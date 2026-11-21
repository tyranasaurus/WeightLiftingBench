"""Data-free, hand-authored clothing shape network.

Every template below is a generic geometric drawing, not a training example.
All dimensions are fractions of a hypothetical object's box. No data are read.
"""
import sys
import json
from pathlib import Path
import itertools
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def op(self, name, op, inputs, **settings):
        self.layers.append(dict(name=name, op=op, inputs=[inputs] if isinstance(inputs, str) else inputs, **settings))
        return name

    def linear(self, name, inp, w, b=None, conv=False, padding=0):
        w = np.asarray(w, dtype=np.float32)
        if b is None:
            b = np.zeros(w.shape[0], np.float32)
        self.weights[name+'.W'] = w
        self.weights[name+'.b'] = np.asarray(b, dtype=np.float32)
        if conv:
            return self.op(name, 'conv', inp, out=w.shape[0], kernel=list(w.shape[-2:]), stride=1, padding=padding)
        return self.op(name, 'dense', inp, out=w.shape[0])


# Generic drawings are supersampled, then area-averaged, not fitted.
S = 56
Y, X = np.mgrid[0:S, 0:S].astype(np.float64)
X = (X + .5)/S - .5
Y = (Y + .5)/S - .5


def polygon(x, y, points):
    inside = np.zeros(x.shape, dtype=bool)
    a = points[-1]
    for b in points:
        if a[1] != b[1]:
            cross = ((a[1] > y) != (b[1] > y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1]) + a[0])
            inside ^= cross
        a = b
    return inside


def ellipse(x, y, cx, cy, rx, ry):
    return ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1


def line(x, y, a, b, thickness):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = np.clip(((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy), 0, 1)
    return (x-a[0]-t*dx)**2+(y-a[1]-t*dy)**2 <= thickness**2


def frame(w, h, flip=1, tilt=0, offset=0):
    xx = X*flip
    yy = Y - offset
    return (xx+tilt*yy)/w, (yy-tilt*xx)/h + .5


def garment(w, h, body, sleeve, neck, hem=1, flare=0, style='round'):
    x,y = frame(w,h)
    # Torso, shoulders and two sleeves. Long sleeves hang down at the sides;
    # short sleeves point partly outwards. This is an idealized sewing pattern.
    bot = body+flare
    if sleeve <= .5:
        points = [(-body,0),(-.37,.07),(-.5,sleeve*.75),(-.39,sleeve),
                  (-body,.24),(-bot,hem),(bot,hem),(body,.24),
                  (.39,sleeve),(.5,sleeve*.75),(.37,.07),(body,0)]
    else:
        points = [(-body,0),(-.41,.09),(-.5,sleeve-.04),(-.37,sleeve),
                  (-body,.33),(-bot,hem),(bot,hem),(body,.33),
                  (.37,sleeve),(.5,sleeve-.04),(.41,.09),(body,0)]
    m = polygon(x,y,points)
    if style == 'v':
        hole = polygon(x,y,[(-neck,-.03),(neck,-.03),(0,.22)])
    else:
        hole = ellipse(x,y,0,-.035,neck,.13)
    return m & ~hole


def tank(w,h,body,neck):
    x,y = frame(w,h)
    m = polygon(x,y,[(-.29,0),(-.46,.24),(-body,.38),(-body,1),
                        (body,1),(body,.38),(.46,.24),(.29,0)])
    return m & ~ellipse(x,y,0,-.01,neck,.28)


def pants(w,gap,crotch,taper,slant=0):
    x,y = frame(w,.98,tilt=slant)
    outer = polygon(x,y,[(-.48,0),(.48,0),(.5,.20),(.42*taper,1),
                         (.10+gap,1),(0,crotch),(-.10-gap,1),(-.42*taper,1),(-.5,.20)])
    return outer


def dress(w,waist,hem,sleeve,waist_y=.4):
    x,y = frame(w,.98)
    m = polygon(x,y,[(-.21,0),(-.36,.06),(-.5,.20),(-.39,.29),
                        (-.25,.20),(-waist,waist_y),(-hem,1),(hem,1),
                        (waist,waist_y),(.25,.20),(.39,.29),(.5,.20),(.36,.06),(.21,0)])
    if sleeve == 'none':
        m = polygon(x,y,[(-.20,0),(-.29,.03),(-.31,.23),(-waist,waist_y),
                           (-hem,1),(hem,1),(waist,waist_y),(.31,.23),(.29,.03),(.20,0)])
    elif sleeve == 'long':
        m |= polygon(x,y,[(-.31,.03),(-.40,.08),(-.5,.6),(-.39,.64),(-.26,.20)])
        m |= polygon(-x,y,[(-.31,.03),(-.40,.08),(-.5,.6),(-.39,.64),(-.26,.20)])
    return m & ~ellipse(x,y,0,-.02,.14,.13)


def bag(w,h,top,roundness,handle):
    x,y = frame(w,h)
    if roundness:
        body = ellipse(x,y,0,(top+1)/2,.50,(1-top)/2)
        body |= (np.abs(x)<.40)&(y>top)&(y<.85)
    else:
        body = polygon(x,y,[(-.44,top),(.44,top),(.5,.90),(.40,1),(-.40,1),(-.5,.90)])
    if handle:
        outer = ellipse(x,y,0,top,.28,top)
        inner = ellipse(x,y,0,top,.21,max(top-.07,.02))
        body |= outer & ~inner & (y<top+.03)
    else:
        # A small short handle on a clutch or briefcase.
        body |= (np.abs(x)<.15)&(y>top-.08)&(y<top+.02)
        body &= ~((np.abs(x)<.09)&(y>top-.05)&(y<top))
    return body


def sneaker(h,ankle,high,flip=1,tilt=0):
    x,y = frame(.98,h,flip,tilt)
    # Toe is on the left, heel on the right.
    return polygon(x,y,[(-.50,.78),(-.45,.60),(-.17,.48),(ankle-.18,.05),
                        (ankle+.03,high),(.39,high-.03),(.46,high+.12),
                        (.48,.88),(.40,1),(-.42,1),(-.50,.91)])


def boot(w,h,shaft,ankle,heel,flip=1,tilt=0):
    x,y = frame(w,h,flip,tilt)
    m = polygon(x,y,[(-.5,.82),(-.43,.72),(-.10,.59),(ankle-.18,.42),
                      (ankle-.18,0),(.46,0),(.46,.72),(.43,1),
                      (.25,1),(.22,.85),(-.43,.94),(-.5,.90)])
    if shaft < .95:
        m &= y >= (1-shaft)*.5
    if heel:
        m &= ~polygon(x,y,[(-.20,.93),(.25,.77),(.25,1),(-.20,1)])
    return m


def sandal(h,style,flip=1,tilt=0):
    x,y = frame(.98,h,flip,tilt)
    m = polygon(x,y,[(-.5,.87),(-.43,.81),(.45,.81),(.48,.94),(-.43,1),(-.5,.96)])
    if style == 'flat':
        for a,b,t in [((-.30,.85),(-.10,.40),.055),((-.10,.40),(.15,.84),.055),
                      ((-.31,.83),(-.13,.72),.07),((.20,.83),(.35,.24),.05)]:
            m |= line(x,y,a,b,t)
    elif style == 'cross':
        for a,b in [((-.36,.83),(.12,.15)),((-.20,.40),(.36,.83)),((.17,.83),(.37,.06))]:
            m |= line(x,y,a,b,.06)
    elif style == 'slide':
        m |= polygon(x,y,[(-.35,.82),(-.30,.24),(-.04,.10),(.14,.60),(.18,.84)])
    elif style == 'heel':
        m = polygon(x,y,[(-.5,.86),(-.45,.73),(-.20,.80),(.20,.46),(.43,.44),
                          (.41,1),(.33,1),(.27,.62),(-.17,.96),(-.47,.98)])
        m |= line(x,y,(.22,.47),(.36,.02),.045)
        m |= line(x,y,(.36,.02),(.12,.02),.04)
    elif style == 'strappy':
        for yy in [.2,.4,.6]:
            m |= line(x,y,(.03,yy),(.39,yy-.10),.04)
        m |= line(x,y,(.05,.80),(.05,.1),.04)
        m |= line(x,y,(.38,.82),(.38,.02),.04)
        m |= line(x,y,(-.4,.82),(-.2,.62),.065)
    return m


def templates():
    ts=[[] for _ in range(10)]
    def put(c,m,bias=0):
        # Supersampling yields soft silhouette boundaries at 14 by 14.
        # Shoes and handbags may be centered or rest on a common baseline.
        placements=[m]
        if c in (5,7,8,9):
            occupied=np.flatnonzero(m.any(axis=1))
            if len(occupied):
                shift=S-1-int(occupied[-1])
                if shift>2:
                    baseline=np.zeros_like(m)
                    baseline[shift:]=m[:-shift]
                    placements.append(baseline)
        for drawing in placements:
            t=drawing.astype(np.float32).reshape(14,4,14,4).mean((1,3))
            ts[c].append((t,bias))
    # Parameter ranges encode general variation in clothing cuts and proportions.
    for w,h,body,sl,neck in itertools.product([.74,.87,1.0],[.90,1.0],[.26,.32,.38],[.27,.39,.49],[.10,.19]):
        if w<.8 and h<1: continue
        put(0,garment(w,h,body,sl,neck),.05)
    for w,body,neck in itertools.product([.48,.60,.73],[.32,.42],[.16,.25]):
        put(0,tank(w,.98,body,neck),-.05)
    for w,gap,crotch,taper,slant in itertools.product([.30,.40,.52,.65],[.025,.08,.14],[.30,.43,.55],[.75,1.0,1.15],[-.06,0,.06]):
        put(1,pants(w,gap,crotch,taper,slant),.05)
    for w,body,sl,hem,neck in itertools.product([.76,.88,1.0],[.25,.31,.37],[.83,.98],[.88,1.0],[.09,.17]):
        put(2,garment(w,.98,body,sl,neck,hem),.04)
        put(4,garment(w,.98,body,sl,neck,hem,flare=.02,style='v'),-.16)
        put(6,garment(w,.98,body,sl,neck,hem,style='v'),-.17)
    for w,body,sl,opening in itertools.product([.76,.88,1.0],[.30,.37,.42],[.87,.99],[.04,.10,.17]):
        m=garment(w,.98,body,sl,.12,1.0,style='v')
        x,y=frame(w,.98)
        gap=(abs(x)<opening*(.3+.7*y))&(y>.23)
        put(4,m & ~gap,-.16)
        # A coat may have a continuous broad torso and no arm/body notch.
        if body>.35:
            broad=m | ((abs(x)<body)&(y>.20)&(y<1))
            put(4,broad & ~gap,-.24)
    for w,waist,hem,sleeve,wy in itertools.product([.44,.59,.74,.89],[.15,.23,.30],[.32,.42,.50],['none','short','long'],[.36,.49]):
        put(3,dress(w,waist,hem,sleeve,wy),.02)
    for w,body,sl in itertools.product([.78,.90,1.0],[.28,.35],[.29,.43]):
        put(6,garment(w,.98,body,sl,.14,style='v'),-.16)
    # Shirttails, wider woven-fabric cuts and lower cuffs are generic sewing
    # variations not well represented by a pullover's straight ribbed hem.
    for w,body,sl,tail,neck in itertools.product([.66,.80,.94],[.29,.36,.42],[.38,.82,.94],[.06,.14],[.11,.20]):
        x,y=frame(w,.98)
        m=garment(w,.98,body,sl,neck,style='v')
        curve=1-tail*(np.abs(x)/body)**2
        torso=(np.abs(x)<body)&(y>.4)
        m &= ~(torso & (y>curve))
        put(6,m,-.12)
    # Coats retain the explicit shoulder and sleeve proportions above.
    # A simple pair of narrow vertical panels is not a sufficient coat model.
    for h,style,flip,tilt in itertools.product([.24,.40,.58,.76],['flat','cross','slide','heel','strappy'],[-1,1],[-.08,0,.08]):
        put(5,sandal(h,style,flip,tilt),.03)
    for h,ankle,high,flip,tilt in itertools.product([.28,.39,.52],[.02,.18,.30],[.10,.30,.46],[-1,1],[-.08,0,.08]):
        put(7,sneaker(h,ankle,high,flip,tilt),.03)
    for w,h,top,rnd,handle in itertools.product([.60,.80,.98],[.62,.80,.98],[.15,.30,.43],[False,True],[False,True]):
        if max(w,h)<.95: continue
        put(8,bag(w,h,top,rnd,handle),.02)
    for w,h,shaft,ankle,heel,flip,tilt in itertools.product([.72,.98],[.48,.66,.84,.98],[.8,1.0],[.02,.18],[False,True],[-1,1],[-.06,0,.06]):
        if max(w,h)<.95: continue
        put(9,boot(w,h,shaft,ankle,heel,flip,tilt),-.01)
    return ts


def build(out):
    n=Network()
    # A soft foreground indicator suppresses brightness and texture variation.
    g=np.array([[1,2,1],[2,4,2],[1,2,1]],np.float32)/16
    n.linear('blur','image',g[None,None],conv=True,padding=1)
    n.linear('foreground_ramps','blur',np.array([10,10,5,5],np.float32).reshape(4,1,1,1),[-.35,-1.35,-.8,-1.8],conv=True)
    n.op('ramps_positive','relu','foreground_ramps')
    n.linear('foreground','ramps_positive',np.array([[1,-1,0,0],[0,0,1,-1]],np.float32).reshape(2,4,1,1),conv=True)
    n.op('small_shape','avgpool','foreground',size=[2,2],stride=[2,2])
    ts=templates()
    count=max(map(len,ts))
    w=np.zeros((10*count,2,14,14),np.float32)
    b=np.full(10*count,-100,np.float32)
    for c, group in enumerate(ts):
        for j,(t,prior) in enumerate(group):
            idx=c*count+j
            # Negative squared silhouette discrepancy up to a shared input term.
            w[idx,0] = 10*.85*(2*t-1)/196
            w[idx,1] = 10*.15*(2*t-1)/196
            b[idx]=-10*np.mean(t*t)+prior
    n.linear('shape_matches','small_shape',w.reshape(10*count,-1),b)
    n.op('template_groups','reshape','shape_matches',shape=[10,1,count])
    n.op('best_shape','maxpool','template_groups',size=[1,count],stride=[1,count])
    n.op('shape_scores','reshape','best_shape',shape=[10])

    # Contrast-independent intensity supplies central openings and collar cues.
    n.op('contrast','norm','image')
    yy,xx=np.mgrid[0:28,0:28].astype(float)
    xx=(xx+.5)/28-.5
    yy=(yy+.5)/28
    fs=[]; names=[]
    def feature(name,pos,neg):
        p=pos.astype(float); q=neg.astype(float)
        fs.append(p/max(p.sum(),1)-q/max(q.sum(),1))
        names.append(name)
    for wid in [.035,.07]:
        for lo,hi in [(.22,.85),(.35,.90),(.12,.47)]:
            p=(abs(xx)<wid)&(yy>lo)&(yy<hi)
            q=(abs(xx)>wid)&(abs(xx)<wid+.09)&(yy>lo)&(yy<hi)
            feature('front_line',p,q)
    # V-shaped lapel / collar contrast to the adjacent chest.
    for slope in [.35,.60]:
        center=slope*(yy-.05)
        p=(abs(abs(xx)-center)<.035)&(yy>.08)&(yy<.40)
        q=(abs(abs(xx)-center-.07)<.035)&(yy>.08)&(yy<.40)
        feature('lapel',p,q)
    f=np.array(fs,np.float32)
    n.linear('front_contrasts','contrast',np.concatenate([f,-f]).reshape(len(f)*2,-1))
    n.op('front_evidence','relu','front_contrasts')
    # Sum of contrast magnitudes, with conservative hand-set influence.
    sw=np.zeros((10,2*len(f)),np.float32)
    for i in range(len(f)):
        for j in [i,i+len(f)]:
            sw[4,j]=.12 if i<6 else .18
            sw[6,j]=.08 if i<6 else .10
            sw[2,j]=-.04 if i<6 else -.03
    n.linear('front_scores','front_evidence',sw)

    # Fine fabric edges: directional differences, rectified to magnitudes.
    kernels=[]
    for k in [np.array([[0,0,0],[-1,0,1],[0,0,0]])/2,
              np.array([[0,-1,0],[0,0,0],[0,1,0]])/2]:
        kernels.extend([k,-k])
    n.linear('fabric_edges','contrast',np.array(kernels,np.float32)[:,None],conv=True,padding=1)
    n.op('edge_magnitudes','relu','fabric_edges')
    # Only the torso, away from the silhouette boundary and neckline.
    p=(abs(xx)<.20)&(yy>.35)&(yy<.85)
    ew=np.zeros((10,4,28,28),np.float32)
    for c,coef in [(6,.20),(2,.01),(4,.02)]:
        ew[c]=coef*p/max(p.sum(),1)
    n.linear('fabric_scores','edge_magnitudes',ew.reshape(10,-1))
    # Region occupancies are specified semantically in object-centered units.
    regions=[
        (abs(xx)>.25)&(abs(xx)<.47)&(yy>.12)&(yy<.48), # shoulders/upper sleeves
        (abs(xx)>.27)&(abs(xx)<.47)&(yy>.58)&(yy<.90), # long sleeve ends
        (abs(xx)<.18)&(yy>.28)&(yy<.65),              # torso
        (abs(xx)<.18)&(yy>.67)&(yy<.94),              # hem/legs
        (abs(xx)<.055)&(yy>.55)&(yy<.94),            # leg separation
        (abs(xx)<.055)&(yy>.17)&(yy<.50),            # upper central opening
        (abs(xx)>.08)&(abs(xx)<.20)&(yy>.17)&(yy<.50),# chest flanks
    ]
    nr=len(regions)
    rw=np.zeros((nr,2,28,28),np.float32)
    for i,r in enumerate(regions): rw[i,0]=r/max(r.sum(),1)
    n.linear('body_regions','foreground',rw.reshape(nr,-1))
    # Rules express garment constraints; none is estimated from examples.
    rules=[]
    def rule(c,coeff,threshold,gain):
        row=np.zeros(nr,np.float32)
        for k,v in coeff.items(): row[k]=v
        rules.append((c,row,threshold,gain))
    rule(4,{1:1,0:.35},.55,.12) # long sleeves alone are weak coat evidence
    rule(2,{1:1,0:.25},.60,.12)
    rule(6,{0:1,1:.30},.35,.16)
    rule(6,{0:1,5:-.65,6:.65},.65,.40) # collar/chest opening with shoulders
    rule(4,{0:.6,5:-1,6:1},.60,.35)
    rule(0,{0:.6,1:-1},.12,.25) # short sleeves
    # Bag suppression below uses actual arm/body gaps, not breadth alone.
    # A side silhouette with a split down its entire lower center is trousers,
    # not a button front; reduce spurious texture-based shirt scores there.
    rule(1,{2:.6,3:.8,4:-1,0:-.5},.30,.16)
    rule(6,{0:-1},-.15,-2.5)
    rule(4,{0:-1},-.10,-1.0)
    n.linear('garment_rule_tests','body_regions',np.array([r[1] for r in rules]),[-r[2] for r in rules])
    n.op('garment_rule_positive','relu','garment_rule_tests')
    readout=np.zeros((10,len(rules)),np.float32)
    for j,(c,row,thr,gain) in enumerate(rules): readout[c,j]=gain
    n.linear('garment_rule_scores','garment_rule_positive',readout)

    # Morphological closing measures gaps between straps. A solid shoe upper
    # contributes few such gaps, while a sandal has interrupted foreground.
    n.linear('morph_pad','foreground',np.array([1,0],np.float32).reshape(1,2,1,1),conv=True,padding=3)
    n.op('dilation','maxpool','morph_pad',size=[7,7],stride=[1,1])
    n.linear('negative_dilation','dilation',-np.ones((1,1,1,1),np.float32),conv=True,padding=3)
    n.op('negative_closing','maxpool','negative_dilation',size=[7,7],stride=[1,1])
    n.linear('closing_score','negative_closing',np.ones((10,784),np.float32)*np.array([0,0,0,0,0,-5,0,2,0,.5],np.float32)[:,None]/784)
    n.linear('occupancy_score','foreground',np.concatenate([np.ones((10,784)),np.zeros((10,784))],axis=1)*np.array([0,0,0,0,0,-5,0,2,0,.5])[:,None]/784)
    # Fine foreground preserves the holes in open footwear better than a
    # blurred map. Per-shape occupancy measures solidity, not fabric brightness.
    n.linear('fine_ramps','image',np.array([12,12],np.float32).reshape(2,1,1,1),[-.3,-1.3],conv=True)
    n.op('fine_positive','relu','fine_ramps')
    n.linear('fine_foreground','fine_positive',np.array([1,-1],np.float32).reshape(1,2,1,1),conv=True)
    n.op('fine_small','avgpool','fine_foreground',size=[2,2],stride=[2,2])
    xx14=(np.arange(14)+.5)/14-.5
    yy14=(np.arange(14)+.5)/14-.5
    x14,y14=np.meshgrid(xx14,yy14)
    # A horizontal-vs-vertical occupancy test supplies shoe context; total
    # foreground supplies a conservative sandal prior.
    reg=np.zeros((4,196),np.float32)
    reg[0]=((abs(x14)>.28)&(abs(y14)<.30)).reshape(-1)/max(((abs(x14)>.28)&(abs(y14)<.30)).sum(),1)
    reg[1]=((abs(y14)>.30)&(abs(x14)<.25)).reshape(-1)/max(((abs(y14)>.30)&(abs(x14)<.25)).sum(),1)
    reg[2]=np.ones(196)/196
    reg[3]=((abs(x14)<.32)&(abs(y14)<.15)).reshape(-1)/max(((abs(x14)<.32)&(abs(y14)<.15)).sum(),1)
    n.linear('object_occupancies','fine_small',reg)
    # Low area is only evidence for sandals in a horizontally extended object.
    # min(a,b) = a - relu(a-b), implemented using two positive ramps below.
    rr=np.array([[2,-1,0,0],[0,0,-3,0]],np.float32)
    bb=np.array([-.15,.80],np.float32)
    n.linear('open_shoe_tests','object_occupancies',rr,bb)
    n.op('open_shoe_positive','relu','open_shoe_tests')
    # min(relu(a),relu(b)) = relu(a) - relu(relu(a)-relu(b)).
    n.linear('open_shoe_bounds','open_shoe_positive',np.array([[1,0],[1,-1]],np.float32))
    n.op('open_shoe_bound_positive','relu','open_shoe_bounds')
    ow=np.zeros((10,2),np.float32)
    ow[5]=[3,-3]
    ow[7]=[-1.2,1.2]
    ow[9]=[-.5,.5]
    n.linear('open_shoe_scores','open_shoe_bound_positive',ow)
    # Garment/bag disambiguation: a bag does not have two independent sleeves.
    # Measure arm/body notches on each side rather than merely broad shoulders.
    seam=[]
    meta=[]
    for cx in [-.035,0,.035]:
        for wid in [.027,.055]:
            for lo,hi in [(.20,.45),(.45,.70),(.70,.92)]:
                center=(abs(xx-cx)<wid)&(yy>lo)&(yy<hi)
                flank=(abs(xx-cx)>wid)&(abs(xx-cx)<wid+.075)&(yy>lo)&(yy<hi)
                # Positive for a bright center or a dark center; continuity is
                # taken across vertically separated portions of the front.
                v=center/max(center.sum(),1)-flank/max(flank.sum(),1)
                seam.extend([v,-v])
    n.linear('seam_segments','contrast',np.array(seam,np.float32).reshape(len(seam),-1))
    n.op('seam_magnitudes','relu','seam_segments')
    # Minimum of the segment magnitudes encodes a coherent full-length seam.
    pairs=[]
    for j in range(6):
        for sign in range(2):
            a=j*6+sign; b=a+2; c=a+4
            pairs.extend([(a,b),(a,c)])
    m=np.zeros((len(pairs)*2,len(seam)),np.float32)
    for j,(a,b) in enumerate(pairs):
        m[2*j,a]=1
        m[2*j+1,a]=1; m[2*j+1,b]=-1
    n.linear('seam_pair_differences','seam_magnitudes',m)
    n.op('seam_pair_positive','relu','seam_pair_differences')
    mm=np.zeros((len(pairs),len(pairs)*2),np.float32)
    for j in range(len(pairs)): mm[j,2*j:2*j+2]=[1,-1]
    n.linear('continuous_seams','seam_pair_positive',mm)
    # Take strongest interpretation of a coherent front closure, not a sum of
    # multiple nearby probes of the same seam.
    n.op('seam_shape','reshape','continuous_seams',shape=[1,1,len(pairs)])
    n.op('seam_max','maxpool','seam_shape',size=[1,len(pairs)],stride=[1,len(pairs)])

    # Negative foreground inside a narrow arm/torso gap: absent on a bag,
    # present on hanging sleeves, broad opening on short-sleeved garments.
    masks=[]
    for side in [-1,1]:
        for pos in [.24,.30,.36]:
            near=(abs(xx-side*pos)<.035)&(yy>.37)&(yy<.74)
            masks.append(near/max(near.sum(),1))
    aw=np.zeros((len(masks),2,28,28),np.float32)
    aw[:,0]=np.array(masks)
    n.linear('arm_gaps','foreground',aw.reshape(len(masks),-1))
    n.linear('gap_evidence','arm_gaps',-np.eye(len(masks),dtype=np.float32),np.ones(len(masks),np.float32)*.60)
    n.op('gap_positive','relu','gap_evidence')
    # Presence of cloth outside of the gap prevents background from counting.
    sleeve=[]
    for side in [-1,1]:
        sleeve.append(((xx*side>.32)&(xx*side<.46)&(yy>.25)&(yy<.85)).astype(float))
    sw2=np.zeros((2,2,28,28),np.float32)
    for j,r in enumerate(sleeve):sw2[j,0]=r/max(r.sum(),1)
    n.linear('sleeve_presence','foreground',sw2.reshape(2,-1))
    # The score margin over the best bag model supplies an invariant sleeve
    # cue even if the foreground has no clearly visible arm/body gap.
    gm=np.zeros((1,10),np.float32)
    gm[0,2]=1; gm[0,8]=-1
    n.linear('sleeve_shape_margin','shape_scores',gm,[.2])
    n.op('sleeve_margin_positive','relu','sleeve_shape_margin')
    # Cap the garment-context signal so that score magnitude cannot dominate.
    n.linear('context_cap_tests','sleeve_margin_positive',np.ones((2,1),np.float32),[0,-.5])
    n.op('context_cap_positive','relu','context_cap_tests')
    n.linear('garment_context','context_cap_positive',np.array([[1,-1]],np.float32))
    # min(seam, context) ensures a trouser split or bag handle is not a lapel.
    n.linear('seam_context','seam_max',np.array([[1],[-1]],np.float32))
    n.linear('context_to_seam','garment_context',np.array([[0],[1]],np.float32))
    n.op('context_minus_seam','add',['seam_context','context_to_seam'])
    n.op('context_minus_seam_positive','relu','context_minus_seam')
    # min(seam,context) = context-relu(context-seam).
    n.linear('context_min_part','context_minus_seam_positive',np.array([[0,-1]],np.float32))
    n.op('gated_seam','add',['garment_context','context_min_part'])
    seam_out=np.zeros((10,1),np.float32)
    seam_out[4]=.75;seam_out[6]=.35;seam_out[2]=-.20
    n.linear('seam_scores','gated_seam',seam_out)
    # Generic sleeve/torso notch support. max over candidate notch positions
    # on each side, then require both sides; a single bag handle does not pass.
    n.op('gap_by_side','reshape','gap_positive',shape=[2,1,3])
    n.op('best_gap_by_side','maxpool','gap_by_side',size=[1,3],stride=[1,3])
    n.linear('notch_difference','best_gap_by_side',np.array([[1,0],[1,-1]],np.float32))
    n.op('notch_difference_positive','relu','notch_difference')
    n.linear('bilateral_notch','notch_difference_positive',np.array([[1,-1]],np.float32))
    n.linear('sleeve_difference','sleeve_presence',np.array([[1,0],[1,-1]],np.float32))
    n.op('sleeve_difference_positive','relu','sleeve_difference')
    n.linear('bilateral_sleeve','sleeve_difference_positive',np.array([[1,-1]],np.float32))
    n.linear('notch_to_joint','bilateral_notch',np.array([[1],[1]],np.float32))
    n.linear('sleeve_to_joint','bilateral_sleeve',np.array([[0],[-1]],np.float32))
    n.op('sleeve_notch_joint','add',['notch_to_joint','sleeve_to_joint'])
    n.op('sleeve_notch_positive','relu','sleeve_notch_joint')
    ng=np.zeros((10,2),np.float32)
    ng[8]=[-1.8,1.8]
    ng[4]=[.20,-.20]
    n.linear('notch_scores','sleeve_notch_positive',ng)
    # Relative solidity is scale-independent: project the foreground onto
    # each axis and compare area with the corresponding bounding rectangle.
    # Maximum projections only ask whether cloth is present, not its texture.
    n.op('vertical_projection','maxpool','fine_foreground',size=[28,1],stride=[28,1])
    n.op('horizontal_projection','maxpool','fine_foreground',size=[1,28],stride=[1,28])
    n.linear('projected_width','vertical_projection',np.ones((1,28),np.float32)/28)
    n.linear('projected_height','horizontal_projection',np.ones((1,28),np.float32)/28)
    n.linear('foreground_area','fine_foreground',np.ones((1,784),np.float32)/784)
    # Width is about one for a side-view shoe. Linear aspect-conditioned
    # solidity thresholds avoid multiplication or division in the network.
    n.linear('height_deficit','projected_height',np.array([[.48],[.30]],np.float32),[-.035,-.02])
    n.linear('area_deficit','foreground_area',-np.ones((2,1),np.float32))
    n.op('solidity_deficit','add',['height_deficit','area_deficit'])
    n.op('solidity_positive','relu','solidity_deficit')
    n.linear('aspect_from_width','projected_width',np.ones((1,1),np.float32),[-.2])
    n.linear('aspect_from_height','projected_height',np.array([[-.60]],np.float32))
    n.op('horizontal_aspect','add',['aspect_from_width','aspect_from_height'])
    n.op('horizontal_aspect_positive','relu','horizontal_aspect')
    n.linear('aspect_for_solidity','horizontal_aspect_positive',-np.ones((2,1),np.float32))
    n.op('solidity_minus_aspect','add',['solidity_positive','aspect_for_solidity'])
    n.op('solidity_excess','relu','solidity_minus_aspect')
    n.linear('negative_solidity_excess','solidity_excess',-np.eye(2,dtype=np.float32))
    n.op('shoe_solidity','add',['solidity_positive','negative_solidity_excess'])
    n.linear('shoe_solidity_scores','shoe_solidity',np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[3,5],[0,0],[-1,-2],[0,0],[-1,-2]],np.float32))

    # Fine horizontal and vertical periodic fabric contrast is evidence for
    # woven or patterned shirts. Exclude the external silhouette from this cue.
    p=(abs(xx)<.18)&(yy>.30)&(yy<.82)
    tw=np.zeros((1,4,28,28),np.float32)
    tw[0]=p/max(p.sum(),1)
    n.linear('torso_texture','edge_magnitudes',tw.reshape(1,-1))
    n.linear('texture_threshold','torso_texture',np.ones((2,1),np.float32),[-.30,-.65])
    n.op('texture_threshold_positive','relu','texture_threshold')
    tx=np.zeros((10,2),np.float32)
    tx[6]=[.70,-.70]
    n.linear('texture_scores','texture_threshold_positive',tx)

    # A front opening in the upper and lower torso is more informative than
    # a collar alone. Use a signed, dark center relative to both chest flanks.
    upper=(abs(xx)<.05)&(yy>.20)&(yy<.48)
    uflank=(abs(xx)>.07)&(abs(xx)<.17)&(yy>.20)&(yy<.48)
    lower=(abs(xx)<.05)&(yy>.52)&(yy<.90)
    lflank=(abs(xx)>.07)&(abs(xx)<.17)&(yy>.52)&(yy<.90)
    opw=np.array([uflank/max(uflank.sum(),1)-upper/max(upper.sum(),1),
                  lflank/max(lflank.sum(),1)-lower/max(lower.sum(),1)],np.float32)
    n.linear('dark_opening','contrast',opw.reshape(2,-1),[-.10,-.10])
    n.op('dark_opening_positive','relu','dark_opening')
    n.linear('opening_difference','dark_opening_positive',np.array([[1,0],[1,-1]],np.float32))
    n.op('opening_difference_positive','relu','opening_difference')
    n.linear('opening_minimum','opening_difference_positive',np.array([[1,-1]],np.float32))
    n.linear('opening_negation','opening_minimum',-np.ones((1,1),np.float32))
    n.op('opening_context_difference','add',['opening_negation','garment_context'])
    n.op('opening_context_positive','relu','opening_context_difference')
    n.linear('opening_context_negative','opening_context_positive',-np.ones((1,1),np.float32))
    n.op('gated_opening','add',['garment_context','opening_context_negative'])
    oo=np.zeros((10,1),np.float32)
    oo[4]=.25;oo[6]=.12;oo[2]=-.10
    n.linear('opening_scores','gated_opening',oo)
    n.op('scores','add',['shape_scores','front_scores','fabric_scores','garment_rule_scores','closing_score','occupancy_score','open_shoe_scores','seam_scores','notch_scores','shoe_solidity_scores','texture_scores','opening_scores'])
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':n.layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**n.weights)
    print('Procedural template counts:',[len(t) for t in ts])
    print('Weights:',sum(a.size for a in n.weights.values()))


if __name__=='__main__':
    build(sys.argv[1])
