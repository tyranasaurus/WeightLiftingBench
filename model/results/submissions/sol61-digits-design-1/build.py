"""Data-free, analytic handwritten-digit template network.

All paths below describe generic writing conventions, not observed examples.
No files are read. There is no learning, optimization, or fitted readout.
"""
import json
import sys
from pathlib import Path
import numpy as np


def bezier(points, n=24):
    p = np.asarray(points, dtype=np.float64)
    t = np.linspace(0., 1., n)[:, None]
    if len(p) == 2:
        return (1-t)*p[0] + t*p[1]
    if len(p) == 3:
        return (1-t)**2*p[0] + 2*t*(1-t)*p[1] + t*t*p[2]
    return (1-t)**3*p[0] + 3*t*(1-t)**2*p[1] + 3*t*t*(1-t)*p[2] + t**3*p[3]


def path(*segments):
    return np.concatenate([bezier(s) for s in segments], axis=0)


def ellipse(cx, cy, rx, ry, start=0., stop=2*np.pi, n=100):
    a = np.linspace(start, stop, n)
    return np.stack([cx+rx*np.cos(a), cy+ry*np.sin(a)], axis=1)


def glyph(digit, style, rng):
    # Coordinate system: x about [-.5,.5], y from 0 (top) to 1.
    # The few parameters express ordinary pen-path variations.
    u = rng.uniform
    mid = u(.43,.57)
    topw = u(.38,.55)
    botw = u(.38,.55)
    if digit == 0:
        a = np.linspace(0, 2*np.pi, 144)
        power = u(.85,1.18)
        x = np.sign(np.cos(a))*np.abs(np.cos(a))**power
        y = np.sign(np.sin(a))*np.abs(np.sin(a))**(2-power)
        return [np.stack([.5*x*(1+u(-.18,.18)*y), .5+.5*y],axis=1)]
    if digit == 1:
        lean = u(-.11,.13)
        stem = path(((lean,0),(-.03,.52),(0,1)))
        if style % 5 == 0:
            return [path(((-.30,.23),(lean,0))),stem,path(((-.27,1),(.29,1))) ]
        if style % 5 in (1,2):
            return [path(((-u(.12,.35),u(.12,.30)),(lean,0))),stem]
        return [stem]
    if digit == 2:
        # A rounded head, descending diagonal, and independent baseline.
        # Include both a print corner and a cursive retracing/loop junction.
        head_left = -u(.23,.54)
        head_right = u(.24,.55)
        head_bottom = u(.24,.49)
        start = (head_left,u(.15,.35))
        turn = (head_right,head_bottom)
        end = (-u(.32,.55),u(.88,1.02))
        baseline = (u(.40,.62),u(.81,1.02))
        if style % 6 == 0:
            head = path((start,(-.30,-.04),(.18,-.02),(.47,.04)))
            diagonal = path((((.47,.04)),(.16,.42),(-.20,.73),end))
        else:
            head = path((start,(head_left-u(0,.14),-.09),
                         (head_right+u(.05,.27),-.06),turn))
            diagonal = path((turn,(head_right+u(-.18,.12),head_bottom+.16),
                             (-.36,u(.61,.84)),end))
        if style % 3 == 0:
            bottom = path((end,(-.18,u(1.02,1.12)),(-.10,.76),(-.35,.82)),
                          ((-.35,.82),(-.16,.94),(.22,1.01),baseline))
        elif style % 3 == 1:
            bottom = path((end,(-.23,u(.75,.92)),(.18,1.11),baseline))
        else:
            bottom = path((end,(-.14,.97),(.20,.97),baseline))
        return [head,diagonal,bottom]
    if digit == 3:
        if style % 4 == 0:
            return [path(((-.47,.04),(.45,0)),((.45,0),(-.05,mid)),
                         ((-.05,mid),(.70,.38),(.63,1.13),(-.46,.92)))]
        neck = u(-.18,.14)
        return [path(((-topw,.12),(-.12,-.10),(.63,-.03),(.43,.25)),
                     ((.43,.25),(.35,.39),(.10,mid),(neck,mid)),
                     ((neck,mid),(.67,mid-.10),(.67,.91),(.18,.98)),
                     ((.18,.98),(-.13,1.04),(-.37,.97),(-botw,.87)))]
    if digit == 4:
        joint = u(.50,.66)
        right = u(.18,.36)
        if style % 4 == 0:
            return [path(((right,0),(-.5,joint)),((-.5,joint),(.5,joint))),
                    path(((right,0),(right,1)))]
        return [path(((-u(.15,.38),0),(-.5,joint)),((-.5,joint),(.5,joint))),
                path(((right,u(-.05,.10)),(right,1)))]
    if digit == 5:
        left = -u(.25,.48)
        corner = u(.29,.47)
        if style % 5 == 0:
            top = path(((u(-.06,.28),.01),(left-.04,corner)))
        else:
            top = path(((.48,.015),(left,.08)),((left,.08),(left-.04,corner)))
        if style % 3 == 0:
            bowl = path(((left-.04,corner),(.50,.52),(.57,.75),(.21,.95)),
                        ((.21,.95),(-.10,1.10),(-.44,.99),(-.50,.84)))
        else:
            bowl = path(((left-.04,corner),(.08,corner-.05),(.58,.61),(.46,.83)),
                        ((.46,.83),(.27,1.12),(-.20,1.07),(-.49,.89)))
        return [top,bowl]
    if digit == 6:
        loop_y = u(.61,.70)
        stem_x = u(.12,.45)
        if style % 4 == 0:
            stem_x = -.05
        return [path(((stem_x,.015),(-.35,.12),(-.53,.45),(-.48,.76)),
                     ((-.48,.76),(-.44,1.07),(.33,1.12),(.48,.81)),
                     ((.48,.81),(.68,.49),(-.20,.37),(-.47,.69)))]
    if digit == 7:
        end = u(-.3,-.05)
        stem = path(((.48,.025),(.05,.56),(end,1)))
        bar = path(((-.52,.045),(-.20,.015),(.16,.035),(.48,.025)))
        if style % 5 == 0:
            return [bar,stem,path(((-.34,.51),(.34,.51)))]
        if style % 5 == 1:
            return [path(((-.51,.19),(-.52,.045))),bar,stem]
        return [bar,stem]
    if digit == 8:
        waist = u(.43,.57)
        if style % 3 == 0:
            # A continuous crossing figure-eight.
            a = np.linspace(0,2*np.pi,192)
            return [np.stack([.5*np.sin(2*a)*(1+u(-.15,.1)*np.cos(a)),
                              .5-.5*np.cos(a)],axis=1)]
        return [path(((0,waist),(-.63,waist-.12),(-.47,-.04),(.02,.015)),
                     ((.02,.015),(.60,-.02),(.54,waist-.12),(0,waist)),
                     ((0,waist),(-.67,waist+.15),(-.55,1.04),(.02,.985)),
                     ((.02,.985),(.65,1.05),(.60,waist+.12),(0,waist)))]
    if digit == 9:
        base = u(.37,.64)
        right = u(.28,.48)
        left = -u(.32,.52)
        join = base*u(.35,.65)
        loop = path(((right,join),(right,-.12),(left,-.10),(left,base*.52)),
                    ((left,base*.52),(left,base*1.19),(right,base*1.17),(right,join)))
        if style % 4 == 0:
            tail = path(((right,join),(.48,.77),(.14,1.04),(-.30,.97)))
        elif style % 4 == 1:
            tail = path(((right,join),(.32,.64),(.19,1)))
        else:
            tail = path(((right,join),(.38,.62),(-.08,1)))
        return [loop,tail]
    raise ValueError(digit)


GRID_Y, GRID_X = np.mgrid[:28,:28].astype(np.float64)


def draw(digit, index, rng):
    paths = glyph(digit,index,rng)
    # Smooth, small global bending is a writing-style prior, not fitted noise.
    width = rng.uniform(8.5,17.8)
    if digit == 1:
        width = rng.uniform(10.,17.)
    slant = rng.uniform(-.82,.32)
    bend = rng.uniform(-.09,.09)
    thickness = rng.uniform(1.25,3.6)
    h = rng.uniform(17.1,19.1)
    taper = rng.uniform(-.34,.34)
    vertical_warp = rng.uniform(-.14,.16)
    twist = rng.uniform(-.055,.055)
    coords = []
    for p in paths:
        p = p.copy()
        v = p[:,1]
        p[:,0] *= 1+taper*(2*v-1)
        p[:,0] += twist*np.sin(2*np.pi*v)
        p[:,1] += vertical_warp*np.sin(np.pi*v)
        y = p[:,1]-.5
        p[:,0] = width*(p[:,0]+bend*(4*y*y-1)) + slant*h*y
        p[:,1] = h*y
        coords.append(p)
    points = np.concatenate(coords)
    # Apply the task's generic size/centering convention to the synthetic
    # pen trace. All measurements here are of our own analytic drawing.
    span = np.ptp(points,axis=0).max()+thickness
    factor = rng.uniform(19.3,20.5)/span
    points *= factor
    thickness *= factor
    points += 14.-points.mean(axis=0)
    def distances(p):
        dx = GRID_X[:,:,None]-p[None,None,:,0]
        dy = GRID_Y[:,:,None]-p[None,None,:,1]
        return np.sqrt(np.min(dx*dx+dy*dy,axis=2))
    distance = distances(points)
    ink = np.clip(thickness*.5+.5-distance,0,1)
    mass = ink.sum()
    center = np.array([(ink*GRID_X).sum(),(ink*GRID_Y).sum()])/mass
    points += np.array([14.,14.])+rng.uniform(-.5,.5,2)-center
    distance = distances(points)
    ink = np.clip(thickness*.5+.5-distance,0,1).astype(np.float32)
    proximity = np.exp(-.5*(np.maximum(distance-thickness*.5,0)/1.5)**2)
    return ink,proximity.astype(np.float32)


def max3(a):
    pad = np.pad(a,1)
    return np.maximum.reduce([pad[y:y+28,x:x+28] for y in range(3) for x in range(3)])


def blur3(a):
    pad = np.pad(a,1)
    return sum(pad[y:y+28,x:x+28] for y in range(3) for x in range(3))/9.


def down(a):
    return a.reshape(14,2,14,2).mean(axis=(1,3))


def enclosure(a):
    # A soft empty region blocked by ink in four cardinal directions.
    # This is applied only to our generated drawings during construction.
    p = np.pad(a,9)
    rays = []
    for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
        s = sum(p[9+dy*d:37+dy*d,9+dx*d:37+dx*d] for d in range(1,10))
        rays.append(np.minimum(s,1.))
    return np.maximum(np.minimum.reduce(rays)-1.5*a-.25,0.)


def normalized(a):
    a = a.astype(np.float64).reshape(-1)
    a -= a.mean()
    return (a/np.sqrt(np.mean(a*a)+1e-5)/len(a)).astype(np.float32)


def build(folder):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    layers=[]
    weights={}
    def layer(name,op,ins,**kw):
        layers.append(dict(name=name,op=op,inputs=[ins] if isinstance(ins,str) else ins,**kw))
        return name
    def conv(name,ins,w,b,padding=0):
        w=np.asarray(w,dtype=np.float32)
        weights[name+'.W']=w
        weights[name+'.b']=np.asarray(b,dtype=np.float32)
        return layer(name,'conv',ins,out=w.shape[0],kernel=list(w.shape[2:]),stride=1,padding=padding)
    # Both blur and local dilation are fixed, translation-equivariant features.
    padded=conv('border','image',np.ones((1,1,1,1)),[0],padding=1)
    dilated=layer('dilation','maxpool',padded,size=[3,3],stride=[1,1])
    blurred=conv('blur','image',np.ones((1,1,3,3))/9.,[0],padding=1)
    # Soft enclosure detector; sums along rays are clipped at one using ReLU.
    rayw=np.zeros((4,1,19,19),dtype=np.float32)
    for i,(dy,dx) in enumerate([(0,1),(0,-1),(1,0),(-1,0)]):
        for d in range(1,10):
            rayw[i,0,9+dy*d,9+dx*d]=1.
    rays=conv('rays','image',rayw,[0]*4,padding=9)
    excess=conv('ray_excess',rays,np.eye(4).reshape(4,4,1,1),[-1]*4)
    excess=layer('ray_excess_relu','relu',excess)
    neg_rays=conv('negative_rays',rays,-np.eye(4).reshape(4,4,1,1),[0]*4)
    negclip=layer('negative_clipped','add',[excess,neg_rays])
    # Pairwise maxima of negative clipped rays give the negative minimum.
    # max(a,b)=b+ReLU(a-b), implemented with 1x1 convolutions.
    w=np.zeros((2,4,1,1),dtype=np.float32)
    w[0,0,0,0]=1;w[0,1,0,0]=-1
    w[1,2,0,0]=1;w[1,3,0,0]=-1
    diff=conv('pair_difference',negclip,w,[0,0])
    diff=layer('pair_difference_relu','relu',diff)
    w=np.zeros((2,4,1,1),dtype=np.float32);w[0,1,0,0]=1;w[1,3,0,0]=1
    base=conv('pair_base',negclip,w,[0,0])
    pair=layer('pair_max','add',[base,diff])
    diff=conv('last_difference',pair,np.array([1,-1]).reshape(1,2,1,1),[0])
    diff=layer('last_difference_relu','relu',diff)
    base=conv('last_base',pair,np.array([0,1]).reshape(1,2,1,1),[0])
    negative_min=layer('negative_minimum','add',[base,diff])
    positive_min=conv('minimum_rays',negative_min,-np.ones((1,1,1,1)),[-.25])
    negative_ink=conv('negative_ink','image',np.full((1,1,1,1),-1.5),[0])
    holes=layer('empty_blocked_regions','add',[positive_min,negative_ink])
    holes=layer('enclosure','relu',holes)
    features=[]
    labels=['sharp','soft','wide','holes']
    for label,source in zip(labels,['image',blurred,dilated,holes]):
        pool=layer(label+'_small','avgpool',source,size=[2,2],stride=[2,2])
        features.append(layer(label+'_norm','norm',pool))
    count=8192
    total=10*count
    bank=[np.empty((total,196),dtype=np.float32) for _ in features]
    rng=np.random.default_rng(714025)
    scales=[.285,.38,.19,.05]
    for digit in range(10):
        for j in range(count):
            a,proximity=draw(digit,j,rng)
            k=digit*count+j
            for mat,feature,scale in zip(bank,[a,blur3(a),max3(a),enclosure(a)],scales):
                mat[k]=scale*normalized(down(feature))
            bank[0][k] += .095*normalized(down(proximity))
        print('Constructed analytic class',digit,flush=True)
    names=[]
    for label,feature,w in zip(labels,features,bank):
        name=label+'_match'
        weights[name+'.W']=w
        weights[name+'.b']=np.zeros(total,dtype=np.float32)
        names.append(layer(name,'dense',feature,out=total))
    score=layer('template_scores','add',names)
    score=layer('group_by_digit','reshape',score,shape=[10,1,count])
    score=layer('best_writing_style','maxpool',score,size=[1,count],stride=[1,count])
    layer('scores','reshape',score,shape=[10])
    (folder/'network.json').write_text(json.dumps({'layers':layers},indent=2))
    np.savez(folder/'weights.npz',**weights)
    print('Saved',sum(v.size for v in weights.values()),'analytic weights',flush=True)


if __name__=='__main__':
    build(sys.argv[1])
