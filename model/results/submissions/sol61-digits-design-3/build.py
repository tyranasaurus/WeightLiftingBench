"""Data-free handwritten-digit network, built entirely from analytic pen paths.

No input files are opened. All geometry below is a priori handwriting geometry.
"""
import sys
import json
from pathlib import Path
import numpy as np


def bezier(*p):
    p = np.asarray(p, dtype=np.float64)
    t = np.linspace(0, 1, 25)[:, None]
    if len(p) == 2:
        return (1-t)*p[0] + t*p[1]
    if len(p) == 3:
        return (1-t)**2*p[0] + 2*t*(1-t)*p[1] + t*t*p[2]
    return (1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]


def chain(start, *commands):
    result = []
    current = start
    for command in commands:
        result.append(bezier(current, *command))
        current = command[-1]
    return np.concatenate(result)


def oval(cx, cy, rx, ry, egg=0.0, power=1.0):
    t = np.linspace(0, 2*np.pi, 161)
    x = np.sign(np.sin(t))*np.abs(np.sin(t))**power
    y = np.sign(np.cos(t))*np.abs(np.cos(t))**power
    return np.stack([cx + rx*x*(1+egg*y), cy-ry*y], 1)


def glyph_base(digit, style):
    a, b = style//4, style%4
    if digit == 0:
        return [oval(.5,.5,.5,.5,[-.18,0,.18,.30][b],[.80,1.,1.2][a])]
    if digit == 1:
        bow = [-.10,-.035,.035,.10][b]
        stem = bezier((.55,0),(.55+bow,.40),(.55+bow,.70),(.55,1))
        if a == 0:
            return [stem]
        flag = bezier((.10 if b%2 else .25,.20 if b<2 else .30),(.34,.13),(.55,0))
        if a == 1:
            return [flag,stem]
        return [flag,stem,bezier((.10,1),(.95,1))]
    if digit == 2:
        turn = [.25,.33,.41][a]
        base = [0.,-.10,.08,.16][b]
        return [chain((.02,.19),
            ((.10,-.05),(.86,-.10),(.97,.15)),
            ((1.05,turn),(.72,.46),(.06,.91)),
            ((-.01,1.04),(.37,1.0+base),(.99,.94+base)))]
    if digit == 3:
        waist = [.40,.49,.58][a]
        if b < 2:
            return [chain((.03,.10),
                ((.38,-.07),(.92,-.03),(.96,.18)),
                ((1.0,.33),(.64,waist),(.31 if b==0 else .48,waist)),
                ((.91,waist-.08),(1.05,.68),(.91,.85)),
                ((.75,1.06),(.18,1.04),(.02,.86)))]
        return [chain((.02,.05),
            ((.93,0.),),
            ((.48,waist),),
            ((.99,waist-.03),(1.08,.76),(.85,.90)),
            ((.61,1.06),(.19,1.02),(0.,.85)))]
    if digit == 4:
        cross = [.49,.60,.71][a]
        left = [.06,.28,.52,.77][b]
        return [chain((left,0),((0.,cross),),((1.,cross-.02),)),
                bezier((.82,0),(.76,.48),(.73,1))]
    if digit == 5:
        waist = [.34,.43,.52][a]
        top = [0.,.06,-.07,.12][b]
        return [chain((.97,top),((.13,0.),),((.06,waist),),
            ((.42,waist-.13),(.94,waist-.02),(.98,.67)),
            ((1.04,.98),(.35,1.13),(.01,.84 if b%2 else .91)))]
    if digit == 6:
        loop = [.34,.43,.52][a]
        topx = [.56,.75,.94,1.06][b]
        return [chain((topx,0),
            ((.32,.04),(-.02,.39),(.015,.73)),
            ((.04,1.12),(.95,1.11),(.99,.74)),
            ((1.03,loop-.04),(.33,loop-.06),(.04,.69)))]
    if digit == 7:
        bend = [-.12,0.,.13,.25][b]
        strokes = [chain((0,.04),((1.,0.),),
                         ((.87+bend,.28),(.43+bend,.66),(.27,1.)))]
        if a == 1:
            strokes.append(bezier((.23,.50),(.82,.48)))
        if a == 2:
            strokes.append(bezier((0,.04),(.02,.20)))
        return strokes
    if digit == 8:
        mid = [.40,.49,.58][a]
        upper = [.65,.80,.95,1.08][b]
        if b == 3:
            return [oval(.5,mid/2,.5*upper,mid/2),
                    oval(.5,(1+mid)/2,.5,(1-mid)/2)]
        return [chain((.5,mid),
            ((.5-.65*upper,mid-.10),(.5-.5*upper,0),(.5,0)),
            ((.5+.5*upper,0),(.5+.65*upper,mid-.10),(.5,mid)),
            ((-.16,mid+.12),(-.16,1),(.5,1)),
            ((1.16,1),(1.16,mid+.12),(.5,mid)))]
    if digit == 9:
        loop = [.40,.50,.60][a]
        end = [.20,.45,.70,.95][b]
        return [oval(.45,loop/2,.45,loop/2),
                bezier((.90,loop*.30),(.94,.55),(.80,1.),(end,1.))]
    raise ValueError(digit)


def glyph(digit, style):
    if style < 12:
        return glyph_base(digit,style)
    a,b=divmod(style-12,4)
    if digit==0:
        if a==0:
            return [oval(.5,.5,.5,.5,[-.42,-.28,.28,.42][b],.65)]
        if a==1:
            return [oval(.5,.5,.5,.5,[-.25,-.08,.08,.25][b],1.55)]
        return [chain((.50,.00),((-.07,.02),(-.12,.82),(.35,1.)),
                      ((.84,1.13),(1.08,.20),(.66,.02)),
                      ((.55,-.03),(.47,.02),(.43,.07)))]
    if digit==1:
        lean=[-.12,-.04,.04,.12][b]
        return [chain((.12,.28),((.55,0.),),
                      ((.56+lean,.35),(.52+lean,.73),(.47,1.)))]
    if digit==2:
        top=[.32,.44,.56,.68][b]
        loop=[.66,.78,.90][a]
        return [chain((.04,.22),((.10,-.04),(.77,-.09),(.91,.12)),
            ((1.09,.39),(.20,loop-.08),(.03,.88)),
            ((-.16,1.20),(.64,1.09),(.63,loop)),
            ((.60,loop-.20),(.22,loop-.21),(.13,loop-.09)),
            ((.06,loop+.04),(.52,1.),(1.05,.97)))]
    if digit==3:
        waist=[.36,.45,.54][a]
        top=[.20,.42,.68,.89][b]
        return [chain((.02,.09),((top,-.10),(1.04,-.01),(.97,.20)),
            ((.93,.35),(.66,waist),(.35,waist)),
            ((1.03,waist-.02),(1.16,.79),(.86,.94)),
            ((.60,1.08),(.10,1.),(-.03,.72)))]
    if digit==4:
        cross=[.47,.59,.71][a]
        apex=[.50,.66,.82,.98][b]
        return [chain((apex,0),((0.,cross),),((1.,cross-.02),)),
                bezier((apex,0),(.78,.44),(.70,1))]
    if digit==5:
        waist=[.28,.40,.52][a]
        right=[.70,.85,1.0,1.15][b]
        return [chain((1.,.04),((.08,0.),),((.03,waist),),
            ((.41,waist-.14),(right,waist+.02),(right,.72)),
            ((right,.99),(.26,1.12),(-.04,.78)))]
    if digit==6:
        loop=[.24,.38,.55][a]
        top=[.18,.42,.75,1.0][b]
        return [chain((top,0),((.13,.14),(-.02,.49),(.04,.77)),
            ((.15,1.12),(1.01,1.10),(.98,.72)),
            ((.96,loop),(.35,loop-.03),(.06,.73)))]
    if digit==7:
        return [chain((0.,.10),((.50,-.02),(1.04,-.04),(.94,.13)),
            ((.75,.34),(.33,.69),(.15+.07*b,1.)))]
    if digit==8:
        waist=[.34,.48,.62][a]
        top_width=[.60,.80,1.0,1.20][b]
        return [oval(.5,waist/2,.5*top_width,waist/2,0.,.78),
                oval(.5,(1+waist)/2,.50,(1-waist)/2,0.,.78)]
    if digit==9:
        loop=[.38,.52,.66][a]
        end=[-.10,.12,.35,.60][b]
        return [oval(.45,loop/2,.45,loop/2,0.,.78),
                bezier((.90,loop*.35),(.97,.64),(.75,1.10),(end,.97))]
    raise ValueError(digit)


def equal_spacing(strokes, spacing=.55):
    points = []
    tangents = []
    lengths = []
    for p in strokes:
        ds = np.linalg.norm(np.diff(p,axis=0),axis=1)
        s = np.r_[0,np.cumsum(ds)]
        length = s[-1]
        n = max(2,int(np.ceil(length/spacing)))
        targets = (np.arange(n)+.5)*length/n
        q=np.stack([np.interp(targets,s,p[:,0]),np.interp(targets,s,p[:,1])],1)
        v=np.stack([np.interp(np.minimum(targets+.4,length),s,p[:,0])-np.interp(np.maximum(targets-.4,0),s,p[:,0]),
                    np.interp(np.minimum(targets+.4,length),s,p[:,1])-np.interp(np.maximum(targets-.4,0),s,p[:,1])],1)
        points.append(q)
        tangents.append(np.arctan2(v[:,1],v[:,0]))
        lengths.append(length)
    return np.concatenate(points), np.concatenate(tangents), sum(lengths)


def pose(strokes, width, height, slant):
    result = []
    for p in strokes:
        q = p.copy()
        q[:,0] = (p[:,0]-.5)*width + slant*(.5-p[:,1])*height
        q[:,1] = (p[:,1]-.5)*height
        result.append(q)
    bounds = np.ptp(np.concatenate(result),axis=0).max()
    factor = min(1.,19.1/max(bounds,1.e-8))
    result = [p*factor for p in result]
    points,angles,length = equal_spacing(result)
    points = points-points.mean(axis=0)+13.5
    return points,angles,length


def stroke_weights(points, angles, length):
    # A length-normalized quadrature measure for template-to-image distance.
    field = np.zeros((28,28),np.float64)
    x,y = points[:,0],points[:,1]
    ix,iy = np.floor(x).astype(int),np.floor(y).astype(int)
    fx,fy = x-ix,y-iy
    for ox,oy,v in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),
                    (0,1,(1-fx)*fy),(1,1,fx*fy)]:
        xx,yy=ix+ox,iy+oy
        valid=(xx>=0)&(xx<28)&(yy>=0)&(yy<28)
        np.add.at(field,(yy[valid],xx[valid]),v[valid]/len(points))
    # The converse term penalizes unexplained ink outside the pen corridor.
    yy,xx=np.mgrid[:28,:28]
    distance=np.sqrt(((xx[:,:,None]-x)**2+(yy[:,:,None]-y)**2).min(axis=2))
    cost=np.minimum((np.maximum(distance-1.1,0.)/3.5)**1.3,2.5)
    cover=-1.6*cost/100.
    # Enclosed negative space, generated from the analytic pen geometry only.
    free=distance>1.1
    exterior=np.zeros((28,28),bool)
    exterior[[0,-1],:]=free[[0,-1],:]
    exterior[:,[0,-1]]=free[:,[0,-1]]
    for _ in range(55):
        padded=np.pad(exterior,1)
        reach=(padded[:-2,1:-1] | padded[2:,1:-1] |
               padded[1:-1,:-2] | padded[1:-1,2:] | exterior) & free
        if np.array_equal(reach,exterior):
            break
        exterior=reach
    hole=(free & ~exterior).astype(np.float32)
    hpad=np.pad(hole,1)
    hole=sum(hpad[dy:dy+28,dx:dx+28] for dy in range(3) for dx in range(3))/9
    return field.astype(np.float32),cover.astype(np.float32),hole


class Network:
    def __init__(self):
        self.layers=[]
        self.weights={}
    def op(self,name,op,inputs,**kw):
        self.layers.append(dict(name=name,op=op,inputs=inputs,**kw))
        return name
    def conv(self,name,source,W,b=None,padding=0):
        W=np.asarray(W,dtype=np.float32)
        if b is None: b=np.zeros(W.shape[0],np.float32)
        self.weights[name+'.W']=W
        self.weights[name+'.b']=np.asarray(b,dtype=np.float32)
        return self.op(name,'conv',[source],out=W.shape[0],kernel=list(W.shape[-2:]),stride=1,padding=padding)
    def scalar(self,name,source,scale,bias=0,padding=0):
        return self.conv(name,source,np.array([[[[scale]]]],np.float32),[bias],padding)
    def channel_scale(self,name,source,scale,channels=4):
        return self.conv(name,source,(np.eye(channels)*scale)[:,:,None,None])


def build(folder):
    n=Network()
    # Soft binarization reduces differences in pen pressure and ink thickness.
    n.scalar('ink_threshold','image',2.5,-.25)
    n.op('ink_positive','relu',['ink_threshold'])
    n.scalar('ink_excess','ink_positive',1,-1)
    n.op('ink_excess_positive','relu',['ink_excess'])
    n.scalar('ink_negative_excess','ink_excess_positive',-1)
    n.op('ink','add',['ink_positive','ink_negative_excess'])
    fields=['ink']
    for r in range(1,8):
        n.scalar('pad_'+str(r),fields[-1],1,padding=1)
        fields.append(n.op('dilate_'+str(r),'maxpool',['pad_'+str(r)],size=[3,3],stride=[1,1]))
    terms=[]
    for r,(f,c) in enumerate(zip(fields,[.18,.30,.28,.26,.25,.24,.23,.22])):
        terms.append(n.scalar('distance_term_'+str(r),f,c))
    n.op('near_ink','add',terms)

    # Explicit enclosed-background feature: flood the background from the
    # frame, using only max pooling, addition and ReLU minimum gates.
    n.scalar('free_affine','ink',-2.,1.)
    n.op('free','relu',['free_affine'])
    n.scalar('constant_one','image',0.,1.)
    n.conv('frame_seed','constant_one',-np.ones((1,1,3,3),np.float32)/3,[3.],padding=1)
    f='frame_seed'
    for step in range(28):
        prefix='flood_%02d_'%step
        n.scalar(prefix+'pad',f,1.,padding=1)
        n.op(prefix+'expand','maxpool',[prefix+'pad'],size=[3,3],stride=[1,1])
        n.scalar(prefix+'neg',prefix+'expand',-1.)
        n.op(prefix+'diff','add',['free',prefix+'neg'])
        n.op(prefix+'cut','relu',[prefix+'diff'])
        n.scalar(prefix+'ncut',prefix+'cut',-1.)
        f=n.op(prefix+'result','add',['free',prefix+'ncut'])
    n.scalar('negative_exterior',f,-1.)
    n.op('holes','add',['free','negative_exterior'])

    widths=[8.5,11.,13.5,16.,18.5]
    heights=[17.,18.5,20.]
    slants=[-.40,-.25,-.10,.05,.20,.35,.50]
    per_class=24*len(widths)*len(heights)*len(slants)
    count=10*per_class
    skel=np.empty((count,1,28,28),np.float32)
    cover=np.empty((count,1,28,28),np.float32)
    hole_filters=np.empty_like(cover)
    bias=np.full(count,-1.,np.float32)
    hole_bias=np.zeros(count,np.float32)
    i=0
    for digit in range(10):
        for style in range(24):
            strokes=glyph(digit,style)
            for width in ([4.,6.,8.,10.,12.] if digit==1 else widths):
                for height in heights:
                    for slant in slants:
                        points,angles,length=pose(strokes,width,height,slant)
                        skel[i,0],cover[i,0],hole=stroke_weights(points,angles,length)
                        # Reward a hole in the right neighborhood, without
                        # demanding a particular loop's area. Extra holes
                        # elsewhere receive a gentle penalty.
                        hp=np.pad(hole,2)
                        neighborhood=np.maximum.reduce([hp[dy:dy+28,dx:dx+28]
                                                        for dy in range(5) for dx in range(5)])
                        hole_filters[i,0]=.0025*(2*neighborhood-1.)
                        hole_bias[i]=0.
                        i+=1
        print('constructed analytic digit',digit,flush=True)
    n.conv('stroke_presence','near_ink',skel,bias,padding=1)
    n.conv('unexplained_ink','ink',cover,padding=1)
    n.conv('negative_space','holes',hole_filters,hole_bias,padding=1)
    n.op('template_scores','add',['stroke_presence','unexplained_ink','negative_space'])
    n.op('group_by_digit','reshape',['template_scores'],shape=[10,per_class,9])
    n.op('best_explanation','maxpool',['group_by_digit'],size=[per_class,9],stride=[per_class,9])
    n.op('scores','reshape',['best_explanation'],shape=[10])
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'network.json').write_text(json.dumps({'layers':n.layers},indent=2)+'\n')
    np.savez(folder/'weights.npz',**n.weights)
    print('weights',sum(x.size for x in n.weights.values()),'templates',count)


if __name__=='__main__':
    build(sys.argv[1])
