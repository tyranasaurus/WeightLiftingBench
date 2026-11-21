"""A data-free stroke grammar and template-matching network. Usage: build.py OUT"""
import json
import sys
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows

F = np.float32
rng = np.random.default_rng(180719)
U = rng.uniform

def path(*anchors):
    """Interpolating cubic curve, with a little tension at its knots."""
    pts = np.array(anchors, dtype=float)
    d = np.concatenate([pts[:1], pts, pts[-1:]], axis=0)
    t = np.arange(12)[:, None, None] / 12
    a, b = d[1:-2], d[2:-1]
    v, w = .46*(b-d[:-3]), .46*(d[3:]-a)
    h = (2*t**3-3*t*t+1)*a + (t**3-2*t*t+t)*v + (-2*t**3+3*t*t)*b + (t**3-t*t)*w
    return np.concatenate((h.transpose(1,0,2).reshape(-1,2), pts[-1:]))

def corner(*anchors):
    return np.concatenate([np.linspace(a,b,20,endpoint=False) for a,b in zip(anchors[:-1],anchors[1:])] + [np.array(anchors[-1:])])

def loop(cx,cy,rx,ry,start,end):
    t = np.linspace(start,end,101)
    return np.stack((cx+rx*np.cos(t),cy+ry*np.sin(t)),1)

# Coordinates below are in pen-space (x increases right, y increases down).
# Variables like waist, shelf, junction, flag, and foot refer to parts of the
# drawing, not to measurements or outlines of exemplars.
def zero():
    a = U(-2.6,-.9)
    t = np.linspace(a,a+2*np.pi+U(-.38,.55),130)
    e = U(.84,1.35)
    c,s = np.cos(t),np.sin(t)
    x = .5 + .50*np.sign(c)*abs(c)**e * (1+U(-.25,.25)*s)
    y = .5 + .50*np.sign(s)*abs(s)**e
    # A small overlap near the closure, just as when the pen comes round.
    x += np.linspace(0,U(-.10,.10),len(t))
    return [np.stack([x,y],1)]

def one():
    a, b = (.53,0), (.46,1)
    style=U()
    if style < .62:
        return [path(a,(U(.42,.59),.46),b)]
    flag = (U(-.02,.40), U(.12,.38))
    p = corner(flag,a,b) if U()<.7 else path(flag,(.4,.085),a,(.54,.33),b)
    if style>.905:
        return [p, path((U(.02,.24),.99),b,(U(.72,.96),.98))]
    return [p]

def two():
    middle = U(.32,.55)
    if U()<.40:
        # Sweep of a cursive two. The crown may be narrow and the baseline
        # may turn into a long rising flourish. A backwards entry makes a
        # little shoulder, independently of whether there is a loop below.
        a=U(.32,.92);x0=U(-.32,.15);rise=U(.14,.48)
        crown=path((x0,U(.17,.40)),(x0-.03,.16),(x0+.15,.045),(a-.16,.0),(a,.12),(a+.025,.31),(a-.07,.47))
        foot=U(.22,.60); cy=U(.54,.83)
        if U()<.66:
            sweep=path((a-.07,.47),(a-.27,.65),(.055,.91),(-.08,1.0),(-.17,.92),(-.15,cy+.02),(.0,cy),(.25,cy+.10),(.60,.91),(1.12,1.0-rise*.45))
        else:
            sweep=path((a-.07,.47),(a-.28,.62),(.09,.86),(-.18,1.0),(foot,.96),(1.06,1.-rise))
        return [np.concatenate([crown,sweep])]
    top = path((U(-.04,.14),U(.17,.37)), (.18,.07),(.53,0),(.84,.075),(.94,.24),(.83,middle),(.58,.62),(.28,.80),(.05,.96))
    st=U()
    if st<.52:
        foot=path((.05,.96),(.27,.96),(.57,U(.91,1.02)),(U(.90,1.12),U(.85,1.0)))
    elif st<.88:
        # A loop written backwards at the left end of the baseline.
        top=path((.07,U(.20,.35)),(.19,.07),(.55,0),(.85,.09),(.88,.27),(.68,.49),(.29,.72),(.02,.89),(.02,.98),(.22,1.0))
        foot=path((.22,1.0),(.38,.9),(.32,U(.73,.84)),(.13,.80),(.12,.87),(.43,.94),(.78,.98),(1.12,U(.87,.99)))
    else:
        top=path((.03,.22),(.36,.055),(.68,.0),(.94,.08),(.89,.27),(.56,.56),(.02,.98))
        foot=corner((.02,.98),(.45,.91),(1.,U(.80,.98)))
    return [np.concatenate([top,foot])]

def three():
    wy=U(.37,.59); tip=U(.14,.48); upper=U(.72,1.)
    if U()<.32:
        first=path((.03,U(.02,.18)),(.44,.026),(upper,.0),(upper-.1,.13),(.57,wy-.09),(tip,wy))
    else:
        first=path((.03,U(.10,.27)),(.19,.06),(.51,0),(upper-.09,.055),(upper,.19),(upper-.12,.32),(.50,wy-.025),(tip,wy))
    last=path((tip,wy),(.49,wy-.025),(.79,wy+.02),(.98,wy+.17),(.91,.85),(.62,.98),(.27,1.),(.02,U(.73,.95)))
    return [np.concatenate([first,last])]

def four():
    st=U(); y=U(.41,.65); stem=U(.66,.94)
    if st < .70:
        a=(U(.18,.49),U(.0,.20)); j=(U(.0,.13),y)
        if U()<.45:
            bar=corner(a,j,(U(.9,1.24),y+U(-.14,.07)))
        else:
            bar=path(a,(a[0]-.16,(a[1]+y)*.5),j,(.28,y+.035),(.62,y), (U(.88,1.16),y+U(-.11,.035)))
        s=path((stem+U(-.04,.16),U(-.07,.10)),(stem,y),(stem+U(-.13,.13),1.))
        return [bar,s]
    elif st<.88:
        a=(stem,U(-.03,.06)); j=(0,y+U(.02,.12));
        return [corner(a,j,(1.11,y)),path(a,(stem-.02,y),(stem+U(-.1,.09),1.))]
    else:
        # Continuous, curved upright with a valley between two ascenders.
        return [path((.23,U(0,.14)),(.09,.26),(.01,y-.04),(.16,y+.06),(.48,y),(.77,y-.13),(stem,.13),(stem+.045,.0),(stem+.055,.26),(stem+.02,.64),(stem,1.))]

def five():
    wy=U(.36,.57); stem=U(.07,.27); upper=(stem+U(.04,.18),.025)
    lower=path(upper,(stem,.25),(stem-.07,wy),(.33,wy-.09),(.63,wy-.10),(.91,wy),(.99,wy+.20),(.85,.86),(.57,.98),(.23,1.0),(.00,U(.76,.97)))
    if U()<.31:
        # The top can be a separate long, rising stroke, or a short tick.
        start=(U(.00,.4),U(.045,.18)); end=(U(.67,1.24),U(-.16,.035))
        shelf=path(start,(.5*(start[0]+end[0]),.04),end)
        body=path((upper[0],U(.07,.21)),(stem,.25),(stem-.08,wy),(.35,wy-.095),(.72,wy-.02),(.91,wy+.20),(.77,.87),(.36,1.),(.02,U(.82,1.02)))
        return [shelf,body]
    if U()<.62:
        shelf=path((upper[0]-U(.01,.22),U(-.02,.07)),(.54,.028),(.81,.014),(1.03,U(-.08,.06)))
        return [lower,shelf]
    else:
        shelf=path((1.0,U(-.02,.08)),(.62,.01),upper)
        return [np.concatenate([shelf,lower])]

def six():
    h=U(.37,.60); lx=U(.02,.15); a=U(.51,.98)
    entry=path((a,0),(.61,.085),(.33,.28),(lx+.045,.52),(lx,.70),(lx+.095,.9),(.40,1.))
    bowl=path((.40,1.),(.71,.96),(.94,.81),(.94,h+.13),(.78,h+.015),(.54,h),(.31,h+.065),(lx+U(.00,.17),h+U(.14,.28)))
    if U()<.16:
        bowl=path((.40,1.),(.71,.96),(.94,.81),(.94,h+.13),(.77,h+.015),(.5,h),(.24,h+.10),(.04,h+.22),(-.08,h+.19))
    return [np.concatenate([entry,bowl])]

def seven():
    st=U(); a=(U(-.05,.10),U(-.01,.11)); b=(.99,.04); end=(U(.15,.56),1.)
    mid=(U(.54,.76),.52)
    if st<.56:
        top=path(a,(.40,.016),(.75,.035),b)
    elif st<.86:
        top=path((a[0]-.045,U(.15,.32)),a,(.31,.026),(.66,.06),b)
    else:
        top=path(a,(.32,.09),(.65,.08),b)
    stroke=np.concatenate([top,path(b,(.87,.22),mid,end)])
    if U()<.26:
        y=U(.45,.6)
        return [stroke,path((mid[0]-U(.19,.4),y), (mid[0],y+.012),(mid[0]+U(.13,.37),y+U(-.03,.04)))]
    return [stroke]

def eight():
    wy=U(.38,.59); cx=U(.42,.65); tw=U(.60,1.02); bw=U(.69,1.)
    # A narrow-waisted, tilted ribbon with independent widths, lobe
    # displacement, and imperfect joining of the ends of the stroke.
    if U()<.34:
        start=U(-2.9,2.9);t=np.linspace(start,start+2*np.pi+U(-.85,.55),150)
        v=np.cos(t); side=U(-.33,.33); rel=U(.46,.90) if U()<.5 else U(1.1,1.65)
        xx=.5+.55*np.sin(2*t)*(1+(rel-1)*(v+1)*.5)+side*v+.10*np.sin(t)*U(-1.,1.)
        yy=.5-.5*v
        xx+=U(-.22,.22)*np.linspace(0,1,len(t));yy+=U(-.11,.11)*np.linspace(0,1,len(t))
        return [np.stack((xx,yy),axis=1)]
    # Figure-eight, crossing on a diagonal rather than an abrupt junction.
    if U()<.77:
        top=path((cx,wy),(cx-.32*tw,wy-.12),(cx-.50*tw,.18),(cx-.33*tw,.044),(cx-.03,0),(cx+.32*tw,.055),(cx+.46*tw,.18),(cx+.31*tw,wy-.13),(cx,wy))
        bot=path((cx,wy),(cx-.32*bw,wy+.11),(cx-.53*bw,.79),(cx-.40*bw,.95),(cx-.095,1.0),(cx+.28*bw,.94),(cx+.44*bw,.77),(cx+.32*bw,wy+.13),(cx,wy))
        p=np.concatenate([top,bot])
        if U()<.42: p[:,0]=1.-p[:,0]
        # A few cursive eights have a slight gap at the crossing.
        cut=int(U(0,11))
        return [p[cut:]]
    else:
        return [loop(.5,wy*.5,tw*.50,wy*.50,-2.,4.40),loop(.48,(1+wy)*.5,bw*.5,(1-wy)*.5,-2.2,4.2)]

def nine():
    wy=U(.36,.58); stem=U(.78,.99)
    a=(stem,U(.06,.19)); x1=U(.00,.13)
    bowl=path(a,(.73,.025),(.47,0),(.20,.082),(x1,.23),(x1+.01,wy-.075),(.21,wy+.015),(.49,wy),(.72,wy-.08),a)
    st=U()
    if st<.45:
        leg=path(a,(stem-.015,.36),(stem+U(-.08,.055),.65),(stem+U(-.14,.19),1.0))
    elif st<.82:
        leg=path(a,(stem-.012,.34),(stem-.10,.62),(.59,.85),(U(.23,.55),1.0))
    else:
        leg=path(a,(stem+.025,.39),(stem+.02,.69),(.73,.91),(.52,1.),(U(.10,.33),U(.87,1.0)))
    return [np.concatenate([bowl,leg])]

draw = (zero,one,two,three,four,five,six,seven,eight,nine)

Y,X = np.mgrid[0:28,0:28]
STYLES = 5376  # independent templates per digit; not derived from examples

def render(k):
    paths=draw[k]()
    # Vary aspect, slant, mild rotation, and smooth local pen placement.
    aspect=U(.38,.95) if k!=1 else U(.43,.96)
    if U()<.15: aspect=U(.84,1.26)
    slant=U(-.47,.35) + U(-.13,.13)
    if U()<.17: slant=U(-.95,.57)
    if k==0 and U()<.22: aspect=U(.20,.53)
    rotation=U(-.21,.21)
    bend=U(-.14,.14); taper=U(-.29,.29)
    lean=U(-.20,.20); middle=U(.36,.64)
    thick=U(1.25,3.9)
    if U()<.14: thick=U(3.9,5.2)
    # A correlated displacement field shared by all strokes of one drawing.
    C=rng.normal(0,.043,(2,3,3))
    r=rng.uniform(-.20,.20,2)
    allpts=[]; radii=[]
    for p in paths:
        x=p[:,0]; y=p[:,1];
        gx=np.maximum(0,1-abs(2*x[:,None]-np.arange(3)))
        gy=np.maximum(0,1-abs(2*y[:,None]-np.arange(3)))
        delta=np.einsum('ij,kjl,il->ik',gx,C,gy)
        xx=aspect*(x+delta[:,0]-.5)*(1+taper*(2*y-1)) + slant*(y-.5)+bend*(2*y-1)**2
        cy=np.where(y<.5,2*middle*y,middle+2*(1-middle)*(y-.5))
        yy=cy+delta[:,1]-.5 + (rotation+lean*(2*y-1))*aspect*(x-.5)
        q=np.stack([xx, yy],1)
        allpts.append(q)
        phase=np.linspace(0,2*np.pi,len(q))*U(.35,.90)+U(0,2*np.pi)
        radii.append((thick*.5)*(1+r[0]*np.sin(phase)+r[1]*np.linspace(-1,1,len(q))))
    q=np.concatenate(allpts); radius=np.concatenate(radii)
    q*=(U(19.55,20.4)-thick)/np.ptp(q,axis=0).max()
    q-=.5*(q.min(0)+q.max(0)); q+=13.5
    # Soft union of round pen dabs on a regular grid. Scatter only in a small
    # neighbourhood of each dab. Linear coverage ramp for antialiasing.
    def raster(q):
        ix=np.rint(q[:,0]).astype(int); iy=np.rint(q[:,1]).astype(int)
        rr=int(np.ceil(radius.max()+1.0))
        dy,dx=np.mgrid[-rr:rr+1,-rr:rr+1]; dx=dx.ravel();dy=dy.ravel()
        i=ix[:,None]+dx;j=iy[:,None]+dy
        d=np.sqrt((q[:,0,None]-i)**2+(q[:,1,None]-j)**2)
        alpha=(radius[:,None]+.52-d).clip(0,1)
        ok=(i>=0)&(i<28)&(j>=0)&(j<28)
        out=np.zeros(784,F);np.maximum.at(out,(28*j+i)[ok],alpha[ok]);return out.reshape(28,28)
    first=raster(q); mass=first.sum()
    shift=13.5-np.array([(first*X).sum(),(first*Y).sum()])/mass
    shift+=U(-.53,.53,2)
    return raster(q+shift)

# Fixed front end. No parameters of these filters depend on a drawing.
G=8; D=8; S=8

def blur1(radius,sigma):
    g=np.exp(-.5*(np.arange(-radius,radius+1)/sigma)**2);return g/g.sum()

def directional():
    y,x=np.mgrid[-2:3,-2:3]; g=np.exp(-(x*x+y*y)/(2*1.05**2))
    w=[]
    for a in np.arange(G)*2*np.pi/G:
        f=(x*np.cos(a)+y*np.sin(a))*g
        w.append(f/np.maximum(f,0).sum())
    return np.array(w,F)[:,None]

def rays():
    w=np.zeros((D,1,17,17),F)
    for c,(dy,dx) in enumerate([(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]):
        for j in range(1,9):
            w[c,0,8+j*dy,8+j*dx]=(1.0-.04*j)/np.hypot(dy,dx)**.5
    return w

def sign(x): return x.clip(0,1)

# Smooth each isolated channel with a fixed separable spatial kernel.
def spatial(a,g):
    v=np.zeros_like(a);pad=len(g)//2
    b=np.pad(a,[(0,0)]*(a.ndim-2)+[(pad,pad),(0,0)])
    for j,u in enumerate(g): v+=u*b[...,j:j+a.shape[-2],:]
    out=np.zeros_like(v);b=np.pad(v,[(0,0)]*(a.ndim-1)+[(pad,pad)])
    for j,u in enumerate(g):out+=u*b[...,j:j+a.shape[-1]]
    return out

def znorm(a):
    a=a-a.mean(-1,keepdims=True);return a/np.sqrt((a*a).mean(-1,keepdims=True)+1.e-5)

def features(im):
    # Every operation here computes the descriptor of ONE individual drawing;
    # the first dimension only batches independent images for efficiency.
    n=len(im)
    ink=spatial(im,blur1(3,1.05).astype(F))[:,1::2,1::2]
    grad=windows(np.pad(im,((0,0),(2,2),(2,2))),(5,5),axis=(1,2))[:,::2,::2]
    grad=grad.reshape(-1,25) @ directional().reshape(G,-1).T
    grad=grad.reshape(n,14,14,G).transpose(0,3,1,2).clip(0)
    grad=spatial(grad,np.array([1,4,6,4,1],F)/16)
    grad=grad[:,:,1::2,1::2]
    # Directional probes of counterspace: how much clear space has an ink
    # boundary on each side, or on three sides of an open pocket?
    pad=np.pad(im,((0,0),(8,8),(8,8)))
    R=rays(); occ=np.zeros((n,D,14,14),F)
    for d in range(D):
        for y,x in zip(*np.nonzero(R[d,0])):
            occ[:,d]+=R[d,0,y,x]*pad[:,y:y+28:2,x:x+28:2]
    occ=sign(occ);empty=1-sign(1.5*im[:,::2,::2])
    regions=make_regions(); a=np.einsum('cd,ndij->ncij',regions,occ)+empty[:,None] - regions.sum(1)[None,:,None,None]
    a=a.clip(0);a=spatial(a,np.array([1,4,6,4,1],F)/16)
    a=a[:,:,1::2,1::2]
    # A continuous flood fill from each image's border: propagate by maxima
    # across background pixels, never above the local background capacity.
    # Pool the unreachable background to distinguish open and closed loops.
    room=(1-im.reshape(n,14,2,14,2).mean(axis=(2,4)))[:,None]
    rim=np.ones((14,14),F);rim[1:-1,1:-1]=0
    reach=room*rim
    for _ in range(13):
        rr=np.pad(reach,((0,0),(0,0),(1,1),(1,1)))
        reach=np.minimum(room,np.maximum.reduce([rr[:,:,j:j+14,i:i+14] for j in range(3) for i in range(3)]))
    holes=spatial(room-reach,np.array([1,4,6,4,1],F)/16)[:,:,1::2,1::2].reshape(n,-1)
    # Encode no-hole as an explicit coordinate. Balance each coordinate by
    # its negative so the norm op acts as length normalization, not DC removal.
    holes=np.column_stack((holes/7,np.full(n,.11,F)))
    holes/=np.sqrt((holes*holes).sum(-1,keepdims=True)+50.e-5)
    return [znorm(b.reshape(n,-1)) for b in (ink,grad,a)]+[holes]

def make_regions():
    mask=np.zeros((S,D),F)
    for i,group in enumerate(((0,1,2,3),(4,5,6,7),(0,1,2),(0,1,3),(0,2,3),(1,2,3),(0,1),(2,3))):
        mask[i,list(group)]=1
    return mask

layers=[];weights={}
def op(name,kind,ins,**kw):
    if isinstance(ins,str):ins=[ins]
    layers.append(dict(name=name,op=kind,inputs=ins,**kw));return name

def conv(name,src,w,b=None,st=1,pad=0):
    w=np.asarray(w,F);o=len(w); k=w.shape[-1]
    weights[name+'.W']=w;weights[name+'.b']=np.zeros(o,F) if b is None else np.asarray(b,F)
    return op(name,'conv',src,out=o,kernel=[k,k],stride=st,padding=pad)

def normall(name,src,C,H):
    v=op(name+'_vector','reshape',src,shape=[1,C*H,H]);return op(name,'norm',v)

def smooth(name,src,c):
    g=np.array([1,4,6,4,1],F)/16
    # Offset +1 relative to the even front-end sampling sites. Add zero top
    # and left kernel margin, then use padding 3 on the enlarged kernel.
    w=np.zeros((c,c,7,7),F)
    idx=np.arange(c);w[idx,idx,2:7,2:7]=np.outer(g,g)
    return conv(name,src,w,st=2,pad=3)

def front():
    # ink smoothing at positions 1,3,...,27 instead of 0,2,...,26
    g=blur1(3,1.05);w=np.zeros((1,1,9,9),F);w[0,0,2:,2:]=np.outer(g,g)
    ink=conv('ink_blur','image',w,st=2,pad=4)
    ink=normall('ink',ink,1,14)
    d=conv('direction','image',directional(),st=2,pad=2)
    d=op('positive_direction','relu',d)
    d=normall('stroke',smooth('spread_direction',d,G),G,7)
    R=rays(); W=np.zeros((D+1,1,17,17),F);W[:D]=R;W[D,0,8,8]=1.5
    v=conv('probe','image',W,st=2,pad=8)
    # min(ink,1) = 1 - relu(1-ink). A pocket is present only when the
    # selected rays see ink and there is no ink at the query point itself.
    clipped=conv('missing_ink',v,-np.eye(D+1,dtype=F)[:,:,None,None],np.ones(D+1,F))
    clipped=op('clear','relu',clipped)
    mask=make_regions();mix=np.column_stack([-mask,np.ones(S,F)])
    pocket=conv('pocket',clipped,mix[:,:,None,None]);pocket=op('positive_pocket','relu',pocket)
    pocket=normall('space',smooth('spread_space',pocket,S),S,7)
    return (ink,d,pocket,flood_front())

def dense(name,src,w,b):
    weights[name+'.W']=np.asarray(w,F);weights[name+'.b']=np.asarray(b,F)
    return op(name,'dense',src,out=len(w))

def flood_front():
    room=op('mean_ink','avgpool','image',size=[2,2],stride=[2,2]);unit=np.ones((1,1,1,1),F)
    room=conv('room',room,-unit,[1]); seed=np.eye(196,dtype=F).reshape(14,14,196)
    seed[1:-1,1:-1]=0
    reach=dense('rim',room,seed.reshape(196,196),np.zeros(196,F))
    reach=op('border','reshape',reach,shape=[1,14,14]); deficient=seed=None
    # min(room,max-neighbour-reach) = room - relu(room-max-neighbour-reach).
    # The last deficit is the enclosed background; only small scalar maps
    # need be propagated, regardless of the number of drawing templates.
    center=np.zeros((1,1,3,3),F);center[0,0,1,1]=1
    for i in range(13):
        r=conv('pad'+str(i),reach,center,pad=2);r=op('visit'+str(i),'maxpool',r,size=[3,3],stride=[1,1])
        r=conv('neg_visit'+str(i),r,-unit);r=op('deficit'+str(i),'add',[room,r]);deficient=op('unreachable'+str(i),'relu',r)
        if i!=12:
            r=conv('neg_deficit'+str(i),deficient,-unit);reach=op('reach'+str(i),'add',[room,r])
    holes=smooth('spread_holes',deficient,1)
    c=np.zeros((50,49),F);c[:49]=np.eye(49,dtype=F)/7;b=np.zeros(50,F);b[-1]=.11
    h=dense('signed_holes',holes,np.concatenate([c,-c]),np.concatenate([b,-b]))
    h=normall('cavity_norm',h,100,1)
    return dense('cavity',h,np.eye(50,100,dtype=F)/np.sqrt(50.),np.zeros(50,F))

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    src=front(); dims=(196,G*49,S*49,50); banks=[np.empty((10*STYLES,d),F) for d in dims]
    for k in range(10):
        for start in range(0,STYLES,48):
            m=min(48,STYLES-start);pos=k*STYLES+start
            batch=np.array([render(k) for _ in range(m)])
            for bank,v,coef in zip(banks,features(batch),(.29/196,.49/(G*49),.22/(S*49),.135)):
                bank[pos:pos+m]=v*coef
    score=[]
    for tag,srcname,w in zip(('silhouette','gradient','counterspace','enclosed'),src,banks):
        name=tag+'_match'; weights[name+'.W']=w;weights[name+'.b']=np.zeros(len(w),F)
        score.append(op(name,'dense',srcname,out=len(w)))
    s=op('agreement','add',score)
    # Split each class into independent panels and average their winners, a
    # more robust match than assigning a class by a single accidental hit.
    panels=6;t=STYLES//panels
    s=op('style_panels','reshape',s,shape=[10,panels,t]);s=op('panel_best','maxpool',s,size=[1,t],stride=[1,t])
    s=op('consensus','avgpool',s,size=[panels,1],stride=[panels,1]);op('scores','reshape',s,shape=[10])
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**weights)

if __name__=='__main__': main(Path(sys.argv[1]))
