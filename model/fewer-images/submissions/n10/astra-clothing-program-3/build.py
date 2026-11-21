"""Data-free construction. Lengths used by the garment and shoe patterns are in cm.
Each independent drawing is rasterized and saved; there is no template estimation.
"""
import sys
from pathlib import Path
import math
import numpy as np

rng = np.random.default_rng(91731)

def u(a,b):
    return rng.uniform(a,b)

def pick(seq):
    return seq[int(rng.integers(len(seq)))]

class Drawing:
    def __init__(self):
        self.parts=[]
    def poly(self,p,ink=True):
        self.parts.append((np.array(p,dtype=float),ink))
    def symmetric(self,half,ink=True):
        self.poly(half+[(-a,b) for a,b in half[::-1]],ink)
    def raster(self,tilt):
        parts=[]
        for vertices,ink in self.parts:
            v=vertices.copy()
            v[:,0]+=tilt*v[:,1]
            parts.append((v,ink))
        coords=np.concatenate([p for p,k in parts if k])
        lo=coords.min(axis=0); hi=coords.max(axis=0)
        aspect=(hi[0]-lo[0])/max(.01,hi[1]-lo[1])
        t=(np.arange(56)+.5)/56
        x=lo[0]+t*(hi[0]-lo[0]); y=lo[1]+t*(hi[1]-lo[1])
        m=np.zeros((56,56),bool)
        for v,k in parts:
            bit=np.zeros_like(m); x0,y0=v[-1]
            for x1,y1 in v:
                if abs(y1-y0)>1.e-9:
                    rows=(y>=min(y1,y0)) & (y<max(y1,y0))
                    inter=x0+(y[rows]-y0)*((x1-x0)/(y1-y0))
                    bit[rows]^=x[None,:]<inter[:,None]
                x0,y0=x1,y1
            if k: m|=bit
            else: m&=~bit
        # The longest side of the product photo has only 28 pixels: both
        # sides cannot have 28 independent samples when the item is narrow.
        # Model area sampling at native aspect, slight optical blur, and the
        # same soft support and recropping used for an input photograph.
        h=max(3,int(round(min(1,1/aspect)*27))); w=max(3,int(round(min(1,aspect)*27)))
        def area(n):
            a,b=np.arange(n)*56/n,(1+np.arange(n))*56/n
            t=np.arange(56)
            return np.maximum(0,np.minimum(b[:,None],t+1)-np.maximum(a[:,None],t))*n/56
        coverage=np.pad(area(h)@m.astype(float)@area(w).T,2)
        p=np.pad(coverage,1); smooth=(4*coverage+2*(p[1:-1,:-2]+p[1:-1,2:]+p[:-2,1:-1]+p[2:,1:-1])+p[:-2,:-2]+p[:-2,2:]+p[2:,:-2]+p[2:,2:])/16
        coverage=.76*coverage+.24*smooth
        opacity=np.clip((coverage-.040)/.180,0,1)
        row=opacity.sum(1); col=opacity.sum(0)
        ys=np.flatnonzero(row>max(.52,row.max()*.034)); xs=np.flatnonzero(col>max(.52,col.max()*.034))
        if len(ys) and len(xs):
            y0,y1=ys[0],ys[-1]; x0,x1=xs[0],xs[-1]
            h=y1-y0+1; w=x1-x0+1; t=(np.arange(28)+.5)/28
            x=np.clip(x0-.5+w*t,0,opacity.shape[1]-1); y=np.clip(y0-.5+h*t,0,opacity.shape[0]-1)
            xi=x.astype(int); yi=y.astype(int); xj=np.minimum(xi+1,opacity.shape[1]-1); yj=np.minimum(yi+1,opacity.shape[0]-1); fx=x-xi; fy=(y-yi)[:,None]
            z=(1-fy)*((1-fx)*opacity[yi[:,None],xi]+fx*opacity[yi[:,None],xj])+fy*((1-fx)*opacity[yj[:,None],xi]+fx*opacity[yj[:,None],xj])
            return (255*z+.5).astype(np.uint8),w/h
        return (m.reshape(28,2,28,2).sum(axis=(1,3))*63.75+.5).astype(np.uint8),aspect

# Torso block, attached sleeves, neck opening and hem. The same construction is
# shared by the tops: it is their cut, lengths and ease that differ.
def garment(c,i):
    d=Drawing(); bias=0.
    shoulder=u(15,23); chest=shoulder*u(1.01,1.24); pit=u(15,23)
    neck=u(6,11); drop=u(3,8); base=u(56,73)
    waist=chest*u(.84,1.02); hip=chest*u(.90,1.08)
    thigh=hip; curv=u(-1,2.5); arm='long'; collar='round'
    depth=u(1,5); ease=u(-4,3); flare=u(-2,2); wrist=u(7,10.5)
    skew=0.; size=i%100
    if c==0:
        arm='short' if size<68 else ('bare' if size<94 else 'mid')
        curv=u(-1,2.5); depth=u(1,8); drop=u(3,7)
        if arm=='bare':
            shoulder*=u(.67,1); neck=u(.43,.88)*shoulder; depth=u(3,13)
            chest=shoulder*u(1.16,1.53); hip=chest*u(.88,1.16); waist=chest*u(.82,1)
        if arm=='mid': bias=.014
        if i%6==0: collar='vee'
    if c==2:
        base=u(55,74); chest*=u(1,1.13); hip=chest*u(.83,1.04)
        waist=chest*u(.91,1.03); ease=u(-4,4); flare=u(-1,4)
        curv=u(-1,1.7); wrist=u(6.5,10); depth=u(.4,4.5)
        if size<13: collar='high'
        if 13<=size<25: collar='vee'; depth=u(2,8)
        if 25<=size<31: arm='mid'; bias=.013
        if size>=85: base=u(46,60); chest*=1.05
    if c==3:
        base=u(78,125); shoulder*=u(.8,1.02); chest=shoulder*u(1.01,1.25)
        arm='bare' if size<44 else ('short' if size<78 else ('long' if size<93 else 'mid'))
        if arm=='bare': shoulder*=u(.65,1.02); neck=shoulder*u(.45,.88)
        hip=chest*u(.98,1.25); waist=chest*u(.67,.99)
        thigh=hip*pick([u(1.05,1.62),u(.86,1.16),u(.85,1.08)])
        depth=u(1,12); curv=u(-1,4); wrist=u(6,9)
        if i%4==0: collar='vee'
        if i%7==0: base=u(67,83); bias=.011
    if c==4:
        collar=pick(['stand','stand','hood','lapel','lapel','round'])
        base=u(62,91) if size<67 else u(51,65)
        shoulder=u(17,25); chest=shoulder*u(1.01,1.26); waist=chest*u(.84,1.05)
        hip=chest*u(.88,1.20); wrist=u(8.5,13.5); ease=u(-6,3); curv=u(-1.5,2.5)
        if i%8==0: base=u(88,108); hip*=1.08
        neck=u(5,10); depth=u(.1,3); drop=u(3,7.5)
    if c==6:
        arm='long' if size<58 else ('mid' if size<75 else ('short' if size<93 else 'bare'))
        collar=pick(['stand','stand','point','round','vee','round']); base=u(56,82)
        chest=shoulder*u(1.04,1.30); waist=chest*u(.78,1.05); hip=chest*u(.90,1.17)
        wrist=u(6,10.5); ease=u(-5,4); curv=u(1.5,8); depth=u(1,7)
        if arm=='bare': neck=shoulder*u(.45,.82); shoulder*=.88; bias=.017
        if arm=='short': bias=.010
        if i%5==0: curv=u(-1.5,2); skew=u(-.06,.06)
    shoulder=max(shoulder,neck+1.3)
    waistY=min(u(32,43),base*.65); hipY=waistY+(base-waistY)*u(.32,.64)
    edge=thigh if c==3 else hip
    up=0.
    if collar in ('hood','stand','lapel','point','high'):
        up=u(2,6) if collar!='hood' else u(5,13)
        if collar=='lapel': up=u(1,4)
        depth=u(.1,2.5); neck=min(neck,shoulder*.65)
    half=[(neck*.7,-up),(neck,-up*.78),(neck*1.1,.6),((shoulder+neck)/2,drop*.37),(shoulder,drop)]
    if arm=='bare': half.append((chest*u(.77,.92),pit*.65))
    half+= [(chest,pit),(waist,waistY),(hip,hipY),(edge,base-curv),(.77*edge,base-curv*.48),(.40*edge,base-curv*.10),(0.,base)]
    d.symmetric(half)
    # A set-in tee sleeve can lie as a rectangular stub with a nearly level
    # hem; a flutter sleeve instead ends on the oblique grain. Also include a
    # loose blouse's dropped shoulder and broad, gathered upper arm.
    setin=arm=='short' and i%3!=0
    loose=c==6 and i%7==0 and arm!='bare'
    if arm!='bare':
        if arm=='long': length=u(51,65)
        if arm=='mid': length=u(28,45)
        if arm=='short': length=u(10,21)
        theta=u(.48,1.03) if arm=='short' else u(.13,.40)
        if c==6 and arm=='mid': theta=u(.16,.65)
        for s in [-1,1]:
            drift=u(-1.8,1.8); lengthSide=length+u(-1.7,1.7)
            if arm=='short':
                w=u(11,19); dx=lengthSide*math.sin(theta)
                ox=shoulder+dx; oy=drop+lengthSide*math.cos(theta)
                ix=ox-w*math.cos(theta); iy=oy+w*math.sin(theta)
            else:
                w=wrist if arm=='long' else wrist*u(1.06,1.50)
                ix=max(waist*.78, (hip if c!=3 else waist)+ease+drift); ox=ix+w
                oy=drop+lengthSide*math.cos(theta); iy=oy+u(-1.5,2.5)
            bend=flare+u(-1.5,2.5); mx=.46*shoulder+.54*ox+bend
            my=.47*drop+.53*oy
            p=[(shoulder*.93,drop-.25),(shoulder+1,drop+.8),(mx,my),(ox,oy),(ix,iy),(.5*(ix+chest)+bend*.25,.5*(iy+pit)),(chest-.5,pit),(shoulder*.87,drop+2)]
            if setin:
                outer=shoulder+u(7,15); end=drop+lengthSide
                p=[(shoulder*.93,drop-.25),(shoulder+1.2,drop+.8),(shoulder+.68*(outer-shoulder),drop+.39*lengthSide),(outer,end),(chest-.7,end+u(-1,2)),(chest-1,pit),(shoulder*.87,drop+2)]
            if loose:
                p[2]=(p[2][0]+u(2,6),p[2][1]); p[3]=(p[3][0]+u(1,5),p[3][1])
            d.poly([(s*a,b) for a,b in p])
    # Cut the neck after joining the pieces. Negative y denotes collar height.
    top=-up
    if collar=='vee': h=[(neck,-up-2),(neck,top),(.54*neck,top+depth*.48),(0,top+depth)]
    else: h=[(neck*.94,top-2),(neck*.94,top),(.84*neck,top+depth*.57),(.57*neck,top+depth*.9),(0,top+depth)]
    d.symmetric(h,False)
    return d,bias,skew+u(-.045,.045)

def trousers(i):
    d=Drawing(); l=u(87,113); hip=u(16,24); belt=hip*u(.73,1); crotch=u(22,38)
    ankle=hip*u(.18,.48); stance=hip*u(.35,.71); knee=ankle*u(1.02,1.36)
    if i%8==0: ankle=hip*u(.43,.64); stance=ankle+u(.0,3); knee=ankle*u(.74,1)
    # Each leg is suspended below its own hip, even when the ankles are drawn
    # together or the cuffs are so wide that the inseams almost meet.
    inside=max(.05,stance-ankle); kneecenter=(stance+hip*.53)*.5
    d.symmetric([(0,0),(belt,.4),(hip*.97,9),(hip,17),(hip*.97,crotch),(kneecenter+knee,.59*l),(stance+ankle,l),(inside,l),(max(.12,kneecenter-knee),.56*l),(0,crotch)])
    return d,0.,u(-.05,.05)

# A side elevation of a foot: toes on the left, y is height above the ground.
def footwear(c,i):
    d=Drawing(); bias=0.; L=u(24,29); heel=u(.4,2); toe=u(1.9,3.8); sole=u(1,2.6)
    rise=u(.0,.9); block=u(.14,.27); arch=u(.2,1.3); toeX=u(.20,.36)
    end=u(.91,.99); tilt=u(-.09,.09); mode=i%100
    if c==9:
        heel=pick([u(.7,2.7),u(2,5),u(4,8)]); arch=heel*u(.51,.99); block=u(.09,.26)
    if c==5:
        heel=pick([u(.1,2),u(1,4),u(4,10)]); arch=heel*u(.15,.96); sole=u(.5,1.9); block=u(.055,.23)
    # The heel can be a solid wedge, a block, or a separate column below the arch.
    archX=u(.48,.69)
    bottom=[(.0,rise+sole*.48),(.028,rise*.8),(.13,.12),(.38,0),(archX,arch*.77),(.99-block,arch),(.99-block*.9,.14),(.98,.12),(1.,.5)]
    if c==7:
        bottom=[(.0,rise+sole*.80),(.035,rise*.75),(.18,.02),(.4,0),(.61,arch*.26),(.81,.14),(.98,.18),(1.,.48)]
    def shape(points,ink=True): d.poly([(L*a,-b) for a,b in points],ink)
    back=heel+sole
    if c==7:
        cuff=u(4.8,8.5)+sole; tongue=cuff+u(-.65,3.2); low=cuff*u(.70,.94)
        if mode<19: cuff=u(8,12.3); tongue=cuff+u(.3,4); low=cuff*u(.82,.98); bias=.006
        if 19<=mode<31: cuff=u(4,7); tongue=cuff*u(.72,.98); low=cuff*u(.61,.88)
        tongueX=u(.49,.69); lowX=u(.71,.86)
        shape(bottom+[(1.,cuff*.61),(end,cuff),(end-.045,cuff*.99),(lowX,low),(tongueX+.04,tongue*.90),(tongueX,tongue),(tongueX-.06,tongue*.88),(toeX,toe+sole),(.11,toe+sole*.83),(.025,sole+toe*.56),(.0,sole+toe*.18)])
    if c==9:
        shaft=pick([u(6,10),u(9,15),u(13,20)])+back
        front=u(.42,.66); ankle=u(.51,.72); toepeak=toe+sole
        lip=shaft*u(-.10,.10); rear=u(.90,1.00)
        shape(bottom+[(.99,back+1),(rear-.01,back+(shaft-back)*.53),(rear,shaft+lip),((rear+front)*.5,shaft+.3*lip),(front,shaft),(front+.013,shaft*.86),(ankle,back+.33*(shaft-back)),(ankle-.11,back+.20*(shaft-back)),(toeX,toepeak+u(0,.9)),(.13,toepeak),(.025,sole+.61*toe),(.0,sole+toe*.25)])
    if c==7 and i%5==0:
        # Lace-up casual high-top: a pitched tongue ABOVE the heel collar, with
        # a rubber sidewall and plenty of material over the forefoot.
        d=Drawing(); yh=u(6,11); yt=yh+u(1.2,4.2); tip=u(.47,.67); toepeak=sole+u(2.5,4)
        shape(bottom+[(.985,yh*.74),(.94,yh),(.85,yh*u(.82,1.0)),(.74,yh*u(.81,.98)),(tip+.06,yt*.95),(tip,yt),(tip-.07,yt*.85),(.36,toepeak+1),(.19,toepeak),(.065,toepeak*.84),(.01,sole+.8)])
    if c==5:
        deck=[(1.,back),(.86,back+.1),(.72,back-.1),(.54,.61*heel+sole),(.36,sole+.11*heel),(.18,sole),(.03,sole+rise),(.0,sole+rise*.9)]
        shape(bottom+deck)
        def floor(a): return sole+heel*max(0,min(1,(a-.28)/.48))
        def strap(a,b,h,w):
            shape([(a,floor(a)-.4),(a+w,floor(a+w)-.4),(b+w,h),(b,h+.06)])
        if mode<24: # slide, flip-flop or mule, with an open back
            f=u(.03,.21); b=u(.37,.65); h=u(2.4,4.8)+sole
            shape([(f,floor(f)-.4),(b+.10,floor(b+.10)-.3),(b+.05,h),(b-.04,h+.6),(f+.04,sole+toe*.69)])
            if mode<9: shape([(.0,sole+.2),(.14,sole),(.35,h*.91),(.27,h)])
        elif mode<86:
            a=u(.02,.18); b=u(.21,.45); h=u(3,5)+sole
            strap(a,b,h,u(.06,.19)); strap(b+u(-.09,.08),b,h,u(.055,.17))
            cuff=back+u(3.2,9.0); pos=u(.59,.82)
            strap(u(.58,.80),pos,cuff,u(.045,.12)); strap(u(.81,.96),pos+u(.09,.22),cuff,u(.04,.09))
            shape([(pos-.06,cuff-1.3),(min(1,pos+.22),cuff-1.4),(min(1,pos+.25),cuff+.25),(pos-.1,cuff)])
            if i%3==0: strap(u(.31,.50),pos,cuff-1,u(.04,.12))
        else: # closed-toe pump / peep toe shoe with deep cutaway
            shape([(.0,sole),(.02,sole+.6*toe),(.15,sole+toe),(.33,sole+toe*1.3),(.45,sole+toe*.6),(.66,back+.1),(.92,back+u(3,6)),(.99,back+3),(1.,back-.4),(.70,back-.4),(.31,sole-.4)])
    if c==5 and i%6==0:
        # Walking sandal: a rubber outsole with opaque side panels. Windows in
        # the vamp reveal the opposite straps; these shoes lack a boot shaft.
        d=Drawing(); ankle=back+u(2.4,5.4); peak=sole+u(3.1,6.5)
        shape(bottom+[(.99,ankle*.78),(.94,ankle),(.86,ankle*.94),(.75,ankle*.75),(.63,peak*.96),(.54,peak),(.38,peak*.91),(.22,peak*.71),(.06,sole+toe*.64),(.0,sole+.4)])
        for start in (.08,.34,.59,.80):
            height=(peak if start<.68 else ankle)-floor(start)
            pos=start+u(-.04,.04); width=u(.075,.19); low=floor(pos)+u(-.15,.4)
            shape([(pos,low),(pos+width,low+.2),(pos+width*.84,low+height*.80),(pos+width*.56,low+height*.82),(pos+width*.09,low+height*.35)],False)
    if c==5 and i%7==0:
        # Flat leather/crisscross sandal; both attachments of each thin band
        # meet the sole, leaving a triangular air space underneath the band.
        d=Drawing(); heel=u(.2,1.6); back=heel+sole
        shape([(.0,sole*.77),(.08,.05),(.42,0),(.71,heel*.23),(.98,heel*.19),(1.,back*.85),(.91,back),(.65,back*.91),(.37,sole*.98),(.09,sole+rise*.4)])
        for start,w in [(.04,u(.24,.53)),(u(.38,.62),u(.21,.38))]:
            h=floor(start)+u(1.8,4.4); mid=start+w*u(.4,.70); thick=u(.045,.16)
            strap(start,mid,h,thick); strap(min(.96-thick,start+w),mid,h,thick)
        if i%2==0: strap(.90,.83,back+u(2.3,4),.06)
    return d,bias,tilt

def bag(i):
    d=Drawing(); w=u(24,48)*.5; h=w*u(.92,2.04); tip=w*u(.79,1.02); bot=w*u(.70,.98)
    rounded=u(.03,.27); side=u(.36,.78)*h; mouth=h*u(.0,.13)
    d.symmetric([(0,mouth),(tip*.78,mouth*.4),(tip,0),(w,side),(bot+(w-bot)*.68,h*(1-rounded)),(bot,h*.96),(.74*bot,h),(0,h)])
    if i%100<79:
        hh=w*u(.45,1.66); ww=tip*u(.32,.81); thickness=u(1.0,3.3); join=mouth+u(1,4); off=u(-.10,.10)*w
        if i%7==0: hh*=1.6; ww=tip*u(.75,1.0); thickness*=.70
        # Curved leather tape / hoop rising from the body of the bag.
        outer=[]; inner=[]
        for t in np.linspace(0,math.pi,17):
            x=math.cos(t); y=math.sin(t); outer.append((ww*x+off*y,join-(hh+join)*y**.72)); inner.append(((ww-thickness)*x+off*y,join-(hh+join-thickness)*y**.72))
        d.poly(outer+inner[::-1]);
    return d,0.,u(-.12,.12)

SOLUTION = r'''import numpy as np


def blur(a):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    return ((p[:,1:-1,:-2]+p[:,1:-1,2:]+p[:,:-2,1:-1]+p[:,2:,1:-1])*2+p[:,:-2,:-2]+p[:,2:,:-2]+p[:,:-2,2:]+p[:,2:,2:]+4*p[:,1:-1,1:-1])*(1/16)


def pool(a):
    return a.reshape(-1,14,2,14,2).mean(axis=(2,4))


def unit(a):
    return np.clip(a,0,1)


def features(m,aspect):
    # Envelopes from both axes reduce sensitivity to printing and fractured edges.
    m=blur(m)*.50+m*.50
    left=np.maximum.accumulate(m,axis=2); right=np.maximum.accumulate(m[:,:,::-1],axis=2)[:,:,::-1]
    top=np.maximum.accumulate(m,axis=1); bottom=np.maximum.accumulate(m[:,::-1,:],axis=1)[:,::-1,:]
    row=np.minimum(left,right); col=np.minimum(top,bottom)
    measures=[]
    for a,w in [(m,.60),(row,.21),(col,.19)]:
        measures.append(pool(a).reshape(len(m),196)*(w**.5/14))
    for p in (left.mean(2),right.mean(2),top.mean(1),bottom.mean(1)):
        measures.append(p.reshape(-1,14,2).mean(2)*(.095/14)**.5)
    measures.append(np.log(np.maximum(.06,aspect))[:,None]*(.24**.5))
    return np.ascontiguousarray(np.concatenate(measures,axis=1),dtype=np.float32)


def makebank():
    pixels=CONSTANTS['silhouettes']; aspect=CONSTANTS['aspect']; costs=CONSTANTS['costs']; counts=CONSTANTS['count']
    result=[]; offset=0
    for n in counts:
        bank=features(pixels[offset:offset+n].astype(np.float32)*(1/255),aspect[offset:offset+n])
        result.append((np.ascontiguousarray(bank.T),np.sum(bank*bank,axis=1)+costs[offset:offset+n]))
        offset+=n
    return result


banks=makebank()


def sample(a,ys,xs):
    yi=np.floor(ys).astype(int); xi=np.floor(xs).astype(int)
    wy=(ys-yi).astype(np.float32)[:,:,None]; wx=(xs-xi).astype(np.float32)[:,None,:]
    yj=np.minimum(yi+1,27); xj=np.minimum(xi+1,27)
    b=np.arange(len(a))[:,None,None]
    return ((1-wy)*((1-wx)*a[b,yi[:,:,None],xi[:,None,:]]+wx*a[b,yi[:,:,None],xj[:,None,:]])+wy*((1-wx)*a[b,yj[:,:,None],xi[:,None,:]]+wx*a[b,yj[:,:,None],xj[:,None,:]]))


def prepare(a):
    peak=np.maximum(.04,np.quantile(a,.965,axis=(1,2)).astype(np.float32))[:,None,None]
    sm=.73*a+.27*blur(a)
    m=unit((sm-.005-.022*peak)/(.022+.083*peak))
    # Suppress isolated glints when locating the bounding rectangle, but allow
    # a handle or narrow shoulder strap to define the top.
    rows=m.sum(2); cols=m.sum(1)
    rows=rows>np.maximum(.52,rows.max(1,keepdims=True)*.034)
    cols=cols>np.maximum(.52,cols.max(1,keepdims=True)*.034)
    y0=rows.argmax(1); y1=27-rows[:,::-1].argmax(1); x0=cols.argmax(1); x1=27-cols[:,::-1].argmax(1)
    w=x1-x0+1; h=y1-y0+1; t=(np.arange(28,dtype=np.float32)+.5)/28
    xs=np.clip((x0-.5)[:,None]+w[:,None]*t,0,27); ys=np.clip((y0-.5)[:,None]+h[:,None]*t,0,27)
    occ=sample(m,ys,xs); ink=sample(a/peak,ys,xs)
    # One local closing bridges small cracks in otherwise solid fabric.
    p=np.pad(occ,((0,0),(1,1),(1,1)),mode='edge'); hi=occ
    for dy,dx in ((0,1),(2,1),(1,0),(1,2)):
        hi=np.maximum(hi,p[:,dy:dy+28,dx:dx+28])
    p=np.pad(hi,((0,0),(1,1),(1,1)),mode='edge'); close=hi
    for dy,dx in ((0,1),(2,1),(1,0),(1,2)):
        close=np.minimum(close,p[:,dy:dy+28,dx:dx+28])
    return .68*occ+.32*close,ink,(w/h).astype(np.float32)


def matches(m,aspect):
    direct=features(m,aspect); reverse=features(m[:,:,::-1].copy(),aspect)
    pair=np.concatenate([direct,reverse]); base=(pair*pair).sum(1,keepdims=True); n=len(m)
    fits=[]
    for bank,offset in banks:
        dist=(base+offset[None,:])-2*(pair@bank)
        d=np.minimum(dist[:n],dist[n:])
        # Marginalize a little over pose: favor a match with several close cuts,
        # rather than depending entirely on one exact pose.
        near=np.partition(d,5,axis=1)[:,:6]
        fits.append(-(.70*near.min(1)+.30*near.mean(1)))
    return np.stack(fits,axis=1)


def ramp(x,lo,hi):
    return unit((x-lo)/(hi-lo))


def cues(m,a,aspect):
    s=np.zeros((len(a),10)); smooth=blur(a)
    # Find a placket or open front: a vertical edge, ridge, or trench that
    # persists down the center front. Compare against folds away from center.
    gx=smooth[:,:,2:]-smooth[:,:,:-2]; ridge=.5*(a[:,:,2:]+a[:,:,:-2])-a[:,:,1:-1]
    col=np.maximum(np.abs(gx),np.abs(ridge)*1.25)
    side=np.mean(col[:,7:23,5:21],axis=(1,2)); line=np.zeros(len(a))
    for slope in (-2,0,2):
        step=np.rint(np.linspace(-slope/2,slope/2,18)).astype(int)
        for center in range(10,16):
            ix=center+step; iy=np.arange(6,24)
            # Take the strongest polarity-consistent edge in a narrow corridor.
            mag=np.maximum(col[:,iy,ix],.81*np.maximum(col[:,iy,ix-1],col[:,iy,ix+1]))
            pol=gx[:,iy,ix]; coherent=np.maximum(abs(pol.mean(1)),abs(ridge[:,iy,ix].mean(1)))
            spans=mag.reshape(len(a),3,6).mean(2)
            quality=.53*spans.min(1)+.32*spans.mean(1)+.15*coherent
            line=np.maximum(line,quality)
    rel=line/(.055+side)
    closure=ramp(line,.045,.22)*(.42+.58*ramp(rel,.86,1.94))
    # Paired lapel diagonals. A single stripe across the body isn't a lapel.
    diag=np.abs(gx); lapel=np.zeros(len(a))
    for center in (12,13,14):
        for end in (8,12,16):
            iy=np.arange(2,end); dx=np.rint(np.linspace(4.8,1.0,len(iy))).astype(int)
            l=diag[:,iy,center-dx].mean(1); r=diag[:,iy,center+dx].mean(1)
            lapel=np.maximum(lapel,.65*np.minimum(l,r)+.35*(l+r)*.5)
    lapel=ramp(lapel-.66*side,.045,.22)
    # Texture outside the central print area, measured relative to the lighting
    # of the item. Most jackets are not floral/checked like a printed blouse.
    detail=np.abs(a-smooth); cloth=np.maximum(.15,a[:,7:23,7:21].mean(axis=(1,2)))
    texture=(detail[:,7:22,7:11].mean(axis=(1,2))+detail[:,7:22,17:21].mean(axis=(1,2)))*.5
    pattern=ramp(texture/(.36+cloth),.043,.13)
    centerart=detail[:,6:16,10:18].mean(axis=(1,2))
    graphic=ramp(centerart-texture,.02,.11)*(1-closure)
    # A stand-up collar is a central protrusion above both shoulders. A tank
    # or scoop-neck, conversely, has a central dip between narrow straps.
    crown=(m>.35).argmax(1)/28
    center=crown[:,12:16].mean(1); neckside=.5*(crown[:,8:11].mean(1)+crown[:,17:20].mean(1))
    outer=.5*(crown[:,5:8].mean(1)+crown[:,20:23].mean(1))
    raised=ramp(neckside-center,.035,.19)*ramp(outer-center,.085,.28)
    dip=ramp(center-neckside,.038,.23)
    # Curved shirttails terminate earlier at the hips than on the center line.
    end=(m[:,::-1,:]>.40).argmax(1)/28
    curved=ramp(.5*(end[:,7:10].mean(1)+end[:,18:21].mean(1))-end[:,12:16].mean(1),.052,.18)
    # Interpretation of silhouette similarity as negative discrepancy; appearance
    # evidence is deliberately bounded in those same units of silhouette error.
    # A thin center channel between legs, missed by a coarse occupancy grid.
    # Look for two separately supported columns, not merely a dark fold.
    cleft=np.zeros(len(a))
    for x in range(9,19):
        core=1-m[:,11:27,x]; fabric=np.minimum(m[:,11:27,x-7:x-3].mean(2),m[:,11:27,x+4:x+8].mean(2))
        split=(core*fabric).mean(1)
        darkness=np.maximum(0,np.minimum(a[:,11:27,x-6:x-2].mean(2),a[:,11:27,x+3:x+7].mean(2))-a[:,11:27,x])
        cleft=np.maximum(cleft,split+.22*np.minimum(.8,darkness).mean(1))
    belt=m[:,1:4,5:23].mean(axis=(1,2)); tapered=m[:,5:10,:].mean(axis=(1,2))-m[:,20:25,:].mean(axis=(1,2))
    legs=ramp(cleft,.11,.62)*ramp(belt,.44,.82)*(1-ramp(aspect,.58,.83))
    s[:,1]+=.110*legs+.026*ramp(tapered,.04,.26)*ramp(belt,.70,.98)*(1-ramp(aspect,.53,.77))
    waistband=ramp(m[:,:2,4:24].mean(axis=(1,2)),.63,.92)*(1-ramp(aspect,.52,.74))
    s[:,1]+=.045*waistband*(.40+.60*ramp(tapered,.03,.24))
    s[:,3]-=.040*legs
    # A skirt continues below the waist with uninterrupted fabric across its
    # middle. A sleeve ending at waist height is not a trouser leg.
    rowsize=np.minimum(np.maximum.accumulate(m,2),np.maximum.accumulate(m[:,:,::-1],2)[:,:,::-1]).mean(2); waistwidth=rowsize[:,10:19].min(1); hemwidth=rowsize[:,22:26].mean(1)
    skirt=ramp(hemwidth-waistwidth,.052,.26)*ramp(m[:,21:25,8:20].mean(axis=(1,2)),.72,.96)
    slender=1-ramp(aspect,.45,.73)
    s[:,3]+=.037*skirt*(.36+.64*slender)+.019*slender*(1-waistband)*(1-legs)
    s[:,0]+=.005+.014*graphic+.006*dip-.020*closure-.011*raised-.006*lapel
    s[:,2]+=.010+.020*graphic-.041*closure-.013*lapel-.010*pattern-.006*curved
    s[:,3]+=.002-.006*raised
    s[:,4]+=-.004+.037*closure+.017*raised+.020*lapel-.021*pattern-.007*dip-.009*graphic
    s[:,6]+=-.009+.018*closure+.013*lapel+.026*pattern+.021*curved-.008*graphic
    # A sandal has holes between straps: look for nearly empty pixels enclosed
    # both transversely and vertically, not just a curved sole or toe line.
    row=np.minimum(np.maximum.accumulate(m,axis=2),np.maximum.accumulate(m[:,:,::-1],axis=2)[:,:,::-1]); colh=np.minimum(np.maximum.accumulate(m,axis=1),np.maximum.accumulate(m[:,::-1,:],axis=1)[:,::-1,:]); holes=(np.minimum(row,colh)*(1-m))[:,3:23,3:25].mean(axis=(1,2))
    hole=ramp(holes,.053,.235)
    # A change in brightness within a black or shiny shoe can still delineate
    # an opening even when the outer silhouette survives smoothing. Count low
    # regions which have substantially lighter material on opposite sides.
    shade=unit((a-.035)/.17)
    hrow=np.minimum(np.maximum.accumulate(shade,2),np.maximum.accumulate(shade[:,:,::-1],2)[:,:,::-1]); hcol=np.minimum(np.maximum.accumulate(shade,1),np.maximum.accumulate(shade[:,::-1,:],1)[:,::-1,:])
    see=ramp((np.minimum(hrow,hcol)*(1-shade))[:,4:24,3:25].mean(axis=(1,2)),.075,.26)
    # Turn the side elevation toe-left using height of material in the upper.
    flip=m[:,:14,:10].sum(axis=(1,2))>m[:,:14,-10:].sum(axis=(1,2))
    profile=np.where(flip[:,None,None],m[:,:,::-1],m); present=profile>.48
    roof=np.where(present.any(1),present.argmax(1),28)/28
    rear=roof[:,20:25].mean(1); tongue=roof[:,13:18].min(1); back=roof[:,22:26].mean(1); fore=roof[:,4:10].mean(1)
    # Boots rise well above the toe and have a filled shaft beneath a fairly
    # level cuff. Trainers generally have a tongue ahead of a dipped topline.
    shaft=ramp(fore-rear,.30,.61)*(1-ramp(rear,.12,.31))*(1-ramp(aspect,1.85,2.65))
    notch=ramp(back-tongue,.085,.28)
    noheel=ramp(back,.43,.84)*ramp(roof[:,11:18].max(1)-tongue,.17,.42)
    arch=present[:,::-1,:].argmax(1)/28
    heel=ramp(arch[:,16:22].max(1)-arch[:,22:26].min(1),.065,.23)
    # Reject an apparent cuff when there is air directly beneath it. A narrow
    # horizontal strap at ankle height otherwise looks remarkably like a boot.
    ankleband=profile[:,9:21,18:25].mean(axis=(1,2)); shaft*=ramp(ankleband,.64,.96)
    shadow=np.where(flip[:,None,None],shade[:,:,::-1],shade)
    clear=1-shadow
    # Local maximum opening with darker center than the sole and the arching
    # straps above and on either side. Erode support to demand an actual patch
    # of air, not a seam, then search at the scale of a few pixels.
    sky=np.minimum(np.maximum.accumulate(shadow,axis=1),np.maximum.accumulate(shadow[:,::-1,:],axis=1)[:,::-1,:]); walls=np.minimum(np.maximum.accumulate(shadow,axis=2),np.maximum.accumulate(shadow[:,:,::-1],axis=2)[:,:,::-1]); air=clear*np.minimum(sky,walls)
    def patchmax(f,by,bx):
        p=np.pad(f,((0,0),(1,0),(1,0))); integral=p.cumsum(1).cumsum(2)
        avg=(integral[:,by:,bx:]+integral[:,:-by,:-bx]-integral[:,:-by,bx:]-integral[:,by:,:-bx])/(by*bx)
        return avg.max(axis=(1,2))
    window=ramp(patchmax(air[:,3:23,2:26],5,4),.41,.88)
    topline=blur(profile).sum(axis=1)/28
    backopen=ramp(topline[:,9:18].max(1)-topline[:,22:26].mean(1),.28,.62)
    cleartoe=ramp(roof[:,3:6].mean(1),.69,.92)*ramp(roof[:,8:12].mean(1)-tongue,.24,.57)
    # A trainer has a padded tongue in front of the collar. Its topline steps
    # DOWN behind the tongue, unlike the straight rim of a boot shaft.
    padded=ramp(roof[:,18:23].max(1)-tongue,.10,.33)*ramp(aspect,1.20,1.82)
    solid=(1-window)*(1-hole)*ramp(ankleband,.69,.97)
    s[:,5]+=.065*hole+.055*see+.057*noheel+.020*heel*(1-shaft)+.072*window+.036*backopen+.012*cleartoe
    s[:,7]+=.008-.029*hole+.018*notch-.018*heel-.031*noheel+.034*padded*solid-.024*window
    s[:,9]+=.002-.025*hole+.046*shaft*solid+.016*heel-.019*notch-.026*noheel-.027*padded-.042*window
    return s


def cut_and_fabric(m,a,aspect):
    # Return extra evidence for body clothing. Work in local contrast: on white
    # fabric a seam can be faint and on dark fabric it may be a bright line.
    v=np.sqrt(np.maximum(a,0)+.006); v=blur(v)*.32+v*.68
    pad=np.pad(v,((0,0),(0,0),(3,3)),mode='edge'); mid=pad[:,:,3:31]
    def ridge_at(w):
        l=pad[:,:,3-w:31-w]; r=pad[:,:,3+w:31+w]
        return np.maximum(abs(mid-(l+r)*.5),abs(l-r)*.50)
    lines=np.maximum(ridge_at(1),np.maximum(.90*ridge_at(2),.79*ridge_at(3)))
    baseline=np.minimum(lines[:,7:24,6:10].mean(axis=(1,2)),lines[:,7:24,18:22].mean(axis=(1,2)))
    zipstrength=np.zeros(len(a)); placket=zipstrength.copy()
    for x in range(11,17):
        for dx in (-2,0,2):
            y=np.arange(5,25); q=x+np.rint(np.linspace(-dx/2,dx/2,len(y))).astype(int)
            e=lines[:,y,q]; e=np.maximum(e,.84*np.maximum(lines[:,y,q-1],lines[:,y,q+1]))
            bands=e.reshape(len(a),4,5).mean(2); steady=.60*bands.min(1)+.40*e.mean(1)
            excess=steady-baseline*.58
            zipstrength=np.maximum(zipstrength,ramp(excess,.012,.080)*(.35+.65*ramp(steady/(.012+baseline),1.12,2.1)))
            # A discrete row of buttons can be more informative than the
            # continuous placket in a shirt. Look for local extremal dots.
            light=v[:,y,q]; along=light-.5*(v[:,y-2,q]+v[:,np.minimum(y+2,27),q])
            across=light-.5*(v[:,y,q-2]+v[:,y,q+2]); dots=np.maximum(0,np.minimum(along,across))+np.maximum(0,np.minimum(-along,-across))
            distinct=dots.reshape(len(a),4,5).max(2); buttons=np.sort(distinct,axis=1)[:,1:3].mean(1)
            placket=np.maximum(placket,ramp(buttons-baseline*.64,.021,.105)*ramp(excess,.01,.08))
    # Collar fold: a shallow or deep V with two consistent diagonal edges.
    def sided(y,x,delta):
        return v[:,y,x+delta]-v[:,y,x-delta]
    vee=np.zeros(len(a)); spread=vee.copy()
    for center in (13,14):
        for width in (3,5,7):
            for start,stop in ((1,6),(2,11),(2,16)):
                y=np.arange(start,stop+1); shift=np.rint(np.linspace(width,1,len(y))).astype(int)
                l=sided(y,center-shift,1); r=sided(y,center+shift,1)
                both=np.sqrt(abs(l.mean(1)*r.mean(1)))
                vee=np.maximum(vee,ramp(both-.9*baseline,.035,.17))
                if stop==6: spread=np.maximum(spread,ramp(both-.9*baseline,.04,.17))
    # Clothing structure below the collar. At the hem, open coat fronts often
    # make a central vent. At the head, a funnel collar sticks up; for a crew
    # neck, the two shoulder tops are instead connected by a shallow scoop.
    masker=m>.42
    roof=np.where(masker.any(1),masker.argmax(1),28)/28
    floor=np.where(masker.any(1),masker[:,::-1,:].argmax(1),28)/28
    c=roof[:,12:16].mean(1); sid=.5*(roof[:,6:9].mean(1)+roof[:,19:22].mean(1))
    inner=.5*(roof[:,9:11].mean(1)+roof[:,17:19].mean(1)); collar=ramp(sid-c,.068,.19)*(1-ramp(c,.035,.17))
    collar*=.55+.45*ramp(inner-c,.012,.095)
    vent=ramp(floor[:,12:16].max(1)-.5*(floor[:,8:11].mean(1)+floor[:,17:20].mean(1)),.034,.20)
    crew=ramp(c-inner,.022,.13)*(1-zipstrength)
    # Distribution of pattern: prints on button shirts tend to cover the side
    # panels as well as the center. A single graphic on a sweatshirt does not.
    hp=abs(v-blur(v)); rep=np.minimum(hp[:,6:20,6:11].mean(axis=(1,2)),hp[:,6:20,17:22].mean(axis=(1,2)))
    fine=ramp(rep,.015,.053)
    # Cuffs of a heavy jacket are broader than those of a shirt or jumper.
    # Ignore this cue entirely when cuffs do not separate from the torso.
    cuff=np.zeros(len(a)); valid=cuff.copy()
    for flip in (False,True):
        p=m[:,:,::-1] if flip else m
        occupied=p[:,17:26,:]>.45; first=occupied.argmax(2)
        xx=np.arange(28)[None,None,:]; gap=(~occupied)&(xx>first[:,:,None]); sep=np.where(gap.any(2),gap.argmax(2),28)
        width=(sep-first); good=(first<6)&(sep<12)&occupied[:,:,11:18].any(2)
        w=np.sum(good*width,axis=1)/np.maximum(good.sum(1),1)
        cf=unit(good.mean(1)*2); valid+=cf*.5; cuff+=cf*ramp(w*aspect,2.6,5.2)*.5
    # Asymmetric and very wide openings/dramatic sleeves are blouse-like.
    asym=abs(m-m[:,:,::-1]).mean(axis=(1,2)); drape=ramp(asym,.125,.29)
    outer=masker.any(2); xl=masker.argmax(2); xr=masker[:,:,::-1].argmax(2)
    midwidth=(28-xl-xr)*outer/28
    bell=ramp(midwidth[:,9:17].mean(1)-midwidth[:,20:25].mean(1),.09,.34)
    fitted=ramp(midwidth[:,13:18].mean(1)-midwidth[:,2:5].mean(1),.15,.38)
    s=np.zeros((len(a),10)); clasp=zipstrength
    # Woven checks and floral repeats differ from a sweater with broad,
    # horizontal knitted stripes. Test the *direction* of the texture as well
    # as its strength, excluding the outline, sleeve gap and front placket.
    dx=abs(v[:,:,2:]-v[:,:,:-2]); dy=abs(v[:,2:,:]-v[:,:-2,:]); both=m[:,5:20,6:22]>.92
    bx=dx[:,5:20,5:21]; by=dy[:,4:19,6:22]; off=np.ones(16); off[6:10]=0
    inside=both*off[None,None,:]; den=np.maximum(1,inside.sum(axis=(1,2)))
    tx=(inside*bx).sum(axis=(1,2))/den; ty=(inside*by).sum(axis=(1,2))/den
    striped=ramp(ty/(.015+tx),1.35,2.50)*ramp(ty,.035,.11)
    woven=ramp(np.minimum(tx,ty),.045,.134)*(1-striped)
    # Shape of negative space below the cuffs: the close fit of a jacket can
    # bring its sleeves into contact with its hips. A cropped pullover often
    # leaves daylight between cuffs and beneath the straight knitted waistband.
    armextent=.5*(floor[:,1:6].min(1)+floor[:,22:27].min(1)); hem=floor[:,10:18].mean(1)
    cropped=ramp(hem-armextent,.030,.18); skirt=ramp(armextent-hem,.08,.36)
    broadhem=ramp(m[:,22:26,6:22].mean(axis=(1,2)),.73,.96); blocky=collar*np.maximum(broadhem,skirt)
    s[:,2]+=.022*striped+.016*cropped*(1-clasp)
    s[:,4]+=.020*blocky+.014*skirt*clasp
    s[:,6]+=.018*woven-.014*striped
    s[:,0]-=.028*clasp+.015*collar+.014*placket
    s[:,2]+=.014*crew-.043*clasp-.018*vent-.017*vee-.013*collar-.020*placket
    s[:,4]+=.032*clasp+.022*vent+.022*collar+.013*vee+.022*cuff-.012*valid-.017*fine-.010*drape
    s[:,6]+=.020*clasp+.037*placket+.024*fine+.013*spread+.027*drape+.012*bell+.009*fitted-.013*cuff
    # Additional construction lines: welt pockets or diagonal zippers on both
    # hips suggest an outer layer; traditional folded shirt collars have two
    # downward tips. Use matched local contrasts rather than absolute grey.
    pockets=np.zeros(len(a)); tips=pockets.copy()
    for hand in (-1,1):
        view=v if hand==1 else v[:,:,::-1]; peak=np.zeros(len(a))
        for xpos in (6,8,10):
            for y0 in (12,16,19):
                yy=np.arange(y0,y0+6); xx=xpos+np.rint(np.linspace(2,-1,6)).astype(int)
                edge=view[:,yy,xx]; l=view[:,yy,xx-2]; r=view[:,yy,xx+2]
                rr=abs((edge-.5*(l+r)).mean(1)); ss=abs((l-r).mean(1))
                # Only count marks on the cloth, not the sleeve/torso gap.
                support=(m if hand==1 else m[:,:,::-1])[:,yy,xx-2].mean(1)
                pk=ramp(np.maximum(rr,ss*.64)-.9*baseline,.032,.13)*ramp(support,.87,.99)
                peak=np.maximum(peak,pk)
        pockets+=peak*.5
        peak=np.zeros(len(a))
        for deep in (4,6,8):
            yy=np.arange(1,deep+1); tipin=np.rint(np.linspace(12,9,len(yy))).astype(int); tipout=np.rint(np.linspace(7,9,len(yy))).astype(int)
            le=view[:,yy,tipout]-view[:,yy,tipout-1]; ri=view[:,yy,tipin+1]-view[:,yy,tipin]
            peak=np.maximum(peak,ramp(np.minimum(abs(le.mean(1)),abs(ri.mean(1)))-.4*baseline,.021,.105))
        tips+=peak*.5
    # Short-sleeved tee: a symmetric abrupt loss of span at roughly armpit
    # height followed by an almost constant-width, uninterrupted tube.
    transition=np.zeros(len(a)); elbow=np.zeros(len(a))
    for y in range(6,17):
        drop=midwidth[:,y-3:y].mean(1)-midwidth[:,y+1:y+4].mean(1)
        transition=np.maximum(transition,ramp(drop,.12,.34));
        if y>=13: elbow=np.maximum(elbow,ramp(drop,.12,.34))
    tube=1-ramp(midwidth[:,17:25].max(1)-midwidth[:,17:25].min(1),.08,.26)
    tshirt=transition*tube*(1-ramp(asym,.082,.23))*(1-ramp(clasp,.33,.92))*ramp(aspect,.55,.87)
    # A symmetric thin-strapped undershirt is another standard top cut.
    scoop=ramp(c-inner,.070,.26)*(1-ramp(clasp,.4,.95)); tank=scoop*tube*(1-ramp(asym,.09,.27))*ramp(aspect,.46,.64)
    s[:,0]+=.072*tshirt+.046*tank
    s[:,6]+=.026*tips*(.46+.54*clasp)+.012*elbow*(1-tube)-.024*tshirt
    s[:,4]+=.026*pockets*(.35+.65*np.maximum(collar,clasp))
    s[:,2]-=.016*tips*clasp+.012*pockets*clasp
    return s


def predict(x):
    answer=np.empty(len(x),dtype=np.int64)
    # A fixed numeric kernel shape even for the last partial block. No sample
    # participates in any other sample's normalization or vote.
    for start in range(0,len(x),64):
        part=x[start:start+64,0]; n=len(part)
        a=np.zeros((64,28,28),np.float32); a[:n]=part
        m,ink,aspect=prepare(a)
        shape=matches(m,aspect); detail=cues(m,ink,aspect); textile=cut_and_fabric(m,ink,aspect)
        # Hierarchy: buttons, folds and cuffs are meaningful *conditional on*
        # seeing a torso; they must not turn a handbag into a shirt. Likewise a
        # zipper down a boot should not affect its distinction from a jacket.
        tops=[0,2,3,4,6]; shoes=[5,7,9]
        leg=detail[:,1]; torso=shape[:,tops].max(1); footwear=shape[:,shoes].max(1)
        primary=np.stack([torso,shape[:,1]+leg,footwear,shape[:,8]],axis=1).argmax(1)
        # Preserve cut evidence while giving more leeway to shirtlike textures
        # among long sleeves than between, for example, a dress and a blazer.
        longcut=shape[:,[2,4,6]].max(1); shortcut=shape[:,[0,3]].max(1)
        relevant=.24+.76*ramp(longcut-shortcut,-.015,.048)
        weave=detail[:,tops]+textile[:,tops]
        clothes=np.array(tops)[(shape[:,tops]+weave*relevant[:,None]).argmax(1)]
        shoe=np.array(shoes)[(shape[:,shoes]+detail[:,shoes]).argmax(1)]
        result=np.select([primary==0,primary==1,primary==2],[clothes,1,shoe],default=8)
        answer[start:start+n]=result[:n]
    return answer
'''


def main(dest):
    dest.mkdir(exist_ok=True,parents=True)
    (dest/'solution.py').write_text(SOLUTION)
    counts=[1100,500,1500,1900,1600,1800,1800,1050,900,1100]
    templates=[]; aspects=[]; costs=[]
    for c,count in enumerate(counts):
        for i in range(count):
            if c==1: d,bias,tilt=trousers(i)
            elif c in (5,7,9): d,bias,tilt=footwear(c,i)
            elif c==8: d,bias,tilt=bag(i)
            else: d,bias,tilt=garment(c,i)
            pic,ratio=d.raster(tilt)
            templates.append(pic); aspects.append(ratio); costs.append(bias)
    np.savez(dest/'constants.npz',silhouettes=np.array(templates),aspect=np.array(aspects,np.float32),costs=np.array(costs,np.float32),count=np.array(counts,np.int32))

if __name__=='__main__':
    main(Path(sys.argv[1]))
