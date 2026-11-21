"""Hand-designed Fashion item network. Inputs to the builder: a destination, nothing else."""
import json
import sys
from pathlib import Path
import numpy as np

rng = np.random.default_rng(589172)
def u(a,b): return rng.uniform(a,b)
def yes(p): return rng.random()<p
def pick(xs): return xs[rng.integers(len(xs))]

# Render shapes in the garment maker's coordinates. Supersample for smooth silhouettes.
S=42
GY,GX = (np.indices((S,S))+.5)/S

def coords(width,height,foot=False):
    # Photograph framing: centred horizontally; footwear, in particular, rests on a baseline.
    size=u(.89,.995)*28
    scale=size/max(width,height)
    mx=(28-width*scale); my=(28-height*scale)
    ox=mx*.5+u(-.48,.48)
    anchor=(u(.40,.63) if yes(.85) else u(.59,.85)) if foot else u(.36,.64)
    oy=my*anchor+u(-.4,.4)
    x=(GX*28-ox)/scale; y=(GY*28-oy)/scale
    # Modest garment shear, camera and positioning variation.
    x=x+(y-height*.5)*u(-.032,.032)
    return x,y

def interp(q,xx,yy):return np.interp(q,xx,yy)
def poly(x,y,vs):
    m=np.zeros(x.shape,dtype=bool)
    for (a,b),(c,d) in zip(vs,vs[1:]+vs[:1]):
        if b!=d: m^=((y>=min(b,d))&(y<max(b,d))&(x<(a+(c-a)/(d-b)*(y-b))))
    return m

# Dimensions are in approximately cm. The 'cut' indexes a sewing pattern, not an exemplar.
def garment(k,cut):
    if k==0:
        L=u(58,80); bust=u(20,29); hip=bust*u(.85,1.09); waist=bust*u(.82,1.01)
        if cut==0: sleeve='tee'; depth=u(1.0,7); neck=u(7,12); bend=u(-1.3,2.5)
        elif cut==1: sleeve=pick(['tank','tank','strap']); depth=u(6,18);neck=u(9,15);bend=u(-1.5,2.5)
        elif cut==2: sleeve='tee';depth=u(3,12);neck=u(7,14); L=u(68,88); hip*=u(.95,1.14);bend=u(-1,3)
        else: sleeve=pick(['long','threequarter']);depth=u(4,13);neck=u(8,14);bend=u(-1,3)
    elif k==2:
        L=u(53,75);bust=u(22,31);waist=bust*u(.88,1.02);hip=bust*u(.78,1.0)
        sleeve='long';neck=u(6,12);depth=u(0,7);bend=u(-1.8,2)
        if cut==1: # crew-neck sweatshirt, compact hem and long cuffs
            L=u(51,66); hip=bust*u(.73,.94);depth=u(0,4)
        if cut==2: # loose longline or funnel neck knit
            L=u(64,83); hip=bust*u(.91,1.08);depth=u(0,3);neck=u(5,10)
        if cut==3:
            sleeve=pick(['threequarter','threequarter','tee']);neck=u(6,13);depth=u(1,9)
    elif k==3:
        L=u(88,135);bust=u(18,26);waist=bust*u(.63,.94);hip=bust*u(1.05,1.90)
        sleeve=['tank','tee','long','strap'][cut];neck=u(6,13);depth=u(2,11);bend=u(-1,4)
        if cut==1 and yes(.25):sleeve='threequarter'
        if cut==3 and yes(.37):sleeve='tank';depth=u(0,5)
        if yes(.28):hip=bust*u(.88,1.2) # sheath
        if yes(.18):L=u(126,155)
    elif k==4:
        L=u(66,84);bust=u(23,32);waist=bust*u(.87,1.01);hip=bust*u(.91,1.12)
        sleeve='long';neck=u(5.5,11);depth=u(-2,3);bend=u(-2,3.5)
        if cut==1: # cropped outerwear
            L=u(52,67);hip=bust*u(.8,.99)
        if cut==2: # overcoat, parka
            L=u(82,109);hip=bust*u(.99,1.23);waist=bust*u(.79,.98)
        if cut==3: # blazer, cardigan or open jacket
            L=u(61,84);depth=u(2,9);hip=bust*u(.9,1.10);bend=u(0,4)
    else: # shirt and blouse
        L=u(64,84);bust=u(20,27);waist=bust*u(.82,1.01);hip=bust*u(.91,1.13)
        sleeve='long';neck=u(6,11);depth=u(-1,6);bend=u(2,8)
        if cut==1: sleeve=pick(['threequarter','long']);neck=u(8,14);depth=u(3,12);bend=u(.5,7)
        if cut==2: sleeve='tee';neck=u(5,11);depth=u(0,6);bend=u(1,7)
        if cut==3:
            sleeve=pick(['threequarter','tank','tee']);depth=u(4,12);neck=u(8,14);bend=u(0,6);hip*=u(.8,1.13)
    shoulder=bust*u(.78,.99); ys=u(3,7); under=u(17,26); waistY=min(L*.72,u(35,49))
    if sleeve=='tank':shoulder=u(max(neck+1,14),max(neck+3,20));ys=u(.7,4)
    if sleeve=='strap':shoulder=neck+u(.9,3.4);ys=u(.2,2);depth=u(10,19)
    # Long sleeves hang beside the torso. Short sleeves project from the shoulder
    # and terminate in an angled opening. Cuff width denotes the whole flat cuff.
    if sleeve=='long':
        endY=u(56,70); dx=u(5,16); cuff=u(7,12)
        if k==4:cuff=u(9,14); endY=u(58,71);dx=u(1,12) # outerwear drapes heavily
        if k==6:dx=u(4,13)
        if k==2 and cut==1:endY=u(60,73)
        dy=u(-2.0,3.5)
    elif sleeve=='threequarter':endY=u(32,51); dx=u(6,15);cuff=u(8,15);dy=u(-1,5)
    else:endY=u(13,25);dx=u(7,16);cuff=u(7,13);dy=u(3,9)
    outer=shoulder+dx+cuff*.45; inner=outer-cuff
    hasarm=sleeve not in ('tank','strap')
    half=max(bust,hip,outer if hasarm else 0,shoulder)
    collarH=(u(2.5,7.5) if k==4 else u(1.2,4.8) if k==6 and cut in [0,2] else 0)
    H=collarH+max(L,endY+max(dy,0) if hasarm else 0)
    x,y=coords(half*2,H); x-=half;y-=collarH
    a=np.abs(x); side=(x>0)
    # A tiny difference in how the two sleeves hang, never a stored outline.
    yarm=y+side*u(-2.4,2.4)
    width=interp(y,[0,ys,under,waistY,L],[max(neck,shoulder-4),shoulder,bust,waist,hip]);
    neckcurve=(np.maximum(1-(a/neck)**(1 if yes(.3) else 2),0))**(.68)
    shoulderline=ys*np.clip((a-neck)/max(shoulder-neck,.1),0,1)**1.35
    top=shoulderline+max(0,depth)*neckcurve
    if sleeve=='strap':top=np.maximum(top,interp(a,[neck,shoulder,bust],[0,1,under]))
    elif sleeve=='tank':top=np.maximum(top,interp(a,[shoulder,max(shoulder+.1,bust)],[ys,under]))
    hem=L-bend*(np.minimum(a/hip,1.3)**(1.4 if k==6 else 2.4));
    m=(a<=width)&(y>=top)&(y<=hem)
    if collarH:
        m |= (a < interp(y,[-collarH,0,ys],[neck*u(.60,.91),neck,neck+2]))&(y>=-collarH+u(0,2)*(a/neck)**2)&(y<ys)
    if hasarm:
        elbow=ys+(endY-ys)*u(.37,.60)
        q=u(.42,.72) if sleeve!='tee' else u(.43,.62)
        arm=poly(a,yarm,[(neck,0),(shoulder,ys*.85),(shoulder+u(1.2,3.5),ys+u(2.4,5.5)),(shoulder+(outer-shoulder)*q,elbow),(outer,endY),(inner,endY+dy),(bust-u(.3,2.8),under),(shoulder*.75,under*.59)])
        m|=arm
    # Some front-opening garments hang apart; the viewer sees the background
    # through the split at the hem and/or the V at the front.
    if k==4 and ((cut==3 and yes(.70)) or yes(.12)):
        gap=u(.5,6);start=L*u(.35,.92);m &= (a>gap*np.maximum(0,(y-start)/(L-start)))|(y<start)
    return m

def trousers(cut):
    H=u(87,114) if cut!=3 else u(50,94)
    hip=u(21,29);waist=hip*u(.81,1.03);rise=u(.22,.38)*H
    ankle=hip*(u(.20,.40) if cut==0 else u(.32,.53) if cut in [1,3] else u(.46,.64))
    # Half-width of individual pant leg at the ankle. Negative spacing lets
    # fabric touch when trousers are laid flat.
    gap=u(-.8,5) if cut!=2 else u(-1.4,2); inset=u(-2.6,2.6)
    span=max(hip,2*ankle+max(gap,0)+abs(inset))
    x,y=coords(2*span,H);x-=span; y=y+(x>0)*u(-1.5,1.5);a=abs(x)
    t=np.clip((y-rise)/(H-rise),0,1)
    outside=interp(y,[0,rise*.48,rise,rise+.58*(H-rise),H],[waist,hip,hip*u(.90,1.01),(.45*hip+.55*(gap+2*ankle))*u(.90,1.04),gap+2*ankle]);outside+=t*inset*np.sign(x)
    inside=gap*t**.68+inset*t*np.sign(x)
    return (y>=u(0,.8)*(1-(a/hip)**2))&(y<=H)&(a<outside)&((a>inside)|(y<rise))

def shoe(k,cut):
    # All lengths expressed in sole lengths, heel on the right. v is height above ground.
    heelx=u(.78,.91); spring=u(.015,.078); toe=u(.04,.11); sole=u(.034,.09)
    if k==7:
        H=[u(.30,.47),u(.33,.54),u(.42,.66),u(.22,.39)][cut]
        heel=u(.68,.97)*H; tongue=u(.47,.69); bridge=H*u(.83,1.04)
    elif k==9:
        H=[u(.49,.82),u(.65,1.12),u(.56,.98),u(.40,.68)][cut];heel=u(.87,1.0)*H
        tongue=.6;bridge=H
    else:
        H=[u(.16,.38),u(.30,.60),u(.46,.92),u(.24,.48)][cut];heel=H;tongue=.6;bridge=H
    lift=0
    if (k==9 and cut==2) or (k==5 and cut==2):lift=H*u(.21,.60)
    elif k==9:lift=u(.015,.14)
    elif k==5 and cut==1:lift=u(.005,.11)
    else:lift=u(.0,.025)
    x,y=coords(1,H,foot=True);v=H-y
    if yes(.035):x=1-x
    # Slight toe rocker and heel arch; elevated shoes exchange arch support for a heel post.
    base=interp(x,[0,.05,.18,.37,.60,heelx,1],[spring,spring*.44,0,0,lift*.38,lift,lift*.94]);
    inset=interp(x,[0,.30,.51,heelx,1],[0,0,0,max(0,lift-sole*.18),max(0,lift-sole*.18)])
    heelpost=(x>heelx+u(-.025,.05))&(x<.995-u(0,.07)*np.clip(1-v/max(lift,.001),0,1))&(v>=0)&(v<base+sole)
    # Rounded toe and back edges, not an exact polygonal bounding box.
    tip=np.clip(x/u(.035,.090),0,1)**.58; back=np.clip((1-x)/u(.018,.07),0,1)**.48
    if k==7:
        collar=u(.75,.86); backx=u(.91,.97)
        upper=interp(x,[0,.07,.23,.38,tongue,tongue+.042,collar,backx,1],[sole*.75,spring+toe+sole*.52,toe+sole*.90,H*u(.39,.63),bridge,bridge*u(.68,.95),H*u(.43,.73),heel,heel*.85]);
        m=(v<base+(upper-base)*tip*back)&(v>=base)
    elif k==9:
        ankle=u(.49,.66); shaft=u(.39,.63); backx=u(.90,.99)
        upper=interp(x,[0,.08,.24,ankle-.09,ankle,ankle+.08,backx,1],[spring+sole,spring+toe+sole,toe+sole*1.17,H*u(.30,.45),H*u(.43,.57),H*.63,H*.67,H*.52]);
        m=(v<base+(upper-base)*tip*back)&(v>=base)
        front=interp(v/H,[0,.38,.57,1.],[ankle+.04,ankle-.045,ankle,shaft]);
        rear=interp(v/H,[0,.34,.56,1.],[.96,.987,.959,backx]);
        lip=H*(1-u(-.13,.13)*(x-(shaft+backx)*.5)/max(backx-shaft,.2));
        m|=(x>front)&(x<rear)&(v<lip)&(v>=base)
    else:
        z=v-base; m=(z>=0)&(z<sole*tip*back)
        # Open space under arched straps. Footbed has a continuous thin outline.
        def strap(xa,za,xb,zb,thick):
            t=np.clip(((x-xa)*(xb-xa)+(z-za)*(zb-za))/((xb-xa)**2+(zb-za)**2),0,1)
            return ((x-xa-t*(xb-xa))**2+(z-za-t*(zb-za))**2)<thick**2
        fx=u(.15,.34); mid=u(.39,.62);ank=u(.64,.85)
        top=H-lift; ztop=top*u(.74,.98);wide=u(.022,.062)
        if cut==3: # slide: wide forefoot strap, no heel counter
            m|=poly(x,z,[(.05,sole*.63),(fx,top*.81),(mid,top),(mid+u(.02,.15),sole*.42)])
        else:
            m|=strap(u(.05,.20),sole*.75,fx+u(.01,.16),top*u(.36,.70),wide)
            m|=strap(fx+u(.01,.15),top*.46,u(.32,.47),sole*.73,wide)
            if cut==0: # thong sandal
                m|=strap(fx,top*u(.26,.57),mid,ztop,wide*.72);m|=strap(mid,ztop,u(.75,.92),sole,wide*.67)
            else:
                m|=strap(mid-u(.05,.16),sole*.7,ank,ztop,wide)
                m|=strap(ank-u(.07,.18),ztop, u(.88,.98),ztop*u(.67,1.02),wide*.8)
                if yes(.77):m|=strap(u(.86,.98),sole*.8,u(.88,.97),ztop*.82,wide*u(.78,1.55))
                if yes(.37):m|=strap(mid,sole*.6,ank-.10,ztop,wide*.74)
    if k==5 and cut in (1,2):
        # Many dress sandals have a disconnected-looking forefoot and heel:
        # a toe cap, steep arch, and sometimes a filled-in heel counter.
        if yes(.67):
            tx=u(.20,.40);vv=interp(x,[0,.08,tx,tx+.075],[sole*.8,toe+sole,toe+sole*1.5,sole*.8]);m|=(x<tx+.075)&(v<base+vv*tip)&(v>=base)
        if yes(.60):
            m |= poly(x,v,[(u(.68,.84),lift+sole*.8),(.83,H*u(.67,.96)),(.97,H),(.995,lift+sole*.8),(.96,lift)])
        if yes(.22):m|=(v>0)&(v<base+sole*.4)&(x>.26)&(x<.98)
    return ((m|heelpost)&(x>=0)&(x<=1))

def bag(cut):
    W=1.; bh=u(.46,.86) if cut in (0,1) else u(.29,.88) if cut==2 else u(.73,1.27)
    handle=(u(.19,.51) if cut==0 else u(.39,.92) if cut==1 else u(0,.08) if cut==2 else u(.04,.28));H=bh+handle
    x,y=coords(W,H,foot=True);x-=.5; lean=u(-.14,.14);xx=x+lean*(y/H-.5)
    a=abs(xx); topw=u(.33,.50);botw=u(.33,.49); waist=max(topw,botw,u(.43,.51));
    r=(y-handle)/bh
    w=interp(r,[0,.13,.74,.93,1],[topw*u(.81,1),topw,waist,botw,botw*u(.80,.99)])
    lip=u(-.05,.12);m=(a<=w)&(r>=lip*(1-(a/max(topw,.01))**2))&(r<=1-.033*(a/botw)**4)
    sx=u(.20,.38);thick=u(.012,.038);off=u(-.05,.05)
    ellipse=np.sqrt(((xx-off)/sx)**2+((y-(handle+bh*.07))/max(.01,handle+bh*.07))**2)
    m|=(abs(ellipse-1)<thick/sx)&(y<handle+bh*.18)
    if cut==3 and yes(.55):m|=(a<interp(y/H,[0,.12,.4,1],[.09,.32,.48,.38]))&(y>=0)&(y<H*.9)
    return m

# Small network construction helpers
L=[]; W={}
def layer(name,op,src,**opts):
    L.append(dict(name=name,op=op,inputs=src if isinstance(src,list) else [src],**opts));return name
def affine(name,src,w,b=0,conv=False,pad=0):
    w=np.asarray(w,np.float32); W[name+'.W']=w;W[name+'.b']=np.broadcast_to(np.asarray(b,np.float32),(len(w),)).copy();
    return layer(name,'conv' if conv else 'dense',src,out=len(w),**({'kernel':list(w.shape[-2:]),'stride':1,'padding':pad} if conv else {}))
def shape(n,src,*s):return layer(n,'reshape',src,shape=list(s))
def pool(n,src,size,kind='max',stride=None):return layer(n,kind+'pool',src,size=list(size),stride=list(stride or size))
def rel(n,src):return layer(n,'relu',src)
def add(n,*src):return layer(n,'add',list(src))

# Foreground: soft clipping at two illumination levels, with slight blur before clipping.
sm=np.array([[1,2,1],[2,18,2],[1,2,1]],float)/30
k=np.array([sm*13,sm*13,sm*3.8,sm*3.8,sm]);b=[-.32,-1.32,-.16,-1.16,0]
s=rel('ramp',affine('silhouette_ramps','image',k[:,None],b,True,1))
s=affine('soft_cloth',s,np.array([.63,-.63,.29,-.29,.08]).reshape(1,5,1,1),conv=True)
coarse=pool('area_samples',s,(2,2),'avg'); standardized=layer('shape_contrast','norm',coarse)
# A parallel shape representation retains narrow gaps otherwise lost when two
# dark boundaries are blended before thresholding. Gradients, occupancy and
# a little normalized continuous tone all come from the query itself.
identity=np.zeros((3,3));identity[1,1]=1
r=rel('fine_mask_ramp',affine('fine_mask_gate','image',np.stack([identity*28,identity*28])[:,None],[-.27,-1.27],True,1))
hard=affine('fine_foreground',r,np.array([1,-1]).reshape(1,2,1,1),conv=True)
hardcell=pool('fine_cells',hard,(2,2),'avg');hardstd=layer('fine_shape','norm',hardcell)
rawcell=pool('light_cells','image',(2,2),'avg');rawstd=layer('light_shape','norm',rawcell)
vv=[]
for v,wt in [(standardized,.48),(hardstd,.36),(rawstd,.16)]:vv.append(affine(v+'_fraction',v,np.array([[[[wt]]]]),conv=True))
standardized=add('combined_shape',*vv)

# Each row is a *single* independently drawn item. No cross-drawing statistics.
# Separate groups permit a prior for a rare cut, rather than penalizing the whole class.
G=8;P=128
bank=np.empty((10,4,G,P,196),np.float32)
profbank=np.empty((10,4,G,P,28),np.float32)
for k in range(10):
    for c in range(4):
        for j in range(G):
            for i in range(P):
                if k==1:m=trousers(c)
                elif k==8:m=bag(c)
                elif k in (5,7,9):m=shoe(k,c)
                else:m=garment(k,c)
                f=m.reshape(14,3,14,3).mean((1,3));profile=np.r_[f.mean(1),f.mean(0)];profile-=profile.mean();
                profbank[k,c,j,i]=profile/(np.sqrt(28)*np.sqrt(np.sum(profile**2)+1.e-12))
                f-=f.mean(); f/=14*np.sqrt(np.sum(f*f)+1.e-12)
                bank[k,c,j,i]=f.ravel()
# An independent, condensed view of each individual template: row widths
# and column heights. The same per-object normalization is used at test time.
YY,XX=np.indices((14,14));ww=np.array([unitmask.astype(float)/14 for grid in (YY,XX) for unitmask in [grid==j for j in range(14)]])
pq=affine('width_and_height',coarse,ww.reshape(28,196));pq=layer('projection_contrast','norm',shape('scan_directions',pq,1,2,14))
pr=affine('individual_profiles',pq,.19*profbank.reshape(-1,28));r=affine('individual_cut_patterns',standardized,.81*bank.reshape(-1,196));r=shape('patterns_by_cut',add('template_match',r,pr),40,G,P)
r=pool('best_in_group',r,(1,P));rm=pool('typical_match',r,(G,1),'avg');rx=pool('best_match',r,(G,1));
rm=affine('typical_vote',rm,np.eye(40)*.37);rx=affine('closest_vote',rx,np.eye(40)*.63)
parts=add('pattern_evidence',rm,rx)

# Construction detail detectors on intensity (not masks). Contrast normalization makes
# a dark button-down comparable to a light one. Filters here are analytic, not drawn images.
norm=layer('exposure','norm','image'); Y,X=np.indices((28,28),dtype=float);X+=.5;Y+=.5

def band(pos,wd): return np.exp(-.5*pos**2/wd**2)
def unit(a):return a/(a.sum()+1.e-12)
def ridge(cx,ymin,ymax,width,offset,side=0,slope=0):
    t=X-cx-slope*(Y-(ymin+ymax)*.5);win=(Y>=ymin)&(Y<=ymax)
    if side:
        return unit(win*band(t-offset*.5,width))-unit(win*band(t+offset*.5,width))
    return unit(win*band(t,width))-.5*(unit(win*band(t-offset,width))+unit(win*band(t+offset,width)))

def max_features(name,features,src):
    # Array of lists of linear forms; maxima of signed correlations for landmarks.
    width=max(len(q) for q in features);ff=[]
    for q in features:ff.extend(q+[q[0]]*(width-len(q)))
    r=affine(name+'_forms',src,np.asarray(ff).reshape(len(ff),-1));r=shape(name+'_axes',r,len(features),1,width)
    return pool(name,r,(1,width))

det=[];names=[]
def feature(n,fs):names.append(n);det.append(fs)
for n,xcenters in [('placket',[12.9,14,15.1]),('fold_l',[9.4,10.5]),('fold_r',[17.5,18.6])]:
    fs=[]
    for xx in xcenters:
        for sl in (-.065,.065):
            for sd,ww,off in [(0,.53,1.6),(0,.8,2.6),(1,.70,2.4)]:
                a=ridge(xx,6,22,ww,off,sd,sl);fs.extend([a,-a]);
    feature(n,fs)
# Neck points: diagonal boundaries of collar and lapels, both left and right.
for side in [-1,1]:
    fs=[]
    for cx in [13.4,14.6]:
        for h in [4.5,7.5,10.5]:
            for wid in [2.4,4.]:
                a=ridge(cx+side*wid*.5,1,1+h,.63,1.65,1,-side*wid/h)
                fs.extend([a,-a]);
    feature('lapel'+str(side),fs)
# Fabric repeats, with a quadrature-like maximum over phase.
for direction in ['weft','warp']:
    fs=[]; zz=Y if direction=='weft' else X
    window=band(X-14,3.4)*band(Y-14,5.1)
    for period in [2.5,3,3.6,4.4,5.4,6.5,8]:
        for phase in [0,.25,.5,.75]:
            a=np.cos(2*np.pi*(zz/period+phase));a-=np.sum(unit(window)*a);fs.append(2*unit(window)*a)
    feature(direction,fs)
details=max_features('sewing_details',det,norm)

# Texture extent: blouse patterns often cover the whole garment, as opposed to
# a graphic printed in the middle of a T-shirt. Use local high-pass energy.
k=np.array([[-1,-2,-1],[-2,12,-2],[-1,-2,-1]],float)/12
tex=rel('weave_halfwaves',affine('weave_filter',norm,np.stack([k,-k])[:,None],conv=True,pad=1));
win=unit(((X>9)&(X<19)&(Y>9)&(Y<21)).astype(float));w=np.stack([win,win])[None];texture=affine('weave_energy',tex,w.reshape(1,-1))

# Estimate open space inside a side-view shoe. For each coarse cell, test for
# fabric higher and lower in its column. Open straps and heel arches leave gaps.
Hc=14
def cumulative(above):
    a=np.zeros((196,196));
    for yy in range(Hc):
        for xx in range(Hc):
            cols=range(yy) if above else range(yy+1,Hc)
            for y2 in cols:a[yy*14+xx,y2*14+xx]=1
    return a

def clip01(n,src):
    # 1 - relu(1 - relu(src)); here used only on vectors.
    src=rel(n+'_positive',src);d=W[n+'_input.W'].shape[0];
    return affine(n,rel(n+'_extra',affine(n+'_complement',src,-np.eye(d),1)), -np.eye(d),1)
cs=[]
for n,v in [('sky',True),('floor',False)]:
    q=affine(n+'_input',coarse,cumulative(v)*1.55,-.15);cs.append(clip01(n,q))
holes=rel('openings',add('enclosed_dark',*cs,affine('not_material',coarse,-np.eye(196)*1.35,-1.05)))
yc,xc=(np.indices((14,14))+.5)/14
w=unit(((xc>.06)&(xc<.95)&(yc>.18)).astype(float));hole=affine('strap_void_area',holes,w.reshape(1,-1))

# Additional predictors: the shoulder region and knitted waistband. Contrast
# between the hem in the middle and the sleeve ends at either side.
fs=[];hmnames=[]
def region(x0,x1,y0,y1):return unit(((X>=x0)&(X<=x1)&(Y>=y0)&(Y<=y1)).astype(float))
# Combine signals as scalar regional averages. Measurements are of the *query* only.
for n,w in [('high_neck',region(11.5,16.5,1,3.5)),('neck_void',region(11.5,16.5,2,6)),('hem',region(10,18,23,26.5)),('tail',region(11,17,25,27)),('cuff',region(2,6,23,26)+region(22,26,23,26))]:hmnames.append(n);fs.append(w if n!='cuff' else w*.5)
hem=affine('garment_landmarks',s,np.stack(fs).reshape(len(fs),-1))

# Rule expressions: linear/ReLU piecewise decisions. Units of the final additive
# evidence are normalized template-correlation units. Coefficients express priors
# about garment construction, not a fit to examples.
# Also measure opposing contours directly. All dimensions below are fractions
# of the image, not numbers measured from examples.
other=[]; on=[]
def land(n,m):on.append(n);other.append(m)
def rect(a,b,c,d):return unit(((X/28>=a)&(X/28<=b)&(Y/28>=c)&(Y/28<=d)).astype(float))
def wings(inner,outer,top,bot):return .5*(rect(.5-outer,.5-inner,top,bot)+rect(.5+inner,.5+outer,top,bot))
land('cap',wings(.32,.46,.14,.32));land('waist_edge',wings(.32,.46,.44,.67));land('lowedge',wings(.33,.47,.66,.89));land('hemwidth',wings(.17,.28,.71,.87));land('waistwidth',wings(.17,.28,.37,.53))
land('clavicle',wings(.09,.19,.05,.17));land('crown',rect(.35,.65,.055,.17));land('flatwaist',rect(.34,.66,.02,.14));land('hipline',rect(.28,.72,.16,.29));
land('toe',rect(.10,.30,.40,.66));land('instep',rect(.66,.86,.40,.66));land('toetop',rect(.08,.31,.23,.49));land('bootleg',rect(.65,.88,.23,.49));
land('belowtoe',rect(.07,.29,.65,.84));land('belowheel',rect(.72,.94,.65,.84));land('lowest',rect(.10,.90,.88,.98));
land('upperbody',rect(.30,.70,.32,.55));land('belly',rect(.31,.69,.52,.72));
land('toesky',rect(.12,.38,.28,.49));land('tallheel',rect(.70,.92,.16,.38));land('shoeupper',rect(.33,.72,.43,.59));
land('left_pre',wings(.30,.45,.34,.50));land('left_post',wings(.30,.45,.75,.92))
lands=affine('anatomical_regions',s,np.stack(other).reshape(len(on),-1))
# A matched lower-half cleft surrounded by two legs. A single narrow placket
# by itself is not enough: it must have dark space with trouser legs on both sides.
leg=[]
for mid in [12.4,14,15.6]:
    for lean in [-.09,0,.09]:
        t=X-mid-lean*(Y-20); h=(Y>14)&(Y<25)
        for d in [2.5,4.]:
            leg.append(.5*(unit(h*band(t-d,1.0))+unit(h*band(t+d,1.0)))-unit(h*band(t,.72)))
cleft=max_features('between_legs',[leg],s)
# Look for air horizontally enclosed by the sides of an object; unlike full
# handles and sandal straps, textile details usually retain a small but real intensity.
kdet=np.array([sm*36,sm*36]);kk=rel('ink_ramp',affine('ink_threshold','image',kdet[:,None],[-.16,-1.16],True,1));
ink=affine('nonblack',kk,np.array([1,-1]).reshape(1,2,1,1),conv=True)
# Per-pixel horizontal/vertical span closure via maxima above/below (or left/right).
# These comparisons happen on the query, not in the generator.
Y2,X2=np.indices((28,28));
def directional_max(src,n,axis,rev):
    # Binary lifting max over the whole prefix, using max(a,b)=a+relu(b-a).
    # Use a dense route on downsampled cells, to avoid large image convolutions.
    t=src;N=14;P=N*N
    for step in (1,2,4,8):
        w=-np.eye(P);dy,dx=(step,0) if axis==0 else (0,step)
        if rev:dx=-dx;dy=-dy
        for y0 in range(N):
            for x0 in range(N):
                y1=y0-dy;x1=x0-dx
                if 0<=y1<N and 0<=x1<N:w[y0*N+x0,y1*N+x1]+=1
        q=affine(n+str(step)+'_diff',t,w);t=add(n+str(step),t,rel(n+str(step)+'_larger',q))
    return t
bi=shape('ink_cells',pool('ink_mean',ink,(2,2),'avg'),196)
ab=directional_max(bi,'above',0,False);bl=directional_max(bi,'below',0,True)
le=directional_max(bi,'left',1,False);ri=directional_max(bi,'right',1,True)
bkg=affine('empty_cell',bi,-np.eye(196)*1.32,-1.)
vvoid=rel('vertical_void',add('v_span',ab,bl,bkg));hvoid=rel('horizontal_void',add('h_span',le,ri,bkg));
vv=affine('voids_across_sole',vvoid,unit(((xc>.06)&(xc<.94)&(yc>.25)&(yc<.82)).astype(float)).reshape(1,-1));
hv=affine('voids_across_vamp',hvoid,unit(((xc>.07)&(xc<.93)&(yc>.29)&(yc<.75)).astype(float)).reshape(1,-1));
# Maximum over the whole front, measured against the rear at the same height.
# Comparing row maxima rather than means makes this insensitive to the width
# of the ankle shaft. A bag ordinarily has its high point over its middle,
# footwear further back. Use the full vertical span, not a guessed shoe height.
rf=[]
for regionname,low,high in [('toe',0,.33),('rear',.59,1),('center',.36,.64)]:
    regionarray=np.zeros((14,196))
    for yy in range(14):regionarray[yy]=unit(((xc>=low)&(xc<high)&(np.indices((14,14))[0]==yy)).astype(float)).ravel()
    f=affine(regionname+'_by_height',coarse,regionarray*2,-.17)
    # Soft presence over rows of forefoot and back of item
    f=affine(regionname+'_height',rel(regionname+'_material',f),np.ones((1,14))/14);rf.append(f)
# A gap between thin straps remains dark in the actual photons even if faint
# antialiasing is enough to have soft mask support. Compute the lower of the two
# side levels using the identity min(a,b) = a - relu(a-b).
ph=affine('bright_upper_ramps',rawcell,np.stack([np.eye(196)*4,np.eye(196)*4],1).reshape(392,196),np.tile([0,-1],196));
q=np.eye(196).repeat(2,axis=1);q[:,1::2]*=-1; ph=affine('bright_surface',rel('upper_threshold',ph),q)
phab=directional_max(ph,'lit_above',0,False);phbl=directional_max(ph,'lit_below',0,True)
phle=directional_max(ph,'lit_left',1,False);phri=directional_max(ph,'lit_right',1,True)
def midvoid(src1,src2,n):
    diff=add(n+'_minus',src1,affine(n+'_neg',src2,-np.eye(196)));v=add(n+'_min',src1,affine(n+'_clip',rel(n+'_p',diff),-np.eye(196)))
    return rel(n+'_opening', add(n+'_dark',v,affine(n+'_opacity',ph,-np.eye(196)*1.15,-.05)))
vm=midvoid(phab,phbl,'strap');hm=midvoid(phle,phri,'crossing')
more=[('photogap_v',vm,(xc>.06)&(xc<.94)&(yc>.15)&(yc<.88)),('photogap_h',hm,(xc>.07)&(xc<.93)&(yc>.18)&(yc<.85)),('toehole',vm,(xc>.04)&(xc<.49)&(yc>.25)&(yc<.87)),('heelhole',vm,(xc>.53)&(xc<.93)&(yc>.22)&(yc<.86))]
photo=[]
for n,src,win in more:photo.append((affine(n,src,unit(win.astype(float)).reshape(1,-1)),[n]))
extras=photo+[(src,[n]) for src,n in zip(rf,['toe_height','rear_height','mid_height'])]
# Foreground projections separate a hung garment (tall with sculpted shoulders)
# from the box and handle of a bag. Mean *differences between rows* distinguish
# a horizontal waistband from a neckline. These are independent of position.
def columnrow(axis,n):
    # A clipped line integral of the query, equivalent to a soft logical OR.
    a=np.zeros((14,196));ar=np.arange(196).reshape(14,14)
    for i in range(14):a[i,(ar[i,:] if axis==0 else ar[:,i])]=1
    m=affine(n+'_input',coarse,a*1.65,-.18);v=clip01(n,m)
    return affine(n+'_extent',v,np.ones((1,14))/14)
for ax,n in [(0,'height'),(1,'width')]:extras.append((columnrow(ax,n),[n]))
f2=[];fn2=[]
def second(n,m):fn2.append(n);f2.append(m)
second('shoulder_w',rect(.0,1.,.13,.28));second('lower_w',rect(.0,1.,.67,.84));second('belt_w',rect(.0,1.,.04,.16));second('middle_w',rect(.0,1.,.40,.58))
second('armpit',wings(.13,.32,.23,.40));second('outseam',wings(.20,.32,.48,.76));second('sleeve_center',wings(.24,.38,.56,.81));second('base_corners',wings(.39,.50,.60,.82));
second('midchest',rect(.39,.61,.28,.49));second('base_sweep',rect(.17,.83,.72,.89));second('near_bottom',rect(.18,.82,.82,.97));
extras.append((affine('panel_regions',s,np.stack(f2).reshape(len(f2),-1)),fn2))
# Look for the paired gaps down either side of the torso. Unlike pants, a shirt
# or pullover has a central expanse of fabric, gaps on either side, then cuffs.
slitforms=[]
for side in [-1,1]:
    q=[]
    for distance in [4.8,6.4,8.0,9.4]:
        for slope in [-.13,0,.17]:
            xx=14+side*(distance+slope*(Y-19));ww=(Y>=14)&(Y<=23)
            q.append(.5*unit(ww*band(X-xx-2.1,.8))+.5*unit(ww*band(X-xx+2.1,.8))-unit(ww*band(X-xx,.64)))
    slitforms.append(q)
extras.append((max_features('underarms',slitforms,hard),['sleeveL','sleeveR']))
# A garment opening stays at roughly the same horizontal position through multiple
# central panels. Compare a centred family of narrow oriented ridge filters to
# incidental folds to its left and right. A small vertical zipper or button placket
# can be low contrast even on a very light garment.
profile=[]; profnames=[]
for zone,shift in [('front',0),('leftside',-4.2),('rightside',4.2)]:
    a=[]
    for delta in [-1.2,0,1.2]:
        for lean in [-.06,.06]:
            xx=14+shift+delta+lean*(Y-14)
            for rad in [.55,1.15]:
                w0=(Y>5)&(Y<23);cen=unit(w0*band(X-xx,rad));lft=unit(w0*band(X-xx-1.9*rad-.55,rad));rgt=unit(w0*band(X-xx+1.9*rad+.55,rad))
                a.extend([cen-(lft+rgt)*.5,(lft+rgt)*.5-cen,lft-rgt,rgt-lft]);
    profile.append(a);profnames.append(zone)
extras.append((max_features('straight_stitch',profile,norm),profnames))
# In the lower body of a sandal there can be an arch-shaped void even without
# anything below it (a high heel). Count the concavities above and below the
# horizontally spanned portion, retaining the zero background via hard masking.
# Sum vertically enclosed gaps relative to the *occupied* area, not image area:
# comparison hole_fraction > constant * cloth_fraction is linear.
extras.append((affine('cloth_area',coarse,np.ones((1,196))/196),['area']))
extras.append((affine('cen_air',hvoid,unit(((abs(xc-.5)<.10)&(yc>.53)&(yc<.89)).astype(float)).reshape(1,-1)),['leg_air']))
extras.append((affine('heel_arch',hvoid,unit(((xc>.45)&(xc<.83)&(yc>.51)&(yc<.85)).astype(float)).reshape(1,-1)),['arch_space']))
# Repeated directional changes within the region of the shoe: sandals have many
# transitions right down to black. A closed boot has one continuous upper.
YI,XI=np.indices((28,28));near=np.zeros((4,1,3,3));near[:,0,1,1]=1
for i,(dy,dx) in enumerate([(0,1),(1,0),(1,1),(1,-1)]):near[i,0,1+dy,1+dx]=-1
j=rel('starts_of_ink',affine('scan_ink',hard,near,conv=True,pad=1));
ss=[];sn=[]
for n,ax,window in [('breaks_x',0,rect(.05,.95,.17,.83)),('breaks_y',1,rect(.06,.94,.15,.85)),('arch_edges',1,rect(.15,.87,.25,.73))]:
    a=np.zeros((4,28,28));a[ax]=window;ss.append(a);sn.append(n)
extras.append((affine('strap_edges',j,np.stack(ss).reshape(len(ss),-1)),sn))
# Tone-independent visibility: a narrow stitched opening can be a slight
# brightness difference on a bright jacket, or a large one on a dark jacket.
# Contrast is normalized *within the central fabric*, excluding the black field.
for place,cx in [('axial',14),('offaxis',9.5),('outeraxis',18.5)]:
    xx,yy=np.meshgrid(cx+np.arange(8)-3.5,6.0+np.arange(16));sel=np.zeros((128,784));
    for j,(x0,y0) in enumerate(zip(xx.ravel(),yy.ravel())):
        x0-=.5;ix=int(x0);iy=int(y0);t=x0-ix;sel[j,iy*28+ix]=1-t;sel[j,iy*28+ix+1]=t
    strip=layer(place+'_contrast','norm',shape(place+'_strip',affine(place+'_crop','image',sel),1,16,8));yy,xx=np.indices((16,8));
    lineforms=[]
    for cx0 in [2.3,3.45,4.6]:
        for tilt in [-.065,.065]:
            t=xx-cx0-tilt*(yy-7.5);window=(.7+.3*np.sin(np.pi*(yy+.5)/16));
            sm0=unit(band(t,.61)*window);sm1=unit(band(t-1.65,.63)*window);sm2=unit(band(t+1.65,.63)*window);
            for z in [sm0-.5*(sm1+sm2),sm1-sm2]:lineforms.extend([z,-z])
    extras.append((max_features(place+'_persistence',[lineforms],strip),[place]))
# Vertical reach of the sole heel relative to the forefoot. Project the query
# on the horizontal axis; a boot shaft and a sneaker collar have different
# HEIGHTS AT THE REAR even if their uppers share the same toe-to-instep slope.
# Air as a fraction of that span also identifies sandals with open sides.
for n,left,right in [('fore',.05,.35),('back',.67,.94)]:
    sel=np.zeros((14,196));
    for yy in range(14):sel[yy]=unit(((xc>=left)&(xc<=right)&(np.indices((14,14))[0]==yy)).astype(float)).ravel()
    inc=affine(n+'_input',coarse,3.6*sel,-.18);he=clip01(n,inc);extras.append((affine(n+'_span',he,np.ones((1,14))/14),[n+'_span']))
rawnames=names+['texture','holes','vertical_void','horizontal_void','leg_cleft']+hmnames+on+[n for _,nn in extras for n in nn]
D=len(rawnames); blocks=[(details,names),(texture,['texture']),(hole,['holes']),(vv,['vertical_void']),(hv,['horizontal_void']),(cleft,['leg_cleft']),(hem,hmnames),(lands,on)]+extras
embedded=[]
for src,nn in blocks:
    w=np.zeros((D,len(nn)))
    for j,n in enumerate(nn):w[rawnames.index(n),j]=1
    embedded.append(affine(src+'_routing',src,w))
observ=add('observations',*embedded)
# A ramp saturates: this prevents occasional extreme contrasts arbitrarily
# dominating the silhouette score.
conditions=[]; coef=[];bias=[]
def ramp(n,expr,lo,hi):
    a=np.zeros(D)
    for t,v in expr.items():a[rawnames.index(t)]+=v/(hi-lo)
    coef.extend([a,a]);bias.extend([-lo/(hi-lo),-hi/(hi-lo)]);conditions.append(n)
ramp('front_opening',dict(placket=1,fold_l=-.18,fold_r=-.18),.12,.80)
ramp('strong_opening',dict(placket=1,fold_l=-.23,fold_r=-.23),.52,1.40)
ramp('collar',{'lapel-1':.5,'lapel1':.5},.45,1.27)
ramp('print',dict(texture=1),.15,.48)
ramp('checks',dict(weft=.5,warp=.5),.22,.69)
ramp('clearstraps',dict(holes=1),.012,.102)
ramp('veryopen',dict(holes=1),.085,.22)
ramp('knitband',dict(cuff=1,hem=-1),-.12,.48)
ramp('highcollar',dict(high_neck=1),.57,.95)
ramp('scoop',dict(neck_void=-1),-.78,-.22)
ramp('longtail',dict(tail=1,cuff=-1),.22,.83)
ramp('legsplit',dict(leg_cleft=1),.10,.54)
ramp('no_split',dict(leg_cleft=-1),-.26,-.06)
ramp('belt',dict(flatwaist=1),.58,.94)
ramp('not_belt',dict(flatwaist=-1),-.7,-.32)
ramp('skirt',dict(hemwidth=1,waistwidth=-1),.08,.59)
ramp('thinwaist',dict(waistwidth=-1),-.46,-.10)
ramp('shortsleeve',dict(cap=1,waist_edge=-1),.12,.68)
ramp('nocrown',dict(crown=-1),-.70,-.18)
ramp('toe_slope',dict(instep=1,toe=-1),.13,.67)
ramp('shaft_slope',dict(bootleg=1,toetop=-1),.13,.79)
ramp('toe_box',dict(toe=1,belowtoe=-.40),.13,.68)
ramp('holes_v',dict(vertical_void=1),.018,.14)
ramp('holes_h',dict(horizontal_void=1),.023,.18)
ramp('torso',dict(upperbody=1,belly=.25),.55,1.11)
ramp('cuff_air',dict(cuff=-1,tail=.35),.10,.33)
ramp('sleeve_tip',dict(lowedge=1,waist_edge=-.36),.13,.55)
ramp('toe_low',dict(rear_height=1,toe_height=-1),.070,.35)
ramp('bag_arch',dict(mid_height=1,rear_height=-.5,toe_height=-.5),.08,.48)
ramp('bag_flat',dict(toe_height=1,rear_height=-1),-.16,-.015)
ramp('pale_straps',dict(photogap_v=.7,photogap_h=.3),.045,.20)
ramp('over_sole',dict(toehole=.52,heelhole=.48),.040,.17)
ramp('shaft_high',dict(rear_height=1,toe_height=-.28),.62,1.05)
ramp('knitted_hem',dict(tail=-1,lowedge=.60),-.41,.03)
ramp('tallish',dict(height=1,width=-1),.04,.31)
ramp('broad',dict(width=1,height=-1),.10,.45)
ramp('short_body',dict(height=-1),-.90,-.65)
ramp('sleeve_gap',dict(sleeveL=.5,sleeveR=.5),.18,.66)
ramp('broadbelt',dict(belt_w=1,shoulder_w=-1),-.10,.025)
ramp('shoulders',dict(shoulder_w=1,belt_w=-1),.095,.38)
ramp('tapertop',dict(shoulder_w=1,lower_w=-1),.080,.36)
ramp('splayed',dict(lower_w=1,middle_w=-1),.055,.26)
ramp('underarms',dict(sleeve_center=1,waistwidth=-.75,lower_w=.1),.045,.47)
ramp('centerfasten',dict(front=1,leftside=-.39,rightside=-.39),.09,.50)
ramp('no_seam',dict(front=-1,leftside=.16,rightside=.16),-.23,-.04)
ramp('leg_air',dict(leg_air=1),.13,.66)
ramp('strap_density',dict(photogap_v=.62,photogap_h=.38,area=-.25),.006,.093)
ramp('strap_edges',dict(breaks_y=1,breaks_x=.6,area=-.13),.043,.126)
ramp('cutout_heel',dict(arch_space=1),.12,.48)
ramp('axial_seam',dict(axial=1,offaxis=-.26,outeraxis=-.26),.45,1.48)
ramp('dark_zip',dict(front=1),.07,.37)
ramp('rear_tall',dict(back_span=1),.44,.68)
ramp('shoe_is_low',dict(back_span=-1),-.50,-.27)
ramp('side_profile',dict(back_span=1,fore_span=-1),.08,.35)
ramp('center_handle',dict(mid_height=1,toe_height=-.45,rear_height=-.55),.065,.32)
ramp('low_torso',dict(near_bottom=1,cuff=-.63),.19,.76)
r=rel('cue_ramps',affine('interpret_details',observ,coef,bias));
# Rules for prototype families; entries can vary with the cut.
B=[ [0,-.016,-.009,-.050], [0,0,-.005,-.015], [0,-.004,-.012,-.037], [0,0,-.004,-.003], [0,-.012,0,-.007], [0,0,0,-.014], [-.007,-.016,-.037,-.033], [0,0,-.012,-.009], [0,0,0,0], [0,0,0,-.006] ]
A=np.zeros((10,4,len(conditions)))
def vote(n,**kw):
    for k,value in kw.items():A[int(k[1:]),:,conditions.index(n)]=value
vote('front_opening',c0=-.021,c2=-.035,c4=.036,c6=.014)
vote('strong_opening',c2=-.018,c4=.036,c6=.007)
vote('collar',c0=-.014,c2=-.008,c4=.014,c6=.023)
vote('print',c0=-.006,c2=-.014,c4=-.021,c6=.043,c3=.008)
vote('checks',c0=-.008,c2=-.005,c4=-.010,c6=.027)
vote('clearstraps',c5=.069,c7=-.028,c9=-.025)
vote('veryopen',c5=.054,c7=-.012,c9=-.020)
vote('knitband',c2=.025,c4=-.011,c6=-.009)
vote('highcollar',c4=.011,c2=.003)
vote('scoop',c4=-.018,c0=.012,c6=.008)
vote('longtail',c4=.007,c2=-.009,c6=.006)
vote('legsplit',c1=.19,c3=-.030,c4=-.015)
vote('no_split',c1=-.010) # touching trouser legs need not show a gap
vote('belt',c1=.047,c3=-.014)
vote('not_belt',c1=-.008)
vote('skirt',c3=.115,c4=-.025,c1=-.02)
vote('thinwaist',c3=.041,c4=-.019)
vote('shortsleeve',c0=[.094,.015,.085,.003],c6=[-.024,-.012,.048,.036],c3=.014,c2=-.031,c4=-.045)
vote('nocrown',c8=.016,c4=-.019,c0=[0,.045,0,0],c3=.012)
vote('shaft_slope',c8=-.057,c9=.054,c7=-.006)
vote('toe_slope',c7=.042,c8=-.090,c9=.022)
vote('toe_box',c8=.023,c5=-.015)
vote('holes_v',c5=.17,c7=-.051,c9=-.050,c8=-.030)
vote('holes_h',c5=.13,c7=-.035,c9=-.022)
vote('torso',c5=-.030,c7=-.021)
vote('cuff_air',c2=-.032,c4=-.03,c6=.006,c0=.014,c3=.015)
vote('sleeve_tip',c2=.015,c6=[.014,.013,0,0],c0=[-.03,-.014,-.03,0])
vote('toe_low',c8=-.20,c5=.034,c7=.060,c9=.017)
vote('bag_arch',c8=.068,c9=-.043)
vote('bag_flat',c8=.014,c9=-.033)
vote('pale_straps',c5=.14,c7=-.046,c9=-.030)
vote('over_sole',c5=.075,c7=-.017,c9=-.030)
vote('shaft_high',c7=-.085,c9=.065)
vote('knitted_hem',c2=.038,c4=-.012)
vote('tallish',c8=-.085,c1=.015,c5=-.038)
vote('broad',c8=.048,c2=-.037,c4=-.045,c6=-.025)
vote('short_body',c8=.065,c2=-.030,c4=-.052,c1=-.051)
vote('sleeve_gap',c2=.072,c4=.042,c6=.034,c8=-.115,c1=-.022)
vote('broadbelt',c1=.082,c2=-.015)
vote('shoulders',c1=-.135,c8=-.013,c6=.01)
vote('tapertop',c1=.043,c3=-.033,c0=.028)
vote('splayed',c3=.085,c1=-.120,c4=-.008)
vote('underarms',c2=.015,c6=.018,c1=-.035)
vote('centerfasten',c2=-.041,c0=-.015,c4=.073,c6=.034)
vote('no_seam',c2=.031,c4=-.018)
vote('leg_air',c1=.060,c3=-.018)
vote('strap_density',c5=.16,c7=-.055,c9=-.040)
vote('strap_edges',c5=.060,c7=-.022)
vote('cutout_heel',c5=.098,c9=-.030)
# Aspect alone does not identify a bag: inspect whether there is a body and a
# handle instead. A sweater can have a tall center as well, but with distinct arms.
vote('bag_arch',c8=.017,c9=-.025)
# A high collar or detailed neck is only evidence of outerwear when there are
# also sleeves. In particular it says little about an asymmetric shoe or bag.
vote('axial_seam',c4=.102,c2=-.037,c6=.043,c0=-.016)
vote('dark_zip',c4=.029,c6=.014,c2=-.008)
vote('rear_tall',c9=.073,c7=-.069)
vote('shoe_is_low',c7=.040,c9=-.073)
vote('side_profile',c8=-.082,c7=.044,c5=.035,c9=.018)
vote('center_handle',c8=.038)
vote('low_torso',c4=.032,c6=.012,c2=-.010)
# Pool correlated observations into families. The sum of many weak positive
# cues (e.g. several different tests all rediscovering an ankle) is NOT
# independent evidence. Bound the contributions of each family in advance.
families=[
 ('shape',.091,'legsplit no_split belt not_belt skirt thinwaist shortsleeve broadbelt shoulders tapertop splayed leg_air'),
 ('fabric',.075,'print checks collar scoop highcollar nocrown'),
 ('opening',.100,'front_opening strong_opening centerfasten no_seam axial_seam dark_zip'),
 ('hem',.080,'knitband longtail cuff_air sleeve_tip knitted_hem underarms sleeve_gap low_torso'),
 ('perspective',.105,'toe_slope shaft_slope toe_box torso toe_low bag_arch bag_flat shaft_high rear_tall shoe_is_low side_profile center_handle'),
 ('air',.17,'clearstraps veryopen holes_v holes_h pale_straps over_sole strap_density strap_edges cutout_heel'),
 ('proportion',.068,'tallish broad short_body')]
# Sleeve *identity* does not distinguish long-sleeved garment types. Likewise
# a closed-shirt front can occur on coats; the absence of a seam is only weak
# evidence, not proof of pullover construction.
vote('sleeve_gap',c2=.054,c4=.059,c6=.048,c8=-.093,c1=-.012)
vote('broadbelt',c1=.024,c2=-.001)
vote('nocrown',c8=.006,c4=-.008,c0=[0,.032,0,0],c3=.012)
vote('no_seam',c2=.015,c4=-.006)
vote('knitted_hem',c2=.019,c4=-.003)
vote('splayed',c3=.076,c1=-.115,c4=-.005)
vote('short_body',c8=.030,c2=-.018,c4=-.027,c1=-.033)
vote('broad',c8=.025,c2=-.016,c4=-.021,c6=-.011)
vote('tallish',c8=-.050,c1=.002,c5=-.038)
sc=[]
for n,cap,nn in families:
    en=np.zeros(len(conditions));
    for v in nn.split():en[conditions.index(v)]=1
    a=(A*en).reshape(40,-1); w=np.stack([a,-a],axis=-1).reshape(40,-1)
    # clip(x,-c,c) = relu(x+c)-relu(x-c)-c
    w=np.stack([w,w],axis=1).reshape(80,-1); z=rel(n+'_gated',affine(n+'_family',r,w,np.tile([cap,-cap],40))); t=np.eye(40).repeat(2,axis=1);t[:,1::2]*=-1
    sc.append(affine(n+'_vote',z,t,-cap))
f=shape('garment_families',add('combine',affine('cut_prior',parts,np.eye(40),np.asarray(B).ravel()),*sc),10,1,4);f=pool('choose_cut',f,(1,4));initial=shape('item_evidence',f,10)
# Superordinate silhouette evidence: an ambiguous sweatshirt/jacket should
# nevertheless be identifiable as an upper garment compared to a bag. Use a
# modest common family vote, independent of which detailed cut was best.
raw=shape('coarse_families',parts,10,1,4);raw=shape('coarse_items',pool('coarse_cuts',raw,(1,4)),10)
sm=np.zeros((12,10)); groups=[[0,2,4,6],[1]*4,[3]*4];
for j,gg in enumerate(groups):
    for i,k in enumerate(gg):sm[j*4+i,k]=1
family=shape('body_types',affine('body_order',raw,sm),3,1,4);body=pool('body_type',family,(1,4))
# Correlative template evidence for leg cleft vs skirt independently of which
# shirt wins. Combine grouped correlation and per-class detail scores.
w=np.zeros((10,3));
for i in [0,2,4,6]:w[i,0]=1
w[1,1]=1;w[3,2]=1
q=affine('family_evidence',body,w*.24);v=np.zeros((10,10));
for i in [0,1,2,3,4,6]:v[i,i]=-.24
rest=affine('balance_family',raw,v)
add('scores',initial,q,rest)

dest=Path(sys.argv[1]);dest.mkdir(parents=True,exist_ok=True)
(dest/'network.json').write_text(json.dumps({'layers':L},indent=1)+'\n');np.savez(dest/'weights.npz',**W)
