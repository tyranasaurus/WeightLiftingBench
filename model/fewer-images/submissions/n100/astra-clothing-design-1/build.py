"""Data-free, hand-specified fashion recognizer. Only numpy and the standard library.

Draw in physical units relative to garment length (or shoe/bag width), as if
laying out a paper pattern. Each drawing contributes ONE normalized filter.
There is no estimation step and no input file.
"""
import sys, json
from pathlib import Path
import numpy as np

layers=[]; weights={}
def node(name,op,ins,**kw):
    layers.append(dict(name=name,op=op,inputs=[ins] if isinstance(ins,str) else ins,**kw)); return name
def linear(name,src,W,b=None,conv=False,pad=0):
    W=np.asarray(W,'f'); n=len(W)
    weights[name+'.W']=W
    weights[name+'.b']=np.zeros(n,'f') if b is None else np.broadcast_to(b,(n,)).astype('f').copy()
    if conv: return node(name,'conv',src,out=n,kernel=list(W.shape[-2:]),stride=1,padding=pad)
    return node(name,'dense',src,out=n)
def pw(name,src,W,b=0):return linear(name,src,np.asarray(W)[:,:,None,None],b,True)
def pool(name,op,src,h,w,sh,sw):return node(name,op,src,size=[h,w],stride=[sh,sw])

# Foreground measurement. Two soft reflectance cutoffs: a dark fabric still has
# an outline, while daylight through a sandal/handle should remain background.
pw('ink_slopes','image',[[22],[22],[6],[6]],[-.22,-1.22,-.21,-1.21]);node('ink_hinges','relu','ink_slopes')
pw('soft_ink','ink_hinges',[[.64,-.64,.36,-.36]])
pool('downsample','avgpool','soft_ink',2,2,2,2);node('shape_contrast','norm','downsample')

rng=np.random.default_rng(912732)
u=rng.uniform
def pick(a):return a[rng.integers(len(a))]
size=28
axis=(np.arange(size)+.5)/size

class Paper:
    # Polygon scan conversion with a broadcast grid; supersampling is implicit
    # in the final 2x2 bin integral. Curves are piecewise-linear paper edges.
    def __init__(self):
        self.pieces=[]
    def poly(self,points,ink=True):
        self.pieces.append((np.asarray(points,float),ink))
    def both(self,side,ink=True):
        self.poly(side+[(-x,y) for x,y in reversed(side)],ink)
    def pair(self,points,ink=True):
        self.poly(points,ink); self.poly([(-x,y) for x,y in points],ink)
    def draw(self):
        pts=np.concatenate([p for p,ink in self.pieces if ink],axis=0)
        a,b=pts.min(axis=0); c,d=pts.max(axis=0)
        extent=max(c-a,d-b)/u(.94,1.01)
        # Both raw and independently normalized aspects are useful: the latter
        # compares the internal shape while ignoring the size of the margin.
        extent_x=extent;extent_y=extent
        if getattr(self,'canonical',False):
            extent_x=(c-a)/u(.95,1.02);extent_y=(d-b)/u(.95,1.02)
        vertical_slack=max(0., extent_y-(d-b))/extent_y
        offset=u(-.48,.48)*vertical_slack + u(-.023,.023)
        y=(axis[:,None]-.5+offset)*extent_y+(b+d)/2
        x=(axis[None,:]-.5+u(-.016,.016))*extent_x+(a+c)/2+(y-(b+d)/2)*u(-.028,.028)
        mask=np.zeros((size,size),bool)
        for pp,ink in self.pieces:
            inside=np.zeros_like(mask)
            for k in range(len(pp)):
                x1,y1=pp[k-1];x2,y2=pp[k]
                if y1!=y2:
                    inside ^= ((y1>y)!=(y2>y)) & (x < x1+(y-y1)*((x2-x1)/(y2-y1)))
            if ink: mask|=inside
            else: mask&=~inside
        z=mask.reshape(14,2,14,2).mean((1,3));z-=z.mean()
        return (z.ravel()/(np.linalg.norm(z)*14+1.e-9)).astype('f')

# For every top: B half-chest, H half-hem, T half-shoulder, P sleeve end.
# Anatomical variation, independent of observed samples.
def top(style):
    p=Paper()
    wscale=u(.89,1.12); neck=u(.085,.158); shoulder_y=u(.05,.15)
    if style in ('tee','polo','shortknit'):
        B=u(.261,.352); T=B*u(.91,1.13); H=B*u(.90,1.12); waist=B*u(.91,1.03)
        P=T+u(.105,.205); end=u(.233,.38) if style=='tee' else u(.31,.53)
        hemRise=u(.002,.038) if style!='polo' else u(.015,.10)
        side=[(neck,0),(T,shoulder_y),(P,end-u(.028,.074)),(P-u(.08,.145),end),(B,.30),(waist,.64),(H,1-hemRise),(H*.79,1),(0,1)]
    elif style in ('tank','blouse'):
        B=u(.21,.32);H=B*u(.91,1.19);T=B*u(.73,1.14); dep=u(.14,.29)
        side=[(neck,-.001),(T,shoulder_y*.3),(T*.96,dep*.66),(B,dep),(B*u(.85,1),.59),(H,.98),(H*.72,1),(0,1)]
    else:
        # Measurements along a sleeve are relative to the hem, not independent
        # row masks. A cuff may hang below the sweater's waistband.
        if style=='coat':
            long=u(0,1)<.47
            B=u(.172,.265) if long else u(.235,.326)
            H=B*(u(1.06,1.32) if long else u(.99,1.19));T=B*u(.98,1.20)
            end=u(.60,.87) if long else u(.87,1.06); waist=B*u(.93,1.02)
            hemRise=u(0,.044); spacing=u(-.055,.046); cuff=u(.075,.116)
        elif style in ('shirt','long'):
            B=u(.215,.34);H=B*u(.86,1.20);T=B*u(.97,1.25)
            end=u(.77,1.09);waist=B*u(.83,1.06);hemRise=u(.018,.115); spacing=u(-.04,.095);cuff=u(.052,.104)
        else:
            B=u(.24,.338);T=B*u(.99,1.22);H=B*u(.79,1.03);waist=B*u(.90,1.04)
            end=u(.94,1.20) if u(0,1)<.84 else u(.79,.95);hemRise=u(0,.04);spacing=u(.001,.10);cuff=u(.061,.105)
        shY=shoulder_y if style!='coat' else u(.11,.19)
        outer=H+spacing+cuff; elbow=max(T+.05,outer-u(.005,.067))
        side=[(neck,0),(T,shY),(T+u(.024,.068),shY+.09),(elbow,end*.55),(outer,end),(outer-cuff,end+u(-.015,.014)),(B+.015,end*.62),(B,.34),(waist,.67),(H,1-hemRise),(H*.63,1),(0,1)]
    # Close over the collar; the centre of a low neckline can be visibly recessed.
    neckdip=u(.006,.063)
    if style=='tank': neckdip=u(.09,.22)
    if style in ('shirt','polo','coat'): neckdip=u(-.013,.026)
    side=side[:-1];side=[(0,neckdip),(neck*.56,neckdip*.84)]+side
    p.both([(x*wscale,y) for x,y in side]);return p

def pants():
    p=Paper(); hip=u(.144,.245);waist=hip*u(.79,1.04); crotch=u(.23,.40)
    ankle=hip*u(.65,1.42); leg=u(.065,min(.152,ankle*.98));knee=u(.58,.73)
    p.both([(waist,u(0,.03)),(hip,u(.12,.23)),((ankle+hip)*.51,knee),(ankle,u(.974,1)),(ankle-leg,1),(max(.001,(ankle-leg)*.6),knee),(0,crotch)])
    return p

def dress():
    p=Paper();B=u(.105,.215);T=B*u(.88,1.18); H=u(.194,.35) if u(0,1)<.62 else B*u(.86,1.41); waist=B*u(.66,1.03); wy=u(.34,.52);nn=B*u(.29,.69);kind=u(0,1)
    side=[(0,u(.012,.06)),(nn,.0),(T,u(.02,.075))]
    if kind<.31:side += [(T*1.28,u(.105,.19)),(B,u(.16,.25))]
    elif kind<.71:
        e=u(.30,.73);o=T+u(.015,.095);side +=[(T+.04,.15),(o,e),(o-u(.04,.085),e+.008),(B,.20)]
    else:side +=[(T*.90,.103),(B,.19)]
    side +=[(waist,wy),(waist*.63+H*.37,wy+(1-wy)*.39),(H,.97),(H*.65,1)]
    p.both(side);return p

def bag():
    p=Paper(); h=u(.44,.86); bot=u(.30,.49); topw=u(.31,.5); typ=u(0,1)
    if typ<.09: h=u(.87,1.28); bot=u(.37,.48);topw=u(.30,.45)
    p.both([(topw*.80,0),(topw,.034),(max(bot,topw)+u(0,.042),h*.64),(bot,h*.94),(bot*.86,h)])
    if typ<.76:
        spread=u(.13,.39);rise=u(.18,.69); thick=u(.022,.065);outer=spread*u(.51,.91)
        if typ<.14: outer=spread*u(.05,.46)
        p.both([(outer*.54,-rise),(outer,-rise*.91),(spread,-.06),(spread+.01,.025),(spread-thick,.025),(outer-thick,-rise*.86),(outer*.48,-rise+thick)])
    elif typ>.86:
        # A side strap or wrist loop next to the pouch.
        p.poly([(topw*.9,0),(topw+.067,.06),(topw+.136,h*.74),(topw+.080,h*.8),(topw*.91,h*.88),(topw*.98,h*.80),(topw+.106,h*.70),(topw+.04,.08)])
    return p

def footwear(c):
    p=Paper();toe=u(.045,.105); R=u(.85,.975);typ=u(0,1); t=u(.045,.113); heelrise=0
    if c==7:
        hi=typ<.105; rear=u(.18,.305) if not hi else u(.32,.48); tongue=rear*u(.96,1.42)
        p.poly([(0,-.045),(.009,-toe),(.078,-toe*1.26),(.18,-toe*1.55),(.36,-tongue*.70),(u(.46,.63),-tongue),(.65,-tongue*.85),(.76,-rear*.73),(.86,-rear*.81),(R,-rear),(1,-rear*.53),(1,-.028),(.94,0),(.57,-.014),(.25,0),(.079,-.012)])
    elif c==9:
        shaft=u(.42,.90) if typ<.84 else u(.31,.44);left=u(.38,.62); right=u(.84,.99);heelrise=u(.005,.16)
        bot=[(1,-heelrise*.2),(.85,-heelrise*.17),(.82,-heelrise),(.70,-heelrise),(.42,-.02),(.27,0),(.05,-.02)]
        p.poly([(0,-.045),(.012,-toe*.99),(.10,-toe*1.38),(.25,-toe*1.72),(left-.03,-shaft*.48),(left+u(-.08,.06),-shaft),(left+.10,-shaft*u(.91,1.02)),(right,-shaft*u(.91,1.07)),(right-u(-.03,.05),-shaft*.54),(.985,-shaft*.27)]+bot)
    else:
        heelrise=u(.10,.44) if typ<.49 else u(0,.09);heelw=u(.032,.16);ankle=u(.10,.29)+heelrise
        # Sole in profile, with the heel sometimes supported only at the rear.
        p.poly([(0,-.035),(.025,-t),(.22,-t*.93),(.38,-t*.88),(.73,-t-heelrise),(.98,-t-heelrise),(1,-heelrise),(.78,-heelrise),(.43,-.019),(.29,0),(.06,-.012)])
        p.poly([(.98-heelw,-heelrise-.02),(.987,-heelrise-.025),(.98,-.005),(.98-heelw*.73,-.005)])
        # Straps can be broad, but leave an aperture above the footbed.
        front=u(.09,.28); sw=u(.063,.21);lift=u(.066,.16)
        p.poly([(front-sw*.46,-.04),(front,-t-lift),(front+sw,-t-lift*1.06),(front+sw*.74,-.06)])
        if typ<.8:
            # The ankle strap can lie above and behind the instep strap.
            low=u(.36,.70);hi=u(.64,.89); th=u(.035,.09)
            p.poly([(low,-t*.8),(low+.08,-t*.85),(hi,-ankle+th),(.95,-ankle+th*.8),(.95,-ankle),(hi-.04,-ankle)])
            if typ<.28:
                p.poly([(low,-heelrise*.6-t),(low+.06,-heelrise*.6-t),(hi-.025,-ankle+.02),(hi-.055,-ankle)])
            p.poly([(.85,-t-heelrise+.024),(.99,-t-heelrise+.015),(.966,-ankle),(.90,-ankle)])
        else:
            p.poly([(.4,-.052),(.52,-t-.12),(.66,-t-.10),(.56,-.06)])
    return p

# The bank is organized in independent groups to max-pool without relying on
# a single lucky polygon. All samples have equal standing within each group.
def pattern(style):
    if isinstance(style,str):return top(style)
    if style==1:return pants()
    if style==3:return dress()
    if style==8:return bag()
    return footwear(style)
styles=['tee','tank','long',1,'knit','shortknit',3,'coat',5,'shirt','polo','blouse',7,8,9]
ng=3; per=512
W=np.stack([pattern(style).draw() for style in styles for group in range(ng) for i in range(per)])
linear('paper_patterns','shape_contrast',W)
node('grouped','reshape','paper_patterns',shape=[len(styles),ng,per]);pool('best_match','maxpool','grouped',1,per,1,per)
pool('consensus_raw','avgpool','best_match',ng,1,ng,1)
# Spatial transformer implemented with fixed affine samplers and ReLUs. Ray
# maxima estimate the two spans. Neighboring scale hypotheses are interpolated,
# so we need no custom operators or learned localization parameters.
# For the horizontal direction the crop is centered; for the vertical
# direction try two offsets toward a product resting lower in its frame.
def stripspan(name,src,vertical):
    x=pool(name+'_projection','maxpool',src,1 if vertical else 28,28 if vertical else 1,1 if vertical else 28,28 if vertical else 1)
    return linear(name,x,np.ones((1,28))/28)
def sample_axis(name,src,span,scales,vertical,shift):
    scales=np.array(scales,float); l=len(scales);ridge=[]
    # Piecewise linear barycentric coordinates over the scale proposals.
    linear(name+'_dist',span,np.ones((l,1)),-scales);node(name+'_right','relu',name+'_dist')
    der=np.zeros((l,l));
    for j in range(l-1):
        step=1/(scales[j+1]-scales[j]);der[j,j]-=step;der[j+1,j]+=step;der[j,j+1]+=step;der[j+1,j+1]-=step
    gate=linear(name+'_gates',name+'_right',der,[1]+[0]*(l-1))
    # Pack axis fibres as channels and use 1x1 convolutions to resample all
    # rows/columns in parallel. Do not form a giant spatially dense warp.
    if not vertical:
        # transpose the small 14x14 image with a permutation matrix
        trans=np.eye(196)[np.arange(196).reshape(14,14).T.ravel()]
        src=linear(name+'_transpose',src,trans)
    v=node(name+'_fibres','reshape',src,shape=[14,1,14]); mat=np.zeros((l,14,14))
    for j,s in enumerate(scales):
        for i in range(14):
            pos=(i-6.5)*s+6.5 + (1-s)*13*shift;low=int(np.floor(pos));frac=pos-low
            for k,coef in [(low,1-frac),(low+1,frac)]:
                if 0<=k<14: mat[j,i,k]+=coef
    val=pw(name+'_sample',v,mat.reshape(l*14,14)); flat=node(name+'_values','reshape',val,shape=[l*196]);
    # Soft gating using min(x,g), appropriate to almost-binary foreground.
    # Around an edge this interpolation also acts as mild dilation.
    gg=linear(name+'_negative_gate',gate,-np.eye(l).repeat(196,axis=0));diff=node(name+'_excess','add',[flat,gg]);pos=node(name+'_positive','relu',diff)
    vpos=node(name+'_excess_channels','reshape',pos,shape=[l,14,14]);vval=node(name+'_sample_channels','reshape',val,shape=[l,14,14]);
    neg=pw(name+'_remove',vpos,-np.ones((1,l)));yes=pw(name+'_keep',vval,np.ones((1,l)));merged=node(name+'_merged','add',[yes,neg])
    if not vertical:
        return node(name+'_image','reshape',linear(name+'_restore',merged,trans),shape=[1,14,14])
    return merged
spanx=stripspan('span_x','soft_ink',False);spany=stripspan('span_y','soft_ink',True)
canonical_x=sample_axis('horizontal','downsample',spanx,[.30,.45,.60,.75,.89,1],False,0)
canonical=sample_axis('vertical',canonical_x,spany,[.19,.30,.42,.56,.70,.84,1],True,.14)
canon_norm=node('canonical_contrast','norm',canonical)
def canonical_pattern(style):
    p=pattern(style);p.canonical=True;return p.draw()
W=np.stack([canonical_pattern(style) for style in styles for j in range(ng*per)])
linear('normalized_patterns',canon_norm,W);node('normalized_groups','reshape','normalized_patterns',shape=[len(styles),ng,per]);pool('normalized_matches','maxpool','normalized_groups',1,per,1,per);pool('canonical_consensus','avgpool','normalized_matches',ng,1,ng,1)
# Discard absolute aspect only for the style-specific part of the vote. The
# original correlations still supply most of the between-family evidence.
pw('vote_raw','consensus_raw',np.eye(len(styles))*.69);pw('vote_canonical','canonical_consensus',np.eye(len(styles))*.31);node('consensus','add',['vote_raw','vote_canonical'])
# Keep a short-sleeve tee match distinct from that for a long-sleeve tee;
# rare styles should not get the same prior as the defining style.
style_groups=[[(0,.015),(1,0),(2,-.075)],[(3,0)],[(4,0),(5,-.075)],[(6,0)],[(7,0)],[(8,0)],[(9,0),(10,-.012),(11,-.032)],[(12,0)],[(13,0)],[(14,0)]]
row=[];bias=[]
for group in style_groups:
    for j in range(3):
        i,b=group[min(j,len(group)-1)];row.append(np.eye(len(styles))[i]);bias.append(b)
for label,src in [('silhouettes','consensus_raw'),('resized_styles','consensus')]:
    lin=linear(label+'_prior',src,row,bias);v=node(label+'_class_styles','reshape',lin,shape=[10,1,3]);po=pool(label+'_best','maxpool',v,1,3,1,3);node(label,'reshape',po,shape=[10])

# A separate grayscale branch: shirt fronts have a placket and sometimes
# buttons; outerwear often has a prominent zip, lapels, or separated panels.
# Pullovers tend to have uninterrupted fronts, and tees carry chest prints.
node('photo_contrast','norm','image')
filters=[]
def filt(items):
    a=np.zeros((5,5));
    for dy,dx,v in items:a[dy+2,dx+2]+=v
    filters.append(a)
# Three vertical ridge/valley widths (positive = darker than the flanks).
for dx in (1,2):
    filt([(j,i*dx,w/3) for j in (-1,0,1) for i,w in ((-1,.5),(0,-1),(1,.5))])
# paired first derivatives for detecting edges as well as thin lines
filt([(j,i,w/3) for j in (-1,0,1) for i,w in ((-1,-.5),(1,.5))]);filt([(i,j,w/3) for j in (-1,0,1) for i,w in ((-1,-.5),(1,.5))]);
filt([(j,i,.25) for j,i in ((1,0),(0,1),(-1,0),(0,-1))]+[(0,0,-1)])
filt([(i,j,w/3) for j in (-1,0,1) for i,w in ((-2,.5),(0,-1),(2,.5))])
filt([(0,0,1)])
f=np.array(filters);linear('stitches','photo_contrast',np.concatenate((f,-f))[:,None,:,:],conv=True,pad=2);node('stitch_energy','relu','stitches')
# Sparse, hand-written spatial measurements. Give each centre-line detector
# a side-band control to discount prints/plaids covering the entire garment.
Y,X=np.mgrid[:28,:28];n=len(f); observations=[]; names=[]
def measure(name,parts):
    a=np.zeros((n*2,28,28));
    for ch,mask,scale in parts:
        if np.any(mask):a[ch,mask]+=scale/mask.sum()
    observations.append(a.ravel());names.append(name)
def rect(l,r,t,b):return (X>=l)&(X<r)&(Y>=t)&(Y<b)
def signed(ch,mask,coef):return [(ch,mask,coef),(ch+n,mask,-coef)]
def unsigned(ch,mask,coef):return [(ch,mask,coef),(ch+n,mask,coef)]
central=rect(9,19,8,22); flanks=central & (abs(X-13.5)>=2.5)
for i in range(10,18):
    for lo,hi,k in [(7,24,'all'),(7,16,'high'),(16,24,'low')]:
        col=rect(i,i+1,lo,hi);ref=rect(8,20,lo,hi)&(abs(X-13.5)>2.5)
        measure('line%d%s'%(i,k),signed(1,col,1)+signed(1,ref,-.65))
        measure('edge%d%s'%(i,k),signed(2,col,1)+signed(2,ref,-.65))
    measure('button%d'%i,unsigned(4,rect(i,i+1,8,23),1)+unsigned(4,flanks,-.8))
for name,mask in [('chest',rect(8,20,6,15)),('belly',rect(9,19,16,23)),('bib',rect(11,17,3,8)),('sides',(rect(6,10,8,20)|rect(18,22,8,20))),('allfront',central)]:
    for ch in [1,2,3,4,5]:measure(name+str(ch),unsigned(ch,mask,1))
    measure('light'+name,signed(6,mask,1))
# Contrast just under vs inside the neck. A fold-over collar usually has a
# contrasting or dark opening, whereas knit neck bands are shallow.
for i in (12,13,14,15):
    for t in (1,2,3):
        measure('coll%d_%d'%(i,t),signed(6,rect(i-1,i+2,t+1,t+4),-1)+signed(6,rect(8,11,t+3,t+6)|rect(17,20,t+3,t+6),1))
# Fabric print is not the same as a placket. Detect high-frequency texture on
# both halves of the shirt body, and separately a narrow line at its midpoint.
for s,mask in [('lcloth',rect(9,13,8,21)),('rcloth',rect(15,19,8,21)),('waistband',rect(9,19,21,26))]:
    for j in [0,1,2,3,4,5]:measure(s+str(j),unsigned(j,mask,1))
for side in [-1,1]:
    # Two diagonals under the neck: the notched lapels or points of a collar.
    for width in (0,1,2):
        mask=(abs(X-(13.5+side*(width+.47*(8-Y))))<1)&(Y>=2)&(Y<=8)
        measure('lapel'+str(side)+str(width),unsigned(2,mask,.65)+unsigned(1,mask,.35))
# For a front zipper or lapel the same line continues down both the left
# and right front panels. Compare upper/lower central structure with the
# adjacent material rather than counting every isolated fold as a closure.
for half,lo,hi in [('yoke',4,12),('fasten',12,21),('tail',21,25)]:
    for t in [0,1,2,3,4,5]:measure(half+str(t),unsigned(t,rect(12,16,lo,hi),1)+unsigned(t,rect(8,12,lo,hi)|rect(16,20,lo,hi),-.85))
# Broad panel imbalance, lapels and patch pockets. In contrast to a knit,
# even a low-contrast coat often has more structure at the front than sides.
for half,lo,hi in [('u',7,14),('d',14,22)]:
    measure('panel'+half,signed(6,rect(9,13,lo,hi),1)+signed(6,rect(15,19,lo,hi),-1))
    for w in [1,2,3]:
        measure('fly'+half+str(w),signed(6,rect(14-w,14+w,lo,hi),-1)+signed(6,rect(9-w,14-w,lo,hi)|rect(14+w,19+w,lo,hi),1))
linear('details','stitch_energy',observations)

# Small network of hand-authored rectified rules. Compile symbolic expressions
# so there is a clear correspondence between every coefficient and its cue.
class Expr:
    def __init__(self,w,b=0):self.w=np.array(w,float);self.b=b
    def __add__(self,o):
        if not isinstance(o,Expr):return Expr(self.w,self.b+o)
        return Expr(self.w+o.w,self.b+o.b)
    __radd__=__add__
    def __mul__(self,k):return Expr(self.w*k,self.b*k)
    __rmul__=__mul__
    def __neg__(self):return self*-1
    def __sub__(self,o):return self+-o
    def __rsub__(self,o):return -self+o

def relus(label,src,ex):
    linear(label,src,[a.w for a in ex],[a.b for a in ex]);return node(label+'_positive','relu',label)
def linex(label,src,ex):return linear(label,src,[a.w for a in ex],[a.b for a in ex])
def basis(n):return [Expr(x) for x in np.eye(n)]
e=dict(zip(names,basis(len(names))));rules=[]; coeff=[]
def rule(z,cl):
    rules.append(z); coeff.append([cl.get(i,0) for i in range(10)])
# Each contribution here is in correlation-score units. Thresholds are in the
# instance-normalized image's contrast units, not raw 8-bit grey values.
for i in range(10,18):
    center=1 if 12<=i<=15 else .45
    for s in (-1,1):
        for feat,gain in [('line',.076),('edge',.056)]:
            z=e[feat+str(i)+'all']*s
            # Saturating ramp for a vertical seam; edge and ridge evidence add.
            cl={0:-.20,2:-.64,4:.72,6:.34}; g=gain*center
            rule(z-.105,{k:v*g for k,v in cl.items()});rule(z-.69,{k:-v*g for k,v in cl.items()})
        z=e['line%dhigh'%i]*s
        rule(z-.22,{6:.014*center,4:.012*center,2:-.012*center})
    rule(e['button%d'%i]-.07,{6:.062*center,4:.022*center,2:-.032*center})
for i in (12,13,14,15):
    for t in (1,2,3):
        z=e['coll%d_%d'%(i,t)]
        rule(z-.37,{6:.0043,4:.0035,0:-.0014,2:-.0036})
        rule(z-1.55,{6:-.0043,4:-.0035,0:.0014,2:.0036})
# Printed decoration localized to the chest vs an all-over woven pattern.
z=.55*e['chest1']+.45*e['chest5']-.45*e['belly1']-.55*e['belly5']
rule(z-.09,{0:.14,2:.035,4:-.048,6:-.047})
z=.4*e['belly1']+.4*e['belly5']+.2*e['belly4']
rule(z-.155,{6:.18,0:-.062,4:.02});rule(z-.52,{6:-.18,0:.062,4:-.02})
# A lighter contrasting shirt collar or a patch of bare neck.
rule(e['lightbib']-.5*e['lightchest']-.5*e['lightsides']-.41,{6:.027,0:.008})
# A horizontal rib or rolled band at the neck favours a pullover when there's
# not much more texture in the torso.
rule(e['bib5']-.5*e['bib1']-.5*e['chest5']-.17,{2:.05,4:-.017})
for ss in ('lcloth','rcloth'):
    # Woven checks and all-over floral prints are a useful cue for blouses.
    z=.55*e[ss+'4']+.28*e[ss+'0']+.17*e[ss+'3']
    rule(z-.12,{6:.245,2:-.047,4:-.068});rule(z-.53,{6:-.245,2:.047,4:.068})
z=e['waistband5']-e['belly5']
rule(z-.21,{2:.062,4:-.019,6:-.012})
for width in (0,1,2):
    z=.5*(e['lapel-1'+str(width)]+e['lapel1'+str(width)])-.50*e['chest1']
    rule(z-.28,{6:.032,4:.027,2:-.032,0:-.022})
for s in (-1,1):
    rule((e['panelu']+e['paneld'])*(.5*s)-.21,{4:.065,6:.050,2:-.060})
    for w in (1,2,3):
        rule((e['flyu'+str(w)]+e['flyd'+str(w)])*(.5*s)-.26,{4:.035,2:-.031,0:-.008,6:.008})
z=.36*e['fasten0']+.30*e['fasten1']+.34*e['fasten2']
rule(z-.05,{4:.22,2:-.12,6:.047})
z=.4*e['chest1']+.3*e['chest5']+.3*e['chest4']-.5*e['belly1']-.5*e['belly4']
rule(z-.14,{0:.12,6:-.13,4:-.05})
# Low uniform neck band and plain front: avoid treating every little spot or
# print as evidence for a shirt. On a button-down the collar concentrates the
# contrast in a small V below the neck.
rule(.25-e['bib1']-e['fasten0']-.55*e['lapel-10']-.55*e['lapel10'],{0:.095,2:.043,4:-.031,6:-.062})
src=relus('construction_rules','details',rules)
linear('construction_raw',src,np.transpose(coeff),[.002,0,.022,0,-.008,0,-.020,0,0,0])
# Texture by itself cannot turn a sleeved coat outline into a short-sleeve
# blouse. Bound its vote relative to the [-1,1] silhouette correlations.
I=np.eye(10);linear('detail_limits','construction_raw',np.vstack((I,I)),[.065]*10+[-.065]*10);node('bounded_details','relu','detail_limits');linear('construction','bounded_details',np.hstack((I,-I)),-.065)
# Complement global matching by asking whether a ray through the item encounters
# separated pieces of foreground. Across a trouser it encounters two legs;
# up through a sandal it can see daylight between a strap and the sole.
# For a solid cross-section the variation along a line is twice its maximum
# (include the two end edges). Additional up/down transitions indicate gaps.
f=np.zeros((4,1,3,3));f[0,0,1,1]=1; f[0,0,1,0]=-1;f[1]=-f[0];f[2]=f[0].transpose(0,2,1);f[3]=-f[2]
linear('outline_steps','soft_ink',f,conv=True,pad=1);node('step_lengths','relu','outline_steps')
pm=pool('ray_peak_y','maxpool','soft_ink',28,1,28,1)
# Exact soft hole depth, with an erosion in the vertical direction to ignore
# thin dark stripes. A point in a hole has foreground BOTH above and below it.
hz=28;eye=np.eye(28);rays=pw('column_pixels',node('turn_columns','reshape','soft_ink',shape=[28,1,28]),eye)
def sweep(label,src):
    for j in (1,2,4,8,16):
        q=np.zeros((28,28));q[j:,:-j]=np.eye(28-j);q-=eye
        delta=pw(label+'_diff'+str(j),src,q);r=node(label+'_up'+str(j),'relu',delta);src=node(label+'_max'+str(j),'add',[src,r])
    return src
up=sweep('above',rays);rev=pw('reverse_y',rays,eye[::-1]);low=pw('below',sweep('reversed_above',rev),eye[::-1]);neg=pw('neg_below',low,-eye)
differ=node('vertical_difference','add',[up,neg]);difpos=node('ray_vertical_excess','relu',differ)
gaps=node('holes','add',[up,pw('neg_excess',difpos,-eye),pw('neg_ink',rays,-eye)])
# 3-pixel erosion via De Morgan. Also retain ungated soft area.
gap2d=node('holes_image','reshape',gaps,shape=[1,28,28]);ngap=pw('neg_holes',gap2d,[[-1]]);er=pool('interior_dark','maxpool',ngap,3,1,1,1)
hole_regions=[]
for left,right in [(2,26),(6,21),(16,26)]:
    d=np.zeros((26,28));d[:,left:right]=-1/(right-left);hole_regions.append(d.ravel())
linear('clearance',er,hole_regions)
# One more residual term accounts for the far edge when ink touches the frame.
channels={}
for name,cols in [('mid',range(4,24)),('instep',range(9,21)),('back',range(16,26))]:
    cols=list(cols);a=np.zeros((4,28,28));a[2:,: ,cols]=.5/len(cols);b=np.zeros((28,28));b[-1,cols]=.5/len(cols);d=np.zeros(28);d[cols]=-1/len(cols)
    channels[name]=(a,b,d)
linear('crossings','step_lengths',[v[0].ravel() for v in channels.values()]);linear('floor_edges','soft_ink',[v[1].ravel() for v in channels.values()]);linear('subtract_solid',pm,[v[2] for v in channels.values()]);node('apertures','add',['crossings','floor_edges','subtract_solid'])
# Spatial occupancy contrasts. Use multiple physical cues rather than depending
# on very precise sleeve/leg placement in a particular drawing.
forms={}
def region(k,*parts):
    a=np.zeros((28,28));
    for mask,coef in parts:a[mask]+=coef/mask.sum()
    forms[k]=a.ravel()
def R(l,r,t,b):return rect(l,r,t,b)
region('crotch',(R(8,12,15,25)|R(16,20,15,25),1),(R(13,15,15,25),-1))
region('innerthigh',(R(10,13,13,25)|R(15,18,13,25),1),(R(13,15,13,25),-1))
region('skirt',(R(5,10,21,26)|R(18,23,21,26),1),(R(5,10,10,15)|R(18,23,10,15),-1))
region('fullsleeve',(R(2,7,14,22)|R(21,26,14,22),1))
region('shortsleeve',(R(1,6,5,11)|R(22,27,5,11),1),(R(1,6,14,23)|R(22,27,14,23),-1))
region('shoulder',(R(2,8,5,10)|R(20,26,5,10),1))
region('thin',(R(5,9,4,11)|R(19,23,4,11),-1))
region('hem',(R(9,19,25,28),1))
region('cuff',(R(2,6,24,28)|R(22,26,24,28),1))
region('longbody',(R(9,19,24,27),1),(R(3,7,24,27)|R(21,25,24,27),-1))
region('high',(R(10,18,2,8),1))
region('waistwidth',(R(0,28,11,16),1))
region('hipwidth',(R(0,28,21,25),1))
region('upperwidth',(R(0,28,4,8),1))
region('bodywidth',(R(0,28,8,21),1))
region('upper_sides',(R(3,8,6,15)|R(20,25,6,15),1))
region('foot_angle',(R(15,26,3,16),1),(R(2,13,3,16),-1))
region('sole_band',(R(3,25,17,23),1))
region('thin_centre',(R(11,17,6,18),1))
region('leg_opening',(R(10,13,20,26)|R(15,18,20,26),1),(R(13,15,20,26),-1))
region('shoulderwidth',(R(0,28,2,5),1))
region('lower_sides',(R(2,7,17,22)|R(21,26,17,22),1))
region('empty_neck',(R(11,17,1,3),-1),(R(8,11,1,3)|R(17,20,1,3),1))
linear('garment_proportions','soft_ink',list(forms.values()))
# The proportion of the bounding height filled by material (shoes span the
# width), and sudden peaks in column thickness where straps cross the foot.
linear('area_fraction','soft_ink',np.ones((1,28*28))/(28*28));cc=pool('thickness','avgpool','soft_ink',28,1,28,1)
D=[]
for j in range(3,25):
    v=np.zeros(28);v[j]=1;v[j-3]=v[j+3]=-.5;D.extend([v,-v])
linear('thickness_derivatives',cc,D);node('thickness_variation','relu','thickness_derivatives');linear('crossstrap','thickness_variation',np.ones((1,len(D)))*2/len(D))
sizes=[10,len(channels),len(forms),3,1,1,1];total=sum(sizes)
def pack(label,srcs,sizes):
    o=np.eye(sum(sizes));j=0;ins=[]
    for i,(s,m) in enumerate(zip(srcs,sizes)):
        ins.append(linear(label+'_part'+str(i),s,o[:,j:j+m]));j+=m
    return node(label,'add',ins)
feat=pack('shape_clues',['silhouettes','apertures','garment_proportions','clearance','span_y','area_fraction','crossstrap'],sizes)
allv=basis(total);shape=allv[:10]; h=dict(zip(list(channels)+list(forms)+['void','instep_void','heel_void','height','area','strap'],allv[10:]));geo=[];outs=[]
def geo_rule(z,coeff):
    geo.append(z);outs.append([coeff.get(i,0) for i in range(10)])
def interval(z,k,a,b):
    geo_rule(z-a,k);geo_rule(z-b,{i:-v for i,v in k.items()})
# Hinge terms are in correlation units. Cap gap evidence: sandals are the
# category for which several separated surfaces on a ray are expected.
interval(.5*h['instep']+.5*h['mid']-.80*(1-shape[5]),{5:.43,7:-.085,9:-.045},.08,.65)
interval(.6*h['void']+.4*h['instep_void']-1.1*(1-shape[5]),{5:.36,7:-.08,9:-.04},.04,1.1)
interval(.42-.6*h['void']-.4*h['instep_void']+.35*(shape[7]-.8),{7:.12},0,.40)
# A sneaker's upper fills most of its height. Individual sandal straps may
# reach the same height without filling the space around and under them.
interval(.52*h['height']-h['area'],{5:1.02,7:-.22},.02,.20)
interval(h['area']-.58*h['height'],{7:.68,5:-.19},0,.13)
interval(h['strap']-.05*h['height'],{5:.91,7:-.20},.024,.12)
interval(h['crotch']+.4*(shape[1]-.8),{1:.22},.08,.65)
interval(h['innerthigh']+.4*(shape[1]-.8),{1:.17},.09,.55)
interval(h['leg_opening']+.4*(shape[1]-.8),{1:.16},.12,.68)
# All shoes have a toe, instep, and rear, forcing a characteristic side view.
interval(h['foot_angle']+.9*(shape[9]-.87),{9:.15,8:-.04},.08,.59)
interval(h['foot_angle']+.9*(shape[5]-.82),{5:.088},.07,.54)
interval(h['skirt']+.8*(shape[3]-.82),{3:.18},.07,.65)
interval(.55-h['waistwidth']+.6*(h['hipwidth']-h['upperwidth'])+.75*(shape[3]-.86),{3:.32},.035,.39)
# Dresses have an unbroken hem and often a narrow bodice. Undo a spuriously
# wide-leg/outerwear reading when these independent signals agree.
interval(h['hem']*.26-.45*h['upper_sides']+.9*(shape[3]-.87),{3:.14},.10,.32)
# Tee: shoulder overhang ends well above the hem; pullover: sleeves may be
# longer than the body, with a ribbed waistband; a coat often does the reverse.
interval(h['shortsleeve']+1.5*(shape[0]-.87),{0:.20,6:.10,2:-.032,4:-.055},.10,.78)
interval(h['cuff']-.9*h['hem']+.9*(shape[2]-.87),{2:.064,4:-.037},0,.70)
interval(h['longbody']+(shape[4]-.88),{4:.062,2:-.030},.38,.94)
# Straight substantial outerwear vs. a draw-in at a knitted waistband.
interval(h['hipwidth']-.84*h['waistwidth']+.6*(shape[4]-.88),{4:.10,2:-.055},.045,.31)
interval(h['waistwidth']-h['hipwidth']+.7*(shape[2]-.87),{2:.095,4:-.035},.06,.29)
# Thin straps and a scooped neckline can indicate a tank rather than a narrow dress.
interval(h['empty_neck']*.5+h['waistwidth']*.51+.8*(shape[0]-.85),{0:.10,3:-.05},.40,.82)
src=relus('shape_rules',feat,geo);linear('geometry',src,np.transpose(outs))
node('coarse','add',['silhouettes','geometry'])
# The ankle, upper edge and sole imply a connected solid for sneakers. Penalize
# an open-style reading if its high-frequency transitions have no associated
# area of open space. Combine the ray count AND the gap area, not just either.
area=.40*h['instep_void']+.60*h['void'];jumps=.5*h['instep']+.5*h['mid']
clues=[jumps-.21-area*1.1, jumps-.21-area*1.1-.55, .32-area-1.8*(jumps-.19),.32-area-1.8*(jumps-.19)-.55]
c=relus('solid_shoe',feat,clues);d=np.zeros((10,4));d[5]=[-.45,.45,-.18,.18];d[7]=[.14,-.14,.06,-.06];linear('solid_check',c,d)
# Raw silhouette decides the family; aspect rectification, strap counts and
# center plackets act ONLY inside the relevant family. Large purse handles
# and negative space between sleeves cannot win a sandal vote this way.
linear('raw_reverse','silhouettes',-np.eye(10));node('aspect_adjustment','add',['resized_styles','raw_reverse'])
node('fine','add',['coarse','construction','aspect_adjustment','solid_check'])
# Baseline for each family. Only broad silhouette, its main axis, hem/leg
# division and handbag/shoe asymmetry can be used for between-family votes.
base=[];cb=[]
def br(e,k):base.append(e);cb.append([k.get(i,0) for i in range(10)])
def ramp(z,l,r,k):
    br(z-l,k);br(z-r,{i:-p for i,p in k.items()})
ramp(.6*h['crotch']+.4*h['innerthigh']+.4*(shape[1]-.8),.08,.65,{1:.45})
ramp(h['leg_opening']+.4*(shape[1]-.8),.12,.68,{1:.17})
ramp(.48-h['waistwidth']+.56*(h['hipwidth']-h['upperwidth'])+.67*(shape[3]-.86),.035,.37,{3:.29})
ramp(h['skirt']-.46*h['crotch']+.9*(shape[3]-.84),.08,.75,{3:.14})
ramp(h['foot_angle']+.65*(shape[9]-.85),.06,.54,{9:.11,8:-.048})
ramp(h['foot_angle']+.65*(shape[5]-.85),.04,.55,{5:.076})
ramp(1-h['high']+.8*(shape[8]-.85)-.9*h['foot_angle'],.48,1,{8:.11})
ramp(h['shortsleeve']+.8*(shape[0]-.86),.08,.75,{0:.095,6:.06})
bv=relus('family_evidence',feat,base);linear('axis_adjustments',bv,np.transpose(cb),[0,0,0,0,0,0,0,0,.008,0]);node('family_base','add',['silhouettes','axis_adjustments'])
# Replace each within-family maximum by the corresponding gross-shape score.
# The non-winning members keep their within-family ranking unchanged.
ins=['fine']
for group,ids in enumerate(([0,2,4,6],[5,7,9],[1],[3],[8])):
    suffix=str(group);gate=np.isin(np.arange(10),ids).astype(float)[:,None]
    for k,sign in [('family_base',1),('fine',-1)]:
        key=k+suffix;v=linear(key+'_rows',k,np.eye(10)[ids]);rs=node(key+'_view','reshape',v,shape=[1,1,len(ids)]);best=pool(key+'_best','maxpool',rs,1,len(ids),1,len(ids));ins.append(linear(key+'_offset',best,gate*sign))
node('scores','add',ins)

out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True)
(out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
np.savez(out/'weights.npz',**weights)
