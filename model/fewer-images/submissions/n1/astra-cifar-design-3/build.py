"""Build a hand-designed photo recognition network. No input assets or datasets."""
import sys, json, math
from pathlib import Path
import numpy as np

F = np.float32
rng = np.random.default_rng(971403)
layers, weights = [], {}

def op(name,kind,src,**settings):
    layers.append(dict(name=name,op=kind,inputs=src if isinstance(src,list) else [src],**settings))
    return name

def affine(name,src,w,b=0,conv=False,stride=1,padding=0):
    w=np.asarray(w,dtype=F); count=len(w)
    weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(np.asarray(b,dtype=F),(count,)).copy()
    if conv:
        k=w.shape[-1]; return op(name,'conv',src,out=count,kernel=[k,k],stride=stride,padding=padding)
    return op(name,'dense',src,out=count)

def rel(name,x): return op(name,'relu',x)
def shape(name,x,*dims): return op(name,'reshape',x,shape=list(dims))
def pool(name,x,sz,st=None,kind='avg'):
    sz=[sz,sz] if isinstance(sz,int) else list(sz)
    return op(name,kind+'pool',x,size=sz,stride=sz if st is None else st)

def getgrid(n):
    a=(np.arange(n)+.5)*2/n-1
    return np.meshgrid(a,a)

# A small vector drawing language. Coordinates in object space, independent
# of any dataset. Paint values are relative material lightness, not class colors.
class Paper:
    def __init__(self,x,y,background):
        self.x=x; self.y=y; self.a=np.full(x.shape,background); self.m=np.zeros(x.shape,bool)
    def paint(self,m,c): self.a[m]=np.clip(c,0,1); self.m[m]=True
    def oval(self,x,y,rx,ry,c,tilt=0):
        dx=self.x-x; dy=self.y-y; co=math.cos(tilt); si=math.sin(tilt)
        self.paint(((co*dx+si*dy)/rx)**2+((co*dy-si*dx)/ry)**2<=1,c)
    def poly(self,points,c):
        # Even-odd crossing test, including concave outlines.
        m=np.zeros_like(self.m); x1,y1=points[-1]
        for x2,y2 in points:
            if y1!=y2:
                m ^= ((y1>self.y)!=(y2>self.y)) & (self.x < x1+(self.y-y1)*((x2-x1)/(y2-y1)))
            x1,y1=x2,y2
        self.paint(m,c)
    def line(self,x1,y1,x2,y2,r,c):
        dx=x2-x1; dy=y2-y1; t=np.clip(((self.x-x1)*dx+(self.y-y1)*dy)/(dx*dx+dy*dy+1e-10),0,1)
        self.paint((self.x-x1-t*dx)**2+(self.y-y1-t*dy)**2<=r*r,c)
    def rect(self,x1,y1,x2,y2,c): self.paint((self.x>=x1)&(self.x<=x2)&(self.y>=y1)&(self.y<=y2),c)

U=rng.uniform
pick=rng.choice

def car(p,style,truck=False):
    O,P,R,L=p.oval,p.poly,p.rect,p.line
    body=U(.23,.95); glass=U(.14,.53); lower=body*U(.48,.82); tire=U(.035,.18); hub=U(.29,.73)
    if style<.78: # side and oblique views: cab, fenders, far face, visible wheels
        persp=U(.03,.37); nose=1-persp
        if truck:
            box=U(.28,.96); roof=U(-.53,-.30); split=U(.03,.36)
            P([(-.98,roof), (split,roof-.025),(split,.19),(-.98,.21)],box)
            P([(-.97,.20),(split,.17),(split,.25),(-.96,.27)],box*.60)
            P([(split+.02,-.21),(nose-.19,-.24),(nose,-.04),(nose+.03,.26),(split,.26)],body)
            P([(split+.07,-.18),(nose-.21,-.19),(nose-.06,-.055),(split+.06,-.055)],glass)
            P([(nose-.19,-.24),(.93,-.20),(1,.02),(nose,-.04)],glass*.87)
            P([(nose,-.04),(1,.02),(.995,.25),(nose,.30)],lower)
            L(nose+.02,.13,.97,.15,.018,.19)
            wheels=[(-.76,.115),(-.53,.105),(nose-.12,.13)]
        else:
            roof=U(-.38,-.21); front=U(.19,.41); rear=U(-.64,-.31); base=U(-.035,.065)
            P([(-1,base+.025),(-.89,base-.05),(rear-.14,base-.07),(rear,roof+.025),(rear+.12,roof),
               (front-.13,roof),(front+.14,base-.055),(nose-.12,base-.015),(nose,base+.075),
               (nose-.02,.245),(-.93,.26),(-1,.18)],body)
            P([(rear-.065,base-.07),(rear+.035,roof+.065),(rear+.15,roof+.035),(-.13,roof+.035),(-.13,base-.065)],glass)
            P([(-.085,roof+.035),(front-.14,roof+.035),(front+.055,base-.055),(-.075,base-.06)],glass*.93)
            P([(front-.13,roof),(front+persp*.52-.07,roof+.025),(front+persp*.65+.13,base+.015),(front+.14,base-.055)],glass*.83)
            P([(nose-.12,base-.015),(front+persp*.65+.13,base+.015),(.98,base+.055),(nose,base+.08)],min(1,body*1.11))
            P([(nose,base+.08),(.98,base+.055),(1,.195),(nose-.02,.245)],lower)
            P([(-.98,.168),(nose-.01,.16),(nose-.02,.245),(-.93,.26)],body*.79)
            L(nose+.015,.111,.95,.105,.021,.85)
            wheels=[(U(-.77,-.61),U(.11,.17)),(nose-U(.16,.29),U(.12,.174))]
        for x,rad in wheels:
            O(x,.225,rad*(1-.55*persp),rad,tire); O(x,.23,rad*.52,rad*.53,hub)
    else: # frontal view (sometimes from behind)
        top=-.67 if truck else -.49; wd=.64 if truck else .77; sill=-.035 if truck else -.08
        for x in [-wd*.76,wd*.76]: R(x-.11,.17,x+.11,.55 if truck else .43,tire)
        P([(-wd*.98,sill),(-wd*.84,top+.05),(-wd*.65,top),(wd*.65,top),(wd*.86,top+.05),(wd,sill),
           (wd,.32),(wd*.76,.39),(-wd*.77,.39),(-wd,.29)],body)
        P([(-wd*.85,sill),(-wd*.73,top+.075),(wd*.74,top+.075),(wd*.85,sill)],glass)
        P([(-wd,sill),(-wd*.87,.095),(wd*.86,.095),(wd,sill)],min(1,body*1.12))
        R(-wd*.44,.135,wd*.44,.265,.17); R(-wd*.29,.29,wd*.29,.34,.79)
        for x in [-wd*.76,wd*.76]: O(x,.155,.105,.052,.88)

def plane(p,style):
    P,O,L=p.poly,p.oval,p.line; c=U(.22,.94); sh=c*U(.53,.85)
    if style<.48: # in profile
        P([(-.99,-.03),(-.77,-.075),(-.55,-.074),(.68,-.078),(.87,-.05),(1,.018),(.93,.075),
           (.36,.107),(-.57,.072),(-.89,.02)],c)
        P([(-.81,-.02),(-.94,-.325),(-.85,-.34),(-.54,-.055)],sh)
        P([(-.22,.03),(-.61,.24),(-.53,.26),(.28,.056)],sh)
        P([(-.24,-.047),(-.38,-.20),(-.29,-.19),(.20,-.05)],sh*.91)
        O(.78,-.025,.082,.025,c*.44)
    elif style<.84: # underside or planform; x axis along fuselage
        O(.02,0,.96,.107,c)
        for s in [-1,1]:
            P([(.36,s*.06),(-.21,s*.88),(-.34,s*.90),(-.15,s*.105)],sh if s<0 else c)
            P([(-.57,s*.04),(-.89,s*.33),(-.96,s*.32),(-.85,s*.055)],sh)
            O(.015,s*.40,.146,.042,c*.77)
    else: # wingspan seen head-on
        P([(-1,-.07),(-.94,-.086),(-.12,-.019),(.12,-.019),(.94,-.086),(1,-.07),(.12,.071),(-.12,.071)],sh)
        O(0,0,.145,.172,c); O(0,-.059,.087,.039,sh*.72)
        P([(-.018,-.14),(0,-.41),(.023,-.14)],sh)
        for x in [-.39,.39]: O(x,.068,.069,.074,sh*.67)

def ship(p,style):
    O,P,R,L=p.oval,p.poly,p.rect,p.line; hull=U(.10,.55); cabin=U(.57,.95)
    if style<.57:
        deck=U(-.04,.13); keel=deck+U(.14,.29)
        P([(-1,deck),(-.63,deck+.015),(.64,deck-.018),(1,deck-.12),(.80,keel-.025),(.59,keel),(-.83,keel)],hull)
        L(-.81,keel-.023,.68,keel-.046,.019,hull*1.47)
        cx=U(-.58,.28); wid=U(.28,.77); top=deck-U(.16,.31)
        R(cx,top,cx+wid,deck,cabin); R(cx+.06,top-.058,cx+wid-.04,top,cabin*.88)
        R(cx+.04,top+.032,cx+wid-.025,top+.08,cabin*.49)
        R(cx+wid*.25,top-.16,cx+wid*.25+.085,top-.05,hull)
        L(.54,deck,.54,deck-.32,.013,hull)
    elif style<.80:
        P([(-.72,.23),(.76,.17),(.53,.38),(-.52,.405)],hull); mast=U(-.19,.10)
        L(mast,.23,mast,-.91,.018,hull)
        P([(mast-.045,-.82),(mast-.055,.13),(-.66,.18)],cabin)
        P([(mast+.05,-.72),(.66,.108),(mast+.05,.13)],cabin*U(.7,1))
    else:
        P([(-.63,.02),(0,.135),(.73,-.04),(.25,.36),(0,.46),(-.34,.38)],hull)
        P([(-.46,.03),(-.38,-.29),(.30,-.33),(.55,-.04),(0,.11)],cabin)
        P([(-.33,-.20),(-.27,-.285),(.27,-.30),(.36,-.22)],cabin*.45)
        L(-.02,-.28,-.02,-.67,.016,hull)

def bird(p,style):
    O,P,L=p.oval,p.poly,p.line; c=U(.14,.88); wing=c*U(.45,.87); beak=U(.28,.76)
    if style<.55:
        # perched in profile: diagonal egg-shaped belly, tail, small bill
        O(-.075,.075,.47,.31,c,-.51); O(.22,-.245,.222,.211,c*U(.79,1.2))
        P([(.375,-.29),(.63,-.221),(.39,-.18)],beak)
        P([(-.39,.12),(-.86,.53),(-.75,.54),(-.26,.34)],wing)
        O(-.14,.064,.326,.155,wing,-.44); O(.294,-.267,.034,.034,.07)
        for x in [-.12,.02]: L(x,.35,x-.045,.53,.014,beak)
    elif style<.77:
        O(0,.12,.34,.45,c); O(.01,-.29,.245,.25,c*.92)
        for s in [-1,1]:
            O(s*.236,.12,.102,.294,wing,s*.22); O(s*.101,-.332,.030,.032,.074)
        P([(-.062,-.260),(.075,-.263),(.014,-.165)],beak)
    else:
        O(0,.056,.125,.34,c,-.30)
        for s in [-1,1]:
            lift=U(-.48,.03); P([(s*.04,-.10),(s*.43,lift-.12),(s*1,lift-.25),
                (s*.70,lift+.04),(s*.36,.15),(s*.10,.195)],wing)
        P([(-.03,.28),(.23,.49),(.01,.48),(-.12,.39)],c)

def quadruped(p,kind,style):
    O,P,L=p.oval,p.poly,p.line; c=U(.21,.82); sh=c*U(.52,.86); far=c*.66
    horse=kind==7; deer=kind==4; cat=kind==3; dog=kind==5
    face=style < ({3:.74,5:.71,4:.20,7:.13}[kind]); sideface=style < ({3:.10,5:.13,4:.105,7:.085}[kind])
    if face:
        # Head and shoulders: pointed pinnae vs pendulous ears, eye pair,
        # nose bridge, muzzle and chest. Narrower, much longer for ungulates.
        h=.55 if cat else (.63 if dog else .74)
        w=(.54 if cat else (.48 if dog else .30))*U(.89,1.11)
        ey=U(-.235,-.106); sep=w*U(.37,.59); ec=U(.04,.18)
        O(-.09,.91,w*1.43,.64,sh,-.08); O(.01,.75,w*.80,.58,c*.90)
        hx=.07 if sideface else 0
        if cat or deer or horse or (dog and rng.random()<.28):
            for s in [-1,1]:
                bx=s*w*.73; tip=s*w*(1.17 if deer else .97); ty=-h*(1.47 if not deer else 1.15)
                if deer: tip=s*w*1.97
                P([(bx-s*w*.42,-h*.67),(tip,ty),(bx+s*w*.28,-h*.38)],c)
                P([(bx-s*w*.19,-h*.66),(tip*.96,ty+.072),(bx+s*w*.17,-h*.48)],sh)
        else:
            for s in [-1,1]: O(s*w*.93,-.055,w*.42,h*.73,sh*.78,-s*U(.10,.33))
        O(0,-.06,w,h*.80,c); O(hx,.13,w*(.88 if cat else .74),h*.66,c)
        # lighter fur on muzzle, graduated patches under the eyes
        mc=min(1,c*U(1.05,1.45)); my=.18 if cat else (.30 if dog else .42)
        if sideface:
            O(w*.60,my-.02,w*.70,h*.34,mc,-.11); O(w*1.03,my-.06,w*.15,h*.11,ec)
            O(w*.42,ey, .105*w,.056,ec)
        else:
            for s in [-1,1]:
                O(s*sep,ey,w*(.167 if cat else .14),h*.077,ec,.14*s)
                O(s*sep-.008,ey-.012,w*.05,h*.029,.69)
                O(s*w*.19,my,w*(.34 if cat else .40),h*(.20 if cat else .30),mc)
            ny=my-.058
            if cat: P([(-w*.19,ny-.025),(w*.19,ny-.025),(0,ny+.071)],sh*.63)
            elif dog: O(0,ny,w*.244,h*.14,ec)
            else:
                for s in [-1,1]: O(s*w*.33,my+.04,w*.12,.042,ec,-s*.25)
            if cat or dog: L(0,ny+.047,0,my+.10,.012,sh*.65)
        return
    # Standing or walking broadside, with partially overlapping far legs.
    long=horse or deer
    by=-.08 if long else .03; cx=-.09; bw=(.585 if long else .54)*U(.91,1.09); bh=U(.195,.26) if long else U(.18,.28)
    floor=U(.62,.85) if long else U(.38,.62); rad=(.043 if deer else .051) if long else .063
    if dog: floor*=U(.87,1.17)
    for j,(leg,col) in enumerate([(-.48,far),(.26,far),(-.50,c),(.32,c)]):
        x=leg+U(-.073,.065); lean=U(-.24,.22); knee=x+lean*.46+(-.055 if leg<0 else .015)
        base=by+bh*.34; ky=(floor+base)*.51; end=floor-(.043 if j<2 else 0)
        P([(x-rad*1.7,base),(x+rad*1.75,base),(knee+rad*.80,ky),(x+lean+rad*.56,end),
           (x+lean-rad*.69,end),(knee-rad*.88,ky)],col)
        O(x+lean+.009,end,rad*.96,rad*.48,col*.77)
    O(cx,by,bw,bh,c,-.028); O(-.43,by+.014,bw*.48,bh*.97,c)
    if long:
        hx=U(.49,.73); hy=U(-.64,-.48)
        P([(.04,by-.115),(hx-.09,hy-.072),(hx+.07,hy+.102),(.43,by+.14)],c)
        O(hx+.034,hy+.015,.180 if horse else .151,.124 if horse else .101,c,.30)
        O(hx+.173,hy+.096,.148 if horse else .106,.085 if horse else .054,c*.87,.46)
        for d in [-.075,.032]:
            P([(hx+d-.055,hy-.062),(hx+d+(.021 if horse else -.105),hy-(.244 if horse else .186)),(hx+d+.042,hy-.071)],c*.90)
        O(hx+.045,hy-.026,.023,.022,.09)
        if horse:
            L(.17,by-.16,hx-.115,hy-.038,.040,sh*.72)
            P([(-.57,by-.12),(-.696,by-.10),(-.79,.14),(-.87,.43),(-.84,.55),(-.70,.35),(-.63,.06)],sh*.66)
        else:
            O(-.626,by-.045,.112,.067,min(1,c*1.40),-.49)
            if rng.random()<.39:
                for v in [-.07,.025]:
                    L(hx+v,hy-.085,hx+v-.11,hy-.30,.014,sh); L(hx+v-.10,hy-.27,hx+v-.20,hy-.33,.012,sh); L(hx+v-.11,hy-.30,hx+v-.12,hy-.40,.012,sh)
    else:
        hx=.44; hy=U(-.31,-.16)
        O(.29,-.074,.17,.25,c,-.34); O(hx,hy,.208,.193,c)
        if cat:
            for xx in [hx-.11,hx+.08]: P([(xx-.073,hy-.091),(xx-.036,hy-.29),(xx+.084,hy-.114)],sh)
        else:
            O(hx-.12,hy+.014,.090,.153,sh,-.18); O(hx+.165,hy+.08,.16,.083,c*.90)
            O(hx+.282,hy+.060,.047,.044,.12)
        O(hx+.079,hy-.031,.026,.025,.08)
        L(-.57,by-.021,-.79,-.13,.055,c); L(-.79,-.13,-.88,-.34,.044,c)

def frog(p,style):
    O,P,L=p.oval,p.poly,p.line; c=U(.23,.75); sh=.72*c; spot=c*U(.40,.78)
    if style<.53:
        O(-.17,.093,.55,.32,c,-.23); O(.295,-.08,.354,.218,c,-.07)
        O(.345,-.243,.122,.123,c); O(.37,-.264,.067,.053,.13)
        O(-.37,.235,.285,.150,sh,-.31)
        L(-.55,.276,-.20,.39,.055,c); L(-.20,.39,-.71,.471,.031,sh)
        L(.20,.104,.22,.301,.056,sh); L(.22,.301,.54,.37,.035,c)
        L(.27,.025,.587,-.064,.015,sh)
    else:
        O(0,.13,.51,.36,c); O(0,-.031,.54,.27,c)
        for s in [-1,1]:
            O(s*.31,-.21,.146,.154,c); O(s*.324,-.235,.079,.063,.15)
            O(s*.49,.192,.19,.208,sh,-s*.45)
            L(s*.27,.132,s*.36,.379,.058,c); L(s*.36,.379,s*.64,.417,.032,c)
        L(-.34,.051,0,.113,.013,sh); L(0,.113,.34,.051,.013,sh)
    for j in range(5):
        x=U(-.35,.14); y=U(-.02,.19); O(x,y,U(.026,.065),U(.022,.046),spot)

# Paper is sampled above the feature resolution and then box integrated.
# Image and template share the same analytic descriptor, but each drawn
# template is processed/centered/normalized individually, never in groups.
baseX,baseY=getgrid(64)

def draw(k):
    style=rng.random(); sx=U(.61,1.07); sy=sx*U(.77,1.27)
    rot=U(-.11,.11)
    if k==0: rot=U(-.63,.63) if style<.48 else (U(-math.pi,math.pi) if style<.84 else rot)
    if k==2: rot=U(-.39,.39)
    if k in [3,5] and style<(.74 if k==3 else .71): sx=U(.72,1.42); sy=sx*U(.82,1.19); rot=U(-.18,.18)
    if k==6: sx=U(.76,1.35); sy=sx*U(.77,1.26); rot=U(-.58,.58)
    if k in [2,4,7]: sx*=U(.90,1.27); sy*=U(.94,1.19)
    if k in [1,9] and style>=.78: sx*=U(.75,1.36); sy*=U(1.0,1.35)
    dx=U(-.18,.18); dy=U(-.17,.17)
    x=baseX-dx; y=baseY-dy; co=np.cos(rot); si=np.sin(rot)
    X=(co*x+si*y)/sx; Y=(co*y-si*x)/sy
    if rng.integers(2): X=-X
    paper=Paper(X,Y,U(.26,.87))
    if k in [1,9]: car(paper,style,k==9)
    elif k==0: plane(paper,style)
    elif k==8: ship(paper,style)
    elif k==2: bird(paper,style)
    elif k==6: frog(paper,style)
    else: quadruped(paper,k,style)
    return paper.a.reshape(32,2,32,2).mean((1,3)), paper.m.reshape(8,8,8,8).mean((1,3))

# Common calculations on a 16-square grid. A cropped and oriented
# derivative sees no artificial black border. ReLU/box filtering yields
# signed and unsigned edge evidence; cos and sin sample four directions.
nd=4
angles=np.arange(nd)*np.pi/nd
ky,kx=np.mgrid[-1:2,-1:2]; gg=np.exp(-(kx*kx+ky*ky)/1.65)
fil=np.array([(kx*np.cos(t)+ky*np.sin(t))*gg for t in angles]); fil/=np.abs(fil).sum((1,2))[:,None,None]/2
fil=np.concatenate([fil,-fil],axis=0)
lum=np.array([.299,.587,.114]); half=pool('two_pixel_cells','image',2)
gray=affine('luminance',half,lum[None,:,None,None],conv=True)
dir0=affine('differences',gray,fil[:,None],conv=True)
edges=rel('positive_edges',dir0)
tiny=pool('luma_grid',gray,2); stdlum=op('luma_contrast','norm',tiny)
sm=pool('cell_edges',edges,2)
# Mixture of signed and unsigned magnitudes. Compress large edge responses
# with an explicit cap, limiting the leverage of an unrelated clutter edge.
ident=np.eye(2*nd); mix=ident+.52*np.roll(ident,nd,axis=1)
emix=affine('edge_polarities',sm,mix[:,:,None,None],conv=True)
cap=.21
excess=rel('above_edge_cap',affine('edge_cap_delta',emix,ident[:,:,None,None],-cap,conv=True))
neg=affine('negative_excess',excess,-ident[:,:,None,None],conv=True)
ecap=op('bounded_edges','add',[emix,neg])

def ztemplate(a):
    a=a-a.mean(); return a/(np.sqrt((a*a).sum())+1e-10)

def describe(a):
    g=a.reshape(16,2,16,2).mean((1,3)); l=g.reshape(8,2,8,2).mean((1,3))
    patches=np.lib.stride_tricks.sliding_window_view(g,(3,3)); grad=np.einsum('yxij,dij->dyx',patches,fil,optimize=False)
    e=grad.clip(0).reshape(2*nd,7,2,7,2).mean((2,4)); e=np.minimum(cap,e+.52*np.roll(e,nd,axis=0))
    # Center globally, not per orientation: a strong horizontal line really
    # is more diagnostic of a hull than a vertical line in the same place.
    return ztemplate(l).ravel(), ztemplate(e).ravel()

# Dot-products with globally centered unit descriptors. Soft denominator
# floor on the edge descriptor prevents textureless scenes from blowing up.
evec=shape('one_edge_field',ecap,1,2*nd*7,7)
enorm=op('edge_contrast','norm',evec)
NT=640
WL=np.empty((10*NT,64),F); WE=np.empty((10*NT,8*49),F); WC=np.empty_like(WL)
for k in range(10):
    for j in range(NT):
        im,mask=draw(k); l,e=describe(im); row=k*NT+j
        WL[row]=l*(.23/8); WE[row]=e*(.64/np.sqrt(8*49)); WC[row]=ztemplate(mask).ravel()*(.13/8)
ls=affine('silhouette_agreement',stdlum,WL); es=affine('contour_agreement',enorm,WE)
# Is there also a coherent color difference on the pixels occupied by
# this particular outline? Avoid guessing the material color or its sign.
# A fixed noise floor downweights near-monochrome opponent axes.
chromagrid=pool('object_colour_grid',half,2)
axes=np.array([[.8,.2,-1],[-.65,1.,-.35]])*.12
axis=op('normalized_object_colour','norm',affine('object_opponents',chromagrid,axes[:,:,None,None],conv=True))
# Embed both signs of both opponent axes. One pool implements
# max(abs(correlation_warmth), abs(correlation_foliage)).
zeros=np.zeros_like(WC); pos=np.hstack([WC,zeros]); alt=np.hstack([zeros,WC])
projections=affine('chromatic_shape_contrasts',axis,np.vstack([pos,-pos,alt,-alt]))
csim=pool('stronger_colour_axis',shape('opponent_mask_channels',projections,1,4,10*NT),(4,1),kind='max')
csim=shape('chromatic_agreement',csim,10*NT)
sim=op('joint_agreement','add',[ls,es,csim]); simgrid=shape('shape_families',sim,10,1,NT)
peaks=pool('best_in_each_group',simgrid,(1,NT//4),kind='max')
meanpeak=pool('shape_vote',peaks,(1,4)); peak=pool('best_shape',peaks,(1,4),kind='max')
# Real images with different edge densities have different overall match
# levels. Only differences between class matches are of interest.
shapevote=op('shape_consensus','add',[affine('several_matches',meanpeak,np.eye(10)*.58),affine('strongest_match',peak,np.eye(10)*.42)])

# Scene and surface properties. These are explicit fuzzy tests of physical
# cues, not parameters obtained by measuring any photos or drawings.
r=np.array([1.,0,0]); g=np.array([0.,1,0]); b=np.array([0.,0,1]); Y=lum
# name: list of affine conditions which must all hold. Interior margin
# (difference in RGB units) gives continuous rather than binary membership.
hues={
    'blue':[(b-r,.004),(b-g,-.006)],
    'leaf':[(g-b,.019),(g-.77*r-.23*b,.009)],
    'earth':[(r-b,.040),(r-g,.010),(g-b,-.008)],
    'red':[(r-g,.115),(r-b,.14)],
    'bright':[(Y,.63)],'black':[(-Y,-.24)],'lightblue':[(b-r,-.005),(b-g,-.023),(Y,.40)],
}
# Approximate inverse width of the membership transition, then clip to 1.
width={'blue':.14,'leaf':.10,'earth':.13,'red':.22,'bright':.27,'black':.19,'lightblue':.14}
A=[]; B=[]; which=[]
for i,(n,terms) in enumerate(hues.items()):
    row=[]
    for vec,t in terms:
        A.append(vec/width[n]); B.append(-t/width[n]); row.append(len(A)-1)
    which.append(row)
raw=affine('colour_tests',half,np.array(A)[:,:,None,None],B,conv=True)
# All linear operations in the fuzzy minimum are pixelwise. Padding the
# list of tests with copies is an exact no-op for a minimum.
hn=list(hues); nh=len(hn); M=len(A)
C1=np.zeros((nh,M)); C2=C1.copy(); C3=C1.copy()
for i,idx in enumerate(which):
    i1,i2,i3=(idx+[idx[-1],idx[-1]])[:3]; C1[i,i1]=1; C2[i,i2]=1; C3[i,i3]=1
# After rectification, first and second operand can be combined by a
# single affine readout: min(a,b) = a - relu(a-b).
first=rel('first_two_tests',affine('colour_min_terms',raw,np.vstack([C1,C1-C2,C3])[:,:,None,None],conv=True))
I=np.eye(nh); Z=I*0
# max(0,min(a,b)) = relu(relu(a)-relu(a-b)): for negative a,
# either inner result is nonpositive. Intersect next with the third test.
mm=np.hstack([I,-I,Z]); c3=np.hstack([Z,Z,I]);
nextm=rel('third_colour_test',affine('first_intersection',first,np.vstack([mm,mm-c3])[:,:,None,None],conv=True))
# for m>=0 this yields m - relu(m-c); otherwise both are zero since c>=0.
rawmin=affine('positive_colour_margins',nextm,np.hstack([I,-I])[:,:,None,None],conv=True)
# Clip after spatial integration in the low-level color maps, before scene pooling.
colour=op('colour_membership','add',[rawmin,affine('colour_cap_subtract',rel('colour_cap_excess',affine('colour_cap',rawmin,I[:,:,None,None],-1,conv=True)),-I[:,:,None,None],conv=True)])
colgrid=pool('colour_regions',colour,4)

# A finer set of oriented derivatives runs on the original pixel grid.
# Only two axial orientations are needed for the regional high-frequency
# measure; the half-resolution branch supplies additional diagonal tests.
fullgray=affine('pixel_light', 'image',lum[None,:,None,None],conv=True)
finef=fil[[0,2,4,6]]
fin=rel('fine_edge_halves',affine('fine_differences',fullgray,finef[:,None],conv=True))
# Collapse +/- and directions early, keeping evaluation on large batches small.
finen=affine('fine_activity',fin,np.ones((1,4,1,1))*.5,conv=True)
fgrid=pool('fine_regions',finen,5) # 6x6
absdir=affine('edge_magnitude',edges,np.hstack([np.eye(4),np.eye(4)])[:,:,None,None],conv=True)
# Score oriented, coherent lines against rough surfaces. A line persists
# along its tangent; signed average first, rectify second. Valid 3x3
# tangent filters shrink the field by one cell on each border.
straightW=np.zeros((8,8,3,3)); normal_dirs=[(1,0),(1,1),(0,1),(-1,1)]
for i,(nx,ny) in enumerate(normal_dirs):
    for sign in range(2):
        j=i+4*sign
        for h in [-1,0,1]:
            straightW[j,j,1+h*nx,1-h*ny]=1/3
            straightW[j,(j+4)%8,1+h*nx,1-h*ny]=-1/3
st=rel('coherent_sign',affine('tangent_integration',edges,straightW,conv=True))
stmag=affine('straight_activity',st,np.hstack([np.eye(4),np.eye(4)])[:,:,None,None],conv=True)
stgrid=pool('line_regions',stmag,3); cgrid=pool('coarse_regions',absdir,2)

# Each row in this table is a named scalar measurement of the CURRENT
# input. The masks are defined in normalized frame coordinates. All rules
# below are written directly in these feature units.
FEATS={}; sources={}
def reg(n,x,y):
    if n=='all': v=np.ones_like(x)
    elif n=='top': v=np.maximum(0,-y)**1.65
    elif n=='bottom': v=np.maximum(0,y)**1.35
    elif n=='center': v=np.exp(-((x/.64)**2+(y/.58)**2)*1.22)
    elif n=='border': v=1-np.exp(-((x/.70)**2+(y/.61)**2)*1.32)
    elif n=='upperouter': v=np.maximum(0,-y)*(.5+abs(x))
    elif n=='sides': v=np.abs(x)**2*(1-.45*np.abs(y))
    elif n=='waist': v=np.exp(-((y-.04)/.32)**2)*(1-.24*abs(x))
    elif n=='lowwaist': v=np.exp(-((y-.40)/.24)**2)*(1-.34*abs(x))
    return v/v.sum()
def measure(name,src,ch,h,region,channel):
    if name not in FEATS: FEATS[name]=len(FEATS)
    mats=sources.setdefault(src,{})
    x,y=getgrid(h); spatial=reg(region,x,y)
    w=np.asarray(channel).reshape(ch,1,1)*spatial
    mats[name]=mats.get(name,0)+w.ravel()

for i,n in enumerate(hn):
    for a in ['top','bottom','center','border','all']:
        measure(n+'_'+a,colgrid,nh,4,a,I[i])
for a in ['top','bottom','center','border','all','waist','lowwaist']:
    measure('fine_'+a,fgrid,1,6,a,[1])
    measure('coarse_'+a,cgrid,4,7,a,[.25]*4)
    measure('line_'+a,stgrid,4,4,a,[.25]*4)
    measure('axial_'+a,stgrid,4,4,a,[.5,-.5,.5,-.5])
    measure('horiz_'+a,stgrid,4,4,a,[0,-.5,1,-.5])
    measure('vert_'+a,stgrid,4,4,a,[1,-.5,0,-.5])
    measure('Y_'+a,tiny,1,8,a,[1])
    # Two linear opponent axes, useful in slightly desaturated photographs
    measure('cool_'+a,half,3,16,a,b-.75*r-.25*g)
    measure('ochre_'+a,half,3,16,a,r*.48+g*.52-b)
measure('midband',tiny,1,8,'waist',[-1]); measure('midband',tiny,1,8,'top',[.5]); measure('midband',tiny,1,8,'bottom',[.5])

# Spatial concentration / uniformity: average absolute deviation of blocks
# about the per-photograph mean, separately in top, bottom and center.
# 4x4 blocks at 8x8 resolution. A structured sky is often not saturated
# blue; low within-band variation catches an overcast or white background.
cs=pool('colour_blocks',half,2); nr=8
vW=np.zeros((nr*nr,nr,nr)); XX,YY=getgrid(nr)
for j in range(nr):
    for i in range(nr):
        # compare within a short horizontal strip; does not mistake a sky
        # gradient or horizon line for dense, irregular vegetation.
        mask=(np.abs(YY-YY[j,i])<.26).astype(float); mask/=mask.sum()
        vW[j*nr+i]=-mask; vW[j*nr+i,j,i]+=1
varfields=[]
for nm,v in [('light',lum),('chroma',b-r*.7-g*.3)]:
    vproj=affine('block_'+nm,cs,v[None,:,None,None],conv=True)
    q=vW.reshape(64,64); d=rel('rectified_'+nm+'_changes',affine('strip_'+nm+'_changes',vproj,np.vstack([q,-q])))
    mag=affine('absolute_'+nm+'_changes',d,np.hstack([np.eye(64)]*2)); field=shape(nm+'_variation',mag,1,8,8)
    for a in ['top','bottom','center','border','all']:
        measure('var'+nm+'_'+a,field,1,8,a,[1])

# Ring-and-shadow detector. A wheel is not just a dark dot. Reward an
# excess of such compact dark structures along a common lower axle.
# Use valid filters so that framing cannot introduce black rims.
w=Wring=[]
kg=7; yy,xx=np.mgrid[-3:4,-3:4]; rr=np.hypot(xx,yy)
for rad in [1.25,2.05]:
    # A bright core is optional; a dark tire and lighter fender above it.
    ring=np.exp(-.5*((rr-rad*.69)/(.47*rad))**2)
    surround=np.exp(-.5*((rr-1.55*rad)/(.40*rad))**2)*(1-.43*np.tanh(yy))
    w.append(surround/surround.sum()-ring/ring.sum())
wr=rel('compact_dark_parts',affine('dark_discs',gray,np.array(w)[:,None],[-.065,-.065],conv=True))
# Relative separation varies with perspective. Sample wheel candidates
# from left and right windows offset along a shared horizontal strip.
# Min of the two gives no reward if there is just one isolated dark dot.
P1=[]; P2=[]
for j in range(2):
    for y0 in [4,5,6,7,8]:
        for x1,x2 in [(0,5),(1,6),(2,7),(0,7),(1,8),(0,8),(2,8),(3,8)]:
            a=np.zeros((2,10,10)); b0=a.copy(); a[j,y0:y0+2,x1:x1+2]=.25; b0[j,y0:y0+2,x2:x2+2]=.25
            P1.append(a.ravel()); P2.append(b0.ravel())
pa=np.array(P1); pb=np.array(P2); nn=len(pa)
wd=rel('paired_wheel_terms',affine('axle_offsets',wr,np.vstack([pa,pa-pb])))
wp=affine('both_wheels',wd,np.hstack([np.eye(nn),-np.eye(nn)]))
wb=pool('visible_axle',shape('axle_positions',wp,1,1,nn),(1,nn),kind='max')
measure('wheels',wb,1,1,'all',[1])

nf=len(FEATS); contrib=[]
for src,rows in sources.items():
    width0=len(next(iter(rows.values()))); ww=np.zeros((nf,width0))
    for key,row in rows.items(): ww[FEATS[key]]=row
    contrib.append(affine('measure_'+src,src,ww))
features=op('photographic_evidence','add',contrib)

# Smooth membership functions. Every cue ranges from 0 to 1 so that the
# explicit score table below has interpretable per-cue units.
CUES={}; ca=[]; cb=[]
def cue(n,terms,lo,hi):
    if isinstance(terms,str): terms={terms:1}
    row=np.zeros(nf)
    for key,val in terms.items(): row[FEATS[key]]+=val/(hi-lo)
    CUES[n]=len(ca); ca.append(row); cb.append(-lo/(hi-lo))
cue('sky','lightblue_top',.04,.63); cue('water','blue_bottom',.045,.58); cue('coolscene','cool_all',-.020,.112)
cue('coolbase','cool_bottom',-.035,.115); cue('blueobject','blue_center',.11,.68)
cue('plants','leaf_border',.065,.56); cue('pasture','leaf_bottom',.06,.59); cue('greenabove','leaf_top',.045,.57)
cue('greenobject','leaf_center',.085,.59); cue('greencontrast',{'leaf_border':1,'leaf_center':-1},.045,.38)
cue('earthy','earth_center',.085,.65); cue('dryground','earth_bottom',.095,.70); cue('warmroom','ochre_all',.04,.22)
cue('colourfulbody','red_center',.045,.49); cue('vividspot',{'red_center':1,'red_border':-.80},.045,.30)
cue('whitebody','bright_center',.16,.66); cue('sunnytop','bright_top',.17,.72); cue('whitebase','bright_bottom',.17,.69)
cue('blackbody','black_center',.09,.44); cue('darkbase','black_bottom',.11,.56)
cue('shadedscene','Y_all',.46,.23); cue('highkey','Y_all',.52,.76)
cue('calmtop',{'fine_top':.46,'coarse_top':.54},.074,.015)
cue('calmbase',{'fine_bottom':.46,'coarse_bottom':.54},.078,.019)
cue('calmedges',{'fine_border':.46,'coarse_border':.54},.077,.020)
cue('bustop','fine_top',.045,.119); cue('busybase','fine_bottom',.052,.126)
cue('busy','fine_all',.046,.117); cue('furry',{'fine_center':1,'coarse_center':-.49},.018,.075)
cue('coarsetexture','coarse_center',.073,.18)
cue('flattop',{'varlight_top':1,'varchroma_top':.56},.134,.029)
cue('flatbase',{'varlight_bottom':1,'varchroma_bottom':.56},.125,.029)
cue('flatsides','varlight_border',.123,.040)
cue('flatcolour','varchroma_all',.044,.010)
cue('smallsubject',{'coarse_center':1,'coarse_border':-1},.024,.132)
cue('solidsubject',{'fine_border':1,'fine_center':-.90},.012,.080)
cue('axial',{'axial_all':1,'line_all':-.075},.0009,.017)
cue('horiz',{'horiz_waist':1,'line_waist':-.20},.005,.056)
cue('lowline',{'horiz_lowwaist':1,'line_lowwaist':-.17},.005,.052)
cue('verticals',{'vert_center':1,'line_center':-.16},.003,.045)
cue('rectilinear',{'axial_center':1,'line_center':-.075},.002,.022)
cue('darkstripe','midband',.054,.255)
cue('lightoverdark',{'Y_top':1,'Y_bottom':-1},.095,.41)
cue('darkoverlight',{'Y_bottom':1,'Y_top':-1},.065,.32)
cue('axle','wheels',.035,.165)
cue('warmsubject',{'earth_center':1,'earth_border':-.80},.06,.39)
cue('coolback',{'cool_border':1,'cool_center':-.82},.01,.13)
cc=np.array(ca); bb=np.array(cb); nc=len(cc); ci=np.eye(nc)
ch=rel('soft_thresholds',affine('cue_thresholds',features,np.vstack([cc,cc]),np.r_[bb,bb-1]))
cv=affine('visual_cues',ch,np.hstack([ci,-ci]))

# Soft conjunction: fires when a collection of evidence supports a scene
# interpretation. Averages here are of cues on a single input at inference,
# never of exemplars used in weight construction.
GROUPS={}; gws=[]; gbs=[]
def group(n,items,cut=.43):
    w=np.zeros(nc)
    for key,v in items.items(): w[CUES[key]]=v
    den=np.abs(w).sum(); pos=w.clip(0).sum(); w/=den
    # Treat negative cues as (1-cue).
    gws.append(w/(1-cut)); gbs.append(((den-pos)/den-cut)/(1-cut)); GROUPS[n]=len(gws)-1
group('open_sky',dict(sky=.9,calmtop=1,calmbase=1,flatsides=.7),.37)
group('blank_back',dict(calmedges=1,flatsides=1,smallsubject=1,flatcolour=.5),.40)
group('airfield',dict(sky=1,calmtop=1,flatbase=.6,lightoverdark=.6,horiz=.5),.48)
group('ocean',dict(water=1.6,calmbase=.65,flatbase=.85,lowline=.65),.35)
group('hull',dict(flatbase=1,lowline=1,horiz=.7,darkstripe=.63,calmtop=.67),.41)
group('greywater',dict(flatbase=1,coolbase=1,calmtop=.7,whitebody=.42,lowline=.54),.43)
group('roadway',dict(flatbase=1,calmbase=.72,pasture=-.90,water=-.80,axial=.82,axle=.4),.50)
group('carbody',dict(axle=1.2,lowline=.73,rectilinear=1,darkstripe=.65),.39)
group('freight',dict(verticals=1,rectilinear=.8,whitebody=.62,lowline=.75,flatbase=.4),.38)
group('grazing',dict(pasture=1.4,greencontrast=.62,earthy=.60,solidsubject=.38),.40)
group('forest',dict(plants=1.2,greenabove=.77,bustop=.66,dryground=.42),.40)
group('paddock',dict(pasture=1,sky=.62,earthy=.72,verticals=.55),.42)
group('pond',dict(greenobject=1.25,warmroom=.7,shadedscene=.45,sky=-.76,calmbase=-.40),.50)
group('forestfloor',dict(warmroom=.85,dryground=.75,busybase=.6,flattop=-.47,sky=-.72),.48)
group('perched',dict(smallsubject=1,plants=.76,calmedges=.50,rectilinear=-.8),.45)
group('plumage',dict(vividspot=.91,plants=.61,smallsubject=.67,pasture=.4),.42)
group('pet',dict(plants=-1,water=-.60,sky=-.8,horiz=-.46,furry=.63),.50)
group('fur_closeup',dict(furry=1,coarsetexture=.52,flatbase=-.5,axial=-.5),.44)
group('pet_outdoors',dict(pasture=1,earthy=.73,furry=.76,smallsubject=-.5),.40)
group('neutralstudio',dict(highkey=1,calmedges=.8,flatcolour=.6,water=-.6),.55)
gcv=rel('scene_interpretations',affine('scene_rules',cv,np.array(gws),gbs))

# Relative votes, rather than an unconditional boost for every feature a
# class uses. Midpoints below are rough semantic priors (e.g. most photos
# do not have red central paint), explicitly chosen, not data estimates.
wc=np.zeros((10,nc)); wg=np.zeros((10,len(GROUPS))); intercept=np.zeros(10)
def votes(n,prior,entries):
    m,idx=(wc,CUES[n]) if n in CUES else (wg,GROUPS[n]);
    for k,v in entries.items(): m[k,idx]+=v; intercept[k]-=prior*v
# Air, water and vegetation are strong but not mandatory scene evidence.
votes('sky',.16,{0:.80,8:.45,9:.28,7:.10,3:-.22,5:-.19,6:-.34,4:-.12})
votes('water',.12,{8:1.40,0:.52,2:.13,4:-.28,6:-.27,7:-.23,9:-.13})
votes('coolscene',.27,{0:.45,8:.47,1:.16,6:-.37,4:-.15,7:-.12,2:.07})
votes('blueobject',.16,{2:.23,0:.14,8:.18,1:.11,6:-.21})
votes('plants',.24,{4:.79,2:.55,7:.39,6:.38,5:.07,0:-.48,1:-.26,9:-.27,8:-.58,3:-.22})
votes('pasture',.24,{7:.78,4:.55,5:.20,2:.16,6:.25,0:-.28,1:-.29,8:-.47,9:-.23,3:-.17})
votes('greenabove',.21,{4:.38,2:.30,6:.24,7:.14,3:-.18,0:-.27,8:-.30})
votes('greenobject',.22,{6:.71,2:.13,4:.07,3:-.22,1:-.20,9:-.20,5:-.20,7:-.15})
votes('greencontrast',.15,{4:.43,7:.42,2:.25,5:.15,6:-.22})
votes('earthy',.34,{7:.41,4:.38,6:.33,5:.33,3:.25,8:-.25,0:-.29,1:-.12})
votes('warmroom',.36,{6:.68,4:.23,3:.21,5:.21,8:-.32,0:-.30,1:-.10})
votes('dryground',.34,{6:.48,4:.22,7:.16,3:.08,8:-.33,0:-.22})
votes('warmsubject',.20,{7:.61,4:.48,5:.31,3:.13,6:.06,2:.14})
votes('colourfulbody',.13,{1:.55,9:.39,2:.20,4:-.28,6:-.27,7:-.02})
votes('whitebody',.25,{9:.47,5:.23,3:.22,0:.15,6:-.40,4:-.23})
votes('blackbody',.19,{1:.23,7:.24,5:.11,0:-.13})
votes('shadedscene',.27,{6:.64,4:.28,3:.25,7:.05,0:-.43,8:-.37,9:-.21})
votes('highkey',.18,{0:.43,8:.36,5:.21,3:.20,9:.10,6:-.44,4:-.20})
votes('lightoverdark',.31,{0:.28,8:.29,9:.27,7:.13,3:-.22,6:-.26})
votes('darkoverlight',.22,{1:.23,3:.22,8:.09,9:-.12})
votes('smallsubject',.30,{2:.51,0:.39,8:.12,3:-.24,5:-.11,6:-.15})
votes('furry',.31,{3:.70,5:.45,6:.29,0:-.35,8:-.17,1:-.20})
votes('busy',.32,{6:.36,3:.30,4:.27,5:.14,0:-.48,8:-.29})
votes('bustop',.32,{4:.23,6:.26,1:.12,0:-.33,8:-.34})
votes('busybase',.33,{6:.34,4:.16,3:.20,1:-.29,0:-.34,8:-.20})
votes('calmedges',.35,{0:.53,2:.35,8:.26,6:-.29,4:-.13})
votes('calmbase',.35,{0:.43,1:.34,8:.24,9:.15,6:-.19,3:-.17})
votes('flattop',.34,{0:.42,8:.29,9:.16,2:.08,6:-.18})
votes('flatbase',.34,{8:.42,1:.39,0:.29,9:.19,2:-.12,5:-.13})
votes('darkstripe',.32,{1:.28,8:.27,0:.17,9:.10,3:-.22,5:-.18})
votes('darkbase',.22,{9:.21,7:.14,3:.09,8:-.11})
votes('axial',.32,{9:.71,1:.50,8:.35,0:.09,2:-.22,3:-.13,5:-.14,4:-.15,6:-.24})
votes('horiz',.33,{8:.69,1:.38,9:.41,0:.25,2:-.25,3:-.31,5:-.23,6:-.19})
votes('lowline',.32,{1:.50,8:.49,9:.42,0:-.08,3:-.27,2:-.18})
votes('verticals',.25,{9:.53,7:.25,4:.13,0:-.18,8:-.13})
votes('axle',.22,{1:.66,9:.41,7:.08,8:-.15,0:-.12})
votes('open_sky',.29,{0:1.07,8:.34,2:.18,4:-.29,6:-.33})
votes('blank_back',.22,{0:.60,2:.57,8:.17,6:-.21})
votes('airfield',.22,{0:.76,8:.28,9:.28})
votes('ocean',.19,{8:1.48,0:.13,7:-.27,4:-.27})
votes('hull',.21,{8:.99,1:.31,9:.20,3:-.18})
votes('greywater',.18,{8:.80,0:.17})
votes('roadway',.28,{1:1.15,9:.69,8:-.18})
votes('carbody',.19,{1:.94,9:.48})
votes('freight',.21,{9:1.07,1:.13})
votes('grazing',.20,{4:.89,7:.82,5:.22})
votes('forest',.23,{4:.78,2:.41,6:.34,7:-.13})
votes('paddock',.20,{7:.91,4:.29,5:.12})
votes('pond',.22,{6:.94,4:.24})
votes('forestfloor',.23,{6:.74,4:.37,3:.10})
votes('perched',.20,{2:1.04,4:.23})
votes('plumage',.17,{2:.72,7:.11})
votes('pet',.43,{3:.95,5:.74,4:-.25,7:-.19,6:-.10})
votes('fur_closeup',.24,{3:.76,5:.64,6:.19})
votes('pet_outdoors',.17,{5:.77,7:.30,4:.19})
votes('neutralstudio',.22,{5:.46,3:.40,0:.21,1:.10})

# Shape similarity is a normalized agreement rather than a probability.
# Cues share information (e.g. sky hue and low texture); discount their
# sum relative to the direct shape vote.
discount=.57
scene1=affine('colour_and_texture_votes',cv,wc*discount,intercept*discount)
scene2=affine('scene_votes',gcv,wg*discount)
cent=np.eye(10)-np.ones((10,10))/10
form=affine('form_votes',shapevote,cent*11.0)

# A third, local branch: look for eye pairs in animal portraits and
# headlights flanking a grille. Geometry shifts with face position, size
# and head tilt; unrelated isolated spots cannot satisfy a pair.
pages=[]; groupsz=112
xf,yf=getgrid(16)
for typ in ['feline','canine','ungulate','frog','headlights']:
    zrows=[]
    for j in range(groupsz):
        midx=U(-.24,.24); midy=U(-.40,.06); sep=U(.16,.47); tilt=U(-.20,.20)
        if typ=='headlights': midy=U(.0,.37); sep=U(.28,.63); tilt=U(-.11,.11)
        xx=(xf-midx)*np.cos(tilt)+(yf-midy)*np.sin(tilt); yy=(yf-midy)*np.cos(tilt)-(xf-midx)*np.sin(tilt)
        def blob(x,y,a,b):
            z=np.exp(-.5*(((xx-x)/a)**2+((yy-y)/b)**2)); return z/(z.sum()+1.e-12)
        def spot(x,y,a,b): return blob(x,y,a*2.25,b*2.18)-blob(x,y,a,b)
        # Difference-of-Gaussian atoms have a known DC rejection and are
        # normalized by their own lobes. No exemplar outputs are measured.
        if typ=='headlights':
            e=-.5*(spot(-sep,0,.083,.045)+spot(sep,0,.083,.045))
            details=.40*spot(0,.045,.48*sep,.064)+.60*(blob(0,-.19,.66*sep,.095)-blob(0,-.40,.70*sep,.10))
        else:
            er=.036+sep*.073; e=.5*(spot(-sep,0,er*(1.25 if typ=='feline' else 1),er)+spot(sep,0,er*(1.25 if typ=='feline' else 1),er))
            if typ=='feline':
                # Short nose between two lighter whisker pads, separated ears.
                details=.38*spot(0,.78*sep,.078,.055)+.31*((blob(-sep*1.19,-sep*.84,.087,.097)+blob(sep*1.19,-sep*.84,.087,.097))*.5-blob(0,-sep*.93,.10,.086))+.31*((blob(-sep*.44,sep*.91,.10,.068)+blob(sep*.44,sep*.91,.10,.068))*.5-blob(0,1.45*sep,.12,.084))
            elif typ=='canine':
                details=.65*spot(0,sep*U(1.09,1.65),sep*.37+.02,sep*.23+.02)+.35*(blob(0,.02,.14,.20)-.5*(blob(-1.78*sep,.22,.102,.191)+blob(1.78*sep,.22,.102,.191)))
            elif typ=='ungulate':
                details=.46*spot(0,sep*U(1.8,2.75),sep*.49,sep*.35)+.54*(blob(0,.62*sep,.13,.16)-.5*(blob(-1.83*sep,.72*sep,.11,.11)+blob(1.83*sep,.72*sep,.11,.11)))
            else:
                details=blob(0,.55*sep,.72*sep,.061)-blob(0,.99*sep,.72*sep,.057)
        zrows.append((.56*e+.44*details).ravel())
    pages.extend(zrows)
facebase=op('part_illumination','norm',gray)
parts=affine('facial_geometry',facebase,pages)
parts=pool('part_best_in_group',shape('part_configurations',parts,5,4,groupsz//4),(1,groupsz//4),kind='max')
parts=pool('part_family_evidence',parts,(4,1))
# Related part detectors share nuisance responses to texture. Compare
# similar hypotheses directly, with a weak absolute evidence term.
pm=np.array([[1.0,-.52,-.17,-.11,-.10],[-.50,1.,-.22,-.10,-.08],[-.22,-.29,1.,-.22,-.18],[-.21,-.22,-.20,1.,-.18],[-.12,-.10,-.09,-.12,1.]])
read=np.zeros((10,5)); read[3]=pm[0]; read[5]=pm[1]; read[4]=.4*pm[2]; read[7]=.47*pm[2]; read[6]=pm[3]*.64; read[1]=pm[4]*.8; read[9]=pm[4]*.43
partvote=affine('part_votes',parts,read*1.65,-read.sum(1)*.54*1.65)
op('class_scores','add',[form,scene1,scene2,partvote])

out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
(out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
np.savez(out/'weights.npz', **weights)
