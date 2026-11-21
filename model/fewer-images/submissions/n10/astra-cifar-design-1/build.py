"""A fixed shape-and-scene network. No inputs to the design other than this source."""
import sys, json, math
from pathlib import Path
import numpy as np

D = Path(sys.argv[1]); D.mkdir(parents=True, exist_ok=True)
F=np.float32
layers=[]; weights={}
def layer(n,op,ins,**kw):
    layers.append(dict(name=n,op=op,inputs=[ins] if isinstance(ins,str) else ins,**kw)); return n
def aff(n,src,w,b=0,conv=False,stride=1,pad=0):
    w=np.asarray(w,dtype=F); o=len(w); weights[n+'.W']=w
    weights[n+'.b']=np.broadcast_to(np.asarray(b,dtype=F),(o,)).copy()
    if conv:return layer(n,'conv',src,out=o,kernel=list(w.shape[-2:]),stride=stride,padding=pad)
    return layer(n,'dense',src,out=o)
def pool(n,src,k,s=None,how='avg'):
    if isinstance(k,int): k=[k,k]
    if s is None:s=k
    if isinstance(s,int):s=[s,s]
    return layer(n,how+'pool',src,size=k,stride=s)
def rs(n,s,sh):return layer(n,'reshape',s,shape=list(sh))
def dot1(n,s,w,b=0):return aff(n,s,np.asarray(w)[:,:,None,None],b,True)
def rel(n,s):return layer(n,'relu',s)

# Input to the shape bank. Blurred local luminance contrast is insensitive to
# gradual shading. Unsigned oriented edge energy is compared across directions.
grid=np.arange(-2,3); gy,gx=np.meshgrid(grid,grid,indexing='ij')
lum=np.array([.299,.587,.114]); g=np.exp(-(gx*gx+gy*gy)/(2*1.35**2)); g/=g.sum()
hp=np.eye(1,25,12).reshape(5,5)-g
angles=np.arange(4)*np.pi/4
fil=[]
for a in angles:
    v=(gx*np.cos(a)+gy*np.sin(a))*np.exp(-(gx*gx+gy*gy)/(2*.95**2));v=v/v[v>0].sum();fil.extend([v,-v])
# Reduce both contrast and orientation in the same 28x28 support.
fil=np.stack(fil); fullkern=fil[:,None,:,:]*lum[None,:,None,None]
edge=aff('directional_derivatives','image',fullkern,conv=True)
ehalf=pool('edge_cells',rel('signed_edges',edge),2)
ez=np.zeros((4,8));
for i in range(4):ez[i,2*i:2*i+2]=1
ori=dot1('orientations',ehalf,ez)
oz=np.eye(4)-np.ones((4,4))*.25
sel=dot1('orientation_excess',ori,oz)
sel=pool('orientation_blocks',sel,2)
sel=layer('orientation_norm','norm',sel)

# Intensity stream has the same 7 x 7 cells. Sample only valid centers;
# a broad smooth and a band-pass channel complement gradient matching.
sm=np.exp(-(gx*gx+gy*gy)/(2*.83**2));sm/=sm.sum()
lk=np.stack([sm,sm-g]); lc=aff('intensity_filters','image',lk[:,None,:,:]*lum[None,:,None,None],conv=True)
lc=layer('intensity_norm','norm',pool('intensity_blocks',lc,4))

# In addition to the narrow-band orientation residual above, total contour
# magnitude records curvature (a pair of arches, the outline around a face)
# even if the edge direction wanders inside a block.
cont=dot1('total_edge_energy',ori,[[.25,.25,.25,.25]])
cont=layer('contour_norm','norm',pool('contour_blocks',cont,2))

# Draw each object independently with filled polygons and ellipses, in local
# coordinates where the picture frame is [-1,1] on either axis. Draw at twice
# the sensor resolution and use area samples. No shared image statistics.
q=(np.arange(64)+.5)/32-1; Y,X=np.meshgrid(q,q,indexing='ij')
class Drawing:
    def __init__(self,x,y,bg): self.x=x; self.y=y; self.a=np.full_like(x,bg)
    def el(self,cx,cy,rx,ry,v,ang=0):
        u=self.x-cx;t=self.y-cy;c=np.cos(ang);s=np.sin(ang)
        m=((c*u+s*t)/rx)**2+((-s*u+c*t)/ry)**2<=1
        self.a[m]=v
    def poly(self,pts,v):
        m=np.zeros_like(self.x,dtype=bool)
        for (x,y),(xx,yy) in zip(pts,pts[1:]+pts[:1]):
            if yy!=y: m^=((self.y>y)!=(self.y>yy)) & (self.x<(xx-x)*(self.y-y)/(yy-y)+x)
        self.a[m]=v
    def rect(self,l,t,r,b,v):self.poly([(l,t),(r,t),(r,b),(l,b)],v)
    def line(self,p,q,w,v):
        x,y=p;u,t=q;dx=u-x;dy=t-y;r=max(1.e-8,math.hypot(dx,dy)); sx=w*.5*dy/r;sy=-w*.5*dx/r
        self.poly([(x+sx,y+sy),(u+sx,t+sy),(u-sx,t-sy),(x-sx,y-sy)],v)
    def path(self,pts,w,v):
        for p,q in zip(pts,pts[1:]):self.line(p,q,w,v)

# Three independently composed view families per class. j selects a pose within
# a family; it never indexes an example photograph.
def aircraft(d,view,r,b,coat):
    E,P,L=d.el,d.poly,d.line
    if view==0: # side / three quarters, nose right
        P([(-.7,.05),(-.88,-.29),(-.73,-.29),(-.49,-.02),(.54,-.06),(.76,-.015),(.90,.05),(.74,.11),(-.58,.13),(-.84,.09)],coat)
        P([(-.1,.05),(-.58,-.37),(-.44,-.40),(.22,.02)],coat*.91)
        P([(-.13,.09),(-.47,.49),(-.29,.47),(.30,.06)],coat*.78)
        E(.67,.0,.073,.027,.28);E(-.13,.23,.12,.045,coat*.73)
    elif view==1: # ventral or banked planform
        E(0,-.02,.087,.86,coat);P([(-.01,-.44),(-.94,.23),(-.91,.32),(.0,.08),(.91,.32),(.94,.23),(.03,-.44)],coat)
        P([(0,.41),(-.33,.71),(-.32,.77),(0,.66),(.32,.77),(.33,.71)],coat*.94)
        for i in [-1,1]:E(.31*i,.01,.046,.14,coat*.85)
    else: # approaching or receding, long wings and central fuselage
        E(0,.02,.14,.13,coat);P([(-.94,-.10),(0,-.015),(.94,-.10),(.90,-.035),(.10,.065),(-.10,.065),(-.90,-.035)],coat)
        P([(-.035,-.045),(0,-.31),(.03,-.28),(.035,-.04)],coat*.83)
        for i in [-1,1]:E(i*.32,.085,.065,.060,coat*.75)
def wheel(d,x,y,rx,ry):
    d.el(x,y,rx,ry,.105);d.el(x,y,.50*rx,.50*ry,.48);d.el(x,y,.22*rx,.22*ry,.32)
def car(d,v,r,b,coat):
    E,P,R=d.el,d.poly,d.rect
    if v==0:
        wagon=r.random()<.38;roof=-.40 if wagon else -.32
        P([(-.86,-.05),(-.72,-.12),(-.57 if wagon else -.36,roof),(.18,roof-.015),(.48,-.11),(.79,-.055),(.87,.04),(.84,.24),(-.84,.24)],coat)
        P([(-.64 if wagon else -.62,-.105),(-.52 if wagon else -.32,roof+.055),(-.08,roof+.045),(-.08,-.092)],.22)
        P([(-.037,roof+.04),(.15,roof+.035),(.404,-.108),(-.037,-.092)],.27)
        R(-.86,.15,.85,.23,coat*.75);wheel(d,-.52,.215,.155,.164);wheel(d,.53,.215,.155,.164)
        E(.79,.014,.066,.039,.85)
    elif v==1:
        P([(-.65,.16),(-.59,-.16),(-.38,-.45),(.37,-.45),(.60,-.15),(.66,.14),(.59,.38),(-.57,.38)],coat)
        P([(-.50,-.145),(-.33,-.377),(.32,-.377),(.49,-.145)],.22);R(-.52,.21,.52,.35,.19)
        for i in [-1,1]:E(i*.43,.06,.13,.055,.89);R(i*.50-.085,.29,i*.50+.085,.43,.12)
        R(-.15,.22,.15,.29,.72)
    else:
        P([(-.87,-.04),(-.66,-.12),(-.42,-.385),(.14,-.42),(.48,-.19),(.78,-.10),(.84,.12),(.73,.29),(.22,.36),(-.77,.20)],coat)
        P([(-.58,-.12),(-.38,-.335),(-.04,-.36),(.05,-.12)],.19);P([(.02,-.36),(.12,-.366),(.44,-.18),(.12,-.115)],.31)
        P([(.2,.24),(.78,.125),(.72,.26),(.22,.33)],.19);wheel(d,-.59,.17,.116,.15);wheel(d,.08,.275,.121,.164)
        E(.32,.094,.073,.035,.91);E(.73,.01,.075,.034,.91)
def truck(d,v,r,b,coat):
    P,R,E=d.poly,d.rect,d.el
    box=r.choice([.30,.58,.78,.86]); cab=coat
    if v==0:
        R(-.91,-.47,.21,.19,box);R(-.89,.18,.84,.27,.19)
        P([(.27,.20),(.27,-.35),(.58,-.35),(.72,-.16),(.84,-.10),(.85,.20)],cab)
        P([(.38,-.28),(.55,-.28),(.65,-.12),(.38,-.12)],.20)
        wheel(d,-.68,.255,.128,.13);wheel(d,-.42,.255,.12,.13);wheel(d,.60,.255,.131,.14)
    elif v==1:
        R(-.52,-.68,.52,.44,box);R(-.53,-.48,.53,.39,cab)
        P([(-.42,-.38),(.42,-.38),(.446,-.057),(-.45,-.057)],.20);R(-.018,-.38,.018,-.056,cab)
        R(-.31,.078,.31,.31,.23);R(-.49,.33,.49,.43,.39)
        for i in [-1,1]:E(i*.405,.19,.081,.049,.9);R(i*.36-.11,.39,i*.36+.11,.56,.11)
    else:
        P([(-.89,-.36),(.07,-.56),(.39,-.43),(.38,.075),(-.87,.21)],box)
        P([(.07,-.56),(.39,-.43),(.38,.075),(.08,.14)],box*.78)
        P([(.1,.23),(.10,-.28),(.43,-.41),(.72,-.32),(.80,-.05),(.87,.0),(.85,.245),(.48,.35)],cab)
        P([(.17,-.22),(.39,-.31),(.42,-.094),(.17,-.024)],.20)
        P([(.45,-.33),(.67,-.263),(.74,-.062),(.47,-.10)],.26);P([(.49,.17),(.83,.10),(.82,.21),(.50,.28)],.23)
        wheel(d,-.71,.21,.095,.125);wheel(d,-.48,.18,.096,.125);wheel(d,.25,.234,.119,.157)
def ship(d,v,r,b,coat):
    P,E,R,L=d.poly,d.el,d.rect,d.line
    y=.19
    if v==0: # cabin cruiser, cargo vessel or ferry
        P([(-.93,y-.06),(.87,y-.14),(.69,y+.13),(-.68,y+.15)],coat)
        tx=r.uniform(-.42,.05);R(tx,-.18,tx+.46,.13,.85);R(tx+.07,-.30,tx+.38,-.16,.85);R(tx+.08,-.25,tx+.35,-.20,.26)
        R(tx+.15,-.45,tx+.25,-.30,.33);L((.50,.08),(.50,-.32),.026,.35)
    elif v==1: # sailboat
        P([(-.79,.30),(.77,.24),(.59,.405),(-.54,.425)],coat);L((.10,.29),(.10,-.79),.025,.23)
        P([(.066,-.76),(.066,.19),(-.60,.22)],.91);P([(.15,-.60),(.67,.17),(.16,.12)],.79)
    else: # high bow or steamer in perspective
        P([(-.68,-.02),(.33,-.04),(.69,-.19),(.53,.17),(.20,.37),(-.51,.22)],coat)
        P([(-.59,-.035),(-.49,-.25),(.13,-.31),(.38,-.17),(.30,-.045)],.85)
        P([(-.34,-.25),(-.32,-.42),(.0,-.46),(.14,-.30)],.80);L((-.29,-.34),(.005,-.375),.043,.24)
        L((-.05,-.43),(-.07,-.64),.052,.34)

def bird(d,v,r,b,c):
    P,E,L=d.poly,d.el,d.line
    if v==0: # perched songbird, long tail and projecting beak
        sl=r.uniform(-.80,-.26);E(-.04,.055,.48,.285,c,sl);E(.265,-.24,.241,.243,c*.9)
        P([(-.26,.10),(-.82,.40),(-.62,.47),(-.16,.26)],c*.76);P([(.45,-.268),(.695,-.226),(.45,-.175)],.31)
        E(-.07,.035,.325,.146,.68*c,sl);E(.34,-.29,.039,.038,.055)
        for t in [-.14,.04]:L((t,.26),(t+.01,.44),.033,.27);L((t-.08,.44),(t+.09,.44),.027,.25)
    elif v==1:
        E(.0,.085,.32,.50,c,-.22);E(-.12,-.37,.27,.259,c*.87);E(-.115,.10,.14,.34,c*.7,-.29)
        E(-.22,-.40,.044,.04,.06);P([(-.34,-.38),(-.53,-.30),(-.32,-.29)],.28)
        P([(.11,.43),(.23,.78),(.33,.78),(.235,.35)],c*.63)
    else: # bird with bent wings in flight
        P([(-.93,-.39),(-.58,-.28),(-.30,-.32),(.0,.01),(.26,-.20),(.53,-.27),(.91,-.18),(.57,-.115),(.28,-.065),(.04,.22),(-.13,.20),(-.38,-.11),(-.65,-.15)],c)
        E(.01,.12,.104,.24,c);P([(-.07,.25),(-.16,.42),(.13,.40),(.08,.25)],c*.83)

def quadruped(d,sp,v,r,b,c):
    # Long-legged grazer in profile, foreshortened, or a head-and-neck portrait.
    horse=sp=='horse';P,E,L=d.poly,d.el,d.line
    if v==2:
        if horse:
            P([(-.68,.98),(-.51,.25),(-.20,-.40),(.07,-.53),(.32,-.39),(.37,.04),(.14,.19),(.06,.95)],c)
            E(.28,-.135,.205,.455,c,.35);E(.165,.225,.162,.17,c*.82)
            for t in [-.10,.065]:E(.16+t,-.565,.067,.172,c,.16)
            L((-.25,-.33),(-.46,.22),.12,c*.53);E(.255,-.28,.044,.038,.075);E(.22,.25,.042,.035,.13)
        else:
            E(0,.76,.23,.70,c);E(.0,-.14,.276,.405,c);E(0,.14,.148,.16,c*.84)
            E(-.31,-.41,.26,.106,c,-2.58);E(.31,-.41,.26,.106,c,-.56)
            for i in [-1,1]:E(.15*i,-.22,.048,.05,.075)
            E(0,.22,.088,.058,.12)
        return
    near=(v==0);sx=1 if near else .70; legw=.080 if horse else .047
    def path(pts,w,v): d.path([(x*sx,y) for x,y in pts],w,v)
    for i,(hip,foot) in enumerate([(-.39,-.54),(.18,.36),(-.48,-.43),(.26,.18)]):
        shift=r.uniform(-.10,.10);mid=hip+(-.11 if hip<0 else .015)+shift
        path([(hip,.075),(mid,.34),(foot+shift,.62)],legw*(.82 if i<2 else 1),c*(.65 if i<2 else .87))
        E((foot+shift+.01)*sx,.63,.052*sx,.036,c*.55)
    E(-.13*sx,-.065,(.555 if horse else .49)*sx,.233 if horse else .201,c)
    P([(.09*sx,-.22),(.38*sx,-.61),(.53*sx,-.60),(.45*sx,-.20),(.26*sx,.08)],c)
    if horse:
        E(.55*sx,-.50,.137*sx,.241,c,-.57);E(.65*sx,-.335,.10*sx,.095,c*.79)
        path([(.35,-.58),(.20,-.37),(.15,-.24)],.081,c*.54);path([(-.59,-.16),(-.73,.076),(-.79,.32)],.105,c*.49)
        E(.43*sx,-.708,.047*sx,.113,c,-.13)
    else:
        E(.53*sx,-.55,.20*sx,.109,c,.18);E(.36*sx,-.674,.12*sx,.065,c,.65);E(.46*sx,-.68,.055*sx,.11,c,.30)
        E(-.60*sx,-.11,.07,.103,c,-.5)
        if r.random()<.24:
            for sign in [-1,1]:path([(.43,-.63),(.40+.15*sign,-.80),(.41+.20*sign,-.89)],.025,c*.8)
    E(.528*sx,-.56,.026,.024,.09)

def pet(d,sp,v,r,b,c):
    cat=(sp=='cat'); P,E,L=d.poly,d.el,d.line; dark=.075; pale=.76*c+.21
    if v<=1: # face or sitting portrait
        s=1 if v==0 else .73; y=.0 if v==0 else -.25
        if v==1:
            E(0,.47,.42,.57,c*.91);E(-.22,.67,.10,.33,pale);E(.22,.67,.10,.33,pale)
        def ep(x,t,a,z,v,ang=0):E(s*x,y+s*t,s*a,s*z,v,ang)
        def pp(pts,v):P([(s*x,y+s*t) for x,t in pts],v)
        if cat:
            pp([(-.56,-.095),(-.55,-.69),(-.15,-.415)],c);pp([(.56,-.095),(.55,-.69),(.15,-.415)],c)
            pp([(-.475,-.55),(-.44,-.25),(-.246,-.365)],c*.71);pp([(.475,-.55),(.44,-.25),(.246,-.365)],c*.71)
            ep(0,-.01,.546,.435,c)
        else:
            ep(0,.0,.454,.502,c);ep(-.472,-.08,.174,.39,c*.56,.16);ep(.472,-.08,.174,.39,c*.56,-.16)
        sep=.235 if cat else .209
        for i in [-1,1]:
            ep(i*sep,-.095,.104 if cat else .073,.068 if cat else .080,dark,-.11*i)
            if cat:ep(i*sep,-.102,.064,.041,.55);ep(i*sep,-.102,.018,.041,.10)
            ep(i*.108,.15,.137,.117 if cat else .169,pale)
        if cat:pp([(-.075,.081),(.075,.081),(0,.15)],.23)
        else:ep(0,.125,.114,.084,dark);ep(.0,.30,.117,.026,.20)
    else:
        E(-.16,.08,.446,.226,c);E(.29,-.02,.19,.297,c,-.36);E(.414,-.27,.218,.216,c)
        if cat:
            for i in [-1,1]:P([(.41+i*.18,-.31),(.41+i*.157,-.56),(.41+i*.035,-.40)],c)
            d.path([(-.52,.04),(-.72,-.085),(-.79,-.30),(-.74,-.52)],.091,c*.88)
        else:
            E(.58,-.205,.192,.11,c);E(.332,-.28,.113,.197,c*.59);L((-.52,.038),(-.81,-.21),.107,c)
        for x in [-.42,.17]:L((x,.18),(x+.034,.48),.108,c*.9);E(x+.061,.48,.084,.045,c*.80)
        E(.49,-.29,.031,.033,.09)
def frog(d,v,r,b,c):
    E,P=d.el,d.poly;eye=.078
    if v==0:
        E(0,.12,.565,.30,c);E(0,-.064,.466,.242,c)
        for i in [-1,1]:
            E(i*.295,-.252,.143,.139,c*.93);E(i*.295,-.273,.078,.067,.76);E(i*.295,-.266,.057,.044,eye)
            E(i*.53,.227,.224,.139,c*.81,-.55*i);d.path([(i*.30,.06),(i*.40,.29),(i*.56,.36)],.083,c*1.05)
        E(0,.027,.287,.024,c*.66)
    elif v==1:
        E(-.13,.07,.46,.305,c,-.16);E(.26,-.08,.316,.213,c,-.18)
        E(.33,-.248,.112,.095,c);E(.352,-.265,.060,.055,.10);E(-.37,.18,.284,.18,c*.73,-.31)
        d.path([(.19,.054),(.17,.278),(.52,.314)],.080,c*.84);d.path([(-.42,.21),(-.64,.31),(-.27,.33)],.096,c*.9)
    else:
        E(0,.072,.298,.476,c);E(0,-.236,.348,.251,c)
        for i in [-1,1]:
            E(i*.25,-.35,.091,.097,c*.67);E(i*.255,-.367,.050,.047,.09)
            d.path([(i*.205,-.13),(i*.47,-.024),(i*.56,-.28)],.070,c)
            d.path([(i*.18,.30),(i*.46,.48),(i*.45,.14),(i*.63,.36)],.109,c*.88)
    if r.random()<.6:
        for x,t in [(-.17,-.01),(.06,.055),(-.10,.17)]:E(x,t,.054,.036,c*.69)

# Descriptor of ONE drawing, without reference to any other drawing. Spatial
# mean subtraction per channel eliminates feature offsets. Edge norms share a
# single length so a nearly absent orientation is not unduly amplified.
def template_features(img):
    a=img.reshape(32,2,32,2).mean((1,3)); win=np.lib.stride_tricks.sliding_window_view(a,(5,5))
    def blocks(z):return z.reshape(-1,7,4,7,4).mean((2,4))
    oo=blocks(np.abs(np.einsum('xyij,kij->kxy',win,fil[::2],optimize=False)))
    contour=oo.mean(axis=0);contour-=contour.mean();contour/=np.sqrt((contour**2).sum())+1e-9
    oo-=oo.mean(axis=0,keepdims=True); oo-=oo.mean(axis=(1,2),keepdims=True)
    light=blocks(np.einsum('xyij,kij->kxy',win,lk,optimize=False)); light-=light.mean(axis=(1,2),keepdims=True)
    # Unit Euclidean template vectors; the corresponding input vectors have
    # spatial RMS one rather than Euclidean length one.
    light/=np.sqrt((light**2).sum((1,2),keepdims=True))+1e-9
    oo/=np.sqrt((oo**2).sum())+1e-9
    return np.r_[.29/7*light[0].ravel(),.15/7*light[1].ravel()],.40/14*oo.ravel(),.16/7*contour.ravel()

rng=np.random.default_rng(61629); n_templates=480
Wl=[]; Wo=[]; Wc=[]
for k in range(10):
    for j in range(n_templates):
        v=j%3
        if k==0 and v==1:a=rng.uniform(-1.55,1.55)
        elif k==0:a=rng.normal(0,.19)
        elif k in (2,6):a=rng.normal(0,.24) if (v!=2 or k!=6) else rng.uniform(-2.8,2.8)
        else:a=rng.normal(0,.085)
        zoom=rng.uniform(.76,1.28)
        if k in [2,6]:zoom*=rng.uniform(.73,1.12)
        if k in [3,5] and v==0:zoom*=rng.uniform(1.00,1.43)
        if k in [4,7] and v==2: zoom*=rng.uniform(.90,1.32)
        dx,dy=rng.uniform(-.19,.19,2);xx=X-dx;yy=Y-dy
        x=(xx*np.cos(a)+yy*np.sin(a))/(zoom*rng.uniform(.85,1.15));y=(-xx*np.sin(a)+yy*np.cos(a))/(zoom*rng.uniform(.84,1.19))
        if rng.random()<.5:x=-x
        bg=rng.uniform(.30,.78)
        if k in [0,1,8,9]:coat=rng.choice([.17,.32,.60,.81,.89])
        else:coat=rng.choice([.23,.32,.44,.58,.72,.84])
        d=Drawing(x,y,bg)
        if k==0:aircraft(d,v,rng,bg,coat)
        elif k==1:car(d,v,rng,bg,coat)
        elif k==2:bird(d,v,rng,bg,coat)
        elif k in [3,5]:pet(d,'cat' if k==3 else 'dog',v,rng,bg,coat)
        elif k in [4,7]:quadruped(d,'horse' if k==7 else 'deer',v,rng,bg,coat)
        elif k==6:frog(d,v,rng,bg,coat)
        elif k==8:ship(d,v,rng,bg,coat)
        else:truck(d,v,rng,bg,coat)
        u,w,c=template_features(d.a);Wl.append(u);Wo.append(w);Wc.append(c)
sa=aff('match_contrast',lc,Wl);sb=aff('match_orientations',sel,Wo);sc=aff('match_contours',cont,Wc)
matches=layer('template_matches','add',[sa,sb,sc]);maps=rs('by_class_and_pose',matches,[10,n_templates//12,12])
pmax=pool('best_of_local_groups',maps,[1,12],how='max')
best=pool('strongest_match',pmax,[n_templates//12,1],how='max')
consensus=pool('pose_support',pmax,[n_templates//12,1])
# Keep best-match and a smaller vote from multiple independent groups.
shapevote=layer('shape_vote','add',[aff('peak_weight',best,8.5*np.eye(10)),aff('support_weight',consensus,4.0*np.eye(10))])

# LOCAL PART STREAM. The wheel center often looks lighter than its tire, with
# another lighter band outside the tire. Cats have high pointed ears, dogs
# tend to have lower dark ears and a larger dark nose, and a frog's eyes can
# protrude above the top of its head. Test pairs of cues under small affine
# changes instead of requiring the entire outline to be correct.
y14,x14=np.meshgrid(np.arange(14)+.5,np.arange(14)+.5,indexing='ij')
part_k=[]; part_bias=[]; part_layout=[]
def part_group(name,configs,make,base):
    assert len(configs)==108
    start=len(part_k)
    for cfg in configs:
        comps=make(*cfg)
        assert len(comps)==3
        for w,b in comps:part_k.append(w.ravel());part_bias.append(b)
    part_layout.append(name)
def spot(x,y,sx,sy=None,angle=0):
    if sy is None:sy=sx
    c=np.cos(angle);s=np.sin(angle);u=(x14-x)*c+(y14-y)*s;v=-(x14-x)*s+(y14-y)*c
    w=np.exp(-.5*(u*u/sx**2+v*v/sy**2));return w/max(1.e-20,w.sum())
def annulus(x,y,r,aspect=1):
    rad=np.sqrt(((x14-x)/aspect)**2+(y14-y)**2);w=np.exp(-.5*((rad-r)/(.32+.09*r))**2);return w/w.sum()
def smooth_template(w):
    w=w-w.mean();return w/(w[w>0].sum()+1e-12)
def wheel_test(cx,cy,span,r,a):
    # Two hubs and tires above a dark undercarriage. Compare all three clues.
    cues=[]
    for s in [-1,1]:
        x=cx+s*span/2;y=cy+s*a
        ring=annulus(x,y,r*.77,.90);hub=spot(x,y,r*.35);outside=annulus(x,y,r*1.36,.90)
        cues.append((.42*hub+.58*outside-ring,0))
    under=spot(cx,cy-r*.12,span*.21,.48*r);road=spot(cx,cy+r*1.24,span*.24,.53*r);paint=spot(cx,cy-r*1.58,span*.24,.71*r)
    cues.append((.53*paint+.47*road-under,.055));return cues
wheel_cfg=[(x,y,sp,r,tilt) for x in [6.15,7.0,7.85] for y in [8.1,9.35,10.6] for sp,r in [(5.3,.83),(6.8,1.04),(8.4,1.27),(4.4,.82)] for tilt in [-.5,0,.5]]
part_group('wheel_pair',wheel_cfg,wheel_test,0)

def face_test(animal,cx,cy,se,tilt,pose):
    # se=eye-to-eye separation. Compute independent eye/ear/muzzle contrasts.
    def loc(u,v):return cx+u*se,cy+v*se+tilt*u
    def sp(u,v,a,b=None,ang=0):return spot(*loc(u,v),a*se,(b if b is not None else a)*se,ang)
    eye=0
    for s in [-1,1]:
        xx=.5*s
        eye+=.5*(sp(xx,0,.27,.24)-sp(xx,0,.125,.106))
    if animal=='puppy':
        nose=sp(pose*.12,.58,.39,.30)-sp(pose*.12,.54,.18,.133)
        ear=0
        for s in [-1,1]:ear+=.5*(.72*sp(.64*s,.10,.18,.38)+.28*sp(1.34*s,.14,.17,.34)-sp(1.02*s,.12,.18,.38))
        return [(smooth_template(eye),.0),(smooth_template(nose),.0),(smooth_template(ear),.055)]
    if animal=='kitty':
        # Inner ear bordered by two bright oblique ridges; muzzle pads flank
        # the dark inverted triangle at the small cat nose.
        ear=0
        for s in [-1,1]:ear+=.5*(.48*sp(.84*s,-.63,.14,.29,-s*.34)+.52*sp(.48*s,-.61,.09,.26,s*.54)-sp(.70*s,-.60,.10,.21))
        muzzle=.5*(sp(-.23,.43,.18,.13)+sp(.23,.43,.18,.13))-sp(.04*pose,.29,.13,.09)
        return [(smooth_template(eye),0),(smooth_template(ear),.035),(smooth_template(muzzle),.030)]
    # Broad low amphibian snout below the widely separated elevated eyes.
    mouth=.48*sp(0,.35,.41,.08)+.52*sp(0,.59,.41,.09)-sp(0,.48,.42,.065)
    bump=0
    for s in [-1,1]:bump+=.5*(sp(s*.5,-.03,.19,.18)-sp(s*.5,-.036,.09,.09))
    return [(smooth_template(eye),.01),(smooth_template(mouth),.045),(smooth_template(bump),0)]
face_cfg=[(x,y,d,t,p) for x in [5.5,7.,8.5] for y in [4.05,5.58,7.1] for d,p in [(2.6,-.5),(3.5,0),(4.5,.5),(5.7,0)] for t in [-.56,0,.56]]
for ani in ['kitty','puppy','toad']:
    part_group(ani,face_cfg,lambda *v:face_test(ani,*v),0)
# Two relatively long straight legs with the background visible between
# them. Grazers have this feature; most other CIFAR-scale animals do not.
def legs(cx,cy,gap,tilt,height):
    out=[]
    for s in [-1,1]:
        x=cx+s*gap/2
        leg=spot(x,cy,.31,height,tilt*s);l=spot(x-.85,cy,.32,height,tilt*s);r=spot(x+.85,cy,.32,height,tilt*s)
        out.append((.5*(l+r)-leg,0))
    belly=spot(cx,cy-height*1.50,gap*.36,.70);between=spot(cx,cy+.10,gap*.28,.85)
    return out+[(between-belly,.065)]
leg_cfg=[(x,y,g,t,h) for x in [5.65,7.,8.35] for y in [8.4,9.7,10.7] for g,h in [(2.4,.87),(3.5,1.15),(4.6,1.28),(5.8,1.5)] for t in [-.15,0,.15]]
part_group('long_legs',leg_cfg,legs,0)
part_image=aff('part_luminance','image',sm[None,None,:,:]*lum[None,:,None,None],conv=True)
part_image=layer('part_luminance_norm','norm',pool('part_grid',part_image,2))
pt=aff('negative_part_responses',part_image,-np.array(part_k),-np.array(part_bias));pt=rs('part_conditions',pt,[len(part_layout),108,3]);pt=pool('least_part_response',pt,[1,3],how='max')
pt=rel('positive_constellations',dot1('constellation_sign',pt,-np.eye(len(part_layout))));pt=pool('localized_constellations',pt,[108,1],how='max')
# Clip positive evidence to avoid a single coincidental dark pair dominating.
pc0=aff('part_ramp_tests',pt,np.repeat(np.eye(5),2,axis=0),np.tile([-.22,-1.02],5));pc1=rel('part_ramp_activations',pc0)
pw0=np.zeros((5,10))
for i in range(5):pw0[i,2*i:2*i+2]=[1.25,-1.25]
pt=aff('part_certainty',pc1,pw0)
part_vote=aff('part_vote',pt,[[0,0,0,0,0],[1.04,0,0,0,0],[0,0,0,0,0],[0,.86,-.12,0,0],[0,0,0,0,.54],[0,-.09,.84,0,0],[0,0,0,.50,0],[0,0,0,0,.68],[0,0,0,0,0],[.57,0,0,0,0]], [0,-.16,0,-.20,-.12,-.18,-.15,-.16,0,-.09])

# SCENE AND MATERIAL STREAM. Compute soft membership in basic color families
# using clipped linear inequalities on 2x2 area pixels. The minimum of a list
# of inequalities implements their conjunction; color features are in [0,1].
photo=pool('small_rgb','image',2)
R=np.array([1.,0,0]);G=np.array([0.,1,0]);B=np.array([0.,0,1]);Yc=lum
spec={}
def ramps(name,*tests):spec[name]=tests
# test is (linear form, start, transition width)
ramps('blue',(B-R,.013,.15),(B-G,-.008,.085))
ramps('cyan',(B-R,.028,.14),(G-R,.012,.10),(G-B,-.13,.08))
ramps('green',(G-R,-.047,.11),(G-B,.034,.14))
ramps('leaf',(G-R,-.014,.080),(G-B,.028,.095))
ramps('olive',(R-B,.059,.17),(G-B,.055,.15),(G-R,-.105,.10))
ramps('rust',(R-G,.032,.13),(R-B,.064,.175))
ramps('earth',(R-G,.005,.10),(G-B,.008,.10))
ramps('tan',(R-G,.012,.091),(G-B,.011,.086),(Yc,.33,.24))
ramps('red',(R-G,.15,.21),(R-B,.14,.20))
ramps('gray',(R-G,-.14,.105),(G-R,-.14,.105),(R-B,-.16,.123),(B-R,-.16,.123),(G-B,-.13,.099),(B-G,-.13,.099))
ramps('white',(Yc,.62,.20),(G-R,-.15,.09),(B-R,-.18,.105))
ramps('dark',(-Yc,-.34,.23))
ramps('gold',(R-B,.18,.22),(G-B,.10,.16))
ramps('light',(Yc,.48,.31))
cnames=list(spec); count=len(cnames); slots=8
cw=[];cb=[]
for name,ts in spec.items():
    for i in range(slots):
        if i<len(ts):
            w,z,s=ts[i];cw.append(-w/s);cb.append(z/s)
        else:cw.append(np.zeros(3));cb.append(-1)
# Re-express -min across colors as a spatial max, preserving positions.
# Use dense on pooled cells instead of pooling along channels at full size:
# reshape Cx8x256; 8 consecutive entries in original channel order form rows.
# An explicit cap at one is supplied by the spare constant -1 above.
cc=dot1('color_halfspaces',photo,cw,cb);cc=rs('inequality_axes',cc,[count,slots,256]);cc=pool('color_intersection',cc,[slots,1],how='max');cc=rs('color_planes',cc,[count,16,16]);cc=rel('color_positive',dot1('color_sign',cc,-np.eye(count)))
cc=pool('color_grid',cc,4)

# Extra cues: straight versus irregular contours, high spatial frequencies,
# image roughness near the borders, symmetry, and strong horizontal bands.
# s16 luminance has the central 28x28 field and a stride of two like ehalf.
rough_k=[]
# Highpass capturing short-range texture and a softer, wider one.
blur=np.exp(-(gx*gx+gy*gy)/(2*.68**2));blur/=blur.sum();delta=np.eye(1,25,12).reshape(5,5)
for f in [delta-blur,sm-g]:rough_k.extend([f,-f])
rough=aff('detail_filters','image',np.array(rough_k)[:,None,:,:]*lum[None,:,None,None],conv=True)
rough=pool('detail_blocks',rel('detail_magnitude',rough),4)
rough=dot1('unsigned_detail',rough,[[1,1,0,0],[0,0,1,1]])
# Directional energy corrected for diagonal derivatives responding to the
# same cardinal edge. Relu keeps only preferential orientations.
lin=np.array([[0,0,1,0],[1,0,0,0],[0,.5,0,.5],[0,-.5,1,-.5],[1,-.5,0,-.5],[-.5,1,-.5,0],[-.5,0,-.5,1]])
cur=pool('direction_blocks',rel('line_specificity',dot1('line_measure',ori,lin)),2)
# gradients of a 4x4 and 8x8 image encode horizontal repetition and simple
# symmetry. Absolute values of left-right differences in object-centered area.
p8=pool('rgb_8',photo,2)
regions={}
for h in [4,7,8]:
    z=(np.arange(h)+.5)/h; yy,xx=np.meshgrid(z,z,indexing='ij')
    def gaussian(cx,cy,sx,sy):return np.exp(-.5*((xx-cx)**2/sx**2+(yy-cy)**2/sy**2))
    mid=gaussian(.5,.49,.252,.245); top=gaussian(.50,.04,.49,.17); bot=gaussian(.5,.95,.47,.15)
    for n,mask in dict(all=np.ones_like(xx),top=top,bot=bot,mid=mid,edges=1-mid,side=(np.abs(xx-.5)*2)**1.8,
        upmid=gaussian(.5,.31,.27,.22),lowmid=gaussian(.5,.71,.30,.18),corners=(np.abs(xx-.5)*2)**1.3*(np.abs(yy-.5)*2)**.6,
        outer=1-gaussian(.5,.50,.35,.35),band=gaussian(.5,.52,.46,.24)).items():regions[(h,n)]=mask/mask.sum()

# Named regional measurements, all prescribed directly by formulas.
class Extract:
    def __init__(self,src,h,nc):self.src=src;self.h=h;self.nc=nc;self.ws=[];self.names=[]
    def put(self,n,w):self.names.append(n);self.ws.append(np.asarray(w).ravel());return n
    def mean(self,n,c,rg,scale=1):
        u=np.zeros((self.nc,self.h,self.h));u[c]=scale*regions[self.h,rg];return self.put(n,u)
col=Extract(cc,4,count);tex=Extract(rough,7,2);lines=Extract(cur,7,7);rgb=Extract(p8,8,3)
for i,n in enumerate(cnames):
    for reg in ['all','top','bot','mid','edges','upmid','lowmid','side']:
        col.mean(n+'_'+reg,i,reg)
for i,n,s in [(0,'fine',48),(1,'coarse',32)]:
    for reg in ['all','mid','top','bot','edges','side']:tex.mean(n+'_'+reg,i,reg,s)
for i,(n,s) in enumerate([('hgrad',9),('vgrad',9),('diag',9),('hline',32),('vline',32),('slash',25),('back',25)]):
    for reg in ['mid','top','bot','all','edges']:lines.mean(n+'_'+reg,i,reg,s)
for reg in ['all','top','bot','mid','side','edges','upmid','lowmid']:
    rgb.put('lum_'+reg,lum[:,None,None]*regions[8,reg]);rgb.put('temp_'+reg,(R-B)[:,None,None]*regions[8,reg]*4)

# Structured contrast cues on 8x8 grid. Pairwise comparisons across short
# distances distinguish a flat/smooth surround from an interior full of detail.
pairw=[]; pairtags=[]
def difference(tag,m):
    pairw.extend([m.ravel(),-m.ravel()]);pairtags.extend([tag,tag])
for y in range(1,7):
    for x in range(4):
        m=np.zeros((3,8,8));m[:,y,x]=lum;m[:,y,7-x]=-lum;difference('asym',m)
for side in ['skyflat','landflat','midflat']:
    ys={'skyflat':range(2),'landflat':range(6,8),'midflat':range(2,6)}[side]
    for y in ys:
        for x in range(6):
            m=np.zeros((3,8,8));m[:,y,x]=lum;m[:,y,x+2]=-lum;difference(side,m)
for y in range(7):
    m=np.zeros((3,8,8));m[:,y,:]=lum[:,None]/8;m[:,y+1,:]=-lum[:,None]/8;difference('layering',m)
for x in range(7):
    m=np.zeros((3,8,8));m[:,:,x]=lum[:,None]/8;m[:,:,x+1]=-lum[:,None]/8;difference('columns',m)
# Is the middle of the object distinct in color from its immediate setting?
for region in ['edges','top','bot']:
    for c in [lum,(R-B)*.8,(G-.5*(R+B))*.9]:
        m=c[:,None,None]*(regions[8,'mid']-regions[8,region]);difference('contrast_'+region,m)
pr=rel('absolute_differences',aff('paired_regions',p8,pairw));pnames=list(dict.fromkeys(pairtags));pw=[]
for n in pnames:
    row=np.array([v==n for v in pairtags],dtype=F);pw.append(2*row/row.sum()*(6 if n.startswith('contrast') else 7))
pa=aff('contrast_statistics',pr,pw)
# Region measurements are concatenated logically by linearly mapping the
# branches into disjoint slots followed by an add. No examples are involved.
sets=[(ex.src,ex.names,ex.ws) for ex in [col,tex,lines,rgb]]
features=[n for _,ns,_ in sets for n in ns]+pnames
ix={n:i for i,n in enumerate(features)};nf=len(features);parts=[]
for j,(src,ns,ws) in enumerate(sets+[(pa,pnames,np.eye(len(pnames)))]):
    mat=np.zeros((nf,len(ws[0])),F)
    for n,w in zip(ns,ws):mat[ix[n]]=w
    parts.append(aff('regional_features_'+str(j),src,mat))
features_node=layer('regional_features','add',parts)

# Soft cue rules. Represent a clipped ramp f(z)=clip((z-start)/width,0,1)
# exactly with two ReLUs. For conjunctions, take the minimum of those ramps.
# This table expresses hypotheses about outdoor backgrounds, man-made lines,
# animal coats and photographic composition, not statistics of any images.
rulez={}; zer=np.zeros(nf)
def vec(**kw):
    w=zer.copy()
    for n,c in kw.items():w[ix[n]]=c
    return w
def cue(n,start,width,**kw):rulez[n]=(vec(**kw),start,width)
cue('sky',.09,.58,blue_top=1,cyan_top=.22)
cue('sea',.06,.50,blue_bot=1,cyan_bot=.34)
cue('cool',.065,.42,blue_edges=1,cyan_edges=.21)
cue('green',.085,.43,green_edges=.42,leaf_edges=.58)
cue('grass',.075,.46,green_bot=.42,leaf_bot=.58)
cue('woods',.10,.43,olive_edges=.65,leaf_edges=.35)
cue('foliage_all',.10,.48,green_all=.40,leaf_all=.60)
cue('moss',.095,.45,olive_mid=.67,green_mid=.33)
cue('brown',.06,.45,rust_mid=.67,earth_mid=.33)
cue('sorrel',.04,.32,rust_mid=1,rust_side=-.60)
cue('brown_green',.05,.34,earth_mid=.5,rust_mid=.5,earth_edges=-.30)
cue('tawny',.08,.42,tan_mid=1)
cue('warmScene',.075,.42,earth_all=.63,rust_all=.37)
cue('brilliant',.012,.13,red_mid=1)
cue('colorful',.028,.17,gold_mid=.64,red_mid=.8,gold_edges=-.25)
cue('neutral',.33,.50,gray_all=1)
cue('grayground',.32,.53,gray_bot=1)
cue('darkcoat',.13,.43,dark_mid=1)
cue('shadow',.00,.21,dark_lowmid=1,dark_bot=-.60,dark_top=-.40)
cue('whitebody',.085,.38,white_mid=1)
cue('lighttop',.065,.51,white_top=1)
cue('paleground',.14,.54,white_bot=1)
cue('palesides',.16,.55,white_side=1)
cue('highkey',.56,.22,lum_all=1)
cue('lowkey',-.47,.21,lum_all=-1)
cue('openabove',.024,.27,lum_top=1,lum_mid=-1)
cue('silhouette',.014,.16,lum_edges=1,lum_mid=-1)
cue('lightobject',.02,.17,lum_mid=1,lum_edges=-1)
cue('toplight',.02,.31,lum_top=1,lum_bot=-1)
cue('water_below',.017,.25,blue_bot=1,blue_mid=-.62)
cue('warmer_object',.06,.52,temp_mid=1,temp_side=-1)
cue('greensurround',.015,.21,leaf_edges=1,leaf_mid=-1)
cue('lowergrass',.035,.39,leaf_bot=1,leaf_top=-1)
cue('smoothsky',-.88,.68,fine_top=-.60,coarse_top=-.40)
cue('smoothground',-.89,.69,fine_bot=-.60,coarse_bot=-.40)
cue('smoothsides',-.81,.62,fine_side=-.65,coarse_side=-.35)
cue('smooth',-.94,.67,fine_all=-.62,coarse_all=-.38)
cue('textured',.72,1.14,fine_all=.65,coarse_all=.35)
cue('busyback',.62,1.05,fine_edges=.65,coarse_edges=.35)
cue('spotted',.84,1.05,fine_mid=.76,coarse_mid=.24)
cue('fur',.46,.93,fine_mid=.67,coarse_mid=.33)
cue('sharpobject',.16,1.15,coarse_mid=1,coarse_edges=-1)
cue('horiz',.13,.49,hline_all=1,vline_all=-.28,slash_all=-.19,back_all=-.19)
cue('horizMid',.17,.65,hline_mid=1,slash_mid=-.24,back_mid=-.24)
cue('rect',.13,.51,vline_mid=.59,hline_mid=.41,slash_mid=-.47,back_mid=-.47)
cue('vertical',.065,.45,vline_mid=1,slash_mid=-.33,back_mid=-.33)
cue('organic',-.29,.53,slash_mid=.70,back_mid=.70,vline_mid=-.5,hline_mid=-.5)
cue('angled',.16,.52,slash_mid=.72,back_mid=.72,hline_mid=-.38,vline_mid=-.38)
cue('quiettop',-.89,.74,skyflat=-1)
cue('quietbottom',-.93,.71,landflat=-1)
cue('layers',.055,.57,layering=1,columns=-.55)
cue('sym',-1.13,.85,asym=-1)
cue('object',.11,.52,contrast_edges=1)
cue('landmark',.16,.66,contrast_top=1)
cue('facecontrast',.04,.69,midflat=1,skyflat=-.53,landflat=-.22)
rule_names=list(rulez);rw=[];rb=[]
for w,s,t in rulez.values():rw.extend([w/t,w/t]);rb.extend([-s/t,-s/t-1])
rc=rel('soft_cue_ramps',aff('test_scene_cues',features_node,rw,rb));rw=np.zeros((len(rule_names),len(rb)))
for i in range(len(rw)):rw[i,2*i:2*i+2]=[1,-1]
rc=aff('scene_cues',rc,rw)

# Votes measured in arbitrary evidence units. A conjunction is only as strong
# as the least satisfied condition. Negation is 1 minus the soft cue value.
votes=[{} for _ in range(10)]
def V(k,terms):votes[k].update(terms)
V(0,{'cool':.59,'sky':.83,'smoothsky':.23,'smoothsides':.31,'quietbottom':.12,'palesides':.36,'silhouette':.19,'highkey':.22,'sharpobject':.26,'sky smoothground':.54,'smoothsides landmark':.40,'cool angled':.39,'smoothsides silhouette':.30,'busyback':-.41,'green':-.27,'warmScene':-.41,'rect':-.20,'sea water_below':-.43,'grass':-.14})
V(1,{'grayground':.48,'neutral':.10,'rect':.39,'horizMid':.40,'shadow':.32,'brilliant':.33,'sharpobject':.23,'grayground shadow':.40,'horizMid grayground':.29,'openabove':.13,'grayground quietbottom':.16,'lightobject':.15,'paleground':.11,'sea':-.36,'grass':-.39,'moss':-.25,'warmScene':-.13,'smoothsides sky':-.17,'busyback':-.07})
V(2,{'green':.34,'woods':.25,'smoothsides':.40,'silhouette':.22,'sharpobject':.27,'colorful':.26,'warmer_object':.11,'organic':.23,'greensurround':.25,'smoothsides sharpobject':.42,'green smoothsides':.30,'sky angled':.30,'palesides':.13,'rect':-.39,'horizMid':-.23,'grayground shadow':-.28,'sea':-.13})
V(3,{'warmScene':.38,'tawny':.13,'fur':.26,'sym':.28,'facecontrast':.21,'organic':.25,'neutral':.17,'lowkey':.16,'spotted':.17,'darkcoat':.11,'fur sym':.37,'busyback warmScene':.25,'sky':-.50,'grass':-.49,'moss':-.23,'rect':-.31,'horiz':-.19,'cool':-.12,'water_below':-.23})
V(4,{'green':.42,'woods':.49,'grass':.38,'brown':.21,'tawny':.20,'organic':.23,'greensurround':.31,'spotted':.08,'busyback':.13,'tawny woods':.42,'green brown_green':.40,'woods busyback':.22,'sky':-.36,'sea':-.46,'rect':-.36,'brilliant':-.24,'palesides':-.13,'neutral':-.13})
V(5,{'warmScene':.25,'brown':.23,'sym':.20,'fur':.26,'whitebody':.18,'darkcoat':.13,'facecontrast':.21,'organic':.24,'lightobject':.23,'object':.13,'fur facecontrast':.25,'whitebody brown':.22,'tawny':.16,'grass':.11,'sky':-.47,'sea':-.39,'rect':-.32,'horiz':-.18,'moss':-.20})
V(6,{'warmScene':.43,'moss':.49,'woods':.22,'lowkey':.29,'spotted':.34,'busyback':.26,'organic':.24,'moss spotted':.43,'lowkey warmScene':.29,'spotted warmScene':.35,'silhouette':-.12,'greensurround':-.18,'grass lowergrass':-.23,'sea':-.54,'sky':-.49,'rect':-.43,'lighttop':-.31,'palesides':-.26,'brilliant':-.14})
V(7,{'grass':.52,'lowergrass':.30,'sorrel':.29,'darkcoat':.19,'woods':.12,'silhouette':.16,'organic':.24,'warmer_object':.21,'grass brown':.43,'grass darkcoat':.28,'brown_green greensurround':.22,'grass lighttop':.19,'green':.19,'paleground':.06,'moss':-.13,'sea':-.44,'rect':-.31,'horiz':-.15})
V(8,{'sea':1.00,'water_below':.55,'sky':.30,'cool':.28,'smoothsky':.19,'quiettop':.18,'horiz':.55,'layers':.42,'toplight':.18,'horiz sea':.52,'layers quiettop':.28,'water_below lighttop':.37,'horiz smoothsky':.24,'palesides':.09,'warmScene':-.50,'grass':-.55,'woods':-.24,'busyback':-.21})
V(9,{'rect':.67,'vertical':.37,'grayground':.38,'whitebody':.25,'brilliant':.22,'horizMid':.23,'shadow':.17,'openabove':.17,'sky':.12,'rect grayground':.43,'vertical whitebody':.34,'rect shadow':.26,'facecontrast':.09,'sea':-.36,'grass':-.39,'moss':-.31,'warmScene':-.13,'smoothsides angled':-.12})
# With cues mostly in [0,1] but not all equally common, rest evidence is
# slightly lower for ships and grazers; pet scores rely less on scene color.
intercepts=np.array([-.69,-.68,-.46,-.43,-.75,-.63,-.45,-.63,-.72,-.69])
keys=sorted({n for d in votes for n,w in d.items() if w});zz=[];bb=[]
for n in keys:
    ns=n.split(); assert len(ns)<=2
    for name in [ns[0],ns[-1]]:
        a=np.zeros(len(rule_names));a[rule_names.index(name)]=-1;zz.append(a);bb.append(0)
clause=aff('negative_conditions',rc,zz,bb);clause=rs('condition_pairs',clause,[1,len(keys),2]);clause=pool('weakest_condition',clause,[1,2],how='max')
sv=aff('context_vote',clause,[[-votes[c].get(n,0) for n in keys] for c in range(10)],intercepts)
layer('scores','add',[shapevote,sv,part_vote])
(D/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n');np.savez(D/'weights.npz',**weights)
