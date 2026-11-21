"""Build a small-photo recognizer from geometric cartoons and visual rules. No input files."""
import json
import sys
from pathlib import Path
import numpy as np

F = np.float32
layers = []
weights = {}

def layer(name, op, src, **settings):
    layers.append(dict(name=name, op=op, inputs=[src] if isinstance(src,str) else src, **settings))
    return name

def affine(name, src, W, b=0, stride=1, padding=0):
    W = np.asarray(W, dtype=F)
    weights[name+'.W'] = W
    weights[name+'.b'] = np.zeros(len(W),dtype=F) + np.asarray(b,dtype=F)
    if W.ndim == 2:
        return layer(name,'dense',src,out=len(W))
    return layer(name,'conv',src,out=len(W),kernel=list(W.shape[-2:]),stride=stride,padding=padding)

def pool(name, kind, src, k, s):
    pair = lambda x: [x,x] if np.isscalar(x) else list(x)
    return layer(name,kind+'pool',src,size=pair(k),stride=pair(s))

def rect(name, src): return layer(name,'relu',src)
def reshape(name, src, shape): return layer(name,'reshape',src,shape=list(shape))

def unit(v): return v/(np.sqrt((v*v).sum())+1.e-10)

def softclip(x): return np.minimum(1,np.maximum(0,x))

# The renderer's world is [-1,1] square; x increases toward the animal's
# nose in a profile view and y toward the ground. Each ordered layer of
# paint is a material (not an observed color or photometric measurement).
class Drawing:
    def __init__(self,x,y):
        self.x,self.y=x,y
        self.mat=np.zeros(x.shape,np.int16)
    def ellipse(self,cx,cy,rx,ry,mat,angle=0):
        co,si=np.cos(angle),np.sin(angle)
        x,y=self.x-cx,self.y-cy
        self.mat[((x*co+y*si)/rx)**2+((y*co-x*si)/ry)**2<=1]=mat
    def poly(self,pts,mat):
        x,y=self.x,self.y
        inside=np.zeros(x.shape,bool)
        for (a,b),(c,d) in zip(pts,pts[1:]+pts[:1]):
            if b!=d: inside ^= ((b>y)!=(d>y)) & (x<(c-a)*(y-b)/(d-b)+a)
        self.mat[inside]=mat
    def rod(self,start,end,r1,r2,mat):
        a,b=np.array(start),np.array(end)
        v=b-a; normal=unit(np.array([-v[1],v[0]]))
        self.poly([a+r1*normal,b+r2*normal,b-r2*normal,a-r1*normal],mat)

# Material ids: 0 background; 1 main coat/paint; 2 shading/second material;
# 3 black rubber, eyes, etc.; 4 light metal/face marking; 5 glass; 6 warm detail.
def draw(c,v,x,y):
    d=Drawing(x,y); E,P,R=d.ellipse,d.poly,d.rod
    def wheels(xs,y,r):
        for x in xs:
            E(x,y,r,r,3); E(x,y,r*.43,r*.43,4)
    def eyes(cx,cy,rx,ry):
        for s in [-1,1]:
            E(s*cx,cy,rx,ry,3,-s*.12)
            E(s*cx-rx*.22,cy-ry*.35,rx*.19,ry*.18,4)
    if c==0: # wings, tailplane, stabilizer, fuselage
        if v%4==0: # silhouette in plan view
            P([(-.23,-.09),(-.55,-.8),(-.43,-.78),(.2,-.1),(.2,.1),(-.43,.78),(-.55,.8),(-.23,.09)],1)
            P([(-.75,-.04),(-.86,-.32),(-.76,-.31),(-.55,-.06),(-.55,.06),(-.76,.31),(-.86,.32),(-.75,.04)],2)
            E(0,0,.92,.095,1)
        elif v%4==1: # approaching the viewer
            P([(-.98,-.02),(-.1,-.02),(0,-.07),(.1,-.02),(.98,-.02),(.74,.07),(.09,.1),(-.09,.1),(-.74,.07)],1)
            P([(-.05,0),(0,-.38),(.04,-.32),(.05,.01)],2); E(0,.02,.13,.14,1)
            for s in [-1,1]: E(s*.37,.08,.064,.06,3)
        else: # side / three-quarter elevation
            P([(-.82,.02),(-.92,-.35),(-.82,-.34),(-.59,-.06),(.52,-.09),(.81,-.045),(.91,.022),(.79,.068),(.12,.11),(-.6,.09)],1)
            P([(-.07,.02),(-.46,.39),(-.32,.39),(.29,.06)],2)
            P([(-.12,-.04),(-.42,-.27),(-.29,-.27),(.26,-.05)],1)
            P([(-.8,.025),(-.96,.14),(-.69,.12),(-.56,.055)],2)
            E(.59,-.043,.115,.028,5)
    elif c==1: # passenger vehicles
        if v==4 or v==0: # oblique view: visible windshield, hood and flank
            P([(-.94,.13),(-.88,-.10),(-.61,-.20),(-.38,-.40),(.15,-.35),(.46,-.11),(.81,.025),(.85,.24),(.30,.38),(-.71,.19)],1)
            P([(-.39,-.35),(-.13,-.32),(.10,-.125),(-.61,-.239)],5)
            P([(-.080,-.318),(.116,-.292),(.37,-.095),(.134,-.119)],5)
            P([(.113,-.097),(.393,-.065),(.79,.045),(.294,.148),(-.20,.016)],4 if v==0 else 1)
            P([(.29,.151),(.824,.052),(.841,.23),(.303,.358)],2)
            P([(.38,.205),(.736,.144),(.729,.219),(.39,.291)],3)
            E(.35,.177,.066,.03,4,-.2);E(.756,.108,.067,.026,4,-.2)
            for x,y,r in [(-.73,.11,.123),(.12,.27,.151)]:
                E(x,y,r*.76,r,3,-.17);E(x,y,r*.34,r*.46,4,-.15)
        elif v%4 in (1,2):
            tall=.10 if v%4==2 else 0
            P([(-.92,.15),(-.88,-.015),(-.61,-.09),(-.39,-.31-tall),(.18,-.34-tall),(.51,-.1),(.84,-.043),(.92,.06),(.90,.22),(.71,.25),(-.75,.25),(-.92,.21)],1)
            P([(-.53,-.09),(-.35,-.275-tall),(-.09,-.28-tall),(-.08,-.09)],5)
            P([(-.045,-.09),(-.048,-.28-tall),(.15,-.285-tall),(.42,-.105)],5)
            R((-.72,.205),(.79,.19),.035,.035,2)
            wheels([-.58,.57],.205,.16)
            E(.83,.003,.066,.028,4)
        else:
            P([(-.65,.3),(-.7,.15),(-.66,-.05),(-.49,-.39),(-.32,-.46),(.4,-.45),(.56,-.32),(.67,-.04),(.7,.23),(.62,.32)],1)
            P([(-.51,-.12),(-.38,-.385),(.34,-.39),(.51,-.13)],5)
            P([(-.39,.12),(.39,.12),(.3,.25),(-.3,.25)],3)
            for s in [-1,1]:
                E(s*.51,.048,.121,.051,4); E(s*.51,.30,.133,.09,3)
    elif c==2: # birds
        if v%4==0: # wings outstretched, a gull or raptor
            P([(-.98,-.14),(-.66,-.28),(-.39,-.23),(0,-.02),(.39,-.23),(.66,-.28),(.98,-.14),(.62,-.15),(.29,.008),(.08,.12),(-.08,.12),(-.29,.008),(-.62,-.15)],1)
            E(0,.05,.09,.20,2)
        elif v%4==1: # upright small bird
            E(-.04,.08,.26,.42,1,-.28); E(.12,-.25,.211,.202,1)
            E(-.13,.10,.13,.3,2,-.31); P([(.29,-.28),(.53,-.22),(.28,-.19)],6)
            E(.207,-.30,.034,.036,3); P([(-.24,.25),(-.48,.64),(-.13,.44)],2)
            for s in [-1,1]: R((s*.08,.41),(s*.1,.56),.015,.014,6)
        else:
            E(-.09,.06,.41,.247,1,-.38); E(.24,-.16,.193,.188,1)
            P([(-.35,.04),(-.73,.32),(-.47,.31),(-.19,.18)],2)
            E(-.17,.04,.278,.127,2,-.48); E(.31,-.21,.027,.031,3)
            P([(.405,-.18),(.61,-.12),(.41,-.103)],6)
            for s in [-1,1]: R((s*.085,.25),(s*.095-.035,.39),.014,.015,6)
    elif c==3 or c==5: # cats / dogs; faces as well as complete animals
        if v%5 in (0,1,2):
            dog=c==5
            if dog:
                E(0,.0,.52,.56,1)
                for s in [-1,1]:
                    if v%5!=2: E(s*.48,-.035,.168,.415,2,-s*.28)
                    else: P([(s*.53,-.28),(s*.42,-.78),(s*.19,-.44)],2)
                E(0,.21,.313,.266,4); eyes(.226,-.109,.077,.066); E(0,.126,.131,.084,3)
                R((0,.19),(0,.285),.014,.017,3); E(0,.325,.153,.041,2)
            else:
                for s in [-1,1]:
                    P([(s*.53,-.12),(s*.50,-.68),(s*.15,-.38)],1)
                    P([(s*.465,-.25),(s*.455,-.56),(s*.265,-.35)],6)
                E(0,.035,.537,.438,1)
                for s in [-1,1]: E(s*.096,.191,.15,.106,4)
                eyes(.224,-.055,.108,.052); P([(-.079,.126),(.079,.126),(0,.20)],6)
                for s in [-1,1]:
                    R((s*.09,-.10),(s*.32,-.149),.026,.019,2)
        elif v%5==3: # looking at the camera and sitting
            E(.025,.20,.34,.44,1); E(0,-.24,.287,.265,1)
            for s in [-1,1]:
                if c==3: P([(s*.26,-.32),(s*.24,-.62),(s*.06,-.445)],1)
                else: E(s*.262,-.222,.091,.198,2,-s*.25)
                E(s*.125,-.262,.045,.03,3)
                E(s*.125,.33,.085,.276,4 if dog_mark(v) else 1)
            E(0,-.13,.065,.045,3)
        else:
            E(-.07,.015,.49,.22,1); E(.42,-.23,.235,.22,1)
            for xleg in [-.40,-.28,.17,.28]: R((xleg,.12),(xleg-.035,.48),.071,.047,1)
            R((-.45,.04),(-.70,-.23),.073,.04,2)
            if c==3: P([(.247,-.31),(.25,-.57),(.415,-.39)],1)
            else: E(.31,-.20,.095,.206,2,-.14); E(.60,-.17,.155,.09,1)
            E(.49,-.265,.03,.033,3)
    elif c==4 or c==7:
        horse=c==7
        if v%4==0: # front view / portrait
            E(0,.31,.32,.48,1); E(0,-.10,.22,.38,1)
            E(0,.103,.165,.186,2); eyes(.139,-.20,.037,.047)
            for s in [-1,1]:
                E(s*(.18 if horse else .265),-.446,.069 if horse else .084,.19 if horse else .21,1,s*(.25 if horse else .99))
                E(s*.09,.12,.030,.043,3)
            if horse: E(0,-.18,.052,.229,4)
        else:
            E(-.109,-.024,.51,.226 if horse else .198,1,-.04)
            R((.225,-.10),(.424,-.46),.165,.094,1)
            E(.509,-.444,.202,.089,1,.40 if horse else .12)
            E(.421,-.567,.048,.123,1,-.15 if horse else -.6)
            E(.493,-.477,.022,.023,3)
            for j,(a,b) in enumerate([(-.454,-.535),(-.35,-.218),(.141,.083),(.244,.32)]):
                xk=(a+b)/2+(-.059 if j<2 else .025); xf=b+(.125*(-1)**j if v%4==2 else 0)
                R((a,.05),(xk,.316),.073 if horse else .059,.038,1 if j%2 else 2)
                R((xk,.316),(xf,.559),.038,.024,1 if j%2 else 2)
            if horse:
                R((-.549,-.117),(-.675,.246),.087,.067,2); R((-.675,.246),(-.727,.345),.067,.025,2)
                R((.27,-.233),(.381,-.492),.033,.036,2)
            else:
                E(-.535,-.095,.099,.067,4,-.60)
                if v%4==2:
                    R((.40,-.52),(.27,-.74),.017,.014,2); R((.32,-.66),(.35,-.76),.015,.009,2)
    elif c==6:
        if v%3==0: # a frog looking toward the camera
            E(0,.075,.48,.298,1); E(0,-.117,.43,.234,1)
            for s in [-1,1]:
                E(s*.255,-.279,.126,.124,1); E(s*.25,-.303,.069,.066,4); E(s*.25,-.307,.045,.041,3)
                E(s*.45,.176,.237,.109,2,s*.46); R((s*.475,.21),(s*.593,.36),.07,.046,1)
            E(0,.013,.281,.030,2)
        else:
            E(-.1,.04,.405,.251,1,-.18); E(.237,-.099,.274,.186,1,.06)
            E(.234,-.229,.118,.117,1); E(.255,-.24,.072,.066,4); E(.259,-.239,.044,.043,3)
            E(-.277,.164,.277,.13,2,-.55); R((-.436,.22),(-.074,.331),.074,.036,1)
            R((.163,.049),(.20,.232),.069,.041,1); R((.20,.232),(.405,.27),.038,.029,1)
            R((.324,-.049),(.489,-.072),.015,.011,2)
    elif c==8:
        if v%4==0: # sailing boat
            R((-.03,-.68),(-.03,.19),.018,.02,2)
            P([(-.069,-.64),(-.078,.085),(-.49,.085)],4)
            P([(.012,-.55),(.39,.09),(.02,.066)],1)
            P([(-.63,.14),(.66,.13),(.455,.293),(-.43,.292)],2)
        elif v%4==1: # bow toward viewer
            P([(-.41,.075),(0,.14),(.41,.075),(.20,.356),(0,.44),(-.20,.356)],2)
            P([(-.31,.086),(-.23,-.287),(.22,-.287),(.31,.086)],1)
            P([(-.214,-.22),(.214,-.22),(.23,-.112),(-.23,-.112)],5)
            R((0,-.26),(0,-.48),.018,.015,2)
        else:
            P([(-.89,.065),(.899,.032),(.72,.206),(.55,.26),(-.68,.255),(-.826,.185)],2)
            P([(-.60,.064),(-.54,-.088),(-.37,-.095),(-.31,-.235),(.17,-.235),(.26,-.084),(.56,-.071),(.63,.05)],1)
            P([(-.32,-.183),(.14,-.183),(.185,-.125),(-.34,-.125)],5)
            R((-.139,-.225),(-.14,-.383),.054,.049,2); R((.3,-.065),(.28,-.35),.014,.012,2)
    elif c==9:
        if v==4: # oblique view, two planes of the tractor and trailer
            P([(-.90,-.39),(-.17,-.49),(.26,-.382),(.26,.055),(-.18,.13),(-.89,.017)],4)
            P([(-.90,-.39),(-.17,-.293),(-.17,.131),(-.89,.017)],1)
            P([(-.13,-.278),(.37,-.333),(.648,-.233),(.709,.024),(.831,.14),(.822,.332),(.37,.391),(-.122,.233)],1)
            P([(.319,-.277),(.601,-.201),(.651,-.010),(.355,-.085)],5)
            P([(.05,-.196),(.267,-.241),(.282,-.097),(.061,-.063)],5)
            P([(.40,.115),(.793,.16),(.774,.296),(.412,.302)],2)
            for x,y,r in [(-.745,.082,.084),(-.59,.104,.093),(.234,.260,.140)]:
                E(x,y,.64*r,r,3);E(x,y,.26*r,.43*r,4)
        elif v%4==0: # front of the cab
            P([(-.51,.36),(-.53,-.315),(-.418,-.57),(.405,-.57),(.515,-.342),(.525,.36)],1)
            P([(-.43,-.315),(.43,-.315),(.42,-.091),(-.42,-.091)],5)
            P([(-.34,.034),(.34,.034),(.32,.239),(-.325,.239)],2)
            for s in [-1,1]:
                E(s*.407,.16,.08,.05,4); E(s*.375,.37,.129,.101,3)
        else:
            # Tall cargo body, gap, cab and separately resolved axles.
            top=-.445 if v%4!=2 else -.245
            P([(-.928,top),(.228,top+.018),(.228,.102),(-.928,.127)],4 if v%4==1 else 1)
            P([(.267,.13),(.277,-.337),(.617,-.33),(.771,-.109),(.922,-.06),(.928,.178),(.26,.18)],1)
            P([(.446,-.277),(.594,-.275),(.712,-.119),(.444,-.117)],5)
            R((-.83,.158),(.82,.146),.069,.064,2); wheels([-.69,-.453,.638],.215,.133)
    return d.mat

def dog_mark(v): return v%2==0

# Symbolic grayscale palettes: dark or light paint/fur against lighter or
# darker surround. Glass, rubber, muzzles and alternating materials retain
# their own ordering. A bank spans both reflections, locations and sizes.
palettes=np.array([[.73,.29,.16,.055,.57,.38,.24],
                   [.36,.78,.45,.085,.88,.25,.42],
                   [.46,.52,.25,.07,.76,.29,.40]])

def shift(a,dy,dx):
    H,W=a.shape[-2:]; o=np.zeros_like(a)
    o[...,max(0,dy):min(H,H+dy),max(0,dx):min(W,W+dx)]=a[...,max(0,-dy):min(H,H-dy),max(0,-dx):min(W,W-dx)]
    return o

def edge_bank():
    # 2x2 subsampling, then derivatives on the 16x16 grid. Offset taps
    # avoid both artificial borders and opposing ends of the object.
    Y,X=np.mgrid[-1:2,-1:2]; gst=np.exp(-.5*(X*X+Y*Y)/.72**2)
    directions=np.arange(4)*np.pi/4
    kernels=[unit((X*np.cos(a)+Y*np.sin(a))*gst) for a in directions]
    gray=np.array([.26,.51,.23]); weights_=np.array([k[None,:,:]*gray[:,None,None]*sign for k in kernels for sign in [1,-1]])
    return kernels,weights_

kernels,edgeW=edge_bank()
small=pool('small','avg','image',2,2)
# Valid derivatives (no boundary artifacts) are on the inner 14x14 grid.
ed=rect('edge_halves',affine('derivatives',small,edgeW))
# Fold gradient to sign-insensitive direction channels and make them robust
# to image contrast and locally uniform colors.
fold=np.zeros((4,8,1,1))
for i in range(4): fold[i,2*i:2*i+2,0,0]=1
orient=affine('orientations',ed,fold)
po=pool('orientation_cells','avg',orient,2,2)
# Normalize across bin AND spatial indices together, preserving relative
# orientation strength. Reshaping first makes norm perform joint centering.
po=reshape('orientation_vector',po,[1,4*7,7]); po=layer('contrast_normalized','norm',po)
po=reshape('oriented_grid',po,[4,7,7])
# Two low-frequency opponent-color shape channels in addition to brightness.
coarse=pool('coarse_colors','avg','image',4,4)
opp=affine('object_channels',coarse,np.array([[.26,.51,.23],[.5,-.85,.35],[.57,.20,-.77]])[:,:,None,None])
opp=layer('object_contrast','norm',opp)

# Assemble several thousand direct matched filters. Only geometric
# primitives are rendered. Small subcell translations are marginalized.
def shape_bank():
    yy,xx=(np.indices((32,32))+.5)/16-1
    rng=np.random.default_rng(1205)
    Wg=[]; Wo=[]; Wline=[]; bb=[]
    nfamily=5; nposes=42; ncolor=3
    # Look for paint/material transitions AND their polarity at a finer
    # resolution. This is a matched high-pass contour, not a solid cutout.
    line_src=affine('midband_light',small,np.array([.26,.51,.23]).reshape(1,3,1,1))
    line_src=layer('midband_light_normalized','norm',line_src)
    # Spatial taper de-emphasizes uncertain edges at truncation boundaries.
    zy,zx=np.indices((16,16)); taper=.4+.6*np.exp(-.8*(((zx-7.5)/7.5)**4+((zy-7.5)/7.5)**4))
    for c in range(10):
        for vi in range(nfamily):
            idx=vi if c in (1,3,5,9) else [0,1,2,3,2][vi]
            for it in range(nposes):
                # Balanced positions rather than independently sampling location.
                side=1 if it%2==0 else -1
                a=(rng.uniform(-.14,.14) if c not in (0,2,6) else rng.uniform(-.43,.43))
                if c==0 and idx==0: a+=rng.uniform(-1.,1.)
                sizes=[.71,.93,1.12] if c in [0,1,8,9] else [.87,1.21,1.62]
                s=sizes[(it//2)%3]; sx=s*rng.uniform(.90,1.12); sy=s*rng.uniform(.83,1.19)
                if c in (1,9) and idx!=0: sy*=rng.uniform(1.05,1.53)
                if c==0 and idx>=2: sy*=rng.uniform(.85,1.45)
                if c in (3,5) and idx<3: sx*=rng.uniform(.94,1.32);sy*=rng.uniform(1.,1.46)
                u=xx-rng.uniform(-.23,.23); w=yy-rng.uniform(-.22,.22)
                x=side*(np.cos(a)*u+np.sin(a)*w)/sx; y=(-np.sin(a)*u+np.cos(a)*w)/sy
                mat=draw(c,idx,x,y)
                # Color-opponent mask: e.g. warm horse against green grass.
                ob=(mat!=0).reshape(8,4,8,4).mean((1,3)); ob=unit(ob-ob.mean())
                for col,gray in enumerate(palettes):
                    l=gray[mat]; im=l.reshape(16,2,16,2).mean((1,3)); low=l.reshape(8,4,8,4).mean((1,3)); low=unit(low-low.mean())
                    g=np.array([np.abs(sum(im[j:j+14,i:i+14]*k[j,i] for j in range(3) for i in range(3))) for k in kernels]); g=g.reshape(4,7,2,7,2).mean((2,4))
                    # Smooth the descriptor and contrast orientation against
                    # the mean over the bins; downweight generic blob edges.
                    g=.70*g+.075*sum(shift(g,dy,dx) for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)])
                    g=g-.57*g.mean(axis=0,keepdims=True); g=g-.8*g.mean((1,2),keepdims=True)-.2*g.mean()
                    Wg.append(unit(g).ravel()/14)
                    ext=np.pad(im,3,mode='edge'); smooth=0.
                    for dy in range(7):
                        for dx in range(7):
                            smooth+=ext[dy:dy+16,dx:dx+16]*(np.exp(-((dy-3)**2+(dx-3)**2)/5.0))
                    smooth/=np.exp(-(np.arange(-3,4)**2)/5.).sum()**2
                    diff=(im-smooth)*taper; diff-=diff.mean();Wline.append(.94*unit(diff).ravel()/16)
                    # A small fraction of the brightness term can be supplied
                    # by coat/surround chromatic contrast in particular classes.
                    warm={4:.20,7:.18,3:.07,5:.11,6:.0,2:.1}.get(c,0)
                    blue={0:.12,8:.16,6:.09}.get(c,0)
                    Wo.append(np.array([.64*low,.0*low+warm*ob,blue*ob]).ravel()/8)
                    bb.append(0.)
    N=nfamily*nposes*ncolor
    e=affine('contour_matches',po,Wg); l=affine('material_matches',opp,Wo,bb)
    hi=affine('fine_contour_matches',line_src,Wline)
    a=layer('pose_evidence','add',[e,l,hi]); a=reshape('pose_grid',a,[10,nfamily,nposes*ncolor]); a=pool('best_view','max',a,[1,nposes*ncolor],[1,nposes*ncolor]); b=pool('best_shape','max',a,[nfamily,1],[nfamily,1]); av=pool('typical_shape','avg',a,[nfamily,1],[nfamily,1]);
    # Differences, not raw totals, are meaningful to the vote.
    return affine('geometric_vote',layer('shape_support','add',[b,av]),np.eye(10)*2.1)

shapevote=shape_bank()

# A low-level scene vocabulary. Each soft predicate is the intersection of
# linear halfspaces in RGB: e.g. green exceeds red and blue for leaves,
# whereas endmember blue sky is bright AND blue, not just saturated blue.
# Compile the intersection using summed deficits followed by a hinge.
RGB=np.eye(3); r,g,b=RGB; L=np.array([.28,.49,.23]); zero=0*r
# Each item is (normal, threshold, ramp width); positive is compatible.
colors={
 'blue':[(b-r,.065,.12),(b-g,.012,.075)],
 'sky':[(b-r,.085,.14),(b-g,.022,.07),(L,.48,.21)],
 'white':[(r,.73,.21),(g,.73,.21),(b,.73,.21)],
 'pale':[(L,.65,.25),(b-r,-.018,.10),(g-r,-.015,.09)],
 'bluegray':[(b-r,.006,.10),(g-r,-.006,.07),(b-g,-.018,.065),(L,.17,.16)],
 'leaves':[(g-r,.028,.071),(g-b,.064,.10)],
 'grass':[(g-r,.001,.073),(g-b,.105,.12),(L,.26,.15)],
 'earth':[(r-b,.099,.15),(g-b,.067,.103),(r-g,-.029,.085)],
 'brown':[(r-g,.065,.107),(g-b,.045,.10),(r-b,.13,.185)],
 'gold':[(r-g,.055,.105),(r-b,.16,.19),(L,.36,.19)],
 'grey':[(r-b,-.032,.11),(b-r,-.032,.11),(g-r,-.03,.09),(r-g,-.03,.09)],
 'dark': [(-L,-.19,.18)],
 'red': [(r-g,.21,.20),(r-b,.22,.22)],
 'marsh':[(g-b,.10,.10),(r-g,-.041,.09),(g-r,-.045,.10),(-L,-.48,.25)],
 'black': [(-L,-.105,.12)],
}
# want 1 when each q exceeds threshold, zero after summed shortfalls >=1
names=list(colors); hw=[]; hb=[]; out=np.zeros((len(names),sum(map(len,colors.values())),1,1)); j=0
for i,cond in enumerate(colors.values()):
    for normal,th,delta in cond:
        hw.append(-normal/delta); hb.append(th/delta); out[i,j,0,0]=-1; j+=1
hc=rect('color_deficits',affine('color_halfspaces',small,np.array(hw)[:,:,None,None],hb)); color=rect('color_membership',affine('color_conjunctions',hc,out,1)); cp=pool('color_cells','avg',color,2,2)

# Spatial rule compiler: quantities are linear forms on one feature branch.
# Logical rules can refer to quantities from multiple branches via an add.
forms={}
def form(name,src,w,bias=0): forms[name]=(src,np.asarray(w).ravel(),bias)

def spatial(H,kind):
    y,x=(np.indices((H,H))+.5)/H; a=abs(x-.50); v=abs(y-.5)
    if kind=='all': mask=x*0+1
    elif kind=='sky': mask=(y<.30)*(1+.6*(a>.25))
    elif kind=='corners': mask=(a>.25)*(y<.36)
    elif kind=='base': mask=(y>.76)
    elif kind=='floor': mask=(y>.70)*(1+.55*(a>.23))
    elif kind=='center': mask=np.exp(-.5*(((x-.5)/.224)**2+((y-.49)/.231)**2))
    elif kind=='core': mask=np.exp(-.5*(((x-.5)/.177)**2+((y-.47)/.191)**2))
    elif kind=='surround': mask=(a>.30)|(v>.34)
    elif kind=='sides': mask=(a>.27)*(1+.45*(y>.43))
    elif kind=='lowerbody': mask=np.exp(-.5*(((x-.5)/.29)**2+((y-.681)/.112)**2))
    elif kind=='roof': mask=(y<.34)
    elif kind=='middle': mask=(abs(y-.50)<.16)
    else: raise KeyError(kind)
    return mask/mask.sum()

for s in ['all','sky','base','center','core','surround','floor','lowerbody','sides','corners','middle']:
    for i,n in enumerate(names):
        arr=np.zeros((len(names),8,8)); arr[i]=spatial(8,s); form(s+'_'+n,cp,arr)
    for i,(n,vec) in enumerate([('luma',L),('warm',r-b),('veg',g-(.66*r+.34*b))]):
        mask=spatial(8,s); form(s+'_'+n,coarse,vec[:,None,None]*mask)
# Intra-image contrast and relative brightness obtain low weighting, as a
# subject may be light OR dark. The lower bodywork shadow is an exception.
for s in ['base','center','core','sky','lowerbody','middle','surround']:
    t=np.zeros((3,8,8));t[0]=spatial(8,s);form(s+'_brightness',opp,t)

# Locally pooled absolute gradient strength. In natural texture derivative
# directions are roughly balanced. Horizontal manmade edges produce a
# surplus in the y derivative, vertical edges a surplus in the x derivative.
for s in ['all','center','sky','corners','base','sides','lowerbody','surround','middle']:
    ma=spatial(7,s)
    for name,dire in [('detail',[.25,.25,.25,.25]),('horizontal',[-.45,-.275,1.,-.275]),('vertical',[1.,-.275,-.45,-.275]),('axis',[.5,-.5,.5,-.5])]:
        form(s+'_'+name,po if name!='detail' else 'orientation_cells',np.array(dire)[:,None,None]*ma)

# Fine texture versus extended straight edges. Separately average signed
# responses along a line and rectify: the line loses support whenever the
# polarity changes. At full resolution also measure Laplacian/detail.
k5=np.arange(-2,3); gauss=np.exp(-k5**2/2);gauss/=gauss.sum(); der=-k5*gauss;der/=np.abs(der).sum()/2
Y,X=np.mgrid[-2:3,-2:3]; blur=np.exp(-(X*X+Y*Y)/(2*.94**2));blur/=blur.sum(); hp=-blur;hp[2,2]+=1
fs=[np.outer(der,np.ones(5)/5),np.outer(np.ones(5)/5,der),hp*2.5]
fine=rect('fine_halfwaves',affine('fine_detail','image',np.array([sg*k*L[:,None,None] for k in fs for sg in [1,-1]]),stride=2)); fine=pool('fine_cells','avg',fine,2,2)
for s in ['all','center','core','surround','sky','base','lowerbody','middle','sides','corners']:
    m=spatial(7,s)
    for i,n in enumerate(['hline','vline','grain']):
        w=np.zeros((6,7,7));w[2*i:2*i+2]=m;form(s+'_'+n,fine,w)

# Contrast patches for candidate tires and animal facial landmarks. These
# detectors use locally normalized luminance; combining several landmarks
# is much more specific than finding a single isolated black spot.
light=affine('luminance',small,L.reshape(1,3,1,1));light=layer('normalized_light','norm',light)

def local_gauss(X,Y,x,y,wx,wy):
    q=np.exp(-.5*(((X-x)/wx)**2+((Y-y)/wy)**2));return q/q.sum()

def landmark_bank():
    Y,X=np.indices((16,16)); W=[]; counts=[]; which=[]
    def gaussian(x,y,rx,ry):return local_gauss(X,Y,x,y,rx,ry)
    # Define contrast as a small marked patch versus its broader surround.
    def spot(x,y,r,s,outer=2.1):return gaussian(x,y,r*outer,s*outer)-gaussian(x,y,r,s)
    for name,grid in [('tires',[(x,y,dist,tilt,rad) for x in [-.9,0,.9] for y in [9.,10.2,11.4,12.6] for dist in [5.,7.,9.,11.] for tilt in [-.12,0,.12] for rad in [.65,1.05]]),('face',[(x,y,dist,tilt,rad) for x in [-1.4,0,1.4] for y in [4.6,6.4,8.2] for dist in [2.8,4.6,6.3] for tilt in [-.12,.12] for rad in [.50,.85]]),('nose',[(x,y,dist,tilt,rad) for x in [-1.4,0,1.4] for y in [4.4,6.2,8.] for dist in [3.9,5.7,7.2] for tilt in [-.12,.12] for rad in [.53,.90]])]:
        counts.append(len(grid)); which.append(name)
        for x,y,dist,tilt,rad in grid:
            mid=7.5+x; dx=dist/2;dy=dx*tilt
            eyes=.5*(spot(mid-dx,y-dy,rad,rad*.87)+spot(mid+dx,y+dy,rad,rad*.87))
            if name=='tires':
                rim=.5*(spot(mid-dx,y-dy,rad*1.25,rad)+spot(mid+dx,y+dy,rad*1.25,rad));w=.66*rim+.16*(gaussian(mid,y+1.65,dx*1.25,.85)-gaussian(mid,y-.2,dx*1.2,.68))+.18*(gaussian(mid,y-1.8,dx,.85)-gaussian(mid,y-.4,dx,.85))
            elif name=='nose': w=.57*eyes+.43*spot(mid-.62*dist*tilt,y+.62*dist,rad*1.33,rad,2.3)
            else: w=.82*eyes+.18*spot(mid-.31*dist*tilt,y+.31*dist,rad*.8,rad*.75)
            W.append(w.ravel())
    # Preserve each group's maxima without adding huge routing matrices.
    start=0
    for n,num in zip(which,counts):
        a=affine(n+'_response',light,W[start:start+num]);start+=num
        a=reshape(n+'_poses',a,[1,1,num]);a=pool(n+'_best','max',a,[1,num],[1,num]);form(n,a,[1])
landmark_bank()

# Rule language. A feature represents soft membership, with linear ramp
# between the two specified semantic bounds. Logical conjunctions use a
# clipped sum. All coefficients / bounds are judgments made by hand.
feature_specs={}
def cue(name,lo,hi,**terms): feature_specs[name]=(terms,lo,hi)
cue('open_sky',.12,.76,sky_sky=1)
cue('blue_scene',.06,.60,all_blue=1)
cue('cool_top',.29,.86,sky_pale=.66,sky_bluegray=.34)
cue('light_top',.39,.91,sky_pale=1)
cue('blue_floor',.18,.85,base_bluegray=.72,base_blue=.28)
cue('blue_surround',.14,.78,surround_blue=.55,surround_bluegray=.45)
cue('cool_floor',-.027,.062,base_warm=-1)
cue('foliage',.10,.68,surround_leaves=.70,surround_grass=.30)
cue('pasture',.13,.68,floor_grass=.75,floor_leaves=.25)
cue('leafy_top',.09,.67,sky_leaves=.76,sky_grass=.24)
cue('warm_subject',.10,.61,core_brown=.75,core_gold=.25)
cue('gold_subject',.12,.65,core_gold=1)
cue('warm_indoor',.09,.63,surround_brown=.63,surround_earth=.37)
cue('mud',.19,.67,all_earth=.55,all_marsh=.45)
cue('olive',.08,.48,center_marsh=1)
cue('green_subject',.06,.52,core_leaves=.73,core_grass=.27)
cue('colorful',.012,.23,core_red=1)
cue('neutral_road',.46,.91,floor_grey=1)
cue('neutral_body',.37,.88,center_grey=1)
cue('white_box',.08,.46,center_white=1)
cue('dark_chassis',.045,.48,lowerbody_dark=1)
cue('shadow_line',.05,.91,lowerbody_brightness=-1,base_brightness=.66,middle_brightness=.34)
cue('bright_above',.15,1.48,sky_brightness=1,center_brightness=-1)
cue('uniform_light',.37,.86,all_pale=1)
cue('cool_scene',.15,.67,all_bluegray=.7,all_blue=.3)
cue('bright_scene',.44,.75,all_luma=1)
cue('dark_scene',.49,.26,all_luma=1)
cue('nature_palette',.055,.22,all_warm=1)
cue('grass_contrast',.00,.12,core_veg=-1,surround_veg=1)
cue('fine_bg',.036,.125,surround_grain=1)
cue('coarse_bg',.040,.156,surround_detail=1)
cue('empty_sky',.074,.015,sky_detail=1)
cue('open_surround',.082,.021,surround_detail=1)
cue('smooth_floor',.079,.024,base_detail=1)
cue('textured_subject',.042,.152,core_grain=1)
cue('busy_floor',.043,.135,base_grain=1)
cue('isolated',.006,.110,center_detail=1,surround_detail=-1)
cue('focus',.006,.076,core_grain=1,surround_grain=-1)
cue('horiz',.03,.66,all_horizontal=1)
cue('vert',.005,.52,center_vertical=1)
cue('rectilinear',-.04,.32,center_axis=1)
cue('hull_line',.04,.92,lowerbody_horizontal=1)
cue('water_ripple',.015,.65,base_horizontal=1)
cue('legs',.01,.67,lowerbody_vertical=1)
cue('long_horizontal',.004,.080,center_hline=1,center_vline=-.67)
cue('long_vertical',.005,.080,center_vline=1,center_hline=-.62)
cue('soft_edges',.025,-.060,all_axis=1)
cue('tires',.43,1.55,tires=1)
cue('eye_pair',.38,1.40,face=1)
cue('dog_nose',.40,1.37,nose=1)
cue('muzzle',.014,.26,nose=1,face=-.91)
cue('smooth_animal',.112,.047,center_detail=1)
cue('ground_not_sky',.12,.66,sky_pale=.6,sky_blue=.4,base_pale=-.6,base_blue=-.4)
cue('white_belly',-.10,.72,center_brightness=1,surround_brightness=-1)

# Compile bounded evidence into one vector (two hinges per coordinate).
cn=list(feature_specs); cnt=len(cn); branch={}; fbias=[]
for i,n in enumerate(cn):
    terms,lo,hi=feature_specs[n];bias=-lo/(hi-lo)
    for key,wt in terms.items():
        src,w,b_=forms[key];branch.setdefault(src,np.zeros((2*cnt,len(w))))[2*i:2*i+2]+=wt*w/(hi-lo);bias+=wt*b_/(hi-lo)
    fbias.extend([bias,bias-1])
acc=[]
for k,(src,W) in enumerate(branch.items()): acc.append(affine('cue_projection_'+str(k),src,W,fbias if k==0 else 0))
ac=rect('evidence_hinges',layer('cue_thresholds','add',acc)); mw=np.zeros((cnt,2*cnt));mw[np.arange(cnt),2*np.arange(cnt)]=1;mw[np.arange(cnt),2*np.arange(cnt)+1]=-1
members=affine('evidence',ac,mw)

# Positive and negative votes are relative to the ten alternatives. Each
# line names the visual reason for the vote, rather than an image exemplar.
linear=[{} for _ in range(10)]; rules=[]
def lin(c,**kw):linear[c].update(kw)
def rule(c,strength,*items, level=.48):
    # entries with ! mean absence of a cue
    rules.append((c,strength,items,level))
lin(0,open_sky=.95,blue_surround=.40,empty_sky=.48,open_surround=.65,uniform_light=.42,isolated=.42,pasture=-.23,foliage=-.40,shadow_line=-.35,warm_indoor=-.33,fine_bg=-.41)
rule(0,1.55,'open_surround','blue_surround','isolated');rule(0,.80,'open_surround','uniform_light');rule(0,.67,'open_sky','empty_sky','!ground_not_sky');rule(0,-.58,'hull_line','water_ripple')
lin(1,neutral_road=.47,shadow_line=.72,tires=.83,rectilinear=.34,dark_chassis=.43,long_horizontal=.28,horiz=.20,pasture=-.39,blue_floor=-.27,olive=-.28)
rule(1,1.04,'shadow_line','neutral_road','tires');rule(1,.45,'colorful','hull_line');rule(1,-.66,'vert','white_box','light_top');rule(1,.35,'dark_chassis','smooth_floor')
lin(2,foliage=.58,leafy_top=.43,open_surround=.38,isolated=.45,focus=.29,blue_surround=.15,rectilinear=-.35,shadow_line=-.26,mud=-.25)
rule(2,.98,'isolated','foliage');rule(2,.62,'blue_surround','open_surround','!horiz');rule(2,.64,'foliage','colorful');rule(2,.57,'isolated','open_surround','soft_edges')
lin(3,eye_pair=.67,warm_indoor=.41,textured_subject=.50,nature_palette=.15,neutral_body=.14,rectilinear=-.31,open_sky=-.56,pasture=-.45,leafy_top=-.28,shadow_line=-.18)
rule(3,.8,'eye_pair','textured_subject','!muzzle');rule(3,.50,'eye_pair','warm_indoor');rule(3,.44,'textured_subject','focus');rule(3,.37,'neutral_body','!blue_surround','eye_pair')
lin(4,foliage=.66,pasture=.67,leafy_top=.43,warm_subject=.32,grass_contrast=.35,legs=.16,open_sky=-.38,rectilinear=-.25,dog_nose=-.16)
rule(4,1.13,'grass_contrast','pasture','warm_subject');rule(4,.58,'leafy_top','foliage','smooth_animal');rule(4,-.49,'mud','olive','busy_floor')
lin(5,dog_nose=.62,muzzle=.40,warm_subject=.39,warm_indoor=.32,textured_subject=.31,focus=.22,soft_edges=.15,open_sky=-.45,horiz=-.30,olive=-.22)
rule(5,.99,'muzzle','dog_nose','warm_subject');rule(5,.35,'dog_nose','warm_indoor');rule(5,.50,'white_belly','textured_subject','soft_edges')
lin(6,mud=.77,olive=.53,green_subject=.27,dark_scene=.49,busy_floor=.36,nature_palette=.25,fine_bg=.15,open_sky=-.64,blue_surround=-.46,rectilinear=-.31)
rule(6,1.08,'mud','olive','dark_scene');rule(6,.67,'busy_floor','mud','!grass_contrast');rule(6,.51,'green_subject','textured_subject');rule(6,.42,'dark_scene','!bright_above','soft_edges')
lin(7,pasture=.52,warm_subject=.62,legs=.58,grass_contrast=.32,empty_sky=.12,rectilinear=-.15,blue_floor=-.40,olive=-.37)
rule(7,1.03,'legs','pasture','warm_subject');rule(7,.61,'pasture','ground_not_sky','!leafy_top');rule(7,.48,'grass_contrast','legs','dark_chassis')
lin(8,blue_floor=.73,cool_floor=.51,horiz=.58,water_ripple=.46,hull_line=.32,cool_top=.28,bright_above=.32,uniform_light=.14,pasture=-.48,warm_indoor=-.46)
rule(8,1.23,'blue_floor','hull_line');rule(8,.73,'light_top','water_ripple','cool_floor');rule(8,.60,'horiz','bright_above','blue_surround');rule(8,-.52,'shadow_line','tires','neutral_road')
lin(9,rectilinear=.68,vert=.33,white_box=.36,tires=.44,neutral_road=.38,light_top=.42,shadow_line=.42,dark_chassis=.23,pasture=-.31,blue_floor=-.23)
rule(9,1.18,'rectilinear','ground_not_sky','tires');rule(9,.73,'white_box','long_vertical','dark_chassis');rule(9,.54,'colorful','rectilinear','light_top');rule(9,.50,'hull_line','rectilinear','vert')

# Center the main-effect coordinates at semantic neutral points: indicators
# of a special color or structure have a low default, not half coverage.
prior={
 'open_sky':.20,'blue_scene':.22,'cool_top':.45,'light_top':.39,'blue_floor':.29,'blue_surround':.25,'cool_floor':.24,'foliage':.25,'pasture':.32,'leafy_top':.27,'warm_subject':.35,'gold_subject':.30,'warm_indoor':.36,'mud':.39,'olive':.30,'green_subject':.27,'colorful':.16,'neutral_road':.48,'neutral_body':.50,'white_box':.30,'dark_chassis':.45,'shadow_line':.34,'bright_above':.45,'uniform_light':.20,'cool_scene':.35,'bright_scene':.40,'dark_scene':.34,'nature_palette':.43,'grass_contrast':.28,'fine_bg':.35,'coarse_bg':.40,'empty_sky':.44,'open_surround':.36,'smooth_floor':.37,'textured_subject':.47,'busy_floor':.41,'isolated':.43,'focus':.43,'horiz':.24,'vert':.25,'rectilinear':.36,'hull_line':.29,'water_ripple':.28,'legs':.27,'long_horizontal':.48,'long_vertical':.40,'soft_edges':.30,'tires':.42,'eye_pair':.46,'dog_nose':.46,'muzzle':.34,'smooth_animal':.39,'ground_not_sky':.32,'white_belly':.31}
M=np.zeros((10,cnt));B=np.zeros(10)
for c in range(10):
    for n,wt in linear[c].items(): M[c,cn.index(n)]=wt;B[c]-=prior[n]*wt
main=affine('scene_vote',members,M,B)
W=np.zeros((len(rules),cnt));B=np.zeros(len(rules));V=np.zeros((10,len(rules)));C=np.zeros(10)
for i,(c,strength,items,level) in enumerate(rules):
    # A majority of the component evidence must be present.
    B[i]=-level/(1-level); pv=0.
    for item in items:
        neg=item.startswith('!'); n=item.lstrip('!'); mag=1/(len(items)*(1-level));W[i,cn.index(n)]+=(-1 if neg else 1)*mag
        if neg:B[i]+=mag
        pv+=(1-prior[n] if neg else prior[n])/len(items)
    V[c,i]=strength; C[c]-=strength*(.065+max(0,(pv-level)/(1-level)))
conj=rect('joint_evidence',affine('evidence_combinations',members,W,B));joint=affine('joint_vote',conj,V,C)
# Compare full scene descriptions rather than letting several names for the
# same tint accumulate as independent votes. Each row below is an
# alternative world: three desired properties and two counterindications.
# Pool over alternatives, since e.g. a docked liner need not have a blue sea.
worlds=[
 [ # aircraft: airborne, on tarmac, in clouds, silhouette, low over land
 ('open_sky open_surround isolated','busy_floor grass_contrast'),
 ('open_sky empty_sky neutral_road','dark_chassis fine_bg'),
 ('uniform_light open_surround isolated','warm_subject leafy_top'),
 ('open_surround focus cool_top','water_ripple tires'),
 ('empty_sky isolated ground_not_sky','rectilinear textured_subject')],
 [ # car: on road, in a lot, colored paint, side profile, front view
 ('neutral_road tires shadow_line','water_ripple leafy_top'),
 ('smooth_floor dark_chassis tires','olive open_surround'),
 ('colorful neutral_road dark_chassis','warm_indoor foliage'),
 ('hull_line shadow_line smooth_floor','blue_floor white_box'),
 ('neutral_body tires shadow_line','open_sky nature_palette')],
 [ # bird against leaves, at sky, perched, breast and wing, bright plumage
 ('foliage isolated focus','hull_line dark_chassis'),
 ('blue_surround open_surround soft_edges','horiz water_ripple'),
 ('leafy_top isolated smooth_animal','mud tires'),
 ('white_belly focus open_surround','rectilinear neutral_road'),
 ('colorful foliage isolated','shadow_line neutral_road')],
 [ # cats: face portrait, tabby, light or dark house cat, sitting indoors
 ('eye_pair textured_subject focus','rectilinear muzzle'),
 ('eye_pair warm_indoor nature_palette','open_sky pasture'),
 ('white_belly eye_pair textured_subject','grass_contrast hull_line'),
 ('eye_pair dark_scene soft_edges','muzzle olive'),
 ('warm_indoor eye_pair neutral_body','foliage open_sky')],
 [ # deer: tan in meadow, woods, standing, bedding down, deer portrait
 ('grass_contrast pasture warm_subject','blue_scene rectilinear'),
 ('leafy_top foliage nature_palette','muzzle shadow_line'),
 ('legs grass_contrast foliage','blue_surround olive'),
 ('warm_subject smooth_animal pasture','muzzle tires'),
 ('eye_pair warm_subject leafy_top','dark_chassis blue_floor')],
 [ # dog: face, in house, playing outside, fluffy dog, dark dog
 ('dog_nose muzzle textured_subject','hull_line blue_floor'),
 ('dog_nose warm_indoor gold_subject','pasture olive'),
 ('dog_nose grass_contrast white_belly','horiz rectilinear'),
 ('white_belly focus textured_subject','open_sky blue_floor'),
 ('dog_nose dark_scene focus','olive foliage')],
 [ # frog: camouflage among reeds, on mud, on rock, green frog, at night
 ('olive nature_palette busy_floor','grass_contrast long_horizontal'),
 ('mud dark_scene textured_subject','hull_line light_top'),
 ('earthy_rock busy_floor soft_edges','rectilinear blue_surround'),
 ('green_subject olive focus','grass_contrast open_sky'),
 ('dark_scene mud fine_bg','bright_above white_box')],
 [ # horse: paddock, dark horse, ridden, chestnut, horse's face
 ('legs pasture grass_contrast','olive blue_floor'),
 ('dark_chassis pasture legs','neutral_road rectilinear'),
 ('legs ground_not_sky pasture','mud leafy_top'),
 ('warm_subject legs grass_contrast','muzzle olive'),
 ('warm_subject long_vertical eye_pair','blue_surround olive')],
 [ # ship: on blue water, distant, dockside, grey sea, sails
 ('blue_floor water_ripple hull_line','shadow_line busy_floor'),
 ('horiz blue_surround bright_above','tires long_vertical'),
 ('hull_line water_ripple white_box','shadow_line nature_palette'),
 ('horiz cool_floor light_top','foliage dark_chassis'),
 ('blue_floor white_box open_sky','shadow_line neutral_road')],
 [ # truck: box in profile, cab, open cargo, painted body, semitrailer
 ('white_box rectilinear dark_chassis','blue_floor pasture'),
 ('long_vertical tires light_top','olive water_ripple'),
 ('tires hull_line ground_not_sky','blue_floor leafy_top'),
 ('colorful rectilinear neutral_road','open_surround pasture'),
 ('shadow_line rectilinear long_vertical','blue_floor olive')]]
world_W=[]; world_bias=[]
for cs in worlds:
    for yes,no in cs:
        f=np.zeros(cnt)
        for n in yes.replace('earthy_rock','nature_palette').split(): f[cn.index(n)]+=1./3
        for n in no.split(): f[cn.index(n)]-=.26
        world_W.append(f);world_bias.append(-sum(v*prior[n] for v,n in zip(f,cn)))
ww=affine('scene_models',members,world_W,world_bias);ww=reshape('scene_alternatives',ww,[10,1,5]);mx=pool('scene_best','max',ww,[1,5],[1,5]);av=pool('scene_mean','avg',ww,[1,5],[1,5]);sv=affine('scene_agreement',layer('scene_summary','add',[mx,av]),1.7*np.eye(10))
# Scale direct and conjunction-based votes as a *single* source, and cap
# its influence (a landscape can't establish the identity of the subject).
context=layer('context_sources','add',[main,joint]);z=affine('context_limits',context,np.r_[.43*np.eye(10),.43*np.eye(10)],np.r_[np.full(10,1.35),np.full(10,-1.35)]);z=rect('context_hinges',z);z=affine('bounded_context',z,np.c_[np.eye(10),-np.eye(10)],-1.35)
# Generative comparison in a coarse color layout. Instead of adding
# several redundant color adjectives, describe complete plausible views in
# terms of physical materials: sky, clouds, forest, meadow, water, road,
# wall, coat, glass and rubber. An affine minus-half-squared-distance is
# exact up to the same input-only term for EVERY hypothesis. No samples
# from any photo collection are used to design or select these surfaces.
def material_worlds():
    palette={
      'sky':[.52,.70,.86],'blue':[.29,.49,.75],'cloud':[.76,.78,.80],
      'mist':[.61,.64,.65],'water':[.27,.41,.51],'sea':[.20,.34,.46],
      'harbor':[.42,.47,.47],'field':[.42,.49,.26],'green':[.26,.36,.16],
      'leaf':[.27,.32,.20],'wood':[.39,.34,.24],'earth':[.43,.37,.27],
      'dirt':[.32,.27,.19],'sand':[.63,.57,.45],'road':[.49,.49,.48],
      'street':[.36,.36,.36],'wall':[.58,.54,.47],'room':[.42,.36,.32],
      'white':[.78,.77,.73],'silver':[.63,.64,.66],'black':[.15,.15,.16],
      'brown':[.48,.32,.21],'tan':[.64,.52,.36],'gray':[.45,.44,.42],
      'bay':[.40,.25,.17],'rust':[.60,.32,.19],'cream':[.77,.71,.56],
      'red':[.65,.22,.17],'paintblue':[.24,.36,.56],'yellow':[.72,.54,.21],
      'stone':[.44,.43,.38],'olive':[.40,.43,.24],'pool':[.40,.53,.54],
    }
    # Environment choices are deliberately not unrestricted: there is
    # typically sky above water, pavement under tires, grass under hooves.
    settings=[
      [('blue','sky'),('sky','sky'),('cloud','mist'),('cloud','cloud'),('sky','road'),('cloud','field')],
      [('leaf','road'),('street','road'),('mist','street'),('wall','road'),('cloud','road'),('green','street')],
      [('leaf','green'),('field','leaf'),('sky','sky'),('cloud','mist'),('wood','earth'),('green','field')],
      [('room','wall'),('wall','room'),('gray','gray'),('cream','wall'),('wood','earth'),('leaf','field')],
      [('leaf','field'),('green','green'),('field','field'),('wood','earth'),('leaf','earth'),('mist','field')],
      [('wall','room'),('room','wall'),('field','green'),('leaf','field'),('sand','sand'),('gray','wall')],
      [('olive','earth'),('leaf','dirt'),('stone','stone'),('green','olive'),('earth','earth'),('dirt','leaf')],
      [('sky','field'),('leaf','field'),('mist','sand'),('wood','earth'),('field','field'),('cloud','green')],
      [('sky','water'),('cloud','sea'),('mist','harbor'),('sky','sea'),('cloud','water'),('leaf','harbor')],
      [('cloud','road'),('sky','street'),('mist','road'),('leaf','road'),('wall','street'),('mist','sand')]]
    bodies=[['white','silver','black','gray'],['red','silver','white','black','paintblue'],
        ['brown','gray','cream','black','yellow','white'],['gray','cream','rust','black','white'],
        ['tan','brown','gray','bay'],['brown','cream','black','white','tan'],
        ['olive','earth','brown','green','stone'],['bay','black','cream','tan','rust'],
        ['white','silver','black','red'],['white','red','paintblue','silver','yellow']]
    y,x=(np.indices((32,32))+.5)/16-1
    def coarse_map(v): return v.reshape(8,4,8,4).mean((1,3))
    def formcolors(rgb):
        return np.array([np.sum(rgb*ax[:,None,None],axis=0) for ax in feature_axes])
    # Opponent coordinates and deliberately different noise scales for
    # variable brightness vs. relatively stable hue and reflectance layout.
    feature_axes=np.array([[.26,.51,.23],[-.72,-.28,1.],[-.63,1.,-.37]])
    invvar=np.array([1./.24**2,1./.138**2,1./.091**2])[:,None,None]
    rng=np.random.default_rng(45491); W=[];bias=[];families=6;poses=156
    center=.54+.46*np.exp(-((np.indices((8,8))[0]-3.5)**2+(np.indices((8,8))[1]-3.5)**2)/18);center/=center.sum()
    for c in range(10):
        for family,(atm,ground) in enumerate(settings[c]):
            for i in range(poses):
                scale=rng.uniform(.66,1.18) if c in (0,1,8,9) else rng.uniform(.78,1.98)
                a=rng.uniform(-.29,.29) if c in (0,2,6) else rng.uniform(-.105,.105)
                X0=x-rng.uniform(-.17,.17);Y0=y-rng.uniform(-.20,.19)
                X=(1 if i%2 else -1)*(np.cos(a)*X0+np.sin(a)*Y0)/(scale*rng.uniform(.89,1.12));Y=(-np.sin(a)*X0+np.cos(a)*Y0)/(scale*rng.uniform(.86,1.31))
                model=draw(c,i%5,X,Y)
                mc=np.array(palette[rng.choice(bodies[c])]);top=np.array(palette[atm]);bottom=np.array(palette[ground]);
                # A distant transition is soft in woods and rooms, but
                # quite sharp between the ocean and the sky.
                split=rng.uniform(-.26,.23) if c not in (1,9) else rng.uniform(-.17,.41)
                soft=.05 if c in (0,8) else .31
                band=softclip(.5+(y-split+rng.uniform(-.075,.075)*x)/soft)
                horizon=coarse_map(band*(model==0)); amt=coarse_map(model==0)
                rgb=top[:,None,None]*(amt-horizon)+bottom[:,None,None]*horizon
                mats=[mc,mc*.59, np.array([.10,.098,.098]),.52*mc+.48*np.array(palette['white']),.47*top+.53*np.array([.18,.21,.23]),.53*mc+.47*np.array(palette['rust'])]
                if c==8:mats[1]=np.array(palette['street'])
                if c==9:mats[3]=np.array(palette['silver' if i%3 else 'white'])
                for m,albedo in enumerate(mats,1):rgb+=albedo[:,None,None]*coarse_map(model==m)
                # Modest exposure and white-balance variations; these are
                # illuminants of the drawing, not image measurements.
                rgb=(rgb-rng.uniform(-.055,.055))*rng.uniform(.79,1.20)
                rgb+=rng.uniform(-.041,.041)*np.array([1.,.28,-.9])[:,None,None]
                feat=formcolors(rgb)
                # Soft coverage permits partial background violations. An
                # input-only squared norm cancels in nearest-model voting.
                wt=center*invvar
                wtfeat=feat*wt
                coeff=np.einsum('ihw,ij->jhw',wtfeat,feature_axes)
                W.append(coeff.ravel().astype(F));bias.append(-.5*np.sum(feat*wtfeat))
    s=affine('physical_color_layouts',coarse,W,bias);s=reshape('material_world_grid',s,[10,families,poses]);s=pool('best_rendered_scene','max',s,[1,poses],[1,poses]);s=pool('best_material_world','max',s,[families,1],[families,1]);
    # Only relative preferences have meaning for this metric.
    return affine('rendered_scene_vote',s,3.1*(np.eye(10)-np.ones((10,10))*.1))
materialvote=material_worlds()
# Bounded heuristic evidence supplements the two generative branches.
shape_and_scene=layer('generative_models','add',[shapevote,materialvote]);
rest=affine('context_contribution',layer('context_and_alternatives','add',[z,sv]),.75*np.eye(10))
layer('class_scores','add',[shape_and_scene,rest])

outdir=Path(sys.argv[1]);outdir.mkdir(parents=True,exist_ok=True)
(outdir/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n');np.savez(outdir/'weights.npz',**weights)
