"""A picture grammar and a set of scene cues. No data, learned parameters, or external assets."""
import sys, json, math
from pathlib import Path
import numpy as np

layers=[]; weights={}
def layer(n,op,src,**kw):
    layers.append(dict(name=n,op=op,inputs=[src] if isinstance(src,str) else src,**kw)); return n
def affine(n,src,w,b=0,**kw):
    w=np.asarray(w,np.float32); weights[n+'.W']=w; weights[n+'.b']=np.broadcast_to(np.asarray(b,np.float32),(len(w),)).copy()
    if w.ndim==2: return layer(n,'dense',src,out=len(w))
    return layer(n,'conv',src,kernel=list(w.shape[2:]),out=len(w),stride=1,padding=0,**kw)
def pool(n,x,k,s=0,op='avgpool'):
    k=(k,k) if isinstance(k,int) else k
    s=(s,s) if isinstance(s,int) and s else s or k
    return layer(n,op,x,size=list(k),stride=list(s))
def relu(n,x):return layer(n,'relu',x)
def reshape(n,x,*s):return layer(n,'reshape',x,shape=list(s))
def add(n,*x):return layer(n,'add',list(x))

# A little orthographic drawing language. Coordinates are in object space,
# not image coordinates. Arrays are material masks of the idealized objects.
class Drawing:
    def __init__(self,x,y):
        self.x=x; self.y=y; self.z=np.zeros(x.shape)
    def ell(self,x,y,rx,ry,v,a=0):
        ca,sa=math.cos(a),math.sin(a); dx,dy=self.x-x,self.y-y
        self.z[((ca*dx+sa*dy)/rx)**2+((ca*dy-sa*dx)/ry)**2<=1]=v
    def poly(self,pts,v):
        m=False; x,y=self.x,self.y
        for (a,b),(c,d) in zip(pts,pts[1:]+pts[:1]):
            if b!=d: m=m^(((b>y)!=(d>y))&(x<a+(y-b)*(c-a)/(d-b)))
        self.z[m]=v
    def line(self,a,b,r,v):
        x,y=self.x-a[0],self.y-a[1]; dx,dy=b[0]-a[0],b[1]-a[1]
        t=np.clip((x*dx+y*dy)/(dx*dx+dy*dy+1e-12),0,1)
        self.z[(x-t*dx)**2+(y-t*dy)**2<=r*r]=v
    def legs(self,pts,r,v):
        for a,b in zip(pts,pts[1:]):self.line(a,b,r,v)

def vehicle(d,k,pose):
    p,e,l=d.poly,d.ell,d.line
    # tires, hubs, underbody, glass, headlamps and painted panels
    def tire(x,y,rx,ry):
        e(x,y,rx,ry,-.93); e(x,y,rx*.43,ry*.43,.35)
    def front(high):
        top=-.89 if high else -.60; mid=-.03 if high else -.16
        e(-.54,.52,.13,.21,-.95);e(.54,.52,.13,.21,-.95)
        p([(-.62,.53),(-.69,.25),(-.61,mid),(-.48,top),(.47,top),(.64,mid),(.69,.29),(.61,.53)],.65)
        p([(-.56,mid),(-.43,top+.09),(.42,top+.09),(.56,mid)],-.54)
        p([(-.47,.19),(.47,.19),(.37,.44),(-.38,.44)],-.65)
        for s in (-1,1):e(s*.51,.16,.12,.075,1.)
        l((-.60,.47),(.60,.47),.035,-.85)
    if pose==2:front(k==9);return
    if k==1:
        # Three-quarter projection slides the far side of the car backwards.
        q=.25 if pose==1 else 0
        p([(-.95,.32),(-.97,.05),(-.83,-.13),(-.56,-.22),(-.35,-.46-q*.4),(.13,-.50-q*.4),(.49,-.19),(.83,-.10),(.97,.04),(.96,.31),(.79,.40),(-.78,.40)],.72)
        p([(-.53,-.19),(-.32,-.41-q*.4),(.08,-.43-q*.4),(.34,-.19)],-.50)
        l((-.13,-.40),(-.12,-.19),.022,.55)
        p([(-.90,.26),(.92,.26),(.89,.35),(-.85,.38)],-.64)
        if q:
            p([(.35,-.20),(.68,-.10),(.91,.02),(.50,-.04)],.96)
            p([(.43,.15),(.95,.10),(.94,.27),(.47,.31)],-.50)
        tire(-.59,.33,.15-q*.12,.18); tire(.59-q,.33+q*.17,.15-q*.12,.18)
        e(.83,.04,.10,.045,.98)
    else:
        # Rigid box lorry / bonneted tractor-trailer / city bus
        q=.2 if pose==1 else 0
        p([(-.95,-.60),(.30,-.60-q/2),(.32,.21),(-.95,.17)],.85)
        p([(-.95,.17),(.32,.21),(.78,.32),(-.92,.29)],-.7)
        p([(.25,.26),(.26,-.43-q),(.68,-.44-q/2),(.84,-.16),(.98,-.10),(.95,.34),(.35,.34)],.50)
        p([(.34,-.34-q),(.64,-.35-q/2),(.75,-.13),(.35,-.12)],-.73)
        if q:
            p([(.64,-.35-q/2),(.87,-.31),(.94,-.11),(.76,-.13)],-.43)
            p([(.74,.05),(.98,.04),(.97,.23),(.74,.22)],-.69)
        tire(-.72,.30,.12,.15);tire(-.43,.31,.12,.15);tire(.60,.34,.13,.16)
        if pose==3:
            p([(-.91,-.49),(.61,-.49),(.76,-.11),(-.91,-.12)],-.66)
            for x in np.arange(-.68,.66,.28):l((x,-.48),(x,-.13),.022,.65)

def three_quarter(d,k,p):
    # A pinhole projection of a schematic vehicle. Length is the X axis,
    # height is H; positive X and positive Z sides face the camera. Varying
    # yaw now moves windshield, front fascia, body side and wheels together.
    car=k==1; yaw=math.radians([25,48,68,38,65,77][p%6]);el=math.radians([14,18,11,26,23,14][p%6]);co,si=math.cos(yaw),math.sin(yaw);ce,se=math.cos(el),math.sin(el)
    def proj(x,z,h):
        depth=x*si+z*co; scale=1/(1-.095*(depth*ce+h*se));return ((x*co-z*si)*scale,((depth*se-h*ce)*scale+(.40 if car else .59)))
    def panel(pts,c):d.poly([proj(*a) for a in pts],c)
    def quad(x0,x1,z0,z1,h,c):panel([(x0,z0,h),(x1,z0,h),(x1,z1,h),(x0,z1,h)],c)
    def face(x,z0,z1,h0,h1,c):panel([(x,z0,h0),(x,z1,h0),(x,z1,h1),(x,z0,h1)],c)
    def side(z,x0,x1,h0,h1,c):panel([(x0,z,h0),(x1,z,h0),(x1,z,h1),(x0,z,h1)],c)
    def box(x0,x1,z0,z1,h0,h1,t,s,f):
        side(z1,x0,x1,h0,h1,s);face(x1,z0,z1,h0,h1,f);quad(x0,x1,z0,z1,h1,t)
    def disk(x,z,h,rx,ry,c):
        panel([(x+rx*math.cos(t),z,h+ry*math.sin(t)) for t in np.linspace(0,2*math.pi,17)[:-1]],c)
    w=.47 if car else .40;r=.175 if car else .19
    # Wheels of the far side partially occluded by the near-facing panels.
    for z in [-w,w]:
        for x in [-.64,.63]:disk(x,z,r,r,r,-.95)
    if car:
        box(-.97,.96,-w,w,.17,.45,.87,.51,.63)
        # windowed greenhouse, tapered roof and sloped windshields
        a,b=-.76,.43;ar,br=-.49,.12;h=.45;hr=.83+(p%2)*.045;wr=w*.78
        panel([(a,w,h),(b,w,h),(br,wr,hr),(ar,wr,hr)],-.56)
        panel([(br,-wr,hr),(b,-w,h),(b,w,h),(br,wr,hr)],-.75)
        quad(ar,br,-wr,wr,hr,.92)
        d.line(proj(-.24,w,h),proj(-.25,wr,hr),.018,.66)
        quad(b,.94,-w,w,.455,.94);face(.967,-w*.66,w*.66,.21,.33,-.81)
        for z in [-.35,.35]:face(.968,z-.084,z+.084,.33,.426,1.12)
        face(.969,-.12,.12,.19,.25,.84);side(w+.005,-.89,.88,.18,.216,-.40)
    else:
        box(-1,.95,-.36,.36,.135,.32,-.42,-.72,-.7)
        cargo=1.05 if p%2 else 1.21
        box(-1,.34,-w,w,.34,cargo,.95,.69,.49)
        box(.37,.98,-w,w,.28,.96,.78,.47,.64)
        face(.987,-.34,.34,.61,.875,-.73);side(w+.004,.46,.89,.62,.87,-.54)
        face(.989,-.27,.27,.34,.53,-.82)
        for z in [-.32,.32]:face(.99,z-.057,z+.057,.36,.44,1.08)
    for x in [-.64,.64]:
        disk(x,w+.011,r,r,r,-.95);disk(x,w+.012,r,.44*r,.44*r,.27)
    if not car:disk(-.25,w+.011,r,r*.95,r,-.90);disk(-.25,w+.012,r,r*.41,r*.41,.23)

def extra_animal(d,k,p):
    e,g,l=d.ell,d.poly,d.line
    if k==2:
        # long-necked bird: a light S-shaped neck joining body to a small head
        e(-.12,.16,.44,.26,.77,-.2);e(-.22,.15,.34,.16,-.51,-.25)
        d.legs([(.19,.10),(.30,-.15),(.26,-.44),(.38,-.60)],.068,.77)
        e(.44,-.60,.14,.085,.77);g([(.50,-.62),(.80,-.54),(.48,-.55)],-.57)
        for x in [-.16,.03]:d.legs([(x,.36),(x-.01,.58),(x-.10,.77)],.029,.61)
    elif k==6:
        # dorsal frog/toad: tapering snout, bent back legs, scattered warts
        e(0,.02,.40,.56,.78);e(0,-.36,.32,.22,.78)
        for s in [-1,1]:
            e(s*.255,-.42,.079,.12,-.78,s*.25);e(s*.36,.25,.18,.30,.66,s*-.58)
            d.legs([(s*.27,-.11),(s*.50,-.02),(s*.55,-.29)],.055,.77)
            for a,b in [(.12,.14),(.21,-.17),(.19,.30),(.05,-.24)]:e(s*a,b,.042,.060,-.28)
    elif p==5: # broad face, eye sockets, blaze, bridge of nose and chin
        e(0,.62,.76,.57,.71); e(0,-.15,.61,.63,.82)
        for s in [-1,1]:
            if k==5:e(s*.53,-.21,.19,.44,-.57,s*-.34)
            else:g([(s*.54,-.23),(s*.61,-.93),(s*.17,-.58)],.71)
            e(s*.286,-.22,.195,.17,-.30);e(s*.27,-.20,.085,.057,-.96)
        e(0,.065,.27,.36,.93);e(0,.17,.154 if k==5 else .086,.093 if k==5 else .059,-.98)
        l((-.15,.35),(.16,.35),.035,-.68)
        if k==3:
            for s in [-1,1]:
                for i in range(3):l((s*.56,-.06+.11*i),(s*.35,.01+.09*i),.029,-.42)
    elif k==3: # curled up body, small tucked head and a wrapping tail
        e(-.11,.16,.73,.45,.73);e(.46,-.095,.34,.30,.81)
        g([(.17,-.11),(.12,-.51),(.41,-.34),(.66,-.54),(.77,-.15)],.78)
        for x in [.35,.61]:l((x-.048,-.12),(x+.041,-.12),.022,-.69)
        e(.48,.007,.047,.034,-.80);d.legs([(-.72,.17),(-.59,.40),(-.24,.49),(.23,.44)],.079,-.47)
    else: # profile of a canine muzzle in a head-and-shoulders view
        e(-.21,.47,.54,.60,.66);e(-.073,-.18,.45,.42,.78,-.22)
        e(.32,-.073,.36,.17,.84,.13);e(.62,-.09,.090,.069,-.94);e(.17,-.27,.061,.056,-.94)
        e(-.34,-.19,.16,.35,-.61,-.23);l((.26,.03),(.54,.028),.024,-.66)

def plane(d,p):
    e,g,l=d.ell,d.poly,d.line
    if p==0: # side, long fuselage, swept tail and wings
        g([(-.93,-.05),(-.98,-.43),(-.85,-.42),(-.60,-.09),(.65,-.05),(.97,.04),(.80,.13),(-.65,.11)],.8)
        g([(-.20,.05),(-.55,.46),(-.38,.44),(.26,.09)],-.62)
        g([(-.19,0),(-.50,-.28),(-.38,-.27),(.19,-.01)],.45)
        l((-.48,.015),(.76,.045),.025,-.5)
    elif p==1: # overhead
        e(0,0,.115,.93,.8);g([(-.05,-.32),(-.99,.30),(-.94,.41),(0,.20),(.94,.41),(.99,.30),(.05,-.32)],.74)
        g([(0,.53),(-.39,.82),(-.38,.89),(0,.79),(.38,.89),(.39,.82)],.7);e(0,-.43,.061,.23,-.60)
    elif p==2: # approaching, engines under wings
        g([(-1,-.12),(-.99,-.05),(-.14,.10),(.14,.10),(.99,-.05),(1,-.12),(0,-.025)],.7)
        for s in [-1,1]: e(s*.40,.095,.10,.108,-.7)
        g([(-.04,0),(-.023,-.47),(.023,-.47),(.04,0)],.72);e(0,.04,.15,.20,.80);e(0,-.045,.108,.044,-.65)
    else: # banked oblique view
        g([(-.93,.32),(-.77,.10),(.70,-.32),(.97,-.34),(.83,-.18),(-.75,.37)],.85)
        g([(-.10,-.03),(-.58,-.51),(-.40,-.47),(.23,-.13)],.63)
        g([(-.17,.13),(-.05,.72),(.09,.63),(.23,-.10)],.73)
        g([(-.77,.25),(-.92,-.06),(-.79,-.07),(-.56,.16)],.58)

def boat(d,p):
    e,g,l=d.ell,d.poly,d.line
    if p==3: # bow-on
        g([(-.55,.02),(.55,.02),(.31,.51),(0,.65),(-.33,.49)],-.72)
        g([(-.48,-.02),(-.34,-.43),(.31,-.43),(.48,-.02)],.88)
        g([(-.35,-.18),(-.27,-.37),(.24,-.37),(.35,-.18)],-.48);l((0,-.46),(0,-.75),.025,.64);return
    g([(-.98,.06),(.98,.025),(.74,.35),(.56,.43),(-.72,.40)],-.86)
    l((-.89,.10),(.87,.082),.032,.75)
    if p==0: # deckhouse, bridge, stack
        g([(-.65,.04),(-.65,-.15),(-.42,-.15),(-.42,-.29),(.42,-.29),(.42,-.14),(.62,-.12),(.70,.04)],.9)
        l((-.36,-.20),(.37,-.20),.043,-.55);g([(-.24,-.30),(-.23,-.49),(-.07,-.49),(-.05,-.30)],-.58)
    elif p==1:
        g([(-.55,.04),(-.50,-.34),(-.18,-.34),(-.18,-.12),(.63,-.11),(.71,.04)],.88)
        l((-.40,-.34),(-.40,-.64),.022,.62)
    elif p==2:
        l((-.09,.06),(-.09,-.99),.023,-.73);g([(-.13,-.94),(-.15,-.04),(-.79,-.035)],.93);g([(-.033,-.80),(.64,-.045),(-.024,-.045)],.75)

def ungulate(d,k,pose):
    e,g,l=d.ell,d.poly,d.line;horse=k==7
    if pose==3: # head-and-neck profile; face points to the right
        g([(-.76,.95),(-.49,-.09),(-.28,-.63),(.06,-.74),(.29,-.57),(.63,-.04),(.49,.11),(.16,-.03),(.01,.13),(.09,.96)],.75)
        for x in [-.21,.01]:e(x,-.73,.08,.23,.68,.12)
        e(.10,-.49,.047,.044,-.85);e(.49,-.03,.052,.04,-.80)
        if horse:l((-.25,-.54),(-.57,.44),.087,-.65)
        return
    # Leg bundles in profile or slight perspective. Long shin and hoof.
    for hip,knee,foot in [(-.51,-.56,-.66),(-.40,-.29,-.25),(.25,.27,.23),(.37,.48,.53)]:
        dx= (.16 if hip>0 else -.14) if pose==2 else 0
        d.legs([(hip,.05),(knee+dx,.51),(foot+dx,.85)],.040 if horse else .027,.65)
        l((hip,.05),(knee+dx,.42),.068 if horse else .05,.65)
        e(foot+dx-.008,.85,.066 if horse else .043,.044,-.6)
    e(-.11,.015,.60,.245 if horse else .205,.73)
    if pose==1: # coat and face of grazing animal
        l((.28,-.06),(.66,.39),.115,.71);e(.73,.44,.118,.23,.76,-.45);e(.79,.58,.105,.084,-.49)
    else:
        g([(.11,-.13),(.45,-.53),(.54,-.62),(.71,-.56),(.52,-.03),(.37,.15)],.78)
        e(.68,-.53,.227,.115,.72,.50);e(.81,-.44,.11,.083,-.43 if horse else .68)
        for x in [.52,.63]:e(x,-.685,.042 if horse else .061,.129 if horse else .096,.78,-.25)
        if horse:l((.47,-.48),(.24,-.18),.05,-.64)
        if not horse and pose==2:
            for s in [-1,1]:
                d.legs([(.56,-.65),(.53+.13*s,-.80),(.48+.24*s,-.94)],.022,.7)
    if horse:d.legs([(-.62,-.08),(-.73,.11),(-.78,.48)],.075,-.70)
    else:e(-.68,-.02,.11,.055,.85,-.55)

def pet(d,k,p):
    e,g,l=d.ell,d.poly,d.line;cat=k==3
    if p<3: # face grammar: ear shapes, eyes, bridge and muzzle
        if cat:
            g([(-.62,-.27),(-.66,-.87),(-.19,-.55),(.18,-.56),(.63,-.89),(.64,-.22)],.68)
            for s in [-1,1]:g([(s*.55,-.69),(s*.55,-.39),(s*.32,-.49)],-.36)
            e(0,-.04,.68,.56,.67); e(0,.18,.44,.31,.75)
        else:
            e(0,-.1,.53,.62,.74)
            for s in [-1,1]:
                if p==1:g([(s*.16,-.56),(s*.53,-.97),(s*.51,-.21)],.65)
                else:e(s*.51,-.15,.22,.47,-.56,s*-.25)
            e(0,.14,.38,.42,.87)
        for s in [-1,1]:
            e(s*(.28 if cat else .24),-.126,.123 if cat else .086,.050 if cat else .070,-.83,-s*.16)
            if cat:l((s*.165,-.01),(s*.09,.16),.035,-.28)
        if cat:
            g([(-.095,.145),(.095,.145),(0,.23)],-.79)
            for s in [-1,1]:l((s*.055,.265),(s*.32,.25),.019,-.55)
        else:
            e(0,.22,.145,.103,-.98);l((-.17,.41),(.16,.42),.026,-.71)
        if p==2: # bib and chest below face; drawing coordinates used at smaller scale
            e(0,.89,.51,.43,.68)
    elif p==3: # sitting upright
        e(0,.31,.46,.56,.70);e(-.05,-.38,.36,.33,.74)
        if cat:g([(-.35,-.46),(-.33,-.88),(-.08,-.64),(.08,-.65),(.31,-.85),(.32,-.43)],.69)
        else:
            for s in [-1,1]:e(s*.32-.05,-.33,.13,.28,-.54)
        for s in [-1,1]:e(s*.14-.04,-.415,.047,.035,-.85)
        e(-.038,-.277,.06,.046,-.9)
        l((-.16,.25),(-.19,.79),.07,.82);l((.14,.25),(.17,.79),.07,.82)
    else:
        # lateral walking or reclining pose
        e(-.11,.13,.57,.27,.77);e(.47,-.12,.25,.25,.76)
        for x in [-.49,-.32,.14,.26]:l((x,.24),(x-.04,.60),.059,.68)
        if cat:
            g([(.22,-.18),(.25,-.52),(.45,-.33),(.60,-.49),(.66,-.18)],.7)
            d.legs([(-.61,.13),(-.83,-.09),(-.86,-.43)],.052,.81)
        else:e(.33,-.09,.107,.22,-.66);e(.67,-.04,.18,.113,.75);l((-.58,.11),(-.83,-.14),.068,.76)

def birdfrog(d,k,p):
    e,g,l=d.ell,d.poly,d.line
    if k==2:
        if p==2: # wings outstretched, tapered feather tips
            g([(-1,-.35),(-.61,-.30),(-.2,-.12),(0,.04),(.33,-.21),(1,-.39),(.70,-.04),(.19,.18),(-.13,.16),(-.67,-.055)],.72)
            e(0,.09,.135,.28,.8);return
        if p==3: # breast-on profile
            e(0,.14,.35,.55,.72);e(0,-.38,.23,.23,.7);g([(.13,-.40),(.41,-.34),(.14,-.30)],-.62)
            e(.10,-.42,.034,.034,-.9);return
        a=.33 if p==0 else .79
        g([(-.17,.14),(-.67,.70),(-.73,.61),(-.42,.13)],-.61)
        e(-.05,.05,.33,.54,.77,a);e(.21,-.38,.22,.22,.75)
        e(-.11,.10,.22,.37,-.43,a);g([(.40,-.41),(.65,-.32),(.40,-.31)],-.60);e(.28,-.42,.038,.035,-.93)
        for x in [-.04,.13]:l((x,.42),(x-.05,.60),.027,-.55)
    else:
        if p in [0,2]: # profile: hunched back, folded hind limb, flat jaw and round eye
            e(-.16,.08,.57,.32,.74,-.2);e(.35,-.08,.38,.205,.8,.20)
            e(.29,-.255,.12,.115,.86);e(.32,-.266,.066,.065,-.95)
            e(-.43,.20,.34,.184,-.44,-.58);e(-.30,.26,.29,.115,.66,-.38)
            d.legs([(.28,.03),(.20,.34),(.55,.41)],.059,.79);l((.40,.006),(.68,-.004),.02,-.68)
        else: # front: two protruding eyes, broad head and curved mouth
            e(0,.14,.56,.37,.73);e(0,-.046,.57,.25,.83)
            for s in [-1,1]:
                e(s*.34,-.215,.17,.155,.76);e(s*.34,-.23,.089,.08,-.94)
                e(s*.56,.22,.206,.18,.65,-s*.51);l((s*.39,.24),(s*.48,.46),.055,.82)
            d.legs([(-.46,.007),(-.2,.095),(.20,.095),(.46,.007)],.024,-.67)

# Shape-matcher input: zero-mean unit-energy oriented edge maps (unsigned), at
# an 8x8 grid, and normalized luminance at a 16x16 grid. Valid differences
# avoid inventing a border frame. Smoothing is by averaging over edge cells.
lum=np.array([.299,.587,.114]); angles=np.arange(4)*math.pi/4
luma=affine('luminance','image',lum.reshape(1,3,1,1));unit=layer('contrast','norm',luma)
small=pool('light16',unit,2)
y,x=np.mgrid[-1:2,-1:2];env=np.exp(-(x*x+y*y)/1.5)
kernels=[]
for a in angles:
    h=(x*np.cos(a)+y*np.sin(a))*env; h/=np.sum(abs(h))*.5
    kernels.extend([h,-h])
grad=affine('derivatives',unit,np.array(kernels)[:,None]); half=relu('edge_polarities',grad)
combine=np.repeat(np.eye(4),2,axis=1)[:,:,None,None]; ed=affine('unsigned_edges',half,combine)
# HOG-like positive orientation surplus. A little spatial smoothing before
# subsampling makes contour cells insensitive to a one-pixel translation.
orientmix=np.eye(4)-.70*np.ones((4,4))/4
sel=relu('orientation_surplus',affine('orientation_competition',ed,orientmix[:,:,None,None]))
sm=np.outer([.25,.5,.25],[.25,.5,.25]); weights['edge_smoothing.W']=(np.eye(4)[:,:,None,None]*sm).astype('f');weights['edge_smoothing.b']=np.zeros(4,'f')
sel=layer('edge_smoothing','conv',sel,kernel=[3,3],stride=1,padding=1,out=4)
edge8=pool('orientation_cells',sel,(2,2),(4,4)); edge8=layer('orientation_norm','norm',edge8)
opp=affine('opponent_axes','image',.15*np.array([[.55,.45,-1],[-.90,1,-.10]])[:,:,None,None]);opp=pool('pooled_opponents',layer('color_constancy','norm',opp),2)
def blur(a,s):
    n=int(np.ceil(2*s));g=np.exp(-.5*(np.arange(-n,n+1)/s)**2);g/=g.sum()
    for axis in [0,1]: a=np.apply_along_axis(lambda v:np.convolve(np.pad(v,n,mode='edge'),g,'valid'),axis,a)
    return a
shape_funcs=[plane,vehicle,birdfrog,pet,ungulate,pet,birdfrog,ungulate,boat,vehicle]
pose_counts=[4,9,5,7,4,7,4,4,4,10]
# Divide the bank into equally sized groups per class for pooling. Variations
# are deterministically sampled from general pose/scale/translation ranges.
rng=np.random.default_rng(7611); yy,xx=np.mgrid[0:32,0:32]; xx=xx-15.5;yy=yy-15.5
nper=960
HW=[];LW=[];CW=[]
for k in range(10):
    for j in range(nper):
        p=j%pose_counts[k];nom=[14,17,14,18,17,18,17,17,16,17][k]
        if k in [1,9] and p==2:nom=15
        if k in [3,5] and p in [2,3]:nom=13
        if k==8 and p==3:nom=14
        if k in [4,7] and p==3:nom=17
        scale=nom*np.exp(rng.uniform(-.34,.25)); aspect=np.exp(rng.uniform(-.20,.20))
        angle=rng.uniform(-1.4,1.4) if k==0 and p in (1,3) else rng.uniform(-.16,.16) if k in (1,9,8,4,7) else rng.uniform(-.26,.26)
        dx,dy=rng.uniform(-3.8,3.8,2); ca,sa=math.cos(angle),math.sin(angle)
        u=(ca*(xx-dx)+sa*(yy-dy))/scale * rng.choice([-1,1]);v=(ca*(yy-dy)-sa*(xx-dx))/(scale*aspect)
        d=Drawing(u,v)
        if (k==1 and p>=3) or (k==9 and p>=4):three_quarter(d,k,p)
        elif (k in [3,5] and p>=5) or (k==2 and p==4) or (k==6 and p==3):extra_animal(d,k,p)
        elif k in (0,8):shape_funcs[k](d,p)
        else:shape_funcs[k](d,k,p)
        z=d.z
        # Allow either polarity for the outline; glass, tire, eye & nose
        # contrast does not invert together with object vs. background.
        solid=(z!=0); interior=np.where(solid,z,0)
        interior=interior-solid*(interior.sum()/max(1,solid.sum()))
        pic=solid*rng.choice([-.68,.78])+interior*rng.uniform(.70,1.15)
        l=pic.reshape(16,2,16,2).mean((1,3));l-=l.mean();l/=max(.08,np.sqrt((l*l).mean()))
        # Local, center-surround color agreement: grass vs warm fur, sand vs
        # olive skin, silver wings vs blue sky. Also allow noncanonical hues.
        chrom=solid.reshape(16,2,16,2).mean((1,3));chrom=chrom-.75*blur(chrom,2)-.25*chrom.mean();chrom-=chrom.mean();chrom/=max(.065,np.sqrt((chrom*chrom).mean()))
        common=[(.9,-.2),(.3,-.8),(.45,-.85),(.65,-.55),(.25,-1),(.8,-.45),(.4,.90),(.55,-.95),(.95,-.1),(.32,-.75)][k]
        if j%3==0: ang=rng.uniform(-math.pi,math.pi);hue=[math.cos(ang),math.sin(ang)]
        else:hue=common
        CW.append((np.array(hue)[:,None,None]*chrom).ravel()*(.15/chrom.size))
        # Analytic projected edge strengths, rectified in the same opponent
        # orientation basis as the image. Smooth gradient directions, not
        # intensities, across each cell so the two edges of a thin wing survive.
        smooth=(pic+np.roll(pic,1,0)+np.roll(pic,-1,0)+np.roll(pic,1,1)+np.roll(pic,-1,1))*.2
        gu=(smooth[1:-1,2:]-smooth[1:-1,:-2])*.50;gv=(smooth[2:,1:-1]-smooth[:-2,1:-1])*.50
        t=np.stack([abs(gu*np.cos(a)+gv*np.sin(a)) for a in angles]);t=np.maximum(t-.70*t.mean(0),0)
        t=np.pad(t,((0,0),(1,1),(1,1)),mode='edge').reshape(4,8,4,8,4).mean((2,4))
        t-=t.mean((1,2),keepdims=True);t-=.50*t.mean(0,keepdims=True);t/=max(.006,np.sqrt((t*t).mean()))
        HW.append(t.ravel()*(.58/t.size));LW.append(l.ravel()*(.37/l.size))
sh=affine('contour_matches',edge8,np.array(HW));sl=affine('tone_matches',small,np.array(LW));sc=affine('opponent_matches',opp,np.array(CW)); matches=add('joint_matches',sh,sl,sc)
matches=reshape('pose_scores',matches,10,1,nper)
mx=pool('best_drawing',matches,(1,nper),op='maxpool')
# A complementary mean of sub-bank maxima is less vulnerable to one accidental alignment.
sub=pool('pose_groups',matches,(1,nper//8),op='maxpool');mean=pool('pose_consensus',sub,(1,8))
shape=add('shape_evidence',affine('strong_shape',mx,np.eye(10)*3.7),affine('consistent_shape',mean,np.eye(10)*2.0))

# Color maps. Approximate interval/halfspace indicators with clipped ramps;
# conjunction is a positive-part AND, so white / gray are not classified as blue sky.
color_src=pool('color16','image',2)
ramps=[]; rb=[]; colors=[]; cw=[]; cb=[]
def color(name,*tests):
    weights_here={}
    for vec,lo,hi in tests:
        # Intersection of halfspaces: subtract only the shortfall from each
        # passing test. This equals the two-hinge ramp AND after final ReLU.
        slope=-np.array(vec)/(hi-lo);i=len(rb);ramps.append(slope);rb.append(hi/(hi-lo));weights_here[i]=-1
    colors.append(name);cw.append(weights_here);cb.append(1)
RG=[1,-1,0]; GR=[-1,1,0]; GB=[0,1,-1]; BG=[0,-1,1]; BR=[-1,0,1]; RB=[1,0,-1]; R=[1,0,0];G=[0,1,0];B=[0,0,1]
color('blue',(BR,.01,.14),(BG,-.06,.045))
color('sky',(BR,.025,.17),(B,.46,.69),(GR,-.045,.025))
color('leaf',(GR,-.032,.045),(GB,.038,.14))
color('olive',(GR,-.095,-.02),(GB,.035,.14),([0,0,-1],-.68,-.45))
color('brown',(RG,.033,.123),(GB,.015,.09),([-1,0,0],-.90,-.62))
color('tan',(RG,-.005,.085),(GB,.038,.15),(R,.46,.65))
color('rust',(RG,.065,.18),(RB,.10,.24))
color('flash',(RG,.14,.30),(RB,.12,.27));color('flashblue',(BR,.12,.29),(BG,.08,.20))
color('gold',(RB,.19,.36),(GB,.13,.24),(R,.48,.70))
color('ink',(-lum,-.30,-.14));color('white',(R,.63,.82),(G,.63,.82),(B,.63,.82))
color('neutral',(RG,-.12,-.045),(GR,-.12,-.045),(BG,-.12,-.045),(GB,-.12,-.045))
color('slate',(BR,-.04,.075),(GB,-.15,-.04),(-lum,-.69,-.47))
color('mud',(RB,.03,.15),(-lum,-.51,-.29),(RG,-.05,.055))
color('green',(GR,.032,.13),(GB,.065,.16))
color('light',(lum,.48,.75));color('dark',(-lum,-.44,-.22))
ram=relu('color_ramps',affine('color_thresholds',color_src,np.array(ramps)[:,:,None,None],rb));mat=np.zeros((len(colors),len(rb)))
for i,row in enumerate(cw):
    for j,v in row.items():mat[i,j]=v
cm=relu('color_memberships',affine('color_intersections',ram,mat[:,:,None,None],cb));cm=pool('color_cells',cm,2)

# One vector of scalar measurements, with each entry defined as a sum of
# spatial-filter outputs. Coordinates in this section describe generic regions
# (sky above, verge below, subject center), never individual photographs.
feat=[]; contrib={}; widths={}; biases=[]
def feature(name,*terms,bias=0):
    i=len(feat);feat.append(name);biases.append(bias)
    for src,row in terms:
        row=np.asarray(row).ravel();widths[src]=len(row);contrib.setdefault(src,[]).append((i,row))
    return name
ya,xa=np.mgrid[0:8,0:8]; xx=(xa+.5)/4-1; yy=(ya+.5)/4-1
regions={'all':np.ones((8,8)), 'top':yy<-.43, 'bot':yy>.43, 'mid':np.exp(-2.8*xx**2-6*yy**2), 'rim':(abs(xx)>.68)|(abs(yy)>.71),'side':(abs(xx)>.60)&(yy<.50),'upper':yy<-.68,'foot':yy>.69}
for key in regions:regions[key]=regions[key]/regions[key].sum()
for i,c in enumerate(colors):
    for region,m in regions.items():
        row=np.zeros((len(colors),8,8));row[i]=m;feature(c+'_'+region,(cm,row))

# Texture relative to *this photo's* contrast; measure adjacent and medium-
# range variation separately. Unusual strong thin lines alone shouldn't count
# as a whole field of texture: clip their contribution.
clip=relu('over_edge_cap',affine('edge_cap_offset',ed,np.eye(4)[:,:,None,None],-.95))
fine=add('bounded_edges',ed,affine('subtract_excess',clip,-np.eye(4)[:,:,None,None]));fine=pool('texture_cells',fine,2,4)
# Long straight edges: oriented summation BEFORE rectification (cancels
# disordered changes of polarity). Sample overlapping windows on the grid.
longk=[];yg,xg=np.mgrid[-4:5,-4:5]
for a in angles:
    perp=xg*np.cos(a)+yg*np.sin(a);tang=yg*np.cos(a)-xg*np.sin(a)
    h=perp*np.exp(-.5*(perp/1.)**2-.5*(tang/3.2)**2); h/=abs(h).sum()*.5;longk.extend([h,-h])
# valid, stride two -> 12x12
weights['long_filters.W']=np.array(longk,np.float32)[:,None];weights['long_filters.b']=np.zeros(8,np.float32)
le=layer('long_filters','conv',unit,out=8,kernel=[9,9],stride=2,padding=0);le=affine('unsigned_long',relu('line_polarities',le),combine)
lc=relu('over_line_cap',affine('line_cap_offset',le,np.eye(4)[:,:,None,None],-1.65));le=add('bounded_lines',le,affine('subtract_line_excess',lc,-np.eye(4)[:,:,None,None]));le=pool('line_cells',le,2)
for r,m in regions.items():
    feature('tex_'+r,(fine,np.broadcast_to(m/4,(4,8,8))))
    for ch,name in [(0,'vert'),(2,'hor')]:
        row=np.zeros((4,8,8));row[ch]=m;feature(name+'_'+r,(fine,row))
for tag,m in [('all',np.ones((6,6))),('mid',np.exp(-((np.arange(6)[:,None]-2.5)/1.8)**2)*np.ones((6,6))),('top',np.broadcast_to((np.arange(6)[:,None]<2),(6,6))),('bot',np.broadcast_to((np.arange(6)[:,None]>=4),(6,6)))]:
    m=m/m.sum()
    for i,axis in [(0,'v'),(2,'h'),(None,'d')]:
        w=np.zeros((4,6,6))
        if i is None:w[1]=w[3]=m*.5
        else:w[i]=m
        feature('line'+axis+'_'+tag,(le,w))
# An achromatic highway shadow produces a dark band underneath a lighter
# cabin; boat decks and hulls frequently have a similar contrast.
light8=pool('regional_light',small,2)
for r,m in regions.items():feature('lum_'+r,(light8,m))
for name,ypos,band in [('under',.40,.20),('belt',.0,.24),('roof',-.35,.22)]:
    m=np.exp(-((yy-ypos)/band)**2-1.6*xx**2);feature(name,(light8,m/m.sum()))
# Bilateral symmetry at modest shifts. The score uses differences in luminance
# and in contour orientation. Align mirror-symmetric eyes, grille, nose, ears.
symm=[];picks=[]
for mid in [6.5,7.5,8.5]:
    start=len(symm)
    for by in range(3,13,2):
        for d in [1.5,3.5,5.5]:
            row=np.zeros((16,16));left,right=int(mid-d),int(mid+d)
            row[by:by+2,left:left+2]=.25;row[by:by+2,right-1:right+1]=-.25
            symm.extend([row.ravel(),-row.ravel()])
    picks.append(slice(start,len(symm)))
sy=relu('abs_mirror',affine('mirror_pairs',small,np.array(symm)))
for i,s in enumerate(picks):
    w=np.zeros(len(symm));w[s]=2/(s.stop-s.start);feature('asym'+str(i),(sy,w))

# Specialized paired wheel model: annular tire signal with a brighter hub,
# brighter pavement below and a common lower edge. Different track widths and
# perspectives have equal status. It is a *part* cue, independent of bodywork.
yw,xw=np.mgrid[0:16,0:16];xw=xw*2+1;yw=yw*2+1
pairs=[];lineparts=[]
def avgmask(mask):return mask/(mask.sum()+1e-12)
for y0 in [18,21,24,27]:
    for cx in [12,16,20]:
        for sep in [9,14,19,23]:
            for sl in [-.17,0,.17]:
                for rad in [2,3.3]:
                    w=0
                    for side in [-1,1]:
                        xx0=cx+sep*side/2; yy0=y0+sep*side*sl/2
                        ds=((xw-xx0)/(rad*.82))**2+((yw-yy0)/rad)**2
                        tire=avgmask(np.exp(-((np.sqrt(ds)-.73)/.4)**2));hub=avgmask(np.exp(-ds/.24));pave=avgmask(np.exp(-((xw-xx0)/(rad*1.2))**2-((yw-yy0-1.50*rad)/rad)**2));body=avgmask(np.exp(-((xw-xx0)/(rad*1.1))**2-((yw-yy0+1.46*rad)/rad)**2))
                        w=w+(.12*hub+.40*pave+.48*body-tire)*.5
                    pairs.append(w.ravel())
wheel=affine('two_tires',small,np.array(pairs));wheel=reshape('tire_poses',wheel,1,1,len(pairs));wheel=pool('strong_tires',wheel,(1,len(pairs)),op='maxpool');feature('wheels',(wheel,[1]))
# Two separate front parts for domestic mammals. Require both eyes AND a
# dark nose, not merely a high total response at three positions. Distinguish
# the small nose from a dog's broader black nose pad below the eyes.
for petname in ['feline','canine']:
    parts=[]
    for y0 in [8,12,16,20]:
        for cx in [11,14.5,18,21]:
            for sep in [5,8.5,12,15]:
                for tilt in [-.18,.12]:
                    rad=sep*(.17 if petname=='canine' else .16)
                    for j in [-1,1,0]:
                        x0=cx+j*sep/2; y1=y0+j*sep/2*tilt
                        rx=rad;ry=rad*(.70 if petname=='canine' else .49)
                        if j==0:
                            y1+=sep*(.77 if petname=='canine' else .54);rx=sep*(.24 if petname=='canine' else .14);ry=rx*.74;x0-=(y1-y0)*tilt
                        dd=((xw-x0)/rx)**2+((yw-y1)/ry)**2
                        if dd.min()>10: dd=dd-dd.min()
                        center=avgmask(np.exp(-dd));surround=avgmask(np.exp(-dd*.20));parts.append((surround-center).ravel())
    signals=affine(petname+'_spots',small,np.array(parts)); ns=len(parts)//3
    # s = min(t_i) implemented by max pooling over the negated spot scores
    signals=reshape(petname+'_triplets',signals,1,ns,3); neg=affine(petname+'_flip',signals,np.array([[[[-1.]]]]));worst=pool(petname+'_weakest',neg,(1,3),op='maxpool');worst=affine(petname+'_part_min',worst,np.array([[[[-1.]]]]));best=pool(petname+'_best_align',worst,(ns,1),op='maxpool');feature(petname,(best,[1]))

# Context smoothness, color differences and within-strip variations. For
# example, water and sky often differ on opposite sides of a horizontal hull;
# a flying object instead has nearly the same background in both places.
raw8=pool('scene_rgb',color_src,2);D=[]; chunk=[]
for nm,ra,rr in [('tb','top','bot'),('mr','mid','rim')]:
    start=len(D)
    for c in range(3):
        row=np.zeros((3,8,8));row[c]=regions[ra]-regions[rr];D.extend([row.ravel(),-row.ravel()])
    chunk.append((nm,start,len(D)))
for nm,area in [('topspread',[(0,0),(0,3),(0,6),(1,1),(1,5)]),('rimspread',[(0,0),(0,3),(0,6),(3,0),(3,6),(6,0),(6,3),(6,6)]),('botspread',[(6,0),(6,3),(6,6),(5,1),(5,5)])]:
    start=len(D)
    for idx,(a,b) in enumerate(area):
        for c in range(3):
            for e1,e2 in area[idx+1:]:
                row=np.zeros((3,8,8));row[c,a:a+2,b:b+2]=.25;row[c,e1:e1+2,e2:e2+2]-=.25;D.extend([row.ravel(),-row.ravel()])
    chunk.append((nm,start,len(D)))
diffs=relu('scene_color_differences',affine('compare_regions',raw8,np.array(D)))
for nm,start,end in chunk:
    w=np.zeros(len(D));w[start:end]=2/(end-start);feature(nm,(diffs,w))
# Bars at the centers of quadruped shins, of either polarity, in pairs. Search
# for legs joined by a torso but separated by a patch of visible ground.
rng2=np.random.default_rng(881); sk=[]
for i in range(768):
    xc=rng2.uniform(11.5,20.5);sep=rng2.uniform(7,18); ytop=rng2.uniform(13.5,21);length=rng2.uniform(5.5,11);slant=rng2.uniform(-.25,.25)
    body=avgmask(np.exp(-((xw-xc)/(.66*sep))**4-((yw-ytop+2)/3.2)**2));gap=avgmask(np.exp(-((xw-xc)/(.32*sep))**4-((yw-ytop-length*.58)/(length*.37))**4));
    row=.42*(body-gap)
    for s in [-1,1]:
        tilt=slant+rng2.uniform(-.18,.18);x0=xc+s*sep/2;dx=xw-(x0+(yw-ytop)*tilt); wy=np.exp(-((yw-ytop-length*.5)/(length*.53))**6);thick=rng2.uniform(.9,1.6)
        leg=avgmask(wy*np.exp(-(dx/thick)**2));neg1=avgmask(wy*np.exp(-((dx-2.5*thick)/thick)**2));neg2=avgmask(wy*np.exp(-((dx+2.5*thick)/thick)**2))
        row+=.29*(leg-.50*(neg1+neg2))
    sk.append(row.ravel()*rng2.choice([-1,1]))
legs=reshape('leg_poses',affine('two_legs_and_torso',small,np.array(sk)),1,1,len(sk));legs=pool('best_leg_alignment',legs,(1,len(sk)),op='maxpool');feature('legs',(legs,[1]))
# A hull has a long, approximately level waterline. Look for a long pair of
# contrasting strips with little difference between their left & right parts.
strip=[]
for i in range(180):
    xc=rng2.uniform(12,20);yc=rng2.uniform(16,24);slope=rng2.uniform(-.15,.15);length=rng2.uniform(13,26); dx=xw-xc;dy=yw-yc-dx*slope
    wt=np.exp(-(dx/(length*.5))**6);below=avgmask(wt*np.exp(-((dy-1.9)/1.7)**2));above=avgmask(wt*np.exp(-((dy+1.9)/1.7)**2));strip.extend([(above-below).ravel(),(below-above).ravel()])
wl=reshape('waterline_poses',affine('waterline_contrast',small,np.array(strip)),1,1,len(strip));wl=pool('best_waterline',wl,(1,len(strip)),op='maxpool');feature('hull',(wl,[1]))

# A second-order texture descriptor: short oriented bars, center-surround
# blobs and same-polarity neighbors. This distinguishes, e.g., a sharp chrome
# panel from low-contrast fur, rather than equating all high edge counts.
ys,xs=np.mgrid[-2:3,-2:3]; filters=[]
for sigma in [.67,1.35]:
    for a in angles:
        u=xs*np.cos(a)+ys*np.sin(a);v=ys*np.cos(a)-xs*np.sin(a)
        w=(1-(u/sigma)**2)*np.exp(-.5*(u/sigma)**2-.5*(v/(sigma*1.7))**2);w-=w.mean();w/=abs(w).sum()*.5;filters.extend([w,-w])
for sig in [.65,1.15]:
    w=np.exp(-.5*(xs**2+ys**2)/sig**2);w/=w.sum();wide=np.exp(-.5*(xs**2+ys**2)/(1.8*sig)**2);wide/=wide.sum();w-=wide;w/=abs(w).sum()*.5;filters.extend([w,-w])
weights['material_filters.W']=np.array(filters,np.float32)[:,None];weights['material_filters.b']=np.zeros(len(filters),np.float32)
material=layer('material_filters','conv',unit,out=len(filters),kernel=[5,5],stride=2,padding=0);material=relu('material_magnitudes',material);mc=pool('material_cells',material,2)
# Co-occurring responses at neighbors: min(a,b)=(a+b-|a-b|)/2.
# Continuation along an edge, and banding at a perpendicular displacement.
M=[];measure_masks=[]
for i in range(8):
    a=angles[(i//2)%4];vx=int(np.rint(np.cos(a)));vy=int(np.rint(np.sin(a)))
    for group,(dx,dy) in enumerate([(-vy,vx),(vx,vy)]):
        for step in [1,2]:
            w=np.zeros((20,5,5));w[i,2,2]=1;w[i,2+dy*step,2+dx*step]=-1;M.extend([w,-w]);measure_masks.extend([group,group])
md=relu('material_changes',affine('compare_bar_neighbors',material,np.array(M)));mc2=pool('neighbor_cells',md,2)
yt,xt=np.mgrid[0:7,0:7];xt=(xt-3)/3.5;yt=(yt-3)/3.5
for r,m in [('mid',np.exp(-2.8*xt**2-5*yt**2)),('top',yt<-.42),('bot',yt>.42),('rim',(abs(yt)>.62)|(abs(xt)>.62))]:
    m=m/m.sum()
    for nm,q in [('finebar',slice(0,8)),('medbar',slice(8,16)),('fineblob',slice(16,18)),('medblob',slice(18,20))]:
        w=np.zeros((20,7,7));w[q]=m*(2/len(w[q]));feature(nm+'_'+r,(mc,w))
for group,nm in [(0,'broken'),(1,'bandchange')]:
    w=np.zeros((len(M),5,5));ii=np.array(measure_masks)==group;w[ii]=2/(ii.sum()*25);feature(nm,(mc2,w))

# Generic stationary and spatial palette models. Each entry specifies three
# sets of *alternative* material hues (overhead, subject, ground). Distances
# to these color sets are evaluated pixel by pixel in a luminance/chroma
# space. This avoids counting a few blue or green pixels as an entire scene.
palette={
 'cloud':(.83,.86,.89),'snow':(.90,.90,.89),'mist':(.62,.65,.67),'slate':(.33,.37,.40),'gray':(.49,.49,.48),'dusk':(.22,.24,.26),'coal':(.12,.12,.12),
 'azure':(.40,.60,.80),'sky':(.58,.73,.83),'haze':(.71,.77,.80),'navy':(.18,.30,.43),'sea':(.33,.48,.54),'aqua':(.39,.65,.66),'wave':(.52,.61,.65),
 'forest':(.18,.25,.15),'leaf':(.32,.43,.22),'lawn':(.48,.57,.32),'sage':(.54,.57,.43),'moss':(.38,.41,.25),'hay':(.68,.61,.40),'bark':(.33,.29,.23),'sand':(.72,.66,.54),
 'earth':(.49,.39,.27),'chestnut':(.48,.29,.17),'golden':(.68,.51,.32),'cream':(.81,.75,.64),'tan':(.60,.53,.41),'warmgray':(.53,.50,.46),'wood':(.40,.33,.28),
 'red':(.66,.18,.14),'bluepaint':(.15,.28,.51),'olive':(.44,.46,.23),'lime':(.45,.61,.20),'orange':(.80,.45,.16),'metal':(.66,.67,.65)}
names=list(palette); axes=np.stack([lum,np.array([1.,-1,0]),np.array([.5,.5,-1])]);w=np.concatenate([axes,-axes]);PW=[];Pb=[]
for rgb in palette.values():PW.extend(w);Pb.extend(-w@rgb)
# Average colors over 4x4 cells before computing palette distances. A tiny
# checkerboard of blue and yellow should read as desaturated, not as green.
rgb8=pool('palette_grid','image',4);pd=relu('palette_axis_errors',affine('palette_axis_offsets',rgb8,np.array(PW)[:,:,None,None],Pb)); nw=len(names);w=np.zeros((nw,6*nw));
for i in range(nw):w[i,6*i:6*i+6]=[1.15,3.30,2.7,1.15,3.30,2.7]
pd=affine('palette_distances',pd,w[:,:,None,None]);
# Nearest of the *named colors* in a material mixture. All color coordinates
# above are conventional hues, not measured colors from photographs.
mats={
 'sky':'azure sky haze cloud', 'pale':'snow cloud mist haze', 'water':'navy sea wave aqua', 'harbor':'slate wave mist wood', 'tarmac':'gray slate dusk mist', 'street':'gray bark warmgray forest',
 'foliage':'forest leaf moss sage', 'pasture':'lawn sage moss hay','dry':'sand hay earth tan','litter':'bark earth olive wood','home':'warmgray wood cream tan','colored':'red bluepaint snow coal',
 'silver':'mist slate metal coal','markings':'snow navy red metal','deer':'tan earth hay bark','chestnut':'chestnut golden coal tan','blackwhite':'snow coal warmgray cream','fur':'golden tan cream wood','tabby':'warmgray bark gray tan','amphib':'olive moss earth bark','vivid':'lime olive orange forest','plumage':'navy coal tan cream','paleplume':'cloud gray wood tan','freight':'metal snow mist warmgray'}
# Exact minimum via reshape and maxpool across adjacent channels, transposed
# into place by a sparse channel indexing 1x1 and an intermediate dense-free
# arrangement (channel major contiguous spatial dimensions).
w=np.zeros((len(mats)*4,nw));
for i,hues in enumerate(mats.values()):
    for j,h in enumerate(hues.split()):w[4*i+j,names.index(h)]=-1
idx=affine('material_palettes',pd,w[:,:,None,None]);idx=reshape('color_choices',idx,len(mats),4,64);best=pool('closest_color',idx,(4,1),op='maxpool');best=reshape('material_fit',best,len(mats),8,8)
for idx,matname in enumerate(mats):
    for reg in ['top','mid','bot','rim']:
        w=np.zeros((len(mats),8,8));w[idx]=regions[reg];feature('fit_'+matname+'_'+reg,(best,w))

# Project the individual branches into a common vector. The following DSL
# expresses continuous rules; its compilation is just linear algebra, not a fit.
branches=[]
for src,rows in contrib.items():
    a=np.zeros((len(feat),widths[src]));
    for i,w in rows:a[i]+=w
    branches.append(affine('measure_'+src,src,a))
measure=add('cue_vector',*branches)
class Expr:
    def __init__(self,coef,b=0):self.a=np.asarray(coef);self.b=b
    def __add__(self,o):
        if not isinstance(o,Expr):o=Expr(0,o)
        return Expr(self.a+o.a,self.b+o.b)
    __radd__=__add__
    def __mul__(self,k):return Expr(self.a*k,self.b*k)
    __rmul__=__mul__
    def __neg__(self):return self*-1
    def __sub__(self,o):return self+(-o)
    def __rsub__(self,o):return -self+o
F={key:Expr(np.eye(1,len(feat),i).ravel()) for i,key in enumerate(feat)}
def f(name,reg='all'):return F[name+'_'+reg]
rules=[]; rule_names=[]; columns=[]
def rule(name,measure,lo,hi,coeffs):
    # An increasing or decreasing saturating ramp, with zero influence below
    # lo. These table entries express semantic log-evidence, chosen by hand.
    scaled=(measure-lo)*(1/(hi-lo));rules.extend([scaled,scaled-1]);rule_names.append(name);columns.extend([np.asarray(coeffs),-np.asarray(coeffs)])
# Column order: plane car bird cat deer dog frog horse ship lorry.
rule('extensive blue background',f('blue','rim'),.09,.60,[.75,-.14,.20,-.17,-.49,-.18,-.43,-.18,.78,.12])
rule('blue above subject',f('sky','top'),.10,.64,[.62,.06,.02,-.33,-.40,-.30,-.36,.15,.56,.32])
rule('blue water below',f('blue','bot')+.35*f('slate','bot'),.10,.67,[.19,.01,.19,-.20,-.31,-.16,-.36,-.26,1.35,-.06])
rule('many sky colored pixels',f('sky','all'),.18,.61,[.75,-.12,.06,-.17,-.24,-.20,-.33,-.14,.15,-.14])
rule('pale upper background',f('white','top')+.20*F['lum_top'],.19,.74,[.48,-.05,.04,-.12,-.15,-.09,-.33,.12,.37,.26])
rule('empty sky',.60*f('light','rim')+.36*f('blue','rim')-1.8*f('tex','rim'),-.13,.48,[.76,-.17,.32,-.05,-.19,-.14,-.48,-.20,.37,-.26])
rule('bright above dark below',F['lum_top']-F['lum_bot'],.26,1.50,[.05,-.02,-.16,-.13,-.23,-.21,-.39,.10,.53,.41])
rule('green background',f('leaf','rim')+.27*f('olive','rim'),.10,.65,[-.40,-.38,.56,-.30,.97,.01,.32,.56,-.65,-.48])
rule('grass below',f('leaf','bot')+.25*f('olive','bot'),.10,.66,[-.22,-.42,.15,-.27,.85,.12,.18,1.00,-.63,-.47])
rule('green top and bottom',.5*f('leaf','top')+.5*f('leaf','bot')-.27*f('leaf','mid'),.08,.53,[-.17,-.10,.20,-.09,.44,-.02,-.12,.31,-.24,-.18])
rule('green subject',f('green','mid')-.48*f('green','rim'),.025,.35,[-.12,-.17,.40,-.18,-.33,-.20,1.21,-.36,-.23,.05])
rule('ochre undergrowth',f('olive','all')+.26*f('mud','all'),.18,.66,[-.45,-.39,.35,-.02,.65,-.18,.75,.10,-.53,-.48])
rule('warm ground',f('tan','bot')+.7*f('brown','bot'),.20,.84,[-.31,-.26,.10,.13,.37,.08,.40,.30,-.45,-.28])
rule('saturated chestnut coat',f('rust','mid')+.38*f('brown','mid')-.52*f('rust','rim'),.12,.68,[-.21,.07,.01,-.05,.15,.23,.13,.94,-.24,-.16])
rule('subdued brown coat',f('brown','mid')-.35*f('brown','rim'),.08,.50,[-.13,-.28,.16,.09,.51,.22,.33,.36,-.19,-.34])
rule('earth palette',.60*f('brown')+.65*f('mud')+.36*f('tan'),.19,.62,[-.38,-.38,.05,.27,.25,.19,.68,.10,-.48,-.45])
rule('neutral subject',f('neutral','mid')-.40*f('neutral','rim'),.27,.72,[.19,.22,.04,.32,-.35,.19,-.45,-.16,.29,.39])
rule('very white subject',f('white','mid')-.33*f('white','rim'),.11,.48,[.29,.19,.10,.34,-.42,.40,-.59,-.12,.46,.42])
rule('some black and white',f('white','mid')+.7*f('ink','mid'),.23,.58,[.10,.28,.00,.07,-.35,.26,-.48,-.05,.14,.36])
rule('painted surface',f('flash','mid')+f('flashblue','mid')-.29*f('flash','rim'),.042,.32,[.23,.94,.40,-.18,-.46,-.27,-.20,-.33,.20,.64])
rule('bright colored object',f('gold','mid')+.55*f('flashblue','mid'),.06,.42,[.11,.33,.59,-.19,-.21,-.15,.19,-.13,.11,.22])
rule('gray road',f('neutral','bot')-.49*f('white','bot')-.28*f('blue','bot')-.24*f('mud','bot'),.35,.85,[.10,.58,-.13,.07,-.27,.08,-.26,-.16,-.05,.64])
rule('rough background',f('tex','rim'),.22,.46,[-.61,-.26,.14,.20,.39,.13,.62,.12,-.43,-.11])
rule('rough foreground',f('tex','mid'),.29,.54,[-.28,-.01,.17,.38,-.06,.30,.61,-.14,-.20,.12])
rule('forest floor',f('tex','bot')+.17*f('olive','bot')+.12*f('brown','bot'),.31,.60,[-.45,-.34,.07,.20,.17,.22,.88,.08,-.44,-.22])
rule('smooth background rough center',f('tex','mid')-f('tex','rim'),.045,.33,[.19,.15,.72,-.05,-.03,-.05,-.11,-.18,.01,.18])
rule('featureless surround',f('tex','rim'),.28,.085,[.42,-.08,.57,.03,-.29,-.09,-.42,-.23,.29,-.29])
rule('horizontal scene geometry',f('lineh')-.87*f('lined'),.045,.21,[.01,.49,-.29,-.31,-.21,-.26,-.34,.04,.85,.58])
rule('vertical scene geometry',f('linev')-.89*f('lined'),.042,.17,[-.25,-.02,-.10,-.08,.26,-.10,-.22,.36,-.29,.72])
rule('horizontal subject edges',f('lineh','mid')-f('linev','mid'),.09,.42,[.07,.61,-.24,-.26,-.14,-.35,-.17,-.17,.66,.17])
rule('horizontal water ripples',f('hor','bot')-f('vert','bot'),.028,.18,[-.09,.13,-.13,-.18,-.23,-.18,-.21,-.17,.77,.09])
rule('isolated smooth top',f('tex','mid')-f('tex','top'),.12,.42,[.14,.03,.22,-.11,-.13,-.10,-.18,.09,.17,.12])
rule('dark underbody',.63*F['belt']+.37*F['roof']-F['under'],.22,.96,[.12,.73,-.16,-.09,-.15,-.11,-.18,-.04,.54,.65])
rule('light ground under chassis',F['lum_foot']-F['under'],.21,.99,[.15,.44,-.05,-.06,.15,-.08,-.11,.18,-.19,.33])
rule('wheels with road contact',F['wheels'],.49,1.35,[-.07,.78,-.11,-.09,-.02,-.08,-.15,.03,-.12,.68])
rule('paired eyes cat',F['feline'],.22,.84,[-.08,-.02,.00,.82,.00,.25,.24,-.07,-.04,-.03])
rule('eyes and broad nose',F['canine'],.20,.77,[-.08,-.03,-.04,.17,-.06,.95,.06,-.02,-.08,.00])
rule('frontal symmetry',(F['asym0']+F['asym1']+F['asym2'])*(1/3),.91,.44,[.06,.19,-.11,.46,-.18,.34,.03,-.15,-.12,.08])
rule('domestic setting',.46*f('neutral','rim')+.55*f('brown','rim')+.30*f('tan','rim')-.55*f('blue','rim')-.80*f('leaf','rim'),.23,.55,[-.18,.03,-.13,.56,-.21,.47,.01,-.15,-.16,-.11])
rule('dark irregular field',f('dark','rim')+.68*f('tex','rim')-.45*f('blue','rim'),.59,1.06,[-.31,-.19,.14,.16,.18,.14,.54,.03,-.27,-.20])
# A hazy / muted distant coat vs a close-up textured face.
rule('quiet tan subject',.55*f('tan','mid')+.35*f('leaf','rim')-.64*f('tex','mid'),-.09,.30,[-.06,-.13,.13,-.06,.45,.04,-.10,.09,-.13,-.14])
# List of additional propositions used inside composite scene descriptions.
def cue(name,x,lo,hi):rule(name,x,lo,hi,np.zeros(10))
cue('uniform perimeter',F['rimspread'],.19,.045)
cue('one background',F['tb'],.18,.035)
cue('isolated color',F['mr']-.70*F['rimspread'],.03,.18)
cue('vertical layers',F['tb'],.085,.30)
cue('level waterline',F['hull'],.75,1.87)
cue('straight legs',F['legs'],.54,1.43)
cue('daylight at top',f('light','top')+.25*f('sky','top')-.33*f('tex','top'),.29,.76)
cue('natural ground',f('tan','bot')+.8*f('leaf','bot')+.4*f('brown','bot'),.18,.64)
cue('no blue',f('blue','rim'),.23,.035)
cue('top smooth',f('tex','top'),.37,.105)
cue('bottom smooth',f('tex','foot'),.37,.105)
cue('structured top',f('tex','top')+.42*F['topspread'],.29,.54)
cue('rugged diagonals',f('lined','mid')-.36*f('lineh','mid')-.36*f('linev','mid'),.055,.17)
cue('soft contours',f('tex','mid'),.40,.22)
cue('dark eyes x nose',F['canine']-.48*F['feline'],.10,.46)
cue('triangular face',F['feline']-.48*F['canine'],.10,.45)
cue('pale belly',F['under']-.48*F['belt']-.52*F['roof'],.09,.74)
cue('warm sides',f('brown','side')+.50*f('tan','side')+.40*f('mud','side'),.18,.58)
cue('dark coat',f('ink','mid')-.43*f('ink','rim'),.08,.45)
cue('foot contrast',f('vert','bot')-f('hor','bot'),.025,.13)
cue('warm skin',f('mud','mid')+.55*f('olive','mid')+.30*f('tan','mid'),.23,.70)
cue('silvery flanks',f('neutral','mid')+.6*f('slate','mid')-.50*f('neutral','rim'),.30,.76)
cue('dark leg gap',F['legs']-.24*F['hull'],.48,1.19)
# Many rules (e.g. ground and sky color) are correlated observations, not
# independent likelihoods. Limit their additive vote. Most context evidence
# instead comes from a disjunction of conjunctions: any one plausible setting
# is sufficient, with both mandatory and exclusionary cues for that setting.
r=relu('firing_strengths',affine('soft_rule_tests',measure,np.stack([v.a for v in rules]),[v.b for v in rules]));
context=affine('scene_and_part_votes',r,.26*np.array(columns).T)
models=[
# plane: one surround behind wings, or runway and air above, or banked plane
[['uniform perimeter','one background','extensive blue background'],['top smooth','bottom smooth','one background','isolated color'],['blue above subject','daylight at top','grass below','smooth background rough center'],['many sky colored pixels','uniform perimeter'],['pale upper background','one background','rugged diagonals','smooth background rough center'],['empty sky','isolated color','one background']],
# passenger car: paint + glass over pavement; frontal and side views
[['painted surface','gray road','wheels with road contact'],['gray road','dark underbody','structured top','wheels with road contact'],['silvery flanks','horizontal subject edges','wheels with road contact','structured top'],['frontal symmetry','gray road','structured top','some black and white'],['painted surface','horizontal subject edges','light ground under chassis'],['silvery flanks','dark underbody','light ground under chassis','no blue']],
# bird: perched against foliage, solitary bird in sky, bright plumage
[['smooth background rough center','green background','isolated color'],['featureless surround','one background','isolated color'],['green background','bright colored object','rugged diagonals'],['quiet tan subject','smooth background rough center','one background'],['extensive blue background','rugged diagonals','pale belly','one background'],['green subject','featureless surround','smooth background rough center']],
# cat: face, reclined fluffy body or tabby against domestic interior
[['domestic setting','paired eyes cat','triangular face'],['neutral subject','frontal symmetry','triangular face','no blue'],['warm sides','rough foreground','triangular face'],['domestic setting','soft contours','pale belly'],['domestic setting','subdued brown coat','rugged diagonals','paired eyes cat'],['very white subject','paired eyes cat','one background','no blue']],
# deer: slim legs in foliage, often inconspicuous, tawny
[['green background','grass below','subdued brown coat'],['natural ground','straight legs','quiet tan subject'],['green top and bottom','no blue','straight legs'],['warm ground','subdued brown coat','rough background','soft contours'],['ochre undergrowth','quiet tan subject','green background'],['dark leg gap','green background','natural ground']],
# dog: long muzzle, ground or home setting, many coat colors
[['eyes and broad nose','dark eyes x nose','frontal symmetry'],['very white subject','natural ground','eyes and broad nose'],['saturated chestnut coat','eyes and broad nose','soft contours'],['domestic setting','eyes and broad nose','some black and white'],['dark coat','grass below','pale belly','eyes and broad nose'],['dark eyes x nose','warm sides','rough foreground']],
# frog: close to ground, diffuse earth colored field OR vivid green skin
[['forest floor','warm skin','rough foreground'],['earth palette','one background','warm skin'],['green subject','rough foreground','one background'],['warm ground','warm sides','one background','paired eyes cat'],['dark irregular field','warm skin','rugged diagonals'],['ochre undergrowth','warm skin','rough foreground']],
# horse: chestnut or dark horse against grass, open paddock, sky and legs
[['straight legs','natural ground','saturated chestnut coat'],['blue above subject','grass below','straight legs'],['daylight at top','natural ground','dark leg gap'],['dark coat','grass below','straight legs'],['saturated chestnut coat','grass below','vertical scene geometry'],['warm ground','daylight at top','straight legs','foot contrast']],
# ship: horizon + hull breaking the water + sea below; sky above is optional
[['blue water below','vertical layers','level waterline'],['gray road','horizontal water ripples','level waterline','daylight at top'],['very white subject','blue water below','bright above dark below'],['blue above subject','level waterline','horizontal water ripples'],['bottom smooth','vertical layers','level waterline','very white subject'],['blue water below','bright above dark below','dark underbody']],
# truck: broad body with vertical and horizontal lines over road, not any bright object
[['vertical scene geometry','dark underbody','gray road'],['pale upper background','wheels with road contact','vertical scene geometry','daylight at top'],['very white subject','vertical scene geometry','gray road'],['painted surface','wheels with road contact','daylight at top','vertical scene geometry'],['very white subject','gray road','horizontal scene geometry','wheels with road contact'],['painted surface','dark underbody','frontal symmetry','daylight at top']]]
conj=[]
for options in models:
    for terms in options:
        row=np.zeros(len(rules));
        for term in terms:
            ix=rule_names.index(term);row[2*ix]+=1/len(terms);row[2*ix+1]-=1/len(terms)
        conj.append(row)
complexes=affine('scene_descriptions',r,np.array(conj));poses=reshape('scene_alternatives',complexes,10,1,6);alternative=pool('best_setting',poses,(1,6),op='maxpool');alternatives=pool('other_settings',poses,(1,6));joint=add('scene_likelihood',affine('setting_score',alternative,np.eye(10)*2.6),affine('setting_consistency',alternatives,np.eye(10)*.8))
# Three-zone generative scene hypotheses, equally many per class. Keep a
# mean of the two/three local material costs rather than letting a single
# excellent color anywhere dominate. Hedges, road, grass, water and sky are
# background materials, whereas fur, feather, tire and paint are foreground.
scenes=[
 ['sky markings sky','pale silver pale','sky silver pasture','pale markings tarmac','sky colored sky','foliage markings pasture','sky silver sky','pale silver tarmac'],
 ['street silver tarmac','street colored tarmac','foliage silver tarmac','home colored street','tarmac silver tarmac','pale blackwhite tarmac','sky colored tarmac','street blackwhite street'],
 ['foliage plumage foliage','pasture paleplume pasture','pale plumage pale','sky paleplume sky','foliage vivid pasture','dry plumage dry','water paleplume water','foliage paleplume litter'],
 ['home tabby home','home blackwhite home','home fur home','street tabby home','dry fur home','home tabby dry','pasture blackwhite pasture','pale tabby pale'],
 ['foliage deer pasture','pasture deer pasture','foliage deer dry','dry deer dry','foliage deer litter','pasture deer litter','pale deer pale','foliage deer foliage'],
 ['home fur home','foliage blackwhite pasture','street fur street','pasture fur pasture','home blackwhite home','pale blackwhite home','dry fur dry','home chestnut home'],
 ['litter amphib litter','foliage vivid foliage','dry amphib dry','litter vivid litter','home amphib litter','foliage amphib litter','water amphib water','dry amphib litter'],
 ['sky chestnut pasture','foliage chestnut pasture','pale chestnut dry','sky blackwhite pasture','dry chestnut dry','pasture blackwhite pasture','foliage chestnut dry','pale blackwhite dry'],
 ['sky freight water','pale silver water','sky markings harbor','pale markings harbor','foliage freight water','pale silver harbor','sky markings water','harbor silver water'],
 ['sky freight tarmac','pale colored tarmac','foliage freight street','sky colored street','pale freight tarmac','street colored street','sky freight pasture','street silver tarmac']]
sceneRows=[]
for opts in scenes:
    for text in opts:
        a,b,c=text.split(); v=.30*f('fit_'+a,'top')+.43*f('fit_'+b,'mid')+.27*f('fit_'+c,'bot');sceneRows.append(v.a)
matcost=reshape('color_hypotheses',affine('zone_palettes',measure,np.stack(sceneRows)),10,1,8)
mb=pool('plausible_palette',matcost,(1,8),op='maxpool');mtyp=pool('typical_palette',matcost,(1,8));materialScore=add('palette_evidence',affine('palette_match',mb,np.eye(10)*3.4),affine('palette_support',mtyp,np.eye(10)*.9))
# Scale relationships rather than a single texture amplitude. Fine bars that
# do not persist over distance indicate fur/feathers/skin. Lines with strong
# low spatial frequency content and continuity indicate painted structures.
base=f('finebar','mid'); med=f('medbar','mid'); spots=f('fineblob','mid');bigspot=f('medblob','mid');broken=F['broken'];change=F['bandchange']
textures=[];tv=[]
def texture(x,lo,hi,coeff):
    t=(x-lo)*(1/(hi-lo));textures.extend([t,t-1]);tv.extend([coeff,-np.array(coeff)])
texture(broken-.82*change,-.018,.028,[-.19,-.49,.25,.38,.07,.26,.52,-.02,-.36,-.47])
texture(base-.69*med,-.022,.069,[-.35,-.34,.14,.52,.09,.18,.62,-.30,-.31,-.43])
texture(spots-.57*bigspot,-.025,.072,[-.18,-.27,.10,.18,.02,.09,.63,-.18,-.28,-.38])
texture(f('fineblob','rim')-.45*f('medblob','rim'),-.012,.086,[-.31,-.06,.04,.06,.29,.03,.55,.02,-.33,-.13])
texture(med-.70*base,.14,.35,[.15,.39,-.11,-.23,-.19,-.10,-.26,.03,.21,.50])
texture(f('finebar','mid')-f('finebar','rim'),.037,.16,[.10,.02,.50,.12,-.04,.26,-.13,-.08,-.03,.05])
texture(f('finebar','bot')-f('finebar','top'),.0,.18,[.02,-.25,-.07,-.11,.07,-.12,.00,.37,.21,.02])
tr=relu('texture_rules',affine('texture_scales',measure,np.stack([v.a for v in textures]),[v.b for v in textures]));tx=affine('texture_evidence',tr,np.array(tv).T*.82)
add('class_scores',shape,context,joint,materialScore,tx)

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
(out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
np.savez(out/'weights.npz',**weights)