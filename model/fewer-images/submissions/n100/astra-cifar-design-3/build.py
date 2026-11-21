"""Build an image classifier from geometric figure drawings and color/texture rules.
No input files; all lengths in the drawings are generic object-local coordinates.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

LAYERS = []
WEIGHTS = {}

def layer(name, op, src, **kw):
    LAYERS.append(dict(name=name, op=op, inputs=[src] if isinstance(src,str) else src, **kw))
    return name

def affine(name, src, w, b=0, stride=1, padding=0):
    w=np.array(w, dtype='f'); b=np.broadcast_to(b,(len(w),)).astype('f')
    WEIGHTS[name+'.W']=w; WEIGHTS[name+'.b']=b
    if w.ndim==2: return layer(name,'dense',src,out=len(w))
    return layer(name,'conv',src,out=len(w),kernel=list(w.shape[2:]),stride=stride,padding=padding)

def pool(name,src,size,stride=None,maximum=False):
    if isinstance(size,int): size=[size,size]
    if stride is None: stride=size
    if isinstance(stride,int): stride=[stride,stride]
    return layer(name,'maxpool' if maximum else 'avgpool',src,size=size,stride=stride)

def relu(name,src): return layer(name,'relu',src)
def reshape(name,src,shape): return layer(name,'reshape',src,shape=list(shape))

# Inverse projection of a subpixel grid into the local coordinates of the figure.
# Most models occupy approximately [-1,1] x [-1,1]. Features can occlude others.
class Canvas:
    def __init__(self,x,y,bg):
        self.x=x; self.y=y
        self.a=np.array(np.broadcast_to(bg,x.shape),copy=True,dtype='f')
    def ell(self,x,y,rx,ry,v,angle=0):
        X=self.x-x; Y=self.y-y; c=math.cos(angle); s=math.sin(angle)
        test=((c*X+s*Y)/rx)**2+((c*Y-s*X)/ry)**2<=1
        self.a[test]=v
    def poly(self,pts,v):
        inside=np.zeros_like(self.x,dtype=bool)
        x0,y0=pts[-1]
        for x1,y1 in pts:
            if y1!=y0:
                inside ^= ((y0>self.y)!=(y1>self.y)) & (self.x < (x1-x0)*(self.y-y0)/(y1-y0)+x0)
            x0,y0=x1,y1
        self.a[inside]=v
    def box(self,x0,y0,x1,y1,v):
        self.a[(self.x>=x0)&(self.x<=x1)&(self.y>=y0)&(self.y<=y1)]=v
    def line(self,x1,y1,x2,y2,r,v):
        dx=x2-x1; dy=y2-y1
        t=np.clip(((self.x-x1)*dx+(self.y-y1)*dy)/(dx*dx+dy*dy+1.e-10),0,1)
        self.a[(self.x-x1-t*dx)**2+(self.y-y1-t*dy)**2<=r*r]=v

# The following are independent templates, not training samples for a readout.
# The only image-dependent math at build time is a descriptor of one drawing.
def airplane(d,R,m):
    U=R.uniform; body=U(.35,.95) if R.random()<.55 else U(.04,.42)
    trim=body*U(.35,.77); lit=min(1,body+U(.04,.2))
    if m==0: # airliner seen broadside
        d.poly([(-.85,.02),(-.96,-.62),(-.79,-.55),(-.53,-.1),(.58,-.10),(.84,-.04),(.99,.06),(.91,.15),(.57,.23),(-.58,.19),(-.98,.09)],body)
        d.poly([(-.44,.09),(-.64,-.36),(-.53,-.33),(.14,.06)],trim)
        d.poly([(-.08,.12),(-.45,.62),(-.30,.59),(.41,.16)],lit)
        d.ell(.13,.23,.24,.105,trim)
        d.poly([(.59,-.095),(.72,-.072),(.84,-.015),(.61,-.024)],trim*.7)
        d.line(-.53,.021,.52,.018,.021,trim)
    elif m==1: # swept wings from above/below
        d.poly([(.99,0),(.79,-.085),(.24,-.105),(-.29,-.94),(-.44,-.99),(-.15,-.098),(-.69,-.061),(-.91,-.38),(-.99,-.41),(-.87,0),(-.99,.41),(-.91,.38),(-.69,.061),(-.15,.098),(-.44,.99),(-.29,.94),(.24,.105),(.79,.085)],body)
        for side in [-1,1]: d.ell(.025,.39*side,.14,.054,trim)
        d.ell(.51,0,.28,.055,lit)
    elif m==2: # oblique view, with one wing pointing toward the camera
        d.poly([(-.92,-.32),(-.99,-.91),(-.85,-.80),(-.61,-.30),(.36,.01),(.64,.17),(.70,.28),(.58,.39),(.39,.35),(-.72,-.10),(-1,-.23)],body)
        d.poly([(.21,.12),(.25,-.65),(.13,-.69),(-.12,-.02)],trim)
        d.poly([(.31,.20),(-.72,.63),(-.87,.57),(-.16,.04)],lit)
        d.ell(-.11,.26,.15,.09,trim,.25)
        d.ell(.51,.16,.11,.045,trim,.35)
    else: # approaching aircraft
        d.poly([(-1,-.23),(-.94,-.26),(-.19,-.01),(.19,-.01),(.94,-.26),(1,-.23),(.31,.16),(0,.20),(-.31,.16)],body)
        d.poly([(-.026,-.11),(0,-.63),(.04,-.58),(.08,.03)],trim)
        d.ell(0,.04,.18,.25,body); d.ell(0,-.065,.128,.065,trim)
        for s in [-1,1]: d.ell(.43*s,.114,.082,.095,trim)

def car(d,R,m):
    U=R.uniform; b=U(.15,.94); roof=min(1,b+U(.0,.15)); glass=U(.10,.40); tire=U(.015,.15)
    van=(m==3); xroof=U(-.32,-.13); front=U(.28,.48); top=-U(.43,.65)
    if van: xroof=-.67; front=.44; top=-.69
    if m in (0,3):
        d.poly([(-.98,-.02),(-.91,-.20),(xroof-.24,-.25),(xroof,top),(front-.18,top),(front+.14,-.20),(.85,-.13),(.99,.03),(.95,.32),(-.92,.34)],b)
        d.poly([(xroof-.13,-.24),(xroof+.025,top+.045),(front-.20,top+.045),(front+.054,-.21)],glass)
        d.line((xroof+front)*.5-.08,top+.04,(xroof+front)*.5-.10,-.23,.021,b)
        d.line(-.90,.25,.91,.24,.036,b*.54)
        d.ell(.88,-.04,.08,.052,.85)
        for x in [-.61,.60]:
            d.ell(x,.286,.168,.232,tire); d.ell(x,.287,.089,.126,U(.36,.72))
    elif m==1: # near/front or rear, including the tires at either corner
        for s in [-1,1]: d.ell(s*.69,.44,.145,.29,tire)
        d.poly([(-.90,.40),(-.93,-.03),(-.74,-.35),(-.57,-.77),(.55,-.77),(.73,-.36),(.93,-.03),(.89,.41),(.74,.54),(-.74,.54)],b)
        d.poly([(-.67,-.31),(-.51,-.68),(.49,-.68),(.65,-.31)],glass)
        d.poly([(-.83,-.14),(-.67,-.28),(.65,-.28),(.83,-.14),(.66,.04),(-.66,.04)],roof)
        d.box(-.47,.20,.47,.34,tire*1.4); d.box(-.20,.27,.20,.40,.74)
        for s in [-1,1]: d.ell(s*.66,.048,.182,.097,U(.62,.98),-s*.22)
    else: # three-quarter view
        d.poly([(-.98,-.04),(-.78,-.22),(-.50,-.55),(.03,-.67),(.35,-.59),(.62,-.25),(.93,-.04),(.99,.18),(.85,.43),(.15,.57),(-.89,.30)],b)
        d.poly([(-.77,-.22),(-.47,-.50),(-.13,-.55),(.05,-.20)],glass)
        d.poly([(-.07,-.56),(.28,-.53),(.54,-.26),(.14,-.20)],glass*1.16)
        d.poly([(.16,-.15),(.56,-.22),(.90,-.026),(.63,.07),(.17,.10)],roof)
        d.poly([(.17,.31),(.91,.18),(.82,.34),(.18,.48)],b*.48)
        d.ell(.30,.15,.11,.06,.90); d.ell(.87,.063,.076,.047,.88)
        for x,y,r in [(-.69,.25,.13),(.02,.41,.15)]:
            d.ell(x,y,r,r*1.52,tire,-.08); d.ell(x,y,r*.50,r*.75,.54)

def truck(d,R,m):
    U=R.uniform; cab=U(.11,.94); box=U(.30,.96); win=U(.08,.35); tire=U(.015,.13)
    if m==1: # head-on tall cab with upright windscreen, lights and grill
        for s in [-1,1]: d.box(s*.61-.12,.44,s*.61+.12,.89,tire)
        d.poly([(-.77,.63),(-.76,-.60),(-.63,-.84),(.58,-.84),(.75,-.61),(.78,.63),(.61,.73),(-.62,.73)],cab)
        d.box(-.64,-.58,.62,-.11,win); d.box(-.021,-.59,.021,-.10,cab)
        d.box(-.44,.081,.44,.43,win); d.box(-.75,.49,.74,.64,.46)
        for s in [-1,1]: d.box(s*.60-.095,.29,s*.60+.095,.42,.87)
        for y in [.17,.27,.36]: d.box(-.42,y,.42,y+.025,.61)
    else:
        angled=(m==2); hood=(m==3); length=U(.08,.39) if not angled else -.01
        ty=-U(.47,.81); by=.24
        d.box(-.97,.20,.88,.38,.16)
        d.poly([(-.99,ty+.10*angled),(length,ty),(length,by+.08*angled),(-.98,by-.05*angled)],box)
        d.line(-.95,.19,length,.22+.06*angled,.027,box*.55)
        cx=length+.05; cy=ty+U(.035,.19)
        d.poly([(cx,.29),(cx,cy),(.65,cy),(.81,-.18),(.95,-.14 if hood else -.07),(.96,.34),(.78,.40)],cab)
        d.poly([(cx+.07,cy+.07),(.62,cy+.07),(.74,-.19),(cx+.07,-.17)],win)
        if angled:
            d.poly([(.64,cy),(.93,cy+.11),(1,-.13),(.78,-.18)],win+.03)
            d.poly([(.80,-.13),(.99,-.08),(.97,.37),(.79,.42)],cab*.77); d.box(.822,.06,.973,.24,.22)
        for x in [-.78,-.57,length-.01,.72]:
            d.ell(x,.345,.10,.172,tire); d.ell(x,.345,.051,.087,.54)
        d.ell(.90,.16,.063,.041,.9)

def ship(d,R,m):
    U=R.uniform; hull=U(.045,.40) if R.random()<.70 else U(.55,.96); deck=U(.57,.97); dark=U(.08,.38)
    if m==1: # sailboat
        d.poly([(-.78,.37),(.87,.33),(.62,.57),(-.52,.59)],hull)
        d.line(.025,-.98,.025,.38,.018,dark)
        d.poly([(-.045,-.91),(-.072,.227),(-.84,.27)],deck)
        d.poly([(.083,-.77),(.76,.258),(.083,.23)],min(1,deck+.07))
    elif m==2: # approaching bow and bridge
        d.poly([(-.75,.06),(.02,.18),(.75,.026),(.51,.44),(.05,.75),(-.48,.45)],hull)
        d.poly([(-.67,-.02),(-.46,-.36),(.48,-.37),(.70,-.05),(.03,.10)],deck)
        d.box(-.31,-.64,.35,-.26,deck); d.box(-.26,-.53,.29,-.43,dark); d.box(-.084,-.83,.052,-.63,dark)
        d.line(.01,.19,.052,.65,.024,hull*.55)
    else: # liner, cargo vessel or low motorboat
        d.poly([(-.99,.11),(-.58,.15),(.51,.16),(.98,.03),(.84,.32),(.56,.44),(-.70,.42),(-.91,.30)],hull)
        if m==0:
            l=U(-.81,-.54); r=U(.08,.65)
            d.box(l,-.12,r,.14,deck); d.box(l+.09,-.26,r-.06,-.10,deck)
            d.box(l+.12,-.19,r-.09,-.14,dark); d.box(l+.022,-.053,r-.03,-.01,dark)
            d.box(l+.25,-.51,l+.40,-.25,dark); d.line(r+.06,-.53,r+.06,.12,.015,dark)
        else:
            d.poly([(-.62,.13),(-.41,-.19),(.17,-.20),(.49,.14)],deck)
            d.poly([(-.42,-.12),(.15,-.13),(.32,.035),(-.55,.032)],dark)
            d.line(-.50,-.23,.15,-.24,.032,deck)

def bird(d,R,m):
    U=R.uniform; v=U(.09,.60); wing=max(.035,v*U(.46,.88)); belly=min(.98,v+U(.06,.29)); head=v*U(.54,1.22)
    if m==0 or m==3: # perching bird, profile or turned toward viewer
        if m==0:
            d.poly([(-.46,.15),(-1,.62),(-.87,.65),(-.16,.43)],wing)
            d.ell(-.05,.06,.56,.34,belly,-.48); d.ell(-.16,.02,.46,.24,wing,-.44)
            d.ell(.31,-.32,.245,.25,head); d.poly([(.49,-.37),(.79,-.33),(.505,-.26)],v*.67)
            d.ell(.39,-.39,.043,.048,.03)
            for x in [-.14,.06]: d.line(x,.35,x+.005,.56,.024,wing)
        else:
            d.poly([(-.03,.34),(.20,.90),(.40,.89),(.25,.28)],wing)
            d.ell(0,.14,.36,.53,belly,-.10); d.ell(.22,.11,.14,.40,wing,-.16)
            d.ell(-.045,-.34,.266,.26,head); d.poly([(-.25,-.39),(-.48,-.33),(-.24,-.27)],wing)
            d.ell(-.15,-.385,.041,.041,.02)
        if R.random()<.46: d.line(-1,.67,1,.45,.041,U(.16,.41))
    elif m==1: # gliding, bent rather than straight wings
        d.poly([(-.97,-.50),(-.63,-.45),(-.18,-.18),(0,.04),(.21,-.17),(.62,-.39),(.99,-.38),(.65,-.27),(.34,.04),(.13,.27),(.02,.26),(-.16,.25),(-.47,-.10)],wing)
        d.ell(0,.12,.12,.26,v,-.10); d.poly([(-.06,.30),(-.12,.50),(.14,.48),(.055,.27)],wing)
    else: # duck/wading bird, strong curved neck and broad oval body
        d.ell(-.17,.18,.66,.31,belly,-.05); d.ell(-.28,.12,.46,.22,wing)
        d.ell(.35,-.03,.16,.36,v,.17); d.ell(.43,-.34,.218,.20,head); d.ell(.53,-.38,.034,.034,.02)
        d.poly([(.57,-.34),(.88,-.29),(.84,-.23),(.57,-.25)],.62)

def quadruped(d,R,horse,m):
    U=R.uniform
    v=U(.19,.67) if not horse or R.random()<.8 else U(.65,.95)
    hair=v*U(.32,.68); leg=v*U(.64,1.03)
    if m==2: # close head and neck, side-on or in three-quarter view
        d.poly([(-.75,.98),(-.69,.27),(-.39,-.51),(-.18,-.81),(.15,-.62),(.01,-.19),(-.012,.22),(-.12,.98)],v)
        d.ell(-.072,-.33,.37,.40,v,-.31); d.ell(.24,-.057,.56,.224,v,.72)
        d.ell(.60,.26,.22,.16,leg,.45); d.ell(.65,.23,.048,.039,hair*.48)
        d.ell(.073,-.40,.057,.053,.035)
        for x in [-.31,-.082]: d.ell(x,-.705,.084 if horse else .13,.25,leg,-.34 if x<-.1 else .40)
        if horse: d.ell(-.48,-.21,.098,.58,hair,.45)
        elif R.random()<.45:
            for x in [-.19,-.046]:
                d.line(x,-.57,x-.18,-.98,.033,hair); d.line(x-.08,-.77,x+.02,-.95,.027,hair)
        return
    if m==3: # looking directly at us
        d.ell(0,.42,.37,.67,v); d.ell(.01,-.22,.29,.48,v)
        d.ell(.02,.072,.172,.23,leg); d.ell(.03,.181,.13,.094,hair)
        for s in [-1,1]:
            d.ell(s*(.22 if horse else .34),-.64,.092 if horse else .13,.24,leg,s*(.44 if horse else .94))
            d.ell(s*.20,-.35,.048,.041,.03)
        if horse: d.ell(0,-.53,.11,.21,hair,.22)
        return
    # profile, either walking/running or standing
    slender=not horse
    for side in [0,1]:
        for rear in [0,1]:
            x=(-.47 if rear else .25)+(.11 if side else -.02)
            yy=.07; spread=U(-.33,.33) if m==1 else U(-.13,.13)
            kx=x+spread*.55-(.09 if rear else 0); ky=.48+U(-.07,.06)
            tx=x+spread; ty=.80+U(-.08,.09); width=.048 if slender else .061
            val=leg if side else leg*.74
            d.line(x,yy,kx,ky,width*1.15,val); d.line(kx,ky,tx,ty,width*.68,val); d.ell(tx+.015,ty,.055,.039,hair)
    d.ell(-.12,-.021,.57,.244 if slender else .286,v,-.035)
    d.ell(.33,-.27,.198 if slender else .23,.47,v,.42)
    d.ell(.58,-.58,.267,.134 if slender else .163,v,.26)
    d.ell(.73,-.50,.14,.101,leg,.37); d.ell(.58,-.627,.028,.025,.04)
    d.ell(.47,-.763,.062,.141,leg,-.30); d.ell(.57,-.74,.052,.109,v,.39)
    if horse:
        d.ell(.223,-.375,.068,.37,hair,.49); d.ell(-.70,.135,.082,.34,hair,.35)
    else:
        d.ell(-.68,-.063,.103,.051,.83,-.55)
        if m==0 and R.random()<.35:
            d.line(.49,-.67,.34,-1.05,.024,leg); d.line(.40,-.89,.52,-1.02,.022,leg)

def pet(d,R,cat,m):
    U=R.uniform; fur=U(.13,.86); shadow=fur*U(.50,.82); ink=U(.016,.14); cream=min(.99,fur+U(.07,.24))
    if m in (0,1): # broad portrait, close-set muzzle and triangular or floppy ears
        turn=U(-.15,.19); tab=cat and R.random()<.62; pointy=cat or R.random()<.23
        if m==1: d.ell(0,.80,.68,.72,fur)
        for s in [-1,1]:
            if pointy:
                d.poly([(s*.72,-.23),(s*(.60+U(-.05,.07)),-.92),(s*.16,-.48)],fur)
                d.poly([(s*.59,-.39),(s*.57,-.77),(s*.33,-.46)],shadow)
            else: d.ell(s*.65,-.083,.228,.52,shadow,-s*.24)
        d.ell(0,-.015,.72 if cat else .64,.625 if cat else .68,fur)
        if tab:
            for s in [-1,1]:
                for y in [-.023,.13]: d.poly([(s*.685,y-.11),(s*.70,y-.018),(s*.40,y+.04)],shadow)
                d.poly([(s*.34,-.54),(s*.19,-.586),(s*.13,-.30)],shadow)
        if not cat and R.random()<.50:
            for s in [-1,1]: d.ell(s*.34+turn,-.15,.208,.23,shadow,s*.22)
        for s in [-1,1]:
            x=s*(.295 if cat else .274)+turn; y=-.113+turn*s*.10
            d.ell(x,y,.139 if cat else .096,.067 if cat else .081,ink,-s*.17)
            if cat:
                d.ell(x,y,.084,.052,U(.37,.74)); d.ell(x+.012,y,.026,.053,ink)
            else: d.ell(x-.017,y-.02,.023,.023,.86)
        y=.22 if cat else .34; x=turn*1.55
        for s in [-1,1]: d.ell(x+s*.11,y+.01,.153 if cat else .182,.118 if cat else .182,cream,-s*.16)
        if cat: d.poly([(x-.087,y-.09),(x+.088,y-.09),(x,y+.005)],shadow*.70)
        else: d.ell(x,y-.113,.158,.106,ink)
        d.line(x,y-.004,x,y+.083,.015,ink); d.line(x,y+.079,x+.09,y+.109,.016,ink); d.line(x,y+.079,x-.09,y+.109,.016,ink)
    elif m==2: # standing profile
        for x in [-.48,-.32,.27,.44]:
            shift=U(-.15,.14); d.line(x,.09,x+shift,.58,.062 if cat else .076,shadow); d.ell(x+shift+.036,.58,.104,.046,fur)
        d.ell(-.12,.027,.61,.25 if cat else .29,fur,-.09); d.ell(.43,-.142,.208,.30,fur,.33)
        d.ell(.58,-.35,.235,.232,fur)
        if cat:
            for x in [.43,.63]: d.poly([(x-.094,-.48),(x+.002,-.76),(x+.095,-.47)],fur)
            d.line(-.64,.02,-.84,-.35,.064,fur); d.line(-.84,-.35,-.90,-.64,.055,fur)
        else:
            d.ell(.45,-.33,.116,.214,shadow,.19); d.ell(.77,-.29,.20,.104,cream); d.ell(.925,-.32,.059,.05,ink)
            d.line(-.60,.00,-.92,-.32,.076,fur)
        d.ell(.66,-.40,.034,.032,ink)
    else: # sitting upright
        d.ell(-.04,.28,.47,.63,fur,-.10); d.ell(-.37,.55,.31,.31,fur)
        for s in [-1,1]: d.ell(s*.163,.44,.098,.38,cream,s*.08); d.ell(s*.18,.786,.138,.071,cream)
        d.ell(.047,-.37,.37,.34,fur)
        for s in [-1,1]:
            if cat: d.poly([(.05+s*.34,-.46),(.05+s*.31,-.88),(.05+s*.048,-.60)],fur)
            else: d.ell(.042+s*.333,-.38,.119,.25,shadow,-s*.23)
            d.ell(.065+s*.153,-.398,.057,.041,ink)
        d.ell(.073,-.232,.15,.105,cream); d.ell(.075,-.278,.068 if cat else .087,.049,ink)

def frog(d,R,m):
    U=R.uniform; v=U(.23,.78); dark=v*U(.46,.81); light=min(.98,v+U(.06,.23)); gold=U(.48,.89)
    if m in [0,3]: # squat frog facing forward
        for s in [-1,1]:
            d.ell(s*.61,.23,.30,.279,dark,s*.34); d.ell(s*.64,.44,.27,.12,v,-s*.20)
            d.line(s*.48,.06,s*.57,.44,.103,v); d.line(s*.57,.44,s*.76,.52,.058,light)
            for a in [-.08,.02,.12]: d.line(s*.71,.50,s*(.84+a),.59-a*.35,.021,light)
        d.ell(0,.085,.565,.414,v); d.ell(0,-.13,.536,.28,v); d.ell(0,.038,.41,.128,light)
        for s in [-1,1]:
            d.ell(s*.316,-.33,.168,.171,v); d.ell(s*.316,-.363,.098,.094,gold); d.ell(s*.316,-.374,.080,.038,.032)
        d.line(-.356,.010,0,.07,.021,dark); d.line(0,.07,.356,.010,.021,dark)
    else: # a tucked rear leg, large head without a neck, protruding eye
        d.ell(-.12,.111,.681,.343,v,-.17); d.ell(.40,-.107,.45,.241,v,-.10)
        d.ell(.63,-.103,.256,.139,light); d.ell(.38,-.29,.155,.16,v); d.ell(.415,-.309,.099,.098,gold); d.ell(.429,-.318,.075,.055,.035)
        d.ell(-.47,.22,.41,.204,dark,-.21); d.ell(-.38,.30,.371,.114,v,-.30)
        d.line(.35,.052,.22,.38,.074,v); d.line(.22,.38,.54,.443,.048,light)
        for q in [-.06,.015,.091]: d.line(.49,.428,.67+q,.46+q*.45,.018,light)
        if m==3: pass
    # Short irregular mottling strokes on the back (not used to tune anything).
    if m==2:
        for _ in range(7): d.ell(U(-.56,.07),U(-.01,.14),U(.029,.075),U(.023,.058),dark)

DRAW=[airplane,car,bird,lambda d,r,m:pet(d,r,True,m),lambda d,r,m:quadruped(d,r,False,m),lambda d,r,m:pet(d,r,False,m),frog,lambda d,r,m:quadruped(d,r,True,m),ship,truck]

# Nominal bounding radii of each drawing type as a fraction of the image radius.
# Kept in object coordinates so (e.g.) a cow-sized quadruped is not forced to
# have the proportions of a human face when we zoom in on it.
SIZES=[[(.91,.62),(.83,.54),(.88,.60),(.92,.63)],
       [(.93,.78),(.76,.84),(.92,.87),(.92,.82)],
       [(.65,.85),(.84,.71),(.72,.76),(.78,.87)],
       [(.98,1.02),(.76,.80),(.79,.86),(.75,.88)],
       [(.83,.85),(.86,.84),(.81,.87),(.83,.89)],
       [(1.00,1.00),(.79,.86),(.82,.89),(.75,.89)],
       [(.88,.98),(.87,.98),(.83,.96),(.87,.99)],
       [(.85,.94),(.9,.85),(.9,.96),(.85,.94)],
       [(.91,.77),(.80,.85),(.77,.81),(.88,.82)],
       [(.94,.87),(.81,.92),(.95,.91),(.94,.87)]]

N=16
# Evaluate filled solids with subpixel coverage for smooth filter coefficients.
SS=4
zy,zx=np.mgrid[:N*SS,:N*SS]; zx=((zx+.5)/(N*SS)*2-1).astype('f'); zy=((zy+.5)/(N*SS)*2-1).astype('f')

def draw_one(k,m,R):
    U=R.uniform
    mx,my=SIZES[k][m]; zoom=math.exp(U(-.33,.40)); mx*=zoom*math.exp(U(-.14,.14)); my*=zoom*math.exp(U(-.18,.18))
    cx=U(-.13,.13); cy=U(-.13,.13)
    if k in [1,8,9] and m!=1: cy+=.075
    tilt=U(-.15,.15)
    if k in [0,2,6]: tilt=U(-.55,.52)
    if k==0 and m==1: tilt=U(-1.50,1.50)
    if k==2 and m==1: tilt=U(-.83,.83)
    if k in [3,5] and m<2: tilt=U(-.29,.29)
    c=math.cos(tilt); s=math.sin(tilt)
    x=(c*(zx-cx)+s*(zy-cy))/mx; y=(c*(zy-cy)-s*(zx-cx))/my
    bg=U(.22,.83)
    # Background is mostly left featureless. Sometimes include a simple ground
    # plane or waterline, each at a height chosen relative to this drawing.
    if k in [0,2,8]: bg=U(.44,.97) if R.random()<.71 else U(.17,.66)
    if k in [1,9] and R.random()<.64:
        ground_y=cy+my*(.57 if k==1 else .58)
        bg=np.where(zy>ground_y,U(.22,.63),bg)
    if k==8 and R.random()<.67:
        water_y=cy+my*U(-.32,.34); bg=np.where(zy>water_y,U(.19,.59),bg)
    d=Canvas(x,y,bg); backdrop=d.a.copy(); DRAW[k](d,R,m)
    # Each drawing has two planes: shading and foreground coverage. They stay
    # together as one hypothesis and are never combined across drawings.
    planes=np.stack([d.a,d.a!=backdrop]); return planes.reshape(2,N,SS,N,SS).mean((2,4))

def gder(theta):
    yy,xx=np.mgrid[-1:2,-1:2]; f=(xx*np.cos(theta)+yy*np.sin(theta))*np.exp(-(xx*xx+yy*yy)/1.7)
    return (f/np.abs(f).sum()*2).astype('f')
GF=np.stack([gder(j*np.pi/4) for j in range(4)])

def unit(a):
    a=a-a.mean(axis=(-1,-2),keepdims=True); return a/(np.linalg.norm(a)+1.e-7)

def describe(t):
    # Signed brightness and centered oriented-gradient magnitudes of *one*
    # analytic drawing. Each block is unit length, independently of any other
    # drawing. The edge filters agree with the ones in the inference graph.
    a=unit(t).ravel(); win=np.lib.stride_tricks.sliding_window_view(t,(3,3)); e=np.abs(np.einsum('yxij,cij->cyx',win,GF))
    e=e.reshape(4,7,2,7,2).mean((2,4)); b=unit(e).ravel()
    return a,b

LUMA=np.array([.299,.587,.114],dtype='f')
def templates():
    mono=affine('luminance','image',LUMA.reshape(1,3,1,1)); small=pool('reduce_to_cells',mono,2)
    norm=layer('contrast','norm',small)
    edge=affine('contours',small,np.concatenate((GF,-GF))[:,None]); half=relu('half_waves',edge)
    fold=affine('unsigned_contours',half,np.tile(np.eye(4),(1,2))[:,:,None,None]); smooth=pool('contour_cells',fold,2)
    zn=layer('contrast_contours','norm',smooth)
    # Weight units: correlations, scaled by the square root of input dimension
    # because norm sets per-channel RMS to one. Four orientations contribute.
    # Positive center-surround color saliency can match a silhouette even if
    # the object and background happen to have very similar brightness.
    small_rgb=pool('color_cells','image',2)
    yy,xx=np.mgrid[:16,:16]; radial=np.maximum(abs(xx-7.5),abs(yy-7.5))
    surround=np.maximum(radial-4.5,0); surround/=surround.sum()
    I=np.eye(256)-surround.ravel()[None,:]; contrast_rgb=np.kron(np.eye(3),I)
    away=affine('border_difference',small_rgb,np.concatenate([contrast_rgb,-contrast_rgb])); away=relu('positive_differences',away)
    distance=affine('color_distance',away,np.tile(np.eye(256),(1,6))/3)
    dist=reshape('figure_ground',distance,[1,16,16]); dist=layer('saliency_contrast','norm',dist)
    K=512; shape_w=[]; edge_w=[]; support_w=[]
    R=np.random.default_rng(701009)
    for k in range(10):
        for j in range(K//2):
            m=j%4; t=draw_one(k,m,R)
            for v in (t,t[:,:,::-1]):
                a,b=describe(v[0]); shape_w.append(a/16); edge_w.append(b/14); support_w.append(unit(v[1]).ravel()/16)
    lum=affine('figure_tone',norm,np.array(shape_w)*.36); eg=affine('figure_contour',zn,np.array(edge_w)*.49)
    support=affine('figure_silhouette',dist,np.array(support_w)*.15)
    together=layer('figure_match','add',[lum,eg,support]); cube=reshape('by_species_and_view',together,[10,4,K//4])
    # Retain the best alternatives in several independent banks; a chance
    # alignment in one bank cannot wholly dominate the vote.
    bank=pool('best_in_bank',cube,[1,K//4],maximum=True)
    peak=pool('strongest_figure',bank,[4,1],maximum=True); mean=pool('repeat_matches',bank,[4,1])
    result=layer('shape_evidence','add',[affine('peak_vote',peak,np.eye(10)*.54),affine('repeated_vote',mean,np.eye(10)*.46)])
    return result,mono

# Smooth membership in a color word = min of 3 clipped linear half-spaces.
# Thresholds refer to ordinary RGB differences on a 0--1 scale. All pooling
# regions and all readout preferences below were specified by hand.
def colors():
    r,g,b=np.eye(3); Y=LUMA
    blue=(b-r,.018,.15); cyan=(g-r,.006,.11); grass=(g-r,.009,.095); yellow=(g-b,.022,.14)
    warm=(r-g,.018,.13); light=(Y,.57,.25); cool=(b-g,-.005,.085)
    spec={
        'blue':[blue,cool], 'aqua':[blue,cyan,(b-g,-.105,.12)],
        'leaf':[grass,yellow], 'olive':[yellow,(g-r,-.081,.087)],
        'tan':[warm,(g-b,.008,.098),(r-b,.075,.15)],
        'red':[(r-g,.102,.22),(r-b,.13,.23)], 'cobalt':[(b-r,.12,.25),(b-g,.08,.20)],
        'white':[light,(r,.56,.25),(b,.51,.23)], 'black':[(-Y,-.255,.19)],
        'gold':[(r-b,.19,.24),(g-b,.11,.20),(r-g,-.045,.10)],
        'gray':[(Y,.18,.14),(-Y,-.82,.14)],'pale':[light], 'deep':[(-Y,-.42,.22)],
    }
    names=list(spec); A=[]; bias=[]
    for c in names:
        terms=spec[c]; terms=terms+[terms[-1]]*(3-len(terms))
        for vec,lower,width in terms:
            for cut in [0,1]: A.append(vec/width); bias.append(-lower/width-cut)
    # Reduce at two-pixel resolution to avoid classifying isolated noisy pixels.
    src='color_cells'; h=relu('color_halfspaces',affine('color_projection',src,np.array(A)[:,:,None,None],bias))
    n=len(names)*3; sums=np.eye(n).repeat(2,axis=1); sums[:,1::2]*=-1
    h=affine('bounded_color',h,-sums[:,:,None,None]); h=reshape('color_conditions',h,[len(names),3,256]); h=pool('limiting_condition',h,[3,1],maximum=True)
    h=reshape('color_support',h,[len(names),16,16]); h=affine('color_membership',h,-np.eye(len(names))[:,:,None,None]); return pool('color_regions',h,2),names

def texture(mono):
    # Valid convolutions: the image perimeter cannot itself make a line.
    yy,xx=np.mgrid[-3:4,-3:4]; F=[]; nms=[]
    def add(n,f):
        f=f-f.mean(); F.append(f/np.abs(f).sum()*2); nms.append(n)
    for axis in range(4):
        phi=axis*np.pi/4; across=yy*math.cos(phi)+xx*math.sin(phi); along=xx*math.cos(phi)-yy*math.sin(phi)
        for name,length in [('edge',.78),('line',2.55)]:
            add(name+str(axis),across*np.exp(-.5*(across/.74)**2-.5*(along/length)**2))
    tight=np.exp(-(xx*xx+yy*yy)/.53); broad=np.exp(-(xx*xx+yy*yy)/2.70)
    add('detail',tight/tight.sum()-broad/broad.sum())
    F=np.array(F); q=affine('directional_differences',mono,np.concatenate([F,-F])[:,None],stride=2)
    q=relu('directed_edge_strength',q); q=affine('edge_strength',q,np.tile(np.eye(len(nms)),(1,2))[:,:,None,None]); q=pool('texture_regions',q,2)
    return q,nms

# Register a few regional measurements using mostly overlapping soft windows.
# This is inference-time spatial pooling of the unknown image, not pooling
# across templates or across examples to make parameters.
REG={
    'all': lambda x,y: 1+0*x,
    'center':lambda x,y:np.exp(-.5*((x/.46)**2+(y/.46)**2)),
    'upper':lambda x,y:np.clip((-y-.17)*2.8,0,1),
    'lower':lambda x,y:np.clip((y-.21)*3.3,0,1),
    'around':lambda x,y:np.clip((np.maximum(abs(x),abs(y))-.52)*3.5,0,1),
    'sides':lambda x,y:np.clip((abs(x)-.49)*3.4,0,1)*np.exp(-.6*(y/.78)**2),
    'flanks':lambda x,y:np.clip((abs(x)-.35)*3.4,0,1)*np.clip((y-.02)*4,0,1),
    'middle':lambda x,y:np.exp(-.5*(y/.32)**2),
    'under':lambda x,y:np.exp(-.5*(((y-.44)/.18)**2+(x/.70)**2))}

def region(n,name):
    t=(np.arange(n)+.5)/n*2-1; y,x=np.meshgrid(t,t,indexing='ij'); z=REG[name](x,y); return z/z.sum()

def collect(src,names,n,requests):
    dest=[]; mats=[]
    for ch,rs in requests.items():
        for reg in rs.split():
            dest.append(ch+'_'+reg); w=np.zeros((len(names),n,n),'f'); w[names.index(ch)]=region(n,reg); mats.append(w.ravel())
    return affine(src+'_summary',src,mats),dest

class Signals:
    """A compiler for a hand-specified sum of rectangle/ramp votes."""
    def __init__(self,inputs): self.inputs=inputs; self.names=sum([ns for _,ns in inputs],[]); self.items=[]
    def ramp(self,terms,at,width,weights):
        w=np.zeros(len(self.names));
        for key,val in terms.items(): w[self.names.index(key)]+=val/width
        # Every feature saturates. A centered transfer gives no preference at
        # the stated crossover, and opposite votes below/above it.
        self.items.append((w,.5-at/width,np.array(weights,dtype='f')))
    def emit(self):
        A=np.array([w for w,b,v in self.items]).repeat(2,axis=0); B=np.array([b for w,b,v in self.items]).repeat(2); B[1::2]-=1
        z=[]; i=0
        for src,names in self.inputs:
            z.append(affine(src+'_tests',src,A[:,i:i+len(names)],B if not z else 0)); i+=len(names)
        h=relu('scene_tests',layer('combine_scene_cues','add',z)); W=np.stack([v for w,b,v in self.items],axis=1); WV=W.repeat(2,axis=1); WV[:,1::2]*=-1
        return affine('scene_evidence',h,WV,-.5*W.sum(1))

def scene(mono):
    image,names=colors(); c=collect(image,names,8,{
        'blue':'upper lower around center all', 'aqua':'upper lower around center',
        'leaf':'upper lower center around all', 'olive':'upper lower center around all',
        'tan':'center around lower all', 'red':'center around', 'cobalt':'center around',
        'white':'upper center around lower', 'black':'center under lower around',
        'gold':'center around', 'pale':'all upper center lower', 'deep':'all upper center lower'})
    text,names=texture(mono); regs='all upper center around lower middle'; t=collect(text,names,6,{n:regs for n in names})
    S=Signals([c,t]); add=S.ramp
    # Order of preferences: plane car bird cat deer dog frog horse ship truck.
    # Blue surround and untextured sky; cool lower field suggests sky or sea.
    add({'blue_upper':1},.13,.39,      [.145,-.020,.025,-.071,-.071,-.061,-.067,.006,.108,.040])
    add({'blue_lower':.72,'aqua_lower':.28},.095,.33,[.108,-.022,.003,-.041,-.063,-.045,-.067,-.038,.199,-.004])
    add({'blue_around':1,'blue_center':-.55},.076,.34,[.074,0,.028,-.013,-.023,-.018,-.025,0,.044,.006])
    # Sky reflected in water vs. a patch of sky above green ground.
    add({'blue_upper':.6,'pale_upper':.4,'leaf_lower':.50,'olive_lower':.30,'blue_lower':-.66},.29,.62,
        [0,0,.004,-.045,.035,-.013,-.04,.092,-.046,.024])
    # Leaf green and yellow-green are distinct: olive/camouflage overlaps tan
    # fur, while a true green border often signals vegetation behind an animal.
    add({'leaf_around':.65,'leaf_lower':.35},.14,.45,[-.084,-.086,.073,-.057,.117,.008,.081,.091,-.113,-.079])
    add({'olive_around':1},.23,.54,[-.064,-.065,.042,-.024,.081,.011,.067,.055,-.085,-.060])
    add({'olive_center':1},.25,.52,[-.046,-.038,.025,-.027,.007,-.015,.152,-.030,-.048,-.040])
    add({'leaf_center':1,'leaf_around':-.53},.034,.37,[0,-.027,.026,-.03,-.047,-.049,.119,-.061,0,-.021])
    add({'leaf_around':.62,'olive_around':.38,'olive_center':-.60,'leaf_center':-.40},.010,.33,
        [-.022,-.014,.055,-.026,.064,.011,-.039,.051,-.015,-.010])
    # Warm fur / indoor light, brown hide against cool vegetation.
    add({'tan_center':1},.22,.55,[-.076,-.036,.009,.040,.051,.073,.016,.051,-.080,-.026])
    add({'tan_around':1},.19,.50,[-.058,-.039,-.012,.088,.018,.055,.014,-.003,-.059,-.031])
    add({'tan_center':1,'tan_around':-.85},.035,.39,[-.034,-.020,.012,-.009,.060,.027,-.008,.064,-.054,-.012])
    # Paint, vivid plumage, a pale trailer body and a dark undercarriage.
    add({'red_center':1,'red_around':-.40},.055,.31,[-.005,.113,.034,-.034,-.031,-.020,-.030,.012,.014,.104])
    add({'cobalt_center':1,'cobalt_around':-.77},.02,.25,[0,.078,.045,-.022,-.028,-.029,-.014,-.011,.005,.06])
    add({'gold_center':1,'gold_around':-.75},.03,.40,[-.015,.020,.060,-.008,.006,.008,.048,-.012,-.022,.002])
    add({'white_center':1,'white_around':-.87},.013,.37,[.029,.037,0,.016,-.074,.022,-.066,-.023,.045,.090])
    add({'black_under':1,'black_lower':-.60,'black_around':-.28},.024,.36,[-.045,.105,.001,-.055,-.015,-.035,-.024,.012,.052,.081])
    add({'pale_upper':.50,'pale_lower':.50},.19,.52,[.085,.007,.025,.017,-.076,-.015,-.065,-.013,.063,.024])
    add({'deep_upper':1,'deep_lower':-.42},.091,.39,[-.060,.026,.012,.013,.051,.020,.045,.007,-.048,-.017])
    add({'black_center':1},.11,.35,[-.009,.048,-.012,-.012,-.065,.015,-.03,.037,.002,.026])
    # High-frequency contrast, straightness, orientation, and spatial layout.
    # Filters have unit response to a unit step across their preferred axis.
    edge=lambda reg:{'edge%d_%s'%(i,reg):.25 for i in range(4)}
    def lin(*terms):
        out={}
        for coeff,expr in terms:
            for n,x in expr.items(): out[n]=out.get(n,0)+coeff*x
        return out
    add(edge('upper'),.069,.092,[-.117,.031,-.019,.006,.047,.012,.036,.002,-.076,.009])
    add(edge('lower'),.068,.088,[-.107,-.004,-.003,.020,.029,.005,.056,.020,-.076,-.012])
    add(lin((1,edge('center')),(-1,edge('around'))),.024,.11,[.025,.001,.056,.023,-.007,.025,-.006,-.015,.022,-.022])
    add({'detail_all':1},.040,.074,[-.070,-.008,-.008,.039,.026,.026,.071,.017,-.041,-.004])
    # Sustained horizontal / vertical contours compared to diagonal contours.
    def direction(reg,a=0,b=2):
        return {'line%d_%s'%(a,reg):.5,'line%d_%s'%(b,reg):.5,'line1_'+reg:-.50,'line3_'+reg:-.50}
    add(direction('all'),.0015,.031,[.012,.133,-.039,-.054,-.075,-.047,-.077,-.019,.100,.139])
    add({'line0_middle':1,'line1_middle':-.50,'line3_middle':-.50},.006,.077,[.012,.070,-.024,-.033,-.025,-.024,-.036,-.018,.091,.059])
    add({'line0_lower':1,'line2_lower':-.75},.008,.075,[.002,.050,-.022,-.034,-.013,-.022,-.009,-.025,.092,.030])
    add({'line2_center':1,'line1_center':-.50,'line3_center':-.50},.001,.062,[-.018,.003,-.017,-.010,.004,-.009,-.030,.018,-.024,.073])
    add({'line0_all':.50,'line2_all':.50,'edge0_all':-.36,'edge2_all':-.36},.001,.039,
        [.034,.038,-.010,-.025,-.049,-.026,-.052,.003,.032,.048])
    return S.emit()

# Part geometry is independent of the large templates. Each part kernel is a
# single drawing of a wheel, eye or nose (or a contrast ridge). Convolution
# translates it, and a second layer describes allowable layouts of such parts.
def wheels(mono):
    y,x=np.mgrid[-3:4,-3:4]; A=[]
    for radius in [1.45,2.15,2.68]:
        for aspect in [.76,1.08]:
            rr=np.sqrt((x/aspect)**2+y*y)
            for hub in [.12,.60]:
                # Dark tire with an optional light hub and a bright exterior.
                # Ignore road pixels immediately below the contact patch.
                surround=np.exp(-.5*((rr-radius-1.0)/.68)**2)*np.clip(1.32-.29*y,.24,1)
                rubber=np.exp(-.5*((rr-radius*.75)/max(.44,.26*radius))**2)
                center=np.exp(-.5*(rr/(.37*radius))**2)
                f=(1-hub)*surround/surround.sum()+hub*center/center.sum()-rubber/rubber.sum()
                A.append(f/(.5*np.abs(f).sum()))
    z=affine('tire_rings',mono,np.array(A)[:,None],stride=1); z=reshape('tire_shapes',z,[1,len(A),26*26])
    z=pool('best_tire_shape',z,[len(A),1],maximum=True); z=reshape('tire_map',z,[1,26,26]); z=pool('tire_neighborhood',z,2,1,True)
    # Pairs with similar heights, viewed from the side or at a shallow angle.
    pairs=[]; proposals=[]; D=25
    for group in range(4):
        for gap in ([8,10,12,14,16,18] if group<2 else [5,7,9,12,15,18]):
            for y0 in range(10,23,2):
                for xc in range(6,20,2):
                    x0=xc-gap//2; x1=x0+gap
                    if x0<0 or x1>=D: continue
                    for dy in [-3,-1,0,1,3]:
                        if not 0<=y0+dy<D: continue
                        if group==0 and abs(dy)>1: continue
                        if group==1 and abs(dy)<2: continue
                        pts=[(y0,x0),(y0+dy,x1)]
                        if group>=2:
                            # Trailer tandems: a third wheel near one axle.
                            if gap<7: continue
                            step=min(4,gap//3); pts.append((y0 if group==2 else y0+dy,x0+step if group==2 else x1-step))
                        w=np.zeros((D,D),'f');
                        for yy,xx in pts: w[yy,xx]=1/len(pts)
                        pairs.append(w.ravel())
        proposals.append(len(pairs))
    # Avoid measuring the bank to choose any settings: these indices specify
    # the geometry only. A grouped layout pool has equal width by duplication.
    count=np.diff([0]+proposals).tolist(); M=max(count); grouped=[]; idx=0
    for n in count:
        grouped.extend(pairs[idx+i%n] for i in range(M)); idx+=n
    z=affine('axle_layouts',z,grouped); z=reshape('group_axles',z,[4,1,M]); z=pool('aligned_wheels',z,[1,M],maximum=True)
    # Part match is a local Michelson-like brightness difference, not a class
    # score until above the deliberately conservative tire-contrast threshold.
    z=relu('distinct_wheels',affine('threshold_axles',z,np.eye(4),[-.225,-.24,-.215,-.215]));
    a=np.array([-.025,.29,-.025,-.025,-.015,-.02,-.024,-.019,-.032,.135]); t=np.array([-.022,.023,-.014,-.016,-.014,-.016,-.023,-.015,-.023,.19]);
    return affine('wheel_evidence',z,np.stack([a*.60,a*.40,t*.5,t*.5],axis=1))

def faces(mono):
    # Analytical local face drawings: triangular ears vs hanging ears; small
    # short muzzle vs large dark nose. Local matching tolerates displacement
    # independently of the rest of the animal. Each patch is centered and
    # normalized on its own, and no patch is used to estimate a parameter.
    L=10; S=4; y,x=np.mgrid[:L*S,:L*S]; x=(x+.5)/(L*S)*2-1; y=(y+.5)/(L*S)*2-1
    R=np.random.default_rng(100912); W=[]
    for k in [3,5,6,4,7]:
        for i in range(32):
            zoom=R.uniform(.86,1.48); ang=R.uniform(-.21,.21); c=np.cos(ang); s=np.sin(ang)
            X=(c*x+s*y)/zoom; Y=(-s*x+c*y)/(zoom*R.uniform(.91,1.12)); d=Canvas(X,Y,R.uniform(.21,.89))
            if k in [3,5]: pet(d,R,k==3,0)
            elif k==6: frog(d,R,0)
            else: quadruped(d,R,k==7,3 if i%3 else 2)
            a=d.a.reshape(L,S,L,S).mean((1,3));
            for v in [a,a[:,::-1]]: W.append(unit(v))
    W=np.array(W)[:,None]; z=affine('portrait_filters','contrast',W/L,stride=1)
    z=reshape('local_portraits',z,[5,64,49]); z=pool('local_portrait_votes',z,[64,49],maximum=True)
    # Compare like cues rather than allowing the largest local-contrast patch
    # to vote simultaneously for all animals. Keep a small absolute vote.
    A=np.zeros((10,5))
    for i,(k,magnitude) in enumerate([(3,.165),(5,.16),(6,.10),(4,.08),(7,.08)]):
        A[k,:]=-magnitude*.19; A[k,i]+=magnitude
    return affine('portrait_evidence',z,A,-A.sum(1)*.45)

def main():
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    figures,mono=templates(); context=scene(mono); parts=wheels(mono); heads=faces(mono)
    layer('class_scores','add',[figures,context,parts,heads])
    (out/'network.json').write_text(json.dumps({'layers':LAYERS},indent=2)+'\n')
    np.savez(out/'weights.npz',**WEIGHTS)

if __name__=='__main__': main()
