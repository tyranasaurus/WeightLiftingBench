"""Construct a fixed visual classifier. No input files, fitting, or evaluation."""
import sys, json, math
from pathlib import Path
import numpy as np

D = Path(sys.argv[1]); D.mkdir(parents=True, exist_ok=True)
L=[]; W={}
def layer(name, op, inp, **settings):
    L.append(dict(name=name,op=op,inputs=[inp] if isinstance(inp,str) else inp,**settings)); return name
def conv(name, inp, w, b=0, stride=1, padding=0):
    w=np.asarray(w,np.float32); o,_,k,_=w.shape
    W[name+'.W']=w; W[name+'.b']=np.broadcast_to(b,(o,)).astype(np.float32).copy()
    return layer(name,'conv',inp,out=o,kernel=[k,k],stride=stride,padding=padding)
def dense(name, inp, w, b=0):
    w=np.asarray(w,np.float32); o=w.shape[0]
    W[name+'.W']=w.reshape(o,-1); W[name+'.b']=np.broadcast_to(b,(o,)).astype(np.float32).copy()
    return layer(name,'dense',inp,out=o)
def pool(name,op,inp,h,w=None,sh=None,sw=None):
    w=h if w is None else w; sh=h if sh is None else sh; sw=w if sw is None else sw
    return layer(name,op+'pool',inp,size=[h,w],stride=[sh,sw])
def rel(name,inp): return layer(name,'relu',inp)
def norm(name,inp):return layer(name,'norm',inp)
def shape(name,inp,*s):return layer(name,'reshape',inp,shape=list(s))
def plus(name,*inputs):return layer(name,'add',list(inputs))

# Analytic primitives for individual illustrated hypotheses. Coordinates are in a
# canonical object frame, not coordinates in any example photo. Paint at 2x.
rng=np.random.default_rng(157275)
class Paper:
    def __init__(self,scale,stretch,dx,dy,angle,flip,bg):
        y,x=np.mgrid[0:64,0:64].astype(np.float64); x=(x+.5)/32-1-dx; y=(y+.5)/32-1-dy
        c,s=np.cos(angle),np.sin(angle)
        self.x=(x*c+y*s)/scale*flip; self.y=(-x*s+y*c)/(scale*stretch)
        self.a=np.full((64,64),float(bg))
    def ell(self,x,y,rx,ry,v,angle=0):
        X,Y=self.x-x,self.y-y; c,s=np.cos(angle),np.sin(angle)
        mask=((X*c+Y*s)/rx)**2+((-X*s+Y*c)/ry)**2<=1
        self.a[mask]=v
    def poly(self,pts,v):
        xs,ys=self.x,self.y; mask=np.zeros_like(xs,dtype=bool); last=pts[-1]
        for p in pts:
            x,y=p; a,b=last
            if b!=y: mask^=((y>ys)!=(b>ys)) & (xs<(a-x)*(ys-y)/(b-y)+x)
            last=p
        self.a[mask]=v
    def line(self,pts,r,v):
        for (x,y),(a,b) in zip(pts,pts[1:]):
            dx,dy=a-x,b-y
            t=np.clip(((self.x-x)*dx+(self.y-y)*dy)/(dx*dx+dy*dy+1e-12),0,1)
            self.a[(self.x-x-t*dx)**2+(self.y-y-t*dy)**2<=r*r]=v
    def box(self,x,y,X,Y,v): self.a[(self.x>=x)&(self.x<=X)&(self.y>=y)&(self.y<=Y)]=v
    def frame(self):return self.a.reshape(32,2,32,2).mean((1,3))

def wheels(p,xx,level,r,width=1):
    for x in xx:
        p.ell(x,level,r*width,r,.10);p.ell(x,level,r*.52*width,r*.52,.54)

def plane(p,m,t,bg):
    ink=[.26,.78,.92,.48][t%4]; shade=ink*.70
    if m in (0,1): # airliner or fighter in profile, tail, wings, fuselage
        p.poly([(-.78,-.02),(-.95,-.35),(-.82,-.36),(-.52,-.05)],shade)
        p.poly([(-.1,-.05),(-.51,-.29),(-.39,-.3),(.22,-.04)],shade)
        p.poly([(-.93,-.02),(.50,-.08),(.74,-.07),(.98,.035),(.76,.095),(-.50,.115),(-.90,.05)],ink)
        p.poly([(.17,.045),(-.42,.37),(-.53,.36),(-.20,.065)],shade)
        p.line([(-.60,.012),(.65,-.003)],.013,ink*.45)
        if m==0:p.ell(-.15,.165,.16,.046,shade)
    elif m==2: # head on or climbing
        p.poly([(-1,.04),(-.97,-.018),(-.13,.015),(-.06,-.09),(.05,-.09),(.14,.015),(.98,-.055),(1,.015),(.10,.125),(-.09,.125)],ink)
        p.ell(0,.075,.105,.115,shade);p.line([(0,-.26),(0,-.05)],.021,shade)
        for x in [-.35,.35]:p.ell(x,.095,.06,.064,shade)
    else: # underside in plan, with swept wings and stabilizers
        p.poly([(-.84,-.02),(-.98,-.27),(-.88,-.28),(-.62,-.07),(-.19,-.06),(-.50,-.73),(-.36,-.77),(.22,-.08),(.72,-.069),(.98,0),(.73,.07),(.22,.09),(-.39,.73),(-.52,.70),(-.18,.076),(-.61,.068),(-.89,.28),(-.97,.24)],ink)
        p.line([(-.55,.03),(.72,.03)],.019,shade)

def car(p,m,t,bg):
    body=[.76,.27,.88,.43,.57][t%5]; glass=[.28,.49,.32,.60][(t//2)%4]
    if m<3:
        back=[-.57,-.79,-.74][m]; roof=[-.38,-.32,-.41][m]; h=[-.36,-.43,-.49][m]
        p.poly([(-.96,.035),(-.89,-.095),(back,-.13),(roof,h),(.17,h-.006),(.51,-.15),(.85,-.09),(.98,.045),(.96,.235),(.74,.29),(-.84,.29),(-.97,.19)],body)
        p.poly([(back+.06,-.126),(roof+.035,h+.045),(-.11,h+.039),(-.115,-.13)],glass)
        p.poly([(-.065,-.13),(-.078,h+.04),(.14,h+.04),(.435,-.142)],glass)
        p.line([(-.86,.218),(.87,.20)],.027,body*.55)
        p.ell(.886,-.015,.07,.032,.91);p.ell(-.89,-.005,.03,.045,.25)
        wheels(p,[-.60,.60],.234,.164,.98)
        p.line([(-.05,-.094),(-.04,.15)],.009,body*.75)
        if m==1:p.box(-.81,-.09,-.66,-.045,.35)
    else:
        ob=.17 if m==4 else 0
        p.poly([(-.75,.30),(-.80,-.04),(-.64+ob,-.20),(-.44+ob,-.52),(.42+ob,-.52),(.66+ob*.3,-.20),(.80,-.03),(.78,.33),(.57,.43),(-.65,.42)],body)
        p.poly([(-.58+ob,-.20),(-.41+ob,-.46),(.39+ob,-.46),(.56+ob,-.20)],glass)
        p.poly([(-.67,-.026),(-.43,.05),(.45,.05),(.70,-.044),(.61,.115),(-.59,.13)],body*.78)
        p.box(-.43,.17,.45,.29,.17); p.box(-.18,.218,.18,.30,.75)
        for x in [-.59,.59]:p.ell(x,.07,.11,.051,.92);p.box(x-.10,.35,x+.10,.47,.11)

def truck(p,m,t,bg):
    cargo=[.84,.47,.69,.32,.91][t%5];cab=[.78,.34,.86,.42][(t//2)%4];glass=.29 if t%3 else .50
    if m<=2:
        top=-.52 if m!=2 else -.26
        p.box(-.98,top,.31,.14,cargo);p.box(-.97,.125,.30,.18,cargo*.72)
        p.line([(-.95,.22),(.88,.22)],.064,.18)
        if m==1: # American tractor, bonnet
            p.poly([(.24,.20),(.25,-.40),(.48,-.44),(.67,-.32),(.70,-.10),(.96,-.066),(.99,.22)],cab)
            p.poly([(.43,-.35),(.56,-.32),(.63,-.14),(.43,-.16)],glass)
        else:
            p.poly([(.34,.20),(.34,-.37),(.47,-.45),(.77,-.44),(.91,-.27),(.97,.16),(.85,.24)],cab)
            p.poly([(.48,-.37),(.74,-.37),(.85,-.20),(.85,-.13),(.48,-.14)],glass)
        p.box(.88,.045,.96,.095,.91);wheels(p,[-.75,-.48,.19,.73],.255,.139)
        if m==2:
            p.line([(-.94,top+.08),(.24,top+.08)],.022,cargo*.68)
    elif m==3: # three-quarter, box and a tall cab
        p.poly([(-.93,-.46),(.12,-.65),(.21,.13),(-.89,.27)],cargo*.86)
        p.poly([(.12,-.65),(.68,-.55),(.72,.16),(.21,.13)],cargo)
        p.poly([(-.06,-.45),(.24,-.54),(.74,-.45),(.84,-.25),(.83,.29),(.21,.38),(-.06,.25)],cab)
        p.poly([(.22,-.40),(.72,-.36),(.77,-.13),(.24,-.16)],glass)
        p.poly([(-.005,-.36),(.13,-.41),(.15,-.18),(.012,-.13)],glass)
        p.box(.27,.10,.71,.21,.24);wheels(p,[-.69,-.48,.06,.73],.265,.135,.64)
    else:
        p.box(-.51,-.77,.51,.09,cargo);p.poly([(-.64,.38),(-.63,-.30),(-.49,-.54),(.48,-.54),(.62,-.31),(.64,.39)],cab)
        p.box(-.52,-.36,.52,-.078,glass);p.box(-.37,.085,.37,.30,.21)
        for x in [-.47,.47]:p.box(x-.08,.19,x+.08,.27,.88);p.box(x-.11,.38,x+.11,.55,.12)

def ship(p,m,t,bg):
    hull=[.21,.76,.36,.87][t%4]; cabin=[.89,.61,.78][(t//2)%3]
    if m==0 or m==1: # cargo vessel / passenger ship
        p.poly([(-.97,.075),(.69,.09),(.99,.002),(.82,.24),(.68,.30),(-.78,.30)],hull)
        if m==0:
            p.box(-.68,-.16,-.28,.08,cabin);p.box(-.59,-.26,-.35,-.16,cabin);p.box(-.57,-.38,-.47,-.26,.29)
            for x in [-.15,.12,.39]:p.box(x,-.06,x+.24,.079,.52)
        else:
            p.poly([(-.75,.084),(-.62,-.11),(-.42,-.20),(.41,-.20),(.68,.02),(.79,.085)],cabin)
            p.line([(-.56,-.069),(.47,-.069)],.025,.33);p.line([(-.46,-.148),(.37,-.148)],.018,.47)
            p.poly([(-.25,-.20),(-.31,-.40),(-.12,-.40),(-.08,-.20)],.34)
    elif m==2:
        p.poly([(-.79,.23),(.37,.26),(.85,.15),(.65,.39),(-.49,.40)],hull)
        p.line([(-.06,.27),(-.055,-.83)],.018,.30)
        p.poly([(-.09,-.77),(-.12,.18),(-.73,.18)],cabin);p.poly([(-.004,-.59),(.57,.13),(.03,.19)],.84)
    elif m==3:
        p.poly([(-.90,.10),(.99,-.04),(.76,.22),(.51,.29),(-.67,.29)],hull)
        p.poly([(-.55,.071),(-.23,-.18),(.15,-.20),(.40,.012)],cabin)
        p.poly([(-.36,.013),(-.19,-.13),(.11,-.15),(.24,-.038)],.27)
    else:
        p.poly([(-.70,-.12),(.01,.04),(.72,-.11),(.49,.29),(.03,.60),(-.44,.30)],hull)
        p.poly([(-.49,-.11),(-.39,-.38),(.32,-.39),(.49,-.11),(.01,-.017)],cabin)
        p.box(-.29,-.33,.26,-.25,.33);p.line([(0,-.39),(0,-.67)],.022,.33)
    if t%2==0:p.line([(-.95,.34),(-.67,.36),(-.29,.35)],.02,.83)

def bird(p,m,t,bg):
    coat=[.23,.65,.47,.81,.33][t%5];wing=coat*.66; breast=min(.94,coat+.18)
    if m<=2:
        # song bird, shore bird, duck
        duck=m==2; tall=m==1
        hx=.40 if not duck else .51;hy=-.30 if not tall else -.47
        by=.12 if not duck else .17
        p.poly([(-.20,by-.05),(-.86,.22 if duck else .43),(-.74,.31 if duck else .56),(-.03,by+.18)],wing)
        p.ell(-.04,by,.51,.255 if duck else .34,coat,-.10 if duck else -.49)
        p.ell(.12,by+.025,.235,.26,breast,-.38)
        p.line([(.19,.04),(hx-.035,hy+.05)],.14 if not tall else .10,coat)
        p.ell(hx,hy,.19,.183,coat)
        p.poly([(hx+.12,hy-.003),(hx+(.39 if tall or duck else .30),hy+.04),(hx+.13,hy+.075)],.45)
        p.ell(hx+.07,hy-.052,.033,.034,.08);p.ell(-.16,by-.01,.36,.17,wing,-.39)
        if not duck:
            for x in [-.12,.075]:p.line([(x,by+.21),(x-.03,.57 if not tall else .70),(x+.055,.60 if not tall else .72)],.017,.31)
    elif m==3: # wings in flight
        p.poly([(-.97,-.12),(-.63,-.28),(-.28,-.13),(-.04,.018),(.25,-.22),(.55,-.35),(.97,-.30),(.62,-.20),(.34,-.08),(.07,.19),(-.09,.18),(-.41,-.012),(-.69,-.15)],coat)
        p.ell(.017,.122,.105,.20,wing,-.15)
    else:
        p.ell(0,.17,.36,.47,coat);p.ell(.024,-.28,.26,.26,coat);p.ell(.04,.18,.26,.32,breast)
        p.poly([(-.07,-.26),(.12,-.26),(.07,-.15)],.44)
        for x in [-.12,.17]:p.ell(x,-.32,.033,.035,.09)
        for x in [-.14,.15]:p.line([(x,.54),(x,.66),(x+.06,.69)],.018,.29)

def walking(p,kind,t,mode):
    coat={'deer':[.53,.39,.61,.45], 'horse':[.27,.48,.69,.37], 'dog':[.64,.38,.78,.29], 'cat':[.54,.36,.69,.28], 'frog':[.39,.55,.33,.48]}[kind][t%4]
    shade=coat*.74; light=min(.93,coat+.16)
    if kind=='frog':
        p.ell(-.13,.095,.49,.29,coat,-.21);p.ell(.34,-.08,.31,.21,coat,-.21)
        p.ell(.39,-.24,.093,.091,light);p.ell(.41,-.249,.045,.047,.09)
        p.line([(.32,-.02),(.55,-.023),(.64,-.085)],.016,shade)
        p.ell(-.34,.13,.25,.17,shade,-.55)
        p.line([(-.41,.105),(-.66,.37),(-.27,.32),(-.48,.47)],.054,coat)
        p.line([(.14,.10),(.31,.31),(.55,.34)],.049,shade)
        for x in [-.48,.55]: p.line([(x,.35 if x>0 else .47),(x+.16,.39 if x>0 else .49)],.025,coat)
        if t%2:
            for x,y in [(-.30,.04),(-.05,-.05),(.14,-.09),(-.14,.17)]:p.ell(x,y,.061,.042,shade)
        return
    long=kind in ('horse','deer'); dog=kind=='dog'; cat=kind=='cat'; deer=kind=='deer'
    by=-.015 if long else .025; leg=.67 if long else .44; r=.041 if deer else .056
    if cat:r=.053
    # Articulated legs: alternating far and near legs.
    for j,(x,advance) in enumerate([(-.39,-.14),(.28,.12),(-.41,.075),(.24,-.055)]):
        gait=1 if mode==0 else 1.9
        foot=x+advance*gait; knee=x+(-.07 if x<0 else .045)
        p.line([(x,by+.07),(knee,by+.31),(foot,leg-.04),(foot+.045,leg)],r*(.84 if j<2 else 1),shade if j<2 else coat)
        if long:p.ell(foot+.027,leg,.05,.031,.18)
    p.ell(-.075,by,.58,.235 if long else .216,coat,-.045)
    p.ell(-.40,by+.065,.22,.22,coat);p.ell(.24,by+.06,.20,.20,coat)
    if long:
        hx=.62;hy=-.55 if deer else -.53
        p.poly([(.10,-.14),(.35,-.30),(.52,hy-.06),(.67,hy+.01),(.52,-.14),(.39,.09)],coat)
        p.ell(hx+.055,hy+.07,.224 if not deer else .19,.109,coat,.52 if not deer else .26)
        p.ell(.82,hy+.16 if not deer else hy+.11,.085,.066,shade)
        p.ell(.669,hy+.008,.027,.026,.11)
        for x in [.54,.64]:p.poly([(x-.052,hy+.017),(x-.045,hy-.14),(x+.05,hy-.035)],coat)
        if deer:
            p.ell(-.56,-.06,.14,.093,light,-.50)
            if t%3==0:
                p.line([(.56,hy-.05),(.47,hy-.20),(.39,hy-.31)],.022,shade)
                p.line([(.485,hy-.19),(.52,hy-.32)],.020,shade)
        else:
            p.line([(.15,-.17),(.37,-.36),(.52,hy-.07)],.046,shade*.76)
            p.line([(-.56,-.10),(-.67,-.064),(-.71,.15),(-.77,.39)],.081,shade*.75)
    else:
        hx=.45;hy=-.24
        p.ell(.33,-.08,.205,.22,coat,-.48);p.ell(hx,hy,.19,.182,coat)
        p.ell(.58,hy+.062,.158 if dog else .101,.086,light);p.ell(.71 if dog else .66,hy+.055,.041,.032,.13)
        p.ell(.524,hy-.047,.030,.027,.08)
        if cat or t%3==0:p.poly([(.30,hy-.065),(.29,hy-.30),(.47,hy-.15)],coat)
        else:p.ell(.345,hy+.005,.091,.17,shade,-.19)
        p.line([(-.54,-.03),(-.70,-.10),(-.78,-.29),(-.83,-.42 if cat else -.27)],.044 if cat else .060,coat)

def petface(p,kind,t,mode):
    cat=kind=='cat'; wolf=not cat and t%4==0
    coat=[.65,.39,.78,.48,.28,.57][t%6]; pale=min(.94,coat+.17);shade=coat*.69
    three=mode==3; d=.135 if three else 0
    # chest, chin, skull, ears
    p.ell(-d,.68,.57,.56,coat);p.ell(-d+.04,.62,.34,.43,pale)
    if cat or wolf:
        for sg in [-1,1]:
            p.poly([(sg*.54,-.16),(sg*.51,-.78),(sg*.13,-.46)],coat)
            p.poly([(sg*.47,-.33),(sg*.47,-.64),(sg*.25,-.44)],shade)
    p.ell(0,-.09,.54 if cat else .50,.48 if cat else .54,coat)
    if not cat and not wolf:
        for sg in [-1,1]:p.ell(sg*.465,-.15,.16,.39,shade,-sg*.23)
    for sg in [-1,1]:
        x=d+sg*(.237 if cat else .213)*(1-.19*sg if three else 1);y=-.168
        if cat:
            p.line([(x-.104,y-.020*sg),(x,y-.046),(x+.094,y+.011*sg)],.019,.20)
            p.ell(x,y,.079,.039,.71,sg*-.12);p.ell(x,y,.024,.041,.09)
        else:
            p.ell(x,y-.02,.106,.081,shade);p.ell(x,y-.017,.057,.052,.10)
        p.ell(x-.014,y-.034,.014,.013,.91)
    if cat:
        for sg in [-1,1]:p.ell(d+sg*.095,.100,.126,.105,pale)
        p.poly([(d-.077,.044),(d+.079,.044),(d,.117)],.25);p.line([(d,.12),(d,.17),(d+.065,.195)],.012,.22)
        if t%2:
            for x in [-.14,0,.14]:p.poly([(x-.036,-.51),(x+.055,-.49),(x+.016,-.29)],shade)
    else:
        p.ell(d+.015,.095,.235,.196,pale);p.ell(d+.026,.040,.128,.084,.12)
        if t%3==1:p.ell(d+.032,.25,.112,.12,.22);p.ell(d+.046,.303,.079,.094,.52)
        else:p.line([(d-.12,.192),(d+.01,.213),(d+.14,.183)],.016,.24)

def sitting(p,kind,t):
    cat=kind=='cat';coat=[.68,.34,.80,.50][t%4];sh=coat*.72
    p.ell(-.065,.16,.37,.46,coat,-.18);p.ell(-.21,.33,.27,.25,coat)
    p.ell(.145,-.29,.255,.24,coat)
    for s in [-1,1]:
        x=.14+s*.148
        if cat or t%4==0:p.poly([(x-.11,-.38),(x+s*.05,-.64),(x+.10,-.43)],coat)
        else:p.ell(x+s*.05,-.30,.085,.18,sh,s*-.20)
        p.ell(.15+s*.095,-.31,.028,.028,.12)
        p.line([(.03+s*.135,.17),(.05+s*.16,.57),(.10+s*.17,.61)],.054,coat if s==1 else sh)
    p.ell(.162,-.221,.056 if not cat else .038,.035,.17)
    p.line([(-.29,.35),(-.52,.43),(-.62,.34),(-.61,.20)],.065,coat)

def ungulate_portrait(p,kind,t):
    deer=kind=='deer';c=[.57,.36,.66,.47][t%4]; sh=c*.69
    p.ell(-.12,.65,.44,.66,c,-.15)
    p.poly([(-.33,-.28),(-.45,-.72),(-.26,-.63),(-.12,-.38)],c);p.poly([(.11,-.40),(.39,-.75),(.38,-.47),(.23,-.26)],c)
    p.ell(.032,-.18,.30,.41,c);p.ell(.071,.09,.205,.32,c)
    p.ell(.084,.28,.189,.134,sh)
    for s in [-1,1]:p.ell(.03+s*.238,-.24,.037,.047,.10);p.ell(.08+s*.093,.29,.033,.026,.12)
    if not deer and t%2==0:p.ell(.073,-.08,.050,.285,.81)
    if deer and t%2==0:
        for s in [-1,1]:
            p.line([(s*.18,-.43),(s*.43,-.62),(s*.52,-.90)],.024,sh)
            p.line([(s*.43,-.62),(s*.37,-.86)],.024,sh)

def frogfront(p,t):
    c=[.55,.39,.65,.46][t%4];sh=c*.72
    p.ell(0,.18,.53,.39,c);p.ell(0,-.045,.49,.29,c)
    for s in [-1,1]:
        p.ell(s*.31,-.257,.118,.123,c);p.ell(s*.32,-.283,.078,.072,.71);p.ell(s*.32,-.28,.052,.038,.09)
        p.ell(s*.44,.23,.19,.20,sh,-s*.57);p.line([(s*.28,.094),(s*.46,.38),(s*.65,.43)],.056,c)
    p.line([(-.415,-.036),(-.22,.046),(0,.074),(.22,.046),(.415,-.036)],.017,sh*.73)

def illustrate(k,i,n):
    # Stratified part/style choices; continuous variations fixed before any check.
    t=(i//5+2*(i%5))%30;m=i%5
    if k==0 and m==3: m=4
    natural=k in [2,3,4,5,6,7]; face=(k in (3,5) and m in (2,3)) or (k in (4,7) and m==4)
    if face:
        scale=rng.uniform(.98,1.96); stretch=rng.uniform(.84,1.13)
    elif natural:
        scale=rng.uniform(.73,1.34); stretch=rng.uniform(.83,1.22)
    else:
        scale=rng.uniform(.66,1.09); stretch=rng.uniform(.77,1.25)
    if k==0 and m==4:ang=rng.uniform(-1.4,1.4)
    else:ang=rng.uniform(-1,1)*(.23 if natural or k==0 else .085)
    dx=rng.normal(0,.10);dy=rng.normal(0,.112); flip=(-1)**(i//5+i%5)
    bg=[.64,.37,.79,.48][(i//3)%4]
    p=Paper(scale,stretch,dx,dy,ang,flip,bg)
    if k==0:plane(p,m,t,bg)
    elif k==1:car(p,m,t,bg)
    elif k==2:bird(p,m,t,bg)
    elif k in (3,5):
        kind='cat' if k==3 else 'dog'
        if m in (2,3):petface(p,kind,t,m)
        elif m==4:sitting(p,kind,t)
        else:walking(p,kind,t,m)
    elif k in (4,7):
        kind='deer' if k==4 else 'horse'
        if m==4:ungulate_portrait(p,kind,t)
        else:walking(p,kind,t,m%2)
    elif k==6:
        if m>=3:frogfront(p,t)
        else:walking(p,'frog',t,m)
    elif k==8:ship(p,m,t,bg)
    elif k==9:truck(p,m,t,bg)
    return p.frame()

# Solid illustrations. Orthographic camera, polygon faces and analytic ray /
# ellipsoid intersection. Each call produces one more independent hypothesis.
class View:
    def __init__(self,az,el,target,scale,aspect,shift,roll,flip,bg):
        c,s=np.cos(az),np.sin(az);f,g=np.cos(el),np.sin(el)
        self.M=np.array([[c,0,-s],[-s*g,f,-c*g],[s*f,g,c*f]])
        self.target=np.array(target);y,x=np.mgrid[0:32,0:32];x=(x+.5)/16-1-shift[0];y=1-(y+.5)/16-shift[1]
        c,s=np.cos(roll),np.sin(roll);self.u=flip*(x*c+y*s)/scale;self.v=(-x*s+y*c)/(scale*aspect)
        self.pix=np.full((32,32),float(bg));self.depth=np.full((32,32),-100.)
    def cam(self,p):return (np.asarray(p)-self.target)@self.M.T
    def ball(self,center,r,color,axis=None):
        r=np.array(r,dtype=float)
        if axis is not None:
            # Local x follows the specified bone. Other axes span its section.
            e=np.array(axis);e=e/(np.linalg.norm(e)+1e-15)
            helper=[0.,0.,1.] if abs(e[2])<.9 else [0.,1.,0.]
            f=np.cross(helper,e);f/=np.linalg.norm(f);g=np.cross(e,f);ma=self.M@np.stack([e,f,g],axis=1)
        else:ma=self.M
        cx,cy,cz=self.cam(center);u=self.u-cx;v=self.v-cy
        a=(ma/r**2)@ma.T;d=a[0,2]*u+a[1,2]*v
        ds=d*d-a[2,2]*(a[0,0]*u*u+2*a[0,1]*u*v+a[1,1]*v*v-1)
        t=(-d+np.sqrt(np.maximum(0,ds)))/a[2,2];mask=(ds>=0)&(cz+t>=self.depth)
        nx=a[0,0]*u+a[0,1]*v+a[0,2]*t;ny=a[0,1]*u+a[1,1]*v+a[1,2]*t;nz=d+a[2,2]*t
        lit=(-.36*nx+.71*ny+.50*nz)/np.sqrt(nx*nx+ny*ny+nz*nz+1e-14)
        self.pix[mask]=color*(.85+.26*lit[mask]);self.depth[mask]=(cz+t)[mask]
    def bone(self,start,end,r,color,r2=None):
        a,b=np.array(start,dtype=float),np.array(end,dtype=float);d=b-a
        self.ball((a+b)*.5,[np.linalg.norm(d)*.54+r*.23,r,r if r2 is None else r2],color,d)
    def poly(self,pts,color):
        p=self.cam(pts);a,b,c=p[0],p[1],p[2];n=np.cross(b-a,c-a)
        if abs(n[2])<1e-7:return
        u,v=self.u,self.v;q=np.zeros(u.shape,bool)
        for a,b in zip(p,np.roll(p,1,axis=0)):
            if abs(a[1]-b[1])>1e-10:q^=((a[1]>v)!=(b[1]>v))&(u<(b[0]-a[0])*(v-a[1])/(b[1]-a[1])+a[0])
        a=p[0];depth=a[2]-((u-a[0])*n[0]+(v-a[1])*n[1])/n[2];mask=q&(depth>=self.depth-1e-8)
        lit=np.dot(n,[-.30,.70,.52])*np.sign(n[2])/np.linalg.norm(n)
        self.pix[mask]=color*(.86+.22*lit);self.depth[mask]=depth[mask]
    def loft(self,segments,color,top=None):
        # x, bottom, top, half-width at bottom, half-width at top
        prev=None
        for x,b,t,wb,wt in segments:
            pts=[(x,b,-wb),(x,t,-wt),(x,t,wt),(x,b,wb)]
            if prev is None:self.poly(pts,color)
            else:
                for i in range(4):self.poly([prev[i],pts[i],pts[(i+1)%4],prev[(i+1)%4]],top if i==1 and top is not None else color)
            prev=pts
        self.poly(prev,color)
    def box(self,x0,x1,y0,y1,w,color):self.loft([(x0,y0,y1,w,w),(x1,y0,y1,w,w)],color)
    def tire(self,x,h,w,r,hub):
        for side in [-1,1]:
            self.ball((x,h,side*w),(r,r,r*.51),.098)
            self.ball((x,h,side*(w+r*.38)),(r*.58,r*.58,r*.17),hub)

def solid_car(p,t):
    col=[.77,.34,.90,.57,.28,.52][t%6];glass=[.32,.47,.25,.57][(t//3)%4]
    van=t%5==1; ht=.88 if van else .74; rear=-.87 if van or t%4==2 else -.69
    p.loft([(-.99,.16,.32,.37,.38),(-.91,.15,.44,.42,.41),(-.46,.135,.46,.425,.41),(.48,.135,.445,.425,.409),(.91,.17,.397,.402,.391),(.98,.205,.32,.33,.35)],col)
    p.loft([(rear,.426,.45,.39,.388),(rear+.22,.447,ht,.405,.315),(.14,.445,ht-.012,.407,.315),(.51,.43,.46,.402,.387)],col)
    # Windshield planes and side glass. The pillars and roof are left in paint.
    p.poly([(.174,ht-.031,-.296),(.174,ht-.031,.296),(.473,.479,.360),(.473,.479,-.360)],glass)
    p.poly([(rear+.194,ht-.023,-.292),(rear+.194,ht-.023,.292),(rear+.025,.47,.364),(rear+.025,.47,-.364)],glass*.9)
    for s in [-1,1]:
        p.poly([(rear+.086,.477,s*.403),(rear+.248,ht-.042,s*.341),(-.141,ht-.045,s*.343),(-.142,.477,s*.405)],glass*.88)
        p.poly([(-.097,.477,s*.406),(-.098,ht-.043,s*.341),(.113,ht-.054,s*.344),(.407,.477,s*.405)],glass)
        p.poly([(-.69,.20,s*.425),(.70,.20,s*.425),(.67,.232,s*.425),(-.68,.232,s*.425)],col*.70)
        p.ball((.33,.49,s*.422),(.067,.036,.047),col)
        p.ball((.927,.347,s*.292),(.051,.039,.105),.93)
        p.ball((-.953,.352,s*.314),(.047,.041,.079),.32)
    p.box(.965,.986,.231,.286,.233,.21);p.box(.981,.993,.235,.275,.092,.79)
    p.tire(-.615,.184,.400,.184,.69 if t%2 else .48);p.tire(.62,.184,.398,.184,.69 if t%2 else .48)

def solid_truck(p,t):
    col=[.83,.36,.77,.56,.91][t%5];cab=[.78,.42,.30,.86][(t//2)%4];glass=.32 if t%3 else .48
    short=t%3==1;w=.38 if short else .29; h=.99 if short else .78; split=.29 if short else .36
    p.box(-1.,split,.288,h,w,col);p.box(-.97,.90,.164,.255,w*.88,.22)
    p.loft([(split+.03,.206,.68,w,w*.89),(.48,.181,h*.89,w*1.03,w*.91),(.75,.18,h*.89,w*1.03,w*.93),(.85,.18,h*.62,w*1.02,w*.96)],cab)
    hood=.40 if not short and t%2==0 else h*.58
    p.loft([(.78,.18,hood,w,w*.95),(.97,.18,hood-.03,w*.96,w*.90)],cab)
    p.poly([(.76,h*.865,-w*.86),(.76,h*.865,w*.86),(.845,h*.626,w*.87),(.845,h*.626,-w*.87)],glass)
    for s in [-1,1]:
        p.poly([(.49,h*.61,s*w*1.015),(.50,h*.827,s*w*.968),(.718,h*.827,s*w*.97),(.785,h*.61,s*w*1.015)],glass)
        p.ball((.967,.30,s*w*.78),(.018,.039,.049),.91)
    p.box(.964,.983,.219,hood-.05,w*.59,.37)
    for i in range(3):
        y=.23+i*(hood-.29)/3;p.box(.982,.987,y,y+.011,w*.58,.63)
    for x in ([-.73,-.43,.72] if short else [-.82,-.59,.22,.74]):p.tire(x,.164,w*.97,.157,.56)

def solid_plane(p,t):
    col=[.82,.37,.92,.63][t%4];wing=col*.88
    p.ball((.01,0,0),(1.,.087,.092),col)
    for s in [-1,1]:
        p.poly([(.19,.016,s*.04),(-.30,.09,s*.94),(-.43,.095,s*.91),(-.19,.02,s*.055)],wing)
        p.poly([(-.59,.041,s*.036),(-.82,.107,s*.33),(-.94,.105,s*.29),(-.82,.045,s*.02)],wing)
        if t%4!=1:p.ball((-.036,-.010,s*.36),(.153,.059,.049),col*.82)
        p.bone((-.62,.015,s*.077),(.65,.015,s*.064),.014,col*.40)
    p.poly([(-.65,.05,0),(-.84,.39,0),(-.96,.40,0),(-.91,.045,0)],col*.82)
    p.ball((.713,.042,0),(.099,.041,.060),.25)

def solid_ship(p,t):
    col=[.28,.71,.19,.85,.55][t%5];deck=[.88,.71,.83][(t//3)%3]
    p.loft([(-.96,-.10,.105,.105,.22),(-.79,-.15,.09,.24,.29),(.51,-.15,.10,.23,.29),(.81,-.08,.14,.13,.20),(.99,.11,.20,.011,.014)],col,deck*.72)
    if t%4==1:
        p.bone((-.15,.1,0),(-.15,1.21,0),.013,.39)
        p.poly([(-.17,1.19,0),(-.19,.20,0),(-.81,.20,.05)],deck)
        p.poly([(-.10,1.01,0),(.65,.225,0),(-.1,.245,.07)],.78)
    elif t%4==2:
        p.loft([(-.55,.10,.18,.16,.15),(-.22,.10,.36,.19,.16),(.13,.10,.36,.19,.16),(.44,.10,.16,.18,.16)],deck)
        p.poly([(.16,.35,-.15),(.16,.35,.15),(.35,.228,.145),(.35,.228,-.145)],.29)
    elif t%4==0:
        p.box(-.68,-.30,.09,.34,.19,deck);p.box(-.61,-.32,.34,.46,.17,deck);p.box(-.58,-.47,.45,.60,.088,.31)
        for x in [-.21,.03,.27]:p.box(x,x+.22,.10,.21,.235,col*.8)
    else:
        p.loft([(-.77,.09,.17,.23,.19),(-.50,.09,.34,.24,.19),(.35,.11,.34,.23,.18),(.64,.11,.16,.21,.19)],deck)
        for s in [-1,1]:p.poly([(-.54,.255,s*.222),(.43,.255,s*.208),(.40,.286,s*.201),(-.52,.286,s*.215)],.31)
        p.box(-.24,-.06,.34,.52,.094,.32)

def solid_ungulate(p,t,kind):
    deer=kind=='deer';c=([.56,.43,.61,.48,.52] if deer else [.36,.50,.26,.67,.79])[t%5];sh=c*(.83 if deer else .62)
    p.ball((-.08,.60,0),(.53,.215,.166),c);p.ball((-.41,.61,0),(.228,.207,.177),c)
    if deer:p.ball((-.53,.64,0),(.12,.16,.156),min(.91,c+.16))
    for s in [-1,1]:
        for j,x in enumerate([-.40,.24]):
            phase=(1 if j else -1)*s*(.17 if t%2 else .075)
            knee=(x+phase*.48+(-.10 if j==0 else 0),.30,s*.12);hoof=(x+phase,.045,s*.138)
            p.bone((x,.65,s*.107),knee,.090 if not deer else .072,c)
            p.bone(knee,hoof,.040 if not deer else .031,c);p.ball(hoof,(.052,.046,.039),.22)
    hy=1.09 if t%4 else .92;hx=.50 if t%4 else .64
    p.bone((.20,.66,0),(hx-.004,hy,0),.143 if not deer else .119,c,.127 if not deer else .096)
    if not deer:p.bone((.10,.771,0),(hx-.044,hy+.047,0),.056,sh*.75)
    p.ball((hx+.055,hy,0),(.158,.123,.115),c)
    muzzle=(hx+(.270 if not deer else .23),hy-(.17 if not deer else .06),0)
    p.bone((hx+.058,hy+.008,0),muzzle,.099 if not deer else .082,c)
    p.ball(muzzle,(.075,.078,.093),sh)
    for s in [-1,1]:
        p.bone((hx-.018,hy+.060,s*.058),(hx-.069,hy+.217,s*(.145 if deer else .087)),.049 if deer else .040,c)
        p.ball((hx+.10,hy+.025,s*.107),(.026,.024,.017),.11)
        p.ball((muzzle[0]+.053,muzzle[1]+.016,s*.052),(.021,.014,.014),.16)
        if deer and t%3==0:
            p.bone((hx-.036,hy+.10,s*.06),(hx-.16,hy+.26,s*.19),.023,sh)
            p.bone((hx-.15,hy+.25,s*.18),(hx-.20,hy+.38,s*.22),.018,sh)
            p.bone((hx-.115,hy+.21,s*.13),(hx-.068,hy+.34,s*.14),.018,sh)
    if deer:p.bone((-.54,.69,0),(-.67,.62,0),.054,min(.95,c+.19))
    else:
        p.bone((-.52,.72,0),(-.67,.51,0),.084,sh);p.bone((-.65,.54,0),(-.76,.25,.018),.074,sh)

def solid_pet(p,t,cat,sitting):
    c=[.64,.38,.81,.49,.28,.59][t%6];sh=c*.70;cream=min(.94,c+.18)
    bx=-.12;hx=.42;hy=.62
    if sitting:
        hx=.21;hy=.84;p.bone((-.14,.24,0),(.14,.63,0),.22,c,.19)
        p.ball((-.22,.21,0),(.30,.20,.20),c)
    else:p.ball((-.10,.37,0),(.47,.205,.177),c)
    for s in [-1,1]:
        for j,x in enumerate([-.40,hx-.16]):
            xx=x+s*(.049 if j else .081);p.bone((x,.37 if not sitting else (.56 if j else .28),s*.115),(xx,.066,s*.137),.05 if cat else .062,c)
            p.ball((xx+.036,.048,s*.137),(.079,.044,.054),c)
    p.bone((hx-.20,.40,0),(hx-.05,hy,0),.155,c)
    p.ball((hx,hy,0),(.174,.209,.199 if cat else .183),c)
    for s in [-1,1]:
        if cat or t%4==0:
            a=(hx+.068,hy+.112,s*.057);b=(hx+.026,hy+.089,s*.19);tip=(hx-.033,hy+.317,s*.177)
            p.poly([a,b,tip],sh);p.poly([a,tip,(hx-.099,hy+.112,s*.093)],c);p.poly([b,(hx-.099,hy+.112,s*.093),tip],c)
        else:p.bone((hx-.04,hy+.125,s*.14),(hx-.12,hy-.15,s*.205),.078,sh)
        eye=(hx+.131,hy+.044,s*.113)
        p.ball(eye,(.034,.032,.049 if cat else .035),.64 if cat else .14)
        p.ball((eye[0]+.022,eye[1]+.002,eye[2]+s*.009),(.015,.023,.017),.082)
        if cat and t%3!=0:
            p.bone((hx+.005,hy+.199,s*.051),(hx+.117,hy+.126,s*.061),.018,sh*.77)
            p.bone((hx+.044,hy-.01,s*.182),(hx+.118,hy-.030,s*.14),.016,sh*.78)
        p.ball((hx+.152,hy-.048,s*(.063 if cat else .069)),(.071 if cat else .132,.071,.088),cream)
    if cat:p.ball((hx+.219,hy-.008,0),(.027,.024,.045),.30)
    else:
        p.ball((hx+.205,hy-.085,0),(.101,.066,.105),sh*.77);p.ball((hx+.262,hy-.006,0),(.060,.049,.071),.113)
        if t%3==0:p.ball((hx+.263,hy-.124,0),(.053,.069,.051),.52)
    p.bone((-.49,.29,0),(-.65,.43,.045),.052,c);p.bone((-.64,.41,.045),(-.73,.61,.13),.048,c)

def solid_bird(p,t):
    c=[.30,.58,.42,.77,.49][t%5];w=c*.67
    duck=t%4==0; hx=.34;hy=.46 if duck else .59
    p.ball((-.04,.25,0),(.42,.188 if duck else .22,.16),c,(1,.15 if duck else .55,0))
    p.bone((.13,.29,0),(hx-.03,hy,0),.11,c);p.ball((hx,hy,0),(.147,.15,.137),c)
    for s in [-1,1]:
        p.bone((.10,.34,s*.12),(-.35,.19,s*.105),.093,w,.069)
        p.bone((-.27,.21,s*.08),(-.66,.145,s*.093),.051,w)
        p.ball((hx+.069,hy+.022,s*.114),(.027,.027,.015),.07)
        if not duck:
            p.bone((-.06,.15,s*.058),(-.071,-.04,s*.079),.017,.32);p.bone((-.07,-.04,s*.079),(.008,-.059,s*.09),.013,.36)
    beak=.23 if duck or t%3==0 else .159
    for s in [-1,1]:p.poly([(hx+.11,hy+.022,s*.028),(hx+.12,hy-.055,s*.048),(hx+beak+.066,hy-.021,0)],.48)

def solid_frog(p,t):
    c=[.52,.38,.64,.48,.58][t%5];sh=c*.77
    p.ball((-.072,.211,0),(.37,.177,.234),c,(1,.14,0));p.ball((.245,.28,0),(.233,.135,.222),c)
    for s in [-1,1]:
        p.ball((.233,.372,s*.139),(.089,.093,.075),c)
        p.ball((.267,.40,s*.182),(.055,.051,.042),.70);p.ball((.282,.401,s*.209),(.041,.023,.021),.094)
        p.bone((.167,.23,s*.15),(.06,.131,s*.32),.044,sh);p.bone((.06,.131,s*.32),(.31,.044,s*.38),.044,c)
        p.bone((-.34,.19,s*.13),(-.147,.177,s*.28),.112,c);p.bone((-.157,.155,s*.275),(-.51,.047,s*.32),.072,sh)
        for xx,zz in [(-.51,.32),(.31,.38)]:
            for d in [-.055,.065]:p.bone((xx,.05,s*zz),(xx+.12,.021,s*(zz+d)),.015,c)
        if t%3!=0:
            for xx,yy,zz in [(-.26,.304,.12),(-.13,.337,.15),(.002,.333,.14)]:p.ball((xx,yy,s*zz),(.044,.021,.031),sh)

def volume(k,i):
    t=i//5+i%5*7;u=rng.uniform; close=i%5>=3 and k in [3,5,4,7,2,6];sitting=i%3!=0
    az=u(-1.08,1.47);el=u(.045,.37);ctr=[0,.40,0];scale=u(.70,1.12);roll=u(-.065,.065)
    if k==0:
        az=u(-1.5,1.5);el=u(-1.4,.95);ctr=[0,0,0];roll=u(-.31,.31)
    elif k==8:ctr=[0,.25 if t%4==1 else .14,0];el=u(.016,.26)
    elif k==9:ctr=[0,.41,0]
    elif k in [4,7]:
        if close:
            scale=u(1.20,2.49);ctr=[.43,.86,0];az=u(.55,1.68)
        else:scale=u(.81,1.30);ctr=[0,.56,0]
    elif k in [3,5]:
        hy=.84 if sitting else .62;hx=.21 if sitting else .42
        if close:az=u(.91,1.80);ctr=[hx,hy-.045,0];scale=u(2.07,3.86)
        else:ctr=[-.01,.42,0];scale=u(1.10,1.69)
        el=u(-.12,.48);roll=u(-.15,.15)
    elif k==6:
        ctr=[0,.22,0];scale=u(1.55,2.33) if close else u(.97,1.65);el=u(.14,.98)
    elif k==2:
        ctr=[-.01,.31,0];scale=u(1.48,2.12) if close else u(.86,1.45);roll=u(-.17,.17)
    v=View(az,el,ctr,scale,u(.89,1.14),rng.normal(0,.11,2),roll,1 if i%2 else -1,[.44,.72,.56,.79,.34][i%5 if i%7==0 else (i//3)%5])
    if k in [1,9] and i%3!=0:
        # Shallow contact shadow below the chassis.
        v.ball((-.018,.011,0),(.91,.017,.40),.265 if i%3==1 else .383)
    if k==0:solid_plane(v,t)
    elif k==1:solid_car(v,t)
    elif k==2:solid_bird(v,t)
    elif k in [3,5]:solid_pet(v,t,k==3,sitting)
    elif k in [4,7]:solid_ungulate(v,t,'deer' if k==4 else 'horse')
    elif k==6:solid_frog(v,t)
    elif k==8:solid_ship(v,t)
    else:solid_truck(v,t)
    return v.pix

# Feature transform used to describe just ONE template at a time. Eight signed
# orientations are combined into contrast-insensitive orientations with a small
# signed component. Cell histograms get a soft saturation and local competition.
y,x=np.mgrid[-1:2,-1:2]; dker=[]
for a in range(8):
    v=(x*np.cos(a*np.pi/4)+y*np.sin(a*np.pi/4))*np.exp(-(x*x+y*y)/1.8)
    dker.append(v/np.abs(v).sum()*2)
dker=np.array(dker); lum=np.array([.27,.55,.18]); H=8
orient=np.zeros((8,8)); orientopp=np.zeros((8,8))
for a in range(8):
    orient[a,a]=1.;orient[a,(a+4)%8]=.70
for a in range(8):orientopp[a]=orient[a]-.67*orient.mean(0)
z=norm('local_luminance',conv('luminance','image',lum[None,:,None,None]))
grad=conv('signed_gradients',z,dker[:,None],padding=0)
gradpos=rel('gradient_magnitudes',grad)
cells=pool('overlapping_cells','avg',gradpos,6,6,3,3) # 9 x 9
# Cap cell strength: less sensitive to blown highlights. Use a fixed, weak
# subtraction of the other directions rather than divisive normalization.
a=conv('orientation_competition',cells,orientopp[:,:,None,None]); a=rel('orientation_positive',a)
b=rel('orientation_excess',conv('clip_offset',a,np.eye(8)[:,:,None,None],-.65))
b=conv('remove_excess',b,-np.eye(8)[:,:,None,None]); a=plus('oriented_cells',a,b)
hog=norm('orientation_layout',a)
low=pool('coarse_luminance','avg',z,4);low=norm('luminance_layout',low)
def unit(x):
    x=x-x.mean(axis=tuple(range(1,x.ndim)),keepdims=True)
    return x/max(1e-9,np.linalg.norm(x))
def describe(image):
    a=(image-image.mean())/np.sqrt(image.var()+1e-5)
    q=np.lib.stride_tricks.sliding_window_view(a,(3,3))
    g=np.maximum(np.einsum('ijxy,kxy->kij',q,dker,optimize=False),0)
    c=np.lib.stride_tricks.sliding_window_view(g,(6,6),axis=(1,2))[:,::3,::3].mean((-1,-2))
    c=np.clip(np.einsum('ij,jxy->ixy',orientopp,c),0,.65)
    l=a.reshape(8,4,8,4).mean((1,3))[None]
    # Each template gets its own centering and norm, never shared statistics.
    return unit(c)/np.sqrt(c.size),unit(l)/np.sqrt(l.size)
N=600; wt_h=np.empty((10*N,8,9,9),np.float32); wt_l=np.empty((10*N,1,8,8),np.float32)
for k in range(10):
    for i in range(N):
        art=volume(k,i//2) if i%2 else illustrate(k,i//2,N//2)
        wt_h[k*N+i],wt_l[k*N+i]=describe(art)
s1=dense('shape_histogram_vote',hog,wt_h);s2=dense('shape_shading_vote',low,.37*wt_l)
s=shape('illustrated_hypotheses',plus('matched_illustrations',s1,s2),10,1,N)
# Max across alternative poses, mixing the very best with a mean of subgroup maxima.
sbest=pool('best_pose','max',s,1,N);sg=pool('best_in_groups','max',s,1,50);sg=pool('group_consensus','avg',sg,1,N//50)
shapescore=plus('shape_scores',dense('peak_shape',sbest,2.65*np.eye(10)),dense('consistent_shape',sg,1.35*np.eye(10)))

# A separate scene reader. All pixel predicates below are hand-specified soft
# intervals or intersections of half-spaces in RGB. They do not involve the art.
R,G,B=np.eye(3); V=lum; rawcolor=pool('small_color_cells','avg','image',2)
colors={
 'blue':[(B-R,.024,.13),(B-G,-.004,.080)],
 'water':[(B-R,.022,.12),(G-R,.004,.065)],
 'green':[(G-B,.031,.13),(G-R,-.013,.082)],
 'olive':[(G-B,.045,.13),(G-R,-.060,.091)],
 'warm':[(R-B,.055,.17),(R-G,.020,.10)],
 'tan':[(R-B,.075,.17),(R-G,.012,.10),(G-B,.015,.10)],
 'red':[(R-G,.12,.16),(R-B,.12,.16)],
 'white':[(V,.64,.18),(R-B,-.12,.12),(B-R,-.12,.12)],
 'dark':[(-V,-.29,.19)], 'light':[(V,.50,.30)],
 'gray':[(R-G,-.12,.12),(G-R,-.12,.12),(B-G,-.12,.12),(G-B,-.12,.12),(B-R,-.14,.14),(R-B,-.14,.14)],
 'gold':[(R-B,.19,.19),(G-B,.11,.15),(R-G,-.025,.10)]}
# Apply all clipped linear ramp predicates together. The fuzzy AND is
# max(0, sum(memberships) - (n-1)); all thresholds in input intensity units.
chan=[];cb=[];pidx={}; mix=np.zeros((len(colors),2*sum(map(len,colors.values())))); mixb=[]
for j,(key,tests) in enumerate(colors.items()):
    for v,lo,span in tests:
        ito=len(chan);chan.extend([v/span,v/span]);cb.extend([-lo/span,-lo/span-1]);mix[j,ito:ito+2]=[1,-1]
    pidx[key]=j;mixb.append(1-len(tests))
c1=rel('color_ramps',conv('chromatic_halfspaces',rawcolor,np.array(chan)[:,:,None,None],cb))
c2=rel('color_memberships',conv('color_intersections',c1,mix[:,:,None,None],mixb)); colorgrid=pool('color_map','avg',c2,2)

# Assemble a named list of scalars from a handful of small grids. Each grid
# has spatial mask averages; summing zero-extended branch outputs is concat.
obs={}; sources={}
def register(source,name,w,b=0):
    assert name not in obs; obs[name]=(source,np.array(w),b)
def mkmask(h,w,reg):
    yy,xx=np.mgrid[:h,:w];xx=(xx+.5)/w;yy=(yy+.5)/h
    inner=(abs(xx-.5)<.28)&(abs(yy-.51)<.29)
    return {'all':np.ones_like(xx,dtype=bool),'top':yy<.27,'bottom':yy>.74,'mid':inner,'rim':~inner,'high':yy<.50,
            'low':yy>.50,'flank':abs(xx-.5)>.31,'shoulders':(abs(xx-.5)>.24)&(yy>.27)&(yy<.67),
            'crown':(abs(xx-.5)<.31)&(yy<.40),'foot':(abs(xx-.5)<.31)&(yy>.61)&(yy<.89)}[reg].astype(float)
def measure(src,dims,idx,key,reg='all',gain=1):
    c,h,w=dims; wt=np.zeros(dims); mask=mkmask(h,w,reg);wt[idx]=gain*mask/mask.sum();register(src,key,wt)
for key,idx in pidx.items():
    for reg in ['all','top','bottom','mid','rim','high','low','flank','foot','crown']:
        measure(colorgrid,(len(colors),8,8),idx,key+'_'+reg,reg)

# Absolute and relative edge energy, contrast, oriented texture and straightness.
y,x=np.mgrid[-1:2,-1:2]; logker=np.exp(-(x*x+y*y)/1.5);logker/=-logker.sum();logker[1,1]+=1
ker=np.concatenate([dker,np.stack([logker,-logker])]); zraw='luminance'
g0=rel('raw_edge_halves',conv('raw_edge_filters',zraw,ker[:,None]))
rawcell=pool('raw_edge_cells','avg',g0,6,6,3,3)
for reg in ['all','mid','rim','top','bottom','foot','crown']:
    for src,VV,C,fields in [(rawcell,'raw',10,[('e',list(range(8)),1/8),('hf',[8,9],1)]),
                              (cells,'rel',8,[('e',list(range(8)),1/8),('h',[2,6],1),('v',[0,4],1),('dd',[1,3,5,7],.5)])]:
        for tag,ids,factor in fields:
            wt=np.zeros((C,9,9));mk=mkmask(9,9,reg);wt[ids]=factor*mk/mk.sum();register(src,VV+tag+'_'+reg,wt)
# Contrast fraction: average absolute luminance residual from the picture mean
# is not directly present in z, so also measure coarse absolute bandpasses.
# Long straight contours: local signed differences averaged along the tangent
# before rectification; a curved or irregular edge cancels in the integral.
long_scores={}
for direct in ['horizontal','vertical','slash','backslash']:
    yy,xx=np.mgrid[-3:4,-3:4]; ang={'horizontal':np.pi/2,'vertical':0,'slash':np.pi/4,'backslash':-np.pi/4}[direct]
    v=xx*np.cos(ang)+yy*np.sin(ang);u=-xx*np.sin(ang)+yy*np.cos(ang)
    f=v*np.exp(-v*v/(2*.75**2)-u*u/(2*2.9**2));f*=2/np.abs(f).sum();long_scores[direct]=f
longker=np.array([sg*f for f in long_scores.values() for sg in [1,-1]])
straight=rel('long_edge_halves',conv('bar_edges',z,longker[:,None],stride=2));straight=pool('bar_edge_grid','avg',straight,3,3,2,2)
for j,direct in enumerate(long_scores):
    for reg in ['all','mid','rim','bottom','top']:
        mk=mkmask(6,6,reg);wt=np.zeros((8,6,6));wt[2*j:2*j+2]=mk/mk.sum();register(straight,'long_'+direct+'_'+reg,wt)
# Whole-image spatial symmetry at coarse scale. Search a few axes around
# the middle, using channel-un-normalized luminance for the absolute differences.
small=pool('symmetry_cells','avg',z,2);symw=[];items={}
for ax in [6.5,7.5,8.5]:
    for reg in ['mid','top','low']:
        group=[]
        for y in ({'mid':range(3,13,2),'top':range(1,8,2),'low':range(8,15,2)}[reg]):
            for delta in [1.5,3.5,5.5]:
                a,b=int(ax-delta),int(ax+delta);v=np.zeros((1,16,16));v[0,y,a]=1;v[0,y,b]=-1
                group.extend([len(symw),len(symw)+1]);symw.extend([v,-v])
        items['sym_%s_%s'%(reg,ax)]=group
sym=rel('mirror_differences',dense('mirror_pairs',small,symw))
for key,ids in items.items():
    v=np.zeros(len(symw));v[ids]=2/len(ids);register(sym,key,v)

# A dark spot next to a brighter rim, often produced by wheel hubs or an eye.
# Pair contrasts are kept signed until after two wheels and the rocker/sill
# between them have contributed. Each detector is a hand-described part model.
Y,X=np.mgrid[0:16,0:16]; X=X+.5; Y=Y+.5
wheel=[]; facew=[]
def field(x,y,sx,sy):
    g=np.exp(-.5*(((X-x)/sx)**2+((Y-y)/sy)**2));return g/g.sum()
def tire(x,y,r):
    # bottom and sides of rubber darker than the road and body just outside.
    dist=np.sqrt((X-x)**2+(Y-y)**2);g=np.exp(-.5*((dist-r)/(.40+r*.14))**2);g/=g.sum()
    return .8*field(x,y,r*1.70,r*1.48)+.2*field(x,y,r*.39,r*.39)-g
for y in [8.4,10,11.6,13.2]:
    for x in [6.8,8,9.2]:
        for d in [5.4,7.6,9.7]:
            for tilt in [-.055,.055]:
                r=.13*d+.12
                w=(tire(x-d/2,y-d*tilt/2,r)+tire(x+d/2,y+d*tilt/2,r))/2
                w+=.24*(field(x,y+1.65,d*.36,.65)-field(x,y,d*.31,.53));wheel.append(w)
for cy in [4.5,6.5,8.5,10.5]:
    for x in [6.5,8,9.5]:
        for d in [2.8,4.4,6.3,7.8]:
            for tilt in [-.12,.12]:
                out=np.zeros_like(X)
                for sg in [-1,1]:
                    px=x+sg*d/2;py=cy+sg*d*tilt/2
                    out+=.35*(field(px,py,.28*d,.24*d)-field(px,py,.115*d,.083*d))
                out+=.3*(field(x,cy+.46*d,.24*d,.20*d)-field(x,cy+.46*d,.14*d,.095*d));facew.append(out)
for ws,tag in [(wheel,'wheel_pair'),(facew,'facial_triangle')]:
    det=shape(tag+'_bank',dense(tag+'_contrasts',small,ws),1,1,len(ws));det=pool(tag+'_peak','max',det,1,len(ws));register(det,tag,np.ones((1,1,1)))

# Compile a high-level hand-written rule table from linear observables and
# clipped soft predicates. No class-dependent quantities are inferred.
keys=list(obs); keynum={k:i for i,k in enumerate(keys)}; bases=[]
for name in dict.fromkeys(v[0] for v in obs.values()):
    dims=next(v[1].shape for v in obs.values() if v[0]==name)
    mat=np.zeros((len(keys),*dims),np.float32);bias=np.zeros(len(keys))
    for i,k in enumerate(keys):
        node,w,b=obs[k]
        if node==name:mat[i]=w;bias[i]=b
    bases.append(dense('read_'+name,name,mat,bias))
obsnode=plus('observations',*bases)
class Expr:
    def __init__(self,coef=0,const=0):self.w=np.broadcast_to(coef,(len(keys),)).copy();self.b=const
    def __add__(self,o):
        if isinstance(o,Expr):return Expr(self.w+o.w,self.b+o.b)
        return Expr(self.w,self.b+o)
    __radd__=__add__
    def __sub__(self,o):return self+(-o)
    def __rsub__(self,o):return -self+o
    def __neg__(self):return Expr(-self.w,-self.b)
    def __mul__(self,f):return Expr(self.w*f,self.b*f)
    __rmul__=__mul__
    def __truediv__(self,f):return self*(1/f)
E={k:Expr(v) for k,v in zip(keys,np.eye(len(keys)))}
feats={}
def ramp(name,e,lo,hi):feats[name]=(e-lo)/(hi-lo)
def col(key,reg):return E[key+'_'+reg]
# Scalar color memberships (+ geography). Responses saturate rather than
# extrapolating a blue or green cast arbitrarily far.
for k in ['blue','water','green','olive','warm','tan','red','white','dark','light','gray','gold']:
    for reg in ['all','top','bottom','mid','rim','foot','crown','flank']:
        if k in ['blue','green','tan','warm','olive','water']:rr=(.025,.50)
        elif k in ['red','gold']:rr=(.018,.29)
        elif k=='gray':rr=(.24,.82)
        else:rr=(.055,.60)
        ramp(k+'_'+reg,col(k,reg),*rr)
# Positive soft discrepancies in color/brightness between regions.
for k in ['light','blue','dark','white','green','warm','tan','red','gray']:
    ramp('object_'+k,col(k,'mid')-col(k,'rim'),.018,.37)
    ramp('surround_'+k,col(k,'rim')-col(k,'mid'),.018,.37)
ramp('sky_over_land',col('light','top')-col('light','bottom'),.03,.48)
ramp('dark_over_ground',col('light','bottom')-col('light','top'),.06,.51)
ramp('sky_blue_gradient',col('blue','top')-col('blue','bottom'),.035,.42)
ramp('blue_base',col('water','bottom')-col('water','top'),.035,.44)
ramp('field_base',col('green','bottom')-col('green','top'),.02,.39)
ramp('shadow_band',.5*(col('dark','foot')+col('dark','mid'))-col('dark','bottom'),.025,.37)
ramp('roof_sky',col('white','top')-col('white','mid'),.035,.49)
# Texture / smoothness. Input luminance lies on [0,1] before contrast normalization.
for reg in ['all','top','mid','bottom','rim','foot','crown']:
    ramp('rough_'+reg,E['rawe_'+reg],.027,.086)
    ramp('fine_'+reg,E['rawhf_'+reg],.015,.050)
    ramp('detail_'+reg,E['rele_'+reg],.16,.42)
    ramp('flat_'+reg,-E['rawe_'+reg],-.046,-.011)
    ramp('relflat_'+reg,-E['rele_'+reg],-.24,-.061)
    ramp('horizontal_'+reg,E['relh_'+reg]-.52*(E['relv_'+reg]+E['reldd_'+reg]),.007,.21)
    ramp('vertical_'+reg,E['relv_'+reg]-.52*(E['relh_'+reg]+E['reldd_'+reg]),.004,.17)
for reg in ['all','top','mid','bottom','rim']:
    lh=E['long_horizontal_'+reg];lv=E['long_vertical_'+reg];ld=(E['long_slash_'+reg]+E['long_backslash_'+reg])*.5
    ramp('bars_'+reg,lh-.6*ld-.4*lv,.019,.28)
    ramp('posts_'+reg,lv-.60*ld-.40*lh,.017,.22)
    ramp('square_'+reg,(lh+lv)*.5-ld,.013,.106)
    ramp('slant_'+reg,ld-(lh+lv)*.5,.014,.10)
    ramp('strongbars_'+reg,lh,.31,.85)
ramp('mid_detail',E['rele_mid']-E['rele_rim'],.018,.28)
ramp('below_detail',E['rele_bottom']-E['rele_top'],.02,.31)
ramp('rim_detail',E['rele_rim']-E['rele_mid'],.018,.26)
ramp('complex_picture',E['rele_all'],.23,.51)
ramp('wheels',E['wheel_pair'],.29,.95);ramp('eyes_nose',E['facial_triangle'],.36,1.23)
for reg in ['mid','top','low']:
    for ax in [6.5,7.5,8.5]:ramp('mirror_%s_%s'%(reg,ax),-E['sym_%s_%s'%(reg,ax)],-.90,-.23)
fn=list(feats);mat=[];bias=[]
for f in fn:
    v=feats[f];mat.extend([v.w,v.w]);bias.extend([v.b,v.b-1])
rnodes=rel('attribute_ramps',dense('attribute_thresholds',obsnode,mat,bias));fm=np.zeros((len(fn),len(fn)*2))
for i in range(len(fn)):fm[i,i*2:i*2+2]=[1,-1]
memb=dense('attributes',rnodes,fm)

# Center a cue table at the strength expected in an uninformative natural
# picture. These offsets are reasoning priors (not corpus measurements).
# Columns: plane, car, bird, cat, deer, dog, frog, horse, ship, truck.
base=np.zeros((10,len(fn))); bb=np.zeros(10)
def cue(name,neutral,coeff,scale=1):
    q=scale*np.asarray(coeff);base[:,fn.index(name)]+=q;bb[:]-=q*neutral
cue('blue_top',.17, [1.40,.01,.65,-.38,-.27,-.33,-.60,-.15,1.04,.23])
cue('water_bottom',.12,[.38,-.25,.08,-.48,-.48,-.31,-.25,-.45,1.63,-.35])
cue('blue_mid',.10, [.52,.01,.12,-.30,-.36,-.24,-.32,-.27,.32,-.04])
cue('water_rim',.14, [.46,-.13,.13,-.19,-.24,-.11,-.18,-.19,.33,-.08])
cue('green_rim',.22, [-.79,-.60,.58,-.16,.91,.17,.55,.74,-.78,-.46])
cue('green_bottom',.20,[-.52,-.62,.12,-.28,.78,.25,.72,.85,-.64,-.47])
cue('green_mid',.17, [-.32,-.49,.22,-.30,.32,-.12,1.06,.14,-.33,-.39])
cue('olive_mid',.24, [-.27,-.32,.05,.00,.19,-.13,.64,.01,-.22,-.21])
cue('tan_mid',.29, [-.41,-.09,.07,.36,.81,.68,.21,.73,-.39,-.20])
cue('warm_rim',.27, [-.49,-.13,-.07,.82,.12,.48,.22,.01,-.49,-.19])
cue('red_mid',.08, [-.08,.66,.12,.02,-.35,.02,-.31,-.05,-.07,.62])
cue('gold_mid',.12, [-.23,.03,.35,.18,.18,.23,.28,.02,-.25,.09])
cue('object_red',.07,[.03,.65,.47,-.08,-.47,-.14,-.38,-.28,.04,.60])
cue('object_tan',.16,[-.16,-.07,.19,.24,.42,.44,.17,.46,-.14,-.11])
cue('object_white',.17,[.19,.30,-.21,-.03,-.39,.27,-.53,.02,.40,.71])
cue('gray_rim',.42, [.33,.55,-.21,.22,-.46,.15,-.56,-.49,.20,.43])
cue('white_top',.20, [.59,.01,.06,-.16,-.27,.03,-.66,-.03,.54,.32])
cue('light_top',.54, [.50,-.08,.05,-.06,-.12,.06,-.66,.04,.35,.11])
cue('dark_all',.28, [-.38,.11,-.09,.12,.12,-.03,.52,.04,-.31,.05])
cue('dark_bottom',.29, [-.30,.15,-.02,-.08,.21,-.03,.40,.16,-.18,.14])
cue('roof_sky',.20, [.35,.03,.05,-.11,-.08,-.15,-.37,.19,.35,.22])
cue('sky_over_land',.37, [.27,-.05,-.11,-.33,.05,-.21,-.49,.31,.69,.43])
cue('dark_over_ground',.19, [-.16,.39,-.05,.22,-.17,.19,.31,-.16,-.46,.20])
cue('shadow_band',.28, [-.03,.51,.11,-.09,-.07,.02,-.07,.05,.16,.32])
cue('field_base',.12, [-.14,-.30,-.15,-.20,.46,.19,.22,.61,-.26,-.17])
cue('surround_green',.14,[-.14,-.19,.56,-.13,.41,.15,-.18,.34,-.19,-.15])
cue('blue_base',.09, [-.39,-.14,-.15,-.13,-.13,-.11,-.09,-.12,.85,-.12])
cue('flat_top',.40, [1.02,-.06,.37,-.11,-.18,-.16,-.31,-.13,.63,.26])
cue('relflat_rim',.24, [.65,-.13,.41,-.05,-.24,-.10,-.39,-.21,.22,-.10])
cue('flat_bottom',.32, [.92,.34,.20,-.12,-.49,-.05,-.26,-.44,.17,-.03])
cue('fine_mid',.42, [-.45,-.05,.13,.41,.19,.35,.32,.00,-.30,-.07])
cue('fine_rim',.32, [-.52,.04,-.25,.14,.47,.05,.59,.34,-.20,.05])
cue('complex_picture',.33, [-.47,.13,-.04,.28,.26,.06,.67,.17,-.16,.21])
cue('mid_detail',.41,[.23,.17,.33,.23,-.26,.03,-.23,-.11,-.14,.06])
cue('below_detail',.30, [-.16,-.21,-.04,.02,.18,.08,.23,.04,.58,.18])
cue('bars_all',.29, [.40,.71,-.38,-.41,-.28,-.45,-.16,-.04,.93,.71])
cue('horizontal_mid',.22, [.29,.65,-.28,-.39,-.27,-.35,.01,-.06,.52,.51])
cue('bars_bottom',.24, [-.02,.62,-.21,-.23,-.29,-.28,.00,-.21,.81,.41])
cue('posts_mid',.25, [-.36,-.23,.01,.10,.22,.04,-.28,.43,-.30,.47])
cue('square_mid',.24, [-.13,.50,-.27,-.26,-.19,-.29,-.29,.02,.32,.90])
cue('square_all',.24, [-.12,.48,-.32,-.19,-.25,-.29,-.31,-.12,.12,.63])
cue('slant_mid',.21, [.28,-.34,.43,.13,.03,.14,.27,.09,-.17,-.35])
cue('wheels',.38, [-.16,.85,-.07,-.05,-.13,-.01,-.15,.01,-.14,.54])
cue('eyes_nose',.34, [-.30,.11,.02,.72,-.08,.71,-.03,-.07,-.24,-.20])
# Put subtle corrections on typical ambiguity of drawings: the compact,
# horizontal silhouette alone is not sufficient evidence of a vehicle or frog.
bb+=np.array([-.10,-.09,.06,.17,.11,.18,.00,.09,-.05,-.08])
scene=dense('individual_cues',memb,base,bb)

# Context conjunctions. A rectifier means the context is not awarded when a
# single property is present but the other properties contradict the hypothesis.
# For each rule enter (attribute, weight) pairs, a soft threshold (weighted
# average), and class evidence. Negative weight denotes the complemented test.
rr=[];rb=[];rv=[]
def rule(terms,threshold,votes):
    row=np.zeros(len(fn));b=0;sw=sum(abs(w) for _,w in terms)
    for n,w in terms:
        row[fn.index(n)]+=w/sw
        if w<0:b-=w/sw
    rr.append(row/(1-threshold));rb.append((b-threshold)/(1-threshold));rv.append([votes.get(i,0.) for i in range(10)])
rule([('flat_top',1),('flat_bottom',1),('mid_detail',.6),('green_mid',-.5)],.52,{0:1.04,2:.36,9:-.20})
rule([('blue_top',1),('flat_top',.8),('flat_bottom',1),('bars_bottom',-.7)],.49,{0:1.13,2:.30,8:-.33})
rule([('blue_top',1),('blue_bottom',1),('below_detail',-.5),('square_all',-.5)],.49,{0:.74,8:.15})
rule([('white_top',.9),('flat_top',.9),('gray_rim',.45),('bars_mid',1),('fine_rim',-.65)],.52,{0:.57,8:.43})
rule([('bars_bottom',1),('water_bottom',1),('green_mid',-.45)],.43,{8:1.29,0:-.36})
rule([('bars_all',1),('flat_top',1),('below_detail',1),('white_top',.6)],.46,{8:1.11,9:.21,0:-.16})
rule([('blue_base',1),('sky_over_land',1),('bars_bottom',.65)],.40,{8:.86})
rule([('gray_bottom',1),('flat_bottom',1),('square_mid',.7),('wheels',1)],.45,{1:.99,9:.31})
rule([('square_mid',1),('shadow_band',1),('rough_top',.6),('flat_bottom',.8)],.46,{1:.81,9:.28})
rule([('object_red',1),('gray_bottom',.65),('wheels',.7),('horizontal_mid',.6)],.44,{1:.80,9:.48})
rule([('square_mid',1),('posts_mid',1),('object_white',.75),('wheels',.55)],.43,{9:1.08,1:-.22})
rule([('gray_bottom',.6),('white_top',.7),('square_mid',1),('bars_mid',.5)],.47,{9:.68,1:.08})
rule([('object_white',1),('object_red',.7),('square_all',1)],.43,{9:.67})
rule([('green_rim',1),('object_tan',1),('rough_bottom',.75),('red_mid',-.6)],.47,{4:1.00,7:.65})
rule([('green_top',1),('green_bottom',1),('tan_mid',.65),('rough_rim',.85)],.44,{4:.88,7:.23,6:.27})
rule([('green_rim',1),('flat_rim',1),('mid_detail',1)],.44,{2:.99,4:.05,6:-.18})
rule([('mid_detail',1),('flat_rim',.8),('slant_mid',.5),('object_red',.5)],.46,{2:.84,0:.09})
rule([('green_bottom',1),('object_tan',.7),('sky_over_land',.8),('vertical_mid',.7)],.41,{7:1.06,4:.24})
rule([('field_base',1),('shadow_band',.45),('green_top',-.8),('white_top',.55)],.45,{7:.81,5:.13})
rule([('olive_mid',1),('dark_all',.7),('light_top',-.7),('fine_mid',.6)],.44,{6:1.08,4:.22})
rule([('green_mid',1),('complex_picture',.9),('sky_over_land',-.55)],.43,{6:.91,4:.15,2:-.10})
rule([('dark_all',.8),('warm_rim',.8),('rough_rim',.8),('sky_over_land',-.7)],.48,{6:.56,3:.28})
rule([('warm_rim',1),('eyes_nose',1),('green_bottom',-.6)],.47,{3:.87,5:.68})
rule([('eyes_nose',1),('object_white',.7),('square_all',-.7),('fine_mid',.7)],.47,{5:.82,3:.38})
rule([('warm_mid',1),('fine_mid',.8),('slant_mid',.7),('flat_bottom',.4)],.46,{3:.61,5:.41})
rule([('green_rim',.7),('eyes_nose',1),('object_tan',.8),('flat_rim',.65)],.47,{5:.80,2:.24})
for ax in [6.5,7.5,8.5]:
    rule([('mirror_mid_'+str(ax),1),('eyes_nose',1),('detail_mid',.65),('horizontal_mid',-.5)],.58,{3:.47,5:.47})
    rule([('mirror_low_'+str(ax),.85),('horizontal_mid',1),('eyes_nose',.30),('wheels',1)],.58,{1:.47,9:.19})
context=dense('context_votes',rel('supported_context',dense('context_tests',memb,rr,rb)),np.array(rv).T)
plus('class_scores',shapescore,scene,context)
(D/'network.json').write_text(json.dumps({'layers':L},indent=1)+'\n')
np.savez(D/'weights.npz',**W)


