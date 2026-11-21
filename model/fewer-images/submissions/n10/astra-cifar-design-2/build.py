"""Construct a photo classifier from category geometry and photographic priors.
Only numpy and the standard library. No inputs other than the output path.
"""
import json
import sys
from pathlib import Path
import numpy as np

rng = np.random.default_rng(310572)
U, C, I = rng.uniform, rng.choice, rng.integers
L, W = [], {}

def node(name, op, x, **kw):
    L.append(dict(name=name, op=op, inputs=x if isinstance(x,list) else [x], **kw))
    return name

def conv(name, x, w, b=0, s=1, p=0):
    w=np.asarray(w,'f'); o,c,k,_=w.shape
    W[name+'.W']=w; W[name+'.b']=np.broadcast_to(np.float32(b),(o,)).copy()
    return node(name,'conv',x,out=o,kernel=[k,k],stride=s,padding=p)

def dense(name, x, w, b=0):
    w=np.asarray(w,'f'); W[name+'.W']=w.reshape(len(w),-1)
    W[name+'.b']=np.broadcast_to(np.asarray(b,'f'),(len(w),)).copy()
    return node(name,'dense',x,out=len(w))

def rel(name,x): return node(name,'relu',x)
def pool(name,x,k,mode='avg',stride=None):
    k=(k,k) if isinstance(k,int) else k
    return node(name,mode+'pool',x,size=list(k),stride=list(stride or k))
def shape(name,x,*d):return node(name,'reshape',x,shape=list(d))

# Rasterize category-coordinate primitives by transforming the sampling grid.
# Neutral outside a shape; luminance discontinuities and paint are on two plates.
ss=2; N=32*ss
gy,gx=np.mgrid[:N,:N]; gx=(gx+.5)/N; gy=(gy+.5)/N
# Feature extractor maps an image to 8x8 local descriptors. The same fixed
# filters, zero exterior AFTER centering, and spatial binning apply to a drawing.
t=np.arange(5)-2; yy,xx=np.meshgrid(t,t,indexing='ij')
dirs=[]
for a in np.arange(4)*np.pi/4:
    v=xx*np.cos(a)+yy*np.sin(a); u=-xx*np.sin(a)+yy*np.cos(a)
    h=v*np.exp(-.5*(v/.9)**2-.5*(u/1.12)**2); dirs.append(h/np.maximum(h,0).sum())
h=np.exp(-.5*(xx**2+yy**2)/.65**2); h/=h.sum(); broad=np.exp(-.5*(xx**2+yy**2)/1.6**2); broad/=broad.sum(); h-=broad; dirs.append(h/np.maximum(h,0).sum())
dirs=np.array(dirs,'f')

def describe(im):
    im=im.reshape(32,ss,32,ss).mean((1,3)); im=im-im.mean()
    def shrink(a): return a.reshape(-1,8,4,8,4).mean((2,4))
    pad=np.pad(im,2); views=np.lib.stride_tricks.sliding_window_view(pad,(5,5))
    dx=np.einsum('yxij,kij->kyx',views,dirs,optimize=False)
    p=shrink(dx.clip(0)); n=shrink((-dx).clip(0)); ax=p+n
    # Remove the isotropic part of the four oriented energies at each cell.
    orient=ax[:4]-ax[:4].mean(0)
    groups=[shrink(im), p-n, orient, ax[:4].mean(0,keepdims=True), ax[4:5]]
    # Mean-free correlation in each channel, followed by self normalization of
    # each block. All divisors here belong exclusively to this drawing.
    blocks=[]
    for a,g in zip(groups,[.20,.29,.20,.21,.10]):
        a=a-a.mean((1,2),keepdims=True)
        blocks.append(g*a/(np.sqrt((a*a).sum())+1e-7))
    return np.concatenate(blocks).astype('f')

class Pic:
    def __init__(self, size, height, ang, cx,cy,flip):
        a=np.deg2rad(ang); x=gx-cx; y=gy-cy
        self.x=(x*np.cos(a)+y*np.sin(a))/size*flip
        self.y=(-x*np.sin(a)+y*np.cos(a))/height
        self.base=U(.17,.83); self.im=np.full((N,N),self.base,'f')
        self.fg=U(.10,.90)
        if abs(self.fg-self.base)<.15:self.fg=np.clip(self.base+C([-1,1])*.28,.04,.96)
    def paint(self,m,v):self.im[m]=np.clip(v,0,1)
    def ell(self,x,y,rx,ry,v,ang=0):
        a=np.deg2rad(ang); X=self.x-x; Y=self.y-y
        self.paint(((X*np.cos(a)+Y*np.sin(a))/rx)**2+((-X*np.sin(a)+Y*np.cos(a))/ry)**2<1,v)
    def poly(self,pts,v):
        # even-odd rule handles concave shapes as well as convex ones
        inside=np.zeros((N,N),bool); x0,y0=pts[-1]
        for x1,y1 in pts:
            if y0!=y1:
                inside ^= ((self.y>y0)!=(self.y>y1)) & (self.x<(x1-x0)*(self.y-y0)/(y1-y0)+x0)
            x0,y0=x1,y1
        self.paint(inside,v)
    def box(self,x,y,X,Y,v):self.paint((self.x>x)&(self.x<X)&(self.y>y)&(self.y<Y),v)
    def seg(self,x,y,X,Y,r,R,v):
        dx=X-x;dy=Y-y; f=np.clip(((self.x-x)*dx+(self.y-y)*dy)/(dx*dx+dy*dy+1e-9),0,1)
        self.paint((self.x-x-f*dx)**2+(self.y-y-f*dy)**2<(r+(R-r)*f)**2,v)

def atmosphere(p,kind):
    # Plain studio/sky background or one horizon change; parameters are priors.
    if U()<{'air':.20,'water':.90,'road':.72,'field':.45,'pet':.18,'pond':.19}[kind]:
        y={'air':U(.66,.89),'water':U(.39,.65),'road':U(.57,.75),'field':U(.28,.66),'pet':U(.5,.82),'pond':U(.4,.7)}[kind]
        change=U(-.22,.06) if kind!='road' else U(-.12,.17)
        p.im[gy>y+U(-.04,.04)*(gx-.5)]=np.clip(p.base+change,.02,.98)

def airplane(p,view):
    E,G,S=p.ell,p.poly,p.seg; f=p.fg; wing=f+U(-.14,.14); win=U(.11,.43)
    if view==0: # airliner from apron/side, rudder above rear, slightly swept wing
        G([(-.46,.005),(-.50,-.135),(-.465,-.14),(-.36,-.035),(.29,-.036),(.36,-.026),(.49,.005),(.48,.023),(.32,.042),(-.32,.042)],f)
        G([(-.09,.009),(.16,-.001),(-.13,.119),(-.27,.131)],wing)
        G([(-.49,.042),(-.37,.005),(-.30,.01),(-.46,.067)],wing)
        S(-.17,-.010,.305,-.010,.008,.005,win);E(.37,-.008,.048,.012,win,9)
        E(-.02,.058,.067,.020,f-.13)
    elif view==1: # view from below or planform
        E(0,0,.47,.041,f); G([(.16,-.012),(-.05,-.023),(-.255,-.44),(-.20,-.47),(.12,-.19)],wing)
        G([(.16,.012),(-.05,.023),(-.255,.44),(-.20,.47),(.12,.19)],wing)
        for s in [-1,1]:
            G([(-.29,0),(-.42,0),(-.48,s*.17),(-.425,s*.17)],wing); E(.02,s*.17,.057,.019,f-.08)
        E(.34,0,.037,.029,win)
    else: # nearly head on
        G([(-.50,-.058),(-.495,-.087),(0,-.008),(.495,-.087),(.50,-.058),(.055,.050),(-.055,.050)],wing)
        G([(-.015,0),(-.008,-.18),(.008,-.19),(.018,0)],f)
        E(0,.013,.081,.088,f);E(0,-.01,.047,.019,win)
        for s in [-1,1]:E(s*.18,.043,.045,.045,f-.09); E(s*.18,.046,.029,.031,win)

def wheel(p,x,y,rx,ry):
    p.ell(x,y,rx,ry,U(.036,.17)); p.ell(x,y,rx*.48,ry*.48,U(.32,.70))

def car(p,view):
    G,E,S,B=p.poly,p.ell,p.seg,p.box; f=p.fg; glass=U(.12,.5); shine=f+U(.06,.18)
    if view==0 or view==1:
        # Side elevation, subtly shifted rear/front planes for three-quarter.
        roof=U(-.22,-.13); b=U(-.035,.015); cab=U(-.07,.08); back=U(-.39,-.25); nose=U(.19,.29)
        G([(-.49,.025),(-.44,b), (back,b-.025),(back+.08,roof+.035),(back+.14,roof),(cab+.075,roof+.005),(nose,b-.022),(.43,b+.001),(.49,.056),(.483,.145),(.42,.168),(-.43,.168),(-.49,.115)],f)
        G([(back+.024,b-.024),(back+.133,roof+.022),(cab+.067,roof+.024),(nose-.034,b-.024)],glass)
        S(cab-.044,roof+.018,cab-.071,b-.027,.009,.009,f)
        S(-.45,.115,.458,.116,.008,.014,f-.11)
        if view==1:
            G([(.245,.003),(.40,-.014),(.49,.046),(.484,.145),(.29,.160),(.26,.101)],f-.10)
            S(.30,.091,.476,.077,.012,.011,glass)
        E(.425,.042,.046,.014,shine+.18);E(-.456,.042,.018,.015,.32)
        for x in [U(-.34,-.277),U(.255,.336)]:wheel(p,x,.139,U(.059,.083),U(.068,.091))
    else:
        B(-.41,.09,-.25,.237,.11); B(.25,.09,.41,.237,.11)
        G([(-.45,.164),(-.47,-.003),(-.395,-.092),(-.30,-.254),(-.20,-.275),(.24,-.263),(.32,-.229),(.399,-.069),(.47,-.005),(.45,.174),(.33,.205),(-.33,.205)],f)
        G([(-.34,-.073),(-.27,-.225),(.252,-.219),(.34,-.062)],glass)
        G([(-.37,-.033),(.36,-.03),(.28,.047),(-.27,.05)],shine)
        G([(-.23,.102),(.238,.102),(.19,.152),(-.18,.155)],glass-.045)
        for s in [-1,1]: E(s*.329,.055,.084,.037,.85);E(s*.424,-.085,.057,.027,f)
        B(-.079,.115,.079,.158,.8)

def truck(p,view):
    G,E,S,B=p.poly,p.ell,p.seg,p.box; f=p.fg; glass=U(.12,.40); cargo=U(.25,.93)
    if view<2:
        tilt=U(-.04,.035); top=U(-.31,-.18); base=U(.064,.115); cut=U(.075,.23)
        B(-.468,.077,.444,.172,.16)
        if view==0:
            G([(-.49,top),(cut,top+tilt),(cut,base+tilt*.2),(-.49,base)],cargo)
            S(-.482,base-.015,cut,base+tilt*.2-.015,.009,.009,cargo-.16)
        else: # tanker/dump body
            G([(-.50,top+.043),(cut,top),(cut-.04,base),(-.41,base)],cargo)
        G([(cut+.015,.136),(cut+.015,top+.066),(.35,top+.059),(.412,-.077),(.476,-.046),(.492,.135)],f)
        G([(cut+.041,top+.085),(.34,top+.081),(.385,-.057),(cut+.04,-.048)],glass)
        B(.441,.033,.49,.057,.86)
        for x in [-.37,-.244,.352]:wheel(p,x,.153,.054,.071)
    else:
        B(-.34,.20,-.22,.36,.10);B(.22,.20,.34,.36,.10)
        B(-.352,-.365,.343,.20,cargo)
        G([(-.361,.222),(-.346,-.174),(-.277,-.252),(.276,-.251),(.341,-.197),(.372,.236),(.318,.291),(-.323,.291)],f)
        G([(-.287,-.16),(.287,-.16),(.315,-.022),(-.305,-.024)],glass)
        B(-.213,.063,.214,.177,.23);B(-.33,.216,.33,.26,.48)
        for s in [-1,1]:E(s*.277,.149,.048,.035,.82);B(s*.40-.023,-.14,s*.40+.023,-.048,.19)

def ship(p,view):
    G,E,S,B=p.poly,p.ell,p.seg,p.box; f=p.fg
    keel=U(.097,.185); deck=U(-.015,.052); bow=U(.34,.43)
    G([(-.496,deck-.025),(-.38,deck+.004),(.31,deck),(.499,deck-.053),(bow,keel-.012),(.29,keel),(-.39,keel),(-.45,keel-.033)],f)
    if view==0: # cargo, liner, ferry, fishing boat in side view
        a=U(-.39,.03); b=a+U(.16,.42); bridge=U(-.18,-.066); cabin=U(.43,.95)
        B(a,bridge,b,deck,cabin);B(a+.05,bridge-.048,b-.024,bridge,cabin+.03)
        S(a+.014,bridge+.026,b-.02,bridge+.026,.011,.011,cabin-.30)
        x=U(a+.05,b-.033);B(x-.026,bridge-.115,x+.023,bridge-.038,.26)
        if U()<.6:S(.29,deck,.29,-.148,.008,.006,.26)
    elif view==1: # sails and mast
        x=U(-.18,.09); ht=U(-.59,-.34); S(x,deck,x,ht,.012,.008,.28)
        G([(x-.012,ht+.025),(x-.025,deck-.038),(-.39,deck-.025)],U(.68,.97))
        G([(x+.016,ht+.06),(.37,deck-.027),(x+.038,deck-.042)],U(.58,.96))
    else: # oblique bow and pilothouse
        G([(-.42,deck),(-.09,deck-.07),(.49,deck-.05),(.32,keel),(-.11,keel+.05)],f-.14)
        G([(-.29,deck-.02),(-.25,-.16),(.12,-.165),(.26,-.09),(.24,deck-.017)],U(.55,.94))
        B(-.20,-.142,.13,-.09,.25);S(-.07,-.153,-.074,-.28,.008,.004,.32)
    S(-.37,keel+.010,.34,keel+.008,.01,.008,p.base+.13)

# Bird profiles and foreshortened/frontal bodies.
def bird(p,view):
    E,G,S=p.ell,p.poly,p.seg; f=p.fg; wing=f+U(-.27,.23)
    if view<2:
        lean=U(-34,-8) if view==0 else U(-10,12); hx=U(.098,.25);hy=U(-.279,-.099)
        G([(-.15,.035),(-.47,U(.05,.23)),(-.41,.24),(-.06,.12)],wing)
        for x in [-.064,.08]:
            S(x,.134,x+.026,.257,.012,.009,.34);S(x+.026,.254,x+.082,.262,.009,.003,.30)
        E(-.035,.017,U(.20,.291),U(.127,.188),f,lean)
        E(hx-.046,hy+.065,.117,.128,f);E(hx,hy,.123,.119,f+U(-.11,.11))
        G([(hx+.088,hy-.016),(hx+U(.177,.266),hy+.005),(hx+.083,hy+.045)],U(.23,.55))
        E(-.10,-.014,.196,.080,wing,lean);E(hx+.039,hy-.023,.020,.018,.058)
        if U()<.50:S(-.30,.276,.29,.243,.025,.019,.28)
    elif view==2:
        E(0,.065,.167,.287,f);E(.018,-.174,.139,.135,f-.04)
        for s in [-1,1]:E(.14*s,.086,.059,.201,wing,-s*12);E(.018+s*.077,-.201,.015,.018,.09)
        G([(-.029,-.158),(.083,-.159),(.029,-.102)],.45)
    else:
        G([(-.495,-.278),(-.31,-.175),(-.10,-.069),(0,.006),(.14,-.096),(.49,-.24),(.28,-.002),(.09,.101),(-.08,.102),(-.286,-.044)],wing)
        E(.004,.047,.076,.152,f);E(.023,-.046,.062,.058,f);G([(-.04,.10),(.043,.12),(.081,.24),(-.092,.25)],f)

def quadruped(p,k,view):
    E,G,S=p.ell,p.poly,p.seg; f=p.fg
    equ=k==7; deer=k==4; pet=k in [3,5]; dog=k==5
    if view==2 and not pet:
        # Forward-facing grazing animal: narrow chest, long legs, head overhead.
        for s in [-1,1]:
            E(s*.11,-.016,.094,.182,f-.078);S(s*.118,.05,s*.116,.391,.045,.017,f-.06)
            E(s*.116,.399,.026,.017,.13)
        E(0,-.075,.144,.216,f); E(0,-.221,.096,.181,f+.06)
        hx=.012;hy=-.315
        E(hx,hy,.105,.118,f+.04);E(hx,hy+.086,.071,.117,f+.095);E(hx,hy+.152,.060,.041,f-.12)
        for s in [-1,1]:
            E(hx+s*.107,hy-.084,.039,.102,f,s*(44 if deer else 21));E(hx+s*.067,hy-.02,.014,.016,.08)
        return
    # Full body lateral pose; leg hinges, taper, separation and stance variation.
    length=U(.201,.274); belly=U(.085,.144); by=U(-.05,.00); rump=-length*.81; shoulder=length*.70
    hock=U(.206,.284) if not pet else U(.155,.234); foot=hock+U(.085,.16) if not pet else hock+U(.053,.09)
    for i,(x,sg) in enumerate([(rump+.046,-1),(shoulder-.032,1),(rump,-1),(shoulder,1)]):
        near=i>=2; shade=f if near else f-.15; q=U(-.14,.14) if view==1 else U(-.058,.058)
        mid=x+.5*q+(-.012 if sg==1 else .046); ankle=x+q+(-.045 if sg<0 else .0); yt=by+.014
        S(x,yt,mid,.139,.069 if sg<0 else .045,.033,shade)
        S(mid,.139,ankle,hock,.035,.020,shade)
        S(ankle,hock,ankle-.015,foot,.020,.012,shade)
        E(ankle-.006,foot,.027,.015,(shade-.13) if not pet else shade)
    E(-.019,by,length,belly,f,U(-8,5)); E(rump,by+.001,.103,belly*1.07,f)
    if not pet:
        if equ:
            S(rump-.058,by-.078,rump-.145,by+.113,.046,.040,f-.19)
            S(rump-.145,by+.113,rump-.149,by+.228,.04,.018,f-.19)
        else:E(rump-.082,by-.050,.052,.034,f+.18,-36)
        if view==1 and U()<.55: # grazing
            hx=U(.30,.39);hy=U(.09,.17); S(shoulder,-.037,hx,hy,.101,.063,f)
        else:
            hx=U(.206,.315);hy=U(-.294,-.178);S(shoulder,by+.01,hx,hy,.111,.066,f)
            if equ:S(shoulder-.085,by-.071,hx-.049,hy-.049,.029,.023,f-.19)
        E(hx,hy,.082,.068,f)
        S(hx+.019,hy-.008,hx+.117,hy+.057,.066,.043,f+.034)
        E(hx+.131,hy+.064,.038,.03,f-.11);E(hx+.033,hy-.017,.013,.013,.08)
        E(hx-.010,hy-.071,.027,.061,f,-23 if deer else -8)
        if deer and U()<.3:
            S(hx-.012,hy-.036,hx-.103,hy-.203,.012,.007,.32)
            S(hx-.061,hy-.132,hx-.031,hy-.190,.009,.004,.32)
    else:
        S(shoulder,by,shoulder+.098,by-.097,.075,.074,f);hx=shoulder+.110;hy=by-.119
        E(hx,hy,.10,.095,f);E(hx+.089,hy+.031,.082 if dog else .043,.046,f+.081)
        if dog:E(hx-.045,hy+.009,.046,.085,f-.16,14)
        else:G([(hx-.091,hy-.016),(hx-.079,hy-.155),(hx-.002,hy-.063)],f)
        E(hx+.044,hy-.021,.015,.016,.07);E(hx+(.150 if dog else .121),hy+.016,.018,.015,.09)
        S(rump-.051,by-.019,rump-.148,by-.11,.039,.030,f);S(rump-.148,by-.11,rump-.193,by-.231,.031,.013,f)

def petface(p,k,view):
    E,G,S=p.ell,p.poly,p.seg; f=p.fg; cat=k==3; ear=f+U(-.22,.07)
    # width units relative to a unit portrait; forequarters behind head.
    if view==1:
        E(.015,.265,.253,.291,f-.051);E(-.151,.401,.067,.18,f+.058);E(.138,.4,.066,.19,f+.058)
        E(.003,-.153,.206,.18,f);cx=0;cy=-.145; z=.58
    else:
        E(.008,.429,.409,.30,f-.047);cx=U(-.018,.018);cy=U(-.041,.013);z=1
    def e(x,y,rx,ry,v,ang=0):E(cx+z*x,cy+z*y,z*rx,z*ry,v,ang)
    def g(pts,v):G([(cx+z*x,cy+z*y) for x,y in pts],v)
    def s(x,y,X,Y,r,R,v):S(cx+z*x,cy+z*y,cx+z*X,cy+z*Y,z*r,z*R,v)
    pointy=cat or U()<.24; tall=U(.315,.472); broad=U(.276,.362)
    for j in [-1,1]:
        if pointy:
            g([(j*broad,-.048),(j*broad*1.02,-tall),(j*.083,-.197)],ear)
            g([(j*broad*.87,-.17),(j*broad*.97,-tall+.048),(j*.15,-.21)],ear+.14)
        else:
            e(j*.295,-.00,.111,U(.216,.32),ear,-j*12); e(j*.312,.128,.084,.139,ear-.027)
    e(0,-.005,broad,U(.251,.315) if cat else U(.282,.358),f)
    eyes=U(.125,.180);ey=U(-.099,-.053); snout=U(.074,.121) if cat else U(.135,.208)
    if not cat and U()<.48:e(C([-1,1])*eyes,ey,.103,.115,f-.18)
    for j in [-1,1]:
        e(j*eyes,ey,U(.035,.059),U(.023,.041),U(.042,.21),j*8)
        if cat:e(j*eyes,ey+.003,.032,.019,.48);e(j*eyes,ey,.010,.022,.055)
        e(j*eyes-.008,ey-.008,.010,.009,.82)
        if cat and U()<.65:s(j*.075,-.24,j*.067,-.125,.017,.011,f-.16)
    for j in [-1,1]:e(j*.050,snout+.031,.083 if cat else .104,.061 if cat else .086,f+U(.06,.19),j*6)
    if cat:g([(-.044,snout-.009),(.045,snout-.009),(0,snout+.031)],U(.20,.49))
    else:e(0,snout-.015,.067,.048,U(.041,.17))
    s(0,snout+.010,0,snout+.074,.010,.008,.22)
    for j in [-1,1]:s(0,snout+.077,j*.078,snout+.069,.008,.006,.23)

def frog(p,view):
    E,G,S=p.ell,p.poly,p.seg; f=p.fg
    if view==0:
        hx=U(.116,.231);hy=U(-.135,-.054)
        E(-.076,.018,.23,.153,f,-15);E(-.187,.093,.153,.1,f-.09,-15)
        S(-.275,.118,-.034,.178,.052,.019,f+.087); S(.099,.020,.119,.152,.056,.034,f-.064)
        S(.119,.152,.298,.179,.031,.012,f+.045)
        E(hx,hy,.185,.112,f,-8);E(hx+.077,hy+.041,.151,.054,f+.113,-5)
        E(hx+.018,hy-.071,.052,.052,f+.139); E(hx+.026,hy-.071,.031,.029,.17);E(hx+.028,hy-.07,.023,.010,.055)
        S(hx-.025,hy+.034,hx+.191,hy+.015,.009,.006,f-.159)
    else:
        E(0,.06,.266,.202,f);E(-.24,.116,.118,.104,f-.096,25);E(.24,.116,.118,.104,f-.096,-25)
        for j in [-1,1]:
            S(j*.158,.012,j*.272,.185,.054,.028,f+.02);S(j*.272,.185,j*.376,.199,.029,.011,f+.06)
        E(0,-.089,.237,.122,f+.069);E(0,-.044,.209,.067,f+.155)
        for j in [-1,1]:
            e=U(.122,.167)*j;E(e,-.179,.062,.064,f+.11);E(e,-.185,.039,.038,.45);E(e,-.19,.026,.02,.094)
        S(-.161,-.067,0,-.039,.009,.008,f-.126);S(0,-.039,.161,-.067,.008,.009,f-.126)
    if U()<.52:
        for i in range(5):E(U(-.18,.02),U(-.055,.068),U(.013,.032),U(.012,.029),f-.118)

def draw(k,j):
    if k==0:
        view=C([0,0,1,1,2]);size=U(.44,1.02);height=size*(U(.43,1.11) if view==1 else U(.7,1.23));ang=U(-77,77) if view==1 else U(-20,19);env='air'
    elif k in [1,9,8]:
        view=C([0,0,1,1,2,2] if k==1 else ([0,0,0,1,2,2] if k==9 else [0,0,0,1,1,2]));size=U(.68,1.14) if k!=8 else U(.39,1.04);height=size*U(.70,1.24);ang=U(-7,7);env={1:'road',9:'road',8:'water'}[k]
    elif k==2:
        view=C([0,0,0,1,2,3]);size=U(.54,1.26);height=size*U(.77,1.25);ang=U(-27,27);env='field'
    elif k in [4,7]:
        view=C([0,0,0,1,1,2]);size=U(.56,1.13);height=size*U(.76,1.23);ang=U(-9,9);env='field'
    elif k in [3,5]:
        view=C([0,0,0,1,1,2,2]); size=U(.84,1.31) if view!=1 else U(.78,1.16);height=size*U(.81,1.19);ang=U(-21,21);env='pet'
    else:view=C([0,0,0,1,1]);size=U(.76,1.49);height=size*U(.83,1.4);ang=U(-32,32);env='pond'
    cx=U(.39,.61);cy=U(.405,.575)
    p=Pic(size,height,ang,cx,cy,C([-1,1]));atmosphere(p,env)
    if k==0:airplane(p,view)
    elif k==1:car(p,view)
    elif k==9:truck(p,view)
    elif k==8:ship(p,view)
    elif k==2:bird(p,view)
    elif k in [4,7]:quadruped(p,k,view)
    elif k in [3,5]:
        if view==2:quadruped(p,k,0)
        else:petface(p,k,view)
    else:frog(p,view)
    return describe(p.im)

# Network implementation of the multiblock descriptor. Standardize each block
# at test time by flattening its channels into a spatial axis for norm.
def blocknorm(name,x,c):
    x=node(name+'center','norm',shape(name+'flat',x,1,c*8,8))
    return shape(name+'tensor',x,c,8,8)

def mkshapes():
    v=conv('luma','image',np.array([.26,.53,.21])[None,:,None,None]);v=node('light','norm',v)
    lum=pool('lightcells',v,4)
    d=conv('derivatives',v,np.concatenate([dirs,-dirs])[:,None],p=2)
    ds=pool('cellresponses',rel('halfwaves',d),4)
    Id=np.eye(5); absolute=np.concatenate([Id,Id],1); signed=np.concatenate([Id,-Id],1)
    mix=[signed,absolute[:4]-absolute[:4].mean(0),absolute[:4].mean(0,keepdims=True),absolute[4:5]]
    desc=[lum]+[conv(n,ds,a[:,:,None,None]) for n,a in zip(['polarity','steer','density','spots'],mix)]
    counts=[1,5,4,1,1]; channels=sum(counts); merged=[];start=0
    for j,(x,c) in enumerate(zip(desc,counts)):
        # Center each channel, without equalizing its gain, using an 8x8
        # linear mean replicated to the grid by a dense layer.
        avg=shape('avgvec'+str(j),pool('dc'+str(j),x,8),c)
        neg=-np.eye(c).repeat(64,axis=0)
        bias=shape('dcgrid'+str(j),dense('repeatdc'+str(j),avg,neg),c,8,8)
        cent=node('ac'+str(j),'add',[x,bias]);unit=blocknorm('unit'+str(j),cent,c)
        pad=np.zeros((channels,c));pad[start:start+c]=np.eye(c)/(8*np.sqrt(c));start+=c
        merged.append(conv('place'+str(j),unit,pad[:,:,None,None]))
    descriptor=node('descriptor','add',merged)
    K=768 # independent drawings per category; contiguous groups permit pooling
    wt=np.empty((10,K,channels,8,8),'f')
    for k in range(10):
        for j in range(K):wt[k,j]=draw(k,j)
    sims=dense('drawings',descriptor,wt.reshape(10*K,-1))
    sims=shape('catalog',sims,10,K//16,16)
    # Average maxima in subcatalogs -- tolerates printing noise, doesn't depend
    # entirely on one lucky match. Plus the best match across all subcatalogs.
    chunk=pool('best_of_48',sims,(K//16,1),'max');top=pool('best',chunk,(1,16),'max');typ=pool('recurring',chunk,(1,16))
    a=dense('strong_shape',top,np.eye(10)*3.8);b=dense('consistent_shape',typ,np.eye(10)*2.6)
    return [a,b]

# Assemble soft color predicates with ReLU halfspaces. These are hand-coded
# color names, not color centroids or thresholds estimated from a population.
def colors():
    rgb=[np.eye(3)[i] for i in range(3)];r,g,b=rgb; y=.26*r+.53*g+.21*b
    specs={
      'cool':[(b-r,.014,.12),(b-g,-.02,.085)],
      'blue':[(b-r,.08,.17),(b-g,.01,.09)],
      'foliage':[(g-b,.045,.13),(g-r,-.035,.08)],
      'leaf':[(g-b,.055,.14),(g-r,.015,.085)],
      'earth':[(r-b,.075,.16),(r-g,-.005,.095),(2*g-r-b,-.07,.09)],
      'tan':[(r-g,.025,.12),(g-b,.025,.115)],
      'red':[(r-g,.095,.20),(r-b,.09,.21)],
      'vivid':[(r-g,.14,.16),(r-b,.15,.16)],
      'white':[(y,.70,.2),(r,.66,.2),(b,.64,.2)],
      'dark':[(-y,-.29,.20)], 'bright':[(y,.65,.23)],
      'sky':[(b-r,.029,.14),(b-g,-.012,.08),(y,.30,.23)],
      'gray':[(y,.17,.13),(-y,-.88,.14)]+[(v,-.091,.065) for v in [r-b,b-r,g-r,r-g,b-g,g-b]],
      'gold':[(r-g,.054,.15),(g-b,.073,.14)],
    }
    fs=[]; bs=[]; terms=[]; names=list(specs)
    # AND by subtracting average unmet margin. Clip above one by using a
    # positive-part deficit at each halfspace. A failing constraint alone
    # can exclude a color; don't divide by the arity of the whole conjunction.
    for cond in specs.values():
        ids=[]
        for v,thr,margin in cond:
            ids.append(len(fs));fs.append(-v/margin);bs.append(1+thr/margin)
        terms.append(ids)
    sm=pool('antialias_rgb','image',2)
    h=rel('unmet',conv('colorcuts',sm,np.array(fs)[:,:,None,None],bs)); w=np.zeros((len(names),len(fs)))
    for i,ids in enumerate(terms):w[i,ids]=-1 if len(ids)==1 else (-.53 if names[i]!='gray' else -.30)
    out=rel('color',conv('soft_ands',h,w[:,:,None,None],1));return names,pool('colorcells',out,2)

# All context rules are written in terms of averages in analytic regions.
# This class is merely an expression compiler for those dense weights.
Y,X=np.mgrid[:8,:8];Y=(Y+.5)/8;X=(X+.5)/8
regions={'a':np.ones((8,8)), 't':Y<.28, 'b':Y>.72,
         'c':np.exp(-.5*((X-.5)/.265)**2-.5*((Y-.50)/.22)**2),
         'm':(Y>.36)&(Y<.70), 'outer':(np.abs(X-.5)>.28)|(Y<.2)|(Y>.82),
         'lowmid':(Y>.52)&(Y<.80),'side':(np.abs(X-.5)>.30)&(Y>.26)&(Y<.78)}
regions={n:m/m.sum() for n,m in regions.items()}

class Rules:
    def __init__(self,tag,names,node):self.tag=tag; self.names=names; self.node=node;self.rules=[];self.bs=[];self.co=[]
    def measure(self,x):
        w=np.zeros((len(self.names),8,8))
        for key,a in x.items():
            nm,rg=key.split(':');w[self.names.index(nm)]+=a*regions[rg]
        return w
    def put(self,x,b,classes):
        self.rules.append(self.measure(x));self.bs.append(b);self.co.append(classes)
    def done(self):
        expr=rel(self.tag+'_on',dense(self.tag+'_tests',self.node,self.rules,self.bs)); w=np.zeros((10,len(self.rules)))
        for j,t in enumerate(self.co):
            for k,v in t.items():w[k,j]=v
        return dense(self.tag+'_votes',expr,w)

def chromatic():
    names,v=colors(); rules=Rules('color_layout',names,v); put=rules.put
    # Cool surrounding air vs. water under a hull vs. neutral asphalt.
    put({'sky:t':1},0,{0:.85,8:.43,9:.21,2:.14})
    put({'cool:a':1},0,{0:.45,8:.42,9:.13,4:-.25,6:-.30,7:-.15})
    put({'cool:b':1},0,{8:1.26,0:.34,4:-.28,6:-.28})
    put({'blue:b':1},0,{8:.55,0:.16})
    put({'cool:b':1,'cool:c':-.8},-.045,{8:.66})
    put({'sky:t':1,'gray:b':.6,'cool:b':-.9},-.60,{9:.36,0:.10})
    put({'white:t':1},0,{0:.40,8:.26,9:.26})
    put({'white:b':1},0,{0:.39,3:.13,5:.13,4:-.18})
    put({'gray:b':1,'gray:t':.3},0,{1:.53,9:.38,8:.16,2:-.10})
    put({'white:c':1,'white:outer':-.65},-.025,{9:.67,0:.20,5:.20,3:.12,8:.25})
    put({'vivid:c':1},0,{1:.73,9:.84,2:.53,6:-.28,4:-.24})
    put({'dark:lowmid':1,'dark:b':-.8},-.07,{1:.64,9:.36,8:.17})
    put({'dark:t':1},0,{0:-.53,8:-.33,1:.22,6:.18,3:.12,2:.10})
    # Habitat and body color: grass below a tall grazer, leaves surrounding
    # a perched bird, low warm/olive texture for an amphibian.
    put({'foliage:a':1},0,{2:.54,4:.80,6:.69,7:.40,1:-.42,9:-.39,8:-.39,0:-.48,3:-.16})
    put({'leaf:a':1},0,{2:.31,4:.46,6:.32,7:.21})
    put({'foliage:b':1},0,{4:.57,7:.83,2:.15,1:-.23,8:-.27})
    put({'leaf:b':1,'leaf:t':-.65},-.05,{7:.64,4:.23})
    put({'foliage:t':1},0,{2:.43,4:.38,6:.30,3:-.15})
    put({'foliage:outer':1,'foliage:c':-.85},-.045,{2:.67,4:.65,7:.37})
    put({'foliage:c':1,'foliage:outer':-.52},-.02,{6:.88,2:.25})
    put({'earth:a':1},0,{6:.44,4:.22,3:.19,5:.19,2:.08,1:-.17,0:-.18})
    put({'earth:c':1,'earth:outer':-.76},-.045,{7:.62,4:.42,5:.37,3:.22})
    put({'tan:c':1,'foliage:b':.68},-.59,{7:.61,4:.38})
    put({'gold:c':1},0,{5:.50,3:.51,6:.37,7:.13})
    put({'dark:a':1,'earth:a':.35},-.13,{6:.63,3:.33,5:.12})
    put({'sky:t':1,'foliage:b':.85},-.53,{7:.55,4:.24})
    put({'white:c':1,'dark:c':.9},-.42,{5:.48,3:.43,1:.18})
    return rules.done()

# Texture maps: horizontal and vertical structure, abruptness, and a
# coherent (large-neighborhood) gradient vs. fine-scale repeated detail.
def textures():
    f=[];bs=[];names=[]
    def feature(n,w,b=0):names.append(n);f.append(w);bs.append(b)
    a=np.concatenate([np.eye(5),np.eye(5)],1); z=a[:4].mean(0)
    feature('busy',z);feature('grain',a[4]); feature('cross',a[0]+a[2]-a[1]-a[3]);feature('slant',a[1]+a[3]-a[0]-a[2]);feature('h',a[2]-.5*(a[1]+a[3]));feature('v',a[0]-.5*(a[1]+a[3]));feature('quiet',-2.7*z,.86);feature('fine',a[4]-.43*z)
    local=rel('structures',conv('texture_filters','cellresponses',np.array(f)[:,:,None,None],bs))
    rules=Rules('texture_layout',names,local);put=rules.put
    put({'quiet:outer':1},-.08,{0:.91,2:.44,8:.31,6:-.24})
    put({'quiet:b':1},-.10,{0:.51,1:.59,9:.35,8:.21,4:-.28,6:-.21})
    put({'quiet:t':1},-.18,{0:.56,8:.53,9:.16,7:.09})
    put({'busy:t':1},-.20,{0:-.50,8:-.48,4:.34,6:.36,1:.21})
    put({'busy:b':1,'busy:m':-.50},-.09,{4:.73,6:.51,7:.38,1:-.43,0:-.30})
    put({'busy:m':1,'busy:b':-.75,'busy:t':-.35},-.025,{1:.68,9:.53,8:.25,2:.23})
    put({'grain:a':1},-.13,{6:.91,4:.42,2:.14,1:-.29,0:-.24})
    put({'fine:a':1},-.07,{6:1.32,4:.28,3:.26,7:-.13})
    put({'cross:c':1},-.025,{9:1.9,1:.95,8:.92,5:-.34,2:-.34,6:-.50})
    put({'h:m':1},-.030,{8:1.31,1:.79,9:.52,0:.37})
    put({'h:b':1},-.027,{8:1.45,1:.85,9:.59,4:-.33})
    put({'v:c':1},-.04,{9:.97,7:.30,4:.30})
    put({'slant:c':1},-.01,{2:.87,0:.46,3:.57,5:.49,6:.33})
    put({'quiet:c':1,'busy:outer':.40},-.23,{3:.49,5:.56,6:.26})
    return rules.done()

# Independent hand-coded part filters. Measure two circular tire/rim contrasts
# at compatible heights, and use both scale and layout to suppress single
# incidental dark flecks. Similar coincidence detector for eyes and nose.
def parts():
    y,x=np.mgrid[-4:5,-4:5]; filters=[];specs=[]
    def avg(m):return m/max(m.sum(),1e-9)
    def ring(rx,ry):
        rr=np.sqrt((x/rx)**2+(y/ry)**2);rim=(rr>=.48)&(rr<1.15); hub=rr<.48; outer=(rr>=1.15)&(rr<1.72)
        return .70*avg(outer)+.30*avg(hub)-avg(rim)
    for r in [1.65,2.45,3.15]:
        filters.append(ring(r,r));specs.append('tire')
    for rx,ry in [(1.25,1.1),(1.90,1.45)]:
        rr=(x/rx)**2+(y/ry)**2
        filters.append(avg((rr>=1.55)&(rr<5))-avg(rr<1.55));specs.append('eye')
    mp=conv('dark_parts','light',np.array(filters)[:,None],s=2,p=4);mp=rel('part_contrast',mp)
    # Build component arrangements on the 16x16 response grid.
    groups=[[],[],[]]; bs=[[],[],[]]
    def at(w,i,x,y,a,rad=0):
        if rad==0:w[i,y,x]+=a
        else:
            for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                yy=y+dy;xx=x+dx
                if 0<=yy<16 and 0<=xx<16:w[i,yy,xx]+=a*(.56 if dy==dx==0 else .11)
    for i in range(3):
        for sep in ([5,7,9] if i==0 else [6,8,10] if i==1 else [7,9,11]):
            for lx in range(max(1,6-sep//2),min(14-sep,10-sep//2)+1):
                for y0 in range(8,13):
                    for dy in [0,1,-1]:
                        w=np.zeros((5,16,16),'f');at(w,i,lx,y0,.50,1);at(w,i,lx+sep,y0+dy,.50,1)
                        # The opposite locations (axle, underneath) ought not to be wheels as well.
                        at(w,i,lx+sep//2,y0,-.16,1);at(w,i,lx,y0-3,-.10,1);at(w,i,lx+sep,y0+dy-3,-.10,1)
                        groups[0].append(w);bs[0].append(-.40)
    for k in [1,2]: # cat: shorter nose triangle, dog: longer/deeper snout
        for sep in ([3,4,5,6] if k==1 else [3,5,6]):
            for lx in range(4-sep//2,12-sep//2):
                for y0 in range(3,10):
                    for dy in [-1,0,1]:
                        for nose in ([1,2] if k==1 else [3,4]):
                            idx=3+int(sep>=5);w=np.zeros((5,16,16),'f')
                            at(w,idx,lx,y0,.34,1);at(w,idx,lx+sep,y0+dy,.34,1);ny=y0+dy//2+nose;nx=lx+sep//2
                            at(w,idx,nx,ny,.38,1);at(w,idx,nx,y0,-.19,1);at(w,idx,lx,ny,-.10,1);at(w,idx,lx+sep,ny+dy,-.10,1)
                            groups[k].append(w);bs[k].append(-.48)
    outs=[]
    for j in range(3):
        n=len(groups[j]);vec=rel('match_parts'+str(j),dense('constellations'+str(j),mp,groups[j],bs[j]));vec=shape('partbank'+str(j),vec,1,1,n)
        # A bounded reward for a part constellation.
        v=pool('good_parts'+str(j),vec,(1,n),'max');cl=np.zeros((10,1));
        if j==0:cl[1]=.69;cl[9]=.32
        if j==1:cl[3]=.50;cl[2]=.08
        if j==2:cl[5]=.57
        outs.append(dense('partvote'+str(j),v,cl))
    return outs

def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True)
    s=mkshapes();s.extend([chromatic(),textures()]);s.extend(parts());node('class_scores','add',s)
    (out/'network.json').write_text(json.dumps({'layers':L},indent=1)+'\n');np.savez(out/'weights.npz',**W)

if __name__=='__main__':main()
