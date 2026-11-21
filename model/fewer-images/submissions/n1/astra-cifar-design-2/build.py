"""Data-free shape and scene classifier. Usage: python3 build.py OUTPUT_DIRECTORY"""
import sys, json, math
from pathlib import Path
import numpy as np

# The canvas is in arbitrary object coordinates; its camera varies between drawings.
# Everything drawn is luminance. Chromatic evidence is handled separately by rules.
R = np.random.default_rng(197613)
def U(a,b): return R.uniform(a,b)
def pick(*a): return a[R.integers(len(a))]
def prob(p): return R.random()<p
N=48
YY,XX=np.mgrid[0:N,0:N].astype(float)
XX=(XX+.5)*2/N-1; YY=(YY+.5)*2/N-1
class Canvas:
    def __init__(self,k,portrait=False):
        scale=U(.57,1.1) if not portrait else U(.68,1.22)
        cx=U(-.17,.17); cy=U(-.16,.16)
        sx=scale*U(.87,1.15); sy=scale*U(.87,1.14)
        angle=U(-.09,.09) if k in [1,8,9] else U(-.19,.19)
        if k==0: angle=U(-.43,.43)
        c,s=np.cos(angle),np.sin(angle)
        self.x=((XX-cx)*c+(YY-cy)*s)/sx * pick(-1,1)
        self.y=(-(XX-cx)*s+(YY-cy)*c)/sy
        self.bg=U(.22,.84); self.a=np.full((N,N),self.bg)+U(-.10,.10)*YY+U(-.06,.06)*XX
        self.mask=np.zeros((N,N),dtype=bool)
        self.base=np.clip(self.bg+pick(-1,1)*U(.20,.52),.06,.92)
        self.highlight=U(-.04,.16)
        if k in (1,9,8) or (k in (4,7,0) and prob(.30)):
            # A muted, approximately level ground or water plane (not the silhouette).
            y=U(-.18,.50); d=U(-.22,.08)
            self.a+=d*np.clip((self.y-y)*U(5,35),0,1)
    def paint(self,m,v): self.a=np.where(m,v,self.a);self.mask |= m
    def ell(self,x,y,rx,ry,v,ang=0,shade=0):
        c,s=np.cos(ang),np.sin(ang); dx=self.x-x; dy=self.y-y
        u=(c*dx+s*dy)/rx; w=(-s*dx+c*dy)/ry; q=u*u+w*w
        if shade: v=v+shade*(.55-.6*w-.52*q)
        self.paint(q<=1,v)
    def poly(self,pts,v):
        p=list(pts); inside=np.zeros_like(self.x,dtype=bool)
        for (x,y),(xx,yy) in zip(p,p[1:]+p[:1]):
            if abs(yy-y)>1e-9:
                inside ^= ((self.y>y)!=(self.y>yy)) & (self.x < x+(self.y-y)*(xx-x)/(yy-y))
        self.paint(inside,v)
    def line(self,x,y,xx,yy,w,v):
        dx=xx-x;dy=yy-y; t=np.clip(((self.x-x)*dx+(self.y-y)*dy)/(dx*dx+dy*dy+1e-12),0,1)
        self.paint((self.x-x-t*dx)**2+(self.y-y-t*dy)**2<=w*w,v)
    def resample(self,a):
        # Exact 3:2 box resampling from a 48-square canvas to a 32-square image.
        a=np.clip(a,0,1); a=a.reshape(16,3,48)
        a=np.stack(((2*a[:,0]+a[:,1])/3,(a[:,1]+2*a[:,2])/3),1).reshape(32,16,3)
        return np.stack(((2*a[:,:,0]+a[:,:,1])/3,(a[:,:,1]+2*a[:,:,2])/3),2).reshape(32,32)

# Near-orthographic camera. z is upward, l is along the vehicle, d across it.
class Camera:
    def __init__(self,angle,elev,vertical_offset=0):
        self.c=np.cos(angle);self.s=np.sin(angle);self.ce=np.cos(elev);self.se=np.sin(elev);self.y0=vertical_offset
    def p(self,l,d,z):return (l*self.c-d*self.s,self.y0+(l*self.s+d*self.c)*self.se-z*self.ce)

def vehicle(im,truck):
    # Front end is +l; nearest visible side is +d. Heights are above the road.
    cam=Camera(U(.02,1.52),U(.04,.41),U(.35,.52) if not truck else U(.44,.56));p=cam.p
    L=U(.90,1.13); w=U(.28,.40); wheel=U(.135,.18) if not truck else U(.15,.21)
    col=im.base; glass=U(.10,.45); top=U(.04,.19); front=U(-.12,.08)
    def face(points,v): im.poly([p(*q) for q in points],v)
    def box(a,b,d,lo,hi,color):
        face([(a,-d,hi),(b,-d,hi),(b,d,hi),(a,d,hi)],color+top)
        face([(a,d,lo),(b,d,lo),(b,d,hi),(a,d,hi)],color)
        face([(b,-d,lo),(b,d,lo),(b,d,hi),(b,-d,hi)],color+front)
    def disc(l,d,z,r,v):
        t=np.arange(20)*(2*np.pi/20); face([(l+r*np.cos(q),d,z+r*np.sin(q)) for q in t],v)
    if prob(.82): im.ell(*p(0,0,.005),L*cam.c+w*cam.s,.075+.14*cam.se,U(.05,.31))
    if truck:
        box(-L,L,w*.94,.20,.35,U(.09,.29)); split=U(.19,.50)*L
        hc=U(.88,1.21); cabh=hc*U(.75,1.01); cargo=pick(U(.65,.91),col,U(.32,.57))
        box(-L,split-.04,w*1.04,.36,hc,cargo)
        if prob(.28):
            for l in np.linspace(-L,split-.04,8):
                im.line(*p(l,w*1.04,.40),*p(l,w*1.04,hc-.035),.009,cargo-.12)
        box(split,L,w,.23,cabh,col)
        face([(L+.002,-w*.91,cabh*.61),(L+.002,w*.91,cabh*.61),(L+.002,w*.83,cabh-.07),(L+.002,-w*.83,cabh-.07)],glass)
        face([(split+.09,w,cabh*.61),(L-.04,w,cabh*.61),(L-.04,w,cabh-.08),(split+.09,w,cabh-.08)],glass+.04)
        face([(L,-.65*w,.34),(L,.65*w,.34),(L,.65*w,.47),(L,-.65*w,.47)],U(.1,.31))
        axles=[-.72*L,.72*L];
        if prob(.63): axles.insert(1,-.72*L+2*wheel)
        h=.37
    else:
        h=U(.37,.51); roof=h+U(.22,.34); back=U(-.94,-.58)*L; aft=back+U(.09,.27); forward=U(.10,.39)*L; hood=U(.54,.75)*L
        box(-L,L,w,.15,h,col)
        face([(back,w,h),(aft,w*.79,roof),(forward,w*.76,roof),(hood,w,h)],col)
        face([(aft,-w*.79,roof),(forward,-w*.76,roof),(forward,w*.76,roof),(aft,w*.79,roof)],col+top)
        face([(forward,-w*.76,roof),(hood,-w,h),(hood,w,h),(forward,w*.76,roof)],glass+U(.0,.17))
        face([(back+.06,w,h+.035),(aft+.035,w*.81,roof-.04),(forward-.025,w*.79,roof-.04),(hood-.08,w,h+.035)],glass)
        mid=(forward+aft)*.5;im.line(*p(mid,w,h+.035),*p(mid,w*.8,roof-.03),.023,col)
        axles=[U(-.74,-.59)*L,U(.57,.74)*L]
        face([(L,-w*.70,.20),(L,w*.7,.20),(L,w*.67,h*.79),(L,-w*.67,h*.79)],col*.57)
    for side in [-1,1]:
        face([(L,side*w*.63,h-.09),(L,side*w*.94,h-.09),(L,side*w*.94,h-.014),(L,side*w*.63,h-.014)],U(.67,.96))
    tire=U(.035,.18)
    for l in axles:
        disc(l,w*1.014,wheel,wheel,tire);disc(l,w*1.018,wheel,wheel*U(.39,.67),U(.28,.70))

def plane(im):
    cam=Camera(U(-.25,1.52),pick(U(-1.27,-.22),U(-.18,.58)));p=cam.p
    b=im.base; winglen=U(.84,1.27); radius=U(.066,.13)
    def face(v,pts): im.poly([p(*q) for q in pts],v)
    for sign in [-1,1]:
        sweep=U(.25,.65)
        face(b+U(-.10,.08),[(.29,0,0),(.20-sweep,sign*winglen,.05),(-.01-sweep,sign*winglen,.05),(-.31,0,0)])
        face(b-.04,[(-.65,0,.025),(-.92,sign*.44,.10),(-1.06,sign*.43,.10),(-.98,0,.025)])
    nose=p(1.07,0,0);tail=p(-.96,0,.026);dx=nose[0]-tail[0];dy=nose[1]-tail[1]
    im.ell((nose[0]+tail[0])/2,(nose[1]+tail[1])/2,math.hypot(dx,dy)*.52+radius*.15,radius,b,math.atan2(dy,dx),.12)
    face(b+U(-.16,.07),[(-.98,0,.02),(-1.03,0,.39),(-.89,0,.40),(-.58,0,.04)])
    if prob(.72):
        for d in [-.43,.43]:im.ell(*p(.12,d,-.08),.11,.047,b-.09,math.atan2(dy,dx))
    if prob(.7):im.line(*p(.57,radius*.65,.04),*p(.77,radius*.69,.041),.027,b-.26)

def boat(im):
    cam=Camera(U(-.14,1.29),U(.02,.33),U(.16,.37));p=cam.p; col=im.base;w=U(.20,.36)
    def face(pts,v):im.poly([p(*q) for q in pts],v)
    rim=[(-.97,-w,.03),(.52,-w,.07),(1.09,0,.15),(.54,w,.06),(-.97,w,.035)]
    face([rim[4],rim[3],rim[2],(.83,0,-.18),(.46,w*.85,-.24),(-.86,w*.84,-.22)],col)
    face(rim,col+.13); typ=U(0,1); deck=pick(U(.65,.94),col+.12)
    if typ<.27: # sail
        ht=U(.90,1.42);im.line(*p(-.09,0,.03),*p(-.09,0,ht),.018,.18)
        face([(-.12,0,ht),(-.18,0,.18),(-.87,0,.23)],U(.67,.97));face([(-.04,0,ht*.93),(.02,0,.23),(.88,0,.18)],U(.54,.94))
    else:
        a=U(-.86,-.38);b=U(.0,.55) if typ<.72 else U(.59,.84);hi=U(.26,.58)
        for i in range(2 if typ>.42 else 1):
            lo=.04+i*hi*.55;h=hi+i*hi*.44;d=w*(.86-.19*i);l=a+.13*i;r=b-.14*i
            if r-l<.07:break
            face([(l,-d,h),(r,-d,h),(r,d,h),(l,d,h)],deck+.08)
            face([(l,d,lo),(r,d,lo),(r,d,h),(l,d,h)],deck)
            face([(r,-d,lo),(r,d,lo),(r,d,h),(r,-d,h)],deck-.10)
            face([(l+.035,d,h-.105),(r-.04,d,h-.105),(r-.04,d,h-.045),(l+.035,d,h-.045)],deck-U(.19,.46))
        if prob(.73):im.line(*p(a+.17,0,hi),*p(a+.17,0,hi+U(.13,.45)),U(.013,.05),col-.13)
    if prob(.66):im.line(*p(-1.07,w*.8,-.225),*p(.52,w*.8,-.238),.022,U(.62,.90))

# Animal illustrations: ellipse-and-polygon parts are posed relative to joints,
# rather than contours or landmarks digitized from any particular photograph.
def portrait(im,k):
    b=im.base; dark=min(b*.45,U(.05,.21));y=U(-.11,.11); rx=U(.48,.76);ry=rx* (U(.86,1.16) if k==3 else U(1.04,1.33)); turn=U(-.24,.24)
    if k in [4,7]:rx*=.72;ry*=1.16
    im.ell(0,y+ry*1.37,rx*1.25,ry*1.04,b-.05,shade=.11)
    for s in [-1,1]:
        if k==3 or (k==5 and prob(.19)):
            im.poly([(s*.89*rx,y-.23*ry),(s*rx*U(.84,1.19),y-ry*U(1.12,1.48)),(s*.18*rx,y-.76*ry)],b-.015)
            im.poly([(s*.8*rx,y-.57*ry),(s*.95*rx,y-1.12*ry),(s*.50*rx,y-.74*ry)],b-.14)
        elif k in [4,7]:
            im.ell(s*.69*rx,y-.91*ry,.23*rx,ry*(.42 if k==4 else .37),b-.09,s*(.90 if k==4 else .37),.12)
        else:
            im.ell(s*.91*rx,y-.14*ry,.37*rx,U(.60,.96)*ry,b-U(.03,.24),-s*.20,.1)
    im.ell(0,y,rx,ry,b,shade=.17)
    if k in [3,5] and prob(.37): im.ell(turn*rx,y-.15*ry,.22*rx,.80*ry,b+U(.09,.27),shade=.05)
    if k==3 and prob(.59):
        for s in [-1,1]:
            for j in range(3): im.line(s*(.23+.20*j)*rx,y-.79*ry,s*(.14+.22*j)*rx,y-(.49-.06*j)*ry,.023,b-.17)
    for s in [-1,1]:
        ex=(turn+s*(.43 if k in [3,5] else .71))*rx;ey=y-ry*U(.13,.26)
        if k==5 and prob(.48):im.ell(ex,ey,.29*rx,.26*ry,b-.14,s*.22)
        im.ell(ex,ey,(.185 if k==3 else .134)*rx,.103*ry,dark,-s*.14)
        im.ell(ex-.023*rx,ey-.025*ry,.042*rx,.033*ry,U(.62,.93))
    my=y+ry*(.43 if k==3 else .49); mx=turn*rx*1.34
    if k in [4,7]:
        im.ell(mx,my,.62*rx,.60*ry,b-.11,shade=.13)
        for s in [-1,1]:im.ell(mx+s*.34*rx,my+.20*ry,.13*rx,.12*ry,dark,s*.22)
    else:
        for s in [-1,1]: im.ell(mx+s*.188*rx,my,.35*rx,(.21 if k==3 else .36)*ry,min(.95,b+U(.06,.23)),shade=.075)
        if k==3:im.poly([(mx-.17*rx,my-.16*ry),(mx+.17*rx,my-.16*ry),(mx,my+.014*ry)],dark+.09)
        else:im.ell(mx,my-.13*ry,.24*rx,.154*ry,dark)
        im.line(mx,my,mx,my+.13*ry,.015,dark)
        for s in [-1,1]:im.line(mx,my+.14*ry,mx+s*.22*rx,my+.19*ry,.012,dark+.04)

def quadruped(im,k):
    b=im.base; hoof=k in [4,7]; cat=k==3; sgn=-1 if prob(.5) else 1
    bodyrx=U(.45,.65);bodyry=U(.19,.32) if hoof else U(.21,.35); by=U(-.13,.07)
    hip=-.67*bodyrx; shoulder=.70*bodyrx; floor=by+bodyry+ (U(.39,.66) if hoof else U(.21,.43));sit=not hoof and prob(.39)
    def leg(x,far,fore):
        top=by+.02;side=(-.10 if far else 0);x+=side; off=U(-.25,.21)+(sgn*(.16 if fore else -.14) if far else 0)
        knee_x=x+off*.48+(.02 if fore else .14); hock_x=x+off*.68+(.016 if fore else -.045);end=x+off
        w=U(.06,.092) if hoof else U(.067,.108);col=b-(.13 if far else .022);end_y=floor-(.08 if far else 0)
        im.poly([(x-1.4*w,top),(x+1.4*w,top),(knee_x+.66*w,top+.48*(end_y-top)),(hock_x+.43*w,top+.72*(end_y-top)),(end+.43*w,end_y),(end-.46*w,end_y),(hock_x-.53*w,top+.72*(end_y-top)),(knee_x-.75*w,top+.48*(end_y-top))],col)
        im.ell(end+.018,end_y,.055 if hoof else .080,.039,col-.11 if hoof else col)
    for x,fore in [(hip,False),(shoulder,True)]:leg(x,True,fore)
    if k==7: im.ell(hip-bodyrx*.42,by+.21,.112,.36,b-.22,.26)
    elif k==4:im.ell(hip-.30,by-.07,.09,.125,b+.08,-.74)
    else:
        t=U(-.6,.67);im.line(hip-.10,by,hip-.39,by+t*.42,.064,b);im.line(hip-.39,by+t*.42,hip-.57,by+t,.048,b)
    for x,fore in [(hip,False),(shoulder,True)]:leg(x,False,fore)
    if sit:
        im.ell(-.12,by+.12,bodyrx*.74,bodyry*1.36,b,shade=.11);im.ell(.09,by-.13,bodyrx*.68,bodyry*1.35,b,.27,.12)
        hx=.22;hy=by-bodyry*1.56
    else:
        im.ell(0,by,bodyrx,bodyry,b,-.045,.15)
        neck=U(.30,.52) if hoof else U(.08,.24);hx=bodyrx*.97+U(.01,.13);hy=by-neck
        im.poly([(shoulder-.20,by-.11),(hx-.13,hy-.065),(hx+.102,hy+.04),(shoulder+.11,by+.14)],b+.015)
        if k==7:im.line(shoulder-.12,by-.17,hx-.10,hy-.076,.051,b-.22)
    hr=U(.115,.17) if hoof else U(.177,.267)
    if sit:hr*=1.2
    im.ell(hx,hy,hr*1.11,hr,b,shade=.11)
    if hoof:im.ell(hx+hr*1.03,hy+hr*.39,hr*1.41,hr*.65,b-.035,.58,.11)
    elif not cat:im.ell(hx+hr*.81,hy+hr*.23,hr*.81,hr*.55,b+.069,.19,.08)
    if cat or hoof or prob(.23):
        for t in [-.70,.11]:im.poly([(hx+t*hr,hy-.56*hr),(hx+(t-.23)*hr,hy-hr*U(1.32,1.9)),(hx+(t+.66)*hr,hy-.67*hr)],b-.035)
    else:im.ell(hx-.40*hr,hy+.08*hr,.52*hr,.93*hr,b-.15,-.32,.08)
    im.ell(hx+.43*hr,hy-.18*hr,.12*hr,.10*hr,.06)
    if not hoof:im.ell(hx+hr*(1.1 if cat else 1.54),hy+.21*hr,.18*hr,.12*hr,.07)
    if k==4 and prob(.35):
        for t in [-.06,.07]:
            im.line(hx+t,hy-.12,hx+t-.11,hy-.32,.022,b-.15);im.line(hx+t-.11,hy-.32,hx+t-.25,hy-.40,.017,b-.15);im.line(hx+t-.11,hy-.32,hx+t-.14,hy-.48,.012,b-.15)

def bird(im):
    b=im.base
    if prob(.25):
        pitch=U(-.29,.48)
        im.ell(.02,.04,.44,.115,b,pitch,.1);im.ell(.39,.05+.33*pitch,.13,.105,b)
        for s in [-1,1]:
            im.poly([(.17,.04),(-.03+s*.39,s*.52),(-.33+s*.73,s*U(.58,.92)),(-.23+s*.35,s*.40),(-.24,.06)],b+U(-.12,.08))
        im.poly([(-.23,0),(-.66,-.1),(-.64,.16),(-.22,.13)],b-.04);return
    upright=U(.20,1.25);size=U(.83,1.20);rx=.51*size;ry=U(.23,.36)*size;hx=rx*.85*np.cos(upright);hy=-rx*.85*np.sin(upright)
    tx=-rx*.76*np.cos(upright);ty=rx*.76*np.sin(upright)
    if prob(.61):
        im.line(-1.18,.63,1.23,.40,U(.028,.086),im.bg-.20)
        for x in [-.13,.08]:im.line(x,.22,x-.04,.52,.022,b-.19)
    im.poly([(tx-.07,ty-.11),(tx+.13,ty+.06),(tx-.46*np.cos(upright)+.06,ty+.58*np.sin(upright)+.06),(tx-.57*np.cos(upright)-.1,ty+.58*np.sin(upright)-.07)],b-.11)
    im.ell(0,0,rx,ry,b,-upright,.17);im.ell(-.07,-.028,rx*.73,ry*.60,b-U(.08,.27),-upright+.01,.08)
    r=U(.16,.26)*size;im.ell(hx,hy-.02,r*1.06,r,b+U(-.16,.13),shade=.10)
    im.poly([(hx+r*.71,hy-r*.27),(hx+r+U(.11,.26),hy+.014),(hx+r*.76,hy+r*.26)],pick(.14,.54))
    im.ell(hx+.44*r,hy-.27*r,r*.15,r*.15,.055)

def frog(im):
    b=im.base;frontal=prob(.48);rear=-.27 if not frontal else 0;hx=.36 if not frontal else .015;hy=-.07 if not frontal else -.21
    for s in [-1,1]:
        px=rear+(s*.46 if frontal else -.21);py=.23 if frontal else s*.18+.08
        im.ell(px,py,.32,.171,b-.076,s*.61,.1)
        endx=px+(s*.32 if frontal else -.30);endy=py+.16;im.line(px+.11,py,endx,endy,.057,b-.06)
        ax=hx+(s*.35 if frontal else -.05);ay=hy+.13;ex=ax+(s*.20 if frontal else .21);ey=ay+.30
        im.line(ax,ay,ex,ey,.072,b-.075)
        for j in [-1,0,1]:im.line(ex,ey,ex+j*.076+.05,ey+.10+.02*j,.021,b+.034)
    im.ell(rear,.035,.47,.303 if not frontal else .43,b,shade=.175)
    hr=U(.30,.40);im.ell(hx,hy,hr*(1.28 if frontal else 1.1),hr*.76,b+.04,shade=.15)
    for s in ([-1,1] if frontal else [-1]):
        ex=hx+(s*.74*hr if frontal else .14*hr);ey=hy-.53*hr
        im.ell(ex,ey,.12,.127,b+.018,shade=.13);im.ell(ex+.015,ey-.025,.069,.063,.49);im.ell(ex+.018,ey-.024,.053,.029,.07)
    if frontal:im.line(hx-.75*hr,hy+.25*hr,hx+.80*hr,hy+.23*hr,.014,b-.18)
    for j in range(9):
        a=U(0,6.28);r=U(.10,.88);im.ell(rear+r*np.cos(a)*.42,.035+r*np.sin(a)*.27,U(.022,.061),U(.021,.056),b-U(.035,.15))

def render(k):
    port=k in [3,4,5,7] and prob({3:.59,4:.21,5:.60,7:.18}[k]);im=Canvas(k,port)
    if port:portrait(im,k)
    elif k in [3,4,5,7]:quadruped(im,k)
    elif k in [1,9]:vehicle(im,k==9)
    elif k==0:plane(im)
    elif k==2:bird(im)
    elif k==6:frog(im)
    elif k==8:boat(im)
    return im.resample(im.a),im.resample(im.mask.astype(float))

# Network construction utilities. Weights never depend on external files.
layers=[];weights={}
def layer(name,op,src,**kw):
    layers.append(dict(name=name,op=op,inputs=[src] if isinstance(src,str) else src,**kw));return name
def weighted(name,op,src,w,b,**kw):
    weights[name+'.W']=np.asarray(w,np.float32);weights[name+'.b']=np.broadcast_to(np.asarray(b,np.float32),(len(w),)).copy();return layer(name,op,src,out=len(w),**kw)
def conv(name,src,w,b=0,pad=0,stride=1):
    w=np.asarray(w);return weighted(name,'conv',src,w,b,kernel=list(w.shape[2:]),padding=pad,stride=stride)
def dense(name,src,w,b=0):return weighted(name,'dense',src,w,b)
def relu(name,src):return layer(name,'relu',src)
def norm(name,src):return layer(name,'norm',src)
def shape(name,src,x):return layer(name,'reshape',src,shape=list(x))
def pool(name,src,ks,st,op='avgpool'):return layer(name,op,src,size=list(ks),stride=list(st))
def add(name,*src):return layer(name,'add',list(src))

# Local gradient direction indicators with a triangular angular tuning curve.
lum=np.array([.27,.55,.18]); dx=np.outer([.25,.5,.25],[-.5,0,.5]);dy=dx.T
proj=np.stack([math.cos(t*np.pi/4)*dx+math.sin(t*np.pi/4)*dy for t in range(4)])
rot=np.concatenate([proj,-proj]); mix=np.zeros((4,8))
for i in range(4):mix[i,[i,i+4]]=1;mix[i,[(i+2)%4,(i+2)%4+4]]=-1
G=conv('gradient', 'image',rot[:,None]*lum[None,:,None,None]);GR=relu('half_waves',G)
OM=relu('oriented_edges',conv('angle_tuning',GR,mix[:,:,None,None]));HOG=pool('orientation_histograms',OM,(6,6),(3,3));HN=norm('edge_contrast',shape('histogram_array',HOG,(1,36,9)))
SM=pool('color_16','image',(2,2),(2,2));LUM=conv('luminance',SM,lum[None,:,None,None]);LN=norm('exposure_invariant',LUM)
gauss=np.array([1,4,6,4,1])/16;hp=-np.outer(gauss,gauss);hp[2,2]+=1
BP=norm('detail_contrast',conv('local_detail',LN,hp[None,None],pad=2))

def unit(a):return a/(np.sqrt(np.vdot(a,a))+1e-8)
win=np.lib.stride_tricks.sliding_window_view
def describe(im,mask):
    # Construct a matched filter from just this drawing; no other drawing or
    # response enters the calculation. This is not a classifier evaluation.
    grad=win(im,(3,3));gx=(grad*dx).sum(axis=(-1,-2));gy=(grad*dy).sum(axis=(-1,-2))
    vec=np.array([np.abs(np.cos(i*np.pi/4)*gx+np.sin(i*np.pi/4)*gy) for i in range(4)])
    o=np.maximum(0,vec-np.roll(vec,2,axis=0));h=win(o,(6,6),axis=(1,2))[:,::3,::3].mean(axis=(-1,-2))
    h=h-h.mean(axis=(1,2),keepdims=True);h=unit(h)/np.sqrt(h.size)
    l=im.reshape(16,2,16,2).mean(axis=(1,3));l-=l.mean();d=(win(np.pad(l,2),(5,5))*hp).sum(axis=(-1,-2));d-=d.mean()
    silhouette=mask.reshape(8,4,8,4).mean(axis=(1,3));silhouette-=silhouette.mean()
    return h.ravel(),unit(l).ravel()/16,unit(d).ravel()/16,unit(silhouette).ravel()/8

def color_conspicuity():
    # A pixel is conspicuous if it differs from the surround in luminance or
    # either chromatic opponent direction. The reference is computed per photo.
    c8=pool('color_8','image',(4,4),(4,4));basis=np.array([lum,[1,-1,0],[-.44,-.56,1]])
    opp=conv('opponent_axes',c8,basis[:,:,None,None]);rim=regions(8,8)['rim'].ravel();m=np.eye(64)-rim[None,:]
    diff=np.kron(np.eye(3),m);sgn=np.concatenate([diff,-diff]);ab=relu('unsigned_center_surround',dense('center_surround',opp,sgn));ab=shape('surround_deviations',ab,(6,8,8))
    dist=conv('color_distance',ab,np.tile([.68,1.15,1.03],2)[None,:,None,None]);return norm('silhouette_contrast',dist)

def template_bank():
    groups,per=6,192;n=10*groups*per
    sal=color_conspicuity();mats=[np.empty((n,m),np.float32) for m in [324,256,256,64]]
    i=0
    for k in range(10):
        for _ in range(groups*per):
            for mat,row in zip(mats,describe(*render(k))):mat[i]=row
            i+=1
    vals=[dense(name,src,mat*coef) for name,src,mat,coef in zip(['edge_matches','tone_matches','detail_matches','silhouette_matches'],[HN,LN,BP,sal],mats,[.51,.205,.275,.115])]
    allm=shape('illustrations',add('match_evidence',*vals),(10,groups,per));best=pool('best_pose_in_group',allm,(1,per),(1,per),'maxpool')
    av=pool('group_consensus',best,(groups,1),(groups,1));mx=pool('strongest_pose',best,(groups,1),(groups,1),'maxpool')
    return add('shape_scores',dense('broad_match',av,.73*np.eye(10)),dense('specific_match',mx,.27*np.eye(10)))

# A small expression compiler for continuous pixel predicates. A pair of
# rectifiers bounds each predicate in [0,1], so no single extreme pixel wins.
def color_planes():
    E=np.eye(4);r,g,b,one=E;lumv=.27*r+.55*g+.18*b
    aff=[]; expr=[]; names=[]
    def ramp(a,coef=1): aff.append(a);return (len(aff)-1,coef)
    def make(name,a,other=None,sub=()):
        e=[ramp(a)] if other is None else [ramp(a),ramp(a-other,-1)]
        # min(x,y)=x-relu(x-y); 'a' may be negative, so keep its negative half, too.
        e.append(ramp(-a,-1));e.extend(ramp(v,c) for v,c in sub)
        expr.append(e);names.append(name)
    make('blue',7*(b-r)+.03*one,11*(b-g)+.33*one)
    make('green',8.5*(g-b)-.01*one,12*(g-r)+.31*one)
    make('warm',8*(r-g)+.035*one,9*(g-b)+.03*one)
    make('red',5*(r-g)-.46*one,5*(r-b)-.48*one)
    make('yellow',6*(g-b)-.47*one,7*(r-b)-.50*one)
    # max(rgb)-min(rgb) = half the sum of absolute pair differences.
    ds=[r-g,g-b,b-r];make('neutral',1.03*one,sub=[(s*d,-2.6) for d in ds for s in [-1,1]])
    make('white',4.4*lumv-2.50*one,sub=[(s*d,-1.1) for d in ds for s in [-1,1]])
    make('bright',3.1*lumv-1.55*one);make('dark',1.27*one-3.15*lumv);make('tone',lumv)
    a=np.array(aff);first=relu('positive_color_contrasts',conv('color_axes',SM,a[:,:3,None,None],a[:,3]));m=np.zeros((len(names),len(aff)))
    for i,ls in enumerate(expr):
        for j,c in ls:m[i,j]+=c
    cl=relu('color_ramps',conv('color_ranges',first,np.concatenate([m,m])[:,:,None,None],np.repeat([0.,-1.],len(m))))
    return conv('pixel_colors',cl,np.concatenate([np.eye(len(m)),-np.eye(len(m))],axis=1)[:,:,None,None]),names

def regions(h,w,scale=1):
    y,x=np.mgrid[0:h,0:w];x=(2*(x+.5)/w-1)*scale;y=(2*(y+.5)/h-1)*scale
    rad=(x/.68)**2+(y/.55)**2
    masks={'all':np.ones_like(x),'subject':np.exp(-1.50*rad),'surround':1-np.exp(-.65*rad),
           'sky':np.maximum(-y-.14,0)**.65,'ground':np.maximum(y-.18,0)**.63,
           'rim':np.maximum(abs(x)-.55,0)+np.maximum(abs(y)-.54,0),
           'bottom':np.maximum(y-.44,0),'upper_obj':np.exp(-1.8*((x/.70)**2+((y+.22)/.41)**2)),
           'under_obj':np.exp(-1.6*((x/.78)**2+((y-.29)/.35)**2))}
    return {n:v/v.sum() for n,v in masks.items()}

# Pool scalar scene attributes without concatenation (the language has add).
# Each output slot has a single source. The rest of that row is identically 0.
def scene_bank():
    color,cn=color_planes();sources=[];indices={}
    def feature_set(src,names,h,w,zones,scale=1):
        masks=regions(h,w,scale);arr=[];keys=[]
        for i,name in enumerate(names):
            for zone in zones:
                a=np.zeros((len(names),h,w),np.float32);a[i]=masks[zone];arr.append(a.ravel());keys.append(name+'_'+zone)
        sources.append((src,arr,keys))
    feature_set(color,cn,16,16,['all','subject','surround','sky','ground','bottom','upper_obj','under_obj'])
    # Compare coherent long edges with short edges. Vehicles and vessels often
    # have straight boundaries extending much farther than a tuft or leaf.
    feature_set(OM,['vert','diag1','horiz','diag2'],30,30,['all','subject','sky','ground','rim'])
    # Difference between two parallel strips, integrated along their length.
    linefs=[];names=[]
    xx,yy=np.meshgrid(np.arange(-4,5),np.arange(-4,5))
    for i,name in enumerate(['longv','longd1','longh','longd2']):
        th=i*np.pi/4; across=xx*np.cos(th)+yy*np.sin(th);along=-xx*np.sin(th)+yy*np.cos(th)
        f=(np.exp(-.5*((across-1.05)/.72)**2)-np.exp(-.5*((across+1.05)/.72)**2))*np.exp(-.5*(along/3.15)**2);f-=f.mean();f/=np.maximum(f,0).sum()
        linefs.extend([f,-f]);names.append(name)
    lg=relu('unsigned_line_halves',conv('long_edge_filters','image',np.asarray(linefs)[:,None]*lum[None,:,None,None],stride=2));ls=np.repeat(np.eye(4),2,axis=1)
    lg=conv('line_evidence',lg,ls[:,:,None,None]);feature_set(lg,names,12,12,['all','subject','sky','ground'],.78)
    total=sum(len(r) for _,r,_ in sources);start=0;parts=[]
    for src,arr,keys in sources:
        w=np.zeros((total,len(arr[0])),np.float32);w[start:start+len(arr)]=arr
        for i,key in enumerate(keys):indices[key]=start+i
        start+=len(arr);parts.append(dense(src+'_region_features',src,w))
    features=add('scene_measurements',*parts);axes=np.eye(total);val={key:axes[idx] for key,idx in indices.items()}
    classorder=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    aff=[];bias=[];table=[]
    def vote(v,lo,hi,**kw):
        scores=np.array([kw.get(k,0.) for k in classorder]);aff.extend([v/(hi-lo),v/(hi-lo)]);bias.extend([-lo/(hi-lo),-hi/(hi-lo)]);table.extend([scores,-scores])
    def V(c,z):return val[c+'_'+z]
    def texture(z):return sum(V(c,z) for c in ['vert','horiz','diag1','diag2'])
    def cardinal(z):return V('vert',z)+V('horiz',z)-V('diag1',z)-V('diag2',z)
    def long(z):return [V(c,z) for c in ['longv','longh','longd1','longd2']]
    # Sky around an isolated object versus water below a vessel.
    vote(V('blue','surround'),.04,.53,plane=.117,ship=.063,bird=.021,cat=-.037,deer=-.046,dog=-.031,frog=-.054,horse=-.029)
    vote(V('blue','sky'),.11,.70,plane=.047,truck=.030,ship=.041,horse=.010)
    vote(V('blue','ground'),.09,.65,ship=.081,plane=.049,car=-.014,truck=-.014,deer=-.024,horse=-.026)
    vote(V('blue','subject')-.58*V('blue','surround'),.10,.53,car=.049,truck=.028,bird=.054)
    vote(V('white','surround'),.12,.71,plane=.068,ship=.044,bird=.022,frog=-.057,deer=-.026)
    vote(V('white','sky')-V('white','ground'),.14,.72,truck=.030,ship=.048,horse=.012)
    vote(V('tone','ground')-V('tone','subject'),.065,.38,plane=.050,ship=.026,frog=-.026)
    vote(V('tone','sky')-V('tone','ground'),.10,.52,ship=.045,plane=.012,truck=.024,car=.008,frog=-.022)
    # Vegetation, and how it is distributed relative to the subject.
    vote(V('green','surround'),.07,.57,deer=.132,horse=.088,bird=.076,frog=.079,plane=-.082,ship=-.079,car=-.042,truck=-.042,cat=-.029,dog=-.012)
    vote(V('green','all'),.22,.70,frog=.051,deer=.035,bird=.024,car=-.025,ship=-.022)
    vote(V('green','ground')-V('green','sky'),.05,.56,horse=.074,deer=.038,truck=.024,plane=.013,frog=-.016)
    vote(V('green','sky')-.60*V('green','ground'),.075,.42,deer=.039,bird=.038,horse=-.018)
    vote(V('green','subject')-.61*V('green','surround'),.071,.53,frog=.088,bird=.016,horse=-.035,deer=-.026,cat=-.026,dog=-.020)
    vote(V('green','surround')-V('green','subject'),.12,.59,horse=.042,deer=.044,bird=.033,dog=.020)
    vote(V('warm','subject')+.38*V('green','ground'),.20,.79,horse=.073,deer=.052,dog=.037,frog=.032,plane=-.046,ship=-.032)
    vote(V('warm','surround'),.12,.66,cat=.047,dog=.038,frog=.046,deer=.026,plane=-.048,ship=-.047,truck=-.011)
    vote(V('warm','subject')-V('warm','surround'),.09,.61,horse=.044,dog=.045,deer=.029,bird=.013)
    vote(V('yellow','subject'),.07,.52,frog=.044,bird=.040,truck=.015,cat=-.017)
    vote(V('red','subject')-.35*V('red','surround'),.045,.47,car=.091,truck=.079,bird=.058,horse=.016,deer=-.032,frog=-.027)
    # Achromatic subjects and road surfaces vs high-chroma habitats.
    vote(V('neutral','bottom'),.47,.92,car=.068,truck=.058,plane=.025,ship=.033,cat=.015,frog=-.033,deer=-.036,horse=-.023)
    vote(V('neutral','all'),.44,.89,cat=.042,car=.032,truck=.027,plane=.017,ship=.017,frog=-.033,deer=-.024)
    vote(V('neutral','subject')-V('neutral','surround'),.08,.45,cat=.028,dog=.023,truck=.023,car=.021,bird=.010)
    vote(V('dark','surround'),.15,.56,frog=.082,cat=.031,dog=.020,plane=-.063,ship=-.052,truck=-.026)
    vote(V('dark','subject'),.20,.62,frog=.041,cat=.010,dog=.021,car=.015,plane=-.021)
    vote(V('dark','under_obj')-V('dark','bottom'),.064,.46,car=.046,truck=.038,ship=.024)
    vote(V('white','upper_obj')-V('white','surround'),.09,.54,truck=.042,dog=.019,ship=.028,car=.012)
    # Horizontal structure is common to roads, water, and vehicle bodies.
    lv,lh,ld1,ld2=long('subject');av,ah,ad1,ad2=long('all');gv,gh,gd1,gd2=long('ground')
    vote(ah-(ad1+ad2)*.5,.003,.047,ship=.084,car=.050,truck=.034,cat=-.022,dog=-.021,frog=-.023,bird=-.012)
    vote(av+ah-ad1-ad2,.008,.079,truck=.093,car=.059,ship=.041,plane=-.013,deer=-.040,horse=-.033,bird=-.025,frog=-.039,cat=-.025,dog=-.026)
    vote(lv-.51*(ld1+ld2),.004,.06,truck=.059,car=.025,deer=.011,ship=-.012)
    vote(gh-gv,.006,.066,ship=.066,car=.025,truck=.016,cat=-.014)
    vote(cardinal('all'),.001,.029,truck=.047,car=.034,ship=.023,bird=-.013,frog=-.026,deer=-.013)
    vote(cardinal('all'),-.025,-.002,car=.020,truck=.022,ship=.015,bird=-.012)
    vote(V('horiz','ground')-V('vert','ground'),.004,.038,ship=.043,car=.021,truck=.017)
    # Smooth sky, defocused habitats, textured foliage, and indoor closeups.
    vote(-texture('rim'),-.089,-.018,plane=.098,bird=.051,ship=.035,cat=.016,deer=-.035,frog=-.025)
    vote(-texture('sky'),-.078,-.014,plane=.070,ship=.045,bird=.023,truck=.022,cat=-.012,frog=-.024)
    vote(texture('rim'),.080,.203,frog=.050,deer=.045,horse=.022,plane=-.065,ship=-.026,bird=-.019)
    vote(texture('subject')-texture('rim'),.028,.135,bird=.065,cat=.021,dog=.024,car=.019)
    vote(-texture('all'),-.103,-.046,cat=.059,dog=.043,bird=.017,truck=-.019)
    vote(V('green','surround')-2.4*texture('rim'),.025,.38,bird=.051,deer=.019)
    vote(V('green','surround')+2.6*texture('rim'),.50,.96,deer=.057,frog=.047,horse=.013)
    vote(V('dark','all')+.55*V('warm','all')-.7*V('white','sky'),.23,.68,frog=.061,cat=.024,dog=.015)
    votes=relu('rule_strengths',dense('scene_conditions',features,np.array(aff),bias))
    # Intercepts are qualitative priors for missing cues. Aircraft, ships,
    # and vehicles can collect several correlated positive context votes.
    intercept=[-.096,-.080,-.057,.010,-.060,-.005,-.040,-.042,-.114,-.104]
    return dense('scene_scores',votes,np.asarray(table).T,intercept)

def wheel_bank():
    # Tire cross-sections. Below the axle a tire meets the road, so the
    # negative annulus is compared mainly with the outward/downward surround.
    y,x=np.mgrid[-4:5,-4:5];filters=[]
    for r in [1.45,2.20,2.95]:
        dist=np.sqrt((x/ .89)**2+y*y);tire=np.exp(-.5*((dist-.76*r)/(.28*r))**2)*(1+.22*y/r)
        exterior=np.exp(-.5*((dist-1.37*r)/(.30*r))**2)*np.clip(1+y/r,0,2)
        hub=np.exp(-.5*(dist/(.34*r))**2)
        def mass(a):return a/a.sum()
        filters.append(.77*mass(exterior)+.23*mass(hub)-mass(np.maximum(tire,0)))
    gray=conv('full_luminance','image',lum[None,:,None,None]);w=conv('rubber_rings',gray,np.array(filters)[:,None]);w=relu('dark_tires',w);w=pool('axle_locations',w,(2,2),(2,2),'maxpool')
    # A pair has to straddle the lower middle. Limit the separation as a
    # function of radius. min(a,b)=(a+b-|a-b|)/2 prevents one wheel alone
    # from voting for a vehicle. No measured examples set the tolerances.
    entries=[]
    for c,(low,high) in enumerate([(3,6),(4,8),(5,9)]):
        for sep in range(low,high+1):
            for left in range(0,12-sep):
                if not 3.5<=left+.5*sep<=7.5:continue
                for row in range(5,12):
                    for tilt in [-1,0,1]:
                        if 0<=row+tilt<12:entries.append((c,row,left,row+tilt,left+sep))
    n=len(entries);m=np.zeros((3,n,3,12,12),np.float32)
    for j,(c,y1,x1,y2,x2) in enumerate(entries):
        for z,(a,b) in enumerate([(1,1),(1,-1),(-1,1)]):
            m[z,j,c,y1,x1]=a;m[z,j,c,y2,x2]=b
    parts=relu('wheel_pair_parts',dense('paired_wheel_contrasts',w,m.reshape(3*n,-1)));parts=shape('wheelbase_candidates',parts,(3,1,n))
    weaker=conv('joint_wheel_support',parts,np.array([.5,-.5,-.5])[None,:,None,None]);peak=pool('best_wheel_pair',weaker,(1,n),(1,n),'maxpool')
    gate=relu('wheel_pair_ramp',dense('wheel_pair_gate',peak,[[1/.18],[1/.18]],[-.13/.18,-.31/.18]));impact=np.array([-.008,.068,-.007,-.004,-.006,-.004,-.004,-.004,-.007,.051]);return dense('wheel_scores',gate,np.stack([impact,-impact],axis=1))

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    shapescore=template_bank();scene=scene_bank();wheels=wheel_bank();add('class_scores',shapescore,scene,wheels)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n');np.savez(out/'weights.npz',**weights)
if __name__=='__main__':main()
