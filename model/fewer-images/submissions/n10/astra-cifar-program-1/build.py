import sys
from pathlib import Path
import numpy as np
from math import sin, cos, pi, sqrt

# Object coordinates: +x is the nose, z is up, y is the lateral axis.
# Faces carry their own material; no texture assets or measured colours.
def rgb(v):
    if np.isscalar(v): return np.array([v,v,v],float)
    return np.array(v,float)

def bounds(v, a, b): return max(a,min(b,v))

class Draw:
    def __init__(self,u):
        self.u=u
        self.faces=[]
        self.light=np.array([u.uniform(.05,.40), u.uniform(-.58,-.18), u.uniform(.62,.92)])
        self.light[1]*=u.choice([-1,1])
    def face(self, vertices, col, normal=None):
        v=np.array(vertices,float)
        if normal is None: normal=np.cross(v[1]-v[0],v[-1]-v[0])
        normal=np.asarray(normal)
        shade=.79+.22*(normal@self.light)/max(1.e-10,np.linalg.norm(normal))
        self.faces.append((v,np.clip(rgb(col)*shade,0,1)))
    def block(self, rings, colors):
        # Each section lists the corners around a cross-section, back to front.
        rings=np.array(rings); n=rings.shape[1]; center=rings.mean(axis=(0,1))
        def piece(points,color):
            normal=np.cross(points[1]-points[0],points[-1]-points[0]); out=points.mean(axis=0)-center
            self.face(points,color,normal if normal@out>=0 else -normal)
        piece(rings[0],colors[-2]);piece(rings[-1],colors[-1]);
        for a,b in zip(rings,rings[1:]):
            for j in range(n):
                l=(j+1)%n; piece(np.array([a[j],a[l],b[l],b[j]]),colors[j])
    def loft(self,rings,cols):
        # Stations x, bottom and top half-widths, lower and upper heights.
        self.block([[(x,-wb,zb),(x,wb,zb),(x,wt,zt),(x,-wt,zt)] for x,wb,wt,zb,zt in rings],cols)
    def box(self, x0,x1,y0,y1,z0,z1,c):
        self.block([[(x,y0,z0),(x,y1,z0),(x,y1,z1),(x,y0,z1)] for x in [x0,x1]],[c]*6)
    def ell(self,p,r,c,tilt=0):
        # An analytic ellipse under the camera projection; draw at center depth.
        self.faces.append(('e', np.array(p), np.array(r), rgb(c), tilt))
    def bar(self,a,b,ra,rb,c,sides=6):
        a,b=np.array(a),np.array(b); n=b-a;n=n/max(1.e-12,np.linalg.norm(n));
        ax=np.array([0.,1,0]) if abs(n[1])<.85 else np.array([0.,0,1]);v=np.cross(ax,n);v/=max(1.e-12,np.linalg.norm(v)); w=np.cross(n,v)
        t=np.arange(sides)*2*pi/sides;circle=np.cos(t[:,None])*v+np.sin(t[:,None])*w
        self.block([a+ra*circle,b+rb*circle],[c]*(sides+2))
    def stripe(self,a,b,width,c): self.bar(a,b,width,width,c,4)
    def wheel(self,x,y,z,r,wide):
        s=np.sign(y);rim=self.u.uniform(.36,.78);dark=self.u.uniform(.045,.14)
        yz=y+s*wide*.50
        # Stock tire, recessed wheel and hubcap.
        self.bar((x,y-s*wide*.50,z),(x,yz,z),r,r,dark,12)
        for frac,dist,col in ((.60,.007,rim),(.24,.012,.25)):
            at=yz+dist*s;t=np.linspace(0,2*pi,13)[:-1];self.face([(x+r*frac*cos(a),at,z+r*frac*sin(a)) for a in t], col,[0,s,0])

def car(m,u):
    # Long low body on two axles with a raised passenger compartment.
    style=u.randint(5); L=u.uniform(.91,1.1); w=u.uniform(.36,.46);base=u.uniform(.16,.21); hood=u.uniform(.43,.56);roof=hood+u.uniform(.27,.41)
    col=paint(u); glass=glasscolor(u);silver=np.array([.68,.69,.70]);tire=u.uniform(.17,.205)
    back=-L+u.uniform(.04,.32) if style in (1,2) else -L+u.uniform(.02,.10)
    if style==1: roof+=.13
    m.loft([(-L*.99,w*.85,w*.88,base+.035,hood-.045),(-L*.72,w,w,base,hood),(.66*L,w,w*.98,base,hood-.02),(L,w*.88,w*.83,base+.045,hood-.085)],[col*.55,col,col*1.06,col,col,col])
    rear=-L*(.92 if style in (1,2,3) else .68);rearTop=rear+L*(.10 if style in (1,2) else .24);frontTop=L*u.uniform(.05,.22);front=.63*L
    m.loft([(rear,w*.96,w*.96,hood-.02,hood),(rearTop,w*.98,w*.80,hood-.02,roof),(frontTop,w*.98,w*.79,hood-.02,roof-.01),(front,w*.96,w*.95,hood-.02,hood)],[col*.7,glass,col,glass,glass,glass])
    # Windshields are sloped panels between roof and hood; top of loft transitions
    # from painted lid to glass on those sections.
    for x1,z1,w1,x2,z2,w2 in ((rear,hood,w*.96,rearTop,roof,w*.80),(front,hood,w*.95,frontTop,roof-.01,w*.79)):
        m.face([(x1,-w1,z1+.005),(x1,w1,z1+.005),(x2,w2,z2+.005),(x2,-w2,z2+.005)],glass,[(1 if x1>0 else -1),0,1]);
        for sy in [-1,1]:m.stripe((x1,w1*sy,z1),(x2,w2*sy,z2),.016,col)
    for sy in [-1,1]:
        m.stripe((-.30*L,sy*w*.98,hood),(-.30*L,sy*w*.81,roof),.023,col*.75)
        m.stripe((rear,sy*w*.97,hood),(front,sy*w*.97,hood),.018,col)
        for xx in [-.65*L,.67*L]:m.wheel(xx,sy*w*.98,tire,tire,.10)
    for sy in [-1,1]:
        m.face([(L+.002,sy*w*.5,hood-.095),(L+.002,sy*w*.87,hood-.095),(L+.002,sy*w*.87,hood-.18),(L+.002,sy*w*.5,hood-.18)],u.uniform(.70,.98),[1,0,0])
    m.box(L-.01,L+.012,-w*.5,w*.5,base+.065,hood-.11,.16);m.box(L-.04,L+.02,-w*.82,w*.82,base+.03,base+.068,silver)

def paint(u):
    v=u.randint(9)
    if v<4:return rgb(u.uniform(.70,.98) if v<2 else u.uniform(.10,.57))
    return rgb([(.74,.16,.13),(.16,.34,.55),(.29,.44,.35),(.76,.64,.39),(.46,.18,.21)][v-4])*u.uniform(.75,1.12)

def glasscolor(u):
    v=u.uniform(.14,.51);return rgb([.87*v,.98*v,min(.90,1.19*v)])

def truck(m,u):
    L=u.uniform(.98,1.20);w=u.uniform(.28,.40);z=u.uniform(.28,.38); top=u.uniform(.82,1.14);col=paint(u);cab=paint(u);wheel=u.uniform(.135,.19);g=glasscolor(u)
    kind=u.randint(5); boxEnd=L*u.uniform(.14,.38);cabEnd= L*(1 if kind in (0,1,4) else .76);cz=top*u.uniform(.74,.97)
    m.box(-L*.97,L*.97,-w*.7,w*.7,z*.57,z,.17)
    if kind==3:
        m.bar((-L*.90,0,(z+top)*.5),(boxEnd,0,(z+top)*.5),(top-z)*.43,(top-z)*.43, col,12)
    else:
        if kind==4: top=cz*.77
        m.box(-L,boxEnd,-w,w,z,top,col)
        for sy in [-1,1]:
            y=sy*(w+.002);m.stripe((-L,y,z+.025),(boxEnd,y,z+.025),.014,.58)
            if kind==1:
                for xx in np.linspace(-L+.06,boxEnd-.03,11): m.stripe((xx,y,z+.045),(xx,y,top-.035),.007,col*.85)
    c0=boxEnd+u.uniform(.02,.11);w2=w*.94
    m.loft([(c0,w2,w2*.87,z*.78,cz),(cabEnd-.10,w2,w2*.87,z*.78,cz),(cabEnd,w2,w2*.92,z*.78,cz*.76)],[cab,cab,cab,cab,cab,cab])
    for sy in [-1,1]:
        m.face([(c0+.035,w2*sy*.97,cz*.65),(cabEnd-.025,sy*w2*.965,cz*.65),(cabEnd-.12,sy*w2*.89,cz*.94),(c0+.035,sy*w2*.88,cz*.94)],g,[0,sy,0]);
    m.face([(cabEnd+.002,-w2*.86,cz*.66),(cabEnd+.002,w2*.86,cz*.66),(cabEnd-.085,w2*.80,cz*.95),(cabEnd-.085,-w2*.80,cz*.95)],g,[1,0,.35]);
    if cabEnd<L:m.box(cabEnd-.015,L,-w*.78,w*.78,z*.8,cz*.61,cab)
    m.box(L-.035,L+.01,-w2*.60,w2*.60,z*.93,cz*.56,.26)
    for sy in [-1,1]:
        for xx in [-.74*L,.77*L]+([-.46*L] if kind in (1,2,3) else []):m.wheel(xx,sy*w*.92,wheel,wheel,.12)
        m.box(L-.017,L+.012,min(sy*w2*.65,sy*w2*.92),max(sy*w2*.65,sy*w2*.92),cz*.43,cz*.55,.88)

def plane(m,u):
    kind=u.randint(4);c=rgb(u.uniform(.55,.96)) if u.rand()<.75 else paint(u);w=u.uniform(.082,.14) if kind!=3 else .16; span=u.uniform(.82,1.18); sweep=u.uniform(.28,.57) if kind in (0,1) else u.uniform(.06,.30);z=.03
    m.ell((0,0,z),(1,w,w*.99),c)
    m.ell((.72,0,z+w*.40),(.15,w*.79,w*.58),glasscolor(u))
    fin=paint(u) if u.rand()<.5 else c*.85
    for sy in (-1,1):
        # swept wings, tailplanes and engine nacelles below the wings
        pts=[(.32,0,z-.015),(.24-sweep,sy*span,z+.04),(.12-sweep,sy*span,z+.035),(-.20,sy*.18,z-.025),(-.30,0,z-.02)]
        m.face(pts,c*u.uniform(.73,.94),[0,0,1]);m.face([(-.53,0,z),(-.82,sy*span*.42,z+.06),(-.94,sy*span*.42,z+.06),(-.86,0,z)],c*.83,[0,0,1]);
        if kind in (0,1):m.ell((.09,sy*span*.39,z-.09),(.22,.064,.073),c*.72)
        if kind==0:
            for xx in np.arange(-.5,.59,.099):m.ell((xx,w*.96*sy,z+.023),(.025,.009,.017),.21)
    m.face([(-.44,0,z+w*.3),(-.80,0,z+.36),(-.90,0,z+.36),(-.94,0,z+w*.2)],fin,[0,-1,0]);
    if kind==2:m.bar((1,0,z-.21),(1,0,z+.21),.018,.018,.24,4)

def ship(m,u):
    kind=u.randint(6);L=1.08;w=u.uniform(.22,.38);z=u.uniform(.23,.39);c=rgb(u.uniform(.69,.96) if kind in (0,1,4) else u.uniform(.13,.42));deck=rgb(u.uniform(.61,.95));
    m.loft([(-L*.98,w*.53,w*.72,.02,z*1.1),(-L*.75,w*.73,w,0,z),(.53*L,w*.73,w,0,z),(.86*L,w*.47,w*.64,.045,z*1.03),(L,w*.04,.01,.13,z*1.22)],[c*.5,c,c*.86,c,c,c]);
    if kind==4:
        mast=u.uniform(1.28,1.67);m.stripe((-.13,0,z),(-.13,0,mast),.013,.45)
        m.face([(-.15,0,mast-.035),(-.80,0,z+.18),(-.15,0,z+.19)],deck,[0,1,0]);m.face([(-.09,0,mast-.19),(.66,0,z+.15),(-.08,0,z+.12)],.91,[0,1,0]);return
    if kind in (0,1):x0,x1,h=-.86,.59,u.uniform(.10,.18);n=u.randint(2,5)
    else:x0=u.uniform(-.80,.0);x1=x0+u.uniform(.39,.75);h=u.uniform(.15,.24);n=u.randint(1,3)
    for d in range(n):
        ya=w*(.91-.085*d);a=x0+.06*d;b=x1-.055*d
        m.box(a,b,-ya,ya,z+h*d,z+h*(d+1),deck)
        for sy in (-1,1):
            for xx in np.arange(a+.045,b-.02,.118):
                m.face([(xx,sy*(ya+.003),z+h*(d+.38)),(min(xx+.079,b),sy*(ya+.003),z+h*(d+.38)),(min(xx+.079,b),sy*(ya+.003),z+h*(d+.79)),(xx,sy*(ya+.003),z+h*(d+.79))],.23,[0,sy,0])
    m.bar((x0*.5+x1*.5,0,z+n*h),(x0*.53+x1*.47,0,z+n*h+.19),.077,.066,paint(u),8)
    mx=.69 if kind in (2,3,5) else x1-.08;m.stripe((mx,0,z),(mx,0,z+n*h+.27),.013,.38)

def fur(u,kind):
    if kind=='frog':return rgb([( .38,.47,.15),(.52,.38,.20),(.31,.30,.15),(.46,.57,.26),(.50,.45,.29)][u.randint(5)])*u.uniform(.72,1.20)
    if kind in ('cat','dog'):
        i=u.randint(10)
        if i<3:return rgb(u.uniform(.13,.63))
        if i<5:return rgb([.93,.91,.85])*u.uniform(.75,1.05)
    if kind=='horse' and u.rand()<.25:return rgb(u.uniform(.14,.88))
    return rgb([.59,.423,.285])*u.uniform(.60,1.32)*np.array([u.uniform(.91,1.11),1,u.uniform(.82,1.13)])

def leg(m,hip,knee,foot,a,b,c,col,hoof=False):
    m.bar(hip,knee,a,b,col);m.ell(knee,(b,b,b),col);m.bar(knee,foot,b,c,col*.94)
    x,y,z=foot;m.ell((x+.021,y,z),(c*1.50,c*1.06,c*.80),.175 if hoof else col*.89)

def ungulate(m,u,kind):
    deer=kind=='deer'; col=fur(u,kind);w=.18 if deer else .23;high=u.uniform(.81,.97) if deer else u.uniform(.82,.98);rad=u.uniform(.21,.26) if deer else u.uniform(.28,.34); long=u.uniform(.55,.71); hx=.77 if deer else .90;headz=high+(.52 if deer else .58)*u.uniform(.78,1.12);headlen=.17 if deer else .235
    if u.rand()<.14:headz=high+u.uniform(-.33,.15); hx+=.09
    m.ell((-.07,0,high),(long,w,rad),col);m.ell((-.42,0,high+.02),(.31,w*.99,rad*.96),col)
    m.bar((.32,0,high+.015),(hx-.04,0,headz),rad*.93,headlen*.57,col);m.ell((hx,0,headz),(.94*headlen,headlen*.68,headlen*.91),col)
    m.ell((hx+headlen*.75,0,headz-headlen*.48),(headlen*.97,headlen*.55,headlen*.60),col*.95,-.61)
    for sy in [-1,1]:
        m.ell((hx+.049,sy*headlen*.66,headz+.015),(.037,.012,.031),.075)
        m.bar((hx-.075,sy*.048,headz+headlen*.55),(hx-.12,sy*(.16 if deer else .085),headz+headlen*(1.60 if deer else 1.41)),.060,.011,col)
        m.ell((hx+headlen*1.41,sy*headlen*.37,headz-headlen*.96),(.028,.015,.027),.18)
        for xx in [-.48,.31]:
            yy=sy*w*.57; dx=u.uniform(-.135,.135); leg(m,(xx,yy,high),(xx+(-.074 if xx<0 else .013)+dx*.45,yy*1.07,high*.53),(xx+dx,yy*1.19,.04),.098 if xx<0 else .078,.039,.028,col*(.87 if sy>0 else .99),True)
    if deer:
        m.ell((-.67,0,high+.03),(.135,.074,.059),col*.80,.68)
        if u.rand()<.34:
            for sy in [-1,1]:
                a=(hx-.055,.03*sy,headz+.10);b=(hx-.24,.22*sy,headz+.44);m.bar(a,b,.027,.008,col*.73)
                for f in (.49,.83):
                    mid=tuple(np.array(a)*(1-f)+np.array(b)*f);m.bar(mid,(mid[0]+.06,mid[1]+sy*.035,mid[2]+.13),.014,.004,col*.74)
    else:
        mane=col*u.uniform(.28,.67);m.ell((-.72,0,high-.19),(.095,.080,.31),mane,-.23)
        m.bar((.30,0,high+rad*.7),(hx-.13,0,headz+.10),.092,.048,mane)
        if u.rand()<.2:m.ell((hx+headlen*.75,-.052,headz-.085),(.13,.038,.059),rgb(.82),-.7)

def dogcat(m,u,kind,portrait):
    cat=kind=='cat';col=fur(u,kind); L=u.uniform(.46,.59);w=u.uniform(.19,.28);bhigh=u.uniform(.44,.68);hh=bhigh+u.uniform(.25,.42);hx=.48
    sit=portrait or u.rand()<.3
    if sit:m.ell((.02,0,.53),(.33,w*1.09,.53),col,-.23);hx=.22;hh=1.02
    else:m.ell((-.01,0,bhigh),(L,w,w*1.19),col);m.bar((.29,0,bhigh),(hx,0,hh),w*.94,w*.77,col)
    for sy in [-1,1]:
        yy=sy*w*.65
        for xx in (-.37,.33):
            if sit and xx<0:m.ell((-.18,yy,.23),(.26,.13,.21),col*.93);continue
            top=(xx,yy,bhigh*.94);dx=u.uniform(-.08,.08);leg(m,top,(xx+.02+dx,yy*1.05,bhigh*.50),(xx+.055+dx,yy*1.18,.039),w*.43,w*.25,w*.22,col)
    if u.rand()<.81:
        a=(-.42,0,bhigh*.8);b=(-.72,.018,bhigh+.03);c=(-.77,-.015,bhigh+(.36 if cat else .22));m.bar(a,b,.057,.049,col);m.bar(b,c,.05,.028,col)
    hr=u.uniform(.225,.29) if cat else u.uniform(.22,.32);hr*=1.13 if portrait else 1.;facew=hr*(1.05 if cat else .96);hcol=col;dark=col*u.uniform(.43,.77);m.ell((hx,0,hh),(hr*.88,facew,hr),hcol)
    floppy=not cat and u.rand()<.64;patch=not cat and u.rand()<.36;eyeCol=rgb([.62,.69,.37]) if cat else rgb([.24,.17,.13]);muzzle=hr*(.39 if cat else u.uniform(.48,.86));chin=col*.69+(.29 if cat or u.rand()<.42 else .095)
    for sy in [-1,1]:
        if floppy:
            m.ell((hx-.075,sy*facew*.89,hh+hr*.09),(hr*.61,hr*.33,hr*.93),dark,-.14)
        else:
            a=(hx-.10,sy*facew*.26,hh+hr*.68);b=(hx+.050,sy*facew*.89,hh+hr*.45);c=(hx-.056,sy*facew*.80,hh+hr*u.uniform(1.22,1.65));m.face([a,b,c],hcol,[.84,sy*.48,0]);m.face([(hx-.14,b[1],b[2]),b,c],dark,[-.1,sy,0]);
            if cat:m.face([(.04+.71*a[0]+.29*c[0],.71*a[1]+.29*c[1],.71*a[2]+.29*c[2]),(.03+.75*b[0]+.25*c[0],.75*b[1]+.25*c[1],.75*b[2]+.25*c[2]),(c[0]+.018,c[1],c[2]-.021)],col*.67+rgb([.23,.12,.11]),[1,sy*.2,0])
        if patch:m.ell((hx+hr*.61,sy*facew*.51,hh+hr*.19),(hr*.29,hr*.38,hr*.43),dark)
        ex=hx+hr*.74;ey=sy*facew*.48;ez=hh+hr*.15
        m.ell((ex,ey,ez),(hr*.205,hr*.202,hr*(.165 if cat else .16)),.07);m.ell((ex+.022,ey,ez+.004),(hr*.17,hr*.16,hr*.129),eyeCol);m.ell((ex+hr*.23,ey,ez+.005),(hr*.055,hr*.071,hr*.10),.04)
        m.ell((ex+hr*.27,ey-.006,ez+.014),(.008,.008,.008),.93)
    m.ell((hx+hr*.78,0,hh-hr*.30),(muzzle,hr*.59,hr*.39),chin)
    if cat:
        for sy in [-1,1]:m.ell((hx+hr*.92,hr*.25*sy,hh-hr*.25),(hr*.31,hr*.34,hr*.255),chin)
    m.ell((hx+hr*.78+muzzle*.86,0,hh-hr*(.16 if cat else .19)),(hr*.14,hr*(.22 if cat else .35),hr*(.12 if cat else .21)),rgb([.48,.29,.27]) if cat and u.rand()<.6 else rgb(.086))
    if cat and u.rand()<.52:
        for sy in [-1,1]:
            for d in (0,.18):m.ell((hx+hr*(.59-d),sy*hr*(.24+d),hh+hr*(.66+d*.36)),(.034,.019,.059),dark,-.48)
    return np.array((hx,0,hh))

def frog(m,u):
    col=fur(u,'frog');w=u.uniform(.26,.37);bh=.22;hx=.36
    m.ell((-.12,0,bh),(.52,w,.23),col,.08);m.ell((hx,0,.30),(.32,w*.97,.21),col)
    m.ell((hx+.085,0,.221),(.27,w*.94,.108),col*.72+rgb([.18,.17,.10]));
    for sy in [-1,1]:
        y=sy*w*.68;m.ell((hx+.05,y,.423),(.112,.094,.117),col*1.1);m.ell((hx+.092,y+sy*.047,.451),(.081,.064,.080),rgb([.58,.51,.31]));m.ell((hx+.129,y+sy*.076,.453),(.050,.050,.052),.067);m.ell((hx+.154,y+sy*.104,.481),(.014,.014,.012),.91)
        m.ell((-.30,sy*w,.16),(.34,.155,.15),col,-.12)
        a=(.24,sy*w*.77,.26);b=(.105,sy*w*1.45,.09);c=(.37,sy*w*1.66,.042);m.bar(a,b,.081,.057,col);m.bar(b,c,.057,.042,col)
        m.bar((-.45,sy*w*1.20,.14),(-.03,sy*w*1.43,.04),.090,.047,col)
        for j in (-1,0,1):
            end=(c[0]+.16,c[1]+.064*j,.027);m.bar(c,end,.018,.009,col);m.ell(end,(.025,.019,.012),col*.93)
    if u.rand()<.68:
        for j in range(16):
            px=u.uniform(-.53,.18);py=u.uniform(-w,w);z=1-((px+.12)/.52)**2-(py/w)**2
            if z>.07:m.ell((px,py,bh+.221*sqrt(z)),(.046,.038,.023),col*u.uniform(.52,.83))

def bird(m,u,flight):
    kind=u.randint(5);col=rgb([( .43,.32,.23),(.18,.18,.20),(.87,.86,.80),(.37,.47,.52),(.63,.52,.32)][kind])*u.uniform(.80,1.13);wing=col*u.uniform(.53,.86);beak=rgb([.65,.48,.21]) if u.rand()<.51 else rgb(.20)
    if flight:
        m.ell((0,0,.11),(.50,.135,.14),col);m.ell((.39,0,.17),(.147,.121,.13),col);m.bar((.48,0,.16),(.68,0,.153),.054,.004,beak)
        for sy in (-1,1):
            z=u.uniform(-.36,.50);m.face([(.13,0,.145),(.20,sy*.55,.18+z*.55),(-.17,sy*1.06,.20+z),(-.34,sy*1.12,.17+z),(-.23,sy*.43,.14+z*.47),(-.32,0,.15)],wing,[0,0,1]);m.face([(-.32,0,.13),(-.73,sy*.23,.17),(-.76,0,.14)],col*.87,[0,0,1])
    else:
        tilt=u.uniform(.16,.83);hx=.32*cos(tilt);hz=.36+.35*sin(tilt)
        m.ell((0,0,.34),(.44,.226,.265),col,tilt);m.ell((hx,0,hz+.076),(.204,.177,.197),col*.94);m.bar((hx+.15,0,hz+.062),(hx+u.uniform(.28,.42),0,hz+.035),.074,.004,beak)
        m.face([(-.21,-.10,.32),(-.71,-.12,.20-.2*tilt),(-.77,.08,.18-.2*tilt),(-.23,.13,.32)],wing,[0,0,1]);
        for sy in (-1,1):
            m.ell((-.083,sy*.179,.346),(.33,.069,.152),wing,tilt*.83);m.ell((hx+.084,sy*.142,hz+.107),(.036,.031,.039),.057);m.ell((hx+.09,sy*.168,hz+.12),(.010,.007,.011),.95)
            m.stripe((.04,sy*.082,.18),(-.019,sy*.12,-.016),.016,beak);m.stripe((-.019,sy*.12,-.016),(.088,sy*.15,-.031),.013,beak)

SOURCE = "import numpy as np\n\ndef saturate(x):\n    return np.clip(x, 0., 1.)\n\ndef blur(a):\n    z = np.pad(a, [(0,0)]*(a.ndim-2)+[(1,1),(1,1)], mode='edge')\n    h = (z[..., :-2] + 2*z[..., 1:-1] + z[..., 2:])*0.25\n    return (h[..., :-2,:]+2*h[...,1:-1,:]+h[...,2:,:])*0.25\n\ndef mean4(a):\n    return a.reshape(a.shape[:-2]+(8,4,8,4)).mean(axis=(-3,-1))\n\ndef pool(a):\n    return mean4(blur(a))\n\ndef deriv(a):\n    z = np.pad(a, [(0,0)]*(a.ndim-2)+[(1,1),(1,1)], mode='edge')\n    return (z[...,1:-1,2:]-z[...,1:-1,:-2])*0.5, (z[...,2:,1:-1]-z[...,:-2,1:-1])*0.5\n\ndef gray(a):\n    return a[:,0]*0.299 + a[:,1]*0.587 + a[:,2]*0.114\n\ndef normalize(a, weight, floor):\n    b = a.reshape(len(a),-1)\n    return b * (np.sqrt(weight)/(np.sqrt(np.sum(b*b,axis=1,keepdims=True)) + floor))\n\ndef salmap(small):\n    # A pixel in the centre that can also be explained by the colours at the\n    # sides, particularly at the same height, is less likely to be the subject.\n    trans = np.stack((gray(small), (small[:,0]-small[:,1])*0.85, (small[:,2]-small[:,1])*0.8),axis=1)\n    best = np.full((len(small),8,8), 1., np.float32)\n    same = best.copy()\n    ids = np.arange(8,dtype=np.float32)\n    for i in range(8):\n        for j in (0,7):\n            d = np.sum((trans-trans[:,:,i:i+1,j:j+1])**2,axis=1)\n            best = np.minimum(best,d)\n            same = np.minimum(same,d+0.0018*(ids[None,:,None]-i)**2)\n    for i in (0,7):\n        for j in (1,3,5,6):\n            d = np.sum((trans-trans[:,:,i:i+1,j:j+1])**2,axis=1)\n            best = np.minimum(best,d)\n    return np.sqrt(best+.4*same+0.0001)\n\ndef describe(x):\n    # Unsigned edges from the strongest of the three colour channels. The signed\n    # branch keeps luminance polarity (e.g. black rubber below bright bodywork).\n    a = x*0.4 + blur(x)*0.6\n    dx,dy = deriv(a)\n    k = np.argmax(dx*dx+dy*dy, axis=1)[:,None]\n    ux = np.take_along_axis(dx,k,axis=1)[:,0]\n    uy = np.take_along_axis(dy,k,axis=1)[:,0]\n    mag = np.sqrt(ux*ux+uy*uy+1.e-10)\n    energy = pool(mag)\n    div = 0.012+0.45*energy.mean(axis=(1,2),keepdims=True)+0.55*blur(energy)+0.35*energy\n    dirs = []\n    for j in range(6):\n        ang = j*np.pi/6\n        proj = (np.float32(np.cos(ang))*ux+np.float32(np.sin(ang))*uy)/mag\n        dirs.append(np.minimum(1.6,pool((proj**2)**3*mag)/div))\n    h = np.stack(dirs,axis=1)\n    h = h-0.56*h.mean(axis=1,keepdims=True)\n    h = h-h.mean(axis=(1,2,3),keepdims=True)\n    p = np.stack((pool(gray(dx)),pool(gray(dy))),axis=1)/div[:,None]\n    # Coarser structure tolerates imprecise alignment of contours and limbs.\n    small = pool(x)\n    light = gray(small)\n    low = light - light.mean(axis=(1,2),keepdims=True)\n    band = light - blur(blur(light))\n    # Midrange colour self-similarity is unchanged when a differently coloured\n    # animal has the same arrangement of body, face and surrounding background.\n    z = np.pad(small,[(0,0),(0,0),(2,2),(2,2)],mode='edge')\n    comps = []\n    for hy,hx in ((0,2),(2,0),(1,1),(1,-1)):\n        diff = z[:,:,2-hy:10-hy,2-hx:10-hx]-z[:,:,2+hy:10+hy,2+hx:10+hx]\n        d = (np.mean(diff**2, axis=1)+0.00025)**0.36\n        comps.append(d-d.mean(axis=(1,2),keepdims=True))\n    sim = np.stack(comps,axis=1)\n    sal = salmap(small)\n    sal = np.minimum(sal, 0.5)\n    sal = sal-sal.mean(axis=(1,2),keepdims=True)\n    result = np.concatenate((normalize(h,.43,.18), normalize(p,.16,.32), normalize(band,.12,.25), normalize(low,.065,.50), normalize(sim,.155,.4), normalize(sal,.07,.30)),axis=1)\n    return result.astype(np.float32)\n\ndef region(x, ys, xs):\n    return x[:,ys,xs].mean(axis=(1,2))\n\ndef scene(x):\n    # Bounded evidence functions, not hard gates: a plane need not have sky\n    # behind it, nor is every patch of blue water or sky.\n    r,g,b=x[:,0],x[:,1],x[:,2]\n    lum=gray(x)\n    lo=np.minimum(np.minimum(r,g),b)\n    hi=np.maximum(np.maximum(r,g),b)\n    chroma=hi-lo\n    dx,dy=deriv(blur(lum))\n    fine=np.abs(lum-blur(lum))\n    gm=np.sqrt(dx*dx+dy*dy+0.000001)\n    flat=np.exp(-11*blur(gm)-15*blur(fine))\n    blue=saturate((b-r+0.018)/.145)*saturate((b-g+.025)/.080)*saturate((b-.16)/.30)\n    cool=saturate((b-r+.035)/.12)*saturate((g-r+.055)/.10)\n    green=saturate((g-b-.004)/.12)*saturate((g-r+.045)/.10)\n    brown=saturate((r-b-.025)/.18)*saturate((r-g+.012)/.12)*saturate((g-b+.03)/.13)\n    skin=saturate((r-b-.04)/.22)*saturate((r-g-.015)/.12)*saturate((g-b+.01)/.12)\n    bright=saturate((lum-.54)/.28)*saturate((.24-chroma)/.18)\n    grey=saturate((.18-chroma)/.14)\n    top=slice(0,8)\n    bot=slice(24,32)\n    allx=slice(0,32)\n    mid=slice(8,24)\n    def surroundings(a):\n        return .5*(region(a,slice(4,30),slice(0,5))+region(a,slice(4,30),slice(27,32)))\n    def border(a):\n        return .4*surroundings(a)+.30*region(a,top,allx)+.30*region(a,bot,allx)\n    skycol=blue*flat\n    skywhite=bright*flat\n    sky=region(skycol,top,allx)\n    white=region(skywhite,top,allx)\n    air=border((blue+.58*(1-blue)*bright)*flat)\n    smooth=border(flat)\n    lawn=region(green,bot,allx)\n    leaf=border(green)\n    woods=region(green,top,allx)\n    water=region(cool,bot,allx)\n    sea=region(blue,bot,allx)\n    grnd=region(brown,bot,allx)\n    warm=border(brown)\n    road=region(grey,bot,allx)\n    neutral=border(grey)\n    field=lawn*np.maximum(.0,1-woods)\n    centerbrown=region(brown,mid,mid)\n    centergreen=region(green,mid,mid)\n    warmbody=region(skin,mid,mid)\n    colored=region(saturate((chroma-.23)/.22)*(1-green)*(1-brown),mid,mid)\n    whitebody=region(bright,mid,mid)\n    topflat=region(flat,top,allx)\n    bottomflat=region(flat,bot,allx)\n    # Horizontal and vertical lines that persist over multiple pixels indicate\n    # the hull/deck of a ship or the windows, grille and chassis of a vehicle.\n    a=lum-blur(lum)\n    def line(ygrad,xgrad):\n        v=ygrad[:,:,3:-3]\n        p=(v+ygrad[:,:,:-6]+ygrad[:,:,6:])/3\n        q=(np.abs(v)+np.abs(ygrad[:,:,:-6])+np.abs(ygrad[:,:,6:]))/3\n        return np.maximum(0,np.abs(p)-0.52*q-.23*np.abs(xgrad[:,:,3:-3]))\n    hline=line(dy,dx)\n    vline=line(dx.transpose(0,2,1),dy.transpose(0,2,1)).transpose(0,2,1)\n    strength=gm.mean(axis=(1,2))+.016\n    h=(hline.mean(axis=(1,2))/strength-.15)*5\n    v=(vline.mean(axis=(1,2))/strength-.105)*6\n    h=np.clip(h,-.8,1.6)\n    v=np.clip(v,-.7,1.6)\n    wh=line(dy[:,23:,:],dx[:,23:,:]).mean(axis=(1,2))/(gm[:,23:,:].mean(axis=(1,2))+.012)\n    ripple=np.clip((wh-.14)*4,-.4,1.)\n    # Fine texture and the spatial distribution of detail separate grass, fur,\n    # smooth sky and water. Dark wheel wells and tires often form a low band.\n    texture=np.clip((border(fine)-.023)*17,-.4,.8)\n    groundlight=region(lum,bot,allx)\n    belly=region(lum,slice(20,25),slice(4,28))\n    under=np.clip((groundlight-belly)*3,-.6,1.)\n    horizon=np.clip((region(lum,top,allx)-groundlight)*2,-.7,1.)\n    # A pair of dark tires at the same height is more informative than dark\n    # pixels alone. Inner-disk, mid-ring and outer-road samples reject shadows.\n    wheels=np.zeros(len(x),np.float32)\n    wheelmap=blur(lum)\n    rimsrc=np.pad(wheelmap,[(0,0),(5,5),(5,5)],mode='edge')\n    for rad in (2,3,4):\n        out=[];tire=[]\n        for j in range(8):\n            phi=(j+.5)*np.pi/4\n            yy=int(round(np.sin(phi)*(rad+1.3)));xx=int(round(np.cos(phi)*(rad+1.3)))\n            out.append(rimsrc[:,5+yy:37+yy,5+xx:37+xx])\n            yy=int(round(np.sin(phi)*(rad*.71)));xx=int(round(np.cos(phi)*(rad*.71)))\n            tire.append(rimsrc[:,5+yy:37+yy,5+xx:37+xx])\n        radial=sum(tire)*.125\n        contrast=sum(np.clip(outer-inner,-.11,.27) for outer,inner in zip(out,tire))*.125\n        response=saturate((contrast+.40*np.maximum(wheelmap-radial,0)-.052)*7)*saturate((.54-radial)*4)\n        for gap in (9,14,19):\n            n=32-gap\n            for lift in (-2,0,2):\n                first=response[:,16:28,:n];second=response[:,16+lift:28+lift,gap:]\n                wheels=np.maximum(wheels,np.minimum(first,second).max(axis=(1,2)))\n    wheels-=.36\n    # A hull separates the water from a different surface above. Uniform blue\n    # all around a compact silhouette is better evidence of an object in flight.\n    topcool=region(cool,top,allx)\n    waterline=np.clip(water-topcool,-.35,.70)\n    tip=region(flat,mid,mid)\n    focal=np.clip(2*(smooth-tip),-.65,.95)\n    continuous=np.abs(region(lum,top,allx)-groundlight)\n    surroundsky=saturate(.74*air+.3*sky)*saturate(.98-2*continuous)\n    foliage=(1-topflat)*woods\n    solid=region(np.abs(lum-blur(blur(lum))),mid,mid)\n    furry=np.clip((solid-.050)*7,-.33,.60)\n    # Cues to a tightly framed furry face: roughly symmetrical dark eye pair,\n    # nose below it and bright cheeks. Evaluate across several eye spacings.\n    coarse=blur(lum)\n    eyes=coarse*0\n    for d in (2,3):\n        z=np.pad(coarse,[(0,0),(d,d),(d,d)],mode='edge')\n        ring=(z[:,:32, d:d+32]+z[:,2*d:,d:d+32]+z[:,d:d+32,:32]+z[:,d:d+32,2*d:])*.25\n        eyes=np.maximum(eyes, ring-coarse)\n    pair=np.zeros(len(x),np.float32)\n    for d in (3,5,7):\n        e1=eyes[:,7:19,10-d:22-d]\n        e2=eyes[:,7:19,10+d:22+d]\n        nose=eyes[:,7+d:19+d,10:22]\n        eye=(np.minimum(e1,e2)*.78 + .22*np.maximum(e1,e2))\n        pair=np.maximum(pair,(np.minimum(eye,nose+.04)+.35*eye).max(axis=(1,2)))\n    face=saturate((pair-.105)*4.5)\n    # Centred terms are hand-chosen neutral conditions for each soft predicate.\n    sky-=.15; white-=.15; air-=.19; smooth-=.49; water-=.27; sea-=.10\n    leaf-=.18; lawn-=.21; woods-=.16; grnd-=.23; warm-=.22; field-=.14\n    neutral-=.52; road-=.53; centerbrown-=.23; centergreen-=.15; warmbody-=.16\n    colored-=.10; whitebody-=.16; face-=.35\n    natural = 1.2*leaf+.43*warm-.65*h-.42*v-.38*water-.32*neutral\n    engineered = .75*h+.42*v+.53*road-.56*leaf-.36*warm\n    s0=2.9*air+1.55*sky+.95*smooth+.48*white+.4*sea-.38*ripple-.75*leaf-.92*warm-.55*v-.42*h-.34*texture\n    s1=engineered+.48*h+.66*under+.4*colored+.22*road-.64*sky-.42*sea-.2*horizon-.24*white\n    s2=.62*natural+1.0*smooth+.54*leaf+.26*sky+.72*colored-.45*under-.27*v-.12*face\n    s3=.45*natural+.67*neutral+.53*face+.57*texture-.64*sky-.40*lawn-.25*horizon-.24*under+.21*warm\n    s4=natural+.84*leaf+.65*lawn+.44*woods+.59*centerbrown-.43*sky-.57*colored-.32*face-.28*whitebody\n    s5=.40*natural+.54*face+.33*warmbody+.35*whitebody+.16*colored-.37*sea-.4*sky-.19*horizon\n    s6=.50*natural+1.2*warm+.88*grnd+.57*centergreen+.39*texture-.6*white-.80*sky-.42*horizon-.39*sea-.26*face\n    s7=.45*natural+.88*field+.69*centerbrown+.51*warmbody+.44*sky+.36*horizon-.43*sea-.22*face-.15*woods\n    s8=1.85*water+1.48*sea+.76*sky+.62*white+.89*ripple+.76*h+.24*smooth-.52*v-1.08*leaf-.84*warm-.55*under\n    s9=engineered+.76*v+.46*horizon+.49*sky+.38*white+.36*whitebody+.25*colored-.53*sea+.27*under\n    s0+=1.5*(surroundsky-.10)+.36*focal-.64*waterline-.28*wheels\n    s1+=1.13*wheels-.35*furry\n    s2+=.85*focal+.51*foliage+.22*(surroundsky-.1)-.42*wheels\n    s3+=.57*furry-.35*wheels\n    s4+=.64*foliage-.26*wheels\n    s5+=.32*furry-.22*wheels\n    s6+=.40*furry-.35*wheels\n    s7-=.21*wheels\n    s8+=.9*waterline-.76*(surroundsky-.10)-.50*wheels\n    s9+=.76*wheels-.20*furry\n    return np.stack((s0,s1,s2,s3,s4,s5,s6,s7,s8,s9),axis=1)\n\ndef predict(x):\n    labels=np.empty(len(x),np.int64)\n    for start in range(0,len(x),64):\n        a=x[start:start+64]\n        # Fixed matrix geometry also keeps BLAS kernel selection independent of\n        # the caller's batch size, even for the last incomplete batch.\n        d=describe(a)\n        block=np.zeros((64,d.shape[1]),np.float32)\n        block[:len(a)]=d\n        evidence=scene(a)\n        for k in range(10):\n            votes=(block @ CONSTANTS[str(k)].T)[:len(a)]\n            # Keep best support and a few nearby alternatives rather than an\n            # average prototype. A rare but distinctive viewpoint can win.\n            support=np.partition(votes,-7,axis=1)[:,-7:]\n            best=.66*support.max(axis=1)+.34*support.mean(axis=1)\n            evidence[:,k]+=20*best\n        labels[start:start+len(a)]=np.argmax(evidence,axis=1)\n    return labels\n"
exec(SOURCE)

# Turn every convex plate into triangles, then use an orthographic z-buffer.
# Rounded animal features are ellipsoids, intersected analytically with the ray.
def prepare(m):
    triangles=[];colors=[];centers=[];quadrics=[];ecols=[]
    for f in m.faces:
        if len(f)==2:
            v,c=f
            for j in range(1,len(v)-1):triangles.append((v[0],v[j],v[j+1]));colors.append(c)
        else:
            _,p,r,c,t=f;centers.append(p);ecols.append(c)
            rmat=np.array([[cos(t),0,-sin(t)],[0,1,0],[sin(t),0,cos(t)]])
            quadrics.append((rmat*(r*r))@rmat.T)
    return (np.asarray(triangles).reshape(-1,3,3),np.asarray(colors).reshape(-1,3),np.asarray(centers).reshape(-1,3),np.asarray(quadrics).reshape(-1,3,3),np.asarray(ecols).reshape(-1,3),m.light)

def gridbox(x0,x1,y0,y1):
    x0=np.clip(np.ceil(x0),0,80).astype(int);x1=np.clip(np.floor(x1)+1,0,80).astype(int)
    y0=np.clip(np.ceil(y0),0,80).astype(int);y1=np.clip(np.floor(y1)+1,0,80).astype(int)
    wid=np.maximum(0,x1-x0);n=wid*np.maximum(0,y1-y0)
    face=np.repeat(np.arange(len(n)),n);local=np.arange(len(face))-np.repeat(np.cumsum(n)-n,n)
    return face, x0[face]+local%wid[face], y0[face]+local//wid[face]

def render(model,az,elev,roll,target):
    tri,col,cent,Q,ecol,L=model
    normal=np.array([cos(az)*cos(elev),sin(az)*cos(elev),sin(elev)])
    side=np.array([-sin(az),cos(az),0]);up=np.cross(normal,side)
    T=np.stack((side*cos(roll)+up*sin(roll),side*sin(roll)-up*cos(roll),normal),axis=1)
    tr=tri@T;cent=cent@T;cov=T.T@Q@T
    rad=np.sqrt(np.maximum(cov.diagonal(axis1=1,axis2=2),0))[:,:2]
    bound=np.concatenate((tr.reshape(-1,3)[:,:2], cent[:,:2]+rad,cent[:,:2]-rad));lo=bound.min(axis=0);hi=bound.max(axis=0)
    scale=74/max(hi-lo);offset=39.5-(hi+lo)*.5*scale
    allpos=[];alldep=[];allcol=[]
    if len(tr):
        uv=tr[:,:,:2]*scale+offset
        low=uv.min(axis=1);high=uv.max(axis=1);f,x,y=gridbox(low[:,0],high[:,0],low[:,1],high[:,1]);
        a=uv[:,0];b=uv[:,1]-a;c=uv[:,2]-a;px=x-a[f,0];py=y-a[f,1];det=(b[:,0]*c[:,1]-c[:,0]*b[:,1]);den=np.where(np.abs(det)>1.e-12,det,1.e12)
        s=(px*c[f,1]-py*c[f,0])/den[f];t=(py*b[f,0]-px*b[f,1])/den[f];inside=(s>=0)&(t>=0)&(s+t<=1)&(np.abs(det[f])>1.e-12)
        f=f[inside];s=s[inside];t=t[inside];allpos.append((80*y+x)[inside]);alldep.append(tr[f,0,2]*(1-s-t)+tr[f,1,2]*s+tr[f,2,2]*t);allcol.append(col[f]);
    if len(cent):
        uv=cent[:,:2]*scale+offset;rr=rad*scale
        f,x,y=gridbox(uv[:,0]-rr[:,0],uv[:,0]+rr[:,0],uv[:,1]-rr[:,1],uv[:,1]+rr[:,1]);
        px=(x-uv[f,0])/scale;py=(y-uv[f,1])/scale
        # Invert 2x2 screen-space ellipse (solely an individual geometric primitive).
        a,b,c=cov[:,0,0],cov[:,0,1],cov[:,1,1];den=a*c-b*b
        ux=(c[f]*px-b[f]*py)/den[f];uy=(a[f]*py-b[f]*px)/den[f];radic=1-ux*px-uy*py;inside=radic>=0
        f=f[inside];ux=ux[inside];uy=uy[inside];zvar=np.maximum(1.e-12,cov[:,2,2]-(c*cov[:,0,2]**2+a*cov[:,1,2]**2-2*b*cov[:,0,2]*cov[:,1,2])/den)
        zmid=cent[f,2]+cov[f,0,2]*ux+cov[f,1,2]*uy
        dz=np.sqrt(radic[inside]*zvar[f]);nz=dz/zvar[f];sx=(c*cov[:,0,2]-b*cov[:,1,2])/den;sy=(a*cov[:,1,2]-b*cov[:,0,2])/den
        nx=ux-nz*sx[f];ny=uy-nz*sy[f];l=L@T;ndotl=(nx*l[0]+ny*l[1]+nz*l[2])/np.sqrt(nx*nx+ny*ny+nz*nz+1.e-15)
        allpos.append((80*y+x)[inside]);alldep.append(zmid+dz);allcol.append(ecol[f]*(.84+.17*ndotl[:,None]))
    pos=np.concatenate(allpos);depth=np.concatenate(alldep);colors=np.concatenate(allcol)
    front=np.full(6400,-1.e6);np.maximum.at(front,pos,depth);ok=depth>=front[pos];canvas=np.zeros((6400,4),np.float32);canvas[pos[ok],:3]=np.clip(colors[ok],0,1);canvas[pos[ok],3]=1.
    focus=(target@T)[:2]*scale+offset if target is not None else np.array([39.5,39.5]);return canvas.reshape(80,80,4), focus, scale

pix=(np.arange(64)+.5)/2-16; gridX=pix[None,:]; gridY=pix[:,None]

def bgcolour(u,opt):
    if opt=='sky':return rgb([.47,.65,.84])*u.uniform(.70,1.15)+u.uniform(-.05,.08)
    if opt=='water':return rgb([.28,.46,.58])*u.uniform(.56,1.45)
    if opt=='field':return rgb([( .55,.55,.30),(.38,.48,.27),(.27,.38,.19),(.48,.42,.32)][u.randint(4)])*u.uniform(.64,1.38)
    if opt=='earth':return rgb([.50,.42,.31])*u.uniform(.54,1.48)
    if opt=='silver':return rgb([.69,.71,.72])*u.uniform(.83,1.32)
    if opt=='room':return rgb([( .61,.58,.54),(.67,.57,.48),(.53,.48,.47),(.47,.53,.55)][u.randint(4)])*u.uniform(.38,1.51)
    if opt=='trees':return rgb([.31,.36,.255])*u.uniform(.48,1.55)
    return rgb(u.uniform(.23,.70))

def composite(k,u,can,focus,scale,portrait,frac,style):
    # frame is sized by object extent, except that a portrait is sized by the
    # physical size of the head, not by the animal's height including its legs.
    if portrait:
        span=scale*(u.uniform(.60,.95) if k in (3,5) else u.uniform(.61,1.0)); center=focus.copy();center[1]+=u.uniform(-.05,.12)*span
    else:span=74.;center=np.array([39.5,39.5]);
    jitter=u.normal(0,1.65,2).clip(-4.,4.); ratio=span/(32*frac)
    sx=center[0]+(gridX-jitter[0])*ratio;sy=center[1]+(gridY-jitter[1])*ratio
    ix=np.floor(sx).astype(int);iy=np.floor(sy).astype(int);xf=(sx-ix)[:,:,None];yf=(sy-iy)[:,:,None]; ix=np.clip(ix,0,78);iy=np.clip(iy,0,78)
    obj=((1-yf)*((1-xf)*can[iy,ix]+xf*can[iy,ix+1])+yf*((1-xf)*can[iy+1,ix]+xf*can[iy+1,ix+1])).astype(np.float32)
    if k==0:opts=['sky','sky','silver','silver','room']; opts2=['sky','sky','trees','silver','room']
    elif k==8:opts=['sky','silver','silver','trees','room'];opts2=['water','water','water','road','water']
    elif k in (1,9):opts=['silver','trees','room','sky','road'];opts2=['road','room','road','earth','road']
    elif k in (4,7):opts=['trees','field','earth','sky','silver'];opts2=['field','field','earth','field','trees']
    elif k==2:opts=['trees','sky','field','silver','earth'];opts2=['field','sky','earth','trees','room']
    elif k==6:opts=['earth','field','trees','room','earth'];opts2=['field','earth','field','earth','earth']
    else:opts=['room','room','trees','room','field'];opts2=['room','earth','field','room','field']
    i=u.randint(5);a=bgcolour(u,opts[i]);b=bgcolour(u,opts2[i]);
    if style==0 or k==0 and u.rand()<.62:
        b=a*u.uniform(.86,1.13)+u.uniform(-.1,.1)
    slope=u.normal(0,.064 if k in (0,1,8,9) else .22);hor=u.uniform(-8,7);soft=u.uniform(.4,2.6) if style!=0 else 8
    blend=np.clip((gridY-hor+slope*gridX)/soft+.5,0,1)[:,:,None];bg=a*(1-blend)+b*blend
    # Low-frequency background clutter and shading are per drawing. No global
    # measurements, either over rendered examples or over the external photos.
    for j in range(3):
        sz=u.uniform(4,17);sz2=u.uniform(3,19);x0,y0=u.uniform(-18,18,2)
        blob=np.exp(-((gridX-x0)/sz)**2-((gridY-y0)/sz2)**2);bg+=blob[:,:,None]*(u.uniform(-.11,.11)+u.uniform(-.022,.022,3))
    if style==2:
        # Structured context for the profile view: branches below perches,
        # reflected streaks below hulls and softly blurred outdoor scenery.
        if k in (2,4,6,7) or k in (3,5) and opts[i] in ('field','trees'):
            for t in range(4):
                xpos=u.uniform(-23,23);slope2=u.uniform(-.9,.9);w=u.uniform(.65,3)
                branch=np.exp(-((gridX-xpos-slope2*gridY)/w)**2)
                bg+=branch[:,:,None]*u.uniform(-.16,.05)
        if k==2 and not portrait and u.rand()<.57:
            perch=np.exp(-((gridY-u.uniform(6,11)*frac-u.uniform(-.27,.27)*gridX)/u.uniform(.6,1.8))**2)
            bg+=perch[:,:,None]*u.uniform(-.26,-.06)
        if k==8:
            for t in range(6):
                py=u.uniform(-2,18);px=u.uniform(-19,19);w=u.uniform(2,16)
                water=np.exp(-((gridY-py)/u.uniform(.25,1.2))**2-((gridX-px)/w)**2)
                bg+=water[:,:,None]*blend*u.uniform(-.13,.19)
    if k in (1,9) and not portrait:
        shade=np.exp(-((gridX-jitter[0])/(12*frac))**2-((gridY-jitter[1]-frac*8)/(2.2*frac))**2);bg*=1-.23*shade[:,:,None]
    out=bg*(1-obj[:,:,3:4])+obj[:,:,:3];out=np.clip(out,0,1).reshape(32,2,32,2,3).mean(axis=(1,3)).astype(np.float32).transpose(2,0,1)
    return out

def makebank(k):
    u=np.random.RandomState(13001+k*431);features=[];batch=[]
    def flush():
        features.append(describe(np.array(batch)));batch.clear()
    for i in range(144):
        m=Draw(u);focus=None;portrait=False;fly=False
        if k==0:plane(m,u)
        elif k==1:car(m,u)
        elif k==2:fly=u.rand()<.21;bird(m,u,fly)
        elif k in (3,5):portrait=u.rand()<(.57 if k==3 else .46);focus=dogcat(m,u,'cat' if k==3 else 'dog',portrait)
        elif k in (4,7):
            ungulate(m,u,'deer' if k==4 else 'horse');portrait=u.rand()<.14
            # head center relative to shoulders of the standing animal
            focus=np.array([.89 if k==4 else 1.,0,1.39])
        elif k==6:frog(m,u)
        elif k==8:ship(m,u)
        else:truck(m,u)
        model=prepare(m)
        for v in range(7):
            # Side views, front and rear three-quarter views and foreshortened
            # views have different informative edges. Include them explicitly.
            if portrait:az=u.uniform(0,1.35)*(1 if u.rand()<.5 else -1)
            else:
                az=(v+u.uniform(.1,.9))/7*pi
                if k in (2,3,4,5,6,7) and az>2.18:az=pi-az
                az*=u.choice([-1,1])
            if k==0 or fly:elev=u.uniform(-.73,.52);roll=u.uniform(-.57,.57);roll*=2 if k==0 and v==4 else 1.
            elif k in (1,8,9):elev=u.uniform(.018,.43 if k!=8 else .34);roll=u.normal(0,.047)
            elif k==6:elev=u.uniform(.045,1.18);roll=u.normal(0,.16)
            else:elev=u.uniform(-.05,.61 if k in (2,3,5) else .36);roll=u.normal(0,.085)
            can,fc,sc=render(model,az,elev,roll,focus)
            for j,fr in enumerate((.56,.78,.98,1.21)):
                frac=fr*u.uniform(.93,1.07)
                if k in (0,2,8) and j==0:frac*=.82
                draw=composite(k,u,can,fc,sc,portrait,frac,(j+i+v)%3);batch.extend((draw,draw[:,:,::-1]));
                if len(batch)==64:flush()
    if batch:flush()
    return np.concatenate(features)

if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True);(out/'solution.py').write_text(SOURCE)
    banks={}
    for k in range(10):
        banks[str(k)]=makebank(k)
        print('constructed class',k,flush=True)
    np.savez(out/'constants.npz',**banks)
