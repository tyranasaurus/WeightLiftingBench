"""Build a fixed illustration-matching and scene-cue network. No input files."""
import sys, json, math
from pathlib import Path
import numpy as np

# Model coordinates: x points toward the nose, y up, z toward the left side.
# Dimensions are arbitrary, relative to the object, not image measurements.
r=np.random.default_rng(61429)
def u(a,b): return r.uniform(a,b)
def chance(p): return r.random()<p
def pick(xs): return xs[r.integers(len(xs))]
def V(*xs): return np.array(xs,float)
def color(s): return np.array([int(s[i:i+2],16)/255 for i in (0,2,4)])
def palette(s): return color(pick(s.split()))
def tint(c,f): return np.clip(c*f,0,1)
def unit(x): return x/max(np.linalg.norm(x),1e-10)
class Model:
    def __init__(self): self.objs=[]
    def oval(self,c,rad,col,basis=None):
        A=np.diag(rad) if basis is None else basis @ np.diag(rad)
        self.objs.append(('e',np.array(c),A,np.array(col)))
    def limb(self,a,b,w,h,col,extent=1.15):
        a,b=np.array(a),np.array(b); d=unit(b-a); v=unit(np.cross(d,V(0,0,1) if abs(d[2])<.94 else V(1,0,0))); z=np.cross(d,v)
        self.oval((a+b)/2,[max(.01,np.linalg.norm(b-a)*extent/2),w,h],col,np.stack([d,v,z],1))
    def face(self,pts,col): self.objs.append(('f',np.array(pts),None,np.array(col)))
    def box(self,lo,hi,col):
        x,y,z=lo; X,Y,Z=hi
        self.prism([(x,y),(X,y),(X,Y),(x,Y)],z,Z,col)
    def prism(self,profile,z,Z,col):
        a=np.array([(x,y,z) for x,y in profile]); b=a.copy(); b[:,2]=Z
        self.face(a,col);self.face(b,col)
        for i in range(len(a)): self.face([a[i-1],b[i-1],b[i],a[i]],col)
    def rings(self,profiles,col):
        # Width and height of each transverse quadrilateral section.
        rings=[[(x,y,-w),(x,y,w),(x,Y,W),(x,Y,-W)] for x,y,Y,w,W in profiles]
        self.face(rings[0],col);self.face(rings[-1],col)
        for a,b in zip(rings,rings[1:]):
            for i in range(4):self.face([a[i-1],a[i],b[i],b[i-1]],col)

# Lightweight orthographic ray caster. Polygons are convex; ellipsoids have
# exact projected boundaries. One render supplies one view, not a class mean.
S=40
YY,XX=np.mgrid[:S,:S].astype(float)+.5

def render(m,yaw,pitch,roll,focus=None):
    e=V(np.sin(yaw),0,-np.cos(yaw)); f=V(-np.cos(yaw)*np.sin(pitch),np.cos(pitch),-np.sin(yaw)*np.sin(pitch)); d=np.cross(e,f)
    R=np.stack([e*np.cos(roll)+f*np.sin(roll), e*np.sin(roll)-f*np.cos(roll), d]); invR=R.T
    light=unit(V(u(-.55,.55),1,u(-.5,.5))+.40*d); L=R@light
    objs=[]; mi=V(1e5,1e5); ma=-mi
    for typ,c,A,col in m.objs:
        if typ=='e':
            q=R@c; T=R@A; cov=T@T.T
            span=np.sqrt(np.diag(cov)[:2]); lo=q[:2]-span; hi=q[:2]+span
            # inverse of the screen-plane 2x2 covariance; depth conditioned on xy
            a,b=cov[0,:2]; cc=cov[1,1]; det=max(a*cc-b*b,1e-12); ia,ib,ic=cc/det,-b/det,a/det
            g=cov[2,0]*ia+cov[2,1]*ib; h=cov[2,0]*ib+cov[2,1]*ic
            zz=np.sqrt(max(1e-8,cov[2,2]-g*cov[2,0]-h*cov[2,1])); pars=(ia,ib,ic,g,h,zz)
        else:
            q=c@invR; lo=q[:,:2].min(0);hi=q[:,:2].max(0);pars=None
        mi=np.minimum(mi,lo);ma=np.maximum(ma,hi); objs.append((typ,col,q,pars,lo,hi))
    if focus is None:
        scl=(S*.94)/max(ma-mi); off=(mi+ma)/2-(S/2)/scl
    else:
        point,span=focus;scl=S/span;off=(R@point)[:2]-(S/2)/scl
    Z=np.full((S,S),-1e5); img=np.zeros((S,S,3)); opacity=np.zeros((S,S));
    for typ,col,q,pars,lo,hi in objs:
        l,t=np.maximum(0,np.floor((lo-off)*scl).astype(int)); rr,bb=np.minimum(S,np.ceil((hi-off)*scl).astype(int));
        if rr<=l or bb<=t:continue
        sl=np.s_[t:bb,l:rr]; x=XX[sl]/scl+off[0]; y=YY[sl]/scl+off[1]
        if typ=='e':
            ia,ib,ic,g,h,zz=pars; x=x-q[0];y=y-q[1]; dx=ia*x+ib*y;dy=ib*x+ic*y
            v=1-x*dx-y*dy; ok=v>0; rt=np.sqrt(np.maximum(v,.001)); depth=q[2]+g*x+h*y+zz*rt
            nx=zz*dx/rt-g;ny=zz*dy/rt-h
            lighting=.78+.24*(nx*L[0]+ny*L[1]+L[2])/np.sqrt(nx*nx+ny*ny+1)
        else:
            normal=np.cross(q[1]-q[0],q[2]-q[0]);
            if abs(normal[2])<1e-8:continue
            sign=np.sign(normal[2]); ok=np.ones(x.shape,bool)
            for a,b in zip(q,np.roll(q,-1,axis=0)):
                ok &= ((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]))*sign>=-1e-9
            depth=q[0,2]-(normal[0]*(x-q[0,0])+normal[1]*(y-q[0,1]))/normal[2];lighting=.80+.21*np.dot(unit(normal)*sign,L)
        ok &= depth>=Z[sl]-1e-6
        Z[sl][ok]=depth[ok];opacity[sl][ok]=1
        if np.ndim(lighting): img[sl][ok]=lighting[ok,None]*col
        else:img[sl][ok]=lighting*col
    return np.clip(img,0,1),opacity

def wheels(m,axles,rad,width):
    dark=palette('1d2022 302e2f 202020'); rim=palette('9fa1a2 656362 c4c6c5')
    for x in axles:
        for s in [-1,1]:
            m.oval((x,rad,s*width),(rad,rad,.15),dark)
            m.oval((x,rad,s*(width+.145)),(rad*.55,rad*.55,.025),rim)

def vehicle(truck):
    m=Model(); paint=palette('dddcd8 edeeed acaeb0 8b9294 3d4653 33343a 69768c 446d9b 943329 b34537 b5a999 63817a'); glass=palette('333e47 505961 526b78 586879 849aaa 384651'); chrome=palette('81888b a6a7a4 535354'); black=color('27292c')
    if not truck:
        h=u(.79,1.10);w=u(.77,.93); roof=u(1.35,1.72);back=pick(['sedan','sedan','van','hatch']);rear={'sedan':-1.12,'van':-1.93,'hatch':-1.80}[back]; rr=rear+u(.14,.52); front=u(.95,1.35); pillar=u(.30,.64)
        m.rings([(-2.05,.34,h*.88,w*.92,w*.94),(-1.75,.32,h,w,w),(1.45,.32,h*.98,w,w),(2.12,.39,h*.77,w*.88,w*.92)],paint)
        m.rings([(rear,h-.03,h+.03,w*.94,w*.93),(rr,h-.025,roof,w*.98,w*.82),(pillar,h-.025,roof,w*.98,w*.80),(front,h-.025,h+.015,w*.97,w*.95)],paint)
        for s in [-1,1]:
            m.face([(pillar+.035,roof-.035,s*w*.81),(front-.07,h+.06,s*w*.96),(rear+.09,h+.06,s*w*.97),(rr+.045,roof-.04,s*w*.83)],glass)
            m.face([(-.38,h+.04,s*w*.98),(-.24,h+.04,s*w*.98),(-.28,roof-.02,s*w*.84),(-.41,roof-.02,s*w*.84)],black)
            m.oval((front-.14,h+.11,s*(w+.085)),(.15,.095,.12),paint)
        for x,X,ww,WW in [(pillar+.04,front-.045,.78,.87),(rr-.02,rear+.06,.79,.87)]:
            m.face([(x,roof-.018,-w*ww),(x,roof-.018,w*ww),(X,h+.09,w*WW),(X,h+.09,-w*WW)],glass)
        m.box((1.99,.43,-w*.58),(2.13,h*.74,w*.58),black)
        for s in [-1,1]:
            m.oval((2.1,h*.76,s*w*.68),(.04,.085,.19),color('e5dfce'));m.oval((-2.04,h*.76,s*w*.72),(.04,.085,.14),color('b94640'))
        wheels(m,[-1.28,1.32],u(.34,.42),w*.94)
    else:
        paint=palette('d5d4d0 f1efdf b34136 a53224 5079a2 2f4b6b dba845 407b6c');w=.87; cargo=palette('dcdcd5 b8b8ad edede9 a9b5bc 93938a ad5749 577997 c2a46a');typ=pick(['box','box','semi','semi','dump','tanker','delivery']);back=-u(2.4,3.9);top=u(2.25,3.05); nose=u(1.8,2.5); long=chance(.40); cabtop=top*u(.76,.99)
        m.box((back,.37,-.70),(nose,.71,.70),black)
        if typ=='tanker':m.oval(((back+.22)/2,1.72,0),((.25-back)/2,.9,.96),cargo)
        else:
            lower=1.03 if typ=='semi' else .77
            m.box((back,lower,-.99),(.22,top if typ!='dump' else top*.79,.99),cargo)
            for s in [-1,1]:m.box((back,lower,s*.995-.016),(.22,lower+.095,s*.995+.016),chrome)
        mid=nose-(.85 if long else .14)
        m.prism([(.38,.60),(nose,.60),(nose,1.38),(mid+.07,1.52),(mid-.14,cabtop),(.39,cabtop)],-w,w,paint)
        for s in [-1,1]:m.face([(.53,1.51,s*(w+.003)),(mid-.02,1.51,s*(w+.003)),(mid-.22,cabtop-.12,s*(w+.003)),(.53,cabtop-.12,s*(w+.003))],glass)
        m.face([(mid+.045,1.65,-.77),(mid+.045,1.65,.77),(mid-.085,cabtop-.09,.77),(mid-.085,cabtop-.09,-.77)],glass)
        m.box((nose-.01,.72,-.60),(nose+.035,1.29,.6),chrome);m.box((nose,.60,-w),(nose+.10,.75,w),chrome)
        for s in [-1,1]: m.oval((nose+.02,1.15,s*.72),(.05,.10,.13),color('e4dfce'))
        wheels(m,([back+.57,back+1.29,nose-.45] if typ in ('semi','tanker','dump') else [back+.70,nose-.45]),.405,w)
    return m

def airplane():
    m=Model();c=palette('e3e4e4 f0eee6 c2c7c7 a0acb7 b9c9cd 858884 687580 4d555d'); accent=palette('b3453e 3d688b 344973 969ba0 b8bfc1 e1ded2'); w=u(.19,.32); l=u(1.9,2.45); wing=u(1.6,2.55); prop=chance(.24)
    m.oval((0,0,0),(l,w,w),c); m.oval((l*.69,w*.51,0),(l*.185,.52*w,.72*w),tint(accent,.75))
    for s in [-1,1]:
        sweep=u(.02,.12) if prop else .37
        m.face([(.56,.03,0),(.56-wing*sweep,.04,s*wing),(.08-wing*sweep,.045,s*wing),(-.65,.00,0)],tint(c,.88))
        m.face([(-l*.62,.03,0),(-l*.76,.09,s*.87),(-l*.96,.085,s*.82),(-l*.95,.01,0)],c)
        if not prop:m.oval((.24,-.16,s*.85),(.39,.13,.13),tint(c,.86));m.oval((.55,-.16,s*.85),(.07,.10,.10),color('49545a'))
        m.oval((.22,w*.21,s*w*.96),(l*.535,.026,.022),accent)
    m.face([(-l*.54,.05,0),(-l*.84,.81,0),(-l*.96,.83,0),(-l*.96,.04,0)],accent)
    return m

def ship():
    m=Model(); typ=pick(['freight','freight','ferry','liner','sail','motor','warship','fishing']); hull=palette('343e44 35353a 525a64 8b969e b5b7b5 dddcd2 703b35'); deck=palette('e5e0d6 f1f0eb bfc5c2 a9aca9 a4aea9');glass=palette('485c69 364044 677d86');w=u(.52,.94);H=u(.59,.93)
    m.rings([(-2.4,.18,H*.95,w*.55,w*.8),(-1.85,.02,H,w*.87,w),(.98,0,H,w*.95,w),(1.84,.07,H*.99,w*.70,w*.84),(2.49,H*.62,H*1.13,.01,.025)],hull)
    if typ=='sail':
        h=u(2.7,4.1);m.limb((0,H,0),(0,h+.05,0),.035,.035,glass,1.01)
        m.face([(-.10,h,0),(-.10,H+.43,0),(-1.8,H+.52,.10)],deck);m.face([(.055,h-.36,0),(1.89,H+.14,0),(.06,H+.39,.08)],tint(deck,.90))
    else:
        if typ=='freight':x=pick([-1.25,.90]);half=.42;n=3
        elif typ=='motor':x=-.33;half=.95;n=1
        elif typ in ('liner','ferry'):x=-.12;half=1.70;n=r.integers(2,5)
        else:x=.01;half=.7;n=2
        for j in range(n):
            ww=w*(.82-.10*j);hh=half*(1-.10*j);y=H+j*.24
            m.box((x-hh,y,-ww),(x+hh,y+.245,ww),deck)
            for s in [-1,1]:m.box((x-hh*.92,y+.085,s*ww-.007),(x+hh*.92,y+.16,s*ww+.007),glass)
        if typ=='freight' and chance(.7):
            for i in range(4):
                xx=-1.65+i*.91
                if abs(xx-x)>.6:m.box((xx-.37,H,-w*.76),(xx+.40,H+u(.23,.70),w*.76),palette('a3604b 8c7860 6e818e 875e4d'))
        m.box((x-.25,H+.24*n,-.17),(x+.08,H+.24*n+.36,.17),tint(hull,.87))
        m.limb((x+.18,H+.24*n,0),(x+.21,H+.24*n+u(.35,.93),0),.032,.032,glass)
    return m

def fur(k):
    return palette({3:'8d8172 7b7b77 c2a17a b68b58 d1c3a5 e9e2d5 474947 95897f',4:'ac8b62 a28663 8e785f 795f49 9d714f b8a082',5:'ccb389 d6c8b3 645245 8a6b51 565455 b39471 e1dcd2 ac845a 453f37',7:'a16b49 925f3b 694c3d 654d3d 493d37 9d785b b59d7f d4d2c5'}[k])

def quadruped(k,mode):
    cat=k==3; dog=k==5; deer=k==4; horse=k==7; tall=horse or deer
    m=Model(); coat=fur(k); shade=tint(coat,u(.60,.83)); cream=tint(coat,u(1.14,1.40)); eye=palette('241e19 292b25 282321');nose=palette('393330 453930 5b4840'); stripe=tint(coat,.58)
    # Limbs are chains; raised hock, knee and hoof positions can move separately.
    y=1.57 if tall else (.84 if dog else .58); body=1.06 if tall else (1.00 if dog else .84); depth=(.52 if tall else .43)*u(.84,1.17);w=(.36 if tall else .34)*u(.83,1.1);sit=not tall and mode=='sit';
    if sit:y*=1.17
    m.oval((0,y,0),(body,depth,w),coat)
    if dog and chance(.32):m.oval((-.43,y+.04,w*.71),(.42,.30,.13),shade);m.oval((-.43,y+.04,-w*.71),(.42,.30,.13),shade)
    if cat and chance(.58):
        for s in [-1,1]:
            for i in range(4):m.limb((-.57+.25*i,y+.27,s*w*.65),(-.47+.25*i,y+.03,s*w*.99),.047,.035,shade)
    for s in [-1,1]:
        for front in [-1,1]:
            a=front*body*.67;shift=u(-.34,.34)*(1 if tall else .58);knee=a+(-.12 if front==-1 else .04)+.4*shift; foot=a+shift
            if sit and front==-1:knee=a+.33;foot=knee+.08
            z=s*w*.64; th=.175 if (horse or dog) else .14; footy=.08
            m.limb((a,y+.08,z),(knee,y*.48,z*1.08),th,th*.94,coat)
            hock=knee+(-.17 if front<0 else -.055)
            m.limb((knee,y*.50,z*1.08),(hock,y*.27,z*1.1),th*.63,th*.62,coat)
            m.limb((hock,y*.30,z*1.1),(foot,footy,z*1.14),th*(.41 if tall else .66),th*.58,coat)
            m.oval((foot+.044,footy,z*1.14),(.117,.082,th*.65),shade if tall else coat)
    if horse:
        m.limb((-.90,y+.28,0),(-1.23,y-.84,0),.15,.13,tint(shade,.72));m.limb((-1.19,y-.50,0),(-1.31,y-1.04,.015),.11,.13,tint(shade,.72))
    elif deer:m.oval((-1.01,y+.08,0),(.16,.15,.11),cream)
    else:
        a=(-body*.87,y+.14,0);b=(-body-u(.19,.48),y+u(-.18,.61),.06);c=(b[0]-.12,b[1]+u(-.44,.52),.19);m.limb(a,b,.10,.10,coat);m.limb(b,c,.08,.08,coat)
    hy=y+(.82 if tall else .43); hx=body*(1.02 if tall else .88)
    if tall and mode=='graze':hy=y-.58;hx=1.51
    if not tall and sit:hy+=.27;hx-=.23
    m.limb((body*.52,y+.14,0),(hx,hy+.08,0),.29 if tall else .255,w*.78,coat)
    if horse:m.limb((body*.45,y+.37,0),(hx-.065,hy+.31,0),.10,.12,tint(shade,.75))
    # Head primitives are in their own frame so the animal can look at camera.
    head=Model();
    if tall:
        hd=.245 if deer else .28
        head.oval((0,0,0),(.32,.30,hd),coat);head.limb((.05,-.01,0),(.49,-.30,0),.21,hd*.83,coat);head.oval((.49,-.29,0),(.19,.15,.19),shade if horse else coat);head.oval((.61,-.28,0),(.095,.098,.135),nose)
        for s in [-1,1]:
            head.limb((-.15,.18,s*.15),(-.21,.52,s*(.35 if deer else .22)),.105,.095,coat);head.oval((.13,.09,s*(hd*.91)),(.069,.064,.041),eye)
            if deer and mode=='antlers':
                for a,b in [((-.08,.20,.14),(-.20,.61,.28)),((-.20,.61,.28),(-.47,.87,.39)),((-.22,.64,.31),(-.10,.91,.34)),((-.46,.85,.38),(-.43,1.07,.45))]:
                    head.limb((a[0],a[1],s*a[2]),(b[0],b[1],s*b[2]),.032,.032,shade)
    else:
        h=u(.34,.44) if dog else u(.325,.38); hw=h*u(.95,1.20); hd=h*(.93 if dog else .85);head.oval((0,0,0),(hd,h,hw),coat)
        floppy=dog and chance(.69); sn=h*(u(.56,1.05) if dog else .42); mask=chance(.38)
        for s in [-1,1]:
            if floppy:head.limb((-.055,h*.71,s*hw*.84),(-.065,-h*.62,s*hw*1.10),h*.43,h*.27,shade)
            else:
                tip=(-hd*.48,h*u(1.23,1.72),s*hw*u(.63,1.04));a=(hd*.31,h*.43,s*hw*.89);b=(hd*.23,h*.72,s*hw*.22);back=(-hd*.6,h*.42,s*hw*.68)
                for poly in [[tip,a,b],[tip,b,back],[tip,back,a]]:head.face(poly,coat)
                def mid(a,b,t):return tuple(np.array(a)*(1-t)+np.array(b)*t)
                head.face([mid(tip,a,.22),mid(a,b,.18),mid(b,tip,.14)],tint(coat,.75))
            if mask:head.oval((hd*.71,h*.18,s*hw*.52),(h*.17,h*.33,h*.29),shade)
            ex=hd*.83;ey=h*.20;ez=hw*.52
            head.oval((ex,ey,s*ez),(h*.13,h*(.145 if dog else .128),h*.17),eye)
            if cat:head.oval((ex+h*.06,ey,s*(ez+.014)),(h*.089,h*.079,h*.125),palette('8e9967 b4a271'))
            head.oval((ex+h*.145,ey+.01,s*ez),(.014,.024,.016),eye)
            head.oval((hd*.88+sn*.24,-h*.24,s*hw*.205),(sn*.82,h*.27,hw*.34),cream if not dog or chance(.5) else coat)
        head.oval((hd*.9+sn*.93,-h*.125,0),(h*(.23 if dog else .14),h*(.17 if dog else .10),h*(.30 if dog else .16)),nose if dog else tint(nose,1.45))
        if cat and chance(.6):
            for s in [-1,1]:
                head.limb((hd*.36,.80*h,s*hw*.24),(hd*.81,.42*h,s*hw*.20),.03,.026,shade)
                head.limb((hd*.44,.08*h,s*hw*.85),(hd*.83,-.01*h,s*hw*.58),.024,.024,shade)
    turn=u(-.52,.63); c=np.cos(turn);s=np.sin(turn);T=np.array([[c,0,-s],[0,1,0],[s,0,c]]);t=V(hx,hy,0)
    if mode=='portrait':
        # Bust, not a disembodied head; framing is further cropped at composition.
        m=Model();m.oval((hx-.21,hy-.49,0),(.47,.55,.405),coat)
    m.head=t
    for typ,p,A,col in head.objs:m.objs.append((typ,p@T.T+t,None if A is None else T@A,col))
    return m

def frog():
    m=Model();c=palette('7f8051 64763b 78633e 968954 678442 555b3c 847562 93a953 847044');cream=tint(c,1.41);spots=tint(c,.64);m.oval((-.20,.49,0),(.84,.47,.54),c)
    m.oval((.59,.57,0),(.66,.35,.58),cream);m.oval((.66,.65,0),(.68,.32,.59),c)
    for s in [-1,1]:
        m.limb((-.59,.40,s*.34),(.06,.28,s*.78),.24,.21,c);m.limb((.05,.26,s*.79),(-.87,.14,s*.71),.105,.14,c)
        m.limb((.47,.47,s*.42),(.16,.26,s*.76),.124,.12,c);m.limb((.17,.26,s*.76),(.70,.10,s*.78),.09,.095,c)
        for j in range(3):m.limb((.69,.10,s*.79),(.90+.07*j,.065,s*(.56+j*.21)),.031,.034,c)
        m.oval((.65,.86,s*.39),(.23,.25,.19),c);m.oval((.77,.915,s*.49),(.15,.144,.112),color('ac9862'));m.oval((.803,.925,s*.565),(.114,.055,.056),color('2b2924'))
        m.limb((1.28,.60,s*.18),(.67,.54,s*.56),.021,.019,spots)
        for j in range(7):
            x=u(-.76,.22);z=u(.10,.41)*s;y=.49+.47*np.sqrt(max(.04,1-((x+.2)/.84)**2-(z/.54)**2));m.oval((x,y,z),(u(.045,.11),.037,u(.035,.078)),spots)
    return m

def bird(flying):
    m=Model();c=palette('736957 454744 6f8178 a48554 b7ae96 e2e3de b69e78 4b667b 979691 a18160');wing=tint(c,u(.50,.81));belly=tint(c,u(1.12,1.48));long=chance(.20);y=.85
    m.limb((-.71,y-.08,0),(.44,y+.09,0),.45,.37,c);m.oval((.08,y-.07,0),(.51,.36,.33),belly)
    heady=y+(.32 if flying else .54)+(.31 if long else 0);hx=.55 if flying else .42
    m.limb((.33,y+.12,0),(hx,heady,0),.22,.22,c);m.oval((hx,heady,0),(.32,.32,.295),c)
    beak=u(.29,.65) if long else u(.15,.34);bcol=palette('b69b59 625848 4d4941 c2a56a'); base=(hx+.26,heady-.016,0);tip=(hx+.30+beak,heady-.037,0)
    for s in [-1,1]:
        a=(base[0],heady+.066,0);b=(base[0],heady-.079,0);side=(base[0],heady-.016,s*.093);m.face([a,tip,side],bcol);m.face([b,tip,side],tint(bcol,.81));m.oval((hx+.118,heady+.077,s*.262),(.056,.054,.045),color('272822'))
        if flying:
            up=u(-.8,1.3);m.face([(.33,y+.05,s*.19),(.08,y+.21+up*.3,s*.91),(-.35,y+.26+up,s*2.08),(-.69,y+.20+up,s*2.28),(-.64,y+.12+up*.26,s*.90),(-.51,y,s*.20)],wing)
        else:
            m.limb((-.81,y-.16,s*.19),(.32,y+.10,s*.28),.26,.15,wing);m.limb((-.15,y-.27,s*.11),(-.14,y-.71,s*.15),.035,.032,bcol);m.limb((-.27,y-.70,s*.15),(.045,y-.72,s*.16),.029,.030,bcol)
    m.face([(-.37,y-.08,-.18),(-1.28,y-.24,-.23),(-1.36,y-.23,.23),(-.37,y-.08,.18)],wing);return m

# Descriptors of the drawing and of the input live in the same coordinates.
# The drawing routine only ever normalizes the current individual view.
axis=np.arange(4)-1.5; v0,v1=np.meshgrid(axis,axis);gg=np.exp(-(v0*v0+v1*v1)/(2*1.04**2))
ang=np.arange(4)*np.pi/4; edge=np.stack([gg*(v0*np.cos(a)+v1*np.sin(a)) for a in ang]);edge/=np.abs(edge).sum((1,2),keepdims=True)/2
lum=V(.30,.57,.13); opponent=np.array([[1.,0,-1],[-.5,1,-.5]])
feature_width=7*8*8+3*16

def normalized(x):return (x-x.mean(axis=(-2,-1),keepdims=True))/(x.std(axis=(-2,-1),keepdims=True)+1.e-6)
def describe(im):
    # area antialiasing has already been applied. Edge measurements every 2px.
    gray=im@lum; padded=np.pad(gray,1,mode='edge');win=np.lib.stride_tricks.sliding_window_view(padded,(4,4))[::2,::2]; ed=np.abs(np.einsum('xyij,kij->kxy',win,edge));ed=ed.reshape(4,8,2,8,2).mean((2,4));low=im.reshape(8,4,8,4,3).mean((1,3))
    def norm1(x):
        x=x-x.mean(axis=(-2,-1),keepdims=True);return x/max(.00001,np.sqrt((x*x).sum()))
    g=norm1((low@lum)[None]); o=norm1(ed); chrom=(low@opponent.T).transpose(2,0,1);chrom-=chrom.mean((1,2),keepdims=True);chrom=chrom/max(.24,np.linalg.norm(chrom));small=im.reshape(4,8,4,8,3).mean((1,3));C=(small@opponent.T).transpose(2,0,1);sat=np.linalg.norm(C,axis=0,keepdims=True);C=np.concatenate([C,sat]);C=C/max(.20,np.linalg.norm(C));return np.concatenate([g.ravel(),o.ravel(),chrom.ravel(),C.ravel()])

def illumination(c,level,s):
    m=c@lum;return np.clip(level+s*(c-m),.015,.98)
skies='9fbad6 81acd0 b6c6d8 cdcdcc e3e2df 719ec4 b8cddc d4d5d6'; grass='657448 738448 868659 67704a 8e8e64 4f6241 72814a 7e784f 484f36 9b946a'; woods='6f7050 504c35 524d40 72755b 887c65 695e49 485447 576247'; water='647f93 7e949e 92a7b5 707f85 597586 668494 9ca7aa';soil='817157 73634f 928069 a38b6b 645c50'; indoor='b3a494 878075 86756d 6b6760 c1bab2 9d9f99 60594f b09a87 6c7776'
# Background colour and horizon priors, never sampled from photographs.
def setting(k,mode,scale,cy):
    sky=palette(skies); gr=palette(grass);street=palette('878580 777772 989690 6d6f70 aba8a4 6a6864');building=palette('9b9691 7f7975 b2a69c a4a19b 66706b 606259')
    if k==0:
        if mode=='air': top=sky;bottom=sky;h=1.4
        else:top=sky if chance(.85) else building;bottom=street;h=u(.43,.70)
    elif k==8:top=sky if chance(.83) else palette(woods);bottom=palette(water);h=u(.31,.66)
    elif k in (1,9):
        top=pick([building,sky,gr,palette(woods),building] if k==1 else [sky,sky,building,palette(woods)]); bottom=street;h=u(.39,.74)
    elif k in (3,5):
        if chance(.66 if k==3 else .42):top=palette(indoor);bottom=palette(indoor) if chance(.50) else top;h=u(.38,.71)
        else:top=gr if chance(.57) else palette(woods);bottom=gr if chance(.77) else palette(soil);h=u(.20,.52)
    elif k in (4,7):top=pick([sky,gr,palette(woods),gr] if k==4 else [sky,sky,gr,palette(woods)]);bottom=gr if chance(.75) else palette(soil);h=u(.17,.54)
    elif k==2:
        top=pick([sky,gr,palette(woods)]);bottom=top if chance(.65) else pick([gr,palette(woods),palette(soil)]);h=u(.08,.63)
    else:
        top=pick([gr,palette(soil),palette(woods)]);bottom=top if chance(.5) else pick([gr,palette(soil),palette(woods)]);h=u(.24,.66)
    gain=u(.75,1.15); sat=u(.65,1.3);return illumination(top,(top@lum)*gain,sat),illumination(bottom,(bottom@lum)*gain,sat),h

def model(k):
    if k==0:mode='air' if chance(.69) else 'runway';m=airplane();pitch=u(-.80,.45) if mode=='air' else u(-.04,.20);yaw=u(.07,1.85);roll=u(-.49,.49) if mode=='air' else u(-.075,.075)
    elif k in (1,9):mode='road';m=vehicle(k==9);pitch=u(.045,.42) if k==1 else u(-.06,.25);yaw=u(.04,1.94);roll=u(-.058,.058)
    elif k==8:mode='afloat';m=ship();pitch=u(-.03,.32);yaw=u(.12,1.86);roll=u(-.11,.11)
    elif k==6:mode='low';m=frog();pitch=u(.10,1.12);yaw=u(.00,1.97);roll=u(-.32,.32)
    elif k==2:mode='flight' if chance(.235) else 'perch';m=bird(mode=='flight');yaw=u(0,2.35);pitch=u(-.73,.21) if mode=='flight' else u(-.22,.43);roll=u(-.36,.36)
    else:
        mode=pick(({3:['portrait','portrait','portrait','stand','sit','sit'],5:['portrait','portrait','stand','stand','sit'],4:['stand','stand','stand','antlers','graze','portrait'],7:['stand','stand','stand','stand','graze','portrait']}[k]));m=quadruped(k,mode);yaw=u(0,.81) if mode=='portrait' else u(.17,1.95);pitch=u(-.09,.46) if k in (3,5) else u(-.07,.20);roll=u(-.18,.18) if k in (3,5) else u(-.10,.10)
    return m,yaw,pitch,roll,mode

canvas=(np.arange(64)+.5)/2;gy,gx=np.meshgrid(canvas,canvas,indexing='ij'); gx=gx.ravel();gy=gy.ravel()
def compose(img,alpha,k,mode,rep):
    if mode=='portrait':scale=u(.89,1.57);cy=u(.53,.68)
    elif k==6:scale=u(.72,1.25);cy=u(.45,.63)
    elif k==2:scale=u(.47,1.11);cy=u(.43,.61)
    elif k==8:scale=u(.58,1.12);cy=u(.43,.61)
    elif k==1:scale=u(.84,1.26);cy=u(.48,.60)
    elif k==9:scale=u(.77,1.23);cy=u(.46,.60)
    else:scale=u(.64,1.16);cy=u(.43,.61)
    top,bt,h=setting(k,mode,scale,cy);cx=u(.435,.565);a=(gx-cx*32)/(32*scale)*S+S/2-.5;b=(gy-cy*32)/(32*scale)*S+S/2-.5
    a=np.clip(a,0,S-1.00001);b=np.clip(b,0,S-1.00001);x=a.astype(int);y=b.astype(int);fx=a-x;fy=b-y
    bg=np.clip(.5+(gy/32-h)*u(3,14),0,1)[:,None]*(bt-top)+top
    # Soft low-frequency light, not noise or features estimated from other views.
    bg=np.clip(bg+u(-.10,.085)*(gy[:,None]/32-.5)+u(-.075,.075)*(gx[:,None]/32-.5),0,1)
    gain=u(.78,1.25);out=np.zeros((len(a),3)); op=np.zeros(len(a));
    for dx,wx in [(0,1-fx),(1,fx)]:
        for dy,wy in [(0,1-fy),(1,fy)]:
            wt=wx*wy;out+=img[y+dy,x+dx]*wt[:,None];op+=alpha[y+dy,x+dx]*wt
    out=out*gain+bg*(1-op[:,None]);return np.clip(out.reshape(32,2,32,2,3).mean((1,3)),0,1)

layers=[];weights={}
def layer(name,op,inp,**kw):
    layers.append(dict(name=name,op=op,inputs=[inp] if isinstance(inp,str) else inp,**kw));return name
def aff(name,inp,W,b=None,conv=False,stride=1,pad=0):
    W=np.asarray(W,np.float32); weights[name+'.W']=W; weights[name+'.b']=np.zeros(len(W),np.float32) if b is None else np.asarray(b,np.float32)
    return layer(name,'conv' if conv else 'dense',inp,out=len(W),**(dict(kernel=list(W.shape[-2:]),stride=stride,padding=pad) if conv else {}))
def pool(name,inp,k,s=None,kind='avg'):
    k=[k,k] if isinstance(k,int) else list(k); s=k if s is None else ([s,s] if isinstance(s,int) else list(s));return layer(name,kind+'pool',inp,size=k,stride=s)
def conv1(name,inp,W,b=None):return aff(name,inp,np.asarray(W)[:,:,None,None],b,True)
def reshape(name,inp,*shape):return layer(name,'reshape',inp,shape=list(shape))
def plus(name,inp):return layer(name,'add',inp)
def relu(name,inp):return layer(name,'relu',inp)
def norm(name,inp):return layer(name,'norm',inp)

# The real image front end. Remove DC before the derivative bank so padding
# cannot produce a spurious high-energy rectangular outline on bright images.
gray=conv1('luminance','image',[lum]);g0=norm('adjust_contrast',gray); df=np.concatenate([edge,-edge]);ed0=aff('oriented_differences',g0,df[:,None],conv=True,stride=2,pad=1);ep=relu('half_waves',ed0);en=conv1('edge_magnitude',ep,np.tile(np.eye(4),(1,2))); en8=pool('cell_energy',en,2); ho=norm('relative_cell_energy',en8)
low=pool('coarse_RGB','image',4); lo=norm('intensity_layout',conv1('gray_8',low,[lum]));ch=conv1('color_opponents',low,opponent);chrom=norm('relative_chroma',ch)
# Relative chroma is less reliable than relative intensity. Keep an absolute
# opponent channel as well, in physical RGB units and with a soft saturation map.
small=pool('scene_grid',low,2); cp=conv1('scene_opponents',small,opponent); sat=conv1('split_opponents',small,np.concatenate([opponent,-opponent]));sat=conv1('colorfulness',relu('opponent_magnitudes',sat),[[.85,.85,.85,.85]])
# Assemble blocks with disjoint row routing (the instruction set has no concat).
def concat(name,parts):
    total=sum(d for _,d,_ in parts);i=0;sources=[]
    for j,(inp,d,g) in enumerate(parts):
        W=np.zeros((total,d)); W[i:i+d]=g*np.eye(d);i+=d;sources.append(aff(name+'_route'+str(j),inp,W))
    return plus(name,sources)
feat=concat('scene_and_shape',[(lo,64,.34/8),(ho,256,.66/16),(chrom,128,.105/np.sqrt(128)),(cp,32,.11/(.17*4)),(sat,16,.11/(.17*4))])

# Without fitting or looking at any photo: equal-sized groups of views
# per class. Templates remain individual rows. Pool responses, NOT templates.
def bank():
    N=128; nview=4;d=feature_width; BW=np.empty((10*N*nview,d),np.float32); bt=np.zeros(10*N*nview,np.float32)
    for k in range(10):
        for j in range(N):
            m,a,b,c,mode=model(k);img,op=render(m,a,b,c)
            for h in range(nview//2):
                im=compose(img,op,k,mode,h)
                for f in range(2):
                    i=((k*N+j)*nview+h*2+f);BW[i]=describe(im if f==0 else im[:,::-1]);
    v=aff('individual_view_matches',feat,BW,bt);v=reshape('view_groups',v,10,8,N*nview//8);v=pool('best_view_in_group',v,[1,N*nview//8],kind='max');mx=pool('best_view',v,[8,1],kind='max');av=pool('group_consensus',v,[8,1]);return plus('object_evidence',[aff('strongest',mx,np.eye(10)*.63),aff('agreement',av,np.eye(10)*.37)])

# A separate, entirely analytic branch. Bounded ramps count colour/texture
# memberships instead of letting a few extreme pixels dominate the scene prior.
Y,X=np.mgrid[0:8,0:8];Y=(Y+.5)/8;X=(X+.5)/8
zones={'all':np.ones_like(X),'middle':np.exp(-((X-.5)/.34)**4-((Y-.51)/.30)**4),'sky':(Y<.31)*(1+.5*(abs(X-.5)>.31)),'floor':(Y>.71)*(1+.45*(abs(X-.5)>.32)),'flanks':(abs(X-.5)>.31),'rim':((abs(X-.5)>.34)|(abs(Y-.5)>.34)),'uppercorner':(Y<.38)*(abs(X-.5)>.24),'uppermiddle':(Y<.5)*(abs(X-.5)<.31),'lowmiddle':(Y>.5)*(Y<.84)*(abs(X-.5)<.31)}
zone={k:v/np.sum(v) for k,v in zones.items()}

def scene():
    # tan, foliage, water, strong paint colours, soft neutral colours and darks.
    defs=[('blue',[-.76,-.24,1],.015,.155),('leaf',[-.50,1,-.50],.013,.118),('olive',[-.25,1,-.75],.030,.165),('brown',[.8,.2,-1],.06,.29),('red',[1,-.73,-.27],.115,.36),('white',lum,.61,.89),('black',-lum,-.32,-.10)]
    W=[];b=[]
    for _,v,a,z in defs:W.extend([np.array(v)/(z-a)]*2);b.extend([-a/(z-a),-z/(z-a)])
    c0=relu('color_membership_ramps',conv1('color_tests',low,W,b));WW=np.zeros((len(defs),len(W)));idx=np.arange(len(defs));WW[idx,2*idx]=1;WW[idx,2*idx+1]=-1
    maps=conv1('bounded_color_memberships',c0,WW);names=[x[0] for x in defs]; feats=[]; cues=[]
    def measure(who,channels):
        M=[]
        for j,key in enumerate(channels):
            for region,z in zone.items():
                w=np.zeros((len(channels),8,8));w[j]=z;M.append(w.ravel());cues.append(key+'_'+region)
        feats.append((aff(who+'_statistics',who,M),len(M),1))
    measure(maps,names)
    # Absolute-scale local total variation. A smooth corner is strong evidence
    # for sky; a textured foreground on textured ground is a natural habitat.
    ed=aff('physical_differences',gray,np.concatenate([edge,-edge])[:,None],conv=True,stride=2,pad=1)
    ed=pool('physical_cells',conv1('physical_magnitudes',relu('rectify_physical_differences',ed),np.tile(np.eye(4),(1,2))),2)
    v=V(.25,.25,.25,.25);hd=V(-.45,-.275,1,-.275);vd=V(1,-.275,-.45,-.275)
    e_defs=[('flat',-v,-.086,-.025),('rough',v,.095,.225),('horizontal',hd,.015,.11),('vertical',vd,.015,.11)]
    w=[];b=[]
    for _,v,a,z in e_defs:w.extend([v/(z-a)]*2);b.extend([-a/(z-a),-z/(z-a)])
    e0=relu('texture_membership_ramps',conv1('texture_tests',ed,w,b));w=np.zeros((4,8));i=np.arange(4);w[i,2*i]=1;w[i,2*i+1]=-1; em=conv1('texture_types',e0,w);measure(em,[x[0] for x in e_defs]);measure(conv1('lightness_map',low,[lum]),['lum'])
    stat=concat('spatial_evidence',feats);F=len(cues);rules=[];bias=[];read=[]
    def cue(term,threshold,coefs):
        row=np.zeros(F)
        for key,v in term.items():row[cues.index(key)]+=v
        rules.append(row);bias.append(-threshold);read.append(np.array(coefs))
    # Each coefficient is a prior from everyday scene interpretation. Responses
    # have units of area fraction or intensity difference. Class order is fixed.
    cue({'blue_sky':1},.05,[.19,-.015,.022,-.055,-.05,-.048,-.09,.015,.16,.039])
    cue({'blue_floor':1},.04,[.17,.0,.047,-.05,-.065,-.05,-.10,-.056,.22,-.009])
    cue({'blue_flanks':.60,'flat_flanks':.54},.45,[.16,-.025,.060,-.04,-.03,-.04,-.08,-.03,.080,-.03])
    cue({'leaf_rim':1},.08,[-.15,-.07,.115,.005,.22,.026,.14,.125,-.18,-.09])
    cue({'olive_rim':1},.16,[-.12,-.07,.075,.008,.115,.025,.11,.095,-.11,-.07])
    cue({'leaf_middle':1},.10,[-.11,-.095,.087,-.032,.038,-.037,.28,-.05,-.11,-.09])
    cue({'olive_middle':1},.20,[-.07,-.058,.037,-.013,.025,-.018,.095,-.03,-.07,-.07])
    cue({'brown_middle':1},.11,[-.10,-.055,.036,.056,.115,.10,.078,.136,-.09,-.065])
    cue({'brown_floor':1},.13,[-.087,-.044,.017,.078,.064,.090,.083,.034,-.10,-.045])
    cue({'brown_middle':.7,'leaf_rim':.65},.43,[-.065,-.043,.006,-.029,.12,.012,.04,.15,-.06,-.04])
    cue({'red_middle':1},.065,[-.024,.23,.010,-.025,-.07,-.026,-.06,-.035,.04,.235])
    cue({'red_middle':1,'red_rim':-.85},.07,[.0,.18,.01,.0,-.04,-.02,-.04,-.025,.01,.17])
    cue({'white_middle':1,'white_rim':-.65},.09,[.11,.056,.006,.033,-.080,.025,-.11,-.04,.09,.18])
    cue({'black_middle':1,'black_rim':-.74},.12,[.082,.09,.014,-.036,-.031,-.017,-.046,.043,.074,.04])
    cue({'flat_flanks':1},.21,[.175,.052,.024,-.039,-.054,-.036,-.065,-.025,.105,.047])
    cue({'flat_floor':1},.26,[.073,.10,.023,-.017,-.034,-.027,-.057,-.023,.105,.066])
    cue({'rough_rim':1},.13,[-.15,-.067,.048,.032,.097,.053,.12,.067,-.093,-.071])
    cue({'rough_middle':1,'rough_rim':-.55},.15,[-.02,.002,.018,.073,.003,.057,.030,-.019,-.026,.034])
    cue({'flat_rim':1,'flat_middle':-.85},.19,[.076,.028,.092,-.042,-.006,-.04,-.02,.01,.035,.039])
    cue({'horizontal_all':1},.07,[.052,.18,-.09,-.09,-.03,-.084,-.065,-.008,.20,.18])
    cue({'vertical_all':1},.08,[-.084,.015,-.04,-.044,.016,-.049,-.052,.082,-.019,.21])
    cue({'lum_sky':1,'lum_floor':-1},.12,[-.005,-.035,.009,-.025,.038,-.008,-.086,.084,.14,.055])
    cue({'lum_floor':1,'lum_sky':-1},.04,[.011,.16,-.015,.035,-.070,.045,.004,-.079,-.09,.022])
    cue({'white_sky':1},.21,[.078,-.006,.019,-.025,-.006,-.018,-.080,.029,.072,.046])
    cue({'black_all':1},.21,[-.111,-.027,.002,.040,.086,.035,.14,-.02,-.12,-.075])
    cue({'flat_sky':.48,'flat_floor':.46,'horizontal_middle':.80},.54,[.108,.02,-.029,-.022,-.042,-.029,-.080,-.042,.19,.047])
    cue({'flat_sky':.56,'rough_floor':.72,'brown_middle':.18},.54,[-.04,-.025,.004,-.018,.11,-.02,-.034,.13,-.038,-.026])
    return aff('context_logit',relu('scene_conditions',aff('scene_hypotheses',stat,rules,bias)),np.array(read).T)

# Part detectors: contrast-normalized analytical ring templates, in two scales,
# searched for repeated collinear wheel/hub structures above the road. Also a
# separate horizontal-mirror test for pet faces vs side views and moving limbs.
def parts():
    features=[]
    for j,(radius,step) in enumerate([(1.65,1),(2.85,2)]):
        K=(7,11)[j];yy,xx=np.mgrid[:K,:K]-(K-1)/2;F=[]
        for squash in (.62,1.):
            rr=np.sqrt((xx/squash)**2+yy**2);hub=np.exp(-.5*(rr/(radius*.46))**2); tire=np.exp(-.5*((rr-radius*.79)/(radius*.22))**2);outside=np.exp(-.5*((rr-radius*1.43)/(radius*.31))**2)*(1+.48*(yy<0));
            resp=(.50*hub/hub.sum()+.50*outside/outside.sum()-tire/tire.sum());F.append(resp)
        # zero-pad normalized deviations rather than raw luminance
        w=aff('wheel_filters'+str(j),g0,np.array(F)[:,None],conv=True,stride=step,pad=K//2)
        wp=pool('nearby_wheel'+str(j),w,(4,4) if j==0 else (2,2),kind='max'); wp=reshape('flatten_wheels'+str(j),wp,2,8,8)
        # layout evidence uses a pair threshold, expressed by min(A,B)=A-relu(A-B).
        A=[];B=[]
        for y in (4,5,6):
            for x in range(1,5):
                for gap in (2,3,4,5):
                    if x+gap>7:continue
                    for dy in ((0,1,-1) if gap<4 else (0,1)):
                        a=np.zeros((2,8,8));b=a.copy();a[:,y,x]=.5;b[:,y+dy,x+gap]=.5;A.append(a.ravel());B.append(b.ravel())
        A=np.array(A);B=np.array(B);b1=aff('first_tire'+str(j),wp,A);dif=relu('tire_imbalance'+str(j),aff('compare_tires'+str(j),wp,A-B));v=plus('paired_tires'+str(j),[b1,aff('subtract_imbalance'+str(j),dif,-np.eye(len(A)))]);v=reshape('arrange_tires'+str(j),v,1,1,len(A));v=pool('best_axle_pair'+str(j),v,[1,len(A)],kind='max');features.append((v,1,.5))
    tire=plus('wheel_vote',[aff('wheel_scale'+str(j),f,[[s]]) for j,(f,_,s) in enumerate(features)])
    z=relu('wheel_pair_evidence',aff('strong_wheel_pair',tire,[[1]],[-.66]));return aff('wheel_class_vote',z,np.array([-.014,.092,-.017,-.013,-.015,-.016,-.016,-.012,-.016,.065])[:,None])

# Compare paired small-HOG cells across candidate centers of bilateral symmetry.
# The central upper patch of cat and dog portraits often shows two eyes, two
# ears and a central nose/muzzle. Profiles and birds seldom have that pattern.
def symmetry():
    A=[];groups=[]
    for x0 in (3,3.5,4):
        for y0 in (3,4):
            n=0
            for dx in ((1,2) if x0==int(x0) else (.5,1.5,2.5)):
                for y in range(y0-2,y0+2):
                    a=np.zeros((8,8));a[y,int(x0+dx)]=1;a[y,int(x0-dx)]=-1;A.append(a.ravel());n+=1
            groups.append(n)
    A=np.array(A);v=relu('rectify_reflections',aff('compare_reflections',lo,np.concatenate([A,-A])));rows=[];i=0
    for n in groups:
        w=np.zeros(len(A));w[i:i+n]=1/n;rows.append(np.tile(-w,2));i+=n
    v=relu('reflection_candidates',aff('bilateral_agreement',v,rows,[.54]*len(groups)));v=pool('best_reflection',reshape('reflection_grid',v,1,1,len(groups)),[1,len(groups)],kind='max');return aff('portrait_vote',v,np.array([.02,.028,-.046,.174,-.034,.118,-.026,-.044,-.022,.014])[:,None])

# A second scale of matching: one standalone part illustration per filter.
# Prescribed camera targets are in model coordinates, e.g. not photo keypoints.
# The filters slide over the central field rather than requiring exact framing.
def part_descriptor(img):
    img=img.reshape(20,2,20,2,3).mean((1,3));g=img@lum;pad=np.pad(g,1,mode='edge');patch=np.lib.stride_tricks.sliding_window_view(pad,(4,4))[::2,::2];ed=np.abs(np.einsum('xyij,kij->kxy',patch,edge)).reshape(4,5,2,5,2).mean((2,4));g=g.reshape(5,4,5,4).mean((1,3));q=img.reshape(5,4,5,4,3).mean((1,3))@opponent.T;q=q.transpose(2,0,1)
    def vec(x):
        x=x-x.mean((-2,-1),keepdims=True);return x/max(np.linalg.norm(x),1.e-6)
    return np.concatenate([vec(g)[None],vec(ed),vec(q)])

def local_match():
    n=80; rows=[]
    for k in range(10):
        for j in range(n):
            m,a,b,c,mode=model(k)
            if k==0: point=pick([V(.09,0,0),V(.09,0,0),V(1.05,0,0),V(-1.33,.12,0)]);width=u(1.7,3.3)
            elif k in (1,9):
                if k==1:point=pick([V(1.05,.80,.36),V(.05,1.04,.40),V(-.66,.76,.36),V(1.77,.69,.08)]);width=u(1.83,3.55)
                else:point=pick([V(1.10,1.28,.38),V(1.30,1.57,.24),V(-.20,1.60,.32),V(-1.26,.94,.48)]);width=u(2.12,3.99)
            elif k==8:point=pick([V(.03,1.08,0),V(-.69,1.15,0),V(1.11,.81,0)]);width=u(2.15,4.02)
            elif k in (3,4,5,7):point=m.head.copy();point[1]-=u(-.03,.18);width=(u(1.15,1.95) if k in (3,5) else u(1.30,2.15))
            elif k==2:point=pick([V(.32,1.22,0),V(.12,1.18,0),V(-.17,.88,0)]);width=u(1.03,1.97)
            else:point=pick([V(.68,.69,.12),V(.38,.63,.19),V(.12,.63,.19)]);width=u(1.31,2.40)
            lit,alpha=render(m,a,b,c,(point,width));top,bottom,h=setting(k,mode,1.,.5);bg=top+np.clip(.5+(YY/S-h)*4,0,1)[:,:,None]*(bottom-top);view=np.clip(lit*u(.86,1.17)+(1-alpha[:,:,None])*bg,0,1)
            for obj in [view,view[:,::-1]]:rows.append(part_descriptor(obj))
    # Use 28 positions per feature plane: 25 observed cells and 3 guards
    # (mean-plus-contrast-floor, mean-minus-contrast-floor, mean). Those extra
    # cells prevent unrelated low-amplitude input noise becoming high contrast.
    positions=16; dblock=28; inputs=[(conv1('part_light',low,[lum]),1,.065),(en8,4,.12),(ch,2,.038)];sources=[];channel=0;dims=7
    for a,(inp,C,guard) in enumerate(inputs):
        MM=np.zeros((positions,dims*dblock,C,8,8),np.float32);bias=np.zeros((positions,dims*dblock))
        for py in range(4):
            for px in range(4):
                ix=py*4+px
                for chn in range(C):
                    off=(channel+chn)*dblock
                    for dy in range(5):
                        for dx in range(5):MM[ix,off+dy*5+dx,chn,py+dy,px+dx]=1
                    MM[ix,off+25:off+28,chn,py:py+5,px:px+5]=1/25;bias[ix,off+25]=guard*np.sqrt(14);bias[ix,off+26]=-guard*np.sqrt(14)
        sources.append(aff('extract_regions'+str(a),inp,MM.reshape(positions*dims*dblock,-1),bias.ravel()));channel+=C
    v=reshape('stack_regions',plus('assembled_windows',sources),positions*dims,1,dblock);v=norm('local_contrast',v)
    # Each normalized window has exactly 7*28=14^2 elements. Pack the windows
    # vertically as 14x14 tiles: a stride-14 conv scores each tile independently.
    W=np.zeros((len(rows),dims,dblock));gain=np.array([.32/5]+[.60/10]*4+[.08/np.sqrt(50)]*2);W[:,:,:25]=np.array(rows).reshape(len(rows),dims,25)*gain[None,:,None]
    v=reshape('tiles',v,1,positions*14,14);v=aff('part_filters',v,W.reshape(-1,1,14,14),conv=True,stride=14);v=pool('best_patch_alignment',v,[16,1],kind='max');v=reshape('part_group_grid',v,10,8,n*2//8);v=pool('within_part_group',v,[1,n*2//8],kind='max');v1=pool('best_part',v,[8,1],kind='max');v2=pool('recurring_parts',v,[8,1]);return plus('local_object_vote',[aff('part_strength',v1,.22*np.eye(10)),aff('part_consistency',v2,.28*np.eye(10))])

def main():
    evidence=bank();ctx=scene();wheel=parts();portrait=symmetry();local=local_match();plus('class_scores',[evidence,ctx,wheel,portrait,local]);out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);(out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n');np.savez(out/'weights.npz',**weights)
if __name__=='__main__':main()

