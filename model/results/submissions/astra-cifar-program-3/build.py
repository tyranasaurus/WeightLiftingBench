# All coordinates below are in a drawing's own world units. x = noseward,
# y = up, z = across. Drawings are made out of solids, not photo samples.
import sys, math
from pathlib import Path
import numpy as np

S=48
K=12

def unit(v):
    return v/max(np.linalg.norm(v),1e-12)

class Scene:
    def __init__(self): self.objects=[]
    def ell(self,c,r,v,axis=None):
        if axis is None: M=np.diag(r)
        else:
            a=unit(np.array(axis)); side=unit(np.cross(a,[0,0,1]))
            if np.linalg.norm(side)<0.5: side=unit(np.cross(a,[0,1,0]))
            M=np.stack((a,side,np.cross(a,side)),1)@np.diag(r)
        self.objects.append(('e',np.array(c),M,v))
    def stick(self,a,b,r,s,v):
        a,b=np.array(a),np.array(b); d=b-a
        self.ell((a+b)/2,(np.linalg.norm(d)*.56,r,s),v,d)
    def face(self,pts,v): self.objects.append(('p',np.array(pts),v))
    def loft(self,levels,v):
        # Each ring runs around the object; bridge consecutive rings.
        for i in range(len(levels)-1):
            a,b=levels[i],levels[i+1]
            for j in range(len(a)):
                k=(j+1)%len(a); self.face([a[j],a[k],b[k],b[j]],v)
        self.face(levels[0][::-1],v); self.face(levels[-1],v)
    def block(self,x0,x1,y0,y1,w,v):
        self.loft([[(x,y0,-w),(x,y1,-w),(x,y1,w),(x,y0,w)] for x in (x0,x1)],v)
    def cabin(self,xb,xf,xtb,xtf,yb,yt,wb,wt,v):
        self.loft([[(xb,yb,-wb),(xf,yb,-wb),(xf,yb,wb),(xb,yb,wb)],[(xtb,yt,-wt),(xtf,yt,-wt),(xtf,yt,wt),(xtb,yt,wt)]],v)
    def wheel(self,x,y,z,rad,th):
        a=np.arange(16)*2*np.pi/16
        self.loft([[(x+rad*np.cos(t),y+rad*np.sin(t),zz) for t in a] for zz in (z-th,z+th)],.11)
        self.ell((x,y,z),(.52*rad,.52*rad,th*1.18),.57)


def car(style):
    s=Scene(); body=.71 if style!=2 else .42
    # saloon, hatchback, sports coupe, people carrier
    hi=[.89,1.00,.72,1.08][style]; rear=[-.9,-1.26,-.86,-1.40][style]
    rt=[-.60,-1.10,-.46,-1.24][style]; ft=[.36,.45,.30,.66][style]
    s.cabin(-1.50,1.50,-1.59,1.56,-.01,.34,.58,.65,body*.82)
    s.cabin(-1.59,1.56,-1.44,1.46,.34,.49,.65,.63,body)
    s.cabin(rear,1.08,rt,ft,.48,hi,.62,.47,.34)
    s.block(rt,ft,hi-.025,hi+.015,.47,body)
    for k in (-1,1):
        s.face([(-.23,.49,k*.627),(-.11,.49,k*.627),(-.18,hi,k*.476),(-.26,hi,k*.476)],body*.8)
        for x in (-.99,1.00): s.wheel(x,.085,.585*k,.285,.090)
        s.ell((1.54,.348,k*.44),(.050,.073,.155),.91)
        s.ell((-1.54,.338,k*.48),(.050,.074,.12),.33)
        s.ell((.72,.58,k*.65),(.12,.062,.088),body)
    s.face([(1.569,.32,-.36),(1.569,.32,.36),(1.545,.15,.34),(1.545,.15,-.34)],.24)
    s.block(1.52,1.58,.16,.22,.19,.82); return s


def truck(style):
    s=Scene(); rear=-1.78; front=1.65; c=.62
    s.block(rear,front,-.02,.27,.49,.19)
    if style==3: # bus
        s.cabin(rear,front,rear+.06,front-.08,.21,1.48,.62,.61,.76)
        for k in (-1,1):
            for x in np.linspace(-1.50,1.10,6):
                s.face([(x,.86,k*.627),(x+.38,.86,k*.627),(x+.38,1.35,k*.622),(x,1.35,k*.622)],.32)
        s.face([(1.63,.83,-.53),(1.63,.83,.53),(1.586,1.35,.53),(1.586,1.35,-.53)],.30)
    else:
        end=.34 if style!=2 else .12
        if style==1: s.ell((-.75,.99,0),(1.24,.64,.65),.72)
        else: s.block(rear,end,.34,1.78 if style==0 else 1.30,.67,.81 if style==0 else .53)
        s.cabin(end+.03,front,end+.04,front-.04,.24,.85,.58,.60,c)
        ct=front-.13 if style!=2 else front-.49
        s.cabin(end+.05,ct+.08,end+.07,ct-.04,.81,1.49,.60,.555,c)
        s.face([(ct+.083,.88,-.52),(ct+.083,.88,.52),(ct-.018,1.38,.50),(ct-.018,1.38,-.50)],.32)
        for k in (-1,1):
            s.face([(end+.14,.92,.599*k),(ct+.012,.92,.599*k),(ct-.075,1.37,.57*k),(end+.14,1.37,.57*k)],.28)
        s.face([(front+.01,.45,-.34),(front+.01,.45,.34),(front-.024,.76,.34),(front-.024,.76,-.34)],.26)
    for k in (-1,1):
        for x in ([-1.22,1.14] if style==3 else [-1.36,-.80,1.13]): s.wheel(x,.16,k*.545,.34,.12)
        s.ell((front,.43,k*.45),(.04,.085,.10),.94)
    return s


def plane(style):
    s=Scene(); c=.78; s.ell((0,0,0),(1.72,.185,.19),c)
    if style==2: s.ell((.44,.17,0),(.43,.15,.17),.28)
    sw=[.60,.12,.92][style]; span=[1.70,1.46,1.35][style]
    for k in (-1,1):
        s.face([(.47,-.04,0),(.31-sw,.07,k*span),(.02-sw,.07,k*span),(-.45,-.04,.14*k)],.67)
        s.face([(-1.02,.04,0),(-1.39,.14,.67*k),(-1.60,.14,.67*k),(-1.49,.04,0)],.67)
        if style==0: s.ell((.14,-.17,.7*k),(.34,.15,.14),.55)
    s.face([(-.93,.09,0),(-1.41,.64,0),(-1.61,.65,0),(-1.54,.06,0)],.54)
    if style==1: s.ell((1.64,0,0),(.03,.58,.045),.24)
    return s


def ship(style):
    s=Scene(); deck=.04; widths=[.43,.56,.60,.39,.018]
    rings=[]
    for x,w in zip([-1.63,-1.05,.54,1.20,1.7],widths):
        rise=.18 if x>1.6 else 0
        rings.append([(x,-.41+rise,-.57*w),(x,deck+rise,-w),(x,deck+rise,w),(x,-.41+rise,.57*w)])
    s.loft(rings,.40 if style!=1 else .72)
    if style==0: # freighter
        s.block(-1.41,-.82,.04,.56,.42,.84); s.block(-1.35,-.92,.56,.76,.36,.79)
        s.block(-1.34,-1.14,.76,1,.12,.35)
        for xx in (-.70,-.14,.42): s.block(xx,xx+.52,.04,.32,.44,.55)
    if style==1 or style==3: # motor vessel / liner
        for i in range(2 if style==1 else 3):
            xb=-1.38+.18*i; xf=(.64 if style==1 else 1.17)-.28*i
            y=.04+.27*i; w=.48-.052*i; s.cabin(xb,xf,xb+.11,xf-.16,y,y+.26,w,w-.03,.86)
            for k in (-1,1):
                s.face([(xb+.15,y+.10,k*(w-.006)),(xf-.18,y+.10,k*(w-.006)),(xf-.24,y+.21,k*(w-.020)),(xb+.17,y+.21,k*(w-.020))],.32)
        s.stick((-.64,.54,0),(-.72,.94,0),.09,.09,.49)
    if style==2:
        s.stick((.16,.06,0),(.16,2.54,0),.033,.032,.36)
        s.face([(.13,.36,0),(.13,2.55,0),(-1.18,.36,.09)],.88)
        s.face([(.28,2.25,0),(.29,.40,0),(1.47,.27,0)],.80)
    return s


def markedpet(style):
    # Lay markings just outside the ellipsoidal head: pale facial blaze,
    # dark eye patches, a dark mask, or a long-haired ruff.
    s=pet(3 if style in (0,2) else 5,False)
    for k in (-1,1):
        if style!=2: s.ell((.19,.095,k*.226),(.135,.14,.117),.27)
        if style in (0,2):
            s.stick((-.08,.16,k*.29),(-.12,-.30,k*.43),.157,.14,.27 if style==0 else .50)
        if style==1:
            s.stick((.19,.25,k*.19),(.28,.17,k*.18),.069,.066,.73)
        s.ell((.30,.071,k*.214),(.058,.054,.05),.10)
    if style==0: s.stick((.19,.32,0),(.36,-.01,0),.079,.071,.88)
    if style==1: s.ell((.38,-.09,0),(.16,.17,.216),.32)
    s.ell((.54,-.09,0),(.073,.069,.113),.10)
    if style==2: s.ell((-.13,-.55,0),(.42,.42,.40),.78)
    return s

def portrait(deer):
    # Neck and elongated muzzle filling the frame; mane or branching antlers.
    s=Scene(); c=.58 if deer else .48
    s.stick((-.23,-.52,0),(.25,.66,0),.33 if not deer else .23,.25,c)
    s.stick((.27,.71,0),(1.02,.21,0),.22 if not deer else .17,.21,c)
    s.ell((.96,.24,0),(.154,.146,.18),c*.85,(1,-.55,0))
    if not deer: s.stick((-.48,-.40,0),(.16,.80,0),.105,.15,c*.54)
    for k in (-1,1):
        ear(s,(.24,.77,k*.12),.25 if not deer else .29,.17,-.09,c)
        s.ell((.48,.64,k*.169),(.050,.048,.038),.13)
        s.ell((1.036,.254,k*.127),(.042,.048,.033),.19)
        if deer:
            s.stick((.32,.78,k*.095),(.13,1.31,k*.44),.048,.05,.37)
            s.stick((.16,1.21,k*.40),(-.045,1.48,k*.48),.037,.038,.37)
            s.stick((.21,1.07,k*.30),(.32,1.36,k*.34),.035,.035,.37)
    return s

def curl():
    s=Scene(); c=.62
    s.ell((-.13,.43,0),(.88,.50,.52),c)
    s.ell((.53,.65,.27),(.38,.35,.36),c)
    for k in (-1,1):
        ear(s,(.48,.85,.27+k*.16),.22,.24,-.025,c*.88)
        s.ell((.80,.69,.27+k*.17),(.042,.023,.052),.24)
    s.ell((.83,.58,.27),(.103,.075,.12),.77)
    s.stick((-.75,.24,.32),(.25,.17,.48),.129,.12,c*.82)
    return s

def ear(s,c,length,width,lean,v):
    x,y,z=c
    s.loft([[(x-width*.45,y,z-width*.72),(x+width*.55,y,z-width*.57),(x+width*.53,y,z+width*.68),(x-width*.52,y,z+width*.71)],[(x+lean,y+length,z+np.sign(z)*width*.48),(x+lean+.016,y+length,z+np.sign(z)*width*.48),(x+lean+.01,y+length-.016,z+np.sign(z)*width*.51),(x+lean-.01,y+length-.02,z+np.sign(z)*width*.51)]],v)


def ungulate(style,deer):
    s=Scene(); thin=.78 if deer else 1.; coat=.59 if deer else .46
    s.ell((-.19,1.26,0),(1.05,.46*thin,.36*thin),coat)
    s.ell((-.82,1.30,0),(.46,.44*thin,.37*thin),coat)
    s.ell((.55,1.34,0),(.39,.48*thin,.33*thin),coat)
    low=style==1
    a=(.53,1.46,0); b=(1.31,.87,0) if low else (1.00,2.05+(0.17 if deer else 0),0)
    s.stick(a,b,.26*thin,.22*thin,coat)
    bx,by,z=b; s.stick((bx-.08,by+.06,0),(bx+.44,by-.27,0),.19*thin,.17*thin,coat)
    if not deer: s.stick((a[0]-.15,a[1]+.06,0),(bx-.10,by+.12,0),.095,.12,.25)
    for k in (-1,1):
        ear(s,(bx-.06,by+.11,k*.095),.27 if deer else .20,.16,-.071,coat)
        s.ell((bx+.10,by+.003,k*.147*thin),(.041,.039,.036),.13)
        s.ell((bx+.42,by-.225,k*.080*thin),(.056,.055,.057),.26)
        for j,x in enumerate((-.86,.57)):
            step=(.16 if style!=2 else .46)*k*(1 if j else -1)
            lz=k*.23*thin; h=(x,1.23,lz); knee=(x+(-.15 if j==0 else .06)+step*.42,.55,lz*1.09)
            foot=(x+step,.04,lz*1.22)
            s.stick(h,knee,.129*thin,.111*thin,coat); s.stick(knee,foot,.070*thin,.070*thin,coat*.86)
            s.ell(foot,(.084*thin,.064,.080*thin),coat*.53)
    if deer: s.ell((-1.21,1.40,0),(.17,.12,.12),.78)
    else: s.stick((-1.20,1.50,0),(-1.42,.55,0),.139,.133,.25)
    return s


def pet(style,cat):
    s=Scene(); face=style>=3; sit=style==1; c=.60
    if face: hx,hy=0,0
    elif sit: hx,hy=.37,1.59
    else: hx,hy=.85,1.35
    fluffy=style in (2,4); f=1.19 if fluffy else 1
    if not face:
        if sit:
            s.stick((-.26,.48,0),(.26,1.33,0),.53,.40,c)
            s.ell((-.29,.41,0),(.55,.40,.40),c)
        else: s.ell((-.19,.82,0),(.90,.39*f,.36*f),c)
        s.stick((hx-.22,hy-.42,0),(hx,hy,0),.29,.29,c)
        for k in (-1,1):
            for j,x in enumerate((-.72,.40)):
                lx=x if not sit else (-.38 if not j else .42)
                foot=(lx+.11+(.11*k if not sit else 0),.07,k*.29)
                s.stick((lx,.65 if not sit else .53, k*.23),foot,.103*f,.105*f,c)
                s.ell(foot,(.159,.079,.109),c)
        tbase=(-1.0,.84,0) if not sit else (-.65,.34,0)
        s.stick(tbase,(tbase[0]-.35,tbase[1]+(.52 if not sit else -.19),.12),.107*f,.103*f,c*.86)
    wx=.39 if cat else .36; ht=.38 if cat else .43; ww=.39 if cat else .36
    s.ell((hx,hy,0),(wx,ht,ww),c)
    snout=.30 if cat else (.34 if fluffy else .46)
    s.ell((hx+snout*.76,hy-.105,0),(.16 if cat else .26,.16,.215),.72)
    s.ell((hx+snout+(.058 if cat else .125),hy-.085,0),(.048 if cat else .083,.043 if cat else .075,.064 if cat else .115),.37 if cat else .14)
    for k in (-1,1):
        s.ell((hx+.295,hy+.068,k*.21),(.066,.041 if cat else .050,.061),.19)
        if cat or style in (2,5): ear(s,(hx-.065,hy+.22,k*.210),.265 if cat else .34,.27,-.04,c*.89)
        else: s.stick((hx-.045,hy+.21,k*.245),(hx-.081,hy-.21,k*.399),.156,.101,c*.62)
        if cat and style in (2,4):
            s.stick((hx+.13,hy+.32,k*.13),(hx+.29,hy+.18,k*.12),.029,.033,.40)
    return s


def bird(style):
    s=Scene(); fly=style==2; upright=style in (1,3)
    s.ell((-.12,.71,0),(.61,.34,.31),.55,(.64,.73 if upright else .22,0))
    h=(.20,1.27,0) if upright else (.43,.97,0)
    s.ell(h,(.25,.255,.242),.51); hx,hy,z=h
    s.loft([[(hx+.18,hy+.08,-.095),(hx+.18,hy-.067,-.092),(hx+.18,hy-.070,.093),(hx+.18,hy+.08,.096)],[(hx+.50,hy-.04,-.005),(hx+.50,hy-.044,-.005),(hx+.50,hy-.044,.005),(hx+.50,hy-.04,.005)]],.47)
    for k in (-1,1):
        s.ell((hx+.13,hy+.048,k*.19),(.047,.047,.037),.15)
        if fly: s.face([(.27,.76,0),(.30,1.02,k*.84),(-.18,1.21,k*1.58),(-.34,1.04,k*.75),(-.48,.71,0)],.42)
        else: s.stick((.22,1 if upright else .85,k*.18),(-.56,.40 if upright else .58,k*.22),.185,.103,.40)
        s.stick((-.44,.48 if upright else .62,k*.09),(-.86,.08 if upright else .54,k*.13),.092,.11,.36)
        if not fly: s.stick((-.09,.43,k*.10),(-.06,.10,k*.16),.028,.027,.33)
    return s


def frog(style):
    s=Scene(); tall=style==1; c=.53
    s.ell((-.17,.34,0),(.74,.36,.49),c,(1,.42 if tall else .13,0))
    hh=.64 if tall else .47
    s.ell((.35,hh,0),(.49,.283,.50),c)
    s.ell((.46,hh-.094,0),(.40,.16,.45),.68)
    for k in (-1,1):
        s.ell((.34,hh+.178,k*.32),(.158,.16,.151),c)
        s.ell((.40,hh+.235,k*.37),(.094,.095,.092),.26)
        s.ell((-.50,.26,k*.39),(.44,.24,.23),c)
        s.stick((-.71,.20,k*.45),(-.28,.075,k*.64),.104,.10,c*.89)
        s.stick((.22,hh-.12,k*.33),(.02,.10,k*.58),.099,.093,c)
        s.stick((.02,.10,k*.58),(.43,.03,k*.67),.071,.069,c)
    return s


def render(scene,yaw,elev,roll):
    a,b,t=np.deg2rad([yaw,elev,roll]); co=np.cos; si=np.sin
    u=np.array([co(a),0,-si(a)]); v=np.array([-si(b)*si(a),co(b),-si(b)*co(a)])
    cam=np.stack([u*co(t)+v*si(t),-u*si(t)+v*co(t),[co(b)*si(a),si(b),co(b)*co(a)]])
    obj=[]; bounds=[]
    for o in scene.objects:
        if o[0]=='e':
            c,M=cam@o[1],cam@o[2]; r=np.sqrt((M*M).sum(1)); bounds.extend([c-r,c+r]); obj.append(('e',c,M,o[3]))
        else:
            pts=o[1]@cam.T; bounds.extend(pts); obj.append(('p',pts,o[2]))
    bounds=np.array(bounds); mid=(bounds.max(0)+bounds.min(0))/2; span=np.ptp(bounds,axis=0)[:2].max(); scl=span/(S*.91)
    grid=(np.arange(S)+.5-S/2)*scl
    X,Y=np.meshgrid(grid+mid[0],-grid+mid[1]); depth=np.full((S,S),-999.); color=np.zeros((S,S)); light=unit(np.array([-.28,.76,.58]))
    for o in obj:
        if o[0]=='e':
            c,M,base=o[1:]; Q=np.linalg.inv(M@M.T); dx=X-c[0]; dy=Y-c[1]
            line=Q[2,0]*dx+Q[2,1]*dy; delta=line**2-Q[2,2]*(Q[0,0]*dx**2+2*Q[0,1]*dx*dy+Q[1,1]*dy**2-1)
            dz=(np.sqrt(np.maximum(delta,0))-line)/Q[2,2]; dep=dz+c[2]; use=(delta>=0)&(dep>depth)
            if not use.any(): continue
            n=Q@np.stack([dx[use],dy[use],dz[use]]); shading=(light@n)/np.sqrt((n*n).sum(0)+1e-18)
            color[use]=base*(.85+.18*shading); depth[use]=dep[use]
        else:
            p,base=o[1:]; n=np.cross(p[1]-p[0],p[2]-p[0]); n=unit(n)
            if abs(n[2])<.00001: continue
            if n[2]<0: n=-n
            dep=p[0,2]-(n[0]*(X-p[0,0])+n[1]*(Y-p[0,1]))/n[2]; inside=np.zeros_like(X,dtype=bool)
            for pt,qt in zip(p,np.roll(p,-1,0)):
                if pt[1]!=qt[1]: inside^=((pt[1]>Y)!=(qt[1]>Y))&(X<(qt[0]-pt[0])*(Y-pt[1])/(qt[1]-pt[1])+pt[0])
            use=inside&(dep>depth); color[use]=base*(.85+.18*(n@light)); depth[use]=dep[use]
    return color.astype('f')

# The same orientation channels and pooling geometry are used on drawings and inputs.
shared=r'''import numpy as np

def smooth(a):
    v=np.pad(a,[(0,0)]*(a.ndim-2)+[(1,1),(1,1)],mode='edge')
    h=(v[...,2:]+v[...,:-2]+2*v[...,1:-1])*.25
    return (h[...,2:,:]+h[...,:-2,:]+2*h[...,1:-1,:])*.25

def diff(a):
    v=np.pad(a,[(0,0)]*(a.ndim-2)+[(1,1),(1,1)],mode='edge')
    return (v[...,1:-1,2:]-v[...,1:-1,:-2])*.5,(v[...,2:,1:-1]-v[...,:-2,1:-1])*.5

def angles(dx,dy):
    planes=[]
    for a in range(6):
        c=np.float32(np.cos(a*np.pi/6)); s=np.float32(np.sin(a*np.pi/6))
        u=np.abs(dx*c+dy*s); v=np.abs(dx*s-dy*c)
        planes.append(np.maximum(u-1.45*v,0))
    return np.stack(planes,1)

def pool(a):
    return a.reshape(a.shape[:-2]+(12,4,12,4)).mean((-1,-3))

def norm(a,eps):
    return a/np.sqrt(np.sum(a*a,axis=tuple(range(1,a.ndim)),keepdims=True)+eps)
'''
# These operate only on the analytical drawings.
exec(shared)

def sample(src,x,y):
    xx=np.clip(x,0,S-1.0001); yy=np.clip(y,0,S-1.0001); i=xx.astype(int); j=yy.astype(int); u=xx-i; v=yy-j
    return ((1-u)*(1-v)*src[j,i]+u*(1-v)*src[j,i+1]+(1-u)*v*src[j+1,i]+u*v*src[j+1,i+1]).astype('f')

def features(views):
    # Compare gradients both under a light and a dark surround, for outline polarity invariance.
    mask,gray=views[:,0],views[:,1]; hog=0
    for bg in (.22,.78):
        val=gray+(1-mask)*bg
        dx,dy=diff(smooth(val)); hog=hog+angles(dx,dy)
    hog=pool(smooth(hog)); hog=np.sqrt(hog+.0015)-np.float32(np.sqrt(.0015))
    hog=hog-hog.mean((2,3),keepdims=True); hog=norm(hog,.02)
    m=pool(mask); mmean=m.mean((1,2),keepdims=True); shape=norm(m-mmean,.10)
    p=pool(gray); photo=norm(p-m*(p.sum((1,2),keepdims=True)/(m.sum((1,2),keepdims=True)+.001)),.28)
    return [ar.reshape(len(m),-1).astype('f') for ar in (hog,shape,photo)]

def generate():
    byclass=[]
    Y,X=np.mgrid[:S,:S].astype('f')-(S-1)/2
    for k in range(10):
        models=([plane(i) for i in range(3)] if k==0 else [car(i) for i in range(4)] if k==1 else [bird(i) for i in range(3)] if k==2 else [pet(i,k==3) for i in range(6)] if k in (3,5) else [ungulate(i,k==4) for i in range(3)] if k in (4,7) else [frog(i) for i in range(2)] if k==6 else [ship(i) for i in range(4)] if k==8 else [truck(i) for i in range(4)])
        if k==3: models.append(curl())
        if k==5: models.extend(markedpet(i) for i in range(3))
        if k in (4,7): models.append(portrait(k==4))
        rows=[[],[],[]]; cache=[]
        def flush():
            vals=features(np.stack(cache)); cache.clear()
            for dest,src in zip(rows,vals): dest.append(src)
        for j,model in enumerate(models):
            aa=[-38,0,35,66,88]; els=[8,25]; rolls=[0]
            if k==0: aa=[0,40,78]; els=[-25,7,42,77]; rolls=[-23,0,23]
            if k in (2,4,7): aa=[0,38,75]; els=[5,20]
            if k in (3,5) and 3<=j<6: aa=[14,46,69,90]; els=[3,25]
            if k==5 and j>=6: aa=[35,63,90]; els=[3,20]
            if k==3 and j==6: aa=[-20,22,63]; els=[14,40,62]
            if k==6: aa=[0,45,83]; els=[10,37,64]
            if k==8: aa=[0,40,74]; els=[6,19]; rolls=[0]
            for yaw in aa:
                for elev in els:
                    for roll in rolls:
                        im=render(model,yaw,elev,roll); mask=(im!=0).astype('f')
                        for size in ([.69,.92,1.12] if k in (1,9) else [.49,.72,.96,1.14] if k not in (3,5,6) else [.60,.83,1.08]):
                            for flip in (1,-1):
                                # Centered object, plus modest displacements in all directions.
                                for ox,oy in [(0,0),(-.081,-.061),(.081,.061),(-.066,.085),(.066,-.085)]:
                                    xx=(X-S*ox)*flip/size+(S-1)/2; yy=(Y-S*oy)/size+(S-1)/2
                                    cache.append(np.stack([sample(mask,xx,yy),sample(im,xx,yy)]))
                                    if len(cache)>=64: flush()
        if cache: flush()
        byclass.append([np.concatenate(row) for row in rows])
    return dict(h=np.ascontiguousarray(np.concatenate([b[0] for b in byclass]).T),m=np.ascontiguousarray(np.concatenate([b[1] for b in byclass]).T),p=np.ascontiguousarray(np.concatenate([b[2] for b in byclass]).T),ends=np.cumsum([b[0].shape[0] for b in byclass]))

solution=r'''
def clip(a):
    return np.clip(a,0,1)

def between(x,a,b):
    return clip((x-a)/(b-a))

def enlarge(x):
    u=(np.arange(48,dtype=np.float32)+.5)*(32/48)-.5
    u=np.clip(u,0,30.99999); j=u.astype(int); f=u-j.astype(np.float32)
    a=x[...,j]*(1-f)+x[...,j+1]*f
    return a[...,j,:]*(1-f[:,None])+a[...,j+1,:]*f[:,None]

def avg(a):
    return a.mean((-2,-1))

def stretch(a):
    return norm(a-a.mean((-2,-1),keepdims=True),.35)

def borderdiff(rgb):
    # How different is this colour from several nearby background samples?
    # A nearest single boundary pixel is unreliable; allow soft matches.
    n=len(rgb); yy,xx=np.mgrid[:12,:12]; by,bx=np.nonzero((xx==0)|(xx==11)|(yy==0)|(yy==11))
    dx=xx[None,:,:]-bx[:,None,None]; dy=yy[None,:,:]-by[:,None,None]
    spatial=(dx*dx/90+dy*dy/31).astype(np.float32)
    bd=rgb[:,:,by,bx]; d=rgb[:,:,None,:,:]-bd[:,:,:,None,None]
    dl=.27*d[:,0]+.55*d[:,1]+.18*d[:,2]; dg=d[:,1]-d[:,2]; dr=d[:,0]-d[:,1]
    delta=dl*dl+.68*(dg*dg+dr*dr)
    prob=np.exp(-spatial); w=(np.exp(-delta/.020)*prob).sum(1)/prob.sum(0)
    return -np.log(.012+w)

def boxline(a,axis,n):
    if axis==1: a=a.transpose(0,2,1)
    p=np.pad(a,[(0,0),(0,0),(n,n)],mode='edge'); b=p[:,:,:a.shape[2]].copy()
    for j in range(1,2*n+1): b+=p[:,:,j:j+a.shape[2]]
    b*=1/(2*n+1)
    return b.transpose(0,2,1) if axis==1 else b

def blurbox(a,n):
    return boxline(boxline(a,2,n),1,n)

def shift(a,dy,dx):
    yy=np.clip(np.arange(32)+dy,0,31); xx=np.clip(np.arange(32)+dx,0,31)
    return a[:,yy,:][:,:,xx]

def pairparts(g):
    # Tires with hubcaps or uniformly dark tires, viewed side-on or in perspective.
    # A pair of strong responses beats a lone round dark spot.
    s=smooth(g); local=blurbox(s,3); contrast=.055+.6*blurbox(np.abs(s-local),3)
    wheels=[]
    for rad in (2,3,4):
        z=np.zeros_like(g); outer=z.copy(); tire=z.copy()
        for cx,cy in ((1,0),(-1,0),(0,1),(0,-1),(.70,.70),(-.70,.70),(.70,-.70),(-.70,-.70)):
            outer+=shift(s,round(rad*1.5*cy),round(rad*1.5*cx))/8
            tire+=shift(s,round(rad*.78*cy),round(rad*.78*cx))/8
        z=between((outer-tire+.40*np.maximum(s-tire,0))/contrast,.24,2.2)*(.48+.52*between(tire,.51,.13))
        best=np.zeros(len(s),np.float32)
        for sep in (rad*4,rad*6):
            for slope in (-3,0,3):
                l=z[:,14:29,1:31-sep]; rr=shift(z,slope,sep)[:,14:29,1:31-sep]
                v=np.sqrt(l*rr).max((1,2)); best=np.maximum(best,v)
        wheels.append(best)
    wheels=np.maximum.reduce(wheels)
    # Eye pair above a nose: require all three dark details relative to local fur.
    eyes=between((local-s)/contrast,.26,2.1); noses=between((blurbox(s,4)-blurbox(s,1))/contrast,.18,2.0)
    dogface=np.zeros(len(s),np.float32); catface=dogface.copy()
    for half in (2,3,4,6):
        a=shift(eyes,0,-half); b=shift(eyes,0,half)
        pair=np.sqrt(a*b)
        for height in (half,half*2):
            n=shift(noses,height,0); quality=pair*(.3+.7*n)
            qual=quality[:,5:23,5:27].max((1,2))
            if height==half: catface=np.maximum(catface,qual)
            else: dogface=np.maximum(dogface,qual)
    return wheels,dogface,catface

def spectraltexture(g):
    # Structure at three octaves. Discount energy near horizontal/vertical
    # axes when estimating stippling: railings or a truck grille are not fur.
    u=np.arange(32,dtype=np.float32); w=np.sin(np.pi*(u+.5)/32)**1.3
    tap=w[:,None]*w; center=(g*tap).sum((1,2),keepdims=True)/tap.sum()
    p=np.abs(np.fft.rfft2((g-center)*tap))**2
    fy=np.fft.fftfreq(32)[:,None]*32; fx=np.fft.rfftfreq(32)[None,:]*32
    rad=np.sqrt(fx*fx+fy*fy); diag=np.minimum(np.abs(fx),np.abs(fy))/(rad+.001)
    def E(lo,hi): return (p*((rad>=lo)&(rad<hi))).sum((1,2))
    coarse=E(1,4); medium=E(4,8); small=E(8,15)
    isot=(p*((rad>3)&(diag>.26))).sum((1,2))/(medium+small+.01)
    return between((small+.05)/(medium+.1),.16,.76),between(medium/(coarse+1),.12,.74),between(isot,.29,.79)

def localmarks(x,g):
    r,v,b=x[:,0],x[:,1],x[:,2]; dark=between(g,.32,.09); pale=between(g,.59,.88)
    neutral=1-between((x.max(1)-x.min(1))/(g+.3),.07,.31)
    ochre=between((r-b)/(g+.22),.08,.30)*between((r-v)/(g+.22),-.06,.12)
    # A facial dark spot has lighter fur on both sides; a side-lit face
    # need not be bilaterally identical. The same arrangement with a long
    # flat lower radiator is penalized below by the object-family evidence.
    best=np.zeros(len(g),np.float32); nosebest=best.copy()
    for rad in (1,2,3):
        val=g if rad==1 else blurbox(g,rad-1); surround=blurbox(g,2*rad+1); con=between(surround-val,.025,.24)
        nos=con*(.48+.52*between(val,.43,.12)); eye=con
        for w in (rad+2,rad+4):
            a=shift(eye,-w,-w); bresp=shift(eye,-w,w)
            cheeks=.5*(shift(val,rad,-2*rad)+shift(val,rad,2*rad)); bridge=shift(val,-rad*2,0)
            support=between(.65*cheeks+.35*bridge-val,.028,.25)
            f=nos*support*(.43+.35*np.minimum(a,bresp)+.22*np.maximum(a,bresp))
            best=np.maximum(best,f[:,8:27,6:-6].max((1,2)))
            nosebest=np.maximum(nosebest,(nos*support)[:,9:26,8:24].max((1,2)))
    dc=blurbox(dark,2); wh=blurbox(pale*neutral,2); tn=blurbox(ochre,2); face=avg(dc[:,7:23,6:26]); wp=avg(wh[:,7:26,6:26]); brown=avg(tn[:,7:26,6:26])
    coat=between(np.sqrt(face*(wp+.48*brown)),.085,.28)
    # Pigment patches, in contrast to a few dark individual high-frequency spots.
    patches=avg((dc*dc+wh*wh)[:,7:25,5:27]); patches=between(patches,.08,.39)
    return best,nosebest,coat,patches

def habitat(x,g,sal):
    # Class-independent material indicators. Each is a proportion of pixels
    # explained by a material (not a probability model estimated from photos).
    r,v,b=x[:,0],x[:,1],x[:,2]; ma=x.max(1); mi=x.min(1); sat=ma-mi
    den=.22+g; cool=(b-r)/den; blue=between(cool,.015,.18)*between((b-v)/den,-.085,.055)
    leaves=between((v-b)/den,.03,.17)*between((v-r)/den,-.106,.054)
    grass=between((v-b)/den,.09,.32)*between((v-r)/den,-.17,.07)
    dirt=between((r-b)/den,.092,.32)*between((v-b)/den,.02,.18)*between((r-v)/den,.014,.12)
    neutral=1-between(sat/den,.075,.29)
    grayroad=neutral*between(g,.055,.23)*between(g,.94,.61)
    gold=between((r-b)/den,.12,.40)*between((v-b)/den,.10,.31)
    rawdx,rawdy=diff(g); noise=np.sqrt(rawdx*rawdx+rawdy*rawdy)
    micro=blurbox(np.abs(g-smooth(g)),2); smoothness=np.exp(-micro/.023)
    sky=between(b,.33,.72)*between((b-r)/den,-.09,.025)*between((b-v)/den,-.065,.018)*(.50+.50*smoothness)
    # Water includes dark turquoise, grey and brown water; for these colours
    # insist on long horizontal ripples or a clear change from the sky above.
    water=blue*(.55+.45*smoothness)
    colorpaint=between(sat/den,.20,.49)*np.maximum.reduce((between((b-v)/den,.065,.25),between((r-v)/den,.095,.28)*(1-between((v-b)/den,.045,.18)),between((r-b)/den,-.05,-.22)*between((v-r)/den,.12,.40)))
    icy=neutral*between(g,.73,.93)
    skyhi=avg(sky[:,:8]); skylo=avg(sky[:,-8:]); skyedges=.5*(avg(sky[:,:,:4])+avg(sky[:,:,-4:]))
    waterlo=avg(water[:,-9:]); watertop=avg(water[:,:9]); ro=avg(grayroad[:,-6:]); botgreen=avg(grass[:,-8:]); topgreen=avg(leaves[:,:9]); leafy=avg(leaves); ochre=avg(dirt); ochrelow=avg(dirt[:,-9:]); coolall=avg(blue); centerblue=avg(blue[:,9:23,9:23]); paint=avg(colorpaint[:,7:25,5:27]); gray=avg(neutral); dark=avg(between(g[:,6:27,6:27],.30,.08)); white=avg(icy[:,5:26,6:26]); bright=avg(g)
    center=avg(g[:,10:24,9:23]); fgcoat=avg(dirt[:,9:23,9:23]); cgreen=avg(leaves[:,9:24,8:24]); muted=1-avg(between(sat/den,.14,.37)); plain=(avg(smoothness[:,:8])+avg(smoothness[:,-8:]))*.5
    # Relative concentration of texture and colour variation (shallow depth of
    # field, a foreground subject filling the frame, or detailed ground cover).
    border=(avg(noise[:,:6])+avg(noise[:,-6:])+avg(noise[:,:,:5])+avg(noise[:,:,-5:]))*.25
    inedge=avg(noise[:,8:24,7:25]); isolate=np.clip((inedge-border)/(inedge+border+.02),-.65,.75)
    fgstd=x[:,:,6:25,6:26].std((2,3)).mean(1)
    topsd=x[:,:,:8].std((2,3)).mean(1); botsd=x[:,:,-8:].std((2,3)).mean(1)
    topflat=between(topsd,.215,.065); botflat=between(botsd,.21,.065); flat=(topflat+botflat)*.5
    # Horizontal straightness of boundary patches; a horizon is a coherent
    # full-width change rather than merely a horizontal texture element.
    stripe=x.reshape(len(x),3,32,4,8).mean(4)
    change=stripe[:,:,4:]-stripe[:,:,:-4]; dh=change.mean(3); coherent=(dh**2).mean(1)/( (change**2).mean((1,3))+.004)
    amp=np.sqrt((dh**2).mean(1)); horizon=(between(amp,.06,.26)*coherent)[:,3:26].max(1)
    ux=np.abs(rawdx); uy=np.abs(rawdy); strength=avg(noise)+.02
    horizontal=(avg(uy)-avg(ux))/strength
    linedraw=(avg(np.abs(boxline(rawdx,1,5)))+avg(np.abs(boxline(rawdy,2,5))))/strength
    line=between(linedraw,.47,.88); rip=avg(np.abs(boxline(rawdy,2,4))[:,-9:])/(avg(noise[:,-9:])+.024)
    yy=g.mean(2); midblack=between(yy[:,-4:].mean(1)-yy[:,19:28].min(1),.04,.26)
    shotsky=skyhi*(1-skylo); bluesea=waterlo*(.30+.96*skyhi)+.42*waterlo*(1-watertop)
    close=1-isolate; complexion=np.clip(plain-.35,-.35,.65); fog=avg(icy[:,:7])
    # Work in contrasts: a picture with neither affirmative nor contrary
    # evidence should not acquire a large score just from adding weak cues.
    # Colour match to the boundary is a soft measure of foreground occupancy.
    occ=1-np.exp(-sal*.67)
    wy=np.exp(-((np.arange(12,dtype=np.float32)-5.5)/4.5)**2)
    wgt=wy[:,None]*wy; mass=avg(occ*wgt)/wgt.mean()
    outer=.5*(avg(occ[:,:3])+avg(occ[:,-3:])); big=clip(mass*1.75-outer*.55)
    horconc=avg(occ[:,4:8])-avg(occ[:,1:4])*.5-avg(occ[:,8:11])*.5
    # Search for narrow leg-like vertical ridges of a consistent colour.
    leg=smooth(x); sides=(shift(g,0,-2)+shift(g,0,2))*.5
    rid=g-sides; cont=np.maximum(np.abs(blurbox(rid,1)),0)
    spine=np.abs(boxline(rid,1,2))/(boxline(np.abs(rid),1,2)+.026)
    shaft=between(np.abs(rid),.042,.22)*spine
    shaft=shaft[:,18:29,3:29].max(1); legs=0
    for dist in (6,10,14): legs=np.maximum(legs,np.sqrt(shaft[:,:-dist]*shaft[:,dist:]).max(1))
    # Fur gets directional fine ridges, rough skin gets isolated spots of
    # either sign. Check this inside the putative subject, not over sky.
    orig=g[:,7:26,6:26]; fine=g-smooth(g); mx,my=diff(g); tx=blurbox(mx*mx-my*my,2); ty=blurbox(2*mx*my,2)
    coherence=np.sqrt(tx*tx+ty*ty)/(blurbox(mx*mx+my*my,2)+.0014)
    fur=avg(coherence[:,7:26,6:26]); grain=avg(np.abs(fine[:,7:26,6:26]))
    wheels,df,cf=pairparts(g); turf=botgreen*(1-.8*cgreen)
    def C(z,origin): return z-origin
    # Natural kinds vs manufactured objects. Darkness below the object,
    # straight contours, sky and bare pavement are independent kinds of evidence.
    art=1.35*C(line,.5)+.8*C(horizon,.26)+1.55*C(skyhi,.22)+1.3*C(ro,.30)+.85*C(midblack,.35)-1.9*C(leafy,.20)-1.05*C(ochre,.22)+.90*horizontal+.55*C(paint,.12)
    # Local positive support within each family of objects. Center all
    # broadly active cues at a neutral/ambiguous value before combining.
    a=2.0*C(skyedges,.18)+1.2*C(skylo*skyhi,.08)+1.55*C(coolall,.16)+1.35*C(flat,.40)+1.7*C(isolate,.13)-.85*C(horizon,.28)-.95*horizontal-.76*C(midblack,.32)-.95*C(line,.48)+.6*C(topflat,.35)-.75*C(waterlo,.16)
    c=1.45*C(midblack,.32)+1.32*C(ro,.29)+1.0*C(paint,.12)+.72*C(wheels,.55)+.53*C(dark,.21)-1.10*C(skyhi,.25)-.95*C(waterlo,.16)-.72*C(leafy,.21)+.62*C(big,.47)+.52*C(horconc,.05)
    tr=.90*C(line,.48)+.78*C(ro,.29)+.92*C(shotsky,.14)+.67*C(white,.13)+.65*C(midblack,.32)+.42*C(wheels,.55)+.63*C(big,.48)-1.53*C(waterlo,.15)-.69*C(skylo,.18)-.68*C(leafy,.21)-.60*C(flat,.38)
    sh=2.9*C(bluesea,.14)+1.00*C(skyhi,.22)+1.55*C(horizon,.28)+1.85*horizontal+.75*C(rip,.27)+.57*C(flat,.38)-1.05*C(leafy,.20)-1.0*C(ochre,.19)-.93*C(midblack,.32)-.65*C(paint,.11)-.73*C(ro,.30)
    bi=1.7*C(flat,.40)+1.7*C(isolate,.14)+1.15*C(topgreen,.19)+.84*C(coolall,.15)+.6*C(paint,.12)-1.45*C(big,.50)-.85*C(midblack,.33)-.80*C(line,.47)-.7*horizontal-.40*C(turf,.18)
    ca=.75*C(muted,.42)+.53*C(gray,.42)+.57*C(cf,.35)+.85*C(fur,.32)+.62*C(big,.49)-.90*C(botgreen,.25)-.80*C(skyhi,.23)-.56*C(coolall,.15)-.50*C(horizon,.26)-.6*C(turf,.17)-.52*C(dark,.17)
    do=.95*C(df,.35)+.95*C(white,.13)+.95*C(dark,.17)+.70*C(big,.48)+.77*C(fgcoat,.22)+.68*C(turf,.18)+.35*C(isolate,.12)-.97*C(cgreen,.21)-.95*C(skyhi,.23)-.75*C(coolall,.15)-.56*C(line,.5)
    de=1.7*C(turf,.17)+1.5*C(topgreen,.20)+1.18*C(leafy,.20)+.85*C(ochrelow,.25)+.85*C(muted,.43)-.90*C(big,.50)-.88*C(paint,.13)-.55*C(white,.13)-.65*C(skyhi,.23)-.64*C(midblack,.34)+.68*C(legs,.51)
    ho=2.05*C(turf,.17)+.65*C(botgreen,.25)+.74*C(shotsky,.14)+.62*C(horizon,.27)+.78*C(fgcoat,.23)+.48*C(big,.50)+.66*C(dark,.17)-.63*C(cgreen,.20)-.55*C(waterlo,.15)+1.25*C(legs,.51)+.45*C(line,.47)
    fr=1.66*C(ochre,.20)+1.53*C(cgreen,.21)+.92*C(avg(gold),.22)+1.05*C(between(border,.030,.081),.48)+.93*C(between(grain,.016,.054),.39)-1.14*C(skyhi,.23)-.95*C(flat,.40)-.83*C(horizon,.28)-.92*C(turf,.17)-.71*C(coolall,.15)-.82*C(fur,.32)-.48*C(white,.13)
    # A likelihood of seeing an open water *region*, not just a bright line.
    # Check both lower corners independently, including their colour spread.
    q1=x[:,:,-8:,:6].mean((2,3)); q2=x[:,:,-8:,-6:].mean((2,3))
    agreement=np.exp(-((q1-q2)**2).mean(1)/.019); bluelo=avg(blue[:,-6:])
    opensurface=clip(.50*bluelo+.23*botflat+.27*agreement)
    marine=(.68*skyhi+.32*topflat)*opensurface
    sh+=1.55*C(marine,.20)+1.0*C(bluelo,.22)-1.1*horizon*(1-opensurface)-.65*midblack*(1-bluelo)-.65*paint*(1-bluelo)
    a+=.9*(skyhi*skylo)**.5*(1-horizon)+.70*skyedges*isolate-.47*marine
    c+=.75*horconc+.64*midblack*(1-skyhi)+.46*paint*big-.59*marine
    tr+=.65*(1-horconc)*shotsky*line+.57*big*line-.50*marine
    # A dog's fur may be almost any shade. Give support for *combinations*
    # such as a dark nose on a pale muzzle or a pied coat, rather than count
    # a single dark eye or a single white patch as enough.
    mask,nose,coat,piebald=localmarks(x,g); hi,mid,isotropic=spectraltexture(g)
    doglook=mask*(.6+.4*big)*(.5+.5*(1-line))
    do+=1.52*doglook+.68*coat+.5*piebald+.47*nose*(1-skyhi)+.43*big*(1-cgreen)-.35*marine-.35*cgreen
    ca+=.40*(1-coat)*fur+.41*mid*isotropic-.44*doglook-.42*piebald-.38*ochre*cgreen
    neartexture=clip((avg(micro[:,7:26,6:26])-.012)*22)
    fr+=1.12*neartexture*isotropic+.69*hi*isotropic+.54*mid*ochre+.31*(1-skyhi)*cgreen-.55*coat-.38*botgreen*(1-cgreen)
    bi+=.8*isolate+.65*flat*(1-big)+.55*flat*topgreen+.55*topflat*botflat+.43*paint*(1-line)-.37*coat
    de+=.62*turf*(1-big)+.60*muted*(1-skyhi)*turf-.5*big*coat
    ho+=.83*big*turf+.61*legs*shotsky+.52*legs*turf+.40*legs*fgcoat+.34*coat*turf-.40*cgreen*leafy
    # Context should select a family, not overpower the internal object match.
    add=.52*np.clip(art,-1.80,2.20)
    return np.stack((a+add,c+add,bi-add,ca-add,de-add,do-add,fr-add,ho-add,sh+add,tr+add),1)

def scene(x,g,dx,dy,sal):
    r,bg,b=x[:,0],x[:,1],x[:,2]
    ma=x.max(1); mi=x.min(1); sat=ma-mi
    def A(a): return avg(a)
    def high(a): return A(a[:,:9])
    def low(a): return A(a[:,-8:])
    def rim(a): return .5*(A(a[:,:,0:5])+A(a[:,:,-5:]))
    # Estimate which pixels might depict sky, water, plants, road, soil or coat.
    den=.26+g
    blue=between((b-r)/den,.018,.19)*between((b-bg)/den,-.035,.049)
    green=between((bg-b)/den,.018,.17)*between((bg-r)/den,-.062,.078)
    yellow=between((r-b)/den,.070,.30)*between((bg-b)/den,.045,.23)*between((r-bg)/den,-.08,.13)
    warm=between((r-bg)/den,.027,.18)*between((r-b)/den,.047,.26)
    gray=1-between(sat/den,.076,.24)
    red=between((r-bg)/den,.20,.49)*between((r-b)/den,.13,.39)
    vivid=between(sat/den,.32,.63)
    wh=between(g,.60,.85)*gray
    black=between(g,.27,.095)
    noise=np.abs(dx)+np.abs(dy); mag=np.sqrt(dx*dx+dy*dy)
    rough=smooth(smooth(noise)); flat=np.exp(-rough/.044)
    sky=(blue*(.45+.55*between(g,.24,.60))+gray*between(g,.60,.85)*.78)*( .18+.82*flat)
    topsky=high(sky); botsky=low(sky); skyall=rim(sky)
    water=blue*(.55+.45*flat); sea=low(water); grasses=low(green); leaves=high(green)
    earth=A(yellow+warm*.40); woods=A(green); road=low(gray*(.30+.70*flat))
    # Give central, boundary-dissimilar colours greater say about coat/paint than background.
    u=(np.arange(32,dtype=np.float32)+.5-16)/16
    center=np.exp(-2.7*(u[:,None]**2+u[None,:]**2)); j=np.minimum((np.arange(32)*12)//32,11)
    weight=center*(.35+np.minimum(sal[:,j[:,None],j[None,:]],3)); area=weight.sum((1,2))
    def fg(a): return (a*weight).sum((1,2))/area
    fgr=fg(green); fw=fg(warm); fgray=fg(gray); fred=fg(red); fv=fg(vivid); fblue=fg(blue)
    white=fg(wh); ink=fg(black); redpaint=fg(red*between((bg-b)/den,.18,0))
    axisx=np.abs(dx); axisy=np.abs(dy); ene=A(mag)+.018
    horiz=(A(axisy)-A(axisx))/ene
    align=A(np.abs(axisy-axisx)-np.minimum(axisx,axisy)*.75)/ene
    longh=A(np.abs(boxline(dy,2,3)))/ene; longv=A(np.abs(boxline(dx,1,3)))/ene
    lines=(align-.33)*1.7+(longh+longv-.62)*1.65
    linbottom=low(np.abs(boxline(dy,2,3)))/(low(mag)+.024)
    topbottom=high(g)-low(g)
    edges=A(mag[:,7:25,6:26]); frame=.35*high(mag)+.30*low(mag)+.35*rim(mag)
    isolated=np.clip((edges-frame)/(frame+edges+.020),-.6,.8)
    outerflat=(high(flat)+low(flat)+rim(flat))*.3333
    rows=g[:,:,4:-4].mean(2); prof=boxline(rows[:,None,:],2,1)[:,0]
    darkband=np.minimum(prof[:,-4:].mean(1),prof[:,8:17].mean(1))-prof[:,18:28].min(1)
    shadow=between(darkband,.035,.24)
    # Distinguish green foreground on earth from a non-green animal on grass.
    field=grasses*(1-.66*fgr); wild=(leaves+grasses)*.5; pasture=field*(.65+.65*topsky)
    # Relative strength of fine detail inside versus a smooth outline.
    fine=A(np.abs(g-smooth(g)))/(A(mag)+.025)
    busy=between(frame,.029,.094); stipple=between(fine,.28,.60)
    paleground=low(wh); backdrop=(high(blue)+rim(blue))*.5
    # The straight waterline tends to cut horizontally through the picture;
    # the sky behind flying objects instead continues down the sides.
    split=topsky*(1-botsky); bothsky=(topsky*botsky)**.5
    wet=(.57+1.6*topsky)*sea
    mixed=(fg(black)*fg(wh))**.5
    ftexture=fg(rough)
    manufactured=1.70*lines+1.10*horiz+1.05*shadow+.80*road+.55*topsky-1.35*woods-.56*earth-.45*stipple
    groundanimal=1.75*field+.70*leaves+.55*yellow.mean((1,2))-.95*sea-.80*bothsky
    a= -.30+2.35*bothsky+.85*topsky+.9*backdrop+.85*skyall+.86*outerflat+.85*isolated-.47*shadow-.56*horiz-.62*busy-.40*woods-.42*earth-.24*wet
    c= -.27+manufactured+.55*road+.70*redpaint+.36*fv+.41*ink+.28*isolated-.45*topsky-.55*wet-.43*sea-.26*grasses
    d= .19+.75*leaves+.46*backdrop+.71*outerflat+.77*isolated+.57*fv+.19*fblue-.71*lines-.62*shadow-.45*horiz-.52*wet-.32*topbottom-.31*field
    t= .45+.37*fgray+.53*fw+.24*stipple+.37*busy+.19*white-.61*wild-.71*topsky-.51*sea-.73*lines-.44*horiz-.35*shadow-.34*isolated
    e= .17+groundanimal+.42*earth+.36*leaves+.31*fw-.55*fv-.85*redpaint-.25*mixed-.42*topsky-.44*lines-.31*shadow
    dog= .33+.63*fw+.36*white+.85*mixed+.44*field+.21*isolated-.65*fgr-.67*sea-.48*topsky-.61*lines-.32*shadow-.20*horiz
    f= .12+1.45*fgr+.82*earth+.78*busy+.73*stipple+.58*fw-.94*topsky-.57*blue.mean((1,2))-.55*shadow-.57*lines-.44*field-.67*outerflat-.30*white
    h= .13+groundanimal+.60*ink+.47*fw+.65*split+.50*topbottom+.37*shadow+.31*isolated-.48*fgr-.29*stipple-.20*lines
    s= -.20+2.20*wet+.98*topsky+.69*split+.93*horiz+.93*(linbottom-.29)+.22*isolated+.29*outerflat-.37*shadow-.82*woods-.61*earth-.33*redpaint
    tr= -.32+manufactured+.88*split+.75*topbottom+.60*white+.45*(longv-.26)+.36*road-.48*sea-.43*wet-.32*outerflat
    return np.stack((a,c,d,t,e,dog,f,h,s,tr),1)

def classify(x):
    xx=enlarge(x); rgb=pool(xx)
    g=.27*x[:,0]+.55*x[:,1]+.18*x[:,2]
    y=.27*xx[:,0]+.55*xx[:,1]+.18*xx[:,2]
    dx,dy=diff(smooth(y)); h=angles(dx,dy)
    for opponent in (xx[:,0]-xx[:,1],xx[:,2]-xx[:,1]):
        gx,gy=diff(smooth(opponent)); h+=.38*angles(gx,gy)
    h=pool(smooth(h)); h=np.sqrt(h+.0015)-np.float32(np.sqrt(.0015)); h-=h.mean((2,3),keepdims=True); h=norm(h,.02)
    sal=borderdiff(rgb); ysmall=pool(y); val=stretch(ysmall).reshape(len(x),-1); salvec=stretch(sal).reshape(len(x),-1)
    response=10.5*(h.reshape(len(x),-1)@CONSTANTS['h'])
    m=CONSTANTS['m']; lumin=np.abs(val@m)
    # A bird against a similarly bright leaf can have an almost invisible
    # luminance boundary. Treat either opponent axis as an alternative.
    chrom=[]
    for aa,bb in ((0,1),(1,2)):
        d=rgb[:,aa]-rgb[:,bb]; d-=d.mean((1,2),keepdims=True)
        chrom.append(np.abs(norm(d,.60).reshape(len(x),-1)@m))
    response+=2.3*np.maximum(lumin,np.maximum(chrom[0],chrom[1]))+2.25*(salvec@m)+3.25*(val@CONSTANTS['p'])
    scores=[]; start=0
    for end in CONSTANTS['ends']:
        sub=response[:,start:end]; best=sub.max(1)
        # Pool over nearby explanations; avoid relying entirely on one accidental alignment.
        scores.append(best+.29*np.log(np.mean(np.exp((sub-best[:,None])*2.4),axis=1)))
        start=end
    dx,dy=diff(g)
    shape=np.stack(scores,1)
    # Shape is useful, but a local accidental alignment is not by itself
    # good evidence against the rest of the picture.
    # Pool contextual rules independently of how many there are for a class.
    # The second pass emphasizes agreement rather than stacking near-duplicates.
    return .87*shape+.39*scene(x,g,dx,dy,sal)+1.27*habitat(x,g,sal)

def predict(x):
    # Fixed matrix dimensions also for the last chunk, so the numerical work
    # on one item does not depend on the caller's choice of batch size.
    result=np.empty(len(x),np.int64)
    for i in range(0,len(x),32):
        b=np.zeros((32,3,32,32),np.float32); n=min(32,len(x)-i); b[:n]=x[i:i+n]
        result[i:i+n]=classify(b)[:n].argmax(1)
    return result
'''

if __name__=='__main__':
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(shared+solution)
    np.savez(out/'constants.npz',**generate())
