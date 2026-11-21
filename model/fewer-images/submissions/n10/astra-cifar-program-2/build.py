"""Data-free builder. All geometry is in arbitrary object-local modelling units."""
import math
import sys
from pathlib import Path
import numpy as np

# Features are shared between the renderer (one descriptor per view) and inference.
FEATURES = '''import numpy as np
import math

def blur(a):
    p = np.pad(a, ((0,0),(1,1),(1,1)), mode="edge")
    t = p[:,:,:-2] + 2*p[:,:,1:-1] + p[:,:,2:]
    return (t[:,:-2] + 2*t[:,1:-1] + t[:,2:]) * .0625

def deriv(a):
    p = np.pad(a, ((0,0),(1,1),(1,1)), mode="edge")
    dx = (p[:,:-2,2:]-p[:,:-2,:-2] + 2*(p[:,1:-1,2:]-p[:,1:-1,:-2]) + p[:,2:,2:]-p[:,2:,:-2])*.125
    dy = (p[:,2:,:-2]-p[:,:-2,:-2] + 2*(p[:,2:,1:-1]-p[:,:-2,1:-1]) + p[:,2:,2:]-p[:,:-2,2:])*.125
    return dx, dy

def pool(a,k):
    return a.reshape(len(a),32//k,k,32//k,k).mean((2,4))

def norm(a):
    a = a.reshape(len(a),-1)
    return a / np.sqrt(np.sum(a*a,axis=1,keepdims=True)+1.e-8)

def smoothpool(a,k):
    return pool(blur(a),k)

def pos(a):
    return np.maximum(0,a)

def unit(a):
    return np.clip(a,0,1)

def avg(a):
    return a.mean((1,2))

def features(rgb):
    r,g,b = rgb[:,0],rgb[:,1],rgb[:,2]
    l = .30*r+.55*g+.15*b
    dx,dy = deriv(l)
    hh=[]
    ss=[]
    for scale in range(2):
        if scale:
            dx,dy=deriv(blur(blur(l)))
        mag=np.sqrt(dx*dx+dy*dy+1.e-10)
        k=4 if scale==0 else 8
        m=smoothpool(mag,k)
        den=np.sqrt((.018 + .38*m.mean((1,2),keepdims=True) + .65*m)*(.025 + m.mean((1,2),keepdims=True)))
        hh.append(norm((m-m.mean((1,2),keepdims=True))/den)*.45)
        hhpart=[]
        for i in range(6):
            theta=i*math.pi/6
            c=(dx*math.cos(theta)+dy*math.sin(theta))/mag
            q=c*c
            f=smoothpool(q*q*q*mag,k)
            hhpart.append((f-.27*m)/den)
        hh.append(norm(np.stack(hhpart,1))*(1 if scale==0 else .75))
        ss.append(norm(np.stack([smoothpool(dx,k)/den,smoothpool(dy,k)/den],1))*(.70 if scale==0 else .55))
    ll=pool(blur(l),4)
    ss.append(norm(ll-ll.mean((1,2),keepdims=True))*1.05)
    value=np.maximum(np.maximum(r,g),b)
    low=np.minimum(np.minimum(r,g),b)
    # Soft colour categories: foliage, warm earth/fur, blue, vivid red, dark/neutral/light.
    chrom=value-low
    d=.14+.54*value
    gr=unit((g-r+.015)/d*3)*unit((g-b+.025)/d*3)
    earth=unit((r-b)/d*2.5)*unit((r-g+.045)/d*3)
    blue=unit((b-r+.015)/d*3)*unit((b-g+.055)/d*3)
    red=unit((r-g)/d*2.5)*unit((r-b)/d*2)
    gold=unit((g-b)/d*2.8)*unit((r-b)/d*2.8)
    neutral=np.exp(-2.2*chrom/d)
    colours=[gr,earth,blue,red,gold,neutral*unit((.52-l)*3),neutral*unit((l-.40)*2.7),unit((chrom-.12)*2.8)]
    cc=[]
    for v in colours:
        cc.append(pool(v,8)-.18)
    cc.append(pool(neutral,8)-.5)
    cc.append(.52*(pool(l,8)-.5))
    return norm(np.concatenate(hh,1)), norm(np.concatenate(ss,1)), norm(np.stack(cc,1)), (l,dx,dy,gr,earth,blue,red,gold,neutral)
'''
ns = {}
exec(FEATURES,ns)
features = ns['features']

# Projected ellipsoids and planar facets with pixelwise depth buffering.
# Each rendered scene is completely independent of all other scenes.
R = np.random.default_rng(283751)
def U(a,b): return R.uniform(a,b)
def P(p): return R.random()<p
def choose(items): return items[R.integers(len(items))]
def pick(s): return choose(s.split())
def col(s):
    if isinstance(s,str): return np.array([int(s[i:i+2],16) for i in (0,2,4)],np.float32)/255
    return np.asarray(s,dtype=np.float32)
def blend(a,b,t): return (1-t)*col(a)+t*col(b)
def varied(s,f=.12): return np.clip(col(s)*U(1-f,1+f)+U(-f/4,f/4),0,1)
YY,XX=np.mgrid[:32,:32].astype(np.float32)
I=np.eye(3)
def noise(n,m=None):
    if m is None: m=n
    grid=R.normal(0,1,(n+1,m+1)).astype(np.float32)
    x=(XX+.5)*(m/32); y=(YY+.5)*(n/32)
    i=x.astype(int);j=y.astype(int); x-=i; y-=j
    x=x*x*(3-2*x);y=y*y*(3-2*y)
    return (1-y)*((1-x)*grid[j,i]+x*grid[j,i+1])+y*((1-x)*grid[j+1,i]+x*grid[j+1,i+1])

class Scene:
    def __init__(self,kind,mode,extent,centre,az,el,roll,span,texture):
        self.scene=mode
        # Independent illumination, camera and background for this single template.
        ca,sa=np.cos(az),np.sin(az); ce,se=np.cos(el),np.sin(el)
        h=np.array([ca,0,-sa]); v=np.array([sa*se,-ce,ca*se]); d=np.array([sa*ce,se,ca*ce])
        cr,sr=np.cos(roll),np.sin(roll)
        self.mat=np.array([cr*h-sr*v,sr*h+cr*v,d])
        ext=np.asarray(extent)
        sizes=np.abs(self.mat[:2])@ext
        self.scale=span/max(sizes)
        self.mat*=self.scale
        self.shift=np.array([15.5+U(-2.7,2.7),15.5+U(-2.5,2.5),0.])-self.mat@centre
        self.light=np.array([U(-.65,.65),-U(.4,.9),U(.25,.8)])
        self.light/=np.linalg.norm(self.light)
        self.ambient=U(.67,.9); self.diffuse=U(.23,.48)
        self.depth=np.full((32,32),-1.e5,np.float32)
        self.paint=np.zeros((32,32,3),np.float32)
        self.texture=texture
        self.background(mode,kind)
    def background(self,mode,kind):
        sky=pick('77acd1 94c4e3 7f9bb3 c6d8e4 c3c9ca e1e4e3 c5c7c0 6388ad a2bad4')
        grass=pick('627547 75884c 7e8847 556b3d 4d5a38 a49e64 677446 838457 a39563')
        wood=pick('504632 655e42 625742 6b754c 574737 7c7056 8c846c 434f35')
        leaf=pick('365137 495d32 4e6740 738953 30482e 657348 6c6d43 384a30')
        road=pick('777976 737572 9d9c95 9d9589 696867 b0b0a9 84817e 555955 b7afa1')
        water=pick('537c93 546b76 56747d 4a7394 687e93 6e8a91 879c9f 4d6774 839caa 3d627d 617169')
        indoor=pick('8a7769 ae9b83 b0a899 b8b1a3 716b67 797468 6c665e 968378 c8b7a1')
        wall=pick('ababa4 b9b6a8 b3bbc0 8e8981 858b8a 9c8777 787874 bdb8af')
        neutral=pick('ecece8 d7d5d0 b0aba4 3c3d3b 797979 8c99a2 b8babb')
        if mode=='air': hi,lo=sky,sky; line=U(-18,50); th,tl=U(.012,.076),U(.012,.076)
        if mode=='sea': hi,lo=sky,water; line=U(9,22); th,tl=.043,.073
        if mode=='port': hi,lo=choose([wall,sky,leaf]),water;line=U(13,23);th,tl=.12,.071
        if mode=='street': hi,lo=choose([wall,leaf,wood,sky]),road;line=U(13,23);th,tl=U(.06,.16),U(.016,.072)
        if mode=='tarmac': hi,lo=sky,road;line=U(11,23);th,tl=.046,.043
        if mode=='field': hi,lo=choose([leaf,grass,wood,sky]),grass;line=U(4,18);th,tl=.10,U(.05,.13)
        if mode=='wood': hi,lo=choose([leaf,wood,grass]),choose([wood,grass,leaf]);line=U(3,25);th,tl=.14,.13
        if mode=='interior': hi,lo=choose([wall,indoor,neutral]),choose([indoor,wood,wall]);line=U(8,26);th,tl=.085,.10
        if mode=='pond': hi,lo=choose([leaf,wood,grass]),choose([wood,grass,water,leaf]);line=U(5,24);th,tl=.105,.11
        if mode=='plain': hi=lo=neutral;line=0;th=tl=U(.007,.05)
        if mode=='mixed': hi=choose([wood,wall,sky,grass,neutral,indoor]);lo=choose([road,water,wood,neutral,grass]);line=U(6,28);th=tl=.11
        if mode=='air': lo=hi
        top=varied(hi);bottom=varied(lo) if mode not in ('air','plain') else top
        coarse=noise(3,4); med=noise(9,9); fine=noise(25,27)
        tt=coarse*.75+med*.48+fine*.23
        bt=tt if mode not in ('sea','port','air','tarmac') else .48*noise(5,3)+.7*noise(29,5)
        if hi in [wall,indoor] and P(.65): tt+=.55*np.sign(noise(3,6))
        if hi in [leaf,wood,grass]: tt+=.40*fine+.45*np.tanh(2*noise(5))
        if lo in [wood,grass,leaf]: bt=bt+.45*fine+.27*np.tanh(3*med)
        line=line+(XX-16)*U(-.10,.10)+U(0,2.7)*noise(1,5)
        mix=np.clip(.5+(YY-line)/U(.8,6),0,1)[:,:,None]
        self.bg=(1-mix)*(top+th*tt[:,:,None])+mix*(bottom+tl*bt[:,:,None])
        self.grain=.45*coarse+.52*med+.40*fine+.24*R.normal(size=(32,32))
    def proj(self,p): return self.mat@p+self.shift
    def bounds(self,x0,x1,y0,y1):
        xa=max(0,int(math.ceil(x0))); xb=min(32,int(math.floor(x1))+1)
        ya=max(0,int(math.ceil(y0))); yb=min(32,int(math.floor(y1))+1)
        if xa>=xb or ya>=yb: return None
        return (slice(ya,yb),slice(xa,xb))
    def ell(self,p,r,c,axis=None):
        # Covariance of the projected ellipsoid and its depth conditional on screen x,y.
        t=self.mat if axis is None else self.mat@axis
        m=t*np.asarray(r); q=m@m.T
        x,y,z=self.proj(p); rx,ry=math.sqrt(q[0,0]),math.sqrt(q[1,1]); s=self.bounds(x-rx,x+rx,y-ry,y+ry)
        if s is None: return
        a,b=q[0,:2]; f=q[1,1]; det=max(1e-12,a*f-b*b)
        a1,b1,f1=f/det,-b/det,a/det
        xx,yy=XX[s]-x,YY[s]-y
        vx=a1*xx+b1*yy; vy=b1*xx+f1*yy
        z0=q[2,0];z1=q[2,1];lx=z0*a1+z1*b1;ly=z0*b1+z1*f1
        zz=np.sqrt(np.maximum(0,1-xx*vx-yy*vy));rad=math.sqrt(max(1e-12,q[2,2]-lx*z0-ly*z1))
        depth=z+lx*xx+ly*yy+rad*zz
        mask=(zz>0)&(depth>self.depth[s]); self.depth[s][mask]=depth[mask]
        nx=rad*vx-lx*zz; ny=rad*vy-ly*zz
        shading=self.ambient+self.diffuse*np.maximum(0,(self.light[0]*nx+self.light[1]*ny+self.light[2]*zz)/np.sqrt(nx*nx+ny*ny+zz*zz+1e-8))
        self.paint[s][mask]=shading[mask,None]*col(c)
    def facet(self,pp,c):
        vv=np.asarray(pp)@self.mat.T+self.shift; x=vv[:,0];y=vv[:,1];z=vv[:,2]
        s=self.bounds(x.min(),x.max(),y.min(),y.max());
        if s is None: return
        u=vv[1]-vv[0];v=vv[-1]-vv[0];n=np.cross(u,v); nz=n[2]
        if abs(nz)<1.e-7:return
        xx,yy=XX[s],YY[s]; depth=z[0]-(n[0]*(xx-x[0])+n[1]*(yy-y[0]))/nz
        mask=depth>self.depth[s]-.0002
        for i in range(len(vv)):
            j=i-1
            side=(x[i]-x[j])*(yy-y[j])-(y[i]-y[j])*(xx-x[j]); mask &= (side>=-.0001) if nz>0 else (side<=.0001)
        self.depth[s][mask]=depth[mask]; bright=self.ambient+self.diffuse*max(0,np.sign(nz)*(self.light@n)/(np.linalg.norm(n)+1e-8))
        self.paint[s][mask]=col(c)*bright
    def profile(self,outline,w,c):
        # Extruded polygon in the vehicle's longitudinal plane. Outline is convex.
        a=[(x,y,-w) for x,y in outline]; b=[(x,y,w) for x,y in outline]
        self.facet(a,c);self.facet(b,c)
        for i in range(len(a)): self.facet([a[i-1],a[i],b[i],b[i-1]],c)
    def box(self,x0,x1,y0,y1,w,c): self.profile([(x0,y0),(x1,y0),(x1,y1),(x0,y1)],w,c)
    def rod(self,p1,p2,a,b,c):
        v=np.asarray(p2)-p1; length=np.linalg.norm(v)
        if length<1.e-6:return
        e=v/length; f=np.cross(e,[0,0,1] if abs(e[2])<.9 else [1,0,0]); f/=np.linalg.norm(f); h=np.cross(e,f)
        self.ell(.5*(np.asarray(p1)+p2),(length*.53+a*.40,a,b),c,np.column_stack((e,f,h)))
    def shadow(self,centre,rx,rz):
        if self.scene in ('plain','air'): return
        p=self.proj(centre); m=self.mat[:2,[0,2]]*np.array([rx,rz]);q=m@m.T
        a,b=q[0];d=q[1,1]+1.5; xx=XX-p[0];yy=YY-p[1]; dis=(d*xx*xx-2*b*xx*yy+a*yy*yy)/max(.001,a*d-b*b)
        self.bg*=1-U(.22,.53)*np.exp(-1.7*dis)[:,:,None]
    def finish(self):
        obj=self.paint+self.texture*self.grain[:,:,None]
        tex=np.where((self.depth>-9.e4)[:,:,None],obj,self.bg)
        # Downsampling blur, mild camera colour cast and exposure.
        tex*=np.array([U(.93,1.07),U(.96,1.04),U(.92,1.08)])
        tex=(tex-.45)*U(.78,1.22)+U(.36,.54)
        p=np.pad(tex,((1,1),(1,1),(0,0)),mode='edge'); a=U(.05,.16)
        return np.clip((1-4*a)*tex+a*(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]),0,1).transpose(2,0,1).astype(np.float32)

paintcols='c9cace dbdbe0 e6e3d9 979a9d 505356 25272a 353b45 526276 395275 814042 b13b34 a78555 3f5963'
furcols='a77c52 98714e 766150 b49b79 826349 9e815e'
black='252426'; tyre='29292b'; metal='8c8e90'

def wheel(s,x,y,w,rad,dual=False):
    for side in (-1,1):
        z=side*w
        s.ell((x,y,z),(rad,rad,rad*.40),varied(tyre,.13))
        s.ell((x,y,z+side*rad*.33),(rad*.57,rad*.57,rad*.12),varied(metal,.32))
        if dual: s.ell((x,y,z+side*rad*.43),(rad*.22,rad*.22,rad*.045),tyre)

def automobile(s,style):
    w=U(.70,.86); c=varied(pick(paintcols));glass=varied(pick('394952 455761 657782 393e44 2c373e 738791 525b63'))
    chassis=[(-2,.49),(-1.92,.34),(1.85,.34),(2.01,.49),(1.96,.77),(1.67,.91),(.75,1.01),(-1.63,1.00),(-1.99,.82)]
    s.shadow((0,.09,0),2.28,1.12);s.profile(chassis,w,c)
    rear=U(-1.60,-1.22); front=U(.83,1.13); roofh=U(1.44,1.64);rf=U(.31,.57);rr=U(-1.05,-.72)
    if style==1: rear=-1.83;rr=-1.62;roofh+=.18;rf=.57;front=1.29
    if style==2: rear=-1.9;rr=-1.82;roofh+=.37;rf=.79;front=1.59
    if style==3: rear=-1.25;rr=-.63;roofh-=.12;front=1.08
    base=.93
    p=[(rear,base),(front,base),(rf,roofh),(rr,roofh)]
    lower=[(rear,base,-w*.98),(front,base,-w*.98),(front,base,w*.98),(rear,base,w*.98)]
    upper=[(rr,roofh,-w*.79),(rf,roofh,-w*.79),(rf,roofh,w*.79),(rr,roofh,w*.79)]
    for i in range(4):
        panel=np.array([lower[i-1],lower[i],upper[i],upper[i-1]]); s.facet(panel,c)
        mid=panel.mean(0);window=mid+.86*(panel-mid)
        n=np.cross(panel[1]-panel[0],panel[3]-panel[0]);n/=np.linalg.norm(n);window+=.007*n*np.sign(n[1]+.000001)
        s.facet(window,glass)
        if i in (1,3):
            weight=U(.43,.53);u=window[0]*(1-weight)+window[1]*weight;v=window[3]*(1-weight)+window[2]*weight
            s.rod(u,v,.038,.044,blend(c,black,.48))
    s.facet(upper,c)
    for q in (-1,1):
        s.ell((1.91,.76,q*w*.75),(.09,.105,.20),'dedac4');s.ell((-1.97,.71,q*w*.70),(.066,.11,.22),'a83c34')
        s.ell((.62,1.07,q*w*1.05),(.14,.082,.10),c)
    s.facet([(2.007,.50,-.38),(2.007,.50,.38),(1.978,.69,.43),(1.978,.69,-.43)],'4a4845')
    wr=U(.32,.41);wheel(s,-1.27,wr,w*.95,wr);wheel(s,1.26,wr,w*.95,wr)

def truck(s,style):
    w=U(.58,.72); cab=varied(pick(paintcols+' c5af45 b7332e e1ddd4 b8c9ce'));cargo=varied(pick('d6d6cd bdbec4 b7b3a9 c6c2b3 bbcbd3 788897 7c5251 b99a67 466887 a75942'))
    glass=varied(pick('32404c 4c6170 758695 343f45 567887')); s.shadow((0,.04,0),2.8,1.02);s.box(-2.57,2.35,.32,.56,w*.86,'433f3c')
    joint=U(.52,1.03); top=U(1.81,2.34); low=.64
    if style==2: top=U(1.08,1.45); cargo=varied(pick('706956 af9357 696f6a 844c3c'))
    if style==3:
        s.ell((-.92,1.27,0),(1.73,.67,w),cargo);s.box(-2.49,joint,.62,.79,w,cargo)
    else:
        s.box(-2.62,joint,low,top,w,cargo)
        for side in [-1,1]:
            z=side*(w+.006)
            s.facet([(-2.58,low+.04,z),(joint-.02,low+.04,z),(joint-.02,low+.095,z),(-2.58,low+.095,z)],blend(cargo,'524c45',.35))
    nose=2.53; flat=style!=1; ctop=U(1.64,2.04)
    lead=nose if flat else 1.96
    s.profile([(joint+.05,.49),(lead,.49),(lead,1.13),(lead-.18,ctop),(joint+.05,ctop)],w*.96,cab)
    if not flat:s.box(lead-.10,nose,.50,1.08,w*.88,cab)
    s.facet([(lead+.002,1.15,-w*.80),(lead+.002,1.15,w*.80),(lead-.146,ctop-.1,w*.80),(lead-.146,ctop-.1,-w*.80)],glass)
    for side in (-1,1):
        z=side*(w*.96+.003);s.facet([(joint+.18,1.14,z),(lead-.11,1.14,z),(lead-.27,ctop-.10,z),(joint+.18,ctop-.10,z)],glass)
        s.ell((nose,.85,side*w*.67),(.065,.085,.13),'e1dfc9')
    s.facet([(nose+.004,.57,-w*.56),(nose+.004,.57,w*.56),(nose+.004,1.02,w*.56),(nose+.004,1.02,-w*.56)],'726f69')
    rad=U(.31,.39); wheel(s,-1.92,rad,w*.94,rad,True);wheel(s,1.91,rad,w*.91,rad,True)
    if P(.76):wheel(s,-1.19,rad,w*.94,rad)

def plane(s,style):
    # Fuselage, swept main planes, stabilisers, fin and optional engine nacelles.
    c=varied(pick('d1d5d6 ebebe5 a0a4a6 bcbfc2 aeb7bd 81888b 6d7d83 797c73 626566 e0d9c7'))
    trim=varied(pick('546b89 394c65 914942 aa4745 a1a1a0 7f858b b9b9b1'));g='43535e';rad=.20 if style!=2 else .28
    s.ell((0,0,0),(2.29,rad,rad*.97),c)
    sweep=.77 if style==0 else (.22 if style==1 else 1.14); wide=2.00 if style!=2 else 1.36
    for q in (-1,1):
        s.facet([(.57,-.025,.08*q),(.42-sweep,.08,wide*q),(.13-sweep,.07,(wide+.02)*q),(-.59,-.055,.07*q)],blend(c,'888b8d',.17))
        s.facet([(-1.27,.055,q*.06),(-1.84,.13,wide*.44*q),(-2.08,.13,wide*.43*q),(-1.89,.08,q*.06)],c)
        if style==0:
            s.ell((.06,-.17,q*.82),(.39,.13,.14),blend(c,trim,.23));s.ell((.37,-.18,q*.82),(.08,.095,.10),g)
    s.facet([(-1.17,.13,0),(-1.86,.79,0),(-2.07,.80,0),(-1.99,.09,0)],trim)
    s.ell((1.50,rad*.58,0),(.37,rad*.48,rad*.74),g)
    if style==0:
        for q in [-1,1]:s.rod((-1.42,.046,q*rad*.72),(1.43,.048,q*rad*.72),.031,.048,blend(c,trim,.70))
    if style==1:s.ell((2.19,0,0),(.04,.56,.049),'454748')

def ship(s,style):
    hull=varied(pick('3b4651 535b60 334859 6b7173 565a59 875e4d cccbc1 e0dfd3 c7c8c2'));deck=varied(pick('d9d8ca b6bec0 a9abad e0dfd7 d5c9ae')); glass=varied(pick('405464 788c95 455357'))
    w=U(.42,.67); end=2.45
    rim=np.array([(-2.18,.53,-w*.76),(-1.84,.56,-w),(1.53,.53,-w),(end,.70,0),(1.53,.53,w),(-1.84,.56,w),(-2.18,.53,w*.76)])
    keel=rim.copy();keel[:,1]=.02;keel[:,0]*=.88;keel[:,2]*=.75
    s.facet(rim,blend(hull,deck,.6))
    for j in range(len(rim)):s.facet([keel[j-1],keel[j],rim[j],rim[j-1]],hull)
    if style==0 or style==1:
        h=1.10 if style==0 else .89;front=.82 if style==0 else .34
        s.box(-1.42,front,.55,h,w*.84,deck)
        for side in (-1,1):
            z=side*(w*.84+.004);s.facet([(-1.27,h-.22,z),(front-.12,h-.22,z),(front-.12,h-.07,z),(-1.27,h-.07,z)],glass)
        s.box(-1.11,.28,h,h+.22,w*.70,deck);s.box(-.96,-.62,h+.22,h+.53,w*.35,hull)
        s.rod((-.18,h+.22,0),(-.20,h+.76,0),.031,.032,glass)
    if style==2:
        s.rod((0,.55,0),(0,3.05,0),.035,.035,glass)
        s.facet([(-1.44,.79,0),(-.02,2.99,0),(-.034,.80,.12)],'e0ded0');s.facet([(.12,2.74,0),(1.73,.75,0),(.11,.84,.11)],'cbc9bb')
    if style==3:
        s.box(-1.68,-.97,.56,1.53,w*.72,deck);s.box(-1.80,-.91,1.32,1.53,w*.84,deck)
        for j in range(4):s.box(-.80+j*.54,-.32+j*.54,.56,U(.80,1.06),w*.83,varied(pick('99624d 576665 686e7a 967653 a1886d')))
    if style==4:
        s.profile([(-1.07,.54),(.94,.54),(.56,.90),(-.70,.9)],w*.72,deck);s.profile([(-.64,.90),(.34,.90),(.10,1.19),(-.62,1.19)],w*.58,glass)

# Animal heads are local assemblies of volumes. The same model can be viewed
# from the side, in three quarters, head-on, or in a tightly framed portrait.
def petcol(kind):
    if kind==3:return varied(pick('b49a80 8f857a aca394 bf945d c09467 8b8071 838688 555651 484744 3c3936 c9c4b7 d8d5cf 97836b b9a68c'))
    if kind==5:return varied(pick('b79a71 cab18b b69062 9b7854 ab8d69 705e4e 50453e 494744 383632 e0dbcd d1c4a9 85796b a66c45'))
    if kind==4:return varied(pick('997c58 96795d a18969 b49a74 897460 746451 a68a6a 9e7f59'))
    return varied(pick('996b4f 8b5b42 966347 7e5943 674936 664a39 453d35 443c36 ba9771 d0c6b5 8e8274'))

def ear(s,base,tip,width,depth,c,inner):
    a=np.asarray(base);b=np.asarray(tip);da=np.array([-.06,0,-width]);db=np.array([.055,0,width]);back=np.array([-depth,-.035,0.])
    s.facet([a+da,b,a+db],c);s.facet([a+db,b,a+back],c);s.facet([a+back,b,a+da],c)
    o=(a+da+b+a+db)/3.; triangle=np.array([a+da,b,a+db]);s.facet(o+.66*(triangle-o)+np.array([.006,0,0]),inner)

def eye(s,p,r,c,side,cateye=False):
    s.ell(p,(r*.87,r*.76,r),c)
    if cateye:
        s.ell(tuple(np.asarray(p)+[r*.45,0,side*r*.44]),(r*.41,r*.62,r*.36),'292723')
    s.ell(tuple(np.asarray(p)+[r*.52,r*.36,side*r*.54]),(r*.21,r*.21,r*.18),'e0dace')

def pethead(s,kind,h,c,style):
    def pt(x,y,z):return (h[0]+x,h[1]+y,h[2]+z)
    dark=blend(c,'302c28',U(.52,.91));light=blend(c,'e8dfc9',U(.17,.50)); nose=varied(pick('393532 302d2a 51443c'))
    if kind==3:
        s.ell(h,(.37,.40,.40),c); eyeCol=pick('a8ab7c a39d74 829b81 8f9675')
        for q in (-1,1):
            ear(s,pt(-.05,.22,.235*q),pt(-.12,.63,.35*q),.15,.15,c,blend(c,'9f7a71',.46))
            s.ell(pt(.24,-.105,.113*q),(.17,.138,.18),light)
            eye(s,pt(.295,.077,.235*q),.074,eyeCol,q,True)
            if style<2:
                for j in range(3):s.rod(pt(.20-j*.095,.22+j*.035,.248*q),pt(.27-j*.085,.15+j*.024,.266*q),.023,.027,dark)
                s.rod(pt(.04,-.035,.389*q),pt(.245,-.062,.31*q),.022,.027,dark)
        s.ell(pt(.397,-.050,0),(.052,.048,.077),pick('866661 65534e 5b514e'));s.ell(pt(.27,-.208,0),(.12,.091,.13),light)
    if kind==5:
        s.ell(h,(.36,.385,.34),c);muzzle=.27 if style!=3 else .155
        s.ell(pt(.26,-.105,0),(muzzle,.17,.21),light if style!=1 else dark)
        for q in (-1,1):
            if style in (0,1,3):
                s.rod(pt(-.11,.18,.263*q),pt(-.065,-.285,.388*q),.14,.117,dark if style!=3 else c)
            else:ear(s,pt(-.10,.18,.198*q),pt(-.19,.59,.306*q),.14,.14,c,dark)
            eye(s,pt(.267,.085,.193*q),.065,black,q)
            if style==1:s.ell(pt(.238,.158,.219*q),(.060,.033,.049),light)
        s.ell(pt(.24,-.241,0),(muzzle*.76,.068,.164),dark);s.ell(pt(.25+muzzle*.90,-.032,0),(.08,.077,.134),nose)
        if P(.28):s.ell(pt(.31,-.32,0),(.12,.11,.11),'b47b77')

def pet(s,kind,style,portrait):
    cat=kind==3;c=petcol(kind); dark=blend(c,'38332e',.53);light=blend(c,'ded6c6',.52)
    if cat:ln,hi,wd=.79,.37,.32
    else:ln,hi,wd=.9,.45,.37
    sit=portrait or P(.36);back=(0,.74,0) if not sit else (-.20,.64,0)
    if sit:
        s.rod((-.38,.38,0),(.18,1.10,0),hi,wd,c);h=(.32,1.34,0);s.ell((-.33,.42,0),(.56,.37,wd*1.12),c)
    else:
        s.ell(back,(ln,hi,wd),c);h=(ln*.94,1.18 if cat else 1.34,0);s.rod((.48,.80,0),h,.238,.257,c)
    s.shadow((-.06,.025,0),1.27,.54)
    for q in (-1,1):
        z=wd*.62*q
        if sit: hind=-.47; fore=.30; y=.83
        else:hind=-ln*.69;fore=ln*.62;y=.76
        for x in [hind,fore]:
            step=U(-.17,.17); hip=(x,y,z);joint=(x+(-.12 if x<0 else .02),.39,z*1.18);paw=(x+step,.105,z*1.25)
            s.rod(hip,joint,.139,.13,c);s.rod(joint,paw,.081,.084,c);s.ell((paw[0]+.05,.082,paw[2]),(.146,.083,.091),light if P(.28) else c)
    start=(-ln*.88,.78 if not sit else .49,0);mid=(-ln*1.24,U(.64,1.12),.08);tip=(-ln*1.56,U(.62,1.56),.18)
    s.rod(start,mid,.108,.10,c);s.rod(mid,tip,.078,.077,c)
    if cat and style==0 and not sit:
        for j in range(5):
            x=-.49+j*.222
            for q in (-1,1):s.rod((x-.05,1.058,.10*q),(x+.08,.83,.30*q),.029,.029,dark)
    if not cat and style==1:s.ell((-.13,.87,0),(.61,.36,.36),dark)
    pethead(s,kind,h,c,style)

def hoofed(s,kind,style):
    horse=kind==7;c=petcol(kind);dark=blend(c,'332c26',U(.44,.78));soft=blend(c,'dfd3b8',U(.18,.43))
    length=1.00 if horse else .86;top=1.19 if horse else 1.11;width=.35 if horse else .27
    s.shadow((.12,.02,0),1.55,.56);s.ell((-.05,top,0),(length,.44 if horse else .36,width),c)
    for q in (-1,1):
        z=width*.64*q
        for front in (-1,1):
            x=front*length*.65; d=U(-.25,.25); f=front>0
            hip=(x,top+.06,z);knee=(x+(.025 if f else .09)+d*.35,.69,z*1.1);ankle=(x+d+(-.04 if f else -.13),.20,z*1.23);foot=(ankle[0]+.036,.075,ankle[2]);w=.133 if horse else .09
            s.rod(hip,knee,w*1.29,w,c);s.rod(knee,ankle,w*.57,w*.55,c);s.rod(ankle,foot,w*.58,w*.58,dark)
            s.ell(foot,(w*.85,w*.53,w*.70),dark)
    h=(1.10 if horse else .99,1.99 if horse else 1.87,0); chin=(h[0]+(.37 if horse else .29),h[1]-.31,0)
    s.rod((.46,top+.045,0),h,.248 if horse else .173,.235 if horse else .16,c)
    if horse:s.rod((.27,top+.31,0),(h[0]-.11,h[1]+.09,0),.117,.114,dark)
    s.ell(h,(.228,.218,.169),c);s.rod(h,chin,.172,.144,c);s.ell(chin,(.154,.132,.149),dark if horse else soft)
    s.ell((chin[0]+.1,chin[1]-.008,0),(.068,.082,.115),dark)
    for q in (-1,1):
        eye(s,(h[0]+.10,h[1]+.027,q*.149),.045,black,q)
        ear(s,(h[0]-.08,h[1]+.13,q*.095),(h[0]-(.13 if horse else .27),h[1]+(.41 if horse else .30),q*(.17 if horse else .33)),.078,.09,c,dark)
        if not horse and style==0:
            a=(h[0]-.028,h[1]+.12,q*.09);b=(h[0]-.14,h[1]+.38,q*.24);e=(h[0]-.39,h[1]+.65,q*.36)
            s.rod(a,b,.035,.034,dark);s.rod(b,e,.029,.026,dark);s.rod(b,(h[0]-.028,h[1]+.61,q*.28),.026,.026,dark)
            s.rod(e,(h[0]-.36,h[1]+.81,q*.44),.024,.018,dark)
    if horse:
        s.rod((-.91,top+.15,0),(-1.18,.80,.033),.16,.149,dark);s.rod((-1.17,.83,.03),(-1.31,.31,.05),.13,.13,dark)
        if style==1:
            s.ell((.02,top+.32,0),(.43,.17,.34),'54443a');s.rod((.00,top+.49,0),(.12,top+1.05,0),.18,.20,pick('455765 6a696a 8d6657 b6b4ad'));s.ell((.13,top+1.20,0),(.145,.17,.14),'b09279');s.ell((.12,top+1.32,0),(.15,.091,.15),'383736')
    else:
        s.ell((-.85,top+.08,0),(.18,.16,.15),soft);s.rod((-.81,top+.16,0),(-1.04,top+.32,0),.087,.074,c)

def bird(s,style):
    flying=style==2; duck=style==1; long=style==3
    body=varied(pick('756c5e 55514a 4a4c49 9b8a70 b7ad91 cac6ba 45484a d5d4c7 6f777a 816547 92944a 617866 4c768d'))
    wing=blend(body,pick('413f3a 695e50 4d565e 365b70 776953 474b35'),U(.3,.80));head=body if P(.60) else varied(pick('484540 546353 a28f70 5a6b70 6d6252'))
    ht=.53 if duck else (.91 if long else .71);s.ell((0,ht,0),(.63,.32 if not duck else .27,.30),body)
    h=(.40,ht+(.67 if long else (.22 if flying else .32)),0);s.rod((.26,ht,0),h,.17,.168,body);s.ell(h,(.244,.23,.219),head)
    for q in (-1,1):
        eye(s,(h[0]+.123,h[1]+.038,.174*q),.038,'292929',q)
        if flying:
            s.facet([(.30,ht+.11,.10*q),(.12,ht+.26,.86*q),(-.30,ht+.36,1.44*q),(-.51,ht+.29,1.24*q),(-.50,ht,.16*q)],wing)
        else:s.rod((.26,ht+.048,.225*q),(-.59,ht-.055,.185*q),.197,.105,wing)
        if not flying and not duck:
            s.rod((.055,ht-.15,.105*q),(.026,ht-(.62 if long else .44),.14*q),.030,.028,'6e5a43');s.rod((.038,ht-(.62 if long else .44),q*.14),(.18,ht-(.65 if long else .47),q*.15),.024,.020,'6e5a43')
    bill=pick('a59260 a08451 665d4c a48857') if not duck else 'ba9859';s.rod((h[0]+.16,h[1]-.035,0),(h[0]+(.47 if (duck or long) else .36),h[1]-.055,0),.056,.092 if duck else .056,bill)
    s.rod((-.42,ht,0),(-.97,ht+(.10 if duck else -.14),0),.082,.153,wing)
    if style==0 and P(.65):s.rod((-.96,ht-.56,-1),(.47,ht-.46,.8),.058,.062,'675340')

def frog(s,style):
    base=varied(pick('879055 7c8255 606d40 5b673c 877452 6f6547 657641 809759 729248 819f4d 4c713e 87806a 7a7462 9b895e'));light=blend(base,'dfd6a7',.39);dark=blend(base,'382e21',.56)
    s.ell((-.08,.39,0),(.62,.38,.48),base);s.ell((.38,.49,0),(.45,.316,.438),base);s.ell((.53,.393,0),(.349,.166,.375),light);s.ell((.53,.462,0),(.355,.177,.381),base)
    for q in (-1,1):
        s.ell((-.37,.26,.368*q),(.394,.236,.239),base);s.rod((-.48,.17,.42*q),(.044,.105,.57*q),.099,.123,base)
        s.rod((.22,.42,.33*q),(.13,.18,.50*q),.092,.083,base);s.rod((.14,.18,.50*q),(.46,.094,.55*q),.066,.063,light)
        for j in [-1,0,1]:s.rod((.42,.08,.55*q),(.62-abs(j)*.047,.07,(.55+.12*j)*q),.025,.026,light)
        p=(.46,.703,.266*q);s.ell(p,(.136,.127,.12),base);s.ell((.514,.733,.31*q),(.078,.084,.085),pick('aaa263 a89551 9a8b5a'));s.ell((.548,.742,.354*q),(.055,.04,.048),'2c2d24')
        s.ell((.778,.524,.205*q),(.028,.024,.021),dark)
    if style!=2:
        for j in range(12):
            u=U(-.96,.66);v=U(-.79,.79);w=math.sqrt(max(0,1-u*u-v*v));s.ell((-.08+.612*u,.39+.376*w,.47*v),(.075,.031,.064),dark)
    s.shadow((.04,0,0),1.05,.75)

def render(k,i):
    sky=['air','air','air','air','air','tarmac','tarmac','field']
    backgrounds=[sky,['street']*7+['tarmac','field','interior'],['wood','wood','wood','field','field','air','pond','interior'],['interior']*6+['field','wood'],['wood','wood','wood','field','field','field','field'],['interior','interior','interior','field','field','field','wood','street'],['pond','pond','wood','wood','wood','field','field','interior'],['field']*5+['wood','street','interior'],['sea']*6+['port']*3, ['street']*6+['tarmac','field']]
    mode=choose(backgrounds[k]);
    if P(.083):mode='plain'
    elif P(.07):mode='mixed'
    az=U(-math.pi,math.pi);el=U(.012,.35);roll=U(-.062,.062);span=U(21,34);portrait=False
    if k==0:
        style=choose([0,0,0,1,1,2]);ext=(4.55,.98,4.0 if style!=2 else 2.75);mid=(0,.13,0);span=U(18,36);el=U(-.75,.77);roll=U(-.72,.72) if mode=='air' else U(-.15,.15)
        if P(.4):el=U(-.30,.30)
        tex=U(.005,.028)
    if k==1:
        style=choose([0,0,0,0,1,1,2,3]);ext=(4.0,1.56+style*.042,1.57);mid=(0,.80,0);span=U(24,38);tex=U(.008,.034)
    if k==9:
        style=choose([0,0,0,1,1,2,3]);ext=(5.15,2.15,1.41);mid=(0,1.07,0);span=U(23,37);tex=U(.008,.038);el=U(-.057,.27)
    if k==8:
        style=choose([0,0,1,2,2,3,4]);ext=(4.6,2.9 if style==2 else 1.6,1.22);mid=(0,ext[1]*.51,0);span=U(18,37);el=U(-.015,.37);tex=U(.008,.036)
    if k in [3,5]:
        style=i%4;portrait=P(.50 if k==3 else .40);tex=U(.037,.107);el=U(-.06,.51);roll=U(-.18,.18)
        if portrait:
            if P(.78):az=U(.20,1.75)
            if P(.5):az=math.pi-az
            ext=(1.16,1.18,.90);mid=(.22,1.26,0);span=U(18,35)
        else:
            ext=(2.45,1.73,.77);mid=(.025,.84,0);span=U(23,37)
    if k in [4,7]:
        style=choose([0,1,2,2,2,3]) if k==7 else i%4;ext=(2.89,2.41,.8);mid=(.15,1.16,0);span=U(20,35);tex=U(.019,.062);el=U(-.05,.25)
        if P(.15):
            ext=(1.27,1.34,.7);mid=(.96,1.71,0);span=U(18,33);az=U(-.4,2.4)
    if k==2:
        style=choose([0,0,0,1,1,2,3]);ext=(1.88,1.31,2.89 if style==2 else .69);mid=(-.04,.85,0);span=U(17,34);tex=U(.028,.082);el=U(-.39,.67);roll=U(-.22,.22)
        if style==2:roll=U(-.74,.74);el=U(-.85,.80)
        if style==1 and P(.5):mode='pond'
    if k==6:
        style=i%4;ext=(1.64,.79,1.38);mid=(.05,.39,0);span=U(20,41);tex=U(.035,.10);el=U(.05,1.05);roll=U(-.23,.23)
    if P(.10):span*=.74
    if P(.13):span*=1.19
    s=Scene(k,mode,ext,mid,az,el,roll,span,tex)
    if k==0:plane(s,style)
    elif k==1:automobile(s,style)
    elif k==9:truck(s,style)
    elif k==8:ship(s,style)
    elif k in (3,5):pet(s,k,style,portrait)
    elif k in (4,7):hoofed(s,k,style)
    elif k==2:bird(s,style)
    else:frog(s,style)
    im=s.finish()
    return im if i%2 else im[:,:,::-1]

REASON = '''
def partIndices(region):
    row,col = region
    indexes=np.arange(64).reshape(8,8)[row,col].reshape(-1)
    small=np.arange(16).reshape(4,4)[row.start//2:(row.stop+1)//2,col.start//2:(col.stop+1)//2].reshape(-1)
    hi=np.concatenate([indexes]+[64+64*i+indexes for i in range(6)]+[448+small]+[464+i*16+small for i in range(6)])
    si=np.concatenate([indexes,64+indexes,128+small,144+small,160+indexes])
    return hi,si

def focusFeatures(h,s,hi,si):
    return np.concatenate([.76*norm(h[:,hi]),.65*norm(s[:,si])],1)

def focusMatch(h,s,indices,cols):
    # Parts can occupy different poses from the rest of the matching view.
    descriptor=focusFeatures(h,s,indices[0],indices[1])
    scores=[]
    for j in range(10):
        corr=descriptor@cols[j].T
        best=np.partition(corr,-4,axis=1)[:,-4:]
        scores.append(.30*best.max(1)+.70*best.mean(1))
    return np.stack(scores,1)

regions=[partIndices(r) for r in [(slice(2,6),slice(1,7)),(slice(1,5),slice(2,6)),(slice(4,8),slice(1,7))]]
partbanks=[]
for hi,si in regions:
    bank=[]
    for j in range(10):
        h=CONSTANTS["h"+str(j)][::3];s=CONSTANTS["s"+str(j)][::3]
        bank.append(np.ascontiguousarray(focusFeatures(h,s,hi,si)))
    partbanks.append(bank)

def su(v):
    return np.clip(v,-1,1)

def sceneRules(rgb,maps):
    l,dx,dy,gr,earth,blue,red,gold,neutral=maps
    r,g,b=rgb[:,0],rgb[:,1],rgb[:,2]
    n=len(l)
    cen=(slice(None),slice(7,26),slice(6,26))
    top=(slice(None),slice(0,8),slice(None))
    bot=(slice(None),slice(24,32),slice(None))
    gradx,grady=deriv(l)
    ax=np.abs(gradx);ay=np.abs(grady);edge=np.sqrt(ax*ax+ay*ay)
    # Coherent straight edges at two lengths distinguish structures from fur and foliage.
    vals=[]
    for span in (5,11):
        av=0;ah=0
        for j in range(span):
            ah=ah+grady[:,:,j:j+33-span]
            av=av+gradx[:,j:j+33-span,:]
        eh=avg(np.abs(ah))/span;ev=avg(np.abs(av))/span
        vals.append((eh,ev))
    hx,vx=vals[1];h5,v5=vals[0]
    es=avg(edge); axm=avg(ax);aym=avg(ay)
    rect=su(4.7*((hx+vx+.007)/(axm+aym+.014)-.53))
    vert=su(4.0*((vx+.003)/(axm+.009)-.47))
    hori=su(4.6*((hx+.003)/(aym+.009)-.50))
    direction=su(2.4*(hx-vx)/(hx+vx+.014))
    sky=avg(blue[top]); ocean=avg(blue[bot]); green=avg(gr);fol=avg(gr[bot]); bush=avg(gr[top]); warm=avg(earth[cen]); warmfloor=avg(earth[bot]);
    topEdge=avg(edge[top]);botEdge=avg(edge[bot]);midEdge=avg(edge[cen]);
    smooth=su((.034-es)*34); skyflat=su((.025-topEdge)*45); landflat=su((.030-botEdge)*37)
    bright=avg(l);toplight=avg(l[top]); botlight=avg(l[bot]); mlight=avg(l[cen]); chrom=rgb.max(1)-rgb.min(1)
    bgmean=(avg(l[top])+avg(l[bot]))*.5
    darkfeet=avg(unit((.28-l[:,19:28,:])*5));neutralfloor=avg(neutral[bot]); grey=avg(neutral);colour=avg(chrom);lawn=avg(gold[bot]);
    # The road tends to be neutral underneath a band of dark wheels and chassis.
    road=su(2.6*(neutralfloor-.57)+.70*landflat+.62*(darkfeet-.22)-1.12*fol-.86*ocean)
    # Blue or pale sky and blue-grey water, with a horizontal intensity discontinuity.
    skycue=su(2.7*sky+.76*skyflat+1.6*(toplight-.6)-.54*bush)
    sea=su(2.85*ocean+.59*direction+.29*hori+.30*landflat-.58*fol-.51*warmfloor-.28)
    vegetation=su(2.2*green+1.03*fol+.73*avg(gold)-.75*ocean-.31*sky-.34)
    wool=su(2.9*(warm-.24)+.62*(warmfloor-.20)-.60*sky-.61*ocean)
    dim=su(2.1*(.51-bright));glare=avg(unit((l-.73)*4.5));hiContrast=su(4.0*(l.std((1,2))-.205)); close=su(10*(botEdge+topEdge-.082))
    # Repeated horizontal streaks in water, as distinct from uniform asphalt or grass.
    streak=su((avg(ay[bot])-avg(ax[bot]))/(.014+avg(edge[bot]))*3.1)
    skyboth=su(1.65*(sky+ocean)-.53+ .42*(skyflat+landflat))
    # Bilateral matching of luminance detail for animal portraits and head-on vehicles.
    cenL=l[:,5:27,6:26];rev=cenL[:,:,::-1]; dif=avg(np.abs(cenL-rev)); rms=cenL.std((1,2)); sym=su(1.9*(.94-dif/(.055+rms))-.44)
    eyes=np.zeros(n,np.float32)
    fine=blur(l); broad=blur(blur(blur(fine))); spot=pos(broad-fine-.008)
    nose=unit((broad-fine)*8)
    for spacing in (4,7,10,13):
        p=spot[:,8:19,4:28-spacing]; q=spot[:,8:19,4+spacing:28];z=spacing//2
        bridge=l[:,8:19,4+z:28-spacing+z]-.5*(l[:,8:19,4:28-spacing]+l[:,8:19,4+spacing:28]);ns=nose[:,12:23,4+z:28-spacing+z]
        evidence=np.sqrt(p*q)*unit(bridge*8)*(.55+.45*ns)
        eyes=np.maximum(eyes,evidence.max((1,2)))
    face=su(6.0*(eyes-.08)+.55*sym)
    # Vivid paint is common on cars and truck cabs. Fur rarely has that chroma.
    painted=su(3.5*avg(red[cen])+1.3*avg(blue[cen])-.32)
    centreGreen=avg(gr[cen]); greenFrog=su(2.8*(centreGreen-.17)+.8*(avg(gold[cen])-.26)-.45*sky)
    gray=su(2.5*(grey-.54)); turf=su(2.9*fol+1.4*lawn-.72*warmfloor-.46)
    shadow=su(2.7*(.5*(botlight+toplight)-mlight)+1.3*(darkfeet-.23));whites=su(3.3*(glare-.17))
    man=.46*rect+.24*hori+.22*road-.46*vegetation-.19*wool
    animal=-.44*rect-.20*hori-.28*road+.33*vegetation+.19*wool
    a=.90*skyboth+.46*smooth+.43*skycue-.28*vert-.26*hori-.25*streak+.16*whites-.35*vegetation-.17*wool
    c=man+.52*road+.33*shadow+.34*painted+.19*direction+.26*hiContrast-.33*sea-.15*skyflat
    t=man+.44*vert+.26*road+.25*skycue+.20*whites+.24*painted-.25*sea-.27*smooth
    shipscore=.98*sea+.48*skycue+.51*direction+.42*streak+.23*gray+.15*hori-.32*vegetation-.38*wool
    birdscore=animal+.32*skycue+.29*smooth+.21*vegetation-.15*shadow-.10*face-.16*wool+.14*painted
    catscore=animal+.35*wool+.26*face+.18*sym+.27*close-.44*turf-.26*vegetation-.23*skycue+.15*gray
    deerscore=animal+.47*vegetation+.35*turf+.24*wool-.22*painted-.17*face-.20*hiContrast+.18*bush
    dogscore=animal+.35*wool+.35*face+.13*close-.23*skycue-.25*sea-.18*vegetation+.14*hiContrast
    frogscore=animal+.65*greenFrog+.36*dim+.39*close-.33*skycue-.34*whites-.20*sym+.12*warmfloor-.16*hori
    horsescore=animal+.55*turf+.25*skycue+.22*hiContrast+.26*shadow+.11*wool-.17*greenFrog-.16*close
    return np.stack([a,c,birdscore,catscore,deerscore,dogscore,frogscore,horsescore,shipscore,t],1)
'''

INFER = '''
def matchView(h,s,c):
    # Equal numbers of templates per class; match HOG, shading and chroma jointly.
    out=[]
    for j in range(10):
        a=h@CONSTANTS["h"+str(j)].T
        b=s@CONSTANTS["s"+str(j)].T
        d=c@CONSTANTS["c"+str(j)].T
        z=.48*a+.35*b+.17*d
        z=z.reshape(len(h),-1)
        best=np.partition(z,-6,axis=1)[:,-6:]
        geom=np.partition(.67*a+.33*b,-4,axis=1)[:,-4:].mean(1)
        out.append(.26*best.max(1)+.55*best.mean(1)+.19*geom)
    return np.stack(out,1)

def classify(x):
    h,s,c,m=features(x)
    geo=matchView(h,s,c)
    # Mirroring is a new input viewpoint, not another training example.
    hf,sf,cf,mf=features(np.ascontiguousarray(x[:,:,:,::-1]));flip=matchView(hf,sf,cf)
    geo=.70*np.maximum(geo,flip)+.30*np.minimum(geo,flip)
    part=0
    for i,region in enumerate(regions):
        f=focusMatch(h,s,region,partbanks[i]);g=focusMatch(hf,sf,region,partbanks[i])
        part+=np.maximum(f,g)*([.46,.29,.25][i])
    return (14.5*geo+3.5*part+.76*sceneRules(x,m)).argmax(1)

def predict(x):
    # Fixed compute shape also fixes BLAS reduction order on partial batches.
    n=len(x);out=np.empty(n,dtype=np.int64)
    for i in range(0,n,32):
        batch=np.zeros((32,3,32,32),np.float32)
        part=x[i:i+32];batch[:len(part)]=part
        out[i:i+len(part)]=classify(batch)[:len(part)]
    return out
'''

# No aggregation of descriptors across renders. Batching features here is just
# vectorization of the same independent per-picture operations used on the input.
def main():
    dest=Path(sys.argv[1]);dest.mkdir(parents=True,exist_ok=True)
    (dest/'solution.py').write_text(FEATURES+REASON+INFER)
    arrays={}
    for k in range(10):
        parts=[[],[],[]]
        for start in range(0,2304,32):
            x=np.stack([render(k,i) for i in range(start,start+32)])
            h,s,c,m=features(x)
            for p,v in zip(parts,(h,s,c)):p.append(v.astype(np.float32))
        for p,name in zip(parts,'hsc'):arrays[name+str(k)]=np.ascontiguousarray(np.concatenate(p,0))
        print('Rendered class',k,flush=True)
    np.savez(dest/'constants.npz',**arrays)

if __name__=='__main__': main()
