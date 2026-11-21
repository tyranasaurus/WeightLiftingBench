"""Compile an analytic sign recognizer. No assets, fonts, images or saved parameters."""
import json, sys, math
from pathlib import Path
import numpy as np

L=[]; W={}
def layer(name,op,inputs,**kw):
    if isinstance(inputs,str): inputs=[inputs]
    L.append(dict(name=name,op=op,inputs=inputs,**kw)); return name

def dense(name,src,w,b=0.):
    w=np.asarray(w,np.float32); W[name+'.W']=w; W[name+'.b']=np.broadcast_to(b,(len(w),)).astype('f')
    return layer(name,'dense',src,out=len(w))
def conv(name,src,w,b=0.,pad=0,stride=1):
    w=np.array(w,np.float32); W[name+'.W']=w; W[name+'.b']=np.broadcast_to(b,(len(w),)).astype('f')
    return layer(name,'conv',src,out=len(w),kernel=list(w.shape[-2:]),padding=pad,stride=stride)
def point(name,src,w,b=0.):return conv(name,src,np.array(w)[:,:,None,None],b)
def rel(name,x):return layer(name,'relu',x)
def pool(name,x,h,w=None,avg=False):
    if w is None:w=h
    return layer(name,'avgpool' if avg else 'maxpool',x,size=[h,w],stride=[h,w])
def reshape(name,x,*dims):return layer(name,'reshape',x,shape=list(dims))
def add(name,*args):return layer(name,'add',list(args))

def clip(name,x,c,C):
    I=np.eye(C); v=rel(name+'_rect',point(name+'_breaks',x,np.r_[I,I],np.r_[np.ones(C)*c,-np.ones(C)*c]));return point(name,v,np.c_[I,-I],-c)

def gaussian(k,sigma):
    t=np.arange(k)-(k-1)/2; g=np.exp(-t*t/(2*sigma**2));g/=g.sum();return g[:,None]*g

# A piecewise-linear approximation to log radiance: enhance dim reflectors
# without relying on a fixed RGB cutoff in shadow.
t=np.array([0,.016,.04,.09,.18,.35,.65,1.]); s=np.diff(np.log(t+.023))/np.diff(t); co=np.diff(np.r_[0,s]); I=np.eye(3)
a=rel('hinges',point('exposure_breaks','image',np.tile(I,(len(s),1)),-np.repeat(t[:-1],3)))
a=point('logRGB',a,np.concatenate([v*I for v in co],axis=1))
lum=np.array([.30,.50,.20]);# neutral ink contrast, using log luminance
# Color coordinates deliberately keep *sign* even on an achromatic sign.
mat=np.array([lum,[1,-.62,-.38],[-.72,-.28,1],[.4,.6,-1]])
a=point('percept',a,mat); g=gaussian(9,2.15); delta=np.zeros_like(g);delta[4,4]=1
rawlight=point('lightness',a,[[1,0,0,0]])
# Center before filtering: the artificial black outside of a convolution must
# not overwhelm the small contrast inside a sign on an otherwise bright image.
glob=layer('light_contrast','norm',rawlight)
z=conv('detail',glob,np.stack([delta-g,.8*delta+.2*gaussian(9,.9)])[:,None,:,:],pad=4)
z=clip('detail_clip',layer('ink_contrast','norm',z),2.8,2)
# Flattening little chips by a dense map avoids a large sliding-window tensor
# in the reference interpreter. Maps are shared by every symbol/class.
k=20; shifts=[(x,y) for y in (-2,-1,0,1,2) for x in (-2,-1,0,1,2)]; P=len(shifts)
def sampler(name,src,shifts,side,offset):
    A=np.zeros((side,side,len(shifts),32,32),np.float32);yy,xx=np.indices((side,side));
    for p,(dx,dy) in enumerate(shifts): A[yy,xx,p,yy+offset+dy,xx+offset+dx]=1
    return reshape(name+'_array',dense(name,src,A.reshape(side**2*len(shifts),-1)),side**2,len(shifts),1)
b=point('ink',z,[[.74,.36]])
b=sampler('read_center',b,shifts,k,(32-k)//2)

# A separate low-resolution stream for outline, background and color.
colorw=np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1]])
c=clip('saturation',point('hues',a,colorw),.95,3)
lc=point('large_scale_brightness',z,[[0,1]])
# stack channels using embeddings then add
c=add('surface',point('hue_axes',c,[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]),point('light_axis',lc,[[1],[0],[0],[0]]))
c=pool('area_sample',c,2,avg=True)

# Vector drafting primitives, in a unit coordinate system centered on the sign.
# Radius or half-width = 1. Positive y points down; all proportions are design choices.
class Canvas:
    def __init__(self,N):
        u=(np.arange(N)+.5)/N*2-1;self.x,self.y=np.meshgrid(u,u);self.a=np.zeros((N,N),bool)
    def ell(self,x,y,rx,ry,fill=1):self.a[((self.x-x)/rx)**2+((self.y-y)/ry)**2<=1]=fill
    def poly(self,pts,fill=1):
        m=np.zeros_like(self.a);x0,y0=pts[-1]
        for x1,y1 in pts:
            if y1!=y0:m^=((y0>self.y)!=(y1>self.y))&(self.x<(x1-x0)*(self.y-y0)/(y1-y0)+x0)
            x0,y0=x1,y1
        self.a[m]=fill
    def line(self,pts,width,fill=1):
        for (x,y),(v,w) in zip(pts[:-1],pts[1:]):
            dx=v-x;dy=w-y; u=np.clip(((self.x-x)*dx+(self.y-y)*dy)/(dx*dx+dy*dy+1.e-20),0,1);self.a[(self.x-x-u*dx)**2+(self.y-y-u*dy)**2<=(width/2)**2]=fill

def curve(p,closed=False):
    # Catmull-Rom through control points.
    a=np.array(p,dtype=float);p=np.vstack(([a[-1]] if closed else [a[0]],a,[a[0],a[1]] if closed else [a[-1]]));out=[]
    for i in range(1,len(p)-2):
        a,b,c,d=p[i-1:i+3]
        for u in np.arange(5)/5:out.append(.5*(2*b+(-a+c)*u+(2*a-5*b+4*c-d)*u*u+(-a+3*b-3*c+d)*u**3))
    return np.array(out+[p[-2]])
def smooth(q,p,w,fill=1):q.line(curve(p),w,fill)

# Digits: straight DIN-like stems with curved bowls. Glyph space [0,1] x [0,1].
digits={
'0':[[(.5,.075),(.77,.10),(.88,.27),(.88,.72),(.77,.91),(.5,.935),(.23,.90),(.12,.72),(.12,.27),(.23,.10),(.5,.075)]],
'1':[[(.18,.29),(.56,.08),(.57,.93)]],
'2':[[(.10,.28),(.2,.135),(.46,.07),(.72,.09),(.87,.21),(.86,.36),(.72,.49),(.42,.66),(.12,.90)],[(.12,.92),(.89,.92)]],
'3':[[(.12,.18),(.31,.085),(.61,.075),(.84,.16),(.87,.29),(.76,.405),(.47,.47)],[(.45,.47),(.69,.49),(.86,.60),(.88,.76),(.76,.88),(.52,.935),(.29,.9),(.11,.82)]],
'5':[[(.84,.085),(.20,.085),(.16,.47)],[(.16,.47),(.43,.41),(.68,.43),(.85,.54),(.88,.72),(.79,.865),(.55,.93),(.30,.90),(.13,.82)]],
'6':[[(.77,.09),(.53,.12),(.30,.28),(.15,.49),(.14,.72),(.24,.88),(.48,.935),(.72,.90),(.87,.77),(.86,.61),(.72,.50),(.51,.46),(.31,.49),(.15,.6)]],
'7':[[(.10,.085),(.89,.085),(.53,.50),(.32,.93)]],
'8':[[(.50,.455),(.23,.39),(.14,.25),(.21,.12),(.49,.067),(.77,.12),(.85,.25),(.76,.39),(.50,.455),(.26,.52),(.12,.66),(.16,.82),(.32,.915),(.55,.937),(.79,.87),(.89,.73),(.85,.59),(.70,.51),(.50,.455)]]}
def number(q,text,v):
    trip=len(text)==3; ht=(.93,.99,.84,.91)[v%4] if not trip else (.79,.90,.87,.94)[v%4]; wid=ht*((.435,.49,.56,.475) if trip else (.51,.55,.47,.52))[v%4]; gap=ht*(.075,.10,.13,.085)[v%4]; widths=[wid*(.65 if d=='1' else 1.) for d in text]; x=-(sum(widths)+gap*(len(text)-1))/2
    for d,w in zip(text,widths):
        for j,path in enumerate(digits[d]):
            pts=np.array(path)
            if d not in '17' and not(d=='5' and j==0) and len(pts)>2:pts=curve(pts)
            pts=pts*[w,ht]+[x,-.5*ht];q.line(pts,ht*(.108,.139,.114,.14)[v%4]);
        x+=w+gap

def car(q,cx,cy,w,h):
    pts=[(-.50,-.15),(-.42,-.22),(-.31,-.46),(.30,-.46),(.42,-.20),(.50,-.12),(.49,.31),(.4,.35),(.39,.47),(.20,.47),(.19,.35),(-.21,.35),(-.22,.47),(-.43,.47),(-.44,.32),(-.50,.28)]
    def f(p):return [(cx+x*w,cy+y*h) for x,y in p]
    q.poly(f(pts));q.poly(f([(-.32,-.17),(-.24,-.36),(.22,-.36),(.32,-.17)]),0)
def truck(q,cx,cy,w,h,mir=False):
    def f(x,y):return (cx+w*(-x if mir else x),cy+y*h)
    q.poly([f(x,y) for x,y in [(-.5,-.02),(-.36,-.22),(-.22,-.22),(-.22,-.43),(.5,-.43),(.5,.24),(-.5,.24)]]);
    for x in [-.31,.30]: q.ell(*f(x,.26),h*.165,h*.17)
    q.poly([f(x,y) for x,y in [(-.42,-.035),(-.34,-.18),(-.26,-.18),(-.26,-.035)]],0)
def arrow(q,pts,head,sw,hw,hl):
    p=np.array(pts,float); q.line(p,sw);e=np.array(head,float);d=e-p[-1];d/=np.linalg.norm(d);normal=np.array([-d[1],d[0]]);base=e-d*hl;q.line([p[-1],base+.02*d],sw);q.poly([base+hw/2*normal,e,base-hw/2*normal]);
def person(q,x,y,s,sgn=1):
    def f(p):return [(x+sgn*s*a,y+s*b) for a,b in p]
    q.ell(*f([(.06,-.40)])[0],.09*s,.10*s)
    for p,w in [([(.02,-.25),(-.09,.02)],.17),([(-.07,.01),(.1,.18),(.19,.42)],.085),([(-.07,.01),(-.12,.18),(-.32,.39)],.09),([(.0,-.21),(.12,-.06),(.25,-.11)],.065),([(-.02,-.2),(-.22,-.10),(-.29,.05)],.065)]:q.line(f(p),s*w)

N=120
nums=['20','30','50','60','70','80','80','100','120']
# Boolean stencil and region where a clean background is expected.
def make_symbol(k,v):
    q=Canvas(N); x,y=q.x,q.y; area=x*x+y*y<.71**2; light=False
    if k<=8:
        number(q,nums[k],v)
        if k==6:q.line([(-.77,.79),(.77,-.79)],.21 if v%2==0 else .29)
    elif k in [9,10,41,42]:
        if k in [9,41]:car(q,-.32,.015,.48,.43);car(q,.295,.015 if v%2==0 else .075,.46,.43)
        else:truck(q,-.26,-.04,.75,.61,True);car(q,.36,.08,.40,.36)
        if k in [41,42]:q.line([(-.78,.78),(.78,-.78)],.20 if v%2 else .31)
    elif k==13:
        edge=np.maximum((abs(x)-.5*(-y+.98))*.90,-y-.92);q.a=(edge<0)&(edge>-.15);area=edge<.05
    elif k==15:pass # blank field
    elif k==16:truck(q,0,0,1.33,.74,False)
    elif k==11 or 18<=k<=31:
        # Equilateral warning signs carry their symbol below their bounding-box center.
        area=(y<.70)&(y>-.55)&(np.abs(x)<.51*(y+.69)+.025)
        if k==11:
            q.poly([(-.125,.55),(.125,.55),(.125,.28),(.32,.28),(.32,.17),(.125,.17),(.125,-.08),(0,-.28),(-.125,-.08),(-.125,.17),(-.32,.17),(-.32,.28),(-.125,.28)])
        elif k==18:
            q.poly([(-.067,-.31),(.063,-.31),(.033,.31),(-.035,.31)]);q.ell(0,.455,.067,.076)
        elif k in [19,20]:
            s=1 if k==19 else -1
            # A constant-width road ribbon with flat (not round) terminals.
            pts=curve([(.23,.54),(.23,.28),(.20,.12),(.10,-.015),(-.05,-.08),(-.18,-.10)]);dxy=np.gradient(pts,axis=0);dxy/=np.maximum(1e-6,np.linalg.norm(dxy,axis=1,keepdims=True));nrm=np.c_[-dxy[:,1],dxy[:,0]]*(.103 if v%2 else .082);q.poly(np.r_[pts+nrm,(pts-nrm)[::-1]]*[s,1])
        elif k==21:
            p=curve([(.22,.55),(.22,.23),(.13,.12),(-.075,.11),(-.145,.03),(-.15,-.24)]);dxy=np.gradient(p,axis=0);dxy/=np.maximum(1e-6,np.linalg.norm(dxy,axis=1,keepdims=True));nm=np.c_[-dxy[:,1],dxy[:,0]]*.091;q.poly(np.r_[p+nm,(p-nm)[::-1]])
        elif k==22:
            p=[(-.58,.48),(-.47,.43),(-.39,.28),(-.30,.25),(-.21,.29),(-.12,.41),(-.03,.39),(.06,.26),(.19,.24),(.31,.30),(.41,.43),(.56,.48)];q.poly(list(curve(p))+[(.58,.53),(-.58,.53)])
        elif k==23:
            # Rear window and two crossed tire tracks of a car in a skid.
            car(q,-.025,-.012,.56,.47)
            for d in [-.16,.16]:smooth(q,[(d,.20),(d+.09,.29),(d+.045,.36),(d-.14,.40),(d-.21,.49),(d-.13,.54)],.061)
            if v%2: # tilt the car; its tires slide out to one side
                ang=.18;rx=x*np.cos(ang)+y*np.sin(ang);ry=y*np.cos(ang)-x*np.sin(ang);q.a=raster(q.a,rx,ry)>.5
        elif k==24:
            q.line([(-.23,.53),(-.23,-.16)],.09);smooth(q,[(.24,.53),(.24,.34),(.18,.18),(.08,.03),(.07,-.22)],.09)
        elif k==25:
            q.ell(-.02,-.17,.088,.09);q.line([(-.09,-.08),(-.27,.10)],.17);q.line([(-.26,.1),(-.24,.28),(-.42,.54)],.11);q.line([(-.27,.12),(-.12,.29),(-.11,.53)],.105);q.line([(-.08,-.06),(-.02,.10),(.12,.16)],.069);q.line([(-.27,-.02),(.34,.42)],.048);q.poly([(-.005,.54),(.12,.37),(.24,.23),(.32,.27),(.43,.39),(.59,.54)]);q.poly([(-.56,.55),(.58,.55),(.59,.585),(-.58,.585)])
        elif k==26:
            # European warning pictogram: three separate signal lights, no housing.
            for yy in [-.20,.112,.425]:q.ell(0,yy,.108*(1+.08*(v%2)),.112*(1+.08*(v%2)))
        elif k==27:
            person(q,0,.12,.96,-1 if v%2 else 1)
            if v%2:
                # Striding pedestrian profile, with a clearly separated pair of legs.
                q.ell(-.045,-.22,.075,.085);q.line([(-.03,-.10),(.055,.14)],.13);q.line([(.055,.14),(.20,.33),(.30,.51)],.084);q.line([(.04,.14),(-.085,.36),(-.19,.52)],.09)
        elif k==28:
            person(q,-.095,.12,.91,-1);person(q,.22,.24,.65,-1)
            if v%2:q.poly([(-.165,-.07),(-.045,-.05),(.055,.27),(-.20,.24)])
            q.line([(-.07,-.035),(.045,.085),(.18,.05)],.05)
        elif k==29:
            for xx in [-.30,.32]:q.ell(xx,.31,.20,.20);q.ell(xx,.31,.145,.145,0)
            q.line([(-.30,.31),(-.14,.02),(.06,.31),(-.30,.31)],.045);q.line([(.32,.31),(.17,-.02),(-.12,.02)],.05);q.ell(.09,-.21,.078,.083);q.line([(.00,-.10),(-.15,.015),(.005,.11),(-.06,.25)],.086);q.line([(.01,-.09),(.12,.00),(.23,-.01)],.054)
        elif k==30:
            for th in np.arange(6)*np.pi/3-np.pi/2:
                d=np.array([np.cos(th),np.sin(th)]);e=np.array([-d[1],d[0]]);pt=np.array([0,.17]);q.line([pt,pt+.37*d],.050)
                for sg in [-1,1]:q.line([pt+.17*d,pt+.30*d+sg*.10*e],.04)
        elif k==31:
            # Leaping deer: head to the left, rump below the shoulder, legs trailing.
            q.poly([(-.28,-.02),(-.36,-.08),(-.35,-.15),(-.25,-.19),(-.17,-.18),(-.11,-.05),(.01,-.01),(.24,.09),(.31,.21),(.28,.30),(.16,.29),(-.02,.18),(-.19,.12),(-.23,.02)]);
            for p,w in [([(.20,.21),(.32,.34),(.50,.47)],.073),([(.12,.23),(.22,.37),(.42,.50)],.058),([(-.15,.04),(-.21,.20),(-.36,.28)],.065),([(-.08,.11),(-.08,.24),(-.24,.32)],.058),([(-.21,-.15),(-.18,-.26),(-.24,-.36)],.031)]:q.line(p,w)
    elif k==12:
        # White bevel of the yellow diamond has a dark outline against the road.
        area=abs(x)+abs(y)<1.17;q.a=(abs(x)+abs(y)<1.01)&(abs(x)+abs(y)>.72);light=True
    elif k==14:
        # STOP in tall narrow capitals. White strokes against red.
        light=True;area=x*x+y*y<.88**2
        letters=[[[(.9,.10),(.66,.015),(.23,.05),(.06,.23),(.19,.41),(.71,.54),(.94,.74),(.82,.92),(.49,.99),(.08,.89)]],[[(0,.015),(1,.015)],[(.50,.015),(.50,.99)]],[[(.50,.02),(.19,.05),(.07,.25),(.06,.76),(.23,.95),(.55,.99),(.83,.92),(.94,.72),(.93,.24),(.77,.04),(.50,.02)]],[[(.09,1),(.09,.02)],[(.09,.02),(.70,.025),(.91,.13),(.93,.38),(.78,.51),(.09,.53)]]]
        for j,letter in enumerate(letters):
            for p in letter:
                p=np.array(p);p=curve(p) if len(p)>4 else p;p=p*[.27,.62]+[-.675+j*.356,-.31];q.line(p,.052+.009*(v%2))
    elif k==17:
        area=x*x+y*y<.90**2;light=True;q.poly([(-.80,-.16),(.80,-.16),(.80,.16),(-.80,.16)])
    elif k==32:q.line([(-.78,.78),(.78,-.78)],.28)
    elif k in range(33,41):
        area=x*x+y*y<.83**2;light=True
        if k in (33,34):
            sg=1 if k==33 else -1;p=curve([(-.26,.6),(-.26,.05),(-.21,-.14),(-.07,-.26),(.17,-.27)])*[sg,1];arrow(q,p,(sg*.59,-.27),.174,.55,.34)
        if k in (35,36,37):
            xx=(0 if k==35 else (-.21 if k==36 else .21));arrow(q,[(xx,.64),(xx,-.23)],(xx,-.69),.195,.58,.37)
            if k in (36,37):
                sg=1 if k==36 else -1;arrow(q,[(xx,.11),(.20*sg,.11)],(.63*sg,.11),.18,.51,.34)
        if k in (38,39):
            sg=1 if k==38 else -1; # German keep-right arrow has a right-angle tip with two square-ended wings
            if v%2==0:
                q.poly([(sg*x,y) for x,y in [(-.60,-.40),(-.40,-.60),(.27,.07),(.27,-.28),(.51,-.04),(.51,.51),(-.04,.51),(-.28,.27),(.07,.27)]])
            else:arrow(q,[(-sg*.50,-.50),(sg*.15,.15)],(sg*.55,.55),.26,.66,.41)
        if k==40:
            for u in range(3):
                t=np.linspace(.05,1.54,10)-2*np.pi*u/3;p=np.stack([.47*np.cos(t),-.47*np.sin(t)],axis=1); dr=(p[-1]-p[-2]);dr/=np.linalg.norm(dr);arrow(q,p,p[-1]+.17*dr,.16,.39,.25)
    else:return None,None,False
    # Separate extensions (stripes) do not get to sample sign exterior.
    return q.a,area,light

# Bilinear lookup of an antialiased stencil into a square patch. Sampling a
# procedural drawing is not sampling an example. v gives related glyph styles.
px,py=np.meshgrid(np.arange(20)-9.5,np.arange(20)-9.5)
def raster(m,X,Y):
    fx=(X+1)*.5*N-.5;fy=(Y+1)*.5*N-.5; i=np.floor(fx).astype(int);j=np.floor(fy).astype(int);u=fx-i;v=fy-j
    def val(i,j):return m[np.clip(j,0,N-1),np.clip(i,0,N-1)]*((i>=0)&(j>=0)&(i<N)&(j<N))
    return val(i,j)*((1-u)*(1-v))+val(i+1,j)*u*(1-v)+val(i,j+1)*(1-u)*v+val(i+1,j+1)*u*v

def small_render(m,X,Y,rx,ry):
    return sum(raster(m,X+i/rx,Y+j/ry) for i in [-.30,.30] for j in [-.30,.30])*.25

def blur(a,s):
    r=int(math.ceil(2.7*s));w=np.exp(-np.arange(-r,r+1)**2/(2*s*s));w/=w.sum();
    for ax in (0,1): a=np.apply_along_axis(lambda x:np.convolve(x,w,'same'),ax,a)
    return a

# Each pixel in a chip is a channel. 1x1 convolutions score each possible
# location with exactly the same shape detectors.

# Symbol readout. Contrast masks are locally balanced, with both the ink and
# its surround contributing. Unit energy penalizes coincidental tiny matches.
S=48; allw=[]; present=[]
for cls in range(43):
    stencils=[make_symbol(cls,i) for i in range(4)]
    if stencils[0][0] is None:continue
    present.append(cls)
    rows=[]
    for n in range(S):
        v=n%4; m,area,light=stencils[v]; radius=[9.3,11.15,13.0,14.5][(n//4)%4]; shapevar=n//16
        asp=[1,1.10,.90][shapevar]; rx=radius*asp;ry=radius/asp; # perturb mounting angle mildly
        angle=[-.045,.04,0,.015][(n//4+v)%4] if shapevar else 0.;X=(px*np.cos(angle)+py*np.sin(angle))/rx;Y=(py*np.cos(angle)-px*np.sin(angle))/ry
        # Alternate placement of the warning pictogram / size of icon (not of outline)
        if cls==11 or 18<=cls<=31:
            f=[.93,1.11,1.0,1.03][v];X=X/f;Y=(Y-.17)/f+.17;
            if cls==27 and v%2: X=-X
        mask=small_render(m,X,Y,rx,ry);support=small_render(area,X,Y,rx,ry)
        mask=blur(mask,.53); support=np.maximum(mask,support);mask*=np.minimum(support*1.3,1)
        # Matched band-pass stencil: no credit simply for a dark patch of sign.
        avg=blur(mask,1.70); near=(avg+.085)*support
        h=mask-.80*avg;h-=near*h.sum()/(near.sum()+1e-8)
        h/=np.sqrt((h*h).sum())+1e-8
        rows.append(((1 if light else -1)*h).ravel())
    allw.extend(rows)
A=np.array(allw,np.float32)
r=point('pictograms',b,A);r=reshape('pictogram_hypotheses',r,len(present),S,P);r=pool('best_pictogram',r,S,P)

# Whole-sign color/shape matching. Describe surfaces by expected luminance,
# red-green, blue-yellow, and yellow-blue hues; black surroundings unspecified.
# Templates are antialiased and zero-sum in brightness to ignore exposure.
fx,fy=np.meshgrid(np.arange(16)*2-15,np.arange(16)*2-15)
def surfaces(group,x,y):
    dist=np.sqrt(x*x+y*y); aa=.085
    def inside(d):return np.clip(.5-d/aa,0,1)
    outer=inside(dist-1);inner=inside(dist-.755);circ=outer-inner
    if group in ('ring','pass','truckpass','truck','nopass'):
        lum=inner*.95-circ*.52;red=circ;blue=0*x;yellow=0*x
        if group in ('pass','truckpass','truck'):
            if group=='pass':
                l=inside(np.maximum(abs(x+.31)-.24,abs(y+.035)-.205));r=inside(np.maximum(abs(x-.29)-.225,abs(y-.04)-.187));red=red+l*.84;lum=lum-l*.86-r*1.65
            if group=='truckpass':
                l=inside(np.maximum(abs(x+.25)-.38,abs(y+.055)-.26));r=inside(np.maximum(abs(x-.37)-.17,abs(y-.08)-.145));red=red+l*.87;lum=lum-l*.89-r*1.60
            if group=='truck':lum-=1.67*inside(np.maximum(abs(x)-.61,abs(y)-.22))
        if group=='ring': # digits not drawn by the shape stream; give less importance to interior
            lum-=.68*inside(np.maximum(abs(x)-.54,abs(y)-.42))
    elif group in ('warning','yield'):
        Y=y if group=='warning' else -y
        # rounded triangular ring, hollow center
        d=np.maximum((abs(x)-(.50*(Y+.98)))*.90,Y-.91);outer=inside(d);inner=inside(d+.185);red=outer-inner;lum=inner*.96-.61*red
        if group=='warning':lum-=.64*inside(np.maximum(abs(x)-(.18+.20*Y),abs(Y-.18)-.39))
        blue=yellow=0*x
    elif group=='diamond':
        outer=inside((abs(x)+abs(y)-1.04)*.707);inner=inside((abs(x)+abs(y)-.76)*.707);lum=outer*.95-.29*inner;yellow=inner;red=blue=0*x
    elif group=='blue':lum=-.80*outer+.34*inside(dist-.51);red=yellow=0*x;blue=outer
    elif group in ('end','end80','endphp'):
        stripe=inside(abs(x+y)/1.414-.125)*outer;lum=.98*outer-1.4*stripe
        if group=='end80':lum-=.73*inside(np.maximum(abs(x)-.51,abs(y)-.39))*(outer-stripe)
        if group=='endphp':lum-=1.11*inside(np.maximum(abs(x)-.56,abs(y)-.19))*(outer-stripe)
        blue=yellow=red=0*x
    elif group=='stop':
        d=np.maximum(np.maximum(abs(x),abs(y))-.96,(abs(x)+abs(y)-1.355)*.707);outer=inside(d);white=outer-inside(d+.074);text=inside(np.maximum(abs(x)-.71,abs(y)-.25));lum=-.67*outer+1.1*white+.61*text;red=outer-white-.51*text;blue=yellow=0*x
    elif group=='entry':
        bar=inside(np.maximum(abs(x)-.81,abs(y)-.183));red=outer-bar;lum=-.75*red+.98*bar;blue=yellow=0*x
    return lum,red,blue,yellow,outer

groups=['ring','pass','truckpass','warning','diamond','yield','stop','nopass','truck','entry','end80','end','endphp','blue'];G=len(groups);shapes=[]
for group in groups:
    gs=[]
    for radius in (9.4,11.4,13.5,14.7):
        for asp in (.90,1.06):
            for dy in (-1.5,.65,2.5):
                for dx in (-1.7,.2,2.):
                    x=(fx-dx)/(radius*asp);y=(fy-dy)/(radius/asp);l,red,blue,yellow,m=surfaces(group,x,y)
                    # The environment around a road sign is metrically unreliable.
                    region=np.clip(blur(m,.65)*1.25,0,1);region=.73*region+.27*np.exp(-((x*x+y*y)/1.35)**2);l=l-region*l.sum()/region.sum();l/=np.linalg.norm(l)+1.e-5
                    # Match *both* the presence and absence of hue, with mostly
                    # signed local contrasts and some absolute chroma evidence.
                    def hue(h):
                        if not np.any(h):return -.36*m/(np.linalg.norm(m)+1e-8)
                        hc=h-.78*region*h.sum()/region.sum();return hc/(np.linalg.norm(hc)+1e-7)
                    gs.append(np.stack([l*.95,hue(red)*1.25,hue(blue)*.86,hue(yellow)*.37]).ravel())
    shapes.extend(gs)
H=len(gs);f0=dense('silhouettes',c,np.array(shapes));f=reshape('silhouette_hypotheses',f0,G,H,1);f=pool('best_silhouette',f,H,1)

# OCR: for each possible number box, resample a rectangular grid from the
# input. Norm over the box gives cosine matching of the *whole* inscription,
# independent of what happens on the rim or in the roadside background.
# The grid is shared by the different digits: its transform is a latent pose.
geo=[]
for height in (5.8,7.5,9.25,11.3,13.4):
    for width in (7.9,9.85,12.25,14.8,17.5):
        if not .82<=width/height<=2.10:continue
        for dx in (-1.9,-.65,.65,1.9):
            for dy in (-1.9,-.65,.65,1.9):geo.append((width,height,dx,dy))
# Pack normalized square chips into a vertical strip; stride D makes a
# single square kernel apply the same OCR stencil independently to each box.
def warps(name,src,poses,D,sx,sy):
    # All poses reuse a small set of horizontal interpolations. Emit one
    # channel per (x warp, source row); the remaining y interpolation is
    # pointwise across those channels, shared at each target column.
    locs=list(dict.fromkeys((p[0],p[2]) for p in poses));nx=len(locs)
    xproj=np.zeros((nx,32,D,32,32),np.float32);grid=(np.arange(D)+.5)/D-.5
    for ii,(rx,dx) in enumerate(locs):
        pos=15.5+dx+rx*grid*sx;i=np.floor(pos).astype(int)
        for y in range(32):
            for a in (0,1):xproj[ii,y,np.arange(D),y,np.clip(i+a,0,31)]+=1-abs(pos-i-a)
    x=reshape(name+'_columns',dense(name+'_horizontal',src,xproj.reshape(-1,1024)),nx*32,1,D)
    yp=np.zeros((len(poses),D,nx,32),np.float32)
    for j,(rx,ry,dx,dy) in enumerate(poses):
        ii=locs.index((rx,dx));pos=15.5+dy+ry*grid*sy;i=np.floor(pos).astype(int)
        for a in (0,1):yp[j,np.arange(D),ii,np.clip(i+a,0,31)]+=1-abs(pos-i-a)
    return reshape(name+'_chips',point(name+'_vertical',x,yp.reshape(-1,nx*32)),len(poses),D,D)
B=len(geo);D=16
Yg,Xg=np.meshgrid((np.arange(D)+.5)/D-.5,(np.arange(D)+.5)/D-.5,indexing='ij');crop=warps('numbers',glob,geo,D,1.21,1.25)
# The network may have to accommodate two widths for an old vs new DIN face,
# but a two-digit inscription cannot have an arbitrarily long/narrow box.
rm=reshape('number_strip',layer('number_contrast','norm',crop),1,B*D,D)
numcls=[0,1,2,3,4,5,7,8]; digitw=[]; gbias=[];V=4
for i in numcls:
    for v in range(V):
        m=make_symbol(i,v)[0]; iy,ix=np.nonzero(m); xmin,xmax=(ix.min()+.5)/N*2-1,(ix.max()+.5)/N*2-1;ymin,ymax=(iy.min()+.5)/N*2-1,(iy.max()+.5)/N*2-1
        X=Xg*(xmax-xmin)*1.21+.5*(xmax+xmin);Y=Yg*(ymax-ymin)*1.25+.5*(ymax+ymin)
        glyph=sum(raster(m,X+tx*(xmax-xmin)/D,Y+ty*(ymax-ymin)/D) for tx in (-.35,0,.35) for ty in (-.35,0,.35))/9
        glyph=blur(glyph,.5);glyph=glyph.mean()-glyph
        # Whole-digit centered normalized correlation; a soft discriminant
        # can value the changing tens digit slightly more than the common 0.
        glyph*=1.-.23*(Xg>.08);glyph-=glyph.mean();glyph/=np.linalg.norm(glyph)+1e-8;digitw.append(glyph/D)
        aspect=(xmax-xmin)/(ymax-ymin);gbias.append([-.33*max(0,abs(np.log(width/height/aspect))-.14)-.014*(abs(dx)+abs(dy)) for width,height,dx,dy in geo])
ocr=conv('read_numbers',rm,np.array(digitw)[:,None,:,:],stride=D);ocr=reshape('number_hypotheses',ocr,len(numcls),V,B)
# Constant bias broadcast using a linear layer with a single zero input weight.
cb=reshape('glyph_pose_penalty',dense('number_box_priors',pool('light_dc',glob,32,avg=True),np.zeros((len(numcls)*V*B,1)),np.array(gbias).ravel()),len(numcls),V,B)
ocr=pool('best_number',add('constrained_number',ocr,cb),V,B)

# Explicit emptiness cue, invariant to global exposure: low central variation
# favors a plain prohibitory disc or the uninscribed inverted triangle.
em=rel('unsigned_ink',point('both_polarities',z,[[1,0],[-1,0]])); y_,x_=np.mgrid[:32,:32]-15.5
cw=np.exp(-.5*((x_/4.)**2+(y_/3.)**2));cw/=cw.sum();em=dense('center_activity',em,np.tile(cw.ravel(),(1,2)))

def group_for(i):
    if i==6:return 'end80'
    if i<=8:return 'ring'
    if i in (9,10):return ['pass','truckpass'][i-9]
    if i==11 or 18<=i<=31:return 'warning'
    if 12<=i<=17:return ['diamond','yield','stop','nopass','truck','entry'][i-12]
    if i==32:return 'end'
    if i>=41:return 'endphp'
    return 'blue'

# Joint registration: a candidate sign is warped *once*, and its border,
# central ink, and surface colors must all agree at this same pose. Rather
# than a giant dense coordinate transform, factor the resampling into two
# one-dimensional linear interpolations (horizontal then vertical).
D2=24
ux=(np.arange(D2)+.5)/D2*2.08-1.04; X,Y=np.meshgrid(ux,ux)
poses=[(rx*asp,rx/asp,dx,dy) for rx in (8.2,9.7,11.3,12.9,14.6) for asp in (.93,1.06) for dy in (-2.,0.,2.) for dx in (-2.,0.,2.)]; J=len(poses)
rect=warps('sign',glob,poses,D2,2.08,2.08)
rect=reshape('sign_strip',layer('sign_contrast','norm',rect),1,J*D2,D2)

# Candidate chroma log-likelihood: the expected colored areas should have
# the right signed hue, *and* nominally white areas should not. With no hue
# evidence the class-specific color score naturally goes toward zero.
# Use opponent chroma relative to the central white surface as well as an
# absolute term. Restrict attention to the modeled surface inside the sign.
cwts=[]; jointw=[]; jstyles=4
for cls in range(43):
    group=group_for(cls);iswarn=group=='warning';base='ring' if group in ('pass','truckpass','truck','nopass') else group
    m0=make_symbol(cls,0)[0]; colrows=[]
    for rx,ry,dx,dy in poses:
        xx=(fx-dx)/rx; yy=(fy-dy)/ry;ll,red,blue,yellow,outer=surfaces(group,xx,yy)
        if cls==26:red+=np.exp(-((xx/.12)**2+((yy+.20)/.12)**2));yellow+=.7*np.exp(-((xx/.12)**2+((yy-.1)/.12)**2))
        if 33<=cls<=40:blue*=1-small_render(m0,xx,yy,rx*.5,ry*.5)
        # Labeled-colored vs neutral pixels: amplitude-squared expected for
        # each color is not charged in the absence of signal. Area weights
        # trade off small (noise-prone) and large uniform color regions.
        def cweight(mask,weight):
            if np.max(mask)<.01:return -weight*.23*outer/(outer.sum()+1.)
            return weight*(mask/(mask.sum()+1.)-.67*np.maximum(outer-mask,0)/(np.maximum(outer-mask,0).sum()+1.))
        colrows.append(np.stack([0*ll,cweight(red,1.45),cweight(blue,1.75),cweight(yellow,1.70)]).ravel())
    cwts.extend(colrows)
    for v in range(jstyles):
        m,area,light=make_symbol(cls,v);X1=X.copy();Y1=Y.copy()
        if iswarn:
            sc=(.88,1.,1.12,.98)[v];X1=X/sc;Y1=(Y-.18)/sc+.18
            if cls in [27,28,29] and v%2:X1=-X1
        if cls in (33,34,35,36,37,38,39,40):X1/= (.90,1.,1.12,.98)[v];Y1/=(.90,1.,1.12,.98)[v]
        m=blur(small_render(m,X1,Y1,11,11),.48)
        # Smooth material design. Ignore unknown background in the template.
        _,red,blue,yellow,outer=surfaces(base,X,Y);d=np.hypot(X,Y);l=np.zeros_like(d)
        if base=='warning' or base=='yield' or base=='ring':
            l=outer*.85-red*1.35; l-=m*(.85 if cls<=8 else 1.55)
        if base=='diamond':l=.18*outer+.95*m
        if base=='blue':l=-.72*outer+1.82*m
        if base in ('stop','entry'):l=-.70*outer+1.75*m
        if base.startswith('end'):l=.92*outer-1.55*m
        activity=np.minimum(1.,blur(outer,.6)*1.2); # mildly penalize exterior mismatches
        active=activity+.11*(1-activity);l=l-active*l.sum()/active.sum()
        if iswarn:l*=.68+.60*((Y<.71)&(abs(X)<.52*(Y+.62)));l-=active*l.sum()/active.sum()
        if base in ('blue','stop','entry'): # unknown background around disc
            l=(m-(outer-m)*m.sum()/max(1.,(outer-m).sum()));l=blur(l,.35)
        l/=np.linalg.norm(l)+1e-8;jointw.append(l/D2)
JL=conv('read_whole_sign',rect,np.array(jointw)[:,None,:,:],stride=D2);JL=reshape('rendered_matches',JL,43,jstyles,J);JL=pool('choose_drawing',JL,jstyles,1)
JC=reshape('pose_colors',dense('match_materials',c,np.array(cwts)),43,1,J)
# Oriented structural gradients (six signed axes and their opposites) give
# sharp boundaries evidence even when foreground and background colors are
# nearly equal on average. Compare gradients across a small neighborhood,
# not at an exact pixel, using a blurred dual stencil.
axis=6;ang=np.arange(2*axis)*np.pi/axis;gy0,gx0=np.mgrid[-2:3,-2:3];grad=np.array([(gx0*np.cos(t)+gy0*np.sin(t))*np.exp(-(gx0**2+gy0**2)/2.15) for t in ang]);grad/=np.abs(grad[0]).sum()/2
ge=pool('edge_cells',rel('edge_polarities',conv('oriented_gradients',glob,grad[:,None,:,:],pad=2)),2,avg=True)
# Only the *structural* edge signature is used in this branch. The fine
# pictogram edges are a separate cue at higher resolution. In particular
# a yield sign has an uninterrupted horizontal inner edge at the top.
edge_groups=['ring','warning','yield','diamond','blue','stop','entry','end'];eb=[]
for group in edge_groups:
    for rx,ry,dx,dy in poses:
        x=(fx-dx)/rx;y=(fy-dy)/ry; l,red,blue,yellow,outer=surfaces(group,x,y)
        if group=='ring':l=outer-1.30*red
        if group=='warning':l=outer-1.40*red
        if group=='diamond':l=outer-.62*yellow
        if group=='blue':l=-outer
        a=blur(l,.49);gy,gx=np.gradient(a);gn=np.hypot(gy,gx);sig=np.minimum(1,blur(outer,.5)*1.6)
        # For bright vs dark exterior, retain weak unoriented gradients as
        # well as the stronger signed gradients within a sign.
        h=[]
        for theta in ang:
            dot=(gx*np.cos(theta)+gy*np.sin(theta))/(gn+.0001);h.append(blur(sig*gn*(np.maximum(dot,0)**4+.19*abs(dot)**4-.22),.48))
        h=np.array(h);h/=max(np.abs(h).sum()*.24,.001);eb.append(h.ravel())
EH=reshape('edge_signatures',dense('match_boundary_edges',ge,np.array(eb)),len(edge_groups),1,J)
ET=np.zeros((43,len(edge_groups)))
for i in range(43):
    gg=group_for(i);gg=gg if gg in edge_groups else ('end' if gg.startswith('end') else 'ring');ET[i,edge_groups.index(gg)]=1.
JE=point('boundary_agreement',EH,ET)

# An additional pictogram comparison in the registered coordinates gives
# similar warning signs a common triangular white field. The normalization
# here is local to that field, *not* to the whole sign and its surroundings.
# A rectangular region in the lower center is safe for most triangle icons;
# zeros outside the inscribed triangle must not reward matching the border.
wi=[11]+list(range(18,32));spx,spy=np.meshgrid((np.arange(18)+.5)/18-.5,(np.arange(18)+.5)/18-.5);WX=spx*1.28;WY=spy*1.12+.145
# Sample the symbol with the offset of its field relative to the sign center.
wp=[(rx,ry,dx,dy+ry*.145) for rx,ry,dx,dy in poses];wpix=warps('warning',glob,wp,18,1.28,1.12)
# Restrict the support of each zero-mean symbol kernel to the interior;
# its score should not come from a coincidental match to the colored rim.
ws=reshape('warning_strip',layer('warning_contrast','norm',wpix),1,J*18,18); wrows=[]; partial=[]; wstyles=8
for i in wi:
    for v in range(wstyles):
        m,area,_=make_symbol(i,v%4); scale=(.86,1.,1.13,1.02)[v%4]; angle=(-.055,.05,.015,0)[v%4];x=(WX+angle*(WY-.15))/scale;y=(WY-.15-angle*WX)/scale+.15
        if v>=4 and i in (23,27,28,29):x=-x
        if v>=4:y+=.075
        mm=blur(small_render(m,x,y,15,16),.60);ar=np.maximum(mm,small_render(area,WX,WY,15,16));smooth_m=blur(mm,1.7);h=mm-.53*smooth_m;h-=ar*h.sum()/(ar.sum()+1.e-6);h/=np.linalg.norm(h)+1.e-7;wrows.append(-h/14.)
        # Compare head/torso and lower-left/right parts independently. Each
        # subfilter is zero-sum and only sees the white field. Small shifts in
        # the drawing need not displace every limb or wheel at once.
        parts=[]
        for cx,cy,ww,hh in [(0,-.03,.31,.19),(-.21,.36,.22,.18),(.21,.36,.22,.18)]:
            row=[];mx=np.exp(-.5*(((WX-cx)/ww)**2+((WY-cy)/hh)**2));reg=ar*mx
            for dx0,dy0 in [(0,0),(-.080,0),(.080,0),(0,-.068),(0,.068)]:
                mh=blur(small_render(m,x-dx0,y-dy0,15,16),.79);fh=reg*(mh-(mh*reg).sum()/(reg.sum()+1.e-6));fh/=max(1.35,np.linalg.norm(fh));row.append((-fh/9.).ravel())
            parts.append(row)
        partial.append(parts)
wl=conv('warning_symbols',ws,np.array(wrows)[:,None,:,:],stride=18);wl=pool('warning_style',reshape('warning_hypotheses',wl,len(wi),wstyles,J),wstyles,1)
wf=np.zeros((43,len(wi)));wf[wi,np.arange(len(wi))]=1.;JF=point('warning_detail',wl,wf,np.array([-.28 if i in wi else 0. for i in range(43)]))
parts=conv('read_icon_parts',ws,np.asarray(partial).reshape(-1,1,18,18),stride=18);parts=pool('choose_part_shift',reshape('part_offsets',parts,len(wi)*wstyles*3,5,J),5,1);parts=pool('average_icon_parts',reshape('components',parts,len(wi)*wstyles,3,J),3,1,avg=True);parts=pool('choose_part_drawing',reshape('assembled_icons',parts,len(wi),wstyles,J),wstyles,1);JP=point('deformable_warning',parts,wf*.81,np.array([-.18 if i in wi else 0. for i in range(43)]))
# A second reading of the number, this time relative to the *same circle*
# used for the chroma and boundary tests. The surrounding clear space is
# part of a number, preventing the 20 within 120 from being a perfect 20.
Dn=18;ng=(np.arange(Dn)+.5)/Dn-.5;NX,NY=np.meshgrid(ng*1.49,ng*1.17)
nchip=warps('inscription',glob,poses,Dn,1.49,1.17);nstrip=reshape('inscription_strip',layer('inscription_contrast','norm',nchip),1,J*Dn,Dn)
nw=[]; nstyles=32
for idx in numcls:
    masks=[make_symbol(idx,v)[0] for v in range(4)]
    for v in range(nstyles):
        stretch=(.90,1.12)[v//4%2];height=(.90,1.06)[v//8%2];ch=masks[v%4];xx=(NX-(.025 if v//16 else -.025))/stretch;yy=NY/height
        glyph=blur(small_render(ch,xx,yy,Dn/1.49,Dn/1.17),.58); a0=np.clip((.80-np.sqrt(NX*NX+NY*NY))/.085,0,1); h=a0*(glyph-(glyph*a0).sum()/a0.sum());h/=np.linalg.norm(h)+1.e-8;nw.append(-h/Dn)
rn=conv('registered_digits',nstrip,np.array(nw)[:,None,:,:],stride=Dn);rn=pool('inscription_style',reshape('number_styles',rn,len(numcls),nstyles,J),nstyles,1)
nf=np.zeros((43,len(numcls)));nf[numcls,np.arange(len(numcls))]=1.8;JN=point('registered_ocr',rn,nf,np.array([-.65 if i in numcls else 0 for i in range(43)]))

# A cancelled restriction has a slash, extending *beyond* the digit or car.
# Testing its ends separately keeps a single dark diagonal in a numeral or
# a piece of background from sufficing. The average minus half the absolute
# difference is the minimum of the two half scores (computed with ReLUs).
stripe=[]
for a0 in (.25*np.pi-.09,.25*np.pi,.25*np.pi+.09):
    cross=X*np.sin(a0)+Y*np.cos(a0);along=X*np.cos(a0)-Y*np.sin(a0)
    for sg in (-1,1):
        window=np.clip((sg*along-.41)/.14,0,1)*np.clip((.99-sg*along)/.14,0,1);dark=window*np.exp(-.5*(cross/.087)**2);flank=window*(np.exp(-.5*((cross-.225)/.08)**2)+np.exp(-.5*((cross+.225)/.08)**2));stripe.append(flank/(flank.sum()+1.e-7)-dark/(dark.sum()+1.e-7))
sr=conv('stripe_ends',rect,np.array(stripe)[:,None,:,:],stride=D2);neg=[];bases=[]
for i in range(3):
    v=np.zeros(6);v[2*i:2*i+2]=[1,-1];neg.append(v);v=np.zeros(6);v[2*i]=1;bases.append(v)
small=point('both_stripe_ends',rel('end_difference',point('compare_stripe_ends',sr,neg)),-np.eye(3));sr=add('weaker_end',small,point('stripe_start',sr,bases));sr=pool('stripe_angle',reshape('stripe_hypotheses',sr,1,3,J),3,1)
sf=np.zeros((43,1));sf[[6,32,41,42],0]=1.25;JS=point('cancellation',sr,sf,-sf.ravel()*.25)

# Internal hues are unusually informative for a signal warning: red, yellow
# and green lights in that order. Compare the lamp positions against nearby
# neutral surface. Also distinguish one wider red truck from a compact car.
hvec=[]
for rx,ry,dx,dy in poses:
    x=(fx-dx)/rx;y=(fy-dy)/ry
    def local(a,b,wx,wy):
        spot=np.exp(-.5*(((x-a)/wx)**2+((y-b)/wy)**2));return spot/(spot.sum()+1.e-7)
    def lamp(y0):return local(0,y0,.094,.096)-.5*(local(-.215,y0,.082,.08)+local(.215,y0,.082,.08))
    top=lamp(-.21);mid=lamp(.10);bot=lamp(.415);lamps=np.stack([0*x,top-.90*bot-.38*mid,-.25*(mid+bot),.65*mid+.31*bot-.24*top])
    load=.45*local(-.45,-.235,.10,.08)+.4*local(-.15,-.25,.09,.07)+.15*local(-.035,-.075,.06,.11);clear=.55*local(-.03,-.38,.11,.08)+.45*local(-.28,.34,.10,.065);load-=clear
    hvec.append([lamps.ravel(),np.stack([-.25*load,load,0*x,0*x]).ravel()])
colorsig=reshape('symbol_hue_patterns',dense('symbol_colors',c,np.array(hvec).transpose(1,0,2).reshape(2*J,-1)),2,1,J);sigf=np.zeros((43,2));sigf[26,0]=1.25;sigf[10,1]=.72;sigf[9,1]=-.24;JC2=point('pictogram_color_evidence',colorsig,sigf)
# A coarse boundary aids alignment. Apart from triangles, the exact edge
# polarity of the background is unknown, so don't use it as a class bias.
wc=np.eye(43)*.32;wc[13,13]=.85;wc[12,12]=.55
joint=pool('registered_match',add('joint_pose',point('drawing_agreement',JL,np.eye(43)*3.2),JC,point('boundary_weight',JE,wc),JF,JP,JN,JS,JC2),1,J)
# Class-family and symbol logit streams. Compete on family evidence first,
# then compare symbols to their siblings under common appearance conditions.
M=np.zeros((43,G));T=np.zeros((43,len(present)));bias=np.zeros(43);E=np.zeros((43,1));O=np.zeros((43,len(numcls)))
for i in range(43):
    M[i,groups.index(group_for(i))]=.35
    T[i,present.index(i)]=.30
    if i in numcls:
        T[i,present.index(i)]=.12;O[i,numcls.index(i)]=6.;bias[i]=-2.
    if i==15:bias[i]=1.05;E[i,0]=-2.5
    if i==13:bias[i]=.5;E[i,0]=-.65
    if i in (6,32,41,42):bias[i]=.40
    if i==14:bias[i]=-.35
    if i in range(33,41):T[i,present.index(i)]=.17
shape_scores=dense('shape_evidence',f,M);sym_scores=dense('symbol_evidence',r,T,bias);out=add('scores',shape_scores,sym_scores,dense('registered_evidence',joint,np.eye(43)*2.8),dense('ocr_evidence',ocr,O),dense('blank_center',em,E))

folder=Path(sys.argv[1]);folder.mkdir(parents=True,exist_ok=True);(folder/'network.json').write_text(json.dumps({'layers':L},separators=(',',':'))+'\n');np.savez(folder/'weights.npz',**W)
