"""Formula-built traffic sign reader. No resources, font files or example data."""
import json, sys, math
from pathlib import Path
import numpy as np

P=Path(sys.argv[1]); P.mkdir(parents=True,exist_ok=True)
W={}; layers=[]
def layer(n,op,src,**kw):
    layers.append(dict(name=n, op=op, inputs=src if isinstance(src,list) else [src], **kw));return n
def matrix(n,src,m,b=0,conv=False):
    m=np.asarray(m,dtype=np.float32); o=len(m)
    if conv:
        m=m.reshape(o,-1,1,1);layer(n,'conv',src,out=o,kernel=[1,1],padding=0,stride=1)
    else:layer(n,'dense',src,out=o)
    W[n+'.W']=m;W[n+'.b']=np.broadcast_to(np.asarray(b,np.float32),(o,)).copy();return n
def pool(n,src,h,w,op='maxpool',sy=None,sx=None):
    return layer(n,op,src,size=[h,w],stride=[sy or h,sx or w])
def shape(n,src,*s):return layer(n,'reshape',src,shape=list(s))
def norm(n,src):return layer(n,'norm',src)
def add(n,*ins):return layer(n,'add',list(ins))
def dot_rows(n,src,m):return matrix(n,src,np.stack(m))

# Approximate a log photoreceptor response with a concave piecewise-linear map.
# All lighting adaptation in this build is applied anew to the *input* at inference.
knots=np.array([0,.025,.065,.14,.29,.55,1.0]); slopes=np.diff(np.log(knots+.025))/np.diff(knots)
ramp=matrix('exposure_knots','image',np.repeat(np.eye(3),len(slopes),axis=0),-np.tile(knots[:-1],3),True)
ramp=layer('positive_ramps','relu',ramp)
curves=np.kron(np.eye(3),np.r_[slopes[0],np.diff(slopes)])
# Signed red/blue and yellow opponent channels, plus a brightness channel.
op=np.array([[.27,.55,.18],[1,-.62,-.38],[-.53,-.47,1],[.44,.56,-1]])
photo=matrix('perceptual_channels',ramp,op@curves,conv=True)
std=norm('adapted_channels',photo)
# Saturate far outliers (in particular, sky and specular highlights).
def clipped(n,src,c,bound):
    a=matrix(n+'_knees',src,np.r_[np.eye(c),np.eye(c)],np.r_[np.full(c,bound),np.full(c,-bound)],True)
    a=layer(n+'_ramps','relu',a)
    return matrix(n, a, np.c_[np.eye(c),-np.eye(c)],-bound,True)
std=clipped('limited',std,4,2.4)
# Keep absolute hue as well as within-image contrast. A nearly neutral sign
# should not acquire a saturated border just from a tiny color difference.
abscol=matrix('absolute_opponency',photo,np.diag([0,.72,.72,.72]),conv=True)
chrom=add('color_and_contrast',std,abscol)
coarse=pool('low_resolution_color',chrom,2,2,'avgpool')

# Geometric poses are offsets in pixels and radii in pixels, based on a sign
# filling most of the square with a small border. All banks use a fixed grid.
poses=[(r,dx,dy) for r in [10.0,11.8,13.6,15.1] for dy in [-2,0,2] for dx in [-2,0,2]]
Q=len(poses)
families=['ring','warning','blue','cancel','yield','priority','stop','entry']
F=len(families)
family=[0]*11+[1,5,4,6,0,0,7]+[1]*14+[3]+[2]*8+[3,3];family[6]=3
assert len(family)==43

def blur(z,t):
    if t<=.01:return z
    h=int(np.ceil(3*t));g=np.exp(-.5*(np.arange(-h,h+1)/t)**2);g/=g.sum()
    for ax in [0,1]:z=np.apply_along_axis(lambda a:np.convolve(np.pad(a,h,mode='edge'),g,'valid'),ax,z)
    return z

def smoothstep(x):
    u=np.clip(x,0,1);return u*u*(3-2*u)
def under(x,r,aa):return smoothstep(.5+(r-x)/aa)

def triangle(x,y):return np.maximum(y,2*np.abs(x)-y) # center of scaling is halfway up: equals 1 on all edges

def rim(f,x,y,aa):
    rho=np.hypot(x,y)
    def cut(z,v):return under(z,v,aa)
    if f in [0,2,3,6,7]:
        level=rho if f!=6 else np.maximum.reduce([abs(x),abs(y),(abs(x)+abs(y))/np.sqrt(2)])
        outer=cut(level,1); inner=cut(level,.775)
        # Exterior plus the very centre are nuisance pixels for a border detector.
        support=cut(level,1.13)*(1-.91*cut(level,.57))
        if f==0:return outer,inner,outer-inner,support
        if f==2:return outer,.13*outer,-outer,support
        if f==3:return outer,.91*outer,np.zeros_like(x),cut(level,1.10)
        if f==6:return outer,.28*outer,outer,cut(level,1.08)
        bar=cut(abs(y),.12)*cut(abs(x),.83)*outer
        return outer,.23*outer+.74*bar,outer-bar,cut(level,1.09)
    if f==5:
        level=abs(x)+abs(y);outer=cut(level,1.02);inner=cut(level,.73)
        return outer,.95*outer-.20*inner,inner,cut(level,1.25)
    Y=y if f==1 else -y
    level=triangle(x,Y);outer=cut(level,1)
    # Inset of each of the three sides, parallel to the original side.
    inner=cut(Y,.79)*cut(2*abs(x)-Y,.53)
    core=cut(Y,.64)*cut(2*abs(x)-Y,.26)
    return outer,inner,outer-inner,cut(level,1.2)*(1-.84*core)

def unit(a):return a/max(float(np.sqrt(np.sum(a*a))),1e-8)
def contrast(a,mask):return unit(mask*(a-np.sum(mask*a)/max(np.sum(mask),1e-8)))
# Each bank row is an independent, antialiased geometric template. Fixed
# coefficients specify color semantics; no calibration against other rows.
detail=4
sub=(np.arange(16*detail)+.5)*2/detail-16;X,Y=np.meshgrid(sub,sub)
def small(a):return a.reshape(16,detail,16,detail).mean(axis=(1,3))
border=[]
for f in range(F):
 for r,dx,dy in poses:
    x=(X-dx)/r;y=(Y-dy)/r;o,l,c,mask=rim(f,x,y,.12)
    # Grey exterior, dark colored rim or face and a lighter interior.
    light=small(.49*(1-o)+(.25*o+.70*l if f in (0,1,4) else l))
    mask=small(mask); lum=contrast(light,mask)
    coeff=np.zeros((4,16,16)); area=math.sqrt(55.)
    if f in (0,1,4):
        coeff[0]=.73*lum/area;coeff[1]=.45*contrast(small(c),mask)/area
        coeff[1]+=.39*small(c)/max(1,small(c).sum())
    elif f==2:
        coeff[0]=.48*lum/area;coeff[2]=.50*contrast(small(-c),mask)/area+.69*small(-c)/small(-c).sum()
    elif f==3:
        coeff[0]=.60*lum/area
        # A white, rimless disk, without the saturated red rim of a speed sign
        ann=small(o-under(np.hypot(x,y),.79,.12));coeff[1]=-.42*ann/ann.sum();coeff[2]=-.16*small(o)/small(o).sum()
    elif f==5:
        coeff[0]=.55*lum/area;coeff[3]=.55*contrast(small(c),mask)/area+.75*small(c)/small(c).sum()
    else:
        coeff[0]=(.73 if f==7 else .32)*lum/area;coeff[1]=.56*contrast(small(c),mask)/area+.77*small(c)/small(c).sum()
    border.append(coeff.flatten())
border=dot_rows('frame_evidence',coarse,border)

# Hand drawing primitives. Coordinates are sign-relative, not photo-relative.
N=128; grid=(np.arange(N)+.5-N/2)/(N/3); U,V=np.meshgrid(grid,grid)
class Pen:
 def __init__(self,x,y):self.x=x;self.y=y;self.a=np.zeros_like(x)
 def put(self,m,v=1):self.a=np.where(m,v,self.a)
 def poly(self,p,v=1):
    x=self.x;y=self.y;mask=np.zeros(x.shape,bool);px,py=p[-1]
    for qx,qy in p:
        if qy!=py:mask^=((py>y)!=(qy>y))&(x<(qx-px)*(y-py)/(qy-py)+px)
        px,py=qx,qy
    self.put(mask,v)
 def line(self,p,w,v=1):
    for (x0,y0),(x1,y1) in zip(p,p[1:]):
        dx=x1-x0;dy=y1-y0;t=np.clip(((self.x-x0)*dx+(self.y-y0)*dy)/max(1e-9,dx*dx+dy*dy),0,1)
        self.put((self.x-(x0+t*dx))**2+(self.y-(y0+t*dy))**2<(w/2)**2,v)
 def rect(self,x0,y0,x1,y1,v=1):self.put((self.x>=x0)&(self.x<=x1)&(self.y>=y0)&(self.y<=y1),v)
 def oval(self,x,y,rx,ry,v=1):self.put(((self.x-x)/rx)**2+((self.y-y)/ry)**2<=1,v)
 def circle(self,x,y,r,v=1):self.oval(x,y,r,r,v)
 def cubic(self,p,w,v=1):
    pts=[p[0]]
    for i in range(1,len(p),3):
        a,b,c,d=np.array([p[i-1],*p[i:i+3]])
        for t in np.linspace(0,1,12)[1:]:pts.append(tuple((1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d))
    self.line(pts,w,v)

def digit(p,d,variant):
 # Digit coordinates in a 1 by 1 em box; drawn at a condensed aspect ratio.
 # Draw centre lines in physical height units with round joins.
 aspect=.57 if variant%2==0 else .66
 oldx=p.x;p.x=p.x*aspect
 def cb(coords,w=1):p.cubic([(a*aspect,b) for a,b in coords],(.135 if variant<2 else .105)*w)
 def ln(coords,w=1):p.line([(a*aspect,b) for a,b in coords],(.135 if variant<2 else .105)*w)
 if d=='0':
    cb([(.5,.07),(.80,.07),(.88,.18),(.88,.36),(.90,.64),(.92,.93),(.50,.93),(.10,.93),(.10,.77),(.12,.43),(.12,.17),(.20,.07),(.50,.07)])
 elif d=='1':
    ln([(.16,.24),(.55,.07),(.55,.92)])
    if variant%2:ln([(.33,.93),(.78,.93)],.8)
 elif d=='2':
    cb([(.13,.27),(.14,.01),(.86,-.03),(.86,.25),(.88,.40),(.67,.50),(.50,.63)])
    ln([(.50,.63),(.13,.91),(.92,.91)])
 elif d=='3':
    cb([(.13,.18),(.36,-.01),(.84,.04),(.84,.27),(.85,.40),(.67,.47),(.40,.47),(.71,.43),(.92,.56),(.88,.74),(.88,.94),(.41,1.02),(.12,.83)])
 elif d=='5':
    ln([(.84,.08),(.22,.08),(.17,.47)])
    cb([(.17,.47),(.46,.32),(.91,.43),(.87,.67),(.88,.96),(.40,1.02),(.12,.83)])
 elif d=='6':
    if variant%2:cb([(.82,.13),(.37,-.10),(.12,.27),(.13,.63)])
    else:ln([(.68,.07),(.31,.38),(.16,.56)])
    cb([(.16,.57),(.36,.38),(.88,.40),(.88,.68),(.90,.98),(.14,1.05),(.13,.72),(.12,.64),(.13,.59),(.16,.57)])
 elif d=='7':ln([(.10,.07),(.91,.07),(.34,.92)])
 elif d=='8':
    if variant%2:
        # Conventional double bowl, with a broad waist shared by both bowls.
        cb([(.5,.47),(.10,.47),(.09,.33),(.10,.23),(.11,.02),(.90,-.015),(.90,.25),(.90,.41),(.81,.47),(.5,.47)])
        cb([(.5,.47),(.08,.47),(.08,.61),(.08,.74),(.08,1.),(.92,1.),(.92,.74),(.92,.60),(.94,.47),(.5,.47)])
    else:cb([(.50,.46),(.07,.36),(.06,.06),(.50,.06),(.94,.06),(.94,.36),(.50,.46),(.06,.55),(-.02,.93),(.50,.94),(1.01,.93),(.96,.55),(.50,.46)])
 p.x=oldx

def arrow(p,tip,tail,width,head,headlen):
    b=np.array(tip,dtype=float);a=np.array(tail,dtype=float);d=(b-a)/np.linalg.norm(b-a);n=np.array([-d[1],d[0]]);s=b-d*headlen
    p.poly([a+n*width/2,s+n*width/2,s+n*head/2,b,s-n*head/2,s-n*width/2,a-n*width/2])

def frontal_car(p,x,y,s,shade):
    pts=[(-.24,-.03),(-.18,-.20),(-.12,-.24),(.13,-.24),(.19,-.2),(.24,-.03),(.27,.01),(.27,.17),(.21,.17),(.21,.24),(.12,.24),(.12,.18),(-.13,.18),(-.13,.24),(-.22,.24),(-.22,.16),(-.27,.16),(-.27,.01)]
    p.poly([(x+a*s,y+b*s) for a,b in pts],shade)
    p.poly([(x+a*s,y+b*s) for a,b in [(-.19,-.04),(-.13,-.18),(.12,-.18),(.18,-.04)]],.06)

def lorry(p,x,y,s,shade):
    p.rect(x-.46*s,y-.23*s,x+.10*s,y+.13*s,shade)
    p.poly([(x+a*s,y+b*s) for a,b in [(.14,-.13),(.32,-.13),(.44,.00),(.44,.15),(.14,.15)]],shade)
    p.rect(x-.44*s,y+.10*s,x+.43*s,y+.17*s,shade)
    for a in [-.29,.29]:p.circle(x+a*s,y+.16*s,.095*s,shade)
    p.poly([(x+a*s,y+b*s) for a,b in [(.19,-.09),(.29,-.09),(.36,0),(.19,0)]],0)

def walking(p,x,y,s,flip=1):
    def line(a,w):p.line([(x+flip*s*i,y+j*s) for i,j in a],w*s)
    p.circle(x+.11*flip*s,y-.31*s,.084*s)
    line([(.06,-.18),(-.035,.035)],.148)
    line([(.01,-.12),(-.16,-.025),(-.22,.06)],.071)
    line([(.06,-.12),(.16,-.02),(.28,.00)],.070)
    line([(-.035,.015),(.055,.17),(.23,.36)],.086)
    line([(-.04,.02),(-.10,.20),(-.24,.37)],.096)

def render(k,sty):
    x=U.copy();y=V.copy()
    if k in [11,*range(18,32)]:y=y-.22
    if k in [6,41,42]:
        # Without a red surround the canceled restriction can use a larger
        # legend, overlaid by the diagonal bands over the *whole* disk.
        x=x/(1.19 if sty%2 else 1.33);y=y/(1.13 if sty%2 else 1.24)
    # Use handedness, not unrelated templates, for mirror-related signs.
    if k in [20,34,37,39]:x=-x
    p=Pen(x,y);white_arrow=False
    if k in [0,1,2,3,4,5,6,7,8]:
        text=['20','30','50','60','70','80','80','100','120'][k];ht=.84 if len(text)==2 else .76
        widths=[((.33 if sty%2==0 else .27) if d=='1' else (.43 if len(text)==2 else .365)) for d in text];gap=.065 if len(text)==2 else .047
        start=-.5*(sum(widths)+gap*(len(text)-1))
        for d,w in zip(text,widths):
            q=Pen((x-start)/w,y/ht+.5);digit(q,d,sty);p.a=np.maximum(p.a,q.a);start+=w+gap
    elif k in (9,10,41,42):
        frontal_car(p,.33,.03,1.03,1)
        if k in (9,41):frontal_car(p,-.33,.0,1.02,.68)
        else:
            p.rect(-.68,-.31,-.25,.19,.75);p.poly([(-.24,-.20),(-.12,-.2),(-.035,-.07),(-.035,.2),(-.24,.2)],.75)
            p.rect(-.64,.13,-.06,.24,.75);p.circle(-.54,.23,.066,.74);p.circle(-.14,.23,.066,.74)
    elif k==11:
        p.poly([(-.12,.44),(.12,.44),(.12,.11),(.36,.11),(.36,-.04),(.12,-.04),(.12,-.18),(.0,-.48),(-.12,-.18),(-.12,-.04),(-.36,-.04),(-.36,.11),(-.12,.11)])
    elif k==14:
        # Block lettering, drawn without using a font file.
        xs=[-.64,-.30,.04,.38];ww=.255;top=-.24;bot=.24
        letters=[[(1,0),(.18,0),(0,.15),(0,.40),(.18,.5),(.84,.5),(1,.62),(1,.87),(.84,1),(0,1)],[(0,0),(1,0),(.5,0),(.5,1)],[(.18,0),(.82,0),(1,.14),(1,.86),(.82,1),(.18,1),(0,.86),(0,.14),(.18,0)],[(0,1),(0,0),(.75,0),(1,.13),(1,.40),(.78,.52),(0,.52)]]
        for xx,seq in zip(xs,letters):p.line([(xx+ww*a,top+(bot-top)*b) for a,b in seq],.066)
        white_arrow=True
    elif k==16:
        p.x=-p.x;lorry(p,0,0,1.45,1)
    elif k==17:
        p.rect(-.82,-.118,.82,.118);white_arrow=True
    elif k==18:
        p.poly([(-.090,-.55),(.090,-.55),(.067,-.06),(.044,.22),(-.044,.22),(-.067,-.06)]);p.circle(0,.37,.078)
    elif k==19 or k==20:
        p.cubic([(.21,.44),(.21,.33),(.21,.20),(.21,.10),(.21,-.12),(.07,-.26),(-.10,-.26),(-.16,-.26),(-.21,-.26),(-.26,-.26)],.137)
        if sty%2:p.rect(-.28,-.329,-.16,-.195)
    elif k==21:
        if sty%2:
            p.line([(.19,.43),(.19,.025),(-.16,.025),(-.16,-.22),(-.09,-.44)],.14)
        else:p.cubic([(.19,.44),(.18,.17),(.23,.06),(.035,.04),(-.27,.02),(-.18,-.18),(-.12,-.42)],.14)
    elif k==22:
        p.rect(-.48,.29,.48,.37)
        for a in [-.20,.18]:p.oval(a,.30,.22,.19)
    elif k==23:
        xx=p.x; yy=p.y;p.x=.955*xx+.296*yy;p.y=-.296*xx+.955*yy
        frontal_car(p,.005,-.15,1.17,1);p.x=xx;p.y=yy
        for b in [-.18,.18]:p.cubic([(b+.02,.09),(b+.18,.15),(b-.24,.25),(b-.04,.36),(b+.01,.40),(b+.08,.42),(b+.07,.47)],.047)
    elif k==24:
        p.line([(-.22,.45),(-.22,-.40)],.085);p.cubic([(.25,.45),(.26,.03),(.08,-.07),(.08,-.40)],.092)
    elif k==25:
        p.circle(.13,-.32,.076);p.line([(.085,-.21),(-.09,-.06)],.14)
        p.line([(-.09,-.06),(-.13,.15),(-.27,.40)],.091);p.line([(-.06,-.03),(.02,.12),(.04,.39)],.09)
        p.line([(.08,-.19),(.13,-.035),(.21,.01)],.07);p.line([(-.17,-.15),(.31,.26)],.042)
        p.poly([(.09,.43),(.25,.21),(.33,.17),(.48,.40),(.51,.43)])
    elif k==26:
        for j,v in zip([-.32,-.02,.27],[.64,.42,.88]):p.circle(0,j,.14,v)
    elif k==27:walking(p,0,.0,1.12)
    elif k==28:
        # Two children holding hands, the smaller on the left. Their bodies
        # stay distinct even when individual limbs disappear in a small crop.
        walking(p,.16,-.035,1.03,-1);walking(p,-.19,.13,.71,-1)
        p.line([(.07,-.07),(-.04,.03),(-.16,.04)],.055)
    elif k==29:
        if sty%2:p.x=-p.x
        for a in [-.29,.29]:p.circle(a,.23,.206);p.circle(a,.23,.128,0)
        p.line([(-.29,.23),(-.12,-.03),(.02,.22),(.22,-.09),(.29,.23)],.065)
        p.line([(-.29,.23),(.01,.23)],.063);p.circle(.07,-.34,.083)
        p.line([(.01,-.21),(-.14,-.06),(.02,.03),(-.03,.18)],.097)
        p.line([(.00,-.20),(.12,-.075),(.24,-.08)],.065)
    elif k==30:
        for a in np.arange(6)*np.pi/3-(np.pi/2 if sty%2 else 0):
            d=np.array([np.cos(a),np.sin(a)]);n=np.array([-d[1],d[0]])
            p.line([(0,0),d*.40],.055)
            for s in [-1,1]:p.line([d*.235,d*.32+s*.09*n],.048)
    elif k==31:
        p.x=-x*.94-y*.30;p.y=-x*.30+y*.94
        p.poly([(-.4,.10),(-.29,-.06),(-.12,-.12),(.07,-.08),(.22,-.29),(.29,-.29),(.35,-.21),(.26,-.17),(.18,.03),(.04,.11),(-.14,.11),(-.29,.17)])
        for seq in [[(-.29,.09),(-.30,.22),(-.45,.31)],[(-.20,.10),(-.10,.21),(-.17,.33)],[(.08,.05),(.20,.15),(.34,.12)],[(.11,.005),(.26,.08),(.39,.02)],[(.27,-.24),(.20,-.36),(.19,-.43)]]:p.line(seq,.048)
    elif k in [33,34]:
        p.cubic([(-.21,.66),(-.21,.04),(-.30,-.27),(.27,-.27)],.19)
        arrow(p,(.63,-.27),(.08,-.27),.19,.63,.35);white_arrow=True
    elif k in [35,36,37]:
        arrow(p,(0,-.69),(0,.68),.19,.65,.34);white_arrow=True
        if k!=35:
            p.x=x+.18;p.a[:]=0;arrow(p,(0,-.67),(0,.69),.18,.58,.34)
            p.cubic([(0,.24),(.01,.03),(.04,-.016),(.37,-.016)],.16)
            arrow(p,(.79,-.016),(.35,-.016),.17,.57,.32)
    elif k in [38,39]:arrow(p,(.44,.46),(-.46,-.46),.255,.79,.41);white_arrow=True
    elif k==40:
        for t in np.arange(3)*2*np.pi/3:
            a=np.linspace(t+.14,t+1.71,20);r=.51;p.line(np.stack([r*np.cos(a),-r*np.sin(a)],1),.115)
            v=np.array([np.cos(a[-1]),-np.sin(a[-1])]);tang=np.array([v[1],-v[0]]);arrow(p,v*r+tang*.12,v*r-tang*.12,.11,.31,.19)
        white_arrow=True
    if k in [6,32,41,42]:
        p.a*=.83
        # Close oblique parallel stripes merge to a broad grey band. The
        # grey paint on a cancellation sign can be lighter than the legend.
        u=(U+(.76 if sty%2==0 else 1.04)*V)/(1+(.76 if sty%2==0 else 1.04)**2)**.5
        stripes=abs(u)<.215
        p.a=np.maximum(p.a,(1 if k==32 else .62)*stripes*(U*U+V*V<.98**2))
    return p.a if white_arrow else -p.a

# Bilinear evaluation of a single supersampled drawing, with a small footprint
# (anti-aliasing). Only the individual symbol's own values determine its filter.
def sample(im,x,y):
    u=np.clip(x*N/3+N/2-.5,0,N-1.0001);v=np.clip(y*N/3+N/2-.5,0,N-1.0001)
    j=u.astype(int);i=v.astype(int);u-=j;v-=i
    return (1-u)*(1-v)*im[i,j]+u*(1-v)*im[i,j+1]+(1-u)*v*im[i+1,j]+u*v*im[i+1,j+1]

def ink_support(k,x,y):
    f=family[k]
    if f==1:return under(y,.75,.095)*under(2*np.abs(x)-y,.48,.12)
    if f==0 or f==3:return under(np.hypot(x,y),(.745 if f==0 else .90),.12)
    if f==2:return under(np.hypot(x,y),.82,.12)
    return under(abs(x),.77,.10)*under(abs(y),.39,.12)

# Split the image into brightness and a less red-sensitive 'ink' channel:
# a red surround stays relatively bright in red light, so it interferes less
# with reading black-on-white characters. A zero-DC bandpass rejects shadows.
ink=matrix('ink_light',ramp,np.array([[.60,.31,.09]])@curves,conv=True)
ink=norm('ink_adapt',ink);ink=clipped('ink_limit',ink,1,2.5)
# A broad local surround removes slow brightness variation inside the face.
s=np.arange(-4,5);g=np.exp(-.5*(s/2.45)**2);g/=g.sum();kernel=-.77*np.outer(g,g);kernel[4,4]+=1
layer('ink_band','conv',ink,out=1,kernel=[9,9],stride=1,padding=4);W['ink_band.W']=kernel.astype('f').reshape(1,1,9,9);W['ink_band.b']=np.zeros(1,'f')

def patches(n,src,h,offsets):
    rows=[];side=int(round(32));rr,cc=np.indices((h,h));lo=(side-h)//2
    for dx,dy in offsets:
        rows.extend(((rr+lo+dy)*side+cc+lo+dx).flatten())
    mat=np.zeros((len(rows),side*side),'f');mat[np.arange(len(rows)),rows]=1
    return shape(n+'_squares',matrix(n+'_extract',src,mat),len(offsets),h,h)

# Local patch contrast normalization is done at inference, not template-set
# normalization. Each contiguous square is its own channel for the norm op.
# The nine translations are kept as *poses*, shared between symbol and frame.
OFF=[(dx,dy) for dy in [-2,0,2] for dx in [-2,0,2]]
# Instead of a large dense matrix duplicating patches, match in one unshifted
# central window, including the translated versions in the bank's rows.
H=26
center=patches('reading_window','ink_band',H,[(0,0)])
center=norm('local_ink_contrast',center)
axis=np.arange(H)+.5-H/2;xx,yy=np.meshgrid(axis,axis)
ks=[k for k in range(43) if k not in [12,13,15,17]]
# Each variant is a discrete physical hypothesis, not an ensemble average.
# (horizontal scale, vertical scale, shear, roll, font style, vertical shift)
variants=[(1,1,0,0,0,0),(.90,.85,0,0,1,0),(1.11,1.10,0,0,1,0),
          (.94,1.10,-.08,-.05,2,.025),(1.07,.93,.08,.05,3,-.025),
          (1,1,-.05,.10,1,0),(1,1,.05,-.10,0,0),(.91,.96,0,0,2,-.055),
          (1.14,1.24,-.04,0,0,-.07),(.94,1.2,0,.07,1,-.11),(.78,.82,0,0,0,.07),
          (1,1.10,0,-.04,0,.12),(1.18,.98,0,.03,2,-.09),(1.06,1.04,0,0,0,-.15)]
Vn=len(variants)
# Generation is vectorized over the nine translations, never reduced over them.
xoffs=np.array([a for a,b in OFF])[:,None,None];yoffs=np.array([b for a,b in OFF])[:,None,None]
M=np.empty((len(ks),4,9,Vn,H*H),'f')
for ik,k in enumerate(ks):
    drawings=[blur(render(k,st),1.25) for st in range(4)]
    for ri,(r,_,__) in enumerate(poses[::9]):
        x=(xx-xoffs)/r;y=(yy-yoffs)/r;mask=ink_support(k,x,y)
        for vi,(sx,sy,sh,rot,st,oy) in enumerate(variants):
            tx=x+sh*y;ty=y-oy
            if family[k]==1:ty=ty-.22
            a=(math.cos(rot)*tx+math.sin(rot)*ty)/sx;b=(-math.sin(rot)*tx+math.cos(rot)*ty)/sy
            if family[k]==1:b=b+.22
            im=sample(drawings[st],a,b)
            # Weighted zero-mean correlation, confined to the readable field.
            filt=mask*(im-(mask*im).sum(axis=(1,2),keepdims=True)/mask.sum(axis=(1,2),keepdims=True))
            filt/=np.maximum(1e-8,np.sqrt((filt*filt).sum(axis=(1,2),keepdims=True)))
            # Compare evidence per unit of face diameter (not total ink mass).
            M[ik,ri,:,vi]=filt.reshape(9,-1)/(r*.87)
inkrows=matrix('symbol_correlations',center,M.reshape(-1,H*H));del M
s=shape('symbol_hypotheses',inkrows,len(ks),Q,Vn)
s=pool('best_drawing',s,1,Vn)
# Expand into the 43 class slots. Scores for blank faces and unlettered signs
# come from explicit face tests below rather than a non-existent ink filter.
expanded=np.zeros((43,len(ks)))
for j,k in enumerate(ks):expanded[k,j]=1.0
s=matrix('read_symbols',s,expanded,conv=True)

# A separate reading head handles small type. First use local contrast in the
# middle of the sign only (neither sky nor rim should drown out faint print).
# Search at pixel-spaced positions using a true shared convolution. Width,
# height, kerning and stroke weight vary independently of the frame pose.
wordH=23
wordinput=patches('type_window','ink_band',wordH,[(0,0)])
wordinput=norm('type_contrast',wordinput)
wordks=[0,1,2,3,4,5,7,8]
textlist=['20','30','50','60','70','80','100','120']
K=21
wx,wy=np.meshgrid(np.arange(K)-(K-1)/2,np.arange(K)-(K-1)/2)
wr=[]
for k,text in zip(wordks,textlist):
    drawings=[blur(-render(k,st),1.15) for st in range(4)]
    for ht in [7.5,9.25,11.0,12.75]:
      for stretch in [.83,1.,1.19]:
       for st,slant,turn,space in [(0,0,0,1.),(3,-.065,.055,.86),(2,.07,-.065,1.12),(1,0,0,.96)]:
        # Each complete word, rather than unrelated maxima at each digit.
        rx=(wx+wy*turn)/(ht*stretch);ry=(wy-wx*turn)/ht
        hh=.84 if len(text)==2 else .76
        rx=(rx-slant*ry)*hh;ry=ry*hh
        wds=[(.33 if st%2==0 else .27) if ch=='1' else (.43 if len(text)==2 else .365) for ch in text];gap=.065 if len(text)==2 else .047
        total=sum(wds)+gap*(len(text)-1);cent=[];pos=-total/2
        for wid in wds:cent.append(pos+wid*.5);pos+=wid+gap
        # Inverse map a typeset line with slightly changed character spacing.
        rx=rx-np.interp(rx,np.array(cent)*space,np.array(cent)*(space-1))
        pic=-sample(drawings[st],rx,ry)
        # Exclude the colored rim and anything outside the word, but do
        # include a light flank to distinguish a narrow 1 from a wide digit.
        window=under(abs(ry),hh*.64,.10)*under(abs(rx),total*.5+.075,.10)
        filt=contrast(pic,window)
        wr.append((filt/(ht*1.12)).astype('f'))
wordW=np.stack(wr)[:,None]
word=layer('typeset_words','conv',wordinput,out=len(wr),kernel=[K,K],stride=1,padding=3)
W[word+'.W']=wordW;W[word+'.b']=np.zeros(len(wr),'f')
# Max is separable: avoid materializing all 81 positions in a single pool.
word=pool('horizontal_word_location',word,1,9)
word=pool('word_location',word,9,1)
word=shape('fonts_by_word',word,len(wordks),1,len(wr)//len(wordks));word=pool('read_type',word,1,len(wr)//len(wordks))
# Centering is not across generated drawings. This per-photo comparison is
# between alternative word readings *at inference*. It only affects choices
# among speed classes, not the overall prior odds of a speed sign.
decode=np.zeros((43,len(wordks)))
for j,k in enumerate(wordks):
    decode[k,j]=1.6;decode[k]-=1.6/len(wordks)
word=matrix('distinctive_word_parts',word,decode,conv=True)
# Replicate the one readout at all candidate frame poses (1x1 conv on a
# transposed vector, then transpose semantics back with a fixed dense copy).
wmap=np.repeat(np.eye(43,dtype='f'),Q,axis=0)
word=shape('word_at_frame',matrix('share_word_reading',word,wmap),43,Q,1)

# In addition, compare the shapes in face-relative coordinates. Each norm
# channel below is one *input* patch: neither image exposure nor contrast in
# the background can determine the result of this part of the recognition.
# Outside the circular/triangular face the sample coordinates reflect back
# inside rather than sampling the red frame. This implements a boundary
# condition, not a prediction of the positions of any objects in a photo.
domain=19;ax=np.linspace(-1,1,domain);ux,uy=np.meshgrid(ax,ax)

def local_head(n,classes,idx,sizes,triangular):
    if triangular:
        gx=ux*.48;gy=.19+uy*.47
        # At the corners of the patch, reflect about the sloping sides.
        lim=np.maximum(.025,(gy+.48)/2)
        gx=np.sign(gx)*(lim-np.abs(lim-np.minimum(abs(gx),2*lim)))
    else:
        gx=.65*ux;gy=.48*uy
        rr=np.hypot(gx,gy);mag=np.minimum(1.,(.75-np.maximum(rr-.75,0))/np.maximum(rr,1e-6));gx*=mag;gy*=mag
    remap=np.zeros((Q,domain**2,32*32),'f');ij=np.arange(domain**2)
    for q,(r,dx,dy) in enumerate(poses):
        xx=np.clip((gx*r+15.5+dx).flatten(),0,30.999);yy=np.clip((gy*r+15.5+dy).flatten(),0,30.999)
        i=yy.astype(int);j=xx.astype(int);fy=yy-i;fx=xx-j
        for e in [0,1]:
         for d in [0,1]:
            remap[q,ij,(i+e)*32+j+d]=(fy if e else 1-fy)*(fx if d else 1-fx)
    # Positions on the boundary of the remap are shared by normalization
    # and the numerator. Each numerator is an independent unit-length filter.
    a=matrix(n+'_rectify',ink,remap.reshape(-1,1024));a=shape(n+'_images',a,Q,domain,domain);a=norm(n+'_normalize',a)
    a=shape(n+'_columns',a,1,Q,domain**2)
    # Transpose Q x pixels to pixels x Q using an equivalent conv: a dense
    # matrix per patch can be a square-domain convolution on the patches
    # tiled down a 1-channel image. Non-overlapping stride processes all rows.
    a=shape(n+'_tiles',a,1,Q*domain,domain)
    filt=[]
    for k in classes:
        pictures=[blur(render(k,st),1.6) for st in range(4)]
        for sx,sy,tilt,skew,st in sizes:
         for ox,oy in idx:
            y=gy-(.22 if triangular else 0)-oy;x=gx-ox-skew*y
            x,y=(x*np.cos(tilt)+y*np.sin(tilt))/sx,(y*np.cos(tilt)-x*np.sin(tilt))/sy
            pic=sample(pictures[st],x,y+(.22 if triangular else 0));pic=unit(pic-pic.mean())/domain
            filt.append(pic.astype('f'))
    filt=np.stack(filt)[:,None];a=layer(n+'_inner_products','conv',a,out=len(filt),kernel=[domain,domain],stride=domain,padding=0);W[a+'.W']=filt;W[a+'.b']=np.zeros(len(filt),'f')
    a=shape(n+'_variants',a,len(classes),len(idx)*len(sizes),Q)
    a=pool(n+'_correlation',a,len(idx)*len(sizes),1);a=shape(n+'_by_pose',a,len(classes),Q,1)
    m=np.zeros((43,len(classes)));bias=np.zeros(43)
    for j,k in enumerate(classes):
        # A low but positive correlation to noise can occur by chance.
        m[k,j]=2.6;bias[k]=-2.6*.38
    return matrix(n+'_evidence',a,m,bias,True)
shift=[(x,y) for x in [-.087,0,.087] for y in [-.095,0,.095]]
styles=[(1,1,0,0,0),(.84,.82,-.06,.04,1),(1.13,1.20,.065,-.04,0),(.90,1.1,.02,.05,2),(1.16,.88,-.03,-.07,3),(.99,1.,-.12,0,1),(.99,1.,.12,0,2)]
warn_head=local_head('warning_icon',[11,*range(18,32)],shift,styles,True)
reg_head=local_head('circular_legend',[*wordks,9,10,16,6,32,41,42,14,17,*range(33,41)],shift,styles,False)

# Face grammar. Evidence that a sign has a blank area, separate text or a
# horizontal ideogram, provided by signed contrast, darkness and gradients.
# These formula filters also give color-only hypotheses a score commensurate
# with a good symbol match without rewarding them for arbitrary dark objects.
gh=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/8
kern=np.array([gh,-gh,gh.T,-gh.T]);layer('ink_slopes','conv',ink,out=4,kernel=[3,3],stride=1,padding=1);W['ink_slopes.W']=kern.astype('f')[:,None];W['ink_slopes.b']=np.zeros(4,'f')
edge=layer('ink_slope_parts','relu','ink_slopes');edge=matrix('ink_variation',edge,[[1,1,1,1]],conv=True)
edge=pool('coarse_variation',edge,2,2,'avgpool');e=[];av=[]
for r,dx,dy in poses:
    x=(X-dx)/r;y=(Y-dy)/r;roi=small(under(np.hypot(x,y),.61,.09));roi/=roi.sum();e.append(roi.flatten());av.append(roi.flatten())
edge=shape('inner_variation',dot_rows('variation_by_pose',edge,e),1,Q,1)
# Colour and brightness of the inner face and of two vehicles (where present).
reg=[]
for mode in ['brightness','grayred','grayblue','leftcar']:
 for r,dx,dy in poses:
    x=(X-dx)/r;y=(Y-dy)/r;coeff=np.zeros((4,16,16));roi=small(under(np.hypot(x,y),.66,.1));roi/=roi.sum();
    if mode=='leftcar':
        m=small(under(abs(x+.34),.24,.08)*under(abs(y),.21,.08));m/=m.sum();coeff[1]=m-roi
    else:coeff[['brightness','grayred','grayblue'].index(mode)]=roi
    reg.append(coeff.flatten())
reg=shape('face_planes',dot_rows('face_regions',coarse,reg),4,Q,1)
border=shape('frame_planes',border,F,Q,1)
# Classes sharing a sign frame use exactly the same frame rules.
m=np.eye(F)[family];b=matrix('frame_to_class',border,m,conv=True)
em=np.zeros(43);bias=np.zeros(43);rm=np.zeros((43,4))
# Presence/absence of a dark legend on pale signs: generic semantic cues.
for k in range(43):
    f=family[k]
    if f in (0,1,3):rm[k,0]=.20
    if f==0:rm[k,1]=-.08
    if f==1:rm[k,1]=-.11
    if f==2:rm[k,2]=.16
    if f==3:rm[k,1]=-.14;rm[k,2]=-.10;bias[k]=-.06
    if k in (9,10):rm[k,3]=.32
    if k in (12,13,15,17):
        bias[k]=.73;em[k]=-1.65
    if k==14:bias[k]=.10
    if k==12:rm[k,0]=.08
    if k==17:em[k]=-.40;bias[k]=.32
    if k==13:em[k]=-.80;bias[k]=.58
    if k==15:rm[k,0]=.28
    if k==32:em[k]=-.23
    # The pale signs with no legend are the correlation of a constant field:
    # a smooth face gives maximum evidence, but texture quickly removes it.
    if k in [15,13]:bias[k]+=.78;em[k]-=3.6
edges=matrix('blank_face_rule',edge,em[:,None],bias,True)
regions=matrix('face_color_rules',reg,rm,conv=True)
# Check the rim in several separated arcs, not just as a global matched
# filter. A red circle has color or a luminance valley at every azimuth;
# shadows and diagonal cancellation bands generally do not.
arc=[];tri=[];neut=[]
for t in range(8):
 for r,dx,dy in poses:
    x=(X-dx)/r;y=(Y-dy)/r;rad=np.hypot(x,y);theta=t*np.pi/4
    sector=np.exp(4*(np.cos(theta)*x+np.sin(theta)*y)/np.maximum(.02,rad)-4)
    def patch(z):
        a=small(z);return a/max(1e-7,a.sum())
    band=patch(sector*under(abs(rad-.876),.084,.075));inside=patch(sector*under(abs(rad-.655),.065,.07))
    cc=np.zeros((4,16,16));cc[1]=.85*band-.68*inside;cc[0]=.80*(inside-band);arc.append(cc.flatten())
    z=cc.copy()*0;z[1]=band-inside;neut.append(z.flatten())
arc=shape('rim_arcs',dot_rows('arc_contrasts',coarse,arc),1,8,Q)
neg=matrix('incomplete_arcs',arc,[[-1]],.48,True);neg=layer('arc_deficit','relu',neg)
neg=pool('arc_deficit_mean',neg,8,1,'avgpool');neg=shape('rim_deficit',neg,1,Q,1)
m=np.zeros((43,1))
for k,f in enumerate(family):
    if f==0:m[k]=-.91
    if f==3:m[k]=.53
circle=matrix('consistent_rim',neg,m,conv=True)
# Consistency of horizontal: a no-entry bar covers most of the red disc's
# width. STOP letters are vertical strokes separated by negative space.
# Other iconic colors: stop lights have red above yellow above green, even
# when their luminances vary unpredictably.
layout=[]
for k in [26,4,14]:
 for r,dx,dy in poses:
    x=(X-dx)/r;y=(Y-dy)/r;cc=np.zeros((4,16,16))
    if k==26:
        def spot(a):
            z=small(np.exp(-.5*((x/.14)**2+((y-a)/.14)**2)));return z/z.sum()
        cc[1]=.48*(spot(-.09)-spot(.49));cc[3]=.16*(spot(.2)-.5*(spot(-.09)+spot(.49)))
    elif k==4:
        # A triangular yield sign has a red horizontal cross-edge at the
        # TOP, and two converging straight edges rather than a lower rim.
        s1=patch(under(abs(y+.875),.08,.07)*under(abs(x),.68,.08));s2=patch(under(abs(y+.62),.065,.08)*under(abs(x),.60,.08))
        cc[0]=.30*(s2-s1);cc[1]=.32*(s1-s2)
    else:
        # A solid, bright no-entry bar has a dark face on BOTH sides. In
        # a word like STOP the white strokes extend above/below the midline.
        mid=patch(under(abs(y),.095,.07)*under(abs(x),.72,.08));flank=patch(under(abs(abs(y)-.23),.065,.07)*under(abs(x),.7,.08))
        cc[0]=.62*(mid-flank);cc[1]=.35*(flank-mid)
    layout.append(cc.flatten())
layout=shape('color_geometry',dot_rows('light_and_yield_edges',coarse,layout),3,Q,1)
# The traffic light cue is supportive, not sufficient: saturate its
# contribution rather than allowing a bright bit of border to vote unboundedly.
t=matrix('layout_bounds',layout,np.r_[np.eye(3),np.eye(3)],np.r_[[.32,1.2,1.5],[-.32,-1.2,-1.5]],True)
t=layer('layout_ramps','relu',t);layout=matrix('bounded_layout',t,np.c_[np.eye(3),-np.eye(3)],[-.32,-1.2,-1.5],True)
assign=np.zeros((43,3));assign[26,0]=assign[13,1]=assign[17,2]=1;assign[14,2]=-.24
layout=matrix('place_color_geometry',layout,assign,conv=True)
# A cancellation stripe traverses the whole disk and survives even when
# the lettering it crosses is lost. Require the diagonal to continue OUTSIDE
# the word, comparing it to both flanks and combining the two ends.
stripes=[]
for end in [-1,1]:
 for r,dx,dy in poses:
    x=(X-dx)/r;y=(Y-dy)/r;u=(x+y)/np.sqrt(2);v=end*(x-y)/np.sqrt(2)
    z=under(abs(v-.67),.20,.09)
    dark=patch(z*under(abs(u),.155,.10));light=patch(z*under(abs(abs(u)-.33),.12,.10))
    cc=np.zeros((4,16,16));cc[0]=light-dark;stripes.append(cc.flatten())
diag=shape('outer_band_ends',dot_rows('diagonal_cancellation',coarse,stripes),2,Q,1)
# min(a,b) = a - relu(a-b).
aux=matrix('stripe_min_parts',diag,[[1,0],[-1,0],[1,-1]],conv=True);aux=layer('stripe_rect','relu',aux)
diag=matrix('continuous_stripe',aux,[[1,-1,-1]],conv=True)
cm=np.zeros((43,1));cm[[6,32,41,42],0]=1.5
band=matrix('cancellation_evidence',diag,cm,conv=True)
# Check the three sides of the warning triangle separately. Use brightness
# and red/neutral contrast and bound each side's vote: a strong horizontal
# stroke without two sloping strokes is not a triangle.
trirows=[]
for side in range(3):
 for r,dx,dy in poses:
    x=(X-dx)/r;y=(Y-dy)/r
    if side==2:
        h=y;z=under(abs(x),.67,.11);d1=patch(z*under(abs(h-.89),.086,.08));d2=patch(z*under(abs(h-.67),.075,.08))
    else:
        h=2*x*(1 if side==1 else -1)-y;z=under(abs(y-.09),.58,.18)
        d1=patch(z*under(abs(h-.77),.19,.15));d2=patch(z*under(abs(h-.30),.13,.12))
    cc=np.zeros((4,16,16));cc[0]=.80*(d2-d1);cc[1]=.60*(d1-d2);trirows.append(cc.flatten())
tri=shape('triangle_sides',dot_rows('triangle_side_contrasts',coarse,trirows),1,3,Q)
tri=clipped('bounded_side_votes',tri,1,.98)
tri=pool('three_sides',tri,3,1,'avgpool');tri=shape('triangle_by_pose',tri,1,Q,1)
m=np.zeros((43,1));bb=np.zeros(43)
for k,f in enumerate(family):
    if f==1:m[k,0]=1.1;bb[k]=-.6
tritest=matrix('closed_triangle',tri,m,bb,True)

# One more independent view keeps both the outline and the entire face in a
# single normalized coordinate system. This tests whether an apparent icon
# could instead just be a piece of a differently shaped rim. For triangles
# use a star-shaped grid: near a corner we sample closer to the centroid.
# (Custom dense interpolation here is simply four fixed bilinear taps.)
def entire(n,classes,triangle_type=False):
    h=17;ax=np.linspace(-1.01,1.01,h);x,y=np.meshgrid(ax,ax)
    if triangle_type:
        radial=np.maximum(1e-10,np.maximum(abs(x),abs(y)));X0=x/radial;Y0=y/radial
        radius=1/np.maximum.reduce([Y0/.67,(2*abs(X0)-Y0)/1.33,np.zeros_like(x)+1.e-9]);gx=x*radius;gy=.33+y*radius
        if triangle_type=='yield':gy=-gy
    else:
        rho=np.hypot(x,y);g=np.maximum(abs(x),abs(y))/np.maximum(rho,1e-9);gx=x*g;gy=y*g
    mat=np.zeros((Q,h*h,1024),'f');ii=np.arange(h*h)
    for q,(r,dx,dy) in enumerate(poses):
        xx=np.clip((gx*r+15.5+dx).ravel(),0,30.999);yy=np.clip((gy*r+15.5+dy).ravel(),0,30.999)
        i=yy.astype(int);j=xx.astype(int);fx=xx-j;fy=yy-i
        for a in [0,1]:
         for b in [0,1]:mat[q,ii,(i+a)*32+j+b]=(fx if b else 1-fx)*(fy if a else 1-fy)
    a=matrix(n+'_extract',shade,mat.reshape(-1,1024));a=shape(n+'_faces',a,Q,h,h);a=norm(n+'_contrast',a);a=shape(n+'_tiles',a,1,Q*h,h)
    hist=[];opts=[(.99,.99,0,0,0,0),(.90,1.10,0,-.075,0,1),(1.10,.88,-.09,.04,.03,3),(1.06,1.12,.075,.04,-.03,2),(.93,.91,.09,0,.03,1),
        (.98,1.02,-.09,-.06,-.03,1),(.85,.84,0,.055,0,0),(1.2,1.22,0,-.05,0,0)]
    for k in classes:
        f=family[k];pics=[blur(render(k,st),1.6) for st in range(4)]
        for sx,sy,t,offy,offx,st in opts:
            xx=gx-offx;yy=gy-(.22 if f==1 else 0)-offy
            tx=(xx*np.cos(t)+yy*np.sin(t))/sx;ty=(-xx*np.sin(t)+yy*np.cos(t))/sy+(.22 if f==1 else 0)
            mark=sample(pics[st],tx,ty)
            if f in (0,1,4):
                o,i,__,__=rim(f,gx,gy,.15);pic=(.91+.71*mark)*i+.26*(o-i)+.43*(1-o)
            elif f==2:
                o=under(np.hypot(gx,gy),.98,.1);pic=.22+.70*mark*o
            elif f==3:pic=.91+.74*mark
            elif f==6:pic=.28+.67*mark
            elif f==5:
                o,i,c,_=rim(f,gx,gy,.14);pic=.42*(1-o)+i
            elif f==7:
                o,i,c,_=rim(f,gx,gy,.14);pic=i+.4*(1-o)
            hist.append(unit(pic-pic.mean())/h)
    weights=np.stack(hist).astype('f')[:,None];a=layer(n+'_matched', 'conv',a,out=len(weights),kernel=[h,h],stride=h,padding=0);W[a+'.W']=weights;W[a+'.b']=np.zeros(len(weights),'f')
    a=shape(n+'_styles',a,len(classes),len(opts),Q);a=pool(n+'_read',a,len(opts),1);a=shape(n+'_by_pose',a,len(classes),Q,1)
    mat=np.zeros((43,len(classes)));offset=np.zeros(43)
    for j,k in enumerate(classes):mat[k,j]=3.2;offset[k]=-.84
    val=matrix(n+'_score',a,mat,offset,True)
    if triangle_type=='yield':
        # A yield face without any imprint is itself a positive, complete
        # hypothesis, not missing symbol evidence.
        copy=np.zeros_like(mat);copy[13,0]=1.75
        val=add('yield_empty_field',val,matrix('plain_yield_match',a,copy,conv=True))
    return val
shade=matrix('full_sign_gray',ramp, np.array([[.22,.60,.18]])@curves,conv=True)
roundwhole=entire('whole_disc',[k for k in range(43) if family[k] not in (1,4)])
warnwhole=entire('whole_triangle',[k for k in range(43) if family[k]==1],True)
yieldwhole=entire('whole_yield',[13],'yield')
# Bright and dark marks should also agree in color with the *local* ground
# and surround. The background beyond the sign is unknown; ignore it here.
# Instead of measuring the whole image at full resolution, compact the four
# planes to 16x16 and compare ordered adjacent regions. Zero-mean opponent
# patterns make this cue insensitive to the color of the illuminant.
# Use several aspect/roll hypotheses; each has its own filter normalization.
bank=[]; borderposes=[(.99,.99,0),(1.09,.90,0),(.90,1.10,0),(.99,.99,.12),(.99,.99,-.12)]
for f in range(F):
 for r,dx,dy in poses:
  for sx,sy,t in borderposes:
    x0=(X-dx)/r;y0=(Y-dy)/r;x=(x0*np.cos(t)+y0*np.sin(t))/sx;y=(y0*np.cos(t)-x0*np.sin(t))/sy
    rho=np.hypot(x,y);cc=np.zeros((4,16,16));o,i,c,mask=rim(f,x,y,.11)
    if f in [0,1,4]:
        outer=patch(c);inside=patch(i*(.35+.65*(rho>.42)));d=outer-inside
        cc[1]=1.05*d;cc[0]=-.86*d
    elif f==2:
        d=patch(o);cc[2]=1.16*d;cc[1]=-.35*d;cc[0]=-.45*d
    elif f==3:
        # Neutral rimless disk. Compare its outer band to the white field:
        # no red ring irrespective of absolute brightness of the gray disk.
        m=patch(o-under(rho,.79,.1));core=patch(i);cc[1]=-.66*(m-core);cc[0]=.46*m
    elif f==5:
        # Yellow center, then white diamond, frequently with a black keyline.
        yel=patch(c);white=patch(o-c);d=yel-white
        cc[3]=1.15*d+.45*yel;cc[0]=-.40*d
    else:
        cheeks=patch(o*under(abs(abs(y)-.42),.18,.12)*under(abs(x),.58,.1))
        cc[1]=.89*cheeks;cc[0]=-.36*cheeks
        if f==7:
            bar=patch(under(abs(y),.11,.08)*under(abs(x),.76,.1))
            cc[0]+=.4*(bar-cheeks);cc[1]+=.25*(cheeks-bar)
    bank.append(cc.flatten())
extra=shape('relative_color_fields',dot_rows('relative_border_color',coarse,bank),F,Q,len(borderposes));extra=pool('border_deformation',extra,1,len(borderposes))
# For inhibition between sign families, also consider whether a competing
# frame has a better pose than ours. This compares scores only at inference.
base=add('frame_and_field',border,extra);best=pool('any_frame_location',base,Q,1)
dup=np.tile(np.eye(F,dtype='f')[:,None,:],(1,Q,1)).reshape(F*Q,F)
best=shape('frame_strength',matrix('share_frame',best,dup),F,Q,1)
summary=add('frame_global_local',base,best);summary=matrix('framing_evidence',summary,.68*np.eye(F)[family],conv=True)

# Recognize the positive evidence of a speed legend even when its red paint
# is gray in shadow. Correlate the entire word *and the blank space beside
# it* on a pose-normalized face: this stops a 20 read from the right of 120
# competing on equal terms with a complete reading of all three numerals.
# Require the word to explain the horizontal distribution of ink as well as
# the detailed strokes. Integration by columns loses the vertical phase and
# tolerates a tilted baseline, blur and coarse original sampling.
profile=[];pstyles=[(.90,.9,-.045),(.96,1.12,0),(1.,1.,0),(1.10,1.,0),(1.14,.90,.045),(.82,1.10,-.02),(1.04,1.15,.018)]
for k in wordks:
    pics=[blur(render(k,st),1.4) for st in range(4)]
    for sx,sy,off in pstyles:
     for xoff in [-.065,0,.065]:
        gx=.65*ux;gy=.48*uy;rr=np.hypot(gx,gy);mag=np.minimum(1.,(.75-np.maximum(rr-.75,0))/np.maximum(rr,1e-6));gx*=mag;gy*=mag
        a=sample(pics[1],(gx-xoff)/sx,(gy-off)/sy)
        # Each individual template is integrated and normalized on its own.
        a=np.tile(a.mean(0),(domain,1));profile.append(unit(a-a.mean())/domain)
wp=np.stack(profile).astype('f')[:,None]
pr=layer('ink_column_reader','conv','circular_legend_tiles',out=len(wp),kernel=[domain,domain],stride=domain,padding=0);W[pr+'.W']=wp;W[pr+'.b']=np.zeros(len(wp),'f')
pr=shape('ink_column_widths',pr,len(wordks),len(pstyles)*3,Q);pr=pool('glyph_spacing',pr,len(pstyles)*3,1);pr=shape('word_spacing_by_pose',pr,len(wordks),Q,1)
pm=np.zeros((43,len(wordks)));pb=np.zeros(43)
for j,k in enumerate(wordks):pm[k,j]=1.7;pb[k]=-.38
pr=matrix('word_spacing_score',pr,pm,pb,True)
# Is the silhouette blank or does it have a digit? Compare the two high
# horizontal-frequency parts to the whole interior on a per-image basis.
# (The regional blank-face rule already supplies this test for unlettered
# ring and yield images.) For the priority diamond the most invariant feature
# is a warm inner lozenge enclosed by a much less saturated border. Use local
# color contrast, not global yellow saturation or a sky-dependent luminance.
coords=np.linspace(-1.02,1.02,17);gx,gy=np.meshgrid(coords,coords);gain=np.maximum(abs(gx),abs(gy))/np.maximum(1e-8,abs(gx)+abs(gy));gx*=gain;gy*=gain
M0=np.zeros((Q,17*17,16*16),'f');ii=np.arange(17*17)
for q,(r,dx,dy) in enumerate(poses):
    xx=np.clip(((gx*r+16+dx)/2-.5).ravel(),0,14.9999);yy=np.clip(((gy*r+16+dy)/2-.5).ravel(),0,14.9999);i=yy.astype(int);j=xx.astype(int);fy=yy-i;fx=xx-j
    for a in [0,1]:
     for b0 in [0,1]:M0[q,ii,(i+a)*16+j+b0]=(fx if b0 else 1-fx)*(fy if a else 1-fy)
yc=matrix('yellow_field',coarse,[[0,0,0,1]],conv=True);yc=matrix('sample_lozenge',yc,M0.reshape(-1,256));yc=shape('diamond_fields',yc,Q,17,17);yc=norm('diamond_chroma',yc);yc=shape('diamond_tiles',yc,1,Q*17,17)
filters=[]
for r0 in [.63,.73,.82]:
  for tilt in [-.09,0,.09]:
    lo=abs(gx*np.cos(tilt)+gy*np.sin(tilt))+abs(gy*np.cos(tilt)-gx*np.sin(tilt));im=under(lo,r0,.13);filters.append(unit(im-im.mean())/17)
yc=layer('white_boundaries','conv',yc,out=len(filters),kernel=[17,17],stride=17,padding=0);W[yc+'.W']=np.array(filters,'f')[:,None];W[yc+'.b']=np.zeros(len(filters),'f')
yc=shape('diamond_models',yc,1,len(filters),Q);yc=pool('white_keyline',yc,len(filters),1);yc=shape('lozenge_by_pose',yc,1,Q,1);assign=np.zeros((43,1));assign[12]=3.;bb=np.zeros(43);bb[12]=-.54;yc=matrix('priority_pattern',yc,assign,bb,True)
combined=add('score_by_pose',s,b,edges,regions,word,circle,layout,warn_head,reg_head,band,tritest,roundwhole,warnwhole,yieldwhole,summary,pr,yc)
combined=pool('most_consistent_pose',combined,Q,1)
shape('scores',combined,43)
(P/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
np.savez(P/'weights.npz',**W)
