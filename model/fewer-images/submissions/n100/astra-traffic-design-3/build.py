"""Data-free German sign recognizer. Draw signs in ideal sign coordinates, then
write banks of individual masked correlations and explicit pose pooling.
All paths below are drawings, not data samples. No font or graphics assets.
"""
import sys, json, math
from pathlib import Path
import numpy as np

F=np.float32
# Drawing plane: -1 .. 1 across the face of the sign. All drawing tools
# accept coordinate arrays, so that the same paths work at any sign size.
class Pen:
    def __init__(self,x,y):
        self.x=x; self.y=y; self.a=np.zeros(np.broadcast_shapes(x.shape,y.shape),F)
    def oval(self,x,y,rx,ry=None,col=1.):
        if ry is None: ry=rx
        self.a[((self.x-x)/rx)**2+((self.y-y)/ry)**2<=1]=col
    def line(self,pts,w,col=1.):
        for (x,y),(u,v) in zip(pts,pts[1:]):
            dx=u-x;dy=v-y
            t=np.clip(((self.x-x)*dx+(self.y-y)*dy)/(dx*dx+dy*dy+1.e-15),0,1)
            self.a[(self.x-x-t*dx)**2+(self.y-y-t*dy)**2<=w*w*.25]=col
    def poly(self,pts,col=1.):
        m=np.zeros_like(self.a,dtype=bool)
        for (x,y),(u,v) in zip(pts,pts[1:]+pts[:1]):
            if v!=y: m^=((self.y>=min(y,v))&(self.y<max(y,v))&(self.x < x+(self.y-y)*((u-x)/(v-y))))
        self.a[m]=col
    def curve(self,pts,w,col=1.):
        # Piecewise cubic Beziers encoded as p0, c1, c2, p1, c1, c2, p2, ...
        out=[pts[0]]
        for j in range(0,len(pts)-1,3):
            a,b,c,d=np.array(pts[j:j+4]);
            for t in np.arange(1,11)/10:
                out.append(tuple((1-t)**3*a+3*t*(1-t)**2*b+3*t*t*(1-t)*c+t**3*d))
        self.line(out,w,col)
    def box(self,x1,y1,x2,y2,col=1.):
        self.a[(self.x>=x1)&(self.x<=x2)&(self.y>=y1)&(self.y<=y2)]=col

def digit(p,ch,th,style=0):
    # Stroke centres in unit digit box; widths supplied by drawing coordinates.
    # Use curves rather than a seven-segment approximation (DIN road lettering).
    if ch=='0':
        p.curve([(.5,.07),(.78,.07),(.86,.19),(.86,.35),(.86,.47),(.86,.57),(.86,.67),(.86,.85),(.73,.93),(.5,.93),(.24,.93),(.14,.82),(.14,.65),(.14,.53),(.14,.46),(.14,.34),(.14,.16),(.28,.07),(.5,.07)],th)
    elif ch=='1':
        p.line([(.18,.25),(.55,.08),(.55,.93)],th)
    elif ch=='2':
        p.curve([(.14,.26),(.18,.03),(.76,-.01),(.83,.23),(.95,.43),(.51,.59),(.14,.9)],th);p.line([(.14,.92),(.86,.92)],th)
    elif ch=='3':
        p.curve([(.13,.19),(.36,-.05),(.93,.05),(.83,.29),(.82,.42),(.69,.48),(.42,.49),(.67,.46),(.9,.56),(.86,.73),(.88,.95),(.37,1.02),(.12,.82)],th)
    elif ch=='5':
        p.line([(.82,.08),(.22,.08),(.16,.48)],th)
        p.curve([(.16,.48),(.36,.34),(.86,.36),(.86,.59),(.93,.83),(.8,.95),(.46,.93),(.29,.93),(.18,.88),(.12,.81)],th)
    elif ch=='6':
        if style%2==0:p.line([(.70,.06),(.28,.45),(.18,.60)],th)
        else:p.curve([(.78,.13),(.35,-.06),(.13,.23),(.13,.61)],th)
        p.curve([(.16,.65),(.17,.39),(.76,.34),(.84,.58),(.96,.81),(.76,.94),(.48,.93),(.28,.93),(.12,.84),(.16,.65)],th)
    elif ch=='7':
        p.line([(.12,.08),(.87,.08),(.34,.93)],th)
    elif ch=='8':
        for cy,rx,ry in [(.275,.32,.208),(.705,.37,.232)]:
            t=np.linspace(0,np.pi*2,33);p.line(list(zip(.50+rx*np.cos(t),cy+ry*np.sin(t))),th)

# Generate square-coordinate strokes despite non-square digit boxes: draw the
# centreline in local digit coords, then stretch the pen itself vertically less
# than the digit aspect ratio. This preserves the even stroke weight of signage.
class LetterPen:
    def __init__(self,p,l,t,w,h):self.p=p;self.l=l;self.t=t;self.w=w;self.h=h
    def coords(self,pts):return [(self.l+x*self.w,self.t+y*self.h) for x,y in pts]
    def line(self,pts,w):self.p.line(self.coords(pts),w*self.h)
    def curve(self,pts,w):self.p.curve(self.coords(pts),w*self.h)

speeds={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}
def number(p,text,v=0):
    n=len(text);h=1.;w=(.56 if v%3!=1 else .64) if n==2 else (.49 if v%3!=1 else .54)
    widths=[w*(.64 if ch=='1' else 1.) for ch in text];gap=.11 if n==2 else .068
    x=-(sum(widths)+(n-1)*gap)/2
    for ch,ww in zip(text,widths):
        digit(LetterPen(p,x,-h/2,ww,h),ch,(.123 if v%3!=2 else .153),v);x+=ww+gap

def car(p,cx,cy,w,h,col):
    def pts(z):return [(cx+x*w,cy+y*h) for x,y in z]
    p.poly(pts([(-.49,.31),(-.49,-.14),(-.36,-.25),(-.28,-.48),(.28,-.48),(.38,-.22),(.49,-.12),(.49,.31),(.38,.31),(.38,.47),(.21,.47),(.20,.32),(-.22,.32),(-.22,.47),(-.39,.47),(-.39,.31)]),col)
    p.poly(pts([(-.28,-.20),(-.19,-.39),(.19,-.39),(.28,-.20)]),0.)
def truck(p,cx,cy,w,h,col):
    def pts(z):return [(cx+x*w,cy+y*h) for x,y in z]
    p.poly(pts([(-.50,.29),(-.50,-.05),(-.35,-.32),(-.19,-.32),(-.19,-.47),(.5,-.47),(.5,.23),(.36,.23),(.3,.31),(-.18,.31),(-.2,.36)]),col)
    p.poly(pts([(-.43,-.04),(-.32,-.24),(-.24,-.24),(-.24,-.04)]),0.)
    for x in [-.31,.27]:p.oval(cx+x*w,cy+.28*h,.17*h,.18*h,col)

def arrow(p,kind,v):
    # White on blue. Use the sign's local vertical as arrow direction.
    if kind in (33,34):
        if v%3==2:
            p.poly([(-.70,-.12),(.14,-.12),(.14,-.44),(.72,0),(.14,.44),(.14,.13),(-.7,.13)])
        else:
            p.poly([(-.39,.68),(-.39,.18),(-.36,-.04),(-.28,-.20),(-.12,-.31),(.04,-.35),(.15,-.35),(.15,-.65),(.72,-.19),(.15,.28),(.15,-.04),(.03,-.04),(-.08,.02),(-.12,.14),(-.12,.68)])
    elif kind==35:
        p.poly([(-.13,.72),(.13,.72),(.13,-.10),(.41,-.10),(0,-.74),(-.41,-.10),(-.13,-.10)])
    elif kind in (36,37):
        p.poly([(-.34,.69),(-.08,.69),(-.08,-.27),(.10,-.27),(-.21,-.73),(-.54,-.27),(-.34,-.27)])
        p.poly([(-.27,.43),(-.13,.05),(0,-.12),(.16,-.17),(.29,-.17),(.29,-.38),(.70,-.02),(.29,.34),(.29,.12),(.12,.12),(.01,.21),(-.07,.43)])
    elif kind in (38,39):
        p.poly([(-.62,-.43),(-.43,-.62),(.31,.10),(.40,-.20),(.59,.59),(-.20,.40),(.10,.31)]) if v%3==1 else p.poly([(-.64,-.46),(-.46,-.64),(.28,.10),(.52,-.17),(.53,.53),(-.17,.52),(.10,.28)])
    else:
        for a in [0,2*np.pi/3,4*np.pi/3]:
            t=np.linspace(a-.36,a+.72,12);p.line(list(zip(.46*np.cos(t),-.46*np.sin(t))),.16)
            def rot(x,y):return (x*np.cos(a)-y*np.sin(a),-x*np.sin(a)-y*np.cos(a))
            p.poly([rot(.55,.12),rot(.49,.58),rot(.06,.54)])

def warning(p,k,v):
    # Drawings in a common icon box, about as high as wide, centred at zero.
    if k==11:
        if v%2==0:
            p.poly([(-.14,.66),(.14,.66),(.14,.10),(.48,.34),(.48,.17),(.14,-.10),(.14,-.25),(.25,-.25),(0,-.76),(-.25,-.25),(-.14,-.25),(-.14,-.10),(-.48,.17),(-.48,.34),(-.14,.10)])
        else:p.poly([(-.13,.66),(.13,.66),(.13,.17),(.48,.17),(.48,.01),(.13,.01),(.13,-.26),(.25,-.26),(0,-.76),(-.25,-.26),(-.13,-.26),(-.13,.01),(-.48,.01),(-.48,.17),(-.13,.17)])
    elif k==18:
        p.poly([(-.10,-.68),(.10,-.68),(.075,.22),(-.075,.22)]);p.oval(0,.48,.105,.12)
    elif k in (19,20):
        if v%2==0:
            p.poly([(.12,.60),(.42,.60),(.42,.06),(.39,-.18),(.30,-.36),(.12,-.47),(-.14,-.50),(-.39,-.50),(-.39,-.22),(-.15,-.22),(.01,-.19),(.09,-.10),(.12,.06)])
        else:p.poly([(.13,.58),(.41,.58),(.41,.23),(.38,.02),(.32,-.15),(.19,-.32),(-.08,-.52),(-.30,-.62),(-.46,-.40),(-.22,-.26),(-.02,-.10),(.10,.09),(.13,.28)])
    elif k==21:
        p.poly([(-.36,.60),(-.08,.60),(-.08,.10),(.32,.10),(.39,.02),(.39,-.32),(.17,-.61),(-.05,-.46),(.12,-.22),(.12,-.15),(-.26,-.15),(-.36,-.05)])
    elif k==22:
        p.poly([(-.85,.55),(-.77,.43),(-.68,.36),(-.57,.34),(-.45,.36),(-.34,.36),(-.14,.20),(0,.15),(.18,.16),(.31,.26),(.41,.36),(.55,.34),(.70,.38),(.85,.55)])
    elif k==23:
        if v%3!=0:
            # Rear view of a skidding car, tipped off horizontal. The two
            # wandering tracks extend below the body.
            a=(-.21 if v%3==1 else .19);cs=np.cos(a);sn=np.sin(a)
            q=Pen(p.x*cs+(p.y+.17)*sn,(p.y+.17)*cs-p.x*sn);car(q,0,0,.93,.61,1.)
            p.a=q.a
            for dx in [-.33,.29]:
                p.curve([(dx,.10),(dx+.35,.23),(dx-.35,.33),(dx-.18,.46),(dx-.09,.54),(dx+.16,.56),(dx+.10,.64)],.095)
            return
        p.poly([(-.68,-.03),(-.60,-.28),(-.35,-.34),(-.15,-.55),(.12,-.54),(.40,-.40),(.54,-.13),(.64,-.06),(.58,.14),(.44,.13),(.37,.04),(-.36,-.02),(-.45,.04)])
        p.poly([(-.23,-.32),(-.10,-.47),(.10,-.46),(.29,-.30)],0.)
        for dx in [-.35,.30]:
            p.curve([(dx,.08),(dx+.36,.22),(dx-.34,.33),(dx-.15,.48),(dx-.06,.54),(dx+.18,.56),(dx+.12,.64)],.083)
    elif k==24:
        sep=[.30,.23,.36][v%3];w=.16
        p.line([(-sep,-.57),(-sep,.61)],w);p.line([(sep*.20,-.57),(sep*.20,-.30),(sep,.06),(sep,.61)],w)
    elif k==25:
        p.poly([(.02,.59),(.10,.45),(.16,.44),(.23,.27),(.32,.27),(.44,.12),(.60,.23),(.70,.39),(.79,.49),(.89,.59)])
        p.oval(-.16,-.49,.13);p.line([(-.28,-.33),(-.47,-.11),(-.52,.06)],.17)
        p.line([(-.52,.04),(-.34,.18),(-.24,.52)],.12);p.line([(-.50,.04),(-.55,.25),(-.74,.53)],.12)
        p.line([(-.29,-.28),(-.18,-.04),(.01,.04)],.11);p.line([(-.28,-.29),(-.37,.01),(-.15,.06)],.09)
        p.line([(-.54,-.12),(.36,.41)],.045)
    elif k==26:
        if v%2:p.box(-.16,-.66,.16,.62,.73)
        for y,c in [(-.44,.81),(-.02,.33),(.40,.65)]:p.oval(0,y,.195,.196,c)
    elif k==27:
        p.oval(.10,-.52,.13);p.line([(.04,-.31),(-.08,.05)],.19)
        p.line([(-.07,.02),(.04,.24),(.31,.55)],.13);p.line([(-.06,.04),(-.15,.29),(-.36,.56)],.12)
        p.line([(.03,-.29),(.17,-.08),(.34,.01)],.085);p.line([(0,-.29),(-.21,-.19),(-.32,.01)],.09)
        if v%2:
            for x in [-.57,-.24,.11,.46]:p.poly([(x,.49),(x+.14,.49),(x+.08,.62),(x-.16,.62)])
    elif k==28:
        for cx,cy,s in [(.16,-.13,.89),(-.33,.12,.60)]:
            def z(x,y):return (cx+x*s,cy+y*s)
            p.oval(*z(.11,-.48),s*.16);p.poly([z(-.07,-.32),z(.17,-.31),z(.17,-.07),z(.31,.22),z(-.16,.22),z(-.09,-.07)])
            for path,w in [([(.02,.12),(-.12,.39),(-.32,.58)],.12),([(.11,.13),(.27,.34),(.35,.56)],.12),([(.08,-.22),(.28,-.01),(.43,-.06)],.09),([(-.01,-.24),(-.21,-.04),(-.35,.01)],.09)]:p.line([z(*q) for q in path],w*s)
    elif k==29:
        if v%3==1:
            # The bike-only version of the cycle warning: two wheels, a
            # triangular frame, seat post, and raised handlebars.
            for x in [-.44,.44]:
                p.oval(x,.28,.30);p.oval(x,.28,.215,col=0.)
            p.line([(-.44,.28),(-.24,-.10),(.015,.28),(-.44,.28)],.07);p.line([(.015,.28),(.28,-.17),(-.24,-.10)],.07)
            p.line([(.44,.28),(.23,-.34),(.38,-.37)],.072);p.line([(-.20,-.02),(-.27,-.25)],.065);p.line([(-.37,-.25),(-.17,-.25)],.075)
            return
        for x in [-.46,.44]:
            p.oval(x,.29,.28);p.oval(x,.29,.19,col=0.)
        p.line([(-.46,.29),(-.25,-.04),(.01,.29),(-.46,.29)],.063);p.line([(.01,.29),(.26,-.11),(.44,.29)],.07)
        p.oval(.05,-.48,.135);p.line([(-.06,-.32),(-.27,-.08),(.02,.02),(-.08,.25)],.13);p.line([(-.04,-.29),(.12,-.12),(.30,-.11)],.10)
    elif k==30:
        for a in np.arange(6)*np.pi/3:
            cs,sn=np.cos(a),np.sin(a)
            def rot(x,y):return (x*cs-y*sn,x*sn+y*cs)
            p.line([(0,0),rot(0,.65)],.09)
            for y in [.27,.44]:p.line([rot(-.16,y+.09),rot(0,y-.04),rot(.16,y+.09)],.065)
    else:
        # Leaping stag, head and antlers up, hind legs trailing.
        p.poly([(-.47,-.18),(-.35,-.13),(-.15,-.08),(.13,-.10),(.28,-.21),(.38,-.13),(.35,.06),(.23,.18),(-.08,.12),(-.33,.05),(-.44,-.04)])
        p.poly([(-.43,-.04),(-.53,-.24),(-.68,-.26),(-.72,-.34),(-.56,-.40),(-.41,-.40),(-.32,-.20)])
        p.line([(-.42,-.33),(-.37,-.51),(-.19,-.67)],.05);p.line([(-.4,-.44),(-.51,-.62)],.045);p.line([(-.32,-.54),(-.32,-.68)],.04)
        p.line([(.23,.07),(.37,.27),(.59,.38),(.72,.58)],.095);p.line([(.28,.04),(.46,.17),(.72,.24)],.065);p.line([(-.33,-.02),(-.38,.19),(-.17,.32),(-.14,.50)],.075);p.line([(-.27,.01),(-.24,.14),(-.04,.19),(.04,.33)],.06)

# Rendering: one smooth high-resolution sign per symbol/lettering variant.
# The result is sampled per hypothesis; each individual filter will be
# normalised only against its own face mask / its own contrast energy.
M=161
g=(np.arange(M,dtype=F)-(M-1)/2)/((M-1)/2)*1.20
Y,X=np.meshgrid(g,g,indexing='ij')
rad=np.sqrt(X*X+Y*Y)
# Upright triangle with a nearly equilateral profile after square crop resizing.
def triangle(x,y):return np.maximum(y, (2*np.abs(x)-y-1)*2+1)
# More directly, scale from centroid: vertices (0,-1), (+/-1,1).
def tri(x,y):return np.maximum((y-1/3)/(2/3),1.5*np.abs(x)-.75*(y-1/3))
triangles={11,18,*range(19,32)}
blue=set(range(33,41)); cancelled={6,32,41,42}
def render(k,v):
    x,y=X,Y
    white=np.ones_like(x); zero=np.zeros_like(x)
    if k in triangles or k==13:d=tri(x, -y if k==13 else y);edge=.735
    elif k==12:d=np.abs(x)+np.abs(y);edge=.68
    elif k==14:d=np.maximum(np.maximum(np.abs(x),np.abs(y)),(np.abs(x)+np.abs(y))/np.sqrt(2));edge=.91
    else:d=rad;edge=.78
    inside=d<=1
    ring=(d>=edge)&inside
    I=white.copy(); C=zero.copy(); B=zero.copy()
    if k in blue:
        I[:]=.23; B[:]=1
        I[d>.95]=.75; B[d>.95]=.2
    elif k in (14,17):
        I[:]=.28;C[:]=1
        if k==14:I[ring]=.95;C[ring]=0
    elif k==12:
        I[d<edge]=.62;C[d<edge]=.54;B[d<edge]=-.7
        I[(d>.89)&inside]=.50
    elif k in cancelled:
        I[d>.94]=.47
    else:I[ring]=.24;C[ring]=1
    # The sign's safety-colour region is separate from its black ink.
    p=Pen(x,y)
    if k in speeds:
        # scale BOTH dimensions of the glyph by the standard cap height
        cap=(.84 if len(speeds[k])==2 else .76)*(1 if v%3!=2 else .95)
        q=Pen(x/cap,y/cap);number(q,speeds[k],v);p.a=q.a
    elif k in (9,10,41,42):
        istruck=k in (10,42)
        car(p,.30,-.015,.50,.44,1.)
        if istruck:truck(p,-.28,-.024,.64,.57,.68)
        else:car(p,-.30,.035,.51,.44,.68)
        if k not in cancelled:C=np.where((x<.02)&(p.a>.1),1.,C)
    elif k==16:truck(p,0,0,1.28,.67,1.)
    elif k==17:p.box(-.76,-.16,.76,.16)
    elif k==14:
        # Condensed STOP capitals, drawn with paths, not an external font.
        p.curve([(-.46,-.22),(-.71,-.36),(-.76,-.02),(-.58,.008),(-.34,.07),(-.48,.38),(-.70,.21)],.079)
        p.line([(-.39,-.24),(-.12,-.24)],.079);p.line([(-.255,-.24),(-.255,.25)],.085)
        t=np.arange(33)*np.pi/16;p.line(list(zip(.074+.127*np.cos(t),.245*np.sin(t))),.079)
        p.line([(.31,.25),(.31,-.24),(.46,-.24)],.079);p.curve([(.45,-.24),(.72,-.25),(.71,.06),(.46,.06)],.079);p.line([(.46,.06),(.31,.06)],.074)
    elif k in blue:
        if k in (34,37,39):x=-x
        q=Pen(x,y);arrow(q,k,v);p.a=q.a
    elif k in triangles:
        # Centre of the inscribed pictogram area is below the centre of the crop.
        sy=.94 if v%3!=1 else .82; sx=.81 if v%3!=1 else .74
        ex=-x if k==20 or (k in (21,27,28,29,31) and v>=3) else x
        q=Pen(ex/sx,(y-.32)/sy);warning(q,k,v);p.a=q.a
    if k in blue or k in (14,17):
        I=I+(1-I)*p.a;C=C*(1-p.a);B=B*(1-p.a)
    else:I=I*(1-p.a*.96)
    if k==26:
        # Red, yellow, green. The blue deficiency distinguishes an actual
        # green bottom lamp from the black dot of an exclamation mark.
        C[(p.a>.76)&(y<.30)]=1.; C[(p.a>.61)&(y>.41)]=-.72
        B[(p.a>.20)&(p.a<.70)]=-.63
    if k in cancelled:
        a=(x+y)/np.sqrt(2);sw=[.30,.24,.37,.18][v%4]
        mask=(np.abs(a)<sw)&inside
        I[mask]=np.minimum(I[mask],np.where(np.cos(a[mask]*np.pi*5/sw)>.32,.48,.11))
        if k!=32:
            fg=(p.a>0)&(~mask)
            I[fg]=.39 if k==6 else np.where(x[fg]<0,.60,.24)
    # Fixed, class-independent handling of undefined background outside a face.
    I[~inside]=.52;C[~inside]=0;B[~inside]=0
    return np.stack([I,C,B,inside.astype(F),d.astype(F)]).astype(F)

# Network writer
layers=[];weights={}
def layer(name,op,inp,**kw):
    layers.append(dict(name=name,op=op,inputs=inp if isinstance(inp,list) else [inp],**kw));return name
def affine(name,inp,W,b=None,conv=False,pad=0):
    W=np.ascontiguousarray(W,dtype=F);o=len(W)
    if conv:layer(name,'conv',inp,out=o,kernel=[W.shape[-1]]*2,padding=pad,stride=1)
    else:layer(name,'dense',inp,out=o);W=W.reshape(o,-1)
    weights[name+'.W']=W;weights[name+'.b']=np.broadcast_to(np.array(0 if b is None else b,dtype=F),(o,)).copy();return name
def conv1(name,inp,W,b=None):return affine(name,inp,np.array(W,dtype=F)[:,:,None,None],b,True)
def norm(name,inp):return layer(name,'norm',inp)
def add(name,*ins):return layer(name,'add',list(ins))
def relu(name,inp):return layer(name,'relu',inp)
def reshape(name,inp,*shape):return layer(name,'reshape',inp,shape=list(shape))
def pool(name,inp,h,w):return layer(name,'maxpool',inp,size=[h,w],stride=[h,w])

def positive_compress(name,ch):
    # Approximate log-like response for camera intensity. Constant formula,
    # intentionally less compressive than a full logarithm in deep shadow.
    hinges=[0.,.045,.125,.32,.64]; slopes=[4.1,2.5,1.57,.94,.68]
    m=np.tile(np.eye(3,dtype=F),(len(hinges),1));a=conv1(name+'_breaks',ch,m,-np.repeat(hinges,3));a=relu(name+'_ramps',a)
    return conv1(name,a,np.hstack([np.eye(3)*s for s in np.diff([0]+slopes)]))

def clip_symmetric(name,inp,n,c):
    a=conv1(name+'_thresholds',inp,np.vstack([np.eye(n),np.eye(n)]),np.repeat([c,-c],n));a=relu(name+'_thresholded',a)
    return conv1(name,a,np.hstack([np.eye(n),-np.eye(n)]),-c)

# Balanced local template correlation. Subtract the region's own mean and
# normalise its contrast using only this individual synthetic view.
def corr(a,mask,soft=.10):
    mass=mask.sum()+1.e-7; a=a-np.sum(mask*a)/mass
    return mask*a/(mass*np.sqrt(np.sum(mask*a*a)/mass+soft**2))

# Sample single renders with subpixel antialiasing. The grid convention matches
# the interpreter's (no padding) cropped input. Features evaluated at all pixels.
def sample(im,xx,yy):
    u=(xx+1.2)*((M-1)/2.4);v=(yy+1.2)*((M-1)/2.4)
    u=np.clip(u,0,M-1.001);v=np.clip(v,0,M-1.001);i=u.astype(int);j=v.astype(int);fx=u-i;fy=v-j
    return ((1-fy)*((1-fx)*im[:,j,i]+fx*im[:,j,i+1])+fy*((1-fx)*im[:,j+1,i]+fx*im[:,j+1,i+1])).astype(F)

def view(im,grid,cx,cy,rx,ry,angle=0.):
    yy,xx=grid;co=np.cos(angle);si=np.sin(angle); s=None
    for ox,oy in [(-.25,-.25),(.25,-.25),(-.25,.25),(.25,.25)]:
        x=(xx+ox-cx)/rx;y=(yy+oy-cy)/ry
        t=sample(im,x*co+y*si,y*co-x*si)
        if s is None:s=t
        else:s+=t
    return s*.25

def lumrows(k,I,C,B,face,d,xx,yy):
    # Ignore general scenery; use a narrow exterior band only for contour/color.
    mask=np.clip((1.03-d)/.09,0,1)
    if k==12:inner=mask
    elif k in triangles or k==13:inner=np.clip((.75-d)/.11,0,1)
    elif k in blue or k==14 or k==17:inner=np.clip((.94-d)/.14,0,1)
    elif k in cancelled:inner=np.clip((.95-d)/.12,0,1)
    else:inner=np.clip((.77-d)/.11,0,1)
    full=corr(I,mask,.14);symbol=corr(I,inner,.08)
    # Blank signs have no pictogram. Reward a uniform interior separately
    # with a contrast-energy penalty (implemented in signbackground below).
    return full,symbol,mask,inner

# Front end. Normalised compressed R is a good ink channel on red-bordered
# signs; green preserves the outline when the red channel is saturated.
photo=positive_compress('exposure','image')
raw=conv1('spectral',photo,[[.28,.55,.17],[1,-.76,-.24],[-.68,-.32,1],[.80,.16,.04]])
sp=norm('spectral_contrast',raw)
sp=clip_symmetric('bounded_contrast',sp,4,2.4)
# Colour estimates also preserve absolute chroma evidence. This avoids making
# flat grey cancellation signs strongly chromatic merely by normalisation.
ab=conv1('absolute_colour',raw,[[0,0,0,0],[0,3.2,0,0],[0,0,3.2,0],[0,0,0,0]])
sp=add('face_features',sp,ab)
# Local centre-surround, using a DC-rejecting bandpass on compressed red-grey.
# Apply reflection-free valid interior in later dense layers (ignore frame edge).
a=np.arange(-3,4,dtype=F);Y7,X7=np.meshgrid(a,a,indexing='ij')
g1=np.exp(-(X7**2+Y7**2)/(.75**2*2));g1/=g1.sum();g2=np.exp(-(X7**2+Y7**2)/(2.5**2*2));g2/=g2.sum();dog=g1-g2
high=affine('local_ink',photo,np.asarray([[(dog*c) for c in [.68,.26,.06]]]),conv=True,pad=3)
# Use the central portion only for contrast normalisation (zero padding should
# not set the reference for a near-black seal in a very bright environment).
def crop_indices(name,inp,fromsize,n):
    off=(fromsize-n)//2; idx=(np.arange(off,off+n)[:,None]*fromsize+np.arange(off,off+n)[None,:]).ravel()
    # A selector is represented by a single valid convolution when possible.
    k=fromsize-n+1;W=np.zeros((1,1,k,k),F);W[0,0,off,off]=1
    return affine(name,inp,W,conv=True)
# Width 26 includes most of each face; reduce near-DC using a separate local
# normalisation for the smaller central portions later in the glyph reader.
hi=crop_indices('local_middle',high,32,26);hi=norm('ink_strength',hi);hi=clip_symmetric('bounded_ink',hi,1,2.1)

# Small-radius margin and generously-sized variants for perspective/ROI error.
# One individual bank entry for every size, translation, tilt and design style.
# Evaluate luminance/colour on a 16-grid (area-averaged photo); the separate ink
# branch operates at native resolution. No averaging between template images.
coarse=layer('reduce_face','avgpool',sp,size=[2,2],stride=[2,2])
a=2*np.arange(16,dtype=F)-15.;grid16=np.meshgrid(a,a,indexing='ij');a=np.arange(26,dtype=F)-12.5;grid26=np.meshgrid(a,a,indexing='ij')
poses=[(r,ar,dx,dy,ang) for r,ar in [(10.8,1.),(12.8,1.),(14.7,1.),(12.9,.86)] for dx,dy in [(0,0),(-1.35,-.65),(1.35,.65),(-.65,1.35),(.65,-1.35)] for ang in [0.]]
# Small in-plane tilts; the added poses do not replace the straight views.
poses += [(12.8,1.,0.,0.,ang) for ang in [-.13,.13]]
modes=[0,1,2,3]
# Individual soft luminance and colour template weights.
rows=[];details=[];bias=[]
for k in range(43):
    for v in modes:
        im=render(k,v)
        for r,ar,dx,dy,ang in poses:
            A=view(im,grid16,dx,dy,r*ar,r,ang);I,C,B,f,d=A
            full,sy,mask,inner=lumrows(k,*A,*grid16)
            isblue=k in blue; isend=k in cancelled
            # Hue region evidence on both sides of the boundary, including a
            # little exterior: a fixed signed mask, not a fitted classifier.
            cm=np.clip((1.17-d)/.18,0,1)
            wc=corr(C,cm,.17);wb=corr(B,cm,.19)
            if k in (14,17) or isblue:
                # On solid discs inside-vs-outside and mean colour both count.
                if isblue:wb+=.36*B*mask/(np.sum(mask*B)+1.e-7)
                else:wc+=.36*C*mask/(np.sum(mask*C)+1.e-7)
            elif isend:
                # Strong positive red ring or blue disc argues against the end family.
                wc-=.60*mask*(d>.75)/(np.sum(mask*(d>.75))+1.e-7)
                wb-=.35*mask/(mask.sum()+1.e-7)
            elif k!=12:
                wc+=.32*mask*C/(np.sum(mask*C)+1.e-7)
            W=np.stack([.69*full+.16*sy,.38*wc,.34*wb,.28*sy])
            bs=0.
            if k in (13,15):
                W[0]+=.18*full; W[3]+=.12*full
            if k==12:W[0]+=.24*full
            # Use the difference between aligned central ink and its immediate
            # surrounding field; unlike raw intensity it rejects gradients.
            Z=view(im,grid26,dx,dy,r*ar,r,ang);Ii,Ci,Bi,fi,di=Z
            _,_,ma,mi=lumrows(k,*Z,*grid26)
            # Model the same centre-surround filtering on an individual template.
            pad=np.pad(Ii,3,mode='edge');dif=np.zeros_like(Ii)
            for iy in range(7):
                for ix in range(7):dif+=dog[iy,ix]*pad[iy:iy+26,ix:ix+26]
            hd=.22*corr(dif,ma,.065)+.37*corr(dif,mi,.052)
            if k in (13,15):hd+=.15*corr(dif,ma,.065)
            if k==12:hd+=.12*corr(dif,ma,.065)
            # Cancellation signs are achromatic and must not lose just because
            # the colour channel contributes no positive match to their template.
            if isend:W[0]+=.30*full;hd+=.21*corr(dif,ma,.065)
            rows.append(W);details.append(hd);bias.append(bs)

face=affine('match_faces',coarse,rows,bias);ink=affine('match_ink',hi,details)
N=len(poses)*len(modes);s=add('same_pose',face,ink);s=reshape('class_pose',s,1,43,N);s=pool('best_face',s,1,N);face43=reshape('face_scores',s,43)
del rows,details

# Specialised number reader: locally normalised patches, so deep shadow and
# blown-out backgrounds do not suppress recognition of a small first digit.
# Each pose is independently normalised over a central window at inference.
# The dense score matrix operates on fixed central features, so variable pose
# does not require replicating the image explicitly for each template.
numcls=[0,1,2,3,4,5,7,8]
# Normalise ink/character contrast in a small rectangle. Only read inside the
# expected height of the numbers; no competing high-contrast road background.
num_raw=conv1('letter_light',photo,[[.82,.14,.04]])
# A locally range-normalised intensity map. Compare I-lo to fixed fractions
# of hi-lo in a neighbourhood. Soft comparators approximate division using
# only affine layers and ReLUs (and impose a noise floor in deep shadows).
lmax=layer('local_bright','maxpool',num_raw,size=[9,9],stride=[1,1])
lmin=layer('local_dark','maxpool',conv1('reverse_light',num_raw,[[-1.]]),size=[9,9],stride=[1,1]);Lc=crop_indices('local_center',num_raw,32,24)
Ld=add('above_dark',Lc,lmin);Lr=add('local_range',lmax,lmin)
fracs=np.arange(1,9)/9;ww=np.repeat(fracs,2)
ramps=add('range_tests',conv1('range_sampling',Ld,[[1]]*16,np.tile([.018,-.018],8)),conv1('range_threshold',Lr,-ww[:,None]))
u=conv1('local_rank',relu('range_ramps',ramps),[np.tile([1,-1],8)/(8*.036)])
def select_rect(name,inp,h,w):
    # Sparse dense crop with its row-major coordinates explicit.
    W=np.zeros((h*w,32*32),F);y=(32-h)//2;x=(32-w)//2
    W[np.arange(h*w),((y+np.arange(h))[:,None]*32+x+np.arange(w)[None,:]).ravel()]=1
    return reshape(name+'_map',affine(name,inp,W),1,h,w)
nh,nw=16,22
nr=select_rect('glyph_window',num_raw,nh,nw);nr=norm('glyph_contrast',nr);nr=clip_symmetric('glyph_bounded',nr,1,2.2)
nhp=select_rect('glyph_local_window',high,nh,nw);nhp=norm('glyph_band_contrast',nhp);nhp=clip_symmetric('glyph_band_bounded',nhp,1,2.2)
gy,gx=np.meshgrid(np.arange(nh,dtype=F)-(nh-1)/2,np.arange(nw,dtype=F)-(nw-1)/2,indexing='ij')
# Patches for numbers may be larger/smaller than typical; the last digits in
# particular are often narrowly spaced. Read the first and second part as
# separate evidence as well as the whole string.
numrows=[];numhi=[]
for k in numcls:
    text=speeds[k];
    for v in range(4):
        # A digit-string render without a ring; only the string varies here.
        p=Pen(X,Y);number(p,text,v);im=(1.-p.a)[None,:,:]
        for h in [7.2,9.0,10.8,12.6]:
            for aspect in [.80,1.,1.18]:
                for dx in [-1.5,0.,1.5]:
                    for dy in [-1.4,0.,1.4]:
                        for ang in [-.075,.075]:
                            cap=h; Xr=(gx-dx)/(cap*aspect);Yr=(gy-dy)/cap
                            I=view(im,(gy,gx),dx,dy,cap*aspect,cap,ang)[0]
                            half=.70 if len(text)==2 else .86
                            ma=np.clip((half+.04-np.abs(Xr))/.10,0,1)*np.clip((.62-np.abs(Yr))/.10,0,1)
                            first=ma*np.clip((.05-Xr)/.12,0,1)
                            # In three-digit strings the digits distinguishing
                            # 100 and 120 are in the middle, after the narrow 1.
                            if len(text)==3:first=ma*np.clip((.33-np.abs(Xr+.06))/.12,0,1)
                            pe=np.pad(I,3,mode='edge');D=np.zeros_like(I)
                            for yy,xx in np.ndindex(7,7):D+=dog[yy,xx]*pe[yy:yy+nh,xx:xx+nw]
                            numrows.append(.40*corr(I,ma)+.24*corr(I,first));numhi.append(.23*corr(D,ma,.055)+.13*corr(D,first,.055))
ns=add('combined_reading',affine('read_numbers',nr,numrows),affine('read_numbers_band',nhp,numhi));ns=reshape('number_pose',ns,1,len(numcls),len(numrows)//len(numcls));ns=pool('best_reading',ns,1,len(numrows)//len(numcls));ns=reshape('numerals',ns,len(numcls));del numrows,numhi
# Add a centred readout *relative to other plausible numbers*, rather than a
# constant speed-limit bonus on every face with a dark spot in its centre.
m=np.zeros((43,8),F)
for j,k in enumerate(numcls):m[k]=-.81/8;m[k,j]+=.81
read43=affine('letter_evidence',ns,m)
# A second measurement registers the inside of a sign independently of the
# sign surround. Canonical samples are a regular grid on a disk/triangle;
# input vectors get instance normalisation at EACH hypothesised pose. Then a
# shared 1x1 convolution computes cosine similarity in the registered frame.
# A contrast floor protects truly blank fields from becoming large evidence.

def register_input(name,inp,n,srcx,srcy,floor):
    poses,m=srcx.shape;W=np.zeros((poses,m+2,n,n),F);b=np.zeros((poses,m+2),F)
    a=np.arange(poses)[:,None];t=np.arange(m)[None,:];
    midx,midy=0,0
    if n==22 and inp in (num_raw,high):
        midy=2 if name.startswith('warn') else 0
        # Only the central window can contribute to the aperture. Factor the
        # sparse bilinear map into a crop and a smaller dense map.
        C=np.zeros((1,1,11,11),F);C[0,0,5+midy,5]=1.
        inp=affine(name+'_crop',inp,C,conv=True)
    x=np.clip(srcx-midx+(n-1)*.5,0,n-1.001);y=np.clip(srcy-midy+(n-1)*.5,0,n-1.001)
    ix=x.astype(int);iy=y.astype(int);fx=x-ix;fy=y-iy
    for dy,wy in [(0,1-fy),(1,fy)]:
        for dx,wx in [(0,1-fx),(1,fx)]:W[a,t,iy+dy,ix+dx]+=wy*wx
    W[:,m:]=W[:,:m].mean(axis=1,keepdims=True)
    b[:,m:]=np.array([-1.,1.])*floor*np.sqrt((m+2)/2)
    r=affine(name+'_sample',inp,W.reshape(poses*(m+2),-1),b.ravel());r=reshape(name+'_windows',r,poses,1,m+2)
    r=norm(name+'_normalize',r)
    # View the independent pose index as a spatial index, so dot products
    # against every template can share their weights across hypotheses.
    r=reshape(name+'_poses',r,1,poses,m+2)
    return r

def dotbank(name,inp,rows,poses,m,outputs):
    # Fold each vector to a square patch; tile poses vertically and stride
    # one complete patch per dot product. No transpose layer is required.
    side=int(np.ceil(np.sqrt(m+2)))
    # Register_input has a square sample count (padding not needed).
    assert side*side==m+2
    inp=reshape(name+'_tiles',inp,1,poses*side,side)
    packed=np.zeros((len(rows),1,side*side),F);packed[:,0,:m]=np.array(rows,F)
    layer(name,'conv',inp,out=len(rows),kernel=[side,side],stride=side,padding=0)
    weights[name+'.W']=packed.reshape(len(rows),1,side,side);weights[name+'.b']=np.zeros(len(rows),F)
    return name

def icon_network(name,ks,typ):
    # Small fixed aperture in canonical sign coords. Pairs nearest to each
    # shape's centre form the entire sampling aperture (not image-selected).
    if typ=='tri':
        gx,gy=np.meshgrid(np.linspace(-.84,.84,23),np.linspace(-.60,.84,21));level=tri(gx,gy)
    elif typ=='stripe':
        gx,gy=np.meshgrid(np.linspace(-.87,.87,21),np.linspace(-.87,.87,21));level=gx**2+gy**2
    else:
        gx,gy=np.meshgrid(np.linspace(-.76,.76,23),np.linspace(-.64,.64,19));level=(gx/.78)**2+(gy/.68)**2
    side=15 if typ!='stripe' else 17
    m=side**2-2;active=np.argsort(level,axis=None,kind='stable')[:m];x=gx.ravel()[active];y=gy.ravel()[active]
    poses=[(rx,ry,ang,cx,cy) for rx,ry in [(10.,10.),(12.,12.),(14.,14.),(15.,15.),(10.8,13.5)] for ang,cx,cy in [(0,0,0),(0,-1.65,0),(0,1.65,0),(0,0,-1.65),(0,0,1.65),(-.13,-.9,-.9),(.13,.9,.9),(-.13,.9,-.9),(.13,-.9,.9)]]
    pcount=len(poses)
    sx=np.array([cx+rx*(x*np.cos(a)-y*np.sin(a)) for rx,ry,a,cx,cy in poses]);sy=np.array([cy+ry*(x*np.sin(a)+y*np.cos(a)) for rx,ry,a,cx,cy in poses]);
    nr=register_input(name+'_gray',num_raw,22,sx,sy,.017)
    hr=register_input(name+'_band',high,22,sx,sy,.007)
    lr=register_input(name+'_rank',u,24,sx,sy,.065)
    # Integrate the individual ideal at neighbouring points to allow for
    # optics and aliasing at these sizes; no combination of distinct renders.
    rows=[];bands=[];distfg=[];distbg=[];rankrows=[];flanks=[];flanks_band=[];nstyle=8
    for k in ks:
        for v in range(nstyle):
            vs=v if typ=='tri' else v%4
            im=render(k,vs)
            # Local image sampling, in units of height of the drawn face.
            z=im[:1];dy=(.075 if v>=4 else 0) if typ=='tri' else 0
            wx=1. if v%4!=2 else .90;wy=(1.07 if v>=4 else .97)
            iy=(y-dy-(.32 if typ=='tri' else 0))/wy+(.32 if typ=='tri' else 0);ix=x/wx
            # Template for absolute grey and its bandpassed version. Smooth
            # each rendering before comparing with bilinearly sampled pixels.
            def tap(ox,oy):return sample(z,ix+ox,iy+oy)[0]
            smooth=np.zeros(m,F);sm2=np.zeros(m,F)
            for qy,qx in np.ndindex(7,7):
                val=tap((qx-3)/12.5,(qy-3)/12.5);smooth+=g1[qy,qx]*val;sm2+=g2[qy,qx]*val
            w=np.ones_like(x);rows.append(.48*corr(smooth,w,.075));bands.append(.34*corr(smooth-sm2,w,.045))
            # Soft local-rank counterpart of THIS rendering; construct both
            # extrema in the same drawn face, never between different views.
            hi_rank=smooth.copy();lo_rank=smooth.copy()
            for ry,rx in np.ndindex(5,5):
                sample_i=tap((rx-2)*.14,(ry-2)*.14);hi_rank=np.maximum(hi_rank,sample_i);lo_rank=np.minimum(lo_rank,sample_i)
            ra=(smooth-lo_rank+.035)/(hi_rank-lo_rank+.07);rankrows.append(.35*corr(ra,w,.09))
            # Symmetric distance-like matching. For foreground AND background
            # sites ask if there is compatible ink/paper nearby in the photo.
            # Conversely place tiny negative weights only far from the model
            # ink/paper, rather than punishing an edge for shifting one pixel.
            V=(1.-smooth); near=V.copy();lo=V.copy()
            for ox,oy in [(-.09,0),(.09,0),(0,.09),(0,-.09),(-.065,-.065),(.065,.065),(-.065,.065),(.065,-.065)]:
                neighbour=1.-tap(ox,oy);near=np.maximum(near,neighbour);lo=np.minimum(lo,neighbour)
            def bw(pos,neg):
                # A handful of antialiased sites cannot alone certify a match.
                return pos/(pos.sum()+m*.03)-neg/(neg.sum()+m*.03)
            distfg.append(bw(V**1.5,np.maximum(0,1-near)**1.5))
            distbg.append(bw((1-V)**2,lo**1.5))
            if typ=='stripe':
                # The broad central stripe is common to every cancellation.
                # Only the exposed fragments beside it identify what ended.
                fl=np.clip((np.abs(ix+iy)/np.sqrt(2)-.23)/.12,0,1)*np.clip((.77-np.hypot(ix,iy))/.13,0,1)
                flanks.append(.55*corr(smooth,fl,.075));flanks_band.append(.45*corr(smooth-sm2,fl,.05))
    na=dotbank(name+'_symbols',nr,rows,pcount,m,len(ks));ha=dotbank(name+'_lines',hr,bands,pcount,m,len(ks))
    # nr is normalised separately at each pose. Bound its foreground and
    # background responses so a single black border pixel cannot outweigh
    # agreement across an entire glyph. The convolution bank implements
    # tolerances on the template side of this distance.
    z=reshape(name+'_normalfield',nr,1,pcount,m+2)
    pos=relu(name+'_dark_response',conv1(name+'_dark_level',z,[[-1.]],[-.10]));neg=relu(name+'_light_response',conv1(name+'_light_level',z,[[1.]],[.05]))
    def saturate(tag,f,h):
        a=conv1(tag+'_over',f,[[1]],[-h]);a=relu(tag+'_excess',a);return add(tag,f,conv1(tag+'_subtract',a,[[-1]]))
    pos=saturate(name+'_bounded_dark',pos,1.3);neg=saturate(name+'_bounded_light',neg,.90)
    df=dotbank(name+'_ink_distance',pos,distfg,pcount,m,len(ks));db=dotbank(name+'_paper_distance',neg,distbg,pcount,m,len(ks))
    # Preserve topology as well as local contrast. Complement the inner
    # product only gently: distant faint details are not reliable.
    def scale_all(tag,z,a):return conv1(tag,z,np.eye(len(ks)*nstyle,dtype=F)*a)
    ds=scale_all(name+'_distance_vote',add(name+'_two_distances',df,db),.32)
    ss=[na,ha,ds,dotbank(name+'_local_rank',lr,rankrows,pcount,m,len(ks))]
    if typ=='stripe':
        fg=dotbank(name+'_exposed_ink',nr,flanks,pcount,m,len(ks));bg=dotbank(name+'_exposed_band',hr,flanks_band,pcount,m,len(ks));ss.append(scale_all(name+'_exposed_vote',add(name+'_side_fragments',fg,bg),.77))
    s=add(name+'_aligned',*ss);s=reshape(name+'_pose_styles',s,1,len(ks),pcount*nstyle);s=pool(name+'_winner',s,1,pcount*nstyle);s=reshape(name+'_classes',s,len(ks))
    # Absolute correlation recognizes the sign family; relative correlation
    # resolves its members. The offset has the interpretation of a middling
    # cosine match, not a class prior.
    W=np.zeros((43,len(ks)),F);b=np.zeros(43,F)
    for j,k in enumerate(ks):
        W[k]=-.60/len(ks);W[k,j]+=.99;b[k]=-.39*(.70 if typ=='stripe' else .60)
        if k==15:
            # No positive ink required; evidence for another round-sign
            # marking argues against a blank sign. The ink-energy branch
            # compares absolute contrast independently.
            W[k]=-.44/len(ks);b[k]=.23
    return affine(name+'_evidence',s,W,b)

reg_tri=icon_network('warn',sorted(triangles),'tri')
reg_ring=icon_network('restriction',numcls+[9,10,15,16],'round')
reg_end=icon_network('cancellation',[6,32,41,42],'stripe')
# Explicit absence-of-ink evidence for the blank circular sign. The absolute
# band contrast in its centre is measured in whole-sign contrast units.
absv=conv1('ink_signs',hi,[[1],[-1]]);absv=relu('ink_halves',absv);absv=conv1('ink_magnitude',absv,[[1,1]])
y,x=grid26; aperture=np.exp(-.5*((x/4.5)**2+(y/3.8)**2));aperture/=aperture.sum();blank=np.zeros((43,26,26),F);blank[15]=-.80*aperture;bo=np.zeros(43,F);bo[15]=.12
bs=affine('blank_interior',absv,blank,bo)
# Contour-only subsystem. A sign border is a CLOSED curve: a match to
# just one red strip (common in a cluttered crop) is not enough. Split the
# expected margin into four sectors and combine the weakest sector with the
# average. Evaluate the field on both sides, independent of the icon within.
controws=[];families=['ring','warn','yield','diamond','blue','end','stop','entry'];
owners=[[0,1,2,3,4,5,7,8,9,10,15,16],sorted(triangles),[13],[12],sorted(blue),sorted(cancelled),[14],[17]]
major=[0,11,13,12,35,32,14,17];poses2=[(rx,ry,dx,dy,ang) for rx,ry in [(10.2,10.2),(11.8,11.8),(13.1,13.1),(14.5,14.5),(11.0,13.2),(13.7,12.3)] for dx,dy,ang in [(0,0,0),(-1.3,0,0),(1.3,0,0),(0,-1.3,0),(0,1.3,0),(-1.,-1.,-.10),(1.,1.,-.10),(-1.,1.,.10),(1.,-1.,.10)]]
for fam,k in zip(families,major):
    for rx,ry,dx,dy,ang in poses2:
        y,x=grid16
        x=(x-dx)/rx;y=(y-dy)/ry;x,y=x*np.cos(ang)+y*np.sin(ang),y*np.cos(ang)-x*np.sin(ang)
        if k in (11,13):d=tri(x,y if k==11 else -y)
        elif k==12:d=np.abs(x)+np.abs(y)
        elif k==14:d=np.maximum(np.maximum(np.abs(x),np.abs(y)),(np.abs(x)+np.abs(y))/np.sqrt(2))
        else:d=np.sqrt(x*x+y*y)
        edge={'ring':.78,'warn':.72,'yield':.69,'diamond':.68,'end':.95,'stop':.88,'blue':.945,'entry':.95}[fam]
        def step(d,c):return np.clip(.5+(c-d)*6.5,0,1)
        within=step(d,1.);inside=step(d,edge);rim=within-inside
        boundary=np.clip((1.18-d)/.10,0,1)
        if fam in ['ring','warn','yield']:
            # Use the inner and outer rim, not text. Four simultaneous matches
            # limit the benefit of finding an accidental red patch.
            use=np.clip((d-(edge-.26))/.13,0,1)*boundary
            light=-rim+.55*inside-.17*(1-within);red=rim;bl=0*d
        elif fam=='diamond':
            use=boundary;light=.92*rim+.48*inside+.24*(1-within);red=.44*inside;bl=-inside
        elif fam=='end':
            # End-of-restriction signs have opposing light lobes and a dark
            # diagonal band, but no red around the white perimeter.
            use=boundary;stripe=step(np.abs(x+y)/np.sqrt(2),.31);light=within*(1-stripe)-.12*rim+.32*(1-within);red=0*d;bl=0*d
        elif fam=='stop':
            use=boundary*(.20+.80*np.clip((d-.53)/.28,0,1));light=.23*inside+rim+.53*(1-within);red=inside;bl=0*d
        elif fam=='entry':
            use=boundary;stripe=step(np.abs(y),.17)*step(np.abs(x),.74);light=.23*within+.72*stripe+.52*(1-within);red=within*(1-stripe);bl=0*d
        else:
            use=boundary*np.clip((d-.52)/.23,0,1);light=.23*inside+.77*rim+.48*(1-within);red=0*d;bl=inside
        ay=y-1/3 if k==11 else (y+1/3 if k==13 else y);at=np.arctan2(ay,x)
        for sector in range(4):
            # Broad half-lobes overlap slightly at the edges.
            tap=np.clip(1.15+2.1*(np.cos(at-sector*np.pi/2)-.77),0,1)**2
            mask=tap*use
            L=corr(light,mask,.14);R=corr(red,mask,.20);B=corr(bl,mask,.20)
            if fam in ['ring','warn','yield','stop','entry']:
                R+=.33*mask*red/(np.sum(mask*red)+1e-6)
            elif fam=='blue':B+=.36*mask*bl/(np.sum(mask*bl)+1e-6)
            elif fam=='end':R-=.27*mask*rim/(np.sum(mask*rim)+1.e-6)
            if fam=='diamond':lwt,rwt,bwt=.52,.14,.54
            elif fam=='end':lwt,rwt,bwt=1.,.35,0.
            else:lwt,rwt,bwt=.65,.47,.47
            controws.append(np.stack([lwt*L,rwt*R,bwt*B,0*d]))
cc=affine('contour_sectors',coarse,controws);del controws
cc=reshape('sectors_by_pose',cc,1,len(poses2)*len(families),4)
cm=layer('contour_average','avgpool',cc,size=[1,4],stride=[1,4]);neg=conv1('negative_sectors',cc,[[-1.]])
worst=pool('least_agreement',neg,1,4);worst=conv1('closedness',worst,[[-.40]]);cm=conv1('overall_contour',cm,[[.60]])
cl=reshape('contour_poses',add('contour_agreement',cm,worst),1,len(families),len(poses2));cl=reshape('best_contours',pool('contour_pool',cl,1,len(poses2)),len(families))
mat=np.zeros((43,len(major)),F)
for g,group in enumerate(owners):mat[group,g]=.70
shape43=affine('contour_evidence',cl,mat)
# Dedicated part reader: use separate masks for the distinguishing digit
# and the rest of the number. In particular, a centre 0 and centre 2 are
# compared at exactly the same family of small and narrow triple-digit poses.
# Normalize the input with the *local range* instead of the whole crop.
W=np.zeros((nh*nw,24,24),F)
for yy,xx in np.ndindex(nh,nw):W[yy*nw+xx,yy+(24-nh)//2,xx+(24-nw)//2]=1
nrk=norm('numeral_rank_normalised',reshape('numeral_rank_window',affine('crop_number_rank',u,W),1,nh,nw))
pr=[];phr=[]
for k in numcls:
    text=speeds[k];n=len(text)
    for v in range(6):
        # Four independent stroke details: compact 8, flat-topped 3, and
        # transversely compressed stems are frequent DIN lettering features.
        w=(.54 if v%2==0 else .62) if n==2 else (.48 if v%2==0 else .55)
        widths=[w*(.65 if d=='1' else 1) for d in text];gap=[.085,.13,.17][v%3] if n==2 else [.035,.07,.115][v%3]
        left=-(sum(widths)+gap*(n-1))/2;pos=left;p=Pen(X,Y);dj=0 if n==2 else 1
        for j,(ch,ww) in enumerate(zip(text,widths)):
            pen=LetterPen(p,pos,-.5,ww,1.);th=[.106,.135,.16][v%3]
            if ch=='3' and v>=3:
                pen.line([(.13,.08),(.86,.08),(.47,.46)],th);pen.curve([(.47,.46),(.72,.42),(.92,.52),(.86,.72),(.81,.99),(.33,.98),(.12,.81)],th)
            elif ch=='8' and v>=3:
                pen.curve([(.50,.48),(.08,.40),(.13,.06),(.50,.07),(.88,.04),(.95,.34),(.50,.48),(.03,.59),(.06,.96),(.50,.93),(.94,.96),(.97,.60),(.50,.48)],th)
            elif ch=='0' and v%2:
                t=np.arange(41)*np.pi/20;pen.line(list(zip(.5+.37*np.cos(t),.5+.432*np.sin(t))),th)
            else:digit(pen,ch,th,v)
            if j==dj:lft,rgt=pos,pos+ww
            pos+=ww+gap
        im=(1.-p.a)[None,:,:]
        for cap in [6.5,8.,9.5,11.,12.5]:
            for ar in [.78,.96,1.14]:
                for dx in [-1.8,0,1.8]:
                    for dy in [-1.5,0,1.5]:
                        for ang in [-.07,.07]:
                            xx=(gx-dx)/(cap*ar);yy=(gy-dy)/cap;x=xx*np.cos(ang)+yy*np.sin(ang);y=yy*np.cos(ang)-xx*np.sin(ang)
                            A=view(im,(gy,gx),dx,dy,cap*ar,cap,ang)[0]
                            ma=np.clip((x-left+.11)/.11,0,1)*np.clip((-left-x+.11)/.11,0,1)*np.clip((.62-np.abs(y))/.10,0,1)
                            focus=np.clip((x-lft+.08)/.08,0,1)*np.clip((rgt-x+.08)/.08,0,1)*ma
                            # The background at the central gap also informs
                            # two versus three digits. Never reward a small 0
                            # in isolation just because one could be found.
                            pr.append(.29*corr(A,ma)+.36*corr(A,focus,.12))
                            pe=np.pad(A,3,mode='edge');D=np.zeros_like(A)
                            for ty,tx in np.ndindex(7,7):D+=dog[ty,tx]*pe[ty:ty+nh,tx:tx+nw]
                            phr.append(.17*corr(D,ma,.06)+.18*corr(D,focus,.065))
pq=add('registered_parts',affine('compare_part_rank',nrk,pr),affine('compare_part_band',nhp,phr));pn=len(pr)//len(numcls);del pr,phr
pq=reshape('digit_part_poses',pq,1,len(numcls),pn);pq=reshape('digit_part_classes',pool('digit_part_pool',pq,1,pn),len(numcls));mat=np.zeros((43,8),F);base=np.zeros(43,F)
for j,k in enumerate(numcls):
    mat[k]=-.70/8;mat[k,j]+=1.03;base[k]=-.33*.56
part43=affine('digit_part_evidence',pq,mat,base)
# A matched interior for a blue arrow or for the word STOP is useful even
# against a brightly coloured background. White-on-colour semantics are built
# into the drawing; a plain red patch should not be a substitute for STOP.
reg_solid=icon_network('solid',[14,17,33,34,35,36,37,38,39,40],'round')
add('scores',face43,read43,reg_tri,reg_ring,reg_end,bs,shape43,part43,reg_solid)

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
(out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':')))
np.savez(out/'weights.npz',**weights)
