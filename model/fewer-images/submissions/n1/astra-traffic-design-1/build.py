"""A hand-specified, part-based sign reader. No input assets, fonts, or examples.
Coordinates in the drawing code are fractions of a sign's nominal half width.
Each raster is used only for its own matched filter. Geometry hypotheses use
Cartesian products or deterministic low-discrepancy sequences, not fitting.
"""
import sys, json, math
from pathlib import Path
import numpy as np

LAY=[]; WE={}
def op(n,t,i,**kw):
    LAY.append(dict(name=n,op=t,inputs=([i] if isinstance(i,str) else i),**kw)); return n
def dense(n,i,w,b=0):
    w=np.asarray(w,'f'); WE[n+'.W']=w; WE[n+'.b']=np.broadcast_to(np.float32(b), (len(w),)).copy(); return op(n,'dense',i,out=len(w))
def conv(n,i,w,b=0,p=0):
    w=np.asarray(w,'f'); WE[n+'.W']=w; WE[n+'.b']=np.broadcast_to(np.float32(b),(len(w),)).copy(); return op(n,'conv',i,out=len(w),kernel=list(w.shape[-2:]),stride=1,padding=p)
def mix(n,i,w,b=0): return conv(n,i,np.array(w,'f')[:,:,None,None],b)
def rs(n,i,shape):return op(n,'reshape',i,shape=list(shape))
def pool(n,t,i,h,w):return op(n,t,i,size=[h,w],stride=[h,w])
def gauss(r,s):
    u=np.exp(-.5*(np.arange(-r,r+1)/s)**2); u/=u.sum();return np.outer(u,u)
def blur(a,s):
    if s<.05:return a
    r=math.ceil(2.6*s); u=np.exp(-.5*(np.arange(-r,r+1)/s)**2);u/=u.sum()
    for ax in [-1,-2]:
        pd=[(0,0)]*a.ndim;pd[ax]=(r,r); q=np.pad(a,pd,mode='edge');a=sum(w*np.take(q,range(i,i+a.shape[ax]),axis=ax) for i,w in enumerate(u))
    return a

# Rendering primitives: polygons, elliptical spots, and round-ended pen strokes.
S=161
v=(np.arange(S,dtype=float)-(S-1)/2)/((S-1)/2)*1.12
XX,YY=np.meshgrid(v,v)
class Pen:
    def __init__(self,x=XX,y=YY):self.x=x;self.y=y;self.a=np.zeros_like(x)
    def shape(self,inside,val):self.a[inside]=val
    def oval(self,x,y,rx,ry,val=1):self.shape(((self.x-x)/rx)**2+((self.y-y)/ry)**2<=1,val)
    def poly(self,pts,val=1):
        inside=np.zeros(self.x.shape,bool);x0,y0=pts[-1]
        for x,y in pts:
            if y!=y0: inside^=((self.y>y)!=(self.y>y0))&(self.x<(x0-x)*(self.y-y)/(y0-y)+x)
            x0,y0=x,y
        self.shape(inside,val)
    def line(self,pts,width,val=1):
        for (x,y),(a,b) in zip(pts,pts[1:]):
            u=a-x;v=b-y;t=np.clip(((self.x-x)*u+(self.y-y)*v)/(u*u+v*v+1e-25),0,1)
            self.shape((self.x-x-t*u)**2+(self.y-y-t*v)**2<=(width/2)**2,val)
    def box(self,x0,y0,x1,y1,val=1):self.shape((self.x>=x0)&(self.x<=x1)&(self.y>=y0)&(self.y<=y1),val)
    def ring(self,x,y,rx,ry,width,val=1):self.shape((np.sqrt(((self.x-x)/rx)**2+((self.y-y)/ry)**2)-1).__abs__()<width/(rx+ry),val)

def arrow(p,pts,sw,hw,hl):
    # The head ends exactly at the last waypoint.
    pts=np.array(pts,float); tip=pts[-1];d=tip-pts[-2];d/=np.linalg.norm(d); side=np.array([-d[1],d[0]]);base=tip-hl*d
    p.line([*pts[:-1],base+sw*.22*d],sw);p.poly([tip,base+hw*.5*side,base-hw*.5*side])

def car(p,cx,cy,w,h,val=1):
    pts=[(-.48,.46),(-.49,-.01),(-.41,-.12),(-.30,-.44),(-.20,-.5),(.24,-.5),(.32,-.43),(.42,-.14),(.49,-.02),(.49,.42),(.39,.42),(.37,.51),(.26,.51),(.24,.40),(-.26,.40),(-.27,.51),(-.41,.51),(-.44,.42)]
    p.poly([(cx+x*w,cy+y*h) for x,y in pts],val)

def bike(p,cx,cy,s):
    x=lambda x:cx+s*x;y=lambda y:cy+s*y
    for a in [-.6,.59]:p.ring(x(a),y(.3),s*.34,s*.34,s*.085)
    for pts in [[(-.6,.3),(-.31,-.17),(.02,.29),(-.6,.3)],[(-.30,-.18),(.43,-.20),(.02,.29)],[(.36,-.44),(.59,.3)],[(.38,-.42),(.56,-.42)],[(-.30,-.18),(-.36,-.41)],[(-.49,-.42),(-.27,-.42)]]:p.line([(x(a),y(b)) for a,b in pts],s*.092)

def person(p,cx,cy,h,bent=0,sgn=1):
    pts=lambda uv:[(cx+sgn*x*h,cy+y*h) for x,y in uv]
    p.oval(cx+sgn*.095*h,cy-.39*h,.092*h,.10*h)
    p.line(pts([(.055,-.23),(-.02,.05)]),.17*h)
    p.line(pts([(-.025,.04),(-.11,.29),(-.27,.47)]),.094*h)
    p.line(pts([(-.01,.035),(.15,.23),(.18,.47)]),.102*h)
    p.line(pts([(.04,-.22),(.15,-.07),(.29,-.03)]),.089*h)
    p.line(pts([(.04,-.20),(-.12,-.12),(-.18,.035)]),.084*h)

# Interior pictograms in a standard upright warning triangle.
def warning(k,style):
    p=Pen(); v=style%2
    if k==11:
        p.poly([(-.13,.63),(-.13,.38),(-.43,.48),(-.43,.36),(-.125,.14),(-.12,-.01),(-.235,.0),(0,-.44),(.235,0),(.12,-.01),(.125,.14),(.43,.36),(.43,.48),(.13,.38),(.13,.63)])
    if k==18:
        p.poly([(-.09,-.37),(-.06,-.42),(.06,-.42),(.09,-.37),(.052,.34),(-.05,.34)]);p.oval(0,.50,.084,.087)
    if k in [19,20]:
        # Thick road line running from the bottom into a left/right bend.
        f=1 if k==19 else -1
        pts=[(.29,.60),(.29,.18),(.27,.075),(.19,-.02),(-.03,-.12),(-.15,-.20),(-.17,-.30)] if not v else [(.28,.60),(.28,.18),(.25,.07),(.19,-.015),(.09,-.075),(-.16,-.18),(-.22,-.25)]
        p.line([(f*x,y) for x,y in pts],.18);p.a[YY>.61]=0
    if k==21:
        pts=[(.28,.61),(.28,.20),(.24,.12),(.04,-.015),(-.10,-.10),(-.12,-.18),(-.11,-.24),(.01,-.34),(.04,-.40)]
        p.line(pts,.165);p.a[YY>.61]=0
        if v:p.a=p.a[:,::-1].copy()
    if k==22:
        x=XX/.53; z=np.maximum(0,1-((x+.46)/.53)**2)**.7+np.maximum(0,1-((x-.46)/.53)**2)**.7
        p.shape((abs(x)<1)&(YY<=.54)&(YY>.53-.24*z),1)
    if k==23:
        # Rear of a tilting car above two wavering skid marks.
        a=-.28; X=XX*np.cos(a)+YY*np.sin(a);Y=-XX*np.sin(a)+YY*np.cos(a);q=Pen(X,Y);car(q,.0,-.03,.57,.41);p.a=q.a
        for xx in [-.16,.16]:
            ys=np.linspace(.16,.62,18);p.line(list(zip(xx+.105*np.sin((ys-.16)*13+.7)-.04,ys)),.057)
    if k==24:
        p.line([(-.29,.60),(-.29,-.28)],.115);p.line([(.30,.60),(.30,.29),(.27,.19),(.135,-.015),(.12,-.12),(.12,-.28)],.115);p.a[YY>.61]=0
    if k==25:
        p.oval(-.035,-.22,.096,.101);p.poly([(-.03,-.12),(-.18,-.15),(-.34,.05),(-.29,.17),(-.13,.09),(.002,-.045)])
        p.line([(-.27,.105),(-.28,.30),(-.43,.56)],.103);p.line([(-.25,.115),(-.13,.30),(-.105,.56)],.115)
        p.line([(-.095,-.066),(-.055,.105),(.13,.22)],.074);p.line([(-.18,-.055),(-.16,.11),(-.04,.16)],.067);p.line([(-.34,.035),(.34,.44)],.045)
        p.poly([(.02,.57),(.19,.42),(.25,.35),(.34,.36),(.43,.46),(.53,.50),(.58,.57)])
        if v:p.a=p.a[:,::-1].copy()
    if k==26:
        p.box(-.125,-.34,.125,.55);p.oval(0,-.32,.125,.11);p.oval(0,.54,.125,.085)
        for cy,value in [(-.23,.56),(.095,.22),(.413,.62)]:p.oval(0,cy,.086,.092,value)
    if k==27:
        person(p,-.015,.06,.99,sgn=-1 if v else 1)
        for cx in [-.38,-.13,.13,.38]:p.poly([(cx-.06,.51),(cx+.045,.51),(cx+.10,.62),(cx-.11,.62)])
    if k==28:
        person(p,.105,.035,1.00,sgn=1);person(p,-.275,.235,.62,sgn=1)
        p.poly([(-.30,.18),(-.405,.38),(-.19,.38)]);p.line([(.01,-.105),(-.105,.045),(-.205,.11)],.061)
        if v:p.a=p.a[:,::-1].copy()
    if k==29:
        bike(p,0,.264,.57)
        if v: # Mounted cyclist rather than an empty cycle frame.
            p.oval(.03,-.20,.081,.081);p.line([(-.08,-.10),(-.15,.01),(-.205,.115),(-.04,.20),(-.09,.34)],.087);p.line([(-.075,-.095),(.045,.012),(.19,.055)],.065)
    if k==30:
        for i in range(6):
            th=np.pi*(i/3-.5); ux,uy=np.cos(th),np.sin(th); vx,vy=-uy,ux;cy=.18
            p.line([(0,cy),(.446*ux,cy+.446*uy)],.065)
            for loc in [.24,.345]:
                for side in [-1,1]:p.line([(loc*ux,cy+loc*uy),((loc+.095)*ux+.094*side*vx,cy+(loc+.095)*uy+.094*side*vy)],.041)
    if k==31:
        # Leaping deer, facing left. Body, neck and antlers form one silhouette.
        p.poly([(-.52,-.035),(-.45,-.12),(-.34,-.14),(-.30,-.20),(-.245,-.18),(-.18,-.045),(-.09,.048),(.15,.08),(.34,.14),(.39,.21),(.49,.18),(.435,.28),(.33,.32),(.20,.31),(.04,.23),(-.13,.235),(-.28,.15),(-.35,-.02),(-.44,.002)])
        for pts in [[(-.15,.18),(-.30,.27),(-.41,.52)],[(-.055,.20),(-.19,.32),(-.25,.50)],[(.24,.24),(.21,.36),(.43,.50),(.53,.55)],[(.31,.23),(.33,.35),(.51,.46)],[(-.33,-.145),(-.32,-.26),(-.22,-.35)],[(-.315,-.23),(-.40,-.31)]]:p.line(pts,.044)
        if v:p.a=p.a[:,::-1].copy()
    return p.a

# Standard mandatory arrows and the interiors of regulatory signs.
def regulatory(k,v):
    p=Pen()
    if k in [9,10,41,42]:
        if k in [9,41]:car(p,-.285,.015,.51,.42,.58 if k==9 else .65)
        else:
            p.box(-.57,-.30,-.23,.20,.63);p.poly([(-.55,.26),(-.55,.14),(-.22,.14),(-.22,-.04),(-.135,-.04),(-.075,.065),(-.075,.23)],.63);p.oval(-.49,.23,.062,.065,.63);p.oval(-.165,.23,.062,.065,.63)
        car(p,.275,.06,.48,.36)
    if k==16:
        p.box(-.195,-.30,.57,.17);p.poly([(-.57,.20),(-.57,-.05),(-.46,-.20),(-.245,-.20),(-.245,.14),(.54,.14),(.54,.235),(-.56,.235)]);p.oval(-.39,.22,.115,.115);p.oval(.34,.22,.115,.115)
        p.poly([(-.48,-.057),(-.417,-.147),(-.327,-.147),(-.327,-.057)],0)
    if k in [33,34]:
        s=1 if k==33 else -1
        pts=[(-.36,.60),(-.36,.22),(-.35,.055),(-.32,-.07),(-.25,-.18),(-.15,-.252),(-.025,-.28),(.14,-.28),(.32,-.28),(.665,-.28)]
        arrow(p,[(s*x,y) for x,y in pts],.206,.65,.33);p.a[YY>.62]=0
    if k==35:arrow(p,[(0,.65),(0,-.09),(0,-.75)],.20,.89,.45);p.a[YY>.655]=0
    if k in [36,37]:
        s=1 if k==36 else -1
        arrow(p,[(-s*.14,.65),(-s*.14,-.22),(-s*.14,-.75)],.169,.64,.322)
        arrow(p,[(-s*.14,.35),(-s*.08,.21),(s*.03,.133),(s*.155,.105),(s*.35,.105),(s*.67,.105)],.166,.61,.305);p.a[YY>.66]=0
    if k in [38,39]:
        s=1 if k==38 else -1
        arrow(p,[(-s*.475,-.50),(s*.13,.11),(s*.525,.505)],.202,.87,.43)
    if k==40:
        for i in range(3):
            a=np.linspace(i*2*np.pi/3+1.0,i*2*np.pi/3-.55,16); pts=np.array([.55*np.cos(a),.55*np.sin(a)]).T
            tip=pts[-1]+.075*(pts[-1]-pts[-2])/np.linalg.norm(pts[-1]-pts[-2]);arrow(p,np.concatenate([pts,tip[None]],axis=0),.14,.42,.235)
    return p.a

# Single-stroke signs of the DIN road lettering: common height and two radii
# for the tighter and broader curves. Coordinates here are relative to a glyph.
def digit(d,style):
    p=Pen(XX+.5,YY+.5);w=[.15,.167,.137][style%3]; r=w/2; hi=1-r;lo=r
    if d=='0':
        # Straight sides with almost semicircular caps (DIN, not an ellipse).
        X=(p.x-.5)*.64;Y=p.y
        # Geometric outline instead of a contour fitted to a font.
        outer=(X/.32)**2+(np.maximum(abs(Y-.5)-.242,0)/.258)**2<=1
        inner=(X/(.32-w))**2+(np.maximum(abs(Y-.5)-.242,0)/(.258-w))**2<1
        p.shape(outer&~inner,1);return p.a
    paths={
      '1':[(.08,.235),(.28,.175),(.52,.075),(.58,lo),(.58,hi)],
      '2':[(.09,.25),(.13,.17),(.25,.096),(.42,lo),(.61,lo),(.78,.115),(.87,.19),(.89,.28),(.85,.37),(.74,.47),(.16,.835),(.115,hi),(.91,hi)],
      '3':[(.12,.20),(.23,.119),(.40,lo),(.62,lo),(.78,.115),(.855,.195),(.86,.29),(.81,.375),(.68,.442),(.52,.48),(.38,.48),(.55,.478),(.73,.525),(.857,.604),(.90,.695),(.88,.788),(.77,.873),(.62,hi),(.41,hi),(.24,.885),(.12,.80)],
      '5':[(.86,lo),(.213,lo),(.16,.464),(.27,.428),(.46,.395),(.67,.414),(.80,.475),(.88,.573),(.89,.730),(.85,.815),(.75,.88),(.61,hi),(.42,hi),(.25,.89),(.12,.803)],
      '6':[(.74,lo),(.46,.279),(.25,.458),(.15,.583),(.12,.69),(.15,.80),(.24,.875),(.39,hi),(.605,hi),(.75,.89),(.858,.807),(.89,.706),(.865,.599),(.79,.517),(.66,.467),(.48,.45),(.31,.47),(.19,.536)],
      '7':[(.09,lo),(.90,lo),(.90,.115),(.34,hi)],
      '8':[(.495,.484),(.314,.439),(.191,.36),(.147,.258),(.174,.177),(.264,.111),(.406,lo),(.60,lo),(.743,.11),(.837,.18),(.855,.271),(.821,.355),(.703,.43),(.495,.484),(.301,.53),(.183,.606),(.128,.696),(.149,.795),(.247,.878),(.397,hi),(.603,hi),(.752,.881),(.858,.805),(.884,.701),(.85,.611),(.708,.527),(.495,.484)]}
    if d=='6' and style%2:
        paths[d]=[(.81,.148),(.65,.08),(.48,lo),(.33,.109),(.21,.184),(.14,.29),(.119,.455),(.12,.69)]+paths[d][5:]
    if d=='3' and style%3==2:
        paths[d]=[(.15,lo),(.82,lo),(.79,.13),(.41,.442),(.535,.444)]+paths[d][10:]
    if d=='1' and style%3==2:paths[d]=[(.065,.241),(.52,lo),(.55,lo),(.55,hi)]
    # Render with a constant physical pen width although glyph coordinates
    # span a rectangle, not a square.
    p=Pen((XX+.5)*.63,YY+.5);p.line([(x*.63,y) for x,y in paths[d]],w)
    return p.a

def sample(src,x,y):
    # Linear resampling of a single hand drawing; uncovered area has no ink.
    u=(x+1.12)*((S-1)/2.24);v=(y+1.12)*((S-1)/2.24);valid=(u>=0)&(v>=0)&(u<S-1)&(v<S-1)
    i=np.clip(np.floor(u).astype(int),0,S-2);j=np.clip(np.floor(v).astype(int),0,S-2);a=u-i;b=v-j
    return valid*((1-a)*((1-b)*src[j,i]+b*src[j+1,i])+a*((1-b)*src[j,i+1]+b*src[j+1,i+1]))

DIG={(d,s):digit(d,s) for d in '01235678' for s in range(3)}
SPEED={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}
def number(text,style=0):
    # Centers/dimensions depend on typesetting, not a reference photograph.
    h=(.84 if len(text)==2 else .715);rat=[.565,.60,.535][style%3];gap=[.115,.096,.130][style%3]*h
    widths=[h*rat*(.66 if d=='1' else 1) for d in text];L=sum(widths)+gap*(len(text)-1);start=-L/2;img=np.zeros_like(XX)
    for d,w in zip(text,widths):
        img=np.maximum(img,sample(DIG[d,style%3],(XX-start)/w-.5,YY/h));start+=w+gap
    return img

def stopword():
    p=Pen();h=.42;w=.25;space=.062
    for idx,letter in enumerate('STOP'):
        x=(idx-1.5)*(w+space); f=lambda pts:[(x+a*w,b*h) for a,b in pts]
        if letter=='S':p.line(f([(.38,-.34),(.2,-.43),(-.16,-.43),(-.37,-.29),(-.38,-.14),(-.27,-.04),(.22,.09),(.38,.19),(.38,.31),(.20,.435),(-.18,.435),(-.39,.33)]),.057)
        if letter=='T':p.line(f([(-.39,-.43),(.39,-.43)]),.056);p.line(f([(0,-.43),(0,.43)]),.059)
        if letter=='O':p.ring(x,0,w*.39,h*.435,.058)
        if letter=='P':p.line(f([(-.34,.43),(-.34,-.43),(.14,-.43),(.35,-.31),(.35,-.13),(.17,.0),(-.32,0)]),.057)
    return p.a

# Fractional coverage of standard sign faces. Distances from the sides of a
# triangle are in the Euclidean metric, giving a uniform-width red border.
def tri_dist(x,y,down=False):
    if down:y=-y
    return np.minimum(.99-y,((y+1.)-2*np.abs(x))/np.sqrt(5.))
def sat(x):return np.clip(x,0,1)
def faces(k):
    if k==12:return 'diamond'
    if k==13:return 'yield'
    if k==14:return 'stop'
    if k==17:return 'entry'
    if k==11 or 18<=k<=31:return 'warning'
    if 33<=k<=40:return 'blue'
    if k in (6,32,41,42):return 'cancel'
    return 'ring'

def face_result(k,x,y,t=.0):
    # Output channels describe normalized red and green intensity and two
    # colour-opponent signals. Unknown scenery is explicitly masked out.
    rad=np.hypot(x,y); typ=faces(k);dist=1-rad
    if typ in ('warning','yield'):dist=tri_dist(x,y,typ=='yield')
    if typ=='diamond':dist=(1-abs(x)-abs(y))*.7071
    if typ=='stop':dist=np.minimum.reduce([.925-np.abs(x),.925-np.abs(y),.925-abs(x+y)*.7071,.925-abs(x-y)*.7071])
    mask=sat(dist/.06+.65)
    # Symbol-free centre contributes little to the family match for the
    # pictorial families. A uniform white face, however, matters for no vehicles.
    exterior=sat((dist+.155)/.13);interior=sat(dist/.08+.5)
    blend=lambda c0,c1,f:np.stack([a+(b-a)*f for a,b in zip(c0,c1)])
    white=(.94,.94,0,0);red=(.55,.105,1.,0);blue=(.10,.32,-.22,1.);yellow=(.99,.72,.12,-.65);grey=(.34,.34,0,0);black=(.09,.09,0,0)
    if typ in ['ring','warning','yield']:
        width=(.196 if typ=='ring' else (.145 if typ=='warning' else .165))+t*.027
        fill=sat((dist-width)*22+.5);cols=blend(red,white,fill)
        if k!=15 and typ!='yield':exterior*=1-.86*sat((dist-width-.11)/.15)
    elif typ=='diamond':cols=blend(white,yellow,sat((dist-.172)*21+.5))
    elif typ=='blue':
        cols=blend(white,blue,sat((dist-.033)*26+.5));exterior*=1-.82*sat((dist-.16)/.15)
    elif typ=='stop':cols=blend(white,red,sat((dist-.06)*22+.5));exterior*=1-.73*sat((dist-.17)/.16)
    elif typ=='entry':cols=blend(red,white,sat(np.minimum(.695-abs(x),.185-abs(y))*22+.5))
    else:
        stripe=sat((.257-np.abs((x+y)*.707))*18+.5);cols=blend(white,grey,stripe)
    cols=cols*mask[None]+np.array([.46,.46,0.,0.])[:,None,None]*(1-mask[None])
    return cols,exterior,mask

# The single-template normalization removes the DC level only within its
# support and caps the effect of small or intrinsically colourless channels.
def filtw(a,mask,cs,floor):
    u=(a-(a*mask).sum(axis=(-2,-1),keepdims=True)/(mask.sum()+1e-12))*mask
    norm=np.sqrt(np.sum(u*u,axis=(-2,-1),keepdims=True));base=math.sqrt(mask.sum()+1e-9)
    return (u/np.maximum(norm,base*np.array(floor).reshape(-1,1,1))*np.asarray(cs).reshape(-1,1,1)/base).astype('f')

def grid(N,H=2):
    u=(np.arange(N*H)+.5)/H-N/2
    return np.meshgrid(u,u)
def bins(a,N,H=2):return a.reshape(*a.shape[:-2],N,H,N,H).mean(axis=(-1,-3))
def seq(i,b):
    j=i;v=0.;f=1.
    while j:f/=b;v+=f*(j%b);j//=b
    return v

def pose(i):
    # Roughly 3/4 of the crop is sign; cover both small and large signs.
    if i==0:return (13.,13.,0.,0.,0.,0.)
    r=10.2+4.35*seq(i,2);aspect=(seq(i,3)-.5)*.18;dx=(seq(i,5)-.5)*4.6;dy=(seq(i,7)-.5)*4.6
    return r*(1+aspect),r*(1-aspect),dx,dy,(seq(i,11)-.5)*.18,(seq(i,13)-.5)*.095

def coords(x,y,params):
    rx,ry,dx,dy,ang,sk=params
    u=(x-dx)/rx;v=(y-dy)/ry; ca=math.cos(ang);sa=math.sin(ang);return u*ca+v*(sa+sk),v*ca-u*sa

# Input contrast compression: differences between signs and sunlit background
# can be orders of magnitude larger than strokes in shadow. Piecewise-linear
# approximations to log(I+epsilon) keep shadow detail in all three channels.
def preprocess():
    knots=np.array([0,.012,.03,.06,.11,.20,.36,.62,1.0]);slopes=np.diff(np.log(knots+.024))/np.diff(knots);m=len(slopes)
    a=np.repeat(np.eye(3),m,axis=0);b=-np.tile(knots[:-1],3);x=op('log_hinges','relu',mix('exposure_knots','image',a,b));g=np.zeros((3,3*m))
    for c in range(3):g[c,c*m:c*m+m]=np.diff(np.r_[0,slopes]);
    x=mix('log_rgb',x,g);features=np.array([[.86,.105,.035],[.12,.75,.13],[1.,-.71,-.29],[-.68,-.32,1.]])
    raw=mix('pigments',x,features);norm=op('frame_contrast','norm',raw)
    # Static, offset-free colour evidence complements normalized opponents.
    # Saturate the normalized channels to avoid a single outdoor highlight.
    pos=op('bright_tails','relu',mix('above_clip',norm,np.eye(4),-2.0));neg=op('dark_tails','relu',mix('below_clip',norm,-np.eye(4),-2.0))
    trim=op('trim_delta','add',[neg,mix('invert_tails',pos,-np.eye(4))]);clip=op('bounded_contrast','add',[norm,trim]);
    return raw,clip

def peakbank(name,source,bank):
    # Order: one contiguous list of geometric alternatives for each category.
    a=np.array(bank,dtype='f');n,m=a.shape[:2];x=dense(name+'_filters',source,a.reshape(n*m,-1));x=rs(name+'_map',x,[n,1,m]);x=pool(name+'_best','maxpool',x,1,m);return rs(name+'_flat',x,[n])

def route(name,x,ids,gains=1):
    w=np.zeros((43,len(ids)))
    for j,k in enumerate(ids):w[k,j]=np.broadcast_to(gains,(len(ids),))[j]
    return dense(name,x,w)

def locinput(n,i,c,N):
    # Two successive valid delta convolutions crop both axes. Splitting the
    # kernel bounds the temporary memory even with 500 test images at once.
    t=i;offset=(32-N)//2
    for j,r in enumerate([offset//2,offset-offset//2]):
        w=np.zeros((c,c,r*2+1,r*2+1));w[np.arange(c),np.arange(c),r,r]=1
        t=conv(n+'_outer' if j==0 else n,t,w)
    return t

def make_frame(clip,raw):
    # Pool the large, easy-to-see border before comparing family hypotheses.
    av=pool('paint_at_half_scale','avgpool',clip,2,2);ids=[0,11,12,13,14,15,17,32,35];reps=[0 if faces(k)=='ring' and k!=15 else 11 if faces(k)=='warning' else 32 if faces(k)=='cancel' else 35 if faces(k)=='blue' else k for k in range(43)]
    xs,ys=grid(16,3); bank=[]
    for k in ids:
        filters=[]
        for i in range(106):
            params=pose(i);x,y=coords(xs*2,ys*2,params);a,m,_=face_result(k,x,y,2*seq(i,17)-1)
            a=bins(a,16,3);m=bins(m,16,3)
            # Fixed scales for luminance/colour evidence across all families.
            # Achromatic signs use intensity in place of the red/blue detectors.
            typ=faces(k);cs=([.46,.46,.0,.0] if typ=='cancel' else [.21,.46,.33,0] if typ in ['ring','warning','yield','entry','stop'] else [.46,.23,0,.31] if typ=='blue' else [.22,.25,.07,.46]);
            filters.append(filtw(a,m,cs,[.085,.085,.14,.14]))
        bank.append(filters)
    scores=peakbank('face',av,bank);w=np.eye(len(ids))[[ids.index(k) for k in reps]]
    return dense('face_to_class',scores,w)

# Content maps: normalized intensity of the central window, and optionally its
# high-pass detail. The red channel suppresses red rims, improving the visibility
# of black strokes on white, while blue arrows are brightest against log red.
def content(raw):
    crop=locinput('content_window',raw,4,20); lp=gauss(4,2.35);hp=-lp;hp[4,4]+=1
    # Apply high-pass on the full crop, so padding never enters the central ROI.
    h=conv('remove_slow_light',raw,np.eye(4)[:,:,None,None]*hp,p=4)
    hc=locinput('detail_window',h,4,20)
    a=op('ink_contrast','norm',crop);b=op('detail_contrast','norm',hc)
    # Signed projections, with a small additional colour-opponent cue for
    # distinguishing red cars from black trucks. This branch uses no drawings.
    a=mix('local_intensity',a,[[.69,.31,0,0],[0,0,1,0]]);b=mix('local_detail',b,[[.69,.31,0,0],[0,0,1,0]])
    f=op('intensity_with_strokes','add',[mix('intensity_part',a,np.eye(2)*.47),mix('detail_part',b,np.eye(2)*.53)])
    # A clipped linear function approximates a robust (outlier-insensitive)
    # matched filter when, for instance, a few sky pixels enter the window.
    hi=op('ink_upper_tail','relu',mix('ink_above_bound',f,np.eye(2),-2.65));lo=op('ink_lower_tail','relu',mix('ink_below_bound',f,-np.eye(2),-2.65));return op('pigment_and_stroke','add',[f,lo,mix('subtract_ink_tail',hi,-np.eye(2))])

all_icons={}
for k in list(range(11,12))+list(range(18,32)):
    for st in range(2):all_icons[k,st]=warning(k,st)
for k in [9,10,16,41,42]+list(range(33,41)):
    for st in range(2):all_icons[k,st]=regulatory(k,st)
for k,t in SPEED.items():
    for st in range(3):all_icons[k,st]=number(t,st)
all_icons[14,0]=stopword()

def iconbank(feat):
    N=20;xs,ys=grid(N); bank=[];nposes=172;ids=[k for k in range(43) if (k,0) in all_icons]
    for k in ids:
        typ=faces(k);filters=[]
        for i in range(nposes):
            p=pose(i);x,y=coords(xs,ys,p);st=(i//3)%(3 if k in SPEED else 1 if k==14 else 2); icon=all_icons[k,st]
            # Icon placement and lettering size have independent manufacturing
            # variation as well as camera scale and perspective.
            sx=1+(seq(i,17)-.5)*(.32 if k in SPEED else .27) if i else 1;sy=1+(seq(i,19)-.5)*.32 if i else 1;shift=(seq(i,23)-.5)*.105 if i else 0
            u=x/sx;v=(y-shift)/sy;ink=sample(icon,u,v); intensity=ink if typ in ('blue','stop') else -ink
            if k in [6,41,42]:
                stripe=sat((.245-np.abs((x+y)*.707))*18+.5);intensity=np.minimum(-ink*.83,-stripe*.83)
            if typ=='warning':mask=sat((tri_dist(x,y)-.142)/.063)*sat((.81-y)/.08)
            elif typ=='stop':mask=sat((.78-abs(x))/.13)*sat((.39-abs(y))/.105)
            elif k in SPEED and k!=6:mask=sat((.73-np.hypot(x*.97,y))/.10)*sat((.61-abs(y))/.10)
            else:mask=sat(((.82 if typ=='blue' else .76)-np.hypot(x,y))/.1)
            if k in [9,10,41,42]:mask*=sat((.48-abs(y))/.10) if k not in [41,42] else 1
            col=np.zeros_like(ink)
            if k in [9,10]:col=ink*sat((-u+.05)/.10);col-=.10*ink*sat(u/.1)
            if k==26:col=sat(1-((u/.115)**2+((v+.22)/.12)**2));col-=.45*sat(1-((u/.115)**2+((v-.41)/.12)**2))
            a=bins(np.stack([intensity,col]),N);mask=bins(mask,N);a=blur(a,.5+.45*seq(i,29));
            filters.append(filtw(a,mask,[.89 if k in [9,10] else .966 if k==26 else 1.,.155 if k in [9,10] else .075 if k==26 else 0.],[.112,.1]))
        bank.append(filters)
    return route('pictograms_to_class',peakbank('pictogram',feat,bank),ids)

# Separate OCR of the FIRST distinguishing digit. The complete inscription
# still has to match in the icon stream, so an isolated 0 or straight line in
# another sign cannot dominate. Evaluate the second digit of 100 versus 120.
def digitbank(feat):
    xs,ys=grid(20);ids=[0,1,2,3,4,5,7,8];bank=[]
    for k in ids:
        text=SPEED[k];target=text[0] if len(text)==2 else text[1];filters=[]
        for i in range(224):
            x,y=coords(xs,ys,pose(i));st=(i//4)%3
            h=(.84 if len(text)==2 else .715)*(1+(.34*(seq(i,17)-.5) if i else 0)); w=h*[.565,.60,.535][st]*(1+(.30*(seq(i,19)-.5) if i else 0)); gap=h*[.115,.096,.13][st]
            cx=-(w+gap)/2 if len(text)==2 else -(.34*w)/2;cy=.11*(seq(i,23)-.5) if i else 0
            u=(x-cx)/w;v=(y-cy)/h;ink=sample(DIG[target,st],u,v)
            m=sat((.5+gap/w*.78-abs(u))*5.5)*sat((.69-abs(v))*5.3)
            a=blur(bins(-ink,20),.46+.36*seq(i,29));mask=bins(m,20)
            filters.append(filtw(a[None],mask,[1.],[.09]))
        bank.append(filters)
    x=mix('number_ink',feat,[[1,0]])
    return route('digit_to_class',peakbank('numeral',x,bank),ids)

# A second reader retains the pose index until face and pictogram have been
# combined. This rules out, for example, reading an arm of the triangular
# border as an arrow at an inconsistent scale. The unconstrained streams are
# also retained because real installations vary in their pictogram layout.
def joint_parts(clip,feat):
    X,Y=grid(16,3); U,V=grid(20); bank=[];mapping=[];ic=[];n=320
    reps=[0,11,12,13,14,15,17,32,35]
    familyindex=lambda k: reps.index(0 if faces(k)=='ring' and k!=15 else 11 if faces(k)=='warning' else 32 if faces(k)=='cancel' else 35 if faces(k)=='blue' else k)
    for k in reps:
        pp=[];typ=faces(k)
        for i in range(n):
            x,y=coords(X*2,Y*2,pose(i));a,m,_=face_result(k,x,y,(seq(i,17)-.5)*2);m=bins(m,16,3);a=bins(a,16,3)
            cs=([.46,.46,0,0] if typ=='cancel' else [.21,.46,.33,0] if typ in ['ring','warning','yield','entry','stop'] else [.46,.23,0,.31] if typ=='blue' else [.22,.25,.07,.46]);pp.append(filtw(a,m,cs,[.085,.085,.14,.14]).reshape(-1))
        bank.extend(pp)
    border=dense('aligned_borders', 'paint_at_half_scale',np.asarray(bank));border=rs('aligned_families',border,[len(reps),1,n]);wi=np.eye(len(reps))[[familyindex(k) for k in range(43)]];border=mix('aligned_class_border',border,wi)
    for k in range(43):
        typ=faces(k);stmax=3 if k in SPEED else 1 if k==14 else 2
        for i in range(n):
            x,y=coords(U,V,pose(i));st=(i//5)%stmax
            sx=1+(seq(i,19)-.5)*.27 if i else 1;sy=1+(seq(i,23)-.5)*.28 if i else 1;dy=(seq(i,29)-.5)*.13 if i else 0
            u=x/sx;v=(y-dy)/sy
            if k in (12,13,15,17):
                # No separately printed glyph. The border stream for these
                # categories already represents the complete sign face.
                ic.append(np.zeros((2,20,20),'f'));continue
            ink=sample(all_icons[k,st],u,v) if k!=32 else np.zeros_like(u)
            intensity=ink if typ in ['blue','stop'] else -ink
            if typ=='cancel':
                ds=(x+y)*.707;stripe=sat((.245-abs(ds))*22+.5)
                # Distinct narrow oblique strokes coalesce into a grey band in
                # small blurry crops. Overprints leave the central motif visible.
                stripes=(.70+.30*np.cos(ds*65+.2))*stripe;intensity=-np.maximum(.76*ink,.86*stripes)
            if typ=='warning':m=sat((tri_dist(x,y)-.135)/.085)*sat((.85-y)/.10)
            elif k==14:m=sat((.78-abs(x))/.14)*sat((.42-abs(y))/.14)
            else:m=sat((.79-np.hypot(x,y))/.10)
            c=np.zeros_like(intensity)
            if k in [9,10]:c=ink*sat((-u+.06)/.10)-ink*.10*sat(u/.10)
            if k in SPEED and k!=6:m*=sat((.64-abs(y))/.14)
            a=blur(bins(np.stack([intensity,c]),20),.53+.32*seq(i,31));m=bins(m,20)
            ic.append(filtw(a,m,[.89 if k in [9,10] else 1,.15 if k in [9,10] else 0],[.115,.10]))
    symbol=dense('aligned_glyphs',feat,np.array(ic).reshape(43*n,-1));symbol=rs('aligned_class_glyph',symbol,[43,1,n]);
    # A single choice of geometry must serve both cues.
    v=np.ones(43)*.84;v[32]=.54
    symbol=mix('aligned_ink_weight',symbol,np.diag(v));border=mix('aligned_edge_weight',border,np.eye(43)*1.35)
    both=op('same_pose','add',[border,symbol]);return rs('aligned_class',pool('best_joint_pose','maxpool',both,1,n),[43])

def wordbank(feat):
    # Exhaustive subpixel search in a small window for the eight possible
    # numerals. Kernels slide rather than duplicating translated drawings. A
    # match to the whole inscription and one to the distinguishing digit must
    # agree on character size, baseline, yaw and translation.
    X,Y=grid(16); ids=[0,1,2,3,4,5,7,8];n=240; ww=[]
    for k in ids:
        text=SPEED[k]; idx=0 if len(text)==2 else 1
        for i in range(n):
            st=i%3;h=(6.75+5.65*seq(i+1,2))*(.865 if len(text)==3 else 1); w=h*(.44+.225*seq(i+1,3));gap=h*(.055+.14*seq(i+1,5));
            theta=(seq(i+1,7)-.5)*.20;sy=(seq(i+1,11)-.5)*.96;sx=(seq(i+1,13)-.5)*.96
            xx=(X-sx)*math.cos(theta)+(Y-sy)*math.sin(theta);yy=(Y-sy)*math.cos(theta)-(X-sx)*math.sin(theta);xx+=yy*(seq(i+1,17)-.5)*.18
            widths=[w*(.66 if d=='1' else 1) for d in text];total=sum(widths)+(len(text)-1)*gap;left=-total*.5;ink=np.zeros_like(xx)
            for j,(d,wd) in enumerate(zip(text,widths)):
                cx=left+.5*wd;glyph=sample(DIG[d,st if j==idx else (st+1)%3],(xx-cx)/wd,yy/h);ink=np.maximum(ink,glyph);left+=wd+gap
                if j==idx:tc=cx;tw=wd
            # Target mask includes just enough paper to test the stroke edges.
            # Sums within one image only; no measured gains or global statistics.
            m1=sat((total*.5+1.08-abs(xx))/.85)*sat((h*.5+1.16-abs(yy))/.85)
            m2=sat((tw*.5+.67*gap+.25-abs(xx-tc))/.69)*sat((h*.5+1.1-abs(yy))/.78)
            a=blur(bins(-ink,16),.37+.50*seq(i+1,19))[None];mask1=bins(m1,16);mask2=bins(m2,16)
            ww.append(.53*filtw(a,mask1,[1],[.09])+.47*filtw(a,mask2,[1],[.09]))
    ink=mix('word_channel',feat,[[1,0]]);z=conv('typeset_readers',ink,np.array(ww));z=pool('best_word_position','maxpool',z,5,5);z=rs('word_sizes',z,[len(ids),1,n]);z=rs('word_match',pool('best_word_size','maxpool',z,1,n),[len(ids)]);return route('word_to_class',z,ids)

def cancellation_detail(feat):
    # The common diagonal band can provide most of the match in this family.
    # Read what's left on its two sides separately. In the absence of markings,
    # the simpler 'all restrictions end' model is preferred.
    X,Y=grid(20); ids=[6,41,42];rr=[]
    for k in ids:
        ff=[]
        for i in range(260):
            x,y=coords(X,Y,pose(i));sx=1+(seq(i,19)-.5)*.25;sy=1+(seq(i,23)-.5)*.22
            ink=sample(all_icons[k,i%(3 if k==6 else 2)],x/sx,y/sy);d=np.abs((x+y)*.7071);band=.22+.09*seq(i,17)
            m=sat((d-band)/.095)*sat((.76-np.hypot(x,y))/.105); m*=sat((.49-abs(y))/.12)
            a=blur(bins(-ink,20),.56+.20*seq(i,7))[None];ff.append(filtw(a,bins(m,20),[1],[.125]))
        rr.append(ff)
    ink=mix('overprint_channel',feat,[[1,0]]);z=peakbank('unstriped_symbol',ink,rr);w=np.zeros((43,3));w[ids,range(3)]=1
    # Only differential evidence: this stream isn't used to identify the
    # sign family. The free halo around a plain stripe is tested elsewhere.
    return dense('overprint_to_class',z,w)

# Absolute colour and local texture tests cover homogeneous centers where a
# zero-mean pictogram template isn't meaningful (e.g. no vehicles).
def supporting(raw,feat):
    x,y=grid(32,1);R=np.hypot(x,y);ga=lambda cx,cy,rx,ry:np.exp(-.5*((x-cx)**2/rx**2+(y-cy)**2/ry**2));
    def unit(m):return m/(m.sum()+1e-15)
    w=[]; groups=['ring','warning','diamond','yield','stop','entry','cancel','blue']
    for g in groups:
        ww=np.zeros((4,32,32));r=12.45
        if g=='ring':s=np.exp(-.5*((R/r-.88)/.139)**2);ww[2]=unit(s)*.40-unit(np.exp(-.5*(R/(r*.46))**2))*.16
        elif g in ('warning','yield'):
            d=tri_dist(x/r,y/r,g=='yield');s=np.exp(-.5*((d-.085)/.125)**2);ww[2]=unit(s)*.39-unit(sat((d-.185)/.19))*.16
        elif g=='stop':ww[2]=unit(sat((.78-R/r)/.23))*.44
        elif g=='entry':ww[2]=unit(sat((.86-R/r)/.16)*sat((abs(y/r)-.17)/.13))*.46-unit(ga(0,0,5,1.3))*.14
        elif g=='blue':ww[3]=unit(sat((.83-R/r)/.19))*.39;ww[2]=-unit(sat((.83-R/r)/.19))*.16
        elif g=='diamond':ww[3]=-unit(sat((.68-abs(x/r)-abs(y/r))/.22))*.42
        w.append(ww)
    c=dense('absolute_paint',raw,np.array(w).reshape(len(groups),-1));c=op('paint_evidence','relu',c); cmax=op('strong_paint','relu',dense('paint_limit',c,np.eye(len(groups)),-.41)); c=op('bound_paint','add',[c,dense('remove_excess',cmax,-np.eye(len(groups)))]);
    color=dense('paint_to_class',c,np.eye(len(groups))[[groups.index(faces(k)) for k in range(43)]])
    # Central activity (absolute high-frequency signal, with no fitted bias).
    # 1x1 rectification distinguishes a smooth white field from printed ink.
    hp=gauss(3,1.55);hp=-hp;hp[3,3]+=1
    # use log-scale detail rather than dividing by the tiny variance of a blank sign
    d=conv('fine_darkness',raw,np.stack([-.72*hp,-.28*hp,0*hp,0*hp])[None],p=3);d=op('dark_marks','relu',d);u=unit(ga(0,0,4.30,3.85));dw=np.zeros((43,32,32))
    for k in range(43):
        if k in [13,15]:dw[k]=-u*3.6
        if k==32:dw[k]=-u*.40
    texture=dense('test_clear_field',d,dw.reshape(43,-1))
    return color,texture

def build(folder):
    raw,clip=preprocess();base=make_frame(clip,raw);feat=content(raw);pic=iconbank(feat);num=digitbank(feat);joint=joint_parts(clip,feat);word=wordbank(feat);over=cancellation_detail(feat);col,tex=supporting(raw,feat)
    # Weights in the decision rule express the role of each component.
    # Centered correlation with a blank pictogram is undefined, so clear-face
    # classes have a small fixed allowance instead, subject to their ink test.
    pa=np.ones(43)*.86;pa[[0,1,2,3,4,5,7,8]]=.61
    fa=np.ones(43)*1.32;fa[[12,13,15,17,32]]=1.43
    bias=np.zeros(43);bias[[12,13,15,17,32]]=[.21,.37,.35,.23,.21]
    # A blank center on the no-vehicles/yield signs should beat a weak chance
    # positive icon response, but not a clearly resolved printed symbol.
    bias[15]+=.08; bias[13]+=.045
    bias[32]=.51; bias[[6,41,42]]=-.09
    # Use the accurate aligned word detector as an extra within-speed vote;
    # subtract the mean *score at inference* over speed labels so it is not
    # an incentive to hallucinate a speed sign on a plain surface.
    ids=[0,1,2,3,4,5,7,8];vote=np.zeros((43,43));vote[np.ix_(ids,ids)]=np.eye(8)-1./8
    terms=[dense('outline_weight',base,np.diag(fa)*.46,bias),dense('symbol_weight',pic,np.diag(pa)*.46),dense('digit_weight',num,np.eye(43)*.251*.46),dense('coherent_weight',joint,np.eye(43)*.54),dense('read_word',word,vote*.65),dense('read_overprint',over,np.eye(43)*.33),dense('colour_weight',col,np.eye(43)*.83),tex];op('scores','add',terms)
    folder.mkdir(parents=True,exist_ok=True);(folder/'network.json').write_text(json.dumps({'layers':LAY},separators=(',',':'))+'\n');np.savez(folder/'weights.npz',**WE)

if __name__=='__main__':build(Path(sys.argv[1]))
