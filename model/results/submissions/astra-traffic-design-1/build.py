"""A constructive sign grammar and a bank of analytic, pose-indexed comparators.
No assets, fonts, example images, or learned parameters are used.
"""
import numpy as np
import json, sys
from pathlib import Path
from functools import lru_cache

# The coordinate system on a sign runs from -1 to 1. Positive y is down.
N=160
a=(np.arange(N)+.5-N/2)/(N/2.4)
X,Y=np.meshgrid(a,a)
def oval(x,y,rx,ry,cx=0,cy=0): return ((x-cx)/rx)**2+((y-cy)/ry)**2<1
def poly(x,y,pts):
    inside=np.zeros(np.broadcast(x,y).shape,bool)
    for (a,b),(c,d) in zip(pts,pts[1:]+pts[:1]):
        if b!=d: inside ^= ((y>b)!=(y>d)) & (x<(c-a)*(y-b)/(d-b)+a)
    return inside
def stroke(x,y,pts,w):
    m=np.zeros(np.broadcast(x,y).shape,bool)
    for (a,b),(c,d) in zip(pts,pts[1:]):
        dx,dy=c-a,d-b; t=np.clip(((x-a)*dx+(y-b)*dy)/max(dx*dx+dy*dy,1e-10),0,1)
        m|=(x-a-dx*t)**2+(y-b-dy*t)**2<(w/2)**2
    return m
def line(pts,w): return stroke(X,Y,pts,w)
def P(*pts): return poly(X,Y,list(pts))
def O(cx,cy,rx,ry): return oval(X,Y,rx,ry,cx,cy)
def curve(p0,p1,p2,p3):
    return [tuple((1-t)**3*a+3*t*(1-t)**2*b+3*t*t*(1-t)*c+t**3*d for a,b,c,d in zip(p0,p1,p2,p3)) for t in np.linspace(0,1,13)]
def chain(*cs): return sum([curve(*c) for c in cs],[])
def ellipsepts(a,b,rx,ry): return [(a+rx*np.cos(t),b+ry*np.sin(t)) for t in np.linspace(0,2*np.pi,35)]

def digit(ch,x,y,width,height,thick):
    # A constructed, largely monoline DIN-style road numeral. The two bars of
    # the condensed zeros are straight, rather than an elliptical computer font.
    dx=width/2; dy=height/2; r=thick/2; w=dx-r; h=dy-r
    if ch=='0':
        top=curve((-w,-h*.51),(-w,-h*1.16),(w,-h*1.16),(w,-h*.51))
        bot=curve((w,h*.51),(w,h*1.16),(-w,h*1.16),(-w,h*.51))
        pts=top+bot+[top[0]]
    elif ch=='1': pts=[(-w,-h*.64),(w*.46,-h),(w*.46,h)]
    elif ch=='C':
        pts=chain(((w*.86,-.72*h),(.38*w,-1.18*h),(-.76*w,-1.04*h),(-.89*w,-.67*h)),((-.89*w,-.67*h),(-1.03*w,-.18*h),(-.99*w,.46*h),(-.88*w,.66*h)))+ellipsepts(0,.42*h,w,.58*h)[17:]+ellipsepts(0,.42*h,w,.58*h)[:18]
    elif ch=='B':
        pts=chain(((-w,-.66*h),(-.30*w,-1.25*h),(w,-1.03*h),(.90*w,-.55*h)),((.90*w,-.55*h),(.91*w,-.24*h),(.43*w,-.06*h),(-.17*w,-.06*h)),((-.17*w,-.06*h),(1.20*w,-.17*h),(1.50*w,.67*h),(.41*w,.93*h)),((.41*w,.93*h),(-.29*w,1.13*h),(-.90*w,.86*h),(-w,.66*h)))
    elif ch=='2':
        pts=chain(((-w,-.52*h),(-w,-1.17*h),(w,-1.19*h),(w,-.52*h)),((w,-.52*h),(w,-.16*h),(-w,.66*h),(-w,h)))+[(w,h)]
    elif ch=='3':
        pts=[(-w*.87,-h),(.86*w,-h),(-.23*w,-.10*h)]+chain(((-.23*w,-.10*h),(.91*w,-.30*h),(1.26*w,.26*h),(.88*w,.63*h)),((.88*w,.63*h),(.66*w,1.13*h),(-.70*w,1.12*h),(-w,.66*h)))
    elif ch=='5':
        pts=[(w*.9,-h),(-w*.83,-h),(-w*.95,-.06*h)]+chain(((-w*.95,-.06*h),(-.28*w,-.41*h),(w,-.34*h),(w,.25*h)),((w,.25*h),(1.18*w,1.22*h),(-.61*w,1.17*h),(-w,.67*h)))
    elif ch=='6':
        pts=chain(((.55*w,-h),(.15*w,-.65*h),(-1.02*w,.0),(-w,.37*h)),((-w,.37*h),(-w,1.20*h),(w,1.22*h),(w,.39*h)),((w,.39*h),(1.12*w,-.28*h),(-.43*w,-.41*h),(-.91*w,.16*h)))
    elif ch=='7': pts=[(-w,-h),(w,-h),(-.48*w,h)]
    elif ch=='8': pts=ellipsepts(0,-.53*h,.90*w,.47*h)+[]; return stroke(x,y,pts,thick)|stroke(x,y,ellipsepts(0,.47*h,w,.53*h),thick)
    else: raise ValueError(ch)
    return stroke(x,y,pts,thick)

def numbers(text,variant):
    # Alternate standard-width and condensed type. The 1 occupies a narrower advance.
    height=[.84,.76,.91,.81][variant%4] if len(text)==2 else [.78,.73,.86,.81][variant%4]
    width=height*([.55,.56,.50,.61][variant%4] if len(text)==2 else [.52,.57,.455,.50][variant%4]); thick=height*(.112 if len(text)==2 else [.112,.126,.093,.108][variant%4])
    if variant%4==2: thick*=1.09
    widths=[width*(.55 if c=='1' else 1) for c in text]; gap=height*(.095 if len(text)==2 else .065)
    pos=-(sum(widths)+gap*(len(text)-1))/2
    m=np.zeros_like(X,bool)
    for c,w in zip(text,widths):
        c=({'3':'B','6':'C'}.get(c,c) if variant%4 in (1,3) else c)
        m|=digit(c,X-pos-w/2,Y,w,height,thick); pos+=w+gap
    return m

def car(cx,cy,s):
    x=(X-cx)/s; y=(Y-cy)/s
    return poly(x,y,[(-.96,.38),(-.96,-.11),(-.78,-.27),(-.57,-.72),(.59,-.72),(.80,-.28),(.96,-.12),(.96,.38),(.82,.42),(.82,.70),(.53,.70),(.51,.42),(-.54,.42),(-.55,.70),(-.83,.70),(-.84,.41)])
def reartruck(cx,cy,s):
    x=(X-cx)/s; y=(Y-cy)/s
    return ((abs(x)<.95)&(abs(y+.12)<1.06)) | ((abs(x)>.49)&(abs(x)<.86)&(y>.9)&(y<1.17))
def lorry():
    return P((-.65,.19),(-.65,-.01),(-.49,-.21),(-.24,-.21),(-.24,-.33),(.63,-.33),(.63,.15),(.49,.15),(.46,.24),(-.53,.24)) | O(-.43,.20,.13,.13) | O(.39,.20,.13,.13)

def stopword():
    m=np.zeros_like(X,bool)
    for j,c in enumerate('STOP'):
        x=(X-(j-1.5)*.365)/.128; y=Y/.325
        if c=='S': pt=chain(((.83,-.71),(.33,-1.26),(-1.03,-1.1),(-.92,-.45)),((-.92,-.45),(-.86,.01),(.99,-.03),(.90,.55)),((.90,.55),(.82,1.20),(-.68,1.19),(-.94,.71)))
        elif c=='T': m|=stroke(x,y,[(-.96,-.94),(.96,-.94)],.30); pt=[(0,-.90),(0,.97)]
        elif c=='O': pt=curve((-.85,-.51),(-.85,-1.15),(.85,-1.15),(.85,-.51))+curve((.85,.51),(.85,1.15),(-.85,1.15),(-.85,.51))+[(-.85,-.51)]
        else: pt=[(-.76,.95),(-.76,-.91)]+chain(((-.76,-.91),(1.47,-1.35),(1.47,.14),(-.76,.04)))
        # use metric stroke widths, including square upright strokes
        m|=stroke(X-(j-1.5)*.365,Y,[(a*.128,b*.325) for a,b in pt],.09)
    return m

def arrow(k,v=0):
    x=-X if k in [34,37,39] else X
    y=Y
    def p(pts):return poly(x,y,pts)
    def l(pts,w):return stroke(x,y,pts,w)
    if k in (33,34):
        return p([(-.36,.68),(-.17,.68),(-.17,.12),(-.36,.12)])|l([(-.265,.22),(-.265,.10)]+curve((-.265,.10),(-.265,-.23),(-.045,-.31),(.275,-.31)),.19)|p([(.12,-.64),(.63,-.31),(.12,.025),(.18,-.205),(.06,-.205),(.06,-.415),(.18,-.415)])
    if k in (35,36,37):
        a=-.15 if k!=35 else 0
        m=p([(a-.096,.68),(a-.096,-.31),(a-.31,-.195),(a,-.77),(a+.31,-.195),(a+.096,-.31),(a+.096,.68)])
        if k!=35: m|=l([(a,.42)]+curve((a,.42),(a,.047),(.035,.045),(.34,.045)),.17)|p([(.22,-.24),(.72,.045),(.22,.33),(.29,.13),(.19,.13),(.19,-.045),(.29,-.045)])
        return m
    if k in (38,39):
        # a broad diagonal arrow with its flat-cut tail at upper left
        u=(x+y)*.7071; vv=(x-y)*.7071
        return poly(u,vv,[(-.68,-.14),(.12,-.14),(.06,-.44),(.73,0),(.06,.44),(.12,.14),(-.68,.14)])
    m=np.zeros_like(X,bool)
    for i in range(3):
        th=i*2*np.pi/3+[0,.35,-.35,.65][v%4]
        u=-x*np.cos(th)+y*np.sin(th); vv=x*np.sin(th)+y*np.cos(th)
        pts=[(.60*np.cos(t),.60*np.sin(t)) for t in np.linspace(-2.81,-1.41,15)]
        m |= stroke(u,vv,pts,.14)|poly(u,vv,[(.04,-.80),(.37,-.53),(.01,-.35)])
    return m

def person(x,y,h,walk=1,flip=1):
    m=O(x+.045*h*flip,y+.10*h,.075*h,.083*h)
    def L(pts,w):return line([(x+a*h*flip,y+b*h) for a,b in pts],w*h)
    m|=L([(.018,.24),(-.060,.53)],.15)|L([(-.002,.27),(.092,.41),(.22,.43)],.072)
    m|=L([(-.035,.27),(-.16,.38),(-.22,.50)],.065)|L([(-.05,.52),(.10,.71),(.22,.94)],.095)|L([(-.06,.52),(-.15,.74),(-.28,.91)],.095)
    return m

def warn(k,v=0):
    # Reserve clear space between a warning symbol and the sloping red sides.
    # In particular, the lane strokes converge *inside* the white triangle.
    if k==11:
        return P((-.14,.56),(-.14,.30),(-.43,.30),(-.14,-.07),(-.14,-.19),(0,-.40),(.14,-.19),(.14,-.07),(.43,.30),(.14,.30),(.14,.56))
    if k==18:
        return P((-.084,-.38),(.084,-.38),(.048,.25),(-.048,.25))|O(0,.45,.078,.088)
    if k in [19,20]:
        sg=1 if k==19 else -1
        pts=[(.27,.55),(.27,.32)]+curve((.27,.32),(.27,.10),(.14,-.054),(-.077,-.065))
        mark=stroke(sg*X,Y,pts,.18)&(Y<=.55)
        if v%2==0:mark|=poly(sg*X,Y,[(-.07,-.24),(-.23,-.065),(-.062,.095)])
        else:mark|=poly(sg*X,Y,[(-.18,-.156),(-.062,-.155),(-.062,.025),(-.18,.025)])
        return mark
    if k==21:
        if v%2==0:return P((-.25,-.30),(-.073,-.30),(-.073,.07),(.30,-.045),(.30,.56),(.13,.56),(.13,.21),(-.25,.32))
        return line([(.28,.53),(.28,.15)]+curve((.28,.15),(.28,-.08),(-.14,.22),(-.14,-.01))+[(-.14,-.22)],.17)|P((-.23,-.12),(-.14,-.33),(-.035,-.12))
    if k==22:
        return (P((-.54,.50),(-.51,.46),(.50,.46),(.54,.50))|O(-.25,.47,.24,.175)|O(.18,.47,.24,.20))&(Y<=.51)
    if k==23:
        # Front view of a skidding car, wheels and windshield visible.
        th=[.27,.42,.12,.33][v%4]; u=(X-.015)*np.cos(th)+(Y+.016)*np.sin(th);vv=-(X-.015)*np.sin(th)+(Y+.016)*np.cos(th)
        m=poly(u,vv,[(-.31,.12),(-.31,-.035),(-.25,-.07),(-.185,-.21),(.16,-.21),(.24,-.07),(.29,-.035),(.29,.12),(.24,.12),(.22,.18),(.15,.18),(.15,.12),(-.16,.12),(-.17,.18),(-.24,.18),(-.25,.12)])
        m &= ~poly(u,vv,[(-.19,-.065),(-.14,-.164),(.124,-.164),(.176,-.065)])
        for d in [-.13,.12]:
            m |= line([(d+.06,.175),(d+.15,.24),(d+.12,.31),(d-.09,.36),(d-.13,.43),(d-.02,.51)],.045)
        return m
    if k==24:
        return line([(-.19,.60),(-.19,-.27)],.082)|line([(.255,.60),(.255,.24),(.085,-.07),(.085,-.27)],.082)
    if k==25:
        if v%2==0:
            return O(.07,-.195,.087,.092)|line([(-.055,-.125),(-.19,-.055),(-.29,.13)],.15)|line([(-.29,.13),(-.25,.29),(-.44,.53)],.090)|line([(-.26,.115),(-.12,.26),(-.18,.53)],.087)|line([(-.06,-.10),(.05,.055),(.008,.14)],.052)|line([(-.19,-.013),(.28,.36)],.038)|P((.002,.55),(.24,.27),(.35,.245),(.46,.38),(.56,.55))
        return O(-.018,-.20,.093,.10)|line([(-.06,-.08),(-.17,.12)],.15)|line([(-.17,.12),(-.21,.29),(-.36,.50)],.09)|line([(-.14,.14),(-.04,.28),(-.084,.50)],.09)|line([(0,-.055),(.10,.11),(.018,.21)],.065)|line([(-.28,.02),(.34,.39)],.04)|P((.06,.51),(.26,.29),(.37,.27),(.50,.42),(.57,.51))
    if k==26:
        return O(0,-.20,.125,.14)|O(0,.13,.125,.14)|O(0,.46,.125,.14)
    if k==27:return person(.02,-.37,.99,flip=-1 if v%2 else 1)
    if k==28:return person(.12,-.29,.88,flip=-1)|person(-.23,.03,.55,flip=-1)
    if k==29:
        m=line(ellipsepts(-.30,.36,.19,.19),.045)|line(ellipsepts(.32,.36,.19,.19),.045)
        m|=line([(-.30,.36),(-.10,.06),(.06,.35),(-.30,.36)],.045)|line([(.06,.35),(.22,.075),(-.08,.11)],.045)|line([(.31,.35),(.17,-.015),(.24,-.05)],.042)
        return m|O(.025,-.24,.081,.085)|line([(-.02,-.13),(-.16,.025),(-.012,.16),(-.06,.31)],.078)|line([(-.05,-.11),(.05,.025),(.18,.02)],.06)
    if k==30:
        m=np.zeros_like(X,bool)
        for t in (np.arange(6)+(.5 if v%2==0 else 0))*np.pi/3:
            u=X*np.sin(t)+(Y-.14)*np.cos(t);vv=X*np.cos(t)-(Y-.14)*np.sin(t)
            m|=stroke(u,vv,[(0,0),(.43,0)],.076)
            for d in [.151,.28]:m|=stroke(u,vv,[(d+.096,.116),(d,0),(d+.096,-.116)],.077)
        return m
    if k==31:
        return P((-.36,-.18),(-.26,-.20),(-.14,-.12),(-.07,.015),(.18,.13),(.32,.12),(.30,.26),(.17,.27),(-.10,.14),(-.20,.025),(-.27,-.055),(-.39,-.07))|line([(-.25,-.13),(-.22,-.29),(-.28,-.36)],.04)|line([(-.16,.035),(-.29,.15),(-.33,.08)],.05)|line([(.23,.21),(.41,.37),(.46,.48)],.057)|line([(.17,.20),(.17,.34),(.32,.42)],.05)
    raise ValueError(k)

# Values used for lightness are perceptual, relative to a white sign face.
# Color channels describe red-blue opponency and yellowness separately.
white=(.94,0,0); black=(.075,0,0); red=(.29,.90,0); blue=(.28,-.90,0); yellow=(.68,.36,.95); grey=(.50,0,0)
@lru_cache(maxsize=None)
def render(k,v):
    tri=(k==11 or k==13 or 18<=k<=31)
    if tri:
        y=-Y if k==13 else Y
        # Slightly rounded corners by expanding an inset triangle.
        pts=[(0,-.91),(-.89,.905),(.89,.905)]
        outer=poly(X,y,pts)|stroke(X,y,pts+[pts[0]],.16)
        inner=poly(X,y,[(0,-.59),(-.68,.72),(.68,.72)])
    elif k==12: outer=abs(X)+abs(Y)<1.01
    elif k==14: outer=(np.maximum(abs(X),abs(Y))<.98)&((abs(X)+abs(Y))<1.38)
    else:outer=oval(X,Y,1,1)
    im=np.empty((3,N,N),np.float32);im[:]=np.array([.43,0,0])[:,None,None]
    def paint(m,c): im[:,m&outer]=np.array(c)[:,None]
    paint(outer,white)
    m=np.zeros_like(X,bool); fg=black
    if tri:
        paint(~inner,red)
        if k!=13:m=warn(k,v)
        if k==26:
            paint(O(0,-.20,.122,.15),red);paint(O(0,.13,.122,.15),yellow);paint(O(0,.46,.122,.15),(.33,-.18,.15));m[:]=False
    elif k==12:paint(abs(X)+abs(Y)<.69,yellow)
    elif k==14:
        paint((np.maximum(abs(X),abs(Y))<.925)&((abs(X)+abs(Y))<1.30),red);m=stopword();fg=white
    elif k==17:paint(outer,red);m=(abs(X)<.80)&(abs(Y)<.185);fg=white
    elif 33<=k<=40:paint(outer,blue);m=arrow(k,v);fg=white
    else:
        cancelled=k in [6,32,41,42]
        if not cancelled:paint(~oval(X,Y,.795,.795),red)
        if k in [0,1,2,3,4,5,6,7,8]:
            m=numbers(['20','30','50','60','70','80','80','100','120'][k],v)
        if k in [9,10,41,42]:
            s=.259; ux=-.323; uy=.008 if k in [9,41] else .024
            m=car(.32,.035,s)
            obj=reartruck(ux,-.055,.239) if k in [10,42] else car(ux,uy,s)
            paint(obj,grey if cancelled else red)
        if k==16:m=lorry()
        if cancelled:fg=grey
    paint(m,fg)
    if k in (6,32,41,42):
        # A band of parallel dark obliques; low resolution also averages them together.
        u=(X+Y)*.7071
        paint((abs(u)<.196)&((u+.196)%.084<.050),(.16,0,0))
    return np.concatenate((im,outer[None].astype('f')))

# Interpolate a hand-rendered shape, with antialiasing and smooth derivative
# templates. All statistics below refer exclusively to the constructed shapes.
def sample(im,u,v):
    ix=np.clip((u+1.2)*(N/2.4)-.5,0,N-1.0001); iy=np.clip((v+1.2)*(N/2.4)-.5,0,N-1.0001)
    i=ix.astype('i');j=iy.astype('i'); dx=ix-i;dy=iy-j
    return (im[:,j,i]*(1-dx)+im[:,j,i+1]*dx)*(1-dy)+(im[:,j+1,i]*(1-dx)+im[:,j+1,i+1]*dx)*dy

def blur(im,s):
    n=int(np.ceil(2.7*s)); v=np.arange(-n,n+1); g=np.exp(-.5*(v/s)**2);g/=g.sum()
    for axis in [-2,-1]:im=sum(b*np.roll(im,i,axis=axis) for b,i in zip(g,v))
    return im

def transform(im,H,rx,ry,skew,dx=0,dy=0):
    a=(np.arange(H)+.5-H/2); x,y=np.meshgrid(a-dx,a-dy); y=y/ry; x=x/rx-skew*y
    return sum(sample(im,x+(i/rx)-skew*(j/ry),y+j/ry) for i,j in [(-.26,-.26),(.26,-.26),(-.26,.26),(.26,.26)])/4

def group(k):
    if k in (6,32,41,42):return 1
    if k==11 or 18<=k<=31:return 2
    if k==13:return 3
    if k==12:return 4
    if k==14:return 5
    if k==17:return 6
    if 33<=k<=40:return 7
    return 0

layers=[]; weights={}
def layer(name,op,inp,**kw):
    layers.append(dict(name=name,op=op,inputs=inp if isinstance(inp,list) else [inp],**kw));return name
def aff(name,op,inp,W,b,**kw):
    W=np.asarray(W,dtype=np.float32);weights[name+'.W']=W;weights[name+'.b']=np.broadcast_to(np.asarray(b,dtype='f'),(len(W),)).copy();return layer(name,op,inp,out=len(W),**kw)
def conv(name,inp,W,b=0,p=0):return aff(name,'conv',inp,W,b,kernel=list(np.shape(W)[-2:]),stride=1,padding=p)
def mixing(name,inp,W,b=0):return conv(name,inp,np.asarray(W)[:,:,None,None],b)
def dense(name,inp,W,b=0):return aff(name,'dense',inp,W,b)
def pool(name,inp,size,stride=1,op='maxpool'):return layer(name,op,inp,size=list(size),stride=[stride,stride])
def reshape(name,inp,sh):return layer(name,'reshape',inp,shape=list(sh))

def features():
    # A piecewise-linear log compresses intensities. Color differences on log
    # channels respond to ratios, so they work in shadow as well as sunlight.
    knots=np.array([0,.025,.065,.14,.29,.55,1.]);sl=np.diff(np.log(knots+.025))/np.diff(knots); d=np.diff(np.r_[0,sl]);
    w=np.repeat(np.eye(3),len(d),axis=0); k=mixing('log_hinges','image',w,np.tile(-knots[:-1],3)); k=layer('positive','relu',k)
    k=mixing('log_rgb',k,np.kron(np.eye(3),d[None,:])); raw=mixing('log_opponents',k,[[.26,.47,.27],[1,-.28,-.72],[.2,.8,-1]])
    # Opponent magnitudes combine absolute chromaticity and contrast relative
    # to the immediate scene. The normalized luminance channel keeps detail.
    n=layer('relative','norm',raw); n=mixing('relative_mix',n,np.diag([1,.38,.40]));p=mixing('absolute_mix',raw,np.diag([0,.96,.90]));
    f=layer('appearance','add',[n,p]); sm=pool('small_appearance',f,(2,2),2,'avgpool')
    return sm,raw

def center_norm(m,mask):
    m=(m-(m*mask).sum()/max(.001,mask.sum()))*mask
    return m/max(1e-6,np.sqrt((m*m).sum()))

def outline_templates():
    # Each radius denotes a hypothesis about the bounding box of the sign;
    # allowable translations are implemented convolutionally, not memorized.
    radii=[3.7,4.45,5.35,6.35,7.22]; K=14; reps=[15,32,13,13,12,14,17,35]; out=[]
    for k,rep in enumerate(reps):
        im=render(rep,0).copy()
        if k==2:im=im[:,::-1,:]  # upright warning outline instead of yield
        for r in radii:
            for sx,sy,sk in [(1,1,0),(.83,1.06,0),(1.06,.85,0),(1,1,-.15),(1,1,.15)]:
                t=transform(im,K,r*sx,r*sy,sk); face=blur(t[3],.65); surround=blur(t[3],1.5); mask=np.clip(face+.7*surround,0,1)
                # Downweight variable central artwork, except when integral to
                # the family (bar, octagon, diamond, blue ground).
                yy,xx=np.mgrid[:K,:K]-(K-1)/2
                if k in [0,1,2,3,7]:
                    u=xx/(r*sx);v=yy/(r*sy)
                    body=np.exp(-((u/.55)**2+(v/.50)**2)**2)
                    if k!=7:mask*=1-.72*body
                    if k==7:t[0]=.43*(1-t[3])+.28*t[3];t[1]=-.9*t[3]
                    if k==1:t[0]=.43*(1-t[3])+.88*t[3]
                w=[]
                # luminance profile and matched chromaticity (opponent signs
                # carry meaning, not merely orientation, on traffic signs).
                ls=blur(t[0],.44)
                lum=center_norm(ls,mask); chrom=blur(t[1],.45); yl=blur(t[2],.45)
                # Color cannot always be expressed as a zero-mean profile, e.g.
                # a blue disk; compare its face to a partly neutral surround.
                chrom=chrom-.68*mask*(chrom*mask).sum()/max(.01,mask.sum());yl=yl-.58*mask*(yl*mask).sum()/max(.01,mask.sum())
                cn=max(.1,np.sqrt((chrom**2+yl**2).sum()))
                w=[lum*(.76 if k!=1 else .63),chrom/cn*.91,yl/cn*.73]
                out.append(np.array(w,dtype='f')/r)
    return np.asarray(out),len(radii)

def global_templates():
    # Compare the entire standardized sign as well as its outline and its
    # isolated central mark. Complementary characters can no longer drift to
    # a different position/scale than the perimeter in this comparison.
    radii=[3.7,4.45,5.35,6.35,7.22];w=[]
    for k in range(43):
        for r in radii:
            for sx,sy,sk in [(1,1,0),(.83,1.04,0),(1.05,.86,0),(1,1,-.13),(1,1,.13)]:
                t=transform(render(k,0),14,r*sx,r*sy,sk);mask=.30*blur(t[3],1.2)+.70*t[3]
                lum=center_norm(blur(t[0],.41),mask)
                redc=blur(t[1],.36); yel=blur(t[2],.36)
                for z in [redc,yel]:z-=.85*mask*(z*mask).sum()/max(.1,mask.sum())
                cn=max(.2,np.sqrt((redc**2+yel**2).sum()))
                w.append(np.array([lum,redc*.65/cn,yel*.63/cn],dtype='f')/r)
    return np.stack(w)

def inner_templates(k,pose):
    r,asp,v,sk=pose; K=20
    im=render(k,v);t=transform(im,K,r*asp,r,sk)
    yy,xx=np.mgrid[:K,:K]-(K-1)/2; U=xx/(r*asp)-sk*yy/r;V=yy/r; g=group(k)
    # Restrict attention to the sign interior. Give a small amount of support
    # to adjacent white space, which distinguishes sparse from dense symbols.
    if g in [0,1]: mask=np.clip((.78-np.sqrt(U*U+V*V))*r+.5,0,1)
    elif g==2:mask=np.clip(np.minimum((.71-V),(.65*(V+.64)-abs(U)))*r+.75,0,1)
    elif g==7:mask=np.clip((.88-np.sqrt(U*U+V*V))*r+.5,0,1)
    else:mask=np.clip((.92-np.maximum(abs(U),abs(V)))*r+.5,0,1)*t[3]
    z=blur(t[0],.50); low=blur(z,max(1,r*.145)); hi=(z-low)*mask
    c=center_norm(z,mask)
    hi=hi/(np.sqrt((hi*hi).sum())+.0001)
    # A blend of masked correlation and an approximately whitened glyph
    # filter. High-pass the filter, not the image, by linearity.
    w=(.45*c+.82*(hi-blur(hi,max(1,r*.145))))
    # Area-normalised rather than unit-energy alone, to compare hypotheses
    # about the same shape at different magnifications.
    w*=8.6/r
    return w.astype('f')

def contours(src,nr):
    # Simple cells use four signed derivative axes. Half-wave rectification
    # keeps opposite edges distinct; a slightly wider contour prototype gives
    # tolerance to small stroke displacements in the road alphabet.
    u=np.array([1,2,1])/4; du=np.array([-.5,0,.5]); dx=u[:,None]*du;dy=dx.T
    ds=[dx,dy,(dx+dy)*.7071,(dx-dy)*.7071];ds=np.array(ds+[-z for z in ds]);
    q=conv('signed_edges',src,ds[:,None],p=1);q=layer('edge_contrast','norm',q);q=layer('edge_polarities','relu',q)
    def signature(k,r,asp,sk,v):
        im=render(k,v);t=transform(im,20,r*asp,r,sk); y,x=np.mgrid[:20,:20]-9.5; V=y/r;U=x/(r*asp)-sk*V;g=group(k)
        rr=np.sqrt(U*U+V*V)
        if g in [0,1]: mask=np.clip((.73-rr)*r+.5,0,1)
        elif g==2:mask=np.clip(np.minimum((.66-V),(.53*(V+.72)-abs(U)))*r+.5,0,1)
        elif g==7: mask=np.clip((.84-rr)*r+.5,0,1)
        else:mask=t[3]
        z=blur(t[0],.62);gx=(np.roll(z,-1,axis=1)-np.roll(z,1,axis=1))/2;gy=(np.roll(z,-1,axis=0)-np.roll(z,1,axis=0))/2
        ed=np.array([gx,gy,(gx+gy)*.7071,(gx-gy)*.7071]);ed=np.maximum(np.concatenate([ed,-ed]),0)*mask
        ed=blur(ed,.58);ed-=.55*ed.mean(axis=0);ed-=mask*(ed*mask).sum(axis=(1,2),keepdims=True)/max(.1,mask.sum())
        return ed/(.006+np.linalg.norm(ed))
    aspvars=[(1,0,0),(.84,0,1),(1.13,0,2),(.98,-.13,1),(1,.13,3)]
    w=[];radii=[7.3,8.9,10.6,12.45,14.1]
    for k in range(43):
        for r in radii:
            for asp,sk,v in aspvars:w.append(signature(k,r,asp,sk,v)*(.88/r))
    m=conv('contour_layouts',q,np.stack(w));m=pool('contour_position',m,(5,5));m=reshape('contour_grid',m,(43*nr,1,len(aspvars)));m=pool('contour_aspect',m,(1,len(aspvars)));return reshape('contour_evidence',m,(43*nr,))

def number_reader(rect,D,n,nr,U,V):
    # Digit parts have different diagnostic value: the final zero checks the
    # baseline and height, but says nothing about 20 vs 30. Separate masks
    # prevent that shared zero from swamping the identity of the first digit.
    # Three-digit limits are parsed as prefix 1, middle numeral, suffix 0.
    keys=[0,1,2,3,4,5,7,8]; W=[]
    poses=[(ax,ay,sk,v) for ax in [.81,1.,1.19] for ay in [.85,1.03,1.18] for sk,v in [(0,0),(-.10,1),(.11,2),(0,3)]]
    for k in keys:
        for ax,ay,sk,v in poses:
            im=render(k,v);x=U/ax-sk*V;y=V/ay
            z=blur(sample(im[:1],x,y)[0],.60)
            # Clip the model field at the inner edge of the red ring: the
            # task of this module is specifically reading black type.
            z=np.where(x*x+y*y>.73**2,.94,z);base=np.clip((.56-abs(y))*12,0,1)
            if k<7:
                masks=[base*np.clip(.52-13*x,0,1),base*np.clip(.52+13*x,0,1)]; coeff=[.75,.25]
            else:
                masks=[base*np.clip((-.18-x)*14,0,1),base*np.clip((x+.26)*14,0,1)*np.clip((.22-x)*14,0,1),base*np.clip((x-.17)*14,0,1)];coeff=[.22,.52,.26]
            w=0
            for mask,co in zip(masks,coeff):
                part=center_norm(z,mask);w+=co*part
            # Look for both strokes and counters (white holes); subtract a
            # local surround from the mask as in an optical character reader.
            w=.85*w+.85*(w-blur(w,1.3));W.append(w*.34)
    t=aff('read_numerals','conv',rect,np.stack(W)[:,None],0,kernel=[D,D],stride=D,padding=0);t=reshape('numeral_models',t,(len(keys),len(poses),n));t=pool('numeral_style',t,(len(poses),1));t=reshape('numeral_positions',t,(len(keys),nr,n//nr));t=pool('numeral_translation',t,(1,n//nr));w=np.zeros((43*nr,len(keys)*nr));b=np.zeros(43*nr)
    for i,k in enumerate(keys):w[k*nr:(k+1)*nr,i*nr:(i+1)*nr]=np.eye(nr);b[k*nr:(k+1)*nr]=-1.65
    return dense('numeral_evidence',t,w,b)

def warning_parts(rect,D,n,nr,U,V):
    # Decompose each pictogram into a top portion, base, left, and right.
    # Stroke density and orientation are compared in each part rather than
    # relying solely on exact placement of every line in a complex drawing.
    keys=[11]+list(range(18,32)); styles=[(.87,.86,-.05),(1,1,0),(1.08,1.08,.05),(.92,1.1,0),(1.12,.89,0)]
    W=[];yy,xx=np.mgrid[0:D,0:D];rho=.40*D
    hs=[(D*.5,D*.22),(D*.5,D*.74),(D*.285,D*.62),(D*.715,D*.62)]
    # All four components are centered to reject common sign-face brightness.
    for k in keys:
        for ax,ay,sk in styles:
            for v in range(4):
                z=blur(sample(render(k,v)[:1],U/ax-sk*V,(V-.14)/ay+.14)[0],.72);w=0
                for cx,cy in hs:
                    m=np.exp(-((xx-cx)**2+(yy-cy)**2)/(rho*.78)**2);p=(z-(z*m).sum()/m.sum())*m;p=p/(np.linalg.norm(p)+.09);w+=p/4
                W.append(w*.23)
    x=aff('warning_parts','conv',rect,np.stack(W)[:,None],0,kernel=[D,D],stride=D,padding=0);x=reshape('warning_part_styles',x,(len(keys),len(styles)*4,n));x=pool('warning_part_style',x,(len(styles)*4,1));x=reshape('warning_part_positions',x,(len(keys),nr,n//nr));x=pool('warning_part_position',x,(1,n//nr));w=np.zeros((43*nr,len(keys)*nr));b=np.zeros(43*nr)
    for i,k in enumerate(keys):w[k*nr:(k+1)*nr,i*nr:(i+1)*nr]=np.eye(nr);b[k*nr:(k+1)*nr]=-1.4
    return dense('pictogram_evidence',x,w,b)

def local_alphabet(rect,D,n,nr):
    # Place both an individual character box and a full-number box on each
    # hypothesis. Center the filters separately in those boxes: the trailing
    # zero must not determine the response to the discriminative leading digit.
    keys=[0,1,2,3,4,5,7,8]; styles=[(r,t,shift) for r in [.87,1.04,1.20] for t in [.87,1.04,1.17] for shift in [-.044,.044]]
    part=[0,1];S=9;a,b=np.meshgrid((np.arange(S)+.5)*2/S-1,(np.arange(S)+.5)*2/S-1);filters=[];uvs=[]
    for wide in [False,True]:
        for ax,ay,shift in styles:
            for p in part:
                if p==0: cx,hw=(-.014,.225) if wide else (-.257,.278)
                else:cx,hw=0,.695 if wide else .57
                xx=cx+hw*a;yy=.48*b;u=(xx*ax+shift)/.70;v=yy*ay/.70
                # Invert the square-to-disc map analytically. Sampling clips
                # at the face boundary; none of these weights uses a photo.
                u0=np.sqrt(np.maximum(0,2+u*u-v*v+2*np.sqrt(2)*u));u1=np.sqrt(np.maximum(0,2+u*u-v*v-2*np.sqrt(2)*u));v0=np.sqrt(np.maximum(0,2-u*u+v*v+2*np.sqrt(2)*v));v1=np.sqrt(np.maximum(0,2-u*u+v*v-2*np.sqrt(2)*v));ix=np.clip(((u0-u1)*.25+.5)*D-.5,0,D-1.0001);iy=np.clip(((v0-v1)*.25+.5)*D-.5,0,D-1.0001)
                i=ix.astype(int);j=iy.astype(int);fx=ix-i;fy=iy-j
                mat=np.zeros((S*S,D,D),'f');J=np.arange(S*S)
                for h in [0,1]:
                    for l in [0,1]:mat[J,(j+h).ravel(),(i+l).ravel()]=((fy if h else 1-fy)*(fx if l else 1-fx)).ravel()
                filters.extend(mat);uvs.append((xx,yy))
    # Compose bilinear resampling with the character projectors before
    # export; the normalized 16x16 tiles can then be read directly.
    branches=[]
    variants=[(1,0),(1.03,1),(.97,2),(1,3)]
    for wide,ks in [(False,keys[:6]),(True,keys[6:])]:
        for p in part:
            # Project at three widths and three heights; the underlying face
            # sampler already handles both relative translations and aspect.
            inds=[i for i in range(len(styles)) if i%2==((i//2)%2)]
            f=np.stack(filters).reshape(2,len(styles),2,S*S,D,D)[int(wide),inds,p].reshape(-1,D,D)
            pre='part_'+str(int(wide))+str(p)
            xx,yy=uvs[int(wide)*len(styles)*2+p];w=[]
            for k in ks:
                for stretch,vv in variants:
                    z=blur(sample(render(k,vv)[:1],xx/stretch,yy)[0],.44);z-=z.mean();z/=np.linalg.norm(z)+.001;w.append(z)
            sc=np.stack(w).reshape(-1,S*S)
            f=f.reshape(len(inds),S*S,D*D);row=np.einsum('ij,sjk->isk',sc,f).reshape(-1,D,D)
            q=aff(pre+'_match','conv',rect,row[:,None],0,kernel=[D,D],stride=D,padding=0);q=reshape(pre+'_alternates',q,(len(ks),len(variants)*len(inds),n));q=pool(pre+'_style',q,(len(variants)*len(inds),1));q=reshape(pre+'_translation',q,(len(ks),nr,n//nr));q=pool(pre+'_best',q,(1,n//nr));w=np.zeros((43*nr,len(ks)*nr));bias=np.zeros(43*nr)
            for i,k in enumerate(ks):
                w[k*nr:(k+1)*nr,i*nr:(i+1)*nr]=np.eye(nr)*(.23 if p==0 else .12);bias[k*nr:(k+1)*nr]=-.82 if p==0 else -.48
            branches.append(dense(pre+'_evidence',q,w,bias))
    joined=layer('part_readings','add',branches)
    # Remove the common response of the candidate numbers: all contain a
    # trailing zero, so that common response is not independent new evidence.
    w=np.eye(43*nr)
    for k in keys:
        for j in keys:
            for s in range(nr):w[k*nr+s,j*nr+s]-=1/len(keys)
    return dense('numeral_identity',joined,w)

def face_reader(raw,nr):
    # Normalize INSIDE the sign at each hypothesized scale. Contrast between
    # sky and a dark sign should not erase the contrast between its digits.
    # A square sampling lattice is mapped to a disk, or to the inner triangle,
    # so that a normalizing channel contains almost exclusively sign face.
    smallr=[7.3,8.9,10.6,12.45,14.1];D=16; u,v=np.meshgrid((np.arange(D)+.5)*2/D-1,(np.arange(D)+.5)*2/D-1)
    li=mixing('light_for_rectifier',raw,[[1,0,0]])
    contrib=[]
    for geom,keys in [('disk',[k for k in range(43) if group(k)!=2 and k!=15]),('triangle',[11]+list(range(18,32)))]:
        if geom=='disk': Xg=.70*u*np.sqrt(1-v*v/2);Yg=.70*v*np.sqrt(1-u*u/2)
        else:Yg=.19+.455*v;Xg=u*(.60*(Yg+.63))
        # Each normalization channel spans the face under one shared pose.
        trans=[(-1.5,-1.5),(1.5,-1.5),(0,0),(-1.5,1.5),(1.5,1.5)]
        poses=[(r,sx,ix,iy) for r in smallr for sx in [.86,1.05] for ix,iy in trans]; n=len(poses)
        W=np.zeros((n,D*D,1024),'f');j=np.arange(D*D)
        for i,(r,sx,dx,dy) in enumerate(poses):
            x=15.5+dx+Xg.ravel()*r*sx;y=15.5+dy+Yg.ravel()*r;ix=x.astype(int);iy=y.astype(int); ax=x-ix;ay=y-iy
            for a in [0,1]:
                for b in [0,1]:W[i,j,(iy+b)*32+ix+a]=(ax if a else 1-ax)*(ay if b else 1-ay)
        rect=dense(geom+'_sampling',li,W.reshape(-1,1024));rect=reshape(geom+'_canvases',rect,(n,D,D));rect=layer(geom+'_face_contrast','norm',rect)
        # Project each tile independently with the same bank of glyphs.
        rect=reshape(geom+'_face_tiles',rect,(1,n*D,D));
        variants=[(.99,1.,0,0),(.91,.90,1,0),(1.08,1.08,2,0),(1.02,1.05,3,-.10),(.96,.98,1,.10),(1.13,.94,0,0),(.89,1.04,2,0),(1.,.89,3,0)]
        W=[]
        for k in keys:
            for ax,ay,font,sk in variants:
                im=render(k,font)
                # Sample the expected ideal mark through the same map and
                # account for sensor blur before comparing polarities.
                z=sample(im[:1],Xg/ax-sk*Yg,Yg/ay)[0];z=blur(z,.53);z-=z.mean(); z/=np.sqrt((z*z).sum())+.05
                W.append(z*(.168 if k not in [12,13] else .09))
        ps=aff(geom+'_read', 'conv', rect,np.stack(W)[:,None],0,kernel=[D,D],padding=0,stride=D)
        ps=reshape(geom+'_poses',ps,(len(keys),len(variants),n));ps=pool(geom+'_font',ps,(len(variants),1));ps=reshape(geom+'_remaining',ps,(len(keys),nr,n//nr));ps=pool(geom+'_translation',ps,(1,n//nr));
        W=np.zeros((43*nr,len(keys)*nr));
        for i,k in enumerate(keys):
            for s in range(nr):W[k*nr+s,i*nr+s]=1
        contrib.append(dense(geom+'_class_evidence',ps,W))
        if geom=='triangle':contrib.append(warning_parts(rect,D,n,nr,Xg,Yg))
        if geom=='disk':
            # Empty sign faces should have very little variation in *absolute*
            # log-lightness. Center the local window without dividing by its
            # contrast, then take its absolute mean deviation.
            # Adjacent-pixel variation (zero-DC) in the inner face also works
            # on a sloped illumination field.
            contrib.append(number_reader(rect,D,n,nr,Xg,Yg));contrib.append(local_alphabet(rect,D,n,nr))
            tiles=reshape('unscaled_face_tiles',geom+'_sampling',(1,n*D,D))
            # Instead of cancellation between opposite boundaries, use local
            # differences at every interior point before mean pooling.
            w=np.array([[[0,0,0],[-1,0,1],[0,0,0]],[[0,-1,0],[0,0,0],[0,1,0]]],dtype='f');w=np.concatenate([w,-w]);ed=conv('blank_gradients',tiles,w[:,None],p=1);ed=layer('blank_rectified','relu',ed);ed=mixing('blank_gradient_sum',ed,[[1,1,1,1]])
            w=np.zeros((D,D));w[2:-2,2:-2]=1;w/=w.sum();dev=aff('blank_variation','conv',ed,(-w*12)[None,None],0,kernel=[D,D],stride=D,padding=0);dev=reshape('blank_test',dev,(1,nr,n//nr));dev=pool('blank_offsets',dev,(1,n//nr));w=np.zeros((43*nr,nr));w[15*nr:16*nr]=np.eye(nr);contrib.append(dense('unmarked_evidence',dev,w))
            # The inverted triangle also has an unmarked central field. Its
            # narrow tip calls for a smaller test window than the blank disk.
            w=np.zeros((D,D));w[D//4:D*2//3,D*5//16:D*11//16]=1;w/=w.sum();dev=aff('yield_variation','conv',ed,(-w*10)[None,None],0,kernel=[D,D],stride=D,padding=0);dev=reshape('yield_test',dev,(1,nr,n//nr));dev=pool('yield_offsets',dev,(1,n//nr));w=np.zeros((43*nr,nr));w[13*nr:14*nr]=np.eye(nr);contrib.append(dense('yield_clear_field',dev,w))
    return layer('face_reader','add',contrib)

def build():
    sm,raw=features(); W,nr=outline_templates();q=conv('families',sm,W,p=1)
    q=pool('family_position',q,(5,5)); q=reshape('family_grid',q,(8,nr,5)); q=pool('family_aspect',q,(1,5)); fam=reshape('family_vector',q,(8*nr,))
    # Use a compact central window from the luminance map for symbol matching.
    # This is a pure selection matrix (no data-dependent sampling).
    eye=np.eye(32*32,dtype='f').reshape(32,32,1024);w=np.zeros((24*24,3,1024),'f');w[:,0]=eye[4:28,4:28].reshape(-1,1024)
    src=dense('central_light','appearance',w.reshape(576,-1));src=reshape('central_window',src,(1,24,24));src=layer('central_contrast','norm',src)
    kset=[k for k in range(43) if k!=15]
    poses=[(r,asp,v,sk) for r in [7.3,8.9,10.6,12.45,14.1] for asp,v,sk in [(1,0,0),(.82,1,0),(1.14,1,0),(.96,2,0),(1,0,-.13),(1,3,.13)]]
    W=np.stack([inner_templates(k,p) for k in kset for p in poses])[:,None]; m=conv('emblems',src,W);m=pool('emblem_position',m,(5,5));m=reshape('emblem_grid',m,(len(kset)*nr,1,len(poses)//nr));m=pool('emblem_aspect',m,(1,len(poses)//nr));
    w=np.zeros((43*nr,len(kset)*nr));
    for i,k in enumerate(kset):
        for s in range(nr):w[k*nr+s,i*nr+s]=.27
    a=dense('symbol_evidence',m,w)
    w=np.zeros((43*nr,8*nr));
    for k in range(43):
        for s in range(nr):w[k*nr+s,group(k)*nr+s]=1.55
    b=np.zeros(43);b[15]=1.65;b[12]=1.15;b[13]=1.35
    f=dense('shape_evidence',fam,w*.72,np.repeat(b,nr));
    gw=global_templates();g=conv('whole_signs',sm,gw*.94,p=1);g=pool('whole_position',g,(5,5));g=reshape('whole_aspects',g,(43,nr,5));g=pool('whole_aspect',g,(1,5));g=reshape('whole_evidence',g,(43*nr,));
    e=contours(src,nr);reader=face_reader(raw,nr);u=layer('pose_evidence','add',[f,a,g,e,reader]);u=reshape('class_scales',u,(43,nr,1));u=pool('consistent_hypothesis',u,(nr,1));reshape('scores',u,(43,))

if __name__=='__main__':
    build();out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=1));np.savez(out/'weights.npz',**weights)
