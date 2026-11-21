"""Data-free, analytic traffic-sign network. No file is ever read by this program.

All geometry below is in abstract sign-radius units, drawn from the meanings
of the sign classes, not from measurements or outlines of dataset images.
The network performs fixed sampling, instance normalization, correlations
with these mathematical drawings, and maximum pooling. There is no fitting,
optimization, stochastic search, external font, or external asset.
"""
import sys, json, math
from pathlib import Path
import numpy as np

F32=np.float32
LAYERS=[]
WEIGHTS={}

def layer(name,op,inputs,**kw):
    LAYERS.append(dict(name=name,op=op,inputs=[inputs] if isinstance(inputs,str) else inputs,**kw))
    return name

def dense(name,src,w,b=None):
    w=np.asarray(w,dtype=F32)
    WEIGHTS[name+'.W']=w
    WEIGHTS[name+'.b']=np.zeros(len(w),F32) if b is None else np.asarray(b,F32)
    return layer(name,'dense',src,out=len(w))

def conv(name,src,w,b=None,stride=1,padding=0):
    w=np.asarray(w,dtype=F32)
    WEIGHTS[name+'.W']=w
    WEIGHTS[name+'.b']=np.zeros(len(w),F32) if b is None else np.asarray(b,F32)
    return layer(name,'conv',src,out=len(w),kernel=list(w.shape[-2:]),stride=stride,padding=padding)

# Vector drawing primitives: implicit distances and polygon winding.
def line(x,y,points,width):
    d=np.full(np.broadcast(x,y).shape,100.,float)
    for (ax,ay),(bx,by) in zip(points[:-1],points[1:]):
        dx,dy=bx-ax,by-ay
        t=np.clip(((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy+1e-20),0,1)
        d=np.minimum(d,np.hypot(x-ax-t*dx,y-ay-t*dy))
    return d<=width/2

def poly(x,y,pts):
    inside=np.zeros(np.broadcast(x,y).shape,bool)
    for (ax,ay),(bx,by) in zip(pts,pts[1:]+pts[:1]):
        if ay != by:
            inside ^= ((ay>y)!=(by>y)) & (x < (bx-ax)*(y-ay)/(by-ay)+ax)
    return inside

def rect(x,y,a,b,c,d): return (x>=a)&(x<=c)&(y>=b)&(y<=d)
def ellipse(x,y,cx,cy,rx,ry=None):
    return ((x-cx)/rx)**2+((y-cy)/(rx if ry is None else ry))**2<=1

def arc(cx,cy,rx,ry,a,b,n=30):
    t=np.linspace(a,b,n)
    return list(zip(cx+rx*np.cos(t),cy+ry*np.sin(t)))

# Abstract rounded, monoline road-sign numerals. No font file is used.
def digit(x,y,d,thick=0.12,style=0):
    # Coordinates: the numeral is one unit wide and two units tall.
    x=x*2; y=y*2
    w=thick*2
    if d=='0':
        if style==0:
            pts=arc(0,0,.38,.87,0,2*math.pi,45)
        else:
            pts=[(-.36,-.58),(-.30,-.83),(0,-.9),(.30,-.83),(.36,-.58),(.36,.58),(.30,.83),(0,.9),(-.30,.83),(-.36,.58),(-.36,-.58)]
    elif d=='1':
        pts=[(-.30,-.55),(.02,-.87),(.02,.89)]
    elif d=='2':
        pts=[(-.37,-.58),(-.29,-.80),(-.10,-.90),(.16,-.88),(.34,-.70),(.36,-.44),(.24,-.20),(-.33,.70),(-.37,.87),(.39,.87)]
    elif d=='3':
        pts=[(-.37,-.72),(-.16,-.89),(.12,-.89),(.34,-.72),(.35,-.45),(.23,-.17),(-.05,-.04),(.23,.05),(.37,.30),(.35,.64),(.19,.85),(-.11,.90),(-.36,.73)]
    elif d=='5':
        pts=[(.36,-.87),(-.31,-.87),(-.36,-.04),(-.12,-.17),(.17,-.14),(.36,.08),(.38,.48),(.25,.78),(0,.90),(-.28,.81),(-.38,.64)]
    elif d=='6':
        pts=[(.30,-.75),(.09,-.89),(-.14,-.81),(-.32,-.52),(-.39,-.09),(-.38,.49),(-.22,.80),(.01,.89),(.25,.79),(.38,.52),(.35,.15),(.18,-.07),(-.08,-.11),(-.35,.10)]
    elif d=='7':
        pts=[(-.39,-.87),(.39,-.87),(.10,-.28),(-.12,.31),(-.23,.89)]
    elif d=='8':
        p1=arc(0,-.47,.33,.41,0,2*math.pi,32)
        p2=arc(0,.43,.38,.46,0,2*math.pi,32)
        return line(x,y,p1,w)|line(x,y,p2,w)
    else: raise ValueError(d)
    return line(x,y,pts,w)

def number(x,y,text,height=1.05,width=1.,thick=.12,style=0):
    n=len(text)
    dw=height*(.49 if n==2 else .36)*width
    gap=height*.095
    total=dw*n+gap*(n-1)
    a=np.zeros(np.broadcast(x,y).shape,bool)
    for j,ch in enumerate(text):
        cx=-total/2+dw/2+j*(dw+gap)
        # digit primitive's support has height .9 and width .4.
        a |= digit((x-cx)/(dw/.48),y/(height/.96),ch,thick,style)
    return a

def arrow(x,y,kind,th=.15):
    a=np.zeros(np.broadcast(x,y).shape,bool)
    if kind in (34,37,39):
        return arrow(-x,y,{34:33,37:36,39:38}[kind],th)
    if kind==35 or kind==36:
        a |= line(x,y,[(0,.65),(0,-.42)],th)
        a |= poly(x,y,[(0,-.72),(-.34,-.25),(.34,-.25)])
        if kind==36:
            a |= line(x,y,[(0,.22),(0,.12),(.17,-.02),(.47,-.02)],th)
            a |= poly(x,y,[(.75,-.02),(.39,-.32),(.39,.28)])
    elif kind==33:
        a |= line(x,y,[(0,.66),(0,.13),(.035,-.08),(.18,-.24),(.49,-.26)],th)
        a |= poly(x,y,[(.76,-.26),(.38,-.59),(.38,.07)])
    elif kind==38:
        a |= line(x,y,[(-.48,-.52),(.32,.34)],th*1.2)
        a |= poly(x,y,[(.62,.65),(.54,-.04),(-.06,.52)])
    elif kind==40:
        for angle in (0,2*math.pi/3,4*math.pi/3):
            a |= line(x,y,arc(0,0,.49,.49,angle-.9,angle+.55,23),th*.8)
            ang=angle+.55
            p=np.array([.49*math.cos(ang),.49*math.sin(ang)])
            t=np.array([-math.sin(ang),math.cos(ang)])
            n=np.array([math.cos(ang),math.sin(ang)])
            tip=p+t*.17
            a |= poly(x,y,[tuple(tip),tuple(p-t*.13+n*.20),tuple(p-t*.13-n*.20)])
    return a

def car(x,y,cx=0,cy=0,scale=1):
    x=(x-cx)/scale; y=(y-cy)/scale
    a=poly(x,y,[(-.18,-.29),(.18,-.29),(.28,-.04),(.30,.24),(-.30,.24),(-.28,-.04)])
    a &= ~rect(x,y,-.14,-.23,.14,-.10)
    a |= rect(x,y,-.27,.17,-.17,.33)|rect(x,y,.17,.17,.27,.33)
    return a

def truck(x,y):
    a=rect(x,y,-.60,-.29,.15,.23)
    a |= poly(x,y,[(.15,-.15),(.40,-.15),(.58,.05),(.58,.23),(.15,.23)])
    a &= ~poly(x,y,[(.23,-.10),(.36,-.10),(.47,.04),(.23,.04)])
    a |= ellipse(x,y,-.37,.25,.12)|ellipse(x,y,.36,.25,.12)
    return a

def stopword(x,y):
    x=x/.9; y=y/.5
    a=np.zeros(np.broadcast(x,y).shape,bool)
    letters=[('S',-.70),('T',-.23),('O',.24),('P',.71)]
    for ch,cx in letters:
        xx=(x-cx)/.36
        if ch=='S': pts=[(.4,-.7),(.1,-.9),(-.3,-.7),(-.4,-.3),(.3,.1),(.4,.5),(.2,.8),(-.2,.9),(-.4,.7)]
        elif ch=='T':
            a|=line(xx,y,[(-.45,-.86),(.45,-.86)],.18)
            pts=[(0,-.86),(0,.86)]
        elif ch=='O': pts=arc(0,0,.40,.86,0,2*math.pi)
        else: pts=[(-.38,.86),(-.38,-.86),(.20,-.86),(.43,-.56),(.36,-.10),(-.38,-.10)]
        a|=line(xx,y,pts,.18)
    return a

def person(x,y,cx=0,cy=0,s=1,walking=True):
    x=(x-cx)/s; y=(y-cy)/s
    a=ellipse(x,y,.015,-.42,.10)
    a|=line(x,y,[(0,-.25),(-.06,.08)],.18)
    a|=line(x,y,[(-.06,.05),(-.28,.45)],.11)
    a|=line(x,y,[(-.06,.05),(.15,.23),(.28,.45)],.11)
    a|=line(x,y,[(-.01,-.20),(.14,-.01),(.29,.01)],.085)
    a|=line(x,y,[(-.01,-.20),(-.22,-.01),(-.31,.03)],.085)
    return a

def warning(x,y,c,t=1.,style=0):
    a=np.zeros(np.broadcast(x,y).shape,bool)
    if c==11:
        a|=rect(x,y,-.095,-.30,.095,.60)
        a|=poly(x,y,[(0,-.55),(-.22,-.20),(.22,-.20)])
        a|=rect(x,y,-.43,.11,.43,.23)
    elif c==18:
        a|=poly(x,y,[(-.075,-.47),(.075,-.47),(.045,.30),(-.045,.30)])
        a|=ellipse(x,y,0,.50,.085)
    elif c in (19,20):
        if c==20: x=-x
        a|=line(x,y,[(.26,.58),(.26,.12),(.20,-.11),(.06,-.25),(-.25,-.27)],.17*t)
        a|=poly(x,y,[(-.51,-.27),(-.17,-.53),(-.17,-.01)])
    elif c==21:
        a|=line(x,y,[(.25,.57),(.25,.36),(.10,.21),(-.16,.08),(-.24,-.10),(-.20,-.27),(.04,-.39)],.15*t)
        a|=poly(x,y,[(.26,-.42),(-.03,-.59),(.02,-.18)])
    elif c==22:
        a|=rect(x,y,-.60,.46,.60,.57)
        a|=ellipse(x,y,-.29,.46,.29,.24)&(y<.50)
        a|=ellipse(x,y,.29,.46,.29,.24)&(y<.50)
    elif c==23:
        xx=x-.02; yy=y+.16
        ang=-.18
        a|=car(xx*math.cos(ang)+yy*math.sin(ang),-xx*math.sin(ang)+yy*math.cos(ang),scale=.86)
        for shift in (-.20,.20):
            a|=line(x,y,[(shift,.12),(shift-.13,.27),(shift+.07,.41),(shift-.10,.57)],.065*t)
    elif c==24:
        a|=line(x,y,[(-.31,-.48),(-.31,.59)],.095*t)
        a|=line(x,y,[(.03,-.48),(.03,-.20),(.32,.14),(.32,.59)],.095*t)
    elif c==25:
        a|=ellipse(x,y,-.15,-.27,.09)
        a|=line(x,y,[(-.15,-.12),(.02,.13),(-.22,.52)],.14)
        a|=line(x,y,[(.02,.13),(.20,.52)],.10)
        a|=line(x,y,[(-.12,-.05),(.12,.12),(.27,.03)],.08)
        a|=line(x,y,[(.16,-.11),(.40,.47)],.045)
        a|=poly(x,y,[(.30,.56),(.55,.27),(.72,.56)])
    elif c==26:
        a|=rect(x,y,-.18,-.48,.18,.59)
        # The lenses are coloured, not three white holes. In luminance
        # their housing and unlit lens faces both remain dark.
        for cy in (-.29,.055,.40): a &= ~ellipse(x,y,0,cy,.055)
    elif c==27:
        a|=person(x,y,cy=.09,s=.94)
        if style:
            for cx in (-.48,-.24,0,.24,.48): a|=rect(x,y,cx-.07,.58,cx+.07,.64)
    elif c==28:
        a|=person(x,y,cx=.19,cy=.08,s=.93)
        a|=person(-x,y,cx=.29,cy=.22,s=.61)
    elif c==29:
        for cx in (-.34,.34):
            rr=np.hypot(x-cx,y-.35)
            a|=(rr<.23)&(rr>.23-.045*t)
        a|=line(x,y,[(-.34,.35),(-.13,.02),(.15,.35),(-.34,.35)],.045*t)
        a|=line(x,y,[(.15,.35),(.25,-.04),(.34,.35)],.045*t)
        a|=line(x,y,[(.10,-.28),(-.09,-.10),(.09,.04),(-.03,.31)],.09)
        a|=line(x,y,[(.10,-.28),(.21,-.06),(.34,-.05)],.06)
        a|=ellipse(x,y,.13,-.43,.087)
    elif c==30:
        for ang in np.arange(6)*math.pi/3:
            u=np.array([math.cos(ang),math.sin(ang)]); v=np.array([-u[1],u[0]])
            center=np.array([0,.17]); end=center+.51*u; root=center+.29*u
            a|=line(x,y,[tuple(center),tuple(end)],.055*t)
            for sign in (-1,1): a|=line(x,y,[tuple(root),tuple(root+.10*u+sign*.14*v)],.045*t)
    elif c==31:
        # Leaping deer profile: head faces left in the abstract warning
        # symbol, unlike the right-facing construction below.
        x=-x
        a|=poly(x,y,[(-.43,.09),(-.24,-.04),(.14,.02),(.31,-.27),(.43,-.26),(.29,.25),(-.22,.25)])
        a|=poly(x,y,[(.30,-.27),(.39,-.37),(.57,-.29),(.51,-.19)])
        a|=line(x,y,[(-.31,.16),(-.37,.40),(-.61,.55)],.065)
        a|=line(x,y,[(-.18,.18),(-.02,.40),(-.17,.59)],.06)
        a|=line(x,y,[(.17,.18),(.36,.40),(.56,.42)],.06)
        a|=line(x,y,[(.23,.12),(.45,.26),(.58,.18)],.055)
        a|=line(x,y,[(-.39,.10),(-.54,-.02)],.05)
        for sh in (0,.08):
            a|=line(x,y,[(.35+sh,-.32),(.30+sh,-.56),(.22+sh,-.64)],.026)
            a|=line(x,y,[(.31+sh,-.49),(.43+sh,-.57)],.026)
    return a.astype(float)

SPEED={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}
CIRCLE_CLASSES=[0,1,2,3,4,5,7,8,9,10,15,16]
ROUND_CLASSES=[6,12,13,14,17,32,33,34,35,36,37,38,39,40,41,42]
TRI_CLASSES=[11]+list(range(18,32))
VARIANTS=96

def prototype(x,y,c,v):
    # Cartesian product of three icon heights, two widths, two stroke
    # thicknesses and two generic drafting styles; chosen a priori.
    hi=v%3; wi=(v//3)%2; ti=(v//6)%2; st=(v//12)%2
    # Four symmetric drafting offsets are shared by every class, rather
    # than estimating the position of any particular training symbol.
    dx,dy=((-0.045,-0.045),(-0.045,0.045),(0.045,-0.045),(0.045,0.045))[v//24]
    x=x-dx; y=y-dy
    scale=(.82,1.,1.18)[hi]; width=(.86,1.12)[wi]
    thick=(.12,.17)[ti]
    if c in SPEED:
        return 1-number(x,y,SPEED[c],height=1.05*scale,width=width,thick=thick,style=st).astype(float)
    if c in (12,13,15): return np.zeros(np.broadcast(x,y).shape)
    if c in TRI_CLASSES:
        # Regulatory warning pictograms sit wholly inside the triangular
        # border. Their bounding box is appreciably narrower than the base.
        scale = (.75,.90,1.05)[hi]
        return 1-warning(x/(width*scale),(y-.18)/scale+.18,c,(.9,1.15)[ti],st)
    xx=x/(scale*width); yy=y/scale
    if 33<=c<=40: return arrow(xx,yy,c,(.14,.18)[ti]).astype(float)
    if c==14: return stopword(xx,yy).astype(float)
    if c==17: return rect(xx,yy,-.79,-.14,.79,.14).astype(float)
    if c in (6,9,10,16,32,41,42):
        a=np.zeros(np.broadcast(x,y).shape,float)
        if c==6: a=.55*number(xx,yy,'80',height=1.05,width=1.,thick=thick,style=st)
        elif c in (9,41):
            a=np.maximum(.65*car(xx,yy,-.37,0,.78),car(xx,yy,.37,0,.78).astype(float))
        elif c in (10,42):
            tr=rect(xx,yy,-.62,-.35,-.10,.28)|ellipse(xx,yy,-.49,.30,.08)|ellipse(xx,yy,-.21,.30,.08)
            tr &= ~rect(xx,yy,-.54,-.29,-.18,-.17)
            a=np.maximum(.75*tr,car(xx,yy,.36,.05,.77).astype(float))
        elif c==16: a=truck(xx,yy).astype(float)
        if c in (6,32,41,42):
            a *= .65
            # The five parallel cancelling strokes run lower-left to upper-right.
            for shift in np.linspace(-.22,.22,5):
                stripe=np.abs(xx+yy-shift)<(.026 if st==0 else .036)
                a=np.maximum(a,stripe.astype(float)*.85)
        return 1-a
    raise ValueError(c)


def candidates():
    poses=[]
    for diameter in (20.,23.,26.,29.):
        for aspect in (.85,1.,1.18):
            for cy in (-2.5,0.,2.5):
                for cx in (-2.5,0.,2.5):
                    for angle in (-5.,5.):
                        poses.append((diameter,aspect,cx,cy,angle*math.pi/180))
    return poses


def transform(x,y,pose):
    diam,aspect,cx,cy,ang=pose
    xx=x*(diam/2)*math.sqrt(aspect); yy=y*(diam/2)/math.sqrt(aspect)
    return 15.5+cx+xx*math.cos(ang)-yy*math.sin(ang),15.5+cy+xx*math.sin(ang)+yy*math.cos(ang)


def sample_coordinates(kind,n=12):
    u,v=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n)
    if kind in ('circle','round'):
        u=2*u-1; v=2*v-1
        radius=.70 if kind=='circle' else .86
        return radius*u*np.sqrt(1-v*v/2),radius*v*np.sqrt(1-u*u/2)
    # Area-uniform barycentric sampling inside an abstract upright triangle.
    s=np.sqrt(u)
    x=s*(2*v-1); y=-1+1.8*s
    return .75*x,.2+.75*(y-.2)


def sampling_weights(poses,x,y,res):
    n=x.size
    W=np.zeros((len(poses)*n,res*res),F32)
    rows=np.arange(n)
    for k,pose in enumerate(poses):
        px,py=transform(x.ravel(),y.ravel(),pose)
        px=(px+.5)*res/32-.5; py=(py+.5)*res/32-.5
        px=np.clip(px,0,res-1.00001); py=np.clip(py,0,res-1.00001)
        xi=np.floor(px).astype(int); yi=np.floor(py).astype(int)
        fx=px-xi; fy=py-yi
        for dy in (0,1):
            for dx in (0,1):
                W[k*n+rows,(yi+dy)*res+xi+dx]+=(fx if dx else 1-fx)*(fy if dy else 1-fy)
    return W

# Shape categories, independent of internal pictograms.
FAMILIES=['ring','blue','cancel','diamond','yield','solid','triangle']

def family(c):
    if c in TRI_CLASSES: return 6
    if 33<=c<=40: return 1
    if c in (6,32,41,42): return 2
    if c==12: return 3
    if c==13: return 4
    if c in (14,17): return 5
    return 0


def region(x,y,kind):
    if kind in ('ring','blue','cancel','solid'):
        r=np.hypot(x,y)
        inner=r<.74
        full=r<1.
        outer=(r>=1.)&(r<1.22)
    elif kind=='diamond':
        r=np.abs(x)+np.abs(y)
        inner=r<.76; full=r<1.; outer=(r>=1.)&(r<1.22)
    else:
        if kind=='yield': y=-y
        def tri(scale):
            return poly(x/scale,(y-.2)/scale+.2,[(0,-1.),(-1.,.8),(1.,.8)])
        inner=tri(.74); full=tri(1.); outer=tri(1.22)&~full
    return inner,full&~inner,outer,full


def shape_weights(poses,res=16):
    # Features are bounded opponent red, blue, yellow, and normalized luminance.
    W=np.zeros((len(FAMILIES)*len(poses),4*res*res),F32)
    # Integrate masks over pixels to avoid a hard raster boundary.
    yy,xx=np.meshgrid(np.arange(res),np.arange(res),indexing='ij')
    for j,pose in enumerate(poses):
        diam,asp,cx,cy,ang=pose
        for f,kind in enumerate(FAMILIES):
            masks=np.zeros((4,res,res),float)
            for oy in (-.375,-.125,.125,.375):
                for ox in (-.375,-.125,.125,.375):
                    dx=(xx+.5+ox)*32/res-.5-(15.5+cx)
                    dy=(yy+.5+oy)*32/res-.5-(15.5+cy)
                    x=(dx*math.cos(ang)+dy*math.sin(ang))/(diam/2*math.sqrt(asp))
                    y=(-dx*math.sin(ang)+dy*math.cos(ang))/(diam/2/math.sqrt(asp))
                    masks+=np.array(region(x,y,kind),float)/16
            i,r,o,full=[a/(a.sum()+1e-8) for a in masks]
            ww=np.zeros((4,res,res),float)
            if kind in ('ring','triangle','yield'):
                ww[0]=1.65*(r-.55*i-.35*o)
                ww[1]=-.70*full
                ww[3]=.20*(i-.7*r-.3*o)
            elif kind=='blue':
                ww[1]=1.65*(full-.65*o)
                ww[0]=-.85*full
            elif kind=='cancel':
                ww[0]=-.9*full; ww[1]=-.6*full; ww[2]=-.25*full
                ww[3]=.40*(full-o)
            elif kind=='diamond':
                ww[2]=1.8*(i-.65*r-.35*o)
                ww[0]=-.25*r
                ww[3]=.12*(r-o)
            elif kind=='solid':
                ww[0]=1.65*(full-.6*o)
                ww[1]=-.3*full
            W[f*len(poses)+j]=ww.ravel()
    return W


def make_templates(kind,classes):
    x,y=sample_coordinates(kind)
    weights=[]; biases=[]
    # Antialias analytic symbols using a fixed quadrature in sign coordinates.
    for c in classes:
        for v in range(VARIANTS):
            a=np.zeros_like(x)
            # Analytic box integration approximates a pixel footprint on
            # a small sign and retains thin spokes, strokes, and limbs.
            for dx in (-.06,-.03,0.,.03,.06):
                for dy in (-.06,-.03,0.,.03,.06):
                    a+=prototype(x+dx,y+dy,c,v)/25
            a-=a.mean()
            std=np.sqrt((a*a).mean())
            coef=1.20
            bias=-.15
            if c in (12,13): coef=0; bias=.23
            elif c==15: coef=0; bias=.35
            elif c in (6,32,41,42): coef=1.45; bias=.02
            weights.append((coef*a/(max(std,1e-8)*a.size)).astype(F32))
            biases.append(bias)
    return np.array(weights)[:,None,:,:],np.array(biases,F32)


def branch(kind,classes,poses,gray,shape,globalgray):
    n=len(poses)
    x,y=sample_coordinates(kind)
    res=32
    samp=dense(kind+'_sample',gray,sampling_weights(poses,x,y,res))
    patch=layer(kind+'_patches','reshape',samp,shape=[n,12,12])
    norm=layer(kind+'_contrast','norm',patch)
    strip=layer(kind+'_strip','reshape',norm,shape=[1,n*12,12])
    # Bounded contrast prevents single bright/dark sampling outliers from
    # dominating the correlation; this is a fixed symmetric clipping rule.
    clip=conv(kind+'_clip_parts',strip,np.ones((2,1,1,1),F32),[1.5,-1.5])
    clip=layer(kind+'_clip_relu','relu',clip)
    clipped=conv(kind+'_clipped',clip,np.array([1,-1],F32).reshape(1,2,1,1),[-1.5])
    w,b=make_templates(kind,classes)
    match=conv(kind+'_symbols',clipped,w,b,stride=12)
    # A white-on-blue interpretation must not beat clear evidence for a
    # dark-on-white sign of another family just by matching a negative.
    sw=np.zeros((len(FAMILIES)*len(FAMILIES),len(FAMILIES),1,1),F32)
    for f in range(len(FAMILIES)):
        for j in range(len(FAMILIES)):
            sw[f*len(FAMILIES)+j,j,0,0]+=1
            sw[f*len(FAMILIES)+j,f,0,0]-=1
    comp=conv(kind+'_family_comparison',shape,sw)
    comp=layer(kind+'_family_comparison_grid','reshape',comp,shape=[len(FAMILIES),len(FAMILIES),n])
    comp=layer(kind+'_stronger_family','maxpool',comp,size=[len(FAMILIES),1],stride=[len(FAMILIES),1])
    comp=layer(kind+'_family_difference','reshape',comp,shape=[len(FAMILIES),n,1])
    penalty=np.zeros((len(FAMILIES),len(FAMILIES),1,1),F32)
    penalty[1,1,0,0]=-1.5
    penalty[5,5,0,0]=-1.5
    comp=conv(kind+'_polarity_penalty',comp,penalty)
    shape=layer(kind+'_consistent_shape','add',[shape,comp])
    # Require affirmative evidence for the selected sign family; an
    # accidental symbol match in the roadside background is insufficient.
    thresholds=np.array([.20,.20,.04,.20,.20,.20,.20],F32)
    w=np.zeros((len(FAMILIES),len(FAMILIES),1,1),F32)
    for j in range(len(FAMILIES)): w[j,j,0,0]=-1
    deficit=conv(kind+'_family_deficit',shape,w,thresholds)
    deficit=layer(kind+'_family_missing','relu',deficit)
    w=np.zeros((len(classes)*VARIANTS,len(FAMILIES),1,1),F32)
    for i,c in enumerate(classes): w[i*VARIANTS:(i+1)*VARIANTS,family(c),0,0]=1
    context=conv(kind+'_context',shape,w)
    gate=conv(kind+'_family_gate',deficit,-2*w)
    context=layer(kind+'_family_evidence','add',[context,gate])
    if kind=='circle':
        # A blank sign must have a blank interior, not merely a red ring.
        # Use the absolute response to fixed adjacent-sample differences.
        # This is an input-dependent test, not a class statistic.
        nn=12*12
        d=np.zeros((2*nn,1,12,12),F32)
        for iy in range(12):
            for ix in range(12):
                q=iy*12+ix
                jy,jx=iy,min(ix+1,11)
                if ix==11: jy,jx=min(iy+1,11),ix
                d[q,0,iy,ix]=1
                d[q,0,jy,jx]-=1
                d[nn+q]=-d[q]
        # Preserve the amplitude of interior contrast relative to the
        # whole photo: patch normalization would erase blankness itself.
        xx,yy=sample_coordinates('circle')
        raw=dense('blank_samples',globalgray,sampling_weights(poses,xx,yy,16))
        raw=layer('blank_strip','reshape',raw,shape=[1,n*12,12])
        variation=conv('interior_differences',raw,d,stride=12)
        variation=layer('interior_absolute','relu',variation)
        w=np.zeros((len(classes)*VARIANTS,2*nn,1,1),F32)
        i=classes.index(15)
        w[i*VARIANTS:(i+1)*VARIANTS,:,0,0]=-1.5/nn
        penalty=conv('blank_interior_test',variation,w)
        scores=layer(kind+'_joint','add',[match,context,penalty])
    else:
        scores=layer(kind+'_joint','add',[match,context])
    arranged=layer(kind+'_alternatives','reshape',scores,shape=[len(classes),VARIANTS,n])
    best=layer(kind+'_best','maxpool',arranged,size=[VARIANTS,n],stride=[VARIANTS,n])
    w=np.zeros((43,len(classes)),F32)
    for i,c in enumerate(classes): w[c,i]=1
    return dense(kind+'_classes',best,w)


def build(folder):
    gray=conv('luminance','image',np.array([.299,.587,.114],F32).reshape(1,3,1,1))
    smallgray=layer('luminance16','avgpool',gray,size=[2,2],stride=[2,2])
    rgb=layer('rgb_joint','reshape','image',shape=[1,96,32])
    norm=layer('global_light','norm',rgb)
    rgb=layer('rgb_normalized','reshape',norm,shape=[3,32,32])
    # Clip opponent magnitudes to [0,1] and luminance to [-2,2].
    base=np.array([[1.5,-.75,-.75],[-.75,-.75,1.5],[.75,.75,-1.5],[.299,.587,.114]],F32)
    w=np.concatenate([base,base],axis=0).reshape(8,3,1,1)
    b=np.array([0,0,0,2,-1,-1,-1,-2],F32)
    f=conv('opponents',rgb,w,b)
    f=layer('opponent_relu','relu',f)
    w=np.zeros((4,8,1,1),F32)
    for j in range(4): w[j,j,0,0]=1; w[j,j+4,0,0]=-1
    f=conv('bounded_features',f,w,[0,0,0,-2])
    f=layer('colour16','avgpool',f,size=[2,2],stride=[2,2])
    poses=candidates()
    shape=dense('shape_correlations',f,shape_weights(poses))
    shape=layer('shape_grid','reshape',shape,shape=[len(FAMILIES),len(poses),1])
    globalgray=layer('global_gray_contrast','norm',smallgray)
    a=branch('circle',CIRCLE_CLASSES,poses,gray,shape,globalgray)
    b=branch('triangle',TRI_CLASSES,poses,gray,shape,globalgray)
    c=branch('round',ROUND_CLASSES,poses,gray,shape,globalgray)
    layer('class_scores','add',[a,b,c])
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    (folder/'network.json').write_text(json.dumps({'layers':LAYERS},indent=2)+'\n')
    np.savez(folder/'weights.npz',**WEIGHTS)
    print('Built',len(LAYERS),'layers with',sum(w.size for w in WEIGHTS.values()),'weights.')

if __name__=='__main__': build(sys.argv[1])
