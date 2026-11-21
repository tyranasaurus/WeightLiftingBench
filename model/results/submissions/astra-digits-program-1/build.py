"""A drawing grammar, renderer, and recognizer. No input files are needed."""
import sys
from pathlib import Path
import numpy as np

# Coordinates are in pen units, with y increasing down the page.  A family is
# a set of strokes.  The extra copies here are alternative pen motions, not
# stored samples.  Sampling the parameters is quadrature over writing styles.
rng = np.random.default_rng(392641)
B = 64
G = 28
F = np.float32

def scalar(x):
    return np.broadcast_to(np.asarray(x, dtype=float), (B,))

def rand(a=0,b=1):
    return rng.uniform(a,b,B)

def norm(s):
    return np.clip(rng.standard_normal(B),-2.8,2.8)*s

def pt(x,y):
    return np.stack((scalar(x),scalar(y)),axis=1)

def line(a,b,n):
    t = np.linspace(0,1,n)[None,:,None]
    return a[:,None]*(1-t)+b[:,None]*t

def path(nodes):
    # Rounded joins with locally controlled sharpness. Interpolate short
    # straight segments too, so rubber-sheet deformations can bend them.
    nodes=[(pt(p[0],p[1]),scalar(p[2]) if len(p)>2 else scalar(.8)) for p in nodes]
    a=nodes[0][0]; parts=[]
    t = np.linspace(0,1,11)[None,:,None]
    for k in range(1,len(nodes)-1):
        p,r=nodes[k]; r=r[:,None]*.5
        b=p+(nodes[k-1][0]-p)*r; c=p+(nodes[k+1][0]-p)*r
        parts.extend([line(a,b,8), (1-t)**2*b[:,None]+2*t*(1-t)*p[:,None]+t**2*c[:,None]])
        a=c
    parts.append(line(a,nodes[-1][0],16))
    return np.concatenate(parts,axis=1)

def cubic(start,segments):
    a=pt(*start)[:,None]; t=np.linspace(0,1,22)[None,:,None]; parts=[]
    for h,i,end in segments:
        b=pt(*h)[:,None]; c=pt(*i)[:,None]; d=pt(*end)[:,None]
        parts.append(a*(1-t)**3+3*b*t*(1-t)**2+3*c*(1-t)*t**2+d*t**3); a=d
    return np.concatenate(parts,axis=1)

def ellipse(cx,cy,rx,ry,t0,t1,power=1):
    t=scalar(t0)[:,None]+scalar(t1-t0)[:,None]*np.linspace(0,1,76)[None,:]
    c=np.cos(t); s=np.sin(t); e=scalar(power)[:,None]
    return np.stack((scalar(cx)[:,None]+scalar(rx)[:,None]*np.sign(c)*np.abs(c)**e,
                     scalar(cy)[:,None]+scalar(ry)[:,None]*np.sign(s)*np.abs(s)**e),axis=-1)

pi=np.pi

def zero(style):
    # A loop, often with a small overlap, pinched or tapered at one end.
    angle=rand(-1.7,-.5); arc=rand(1.88,2.15)*pi
    p=ellipse(0,.50,.49,.51,angle,angle+arc,rand(.8,1.35))
    u=np.linspace(0,1,p.shape[1])[None,:]
    p[:,:,0] *= 1+rand(-.75,.75)[:,None]*(p[:,:,1]-.5)
    p[:,:,0] += norm(.09)[:,None]*np.maximum(0,1-u*8)**2
    p[:,:,1] += rand(-.12,.15)[:,None]*np.maximum(0,1-u*8)**2
    if style==1:
        # Two tips can cross instead of meeting exactly at the crown.
        p[:,:12,0] += rand(-.38,.23)[:,None]*np.linspace(1,0,12)[None,:]**2
        p[:,:12,1] += rand(.02,.26)[:,None]*np.linspace(1,0,12)[None,:]**2
    if style==2:
        p=cubic((.24,.12),[((.00,-.18),(-.41,.07),(-.49,.43)),((-.69,.94),(-.16,1.16),(.21,.94)),((.63,.72),(.62,.20),(.28,.04)),((.13,-.03),(-.03,.02),(-.12,.12))])
    return [p]

def one(style):
    x=rand(-.28,.23); y=rand(-.03,.04); xb=x+rand(-.12,.12)
    flag=rand(.09,.65); down=rand(.08,.48)
    if style==0:
        p=path([(x+norm(.04),0),(x+norm(.05),.20),(xb+norm(.05),.68),(xb,1)])
    elif style in (1,2):
        p=path([(x-flag,down),(x-.04,y+.02,.75),(x,y,.25),(xb,1)])
    else:
        p=path([(x-.10,.01),(x,.01,.55),(xb+.02,.92,.65),(xb+.12,1)])
    if style==2:
        w=rand(.25,.8); return [p,path([(xb-w*.5,1+norm(.04)),(xb,1,.55),(xb+w*.5,1+norm(.04))])]
    return [p]

def two(style):
    a=rand(.12,.52); low=rand(.65,.94); right=rand(.32,.57); end=rand(.27,.68); h=rand(.00,.20)
    if style==0 or style==3:
        p=cubic((-.49,a),[((-.46,a-.15),(-.28,-.09),(.04,.01)),((.51,-.05),(right+.05,.24),(right-.12,.41)),((.16,.61),(-.34,low-.01),(-.5,1.0)),((-.23,1.00),(end-.15,.98-h),(end,1-h))])
        if style==3:
            n=p.shape[1]//4
            p[:,:n,1]+=rand(-.07,.09)[:,None]*np.linspace(1,0,n)[None,:]
    elif style==1:
        # Cursive two: the lower-left reversal makes a loop, then exits right.
        p=cubic((-.46,a),[((-.40,-.01),(.13,-.11),(.37,.08)),((.68,.28),(.02,.69),(-.31,.89)),((-.77,1.15),(-.50,low-.05),(-.27,low)),((.03,low),(.20,1+h),(end,.98+norm(.055)))])
    elif style==2:
        p=path([(-.52,a),(-.20,.03,.9),(.22,-.02,.9),(.51,.16,.9),(.34,.41,1),(-.50,.98,rand(.12,.80)),(.06,1-h*.45,.9),(end,1-h)])
    elif style==4:
        p=cubic((-.44,.14),[((-.12,-.01),(.31,-.13),(.44,.13)),((.55,.39),(-.05,.67),(-.43,.96)),((-.07,.87),(.26,1.07),(end,.89))])
    elif style in (5,6):
        # See the baseline loop as an ellipse with a long entry from its
        # upper right and a tangential exit, rather than a Z with a bent toe.
        cx=rand(-.37,-.08); rx=rand(.16,.45); ry=rand(.075,.21); cy=1-ry
        bx=rand(.01,.43); tip=rand(-.11,.29); crown=rand(-.42,.08)
        enter=ellipse(cx,cy,rx,ry,0,1.5*pi,rand(.87,1.17))
        cap=cubic((crown,a*.61),[((crown+.18,-.12),(bx+.07,-.04),(bx+.10,.19)),((bx+.18,.37),(cx+rx+.07,cy-.17),(cx+rx,cy))])
        endpath=cubic((cx,cy-ry),[((cx+.21,cy-ry-.06),(end-.18,cy-ry+tip),(end,cy-ry+tip))])
        if style==6:
            cap=cubic((bx-.22,.015),[((bx+.22,.18),(bx+.12,.38),(bx,.53)),((bx-.08,.65),(cx+rx+.06,cy-.12),(cx+rx,cy))])
        p=np.concatenate((cap,enter,endpath),axis=1)
    elif style==7:
        # Z-style two, with a flattened crown and a sharp point at the toe.
        p=path([(-.5,rand(-.05,.16)),(.01,.02,.91),(.46,.015,rand(.12,.80)),(.32,.31,.85),(-.46,.94,rand(.15,.85)),(-.15,1,.98),(end,.96+norm(.09))])
    else:
        # Upright question-mark-shaped cap descends into a rounded shallow foot.
        wx=rand(-.33,.13)
        p=cubic((wx-.17,.01),[((wx+.04,-.04),(wx+.31,.15),(wx+.24,.32)),((wx+.22,.56),(-.28,.73),(-.40,.88)),((-.58,1.11),(.21,.92),(end,.89+norm(.10)))])
    return [p]

def three(style):
    m=rand(.34,.63); notch=rand(-.30,.13); top=rand(.04,.29); end=rand(.72,1.0); w=rand(.27,.54)
    seg=[((-.10,-.04),(.23,-.08),(w,.06)),((w+.25,.22),(notch+.10,m-.03),(notch,m)),((.12,m-.06),(.51,m+.06),(.47,m+.23)),((.43,1.02),(-.14,1.10),(-.48,end))]
    if style==0: p=cubic((-.42,top),seg)
    elif style==1:
        p1=path([(-.48,top*.40),(.04,.03,.85),(.43,.01,rand(.12,.6)),(notch,m)])
        p=np.concatenate((p1,cubic((notch,m),seg[2:])),axis=1)
    elif style==2:
        # A little loop at the junction of the two lobes.
        p=cubic((-.44,top),[seg[0],((w+.21,.24),(notch+.09,m+.06),(notch-.14,m+.02)),((notch-.32,m-.02),(notch-.03,m-.09),(.08,m-.01)),((.53,m+.08),(.62,.78),(.32,.90)),((.02,1.09),(-.30,1.0),(-.48,end))])
    else:
        p=path([(-.40,top),(.05,.0,1),(.49,.07,.9),(.33,.28,.8),(notch,m,.18),(.35,m+.035,.85),(.48,.73,.8),(.20,.99,1),(-.23,1.01,.8),(-.50,end)])
    return [p]

def four(style):
    y=rand(.42,.73); w=rand(.00,.39); stem=rand(.04,.36); l=rand(-.42,-.09); cap=rand(-.03,.24)
    if style<3:
        l= l if style!=1 else stem-rand(.10,.32)
        a=path([(l,rand(-.04,.22)),(-.47,y-.03,rand(.1,.85)),(-.25,y+.01,.75),(stem+w,y+norm(.085))]);
        b=path([(stem+norm(.09),cap),(stem+norm(.035),y,.9),(stem+norm(.08),1)])
        if style==2:
            # Continuous U, right side retraced downwards.
            a=path([(l,.05),(-.51,y,rand(.35,.95)),(-.11,y+.04,.9),(stem,y-.06,.9),(stem+.04,cap,.25),(stem+.03,1)])
            return [a]
        return [a,b]
    if style==3:
        return [path([(stem,.00),(-.49,y,.16),(stem+w,y+.02)]),path([(stem,.00),(stem+norm(.035),.50),(stem+norm(.04),1)])]
    return [path([(stem-.08,.03),(-.48,y,rand(.2,.6)),(stem+w,y+norm(.05))]),path([(stem,.21),(stem,.68),(stem-.03,1)])]

def five(style):
    y=rand(.33,.57); l=rand(-.48,-.22); end=rand(.70,.99); ey=rand(-.08,.06); top=rand(.30,.58)
    bow=cubic((l,y),[((-.18,y-.12),(.28,y-.13),(.43,y+.05)),((.70,.72),(.35,.95),(.06,.99)),((-.16,1.04),(-.42,.99),(-.49,end))])
    if style in (0,1):
        x=l+rand(-.04,.23)
        a=path([(top,ey),(x,.03,rand(.10,.6)),(l,y)])
        return [np.concatenate((a,bow),axis=1)]
    if style==2:
        # Bowl and top bar made in separate pen strokes.
        x=l+rand(-.06,.17)
        a=path([(x,rand(-.02,.18)),(l,y)]); b=path([(x+rand(-.17,.19),rand(.00,.12)),(.10,.065,.85),(top,ey)])
        return [np.concatenate((a,bow),axis=1),b]
    if style==3:
        p=cubic((top,ey),[((.04,.11),(l-.14,-.01),(l,.18)),((l-.03,.30),(l+.08,.35),(-.01,y)),((.42,y+.09),(.60,.66),(.32,.84)),((.10,1.02),(-.27,1.10),(-.52,end))]); return [p]
    if style==4:
        return [path([(top,ey),(l+.04,.04,.35),(l,y,.20),(.11,y-.05,.95),(.48,y+.14,.95),(.36,.86,.95),(-.01,1,.95),(-.33,.97,.90),(-.51,end)])]
    # A fast continuous five can round off, or entirely skip, the corner
    # between the downstroke and the start of the bowl.
    mid=rand(-.30,.08); h=rand(.02,.27); tip=rand(.70,1.05)
    return [cubic((top,ey),[((.13,h),(-.57,-.06),(-.40,.22)),((-.42,.33),(mid-.07,.40),(mid+.12,.55)),((.61,.78),(.07,1.05),(-.25,.98)),((-.40,.97),(-.46,.88),(-.45,tip))])]

def six(style):
    x=rand(-.05,.52); start=rand(-.04,.05); m=rand(.40,.64); join=rand(.60,.87)
    seg=[((x-.24,.12),(-.51,.40),(-.48,.72)),((-.48,.96),(-.22,1.07),(.07,.99)),((.45,.96),(.61,.70),(.36,m+.04)),((.10,m-.10),(-.30,m+.02),(-.44,join))]
    if style==0: p=cubic((x,start),seg)
    elif style==1:
        p=cubic((x+.24,.14),[((x+.20,-.04),(x+.05,-.06),(x-.10,.02))]+seg)
    elif style==2:
        p=cubic((x,start),seg[:-1]+[((.21,m-.04),(-.07,m-.04),(-.20,m+.05))])
    elif style==3:
        p=cubic((x,start),seg+[((-.48,join+.12),(-.27,join+.16),(-.13,.94))])
    else:
        p=cubic((x,start),[((x-.22,.25),(-.23,.45),(-.41,.69)),((-.73,1.11),(.02,1.09),(.28,.92)),((.81,.56),(.15,m-.11),(-.41,join))]);
    return [p]

def seven(style):
    r=rand(.33,.57); l=rand(-.60,-.33); e=rand(-.28,.31); dip=rand(-.045,.12); endx=e+norm(.10)
    nodes=[(l,rand(-.05,.14)),((l+r)*.5,dip,rand(.7,1)),(r,.04,rand(.13,.73)),((r+e)*.5+norm(.075),.52,.9),(endx,1)]
    if style in (1,3): nodes=[(l-rand(-.03,.12),rand(.14,.37))]+nodes
    if style in (4,5,6):
        # The diagonal often continues well to the left of the top bar.
        # A short flag on a one, by contrast, rises towards the start point.
        e=rand(-.96,-.20); h=rand(-.04,.11); slope=rand(.10,.38)
        nodes=[(l,slope),(-.24,h+.03,.86),(.16,h,.87),(r,.015,rand(.15,.8)),((e+r)*.5+norm(.08),.52,.82),(e,1)]
        if style==5: nodes=[(l-.035,slope+.13)]+nodes
        if style==6: nodes[0]=(l,h+norm(.04))
    p=path(nodes)
    if style in (2,3):
        m=rand(.43,.65); cx=r*(1-m)+e*m; return [p,path([(cx-rand(.16,.40),m+norm(.03)),(cx,m,.65),(cx+rand(.11,.32),m+norm(.045))])]
    return [p]

def eight(style):
    cy=rand(.36,.58); notch=rand(-.15,.18); w=rand(.27,.53); b=rand(.31,.57)
    # A continuous figure eight starts high, passes through the waist, goes
    # around the bottom and returns up the other diagonal.
    xp=rand(-.16,.22); yp=rand(-.025,.05)
    if style in (0,1,2):
        p=cubic((xp,yp),[((xp-.12,-.07),(-w-.18,.03),(-w,.21)),((-w+.05,.36),(.01,cy-.06),(notch+.07,cy+.045)),((b,cy+.17),(b+.19,.89),(.20,.97)),((-.10,1.09),(-b-.14,1.00),(-b,.78)),((-b+.02,.65),(-.10,cy+.07),(notch,cy)),((.20,cy-.12),(w+.15,.23),(w,.09)),((w-.07,-.02),(xp+.06,-.02),(xp+norm(.08),yp+norm(.05)))])
        if style==1: p=p[:,int(p.shape[1]*.10):]
        if style==2: p[:,:,0]*=-1
        return [p]
    if style==3:
        return [ellipse(notch*.55,cy*.49,w,cy*.51,-pi/2,1.53*pi,rand(.84,1.18)),ellipse(-notch*.35,(1+cy)*.5,b,(1-cy)*.51,-pi/2,1.5*pi,rand(.85,1.18))]
    if style>=5:
        # Use a waist-to-waist parameterization as well. A near-straight
        # diagonal, ending a little beyond the crossing, is common in a fast 8.
        cy=rand(.24,.68); w=rand(.18,.55); b=rand(.20,.57)
        p=cubic((notch,cy),[((notch-.12,cy-.09),(-w-.15,.23),(-w,.10)),((-w+.10,-.09),(w+.11,-.04),(w,.17)),((w-.01,.28),(notch+.15,cy-.12),(notch,cy)),((notch-.19,cy+.13),(-b-.19,.82),(-b,.93)),((-b+.07,1.09),(b+.17,1.02),(b,.82)),((b-.02,.73),(notch+.08,cy+.05),(notch+norm(.065),cy+norm(.065)))])
        if style==6: p[:,:,0]*=-1
        return [p]
    return [cubic((xp,yp),[((-w,-.05),(-w-.21,.25),(-.16,cy-.04)),((.20,cy+.09),(b+.12,.68),(b,.85)),((b-.04,1.03),(-b,1.10),(-b,.87)),((-b,.65),(-.14,cy+.05),(notch,cy)),((.29,cy-.15),(w+.20,.07),(xp+.04,.00))])]

def nine(style):
    m=rand(.38,.65); r=rand(.18,.43); e=rand(-.02,.47); tipx=e-rand(.12,.48)
    loop=cubic((r,.17),[((r-.04,-.045),(-.26,-.08),(-.44,.16)),((-.76,m-.03),(-.32,m+.14),(-.06,m)),((.12,m-.05),(r,.25),(r,.13))])
    if style==0:
        s=path([(r,.13),(r+.01,m),(e,1)]); return [np.concatenate((loop,s),axis=1)]
    if style==1:
        s=cubic((r,.13),[((r+.02,.39),(e+.12,.75),(e,.90)),((e-.04,1.02),(tipx+.10,1.02),(tipx,.95+norm(.065)))])
        return [np.concatenate((loop,s),axis=1)]
    if style==2:
        return [ellipse(-.07,m*.5,.42,m*.5,-.55*pi,1.44*pi,rand(.8,1.22)),path([(r,rand(-.04,.14)),(r+.03,m),(e,1)])]
    if style==3:
        return [cubic((r-.13,.085),[((-.19,-.10),(-.53,.12),(-.49,.32)),((-.53,m+.16),(.04,m+.10),(r,.15)),((r+.07,-.04),(r+.05,.16),(r+.04,.35)),((r+.02,.53),(e+.04,.81),(e,1))])]
    if style==4:
        return [cubic((r,.16),[((r-.34,-.17),(-.52,.13),(-.47,m-.07)),((-.41,m+.19),(.10,m+.04),(r,.10)),((r+.20,.34),(e+.11,.77),(e,1))])]
    if style in (7,8,9):
        # A compact ring may sit on either side of an angled downstroke. Let
        # its size and eccentricity vary independently of the long descender.
        height=rand(.24,.58); width=rand(.34,.96); ex=rand(-.53,.40); start=rand(-1.3,-.4)
        ring=ellipse(0,height*.52,width*.50,height*.52,start,start+rand(1.80,2.12)*pi,rand(.8,1.20))
        ring[:,:,0]+=(ring[:,:,1]-height[:,None]/2)*norm(.50)[:,None]
        topx=width*.42; topy=height*rand(.03,.36); bend=rand(-.10,.15)
        if style==7: desc=path([(topx,topy),(.5*(topx+ex)+bend,.56,.9),(ex,1)])
        elif style==8: desc=cubic((topx,topy),[((topx+.03,.42),(ex-.30,.82),(ex,1))])
        else: desc=cubic((topx,topy),[((topx+.07,.43),(ex+.18,.81),(ex,.92)),((ex-.11,1.02),(ex-.24,1.02),(ex-.32,.90))])
        return [ring,desc]
    if style==5:
        # Circle with a sweeping tangent continuing the right edge of the loop.
        e=rand(-.81,-.10); s=cubic((r,.13),[((r+.04,.41),(e+.07,.84),(e,1))]); return [loop,s]
    # The pen can lift before reaching the top of the loop, leaving an aperture.
    gap=rand(-.11,.22); a=cubic((r-gap,.07),[((-.06,-.10),(-.52,.03),(-.46,.29)),((-.51,m+.18),(.18,m+.08),(r,.12))]);
    return [a,path([(r,.12),(r+norm(.055),.32,.8),(e-.15,1)])]


def draw(digit,style):
    if digit==0: strokes=zero(style)
    else: strokes=[None,one,two,three,four,five,six,seven,eight,nine][digit](style)
    z=np.concatenate(strokes,axis=1)
    x=z[:,:,0]; y=z[:,:,1]; x=x-(x.max(1)+x.min(1))[:,None]/2; y=y-(y.max(1)+y.min(1))[:,None]/2
    # Smooth distortions, correlated between pen strokes, and axial rotations.
    def n(s): return norm(s)[:,None]
    def u(a,b): return rand(a,b)[:,None]
    ax=1+x*n(.20)+y*n(.28)+y*y*n(.40); ay=1+x*n(.24)+y*n(.18)
    xx=ax*x+n(.32)*(y*y-.12)+n(.040)*np.cos(y*5+u(-pi,pi))+n(.023)*np.sin(x*4+y*5+u(-pi,pi))
    yy=ay*y+n(.17)*x*x+n(.032)*np.sin(x*4+u(-pi,pi))+n(.018)*np.cos(y*5-x*4+u(-pi,pi))
    basewidth=[.66,.76,.81,.79,.78,.80,.71,.79,.71,.74][digit]
    xx*=basewidth*np.exp(n(.22)); yy+=n(.19)*xx
    xx+=(n(.23)-.13)*yy
    scale=1/np.maximum(np.ptp(xx,axis=1),np.ptp(yy,axis=1))[:,None]
    rad=u(.039,.108)*np.exp(n(.09)); rad*=scale**.25
    xx*=scale; yy*=scale
    xx-=.5*(xx.max(1)+xx.min(1))[:,None]; yy-=.5*(yy.max(1)+yy.min(1))[:,None]
    scale2=21.0/(1+2*rad)
    xx=13.5+xx*scale2; yy=13.5+yy*scale2
    # Rasterize by splatting small anti-aliased discs along each path. Radius
    # variation and an elliptical nib model changes of pressure and pen angle.
    rad=rad*scale2; dx=u(.85,1.18); a=u(-pi,pi)
    rr=rad*(1+n(.10)*np.sin(yy*.17+a)+n(.07)*np.cos(xx*.28-a))
    res=np.zeros((B,G*G),F)
    off=np.arange(B)[:,None]*G*G
    bx=np.floor(xx).astype(np.int32); by=np.floor(yy).astype(np.int32)
    for k in range(-3,5):
        py=by+k
        for j in range(-3,5):
            px=bx+j
            dist=np.sqrt(((py-yy)*dx)**2+((px-xx)/dx)**2)
            ink=np.clip(.5+rr-dist,0,1).astype(F)
            idx=off+np.clip(py,0,G-1)*G+np.clip(px,0,G-1)
            np.maximum.at(res.ravel(),idx.ravel(),ink.ravel())
    return res.reshape(B,G,G)

# Included verbatim in the recognizer; shared by both sides of the comparison.
features=r'''
import numpy as np

def samplegrid(a,xx,yy):
    n=len(a)
    xx=np.clip(xx,0,26.99999); yy=np.clip(yy,0,26.99999)
    i=xx.astype(np.int32); j=yy.astype(np.int32)
    u=(xx-i).astype(np.float32); v=(yy-j).astype(np.float32)
    b=np.arange(n)[:,None,None]
    return (1-v)*((1-u)*a[b,j,i]+u*a[b,j,i+1])+v*((1-u)*a[b,j+1,i]+u*a[b,j+1,i+1])

coord=np.arange(28,dtype=np.float32)
ux=(coord[None,None,:]-13.5)
uy=(coord[None,:,None]-13.5)

def canonical(a):
    mass=np.maximum(a.sum((1,2),keepdims=True),.001)
    mx=(a*ux).sum((1,2),keepdims=True)/mass; my=(a*uy).sum((1,2),keepdims=True)/mass
    dx=ux-mx; dy=uy-my
    slant=(a*dx*dy).sum((1,2),keepdims=True)/np.maximum((a*dy*dy).sum((1,2),keepdims=True),.01)
    slant=np.clip(slant,-1.4,1.4)
    ds=dx-slant*dy
    sx=np.maximum(np.sqrt((a*ds*ds).sum((1,2),keepdims=True)/mass),.6)
    sy=np.maximum(np.sqrt((a*dy*dy).sum((1,2),keepdims=True)/mass),.6)
    # Only partial width normalization: a narrow one must stay narrow.
    fx=np.clip((sx/4.1)**.73,.70,1.60); fy=np.clip((sy/5.8)**.80,.66,1.50)
    y=uy*fy; x=ux*fx+slant*y
    c=samplegrid(a,13.5+mx+x,13.5+my+y)
    return c,np.log(sx/sy).reshape(-1),slant.reshape(-1)

def blur(a):
    b=np.pad(a,((0,0),(1,1),(1,1)),mode='constant'); c=(b[:,:-2]+2*b[:,1:-1]+b[:,2:])*.25
    return (c[:,:,:-2]+2*c[:,:,1:-1]+c[:,:,2:])*.25

def orientations(a):
    a=blur(a); p=np.pad(a,((0,0),(1,1),(1,1)),mode='constant')
    gx=(p[:,1:-1,2:]-p[:,1:-1,:-2]); gy=(p[:,2:,1:-1]-p[:,:-2,1:-1]); mag=np.sqrt(gx*gx+gy*gy+.0000001)
    res=[]
    for c,s in [(1,0),(.7071068,.7071068),(0,1),(-.7071068,.7071068)]:
        t=gx*c+gy*s
        res.append(np.maximum(0,2*t*t/mag-mag))
    return res

def unit(v):
    return v/np.maximum(.00001,np.sqrt((v*v).sum(tuple(range(1,v.ndim)),keepdims=True)))

def cells(a):
    # Gaussian integration windows on a regular lattice of stroke locations.
    return (cellmat @ a) @ cellmat.T

cellpos=np.arange(6,dtype=np.float32)*4+3.5
cellmat=np.exp(-.5*((cellpos[:,None]-coord)/2.4)**2).astype(np.float32)
cellmat/=cellmat.sum(1,keepdims=True)

def cavities(c):
    # Escape cost of the background: the lowest ink barrier on a path from
    # this pixel to the border. In an enclosed counter the barrier is high.
    # Alternating row/column scans propagate around open ends of long strokes.
    d=np.ones_like(c); d[:,0,:]=c[:,0,:]; d[:,-1,:]=c[:,-1,:]
    d[:,:,0]=c[:,:,0]; d[:,:,-1]=c[:,:,-1]
    for sweep in range(2):
        for i in range(1,28):
            d[:,i,:]=np.minimum(d[:,i,:],np.maximum(c[:,i,:],d[:,i-1,:]))
            d[:,:,i]=np.minimum(d[:,:,i],np.maximum(c[:,:,i],d[:,:,i-1]))
        for i in range(26,-1,-1):
            d[:,i,:]=np.minimum(d[:,i,:],np.maximum(c[:,i,:],d[:,i+1,:]))
            d[:,:,i]=np.minimum(d[:,:,i],np.maximum(c[:,:,i],d[:,:,i+1]))
    return np.maximum(0,d-c)

def topology(c):
    # Thin the strokes by deleting only simple boundary points (the two
    # directional passes of Zhang-Suen). A loop then has no stroke ends; a
    # line has two, and the protruding arms of a crossing have their own ends.
    a=np.pad(c>.32,((0,0),(1,1),(1,1)),mode='constant')
    def neighbors():
        return (a[:,:-2,1:-1],a[:,:-2,2:],a[:,1:-1,2:],a[:,2:,2:],a[:,2:,1:-1],a[:,2:,:-2],a[:,1:-1,:-2],a[:,:-2,:-2])
    def crossings(ns):
        t=np.zeros_like(ns[0],dtype=np.uint8)
        for i in range(8): t+=(~ns[i-1])&ns[i]
        return t
    for sweep in range(12):
        changed=False
        for step in range(2):
            ns=neighbors(); north,ne,east,se,south,sw,west,nw=ns
            deg=np.add.reduce(ns,dtype=np.uint8); tr=crossings(ns)
            if step==0: keep=(north&east&south)|(east&south&west)
            else: keep=(north&east&west)|(north&south&west)
            remove=a[:,1:-1,1:-1] & (deg>=2) & (deg<=6) & (tr==1) & ~keep
            if np.any(remove):
                changed=True; a[:,1:-1,1:-1]&=~remove
        if not changed: break
    ns=neighbors(); deg=np.add.reduce(ns,dtype=np.uint8); tr=crossings(ns)
    thin=a[:,1:-1,1:-1]; tip=thin & (deg==1); branch=thin & (tr>=3)
    maps=[]
    for target in [tip,branch]:
        v=topwin @ target.astype(np.float32) @ topwin.T
        maps.append(np.minimum(1.,v).reshape(len(c),-1))
    return np.rint(np.concatenate(maps,axis=1)*127).astype(np.uint8)

topwin=np.exp(-.5*((np.arange(5,dtype=np.float32)[:,None]*5+3.5-coord)/3.5)**2)

def coarse(c,aspect,slant,holes):
    v=unit(cells(c)).reshape(len(c),-1)
    h=np.stack([cells(o) for o in orientations(c)],axis=1)
    h=unit(np.sqrt(np.maximum(h,0)+.008)-.08944272).reshape(len(c),-1)
    # The small RBF embeddings preserve width/slant without scalar cross terms.
    ar=np.exp(-.5*((aspect[:,None]-np.linspace(-2.4,.8,10,dtype=np.float32))/.35)**2)
    sk=np.exp(-.5*((slant[:,None]-np.linspace(-1.6,1.6,8,dtype=np.float32))/.45)**2)
    hh=cells(holes)*1.4
    cav=unit(np.concatenate((hh.reshape(len(c),-1),np.ones((len(c),1),dtype=np.float32)),axis=1))
    v=np.concatenate((.59*v,.76*h,.235*unit(ar),.12*unit(sk),.21*cav),axis=1)
    return np.rint(v*2048).astype(np.int16)

def pool2(a):
    return (a[:,0::2,0::2]+a[:,1::2,0::2]+a[:,0::2,1::2]+a[:,1::2,1::2])*.25

def fine(c,holes):
    # All maps share the same displacement in the elastic comparison. Opposite
    # directions share a bin, but different orientation axes are independent.
    o=orientations(c); maps=[pool2(blur(c))]+[pool2(blur(q))*1.75 for q in o]+[.75*pool2(blur(holes))]
    return np.rint(np.clip(np.stack(maps,axis=-1),0,1)*31).astype(np.uint32)

powers=np.array([1,32,1024,32768,1048576,33554432],dtype=np.uint32)
'''

inference=r'''
lookup=CONSTANTS['coarse'].astype(np.float32).T.copy()
low=CONSTANTS['low'].astype(np.uint32)
high=CONSTANTS['high'].astype(np.uint32)
detail=low+(high<<15)
del low,high
ends=CONSTANTS['topology']
offsets=np.arange(6,dtype=np.uint32)*5
perdigit=detail.shape[0]//10

# A 3 by 3 lattice of overlapping Gaussian patches. Patch offsets have a
# displacement prior and must mostly agree with neighboring patches.
pg=np.arange(14,dtype=np.float32)
pm=np.exp(-.5*((pg[None,:]-np.array([2.8,6.5,10.2],dtype=np.float32)[:,None])/2.15)**2)
patch=(pm[:,None,:,None]*pm[None,:,None,:]).reshape(9,196)
patch/=np.maximum(patch.sum(0,keepdims=True),.22)
patch=patch.T.copy()

def elastic(q,idx):
    # Work on just one writing sample at a time in this more detailed stage.
    t=((detail[idx][:,:,:,None] >> offsets) & 31).astype(np.float32)/31
    q=q.astype(np.float32)/31
    quilt=np.pad(q,((1,1),(1,1),(0,0)),mode='constant')
    many=len(idx); loc=[]
    for y in range(3):
        for x in range(3):
            loc.append(quilt[y:y+14,x:x+14].reshape(196,6))
    loc=np.stack(loc,axis=1)
    # SSD for vector-valued cells. The matrix operations are over templates,
    # not different input samples. Distances have symmetric ink normalization.
    a=t.reshape(many,196,6)
    dots=np.matmul(a.transpose(1,0,2),loc.transpose(0,2,1)).transpose(1,2,0)
    qm=(loc*loc).sum(2).T; tm=(a*a).sum(2)
    cost=(tm[:,None,:]+qm[None,:,:]-2*dots) @ patch
    mass=(tm+qm[4]) @ patch
    shift=np.array([2,1,2,1,0,1,2,1,2],dtype=np.float32)
    cost+=shift[None,:,None]*(.034*mass[:,None,:]+.045)
    # Smooth displacement field, via min-sum on a tree rooted in the centre.
    coords=np.arange(9); sy=coords//3; sx=coords%3
    smooth=np.minimum(3,(sy[:,None]-sy)**2+(sx[:,None]-sx)**2).astype(np.float32)
    for child,parent in [(0,1),(2,1),(6,7),(8,7),(1,4),(3,4),(5,4),(7,4)]:
        w=.025*np.minimum(mass[:,child],mass[:,parent])+.032
        cost[:,:,parent]+=np.min(cost[:,:,child,None]+w[:,None,None]*smooth,axis=1)
    return np.min(cost[:,:,4],axis=1)/np.maximum(1,mass.sum(1))

def predict(x):
    n=len(x); out=np.empty(n,dtype=np.int64); k=16
    for start in range(0,n,64):
        a=x[start:start+64,0]; m=len(a)
        # Fixed matrix shape and exact integer-valued coarse features make the
        # retrieval independent of the position of the end of the caller's batch.
        c,aspect,slant=canonical(a)
        holes=cavities(c)
        v=coarse(c,aspect,slant,holes); block=np.zeros((64,v.shape[1]),np.float32); block[:m]=v
        score=(block @ lookup)[:m].reshape(m,10,perdigit)
        ix=np.argpartition(score,perdigit-k,axis=2)[:,:,-k:]
        cs=np.take_along_axis(score,ix,axis=2)/4194304
        ids=ix+perdigit*np.arange(10)[None,:,None]
        qs=fine(c,holes); ep=topology(c).astype(np.float32)
        for j in range(m):
            err=elastic(qs[j],ids[j].reshape(-1)).reshape(10,k)
            # A bounded extra cost: stroke ends provide useful evidence, but
            # digitization can introduce a false spur or bridge a narrow gap.
            diff=(ends[ids[j]].astype(np.float32)-ep[j])/127
            delta=np.minimum(.65,np.abs(diff))**2
            extra=.014*delta[:,:,:25].sum(2)+.007*delta[:,:,25:].sum(2)
            cost=err-.43*cs[j]+np.minimum(extra,.070)
            best=np.sort(cost,axis=1)
            # Two independent plausible stroke renderings must support the match.
            out[start+j]=np.argmin(.77*best[:,0]+.23*best[:,1])
    return out
'''

def main():
    folder=Path(sys.argv[1]); folder.mkdir(parents=True,exist_ok=True)
    (folder/'solution.py').write_text(features+inference)
    shared={}; exec(features,shared)
    feats=[]; lo=[]; hi=[]; topo=[]
    styles=[[0,0,0,1,2],[0,0,0,0,0,0,0,0,1,1,2,3],[0,0,1,2,3,4,5,5,6,7,7,8],[0,0,1,1,2,3],
            [0,0,0,1,1,2,3,4],[0,1,2,2,3,4,5,5],[0,0,1,2,3,4],
            [0,0,1,1,2,3,4,4,5,6],[0,0,1,2,3,4,5,6],[0,0,1,1,2,3,4,5,6,7,7,8,9]]
    for d in range(10):
        for k in range(256):
            im=draw(d,styles[d][k%len(styles[d])]); c,ar,sk=shared['canonical'](im); holes=shared['cavities'](c)
            feats.append(shared['coarse'](c,ar,sk,holes)); f=shared['fine'](c,holes); topo.append(shared['topology'](c))
            packed=np.sum(f*shared['powers'],axis=-1,dtype=np.uint32)
            lo.append((packed%32768).astype(np.uint16)); hi.append((packed>>15).astype(np.uint16))
    np.savez(folder/'constants.npz',coarse=np.concatenate(feats),low=np.concatenate(lo),high=np.concatenate(hi),topology=np.concatenate(topo))

if __name__=='__main__': main()
