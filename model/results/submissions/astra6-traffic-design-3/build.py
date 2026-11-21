"""Data-free, analytic traffic sign recognizer.

All geometry below is an idealized drawing in sign-centred coordinates, not a
measurement or tracing of a photograph. There is no fitting, optimization,
font file, asset file, random example, or data access. Only numpy and the
standard library are used. Run: python3 build.py DESTINATION.
"""
import json
import sys
from pathlib import Path
import numpy as np

# Drawing functions return geometric coverage masks. Coordinates refer to an
# abstract sign whose outer circumradius is one, with downward-positive y.
def stroke(x, y, points, width):
    d = np.full_like(x, 100.0)
    for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
        vx, vy = bx-ax, by-ay
        t = np.clip(((x-ax)*vx+(y-ay)*vy)/(vx*vx+vy*vy+1e-12), 0, 1)
        d = np.minimum(d, (x-ax-t*vx)**2+(y-ay-t*vy)**2)
    return d <= (width/2)**2

def polygon(x, y, pts):
    inside = np.zeros_like(x, dtype=bool)
    for (ax, ay), (bx, by) in zip(pts, pts[1:]+pts[:1]):
        inside ^= ((ay > y) != (by > y)) & (x < (bx-ax)*(y-ay)/(by-ay+1e-15)+ax)
    return inside

def ellipse(x, y, cx, cy, rx, ry):
    return ((x-cx)/rx)**2+((y-cy)/ry)**2 <= 1

def rect(x, y, x0, y0, x1, y1):
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

def arc(x, y, cx, cy, rx, ry, start, end, width):
    a = np.linspace(start, end, 25)
    pts = list(zip(cx+rx*np.cos(a),cy+ry*np.sin(a)))
    return stroke(x,y,pts,width)

def digit(x,y,d,w):
    # A rounded, narrow, monoline road-sign alphabet, in a unit-height cell.
    if d == '0':
        return (arc(x,y,0,-.20,.25,.25,np.pi,2*np.pi,w) |
                arc(x,y,0,.20,.25,.25,0,np.pi,w) |
                stroke(x,y,[(-.25,-.20),(-.25,.20)],w) |
                stroke(x,y,[(.25,-.20),(.25,.20)],w))
    if d == '1':
        return stroke(x,y,[(-.17,-.28),(.04,-.45),(.04,.45)],w)
    if d == '2':
        return (arc(x,y,0,-.23,.26,.22,np.pi,2*np.pi,w) |
                stroke(x,y,[(.26,-.23),(.22,-.10),(-.24,.36),(-.25,.44),(.27,.44)],w))
    if d == '3':
        return (arc(x,y,-.02,-.23,.27,.22,-np.pi/2,np.pi/2,w) |
                arc(x,y,-.02,.22,.28,.23,-np.pi/2,np.pi/2+.3,w) |
                stroke(x,y,[(-.23,-.41),(-.02,-.45)],w) |
                stroke(x,y,[(-.23,.40),(-.02,.45)],w))
    if d == '5':
        return (stroke(x,y,[(.25,-.44),(-.23,-.44),(-.25,-.04),(.02,-.05)],w) |
                arc(x,y,-.01,.20,.28,.25,-np.pi/2,np.pi*.85,w))
    if d == '6':
        return (arc(x,y,.035,-.12,.28,.33,np.pi*.95,np.pi*1.8,w) |
                stroke(x,y,[(-.245,-.12),(-.25,.20)],w) |
                arc(x,y,0,.20,.25,.245,0,2*np.pi,w))
    if d == '7':
        return stroke(x,y,[(-.28,-.44),(.27,-.44),(-.12,.45)],w)
    if d == '8':
        return (arc(x,y,0,-.24,.245,.21,0,2*np.pi,w) |
                arc(x,y,0,.215,.27,.235,0,2*np.pi,w))
    raise ValueError(d)

def number(x,y,text,style):
    n=len(text)
    height = 1.02 if n==2 else .94
    advance = .62 if n==2 else .42
    widthscale = .80 if n==2 else .64
    widthscale *= (0.94 if style==0 else 1.06)
    w = .14 if style==0 else .18
    out=np.zeros_like(x,dtype=bool)
    for j,d in enumerate(text):
        cx=(j-(n-1)/2)*advance
        out |= digit((x-cx)/(height*widthscale), y/height, d, w)
    return out

def wheel(x,y,cx,cy,r):
    return ellipse(x,y,cx,cy,r,r)

def car_front(x,y,cx=0,cy=0,s=1):
    x=(x-cx)/s; y=(y-cy)/s
    body=polygon(x,y,[(-.24,-.25),(.24,-.25),(.34,.02),(.34,.27),(-.34,.27),(-.34,.02)])
    window=polygon(x,y,[(-.18,-.18),(.18,-.18),(.24,-.015),(-.24,-.015)])
    body &= ~window
    body |= rect(x,y,-.29,.23,-.18,.35)|rect(x,y,.18,.23,.29,.35)
    return body

def truck_front(x,y,cx=0,cy=0,s=1):
    x=(x-cx)/s; y=(y-cy)/s
    body=rect(x,y,-.30,-.42,.30,.26)
    body &= ~rect(x,y,-.23,-.13,.23,.04)
    body |= rect(x,y,-.26,.23,-.15,.36)|rect(x,y,.15,.23,.26,.36)
    return body

def truck_side(x,y):
    body=rect(x,y,-.58,-.28,.19,.19)
    body |= polygon(x,y,[(.19,-.17),(.42,-.17),(.57,.02),(.57,.22),(.17,.22)])
    body &= ~polygon(x,y,[(.26,-.11),(.39,-.11),(.47,.01),(.26,.01)])
    return body|wheel(x,y,-.34,.24,.12)|wheel(x,y,.35,.24,.12)

def person(x,y,cx=0,cy=0,s=1,mirror=1):
    x=(x-cx)/s*mirror; y=(y-cy)/s
    return (ellipse(x,y,.06,-.38,.095,.10) |
            stroke(x,y,[(.02,-.23),(-.04,.05),(-.20,.34)],.105) |
            stroke(x,y,[(-.04,.04),(.13,.15),(.25,.34)],.10) |
            stroke(x,y,[(.02,-.18),(-.17,-.03),(-.31,-.08)],.075) |
            stroke(x,y,[(.01,-.15),(.20,-.05),(.28,-.15)],.075))

def arrow(x,y,points,width=.15,head=.29):
    a,b=points[-2],points[-1]
    vx,vy=b[0]-a[0],b[1]-a[1]
    length=np.hypot(vx,vy); vx/=length; vy/=length
    tip=np.array(b); base=tip-np.array([vx,vy])*head
    p=[tuple(tip),tuple(base+np.array([-vy,vx])*head*.75),tuple(base-np.array([-vy,vx])*head*.75)]
    return stroke(x,y,points,width)|polygon(x,y,p)

def blue_symbol(x,y,k):
    if k in (33,34):
        if k==34: x=-x
        return arrow(x,y,[(0,.58),(0,-.16),(.16,-.32),(.61,-.32)],.16,.30)
    if k==35:
        return arrow(x,y,[(0,.61),(0,-.64)],.16,.32)
    if k in (36,37):
        if k==37: x=-x
        return (arrow(x,y,[(0,.62),(0,-.63)],.14,.28)|
                arrow(x,y,[(0,.10),(.18,-.12),(.62,-.12)],.14,.27))
    if k in (38,39):
        if k==39: x=-x
        return arrow(x,y,[(-.42,-.50),(.43,.48)],.21,.39)
    out=np.zeros_like(x,dtype=bool)
    for a in (-np.pi/2,np.pi/6,5*np.pi/6):
        angles=np.linspace(a,a+1.25,16)
        pts=list(zip(.48*np.cos(angles),.48*np.sin(angles)))
        out |= arrow(x,y,pts,.12,.23)
    return out

def warning_symbol(x,y,k,style):
    # The available symbol area is the interior of an equilateral triangle.
    # Transform its abstract drawing cell to sit around the triangle centroid.
    x=x/.90; y=(y+.04)/.82
    w=.105 if style==0 else .125
    if k==11:
        return (arrow(x,y,[(0,.45),(0,-.57)],.18,.23) |
                stroke(x,y,[(-.37,.02),(.37,.02)],.12))
    if k==18:
        return polygon(x,y,[(-.11,-.55),(.11,-.55),(.065,.18),(-.065,.18)]) | ellipse(x,y,0,.37,.095,.095)
    if k in (19,20):
        if k==20: x=-x
        return arrow(x,y,[(.19,.45),(.19,.02),(.10,-.18),(-.06,-.27),(-.42,-.27)],.19,.28)
    if k==21:
        return arrow(x,y,[(.24,.45),(.24,.13),(.14,.02),(-.13,-.02),(-.24,-.15),(-.24,-.38)],.17,.25)
    if k==22:
        return (rect(x,y,-.65,.31,.65,.45) |
                ellipse(x,y,-.31,.34,.25,.24) | ellipse(x,y,.28,.34,.25,.24))
    if k==23:
        c=car_front(x,y,0,-.11,.72)
        c |= stroke(x,y,[(-.17,.15),(-.34,.29),(-.22,.40),(-.40,.52)],.07)
        c |= stroke(x,y,[(.16,.15),(.02,.29),(.17,.40),(.02,.52)],.07)
        return c
    if k==24:
        return (stroke(x,y,[(-.24,.46),(-.24,-.45)],.15) |
                stroke(x,y,[(.35,.46),(.35,.16),(.12,-.15),(.12,-.45)],.15))
    if k==25:
        return (ellipse(x,y,-.06,-.37,.095,.10) |
                stroke(x,y,[(-.09,-.21),(-.28,.04),(-.48,.37)],.13) |
                stroke(x,y,[(-.25,.02),(-.05,.14),(-.10,.41)],.12) |
                stroke(x,y,[(-.08,-.20),(.18,-.04),(.37,.03)],.085) |
                stroke(x,y,[(.04,-.14),(.42,.36)],.045) |
                polygon(x,y,[(.22,.43),(.43,.19),(.67,.43)]))
    if k==26:
        return rect(x,y,-.16,-.59,.16,.46)
    if k==27:
        return person(x,y,0,-.01,1.0)|stroke(x,y,[(-.60,.47),(.57,.47)],.05)
    if k==28:
        return person(x,y,-.21,-.04,.87,-1)|person(x,y,.28,.12,.67,-1)
    if k==29:
        out=(arc(x,y,-.35,.23,.23,.23,0,2*np.pi,.055)|arc(x,y,.37,.23,.23,.23,0,2*np.pi,.055))
        out |= stroke(x,y,[(-.35,.23),(-.10,-.02),(.18,.23),(-.35,.23)],.045)
        out |= stroke(x,y,[(.18,.23),(.29,-.12),(.37,.23)],.045)
        out |= ellipse(x,y,.05,-.45,.085,.085)
        out |= stroke(x,y,[(.02,-.31),(-.16,-.15),(-.02,.02),(-.11,.24)],.09)
        out |= stroke(x,y,[(.02,-.29),(.17,-.15),(.32,-.15)],.065)
        return out
    if k==30:
        out=np.zeros_like(x,dtype=bool)
        for a in np.arange(6)*np.pi/3:
            ux,uy=np.cos(a),np.sin(a)
            out |= stroke(x,y,[(0,0),(.55*ux,.55*uy)],.055)
            for s in (-1,1):
                vx,vy=np.cos(a+s*.8),np.sin(a+s*.8)
                out |= stroke(x,y,[(.31*ux,.31*uy),(.31*ux-.18*vx,.31*uy-.18*vy)],.05)
        return out
    if k==31:
        out=ellipse(x,y,-.07,.04,.34,.14)
        out |= stroke(x,y,[(.19,.04),(.30,-.28),(.41,-.35)],.13)
        out |= ellipse(x,y,.43,-.35,.14,.08)
        out |= stroke(x,y,[(.33,-.39),(.30,-.57),(.22,-.64)],.035)
        out |= stroke(x,y,[(.36,-.40),(.46,-.56),(.44,-.65)],.035)
        out |= stroke(x,y,[(-.29,.09),(-.40,.29),(-.59,.36)],.065)
        out |= stroke(x,y,[(-.18,.13),(-.20,.32),(-.07,.40)],.06)
        out |= stroke(x,y,[(.16,.10),(.30,.23),(.55,.26)],.06)
        out |= stroke(x,y,[(.08,.12),(.19,.34),(.38,.40)],.055)
        out |= stroke(x,y,[(-.38,-.01),(-.51,-.12)],.055)
        return out
    raise ValueError(k)

def stop_text(x,y):
    text=np.zeros_like(x,dtype=bool)
    for j,c in enumerate('STOP'):
        xx=(x-(j-1.5)*.31)/.45; yy=y/.55
        if c=='S':
            t=(arc(xx,yy,0,-.23,.23,.22,np.pi/2,3*np.pi/2,.085)|
               arc(xx,yy,0,.21,.23,.22,-np.pi/2,np.pi/2,.085)|
               stroke(xx,yy,[(0,-.45),(.23,-.40)],.085)|
               stroke(xx,yy,[(-.23,.40),(0,.43)],.085))
        elif c=='T':
            t=stroke(xx,yy,[(-.26,-.44),(.26,-.44)],.085)|stroke(xx,yy,[(0,-.44),(0,.44)],.085)
        elif c=='O': t=arc(xx,yy,0,0,.23,.44,0,2*np.pi,.085)
        else:
            t=stroke(xx,yy,[(-.22,.44),(-.22,-.44),(.02,-.44)],.085)|arc(xx,yy,.02,-.20,.22,.24,-np.pi/2,np.pi/2,.085)|stroke(xx,yy,[(.02,.04),(-.22,.04)],.085)
        text |= t
    return text

RED=np.array([.70,.045,.035],np.float32)
BLUE=np.array([.035,.14,.65],np.float32)
WHITE=np.array([.92,.92,.92],np.float32)
BLACK=np.array([.035,.035,.035],np.float32)
YELLOW=np.array([.95,.78,.045],np.float32)
GRAY=np.array([.50,.50,.50],np.float32)
SPEED={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}
WARN={11,*range(18,32)}

def draw(k,x,y,style,symbol_scale=1.):
    r=np.hypot(x,y)
    sx=x/symbol_scale; sy=y/symbol_scale
    rgb=np.zeros((3,)+x.shape,dtype=np.float32)
    def paint(mask,c):
        rgb[:,mask]=c[:,None]
    if k in WARN:
        outer=[(0,-1),(.866,.5),(-.866,.5)]
        inner=[(0,-.72),(.624,.36),(-.624,.36)]
        support=polygon(x,y,outer)
        paint(support,RED); paint(polygon(x,y,inner),WHITE)
        paint(warning_symbol(sx,sy,k,style)&polygon(x,y,inner),BLACK)
        if k==26:
            for cy,col in ((-.10,RED),(.12,YELLOW),(.34,np.array([.05,.40,.14],np.float32))):
                paint(ellipse(x,y,0,cy,.075,.075),col)
    elif k==13:
        support=polygon(x,y,[(0,1),(-.866,-.5),(.866,-.5)])
        paint(support,RED)
        paint(polygon(x,y,[(0,.69),(-.598,-.345),(.598,-.345)]),WHITE)
    elif k==12:
        support=np.abs(x)+np.abs(y)<=1.15
        paint(support,WHITE)
        paint(np.abs(x)+np.abs(y)<=1.035,BLACK)
        paint(np.abs(x)+np.abs(y)<=.97,WHITE)
        paint(np.abs(x)+np.abs(y)<=.82,YELLOW)
    elif k==14:
        support=(np.maximum(np.abs(x),np.abs(y))<=.925)&(np.abs(x)+np.abs(y)<=1.31)
        paint(support,WHITE)
        red=(np.maximum(np.abs(x),np.abs(y))<=.865)&(np.abs(x)+np.abs(y)<=1.22)
        paint(red,RED); paint(stop_text(sx,sy)&red,WHITE)
    elif k>=33 and k<=40:
        support=r<=1
        paint(support,WHITE); paint(r<=.965,BLUE)
        paint(blue_symbol(sx,sy,k)&support,WHITE)
    elif k==17:
        support=r<=1
        paint(support,WHITE); paint(r<=.96,RED)
        paint(rect(x,y,-.77,-.18,.77,.18)&support,WHITE)
    else:
        support=r<=1
        paint(support,RED if k not in (6,32,41,42) else GRAY)
        paint(r<=.81,WHITE)
        if k in SPEED:
            paint(number(sx,sy,SPEED[k],style)&(r<.8),BLACK if k!=6 else GRAY)
        if k in (9,10,41,42):
            if k in (10,42):
                a=truck_front(x,y,-.34,0,.78)
            else: a=car_front(x,y,-.34,0,.78)
            b=car_front(x,y,.34,0,.78)
            paint(a,RED if k in (9,10) else GRAY)
            paint(b,BLACK if k in (9,10) else GRAY)
        if k==16: paint(truck_side(x,y),BLACK)
        if k in (6,32,41,42):
            # Cancellation: parallel diagonal stripes rising to the right.
            if k==32:
                stripes=np.abs(y+x)<.22
            else:
                stripes=np.zeros_like(x,dtype=bool)
                for a in (-.23,-.115,0,.115,.23):
                    stripes |= np.abs(y+x-a)<.027
            paint(stripes&support,BLACK)
    # A uniform version of the same field isolates the semantic symbol from
    # the enclosing shape. This is an analytic decomposition, not a fitted one.
    base=rgb.copy()
    inner=np.zeros_like(x,dtype=bool)
    if k in WARN:
        inner=polygon(x,y,[(0,-.72),(.624,.36),(-.624,.36)])
        base[:,inner]=WHITE[:,None]
    elif k in (0,1,2,3,4,5,6,7,8,9,10,15,16,32,41,42):
        inner=r<=.81
        base[:,inner]=WHITE[:,None]
    elif 33<=k<=40:
        inner=r<=.965
        base[:,inner]=BLUE[:,None]
    elif k==14:
        inner=(np.maximum(np.abs(x),np.abs(y))<=.865)&(np.abs(x)+np.abs(y)<=1.22)
        base[:,inner]=RED[:,None]
    elif k==17:
        inner=r<=.96
        base[:,inner]=RED[:,None]
    return rgb,support,base,inner

def down(a):
    return a.reshape(a.shape[:-2]+(32,2,32,2)).mean(axis=(-3,-1))

def contrast_kernel(signal,mask):
    area=mask.sum()+1e-8
    z=signal-mask*(signal.sum()/area)
    return z/(np.sqrt((z*z).sum()/area+1e-8)*area)

def template(k,x,y,style):
    # The triangle's bounding-box centre differs from its centroid.
    if k in WARN: y=.90*y-.25
    if k==13: y=.90*y+.25
    rgb,support,base,inner=draw(k,x,y,style)
    m=down(support.astype(np.float32)); inner=down(inner.astype(np.float32))
    rgb=down(rgb); base=down(base)
    lum=np.einsum('c,cyx->yx',np.array([.299,.587,.114]),rgb)
    baselum=np.einsum('c,cyx->yx',np.array([.299,.587,.114]),base)
    full=contrast_kernel(lum,m)
    shape=contrast_kernel(baselum,m)
    detail=contrast_kernel(lum-baselum,inner)
    red=base[0]-.5*base[1]-.5*base[2]
    blue=base[2]-.5*base[0]-.5*base[1]
    area=m.sum()+1e-8
    color_scale=np.sqrt((red*red+blue*blue).sum()/area+1e-8)*area
    # Detail receives its own normalization instead of competing with a large
    # white field in the same normalized correlation.
    gray=.28*full+.20*shape+.52*detail
    out=np.stack([gray,.30*red/color_scale,.30*blue/color_scale])
    return out.astype(np.float32)

class Network:
    def __init__(self): self.layers=[]; self.weights={}
    def layer(self,name,op,inputs,**settings):
        self.layers.append(dict(name=name,op=op,inputs=[inputs] if isinstance(inputs,str) else inputs,**settings))
        return name
    def linear(self,name,source,W,b=None,conv=False):
        W=np.asarray(W,dtype=np.float32)
        if b is None: b=np.zeros(W.shape[0],dtype=np.float32)
        self.weights[name+'.W']=W
        self.weights[name+'.b']=np.asarray(b,dtype=np.float32)
        if conv: return self.layer(name,'conv',source,out=W.shape[0],kernel=[W.shape[2],W.shape[3]],stride=1,padding=0)
        return self.layer(name,'dense',source,out=W.shape[0])

def grid(n):
    y,x=np.mgrid[:n,:n].astype(np.float32)
    return (x+.5)*2.2/n-1.1,(y+.5)*2.2/n-1.1

def sample_matrix(n,side,r,aspect,dx,dy,factor=1.,yoffset=0.):
    x,y=grid(n)
    x*=factor; y=y*factor+yoffset
    x=(15.5+dx+r*np.sqrt(aspect)*x+.5)*side/32-.5
    y=(15.5+dy+r/np.sqrt(aspect)*y+.5)*side/32-.5
    x=np.clip(x,0,side-1); y=np.clip(y,0,side-1)
    ix=np.floor(x).astype(int); iy=np.floor(y).astype(int)
    fx=x-ix; fy=y-iy
    mat=np.zeros((n*n,side*side),dtype=np.float32)
    rows=np.arange(n*n)
    for ox,oy in ((0,0),(1,0),(0,1),(1,1)):
        cols=np.minimum(iy+oy,side-1)*side+np.minimum(ix+ox,side-1)
        coeff=(fx if ox else 1-fx)*(fy if oy else 1-fy)
        np.add.at(mat,(rows,cols.ravel()),coeff.ravel())
    return mat

def local_normalize(a,mask):
    a=a.copy()
    a-=a[mask].mean()
    a[~mask]=0
    a/=np.sqrt(np.mean(a*a)+1e-5)
    return a

def canonical(k,n,angle,style,variant):
    # Supersample analytic geometry, never a photograph.
    x,y=grid(n*4)
    th=np.deg2rad(angle)
    u=np.cos(th)*x+np.sin(th)*y
    v=-np.sin(th)*x+np.cos(th)*y
    if k in WARN: v=.90*v-.25
    if k==13: v=.90*v+.25
    rgb,support,base,inner=draw(k,u,v,style,variant)
    bg=np.full_like(rgb,.50)
    bg[:,support]=rgb[:,support]
    basebg=np.full_like(base,.50)
    basebg[:,support]=base[:,support]
    def reduce(a):return a.reshape(3,n,4,n,4).mean(axis=(2,4))
    return reduce(bg),reduce(basebg)

def interpolate_plane(a,x,y):
    n=a.shape[0]
    xx=np.clip((x+1.1)*n/2.2-.5,0,n-1)
    yy=np.clip((y+1.1)*n/2.2-.5,0,n-1)
    ix=xx.astype(int); iy=yy.astype(int); fx=xx-ix; fy=yy-iy
    return ((1-fx)*(1-fy)*a[iy,ix]+fx*(1-fy)*a[iy,np.minimum(ix+1,n-1)]
            +(1-fx)*fy*a[np.minimum(iy+1,n-1),ix]+fx*fy*a[np.minimum(iy+1,n-1),np.minimum(ix+1,n-1)])

def detail_patch(gray):
    # Shared multiresolution interior representation: one circular field and
    # one smaller, downward-shifted field for the interior of a warning sign.
    x,y=grid(10)
    circle=interpolate_plane(gray,x*.70,y*.70)
    triangle=interpolate_plane(gray,x*.40,y*.40+.22)
    out=np.zeros((20,20),dtype=np.float32)
    out[:10,:10]=circle-circle.mean()
    out[10:,:10]=out[:10,:10]
    out[:10,10:]=triangle-triangle.mean()
    out[10:,10:]=out[:10,10:]
    return out/np.sqrt(np.mean(out*out)+1e-5)

def build(dest):
    net=Network()
    # Geometric sampling, with local (per pose) rather than global contrast.
    # The sample matrices implement bilinear interpolation and subtraction
    # of the local masked mean. Every row comes from a coordinate formula.
    radii=(8.,9.5,11.,12.5,14.)
    aspects=(.86,1.,1.16)
    shifts=(-2.,0.,2.)
    poses=[(r,a,dx,dy) for r in radii for a in aspects for dx in shifts for dy in shifts]
    P=len(poses)
    x,y=grid(20); mask20=(x*x+y*y<=1.05**2)
    x,y=grid(10); mask10=(x*x+y*y<=1.05**2)
    mask_rgb=np.tile(mask10,(2,2))
    WG=np.zeros((P*400,1024),dtype=np.float32)
    WC=np.zeros((P*400,3*256),dtype=np.float32)
    WD=np.zeros((P*400,256),dtype=np.float32)
    for p,pose in enumerate(poses):
        g=sample_matrix(20,32,*pose)
        g-=g[mask20.ravel()].mean(axis=0)
        g[~mask20.ravel()]=0
        WG[p*400:(p+1)*400]=g
        c=sample_matrix(10,16,*pose).reshape(10,10,256)
        tiles=np.zeros((20,20,3*256),dtype=np.float32)
        # Chromatic opponent coordinates; white and gray give exactly zero.
        for channel,amount in enumerate((1.,-.5,-.5)):
            tiles[:10,:10,channel*256:(channel+1)*256]=c*amount
        for channel,amount in enumerate((-.5,-.5,1.)):
            tiles[:10,10:,channel*256:(channel+1)*256]=c*amount
        tiles[10:,:10]=-tiles[:10,:10]
        tiles[10:,10:]=-tiles[:10,10:]
        tiles[~mask_rgb]=0
        WC[p*400:(p+1)*400]=tiles.reshape(400,768)
        dc=sample_matrix(10,16,*pose,factor=.70).reshape(10,10,256)
        dt=sample_matrix(10,16,*pose,factor=.40,yoffset=.22).reshape(10,10,256)
        dc-=dc.mean(axis=(0,1)); dt-=dt.mean(axis=(0,1))
        d=np.empty((20,20,256),dtype=np.float32)
        d[:10,:10]=dc; d[10:,:10]=dc
        d[:10,10:]=dt; d[10:,10:]=dt
        WD[p*400:(p+1)*400]=d.reshape(400,256)
    net.linear('luminance','image',np.array([.299,.587,.114]).reshape(1,3,1,1),conv=True)
    net.linear('sample_gray','luminance',WG)
    net.layer('gray_patches','reshape','sample_gray',shape=[P,20,20])
    net.layer('local_gray_normalized','norm','gray_patches')
    net.layer('gray_strip','reshape','local_gray_normalized',shape=[1,P*20,20])
    net.layer('color_downsample','avgpool','image',size=[2,2],stride=[2,2])
    net.linear('sample_color','color_downsample',WC)
    net.layer('color_patches','reshape','sample_color',shape=[P,20,20])
    net.layer('local_color_normalized','norm','color_patches')
    net.layer('color_strip','reshape','local_color_normalized',shape=[1,P*20,20])
    net.layer('gray_downsample','avgpool','luminance',size=[2,2],stride=[2,2])
    net.linear('sample_detail','gray_downsample',WD)
    net.layer('detail_patches','reshape','sample_detail',shape=[P,20,20])
    net.layer('local_detail_normalized','norm','detail_patches')
    net.layer('detail_strip','reshape','local_detail_normalized',shape=[1,P*20,20])
    # A blank field is a positive absence test, not an arbitrary class bias.
    # Extract its centred interior from the already normalized full patch and
    # take mean absolute contrast using ReLU(z)+ReLU(-z).
    x10,y10=grid(10)
    z=sample_matrix(10,20,32/2.2,1.,0.,0.,factor=.70)
    z-=z.mean(axis=0)
    net.weights['interior_extract.W']=z.reshape(100,1,20,20)
    net.weights['interior_extract.b']=np.zeros(100,dtype=np.float32)
    net.layer('interior_extract','conv','gray_strip',out=100,kernel=[20,20],stride=20,padding=0)
    net.layer('interior_positive','relu','interior_extract')
    net.linear('interior_negative','interior_extract',(-np.eye(100)).reshape(100,100,1,1),conv=True)
    net.layer('interior_negative_relu','relu','interior_negative')
    net.layer('interior_absolute','add',['interior_positive','interior_negative_relu'])
    # Common local normalization gives constant input energy; normalized dot
    # products are therefore equivalent to symmetric squared-distance scores.
    variants=[(angle,style,scale) for angle in (-7.,0.,7.) for style in (0,1)
              for scale in (.88,1.,1.12)]
    S=len(variants)
    TG=np.empty((43*S,1,20,20),dtype=np.float32)
    TC=np.empty_like(TG)
    TD=np.empty_like(TG)
    for k in range(43):
        for j,(angle,style,scale) in enumerate(variants):
            rgb,base=canonical(k,20,angle,style,scale)
            gray=np.einsum('c,cyx->yx',np.array([.299,.587,.114]),rgb)
            details=detail_patch(gray)
            gray=local_normalize(gray,mask20)
            rgb,_=canonical(k,10,angle,style,scale)
            col=np.empty((20,20),dtype=np.float32)
            col[:10,:10]=rgb[0]-.5*rgb[1]-.5*rgb[2]
            col[:10,10:]=rgb[2]-.5*rgb[0]-.5*rgb[1]
            col[10:,:10]=-col[:10,:10]; col[10:,10:]=-col[:10,10:]
            col=local_normalize(col,mask_rgb)
            # An empty field has no interior contrast. Reserve the same total
            # evidence weight for it, without inventing a symbol.
            no_detail=k in (12,13,15)
            achromatic=k in (6,32,41,42)
            gw,dw,cw=(.50,.50,0.) if achromatic else ((.60,0.,.40) if no_detail else (.30,.30,.40))
            TG[k*S+j,0]=gray*gw/400
            TC[k*S+j,0]=col*cw/400
            TD[k*S+j,0]=details*dw/400
    net.weights['gray_matches.W']=TG; net.weights['gray_matches.b']=np.zeros(43*S,dtype=np.float32)
    net.layer('gray_matches','conv','gray_strip',out=43*S,kernel=[20,20],stride=20,padding=0)
    net.weights['color_matches.W']=TC; net.weights['color_matches.b']=np.zeros(43*S,dtype=np.float32)
    net.layer('color_matches','conv','color_strip',out=43*S,kernel=[20,20],stride=20,padding=0)
    net.weights['detail_matches.W']=TD; net.weights['detail_matches.b']=np.zeros(43*S,dtype=np.float32)
    net.layer('detail_matches','conv','detail_strip',out=43*S,kernel=[20,20],stride=20,padding=0)
    absence=np.zeros((43*S,100,1,1),dtype=np.float32)
    absence[15*S:16*S,:,0,0]=-.60/100
    net.linear('blank_field_evidence','interior_absolute',absence,conv=True)
    net.layer('joint_matches','add',['gray_matches','color_matches','detail_matches','blank_field_evidence'])
    net.layer('hypothesis_grid','reshape','joint_matches',shape=[43,S*P,1])
    net.layer('best_hypothesis','maxpool','hypothesis_grid',size=[S*P,1],stride=[S*P,1])
    net.layer('class_scores','reshape','best_hypothesis',shape=[43])
    count=sum(a.size for a in net.weights.values())
    assert count<=120000000
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    (dest/'network.json').write_text(json.dumps({'layers':net.layers},indent=2)+'\n')
    np.savez(dest/'weights.npz',**net.weights)
    print('Wrote',len(net.layers),'layers;',count,'weights;',S*P,'analytic hypotheses per class.')

if __name__=='__main__':
    build(sys.argv[1])
