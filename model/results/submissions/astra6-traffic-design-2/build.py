"""Data-free traffic-sign network: analytic colours, geometry and hand-drawn symbols.
No files are read. No optimizer, examples, training statistics, or fitted parameters.
All coordinates below describe idealized road-sign artwork in radius-one units.
"""
import json
import sys
from pathlib import Path
import numpy as np

LAYERS = []
WEIGHTS = {}

def layer(name, op, inputs, **kw):
    LAYERS.append(dict(name=name, op=op, inputs=[inputs] if isinstance(inputs,str) else inputs, **kw))
    return name

def linear(name, source, W, b=None, conv=False, stride=1):
    W = np.asarray(W, dtype=np.float32)
    if b is None: b = np.zeros(W.shape[0],np.float32)
    WEIGHTS[name+'.W']=W
    WEIGHTS[name+'.b']=np.asarray(b,np.float32)
    if conv:
        return layer(name,'conv',source,out=len(W),kernel=list(W.shape[-2:]),stride=stride,padding=0)
    return layer(name,'dense',source,out=len(W))

N=16
SS=4
q=(np.arange(N*SS)+.5)/(N*SS)*1.8-.9
XX,YY=np.meshgrid(q,q)

def segment(x,y,a,b,width):
    dx,dy=b[0]-a[0],b[1]-a[1]
    t=np.clip(((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy+1e-20),0,1)
    return (x-a[0]-t*dx)**2+(y-a[1]-t*dy)**2 <= (width/2)**2

def poly(x,y,points):
    inside=np.zeros(x.shape,bool)
    for a,b in zip(points,points[1:]+points[:1]):
        cross=((a[1]>y)!=(b[1]>y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1]+1e-20)+a[0])
        inside ^= cross
    return inside

class Art:
    def __init__(self,x,y):
        self.x=x;self.y=y;self.ink=np.zeros(x.shape,np.float64)
    def put(self,m,value=1):self.ink=np.maximum(self.ink,m*value)
    def line(self,pts,w=.09,value=1):
        for a,b in zip(pts,pts[1:]):self.put(segment(self.x,self.y,a,b,w),value)
    def polygon(self,pts,value=1):self.put(poly(self.x,self.y,pts),value)
    def disk(self,c,r,value=1):self.put((self.x-c[0])**2+(self.y-c[1])**2<=r*r,value)
    def ellipse(self,c,rx,ry,w=None,value=1):
        d=((self.x-c[0])/rx)**2+((self.y-c[1])/ry)**2
        if w is None:self.put(d<=1,value)
        else:self.put((d<=1)&((((self.x-c[0])/max(rx-w,.001))**2+((self.y-c[1])/max(ry-w,.001))**2)>=1),value)
    def arc(self,c,rx,ry,start,end,w=.09):
        t=np.linspace(start,end,36)
        self.line(list(zip(c[0]+rx*np.cos(t),c[1]+ry*np.sin(t))),w)
    def arrow(self,pts,w=.12,head=.25):
        self.line(pts,w)
        a,b=np.array(pts[-2]),np.array(pts[-1]);d=b-a;d=d/np.linalg.norm(d)
        side=np.array([-d[1],d[0]])
        base=b-d*head
        self.polygon([tuple(b),tuple(base+side*head*.78),tuple(base-side*head*.78)])

# Skeletons for DIN-like numerals: generic typography, not digit crops.
def digit(a,d,cx,cy,w,h,stroke,style=0):
    X=a.x;Y=a.y
    z=Art((X-cx)/w,(Y-cy)/h)
    t=stroke/w
    # Work in coordinates of width one, height one; segment thickness is
    # rendered in physical coordinates below to avoid anisotropic pen widths.
    def ln(p):a.line([(cx+u*w,cy+v*h) for u,v in p],stroke)
    def arc(c,rx,ry,s,e):
        ts=np.linspace(s,e,40)
        ln([(c[0]+rx*np.cos(t),c[1]+ry*np.sin(t)) for t in ts])
    if d=='0':
        # Rounded rectangular zero, straight sides characteristic of road lettering.
        ln([(-.39,-.28),(-.39,.28)]);ln([(.39,-.28),(.39,.28)])
        arc((0,-.28),.39,.19,np.pi,2*np.pi)
        arc((0,.28),.39,.19,0,np.pi)
    elif d=='1':
        ln([(-.28,-.29),(.04,-.47),(.04,.47)])
        if style:ln([(-.19,.47),(.25,.47)])
    elif d=='2':
        arc((0,-.27),.39,.20,np.pi,2*np.pi+.48)
        ln([(.35,-.18),(.29,-.02),(-.38,.43),(-.38,.47),(.40,.47)])
    elif d=='3':
        arc((-.03,-.25),.40,.23,-np.pi*.86,np.pi*.55)
        arc((-.03,.23),.41,.24,-np.pi*.53,np.pi*.86)
        ln([(-.06,-.015),(.10,-.015)])
    elif d=='5':
        ln([(.37,-.47),(-.34,-.47),(-.38,-.035),(-.12,-.085)])
        arc((-.015,.21),.405,.26,-np.pi*.69,np.pi*.90)
    elif d=='6':
        arc((.01,.205),.385,.265,0,2*np.pi)
        arc((.02,-.02),.40,.45,np.pi*.90,np.pi*1.75)
    elif d=='7':
        ln([(-.41,-.47),(.41,-.47),(-.20,.47)])
    elif d=='8':
        arc((0,-.255),.35,.215,0,2*np.pi)
        arc((0,.225),.40,.245,0,2*np.pi)
    else:raise ValueError(d)

def number(a,text,scale=1,widthfactor=1,stroke=.083,style=0):
    h=.94*scale
    dw=(.47 if len(text)==2 else .32)*scale*widthfactor
    gap=.075*scale
    total=len(text)*dw+(len(text)-1)*gap
    for j,ch in enumerate(text):
        digit(a,ch,-total/2+dw/2+j*(dw+gap),.015,dw,h,stroke*scale,style)

def car(a,cx,cy,s=1,front=True,value=1):
    if front:
        a.polygon([(cx-.20*s,cy-.08*s),(cx-.14*s,cy-.24*s),(cx+.14*s,cy-.24*s),(cx+.20*s,cy-.08*s),(cx+.24*s,cy-.04*s),(cx+.24*s,cy+.15*s),(cx-.24*s,cy+.15*s),(cx-.24*s,cy-.04*s)],value)
        a.line([(cx-.17*s,cy+.10*s),(cx-.17*s,cy+.23*s)],.09*s,value)
        a.line([(cx+.17*s,cy+.10*s),(cx+.17*s,cy+.23*s)],.09*s,value)
        # Window is white, a hole in the dark silhouette.
        hole=poly(a.x,a.y,[(cx-.13*s,cy-.08*s),(cx-.10*s,cy-.18*s),(cx+.10*s,cy-.18*s),(cx+.13*s,cy-.08*s)])
        a.ink[hole]=0
    else:
        a.polygon([(cx-.37*s,cy+.10*s),(cx-.37*s,cy-.02*s),(cx-.22*s,cy-.05*s),(cx-.10*s,cy-.20*s),(cx+.17*s,cy-.20*s),(cx+.30*s,cy-.06*s),(cx+.39*s,cy-.01*s),(cx+.39*s,cy+.10*s)],value)
        a.disk((cx-.23*s,cy+.11*s),.09*s,value);a.disk((cx+.24*s,cy+.11*s),.09*s,value)

def truck(a,cx,cy,s=1,front=False,value=1):
    if front:
        a.polygon([(cx-.22*s,cy-.38*s),(cx+.22*s,cy-.38*s),(cx+.22*s,cy+.19*s),(cx-.22*s,cy+.19*s)],value)
        a.line([(cx-.15*s,cy+.12*s),(cx-.15*s,cy+.28*s)],.105*s,value)
        a.line([(cx+.15*s,cy+.12*s),(cx+.15*s,cy+.28*s)],.105*s,value)
        hole=(np.abs(a.x-cx)<.15*s)&(a.y>cy-.16*s)&(a.y<cy+.005*s)
        a.ink[hole]=0
    else:
        a.polygon([(cx-.52*s,cy-.25*s),(cx+.13*s,cy-.25*s),(cx+.13*s,cy-.14*s),(cx+.35*s,cy-.14*s),(cx+.48*s,cy+.015*s),(cx+.48*s,cy+.20*s),(cx-.52*s,cy+.20*s)],value)
        a.disk((cx-.32*s,cy+.20*s),.105*s,value);a.disk((cx+.31*s,cy+.20*s),.105*s,value)
        hole=poly(a.x,a.y,[(cx+.21*s,cy-.10*s),(cx+.31*s,cy-.10*s),(cx+.40*s,cy+.005*s),(cx+.21*s,cy+.005*s)])
        a.ink[hole]=0

def person(a,cx,cy,s=1,walk=1):
    a.disk((cx,cy-.27*s),.075*s)
    a.line([(cx-.015*s,cy-.17*s),(cx-.07*s,cy+.05*s)],.11*s)
    a.line([(cx-.015*s,cy-.13*s),(cx+.12*s,cy-.01*s),(cx+.23*s,cy-.02*s)],.06*s)
    a.line([(cx-.04*s,cy-.11*s),(cx-.18*s,cy+.02*s)],.065*s)
    a.line([(cx-.065*s,cy+.035*s),(cx+.11*s,cy+.22*s),(cx+.22*s,cy+.26*s)],.075*s)
    a.line([(cx-.065*s,cy+.035*s),(cx-.17*s,cy+.26*s),(cx-.24*s,cy+.28*s)],.075*s)

def bicycle(a,cx,cy,s=1):
    for u in [-.22,.22]:a.ellipse((cx+u*s,cy+.12*s),.145*s,.145*s,.035*s)
    a.line([(cx-.22*s,cy+.12*s),(cx-.04*s,cy-.10*s),(cx+.08*s,cy+.12*s),(cx-.22*s,cy+.12*s),(cx-.10*s,cy-.12*s)],.035*s)
    a.line([(cx+.08*s,cy+.12*s),(cx+.18*s,cy-.12*s),(cx+.22*s,cy+.12*s)],.04*s)
    a.line([(cx+.18*s,cy-.12*s),(cx+.27*s,cy-.12*s)],.045*s)
    a.line([(cx-.16*s,cy-.12*s),(cx-.05*s,cy-.12*s)],.045*s)

def artwork(k,scale,width,stroke,angle,style,dx=0.,dy=0.):
    th=np.deg2rad(angle)
    x=((XX-dx)*np.cos(th)+(YY-dy)*np.sin(th))/scale
    y=(-(XX-dx)*np.sin(th)+(YY-dy)*np.cos(th))/scale
    a=Art(x/width,y)
    # Black on white for restrictions and warnings; white on blue for mandates.
    positive = 33<=k<=40 or k in (14,17)
    speed={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}
    if k in speed:
        number(a,speed[k],stroke=stroke,style=style)
        if k==6:
            a.ink*=.45
            for off in [-.21,-.105,0,.105,.21]:a.put(np.abs(a.x+a.y-off)<.025,.80)
    elif k in (9,10,41,42):
        if k in (9,41):car(a,-.32,.015,.91,value=.75);car(a,.32,.015,.91)
        else:truck(a,-.30,.015,.87,True,value=.8);car(a,.33,.015,.85)
        if k in (41,42):
            a.ink*=.45
            for off in [-.21,-.105,0,.105,.21]:a.put(np.abs(a.x+a.y-off)<.025,.8)
    elif k==16:truck(a,0,0,1.13)
    elif k==17:a.polygon([(-.68,-.15),(.68,-.15),(.68,.15),(-.68,.15)])
    elif k==14:
        # Compact block-letter STOP, described by strokes and loops.
        ww=.23; h=.48
        for j,ch in enumerate('STOP'):
            cx=(j-1.5)*.30
            if ch=='S':
                a.line([(cx+.09,-.22),(cx-.07,-.22),(cx-.10,-.16),(cx-.09,-.04),(cx+.09,.04),(cx+.10,.16),(cx+.07,.22),(cx-.10,.22)],.045)
            elif ch=='T':a.line([(cx-.11,-.22),(cx+.11,-.22)],.045);a.line([(cx,-.22),(cx,.22)],.045)
            elif ch=='O':a.ellipse((cx,0),.11,.24,.045)
            elif ch=='P':
                a.line([(cx-.10,.22),(cx-.10,-.22),(cx+.05,-.22),(cx+.11,-.16),(cx+.11,-.05),(cx+.05,.015),(cx-.10,.015)],.045)
    elif k==32:
        for off in [-.22,-.11,0,.11,.22]:a.put(np.abs(a.x+a.y-off)<.030)
    elif k in (33,34):
        if k==34:a.x=-a.x
        # Bent arrow, right-angle path with a rounded elbow.
        a.line([(-.22,.59),(-.22,.03)],.15)
        t=np.linspace(np.pi,1.5*np.pi,24)
        a.line(list(zip(.04+.26*np.cos(t),.03+.26*np.sin(t))),.15)
        a.arrow([(.04,-.23),(.57,-.23)],.15,.32)
    elif k==35:a.arrow([(0,.59),(0,-.62)],.17,.37)
    elif k in (36,37):
        if k==37:a.x=-a.x
        a.arrow([(0,.58),(0,-.62)],.14,.31)
        a.arrow([(0,.12),(.23,-.08),(.61,-.08)],.14,.29)
    elif k in (38,39):
        if k==39:a.x=-a.x
        a.arrow([(-.43,-.46),(.49,.49)],.21,.43)
    elif k==40:
        for d in [-90,30,150]:
            t=np.deg2rad(np.linspace(d,d+82,25))
            pts=list(zip(.45*np.cos(t),.45*np.sin(t)))
            a.arrow(pts,.12,.23)
    elif k==11:
        a.line([(0,.51),(0,-.29)],.17)
        a.line([(-.36,.05),(.36,.05)],.13)
        a.polygon([(-.16,-.20),(0,-.43),(.16,-.20)])
    elif k==18:
        a.polygon([(-.065,-.38),(.065,-.38),(.042,.24),(-.042,.24)])
        a.disk((0,.43),.067)
    elif k in (19,20):
        if k==19:a.x=-a.x
        a.line([(-.14,.48),(-.14,.09)],.16)
        a.arc((.08,.09),.22,.25,np.pi,1.5*np.pi,.16)
        a.arrow([(.08,-.16),(.30,-.16)],.16,.20)
    elif k==21:
        # The class name permits either first-bend direction.
        if style:a.x=-a.x
        a.line([(.17,.49),(.17,.27),(-.17,.01),(-.17,-.16)],.15)
        a.arrow([(-.17,-.16),(.10,-.32)],.15,.22)
    elif k==22:
        a.line([(-.47,.47),(.47,.47)],.065)
        a.polygon([(-.46,.47),(-.36,.38),(-.28,.23),(-.19,.21),(-.10,.29),(-.02,.40),(.06,.39),(.14,.28),(.23,.21),(.31,.25),(.40,.40),(.46,.47)])
    elif k==23:
        # A tilted car over two separate serpentine skid tracks.
        ax,ay=a.x.copy(),a.y.copy()
        t=np.deg2rad(-22)
        a.x=ax*np.cos(t)+ay*np.sin(t)
        a.y=-ax*np.sin(t)+ay*np.cos(t)
        car(a,.0,-.015,.83)
        a.x=ax;a.y=ay
        a.line([(-.13,.19),(-.32,.28),(-.11,.37),(-.31,.49)],.065)
        a.line([(.14,.20),(-.02,.29),(.19,.39),(.02,.49)],.065)
    elif k==24:
        a.line([(-.22,.50),(-.22,-.28)],.10)
        a.line([(.25,.50),(.25,.12),(.09,-.10),(.09,-.28)],.10)
    elif k==25:
        a.disk((-.06,-.24),.078)
        a.line([(-.12,-.13),(.07,.035),(.18,.06)],.12)
        a.line([(-.105,-.10),(-.22,.16),(-.36,.45)],.085)
        a.line([(-.22,.16),(-.01,.22),(.04,.47)],.09)
        a.line([(.06,.02),(.19,.17),(.32,.40)],.055)
        a.polygon([(.12,.48),(.30,.31),(.47,.48)])
    elif k==26:
        a.polygon([(-.11,-.39),(.11,-.39),(.11,.49),(-.11,.49)])
        for v in [-.24,.055,.35]:
            a.polygon([(-.11,v-.08),(-.18,v-.10),(-.16,v+.06),(-.11,v+.10)])
            a.polygon([(.11,v-.08),(.18,v-.10),(.16,v+.06),(.11,v+.10)])
            m=(a.x*a.x+(a.y-v)**2)<.080**2
            a.ink[m]=.32 if v!=.055 else .10
    elif k==27:
        person(a,0,.11,1.04)
        for dx in [-.32,-.13,.06,.25]:a.line([(dx-.06,.47),(dx+.04,.39)],.035)
    elif k==28:
        person(a,-.14,.05,.93)
        person(a,.23,.24,.63)
        a.line([(-.01,.02),(.17,.13)],.05)
    elif k==29:
        bicycle(a,0,.19,1.32)
        # The standard cycle pictogram need not include a rider.
        if style:
            a.disk((.06,-.25),.067)
            a.line([(.015,-.14),(-.12,-.04),(.04,.15)],.075)
            a.line([(.015,-.14),(.19,-.04),(.23,.10)],.055)
            a.line([(-.10,-.04),(-.05,.13),(-.16,.22)],.06)
    elif k==30:
        for t in np.deg2rad([0,60,120]):
            d=np.array([np.sin(t),np.cos(t)])
            a.line([tuple(-.39*d+[0,.14]),tuple(.39*d+[0,.14])],.055)
            for sign in [-1,1]:
                p=.25*sign*d+np.array([0,.14]);v=np.array([-d[1],d[0]])
                a.line([tuple(p-.11*sign*d+.10*v),tuple(p),tuple(p-.11*sign*d-.10*v)],.045)
    elif k==31:
        # Leaping deer: horizontal body, upright neck, four bent legs, antlers.
        a.ellipse((-.035,.12),.27,.12)
        a.line([(.18,.13),(.25,-.15),(.35,-.20)],.095)
        a.line([(-.23,.13),(-.35,.32),(-.46,.33)],.055)
        a.line([(-.12,.17),(-.20,.37),(-.31,.44)],.05)
        a.line([(.13,.19),(.29,.29),(.23,.44)],.055)
        a.line([(.20,.13),(.39,.18),(.44,.33)],.05)
        a.line([(.28,-.19),(.24,-.35),(.14,-.40)],.035)
        a.line([(.26,-.30),(.34,-.38)],.035)
        a.line([(-.26,.10),(-.37,.005)],.045)
    elif k in (12,13,15):pass
    else:raise ValueError(k)
    image=a.ink if positive else -a.ink
    return image.reshape(N,SS,N,SS).mean((1,3))

# 108 geometry hypotheses: fixed translations, diameters and aspect ratios.
POSES=[(15.5+dx,15.5+dy,d,aspect)
       for d in (19.,23.,27.,31.)
       for aspect in (.84,1.,1.16)
       for dy in (-2.,0.,2.) for dx in (-2.,0.,2.)]
P=len(POSES)
CIRC=[k for k in range(43) if k not in [11,12,13,15,*range(18,32)]]
TRI=[11,*range(18,32)]

def triangle(x,y,inner=False,down=False):
    if down:y=-y
    top=-.76 if inner else -.99
    bottom=.65 if inner else .88
    half=.69 if inner else 1.02
    return (y>=top)&(y<=bottom)&(np.abs(x)<=half*(y-top)/(bottom-top))

def mask_grid(group):
    z=(np.arange(N)+.5)/N*1.8-.9
    x,y=np.meshgrid(z,z)
    return (x*x+y*y<.79**2) if group=='circle' else triangle(x,y,True,down=(group=='yield'))

def interior_projection(group):
    # Analytic projection removes constant and linear illumination gradients.
    mask=mask_grid(group).astype(np.float64).reshape(-1)
    t=(np.arange(N)+.5)/N*1.8-.9
    u,v=np.meshgrid(t,t)
    basis=np.stack([np.ones(N*N),u.ravel(),v.ravel()])
    basis=basis*mask
    # Gram-Schmidt uses only the ideal geometric mask, never photographs.
    ortho=[]
    for row in basis:
        for prev in ortho:row=row-prev*np.dot(row,prev)
        row=row/np.sqrt(np.dot(row,row))
        ortho.append(row)
    Q=np.stack(ortho)
    return np.diag(mask)-Q.T@Q

def resampling_matrix(group):
    mask=mask_grid(group).astype(np.float32).reshape(-1)
    projection=interior_projection(group)
    rows=[]
    for cx,cy,diam,aspect in POSES:
        # Area-integrated bilinear resampling is a fixed, linear operation.
        A=np.zeros((N*N,32*32),np.float32)
        for oy in (.25,.75):
            for ox in (.25,.75):
                ux=(np.arange(N)+ox)/N*1.8-.9
                uy=(np.arange(N)+oy)/N*1.8-.9
                x,y=np.meshgrid(cx+ux*diam*.5*aspect,cy+uy*diam*.5)
                x=x.reshape(-1);y=y.reshape(-1)
                ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
                fx=x-ix;fy=y-iy
                for dx,dy,w in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
                    xx=np.clip(ix+dx,0,31);yy=np.clip(iy+dy,0,31)
                    np.add.at(A,(np.arange(N*N),yy*32+xx),w*.25)
        # Remove only the mean of the analytic interior; never image statistics
        # at build time. Runtime norm provides illumination/contrast invariance.
        A=projection@A
        rows.append(A)
    return np.concatenate(rows,axis=0)

def region(m):
    m=m.astype(np.float64)
    return m/max(m.sum(),1.)

def family_matrix():
    W=np.zeros((7,P,4,32,32),np.float32)
    y,x=np.mgrid[:32,:32]
    for j,(cx,cy,d,aspect) in enumerate(POSES):
        u=(x-cx)/(d*.5*aspect);v=(y-cy)/(d*.5)
        r=np.sqrt(u*u+v*v)
        core=region(r<.72);ring=region((r>.82)&(r<1.04));outer=region((r>1.06)&(r<1.27))
        W[0,j,0]=ring-.40*core-.20*outer
        W[0,j,3]=.08*(core-ring)
        for f,down in [(1,False),(2,True)]:
            inside=triangle(u,v,True,down);whole=triangle(u,v,False,down)
            rim=region(whole&~inside);inter=region(inside)
            outside=region((np.abs(u)<1.15)&(np.abs(v)<1.15)&~whole)
            W[f,j,0]=rim-.35*inter-.18*outside
            W[f,j,3]=.08*(inter-rim)
        W[3,j,1]=region(r<.83)-.20*outer
        W[4,j,2]=region(np.abs(u)+np.abs(v)<.80)-.20*region((np.abs(u)+np.abs(v)>1.15)&(np.abs(u)+np.abs(v)<1.5))
        W[5,j,0]=region(r<.83)-.20*outer
        W[6,j,3]=.50*(core-ring)
    return W.reshape(7*P,-1)

def family_rule(k):
    if k in (6,32,41,42):return [-.30,-.20,-.20,-.50,-.40,-.30,.20]
    if k in (12,):return [-.3,-.3,-.3,-.5,1.60,-.3,0]
    if k==13:return [-.25,-.30,1.45,-.6,-.5,-.25,0]
    if k in (14,17):return [-.30,-.25,-.25,-.55,-.45,1.35,0]
    if k in TRI:return [-.25,1.30,-.30,-.6,-.4,-.25,0]
    if 33<=k<=40:return [-.35,-.35,-.35,1.50,-.40,-.35,0]
    return [1.25,-.25,-.25,-.55,-.45,-.25,0]

def build():
    # Continuous piecewise-linear log(v+.025), identical for the RGB channels.
    knots=np.array([0,.015,.04,.085,.16,.28,.46,.70,1.],np.float64)
    values=np.log(knots+.025)
    slopes=np.diff(values)/np.diff(knots)
    delta=np.r_[slopes[0],np.diff(slopes)]
    W=np.zeros((3*len(delta),3,1,1),np.float32);b=np.zeros(len(W),np.float32)
    for c in range(3):
        for j in range(len(delta)):
            W[c*len(delta)+j,c,0,0]=1;b[c*len(delta)+j]=-knots[j]
    v=linear('log_hinges','image',W,b,True)
    v=layer('log_relu','relu',v)
    W=np.zeros((3,3*len(delta),1,1),np.float32)
    for c in range(3):W[c,c*len(delta):(c+1)*len(delta),0,0]=delta
    logs=linear('log_rgb',v,W,np.full(3,values[0]),True)
    lum=linear('luminance',logs,np.array([.30,.55,.15]).reshape(1,3,1,1),conv=True)
    global_lum=layer('global_luminance_norm','norm',lum)
    # Red, blue and yellow log-chroma, softly clipped to [0,1].
    colour=np.array([[1,-.5,-.5],[-.8,-.2,1],[.5,.5,-1]],np.float32)
    W=np.repeat(colour,2,axis=0).reshape(6,3,1,1)
    b=np.tile([-.10,-1.10],3)
    colour_h=linear('colour_hinges',logs,W,b,True)
    colour_h=layer('colour_relu','relu',colour_h)
    W=np.zeros((4,6,1,1),np.float32)
    for i in range(3):W[i,2*i,0,0]=1;W[i,2*i+1,0,0]=-1
    colour_v=linear('bounded_colours',colour_h,W,conv=True)
    W=np.zeros((4,1,1,1),np.float32);W[3,0,0,0]=1
    grey_v=linear('grey_feature',global_lum,W,conv=True)
    features=layer('colour_and_grey','add',[colour_v,grey_v])
    families=linear('family_geometry',features,family_matrix())
    families=layer('family_grid','reshape',families,shape=[7,P,1])
    outputs=[]
    blank_inputs={}
    # Artwork variations are enumerated, not learned or selected from data.
    variants=[(s,w,t,rot,style,dx,dy)
              for s,w,t,style in [(.78,1.,.082,0),(1.,1.,.086,0),(1.22,1.,.086,0),
                                  (1.,.84,.076,1),(1.,1.16,.100,0)]
              for rot in (-8.,0.,8.)
              for dy in (-.07,0.,.07) for dx in (-.07,0.,.07)]
    V=len(variants)
    for group,classes in [('circle',CIRC),('triangle',TRI)]:
        crop=linear(group+'_resample',global_lum,resampling_matrix(group))
        crop=layer(group+'_patches','reshape',crop,shape=[P,N,N])
        if group=='circle':blank_inputs[15]=crop
        crop=layer(group+'_contrast','norm',crop)
        crop=layer(group+'_strip','reshape',crop,shape=[1,P*N,N])
        mask=mask_grid(group).astype(np.float64)
        projection=interior_projection(group)
        weights=[];bias=[]
        for k in classes:
            for s,w,t,rot,style,dx,dy in variants:
                im=artwork(k,s,w,t,rot,style,dx,dy)
                im=(projection@im.reshape(-1)).reshape(N,N)
                energy=np.sqrt((im*im).sum()*N*N)
                coef=im/max(energy,1e-12)
                weights.append(coef*1.30)
                bias.append({12:.56,13:.56,15:.48}.get(k,0)*1.30)
        glyph=linear(group+'_symbols',crop,np.asarray(weights)[:,None],bias,True,N)
        glyph=layer(group+'_symbol_variants','reshape',glyph,shape=[len(classes),V,P])
        glyph=layer(group+'_best_artwork','maxpool',glyph,size=[V,1],stride=[V,1])
        glyph=layer(group+'_symbol_grid','reshape',glyph,shape=[len(classes),P,1])
        rules=3.0*np.array([family_rule(k) for k in classes],np.float32)
        colour_score=linear(group+'_colour_rules',families,rules[:,:,None,None],conv=True)
        score=layer(group+'_joint_scores','add',[glyph,colour_score])
        score=layer(group+'_best','maxpool',score,size=[P,1],stride=[P,1])
        W=np.zeros((43,len(classes)),np.float32)
        for i,k in enumerate(classes):W[k,i]=1
        outputs.append(linear(group+'_outputs',score,W))
    # A blank sign interior is an absence-of-ink hypothesis, not a constant
    # high correlation. Positive and negative deviations implement mean L1
    # contrast. Yield is sampled with its own downward-pointing interior.
    crop=linear('yield_resample',global_lum,resampling_matrix('yield'))
    blank_inputs[13]=layer('yield_patches','reshape',crop,shape=[P,N,N])
    for k in (13,15):
        tag='blank_'+str(k)
        crop=layer(tag+'_strip','reshape',blank_inputs[k],shape=[1,P*N,N])
        w=np.array([1.,-1.],np.float32).reshape(2,1,1,1)
        dev=linear(tag+'_signed',crop,w,conv=True)
        dev=layer(tag+'_absolute_halves','relu',dev)
        dev=layer(tag+'_mean_deviations','avgpool',dev,size=[N,N],stride=[N,N])
        mask=mask_grid('yield' if k==13 else 'circle')
        coef=-2.0*N*N/mask.sum()
        penalty=linear(tag+'_flatness',dev,np.full((1,2,1,1),coef),[.90],True)
        rule=linear(tag+'_family',families,3.0*np.array(family_rule(k)).reshape(1,7,1,1),conv=True)
        score=layer(tag+'_joint','add',[penalty,rule])
        score=layer(tag+'_best','maxpool',score,size=[P,1],stride=[P,1])
        w=np.zeros((43,1),np.float32);w[k,0]=1
        outputs.append(linear(tag+'_output',score,w))
    # Whole-crop family context prevents a local colour fragment from being
    # interpreted as a complete sign of a conflicting family. Geometry scores
    # are pooled before the hand-written semantic compatibility rules.
    context=layer('whole_sign_family','maxpool',families,size=[P,1],stride=[P,1])
    w=np.zeros((43,7),np.float32)
    for k in range(43):
        if k in (6,32,41,42):w[k]=[-1.2,-.5,-.5,-.7,-.7,-.5,0]
        elif k==12:w[k]=[-.3,-.3,-.3,0,1.,-.3,0]
        elif k==13:w[k]=[-.3,-.7,.8,-.3,-.5,-.3,0]
        elif k in (14,17):w[k]=[-.2,-.5,-.5,-.3,-.4,.8,0]
        elif k in TRI:w[k]=[-.6,.8,-.6,-.3,-.5,-.3,0]
        elif 33<=k<=40:w[k]=[-1.2,-1.2,-.7,.6,-.7,-.8,0]
        else:w[k]=[.8,-.6,-.6,-.3,-.5,-.5,0]
    outputs.append(linear('whole_sign_compatibility',context,w))
    # Priority road is a yellow diamond; its centre need not contain a glyph.
    rule=linear('priority_family',families,3.0*np.array(family_rule(12)).reshape(1,7,1,1),[.60],True)
    rule=layer('priority_best','maxpool',rule,size=[P,1],stride=[P,1])
    w=np.zeros((43,1),np.float32);w[12,0]=1
    outputs.append(linear('priority_output',rule,w))
    layer('scores','add',outputs)

if __name__=='__main__':
    build()
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':LAYERS},indent=2)+'\n')
    np.savez(out/'weights.npz',**WEIGHTS)
    print('Wrote',len(LAYERS),'layers and',sum(w.size for w in WEIGHTS.values()),'float32 weights.')
