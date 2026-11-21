#!/usr/bin/env python3
"""Build a data-free, hand-designed traffic-sign template network.

All pictures used below are analytic drawings made from elementary geometry and a
small, conventional 5x7 bitmap alphabet.  No data files are read.
"""
import json, os, sys
from pathlib import Path
import numpy as np

# Template count.  Geometry, illumination and stroke variations are generated
# from this fixed seed; the random numbers select nuisance parameters only.
T = 180
SEED = 20261119
S = 64                         # 2x rendering canvas

# Conventional 5x7 glyphs, written by hand (not obtained from photographs).
FONT = {
 '0':("01110","10001","10011","10101","11001","10001","01110"),
 '1':("00100","01100","00100","00100","00100","00100","01110"),
 '2':("01110","10001","00001","00010","00100","01000","11111"),
 '3':("11110","00001","00001","01110","00001","00001","11110"),
 '4':("00010","00110","01010","10010","11111","00010","00010"),
 '5':("11111","10000","10000","11110","00001","00001","11110"),
 '6':("01110","10000","10000","11110","10001","10001","01110"),
 '7':("11111","00001","00010","00100","01000","01000","01000"),
 '8':("01110","10001","10001","01110","10001","10001","01110"),
 '9':("01110","10001","10001","01111","00001","00001","01110"),
 'A':("01110","10001","10001","11111","10001","10001","10001"),
 'P':("11110","10001","10001","11110","10000","10000","10000"),
 'S':("01111","10000","10000","01110","00001","00001","11110"),
 'T':("11111","00100","00100","00100","00100","00100","00100"),
 'O':("01110","10001","10001","10001","10001","10001","01110"),
}

RED=np.array((.92,.035,.025)); BLUE=np.array((.025,.18,.78))
WHITE=np.array((.96,.96,.92)); BLACK=np.array((.025,.025,.022))
GRAY=np.array((.53,.54,.52)); YELLOW=np.array((.98,.72,.03))

# Feature convolutions.  These formulas encode luminance, red/blue opponent
# colour and two ordinary Sobel edge directions.
LUM=np.array((.299,.587,.114),np.float64)
FULL_K=np.zeros((8,3,3,3),np.float64)
FULL_K[0,:,1,1]=LUM
FULL_K[1,0,1,1]=1; FULL_K[1,1:,1,1]=-.5
FULL_K[2,2,1,1]=1; FULL_K[2,:2,1,1]=-.5
FULL_K[3,0,1,1]=1; FULL_K[3,1,1,1]=-1
FULL_K[4,0,1,1]=1
FULL_K[5,1,1,1]=1
sobx=np.array(((-1,0,1),(-2,0,2),(-1,0,1)),float)/4
soby=sobx.T
for c in range(3):
 FULL_K[6,c]=LUM[c]*sobx; FULL_K[7,c]=LUM[c]*soby
CROP_K=np.zeros((5,3,13,13),np.float64)
CROP_K[0,:,6,6]=LUM
CROP_K[1,0,6,6]=1; CROP_K[1,1:,6,6]=-.5
CROP_K[2,2,6,6]=1; CROP_K[2,:2,6,6]=-.5
for c in range(3):
 CROP_K[3,c,5:8,5:8]=LUM[c]*sobx
 CROP_K[4,c,5:8,5:8]=LUM[c]*soby

# Pixel-centre coordinates.  Every icon is expressed in a sign-local system.
py,px=np.mgrid[0:S,0:S]


def polygon(x,y,pts):
    """Vectorised even/odd polygon fill."""
    inside=np.zeros(x.shape,bool)
    xj,yj=pts[-1]
    for xi,yi in pts:
        cross=((yi>y)!=(yj>y)) & (x < (xj-xi)*(y-yi)/(yj-yi+1e-30)+xi)
        inside ^= cross
        xj,yj=xi,yi
    return inside


def seg(x,y,x1,y1,x2,y2,w):
    vx=x2-x1; vy=y2-y1
    t=np.clip(((x-x1)*vx+(y-y1)*vy)/(vx*vx+vy*vy+1e-20),0,1)
    return (x-(x1+t*vx))**2+(y-(y1+t*vy))**2 <= w*w


def lines(x,y,pts,w):
    z=np.zeros(x.shape,bool)
    for a,b in zip(pts[:-1],pts[1:]): z |= seg(x,y,*a,*b,w)
    return z


def ellipse(x,y,cx,cy,rx,ry):
    return ((x-cx)/rx)**2+((y-cy)/ry)**2 <= 1


def make_coords(rng):
    # General crop knowledge: road signs occupy most of a roughly square crop.
    rad=rng.uniform(11.7,14.8)*2
    ar=rng.uniform(.91,1.09)
    cx=32+rng.uniform(-1.35,1.35)*2
    cy=32+rng.uniform(-1.25,1.25)*2
    ang=rng.uniform(-.105,.105)
    ca,sa=np.cos(ang),np.sin(ang)
    xx=(px+.5-cx); yy=(py+.5-cy)
    x=(ca*xx+sa*yy)/(rad*ar)
    y=(-sa*xx+ca*yy)/rad
    # A little affine slant models an oblique view without copying a photo.
    x=x-rng.uniform(-.055,.055)*y
    return x,y


def render(k,rng):
    x,y=make_coords(rng)
    im=np.empty((S,S,3),float); im[:]=rng.uniform(.30,.62)
    sym=np.zeros((S,S),bool)
    shape=np.zeros((S,S),bool)
    stroke=rng.uniform(.92,1.10)

    def put(mask,col,detail=False):
        nonlocal sym
        im[mask]=col
        if detail: sym |= mask
    def line(a,b,w=.055,col=BLACK,detail=True): put(seg(x,y,*a,*b,w*stroke),col,detail)
    def path(q,w=.055,col=BLACK,detail=True): put(lines(x,y,q,w*stroke),col,detail)
    def circ(cx,cy,rx,ry=None,col=BLACK,detail=True):
        if ry is None: ry=rx
        put(ellipse(x,y,cx,cy,rx,ry),col,detail)
    def poly(q,col=BLACK,detail=True): put(polygon(x,y,q),col,detail)

    circle_classes=set(range(0,11))|{15,16,17,32,41,42}
    tri_classes={11}|set(range(18,32))
    blue_classes=set(range(33,41))

    if k in circle_classes:
        shape=(x*x+y*y)<=1.05**2
        if k==17:
            put(x*x+y*y<=1,RED)
        elif k in (32,41,42,6):
            put(x*x+y*y<=1,WHITE)
            put((x*x+y*y<=1)&(x*x+y*y>=.94**2),GRAY)
        else:
            put(x*x+y*y<=1,RED)
            put(x*x+y*y<=rng.uniform(.72,.78)**2,WHITE)
    elif k in tri_classes:
        outer=[(0,-1.03),(-.94,.73),(.94,.73)]
        inner=[(0,-.75),(-.66,.57),(.66,.57)]
        shape=polygon(x,y,[(-1.03,.82),(0,-1.12),(1.03,.82)])
        put(polygon(x,y,outer),RED)
        put(polygon(x,y,inner),WHITE)
    elif k==12:
        outer=[(0,-1),(-.92,0),(0,1),(.92,0)]
        mid=[(0,-.82),(-.75,0),(0,.82),(.75,0)]
        inn=[(0,-.68),(-.62,0),(0,.68),(.62,0)]
        shape=polygon(x,y,outer); put(shape,WHITE); put(polygon(x,y,mid),BLACK); put(polygon(x,y,inn),YELLOW)
    elif k==13:
        outer=[(-.94,-.72),(.94,-.72),(0,1.02)]
        inner=[(-.66,-.54),(.66,-.54),(0,.72)]
        shape=polygon(x,y,[(-1.03,-.82),(1.03,-.82),(0,1.12)])
        put(polygon(x,y,outer),RED); put(polygon(x,y,inner),WHITE)
    elif k==14:
        q=[]
        for j in range(8):
            a=np.pi/8+j*np.pi/4; q.append((np.sin(a),-np.cos(a)))
        shape=polygon(x,y,q); put(shape,RED)
    elif k in blue_classes:
        shape=x*x+y*y<=1.05**2; put(x*x+y*y<=1,BLUE)

    # Bitmap text.  The alphabet is deliberately simple and canonical.
    def text(s,cx0=0,cy0=0,h=.73,total=None,col=BLACK):
        nonlocal sym
        n=len(s); total=(1.12 if n==2 else 1.29 if n==3 else .78) if total is None else total
        gap=total*.06; gw=(total-gap*(n-1))/n; gh=h/7
        left=cx0-total/2
        m=np.zeros((S,S),bool)
        for q,ch in enumerate(s):
            rows=FONT[ch]
            cw=gw/5
            for iy,row in enumerate(rows):
                for ix,v in enumerate(row):
                    if v=='1':
                        xa=left+q*(gw+gap)+ix*cw; xb=xa+cw*.91
                        ya=cy0-h/2+iy*gh; yb=ya+gh*.91
                        m |= (x>=xa)&(x<=xb)&(y>=ya)&(y<=yb)
        put(m,col,True)

    speeds={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}
    if k in speeds: text(speeds[k],h=rng.uniform(.68,.78))
    elif k==6:
        text('80',h=.70,col=GRAY)
        for d in np.linspace(-.65,.65,5): line((d-.42,.70),(d+.42,-.70),.035,BLACK)
    elif k in (9,10,16,41,42):
        def car(cx0,col=BLACK,truck=False):
            if truck:
                poly([(cx0-.38,.05),(cx0+.08,.05),(cx0+.08,.35),(cx0-.38,.35)],col)
                poly([(cx0+.08,.13),(cx0+.28,.13),(cx0+.38,.35),(cx0+.08,.35)],col)
                circ(cx0-.24,.39,.075,col=BLACK); circ(cx0+.23,.39,.075,col=BLACK)
            else:
                poly([(cx0-.25,.02),(cx0+.25,.02),(cx0+.34,.38),(cx0-.34,.38)],col)
                poly([(cx0-.16,-.18),(cx0+.14,-.18),(cx0+.25,.02),(cx0-.25,.02)],col)
                circ(cx0-.22,.40,.06,col=BLACK); circ(cx0+.22,.40,.06,col=BLACK)
        if k==9: car(-.28,RED); car(.30,BLACK)
        elif k==10: car(-.23,RED,True); car(.36,BLACK)
        elif k==16: car(0,BLACK,True)
        elif k==41: car(0,GRAY)
        else: car(0,GRAY,True)
        if k in (41,42):
            for d in np.linspace(-.65,.65,5): line((d-.42,.70),(d+.42,-.70),.035,BLACK)
    elif k==11:
        line((0,.56),(0,-.57),.105)
        line((-.45,-.16),(.45,-.16),.052)
    elif k==14:
        text('STOP',h=.40,total=1.25,col=WHITE)
    elif k==17:
        put((np.abs(y)<.17)&(np.abs(x)<.72),WHITE,True)
    elif k==18:
        line((0,-.43),(0,.20),.085); circ(0,.43,.095)
    elif k in (19,20):
        sg=-1 if k==19 else 1
        path([(sg*.20,.55),(sg*.20,.18),(sg*.08,-.02),(sg*-.27,-.34),(sg*-.30,-.54)],.085)
    elif k==21:
        path([(0,.58),(-.24,.25),(.24,-.08),(-.20,-.50)],.085)
    elif k==22:
        # Two rounded-looking humps, represented by short polylines.
        path([(-.55,.20),(-.40,-.12),(-.20,-.27),(0,-.05),(.20,-.27),(.40,-.12),(.55,.20)],.075)
    elif k==23:
        # Car and two sinuous skid traces.
        poly([(-.36,-.30),(.18,-.30),(.32,-.05),(-.42,-.05)],BLACK)
        circ(-.27,-.01,.065); circ(.20,-.01,.065)
        path([(-.20,.15),(-.38,.27),(-.18,.38),(-.40,.52)],.035)
        path([(.24,.12),(.06,.25),(.27,.36),(.08,.52)],.035)
    elif k==24:
        path([(-.50,-.52),(-.50,.53)],.075)
        path([(.52,-.52),(.23,.53)],.075)
    elif k==25:
        circ(-.06,-.40,.09)
        line((-.05,-.30),(-.13,.05),.065)
        line((-.10,-.08),(-.38,.10),.05); line((-.08,-.07),(.19,.10),.05)
        line((-.13,.04),(-.34,.40),.055); line((-.12,.04),(.08,.42),.055)
        line((.20,-.10),(.38,.46),.035); poly([(.25,.45),(.50,.45),(.40,.28)],BLACK)
        line((-.55,.51),(.55,.51),.035)
    elif k==26:
        put((np.abs(x)<.20)&(y>-.55)&(y<.55),BLACK,True)
        circ(0,-.34,.105, col=RED); circ(0,0,.105,col=YELLOW); circ(0,.34,.105,col=np.array((.04,.55,.08)))
    elif k in (27,28):
        def person(cx0,cy0,sc=.75):
            circ(cx0,cy0-.38*sc,.09*sc)
            line((cx0,cy0-.28*sc),(cx0,cy0+.08*sc),.055*sc)
            line((cx0,cy0-.16*sc),(cx0-.22*sc,cy0+.02*sc),.04*sc)
            line((cx0,cy0-.12*sc),(cx0+.23*sc,cy0+.03*sc),.04*sc)
            line((cx0,cy0+.06*sc),(cx0-.19*sc,cy0+.39*sc),.05*sc)
            line((cx0,cy0+.06*sc),(cx0+.22*sc,cy0+.35*sc),.05*sc)
        if k==27: person(0,.05,1)
        else: person(-.18,.08,.82); person(.23,-.03,.62)
    elif k==29:
        # Bicycle: two wheels, frame, rider-less conventional bicycle icon.
        ring1=ellipse(x,y,-.34,.25,.22,.22)&~ellipse(x,y,-.34,.25,.17,.17)
        ring2=ellipse(x,y,.36,.25,.22,.22)&~ellipse(x,y,.36,.25,.17,.17)
        put(ring1|ring2,BLACK,True)
        path([(-.34,.25),(-.10,-.08),(.12,.25),(-.34,.25)],.035)
        line((-.10,-.08),(.26,-.08),.035); line((.12,.25),(.28,-.20),.035); line((.20,-.20),(.38,-.20),.035)
    elif k==30:
        for a in (0,np.pi/3,-np.pi/3):
            dx=.55*np.cos(a); dy=.55*np.sin(a); line((-dx,-dy),(dx,dy),.045)
            for sg in (-1,1):
                ex=sg*dx; ey=sg*dy
                line((ex,ey),(ex-.16*np.cos(a+.55)*sg,ey-.16*np.sin(a+.55)*sg),.035)
                line((ex,ey),(ex-.16*np.cos(a-.55)*sg,ey-.16*np.sin(a-.55)*sg),.035)
    elif k==31:
        put(ellipse(x,y,-.05,.03,.42,.20),BLACK,True)
        line((.22,-.05),(.40,-.40),.09); circ(.45,-.46,.12,.085)
        line((-.28,.13),(-.40,.52),.045); line((-.05,.16),(-.10,.54),.045); line((.16,.12),(.28,.51),.045)
        line((.46,-.52),(.37,-.69),.028); line((.46,-.52),(.55,-.70),.028)
    elif k==32:
        for d in np.linspace(-.72,.72,6): line((d-.43,.72),(d+.43,-.72),.040,BLACK)
    elif k in blue_classes:
        def arrow(a,b,w=.105):
            line(a,b,w,WHITE)
            vx=b[0]-a[0]; vy=b[1]-a[1]; n=(vx*vx+vy*vy)**.5; ux=vx/n; uy=vy/n
            px0=-uy; py0=ux
            poly([b,(b[0]-.30*ux+.20*px0,b[1]-.30*uy+.20*py0),(b[0]-.30*ux-.20*px0,b[1]-.30*uy-.20*py0)],WHITE)
        if k==33:
            path([(0,.55),(0,-.05),(.35,-.05)],.105,WHITE); poly([(.58,-.05),(.28,-.27),(.28,.17)],WHITE)
        elif k==34:
            path([(0,.55),(0,-.05),(-.35,-.05)],.105,WHITE); poly([(-.58,-.05),(-.28,-.27),(-.28,.17)],WHITE)
        elif k==35: arrow((0,.55),(0,-.57))
        elif k in (36,37):
            sg=1 if k==36 else -1
            arrow((0,.58),(0,-.58)); arrow((0,.10),(sg*.55,.10),.09)
        elif k in (38,39):
            sg=1 if k==38 else -1; arrow((sg*-.42,-.43),(sg*.43,.43),.12)
        else:
            # Three tangential arrows around a circle.
            for a in (-np.pi/2,np.pi/6,5*np.pi/6):
                a1=a-.55; a2=a+.42
                arrow((.44*np.cos(a1),.44*np.sin(a1)),(.44*np.cos(a2),.44*np.sin(a2)),.07)

    # Broad masks used only to say which analytic pixels should participate in
    # correlation.  They are based on sign geometry, not on any photograph.
    detail=sym.copy()
    if not detail.any(): detail = shape & (x*x+y*y<.60)
    # Include a margin around symbols so that their white/blue surround matters.
    for _ in range(3):
        p=np.pad(detail,1)
        detail=(p[1:-1,1:-1]|p[:-2,1:-1]|p[2:,1:-1]|p[1:-1,:-2]|p[1:-1,2:])
    return im,shape,detail


def conv2(a,k,pad):
    # a is HWC, k is OIkk, and this is only used on synthetic drawings.
    h,w,c=a.shape; kk=k.shape[2]
    q=np.pad(a,((pad,pad),(pad,pad),(0,0)))
    out=np.zeros((k.shape[0],h+2*pad-kk+1,w+2*pad-kk+1),float)
    for dy in range(kk):
      for dx in range(kk):
       out += np.einsum('oi,hwi->ohw',k[:,:,dy,dx],q[dy:dy+out.shape[1],dx:dx+out.shape[2]])
    return out


def norm_channels(a):
    return (a-a.mean((1,2),keepdims=True))/np.sqrt(a.var((1,2),keepdims=True)+1e-5)


def weight_features(f,mask,channel_weights):
    """Zero-DC, unit-energy correlation filters within an analytic mask."""
    m=mask.astype(float)
    ans=np.zeros_like(f)
    for c in range(len(f)):
        # Keep the template's expected signed structure, including filled
        # symbols; put zero-DC compensation in the analytic surround.
        z=m*f[c]
        z -= z.sum()*(1-m)/((1-m).sum()+1e-12)
        n=np.sqrt((z*z).sum())
        if n>1e-8: ans[c]=z*(channel_weights[c]/n)
    return ans.astype(np.float32)


def template_weights(k,rng):
    im,shape,detail=render(k,rng)
    # Area averaging mimics a robust low-resolution view of the full sign.
    lo=im.reshape(16,4,16,4,3).mean((1,3))
    fs=norm_channels(conv2(lo,FULL_K,1))
    ms=shape.reshape(16,4,16,4).mean((1,3))>.18
    # Correlation includes both silhouette and interior, with edge channels.
    wf=weight_features(fs,ms,[.31,.27,.27,.18,.16,.16,.25,.25])

    # A high-resolution central view emphasizes pictograms, arrows and digits.
    fc=norm_channels(conv2(im,CROP_K,0))       # 5 x 52 x 52
    # Crop feature corresponds to source centres 6..57 at 2x resolution, then
    # area average to the network's 20x20 crop convention via a 32x render.
    # First reduce the rendered icon to 32, then use exactly the scorer kernels.
    hi=im.reshape(32,2,32,2,3).mean((1,3))
    fc=norm_channels(conv2(hi,CROP_K,0))       # 5 x 20 x 20
    md=detail.reshape(32,2,32,2).mean((1,3))[6:26,6:26]>.10
    wc=weight_features(fc,md,[.55,.28,.28,.44,.44])
    return wf,wc


def build(dst):
    dst=Path(dst); dst.mkdir(parents=True,exist_ok=True)
    rows=43*T
    WF=np.empty((rows,8*16*16),np.float32)
    WC=np.empty((rows,5*20*20),np.float32)
    # One nuisance sequence is shared by all classes, preventing accidental
    # class-specific priors while covering scale, pose, and stroke width.
    # Hand calibration compensates for predictable template complexity. Filled
    # or nearly blank icons otherwise have more correlation energy than fine
    # line drawings. These are semantic design constants, not image statistics.
    calibration=np.array([
      1.04,1.04,1.06,1.07,1.06,1.06,.80,1.06,1.06,1.03,1.03,1.05,
      1.08,1.00,1.04,.78,1.04,1.03,1.08,1.13,1.13,1.12,1.09,1.12,
      1.10,1.08,1.08,1.04,1.08,.92,.94,1.06,.78,1.06,1.06,1.00,
      1.02,1.05,1.03,1.03,.94,.82,.84],np.float32)
    bias=np.array([
      0,0,0,0,0,0,-1,0,0,0,0,0,1,0,0,-2,0,0,0,0,0,0,0,0,
      0,0,0,0,0,-1,-1,0,-1,1,1,0,0,0,1,1,0,-1,-1],np.float32)
    for k in range(43):
        for t in range(T):
            rng=np.random.default_rng(SEED+1009*t)
            wf,wc=template_weights(k,rng)
            j=k*T+t
            WF[j]=calibration[k]*wf.reshape(-1)
            WC[j]=calibration[k]*wc.reshape(-1)

    layers=[
      {"name":"small","op":"avgpool","inputs":["image"],"size":[2,2],"stride":[2,2]},
      {"name":"fullfeat","op":"conv","inputs":["small"],"out":8,"kernel":[3,3],"stride":1,"padding":1},
      {"name":"fullnorm","op":"norm","inputs":["fullfeat"]},
      {"name":"cropfeat","op":"conv","inputs":["image"],"out":5,"kernel":[13,13],"stride":1,"padding":0},
      {"name":"cropnorm","op":"norm","inputs":["cropfeat"]},
      {"name":"fullscore","op":"dense","inputs":["fullnorm"],"out":rows},
      {"name":"cropscore","op":"dense","inputs":["cropnorm"],"out":rows},
      {"name":"combined","op":"add","inputs":["fullscore","cropscore"]},
      {"name":"bank","op":"reshape","inputs":["combined"],"shape":[43,T,1]},
      {"name":"best","op":"maxpool","inputs":["bank"],"size":[T,1],"stride":[T,1]},
      {"name":"scores","op":"reshape","inputs":["best"],"shape":[43]},
    ]
    (dst/'network.json').write_text(json.dumps({"layers":layers},separators=(',',':')))
    weights={
      'fullfeat.W':FULL_K.astype(np.float32),'fullfeat.b':np.zeros(8,np.float32),
      'cropfeat.W':CROP_K.astype(np.float32),'cropfeat.b':np.zeros(5,np.float32),
      'fullscore.W':WF,'fullscore.b':np.repeat(bias,T)/2,
      'cropscore.W':WC,'cropscore.b':np.repeat(bias,T)/2,
    }
    np.savez(dst/'weights.npz',**weights)

if __name__=='__main__':
    build(sys.argv[1] if len(sys.argv)>1 else '.')
