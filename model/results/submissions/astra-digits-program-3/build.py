"""Generate digit hypotheses from a grammar of hand-drawn pen strokes. No input files."""
from pathlib import Path
import sys
import numpy as np

out = Path(sys.argv[1]); out.mkdir(exist_ok=True, parents=True)
rng = np.random.default_rng(35771)
U = rng.uniform

def normal(s=1):
    return np.clip(rng.normal(), -2.35, 2.35)*s

# All pen coordinates are in units of glyph height, with y increasing downwards.
# Segment controls are relative to the same coordinate system as the endpoints.
def start(x,y):
    return [np.array([[x,y]])]

def cubic(p,a,b,end):
    t = np.linspace(0, 1, 25)[1:,None]; u = 1-t
    p.append(u**3*p[-1][-1] + 3*t*u*u*a + 3*u*t*t*b + t**3*end)

def line(p,end):
    t=np.linspace(0,1,18)[1:,None]; p.append((1-t)*p[-1][-1]+t*end)

def oval(cx, cy, rx, ry, a, b, flat=1):
    t=np.linspace(a,b,100)*np.pi
    c=np.cos(t); s=np.sin(t)
    return np.stack((cx+rx*np.sign(c)*abs(c)**flat,cy+ry*np.sign(s)*abs(s)**flat),axis=1)

def digit(d):
    paths=[]; st=U()
    if d==0:
        ang=U(-.95,-.3); span=U(1.90,2.28) if st>.18 else U(1.67,1.96)
        p=oval(.5,.5,.50,.50,ang,ang-span,U(.76,1.25))
        # A small inward or outward turn where the oval closes.
        w=np.linspace(0,1,len(p))[:,None]**8
        p+=w*[normal(.12),normal(.07)]
        paths.append(p)
    elif d==1:
        bx=.55+normal(.06); tx=.48+normal(.045)
        if st<.73:
            p=start(tx+normal(.04),U(.0,.12)); cubic(p,(tx+normal(.045),.22),(bx+normal(.04),.78),(bx,1))
        else:
            ax=tx-U(.08,.44); ay=U(.16,.47)
            p=start(ax,ay); cubic(p,(ax+.05,ay+.01),(tx-.06,.08),(tx,0)); cubic(p,(tx+normal(.03),.3),(bx,.8),(bx,1))
        if st>.95 or st<.012:
            paths.append(np.concatenate(p)); l=bx-U(.19,.50); p=start(l,1+normal(.018)); cubic(p,(bx-.03,1),(bx+.10,1),(bx+U(.14,.55),1+normal(.05)))
        paths.append(np.concatenate(p))
    elif d==2:
        # Arching entry, oblique descent, and a baseline with or without a loop.
        ax=U(-.02,.22); ay=U(.13,.46); h=U(.08,.29); rx=U(.64,1.04)
        if st<.10:
            p=start(ax+U(.06,.28),ay+U(.01,.14)); cubic(p,(ax-.13,ay+.14),(ax-.10,ay),(ax,ay-.10)); ay-=.1
        else: p=start(ax,ay)
        cubic(p,(ax+U(.0,.22),-.05),(rx+U(-.23,.09),-.14),(rx,h))
        lx=U(-.12,.13); ly=U(.85,1.03)
        if st<.42:
            cubic(p,(rx+.03,h+.16),(lx+.25,ly-.18),(lx,ly))
            ex=U(.76,1.25); ey=ly+U(-.13,.12); cubic(p,(lx+.18,ly+U(-.05,.08)),(ex-.18,ey+normal(.04)),(ex,ey))
        elif st<.93:
            # Lower-left cursive loop. The descent can go almost straight down
            # from the cap before curving back to the left, crossing the exit.
            ex=rx+U(-.14,.52); ey=ly+U(-.20,.14); q=U(.19,.52)
            v=U(.00,.62); top=ly-q
            cubic(p,(rx+v*.6,h+.21),(rx+v,ly-.09),(lx+U(.08,.28),ly))
            cubic(p,(lx-U(.04,.29),ly+.04),(lx-U(.04,.28),top+.035),(lx+.10,top))
            cubic(p,(lx+.38,top-.095),(ex-U(.12,.48),ey+U(-.07,.15)),(ex,ey))
        else:
            cubic(p,(rx,h+.28),(lx+.32,.74),(lx,ly)); line(p,(U(.37,.62),ly-.08)); line(p,(U(.78,1.13),ly+normal(.08)))
        paths.append(np.concatenate(p))
    elif d==3:
        m=U(.37,.59); wx=U(.16,.51); a=U(-.08,.19); b=U(.77,1.1); c=U(.72,1.1)
        ay=U(.04,.31) if st>.22 else U(.0,.13)
        p=start(a,ay)
        if st<.23:
            cubic(p,(a+.2,ay-.07),(b-.15,-.02),(b,.03)); line(p,(wx,m))
        else:
            cubic(p,(a+.17,ay-.25),(b+.02,-.12),(b,U(.12,.24))); cubic(p,(b+.04,m-.13),(wx+.17,m-.03),(wx,m))
        if .48<st<.78:
            v=U(.09,.27); cubic(p,(wx-v,m+.12),(wx-v*1.3,m-.07),(wx+.10,m-.014))
        cubic(p,(wx+.22,m-.06),(c+.10,m+.03),(c,m+(.98-m)*.50))
        endy=U(.70,.98); cubic(p,(c-.04,1.05),(a+.16,1.10),(a+normal(.11),endy)); paths.append(np.concatenate(p))
    elif d==4:
        # Left downstroke and cross stroke, independently placed upright.
        jy=U(.49,.69); jx=U(-.09,.13); ax=U(.05,.62); ay=U(-.03,.20); ex=U(.69,1.17); ey=jy+U(-.13,.095)
        tx=U(.65,.93); bx=tx+normal(.095)
        if st<.24:
            ax=tx+normal(.036); ay=U(-.045,.08)
        p=start(ax,ay); cubic(p,(ax+(jx-ax)*.28,ay+(jy-ay)*.35),(jx+normal(.04),jy-.07),(jx,jy)); cubic(p,(jx+.1,jy+U(-.01,.08)),(ex-.14,ey),(ex,ey)); paths.append(np.concatenate(p))
        p=start(tx,ay if st<.24 else U(-.03,.27)); cubic(p,(tx+normal(.065),.32),(bx+normal(.04),.77),(bx,1)); paths.append(np.concatenate(p))
    elif d==5:
        ly=U(.34,.55); lx=U(.0,.23); tx=lx+U(-.03,.30); ty=U(.0,.14); ex=U(.71,1.14); ey=ty+U(-.14,.08)
        rx=U(.79,1.05); endx=U(-.12,.22); endy=U(.72,1.00)
        p=start(ex,ey); cubic(p,(ex-.21,ey+normal(.06)),(tx+.17,ty+normal(.045)),(tx,ty))
        if st<.36:
            paths.append(np.concatenate(p)); p=start(tx+U(-.15,.15),ty+U(-.09,.09))
        if .30<st<.68:
            # Quick continuous form: the stem rounds into a diagonal back,
            # with no sharp corner above the bowl, much like a capital S.
            h=U(.18,.38); cubic(p,(tx-.13,ty+.03),(lx-.12,h-.05),(lx,h))
            cubic(p,(lx+.06,h+.15),(rx+.07,.58),(rx,.75))
        else:
            cubic(p,(tx+normal(.03),ty+.16),(lx-.014,ly-.11),(lx,ly))
            if st>.82:
                cubic(p,(lx-.01,ly+.14),(lx+.19,ly+.03),(lx+.31,ly)); lx+=.31
            cubic(p,(lx+.37,ly-U(.05,.22)),(rx+.05,ly-.025),(rx,ly+U(.14,.30)))
        cubic(p,(rx+.01,1.00),(endx+U(.15,.35),1.14),(endx,endy)); paths.append(np.concatenate(p))
    elif d==6:
        # Descending arc sweeps under the counter. The oval can close, gap or overshoot.
        ax=U(.43,.99); ay=U(-.04,.085); j=U(.47,.66); lx=U(-.06,.12); rx=U(.77,1.05); cx=U(.37,.58)
        p=start(ax,ay); cubic(p,(ax-U(.18,.50),ay+U(.015,.20)),(lx-.03,j-.16),(lx,j+U(.02,.15))); cubic(p,(lx-.02,.93),(cx-.20,1.07),(cx,1)); cubic(p,(cx+.21,1.01),(rx, .91),(rx,j+.17))
        if st<.20:
            cubic(p,(rx-.02,j-.02),(cx+.06,j-U(.03,.16)),(cx-U(.03,.20),j+.02))
        else:
            qx=lx+U(-.04,.16); qy=j+U(.01,.20); cubic(p,(rx+.03,j-.08),(cx-.14,j-.1),(qx,qy))
            if st>.74: cubic(p,(qx-.08,qy+.08),(qx+.02,qy+.16),(qx+.12,qy+.21))
        if .14<st<.21:
            s=start(ax+.1,ay+.13); cubic(s,(ax+.12,ay+.01),(ax+.07,ay-.04),(ax,ay)); paths.append(np.concatenate(s))
        paths.append(np.concatenate(p))
    elif d==7:
        ax=U(-.05,.16); ay=U(-.01,.15); tx=U(.82,1.08); ty=U(-.04,.15); bx=U(.28,.62)
        if st>.66:
            # Short cap swept upwards into an abrupt change of direction.
            ax=tx-U(.33,.83); ay=ty+U(.08,.24)
        if st<.24: p=start(ax-.04,ay+U(.11,.29)); line(p,(ax,ay))
        else: p=start(ax,ay)
        cubic(p,(ax+.24,ay+normal(.06)),(tx-.20,ty+normal(.065)),(tx,ty)); cubic(p,(tx+U(-.14,.0),ty+.20),(bx+U(.01,.20),.74),(bx,1)); paths.append(np.concatenate(p))
        if .18<st<.44:
            yy=U(.40,.61); xx=tx+(bx-tx)*yy; p=start(xx-U(.20,.47),yy+normal(.03)); line(p,(xx+U(.09,.33),yy+normal(.04))); paths.append(np.concatenate(p))
    elif d==8:
        # A continuous figure eight, with independent sizes and curvatures of its lobes.
        zy=U(.37,.57); zx=U(.35,.64); lt=U(-.01,.26); rt=U(.74,1.06); lb=U(-.11,.18); rb=U(.76,1.08)
        if st<.18:
            for (lo,hi,top,bottom) in [(lt,rt,.0,zy+.04),(lb,rb,zy-.05,1.)]:
                p=oval((lo+hi)/2,(top+bottom)/2,(hi-lo)/2,(bottom-top)/2,-.2,1.8+normal(.06),U(.86,1.1)); paths.append(p)
        elif st>.68:
            # Ribbon form: one side of the crossing is a long, nearly
            # straight pen movement, rather than an arc for each quadrant.
            zy+=U(-.10,.12); lb+=U(-.18,.12); rt+=U(-.08,.21)
            p=start(rt,.05); cubic(p,(rt-.45,zy*.73),(lb+.18,.72),(lb,.83)); cubic(p,(lb-.14,1.05),(rb-.07,1.07),(rb,.83))
            cubic(p,(rb+.08,.64),(lt-.03,zy-.05),(lt,.19)); cubic(p,(lt-.08,-.05),(rt-.22,-.04),(rt+U(-.20,.17),U(-.02,.24))); paths.append(np.concatenate(p))
        else:
            p=start(zx,zy); cubic(p,(zx-.22,zy-.09),(lt,zy-.12),(lt,zy*.44)); cubic(p,(lt,-.04),(rt-.05,-.08),(rt,zy*.38)); cubic(p,(rt+.05,zy*.68),(zx+.26,zy-.15),(zx,zy)); cubic(p,(zx-.18,zy+.13),(lb,zy+.18),(lb,.83)); cubic(p,(lb+.015,1.08),(rb-.03,1.07),(rb,.79)); cubic(p,(rb,.66),(zx+.24,zy+.10),(zx,zy)); a=np.concatenate(p)
            # Start and finish away from the crossing, not always precisely joined.
            a=np.roll(a[:-1],rng.integers(0,len(a)-1),axis=0); a=a[:len(a)-rng.integers(0,15)]; a+=np.linspace(0,1,len(a))[:,None]**7*[normal(.08),normal(.035)]; paths.append(a)
            if st<.46: paths[-1][:,0]=1-paths[-1][:,0]
    else:
        # Counter above a tall downstroke, lifted pen or joined at the upper right.
        h=U(.38,.60); tx=U(.69,.98); lx=U(-.10,.14); mx=(lx+tx)*.5; jy=U(.10,.28); bx=tx+U(-.37,.13)
        p=start(tx+normal(.05),jy+normal(.05)); cubic(p,(tx-.005,jy-.20),(mx+.05,-.10),(lx+.11,.08)); cubic(p,(lx-.12,.18),(lx-.07,h+.04),(mx-.08,h));
        if .18<st<.32:
            # A counter that curls toward a separate stem but isn't fully shut.
            t=oval(mx,h*.48,(tx-lx)*.51,h*.52,U(-.57,-.36),-1.50); p=[t]
        if st<.18:
            cubic(p,(mx+.11,h+.02),(tx-.10,h-.08),(tx-.07,jy+.10)); paths.append(np.concatenate(p)); p=start(tx+U(-.04,.08),U(.0,.14))
        else:
            cubic(p,(mx+.12,h+.025),(tx-.03,jy+.14),(tx,U(-.035,.19)))
        if st>.79:
            cubic(p,(tx+normal(.05),.55),(bx+.26,1.04),(bx-.14,1)); cubic(p,(bx-.28,.99),(bx-.34,.88),(bx-.37,U(.73,.91)))
        else:
            cubic(p,(tx+normal(.07),.42),(bx+normal(.08),.79),(bx,1));
        paths.append(np.concatenate(p))
    return paths

# Batched methods here are shared *verbatim* by the builder and predictor. They
# operate independently on each glyph. No parameters are estimated across glyphs.
features = r'''import numpy as np

def sample(im,x,y):
    n,h,w=im.shape
    ii=np.arange(n)[:,None,None]
    x=np.clip(x,0,w-1.001); y=np.clip(y,0,h-1.001)
    ix=x.astype(np.int32); iy=y.astype(np.int32)
    dx=(x-ix).astype(np.float32); dy=(y-iy).astype(np.float32)
    return (1-dy)*((1-dx)*im[ii,iy,ix]+dx*im[ii,iy,ix+1])+dy*((1-dx)*im[ii,iy+1,ix]+dx*im[ii,iy+1,ix+1])

def normalize(im):
    g=np.arange(28,dtype=np.float32); xx=g[None,None,:]; yy=g[None,:,None]
    mass=np.maximum(im.sum(axis=(1,2),keepdims=True),.0001)
    cx=(im*xx).sum(axis=(1,2),keepdims=True)/mass; cy=(im*yy).sum(axis=(1,2),keepdims=True)/mass
    tilt=(im*(xx-cx)*(yy-cy)).sum(axis=(1,2),keepdims=True)/np.maximum((im*(yy-cy)**2).sum(axis=(1,2),keepdims=True),.001)
    tilt=np.clip(tilt,-1.3,1.3)
    # Deskew and bound the glyph by the high-confidence part of its ink.
    wide=np.arange(44,dtype=np.float32)[None,None,:]-8
    part=sample(im,wide+.88*tilt*(yy-cy),yy+np.zeros_like(wide))
    binary=part>np.maximum(.075,part.max(axis=(1,2),keepdims=True)*.20)
    xs=binary.any(axis=1); ys=binary.any(axis=2)
    x0=xs.argmax(axis=1); x1=43-xs[:,::-1].argmax(axis=1)
    y0=ys.argmax(axis=1); y1=27-ys[:,::-1].argmax(axis=1)
    w=x1-x0+1; h=y1-y0+1
    # Partial rather than complete aspect normalization; don't inflate a straight 1.
    sh=h/23.; sw=sh*np.clip((w/np.maximum(.66*h,1))**.66,.65,1.62)
    sh=sh.astype(np.float32)[:,None,None]; sw=sw.astype(np.float32)[:,None,None]
    oy=(y0+y1).astype(np.float32)[:,None,None]*.5; ox=(x0+x1-16).astype(np.float32)[:,None,None]*.5
    y=oy+(yy-13.5)*sh
    x=ox+(xx-13.5)*sw+.88*tilt*(y-cy)
    z=sample(im,x,y)
    aspect=np.log(np.maximum(w,.5)/np.maximum(h,.5)).astype(np.float32)
    return z,aspect

def pool(a,weights):
    return (weights @ a @ weights.T)

def gaussianbins(n,scale):
    axis=np.arange(28,dtype=np.float32)
    c=(np.arange(n,dtype=np.float32)+.5)*28/n-.5
    w=np.exp(-.5*((axis[None,:]-c[:,None])/scale)**2)
    return w/w.sum(axis=1,keepdims=True)

coarse=gaussianbins(6,2.25); fine=gaussianbins(9,1.35)

def unit(a):
    return a/np.sqrt(np.maximum((a*a).sum(axis=tuple(range(1,a.ndim)),keepdims=True),.000001))

def holes(z):
    # Ink counters at several brightness levels. Flood through the background
    # from all four borders, then locate any unreachable background by height.
    n=len(z); q=z.max(axis=(1,2),keepdims=True)
    levels=[]
    for v in (.26,.50,.73):
        free=z<=v*q
        a=np.ones_like(free); a[:,1:-1,1:-1]=False; a &= free
        for it in range(80):
            b=a.copy(); b[:,1:]|=a[:,:-1]; b[:,:-1]|=a[:,1:]; b[:,:,1:]|=a[:,:,:-1]; b[:,:,:-1]|=a[:,:,1:]; b &= free
            if np.array_equal(a,b): break
            a=b
        trapped=(free & ~a).sum(axis=2).astype(np.float32)
        h=trapped@coarse.T
        # Saturate once a counter is unmistakable; the metric isn't meant
        # to enforce equal size holes in thick and thin handwriting.
        levels.append(np.minimum(np.sqrt(h*.50),1.0))
    return sum(levels)/3

def descriptors(z):
    # Smooth centered differences, then six soft directed orientation channels.
    zp=np.pad(z,((0,0),(1,1),(1,1)),mode='constant')
    dx=.25*((zp[:,:-2,2:]-zp[:,:-2,:-2])+2*(zp[:,1:-1,2:]-zp[:,1:-1,:-2])+(zp[:,2:,2:]-zp[:,2:,:-2]))
    dy=.25*((zp[:,2:,:-2]-zp[:,:-2,:-2])+2*(zp[:,2:,1:-1]-zp[:,:-2,1:-1])+(zp[:,2:,2:]-zp[:,:-2,2:]))
    den=1/(.035+np.sqrt(dx*dx+dy*dy))
    small=[]; large=[]
    for k in range(6):
        angle=k*np.pi/3
        p=np.maximum(np.float32(np.cos(angle))*dx+np.float32(np.sin(angle))*dy,0)
        p=p*p*den
        small.append(np.sqrt(pool(p,coarse)+.00001)); large.append(np.sqrt(pool(p,fine)+.00001))
    s=unit(np.stack(small,axis=1)); l=unit(np.stack(large,axis=1))
    ink=unit(np.sqrt(np.maximum(pool(z,fine),0)))
    return s.reshape(len(z),-1),l,ink
'''
exec(features)

def render(d):
    strokes=digit(d)
    # Smooth low-order coordinate fields model bends, changed waist heights and writer slant.
    width=.72*np.exp(normal(.25)); lean=.17+normal(.29); yshear=normal(.115)
    thickness=U(.065,.25); tip=U(-.18,.39); brush=normal(.15)
    bend=normal(.086); taper=normal(.19); waist=normal(.115); wave=normal(.042)
    left=normal(.048); right=normal(.048); squash=normal(.10)
    ripple=normal(.032); phase=U(-np.pi,np.pi); across=normal(.034); tilted=normal(.05)
    more=[]; radius=[]
    for stroke in strokes:
        n=len(stroke); t=np.linspace(0,1,n)
        x,y=stroke.T; x=x-.5; y=y-.5
        a=(width*(x*(1+taper*2*y)+bend*np.cos(np.pi*y)+wave*np.sin(2*np.pi*y)) - lean*y
           +ripple*np.sin(2*np.pi*(x+y)+phase)+normal(.017)*np.sin(2*np.pi*t))
        b=(y+yshear*x+waist*np.cos(np.pi*y)+squash*x*y+left*np.cos(np.pi*(x+.55))+right*np.sin(np.pi*2*x)+across*np.sin(2*np.pi*(y-x)+phase)+tilted*y*x*x)
        more.append(np.stack((a,b),axis=1)); r=thickness*.5*(1+U(.01,.19)*np.sin(U(0,6)+2*np.pi*t)-tip*(np.maximum(0,1-8*t)**2+np.maximum(0,8*t-7)**2))
        radius.append(r)
    points=np.concatenate(more); r=np.concatenate(radius); bx=1+brush
    low=np.min(points-r[:,None]*[bx,1],axis=0); high=np.max(points+r[:,None]*[bx,1],axis=0)
    scale=U(19.3,20.7)/max(high-low)
    point=(points-(low+high)/2)*scale+13.5+[U(-.48,.48),U(-.48,.48)]; rad=r*scale
    # Stamp antialiased moving discs, denser samples on long Bezier segments.
    ix=np.floor(point[:,0]).astype(int); iy=np.floor(point[:,1]).astype(int)
    border=int(np.ceil(np.max(rad)*max(bx,1)+1)); grid=np.arange(-border,border+1); gx,gy=np.meshgrid(grid,grid); gx=gx.ravel(); gy=gy.ravel()
    x=ix[:,None]+gx; y=iy[:,None]+gy
    dist=np.sqrt(((x-point[:,0,None])/bx)**2+(y-point[:,1,None])**2)
    intensity=np.clip((rad[:,None]-dist)*U(.86,1.40)+.5,0,1)
    im=np.zeros(784,dtype=np.float32); np.maximum.at(im,np.clip(y,0,27)*28+np.clip(x,0,27),intensity); return im.reshape(28,28)

predictor = r'''
# Scaled integers in the coarse comparison keep the candidate set independent
# of reduction order (all of the dot products are float32-exact integers).
shelf=CONSTANTS['coarse'].astype(np.float32)
shapes=CONSTANTS['fine']; silhouettes=CONSTANTS['ink']; aspects=CONSTANTS['aspect']

def classify(z,aspect):
    n=len(z); s,l,ink=descriptors(z)
    s=np.rint(512*s)
    score=s@shelf.T
    # Preliminary comparison only: use a second scale and elastic matching below.
    per=shelf.shape[0]//10
    score=score.reshape(n,10,per)
    score-=np.minimum(np.abs(aspect[:,None,None]-aspects.reshape(1,10,per)),1.5)*4200
    top=holes(z)
    counter=CONSTANTS['counter']
    # Use the counter descriptor gently in the search; occasional open loops
    # or missing ink should not disqualify an entire shape family.
    for t in range(6):
        score-=np.abs(top[:,t,None,None]*127-counter[:,t].reshape(1,10,per))*6
    k=32
    ix=np.argpartition(score,per-k,axis=2)[:,:,per-k:]+np.arange(10)[None,:,None]*per
    ids=ix.reshape(n,-1); out=np.empty((n,10),dtype=np.float32)
    for j in range(n):
        # Local block descriptors can move by one cell with a small spring cost.
        # Both rigid and non-rigid evidence matter; the latter is particularly
        # useful at hooks and junctions written at varying heights.
        f=shapes[ids[j]].astype(np.float32)/1024
        b=silhouettes[ids[j]].astype(np.float32)/1024
        target=l[j]; ib=ink[j]
        rigid=(f*target).sum(axis=(1,2,3)); ir=(b*ib).sum(axis=(1,2))
        af=(f*f).sum(axis=1); at=(target*target).sum(axis=0)
        # Aggregate overlapping 3x3 patches of the square-root orientation map.
        # Candidate and query energies are both included (unlike a one-sided
        # local correlation). The outermost cells mostly cover the margin.
        best=None
        for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)):
            tt=np.roll(target,(dy,dx),axis=(1,2)); it=np.roll(ib,(dy,dx),axis=(0,1)); en=np.roll(at,(dy,dx),axis=(0,1))
            e=(af+en-2*(f*tt).sum(axis=1))+.32*(b-it)**2
            # The patch centers are at cells 1, 3, 5 and 7.
            local=e[:,0:7:2,0:7:2]+e[:,0:7:2,2:9:2]+e[:,2:9:2,0:7:2]+e[:,2:9:2,2:9:2]+2*(e[:,1:8:2,0:7:2]+e[:,1:8:2,2:9:2]+e[:,0:7:2,1:8:2]+e[:,2:9:2,1:8:2])+4*e[:,1:8:2,1:8:2]
            local+=.004*(abs(dx)+abs(dy))
            if best is None: best=local
            else: best=np.minimum(best,local)
        elastic=best.sum(axis=(1,2))*.25
        ar=np.minimum(np.abs(aspect[j]-aspects[ids[j]]),1.6)
        coarseval=score[j,np.arange(10)[:,None], ix[j]%per].reshape(-1)/262144
        hd=np.abs(top[j]-counter[ids[j]].astype(np.float32)/127).mean(axis=1)
        val=(.31*rigid+.12*ir+.23*coarseval-.37*elastic-.012*ar-.054*hd).reshape(10,k)
        v=np.sort(val,axis=1)
        out[j]=.55*v[:,-1]+.27*v[:,-2]+.18*v[:,-3]
    return np.argmax(out,axis=1)

def predict(x):
    labels=np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),64):
        stop=min(len(x),start+64)
        z,a=normalize(x[start:stop,0]); labels[start:stop]=classify(z,a)
    return labels
'''

if __name__ == '__main__':
    (out/'solution.py').write_text(features+predictor)
    size=14336
    banks={k:[] for k in ('coarse','fine','ink','aspect','counter')}
    for d in range(10):
        for base in range(0,size,64):
            ims=np.stack([render(d) for j in range(64)])
            z,a=normalize(ims); c,f,b=descriptors(z)
            banks['coarse'].append(np.uint8(np.rint(c*512)))
            banks['fine'].append(np.uint8(np.minimum(np.rint(f*1024),255)))
            banks['ink'].append(np.uint8(np.minimum(np.rint(b*1024),255)))
            banks['aspect'].append(a); banks['counter'].append(np.uint8(np.rint(holes(z)*127)))
        print('Rendered pen family',d,flush=True)
    np.savez(out/'constants.npz',**{k:np.concatenate(v) for k,v in banks.items()})
