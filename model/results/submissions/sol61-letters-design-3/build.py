"""Data-free ASL recognizer: analytic oriented edges and anatomical shape rules.

No input files are opened.  All geometry below is a schematic hand in palm-width
units, not a trace of a photograph.  There is no optimizer or fitted readout.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np


class Hand:
    def __init__(self):
        self.parts = []
        self.cuts = []
        self.lines = []

    def ellipse(self, x, y, rx, ry, hole=False):
        p = ('ellipse', x, y, rx, ry)
        (self.cuts if hole else self.parts).append(p)

    def bone(self, a, b, r, r2=None, wrist=False):
        self.parts.append(('bone', *a, *b, r, r if r2 is None else r2, wrist))

    def line(self, points, strength=0.35):
        for a, b in zip(points[:-1], points[1:]):
            self.lines.append((*a, *b, strength))

    def finger(self, points, radius, taper=0.78, crease=True):
        n = len(points) - 1
        for j in range(n):
            self.bone(points[j], points[j+1], radius*(1-(1-taper)*j/n),
                      radius*(1-(1-taper)*(j+1)/n))
        if crease and n > 1:
            for j in range(1, n):
                x,y = points[j]
                dx,dy = np.subtract(points[j+1],points[j-1])
                q = math.hypot(dx,dy)
                nx,ny = -dy/q,dx/q
                rr = radius * 0.65
                self.line([(x-nx*rr,y-ny*rr),(x+nx*rr,y+ny*rr)],0.15)

    def bounds(self):
        left,right,top = 1e9,-1e9,-1e9
        for p in self.parts:
            if p[0] == 'ellipse':
                _,x,y,rx,ry=p
                l,r,t=x-rx,x+rx,y+ry
            else:
                _,ax,ay,bx,by,r1,r2,wrist=p
                if wrist:
                    continue
                l,r,t=min(ax-r1,bx-r2),max(ax+r1,bx+r2),max(ay+r1,by+r2)
            left,right,top=min(left,l),max(right,r),max(top,t)
        return left,right,top


def anatomy(letter, style):
    """Conventional static ASL gestures, pinky on the left, thumb on the right.

    style changes proportions and finger articulation by explicit rules.  These
    alternatives express anatomical uncertainty; no observations choose them.
    """
    h=Hand()
    z=style
    xs=np.array([-.36,-.13,.11,.33])
    roots=np.array([.34,.46,.49,.40])
    lens=np.array([.83,1.08,1.20,1.08])*(1+0.12*z)
    radii=np.array([.092,.108,.113,.117])*(1+0.08*z)

    def palm(narrow=1.0):
        h.ellipse(0,-.05,.475*narrow,.58)
        h.bone((0,-.40),(-.05,-1.75),.265*narrow,.23*narrow,wrist=True)
        h.line([(-.28,-.1),(-.10,-.20),(.20,-.18)],.12)
        h.line([(.31,.18),(.18,-.08),(.18,-.35)],.14)

    def folded(j, lift=0, internal=True):
        x=xs[j]
        y=roots[j]
        h.finger([(x,y-.27),(x-.025*z,y+.14+lift),(x+.025,y-.03+lift)],
                 radii[j]*1.08,0.92,False)
        if internal:
            h.line([(x-.075,y+.02+lift),(x+.06,y-.015+lift)],.34)
            h.line([(x-.085,y+.08),(x-.085,y-.16)],.23)

    def extended(j, angle=0, length=1.0):
        x,y=xs[j],roots[j]
        a=math.radians(angle)
        ll=lens[j]*length
        bend=math.radians(3*z)
        pts=[(x,y), (x+math.sin(a)*ll*.52,y+math.cos(a)*ll*.52),
             (x+math.sin(a+bend)*ll,y+math.cos(a+bend)*ll)]
        h.finger(pts,radii[j])

    def thumb_across(high=0):
        h.finger([(.43,-.24),(.27,.04+high),(-.15,.13+high)],.13,.83,False)
        h.line([(.38,-.13),(.20,.10+high),(-.15,.19+high)],.47)

    if letter in 'CGHOPQ':
        # Side views use a different palm silhouette, rather than rotating a
        # frontal open hand, because the thumb/index opposition remains visible.
        if letter=='C':
            h.bone((-.45,-.45),(-.48,-1.8),.235,.20,wrist=True)
            h.bone((-.40,-.38),(-.38,.28),.26,.25)
            h.finger([(-.38,.23),(-.22,.67),(.14,.86),(.48,.75-.06*z)],.18,.76,False)
            h.finger([(-.26,-.24),(.08,-.10),(.49,.14+.05*z)],.13,.74,False)
            h.line([(-.37,.16),(-.27,.45),(.07,.67),(.38,.65)],.26)
            h.line([(-.35,-.25),(-.12,-.08),(.13,.03)],.20)
        elif letter=='O':
            h.bone((-.22,-.38),(-.47,-1.8),.24,.20,wrist=True)
            h.ellipse(-.05,.08,.46,.59)
            h.ellipse(.14,.43,.42,.43)
            h.ellipse(.20,.39,.22+.025*z,.235+.025*z,hole=True)
            h.line([(-.30,.24),(-.21,.53),(.03,.70),(.29,.67)],.25)
            h.line([(.04,.03),(.35,.10),(.52,.30)],.43)
        elif letter in 'GH':
            h.bone((-.50,-.28),(-1.35,-1.28),.23,.24,wrist=True)
            h.ellipse(-.30,.02,.34,.45)
            if letter=='G':
                h.finger([(-.30,.36),(.48,.43),(1.26,.47+.06*z)],.105,.73,False)
                h.finger([(-.14,-.13),(.32,-.02),(.83,.09+.05*z)],.115,.78,False)
                h.line([(-.48,.13),(-.19,.20),(.08,.15)],.30)
            else:
                gap=.21+.025*z
                h.finger([(-.28,.37),(.47,.49),(1.25,.57)],.105,.75,False)
                h.finger([(-.20,.37-gap),(.53,.49-gap),(1.32,.57-gap)],.112,.75,False)
                h.line([(-.50,.0),(-.25,.08),(.14,.06)],.30)
            h.line([(-.47,-.11),(-.18,-.03),(-.03,.19)],.22)
        elif letter=='P':
            h.bone((-.23,-.23),(-.61,-1.65),.245,.23,wrist=True)
            h.ellipse(-.06,.24,.37,.40)
            h.finger([(-.07,.47),(-.44,.83),(-.83,1.11+.08*z)],.115,.76,False)
            h.finger([(.17,.28),(.57,-.02),(.96,-.34-.08*z)],.125,.75,False)
            h.finger([(-.25,.03),(.13,-.02),(.40,.01)],.13,.82,False)
            h.line([(-.29,.29),(-.10,.04),(.20,-.03)],.42)
            h.line([(-.01,.37),(.30,.22),(.51,.01)],.26)
        else: # Q: G with the extended index pointing down.
            h.bone((-.26,-.24),(-.38,-1.75),.24,.22,wrist=True)
            h.ellipse(-.07,.35,.40,.48)
            h.finger([(.23,.55),(.45,.04),(.70,-.57-.10*z)],.12,.74,False)
            h.finger([(-.25,.30),(-.13,-.04),(.04,-.34)],.12,.77,False)
            h.line([(-.28,.49),(-.09,.32),(.20,.30)],.36)
            h.line([(.09,.36),(.22,.03),(.41,-.26)],.23)
        return h

    palm(.94 if letter in 'BDRU' else 1.0)
    if letter=='B':
        for j in range(4):
            extended(j,(-3+2*j)*(1+.2*z))
        thumb_across(-.13)
    elif letter=='D':
        for j in range(3): folded(j,-.07)
        extended(3,-3+4*z,1.08)
        h.ellipse(.10,.15,.31,.30)
        h.ellipse(.12,.20,.14,.13,hole=True)
        h.finger([(.44,-.18),(.39,.10),(.12,.37)],.12,.83,False)
        h.line([(-.26,.13),(-.04,.02),(.25,.02)],.30)
    elif letter=='F':
        for j,angle in enumerate([-28,-12,5]): extended(j,angle+3*z)
        h.finger([(.31,.41),(.62,.39),(.64,.13),(.46,.03)],.115,.83,False)
        h.finger([(.36,-.24),(.61,-.02),(.64,.17)],.13,.83,False)
        h.ellipse(.43,.17,.13,.12,hole=True)
        h.line([(.32,-.17),(.44,-.01),(.60,.06)],.34)
    elif letter in 'IY':
        extended(0,-12-4*z,1.05)
        for j in [1,2,3]: folded(j)
        if letter=='Y':
            h.finger([(.39,-.18),(.74,.16),(1.12,.43+.06*z)],.13,.75,False)
            h.line([(.36,-.23),(.56,-.01),(.72,.08)],.27)
        else: thumb_across(-.02)
    elif letter=='L':
        for j in range(3): folded(j,-.06)
        extended(3,-3+4*z,1.04)
        h.finger([(.35,-.25),(.75,.02),(1.17,.19+.06*z)],.13,.75,False)
        h.line([(.26,-.22),(.49,-.05),(.62,-.01)],.28)
    elif letter in 'KRUVW':
        for j in range(4):
            if j < (1 if letter=='W' else 2): folded(j)
        if letter=='R':
            h.finger([(xs[2],roots[2]),(.30,1.13),(.33,1.58+.04*z)],radii[2])
            h.finger([(xs[3],roots[3]),(.15,1.12),(-.01,1.51+.05*z)],radii[3])
            h.line([(.12,.90),(.23,1.17),(.31,1.30)],.43)
        elif letter=='U':
            extended(2,-3+3*z)
            extended(3,-2+3*z,1.06)
        elif letter=='V':
            extended(2,-17-4*z)
            extended(3,16+4*z,1.04)
        elif letter=='W':
            extended(1,-23-3*z)
            extended(2,-2)
            extended(3,21+3*z,1.04)
        else:
            extended(2,-25-4*z)
            extended(3,2+3*z,1.04)
        if letter=='K':
            h.finger([(.43,-.20),(.24,.20),(.06,.61)],.125,.76,False)
            h.line([(.34,-.14),(.13,.20),(-.02,.47)],.45)
        else: thumb_across(-.03)
    elif letter=='X':
        for j in range(3): folded(j,-.03)
        h.finger([(.32,.40),(.34,.97),(.11,1.22),(-.13,1.12-.06*z)],.125,.78,False)
        h.line([(.23,.77),(.23,1.00),(.04,1.10)],.28)
        thumb_across(-.03)
    else:
        # A, E, M, N, S, T: closed hand with distinctive thumb placement.
        for j in range(4): folded(j,.05 if letter=='E' else 0)
        if letter=='A':
            h.finger([(.43,-.28),(.53,.13),(.49,.58+.04*z)],.13,.76,False)
            h.line([(.35,-.23),(.40,.15),(.39,.51)],.47)
        elif letter=='E':
            h.finger([(.43,-.22),(.17,-.16),(-.23,-.13)],.13,.85,False)
            h.line([(-.32,.15),(-.08,.05),(.26,.09),(.39,.19)],.55)
            for j in range(4):
                x=xs[j]
                h.line([(x-.065,.23),(x+.045,.20)],.37)
        elif letter=='S':
            h.finger([(.43,-.13),(.14,.21),(-.22,.25+.04*z)],.14,.85,False)
            h.line([(.35,-.13),(.10,.12),(-.24,.16)],.51)
            h.line([(-.25,.40),(-.05,.43),(.25,.42)],.30)
        else:
            # Thumb tip appears between the third, second or first finger.
            tip={'M':(-.25,.40),'N':(-.04,.47),'T':(.19,.66)}[letter]
            h.finger([(.40,-.17),(.19,.10),tip],.12,.78,False)
            h.line([(.35,-.20),(.12,.05),(tip[0]-.07,tip[1]-.08)],.45)
            if letter in 'MN':
                count=3 if letter=='M' else 2
                for j in range(4-count,4):
                    x=xs[j]
                    h.line([(x-.08,.43),(x-.08,.12),(x-.01,-.02)],.41)
            else:
                h.line([(.05,.48),(.11,.25),(.28,.19)],.39)
    return h


def primitive(p,x,y):
    """Signed interior distance and its unit normal, from closed formulas."""
    if p[0]=='ellipse':
        _,cx,cy,rx,ry=p
        xx=(x-cx)/rx; yy=(y-cy)/ry
        q=np.sqrt(xx*xx+yy*yy+1e-18)
        d=(1-q)*min(rx,ry)
        nx=-xx/rx; ny=-yy/ry
    else:
        _,ax,ay,bx,by,r1,r2,_=p
        dx=bx-ax;dy=by-ay
        t=np.clip(((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy+1e-18),0,1)
        xx=x-ax-t*dx; yy=y-ay-t*dy
        q=np.sqrt(xx*xx+yy*yy+1e-18)
        d=r1+(r2-r1)*t-q
        nx=-xx;ny=-yy
    nn=np.sqrt(nx*nx+ny*ny+1e-18)
    return d,nx/nn,ny/nn


def field(h,x,y):
    d=np.full(x.shape,-1e6);nx=np.zeros_like(x);ny=np.zeros_like(x)
    for p in h.parts:
        dd,xx,yy=primitive(p,x,y)
        take=dd>d
        d=np.maximum(d,dd);nx=np.where(take,xx,nx);ny=np.where(take,yy,ny)
    for p in h.cuts:
        dd,xx,yy=primitive(p,x,y)
        take=-dd<d
        d=np.minimum(d,-dd);nx=np.where(take,-xx,nx);ny=np.where(take,-yy,ny)
    return d,nx,ny


def model_weights(h,angle,scale,tx,ty,aspect):
    # The grid integrates analytic functions over 2x2 pixel descriptor cells.
    # No input photograph, including a synthetic one, is passed through a net.
    g=(np.arange(56,dtype=np.float64)+.5)/2-.5
    xx,yy=np.meshgrid(g,g)
    co,si=math.cos(angle),math.sin(angle)
    u=(xx-13.5-tx)/scale; v=(yy-13.5-ty)/scale
    bx=(co*u+si*v)/aspect+13.5; by=-si*u+co*v+13.5
    left,right,top=h.bounds()
    natural=min(24/(right-left),25/(top+.95))
    cx=(left+right)/2
    x=cx+(bx-13.5)/natural
    y=top-(by-2)/natural
    pixel=natural*scale
    d,nx,ny=field(h,x,y)
    # Normal vectors transform by the inverse transpose of aspect scaling.
    nx=nx/aspect
    nn=np.sqrt(nx*nx+ny*ny+1e-18)
    nx/=nn;ny/=nn
    # Convert mathematical y-up normals to image y-down, then rotate them.
    sx=co*nx+si*ny; sy=si*nx-co*ny
    # Boundary-integral quadrature, including signed normal cancellation.
    # Adjacent fingers do not give two strong contours when their gap is
    # narrower than the optical blur.  This computes a continuous shape model,
    # not a forward pass on any photograph or training example.
    contour=np.exp(-.5*(d*pixel/.35)**2)/(.35*math.sqrt(2*math.pi))
    gx=contour*sx;gy=contour*sy
    ink=np.zeros_like(x)
    cot=1/math.tan(math.pi/8)
    for ax,ay,bx,by,strength in h.lines:
        dx,dy=bx-ax,by-ay
        t=np.clip(((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy+1e-18),0,1)
        dist=np.sqrt((x-ax-t*dx)**2+(y-ay-t*dy)**2)
        line=np.exp(-.5*(dist*pixel/.67)**2)
        inside=np.clip(d*pixel+0.5,0,1)
        line*=inside
        ink=np.maximum(ink,line*strength)
        q=math.hypot(dx,dy)
        nnx,nny=-dy/q/aspect,dx/q
        nq=math.hypot(nnx,nny);nnx/=nq;nny/=nq
        nsx=co*nnx+si*nny;nsy=si*nnx-co*nny
        side=((x-ax)*(-dy/q)+(y-ay)*(dx/q))*pixel
        dg=line*strength*side/(.67*.67)
        gx+=dg*nsx;gy+=dg*nsy
    radius=5
    offsets=np.arange(-radius,radius+1)
    kernel=np.exp(-.5*(offsets/(2*.85))**2);kernel/=kernel.sum()
    def smooth(v):
        for axis in (0,1):
            pad=[(0,0),(0,0)];pad[axis]=(radius,radius)
            vp=np.pad(v,pad,mode='edge')
            out=np.zeros_like(v)
            for j,k in enumerate(kernel):
                sl=[slice(None),slice(None)];sl[axis]=slice(j,j+v.shape[axis])
                out+=k*vp[tuple(sl)]
            v=out
        return v
    gx=smooth(gx);gy=smooth(gy)
    edges=[]
    for j in range(8):
        a=j*math.pi/8
        along=np.abs(gx*math.cos(a)+gy*math.sin(a))
        across=np.abs(-gx*math.sin(a)+gy*math.cos(a))
        edges.append(np.maximum(along-cot*across,0))
    edges=np.stack(edges)
    # Suppress the common wrist without removing informative palm boundaries.
    edges*=.65+.35/(1+np.exp(-np.clip((y+.20)*12,-45,45)))
    edges=edges.reshape(8,14,4,14,4).mean((2,4))[:,1:-1,1:-1]
    # Surround inhibition rewards aligned edges and penalizes nearby edges with
    # the wrong direction.  It is a fixed angular unsharp mask, not a fit.
    edges=edges-0.20*edges.mean(axis=0,keepdims=True)
    e=edges.ravel()
    e=e-e.mean()
    e/=math.sqrt(float(e@e)+1e-12)
    # A low-weight silhouette term, invariant to the foreground contrast sign.
    mask=1/(1+np.exp(-np.clip(d*pixel/.55,-45,45)))
    shade=mask*(1-.55*ink)
    v=shade.reshape(14,4,14,4).mean((1,3)).ravel()
    v-=v.mean();v/=math.sqrt(float(v@v)+1e-12)
    # A pair of opposite edges is a finger-width ridge.  The theoretical ridge
    # response is accumulated along the declared finger center lines only.
    rg=np.arange(28,dtype=np.float64)
    rxx,ryy=np.meshgrid(rg,rg)
    ru=(rxx-13.5-tx)/scale;rv=(ryy-13.5-ty)/scale
    rbx=(co*ru+si*rv)/aspect+13.5;rby=-si*ru+co*rv+13.5
    rx=cx+(rbx-13.5)/natural;ry=top-(rby-2)/natural
    ridge=np.zeros((8,28,28),np.float64)
    for p in h.parts:
        if p[0]!='bone':continue
        _,ax,ay,bx,by,r1,r2,wrist=p
        dx,dy=bx-ax,by-ay
        length=math.hypot(dx,dy)
        if wrist or length<2.6*max(r1,r2):continue
        t=np.clip(((rx-ax)*dx+(ry-ay)*dy)/(length*length),0,1)
        dist=np.sqrt((rx-ax-t*dx)**2+(ry-ay-t*dy)**2)
        rr=(r1+(r2-r1)*t)*pixel*math.sqrt(aspect)
        response=np.exp(-.5*(dist*pixel/.72)**2)
        response*=np.minimum(length*pixel/5,1)
        nnx,nny=-dy/length/aspect,dx/length
        q=math.hypot(nnx,nny);nnx/=q;nny/=q
        nsx=co*nnx+si*nny;nsy=si*nnx-co*nny
        for j in range(8):
            a=j*math.pi/8
            val=max(abs(nsx*math.cos(a)+nsy*math.sin(a))-
                    cot*abs(-nsx*math.sin(a)+nsy*math.cos(a)),0)
            width=np.exp(-.5*((rr-1.15)/.80)**2)
            ridge[j]=np.maximum(ridge[j],response*val*width)
    ridge=ridge[:,2:-2,2:-2].reshape(8,12,2,12,2).mean((2,4))
    r=ridge.ravel();r-=r.mean();r/=math.sqrt(float(r@r)+1e-12)
    # Signed normal flow distinguishes a coherent silhouette from unrelated
    # edges.  The network takes its absolute correlation to allow either
    # foreground/background contrast polarity.
    flow=np.stack((gx,gy))
    flow*=.65+.35/(1+np.exp(-np.clip((y+.20)*12,-45,45)))
    flow=flow.reshape(2,14,4,14,4).mean((2,4))[:,1:-1,1:-1]
    f=flow.ravel();f-=f.mean();f/=math.sqrt(float(f@f)+1e-12)
    return (e.astype(np.float32),(.35*v).astype(np.float32),
            (.12*r).astype(np.float32),(.65*f).astype(np.float32))


def build(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    layers=[];weights={}
    def op(name,kind,inputs,**kw):
        layers.append(dict(name=name,op=kind,inputs=inputs,**kw));return name
    def conv(name,src,w,b=None,padding=0):
        w=np.asarray(w,np.float32)
        op(name,'conv',[src],out=w.shape[0],kernel=[w.shape[2],w.shape[3]],stride=1,padding=padding)
        weights[name+'.W']=w
        weights[name+'.b']=np.zeros(w.shape[0],np.float32) if b is None else np.asarray(b,np.float32)
        return name
    def dense(name,src,w,b=None):
        w=np.asarray(w,np.float32)
        op(name,'dense',[src],out=w.shape[0]);weights[name+'.W']=w
        weights[name+'.b']=np.zeros(w.shape[0],np.float32) if b is None else np.asarray(b,np.float32)
        return name

    op('contrast','norm',['image'])
    q=np.arange(-2,3,dtype=np.float64);xx,yy=np.meshgrid(q,q)
    gaussian=np.exp(-(xx*xx+yy*yy)/(2*.85**2))
    derivative=np.stack((xx*gaussian,yy*gaussian))
    derivative/=np.sum(xx*xx*gaussian)
    filters=[]
    for j in range(8):
        a=j*math.pi/8
        f=math.cos(a)*derivative[0]+math.sin(a)*derivative[1]
        filters.extend([f,-f])
    conv('gradients','contrast',np.array(filters)[:,None])
    conv('normal_flow','contrast',derivative[:,None])
    op('flow_cells','avgpool',['normal_flow'],size=[2,2],stride=[2,2])
    op('flow_strip','reshape',['flow_cells'],shape=[1,24,12])
    op('flow_energy','norm',['flow_strip'])
    op('signed_edges','relu',['gradients'])
    w=np.zeros((8,16,1,1),np.float32)
    for j in range(8):w[j,2*j:2*j+2,0,0]=1
    conv('absolute_projections','signed_edges',w)
    w=np.eye(8,dtype=np.float32)
    for j in range(8):w[j,(j+4)%8]=-1/math.tan(math.pi/8)
    conv('angular_bins','absolute_projections',w[:,:,None,None])
    op('edge_bins','relu',['angular_bins'])
    conv('edge_headroom','edge_bins',-np.eye(8)[:,:,None,None],np.full(8,.70))
    op('positive_headroom','relu',['edge_headroom'])
    conv('bounded_edges','positive_headroom',-np.eye(8)[:,:,None,None],np.full(8,.70))
    op('edge_cells','avgpool',['bounded_edges'],size=[2,2],stride=[2,2])
    op('edge_strip','reshape',['edge_cells'],shape=[1,96,12])
    op('edge_energy','norm',['edge_strip'])
    op('gray_cells','avgpool',['contrast'],size=[2,2],stride=[2,2])

    # Contrast-independent ridge detectors: a central strip opposed by two
    # flanking strips.  All kernels are Gaussian derivative formulas.
    rq=np.arange(-2,3,dtype=np.float64);rxx,ryy=np.meshgrid(rq,rq)
    rf=[]
    for j in range(8):
        a=j*math.pi/8
        across=rxx*math.cos(a)+ryy*math.sin(a)
        along=-rxx*math.sin(a)+ryy*math.cos(a)
        f=(1-(across/.85)**2)*np.exp(-.5*(across/.85)**2-.5*(along/1.7)**2)
        f-=f.mean();f/=np.sum(np.maximum(f,0))
        rf.extend([f,-f])
    conv('finger_widths','contrast',np.asarray(rf)[:,None])
    op('ridge_polarities','relu',['finger_widths'])
    rw=np.zeros((8,16,1,1),np.float32)
    for j in range(8):rw[j,2*j:2*j+2,0,0]=1
    conv('ridge_magnitudes','ridge_polarities',rw)
    op('ridge_cells','avgpool',['ridge_magnitudes'],size=[2,2],stride=[2,2])
    op('ridge_strip','reshape',['ridge_cells'],shape=[1,96,12])
    op('ridge_energy','norm',['ridge_strip'])

    letters='ABCDEFGHIKLMNOPQRSTUVWXY'
    nvariant=3*7*3*3*3*3
    nmodels=24*nvariant
    ew=np.empty((nmodels,8*12*12),np.float32)
    vw=np.empty((nmodels,14*14),np.float32)
    rw=np.empty((nmodels,8*12*12),np.float32)
    fw=np.empty((nmodels,2*12*12),np.float32)
    row=0
    for letter in letters:
        for style in [-1,0,1]:
            h=anatomy(letter,style)
            for degrees in [-36,-24,-12,0,12,24,36]:
                for scale in [.78,1.,1.22]:
                    for ty in [-2.5,0.,2.5]:
                        for tx in [-2.5,0.,2.5]:
                            for aspect in [.65,.90,1.15]:
                                ew[row],vw[row],rw[row],fw[row]=model_weights(
                                    h,math.radians(degrees),scale,tx,ty,aspect)
                                row+=1
        print('Constructed anatomical models:',letter,flush=True)
    dense('contour_match','edge_energy',ew)
    dense('silhouette_match','gray_cells',vw)
    dense('finger_match','ridge_energy',rw)
    dense('flow_match','flow_energy',fw)
    op('flow_column','reshape',['flow_match'],shape=[1,nmodels,1])
    op('positive_flow','relu',['flow_column'])
    conv('invert_flow','flow_column',np.array([[[[-1.]]]]))
    op('negative_flow','relu',['invert_flow'])
    op('either_flow','add',['positive_flow','negative_flow'])
    op('flow_scores','reshape',['either_flow'],shape=[nmodels])
    op('silhouette_column','reshape',['silhouette_match'],shape=[1,nmodels,1])
    op('bright_hand','relu',['silhouette_column'])
    conv('invert_silhouette','silhouette_column',np.array([[[[-1.]]]]))
    op('dark_hand','relu',['invert_silhouette'])
    op('either_contrast','add',['bright_hand','dark_hand'])
    op('silhouette_scores','reshape',['either_contrast'],shape=[nmodels])
    op('model_scores','add',['contour_match','silhouette_scores','finger_match','flow_scores'])
    op('class_models','reshape',['model_scores'],shape=[24,nvariant,1])
    op('best_geometry','maxpool',['class_models'],size=[nvariant,1],stride=[nvariant,1])
    op('scores','reshape',['best_geometry'],shape=[24])
    with (folder/'network.json').open('w') as f:json.dump({'layers':layers},f,indent=2)
    np.savez(folder/'weights.npz',**weights)
    print('Weights:',sum(a.size for a in weights.values()),flush=True)


if __name__=='__main__':
    build(sys.argv[1] if len(sys.argv)>1 else '.')
