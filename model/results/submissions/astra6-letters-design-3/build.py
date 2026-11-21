"""Data-free, analytic ASL classifier. No inputs other than the output directory.

All lengths below are dimensionless anatomical drawing conventions, not image
measurements. Templates are generated directly from ellipses, tapered finger
segments and crease lines. No training, optimization, or stored examples.
"""
import sys
import json
import math
from pathlib import Path
import numpy as np

LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'
VARIANTS = 768


def hand(letter, variation):
    """A small vector drawing, in palm-width units, seen from the front.

    Each primitive is an ellipse or a tapered capsule. The wrist is marked
    separately so it does not determine the crop of the hand. Creases describe
    anatomical finger separations and folds rather than photographic texture.
    """
    shapes, creases = [], []
    spread, longness, side = variation

    def ellipse(x, y, rx, ry, wrist=False):
        shapes.append(('e', x, y, rx, ry, bool(wrist)))

    def segment(a, b, r, r2=None, wrist=False):
        shapes.append(('s', *a, *b, r, r if r2 is None else r2, bool(wrist)))

    def chain(points, radius, taper=0.88):
        for j, (a, b) in enumerate(zip(points[:-1], points[1:])):
            segment(a, b, radius, radius * (taper if j == len(points)-2 else 1))

    def crease(points, strength=0.45):
        for a, b in zip(points[:-1], points[1:]):
            creases.append((*a, *b, strength))

    xs = [0.42, 0.14, -0.14, -0.42]
    lengths = [1.20, 1.38, 1.28, 1.02]
    radii = [0.145, 0.15, 0.145, 0.125]

    def finger(i, angle=0, length=1, base_y=-0.20):
        a = (xs[i], base_y)
        theta = math.radians(angle)
        length = lengths[i] * longness * length
        b = (a[0] + math.sin(theta)*length, a[1] - math.cos(theta)*length)
        segment(a, b, radii[i], radii[i]*0.83)
        # Small transverse interphalangeal folds.
        t = 0.47
        x, y = (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]))
        dx, dy = math.cos(theta)*radii[i]*0.65, math.sin(theta)*radii[i]*0.65
        crease([(x-dx,y-dy),(x+dx,y+dy)],0.20)

    def folded(i, high=0.0, deep=0.0):
        x, r = xs[i], radii[i]
        top = -0.43 + high - 0.10*(i in (1,2))
        segment((x,0.12),(x,top),r*1.12,r)
        crease([(x-r*0.7,top+0.18),(x-r*0.65,0.08+deep),
                (x+r*0.45,0.14+deep)],0.62)

    def thumb(mode='across'):
        if mode == 'out':
            chain([(0.37,0.65),(0.72,0.35),(1.29,0.22)],0.19)
            crease([(0.43,0.57),(0.54,0.22)],0.35)
        elif mode == 'up':
            chain([(0.53,0.66),(0.72,0.19),(0.66,-0.48)],0.19)
            crease([(0.48,0.48),(0.49,-0.25)],0.60)
        elif mode == 'across':
            chain([(0.55,0.51),(0.31,0.16),(-0.14,0.03)],0.19)
            crease([(0.53,0.29),(0.22,-0.03),(-0.19,-0.12)],0.65)
        elif mode == 'under':
            chain([(0.53,0.48),(0.20,0.52),(-0.27,0.38)],0.18)
            crease([(-0.37,0.20),(0.10,0.27),(0.50,0.32)],0.68)

    # C and O are naturally described by the side-view curve of the hand.
    if letter in 'CO':
        ellipse(-0.39,0.24,0.37,0.67)
        chain([(-0.50,0.28),(-0.51,-0.36),(-0.25,-0.84),
               (0.27,-0.98),(0.68,-0.76)],0.23)
        if letter == 'C':
            chain([(-0.35,0.68),(0.08,0.76),(0.50,0.51),(0.71,0.15)],0.18)
        else:
            chain([(0.68,-0.76),(0.81,-0.36),(0.68,-0.03)],0.21)
            chain([(-0.35,0.68),(0.11,0.56),(0.52,0.17),(0.68,-0.03)],0.21)
        crease([(-0.42,-0.36),(-0.14,-0.67),(0.28,-0.78),(0.56,-0.62)],0.36)
        segment((-0.34,0.67),(-0.42,2.5),0.36,0.40,True)
    # The sideways pointing signs use a closed palm and horizontal digits.
    elif letter in 'GH':
        ellipse(-0.32,0.25,0.50,0.59)
        ellipse(-0.16,-0.16,0.49,0.30)
        if letter == 'G':
            chain([(-0.05,-0.26),(0.53,-0.45),(1.53,-0.45)],0.13)
            chain([(-0.35,0.17),(0.18,0.01),(0.87,-0.11)],0.16)
            crease([(-0.27,-0.12),(0.0,0.14),(0.04,0.49)],0.55)
        else:
            chain([(-0.10,-0.29),(0.57,-0.53),(1.55,-0.58)],0.14)
            chain([(-0.03,-0.03),(0.56,-0.22),(1.45,-0.24)],0.14)
            crease([(0.08,-0.17),(0.65,-0.36),(1.27,-0.40)],0.65)
            crease([(-0.65,0.13),(-0.23,0.22),(0.06,0.14)],0.40)
        segment((-0.40,0.65),(-0.88,2.2),0.35,0.43,True)
    elif letter in 'PQ':
        ellipse(-0.12,-0.20,0.53,0.43)
        if letter == 'P':
            chain([(-0.17,-0.40),(0.45,-0.29),(1.13,-0.13)],0.14)
            chain([(0.13,-0.15),(0.48,0.28),(0.54,1.32)],0.14)
            chain([(-0.41,-0.15),(-0.21,0.35),(0.28,0.54)],0.18)
            crease([(-0.19,-0.33),(0.11,0.10),(0.24,0.56)],0.50)
        else:
            chain([(0.28,-0.18),(0.51,0.45),(0.53,1.33)],0.145)
            chain([(-0.38,-0.09),(-0.40,0.54),(-0.20,1.03)],0.18)
            crease([(-0.10,-0.47),(0.05,-0.06),(0.19,0.16)],0.50)
        segment((-0.43,-0.32),(-1.22,1.70),0.35,0.42,True)
    else:
        ellipse(0,0.36,0.59,0.67)
        segment((0.0,0.83),(-0.03,2.7),0.36,0.42,True)
        if letter == 'A':
            for i in range(4): folded(i)
            thumb('up')
        elif letter == 'B':
            for i in range(4): finger(i, (1.5-i)*spread*1.6)
            thumb('under')
            for i in range(3):
                x=(xs[i]+xs[i+1])/2
                crease([(x,-0.16),(x,-1.02*longness)],0.62)
        elif letter == 'D':
            finger(0,-9*side)
            for i in range(1,4): folded(i,0.08)
            chain([(0.54,0.54),(0.48,0.06),(0.07,-0.25)],0.17)
            crease([(-0.29,-0.27),(-0.32,0.17),(0.11,0.24),(0.33,-0.05)],0.58)
        elif letter == 'E':
            for i in range(4): folded(i,-0.10,0.12)
            thumb('under')
            crease([(-0.48,0.05),(0.0,0.20),(0.41,0.12)],0.72)
        elif letter == 'F':
            for i in (1,2,3): finger(i,(1.5-i)*spread*13)
            chain([(0.40,-0.19),(0.76,-0.32),(0.92,-0.03),(0.73,0.28)],0.14)
            chain([(0.38,0.66),(0.77,0.41),(0.73,0.28)],0.17)
            crease([(0.28,0.07),(0.45,0.30),(0.68,0.38)],0.36)
        elif letter == 'I':
            finger(3,-14*spread)
            for i in range(3): folded(i)
            thumb('across')
        elif letter == 'K':
            finger(0,10*spread)
            finger(1,-31*spread,0.85)
            for i in (2,3): folded(i)
            chain([(0.52,0.55),(0.29,0.04),(0.08,-0.56)],0.18)
            crease([(0.01,-0.48),(0.11,0.05),(0.40,0.35)],0.56)
        elif letter == 'L':
            finger(0,0)
            for i in (1,2,3): folded(i,0.16)
            thumb('out')
        elif letter in 'MN':
            for i in range(4): folded(i,-0.10,0.12)
            # The thumb is tucked beneath three or two fingers, respectively.
            i = 2 if letter == 'M' else 1
            x = xs[i]-0.10
            segment((0.48,0.52),(x,0.19),0.17,0.14)
            crease([(x-0.08,0.03),(x,0.25),(x+0.10,0.27)],0.80)
            crease([(-0.43,0.38),(0.02,0.46),(0.38,0.33)],0.40)
        elif letter == 'R':
            # Crossed index and middle fingers: one bends across the other.
            chain([(xs[0],-0.16),(0.30,-0.77),(-0.01,-1.45)],0.145)
            chain([(xs[1],-0.18),(0.17,-0.90),(0.40,-1.55)],0.145)
            for i in (2,3): folded(i)
            thumb('across')
            crease([(0.12,-0.73),(0.28,-1.07)],0.65)
        elif letter == 'S':
            for i in range(4): folded(i)
            thumb('across')
        elif letter == 'T':
            for i in range(4): folded(i,0.09)
            chain([(0.55,0.50),(0.34,0.05),(0.27,-0.67)],0.19)
            crease([(0.12,-0.51),(0.12,0.04),(0.36,0.23)],0.66)
        elif letter in 'UVW':
            active = (0,1,2) if letter == 'W' else (0,1)
            angles = {'U': (0,-2,0), 'V': (15,-16,0), 'W': (20,0,-22)}[letter]
            for i in active: finger(i,angles[i]*spread)
            for i in range(4):
                if i not in active: folded(i,0.10)
            thumb('across')
            if letter == 'U':
                crease([(0.28,-0.28),(0.28,-1.29*longness)],0.70)
        elif letter == 'X':
            chain([(0.40,-0.12),(0.38,-0.72),(0.66,-1.03),(0.92,-0.85)],0.145)
            for i in (1,2,3): folded(i)
            thumb('across')
            crease([(0.53,-0.83),(0.71,-0.83)],0.40)
        elif letter == 'Y':
            finger(3,-30*spread)
            for i in (0,1,2): folded(i,0.1)
            thumb('out')
        else:
            raise ValueError(letter)
        # A faint curved palm fold, common to open hands.
        if letter in 'BDFIKLUVWY':
            crease([(-0.40,0.32),(-0.13,0.40),(0.17,0.31)],0.22)
    return shapes, creases


def shape_bounds(shapes):
    points=[]
    for s in shapes:
        if s[-1]:
            continue
        if s[0]=='e':
            _,x,y,rx,ry,_=s
            points.extend([(x-rx,y-ry),(x+rx,y+ry)])
        else:
            _,ax,ay,bx,by,r1,r2,_=s
            points.extend([(ax-r1,ay-r1),(ax+r1,ay+r1),
                           (bx-r2,by-r2),(bx+r2,by+r2)])
    p=np.asarray(points)
    return p.min(0),p.max(0)


def field(shapes, creases, x, y):
    """Signed distance and normal to an analytic union, plus crease geometry."""
    best=np.full(x.shape,1e9,dtype=np.float64)
    nx=np.zeros_like(x); ny=np.zeros_like(y)
    for s in shapes:
        if s[0]=='e':
            _,cx,cy,rx,ry,_=s
            u,v=(x-cx)/rx,(y-cy)/ry
            rho=np.sqrt(u*u+v*v+1e-10)
            # First-order metric correction to elliptical radial distance.
            gx,gy=u/rx,v/ry
            g=np.sqrt(gx*gx+gy*gy+1e-10)
            d=(rho-1)*rho/(g+1e-10)
            dx,dy=gx/g,gy/g
        else:
            _,ax,ay,bx,by,r1,r2,_=s
            vx,vy=bx-ax,by-ay
            t=np.clip(((x-ax)*vx+(y-ay)*vy)/(vx*vx+vy*vy+1e-10),0,1)
            u,v=x-ax-t*vx,y-ay-t*vy
            rho=np.sqrt(u*u+v*v+1e-10)
            d=rho-r1-t*(r2-r1)
            dx,dy=u/rho,v/rho
        sel=d<best
        best=np.where(sel,d,best)
        nx=np.where(sel,dx,nx); ny=np.where(sel,dy,ny)
    return best,nx,ny


def parameters(j):
    """Fixed low-discrepancy coverage, not a search or fitted distribution."""
    def u(n):
        primes=[2,3,5,7,11,13,17,19,23,29,31,37]
        b=primes[n]; k=j+1; value=0.; denom=1.
        while k:
            denom*=b
            value+=(k%b)/denom
            k//=b
        return value
    # Repeated basic frontal and oblique views. Either hand is admitted.
    return dict(spread=0.65+0.75*u(0),longness=0.84+0.32*u(1),
                side=u(2),width=0.40+0.75*u(3),angle=-32+64*u(4),
                scale=0.78+0.30*u(5),tx=-2.2+4.4*u(6),ty=-2.1+4.2*u(7),
                mirror=(-1 if j%2 else 1),shear=-0.18+0.36*u(8))


def template(letter,j):
    p=parameters(j)
    shapes,creases=hand(letter,(p['spread'],p['longness'],p['side']))
    lo,hi=shape_bounds(shapes)
    center=(lo+hi)/2
    # Camera foreshortening of the palm's transverse dimension.
    width=p['width']
    scale=22.5*p['scale']/max((hi[0]-lo[0])*width,hi[1]-lo[1])
    angle=math.radians(p['angle'])
    ca,sa=math.cos(angle),math.sin(angle)
    mirror=p['mirror']
    def inverse(x,y):
        x=x-(13.5+p['tx']); y=y-(12.5+p['ty'])
        u=(ca*x+sa*y)/scale
        v=(-sa*x+ca*y)/scale
        u=(u-p['shear']*v)/(width*mirror)
        return u+center[0],v+center[1]
    # Evaluate a smooth silhouette directly on the input grid.
    yy,xx=np.mgrid[:28,:28].astype(float)
    x,y=inverse(xx,yy)
    d,_,_=field(shapes,creases,x,y)
    mask=np.clip(0.5-d*scale/1.2,0,1)
    def edge_drawing(size, origin, sigma, signed):
        # Direct vector-geometry evaluation, not features extracted from images.
        yy,xx=np.mgrid[:size,:size].astype(float)
        xx=origin+2*xx; yy=origin+2*yy
        x,y=inverse(xx,yy)
        d,nx,ny=field(shapes,creases,x,y)
        # Transform normals using the inverse-transpose camera matrix.
        gx=nx/(width*mirror)
        gy=ny-p['shear']*gx
        a,b=ca*gx-sa*gy,sa*gx+ca*gy
        n=np.sqrt(a*a+b*b+1e-10)
        a,b=a/n,b/n
        dp=d*scale/n
        amplitude=np.exp(-0.5*(dp/sigma)**2)
        angles=np.arange(8 if signed else 4)*np.pi/4
        co,si=np.cos(angles)[:,None,None],np.sin(angles)[:,None,None]
        projection=co*a+si*b
        transverse=np.abs(-si*a+co*b)
        unsigned=np.maximum(np.abs(projection)-0.7*transverse,0)
        inward=np.maximum(-projection-0.7*transverse,0)
        edges=(0.45*unsigned+0.55*inward if signed else unsigned)*amplitude
        # Union contours alone omit the visible sides of touching fingers and
        # thumbs resting across a palm. Add weak, polarity-free surface edges
        # from each anatomical segment, restricted to the inside of the hand.
        inside=np.clip(0.5-d*scale,0,1)
        for primitive in shapes:
            if primitive[0]!='s' or primitive[-1]:
                continue
            sd,sx,sy=field([primitive],[],x,y)
            gx=sx/(width*mirror)
            gy=sy-p['shear']*gx
            a,b=ca*gx-sa*gy,sa*gx+ca*gy
            n=np.sqrt(a*a+b*b+1e-10)
            a,b=a/n,b/n
            distance=sd*scale/n
            response=np.exp(-0.5*(distance/sigma)**2)*inside
            orientation=np.maximum(np.abs(co*a+si*b)-0.7*np.abs(-si*a+co*b),0)
            edges=np.maximum(edges,0.38*orientation*response)
        # Anatomical crease lines are weaker than the outside boundary.
        for ax,ay,bx,by,strength in creases:
            vx,vy=bx-ax,by-ay
            t=np.clip(((x-ax)*vx+(y-ay)*vy)/(vx*vx+vy*vy+1e-10),0,1)
            distance=np.sqrt((x-ax-t*vx)**2+(y-ay-t*vy)**2)
            gx=-vy/(width*mirror)
            gy=vx-p['shear']*gx
            a,b=ca*gx-sa*gy,sa*gx+ca*gy
            n=math.sqrt(a*a+b*b+1e-10)
            a,b=a/n,b/n
            projection=co*a+si*b
            transverse=np.abs(-si*a+co*b)
            oriented=np.maximum(np.abs(projection)-0.7*transverse,0)
            # Use physical distance perpendicular to the projected line.
            distance*=scale*math.hypot(vx,vy)/n
            line=strength*np.exp(-0.5*(distance/(sigma*0.8))**2)
            edges=np.maximum(edges,oriented*line)
        # Cropped wrists and clothing carry little class evidence.
        spatial=np.clip((29.0-yy)/7.0,0.3,1.0)
        edges*=spatial
        return edges
    # A separate dark-line map specifies interdigital valleys and flexion folds.
    # Ridge evidence cannot be replaced by a single silhouette edge, so it helps
    # distinguish a pair of touching fingers from one extended finger.
    yy,xx=np.mgrid[:14,:14].astype(float)
    x,y=inverse(0.5+2*xx,0.5+2*yy)
    ridges=np.zeros((4,14,14))
    angles=np.arange(4)*np.pi/4
    co,si=np.cos(angles)[:,None,None],np.sin(angles)[:,None,None]
    for ax,ay,bx,by,strength in creases:
        vx,vy=bx-ax,by-ay
        t=np.clip(((x-ax)*vx+(y-ay)*vy)/(vx*vx+vy*vy+1e-10),0,1)
        distance=np.sqrt((x-ax-t*vx)**2+(y-ay-t*vy)**2)
        gx=-vy/(width*mirror)
        gy=vx-p['shear']*gx
        a,b=ca*gx-sa*gy,sa*gx+ca*gy
        n=math.sqrt(a*a+b*b+1e-10)
        a,b=a/n,b/n
        distance*=scale*math.hypot(vx,vy)/n
        alignment=np.maximum(np.abs(co*a+si*b)-0.7*np.abs(-si*a+co*b),0)
        line=strength*np.exp(-0.5*(distance/1.0)**2)
        ridges=np.maximum(ridges,alignment*line)
    ridges*=np.clip((29.0-(0.5+2*yy))/7.0,0.3,1.0)
    return mask,edge_drawing(13,1.5,1.8,True),edge_drawing(14,0.5,0.9,False),ridges


def unit(v):
    v=v.astype(np.float64).reshape(-1)
    v-=v.mean()
    v/=max(np.linalg.norm(v),1e-8)*math.sqrt(len(v))
    return v.astype(np.float32)


def build(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    layers=[]; weights={}
    def layer(name,op,inputs,**kw):
        layers.append(dict(name=name,op=op,inputs=inputs,**kw))
    def conv(name,src,w,b=None,padding=0):
        w=np.asarray(w,dtype=np.float32)
        weights[name+'.W']=w
        weights[name+'.b']=np.zeros(w.shape[0],np.float32) if b is None else np.asarray(b,np.float32)
        layer(name,'conv',[src],out=w.shape[0],kernel=[w.shape[2],w.shape[3]],stride=1,padding=padding)
    def dense(name,src,w,b=None):
        w=np.asarray(w,dtype=np.float32)
        weights[name+'.W']=w
        weights[name+'.b']=np.zeros(w.shape[0],np.float32) if b is None else np.asarray(b,np.float32)
        layer(name,'dense',[src],out=w.shape[0])

    layer('contrast','norm',['image'])
    dx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)/8
    dy=dx.T
    kernels=[]
    for theta in np.arange(8)*np.pi/4:
        kernels.append(np.cos(theta)*dx+np.sin(theta)*dy)
    conv('derivatives','contrast',np.asarray(kernels)[:,None],padding=1)
    layer('positive_derivatives','relu',['derivatives'])
    competition=np.zeros((8,8,1,1),np.float32)
    for k in range(8):
        competition[k,k,0,0]=1
        competition[k,(k+2)%8,0,0]=-0.7
        competition[k,(k+6)%8,0,0]=-0.7
    conv('orientation_bins','positive_derivatives',competition)
    layer('oriented_edges','relu',['orientation_bins'])
    layer('edge_tolerance','maxpool',['oriented_edges'],size=[3,3],stride=[1,1])
    layer('edge_lattice','avgpool',['edge_tolerance'],size=[2,2],stride=[2,2])
    layer('edge_normalization','norm',['edge_lattice'])
    unsigned=np.zeros((4,8,1,1),np.float32)
    for k in range(4):
        unsigned[k,k,0,0]=unsigned[k,k+4,0,0]=1
    conv('fine_unsigned_edges','oriented_edges',unsigned)
    layer('fine_lattice','avgpool',['fine_unsigned_edges'],size=[2,2],stride=[2,2])
    layer('fine_normalization','norm',['fine_lattice'])
    # Gaussian second derivatives are fixed dark-ridge detectors.
    yy,xx=np.mgrid[-2:3,-2:3].astype(float)
    ridge_kernels=[]
    for theta in np.arange(4)*np.pi/4:
        u=xx*np.cos(theta)+yy*np.sin(theta)
        v=-xx*np.sin(theta)+yy*np.cos(theta)
        g=np.exp(-0.5*((u/0.85)**2+(v/1.4)**2))
        kernel=((u/0.85)**2-1)*g
        kernel-=kernel.mean()
        kernel/=np.abs(kernel).sum()
        ridge_kernels.append(kernel)
    conv('dark_ridges','contrast',np.asarray(ridge_kernels)[:,None],padding=2)
    layer('ridge_activation','relu',['dark_ridges'])
    layer('ridge_lattice','avgpool',['ridge_activation'],size=[2,2],stride=[2,2])
    layer('ridge_normalization','norm',['ridge_lattice'])
    layer('silhouette_lattice','avgpool',['contrast'],size=[2,2],stride=[2,2])

    count=24*VARIANTS
    edge_w=np.empty((count*2,8*13*13),np.float32)
    fine_w=np.empty((count*2,4*14*14),np.float32)
    ridge_w=np.empty((count*2,4*14*14),np.float32)
    shape_w=np.empty((count*2,14*14),np.float32)
    for k,letter in enumerate(LETTERS):
        for j in range(VARIANTS):
            i=k*VARIANTS+j
            mask,edges,fine,ridges=template(letter,j)
            # Average per-orientation cosine agreement, with no fitted readout.
            ew=np.stack([unit(e) for e in edges])*0.65/8
            edge_w[2*i]=ew.reshape(-1)
            edge_w[2*i+1]=np.roll(ew,4,axis=0).reshape(-1)
            fw=np.stack([unit(e) for e in fine])*0.35/4
            fine_w[2*i]=fine_w[2*i+1]=fw.reshape(-1)
            rw=np.stack([unit(e) for e in ridges])*0.22/4
            ridge_w[2*i]=ridge_w[2*i+1]=rw.reshape(-1)
            silhouette=0.18*unit(mask.reshape(14,2,14,2).mean((1,3)))
            shape_w[2*i]=silhouette
            shape_w[2*i+1]=-silhouette
        print('constructed',letter,flush=True)
    dense('contour_agreement','edge_normalization',edge_w)
    dense('fine_agreement','fine_normalization',fine_w)
    dense('ridge_agreement','ridge_normalization',ridge_w)
    dense('silhouette_agreement','silhouette_lattice',shape_w)
    layer('pose_agreement','add',['contour_agreement','fine_agreement','ridge_agreement','silhouette_agreement'])
    layer('class_pose_table','reshape',['pose_agreement'],shape=[24,1,VARIANTS*2])
    layer('best_pose','maxpool',['class_pose_table'],size=[1,VARIANTS*2],stride=[1,VARIANTS*2])
    layer('scores','reshape',['best_pose'],shape=[24])
    (output/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(output/'weights.npz',**weights)
    print('Weights:',sum(v.size for v in weights.values()))


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('usage: python3 build.py OUTPUT_DIRECTORY')
    build(sys.argv[1])
