"""Compile procedural ASL hand poses into a fixed analysis-by-synthesis network.

Geometry is in palm half-widths, in a palm-facing frame (positive y toward
wrist). All the variability here is a prior on the model, not a sample of
measurements. There is no input to the compiler other than its output path.
"""
import sys, json, math
from pathlib import Path
import numpy as np

LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'
NPOSE = 128
SHIFTS = (-2.7, 0., 2.7)
U = np.arange(28, dtype=np.float32) - 13.5
GA = np.exp(-.5*(np.arange(-2,3)/.92)**2); GA /= GA.sum()
DER = np.arange(-2,3)*GA; DER /= np.abs(DER).sum()
DX, DY = np.outer(GA,DER), np.outer(DER,GA)
THETA = np.arange(4)*np.pi/4
EDGE = np.stack([f for t in THETA for f in (DX*np.cos(t)+DY*np.sin(t), -DX*np.cos(t)-DY*np.sin(t))]).astype('f')


def model(c, rng, variant):
    """Build rear-to-front layers of a schematic shaded hand, and an anchor
    extent for the camera. The extent covers the hand, not the entire arm."""
    pieces=[]
    # A chain has round tapered links; a disc is an elliptical volume.
    def chain(points, r=.22, light=0., arm=False):
        p=np.array(points,dtype=float)
        r=np.broadcast_to(r,len(p)).copy()
        pieces.append(('chain',p,r,light,arm))
    def disc(cx,cy,rx,ry,angle=0.,light=0.):
        pieces.append(('disc', np.array([cx,cy,rx,ry,angle]), None,light,False))
    def wrist(x,y,ex,ey,r=.64):
        chain([(x,y),(ex,ey)],[r,r*1.1],-.02,True)
    # Global continuous variation, mirrored neither horizontally nor vertically:
    # handedness matters in the frontal projections.
    length = rng.uniform(.89,1.13); fat=rng.uniform(.9,1.12)
    splay=rng.uniform(.72,1.45); bend=rng.uniform(-.14,.14)
    lean=rng.normal(0,.10)   # shared direction of MCP joints; small independent PIP flexion
    def finger(base, direction, r=.22, offset=0., tone=0.):
        x,y=base; dx,dy=direction
        chain([(x,y),(x+dx*.55+offset,y+dy*.54),(x+dx,y+dy)],
              np.array([r*1.04,r,r*.79])*fat,tone)
    # Palm a slightly oblique egg; a narrower heel joins it to the forearm.
    def palm(narrow=1.):
        wrist(-.12*narrow,.69, (-.70+rng.uniform(-.55,.55))*narrow,3.6, .64*narrow)
        disc(-.01,.01,narrow,1.16,.12)
    base=np.array([[-.73,-.59],[-.32,-.85],[.13,-.97],[.57,-.83]])
    fl=np.array([1.51,1.89,2.12,1.95])*length
    rad=np.array([.194,.217,.226,.218])*fat
    base[:,0]+=rng.uniform(-.035,.035,4)
    if c in 'AMNSEIYLD':
        # At oblique views the outer knuckles stack instead of forming a
        # straight row. The radial side of the hand is above its ulnar side.
        slant=rng.uniform(-.12,.42) if variant%3!=0 else 0.
        base[:,1]-=slant*base[:,0]
    if c in 'RUKVW' and variant%4!=0:
        # Folding three digits into a fist tilts the metacarpal fan. The
        # straight digits start together on one side, not above the palm center.
        base[:,1]+=.27*(.5-base[:,0]);base[:,0]+=.08
    def curl(i,kind='normal',shift=0.):
        # Raised MCP knuckle -> projecting PIP -> folded tip visible on palm.
        x,y=base[i]; r=rad[i]*1.09
        lift=(.40 if kind=='low' else .60) * rng.uniform(.81,1.2)
        drift=(-.08 if i==0 else -.13)+bend
        chain([(x,y+.10),(x+drift,y-lift*.64),(x+drift-.025,y-lift)],
              [r,r,r*.94],-.03)
        chain([(x+drift-.025,y-lift),(x+.02+shift,y-lift+.28),
               (x+.06+shift,y+(.02 if kind!='low' else .23))],
              [r*.96,r*.96,r*.84],.035)
    def up(i, dx=0, dy=1, off=0.):
        finger(base[i],(dx*splay+lean+rng.uniform(-.055,.055),-fl[i]*dy*rng.uniform(.94,1.06)),rad[i]/fat,off,.03)
    def foldedthumb(high=.0):
        chain([(.71,.40),(.86,-.23),(.46,-.66+high),(-.19,-.77+high)],
              np.array([.33,.295,.247,.192])*fat,.04)
    if c in 'ABDEIKLMNR SUVWXYF'.replace(' ',''):
        palm()
        if c=='B':
            # Four apposed fingers; the outer two incline toward the central pair.
            for i,d in enumerate([.11,.065,-.04,-.10]): up(i,d)
            foldedthumb(.20)
        elif c in 'AMNSE':
            # Fists are not flat: a thumb under the knuckles lifts the
            # overlying fingers. Some faces of curled fingers are in deep shade.
            if c in 'MN':
                cut=1 if c=='M' else 2
                for i in range(cut):
                    base[i,1]+=.12*(cut-i); curl(i,'low')
                x=base[cut,0]-.23
                chain([(.73,.42),(.79,-.14),(.32,-.64),(x,-.43)],
                      [.35,.31,.26,.21],.045)
                for i in reversed(range(cut,4)):
                    base[i,1]-=.19; curl(i,shift=.10)
            else:
                for i in range(4):
                    if c=='E':base[i,0]*=.87
                    curl(i,'low' if c=='S' else 'normal')
                if c=='A':
                    chain([(.62,.55),(.98,-.02),(1.01,-.59),(.96,-1.21-bend)],
                          [.34,.28,.24,.194],.06)
                elif c=='S':
                    chain([(.61,.42),(.86,-.24),(.52,-.80),(-.10,-1.02)],
                          [.35,.31,.265,.211],.06)
                else:
                    chain([(.65,.51),(.81,-.04),(.42,-.35),(-.34,-.42)],
                          [.33,.28,.25,.20],.045)
        elif c in 'DUVRWK':
            active={'D':(3,), 'U':(2,3),'V':(2,3),'R':(2,3),'W':(1,2,3),'K':(2,3)}[c]
            for i in range(4):
                if i not in active: curl(i)
            if c=='D':
                up(3,.08,1.07)
                chain([(.68,.41),(.88,-.17),(.65,-.69),(.17,-.97)],
                      [.32,.29,.23,.185],.045)
            else:
                dr={'U':(-.0,-.065),'R':(.46,-.35),'V':(-.49,.67),
                    'W':(-.53,.02,.71),'K':(-.55,.31)}[c]
                if c=='R':
                    # Overlap the distal halves. Just one fingertip is often
                    # visible where the fingertips cross, not a symmetric X.
                    short=rng.uniform(.58,.91);sign=1 if variant%4 else -1
                    for i in ((2,3) if sign>0 else (3,2)):
                        if i==2: finger(base[i],(.49+lean,-fl[i]*short),.235,-.06)
                        else: finger(base[i],(.05+lean,-fl[i]*1.06),.23,.08)
                elif c=='K':
                    # One finger is foreshortened pointing toward the camera.
                    up(2,-.93,rng.uniform(.51,.89));up(3,.08,1.07)
                else:
                    for i,d in zip(active,dr): up(i,d)
                if c=='K':
                    chain([(.68,.38),(.82,-.28),(.54,-.87),(.21,-1.43)],
                          [.33,.29,.235,.19],.06)
                else: foldedthumb(.10)
        elif c in 'IYL':
            ext=3 if c=='L' else 0
            for i in range(4):
                if i!=ext: curl(i)
            up(ext,.02 if ext==3 else -.43,1.11, -.05)
            if c=='I':
                chain([(.63,.52),(.94,-.13),(.92,-.70),(.69,-.90)],
                      [.32,.30,.23,.19],.04)
            else:
                rise= .78 if c=='L' else 1.18
                finger((.46,.56),(1.62*splay,-rise),.295,.06,.03)
        elif c=='X':
            # Side view: curled digits lie to the far side of the proximal
            # index; the distal index arches forward over the thumb.
            for i in range(3):
                base[i,0]=(.47-base[i,0]*.54);base[i,1]+=.26
                curl(i,'low')
            chain([(-.56,-.44),(-.61,-1.20),(-.46,-1.72),(-.11,-2.06),
                   (.45,-2.15),(.68,-1.96+bend)],
                  [.37,.33,.29,.255,.215,.17],.055)
            chain([(-.48,.46),(-.32,-.30),(.0,-.97),(.34,-1.18)],
                  [.33,.30,.26,.195],.04)
        elif c=='F':
            for i,d in zip((0,1,2),(-.79,-.26,.47)): up(i,d,1.05,rng.uniform(-.29,.08))
            # The two bent digits close a small circular aperture.
            chain([(.43,-.69),(.86,-1.31),(1.39,-1.67),(1.69,-1.39)],
                  [.26,.245,.215,.168],.02)
            chain([(.62,.50),(1.18,-.02),(1.66,-.75),(1.71,-1.32)],
                  [.34,.285,.255,.186],.045)
    elif c in 'CO':
        # Side elevation. Four flexed fingers overlap, leaving an opening
        # above the thumb. Closing the aperture gives O rather than C.
        wrist(-.51,.80,-1.50,3.8,.62)
        disc(-.45,.08,.68,1.16,.30)
        gap=(.72 if c=='C' else .0)
        for i in range(4):
            f=i/3
            chain([(-.62+f*.12,-.14),(-.40+f*.13,-1.10),(.26+f*.08,-1.74),
                   (.90+f*.15,-1.83+.20*f),(1.28+f*.10,-1.64+f*.30)],
                  np.array([.36,.31,.265,.24,.19])*fat, -.09+f*.085)
        chain([(-.35,.79),(.30,.38),(.98,-.20+gap*.47),(1.36,-1.19+gap*1.33)],
              [.45,.34,.26,.198],.04)
    elif c in 'GHT':
        # Sideways index (index and middle for H). Wrist enters on the left.
        if c=='T': wrist(-.33,.49,-1.52,3.70,.67)
        else: wrist(-.53,.26,-3.75,1.72,.64)
        disc(-.04,.00,1.13,.90,-.47)
        for i in range(3 if c!='H' else 2):
            y=.70-i*.38
            chain([(.24,y+.10),(.72,y),(.97,y-.18),(.61,y-.31)],
                  [.32,.28,.23,.195],-.03+i*.015)
        if c=='G' and variant%3==0:
            finger((.47,.42),(1.78,-.25),.25)
        else:
            chain([(.25,.34),(.69,-.12),(.75,-.49),(.62,-.62)],
                  [.34,.29,.24,.19],.035)
        if c=='H': finger((.48,-.52),(2.36*length,-.13),.232)
        finger((.34,-.85),(2.48*length, -.16+bend), .226)
        if c=='T':
            # In this side view the thumb protrudes between the fingers.
            finger((.24,-.58),(-.10,-.94*length),.25)
    elif c=='P':
        wrist(.62,.43,2.55,3.32,.65)
        disc(.17,.00,1.00,.91,.53)
        finger((-.17,-.62),(-2.28*length,-.58+bend),.236)
        finger((-.33,-.34),(-1.56*length,1.04*splay),.238)
        for i in range(2):
            y=.19+i*.44
            chain([(-.20,y),(-.81,y-.13),(-1.08,y+.11),(-.62,y+.44)],
                  [.30,.27,.235,.21],.02)
        chain([(.38,-.41),(-.11,-.53),(-.70,-.08),(-1.05,.41)],
              [.34,.30,.25,.20],.04)
    elif c=='Q':
        # Back/side of a pronated wrist, flexed knuckles above two downward rays.
        # The forearm behind the thumb is strongly foreshortened or even
        # out of frame when the back of the hand is presented to the camera.
        outboard=1. if variant%3 else -1.
        wrist(outboard*.49,.25,outboard*rng.uniform(.95,2.05),3.9,.57)
        disc(-.16,-.05,.91,1.07,.49)
        finger((-.64,-.12),(-1.01*splay,2.17*length),.26)
        finger((.36,-.20),(.63*splay,1.72*length),.30)
        for i in range(3):
            x=.16-i*.34
            chain([(x+.32,-.63),(x+.07,-.86),(x-.15,-.49),(x-.22,.04)],
                  [.28,.27,.245,.20],.02)
    else: raise ValueError(c)
    # Vary yaw (projected width) and roll, independently of the camera framing.
    yaw=(.88 if c in 'DUVRKW' else 1.) * rng.uniform(.79,1.26)
    if c=='R' and variant%3!=0: yaw*=.89
    if c=='X': yaw*=rng.uniform(.85,1.2)
    ang=rng.normal(0,.15)+({'K':-.16,'P':.16,'G':-.045,'I':.12,'Y':.16,'A':.18,'C':.06,'T':.10,'M':.21,'N':.27,'S':.12,'D':.055,'O':.045,'R':.08}.get(c,0.))
    if c in 'DFRKU' and variant%3!=0:
        # View yaw gives an apparent displacement between the near folded
        # digits and the heel of the palm. Keep the central articulated rays.
        shift=rng.uniform(.12,.32)
        for form,p,r,light,arm in pieces:
            if form=='chain' and not arm and np.max(p[:,1])<.05 and np.min(p[:,1])<-1.60:
                p[:,0]+=shift
        if c=='R': ang+=rng.uniform(.03,.20)
    if c=='T':
        # Metacarpals are vertical while the superposed flexed digit is
        # horizontal, and the thumb peeks out above that flexed digit.
        for form,p,r,light,arm in pieces:
            if form=='chain' and not arm and len(p)==3 and p[-1,0]>1.8:
                p[:,0]=p[0,0]+.74*(p[:,0]-p[0,0])
        pieces[1][1][2:5]=(.86,1.09,.13)
    return pieces,yaw,ang


def raster(c,rng,variant):
    pieces,yaw,ang=model(c,rng,variant)
    ca,sa=math.cos(ang),math.sin(ang)
    ax=[]; ay=[]; bx=[]; by=[]
    def ext(x,y,r):
        X=x*yaw*ca-y*sa; Y=x*yaw*sa+y*ca
        ax.append(np.min(X-r)); bx.append(np.max(X+r));ay.append(np.min(Y-r));by.append(np.max(Y+r))
    for form,p,r,light,arm in pieces:
        if form=='disc':
            x,y,rx,ry,t=p
            u=np.arange(16)*np.pi/8
            ext(x+rx*np.cos(u)*np.cos(t)-ry*np.sin(u)*np.sin(t),
                y+rx*np.cos(u)*np.sin(t)+ry*np.sin(u)*np.cos(t),0.)
        elif not arm:ext(p[:,0],p[:,1],r)
    x0,x1=np.min(ax),np.max(bx); y0,y1=np.min(ay),np.max(by)
    # Include some proximal wrist in the field of view. The extra width on
    # a side-pointing hand is on the opposite side to the extended digit.
    if c in 'GH': x0-=.95;y1+=.42
    elif c=='T':
        x0+=.07; y1+=.7
        # A palm-forward view of the tucked-thumb letter, versus an edge-on
        # pointing pose: rotate the wrist toward the base of the palm.
    elif c=='P': x1+=.52;y1+=.72
    else: y1+=rng.uniform(.32,1.04) if c!='Q' else .04
    zoom=rng.uniform(.82,1.23)*26/max(x1-x0,y1-y0)
    return pieces,(x0+x1)/2,(y0+y1)/2,zoom,ca,sa,yaw


def paint(c,rng,variant):
    pieces,cx,cy,zoom,ca,sa,yaw=raster(c,rng,variant)
    # Render one slightly oversampled extended canvas; all camera translations
    # below are crops of this same constructed object, never photographic data.
    n=40; res=2
    a=(np.arange(n*res,dtype='f')+.5)/res-n/2
    X=cx+a[None,:]/zoom; Y=cy+a[:,None]/zoom
    x=(ca*X+sa*Y)/yaw; y=-sa*X+ca*Y
    bg=.81; img=np.full(x.shape,bg,'f'); mask=np.zeros_like(img)
    key=rng.uniform(.8,1.16); ambient=rng.uniform(.30,.46)
    side=rng.uniform(-.26,-.04) if variant%7 else rng.uniform(.03,.19)
    sy=rng.uniform(-.14,-.04)
    for form,p,r,light,arm in pieces:
        if form=='disc':
            px,py,rx,ry,t=p
            u=((x-px)*np.cos(t)+(y-py)*np.sin(t))/rx
            v=(-(x-px)*np.sin(t)+(y-py)*np.cos(t))/ry
            d=u*u+v*v; nx=u*np.cos(t)-v*np.sin(t); ny=u*np.sin(t)+v*np.cos(t)
        else:
            d=np.full(x.shape,1e3,dtype='f'); nx=np.zeros_like(x);ny=np.zeros_like(y)
            for j,(p0,p1) in enumerate(zip(p,p[1:])):
                dx,dy=p1-p0
                f=np.clip(((x-p0[0])*dx+(y-p0[1])*dy)/(dx*dx+dy*dy+1e-9),0,1)
                R=r[j]+f*(r[j+1]-r[j]); u=(x-p0[0]-f*dx)/R;v=(y-p0[1]-f*dy)/R
                dd=u*u+v*v; better=dd<d
                nx=np.where(better,u,nx);ny=np.where(better,v,ny); d=np.minimum(d,dd)
        nz=np.sqrt(np.maximum(0.,1.-d))
        val=ambient + light + (side*nx+sy*ny+.26*nz)*key
        al=np.clip((1.-d)*6.,0.,1.)
        img+=al*(val-img);mask+=al*(1-mask)
    return (np.stack([img,mask]).reshape(2,n,res,n,res).mean((2,4))).astype('f')


def blur(a,g):
    k=len(g)//2
    def along(a,ax):
        pad=[(0,0)]*a.ndim; pad[ax]=(k,k)
        p=np.pad(a,pad,mode='edge'); s=[slice(None)]*a.ndim
        v=np.zeros_like(a)
        for i,b in enumerate(g):
            s[ax]=slice(i,i+a.shape[ax]); v+=b*p[tuple(s)]
        return v
    return along(along(a,-1),-2)


def translate(a,dx,dy,size,start):
    # Bilinear translation of a constructed receiver field, with a valid crop.
    x=start-dx;y=start-dy; ix=math.floor(x);iy=math.floor(y); fx=x-ix;fy=y-iy
    return sum((b*d)*a[...,iy+j:iy+j+size,ix+i:ix+i+size] for i,b in enumerate((1-fx,fx)) for j,d in enumerate((1-fy,fy)))


def pose_weights(c,rng,k):
    img,mask=paint(c,rng,k)
    g=GA.astype('f'); d=DER.astype('f')
    def sep(a,u,v):
        return sum(q*sum(w*a[j:j+36,i:i+36] for i,w in enumerate(u)) for j,q in enumerate(v))
    gx=sep(img,d,g);gy=sep(img,g,d)
    # Synthetic local gradient distribution. Signed as well as phase-insensitive
    # components allow moderate changes of skin/background contrast.
    e=np.stack([f for t in THETA for f in ((gx*np.cos(t)+gy*np.sin(t)), -(gx*np.cos(t)+gy*np.sin(t))) ]).astype('f')
    e=np.maximum(e,0);e=e+.25*e[np.arange(8)^1]
    # Smooth before subsampling to tolerate about one pixel of edge displacement.
    e=blur(e,np.array([.23,.54,.23],'f'))
    # Pattern inference with segregated boundary/interior evidence: where
    # there is an occluding contour, do not mistake synthetic lighting for
    # a necessary condition. Inside the mask, keep knuckle and tendon edges.
    inside=blur(mask,np.array([.25,.5,.25],'f'))[2:-2,2:-2]
    inner=inside**4
    sx=sep(mask,d,g);sy_=sep(mask,g,d)
    contour=np.stack([f for t in THETA for f in (-(sx*np.cos(t)+sy_*np.sin(t)),sx*np.cos(t)+sy_*np.sin(t))]).astype('f');contour=np.maximum(0.,contour)
    r=np.abs(gy)+np.abs(gx); limits=np.sqrt(gx**2+gy**2+.028**2)
    bounded=np.maximum(0.,np.stack([f for t in THETA for f in (gx*np.cos(t)+gy*np.sin(t),-(gx*np.cos(t)+gy*np.sin(t))) ]))/limits
    interior=(bounded+.52*bounded[np.arange(8)^1])*inner
    # Normalize the three sources separately on geometry alone, before
    # summing. Thus a dim digital finger seam still votes on its orientation.
    def unit(q):return q/(np.sqrt(np.sum(q*q))+1e-6)
    e=.34*unit(e) + .39*unit(blur(interior,np.array([.15,.70,.15],'f'))) + .27*unit(contour+.28*contour[np.arange(8)^1])
    # Soft priors: side/top background usually more reliable than bottom
    # corners that might contain clothes; avoid treating the sleeve as a digit.
    pos=np.arange(14)*2+.5; dep=np.clip((pos-18)/8,0,1)
    wt=(1.-.50*dep[:,None])*(1.-.20*np.clip((abs(pos[None,:]-13.5)-9)/4,0,1))
    out1=[];out2=[];part1=[];part2=[]
    for dy in SHIFTS:
        for dx in SHIFTS:
            p=translate(e,dx,dy,28,4).reshape(8,14,2,14,2).mean((2,4))
            # Subtract spatial DC per direction so sheer amount of texture
            # cannot win; subtract orientation DC in part, to favor aligned edges.
            p=p-.83*(p*wt).sum((1,2),keepdims=True)/wt.sum()
            p=p-.22*p.mean(0,keepdims=True)
            p=p*wt
            p/=max(np.linalg.norm(p),1e-8)
            out1.append(p.ravel())
            m=translate(mask,dx,dy,28,6); im=translate(img,dx,dy,28,6)
            m=m.reshape(14,2,14,2).mean((1,3)); im=im.reshape(14,2,14,2).mean((1,3))
            wt2=(1-.60*np.clip((np.arange(14)*2-18)/9,0,1))[:,None]
            # Emphasize the outline, with an additional contribution from
            # directional lighting within the hand.
            fg=.4*im-.60*m
            fg=fg-(fg*wt2).sum()/(wt2.sum()*14);fg=fg*wt2
            fg/=max(np.linalg.norm(fg),1e-8)
            out2.append(fg.ravel());a,b=part_fields(img,mask,dx,dy,c);part1.append(a);part2.append(b)
    return out1,out2,part1,part2,mask


def part_fields(img, mask, dx,dy,c):
    # Deformable parts: resolve a near tie in whole-hand agreement by giving
    # each anatomical subregion its own short search and its own normalization.
    # The subdivision is in hand coordinates, not absolute image coordinates.
    im=translate(img,dx,dy,28,6); m=translate(mask,dx,dy,28,6)
    m=m.reshape(14,2,14,2).mean((1,3));im=im.reshape(14,2,14,2).mean((1,3))
    # Gradient/light pattern local to a phalanx has both a diffuse side and
    # a crease. Remove slowly varying albedo for the crease-sensitive field.
    field=.75*im-.25*m
    Y,X=np.mgrid[0:14,0:14];x=(X-6.5-dx*.5)/6.5;y=(Y-6.5-dy*.5)/6.5
    if c in 'GHPQT':
        axis=x*((-1 if c=='P' else 1) if c!='Q' else 0)+y*(1 if c=='Q' else 0)
        finger=np.clip(.6+.83*axis,.13,1);palm=np.clip(.8-.85*axis,.13,1)
    else:
        finger=np.clip(.33-1.4*y,.06,1);palm=np.clip(1.-1.3*abs(y-.06),.05,1)
    f=field-blur(field,np.array([1,4,6,4,1],'f')/16)
    # Two quadrature-like comparisons: an unsigned internal highpass and a
    # signed figure/ground field emphasize different parts of the sign.
    env=(.85*m+.15)**.5
    a=(f-.35*(f*palm).sum()/palm.sum())*palm*env
    # Fine creases and hole boundaries, including those around the fingertips.
    b=(f-.35*(f*finger).sum()/finger.sum())*finger
    return [q.ravel()/max(np.linalg.norm(q),1e-8) for q in (a,b)]


def figure_map(low,norm,layer,param,pool,weights):
    # Approximate a flood fill from the boundary in an edge-barrier image.
    # For each pixel, ask for the weakest barrier on an escape path. Use
    # only upward and lateral directions; the wrist can reach the bottom.
    # Paths are nearly straight but may drift a cell to route around noise.
    n=14
    yy,xx=np.mgrid[0:n,0:n]; smooth=np.zeros((n*n,28,28),'f');
    for y in range(n):
        for x in range(n):
            a=np.exp(-.5*((np.arange(28)-2*y-.5)/.88)**2)
            b=np.exp(-.5*((np.arange(28)-2*x-.5)/.88)**2)
            smooth[n*y+x]=np.outer(a/a.sum(),b/b.sum())
    w=smooth.reshape(n,n,-1)
    dirs=[(0,-1),(0,1),(-1,0),(-1,-1),(-1,1)]
    # A path's barrier is the largest directional luminance step; measure
    # also a centered two-step difference for softly blurred contours.
    barriers=[]; silhouettes=[]
    for dy,dx in dirs:
        nb=w[np.clip(yy+dy,0,n-1),np.clip(xx+dx,0,n-1)];back=w[np.clip(yy-dy,0,n-1),np.clip(xx-dx,0,n-1)]
        steps=(nb-w)*(.76 if dx*dy else 1.);wide=(nb-back)*(.54 if dx*dy else .73)
        mat=np.stack([steps,-steps,wide,-wide],axis=2).reshape(n*n*4,-1)
        tag=str(len(barriers)); d=param('barrier_'+tag,'dense',norm,mat)
        d=layer('four_steps_'+tag,'reshape',d,shape=[1,n*n,4]);d=pool('step_max_'+tag,'maxpool',d,[1,4],[1,4]);d=layer('barrier_field_'+tag,'reshape',d,shape=[n*n]);
        # Parallel-prefix max along each ray. First include a short run and
        # then extend it in strides of 2,4,8 to reach the image boundary.
        ids=yy*n+xx
        for j,s in enumerate((1,2,4,8)):
            Y=yy+s*dy;X=xx+s*dx;other=np.clip(Y,0,n-1)*n+np.clip(X,0,n-1)
            valid=(X>=0)&(X<n)&(Y>=0)&(Y<n)
            # For a diagonal, clamp at the FIRST boundary, never turn at it.
            if dx and dy:
                limit=np.minimum(yy,xx if dx<0 else n-1-xx);u=np.minimum(s,limit)
                other=(yy+dy*u)*n+xx+dx*u
            m=np.zeros((n*n,2,n*n),'f'); m[ids,0,ids]=1;m[ids,1,other]=1
            key=tag+'_'+str(j);br=param('shift_'+key,'dense',d,m.reshape(-1,n*n));br=layer('neighbors_'+key,'reshape',br,shape=[1,n*n,2]);br=pool('farthest_'+key,'maxpool',br,[1,2],[1,2]);d=layer('prefix_'+key,'reshape',br,shape=[n*n])
        if len(barriers)<3:
            rim=layer('view_'+tag,'reshape',d,shape=[1,n,n]);rim=layer('view_contrast_'+tag,'norm',rim);silhouettes.append(rim)
        # Gather escape likelihood for each of the n*n cells. Pack different
        # directions into separate lanes using add (there is no concat op).
        m=np.zeros((n*n,len(dirs),n*n),'f');m[ids,len(barriers),ids]=-1
        out=param('escape_'+tag,'dense',d,m.reshape(n*n*len(dirs),n*n)); barriers.append(out)
    b=layer('escape_rays','add',barriers);b=layer('ray_choices','reshape',b,shape=[1,n*n,len(dirs)]);b=pool('easiest_ray','maxpool',b,[1,len(dirs)],[1,len(dirs)]);b=layer('inside','reshape',b,shape=[1,n,n]);return layer('enclosure_contrast','norm',b),silhouettes


def distance_fields(src,layer,param,pool,weights,n):
    # Log-likelihood-like tolerance to small shape errors. In each cell keep
    # the best foreground (and best background) within a short displacement,
    # with a cost proportional to distance. Unlike ordinary correlation this
    # forgives skinny digit registration without forgiving a missing digit.
    yy,xx=np.mgrid[:n,:n];idx=yy*n+xx
    out=[]
    for fg,sgn in [('hand',-1),('background',1)]:
        name=fg+'_neighborhood';steps=[(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)]
        m=np.zeros((n*n,len(steps),n*n),'f');b=np.zeros((n*n,len(steps)),'f')
        for k,(dy,dx) in enumerate(steps):
            m[idx,k,np.clip(yy+dy,0,n-1)*n+np.clip(xx+dx,0,n-1)]=sgn
            b[:,k]=-.85*math.hypot(dx,dy)
        v=param(name,'dense',src,m.reshape(-1,n*n));weights[name+'.b']=b.ravel()
        v=layer(fg+'_alternatives','reshape',v,shape=[1,n*n,len(steps)]);v=pool(fg+'_near','maxpool',v,[1,len(steps)],[1,len(steps)]);v=layer(fg+'_field','reshape',v,shape=[1,n,n]);out.append(layer(fg+'_contrast','norm',v))
    return out


def build(out):
    out.mkdir(exist_ok=True,parents=True)
    weights={};layers=[]
    def layer(name,op,src,**args):
        layers.append(dict(name=name,op=op,inputs=[src] if isinstance(src,str) else src,**args));return name
    def param(name,op,src,w,**args):
        weights[name+'.W']=np.asarray(w,dtype='f');weights[name+'.b']=np.zeros(len(w),'f');return layer(name,op,src,out=len(w),**args)
    def pool(name,op,src,k,s): return layer(name,op,src,size=k,stride=s)
    norm=layer('contrast','norm','image')
    # Replicate the rim instead of throwing away fingertip information on it.
    iy,ix=np.mgrid[-2:30,-2:30]; pad=np.zeros((32*32,784),'f'); pad[np.arange(32*32),(28*iy.clip(0,27)+ix.clip(0,27)).ravel()]=1
    padded=param('extend_rim','dense',norm,pad);padded=layer('extended_frame','reshape',padded,shape=[1,32,32]);
    ed=param('gradient','conv',padded,EDGE[:,None],kernel=[5,5],stride=1,padding=0)
    ed=layer('half_wave','relu',ed)
    ed=pool('oriented_cells','avgpool',ed,[2,2],[2,2]); rawedge=ed
    # Limit sensitivity to the extremely dark sleeve at the bottom: sign of
    # a step is more important than its amplitude after it is clearly visible.
    bend=param('edge_excess','conv',ed,np.eye(8,dtype='f')[:,:,None,None],kernel=[1,1],stride=1,padding=0);weights[bend+'.b'][:]=-.33
    bend=layer('supra_threshold','relu',bend);bend=param('compress_high_contrast','conv',bend,-.81*np.eye(8,dtype='f')[:,:,None,None],kernel=[1,1],stride=1,padding=0);ed=layer('compressed_gradients','add',[ed,bend]); ed=layer('edge_contrast','norm',ed)
    # Mechanism for finding material different from the background: estimate a
    # smooth background lighting ramp using the top and lateral boundaries.
    # Steep departures in either direction can be the hand. The estimate blends
    # an interpolated side boundary with an upper rim prior that survives a sleeve.
    low=pool('lowpass','avgpool',norm,[2,2],[2,2]); n=14
    yy,xx=np.mgrid[0:n,0:n]; light=np.zeros((n*n,n*n),'f')
    for y in range(n):
        for x in range(n):
            def gaussian(u,v,sx,sy):
                a=np.exp(-.5*(((xx-u)/sx)**2+((yy-v)/sy)**2));return a/a.sum()
            t=(x+.5)/n
            l=(1-t)*gaussian(0,y,.78,2.8)+t*gaussian(13,y,.78,2.8)
            u=(1-t)*gaussian(.6,1.0,1.8,1.6)+t*gaussian(12.4,1.0,1.8,1.6)
            light[y*n+x]=(.50*l+.50*u).ravel()
    residual=light-np.eye(n*n,dtype='f')
    # Soft clipping |local contrast| so a dark garment doesn't dominate.
    cut=param('material_tests','dense',low,np.tile(np.concatenate([residual,-residual]),(2,1)))
    weights[cut+'.b'][2*n*n:]=-1.15
    h=layer('positive_tests','relu',cut); eye=np.eye(n*n,dtype='f')
    material=param('other_material','dense',h,np.concatenate([-eye,-.56*eye,.83*eye,.44*eye],axis=1))
    material=layer('material_map','reshape',material,shape=[1,n,n]);material=layer('material_contrast','norm',material)
    enclosure,views=figure_map(low,norm,layer,param,pool,weights)
    material=layer('enclosed_material','add',[material,enclosure]);posmap,negmap=distance_fields(material,layer,param,pool,weights,n)
    # Camera/pose disjunction: for each analytic pose, the two fields must
    # match at the *same* position. Normalize the scores by field length.
    nper=NPOSE*len(SHIFTS)**2; nr=len(LETTERS)*nper
    W1=np.empty((nr,8*14*14),'f');W2=np.empty((nr,14*14),'f');W3=np.empty_like(W2);W4=np.empty_like(W2)
    W5=np.empty_like(W2);W6=np.empty_like(W2);yy,xx=np.mgrid[0:n,0:n]
    vistas=[np.empty_like(W2) for _ in range(3)]
    for j,c in enumerate(LETTERS):
        for k in range(NPOSE):
            rng=np.random.default_rng(280000+j*10000+k)
            g,b,hand,fingers,mask=pose_weights(c,rng,k); i=j*nper+k*len(SHIFTS)**2
            W1[i:i+len(g)]=g; W2[i:i+len(b)]=b;W3[i:i+len(b)]=hand;W4[i:i+len(b)]=fingers
            for j0,dy in enumerate(SHIFTS):
                for j1,dx in enumerate(SHIFTS):
                    ix=i+j0*len(SHIFTS)+j1
                    m=translate(mask,dx,dy,28,6).reshape(n,2,n,2).mean((1,3));v=blur(m,np.array([1,4,6,4,1],'f')/16)
                    # Thin hand strips on either side of actively bending
                    # fingers, and negative space bounded by the hand.
                    fine=m*(1.-v);gap=(1.-m)*v
                    wt=(1.-.68*np.clip((yy-8)/5,0,1));
                    if c not in 'GHPTQ':wt=wt*(1.-.26*np.clip((yy-dy*.5-4)/6,0,1))
                    # Analytic shadow from the hand occluder. At deployment
                    # approximate the shadow using the edge barrier on each ray.
                    # Rays with tangent incidence have weaker barriers.
                    for jv,(step,axis) in enumerate([(-1,1),(1,1),(-1,0)]):
                        # Prefix scan starts at the outside and moves inward.
                        f=np.flip(m,axis=axis) if step==1 else m
                        diff=np.diff(f,axis=axis,prepend=np.take(f,[0],axis=axis));bar=np.maximum.accumulate(np.abs(diff),axis=axis)
                        s=.62*np.maximum.accumulate(f,axis=axis)+.38*bar
                        if step==1:s=np.flip(s,axis=axis)
                        vwt=1.-.62*np.clip((yy-8)/5,0.,1.)
                        s=(s-(s*vwt).sum()/vwt.sum())*vwt
                        vistas[jv][ix]=s.ravel()/max(np.linalg.norm(s),1e-8)
                    for w,q in [(W5,fine), (W6,gap)]:
                        q=(q*q+.012*q)*wt;q-=q.mean();w[ix]=q.ravel()/max(np.linalg.norm(q),1e-8)
    W1*=1/np.sqrt(W1.shape[1]);W2*=.31/np.sqrt(W2.shape[1])
    a=param('contour_agreement','dense',ed,W1);b=param('figure_agreement','dense',material,W2)
    W5*=.24/n;W6*=.24/n;inside=param('finger_strips','dense',posmap,W5);outside=param('interdigital_gaps','dense',negmap,W6)
    profile=[]
    for jv,(view,ww) in enumerate(zip(views,vistas)):
        ww*=.36/n;profile.append(param('skyline_'+str(jv),'dense',view,ww))
    v=layer('agreement','add',[a,b,inside,outside]+profile);v=layer('by_pose','reshape',v,shape=[24,NPOSE,len(SHIFTS)**2]);v=pool('best_alignment','maxpool',v,[1,len(SHIFTS)**2],[1,1]);
    # Local contrast normalization for the independently registered parts.
    base=param('approximation','dense',low,np.stack([-(lambda q:q/q.sum())(np.exp(-.5*(((xx-x)/1.24)**2+((yy-y)/1.24)**2))).ravel() for y in range(n) for x in range(n)]));base=layer('local_dc','reshape',base,shape=[1,n,n]);base=layer('bandpass','add',[base,low]);base=layer('detail_contrast','norm',base)
    part=[]
    for key,ww in [('knuckles',W3),('digits',W4)]:
        ww*=.32/np.sqrt(n*n);br=param(key+'_detail','dense',base,ww);br=layer(key+'_index','reshape',br,shape=[24,NPOSE,len(SHIFTS)**2]);br=pool(key+'_alignment','maxpool',br,[1,len(SHIFTS)**2],[1,1]);part.append(br)
    v=layer('whole_and_parts','add',[v]+part)
    # Combine peak and typical best-in-small-group agreement: less sensitivity
    # to an accidental match with just one oddly oriented model.
    peak=pool('best_pose','maxpool',v,[NPOSE,1],[1,1]);peak=layer('peak','reshape',peak,shape=[24]);
    groups=pool('group_peaks','maxpool',v,[8,1],[8,1]);m=pool('pose_support','avgpool',groups,[NPOSE//8,1],[1,1]);m=layer('support','reshape',m,shape=[24]);layer('class_scores','add',[peak,m])
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n');np.savez(out/'weights.npz',**weights)

if __name__=='__main__':build(Path(sys.argv[1]))
