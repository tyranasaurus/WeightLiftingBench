"""Construct an ASL classifier from anatomical primitives. No data inputs.

Coordinates are in palm half-widths; y increases toward the wrist. Positive x
is the thumb side of the open palm. Shapes describe projections, not pixels.
"""
import sys
import json
from pathlib import Path
import numpy as np

LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'
# Generic hand. Listed from little finger to index finger.
BASE_X = np.array([-.77, -.30, .19, .67])
BASE_Y = np.array([.35, .10, 0., .10])
LENGTH = np.array([1.38, 1.90, 2.15, 1.91])
RADIUS = np.array([.190, .219, .237, .226])
RAD = np.pi/180.
# For every class use the same deterministic sequence of nuisance variables.
NMODELS = 768

def sequence(n, d):
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]
    out = np.empty((n,d)); indices=np.arange(1,n+1)
    shift=np.random.default_rng(7331).uniform(0,1,d)
    for col,p in enumerate(primes[:d]):
        v=np.zeros(n); i=indices.copy(); f=1/p
        while np.any(i):
            v+=f*(i%p); f/=p; i//=p
        out[:,col]=(v+shift[col])%1
    return out

class Pose:
    def __init__(self):
        self.parts=[]; self.extra=[]
        self.wrist=(-.23,2.25,.62,-.25)
        self.rot=0.; self.sx=1.; self.height=1.
    def tube(self,pts,rs,kind='finger',shade=0.):
        p=np.array(pts,dtype=float)
        r=np.broadcast_to(np.asarray(rs,dtype=float),(len(p),)).copy()
        self.parts.append((p,r,kind,shade))
    def palm(self,cx=0.,cy=1.03,rx=1.,ry=1.37,lean=-.08):
        # A palmar slab, tapered at the heel and with softly rounded sides.
        self.palmpar=(cx,cy,rx,ry,lean)
    def padding(self,pts): self.extra.extend(pts)


def make_pose(ch,u):
    P=Pose(); P.palm(); bx=BASE_X.copy(); by=BASE_Y.copy()
    l=LENGTH*(.88+.25*u[7]); r=RADIUS*(.90+.21*u[8]); bend=(u[6]-.5)
    def finger(i,angle=0.,factor=1.,curl=0.):
        # Project a gently curved tapering finger from its MCP joint.
        a=RAD*angle; b=RAD*(angle+curl); c=RAD*(angle+1.5*curl)
        d=lambda t:np.array([np.sin(t),-np.cos(t)])
        q=np.array([bx[i],by[i]]); q1=q+.44*l[i]*factor*d(a)
        q2=q1+.31*l[i]*factor*d(b); q3=q2+.25*l[i]*factor*d(c)
        P.tube([q,q1,q2,q3],r[i]*np.array([1.10,.99,.89,.76]))
    def folded(i,reach=1.,angle=0.,dip=0.):
        # Bent proximal phalanx visible under the rounded knuckle, with
        # the fingertip facing the palm. Obliquity displaces the fingertip.
        x=bx[i]; y=by[i]; a=angle*RAD
        k=np.array([x+.45*np.sin(a),y-.54*reach]);
        end=k+np.array([-.09+.21*bend+.60*np.sin(a+.16),.69*reach+dip])
        P.tube([[x,y+.48],k],r[i]*np.array([1.18,1.05]),shade=-.04)
        P.tube([k,k+.54*(end-k),end],r[i]*np.array([1.02,1.02,.84]),kind='closed',shade=.025)
    def thumb(end,via=None,root=(.42,1.70),extra=None):
        if via is None: via=(.87,.87)
        if extra is None:
            P.tube([root,via,end],[.43,.294,.203],kind='thumb')
        else: P.tube([root,via,extra,end],[.43,.30,.24,.196],kind='thumb')
    def covered(): thumb((-.35,.46),(.85,.94),extra=(.27,.52))
    def oval_bridge(i):
        # Foreshortened, bent finger facing the tip of the thumb.
        x=bx[i]; y=by[i]
        P.tube([(x,y+.44),(x-.08,y-.33),(x+.14,y-.47),(x+.38,y-.17)],r[i]*np.array([1.10,1.02,.95,.78]))

    if ch in 'ABDEIKLMNRSUVWY':
        if ch=='B':
            # Closed together: less space between the distal phalanges.
            for i,a in enumerate([7,4,0,-5]): finger(i,a+4*bend,1.,-1)
            covered()
        elif ch in 'AEMNS':
            # Whether the thumb lies beside, under, or in front of the
            # folded fingers is the major difference among these signs.
            P.palm(-.02,.65,1.05,1.16,-.13); P.wrist=(-.31,1.82,.66,-.36)
            P.sx=.96+.22*u[10]
            if ch=='E': thumb((-.60,.48),(.80,.68),extra=(.18,.48))
            if ch in 'MN':
                gap=0 if ch=='M' else 1
                thumb((float((bx[gap]+bx[gap+1])/2), .48),(.82,.84),extra=(.18,.69))
            for i in range(4):
                # When the thumb is enclosed, the fingers over the thumb
                # project farther forward than the remaining tucked ones.
                reach={'A':.71,'E':1.15,'M':1.25 if i>0 else .65,'N':1.25 if i>1 else .62,'S':.76}[ch]
                by[i]+=(3-i)*(.12 if ch in 'AMN' else .04)
                if ch=='E': by[i]-=.15; bx[i]*=.88
                folded(i,reach,angle=(-18+17*bend if ch in 'EMN' else -6+9*bend),dip=.20 if ch in 'MN' else -.18 if ch=='E' else 0.)
            if ch=='A': thumb((1.12+.20*bend,-.38-.23*u[9]),(.99,.63),root=(.36,1.27),extra=(1.02,.04))
            elif ch=='S': thumb((-.54,-.10+.18*bend),(.75,.65),root=(.34,1.25),extra=(.14,.07))
            P.rot={'A':-5,'M':-8,'N':-13,'S':-4,'E':0}[ch]
            if ch=='N':
                # A partially pronated fist can present the two tucked
                # fingers almost end-on, with the index and middle side by side.
                P.sx*=.78+.23*u[7]; P.rot-=10*u[7]
            if ch=='E': P.rot+=6*u[7]
        elif ch in 'IY':
            for i in (1,2,3): folded(i,1.)
            finger(0,-15-17*u[9] if ch=='Y' else -6-14*u[9],1.06,4*bend)
            if ch=='Y': thumb((2.14+.43*u[10],-.05-.40*u[9]),(1.12,1.07),extra=(1.70,.47))
            else: thumb((.21,.29),(.81,.73))
        elif ch=='L':
            for i in (0,1,2): folded(i)
            finger(3,4+8*bend,1.10)
            thumb((2.23+.40*u[10],.85-.59*u[9]),(1.17,1.17),extra=(1.76,.88-.20*u[9]))
        elif ch=='D':
            finger(3,3+9*bend,1.14)
            for i in (0,1,2): by[i]+=.11*(2-i); oval_bridge(i)
            thumb((.43,-.16),(.91,.67))
            P.sx=.91
        elif ch in 'URVWK':
            if ch!='W': folded(1,1.)
            folded(0,1.)
            spread=.70+.60*u[9]
            if ch=='W':
                finger(1,-19*spread+.8*bend); finger(2,-1+4*bend); finger(3,18*spread)
            elif ch=='V':
                finger(2,-17*spread-3); finger(3,15*spread+2)
            elif ch=='U':
                finger(2,2+6*bend); finger(3,-1+6*bend)
            elif ch=='R':
                # Two diagonals at the crossover. The longer finger projects
                # past the shorter one, then bends back toward the midline.
                finger(2,19+8*bend,1.05,-12-10*u[10]); finger(3,-18+8*bend,.92,5)
            elif ch=='K':
                finger(2,-33-23*u[9],.62+.25*u[10],-4); finger(3,2+4*bend,1.05)
            if ch=='K': thumb((.16,-.90),(.71,.62),extra=(.45,-.32))
            else: covered()
            if ch in 'URK': P.sx=.83+.23*u[10]
            if ch in 'UR':
                # The axis of the pair continues into the radial half of the
                # palm; vary how much of the folded fingers is exposed.
                P.palm(.07,.99,.87+.14*u[7],1.30,-.05); P.wrist=(-.05,2.17,.60,-.08)
    elif ch in 'COF':
        # Three-quarter / profile view. The C and O are bent arcs opposite
        # a second, shorter arc of the thumb, open or closed respectively.
        P.palm(-.30,1.03,.68,1.22,-.16); P.wrist=(-.59,2.12,.57,-.30)
        if ch=='O':
            # Oppose the thumb pad to all fingertips. At this viewing angle
            # the thenar eminence fills the base of the circular opening.
            P.palm(-.31,.72,.74,.96,-.20); P.wrist=(-.56,1.78,.60,-.38)
        sep = 1. if ch=='C' else 0.
        tx=1.12+.25*bend; ty=-.60+.18*bend
        if ch=='F':
            # Three extended fingers fan from the ulnar side; the index
            # curls round to contact the thumb (the 'OK' handshape).
            for i,a in enumerate([-21,1,23]):
                bx[i]=[-.84,-.52,-.20][i]; by[i]=[.22,-.01,-.07][i]
                finger(i,a+10*bend,1.,8+7*u[9])
            P.tube([(-.13,.42),(.14,-.16),(.60,-.45),(tx,-.30)],[.28,.25,.217,.19]); ty=-.29
        else:
            for i in range(4):
                # Project the four touching fingers at slightly different
                # depths: their tips can largely eclipse each other.
                off=(3-i); dx=-off*(.068+.064*u[9]); dy=off*(.10+.055*u[10]);
                top=-.86-.28*u[11]-.11*sep
                P.tube([(-.42+dx,.68+dy),(-.29+dx,-.06+dy),(.02+dx,top+.19+dy),(.55+dx,top+dy),(tx+dx,ty-.32*sep+dy)],r[i]*np.array([1.40,1.32,1.15,1.01,.84]),shade=-.027*off)
        ey=ty+sep*(1.00+.49*u[9]); ex=tx+sep*(.08+.25*u[10]);
        thumb((ex,ey),(.37,1.16+.20*sep),root=(-.35,1.64),extra=(.79,.45+.55*sep))
        if ch=='O':
            # Shorten the round aperture vertically; in three-quarter view
            # see several knuckles along the upper arc, not a long C.
            for pts,rs,kind,_ in P.parts:
                if kind=='thumb':
                    pts[0]=[-.26,1.18]; pts[1:,1]=ty+(.68+.18*u[6])*(pts[1:,1]-ty); rs[:2]*=1.13
                else:
                    pts[:,0]+=.11*(pts[:,1]+.4); pts[:,1]=-.45+.78*(pts[:,1]+.45)
            P.sx=1.04+.23*u[8]
        P.rot=9. if ch=='C' else 6.
    elif ch in 'GHT':
        # Lateral pointing hand, seen from its dorsum or side. G extends
        # the index, H both index and middle; for T this pictured gesture
        # has an upright thumb above a side-pointing index.
        P.palm(-.47,.48,1.03,.88,.05); P.wrist=(-1.35, .78,.59,-1.0)
        # Wrist direction stored explicitly for the lateral projections.
        slope=.62+1.15*u[10] if ch=='T' else .10+1.25*u[10]
        P.forearm=(-1.,slope); P.wrist=(-1.09,.90+.32*u[10],.57,-1.)
        for i in (0,1,2):
            if i==2 and ch=='H': continue
            y=.94-.36*i; x=.22+.16*i
            P.tube([(-.53,y-.04),(x,y-.15),(x+.09,y+.10),(x-.32,y+.28)],[.25,.25,.21,.17])
        # The extended digits and metacarpals are nearly collinear. The
        # curled fingers form a mass below, rather than a row of visible tips
        # of equal contrast, especially from the back of the hand.
        P.palm(-.43,.40,1.11,.74,.18)
        base=np.array([-.52,-.19]); angle=(-3 if ch=='G' else 1)+11*bend
        def lateral(y,le,ang,rr):
            a=ang*RAD; d=np.array([np.cos(a),np.sin(a)]); q=base+[0,y]
            P.tube([q,q+.89*d,q+1.62*d,q+le*d],[rr*1.25,rr,rr*.89,rr*.76])
        lateral(0.,2.76+.38*u[9],angle,.234)
        if ch=='H':
            # Together, not a horizontal V: the opposing edges largely
            # coincide in projection. Depth offset gives one dark seam.
            lateral(.27+.13*u[10],2.87+.28*u[9],angle+1.6*bend,.265)
        if ch=='G':
            lateral(.17,1.20+.65*u[9],angle-3-8*u[10],.21)
        if ch=='T':
            P.tube([(-.60,.43),(-.35,-.10),(-.50,-.66-.26*u[9])],[.39,.26,.188]); P.rot=7.
            # Side-on view of the upturned fist. The arm extends roughly
            # along the thumb line, not along the projecting index finger.
            P.palm(-.36,.48,.80,.98,.12); P.wrist=(-.66,1.40,.61,-.20); P.forearm=(-.28-.65*u[10],1.)
        elif ch=='G': P.rot=-6.
    elif ch in 'PQ':
        # Wrist enters from below/right. Knuckles turn over in flexion;
        # distal fingers project down for Q, and out and down for P.
        P.palm(.42,-.20,.94,.78,.16); P.wrist=(1.09,.49,.56,.64); P.forearm=(.58,1.)
        for i in (0,1):
            x=.53+.25*i
            P.tube([(x,-.50),(x+.15,.07),(x-.18,.53),(x-.34,.38)],[.26,.23,.20,.17])
        if ch=='P':
            P.tube([(.35,-.26),(-.44,-.57),(-1.33,-.83),(-2.09,-1.02-.41*u[9])],[.33,.26,.227,.17]);
            # The middle finger can point almost straight at the camera;
            # its visible length then foreshortens towards the thumb tip.
            reach=.42+.68*u[5]
            P.tube([(.24,-.30),(-.37,.06),(-.42-.49*reach,.12+.56*reach),(-.47-.75*reach,.16+1.33*reach)],[.31,.26,.22,.17]);
            P.tube([(.86,.17),(.26,.29),(-.42,.40),(-.81,.72)],[.38,.31,.23,.18]); P.rot=7.
        else:
            P.tube([(.35,-.56),(-.33,-.48),(-.77,.17),(-1.31,1.17+.31*u[9])],[.32,.27,.234,.18]);
            P.tube([(.82,-.08),(.37,.36),(.12,1.03),(-.13,1.67)],[.38,.30,.235,.19]); P.rot=-4.
    elif ch=='X':
        P.palm(-.02,1.01,.88,1.33,-.17); P.wrist=(-.19,2.56,.60,-.19)
        for i in (0,1,2): folded(i,.88)
        # Hook: upright proximal index, bent middle phalanx, bent tip.
        hook=1.10+.48*u[9]
        P.tube([(-.03,.32),(.02,-.53),(.19,-hook),(.71,-hook-.10),(1.03,-hook+.26+.22*bend)],[.29,.245,.229,.20,.158])
        # Viewing the hooked index from the radial side: folded fingers
        # overlap, and the oblique thumb crosses them below the bend.
        thumb((.36,-.16),(.78,.64),root=(-.04,1.32),extra=(.62,.20)); P.sx=.83+.14*u[7]; P.rot=12.
    else: raise ValueError(ch)
    # Short broad hands and long narrow hands; fingers also change projected
    # length when the palm tilts. Scale and translation below are camera DOFs.
    P.sx*=.88+.29*u[11]
    # Approximate perspective under axial rotation. The fingertips are at
    # different depths: flexion projects them in *front* of the MCP line.
    # Simply squeezing a frontal silhouette cannot describe this parallax.
    if ch in 'ABDEIKLMNRSUVWYX':
        yaw=(u[5]-.5)*.76
        if ch in 'MN': yaw+=.35
        if ch=='X': yaw+=.37
        P.sx*=np.cos(yaw)
        for pts,rs,kind,_ in P.parts:
            if kind=='thumb': depth=np.linspace(.15,.72,len(rs))
            elif kind=='closed': depth=np.linspace(.57,.96,len(rs))
            elif len(rs)==2: depth=np.array([.04,.57])
            else: depth=np.linspace(.06,.23,len(rs))
            pts[:,0]+=np.tan(yaw)*depth
    P.letter=ch
    return P

# Primary geometric rendering uses tapered capsules and an elliptical slab.
# The two outputs are synthetic reflectance and the union of the primitives.
def render(p,u,size=34):
    cx,cy,rx,ry,lean=p.palmpar
    wx,wy,wr,wlean=p.wrist
    q=[(cx-rx,cy),(cx+rx,cy),(cx,cy-ry), (wx,wy)]
    for pts,rs,_,_ in p.parts:
        q.extend(pts+[[-1,0]]*rs[:,None]); q.extend(pts+[[1,0]]*rs[:,None]); q.extend(pts+[[0,-1]]*rs[:,None]); q.extend(pts+[[0,1]]*rs[:,None])
    q=np.array(q); q[:,0]*=p.sx
    theta=(p.rot+(u[0]-.5)*31)*RAD; co=np.cos(theta); si=np.sin(theta)
    rot=np.array([[co,si],[-si,co]])
    q=q@rot
    lo=q.min(axis=0); hi=q.max(axis=0)
    # Frame the hand, leaving room for some of the forearm and for motion.
    center=(hi+lo)/2; center[1]+=.08
    scale=max((hi[0]-lo[0])/(.91+.38*(u[1]-.5)),(hi[1]-lo[1])/(.90+.47*(u[2]-.5)))
    if p.letter in 'GHT':
        center[1]+=(.035+.09*u[18])*scale; scale*=.80+.26*u[19]
        center[0]+=(u[5]-.5)*.21*scale
    if p.letter in 'PQ': scale*=.65+.42*u[19]
    center += scale*np.array([.27*(u[3]-.5),.28*(u[4]-.5)])
    ys,xs=np.mgrid[:size,:size]; X=((xs+.5)/size-.5)*scale+center[0]; Y=((ys+.5)/size-.5)*scale+center[1]
    x=(X*co+Y*si)/p.sx; y=-X*si+Y*co
    aa=size/scale
    # Flat background; the matching representation intentionally does not
    # require the rendered background to have the true photographic intensity.
    im=np.full(x.shape,.69+.13*(u[15]-.5)); mask=np.zeros_like(im)
    lx=-(.54+.36*u[12]); ly=-(.13+.38*u[13]); lz=.36
    if u[14]<.14: lx=-lx
    base=.47+.24*(u[15]-.5); contrast=.55+.21*u[16]
    def layer(sd,nx,ny,nz,tint=0.,crease=0.):
        a=np.clip(.5+sd*aa,0,1)
        light=lx*nx+ly*ny+lz*nz
        val=base+contrast*(light-.25)+tint+crease
        im[:]=im*(1-a)+a*np.clip(val,.04,.98); mask[:]=np.maximum(mask,a)
    dx,dy=getattr(p,'forearm',(wlean,1.)); v=np.array([dx,dy]); v/=np.linalg.norm(v)
    fx=x-wx; fy=y-wy; axial=fx*v[0]+fy*v[1]; side=fx*v[1]-fy*v[0]; width=wr+.066*np.clip(axial,0,4)
    f=side/width; fn=np.sqrt(np.maximum(0,1-f*f)); sd=np.minimum(width-np.abs(side),axial+.57)
    layer(sd, f*v[1],-f*v[0],fn,-.035)
    # Optional sleeve beyond the wrist, not used for determining the frame.
    if u[17] < .35:
        a=np.clip(.5+np.minimum(width+.10-np.abs(side),axial-.45-.9*u[18])*aa,0,1); im[:]=im*(1-a)+a*(.17+.12*fn); mask[:]=np.maximum(mask,a)
    v=(y-cy)/ry; nx=(x-cx-lean*(y-cy))/(rx*np.clip(1-.11*v,.72,1.27)); ny=v; nz=np.sqrt(np.maximum(0,1-nx**2-ny**2));
    # The open central palm is usually less illuminated than the rounded
    # thenar pad; light can still spill onto both. The back is more even.
    palmshadow=-.17*np.exp(-.5*(((x-cx+.18)/.62)**2+((y-cy+.31)/.72)**2)) if p.letter not in 'GHPQ' else 0.
    layer((1-np.sqrt(nx**2+ny**2))*min(rx,ry),nx,ny*.74,nz,crease=palmshadow)
    for pts,rs,kind,tint in p.parts:
        # Smooth union within a digit; depth order between digits gives the
        # small occlusion boundaries inside a closed hand.
        dist=np.full(x.shape,1e6); bestx=np.zeros_like(x); besty=bestx.copy(); bestr=bestx.copy(); bestt=bestx.copy()
        for j in range(len(rs)-1):
            ax,ay=pts[j]; vx,vy=pts[j+1]-pts[j]; d2=vx*vx+vy*vy
            t=np.clip(((x-ax)*vx+(y-ay)*vy)/max(d2,1e-7),0,1)
            dx=x-ax-t*vx; dy=y-ay-t*vy; r=rs[j]*(1-t)+rs[j+1]*t
            dd=(dx*dx+dy*dy)/(r*r); use=dd<dist
            dist=np.minimum(dd,dist); bestx=np.where(use,dx/r,bestx); besty=np.where(use,dy/r,besty); bestr=np.where(use,r,bestr); bestt=np.where(use,t+j,bestt)
        nz=np.sqrt(np.maximum(0,1-dist)); sd=(1-np.sqrt(dist))*bestr
        # Slight creases at the joints, perpendicular to the phalanx.
        crease=0.
        if len(rs)>2:
            crease=-.055*np.exp(-((bestt-1.02)/.095)**2)
        if kind=='closed':
            # Nails are small diffuse patches, while the occlusion between
            # curled fingertips and the palm itself makes a dark rim.
            crease+=.11*np.exp(-4*((bestt-1.48)**2+(.8*bestx)**2))-.10*np.clip(bestt-1.53,0,.5)*2
        if kind=='thumb':
            # The thenar pad emerges smoothly from the palm rather than as
            # a separate cylinder with an outlined base.
            crease+=.045*np.clip(1-bestt,0,1)
        layer(sd,bestx,besty,nz,tint+.025,crease)
    # Diffuse shadow across the volume of the hand; finger creases on the
    # far side receive less light than surfaces facing the source.
    im-=mask*(.08+.16*u[16])*np.clip(.50+.43*(x-cx)*(-np.sign(lx)),0,1)
    return np.clip(im,0,1),mask

# Build low-contrast-tolerant oriented energy coefficients. Valid convolutions
# avoid mistaking the frame border for part of the hand.
def kernels():
    y,x=np.mgrid[-2:3,-2:3]; fs=[]
    for i in range(4):
        a=i*np.pi/4; n=x*np.cos(a)+y*np.sin(a); t=y*np.cos(a)-x*np.sin(a)
        f=n*np.exp(-.5*((n/.84)**2+(t/1.07)**2)); f-=f.mean(); f/=np.abs(f).sum()/2; fs.append(f)
    return np.array(fs,dtype=np.float32)

def ridge_kernels():
    # Odd filters respond to transitions; the even second derivative instead
    # detects a narrow phalanx or a dark groove between two adjacent fingers.
    y,x=np.mgrid[-2:3,-2:3]; fs=[]
    for i in range(4):
        a=i*np.pi/4; n=x*np.cos(a)+y*np.sin(a); t=y*np.cos(a)-x*np.sin(a)
        f=(1-(n/.87)**2)*np.exp(-.5*((n/.87)**2+(t/1.5)**2)); f-=f.mean(); f/=np.abs(f).sum()/2; fs.append(f)
    return np.array(fs,dtype='f')

def resize(a,n):
    h=a.shape[0]; t=(np.arange(n)+.5)*h/n-.5; i=np.clip(np.floor(t).astype(int),0,h-2); f=np.clip(t-i,0,1)
    b=(1-f)[None,:]*a[:,i]+f[None,:]*a[:,i+1]; return (1-f)[:,None]*b[i]+f[:,None]*b[i+1]

def energies(a,fs):
    # Only called for procedurally constructed geometric objects in this file.
    v=np.lib.stride_tricks.sliding_window_view(a,(5,5)); g=v.reshape(24*24,25)@fs.reshape(4,25).T
    return np.abs(g.reshape(24,24,4).transpose(2,0,1))

def reduced(e):
    # 4x4 patches on a stride of 2; smoothing before the scan allows some
    # deformation. This is also the spacing of the translation search.
    return np.lib.stride_tricks.sliding_window_view(e,(4,4),axis=(1,2))[:,::2,::2].mean(axis=(-1,-2))

def foreground_filter():
    # Approximate the smooth horizontal lighting gradient of the background
    # using two broad regions at the upper corners. Remove only part of the
    # slope, since those regions themselves may include fingers.
    w=np.zeros((11,11,28,28)); ramp=np.zeros((28,28)); ramp[:8,:6]=-1/48; ramp[:8,-6:]=1/48
    for y in range(11):
        for x in range(11):
            w[y,x]=(x-5)*2/22*.72*ramp; w[y,x,2+2*y:6+2*y,2+2*x:6+2*x]-=1/16
    return w.reshape(121,784).astype('f')

def selective(a): return np.maximum(a-.61*np.roll(a,2,axis=0),0.)

def unit(a): return a/max(.001,np.sqrt(np.sum(a*a)))

def template(p,u,fs,rs,fg):
    im,mask=render(p,u); im=resize(im,28); mask=resize(mask,28)
    a=selective(energies(im,fs)); b=selective(energies(mask,fs))
    e=np.minimum(a/.22,1.)*.74+np.minimum(b/.62,1.)*.26
    e=reduced(e)
    e=e-.68*e.mean()-.32*e.mean(axis=(1,2),keepdims=True)
    # Use two sources of evidence for occupancy: modeled shading and the
    # silhouette. Both get the same corner correction as the image branch.
    occ=(fg @ (.81*im-.19*mask).ravel()).reshape(11,11)
    occ=unit(occ-occ.mean()); e=unit(e)
    ra=selective(energies(im,rs)); rb=selective(energies(mask,rs))
    ridge=reduced(.78*np.minimum(ra/.15,1.)+.22*np.minimum(rb/.40,1.)); ridge=unit(ridge-ridge.mean())
    return np.concatenate([.48*e, .28*ridge, 2*.24*occ[None]])

class Network:
    def __init__(self): self.layers=[]; self.weights={}
    def add(self,name,op,source,**kw):
        self.layers.append(dict(name=name,op=op,inputs=source if isinstance(source,list) else [source],**kw)); return name
    def linear(self,name,op,source,w,b=None,**kw):
        w=np.asarray(w,dtype='f'); self.weights[name+'.W']=w
        self.weights[name+'.b']=np.zeros(len(w),'f') if b is None else np.asarray(b,dtype='f'); return self.add(name,op,source,out=len(w),**kw)

def build(dest):
    net=Network(); fs=kernels(); L=net.add; lin=net.linear
    src=L('adjust_exposure','norm','image')
    g=lin('oriented_derivatives','conv',src,np.concatenate([fs,-fs])[:,None],kernel=[5,5],padding=0,stride=1)
    g=L('rectify','relu',g); w=np.tile(np.eye(4),(1,2))[:,:,None,None]; g=lin('unsigned','conv',g,w,kernel=[1,1]);
    w=(np.eye(4)-.61*np.roll(np.eye(4),2,axis=1))[:,:,None,None]
    g=lin('orientation_competition','conv',g,w,kernel=[1,1]); g=L('select_orientation','relu',g)
    # Saturating piecewise-linear contrast response: strong boundaries do not
    # overwhelm weaker internal ridges or a fingertip against a pale wall.
    cap=.86; h=lin('contrast_threshold','conv',g,np.eye(4)[:,:,None,None],np.full(4,-cap),kernel=[1,1]); h=L('excess','relu',h)
    h=lin('discount_strong_edges','conv',h,(-.88*np.eye(4))[:,:,None,None],kernel=[1,1]); g=L('compressed','add',[g,h])
    g=L('local_histogram','avgpool',g,size=[4,4],stride=[2,2])
    # Separate per-channel mean removal is inappropriate (a V has more
    # diagonals than a U). Normalise across orientations and space together.
    g=L('one_histogram','reshape',g,shape=[1,4*11,11]);g=L('unit_contrast','norm',g);g=L('field','reshape',g,shape=[4,11,11])
    # Add low-frequency shading and curvature before scanning the field.
    fg=foreground_filter(); v=lin('detrend_lighting','dense',src,fg);v=L('silhouette_image','reshape',v,shape=[1,11,11]);v=L('tone_contrast','norm',v)
    rs=ridge_kernels(); r=lin('ridge_curvature','conv',src,np.concatenate([rs,-rs])[:,None],kernel=[5,5]);r=L('ridge_polarities','relu',r)
    r=lin('ridge_strength','conv',r,np.tile(np.eye(4),(1,2))[:,:,None,None],kernel=[1,1]);r=lin('ridge_competition','conv',r,w,kernel=[1,1]);r=L('ridge_positive','relu',r)
    h=lin('ridge_knee','conv',r,np.eye(4)[:,:,None,None],np.full(4,-.68),kernel=[1,1]);h=L('ridge_excess','relu',h);h=lin('ridge_discount','conv',h,(-.90*np.eye(4))[:,:,None,None],kernel=[1,1]);r=L('ridge_compression','add',[r,h]);r=L('ridge_histogram','avgpool',r,size=[4,4],stride=[2,2]);r=L('ridge_vector','reshape',r,shape=[1,44,11]);r=L('ridge_unit','norm',r);r=L('ridge_field','reshape',r,shape=[4,11,11])
    I=np.eye(9);g=lin('histogram_channels','conv',g,I[:,:4,None,None],kernel=[1,1]);r=lin('ridge_channels','conv',r,I[:,4:8,None,None],kernel=[1,1]);v=lin('tone_channel','conv',v,I[:,8:,None,None],kernel=[1,1]);g=L('combined_fields','add',[g,r,v])
    # Each 11x11 filter scans a 3x3 translation grid via one cell of
    # padding. This is the same translation search without a dense gather.
    seq=sequence(NMODELS,20); w=np.empty((len(LETTERS),NMODELS,9,11,11),'f'); fg=fg.astype(float)
    for k,ch in enumerate(LETTERS):
        for j,u in enumerate(seq): w[k,j]=template(make_pose(ch,u),u,fs,rs,fg)
    def consensus(z,s):
        z=L('position_max'+s,'maxpool',z,size=[3,3],stride=[3,3]);z=L('class_pose'+s,'reshape',z,shape=[24,NMODELS,1]);z=L('pose_max'+s,'maxpool',z,size=[NMODELS//4,1],stride=[NMODELS//4,1]);return L('agreement'+s,'avgpool',z,size=[4,1],stride=[4,1])
    global_vote=consensus(lin('geometric_matches','conv',g,w.reshape(-1,9,11,11),kernel=[11,11],padding=1),'')
    # A second vote compares directional structure within small regions.
    # Subtracting the local mean of ALL orientation bins preserves the
    # distinction between e.g. a vertical finger and a horizontal one.
    # It discounts broad differences in contrast between patches of skin,
    # fabric and wall which the synthetic lighting cannot predict.
    def region_normalize(a):
        ans=np.zeros_like(a)
        for iy in range(3):
            for ix in range(3):
                y,x=4*iy,4*ix; t=a[:,np.minimum(y+np.arange(4),10)][:,:,np.minimum(x+np.arange(4),10)]; t=t-t.mean(); k=min(4,11-y); m=min(4,11-x); ans[:,y:y+4,x:x+4]=t[:,:k,:m]/np.sqrt((t*t).mean()+.24**2)
        return ans
    loc=np.zeros((24,NMODELS,8,11,11),dtype='f')
    for k in range(24):
        for j in range(NMODELS):
            # Input coefficients here have global L2 norm of order one;
            # restore RMS-one units before using a local contrast floor.
            a=w[k,j,:4]*44; b=w[k,j,4:8]*78
            def detail(a):
                # A surface with nothing on it cannot predict a local
                # orientation: do not amplify its low-energy neighborhoods.
                z=region_normalize(a); support=np.minimum(1,np.abs(a-a.min()).sum(axis=0)/1.5)
                return unit(z*(.18+.82*support))
            loc[k,j,:4]=.64*detail(a); loc[k,j,4:]=.36*detail(b)
    # Implement the same local centering on the observed edge and ridge
    # fields. Adjacent windows tile rather than wrap at the image borders.
    def local_fields(node,s):
        ids=np.arange(484).reshape(4,11,11); groups=[]
        for y in (0,4,8):
            for x in (0,4,8):
                groups.append(ids[:,np.minimum(y+np.arange(4),10)][:,:,np.minimum(x+np.arange(4),10)].ravel())
        order=np.tile(groups,(1,2)).ravel(); floor=np.tile(np.repeat([.24,-.24],64),9)
        mat=np.eye(484,dtype='f')[order]; back=(mat/np.sum(mat,axis=0)).T
        z=lin('take_'+s,'dense',node,mat,floor); z=L('patch_'+s,'reshape',z,shape=[9,2,64]);z=L('contrast_'+s,'norm',z);z=lin('replace_'+s,'dense',z,back);return L('local_'+s,'reshape',z,shape=[4,11,11])
    a=local_fields('field','edges');b=local_fields('ridge_field','ridges')
    a=lin('local_edge_channels','conv',a,np.eye(8)[:,:4,None,None],kernel=[1,1]);b=lin('local_ridge_channels','conv',b,np.eye(8)[:,4:,None,None],kernel=[1,1]);z=L('local_features','add',[a,b])
    local_vote=consensus(lin('part_structure','conv',z,loc.reshape(-1,8,11,11),kernel=[11,11],padding=1),'_parts')
    # The region vote is supporting evidence, not a substitute for matching
    # the silhouette and layout of the complete hand.
    local_vote=lin('detail_weight','conv',local_vote,(.60*np.eye(24))[:,:,None,None],kernel=[1,1]);out=L('all_evidence','add',[global_vote,local_vote]);L('letters','reshape',out,shape=[24])
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True); (dest/'network.json').write_text(json.dumps({'layers':net.layers},indent=2)); np.savez(dest/'weights.npz',**net.weights)

if __name__=='__main__': build(sys.argv[1])
