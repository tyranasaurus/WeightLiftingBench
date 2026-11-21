"""Data-free build. All coordinates below are in anatomical units, with y up,
+z out of the palm, and +x on the thumb side. Nothing is loaded from disk.
"""
import math
import sys
from pathlib import Path
import numpy as np

runtime = '''import math
import numpy as np

def blurone(a):
    b = np.pad(a, ((0,0),(1,1),(1,1)), mode='edge')
    c = b[:,:-2] + 2*b[:,1:-1] + b[:,2:]
    return (c[:,:,:-2] + 2*c[:,:,1:-1] + c[:,:,2:])*.0625

def box(a,r):
    p = np.pad(a, ((0,0),(r,r),(r,r)), mode='edge')
    p = np.pad(p, ((0,0),(1,0),(1,0)))
    p = p.cumsum(1).cumsum(2)
    k = 2*r+1
    return (p[:,k:,k:] - p[:,:-k,k:] - p[:,k:,:-k] + p[:,:-k,:-k])/(k*k)

def cells(a,k):
    return a.reshape(len(a), 28//k,k,28//k,k).mean((2,4))

def smoothcells(a):
    b = np.pad(a, ((0,0),(1,1),(1,1)),mode='edge')
    return (.8*b[:,1:-1,1:-1] + .05*(b[:,:-2,1:-1]+b[:,2:,1:-1]+b[:,1:-1,:-2]+b[:,1:-1,2:]))

def unitvec(a):
    a = a.reshape(len(a),-1)
    return a / np.sqrt((a*a).sum(1,keepdims=True)+1e-12)

def features(a, mask):
    # Local gradient strength, unoriented contour directions and signed
    # curvature. No pixel depends on other images in the batch.
    b = blurone(a)
    p = np.pad(b, ((0,0),(1,1),(1,1)), mode='edge')
    dx = p[:,1:-1,2:] - p[:,1:-1,:-2]
    dy = p[:,2:,1:-1] - p[:,:-2,1:-1]
    mag = np.sqrt(dx*dx+dy*dy+1e-12)
    local = np.maximum(box(mag,3),0)
    w = 1/np.sqrt(.045+local+.18*mag)
    edge = np.sqrt(np.maximum(mag*w,0))*np.minimum(mag/.028,1)
    inv = 1/mag
    gx,gy = dx*inv,dy*inv
    coarse = []
    fine = []
    for j in range(6):
        z = np.maximum(np.abs(math.cos(j*math.pi/6)*gx+math.sin(j*math.pi/6)*gy)-.68,0)*(1/.32)
        e = edge*z*z
        coarse.append(smoothcells(cells(e,4)))
        fine.append(smoothcells(cells(e,2)))
    # Per-cell normalization with a stabilizer retains the distinction
    # between smooth background and hand texture.
    h = np.stack(coarse,axis=1)
    h = h / np.sqrt(.027+(h*h).sum(1,keepdims=True))**.65
    f = np.stack(fine,axis=1)
    f = f / np.sqrt(.035+(f*f).sum(1,keepdims=True))**.62
    # Greyscale ridges and grooves, locally normalized; insensitive to a
    # linear illumination ramp. Keep both signs in their original location.
    d = b-box(b,3)
    ridge = np.clip(d/np.sqrt(.0016+np.maximum(box(d*d,3),0)), -2,2)
    s = unitvec(smoothcells(cells(ridge,2)))
    h = unitvec(h)
    f = unitvec(f)
    m = unitvec(cells(mask,2)-cells(mask,2).mean((1,2),keepdims=True))
    return h,f,s,m

def passage(a,v):
    # One scan in the minimum-barrier metric. A path may follow an oblique
    # edge: a background route does not have to be an axis-aligned ray.
    lo=a[:,0].copy()
    hi=lo.copy()
    for i in range(1,28):
        row=a[:,i]
        ls=np.stack((lo,np.concatenate((lo[:,:1],lo[:,:-1]),1),np.concatenate((lo[:,1:],lo[:,-1:]),1)),1)
        hs=np.stack((hi,np.concatenate((hi[:,:1],hi[:,:-1]),1),np.concatenate((hi[:,1:],hi[:,-1:]),1)),1)
        ls=np.minimum(ls,row[:,None]); hs=np.maximum(hs,row[:,None]); d=hs-ls
        sel=np.argmin(d,1)[:,None]
        lo=np.take_along_axis(ls,sel,1)[:,0]; hi=np.take_along_axis(hs,sel,1)[:,0]
        v[:,i]=np.minimum(v[:,i],hi-lo)

def foreground(a):
    b=blurone(a)
    v=np.ones_like(a)
    passage(b,v)
    passage(b.transpose(0,2,1),v.transpose(0,2,1))
    passage(b.transpose(0,2,1)[:,::-1],v.transpose(0,2,1)[:,::-1])
    v[:,0]*=.35
    # Texture is useful even at a border crossed by the wrist or fingertips.
    c=np.abs(b-box(b,2))
    e=np.maximum(box(c,1),0)
    return blurone(1-np.exp(-12*(v+.9*e)))

def contourfield(a,fg):
    # Steerable edge evidence with one pixel of registration tolerance. An
    # isolated boundary between two background objects is not very diagnostic.
    b=.7*a+.3*blurone(a)
    p=np.pad(b,((0,0),(1,1),(1,1)),mode='edge')
    gx=p[:,1:-1,2:]-p[:,1:-1,:-2]; gy=p[:,2:,1:-1]-p[:,:-2,1:-1]
    mag=np.sqrt(gx*gx+gy*gy+1e-12)
    lm=np.maximum(box(mag,3),0)
    contrast=mag/(.037+.55*lm+mag)
    contrast*=.75+.25*np.minimum(1,2*blurone(fg))
    fields=[]
    for j in range(6):
        rate=np.abs((math.cos(j*math.pi/6)*gx+math.sin(j*math.pi/6)*gy)/mag)
        z=contrast*np.maximum(0,rate*rate*2-1)**.8
        p=np.pad(z,((0,0),(1,1),(1,1)),mode='edge')
        near=np.maximum(np.maximum(p[:,1:-1,:-2],p[:,1:-1,2:]),np.maximum(p[:,:-2,1:-1],p[:,2:,1:-1]))
        diag=np.maximum(np.maximum(p[:,:-2,:-2],p[:,2:,:-2]),np.maximum(p[:,:-2,2:],p[:,2:,2:]))
        fields.append(np.maximum(z,np.maximum(.85*near,.75*diag)))
    return np.stack(fields,axis=1).reshape(len(a),-1)
'''

infer = '''
def predict(x):
    answers = np.empty(len(x),np.int64)
    # Fixed-size matrix multiplies, even for a last partial block. Integers
    # are returned, not floating scores with a batch-dependent roundoff.
    stride = 64
    nclass = 24
    for start in range(0,len(x),stride):
        n = min(stride,len(x)-start)
        a = np.zeros((stride,28,28),np.float32)
        a[:n] = x[start:start+n,0]
        fg=foreground(a)
        h,f,s,m = features(a,fg)
        field=contourfield(a,fg)
        q=np.concatenate((h*.70,m*.30),1)
        coarse=(q @ searchbank.T).reshape(stride,nclass,-1)
        for i in range(n):
            # Search independently within each letter, then verify high-
            # scoring views at finer resolution. All rows are individuals.
            top=np.argpartition(coarse[i],-56,axis=1)[:,-56:]
            idx=(top + baseidx).reshape(-1)
            ed=field[i][CONSTANTS['points'][idx]].mean(1)
            cc=np.take_along_axis(coarse[i],top,axis=1).ravel()
            pre=(.63*cc+.37*ed).reshape(nclass,-1)
            subset=np.argpartition(pre,-12,axis=1)[:,-12:]
            choose=(subset+np.arange(nclass)[:,None]*56).ravel()
            idx=idx[choose]; cc=cc[choose]; ed=ed[choose]
            fd=CONSTANTS['fine'][idx].astype(np.float32)
            sd=CONSTANTS['shade'][idx].astype(np.float32)
            ff=(fd*f[i]).sum(1)/8192
            ss=(sd*s[i]).sum(1)/8192
            sc=(.23*cc+.37*ff+.29*ed+.11*ss).reshape(nclass,-1)
            best=np.sort(sc,axis=1)[:,-3:]
            answers[start+i]=np.argmax(best[:,2]*.65+best[:,1]*.25+best[:,0]*.10)
    return answers

searchbank = CONSTANTS['search'].astype(np.float32)/8192
baseidx = np.arange(24)[:,None]*(len(searchbank)//24)
'''

# Numerical helpers for the builder only.
def unit(a):
    return a/np.sqrt(np.sum(a*a,axis=-1,keepdims=True)+1e-18)

def rot(a,axis):
    c=math.cos(math.radians(a)); s=math.sin(math.radians(a))
    if axis==0: return np.array(((1,0,0),(0,c,-s),(0,s,c)))
    if axis==1: return np.array(((c,0,s),(0,1,0),(-s,0,c)))
    return np.array(((c,-s,0),(s,c,0),(0,0,1)))

# Viewing convention: the thumb is on the right of a palm-facing fist.
# A rotation about Y toward +90 presents the radial profile (C or O).
views={
 'A':(5,4,-7), 'B':(0,0,0), 'C':(85,-4,-8), 'D':(12,0,0),
 'E':(2,0,-3), 'F':(40,0,-6), 'G':(44,14,-84), 'H':(52,-5,-82),
 'I':(0,5,0), 'K':(-28,0,4), 'L':(0,2,0), 'M':(-6,14,-12),
 'N':(-12,12,-13), 'O':(79,2,-9), 'P':(-80,18,65), 'Q':(78,-18,202),
 'R':(-8,0,0), 'S':(0,8,-10), 'T':(-3,13,-84), 'U':(1,0,0),
 'V':(0,1,0), 'W':(4,1,0), 'X':(-83,-5,-5), 'Y':(0,5,0)}

# Nodes have xyz, radius. Primitives are tapered round bones, or ellipsoids.
def hand(letter, rng, variant):
    un=rng.uniform
    finger_scale=un(.87,1.15)
    thick=un(.91,1.10)
    palmw=un(.92,1.07)
    spread=un(.85,1.18)
    bones=[]; solids=[]
    def chain(nodes,rad,kind=0):
        for a,b,ra,rb in zip(nodes,nodes[1:],rad,rad[1:]):
            bones.append((np.array(a,dtype=float),np.array(b,dtype=float),ra*thick,rb*thick,kind))
    def ell(pos,r,kind=0,angle=0):
        solids.append((np.array(pos),np.array(r),kind,rot(angle,2)))
    # Palm and the fleshy thenar and hypothenar pads.
    ell((0,-.04,0),(1.02*palmw,1.16,.39))
    ell((.46*palmw,-.23,.155),(.46,.82,.285),angle=-24)
    ell((-.54*palmw,-.24,.095),(.39,.78,.27),angle=15)
    # The carpal-to-metacarpal bridge joins the knuckles with a rounded bar.
    chain([(-.70*palmw,.55,-.025),(-.25*palmw,.72,-.025),(.23*palmw,.76,-.025),(.68*palmw,.69,-.025)], [.265,.305,.30,.255])
    straight=(4,1,1); folded=(64,100,74); two=(66,72,62)
    fs=[folded]*4
    splay=[-6,-2,1,3]          # from little through index
    if letter=='B': fs=[straight]*4; splay=[3,1,-1,-4]
    if letter=='C': fs=[(14,38,42)]*4; splay=[4,2,0,-4]
    if letter=='D': fs=[two,two,two,straight]; splay[-1]=-3
    if letter=='E': fs=[(33,101,82)]*4
    if letter=='F': fs=[straight,straight,straight,(54,75,54)]; splay=[-27,-8,12,-7]
    if letter in 'GQ': fs[-1]=straight
    if letter in 'HU': fs[-2:]=[straight,straight]; splay[-2:]=[3,-4]
    if letter in 'IY': fs[0]=straight; splay[0]=-18 if letter=='Y' else -10
    if letter in 'KP': fs[-2:]=[(38,4,2),straight]; splay[-2:]=[-15,10]
    if letter=='P': fs[-2:]=[(79,5,1),(9,0,0)]
    if letter=='L': fs[-1]=straight; splay[-1]=-7
    if letter=='M': fs=[(80,96,65),(37,78,53),(34,76,50),(36,83,54)]
    if letter=='N': fs=[(79,95,65),(73,96,68),(36,75,55),(38,84,58)]
    if letter=='O': fs=[(32,51,45)]*4; splay=[12,7,0,-8]
    if letter=='R': fs[-2:]=[(14,-4,-3),(-2,2,2)]; splay[-2:]=[12,-11]
    if letter=='T' and variant%4!=0:
        # Extended-index, thumb-raised pointing handshape.
        fs[-1]=straight
    if letter=='S': fs=[(70,88,79)]*4
    if letter=='T' and variant%4==0: fs[-1]=(35,65,40)
    if letter=='V': fs[-2:]=[straight,straight]; splay[-2:]=[-14,16]
    if letter=='W': fs[1:]=[straight]*3; splay[1:]=[-17,0,19]
    if letter=='X': fs[-1]=(3,65,65); splay[-1]=2
    # Add variation in curl, correlated across fingers, and tiny variations
    # in individual joints. A closed fist is not four parallel pillars.
    bend=un(-8,8); tips=[]
    for j,(u,y,ln,r) in enumerate(zip((-.72,-.27,.19,.65),(.54,.76,.83,.72),(1.28,1.69,1.83,1.62),(.192,.218,.224,.219))):
        p=np.array((u*palmw,y,-.015)); nodes=[p]
        phi=math.radians(splay[j]*spread+un(-3,3)); theta=0
        for i,(f,q) in enumerate(zip(fs[j],(.44,.315,.245))):
            theta+=math.radians(f+(bend if i==0 else .32*bend)+un(-2.5,2.5))
            # Curl in a plane directed slightly toward the center of the palm.
            dr=np.array((math.sin(phi)*math.cos(theta)-.09*u*math.sin(theta),math.cos(phi)*math.cos(theta),math.sin(theta)))
            p=p+unit(dr)*ln*finger_scale*q; nodes.append(p)
        chain(nodes,[r,r*.98,r*.865,r*.77]); tips.append(p)
    # Opposable thumb. Its pad hinges at the base of the palm, not the
    # rightmost finger knuckle. Terminal targets describe contacts in ASL.
    root=np.array((.48*palmw,-.57,.23))
    if letter=='A': thumb=[(.90,-.08,.33),(1.04,.47,.36),(1.04,.99,.34)]
    elif letter=='B': thumb=[(.69,-.05,.43),(.23,.29,.56),(-.31,.45,.51)]
    elif letter in 'CO':
        if letter=='C': thumb=[(.80,-.10,.63),(.67,.18,1.16),(.54,.41,1.58)]
        else:
            target=tips[3]+np.array((.01,-.19,.045))
            thumb=[(.83,-.09,.61),(.77,.25,1.11),target]
    elif letter=='D': thumb=[(.84,-.07,.46),(.67,.45,.80),tips[2]+np.array((0,-.05,.04))]
    elif letter=='E': thumb=[(.74,-.08,.42),(.21,.10,.56),(-.32,.13,.52)]
    elif letter=='F':
        target=tips[3]+np.array((.03,-.08,.08)); thumb=[(.91,-.19,.69),(.96,.04,1.12),target]
    elif letter in 'GQ':
        thumb=[(.89,-.04,.45),(1.06,.47,.58),(1.08,1.05,.71)] if letter=='G' else [(.94,-.12,.64),(1.12,.10,1.10),(1.21,.48,1.46)]
    elif letter in 'KP':
        target=np.array((.40,1.43,.61)) if letter=='K' else np.array((.42,1.04,1.01))
        thumb=[(.86,-.12,.42),(.80,.50,.69),target]
    elif letter in 'LY':
        thumb=[(1.01,-.35,.29),(1.48,-.14,.30),(1.96,.04 if letter=='L' else .20,.28)]
    elif letter=='T' and variant%4!=0:
        thumb=[(.94,-.33,.21),(1.35,-.22,.22),(1.77,-.18,.21)]
    elif letter in 'MNT':
        k={'M':-.53,'N':-.09,'T':.44}[letter]
        target=(k,.93 if letter!='T' else 1.32,.63)
        thumb=[(.79,-.11,.43),((k+.80)*.5,.46,.71),target]
    elif letter=='S': thumb=[(.74,.01,.49),(.32,.64,.82),(-.15,.83,.81)]
    elif letter=='W': thumb=[(.56,-.13,.45),(-.01,.12,.53),tips[0]+np.array((0,.04,.22))]
    else: thumb=[(.72,-.05,.46),(.35,.38,.67),(-.14,.60,.68)]
    t=[root]+[np.asarray(p)+rng.normal(0,.055,3) for p in thumb]
    chain(t,[.33,.292,.236,.188])
    return solids,bones

# Camera-space sphere-swept segments, with a height buffer and an object id
# buffer. Ellipsoids use the actual ray/quadric quadratic, not flat polygons.
def render(letter,rng,variant,count,side=56):
    solids,bones=hand(letter,rng,variant)
    yaw,pitch,roll=views[letter]
    if letter=='T':
        if variant%4: yaw,roll=170,-95
        else: roll=-9
    if letter=='X' and variant%3==2: yaw=43
    if letter=='K' and variant%2==0: yaw=-63
    if letter in 'GH' and variant%4==0: yaw=-yaw
    if letter in 'MNS' and variant%3==1: pitch+=22
    yaw+=rng.uniform(-32,32); pitch+=rng.uniform(-20,20); roll+=rng.uniform(-14,14)
    if letter in 'CDOFX': yaw+=rng.uniform(-12,12)
    if letter=='F': yaw-=15 if variant%3==0 else 0
    if letter=='Q': roll+=rng.uniform(-22,22)
    cam=rot(roll,2) @ rot(pitch,0) @ rot(yaw,1)
    shapes=[]; bounds=[]
    def spherebox(p,r):
        bounds.extend([p[:2]-r,p[:2]+r])
    for pos,r,kind,local in solids:
        p=cam@pos; axes=cam@local
        shapes.append(('e',p,r,axes,kind)); ext=np.sqrt(((axes*r)**2).sum(1))
        spherebox(p,ext[:2])
    for a,b,r,s,kind in bones:
        a=cam@a; b=cam@b
        shapes.append(('b',a,b,r,s,kind)); spherebox(a,r); spherebox(b,s)
    bounds=np.array(bounds); low=bounds.min(0); high=bounds.max(0); size=high-low
    wrist=cam@np.array((0,-.83,-.055))
    # Forearm normally hangs down; a flexed wrist can form a substantial
    # angle with the metacarpals. Radial pointing comes with a lateral arm.
    if letter in 'GHT': off=rng.uniform(-1.9,-.34)
    elif letter=='P': off=rng.uniform(.65,2.4)
    elif letter=='Q': off=rng.uniform(-.15,1.25)
    else: off=np.clip(-cam[0,1]*.65 + rng.uniform(-.33,.33),-.78,.78)
    far=wrist+np.array((off*4,-4,-4.15 if letter in 'PQ' else -.60))
    armrad=rng.uniform(.57,.72)
    shapes.insert(0,('b',wrist,far,armrad,armrad*1.19,1))
    shapes.insert(1,('b',wrist,cam@np.array((0,-.40,-.08)),armrad,.78,1))
    # Vary framing without treating the elongated forearm as part of the
    # bounding box of the sign. Include a length of forearm below the hand.
    bottom=rng.uniform(.25,1.48) if letter not in 'GHPQT' else rng.uniform(.15,.8)
    low[1]-=bottom
    frame=max(high[0]-low[0],high[1]-low[1])*rng.uniform(.94,1.18)
    mid=(high+low)*.5
    mid+=rng.uniform(-.054,.054,2)*frame
    x=(np.arange(side)+.5-side/2)*(frame/side)+mid[0]
    y=mid[1]-(np.arange(side)+.5-side/2)*(frame/side)
    X,Y=np.broadcast_arrays(x[None,:],y[:,None]); Z=np.full((side,side),-100.)
    I=np.zeros_like(Z); light=unit(np.array((rng.uniform(-1.4,-.40),rng.uniform(.45,1.5),rng.uniform(.32,1.2))))
    if variant%9==2: light[0]*=-1
    def paint(z,nx,ny,nz,ok,kind,tone):
        sel=ok & (z>Z)
        if not np.any(sel):return
        intensity=(nx*light[0]+ny*light[1]+nz*light[2]); l=np.maximum(intensity,0)
        value=tone*(.59*l+.09*np.minimum(intensity,0)+.085*nz+.04*l**8)
        Z[sel]=z[sel]; I[sel]=value[sel]
    for ob in shapes:
        if ob[0]=='e':
            p,r,axes,kind=ob[1:]; A=(axes/(r*r))@axes.T
            dx=X-p[0]; dy=Y-p[1]
            b=(A[2,0]*dx+A[2,1]*dy); c=A[0,0]*dx*dx+(2*A[0,1]*dx+A[1,1]*dy)*dy-1
            disc=b*b-A[2,2]*c; dz=(np.sqrt(np.maximum(disc,0))-b)/A[2,2]
            nx=A[0,0]*dx+A[0,1]*dy+A[0,2]*dz; ny=A[1,0]*dx+A[1,1]*dy+A[1,2]*dz; nz=A[2,2]*dz+b
            n=1/np.sqrt(nx*nx+ny*ny+nz*nz+1e-14)
            paint(p[2]+dz,nx*n,ny*n,nz*n,disc>=0,kind,1.)
        else:
            a,b,r,s,kind=ob[1:]; vx,vy,vz=b-a; dx=X-a[0]; dy=Y-a[1]; dr=s-r
            # Upper envelope of the spheres along the bone: maximize
            # a.z+t*v.z+sqrt((r+t*dr)^2-|p-a.xy-t*v.xy|^2).
            A=vx*vx+vy*vy-dr*dr; B=dx*vx+dy*vy+r*dr; C=r*r-dx*dx-dy*dy
            if A>1e-7:
                tc=B/A; disc=C+B*tc
                t=np.clip(tc+vz*np.sqrt(np.maximum(disc,0)/(A*(A+vz*vz))),0,1)
            else:t=np.zeros_like(Z) if vz<0 else np.ones_like(Z)
            # If the ray does not hit the infinite cone/cylinder, reject.
            rr=r+t*dr; nx=dx-t*vx; ny=dy-t*vy
            sq=rr*rr-nx*nx-ny*ny; dz=np.sqrt(np.maximum(sq,0))
            paint(a[2]+t*vz+dz,nx/rr,ny/rr,dz/rr,sq>=0,kind,1.)
    mask=(Z>-99); skin=rng.uniform(.25,.42); bg=rng.uniform(.47,.82)
    # Depth discontinuities make soft contact shadows between fingers.
    occ=np.zeros_like(Z)
    zp=np.pad(Z,2,mode='edge')
    for sy,sx in ((0,2),(2,0),(4,2),(2,4),(1,1),(1,3)):
        occ+=np.clip((zp[sy:sy+side,sx:sx+side]-Z)*3-.10,0,1)
    image=np.where(mask, skin+I*(1-skin)-.032*occ,bg)
    return image,mask.astype(float),bg

# Each row is an independent reference; never center, normalize or otherwise
# summarize across different renders. Image warps also remain separate rows.
def build(folder):
    folder.mkdir(exist_ok=True,parents=True)
    (folder/'solution.py').write_text(runtime+infer)
    scope={}; exec(runtime,scope); describe=scope['features']
    rng=np.random.default_rng(802711)
    letters='ABCDEFGHIKLMNOPQRSTUVWXY'; poses=208; warps=10
    point_rng=np.random.default_rng(123776)
    n=len(letters)*poses*warps
    bank=np.empty((n,6*49+196),np.int16)
    fine=np.empty((n,6*196),np.int16); shade=np.empty((n,196),np.int16)
    points=np.empty((n,72),np.uint16)
    rr=np.arange(28)*2+.5; xx,yy=np.meshgrid(rr,rr)
    def outline(mask,image):
        # Sample an oriented contour with spatial repulsion. Selection,
        # including the interior crease samples, is local to this ONE view.
        def gradients(a):
            z=np.pad(a,1,mode='edge'); dx=z[1:-1,2:]-z[1:-1,:-2]; dy=z[2:,1:-1]-z[:-2,1:-1]
            mag=np.sqrt(dx*dx+dy*dy)
            ori=np.floor(np.arctan2(dy,dx)*(6/np.pi)+.5).astype(int)%6
            return mag,ori
        def pick(mag,ori,k):
            mag=mag*point_rng.uniform(.85,1.15,(28,28))*np.linspace(1.15,.72,28)[:,None]
            mag[[0,-1]]*=.15; mag[:,[0,-1]]*=.15
            take=[]
            for i in range(k):
                q=mag.argmax(); y,x=divmod(q,28); take.append(ori[y,x]*784+q)
                mag[max(0,y-1):y+2,max(0,x-1):x+2]*=.24; mag[y,x]=0
            return take
        strength,angle=gradients(mask)
        m,o=gradients(image)
        inside=scope['blurone'](mask[None])[0]
        return pick(strength,angle,48)+pick(m*inside**3*np.exp(-4*strength),o,24)
    def interp(a,x,y,outside):
        p=np.pad(a,((1,1),(1,1)),mode='edge'); x=np.clip(x+1,0,56.9999); y=np.clip(y+1,0,56.9999)
        i=x.astype(int); j=y.astype(int); u=x-i; v=y-j
        return ((1-v)*((1-u)*p[j,i]+u*p[j,i+1])+v*((1-u)*p[j+1,i]+u*p[j+1,i+1]))
    def quant(z):return np.rint(z*8192).astype(np.int16)
    for j,letter in enumerate(letters):
        for v in range(poses):
            image,mask,bg=render(letter,rng,v,poses)
            images=[]; masks=[]
            for w in range(warps):
                scale=1 if w==0 else rng.uniform(.84,1.20)
                sx=scale*rng.uniform(.94,1.06); sy=scale*rng.uniform(.95,1.05)
                angle=0 if w==0 else rng.uniform(-.11,.11); ca=math.cos(angle); sa=math.sin(angle)
                tx,ty=(0,0) if w==0 else rng.uniform(-5.2,5.2,2)
                dx=(xx-27.5)*sx; dy=(yy-27.5)*sy
                x=27.5+ca*dx-sa*dy+tx; y=27.5+ca*dy+sa*dx+ty
                def sample(a,fill):
                    return sum(interp(a,x+i,y+k,fill) for i in (-.5,.5) for k in (-.5,.5))*.25
                images.append(sample(image,bg)); masks.append(sample(mask,0))
            h,f,s,m=describe(np.array(images,np.float32),np.array(masks,np.float32))
            lo=(j*poses+v)*warps; sl=slice(lo,lo+warps)
            bank[sl]=quant(np.concatenate((h,m),axis=1)); fine[sl]=quant(f); shade[sl]=quant(s)
            points[sl]=[outline(ma,im) for ma,im in zip(masks,images)]
        print('Built independent poses for',letter,flush=True)
    np.savez(folder/'constants.npz',search=bank,fine=fine,shade=shade,points=points)

if __name__=='__main__':build(Path(sys.argv[1]))
