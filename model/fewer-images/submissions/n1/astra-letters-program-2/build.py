"""Build a bank of individual, procedurally rendered ASL hand templates. No input assets."""
from pathlib import Path
import sys
import math
import numpy as np

# Feature code is shared literally with the submitted program. For the builder it
# sees just one procedural photo per row. There are no cross-row reductions.
features = '''import numpy as np

def blur(a):
    b = np.pad(a, ((0,0),(0,0),(1,1)), mode='edge')
    a = .25*(b[:,:,:-2] + 2*b[:,:,1:-1] + b[:,:,2:])
    b = np.pad(a, ((0,0),(1,1),(0,0)), mode='edge')
    return .25*(b[:,:-2] + 2*b[:,1:-1] + b[:,2:])

def block(a, n):
    b = a.reshape(len(a),28//n,n,28//n,n)
    return b.mean(axis=(2,4))

def grad(a):
    b = np.pad(a, ((0,0),(1,1),(1,1)), mode='edge')
    return .5*(b[:,1:-1,2:]-b[:,1:-1,:-2]), .5*(b[:,2:,1:-1]-b[:,:-2,1:-1])

def activity(im, dx, dy):
    # Weighted multi-source flood from the exposed borders. Path cost is high
    # through a contour, a finger seam, or a region of sustained texture.
    pad = np.pad(im, ((0,0),(1,1),(1,1)), mode='edge')
    edge = np.sqrt(dx*dx+dy*dy)
    scale = np.maximum(.008, np.sqrt(np.mean(edge**2,axis=(1,2),keepdims=True)))
    e = edge/scale
    q = np.full_like(im, 30.)
    q[:,0,:]=0.
    q[:,:,0]=0.
    q[:,:,-1]=0.
    q[:,-1,:]=np.minimum(q[:,-1,:],4.)
    links=[]
    for j,i in ((0,1),(2,1),(1,0),(1,2)):
        diff=np.abs(pad[:,j:j+28,i:i+28]-im)/scale
        links.append(.025 + .27*e + .75*diff)
    for step in range(28):
        prev=np.pad(q, ((0,0),(1,1),(1,1)), mode='edge')
        for t,(j,i) in enumerate(((0,1),(2,1),(1,0),(1,2))):
            np.minimum(q, prev[:,j:j+28,i:i+28]+links[t], out=q)
    # Low-frequency region cue; normalized independently on each crop.
    a=1.-np.exp(-.32*q)
    a=a+.6*np.minimum(blur(blur(e)),1.6)
    a=a-a.mean(axis=(1,2),keepdims=True)
    return a/(np.sqrt((a*a).mean(axis=(1,2),keepdims=True))+.08)

def collect(im):
    im = im.astype(np.float32)
    sm = .4*im + .6*blur(im)
    x,y=grad(sm)
    x1,y1=grad(blur(blur(sm)))
    mag=np.sqrt(x*x+y*y)
    noise = .003 + .20*mag.mean(axis=(1,2),keepdims=True)
    reg=activity(sm,x,y)
    # Orientations are axial, not directed: reversing skin/background contrast
    # or swapping the lit and shaded sides of a finger need not change the match.
    fine=[]
    for k in range(6):
        a=k*np.pi/6
        ca,sa=np.float32(np.cos(a)),np.float32(np.sin(a))
        e=np.maximum(0., np.abs(ca*x+sa*y)-.75*np.abs(sa*x-ca*y))
        e1=np.maximum(0., np.abs(ca*x1+sa*y1)-.75*np.abs(sa*x1-ca*y1))
        e=blur(.75*e+.4*e1)
        fine.append(block(e,2))
    fine=np.stack(fine,axis=-1)
    den=fine.sum(axis=-1,keepdims=True)/2.8
    ref=noise.reshape(-1,1,1,1)
    # A saturating magnitude preserves empty space without letting a single
    # black background edge dominate all the internal contours of the hand.
    fine=fine/(.45*den+1.8*ref+np.sqrt(den*ref))
    r=block(blur(reg),2)
    # A signed center-surround channel distinguishes a highlighted nail or
    # phalanx from a dark inter-finger crease with similar axial edges.
    surround=sm
    for step in range(12):
        surround=blur(surround)
    relief=block(sm-surround,2)
    relief=relief/(.012+np.sqrt((relief*relief).mean(axis=(1,2),keepdims=True)))
    f=np.concatenate([fine, r[:,:,:,None]*.63, np.clip(relief,-2.,2.)[:,:,:,None]*.45],axis=3)
    return f

def summary(f):
    # A fixed coarse grid for the exhaustive first pass.
    s=f.reshape(len(f),7,2,7,2,8).mean(axis=(2,4))
    # Mild high pass: unrelated smooth edge density should not give a strong match.
    s=s-.70*s.mean(axis=(1,2),keepdims=True)
    s=s.reshape(len(s),-1)
    return s / (1.e-8+np.sqrt((s*s).sum(axis=-1,keepdims=True)))
'''

inference = '''
coarsebank = CONSTANTS['coarse'].astype(np.float32)
# The fixed quantization scales are a storage convention, not corpus statistics.
coarsebank *= np.float32(1./32767.)
finebank = CONSTANTS['fine']
poses = finebank.shape[0]//24

def refine(query, cand, approx):
    # Score a few alignments of each shortlisted pose. Neighboring grid patches
    # can move by one cell to allow small changes in joint angle and anatomy.
    m=len(cand)
    ts=finebank[cand].astype(np.float32)*(1./50.)
    qpad=np.pad(query,((1,1),(1,1),(0,0)), mode='edge')
    ans=np.zeros((m,4,4),np.float32)-1.e9
    # fixed center-surround descriptors. Cell weights vary with detail rather
    # than screen position, except for a gentle discount at the crop boundary.
    avg=ts.mean(axis=(1,2),keepdims=True)
    t=(ts-.70*avg)
    qbase=query.mean(axis=(0,1),keepdims=True)*.70
    tt=(t*t).sum(axis=-1)
    border=np.ones(14,np.float32); border[0]=.55; border[-1]=.55
    weight=border[:,None]*border[None,:]
    # Four overlapping bands in each direction, implemented with integral sums.
    # Compare normalized orientations plus area cue using a bounded similarity.
    low=np.array([0,3,6,9]); high=np.array([5,8,11,14])
    for yy,xx in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
        q=qpad[1+yy:15+yy,1+xx:15+xx]-qbase
        qq=(q*q).sum(axis=-1)
        dot=np.einsum('mijk,ijk->mij',t,q)
        # A softened Dice coefficient whose effective weight grows with evidence.
        d=(2*dot - .56*(tt+qq)) / (1.4+.28*(tt+qq))
        u=np.pad(d*weight,((0,0),(1,0),(1,0))).cumsum(1).cumsum(2)
        v=u[:,high[:,None],high] - u[:,low[:,None],high] - u[:,high[:,None],low] + u[:,low[:,None],low]
        v=v/25.-.031*(xx*xx+yy*yy)
        ans=np.maximum(ans,v)
    local=ans.mean(axis=(1,2))
    return .53*approx+.47*local

def predict(x):
    result=np.empty(len(x),dtype=np.int64)
    for offset in range(0,len(x),32):
        n=min(32,len(x)-offset)
        fine=collect(x[offset:offset+n,0])
        # Fixed GEMM dimensions even for a singleton call.
        b=np.zeros((32,coarsebank.shape[1]),np.float32)
        b[:n]=summary(fine)
        coarse=(b@coarsebank.T).reshape(32,24,poses)
        for i in range(n):
            vals=coarse[i]
            pick=np.argpartition(vals,poses-4,axis=1)[:,-4:]
            ind=(pick+np.arange(24)[:,None]*poses).ravel()
            a=np.take_along_axis(vals,pick,axis=1).ravel()
            score=refine(fine[i],ind,a).reshape(24,4)
            ss=np.sort(score,axis=-1)
            result[offset+i]=np.argmax(.75*ss[:,-1]+.25*ss[:,-2])
    return result
'''

# All lengths below are in arbitrary anatomical units. Coordinates are (thumbward,
# along the metacarpals, out of the palm), with origin at the wrist crease.
letters = 'ABCDEFGHIKLMNOPQRSTUVWXY'

def unit(v):
    return v/(np.sqrt(np.sum(v*v))+1.e-12)

def rx(t):
    c,s=np.cos(t),np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def ry(t):
    c,s=np.cos(t),np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rz(t):
    c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def pose(letter, rng):
    u,n=rng.uniform,rng.normal
    rad=np.pi/180
    splay=np.array([2.,0.,-2.,-4.]); ang=np.empty((4,3)); ang[:]=[58.,147.,225.]
    yaw,tilt,roll=0.,2.,0.
    # Handedness: for a frontal right palm the index is on the viewer's right.
    # Values are absolute phalanx inclinations in that finger's flexion plane.
    if letter=='A':
        ang[:]=[65,154,230]; yaw=-9
    if letter=='B':
        ang[:]=[1,3,5]; splay[:]=[0,0,0,-1]; yaw=5
    if letter=='C':
        ang[:]=[25,67,108]; yaw=71; tilt=6; roll=12
    if letter=='D':
        ang[0]=[0,2,4]; ang[1:]=[34,98,158]; yaw=20
    if letter=='E':
        ang[:]=[16,116,183]; tilt=6
    if letter=='F':
        ang[0]=[24,95,162]; ang[1:]=[2,3,5]; splay[:]=[1,3,-6,-16]; yaw=3
    if letter=='G':
        ang[0]=[-1,1,4]; yaw=-52; tilt=-12; roll=79
    if letter=='H':
        ang[:2]=[2,4,7]; yaw=-78; tilt=-8; roll=79; splay[:2]=[1,-2]
    if letter=='I':
        ang[3]=[0,2,4]; splay[3]=-8; yaw=5
    if letter=='K':
        ang[0]=[1,2,3]; ang[1]=[33,41,45]; splay[:2]=[7,-18]; yaw=12
    if letter=='L':
        ang[0]=[-1,1,2]; splay[0]=0
    if letter in 'MN':
        ang[:]=[61,139,205]; yaw=-13; tilt=8
    if letter=='O':
        ang[:]=[32,88,150]; yaw=64; roll=8
    if letter=='P':
        ang[0]=[0,0,2]; ang[1]=[48,58,64]; splay[:2]=[8,-12]; yaw=-74; tilt=-19; roll=-81
    if letter=='Q':
        ang[0]=[1,4,9]; yaw=-65; tilt=-9; roll=-139
    if letter=='R':
        ang[0]=[14,5,-3]; ang[1]=[1,-2,1]; splay[:2]=[-10,12]; yaw=10
    if letter=='S':
        ang[:]=[68,158,228]; yaw=1
    if letter=='T':
        ang[:]=[72,147,219]; ang[0]=[34,117,194]; yaw=5
    if letter=='U':
        ang[:2]=[0,2,4]; splay[:2]=[-.9,.9]
    if letter=='V':
        ang[:2]=[0,2,4]; splay[:2]=[14,-13]
    if letter=='W':
        ang[:3]=[1,2,4]; splay[:3]=[16,0,-16]; yaw=-1
    if letter=='X':
        ang[0]=[1,78,122]; yaw=35
    if letter=='Y':
        ang[3]=[0,1,4]; splay[3]=-23; yaw=0
    # Anatomical shape and gesture variation, not augmentation of any photograph.
    width=u(.91,1.11); plen=u(.94,1.07); flen=u(.90,1.10); thick=u(.90,1.10)
    ax=np.array([.66,.215,-.235,-.65])*width
    ay=np.array([1.76,1.84,1.76,1.54])*plen
    az=np.array([.08,.015,.015,.09]); seglen=flen*np.array([[.80,.49,.39],[.88,.55,.42],[.81,.54,.39],[.62,.36,.32]])*u(.96,1.04,(4,1))
    frad=np.array([.225,.234,.218,.184])*thick
    common=n(0,6)
    ang=ang+common+n(0,4,(4,1))+n(0,3,(4,3))
    if letter in 'BHU':
        splay+=n(0,.7,4)
    else:
        splay+=n(0,2,4)
    if letter in 'CVWYK': splay*=u(.75,1.25)
    # Flexion of all fingers on a side view is correlated, but pinky and ring
    # usually curl slightly more tightly towards the heel of the hand.
    if letter in 'CO': ang+=np.arange(4)[:,None]*u(0.,3.)
    xyz=[]
    for i in range(4):
        a=ang[i]*rad; s=splay[i]*rad
        delta=seglen[i,:,None]*np.stack([np.sin(s)*np.cos(a),np.cos(s)*np.cos(a),np.sin(a)],axis=1)
        root=np.array([ax[i],ay[i],az[i]])
        xyz.append(np.vstack([root,root+np.cumsum(delta,axis=0)]))
    # Thumb paths from the basal saddle joint to the MCP, IP, and thumb tip.
    root=np.array([.45,.43,.05])
    thumb=np.array([[.78,.90,.36],[.45,1.42,.67],[-.04,1.53,.73]])
    if letter=='A': thumb=np.array([[.83,.97,.16],[.95,1.55,.32],[.89,2.07,.39]])
    if letter=='B': thumb=np.array([[.74,.90,.37],[.42,1.28,.62],[-.03,1.37,.57]])
    if letter in 'CYL':
        thumb=np.array([[.94,.66,.17],[1.49,.83,.29],[1.98,.96,.40]])
        if letter=='C': thumb=np.array([[.88,.71,.51],[1.05,.99,1.10],[1.04,1.32,1.48]])
        if letter=='Y': thumb[1:,1]+=[.12,.26]
    if letter in 'ODF':
        i=1 if letter=='D' else 0; targ=xyz[i][-1]+np.array([.04,-.18,-.04]); mid=np.array([.96,.86,.52]); v=targ-mid
        thumb=np.array([mid,mid+.55*v+np.array([.09,-.02,.09]),targ])
    if letter=='E': thumb=np.array([[.74,.94,.38],[.36,1.26,.70],[-.17,1.31,.71]])
    if letter in 'GQ': thumb=np.array([[.86,.94,.22],[1.09,1.51,.51],[1.15,2.02,.77]])
    if letter in 'KP':
        targ=.5*(xyz[0][1]+xyz[1][1])+np.array([.02,-.12,.18]); mid=np.array([.83,1.04,.38]); thumb=np.array([mid,.47*mid+.53*targ+np.array([.06,0,.12]),targ])
    if letter in 'MNT':
        tipx={'M':-.47,'N':-.02,'T':.45}[letter]
        tipy={'M':1.84,'N':1.95,'T':2.19}[letter]
        thumb=np.array([[.78,1.03,.35],[(tipx+.70)/2,1.52,.48],[tipx,tipy,.51]])
    if letter=='S': thumb=np.array([[.82,1.12,.40],[.44,1.69,.91],[-.08,1.84,.95]])
    if letter=='W': thumb=np.array([[.69,.83,.37],[.14,1.22,.64],[-.43,1.45,.60]])
    if letter in 'RUV': thumb=np.array([[.72,.93,.48],[.17,1.32,.79],[-.37,1.48,.78]])
    # Thenar-first flexion and opposition of a thumb are not in the fingers' plane.
    thumb[:,0]*=width; thumb[:,1]*=plen
    thumb+=np.linspace(.35,1,3)[:,None]*n(0,.095,(1,3))
    thumb=np.vstack([root,thumb]); xyz.append(thumb)
    # All rendering primitives are convex, tapered sweeps. The palm is a squashy
    # superelliptical surface with cross sections at the heel, arch and knuckles.
    psec=np.array([[-.15,.49,.255,-.015],[.10,.56,.32,0.],[.62,.77,.40,0.],[1.19,.82,.40,0.],[1.55,.81,.31,0.],[1.73,.72,.21,.01],[1.82,.44,.14,.015]])
    psec[:,0]*=plen; psec[:,1]*=width; psec[:,2]*=thick
    # Wrist bends separately from the palm, especially in sideways and downward letters.
    wristx,wristz=0.,0.
    if letter in 'GH': wristx=-.30; wristz=.85
    if letter in 'PQ': wristx=.28; wristz=-.95
    wristx+=n(0,.19); wristz+=n(0,.18)
    # Rotation: first view the knuckle arch from the side, then tilt and roll.
    yaw+=np.clip(n(0,17),-44,44); tilt+=np.clip(n(0,11),-30,30); roll+=np.clip(n(0,10),-29,29)
    rot=rz(-roll*rad)@rx(tilt*rad)@ry(yaw*rad)
    return xyz,frad,psec,rot,(wristx,wristz)

# Render a batch in parallel. An orthographic ray/surface intersection carries
# the depth and the normal of each visible primitive to the shading stage.
# A swept sphere with linearly changing radius gives a tapered capsule; its
# radius profile is slightly pinched at the joints of adjacent phalanges.
def render(letters_batch,rng):
    m=len(letters_batch); rand=rng.uniform
    poses=[pose(c,rng) for c in letters_batch]
    keys=np.array([p[0] for p in poses],dtype=np.float32)
    radii=np.array([p[1] for p in poses],dtype=np.float32)
    rings=np.array([p[2] for p in poses],dtype=np.float32)
    rot=np.array([p[3] for p in poses],dtype=np.float32)
    wrist=np.array([p[4] for p in poses],dtype=np.float32)
    def project(v):
        w=v.reshape(m,-1,3)@rot.transpose(0,2,1)
        return w.reshape(v.shape)
    keys=project(keys)
    ends=keys.reshape(m,-1,3)
    er=np.repeat(np.column_stack([radii,np.full(m,.25)]),4,axis=1)
    lo=(ends[:,:,:2]-er[:,:,None]).min(1); hi=(ends[:,:,:2]+er[:,:,None]).max(1)
    hull=np.array([[-.52,0,0],[.52,0,0],[.78,1.25,0],[-.78,1.25,0],[0,1.4,.34],[0,1.4,-.34]])
    hull=project(np.broadcast_to(hull,(m,6,3)).astype(np.float32)); lo=np.minimum(lo,hull[:,:,:2].min(1)); hi=np.maximum(hi,hull[:,:,:2].max(1))
    # Frame the signed gesture, not its whole forearm. Extra headroom, modest
    # off-center crops and incomplete wrists are legitimate camera variation.
    wh=hi-lo; h=np.maximum(wh[:,0],wh[:,1]); box=h*rand(1.01,1.39,m)
    center=.5*(lo+hi)+box[:,None]*np.column_stack([rng.normal(0,.065,m),rng.normal(-.045,.059,m)])
    # Pixel centers in camera space, two samples per dimension after rasterization.
    side=56
    axis=((np.arange(side)+.5)/side-.5).astype(np.float32)
    box=box.astype(np.float32); center=center.astype(np.float32)
    X=center[:,0,None,None]+box[:,None,None]*axis[None,None,:]; Y=center[:,1,None,None]-box[:,None,None]*axis[None,:,None]
    Z=np.zeros((m,side,side),np.float32)-20
    norm=np.zeros((m,side,side,3),np.float32); ink=np.ones_like(Z)
    def surface(z,nx,ny,nz,color,priority=0.):
        select=z>Z+priority
        np.copyto(Z,z,where=select)
        for i,v in enumerate((nx,ny,nz)): np.copyto(norm[:,:,:,i],v,where=select)
        np.copyto(ink,color,where=select)
    def field(a): return np.asarray(a,dtype=np.float32).reshape(-1,1,1)
    def capsule(a,b,r0,r1,color=1.,joint=False):
        # Union of balls on a line. Maximize the upper ray intersection along
        # the line analytically for a cylinder, then account for taper once.
        px=X-field(a[:,0]); py=Y-field(a[:,1]); vx,vy,vz=[field(z) for z in (b-a).T]
        l2=vx*vx+vy*vy+1.e-9; ll=l2+vz*vz
        s=(px*vx+py*vy)/l2; perp=(px*vy-py*vx)**2/l2
        ra,rb=field(r0),field(r1); dr=rb-ra
        rr=ra+np.clip(s,0,1)*dr
        t=np.clip(s+(vz+dr)*np.sqrt(np.maximum(0,rr*rr-perp)/(l2*ll+1.e-12)),0,1)
        r=ra+t*dr
        nx=px-t*vx; ny=py-t*vy; nz2=r*r-nx*nx-ny*ny
        nz=np.sqrt(np.maximum(0,nz2)); z=np.where(nz2>0,field(a[:,2])+t*vz+nz,-30.)
        if joint: color=color*(.935+.065*np.minimum(1.,np.minimum(t,1.-t)*7))
        surface(z,nx/r,ny/r,nz/r,color)
    # Ellipsoid intersection under arbitrary rigid rotation; soft elongated
    # ellipsoids blend the heel, metacarpal pad, thenar/hypothenar muscles.
    def ell(center_local,axes,localrot=None,color=1.):
        cent=project(np.broadcast_to(center_local,(m,3)).astype(np.float32))
        mat=rot if localrot is None else rot@localrot
        scaled=mat/np.broadcast_to(axes,(m,3))[:,None,:]
        quad=scaled@scaled.transpose(0,2,1)
        x=X-field(cent[:,0]);y=Y-field(cent[:,1]); aa=field(quad[:,2,2]); bb=field(quad[:,0,2])*x+field(quad[:,1,2])*y
        cc=field(quad[:,0,0])*x*x+2.*field(quad[:,0,1])*x*y+field(quad[:,1,1])*y*y-1.
        disc=bb*bb-aa*cc; v=(np.sqrt(np.maximum(disc,0))-bb)/aa
        z=np.where(disc>0,field(cent[:,2])+v,-30.)
        nx=field(quad[:,0,0])*x+field(quad[:,0,1])*y+field(quad[:,0,2])*v
        ny=field(quad[:,1,0])*x+field(quad[:,1,1])*y+field(quad[:,1,2])*v
        nz=bb+aa*v; inv=1./np.sqrt(nx*nx+ny*ny+nz*nz+1.e-8)
        surface(z,nx*inv,ny*inv,nz*inv,color)
    # Cast the palm and forearm as a stack of truncated elliptical cones.
    # Transform each camera ray into anatomical coordinates and intersect the
    # section equation at constant aspect ratio (a quadratic in ray depth).
    def metacarpus(levels):
        # columns: height, lateral center, front center, halfwidth, thickness
        bx=field(rot[:,0,0])*X+field(rot[:,1,0])*Y; dx=field(rot[:,2,0]); by=field(rot[:,0,1])*X+field(rot[:,1,1])*Y; dy=field(rot[:,2,1]); bz=field(rot[:,0,2])*X+field(rot[:,1,2])*Y; dz=field(rot[:,2,2]);
        for l,r in zip(levels[:-1],levels[1:]):
            length=field(r[:,0]-l[:,0]); y0=field(l[:,0]);
            kx=field(r[:,1]-l[:,1])/length; kz=field(r[:,2]-l[:,2])/length; kw=field(r[:,3]-l[:,3])/length
            aspect=field((l[:,3]+r[:,3])/(l[:,4]+r[:,4]));
            sx=bx-field(l[:,1])-kx*(by-y0); sz=(bz-field(l[:,2])-kz*(by-y0))*aspect; wx=field(l[:,3])+kw*(by-y0)
            vx=dx-kx*dy; vz=(dz-kz*dy)*aspect; vw=kw*dy
            aa=vx*vx+vz*vz-vw*vw; bb=sx*vx+sz*vz-wx*vw; cc=sx*sx+sz*sz-wx*wx
            disc=bb*bb-aa*cc; sq=np.sqrt(np.maximum(disc,0)); a_safe=np.where(np.abs(aa)<1.e-8,1.e-8,aa)
            for v in ((-bb+sq)/a_safe,(-bb-sq)/a_safe):
                height=by+dy*v; valid=(disc>=0)&(height>=y0)&(height<=field(r[:,0]))
                nx=sx+vx*v; nz=(sz+vz*v)*aspect; ny=-kx*nx-kz*nz-kw*(wx+vw*v)
                inv=1./np.sqrt(nx*nx+ny*ny+nz*nz+1.e-8)
                ns=[(field(rot[:,j,0])*nx+field(rot[:,j,1])*ny+field(rot[:,j,2])*nz)*inv for j in range(3)]
                surface(np.where(valid,v,-30.),*ns,1.)
    def row(y,cx,cz,w,d): return np.column_stack(np.broadcast_arrays(np.ones(m)*y,cx,cz,w,d)).astype(np.float32)
    levels=[row(-6,5.2*wrist[:,0],5.2*wrist[:,1],.78,.61),row(-1.1,.94*wrist[:,0],.94*wrist[:,1],.55,.36)]
    for r in rings.transpose(1,0,2): levels.append(row(r[:,0],0,r[:,3],r[:,1],r[:,2]))
    metacarpus(levels)
    ell([.37,.72,.24],[.37,.73,.27],rz(-.36).astype(np.float32),.98)
    ell([-.44,.76,.20],[.28,.69,.22],rz(.21).astype(np.float32),.99)
    # Small pads at the knuckles join the finger roots to the dorsal palm.
    for i in range(4):
        a=keys[:,i,0,:]; v=np.zeros((m,3),np.float32); v[:,1]=-.30
        capsule(a+project(v),a,radii[:,i]*1.13,radii[:,i]*1.01)
    for i in range(5):
        radius=radii[:,i] if i<4 else radii[:,0]*1.15
        rs=(1.,.91,.80,.68) if i<4 else (1.26,1.04,.87,.72)
        for j in range(3): capsule(keys[:,i,j],keys[:,i,j+1],radius*rs[j],radius*rs[j+1],joint=True)
    # Lambertian illumination + rim darkening and contact occlusion. Background
    # is featureless with a smooth slope, independent of the letter and mesh.
    light=np.column_stack([rand(-1.15,.85,m),rand(.28,1.25,m),rand(.5,1.3,m)]).astype(np.float32)
    light/=np.sqrt((light*light).sum(1,keepdims=True))
    nlight=(norm*light[:,None,None,:]).sum(-1); skin=(.39+.34*nlight+.10*norm[:,:,:,2])*ink
    # A local depth discontinuity occludes light on the farther surface only.
    foreground=Z>-10
    adj=np.pad(Z,((0,0),(2,2),(2,2)),mode='edge'); occ=0.
    for iy,ix in ((0,2),(2,0),(4,2),(2,4)):
        occ=occ+np.clip((adj[:,iy:iy+side,ix:ix+side]-Z-.095)*2.4,0.,1.)
    skin=skin-.050*occ+field(rand(-.12,.12,m))
    bg=field(rand(.40,.87,m))+field(rand(-.12,.12,m))*axis[None,:,None]+field(rand(-.13,.13,m))*axis[None,None,:]
    img=np.where(foreground,skin,bg)
    img=np.clip(img,0.,1.).reshape(m,28,2,28,2).mean((2,4))
    return img.astype(np.float32)


# An alternative, diagrammatic palm projection. This treats the visible length
# of an opposed or flexed digit as an independent pose variable. A full 3-D
# rotation couples it very tightly to the visibility of *all* other digits.
# Units are again wrist-to-knuckle and phalanx lengths, not picture coordinates.
def diagram(c,rng):
    u,n=rng.uniform,rng.normal; rad=math.pi/180
    curl=['fist']*4; spread=np.zeros(4); turn=u(-.32,.43); palm=1.
    roll=0.; tmode='opposed'
    if c=='A': tmode='beside'; curl=['closed']*4
    if c=='B': curl=['erect']*4; tmode='inward'; spread[:]=[0,0,0,-1]
    if c=='C': curl=['arch']*4; turn=u(.82,1.1); palm=.55; tmode='cup'; roll=12
    if c=='D': curl=['erect','circle','circle','circle']; turn=u(.18,.85); palm=.87; tmode='touchmiddle'
    if c=='E': curl=['claw']*4; tmode='beneath'
    if c=='F': curl=['circle','erect','erect','erect']; spread[:]=[4,6,-5,-18]; tmode='touchindex'
    if c=='G': curl[0]='erect'; turn=u(-.9,.45); palm=.79; roll=81; tmode='measure'
    if c=='H': curl[:2]=['erect']*2; turn=u(-.98,-.30); palm=u(.30,.72); roll=80; spread[:2]=[1,-1]
    if c=='I': curl[3]='erect'; spread[3]=-8
    if c in 'KP':
        curl[:2]=['erect','oblique']; spread[:2]=[8,-24]; tmode='between'
        if c=='P': palm=.70; roll=-79; spread[1]=u(-84,-43); turn=-.69
    if c=='L': curl[0]='erect'; tmode='abducted'
    if c in 'MN': curl=['draped']*4; tmode='under'+c; turn=u(-.72,.50); palm=.94
    if c=='O': curl=['circle']*4; turn=u(.77,1.08); palm=.57; roll=9; tmode='touchindex'
    if c=='Q': curl[0]='erect'; tmode='measure'; turn=u(-.95,.15); palm=.78; roll=-152
    if c=='R': curl[0:2]=['cross','erect']; spread[:2]=[-14,10]
    if c=='S': tmode='across'
    if c=='T': tmode='underT'; curl[0]='roof'
    if c=='U': curl[0:2]=['erect']*2; spread[:2]=[-1,1]
    if c=='V': curl[0:2]=['erect']*2; spread[:2]=[14,-14]
    if c=='W': curl[0:3]=['erect']*3; spread[:3]=[16,1,-16]; tmode='toPinky'
    if c=='X': curl[0]='hook'; turn=u(.21,1.)
    if c=='Y': curl[3]='erect'; spread[3]=-24; tmode='abducted'
    # Independent drawing-width, foreshortening, and abduction variations.
    w=palm*u(.89,1.09); h=u(.91,1.07); big=u(.88,1.11); fat=u(.90,1.12)
    base=np.array([[.64,1.63],[.21,1.74],[-.22,1.68],[-.62,1.48]])*[w,h]
    spread*=u(.71,1.29); spread+=n(0,.8 if c in 'RUBH' else 2.,4)
    bones=[]
    for i,style in enumerate(curl):
        leng=[1.65,1.78,1.66,1.31][i]*big*u(.94,1.06); a=spread[i]*rad
        def projected(xs,ys): return np.column_stack([xs,ys])
        # Per-joint displacement in units of the unfolded digit length. A folded
        # digit is modelled with the dorsal knuckle above two shorter nail pads.
        if style in ('erect','oblique','cross'):
            pts=np.array([[0,0],[0,.47],[.006,.78],[.012,1.]])
            if style=='oblique': pts[:,1]*=u(.56,.88) if c=='K' else u(.76,1.0)
            if style=='cross': pts[1:,0]+=[-.035,-.045,-.025]
        elif style=='arch':
            pts=projected(np.array([0,.17,.46,.66])*turn,[0,.42,.58,.49])
        elif style=='circle':
            pts=projected(np.array([0,.25,.52,.51])*turn,[0,.39,.43,.22])
            if c=='F': pts[:,0]+=[0,.03,.03,-.04]
        elif style=='hook':
            pts=projected(np.array([0,0,.23,.35])*turn,[0,.48,.60,.44])
        elif style=='claw':
            pts=projected(np.array([0,.11,.27,.24])*turn,[0,.31,.19,-.04])
        elif style=='roof':
            pts=projected(np.array([0,.20,.42,.37])*turn,[0,.34,.23,.03])
        elif style=='draped':
            pts=projected(np.array([0,.24,.44,.41])*turn,[0,.25,.13,-.10])
        else:
            pts=projected(np.array([0,.29,.44,.35])*turn,[0,.19,.07,-.13])
            if style=='closed': pts[:,1]+=[0,.03,.01,-.015]
        pts[:,1]*=u(.92,1.07); pts[1:]+=n(0,.013,(3,2)); pts=base[i]+leng*(pts@rz(-a)[:2,:2].T)
        bones.append(pts)
    root=np.array([.47*w,.39*h]); p=np.array([[.80*w,1.00*h],[.38*w,1.37*h],[-.12*w,1.51*h]])
    if tmode=='beside': p=[[.84*w,.96*h],[.96*w,1.54*h],[.91*w,2.08*h]]
    if tmode=='inward': p=[[.80*w,.92*h],[.42*w,1.22*h],[-.03*w,1.37*h]]
    if tmode=='across': p=[[.86*w,1.15*h],[.49*w,1.66*h],[-.07*w,1.78*h]]
    if tmode=='beneath': p=[[.78*w,.94*h],[.36*w,1.32*h],[-.19*w,1.29*h]]
    if tmode=='cup': p=[[.75*w+.14,.71*h],[.76*w+.64,.92*h],[.76*w+1.02,1.29*h]]
    if tmode=='abducted':
        rise=u(.32,.90) if c=='L' else u(.63,1.15)
        p=[[.90*w,.66*h],[1.42*w,(.66+.50*rise)*h],[1.91*w,(.66+rise)*h]]
    if tmode in ('touchindex','touchmiddle'):
        tip=bones[0 if tmode=='touchindex' else 1][-1]+[.08,-.15]
        mid=[.82*w+max(0,turn)*.20,.84*h]; p=[mid,.45*np.array(mid)+.55*tip+[.16,-.03],tip]
    if tmode=='between':
        tip=.50*(bones[0][1]+bones[1][1])+[0,-.06]; mid=[.77*w,1.00*h]
        p=[mid,.48*np.array(mid)+.52*tip+[.07,.07],tip]
    if tmode.startswith('under'):
        tx={'M':-.44,'N':-.015,'T':.44}[c]; ty={'M':1.77,'N':1.90,'T':2.15}[c]
        p=[[.72*w,1.0*h],[(.62+tx)*.5*w,1.48*h],[tx*w,ty*h]]
    if tmode=='toPinky': p=[[.70*w,.87*h],[.16*w,1.24*h],[-.46*w,1.43*h]]
    if tmode=='measure':
        tx=.95*w+turn*u(.25,.85); ty=u(2.00,2.60)*h
        p=[[.84*w+.10*turn,.95*h],[tx+.07,ty-.53],[tx,ty]]
    p=np.array(p)+np.linspace(.3,1,3)[:,None]*n(0,.083,(1,2)); bones.append(np.vstack([root,p]))
    roll+=np.clip(n(0,11),-30,30); R=rz(-roll*rad)[:2,:2]
    arms={'G':[-.70,-.8],'H':[-.70,-.8],'P':[.85,-.55],'Q':[.85,-.55],'C':[-.28,-1.],'O':[-.28,-1.]}
    arm=unit(.37*R[:,1]*-1.+.63*np.array(arms.get(c,[0.,-1.]))+n(0,.12,2))
    return bones,w,h,fat,R,arm,turn,curl,tmode


def drawbatch(chars,rng):
    m=len(chars); ds=[diagram(c,rng) for c in chars]; u=rng.uniform
    bones=np.array([d[0] for d in ds],np.float32); w,h,fat=[np.array([d[j] for d in ds],np.float32) for j in (1,2,3)]
    R=np.array([d[4] for d in ds],np.float32); arm=np.array([d[5] for d in ds],np.float32); side=56
    end=bones.reshape(m,-1,2)@R.transpose(0,2,1)
    hull=np.stack([np.column_stack([s*w,yy*h]) for s,yy in ((-.49,0),(.49,0),(-.8,1.35),(.8,1.35))],axis=1)@R.transpose(0,2,1)
    lo=np.minimum(end.min(1)-.19,hull.min(1)); hi=np.maximum(end.max(1)+.19,hull.max(1))
    box=(hi-lo).max(1)*u(1.,1.40,m); center=(hi+lo)*.5+box[:,None]*np.column_stack([rng.normal(0,.064,m),rng.normal(-.048,.056,m)])
    axis=((np.arange(side)+.5)/side-.5).astype(np.float32)
    def D(a): return np.asarray(a,np.float32).reshape(-1,1,1)
    X=D(center[:,0])+D(box)*axis[None,None,:]; Y=D(center[:,1])-D(box)*axis[None,:,None]
    x=D(R[:,0,0])*X+D(R[:,1,0])*Y; y=D(R[:,0,1])*X+D(R[:,1,1])*Y
    lx=D(u(-.9,.65,m)); ly=D(u(.35,.95,m)); lz=D(u(.45,1.05,m)); ll=(lx*lx+ly*ly+lz*lz)**-.5
    lx,ly,lz=lx*ll,ly*ll,lz*ll
    zbuf=np.full((m,side,side),-12.,np.float32)
    bg=D(u(.41,.87,m))+D(u(-.13,.13,m))*axis[None,:,None]+D(u(-.13,.13,m))*axis[None,None,:]
    im=np.broadcast_to(bg,zbuf.shape).copy(); tone=D(u(-.12,.12,m))
    def cover(mask,z,nx,ny,nz,tint):
        sel=mask & (z>zbuf); light=nx*lx+ny*ly+nz*lz
        np.copyto(zbuf,z,where=sel)
        np.copyto(im,.35+tone+.32*light+.12*nz+tint,where=sel)
    def tube(a,b,ra,rb,za,zb,contrast):
        vx=D(b[:,0]-a[:,0]); vy=D(b[:,1]-a[:,1]); px=x-D(a[:,0]); py=y-D(a[:,1]); ll=vx*vx+vy*vy+.00001
        t=np.clip((px*vx+py*vy)/ll,0,1); rd=(1-t)*D(ra)+t*D(rb); nx=(px-vx*t)/rd; ny=(py-vy*t)/rd
        nz2=1.-nx*nx-ny*ny; nz=np.sqrt(np.maximum(0,nz2)); z=(1-t)*D(za)+t*D(zb)+nz*rd
        cover(nz2>0,z,nx,ny,nz,-contrast*np.maximum(0.,1.-6*np.minimum(t,1.-t)))
    start=np.column_stack([w*0.,h*.25]); end=start+6*np.einsum('ni,nij->nj',arm,R)
    tube(start,end,.48*w,.78*w,-.15,-.20,.0)
    # Anatomical sections describe tapered heel, palm arch, and knuckle line.
    Yh=y/D(h); sections=[(-.14,.45),(.14,.54),(.67,.76),(1.27,.82),(1.57,.79),(1.75,.59),(1.82,.22)]
    width=np.interp(Yh,[p[0] for p in sections],[p[1] for p in sections])*D(w)
    nx=x/(width+.0001); ny=np.clip((Yh-.87)*.40,-.5,.5); nz=np.sqrt(np.maximum(0,1.-nx*nx))
    cover((np.abs(nx)<1.)&(Yh>-.14)&(Yh<1.82),nz*.32,nx,ny,nz,.0)
    # Soft thenar eminence: fill only the visible pad, followed by its thumb.
    dx=(x/D(w)-.39)-.32*(Yh-.78); dy=(Yh-.78)/.65; dx=dx/.32; d=1.-dx*dx-dy*dy; nz=np.sqrt(np.maximum(0,d))
    cover(d>0,.14+.27*nz,.48*dx,.48*dy,nz,.015)
    rad=[.216,.226,.210,.178,.237]
    for i in range(5):
        if i<4:
            lev=np.array([([.10,.12,.15,.17] if d[7][i] in ('erect','oblique','arch') else [.14,.25,.44,.57]) for d in ds]);
            # Side views occlude the ulnar fingers; crossed R has index in front.
            lev+=np.array([.095*d[6]*(1.5-i) for d in ds])[:,None]
            rs=[1.,.91,.82,.71]
        else:
            lev=np.array([([.14,.29,.37,.46] if d[8].startswith('under') else [.19,.38,.61,.73]) for d in ds]); rs=[1.37,1.07,.89,.77]
        for j in range(3): tube(bones[:,i,j],bones[:,i,j+1],rad[i]*fat*rs[j],rad[i]*fat*rs[j+1],lev[:,j],lev[:,j+1],.036)
    adj=np.pad(zbuf,((0,0),(2,2),(2,2)),mode='edge'); shadow=0.
    for iy,ix in ((2,0),(2,4),(0,2),(4,2)):
        shadow=shadow+np.clip((adj[:,iy:iy+side,ix:ix+side]-zbuf-.095)*2.,0.,1.)
    im-=.048*shadow*(zbuf>-8)
    return np.clip(im,0,1).reshape(m,28,2,28,2).mean((2,4))


def main():
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(features+inference)
    shared={}; exec(features,shared)
    rng=np.random.default_rng(87439)
    # Each row stored below is derived from one independent draw only.
    k=1536; m=16
    coarse=np.empty((24*k,392),np.int16); fine=np.empty((24*k,14,14,8),np.int8)
    for ci,c in enumerate(letters):
        for j in range(0,k,m):
            ims=(render if j<k//2 else drawbatch)([c]*m,rng); f=shared['collect'](ims)
            sl=slice(ci*k+j,ci*k+j+m)
            fine[sl]=np.clip(np.rint(f*50.),-127,127).astype(np.int8)
            coarse[sl]=np.rint(shared['summary'](f)*32767.).astype(np.int16)
    np.savez(out/'constants.npz',coarse=coarse,fine=fine)

if __name__=='__main__': main()
