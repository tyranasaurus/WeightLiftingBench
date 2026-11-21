"""Construct an anatomical template matcher. All lengths in pose() are in
model space (roughly cm); the model is not specified in image coordinates.
Positive x is thumbward, positive y distal and positive z palmar.
"""
import sys, json, math
from pathlib import Path
import numpy as np

ABC = 'ABCDEFGHIKLMNOPQRSTUVWXY'
PI = np.pi
# Number of pose/camera samplings per sign. Same prior sample sequence for all signs.
COUNT = 2560
BATCH = 16
SS = 2
S = 28 * SS
F = np.float32


def pose(letter, rng):
    u=rng.uniform
    # Carpal end at (0,0); knuckle heights follow the metacarpal arch.
    roots=np.array([[2.45,7.35,0],[.86,7.65,0],[-.82,7.28,-.1],[-2.35,6.5,-.25]])
    bones=np.array([[3.55,2.4,1.72],[3.90,2.63,1.88],[3.62,2.42,1.70],[2.76,1.83,1.36]])
    radii=np.array([.83,.87,.82,.71]) * u(.91,1.10)
    # A folded finger: first phalanx rises and projects toward the camera,
    # second folds downward, and distal phalanx turns back toward palm.
    fold=np.array([53.,141.,199.])
    if letter in 'A': fold=np.array([57.,144.,207.])
    elif letter in 'E': fold=np.array([5.,107.,177.])
    elif letter in 'MNST': fold=np.array([63.,141.,204.])
    elif letter in 'GH': fold=np.array([72.,150.,202.])
    flex=np.tile(fold,(4,1)); spread=np.array([2.,0.,-1.,-4.]); ext=[]
    if letter=='B': ext=[0,1,2,3]; spread[:]=[1,0,-1,-2]
    elif letter in 'DGLX': ext=[0]
    elif letter in 'F': ext=[1,2,3]; spread[:]=[0,10,-2,-17]
    elif letter in 'HUVRKP': ext=[0,1]
    elif letter in 'IY': ext=[3]; spread[3]=-12 if letter=='Y' else -7
    elif letter=='W': ext=[0,1,2]; spread[:]=[19,0,-20,0]
    elif letter=='Q': ext=[0]
    for i in ext: flex[i]=[0,1,4]
    if letter in 'MN':
        # Fingers over the thumb have to accommodate its thickness.
        n_over=3 if letter=='M' else 2
        flex[:n_over,:]=[46.,117.,170.]; flex[n_over:,:]=[78.,162.,219.]
        spread[:n_over]-=4
    elif letter=='T':
        flex[0]=[41.,129.,176.]; flex[1:]=[63.,143.,195.]
    elif letter=='Y':
        # The thumb and little finger fan out in opposite directions.
        spread[3]=u(-42,-18);flex[3]=[8,7,8]
    elif letter in 'V': spread[:2]=[16,-13]
    elif letter in 'R': spread[:2]=[-10,5]; flex[0]=[10,-3,0]
    elif letter in 'U': spread[:2]=[-1,-2]
    elif letter in 'KP':
        spread[:2]=[3,-9]; flex[1]=[39,54,58]; flex[0]=[-4,0,3]
    elif letter=='X': flex[0]=[-6,54,133]; spread[0]=-5
    elif letter=='D': flex[1:]=[26,87,147]
    elif letter=='F': flex[0]=[30,88,153]
    elif letter in 'CO':
        flex[:]=[15,61,102] if letter=='C' else [32,89,142]
        # Smaller fingers follow the curve of the larger fingers.
        flex += np.arange(4)[:,None]*1.4
    elif letter in 'GQ':
        # Pointing is chiefly a movement at the MCP joint, not a sweep
        # through 90 degrees of wrist deviation. Keep the finger itself straight.
        flex[0]=u(57.,81.) + np.array([-2,0,2]); spread[0]=3
    elif letter=='H':
        spread[:2]=[1,-1];flex[:2]=u(51.,77.)+np.array([[-2,0,3],[0,2,4]])
    # Independent within-pose variation in hand proportions and joint angles.
    stretch=u(.92,1.08); width=u(.91,1.09); roots[:,0]*=width
    roots[:,1]*=stretch; bones*=u(.94,1.06,(4,1))*u(.93,1.07)
    # A folded hand can be signed loosely, or in a tight curl. Change
    # flexion jointly to preserve the alignment of its knuckles.
    if letter in 'MNE':
        bend=u(-8.,31.); flex+=np.array([bend,.57*bend,.35*bend])
    if letter=='Y':
        # Abduction at both the little finger and thumb widens the hand.
        spread[3]+=u(-13.,4.);flex[:3]+=np.array([-10.,-8.,-5.])
    flex+=u(-7,7,(4,1))+u(-5,5,(4,3)); spread+=u(-2.3,2.3,4)
    spread=np.deg2rad(spread); flex=np.deg2rad(flex)
    points=[]; joint_r=[]
    for i in range(4):
        d=np.stack([np.cos(flex[i])*np.sin(spread[i]),np.cos(flex[i])*np.cos(spread[i]),np.sin(flex[i])],axis=1)
        points.append(np.vstack([roots[i],roots[i]+np.cumsum(d*bones[i,:,None],axis=0)]))
        joint_r.append(radii[i]*np.array([1.,.95,.86,.78]))
    # Thumb originates lower on the radial side. Its positions are defined by
    # their articulatory relation to the other fingers: tucked, opposed, etc.
    base=np.array([2.10*width,2.30*stretch,.35]); end=lambda i: points[i][-1]
    if letter=='A':
        th=[[2.85,3.9,1.0],[3.60,6.0,1.55],[3.65,8.2,1.9],[3.40,9.9,2.25]]
    elif letter in 'BISUVWXR':
        th=[[2.8,3.5,1.0],[2.4,4.9,2.05],[.85,5.9,3.05],[-1.15,6.5,3.5]]
        if letter=='S':
            # The thumb crosses in *front* of the middle phalanges, unlike A.
            reach=u(.70,1.1); th=[[3.15,4.3,1.4],[3.2,5.65,3.4],[1.75,7.,5.2],[1.75-2.65*reach,8.6,5.8]]
        if letter=='W': th[-1]=[-2.05,6.25,2.95]
    elif letter=='E':
        th=[[2.6,3.5,1.2],[2.45,5.15,2.45],[.55,5.65,2.95],[-1.60,5.8,2.90]]
    elif letter in 'MNT':
        # Thumb emerges between different pairs of curled fingers.
        gap={'M':-1.85,'N':0.,'T':1.70}[letter]*width
        th=[[2.65,4.,1.15],[1.75+.25*gap,5.75,1.8],[gap+.28,7.15,2.5],[gap,8.65,3.0]]
        if letter=='T' and rng.random()<.75:
            # Side-on version of T: thumb stands up, index folds across it.
            th=[[2.75,4.0,1.3],[2.6,5.8,2.2],[2.35,7.8,2.2],[2.20,10.0,2.2]]
            p=points[0][0];points[0]=np.array([p,p+[0,1.5,3.1],p+[-.2,1.25,5.3],p+[-.38,.6,6.75]])
    elif letter in 'LY':
        th=[[3.0,3.25,.75],[4.95,4.1,1.05],[6.85,4.95,1.25],[8.5,5.60,1.4]]
    elif letter in 'DFO':
        goal=end(0 if letter=='F' else 1)+[.22,-.55,0]
        if letter=='O': goal=end(0)+[0,-.45,0]
        th=[[2.95,3.8,1.25],[3.60,5.,2.7],.50*goal+.50*np.array([3.9,5.,3.7]),goal]
    elif letter=='C':
        th=[[2.95,3.4,1.1],[3.55,4.55,2.80],[3.70,5.40,4.80],[3.40,6.55,6.5]]
    elif letter in 'GQ':
        if letter=='G':
            th=[[3.15,3.5,.75],[3.95,5.5,1.2],[4.0,7.35,2.1],[3.85,8.8,3.6]]
        else:
            th=[[3.15,3.5,.8],[4.15,4.5,2.3],[5.0,5.4,4.1],[5.7,6.1,5.6]]
    elif letter in 'KP':
        goal=.25*points[1][2]+.75*points[1][1]+[.4,-.3,.5]
        th=[[2.9,4.20,1.2],[2.65,6.2,2.7],.48*goal+.52*np.array([2.65,6.2,2.7]),goal]
    elif letter=='H':
        th=[[2.7,3.8,1.1],[1.95,5.1,2.1],[.1,5.85,2.9],[-1.9,6.05,3.1]]
    else: raise ValueError(letter)
    th=np.vstack([base,np.array(th)]);th[1:]+=u(-.24,.24,(4,3)); points.append(th);joint_r.append(np.array([1.25,1.27,1.06,.92,.80]))
    # Euler angles: yaw shows the palmar surface or a side profile; roll
    # points the fingers along the letter's customary signing direction.
    yaw,tilt,roll= (15.,-5.,0.)
    if letter in 'CO': yaw,tilt,roll=(78.,-3.,14.)
    elif letter=='D': yaw,tilt,roll=(16.,3.,-4.)
    elif letter in 'GH': yaw,tilt,roll=(115.,-8.,25.)
    elif letter=='Q':
        # Palmar normal points down. Roll gives a three-quarter view of the
        # gap between the downward-pointing index and opposed thumb.
        yaw,tilt,roll=(-43.,75.,-9.)
    elif letter=='P': yaw,tilt,roll=(49.,-10.,-73.)
    elif letter=='K': yaw,tilt,roll=(-57.,12.,-6.)
    elif letter=='R': yaw,tilt,roll=(20.,-2.,5.)
    elif letter=='T': yaw,tilt,roll=(64.,3.,1.)
    elif letter in 'MNSAE':
        # Curling over an interposed thumb changes the amount of supination.
        yaw={'M':-25.,'N':-40.,'S':15.,'A':-7.,'E':8.}[letter]
        tilt,roll=(-12.,12.)
        if letter=='E':tilt,roll=(-12.,7.)
        if letter=='M':yaw+=u(0.,47.); roll+=u(-3.,11.)
        if letter=='S':yaw+=u(-29.,8.)
    elif letter=='F': yaw,tilt,roll=(24.,1.,-3.)
    elif letter=='X': yaw,tilt,roll=(75.,-4.,1.)
    elif letter=='L': yaw,tilt,roll=(10.,-4.,-5.)
    elif letter=='Y': yaw,tilt,roll=(1.,-13.,8.)
    if letter in 'GH':
        # Allow both a straight horizontal metacarpus and a flexed wrist.
        tilt+=u(-28,31);roll+=u(-5,25)
    yaw+=u(-28.,28.); tilt+=u(-15.,16.); roll+=u(-14.,15.)
    def rot(a,axis):
        a=np.deg2rad(a);c,s=np.cos(a),np.sin(a)
        if axis==0:return np.array([[1,0,0],[0,c,-s],[0,s,c]])
        if axis==1:return np.array([[c,0,s],[0,1,0],[-s,0,c]])
        return np.array([[c,s,0],[-s,c,0],[0,0,1]])
    rot3=rot(roll,2)@rot(tilt,0)@rot(yaw,1)
    if letter in 'GHQP':
        # At oblique elevations Euler angles couple strongly: specifying the
        # world-space pointing directions avoids unintended foreshortening.
        def unit(a):return a/np.linalg.norm(a)
        def point(angle,depth):
            a=np.deg2rad(angle);return unit(np.array([np.sin(a),np.cos(a),depth]))
        if letter in 'GHQ':
            if letter=='Q': dpoint=point(u(172,210),u(-.40,.30)); dhand=point(u(-35,25),u(.50,1.65))
            else: dpoint=point(u(64,95),u(-.55,.55)); dhand=point(u(-10,48),u(.15,.95))
            # Aim the index (z-ish), and set the roll by projecting the
            # metacarpal axis onto its orthogonal complement.
            def frame(p,q):
                p=unit(p);q=unit(q-np.dot(p,q)*p);return np.stack([p,q,np.cross(p,q)],axis=1)
            tipaxis=points[0][-1]-points[0][0]
            rot3=frame(dpoint,dhand)@frame(tipaxis,np.array([0,1.,0])).T
        elif rng.random()<.6:
            rot3=rot(u(-86,-59),2)@rot(u(-20,12),0)@rot(u(23,80),1)
    # Bevelled soft palm, heel of palm, thenar and hypothenar pads.
    # Ellipsoid list: centre and principal radii. Palm tapers into the wrist.
    ell=[([0,3.9*stretch,-.08],[3.24*width,4.48*stretch,1.29]),
         ([.04,6.0*stretch,-.08],[3.17*width,2.24*stretch,1.16]),
         ([1.38*width,3.18*stretch,.55],[1.65*width,2.95*stretch,1.22]),
         ([-1.60*width,3.22*stretch,.18],[1.43*width,2.92*stretch,1.04])]
    # Connect the four proximal phalanges to their metacarpals.
    bones3=[]
    for i,(path,rr) in enumerate(zip(points,joint_r)):
        if i<4:
            bones3.append((path[0]-[.08*path[0,0],1.85,0],path[0],rr[0]*1.08,rr[0],0))
        for j in range(len(path)-1):bones3.append((path[j],path[j+1],rr[j],rr[j+1],1 if i<4 else 2))
    # In side-pointing positions the wrist bends instead of following the
    # direction of the extended fingers all the way out of the picture.
    wristbend={'G':4,'H':4,'P':22,'Q':56}.get(letter,4)
    return ell,bones3,rot3,wristbend


def sample(rng,letter):
    ell,bones,rot,bend=pose(letter,rng); u=rng.uniform
    # Auto-frame each *model*, not the photographs. Include the palm heel
    # and fingertips but not the whole forearm in the crop.
    lo=[]; hi=[]
    for c,r in ell:
        c=rot@c; v=np.sqrt(((rot*np.asarray(r))**2).sum(1));lo.append(c[:2]-v[:2]);hi.append(c[:2]+v[:2])
    for a,b,ra,rb,_ in bones:
        a,b=rot@a,rot@b;lo.extend([a[:2]-ra,b[:2]-rb]);hi.extend([a[:2]+ra,b[:2]+rb])
    lo=np.min(lo,axis=0);hi=np.max(hi,axis=0);ctr=(lo+hi)*.5
    # Framing can cut the palm heel off, but rarely follows the entire arm.
    # For a broad transverse hand, allow the wrist into the square frame.
    observed_span=max(hi-lo)
    size=observed_span*u(.86,1.35);ctr+=size*np.array([u(-.12,.12),u(-.16,.06)])
    pixel=size/S
    axis=(np.arange(S,dtype=F)+.5-S/2)*pixel
    X=(ctr[0]+axis[None,:]).astype(F); Y=(ctr[1]-axis[:,None]).astype(F)
    depth=np.full((S,S),-100.,dtype=F); bright=np.zeros((S,S),dtype=F)
    # Key light, soft fill and skin/background albedos varied independently.
    light=np.array([u(-.96,.3),u(.25,1.),u(.25,1.)]);light/=np.linalg.norm(light)
    bg=u(.48,.84); ambient=u(.15,.34); diffuse=u(.45,.80)
    if rng.random()<.15: bg=u(.13,.45)
    def layer(z,inside,nx,ny,nz,albedo=1):
        mask=inside & (z>depth)
        strength=nx*light[0]+ny*light[1]+nz*light[2];shade=(ambient+diffuse*np.maximum(0,strength))*albedo
        np.copyto(bright,shade,where=mask);np.copyto(depth,z,where=mask)
    def ellipsoid(c,r):
        c=rot@c; m=rot@np.diag(np.asarray(r)**-2)@rot.T
        x=X-c[0];y=Y-c[1]; b=m[0,2]*x+m[1,2]*y
        delta=b*b-m[2,2]*(m[0,0]*x*x+2*m[0,1]*x*y+m[1,1]*y*y-1)
        z=(-b+np.sqrt(np.maximum(delta,0)))/m[2,2]
        nx=m[0,0]*x+m[0,1]*y+m[0,2]*z;ny=m[1,0]*x+m[1,1]*y+m[1,2]*z;nz=b+m[2,2]*z
        norm=np.sqrt(nx*nx+ny*ny+nz*nz+1e-12)
        layer(z+c[2],delta>=0,nx/norm,ny/norm,nz/norm)
    def tube(a,b,ra,rb,albedo=1.):
        # Rounded tapered stick. Compute the front depth of its envelope of
        # spheres, incl. foreshortening when a finger points at the viewer.
        d=b-a;x=X-a[0];y=Y-a[1];xy2=d[0]**2+d[1]**2+1e-12
        t0=(x*d[0]+y*d[1])/xy2
        rad=ra+(rb-ra)*np.clip(t0,0,1)
        perp=x*d[1]-y*d[0];sq=np.maximum(0,(rad*rad-perp*perp/xy2)/(xy2+d[2]**2))
        t=np.clip(t0+d[2]*np.sqrt(sq/xy2),0.,1.)
        # Correct sphere-envelope optimum: t0 + dz * sqrt((r²-perp²)/(xy²*|d|²)).
        nx=x-t*d[0];ny=y-t*d[1];rad=ra+(rb-ra)*t
        nz2=rad*rad-nx*nx-ny*ny;nz=np.sqrt(np.maximum(0,nz2));z=a[2]+d[2]*t+nz
        layer(z,nz2>=0,nx/rad,ny/rad,nz/rad,albedo)
    # Include a forearm going out of frame, in a natural range of wrist flexion.
    wrist=rot@np.array([-.12,.15,-.2]); direction=rot@np.array([0.,-1.,-.16]);
    a=np.deg2rad(bend+u(-15,15));cs,sn=np.cos(a),np.sin(a);dx,dy,dz=direction
    direction=np.array([dx*cs+dy*sn,-dx*sn+dy*cs,dz])
    if letter in 'GHPQ':
        # Letter orientation is mostly changed by movements at the wrist.
        # The supporting forearm still descends toward the signer's body.
        amin,amax={'G':(-65,0),'H':(-75,-5),'P':(-10,74),'Q':(-25,40)}[letter]
        a=np.deg2rad(u(amin,amax)); direction=np.array([np.sin(a),-np.cos(a),-u(.2,.9)])
    elbow=wrist+(size*1.6)*direction
    tube(wrist,elbow,2.12,3.22)
    for c,r in ell:ellipsoid(c,r)
    for a,b,ra,rb,kind in bones: tube(rot@a,rot@b,ra,rb)
    # A shallow contact shadow at a depth discontinuity, on the surface
    # behind it. Also marks the overlap of fingers in front of the palm.
    pad=np.pad(depth,1,mode='edge'); occ=np.maximum.reduce([pad[:-2,1:-1],pad[1:-1,:-2],pad[2:,1:-1],pad[1:-1,2:]])-depth
    mask=depth>-90
    bright*=1.-.26*np.clip(occ/(pixel*3),0.,1.)
    # Cast soft self-shadows by walking the depth buffer toward the light.
    # In particular, bent fingers shadow the palm behind them; light cannot
    # pass through the thumb where it crosses the fist.
    drops=[];ll=max(.16,np.hypot(*light[:2]));padn=8;pd=np.pad(depth,padn,constant_values=-100.)
    for t in [2,4,7]:
        dx=int(round(t*light[0]/ll));dy=-int(round(t*light[1]/ll)); ahead=pd[padn+dy:padn+dy+S,padn+dx:padn+dx+S]
        drops.append(np.clip((ahead-depth-pixel*(t*light[2]/ll+1.))/(pixel*3+.24),0,1))
    bright*=1.-.36*np.maximum.reduce(drops)
    img=np.where(mask,bright,bg).clip(0,1).reshape(28,SS,28,SS).mean((1,3)).astype(F)
    # Independent per-view support, not a crop from an external image. Do
    # not expect all of the empty part of the frame to be a smooth wall.
    support=mask.reshape(28,SS,28,SS).mean((1,3)).astype(F)
    return img,support


# Analytic filter bank. Channels encode oriented edge energy (unsigned),
# a smaller signed term, and regional shadow/highlight relative to the image.
def filters():
    y,x=np.mgrid[-2:3,-2:3];g=np.exp(-(x*x+y*y)/(2*.92**2));fs=[]
    for i in range(4):
        a=i*PI/4;k=(x*np.cos(a)+y*np.sin(a))*g;k-=k.mean();k/=np.abs(k).sum()/2
        fs.extend([k,-k])
    g=g/g.sum();fs.extend([g,-g]);return np.array(fs,dtype=F)

K=filters(); NFEAT=10
# Four axes of edge energy, four directed edge components, two region components.
MIX=np.zeros((NFEAT,10),dtype=F)
for i in range(4):
    MIX[i,2*i:2*i+2]=1.; MIX[i+4,2*i:2*i+2]=[1.,-1.]
MIX[-2,-2]=1;MIX[-1,-1]=1

def norm(x,axis):return (x-x.mean(axis=axis,keepdims=True))/np.sqrt(x.var(axis=axis,keepdims=True)+1e-5)

# Sharpen the orientation bandwidth before integrating over space. Each
# orientation competes with its orthogonal response at the same location.
SELECT=np.eye(10,dtype=F)
for i in range(8):
    j=(i//2+2)%4; SELECT[i,j*2:j*2+2]=-.60
KNEE=np.array([.60]*8+[1.25]*2,dtype=F)
CAP=np.array([1.75]*8+[2.2]*2,dtype=F)

def descriptor(img):
    # Independent calculations for each individual render. Batch axis is
    # never reduced. Operations here correspond to the feature layers below.
    img=norm(img,(1,2));patch=np.lib.stride_tricks.sliding_window_view(img,(5,5),axis=(1,2))
    raw=(patch.reshape(len(img),24*24,25)@K.reshape(10,-1).T).transpose(0,2,1).reshape(-1,10,24,24)
    raw=np.maximum(raw,0)
    raw=(SELECT@raw.reshape(len(img),10,-1)).reshape(-1,10,24,24)
    raw=np.maximum(raw,0.)
    # Piecewise linear contrast compression preserves fine finger valleys.
    raw=raw-.72*np.maximum(raw-KNEE[None,:,None,None],0)-.28*np.maximum(raw-CAP[None,:,None,None],0)
    # Overlapping spatial cells, centered on a 2-pixel grid.
    raw=np.lib.stride_tricks.sliding_window_view(raw,(4,4),axis=(2,3))[:,:,::2,::2].mean((-1,-2))
    desc=(MIX@raw.reshape(len(img),10,-1)).reshape(-1,10,11,11)
    return desc

GROUPS=[(0,4,.51),(4,8,.30),(8,10,.19)]

# A pair of broad, overlapping spatial parts. Whole-view matching and parts
# share each rendered view, but the two parts can independently choose the
# best view at test time. This exploits articulation without a learned readout.
PARTS=[('whole',slice(0,11)),('upper',slice(0,6)),('lower',slice(5,11))]

def support_cells(mask):
    win=np.lib.stride_tricks.sliding_window_view(mask,(8,8),axis=(1,2))[:,::2,::2]
    return .25+.75*win.max(axis=(-1,-2))


def network(outdir):
    layers=[];w={}
    def put(name,op,i,**kw):
        layers.append(dict(name=name,op=op,inputs=([i] if isinstance(i,str) else i),**kw));return name
    def conv(name,i,K,b=None):
        o,c,h,t=K.shape;w[name+'.W']=K.astype(F);w[name+'.b']=np.zeros(o,F) if b is None else np.asarray(b,dtype=F)
        return put(name,'conv',i,out=o,kernel=[h,t],stride=1,padding=0)
    # Clip positive filter responses with a single expanded filter bank and
    # a linear combination after ReLU: min(max(x,0),h)=relu(x)-relu(x-h).
    g=put('contrast','norm','image');g=conv('derivatives',g,K[:,None]);g=put('rectify','relu',g)
    g=conv('angular_tuning',g,np.tile(SELECT,(3,1))[:,:,None,None],np.concatenate([np.zeros(10),-KNEE,-CAP]));g=put('compress','relu',g)
    g=put('cells','avgpool',g,size=[4,4],stride=[2,2]);g=conv('orientations',g,np.concatenate([MIX,-.72*MIX,-.28*MIX],axis=1)[:,:,None,None]);
    # Flatten each *family* into the spatial dimensions of a single channel;
    # norm computes NCC, not a stored background estimate.
    def dense(name,v,mat):
        w[name+'.b']=np.zeros(len(mat),F);w[name+'.W']=mat;return put(name,'dense',v,out=len(mat))
    g_in=g;outputs=[]
    for part,rs in PARTS:
        h=rs.stop-rs.start; area=11*h
        step=1 if part=='whole' else 2; count=COUNT//step;rows=24*count
        channels=[]
        for a,b,weight in GROUPS:
            key=part+'_family'+str(a);sel=np.eye(10,dtype=F)[a:b,:,None,None];v=conv(key,g_in,sel)
            if step!=1:
                indices=np.arange((b-a)*121).reshape(b-a,11,11)[:,rs].ravel()
                mask=np.eye((b-a)*121,dtype=F)[indices];v=dense(key+'_crop',v,mask)
            v=put(key+'_vector','reshape',v,shape=[1,b-a,area]);v=put(key+'_normalized','norm',v)
            v=put(key+'_map','reshape',v,shape=[b-a,h,11]);v=conv(key+'_place',v,sel.transpose(1,0,2,3));channels.append(v)
        v=put(part+'_descriptor','add',channels)
        v=dense(part+'_match',v,np.empty((rows,10*area),F))
        v=put(part+'_views','reshape',v,shape=[24,1,count])
        v=put(part+'_search','maxpool',v,size=[1,count//4],stride=[1,count//4]);v=put(part+'_consensus','avgpool',v,size=[1,4],stride=[1,4]);outputs.append(v)
    for ci,letter in enumerate(ABC):
        rng=np.random.default_rng(17031)
        for start in range(0,COUNT,BATCH):
            n=min(COUNT-start,BATCH); samples=[sample(rng,letter) for _ in range(n)]
            x,m=map(np.stack,zip(*samples));desc=descriptor(x);supp=support_cells(m)
            for part,rs in PARTS:
                area=(rs.stop-rs.start)*11; step=1 if part=='whole' else 2
                startrow=(ci*COUNT+start)//step
                dest=w[part+'_match.W'][startrow:startrow+n//step].reshape(-1,10,area); mask=supp[::step,rs].reshape(-1,1,area)
                for a,b,gain in GROUPS:
                    # Weighted mean depends only on this view's own apparent
                    # hand, with a halo for silhouette edges and weaker wall evidence.
                    vec=desc[::step,a:b,rs].reshape(-1,b-a,area)
                    mu=(vec*mask).sum((1,2),keepdims=True)/(mask.sum(2,keepdims=True)*(b-a))
                    vec=(vec-mu)*mask
                    vec/=np.sqrt(np.mean(vec*vec,axis=(1,2),keepdims=True)+1e-5)
                    dest[:,a:b]=vec*(gain/(area*(b-a)))
    v=put('evidence','add',[outputs[0],outputs[0],outputs[0],outputs[0],outputs[1],outputs[2]]);put('letters','reshape',v,shape=[24]);
    outdir.mkdir(parents=True,exist_ok=True);np.savez(outdir/'weights.npz',**w);(outdir/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')

if __name__=='__main__': network(Path(sys.argv[1]))
