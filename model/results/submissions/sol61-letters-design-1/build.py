"""Data-free ASL network: anatomical pose generators and fixed edge matching.

No data files are opened.  All model geometry is expressed in palm-width units.
The skeleton uses elementary capsule/ellipsoid geometry, not recorded images.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'
SEED = 17319
PER_CLASS = 4800
GRID = 14


def rotation(yaw, pitch, roll):
    a, b, c = np.deg2rad([yaw, pitch, roll])
    Y = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    X = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    Z = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Z @ X @ Y


# Index, middle, ring, little finger. Flexion is toward the palm/camera.
# MCP / PIP / DIP angles and abduction are joint angles in degrees.
OPEN = [2, 3, 2]
CLOSED = [88, 100, 40]
GESTURES = {
 'A': ([CLOSED]*4, [0,0,0,0], 'upright', -12, 0),
 'B': ([OPEN]*4, [-1,0,1,2], 'across', 0, 0),
 'C': ([[25,45,30]]*4, [0,0,0,0], 'c', 68, 0),
 'D': ([OPEN,CLOSED,CLOSED,CLOSED], [0,0,0,0], 'circle', -18, 0),
 'E': ([[88,100,65]]*4, [0,0,0,0], 'lowacross', 0, 0),
 'F': ([[65,70,20],OPEN,OPEN,OPEN], [0,2,10,23], 'circle', 0, 0),
 'G': ([OPEN,CLOSED,CLOSED,CLOSED], [0,0,0,0], 'parallel', 70, -85),
 'H': ([OPEN,OPEN,CLOSED,CLOSED], [0,0,0,0], 'across', 60, -80),
 'I': ([CLOSED,CLOSED,CLOSED,OPEN], [0,0,0,10], 'across', 0, 0),
 'K': ([OPEN,[30,3,2],CLOSED,CLOSED], [0,0,0,0], 'between', 57, 0),
 'L': ([OPEN,CLOSED,CLOSED,CLOSED], [0,0,0,0], 'out', 0, 0),
 'M': ([[82,100,45]]*4, [0,0,0,0], 'underthree', 0, 0),
 'N': ([[82,100,45]]*4, [0,0,0,0], 'undertwo', 0, 0),
 'O': ([[40,65,45]]*4, [0,0,0,0], 'o', 65, 0),
 'P': ([OPEN,[45,3,2],CLOSED,CLOSED], [0,0,0,0], 'between', 57, 100),
 'Q': ([OPEN,CLOSED,CLOSED,CLOSED], [0,0,0,0], 'parallel', 70, 180),
 'R': ([[0,5,0],[0,5,0],CLOSED,CLOSED], [12,-10,0,0], 'across', 0, 0),
 'S': ([CLOSED]*4, [0,0,0,0], 'across', 0, 0),
 'T': ([CLOSED]*4, [0,0,0,0], 'betweenclosed', 0, 0),
 'U': ([OPEN,OPEN,CLOSED,CLOSED], [0,0,0,0], 'across', 0, 0),
 'V': ([OPEN,OPEN,CLOSED,CLOSED], [-18,15,0,0], 'across', 0, 0),
 'W': ([OPEN,OPEN,OPEN,CLOSED], [-19,0,19,0], 'across', 0, 0),
 'X': ([[15,100,25],CLOSED,CLOSED,CLOSED], [0,0,0,0], 'across', -25, 0),
 'Y': ([CLOSED,CLOSED,CLOSED,OPEN], [0,0,0,24], 'out', 0, 0),
}


def geometry(letter, rng):
    flexes, spreads, thumb, yaw, roll = GESTURES[letter]
    flexes=[list(v) for v in flexes]
    spreads=list(spreads)
    # Semantic alternatives cover conventional articulation, not particular
    # signers. The ASL T uses an opposed thumb between the index and middle;
    # its apparent thumb direction changes strongly with forearm pronation.
    if letter=='T' and rng.random()<.65:
        yaw=70;roll=-75
    if letter=='K' and rng.random()<.50:
        flexes[1]=[rng.uniform(20,55),3,2]
        spreads[0]=-rng.uniform(4,14)
        spreads[1]=rng.uniform(4,14)
        yaw=rng.uniform(-15,35)
    if letter in 'GH' and rng.random()<.35:
        yaw=rng.uniform(25,65)
    if letter in 'PQ' and rng.random()<.35:
        yaw=-yaw
    # These ranges represent ordinary anatomical and camera variation.
    hand_w = rng.uniform(.86, 1.14)
    hand_l = rng.uniform(.88, 1.12)
    finger_l = rng.uniform(.85, 1.12)
    thickness = rng.uniform(.90, 1.20)
    parts = []
    # (centre, radii, rotation) ellipsoids; capsules are added separately.
    ellipses = [(np.array([0., -.12, -.06]), np.array([.76*hand_w, .91*hand_l, .30]), np.eye(3)),
                (np.array([.38,-.25,.17]), np.array([.34*hand_w,.55*hand_l,.25]), np.eye(3))]
    segments = []
    # Wrist enters from the heel of the palm; its distant end is cropped.
    segments.append((np.array([.0,-.65,-.08]),np.array([.04,-2.1,-.12]),.43*hand_w))
    xs = [.53, .18, -.18, -.50]
    ys = [.57, .70, .65, .46]
    lengths = [1.67, 1.84, 1.70, 1.31]
    radii = [.185,.193,.182,.158]
    tips = []
    for i in range(4):
        pos=np.array([xs[i]*hand_w, ys[i]*hand_l, .0])
        angles=np.array(flexes[i],dtype=float)
        # Independent modest articulation variation, chosen without examples.
        if angles[0]>50:
            angles += rng.uniform(-13,13,3)
        else:
            angles += rng.uniform(-4,4,3)
        angles=np.maximum(angles,0)
        spread=spreads[i]+rng.uniform(-2,2)
        if letter in 'BU' and (letter=='B' or i<2):
            # Adducted fingers can meet along their sides without necessarily
            # occluding every inter-finger groove.
            spread += (i-1.5)*rng.uniform(-.5,1.5)
        direction=np.deg2rad(spread)
        flex=0.
        joints=[pos.copy()]
        for j, frac in enumerate([.47,.31,.22]):
            flex+=np.deg2rad(angles[j])
            length=lengths[i]*finger_l*frac
            d=np.array([-np.sin(direction)*np.cos(flex),np.cos(direction)*np.cos(flex),np.sin(flex)])
            end=pos+length*d
            r=radii[i]*thickness*(1-.10*j)
            segments.append((pos.copy(),end.copy(),r))
            joints.append(end.copy())
            pos=end
        tips.append(joints[-1])
    # The thumb has a separate opposable metacarpal. Its paths are anatomical
    # semantic poses, not traced image contours.
    root=np.array([.55,-.40,.06])
    paths={
      'upright': [[.83,-.10,.26],[.85,.48,.30],[.80,.87,.26]],
      'across': [[.66,-.02,.72],[.21,.27,1.02],[-.23,.37,1.02]],
      'lowacross': [[.68,-.22,.57],[.23,-.05,.92],[-.25,-.06,.93]],
      'underthree': [[.64,-.03,.37],[.05,.37,.53],[-.40,.65,.55]],
      'undertwo': [[.65,-.02,.39],[.24,.41,.55],[-.04,.72,.55]],
      'betweenclosed': [[.68,-.04,.62],[.45,.45,.96],[.37,1.05,1.10]],
      'between': [[.65,.00,.45],[.39,.62,.56],[.33,1.07,.48]],
      'out': [[1.02,-.05,.10],[1.48,.25,.10],[1.78,.41,.08]],
      'parallel': [[.75,-.05,.36],[.82,.55,.52],[.77,1.17,.48]],
      'c': [[.82,-.18,.26],[.88,-.02,.82],[.81,.24,1.17]],
      'o': [[.82,-.06,.30],[.80,.20,.95],[.69,.52,1.35]],
      'circle': [[.75,-.04,.40],[.62,.43,.83],[.52,.76,1.00]],
    }
    path=np.array(paths[thumb],float)
    path[:,:2]*=[hand_w,hand_l]
    # Thumb-to-finger contact is a kinematic constraint of these letters.
    # It is enforced in 3D rather than by a drawing of the projected circle.
    if letter=='F':
        path[-1]=tips[0]+np.array([.04,-.04,-.04])
        path[-2]=.5*(path[-1]+path[0])+np.array([.20,-.15,.12])
    elif letter=='D':
        path[-1]=tips[1]+np.array([.08,-.04,-.04])
        path[-2]=.5*(path[-1]+path[0])+np.array([.18,-.15,.10])
    elif letter=='O':
        path[-1]=tips[0]+np.array([.04,-.04,-.04])
        path[-2]=.5*(path[-1]+path[0])+np.array([.20,-.15,.10])
    path += rng.uniform(-.035,.035,path.shape)
    for j,end in enumerate(path):
        segments.append((root.copy(),end.copy(),(.24-.035*j)*thickness))
        root=end
    if letter in 'EFX':
        yaw += rng.uniform(-48,48)
    else:
        yaw += rng.uniform(-26,26)
    pitch=rng.uniform(-20,20)
    roll += rng.uniform(-20,20)
    # Front/back variations are small; C/O and the horizontal/downward letters
    # retain their conventional lateral orientation.
    if letter in 'ADIKLNRSTUVWX':
        yaw += rng.choice([-1,1])*rng.uniform(0,10)
    R=rotation(yaw,pitch,roll)
    ellipses=[(R@c,rad,R@r) for c,rad,r in ellipses]
    segments=[(R@a,R@b,r) for a,b,r in segments]
    # The forearm is not rigidly rotated along with the palm: a signer bends
    # their wrist for horizontal and downward pointing signs.
    wrist=segments[0][0]
    side=0.
    if letter in 'GH':side=-.85
    if letter in 'PQ':side=.70
    segments[0]=(wrist,wrist+np.array([side,-1.8,-.1]),.43*hand_w)
    return ellipses,segments


def render(letter,rng,n=56):
    ellipses,segs=geometry(letter,rng)
    # Geometric auto-framing uses only the generated model, not any real image.
    # Frame above wrist: the wrist is intentionally permitted to cross the edge.
    bounds=[]
    for a,b,r in segs[1:]:
        bounds.extend([a[:2]-r,a[:2]+r,b[:2]-r,b[:2]+r])
    for c,rad,R in ellipses:
        ext=np.sqrt((R[:2]**2)@(rad**2))
        bounds.extend([c[:2]-ext,c[:2]+ext])
    bounds=np.array(bounds)
    lo,hi=bounds.min(0),bounds.max(0)
    centre=(lo+hi)/2
    extent=max(hi-lo)*rng.uniform(1.06,1.46)
    centre+=rng.uniform(-.10,.10,2)*extent
    yy,xx=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    X=(xx+.5-n/2)*extent/n+centre[0]
    Y=-(yy+.5-n/2)*extent/n+centre[1]
    Z=np.full((n,n),-100.,float)
    norm=np.zeros((n,n,3),float)
    # Ray / ellipsoid intersection, camera looking along -z.
    def put_ellipsoid(c,rad,R):
        M=R@np.diag(1/rad**2)@R.T
        dx=X-c[0];dy=Y-c[1]
        aa=M[2,2]
        bb=2*(M[0,2]*dx+M[1,2]*dy)
        cc=M[0,0]*dx**2+2*M[0,1]*dx*dy+M[1,1]*dy**2-1
        disc=bb**2-4*aa*cc
        zz=c[2]+(-bb+np.sqrt(np.maximum(disc,0)))/(2*aa)
        valid=(disc>=0)&(zz>Z)
        q=np.stack([dx,dy,zz-c[2]],-1)
        nn=q@M
        nn/=np.maximum(np.linalg.norm(nn,axis=-1,keepdims=True),1e-10)
        Z[valid]=zz[valid];norm[valid]=nn[valid]
    for c,rad,R in ellipses:
        put_ellipsoid(c,rad,R)
    # Capsule = cylinder with hemispherical caps. Exact projected cylinder
    # intersections avoid polygon meshes and need no renderer dependency.
    for a,b,r in segs:
        axis=b-a;length=np.linalg.norm(axis);v=axis/length
        dx=X-a[0];dy=Y-a[1]
        # q=(dx,dy,t); perpendicular distance to the cylinder axis is r.
        A=1-v[2]**2
        B=-2*v[2]*(dx*v[0]+dy*v[1])
        C=dx**2+dy**2-(dx*v[0]+dy*v[1])**2-r*r
        disc=B*B-4*A*C
        if A>1e-9:
            t=(-B+np.sqrt(np.maximum(disc,0)))/(2*A)
            along=dx*v[0]+dy*v[1]+t*v[2]
            zz=a[2]+t
            valid=(disc>=0)&(along>=0)&(along<=length)&(zz>Z)
            nn=np.stack([dx-along*v[0],dy-along*v[1],t-along*v[2]],-1)/r
            Z[valid]=zz[valid];norm[valid]=nn[valid]
        for c in [a,b]:
            dd=(X-c[0])**2+(Y-c[1])**2
            dz=np.sqrt(np.maximum(r*r-dd,0))
            zz=c[2]+dz
            valid=(dd<=r*r)&(zz>Z)
            nn=np.stack([(X-c[0])/r,(Y-c[1])/r,dz/r],-1)
            Z[valid]=zz[valid];norm[valid]=nn[valid]
    mask=Z>-99
    # Merge capsule junctions into a continuous skin surface before shading.
    # Two-sided normalized depth smoothing supplies normals without the seams
    # of an unblended union of cylinders.
    den=smooth(mask.astype(float),1.05)
    depth=smooth(np.where(mask,Z,0),1.05)/np.maximum(den,1e-8)
    dy,dx=np.gradient(depth)
    surf=np.stack([-dx*n/extent,dy*n/extent,np.ones_like(dx)],-1)
    surf/=np.maximum(np.linalg.norm(surf,axis=-1,keepdims=True),1e-8)
    L=np.array([rng.uniform(-.8,.1),rng.uniform(.1,.8),1.])
    L/=np.linalg.norm(L)
    relief=rng.uniform(.13,.38)
    contrast=rng.uniform(.14,.50)
    shade=(contrast+relief*(surf@L))*mask
    # The sign of contrast changes only the silhouette-vs-relief relationship.
    # Dark hands on a bright background and the reverse are equally plausible.
    if rng.random()<.25:
        shade=(-contrast+relief*(surf@L))*mask
    # Joint relief and knuckle shading also remain visible when the hand and
    # background happen to have similar brightness.
    if rng.random()<.45:
        shade=(.24+.36*(surf@L)+.12*surf[:,:,2])*mask
    return shade,mask,Z,surf


def smooth(a,sigma):
    r=int(np.ceil(3*sigma));xx=np.arange(-r,r+1)
    k=np.exp(-xx**2/(2*sigma*sigma));k/=k.sum()
    p=np.pad(a,((r,r),(0,0)),mode='edge')
    t=sum(k[i]*p[i:i+a.shape[0],:] for i in range(len(k)))
    p=np.pad(t,((0,0),(r,r)),mode='edge')
    return sum(k[i]*p[:,i:i+a.shape[1]] for i in range(len(k)))


def template(letter,rng):
    shade,mask,depth,norm=render(letter,rng)
    # Use a small intrinsic scale tolerance while generating the fixed poses.
    # Apply exactly the same analytic front end to the independently generated
    # surface. This is construction, not a call on any training image.
    a=shade.reshape(28,2,28,2).mean((1,3))
    a=(a-a.mean())/np.sqrt(a.var()+1e-5)
    yy,xx=np.mgrid[-2:3,-2:3]
    gauss=np.exp(-(xx*xx+yy*yy)/(2*.95**2))
    p=np.pad(a,2)
    fs=[]
    for angle in np.arange(4)*np.pi/4:
        kernel=(xx*np.cos(angle)+yy*np.sin(angle))*gauss
        kernel/=np.abs(kernel).sum()
        g=sum(kernel[dy,dx]*p[dy:dy+28,dx:dx+28] for dy in range(5) for dx in range(5))
        g=np.abs(g)
        # Saturation suppresses spurious very strong background lines.
        cap=.22
        g=np.minimum(g,cap)
        g=g.reshape(GRID,2,GRID,2).max((1,3))
        g=(g-g.mean())/np.sqrt(g.var()+1e-5)
        fs.append(g)
    f=np.array(fs)
    # The same fixed saturation in templates and images limits the influence
    # of isolated edges and retains a penalty for edges absent in the model.
    f=np.clip(f,-1.0,2.0)
    f-=f.mean((1,2),keepdims=True)
    f/=np.sqrt((f*f).sum()+1e-10)
    return f.astype(np.float32).ravel()


def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    layers=[];weights={}
    def layer(name,op,inputs,**kw):
        layers.append(dict(name=name,op=op,inputs=inputs,**kw))
    layer('illumination','norm',['image'])
    # Derivative-of-Gaussian filters, both signs, with unit positive mass.
    k=5;yy,xx=np.mgrid[-2:3,-2:3];sigma=.95
    gauss=np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma))
    filters=[]
    for angle in np.arange(4)*np.pi/4:
        f=(xx*np.cos(angle)+yy*np.sin(angle))*gauss
        f/=np.maximum(np.abs(f).sum(),1e-9)
        filters.extend([f,-f])
    layer('derivatives','conv',['illumination'],out=8,kernel=[k,k],stride=1,padding=2)
    weights['derivatives.W']=np.array(filters,dtype=np.float32)[:,None]
    weights['derivatives.b']=np.zeros(8,np.float32)
    layer('half_waves','relu',['derivatives'])
    W=np.zeros((4,8,1,1),np.float32)
    for i in range(4):W[i,2*i:2*i+2]=1
    layer('edge_energy','conv',['half_waves'],out=4,kernel=[1,1],stride=1,padding=0)
    weights['edge_energy.W']=W;weights['edge_energy.b']=np.zeros(4,np.float32)
    layer('cap_shift','conv',['edge_energy'],out=4,kernel=[1,1],stride=1,padding=0)
    weights['cap_shift.W']=np.eye(4,dtype=np.float32)[:,:,None,None]
    weights['cap_shift.b']=np.full(4,-.22,np.float32)
    layer('cap_excess','relu',['cap_shift'])
    layer('negative_excess','conv',['cap_excess'],out=4,kernel=[1,1],stride=1,padding=0)
    weights['negative_excess.W']=-np.eye(4,dtype=np.float32)[:,:,None,None]
    weights['negative_excess.b']=np.zeros(4,np.float32)
    layer('capped_edges','add',['edge_energy','negative_excess'])
    layer('local_tolerance','maxpool',['capped_edges'],size=[2,2],stride=[2,2])
    layer('edge_contrast','norm',['local_tolerance'])
    layer('low_shift','conv',['edge_contrast'],out=4,kernel=[1,1],stride=1,padding=0)
    weights['low_shift.W']=np.eye(4,dtype=np.float32)[:,:,None,None]
    weights['low_shift.b']=np.ones(4,np.float32)
    layer('low_clip','relu',['low_shift'])
    layer('high_shift','conv',['low_clip'],out=4,kernel=[1,1],stride=1,padding=0)
    weights['high_shift.W']=np.eye(4,dtype=np.float32)[:,:,None,None]
    weights['high_shift.b']=np.full(4,-3,np.float32)
    layer('high_clip','relu',['high_shift'])
    layer('high_neg','conv',['high_clip'],out=4,kernel=[1,1],stride=1,padding=0)
    weights['high_neg.W']=-np.eye(4,dtype=np.float32)[:,:,None,None]
    weights['high_neg.b']=np.full(4,-1,np.float32)
    layer('robust_edges','add',['low_clip','high_neg'])
    bank=[]
    for letter in LETTERS:
        for j in range(PER_CLASS):
            bank.append(template(letter,rng))
    bank=np.array(bank,np.float32)
    layer('anatomical_matches','dense',['robust_edges'],out=len(bank))
    weights['anatomical_matches.W']=bank
    weights['anatomical_matches.b']=np.zeros(len(bank),np.float32)
    layer('pose_axis','reshape',['anatomical_matches'],shape=[24,1,PER_CLASS])
    layer('best_pose','maxpool',['pose_axis'],size=[1,PER_CLASS],stride=[1,PER_CLASS])
    layer('scores','reshape',['best_pose'],shape=[24])
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**weights)
    print(f'Built {len(bank)} analytic poses, {sum(a.size for a in weights.values()):,} weights.')

if __name__=='__main__':
    build(sys.argv[1])
