import numpy as np

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
