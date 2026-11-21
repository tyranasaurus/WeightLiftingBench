
import numpy as np

def samplegrid(a,xx,yy):
    n=len(a)
    xx=np.clip(xx,0,26.99999); yy=np.clip(yy,0,26.99999)
    i=xx.astype(np.int32); j=yy.astype(np.int32)
    u=(xx-i).astype(np.float32); v=(yy-j).astype(np.float32)
    b=np.arange(n)[:,None,None]
    return (1-v)*((1-u)*a[b,j,i]+u*a[b,j,i+1])+v*((1-u)*a[b,j+1,i]+u*a[b,j+1,i+1])

coord=np.arange(28,dtype=np.float32)
ux=(coord[None,None,:]-13.5)
uy=(coord[None,:,None]-13.5)

def canonical(a):
    mass=np.maximum(a.sum((1,2),keepdims=True),.001)
    mx=(a*ux).sum((1,2),keepdims=True)/mass; my=(a*uy).sum((1,2),keepdims=True)/mass
    dx=ux-mx; dy=uy-my
    slant=(a*dx*dy).sum((1,2),keepdims=True)/np.maximum((a*dy*dy).sum((1,2),keepdims=True),.01)
    slant=np.clip(slant,-1.4,1.4)
    ds=dx-slant*dy
    sx=np.maximum(np.sqrt((a*ds*ds).sum((1,2),keepdims=True)/mass),.6)
    sy=np.maximum(np.sqrt((a*dy*dy).sum((1,2),keepdims=True)/mass),.6)
    # Only partial width normalization: a narrow one must stay narrow.
    fx=np.clip((sx/4.1)**.73,.70,1.60); fy=np.clip((sy/5.8)**.80,.66,1.50)
    y=uy*fy; x=ux*fx+slant*y
    c=samplegrid(a,13.5+mx+x,13.5+my+y)
    return c,np.log(sx/sy).reshape(-1),slant.reshape(-1)

def blur(a):
    b=np.pad(a,((0,0),(1,1),(1,1)),mode='constant'); c=(b[:,:-2]+2*b[:,1:-1]+b[:,2:])*.25
    return (c[:,:,:-2]+2*c[:,:,1:-1]+c[:,:,2:])*.25

def orientations(a):
    a=blur(a); p=np.pad(a,((0,0),(1,1),(1,1)),mode='constant')
    gx=(p[:,1:-1,2:]-p[:,1:-1,:-2]); gy=(p[:,2:,1:-1]-p[:,:-2,1:-1]); mag=np.sqrt(gx*gx+gy*gy+.0000001)
    res=[]
    for c,s in [(1,0),(.7071068,.7071068),(0,1),(-.7071068,.7071068)]:
        t=gx*c+gy*s
        res.append(np.maximum(0,2*t*t/mag-mag))
    return res

def unit(v):
    return v/np.maximum(.00001,np.sqrt((v*v).sum(tuple(range(1,v.ndim)),keepdims=True)))

def cells(a):
    # Gaussian integration windows on a regular lattice of stroke locations.
    return (cellmat @ a) @ cellmat.T

cellpos=np.arange(6,dtype=np.float32)*4+3.5
cellmat=np.exp(-.5*((cellpos[:,None]-coord)/2.4)**2).astype(np.float32)
cellmat/=cellmat.sum(1,keepdims=True)

def cavities(c):
    # Escape cost of the background: the lowest ink barrier on a path from
    # this pixel to the border. In an enclosed counter the barrier is high.
    # Alternating row/column scans propagate around open ends of long strokes.
    d=np.ones_like(c); d[:,0,:]=c[:,0,:]; d[:,-1,:]=c[:,-1,:]
    d[:,:,0]=c[:,:,0]; d[:,:,-1]=c[:,:,-1]
    for sweep in range(2):
        for i in range(1,28):
            d[:,i,:]=np.minimum(d[:,i,:],np.maximum(c[:,i,:],d[:,i-1,:]))
            d[:,:,i]=np.minimum(d[:,:,i],np.maximum(c[:,:,i],d[:,:,i-1]))
        for i in range(26,-1,-1):
            d[:,i,:]=np.minimum(d[:,i,:],np.maximum(c[:,i,:],d[:,i+1,:]))
            d[:,:,i]=np.minimum(d[:,:,i],np.maximum(c[:,:,i],d[:,:,i+1]))
    return np.maximum(0,d-c)

def topology(c):
    # Thin the strokes by deleting only simple boundary points (the two
    # directional passes of Zhang-Suen). A loop then has no stroke ends; a
    # line has two, and the protruding arms of a crossing have their own ends.
    a=np.pad(c>.32,((0,0),(1,1),(1,1)),mode='constant')
    def neighbors():
        return (a[:,:-2,1:-1],a[:,:-2,2:],a[:,1:-1,2:],a[:,2:,2:],a[:,2:,1:-1],a[:,2:,:-2],a[:,1:-1,:-2],a[:,:-2,:-2])
    def crossings(ns):
        t=np.zeros_like(ns[0],dtype=np.uint8)
        for i in range(8): t+=(~ns[i-1])&ns[i]
        return t
    for sweep in range(12):
        changed=False
        for step in range(2):
            ns=neighbors(); north,ne,east,se,south,sw,west,nw=ns
            deg=np.add.reduce(ns,dtype=np.uint8); tr=crossings(ns)
            if step==0: keep=(north&east&south)|(east&south&west)
            else: keep=(north&east&west)|(north&south&west)
            remove=a[:,1:-1,1:-1] & (deg>=2) & (deg<=6) & (tr==1) & ~keep
            if np.any(remove):
                changed=True; a[:,1:-1,1:-1]&=~remove
        if not changed: break
    ns=neighbors(); deg=np.add.reduce(ns,dtype=np.uint8); tr=crossings(ns)
    thin=a[:,1:-1,1:-1]; tip=thin & (deg==1); branch=thin & (tr>=3)
    maps=[]
    for target in [tip,branch]:
        v=topwin @ target.astype(np.float32) @ topwin.T
        maps.append(np.minimum(1.,v).reshape(len(c),-1))
    return np.rint(np.concatenate(maps,axis=1)*127).astype(np.uint8)

topwin=np.exp(-.5*((np.arange(5,dtype=np.float32)[:,None]*5+3.5-coord)/3.5)**2)

def coarse(c,aspect,slant,holes):
    v=unit(cells(c)).reshape(len(c),-1)
    h=np.stack([cells(o) for o in orientations(c)],axis=1)
    h=unit(np.sqrt(np.maximum(h,0)+.008)-.08944272).reshape(len(c),-1)
    # The small RBF embeddings preserve width/slant without scalar cross terms.
    ar=np.exp(-.5*((aspect[:,None]-np.linspace(-2.4,.8,10,dtype=np.float32))/.35)**2)
    sk=np.exp(-.5*((slant[:,None]-np.linspace(-1.6,1.6,8,dtype=np.float32))/.45)**2)
    hh=cells(holes)*1.4
    cav=unit(np.concatenate((hh.reshape(len(c),-1),np.ones((len(c),1),dtype=np.float32)),axis=1))
    v=np.concatenate((.59*v,.76*h,.235*unit(ar),.12*unit(sk),.21*cav),axis=1)
    return np.rint(v*2048).astype(np.int16)

def pool2(a):
    return (a[:,0::2,0::2]+a[:,1::2,0::2]+a[:,0::2,1::2]+a[:,1::2,1::2])*.25

def fine(c,holes):
    # All maps share the same displacement in the elastic comparison. Opposite
    # directions share a bin, but different orientation axes are independent.
    o=orientations(c); maps=[pool2(blur(c))]+[pool2(blur(q))*1.75 for q in o]+[.75*pool2(blur(holes))]
    return np.rint(np.clip(np.stack(maps,axis=-1),0,1)*31).astype(np.uint32)

powers=np.array([1,32,1024,32768,1048576,33554432],dtype=np.uint32)

lookup=CONSTANTS['coarse'].astype(np.float32).T.copy()
low=CONSTANTS['low'].astype(np.uint32)
high=CONSTANTS['high'].astype(np.uint32)
detail=low+(high<<15)
del low,high
ends=CONSTANTS['topology']
offsets=np.arange(6,dtype=np.uint32)*5
perdigit=detail.shape[0]//10

# A 3 by 3 lattice of overlapping Gaussian patches. Patch offsets have a
# displacement prior and must mostly agree with neighboring patches.
pg=np.arange(14,dtype=np.float32)
pm=np.exp(-.5*((pg[None,:]-np.array([2.8,6.5,10.2],dtype=np.float32)[:,None])/2.15)**2)
patch=(pm[:,None,:,None]*pm[None,:,None,:]).reshape(9,196)
patch/=np.maximum(patch.sum(0,keepdims=True),.22)
patch=patch.T.copy()

def elastic(q,idx):
    # Work on just one writing sample at a time in this more detailed stage.
    t=((detail[idx][:,:,:,None] >> offsets) & 31).astype(np.float32)/31
    q=q.astype(np.float32)/31
    quilt=np.pad(q,((1,1),(1,1),(0,0)),mode='constant')
    many=len(idx); loc=[]
    for y in range(3):
        for x in range(3):
            loc.append(quilt[y:y+14,x:x+14].reshape(196,6))
    loc=np.stack(loc,axis=1)
    # SSD for vector-valued cells. The matrix operations are over templates,
    # not different input samples. Distances have symmetric ink normalization.
    a=t.reshape(many,196,6)
    dots=np.matmul(a.transpose(1,0,2),loc.transpose(0,2,1)).transpose(1,2,0)
    qm=(loc*loc).sum(2).T; tm=(a*a).sum(2)
    cost=(tm[:,None,:]+qm[None,:,:]-2*dots) @ patch
    mass=(tm+qm[4]) @ patch
    shift=np.array([2,1,2,1,0,1,2,1,2],dtype=np.float32)
    cost+=shift[None,:,None]*(.034*mass[:,None,:]+.045)
    # Smooth displacement field, via min-sum on a tree rooted in the centre.
    coords=np.arange(9); sy=coords//3; sx=coords%3
    smooth=np.minimum(3,(sy[:,None]-sy)**2+(sx[:,None]-sx)**2).astype(np.float32)
    for child,parent in [(0,1),(2,1),(6,7),(8,7),(1,4),(3,4),(5,4),(7,4)]:
        w=.025*np.minimum(mass[:,child],mass[:,parent])+.032
        cost[:,:,parent]+=np.min(cost[:,:,child,None]+w[:,None,None]*smooth,axis=1)
    return np.min(cost[:,:,4],axis=1)/np.maximum(1,mass.sum(1))

def predict(x):
    n=len(x); out=np.empty(n,dtype=np.int64); k=16
    for start in range(0,n,64):
        a=x[start:start+64,0]; m=len(a)
        # Fixed matrix shape and exact integer-valued coarse features make the
        # retrieval independent of the position of the end of the caller's batch.
        c,aspect,slant=canonical(a)
        holes=cavities(c)
        v=coarse(c,aspect,slant,holes); block=np.zeros((64,v.shape[1]),np.float32); block[:m]=v
        score=(block @ lookup)[:m].reshape(m,10,perdigit)
        ix=np.argpartition(score,perdigit-k,axis=2)[:,:,-k:]
        cs=np.take_along_axis(score,ix,axis=2)/4194304
        ids=ix+perdigit*np.arange(10)[None,:,None]
        qs=fine(c,holes); ep=topology(c).astype(np.float32)
        for j in range(m):
            err=elastic(qs[j],ids[j].reshape(-1)).reshape(10,k)
            # A bounded extra cost: stroke ends provide useful evidence, but
            # digitization can introduce a false spur or bridge a narrow gap.
            diff=(ends[ids[j]].astype(np.float32)-ep[j])/127
            delta=np.minimum(.65,np.abs(diff))**2
            extra=.014*delta[:,:,:25].sum(2)+.007*delta[:,:,25:].sum(2)
            cost=err-.43*cs[j]+np.minimum(extra,.070)
            best=np.sort(cost,axis=1)
            # Two independent plausible stroke renderings must support the match.
            out[start+j]=np.argmin(.77*best[:,0]+.23*best[:,1])
    return out
