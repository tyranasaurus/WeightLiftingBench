import numpy as np

pixel = np.arange(28, dtype=np.float32)
xx = pixel[None,None,:]
yy = pixel[None,:,None]

def normalize(a):
    # Partial moment deskew, then normalize position and partially normalize
    # aspect. A straight one should stay narrow, not become a thick column.
    mass = np.maximum(a.sum((1,2)), 1e-6)
    cy = (a*yy).sum((1,2))/mass
    cx = (a*xx).sum((1,2))/mass
    dy = yy-cy[:,None,None]
    dx = xx-cx[:,None,None]
    slope = .81*np.clip((a*dx*dy).sum((1,2))/(1e-5+(a*dy*dy).sum((1,2))), -1.15,1.15)
    pos = xx-slope[:,None,None]*dy
    mask = a > np.maximum(1e-6,.19*a.max((1,2),keepdims=True))
    left = np.where(mask,pos,64).min((1,2)); right = np.where(mask,pos,-64).max((1,2))
    top = np.where(mask,yy,64).min((1,2)); bot = np.where(mask,yy,-64).max((1,2))
    h = np.maximum(bot-top+1,1); w=np.maximum(right-left+1,1)
    sy=22/h
    sx=np.minimum(sy*1.65,np.minimum(24,17.5*(w/(.74*h))**.42)/w)
    y = (yy-13.5)/sy[:,None,None]+((top+bot)*.5)[:,None,None]
    x = (xx-13.5)/sx[:,None,None]+((right+left)*.5)[:,None,None]+slope[:,None,None]*(y-cy[:,None,None])
    i=np.floor(x).astype(np.int32); j=np.floor(y).astype(np.int32); fx=x-i; fy=y-j
    n=np.arange(len(a))[:,None,None]
    def sample(j,i):
        return a[n,j.clip(0,27),i.clip(0,27)]*((i>=0)&(i<28)&(j>=0)&(j<28))
    out=(1-fy)*((1-fx)*sample(j,i)+fx*sample(j,i+1))+fy*((1-fx)*sample(j+1,i)+fx*sample(j+1,i+1))
    return out.astype(np.float32),np.log(np.maximum(w/h,.05))

def gauss(s,n):
    centers=13.5 + (np.arange(n,dtype=np.float32)-(n-1)*.5)*(27./n)
    g=np.exp(-.5*((centers[:,None]-pixel[None,:])/s)**2)
    return (g/g.sum(1,keepdims=True)).astype(np.float32)

finegrid=gauss(1.38,10)
coarsegrid=gauss(2.15,6)
voidgrid=gauss(2.5,5)

def pool(a,g):
    return (g @ a @ g.T)

def smooth(a):
    p=np.pad(a,((0,0),(1,1),(1,1)))
    return (.25*p[:,1:-1,:-2]+.50*p[:,1:-1,1:-1]+.25*p[:,1:-1,2:])*.5 + (.25*p[:,:-2,1:-1]+.50*p[:,1:-1,1:-1]+.25*p[:,2:,1:-1])*.5

def holes(a):
    # Flood exterior paper, not ink. Small isolated pinholes carry little
    # weight. Padding models the unbounded region outside the writing box.
    bg=np.pad(a,((0,0),(1,1),(1,1)),constant_values=False)
    reach=np.zeros_like(bg); reach[:,1,1:-1]=True; reach[:,-2,1:-1]=True
    reach[:,1:-1,1]=True; reach[:,1:-1,-2]=True; reach &= bg
    for step in range(60):
        prev=reach[:,1:-1,1:-1]
        nex=(prev|reach[:,:-2,1:-1]|reach[:,2:,1:-1]|reach[:,1:-1,:-2]|reach[:,1:-1,2:]) & a
        if np.array_equal(prev,nex): break
        reach[:,1:-1,1:-1]=nex
    return a & ~reach[:,1:-1,1:-1]

def unit(a):
    return a/np.maximum(np.sqrt(np.sum(a*a,axis=1,keepdims=True)),1e-6)

def describe(arr):
    a,aspect=normalize(arr)
    a=a/np.maximum(a.max((1,2),keepdims=True),.001)
    b=smooth(smooth(a)); p=np.pad(b,((0,0),(1,1),(1,1)))
    gx=p[:,1:-1,2:]-p[:,1:-1,:-2]; gy=p[:,2:,1:-1]-p[:,:-2,1:-1]
    mag=np.sqrt(gx*gx+gy*gy)
    co=[]; fi=[]
    angle=(np.arctan2(gy,gx)*(4/np.pi))%8
    for k in range(8):
        d=np.abs(angle-k); d=np.minimum(d,8-d)
        vote=mag*(np.maximum(1-d,0)+.22*np.maximum(np.abs(d-2)-1,0))
        co.append(pool(vote,coarsegrid)); fi.append(pool(vote,finegrid))
    def hog(ch):
        z=np.stack(ch,axis=-1)
        # Local normalization tempers differences in width, ink, and contrast.
        return np.sqrt(z/(.14+z.sum(-1,keepdims=True)))
    cg=hog(co); fg=hog(fi)
    # Visibility from the four page edges: a notch open to the left is
    # different from a hole and from a notch open to the right.
    near=a>.30
    l=np.maximum.accumulate(near,axis=2); r=np.maximum.accumulate(near[:,:,::-1],axis=2)[:,:,::-1]
    t=np.maximum.accumulate(near,axis=1); btm=np.maximum.accumulate(near[:,::-1],axis=1)[:,::-1]
    paper=1-a
    h1=holes(a<.45).astype(np.float32); h2=holes(a<.17).astype(np.float32)
    hole=.55*h1+.45*h2
    hollow=pool(hole,voidgrid)
    notches=[]
    for v in ((l&r),(t&btm),(~l&r&t&btm),(l&~r&t&btm),(l&r&~t&btm),(l&r&t&~btm)):
        notches.append(pool(paper*v,voidgrid))
    cavity=np.stack(notches,-1)
    n=len(a)
    cv=cavity.reshape(n,-1); hv=hollow.reshape(n,-1)
    # Weights are fixed from the semantics of each channel. Everything is
    # normalized within one drawing, never across drawings.
    main=unit(cg.reshape(n,-1)); area=unit(pool(a,coarsegrid).reshape(n,-1))
    extra=cv/np.sqrt(7.+(cv*cv).sum(1,keepdims=True))
    hcode=np.sqrt(np.maximum(hv,0)); hcode=hcode/np.sqrt(2.5+(hcode*hcode).sum(1,keepdims=True))
    aspectcode=np.exp(-.5*((aspect[:,None]-np.linspace(-2.5,.65,10)[None,:])/.37)**2).astype(np.float32)
    desc=np.concatenate([main,.47*area,.62*extra,.48*hcode,.22*unit(aspectcode)],axis=1)
    # Euclidean distance includes the lengths of the notch and hole codes.
    coarse=np.rint(desc*1450).astype(np.int16)
    fm=pool(a,finegrid); fh=pool(hole,finegrid)
    fine=np.concatenate([fg,.72*np.sqrt(fm[:,:,:,None]),.56*np.sqrt(fh[:,:,:,None])],axis=3)
    fine=np.rint(np.clip(fine,0,1)*200).astype(np.uint8)
    return coarse,fine

# All rows in the alphabet remain separate; there is no pooled class model.
coarseBank=CONSTANTS['coarse'].astype(np.float32)
coarseLen=(coarseBank*coarseBank).sum(1)
fineBank=CONSTANTS['fine']
perDigit=len(fineBank)//10

def localcost(q,bank):
    # Local elastic matching: 25 overlapping-context cells grouped into a
    # 5-by-5 mesh. Compare 2-by-2 patches at each neighboring displacement.
    # A second, coarser mesh encourages nearby patches to move together.
    m=len(bank)
    b=np.pad(bank.astype(np.float32)/200,((0,0),(1,1),(1,1),(0,0)))
    q=q.astype(np.float32)/200
    ds=[]
    for y in range(3):
        for x in range(3):
            dif=b[:,y:y+10,x:x+10,:]-q
            e=np.sum(dif*dif,axis=3).reshape(m,5,2,5,2).mean((2,4))
            ds.append(e)
    dist=np.stack(ds,-1)
    local=dist[:,:,:,4]
    # Penalize displacements but don't penalize zero displacement. The cost
    # is expressed in units of squared normalized patch-feature differences.
    pen=np.array([.072,.045,.072,.045,0,.045,.072,.045,.072],np.float32)
    best=(dist+pen).min(3)
    pad=np.pad(dist,((0,0),(1,1),(1,1),(0,0)),mode='edge')
    neighbor=(2*dist+pad[:,:-2,1:-1]+pad[:,2:,1:-1]+pad[:,1:-1,:-2]+pad[:,1:-1,2:])/6
    group=(neighbor+pen*.76).min(3)
    return (.43*best+.42*group+.15*local).mean((1,2))

def predict(x):
    out=np.empty(len(x),np.int32)
    for start in range(0,len(x),64):
        stop=min(len(x),start+64); n=stop-start
        c,f=describe(x[start:stop,0]); c=c.astype(np.float32)
        # Integer descriptors ensure all inner products are exact in float32.
        dots=c @ coarseBank.T
        raw=(coarseLen[None,:]+(c*c).sum(1,keepdims=True)-2*dots)/(1450*1450)
        raw=raw.reshape(n,10,perDigit)
        for i in range(n):
            # Rerank a small, separate shortlist from every class.
            ix=np.argpartition(raw[i],8,axis=1)[:,:8]
            ci=np.arange(10)[:,None]
            cd=raw[i][ci,ix]; ids=(ci*perDigit+ix).ravel()
            ld=localcost(f[i],fineBank[ids]).reshape(10,-1)
            dist=.50*cd+1.6*ld
            # A little corroboration from other independent pen renditions.
            first=np.sort(dist,axis=1)
            out[start+i]=np.argmin(.72*first[:,0]+.19*first[:,1]+.09*first[:,2])
    return out
