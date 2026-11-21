import numpy as np

def sample(im,x,y):
    n,h,w=im.shape
    ii=np.arange(n)[:,None,None]
    x=np.clip(x,0,w-1.001); y=np.clip(y,0,h-1.001)
    ix=x.astype(np.int32); iy=y.astype(np.int32)
    dx=(x-ix).astype(np.float32); dy=(y-iy).astype(np.float32)
    return (1-dy)*((1-dx)*im[ii,iy,ix]+dx*im[ii,iy,ix+1])+dy*((1-dx)*im[ii,iy+1,ix]+dx*im[ii,iy+1,ix+1])

def normalize(im):
    g=np.arange(28,dtype=np.float32); xx=g[None,None,:]; yy=g[None,:,None]
    mass=np.maximum(im.sum(axis=(1,2),keepdims=True),.0001)
    cx=(im*xx).sum(axis=(1,2),keepdims=True)/mass; cy=(im*yy).sum(axis=(1,2),keepdims=True)/mass
    tilt=(im*(xx-cx)*(yy-cy)).sum(axis=(1,2),keepdims=True)/np.maximum((im*(yy-cy)**2).sum(axis=(1,2),keepdims=True),.001)
    tilt=np.clip(tilt,-1.3,1.3)
    # Deskew and bound the glyph by the high-confidence part of its ink.
    wide=np.arange(44,dtype=np.float32)[None,None,:]-8
    part=sample(im,wide+.88*tilt*(yy-cy),yy+np.zeros_like(wide))
    binary=part>np.maximum(.075,part.max(axis=(1,2),keepdims=True)*.20)
    xs=binary.any(axis=1); ys=binary.any(axis=2)
    x0=xs.argmax(axis=1); x1=43-xs[:,::-1].argmax(axis=1)
    y0=ys.argmax(axis=1); y1=27-ys[:,::-1].argmax(axis=1)
    w=x1-x0+1; h=y1-y0+1
    # Partial rather than complete aspect normalization; don't inflate a straight 1.
    sh=h/23.; sw=sh*np.clip((w/np.maximum(.66*h,1))**.66,.65,1.62)
    sh=sh.astype(np.float32)[:,None,None]; sw=sw.astype(np.float32)[:,None,None]
    oy=(y0+y1).astype(np.float32)[:,None,None]*.5; ox=(x0+x1-16).astype(np.float32)[:,None,None]*.5
    y=oy+(yy-13.5)*sh
    x=ox+(xx-13.5)*sw+.88*tilt*(y-cy)
    z=sample(im,x,y)
    aspect=np.log(np.maximum(w,.5)/np.maximum(h,.5)).astype(np.float32)
    return z,aspect

def pool(a,weights):
    return (weights @ a @ weights.T)

def gaussianbins(n,scale):
    axis=np.arange(28,dtype=np.float32)
    c=(np.arange(n,dtype=np.float32)+.5)*28/n-.5
    w=np.exp(-.5*((axis[None,:]-c[:,None])/scale)**2)
    return w/w.sum(axis=1,keepdims=True)

coarse=gaussianbins(6,2.25); fine=gaussianbins(9,1.35)

def unit(a):
    return a/np.sqrt(np.maximum((a*a).sum(axis=tuple(range(1,a.ndim)),keepdims=True),.000001))

def holes(z):
    # Ink counters at several brightness levels. Flood through the background
    # from all four borders, then locate any unreachable background by height.
    n=len(z); q=z.max(axis=(1,2),keepdims=True)
    levels=[]
    for v in (.26,.50,.73):
        free=z<=v*q
        a=np.ones_like(free); a[:,1:-1,1:-1]=False; a &= free
        for it in range(80):
            b=a.copy(); b[:,1:]|=a[:,:-1]; b[:,:-1]|=a[:,1:]; b[:,:,1:]|=a[:,:,:-1]; b[:,:,:-1]|=a[:,:,1:]; b &= free
            if np.array_equal(a,b): break
            a=b
        trapped=(free & ~a).sum(axis=2).astype(np.float32)
        h=trapped@coarse.T
        # Saturate once a counter is unmistakable; the metric isn't meant
        # to enforce equal size holes in thick and thin handwriting.
        levels.append(np.minimum(np.sqrt(h*.50),1.0))
    return sum(levels)/3

def descriptors(z):
    # Smooth centered differences, then six soft directed orientation channels.
    zp=np.pad(z,((0,0),(1,1),(1,1)),mode='constant')
    dx=.25*((zp[:,:-2,2:]-zp[:,:-2,:-2])+2*(zp[:,1:-1,2:]-zp[:,1:-1,:-2])+(zp[:,2:,2:]-zp[:,2:,:-2]))
    dy=.25*((zp[:,2:,:-2]-zp[:,:-2,:-2])+2*(zp[:,2:,1:-1]-zp[:,:-2,1:-1])+(zp[:,2:,2:]-zp[:,:-2,2:]))
    den=1/(.035+np.sqrt(dx*dx+dy*dy))
    small=[]; large=[]
    for k in range(6):
        angle=k*np.pi/3
        p=np.maximum(np.float32(np.cos(angle))*dx+np.float32(np.sin(angle))*dy,0)
        p=p*p*den
        small.append(np.sqrt(pool(p,coarse)+.00001)); large.append(np.sqrt(pool(p,fine)+.00001))
    s=unit(np.stack(small,axis=1)); l=unit(np.stack(large,axis=1))
    ink=unit(np.sqrt(np.maximum(pool(z,fine),0)))
    return s.reshape(len(z),-1),l,ink

# Scaled integers in the coarse comparison keep the candidate set independent
# of reduction order (all of the dot products are float32-exact integers).
shelf=CONSTANTS['coarse'].astype(np.float32)
shapes=CONSTANTS['fine']; silhouettes=CONSTANTS['ink']; aspects=CONSTANTS['aspect']

def classify(z,aspect):
    n=len(z); s,l,ink=descriptors(z)
    s=np.rint(512*s)
    score=s@shelf.T
    # Preliminary comparison only: use a second scale and elastic matching below.
    per=shelf.shape[0]//10
    score=score.reshape(n,10,per)
    score-=np.minimum(np.abs(aspect[:,None,None]-aspects.reshape(1,10,per)),1.5)*4200
    top=holes(z)
    counter=CONSTANTS['counter']
    # Use the counter descriptor gently in the search; occasional open loops
    # or missing ink should not disqualify an entire shape family.
    for t in range(6):
        score-=np.abs(top[:,t,None,None]*127-counter[:,t].reshape(1,10,per))*6
    k=32
    ix=np.argpartition(score,per-k,axis=2)[:,:,per-k:]+np.arange(10)[None,:,None]*per
    ids=ix.reshape(n,-1); out=np.empty((n,10),dtype=np.float32)
    for j in range(n):
        # Local block descriptors can move by one cell with a small spring cost.
        # Both rigid and non-rigid evidence matter; the latter is particularly
        # useful at hooks and junctions written at varying heights.
        f=shapes[ids[j]].astype(np.float32)/1024
        b=silhouettes[ids[j]].astype(np.float32)/1024
        target=l[j]; ib=ink[j]
        rigid=(f*target).sum(axis=(1,2,3)); ir=(b*ib).sum(axis=(1,2))
        af=(f*f).sum(axis=1); at=(target*target).sum(axis=0)
        # Aggregate overlapping 3x3 patches of the square-root orientation map.
        # Candidate and query energies are both included (unlike a one-sided
        # local correlation). The outermost cells mostly cover the margin.
        best=None
        for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)):
            tt=np.roll(target,(dy,dx),axis=(1,2)); it=np.roll(ib,(dy,dx),axis=(0,1)); en=np.roll(at,(dy,dx),axis=(0,1))
            e=(af+en-2*(f*tt).sum(axis=1))+.32*(b-it)**2
            # The patch centers are at cells 1, 3, 5 and 7.
            local=e[:,0:7:2,0:7:2]+e[:,0:7:2,2:9:2]+e[:,2:9:2,0:7:2]+e[:,2:9:2,2:9:2]+2*(e[:,1:8:2,0:7:2]+e[:,1:8:2,2:9:2]+e[:,0:7:2,1:8:2]+e[:,2:9:2,1:8:2])+4*e[:,1:8:2,1:8:2]
            local+=.004*(abs(dx)+abs(dy))
            if best is None: best=local
            else: best=np.minimum(best,local)
        elastic=best.sum(axis=(1,2))*.25
        ar=np.minimum(np.abs(aspect[j]-aspects[ids[j]]),1.6)
        coarseval=score[j,np.arange(10)[:,None], ix[j]%per].reshape(-1)/262144
        hd=np.abs(top[j]-counter[ids[j]].astype(np.float32)/127).mean(axis=1)
        val=(.31*rigid+.12*ir+.23*coarseval-.37*elastic-.012*ar-.054*hd).reshape(10,k)
        v=np.sort(val,axis=1)
        out[j]=.55*v[:,-1]+.27*v[:,-2]+.18*v[:,-3]
    return np.argmax(out,axis=1)

def predict(x):
    labels=np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),64):
        stop=min(len(x),start+64)
        z,a=normalize(x[start:stop,0]); labels[start:stop]=classify(z,a)
    return labels
