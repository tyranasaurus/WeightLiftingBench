import numpy as np

def smooth(a):
    v = np.pad(a, ((0,0),(1,1),(0,0)), mode="edge")
    v = (v[:,:-2] + 2*v[:,1:-1] + v[:,2:]) * .25
    w = np.pad(v, ((0,0),(0,0),(1,1)), mode="edge")
    return (w[:,:,:-2] + 2*w[:,:,1:-1] + w[:,:,2:]) * .25

def diff(a):
    q=np.pad(a,((0,0),(1,1),(1,1)),mode="edge")
    return (q[:,1:-1,2:]-q[:,1:-1,:-2])*.5,(q[:,2:,1:-1]-q[:,:-2,1:-1])*.5

def pooling(n):
    t=(np.arange(28,dtype=np.float32)+.5)*(n/28)-.5
    p=np.maximum(1-np.abs(t-np.arange(n,dtype=np.float32)[:,None]),0)
    return p/p.sum(1,keepdims=True)

pool7=pooling(7)
pool10=pooling(10)

def cells(v,p):
    return (p@v@p.T).reshape(len(v),-1)

def norm(v):
    return v/np.sqrt((v*v).sum(1,keepdims=True)+1.e-9)

def foreground(a):
    # Background is often a slowly changing wall. Interpolate from both sides
    # on each scan line and combine the residual with local texture evidence.
    v=smooth(a)
    t=(np.arange(28,dtype=np.float32)/27)[None,None,:]
    l=v[:,:,:2].mean(2,keepdims=True);r=v[:,:,-2:].mean(2,keepdims=True)
    # Large jumps between consecutive rows need not describe the background.
    u=v[:,:3,:4].mean((1,2),keepdims=True);w=v[:,:3,-4:].mean((1,2),keepdims=True)
    level=u*(1-t)+w*t
    dl=l-u;dr=r-w
    dl=np.where(np.abs(dl)<np.abs(dr),dl,np.clip(dl,dr-.11,dr+.11))
    dr=np.where(np.abs(dr)<np.abs(l-u),dr,np.clip(dr,l-u-.11,l-u+.11))
    back=level+dl*(1-t)+dr*t
    gx,gy=diff(v); mag=np.sqrt(gx*gx+gy*gy)
    texture=smooth(smooth(mag)); err=np.abs(v-back)
    signal=(err+.85*texture)/( .045+ .20*np.abs(l-r))
    mask=1-np.exp(-signal*signal*.53)
    tex=1-np.exp(-(texture/.034)**2)
    photometric=smooth(.71*mask+.29*tex)
    # Minimax barrier paths from the image frame. A patch that differs in
    # brightness from the distant wall can still be reached along a smooth
    # path, while the inside of a hand is often enclosed by a contrast edge.
    wall=np.ones_like(a)*.22
    wall[:,0,:]=0;wall[:,-1,:]=0;wall[:,:,0]=0;wall[:,:,-1]=0
    cost=(.65*a+.35*v)
    cx=np.abs(cost[:,:,1:]-cost[:,:,:-1]);cy=np.abs(cost[:,1:,:]-cost[:,:-1,:])
    for it in range(24):
        new=wall.copy()
        new[:,1:,:]=np.minimum(new[:,1:,:],np.maximum(cy,wall[:,:-1,:]))
        new[:,:-1,:]=np.minimum(new[:,:-1,:],np.maximum(cy,wall[:,1:,:]))
        new[:,:,1:]=np.minimum(new[:,:,1:],np.maximum(cx,wall[:,:,:-1]))
        new[:,:,:-1]=np.minimum(new[:,:,:-1],np.maximum(cx,wall[:,:,1:]))
        wall=new
    barrier=1-np.exp(-(wall/.031)**1.55)
    return .70*photometric+.30*smooth(barrier)

def features(a,m):
    a=np.asarray(a,dtype=np.float32);m=np.asarray(m,dtype=np.float32)
    # Compressive oriented energy and unsigned orientation channels are
    # insensitive to a reversal of contrast at the silhouette.
    s=smooth(a);gx,gy=diff(.6*s+.4*a); mag=np.sqrt(gx*gx+gy*gy+1.e-11)
    strength=mag/(.045+mag); power=np.sqrt(mag+1.e-8)*strength
    hog=[]
    for i in range(6):
        angle=i*np.pi/6
        cs=np.abs(gx*np.float32(np.cos(angle))+gy*np.float32(np.sin(angle)))/mag
        hog.append(cells(power*np.maximum(0,cs-.76)**2,pool7))
    h=norm(np.sqrt(np.maximum(0,np.concatenate(hog,1))+1.e-8))
    # Retain spatial variation, rather than correlating constant histograms.
    h=norm(h-.60*h.mean(1,keepdims=True))
    energy=cells(power,pool10);en=norm(energy-energy.mean(1,keepdims=True))
    mask=cells(m,pool10);mask=norm(mask-mask.mean(1,keepdims=True))
    # Bandpass shading makes the crease between digits useful, even if
    # parts of the hand have the same intensity as the wall.
    blur=s
    for j in range(5):blur=smooth(blur)
    band=s-blur;band=band/(.045+np.abs(band));bp=norm(cells(band,pool10))
    return np.concatenate((h*.70,en*.41,mask*.50,bp*.30),axis=1).astype(np.float32)

def curvefield(a):
    # Soft oriented chamfer fields. Each channel asks if an edge of the
    # specified normal passes nearby, irrespective of unrelated edges elsewhere.
    gx,gy=diff(.78*smooth(a)+.22*a);mag=np.sqrt(gx*gx+gy*gy+1.e-12)
    response=mag/(mag+.023)
    channels=[]
    for i in range(6):
        angle=i*np.pi/6
        direction=np.abs(gx*np.float32(np.cos(angle))+gy*np.float32(np.sin(angle)))/mag
        e=response*direction**6; base=e
        for j in range(2):
            p=np.pad(e,((0,0),(1,1),(1,1)),mode="edge")
            e=np.maximum(e,np.maximum(np.maximum(p[:,:-2,1:-1],p[:,2:,1:-1]),np.maximum(p[:,1:-1,:-2],p[:,1:-1,2:]))*.75)
        e=.75*e+.25*base
        channels.append(e.reshape(len(e),-1))
    return np.concatenate(channels,1)

def localcontrast(a):
    s=.6*smooth(a)+.4*a
    p=np.pad(s,((0,0),(2,2),(2,2)),mode="edge")
    pair=[]
    # Signed ordinal comparisons at two radii, along both cardinal and
    # diagonal directions. A concave knuckle, a fingertip and the edge of
    # a thumb crossing a finger have different patterns of these signs.
    for radius in (1,2):
        for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
            x=dx*radius;y=dy*radius
            left=p[:,2-y:30-y,2-x:30-x];right=p[:,2+y:30+y,2+x:30+x]
            linear=right-left;curve=s-(left+right)*.5
            pair.append((linear/(.055+np.abs(linear))).reshape(len(s),-1))
            pair.append((curve/(.032+np.abs(curve))).reshape(len(s),-1))
    return np.concatenate(pair,axis=1)

bank=CONSTANTS["templates"].astype(np.float32)*(1/32767.)
poses=bank.shape[1]
width=bank.shape[2]
bank=bank.reshape(-1,width).T.copy()
probe=CONSTANTS["probes"].reshape(24*poses,-1)
weight=CONSTANTS["weights"].reshape(24*poses,-1).astype(np.float32)*(1/32767.)
land=CONSTANTS["land"].reshape(24*poses,-1)
ordinalweight=CONSTANTS["ordinal"].reshape(24*poses,-1).astype(np.float32)*(1/32767.)

def shift(a,dy,dx):
    if dx==0 and dy==0:return a
    ypos=np.clip(np.arange(28)+dy,0,27);xpos=np.clip(np.arange(28)+dx,0,27)
    return a[:,ypos[:,None],xpos[None,:]]

def predict(x):
    labels=np.empty(len(x),dtype=np.int32)
    # Fixed matrix geometry, including for an incomplete batch, so BLAS
    # uses the same accumulation path for every query.
    for start in range(0,len(x),32):
        n=min(32,len(x)-start);a=np.zeros((32,28,28),np.float32);a[:n]=x[start:start+n,0]
        m=foreground(a)
        best=np.full((32,24,4),-10.,dtype=np.float32)
        offsets=np.arange(24)[None,:,None]*poses
        # Compute the translation once in the edge field, not separately for
        # each candidate. The directed score is calculated at full resolution.
        field=curvefield(a).reshape(32,6,28,28)
        lf=localcontrast(a).reshape(32,16,28,28)
        for dy,dx in ((0,0),(0,-2),(0,2),(-2,0),(2,0),(-2,-2),(-2,2),(2,-2),(2,2)):
            f=features(shift(a,dy,dx),shift(m,dy,dx));s=(f@bank).reshape(32,24,poses)
            idx=np.argpartition(s,-6,axis=2)[:,:,-6:];coarse=np.take_along_axis(s,idx,axis=2)
            idx=idx+offsets;ef=shift(field.reshape(-1,28,28),dy,dx).reshape(32,-1)
            hit=ef[np.arange(32)[:,None,None,None],probe[idx]]
            contour=(hit*weight[idx]).sum(3)
            detail=shift(lf.reshape(-1,28,28),dy,dx).reshape(32,-1)
            d=detail[np.arange(32)[:,None,None,None],land[idx]]
            anatomical=(d*ordinalweight[idx]).sum(3)
            candidate=.49*coarse+.36*contour+.15*anatomical - .0015*(abs(dx)+abs(dy))
            best=np.partition(np.concatenate((best,candidate),2),-4,axis=2)[:,:,-4:]
        score=.55*best.max(2)+.45*best.mean(2)
        labels[start:start+n]=score[:n].argmax(1)
    return labels
