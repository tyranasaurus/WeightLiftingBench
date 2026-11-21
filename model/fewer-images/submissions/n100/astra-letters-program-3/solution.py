import numpy as np

def blur(a, axis, w):
    k=len(w)//2
    pad=[(0,0)]*a.ndim
    pad[axis]=(k,k)
    b=np.pad(a,pad,mode='edge')
    sl=[slice(None)]*a.ndim
    out=np.zeros_like(a)
    for j,c in enumerate(w):
        sl[axis]=slice(j,j+a.shape[axis]);out+=c*b[tuple(sl)]
    return out

def filt(a,w):
    return blur(blur(a,2,w),1,w)

def features(a):
    # Cells describe edges as both unoriented axes and signed contrasts.
    # The luminance channel is a band pass, not absolute skin brightness.
    small=[.125,.75,.125]; wide=[.0625,.25,.375,.25,.0625]
    a=filt(a,small)
    b=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    gx=(b[:,1:-1,2:]-b[:,1:-1,:-2])*.5
    gy=(b[:,2:,1:-1]-b[:,:-2,1:-1])*.5
    mag=np.sqrt(gx*gx+gy*gy+.000001)
    power=filt(mag,wide)
    amp=np.minimum(mag/(.016+.75*power)**.78,2.)
    chans=[]; signed=[]
    for k in range(6):
        v=(gx*np.cos(k*np.pi/6)+gy*np.sin(k*np.pi/6))/mag
        axis=np.maximum(0.,(np.abs(v)-.78)/.22)*amp
        chans.append(axis);signed.append(axis*np.sign(v))
    gray=(a-filt(filt(a,wide),wide))/(.055+power)
    maps=np.stack(chans+signed+[amp,np.clip(gray,-2.5,2.5)],axis=3).astype(np.float32)
    def make(m):
        axes=m[:,:,:,:6].copy()
        level=m[:,:,:,12:13]
        axes-=.145*level
        m=np.concatenate([axes, .64*m[:,:,:,6:12], .48*level, .14*m[:,:,:,13:14]],axis=3)
        return m/np.sqrt(.055+level)
    fine=make(filt(maps,[.08,.23,.38,.23,.08]))
    coarse=make(filt(filt(maps,wide),wide))
    # Saliency for the rendered outer contour: allow a locally light or dark
    # background, and tolerate an edge location uncertain by about one pixel.
    edge=(mag/(mag+.025))*(power/(power+.020))
    oriented=np.stack([z/(amp+.0001)*edge for z in chans],axis=3)
    b=np.pad(oriented,((0,0),(1,1),(1,1),(0,0)),mode='edge')
    pool=np.maximum.reduce([b[:,j:j+28,i:i+28] for j,i in ((0,1),(1,0),(2,1),(1,2),(1,1))])
    oriented=filt(.65*pool+.35*oriented,[.2,.6,.2])
    return fine,coarse,oriented.astype(np.float32)

def sample(a,y,x):
    # All points are pixel centers in the original frame. Do not hallucinate
    # edges in the unobserved region beyond the photograph.
    yy=np.clip(y,0,27);xx=np.clip(x,0,27)
    iy=yy.astype(np.int32);ix=xx.astype(np.int32)
    wy=(yy-iy).astype(np.float32);wx=(xx-ix).astype(np.float32)
    v=((1-wx)[None,None,:,None]*a[:,:,ix]+wx[None,None,:,None]*a[:,:,np.minimum(ix+1,27)])
    v=(1-wy)[None,:,None,None]*v[:,iy]+wy[None,:,None,None]*v[:,np.minimum(iy+1,27)]
    ey=np.clip(1-np.abs(yy-y)*.7,0,1);ex=np.clip(1-np.abs(xx-x)*.7,0,1)
    return v*(ey[:,None]*ex[None,:])[None,:,:,None]

def grid(n):
    if n==7: return np.arange(n,dtype=np.float32)*4+1.5
    return np.arange(n,dtype=np.float32)*2+2.5

def descript(m,n,dx,dy,sc):
    t=grid(n)-13.5
    w=(1-.25*(np.abs(t)/13.5)**3).astype(np.float32)
    v=sample(m,13.5+dy+t*sc,13.5+dx+t*sc)*(w[None,:,None,None]*w[None,None,:,None])
    return v/np.sqrt(np.maximum(.000001,(v*v).sum(axis=(1,2,3),keepdims=True)))

def contour(body,skin,n):
    b=np.pad(skin,((0,0),(1,1),(1,1)),mode='edge')
    gx=b[:,1:-1,2:]-b[:,1:-1,:-2];gy=b[:,2:,1:-1]-b[:,:-2,1:-1]
    mag=np.sqrt(.000001+gx*gx+gy*gy)
    field=np.stack([np.maximum(0,(np.abs(gx*np.cos(i*np.pi/6)+gy*np.sin(i*np.pi/6))/mag-.85)/.15)*mag for i in range(6)],axis=3)
    field*=np.clip(filt(body,[.25,.5,.25])*2.,0,1)[:,:,:,None]
    w=[.15,.70,.15] if n==12 else [.10,.23,.34,.23,.10]
    field=sample(filt(field,w),grid(n),grid(n))
    # Contrast with an annulus on both sides of the edge. A cluttered patch
    # rich in all orientations should not be evidence for every contour.
    center=field/(field.sum(axis=(1,2,3),keepdims=True)+.000001)
    s=filt(field,[.18,.20,.24,.20,.18])
    surround=s/(s.sum(axis=(1,2,3),keepdims=True)+.000001)
    return center-.80*surround

def support(a):
    # A soft activity image: even weak narrow fingers disturb an otherwise
    # smooth background. Compress contrast to avoid discarding them.
    d=a-filt(a,[.12,.23,.30,.23,.12]);e=filt(np.abs(d),[.25,.50,.25])
    return (e/(e+.015))[:,:,:,None]

def occupancy(body,skin,n):
    w=[.12,.23,.30,.23,.12]
    field=(.74*filt(skin,w)+.26*filt(body,w))[:,:,:,None]
    return sample(field,grid(n),grid(n))

def ridges(a):
    # Hessian directional curvature: even-symmetric line filters complement
    # first-order edges, specifically when the two edges of a digit are close.
    planes=[]
    for h in (1,2):
        sm=filt(a,[.15,.70,.15] if h==1 else [.06,.24,.40,.24,.06])
        b=np.pad(sm,((0,0),(h,h),(h,h)),mode='edge');n=28
        xx=b[:,h:h+n,:n]+b[:,h:h+n,2*h:]-2*sm
        yy=b[:,:n,h:h+n]+b[:,2*h:,h:h+n]-2*sm
        xy=(b[:,2*h:,2*h:]+b[:,:n,:n]-b[:,2*h:,:n]-b[:,:n,2*h:])*.25
        level=np.sqrt(.0004+filt(xx**2+yy**2+2*xy**2,[.12,.23,.30,.23,.12]))
        for k in range(6):
            c=np.cos(np.pi*k/6);s=np.sin(np.pi*k/6)
            v=(c*c*xx+2*s*c*xy+s*s*yy)/level**.73
            perp=(s*s*xx-2*s*c*xy+c*c*yy)/level**.73
            ridge=np.maximum(0,np.abs(v)-.65*np.abs(perp))
            planes.append(ridge);planes.append(ridge*np.sign(v)*.72)
    return filt(np.stack(planes,axis=3).astype(np.float32),[.22,.56,.22])

def ridgevec(m,dy,dx,sc):
    t=np.arange(7,dtype=np.float32)*3.5+3.-13.5
    v=sample(m,13.5+dy+t*sc,13.5+dx+t*sc)
    return (v/np.sqrt(.000001+(v*v).sum(axis=(1,2,3),keepdims=True))).reshape(len(m),-1)

def outline(shape):
    # Unit-mass angular histograms inside the putative foreground, measured
    # about its own centroid. Coarse rings describe the palm; thin distant
    # sectors describe the radial ordering of the fingertips and thumb.
    b=len(shape);z=shape.reshape(b,784)
    mass=np.maximum(.000001,z.sum(axis=1,keepdims=True));p=z/mass
    t=np.arange(28,dtype=np.float32)-13.5
    x=np.broadcast_to(t[None,:],(28,28)).reshape(1,784)
    y=x.reshape(28,28).T.reshape(1,784)
    xx=x-(x*p).sum(axis=1,keepdims=True);yy=y-(y*p).sum(axis=1,keepdims=True)
    rr=np.sqrt(xx*xx+yy*yy);scale=np.sqrt(.5+(p*rr**2).sum(axis=1,keepdims=True))
    rr=rr/scale; ang=(np.arctan2(yy,xx)+np.pi)*(8/np.pi)
    ai=ang.astype(np.int32);rem=ang-ai;idx=np.arange(b)[:,None]*16
    desc=[]
    for j in range(5):
        w=p*np.maximum(0.,1.-np.abs(rr*2-j))*((1.+j)*.5)
        aa=np.bincount((idx+ai%16).ravel(),weights=(w*(1-rem)).ravel(),minlength=b*16)
        bb=np.bincount((idx+(ai+1)%16).ravel(),weights=(w*rem).ravel(),minlength=b*16)
        h=(aa+bb).reshape(b,16);desc.append(.7*h+.15*(np.roll(h,1,1)+np.roll(h,-1,1)))
    v=np.concatenate(desc,axis=1)
    # Hellinger affinity, with no data-dependent population normalization.
    v=np.sqrt(np.maximum(0.,v));return (v/np.sqrt(.000001+(v*v).sum(axis=1,keepdims=True))).astype(np.float32)

def foreground(x):
    # Smooth backgrounds are common, even when they are not uniform. Compare
    # each pixel with a plane extrapolated from a quiet upper-side patch. Keep
    # a background estimate from each side, deciding between them per image.
    b=len(x);y,z=np.mgrid[0:28,0:28].astype(np.float32)
    activity=support(x)[:,:,:,0];cent=np.exp(-.5*((z-13.5)/7.5)**2-.5*((y-15)/9.)**2)
    smooth=filt(x,[.2,.6,.2]);masks=[];good=[]
    for left in (True,False):
        corners=[(0,0),(0,5),(5,0),(10,0)] if left else [(0,23),(0,18),(5,23),(10,23)]
        noise=[];planes=[]
        for j,i in corners:
            patch=smooth[:,j:j+5,i:i+5]
            noise.append(activity[:,j:j+5,i:i+5].mean(axis=(1,2)))
            vx=np.clip(np.median(patch[:,:,1:]-patch[:,:,:-1],axis=(1,2)),-.014,.014)
            vy=np.clip(np.median(patch[:,1:]-patch[:,:-1],axis=(1,2)),-.014,.014)
            planes.append(np.median(patch,axis=(1,2))[:,None,None]+(z-i-2)*vx[:,None,None]+(y-j-2)*vy[:,None,None])
        best=np.argmin(np.stack(noise,axis=1),axis=1)
        model=np.stack(planes,axis=1)[np.arange(b),best]
        delta=np.abs(smooth-model);detail=activity
        def grow(v):
            p=np.pad(v,((0,0),(1,1),(1,1)),mode='edge')
            return np.maximum.reduce([p[:,i:i+28,j:j+28] for i,j in ((0,1),(1,0),(2,1),(1,2),(1,1))])
        # Interior support from both tone and texture; morphological closing
        # keeps weakly contrasted gaps between knuckles in the same hand.
        m=np.maximum(np.clip((delta-.018)/.105,0,1),np.clip((detail-.27)/.30,0,1))
        m=-grow(-grow(m));m=filt(m,[.2,.6,.2]);m=np.clip((m-.20)/.64,0,1)
        mass=m.sum(axis=(1,2));central=(m*cent).sum(axis=(1,2))/np.maximum(1,mass)
        reliability=np.clip(central*2-.46,0,1)*np.clip(1.50-mass/440,0,1)*np.clip(mass/105,0,1)
        masks.append(m);good.append(reliability)
    rel=np.stack(good,axis=1);chosen=np.argmax(rel,axis=1)
    m=np.stack(masks,axis=1)[np.arange(b),chosen]
    return outline(m),rel.max(axis=1),m,activity

bankC=CONSTANTS['coarse'].astype(np.float32)
bankF=CONSTANTS['fine'].astype(np.float32)
bankS=CONSTANTS['shape']
bankR=CONSTANTS['ridge']

def classify(x):
    fine,coarse,oriented=features(x)
    shape,conf,mask,act=foreground(x)
    act=act[:,:,:,None];inv=1/(.001+act.sum(axis=(1,2),keepdims=True))
    mass=(mask.sum(axis=(1,2))+.001)[:,None,None,None]
    evidence=act*inv-1/784+.85*conf[:,None,None,None]*(mask[:,:,:,None]/mass-1/784)
    rv=ridges(x);rc=[]
    fc=[];cc=[];ec=[];ac=[]
    for sc in (.77,.98,1.22):
        for dy in (-5.,-2.5,0.,2.5,5.):
            for dx in (-3.25,0.,3.25):
                def edged(n):
                    t=(grid(n)-13.5)*sc+13.5
                    return sample(oriented,t+dy,t+dx)
                cv=descript(coarse,7,dx,dy,sc)
                def activity(n):
                    t=(grid(n)-13.5)*sc+13.5
                    # Include activity in the denominator even outside the proposed crop.
                    # Center against a uniform image to correct for projected area.
                    h=sample(evidence,t+dy,t+dx)
                    return h*((4*sc)**2 if n==7 else (2*sc)**2)
                cc.append(np.concatenate([cv,edged(7),activity(7)],axis=3).reshape(len(x),-1))
                fc.append(descript(fine,12,dx,dy,sc));ec.append(edged(12));ac.append(activity(12))
                rc.append(ridgevec(rv,dy,dx,sc))
    rc=np.stack(rc,axis=1)
    c=np.stack(cc,axis=1);f=np.stack(fc,axis=1);e=np.stack(ec,axis=1);actf=np.stack(ac,axis=1)
    b,aligns,length=c.shape; rows=np.arange(b)[:,None]
    c=c.reshape(b*aligns,length)
    answer=np.empty((b,24),np.float32)
    for k in range(24):
        bank=bankC[k]; count=len(bank)
        shapeval=(shape@bankS[k].reshape(-1,80).T).reshape(b,count,2).max(axis=2)
        shapeval=conf[:,None]*shapeval
        s=(c@bank.T).reshape(b,aligns,count)
        s=(s+.68*shapeval[:,None,:]).reshape(b,-1)
        keep=20
        ids=np.argpartition(s,-keep,axis=1)[:,-keep:]
        coarseval=s[rows,ids];p=bankF[k,ids%count].reshape(-1,12,12,21)
        q=f[rows,ids//count].reshape(-1,12,12,14)
        uv=p[:,:,:,20:21];disk=.28+.72*uv
        # Normalize over the modeled hand and an attenuated background, not
        # the unweighted whole crop. Sleeves should not swamp fingertip detail.
        qn=(q*q*disk).sum(axis=(1,2,3),keepdims=True)
        pn=(p[:,:,:,:14]**2*disk).sum(axis=(1,2,3),keepdims=True)
        delta=disk/np.sqrt(.00001+qn*pn)
        ev=e[rows,ids//count].reshape(-1,12,12,6)
        boundary=(p[:,:,:,14:20]*ev).sum(axis=(1,2,3)).reshape(b,keep)
        coverage=(p[:,:,:,20:]*actf[rows,ids//count].reshape(-1,12,12,1)).sum(axis=(1,2,3)).reshape(b,keep)
        p=p[:,:,:,:14]*(.35+.65*delta)
        # Slightly elastic comparison: each of nine parts may move by one
        # fine cell. Share most of the score with rigid registration.
        def parts(q,p):
            return np.einsum('bhiwjc,bhiwjc->bhw',q.reshape(-1,3,4,3,4,14),p.reshape(-1,3,4,3,4,14),optimize=False)
        rigid=parts(q,p);best=rigid.copy()
        pad=np.pad(q,((0,0),(1,1),(1,1),(0,0)),mode='constant')
        for yy,xx in ((0,1),(2,1),(1,0),(1,2)):
            best=np.maximum(best,parts(pad[:,yy:yy+12,xx:xx+12],p))
        sim=(.52*rigid+.48*best).sum(axis=(1,2)).reshape(b,keep)
        ridgematch=(rc[rows,ids//count]*bankR[k,ids%count]).sum(axis=2)
        vals=.68*sim+.32*coarseval+.22*boundary+.25*coverage+.32*shapeval[rows,ids%count]+.28*ridgematch
        # Marginalize over several excellent alignments, damping an isolated
        # accidental edge match without averaging any template descriptors.
        votes=np.partition(vals,-4,axis=1)[:,-4:]
        answer[:,k]=.64*vals.max(axis=1)+.36*votes.mean(axis=1)
    return np.argmax(answer,axis=1)

def predict(x):
    # A fixed internal block size also fixes the BLAS computation for each
    # image regardless of the outer batch size.
    labels=np.empty(len(x),np.int64)
    for i in range(0,len(x),16):
        n=min(16,len(x)-i)
        a=np.zeros((16,28,28),np.float32);a[:n]=x[i:i+n,0]
        labels[i:i+n]=classify(a)[:n]
    return labels
