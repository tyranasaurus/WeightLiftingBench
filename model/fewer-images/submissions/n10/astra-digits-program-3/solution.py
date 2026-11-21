import numpy as np

axis = np.arange(28,dtype=np.float32)
wide = np.arange(-8,36,dtype=np.float32)

def quantiles(p):
    cs = np.cumsum(p,axis=1)
    def loc(f):
        v = cs[:,-1:] * f
        j = np.minimum(np.sum(cs<v,axis=1),p.shape[1]-1)
        n = np.arange(len(p))
        return (j + (v[:,0]-cs[n,j])/np.maximum(.0001,p[n,j])+.5).astype(np.float32)
    return loc(.008),loc(.992)

def canon(a):
    n = np.arange(len(a))
    a = a/np.maximum(.001,np.max(a,axis=(1,2),keepdims=True))
    rows = a.sum(axis=2); cols=a.sum(axis=1)
    mass = np.maximum(.001,rows.sum(1))
    cy = (rows*axis).sum(1)/mass
    cx = (cols*axis).sum(1)/mass
    yd = axis-cy[:,None]; xd = axis-cx[:,None]
    shear = np.clip((a*(yd[:,:,None]*xd[:,None,:])).sum(axis=(1,2))/np.maximum(.001,(rows*yd**2).sum(1)),-1.4,1.4)
    offset = yd*shear[:,None]
    grid = wide+offset[:,:,None]; j = np.floor(grid).astype(np.int32); f=grid-j
    yy=np.arange(28)
    proj = ((1-f)*a[n[:,None,None],yy[:,None],np.clip(j,0,27)] + f*a[n[:,None,None],yy[:,None],np.clip(j+1,0,27)]).sum(1).astype(np.float32)
    l,r=quantiles(proj); l-=8; r-=8
    top,bot=quantiles(rows)
    height=np.maximum(1.,bot-top); width=np.maximum(1.,r-l)
    ratio=np.clip(width/height,.06,3.5)
    nh = 21./np.maximum(1.,ratio/1.45)**.28
    nw = np.clip(nh*ratio*(.73/np.maximum(.15,ratio))**.64,2.7,24.)
    sy = (axis-13.5)*(height/nh)[:,None]+((top+bot)*.5)[:,None]
    sx = (axis-13.5)*(width/nw)[:,None]+((l+r)*.5)[:,None]
    sx = sx[:,None,:]+shear[:,None,None]*(sy-cy[:,None])[:,:,None]
    py=np.floor(sy).astype(np.int32); px=np.floor(sx).astype(np.int32)
    fy=(sy-py).astype(np.float32)[:,:,None]; fx=(sx-px).astype(np.float32)
    p0=np.clip(py,0,27)[:,:,None]; p1=np.clip(py+1,0,27)[:,:,None]
    q0=np.clip(px,0,27); q1=np.clip(px+1,0,27)
    im=(1-fy)*((1-fx)*a[n[:,None,None],p0,q0]+fx*a[n[:,None,None],p0,q1])+fy*((1-fx)*a[n[:,None,None],p1,q0]+fx*a[n[:,None,None],p1,q1])
    meta=np.stack((np.log(ratio),shear,np.log(np.clip(mass/(width*height),.025,2))),axis=1)
    return im,meta

def kernel(pos,sigma):
    v=np.exp(-.5*((np.asarray(pos,dtype=np.float32)[:,None]-axis)/sigma)**2)
    return v/v.sum(1,keepdims=True)

fineker=kernel(1.5+2*np.arange(13),1.10)
coarseker=kernel(1.5+4*np.arange(7),2.05)

def pool(k,f):
    return (k@f)@k.T

def trapped(a,level):
    space=a<level
    exterior=space.copy(); exterior[:,1:-1,1:-1]=False
    for i in range(58):
        v=exterior.copy()
        v[:,1:] |= exterior[:,:-1]; v[:,:-1] |= exterior[:,1:]
        v[:,:,1:] |= exterior[:,:,:-1]; v[:,:,:-1] |= exterior[:,:,1:]
        v &= space
        if i%8==7 and np.array_equal(v,exterior): break
        exterior=v
    return space & ~exterior

def features(raw):
    a,meta=canon(raw)
    v=np.pad(a,((0,0),(1,1),(1,1)))
    im=(4*a+v[:,1:-1,:-2]+v[:,1:-1,2:]+v[:,:-2,1:-1]+v[:,2:,1:-1])*.125
    v=np.pad(im,((0,0),(1,1),(1,1)))
    gx=v[:,1:-1,2:]-v[:,1:-1,:-2]; gy=v[:,2:,1:-1]-v[:,:-2,1:-1]
    floor=.66*np.sqrt(gx**2+gy**2)
    cparts=[]; fparts=[]
    for z in (gx,.70710678*(gx+gy),gy,.70710678*(gy-gx)):
        for sign in (1.,-1.):
            h=np.maximum(0.,(sign*z-floor)*2.94)
            cparts.append(np.sqrt(pool(coarseker,h)+.008)-.08944272)
            fparts.append(np.sqrt(pool(fineker,h)+.008)-.08944272)
    cparts.append(.55*pool(coarseker,a)); fparts.append(.55*pool(fineker,a))
    # A whitespace component closed to the exterior, or open to one side only.
    hole=.5*(trapped(a,.28).astype(np.float32)+trapped(a,.52))
    cparts.append(.50*pool(coarseker,hole))
    bars = (np.maximum.accumulate(a,2),np.maximum.accumulate(a[:,:,::-1],2)[:,:,::-1],
            np.maximum.accumulate(a,1),np.maximum.accumulate(a[:,::-1],1)[:,::-1])
    white=1-a
    m=np.minimum(np.minimum(bars[0],bars[1]),np.minimum(bars[2],bars[3]))
    cparts.append(.28*pool(coarseker,white*m))
    for j in range(4):
        b=np.minimum(np.minimum(bars[(j+1)%4],bars[(j+2)%4]),bars[(j+3)%4])
        cparts.append(.34*pool(coarseker,white*b*(1-bars[j])))
    coarse=np.stack(cparts,axis=1).reshape(len(a),-1)
    coarse/=np.maximum(.0001,np.sqrt((coarse**2).sum(1,keepdims=True)))
    fine=np.stack(fparts,axis=3)
    fine/=np.maximum(.0001,np.sqrt((fine**2).sum(axis=(1,2,3),keepdims=True)))
    return coarse,np.clip(np.rint(1900*fine),0,255).astype(np.uint8),meta

nper=CONSTANTS['drawings'].shape[1]
num=10*nper
rawbank=CONSTANTS['drawings'].reshape(num,28,28)
bank=np.empty((num,15*49),np.float32)
inkbank=np.empty((num,13,13,9),np.uint8)
measure=np.empty((num,3),np.float32)
for start in range(0,num,64):
    section=slice(start,min(start+64,num))
    bank[section],inkbank[section],measure[section]=features(rawbank[section].astype(np.float32)/255.)

def patchmean(e):
    # Compare overlapping neighbourhoods, not isolated gradient samples.
    p=np.pad(e,((0,0),(0,0),(1,1),(1,1)))
    v=.36*e+.12*(p[:,:,:-2,1:-1]+p[:,:,2:,1:-1]+p[:,:,1:-1,:-2]+p[:,:,1:-1,2:])
    return v+.04*(p[:,:,:-2,:-2]+p[:,:,2:,:-2]+p[:,:,:-2,2:]+p[:,:,2:,2:])

def elastic(q,ids):
    b=len(q); k=ids.shape[1]; area=13
    temp=inkbank[ids].astype(np.float32)*(1./1900.)
    pad=np.pad(q.astype(np.float32)*(1./1900.),((0,0),(1,1),(1,1),(0,0)))
    best=np.full((b,k,area,area),4.,np.float32); reverse=best.copy()
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            v=pad[:,1+dy:14+dy,1+dx:14+dx,:]
            dif=temp-v[:,None,:,:,:]
            err=np.einsum('bkhwc,bkhwc->bkhw',dif,dif)
            cost=patchmean(err)+(.028/(13*13))*(dx*dx+dy*dy)
            best=np.minimum(best,cost)
            ty=slice(max(0,dy),min(13,13+dy)); tx=slice(max(0,dx),min(13,13+dx))
            sy=slice(max(0,-dy),min(13,13-dy)); sx=slice(max(0,-dx),min(13,13-dx))
            reverse[:,:,ty,tx]=np.minimum(reverse[:,:,ty,tx],cost[:,:,sy,sx])
    return .5*(best+reverse).sum(axis=(2,3))

def classify(a):
    desc,q,meta=features(a)
    b=len(q)
    # Use a fixed shape for the matrix product, independent of the outer batch.
    query=np.zeros((48,desc.shape[1]),np.float32); query[:b]=desc
    dist=2.-2.*(query@bank.T)[:b]
    extra=(.032*np.minimum(1.6,abs(meta[:,0:1]-measure[:,0]))**2+
           .013*np.minimum(1.6,abs(meta[:,1:2]-measure[:,1]))**2+
           .021*np.minimum(1.4,abs(meta[:,2:3]-measure[:,2]))**2)
    dist+=extra
    d=dist.reshape(b,10,nper)
    # Keep several independent exemplars per promising class, rather than
    # allowing a common pen style to crowd out a rarer competing hypothesis.
    cls=np.argsort(d.min(2),axis=1,kind='stable')[:,:4]
    n=np.arange(b)[:,None]
    dv=d[n,cls]
    ind=np.argpartition(dv,7,axis=2)[:,:,:8]+nper*cls[:,:,None]
    ind=ind.reshape(b,-1)
    cost=dist[n,ind]*.50+extra[n,ind]*.50
    for st in range(0,b,8):
        ss=slice(st,st+8)
        cost[ss]+=elastic(q[ss],ind[ss])
    scores=np.sort(cost.reshape(b,4,8),axis=2)
    loss=.68*scores[:,:,0]+.22*scores[:,:,1]+.10*scores[:,:,2]
    return cls[np.arange(b),np.argmin(loss,axis=1)]

def predict(x):
    result=np.empty(len(x),np.int64)
    for st in range(0,len(x),48):
        result[st:st+48]=classify(x[st:st+48,0])
    return result
