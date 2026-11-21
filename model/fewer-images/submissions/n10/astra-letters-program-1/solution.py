import math
import numpy as np

def blurone(a):
    b = np.pad(a, ((0,0),(1,1),(1,1)), mode='edge')
    c = b[:,:-2] + 2*b[:,1:-1] + b[:,2:]
    return (c[:,:,:-2] + 2*c[:,:,1:-1] + c[:,:,2:])*.0625

def box(a,r):
    p = np.pad(a, ((0,0),(r,r),(r,r)), mode='edge')
    p = np.pad(p, ((0,0),(1,0),(1,0)))
    p = p.cumsum(1).cumsum(2)
    k = 2*r+1
    return (p[:,k:,k:] - p[:,:-k,k:] - p[:,k:,:-k] + p[:,:-k,:-k])/(k*k)

def cells(a,k):
    return a.reshape(len(a), 28//k,k,28//k,k).mean((2,4))

def smoothcells(a):
    b = np.pad(a, ((0,0),(1,1),(1,1)),mode='edge')
    return (.8*b[:,1:-1,1:-1] + .05*(b[:,:-2,1:-1]+b[:,2:,1:-1]+b[:,1:-1,:-2]+b[:,1:-1,2:]))

def unitvec(a):
    a = a.reshape(len(a),-1)
    return a / np.sqrt((a*a).sum(1,keepdims=True)+1e-12)

def features(a, mask):
    # Local gradient strength, unoriented contour directions and signed
    # curvature. No pixel depends on other images in the batch.
    b = blurone(a)
    p = np.pad(b, ((0,0),(1,1),(1,1)), mode='edge')
    dx = p[:,1:-1,2:] - p[:,1:-1,:-2]
    dy = p[:,2:,1:-1] - p[:,:-2,1:-1]
    mag = np.sqrt(dx*dx+dy*dy+1e-12)
    local = np.maximum(box(mag,3),0)
    w = 1/np.sqrt(.045+local+.18*mag)
    edge = np.sqrt(np.maximum(mag*w,0))*np.minimum(mag/.028,1)
    inv = 1/mag
    gx,gy = dx*inv,dy*inv
    coarse = []
    fine = []
    for j in range(6):
        z = np.maximum(np.abs(math.cos(j*math.pi/6)*gx+math.sin(j*math.pi/6)*gy)-.68,0)*(1/.32)
        e = edge*z*z
        coarse.append(smoothcells(cells(e,4)))
        fine.append(smoothcells(cells(e,2)))
    # Per-cell normalization with a stabilizer retains the distinction
    # between smooth background and hand texture.
    h = np.stack(coarse,axis=1)
    h = h / np.sqrt(.027+(h*h).sum(1,keepdims=True))**.65
    f = np.stack(fine,axis=1)
    f = f / np.sqrt(.035+(f*f).sum(1,keepdims=True))**.62
    # Greyscale ridges and grooves, locally normalized; insensitive to a
    # linear illumination ramp. Keep both signs in their original location.
    d = b-box(b,3)
    ridge = np.clip(d/np.sqrt(.0016+np.maximum(box(d*d,3),0)), -2,2)
    s = unitvec(smoothcells(cells(ridge,2)))
    h = unitvec(h)
    f = unitvec(f)
    m = unitvec(cells(mask,2)-cells(mask,2).mean((1,2),keepdims=True))
    return h,f,s,m

def passage(a,v):
    # One scan in the minimum-barrier metric. A path may follow an oblique
    # edge: a background route does not have to be an axis-aligned ray.
    lo=a[:,0].copy()
    hi=lo.copy()
    for i in range(1,28):
        row=a[:,i]
        ls=np.stack((lo,np.concatenate((lo[:,:1],lo[:,:-1]),1),np.concatenate((lo[:,1:],lo[:,-1:]),1)),1)
        hs=np.stack((hi,np.concatenate((hi[:,:1],hi[:,:-1]),1),np.concatenate((hi[:,1:],hi[:,-1:]),1)),1)
        ls=np.minimum(ls,row[:,None]); hs=np.maximum(hs,row[:,None]); d=hs-ls
        sel=np.argmin(d,1)[:,None]
        lo=np.take_along_axis(ls,sel,1)[:,0]; hi=np.take_along_axis(hs,sel,1)[:,0]
        v[:,i]=np.minimum(v[:,i],hi-lo)

def foreground(a):
    b=blurone(a)
    v=np.ones_like(a)
    passage(b,v)
    passage(b.transpose(0,2,1),v.transpose(0,2,1))
    passage(b.transpose(0,2,1)[:,::-1],v.transpose(0,2,1)[:,::-1])
    v[:,0]*=.35
    # Texture is useful even at a border crossed by the wrist or fingertips.
    c=np.abs(b-box(b,2))
    e=np.maximum(box(c,1),0)
    return blurone(1-np.exp(-12*(v+.9*e)))

def contourfield(a,fg):
    # Steerable edge evidence with one pixel of registration tolerance. An
    # isolated boundary between two background objects is not very diagnostic.
    b=.7*a+.3*blurone(a)
    p=np.pad(b,((0,0),(1,1),(1,1)),mode='edge')
    gx=p[:,1:-1,2:]-p[:,1:-1,:-2]; gy=p[:,2:,1:-1]-p[:,:-2,1:-1]
    mag=np.sqrt(gx*gx+gy*gy+1e-12)
    lm=np.maximum(box(mag,3),0)
    contrast=mag/(.037+.55*lm+mag)
    contrast*=.75+.25*np.minimum(1,2*blurone(fg))
    fields=[]
    for j in range(6):
        rate=np.abs((math.cos(j*math.pi/6)*gx+math.sin(j*math.pi/6)*gy)/mag)
        z=contrast*np.maximum(0,rate*rate*2-1)**.8
        p=np.pad(z,((0,0),(1,1),(1,1)),mode='edge')
        near=np.maximum(np.maximum(p[:,1:-1,:-2],p[:,1:-1,2:]),np.maximum(p[:,:-2,1:-1],p[:,2:,1:-1]))
        diag=np.maximum(np.maximum(p[:,:-2,:-2],p[:,2:,:-2]),np.maximum(p[:,:-2,2:],p[:,2:,2:]))
        fields.append(np.maximum(z,np.maximum(.85*near,.75*diag)))
    return np.stack(fields,axis=1).reshape(len(a),-1)

def predict(x):
    answers = np.empty(len(x),np.int64)
    # Fixed-size matrix multiplies, even for a last partial block. Integers
    # are returned, not floating scores with a batch-dependent roundoff.
    stride = 64
    nclass = 24
    for start in range(0,len(x),stride):
        n = min(stride,len(x)-start)
        a = np.zeros((stride,28,28),np.float32)
        a[:n] = x[start:start+n,0]
        fg=foreground(a)
        h,f,s,m = features(a,fg)
        field=contourfield(a,fg)
        q=np.concatenate((h*.70,m*.30),1)
        coarse=(q @ searchbank.T).reshape(stride,nclass,-1)
        for i in range(n):
            # Search independently within each letter, then verify high-
            # scoring views at finer resolution. All rows are individuals.
            top=np.argpartition(coarse[i],-56,axis=1)[:,-56:]
            idx=(top + baseidx).reshape(-1)
            ed=field[i][CONSTANTS['points'][idx]].mean(1)
            cc=np.take_along_axis(coarse[i],top,axis=1).ravel()
            pre=(.63*cc+.37*ed).reshape(nclass,-1)
            subset=np.argpartition(pre,-12,axis=1)[:,-12:]
            choose=(subset+np.arange(nclass)[:,None]*56).ravel()
            idx=idx[choose]; cc=cc[choose]; ed=ed[choose]
            fd=CONSTANTS['fine'][idx].astype(np.float32)
            sd=CONSTANTS['shade'][idx].astype(np.float32)
            ff=(fd*f[i]).sum(1)/8192
            ss=(sd*s[i]).sum(1)/8192
            sc=(.23*cc+.37*ff+.29*ed+.11*ss).reshape(nclass,-1)
            best=np.sort(sc,axis=1)[:,-3:]
            answers[start+i]=np.argmax(best[:,2]*.65+best[:,1]*.25+best[:,0]*.10)
    return answers

searchbank = CONSTANTS['search'].astype(np.float32)/8192
baseidx = np.arange(24)[:,None]*(len(searchbank)//24)
