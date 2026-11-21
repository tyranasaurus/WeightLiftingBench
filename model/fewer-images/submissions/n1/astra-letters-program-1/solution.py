import numpy as np


def smooth(v):
    a = np.pad(v, ((0,0),(1,1),(1,1)), mode='edge')
    b = (a[:,:,:-2]+a[:,:,2:]+2*a[:,:,1:-1])*.25
    return (b[:,:-2]+b[:,2:]+2*b[:,1:-1])*.25


def describe(pic):
    v = (.5*pic + .5*smooth(pic)).astype(np.float32)
    a = np.pad(v, ((0,0),(1,1),(1,1)), mode='edge')
    dx = (a[:,1:-1,2:]-a[:,1:-1,:-2])*.5
    dy = (a[:,2:,1:-1]-a[:,:-2,1:-1])*.5
    mag = np.sqrt(dx*dx+dy*dy+1.e-12)
    # Soft saturation preserves weak interior creases without letting noise
    # dominate the flat part of the frame.
    avg = mag.mean(axis=(1,2),keepdims=True)
    amp = np.sqrt(mag/(mag+.9*avg+.004))*mag/(mag+.44*avg+.003)
    theta = np.arctan2(dy,dx)
    hog = []
    for j in range(6):
        c = np.maximum(1-np.abs((theta*(6/np.pi)-j+3)%6-3),0)
        hog.append(smooth(smooth(amp*c)))
    # Linear inhibition of the other orientations improves selectivity but
    # allows a weaker match with an edge of slightly different orientation.
    h = np.stack(hog,axis=-1)
    h = h-.11*np.sum(h,axis=-1,keepdims=True)
    mass = smooth(smooth(amp))
    # Additional channel: interiors that differ from the background plane.
    bg = backdiff(v)
    return np.concatenate((h,(mass-mass.mean(axis=(1,2),keepdims=True))[:,:,:,None]*.48,bg[:,:,:,None]*.30),axis=-1).astype(np.float32)


def backdiff(v):
    # Use robust upper-side samples, where the hand least often fills both
    # borders. Interpolate lighting from left to right, rejecting a bad side.
    l = np.median(v[:,:19,:2].reshape(len(v),-1),axis=1)[:,None,None]
    r = np.median(v[:,:19,-2:].reshape(len(v),-1),axis=1)[:,None,None]
    weight = np.linspace(0,1,28,dtype=np.float32)[None,None,:]
    d = np.abs(v-(l+(r-l)*weight))
    d = d/(d+.65*np.mean(d,axis=(1,2),keepdims=True)+.017)
    return d-np.mean(d,axis=(1,2),keepdims=True)


def unit(v):
    return v/np.sqrt(np.sum(v*v,axis=-1,keepdims=True)+1.e-10)


def coarse(h):
    # Seven-by-seven, 4-pixel cells.
    v = h.reshape(len(h),7,4,7,4,8).mean(axis=(2,4))
    return unit(v.reshape(len(h),-1))


def predict(x):
    out = np.empty(len(x),np.int32)
    if len(x)==0:
        return out
    # Fixed-size GEMM even for single-image calls; a row's answer must not
    # depend on the shape-dependent floating point path of a BLAS library.
    bank = CONSTANTS['coarse']
    fine = CONSTANTS['fine']
    count = len(bank)//24
    keep = 12
    batch = 16
    offsets = ((0,0),(-3,0),(3,0),(0,-3),(0,3))
    rows = len(offsets)
    for start in range(0,len(x),batch):
        m = min(batch,len(x)-start)
        planes = describe(x[start:start+m,0])
        border = np.pad(planes,((0,0),(3,3),(3,3),(0,0)),mode='edge')
        q = np.zeros((batch,rows,392),np.float32)
        fi = np.empty((m,rows,14,14,8),np.float32)
        for v,(dy,dx) in enumerate(offsets):
            part = border[:,3+dy:31+dy,3+dx:31+dx]
            q[:m,v] = coarse(part)
            fi[:,v] = part.reshape(m,14,2,14,2,8).mean(axis=(2,4))
        sim = (q.reshape(batch*rows,392)@bank.T).reshape(batch,rows,24,count)
        for k in range(m):
            s = sim[k]
            s[1:] -= .012
            # Choose a single best alignment for each individual template.
            simk = s.max(axis=0)
            ix = np.argpartition(simk,-keep,axis=1)[:,-keep:]
            ids = ix+np.arange(24)[:,None]*count
            view = s.reshape(rows,-1)[:,ids].argmax(axis=0).reshape(-1)
            ts = fine[ids.reshape(-1)].astype(np.float32)*(1/127.)
            ce = np.einsum('nhwc,nhwc->nhw',ts,ts).reshape(len(ts),7,2,7,2).sum(axis=(2,4))
            # For the shortlisted poses, allow individual palm and finger
            # patches to move slightly relative to the global alignment.
            pad = np.pad(fi[k],((0,0),(1,1),(1,1),(0,0)),mode='edge')
            central = np.empty_like(ce)
            best = np.empty_like(ce)
            for v in range(rows):
                mask = np.flatnonzero(view==v)
                if len(mask)==0:
                    continue
                sub = ts[mask]
                energy = ce[mask]+.045
                for oy,ox in ((0,0),(-1,0),(1,0),(0,1),(0,-1)):
                    part = pad[v,1+oy:15+oy,1+ox:15+ox]
                    ue = np.sum(part*part,axis=2).reshape(7,2,7,2).sum(axis=(1,3))
                    dot = np.einsum('nhwc,hwc->nhw',sub,part).reshape(len(sub),7,2,7,2).sum(axis=(2,4))
                    patch = dot/np.sqrt(energy*(ue+.045))
                    if oy==0 and ox==0:
                        best[mask] = patch
                        central[mask] = patch
                    else:
                        best[mask] = np.maximum(best[mask],patch-.09)
            weight = ce/(ce+.10)+.14
            refined = np.sum(weight*(.64*best+.36*central),axis=(1,2))/np.sum(weight,axis=(1,2))
            score = .52*np.take_along_axis(simk,ix,axis=1)+.48*refined.reshape(24,keep)
            top = np.partition(score,-3,axis=1)[:,-3:]
            out[start+k] = np.argmax(.73*top.max(axis=1)+.27*top.mean(axis=1))
    return out
