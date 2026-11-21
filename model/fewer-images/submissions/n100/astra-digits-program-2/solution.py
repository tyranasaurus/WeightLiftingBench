import numpy as np

def masslimits(p, start):
    # Quantile of a piecewise uniform density (one bin per pixel).
    a = np.cumsum(p, axis=1)
    m = a[:, -1:]
    out = []
    for q in (.006, .994):
        t = m*q
        i = np.minimum(np.sum(a < t, axis=1), p.shape[1]-1)
        r = np.arange(len(p))
        f = 1 - (a[r,i] - t[:,0]) / np.maximum(p[r,i], 1.e-8)
        out.append(start + i - .5 + np.clip(f, 0, 1))
    return out

def canonical(img):
    img = np.asarray(img, dtype=np.float32)
    b = len(img)
    g = np.arange(28, dtype=np.float32)
    v = g[None,:,None]
    u = g[None,None,:]
    p = img.sum(axis=2)
    m = np.maximum(p.sum(axis=1), 1.e-6)
    ym = (p * g).sum(axis=1)/m
    xm = (img.sum(axis=1) * g).sum(axis=1)/m
    yc = v - ym[:,None,None]
    xc = u - xm[:,None,None]
    tilt = np.clip((img*yc*xc).sum(axis=(1,2)) / np.maximum((img*yc*yc).sum(axis=(1,2)), 1.e-6), -1.15, 1.15)
    shear = .96*tilt
    ux = u - shear[:,None,None]*yc
    # Project ink onto the de-slanted x axis, splatting between adjacent bins.
    h = 33 + ux
    ix = h.astype(np.int32)
    f = h-ix
    pos = ix + 94*np.arange(b)[:,None,None]
    hist = np.bincount(pos.ravel(), (img*(1-f)).ravel(), minlength=b*94)
    hist += np.bincount((pos+1).ravel(), (img*f).ravel(), minlength=b*94)
    l,r = masslimits(hist.reshape(b,94), -33)
    t,d = masslimits(p, 0)
    w = np.maximum(r-l, 1.)
    ht = np.maximum(d-t, 1.)
    aspect = np.clip(w/ht, .06, 3.)
    # Partial rather than full aspect normalization keeps a bare 1 narrow.
    tgtw = np.minimum(21.5, np.minimum(20.5*aspect*1.55, 16*(aspect/.72)**.46))
    sy = (v-13.5) * (ht/20.5)[:,None,None] + ((t+d)/2)[:,None,None]
    sx = (u-13.5) * (w/tgtw)[:,None,None] + ((r+l)/2)[:,None,None] + (sy-ym[:,None,None])*shear[:,None,None]
    ix = np.floor(sx).astype(np.int32)
    iy = np.floor(sy).astype(np.int32)
    fx = (sx-ix).astype(np.float32)
    fy = (sy-iy).astype(np.float32)
    j = np.arange(b)[:,None,None]
    def sample(yy,xx):
        return img[j,np.clip(yy,0,27),np.clip(xx,0,27)]*((xx>=0)&(xx<28)&(yy>=0)&(yy<28))
    z = (1-fy)*((1-fx)*sample(iy,ix)+fx*sample(iy,ix+1)) + fy*((1-fx)*sample(iy+1,ix)+fx*sample(iy+1,ix+1))
    z = np.minimum(1., z / np.maximum(z.max(axis=(1,2),keepdims=True), 1.e-6))
    geom = np.stack((145*tilt, 155*np.log(aspect)),axis=1)
    return z, np.rint(geom).astype(np.int16)

def ringbits(sk):
    p = np.pad(sk,((0,0),(1,1),(1,1)),mode='constant')
    return (p[:,:-2,1:-1]+2*p[:,:-2,2:]+4*p[:,1:-1,2:]+8*p[:,2:,2:]
            +16*p[:,2:,1:-1]+32*p[:,2:,:-2]+64*p[:,1:-1,:-2]+128*p[:,:-2,:-2])

def make_tables():
    bits = (np.arange(256)[:,None] >> np.arange(8)[None,:]) & 1
    nextbits = np.roll(bits,-1,axis=1)
    arcs = np.sum((bits==0)&(nextbits==1),axis=1)
    n = bits.sum(axis=1)
    removable = (arcs==1)&(n>=2)&(n<=6)
    top,right,bot,left = bits[:,::2].T
    keep1 = ~(removable & ((top*right*bot)==0) & ((right*bot*left)==0))
    keep2 = ~(removable & ((top*right*left)==0) & ((top*bot*left)==0))
    return keep1,keep2,n,arcs

tables = make_tables()

def ridgefields(z):
    # Parallel topology-preserving thinning. Compute at two isophotes so a
    # nearly joined stroke isn't forced into an irrevocable topology decision.
    total = np.zeros((len(z),7,28,28),np.float32)
    n,arcs = tables[2:]
    m = np.arange(28)
    smooth = np.exp(-.5*((m[:,None]-m[None,:])/1.25)**2).astype(np.float32)
    for threshold in (.30,.58):
        sk = (z>threshold).astype(np.uint8)
        for step in range(20):
            old = sk
            sk = sk*tables[0][ringbits(sk)]
            sk = sk*tables[1][ringbits(sk)]
            if np.array_equal(sk,old):
                break
        bits = ringbits(sk)
        # Estimate tangent from a tensor of links on the lattice skeleton.
        # Use diagonal links only if not redundant with an axial neighbour.
        vertical = ((bits&1)>0).astype(np.float32) + ((bits&16)>0)
        horiz = ((bits&4)>0).astype(np.float32) + ((bits&64)>0)
        desc = ((bits&28)==8).astype(np.float32) + ((bits&193)==128)
        asc = ((bits&7)==2).astype(np.float32) + ((bits&112)==32)
        tx = smooth @ ((horiz-vertical)*sk) @ smooth.T
        ty = smooth @ ((desc-asc)*sk*1.41421356) @ smooth.T
        disp = smooth @ ((horiz+vertical+(desc+asc)*1.41421356)*sk) @ smooth.T
        # cos(2*theta), sin(2*theta) parameterize unoriented tangents.
        fac = sk / np.maximum(np.abs(tx)+np.abs(ty),.25*disp+1.e-5)
        dirs = [np.maximum(0,v)*fac for v in (tx,ty,-tx,-ty)]
        ends = (n[bits]==1)*sk
        forks = (arcs[bits]>=3)*sk
        total += np.stack(dirs+[sk,ends,forks],axis=1)
    return total*.5

def fields(z):
    # Blur perpendicular to each derivative as in a Sobel filter.
    p = np.pad(z,((0,0),(1,1),(1,1)),mode='constant')
    v = p[:,:-2,:] + 2*p[:,1:-1,:] + p[:,2:,:]
    h = p[:,:,:-2] + 2*p[:,:,1:-1] + p[:,:,2:]
    gx = (v[:,:,2:] - v[:,:,:-2])*.25
    gy = (h[:,2:,:] - h[:,:-2,:])*.25
    diagon = .70710678
    a = (gx+gy)*diagon
    b = (gy-gx)*diagon
    grads = []
    for dot, cross in ((gx,gy),(a,b),(gy,gx),(b,a),(-gx,gy),(-a,b),(-gy,gx),(-b,a)):
        grads.append(np.maximum(dot-np.abs(cross),0))
    # Distinguish inside a ring from inside a bay facing each compass point.
    ink = np.minimum(1., 1.4*z)
    l = np.maximum.accumulate(ink,axis=2)
    r = np.maximum.accumulate(ink[:,:,::-1],axis=2)[:,:,::-1]
    t = np.maximum.accumulate(ink,axis=1)
    d = np.maximum.accumulate(ink[:,::-1,:],axis=1)[:,::-1,:]
    lr = np.minimum(l,r)
    td = np.minimum(t,d)
    mid = np.minimum(lr,td)
    bg = 1-ink
    bay = [(a-ink)*bg for a in (mid,lr,td)]
    bay += [(a-mid)*bg for a in (np.minimum(td,r),np.minimum(td,l),np.minimum(lr,d),np.minimum(lr,t))]
    # Soft flood fill: the value is the least ink one has to cross to escape.
    cover = np.minimum(1., z*1.5)
    fill = np.ones_like(z)
    fill[:,0,:] = cover[:,0,:]
    fill[:,-1,:] = cover[:,-1,:]
    fill[:,:,0] = cover[:,:,0]
    fill[:,:,-1] = cover[:,:,-1]
    for step in range(48):
        n = np.minimum(fill[:,2:,1:-1],fill[:,:-2,1:-1])
        n = np.minimum(n,np.minimum(fill[:,1:-1,:-2],fill[:,1:-1,2:]))
        n = np.maximum(cover[:,1:-1,1:-1],n)
        if step >= 16 and step%4 == 0 and np.array_equal(n,fill[:,1:-1,1:-1]):
            break
        fill[:,1:-1,1:-1] = n
    hole = np.maximum(fill-cover,0.)
    return np.concatenate((np.stack(grads+[z]+bay+[hole],axis=1),ridgefields(z)),axis=1)

def poolmat(shift):
    ds = (np.arange(6)[:,None]*4+3.5+shift) - np.arange(28)[None,:]
    a = np.exp(-.5*(ds/1.85)**2)
    return (a/a.sum(axis=1,keepdims=True)).astype(np.float32)

def norms(parts):
    return [np.sqrt((q*q).sum(axis=(1,2,3),keepdims=True)+floor) for q,floor in zip(parts,(.1,.1,.1,1.,1.1,.05,.05,.10,.13))]

def packed(maps,py,px,den):
    a = (py @ maps) @ px.T
    signed = np.sqrt(np.maximum(a[:,:8],0.))
    orient = np.sqrt(np.maximum(a[:,:4]+a[:,4:8],0.))
    # Keep some magnitude information for enclosed spaces, including absence.
    parts = [signed,orient,np.sqrt(a[:,8:9]),np.maximum(a[:,9:16],0.)**.8,np.sqrt(np.maximum(a[:,16:17],0.)),
             np.sqrt(a[:,17:21]),np.sqrt(a[:,21:22]),np.sqrt(a[:,22:23]),np.sqrt(a[:,23:24])]
    if den is None:
        den = norms(parts)
    weights = (.69,.33,.39,.435,.28,.44,.28,.17,.16)
    blocks = [a*((1024*w)/d) for a,d,w in zip(parts,den,weights)]
    out = np.concatenate(blocks,axis=1).transpose(0,2,3,1)
    return np.ascontiguousarray(np.rint(out).astype(np.int16)),den

# Each row is ONE rendered glyph. Its entries and norm depend on no other glyph.
local = CONSTANTS['glyphs'].reshape(-1,6,6,28)
meta = CONSTANTS['geometry'].reshape(-1,2)
full = np.concatenate((local.reshape(len(local),-1),meta),axis=1).astype(np.float32)
lengths = np.sum(full*full,axis=1)
stylecost = CONSTANTS['stylecost'].reshape(-1)

def classify(img):
    z,g = canonical(img)
    maps = fields(z)
    center = poolmat(0)
    a,den = packed(maps,center,center,None)
    q = np.concatenate((a.reshape(len(a),-1),g),axis=1).astype(np.float32)
    # Dot products are small integers exactly representable as float32.
    dist = lengths[None,:] - 2*(q @ full.T) + stylecost[None,:]
    dclass = dist.reshape(len(a),10,-1)
    k = 16
    order = np.argpartition(dclass,k-1,axis=2)[:,:,:k]
    idx = (order + np.arange(10)[None,:,None]*dclass.shape[2]).reshape(len(a),10*k)
    selected = local[idx].astype(np.float32)
    # Elastic match. Consider four placements of the boundaries between
    # patches so a pen tip at a boundary is not arbitrarily cut in two.
    fixed = None
    best = None
    for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
        if dx == 0 and dy == 0:
            shifted = a
        else:
            shifted,unused = packed(maps,poolmat(1.65*dy),poolmat(1.65*dx),den)
        dif = selected - shifted[:,None,:,:,:]
        cell = np.einsum('bijhc,bijhc->bijh',dif,dif)
        # Sum over every overlapping 2x2 patch, including clipped ones
        # around the border. Each cell participates exactly four times.
        p = np.pad(cell,((0,0),(0,0),(1,1),(1,1)),mode='constant')
        region = p[:,:,:-1,:-1]+p[:,:,1:,:-1]+p[:,:,:-1,1:]+p[:,:,1:,1:]
        if fixed is None:
            fixed = cell.sum(axis=(2,3))
            best = region
        else:
            area = np.array([1,2,2,2,2,2,1],np.float32)
            best = np.minimum(best, region + 300*(dx*dx+dy*dy)*(area[:,None]*area))
    final = .25*fixed + (.75/4)*np.sum(best,axis=(2,3))
    delta = meta[idx].astype(np.float32)-g[:,None,:]
    final += np.sum(delta*delta,axis=2) + stylecost[idx]
    ranks = np.sort(final.reshape(len(a),10,k),axis=2)
    return np.argmin(.65*ranks[:,:,0]+.25*ranks[:,:,1]+.10*ranks[:,:,2],axis=1)

def predict(x):
    labels = np.empty(len(x), dtype=np.int64)
    for j in range(0,len(x),48):
        labels[j:j+48] = classify(x[j:j+48,0])
    return labels
