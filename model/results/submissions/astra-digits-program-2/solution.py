import numpy as np

axis28 = np.arange(28, dtype=np.float32)
axis32 = np.arange(32, dtype=np.float32)

def blur(a):
    b = np.pad(a, ((0,0),(1,1),(1,1)))
    c = (b[:,2:] + 2*b[:,1:-1] + b[:,:-2])*.25
    return (c[:,:,2:] + 2*c[:,:,1:-1] + c[:,:,:-2])*.25

def bounds(mass):
    mass = mass / np.maximum(mass.sum(1,keepdims=True),1e-8)
    # A soft extent: exceptionally light incidental marks do not fix the scale.
    cs = np.cumsum(mass,axis=1)
    idx = np.minimum((cs < .007).sum(1),27)
    lo = idx-.5 + 1 - np.clip((cs[np.arange(len(idx)),idx]-.007) / np.maximum(1e-8,mass[np.arange(len(idx)),idx]),0,1)
    idx = np.minimum((cs < .993).sum(1),27)
    hi = idx-.5 + 1 - np.clip((cs[np.arange(len(idx)),idx]-.993) / np.maximum(1e-8,mass[np.arange(len(idx)),idx]),0,1)
    return lo.astype(np.float32),hi.astype(np.float32)

def normalize(a):
    n = len(a)
    r = np.arange(n)[:,None,None]
    den = np.maximum(a.sum((1,2)),1e-6)
    rows = a.sum(2)
    my = (rows*axis28).sum(1)/den
    dy = axis28[None,:]-my[:,None]
    xy = (a * (axis28[None,None,:]-13.5)).sum(2)
    cov = (xy*dy).sum(1)
    var = (rows*dy**2).sum(1)
    shear = .87*np.clip(cov/np.maximum(var,1e-6),-1.35,1.35)
    # Re-sample just to measure the horizontal extent after shearing. The
    # final image is taken directly from the input, not from this interpolant.
    sx = axis28[None,None,:] + (shear[:,None]*dy)[:,:,None]
    ix = np.floor(sx).astype(np.int32)
    f = sx-ix
    ry = np.arange(28)[None,:,None]
    z = a[r,ry,np.clip(ix,0,27)]*(1-f) + a[r,ry,np.clip(ix+1,0,27)]*f
    lx,hx = bounds(z.sum(1))
    ly,hy = bounds(rows)
    w,h = np.maximum(2,hx-lx),np.maximum(2,hy-ly)
    sy = h/24
    # Partial aspect normalization: allow width to vary less than in the input,
    # without stretching a vertical pen stroke into the shape of a block.
    sw = np.sqrt(np.maximum(w/18.0,sy*.33)*sy)
    yy = ((ly+hy)*.5)[:,None] + sy[:,None]*(axis32-15.5)
    xx = ((lx+hx)*.5)[:,None,None] + sw[:,None,None]*(axis32-15.5) + (shear[:,None]*(yy-my[:,None]))[:,:,None]
    iy = np.floor(yy).astype(np.int32)[:,:,None]
    ix = np.floor(xx).astype(np.int32)
    fy = (yy[:,:,None]-iy).astype(np.float32)
    fx = (xx-ix).astype(np.float32)
    b = np.pad(a,((0,0),(1,1),(1,1)))
    yl = np.clip(iy+1,0,29)
    xl = np.clip(ix+1,0,29)
    yh = np.clip(iy+2,0,29)
    xh = np.clip(ix+2,0,29)
    return ((b[r,yl,xl]*(1-fx)+b[r,yl,xh]*fx)*(1-fy)+(b[r,yh,xl]*(1-fx)+b[r,yh,xh]*fx)*fy).astype(np.float32),np.stack((shear,w/h),1).astype(np.float32)

def bits(a):
    b = blur(a)
    p = np.pad(b,((0,0),(1,1),(1,1)))
    gx = p[:,1:-1,2:]-p[:,1:-1,:-2]
    gy = p[:,2:,1:-1]-p[:,:-2,1:-1]
    return b,gx,gy

def pooled(a,s):
    return a.reshape(len(a),32//s,s,32//s,s).mean((2,4))

def unit(a):
    a = a.reshape(len(a),-1)
    return a/np.sqrt(1e-8+(a*a).sum(1,keepdims=True))

def describe(a,extra):
    b,gx,gy = bits(a)
    v = []
    for j in range(8):
        t = np.pi*j/4
        d = np.cos(t)*gx+np.sin(t)*gy
        # A narrow signed-edge orientation channel.
        e = np.maximum(0,d-.51*np.sqrt(gx*gx+gy*gy))
        v.append(np.sqrt(pooled(blur(blur(e.astype(np.float32))),4)))
    ori = np.stack(v,1)
    img = np.sqrt(pooled(b,2))
    return np.concatenate((unit(ori)*.80,unit(img)*.60),axis=1).astype(np.float32)

def counters(a):
    # Where is white paper inaccessible from the frame? Repeat at several
    # ink levels so a gap just one grey pixel wide is not a hard decision.
    maps = []
    peak = a.max((1,2),keepdims=True)
    for lev in (.17,.40,.64):
        paper = a < lev*peak
        exterior = np.zeros_like(paper)
        exterior[:,0,:] = paper[:,0,:]
        exterior[:,-1,:] = paper[:,-1,:]
        exterior[:,:,0] = paper[:,:,0]
        exterior[:,:,-1] = paper[:,:,-1]
        for step in range(65):
            nxt = exterior.copy()
            nxt[:,1:] |= exterior[:,:-1]
            nxt[:,:-1] |= exterior[:,1:]
            nxt[:,:,1:] |= exterior[:,:,:-1]
            nxt[:,:,:-1] |= exterior[:,:,1:]
            nxt &= paper
            if np.array_equal(nxt,exterior):
                break
            exterior = nxt
        rows = (paper & ~exterior).sum(2).astype(np.float32)
        centers = np.arange(7,dtype=np.float32)*4+3.5
        win = np.maximum(0,1-np.abs(axis32[:,None]-centers[None,:])/5)
        areas = (rows[:,:,None]*win).sum(1)
        maps.append(areas/(6+areas))
    return (maps[0]+maps[1]+maps[2])/3

def localfields(a):
    b,gx,gy = bits(a)
    v = [np.maximum(gx,0),np.maximum(-gx,0),np.maximum(gy,0),np.maximum(-gy,0),b*.59]
    return np.stack([pooled(blur(ch),2) for ch in v],axis=-1)

def elastic(q,templates):
    # A correspondence lattice. Each patch may move a cell up/down and
    # left/right; a displacement has a cost, as does a discontinuity between
    # neighboring patches. The final distance uses the full pairwise energy.
    n,k = templates.shape[:2]
    temp = templates.reshape(-1,16,16,5).astype(np.float32)*(1/255)
    temp = np.pad(temp,((0,0),(1,1),(1,1),(0,0)))
    q = q[:,None,:,:, :]
    unary = []
    for dy in range(-1,2):
        for dx in range(-1,2):
            tv = temp[:,dy+1:dy+17,dx+1:dx+17].reshape(n,k,16,16,5)
            df = (tv-q).reshape(n,k,4,4,4,4,5)
            unary.append((df*df).sum((3,5,6))+.07*(dy*dy+dx*dx))
    cost = np.stack(unary,axis=-1)
    # Two alternating checkerboard relaxations, with starts given by the
    # independent-patch optima. Only non-neighbors move in each half-sweep.
    labels = np.argmin(cost,axis=-1)
    grid = np.arange(9)
    pot = .092*((grid[:,None]//3-grid//3)**2+(grid[:,None]%3-grid%3)**2)
    parity = (np.arange(4)[:,None]+np.arange(4))%2
    for phase in range(4):
        edges = pot[labels].astype(np.float32)
        u = cost.copy()
        u[:,:,1:] += edges[:,:,:-1]
        u[:,:,:-1] += edges[:,:,1:]
        u[:,:,:,1:] += edges[:,:,:,:-1]
        u[:,:,:,:-1] += edges[:,:,:,1:]
        labels = np.where(parity==phase%2,u.argmin(-1),labels)
    energy = np.take_along_axis(cost,labels[:,:,:,:,None],axis=-1).sum((2,3,4))
    energy += pot[labels[:,:,1:],labels[:,:,:-1]].sum((2,3))
    energy += pot[labels[:,:,:,1:],labels[:,:,:,:-1]].sum((2,3))
    norm = (q*q).sum((2,3,4))+1e-5
    return energy/norm

def predict(x):
    result = np.empty(len(x),np.int64)
    # Use fixed-size microkernels. Quantizing the unit query vector to a
    # power-of-two grid makes the coarse dot products exact in float32:
    # the sum of their nonnegative integer numerators is under 2**24.
    tile = 4 if len(x)<8 else 32
    for start in range(0,len(x),tile):
        n = min(tile,len(x)-start)
        inp = np.zeros((tile,28,28),np.float32)
        inp[:n] = x[start:start+n,0]
        img,extra = normalize(inp)
        f = np.rint(16384*describe(img,extra))
        scores = []
        shortlist = []
        cq = counters(img)
        qfields = localfields(img)
        for d in range(10):
            dot = (f @ features[d].T)*(1/(16384*240))
            dw = (extra[:,1:2]-aspects[d][None,:]) / (.32+.25*extra[:,1:2])
            ds = (extra[:,0:1]-slants[d][None,:]) / .65
            dist = 1-dot + .009*np.minimum(3,dw**2) + .006*np.minimum(3,ds**2)
            sel = np.argpartition(dist,64,axis=1)[:,:64]
            first = np.take_along_axis(dist,sel,axis=1)
            gap = cq[:,None,:]-countermaps[d,sel]
            first += .010*(gap*gap).sum(2)
            idx = np.argsort(first,axis=1)[:,:12]
            e = np.take_along_axis(first,idx,axis=1)
            scores.append(e)
            shortlist.append(np.take_along_axis(sel,idx,axis=1))
        e = np.stack(scores,1)
        which = np.stack(shortlist,1)
        coarse = .70*e[:,:,0]+.30*e.mean(2)
        # Use the displacement lattice on the three plausible explanations.
        rivals = np.argsort(coarse,axis=1)[:,:3]
        qrow = np.arange(tile)[:,None]
        score = []
        for rj in range(3):
            ds = rivals[:,rj:rj+1]
            ix = which[qrow,ds][:,0,:]
            de = elastic(qfields,fieldmaps[ds,ix])
            combined = e[qrow,ds][:,0,:] + .22*de
            best = np.sort(combined,axis=1)
            score.append(.74*best[:,0]+.26*best[:,:5].mean(1))
        labels = rivals[np.arange(tile),np.argmin(np.stack(score,1),axis=1)]
        result[start:start+n] = labels[:n]
    return result

features = [np.ascontiguousarray(a,dtype=np.float32) for a in CONSTANTS['directions']]
aspects = CONSTANTS['attributes'][:,:,1]
slants = CONSTANTS['attributes'][:,:,0]
countermaps = CONSTANTS['counters'].astype(np.float32)*(1/255)
fieldmaps = CONSTANTS['fields']
