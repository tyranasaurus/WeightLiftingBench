import numpy as np

axis = np.arange(28, dtype=np.float32)
sy, sx = np.mgrid[:28, :28].astype(np.float32)
grid = np.arange(24, dtype=np.float32) - 11.5

def extent(a):
    s = np.cumsum(a, axis=1)
    total = s[:, -1:]
    left = np.sum(s < total * .009, axis=1)
    right = np.sum(s < total * .991, axis=1)
    return left.astype(np.float32), right.astype(np.float32)

def sample(a, y, x):
    x = np.clip(x, 0, 26.999)
    y = np.clip(y, 0, 26.999)
    ix = x.astype(np.int32)
    iy = y.astype(np.int32)
    dx = (x-ix).astype(np.float32)
    dy = (y-iy).astype(np.float32)
    n = np.arange(len(a))[:,None,None]
    return ((1-dy)*((1-dx)*a[n,iy,ix] + dx*a[n,iy,ix+1]) + dy*((1-dx)*a[n,iy+1,ix] + dx*a[n,iy+1,ix+1]))

def canonical(a):
    a = np.maximum(a - .025, 0)
    a = a / np.maximum(.01, a.max(axis=(1,2),keepdims=True))
    ry = a.sum(axis=2)
    mass = np.maximum(.001, ry.sum(axis=1))
    cy = (ry*axis).sum(axis=1)/mass
    cx = (a.sum(axis=1)*axis).sum(axis=1)/mass
    dys = sy-cy[:,None,None]
    sl = (a*dys*(sx-cx[:,None,None])).sum(axis=(1,2)) / np.maximum(1,(a*dys*dys).sum(axis=(1,2)))
    sl = .9*np.clip(sl,-1.6,1.6)
    # Expand the sheared canvas when estimating its width (no clipping).
    proj = np.maximum(0, a-.1)
    kx = np.floor(sx-sl[:,None,None]*dys+32.5).astype(np.int32).clip(0,95)
    offset = 96*np.arange(len(a))[:,None,None]
    hist = np.bincount((kx+offset).ravel(), weights=proj.ravel(), minlength=96*len(a)).reshape(-1,96)
    l,r = extent(hist)
    t,b = extent(proj.sum(axis=2))
    xc = .76*((l+r)*.5-32)+.24*cx
    yc = .76*((t+b)*.5)+.24*cy
    w = np.maximum(1,r-l+1)
    h = np.maximum(1,b-t+1)
    # Partial aspect normalization: do not make a bare 1 fill a square.
    yy = yc[:,None,None] + grid[None,:,None] * (h/19)[:,None,None]
    xs = np.maximum(.28*h, (.76*w+.24*h*.75))/18
    xx = xc[:,None,None] + grid[None,None,:]*xs[:,None,None] + (yy-cy[:,None,None])*sl[:,None,None]
    norm = sample(a, yy, xx)
    aspect = np.clip(np.log(w/h), -2.7, 1.2)
    return norm, aspect.astype(np.float32)

def smooth(a):
    b = np.pad(a, ((0,0),(0,0),(1,1)),mode='constant')
    b = .25*(b[:,:,:-2]+2*b[:,:,1:-1]+b[:,:,2:])
    b = np.pad(b, ((0,0),(1,1),(0,0)),mode='constant')
    return .25*(b[:,:-2]+2*b[:,1:-1]+b[:,2:])

def pooling(a,side):
    b = a.reshape(len(a),24//side,side,24//side,side)
    return b.mean(axis=(2,4)).reshape(len(a),-1)

def unit(a):
    return a / np.sqrt(np.maximum(.00001,(a*a).sum(axis=1,keepdims=True)))

def holes(a, cut):
    # Mark the outside by expanding through background. The area and vertical
    # position of unreached pockets distinguish bowls from open arches.
    bg = a < cut
    bg = np.pad(bg,((0,0),(1,1),(1,1)),constant_values=True)
    wet = np.zeros_like(bg)
    wet[:,0] = True
    wet[:,-1] = True
    wet[:,:,0] = True
    wet[:,:,-1] = True
    for step in range(56):
        old = wet.copy()
        wet[:,1:] |= old[:,:-1]
        wet[:,:-1] |= old[:,1:]
        wet[:,:,1:] |= old[:,:,:-1]
        wet[:,:,:-1] |= old[:,:,1:]
        wet &= bg
        if step%8 == 7 and np.array_equal(wet,old):
            break
    pocket = (bg & ~wet)[:,1:-1,1:-1].astype(np.float32)
    # Smoothed vertical histogram rather than exact component count/centroids.
    row = pocket.sum(axis=2)
    out = []
    for y in [4,8,12,16,20]:
        mask = np.maximum(0,1-abs(grid+11.5-y)/5)
        area = (row*mask).sum(axis=1)
        out.append(np.sqrt(area/(area+6)))
    return np.stack(out,axis=1)

def describe(a):
    a, asp = canonical(a)
    b = smooth(a)
    c = smooth(b)
    # Multiscale, sign-sensitive gradient distributions and soft occupancy.
    features = [unit(pooling(c,3))*.57]
    hog = []
    for level,block,weight in [(b,4,.65),(c,6,.50)]:
        pad = np.pad(level,((0,0),(1,1),(1,1)))
        gx = pad[:,1:-1,2:] - pad[:,1:-1,:-2]
        gy = pad[:,2:,1:-1] - pad[:,:-2,1:-1]
        mag = np.sqrt(gx*gx+gy*gy+.000001)
        chans = []
        for k in range(8):
            ang = k*np.pi/4
            response = np.maximum(0, gx*np.float32(np.cos(ang)) + gy*np.float32(np.sin(ang)) - .67*mag)
            chans.append(pooling(smooth(response),block))
        vectors = np.stack(chans,axis=2)
        local = np.sqrt((vectors*vectors).sum(axis=2,keepdims=True))
        vectors = vectors / np.sqrt(local+.026)
        features.append(weight*unit(vectors.reshape(len(a),-1)))
    f = unit(np.concatenate(features,axis=1))
    hole = .42*holes(a,.31) + .38*holes(a,.50) + .20*holes(a,.68)
    # A lower contrast medial map: fill small pen-width differences without
    # losing the sharp silhouette needed by the deformable local matcher.
    ink = np.minimum(1, .65*a+.55*b)
    aux = np.concatenate([asp[:,None],hole],axis=1)
    return f, ink, aux

bank = CONSTANTS['f'].astype(np.float32)*(1/127)
images = CONSTANTS['ink']
auxiliary = CONSTANTS['aux']
count = bank.shape[0]//10

def localdist(q, t):
    # Compare 4x4 blocks, allowing neighboring blocks to move by up to two
    # pixels. Compute the symmetric distance; a short displacement has a cost.
    pad = np.pad(q, ((2,2),(2,2)))
    tp = np.pad(t, ((0,0),(2,2),(2,2)))
    norm = np.maximum(1, (q*q).sum()+(t*t).sum(axis=(1,2)))
    best1 = np.zeros((len(t),6,6),dtype=np.float32)+100
    best2 = best1.copy()
    for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1),(-2,0),(2,0),(0,-2),(0,2)]:
        z = t-pad[2+dy:26+dy,2+dx:26+dx]
        z *= z
        cost = z.reshape(-1,6,4,6,4).sum(axis=(2,4)) + .060*(dx*dx+dy*dy)
        best1 = np.minimum(cost,best1)
        z = q-tp[:,2+dy:26+dy,2+dx:26+dx]
        z *= z
        cost = z.reshape(-1,6,4,6,4).sum(axis=(2,4)) + .060*(dx*dx+dy*dy)
        best2 = np.minimum(cost,best2)
    return .5*(best1.sum(axis=(1,2))+best2.sum(axis=(1,2)))/norm

def classify(a):
    n = len(a)
    f, ink, aux = describe(a)
    # Fixed dimensions for the candidate search, including for singleton calls.
    fq = np.zeros((48,bank.shape[1]),dtype=np.float32)
    fq[:n] = np.rint(f*127)
    sim = (fq @ bank.T)[:n]/127
    # Quantization served only to accelerate search; correct the template norms.
    sim *= correction
    sim /= np.maximum(.01,np.sqrt((fq[:n]*fq[:n]).sum(axis=1,keepdims=True))/127)
    diff = np.minimum(1.4,np.abs(aux[:,:1]-auxiliary[None,:,0]))
    cost = 1-sim + .039*diff*diff
    for j in range(1,6):
        delta = aux[:,j:j+1]-auxiliary[None,:,j]
        cost += .014*delta*delta
    cost = cost.reshape(n,10,count)
    res = np.zeros(n,dtype=np.int32)
    for i in range(n):
        # Shortlist independently for each class, then rerank the strongest
        # hypotheses. Each cost remains a distance to one particular drawing.
        v = cost[i]
        ids = np.argpartition(v,7,axis=1)[:,:8]
        fixed = np.take_along_axis(v,ids,axis=1)
        cls = np.argsort(fixed.min(axis=1),kind='stable')[:4]
        idx = (ids[cls]+cls[:,None]*count).ravel()
        flexible = localdist(ink[i], images[idx].astype(np.float32)*(1/255)).reshape(len(cls),8)
        total = np.sort(.66*fixed[cls] + .53*flexible,axis=1)
        res[i] = cls[np.argmin(.66*total[:,0]+.23*total[:,1]+.11*total[:,2])]
    return res

correction = 1/np.sqrt(np.maximum(.0001,(bank*bank).sum(axis=1)))

def predict(x):
    result = np.zeros(len(x),dtype=np.int32)
    for start in range(0,len(x),48):
        result[start:start+48] = classify(x[start:start+48,0])
    return result
