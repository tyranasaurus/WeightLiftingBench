import numpy as np

axis28 = np.arange(28, dtype=np.float32)
axis24 = np.arange(24, dtype=np.float32) - 11.5

def ink_limits(proj):
    cs = proj.cumsum(1)
    total = cs[:, -1:]
    bounds = []
    for frac in (.012, .988):
        q = frac * total
        i = np.minimum((cs < q).sum(1), proj.shape[1]-1)
        n = np.arange(len(i))
        bounds.append(i.astype(np.float32) + .5 - (cs[n,i] - q[:,0]) / np.maximum(proj[n,i], 1.e-7))
    return bounds

def bilinear(im, y, x):
    y = np.clip(y, 0, im.shape[1]-1.0001)
    x = np.clip(x, 0, im.shape[2]-1.0001)
    j = y.astype(np.int32)
    i = x.astype(np.int32)
    b = y-j.astype(np.float32)
    a = x-i.astype(np.float32)
    ind = np.arange(len(im))[:,None,None]
    return (1-b)*((1-a)*im[ind,j,i]+a*im[ind,j,i+1]) + b*((1-a)*im[ind,j+1,i]+a*im[ind,j+1,i+1])

def canonical(im):
    # Undo horizontal shear using ink moments, without rotating the baseline.
    yproj = im.sum(2)
    mass = np.maximum(yproj.sum(1), 1.e-6)
    cy = (yproj*axis28).sum(1) / mass
    dy = axis28[None,:] - cy[:,None]
    slope = ((im*axis28).sum(2)*dy).sum(1) / np.maximum((yproj*dy*dy).sum(1),1.e-5)
    slope = np.clip(slope,-1.2,1.2)
    offsets = np.arange(44, dtype=np.float32) - 8
    xx = offsets[None,None,:] + dy[:,:,None]*slope[:,None,None]
    sh = bilinear(im, axis28[None,:,None],xx)
    left,right = ink_limits(sh.sum(1))
    top,bottom = ink_limits(yproj)
    w = np.maximum(right-left,1.)
    h = np.maximum(bottom-top,1.)
    # Partial rather than full width normalization keeps a 1 narrow.
    targetw = np.clip(16.*(w/h/.75)**.45,5.,21.)
    xs = (left+right)[:,None,None]*.5 - 8. + axis24[None,None,:]*(w/targetw)[:,None,None]
    ys = (top+bottom)[:,None,None]*.5 + axis24[None,:,None]*(h/19.)[:,None,None]
    xs = xs + (ys-cy[:,None,None])*slope[:,None,None]
    norm = bilinear(im,ys,xs)
    return norm, w/h

def center_features(im):
    p = np.pad(im,((0,0),(1,1),(1,1)))
    smooth = (p[:,:-2,1:-1]+p[:,2:,1:-1]+p[:,1:-1,:-2]+p[:,1:-1,2:]+4*im)*.125
    p = np.pad(smooth,((0,0),(1,1),(1,1)))
    gx = (p[:,1:-1,2:]-p[:,1:-1,:-2])*.5
    gy = (p[:,2:,1:-1]-p[:,:-2,1:-1])*.5
    mag = np.sqrt(gx*gx + gy*gy)
    # A modest compression of contrast lets weak pen marks contribute too.
    mag = np.power(mag, .8)
    angle = np.arctan2(gy,gx) * (4./np.pi)
    cells = []
    for i in range(8):
        a = mag*np.maximum(0., 1.-np.abs((angle-i+4.)%8.-4.))
        cells.append(a.reshape(len(im),12,2,12,2).sum((2,4)))
    return np.stack(cells,axis=1), smooth

def unit(v):
    return v / np.sqrt(np.maximum((v*v).sum(1,keepdims=True),1.e-10))

def windowmat(n, count, sigma):
    centers = (np.arange(count)+.5)*n/count - .5
    a = np.exp(-.5*((np.arange(n)[None,:]-centers[:,None])/sigma)**2)
    return a.astype(np.float32)

poolgrad = windowmat(12,6,1.)
poolink = windowmat(24,8,1.35)
poolspace = windowmat(24,5,1.8)

# The shadow on each side of a stroke encodes concavities that a local
# gradient cannot see (e.g. the open space between the stems of a 4).
def negative_space(smooth):
    left = np.maximum.accumulate(smooth,axis=2)
    right = np.maximum.accumulate(smooth[:,:,::-1],axis=2)[:,:,::-1]
    above = np.maximum.accumulate(smooth,axis=1)
    below = np.maximum.accumulate(smooth[:,::-1],axis=1)[:,::-1]
    horizontal = np.minimum(left,right)-smooth
    vertical = np.minimum(above,below)-smooth
    concavities = np.stack((horizontal,vertical,np.minimum(horizontal,vertical)),axis=1)
    pooled = pool(concavities,poolspace)/16.
    v = (np.sqrt(pooled+.05)-np.float32(np.sqrt(.05))).reshape(len(smooth),-1)
    return unit(np.concatenate((v,np.ones((len(v),1),dtype=np.float32)),axis=1))

def localmap(smooth):
    ink = smooth.reshape(len(smooth),12,2,12,2).mean((2,4))
    p = np.pad(ink,((0,0),(1,1),(1,1)))
    dx = p[:,1:-1,2:]-p[:,1:-1,:-2]
    dy = p[:,2:,1:-1]-p[:,:-2,1:-1]
    grid = np.stack((.85*ink,dx,dy),axis=1)
    grid /= np.sqrt(np.maximum((grid*grid).sum((1,2,3),keepdims=True),1.e-8))
    return np.clip(np.rint(384*grid),-127,127).astype(np.int8)

def pool(grid, win):
    # Explicit order of the short contractions, independent of batch layout.
    tmp = np.einsum('py,ncyx->ncpx',win,grid,optimize=False)
    return np.einsum('qx,ncpx->ncpq',win,tmp,optimize=False)

def descriptors(im):
    canon,ratio = canonical(im)
    cells, smooth = center_features(canon)
    hist = pool(cells,poolgrad)
    hist = np.sqrt(hist+.012) - np.float32(np.sqrt(.012))
    # Normalize both over the local patch and over the glyph.
    hist /= (.45+(hist*hist).sum(1,keepdims=True))**.19
    occ = pool(smooth[:,None],poolink)[:,0]
    occ = np.sqrt(occ+.10)-np.float32(np.sqrt(.10))
    hg = unit(hist.reshape(len(im),-1))
    hk = unit(occ.reshape(len(im),-1))
    space = negative_space(smooth)
    full = np.concatenate((np.sqrt(np.float32(.67))*hg,np.sqrt(np.float32(.21))*hk,np.sqrt(np.float32(.12))*space),axis=1)
    return full, canon, ratio, localmap(smooth)

def components(im, threshold):
    # Flood exterior background, then measure the largest few enclosed regions.
    v = im < threshold
    v[0] = True
    v[-1] = True
    v[:,0] = True
    v[:,-1] = True
    def grow(seed):
        old = -1
        for k in range(96):
            a = seed.copy()
            a[1:] |= seed[:-1]
            a[:-1] |= seed[1:]
            a[:,1:] |= seed[:,:-1]
            a[:,:-1] |= seed[:,1:]
            seed = a & v
            now = seed.sum()
            if old == now:
                break
            old = now
        return seed
    outside = np.zeros_like(v)
    outside[0] = True
    outside[-1] = True
    outside[:,0] = True
    outside[:,-1] = True
    v &= ~grow(outside)
    holes = []
    for k in range(8):
        if not v.any():
            break
        seed = np.zeros_like(v)
        seed.flat[np.argmax(v)] = True
        part = grow(seed)
        v &= ~part
        ys,xs = np.nonzero(part)
        if len(ys)>=2:
            holes.append((len(ys),float(ys.mean())))
    return holes

def topology(im):
    # Soft evidence: no missing-hole penalty (loops can be open or filled in).
    evidence = np.zeros(10,dtype=np.float32)
    for t in (.25,.54):
        holes = components(im,t)
        if not holes:
            continue
        holes = sorted(holes,reverse=True)
        area,yy = holes[0]
        power = np.clip((area-2.)/13.,0,1)
        # A large central counter is typical of 0; counters on 6 and 9
        # occupy the lower and upper zones respectively.
        d = (yy-11.5)/8.
        small = np.clip(1.-(area-6.)/28.,0,1)
        e = np.array([.012-.018*abs(d),-.052,-.030,-.031,-.012,
                      -.027,.010+.025*d,-.044,.002,.010-.025*d], dtype=np.float32)
        e[0] += .016*(1.-small)-.014*small
        e[4] += .015*np.clip(-d,0,1)
        e[2] += .014*np.clip(d,0,1)
        e[8] += .008*small
        if len(holes)>1:
            a2,y2 = holes[1]
            two = np.clip((a2-1.)/8.,0,1)*np.clip((abs(yy-y2)-2.)/4.,0,1)
            e -= .015*two
            e[8] += .085*two
        evidence += .5*power*e
    return evidence

# Integer-valued descriptors and dots make shortlist selection independent of
# floating point BLAS summation order. The exact scores fit inside 24 bits.
bank = CONSTANTS['glyph'].astype(np.float32)
logaspect = CONSTANTS['aspect'].astype(np.float32)
count = len(bank)//10

def match_block(im):
    # Keep even the sizes of elementwise math operations in preprocessing
    # independent of the number and positions of neighboring query images.
    rows = [descriptors(np.ascontiguousarray(im[j:j+1])) for j in range(len(im))]
    desc,canon,ratios,local = [np.concatenate(a,axis=0) for a in zip(*rows)]
    q = np.rint(desc*4096.).astype(np.float32)
    # Templates quantized separately with scale 1024; all dots are positive.
    scores = q @ bank.T
    scores *= (1./4194304.)
    scores -= .018*np.minimum(1.6,np.abs(np.log(ratios)[:,None]-logaspect[None,:]))**1.4
    scores = scores.reshape(len(im),10,count)
    shortlist = np.argpartition(scores,count-6,axis=2)[:,:,-6:]
    s = np.take_along_axis(scores,shortlist,axis=2)
    picks = shortlist + np.arange(10)[None,:,None]*count
    best = np.empty((len(im),10),dtype=np.float32)
    for n in range(len(im)):
        templates = CONSTANTS['local'][picks[n].ravel()].astype(np.float32)
        loss = elastic(local[n].astype(np.float32),templates).reshape(10,6)
        z = np.sort(s[n]-.19*loss,axis=1)
        best[n] = z[:,-1]*.68 + z[:,-3:].sum(1)*(.32/3.) + topology(canon[n])
    return best.argmax(1)

def elastic(q,t):
    # Squared edge/ink mismatch after moving each 4x4 patch at most one
    # sample. Compare both directions so a template cannot hide extra ink.
    def patchsum(a):
        return a.reshape(len(a),3,4,3,4).sum((2,4))
    qt = q[None,:,:,:]-t
    center = patchsum((qt*qt).sum(1))
    best1 = center.copy()
    best2 = center.copy()
    qp = np.pad(q,((0,0),(1,1),(1,1)))
    tp = np.pad(t,((0,0),(0,0),(1,1),(1,1)))
    for dy,dx in ((0,1),(0,-1),(-1,0),(1,0),(-1,-1),(1,1),(-1,1),(1,-1)):
        y,x = 1+dy,1+dx
        d1 = t-qp[None,:,y:y+12,x:x+12]
        d2 = q[None]-tp[:,:,y:y+12,x:x+12]
        cost = 360*(abs(dy)+abs(dx))
        best1 = np.minimum(best1,patchsum((d1*d1).sum(1))+cost)
        best2 = np.minimum(best2,patchsum((d2*d2).sum(1))+cost)
    return (.17*center.sum((1,2))+.415*(best1+best2).sum((1,2)))/(384.*384.)

def predict(x):
    out = np.empty(len(x),dtype=np.int64)
    for i in range(0,len(x),96):
        out[i:i+96] = match_block(x[i:i+96,0])
    return out
