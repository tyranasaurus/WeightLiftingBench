import numpy as np

side = 28
mid = 13.5
axis = np.arange(side, dtype=np.float32)
planeY, planeX = np.meshgrid(axis, axis, indexing="ij")


def convaxis(a, k, ax):
    r = len(k) // 2
    padding = [(0, 0)] * a.ndim
    padding[ax] = (r, r)
    b = np.pad(a, padding)
    v = np.zeros_like(a)
    sl = [slice(None)] * a.ndim
    for j in range(len(k)):
        sl[ax] = slice(j, j + a.shape[ax])
        v += k[j] * b[tuple(sl)]
    return v


def blur(a, s):
    k = np.arange(-int(np.ceil(2.8*s)), int(np.ceil(2.8*s))+1, dtype=np.float32)
    k = np.exp(-.5*(k/s)**2)
    k /= k.sum()
    return convaxis(convaxis(a, k, -1), k, -2)


def sample(a, y, x):
    # The extra zero border ensures affine sampling outside the frame reads black.
    y = np.clip(y+1, 0, side+.999)
    x = np.clip(x+1, 0, side+.999)
    iy = y.astype(np.int32)
    ix = x.astype(np.int32)
    fy = (y-iy).astype(np.float32)
    fx = (x-ix).astype(np.float32)
    b = np.pad(a, ((0,0),(1,1),(1,1)))
    row = np.arange(len(a))[:,None,None]
    return ((1-fy)*((1-fx)*b[row,iy,ix]+fx*b[row,iy,ix+1]) +
            fy*((1-fx)*b[row,iy+1,ix]+fx*b[row,iy+1,ix+1]))


def extent(a):
    # Relative grey-level contour, with interpolation between successive pixels.
    h = a.max(axis=1,keepdims=True)*.22
    c = a > h
    l = c.argmax(axis=1)
    r = a.shape[1]-1 - c[:,::-1].argmax(axis=1)
    row = np.arange(len(a))
    v = a[row,l]
    w = a[row,np.maximum(0,l-1)]
    left = l - np.clip((v-h[:,0])/np.maximum(v-w,1e-5),0,1)
    v = a[row,r]
    w = a[row,np.minimum(a.shape[1]-1,r+1)]
    right = r + np.clip((v-h[:,0])/np.maximum(v-w,1e-5),0,1)
    return left.astype(np.float32), right.astype(np.float32)


def canonical(a):
    # Remove predominant slant and part of the variation in height and width.
    mass = np.maximum(a.sum(axis=(1,2)),1e-6)
    mx = (a.sum(axis=1)*axis).sum(axis=1)/mass
    my = (a.sum(axis=2)*axis).sum(axis=1)/mass
    y = planeY - my[:,None,None]
    x = planeX - mx[:,None,None]
    shear = np.clip((a*x*y).sum(axis=(1,2))/np.maximum((a*y*y).sum(axis=(1,2)),1e-5),-1.3,1.3)
    shear *= .94
    shift = shear[:,None,None]*y
    flat = sample(a,planeY,planeX+shift)
    l,r = extent(flat.max(axis=1))
    t,b = extent(a.max(axis=2))
    height = np.maximum(2.5,b-t)
    width = np.maximum(1.5,r-l)
    aspect = width/height
    h = 22*np.minimum(1,1.36/np.maximum(aspect,.001))**.4
    w = 18*(np.maximum(aspect,.05)/.76)**.49
    w = np.clip(w,5.3,24.0)
    h = np.clip(h,14.0,22.0)
    cx = .76*(l+r)/2+.24*mx
    cy = .76*(t+b)/2+.24*my
    yy = (planeY-mid)*(height/h)[:,None,None]+cy[:,None,None]
    xx = (planeX-mid)*(width/w)[:,None,None]+cx[:,None,None]
    xx += shear[:,None,None]*(yy-my[:,None,None])
    img = sample(a,yy,xx)
    img /= np.maximum(img.max(axis=(1,2),keepdims=True),1e-5)
    # As with shear, ignore a modest change of aspect when comparing model to image.
    return img,np.stack([np.log(np.maximum(aspect,.035)),shear],axis=1)


def channels(a):
    # Signed edge directions. A short smoothing step reduces pixel noise but
    # keeps the two sides of a thin line separate.
    a = blur(a,.62)
    gx = convaxis(a,[-.5,0,.5],-1)
    gy = convaxis(a,[-.5,0,.5],-2)
    mag = np.sqrt(gx*gx+gy*gy)
    v = []
    for k in range(8):
        ang = k*(np.pi/4)
        p = gx*np.float32(np.cos(ang))+gy*np.float32(np.sin(ang))
        v.append(np.maximum(p-.68*mag,0)*3.125)
    v.append(a*.75)
    return np.stack(v,axis=1)


def spatial(a, step, n, sigma):
    centers = mid + (np.arange(n,dtype=np.float32)-(n-1)*.5)*step
    k = np.exp(-.5*((centers[:,None]-axis[None,:])/sigma)**2).astype(np.float32)
    k /= k.sum(axis=1,keepdims=True)
    return (k @ a) @ k.T


def unit(v, axes):
    return v/np.sqrt(np.maximum((v*v).sum(axis=axes,keepdims=True),1e-9))


def inside(mask):
    # Propagate exterior background from the four sides. Anything not reached
    # belongs to an enclosed counter (hole), with four-connected background.
    allowed = ~mask
    seen = np.zeros_like(mask)
    seen[:,0] = allowed[:,0]
    seen[:,-1] = allowed[:,-1]
    seen[:,:,0] = allowed[:,:,0]
    seen[:,:,-1] = allowed[:,:,-1]
    for j in range(2*side+4):
        nxt = seen.copy()
        nxt[:,1:] |= seen[:,:-1]
        nxt[:,:-1] |= seen[:,1:]
        nxt[:,:,1:] |= seen[:,:,:-1]
        nxt[:,:,:-1] |= seen[:,:,1:]
        nxt &= allowed
        if np.array_equal(nxt,seen):
            break
        seen = nxt
    return allowed & ~seen


def topology(a):
    # Three levels of persistent enclosure. A very small hole casts a weaker vote.
    scaled = blur(a,.4)
    scaled /= np.maximum(scaled.max(axis=(1,2),keepdims=True),1e-6)
    records = []
    for level in (.20,.44,.68):
        holes = inside(scaled>=level)
        rows = holes.sum(axis=2)
        r = np.arange(side)
        upper = np.minimum(1, np.maximum(0,(17-r)/6))
        lower = 1-upper
        center = np.minimum(1, np.maximum(0,(8-np.abs(r-mid))/4))
        areas = np.stack([rows@upper,rows@lower,rows@center,rows.sum(axis=1)],axis=1)
        presence = (1-np.exp(-areas/8))
        size = np.sqrt(areas/100)
        # Occupancy on a coarse grid encodes the location, not just the size.
        loc = np.sqrt(np.maximum(0,spatial(holes.astype(np.float32),5,5,2.5)))
        rec = np.concatenate([presence,size,loc.reshape(len(a),25)],axis=1)
        records.append(rec)
    return np.stack(records,axis=1).astype(np.float32)


def describe(a):
    a,geometry = canonical(a)
    v = channels(a)
    # Power compression and per-block normalisation stop one heavily inked
    # stroke from drowning out all the other strokes.
    coarse = []
    for step,n,sigma in ((3.5,7,1.85),(5.5,5,2.9)):
        grid = np.maximum(spatial(v,step,n,sigma),0)
        part = grid[:,:8]**.76
        energy = (part*part).sum(axis=1,keepdims=True)
        part /= (energy+.003)**.23
        part = unit(part.reshape(len(a),-1),(1,))
        ink = unit(grid[:,8].reshape(len(a),-1),(1,))
        if n==7:
            coarse.extend([part*.78,ink*.29])
        else:
            coarse.extend([part*.48,ink*.25])
    desc = unit(np.concatenate(coarse,axis=1),(1,))
    # SIFT-like patches centred on a 5x5 grid. Keep a one-step halo for
    # bounded local displacement in the elastic verification stage.
    grid = np.maximum(spatial(v,2,14,1.2),0)
    grid[:,:8] = grid[:,:8]**.8
    grid[:,8] = grid[:,8]**.85*.52
    pads = np.pad(grid,((0,0),(0,0),(2,2),(2,2)))
    coords = np.arange(3,12,2)
    patches = []
    for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
        feat = []
        for iy,ix in ((-1,-1),(-1,1),(1,-1),(1,1),(0,0)):
            y = coords+iy+dy+2
            x = coords+ix+dx+2
            feat.append(pads[:,:,y[:,None],x].transpose(0,2,3,1))
        feat = np.concatenate(feat,axis=-1)
        norm = np.sqrt((feat*feat).sum(axis=-1,keepdims=True))
        # Introduce an explicit blank component, so aligning ink to blank
        # costs something even when one of the patches has no gradient.
        blank = np.full_like(norm,.11)
        feat = unit(np.concatenate([feat,blank],axis=-1),(-1,))
        patches.append(feat.reshape(len(a),25,-1))
    return desc, np.stack(patches,axis=1), topology(a), geometry

coarseModel = CONSTANTS["global"].astype(np.float32)/255
coarseModel = unit(coarseModel,(1,))
patchModel = CONSTANTS["patch"]
holeModel = CONSTANTS["hole"]
geometryModel = CONSTANTS["geo"]
cuts = CONSTANTS["cuts"]
variantCost = CONSTANTS["cost"]


def batch(a):
    d,patch,hole,geo = describe(a)
    # Stage 1: compare whole shapes independently. Coarse screening is
    # separate for each class, so every class gets an elastic comparison.
    similarity = d @ coarseModel.T
    scores = np.zeros((len(a),10),dtype=np.float32)
    rows = np.arange(len(a))[:,None]
    disp = np.array([0,.020,.020,.020,.020,.032,.032,.032,.032],dtype=np.float32)
    for digit in range(10):
        start = int(cuts[digit])
        end = int(cuts[digit+1])
        gap = 1-similarity[:,start:end]
        delta = np.abs(geo[:,None,:]-geometryModel[None,start:end,:])
        geoCost = .017*np.minimum(delta[:,:,0],1.7)**1.4 + .010*np.minimum(delta[:,:,1],1.7)**1.4
        gap = gap + geoCost + variantCost[None,start:end]
        # Check topology on the closest whole shapes, using weak alignment
        # tolerance in grey level for variations in stroke pressure.
        take = np.argpartition(gap,39,axis=1)[:,:40]
        idx = start+take
        cost = gap[rows,take]
        ht = holeModel[idx]
        diff = np.abs(ht-hole[:,None,:,:])
        hd = (.047*diff[:,:,:,:4].mean(axis=(2,3)) + .031*diff[:,:,:,4:8].mean(axis=(2,3))
              +.082*diff[:,:,:,8:].mean(axis=(2,3)))
        cost += hd
        pick = np.argpartition(cost,7,axis=1)[:,:8]
        idx = idx[rows,pick]
        cost = cost[rows,pick]
        template = patchModel[idx].astype(np.float32)*(1/255)
        # Each model patch can move by one fine-grid cell; non-zero moves
        # have a small deformation cost. Only the input has to be shifted.
        best = np.full(template.shape[:3],2,dtype=np.float32)
        for j in range(9):
            ds = np.einsum("bkpv,bpv->bkp",template,patch[:,j],optimize=False)
            np.minimum(best,1-ds+disp[j],out=best)
        # Use both the mean patch error and the worst discrepancies. A
        # misplaced crossbar or an extra tail must matter to the decision.
        elastic = .64*best.mean(axis=2)+.36*np.sqrt((best*best).mean(axis=2))
        final = cost + 1.10*elastic
        scores[:,digit] = -final.min(axis=1)
    return scores.argmax(axis=1)


def predict(x):
    ans = np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),48):
        # Fixed internal batch size, including the final chunk, for identical
        # reduction kernels on singleton and full-batch calls.
        end = min(start+48,len(x))
        a = np.zeros((48,side,side),dtype=np.float32)
        a[:end-start] = x[start:end,0]
        ans[start:end] = batch(a)[:end-start]
    return ans
