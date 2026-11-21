import numpy as np

nside = 28
a = np.arange(nside, dtype=np.float32)
xx = a[None, :]
yy = a[:, None]

def edges(v):
    c = np.cumsum(v, axis=1)
    total = c[:, -1:]
    q = total * .008
    ans = []
    for target in (q, total-q):
        i = np.minimum(np.sum(c < target, axis=1), 27)
        n = np.arange(len(v))
        f = (c[n, i]-target[:,0]) / np.maximum(v[n,i], 1.e-7)
        ans.append(i+.5-np.clip(f, 0, 1))
    return ans

def sample(im, x, y):
    x, y = np.broadcast_arrays(x, y)
    valid = (x > -1) & (x < 28) & (y > -1) & (y < 28)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = (x-x0).astype(np.float32), (y-y0).astype(np.float32)
    n = np.arange(len(im))[:,None,None]
    l, r = np.clip(x0,0,27), np.clip(x0+1,0,27)
    t, b = np.clip(y0,0,27), np.clip(y0+1,0,27)
    return valid*((1-fy)*((1-fx)*im[n,t,l]+fx*im[n,t,r])+fy*((1-fx)*im[n,b,l]+fx*im[n,b,r]))

def normalize(im):
    mass = np.maximum(im.sum(axis=(1,2)), 1.e-5)
    row, col = im.sum(axis=2), im.sum(axis=1)
    mx, my = (col*a).sum(axis=1)/mass, (row*a).sum(axis=1)/mass
    dy = yy-my[:,None,None]
    cov = (im*dy*(xx-mx[:,None,None])).sum(axis=(1,2))
    v = (im*dy*dy).sum(axis=(1,2))
    s = (.93*np.clip(cov/np.maximum(v,1.e-5),-1.25,1.25))[:,None,None]
    upright = sample(im,xx+s*dy, np.broadcast_to(yy, im.shape))
    l,r = edges(upright.sum(axis=1))
    t,b = edges(row)
    w,h = np.maximum(1.,r-l),np.maximum(1.,b-t)
    rat = np.clip(w/h,.08,3.)
    # Partial rather than full aspect normalization: a bare '1' must stay narrow.
    sx = w/(17*np.clip((rat/.73)**.38,.44,1.28))
    sy = h/(21*np.clip((1.35/np.maximum(rat,1.35))**.45,.68,1.))
    y = ((t+b)/2)[:,None,None]+(yy-13.5)*sy[:,None,None]
    x = ((l+r)/2)[:,None,None]+(xx-13.5)*sx[:,None,None]+s*(y-my[:,None,None])
    canon = sample(im,x,y)
    rawl,rawr = edges(col)
    rawratio = np.clip((rawr-rawl)/h,.05,3.)
    return canon, np.log(rat).astype(np.float32), s[:,0,0], np.log(rawratio).astype(np.float32)

def spread(x, radius, sig):
    k = np.exp(-.5*(np.arange(-radius,radius+1)/sig)**2).astype(np.float32)
    k = k/k.sum()
    n,h,w = x.shape[:3]
    pad = np.zeros((n,h+2*radius,w)+x.shape[3:], np.float32)
    pad[:,radius:radius+h] = x
    v = sum(pad[:,i:i+h]*k[i] for i in range(len(k)))
    pad = np.zeros((n,h,w+2*radius)+x.shape[3:], np.float32)
    pad[:,:,radius:radius+w] = v
    return sum(pad[:,:,i:i+w]*k[i] for i in range(len(k)))

def directions(im):
    smooth = spread(im,1,.70)
    gy = (np.roll(smooth,-1,axis=1)-np.roll(smooth,1,axis=1))*.5
    gx = (np.roll(smooth,-1,axis=2)-np.roll(smooth,1,axis=2))*.5
    mag = np.sqrt(gx*gx+gy*gy)
    ang = np.arctan2(gy,gx).astype(np.float32)*np.float32(4/np.pi)
    # Four unsigned normal orientations, with linear angular voting.
    channels = []
    for j in range(4):
        d = np.abs((ang-j+2)%4-2)
        channels.append(mag*np.maximum(0,1-d))
    return np.stack(channels,axis=3)

def features(im):
    ink = im[:,:,:,None]
    maps = np.concatenate([ink, directions(im)],axis=3)
    # Grid centres at 1.5,5.5,...,25.5. A coarser description is used only
    # to look up templates on which to do the deformable fine comparison.
    pool = spread(maps,4,1.65)
    c = .25*(pool[:,1::4,1::4]+pool[:,2::4,1::4]+pool[:,1::4,2::4]+pool[:,2::4,2::4])
    vecs = []
    for k,weight in ((c[:,:,:,0:1],.49),(c[:,:,:,1:],.51)):
        f = np.sqrt(np.maximum(k.reshape(len(im),-1),0))
        vecs.append(f * (np.sqrt(weight)/np.maximum(np.sqrt((f*f).sum(axis=1,keepdims=True)),1.e-7)))
    coarse = np.concatenate(vecs,axis=1).astype(np.float32)
    g = spread(maps,2,1.0)
    g = g.reshape(len(im),14,2,14,2,5).mean(axis=(2,4))
    # Square-root densities suppress differences in pen darkness. Keeping the
    # two channel families separately normalized avoids a pen-width prior.
    parts = []
    for k,weight in ((g[:,:,:,0:1],.52),(g[:,:,:,1:],.48)):
        f = np.sqrt(np.maximum(k,0))
        norm = np.sqrt((f*f).sum(axis=(1,2,3),keepdims=True))
        parts.append(f*(np.sqrt(weight)/np.maximum(norm,1.e-7)))
    fine = np.concatenate(parts,axis=3).astype(np.float32)
    return coarse, fine

def enclosed(bg):
    # Flood from all four borders in parallel. Four-connected white regions
    # respect diagonal pen strokes. What remains is enclosed background.
    f = bg.copy()
    f[:,1:-1,1:-1] = False
    for step in range(64):
        p = f.copy()
        p[:,1:] |= f[:,:-1]
        p[:,:-1] |= f[:,1:]
        p[:,:,1:] |= f[:,:,:-1]
        p[:,:,:-1] |= f[:,:,1:]
        p &= bg
        if np.array_equal(p,f):
            break
        f = p
    return (bg & ~f).astype(np.float32)

def topology(im):
    ref = im.max(axis=(1,2),keepdims=True)
    hole = .5*(enclosed(im < .28*ref)+enclosed(im < .57*ref))
    # Soft vertical location of closed counters; a tiny pinhole provides
    # less evidence than an easily visible counter. A tall hole in a zero
    # can activate both halves; the image terms distinguish it from an eight.
    levels = [np.clip((16-yy)/5,0,1), np.clip((yy-11)/5,0,1), np.clip((yy-9)/3,0,1)*np.clip((18-yy)/3,0,1)]
    ans = []
    for w in levels:
        vol = (hole*w).sum(axis=(1,2))
        ans.append(1-np.exp(-vol/8))
    return np.stack(ans,axis=1)

ct = CONSTANTS['coarse'].astype(np.float32)
ft = CONSTANTS['fine']
rt = CONSTANTS['aspect']
st = CONSTANTS['slope']
rawt = CONSTANTS['rawratio']
ht = CONSTANTS['loops']
styles = CONSTANTS['styles'].astype(np.int32)
num = ct.shape[0]//10

# Little overlapping windows on a 7 by 7 displacement grid. The 14 by 14
# descriptor field has two nodes per grid cell in each direction. For each
# translation compute local squared feature distance, then pool and minimize.
def elastic(q,t):
    # q is one image, t is a modest shortlist of templates.
    pad = np.pad(q,((2,2),(2,2),(0,0)))
    small = np.full((len(t),7,7), 100.,np.float32)
    rigid = np.zeros(len(t),np.float32)
    for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1),(-2,0),(2,0),(0,-2),(0,2)):
        v = pad[2+dy:16+dy,2+dx:16+dx]
        dif = t-v
        d = np.einsum('nhwf,nhwf->nhw',dif,dif)
        cost = d.reshape(len(t),7,2,7,2).sum(axis=(2,4))
        # Neighbouring blocks vote on a shift, to discourage incompatible
        # motion of immediately adjacent strokes. Sum-preserving at borders.
        c = .5*cost
        c[:,1:] += .125*cost[:,:-1]
        c[:,:-1] += .125*cost[:,1:]
        c[:,:,1:] += .125*cost[:,:,:-1]
        c[:,:,:-1] += .125*cost[:,:,1:]
        c[:,0] += .125*cost[:,0]
        c[:,-1] += .125*cost[:,-1]
        c[:,:,0] += .125*cost[:,:,0]
        c[:,:,-1] += .125*cost[:,:,-1]
        if dx==0 and dy==0:
            rigid = c.sum(axis=(1,2))
        small = np.minimum(small,c+.00015*(dx*dx+dy*dy))
    return .83*small.sum(axis=(1,2))+.17*rigid

def block(x):
    im,r,s,raw = normalize(x[:,0].astype(np.float32))
    coarse,fine = features(im)
    loop = topology(im)
    # Exact integer inner products give the same first-stage decisions even
    # when the BLAS microkernel or the outer batch size changes. Feature
    # norms are below 2**24 after quantization, so every sum is exact.
    corr = np.rint(coarse*1100).astype(np.float32) @ ct.T
    cq = (np.rint(coarse*1100)**2).sum(axis=1)
    approx = (cq[:,None]+cn[None,:]-2*corr)/1210000
    dist = .017*np.minimum((r[:,None]-rt)**2,3.)
    dist += .032*(s[:,None]-st)**2+.013*np.minimum((raw[:,None]-rawt)**2,3.)
    dist += .015*((loop[:,0:1]-ht[:,0])**2+(loop[:,1:2]-ht[:,1])**2)+.008*(loop[:,2:3]-ht[:,2])**2
    approx = approx+dist
    res = np.empty(len(im),np.int32)
    for i in range(len(im)):
        allcost = approx[i].reshape(10,num)
        # Keep three distinct close variants and a fourth that may have a
        # different stroke construction from the closest variant.
        k = 14
        ix = np.argpartition(allcost,k,axis=1)[:,:k]
        vals = np.take_along_axis(allcost,ix,axis=1)
        ix = np.take_along_axis(ix,np.argsort(vals,axis=1,kind='stable'),axis=1)+np.arange(10)[:,None]*num
        first = ix[:,:3]
        diffstyle = styles[ix] != styles[ix[:,:1]]
        diffstyle[:,:3] = False
        loc = np.where(diffstyle.any(axis=1),diffstyle.argmax(axis=1),3)
        idx = np.concatenate([first,ix[np.arange(10),loc][:,None]],axis=1).ravel()
        d = elastic(fine[i],ft[idx].astype(np.float32)/1800.)
        d += .55*dist[i,idx]+.22*approx[i,idx]
        ordered = np.sort(d.reshape(10,4),axis=1)
        scores = .88*ordered[:,0]+.12*ordered[:,1]
        res[i] = np.argmin(scores)
    return res

cn = (ct*ct).sum(axis=1)

def predict(x):
    out = np.empty(len(x),np.int32)
    for start in range(0,len(x),48):
        out[start:start+48] = block(x[start:start+48])
    return out
