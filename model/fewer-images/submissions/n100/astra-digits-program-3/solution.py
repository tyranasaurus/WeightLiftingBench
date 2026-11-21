import numpy as np

side = 32
grid = np.arange(side, dtype=np.float32) - (side - 1) / 2
sourcegrid = np.arange(28, dtype=np.float32)

def quantilepos(hist, q):
    cumul = np.cumsum(hist, axis=1)
    target = cumul[:, -1:] * q
    k = np.minimum((cumul < target).sum(1), hist.shape[1] - 1)
    rows = np.arange(len(hist))
    delta = (target[:, 0] - cumul[rows, k]) / np.maximum(hist[rows, k], .00001)
    return (k + .5 + delta).astype(np.float32)

def sample(im, y, x):
    n = len(im)
    iy = np.floor(y).astype(np.int32)
    ix = np.floor(x).astype(np.int32)
    dy = (y - iy).astype(np.float32)
    dx = (x - ix).astype(np.float32)
    rows = np.arange(n)[:, None, None]
    ya = np.clip(iy, 0, 27)
    yb = np.clip(iy + 1, 0, 27)
    xa = np.clip(ix, 0, 27)
    xb = np.clip(ix + 1, 0, 27)
    a = im[rows, ya, xa] * (1 - dx) + im[rows, ya, xb] * dx
    b = im[rows, yb, xa] * (1 - dx) + im[rows, yb, xb] * dx
    return ((1 - dy) * a + dy * b) * ((y > -.5) & (y < 27.5) & (x > -.5) & (x < 27.5))

def normalize(im):
    # Estimate slant from row moments, then use robust bounds in the deskewed frame.
    h = im.sum(2)
    w = im.sum(1)
    mass = np.maximum(h.sum(1), 1e-5)
    cy = (h * sourcegrid).sum(1) / mass
    cx = (w * sourcegrid).sum(1) / mass
    yy = sourcegrid - cy[:, None]
    xx = sourcegrid - cx[:, None]
    slope = ((im * xx[:, None, :]).sum(2) * yy).sum(1) / np.maximum((h * yy**2).sum(1), 1e-5)
    slope = np.clip(.96 * slope, -1.65, 1.65)
    # Project directly onto the de-slanted horizontal axis in a padded histogram.
    pos = sourcegrid[None, None, :] + 24 - slope[:, None, None] * yy[:, :, None]
    pos = np.clip(pos, 0, 78)
    k = np.floor(pos).astype(np.int32)
    f = (pos - k).astype(np.float32)
    offset = (np.arange(len(im), dtype=np.int32) * 80)[:, None, None]
    k += offset
    bins = np.bincount(k.ravel(), (im * (1 - f)).ravel(), minlength=len(im)*80)
    bins += np.bincount((k+1).ravel(), (im*f).ravel(), minlength=len(im)*80)
    proj = bins.reshape(-1, 80).astype(np.float32)
    x0 = quantilepos(proj, .007) - 24
    x1 = quantilepos(proj, .993) - 24
    y0 = quantilepos(h, .007)
    y1 = quantilepos(h, .993)
    ht = np.maximum(y1 - y0, 2)
    wd = np.maximum(x1 - x0, 1.5)
    aspect = wd / ht
    # Moderate width normalization: preserve narrow bars without penalizing wide handwriting.
    sx = (np.maximum(wd, ht*.34)/22) * np.clip(.74 / np.maximum(aspect,.34), .30, 3)**.18
    sy = ht / 25
    y = ((y0+y1)[:, None, None]*.5 + sy[:, None, None]*grid[None, :, None])
    x = (x0+x1)[:, None, None]*.5 + sx[:, None, None]*grid[None, None, :] + slope[:, None, None]*(y-cy[:, None, None])
    image = sample(im, y, x)
    return image / np.maximum(image.max((1,2), keepdims=True), .001), aspect

def blur(a):
    # Separable binomial filter, zero extension.
    b = .5*a
    b[:,1:] += .25*a[:,:-1]
    b[:,:-1] += .25*a[:,1:]
    c = .5*b
    c[:,:,1:] += .25*b[:,:,:-1]
    c[:,:,:-1] += .25*b[:,:,1:]
    return c

def coarse(a):
    return a.reshape(len(a),8,4,8,4).mean((2,4))

def cavity(im, threshold):
    # Fill the exterior by four-connected propagation. What is left is enclosed paper.
    paper = im < threshold
    reach = paper.copy()
    reach[:,1:-1,1:-1] = False
    for step in range(100):
        old = reach.copy()
        reach[:,1:] |= old[:,:-1]
        reach[:,:-1] |= old[:,1:]
        reach[:,:,1:] |= old[:,:,:-1]
        reach[:,:,:-1] |= old[:,:,1:]
        reach &= paper
        if np.array_equal(reach, old):
            break
    return (paper & ~reach).astype(np.float32)

# Two simultaneous-deletion passes, preserving connectivity and endpoints (Zhang-Suen).
# Compile both eight-neighbor truth tables directly from the thinning conditions.
pattern = np.arange(256)
bits = ((pattern[:,None] >> np.arange(8)) & 1).astype(bool)
transitions = ((~bits) & np.roll(bits,-1,axis=1)).sum(1)
eligible = (bits.sum(1)>=2) & (bits.sum(1)<=6) & (transitions==1)
north,east,south,west = bits[:,0],bits[:,2],bits[:,4],bits[:,6]
remove = np.stack((eligible & ~(north & east & south) & ~(east & south & west),
                   eligible & ~(north & east & west) & ~(north & south & west)))

def neighbors(s):
    p = np.pad(s.astype(np.uint8),((0,0),(1,1),(1,1)))
    return (p[:,:-2,1:-1] + 2*p[:,:-2,2:] + 4*p[:,1:-1,2:] + 8*p[:,2:,2:]
           + 16*p[:,2:,1:-1] + 32*p[:,2:,:-2] + 64*p[:,1:-1,:-2] + 128*p[:,:-2,:-2])

def strokes(a):
    s = a > .37
    for it in range(24):
        old = s.copy()
        for passnum in range(2):
            s &= ~remove[passnum,neighbors(s)]
        if np.array_equal(s,old):
            break
    hood = neighbors(s)
    ends = s & (bits.sum(1)[hood] == 1)
    joints = s & (transitions[hood] >= 3)
    # Distance coverage of the thin line and its salient points. Coarse coverage is
    # insensitive to moving a pen lift or the joining point by a few pixels.
    def spread(mask,steps):
        wave = mask.astype(np.float32)
        for i in range(steps):
            pre = wave.copy()
            wave[:,1:] = np.maximum(wave[:,1:],pre[:,:-1]-.92/steps)
            wave[:,:-1] = np.maximum(wave[:,:-1],pre[:,1:]-.92/steps)
            wave[:,:,1:] = np.maximum(wave[:,:,1:],pre[:,:,:-1]-.92/steps)
            wave[:,:,:-1] = np.maximum(wave[:,:,:-1],pre[:,:,1:]-.92/steps)
        return coarse(blur(wave))
    line = spread(s,4)
    line /= np.maximum(np.sqrt((line**2).sum((1,2),keepdims=True)),.001)
    end = spread(ends,7)
    joint = spread(joints,7)
    end /= np.maximum(np.sqrt((end**2).sum((1,2),keepdims=True)),1.7)
    joint /= np.maximum(np.sqrt((joint**2).sum((1,2),keepdims=True)),1.7)
    return np.stack((line,end*.4,joint*.4),axis=3)

def fields(im):
    a, aspect = normalize(im)
    smooth = blur(a)
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:,:,1:-1] = smooth[:,:,2:] - smooth[:,:,:-2]
    gy[:,1:-1] = smooth[:,2:] - smooth[:,:-2]
    angles = np.arange(8) * (np.pi/4)
    mag = np.sqrt(gx*gx+gy*gy+.000001)
    maps = []
    for angle in angles:
        # Soft positive projections raised to the third power give orientation selectivity.
        p = np.maximum(gx*np.float32(np.cos(angle))+gy*np.float32(np.sin(angle)), 0) / mag
        maps.append(coarse(blur(blur(p*p*p*mag))))
    maps = np.stack(maps, axis=3)
    # Per-image local contrast compression prevents a heavy pen from dominating the match.
    maps /= np.sqrt(.045 + maps.sum(axis=3,keepdims=True))
    maps /= np.maximum(np.sqrt((maps*maps).sum((1,2,3),keepdims=True)), .0001)
    ink = coarse(blur(smooth))
    ink /= np.maximum(np.sqrt((ink*ink).sum((1,2),keepdims=True)), .0001)
    shape = np.concatenate((maps*np.float32(np.sqrt(.79)),ink[:,:,:,None]*np.float32(np.sqrt(.21))),axis=3)
    holes = .65*cavity(a,.23) + .35*cavity(a,.53)
    holerows = holes.sum(2)
    bands = []
    for center in (6, 12, 19, 25):
        window = np.exp(-.5*((grid+15.5-center)/4.5)**2)
        bands.append(1-np.exp(-(holerows*window).sum(1)/20))
    geometry = np.stack([np.log(np.maximum(aspect,.05))]+bands,axis=1)
    return shape.astype(np.float32), geometry.astype(np.float32), strokes(smooth)

def packedfield(f):
    return np.rint(f*1200).clip(0,255).astype(np.uint8)

shapes = CONSTANTS['forms']
geometry = CONSTANTS['geom']
graphs = CONSTANTS['strokes']
# Compact first pass: signed-edge histograms on a 4x4 grid, plus 8x8 ink coverage.
def indexvec(f):
    pooled = f.reshape(len(f),4,2,4,2,9).sum((2,4))*.5
    v = np.concatenate((pooled.reshape(len(f),-1), f[:,:,:,8].reshape(len(f),-1)*.8),axis=1)
    return v / np.maximum(np.sqrt((v*v).sum(1,keepdims=True)),1e-7)

index = np.empty((208,len(shapes)),dtype=np.float32)
for start in range(0,len(shapes),512):
    group = shapes[start:start+512].astype(np.float32)/1200
    index[:,start:start+len(group)] = indexvec(group).T

# Fixed matrix sizes and internal padding also make the dot-product path independent of the caller's batch shape.
inbatch = 24
keep = 48
perclass = len(shapes)//10

def selectforms(f, g):
    sim = indexvec(f) @ index
    for j in range(5):
        d = np.abs(g[:,j:j+1]-geometry[None,:,j])
        if j == 0:
            sim -= .028*np.minimum(d,2)**1.5
        else:
            sim -= .021*d*d
    sim = sim.reshape(inbatch,10,perclass)
    return np.argpartition(sim,-keep,axis=2)[:,:,-keep:] + (np.arange(10)*perclass)[None,:,None]

def slide(a,dy,dx):
    b = np.zeros_like(a)
    ya,yb = max(0,-dy),min(8,8-dy)
    xa,xb = max(0,-dx),min(8,8-dx)
    b[:,ya:yb,xa:xb] = a[:,ya+dy:yb+dy,xa+dx:xb+dx]
    return b

def blocks(a):
    # Twenty-five patch sums, including small windows centered on the border.
    p = np.pad(a,((0,0),(1,1),(1,1)))
    b = p.reshape(len(p),5,2,5,2).sum((2,4))
    return b

def compare(a,g,graph,choices):
    cand = choices.ravel()
    b = shapes[cand].astype(np.float32)/1200
    # Half-cell motions of each local patch, with capped deformation freedom.
    # Penalize a mismatch twice, in both coordinate frames (forms and observation).
    ds = ((0,0),(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1))
    a = a[None,:,:,:]
    def sqerror(q,r):
        diff = q-r
        return (diff*diff).sum(axis=3)
    direct = sqerror(a,b)
    Best = blocks(direct)
    Rev = Best.copy()
    for dy,dx in ds[1:]:
        sa = slide(a,dy,dx)
        # At the halfway point between cells, compare a linearly blended map. In the
        # reverse frame use the same unblended cell comparisons and the exact variance identity.
        error = sqerror(sa,b)
        forward = .5*(direct+error) - .25*sqerror(sa,a)
        shifted = slide(error[:,:,:,None],-dy,-dx)[:,:,:,0]
        sb = slide(b,-dy,-dx)
        reverse = .5*(direct+shifted) - .25*sqerror(b,sb)
        # Where the shared comparison was shifted out, compare to the zero padding.
        if dy != 0:
            edge = 0 if dy == 1 else 7
            reverse[:,edge,:] += .5*(a[:,edge,:]**2).sum(2)
        if dx != 0:
            col = 0 if dx == 1 else 7
            correction = .5*(a[:,:,col]**2).sum(2)
            if dy != 0:
                correction[:,edge] = 0
            reverse[:,:,col] += correction
        penalty = .00032*(abs(dx)+abs(dy))
        Best = np.minimum(Best, blocks(forward)+penalty)
        Rev = np.minimum(Rev, blocks(reverse)+penalty)
    geo = np.abs(geometry[cand] - g)
    spatial = .43*(Best+Rev).sum((1,2))+.14*direct.sum((1,2))
    lineerror = np.square(graphs[cand].astype(np.float32)/750 - graph).sum((1,2,3))
    score = spatial + .14*lineerror + .043*np.minimum(geo[:,0], 2)**1.5 + .038*(geo[:,1:]**2).sum(1)
    # Use several individual matches, without estimating any parameters from the template bank.
    best = np.partition(score.reshape(10,keep),2,axis=1)[:,:3]
    return np.argmin(.78*best.min(1)+.22*best.mean(1))

def predict(x):
    out = np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),inbatch):
        part = x[start:start+inbatch,0]
        buf = np.zeros((inbatch,28,28),dtype=np.float32)
        buf[:len(part)] = part
        f,g,graph = fields(buf)
        graph = np.rint(graph*750).clip(0,255)/750
        f = packedfield(f).astype(np.float32)/1200
        choice = selectforms(f,g)
        for row in range(len(part)):
            out[start+row] = compare(f[row],g[row],graph[row],choice[row])
    return out
