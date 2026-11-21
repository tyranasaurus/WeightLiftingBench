import numpy as np

axis = np.arange(28, dtype=np.float32)
rows = axis.reshape(1,28,1)
cols = axis.reshape(1,1,28)

def normalize(v):
    v = v.astype(np.float32)
    v = v / np.maximum(v.max((1,2),keepdims=True),.001)
    m = np.maximum(v.sum((1,2),keepdims=True),.001)
    cx = (cols*v).sum((1,2),keepdims=True)/m
    cy = (rows*v).sum((1,2),keepdims=True)/m
    ry = rows-cy
    rx = cols-cx
    lean = (v*rx*ry).sum((1,2),keepdims=True)/np.maximum(1,(v*ry**2).sum((1,2),keepdims=True))
    lean = np.clip(lean,-1.25,1.25)
    # Bounding rectangle after removing slant, excluding faint edge pixels.
    mark = v>.17
    xx = rx-lean*ry
    xl = np.min(np.where(mark,xx,40),axis=(1,2),keepdims=True)
    xr = np.max(np.where(mark,xx,-40),axis=(1,2),keepdims=True)
    yt = np.min(np.where(mark,ry,40),axis=(1,2),keepdims=True)
    yb = np.max(np.where(mark,ry,-40),axis=(1,2),keepdims=True)
    w = np.maximum(xr-xl+1,1)
    h = np.maximum(yb-yt+1,1)
    # Partial, not full, aspect normalization: a bare 1 stays narrow.
    scale = h/20*np.clip((w/h/.68)**.65,.61,1.57)
    sy = (rows-13.5)*h/20 + (yt+yb)*.42
    sx = (cols-13.5)*scale + (xl+xr)*.42 + lean*sy + cx
    sy = sy+cy
    ix = np.floor(sx).astype(np.int32)
    iy = np.floor(sy).astype(np.int32)
    fx = (sx-ix).astype(np.float32)
    fy = (sy-iy).astype(np.float32)
    b = np.arange(len(v))[:,None,None]
    l = np.clip(ix,0,27)
    r = np.clip(ix+1,0,27)
    a = np.clip(iy,0,27)
    d = np.clip(iy+1,0,27)
    n = ((1-fy)*((1-fx)*v[b,a,l]+fx*v[b,a,r])+fy*((1-fx)*v[b,d,l]+fx*v[b,d,r]))
    n *= (sx>-.5)&(sx<27.5)&(sy>-.5)&(sy<27.5)
    return n, w[:,0,0]/h[:,0,0]

def unit(v):
    return v/np.maximum(.0001,np.sqrt((v*v).sum(1,keepdims=True)))

def average(v, step):
    s = v.shape
    return v.reshape(s[0],s[1]//step,step,s[2]//step,step).mean((2,4))

def smooth(v):
    a = np.pad(v,((0,0),(1,1),(1,1)))
    h = (a[:,:,:-2] + 2*a[:,:,1:-1] + a[:,:,2:])*.25
    return (h[:,:-2] + 2*h[:,1:-1] + h[:,2:])*.25

def enclosed(n, level):
    # Fill from the outside with horizontal and vertical sweeps. Regions
    # that cannot see the exterior are enclosed counters (white spaces).
    wall = n>level
    ix = np.arange(28,dtype=np.int16).reshape(1,1,28)
    iy = ix.reshape(1,28,1)
    lw = np.maximum.accumulate(np.where(wall,ix,-2),axis=2)
    rw = np.minimum.accumulate(np.where(wall,ix,29)[:,:,::-1],axis=2)[:,:,::-1]
    tw = np.maximum.accumulate(np.where(wall,iy,-2),axis=1)
    bw = np.minimum.accumulate(np.where(wall,iy,29)[:,::-1,:],axis=1)[:,::-1,:]
    reach = (lw<0)|(rw>27)|(tw<0)|(bw>27)
    for turn in range(6):
        le = np.maximum.accumulate(np.where(reach,ix,-3),axis=2)
        ri = np.minimum.accumulate(np.where(reach,ix,30)[:,:,::-1],axis=2)[:,:,::-1]
        reach = (le>lw)|(ri<rw)
        up = np.maximum.accumulate(np.where(reach,iy,-3),axis=1)
        dn = np.minimum.accumulate(np.where(reach,iy,30)[:,::-1,:],axis=1)[:,::-1,:]
        reach = (up>tw)|(dn<bw)
    return (~reach)&(~wall)

def descriptors(ink):
    n, aspect = normalize(ink)
    a = smooth(n)
    b = smooth(smooth(a))
    mass = unit(average(b,2).reshape(len(n),196))
    # Signed edge directions: a dark counter has the opposite polarity
    # from a blob of ink. Soft bins avoid discontinuities on diagonals.
    gradx = np.roll(a,-1,2)-np.roll(a,1,2)
    grady = np.roll(a,-1,1)-np.roll(a,1,1)
    mag = np.sqrt(gradx**2+grady**2)
    dirs = []
    for i in range(8):
        angle = i*np.pi/4
        resp = np.maximum(0,gradx*np.float32(np.cos(angle))+grady*np.float32(np.sin(angle))-.55*mag)
        dirs.append(average(smooth(resp),4))
    hv = np.sqrt(np.stack(dirs,axis=-1)+.00002)
    hv = unit(hv.reshape(len(n),392))
    feature = np.concatenate((mass*np.float32(.62),hv*np.float32(.785)),axis=1)
    holes = (enclosed(n,.2).astype(np.float32)+enclosed(n,.48).astype(np.float32))*.5
    def region(mask):
        return (holes*mask).sum((1,2))
    top = region(rows<13.5)
    bot = region(rows>13.5)
    # Apertures have location as well as area. Saturate large areas: the
    # existence of a loop is better evidence than its precise area.
    topology = np.stack([1-np.exp(-top/13),1-np.exp(-bot/13),
                         1-np.exp(-(top+bot)/16),
                         1-np.exp(-region((rows>=11)&(rows<=16))/12)],axis=1)
    metadata = np.concatenate((np.log(np.maximum(aspect,.02))[:,None],topology),axis=1)
    return feature, metadata, n

# Kernels used for symmetric matching of local stroke directions. A line
# detector takes a second difference across the line, at two radii. Small
# direction and position errors are discounted rather than counted twice.
def fine_maps(v):
    a = smooth(v)
    halo = np.pad(a,((0,0),(4,4),(4,4)))
    def at(y,x):
        return halo[:,4+y:32+y,4+x:32+x]
    def diff(y,x):
        return a-(at(y,x)+at(-y,-x))*.5
    vertical = np.maximum(diff(0,2),diff(0,3)*.85)
    horiz = np.maximum(diff(2,0),diff(3,0)*.85)
    diag = np.maximum(diff(1,-1)*1.35,diff(2,-2))
    other = np.maximum(diff(1,1)*1.35,diff(2,2))
    oriented = [vertical,other,horiz,diag]
    responses = []
    fields = []
    for i in range(4):
        stroke = np.maximum(0,oriented[i]-.5*np.maximum(0,oriented[(i+2)%4]))
        responses.append(average(stroke,2))
    for i in range(4):
        p = np.maximum(responses[i],.75*np.maximum(responses[(i+1)%4],responses[(i+3)%4]))
        p = np.maximum(p,.46*responses[(i+2)%4])
        # A capped chamfer transform (quadratic penalty in the first cell,
        # linear tails). Distance to any good nearby line response.
        a = np.pad(p,((0,0),(3,3),(3,3)))
        field = p
        for y in range(-3,4):
            for x in range(-3,4):
                rr=x*x+y*y
                if rr>0 and rr<=10:
                    field=np.maximum(field,a[:,3+y:17+y,3+x:17+x]*np.float32(np.exp(-.24*rr)))
        fields.append(field)
    response = np.stack(responses,axis=1)
    field = np.stack(fields,axis=1)
    # Normalize each image by itself. Clipping above a typical strong
    # ridge keeps pen pressure from dominating coverage of the strokes.
    s = np.maximum(.015,response.max((1,2,3),keepdims=True)*.65)
    return np.minimum(response/s,1), np.minimum(field/s,1)

coarsebank = CONSTANTS['coarse'].astype(np.float32)/32767
meta = CONSTANTS['meta']
labels = CONSTANTS['labels']
count = len(labels)//10

def recognize(chunk):
    f, m, n = descriptors(chunk)
    # Fixed row count even for a singleton call: the matrix multiplication
    # uses the same kernel irrespective of surrounding images in the batch.
    q = np.zeros((32,f.shape[1]),np.float32)
    q[:len(f)] = f
    sims = (q @ coarsebank.T)[:len(f)]
    # Coarse hypothesis cost. Metadata is used gently: a break caused by
    # light pen pressure must not turn an eight irrevocably into a three.
    cost = 1-sims
    for i,weight in enumerate((.019,.011,.011,.014,.008)):
        delta = np.abs(m[:,i,None]-meta[None,:,i])
        cost += weight*np.minimum(delta,1.5)**1.45
    few = 8
    cc = cost.reshape(len(f),10,count)
    pick = np.argpartition(cc,few-1,axis=2)[:,:,:few] + np.arange(10)[None,:,None]*count
    testR,testF=fine_maps(n)
    testR=testR.reshape(len(n),784)
    testF=testF.reshape(len(n),784)
    result=np.zeros(len(f),dtype=np.int64)
    for j in range(len(f)):
        idx=pick[j].reshape(-1)
        r = CONSTANTS['ridge'][idx].astype(np.float32)*(1/255)
        g = CONSTANTS['near'][idx].astype(np.float32)*(1/255)
        # Distance from an input ridge to the template and vice versa;
        # subtract each image's own unmatched cost (a soft Dice distance).
        e = (r*(g-testF[j])+testR[j]*(testF[j]-g)).sum(1)
        d = e/np.maximum(1,r.sum(1)+testR[j].sum())
        total=(cost[j,idx]+.22*np.maximum(-.015,d)).reshape(10,few)
        ranked=np.sort(total,axis=1)
        # Similarity to several INDIVIDUAL independent renders, very
        # localized to the best explanation, not a synthesized prototype.
        value=.69*ranked[:,0]+.21*ranked[:,1]+.10*ranked[:,2]
        result[j]=np.argmin(value)
    return result

def predict(x):
    result=np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),32):
        result[start:start+32]=recognize(x[start:start+32,0])
    return result
