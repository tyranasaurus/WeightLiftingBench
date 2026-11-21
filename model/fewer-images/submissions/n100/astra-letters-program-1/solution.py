import numpy as np

def blur(a, n):
    for k in range(n):
        p = np.pad(a, ((0,0),(0,0),(1,1)), mode='edge')
        q = (p[:,:,:-2] + p[:,:,2:] + 2*a)*0.25
        p = np.pad(q, ((0,0),(1,1),(0,0)), mode='edge')
        a = (p[:,:-2] + p[:,2:] + 2*q)*0.25
    return a

def derivatives(a):
    p = np.pad(a, ((0,0),(1,1),(1,1)), mode='edge')
    return (p[:,1:-1,2:]-p[:,1:-1,:-2])*0.5, (p[:,2:,1:-1]-p[:,:-2,1:-1])*0.5

def sample(a, xs, ys):
    n,h,w = a.shape
    xs = np.clip(xs, 0, w-1.0001)
    ys = np.clip(ys, 0, h-1.0001)
    i = xs.astype(np.int32)
    j = ys.astype(np.int32)
    fx = xs-i
    fy = ys-j
    r = np.arange(n)[:,None,None]
    return ((1-fy)*((1-fx)*a[r,j,i]+fx*a[r,j,i+1]) + fy*((1-fx)*a[r,j+1,i]+fx*a[r,j+1,i+1])).astype(np.float32)

def normalize(v):
    return v / np.sqrt(0.000001+(v*v).sum(1,keepdims=True))

def featuremaps(gray, mask):
    # Undirected orientations for contours of either polarity, and signed
    # bandpass for dark creases between digits. Bound the response of strong
    # edges so clothing cannot dominate merely through higher contrast.
    ax,ay = derivatives(blur(gray,1))
    mag = np.sqrt(ax*ax+ay*ay+0.00000001)
    gain = 1 / (0.019 + 0.6*blur(mag,4) + 0.4*mag)
    mx,my = derivatives(blur(mask,1))
    mm = np.sqrt(mx*mx+my*my+0.00000001)
    channels = []
    for cs,sn in [(1,0),(0.70710678,0.70710678),(0,1),(-0.70710678,0.70710678)]:
        z = np.maximum(0,(ax*cs+ay*sn)**2/mag-0.28*mag)*gain
        edge = np.maximum(0,(mx*cs+my*sn)**2/mm-0.28*mm)
        channels.append(blur(z+0.9*edge,2))
    base = blur(gray,1)
    band = base-blur(base,10)
    adjust = np.maximum(0.022,blur(abs(band),5))
    band = np.clip(band/adjust,-2,2)
    return channels+[blur(mask,1),band]

def extract(maps,xx,yy):
    f = np.stack([sample(a,xx,yy) for a in maps],axis=3)
    return f

def describe(f):
    # A local descriptor for hand shape, independent of other windows. The
    # patch Hann-like weighting reduces contamination from the background.
    n,h,w,c=f.shape
    wx=1-.6*(2*(np.arange(w,dtype=np.float32)+.5)/w-1)**4
    wy=1-.6*(2*(np.arange(h,dtype=np.float32)+.5)/h-1)**4
    wt=wx[None,None,:,None]*wy[None,:,None,None]
    edge=f[:,:,:,:4]; edge=(edge-0.82*edge.mean((1,2),keepdims=True))*wt
    sh=f[:,:,:,4:5];sh=(sh-sh.mean((1,2),keepdims=True))*wt
    band=f[:,:,:,5:6]*wt
    return np.concatenate((normalize(edge.reshape(n,-1)),.76*normalize(sh.reshape(n,-1)),.55*normalize(band.reshape(n,-1))),1)

def observed(a):
    # Smooth background field inferred from the two upper corners, independently
    # per photograph. The cap prevents a dark object at one side dominating it.
    top = blur(a,2)
    l = np.median(top[:,:4,:4].reshape(len(a),-1),axis=1)
    r = np.median(top[:,:4,-4:].reshape(len(a),-1),axis=1)
    l,r = np.maximum(l,r-0.16),np.maximum(r,l-0.16)
    row = np.arange(28, dtype=np.float32)[None,:,None]/27
    col = np.arange(28, dtype=np.float32)[None,None,:]/27
    bg = l[:,None,None]*(1-col)+r[:,None,None]*col
    s = blur(a,1)
    # Estimate gentle vertical change separately on each side. Avoid carrying
    # a sleeve or a person at the boundary across the entire photograph.
    lside = np.median(s[:,:,:3],axis=2)[:,:,None]
    rside = np.median(s[:,:,-3:],axis=2)[:,:,None]
    drift = np.clip(lside-l[:,None,None],-.13,.13)*(1-col)+np.clip(rside-r[:,None,None],-.13,.13)*col
    residual = s-bg-drift
    gx,gy = derivatives(s)
    texture = blur(np.sqrt(gx*gx+gy*gy),3)
    # Photographic backdrops are smoother than the fingers. Use both signs
    # of contrast and also retain very dark regions next to a pale hand.
    dark = (-residual + .055*row - .035)/.115
    light = (residual - .055*row - .060)/.14
    rough = (texture-.022)/.066
    strength = np.clip(np.maximum(np.maximum(dark,light),rough),0,1)
    strength = blur(strength,2)
    # Close narrow dark/light bands inside the hand without closing large
    # gaps such as the space between two separated fingers.
    p = np.pad(strength,((0,0),(1,1),(1,1)),mode='edge')
    lifted = np.maximum.reduce([p[:,i:i+28,j:j+28] for i,j in [(0,1),(1,0),(1,1),(2,1),(1,2)]])
    p = np.pad(lifted,((0,0),(1,1),(1,1)),mode='edge')
    strength = np.minimum.reduce([p[:,i:i+28,j:j+28] for i,j in [(0,1),(1,0),(1,1),(2,1),(1,2)]])
    return strength

def localmatch(maps, windows, cells, model):
    # Only compare homologous patches within a single pose: left/right,
    # distal/proximal. Allow each patch one cell of relative movement.
    h = len(windows)
    pos = cells[None,None,None,:]
    xx = windows[:,:,0,None,None] + windows[:,:,2,None,None]*pos
    yy = windows[:,:,1,None,None] + windows[:,:,2,None,None]*pos.transpose(0,1,3,2)
    xx=xx+np.zeros_like(yy);yy=yy+np.zeros_like(xx)
    xx=xx.reshape(h,-1,len(cells));yy=yy.reshape(h,-1,len(cells))
    f=extract(maps,xx,yy).reshape(h,24,len(cells),len(cells),6)
    ps=[]
    # Use a grip patch and two patches spanning the upper digits. They
    # overlap slightly so a thin finger is not lost at a partition edge.
    for y,x in [(2,2),(2,8),(8,5)]:
        best=np.full((h,24),-4,np.float32)
        for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
            q=f[:,:,y+dy:y+dy+8,x+dx:x+dx+8,:].reshape(h*24,8,8,6)
            d=describe(q).reshape(h,24,-1)
            best=np.maximum(best,(d*model[:,:,len(ps)]).sum(2))
        ps.append(best)
    return (ps[0]+ps[1]+ps[2])/3

def predict(x):
    n = len(x)
    result = np.empty(n, dtype=np.int32)
    bank = CONSTANTS['poses']
    # Fixed block size, including the final partial block, ensures identical
    # floating point operations for any image order or call size.
    block = 64
    coord = (np.arange(14,dtype=np.float32)*2+0.5)-13.5
    gridx = coord[None,None,:]
    gridy = coord[None,:,None]
    for start in range(0,n,block):
        size = min(n-start,block)
        a = np.zeros((block,28,28),np.float32)
        a[:size] = x[start:start+size,0]
        m = observed(a)
        maps = featuremaps(a,m)
        score = np.full((block,24),-100,np.float32)
        windows = np.zeros((block,24,3),np.float32)
        index = np.zeros((block,24),np.int32)
        # Search modest position changes as well as a little extra zoom.
        for zoom in (1.0,0.83,1.17):
            for dx,dy in [(0,0),(-2,-1),(2,-1),(-1.5,1.5),(1.5,1.5)]:
                # Border replication would extend a tip or a wrist. This is
                # suitable for a wrist; the top and sides have little weight.
                xx = 13.5+gridx*zoom+dx
                yy = 13.5+gridy*zoom+dy
                feat = describe(extract(maps,xx,yy))
                sim = (feat @ bank.T).reshape(block,24,-1)
                best=sim.argmax(2)
                fit=np.take_along_axis(sim,best[:,:,None],axis=2)[:,:,0]
                better=fit>score
                windows[better]=np.array([13.5+dx,13.5+dy,zoom],np.float32)
                index=np.where(better,best,index)
                score=np.maximum(score,fit)
        ids=index+np.arange(24)[None,:]*(len(bank)//24)
        patch=localmatch(maps,windows,np.arange(18,dtype=np.float32)*2-17,CONSTANTS['parts'][ids])
        result[start:start+size] = (score+0.85*patch)[:size].argmax(1)
    return result
