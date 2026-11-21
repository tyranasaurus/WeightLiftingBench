import numpy as np

def surround(a):
    p = np.pad(a, ((0,0),(1,1),(1,1)), mode='edge')
    h = p[:,:,:-2] + 2*p[:,:,1:-1] + p[:,:,2:]
    return (h[:,:-2] + 2*h[:,1:-1] + h[:,2:]) * .0625

def evidence(im):
    # How deep is each pixel inside an island of fine-scale detail? A calm
    # background remains connected to the frame; even a smooth palm has an edge.
    sm = surround(im)
    dx = np.pad(sm, ((0,0),(0,0),(1,1)),mode='edge')
    dy = np.pad(sm, ((0,0),(1,1),(0,0)),mode='edge')
    dx = dx[:,:,2:] - dx[:,:,:-2]
    dy = dy[:,2:] - dy[:,:-2]
    mag = np.sqrt(dx*dx+dy*dy)
    strong = np.maximum(.014, mag.max((1,2),keepdims=True)*.36)
    c = .075*np.minimum(1.3,mag/strong)**1.5
    # Horizontal and vertical frame scans give an estimate of the outer envelope.
    a = np.cumsum(c,axis=2); b = np.cumsum(c[:,:,::-1],axis=2)[:,:,::-1]
    t = np.cumsum(c,axis=1)
    env = np.minimum(np.minimum(a/.105,b/.105),t/.095)
    # Also allow reaching the frame on a curved path (soft flood fill).
    cost = .002 + .65*(np.minimum(1.3,mag/strong)**1.8)
    cost[:,0,:] *= .5
    d = np.zeros_like(c)+2
    d[:,0] = cost[:,0]; d[:,:,0]=cost[:,:,0]; d[:,:,-1]=cost[:,:,-1]
    for j in range(18):
        d[:,1:] = np.minimum(d[:,1:], d[:,:-1]+cost[:,1:]); d[:,:-1] = np.minimum(d[:,:-1],d[:,1:]+cost[:,:-1]); d[:,:,1:] = np.minimum(d[:,:,1:],d[:,:,:-1]+cost[:,:,1:]); d[:,:,:-1] = np.minimum(d[:,:,:-1],d[:,:,1:]+cost[:,:,:-1])
    hull=np.minimum(1,env)*.26 + .74*np.minimum(1,d/.59)
    # A companion cue for holes and narrow finger gaps. A candidate stretch
    # of background should resemble a nearby parallel stretch reaching the
    # image border, in both brightness and smoothness. Work per scan line so
    # that a gradual vertical lighting gradient doesn't create a fake hand.
    left=np.median(sm[:,:,:3],axis=2,keepdims=True); right=np.median(sm[:,:,-3:],axis=2,keepdims=True)
    u=(np.arange(28,dtype=np.float32)/27)[None,None,:]
    bg=(1-u)*left+u*right
    detail=np.minimum(1,mag/(.5*strong+.001)); dev=np.abs(sm-bg)
    fl=1-np.exp(-2.7*(dev/(.042+.55*strong))**2-1.5*detail**2)
    agree=np.exp(-np.abs(left-right)/(.062+.32*strong))
    return hull*(1-.47*agree)+fl*.47*agree

def edgechan(im):
    sm = .45*im+.55*surround(im)
    xp=np.pad(sm,((0,0),(0,0),(1,1)),mode='edge'); yp=np.pad(sm,((0,0),(1,1),(0,0)),mode='edge')
    dx=xp[:,:,2:]-xp[:,:,:-2]; dy=yp[:,2:]-yp[:,:-2]
    norm=np.sqrt(dx*dx+dy*dy)+.0001
    amp=np.minimum(1.25,(norm/(.025+np.mean(norm,(1,2),keepdims=True)*1.4))**.75)
    out=[]
    for i in range(4):
        v=(dx*np.cos(i*np.pi/4)+dy*np.sin(i*np.pi/4))/norm
        out.append((v**4*amp).astype(np.float32))
    return out

def pooled(a,n):
    # Triangular interpolation between spatial histogram cells.
    f=np.maximum(0, 1-abs((np.arange(28,dtype=np.float32)[None,:]+.5)*n/28-(np.arange(n,dtype=np.float32)[:,None]+.5)))
    f=f/f.sum(1,keepdims=True)
    return (f @ a @ f.T).reshape(len(a),-1)

def unit(a,center):
    a = a-center*a.mean(1,keepdims=True)
    return a / np.sqrt(.00001+np.sum(a*a,1,keepdims=True))

def features(im,mask):
    n=len(im); seg=unit(pooled(mask,10),1)
    weight=np.array([.82,1,1,1,1,1,1,1,.93,.79,.64,.49],np.float32)[None,:,None]
    edges=[pooled(e,12).reshape(n,12,12)*weight for e in edgechan(im)]
    edges=np.stack(edges,3)
    # Equalize inter-finger valleys and the outline, which otherwise
    # dominates all inner edges. Include a separate, signed crease channel.
    edges = edges/(.11+np.sum(edges**2,axis=3,keepdims=True))**.29
    a=surround(im); b=surround(surround(surround(a))); crease=(a-b)*(.2+.8*mask)
    crease=np.tanh(crease/(.012+np.mean(np.abs(crease),(1,2),keepdims=True)))
    detail=pooled(crease,13)
    # An elongated narrow ridge has similar intensities along its length and
    # contrasts with one or both of its sides. Preserve its direction and
    # location even when a more prominent edge is a wall or a sleeve.
    ridges=[];pad=np.pad(a,((0,0),(3,3),(3,3)),mode='edge')
    for v,h in ((2,0),(2,2),(0,2),(2,-2)):
        plus=pad[:,3+v:31+v,3+h:31+h];minus=pad[:,3-v:31-v,3-h:31-h]
        nex=pad[:,3+h:31+h,3-v:31-v];prev=pad[:,3-h:31-h,3+v:31+v]
        thin=np.abs(2*a-nex-prev)-.55*np.abs(2*a-plus-minus)
        thin=np.clip(thin/(.045+np.mean(np.abs(thin),(1,2),keepdims=True)),0,2.5)
        ridges.append(pooled(thin,9))
    ridge=unit(np.stack(ridges,2).reshape(n,-1),.77)
    return np.concatenate([.61*seg,.76*unit(edges.reshape(n,-1),.68),.32*unit(detail,1),.355*ridge],1).astype(np.float32)

def cleanbox(mask):
    # Crop by robust one-sided extents of a connected hand, ignoring loose marks.
    n=len(mask)
    p=np.maximum(0,mask-.42)**1.25
    y,x=np.indices((28,28),dtype=np.float32)
    # Some photos have wallpaper or furniture touching the border. Reduce its
    # influence, but never assume the hand is restricted to a fixed window.
    central=.25+.75*np.exp(-((x-13.5)/11)**4)
    p *= central
    sm=surround(p)
    main=np.maximum(0,sm-.085)
    w=main.sum((1,2),keepdims=True)
    cx=(main*x).sum((1,2),keepdims=True)/(w+.0001)
    focus=np.exp(-.5*((x-cx)/8)**4)
    p=main*(.18+.82*focus)
    # Locate the top and the two sides above the end of the forearm.
    def bound(profile,q):
        s=profile.cumsum(1); return np.argmax(s>=s[:,-1:] * q,axis=1)
    v=p.sum(2); top=bound(v,.014)-1.
    last=bound(v,.995)+1.
    h=np.maximum(9,last-top)
    upper=(y[None,:,:]-top[:,None,None])/h[:,None,None]
    hp=(p*np.clip((1.06-upper)*3,.18,1)).sum(1)
    left=bound(hp,.018)-1.; right=bound(hp,.982)+1.
    mid=(left+right)*.5; width=right-left
    return mid.astype(np.float32),top.astype(np.float32),h.astype(np.float32),width.astype(np.float32)

def sample(a,x,y):
    x=np.clip(x,0,27); y=np.clip(y,0,27)
    ix=x.astype(int); iy=y.astype(int); ix1=np.minimum(27,ix+1); iy1=np.minimum(27,iy+1)
    b=np.arange(len(a))[:,None,None]; fx=x-ix; fy=y-iy
    return ((a[b,iy,ix]*(1-fx)+a[b,iy,ix1]*fx)*(1-fy)+(a[b,iy1,ix]*(1-fx)+a[b,iy1,ix1]*fx)*fy).astype(np.float32)

def contour(ext):
    # Fourier magnitudes of soft row extents are blind to vertical displacement.
    # Retain relative left/right phase via center and width as separate signals.
    n=len(ext); m=ext-ext.min((1,2),keepdims=True); m/=np.maximum(.0001,m.max((1,2),keepdims=True))
    pos=np.arange(m.shape[2],dtype=np.float32)[None,None,:]
    def bound(a,coord):
        p=np.maximum(0,a-.28)**1.1
        den=p.sum(2);ok=np.minimum(1,den/.82)
        c=p.cumsum(2); left=np.argmax(c>=den[:,:,None]*.058,axis=2)
        right=np.argmax(c>=den[:,:,None]*.942,axis=2)
        width=(right-left+.7)*ok;center=np.sum(p*coord,axis=2)/(den+.0001)
        mid=(center*ok).sum(1,keepdims=True)/(ok.sum(1,keepdims=True)+.001)
        return width,(center-mid)*ok
    w,b=bound(m,pos);h,a=bound(m.transpose(0,2,1),pos)
    def spectra(w):
        v=np.abs(np.fft.rfft(w,axis=1)[:,1:9]);return unit(np.sqrt(v+.001),0)
    return np.concatenate([spectra(w),spectra(b),spectra(h),spectra(a)],1).astype(np.float32)*.5

outlines=[contour(m[:,:100].reshape(len(m),10,10)) for m in CONSTANTS['models']]

def scores(im):
    seg=evidence(im); mid,top,h,width=cleanbox(seg)
    y,x=np.indices((28,28),dtype=np.float32)
    poses=[]; shapes=[]
    # Canonical square, parameterized primarily by the top of the hand and
    # the width of its finger span. Include a conservative, less cropped view.
    for tight,shift,zoom,above in ((1.,0.,1.,.045),(.0,0.,1.,0.),(.9,-.055,.85,.04),(.9,.055,.85,.04),(.75,0.,.77,.025),(.8,0.,1.13,.095)):
        side=np.maximum(np.maximum(h,width*1.08),15.)
        side=(side*tight+27*(1-tight))*zoom
        cx=mid*tight+13.5*(1-tight)+side*shift
        ty=top*tight-side*above
        xx=cx[:,None,None]+(x-13.5)*side[:,None,None]/27
        yy=ty[:,None,None]+y*side[:,None,None]/27
        vec=features(sample(im,xx,yy),sample(seg,xx,yy))
        poses.append(vec); shapes.append(contour(vec[:,:100].reshape(len(im),10,10)))
    inp=np.concatenate(poses); descriptors=np.concatenate(shapes)
    out=[]
    for i in range(24):
        m=CONSTANTS['models'][i]
        # Each comparison is a direct dot product with a mathematical pose,
        # never with a training photograph. Keep multiple supporting poses.
        sim=inp @ m.T + .21*(descriptors @ outlines[i].T)
        vals=np.partition(sim,sim.shape[1]-6,axis=1)[:,-6:]
        val=.6*vals.max(1)+.4*vals.mean(1)
        out.append(val.reshape(len(poses),len(im)).max(0))
    return np.stack(out,1)

def predict(x):
    ans=np.empty(len(x),np.int64)
    for i in range(0,len(x),32):
        block=x[i:i+32,0]
        # Fix the matrix multiplication dimensions even for the final block.
        a=np.zeros((32,28,28),np.float32); a[:len(block)]=block
        ans[i:i+32]=scores(a)[:len(block)].argmax(1)
    return ans
