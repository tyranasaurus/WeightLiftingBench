import numpy as np


def smooth(a):
    l = np.concatenate((a[:, :, :1], a[:, :, :-1]), axis=2)
    r = np.concatenate((a[:, :, 1:], a[:, :, -1:]), axis=2)
    b = (l + 2*a + r)*0.25
    t = np.concatenate((b[:, :1], b[:, :-1]), axis=1)
    d = np.concatenate((b[:, 1:], b[:, -1:]), axis=1)
    return (t + 2*b + d)*0.25


def descriptor(a, occupancy):
    a = a.astype(np.float32)
    b = 0.5*(a + smooth(a))
    dx = np.concatenate((b[:,:,1:],b[:,:,-1:]),axis=2) - np.concatenate((b[:,:,:1],b[:,:,:-1]),axis=2)
    dy = np.concatenate((b[:,1:],b[:,-1:]),axis=1) - np.concatenate((b[:,:1],b[:,:-1]),axis=1)
    mag = np.sqrt(dx*dx+dy*dy+1e-12)
    noise = np.quantile(mag.reshape(len(a),-1),0.25,axis=1).astype(np.float32)[:,None,None]
    magc = np.maximum(mag-0.75*noise,0)
    energy = np.sqrt(magc)*np.minimum(magc/(noise+0.012),1.0)
    # Soft, unsigned bins with an angular width of one bin. Pool locally;
    # truncate local amplitudes to limit the effect of cast shadows.
    angle = np.arctan2(dy,dx)*np.float32(5/np.pi)
    bins = []
    broad = smooth(smooth(smooth(magc)))
    strength = np.minimum(2.5,magc / (0.04 + noise + broad))
    for k in range(5):
        d = np.abs((angle-k+2.5)%5-2.5)
        field = np.maximum(1-d,0)*np.sqrt(energy*strength)
        bins.append(smooth(field))
    occupancy = smooth(occupancy.astype(np.float32))
    # Signed skin relief: compare a point with neighbouring skin, not with
    # the background. Cast shadows between fingers tend to be dark whatever
    # the polarity of the silhouette against the wall.
    acc=b*occupancy; weight=occupancy
    for step in range(7):
        acc=smooth(acc); weight=smooth(weight)
    tone=(b-acc/(weight+1e-6))*occupancy
    # Enclosed negative space. On each ray ask if skin is present before a
    # finger-length distance; a hole has skin in even its weakest direction.
    h,w=a.shape[1:]
    pad=np.pad(occupancy,((0,0),(8,8),(8,8)),mode='constant')
    rays=[]
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
        cover=np.zeros_like(occupancy)
        for r in (1,2,3,4,6,8):
            rr=int(0.72*r+0.5) if dy*dx else r
            oy=8+dy*rr; ox=8+dx*rr
            cover=np.maximum(cover,pad[:,oy:oy+h,ox:ox+w])
        rays.append(cover)
    enclosure=np.min(np.stack(rays),axis=0)
    hole=np.maximum(enclosure-occupancy,0)
    return np.concatenate((np.stack(bins,axis=3),occupancy[:,:,:,None],contourfield(occupancy),tone[:,:,:,None],hole[:,:,:,None]),axis=3)


def contourfield(m):
    h = smooth(m)
    dx = np.concatenate((h[:,:,1:],h[:,:,-1:]),axis=2) - np.concatenate((h[:,:,:1],h[:,:,:-1]),axis=2)
    dy = np.concatenate((h[:,1:],h[:,-1:]),axis=1) - np.concatenate((h[:,:1],h[:,:-1]),axis=1)
    return np.stack((np.maximum(dx,0),np.maximum(-dx,0),np.maximum(dy,0),np.maximum(-dy,0)),axis=3)


def cellbank(n,g):
    step = n/g
    c = (np.arange(g)+0.5)*step-0.5
    w = np.maximum(1-np.abs(np.arange(n)[None,:]-c[:,None])/step,0)
    return (w/w.sum(axis=1,keepdims=True)).astype(np.float32)


def pooldesc(field,g):
    n,h,w,c = field.shape
    bank = cellbank(h,g)
    temp = (bank @ field.reshape(n,h,w*c)).reshape(n,g,w,c).transpose(0,1,3,2)
    v = (temp @ bank.T).transpose(0,1,3,2)
    edge = v[:,:,:,:5]
    # Local HOG normalization with a floor relative to this image's energy.
    s = np.sum(edge*edge,axis=3,keepdims=True)
    floor = 0.50*s.mean(axis=(1,2),keepdims=True)+1e-8
    edge = edge/(s+floor)**0.33
    edge = edge.reshape(n,-1)
    edge = edge/np.sqrt(np.sum(edge*edge,axis=1,keepdims=True)+1e-9)
    occ = v[:,:,:,5].reshape(n,-1)
    occ = occ/np.sqrt(np.sum(occ*occ,axis=1,keepdims=True)+1e-9)
    contour=v[:,:,:,6:10].reshape(n,-1)
    contour=contour / np.sqrt(np.sum(contour*contour,axis=1,keepdims=True)+1e-9)
    # Concatenated groups with a fixed ratio of energy. Boundary normals
    # distinguish a gap between fingers from a peak of similar orientation.
    tone=v[:,:,:,10].reshape(n,-1)
    tone=tone / np.sqrt(np.sum(tone*tone,axis=1,keepdims=True)+.015*g*g*.015)
    hole=v[:,:,:,11].reshape(n,-1)
    # A little hole evidence at every one of many edge troughs is not as
    # compelling as a window with roughly its own inscribed disk.
    hole=np.maximum(hole-.08,0)
    hole=hole / np.sqrt(np.sum(hole*hole,axis=1,keepdims=True)+(.060*g)**2)
    return np.concatenate((edge*.72,occ*.45,contour*.43,tone*.235,hole*.25),axis=1)


def foreground(a):
    # A border pixel lying in a smooth patch may explain other pixels as
    # background. Extrapolate its intensity using its local first derivatives.
    # Several depths into the border accommodate frames around a photo.
    b=smooth(a)
    dx=(np.roll(b,-1,axis=2)-np.roll(b,1,axis=2))*0.5
    dy=(np.roll(b,-1,axis=1)-np.roll(b,1,axis=1))*0.5
    dx[:,:,[0,-1]]=0; dy[:,[0,-1],:]=0
    # Certificate of local planarity: approximation error in a neighbourhood.
    tex=smooth(np.abs(a-b)+1.5*(np.abs(dx-smooth(smooth(dx)))+np.abs(dy-smooth(smooth(dy)))))
    dx=smooth(dx); dy=smooth(dy)
    v,u=np.mgrid[:28,:28].astype(np.float32)
    best=np.full_like(a,10)
    noise=np.quantile(tex.reshape(len(a),-1),.25,axis=1).astype(np.float32)[:,None,None]
    for c in (1,4,23,26):
        for t in range(2,27,4):
            for y,z in ((c,t),(t,c)):
                du=u-z; dv=v-y
                dist=np.sqrt(du*du+dv*dv)
                err=np.abs(b-b[:,y:y+1,z:z+1]-np.clip(dx[:,y:y+1,z:z+1],-.018,.018)*du-np.clip(dy[:,y:y+1,z:z+1],-.018,.018)*dv)
                # Smooth-field extrapolation becomes less certain further from
                # the seed. Cost of explaining texture as a plane is local.
                cost=err/(.025+noise+.0025*dist)+tex[:,y:y+1,z:z+1]/(.012+1.8*noise)+.027*dist
                best=np.minimum(best,cost)
    residual=(best-.9)*.58+(tex-noise)/(.036+1.7*noise)*.51
    m=np.clip(residual,0,1)
    # Diffuse confidence within patches of similar intensity, not across
    # strong boundaries. Per-image, no sharing amongst the batch.
    ref=b
    wt=[]
    for axis,shift in ((1,-1),(1,1),(2,-1),(2,1)):
        d=np.abs(ref-np.roll(ref,shift,axis=axis))
        w=np.exp(-d/(.030+noise))
        if axis==1: w[:,0 if shift==1 else -1]=0
        else: w[:,:,0 if shift==1 else -1]=0
        wt.append(w)
    init=m.copy(); denom=1.38+sum(wt)
    for it in range(8):
        m=(m+.38*init+wt[0]*np.roll(m,-1,axis=1)+wt[1]*np.roll(m,1,axis=1)+wt[2]*np.roll(m,-1,axis=2)+wt[3]*np.roll(m,1,axis=2))/denom
    # A low-texture region can run from the background into a hole in an
    # O or F, but it cannot run through several strong digit boundaries.
    # Compute a soft minimax path barrier to the image border. Smooth skin
    # enclosed by the contour should not be explained by an unrelated seed.
    barrier=np.full_like(b,1)
    barrier[:,0]=0;barrier[:,-1]=0;barrier[:,:,0]=0;barrier[:,:,-1]=0
    difx=np.abs(b[:,:,1:]-b[:,:,:-1]);dify=np.abs(b[:,1:]-b[:,:-1])
    for it in range(20):
        left=np.maximum(barrier[:,:,:-1],difx); right=np.maximum(barrier[:,:,1:],difx)
        up=np.maximum(barrier[:,:-1],dify); down=np.maximum(barrier[:,1:],dify)
        barrier[:,:,1:]=np.minimum(barrier[:,:,1:],left);barrier[:,:,:-1]=np.minimum(barrier[:,:,:-1],right)
        barrier[:,1:]=np.minimum(barrier[:,1:],up);barrier[:,:-1]=np.minimum(barrier[:,:-1],down)
    seal=np.clip((barrier-.005-noise*.5)/(.055+noise*1.5),0,1)
    return m+(1-m)*seal*.6*np.minimum(1,best*.8)


def warps(a, geometries):
    n,h,w=a.shape
    gy,gx=np.mgrid[:h,:w].astype(np.float32)
    stack=[]
    for sx,sy,dx,dy in geometries:
        u=(gx-(w-1)/2)*sx+(w-1)/2+dx
        v=(gy-(h-1)/2)*sy+(h-1)/2+dy
        u=np.clip(u,0,w-1); v=np.clip(v,0,h-1)
        j=u.astype(np.int32); i=v.astype(np.int32)
        fu=(u-j).astype(np.float32); fv=(v-i).astype(np.float32)
        jj=np.minimum(j+1,w-1); ii=np.minimum(i+1,h-1)
        stack.append((1-fv)*((1-fu)*a[:,i,j]+fu*a[:,i,jj])+fv*((1-fu)*a[:,ii,j]+fu*a[:,ii,jj]))
    return np.stack(stack,axis=1).reshape(-1,h,w)


def verify(template, image, outside):
    # Recompute normalized inner products using each candidate's own spatial
    # support. Edges well outside its silhouette can be background objects.
    # A soft support is important: an absent finger still counts against a
    # match, even though the model does not exactly align every joint.
    cells=144
    p=template[:,:,:5*cells].reshape(24,-1,cells,5)
    q=image[:,:,:5*cells].reshape(24,-1,cells,5)
    pm=template[:,:,5*cells:6*cells]; qm=image[:,:,5*cells:6*cells]
    m=pm/(np.max(pm,axis=2,keepdims=True)+1e-8)
    w=.22+.78*np.minimum(1,m*2.5)
    def corr(p,q,weight):
        num=(weight*np.einsum('abch,abch->abc',p,q)).sum(axis=2)
        aa=(weight*np.einsum('abch,abch->abc',p,p)).sum(axis=2)
        bb=(weight*np.einsum('abch,abch->abc',q,q)).sum(axis=2)
        return num/np.sqrt(aa*bb+1e-10)
    texture=corr(p,q,w)
    pp=template[:,:,6*cells:10*cells].reshape(24,-1,cells,4); qq=image[:,:,6*cells:10*cells].reshape(24,-1,cells,4)
    w=.4+.6*np.minimum(1,m*4)
    boundary=corr(pp,qq,w)
    # Foreground agreement, with less penalty for an explained background
    # patch at the edge of the camera frame than one in the hand interior.
    pm=pm-(pm*outside).sum(axis=2,keepdims=True)/outside.sum()
    qm=qm-(qm*outside).sum(axis=2,keepdims=True)/outside.sum()
    region=(pm*qm*outside).sum(axis=2)/np.sqrt((pm*pm*outside).sum(axis=2)*(qm*qm*outside).sum(axis=2)+1e-10)
    shadow=(template[:,:,10*cells:11*cells]*image[:,:,10*cells:11*cells]).sum(axis=2)/(.235**2)
    ht=template[:,:,11*cells:];hq=image[:,:,11*cells:]
    vacancy=1-np.sum((ht-hq)**2,axis=2)/(.25**2)
    return .44*texture+.21*boundary+.20*region+.085*shadow+.065*vacancy


coarse = CONSTANTS['coarse'].astype(np.float32) / 16384
fine = CONSTANTS['fine']


def predict(x):
    n=len(x)
    pred=np.empty(n,np.int64)
    if n==0:
        return pred
    translations=[(1,1,0,0),(1,1,2.4,0),(1,1,-2.4,0),(1,1,0,2.2),(1,1,0,-2.2),
                  (0.85,0.88,-1.5,-1.5),(0.85,0.88,1.5,1.5),(0.85,0.88,-1.5,1.5),(0.85,0.88,1.5,-1.5),
                  (1.16,1.14,0,-1),(1.16,1.14,0,1.5)]
    g=len(translations)
    count=len(coarse)//24
    row,col=np.mgrid[:12,:12].astype(np.float32)
    outside=(.4+.6*np.exp(-((row-5.5)**2+(col-5.5)**2)/42)).reshape(1,1,-1)
    for start in range(0,n,8):
        b=x[start:start+8,0]
        nb=len(b)
        # Fixed matrix dimensions, irrespective of how the caller split the batch.
        if nb<8:
            b=np.concatenate((b,np.repeat(b[-1:],8-nb,axis=0)),axis=0)
        m=foreground(b)
        f=descriptor(warps(b,translations),warps(m,translations))
        small=pooldesc(f,6)
        large=pooldesc(f,12).reshape(8,g,-1)
        match=(small @ coarse.T).reshape(8,g,24,count)
        for i in range(nb):
            # Shortlist pose/registration pairs separately for each letter.
            # Use the fine descriptor for verification before pooling poses.
            s=match[i].transpose(1,0,2).reshape(24,g*count)
            choices=np.argpartition(s,-18,axis=1)[:,-18:]
            pose=choices%count+np.arange(24)[:,None]*count
            query=choices//count
            tf=fine[pose].astype(np.float32)/16384
            high=verify(tf,large[i,query],outside)
            low=np.take_along_axis(s,choices,axis=1)
            both=0.20*low+0.80*high
            best=np.sort(both,axis=1)[:,-3:]
            answer=(2*best[:,2]+best[:,1]+best[:,0])*0.25
            pred[start+i]=np.argmax(answer)
    return pred
