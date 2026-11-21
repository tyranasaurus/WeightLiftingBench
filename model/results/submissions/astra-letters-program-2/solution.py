
import numpy as np

def blur(a):
    v=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    t=(v[:,:,:-2]+v[:,:,2:]+2*v[:,:,1:-1])*.25
    return (t[:,:-2]+t[:,2:]+2*t[:,1:-1])*.25

def deriv(a):
    v=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    return .5*(v[:,1:-1,2:]-v[:,1:-1,:-2]),.5*(v[:,2:,1:-1]-v[:,:-2,1:-1])

def background(x):
    # Suppress slowly varying illumination with a robust local estimate
    # from the two upper corners: prefer the less textured extrapolation.
    n=len(x);t=(np.arange(28,dtype=np.float32)-13.5)/28
    patches=[x[:,:6,:6],x[:,:6,-6:],x[:,6:12,:5],x[:,6:12,-5:]]
    v=np.stack([np.median(p.reshape(n,-1),axis=1) for p in patches],axis=1)
    mad=np.stack([np.mean(abs(p-v[:,i,None,None]),axis=(1,2)) for i,p in enumerate(patches)],1)
    edge=np.mean(mad,1); dt=np.clip(v[:,1]-v[:,0],-.35,.35); reliable=1/(1+(np.maximum(mad[:,0],mad[:,1])/(.024+.18*abs(dt)))**2)
    dx=dt*reliable/.78; conf=1/(.010+mad)**2
    off=(v[:,0]*conf[:,0]+v[:,1]*conf[:,1])/(conf[:,0]+conf[:,1]); off=off+dx*.39*(conf[:,0]-conf[:,1])/(conf[:,0]+conf[:,1]);
    dy=np.clip((v[:,2:]-v[:,:2])/.21,-.35,.35); wy=1/(.015+mad[:,:2]+mad[:,2:]+.17*abs(dy))**2;dy=np.sum(dy*wy,1)/np.sum(wy,1)
    return off[:,None,None]+dx[:,None,None]*t[None,None,:]+dy[:,None,None]*np.clip(t+.39,-.2,.69)[None,:,None]

def fillmask(x):
    # Local tangents to the smooth background, cast inwards from several
    # borders. A ray loses credibility on crossing a textured surface.
    dx,dy=deriv(blur(x)); mag=np.sqrt(dx*dx+dy*dy)
    sm=blur(x); n=len(x); t=np.arange(28,dtype=np.float32)[None,:,None]
    def ray(v,gg):
        slope=np.clip(np.median(v[:,2:5]-v[:,:3],axis=1,keepdims=True)*.5,-.014,.014)
        b=np.median(v[:,:4]-slope*(t[:,:4]-1.5),axis=1,keepdims=True)
        er=np.abs(v-(b+slope*(t-1.5)));peak=np.maximum.accumulate(gg,axis=1)
        return er+.32*peak+1.4*np.mean(gg[:,:3],axis=1,keepdims=True)+.30*np.maximum.accumulate(er,axis=1)
    top=ray(sm,mag);left=ray(sm.transpose(0,2,1),mag.transpose(0,2,1)).transpose(0,2,1)
    right=ray(sm[:,:,::-1].transpose(0,2,1),mag[:,:,::-1].transpose(0,2,1)).transpose(0,2,1)[:,:,::-1]
    loc=np.minimum(np.minimum(top,left),right)
    residual=np.minimum(abs(x-background(x)),loc+.018);
    # Dark or light deviations and fine surface detail are both useful.
    contrast=np.mean(abs(residual),axis=(1,2),keepdims=True)+.03
    e=np.clip(abs(residual)/(.055+.37*contrast)-.20,0,1)
    texture=np.clip(blur(mag)/(.024+.09*contrast)-.22,0,1)
    m=np.maximum(e*.94,texture*.91)
    # Penalize large closed-looking regions supported only by a gradual
    # background slope. Local curvature/texture survives illumination ramps.
    sm=blur(x); low=sm
    for j in range(5):low=blur(low)
    local=np.minimum(1,abs(x-low)/(.020+.05*contrast))
    rad=blur(blur(np.maximum(local,texture)))
    return blur(m)*(.55+.45*np.minimum(1,rad*1.65))

def pool(a):
    v=np.pad(a,((0,0),(2,2),(2,2)),mode='edge')
    return v.reshape(len(a),8,4,8,4).mean(axis=(2,4))

def fine(a):
    return a.reshape(len(a),14,2,14,2).mean(axis=(2,4))

def desc(m,x):
    # Occupancy plus undirected, contrast-normalized local edge histograms.
    # Occupancy gets a signed correlation; orientation evidence saturates
    # so that a sleeve or stray background edge cannot dominate the score.
    dx,dy=deriv(blur(x)*.55+x*.45); mag=np.sqrt(dx*dx+dy*dy+1e-10); scale=np.mean(mag,axis=(1,2),keepdims=True)+.009
    amp=np.minimum(mag/(scale*.9),3.5)**.73
    dx=dx/mag;dy=dy/mag
    chans=[]
    for ca,sa in ((1,0),(.70710678,.70710678),(0,1),(-.70710678,.70710678)):
        chans.append(pool(amp*np.maximum(0,abs(dx*ca+dy*sa)-.7)*3.33333))
    # Give the lower corners less weight since forearm and clothing vary.
    wt=np.array([[1,1,1,1,1,1,1,1]]*5+[[.85,.95,1,1,1,1,.95,.85],[.60,.78,.88,.9,.9,.88,.78,.60],[.44,.6,.66,.7,.7,.66,.6,.44]],np.float32)
    p=pool(m);p=(p-np.mean(p,axis=(1,2),keepdims=True))*wt
    h=np.stack(chans,1);h=(h-.66*np.mean(h,axis=(2,3),keepdims=True))*wt
    p=p.reshape(len(p),-1); h=h.reshape(len(h),-1)
    p=p/np.sqrt(np.sum(p*p,axis=1,keepdims=True)+.001);h=h/np.sqrt(np.sum(h*h,axis=1,keepdims=True)+.001)
    return np.ascontiguousarray(np.concatenate([.74*p,.673*h],axis=1),np.float32)

def cavities(m):
    # Flood background in from outside: the interior of the O and F
    # contacts cannot be reached except through a high foreground barrier.
    free=1-np.clip(m,0,1);flow=np.zeros_like(free)
    flow[:,0]=free[:,0];flow[:,-1]=free[:,-1];flow[:,:,0]=free[:,:,0];flow[:,:,-1]=free[:,:,-1]
    for j in range(17):
        p=np.pad(flow,((0,0),(1,1),(1,1)),mode='edge')
        flow=np.minimum(free,np.maximum(np.maximum(p[:,1:-1,:-2],p[:,1:-1,2:]),np.maximum(p[:,:-2,1:-1],p[:,2:,1:-1])))
    return np.maximum(0,free-flow)

def detailed(m,x):
    sx=blur(x)*.6+.4*x;dx,dy=deriv(sx);g=np.sqrt(dx*dx+dy*dy+1e-9)
    low=sx
    for j in range(6):low=blur(low)
    sc=np.mean(g,axis=(1,2),keepdims=True)+.007
    channels=[fine(m),fine(np.clip((sx-low)/sc,-3.,3.))]
    for ca,sa in ((1,0),(.70710678,.70710678),(0,1),(-.70710678,.70710678)):
        channels.append(fine(blur(np.minimum(3.,np.maximum(0,(abs(dx*ca+dy*sa)-.64*g)*3/sc)))))
    channels.append(fine(cavities(m)))
    return np.stack(channels,1)

def view(a,dx,dy,scale):
    # Transform one input view to a canonical crop, clamping out-of-frame pixels.
    t=(np.arange(28,dtype=np.float32)-13.5)*scale+13.5
    x=np.clip(t+dx,0,27);y=np.clip(t+dy,0,27);i=x.astype(np.int32);j=y.astype(np.int32);sx=(x-i).astype(np.float32);sy=(y-j).astype(np.float32)
    v=a[:,:,i]*(1-sx)+a[:,:,np.minimum(i+1,27)]*sx
    return v[:,j,:]*(1-sy[:,None])+v[:,np.minimum(j+1,27),:]*sy[:,None]

bank=CONSTANTS['poses'].astype(np.float32).reshape(-1,320).T.copy()
fd=CONSTANTS['fine']

def localnorm(a):
    a=a-a.mean(axis=2,keepdims=True)
    norm=np.sqrt(.14+np.sum(a*a,axis=(1,2),keepdims=True))
    return a/norm

partweight=np.array([1.05,1.25,1.05,1.10,1.1,1.10,.65,.65,.65],np.float32)

def partscore(q,m):
    # Slide overlapping palm/finger regions independently by a single cell.
    # Normalize inside each region so a shirt or an unrelated edge in
    # another part of the image cannot suppress the evidence for a digit.
    model=np.zeros((len(m),9),np.float32); confidence=np.zeros_like(model)
    qpad=np.pad(q,((0,0),(0,0),(1,1),(1,1)),mode='edge'); ci=0
    for y in (0,4,8):
        for x in (0,4,8):
            v=m[:,:,y:y+6,x:x+6].reshape(len(m),6,36);v=v-v.mean(axis=2,keepdims=True)
            energy=np.sum(v*v,axis=(1,2))
            qq=q[:,:,y:y+6,x:x+6].reshape(len(m),6,36);qq=qq-qq.mean(axis=2,keepdims=True)
            seen=np.sum(qq*qq,axis=(1,2));confidence[:,ci]=np.minimum(1,(energy+.85*seen)/4+.08)*partweight[ci]
            v=v/np.sqrt(.15+energy[:,None,None]); s=np.full(len(m),-1.,np.float32)
            for iy,ix in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
                w=qpad[:,:,y+1+iy:y+7+iy,x+1+ix:x+7+ix].reshape(len(m),6,36)
                h=np.sum(localnorm(w)*v,axis=(1,2))-.045*(abs(iy)+abs(ix));s=np.maximum(h,s)
            model[:,ci]=s;ci+=1
    return np.sum(model*confidence,1)/(np.sum(confidence,1)+.001)
def predict(x):
    result=np.empty(len(x),np.int64)
    count=bank.shape[1]//24
    for st in range(0,len(x),16):
        # A fixed block size also fixes the reduction kernels across batch sizes.
        n=min(16,len(x)-st);a=np.zeros((16,28,28),np.float32);a[:n]=x[st:st+n,0];m=fillmask(a)
        f=[]
        for scale,delta in ((1.,2.8),(.82,3.),(1.21,2.8)):
            for dx in (-delta,0,delta):
                for dy in (-delta,0,delta):
                    f.append(desc(view(m,dx,dy,scale),view(a,dx,dy,scale)))
        scores=(np.concatenate(f)@bank).reshape(len(f),16,24,count)
        # Rerank a shortlist for each sign on the high-resolution grid.
        # Different signs compete with the same number of verifications.
        best=np.max(scores,axis=0); which=np.argmax(scores,axis=0); keep=6
        sel=np.argpartition(best,count-keep,axis=2)[:,:,-keep:];vv=np.take_along_axis(which,sel,2)
        top=np.take_along_axis(best,sel,2);f=[]
        for scale,delta in ((1.,2.8),(.82,3.),(1.21,2.8)):
            for dx in (-delta,0,delta):
                for dy in (-delta,0,delta):
                    f.append(detailed(view(m,dx,dy,scale),view(a,dx,dy,scale)))
        chan=np.array([1.9,.64,.87,.87,.87,.87],np.float32)[None,:,None,None]; fineviews=np.stack(f,1);rows=np.arange(24)[:,None]
        for idx in range(n):
            vec=fd[rows,sel[idx]].astype(np.float32).reshape(-1,7,14,14)/42
            q=fineviews[idx,vv[idx]].reshape(-1,7,14,14)
            hole=q[:,6];ring=vec[:,6];support=ring.sum((1,2));observed=hole.sum((1,2));overlap=(ring*hole).sum((1,2))
            evidence=(1.7*overlap/(.9+support+observed)-.18*np.minimum(1,support/3)).reshape(24,keep)
            vec=vec[:,:6];q=q[:,:6];t=partscore(q*chan,vec*chan).reshape(24,keep)
            # A global symmetric comparison penalizes an observed finger in
            # the blank area of a fist model (or a model finger in air).
            q=q-q.mean(axis=(2,3),keepdims=True);vec=vec-vec.mean(axis=(2,3),keepdims=True)
            q=q*chan;vec=vec*chan
            rigid=(np.sum(q*vec,axis=(1,2,3))/np.sqrt(.01+np.sum(q*q,axis=(1,2,3))*np.sum(vec*vec,axis=(1,2,3)))).reshape(24,keep)
            both=.30*top[idx]+.40*t+.30*rigid+.085*evidence
            result[st+idx]=np.argmax(.80*np.max(both,1)+.20*np.mean(both,1))
    return result
