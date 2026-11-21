import numpy as np

def blur(a):
    p = np.pad(a,((0,0),(1,1),(1,1)),mode="edge")
    b=(p[:,:-2]+2*p[:,1:-1]+p[:,2:])*.25
    return (b[:,:,:-2]+2*b[:,:,1:-1]+b[:,:,2:])*.25

def pooled(a,n):
    # Bilinear spatial voting, equivalent to a triangular cell window.
    pts=(np.arange(n)+.5)*28/n-.5
    w=np.maximum(1-np.abs(pts[:,None]-np.arange(28))/(28/n),0).astype(np.float32)
    w/=w.sum(1,keepdims=True)
    return w @ a @ w.T

def vector(a):
    return a.reshape(len(a),-1)

def unit(a):
    return a / np.sqrt(np.sum(a*a,axis=1,keepdims=True)+1e-10)

def describe(a):
    smooth=blur(a); grad=.55*a+.45*smooth
    pad = np.pad(grad,((0,0),(1,1),(1,1)),mode="edge")
    dx=(pad[:,1:-1,2:]-pad[:,1:-1,:-2]); dy=pad[:,2:,1:-1]-pad[:,:-2,1:-1]
    mag=np.sqrt(dx*dx+dy*dy+1e-10)
    # Clip rare high-contrast texture boundaries. Keep smooth regions weak.
    ref=np.sqrt(np.mean(mag*mag,axis=(1,2),keepdims=True))+.008
    strength=np.minimum(mag/ref,2.5)**.75
    dx/=mag;dy/=mag
    ax=np.linspace(-1,1,28,dtype=np.float32)
    weight=1-.30*np.abs(ax)[None,:]**3-.21*((ax[:,None]+1)*.5)**4
    strength*=weight
    # Separate fine, directed finger detail from large silhouettes. These
    # oriented even/odd filters respond to ridges as well as to step edges.
    p=np.pad(a,((0,0),(3,3),(3,3)),mode="edge")
    def shift(oy,ox): return p[:,3+oy:31+oy,3+ox:31+ox]
    fingers=[]
    for oy,ox in ((0,1),(1,0),(1,1),(1,-1)):
        t1=shift(oy,ox);t2=shift(-oy,-ox)
        d1=(t1-t2); d2=t1+t2-2*a
        # Average ridge energy along the hypothetical long axis of the digit.
        # The principal fine-scale response depends on width, not skin tone.
        sig=np.minimum(2.5,np.sqrt(.42*d1*d1+1.6*d2*d2)/(ref+.01))
        pp=np.pad(sig,((0,0),(2,2),(2,2)),mode="edge")
        ridge=(2*sig+pp[:,2+ox:30+ox,2-oy:30-oy]+pp[:,2-ox:30-ox,2+oy:30+oy])*.25
        fingers.append(vector(pooled(ridge*weight,7)))
    orient=[]; signed=[]
    for i in range(4):
        phi=i*np.pi/4; dot=dx*np.cos(phi)+dy*np.sin(phi)
        ch=(strength*np.maximum(np.abs(dot)-.57,0)).astype(np.float32)
        orient.append(ch);signed.append(vector(pooled(ch*np.sign(dot),9)))
    fine=[];coarse=[]
    for ch in orient:
        fine.append(vector(pooled(ch,10)));coarse.append(vector(pooled(ch,6)))
    # Soft detection of relief: departures from a locally smooth background.
    # A photo's background may be darker OR lighter than the skin.
    wide=smooth
    for k in range(8):wide=blur(wide)
    hp=np.clip((smooth-wide)/(ref+.02),-2,2)
    dark=pooled(np.maximum(-hp,0)*weight,9);lite=pooled(np.maximum(hp,0)*weight,9)
    edge=pooled(strength,8)
    # Propagate confidence in the background from the outside edges. Penalize
    # paths crossing high gradients, not the absolute color at the endpoint:
    # this can follow continuous changes of illumination across a wall.
    barrier=np.minimum(1,(mag/(ref*1.20))**1.6)
    distance=np.ones_like(a)
    distance[:,0,:]=barrier[:,0,:];distance[:,:,0]=barrier[:,:,0];distance[:,:,-1]=barrier[:,:,-1]
    for k in range(16):
        p=np.pad(distance,((0,0),(1,1),(1,1)),mode="edge")
        neigh=np.minimum(np.minimum(p[:,1:-1,2:],p[:,1:-1,:-2]),np.minimum(p[:,2:,1:-1],p[:,:-2,1:-1]))
        distance=np.minimum(distance,np.maximum(neigh,barrier))
    support=pooled((.75*distance+.25*np.minimum(1,strength))*weight,8)
    def contrast(b):
        return unit(b-.85*np.mean(b,axis=1,keepdims=True))
    h=contrast(np.concatenate(fine,axis=1));g=contrast(np.concatenate(coarse,axis=1));sg=unit(np.concatenate(signed,axis=1))
    bands=np.concatenate([vector(dark),vector(lite)],axis=1)
    ridgevec=contrast(np.concatenate(fingers,axis=1))
    return np.concatenate((.50*h,.29*g,.37*sg,.31*contrast(vector(edge)),.34*contrast(vector(support)),.38*contrast(bands),.34*ridgevec),axis=1).astype(np.float32)

def keyplanes(a):
    b=blur(a); c=blur(b); w=c
    for k in range(5): w=blur(w)
    p=np.pad(.5*(a+b),((0,0),(1,1),(1,1)),mode="edge")
    dx=p[:,1:-1,2:]-p[:,1:-1,:-2];dy=p[:,2:,1:-1]-p[:,:-2,1:-1]
    m=np.sqrt(dx*dx+dy*dy+.000016); ref=np.sqrt(np.mean(m*m,axis=(1,2),keepdims=True))+.01
    volume=[]
    for i in range(4):
        angle=i*np.pi/4;g=(dx*np.cos(angle)+dy*np.sin(angle)).astype(np.float32)
        volume.append(blur(np.maximum(np.abs(g)-.58*m,0)/ref))
    volume.append(blur(np.maximum(w-b,0)/ref));volume.append(blur(np.maximum(b-w,0)/ref))
    volume=np.stack(volume,axis=1)
    return np.minimum(volume,1.6)

def localfield(planes):
    # Endpoint validation: soft maximum in a small neighborhood tolerates
    # foreshortening and articulation error between nearby poses.
    q=planes.shape
    p=np.pad(planes,((0,0),(0,0),(1,1),(1,1)),mode="edge")
    z=np.maximum.reduce([p[:,:,j:j+28,i:i+28] for i,j in ((0,1),(1,0),(1,1),(1,2),(2,1))])
    z=.55*planes+.45*z
    z=z/(z+.18)
    return z.reshape(q[0],-1)

def predict(x):
    ans=np.empty(len(x),dtype=np.int64)
    # Padding makes the matrix product and all reductions identical for a
    # given image even when its position in the caller's batch changes.
    count=64
    for start in range(0,len(x),count):
        a=x[start:start+count,0];n=len(a); block=np.zeros((count,28,28),dtype=np.float32);block[:n]=a
        d=describe(block); scores=[]; planes=keyplanes(block);field=localfield(planes)
        quiet=np.max(planes[:,:4],axis=1)+.7*np.maximum(planes[:,4],planes[:,5]);quiet=(quiet/(quiet+.23)).reshape(count,784)
        align=CONSTANTS["align"]; poses=CONSTANTS["points"]
        lean=CONSTANTS["lean"]; penalty=CONSTANTS["penalty"]
        rows=np.arange(count)[:,None,None]
        for label in range(24):
            bank=CONSTANTS[str(label)]
            sim=d @ bank
            ids=np.argpartition(sim,-20,axis=1)[:,-20:]
            kp=poses[label][ids]; bg=CONSTANTS["spaces"][label][ids];base=np.take_along_axis(sim,ids,axis=1)
            # A small global translation, plus independent small motions of the
            # upper and lower articulations, are nuisance variables.
            best=np.full(base.shape,-2,dtype=np.float32)
            for j in range(len(align)):
                up=field[rows,align[j][kp[:,:,:32]]].mean(2)
                dn=field[rows,align[j][kp[:,:,32:]]].mean(2)
                if j==0: fixedup=up;fixeddn=dn
                v=.5*(np.maximum(up,fixedup-lean[j])+np.maximum(dn,fixeddn-lean[j]))-penalty[j]
                stray=quiet[rows,align[j][bg]]
                # Permit partial occlusion of the nominal background.
                v-=.37*np.mean(np.minimum(stray,.60),axis=2)
                best=np.maximum(best,v)
            merits=base+.70*best
            top=np.partition(merits,-4,axis=1)[:,-4:]
            scores.append(.45*np.max(top,axis=1)+.55*np.mean(top,axis=1))
        ans[start:start+n]=np.argmax(np.stack(scores,axis=1),axis=1)[:n]
    return ans
