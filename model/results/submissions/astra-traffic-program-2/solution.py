import numpy as np

poses=CONSTANTS['poses']
delta=CONSTANTS['delta']

def mesh(n,scale):
    t=(np.arange(n,dtype=np.float32)-(n-1)/2)*scale
    return np.meshgrid(t,t)
coarsex,coarsey=mesh(18,32/18)
finex,finey=mesh(29,1/11)
iconx,icony=mesh(25,1/14)

def sample(v,x,y):
    n=len(v)
    if x.ndim==2:x=x[None];y=y[None]
    x=np.clip(x+15.5,0,30.99999);y=np.clip(y+15.5,0,30.99999)
    ix=x.astype(np.int32); iy=y.astype(np.int32);dx=x-ix;dy=y-iy
    j=np.arange(n)[:,None,None]; k=ix+iy*32
    a=v.reshape(n,-1,1024).transpose(0,2,1)
    d=dx[:,:,:,None];e=dy[:,:,:,None]
    return (((1-d)*a[j,k]+d*a[j,k+1])*(1-e)+((1-d)*a[j,k+32]+d*a[j,k+33])*e).transpose(0,3,1,2).astype(np.float32)

def view(v,p,u,w):
    a,b,s,t,h,k=[p[:,i,None,None] for i in range(6)]
    return sample(v,a+s*(u+h*w),b+t*(w+k*u))

def features(x):
    # Soft reflectance coordinates: luminance and opponents scaled between
    # dark and bright pixels by a log transform. Remove part of the cast.
    r,g,b=np.log(x+.027).transpose(1,0,2,3)
    lum=.24*r+.53*g+.23*b
    flat=lum[:,:, :].reshape(len(x),-1)
    lo,hi=np.percentile(flat,[10,90],axis=1).astype(np.float32)[:,:,None,None]
    lum=np.clip((lum-lo)/(.19+hi-lo),-.55,1.55)
    red=r-.65*g-.35*b; blue=b-.51*r-.49*g; yellow=.51*r+.49*g-b
    rgb=[]
    for ch in (red,blue,yellow):
        off=np.percentile(ch.reshape(len(x),-1),25,axis=1).astype(np.float32)[:,None,None]
        rgb.append(np.clip((ch-.68*np.clip(off,-.65,.65))/.66,-1.2,1.8))
    return np.stack([lum]+rgb,axis=1)

def prob(v):return np.minimum(np.maximum(v,0),1)

def geometry(f,maps,mode):
    n=len(maps); flat=maps.reshape(n*4,-1)
    q=(flat@CONSTANTS[mode+str(f)]).reshape(n,4,-1,3)
    rim,inside,outside=q[:,:,:,0],q[:,:,:,1],q[:,:,:,2]
    l,r,b,y=rim.transpose(1,0,2); li,ri,bi,yi=inside.transpose(1,0,2); le,re,be,ye=outside.transpose(1,0,2)
    if f in (0,1,2):
        ns=6 if f==0 else 3
        differential=(.48*maps[:,1]-.52*maps[:,0]).reshape(n,-1)
        parts=(differential@CONSTANTS['s'+mode+str(f)]).reshape(n,-1,ns)
        evidence=parts.mean(2)*.60+parts.min(2)*.40
        full=.38*(li-l)+.24*(r-ri)+.15*r+.13*(r-re)+.08*abs(l-le)+.08*(li-le)-.06*np.maximum(ri,0)
        return .55*full+.55*evidence
    if f==3:
        return .31*b+.12*bi+.21*(b-be)+.12*(bi-be)+.12*abs(l-le)+.15*(li-l)
    if f==4 or f==6:
        return .26*r+.16*ri+.22*(r-re)+.08*(ri-re)+.09*abs(l-le)+.13*(li-l)
    if f==5:
        return .35*(y-yi)+.23*y+.11*(y-ye)+.34*(li-l)+.12*(li-le)-.09*np.maximum(r,0)
    diag=(-maps[:,0].reshape(n,-1))@CONSTANTS['d'+mode]
    return .28*(l-le)+.12*(l-li)+.08*abs(l-le)+.45*diag-.18*np.maximum(r,0)-.15*np.maximum(b,0)

def combine(p,d):
    v=p.copy();v[:,0]+=p[:,2]*d[:,0];v[:,1]+=p[:,3]*d[:,1];v[:,2:4]*=d[:,2:4];v[:,4:6]=d[:,4:6];return v

def locate(maps):
    n=len(maps);idx=np.arange(n); small=sample(maps,coarsex,coarsey); params=[];scores=[];backup=[];secondary=[]
    for f in range(8):
        g=geometry(f,small,'c')-.007*(np.abs(poses[:,0])+np.abs(poses[:,1]))-.012*np.abs(poses[:,2]-poses[:,3])-.004*np.abs(poses[:,2]+poses[:,3]-25)
        p=poses[g.argmax(1)]; modes=2 if f in (0,1) else 1
        for k in range(modes):
            if k:
                # Retain a spatially distinct alternative for occluded borders.
                dif=abs(poses[None,:,:4]-p[:,None,:4]);far=dif[:,:,:2].sum(2)+.65*dif[:,:,2:].sum(2)>3.5
                g=g-5*(~far);p=poses[g.argmax(1)]
            crop=view(maps,p,finex,finey)
            h=geometry(f,crop,'f')-.075*(abs(delta[:,4])+abs(delta[:,5]))-.045*(abs(delta[:,0])+abs(delta[:,1]))
            j=h.argmax(1); fit=combine(p,delta[j]);sc=.25*g.max(1)+.75*h[idx,j]
            if k:backup.append(fit);secondary.append(sc)
            else:params.append(fit); scores.append(sc)
    return params+backup,np.stack(scores+secondary,axis=1)

def section(v,group):
    a=v[:,CONSTANTS['m'+str(group)]]
    a=a-a.mean(axis=1,keepdims=True); sig=np.sqrt((a*a).mean(axis=1));return a/np.sqrt((a*a).sum(axis=1,keepdims=True)+1e-8),sig

classes=[[0,1,2,3,4,5,7,8,9,10,15,16],[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31],[33,34,35,36,37,38,39,40],[6,32,41,42]]

def inkread(maps,p,group):
    l,r,b,y=view(maps,p,iconx,icony).transpose(1,0,2,3)
    if group==0:
        mask=iconx**2+icony**2<.73**2; mask=mask&(abs(icony)<.57); cy=0.
    else:
        mask=(1+icony-2*abs(iconx)>.40)&(icony<.77)&(icony>-.63);cy=.20
    # Use red-reflectance to separate dark ink from red paint. Soft threshold
    # relative to the paper and ink levels within this candidate sign only.
    v=-l-.26*r
    paper,ink=np.percentile(v[:,mask],[24,92],axis=1).astype(np.float32)[:,:,None,None]
    cut=paper+.34*(ink-paper);mass=mask*np.minimum(np.maximum(v-cut,0),np.maximum(ink-cut,.001))
    total=mass.sum((1,2))+.000001;mx=(mass*iconx).sum((1,2))/total;my=(mass*icony).sum((1,2))/total
    sx=np.sqrt((mass*(iconx-mx[:,None,None])**2).sum((1,2))/total);sy=np.sqrt((mass*(icony-my[:,None,None])**2).sum((1,2))/total)
    dx=np.clip(mx,-.19,.19);dy=np.clip(my,cy-.25,cy+.25)
    pp=p.copy();pp[:,0]+=p[:,2]*(dx+p[:,4]*dy);pp[:,1]+=p[:,3]*(dy+p[:,5]*dx)
    sx=np.clip(2*sx,.11,.75);sy=np.clip(2*sy,.10,.65);pp[:,2]*=sx;pp[:,3]*=sy;pp[:,4]*=sy/sx;pp[:,5]*=sx/sy
    l,r,b,y=view(maps,pp,iconx*14/12,icony*14/12).transpose(1,0,2,3)
    v=-l+.06*r if group==0 else -l-.28*r
    mask=CONSTANTS['inkmask'];z=v[:,mask];z=z-z.mean(1,keepdims=True);z/=np.sqrt((z*z).sum(1,keepdims=True)+1e-8)
    gy,gx=np.gradient(v,axis=(1,2));gy=gy[:,mask];gx=gx[:,mask];en=np.sqrt((gx*gx+gy*gy).sum(axis=1,keepdims=True)+1e-8)
    sc=.68*(z@CONSTANTS['ink'])+.32*(gx@CONSTANTS['inkx']+gy@CONSTANTS['inky'])/en
    return sc,(ink-paper).reshape(-1)

def edges(v,group):
    # Signed gradient matching responds to stroke edges but not an additive
    # illumination offset. Average neighbouring pixels to suppress aliasing.
    z=np.pad(v,((0,0),(1,1),(1,1)),mode='edge')
    sm=(z[:,1:-1,1:-1]*4+2*(z[:,:-2,1:-1]+z[:,2:,1:-1]+z[:,1:-1,:-2]+z[:,1:-1,2:])+z[:,:-2,:-2]+z[:,2:,:-2]+z[:,:-2,2:]+z[:,2:,2:])/16
    gy,gx=np.gradient(sm,axis=(1,2));mk=CONSTANTS['m'+str(group)];gx=gx[:,mk];gy=gy[:,mk]
    en=np.sqrt(np.sum(gx*gx+gy*gy,axis=1,keepdims=True)+1e-8)
    return (gx@CONSTANTS['x'+str(group)]+gy@CONSTANTS['y'+str(group)])/en

def recognize(x):
    n=len(x); maps=features(x);ps,geo=locate(maps);scores=np.full((n,43),-10,np.float32)
    for group,f in [(0,0),(1,1),(2,3),(3,7),(0,8),(1,9)]:
        crop=view(maps,ps[f],iconx,icony); l,r,b,y=crop.transpose(1,0,2,3)
        if group==0 or group==1:
            # Suppress the red painted border, using its brightness in red
            # light rather than green. Retain red vehicles in circular signs.
            v=-l+.06*r if group==0 else -l-.39*r
        if group==2:v=l-.28*b
        if group==3:v=-l
        a,sig=section(v,group);corr=a@CONSTANTS['i'+str(group)]-CONSTANTS['p'+str(group)]
        if group in (0,1):
            corr=.67*corr+.33*edges(v,group)
            inkcorr,contrast=inkread(maps,ps[f],group)
        labs=CONSTANTS['l'+str(group)]; base=geo[:,f];w=.85
        if f>=8:base=base-.035
        fresh=np.full_like(scores,-10)
        for c in classes[group]:
            if c==15:
                if f==0:fresh[:,c]=base+w*(.70-2.35*sig-.56*contrast)
                continue
            s=corr[:,labs==c].max(1)
            if group in (0,1):
                ic=inkcorr[:,CONSTANTS['incl']==c].max(1)
                s=.50*s+.28*ic+.22*np.maximum(s,ic)
            if group==3:fresh[:,c]=.66*base+1.23*s-.06
            else:fresh[:,c]=base+w*s
        if group==1:
            # A German traffic-light warning has three coloured lights. Compare
            # the upper and lower lobes to one another, canceling colour cast.
            dx=abs(iconx)<.125
            top=dx&(icony>-.35)&(icony<.0);btm=dx&(icony>.32)&(icony<.62)
            clr=(r[:,top].mean(1)-r[:,btm].mean(1))*.70+(y[:,top].mean(1)-y[:,btm].mean(1))*.20
            fresh[:,26]+=.35*np.clip(clr-.10,-.35,.8)
        scores=np.maximum(scores,fresh)
    # Special, mostly uniform signs. Use their known spatial divisions too.
    yl,yr,yb,yy=view(maps,ps[2],iconx,icony).transpose(1,0,2,3)
    mask=(icony>-.76)&(1-icony-2*abs(iconx)>.39)
    ink=(yl-.15*yr)[:,mask]
    # An inverted triangle must have an empty, uniform interior, rather
    # than a speed numeral or an upright triangular pictogram.
    scores[:,13]=geo[:,2]+.61-1.6*ink.std(1)
    dl,dr,db,dy=view(maps,ps[5],iconx,icony).transpose(1,0,2,3)
    mask=abs(iconx)+abs(icony)<.60
    scores[:,12]=geo[:,5]+.60-1.05*dl[:,mask].std(1)-.45*dy[:,mask].std(1)
    rad=np.sqrt(iconx**2+icony**2)
    for f,c in [(4,14),(6,17)]:
        l,r,b,y=view(maps,ps[f],iconx,icony).transpose(1,0,2,3)
        v=l-.35*r
        a,sig=section(v,4); mask=CONSTANTS['l4']==c
        s=(a@CONSTANTS['i4'][:,mask]).max(1)
        # The quadrants above and below the word or stripe must be red, too.
        patches=[]
        for side in [-1,1]:
            for vert in [-1,1]:
                m=(iconx*side>0)&(icony*vert>.38)&(rad<.79)
                patches.append(r[:,m].mean(1))
        whole=np.stack(patches,axis=1).min(1)
        scores[:,c]=.79*geo[:,f]+.13*whole+.85*s
        if c==17:
            # A no-entry stripe is uniformly bright; letters/digits create
            # dark holes, even if the coarse stripe correlation is high.
            bar=v[:,(abs(iconx)<.62)&(abs(icony)<.135)]
            hi=v[:,(abs(icony)>.34)&(abs(icony)<.57)&(abs(iconx)<.55)]
            scores[:,c]+=.16*(bar.mean(1)-hi.mean(1))-.78*bar.std(1)-.28*hi.std(1)
    return scores.argmax(1)

def predict(x):
    res=np.empty(len(x),np.int64)
    # Fixed-size work blocks make matrix reductions independent of the caller's batch.
    for i in range(0,len(x),64):
        b=x[i:i+64]; n=len(b)
        if n<64:b=np.concatenate((b,np.repeat(b[-1:],64-n,axis=0)),axis=0)
        res[i:i+n]=recognize(b)[:n]
    return res
