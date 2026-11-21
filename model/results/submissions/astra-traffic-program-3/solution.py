import numpy as np

families = [[0,1,2,3,4,5,7,8,9,10,15,16],[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31],[33,34,35,36,37,38,39,40],[6,32,41,42],[12],[13],[14],[17]]

def contrast(z):
    q=np.percentile(z,[8,92],axis=(1,2)).astype(np.float32)
    return np.clip((z-q[0,:,None,None])/(.027+q[1,:,None,None]-q[0,:,None,None]),-.28,1.35)

def fields(x):
    r,g,b=x[:,0],x[:,1],x[:,2]
    log=np.log(x+.019)
    lr,lg,lb=log[:,0],log[:,1],log[:,2]
    # Chromatic opponent channels: soft membership, robust to illumination.
    red=np.clip((lr-.76*lg-.24*lb-.035)*1.72,-.28,1.2)
    blue=np.clip((lb-.58*lr-.42*lg-.095)*1.8,-.3,1.2)
    yellow=np.clip((.5*(lr+lg)-lb-.14-1.4*np.maximum(lr-lg-.25,0))*1.8,-.26,1.2)
    y=contrast(.68*lg+.18*lr+.14*lb)
    return [red,blue,yellow,y],log

def positions(maps):
    a,b,c,y=[z.reshape(len(z),16,2,16,2).mean(axis=(2,4)).reshape(len(z),256) for z in maps]
    poses=CONSTANTS['poses'];cx,cy,rx,ry,s=poses.T
    prior=.013*((cx-15.5)**2+(cy-15.5)**2)/8+.008*((ry-12)/2.5)**2+.012*(abs(rx/ry-1)+abs(s))
    scores=[];align=[]
    for g in range(8):
        def op(m,j):return m@CONSTANTS['p'+str(g)+str(j)]
        if g in (0,1,5):
            col=op(a,0);dc=op(a,1);dy=-op(y,1); edge=op(a,3)
            h=.27*col+.64*dc+.70*dy+.24*edge+.16*op(y,2)
        elif g==4:
            # Yellow diamond with white surround: reverse chromatic contrast.
            col=op(c,0);dc=-op(c,1);silhouette=y@CONSTANTS['dia'];h=.66*col+.62*dc+.22*op(y,1)+.20*op(y,2)+.28*silhouette
        elif g==3:
            h=.76*op(y,1)+.27*op(y,2)-.24*op(a,0)-.16*op(b,0)+.26
        else:
            m=b if g==2 else a
            h=.55*op(m,0)+.52*op(m,1)+(.30 if g==2 else .59)*op(y,2)
        h=h-prior
        best=np.argmax(h,axis=1);align.append(poses[best]);scores.append(h[np.arange(len(a)),best])
    return np.stack(scores,1),np.stack(align,1)

grid=(np.arange(26,dtype=np.float32)+.5)/26*2-1
U,V=np.meshgrid(grid,grid)
U=U.reshape(1,-1);V=V.reshape(1,-1)

def register(maps,pose,g):
    # Local coordinate search for the sign outline, on the individual image.
    probe=CONSTANTS['probe'+str(g)];u=probe[:,:,0].reshape(1,-1);v=probe[:,:,1].reshape(1,-1);n=len(pose)
    a,blue,yellow,y=maps;signal=.53*a-.47*y
    def rate(pp):
        cx,cy,rx,ry,sl,tilt=pp.T[:,:,None];xx=np.clip(cx+rx*(u+sl*v),0,30.999);yy=np.clip(cy+ry*(v+tilt*u),0,30.999)
        ix=xx.astype(np.int32);iy=yy.astype(np.int32); ax=(xx-ix).astype(np.float32);ay=(yy-iy).astype(np.float32);i=np.arange(n)[:,None]
        z=((1-ax)*((1-ay)*signal[i,iy,ix]+ay*signal[i,iy+1,ix])+ax*((1-ay)*signal[i,iy,ix+1]+ay*signal[i,iy+1,ix+1])).reshape(n,4,-1)
        inner=np.clip(z[:,1]-z[:,0],-.25,.7);outer=np.clip(z[:,1]-.5*(z[:,2]+z[:,3]),-.18,.52)
        return (inner+.40*outer).mean(1)-.0018*((cx[:,0]-15.5)**2+(cy[:,0]-15.5)**2)/4-.009*(np.abs(sl[:,0])+np.abs(tilt[:,0]))
    best=np.column_stack((pose,np.zeros(n,np.float32)));q=rate(best)
    for step in [1.0,.55,.28]:
        for d in [0,1,2,3,4,5]:
            base=best.copy()
            for direction in [-1,1]:
                p=base.copy();p[:,d]+=step*direction*(.072 if d>=4 else 1.)
                score=rate(p);take=score>q;q=np.maximum(q,score);best=np.where(take[:,None],p,best)
    return best

def rectify(field,pose):return warp(field,pose,U,V)

def warp(field,pose,u,v):
    cx,cy,rx,ry,sl=pose[:,:5].T[:,:,None]
    tilt=pose[:,5,None] if pose.shape[1]==6 else 0
    xx=np.clip(cx+rx*(u+sl*v),0,30.999);yy=np.clip(cy+ry*(v+tilt*u),0,30.999)
    ix=xx.astype(np.int32);iy=yy.astype(np.int32); a=(xx-ix).astype(np.float32);b=(yy-iy).astype(np.float32)
    n=np.arange(len(cx))[:,None]
    return (1-a)*((1-b)*field[n,iy,ix]+b*field[n,iy+1,ix])+a*((1-b)*field[n,iy,ix+1]+b*field[n,iy+1,ix+1])

def readnumbers(z,pose):
    dx,dy=np.meshgrid((np.arange(11,dtype=np.float32)+.5)*2/11-1,(np.arange(15,dtype=np.float32)+.5)*2/15-1)
    dx=dx.ravel();dy=dy.ravel();n=len(pose);w=CONSTANTS['dw'];bank=CONSTANTS['digits'];labels=CONSTANTS['dlabel']
    # Find the best common size and spacing for each inscription. No part of
    # one photograph is ever compared with another photograph.
    readings=[]
    for count in [2,3]:
        slots=[];targets=[0,2,3,5,6,7,8] if count==2 else [0,1,2]
        for s,tabular in ([(a,False) for a in [.84,1.,1.16,1.30]] if count==2 else [(a,b) for a in [.85,1.,1.16,1.30] for b in [False,True]]):
            centers=[-.262,.262] if count==2 else ([-.40,0.,.40] if tabular else [-.405,-.065,.362])
            widths=[.30,.30] if count==2 else ([.215,.222,.222] if tabular else [.191,.242,.242])
            results=[]
            for cx,width in zip(centers,widths):
                y=warp(z,pose,s*(cx+width*dx),s*.53*dy)
                q=np.percentile(y,[5,87],axis=1).astype(np.float32);y=np.clip(y,q[0,:,None]-.16*(q[1]-q[0])[:,None],q[1,:,None]);y-=np.sum(y*w,1)[:,None]/w.sum()
                y/=-np.maximum(np.sqrt(np.sum(y*y*w,1))[:,None],.002)
                use=targets if len(results)!=count-1 else [0]
                result={}
                for digit in use:
                    start=np.searchsorted(labels,digit);stop=np.searchsorted(labels,digit,side='right');result[digit]=(y@bank[:,start:stop]).max(1)
                results.append(result)
            if count==2: slots.append(np.stack([(.60*results[0][i]+.40*results[1][0]) for i in [2,3,5,6,7,8]],1))
            else:slots.append(np.stack([(.17*results[0][1]+.50*results[1][i]+.33*results[2][0]) for i in [0,2]],1))
        readings.append(np.max(np.stack(slots),axis=0))
    return np.concatenate(readings,axis=1)

def segmented(z,pose):
    h=(np.arange(23,dtype=np.float32)-11)/20;v=(np.arange(23,dtype=np.float32)-11)/19
    xx,yy=np.meshgrid(h*1.21,v);n=len(pose);pic=warp(z,pose,xx.ravel(),yy.ravel()).reshape(n,23,23)
    # Whiten against a local paper level. In particular, a shadow over the
    # entire left half should not create a spurious first character.
    paper=np.maximum(np.percentile(pic,86,axis=1)[:,None,:],np.percentile(pic,65,axis=(1,2))[:,None,None]);val=np.maximum(paper-pic,0)
    strong=np.percentile(val,93,axis=(1,2));ink=np.clip(val/(strong[:,None,None]+.002)-.18,0,1.)
    # Use the ink projections to locate the inscription and the valleys
    # separating the digits. Bounds are soft to accommodate motion blur.
    proj=ink[:,4:-4].sum(1);proj=np.maximum(proj-.14*proj.max(1)[:,None],0);total=proj.sum(1)+1.e-6;cum=np.cumsum(proj,1)/total[:,None]
    left=np.minimum((cum<.019).sum(1),10);right=np.maximum(22-(cum>.981).sum(1),12)
    rows=ink[:,:,3:-3].sum(2);rows=np.maximum(rows-.17*rows.max(1)[:,None],0);cs=np.cumsum(rows,1)/(rows.sum(1)[:,None]+1.e-6);top=np.minimum((cs<.025).sum(1),8);bot=np.maximum(22-(cs>.975).sum(1),14)
    xs=xx[0];ys=yy[:,0];topY=ys[top]-.03;botY=ys[bot]+.03
    dx,dy=np.meshgrid((np.arange(11,dtype=np.float32)+.5)*2/11-1,(np.arange(15,dtype=np.float32)+.5)*2/15-1)
    dw=CONSTANTS['dw'];bank=CONSTANTS['digits'];labels=CONSTANTS['dlabel'];a=np.arange(23)[None,:];span=np.maximum(right-left,5);pos=(a-left[:,None])/span[:,None]
    def cut(p):
        loss=proj/(proj.max(1)[:,None]+1.e-5)+.62*((pos-p)/.20)**2
        return np.argmin(loss,axis=1)
    def evidence(lo,hi,digits):
        xlo=xs[lo];xhi=xs[hi];cen=(xlo+xhi)*.5;rangeX=np.maximum(.065,(xhi-xlo)*.58);cy=(topY+botY)*.5;rangeY=(botY-topY)*.62
        data=warp(z,pose,cen[:,None]+dx.ravel()*rangeX[:,None],cy[:,None]+dy.ravel()*rangeY[:,None]);data-=np.sum(data*dw,1)[:,None]/dw.sum();data/=-np.maximum(.001,np.sqrt(np.sum(data*data*dw,1))[:,None]);out={}
        for k in digits:
            i=np.searchsorted(labels,k);j=np.searchsorted(labels,k,side='right');out[k]=(data@bank[:,i:j]).max(1)
        return out
    # Verify each proposed gap is white along most of the character height.
    def gap(i):return 1-np.clip(proj[np.arange(n),i]/(proj.max(1)+1.e-5),0,1)
    s=cut(.50);lef=evidence(left,s,[2,3,5,6,7,8]);rt=evidence(s,right,[0]);two=np.stack([.58*lef[i]+.42*rt[0]+.095*(gap(s)-.75) for i in [2,3,5,6,7,8]],1)
    s1=cut(.27);s2=cut(.64);one=evidence(left,s1,[1]);mid=evidence(s1,s2,[0,2]);rt=evidence(s2,right,[0]);three=np.stack([.18*one[1]+.48*mid[i]+.34*rt[0]+.095*(.5*(gap(s1)+gap(s2))-.75) for i in [0,2]],1)
    return np.concatenate([two,three],1)

def symbols(g,gray,pose):
    ix=CONSTANTS['sel'+str(g)]; w=CONSTANTS['w'+str(g)]
    z=rectify(gray,pose)[:,ix];q=np.percentile(z,[5,85],axis=1).astype(np.float32)
    spread=q[1]-q[0];z=np.clip(z,q[0,:,None]-.2*spread[:,None],q[1,:,None]+.3*spread[:,None]);avg=np.sum(z*w,1)/w.sum();z-=avg[:,None]
    dev=np.sqrt(np.sum(z*z*w,1)/w.sum())
    if g in (0,1,3):z=-z
    z/=np.maximum(dev[:,None]*np.sqrt(w.sum()),.001)
    t=z@CONSTANTS['bank'+str(g)]
    if g in (1,3):
        sm=z@CONSTANTS['sm'+str(g)];sm-=np.sum(sm*w,1)[:,None]/w.sum()
        sm/=np.maximum(np.sqrt(np.sum(w*sm*sm,1))[:,None],1.e-6)
        grad=(z[:,CONSTANTS['e1'+str(g)]]-z[:,CONSTANTS['e2'+str(g)]])*CONSTANTS['edgeweight'+str(g)]
        grad/=np.maximum(np.sqrt(np.sum(grad*grad,1))[:,None],1.e-6)
        t=.50*t+.28*(sm@CONSTANTS['low'+str(g)])+.22*(grad@CONSTANTS['grad'+str(g)])
    if g==1:
        zones=CONSTANTS['zones'];means=(z@zones.T)/zones.sum(1);energy=np.maximum((z*z)@zones.T-means*means*zones.sum(1),0)
        ref=CONSTANTS['wlocal'];delta=np.abs(np.sqrt(energy+.005)[:,:,None]-np.sqrt(ref+.005)[None,:,:]).sum(1)
        t-=.17*delta
    t-=CONSTANTS['bias'+str(g)]
    labs=CONSTANTS['ix'+str(g)];ss=[]
    for k in families[g]:
        if k==15:
            # A genuinely blank disk has no dark strokes in its central box.
            center=z[:,(np.abs(U[0,ix])<.56)&(np.abs(V[0,ix])<.44)]
            variation=np.percentile(center,95,axis=1)-np.percentile(center,15,axis=1)
            ss.append(.71-2.90*dev-.18*np.clip(variation*np.sqrt(w.sum())-1.,0,3))
        else:
            left=np.searchsorted(labs,k);right=np.searchsorted(labs,k,side='right');ss.append(t[:,left:right].max(1))
    return np.stack(ss,1)

def priorityedge(maps,pose):
    # Reward four supported straight sides, not just one contrast edge across
    # the neighborhood. Ignore the four corners where damage is common.
    cx,cy,rx,ry,sl=pose.T;n=len(cx);t=np.linspace(.17,.83,12,dtype=np.float32);sides=[]
    for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        uu=a*t;vv=b*(1-t);vs=[]
        for field in [maps[3],maps[2]]:
            inside=warp(field,pose,.88*uu,.88*vv);outside=warp(field,pose,1.18*uu,1.18*vv)
            vs.append(np.clip(inside-outside,-.18,.65).mean(1))
        sides.append(.74*vs[0]+.26*vs[1])
    sides=np.sort(np.stack(sides,1),axis=1);return .60*sides[:,1]+.40*sides[:,2]

def chunk(x):
    maps,log=fields(x);p,poses=positions(maps);score=np.zeros((len(x),43),np.float32)
    gray=maps[3];inkgray=contrast(.60*log[:,0]+.34*log[:,1]+.06*log[:,2]);lgreen=log[:,1]-.5*(log[:,0]+log[:,2]);lred=log[:,0]-.6*log[:,1]-.4*log[:,2]
    for g,ids in enumerate(families):
        pose=poses[:,g]
        if g in (0,1,5):pose=register(maps,pose,g)
        if g in (4,5):
            detail=np.full((len(x),1),.63,dtype=np.float32)
            if g==4:detail[:,0]+=.72*priorityedge(maps,pose)
        else:detail=symbols(g,gray,pose)
        if g in (0,1):
            rr=symbols(g,inkgray,pose);detail[:,:8 if g==0 else len(ids)]=(.27*detail+.73*rr)[:,:8 if g==0 else len(ids)]
            if g==0:
                ocr=readnumbers(inkgray,pose);seg=segmented(inkgray,pose);detail[:,:8]=.45*detail[:,:8]+.31*ocr+.24*seg
        if g in (0,1,5):
            # Look inside the sign as well. Red and white should change places
            # at the rim, not in the field enclosed by the rim.
            rz=rectify(maps[0],pose)
            cz=rectify(gray,pose)
            if g==0:dist=1-np.sqrt(U*U+V*V)
            else:
                v=V if g==1 else -V;dist=np.minimum((1+v-2*np.abs(U))*.447,1-v)
            ringmask=(dist[0]>.04)&(dist[0]<(.17 if g==0 else .13));coremask=dist[0]>(.23 if g==0 else .18)
            diff=rz[:,ringmask].mean(1)-rz[:,coremask].mean(1)+cz[:,coremask].mean(1)-cz[:,ringmask].mean(1)
            p[:,g]=.75*p[:,g]+.30*diff
            if g==5:
                inside=cz[:,coremask];texture=np.percentile(inside,88,axis=1)-np.percentile(inside,12,axis=1)
                detail[:,0]=.78-.65*texture
            else:
                def region(z,m):return z[:,m[0]].mean(1)
                if g==0:
                    left=region(rz,(U>-.61)&(U<-.12)&(np.abs(V)<.18));right=region(rz,(U<.61)&(U>.12)&(np.abs(V)<.18))
                    vehicle=np.clip(left-right,-.15,.8)
                    detail[:,8:10]+=.31*vehicle[:,None];detail[:,11]-=.17*vehicle
                else:
                    green=rectify(lgreen,pose);redlog=rectify(lred,pose);mid=(np.abs(U)<.105)&(np.abs(V-.46)<.10);surround=(np.abs(U)>.15)&(np.abs(U)<.31)&(np.abs(V-.46)<.10)
                    signal=region(green,mid)-region(green,surround)
                    rzsignal=region(redlog,(np.abs(U)<.10)&(np.abs(V+.08)<.09))-region(redlog,(np.abs(U)>.14)&(np.abs(U)<.23)&(np.abs(V+.08)<.09))
                    # Require the full red-over-green color sequence; the white
                    # region around a dark symbol can take on a color cast.
                    traffic=np.minimum(signal-.058,rzsignal-.09)
                    detail[:,9]+=.80*np.clip(traffic,-.18,.42)
        score[:,ids]=p[:,g,None]+.86*detail
    return score

def smooth(z):
    a=np.pad(z,((0,0),(0,0),(1,1),(1,1)),mode='edge');m=(4*a[:,:,1:-1,1:-1]+a[:,:,:-2,1:-1]+a[:,:,2:,1:-1]+a[:,:,1:-1,:-2]+a[:,:,1:-1,2:])*.125
    return .20*z+.80*m

def classify(z):
    # Evaluate marginal matches at two optical resolutions. Make the control
    # decision for each photograph separately, not at the batch level.
    main=chunk(z);pair=np.sort(main,axis=1)[:,-2:];ambiguous=pair[:,1]-pair[:,0]<.18
    if np.any(ambiguous):
        ids=np.flatnonzero(ambiguous);aug=np.zeros_like(z);aug[:len(ids)]=smooth(z[ids]);secondary=chunk(aug)
        main[ids]=.64*main[ids]+.36*secondary[:len(ids)]
    return np.argmax(main,axis=1)

def predict(x):
    result=np.empty(len(x),np.int64)
    for start in range(0,len(x),32):
        # Constant workgroup size, including the final partial group. This also
        # keeps the reduction kernels independent of external batch sizes.
        n=min(32,len(x)-start); block=np.zeros((32,3,32,32),np.float32);block[:n]=x[start:start+n]
        result[start:start+n]=classify(block)[:n]
    return result
