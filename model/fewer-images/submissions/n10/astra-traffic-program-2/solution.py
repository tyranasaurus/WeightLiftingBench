import numpy as np

families=[[0,1,2,3,4,5,7,8,9,10,15,16],[11]+list(range(18,32)),[12],[13],[14],[17],[6,32,41,42],list(range(33,41))]

def product(a,b):
    # Fixed row tiles even when the caller changes the batching.
    a=np.ascontiguousarray(a,dtype=np.float32); n=len(a);k=32
    padded=np.zeros(((n+k-1)//k*k,a.shape[1]),np.float32);padded[:n]=a
    return (padded.reshape(-1,k,a.shape[1])@b.T).reshape(-1,b.shape[0])[:n]

def smooth(im):
    a=np.pad(im,((0,0),(1,1),(1,1)),mode='edge'); m=a[:,1:-1]; b=m*.68+(a[:,:-2]+a[:,2:])*.16
    return b[:,:,1:-1]*.68+(b[:,:,:-2]+b[:,:,2:])*.16

def center(im):
    m=im[:,5:27,5:27].reshape(len(im),-1);q=np.quantile(m,[.08,.50,.92],axis=1).astype(np.float32)[:,:,None,None]
    return np.clip((im-q[1])/(q[2]-q[0]+.032),-1.6,1.6)

def features(x):
    r,g,b=x[:,0],x[:,1],x[:,2]; a=.28*r+.50*g+.22*b
    # Square root keeps detail in the shadows without greatly amplifying black noise.
    lum=np.sqrt(a+.008); lum=center(lum)
    v=.012+.36*r+.42*g+.22*b; red=np.clip((r-.64*g-.36*b)/v,-1.3,1.55);blue=np.clip((b-.62*r-.38*g)/v,-1.45,1.55);yellow=np.clip((.48*r+.52*g-b)/v,-1.4,1.55)
    return lum,red,blue,yellow

def rimscore(lum):
    ap=np.pad(smooth(lum),((0,0),(1,1),(1,1)),mode='edge');gx=ap[:,1:-1,2:]-ap[:,1:-1,:-2];gy=ap[:,2:,1:-1]-ap[:,:-2,1:-1]
    mats=CONSTANTS['discRim'];s=0
    for h,v in enumerate([gx,(gx+gy)*.707,gy,(gy-gx)*.707]):
        s+=product(np.minimum(abs(v),1.).reshape(len(lum),-1),mats[:,h])
    return s

def coarse(lum,red,blue,yellow):
    m=len(lum);n=CONSTANTS['shapes'].shape[0]//8; pos=[]; scores=[]
    for g in range(8):
        f=CONSTANTS['shapes'][g*n:(g+1)*n]; col=blue if g==7 else yellow if g==2 else red
        a=product(col.reshape(m,-1),f[:,0]); l=product(lum.reshape(m,-1),f[:,1]);
        if g in (0,1,3):s=.81*a+.71*l
        elif g==2:s=1.15*a+.41*l
        elif g==6:s=.82*l-.15*abs(a)+.94*rimscore(lum)
        else:s=(1.14 if g==7 else .99)*a+(.30 if g==7 else .49)*l
        p=CONSTANTS['pose'][g*n:(g+1)*n]; s=s-.015*((p[:,0]-15.5)**2+(p[:,1]-15.5)**2)/9-.032*((p[:,2]-11.9)/5)**2-.025*((p[:,3]-p[:,2])/4)**2
        ind=s.argmax(1);scores.append(s[np.arange(m),ind]);pos.append(p[ind]);
    return np.stack(scores,axis=1),np.stack(pos,axis=1)

def interp(im,x,y):
    n=len(im); x=np.clip(x,0,30.9999);y=np.clip(y,0,30.9999);a=x.astype(np.int32);b=y.astype(np.int32);s=x-a;t=y-b
    s=s.astype(np.float32);t=t.astype(np.float32);i=b*32+a+np.arange(n,dtype=np.int32).reshape((n,)+(1,)*(a.ndim-1))*1024;v=im.reshape(-1)
    return (1-t)*((1-s)*v[i]+s*v[i+1])+t*((1-s)*v[i+32]+s*v[i+33])

def positions(p,x,y):
    return p[:,:,0,None]+p[:,:,2,None]*(x+p[:,:,4,None]*y),p[:,:,1,None]+p[:,:,3,None]*(y+p[:,:,5,None]*x)

def posescore(p,img,lum,g):
    e=CONSTANTS['edge'+str(g)]; a,b=e[:2];nx,ny=e[2:]; v=.19 if g==0 else .152 if g in (1,3) else .18
    def val(d,im):
        x,y=positions(p,a-d*nx,b-d*ny);return interp(im,x,y)
    if g in (0,1,3,2):
        ring=val(v*.53,img); inner=val(v+.095,img);out=val(-.115,img)
        diff=ring-inner;outer=ring-out
        s=np.mean(np.clip(diff,-.44,1.45)+.26*np.clip(outer,-.52,1.15),axis=-1)
    else:
        inner=val(.105,img);out=val(-.095,img); outerl=val(-.078,lum);innerl=val(.084,lum)
        s=np.mean(.85*np.clip(inner-out,-.66,1.35)+.52*np.minimum(abs(innerl-outerl),.95),axis=-1)
        if g==6:
            t=np.linspace(-.64,.64,21,dtype=np.float32);vx=-.64*t;vy=.77*t
            def across(w):
                j,i=positions(p,vx+.77*w,vy+.64*w);return interp(lum,j,i)
            s+=.72*np.mean(np.clip((across(.29)+across(-.29))*.5-across(0),-.42,1.25),axis=-1)+.60*np.mean(np.minimum(abs(innerl-outerl),.95),axis=-1)
    prior=.014*((p[:,:,0]-15.5)**2+(p[:,:,1]-15.5)**2)/7+.025*((p[:,:,2]-p[:,:,3])/4)**2+.020*((p[:,:,2]+p[:,:,3]-24)/10)**2+.11*(abs(p[:,:,4])+abs(p[:,:,5]))
    return s-prior

def refine(p,col,l,g):
    if g in (0,1,3):sig=.76*col-.72*l
    elif g==2:sig=-col+.42*l
    elif g==6:sig=l*.22
    else:sig=col
    cand=np.array([[dx,dy,ax,ay,tx,ty] for dx,dy,ax,ay,tx,ty in [(0,0,0,0,0,0),(-1,0,0,0,0,0),(1,0,0,0,0,0),(0,-1,0,0,0,0),(0,1,0,0,0,0),(0,0,-1,0,0,0),(0,0,1,0,0,0),(0,0,0,-1,0,0),(0,0,0,1,0,0),(0,0,-1,-1,0,0),(0,0,1,1,0,0),(0,0,0,0,.105,-.105),(0,0,0,0,-.105,.105),(0,0,0,0,.115,0),(0,0,0,0,-.115,0)]],np.float32)
    start=p;p=p.copy();ii=np.arange(len(p));
    for step in (1.,.80,.55,.40):
        pp=p[:,None,:]+step*cand[None];s=posescore(pp,sig,l,g)
        s-=.024*(((pp[:,:,:4]-start[:,None,:4])**2).sum(axis=2)/6)
        idx=s.argmax(1);p=pp[ii,idx]
    return p

def normal(v,dust=0.):return v/(np.sqrt(np.sum(v*v,axis=-1,keepdims=True)+dust)+1e-6)

def detrend(v,x,y):
    v=v-v.mean(-1,keepdims=True);x=x-x.mean();y=y-y.mean()
    return v-(v*x).sum(-1,keepdims=True)*x/((x*x).sum()+1e-7)-(v*y).sum(-1,keepdims=True)*y/((y*y).sum()+1e-7)

def fullscore(p,lum,col,g):
    k=str(g); x,y=CONSTANTS['whole'+k]; pp=p[:,None];j,i=positions(pp,x,y);v=interp(lum,j,i)[:,0];h=interp(col,j,i)[:,0]
    z=product(normal(detrend(v,x,y),len(x)*.025**2),CONSTANTS['wf'+k]); f=product(normal(detrend(h,x,y),len(x)*.095**2),CONSTANTS['wc'+k]); ids=CONSTANTS['wi'+k]
    z=.66*z+(.22 if g!=6 else 0)*f
    if g==2:
        # The outer diamond is white, its middle yellow. They can have nearly the same brightness.
        z=.38*product(normal(detrend(v,x,y),len(x)*.045**2),CONSTANTS['wf'+k])+.53*f
    return np.stack([np.max(z[:,ids==c],1) for c in families[g]],axis=1)

def symbols(p,lum,clr,g):
    key=str(g); x,y=CONSTANTS['points'+key];n=len(p);m=len(x)
    dx=np.array([0,-1,1,0,0,-1,1,-1,1],np.float32);dy=np.array([0,0,0,-1,1,-1,-1,1,1],np.float32)
    pp=np.broadcast_to(p[:,None,:],(n,len(dx),6)).copy();pp[:,:,0]+=.073*p[:,2,None]*dx;pp[:,:,1]+=.073*p[:,3,None]*dy
    u,v=positions(pp,x,y); z=interp(lum,u,v); zn=detrend(z,x,y);rms=np.sqrt(np.mean(zn*zn,axis=-1));main=normal(zn,m*.022**2)
    # Orientations measured in sign coordinates; a one-pixel-scale symmetric derivative.
    ax=.06*p[:,2,None,None];ay=.06*p[:,3,None,None];gx=interp(lum,u+ax,v)-interp(lum,u-ax,v);gy=interp(lum,u,v+ay)-interp(lum,u,v-ay)
    scale=(gx*gx+gy*gy+.0012)**.25;vec=normal(np.concatenate([gx/scale,gy/scale],axis=-1),m*.012**2)
    ids=CONSTANTS['cid'+key];f=product(main.reshape(-1,m),CONSTANTS['gray'+key]); f=.74*f+.26*product(vec.reshape(-1,2*m),CONSTANTS['grad'+key]);f=f.reshape(n,len(dx),-1)
    if g==1:
        agg=0
        for h in range(4):
            zv=product(main.reshape(-1,m),CONSTANTS['strips'][h]);zv-=zv.mean(1,keepdims=True)
            agg+=product(normal(zv,.012**2),CONSTANTS['stripTemplate'+str(h+1)])
        f=.80*f+.20*agg.reshape(n,len(dx),-1)/4
    if g in (0,6):
        patches=[]
        for h in range(3 if g==0 else 1):
            keep=CONSTANTS['select'+key+str(h)];px=x[keep];py=y[keep];px=px-px.mean();py=py-py.mean();py=py-(px*py).sum()*px/(px*px).sum();vm=detrend(z[:,:,keep],px,py); count=len(px)
            if g==6:unused=np.sqrt(np.mean(vm*vm,axis=-1))
            patches.append(product(normal(vm,count*.020**2).reshape(-1,count),CONSTANTS['patch'+key+str(h)]).reshape(n,len(dx),-1))
        if g==0:
            sel=(ids<9);f[:,:,sel]=.64*f[:,:,sel]+.19*patches[0][:,:,sel]+.085*patches[1][:,:,sel]+.085*patches[2][:,:,sel]
        else:
            sel=(ids!=32);f[:,:,sel]=.52*f[:,:,sel]+.48*patches[0][:,:,sel];f[:,:,ids==32]=.52*f[:,:,ids==32]+.48*np.exp(-unused[:,:,None]/.12)
    if g in (0,1):
        cm=detrend(interp(clr,u,v),x,y); cw=product(normal(cm,m*.14**2).reshape(-1,m),CONSTANTS['col'+key]).reshape(n,len(dx),-1)
        f+=.13*np.maximum(cw,0)
    f-=CONSTANTS['pen'+key][None,None,:]+.011*(abs(dx)+abs(dy))[None,:,None]; scores=[]
    for c in families[g]:
        if c!=15:score=np.max(f[:,:,ids==c],axis=(1,2))
        else:
            # No vehicles: no symbol against the otherwise identical enamel.
            # Estimate the local noise floor from the unprinted top and bottom
            # of the pale centre. This avoids using a global exposure cutoff.
            wing=zn[:,0,abs(y)>.455]; noise=np.median(abs(wing-np.median(wing,axis=1,keepdims=True)),axis=1)*1.4826
            observed=np.quantile(zn[:,0],.85,axis=1)-np.quantile(zn[:,0],.055,axis=1)
            score=.82*np.exp(-np.maximum(0,observed-noise*.6)/.21)
            score-=.46*np.maximum(0,np.max(f[:,:,ids!=15],axis=(1,2))-.30)
        scores.append(score)
    return np.stack(scores,axis=1)

def classify(im):
    lum,red,blue,yellow=features(im);s,ps=coarse(lum,red,blue,yellow);n=len(im);result=np.full((n,43),-20,np.float32)
    cutoff=np.sort(s,axis=1)[:,-4]; sl=smooth(lum);ink=smooth(center(np.sqrt(.70*im[:,0]+.20*im[:,1]+.10*im[:,2]+.006)))
    for g in range(8):
        which=np.nonzero(s[:,g]>=cutoff)[0]
        if len(which)==0:continue
        col=(blue if g==7 else yellow if g==2 else red)[which];l=sl[which];p=refine(ps[which,g],col,l,g)
        # Compare the aligned outline against its immediate neighbourhood.
        # A locally plausible but backwards/upside-down boundary should not
        # win only because it encloses an untextured patch of road or sky.
        dx=p[:,0]-15.5;dy=p[:,1]-15.5
        linefeat=.76*col-.72*l if g in (0,1,3) else (-col+.42*l if g==2 else l*.22 if g==6 else col)
        verified=posescore(p[:,None],linefeat,l,g)[:,0]
        w=fullscore(p,l,col,g);t=s[which,g,None]+.28*verified[:,None]
        if len(families[g])>1:
            sym=(ink[which]*.62+l*.38) if g in (0,1,6) else l
            sy=symbols(p,sym,col,g);total=.69*t+.83*w+.87*sy
            if g in (0,1,6,7):
                # Foreshortening, blur and changes in icon layout need not
                # affect ink and frame in the same way. Rectify again at
                # nearby poses and retain the best joint agreement.
                offset=np.array([[0,-.08,.90,.92],[0,.08,1.10,1.08],[-.065,0,1,1],[.065,0,1,1],[0,0,.82,1.11],[0,0,1.14,.87]],np.float32)
                for off in offset:
                    alt=p.copy();alt[:,0]+=p[:,2]*off[0];alt[:,1]+=p[:,3]*off[1];alt[:,2]*=off[2];alt[:,3]*=off[3]
                    local=symbols(alt,sym,col,g)
                    # Do not move the no-ink test onto an arbitrary blank patch.
                    if g==0:local[:,families[g].index(15)]=sy[:,families[g].index(15)]
                    localw=fullscore(alt,l,col,g) if g==6 else w
                    total=np.maximum(total,.69*t+.83*localw+.87*local-.037)
        else:total=.69*t+1.55*w+.19
        if g in (1,3):
            # A triangle has three sides, not just one strong horizontal bar.
            e=CONSTANTS['edge'+str(g)]; pp=p[:,None]
            def sidelevel(d):
                j,i=positions(pp,e[0]-d*e[2],e[1]-d*e[3]);return interp(linefeat,j,i)[:,0]
            side=(sidelevel(.081)-sidelevel(.239)).reshape(len(p),3,-1).mean(2)
            total+=.36*(np.min(side,axis=1)-side.mean(1))[:,None]
        if g in (1,3):
            # An upright and an inverted warning triangle cannot both explain
            # the same coloured contour. Verify vertex/flat-base orientation
            # by comparing the correct boundary to its vertical reflection.
            opposite=3 if g==1 else 1
            anti=posescore(p[:,None],linefeat,l,opposite)[:,0]
            total+=.28*np.clip(verified-anti,-.65,.52)[:,None]
        if g==3:
            # The wide red side of a yield sign is ABOVE its point. Test the
            # red cap against the lower centre, in addition to the border.
            xx=np.linspace(-.44,.44,17,dtype=np.float32)
            j,i=positions(p[:,None],xx,-.919+xx*0);redtop=interp(col,j,i).mean(axis=(1,2))
            j,i=positions(p[:,None],xx*.48,.16+xx*0);inner=interp(col,j,i).mean(axis=(1,2))
            total+=.26*np.clip(redtop-inner-.25,-.75,.22)[:,None]
        if g==6:
            # Cancelled bans retain both the white disc and the central stripe.
            # Normalise partial-ink agreement independently from those two.
            total=np.maximum(total,.57*t+1.10*w+.70*sy)
        result[which[:,None],np.array(families[g])[None,:]]=total
    return result.argmax(1)

def predict(x):
    output=np.empty(len(x),np.int64)
    for j in range(0,len(x),32):output[j:j+32]=classify(x[j:j+32])
    return output
