import numpy as np

pose = CONSTANTS['pose']

def coords(n,ext):
    q=((np.arange(n,dtype=np.float32)+.5)*2/n-1)*ext
    a,b=np.meshgrid(q,q)
    return a.ravel(),b.ravel()

fineX,fineY=coords(25,1.24)
picX,picY=coords(30,1.)

def values(im):
    r,g,b=im
    # Log-cones reduce the exposure dependence of both colour and contrast.
    lr=np.log(r+.022); lg=np.log(g+.022); lb=np.log(b+.022)
    lum=.25*lr+.51*lg+.24*lb
    dr=lr-(.61*lg+.39*lb); db=lb-(.61*lr+.39*lg); dy=.52*lr+.48*lg-lb
    lev=np.percentile(lum,(12,88)); span=max(.30,lev[1]-lev[0]); center=(lev[0]+lev[1])*.5
    light=np.clip((lum-center)/span,-1.25,1.25)
    red=np.clip(dr/.56,-1.,1.7); blue=np.clip(db/.53,-1.,1.7); yellow=np.clip(dy/.55,-1.,1.7)
    return np.stack((red,blue,yellow,light,lum,lr,lg,lb)).astype(np.float32).reshape(8,-1)

def contrast(k,aa,bb,oo):
    # Common family evidence: region A (ink), B (paper), O (outside).
    ra,ba,ya,la=aa; rb,bbv,yb,lb=bb; ro,bo,yo,lo=oo
    if k in (0,1,2):
        return .47*(ra-.70*rb-.30*ro)+.13*np.clip(ra,-.65,1.35)+.62*(lb-la)+.13*(lb-lo)+.10*np.maximum(0,abs(la-lo)-.13)
    if k==3:
        return .49*(ya-.70*yb-.30*yo)+.20*np.clip(ya,-.65,1.35)+.51*(lb-la)+.28*(lb-lo)+.11*(la-lo)+.14*(ra-rb)
    if k==4:
        return .54*(ba-.70*bo-.30*bbv)+.29*np.clip(ba,-.65,1.35)+.25*np.abs(la-lo)+.21*(lb-la)
    if k==5:
        return .46*(ra-.50*rb-.50*ro)+.26*np.clip(ra,-.65,1.35)+.46*(lb-la)+.16*(lb-lo)+.10*np.abs(la-lo)
    if k==7:
        return .41*(ra-.68*rb-.32*ro)+.19*np.clip(ra,-.65,1.35)+.62*(lb-la)+.14*(lb-lo)+.12*abs(la-lo)
    return .96*(lb-la)+.27*(lb-lo)+.11*abs(lb-lo)-.23*np.maximum(0,np.maximum(ra,ba)-.17)

def pull(f,cx,cy,rx,ry,sx,sy,qx,qy):
    x=np.clip(cx[...,None]+rx[...,None]*(qx+sx[...,None]*qy),0,30.9999)
    y=np.clip(cy[...,None]+ry[...,None]*(qy+sy[...,None]*qx),0,30.9999)
    i=x.astype(np.int32); j=y.astype(np.int32); x=x-i; y=y-j; p=i+32*j
    x=x.astype(np.float32); y=y.astype(np.float32)
    flat=p.ravel(); shape=(len(f),)+p.shape
    u=f.take(flat,axis=1).reshape(shape); h=f.take(flat+1,axis=1).reshape(shape); v=f.take(flat+32,axis=1).reshape(shape); j=f.take(flat+33,axis=1).reshape(shape)
    u+=(h-u)*x; v+=(j-v)*x; return u+(v-u)*y

def prior(v):
    return -.035*((v[...,0]-15.5)**2+(v[...,1]-15.5)**2)/9-.026*(np.log(v[...,2]/v[...,3])/.30)**2-.032*(np.log(np.sqrt(v[...,2]*v[...,3])/11.9)/.27)**2

families=[[0,1,2,3,4,5,7,8,9,10,15,16],[11]+list(range(18,32)),[13],[12],list(range(33,41)),[14],[6,32,41,42],[17]]
mapgroup=(0,1,-1,-1,2,4,3,4)
coarseprior=prior(pose)

def refine(f,k,v):
    w=CONSTANTS['fine'+str(k)]; v=np.array([v[0],v[1],v[2],v[3],0,0],np.float32)
    def score(p):
        t=pull(f[:4],p[:,0],p[:,1],p[:,2],p[:,3],p[:,4],p[:,5],fineX,fineY)
        m=t@w.T
        return contrast(k,m[:,:,0],m[:,:,1],m[:,:,2])+prior(p)-.022*((p[:,4]/.20)**2+(p[:,5]/.20)**2)
    for rnd in range(2):
        step=1. if rnd==0 else .54
        for i,j,di,dj in [(0,1,1.35,1.35),(2,3,1.15,1.15),(4,5,.095,.080),(0,1,.68,.68)]:
            p=np.tile(v,(9,1)); p[:,i]+=gridA*di*step; p[:,j]+=gridB*dj*step
            vals=score(p); best=np.argmax(vals); v=p[best]; val=vals[best]
    return v,val

gridA,gridB=np.meshgrid(np.array([-1.,0.,1.],np.float32),np.array([-1.,0.,1.],np.float32));gridA=gridA.ravel();gridB=gridB.ravel()

interior=((picY<.73)&(abs(picX)<.5*(picY+.66))).reshape(30,30).astype(np.float32)

def gradients(z,interior):
    dx=np.zeros_like(z); dy=np.zeros_like(z); dx[:,:,1:-1]=z[:,:,2:]-z[:,:,:-2]; dy[:,1:-1]=z[:,2:]-z[:,:-2]
    planes=[]
    for u,v in ((1,0),(.7071,.7071),(0,1),(-.7071,.7071)):
        g=u*dx+v*dy
        for signed in (g,-g):
            cell=(np.maximum(0,signed)*interior).reshape(len(z),6,5,6,5).sum(axis=(2,4))
            planes.append(cell)
    h=np.stack(planes,axis=-1); energy=np.sqrt((h*h).sum(axis=-1,keepdims=True)); h=h/np.sqrt(energy+.16*energy.max(axis=(1,2),keepdims=True)+1e-8)
    h=h.reshape(len(z),-1); return h/(np.sqrt(np.sum(h*h,axis=1,keepdims=True))+1e-8)


def ocr(f,v):
    # Compare each candidate inscription against individual glyphs. A narrow
    # digit 1 is centred on its stem; the last zero provides a height anchor.
    gx=CONSTANTS['gx']; gy=CONSTANTS['gy']; cx,cy,rx,ry,sx,sy=v
    ink=(-.52*f[5]-.37*f[6]-.11*f[7])[None,:]; out=[]
    for narrow in (0,1):
        if narrow:
            origins=(-.405,0.,.405); text=((1,0,0),(1,2,0)); weights=(.31,.45,.24); w0=.221
        else:
            origins=(-.265,.265); text=((2,0),(3,0),(4,0),(5,0),(6,0),(7,0)); weights=(.65,.35); w0=.274
        ox,oy,span,hi=np.meshgrid(np.array([-.068,0,.068],np.float32),np.array([-.084,0,.084],np.float32),np.array([.82,1.,1.18],np.float32),np.array([.80,1.,1.20],np.float32),indexing='ij'); ox=ox.ravel(); oy=oy.ravel(); span=span.ravel(); hi=hi.ravel(); ps=[]
        for origin in origins:
            tx=ox+span*origin; ty=oy
            a=pull(ink,cx+rx*(tx+sx*ty),cy+ry*(ty+sy*tx),rx*w0*span,ry*.548*hi,sx*hi*.548/(w0*span),sy*span*w0/(.548*hi),gx,gy)[0]
            # Remove an illumination plane, normalize each glyph separately.
            a-=a.mean(axis=1,keepdims=True); a-=.65*(a@gx/(gx@gx))[:,None]*gx+.65*(a@gy/(gy@gy))[:,None]*gy
            a/=np.maximum(.048*np.sqrt(len(gx)),np.sqrt(np.sum(a*a,axis=1,keepdims=True))); bank=CONSTANTS['glyph'+str(narrow)]
            ps.append((a@bank.reshape(-1,len(gx)).T).reshape(len(a),8,-1).max(axis=2))
        for string in text:
            sc=sum(w*q[:,d] for w,q,d in zip(weights,ps,string)); out.append(np.max(sc-.012*((ox/.068)**2+(oy/.084)**2)))
    return np.array(out,np.float32)

def symbol(f,k,v):
    g=mapgroup[k]; ids=families[k]
    if g<0: return np.array([.16 if k==2 else .11],np.float32)
    r=CONSTANTS['roi'+str(g)]; qx=picX[r]; qy=picY[r]; cx,cy,rx,ry,sx,sy=v
    # Marginal registration error is most consequential for the tiny digits.
    shift=.070 if g==0 or g==1 else .085
    cx=cx+rx*shift*gridA; cy=cy+ry*shift*gridB
    if g==1:
        points=np.array([-.135,-.0675,0,.0675,.135],np.float32); ix,iy=np.meshgrid(points,points); ix=ix.ravel(); iy=iy.ravel()
        cx=v[0]+rx*ix; cy=v[1]+ry*iy; raw=pull(f,cx,cy,rx,ry,sx,sy,picX,picY); m=raw[:,:,r]; disp=.005*((ix/.07)**2+(iy/.07)**2)
    elif g==0:
        raw=pull(f,cx,cy,rx,ry,sx,sy,picX,picY); m=raw[:,:,r]; disp=.012*(gridA*gridA+gridB*gridB)
    else:
        m=pull(f,cx,cy,rx,ry,sx,sy,qx,qy); disp=.012*(gridA*gridA+gridB*gridB)
    # On light signs red-channel intensity separates black ink from its paper
    # without treating a misplaced red rim as a black stroke. Blue signs use
    # the red/green channels, in which white is distinct from blue paint.
    if k==4: z=.54*m[5]+.34*m[6]+.12*m[7]-.10*m[1]
    elif k in (5,7): z=.12*m[5]+.56*m[6]+.32*m[7]-.06*m[0]
    elif k==6: z=-m[4]
    else: z=-.62*m[5]-.27*m[6]-.11*m[7]
    # Winsorization and gentle background-slope cancellation, per query.
    qs=np.percentile(z,[3,16,75,97],axis=1).astype(np.float32); z=np.clip(z,qs[0,:,None]-.10,qs[3,:,None]+.06)
    cen=z.mean(axis=1,keepdims=True); z=z-cen
    a=qx-qx.mean(); b=qy-qy.mean(); z-=.72*(z@a/(a@a))[:,None]*a+.72*(z@b/(b@b))[:,None]*b
    std=np.sqrt(np.mean(z*z,axis=1)); amp=np.maximum(.045,qs[3]-qs[1]); nz=z/np.maximum(.0001,std[:,None]*np.sqrt(r.size))
    templates=CONSTANTS['template'+str(g)]; lab=CONSTANTS['label'+str(g)]; s=nz@templates.T
    # Signal reliability: the correlation of almost constant paper is noise.
    if g in (0,1):
        rawink=(-.62*raw[5]-.27*raw[6]-.11*raw[7]).reshape(-1,30,30); mask=(picX**2+picY**2<.70**2).reshape(30,30) if g==0 else interior
        gh=gradients(rawink,mask)@CONSTANTS['grad'+str(g)].T; bs=nz@CONSTANTS['blur'+str(g)].T; s=np.maximum(s,bs-.035)
        # Equal shapes at low resolution can carry different oriented edges.
        s=.68*s+.32*(1.45*gh-.49)
    s*=np.minimum(1,std/.078)[:,None]; s-=disp[:,None]
    if g==0:
        sub=z[:,CONSTANTS['left']]; sub=sub-sub.mean(axis=1,keepdims=True)
        details=CONSTANTS['numbers']; ns=len(details); ss=sub@details.T/np.maximum(.048*np.sqrt(sub.shape[1]),np.sqrt(np.sum(sub*sub,axis=1,keepdims=True)))
        s[:,:ns]=.57*s[:,:ns]+.43*(ss-.012*(gridA*gridA+gridB*gridB)[:,None])
        ci=CONSTANTS['caridx']; rc=m[0]@CONSTANTS['carred'].T
        s[:,ci]+=.16*np.clip(rc,-.70,1.15)
    if k==1:
        w=CONSTANTS['lamps']; rr=m[5]-m[6]; gg=m[6]-m[5]; yy=.5*(m[5]+m[6])-m[7]
        # Ordered red, amber and green lamps, distinct from an exclamation
        # mark even when both look like a short vertical stroke in grayscale.
        col=(rr@(w[:,0]-w[:,2]).T + .37*yy@(w[:,1]-w[:,2]).T)
        s[:,lab==26]+=.48*np.clip(col,-.65,1.3)-.11
    ret=[]
    for j in ids:
        if j==15:
            # Blank circular sign: use interior contrast relative to observed
            # contrast at the coloured border, and do not reward rim leakage.
            center=(qx*qx+qy*qy<.57**2); zz=(-.55*m[5]-.45*m[6])[:,center]
            pct=np.percentile(zz,[12,91],axis=1); delta=np.min(pct[1]-pct[0]); border=pull(f[4:5],v[0],v[1],v[2],v[3],v[4],v[5],fineX,fineY)@CONSTANTS['fine0'].T
            scale=max(.22,border[0,1]-border[0,0]); corr=.72-.63*np.minimum(1.5,delta/scale)
        else: corr=np.max(s[:,lab==j]);
        ret.append(corr)
    if k==0:
        ret=np.array(ret,np.float32); glyph=ocr(f,v); chosen=max(ret[:8]); identity=.70*ret[:8]+.30*glyph; ret[:8]=identity+chosen-max(identity)
    return np.array(ret,dtype=np.float32)

def classify(f,loc):
    base=np.max(loc,axis=1); best=np.argmax(loc,axis=1)
    # Always retain the strongest border hypotheses; a plausible pictogram
    # can disambiguate a weak or discoloured rim.
    order=np.argsort(-base); top=base[order[0]]; cs=np.full(43,-100.,np.float32)
    for rank,k in enumerate(order):
        if rank>=3 and base[k]<top-.55: break
        if rank>=5: break
        v,shape=refine(f,k,pose[best[k]])
        s=symbol(f,k,v)
        if k!=6:
            sector=CONSTANTS['sectors'+str(k)]; pix=pull(f[:4],v[0],v[1],v[2],v[3],v[4],v[5],fineX,fineY); side=(pix@sector.T).reshape(4,-1,3); side=contrast(k,side[:,:,0],side[:,:,1],side[:,:,2]); shape=.72*shape+.28*(np.sort(side)[:max(2,len(side)//2)].mean()+prior(v))
        if k==7:
            # The bar on a no-entry disc has red paint on BOTH sides. A
            # triangular rim or a sky/background horizon is only one edge.
            upper=(abs(fineX)<.55)&(fineY<-.28)&(fineY>-.75); bot=(abs(fineX)<.55)&(fineY>.28)&(fineY<.75); bar=(abs(fineX)<.55)&(abs(fineY)<.14)
            up=pix[:,upper].mean(axis=1); low=pix[:,bot].mean(axis=1); mid=pix[:,bar].mean(axis=1); s-=.58*min(.85,abs(up[0]-low[0])*.39+abs(up[3]-low[3])*.74)
            h=min(mid[3]-up[3]+.65*(up[0]-mid[0]),mid[3]-low[3]+.65*(low[0]-mid[0])); s-=.42*max(0,.35-h)
        if k in (2,3):
            m=pull(f,v[0],v[1],v[2],v[3],v[4],v[5],fineX,fineY); roi=(abs(fineX)+abs(fineY)<.59) if k==3 else ((-fineY+.70>2*abs(fineX))&(fineY>-.66)); paper=m[4,roi]; fill=np.percentile(paper,85)-np.percentile(paper,15)
            # No pictogram is expected here: reward the homogeneous paper
            # relative to boundary contrast rather than an arbitrary glyph.
            gap=m[4]@CONSTANTS['fine'+str(k)].T
            s+=.53-.55*min(1.4,fill/max(.35,abs(gap[0]-gap[1])+.15))
        # Correlation is evaluated inside a mask independent of photo content.
        bias=(0.,.00,.02,-.035,.01,-.11,-.07,-.07)[k]
        cs[families[k]]=shape*.80+s*.95+bias
    return np.argmax(cs)

# Matrix of geometric masks for the coarse pose grid.
local=np.concatenate([CONSTANTS['loc'+str(k)].reshape(-1,1024) for k in range(8)])

def predict(x):
    ans=np.empty(len(x),np.int64)
    for start in range(0,len(x),8):
        fs=np.stack([values(im) for im in x[start:start+8]])
        # Fixed contraction shape regardless of batch boundary.
        tmp=np.zeros((8,4,1024),np.float32); tmp[:len(fs)]=fs[:,:4]
        s=(tmp.reshape(32,1024)@local.T).reshape(8,4,8,len(pose),3)
        for i,f in enumerate(fs):
            l=[]
            for k in range(8):
                m=s[i,:,k]; l.append(contrast(k,m[:,:,0],m[:,:,1],m[:,:,2])+coarseprior)
            ans[start+i]=classify(f,np.stack(l))
    return ans
