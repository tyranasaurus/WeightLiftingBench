import numpy as np

# Convert the small template banks once. None of the following state changes between calls.
kernels = [[CONSTANTS['k'+str(f)+str(s)].astype(np.float32) for s in range(2)] for f in range(8)]
poses = [[CONSTANTS['p'+str(f)+str(s)] for s in range(2)] for f in range(8)]
families = [[0,1,2,3,4,5,7,8,9,10,16],[11]+list(range(18,32)),list(range(33,41)),[13],[12],[14],[17],[6,32,41,42]]

def central(a):return a[:,6:26,6:26].reshape(len(a),400)
def percentile(a,q):return np.percentile(a,q,axis=1).astype(np.float32)[:,None,None]
def cropfit(v,a,b):return np.minimum(np.maximum(v,a),b)
def smooth(a):
    b=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    return (b[:,:-2,:-2]+b[:,:-2,2:]+b[:,2:,:-2]+b[:,2:,2:]+2*(b[:,1:-1,:-2]+b[:,1:-1,2:]+b[:,:-2,1:-1]+b[:,2:,1:-1])+4*b[:,1:-1,1:-1])/16

def hog(z):
    g=smooth(z);a=np.pad(g,((0,0),(1,1),(1,1)),mode='edge');dx=a[:,1:-1,2:]-a[:,1:-1,:-2];dy=a[:,2:,1:-1]-a[:,:-2,1:-1];mag=np.sqrt(dx*dx+dy*dy)
    feat=[]
    for w in (dx,dy,.7071*(dx+dy),.7071*(dx-dy)):
        for sign in (1,-1):
            response=np.maximum(sign*w-.58*mag,0);response=smooth(smooth(response));feat.append(response.reshape(-1,5,6,5,6).mean((2,4)))
    v=np.stack(feat,axis=3);ss=(v*v).sum(3,keepdims=True);v=v/(ss+.45*ss.mean((1,2),keepdims=True)+.000002)**.30;v=v.reshape(-1,200)
    return v/(np.sqrt((v*v).sum(1,keepdims=True))+.00001)

def features(batch):
    r,g,b=[batch[:,k] for k in range(3)]
    brightness=.29*r+.53*g+.18*b
    floor=.006+.29*percentile(central(brightness),55)
    up=percentile(central(brightness),85)+.012
    def tone(a):
        raw=np.log1p(a/floor);res=[]
        for v in (raw,a):
            q0=percentile(central(v),8);q1=percentile(central(v),89)
            res.append(cropfit((v-q0)/(q1-q0+.014),-.38,1.42))
        return .60*res[0]+.40*res[1]
    l=tone(brightness);gg=tone(g);rr=tone(r)
    offs=.006+.033*up
    def chrom(a,bb):return np.log((a+offs)/(bb+offs))
    def paint(v):
        base=percentile(central(v),30)
        relative=v-cropfit(base,-.38,.55)
        return cropfit(.71*relative+.29*v,-.48,1.45)
    red=paint(np.minimum(chrom(r,g),chrom(r,b)+.15))
    blue=paint(np.minimum(chrom(b,r),1.5*chrom(b,g)+.20))
    yellow=paint(np.minimum(chrom(r,b),chrom(g,b)+.17))
    saturate=np.maximum.reduce([red,blue,yellow,np.zeros_like(red)])
    edge=0
    for a,weight in ((l,.64),(red,.20),(blue,.16)):
        z=np.pad(a,((0,0),(1,1),(1,1)),mode='edge');dx=z[:,1:-1,2:]-z[:,1:-1,:-2];dy=z[:,2:,1:-1]-z[:,:-2,1:-1]
        edge=edge+weight*np.minimum(1.6,np.sqrt(dx*dx+dy*dy))
    return [red,blue,yellow,saturate,l,gg,rr,edge,r,g,b]

def groupmaps(m,f):
    red,blue,yellow,sat,l,gg,rr,edge=m[:8]
    if f==2:return [blue,.64*rr+.36*l,edge,red]
    if f==4:return [yellow,l,edge,blue]
    if f==7:return [sat,l,edge,sat]
    return [red,.71*gg+.29*l,edge,blue]

# Hold the BLAS row dimension fixed, even at a batch boundary. This keeps the
# same arithmetic path for a given example in different calling contexts.
def product(a,b):
    nr=a.shape[0];n=((nr+31)//32)*32
    if nr==n:return np.ascontiguousarray(a)@b
    t=np.zeros((n,a.shape[1]),np.float32);t[:nr]=a
    return (t@b)[:nr]

def location(m,f,stage):
    k=kernels[f][stage];p=poses[f][stage]
    def dot(j):return product(m[j].reshape(len(m[j]),1024),k[j])
    color=dot(0) if f!=7 else 0
    lig=dot(1);edge=dot(2)
    # Evidence for ink/paint, for the pigment boundary, and for the sign outline.
    if f in (0,1,3):v=.75*color+.64*lig+.39*edge
    elif f==2:v=.88*color+.62*lig+.54*edge+.02
    elif f==4:v=.83*color+.67*lig+.52*edge
    elif f==5:v=.85*color+.71*lig+.50*edge-.025
    elif f==6:v=.83*color+.70*lig+.50*edge-.025
    else:v=.95*lig+.47*edge-.41*dot(3)+.045
    if stage==0:
        v=v-.002*((p[:,0]-15.5)**2+(p[:,1]-15.5)**2)-.008*np.abs(p[:,2]-12.5)
    else:
        v=v-.014*(abs(p[:,4])+abs(p[:,5]))-.003*(abs(p[:,0]-15.5)+abs(p[:,1]-15.5))
    idx=v.argmax(1);return v[np.arange(len(idx)),idx],p[idx]

def coordinates(p,x,y,s=1):
    x=x*s;y=y*s
    return (p[:,0,None]+p[:,2,None]*(x+p[:,4,None]*y),p[:,1,None]+p[:,3,None]*(y+p[:,5,None]*x))

def sampler(x,y):
    x=cropfit(x,0,30.99999);y=cropfit(y,0,30.99999);i=x.astype(np.int32);j=y.astype(np.int32);fx=(x-i).astype(np.float32);fy=(y-j).astype(np.float32)
    return i+j*32+np.arange(len(i),dtype=np.int32)[:,None]*1024,fx,fy

def sample(a,where):
    ij,fx,fy=where;v=a.reshape(-1);return (v[ij]*(1-fx)+v[ij+1]*fx)*(1-fy)+(v[ij+32]*(1-fx)+v[ij+33]*fx)*fy

y32,x32=np.mgrid[:32,:32].astype(np.float32);x32=(x32.ravel()-15.5)/12;y32=(y32.ravel()-15.5)/12
yg,xg=np.mgrid[:28,:28].astype(np.float32);xg=(xg.ravel()-13.5)/14;yg=(yg.ravel()-13.5)/14

def refine(m,f,p):
    w=sampler(*coordinates(p,x32,y32));mm=[sample(v,w).reshape(len(p),32,32) for v in m]
    val,q=location(mm,f,1)
    # Compose coarse diagonal transform with the interior affine transform.
    pose=q.copy();pose[:,0]=p[:,0]+(q[:,0]-15.5)*p[:,2]/12;pose[:,1]=p[:,1]+(q[:,1]-15.5)*p[:,3]/12
    pose[:,2:4]=q[:,2:4]*p[:,2:4]/12
    return val,pose

# Small residual displacements account for variations in centring of the printed symbol.
shifts=[(a,b,t) for a in (-.057,0,.057) for b in (-.057,0,.057) for t in (0,)]
shifts+= [(0,0,-.095),(0,0,.095),(-.095,0,0),(.095,0,0),(0,.103,0),(0,-.103,0)]
shiftx=np.concatenate([xg-t*yg+a for a,b,t in shifts]);shifty=np.concatenate([yg+t*xg+b for a,b,t in shifts]);ns=len(shifts)

# The second alignment does not assume a particular printed symbol size. Bound
# the foreground by the marginal distribution of contrast within the sign face.
y30,x30=np.mgrid[:30,:30].astype(np.float32);x30=(x30.ravel()-14.5)/15;y30=(y30.ravel()-14.5)/15
fine=[(u,v,0,0) for u,v in ((1.,1.),(.87,.87),(1.14,1.14),(.88,1.10),(1.10,.88))]+[(1,1,a,b) for a,b in ((.12,0),(-.12,0),(0,-.12),(0,.12))]
finex=np.concatenate([x30*u+a for u,v,a,b in fine]);finey=np.concatenate([y30*v+b for u,v,a,b in fine]);nf=len(fine)

def extent(mass):
    count=len(mass);cs=np.cumsum(mass,axis=1);total=cs[:,-1:]
    def quantile(q):
        j=np.argmax(cs>=q*total,axis=1);massj=mass[np.arange(count),j];rel=(q*total[:,0]-cs[np.arange(count),j])/np.maximum(massj,.00001)+1
        return (j.astype(np.float32)+cropfit(rel,0,1))*2/28-1
    return quantile(.025),quantile(.975)

def letters(a,p,ink,whole):
    # Parse a two- or three-character word; optimize cuts only within the likely
    # inter-character gaps. The last zero provides a consistency check.
    cnt=len(p);row=np.arange(cnt);left,right,top,bot=whole;wid=right-left
    grid=xg.reshape(28,28)[0];ink=ink.reshape(cnt,28,28);hist=ink.sum(1)
    def gap(frac):
        center=left+wid*frac;dist=abs(grid[None,:]-center[:,None])/(wid[:,None]+.02)
        local=hist/(.10+hist.max(1,keepdims=True))+.85*dist+10*(dist>.125)
        return grid[local.argmin(1)]
    def letter(lo,hi):
        region=(grid[None,:]>=lo[:,None])&(grid[None,:]<hi[:,None]);mass=ink*region[:,None,:];l,r=extent(mass.sum(1));t,b=extent(mass.sum(2));cx=.5*(l+r);cy=.22*(t+b)+.28*(bot+top)
        rx=.63*cropfit(r-l,.075,.64);ry=.63*cropfit(.62*(b-t)+.38*(bot-top),.26,1.20)
        w=sampler(*coordinates(p,cx[:,None]+rx[:,None]*finex,cy[:,None]+ry[:,None]*finey));im=sample(a,w).reshape(cnt*nf,30,30)
        def unit(z):
            z=z.reshape(-1,900);z=z-z.mean(1,keepdims=True);return z/np.sqrt((z*z).sum(1,keepdims=True)+.045)
        g=smooth(smooth(im));slow=smooth(smooth(smooth(g)));db=CONSTANTS['digits'];ids=CONSTANTS['didx']
        v=.52*product(unit(im),db[0])+.34*product(unit(im-slow),db[1])+.14*product(unit(g),db[2]);oh=product(hog(im),CONSTANTS['dh']);v=(.75*v+.25*(1.6*oh-.60)).reshape(cnt,nf,len(ids)).max(1)
        return dict((k,v[:,ids==k].max(1)) for k in (0,1,2,3,5,6,7,8))
    mid=gap(.50);ll=letter(left-.06,mid);rr=letter(mid,right+.06)
    double=[.65*ll[n]+.35*rr[0] for n in (2,3,5,6,7,8)]
    i=gap(.265);j=gap(.638);la=letter(left-.06,i);lb=letter(i,j);lc=letter(j,right+.06)
    triple=[.27*la[1]+.47*lb[n]+.26*lc[0] for n in (0,2)]
    return np.stack(double+triple,axis=1)

def foreground(a,m,f,p,mask):
    cnt=len(p);w=sampler(*coordinates(p,xg,yg));z=sample(a,w);rd=sample(m[0],w);bl=sample(m[1],w)
    if f in (0,1): ink=z-.26*np.maximum(rd,0)
    else:ink=z
    core=(mask.ravel()>.80);m0=ink[:,core];bg=np.percentile(m0,20,axis=1).astype(np.float32)[:,None];fg=np.percentile(m0,94,axis=1).astype(np.float32)[:,None]
    ink=cropfit((ink-bg)/(.015+fg-bg),0,1.5)
    if f in (0,1):
        # More conservative frame exclusion for segmentation than for matching.
        safe=(mask>.91)
        if f==0:safe=safe&(xg*xg+yg*yg<.725**2)
        else:safe=safe&(yg<.80)&(yg>2*np.abs(xg)-.58)
    else:safe=mask>.85
    bank=CONSTANTS['t'+str(f)];ids=CONSTANTS['j'+str(f)];best=None;ocr=None
    for th in (.28,.52):
        mass=(cropfit((ink-th)*2.7,0,1)*safe).reshape(cnt,28,28);left,right=extent(mass.sum(1));top,bot=extent(mass.sum(2))
        if f==0 and ocr is None:ocr=letters(a,p,mass,(left,right,top,bot))
        cx=(left+right)*.5;cy=(top+bot)*.5;rx=.63*cropfit(right-left,.12,1.72);ry=.63*cropfit(bot-top,.17,1.72)
        sx=cx[:,None]+rx[:,None]*finex;sy=cy[:,None]+ry[:,None]*finey
        w2=sampler(*coordinates(p,sx,sy));image=sample(a,w2).reshape(cnt*nf,30,30)
        def unit(z):
            z=z.reshape(-1,900);z=z-z.mean(1,keepdims=True);return z/np.sqrt((z*z).sum(1,keepdims=True)+.035)
        g=smooth(smooth(image));slow=smooth(smooth(smooth(g)));v=.51*product(unit(image),bank[0])+.28*product(unit(image-slow),bank[1])+.21*product(unit(g),bank[2]);oh=product(hog(image),CONSTANTS['h'+str(f)]);v=.67*v+.33*(1.6*oh-.60)
        if f==1:
            # Symmetric soft overlap: account for ink not explained by the sketch
            # as well as sketch strokes missing from the photograph.
            lo=np.repeat(bg,nf,axis=0);hi=np.repeat(fg,nf,axis=0);inked=cropfit(((image.reshape(-1,900)-lo)/(.025+hi-lo)-.25)*1.55,0,1).reshape(-1,30,30);pi=inked.reshape(-1,900);prec=product(pi/(pi.sum(1,keepdims=True)+.01),CONSTANTS['distance'])
            wide=np.pad(inked,((0,0),(2,2),(2,2)),mode='edge');near=np.maximum.reduce([wide[:,i:i+30,j:j+30] for i,j in ((0,2),(2,0),(4,2),(2,4),(1,1),(3,3),(3,1),(1,3),(2,2))]);rec=product((.25*inked+.75*near).reshape(-1,900),CONSTANTS['stencil']);chamfer=2*prec*rec/(prec+rec+.00001)
            v=.75*v+.25*(1.9*chamfer-.90)
        v=v.reshape(cnt,nf,len(ids)).max(1)
        best=v if best is None else np.maximum(best,v)
    return np.stack([best[:,ids==c].max(1) for c in families[f]],axis=1),ocr

def localtone(a,idx):
    h=a[:,idx];lo=np.percentile(h,7,axis=1).astype(np.float32)[:,None];hi=np.percentile(h,86,axis=1).astype(np.float32)[:,None];med=np.percentile(h,50,axis=1).astype(np.float32)[:,None]
    # Differences within a dark sign should not be compressed by a bright sky.
    lin=cropfit((a-lo)/(hi-lo+.006),-.28,1.30);floor=.004+.32*med
    l0=np.log(lo+floor);log=cropfit((np.log(a+floor)-l0)/(np.log(hi+floor)-l0+.028),-.28,1.30)
    return .58*log+.42*lin

def canvas(m,f,p):
    w=sampler(*coordinates(p,x32,y32));raw=[sample(z,w) for z in m[8:11]]
    mask=CONSTANTS['m'+str(f)].reshape(28,28);ii=np.clip(np.rint(x32*14+13.5).astype(np.int32),0,27);jj=np.clip(np.rint(y32*14+13.5).astype(np.int32),0,27);inside=(mask[jj,ii]>.59)&(abs(x32)<1)&(abs(y32)<1)
    lum=.30*raw[0]+.52*raw[1]+.18*raw[2]
    local=[sample(z,w).reshape(-1,32,32) for z in m[:8]]
    for i,ch in ((4,lum),(5,raw[1]),(6,raw[0])):local[i]=.88*localtone(ch,inside).reshape(-1,32,32)+.12*local[i]
    pp=np.zeros_like(p);pp[:,:2]=15.5;pp[:,2:4]=12
    # A uniform blank face may have a strong illumination gradient; test fine
    # structure relative to its *own red rim*, rather than absolute brightness.
    L=(.65*lum+.35*raw[0]).reshape(-1,32,32);lm=smooth(smooth(L));rad=np.sqrt(x32*x32+y32*y32);inn=rad<.63;rim=(rad>.835)&(rad<.955)
    hp=(smooth(L)-smooth(smooth(smooth(lm)))).reshape(-1,1024);swing=np.percentile(hp[:,inn],95,axis=1)-np.percentile(hp[:,inn],5,axis=1)
    val=L.reshape(-1,1024);paper=np.percentile(val[:,inn],80,axis=1);paint=np.median(val[:,rim],axis=1);activity=(swing/(.004+.15*paper+abs(paper-paint))).astype(np.float32)
    return local,pp,activity

def scoresymbol(m,f,p,activity):
    red,blue,yellow,sat,l,gg,rr,edge=m[:8];cnt=len(p)
    mask=CONSTANTS['m'+str(f)].reshape(1,-1);mw=np.sqrt(mask);den=mask.sum()
    # Invert dark figures on paper; for blue or solid red plates seek light figures.
    if f in (0,1):
        a=-(.40*l+.38*rr+.22*gg)
        # A red outline is not black ink. Keep it from becoming a false stroke.
        a=a+.22*np.maximum(red,0)
    elif f==2:a=.48*rr+.52*l-.32*blue
    elif f in (5,6):a=.64*gg+.36*l-.22*red
    else:a=-l
    if f in (0,1,2):
        # Subtract a coarse reflected background. Outside the paper use its typical
        # light tone, not the road or sky behind the sign.
        interior=(mask.reshape(28,28)[np.clip(np.rint(y32*14+13.5).astype(np.int32),0,27),np.clip(np.rint(x32*14+13.5).astype(np.int32),0,27)]>.65)&(abs(x32)<1)&(abs(y32)<1)
        paper=np.percentile(a.reshape(cnt,1024)[:,interior],20,axis=1).astype(np.float32)[:,None,None];field=np.where(interior.reshape(1,32,32),a,paper)
        # Integral image box mean; padding carries the paper level.
        acc=np.pad(field,((0,0),(4,5),(4,5)),mode='edge').cumsum(1).cumsum(2);low=(acc[:,9:,9:]+acc[:,:-9,:-9]-acc[:,9:,:-9]-acc[:,:-9,9:])/81
        a=a-.45*(low-paper)
    q=sampler(*coordinates(p,shiftx,shifty));raw=sample(a,q).reshape(cnt*ns,28,28)
    # Subtract a broad, within-photo background estimate; preserve weak features in deep shade.
    x=raw.reshape(-1,784);midx=(mask.ravel()>.60)
    v0=np.percentile(x[:,midx],10,axis=1).astype(np.float32)[:,None];v1=np.percentile(x[:,midx],93,axis=1).astype(np.float32)[:,None]
    width=v1-v0+.014;x=cropfit(x,v0-.30*width,v1+.27*width);raw=x.reshape(-1,28,28)
    slow=smooth(smooth(smooth(smooth(raw))));h=(raw-slow).reshape(-1,784)
    def normal(z,floor):
        z=(z-(z*mask).sum(1,keepdims=True)/den)*mw
        return z/np.sqrt((z*z).sum(1,keepdims=True)+floor)
    bank=CONSTANTS['b'+str(f)];idx=CONSTANTS['i'+str(f)]
    sc=.76*product(normal(x,.020),bank[0])+.24*product(normal(h,.008),bank[1]);sc=sc.reshape(cnt,ns,len(idx)).max(1)
    direct=np.stack([sc[:,idx==c].max(1) for c in families[f]],axis=1)
    if f!=7:
        tight,ocr=foreground(a,m,f,p,mask)
        direct=.49*direct+.37*tight+.14*np.maximum(direct,tight)
        if f==0:direct[:,:8]=.70*direct[:,:8]+.30*ocr
    ans=[direct[:,i] for i in range(len(families[f]))]
    if f==0:
        # Blank face: the body has no lettering or pictogram. Measure its internal
        # contrast relative to the surrounding ring rather than the camera exposure.
        w=sampler(*coordinates(p,xg,yg));z=sample(a,w);rad=np.sqrt(xg*xg+yg*yg);inner=z[:,rad<.64];outer=z[:,(rad>.845)&(rad<.97)]
        qlo=np.percentile(inner,20,axis=1);hi=np.percentile(inner,93,axis=1);contrast=hi-qlo
        border=np.maximum(.04,outer.mean(1)-qlo);blank=.84-.62*contrast/(.13+border)-.18*contrast
        blank=.35*blank+.65*(.83-1.75*activity)
        # A featureless face should not have a convincing non-blank matched symbol.
        blank=blank-.65*np.maximum(0,direct.max(1)-.40);ans.append(cropfit(blank,-.25,.77))
        # The left vehicle of an overtaking-ban sign is painted red, not black.
        z=sample(red,w);left=(xg<-.07)&(xg>-.56)&(abs(yg)<.24);right=(xg>.07)&(xg<.56)&(abs(yg)<.24)
        bonus=cropfit(z[:,left].mean(1)-z[:,right].mean(1),-.30,1.0)
        ans[8]=ans[8]+.19*bonus;ans[9]=ans[9]+.19*bonus
    return np.stack(ans,axis=1)

def support(local,f):
    # A chance patch of the right colour should not substitute for a complete rim.
    co,li,ed,re=groupmaps(local,f);k=CONSTANTS['pro'+str(f)];n=len(co);color=product(co.reshape(n,1024),k[0]);absolute=product(co.reshape(n,1024),k[2]);lig=product(li.reshape(n,1024),k[1]);level=product(li.reshape(n,1024),k[2])
    if f in (0,1,3):ev=.70*color+.57*lig
    elif f==2:ev=.85*color
    elif f==4:ev=.79*color+.60*lig
    elif f in (5,6):ev=.85*color
    else:return -.20*np.median(absolute,axis=1)
    ordered=np.sort(ev,axis=1)
    return .20*ordered[:,1:4].mean(1)+.075*ordered[:,4:7].mean(1)-.045

def process(batch):
    m=features(batch);count=len(batch);base=np.zeros((count,8),np.float32);pos=[]
    for f in range(8):
        v,p=location(groupmaps(m,f),f,0);base[:,f]=v;pos.append(p)
    # Hierarchical verification: shortlist face shapes, then consider their interiors.
    order=np.argsort(-base,axis=1);result=np.full((count,43),-100.,np.float32)
    for f in range(8):
        sel=np.nonzero((order[:,:4]==f).any(1))[0]
        for start in range(0,len(sel),16):
            take=sel[start:start+16];s=[v[take] for v in m];cm=groupmaps(s,f);v,p=refine(cm,f,pos[f][take]);geo=.42*base[take,f]+.58*v
            orig=s;originalpose=p;s,p,activity=canvas(s,f,p);geo=geo+support(s,f)
            if f in (3,4):
                # Both have a nearly featureless, uniform face.
                w=sampler(*coordinates(originalpose,xg,yg));z=sample(orig[4],w);inside=(abs(xg)+abs(yg)<.58) if f==4 else ((yg>-.65)&(yg<.32-1.94*abs(xg)))
                iz=z[:,inside];span=np.percentile(iz,90,axis=1)-np.percentile(iz,10,axis=1)
                match=cropfit(.61-.51*span,.07,.61)[:,None]
            else:match=scoresymbol(s,f,p,activity)
            cls=families[f]+([15] if f==0 else [])
            # Template NCC is a second, independent witness in addition to the frame.
            result[np.ix_(take,cls)]=geo[:,None]+.86*(match-.50)
    return result.argmax(1)

def predict(x):
    result=np.empty(len(x),dtype=np.int64)
    for i in range(0,len(x),32):result[i:i+32]=process(x[i:i+32])
    return result
