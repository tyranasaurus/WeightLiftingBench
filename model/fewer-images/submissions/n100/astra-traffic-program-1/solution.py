import numpy as np
C=CONSTANTS

def sat(a):return np.clip(a,0,1)
def grid(n,extent):
    a=((np.arange(n,dtype=np.float32)+.5)*(2/n)-1)*extent
    return np.meshgrid(a,a)

def prepare(a):
    r,g,b=a[:,0],a[:,1],a[:,2]
    lum=np.sqrt(.16*r+.62*g+.22*b+.003)
    box=lum[:,5:27,5:27].reshape(len(a),-1)
    pp=np.partition(box,(28,450),axis=1);lo=pp[:,28,None,None];hi=pp[:,450,None,None]
    light=np.clip((lum-lo)/np.maximum(hi-lo,.068),-.3,1.3)
    red=sat(3.25*(r-.62*g-.38*b)/(r+.62*g+.38*b+.025))
    blue=sat(2.5*(.75*b+.25*g-r)/(r+.75*b+.25*g+.025))
    yellow=sat(3*(.38*r+.62*g-b)/(.38*r+.62*g+b+.025))*sat(1.3-1.35*red)
    return np.stack((light,red,blue,yellow),axis=1)

def warp(a,p,u,v):
    # Bilinear sampler shared by detector and interior rectifier. x,y axes have independent scale.
    n,ch,h,w=a.shape
    ux=u.reshape(1,-1);vy=v.reshape(1,-1)
    xx=np.clip(p[:,0:1]+p[:,2:3]*ux+p[:,4:5]*vy,0,w-1.0001); yy=np.clip(p[:,1:2]+p[:,3:4]*vy+p[:,5:6]*ux,0,h-1.0001)
    ix=xx.astype(np.int32);iy=yy.astype(np.int32);dx=(xx-ix).astype(np.float32)[:,None,:];dy=(yy-iy).astype(np.float32)[:,None,:]
    ij=(ix+w*iy)[:,None,:]; a=a.reshape(n,ch,-1)
    top=(1-dx)*np.take_along_axis(a,ij,axis=2)+dx*np.take_along_axis(a,ij+1,axis=2)
    bot=(1-dx)*np.take_along_axis(a,ij+w,axis=2)+dx*np.take_along_axis(a,ij+w+1,axis=2)
    return ((1-dy)*top+dy*bot).reshape((n,ch)+u.shape)

# Pose-complexity costs favor an ordinary, central crop in the absence of evidence.
cp=C['coarsep'];fp=C['finep'];nf=len(C['coarse']);ug,vg=grid(20,1.30)
prior=.003*((cp[:,0]-15.5)**2+(cp[:,1]-15.5)**2)/5+.012*((cp[:,2]+cp[:,3]-24)/7)**2
fprior=.003*((fp[:,0]/.1)**2+(fp[:,1]/.1)**2)+.002*((fp[:,4]/.10)**2+(fp[:,5]/.09)**2)
def poses(a):
    n=len(a);small=a.reshape(n,4,16,2,16,2).mean((3,5)).reshape(n,-1)
    sc=(small @ C['coarse'].reshape(-1,1024).T).reshape(n,nf,-1)-prior
    inds=np.argmax(sc,axis=2);allp=[];alls=[]
    for f in range(nf):
        p=cp[inds[:,f]];obs=warp(a,p,ug,vg).reshape(n,-1)
        s=(obs @ C['fine'][f].T)-fprior; ix=s.argmax(1);q=fp[ix];v=s[np.arange(n),ix]-prior[inds[:,f]]
        p=np.stack((p[:,0]+p[:,2]*q[:,0],p[:,1]+p[:,3]*q[:,1],p[:,2]*q[:,2],p[:,3]*q[:,3],p[:,2]*q[:,2]*q[:,4],p[:,3]*q[:,3]*q[:,5]),axis=1)
        allp.append(p); alls.append(v)
    return np.stack(allp,1),np.stack(alls,1)

def unit(q,w):
    q=q-(q*w).sum((1,2),keepdims=True)/(w.sum()+.000001);q=q*np.sqrt(w);return q/(np.sqrt(np.sum(q*q,(1,2),keepdims=True))+.00001)

def smooth(a,n):
    h,w=a.shape[1:];p=np.pad(a,((0,0),(n,n),(n,n)),mode='edge')
    return sum(p[:,i:i+h,j:j+w] for i in range(2*n+1) for j in range(2*n+1))/(2*n+1)**2

def shapevec(t,w):
    t=t*w; z=smooth(t,1); gy,gx=np.gradient(z,axis=(1,2));kx=smooth(abs(gx)*w,1);ky=smooth(abs(gy)*w,1)
    def bn(m):
        q=m.reshape(len(m),6,4,6,4).sum((2,4)).reshape(len(m),-1);q=np.sqrt(q+.00001);return q/(np.sqrt((q*q).sum(1,keepdims=True))+.00001)
    return np.concatenate([bn(t),bn(kx),bn(ky)],axis=1)/3**.5

def descriptors(a,feature,pose,group):
    w=C[group+'w'];ny,nx=w.shape
    if group=='warn':xm,yl,yr=.81,-.72,.82
    else:xm,yl,yr=.765,-.567,.567
    u,v=np.meshgrid(((np.arange(nx,dtype=np.float32)+.5)*2/nx-1)*xm,yl+(np.arange(ny,dtype=np.float32)+.5)*(yr-yl)/ny)
    raw=warp(a,pose,u,v);wr,wg,wb=raw[:,0],raw[:,1],raw[:,2]
    # Red ink should register as ink in the paired car symbols too. Square root
    # compresses highlights without blowing up quantization noise in heavy shade.
    gray=np.sqrt(.37*wr+.47*wg+.16*wb+.003); strong=np.sqrt(.72*wr+.20*wg+.08*wb+.003)
    if group=='end': gray=np.sqrt(.28*wr+.53*wg+.19*wb+.003)
    def inks(g):
        # Estimate local paper brightness by propagating light values within the face.
        z=np.where(w>.2,g,0);pp=np.pad(z,((0,0),(3,3),(3,3)),mode='edge');env=z
        for i in (0,2,4,6):
            for j in (0,2,4,6):env=np.maximum(env,pp[:,i:i+ny,j:j+nx])
        # Soft reference to the global paper brightness stabilizes broad objects.
        ref=np.max(np.where(w>.5,g,0),axis=(1,2),keepdims=True)
        ink=(.84*env+.16*ref-g)/(.55*env+.45*ref+.045)
        return ink
    ink=inks(gray); alt=inks(strong)
    mix=.4*ink+.6*alt
    if group=='warn':
        # The painted frame is red, not black ink. The red-channel test suppresses it even
        # when pose rectification puts a little of the frame inside the triangular mask.
        hue=warp(feature[:,1:2],pose,u,v)[:,0];mix=(.18*ink+.82*alt)*(1-.78*hue);ink=mix
    obj=unit(ink,w).reshape(len(a),-1); robj=unit(mix,w).reshape(len(a),-1)
    hp=unit(mix-smooth(mix,2),w).reshape(len(a),-1);hp2=unit(ink-smooth(ink,2),w).reshape(len(a),-1)
    if group=='warn':
        core=ink[:,w>.65];a0=np.partition(core,core.shape[1]//3,axis=1)[:,core.shape[1]//3,None,None]
        shape=shapevec(np.maximum(ink-a0,0),w)
    dim=nx*ny;templates=C[group+'s']; ids=C[group+'ids'];labels=np.unique(ids);sc=np.full((len(a),43),-2,dtype=np.float32)
    for c in labels:
        start=np.searchsorted(ids,c);stop=np.searchsorted(ids,c,side='right');t=templates[start:stop]
        s=.76*((robj if c<=8 else obj) @ t.T)+.24*((hp if c<=8 else hp2) @ C[group+'e'][start:stop].T)
        if group=='warn':s+=.82*((shape @ C['warnshape'][start:stop].T)-.90)
        if group=='red' and c<6:
            part=unit(.4*ink+.6*alt,w*sat((.035-u)/.085)).reshape(len(a),dim)
            s=.64*s+.36*(part @ C['fdigs'][start:stop].T)
        if group=='red' and c in (7,8):
            mid=unit(mix,w*sat((.35-abs(u))/.12)).reshape(len(a),dim);off=np.searchsorted(ids,7)
            s=.72*s+.28*(mid @ C['mdigs'][start-off:stop-off].T)
        sc[:,c]=np.max(s,axis=1)
    core=(w>.5);mag=np.sort(ink[:,core],axis=1);amp=mag[:,int(.88*mag.shape[1])]-mag[:,int(.26*mag.shape[1])];
    if group=='red':
        # Paper contrast is judged relative to the ring in the same photograph;
        # absolute contrast alone is not reliable under washed-out illumination.
        rx,ry=grid(24,1);full=warp(a,pose,rx,ry);g=np.sqrt(.26*full[:,0]+.52*full[:,1]+.22*full[:,2]+.003);radius=np.hypot(rx,ry);ref=np.percentile(g[:,radius<.72],78,axis=1);ring=g[:,(radius>.82)&(radius<.93)].mean(1);inkref=np.maximum(.06,(ref-ring)/(.08+ref));amp=amp/np.sqrt(.20+.80*inkref)
    hue=warp(feature[:,1:2],pose,u,v)[:,0];carred=(hue*w*((u<-.03)&(abs(v)<.3))).sum((1,2))/(w*((u<-.03)&(abs(v)<.3))).sum()-(hue*w*((u>.03)&(abs(v)<.3))).sum((1,2))/(w*((u>.03)&(abs(v)<.3))).sum()
    if group=='warn':
        # A traffic-light stack has a red upper lamp and a green lower lamp. In the other
        # narrow (exclamation / human) symbols this vertical chroma contrast is absent.
        def meanpatch(t,cx,cy,rx,ry):
            m=((u-cx)/rx)**2+((v-cy)/ry)**2<1;return t[:,m].mean(1)
        opponent=(wr-wg)/(wr+wg+.025);col=meanpatch(opponent,0,-.11,.16,.16)-meanpatch(opponent,0,.50,.15,.16)
        sc[:,26]+=.72*col-.075;sc[:,18]-=.14*np.maximum(col,0)
        # Bike wheels are a roughly symmetric pair low in the triangular field.
        # The skidding vehicle has a broad dark roof above two lighter tire traces;
        # a deer has its body and rump nearly on the same diagonal as its head.
        def rectmean(z,l,r,t,b):return z[:,(u>l)&(u<r)&(v>t)&(v<b)].mean(1)
        q=np.maximum(0,ink-a0);q=q/(.035+np.max(q*w,(1,2),keepdims=True));lc=rectmean(q,-.48,-.17,.22,.61);rc=rectmean(q,.17,.48,.22,.61);tc=rectmean(q,-.18,.18,-.23,.02)
        bilateral=np.minimum(lc,rc);sc[:,29]+=.18*bilateral-.15*tc-.08*abs(lc-rc)
        top=rectmean(q,-.34,.32,-.08,.23);bottom=rectmean(q,-.34,.32,.40,.63);diag=rectmean(q,.21,.48,.30,.53)-rectmean(q,-.48,-.21,.30,.53)
        sc[:,23]+=.12*(top-bottom)-.085*abs(diag);sc[:,31]+=.09*abs(diag)
    return sc,amp,carred

# Score simple observables appropriate to a single sign (e.g., a horizontal bar has no
# vertical lettering texture). The poses are taken from the independent outline searches.
def cues(ft,p,f):
    u,v=grid(26,1.16);m=warp(ft,p,u,v);l,r,b,y=m[:,0],m[:,1],m[:,2],m[:,3]
    def avg(t,w):return t[:,w].mean(1)
    def diff(t,u1,u2):return avg(t,u1)-avg(t,u2)
    if f==3:
        d=abs(u)+abs(v);core=d<.66;band=(d>.79)&(d<.92);outside=(d>1.065)&(d<1.23)
        # Four high brightness transitions along the sides of the diamond independent
        # of the central yellow hue (which is nearly neutral under some illuminants).
        octant=[]
        for sx,sy in ((1,1),(1,-1),(-1,1),(-1,-1)):
            mask=(u*sx>0)&(v*sy>0);octant.append(abs(diff(l,band&mask,outside&mask)))
        edge=np.median(np.stack(octant),axis=0)
        paper=l[:,band];ctr=l[:,core];darkborder=np.maximum(0,paper.mean(1)-ctr.mean(1));sym=[]
        for sx,sy in ((1,1),(1,-1),(-1,1),(-1,-1)):
            mask=(u*sx>0)&(v*sy>0);sym.append(diff(l,band&mask,core&mask))
        border=np.sort(np.stack(sym,1),axis=1)[:,1]
        return .32*avg(y,core)+.28*edge+.18*darkborder+.20*np.maximum(border,0)+.13*diff(y,core,band)-.10*avg(r,band)
    if f==6:
        along=(u-v)/1.414;across=(v+u)/1.414;circle=u*u+v*v<.87**2
        # Local normalization is essential for a gray stripe on a dark, backlit disk.
        # Require the diagonal to continue through both halves of the circle.
        ds=[]
        for v0 in (-.57,-.18,.18,.57):
            zone=(abs(along-v0)<.18)&circle;mid=zone&(abs(across)<.12);side=zone&(abs(across)>.24)&(abs(across)<.46)
            high=avg(l,side);low=avg(l,mid);ds.append((high-low)/(.08+abs(high-low)))
        # Cancellation signs have no colored rim. Inspect the candidate's rim as well
        # as its center: a black stroke in the middle of a red warning sign is not
        # evidence that the red frame belongs to the background.
        rim=(u*u+v*v>.78**2)&(u*u+v*v<1.04**2)
        return .34*np.mean(ds,axis=0)+.13*np.min(ds,axis=0)-.13*avg(b,circle)-.26*avg(r,rim)
    # Variation along a horizontal line distinguishes STOP letters from no-entry bar.
    texture=np.abs(l[:,:,2:]-l[:,:,:-2]);iw=(abs(v[:,1:-1])<.26)&(abs(u[:,1:-1])<.72)
    if f==4:return .55*avg(texture,iw)-.06
    return -.59*avg(texture,iw)+.04

def ocr(a,p):
    u,v=np.meshgrid((np.arange(14,dtype=np.float32)+.5)/14-.5,(np.arange(22,dtype=np.float32)+.5)/22-.5)
    # Normalize each possible character box separately. Use red light to suppress
    # bleeding from the red border; preserve genuine black figures in all channels.
    def readchar(cx,width,indices):
        g=warp(a,p,u*width+cx,v*1.05);g=np.sqrt(.62*g[:,0]+.24*g[:,1]+.14*g[:,2]+.002)
        pp=np.pad(g,((0,0),(3,3),(3,3)),mode='edge');env=g
        for dy in (0,3,6):
            for dx in (0,3,6):env=np.maximum(env,pp[:,dy:dy+22,dx:dx+14])
        z=(env-g)/(.55*env+.45*g.max((1,2),keepdims=True)+.035)-.25*g/(.2+g.mean((1,2),keepdims=True))
        z-=z.mean((1,2),keepdims=True);z=(z/(np.sqrt((z*z).sum((1,2),keepdims=True))+.0001)).reshape(len(a),-1)
        out=[]
        for j in indices:
            l,r=C['ocrsplit'][j:j+2];s=z @ C['ocr'][l:r].T-C['ocrcost'][l:r];out.append(s.max(1))
        return np.stack(out,1)
    d2=readchar(-.285,.72,[2,3,4,5,6,7]);zero=readchar(.285,.72,[0]);two=.70*d2+.30*zero
    left=readchar(-.45,.55,[1]);mid=readchar(0,.60,[0,2]);right=readchar(.43,.60,[0]);three=.27*left+.49*mid+.24*right
    return np.concatenate([two,three],axis=1)

def counters(a,p):
    # Measure enclosed light counters without binarizing the photograph. For each
    # light candidate, compare it to the darkest points along four paths out of it.
    # If all four paths cross a darker stroke, this is evidence of a closed loop.
    ux=np.linspace(-.83,.83,61,dtype=np.float32);vy=np.linspace(-.66,.66,51,dtype=np.float32);u,v=np.meshgrid(ux,vy);g=warp(a,p,u,v);g=np.sqrt(.65*g[:,0]+.22*g[:,1]+.13*g[:,2]+.002)
    def loops(t,l,r,mid):
        q=g[:,t:t+mid,l:r];n,h,w=q.shape
        # Grayscale reconstruction from the four sides. A diagonal or winding
        # white path is just as valid an escape route as a straight one.
        fill=q.copy();fill[:,1:-1,1:-1]=0
        for step in range(h+w):
            pp=np.pad(fill,((0,0),(1,1),(1,1)),mode='edge');spread=np.maximum.reduce((pp[:,:h,:w],pp[:,:h,1:w+1],pp[:,:h,2:],pp[:,1:h+1,:w],pp[:,1:h+1,2:],pp[:,2:,:w],pp[:,2:,1:w+1],pp[:,2:,2:],fill));nxt=np.minimum(q,spread)
            if np.array_equal(fill,nxt):break
            fill=nxt
        contrast=q-fill
        return contrast[:,h//4:h*3//4,w//5:w*4//5].max((1,2))
    top=loops(3,10,27,24);bot=loops(22,10,27,25);other=loops(5,32,52,40)
    # Relative to the counter of the common trailing zero. A small margin protects
    # against flat paper highlights or isolated compression noise.
    return sat((top-.012)/(.035+other)),sat((bot-.012)/(.035+other))

def localblue(ft,poses):
    u,v=grid(24,1);w=C['bluew'];res=[]
    for c in range(33,41):
        m=warp(ft,poses[:,c-26],u,v);l,r,b=m[:,0],m[:,1],m[:,2];ink=l-.4*b
        obj=unit(ink,w).reshape(len(l),-1);lo,hi=np.searchsorted(C['blueids'],[c,c+1]);sc=(obj @ C['blues'][lo:hi].T).max(1)
        if c==40:
            # A ring of white paper around black ink is not three white arrows:
            # there should be blue ground on BOTH sides of the orbit.
            rad=np.hypot(u,v);ground=np.minimum(b[:,rad<.39].mean(1),b[:,(rad>.75)&(rad<.90)].mean(1));s=.36*(sc-.50)+.16*(ground-.37)-.28*r[:,w>.4].mean(1)
        else:s=.36*(sc-.50)
        res.append(s)
    return np.stack(res,axis=1)

def classify(a):
    n=len(a); ft=prepare(a); p,base=poses(ft);scores=np.full((n,43),-2,np.float32)
    scores[:,12]=base[:,3]+cues(ft,p[:,3],3);scores[:,13]=base[:,2];scores[:,14]=base[:,4]+cues(ft,p[:,4],4);scores[:,17]=base[:,5]+cues(ft,p[:,5],5)
    base[:,6]+=.85*cues(ft,p[:,6],6)
    scores[:,33:41]=base[:,7:15]+localblue(ft,p)
    for group,k,ls in [('red',0,[0,1,2,3,4,5,7,8,9,10,16]),('warn',1,[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]),('end',6,[6,41,42])]:
        val,amp,carred=descriptors(a,ft,p[:,k],group);q=base[:,k]
        if k==6:
            scores[:,32]=q-.02+.055*sat((.15-amp)/.12)
            for c in ls:scores[:,c]=q-.03+.21*(val[:,c]-.36)+.045*sat((amp-.08)/.14)
        else:
            if k==0:
                numbers=[0,1,2,3,4,5,7,8];digits=ocr(a,p[:,0]);word=val[:,numbers]
                # The character classifier contributes mostly within its string length;
                # the whole-word recognizer is the one that can detect omitted ink.
                for start,stop in ((0,6),(6,8)):
                    d=digits[:,start:stop];z=word[:,start:stop];val[:,numbers[start:stop]]=.70*z+.30*(d-d.max(1,keepdims=True)+z.max(1,keepdims=True))
                upper,lower=counters(a,p[:,0]);val[:,5]+=.13*np.minimum(upper,lower);val[:,3]+=.055*lower*(1-upper)-.04*upper
                for c in (0,1,2,4):val[:,c]-=.025*(upper+lower)
                val[:,9]+=.10*carred;val[:,10]+=.10*carred
                # Lack of ink alone is ambiguous under low contrast: a blank face must
                # also lack coherent strokes at every tested numeral / vehicle position.
                coherent=val[:,ls].max(1);scores[:,15]=q+.075-.23*sat((amp-.043)/.19)-.27*np.maximum(0,coherent-.42)
            for c in ls:scores[:,c]=q+.31*(val[:,c]-.54)
    return scores.argmax(1)

def predict(x):
    # Pad partial blocks so every matrix product uses the same shape independently of batching.
    labels=np.empty(len(x),dtype=np.int64);bs=24
    for j in range(0,len(x),bs):
        a=x[j:j+bs];m=len(a)
        if m<bs:a=np.concatenate((a,np.repeat(a[-1:],bs-m,axis=0)))
        labels[j:j+m]=classify(a)[:m]
    return labels
