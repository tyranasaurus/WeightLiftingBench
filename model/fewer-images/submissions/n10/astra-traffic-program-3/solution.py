import numpy as np
import math

poses=CONSTANTS['pose']
outerL=CONSTANTS['light']
outerC=CONSTANTS['color']
famlist=[0]*6+[4,0,0,0,0,1,7,2,5,0,0,6]+[1]*14+[4]+[3]*8+[4,4]

def normall(a):
    a=a.reshape(len(a),-1)
    a=a-a.mean(axis=1,keepdims=True)
    return a/(np.sqrt(np.sum(a*a,axis=1,keepdims=True))+.0001)

def chroma(a):
    # Channel ratios reduce sensitivity to irradiance. Subtract a per-image
    # neutral estimate to adapt the opponent channels to the illuminant.
    r,g,b=a[:,0],a[:,1],a[:,2]
    red=r-.70*g-.30*b;blue=b-.78*r-.22*g;yellow=(r+g)*.50-b
    v=(r+g+b)/3+.023
    red,blue,yellow=red/v,blue/v,yellow/v
    def normc(c):
        baseline=np.quantile(c,.25,axis=(1,2)).astype(np.float32)[:,None,None]
        return np.clip((c-.65*baseline-.035)/.55,0,1.5)
    return normc(red),normc(blue),normc(yellow)

def grids(f,n):
    v=(np.arange(n,dtype=np.float32)+.5)*2/n-1;x,y=np.meshgrid(v,v)
    if f==1: x,y=x*.70,y*.62+.17
    elif f==3 or f==4: x,y=x*.83,y*.83
    else:x,y=x*.71,y*.58
    if f==1:m=np.clip((.81-np.maximum((y-1/3)/(2/3),(2*abs(x)-y+1/3)/(4/3)))/.20,0,1)
    else:m=np.clip(((.79 if f==0 else .90)-np.sqrt(x*x+y*y))/.12,0,1)
    return x,y,m

insidegrids=[grids(f,24) for f in range(8)]

def sample(img,x,y):
    x=np.clip(x,0,30.9999);y=np.clip(y,0,30.9999);ix=x.astype(np.int32);iy=y.astype(np.int32);u=x-ix;v=y-iy
    k=np.arange(len(img)).reshape((-1,)+(1,)*(x.ndim-1))
    return ((1-u)*((1-v)*img[k,iy,ix]+v*img[k,iy+1,ix])+u*((1-v)*img[k,iy,ix+1]+v*img[k,iy+1,ix+1])).astype(np.float32)

def boundary(f,theta):
    x=np.cos(theta);y=np.sin(theta)
    if f==1 or f==2:
        s=1 if f==1 else -1
        d=np.maximum(s*y,2*abs(x)-s*y)
        # clipped vertices; use separate distance to the base/oblique edges below
        return x/d,y/d
    if f==7:d=abs(x)+abs(y)
    elif f==5:d=np.maximum(np.maximum(abs(x),abs(y)),(abs(x)+abs(y))/math.sqrt(2))
    else:d=np.sqrt(x*x+y*y)
    return x/d,y/d

# Uniform arc-length sampling, afocal contrast across the INSIDE of the coloured
# border. This is much less sensitive to the background than the exterior edge.
def rimpoints(f):
    if f in (1,2):
        s=1 if f==1 else -1
        pts=[(0,-1*s),(1,1*s),(-1,1*s)]; groups=[]
        for j in range(3):
            a,b=pts[j];c,d=pts[(j+1)%3];t=np.linspace(.09,.91,12)
            groups.extend(zip(a+(c-a)*t,b+(d-b)*t))
        xy=np.array(groups);x,y=xy.T
    else:
        x,y=boundary(f,(np.arange(40)+.5)*2*math.pi/40)
    # In a triangle, scaling must be about its incenter rather than its box center.
    y0=(.32 if f==1 else -.32 if f==2 else 0)
    return x.astype(np.float32), (y-y0).astype(np.float32), y0
rimgrids=[rimpoints(f) for f in range(8)]

def contrast(v):
    return np.mean(np.clip(v,-1,1),axis=-1)

def geomvalue(img,col,f,pose):
    cx,cy,w,h,sk,sl=[pose[:,:,j,None] for j in range(6)];x,y,y0=rimgrids[f]
    def at(z,a):
        X=x*a;Y=y*a+y0
        return sample(z,cx+w*(X+sk*Y),cy+h*(Y+sl*X))
    if f in (0,1,2):
        # all signs have an approximately uniform coloured rim
        inner=.73 if f==0 else .65
        mid=.89 if f==0 else .865
        lr=at(img,mid);li=at(img,inner);cr=at(col,mid);ci=at(col,inner)
        diff=contrast((li-lr)*1.4)+.53*contrast(cr-ci)
        return diff+.13*contrast(cr)-.12*np.std(np.clip(li-lr,-.7,.7),axis=-1)
    if f==6:
        # Follow both long edges of the white bar, not just the outside disc.
        xx=np.linspace(-.64,.64,15,dtype=np.float32)
        def band(a,b):return sample(a,cx+w*(xx+sk*b),cy+h*(b+sl*xx))
        li=band(img,0);lu=band(img,-.26);ld=band(img,.26);cc=band(col,.26)+band(col,-.26)-2*band(col,0)
        return .70*contrast(li-.5*(lu+ld))+.20*contrast(cc)+.25*contrast(at(col,.86)-at(col,1.08))
    if f==4:
        li=at(img,.86);le=at(img,1.075)
        return .65*contrast((li-le)*1.6)+.27*np.mean(np.minimum(1,abs(li-le)*1.8),axis=-1)+.20*contrast(li)
    if f==7:
        delta=at(col,.57)-at(col,.89);li=at(img,.87);le=at(img,1.10)
        return .75*contrast(delta)+.4*contrast(li-le)+.22*contrast(li-at(img,.57))
    # Coloured disc or octagon with white narrow edge; outside may be anything.
    lc=at(img,.85);cc=at(col,.85);ce=at(col,1.09);le=at(img,1.08)
    wh=at(img,.969)-lc
    return .62*contrast(cc-ce)+.38*contrast(le-lc)+.36*contrast(wh)+.24*contrast(cc)

def refiner(img,col,f,p):
    p=np.pad(p,((0,0),(0,2)));best=geomvalue(img,col,f,p[:,None,:])[:,0]
    base=p.copy()
    for step in (1.3,.68):
        for dims in ((0,1),(2,3),(4,5),(0,1),(2,3),(4,5)):
            a,b=dims;opts=np.array([(u,v) for u in (-1,0,1) for v in (-1,0,1)],np.float32)*step
            cand=np.repeat(p[:,None,:],9,axis=1)
            mult=.075 if a==4 else 1
            cand[:,:,a]+=opts[:,0]*mult;cand[:,:,b]+=opts[:,1]*mult
            score=geomvalue(img,col,f,cand)
            score-=.0025*((cand[:,:,0]-15.5)**2+(cand[:,:,1]-15.5)**2)
            score-=.30*(cand[:,:,4]**2+cand[:,:,5]**2)+.002*((cand[:,:,2]-base[:,None,2])**2+(cand[:,:,3]-base[:,None,3])**2)
            ii=np.argmax(score,axis=1);p=cand[np.arange(len(p)),ii];best=score[np.arange(len(p)),ii]
    return p,best

def crop(a,p,f):
    x,y,m=insidegrids[f];cx,cy,w,h,s,t=[p[:,j,None,None] for j in range(6)]
    return sample(a,cx+w*(x+s*y),cy+h*(y+t*x))

def histgrad(a,m):
    v=np.pad(a,((0,0),(1,1),(1,1)),mode='edge');gx=(v[:,1:-1,2:]-v[:,1:-1,:-2])*m*m;gy=(v[:,2:,1:-1]-v[:,:-2,1:-1])*m*m
    channels=[gx,gy,(gx+gy)*.7071,(gx-gy)*.7071];n=len(a)
    b=np.stack([np.maximum(u*t,0).reshape(n,8,3,8,3).mean((2,4)) for u in channels for t in (-1,1)],axis=-1)
    v=np.pad(b,((0,0),(1,1),(1,1),(0,0)));b=.60*b+.10*(v[:,1:-1,2:]+v[:,1:-1,:-2]+v[:,2:,1:-1]+v[:,:-2,1:-1]);e=np.sum(b*b,axis=3,keepdims=True);b=b/(e+.28*e.mean((1,2,3),keepdims=True)+1e-12)**.25
    return (b/(np.sqrt(np.sum(b*b,axis=(1,2,3),keepdims=True))+1e-8)).reshape(n,-1)

def interior(img,red,blue,f,p):
    x,y,m=insidegrids[f];ms=m.sum();v=crop(img,p,f)
    if f==3:v=v-.20*crop(blue,p,f)
    v=v*(1 if f==3 else -1)
    if f==0:v=v+.18*crop(red,p,f)
    v=v-(v*m).sum((1,2),keepdims=True)/ms
    # A bounded contrast stretch keeps highlights and deep shadow edges from
    # dominating and improves the match of saturated or partially faded symbols.
    dev=np.sqrt((v*v*m).sum((1,2),keepdims=True)/ms)+.008
    v=np.clip(v/dev,-2.6,2.8)
    v=v-(v*m).sum((1,2),keepdims=True)/ms
    v=v/np.sqrt((v*v*m).sum((1,2),keepdims=True)+.008)
    feats=CONSTANTS['in'+str(f)];keys=CONSTANTS['keys'+str(f)];q=v.reshape(len(p),-1)@feats.T
    hq=histgrad(v,m)@CONSTANTS['hg'+str(f)].T
    q=.68*q+.32*(1.50*hq-.50)
    start=0;ans=np.full((len(p),43),-3,np.float32)
    for k,n in keys:
        ans[:,k]=q[:,start:start+n].max(axis=1);start+=n
    if f==0:
        ans[:,15]=np.clip(.90-3.8*dev[:,0,0],-.15,.85)
    return ans

def checkpose(img,col,f,p):
    # Make a final local comparison by the whole-sign appearance. Border evidence
    # alone can put a thin, high-contrast rim slightly inside the sign boundary.
    n=len(p);delta=np.array([(0,0,1,1),(-.8,0,1,1),(.8,0,1,1),(0,-.8,1,1),(0,.8,1,1),(0,0,.91,.91),(0,0,1.09,1.09),(0,0,.94,1.06),(0,0,1.06,.94)],np.float32)
    cand=np.repeat(p[:,None,:],len(delta),axis=1);cand[:,:,:2]+=delta[None,:,:2];cand[:,:,2:4]*=delta[None,:,2:]
    g=geomvalue(img,col,f,cand)
    g-=.0015*((cand[:,:,0]-15.5)**2+(cand[:,:,1]-15.5)**2)
    ix=np.repeat(np.arange(n),len(delta));cm=cand.reshape(-1,6);q=apmatch(img[ix],col[ix],f,cm).max(1).reshape(n,-1)
    merit=.66*q+.34*g;best=merit.argmax(1);return cand[np.arange(n),best]

def apmatch(lum,col,f,p):
    v=(np.arange(24,dtype=np.float32)+.5)/12-1;x,y=np.meshgrid(v*1.01,v*1.01);cx,cy,w,h,s,t=[p[:,j,None,None] for j in range(6)]
    X=cx+w*(x+s*y);Y=cy+h*(y+t*x);l=sample(lum,X,Y);c=sample(col,X,Y)
    m=CONSTANTS['am'+str(f)];ms=m.sum();kc=(c*m).sum((1,2))/ms
    # A plane of lighting can cross an entire sign. Local subtraction gives a
    # second view of its structure without making a bad shadow decisive.
    def comp(a,key,floor):
        a=a-(a*m).sum((1,2),keepdims=True)/ms;sd=np.sqrt((a*a*m).sum((1,2),keepdims=True)/ms)
        a=np.clip(a/(sd+floor),-2.6,2.6);a=a-(a*m).sum((1,2),keepdims=True)/ms
        a=a/np.sqrt((a*a*m).sum((1,2),keepdims=True)+.001)
        return a.reshape(len(a),-1)@CONSTANTS[key+str(f)].T,sd[:,0,0]
    q,ld=comp(l,'al',.02)
    if f!=4:
        qc,cd=comp(c,'ac',.03);rc=np.minimum(1,cd*5)[:,None]
        q=q+.57*rc*qc
        if f in (3,5,6,7):q+=.26*np.minimum(1,kc*2)[:,None]-.13
    else:q-=.32*kc[:,None]
    ans=np.full((len(p),43),-3,np.float32);st=0
    for k,n in CONSTANTS['ak'+str(f)]:ans[:,k]=q[:,st:st+n].max(1);st+=n
    return ans

def readnumber(lum,p):
    # Read each candidate at high resolution, demanding a common type size and
    # baseline. In particular, a 120 must explain three glyphs, not just the 20.
    n=len(p);v=(np.arange(24,dtype=np.float32)+.5)/12-1;gx,gy=np.meshgrid(v*1.25,v*1.18)
    cx,cy,w,h,s,t=[p[:,j,None,None,None] for j in range(6)];out=np.zeros((n,43),np.float32)
    ds=CONSTANTS['digits'];keys=CONSTANTS['dkey'];first=[2,3,5,6,7,8]
    for length in (2,3):
        scores=[[] for j in range(8)]
        for scale in (.84,1,1.17):
            for width in ((.86,1.06) if length==2 else (.83,1.03,1.21)):
                H=.455*scale;W=(.235 if length==2 else .185)*width;gap=(.085 if length==2 else .062)*width
                A=W+.5*gap
                centers=(-A,A) if length==2 else (-1.68*W-gap,-.32*W,1.68*W+gap)
                loc=[(c,dx,dy) for c in centers for dx in (-.042,.042) for dy in (-.055,.055)]
                xp=np.array([a+b for a,b,d in loc],np.float32)[:,None,None]+W*gx;yp=np.array([d for a,b,d in loc],np.float32)[:,None,None]+H*gy
                patch=-sample(lum,cx+w*(xp+s*yp),cy+h*(yp+t*xp));patch=patch-patch.mean((2,3),keepdims=True)
                patch=patch/(np.sqrt(np.sum(patch*patch,axis=(2,3),keepdims=True))+.045)
                q=patch.reshape(-1,576)@ds.T;st=0;sc=[]
                for d,cnt in keys:
                    sc.append(q[:,st:st+cnt].max(1).reshape(n,length,4));st+=cnt
                if length==2:
                    for j,fd in enumerate(first):
                        di=(2,3,4,5,6,7)[j]
                        val=(.57*sc[di][:,0]+.43*sc[0][:,1]).max(1)
                        scores[j].append(val)
                else:
                    for j,mid in ((6,0),(7,2)):
                        val=(.31*sc[1][:,0]+.37*sc[mid][:,1]+.32*sc[0][:,2]).max(1)
                        scores[j].append(val)
        for j,arr in enumerate(scores):
            if len(arr)>0:out[:,j if length==2 else j+1]=np.max(np.stack(arr),0)
    return out

def flattenlight(u):
    v=np.pad(u,((0,0),(3,3),(3,3)),mode='edge');v=np.pad(v,((0,0),(1,0),(1,0)));v=v.cumsum(1).cumsum(2)
    smooth=(v[:,7:,7:]-v[:,:-7,7:]-v[:,7:,:-7]+v[:,:-7,:-7])/49
    return u-.73*smooth

def scorebatch(a):
    n=len(a);red,blue,yellow=chroma(a);cols=(red,red,red,blue,red,red,red,yellow)
    rgb=.27*a[:,0]+.57*a[:,1]+.16*a[:,2]
    # Compress luminance mildly, then scale from this frame's central range.
    lum=np.sqrt(rgb+.012)
    lo,hi=np.quantile(lum[:,4:28,4:28],[.08,.92],axis=(1,2)).astype(np.float32)
    lum=np.clip((lum-lo[:,None,None])/(.036+hi[:,None,None]-lo[:,None,None]),-.8,1.65)
    l=normall(lum);nc=[normall(c) for c in cols];vals=[];seeds=[]
    for f in range(8):
        g=l@outerL[f].T
        if f!=4:
            c=cols[f];strength=np.minimum(1,c.std((1,2))*4.5)[:,None]
            g+=.76*strength*(nc[f]@outerC[f].T)
        g-=.002*((poses[f,:,0]-15.5)**2+(poses[f,:,1]-15.5)**2)
        # Whole-frame correlation is biased by clutter and parking signs in the
        # margin. Rerank candidate locations by edge polarity along the perimeter.
        candidates=np.argsort(g,axis=1)[:,-28:];cand=poses[f,candidates];trial=np.pad(cand,((0,0),(0,0),(0,2)))
        edge=geomvalue(lum,cols[f],f,trial);local=np.take_along_axis(g,candidates,axis=1)+.68*edge
        j=candidates[np.arange(n),local.argmax(1)];seeds.append(poses[f,j]);vals.append(g[np.arange(n),j])
    out=np.full((n,43),-8,np.float32)
    detail=flattenlight(lum)
    for f in range(8):
        p,geom=refiner(lum,cols[f],f,seeds[f]);base=.28*vals[f]+.26*geom
        p=checkpose(lum,cols[f],f,p)
        whole=apmatch(lum,cols[f],f,p)
        if f in (0,1,3,4):
            ii=interior(.53*lum+.47*detail,red,blue,f,p)
            if f==1:
                X,Y,M=insidegrids[1];R,G,B=[crop(a[:,ch],p,1) for ch in (0,1,2)]
                opp=(R-.65*G-.35*B)/(.04+(R+G+B)/3)
                region=np.exp(-((X/.15)**2+((Y-.03)/.16)**2));low=np.exp(-((X/.15)**2+((Y-.61)/.16)**2));flank=np.exp(-(((abs(X)-.23)/.10)**2+((Y-.27)/.28)**2))
                rv=(opp*region).sum((1,2))/region.sum();gv=(opp*low).sum((1,2))/low.sum();fv=(opp*flank).sum((1,2))/flank.sum()
                val=np.minimum(rv-gv,rv-fv+.035)
                ii[:,26]+=.72*np.clip(val,-.23,.78)-.07
                ii[:,26]+=.54*np.clip(np.minimum(rv-fv,fv-gv),-.22,.45)
            if f==0:
                digits=readnumber(.45*lum+.55*detail,p)
                for j in (0,1,2,3,4,5,7,8):ii[:,j]=.48*ii[:,j]+.52*digits[:,j]
                # A blank disc lacks local edges. Uniformity alone is ambiguous
                # if a candidate pose has locked onto a patch of background.
                vv=crop(lum,p,f);X,Y,M=insidegrids[0];mask=np.hypot(X,Y)<.65
                ctr=vv[:,mask];dr=np.quantile(ctr,.88,axis=1)-np.quantile(ctr,.06,axis=1)
                ii[:,15]=np.clip(.86-2.1*dr,-.20,.86)
                # Count separate columns of ink by projecting the central band.
                # An extra narrow '1' creates a third dark group and second gap;
                # this check does not depend on the detailed curve of the '2'.
                xx=np.linspace(-.75,.75,40,dtype=np.float32);yy=np.linspace(-.30,.30,13,dtype=np.float32);xx,yy=np.meshgrid(xx,yy)
                cx,cy,w,h,s,t=[p[:,j,None,None] for j in range(6)]
                lineimg=sample(lum,cx+w*(xx+s*yy),cy+h*(yy+t*xx));bright=np.quantile(lineimg,.85,axis=(1,2)).astype(np.float32)[:,None,None]
                ink=np.maximum(bright-lineimg,0);prof=ink.mean(1);noise=.045+prof.std(1)
                # Marginal left stem: compare to the gap just to its right, and
                # require ink in the middle character as well as the final zero.
                evidence=np.full(n,-2,np.float32)
                for shift in (-2,0,2):
                    for sep in (8,10,12):
                        mid=20+shift;left=mid-sep;gap=left+sep//2
                        v1=prof[:,left-2:left+2].max(1);v2=prof[:,gap-2:gap+2].min(1);v3=prof[:,mid-2:mid+3].max(1)
                        evidence=np.maximum(evidence,np.minimum(v1-v2,v3-v2)/noise)
                factor=np.clip((evidence-.90)*.095,-.09,.10);ii[:,7]+=factor;ii[:,8]+=factor
            if f==4:
                # A cancellation is achromatic and carries a strong positive-slope slash.
                bx,by,ww=insidegrids[f];cx=crop(red,p,f);cb=crop(blue,p,f)
                cv=(ww*(cx+.65*cb)).sum((1,2))/ww.sum();base-=.70*cv
                base+=.24
            for j in range(43):
                if famlist[j]==f:out[:,j]=base+.83*ii[:,j]+.74*whole[:,j]
        else:
            j=(13 if f==2 else 14 if f==5 else 17 if f==6 else 12)
            out[:,j]=base+1.13*whole[:,j]+.12
    return out.argmax(axis=1)

def predict(x):
    # Fixed internal block size, padding the last block to keep BLAS calls identical.
    result=np.empty(len(x),np.int64);size=32
    for begin in range(0,len(x),size):
        a=x[begin:begin+size];n=len(a)
        if n<size:a=np.concatenate((a,np.repeat(a[:1],size-n,axis=0)),axis=0)
        result[begin:begin+n]=scorebatch(a)[:n]
    return result
