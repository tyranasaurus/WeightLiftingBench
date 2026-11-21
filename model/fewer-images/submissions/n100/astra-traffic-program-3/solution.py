import numpy as np

posebank=CONSTANTS['poses']
shapeweights=[CONSTANTS['shape'+str(g)].T.copy() for g in range(7)]

def product(a,b):
    # Force the same GEMM tile dimensions and strides even when pruning
    # leaves a partial batch. Each dot product is still over one image only.
    n=len(a);out=np.empty((n,b.shape[1]),np.float32);chunk=64;p=np.zeros((chunk,a.shape[1]),np.float32)
    for i in range(0,n,chunk):
        k=min(n-i,chunk);p[:k]=a[i:i+k];out[i:i+k]=(p@b)[:k]
    return out
spatialprior=(.018*((posebank[:,0]-15.5)**2+(posebank[:,1]-15.5)**2)/9+.012*((posebank[:,2]-12)**2+(posebank[:,3]-12)**2)/16+.013*((posebank[:,2]-posebank[:,3])/3)**2).astype(np.float32)

def grid(g):
    if g==0: bx,by,cy,s=.77,.61,.01,25
    if g==1: bx,by,cy,s=.76,.69,.18,24
    if g==4 or g==6: bx,by,cy,s=.99,.99,0,26
    if g==5: bx,by,cy,s=.98,.6,0,25
    t=(np.arange(s)+.5)*2/s-1;x,y=np.meshgrid(t*bx,t*by+cy)
    if g==1:mask=(y<.78)&(y>2.03*np.abs(x)-.71)
    else:mask=x*x+y*y<({0:.79,4:.91,5:.91,6:.91}[g])**2
    return x[mask].astype(np.float32),y[mask].astype(np.float32)

icongroups=(0,1,4,5,6)
grids={g:grid(g) for g in icongroups}
classes={g:np.unique(CONSTANTS['labels'+str(g)]) for g in icongroups}
starts={g:np.searchsorted(CONSTANTS['labels'+str(g)],classes[g]) for g in icongroups}
shifts=np.array([(u,v,s,t) for s,t in [(1,1),(.94,.94),(1.06,1.06)] for u in (-1.05,0,1.05) for v in (-1.05,0,1.05)],np.float32)
shiftcost=(.012*(shifts[:,0]**2+shifts[:,1]**2)+.13*abs(shifts[:,2]-1)).astype(np.float32)

def sample(im,x,y):
    n=im.shape[0];xx=np.clip(x,0,30.9999);yy=np.clip(y,0,30.9999);u=xx.astype(np.int32);v=yy.astype(np.int32);xx=xx-u;yy=yy-v
    i=np.arange(n)[:,None,None];return (((1-xx)*im[i,v,u]+xx*im[i,v,u+1])*(1-yy)+((1-xx)*im[i,v+1,u]+xx*im[i,v+1,u+1])*yy).astype(np.float32)

def innerhist(a,g):
    e=CONSTANTS['he'+str(g)];pool=CONSTANTS['hp'+str(g)];dx=a[:,e[1]]-a[:,e[0]];dy=a[:,e[3]]-a[:,e[2]];mag=np.sqrt(dx*dx+dy*dy);hs=[]
    for theta in (0,.78539816,1.57079633,2.35619449):
        p=dx*np.float32(np.cos(theta))+dy*np.float32(np.sin(theta));hs.extend([product(np.maximum(0,p-.5*mag),pool),product(np.maximum(0,-p-.5*mag),pool)])
    z=np.concatenate(hs,axis=1);z/=np.maximum(1e-8,z.sum(1,keepdims=True));return np.sqrt(z)

def interior(g,lumin,poses,noise):
    n=len(lumin);u,v=grids[g];p=poses[:,None,:]+np.zeros((1,len(shifts),1),np.float32)
    p[:,:,0:2]+=shifts[:,0:2];p[:,:,2:4]*=shifts[:,2:4]
    x=p[:,:,0:1]+p[:,:,2:3]*u;y=p[:,:,1:2]+p[:,:,3:4]*v
    a=sample(lumin,x,y);mean=a.mean(2,keepdims=True);a-=mean
    # Remove a smooth brightness gradient so a shadow need not look like ink.
    for t in (u,v):
        z=t-t.mean();a-=.80*(a*z).sum(2,keepdims=True)/(z*z).sum()*z
    std=np.sqrt((a*a).mean(2,keepdims=True));a=np.maximum(np.minimum(a,2.85*std),-2.85*std);a-=a.mean(2,keepdims=True)
    if g not in (4,5):a=-a
    norm=np.sqrt((a*a).sum(2,keepdims=True));a/=np.maximum(norm,.00001)
    mat=a.reshape(n*len(shifts),-1);b=product(mat,CONSTANTS['icons'+str(g)].T)
    # For numbers, an extra local correlation compares the leading digit and
    # reduces interference from the zero shared between almost all limits.
    if g==0:
        for j,xr in enumerate((-.33,.13)):
            w=1/(1+np.exp((u-xr-.28)/.045))/(1+np.exp((-u+xr-.28)/.045));f=(mat-(mat*w).sum(1,keepdims=True)/w.sum())*w;f/=np.maximum(.00001,np.sqrt((f*f).sum(1,keepdims=True)))
            ab=starts[0][0] if j==0 else starts[0][6];end=starts[0][6] if j==0 else starts[0][8]
            b[:,ab:end]=.76*b[:,ab:end]+.24*product(f,CONSTANTS['digitparts'][j,ab:end].T)
    if g in (0,1,4):
        part=np.zeros_like(b)
        for j,w in enumerate(CONSTANTS['regions'+str(g)]):
            f=(mat-(mat*w).sum(1,keepdims=True)/w.sum())*np.sqrt(w);f/=np.maximum(.07,np.sqrt((f*f).sum(1,keepdims=True)));part+=product(f,CONSTANTS['local'+str(g)][j].T)
        b=.68*b+.16*part
    hog=product(innerhist(mat,g),CONSTANTS['hist'+str(g)].T)
    b=.77*b+.23*(hog*2-1);b-=CONSTANTS['cost'+str(g)]; corr=np.maximum.reduceat(b,starts[g],axis=1).reshape(n,len(shifts),-1)
    # Quantisation noise is local to the sign face: the brightness of the
    # surrounding sky need not predict the readability of a number.
    strength=std[:,:,0]/(noise[:,None]+.004)
    reliab=strength/(strength+.65);result=((corr-.23)*reliab[:,:,None]-shiftcost[None,:,None]).max(axis=1)
    if g==0:
        # Test blankness on a central disk, away from any painted border.
        mid=(u*u+v*v<.57**2);blankstd=(norm[:,4,0]/np.sqrt(a.shape[2]))*np.std(a[:,4,mid],axis=1)*np.sqrt(a.shape[2]);flat=blankstd/(noise+.004)
        blank=.40/(1+flat*flat*.30)-.06
        result=np.column_stack((result,blank))
    return result

def quant(a,q):
    n=a.shape[0];v=a.reshape(n,-1);index=int((v.shape[1]-1)*q);return np.partition(v,index,axis=1)[:,index,None,None]

def batch(images):
    r=images[:,0];g=images[:,1];b=images[:,2];n=len(r)
    raw=.27*r+.53*g+.20*b
    lumin=np.arcsinh(raw/.17);centre=lumin[:,5:27,5:27];lo=quant(centre,.09);hi=quant(centre,.91)
    gray=np.clip((lumin-lo)/(hi-lo+.054),-.42,1.38)
    dx=np.empty_like(gray);dy=np.empty_like(gray);dx[:,:,1:-1]=(gray[:,:,2:]-gray[:,:,:-2])*.5;dx[:,:,0]=0;dx[:,:,-1]=0;dy[:,1:-1,:]=(gray[:,2:,:]-gray[:,:-2,:])*.5;dy[:,0,:]=0;dy[:,-1,:]=0
    edge=np.minimum(.9,np.sqrt(dx*dx+dy*dy)*2.5)
    red=(r-.64*g-.36*b)/(r+.64*g+.36*b+.04)
    blue=(b-.58*r-.42*g)/(b+.58*r+.42*g+.04)
    yellow=(.65*r+.35*g-b)/(.65*r+.35*g+b+.045)-.63*np.maximum(red,0)
    # Relative chroma catches a red ring even with a colour cast. Absolute
    # chroma prevents turning arbitrary gray variations into saturated paint.
    def colour(t,level):
        base=quant(t[:,5:27,5:27],.18);return np.clip((t-.58*base-.025)/level,-.32,1.3)
    red=colour(red,.33);blue=colour(blue,.30);yellow=colour(yellow,.36)
    # Estimate a noise floor from diagonal high-pass differences. Using a
    # central region keeps the background separate from the sign material.
    hp=lumin[:,11:21,11:21]+lumin[:,10:20,10:20]-lumin[:,10:20,11:21]-lumin[:,11:21,10:20];noise=.40*quant(np.abs(hp),.4)[:,0,0]+.012/(.24+quant(raw[:,11:21,11:21],.6)[:,0,0]*4)
    arr=np.zeros((n,7),np.float32);ps=np.zeros((n,7,4),np.float32)
    for j in range(7):
        cm=red
        if j==3:cm=yellow
        if j==4:cm=blue
        if j==6:cm=np.maximum(0,np.maximum(red,blue))
        feat=np.concatenate([cm.reshape(n,-1),gray.reshape(n,-1),edge.reshape(n,-1)],axis=1)
        # Multiply a fixed number of rows each time, independent of batch size.
        val=feat@shapeweights[j].T-spatialprior
        best=val.argmax(1);ps[:,j]=refinepose(val,best,posebank[best]);arr[:,j]=val[np.arange(n),best]
    # Family hypotheses compete with each other after the internal symbol has
    # been read. Ambiguous borders are not a hard veto on an excellent symbol.
    score=np.full((n,43),-4,np.float32)
    score[:,12]=arr[:,3]+.32;score[:,13]=arr[:,2]+.32
    # Yield has a plain inner white field, not a symbol. Look for internal
    # high-frequency structure using points strictly inside its white triangle.
    u,v=np.meshgrid(np.linspace(-.6,.6,12,dtype=np.float32),np.linspace(-.69,.32,12,dtype=np.float32));m=v<.53-2*abs(u);u=u[m];v=v[m];p=ps[:,2];sx=p[:,0,None]+p[:,2,None]*u;sy=p[:,1,None]+p[:,3,None]*v
    edges=sample(edge,sx[:,None,:],sy[:,None,:])[:,0].mean(axis=1);score[:,13]-=.78*edges
    for j in icongroups:
        # Rule out only very poor outline/colour matches before expensive OCR.
        ii=np.where(arr[:,j]>arr.max(1)-.78)[0]
        if len(ii)==0:continue
        if j in (4,5): fg=np.arcsinh((.47*r[ii]+.5*g[ii]+.03*b[ii])/.17)
        else:fg=lumin[ii]
        if j==0 or j==1:fg=np.arcsinh((.24*r[ii]+.55*g[ii]+.21*b[ii])/.18)
        contrast=noise[ii]; labels=classes[j]
        if j==0:labels=np.append(labels,15)
        for begin in range(0,len(ii),8):
            sl=slice(begin,begin+8);idx=ii[sl]
            read=interior(j,fg[sl],ps[idx,j],contrast[sl])
            if j==0:
                words=independentdigits(fg[sl],ps[idx,j],contrast[sl]);whole=read[:,:8].copy();separate=.58*whole+.42*words
                # The component read is a second way to recognize the same
                # text, not a requirement that each tiny numeral be legible.
                read[:,:8]=np.maximum(separate,whole-.025)
            score[idx[:,None],labels]=arr[idx,j,None]+read
    colouredsymbols(images,ps,noise,score)
    return score.argmax(axis=1)

def predict(x):
    out=np.empty(len(x),np.int64)
    for i in range(0,len(x),32):
        # Identical GEMM panel sizes even for a singleton call.
        k=min(len(x)-i,32);a=np.zeros((32,3,32,32),np.float32);a[:k]=x[i:i+k];out[i:i+k]=batch(a)[:k]
    return out

def region(im,p,ux,uy,wx,wy):
    # Mean over a small elliptical area; support points are a deterministic
    # quadrature rule. There is no dependence on other inputs in the batch.
    t=np.array([0,.70,-.70,0,0,.50,.50,-.50,-.50],np.float32);s=np.array([0,0,0,.70,-.70,.50,-.50,.50,-.50],np.float32)
    x=p[:,0,None]+p[:,2,None]*(ux+wx*t);y=p[:,1,None]+p[:,3,None]*(uy+wy*s)
    return sample(im,x[:,None,:],y[:,None,:])[:,0].mean(1)

def colouredsymbols(images,ps,noise,score):
    r,g,b=images[:,0],images[:,1],images[:,2]
    # Paint hypotheses specific to the sign face: a red vehicle should be
    # visible on the left, even if its gray value matches the black vehicle.
    redness=np.log((r+.025)/(.66*g+.34*b+.025));p=ps[:,0]
    left=region(redness,p,-.33,.10,.16,.14);right=region(redness,p,.33,.10,.16,.14);paper=region(redness,p,0,-.36,.26,.10)
    # Log chroma can also change with brightness through the camera response.
    # Compare two objects of similar brightness as well as the local paper,
    # and require colour to be localized rather than spilling in from a rim.
    outside=region(redness,p,-.33,-.36,.15,.09);car=np.clip(np.minimum(left-right,left-.5*(outside+paper))/.28,-.4,1.15);score[:,9]+=.17*car;score[:,10]+=.17*car
    # Red, amber, green arranged vertically distinguish signals from a black
    # exclamation mark. Relative colour rejects a uniform yellow cast.
    p=ps[:,1];xg=np.log((g+.025)/(r+.025));bottom=region(xg,p,0,.57,.10,.10);top=region(xg,p,0,-.13,.09,.10);white=.5*(region(xg,p,-.26,.53,.09,.09)+region(xg,p,.26,.53,.09,.09));tgreen=bottom-white
    amber=np.log((.50*r+.50*g+.025)/(b+.025));gold=region(amber,p,0,.21,.10,.1)-.5*(region(amber,p,-.22,.24,.06,.07)+region(amber,p,.22,.24,.06,.07))
    chrom=np.clip(np.minimum(bottom-top,2*tgreen+.35*np.maximum(gold,0))/.32,0,1.3);score[:,26]+=.26*chrom-.12;score[:,18]+=.055*(1-np.minimum(chrom,1))

def refinepose(val,best,ps):
    # Sub-grid pose estimate from the local response surface. Keep widely
    # separated maxima (e.g. another sign above this one) out of the centroid.
    d=posebank[None,:,:]-ps[:,None,:];near=(abs(d).max(axis=2)<1.56);v=val-val[np.arange(len(val)),best,None]
    weight=near*np.exp(np.maximum(v*19,-40));return (weight@posebank)/weight.sum(axis=1,keepdims=True)

def independentdigits(fg,poses,noise):
    # Read a word as two or three independent glyphs on a common baseline.
    # This admits condensed type without mistaking narrow digit spacing for
    # thin strokes in a wide character. Intensities are normalized per glyph.
    n=len(fg);digits=(0,1,2,3,5,6,7,8);ts=(np.arange(16,dtype=np.float32)+.5)/16;u,v=np.meshgrid(ts*1.36-.68,ts*1.24-.62);u=u.ravel();v=v.ravel()
    ans=[]
    for length in (2,3):
        layout=np.array([(pitch,dx,dy,h) for pitch in ((.51,.61,.71) if length==2 else (.315,.39,.465,.54)) for dx in (-.075,0,.075) for dy in (-.065,.035) for h in (.77,.95,1.08)],np.float32);pitch,dx,dy,h=layout.T
        per=[];strengths=[]
        for j in range(length):
            px=dx+(j-(length-1)/2)*pitch;w=pitch*.83
            if length==3:px=px+(.12 if j==0 else -.04)*pitch
            a=sample(fg,poses[:,None,0:1]+poses[:,None,2:3]*(px[:,None]+w[:,None]*u),poses[:,None,1:2]+poses[:,None,3:4]*(dy[:,None]+h[:,None]*v));a=a.mean(axis=2,keepdims=True)-a
            for t in (u,v):a-=.55*(a*t).sum(axis=2,keepdims=True)/(t*t).sum()*t
            std=np.sqrt((a*a).mean(axis=2));a=np.clip(a,-2.8*std[:,:,None],2.8*std[:,:,None]);a-=a.mean(axis=2,keepdims=True);norm=np.sqrt((a*a).sum(axis=2,keepdims=True));a/=np.maximum(norm,.00001)
            z=product(a.reshape(-1,len(u)),CONSTANTS['ocr'].T).reshape(n,len(px),8,9).max(3);rel=std/(std+.7*noise[:,None]+.004);per.append((z-.25)*rel[:,:,None]);strengths.append(std)
        for number in (('20','30','50','60','70','80') if length==2 else ('100','120')):
            cs=np.stack([per[j][:,:,digits.index(int(ch))] for j,ch in enumerate(number)]);val=.70*cs.mean(0)+.30*cs.min(0)
            if length==3:
                # Beyond contrast alone, use the repeated shape in '100'.
                # Two clipped or defocused zeroes may still match each other
                # closely, even when neither matches an ideal unblurred zero.
                xx=poses[:,None,0:1]+poses[:,None,2:3]*((dx+pitch*.46)[:,None]+(pitch*.83)[:,None]*u);yy=poses[:,None,1:2]+poses[:,None,3:4]*(dy[:,None]+h[:,None]*v)
                off=poses[:,None,2:3]*pitch[:,None]*.5;left=sample(fg,xx-off,yy);right=sample(fg,xx+off,yy);left-=left.mean(2,keepdims=True);right-=right.mean(2,keepdims=True)
                repeat=(left*right).sum(2)/np.sqrt(np.maximum(1e-10,(left*left).sum(2)*(right*right).sum(2)));val+=.22*(repeat-.35)*(1 if number=='100' else -.35)
            # Large pose corrections cost a little, but are sometimes needed
            # with asymmetric glyph side bearings or a sign seen at an angle.
            val-=.09*(abs(dx)+abs(dy));ans.append(val.max(1))
    return np.stack(ans,1)
