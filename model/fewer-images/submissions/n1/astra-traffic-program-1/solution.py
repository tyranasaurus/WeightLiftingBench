import numpy as np

classes=((0,1,2,3,4,5,7,8,9,10,15,16),(11,18,19,20,21,22,23,24,25,26,27,28,29,30,31),(33,34,35,36,37,38,39,40),(14,17),(6,32,41,42))

def unit(a):
    return np.clip(a,0,1)

def blur(p):
    q=np.pad(p,((0,0),(1,1),(1,1)),mode='edge')
    return (4*q[:,1:-1,1:-1]+2*(q[:,:-2,1:-1]+q[:,2:,1:-1]+q[:,1:-1,:-2]+q[:,1:-1,2:])+q[:,:-2,:-2]+q[:,2:,:-2]+q[:,:-2,2:]+q[:,2:,2:])*.0625

def contrast(v,region,percent=10):
    sub=v[:,region,region].reshape(len(v),-1)
    vals=np.percentile(sub,[percent,50,100-percent],axis=1).astype(np.float32)[:,:,None,None]
    return np.clip((v-vals[1])/(vals[2]-vals[0]+.12),-1.1,1.1)

def maps(pic):
    r,g,b=pic[:,0],pic[:,1],pic[:,2]
    lr,lg,lb=[np.log(t+.007) for t in (r,g,b)]
    # Log ratios discount exposure changes. A grey-green channel preserves
    # contrast between a red rim and the retroreflective white face in shadow.
    R=np.clip(np.minimum(lr-lg,(lr-lb)*.85+.10),-.9,1.8)
    B=np.clip(np.minimum(lb-lr,(lb-lg)*1.7+.12),-1.1,1.7)
    Y=np.clip(np.minimum((lr-lb)*.78,(lg-lb)), -1,1.6)-.55*np.maximum(0,lr-lg-.34)
    light=np.log(.19*r+.61*g+.20*b+.006)
    green=np.log(.05*r+.75*g+.20*b+.006)
    L=contrast(green,slice(4,28),9); L2=contrast(light,slice(3,29),10)
    ri=.67*contrast(R,slice(4,28))+.40*np.clip(R/.8,-1,1)
    bi=.61*contrast(B,slice(4,28))+.44*np.clip(B/.8,-1,1)
    yi=.55*contrast(Y,slice(4,28))+.50*np.clip(Y/.7,-1,1)
    # Return two densities for each border: a colour/brightness likelihood
    # for the material, and evidence for whether this is the right colour.
    return [(ri-.57*L,R),(ri-.57*L,R),(ri-.58*L,R),(bi-.55*L,B),(ri-.49*L,R),(yi-.09*L,Y),(L2-.19*np.maximum(0,ri),-.75*np.maximum(R-.10,0)-.38*np.maximum(B-.13,0))],(light,green,R,B,Y)

def perimeter(shape):
    if shape in (0,3,4,6):
        t=(np.arange(48)+.5)*np.pi/24
        return np.cos(t),np.sin(t),0.
    if shape in (1,2):
        # Three corner fillets of a triangle; outer extents are measured at
        # the rounded vertices. Its homothetic centre is below its box centre.
        v=np.array([[0,-1.10],[1.07,1],[-1.07,1]],dtype=float);fillet=.083;cy=.30
    else:v=np.array([[0,-1],[1,0],[0,1],[-1,0]],dtype=float);fillet=.035;cy=0
    pts=[]
    for i,p in enumerate(v):
        q=v[(i+1)%len(v)];prev=v[i-1];u=(1-fillet)*p+fillet*prev;w=(1-fillet)*p+fillet*q
        t=(np.arange(3)[:,None]+.5)/3;pts.extend((1-t)**2*u+2*t*(1-t)*p+t*t*w)
        t=(np.arange(13)[:,None]+.5)/13;pts.extend(w*(1-t)+t*((1-fillet)*q+fillet*p))
    x,y=np.array(pts).T
    if shape==2:y=-y;cy=-cy
    return x,y,cy

def profiles(shape):
    if shape==0:return ([.84,.94],[.63,.73,1.09],[.39,.41,.20])
    if shape in (1,2):return ([.79,.94],[.52,.62,1.12],[.34,.44,.22])
    if shape in (3,4):return ([.69,.86],[1.03,1.12],[.66,.34])
    if shape==5:return ([.55,.70],[.85,.97,1.13],[.45,.46,.09])
    return ([.84,.94],[1.06,1.18],[.67,.33])

def gridposes():
    c=(np.arange(3)-1)*2.15+15.5; rad=np.array([9.1,10.9,12.8,14.6]);cx,cy,rx,ry=np.meshgrid(c,c,rad,rad,indexing='ij'); keep=abs(rx-ry)<3.85
    return np.stack((cx[keep],cy[keep],rx[keep],ry[keep],0*rx[keep],0*ry[keep]),1).astype(np.float32)

poses=gridposes()

def outlinebanks(shape):
    v,w,cy=perimeter(shape);plus,minus,ww=profiles(shape)
    p=poses.astype(float);arr=[]
    for points,weights in ((plus+minus,[1/len(plus)]*len(plus)+[-a for a in ww]),(plus,[1/len(plus)]*len(plus))):
        hist=np.zeros((len(p),1024))
        for scale,weight in zip(points,weights):
            x=np.clip(p[:,0:1]+p[:,2:3]*v*scale,0,30.9999);y=np.clip(p[:,1:2]+p[:,3:4]*(cy+(w-cy)*scale),0,30.9999);ix=x.astype(int);iy=y.astype(int);dx=x-ix;dy=y-iy
            for j,coef in [(0,(1-dx)*(1-dy)),(1,dx*(1-dy)),(32,(1-dx)*dy),(33,dx*dy)]:np.add.at(hist,(np.arange(len(p))[:,None],iy*32+ix+j),coef*(weight/len(v)))
        arr.append(np.ascontiguousarray(hist.T,dtype=np.float32))
    return arr

outlines=[outlinebanks(i) for i in range(7)]
linepts=[perimeter(i) for i in range(7)]
strength=[.30,.30,.30,.43,.39,.48,.85]

def interp(pix,x,y):
    n=len(pix);xx=np.clip(x,0,30.99999);yy=np.clip(y,0,30.99999);ix=xx.astype(np.int32);iy=yy.astype(np.int32);a=xx-ix.astype(np.float32);b=yy-iy.astype(np.float32)
    index=ix+32*iy+np.arange(n,dtype=np.int32).reshape((n,)+(1,)*(ix.ndim-1))*1024;img=pix.reshape(-1)
    return (img[index]*(1-a)+img[index+1]*a)*(1-b)+(img[index+32]*(1-a)+img[index+33]*a)*b

def posecost(p):
    return .010*((p[...,0]-15.5)**2+(p[...,1]-15.5)**2)/4+.009*((p[...,2]-12.2)**2+(p[...,3]-12.2)**2)/6+.015*((p[...,2]-p[...,3])/3)**2+.15*(p[...,4]**2+p[...,5]**2)

def trace(m,p,sh):
    # Sample finer poses directly. Use the same signed profile as the banks.
    vx,vy,cy=linepts[sh];plus,minus,ww=profiles(sh);x0=p[:,:,0,None];y0=p[:,:,1,None];rx=p[:,:,2,None];ry=p[:,:,3,None];kx=p[:,:,4,None];ky=p[:,:,5,None]
    sc=-posecost(p)
    for scale,weight,positive in zip(plus+minus,[1/len(plus)]*len(plus)+[-a for a in ww],[True]*len(plus)+[False]*len(minus)):
        u=(scale*vx).astype(np.float32);v=(cy+(vy-cy)*scale).astype(np.float32);xx=x0+rx*(u+kx*v);yy=y0+ry*(v+ky*u)
        if positive:q=m[2]
        else:q=m[0]
        sc+=weight*interp(q,xx,yy).mean(axis=-1)
    return sc

def cancellation(m,col,p,pref):
    n=len(p);u,v,cy=linepts[6];positions=np.broadcast_to(poses,(n,len(poses),6));u=u.astype('f');v=v.astype('f')
    def evidence(positions):
        def samp(img,u,v):
            return interp(img,positions[:,:,0,None]+positions[:,:,2,None]*(u+positions[:,:,4,None]*v),positions[:,:,1,None]+positions[:,:,3,None]*(v+positions[:,:,5,None]*u))
        inner=samp(m,.88*u,.88*v);outer=samp(m,1.095*u,1.095*v);edge=inner-outer
        q=np.linspace(-.53,.53,15,dtype='f');ink=samp(m,q,-q);a=samp(m,q+.19,-q+.19);b=samp(m,q-.19,-q-.19);stripe=(a+b)*.5-ink
        return .52*np.mean(np.minimum(abs(edge),1),-1)+.20*np.abs(np.mean(edge,-1))+.75*stripe.mean(-1)+.51*samp(col,.9*u,.9*v).mean(-1)-posecost(positions)
    s=evidence(positions);v0=evidence(p[:,None,:])[:,0];j=s.argmax(1);p=np.where((s[np.arange(n),j]>v0)[:,None],poses[j],p);pref=np.maximum(s[np.arange(n),j],v0)
    for d in [.85,.44]:
        delta=np.eye(6,dtype='f')*np.array([d,d,d,d,d*.10,d*.10],dtype='f');c=p[:,None,:]+np.concatenate([delta,-delta]);s=evidence(c);j=s.argmax(1);score=s[np.arange(n),j];p=np.where((score>pref)[:,None],c[np.arange(n),j],p);pref=np.maximum(score,pref)
    return p,pref

def localize(ms):
    n=len(ms[0][0]);allp=[];allsc=[]
    for sh,(m,col) in enumerate(ms):
        h,hc=outlines[sh];s=m.reshape(n,-1)@h+strength[sh]*(col.reshape(n,-1)@hc)-posecost(poses)
        j=s.argmax(1);p=poses[j].copy();pref=s[np.arange(n),j]
        mat=(m,col,m+col*strength[sh]);eye=np.eye(6,dtype=np.float32)
        for step in range(3):
            if step==0:delta=eye*np.array([1.14,1.14,1.0,1.0,.119,.095],dtype='f')
            elif step==1:delta=eye*np.array([.64,.64,.63,.63,.071,.059],dtype='f')
            else:delta=eye*np.array([.35,.35,.37,.37,.043,.039],dtype='f')
            # Parallel one-axis probes, then try their combined displacement.
            opts=p[:,None,:]+np.concatenate([delta,-delta],axis=0);ss=trace(mat,opts,sh);j=ss.argmax(1);rows=np.arange(n);choice=opts[rows,j];best=ss[rows,j]
            sign=(ss[:,:6]>ss[:,6:]).astype('f')*2-1;shift=sign*(np.maximum(ss[:,:6],ss[:,6:])>pref[:,None])*np.diag(delta)
            combined=p+shift;sc=trace(mat,combined[:,None,:],sh)[:,0];keep=sc>best;choice=np.where(keep[:,None],combined,choice);best=np.maximum(best,sc);p=np.where((best>pref)[:,None],choice,p);pref=np.maximum(pref,best)
        if sh==6:
            p,pref=cancellation(m,col,p,pref)
        allp.append(p);allsc.append(pref)
    return allp,allsc

# Multiple samples of a rectified sign permit subpixel translation and a
# small residual roll without quantising the numbers or arrowheads.
shifts=np.array([(u,v,r) for u in [-.085,0,.085] for v in [-.085,0,.085] for r in [0]]+[(u,v,r) for r in [-.115,.115] for u,v in [(0,0),(-.055,0),(.055,0),(0,-.055),(0,.055)]],dtype='f')

def coords(p,u,v):
    x=p[:,0,None,None]+p[:,2,None,None]*(u+p[:,4,None,None]*v);y=p[:,1,None,None]+p[:,3,None,None]*(v+p[:,5,None,None]*u);return x,y

def readglyphs(im,p,rect):
    ax=CONSTANTS['gx'];ay=CONSTANTS['gy'];c=CONSTANTS['ga'];base=CONSTANTS['gb'];T=CONSTANTS['glyph'];n=len(p)
    cx,cy,ww,hh=[r.reshape(-1,1) for r in rect]
    u=cx+ax*ww;v=cy+ay*hh;z=interp(im,*coords(p,u,v));z=(z-((z*c)@base.T)@base)*np.sqrt(c);norm=np.sqrt((z*z).sum(2,keepdims=True)+.00004);z=z/norm
    return (z.reshape(-1,len(ax))@T).reshape(n,len(u),8,8).max(3)

def ocr(im,p):
    n=len(p);scores=np.zeros((n,8),dtype='f')
    for k in range(2):
        widths=[.95,1.10,1.25,1.38] if k==0 else [1.05,1.19,1.34,1.44];heights=[.67,.82,.97] if k==0 else [.56,.71,.84]
        args=np.meshgrid(np.array(widths,'f'),np.array(heights,'f'),np.array([-.070,0,.070],'f'),np.array([-.08,0,.08],'f'),np.array([.14,.27],'f'),indexing='ij')
        w,h,cx,cy,gap=[a.ravel() for a in args];d=w/((2+gap) if k==0 else (2.61+2*gap));x0=cx-w/2
        if k==0:
            v=readglyphs(im,p,(x0+.5*d,cy,d,h))
            zero=readglyphs(im,p,(cx+w/2-.5*d,cy,d,h))[:,:,0]
            scores[:,:6]=(.61*v[:,:,2:]+.39*zero[:,:,None]-.10*(np.abs(cx)+np.abs(cy))[None,:,None]).max(1)
        else:
            one=readglyphs(im,p,(x0+.305*d,cy,.61*d,h))[:,:,1];mid=readglyphs(im,p,(x0+(1.11+gap)*d,cy,d,h));zero=readglyphs(im,p,(cx+w/2-.5*d,cy,d,h))[:,:,0]
            scores[:,6:]=(.28*one[:,:,None]+.43*mid[:,:,[0,2]]+.29*zero[:,:,None]-.10*(np.abs(cx)+np.abs(cy))[None,:,None]).max(1)
    return scores

def match(features,p,g):
    light,green,R,B,Y=features;st=str(g);ax,ay,a,basis,T,cost,starts=[CONSTANTS[s+st] for s in ('x','y','a','b','t','cost','offset')]
    u=ax+shifts[:,0,None]-shifts[:,2,None]*ay;v=ay+shifts[:,1,None]+shifts[:,2,None]*ax
    xx,yy=coords(p,u,v);w=np.sqrt(a);n=len(p)
    if g==2:im=light-.55*B
    elif g==3:im=green-.32*R
    elif g==0:im=-np.log(np.exp(light)+.03)+.07*R
    elif g==1:im=-light
    else:im=-light
    im=.84*im+.16*blur(im);data=interp(im,xx,yy)
    # Remove a plane (constant and two shading gradients) using a fixed,
    # analytic aperture basis. Clip extremes to resist dirt and occlusion.
    trend=((data*a)@basis.T)@basis;z=data-trend
    power=np.sqrt((z*z*a).sum(-1,keepdims=True)+.00003);z=np.clip(z/power,-2.9,2.9)
    z=(z-((z*a)@basis.T)@basis)*w;z=z/np.sqrt((z*z).sum(-1,keepdims=True)+.001)
    sc=(z.reshape(-1,len(a))@T).reshape(n,len(shifts),-1);detail=z.reshape(-1,len(a))@CONSTANTS['f'+st].T;detail/=np.sqrt((detail*detail).sum(1,keepdims=True)+.001)
    high=(detail@CONSTANTS['detail'+st]).reshape(n,len(shifts),-1);sc=.62*sc+.38*high;sc-=cost;sc-=.12*np.abs(shifts[:,:2]).sum(1)[None,:,None]
    # Directed distance to an ink stroke, with a smooth saturation. This
    # score permits small displacements of individual parts of a symbol
    # (e.g. the rider's limbs) without moving the whole symbol.
    if g==1:
        prob=unit((data-trend)/power*.54+.18)
        # Work on the best rigid shift per symbol family afterwards; the
        # distance score is evaluated on the same shifts as correlation.
        tang=CONSTANTS['neighbors'];near=prob[:,:,tang[:,0]]
        for j in range(1,tang.shape[1]):
            near=np.maximum(near,prob[:,:,tang[:,j]]*CONSTANTS['decay'][:,j])
        soft=(near.reshape(-1,len(a))@CONSTANTS['foreground']).reshape(n,len(shifts),-1)
        recur=((prob*a).reshape(-1,len(a))@CONSTANTS['distance']).reshape(n,len(shifts),-1)/np.maximum(.013,(prob*a).sum(2,keepdims=True))
        cover=(soft+recur)*.5
        # Coverage near random ink has a nonzero baseline; demand coverage
        # in both directions, especially when the image itself is quite dark.
        sc=.71*sc+.29*(cover*1.65-.62)
    vals=np.maximum.reduceat(sc.max(1),starts,axis=1)
    if g==0:
        glyphs=ocr(im,p);vals[:,:8]=.67*vals[:,:8]+.33*glyphs
        # A blank regulation disc is a null hypothesis, not a highly
        # correlated symbol. Test the central area against an unprinted
        # annulus, which shares its lighting and imaging noise. Detect ink
        # at a finer scale as well, in case there is a shadow across the face.
        u,v=np.meshgrid(np.linspace(-.72,.72,19,dtype='f'),np.linspace(-.72,.72,19,dtype='f'));rad=np.hypot(u,v);core=rad<.57;surround=(rad>.60)&(rad<.73)&(abs(v)>.42)
        clean=-light;allpix=interp(clean,*coords(p,u,v));refs=allpix[:,surround];inside=allpix[:,core]
        q=np.percentile(refs,[20,50,80],axis=1);iq=np.percentile(inside,[48,88],axis=1);margin=.022+.5*(q[2]-q[0]);ink=np.maximum(iq[1]-np.minimum(q[1]+.15*(q[2]-q[0]),iq[0]),0)/(margin+.02)
        detailmap=clean-blur(blur(clean));hp=interp(detailmap,*coords(p,u,v));fine=np.percentile(hp[:,core],89,axis=1);base=np.percentile(abs(hp[:,surround]),60,axis=1)
        texture=np.maximum(0,fine-1.6*base)/(.011+base);disorder=np.maximum(ink*.78,texture*.71)
        vals[:,10]=.82-.55*disorder/(.72+disorder)
    if g==3:
        # A word leaves gaps between its vertical stems; a white no-entry bar does not.
        ux=np.linspace(-.63,.63,25,dtype='f');vy=np.array([-.075,0,.075],dtype='f');u,v=np.meshgrid(ux,vy);prof=interp(im,*coords(p,u,v)).mean(1)
        trend=(prof[:,:1]+(prof[:,-1:]-prof[:,:1])*np.linspace(0,1,25,dtype='f'));res=prof-trend;weave=((res[:,4:]-res[:,:-4])**2).mean(1)**.5
        up=interp(im,*coords(p,ux[None,:],np.full((1,25),-.38,dtype='f'))).mean((1,2));down=interp(im,*coords(p,ux[None,:],np.full((1,25),.38,dtype='f'))).mean((1,2));height=prof.mean(1)-(up+down)*.5
        uneven=weave/(.09+np.maximum(height,.07));vals[:,1]-=.34*np.clip(uneven-.16,0,1.1);vals[:,0]+=.12*np.clip(uneven-.20,0,1)
    return vals

def meanface(a,p,scale=1.):
    q=(np.arange(12,dtype='f')+.5)/6-1;u,v=np.meshgrid(q,q);keep=u*u+v*v<1
    return interp(a,*coords(p,(u[keep]*scale)[None,:],(v[keep]*scale)[None,:]))[:,0].mean(1)

def points(sh,kind):
    u=(np.arange(30,dtype='f')+.5)/15-1;x,y=np.meshgrid(u,u)
    if sh in (1,2):
        h=y if sh==1 else -y
        # Barycentric shrinkage relative to triangular centroid.
        z=np.maximum((h-.3)/.70,(abs(x)*2.10-h+.30)/1.40)
    elif sh==5:z=np.abs(x)+np.abs(y)
    else:z=np.sqrt(x*x+y*y)
    mask=(z<(.57 if sh in (1,2) else .65)) if kind==0 else ((z>.81)&(z<.96))
    return x[mask][None,:],y[mask][None,:]

facecoords=[points(k,0) for k in range(7)]

def structural(ms,features,pp,ss):
    light,green,R,B,Y=features;out=[]
    for sh,p in enumerate(pp):
        m,col=ms[sh];vx,vy,cy=linepts[sh];plus,minus,ww=profiles(sh)
        def at(image,sc):
            tx=(sc*vx).astype('f')[None,:];ty=(cy+sc*(vy-cy)).astype('f')[None,:]
            return interp(image,*coords(p,tx,ty))[:,0]
        # Integrate support around the perimeter. An isolated piece of
        # background with extreme saturation cannot substantiate a whole rim.
        diff=(at(m,plus[0])+at(m,plus[1]))*.5
        for radius,weight in zip(minus,ww):diff-=weight*at(m,radius)
        med=np.median(diff,axis=1);coherent=np.mean(np.clip(diff,-.65,1.25),axis=1)
        colrim=at(col,sum(plus)/2);rim=colrim.mean(1);score=.22*med+.28*coherent+.5*(ss[sh]-strength[sh]*rim)-.035*posecost(p)
        cf=coords(p,*facecoords[sh]);coreR,coreB,coreY,faceG=[interp(a,*cf)[:,0] for a in (R,B,Y,green)];face=faceG.mean(1);shade=np.sqrt(((faceG-face[:,None])**2).mean(1));rimG=at(green,sum(plus)/2).mean(1)
        if sh<=2:
            # White is usually brighter than red in green light. Use the
            # upper half of the face to avoid counting black printing as white.
            white=np.percentile(faceG,69,axis=1);contrast=white-rimG
            chrom=(np.median(colrim,axis=1)-np.median(coreR,axis=1))
            score+=.30*np.clip(chrom,-.9,1.1)+.22*np.clip(contrast,-.85,1.25)+.10*np.clip(rim,-.65,1)
            if sh==2:score+=.14-.47*np.clip(shade/(.08+np.maximum(.06,contrast)),0,1.5)
        elif sh in (3,4):
            ct=coreB if sh==3 else coreR; body=np.percentile(ct,64,axis=1);bad=np.percentile(coreR if sh==3 else coreB,64,axis=1)
            score+=.44*np.clip(body,-.65,1.15)+.15*np.clip(rim,-.65,1.1)-.23*np.clip(bad,0,1)
        elif sh==5:
            # A priority sign specifically has a white band outside its yellow
            # centre. Uniform yellow light on a circle cannot supply this edge.
            inner=np.median(coreY,axis=1);outerY=at(Y,.89);delta=inner-np.median(outerY,axis=1)
            lumdiff=np.median(at(light,.88),axis=1)-np.median(interp(light,*cf)[:,0],axis=1)
            score+=.36*np.clip(delta,-1,1.2)+.20*np.clip(inner,-.7,1)+.27*np.clip(lumdiff,-1,1)-.17*np.clip(shade,0,1)
        else:
            # Cancellation signs are permitted a colour cast, but lack a
            # chromatic rim relative to the face and have no vivid colours.
            score=.86*ss[sh];sat=np.maximum(np.mean(abs(at(R,.90)-np.median(coreR,axis=1)[:,None]),axis=1),np.mean(abs(at(B,.90)-np.median(coreB,axis=1)[:,None]),axis=1))
            score-=.37*np.minimum(sat,1.2)+.12*np.minimum(np.maximum(np.abs(coreR).mean(1),np.abs(coreB).mean(1)),1.)
        out.append(score)
    return np.stack(out,1).astype('f')

def batch(x):
    ms,feat=maps(x);pp,ss=localize(ms);n=len(x);sf=structural(ms,feat,pp,ss)
    # A broad uncoloured disc does not in itself identify a cancellation:
    # require matching the diagonal stripe. Use presence of colour rather
    # than assuming a purely achromatic illuminant for those signs.
    sf[:,6]-=.06;sf[:,4]-=.04
    groups=(0,1,3,4,6);out=np.full((n,43),-12,dtype=np.float32)
    out[:,12]=sf[:,5];out[:,13]=sf[:,2]
    for gr,sh in enumerate(groups):
        use=sf[:,sh]>=sf.max(1)-1.35
        if not np.any(use):continue
        values=match(feat,pp[sh],gr)[use];mult=(1.03,.89,1.10,.98,1.26)[gr];adj=(.47,.44,.47,.46,.44)[gr];scores=sf[use,sh,None]+mult*(values-adj)
        out[np.ix_(use,np.array(classes[gr]))]=scores
    return out.argmax(1)

def predict(x):
    labels=np.empty(len(x),dtype=np.int64)
    # A fixed working width also makes the BLAS arithmetic independent of
    # the caller's batch size. The final partial block is padded, not mixed.
    for i in range(0,len(x),16):
        a=x[i:i+16];n=len(a)
        if n!=16:a=np.concatenate((a,np.broadcast_to(a[-1:],(16-n,3,32,32))))
        labels[i:i+n]=batch(a)[:n]
    return labels
