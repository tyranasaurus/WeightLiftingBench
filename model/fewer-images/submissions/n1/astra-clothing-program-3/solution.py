import numpy as np

def sample_box(a, lo, hi, left, right, n):
    phase=(np.arange(n,dtype=np.float64)+.5)/n
    yy=lo-.5+phase*(hi-lo+1)
    xx=left-.5+phase*(right-left+1)
    yy=np.clip(yy,0,a.shape[0]-1); xx=np.clip(xx,0,a.shape[1]-1)
    iy=yy.astype(np.int32); ix=xx.astype(np.int32)
    jy=np.minimum(iy+1,a.shape[0]-1); jx=np.minimum(ix+1,a.shape[1]-1)
    fy=(yy-iy)[:,None]; fx=(xx-ix)[None,:]
    return (1-fy)*((1-fx)*a[iy[:,None],ix]+fx*a[iy[:,None],jx])+fy*((1-fx)*a[jy[:,None],ix]+fx*a[jy[:,None],jx])

def pool(a, g):
    return a.reshape(g,a.shape[0]//g,g,a.shape[1]//g).mean(axis=(1,3))

def features(a):
    # Occupancy, exterior silhouette ignoring internal holes, and four
    # ordered boundary functions. A single squared L2 distance combines
    # the three cues; coordinate weights implement a hand-specified metric.
    n=a.shape[0]
    inside=a>.42
    hull=np.maximum.accumulate(a,axis=1)*np.maximum.accumulate(a[:,::-1],axis=1)[:,::-1]
    vert=np.maximum.accumulate(a,axis=0)*np.maximum.accumulate(a[::-1],axis=0)[::-1]
    sides=[]
    for b in (inside,inside[:,::-1],inside.T,inside.T[:,::-1]):
        p=np.where(b.any(1),np.argmax(b,axis=1),n*.5)/n
        sides.append(p.reshape(12,-1).mean(1))
    bounds=np.concatenate(sides)
    occ=pool(a,12).ravel(); envelope=pool(hull*.63+vert*.37,8).ravel()
    density=np.concatenate([a.mean(axis=i).reshape(12,-1).mean(1) for i in (0,1)])
    return np.concatenate((occ*(.73/12),envelope*(.47/8),bounds*(1.22/np.sqrt(48)),density*(.72/np.sqrt(24))))

bank=CONSTANTS['shapes']
aspects=CONSTANTS['aspects']
square=(bank*bank).sum(2)

def saturate(x):
    return np.clip(x,0,1)

def ramp(x,l,h):
    return saturate((x-l)/(h-l))

def smooth(a):
    p=np.pad(a,1,mode='edge')
    return (p[1:-1,1:-1]*4+p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])/8

def measurements(img):
    # Per-photo foreground likelihood. The smoothed channel connects fine
    # fabric texture without closing the space below a handle or between legs.
    level=max(.012,np.max(img)*.18,np.quantile(img,.91)*.44)
    blurred=smooth(img)
    z=np.maximum(img*.70,blurred)
    presence=saturate((z-level*.047)/(level*.44+.008))
    ys,xs=np.nonzero((presence>.33)&((img>.011)|(blurred>level*.22)))
    if not len(ys):
        lo=0; hi=27; left=0; right=27
    else:
        lo=ys.min(); hi=ys.max(); left=xs.min(); right=xs.max()
        # Ignore isolated faint points in empty border rows/columns.
        row=presence.sum(1); col=presence.sum(0)
        while lo<hi and row[lo]<.90: lo+=1
        while lo<hi and row[hi]<.90: hi-=1
        while left<right and col[left]<.90: left+=1
        while left<right and col[right]<.90: right-=1
    aspect=(right-left+1)/(hi-lo+1)
    a=sample_box(presence,lo,hi,left,right,24)
    solid=saturate((a-.03)/.87)
    light=sample_box(img,lo,hi,left,right,24)
    # Orient laterally asymmetric objects toe first; also compare reflection
    # independently to the bank, which does not assume catalogue orientation.
    feat=np.stack((features(solid),features(solid[:,::-1])))
    bias=visual_cues(solid,light,aspect)
    return feat,np.log(aspect),bias

def visual_cues(m,img,ratio):
    # Additive evidence in log-likelihood units. Global shape is handled
    # separately by the template matcher, so these cues are mostly local.
    scores=np.zeros(10)
    body=m[6:22,7:17]
    midlight=img[5:22,6:18]
    fabric=max(.042,float(np.quantile(midlight,.69)))
    v=img/(fabric+.055)
    ref=smooth(v)
    idx=np.arange(24)
    # Silhouette symmetry, allowing half a pixel of misregistration.
    asym=(abs(m-m[:,::-1])).mean()
    symtex=abs(ref[:,3:21]-ref[:,20:2:-1]).mean()
    # A persistent central placket, zipper or opening is especially likely
    # on outerwear and woven shirts. Subtract the strongest nearby control
    # response: vertical stripes alone shouldn't count as a fastening.
    seam=0.; seamlit=0.
    for slope in [-1,0,1]:
        drift=np.rint(slope*(idx-12)/10).astype(np.int32)
        rows=idx[5:21]; off=drift[5:21]; strengths=[]; lights=[]
        for col in range(4,20):
            at=col+off; line=v[rows,at]*.68+v[rows,at+1]*.32
            fl=(v[rows,at-2]+v[rows,at+3])*.5
            diff=(line-fl)/(fl+.38)
            strengths.append(np.mean(np.maximum(-diff,0))*.7+np.maximum(0,-np.mean(diff))*.3)
            lights.append(np.mean(np.maximum(diff,0))*.7+np.maximum(0,np.mean(diff))*.3)
        neg=np.array(strengths); pos=np.array(lights)
        seam=max(seam,neg[5:10].max()-.70*np.median(neg))
        seamlit=max(seamlit,pos[5:10].max()-.70*np.median(pos))
    opening=max(ramp(seam,.048,.38),ramp(seamlit,.075,.48)*.82)
    # Shirt points / lapels: coherent pair of diagonal edges meeting near
    # the throat. Look for contrast across the edge, not a specific colour.
    diag=0.
    for depth in [5,8,11]:
        ys=np.arange(2,depth+1); dx=(depth-ys)*.44+1.
        for c in [11,12]:
            xl=np.rint(c-dx).astype(np.int32); xr=np.rint(c+dx).astype(np.int32)
            dl=ref[ys,xl+1]-ref[ys,xl-1]; dr=ref[ys,xr-1]-ref[ys,xr+1]
            diag=max(diag,abs(np.mean(dl+dr))*.5)
    lapel=ramp(diag,.105,.47)
    # Two-dimensional repeating edge structure favours woven shirts;
    # isolate the interior to avoid counting the side of a sleeve as plaid.
    crop=v[6:21,6:18]
    dx=np.abs(crop[:,2:]-crop[:,:-2]).mean()
    dy=np.abs(crop[2:]-crop[:-2]).mean()
    colstr=np.abs(np.diff(crop.mean(axis=0))).mean()
    rowstr=np.abs(np.diff(crop.mean(axis=1))).mean()
    check=ramp(min(dx,dy),.12,.42)*ramp(min(colstr,rowstr),.025,.105)
    # Ribbing: a fairly horizontal boundary at a close-fitting hem.
    edge=np.diff(ref[18:24,7:17],axis=0)
    band=ramp(np.max(np.abs(edge[:4].mean(1))),.14,.46)
    clothing=ramp(m[6:17,7:17].mean(),.52,.89)*ramp(1.58-ratio,0,.4)
    # Low contrast through the chest is typical of a plain knit. A shirt
    # has weaker shape cues, so use its collar and fabric as positive cues.
    scores[2]+=clothing*(.48-1.75*opening-.57*lapel-.37*check+.31*band)
    scores[4]+=clothing*(-.31+1.71*opening+.50*lapel-.52*check)
    scores[6]+=clothing*(-.54+1.02*opening+.72*lapel+1.32*check+.21*ramp(symtex,.17,.48))
    scores[0]+=clothing*(.25-.63*opening-.52*lapel-.42*check)
    scores[3]+=clothing*(.04-.32*opening)
    # Relative width at several heights, with holes filled only along each
    # scanline. This distinguishes a flared skirt from long dangling sleeves.
    hard=m>.40
    span=(np.maximum.accumulate(hard,axis=1)&np.maximum.accumulate(hard[:,::-1],axis=1)[:,::-1]).mean(1)
    skirt=span[18:22].mean()-span[8:13].mean()
    flare=ramp(skirt,.065,.32)*ramp(m[16:22,9:15].mean(),.55,.95)
    scores[3]+=1.95*flare
    scores[4]-=.62*flare; scores[2]-=.79*flare; scores[6]-=.64*flare; scores[0]-=.55*flare
    # Cuff gaps below a short sleeve, person-sized pant legs separated at
    # the inseam, single body suspended below a bag handle.
    widthdrop=span[3:8].max()-span[12:18].mean()
    sleeves=ramp(widthdrop,.096,.32)*ramp(span[3:8].max(),.69,.95)
    scores[0]+=.87*sleeves; scores[2]-=.79*sleeves; scores[4]-=.79*sleeves
    centerlower=m[14:23,10:14].mean(); legs=(m[14:23,5:9].mean()+m[14:23,15:19].mean())*.5
    divide=ramp(legs-centerlower,.13,.61)*ramp(m[2:8,9:15].mean(),.39,.84)
    scores[1]+=2.15*divide+.64*ramp(.61-ratio,0,.23)
    scores[3]-=1.05*divide
    roof=m[:9,8:16]; handbag=ramp(m[13:21,6:18].mean()-roof.mean(),.42,.78)
    scores[8]+=1.54*handbag
    # Footwear: toe-to-heel imbalance, a heel/arch cutout, and openings
    # in the upper. A sneaker has a continuous vamp even on a dark shoe.
    feet=m if m[:11,3:10].sum()<=m[:11,14:21].sum() else m[:,::-1]
    fh=feet>.34
    top=np.where(fh.any(0),fh.argmax(0),24)
    bot=np.where(fh.any(0),24-fh[::-1].argmax(0),0)
    between=(idx[:,None]>=top)&(idx[:,None]<bot)
    foot=feet[:,2:22]; ins=between[:,2:22]
    holes=((1-foot)*ins).sum()/(ins.sum()+.01)
    holecue=ramp(holes,.096,.36)
    scores[5]+=2.75*holecue; scores[7]-=1.15*holecue; scores[9]-=.68*holecue
    ankle=(top[3:8].mean()-top[16:20].mean())/24
    step=ramp(ankle,.32,.70)*ramp(1.95-ratio,0,.57)
    scores[9]+=.97*step
    # Deep arch with a heel and thin toe strap tends to be an open shoe.
    arch=(bot[3:8].mean()*.45+bot[18:22].mean()*.55-bot[12:17].mean())/24
    archcue=ramp(arch,.062,.245)
    scores[5]+=.76*archcue; scores[7]-=.96*archcue
    flat=ramp(ratio,1.68,2.6)
    scores[7]+=.45*flat*(1-holecue)-.40*archcue
    scores[9]-=.72*flat
    # Symmetric lower-body shapes and side-profile footwear are rarely
    # confused when shape is clean; use that as weak independent evidence.
    sw=ramp(asym,.13,.39)
    for k in (0,1,2,3,4,6): scores[k]-=1.35*sw
    for k in (5,7,9): scores[k]+=.57*sw
    return scores

def compare(feat,la,bias):
    # Independent squared distances. Include uncertainty in the box aspect
    # ratio, never stretch a trouser template freely into a shoe. Use a few
    # nearby designs to reduce sensitivity to a single accidental match.
    # The contraction runs over feature coordinates only, in a fixed order.
    out=np.empty(10)
    qnorm=(feat*feat).sum(1)
    for k in range(10):
        d0=square[k]+qnorm[0]-2*(bank[k]@feat[0])
        d1=square[k]+qnorm[1]-2*(bank[k]@feat[1])
        d=np.minimum(d0,d1)
        d+=.41*(la-aspects[k])**2
        nearest=np.partition(d,9)[:10]
        out[k]=-99*(.48*nearest.min()+.52*nearest.mean())+bias[k]
    return np.argmax(out)

def predict(x):
    ans=np.empty(len(x),dtype=np.int64)
    for i in range(len(x)):
        f,a,b=measurements(x[i,0]); ans[i]=compare(f,a,b)
    return ans
