import numpy as np

def clip(x):
    return np.clip(x,0,1)


def ramp(x,low,high):
    return clip((x-low)/(high-low))


def smooth(a):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    return (4*a+2*(p[:,1:-1,:-2]+p[:,1:-1,2:]+p[:,:-2,1:-1]+p[:,2:,1:-1])+p[:,:-2,:-2]+p[:,:-2,2:]+p[:,2:,:-2]+p[:,2:,2:])/16


def bounds(m):
    px=m.sum(1); py=m.sum(2)
    cols=px>np.maximum(.48,px.max(1,keepdims=True)*.043)
    rows=py>np.maximum(.48,py.max(1,keepdims=True)*.043)
    x0=cols.argmax(1); x1=27-cols[:,::-1].argmax(1)
    y0=rows.argmax(1); y1=27-rows[:,::-1].argmax(1)
    return x0,x1,y0,y1


def sample(a,cx,cy,w,h):
    grid=(np.arange(28,dtype=np.float32)-13.5)/28
    xx=cx[:,None]+w[:,None]*grid
    yy=cy[:,None]+h[:,None]*grid
    return lookup(a,xx,yy)


def lookup(a,xx,yy):
    xx=np.clip(xx+1,0,28.9999); yy=np.clip(yy+1,0,28.9999)
    ix=xx.astype(np.int32); iy=yy.astype(np.int32)
    rx=(xx-ix)[:,None,:].astype(np.float32); ry=(yy-iy)[:,:,None].astype(np.float32)
    p=np.pad(a,((0,0),(1,1),(1,1)))
    n=np.arange(len(a))[:,None,None]
    i=iy[:,:,None]; j=ix[:,None,:]
    return (1-ry)*((1-rx)*p[n,i,j]+rx*p[n,i,j+1])+ry*((1-rx)*p[n,i+1,j]+rx*p[n,i+1,j+1])


def unit(a):
    v=a.reshape(len(a),-1)
    v=v-v.mean(1,keepdims=True)
    return v/np.maximum(.5,np.sqrt((v*v).sum(1,keepdims=True)))


def descriptor(m):
    p=np.pad(m,((0,0),(1,1),(1,1)))
    s=(4*m+2*(p[:,1:-1,:-2]+p[:,1:-1,2:]+p[:,:-2,1:-1]+p[:,2:,1:-1])+p[:,:-2,:-2]+p[:,:-2,2:]+p[:,2:,:-2]+p[:,2:,2:])/16
    cells=s.reshape(len(m),14,2,14,2).mean(axis=(2,4))
    proj=np.concatenate([m.mean(1),m.mean(2)],axis=1)
    return np.concatenate([unit(cells),unit(proj)],axis=1)


def match(views):
    # Use a fixed GEMM row shape even for a singleton so roundoff does not
    # depend on which other items have been supplied by the caller.
    n=len(views[0]); nv=len(views)
    v=np.zeros((24*nv,252),np.float32)
    for j,m in enumerate(views):
        v[24*j:24*j+n]=descriptor(m)
    sims=v @ CONSTANTS['patterns']
    sims-=CONSTANTS['costs']
    out=np.zeros((n,10),np.float32); pos=0
    for k,size in enumerate(CONSTANTS['lengths']):
        s=sims[:,pos:pos+size].max(1).reshape(nv,24)[:,:n]
        # Only mirror the query, never infer a label from other batch members.
        main=np.maximum(s[0],s[1]); alt=np.maximum(s[2],s[3])
        out[:,k]=.79*np.maximum(main,alt-.018)+.21*np.minimum(main,alt)
        pos+=size
    return out


def extent(b):
    live=b.any(axis=2)
    l=b.argmax(axis=2); r=27-b[:,:,::-1].argmax(axis=2)
    return l,r,(r-l+1)*live


def regionmean(a,b):
    return (a*b).sum(axis=(1,2))/np.maximum(1,b.sum(axis=(1,2)))


def appearance(scores, gray, mask, w, h):
    # Here the axes are fractional bounding-box coordinates, independent of
    # item size. Measurements refer to this query alone.
    b=mask>.43
    l,r,span=extent(b)
    rowwidth=span/28
    asp=w/h
    col=np.arange(28)[None,None,:]
    row=np.arange(28)[None,:,None]
    sym=1-np.abs(mask-mask[:,:,::-1]).sum((1,2))/(mask.sum((1,2))+2)
    asym=ramp(1-sym,.16,.62)
    wide=ramp(asp,1.08,1.80); tall=ramp(1/asp,1.05,1.9)
    # Thin handles, straps and gaps: distinguish a convex hull from its contents.
    hull=(col>=l[:,:,None])&(col<=r[:,:,None])&(span[:,:,None]>0)
    top=b.argmax(axis=1); bot=27-b[:,::-1].argmax(axis=1)
    vhull=(row>=top[:,None,:])&(row<=bot[:,None,:])&b.any(1)[:,None,:]
    holes=hull&vhull
    void=regionmean(1-mask,holes)
    below=regionmean(1-mask,holes&(row>=11))
    shapefill=mask.mean(axis=(1,2))
    upper=mask[:,2:12,8:20].mean((1,2)); core=mask[:,13:22,8:20].mean((1,2))
    corebottom=mask[:,20:26,10:18].mean((1,2))
    mid=mask[:,13:20,11:17].mean((1,2))
    gap=1-mask[:,17:26,12:16].mean((1,2))
    legs=(mask[:,18:26,4:11].mean((1,2))+mask[:,18:26,17:24].mean((1,2)))*.5
    trouser=ramp(gap,.26,.80)*ramp(legs,.33,.78)*ramp(upper,.72,.96)
    # Side view of shoe: the toe is lower than the ankle. Measure the two
    # ends away from the rounded-over pixels at the extremities.
    columns=b.any(1)
    lo=(top[:,3:10]*columns[:,3:10]).sum(1)/np.maximum(1,columns[:,3:10].sum(1))
    hi=(top[:,18:25]*columns[:,18:25]).sum(1)/np.maximum(1,columns[:,18:25].sum(1))
    slope=np.abs(hi-lo)/28
    toeslope=ramp(slope,.14,.50)
    footwear=np.maximum(wide*.50,asym*toeslope)
    shoeprior=.045*wide+.052*footwear-.052*(1-wide)*(1-asym)-.035*tall
    for k in [5,7,9]: scores[:,k]+=shoeprior
    for k in [0,1,2,3,4,6]: scores[:,k]-=.053*wide+.035*asym
    scores[:,1]+=.109*trouser+.037*ramp(1/asp,1.8,2.8)+.028*ramp(upper,.84,.98)-.047*ramp(.63-upper,0,.4)
    # High narrow handles above a substantial, mostly filled bag body.
    handle=ramp(core-upper,.22,.68)*ramp(core,.65,.94)*(1-.8*toeslope)
    scores[:,8]+=.085*handle+.036*wide*(1-toeslope)+.020*ramp(shapefill,.77,.91)-.025*tall*sym*(1-handle)
    for k in [0,2,3,4,6]: scores[:,k]-=.040*handle
    # Upright heel with an air space ahead of it and under the arch.
    bottom = np.where(columns, bot, 0)
    groundleft=bottom[:,2:11].max(1); groundright=bottom[:,18:27].max(1)
    arch=(np.minimum(groundleft,groundright)-bottom[:,12:21].min(1))/28
    arch=ramp(arch,.13,.40)
    strap=ramp(void,.045,.22); openarea=ramp(1-shapefill,.40,.69)
    scores[:,5]+=.105*strap+.047*ramp(below,.08,.25)+.040*openarea+.034*arch-.022*(1-strap)*(1-openarea)
    scores[:,7]+=.022*(1-strap)-.082*strap-.044*arch+.021*ramp(asp,1.85,3.0)
    scores[:,9]+=.026*(1-wide)+.018*toeslope-.062*strap-.015*ramp(asp,1.75,2.40)
    # A long level topline over the entire back quarter is characteristic of a shaft.
    back=np.where((hi<lo)[:,None],top[:,17:25],top[:,3:11][:,::-1])
    shaft=ramp((back<4).mean(1),.38,.88)*ramp(slope,.26,.60)
    scores[:,9]+=.047*shaft
    scores[:,7]-=.031*shaft
    # A dangling backpack strap or a second handle can make the article
    # asymmetric, but the body walls are often nearly vertical and turn into
    # a broad, level base. An outsole instead follows a much shallower body.
    flatbody=ramp(rowwidth[:,17:24].mean(1),.77,.95)*ramp(mask[:,16:24,4:24].mean((1,2)),.80,.98)
    scores[:,8]+=.038*flatbody*ramp(1-slope,.65,.90)*ramp(1.1-tall,.04,.65)
    # Shape cues for dresses and short-sleeved garments.
    bust=rowwidth[:,5:10].mean(1); waist=rowwidth[:,11:16].mean(1)
    hem=rowwidth[:,21:26].mean(1); low=rowwidth[:,18:23].mean(1)
    flare=ramp(hem-waist,.055,.31)
    dressform=flare*ramp(corebottom,.65,.98)
    scores[:,3]+=.066*dressform+.043*ramp(1/asp,1.62,2.45)*(1-trouser)-.055*trouser-.022*ramp(asp,.82,1.13)
    shortarm=ramp(bust-low,.13,.34)*ramp(core,.58,.91)
    scores[:,0]+=.031*shortarm-.038*dressform
    for k in [2,4,6]: scores[:,k]-=.025*dressform
    # Fabric evidence uses original tonal differences, not the binary silhouettes.
    # Search for a persistent narrow central opening/placket across several
    # heights. Subtract broad curvature and normalize by local fabric contrast.
    inner=b&(col>l[:,:,None]+1)&(col<r[:,:,None]-1)
    sm=smooth(gray)
    avg=regionmean(gray,inner&(row>4)&(row<25))
    norm=.16+.65*avg
    centers=[]; edges=[]
    for j in range(11,17):
        q=(sm[:,:,j]+sm[:,:,j+1])*.5
        left=(sm[:,:,j-3]+sm[:,:,j-2])*.5
        right=(sm[:,:,j+3]+sm[:,:,j+4])*.5
        contrast=(left+right)*.5-q
        lr=np.abs(left-right)
        sections=[]; side=[]
        for a,z in [(7,13),(13,19),(19,25)]:
            sections.append(contrast[:,a:z].mean(1)); side.append(lr[:,a:z].mean(1))
        sec=np.stack(sections,1); ed=np.stack(side,1)
        centers.append((np.abs(sec.mean(1))*.63+np.minimum(np.abs(sec).min(1),np.abs(sec.mean(1)))*.37)/norm)
        edges.append(ed.min(1)/norm)
    pl=np.stack(centers,1).max(1); ed=np.stack(edges,1).max(1)
    seam=ramp(pl,.048,.20); opening=ramp(pl,.15,.42); step=ramp(ed,.09,.30)
    # Difference between shoulders and center at the neck; V-shaped lapels often
    # continue from the neck almost to mid-chest, unlike a pullover crew neck.
    neck=sm[:,2:8,11:17].mean((1,2)); chest=sm[:,9:15,10:18].mean((1,2))
    ns=.5*(sm[:,2:8,6:10].mean((1,2))+sm[:,2:8,18:22].mean((1,2)))
    notch=ramp((.56*ns+.44*chest-neck)/norm,.11,.55)
    # Texture of interior cloth. Directional coherence separates checks and
    # prints from slowly changing shading and generally uniform knitwear.
    hp=np.abs(gray-sm)
    grain=regionmean(hp,inner&(row>=6)&(row<24))/norm
    grain=ramp(grain,.051,.145)
    dx=np.abs(sm[:,:,2:]-sm[:,:,:-2]); dy=np.abs(sm[:,2:]-sm[:,:-2])
    dx=regionmean(dx,inner[:,:,2:]&inner[:,:,:-2]&(row>=8)&(row<23))/norm
    dy=regionmean(dy,inner[:,2:]&inner[:,:-2]&(row[:,2:]>=8)&(row[:,2:]<23))/norm
    plaid=ramp(dx,.11,.29)*ramp(dy,.11,.27)
    # Compare the shirt-tail with the ends of the sleeves, measured relative to
    # the item box. A ribbed jumper often finishes a little above the cuffs.
    middle=bot[:,10:18].mean(1)
    sidebot=(bot[:,1:5].max(1)+bot[:,23:27].max(1))*.5
    cuffbelow=ramp((sidebot-middle)/28,-.025,.145)
    longcoat=ramp((middle-sidebot)/28,.08,.27)*(1-shortarm)
    # A curved bottom hem is common in shirts. Look at the central body, not
    # the end of the sleeves, for a shirttail that hangs lower in the middle.
    sidetail=(bot[:,6:9].mean(1)+bot[:,19:22].mean(1))*.5
    tail=ramp((middle-sidetail)/28,.034,.15)
    boxy=ramp(low,.76,.96)
    longarm=ramp(low/(bust+.03),.73,.99)
    scores[:,2]+=.034*(1-seam)+.020*(1-step)+.028*cuffbelow+.015*boxy-.042*opening-.021*notch-.021*plaid-.016*tail
    scores[:,4]+=.048*seam+.046*opening+.026*step+.021*notch+.030*longcoat+.013*tall-.022*shortarm-.025*plaid-.012
    scores[:,6]+=.033*seam+.012*step+.027*plaid+.028*grain+.028*tail+.020*notch-.015*cuffbelow-.014*boxy-.006
    # Track the two converging inner edges of open lapels. Local gradient
    # direction matters: scattered printed marks give less consistent support
    # than paired left/right diagonals which extend downwards toward the center.
    gx=(np.roll(sm,-1,2)-np.roll(sm,1,2))*.5
    gy=(np.roll(sm,-1,1)-np.roll(sm,1,1))*.5
    lapels=[]
    for depth in [8,12,16]:
        rr=np.arange(2,depth+1); t=(rr-2)/(depth-2)
        for radius in [4.5,6.5]:
            rad=radius*(1-t)+.8*t; sl=(radius-.8)/(depth-2)
            lcol=np.rint(13.5-rad).astype(np.int32); rcol=27-lcol
            evidence=[]
            for shift in [-1,0,1]:
                la=gx[:,rr,lcol+shift]-sl*gy[:,rr,lcol+shift]
                ra=-gx[:,rr,rcol+shift]-sl*gy[:,rr,rcol+shift]
                evidence.append((la+ra)*.5)
            # Minor alignment deviations are allowed; polarity must be
            # consistent up and down both lapels, whether dark or light.
            e=np.stack(evidence,1)/(1+sl*sl)
            a=e.max(1); d=-e.min(1); split=len(rr)//2
            support=np.maximum(np.minimum(a[:,:split].mean(1),a[:,split:].mean(1)),np.minimum(d[:,:split].mean(1),d[:,split:].mean(1)))
            lapels.append(support/norm)
    lapel=ramp(np.stack(lapels,1).max(1),.085,.24)
    scores[:,4]+=.031*lapel*ramp(longarm,.18,.80)
    scores[:,6]+=.012*lapel
    scores[:,2]-=.019*lapel
    # A short plain tee is the default short-sleeved garment. Distinguish a
    # woven collared short-sleeve shirt when central opening/texture is visible.
    scores[:,0]+=.014*(1-seam)*shortarm-.028*opening*shortarm-.013*tail
    # We only compare the fine fabric and hem evidence inside the upper-garment
    # family. Avoid raising a poor coat match above a good bag or shoe match
    # just because a zipper or strap produced a strong placket response.
    return scores


def block(a):
    # Robust foreground segmentation with a permissive and a stricter view.
    n=len(a); peak=np.maximum(.055,np.sort(a.reshape(n,-1),axis=1)[:,-18])
    sm=smooth(a); frac=.65*a+.35*sm
    base=.008+.033*peak
    first=clip((frac-base[:,None,None])/(.020+.091*peak[:,None,None]))
    # Smooth away isolated accent marks, but keep the edges of thin straps.
    m=.77*first+.23*smooth(first)
    second=clip((frac-(.014+.082*peak[:,None,None]))/(.030+.14*peak[:,None,None]))
    x0,x1,y0,y1=bounds(m)
    w=(x1-x0+1).astype(np.float32); h=(y1-y0+1).astype(np.float32)
    cx=(x0+x1).astype(np.float32)*.5; cy=(y0+y1).astype(np.float32)*.5
    size=np.maximum(w,h)
    m1=sample(m,cx,cy,size,size); m2=sample(second,cx,cy,size,size)
    shape=match([m1,m1[:,:,::-1],m2,m2[:,:,::-1]])
    scores=appearance(shape.copy(),sample(a,cx,cy,w,h),sample(m,cx,cy,w,h),w,h)
    ids=[0,2,4,6]
    # Within-family cues are useful even when all four silhouette correlations
    # are nearly tied. Remove their shared uplift in cross-family comparison.
    uplift=scores[:,ids].max(1)-shape[:,ids].max(1)
    scores[:,ids]-=np.maximum(.025,uplift-.048)[:,None]
    return scores.argmax(1)


def predict(x):
    result=np.empty(len(x),np.int64)
    for i in range(0,len(x),24):
        result[i:i+24]=block(np.ascontiguousarray(x[i:i+24,0],dtype=np.float32))
    return result
