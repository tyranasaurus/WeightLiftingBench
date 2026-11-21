import numpy as np

def ramp(a,lo,hi):
    return np.clip((a-lo)/(hi-lo),0.,1.)

def blur(a):
    p=np.pad(a,((1,1),(1,1)),mode="edge")
    return (4*a+2*(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])+p[:-2,:-2]+p[:-2,2:]+p[2:,:-2]+p[2:,2:])/16

def foreground(im):
    ref=np.maximum(.035,np.sort(im.ravel())[704])
    raw=im/ref
    local=blur(raw)
    # Interpolate coverage through the antialiased edge but do not grow a
    # light garment past the boundary onto a perfectly black background.
    m=.79*ramp(raw,.021,.19)+.21*ramp(local,.032,.17)
    m=.82*m+.18*np.maximum(m,blur(m))
    return m, np.minimum(raw,2.)

def framed(m):
    z=np.arange(28)
    def bounds(a):
        ok=(a.max(1)>.45)&(a.sum(1)>.85)
        lo=np.argmax(ok);hi=27-np.argmax(ok[::-1]);return lo,hi
    y0,y1=bounds(m);x0,x1=bounds(m.T)
    sc=max(8,y1-y0+1,x1-x0+1)/27.5
    x=(z-13.5)*sc+(x0+x1)/2;y=(z-13.5)*sc+(y0+y1)/2
    a=np.floor(x).astype(int);b=np.floor(y).astype(int);vx=x-a;vy=y-b
    p=np.pad(m,((2,2),(2,2)));a=np.clip(a+2,0,30);b=np.clip(b+2,0,30)
    return ((1-vy[:,None])*((1-vx)*p[b[:,None],a]+vx*p[b[:,None],a+1])+vy[:,None]*((1-vx)*p[b[:,None]+1,a]+vx*p[b[:,None]+1,a+1]))

def describe(mask):
    h=len(mask);f=h//14
    a=mask.reshape(14,f,14,f).mean(axis=(1,3));p=np.pad(a,((1,1),(1,1)))
    sm=(4*a+2*(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])+p[:-2,:-2]+p[:-2,2:]+p[2:,:-2]+p[2:,2:])/16
    v=[(.55*a+.45*sm).ravel()]
    for z in (mask,mask.T):
        for reverse in (False,True):
            # First-hit positions, pooled over strips, describe a contour
            # even when much of the interior is dark, as on an open sandal.
            w=(z[:,::-1] if reverse else z)>.46
            # Give each empty scanline a neutral midpoint before pooling:
            # one stray pixel should not define the edge of a whole strip.
            lim=.50;loc=np.where(w.any(1),np.minimum(lim,(np.argmax(w,axis=1)+.5)/h),lim)
            v.append(.90*loc.reshape(14,f).mean(1))
    return np.concatenate(v)

def shapes(x):
    m=CONSTANTS["models"];n=len(x);end=CONSTANTS["ends"]
    vals=np.empty((n,len(end)),np.float64)
    # Fixed padded block size makes the matrix multiplication independent of N.
    for k in range(0,n,48):
        b=x[k:k+48];data=np.zeros((48,m.shape[0]),np.float32)
        for j,im in enumerate(b):
            data[j]=describe(framed(im))
        dst=(data@ m)*(-2.)+CONSTANTS["norm"]+(data*data).sum(1)[:,None]
        start=0
        for i,stop in enumerate(end):
            v=dst[:len(b),start:stop]
            # Soft minimum approximated with a uniform mixture of nearest styles.
            small=np.partition(v,7,axis=1)[:,:8];values=-.58*(.7*small.mean(1)+.3*small.min(1))
            vals[k:k+len(b),i]=values
            start=stop
    return vals

def mix(s,weights):
    z=max(s);return z+np.log(sum(w*np.exp(v-z) for w,v in zip(weights,s)))

def detail(im,m,b):
    # Estimates of structure use per-image lightness. A button band can be
    # dark on light fabric or light on dark fabric, and is often only one pixel wide.
    smooth=blur(im)
    xx=np.arange(28);yy=xx[:,None]
    wm=m.sum(1)+.001
    center=(m*xx).sum(1)/wm
    axis=np.clip(np.median(center[7:22]),10.,17.)
    # Contrast of a straight placket/zip against the neighboring material.
    texture=np.abs(im-smooth)
    patch=(yy>=7)&(yy<24)&(np.abs(xx-axis)<5.5)&(m>.8)
    grain=(texture*patch).sum()/(patch.sum()+1)
    broad=np.abs(im[:,2:]-im[:,:-2]); rough=broad[7:24,8:18].mean()
    a=im[7:24,7:21];l=(im[7:24,5:19]+im[7:24,6:20])*.5;r=(im[7:24,8:22]+im[7:24,9:23])*.5
    ridge=a-(l+r)*.5;step=l-r
    evidence=np.maximum(abs(ridge.mean(0)),.68*abs(step.mean(0)))
    persist=np.maximum(np.sort(np.abs(ridge),axis=0)[5],.63*np.sort(np.abs(step),axis=0)[5])
    val=(.70*evidence+.30*persist)/(.12+1.5*grain+.32*rough)
    dist=np.abs(np.arange(7,21)-axis);seam=max(0.,np.max(np.where(dist<2.8,val-.08*dist,0)));off=np.max(np.where(dist>=2.8,val,0))
    zipline=ramp(seam-.23*off,.20,1.12)
    # Two diagonal lapels, folds or collar edges falling toward the middle.
    edge=np.abs(im[:,2:]-im[:,:-2]);lapels=0.
    for depth in (4,7,11):
        for spread in (2.7,4.6):
            z=np.arange(depth);xs=spread*(1-z/(depth+1))
            a=np.clip(np.rint(axis-xs).astype(int),1,26);d=np.clip(np.rint(axis+xs).astype(int),1,26)
            left=edge[z+2,a-1].mean();right=edge[z+2,d-1].mean()
            lapels=max(lapels,min(left,right))
    vee=ramp(lapels-.36*rough,.09,.39)
    # Relative shoulder height and the start of continuous cloth across the neck.
    ytop=np.argmax(m>.45,axis=0);ytop=np.where(m.max(0)>.45,ytop,28)
    c=int(np.rint(axis));atneck=np.median(ytop[c-1:c+2]);side=min(np.median(ytop[c-6:c-3]),np.median(ytop[c+4:c+7]))
    collar=ramp(side-atneck,1,4)
    scoop=ramp(atneck-side,.2,4.)
    # Patterned cloth and chest graphics. Compare high-frequency content at
    # similar brightness and look for prints localized to the center chest.
    body=im[7:22,c-4:c+5];tx=texture[7:22,c-4:c+5]
    ink=tx.mean()/(.16+body.mean())
    upper=texture[6:14,c-4:c+5].mean();lower=texture[16:24,c-4:c+5].mean();graphic=ramp(upper-1.5*lower,.018,.12)
    patterned=ramp(ink,.09,.27)
    # Separate central garment hem from sleeve tips: long cuffs below a
    # waistband suggest knitwear; a dipped shirt tail suggests a woven shirt.
    bottom=27-np.argmax((m>.49)[::-1],axis=0)
    hem=np.median(bottom[c-2:c+3]); quarter=np.median(bottom[[c-4,c-3,c+3,c+4]])
    roundhem=ramp(hem-quarter,.40,2.8)
    # Sleeve/body air gaps, estimated independently of cloth brightness.
    gaps=0.
    for row in (15,18,21,24):
        r=m[row];left=np.maximum.accumulate(r);right=np.maximum.accumulate(r[::-1])[::-1]
        inside=np.minimum(left,right)-r
        gaps+=.125*(max(inside[:c])+max(inside[c:]))
    def sm(a,w):return mix([b[i] for i in a],w)
    t=sm([0,1,2,32,37,40],[.42,.13,.055,.055,.26,.08]); p=sm([15,16,35],[.70,.16,.14]);d=sm([12,13,14,33,38],[.21,.23,.16,.12,.28])
    s=sm([4,5,3,11],[.71,.12,.10,.07]);coat=sm([6,7,11,41],[.53,.28,.06,.13]);shirt=sm([8,9,10,0,1,4,34],[.31,.20,.15,.105,.045,.07,.12]);
    sand=sm([28,29,30,31,36,39],[.23,.22,.12,.17,.09,.17]);trainer=sm([23,24],[.85,.15]);boot=sm([25,26,27],[.47,.31,.22]);bag=sm([17,18,19,20,21,22,42],[.20,.13,.16,.09,.21,.11,.10])
    s+=.68*gaps+1.05*graphic+.42*scoop-2.45*zipline-1.05*collar-.77*vee-.48*patterned
    coat+=2.1*zipline+1.23*collar+1.1*vee-.62*gaps-.58*graphic-.32*scoop-.44*roundhem-.46
    shirt+=1.43*patterned+.58*roundhem+.60*vee+.35*zipline-.53*graphic-.32
    t+=.50*graphic-.69*zipline-.54*patterned-.45*vee
    # Width profile through the trunk: a skirt may flare below its waist,
    # whereas the separated legs of trousers form a slit deep in the center.
    rmask=m>.4
    widths=28-np.argmax(rmask,axis=1)-np.argmax(rmask[:,::-1],axis=1)
    widths=np.where(rmask.any(1),widths,0)
    topw=np.median(widths[4:10]); ww=np.median(widths[11:16]);hw=np.median(widths[21:25])
    flare=ramp(hw-ww,.5,5.0);slim=ramp(19-topw,0,8.)
    d+=1.3*flare+.95*slim-.55*zipline-.6*gaps
    t-=.75*flare+.57*slim
    inseam=m[16:26,c-2:c+3].min(1);thigh=(m[16:26,c-5:c-1].max(1)+m[16:26,c+2:c+6].max(1))/2
    fork=ramp((thigh-inseam).mean(),.34,.83)
    waistband=ramp((m[1:4,c-3:c+4].mean()-m[1:4,c-6:c-3].mean()),-.2,.45)
    p+=3.5*fork+.48*waistband-1.2*ramp(topw-ww,1,6.)
    s-=1.9*fork;coat-=1.9*fork;d-=2.2*fork
    # Plaid, all-over prints and lace put horizontal and vertical detail
    # throughout a shirt, unlike a sweatshirt graphic limited to the chest.
    lowtx=texture[14:24,c-5:c+6].mean()/(.16+im[14:24,c-5:c+6].mean())
    lowerprint=ramp(lowtx,.095,.25)
    shirt+=1.15*lowerprint+1.05*patterned+.52*roundhem+.60*ramp(hw-topw,-2,4.)
    s-=.75*lowerprint;coat-=.62*lowerprint
    # Estimate sleeve length independently of the template: use the lowest
    # fabric near each outer side, relative to the lowest central fabric.
    colok=(m>.49).sum(0)>=2;edgeL=np.argmax(colok);edgeR=27-np.argmax(colok[::-1]);radius=(edgeR-edgeL+1)*.5
    leftq=edgeL+np.array([.05,.11,.17])*(c-edgeL);rightq=edgeR+np.array([.05,.11,.17])*(c-edgeR)
    cuff=(np.median(bottom[np.clip(np.rint(leftq).astype(int),0,27)])+np.median(bottom[np.clip(np.rint(rightq).astype(int),0,27)]))*.5
    sleevedrop=hem-cuff
    wings=ramp(sleevedrop,9,17)*ramp(topw-ww,1.,5.)
    t+=2.6*wings*(1-.45*roundhem)-1.15*roundhem
    d-=1.65*wings*ramp(topw,19,25)
    shirt-=.85*wings*(1-.70*patterned)
    drop=ramp(sleevedrop,1.,6.5); longarm=ramp(cuff-hem,-2.,1.5)
    s+=.85*longarm*(1-.65*zipline)-1.10*drop-.75*roundhem
    shirt+=.66*drop+.66*ramp(sleevedrop,2,7)*ramp(15-sleevedrop,0,7)
    # Flat knit, collared outerwear, and printed/bloused shirts can share a
    # silhouette. A closely fitted rib-knit waistband runs horizontally;
    # a collared jacket has a smaller neck platform and converging panels.
    neckpeak=ramp((np.median(ytop[c-8:c-4])+np.median(ytop[c+5:c+9]))/2-np.mean(ytop[c-2:c+3]),1.6,5.)
    bandedge=np.abs(im[1:]-im[:-1]);br=np.clip(int(hem)-5,6,22)
    band=np.max(np.abs((im[br+1:br+5,c-4:c+5]-im[br:br+4,c-4:c+5]).mean(1)))
    elastic=ramp(band-rough*.33,.045,.23)*longarm*(1.-zipline)
    s+=.60*elastic+.36*(1.-neckpeak)*(1.-vee)
    coat+=.85*neckpeak+.45*vee*zipline-.5*elastic
    # A waistband is wide and level at the top, unlike a pair of shoulders.
    wy=np.median(ytop[c-3:c+4]);flat=np.mean(np.abs(ytop[c-4:c+5]-wy)<1.5)
    p+=1.7*ramp(flat,.65,1)*ramp(21-topw,0,7)-1.15*ramp(wy,2,6)
    # Is the bottom two separated cuffs, or one contiguous skirt hem?
    # Allow the split to curve off center but require real cloth on both sides.
    leg=m[19:26];middle=leg[:,c-3:c+4];gapcol=np.argmin(middle,axis=1)+c-3
    leftleg=np.max(np.where(xx[None,:]<gapcol[:,None]-1,leg,0),axis=1)
    rightleg=np.max(np.where(xx[None,:]>gapcol[:,None]+1,leg,0),axis=1)
    divided=np.mean(np.minimum(leftleg,rightleg)-np.min(middle,1))
    # Photometric crease for legs which touch: a dark line surrounded on
    # both sides by appreciably brighter fabric, absent near the waistband.
    crease=0.
    for shift in (-2,0,2):
        pos=c+shift; dark=im[16:26,pos-1:pos+2].min(1);sur=np.minimum(im[16:26,pos-4:pos-1].max(1),im[16:26,pos+2:pos+5].max(1))
        crease=max(crease,np.mean((sur-dark)/(.15+sur)))
    legproof=max(ramp(divided,.21,.76),.7*ramp(crease,.28,.74))
    rowtop=np.argmax(widths>3);lip=widths[rowtop:min(rowtop+3,28)].mean()/(np.mean(widths[6:11])+.5)
    jeans=ramp(lip,.60,.94)*ramp(flat,.62,.99)
    p+=3.2*legproof-5.5*(1-legproof)*(1-.70*jeans)-2.9*flare-1.5*neckpeak
    d+=1.55*flare*(1-legproof)+.90*ramp(21-topw,0,6)*(1-legproof)
    # Sleeve-free tops and standard short-sleeved tees need neither buttons
    # nor a shirt collar. A plain crew neck and a horizontal hem distinguish
    # them from printed woven blouses and dip-hem shirts.
    plain=ramp(.15-ink,.015,.10)*(1-.65*zipline)
    tankshape=ramp(max(b[1],b[2])-max(b[0],b[4],b[8]),0,3)
    t+=1.25*wings+.95*plain*wings+1.25*tankshape-.55*roundhem*patterned
    shirt-=.65*plain*wings+.7*tankshape*(1-patterned)
    # Check for a sustained front opening even when one panel is off center:
    # jackets often show an open V at the top leading to a narrow lower slit.
    zz=im[12:24,c-4:c+5];sideavg=(im[12:24,c-6:c-3].mean(1)+im[12:24,c+4:c+7].mean(1))*.5
    oneline=np.max(np.abs((zz-sideavg[:,None]).mean(0)))/(.19+rough*1.5+np.std(zz))
    opening=ramp(oneline,.39,1.48)*ramp(np.mean(m[9:21,c-4:c+5]),.72,.98)
    # Shirt fabrics may be patterned in both directions (checks, florals);
    # a horizontal-only stripe is just as plausible on a pullover or tee.
    dx=np.abs(body[:,1:]-body[:,:-1]).mean();dy=np.abs(body[1:]-body[:-1]).mean()
    twoway=ramp(dx/(.01+dy),.42,1.00);stripe=1-twoway
    coat+=1.05*opening+.48*zipline+.72*neckpeak*ramp(3.5-sleevedrop,0,4)
    s-=1.20*opening+.4*vee+.64*neckpeak
    shirt+=.64*twoway*lowerprint-.80*stripe*patterned
    t+=.80*stripe*patterned;s+=.70*stripe*patterned
    # Strong horizontal shoulder line above a handle aperture is evidence
    # for luggage even if the body itself is rather tall and coat-shaped.
    ylo=np.argmax(m.max(1)>.46);yhi=27-np.argmax(m.max(1)[::-1]>.46);hei=max(1,yhi-ylo+1)
    h1=min(yhi,ylo+int(.32*hei));h2=min(yhi,ylo+int(.60*hei))
    upperbody=m[h1:h2+1];handle=(upperbody[:,:c-2].max(1)+upperbody[:,c+3:].max(1))*.5-upperbody[:,c-2:c+3].mean(1)
    handleloop=ramp(handle.mean(),.22,.83)*ramp(m[h2:min(yhi+1,h2+6),c-3:c+4].mean(),.25,.80) if h2<=yhi else 0.
    broadbody=ramp(hw,21,27)*ramp(7-abs(hw-ww),0,6)
    bag+=3.8*handleloop+1.45*broadbody*(1-wings)*ramp(1-gaps,.44,.95)
    # The less material within the shoe outline, the more likely a strapped
    # sandal. Boots usually have a steep vamp and a high rear shaft.
    cols=(m>.48).any(0);ht=28-ytop;bot=bottom
    span=np.maximum(0,bot-ytop+1)*cols
    holes=(span.sum()-m.sum())/(span.sum()+1)
    # A column hull misses the air below a slingback or between widely
    # spaced straps. Examine the interior of an elastic envelope over the
    # upper and the sole separately from the concavity under a high heel.
    foot=framed(m);on=foot>.33
    topedge=np.where(on.any(0),np.argmax(on,0),27);base=np.where(on.any(0),27-np.argmax(on[::-1],0),0)
    if topedge[3:9].mean()<topedge[-9:-3].mean():
        foot=foot[:,::-1];topedge=topedge[::-1];base=base[::-1]
    roof=topedge.astype(np.float64);floor=base.astype(np.float64)
    for step in range(3,24):
        xp=np.arange(28-step);a=roof[xp];z=roof[xp+step]
        for frac in (.25,.5,.75):
            j=int(np.rint(frac*step));v=frac*z+(1-frac)*a
            roof[xp+j]=np.minimum(roof[xp+j],v)
    interior=(yy>=roof+.4)&(yy<=floor-.3)&(xx>0)&(xx<27)
    air=((1-foot)*interior).sum()/(interior.sum()+.01)
    openstraps=ramp(air,.085,.34)
    # Deep, repeated holes between a strap and the sole carry more weight
    # than a single depression at the mouth of a closed shoe.
    columnspace=np.maximum(0,base-topedge+1)-foot.sum(0)
    through=np.sort(columnspace[2:26])[-7:-2].mean();window=ramp(through,.85,3.3)
    above=interior&(yy<base-1);darkspace=((foot<.27)&above).sum()/(above.sum()+1)
    openstraps*=.47+.53*max(window,ramp(darkspace,.10,.30))
    # Slides are almost all sole in the rear, or else have two detached
    # islands on it; a closed sneaker maintains a solid heel counter.
    filled=foot.sum(0);toeheight=filled[2:12].mean();heelheight=filled[19:26].mean()
    backless=ramp(toeheight-heelheight,0,3.0)*ramp(5-heelheight,0,3)
    thinsole=ramp(9.0-(base.max()-topedge.min()+1),0,5.0)
    sand+=8.0*openstraps+5.5*backless+1.6*ramp(holes,.12,.39)+2.6*window+3.3*thinsole
    trainer-=4.5*openstraps+3*backless;boot-=4.8*openstraps+3.5*backless
    # A sneaker's tongue rises in front of the scooped opening; a boot's
    # ankle tube is more nearly level across its top, or taller at the back.
    tongue=np.median(np.sort(topedge[10:19])[:3]);notch=np.max(topedge[18:24]);rear=np.median(topedge[22:26])
    tonguecue=ramp(notch-tongue,.8,3.5)*ramp(notch-rear,.35,2.)
    shortshoe=ramp(15-(base.max()-topedge.min()),0,5.)
    trainer+=1.5*tonguecue+1.20*shortshoe;boot-=1.2*tonguecue+.75*shortshoe
    # Final decisions within type: short sleeves produce a sharp drop in
    # row span; a skirt flares progressively below the waist instead. The
    # neckline of a sleeveless top usually has two relatively high straps.
    shoulderwidth=np.sort(widths[3:12])[-3:].mean();trunk=np.median(widths[14:19]);hemwidth=np.median(widths[21:25])
    overhang=ramp(shoulderwidth-max(trunk,hemwidth*.93),2.,7.)*ramp(sleevedrop,6.,15.)
    teeproportion=ramp(shoulderwidth,18.,25.)
    shortT=overhang*(.42+.58*teeproportion)
    tankneck=max(scoop,ramp(np.mean(ytop[c-2:c+3])-min(np.median(ytop[c-5:c-2]),np.median(ytop[c+3:c+6])),.4,4))
    tankcue=ramp(max(b[1],b[2],b[40])-max(b[4],b[6],b[8],b[14]),1,6)*tankneck
    narrowskirt=ramp(20.-shoulderwidth,0,7)*ramp(3.0-abs(hemwidth-trunk),-.5,3)
    t+=1.9*shortT+1.4*tankcue+.9*ramp(b[0]-max(b[8],b[10]),0,3.)
    d-=2.1*shortT+.8*tankcue+1.25*zipline+.75*neckpeak+.85*gaps+.75*ramp(shoulderwidth,19,25)
    shirt-=1.2*shortT*(1-.53*patterned)
    # The garment prefix shares a strong prior in ambiguous upper-body
    # silhouettes. A printed front on a long sleeve tee may be knitwear,
    # but texture covering sleeves as well as the torso supports a blouse.
    armzone=(np.abs(xx-axis)>.31*(edgeR-edgeL))&(m>.86)&(yy>=6)&(yy<=19)
    armtexture=(texture*armzone).sum()/((im*armzone).sum()+.19*armzone.sum()+.1)
    allprint=lowerprint*ramp(armtexture,.085,.22)*twoway
    shirt+=1.30*allprint+1.1*roundhem*(1-zipline)+.70*drop*ramp(hw,14,21)-.7*tankcue
    s-=.7*allprint+.52*roundhem;coat-=.85*allprint
    # Analyze broad collar/hood shading as a ratio of upper and lower
    # centered contrasts. This avoids mistaking full-height plaid for lapels.
    sidevar=im[3:10,c-6:c+7].std();trunkvar=im[12:21,c-5:c+6].std()
    throat=ramp(sidevar-.60*trunkvar,.09,.27)*vee
    coat+=.84*throat+scoop*(-.35)-.65*roundhem*(1-zipline)
    # Side profiles with a flat, low sole are more likely to be sneakers
    # when their toes and heel counters are solid. A boot shaft has a
    # nearly vertical front above the vamp. Search the actual upper outline.
    tallr=max(0.,np.mean(base[19:25]-topedge[19:25]));lowfoot=ramp(13.-tallr,0.,5.)
    topdiff=topedge[8:21]-topedge[10:23]
    shaft=ramp(np.max(topdiff),3.0,7.)*ramp(tallr,9.,16.)
    enclosed=ramp(air,.035,.15)*ramp(through,.8,2.6)
    heelcut=ramp(base[7:12].mean()-np.min(base[15:23]),2.0,6.)
    trainer+=1.45*lowfoot*(1-openstraps)*(1-shaft)-1.75*heelcut-1.3*shaft
    boot+=1.35*shaft+1.35*heelcut-.78*lowfoot
    sand+=1.55*enclosed+2.15*thinsole*(1.-tonguecue)-1.0*tonguecue*(1-window)
    # Apply fabric rules within the clothes group. Their sometimes large
    # bonuses are not evidence that a rectangular piece of luggage is cloth.
    garmentbase=max(b[i] for i in (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,33,34,35,37,38,40,41))
    gracescore=max(t,p,s,d,coat,shirt)-garmentbase
    bag+=.62*max(gracescore,0.)
    bag+=1.4*handleloop+1.2*broadbody*(1-overhang)*(1.-gaps)
    return np.argmax(np.array([t,p,s,d,coat,sand,shirt,trainer,bag,boot]))

def predict(x):
    out=np.empty(len(x),np.int64)
    for k in range(0,len(x),240):
        ims=x[k:k+240,0]; masks=np.empty_like(ims);norm=np.empty_like(ims)
        for j in range(len(ims)):
            masks[j],norm[j]=foreground(ims[j])
        vals=shapes(masks)
        for j in range(len(ims)):
            out[k+j]=detail(norm[j],masks[j],vals[j])
    return out
