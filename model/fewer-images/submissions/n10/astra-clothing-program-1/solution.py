import numpy as np

def sample(im, box, side):
    left,top,right,bottom = box
    t=(np.arange(side)+.5)/side
    xx=left-.5+(right-left+1)*t; yy=top-.5+(bottom-top+1)*t
    xx=np.clip(xx,0,im.shape[1]-1); yy=np.clip(yy,0,im.shape[0]-1)
    ix=xx.astype(int); iy=yy.astype(int)
    ax=(xx-ix).astype(np.float32); ay=(yy-iy).astype(np.float32)[:,None]
    p=im[:,ix]*(1-ax)+im[:,np.minimum(ix+1,im.shape[1]-1)]*ax
    return p[iy]*(1-ay)+p[np.minimum(iy+1,im.shape[0]-1)]*ay

def bounds(binary):
    rows=np.flatnonzero(binary.any(1)); cols=np.flatnonzero(binary.any(0))
    if len(rows)==0 or len(cols)==0: return (0,0,binary.shape[1]-1,binary.shape[0]-1)
    return (cols[0],rows[0],cols[-1],rows[-1])

def pool(a,s):
    return a.reshape(a.shape[0]//s,s,a.shape[1]//s,s).mean((1,3))

def describe(a):
    p=pool(a,2)
    b=a>.44
    h=np.maximum.accumulate(b,axis=1)&np.maximum.accumulate(b[:,::-1],axis=1)[:,::-1]
    v=np.maximum.accumulate(b,axis=0)&np.maximum.accumulate(b[::-1],axis=0)[::-1]
    # A blur for silhouette agreement, plus envelopes which ignore interior detail.
    padded=np.pad(p,1,mode='edge')
    blur=.5*p+.125*(padded[:-2,1:-1]+padded[2:,1:-1]+padded[1:-1,:-2]+padded[1:-1,2:])
    h=h.astype(np.float32); v=v.astype(np.float32)
    pieces=[blur.ravel()*12, pool(h,4).ravel()*13, pool(v,4).ravel()*13,
            a.mean(0).reshape(16,2).mean(1)*19, a.mean(1).reshape(16,2).mean(1)*19]
    return np.rint(np.concatenate(pieces)).astype(np.int8)

def fabric(a,v):
    # Contrast-normalized gradient histograms of the neck, front closure, hem,
    # and two contiguous fabric panels. Each descriptor is image-local.
    lum=np.log(.07+v)
    collar=np.maximum(a,.07)
    pieces=[]
    for stride in (1,2):
        p=np.pad(lum,stride,mode='edge'); n=len(lum)
        gx=p[stride:stride+n,2*stride:]-p[stride:stride+n,:n]
        gy=p[2*stride:,stride:stride+n]-p[:n,stride:stride+n]
        mag=np.sqrt(gx*gx+gy*gy)+.00001
        strength=np.minimum(.9,mag)*collar
        angle=np.arctan2(gy,gx)
        orient=np.stack([strength*np.maximum(0,np.abs(np.cos(angle-th*np.pi/4))-.65)/.35 for th in range(4)])
        for y0,y1,x0,x1,rows,cols,weight in ((0,12,4,28,3,6,1.20),(8,28,10,22,4,3,1.28),(24,32,8,24,2,4,.68),(8,24,4,28,2,6,.77)):
            blocks=orient[:,y0:y1,x0:x1].reshape(4,rows,(y1-y0)//rows,cols,(x1-x0)//cols).mean((2,4))
            # The balance of orientations matters even with faint stitching.
            # Retain some absolute contrast; a plain chest is also evidence.
            norm=.053+.48*blocks.sum(0)+.18*blocks.sum(0).mean()
            pieces.append(np.minimum(2.0,blocks/norm).ravel()*weight)
    return np.rint(22*np.concatenate(pieces)).astype(np.int8)


def sat(x):
    return np.clip(x,0,1)

def largest(v):
    # Keep the main article, not isolated sharpening specks or a detached tag.
    h,w=v.shape
    seed=np.flatnonzero(v)
    seen=np.zeros((h,w),bool)
    keep=np.zeros((h,w),bool); best=0
    for s in seed:
        row=int(s//w); col=int(s%w)
        if seen[row,col]: continue
        stack=[(row,col)]; seen[row,col]=True; group=[]
        while stack:
            r,c=stack.pop(); group.append((r,c))
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                rr=r+dr; cc=c+dc
                if 0<=rr<h and 0<=cc<w and v[rr,cc] and not seen[rr,cc]:
                    seen[rr,cc]=True; stack.append((rr,cc))
        if len(group)>best:
            best=len(group); keep[:]=False
            for rr,cc in group: keep[rr,cc]=True
    return keep

def readitem(raw):
    level=np.sort(raw.ravel())[-16]
    im=np.minimum(1.4,raw/max(.005,level))
    p=np.pad(im,1,mode='edge')
    sm=.64*im+.09*(p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])
    weak=sm>.029; m=largest(weak)
    # Reattach details near the main component. Do not discard a detached strap
    # or a second piece of a dark object that is otherwise well supported.
    p=np.pad(m,2); near=m.copy()
    for r in range(5):
        for c in range(5): near |= p[r:r+28,c:c+28]
    m |= weak & near
    if m.sum()<.77*weak.sum(): m=weak
    left,top,right,bottom=bounds(m & (sm>.056))
    # Estimate placement from the substantial silhouette of the item. A thread
    # or one projecting corner should not change the aspect ratio very much.
    row=m[:,left:right+1].sum(1); col=m[top:bottom+1].sum(0)
    rr=np.flatnonzero(row>=2); cc=np.flatnonzero(col>=2)
    if len(rr)>0:
        top=max(top,rr[0]-1); bottom=min(bottom,rr[-1]+1)
    if len(cc)>0:
        left=max(left,cc[0]-1); right=min(right,cc[-1]+1)
    box=(left,top,max(left,right),max(top,bottom))
    ar=(box[2]-box[0]+1)/(box[3]-box[1]+1)
    soft=sat((sm-.019)/.13)*m
    a=sample(soft,box,32)
    val=sample(im,box,32)
    return a,val,ar

def detect(a,v,ar):
    # Semantic evidence, in dimensionless units within the object's own box.
    # Texture and placket detectors use local contrast, not garment colour.
    m=a>.46; yy,xx=np.indices((32,32)); pos=np.arange(32)
    rev=m[:,::-1]
    sym=1-np.mean(abs(a-a[:,::-1]))/(.16+a.mean())
    b=m.any(0); start=np.where(b,m.argmax(0),32)
    end=np.where(b,32-m[::-1].argmax(0),0)
    ymin=m.argmax(1); ymax=31-rev.argmax(1)
    widths=np.where(m.any(1),ymax-ymin+1,0)/32
    mh=np.maximum.accumulate(m,1)&np.maximum.accumulate(rev,1)[:,::-1]
    mv=np.maximum.accumulate(m,0)&np.maximum.accumulate(m[::-1],0)[::-1]
    inside=mh & mv
    holes=(1-a)*inside
    topvoid=np.maximum(0, (1-a[:16,8:24]).mean()-(1-a[16:28,8:24]).mean())
    shoulder=widths[5:11].mean(); waist=widths[14:20].mean(); hips=widths[24:29].mean()
    # A handbag has a loop over a wide, largely contiguous pouch. A backpack
    # with no loop typically has rounded shoulders rather than upper sleeves.
    sides=(a[18:27,:6].mean()+a[18:27,26:].mean())*.5
    loop=sat((topvoid-.18)/.48)*sat((holes[:19,6:26].mean()-.09)/.42)
    topcenter=start[12:20].mean(); flanks=(start[4:10].mean()+start[22:28].mean())/2
    # Footwear: low forward toe box, long sole, raised quarters at the rear.
    heights=np.maximum(0,end-start)
    hLeft=heights[2:10].mean(); hRight=heights[22:30].mean()
    diff=abs(hRight-hLeft)/32
    symmetry=sat((sym-.70)/.21)
    shoeShape=sat((diff-.13)/.30)*sat((.93-sym)/.22)
    if hLeft>hRight: a=a[:,::-1]; v=v[:,::-1]; m=m[:,::-1]; start=start[::-1]; end=end[::-1]; mv=mv[:,::-1]; holes=holes[:,::-1]
    # Is the sole separated from the upper by open air? Vertical fill is
    # especially informative about sandals: it doesn't count the arch below.
    windows=np.sum((1-a[4:28,3:29])*mv[4:28,3:29])/max(10,mv[4:28,3:29].sum())
    apertures=sat((windows-.14)/.36)
    # Monotone shaft atop the heel versus a padded heel below a sneaker tongue.
    roofFront=start[2:10].mean()/32; roofBack=start[23:29].mean()/32
    roofMid=start[16:22].mean()/32
    shaft=sat((.30-roofBack)/.25)*sat((roofFront-.45)/.26)
    collarDip=sat((roofBack-roofMid-.025)/.23)
    arch=sat(((.5*(end[3:10].mean()+end[27:31].mean())-end[18:25].mean())/32-.055)/.20)
    mass=a.mean(); shoe=sat((ar-1.06)/.85)*sat((.97-sym)/.20)
    score=np.zeros(10,np.float64)
    score[5]+=3.8*apertures+1.3*sat((.42-mass)/.22)
    score[7]+=.9*collarDip+1.2*sat((ar-1.85)/.78)*(1-apertures)-1.8*apertures
    score[9]+=1.7*shaft+1.0*arch-.9*collarDip-2.6*apertures
    score[[5,7,9]]+=1.4*shoeShape+.8*shoe-2.6*symmetry*sat((2-ar)/.75)
    score[8]+=4*loop+1.4*symmetry*sat((ar-.88)/.45)+1.1*sides*sat((ar-.8)/.55)-2.4*shoeShape
    score[[0,2,3,4,6]]-=2.3*shoeShape+2.8*loop
    # Trouser leg separation extends upwards from the hem. Unlike a handle,
    # the bottom of this gap is open, and it has material on BOTH sides.
    mid=a[19:31,14:18].mean(); legs=min(a[19:31,5:12].mean(),a[19:31,20:27].mean())
    split=sat((legs-mid-.12)/.55)
    score[1]+=4.3*split+1.1*sat((.54-ar)/.18)-2*sat((ar-.65)/.30)
    score[[0,2,3,4,6]]-=2.2*split
    # Dress proportions. A waist indentation with a flaring skirt is strong
    # evidence, even in a relatively short dress with cap sleeves.
    flare=sat((hips-waist-.04)/.25)
    dress=sat((.64-ar)/.24)
    score[3]+=2.3*dress+1.7*flare*sat((.80-ar)/.3)-.8*sat((ar-.80)/.18)
    # Re-establish left/right garment coordinates; the statistics below are
    # mirror invariant apart from rounding of the central strip.
    p=np.pad(v,((0,0),(3,3)),mode='edge')
    axial=np.zeros(32)
    for dx in (-2,0,2):
        strip=v[:,15+dx:17+dx].mean(1)
        l=v[:,12+dx:14+dx].mean(1); r=v[:,18+dx:20+dx].mean(1)
        # A vertical ridge or trough, not just a light half and dark half.
        d=((l+r)*.5-strip)/(.15+(strip+l+r)/3)
        axial[8:28]=np.maximum(axial[8:28],abs(d[8:28]))
    # Compare the sustained central opening with general vertical folds.
    smooth=(p[:,1:33]+2*p[:,2:34]+3*p[:,3:35]+2*p[:,4:36]+p[:,5:37])/9
    field=(np.abs(.5*(smooth[:,8:22]+smooth[:,14:28])-smooth[:,11:25]))/(smooth[:,11:25]+.17)
    ridge=(.5*(p[:,:32]+p[:,6:38])-smooth)/(.15+smooth)
    def line(lo,hi,c1,c2):
        reg=ridge[lo:hi,c1:c2]
        energy=(np.abs(reg).mean(0)*.52 + np.abs(reg.mean(0))*.48)
        return energy.max()
    background=(line(9,27,7,12)+line(9,27,20,25))/2
    plack=line(9,28,13,19)
    persistence=min(line(9,18,13,19),line(18,28,13,19))
    opening=sat((.62*plack+.38*persistence-.42*background-.046)/.19)
    # Paired diagonals running down from either side of the collar (lapels).
    lap=0.
    for depth in (7,11,15):
        ry=np.arange(2,depth+2); off=np.rint(np.linspace(4.7,.8,depth)).astype(int)
        for center in (15,16):
            left=ridge[ry,center-off]; right=ridge[ry,center+off]
            lap=max(lap,.5*abs(left.mean()+right.mean())+.25*(abs(left).mean()+abs(right).mean()))
    lapel=sat((lap-.095)/.29)
    raised=sat((flanks-topcenter-1.9)/4.4)
    collar=raised*sat((.75-holes[:6,11:21].mean())/.45)
    # Neck band vs broad open shirt collar: an open top is either V-shaped
    # or cut deeply into the body, not a high-standing neck.
    lowneck=sat((topcenter+.5-flanks)/5.0)
    # Repeated checks or stripes are more typical of shirting than outerwear.
    lum=v[9:27,9:23]; cont=lum+.19
    vx=np.abs(lum[:,2:]-lum[:,:-2])/(.5*(cont[:,2:]+cont[:,:-2]))
    vy=np.abs(lum[2:]-lum[:-2])/(.5*(cont[2:]+cont[:-2]))
    checks=sat((min(vx.mean(),vy.mean())-.12)/.28)
    gx=np.abs(v[:,2:]-v[:,:-2]); gy=np.abs(v[2:]-v[:-2]);
    vdetail=np.mean(gx[9:25,8:22])/(.13+lum.mean()); hdetail=np.mean(gy[9:25,9:23])/(.13+lum.mean())
    smoothKnit=sat((.16-max(vdetail,hdetail))/.13)
    # Cuffs, high hem at the sides, and soft shirttails.
    centerHem=np.median(end[12:20]); edgeHem=.5*(np.median(end[8:11])+np.median(end[21:24]))
    tail=sat((centerHem-edgeHem-.8)/2.8)
    sleeveBottom=a[21:28,:5].mean()+a[21:28,27:].mean()
    short=sat((shoulder-hips-.11)/.28)*sat((.5-sleeveBottom)/.32)
    score[0]+=1.2*short+.45*lowneck+.48*(1-opening)-.8*collar*opening
    score[2]+=1.12*(1-opening)+.57*smoothKnit+.55*lowneck-.95*opening-.62*lapel*collar
    score[4]+=2.05*opening+1.30*collar+1.05*lapel+.52*sat((.81-ar)/.24)-.72*tail-.42*checks-.8*lowneck
    score[6]+=.95*opening+.69*tail+1.30*checks+.40*lapel+.38*sat((.8-ar)/.28)-.42*collar*lapel
    return score

def construction(a,v,ar):
    # Long garment fasteners can be visible as just one edge: not every coat
    # has a dark stripe bounded on both sides. Look for a coherent edge along
    # gently tilted center lines. Contrast is relative to noise on that image.
    eps=.035
    lum=np.log(.09+v)
    lum=(lum+np.roll(lum,1,axis=0)*.5+np.roll(lum,-1,axis=0)*.5)/2
    y=np.arange(8,28)
    gx=np.roll(lum,-1,axis=1)-np.roll(lum,1,axis=1)
    def tracks(centers):
        strengths=[]
        for slope in (-.10,0,.10):
            xx=np.rint((y[:,None]-17)*slope+centers[None,:]).astype(int)
            d=gx[y[:,None],xx]
            # Consecutive sections should agree. Separate sign before pooling
            # so that a shaded side of the placket also counts as a fastener.
            for sign in (-1,1):
                val=d*sign
                parts=np.maximum(0,val.reshape(4,5,-1).mean(1))
                strengths.append(.5*np.sort(parts,axis=0)[1]+.5*np.minimum(.6,parts).mean(0))
        return np.max(np.array(strengths),axis=0)
    strength=tracks(np.arange(7,25)); center=strength[6:13].max()
    flank=np.median(np.concatenate((strength[:5],strength[-5:])))
    vertical=sat((center-.038-flank*.85)/(.12+flank*.45))
    # Local bilateral symmetry of a visible button/fly strip. A two-panel
    # closure extends through a majority of the torso, below a potential lapel.
    collarArea=np.mean(a[:3,12:20]); shoulders=np.mean(a[:3,5:10])+np.mean(a[:3,22:27])
    raised=sat((collarArea-shoulders*.5-.10)/.63)
    sect=a>.47; st=sect.argmax(0); en=np.where(sect.any(0),32-sect[::-1].argmax(0),0)
    st=np.where(sect.any(0),st,32)
    # Contrast along mirror-paired obliques: coat lapels and pressed collars
    # usually run into the opening rather than lying horizontally.
    diag=0.
    for depth in (7,12,17):
        ry=np.arange(2,depth+2)
        for w in (4,7):
            off=np.rint(np.linspace(w,1,depth)).astype(int)
            le=gx[ry,15-off]; ri=gx[ry,16+off]
            paired=.5*(le-ri)
            diag=max(diag, .60*abs(paired.mean())+.40*np.abs(paired).mean())
    lapel=sat((diag-.09)/.30)
    # A pullover can have a collar but has no front fastener. A crew-neck or
    # V-neck sweater usually has its contrast confined to the top of the chest.
    neckContrast=np.std(lum[1:8,11:21]); torsoContrast=np.std(lum[10:26,12:20])
    neckOnly=sat((neckContrast-torsoContrast-.03)/.22)*(1-vertical)
    # Cloth pattern over the two front panels, omitting center, cuffs and hem.
    body=np.concatenate((lum[8:26,9:14],lum[8:26,18:23]),1)
    dx=np.concatenate((np.diff(body[:,:5],axis=1),np.diff(body[:,5:],axis=1)),axis=1)
    dy=np.diff(body,axis=0)
    printStrength=sat((min(abs(dx).mean(),abs(dy).mean())-.054)/.16)
    # Dissimilar button-front panels and rounded shirttails favor a shirt.
    panel=np.mean(abs(lum[8:26,10:15]-lum[8:26,17:22][:,::-1]))
    asymmetric=sat((panel-.14)/.26)
    centerHem=np.median(en[13:19]); tail=sat((centerHem-(np.median(en[8:12])+np.median(en[20:24]))*.5-.6)/2.9)
    hips=a[22:27].mean(0); chest=a[6:11].mean(0)
    # Short sleeves abruptly end to either side of the chest. Tank tops are
    # handled mainly by the outline matching, with a broad cut-out neck.
    sideLow=.5*(hips[:5].mean()+hips[-5:].mean()); sideHi=.5*(chest[:5].mean()+chest[-5:].mean())
    short=sat((sideHi-sideLow-.14)/.40)
    openNeck=sat((np.median(st[12:20])-.5*(np.median(st[7:11])+np.median(st[21:25]))-.4)/4.5)
    # Measure the *shape* of the collar pedestal, not just whether its
    # central pixels are filled. Neighboring neck and shoulder contours move
    # together when a high collar is added, unlike a scooped ribbed neck.
    shoulderline=np.r_[st[6:10],st[22:26]]
    leftNeck=min(st[10:16]); rightNeck=min(st[16:22])
    abrupt=sat((np.median(shoulderline)-(leftNeck+rightNeck)/2-1.65)/3.15)
    notch=sat((np.mean(st[14:18])-.5*(leftNeck+rightNeck)-.6)/2.6)
    stand=abrupt*(1-.7*notch)
    # Lapel edges are not necessarily parallel to the center-line. Look for
    # an inverted chevron on either side of the neck (points of a collar) and
    # downward-converging lines of longer lapels on both fronts.
    fold=0.; pointed=0.
    for depth in (5,9,14):
        ry=np.arange(1,depth+2); frac=np.arange(depth+1)/depth
        for spread in (3,6):
            for ctr in (15,16):
                margin=1+spread*(1-frac)
                lx=np.rint(ctr-margin).astype(int); rx=np.rint(ctr+margin).astype(int)
                lsign=gx[ry,lx]; rsign=gx[ry,rx]
                fold=max(fold,.5*abs(lsign.mean()-rsign.mean())+.25*(abs(lsign).mean()+abs(rsign).mean()))
                if depth<10:
                    lx=np.rint(ctr-2-spread*frac).astype(int); rx=np.rint(ctr+2+spread*frac).astype(int)
                    pointed=max(pointed,min(abs(gx[ry,lx].mean()),abs(gx[ry,rx].mean())))
    folds=sat((fold-.085-flank*.25)/(.24+flank))
    points=sat((pointed-.07)/.22)
    # Plackets can be low-contrast, interrupted by dots (buttons), or curved.
    # Compare local ridge evidence near the midline with adjacent fabric at
    # the same rows, and require support down the front, not just on the chest.
    ridge=np.abs(lum-.5*(np.roll(lum,-2,axis=1)+np.roll(lum,2,axis=1)))
    lines=(ridge[9:29,:]*.5+abs(gx[9:29,:])*.5).reshape(4,5,32).mean(1)
    stripe=np.median(lines[:,13:19].max(1)); textile=np.median(lines[:,[8,9,10,11,20,21,22,23]])
    seam=sat((stripe-.035-1.2*textile)/(.095+.7*textile))
    button=1-(1-vertical)*(1-.78*seam)
    # Flat cuffs form a band crossing the sleeve near its lower end. A ribbed
    # waistband often echoes it on the torso of a pullover.
    band=0.
    for ry in range(23,29):
        delta=lum[ry-1,11:21]-lum[ry+1,11:21]
        band=max(band,abs(delta.mean())-.6*np.std(delta))
    rib=sat((band-.09)/.27)*(1-tail)
    # Strong, spatially repeated contrasts and a gently rounded shirttail are
    # useful even on a shirt whose placket is concealed. Sleeve and torso
    # panels may be lit asymmetrically on softly draped woven fabric.
    woven=sat((min(abs(dx).mean(),abs(dy).mean())-.040)/.12)
    fine=.5*(abs(dx).mean()+abs(dy).mean()); panelStd=np.std(body)
    drape=sat((panelStd-fine*1.35-.07)/.22)
    collar=stand
    # Style proportions from the actual hem, rather than the row envelope:
    # sleeves can project below a cropped sweater, or a long body can extend
    # down well past the sleeves of a coat. On shirting the body often ends
    # in a rounded tail, while the straight ribbed hem of a knit stops abruptly.
    torsoHem=sat((np.median(en[11:21])-(np.quantile(en[2:7],.75)+np.quantile(en[25:30],.75))*.5-1.3)/5)
    straight=sat((1.9-np.std(en[10:22]))/1.6)
    crop=sat(((np.quantile(en[2:7],.75)+np.quantile(en[25:30],.75))*.5-np.median(en[11:21])-.60)/3.6)
    slight=sat((.83-ar)/.24); boxy=sat((ar-.70)/.21)
    slim=slight*(1-.65*short); longLine=torsoHem*(1-short)
    # Differentiate a weak but continuous front closure from a genuinely
    # uniform center panel. A contrast ratio within this one image is more
    # reliable on a light coat than a fixed minimum ridge amplitude.
    ratio=sat((stripe-.009-1.4*textile)/(.015+1.4*textile))
    closure=max(button,.82*ratio)
    # Compare how well the item satisfies each construction, without treating
    # an invisible fastener alone as evidence for a pullover.
    knit=.50*(1-closure)+.52*neckOnly+.45*notch+.65*rib+.45*boxy+.57*crop+.30*straight-.46*collar-.39*folds
    outer=1.22*closure+1.05*collar+.65*folds+.49*closure*collar+.22*straight+.68*slim+.48*longLine-.42*short-.52*woven+.58*points*collar
    shirt=.78*closure+1.04*woven+.80*tail+.56*asymmetric+.42*points+.66*drape+.47*points*(1-collar)+.72*closure*short+.58*slim+.42*longLine-.3*collar
    tee=1.05*short+.70*(1-button)+.43*openNeck+.4*(1-tail)-.6*points*button
    wear=np.array([tee,knit,outer,shirt]); wear-=wear.max()
    # Leg fork: search a connected opening entering from the lower hem. The
    # center of a cropped pair of pants need not run down the column midline.
    lower=a[15:31]
    off=np.arange(9,23)
    gap=1-lower[:,off]
    flank1=lower[:,off-5]; flank2=lower[:,off+5]
    gap=np.maximum(0,np.minimum(flank1,flank2)-lower[:,off])
    perRow=gap.max(1); fork=.6*perRow[-8:].mean()+.4*perRow.mean()
    belt=a[:4,5:27].mean(); neck=a[:3,13:19].mean()
    # Evidence for narrow unbroken legs (front-view trousers with touching
    # ankles) can be a flat waistband and a taper from hip to hem.
    breadth=lambda z: np.mean(np.sum(np.maximum.accumulate(z>.5,axis=1)&np.maximum.accumulate(z[:,::-1]>.5,axis=1)[:,::-1],axis=1))/32
    taper=breadth(a[6:13])-breadth(a[23:30]); belt=sat((belt-.40)/.43)
    pants=sat((fork-.24)/.53)*sat((.91-ar)/.35)
    pants=max(pants,.70*belt*sat((taper+.03)/.23)*sat((.61-ar)/.20))
    # Air under a strap indicates a sandal. Measure holes also at a higher
    # contrast level, for nearly black voids adjacent to sharpened straps.
    air=0.
    for mask in (a>.5,v>.12):
        col=np.maximum.accumulate(mask,0)&np.maximum.accumulate(mask[::-1],0)[::-1]
        row=np.maximum.accumulate(mask,1)&np.maximum.accumulate(mask[:,::-1],1)[:,::-1]
        inside=col[3:28,3:29]; dark=(~mask)[3:28,3:29]
        # A connected gap with a finite span across and down is stronger than
        # isolated dark pixels in a patterned textile or gaps along the laces.
        g=inside&dark
        square=g[1:,1:]&g[:-1,1:]&g[1:,:-1]&g[:-1,:-1]
        air=max(air,sat((g.sum()+square.sum()*1.5-18)/max(28,.33*inside.sum())))
    # Side-view closed shoes have one continuous vamp. An open heel leaves a
    # tail of bare sole, often with its top nearly parallel to its bottom.
    tall=(en-st)/32
    if tall[3:9].mean()>tall[23:29].mean(): tall=tall[::-1]; st=st[::-1]; en=en[::-1]
    tip=np.mean(tall[2:7]); back=np.mean(tall[24:30]); highest=np.mean(tall[12:22]); rearLow=sat((highest-back-.05)/.32)
    soleOnly=sat((.46-back)/.26)*rearLow
    # Use the shape of the rear collar as well as absolute proportions. A boot
    # shaft tends to be an upright plateau rather than a tongue and heel cup.
    shaft=sat((back-tip-.20)/.36)*sat((.35-np.mean(st[24:29])/32)/.28)
    flat=sat((ar-1.77)/.86); upright=sat((2.15-ar)/.71)
    heelArch=sat(((.5*(en[4:10].mean()+en[27:31].mean())-en[18:24].mean())/32-.035)/.24)
    # Sneakers usually have a concave topline between the tongue and heel
    # counter. Even high tops often have this ankle opening. Boot shafts stay
    # high over a continuous interval; merely having a tall tongue is weak
    # evidence. Sandals can have an almost bare midfoot over a continuous sole.
    dip=sat((st[20:27].mean()-.5*(st[15:20].min()+st[26:31].min())-1.8)/5)
    plateau=sat((.28-st[19:29].mean()/32)/.23)
    midLow=sat((.43-tall[13:21].min())/.30)*sat((tall[6:14].max()-.50)/.30)
    boot=1.10*shaft+.72*upright+.74*heelArch+.81*plateau-.84*dip-2.25*air-1.05*soleOnly
    sneaker=.95*flat+.82*(1-air)+.42*(1-shaft)+1.0*dip-1.45*soleOnly-.70*heelArch
    sandal=2.90*air+2.30*soleOnly+.45*heelArch+.76*midLow
    foot=np.array([sandal,sneaker,boot]); foot-=foot.max()
    return wear,foot,pants

# Exact integer squared distances to individual drawings. The aspect ratio is
# matched separately in log space so that a proportional resizing costs the
# same for e.g. a narrow dress and a wide shoe.
patterns=CONSTANTS['pattern'].astype(np.float32)
norms=(patterns*patterns).sum(1)
logasp=CONSTANTS['aspect']
sections=[np.flatnonzero(CONSTANTS['label']==k) for k in range(10)]
bank=patterns.T.copy()
tiles=CONSTANTS['textile'].astype(np.float32)
tileNorm=(tiles*tiles).sum(1)
tileBank=tiles.T.copy(); links=CONSTANTS['links']
tileSections=[np.flatnonzero(CONSTANTS['label'][links]==k) for k in (0,2,4,6)]

def classify(x):
    num=len(x)
    features=np.empty((num*2,bank.shape[0]),np.float32)
    scores=np.empty((num,10),np.float64)
    shape=np.empty((num,10),np.float64)
    cloth=np.empty((num,4)); shoes=np.empty((num,3)); pants=np.empty(num)
    ar=np.empty(num,np.float32); textile=np.empty((num*2,tileBank.shape[0]),np.float32)
    for i in range(num):
        a,v,aspect=readitem(x[i,0]); ar[i]=np.log(aspect)
        features[2*i]=describe(a); features[2*i+1]=describe(a[:,::-1]); scores[i]=detect(a,v,aspect)
        cloth[i],shoes[i],pants[i]=construction(a,v,aspect)
        textile[2*i]=fabric(a,v); textile[2*i+1]=fabric(a[:,::-1],v[:,::-1])
    dist=features@bank
    dist *= -2
    dist += (features*features).sum(1)[:,None]
    dist=norms+np.minimum(dist[::2],dist[1::2])
    delta=np.abs(ar[:,None]-logasp)
    dist += 46000*delta**2/(1+delta)
    for k,indices in enumerate(sections):
        best=dist[:,indices].min(1)
        shape[:,k]=-best/165
    scores+=shape
    dtex=-2*(textile@tileBank)+(textile*textile).sum(1)[:,None]
    dtex=tileNorm+np.minimum(dtex[::2],dtex[1::2])
    # Compare edge patterns with a modest alignment cost from the corresponding
    # drawing, so that a neckline is not matched to the end of a short sleeve.
    dtex+=.34*dist[:,links]
    fabricCost=np.stack([dtex[:,sec].min(1) for sec in tileSections],1)
    # Simulated appearance is a supporting vote; style and lighting vary much
    # more in the photographs than in the schematic textile drawings.
    cloth-=1.15*np.tanh((fabricCost-fabricCost.min(1,keepdims=True))/970)
    # Group first: a jacket and a sweater can have indistinguishable outlines.
    # Once inside a group, an imperfect match to its closest pattern is weak
    # evidence. Bound its effect and use the independent construction cues.
    groups=([0,2,4,6],[5,7,9])
    for cols,details,scale in zip(groups,(cloth,shoes),(1.85,2.20)):
        sh=shape[:,cols]; old=scores[:,cols]; prior=old-sh; peak=old.max(1,keepdims=True)
        loss=sh.max(1,keepdims=True)-sh
        if len(cols)==4:
            # Distinguishing a tee from a coat by its short sleeves is much
            # more reliable than distinguishing a long-sleeve shirt from one.
            longLoss=sh[:,1:].max(1,keepdims=True)-sh[:,1:]
            # A long-sleeved garment has two separable shape decisions: sleeve
            # length versus a tee, and a much weaker silhouette preference
            # among collared shirts, jackets and knits of similar proportions.
            commonLoss=sh.max(1,keepdims=True)-sh[:,1:].max(1,keepdims=True)
            penalty=np.empty_like(sh)
            penalty[:,1:]=.68*commonLoss+1.35*np.tanh(longLoss/4)
            penalty[:,0]=.60*loss[:,0]+.40*1.8*np.log1p(loss[:,0]/2.5)
        else:
            penalty=2.7*np.log1p(loss/3.5)
        final=scale*details+.56*prior-penalty
        scores[:,cols]=peak+final-final.max(1,keepdims=True)
    # Discount an accidental dress-outline match when there is a leg fork.
    scores[:,1]+=5*pants
    scores[:,3]-=5*pants
    return np.argmax(scores,axis=1)

def predict(x):
    out=np.empty(len(x),np.int64)
    for i in range(0,len(x),64): out[i:i+64]=classify(x[i:i+64])
    return out
