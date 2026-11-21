import numpy as np


def sat(z):
    return np.clip(z,0.,1.)


def ramp(z,lo,hi):
    return sat((z-lo)/(hi-lo))


def strips(a,y0,y1,x0,x1):
    return np.mean(a[:,y0:y1,x0:x1],axis=(1,2))


def morph(a,op):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='constant')
    b=a.copy()
    for dy,dx in ((0,1),(1,0),(2,1),(1,2)):
        b=op(b,p[:,dy:dy+28,dx:dx+28])
    return b


def shapeScores(m):
    coarse=m.reshape(-1,7,4,7,4).mean((2,4)).reshape(-1,49)
    row=m.mean(2).reshape(-1,14,2).mean(2); col=m.mean(1).reshape(-1,14,2).mean(2)
    t=m>.36; grid=np.arange(28)
    bounds=(np.min(np.where(t,grid[None,:,None],28),1),28-np.max(np.where(t,1+grid[None,:,None],0),1),
            np.min(np.where(t,grid[None,None,:],28),2),28-np.max(np.where(t,1+grid[None,None,:],0),2))
    outlines=np.concatenate([q.reshape(-1,14,2).mean(2) for q in bounds],1)*.64/28.
    a=np.concatenate((coarse,.85*row,.85*col,outlines),axis=1)
    vec=CONSTANTS['drawing']; norms=CONSTANTS['energy']; sizes=CONSTANTS['sizes']
    # Integer dot products of fixed-point descriptors, exact irrespective of
    # the SIMD or matrix kernel chosen for the number of images presented.
    a=np.rint(256*a).astype(np.float64)
    d=(np.sum(a*a,1)[:,None]+norms[None,:]-2.*(a@vec))/(256.*256.)
    scores=[]; start=0
    for size in sizes:
        stop=start+int(size); best=np.partition(d[:,start:stop],4,axis=1)[:,:5]
        scores.append(-np.mean(np.sqrt(np.maximum(best,0.)+.01),axis=1))
        start=stop
    return np.stack(scores,1)


def details(x,m,b):
    n=len(x)
    # Bilateral seam detector. A placket or zipper has a long consistent
    # vertical contrast at the centre. Compare to nearby fabric so plaid,
    # folds, and all-over print are not mistaken for front closures.
    p=(x[:,5:24,:]+x[:,4:23,:]+x[:,6:25,:])/3.
    ridge=p[:,:,2:-2]-(p[:,:,:-4]+p[:,:,4:])*.5
    change=p[:,:,2:-2]-p[:,:,1:-3]
    def persistence(z):
        return np.maximum(np.abs(z.mean(1)),.70*np.abs(z).mean(1))
    edge=persistence(ridge)+.55*persistence(change)
    central=edge[:,9:15].max(1); flank=np.maximum(edge[:,5:9].mean(1),edge[:,15:19].mean(1))
    luminous=strips(x,6,24,7,21)
    seam=ramp((central-.95*flank)/(.16+luminous),.037,.29)
    # Collar / lapels form a tapered V converging on the centre front.
    level=np.abs(x[:,2:14,1:]-x[:,2:14,:-1])
    vlist=[]
    for length in (5,8,12):
        for width in (3,5):
            index=np.arange(length); span=np.rint(width*(1-index/length)).astype(int)
            vlist.append(np.mean(level[:,index,13-span]+level[:,index,13+span],1)*.5)
    lapel=ramp((np.max(vlist,axis=0)-.90*level[:,:,6:21].mean((1,2)))/(.18+luminous),.00,.21)
    # The contrast of a pattern on lightweight fabric; stripes or a chest
    # graphic alone are not the same as variation covering the whole body.
    tex=np.abs(x[:,1:,1:]-x[:,:-1,1:])+np.abs(x[:,1:,1:]-x[:,1:,:-1])
    allprint=ramp(np.minimum(strips(tex,7,14,8,19),strips(tex,16,23,8,19))/(.22+luminous),.14,.52)
    graphic=ramp((strips(tex,5,14,8,19)-strips(tex,15,23,8,19))/(.2+luminous),.025,.36)
    # Bottom of sleeves relative to torso hem; shirt tails tend to be curved,
    # and knitted waistbands can finish well above the dangling cuffs.
    bot=np.max(np.where(b,np.arange(1,29)[None,:,None],0),axis=1)
    middle=np.median(bot[:,11:17],1)
    def portions(lo,hi):
        a=bot[:,lo:hi]; c=bot[:,28-hi:28-lo]
        return np.maximum(a.max(1),c.max(1))*.35+np.minimum(a.max(1),c.max(1))*.65
    sleeves=portions(0,6); hem=portions(6,10)
    tail=ramp(middle-hem,.8,3.8)
    knitband=ramp(sleeves-middle,-.65,2.9)
    # A narrow zipper, a wide opening between two lapels, and the two
    # halves of a button-front shirt all give different local signatures.
    closure=[]
    for wid in (2,4):
        for c in (12,13,14,15):
            for end in (15,24):
                strip=x[:,5:end,c-1:c+1].mean(2)
                side=(x[:,5:end,c-wid-1:c-wid+1].mean(2)+x[:,5:end,c+wid-1:c+wid+1].mean(2))*.5
                closure.append(abs(((strip-side)/(.12+.5*(strip+side))).mean(1)))
    opening=ramp(np.max(closure,axis=0),.07,.45)
    # Outerwear often has raised shoulders leading up to a hood or a
    # standing collar, in contrast to the hollow scooped neck of a tee.
    ymax=np.arange(1,29)[None,:,None]
    top=np.min(np.where(b,np.arange(28)[None,:,None],28),1)
    peak=np.mean(top[:,8:11]+top[:,17:20],1)*.5-np.mean(top[:,12:16],1)
    hood=ramp(peak,.25,2.8)
    # A terminating shirt cuff leaves the torso as the sole component at
    # the hem; many sweatshirts have two narrow cuffs below the main body.
    shirtcuff=ramp(middle-sleeves,1.,5.)
    # Paired endpoints relative to *this* silhouette's sleeve/torso gap.
    # Find the last occupied pixel when moving out from the spine; unlike
    # a fixed column range this does not take a loose sleeve for the hip.
    sidewidth=[]; returns=[]; gaps=[]; cuffs=[]; falls=[]
    for half in (m[:,:,14:],m[:,:,13::-1]):
        run=np.sum(np.minimum.accumulate(half,axis=2),axis=2)
        extent=np.max((half>.45)*np.arange(1,15)[None,None,:],axis=2)
        slot=(extent>run+1.6)&(run>1.8)
        gaps.append(slot[:,15:24].mean(1))
        # Persistent black space just beyond the side of the torso, ending
        # in a sleeve (not a scan-line void out in the background).
        proj=np.sum(half[:,16:23,:],axis=1)
        fw=np.maximum.accumulate(proj,axis=1); bw=np.maximum.accumulate(proj[:,::-1],axis=1)[:,::-1]
        returns.append(np.max(np.minimum(fw,bw)-proj,axis=1)/7.)
        sidewidth.append(run[:,15:23].mean(1))
        reach=np.max(extent,1); fraction=np.arange(1,15)[None,:]/np.maximum(1,reach[:,None])
        depths=np.max(np.where(half>.45,ymax,0),1)
        cuffs.append(np.max(depths*(fraction>.73),1))
        flank=(fraction>=.36)&(fraction<=.60)
        falls.append((flank*depths).sum(1)/np.maximum(1,flank.sum(1)))
    sleeveend=np.minimum(cuffs[0],cuffs[1])*.7+np.maximum(cuffs[0],cuffs[1])*.3
    knitband=ramp(sleeveend-middle,-.65,2.9); shirtcuff=ramp(middle-sleeveend,1.,5.)
    tail=ramp(middle-.5*(falls[0]+falls[1]),.8,3.8)
    drape=np.maximum(np.minimum(gaps[0],gaps[1]),.7*np.minimum(returns[0],returns[1]))
    # Woven/checkered fabric varies in BOTH axes, while knitted stripes
    # often only change vertically. Near-periodicity is additional evidence.
    crop=x[:,6:23,8:20]; vgrad=np.abs(np.diff(crop,axis=1)).mean((1,2)); hgrad=np.abs(np.diff(crop,axis=2)).mean((1,2))
    weave=ramp(np.minimum(vgrad,hgrad)/(.14+luminous),.052,.18)
    interior=morph(morph(m,np.minimum),np.minimum)[:,1:,1:]>.59
    def fabricpatch(a,b):
        ins=interior[:,5:23,a:b]; return np.sum(tex[:,5:23,a:b]*ins,(1,2))/(8.+ins.sum((1,2)))
    sideprint=ramp(np.minimum(fabricpatch(6,10),fabricpatch(17,21))/(.14+luminous),.10,.40)
    vertical=ramp((hgrad-vgrad)/(.13+luminous),.024,.145)
    horizontal=ramp((vgrad-hgrad)/(.13+luminous),.025,.17)
    yoke=ramp(np.mean(abs(np.diff(x[:,2:8,10:18],axis=1)),(1,2))/(.15+luminous),.066,.240)
    neckline=ramp(-peak,-.25,2.5)
    return (seam,lapel,allprint,graphic,tail,knitband,opening,hood,shirtcuff,
            drape,weave,sideprint,vertical,horizontal,yoke,neckline)


def classify(raw):
    n=len(raw)
    bright=np.maximum(.02,np.max(raw,(1,2)))[:,None,None]; x=raw/bright
    # High-pass sharpening in the source photos can leave dark cracks in a
    # solid item. Close the cracks but keep strap and handle holes.
    padded=np.pad(x,((0,0),(1,1),(1,1)),mode='edge')
    sm=(4*x+padded[:,:28,1:29]+padded[:,2:,1:29]+padded[:,1:29,:28]+padded[:,1:29,2:])/8.
    signal=.6*x+.4*sm
    thresh=np.minimum(.12,.26*np.max(sm,(1,2)))[:,None,None]
    m=ramp(signal,.27*thresh,1.50*thresh)
    openmask=m
    closed=morph(morph(m,np.maximum),np.minimum)
    m=np.maximum(m,.83*closed); b=m>.47
    templates=shapeScores(m)
    tee,tank,knit,outer,shirt,blouse,dress,pants,bag,sandal,sneaker,boot=templates.T
    rows=b.sum(2); cols=b.sum(1)
    ys=np.arange(28)[None,:]; xs=ys
    top=np.min(np.where(rows>=2,ys,28),1); bottom=np.max(np.where(rows>=2,ys,-1),1)
    left=np.min(np.where(cols>=2,xs,28),1); right=np.max(np.where(cols>=2,xs,-1),1)
    w=np.maximum(1,right-left+1); h=np.maximum(1,bottom-top+1); aspect=w/h
    # Compare a profile to its horizontal reflection; symmetry is measured on
    # the silhouette, not on logos, zippers or directional lighting.
    symmetrics=[]
    for center in (26,27,28):
        j=center-np.arange(28); mirror=m[:,:,np.clip(j,0,27)]*((j>=0)&(j<28))
        symmetrics.append(np.sum(abs(m-mirror),(1,2))/(1.+np.sum(m+mirror,(1,2))))
    asym=np.min(symmetrics,axis=0)
    def rowzone(lo,hi):
        select=(ys>=top[:,None]+lo*h[:,None])&(ys<top[:,None]+hi*h[:,None]); count=np.maximum(1,np.sum(select,1))
        return (select*rows).sum(1)/count
    bust=rowzone(.15,.32); waist=rowzone(.38,.57); hip=rowzone(.62,.78); hem=rowzone(.80,.92)
    flare=ramp((hem-waist)/w,.035,.30)
    # Disconnected legs: a central dark column continuing toward the hem,
    # bordered on both sides by fabric. A skirt usually has no such gap.
    leg=np.zeros(n)
    for c in range(11,17):
        score=(np.minimum(m[:,10:26,c-5:c].max(2),m[:,10:26,c+1:c+6].max(2))*(1.-m[:,10:26,c]))
        leg=np.maximum(leg,.65*score[:,5:].mean(1)+.35*score.mean(1))
    trows=rows>=2; topcols=np.min(np.where(b,np.arange(28)[None,:,None],28),axis=1)
    botcols=np.max(np.where(b,np.arange(28)[None,:,None],-1),axis=1)
    def colheight(lo,hi):
        select=(xs>=left[:,None]+lo*w[:,None])&(xs<left[:,None]+hi*w[:,None])&(cols>=2)
        return ((bottom[:,None]-topcols+1)*select).sum(1)/np.maximum(1,select.sum(1))
    toe=colheight(.05,.25); heel=colheight(.74,.94); ankle=colheight(.55,.74)
    tilt=(heel-toe)/h; neckheight=(heel-ankle)/h
    # Fill within the outer upper and sole profile at each column. Sandals
    # expose background inside this envelope; closed shoes fill it with fabric.
    void=np.maximum(0,botcols-topcols+1)-np.sum(m,1)
    toearea=(xs>=left[:,None]+.10*w[:,None])&(xs<left[:,None]+.92*w[:,None])
    porous=(void*toearea).sum(1)/np.maximum(8,((np.maximum(0,botcols-topcols+1))*toearea).sum(1))
    fill=m.sum((1,2))/(w*h)
    (seam,lapel,allprint,graphic,tail,knitband,opening,hood,shirtcuff,
     drape,weave,sideprint,vertical,horizontal,yoke,neckline)=details(x,m,b)
    # Look at the actual waistband, rather than depending on seeing a
    # black crotch. Trousers with adjacent legs can look like a solid bar.
    belt=rowzone(0.,.09); narrow=ramp(aspect,.72,.49)
    flatbelt=ramp(belt/(bust+.5),.53,.94)
    straight=ramp((bust-hem)/w,-.04,.25)
    split=np.zeros(n)
    for c in range(11,17):
        central=x[:,13:26,c]; sides=np.minimum(x[:,13:26,c-3:c].max(2),x[:,13:26,c+1:c+4].max(2))
        split=np.maximum(split,ramp(((sides-central)/(.18+sides)).mean(1),.19,.70))
    # An inseam is allowed to slant as a leg tapers. Accumulate the
    # best connected dark path from ankle to crotch, scoring it only
    # where there is a left and a right piece of fabric on the same row.
    cand=np.arange(9,19); g=x[:,11:27,cand]; leftfabric=np.zeros_like(g); rightfabric=np.zeros_like(g)
    for step in (1,2,3,4):
        leftfabric=np.maximum(leftfabric,x[:,11:27,cand-step]); rightfabric=np.maximum(rightfabric,x[:,11:27,cand+step])
    side=np.minimum(leftfabric,rightfabric)
    valley=sat((side-g)/(.10+side))*(.4+.6*(1.-openmask[:,11:27,cand]))
    path=valley[:,0]
    for k in range(1,16):
        pad=np.pad(path,((0,0),(1,1)),mode='edge')
        path=valley[:,k]+np.maximum(path,np.maximum(pad[:,:-2],pad[:,2:])-.10)
    seamlegs=ramp(path.max(1)/16,.20,.52)
    legs=np.maximum(seamlegs,leg)
    bodycurve=ramp((np.maximum(hip,hem)-waist)/w,-.02,.18)
    # Use the undecorated, open mask for shoe holes. Small cutouts are
    # structural even when narrower than the smoothing element.
    void2=np.maximum(0,botcols-topcols+1)-openmask.sum(1)
    porous=np.maximum(porous,(void2*toearea).sum(1)/np.maximum(8,((np.maximum(0,botcols-topcols+1))*toearea).sum(1)))
    # A deep notch between separated straps may be open to the exterior,
    # and hence not counted by a vertical hole detector. Close the upper
    # envelope horizontally between pairs of raised points to detect it.
    tt=np.where(cols>=2,topcols,bottom[:,None])
    cover=np.maximum(0,tt-np.maximum(np.minimum.accumulate(tt,axis=1),np.minimum.accumulate(tt[:,::-1],axis=1)[:,::-1]))
    notch=ramp((cover*toearea).max(1)/h,.21,.58)
    rear=ramp((ankle-heel)/h,.16,.54)
    openfoot=np.maximum.reduce([ramp(porous,.08,.37),notch,rear,ramp(fill,.52,.29),ramp(aspect,2.95,4.2)])
    # For each shoe cross-section, look for empty intervals along both
    # directions. A sandal may have a cavern between an ankle strap and
    # the vamp in a *row*, or between the toe strap and sole in a *column*.
    soleband=(ys>=top[:,None]+.16*h[:,None])&(ys<bottom[:,None]-.04*h[:,None])
    item=x>.09
    rowhole=np.cumsum(item,axis=2).astype(bool)&np.cumsum(item[:,:,::-1],axis=2)[:,:,::-1].astype(bool)&(~item)
    colhole=np.cumsum(item,axis=1).astype(bool)&np.cumsum(item[:,::-1,:],axis=1)[:,::-1,:].astype(bool)&(~item)
    cavity=np.maximum(ramp(((rowhole.sum(2)/w[:,None])*soleband).max(1),.19,.55),
                      ramp(((colhole.sum(1)/h[:,None])*toearea).max(1),.23,.65))
    # Average thickness of a continuous wedge of fabric over the instep:
    # straps in that region expose the foot, even on a platform sandal.
    instep=(xs>=left[:,None]+.34*w[:,None])&(xs<left[:,None]+.75*w[:,None])
    midsole=(openmask.sum(1)*instep).sum(1)/np.maximum(1,instep.sum(1))
    slender=ramp(midsole/h,.53,.24)
    # Require area as well as length: a deep but single-row line may be
    # lacing or an outsole stripe, not a hole that exposes the foot.
    rlength=np.maximum.accumulate(item,axis=2)&np.maximum.accumulate(item[:,:,::-1],axis=2)[:,:,::-1]
    length=np.maximum(1,rlength.sum(2)); gm=rowhole.sum(2)/length
    rvoid=((gm*soleband).sum(1)/np.maximum(1,soleband.sum(1)))
    coherent=ramp(rvoid,.08,.31)
    bracket=np.sqrt(cavity*np.maximum(coherent,ramp(porous,.07,.23)))
    flatness=ramp(aspect,3.05,4.25)
    openfoot=np.maximum.reduce([openfoot,bracket,slender*.72,flatness])
    # Sneakers have an independent tongue which crests ahead of the heel
    # cup. A boot's shaft usually maintains its height back to the heel.
    tongue=colheight(.46,.64); collar=colheight(.77,.97)
    tongueup=ramp((tongue-collar)/h,-.05,.21)
    shaft=ramp(collar/h,.62,.91)*ramp(tilt,.16,.49)
    uppers=ramp(aspect,.54,.75)*ramp(aspect,1.45,1.05)
    # A pulled-over body tends to be continuous across the chest. Opening,
    # collar and hem observations provide explicitly relative style votes.
    centerline=.62*seam+.38*opening
    knitCue=.56*knitband+.24*graphic+.20*(1.-hood)
    coatCue=.52*centerline+.27*hood+.21*lapel
    shirtCue=.35*allprint+.26*tail+.23*shirtcuff+.16*lapel
    base=np.maximum.reduce([knit,outer,shirt])
    kt=.48*knit+.52*base; ct=.48*outer+.52*base; st=.48*shirt+.52*base
    shortsleeve=ramp((bust-hip)/w,.08,.29)
    sleeveless=np.maximum(tank-tee,0.)
    # Biases are log-evidence-like votes, not estimated probabilities. Most
    # fine-grained cues only apply after the basic silhouette is plausible.
    front=ramp(asym,.15,.48); wide=ramp(aspect,1.05,1.65)
    upright=1.-wide
    common=-.72*front-.80*wide
    sc=np.zeros((n,10))
    sc[:,0]=np.maximum(tee,tank-.02)+common+.18+.17*graphic+.26*shortsleeve-.16*centerline-.20*flare
    sc[:,1]=pants-.32*front-.8*wide+.70*leg+.17*split+narrow*(.12+.56*flatbelt+.40*straight)+.85*legs-.58*(1.-legs)-.64*flare-.26*bodycurve
    sc[:,2]=kt+common+.09+.83*knitCue-.86*coatCue-.37*shirtCue+.32*drape+.16*neckline+.16*horizontal-.30*weave
    sc[:,3]=dress+common+.60*flare+.32*narrow-.52*legs-.24*shortsleeve*uppers+.28*bodycurve-.12*flatbelt*narrow
    sc[:,4]=ct+common-.04+1.26*coatCue-.54*knitCue-.30*shirtCue-.18*allprint+.24*yoke*hood+.15*(1.-drape)-.20*weave
    sc[:,6]=np.maximum(st,blouse-.05)+common-.08+1.02*shirtCue+.19*centerline-.49*knitCue-.31*hood-.10*graphic+.51*weave+.31*vertical+.26*sideprint+.12*tail*centerline-.22*horizontal
    sc[:,8]=bag+.18+.16*wide-.19*ramp(tilt,.21,.59)
    sh=.37*ramp(tilt,.11,.52)+.19*front+.10*wide-.39*ramp(tilt,.18,-.12)*upright
    closedfoot=np.maximum(boot,sneaker)
    # Recognize the garment family by shape before resolving fabric style.
    # A cropped photo of a dress often puts its edges inside the region in
    # which a shirt could have a placket; such apparent texture must not
    # be able to select the shirt family in the first place.
    shortform=np.maximum(tee,tank-.02)+common+.20+.22*shortsleeve+.12*graphic-.19*flare
    longform=base+common+.25+.13*drape-.14*narrow
    midform=blouse+common+.10
    upperform=np.maximum.reduce([shortform,longform,midform])
    solidlegs=ramp(aspect,.51,.38)*flatbelt
    sc[:,1]+=.30*solidlegs
    longdress=narrow*(1.-legs)*(.33+.35*bodycurve+.22*(1.-flatbelt))
    sc[:,3]+=longdress
    look=np.stack([sc[:,0],sc[:,2],sc[:,4],sc[:,6]],1)
    # Look for continuous, symmetric arcs of contrast at the neckline.
    # Crew necks bend across the spine near the top; an open collar has
    # two diagonal edges that run farther down the front of the chest.
    crew=[]; deep=[]
    for rad in (3,5):
        offsets=np.arange(1,rad+1); jj=offsets/rad
        for depth in (2,4,7,10):
            for sy in (1,3):
                yy=np.rint(sy+depth*(1.-jj**2)).astype(int)
                ll=13-offsets; rr=14+offsets
                dl=x[:,yy+1,ll]-x[:,yy-1,ll]; dr=x[:,yy+1,rr]-x[:,yy-1,rr]
                paired=.5*(abs((dl+dr).mean(1))+.6*np.minimum(abs(dl.mean(1)),abs(dr.mean(1))))
                if depth<=4: crew.append(paired)
                else: deep.append(paired)
    broad=ramp((np.max(crew,axis=0)-.7*np.max(deep,axis=0))/(.16+strips(x,3,15,8,20)),.02,.36)
    # Heavy coats tend to have both a structured upper and broad, bulky
    # hips. A lightweight shirt can have a placket without either.
    longcut=ramp(base-np.maximum(tee,tank),-.20,.33)
    shortcut=1.-longcut
    bulk=ramp(hem/w,.55,.83)*(1.-drape)
    wornopen=centerline*np.maximum(lapel,hood)
    cottonprint=weave*np.maximum(sideprint,vertical)
    look[:,1]+=.36*broad+.16*knitband*(1.-centerline)-.18*bulk
    look[:,2]+=.32*wornopen+.25*bulk*longcut-.34*shirtcuff*(.25+.75*tail)-.22*cottonprint-.19*broad
    look[:,3]+=.28*vertical*centerline+.20*shirtcuff*longcut+.30*cottonprint-.25*shortcut*(1.-centerline)*(1.-tail)
    look[:,0]+=.35*shortcut*(1.-centerline)+.20*shortcut*broad
    upperform+=.32*shortcut*(.45+.55*broad)*(1.-bodycurve)+.24*longcut*drape
    # The neckline of a coat rises, and a coat is worn open or fastened in
    # front. These cues can occur on a shirt too; a dropped, curved hem,
    # shortened sleeves and woven fabric then favor the shirt instead.
    look[:,2]+=.29*hood+.20*centerline*(1.-knitband)+.16*ramp(aspect,.84,.64)*(1.-shirtcuff)
    look[:,3]+=.21*shirtcuff*tail-.16*centerline*hood-.22*graphic
    look[:,0]+=.20*shortsleeve+.15*neckline-.33*vertical*centerline
    look-=look.max(1)[:,None]
    for j,label in enumerate((0,2,4,6)):
        sc[:,label]=upperform+look[:,j]
    # Treat the three kinds of footwear as one family too. Decide between
    # open and closed construction first, and ankle height vs a low vamp
    # second. All cues are dimensionless relative to the query shoe.
    shoefamily=np.maximum.reduce([sandal,sneaker,boot])+sh+.30
    openstyle=1.35*openfoot+.45*coherent+.28*flatness+.21*ramp(aspect,2.7,3.6)+.36*(sandal-closedfoot)+.20*ramp(fill,.52,.30)
    shutstyle=.91+.23*ramp(fill,.42,.65)+.14*tongueup*(1.-notch)
    bootstyle=(boot-sneaker)*.45+1.28*ramp(aspect,2.45,1.30)+.68*shaft-.72*tongueup+.42*ramp(tilt,.24,.62)
    # A strap enclosing an actual patch of background creates both
    # horizontal and vertical intervals with area. A printed line on a
    # sneaker usually has a long interval in only one direction.
    inner=rowhole&colhole
    interior=morph(inner.astype(np.float32),np.minimum)
    eye=ramp(interior.sum((1,2))/(w*h),.002,.028)
    aperture=ramp(inner.sum((1,2))/(w*h),.021,.10)
    # Detect a suspended strap across a gap even when only its diagonal
    # arms bracket the void. This also handles wedge-heeled sandals.
    fine=x>.065
    gap2=~(fine[:,:-1,:-1]|fine[:,1:,:-1]|fine[:,:-1,1:]|fine[:,1:,1:])
    flap=np.maximum.accumulate(fine,axis=1)[:,:-1,:-1]&np.maximum.accumulate(fine[:,::-1],axis=1)[:,::-1,:][:,1:,1:]
    suspended=ramp((gap2*flap).sum((1,2))/(w*h),.015,.085)
    nearheel=(xs>left[:,None]+.74*w[:,None])&(xs<right[:,None]-.025*w[:,None])
    rearheight=(openmask.sum(1)*nearheel).sum(1)/np.maximum(1,nearheel.sum(1))
    lowback=ramp(rearheight/h,.52,.24)*ramp(aspect,1.55,2.45)
    openstyle+=.46*aperture+.48*eye+.51*suspended+.51*lowback+.29*slender
    # Corroborate a boot shaft by its broad elevated top *and* contrast
    # with the low toe. A peaked tongue and a thick toe favor a sneaker.
    crown=(topcols<=(top+.23*h)[:,None])&(cols>=2)
    plateau=ramp((crown*nearheel).sum(1)/np.maximum(1,nearheel.sum(1)),.38,.88)
    bootstyle+=.31*plateau*shaft-.37*(1.-plateau)*tongueup-.33*ramp(toe/h,.22,.50)-.22*ramp(aspect,1.63,2.19)
    shoeclass=np.where(openstyle>shutstyle,5,np.where(bootstyle>.84,9,7))
    for lab in (5,7,9):
        sc[:,lab]=np.where(shoeclass==lab,shoefamily,-100.)
    return np.argmax(sc,axis=1)


def predict(x):
    result=np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),64):
        # Work in double precision; reductions never pool different photos.
        work=np.ascontiguousarray(x[start:start+64,0],dtype=np.float64)
        result[start:start+64]=classify(work)
    return result
