import numpy as np

def unit(v):
    return v / np.sqrt(np.maximum(np.sum(v*v,axis=1,keepdims=True),.01))

def describe(mask):
    # Anti-aliased occupancy and two signed components of the contour normal.
    n=len(mask)
    m=mask.reshape(n,14,2,14,2).mean(axis=(2,4))
    p=np.pad(m,((0,0),(1,1),(1,1)))
    dx=(p[:,1:-1,2:]-p[:,1:-1,:-2])*.5
    dy=(p[:,2:,1:-1]-p[:,:-2,1:-1])*.5
    area=unit((m-.5).reshape(n,-1))
    normal=unit(np.concatenate((dx.reshape(n,-1),dy.reshape(n,-1)),axis=1))
    return np.rint(512*np.concatenate((.89*area,.456*normal),axis=1)).astype(np.float32)

def blur(x):
    p=np.pad(x,((0,0),(1,1),(1,1)),mode='edge')
    a=(p[:,:-2]+2*p[:,1:-1]+p[:,2:])*.25
    return (a[:,:,:-2]+2*a[:,:,1:-1]+a[:,:,2:])*.25

def morph(x,op):
    p=np.pad(x,((0,0),(1,1),(1,1)),mode='edge')
    a=op(op(p[:,:-2],p[:,1:-1]),p[:,2:])
    return op(op(a[:,:,:-2],a[:,:,1:-1]),a[:,:,2:])

def bounds(profile):
    yes=profile>np.minimum(.65,profile.max(1,keepdims=True)*.095)
    a=np.argmax(yes,axis=1)
    b=27-np.argmax(yes[:,::-1],axis=1)
    return a,np.maximum(a,b)

def resize(x,l,r,t,b):
    f=(np.arange(28,dtype=np.float32)+.5)/28
    cx=np.clip(l[:,None]-.5+(r-l+1)[:,None]*f,0,27)
    cy=np.clip(t[:,None]-.5+(b-t+1)[:,None]*f,0,27)
    xx=cx.astype(np.int32); yy=cy.astype(np.int32)
    fx=(cx-xx).astype(np.float32)[:,None,:]; fy=(cy-yy).astype(np.float32)[:,:,None]
    k=np.arange(len(x))[:,None,None]; ya=yy[:,:,None]; xa=xx[:,None,:]
    yb=np.minimum(27,ya+1); xb=np.minimum(27,xa+1)
    return ((1-fy)*((1-fx)*x[k,ya,xa]+fx*x[k,ya,xb])+fy*((1-fx)*x[k,yb,xa]+fx*x[k,yb,xb]))

def sat(x):
    return np.clip(x,0,1)

def rules(p,m,aspect):
    n=len(p); s=np.zeros((n,10),np.float32)
    # Coordinates here are fractions of the observation's own bounding box.
    # Measure a persistent front opening, not just any single shadow or logo.
    linear=p
    p=np.power(np.maximum(0,p),.8)
    cloth=p[:,7:24,:]
    light=np.maximum(.16,cloth[:,:,8:20].mean((1,2)))
    cloth=cloth/light[:,None,None]
    def line(z):
        step=z[:,:,4:]-z[:,:,:-4]
        ridge=(z[:,:,4:]+z[:,:,:-4])*.5-z[:,:,2:-2]
        return (np.abs(step.mean(1))*.60+np.abs(ridge.mean(1))*.85+.35*np.abs(ridge).mean(1))
    lines=line(cloth); middle=lines[:,9:15].max(1)
    side=.5*(lines[:,5:9].max(1)+lines[:,15:19].max(1))
    seam=sat((middle-.50*side-.055)*3.6)
    # Regular fabric pattern and small-scale detail (measured away from the silhouette).
    core=cloth[:,1:-1,7:21]
    gx=np.abs(core[:,:,2:]+core[:,:,:-2]-2*core[:,:,1:-1]).mean((1,2))
    gy=np.abs(core[:,2:,:]+core[:,:-2,:]-2*core[:,1:-1,:]).mean((1,2))
    detail=sat((np.minimum(gx,gy)-.105)*3.7)
    busy=sat((.5*(gx+gy)-.11)*2.5)
    # Do the two sloping revers outline a V that extends onto the chest?
    q=p/light[:,None,None]; pairs=[]
    for shift in [-1,0,1]:
        for end in [7,10,13]:
            y=np.arange(2,end+1); off=np.rint(1.4+(end-y)*.52).astype(np.int32)
            a=13+shift-off; b=14+shift+off
            left=q[:,y,a-1]-q[:,y,a+1]; right=q[:,y,b+1]-q[:,y,b-1]
            pairs.append(np.maximum(np.minimum(left.mean(1),right.mean(1)),np.minimum(-left.mean(1),-right.mean(1))))
    lapel=sat((np.maximum.reduce(pairs)-.13)*2.6)
    # Standing collar versus a broad crew neckline. A narrow upper tower must widen below.
    width=m.mean(2)
    stand=sat((width[:,4:8].mean(1)-width[:,:3].mean(1)-.22)*2.8)
    cowl=sat((width[:,1:4].mean(1)-.39)*3.5)
    # A rounded shirttail: a lower centre compared with the flanks of the body.
    yy=np.arange(28)[None,:,None]
    bottom=np.max(np.where(m>.47,yy,-1),axis=1)
    roundhem=sat((bottom[:,11:17].mean(1)-.5*(bottom[:,7:10].mean(1)+bottom[:,18:21].mean(1))-.7)/3.5)
    # A zipper/placket produces a slender coherent edge down the front.
    # Test both transition polarities at multiple widths, allowing the line to
    # slant by a few pixels from chest to hem. Compare against the same filter
    # on the side panels to cancel all-over stripes and shadow contrast.
    y=np.arange(6,24); openings=[]; sidelines=[]
    for run in [-2,0,2]:
        axis=np.arange(6,22)[None,:]+np.rint((y-14)*run/17).astype(np.int32)[:,None]
        for radius in [1,2,3]:
            mid=q[:,y[:,None],axis]; ll=q[:,y[:,None],axis-radius]; rr=q[:,y[:,None],axis+radius]
            step=(ll-rr).mean(1); line=((ll+rr)*.5-mid).mean(1)
            lineval=np.maximum(np.abs(line),np.abs(step)*.55)
            openings.append(lineval[:,5:11].max(1)); sidelines.append((lineval[:,:4].max(1)+lineval[:,12:].max(1))*.5)
    zipval=np.maximum.reduce(openings); other=np.maximum.reduce(sidelines)
    fastener=sat((zipval-.48*other-.035)*3.6)
    # Lapel or shirt collar surrounds the neck and changes tone locally, even
    # on a plain garment. A longer coat leaves cloth *below* the cuffs.
    neckpatch=q[:,2:9,9:19]; neckcontrast=np.abs(neckpatch-blur(neckpatch)).mean((1,2))
    collar=sat((neckcontrast-.04)*5)
    cuffs=np.maximum(bottom[:,:5].max(1),bottom[:,23:].max(1))
    coatlength=sat((bottom[:,10:18].mean(1)-cuffs-1)/5)
    # Tee sleeves end abruptly high up; under the armpits the silhouette of
    # an untucked tee stays close to the same width all the way to the hem.
    row=(m>.48); span=(row.shape[2]-np.argmax(row,2)-np.argmax(row[:,:,::-1],2))/28
    shortarm=sat((span[:,3:10].max(1)-span[:,12:18].mean(1)-.07)*6)
    flare=span[:,22:26].mean(1)-span[:,12:18].mean(1)
    straight=sat(1-np.abs(flare)*7)
    block=shortarm*straight
    # Thin, loose cloth develops fine vertical folds over the lower torso.
    # Use the outer panels, as opposed to the single central coat fastening.
    panel=q[:,12:24,6:22]; wav=panel[:,:,2:] + panel[:,:,:-2]-2*panel[:,:,1:-1]
    folds=sat((np.abs(wav.mean(1)).mean(1)-.045)*4)
    # A graphic printed centrally on an otherwise plain tee is not plaid.
    ht=np.abs(q-blur(q)); printval=ht[:,5:17,9:19].mean((1,2)); outside=ht[:,14:23,7:21].mean((1,2))
    graphic=sat((printval-1.5*outside-.015)*7)
    # A long coat and a blouse may both have buttons. Distinguish a single
    # nearly uninterrupted closure from a row of little shirt buttons or a
    # placket on plaid fabric. Use agreement of three separate vertical bands.
    def bandlines(z):
        a=z[:,:,:-4]; b=z[:,:,2:-2]; c=z[:,:,4:]
        return np.stack(((a-c).mean(1)*.60,((a+c)*.5-b).mean(1)),axis=2)
    bands=np.stack([bandlines(q[:,i:i+6]) for i in (6,12,18)],axis=1)
    pos=np.min(bands,axis=1); neg=np.min(-bands,axis=1)
    persistence=np.maximum(pos,neg).max(2)
    main=persistence[:,9:15].max(1); sides=(persistence[:,5:9].max(1)+persistence[:,15:19].max(1))*.5
    continuous=sat((main-.55*sides-.022)*5)
    # Knit tops tend to have a broad, shallow neck opening; a coat puts
    # collar material high up in the centre. Read the relative height of
    # the neck and both shoulders, not just the total width of the top row.
    rooftop=np.argmax(m>.5,axis=1)
    neckheight=(rooftop[:,6:10].mean(1)+rooftop[:,18:22].mean(1))*.5-rooftop[:,12:16].mean(1)
    highneck=sat((neckheight-1)/3); scoop=sat((1-neckheight)/3)
    # Detect checked or floral fabric by energy over BOTH flanks and multiple
    # rows. In comparison a printed sweatshirt often has a flat back panel,
    # sleeves and hem around a single graphic on the chest.
    checker=ht[:,8:23,6:22]
    allover=sat((np.minimum(checker[:,:,:6].mean((1,2)),checker[:,:,10:].mean((1,2)))-.045)*7)
    # All evidence below is relative to the chosen garment family. In
    # particular a shadow on a skirt or split between legs isn't a zipper.
    s[:,0]+=.012-.039*seam-.018*lapel+.076*block+.061*graphic-.021*folds+.010*scoop
    # Paired, touching banks of a zip can be VERY low contrast. A dotted
    # row of buttons instead makes several compact local extrema surrounded
    # by similar cloth. Look at each row in relation to its neighbours.
    sm=blur(q); marks=np.abs(q-sm)
    dots=marks[:,7:24,11:17].max(2)
    flanks=(marks[:,7:24,6:11].max(2)+marks[:,7:24,17:22].max(2))*.5
    rows=np.maximum(0,dots-flanks*.8)
    button=sat((np.minimum(rows[:,:8].max(1),rows[:,8:].max(1))-.038)*4)
    # Wide jacket cuffs and thick body panels leave little visible space
    # between arm and body; relaxed sweater sleeves taper toward the wrist.
    wleft=m[:,17:24,:8].sum(2); wright=m[:,17:24,20:].sum(2)
    filled=sat(((wleft+wright).mean(1)/16-.56)*3)
    # A vertical crease of the cloth can be as strong as a front fastening.
    # Use a line-vs-parallel-lines contrast ratio to discount such folds.
    prominence=sat((zipval-other*.87)/(.06+other))
    outer=.48*fastener*(.38+.62*prominence)+.38*continuous+.14*seam
    button=button*sat(1-allover)*(.35+.65*prominence)
    # Thin/fluid blouses have curved hems, and a thinner draping sleeve
    # ending above the wrist. Sleeve length is apparent on both sides.
    mediumarm=sat((23-cuffs)*.35)*sat((cuffs-9)*.35)
    openfront=outer*(.8+.2*filled)
    s[:,2]+=.012-.080*openfront-.030*lapel-.012*collar-.014*detail-.018*roundhem-.034*coatlength+.016*scoop-.012*button
    s[:,4]+=-.012+.098*openfront+.037*lapel+.012*stand+.016*highneck+.040*coatlength-.044*allover-.027*roundhem-.015*folds+.016*button
    s[:,6]+=-.014+.028*fastener+.018*lapel+.026*detail+.056*allover+.045*roundhem+.044*folds+.013*collar+.014*button-.027*graphic+.023*mediumarm
    local=np.array([0,2,4,6]); dressgroup=s[:,local].copy()
    s[:,local]-=.90*np.max(dressgroup,axis=1,keepdims=True)
    s[:,3]+=.050*sat(flare*4)*sat((-.03-aspect)*2)-.065*block-.030*sat((aspect+.16)*3)
    # Distinguish rectangular tees from straight shift dresses of similar
    # outline using the length from clavicle to hem in shoulder-width units.
    # Front-view tees end around the hip; a dress extends past the seat.
    trunk=span[:,13:21].mean(1)*np.exp(aspect)
    splitlow=(m[:,16:25,6:10].mean((1,2))+m[:,16:25,18:22].mean((1,2)))*.5-m[:,16:25,12:16].mean((1,2))
    s[:,3]+=.038*sat((.52-trunk)*5)-.075*sat((trunk-.54)*5)-.047*graphic-.074*sat((splitlow-.12)*3)
    # A sleeve ends at an outward-pointing corner on BOTH sides. Its extent
    # as a fraction of total torso length separates a tee from a shift dress.
    tip=np.argmax(span[:,2:14],axis=1)+2
    tips=sat((span[:,2:14].max(1)-span[:,15:21].mean(1)-.10)*5)
    tshape=tips*sat((tip-3)/4)*sat((trunk-.52)*6)*(1-sat(flare*5))
    s[:,0]+=.066*tshape; s[:,3]-=.057*tshape
    # Split legs give evidence even when one leg leans inwards.
    low=m[:,18:26]; split=(low[:,:,4:10].mean((1,2))+low[:,:,18:24].mean((1,2)))*.5-low[:,:,12:16].mean((1,2))
    s[:,1]+=.027*sat((split-.2)*2)
    # Negative spaces bounded above, below and to both sides are characteristic
    # of straps. Ignore the borders and the top, where sneaker laces can be dark.
    above=np.maximum.accumulate(m,axis=1); below=np.maximum.accumulate(m[:,::-1],axis=1)[:,::-1]
    left=np.maximum.accumulate(m,axis=2); right=np.maximum.accumulate(m[:,:,::-1],axis=2)[:,:,::-1]
    hole=np.maximum(0,np.minimum(np.minimum(above,below),np.minimum(left,right))-m)
    air=hole[:,8:24,2:26].mean((1,2)); straps=sat((air-.044)*6)
    # Open toes, T-straps and slingbacks may not enclose a hole in 2-D, but
    # they remove material from the hull of a shoe. Measure column solidity,
    # sparse regions between the straps, and upper material over the footbed.
    level=np.maximum(.07,np.sort(linear.reshape(n,-1),axis=1)[:,675])
    ink=sat((linear/level[:,None,None]-.055)*4.2)
    aircol=np.minimum(np.maximum.accumulate(ink,1),np.maximum.accumulate(ink[:,::-1],1)[:,::-1])-ink
    gap=sat((aircol[:,3:24,2:26].mean((1,2))-.045)*7)
    # How much of the toe is bare? Only consider the toe, not the back end,
    # and allow the more massive ankle/heel to be at either side.
    flip=m[:,:14,:14].sum((1,2))>m[:,:14,14:].sum((1,2))
    oriented=np.where(flip[:,None,None],ink[:,:,::-1],ink)
    profile=oriented.sum(1)/28
    toeheight=profile[:,2:10].mean(1)
    sparse=sat((.51-ink.mean((1,2)))*4.4)
    baretoe=sat((.26-toeheight)*5.5)
    # In a sneaker the tongue then the low collar make a notch; a boot has
    # a much higher continuous shaft, normally almost vertical at the front.
    topedge=np.argmax(oriented>.42,axis=1)
    stepheight=sat((np.median(topedge[:,5:11],axis=1)-np.median(topedge[:,18:24],axis=1)-10)/9)
    # If a strap opens sideways it leaves an indentation instead of an
    # enclosed hole. The convex hull of a column-height graph bridges it.
    # Account for both the arch under the foot and gaps above the sole.
    solid=oriented>.42
    roof=np.argmax(solid,axis=1).astype(np.float32)
    floor=27-np.argmax(solid[:,::-1],axis=1).astype(np.float32)
    roof=np.where(solid.any(1),roof,27); floor=np.where(solid.any(1),floor,0)
    bridged=roof.copy(); base=floor.copy()
    for spanlen in (4,8,12,16,20,24):
        for part in (.25,.5,.75):
            cut=int(part*spanlen); remain=spanlen-cut
            bridged[:,cut:28-remain]=np.minimum(bridged[:,cut:28-remain],(1-part)*roof[:,:28-spanlen]+part*roof[:,spanlen:])
            base[:,cut:28-remain]=np.maximum(base[:,cut:28-remain],(1-part)*floor[:,:28-spanlen]+part*floor[:,spanlen:])
    emptytop=(roof-bridged)[:,2:26]; arch=(base-floor)[:,3:25]
    sunk=sat((emptytop.mean(1)-1.35)*.62); heel=sat((arch.mean(1)-.8)*.65)
    column=aircol[:,3:24].mean(1)
    biggap=sat((np.sort(column[:,2:26],axis=1)[:,-7:-2].mean(1)-.105)*4.0)
    s[:,5]+=.25*straps+.21*gap+.24*sparse+.115*baretoe+.16*sunk+.14*biggap+.078*heel-.046
    s[:,7]+=.010-.102*straps-.085*gap-.125*sparse-.040*baretoe-.055*stepheight-.052*heel
    s[:,9]+=.005-.098*straps-.105*gap-.110*sparse+.034*stepheight-.044*sunk-.046*biggap
    # A bag with a handle has most of its interior space above the pouch.
    grip=hole[:,1:12,:].mean((1,2)); pouch=m[:,15:24,4:24].mean((1,2))
    s[:,8]+=.082*sat((grip-.07)*5)*sat((pouch-.62)*3)
    # A clutch has a broad closed top without shoulders, whereas a handbag
    # can have a narrow top but a near-rectangular base below the handle.
    rectangle=np.minimum(width[:,4:9].mean(1),width[:,19:24].mean(1))
    s[:,8]+=.047*sat((rectangle-.82)*8)*sat((aspect+.18)*3)
    return s

def batch(images):
    raw=images[:,0]
    # Black fabric still has a visible outline. Use a soft dark-pixel test and
    # a one-pixel closing, keeping genuine spaces between straps and sleeves.
    scale=np.sort(raw.reshape(len(raw),-1),axis=1)[:,735,None,None]
    weak=sat((raw-.012)/(.015+.105*scale)); firm=sat((raw-.025)/(.028+.22*scale))
    mask=weak*.73+firm*.27
    closed=morph(morph(mask,np.maximum),np.minimum)
    mask=.64*mask+.36*closed
    l,r=bounds(mask.sum(1)); t,b=bounds(mask.sum(2))
    aspect=np.log((r-l+1)/(b-t+1)).astype(np.float32)
    m=resize(mask,l,r,t,b); picture=resize(raw,l,r,t,b)
    f=describe(m)
    # Look at both the full footwear pattern and the filled side profile.
    above=np.maximum.accumulate(m,axis=1); below=np.maximum.accumulate(m[:,::-1],axis=1)[:,::-1]
    left=np.maximum.accumulate(m,axis=2); right=np.maximum.accumulate(m[:,:,::-1],axis=2)[:,:,::-1]
    boundary=describe(np.minimum(np.minimum(above,below),np.minimum(left,right)))
    footstart=0
    bank=CONSTANTS['patterns']; rat=CONSTANTS['aspect']; cost=CONSTANTS['cost']; ends=CONSTANTS['ends']
    adjustment=rules(picture,m,aspect)
    shape=np.empty_like(adjustment)
    start=0
    for k in range(10):
        last=int(ends[k]); d=f @ bank[start:last].T
        if k in (5,7,9):
            footend=footstart+last-start
            contour=boundary @ CONSTANTS['outlines'][footstart:footend].T
            d=.55*d+.45*contour; footstart=footend
        error=np.abs(aspect[:,None]-rat[None,start:last]); d=d*(1/262144)-.26*error-cost[None,start:last]
        # Require support from several similar sewing patterns as well as the
        # single best correspondence; one lucky extreme pose is weak evidence.
        best=np.partition(d,-8,axis=1)[:,-8:]
        shape[:,k]=.5*(best.max(1)+best.mean(1)); start=last
    scores=shape+adjustment
    clothing=[0,2,4,6]; shoes=[5,7,9]
    # The family comparison should reflect the garment's cut rather than its
    # surface detail. Keep a little evidence for a particularly clear detail.
    bestcut=shape[:,clothing].max(1); bestscore=scores[:,clothing].max(1)
    scores[:,clothing]+=.82*(bestcut-bestscore)[:,None]
    gain=scores[:,shoes].max(1)-shape[:,shoes].max(1)
    scores[:,shoes]-=.85*np.maximum(0,gain-.045)[:,None]
    return np.argmax(scores,axis=1)

def predict(x):
    result=np.empty(len(x),dtype=np.int64)
    for i in range(0,len(x),64):
        result[i:i+64]=batch(x[i:i+64])
    return result
