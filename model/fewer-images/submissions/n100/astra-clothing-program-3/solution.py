import numpy as np

def clip(a):
    return np.clip(a, 0., 1.)

def blur(a):
    p = np.pad(a, ((0,0),(1,1),(1,1)), mode='constant')
    v = p[:,:-2,:]+2*p[:,1:-1,:]+p[:,2:,:]
    return (v[:,:,:-2]+2*v[:,:,1:-1]+v[:,:,2:])*(1/16)

def grid(a, xx, yy):
    n = np.arange(len(a))[:,None,None]
    xx = np.clip(xx, 0, 27)
    yy = np.clip(yy, 0, 27)
    ix = xx.astype(np.int32)
    iy = yy.astype(np.int32)
    fx = (xx-ix).astype(np.float32)
    fy = (yy-iy).astype(np.float32)
    jx = np.minimum(ix+1,27)
    jy = np.minimum(iy+1,27)
    return ((1-fy)*((1-fx)*a[n,iy,ix]+fx*a[n,iy,jx]) + fy*((1-fx)*a[n,jy,ix]+fx*a[n,jy,jx]))

def framing(m):
    extents = []
    for axis in (1,2):
        support = m.sum(axis=axis)
        present = support > np.maximum(.55, .032*support.max(axis=1,keepdims=True))
        first = np.argmax(present,axis=1).astype(np.float32)
        last = 27-np.argmax(present[:,::-1],axis=1).astype(np.float32)
        extents.append((first,last))
    (l,r),(t,b) = extents
    w = np.maximum(2, r-l+1)
    h = np.maximum(2, b-t+1)
    z = (np.arange(28,dtype=np.float32)+.5)/28-.5
    xx = (l+r)[:,None,None]*.5 + w[:,None,None]*z[None,None,:]
    yy = (t+b)[:,None,None]*.5 + h[:,None,None]*z[None,:,None]
    return xx, yy, w/h

def block2(a):
    n,h,w = a.shape
    return a.reshape(n,h//2,2,w//2,2).mean(axis=(2,4))

def margins(a):
    # Integrate the probability of still being outside on a scan from each end.
    return np.minimum(.66, (1-clip(np.cumsum(a,axis=2))).mean(axis=2))

def describe(a, aspect):
    p=block2(a)
    q=block2(p)
    parts=[(p/14).reshape(len(a),196),(.65*q/7).reshape(len(a),49)]
    for b in (a,a.transpose(0,2,1)):
        for c in (b,b[:,:,::-1]):
            e=margins(c).reshape(len(a),14,2).mean(axis=2)
            parts.append(.18*e)
    parts.append((.58*np.log(np.maximum(.06,aspect)))[:,None])
    return np.ascontiguousarray(np.concatenate(parts,axis=1),dtype=np.float32)

# The distance is evaluated in fixed-size blocks so arithmetic for one row
# does not depend on how the caller groups images. Convert to integer cost
# units only after summing a complete per-pattern squared distance.
norms = {k:np.sum(v*v,axis=1) for k,v in CONSTANTS.items()}

def match(v):
    fixed = np.zeros((64,v.shape[1]),np.float32)
    fixed[:len(v)] = v
    sq = np.sum(fixed*fixed, axis=1)
    result = {}
    for k,table in CONSTANTS.items():
        d = (sq[:,None]+norms[k][None,:]-2*(fixed@table.T))[:len(v)]
        # A few nearby patterns mitigate chance coincidences of individual
        # corners. This is a query statistic, not a constructed prototype.
        near = np.partition(d,3,axis=1)[:,:4]
        cost = .6*near.min(axis=1)+.4*near.mean(axis=1)
        result[k] = np.rint(cost*100000)*.01
    return result

def unit(v,k,span):
    return clip((v-k)/span)

def imaging(raw):
    # Avoid basing the exposure on a single white pixel of piping or a button.
    peak = np.partition(raw.reshape(len(raw),784),-32,axis=1)[:,-32,None,None]
    peak = np.maximum(peak,.2*raw.max(axis=(1,2),keepdims=True))
    rel = np.minimum(2,raw/np.maximum(peak,1.e-5))
    soft = clip((rel-.024)/.115)
    smooth = blur(soft)
    soft = .72*soft+.28*clip((smooth-.14)/.65)
    xx,yy,aspect = framing(soft)
    a = grid(soft,xx,yy)
    im = grid(rel,xx,yy)
    return a,im,aspect

def trimmean(a):
    return a.mean(axis=(1,2))

def detailing(a, im):
    # Work in the object's own coordinate system. Compare local contrast,
    # not photographic exposure. Examine the two *sides* of a possible
    # closure; a centre front can be a light line, dark line or step edge.
    n=len(im)
    textile=im[:,7:23,9:19].mean(axis=(1,2),keepdims=True)
    light=np.minimum(2.8,im/(textile+.065))
    sm=blur(light)
    placket=np.zeros(n); continuous=np.zeros(n)
    dy=np.arange(7,24)
    for mid in (12,13,14,15):
        for tilt in (-1,0,1):
            cx=mid+np.rint(tilt*(dy-15)/8).astype(np.int32)
            col=light[:,dy,cx]*.72+sm[:,dy,cx]*.28
            l=(sm[:,dy,cx-2]+sm[:,dy,cx-3])*.5
            r=(sm[:,dy,cx+2]+sm[:,dy,cx+3])*.5
            bend=col-.5*(l+r)
            step=r-l
            strength=np.maximum(np.abs(bend)*.95,np.abs(step)*.62)
            placket=np.maximum(placket,.60*strength.mean(axis=1)+.40*np.maximum(abs(bend.mean(axis=1)),.62*abs(step.mean(axis=1))))
            continuous=np.maximum(continuous,np.mean(strength>.13,axis=1))
    # Offset the centre-line response by the contrast expected on any fold
    # of the fabric. This makes a plain coat with a zip quite different
    # from an entirely patterned sweater.
    off=np.concatenate((abs(sm[:,7:24,7:10]-sm[:,7:24,10:13]),abs(sm[:,7:24,15:18]-sm[:,7:24,18:21])),axis=2)
    baseline=trimmean(off)
    fastening=unit(placket-.48*baseline,.065,.25)*(.55+.45*continuous)
    dx=abs(light[:,:,1:]-light[:,:,:-1]); dy=abs(light[:,1:,:]-light[:,:-1,:])
    tex=.5*(trimmean(dx[:,7:24,8:19])+trimmean(dy[:,7:23,8:20]))
    printed=unit(tex,.11,.22)
    vertical=trimmean(dx[:,7:23,7:20]); horizontal=trimmean(dy[:,7:23,7:21])
    knitstripe=unit(horizontal-vertical,.025,.12)
    # Neckline: a round or V-neck tends to expose a bright or dark crescent.
    # A collar or a hood rises above the sloping line of the shoulders.
    top=margins(a.transpose(0,2,1))
    crown=.5*(top[:,6:9].mean(axis=1)+top[:,19:22].mean(axis=1))
    crown-=top[:,11:17].mean(axis=1)
    raised=unit(crown,.07,.16)
    scooped=unit(-crown,-.01,.12)
    neck=sm[:,1:6,11:17]
    neckcontrast=unit(np.mean(abs(neck-neck.mean(axis=(1,2),keepdims=True)),axis=(1,2)),.11,.30)
    # Localised central graphic, compared with its surrounding fabric.
    upper=sm[:,7:17,9:19]; sided=.5*(sm[:,7:17,7:9].mean(axis=2)+sm[:,7:17,19:21].mean(axis=2))
    lower=sm[:,18:23,9:19]
    graphic=unit(trimmean(abs(upper-sided[:,:,None]))-.45*lower.std(axis=(1,2)),.13,.33)
    # Separate cut (silhouette) from visible details such as a jersey cuff,
    # woven collar or button strip. The latter do not exist on our patterns.
    return fastening,printed,knitstripe,raised,scooped,neckcontrast,graphic,continuous,placket,baseline

def tailoring(a,im,aspect):
    # A raised V of lapels is different from a single round-neck opening.
    # Use two independent sloping edge segments with paired left/right
    # support; either contrast polarity is possible in catalogue lighting.
    n=len(a)
    cloth=im[:,7:20,8:20].mean(axis=(1,2),keepdims=True)
    light=np.minimum(3,im/(cloth+.08))
    sm=.6*blur(light)+.4*light
    lap=np.zeros(n); stand=np.zeros(n)
    for mid in (13,14):
        for width in (3,5,7):
            for depth in (5,9,13):
                y=np.arange(2,depth+2)
                side=np.rint(width*(1-(y-2)/(depth+1.))).astype(np.int32)+1
                le=sm[:,y,mid-side]-sm[:,y,mid-side+2]
                ri=sm[:,y,mid+side]-sm[:,y,mid+side-2]
                coupled=.5*(le+ri)-.25*abs(le-ri)*np.sign(le+ri)
                lap=np.maximum(lap,abs(coupled.mean(axis=1)))
    lapel=unit(lap,.17,.34)
    # Raised centre on an insulating stand collar; the material in a crew
    # neck, in contrast, usually has a notch *between* two higher points.
    z=margins(a.transpose(0,2,1))
    notch=z[:,12:16].mean(axis=1)-.5*(z[:,8:11].mean(axis=1)+z[:,17:20].mean(axis=1))
    crew=unit(notch,.012,.065)
    peak=unit(-notch,.018,.073)
    # Estimate cuff level without assuming a particular body width. Read
    # the outside two slivers of each sleeve; torso tails below cuffs are
    # characteristic of an overcoat, a tunic, or a dress with sleeves.
    h=margins(a[:,::-1,:].transpose(0,2,1))
    outer=.5*(h[:,2:5].min(axis=1)+h[:,23:26].min(axis=1))
    center=h[:,11:17].mean(axis=1)
    tail=outer-center
    longer=unit(tail,.075,.22)
    cropped=unit(-tail,.035,.18)
    # Room at the waist, shape of the hem, and sleeve reach. Short sleeves
    # broaden a tee above a fairly straight, untucked torso. A dress is
    # usually longer per unit hem width or flares out below the waist.
    occupancy=a.mean(axis=2)
    belly=occupancy[:,12:16].mean(axis=1)
    hips=occupancy[:,21:25].mean(axis=1)
    flare=unit(hips-belly,.05,.20)
    chest=occupancy[:,5:12].max(axis=1)
    short=unit(tail,.36,.24)*unit(chest-hips,.075,.21)
    squat=unit(hips*aspect,.51,.22)
    # An all-over fabric print or plaid (unlike a locally printed tee) makes
    # edge energy persist in all quadrants of the torso.
    detail=abs(light-blur(light))
    panels=np.stack([trimmean(detail[:,b:b+7,c:c+5]) for b in (7,16) for c in (8,15)],axis=1)
    allover=unit(np.sort(panels,axis=1)[:,1],.038,.11)
    bold=unit(panels.max(axis=1),.15,.16)*allover
    return lapel,crew,peak,longer,cropped,flare,short,squat,allover,bold

def fabric(a,im):
    # A jersey band runs all the way across the front. A shirt tail can be
    # curved too, but doesn't have a contrasting horizontal knit binding.
    cloth=im[:,8:23,8:20].mean(axis=(1,2),keepdims=True)
    light=np.minimum(3.,im/(.08+cloth))
    y=np.arange(28,dtype=np.float32)[None,:,None]
    cols=np.arange(8,20)
    end=27-np.argmax(a[:,::-1,8:20]>.48,axis=1)
    rows=np.arange(len(a))[:,None]
    hem=np.zeros(len(a))
    for depth in (1,2,3,4):
        p=light[rows,np.maximum(0,end-depth),cols]
        q=light[rows,np.maximum(0,end-depth-2),cols]
        change=p-q
        hem=np.maximum(hem,abs(change.mean(axis=1))-.32*change.std(axis=1))
    binding=unit(hem,.075,.23)
    # A neck-high layer rises between the shoulder seams; distinguish
    # a narrow collar from the broad, shallow arc of a pullover.
    z=margins(a.transpose(0,2,1))
    rise=.5*(z[:,5:9].mean(axis=1)+z[:,19:23].mean(axis=1))-z[:,11:17].mean(axis=1)
    shoulder=unit(rise,.095,.14)
    flat=unit(.13-rise,0,.10)
    # Coarse creases in lightweight cloth, after discounting a centre zip.
    sm=blur(light); broad=blur(blur(sm))
    stacks=[]
    for left in (7,15):
        region=broad[:,8:23,left:left+6]
        # Variation between panels that persists for several rows, as for
        # folds from a hanging shoulder or gathers below a blouse yoke.
        folds=region.mean(axis=1).std(axis=1)
        wav=np.mean(abs(region[:,2:]-region[:,:-2]),axis=(1,2))
        stacks.append(folds-.33*wav)
    folds=unit(.5*(stacks[0]+stacks[1]),.035,.12)
    # Search for dark-light alternation in both directions. Where the
    # pattern repeats across arms AND torso, it is not a central tee print.
    def pattern(v):
        x=(v-v.mean(axis=2,keepdims=True))
        u=(v-v.mean(axis=1,keepdims=True))
        return np.sqrt((x*x).mean(axis=(1,2))*(u*u).mean(axis=(1,2)))
    checks=np.minimum(pattern(light[:,6:15,8:20]),pattern(light[:,15:24,8:20]))
    weave=unit(checks,.045,.20)
    return binding,shoulder,flat,folds,weave

def neckstructure(a,im):
    # Measure two independent properties of a front neckline: how deep
    # the opening is and whether a lapel continues into a vertical edge.
    # Ratios to nearby cloth allow subtle detail on a nearly white shirt.
    cloth=im[:,7:21,8:20].mean(axis=(1,2),keepdims=True)
    sm=blur(im/(cloth+.06))
    n=len(a)
    op=np.zeros(n); deep=np.zeros(n); inline=np.zeros(n)
    for mid in (12,13,14,15):
        y=np.arange(1,13)
        wid=np.maximum(1,np.rint(5-.4*y).astype(np.int32))
        both=.5*(sm[:,y,mid-wid]+sm[:,y,mid+wid])
        centre=sm[:,y,mid]
        v=abs(centre-both)
        # Outside the product is not a contrast marking on its cloth.
        v*=np.minimum(a[:,y,mid-wid],a[:,y,mid+wid])*a[:,y,mid]
        op=np.maximum(op,v[:,:5].mean(axis=1))
        deep=np.maximum(deep,v[:,4:10].mean(axis=1))
        # Extract a long central crease at two scales. A zip produces
        # a line plus nearby shading, rather than merely a bright collar.
        l=sm[:,8:24,mid-3]; r=sm[:,8:24,mid+3]; c=sm[:,8:24,mid]
        lap=c-(l+r)*.5
        step=r-l
        stripe=np.maximum(abs(lap.mean(axis=1)),.64*abs(step.mean(axis=1)))
        inline=np.maximum(inline,stripe)
    return unit(op,.09,.27),unit(deep,.085,.25),unit(inline,.049,.21)

def recognize(raw):
    a,im,aspect=imaging(raw)
    # Orient a shoe toe-left for the side-profile tests. Retain mild
    # asymmetry on front-on garments rather than mirroring their support.
    yy=np.linspace(1,0,28,dtype=np.float32)[None,:,None]
    left=np.sum(a[:,:,:12]*yy,axis=(1,2)); right=np.sum(a[:,:,16:]*yy,axis=(1,2))
    flip=(left>right)[:,None,None]
    a=np.where(flip,a[:,:,::-1],a)
    im=np.where(flip,im[:,:,::-1],im)
    dist=match(describe(a,aspect))
    closure,printtex,stripes,raised,scoop,neck,graphic,persist,line,foldline=detailing(a,im)
    sym=trimmean(abs(a-a[:,:,::-1])); asym=unit(sym,.10,.25)
    # A flexible piece may hang askew, but shoes almost always rise more at
    # the back than at the toe. Patterns capture the detailed shape; these
    # features give stable evidence under imperfect segmentation.
    z=margins(a.transpose(0,2,1))
    toedrop=z[:,2:9].mean(axis=1)-z[:,18:25].mean(axis=1)
    shoe=unit(toedrop,.06,.31)
    hollow=trimmean((1-a[:,8:22,4:24])*clip(4*blur(a)[:,8:22,4:24]))
    # Count supported voids, e.g. below straps or inside a bag handle.
    am=a>.36
    enclosed=(np.maximum.accumulate(am,axis=1)&np.maximum.accumulate(am[:,::-1],axis=1)[:,::-1]&np.maximum.accumulate(am,axis=2)&np.maximum.accumulate(am[:,:,::-1],axis=2)[:,:,::-1])
    holes=enclosed*(1-a)
    strap=unit(trimmean(holes[:,5:23,2:26]),.027,.17)
    handle=unit(trimmean(holes[:,:16,4:24]),.07,.31)
    speckle=unit(hollow,.10,.20)
    high=unit(1/np.maximum(.01,aspect),.47,.25)
    narrow=unit(1/np.maximum(.01,aspect),1.32,.55)
    # A trouser crotch divides a pair of legs for much of their length.
    gap=trimmean(1-a[:,19:27,13:15]); thighs=.5*(trimmean(a[:,15:25,6:10])+trimmean(a[:,15:25,18:22]))
    bifid=unit(gap*thighs,.18,.52)
    # Hem shapes: a sweater waistband is short and straight, while woven
    # shirts can have round tails. A coat often extends well below the cuffs.
    hem=margins(a[:,::-1,:].transpose(0,2,1))
    midhem=hem[:,10:18].mean(axis=1)
    hemcurve=unit(.5*(hem[:,7:10].mean(axis=1)+hem[:,18:21].mean(axis=1))-midhem,.025,.11)
    # Sum negative-log-like evidence in units of silhouette mismatch.
    # Fixed subtype offsets express uncommon cuts (e.g., a long-sleeve tee).
    tee=np.minimum.reduce([dist['tee']-4,dist['tank']+3,dist['captee']+2,dist['longtee']+18])
    trouser=np.minimum.reduce([dist['jeans'],dist['slacks'],dist['shorts']+10])
    knit=np.minimum.reduce([dist['jumper']-2,dist['boxknit'],dist['highneck']+6,dist['cardigan']+9])
    dress=np.minimum.reduce([dist['daydress'],dist['sundress'],dist['sheath']+3,dist['sleevedress']+3])
    coat=np.minimum.reduce([dist['jacket'],dist['overcoat'],dist['parka']+2])
    sandal=np.minimum.reduce([dist['flatstrap'],dist['heelstrap'],dist['slide'],dist['wedge']+2])
    shirt=np.minimum.reduce([dist['buttonshirt']+4,dist['blouse']+7,dist['shortshirt']+8,dist['smock']+11])
    trainer=np.minimum(dist['trainer'],dist['hightop']+6)
    bag=np.minimum.reduce([dist['purse'],dist['tote'],dist['clutch'],dist['rucksack']+8])
    boot=np.minimum.reduce([dist['boot'],dist['heelboot'],dist['lowboot']+4])
    tee=np.minimum(tee,dist['jerseyfront'])
    coat=np.minimum(coat,dist['coatfront'])
    knit=np.minimum(knit,dist['knitfront'])
    shirt=np.minimum(shirt,dist['shirtfront']+4)
    tee+=9*closure-3*scoop-5*graphic+2*raised
    trouser+=8*(1-bifid)-13*bifid-2*narrow
    knit+=13*closure+4*raised-3*scoop-4*stripes+4*hemcurve
    dress+=4*closure-5*narrow-2*hemcurve
    coat+=6-10*closure-13*raised+5*scoop-3*neck+2*printtex
    shirt+=4-8*printtex-8*hemcurve-5*closure-3*neck+3*stripes
    sandal+=4-20*strap-8*speckle-5*shoe
    trainer+=24*strap+7*high-4*shoe
    bag+=11*shoe-16*handle+7*asym*(1-handle)
    boot+=17*strap+5*(1-high)-6*shoe
    lapel,crew,peak,longer,cropped,flare,short,squat,allover,bold=tailoring(a,im,aspect)
    tee-=18*short*squat*(1-.7*flare)
    dress+=13*short*squat-7*flare*narrow
    # Closures, lapels and a raised neckline are evidence about layering;
    # combine them with the cut rather than requiring any single cue.
    jacket=clip(.56*closure+.48*lapel+.62*peak+.32*longer-.38*crew)
    knit+=23*jacket+8*longer-7*crew-4*cropped
    coat-=22*jacket+6*longer-5*cropped
    shirt-=12*allover+9*bold+5*lapel
    # Match physical cut and fabric jointly. A tailored long-sleeve shirt
    # can have the outline of a short jacket, but usually lacks the bulk
    # of a raised collar or the ribbing of a jumper.
    binding,shoulder,flat,folds,weave=fabric(a,im)
    width=a.sum(axis=2)*aspect[:,None]/28
    body=width[:,15:20].mean(axis=1)
    hip=width[:,22:25].mean(axis=1)
    breadth=unit(body,.48,.22)
    flareout=unit(hip-body,.035,.13)
    bodice=unit(.59-body,0,.18)
    tankcut=unit(hip,.43,.18)*unit(.16-abs(hip-body),0,.12)
    # Both free sleeve tips end at, or above, the ribs on a short-sleeve
    # tee. Compare the outer envelope rather than counting foreground:
    # trouser-leg splits and the slit beside a long cuff also reduce area.
    outline=.5*(margins(a)+margins(a[:,:,::-1]))
    atcuff=outline[:,4:13].min(axis=1)
    straight=outline[:,15:24].mean(axis=1)
    reach=unit(straight-atcuff,.052,.13)
    teehem=unit(.14-outline[:,14:23].std(axis=1),.055,.065)
    cutshort=reach*teehem
    tshape=unit(aspect,.65,.26)*cutshort
    armhole=unit(outline[:,4:9].mean(axis=1)-straight,.035,.12)
    tee-=24*tshape+12*breadth*(1-flareout)*cutshort+7*tankcut*armhole*(1-closure)
    dress+=16*tshape+7*breadth*cutshort*(1-flareout)-5*bodice*flareout
    cuffs=unit(.55-short,0,.55)
    knitted=clip(.55*flat+.43*crew+.42*binding+.55*cropped-.60*closure-.28*peak)
    outerwear=clip(.62*shoulder+.44*peak+.45*closure+.32*longer-.42*cropped)
    tailored=clip(.42*allover+.45*weave+.40*hemcurve+.42*folds+.30*lapel-.32*cropped-.36*shoulder)
    shirt-=20*tailored+6*folds*(1-binding)
    shirt+=6*binding*cuffs+6*graphic*short*(1-allover)
    knit-=13*knitted*cuffs
    coat+=9*knitted*cuffs-13*outerwear*cuffs
    # A soft blouse often drapes off the shoulders or has a short wing
    # sleeve, yet is very different from a jersey print. Give these fabric
    # cues their own interpretation rather than counting all Vs as lapels.
    shirtprint=clip(.65*weave+.55*allover+.32*folds-.46*stripes)
    shirt-=19*shirtprint*(1-.55*shoulder)+5*hemcurve
    # Detect two especially characteristic structures directly: the
    # separate limbs on either side of a high, ribbed jumper hem, and
    # a coat body projecting below both sleeves like a straight skirt.
    bot=margins(a[:,::-1,:].transpose(0,2,1))
    cuff=.5*(bot[:,1:6].min(axis=1)+bot[:,22:27].min(axis=1))
    torsoend=bot[:,11:17].mean(axis=1)
    below=unit(cuff-torsoend,.06,.19)*(1-cutshort)
    above=unit(torsoend-cuff,.03,.17)
    coat-=16*below*outerwear
    knit-=15*above*(1-closure)
    # A level hem on a pullover by itself is weak evidence; use shoulder
    # rise and a front closure together to recognise a zip jacket instead.
    coat-=13*closure*shoulder
    shirt+=7*shoulder*below
    # A dress hem need not flare, but the skirt tends to be narrow relative
    # to its length, whereas standard short sleeves span a tee's body.
    dress+=10*cutshort*breadth
    # Distinguish a graphic decorating the chest of an intact jersey from
    # a light woven blouse printed all the way down to the shirttail.
    tee+=9*weave*allover
    # A sleeve-shaped object cannot be a clutch solely because the cuffs
    # happen to fill in the lower corners. Test for the pinched shoulders
    # and for a continuous, substantial body beginning under the neck.
    shoulder=unit(trimmean(a[:,2:7,9:19])-trimmean(a[:,1:5,0:5]),.35,.42)
    neckmass=trimmean(a[:,1:7,9:19]); fullneck=unit(neckmass,.52,.40)
    bag+=16*shoulder*fullneck*(1-handle)*(1-shoe)
    # The nearly straight horizontal lip of a pocketbook, with a rapid
    # transition into a rectangular full bag body, isn't a shoulder line.
    topwidth=a.mean(axis=2)
    ease=topwidth[:,4:19]-topwidth[:,1:16]
    ledge=unit(ease.max(axis=1),.28,.32)
    solid=unit(trimmean(a[:,16:25,3:25]),.75,.22)
    bag-=11*ledge*solid+8*handle*solid
    # The negative space under/between straps can be open to the sky.
    # Detect interruption of the top line or of the footbed separately
    # from the completely enclosed spaces measured above.
    rim=margins(a.transpose(0,2,1))
    cuts=np.maximum(0,rim[:,3:25]-.5*(rim[:,:22]+rim[:,6:]))
    notches=unit(np.sort(cuts,axis=1)[:,-3],.047,.13)
    density=trimmean(a)
    airy=unit(.66-density,.0,.22)
    sandal-=15*notches+15*airy
    trainer+=8*notches+9*airy
    boot+=8*notches+6*airy
    # In a boot the ankle is a continuous shaft even when there is a
    # large open arch under a high heel. Straps leave cuts ABOVE the foot.
    shaft=unit(trimmean(a[:,6:16,18:24]),.68,.29)*high
    boot-=14*shaft*(1-notches)
    sandal+=7*shaft*(1-notches)
    # On very low footwear the same idea distinguishes the uninterrupted
    # vamp of a trainer from the separated bands of a walking sandal.
    # The pair of strap ends often separates near the front instep.
    vamp=trimmean(holes[:,9:23,3:17]); openvamp=unit(vamp,.04,.19)
    sandal-=13*openvamp+7*strap*notches
    trainer+=7*openvamp
    front=13*shoe+9*asym
    side=9*(1-shoe)*(1-asym)
    values=np.stack([tee+front,trouser+front,knit+front,dress+front,coat+front,sandal+side,shirt+front,trainer+side,bag,boot+side],axis=1)
    # Hierarchical readout. A good match to continuous sleeves establishes
    # the parent category, but is NOT three independent pieces of evidence
    # for a pullover. Use a separate competition inside the long-top family
    # where the differences in cut are modest and the seams matter more.
    p=np.argmin(values,axis=1)
    longfamily=(p==2)|(p==4)|(p==6)
    nearshirt=np.minimum.reduce([dist['buttonshirt'],dist['shirtfront'],dist['blouse']])
    drape=unit(above,.09,.59)
    shallow,vneck,zipline=neckstructure(a,im)
    opening=clip(.55*zipline+.41*closure+.28*vneck+.16*persist-.19*unit(foldline,.13,.22))
    raisedneck=clip(.55*shoulder+.37*peak+.26*raised-.27*crew)
    neat=clip(.59*crew+.37*flat+.25*binding+.35*above-.37*opening)
    looseprint=clip(.49*allover+.42*weave+.23*bold-.37*stripes)
    # Differences of cut costs are attenuated within this family, not
    # between shoes/bags/dresses and apparel. A close match to a sleeve
    # arrangement should not overrule positive evidence of a front opening.
    kcut=np.minimum.reduce([dist['jumper'],dist['highneck']+6,dist['knitfront'],dist['boxknit'],dist['cardigan']+8])
    ccut=np.minimum.reduce([dist['coatfront'],dist['jacket'],dist['parka'],dist['overcoat']])
    scut=np.minimum(nearshirt,np.minimum(dist['shortshirt'],dist['smock']))
    length=unit(cuff-torsoend,.045,.18)
    rib=binding*(1-opening)
    pul=.36*kcut-13*neat-13*above-6*stripes-5*rib+21*opening+13*raisedneck+10*length
    outer=.36*ccut+6-16*opening-18*raisedneck-12*length+11*above+5*crew+5*looseprint
    woven=.36*scut+9-14*looseprint-10*hemcurve-6*folds-10*vneck-6*lapel-5*opening+8*raisedneck+8*above+5*stripes
    # A print covering a short sleeve rather than a centre chest graphic
    # is typical of a lightweight blouse. For other short sleeves, leave
    # the main classifier's T-shirt decision intact.
    collared=clip(.56*vneck+.48*zipline+.30*peak)
    woven-=10*cutshort
    fine=np.argmin(np.stack([pul,outer,woven],axis=1),axis=1)
    p=np.where(longfamily,np.array([2,4,6])[fine],p)
    blouse=(p==0)&(shirtprint>.58)&((collared>.40)|(hemcurve>.38))&(stripes<.52)&(graphic<.70)
    p=np.where(blouse,6,p)
    # On a structured handbag, material fills the entire lower panel,
    # uninterrupted by either an armpit or a shirt hem. A strap or an
    # abrupt rim with near-vertical sides supplies the other evidence.
    boxlike=unit(topwidth[:,10:23].min(axis=1),.70,.21)
    panel=unit(.06-outline[:,12:24].std(axis=1),0,.045)*boxlike
    supportedbag=solid*panel*clip(.83*handle+.72*ledge)
    bcut=np.minimum.reduce([dist['purse'],dist['clutch'],dist['tote'],dist['rucksack']+4])
    gcut=np.minimum.reduce([kcut,ccut,scut])
    bagbody=longfamily&(supportedbag>.37)&(bcut<gcut+17)&(opening<.68)
    p=np.where(bagbody,8,p)
    return p

def predict(x):
    result = np.empty(len(x),np.int64)
    for i in range(0,len(x),64):
        result[i:i+64] = recognize(x[i:i+64,0])
    return result
