import numpy as np

def pack(m, aspect):
    solid=(m>.48).astype(np.float32)
    row=np.minimum(np.maximum.accumulate(solid,axis=1),np.maximum.accumulate(solid[:,::-1],axis=1)[:,::-1])
    col=np.minimum(np.maximum.accumulate(solid,axis=0),np.maximum.accumulate(solid[::-1],axis=0)[::-1])
    def pool(a):
        return a.reshape(16,2,16,2).mean(axis=(1,3)).ravel()
    # Soft foreground, two directional hulls and four outline profiles. Squared
    # distance measures disagreement in area and in boundary location.
    prof=[]
    for a in (solid,solid[:,::-1],solid.T,solid.T[:,::-1]):
        dist=(1-np.maximum.accumulate(a,axis=1)).mean(1)
        prof.extend((dist.reshape(16,2).mean(1)*72).tolist())
    return np.rint(np.concatenate((pool(m)*16,pool(row)*10,pool(col)*10,prof,
                  [np.clip(np.log(aspect),-2.5,2.5)*310]))).astype(np.int16)

patterns=CONSTANTS['patterns'].astype(np.float32)
bounds=CONSTANTS['bounds']
lengths=(patterns*patterns).sum(axis=1)
pocketgrid=[(angle,y,x) for angle in (0,.65,1.25,-.75) for y in (17,20,23) for x in (6,8,10)]
pockety=np.array([np.rint(y+angle*(np.arange(5)-2)) for angle,y,x in pocketgrid],int)
pocketx=np.array([x+np.arange(5) for angle,y,x in pocketgrid],int)

def ramp(a,lo,hi):
    return np.clip((a-lo)/(hi-lo),0.,1.)

def smooth(a):
    b=np.pad(a,1,mode='edge')
    return (b[1:-1,1:-1]*4+b[1:-1,:-2]+b[1:-1,2:]+b[:-2,1:-1]+b[2:,1:-1])*.125

def view(a,x,y):
    x=np.clip(x,0,27); y=np.clip(y,0,27)
    i=x.astype(int); j=y.astype(int); dx=x-i; dy=y-j
    return (1-dy)*((1-dx)*a[j,i]+dx*a[j,np.minimum(27,i+1)])+dy*((1-dx)*a[np.minimum(j+1,27),i]+dx*a[np.minimum(j+1,27),np.minimum(i+1,27)])

def photo(raw):
    # Work relative to local foreground and the object's own bounding rectangle.
    sm=smooth(raw); peak=max(.01,np.max(sm)); sup=(raw>max(.016,peak*.043)) & (sm>peak*.049)
    iy,ix=np.nonzero(sup)
    if len(ix)==0:
        x0=0; x1=27; y0=0; y1=27
    else:
        x0=ix.min(); x1=ix.max(); y0=iy.min(); y1=iy.max()
    w=x1-x0+1; h=y1-y0+1; t=(np.arange(32)+.5)/32
    x=(x0-.5+w*t)[None,:]; y=(y0-.5+h*t)[:,None]
    fg=ramp(raw,.009+peak*.012,.022+peak*.087)
    m=view(fg,x,y).astype(np.float32); a=view(raw,x,y).astype(np.float32)
    m=np.clip((m-.10)/.81,0,1)
    # Registration: the soft upper and lower boundaries of a photographed item
    # can touch only one pixel in the extreme rows. Trim subpixel fringe without
    # eroding small, connected garment details.
    if np.max(m)>0:
        rowok=m.sum(1)>1.5; colok=m.sum(0)>1.5
        ri=np.flatnonzero(rowok); ci=np.flatnonzero(colok)
        if len(ri)>2 and len(ci)>2:
            lo=ri[0]; hi=ri[-1]; left=ci[0]; right=ci[-1]
            wr=w*(right+1-left)/32; hr=h*(hi+1-lo)/32
            xp=(x0-.5+w*left/32+wr*t)[None,:]; yp=(y0-.5+h*lo/32+hr*t)[:,None]
            m=np.clip((view(fg,xp,yp).astype(np.float32)-.10)/.81,0,1)
            a=view(raw,xp,yp).astype(np.float32); w=wr; h=hr
    # Prefer toe at the left for side views; bilateral cuts need no preferred side.
    if m[:18,:16].sum()>m[:18,16:].sum():
        m=m[:,::-1].copy(); a=a[:,::-1].copy()
    aspect=w/h; descriptor=pack(m,aspect)
    return descriptor,cues(a,m,aspect,peak)


def cues(a,m,aspect,peak):
    # Intrinsic surface cues, separated from the silhouette so that a dark coat
    # can still have a strong opening even if its outline is sweater-shaped.
    center=a[6:27,12:20]; lum=float(np.mean(a*m)/(np.mean(m)+.0001)); denom=.075+.68*lum
    reflection=np.abs(m-m[:,::-1]).mean()
    asym=ramp(reflection,.125,.42)
    bilateral=1-asym
    # Search a small fan of nearly vertical closure lines. A placket may be a
    # light line, a dark line, or a single step in tone. Follow it down the torso.
    seam=0.; gap=0.; line=0.; button=0.
    yy=np.arange(7,28); rf=np.arange(21)
    for tilt in (-.10,0.,.10):
        for off in (-2,-1,0,1,2):
            j=(15.5+off+tilt*(yy-17)).round().astype(int)
            middle=(a[yy,j]+a[yy,j+1])*.5
            left=(a[yy,j-2]+a[yy,j-3])*.5; right=(a[yy,j+3]+a[yy,j+4])*.5
            contrast=((left+right)*.5-middle)/denom
            # Require evidence in more than one horizontal slice, ignoring collar and hem.
            groups=contrast.reshape(3,7).mean(1); sign=1 if contrast.mean()>=0 else -1
            q=max(0.,np.mean(groups*sign)-.48*np.std(groups))
            edge=(left-right)/denom
            eg=edge.reshape(3,7).mean(1)
            step=max(0.,abs(edge.mean())-.72*np.std(eg))
            seam=max(seam,q+.23*step); line=max(line,step)
            gap=max(gap,np.maximum(np.minimum(left,right)-middle,0).mean()/denom)
            spots=(contrast[1:-1]-.5*(contrast[:-2]+contrast[2:]))
            button=max(button,float(np.mean(np.maximum(spots,0))))
    seams=ramp(seam,.035,.31); strong=ramp(seam,.22,.60); opening=ramp(gap,.22,.76)
    # Look for aligned pairs of diagonal lapel borders which meet below the neck.
    lapel=0.
    for bottom in (9,13,17):
        yy=np.arange(2,bottom); wid=np.rint((bottom-yy)*5/(bottom-2)).astype(int)+1
        lo=15-wid; hi=16+wid
        lc=a[yy,lo+1]-a[yy,lo-1]; rc=a[yy,hi-1]-a[yy,hi+1]
        lapel=max(lapel,abs(float(lc.mean()+rc.mean()))/(2*denom))
    lap=ramp(lapel,.12,.54)
    # Texture: interior edges rather than the object's outer silhouette.
    gx=np.abs(a[:,2:]-a[:,:-2]); gy=np.abs(a[2:,:]-a[:-2,:]);
    vx=m[:,2:]*m[:,:-2]; vy=m[2:,:]*m[:-2,:]
    gx=gx[6:26,7:23]; vx=vx[6:26,7:23]; gy=gy[5:25,8:24]; vy=vy[5:25,8:24]
    tx=float((gx*vx).sum()/max(1.,vx.sum()))/denom; ty=float((gy*vy).sum()/max(1.,vy.sum()))/denom
    busy=ramp((tx+ty)*.5,.18,.54)
    # A plaid's vertical edges recur down the whole shirt, its horizontal ones
    # across the body. This separates a weave/check from one isolated graphic.
    ax=np.abs(a[7:25,8:23]-a[7:25,10:25]); ay=np.abs(a[5:24,9:23]-a[7:26,9:23]);
    repeated=(np.minimum(ax[:9].mean(0),ax[9:].mean(0)).mean()+np.minimum(ay[:,:7].mean(1),ay[:,7:].mean(1)).mean())/(2*denom)
    weave=ramp(repeated,.17,.46)
    # Wide flat tone under the throat, round ribbed neckline, and cuffs typify
    # pullovers. A dress shirt's neckline often has a collar framing a V.
    neck=(a[2:6,13:19].mean()-a[7:11,13:19].mean())/denom
    neckband=ramp(abs(neck),.12,.60)
    # Shape measurements use quantile-free soft masks, not absolute greys.
    row=m.mean(1); inner=m[:,10:22].mean(1)
    bodyw=row[12:19].mean(); hem=row[26:30].mean()
    flare=ramp(hem-bodyw,.045,.29)
    shoulder=row[5:11].mean(); short=ramp(shoulder-hem,.093,.34)
    narrow=ramp(.78-aspect,0,.31)
    darkbody=1-ramp(lum,.25,.65)
    # A shirt-tail curves up at the flanks; a waistband (knit top) has a
    # continuous horizontal edge, and can sit slightly above the sleeve ends.
    ybot=np.maximum.accumulate((m>.50)[::-1],axis=0).sum(0)/32.
    shirtail=ramp(ybot[13:19].mean()-(ybot[7:11].mean()+ybot[21:25].mean())*.5,.029,.13)
    hemstrip=(a[26:29,10:22].mean()-a[22:25,10:22].mean())/denom
    rib=ramp(abs(hemstrip),.11,.48)
    # Vertical extent and interior holes discriminate empty straps from closed
    # uppers even when the silhouettes share the same convex envelope.
    hard=(m>.50).astype(np.float32)
    hrow=np.minimum(np.maximum.accumulate(hard,axis=1),np.maximum.accumulate(hard[:,::-1],axis=1)[:,::-1])
    hcol=np.minimum(np.maximum.accumulate(hard,axis=0),np.maximum.accumulate(hard[::-1],axis=0)[::-1])
    hull=np.minimum(hrow,hcol)
    holes=np.maximum(hull-m,0).sum()/max(1.,hull.sum())
    # Also detect dark negative space above a continuous sole (thin black straps).
    rel=a/(peak+.005); deep=(rel<.092)*hull
    strap=ramp(holes,.025,.235); void=ramp(deep.sum()/max(1.,hull.sum()),.11,.39)
    openness=max(strap,void*.80)
    lefttop=(hcol[:,:8].mean()); righttop=hcol[:,22:30].mean(); backrise=righttop-lefttop
    shaft=ramp(backrise,.28,.64)*ramp(2.55-aspect,0.,1.22)
    # Trousers need not have air between the legs: a long medial dark valley
    # is also diagnostic if surrounded by cloth on both sides at the ankles.
    legs=0.
    for off in (-3,0,3):
        for lean in (-.15,.0,.15):
            yy=np.arange(16,31); j=(15.5+off+(yy-23)*lean).round().astype(int)
            v=np.minimum((a[yy,j-5]+a[yy,j-6])/2,(a[yy,j+5]+a[yy,j+6])/2)-(a[yy,j]+a[yy,j+1])/2
            legs=max(legs,float(v.mean())/denom)
    split=ramp(legs,.20,.70); trouser=ramp(.86-aspect,0,.38)*split
    # A handle tends to have a large hollow above a wide, mostly solid base.
    handle=ramp((hull[:15]*(1-m[:15])).mean(),.045,.29)*ramp(m[19:27].mean(),.48,.80)
    rect=ramp(min(row[7:14].mean(),row[18:27].mean()),.76,.95)
    # Lower sleeve boundaries: for a hanging long sleeve the wrist can protrude
    # beyond the hem; for a tee there is an abrupt lateral step high on the trunk.
    outer=(hrow.mean(1)); ledge=0.
    for j in range(6,15):
        ledge=max(ledge,float(outer[j-3:j].mean()-outer[j+1:j+4].mean()))
    sleeve=ramp(ledge,.10,.32)
    # Rounded shoulder slope vs the abrupt, full-width waistband of trousers.
    belt=ramp(row[1:4].mean()-.5*row[7:11].mean(),.17,.45)
    # The cutoff of a short sleeve is a bilateral narrowing, distinct from a
    # gradual waist. A dress's skirt usually extends beyond that bodice again.
    armstep=0.; armlevel=0.
    for j in range(6,16):
        above=outer[j-3:j].mean(); below=outer[j+1:j+5].mean(); delta=above-below
        armstep=max(armstep,min(delta,above-outer[20:25].mean()))
        armlevel=max(armlevel,min(delta,above-outer[25:29].mean()))
    straight=ramp(.21-abs(outer[17:21].mean()-outer[26:29].mean()),0,.17)
    teeform=ramp(armstep,.07,.24)*straight*ramp(armlevel,.05,.23)*ramp(aspect,.56,.84)
    # Assess a narrow opening against immediately adjacent strips, over short
    # segments. This is insensitive to line polarity, buttons, and seam drift.
    piece=[]; lid=[]
    for begin,end in ((6,11),(11,16),(16,21),(21,26)):
        strip=a[begin:end]; wide=strip[:,9:23].mean(0)
        light=np.maximum(wide[2:-2]-.5*(wide[0:-4]+wide[4:]),0)
        shade=np.maximum(.5*(wide[0:-4]+wide[4:])-wide[2:-2],0)
        narrowedge=np.maximum(light,shade); dy=np.abs(np.diff(wide)); relbase=np.mean(wide)+.07
        loc=np.max(narrowedge[2:8]); bkg=np.mean(narrowedge[[0,1,8,9]])
        piece.append(max(0.,loc-.8*bkg)/relbase)
        lid.append((np.max(dy[3:10])-.65*np.mean(dy))/relbase)
    zipline=ramp(np.mean(piece)+.20*min(piece)+.20*np.mean(lid),.030,.19)
    # Coats often have a raised or pointed neck and a long line of shoulders
    # below it. A pullover more often reaches its highest point across the neck.
    first=(1-np.maximum.accumulate(hard,axis=0)).mean(0)
    collarheight=(first[4:8].mean()+first[24:28].mean())*.5-first[12:20].mean()
    highneck=ramp(collarheight,.080,.25)
    # At the sides of a knit top the cuff and welt end at visibly different
    # heights; straight-cut closing jackets often have a single blocky hem.
    wristbottom=(ybot[3:7].mean()+ybot[25:29].mean())*.5
    cufflow=ramp(wristbottom-ybot[11:21].mean(),.008,.12)
    # Missing vertical sections or a high arch between heel and sole. Count only
    # gaps enclosed in a column, not the black background above a low shoe.
    verticalgap=(hcol*(1-m))[:,4:29].mean()
    thinupper=ramp(verticalgap,.027,.155)
    cavity=ramp(max(holes,verticalgap),.07,.24)
    # Look for daylight *inside* the upper. A sneaker with a sprung toe can have
    # space beneath it, and a pattern printed on leather is not a cutout.
    box=np.arange(32)[:,None]; solidcol=hard>0
    ceiling=np.argmax(solidcol,axis=0); floor=31-np.argmax(solidcol[::-1],axis=0)
    within=(box>=ceiling+2)&(box<=floor-2)&(box<27)
    interior=within*(rel<.032)
    inside=(interior[:-1,:-1]*interior[1:,:-1]*interior[:-1,1:]*interior[1:,1:])
    daylight=ramp(inside[:,2:30].mean(),.008,.085)
    leaky=ramp(holes,.13,.33)
    # Sole/shoe body should face sideways: less symmetry than a front-facing
    # bag. Rectangular clutches, unlike coats, have their mass at or below the top.
    handbag=ramp(m[19:28,2:30].mean()-m[4:12,2:30].mean(),.20,.57)*ramp(hem,.55,.85)
    longskirt=ramp(1/aspect,1.43,2.15)*ramp(inner[24:29].mean(),.75,.98)
    skirtshape=ramp((outer[25:29].mean()-outer[12:16].mean()),.0,.32)
    # Search a row for the two shoulder points flanking the brighter torso, then
    # compare sleeves to bodice and hem in relative coordinates. Slight softness
    # at the outer cuff should not remove the evidence of a short sleeve.
    steppart=0.
    for j in range(6,16):
        top=outer[max(0,j-5):j].max(); low=outer[j+2:j+6].mean()
        steppart=max(steppart,min(top-low,top-outer[23:28].mean()))
    capped=ramp(steppart,.085,.25)*straight*ramp(aspect,.62,.84)
    # Distinguish textile prints from knitted stripes and from a chest graphic.
    # Compare upper and lower regions as well as edges in the two directions.
    def cloth(y,x):
        sample=a[y:y+8,x:x+5]; support=m[y:y+8,x:x+5]
        intensity=sample.mean()+.08
        g=np.abs(sample[1:,1:]-sample[1:,:-1])*support[1:,1:]*support[1:,:-1]
        f=np.abs(sample[1:,1:]-sample[:-1,1:])*support[1:,1:]*support[:-1,1:]
        return (min(g.mean(),f.mean())+.25*max(g.mean(),f.mean()))/intensity
    patches=[cloth(y,x) for x in (8,19) for y in (6,17)]
    allover=ramp(np.sort(patches)[1],.068,.25)
    stripe=ramp((ty-tx)/(ty+tx+.03),.18,.60)
    # Look for an actual neck opening (dark top connected across the throat),
    # not a line somewhere on the chest. Two collar points frame a lower notch.
    backdrop=(a[1:4,11:21].mean()-a[5:8,12:20].mean())/denom
    vneck=ramp(-backdrop,.14,.72)*ramp(neckband,.1,.7)
    prominent=0.
    for y in (2,3,4):
        zone=a[y:y+4]; centerneck=zone[:,14:18].mean(); flank=min(zone[:,10:14].mean(),zone[:,18:22].mean())
        prominent=max(prominent,(flank-centerneck)/denom)
    collarpoints=ramp(prominent,.13,.63)*lap
    # Inspect lateral notches in the silhouette: full-length sleeves and bifurcated
    # hems often leave long notches on both sides of the trunk, not in its center.
    slot=(1-m[22:28])*hrow[22:28]
    armnotch=min(slot[:,:12].mean(),slot[:,20:].mean()); arms=ramp(armnotch,.045,.24)
    # Follow a neckline in the appearance image. V and round knits commonly
    # have one continuous, roughly even band; lapels give two longer edges.
    mid=(np.arange(32)-15.5)/16; roundband=0.; turn=0.
    for radius in (.29,.42):
        for depth in (2,4,6):
            xx=np.flatnonzero(abs(mid)<radius); unit=mid[xx]/radius
            yy=(1+depth*np.sqrt(np.maximum(0,1-unit**2))).round().astype(int)
            inn=a[np.maximum(0,yy-1),xx]; rim=a[yy,xx]; body=a[yy+2,xx]
            contrast=(inn-body)/denom; midrim=(rim-(body+inn)/2)/denom
            roundband=max(roundband,abs(contrast.mean())-.2*np.abs(contrast-contrast.mean()).mean(),abs(midrim.mean())*.9)
    roundneck=ramp(roundband,.09,.39)
    # A raised coat collar often rises as a narrow pedestal above both shoulder
    # seams. A knit sweater has a broader top; its sleeves frequently drop past
    # a horizontal welt. A shirt's two pointed collar leaves surround a notch.
    shoulderRise=(first[7:11].mean()+first[21:25].mean())*.5-first[13:19].mean()
    pedestal=ramp(shoulderRise,.023,.13)
    narrowtop=ramp(outer[5:8].mean()-outer[0:2].mean(),.35,.72)
    # Find side openings from the cuffs up: unlike a bag's single loop, these
    # are paired and meet the lower edge of the image.
    yy=np.arange(32)[:,None]; xx=np.arange(32)[None,:]
    gapbottom=np.maximum(0.,hrow-m)*np.minimum.accumulate(1-hard[::-1],axis=0)[::-1]
    cut=max(min(gapbottom[21:31,3:12].mean(),gapbottom[21:31,20:29].mean()),.6*armnotch)
    sleeveslots=ramp(cut,.015,.21)
    # Follow the convex envelope across concavities that a column or row alone
    # cannot close (an inclined instep strap meeting the sole, for example).
    envelope=hard.copy()
    for slope in (-1.5,-.75,-.4,.4,.75,1.5):
        offs=np.rint(slope*(np.arange(32)-15.5)).astype(int)[None,:]
        lifted=np.zeros((82,32),np.float32)
        lifted[yy+25+offs,xx]=hard
        enclosed=np.minimum(np.maximum.accumulate(lifted,axis=1),np.maximum.accumulate(lifted[:,::-1],axis=1)[:,::-1])
        envelope=np.maximum(envelope,enclosed[yy+25+offs,xx])
    missing=np.maximum(envelope-m,0)
    # A low trainer has a single dipping collar; a sandal has broken intervals
    # across the instep or a largely empty toe box above the footbed.
    instep=missing[6:25,4:25].mean(); mouth=missing[9:22,3:17].mean()
    broken=ramp(instep,.105,.37)*ramp(mouth+.5*verticalgap,.07,.30)
    airloop=ramp((missing*within)[3:26,3:29].mean(),.035,.19)
    toeempty=ramp(1-lefttop,.57,.88)*ramp(m[10:24,0:12].mean(),.24,.01)
    # A handbag's loop occupies the top of the box; its body has broad upper
    # corners, unlike the lower corners of drooping shirt sleeves.
    wideobject=ramp(aspect,1.02,1.56)
    bagloop=handle*ramp(m[16:26,6:26].mean(),.72,.96)
    # Pocket welts are short slanted or horizontal strokes in mirror positions
    # near the hips. A stripe spanning the whole torso is not a pair of pockets.
    def welt(xx):
        vals=a[pockety,xx]; other=.5*(a[pockety-2,xx]+a[pockety+2,xx]); return (vals-other).mean(1)
    l=welt(pocketx); r=welt(31-pocketx); middle=welt(14+np.arange(5))
    pocket=np.maximum(0,(np.minimum(abs(l),abs(r))-.65*abs(middle))*(l*r>0)).max()/denom
    pockets=ramp(pocket,.085,.32)*(1-.65*allover)
    # Wide lapel facets instead of only narrow linear borders: diagonal wedges
    # run from the neck sides toward the central closure on a tailored garment.
    facets=0.
    for length in (7,11,15):
        z=np.arange(length); xleft=(10+5*z/length).round().astype(int); yline=z+2
        for dx in (-1,1):
            c1=a[yline,xleft+dx]-a[yline,xleft+dx-3]; c2=a[yline,31-xleft-dx]-a[yline,31-xleft-dx+3]
            if c1.mean()*c2.mean()>0:
                facets=max(facets,min(abs(c1.mean()),abs(c2.mean()))/denom)
    facets=ramp(facets,.09,.39)
    # Shirt plackets are thin; a broad opening, visibly different left and right
    # panels, or a pair of pocket welts supports outerwear instead.
    blousecue=max(weave,allover)*(1-.7*stripe)
    # Below the waist a dress has one connected sheet even when it has long
    # sleeves above. A tank often has a deep, broad open neckline instead.
    skirtpart=ramp((outer[22:27]-outer[12:17]).mean(),.035,.22)*ramp(m[19:27,10:22].mean(),.72,.97)
    longbody=ramp(1/aspect,1.38,2.10)*(1-.65*sleeveslots)
    # A sneaker's instep often has short alternating lace bars, concentrated
    # in the rising midfoot, along with a lower collar than a boot shaft.
    lace=a[8:21,12:22]; pairdiff=np.abs(lace[2:,1:]-lace[:-2,:-1]); laceink=m[10:21,13:22]*m[8:19,12:21]
    laces=ramp((pairdiff*laceink).sum()/max(1.,laceink.sum())/denom,.15,.43)
    return np.array([asym,bilateral,seams,strong,opening,ramp(line,.15,.55),lap,busy,weave,neckband,flare,short,narrow,darkbody,shirtail,rib,openness,strap,shaft,trouser,handle,rect,sleeve,belt,aspect,ramp(button,.015,.09),teeform,zipline,highneck,cufflow,thinupper,cavity,daylight,leaky,handbag,longskirt,skirtshape,capped,allover,stripe,vneck,collarpoints,arms,roundneck,pedestal,narrowtop,sleeveslots,broken,airloop,toeempty,wideobject,bagloop,pockets,facets,blousecue,skirtpart,longbody,laces],np.float32)

def label(tile,extra):
    # Integer descriptors are represented in float32. Dot products and partial
    # sums remain small exact integers, independent of the BLAS batch kernel.
    dd=tile@patterns.T
    dd=(-2*dd+lengths[None,:]+(tile*tile).sum(1)[:,None])/(16*16*256)
    vals=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        v=np.partition(dd[:,lo:hi],7,axis=1)[:,:8]
        vals.append(-(.55*v.min(1)+.45*v.mean(1)))
    tee,tank,jersey,pants,sweater,dress,jacket,overcoat,sandal,slide,shirt,shortshirt,blouse,trainer,hightrainer,bag,boot=vals
    asym,bilateral,seam,strong,gap,step,lap,busy,weave,neck,flare,short,narrow,dark,tailshape,rib,openness,strap,shaft,trouser,handle,rect,sleeve,belt,aspect,buttons,teeform,zipline,highneck,cufflow,thinupper,cavity,daylight,leaky,handbag,longskirt,skirtshape,capped,allover,stripe,vneck,collarpoints,arms,roundneck,pedestal,narrowtop,sleeveslots,broken,airloop,toeempty,wideobject,bagloop,pockets,facets,blousecue,skirtpart,longbody,laces=extra.T
    # Use alternative cuts within a class, penalizing less characteristic cuts.
    # Surface terms distinguish otherwise nearly exchangeable long-sleeved cuts.
    closure=.052*seam+.058*strong+.055*gap+.027*step
    tailoring=.026*lap+.018*buttons
    sc=np.zeros((len(tile),10),np.float32)
    sc[:,0]=np.maximum.reduce([tee+.021,tank+.002,jersey-.067])+.025*sleeve-.041*seam-.024*weave+.012*short
    sc[:,2]=sweater+.040+.020*rib-.91*closure-.48*tailoring-.038*weave-.021*tailshape
    sc[:,4]=np.maximum(jacket,overcoat-.012)-.029+closure+tailoring-.050*weave-.019*tailshape-.020*busy+.012*dark
    sc[:,5]=np.maximum(sandal,slide-.009)+.115*openness+.033*strap-.022*(1-openness)
    sc[:,6]=np.maximum.reduce([shirt,shortshirt,blouse-.026])-.040+.042*weave+.029*busy+.022*tailshape+.022*lap+.022*buttons+.026*seam-.023*strong-.019*gap
    sc[:,7]=np.maximum(trainer,hightrainer-.026)+.025-.12*openness-.034*shaft
    sc[:,9]=boot+.050*shaft-.092*openness-.009
    # Separate the family decision from fine distinctions. Inside an upper-body
    # garment family the very same cutting pattern can be used for a knit, woven,
    # or outer garment; its small silhouette advantage is not conclusive.
    outerwear=zipline*(.55+.45*highneck)
    sc[:,0]+=.17*teeform
    sc[:,2]+=.035*cufflow-.12*outerwear-.06*highneck-.045*weave
    sc[:,4]+=.15*outerwear+.075*highneck-.04*cufflow
    sc[:,6]+=.065*weave+.040*busy+.045*tailshape+.026*zipline
    # Empty sandal straps are diagnostic even when specific strap layouts differ
    # from a drawing. Borrow the closed-shoe outline evidence in that case only.
    foot=np.maximum(sc[:,7],sc[:,9])
    sc[:,5]=np.maximum(sc[:,5],foot-.25+.35*daylight+.10*leaky+.045*thinupper-.09*(1-openness))
    # Each garment cue votes only in its appropriate recognition problem. A
    # bag handle should not compete against a jacket's seam detector, for example.
    app=np.zeros((len(tile),10),np.float32)
    app[:,3]=dress+.04*flare+.08*longskirt+.06*skirtshape-.055*teeform
    surface=.12*allover*(1-.7*stripe)+.07*collarpoints+.04*vneck
    # Lighter shirting can share the cut of a polo/tee or a straight jacket;
    # a woven all-over print and a pair of collar tips can disambiguate them.
    sc[:,6]=np.maximum.reduce([sc[:,6],tee-.070+surface, sweater-.090+surface,jacket-.10+surface])+surface*.55
    sc[:,0]+=.070*capped-.055*allover
    sc[:,2]+=.037*stripe-.042*collarpoints
    # The common family likelihood is based on the best valid cut, *before*
    # fine-grained texture rules change labels inside that family.
    family=np.maximum.reduce([tee+.025*capped,tank,jersey-.04,sweater,jacket,overcoat,shirt,shortshirt,blouse-.015])
    # Confirm apparel-like shoulder or armpit geometry before letting texture
    # enter the family decision; neither fabric texture nor a zip rules out bags.
    garment=np.maximum.reduce([sleeveslots,capped,pedestal*(1-handle),highneck*(1-belt)*(1-asym)])
    family+=.12*garment+.045*roundneck*(1-belt)*(1-asym)
    app[:,8]=bag+.16*bagloop+.065*rect*wideobject+.060*handbag*np.maximum(bagloop,wideobject)-.09*sleeveslots-.070*asym*(1-handle)
    app[:,1]=pants+.11*trouser+.055*belt-.055*sleeveslots-.05*teeform
    app[:,3]+=.035*longskirt*(1-capped)
    # Forward-dipping lapels and a narrow collar pedestal are construction cues
    # even on uniform pale cloth; avoid treating busy cloth alone as a closure.
    sc[:,4]+=.11*pedestal*(.4+.6*narrowtop)+.055*lap*zipline-.045*roundneck*(1-pedestal)-.045*allover*(1-zipline)
    sc[:,2]+=.052*roundneck*(1-zipline)+.04*cufflow-.075*pedestal*(.55+.45*narrowtop)
    sc[:,6]+=.065*collarpoints+.047*tailshape*(1-cufflow)+.045*allover*(1-stripe)
    sc[:,5]=np.maximum(sc[:,5],foot-.23+.33*broken+.21*airloop+.045*toeempty)
    # Accept a pocket or lapel only in conjunction with a central opening or
    # raised collar. A repeating pattern is better evidence for woven shirting.
    tailored=(.105*pockets+.085*facets)*(.4+.6*np.maximum(zipline,pedestal))
    sc[:,4]+=tailored-.055*blousecue*(1-strong)
    sc[:,2]-=.55*tailored
    # For an ambiguous cut, distinguish a narrow shirt placket under a collar
    # from a broad coat opening or the plain front of a knitted pullover.
    placket=zipline*(1-gap)*(1-strong)*np.maximum(facets,collarpoints)
    shirtprint=blousecue*(1-.5*pockets*strong)
    sc[:,6]+=.088*placket+.095*shirtprint+.055*collarpoints*(1-pedestal)
    # Tapered, elongated skirt and unbroken lower torso vs sleeve slits that
    # continue almost to the hem. A capped shoulder alone is not a skirt.
    app[:,3]+=.11*longbody*skirtpart-.09*capped*(1-longbody)
    family+=.045*capped*(1-longbody)
    sc[:,7]+=.065*laces*(1-openness)*ramp(aspect,1.65,2.65)
    non=np.maximum.reduce([app[:,1],app[:,3],app[:,8],sc[:,5],sc[:,7],sc[:,9]])
    winner=np.argmax(sc[:,[0,2,4,6]],axis=1); lab=np.array([0,2,4,6])[winner]
    other=np.argmax(np.stack([app[:,1],app[:,3],sc[:,5],sc[:,7],app[:,8],sc[:,9]],axis=1),axis=1)
    return np.where(family>non,lab,np.array([1,3,5,7,8,9])[other])

def predict(x):
    out=np.empty(len(x),np.int64)
    for pos in range(0,len(x),48):
        pairs=[photo(im[0]) for im in x[pos:pos+48]]
        d=np.stack([p[0] for p in pairs]).astype(np.float32)
        ex=np.stack([p[1] for p in pairs])
        out[pos:pos+len(d)]=label(d,ex)
    return out
