import numpy as np


def blur(a):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    return ((p[:,1:-1,:-2]+p[:,1:-1,2:]+p[:,:-2,1:-1]+p[:,2:,1:-1])*2+p[:,:-2,:-2]+p[:,2:,:-2]+p[:,:-2,2:]+p[:,2:,2:]+4*p[:,1:-1,1:-1])*(1/16)


def pool(a):
    return a.reshape(-1,14,2,14,2).mean(axis=(2,4))


def unit(a):
    return np.clip(a,0,1)


def features(m,aspect):
    # Envelopes from both axes reduce sensitivity to printing and fractured edges.
    m=blur(m)*.50+m*.50
    left=np.maximum.accumulate(m,axis=2); right=np.maximum.accumulate(m[:,:,::-1],axis=2)[:,:,::-1]
    top=np.maximum.accumulate(m,axis=1); bottom=np.maximum.accumulate(m[:,::-1,:],axis=1)[:,::-1,:]
    row=np.minimum(left,right); col=np.minimum(top,bottom)
    measures=[]
    for a,w in [(m,.60),(row,.21),(col,.19)]:
        measures.append(pool(a).reshape(len(m),196)*(w**.5/14))
    for p in (left.mean(2),right.mean(2),top.mean(1),bottom.mean(1)):
        measures.append(p.reshape(-1,14,2).mean(2)*(.095/14)**.5)
    measures.append(np.log(np.maximum(.06,aspect))[:,None]*(.24**.5))
    return np.ascontiguousarray(np.concatenate(measures,axis=1),dtype=np.float32)


def makebank():
    pixels=CONSTANTS['silhouettes']; aspect=CONSTANTS['aspect']; costs=CONSTANTS['costs']; counts=CONSTANTS['count']
    result=[]; offset=0
    for n in counts:
        bank=features(pixels[offset:offset+n].astype(np.float32)*(1/255),aspect[offset:offset+n])
        result.append((np.ascontiguousarray(bank.T),np.sum(bank*bank,axis=1)+costs[offset:offset+n]))
        offset+=n
    return result


banks=makebank()


def sample(a,ys,xs):
    yi=np.floor(ys).astype(int); xi=np.floor(xs).astype(int)
    wy=(ys-yi).astype(np.float32)[:,:,None]; wx=(xs-xi).astype(np.float32)[:,None,:]
    yj=np.minimum(yi+1,27); xj=np.minimum(xi+1,27)
    b=np.arange(len(a))[:,None,None]
    return ((1-wy)*((1-wx)*a[b,yi[:,:,None],xi[:,None,:]]+wx*a[b,yi[:,:,None],xj[:,None,:]])+wy*((1-wx)*a[b,yj[:,:,None],xi[:,None,:]]+wx*a[b,yj[:,:,None],xj[:,None,:]]))


def prepare(a):
    peak=np.maximum(.04,np.quantile(a,.965,axis=(1,2)).astype(np.float32))[:,None,None]
    sm=.73*a+.27*blur(a)
    m=unit((sm-.005-.022*peak)/(.022+.083*peak))
    # Suppress isolated glints when locating the bounding rectangle, but allow
    # a handle or narrow shoulder strap to define the top.
    rows=m.sum(2); cols=m.sum(1)
    rows=rows>np.maximum(.52,rows.max(1,keepdims=True)*.034)
    cols=cols>np.maximum(.52,cols.max(1,keepdims=True)*.034)
    y0=rows.argmax(1); y1=27-rows[:,::-1].argmax(1); x0=cols.argmax(1); x1=27-cols[:,::-1].argmax(1)
    w=x1-x0+1; h=y1-y0+1; t=(np.arange(28,dtype=np.float32)+.5)/28
    xs=np.clip((x0-.5)[:,None]+w[:,None]*t,0,27); ys=np.clip((y0-.5)[:,None]+h[:,None]*t,0,27)
    occ=sample(m,ys,xs); ink=sample(a/peak,ys,xs)
    # One local closing bridges small cracks in otherwise solid fabric.
    p=np.pad(occ,((0,0),(1,1),(1,1)),mode='edge'); hi=occ
    for dy,dx in ((0,1),(2,1),(1,0),(1,2)):
        hi=np.maximum(hi,p[:,dy:dy+28,dx:dx+28])
    p=np.pad(hi,((0,0),(1,1),(1,1)),mode='edge'); close=hi
    for dy,dx in ((0,1),(2,1),(1,0),(1,2)):
        close=np.minimum(close,p[:,dy:dy+28,dx:dx+28])
    return .68*occ+.32*close,ink,(w/h).astype(np.float32)


def matches(m,aspect):
    direct=features(m,aspect); reverse=features(m[:,:,::-1].copy(),aspect)
    pair=np.concatenate([direct,reverse]); base=(pair*pair).sum(1,keepdims=True); n=len(m)
    fits=[]
    for bank,offset in banks:
        dist=(base+offset[None,:])-2*(pair@bank)
        d=np.minimum(dist[:n],dist[n:])
        # Marginalize a little over pose: favor a match with several close cuts,
        # rather than depending entirely on one exact pose.
        near=np.partition(d,5,axis=1)[:,:6]
        fits.append(-(.70*near.min(1)+.30*near.mean(1)))
    return np.stack(fits,axis=1)


def ramp(x,lo,hi):
    return unit((x-lo)/(hi-lo))


def cues(m,a,aspect):
    s=np.zeros((len(a),10)); smooth=blur(a)
    # Find a placket or open front: a vertical edge, ridge, or trench that
    # persists down the center front. Compare against folds away from center.
    gx=smooth[:,:,2:]-smooth[:,:,:-2]; ridge=.5*(a[:,:,2:]+a[:,:,:-2])-a[:,:,1:-1]
    col=np.maximum(np.abs(gx),np.abs(ridge)*1.25)
    side=np.mean(col[:,7:23,5:21],axis=(1,2)); line=np.zeros(len(a))
    for slope in (-2,0,2):
        step=np.rint(np.linspace(-slope/2,slope/2,18)).astype(int)
        for center in range(10,16):
            ix=center+step; iy=np.arange(6,24)
            # Take the strongest polarity-consistent edge in a narrow corridor.
            mag=np.maximum(col[:,iy,ix],.81*np.maximum(col[:,iy,ix-1],col[:,iy,ix+1]))
            pol=gx[:,iy,ix]; coherent=np.maximum(abs(pol.mean(1)),abs(ridge[:,iy,ix].mean(1)))
            spans=mag.reshape(len(a),3,6).mean(2)
            quality=.53*spans.min(1)+.32*spans.mean(1)+.15*coherent
            line=np.maximum(line,quality)
    rel=line/(.055+side)
    closure=ramp(line,.045,.22)*(.42+.58*ramp(rel,.86,1.94))
    # Paired lapel diagonals. A single stripe across the body isn't a lapel.
    diag=np.abs(gx); lapel=np.zeros(len(a))
    for center in (12,13,14):
        for end in (8,12,16):
            iy=np.arange(2,end); dx=np.rint(np.linspace(4.8,1.0,len(iy))).astype(int)
            l=diag[:,iy,center-dx].mean(1); r=diag[:,iy,center+dx].mean(1)
            lapel=np.maximum(lapel,.65*np.minimum(l,r)+.35*(l+r)*.5)
    lapel=ramp(lapel-.66*side,.045,.22)
    # Texture outside the central print area, measured relative to the lighting
    # of the item. Most jackets are not floral/checked like a printed blouse.
    detail=np.abs(a-smooth); cloth=np.maximum(.15,a[:,7:23,7:21].mean(axis=(1,2)))
    texture=(detail[:,7:22,7:11].mean(axis=(1,2))+detail[:,7:22,17:21].mean(axis=(1,2)))*.5
    pattern=ramp(texture/(.36+cloth),.043,.13)
    centerart=detail[:,6:16,10:18].mean(axis=(1,2))
    graphic=ramp(centerart-texture,.02,.11)*(1-closure)
    # A stand-up collar is a central protrusion above both shoulders. A tank
    # or scoop-neck, conversely, has a central dip between narrow straps.
    crown=(m>.35).argmax(1)/28
    center=crown[:,12:16].mean(1); neckside=.5*(crown[:,8:11].mean(1)+crown[:,17:20].mean(1))
    outer=.5*(crown[:,5:8].mean(1)+crown[:,20:23].mean(1))
    raised=ramp(neckside-center,.035,.19)*ramp(outer-center,.085,.28)
    dip=ramp(center-neckside,.038,.23)
    # Curved shirttails terminate earlier at the hips than on the center line.
    end=(m[:,::-1,:]>.40).argmax(1)/28
    curved=ramp(.5*(end[:,7:10].mean(1)+end[:,18:21].mean(1))-end[:,12:16].mean(1),.052,.18)
    # Interpretation of silhouette similarity as negative discrepancy; appearance
    # evidence is deliberately bounded in those same units of silhouette error.
    # A thin center channel between legs, missed by a coarse occupancy grid.
    # Look for two separately supported columns, not merely a dark fold.
    cleft=np.zeros(len(a))
    for x in range(9,19):
        core=1-m[:,11:27,x]; fabric=np.minimum(m[:,11:27,x-7:x-3].mean(2),m[:,11:27,x+4:x+8].mean(2))
        split=(core*fabric).mean(1)
        darkness=np.maximum(0,np.minimum(a[:,11:27,x-6:x-2].mean(2),a[:,11:27,x+3:x+7].mean(2))-a[:,11:27,x])
        cleft=np.maximum(cleft,split+.22*np.minimum(.8,darkness).mean(1))
    belt=m[:,1:4,5:23].mean(axis=(1,2)); tapered=m[:,5:10,:].mean(axis=(1,2))-m[:,20:25,:].mean(axis=(1,2))
    legs=ramp(cleft,.11,.62)*ramp(belt,.44,.82)*(1-ramp(aspect,.58,.83))
    s[:,1]+=.110*legs+.026*ramp(tapered,.04,.26)*ramp(belt,.70,.98)*(1-ramp(aspect,.53,.77))
    waistband=ramp(m[:,:2,4:24].mean(axis=(1,2)),.63,.92)*(1-ramp(aspect,.52,.74))
    s[:,1]+=.045*waistband*(.40+.60*ramp(tapered,.03,.24))
    s[:,3]-=.040*legs
    # A skirt continues below the waist with uninterrupted fabric across its
    # middle. A sleeve ending at waist height is not a trouser leg.
    rowsize=np.minimum(np.maximum.accumulate(m,2),np.maximum.accumulate(m[:,:,::-1],2)[:,:,::-1]).mean(2); waistwidth=rowsize[:,10:19].min(1); hemwidth=rowsize[:,22:26].mean(1)
    skirt=ramp(hemwidth-waistwidth,.052,.26)*ramp(m[:,21:25,8:20].mean(axis=(1,2)),.72,.96)
    slender=1-ramp(aspect,.45,.73)
    s[:,3]+=.037*skirt*(.36+.64*slender)+.019*slender*(1-waistband)*(1-legs)
    s[:,0]+=.005+.014*graphic+.006*dip-.020*closure-.011*raised-.006*lapel
    s[:,2]+=.010+.020*graphic-.041*closure-.013*lapel-.010*pattern-.006*curved
    s[:,3]+=.002-.006*raised
    s[:,4]+=-.004+.037*closure+.017*raised+.020*lapel-.021*pattern-.007*dip-.009*graphic
    s[:,6]+=-.009+.018*closure+.013*lapel+.026*pattern+.021*curved-.008*graphic
    # A sandal has holes between straps: look for nearly empty pixels enclosed
    # both transversely and vertically, not just a curved sole or toe line.
    row=np.minimum(np.maximum.accumulate(m,axis=2),np.maximum.accumulate(m[:,:,::-1],axis=2)[:,:,::-1]); colh=np.minimum(np.maximum.accumulate(m,axis=1),np.maximum.accumulate(m[:,::-1,:],axis=1)[:,::-1,:]); holes=(np.minimum(row,colh)*(1-m))[:,3:23,3:25].mean(axis=(1,2))
    hole=ramp(holes,.053,.235)
    # A change in brightness within a black or shiny shoe can still delineate
    # an opening even when the outer silhouette survives smoothing. Count low
    # regions which have substantially lighter material on opposite sides.
    shade=unit((a-.035)/.17)
    hrow=np.minimum(np.maximum.accumulate(shade,2),np.maximum.accumulate(shade[:,:,::-1],2)[:,:,::-1]); hcol=np.minimum(np.maximum.accumulate(shade,1),np.maximum.accumulate(shade[:,::-1,:],1)[:,::-1,:])
    see=ramp((np.minimum(hrow,hcol)*(1-shade))[:,4:24,3:25].mean(axis=(1,2)),.075,.26)
    # Turn the side elevation toe-left using height of material in the upper.
    flip=m[:,:14,:10].sum(axis=(1,2))>m[:,:14,-10:].sum(axis=(1,2))
    profile=np.where(flip[:,None,None],m[:,:,::-1],m); present=profile>.48
    roof=np.where(present.any(1),present.argmax(1),28)/28
    rear=roof[:,20:25].mean(1); tongue=roof[:,13:18].min(1); back=roof[:,22:26].mean(1); fore=roof[:,4:10].mean(1)
    # Boots rise well above the toe and have a filled shaft beneath a fairly
    # level cuff. Trainers generally have a tongue ahead of a dipped topline.
    shaft=ramp(fore-rear,.30,.61)*(1-ramp(rear,.12,.31))*(1-ramp(aspect,1.85,2.65))
    notch=ramp(back-tongue,.085,.28)
    noheel=ramp(back,.43,.84)*ramp(roof[:,11:18].max(1)-tongue,.17,.42)
    arch=present[:,::-1,:].argmax(1)/28
    heel=ramp(arch[:,16:22].max(1)-arch[:,22:26].min(1),.065,.23)
    # Reject an apparent cuff when there is air directly beneath it. A narrow
    # horizontal strap at ankle height otherwise looks remarkably like a boot.
    ankleband=profile[:,9:21,18:25].mean(axis=(1,2)); shaft*=ramp(ankleband,.64,.96)
    shadow=np.where(flip[:,None,None],shade[:,:,::-1],shade)
    clear=1-shadow
    # Local maximum opening with darker center than the sole and the arching
    # straps above and on either side. Erode support to demand an actual patch
    # of air, not a seam, then search at the scale of a few pixels.
    sky=np.minimum(np.maximum.accumulate(shadow,axis=1),np.maximum.accumulate(shadow[:,::-1,:],axis=1)[:,::-1,:]); walls=np.minimum(np.maximum.accumulate(shadow,axis=2),np.maximum.accumulate(shadow[:,:,::-1],axis=2)[:,:,::-1]); air=clear*np.minimum(sky,walls)
    def patchmax(f,by,bx):
        p=np.pad(f,((0,0),(1,0),(1,0))); integral=p.cumsum(1).cumsum(2)
        avg=(integral[:,by:,bx:]+integral[:,:-by,:-bx]-integral[:,:-by,bx:]-integral[:,by:,:-bx])/(by*bx)
        return avg.max(axis=(1,2))
    window=ramp(patchmax(air[:,3:23,2:26],5,4),.41,.88)
    topline=blur(profile).sum(axis=1)/28
    backopen=ramp(topline[:,9:18].max(1)-topline[:,22:26].mean(1),.28,.62)
    cleartoe=ramp(roof[:,3:6].mean(1),.69,.92)*ramp(roof[:,8:12].mean(1)-tongue,.24,.57)
    # A trainer has a padded tongue in front of the collar. Its topline steps
    # DOWN behind the tongue, unlike the straight rim of a boot shaft.
    padded=ramp(roof[:,18:23].max(1)-tongue,.10,.33)*ramp(aspect,1.20,1.82)
    solid=(1-window)*(1-hole)*ramp(ankleband,.69,.97)
    s[:,5]+=.065*hole+.055*see+.057*noheel+.020*heel*(1-shaft)+.072*window+.036*backopen+.012*cleartoe
    s[:,7]+=.008-.029*hole+.018*notch-.018*heel-.031*noheel+.034*padded*solid-.024*window
    s[:,9]+=.002-.025*hole+.046*shaft*solid+.016*heel-.019*notch-.026*noheel-.027*padded-.042*window
    return s


def cut_and_fabric(m,a,aspect):
    # Return extra evidence for body clothing. Work in local contrast: on white
    # fabric a seam can be faint and on dark fabric it may be a bright line.
    v=np.sqrt(np.maximum(a,0)+.006); v=blur(v)*.32+v*.68
    pad=np.pad(v,((0,0),(0,0),(3,3)),mode='edge'); mid=pad[:,:,3:31]
    def ridge_at(w):
        l=pad[:,:,3-w:31-w]; r=pad[:,:,3+w:31+w]
        return np.maximum(abs(mid-(l+r)*.5),abs(l-r)*.50)
    lines=np.maximum(ridge_at(1),np.maximum(.90*ridge_at(2),.79*ridge_at(3)))
    baseline=np.minimum(lines[:,7:24,6:10].mean(axis=(1,2)),lines[:,7:24,18:22].mean(axis=(1,2)))
    zipstrength=np.zeros(len(a)); placket=zipstrength.copy()
    for x in range(11,17):
        for dx in (-2,0,2):
            y=np.arange(5,25); q=x+np.rint(np.linspace(-dx/2,dx/2,len(y))).astype(int)
            e=lines[:,y,q]; e=np.maximum(e,.84*np.maximum(lines[:,y,q-1],lines[:,y,q+1]))
            bands=e.reshape(len(a),4,5).mean(2); steady=.60*bands.min(1)+.40*e.mean(1)
            excess=steady-baseline*.58
            zipstrength=np.maximum(zipstrength,ramp(excess,.012,.080)*(.35+.65*ramp(steady/(.012+baseline),1.12,2.1)))
            # A discrete row of buttons can be more informative than the
            # continuous placket in a shirt. Look for local extremal dots.
            light=v[:,y,q]; along=light-.5*(v[:,y-2,q]+v[:,np.minimum(y+2,27),q])
            across=light-.5*(v[:,y,q-2]+v[:,y,q+2]); dots=np.maximum(0,np.minimum(along,across))+np.maximum(0,np.minimum(-along,-across))
            distinct=dots.reshape(len(a),4,5).max(2); buttons=np.sort(distinct,axis=1)[:,1:3].mean(1)
            placket=np.maximum(placket,ramp(buttons-baseline*.64,.021,.105)*ramp(excess,.01,.08))
    # Collar fold: a shallow or deep V with two consistent diagonal edges.
    def sided(y,x,delta):
        return v[:,y,x+delta]-v[:,y,x-delta]
    vee=np.zeros(len(a)); spread=vee.copy()
    for center in (13,14):
        for width in (3,5,7):
            for start,stop in ((1,6),(2,11),(2,16)):
                y=np.arange(start,stop+1); shift=np.rint(np.linspace(width,1,len(y))).astype(int)
                l=sided(y,center-shift,1); r=sided(y,center+shift,1)
                both=np.sqrt(abs(l.mean(1)*r.mean(1)))
                vee=np.maximum(vee,ramp(both-.9*baseline,.035,.17))
                if stop==6: spread=np.maximum(spread,ramp(both-.9*baseline,.04,.17))
    # Clothing structure below the collar. At the hem, open coat fronts often
    # make a central vent. At the head, a funnel collar sticks up; for a crew
    # neck, the two shoulder tops are instead connected by a shallow scoop.
    masker=m>.42
    roof=np.where(masker.any(1),masker.argmax(1),28)/28
    floor=np.where(masker.any(1),masker[:,::-1,:].argmax(1),28)/28
    c=roof[:,12:16].mean(1); sid=.5*(roof[:,6:9].mean(1)+roof[:,19:22].mean(1))
    inner=.5*(roof[:,9:11].mean(1)+roof[:,17:19].mean(1)); collar=ramp(sid-c,.068,.19)*(1-ramp(c,.035,.17))
    collar*=.55+.45*ramp(inner-c,.012,.095)
    vent=ramp(floor[:,12:16].max(1)-.5*(floor[:,8:11].mean(1)+floor[:,17:20].mean(1)),.034,.20)
    crew=ramp(c-inner,.022,.13)*(1-zipstrength)
    # Distribution of pattern: prints on button shirts tend to cover the side
    # panels as well as the center. A single graphic on a sweatshirt does not.
    hp=abs(v-blur(v)); rep=np.minimum(hp[:,6:20,6:11].mean(axis=(1,2)),hp[:,6:20,17:22].mean(axis=(1,2)))
    fine=ramp(rep,.015,.053)
    # Cuffs of a heavy jacket are broader than those of a shirt or jumper.
    # Ignore this cue entirely when cuffs do not separate from the torso.
    cuff=np.zeros(len(a)); valid=cuff.copy()
    for flip in (False,True):
        p=m[:,:,::-1] if flip else m
        occupied=p[:,17:26,:]>.45; first=occupied.argmax(2)
        xx=np.arange(28)[None,None,:]; gap=(~occupied)&(xx>first[:,:,None]); sep=np.where(gap.any(2),gap.argmax(2),28)
        width=(sep-first); good=(first<6)&(sep<12)&occupied[:,:,11:18].any(2)
        w=np.sum(good*width,axis=1)/np.maximum(good.sum(1),1)
        cf=unit(good.mean(1)*2); valid+=cf*.5; cuff+=cf*ramp(w*aspect,2.6,5.2)*.5
    # Asymmetric and very wide openings/dramatic sleeves are blouse-like.
    asym=abs(m-m[:,:,::-1]).mean(axis=(1,2)); drape=ramp(asym,.125,.29)
    outer=masker.any(2); xl=masker.argmax(2); xr=masker[:,:,::-1].argmax(2)
    midwidth=(28-xl-xr)*outer/28
    bell=ramp(midwidth[:,9:17].mean(1)-midwidth[:,20:25].mean(1),.09,.34)
    fitted=ramp(midwidth[:,13:18].mean(1)-midwidth[:,2:5].mean(1),.15,.38)
    s=np.zeros((len(a),10)); clasp=zipstrength
    # Woven checks and floral repeats differ from a sweater with broad,
    # horizontal knitted stripes. Test the *direction* of the texture as well
    # as its strength, excluding the outline, sleeve gap and front placket.
    dx=abs(v[:,:,2:]-v[:,:,:-2]); dy=abs(v[:,2:,:]-v[:,:-2,:]); both=m[:,5:20,6:22]>.92
    bx=dx[:,5:20,5:21]; by=dy[:,4:19,6:22]; off=np.ones(16); off[6:10]=0
    inside=both*off[None,None,:]; den=np.maximum(1,inside.sum(axis=(1,2)))
    tx=(inside*bx).sum(axis=(1,2))/den; ty=(inside*by).sum(axis=(1,2))/den
    striped=ramp(ty/(.015+tx),1.35,2.50)*ramp(ty,.035,.11)
    woven=ramp(np.minimum(tx,ty),.045,.134)*(1-striped)
    # Shape of negative space below the cuffs: the close fit of a jacket can
    # bring its sleeves into contact with its hips. A cropped pullover often
    # leaves daylight between cuffs and beneath the straight knitted waistband.
    armextent=.5*(floor[:,1:6].min(1)+floor[:,22:27].min(1)); hem=floor[:,10:18].mean(1)
    cropped=ramp(hem-armextent,.030,.18); skirt=ramp(armextent-hem,.08,.36)
    broadhem=ramp(m[:,22:26,6:22].mean(axis=(1,2)),.73,.96); blocky=collar*np.maximum(broadhem,skirt)
    s[:,2]+=.022*striped+.016*cropped*(1-clasp)
    s[:,4]+=.020*blocky+.014*skirt*clasp
    s[:,6]+=.018*woven-.014*striped
    s[:,0]-=.028*clasp+.015*collar+.014*placket
    s[:,2]+=.014*crew-.043*clasp-.018*vent-.017*vee-.013*collar-.020*placket
    s[:,4]+=.032*clasp+.022*vent+.022*collar+.013*vee+.022*cuff-.012*valid-.017*fine-.010*drape
    s[:,6]+=.020*clasp+.037*placket+.024*fine+.013*spread+.027*drape+.012*bell+.009*fitted-.013*cuff
    # Additional construction lines: welt pockets or diagonal zippers on both
    # hips suggest an outer layer; traditional folded shirt collars have two
    # downward tips. Use matched local contrasts rather than absolute grey.
    pockets=np.zeros(len(a)); tips=pockets.copy()
    for hand in (-1,1):
        view=v if hand==1 else v[:,:,::-1]; peak=np.zeros(len(a))
        for xpos in (6,8,10):
            for y0 in (12,16,19):
                yy=np.arange(y0,y0+6); xx=xpos+np.rint(np.linspace(2,-1,6)).astype(int)
                edge=view[:,yy,xx]; l=view[:,yy,xx-2]; r=view[:,yy,xx+2]
                rr=abs((edge-.5*(l+r)).mean(1)); ss=abs((l-r).mean(1))
                # Only count marks on the cloth, not the sleeve/torso gap.
                support=(m if hand==1 else m[:,:,::-1])[:,yy,xx-2].mean(1)
                pk=ramp(np.maximum(rr,ss*.64)-.9*baseline,.032,.13)*ramp(support,.87,.99)
                peak=np.maximum(peak,pk)
        pockets+=peak*.5
        peak=np.zeros(len(a))
        for deep in (4,6,8):
            yy=np.arange(1,deep+1); tipin=np.rint(np.linspace(12,9,len(yy))).astype(int); tipout=np.rint(np.linspace(7,9,len(yy))).astype(int)
            le=view[:,yy,tipout]-view[:,yy,tipout-1]; ri=view[:,yy,tipin+1]-view[:,yy,tipin]
            peak=np.maximum(peak,ramp(np.minimum(abs(le.mean(1)),abs(ri.mean(1)))-.4*baseline,.021,.105))
        tips+=peak*.5
    # Short-sleeved tee: a symmetric abrupt loss of span at roughly armpit
    # height followed by an almost constant-width, uninterrupted tube.
    transition=np.zeros(len(a)); elbow=np.zeros(len(a))
    for y in range(6,17):
        drop=midwidth[:,y-3:y].mean(1)-midwidth[:,y+1:y+4].mean(1)
        transition=np.maximum(transition,ramp(drop,.12,.34));
        if y>=13: elbow=np.maximum(elbow,ramp(drop,.12,.34))
    tube=1-ramp(midwidth[:,17:25].max(1)-midwidth[:,17:25].min(1),.08,.26)
    tshirt=transition*tube*(1-ramp(asym,.082,.23))*(1-ramp(clasp,.33,.92))*ramp(aspect,.55,.87)
    # A symmetric thin-strapped undershirt is another standard top cut.
    scoop=ramp(c-inner,.070,.26)*(1-ramp(clasp,.4,.95)); tank=scoop*tube*(1-ramp(asym,.09,.27))*ramp(aspect,.46,.64)
    s[:,0]+=.072*tshirt+.046*tank
    s[:,6]+=.026*tips*(.46+.54*clasp)+.012*elbow*(1-tube)-.024*tshirt
    s[:,4]+=.026*pockets*(.35+.65*np.maximum(collar,clasp))
    s[:,2]-=.016*tips*clasp+.012*pockets*clasp
    return s


def predict(x):
    answer=np.empty(len(x),dtype=np.int64)
    # A fixed numeric kernel shape even for the last partial block. No sample
    # participates in any other sample's normalization or vote.
    for start in range(0,len(x),64):
        part=x[start:start+64,0]; n=len(part)
        a=np.zeros((64,28,28),np.float32); a[:n]=part
        m,ink,aspect=prepare(a)
        shape=matches(m,aspect); detail=cues(m,ink,aspect); textile=cut_and_fabric(m,ink,aspect)
        # Hierarchy: buttons, folds and cuffs are meaningful *conditional on*
        # seeing a torso; they must not turn a handbag into a shirt. Likewise a
        # zipper down a boot should not affect its distinction from a jacket.
        tops=[0,2,3,4,6]; shoes=[5,7,9]
        leg=detail[:,1]; torso=shape[:,tops].max(1); footwear=shape[:,shoes].max(1)
        primary=np.stack([torso,shape[:,1]+leg,footwear,shape[:,8]],axis=1).argmax(1)
        # Preserve cut evidence while giving more leeway to shirtlike textures
        # among long sleeves than between, for example, a dress and a blazer.
        longcut=shape[:,[2,4,6]].max(1); shortcut=shape[:,[0,3]].max(1)
        relevant=.24+.76*ramp(longcut-shortcut,-.015,.048)
        weave=detail[:,tops]+textile[:,tops]
        clothes=np.array(tops)[(shape[:,tops]+weave*relevant[:,None]).argmax(1)]
        shoe=np.array(shoes)[(shape[:,shoes]+detail[:,shoes]).argmax(1)]
        result=np.select([primary==0,primary==1,primary==2],[clothes,1,shoe],default=8)
        answer[start:start+n]=result[:n]
    return answer
