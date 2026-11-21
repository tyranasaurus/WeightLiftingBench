"""Build a collection of individual parametric paper cutouts and their matcher.

There are no input files. All lengths below are proportions, not pixel traces.
"""
import math
import sys
from pathlib import Path
import numpy as np

rng = np.random.default_rng(271828)
R = rng.uniform

def pick(*vs):
    return vs[int(rng.integers(len(vs)))]

# Coordinates of the cutout before trim/scale. The paper is supersampled.
S = 56
Y, X = np.mgrid[0:S, 0:S].astype(np.float64)
X = (X + .5) / S
Y = (Y + .5) / S

def curve(x, knots, values):
    return np.interp(x, knots, values)

def panel(x, y, knots, left, right):
    return ((y >= knots[0]) & (y <= knots[-1]) &
            (x >= curve(y, knots, left)) & (x <= curve(y, knots, right)))

def band(x, y, knots, top, bottom):
    return panel(y, x, knots, top, bottom)

def ellipse(x,y,cx,cy,rx,ry):
    return ((x-cx)/rx)**2 + ((y-cy)/ry)**2 < 1

def upper(family):
    # Normalized by garment length, axis running down the spine. Body and
    # sleeves overlap at the armholes. Sleeves can separate from the waist.
    demi = R(.235,.34)                  # half chest relative to length
    shoulder = demi * R(.82,1.04)
    armhole = R(.18,.28)
    waist = demi * R(.82,1.02)
    seat = demi * R(.88,1.10)
    neck = shoulder * R(.37,.64)
    drop = R(.015,.12)
    style = 'short'
    if family == 'tank':
        style = 'none'
        shoulder = demi * R(.57,.94)
        neck = shoulder * R(.56,.90)
        drop = R(.09,.25)
    if family in ('pullover','jacket','overcoat','woven'):
        style = 'long'
    if family == 'blouse':
        style = pick('none','short','elbow','elbow','long')
        seat *= R(1,1.24)
        drop = R(.05,.23)
    if family == 'jacket':
        demi *= R(1,1.09)
        drop = R(0,.045)
    if family == 'overcoat':
        demi *= R(.70,.89)
        shoulder = demi*R(.89,1.05)
        seat,waist = demi*R(1.02,1.24),demi*R(.86,1)
        drop = R(0,.05)
    if family == 'woven':
        drop = R(0,.11)
        style = pick('long','long','long','short','elbow')
    if family == 'dress':
        demi = R(.118,.218)
        shoulder = demi*R(.72,1.03)
        neck = shoulder*R(.39,.78)
        drop = R(.015,.14)
        armhole = R(.11,.195)
        waist = demi*R(.70,.97)
        seat = demi*R(1.04,1.32)
        style = pick('none','none','none','short','short','short','elbow','long')
    # Use uniform units along x and y until the final isotropic rescale.
    stature = R(.89,1.13)
    # Leave headroom below the torso: some cuffs extend below the body hem.
    y = 1.18*Y
    asym = R(-.025,.025)
    center = .5 + asym * (y-.5) + R(-.009,.009)*np.sin(2*np.pi*y)
    x = np.abs(X-center)*1.35/stature
    if family == 'dress':
        bottom = seat * pick(R(.92,1.13),R(1.12,1.53),R(1.45,1.94))
        wprof = curve(y,[0,armhole,.34,.57,1],[shoulder,demi,waist,seat,bottom])
    else:
        if family=='pullover':
            waist = demi * R(.92,1.08)
            seat = demi * R(.90,1.05)
        wprof = curve(y,[0,armhole,.6,.91,1],[shoulder,demi,waist,seat,seat*R(.92,1.02)])
    shoulder_drop = R(.032,.091) * (.65 if family=='dress' else 1)
    if style=='none':
        wprof = np.minimum(wprof,curve(y,[0,armhole*.54,armhole],[shoulder,shoulder*R(.90,1.03),demi]))
    neckshape = pick(.55,1.,1.7)
    topline = np.where(x<neck,drop*np.maximum(0,1-(x/neck)**2)**neckshape,
                       shoulder_drop * np.clip((x-neck)/max(.03,shoulder-neck),0,1))
    # Shirttails, flat knits and slightly wavy hems.
    rounding = R(.035,.125) if family=='woven' else R(0,.037)
    hem = 1-rounding*(x/np.maximum(wprof,.05))**3 + R(-.016,.016)*np.sin(23*(X-center))
    body = (x<wprof) & (y>=topline) & (y<hem)
    if style!='none':
        if style=='long':
            sleeve_end = R(.86,1.09)
            if family=='overcoat': sleeve_end=R(.65,.89)
            if family=='dress': sleeve_end=R(.45,.68)
            if family=='woven': sleeve_end=R(.75,1.03)
            flare = R(.034,.13)
            cuff = R(.075,.145)
        else:
            sleeve_end = R(.24,.43) if style=='short' else R(.45,.68)
            if family=='dress': sleeve_end*=.67
            flare = R(.065,.17) * (.73 if family=='dress' else 1)
            cuff = R(.105,.19) * (.75 if family=='dress' else 1)
        sleeve_out = demi + flare
        sleeve_in = sleeve_out-cuff
        ang = R(.02,.105) if style!='long' else R(-.022,.039)
        # The cuff can have an oblique hem. The outer edge bows a little;
        # the inner edge leaves a narrow triangle of background by the hip.
        midy = min(sleeve_end*.70,armhole)
        knots = [shoulder_drop*.7,midy,sleeve_end-ang,sleeve_end+.015]
        if knots[2]<=midy: knots[2]=(midy+sleeve_end+.015)/2
        outside=[shoulder*.96,demi+flare*R(.38,.92),sleeve_out,sleeve_in+.008]
        inside=[shoulder*.55,demi*R(.69,.89),sleeve_in,sleeve_in]
        body |= panel(x,y,knots,inside,outside)
    # Standing collar on outerwear or a button shirt, usually projecting up.
    # Compress shoulders slightly downward to leave room for it.
    if family in ('jacket','overcoat','woven') and R()<.83:
        shift=int(R(.025,.093)*S/1.18)+1
        body=np.roll(body,shift,axis=0)
        body[:shift]=False
        cwidth=shoulder*R(.40,.76)
        body |= (x<cwidth*(.82+.26*np.minimum(Y*S/shift,1))) & (y < 1.18*shift/S+shoulder_drop+.025)
    return body, 1.35, 1.18

def trousers():
    y=Y
    x=np.abs(X-(.5+R(-.024,.024)*(y-.5)))*.76
    hip=R(.135,.236)
    crotch=R(.29,.48)
    # Include tapered jeans and trousers displayed with the legs together.
    floor=hip*R(.46,1.12)
    outer=curve(y,[0,.12,crotch,.67,1],[hip*R(.75,.98),hip,hip*R(.82,1.07),.52*floor+.48*hip,floor])
    gap=pick(R(.001,.021),R(.018,.078))
    inner=curve(y,[0,crotch,crotch+.12,1],[-.001,-.001,gap*R(.18,.65),gap])
    inner=inner+R(-.012,.012)*np.sin(4*y)
    cut=(x<outer)&((y<crotch)|(x>inner))
    cut &= y < 1-R(0,.035)*(X>.5)
    return cut,.76,1

def bag(style):
    # A pouch, optionally with one or two narrow hoop handles or a strap.
    x = X + R(-.055,.055)*(Y-.5)
    y = Y
    handle = R(.22,.60) if style=='tote' else R(.05,.36) if style=='purse' else R(0,.12)
    wh=R(.94,1.78) if style=='case' else R(.67,1.60) if style=='purse' else R(.71,1.28)
    width=R(.45,.50)
    top=width*R(.77,1.09)
    base=width*R(.70,1.01)
    shoulder=handle+.14*(1-handle)
    rad=curve(y,[handle,shoulder,.54+.46*handle,.96,1],[top*.90,width,width,base,base*.85])
    sag=R(-.035,.12)*(1-(np.abs(x-.5)/width)**2)
    body=(np.abs(x-.5)<rad)&(y>=handle+sag)&(y<1-.025*((x-.5)/width)**2)
    if handle>.015:
        offset=R(-.085,.085)
        rx=R(.18,.37)
        ry=handle*R(.96,1.37)
        thick=R(.015,.057) if style!='case' else R(.025,.075)
        def hoop(dx,dy):
            ex=x-.5-dx-offset*(1-y)
            ey=y-ry-dy
            outer=(ex/rx)**2+(ey/ry)**2<1
            inner=(ex/max(.015,rx-thick))**2+(ey/max(.01,ry-thick*wh))**2<1
            return outer&~inner&(ey<.22)
        body |= hoop(0,0)
        if R()<.35: body |= hoop(R(-.09,.09),R(.02,.10))
    return body,wh,1

def footwear(style):
    # Toe on the left; heights expressed in fractions of heel-to-toe length.
    # Shoe base may be flat, a wedge, a separate block heel or a thin heel.
    x=X
    if style=='sneaker':
        height=R(.29,.53)
    elif style=='hightop':
        height=R(.46,.69)
    elif style=='boot':
        height=R(.51,.92)
    elif style=='tallboot':
        height=R(.87,1.24)
    elif style=='heeled':
        height=R(.46,.93)
    else:
        height=R(.18,.50)
    # h above the ground, keeping a small curved sole at the toe and heel.
    h=(1-Y)*height
    toe=R(.05,.13); heelx=R(.78,.90)
    sole_thick=R(.04,.113)
    heels = style=='heeled' or (style in ('boot','tallboot') and R()<.75)
    heel_lift=R(.105,min(.48,height*.65)) if heels else R(.0,.052)
    arch=curve(x,[0,.08,.21,.38,heelx,.97,1],[.045,.011,0,.002,heel_lift,heel_lift+.012,heel_lift+.033])
    sole=(h>=arch)&(h<arch+sole_thick)&(x>.005)&(x<.994)
    if heels:
        heelw=R(.07,.25)
        heelctr=R(.86,.94)
        heel=panel(x,h,[0,heel_lift+.012,heel_lift+sole_thick],
                   [heelctr-heelw*.42,heelctr-heelw*.58,heelctr-heelw*.70],
                   [min(.996,heelctr+heelw*.46),.985,.985])
        sole |= heel
    if style in ('sneaker','hightop'):
        tongue=R(.46,.68)
        peak=height*R(.87,1)
        counter=R(.84,.95)
        notch=tongue+R(.32,.83)*(counter-tongue)
        topline=curve(x,[0,.05,.18,tongue-.11,tongue,notch,counter,.986,1],
            [arch[0,0]+sole_thick,height*R(.18,.47),height*R(.28,.59),peak*R(.66,.89),peak,height*R(.53,.91),height*R(.67,1.0),height*R(.40,.73),height*.30])
        upper=(h>=arch)&(h<topline)&(x>.008+.030*(h/height)**2)&(x<.996-.044*(h/height)**3)
        return sole|upper,1,height
    if style in ('boot','tallboot'):
        shaft=R(.49,.71)
        back=R(.88,.995)
        vamp=R(.12,.23)
        upper=band(x,h,[.008,.06,.22,shaft-.08,shaft,shaft+.035,back, .998],
            [.055,.033,.012,.014,.02,.02,.02,.06],
            [sole_thick+.033,vamp,vamp*R(.94,1.2),max(vamp+.06,height*.40),height*R(.64,.87),height,height*R(.87,1.04),vamp+heel_lift])
        upper &= h>=arch
        return sole|upper,1,height
    # Sandals: straps crossing over the forefoot with an open instep;
    # optional rear counter and ankle strap. Slides have only a wide band.
    if style=='flat':
        # Ballet flats and open pumps have a pointed forepart, low central
        # topline and often a small counter; no laced tongue over the instep.
        vamp=R(.16,.33)
        back=R(.86,.97)
        upper=band(x,h,[.006,.065,vamp,vamp+.12,back-.16,back,.992],
                     [.04,.015,0,0,0,.018,.043],
                     [sole_thick+.031,height*.47,height*R(.64,1),height*R(.26,.54),height*.44,height,height*.72])
        return sole|(upper&(h>=arch)),1,height
    front=R(.22,.42)
    def strap(a,b,c,w,up):
        # Three points describe a strap's high point and two attachment points.
        z=curve(x,[a,b,c],[0,up,0])
        return (x>a)&(x<c)&(h<arch+sole_thick+z)&(h>arch+sole_thick+z-w)
    if style=='slide':
        rising = curve(x,[.02,front,front+.20,.96],[.005,height-sole_thick,.0,0])
        upper=(h<arch+sole_thick+rising)&(h>=arch)
        if R()<.50: upper &= ~ellipse(x,h,front+.13,arch[0,S//2]+sole_thick,.088,height*.44)
    else:
        toepeak = max(.045,R(.16,.30)-sole_thick) if style=='heeled' else (height-sole_thick)*R(.45,1)
        upper=strap(R(.015,.15),front,R(.45,.65),R(.04,.14),toepeak)
        if R()<.62: upper |= strap(.15,front+.10,.67,R(.03,.08),toepeak*R(.62,.98))
        rear=R(.69,.86)
        rearh=max(.05,height-heel_lift-sole_thick)
        upper |= strap(R(.36,.66),rear,.985,R(.033,.13),rearh)
        if R()<.63:
            upper |= (x>.87-.10*(h/height))&(x<.98-.05*(h/height))&(h>=arch)&(h<height*R(.76,1))
        if style=='heeled' and R()<.74:
            upper |= (x>rear-.17)&(x<.95)&(h>height*.86)&(h<height*.98)
    return upper|sole,1,height

def represent(shape, physw, physh):
    iy,ix=np.nonzero(shape)
    l,r=ix.min(),ix.max()+1
    t,b=iy.min(),iy.max()+1
    # Each paper model is centered on its own bbox. Encode soft occupancy in
    # a 16 x 16 uniform grid, plus horizontal/vertical extent projections.
    # Downsampling uses supersampling within each grid cell.
    lr=np.log((r-l)*physw/((b-t)*physh))
    # Render on a square-pixel camera before computing projections. This is
    # especially important for narrow objects: a one-pixel gap between two
    # trouser legs is very different from a continuous vector-space slit.
    scale=R(25,28)/max((r-l)*physw,(b-t)*physh)
    w=max(2,round((r-l)*physw*scale)); h=max(2,round((b-t)*physh*scale))
    phaseX=R(-.25,.25); phaseY=R(-.25,.25)
    cx=(l+((np.arange(w*4)+.5)/4+phaseX)*(r-l)/w).astype(int).clip(0,S-1)
    cy=(t+((np.arange(h*4)+.5)/4+phaseY)*(b-t)/h).astype(int).clip(0,S-1)
    pix=shape[np.ix_(cy,cx)].reshape(h,4,w,4).mean((1,3))
    # Edge mixing varies with garment/background contrast and antialiasing.
    pix=np.clip((pix-R(.035,.22))/R(.36,.71),0.,1.)
    ux=((np.arange(48)+.5)*w/48-.5).clip(0,w-1.0001)
    uy=((np.arange(48)+.5)*h/48-.5).clip(0,h-1.0001)
    ix=ux.astype(int); iy=uy.astype(int); dx=ux-ix; dy=uy-iy
    row=(1-dx)*pix[:,ix]+dx*pix[:,ix+1]
    m=((1-dy[:,None])*row[iy,:]+dy[:,None]*row[iy+1,:]).astype(np.float32)
    # Apply the same canonical orientation used at inference (in particular,
    # slides with a single front vamp can be taller at the toe than heel).
    if m[:20,:21].sum()>m[:20,-21:].sum(): m=m[:,::-1]
    return descriptor(m),lr

def descriptor(m):
    c=m.reshape(16,3,16,3).mean(axis=(1,3))
    row=m.mean(axis=1).reshape(16,3).mean(axis=1)
    col=m.mean(axis=0).reshape(16,3).mean(axis=1)
    env=[]
    for a in (m,m.T):
        hit=a>.32
        has=hit.any(1)
        low=np.where(has,hit.argmax(1)/48,.5)
        high=np.where(has,1-hit[:,::-1].argmax(1)/48,.5)
        env.extend([low.reshape(16,3).mean(1),high.reshape(16,3).mean(1)])
    return np.concatenate([c.ravel(),2*row,2*col]+[1.6*p for p in env]).astype(np.float32)

families=[('tee',1100),('tank',450),('trouser',640),('pullover',1300),('dress',1900),
          ('jacket',1150),('overcoat',700),('sandal',850),('slide',550),('heeled',850),
          ('woven',1150),('blouse',750),('sneaker',950),('hightop',440),('tote',650),
          ('purse',650),('case',640),('boot',1100),('tallboot',520),('flat',680)]

def generate():
    features=[]; aspects=[]; starts=[]
    for name,count in families:
        starts.append(len(features))
        for i in range(count):
            if name=='trouser': model=trousers()
            elif name in ('case','purse','tote'): model=bag(name)
            elif name in ('sneaker','hightop','boot','tallboot','sandal','slide','heeled','flat'): model=footwear(name)
            else: model=upper(name)
            v,a=represent(*model)
            features.append(v); aspects.append(a)
    starts.append(len(features))
    return dict(shapes=np.stack(features),aspect=np.array(aspects,dtype=np.float32),ranges=np.array(starts))

source = '''import numpy as np

# The cutouts are compared separately, never averaged within or between classes.
templates = CONSTANTS['shapes']
lengths = np.einsum('ij,ij->i',templates,templates)
aspects = CONSTANTS['aspect']
splits = CONSTANTS['ranges']

def unit(z):
    return np.clip(z,0.,1.)

def envelope(a):
    hit=a>.32
    has=hit.any(2)
    l=np.where(has,hit.argmax(2)/48,.5)
    r=np.where(has,1-hit[:,:,::-1].argmax(2)/48,.5)
    return l,r

def descriptor(m):
    n=len(m)
    c=m.reshape(n,16,3,16,3).mean(axis=(2,4)).reshape(n,256)
    row=m.mean(2).reshape(n,16,3).mean(2)
    col=m.mean(1).reshape(n,16,3).mean(2)
    env=[]
    for a in (m,m.transpose(0,2,1)):
        l,r=envelope(a)
        env.extend([1.6*l.reshape(n,16,3).mean(2),1.6*r.reshape(n,16,3).mean(2)])
    return np.concatenate([c,2*row,2*col]+env,axis=1).astype(np.float32)

def resample(a,left,right,top,bottom):
    g=(np.arange(48)+.5)/48
    xx=left[:,None]+(right-left)[:,None]*g-.5
    yy=top[:,None]+(bottom-top)[:,None]*g-.5
    xx=np.clip(xx,0,26.99999); yy=np.clip(yy,0,26.99999)
    ix=xx.astype(int); iy=yy.astype(int)
    wx=(xx-ix)[:,None,:]; wy=(yy-iy)[:,:,None]
    idx=np.arange(len(a))[:,None,None]
    ix=ix[:,None,:]; iy=iy[:,:,None]
    pair1=(1-wx)*a[idx,iy,ix]+wx*a[idx,iy,ix+1]
    pair2=(1-wx)*a[idx,iy+1,ix]+wx*a[idx,iy+1,ix+1]
    return ((1-wy)*pair1+wy*pair2).astype(np.float32)

def blur(a):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    return (.25*a+.125*(p[:,1:-1,:-2]+p[:,1:-1,2:]+p[:,:-2,1:-1]+p[:,2:,1:-1])
            +.0625*(p[:,:-2,:-2]+p[:,:-2,2:]+p[:,2:,:-2]+p[:,2:,2:]))

def prepare(x):
    # Gentle local hole repair at sensor scale. Preserve openings in handles,
    # between trouser legs, or under a heel. Use a low black-point for dark cloth.
    peak=x.max(axis=(1,2))[:,None,None]
    t=.014+.043*peak
    occ=unit((x-.32*t)/(1.35*t))
    neigh=blur(occ)
    occ=np.maximum(.78*occ+.22*neigh,unit((neigh-.22)/.57)*.78)
    hit=(occ>.37)
    row=hit.sum(2)>=2
    col=hit.sum(1)>=2
    l=np.where(col.any(1),col.argmax(1),0)
    r=np.where(col.any(1),28-col[:,::-1].argmax(1),28)
    t=np.where(row.any(1),row.argmax(1),0)
    b=np.where(row.any(1),28-row[:,::-1].argmax(1),28)
    m=resample(occ,l,r,t,b)
    raw=resample(x,l,r,t,b)
    # In profile the tall ankle is ordinarily opposite the toe. Orient such
    # asymmetric objects consistently before comparing the individual models.
    tilt=m[:,:20,:21].sum((1,2))-m[:,:20,-21:].sum((1,2))
    flip=(tilt>0)[:,None,None]
    return (np.where(flip,m[:,:,::-1],m), np.where(flip,raw[:,:,::-1],raw),
            ((r-l)/(b-t)).astype(np.float32))

def match(m,ratio):
    v=descriptor(m)
    # Squared Euclidean distances between individual grid descriptions. A
    # fixed row count also fixes the BLAS path for the final partial batch.
    dist=np.einsum('ij,ij->i',v,v)[:,None]+lengths[None,:]-2*(v@templates.T)
    dist+=98*(np.log(ratio[:,None])-aspects[None,:])**2
    return np.stack([-dist[:,a:b].min(1)*.22 for a,b in zip(splits[:-1],splits[1:])],axis=1)

def meanbox(a,y1,y2,x1,x2):
    return a[:,y1:y2,x1:x2].mean(axis=(1,2))

def hump(x,low,high):
    return unit((x-low)/(high-low))

def cues(m,gray,ratio):
    n=len(m)
    mean=m.mean((1,2))
    cloth=gray.sum(axis=(1,2))/(m.sum(axis=(1,2))+1)
    # Compress the dynamic range but retain locally salient seams and borders.
    v=np.log1p(gray/(.035+.48*cloth[:,None,None]))
    # Continuous vertical contrast near the center. Search a small range of
    # slopes and offsets: a zipper or button stand may not be exactly straight.
    placket=np.zeros(n)
    yy=np.arange(48)
    for off in (-3,-1,0,1,3):
        for lean in (-3,0,3):
            xx=24+off+np.rint(lean*(yy-24)/24).astype(int)
            middle=.5*(v[:,yy,xx]+v[:,yy,xx-1])
            sideL=.5*(v[:,yy,xx-3]+v[:,yy,xx-4])
            sideR=.5*(v[:,yy,xx+2]+v[:,yy,xx+3])
            linear=middle-.5*(sideL+sideR)
            # Both contrasts of a stripe should have the same sign over a
            # substantial length. Also allow a broad two-tone opening.
            strip=0.; jump=0.
            for start in (12,21,30):
                strip=strip+linear[:,start:start+9].mean(1)/3
                jump=jump+(sideL-sideR)[:,start:start+9].mean(1)/3
            repeated=(np.abs(linear[:,12:39]).mean(1)+np.abs(strip))*.5
            strength=np.maximum(repeated,.60*np.abs(jump))
            placket=np.maximum(placket,strength)
    dx=np.abs(v[:,:,4:]-v[:,:,:-4]); dy=np.abs(v[:,4:,:]-v[:,:-4,:])
    gx=meanbox(dx,14,39,13,31); gy=meanbox(dy,12,37,15,33)
    local=(gx+gy)*.5
    # Discount repeated stripes or print: they are not evidence of a fastening.
    fast=hump(placket-.38*gx,.018,.16)
    quiet=1-hump(local,.039,.17)
    noisy=hump(local,.085,.25)
    highdetail=(meanbox(dx,3,16,11,33)+meanbox(dy,1,14,13,35))*.5
    lapel=hump(highdetail-.5*local,.045,.22)
    # Vertical changes in the height of the shoulder (standing collar), and
    # the opposite sign for a low, scooped neckline.
    top,base=envelope(m.transpose(0,2,1))
    left,right=envelope(m)
    shoulder=.5*(top[:,9:15].mean(1)+top[:,33:39].mean(1))
    stand=hump(shoulder-top[:,19:29].mean(1),.045,.17)
    scoop=hump(top[:,20:28].mean(1)-.5*(top[:,13:18].mean(1)+top[:,30:35].mean(1)),.012,.15)
    span=right-left
    # For shirttails compare the center hem to the hem on either side, but
    # stay inside the body rather than sampling the gap next to a sleeve.
    tails=hump(base[:,20:28].mean(1)-.5*(base[:,15:19].mean(1)+base[:,29:33].mean(1)),.028,.11)
    pocketxy=meanbox(dx,20,34,12,31)
    waistdetail=hump(pocketxy,.10,.27)
    # Soft evidence of a horizontal ribbed waistband, common on knit pullovers.
    hemline=hump(meanbox(dy,37,43,15,33)-gy,.025,.18)
    # Differential shape cues for dresses vs tops: broadening below the waist,
    # slim bodices, long skirts. Row medians ignore wavy seam ends.
    flare=hump(np.median(span[:,37:45],axis=1)-np.median(span[:,17:27],axis=1),.06,.30)
    dresslike=flare+.6*hump(.69-ratio,.02,.24)
    # Two separate branches extending from waistband to the feet.
    midgap=1-meanbox(m,31,46,21,27)
    pair=.5*(meanbox(m,30,44,8,17)+meanbox(m,30,44,31,40))
    leg=midgap*pair
    # Holes within the outer contour, measured column-wise; a closed shoe is
    # normally filled from vamp to sole even when ankle is very low.
    ys=(np.arange(48)+.5)[None,:,None]/48
    within=(ys>top[:,None,:]+.022)&(ys<base[:,None,:]-.022)
    hollow=((1-m)*within).sum((1,2))/(within.sum((1,2))+1.)
    def voidpart(y1,y2,x1,x2):
        z=within[:,y1:y2,x1:x2]
        return (z*(1-m[:,y1:y2,x1:x2])).sum((1,2))/(z.sum((1,2))+1)
    air=hump(hollow,.045,.26)
    strap=hump(voidpart(6,38,5,44),.09,.35)
    # Heel separated from the ground at the arch, and profile rises toward
    # the back. Deep, nearly upright shaft distinguishes a boot from a tongue.
    under=hump(.5*(base[:,6:16].mean(1)+base[:,39:45].mean(1))-base[:,25:34].mean(1),.045,.21)
    rear=1-top[:,35:44].mean(1)
    toe=1-top[:,4:13].mean(1)
    ankle=hump(rear-toe,.22,.64)
    notch=hump(top[:,33:38].mean(1)-top[:,24:31].min(1),.09,.32)
    flatback=hump(rear,.72,.94)*(1-notch)
    # One-piece crossed vamp of a slide, no material at all at rear upper.
    slide=hump(top[:,35:44].mean(1)-top[:,13:22].mean(1),.22,.55)
    shoeasym=meanbox(np.abs(m-m[:,:,::-1]),0,48,0,48)
    symmetric=1-hump(shoeasym,.085,.28)
    # Handle: background above a broad body and below the arch of a thin rim.
    handle=hump(voidpart(1,27,12,36),.20,.65)*hump(meanbox(m,31,42,9,39),.59,.91)
    square=hump(mean,.69,.88)*hump(meanbox(m,3,10,2,8)+meanbox(m,3,10,40,46),.40,1.38)
    return (fast,quiet,noisy,lapel,stand,scoop,tails,waistdetail,hemline,dresslike,
            leg,air,strap,under,ankle,notch,flatback,slide,symmetric,handle,square)

def tailoring(m,raw,ratio):
    # Plaid runs through the fabric on the lateral panels, unlike a central
    # printed graphic. Look for numerous alternating changes in both axes.
    v=np.log1p(raw/(.03+.40*(raw.sum((1,2))/(m.sum((1,2))+1))[:,None,None]))
    def variation(a,step):
        num=0.; den=0.
        for l,r in ((11,19),(29,37)):
            patch=a[:,12:38,l:r]
            ok=m[:,12:38,l:r]>.8
            diff=patch[:,step:,:]-patch[:,:-step,:]
            mask=ok[:,step:,:]&ok[:,:-step,:]
            num=num+(np.abs(diff)*mask).sum((1,2))
            den=den+mask.sum((1,2))
        return num/(den+1.)
    ver=variation(v,3)
    def lateral(a):
        return .5*(np.abs(a[:,14:38,11:18]-a[:,14:38,14:21]).mean((1,2))+
                   np.abs(a[:,14:38,27:34]-a[:,14:38,30:37]).mean((1,2)))
    horiz=lateral(v)
    plaid=hump(np.sqrt(ver*horiz),.065,.20)
    # Thin fabric makes long vertical folds; a coat is more structured and
    # often has an uninterrupted front fastening and a standing collar.
    drape=hump(horiz-.70*ver,.024,.105)
    # Visible contrast across the end of a long sleeve or a folded cuff.
    endflare=0.
    for a,b in ((3,10),(38,45)):
        d=np.abs(v[:,32:43,a:b]-v[:,35:46,a:b])
        overlap=(m[:,32:43,a:b]>.75)&(m[:,35:46,a:b]>.75)
        endflare=endflare+.5*hump((d*overlap).sum((1,2))/(overlap.sum((1,2))+2),.07,.19)
    # Relative drop of the body hem and sleeve ends. Work from local column
    # bottoms, so these measurements also work for embroidered fabrics.
    top,base=envelope(m.transpose(0,2,1))
    bodyend=base[:,19:29].mean(1)
    cuffend=np.maximum(base[:,:9].max(1),base[:,-9:].max(1))
    coatlength=hump(bodyend-cuffend,.035,.24)*hump(cuffend,.53,.77)
    overhang=hump(cuffend-bodyend,.025,.13)
    # Sleeve openings have their own dark troughs below the armpits. Locate
    # them first, then estimate the shape of the hem relative to the torso.
    panel=m[:,26:41].mean(1)
    left=panel[:,5:16].argmin(1)+5
    right=panel[:,32:43].argmin(1)+32
    row=np.arange(len(m))
    tailbend=bodyend-.5*(base[row,left+3]+base[row,right-3])
    shirttail=hump(tailbend,.027,.13)*hump(cuffend,.51,.75)
    # Find the zipper-to-collar V and a button shirt's two collar points. A
    # V is two diagonal ridges converging at the same position on the chest.
    ys=np.arange(14)
    vneck=np.zeros(len(m))
    for depth in (10,17,23):
        for spread in (4,7,10):
            yy=np.rint(3+(depth-3)*ys/13).astype(int)
            sep=np.rint(spread*(1-ys/13)).astype(int)
            rr=24+sep; ll=23-sep
            contrast=.5*(v[:,yy,rr]-v[:,yy,rr+3]+v[:,yy,ll]-v[:,yy,ll-3])
            vneck=np.maximum(vneck,np.abs(contrast.mean(1)))
    vneck=hump(vneck,.10,.34)
    # Trouser waistlines are broad and nearly level across the top, above
    # tapering hips; shirts, by contrast, slope from neck to shoulder.
    leftedge,rightedge=envelope(m)
    span=rightedge-leftedge
    belt=hump(span[:,1:4].mean(1)/(.04+span[:,8:17].mean(1)),.68,.92)
    skinnylegs=hump(.61-ratio,.01,.24)*belt
    taper=hump(span[:,9:15].mean(1)-span[:,38:43].mean(1),.04,.32)
    # Even if the legs touch, the central division can leave a vertical seam.
    cent=v[:,25:44,17:31].min(2).mean(1)
    thighs=.5*(meanbox(v,25,44,10,16)+meanbox(v,25,44,32,38))
    legseam=hump(thighs-cent,.18,.64)*belt
    return plaid,drape,endflare,coatlength,overhang,shirttail,vneck,skinnylegs,taper,legseam

def shoeparts(m,raw,ratio):
    # A sneaker's welt tends to be a distinct band parallel to its *own*
    # outsole. Following the outsole avoids confusing a sloping vamp with it.
    top,base=envelope(m.transpose(0,2,1))
    cols=np.arange(5,44)
    row=np.arange(len(m))[:,None]
    floor=np.clip(np.rint(48*base[:,cols]).astype(int)-1,0,47)
    vv=np.log1p(raw/(.035+.48*(raw.sum((1,2))/(m.sum((1,2))+1))[:,None,None]))
    welt=np.zeros(len(m))
    for dy in (3,6,9,12):
        v1=vv[row,np.maximum(0,floor-dy),cols]
        v2=vv[row,np.maximum(0,floor-dy-5),cols]
        welt=np.maximum(welt,np.abs((v1-v2).mean(1)))
    welt=hump(welt,.08,.35)
    # A flat sneaker has a shoe tongue in front of an opening. The lowest
    # part of that opening should be between the tongue and the heel tab.
    rearDip=top[:,30:41].max(1)
    tongueDip=hump(np.minimum(rearDip-top[:,20:31].min(1),rearDip-top[:,40:46].min(1)),.035,.25)
    rise=1/(ratio+.0001)
    trainer=(1-hump(rise,.42,.65))*(.55+.45*welt)+.8*tongueDip
    # A sandal may have the heel cup AND a toe strap, with a deep V-shaped
    # gap open to the sky in between: not captured by an enclosed-hole test.
    scallop=top[:,23:35].max(1)-np.maximum(top[:,7:22].min(1),top[:,37:46].min(1))
    opening=hump(scallop,.22,.57)
    toeLow=hump(top[:,3:12].mean(1),.64,.88)
    shaft=hump(rise,.43,.72)*hump(1-top[:,33:43].mean(1),.76,.95)
    return trainer,opening,toeLow,shaft

def optical(m,raw,ratio):
    # Use an illumination-relative silhouette as a second view. Some black
    # shoes are mostly piping and highlights; thresholding them at a fixed
    # fraction of the brightest pixel erases the material between the seams.
    level=raw.sum((1,2))/(m.sum((1,2))+1.)
    low=.004+.022*level[:,None,None]
    ink=blur(unit((raw-.25*low)/(1.3*low)))
    # Frame follows occupied pixels rather than recropping this second view.
    alt=np.maximum(m,ink)
    return alt

def surface(m,raw,ratio):
    v=np.log1p(raw/(.028+.45*(raw.sum((1,2))/(m.sum((1,2))+1))[:,None,None]))
    # Contrast of a narrow placket at the actual sensor resolution, compared
    # with other vertical edges in the same cross-section. Count support down
    # the torso rather than treating one printed motif as a front fastening.
    contrast=np.maximum(np.abs(v[:,:,5:-1]-v[:,:,1:-5]),np.abs(v[:,:,6:]-v[:,:,:-6]))
    contrast=(contrast[:,:,:-1]+contrast[:,:,1:])*.5
    peak=0.
    for shift in (-2,0,2):
        for slant in (-2,0,2):
            yy=np.arange(12,42)
            xx=21+shift+np.rint(slant*(yy-25)/16).astype(int)
            ridge=np.maximum(contrast[:,yy,xx-1],contrast[:,yy,xx+1])
            flank=.25*(contrast[:,yy,xx-8]+contrast[:,yy,xx-6]+contrast[:,yy,xx+6]+contrast[:,yy,xx+8])
            relative=unit((ridge-.7*flank-.018)/(.035+.47*flank))
            support=np.minimum(relative[:,:15].mean(1),relative[:,15:].mean(1))
            peak=np.maximum(peak,support)
    # Variation spread across a thin woven shirt; exclude the center where a
    # printed tee or jumper can carry a high-contrast graphic.
    rms=0.
    for x0,x1 in ((10,19),(29,38)):
        patch=v[:,15:40,x0:x1]
        ok=m[:,15:40,x0:x1]>.75
        w=ok.sum((1,2))+1
        avg=(patch*ok).sum((1,2))/w
        rms+=.5*np.sqrt((((patch-avg[:,None,None])**2)*ok).sum((1,2))/w)
    rumple=hump(rms,.067,.21)
    # An irregular raised edge sticks out beyond the shoulder line for a
    # bulky stand collar. Compare at several widths, from the collar to the
    # AC joint; do not mistake empty outer corners for a raised collar.
    top,end=envelope(m.transpose(0,2,1))
    ridge=top[:,18:30].mean(1)
    slow=.5*(top[:,10:15].mean(1)+top[:,33:38].mean(1))
    inner=.5*(top[:,14:17].mean(1)+top[:,31:34].mean(1))
    pillar=hump(.5*(inner+slow)-ridge,.026,.113)
    neckline=hump(ridge-inner,.025,.12)
    shoulder=.5*(top[:,8:11].mean(1)+top[:,37:40].mean(1))
    collarstep=hump((inner-ridge)-.32*(shoulder-inner),.012,.072)
    # Sweater hem or skirt/jacket hem? The bottom of the body of a pullover
    # can stop well above the cuffs; a coat often reaches below the hands.
    body=np.median(end[:,17:31],axis=1)
    cuff=.5*(end[:,:10].max(1)+end[:,-10:].max(1))
    sleeved=hump(cuff,.57,.81)
    ribcut=hump(cuff-body,.018,.105)*sleeved
    short=hump(body-cuff,.35,.61)
    # Shirt collars, unlike round necks, often form two pointed high-contrast
    # tabs or part around a V. Assess light/dark lapels relative to the chest.
    center=v[:,4:14,21:27].mean((1,2))
    wings=.5*(meanbox(v,4,14,15,20)+meanbox(v,4,14,28,33))
    chest=meanbox(v,16,26,16,32)
    collarink=hump(np.abs(center-wings)+.45*np.abs(wings-chest),.09,.34)
    # A bright patch only on the sternum is not the same as all-over plaid.
    field=v[:,13:32,18:30]
    motif=hump(field.std((1,2))-.78*rms,.075,.25)
    return peak,rumple,pillar,neckline,collarstep,sleeved,ribcut,short,collarink,motif

def construction(m,raw):
    # Sleeve seams can still be visible as shading when the sleeves touch the
    # body. Compare sleeve length with the inner hem rather than mistaking
    # arbitrary lower corners of a wide tank top for cuffs.
    level=raw.sum((1,2))/(m.sum((1,2))+1)
    v=np.log1p(raw/(.025+.40*level[:,None,None]))
    top,base=envelope(m.transpose(0,2,1))
    hem=np.median(base[:,18:30],axis=1)
    cuffs=np.minimum(base[:,1:11].max(1),base[:,37:47].max(1))
    separate=.5*((1-m[:,26:43,6:18].mean(1)).max(1)+(1-m[:,26:43,30:42].mean(1)).max(1))
    # An open-front coat/jacket can have a broad facing on each side of its
    # center line. Take contrast relative to flank folds in the same garment.
    split=0.; line=0.
    for k in (20,22,24,26,28):
        c=v[:,16:43,k-1:k+1].mean(2)
        a=v[:,16:43,k-6:k-3].mean(2); b=v[:,16:43,k+3:k+6].mean(2)
        mid=c-.5*(a+b)
        # Differences repeated above and below the belt are more diagnostic
        # than contrast confined to a printed chest graphic.
        signed=np.abs(mid.mean(1))
        multi=np.minimum(np.abs(mid[:,:13]).mean(1),np.abs(mid[:,13:]).mean(1))
        ref=.5*(np.abs(v[:,16:43,11]-v[:,16:43,15]).mean(1)+np.abs(v[:,16:43,36]-v[:,16:43,32]).mean(1))
        line=np.maximum(line,hump(.5*(signed+multi)-.35*ref,.014,.102))
        asym=a-b
        split=np.maximum(split,hump(np.abs(asym.mean(1))-.4*ref,.06,.19))
    cuffbelow=hump(cuffs-hem,.008,.12)*hump(separate,.25,.72)
    # Standing collar relative to a tangent to the shoulder, independent of
    # the overall slope of that shoulder. A knit crew neck has a dip instead.
    ridge=np.median(top[:,18:30],axis=1)
    sides=.5*(np.median(top[:,12:17],axis=1)+np.median(top[:,31:36],axis=1))
    collar=hump(sides-ridge,.013,.097)
    # Fraction of each lower row occupied by the central torso (sleeve gaps
    # and torso width are independent of whether the silhouette is tapered).
    midwidth=m[:,25:38].mean((1,2))
    slimbody=1-hump(midwidth,.68,.91)
    return line,split,cuffbelow,collar,slimbody

def classify(x):
    m,raw,ratio=prepare(x)
    fam=match(m,ratio)
    alt=optical(m,raw,ratio)
    other=match(alt,ratio)
    # For dark cloth either mask can be right. For very open shapes one
    # should not have to fill in the empty space just to match a template.
    fam=.32*fam+.68*np.maximum(fam,other-.45)
    (fast,quiet,noisy,lapel,stand,scoop,tails,pocket,hemline,dresslike,
     leg,air,strap,under,ankle,notch,flatback,slide,symmetric,handle,square)=cues(m,raw,ratio)
    plaid,drape,cuff,skirt,overhang,tailbend,vneck,slim,taper,legseam=tailoring(m,raw,ratio)
    trainer,opening,toeLow,shaft=shoeparts(m,raw,ratio)
    peak,rumple,pillar,neckline,step,sleeved,ribcut,short,collarink,motif=surface(m,raw,ratio)
    # First decide what anatomy the silhouette has. No coefficient for any
    # interior shirt feature participates in the bag, dress or trouser test.
    tshape=np.maximum(fam[:,0],fam[:,1]-.25)
    outer=np.maximum(fam[:,5],fam[:,6]-.22)
    bshape=np.maximum(fam[:,10],fam[:,11]-.20)
    sweater=fam[:,3]
    upper=np.maximum(np.maximum(tshape,sweater),np.maximum(outer,bshape))
    shoes=fam[:,[7,8,9,12,13,17,18,19]].max(1)
    case=fam[:,14:17].max(1)
    # Being narrow is not evidence for pants by itself: a skirt can present
    # the same aspect ratio. Require the putative waistband AND two legs.
    pants=fam[:,2]+3.3*leg+slim*(1.3+2*taper)*np.maximum(leg,legseam)+1.3*legseam-.8*(1-leg)
    coarse=np.stack([upper+.95,pants,
                     fam[:,4]+1.55*dresslike+.55-1.6*leg,shoes+.36-.9*symmetric,
                     case+.50+2.2*handle+.95*square+.48*symmetric],axis=1).argmax(1)
    # Bound the influence of surface models. Evidence for an opening on a
    # jersey does not remove the basic evidence for a T-shaped cut.
    L=hump(sleeved,.07,.87)*(1-hump(tshape-np.maximum(sweater,outer),.5,5))
    common=np.maximum(np.maximum(sweater,outer),fam[:,10])
    def shared(s):
        return np.maximum(s,common-5)*.32+common*.68
    # Sweaters favor a straight, ribbed hem sitting ABOVE the sleeves. Shirts
    # often drape past the cuff, especially at the center of a curved hem.
    sewnline=hump(peak,.28,.81)
    closure=.5*fast+.5*sewnline
    stiff=.50*pillar+.25*step+.25*stand
    longknit=shared(sweater)+.43+1.56*ribcut+.52*hemline+.52*quiet+.32*motif-2.70*closure-.88*stiff-.55*rumple
    jacket=shared(outer)-.25+2.22*closure+1.68*stiff+.68*skirt+.55*vneck-.72*neckline-.87*tailbend-.54*motif
    button=shared(bshape)-.42+1.28*closure+.98*rumple+1.18*tailbend+.55*collarink+.62*plaid+.57*drape+.45*cuff-.95*stiff-.93*ribcut-.39*hemline
    line,split,cuffbelow,collar,slimbody=construction(m,raw)
    # Light woven shirts have less sharply bounded knit welts and are often
    # slim; coats have comparatively stiff fronts and built-up collars.
    # Set a plain pullover apart by requiring an unbroken front panel.
    longknit+=.75*cuffbelow+.4*motif-1.6*line-.80*collar
    jacket+=1.45*line+1.1*collar+1.1*split-.55*slimbody-.70*cuffbelow
    button+=1.15*line+.75*slimbody+.72*rumple+.60*drape-.65*split-.48*collar
    # A ribbed waistband is a horizontal line INSIDE the material. Ignore
    # the contrast from background just below the hem when evaluating it.
    bandmask=(m[:,38:43,15:33]>.82)&(m[:,42:47,15:33]>.82)
    bandedge=np.abs(raw[:,38:43,15:33]-raw[:,42:47,15:33])
    band=(bandedge*bandmask).sum((1,2))/(bandmask.sum((1,2))+1.)
    internal=hump(band,.025,.085)
    longknit+=.62*(internal-hemline)
    button+=.40*(hemline-internal)
    # For a short sleeve or no sleeve, a shirt needs a positive shirt cue,
    # not simply the absence of a sweatshirt cuff. Do not share its shape
    # model with long-sleeve classes unless there is actual sleeve evidence.
    longknit=L*longknit+(1-L)*(sweater+.25)
    jacket=L*jacket+(1-L)*(outer+.35*closure+.45*stiff-.50)
    button=L*button+(1-L)*(bshape-.65+1.0*closure+.91*tailbend+.53*plaid-.70*motif)
    tee=tshape+1.15+.56*short+.67*motif+.25*scoop-.47*closure
    up=np.array([0,2,4,6])[np.stack([tee,longknit,jacket,button],axis=1).argmax(1)]
    # Shape residuals within footwear are noisy (a strap may only cover one
    # sensor pixel); combine their relative ordering with foot anatomy.
    sandshape=np.maximum.reduce([fam[:,7],fam[:,8],fam[:,9],fam[:,19]-.65])
    trainshape=np.maximum(fam[:,12],fam[:,13]-.72)
    bootshape=np.maximum(fam[:,17],fam[:,18]-.18)
    rise=1/(ratio+.0001)
    low=1-hump(rise,.34,.48)
    tall=hump(rise,.44,.69)
    sandal=.45*(sandshape-shoes)+2.5*strap+1.4*air+1.65*opening+.58*under+.37*slide+.54*hump(ratio,3.0,4.1)
    sneaker=.45*(trainshape-shoes)+.45+1.9*trainer+1.10*low+.45*notch-1.94*under-1.9*strap-1.05*air-.75*opening
    boot=.45*(bootshape-shoes)+.23+1.60*shaft+1.83*tall+.70*ankle+.72*under-2.70*low-1.4*strap-.90*notch
    foot=np.array([5,7,9])[np.stack([sandal,sneaker,boot],axis=1).argmax(1)]
    return np.where(coarse==0,up,np.where(coarse==1,1,np.where(coarse==2,3,np.where(coarse==3,foot,8))))

def predict(x):
    n=len(x)
    out=np.empty(n,dtype=np.int64)
    bs=64
    for i in range(0,n,bs):
        rows=min(bs,n-i)
        chunk=np.zeros((bs,28,28),dtype=np.float32)
        chunk[:rows]=x[i:i+rows,0]
        out[i:i+rows]=classify(chunk)[:rows]
    return out
'''

def main():
    dest=Path(sys.argv[1]); dest.mkdir(parents=True, exist_ok=True)
    (dest/'solution.py').write_text(source)
    np.savez(dest/'constants.npz',**generate())

if __name__=='__main__': main()
