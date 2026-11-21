"""Compile independently drawn clothing shapes. No example files are inputs."""
import sys
from pathlib import Path
import numpy as np

# All measurements below describe ideal objects in coordinates relative to their
# own bounding boxes, not in coordinates of any photograph. Rasterise larger
# than the source images to avoid grid-dependent outlines on thin straps.
side = 56
Y,X = np.mgrid[:side,:side] / float(side) + .5 / side
rng = np.random.default_rng(407)
u = rng.uniform

def poly(xs,ys,points):
    inside = np.zeros(xs.shape, bool)
    a,b = points[-1]
    for c,d in points:
        if b != d:
            inside ^= ((ys >= min(b,d)) & (ys < max(b,d)) & (xs < a+(c-a)*(ys-b)/(d-b)))
        a,b=c,d
    return inside

def interp(y,knots,values):
    return np.interp(y,knots,values)

def segment(x,y,x1,y1,x2,y2,r):
    dx,dy=x2-x1,y2-y1
    t=np.clip(((x-x1)*dx+(y-y1)*dy)/(dx*dx+dy*dy+1e-10),0,1)
    return (x-x1-t*dx)**2+(y-y1-t*dy)**2 < r*r

def upper(k):
    # Coordinates of the bodice relative to length from shoulder to hem.
    # Position sleeves independently: dropped, slightly splayed, or alongside
    # the body with no clear background between the cuff and the hem.
    style=rng.random()
    if k==0:
        short=True
        sleeveless=style<.25
        h=u(.86,.995)
        sleeve=u(.22,.43) if style>.15 else u(.12,.26)
        ratio=u(.80,1.30) if not sleeveless else u(.52,.92)
        if style>.96: sleeve=u(.43,.57)
    elif k==3:
        short=style<.77; sleeveless=style<.43
        h=u(.925,1.00)
        sleeve=u(.13,.32) if short else u(.40,.77)
        ratio=u(.43,.81) if style<.90 else u(.76,.99)
    elif k==2:
        short=style<.08; sleeveless=style<.025
        h=u(.81,.99)
        sleeve=u(.24,.48) if short else (u(.55,.84) if style<.24 else u(.86,1.10))
        ratio=u(.74,1.08)
    elif k==4:
        short=style<.03; sleeveless=style<.015
        h=u(.91,1.)
        sleeve=u(.32,.56) if short else (u(.69,.91) if style<.47 else u(.85,1.04))
        ratio=u(.63,.95) if style<.58 else u(.82,1.08)
    else:
        short=style<.26; sleeveless=style<.07
        h=u(.88,1.)
        sleeve=u(.19,.46) if short else (u(.51,.82) if style<.43 else u(.83,1.04))
        ratio=u(.7,1.05) if not short else u(.80,1.20)
    if k==3 and rng.random()<.12:
        # A fitted sheath, including straight strapless dresses.
        ratio=u(.34,.54)
    y=Y*1.13-.035
    cent=.5+u(-.025,.025)*(y-.5)+u(-.012,.012)*np.sin(2*np.pi*y)
    x=abs(X-cent)
    shoulder=u(.254,.382); shoulderY=u(.036,.118)
    chest=u(.25,.357); pit=u(.22,.35)
    if short and not sleeveless and k in (0,6): chest=u(.245,.332)
    waistY=u(.42,.62); waist=chest*u(.83,1.08)
    hip=chest*u(.91,1.14); hem=hip*u(.93,1.04)
    if k==3:
        shoulder=u(.183,.323); chest=u(.193,.308); pit=u(.125,.24)
        waistY=u(.28,.50); waist=chest*u(.66,.98)
        hip=waist*u(1.12,1.62); hem=hip*u(1.04,1.65)
        if ratio<.55 or rng.random()<.16: hem=min(hem,chest*u(1.03,1.34))
        hem=min(.49,hem); hip=min(hip,hem*1.05)
    elif k==2:
        waist=chest*u(.93,1.09); hip=chest*u(.95,1.11); hem=hip*u(.84,.99)
    elif k==4:
        hip=chest*u(1.,1.22); hem=hip*u(.97,1.08)
    neckW=u(.085,.185); neckline=u(.005,.105)
    if k in (4,6) and not sleeveless: neckW=u(.065,.126); neckline=u(.001,.065)
    if sleeveless:
        shoulder=u(.20,.33); chest=u(.29,.40) if k!=3 else chest
        neckline=u(.055,.225); neckW=shoulder*u(.45,.79)
    innerShoulder=min(shoulder*.81,neckW*u(1.,1.19))
    neck=neckline*np.maximum(0.,1.-(x/neckW)**2)**u(.45,1.0)
    top=neck+shoulderY*np.clip((x-innerShoulder)/(shoulder-innerShoulder),0,1)
    # Round or shirt-tail hem and slightly tilted hanging fabric.
    rnd=u(-.014,.031) if k!=6 else u(.0,.080)
    hemline=h-rnd*(x/max(hem,.15))**2+u(-.025,.025)*(X-.5)
    if sleeveless:
        bodyw=interp(y,[0,.09,pit*.75,pit,waistY,.78,h],[shoulder,shoulder*.96,chest*.87,chest,waist,hip,hem])
    else:
        bodyw=interp(y,[0,shoulderY,pit,waistY,.78,max(.80,h-.075),h],[shoulder,shoulder,chest,waist,hip,hip,hem])
    mask=(x<bodyw)&(y>=top)&(y<=hemline)
    if not sleeveless:
        cuffwid=u(.063,.121) if not short else u(.105,.189)
        cuffx=u(.382,.499) if not short else u(.430,.501)
        if k==3: cuffx=u(.32,.464); cuffwid*=.77
        if k==4: cuffwid*=1.14
        elbow=(shoulder+cuffx)*.5+u(.012,.061)
        elbow=min(.49,elbow)
        for hand in [-1,1]:
            xx=(X-cent)*hand
            length=sleeve+u(-.021,.021)
            endx=min(.514,cuffx+u(-.022,.022))
            mid=shoulderY+(length-shoulderY)*u(.38,.62)
            tilt=u(.045,.132) if short else u(-.013,.06)
            first=(shoulder-.016,shoulderY-.006)
            corners=[first,((shoulder+elbow)*.5+.016,(shoulderY+mid)*.5),(elbow,mid),(endx,length),
                     (endx-cuffwid,length+tilt),((chest+endx-cuffwid)*.5,mid+.045),
                     (chest*.94,pit*.92),(shoulder*.84,shoulderY+.024)]
            mask |= poly(xx,y,corners)
    return mask,ratio

def trousers():
    y=Y
    x=abs(X-.5-u(-.045,.045)*(y-.5))
    belt=u(.25,.43); seat=min(.485,belt*u(1.02,1.29)); crotch=u(.22,.425)
    manner=rng.random()
    outer=u(.20,.418) if manner<.77 else u(.36,.49)
    gap=u(.009,.124) if manner<.80 else u(.10,.21)
    if manner<.16: gap=u(-.039,.018)
    knee=(seat+outer)*.5*u(.83,1.08)
    width=interp(y,[0,.10,.21,crotch,.66,1],[belt,seat*.97,seat,seat*.98,knee,outer])
    inner=interp(y,[0,crotch,.60,1],[-.05,-.016,gap*u(.21,.89),gap])
    return (x<width)&(x>inner)&(y>u(-.012,.032)*(1-x/belt))&(y<.995-.014*np.cos(x*11)),u(.34,.61) if manner<.78 else u(.50,.75)

def bag():
    y=Y; x=X+u(-.058,.058)*(Y-.5); kind=rng.random()
    if kind<.16: # clutch or zippered case
        top=u(.016,.20); ratio=u(1.2,2.25)
    elif kind<.30: # small purse on a long strap
        top=u(.49,.72); ratio=u(.54,.94)
    elif kind<.48: # rucksack or soft satchel
        top=u(.05,.27); ratio=u(.62,1.08)
    else:
        top=u(.21,.49); ratio=u(.85,1.55)
    center=.5+u(-.032,.032)*(y-.52)
    hw=.48; t=top; b=.99
    a=u(.74,1.00); foot=u(.72,1.0); mid=u(.28,.66)
    dx=abs(x-center)/hw
    topcurve=u(-.067,.047) if kind>.47 else u(.004,.13)
    w=interp((y-t)/(b-t),[0,.1,mid,.87,1],[a*.94,a,1.,foot,foot*.88])
    if .30<kind<.48:
        w=interp((y-t)/(b-t),[0,.06,.19,.38,.85,1],[.40,.65,.86,.97,1,.86])
    body=(dx<w)&(y>=t+topcurve*dx**2)&(y<b-u(.012,.048)*dx**4)
    if kind>.16 or rng.random()<.5:
        rad=u(.155,.337) if kind>.30 else u(.32,.47)
        wide=u(.021,.063) if kind>.30 else u(.017,.040)
        cx=u(.43,.57); cy=t+u(.006,.12); height=cy-u(.005,.048)
        # Closed arch with straight sides down to the bag mouth.
        arch=((x-cx)/rad)**2+((y-cy)/height)**2
        hole=((x-cx)/max(.06,rad-wide))**2+((y-cy)/max(.04,height-wide))**2
        body |= (arch<=1)&(hole>=1)&(y<=cy+.045)
    return body,ratio

def stripe(x,y,points,width):
    res=np.zeros(x.shape,bool)
    for (xa,ya),(xb,yb) in zip(points,points[1:]):
        res |= segment(x,y,xa,ya,xb,yb,width)
    return res

def footwear(k):
    # Side elevation, toe to the left. Height here is measured downwards
    # from a separate reference plane above the collar.
    x=X; y=Y; v=rng.random(); rise=u(.021,.10)
    bottom=interp(x,[0,.07,.23,.48,.68,.85,1],[.90,.965,.993,.988,.967,.961,.951])
    bottom-=rise*np.maximum(0,1-x/.13)**2
    if k==7:
        ratio=u(1.82,3.17) if v>.115 else u(1.45,2.08)
        tongue=u(.475,.69); notch=u(.675,.815); ht=u(.16,.405)
        if v<.115: ht=u(.085,.27)
        topline=interp(x,[0,.045,.14,.31,tongue-.095,tongue,tongue+.047,notch,.88,.96,1],
            [.77,.70,u(.55,.68),u(.47,.60),u(.15,.28),.045,.11,ht,u(.04,.165),u(.105,.22),.47])
        rear=1-.05*np.maximum(0,.65-y)/.65
        shoe=(y>topline)&(y<bottom)&(x<rear)&(x>.02*np.maximum(0,.83-y)/.3)
    elif k==9:
        ratio=u(.955,1.74) if v<.78 else u(1.50,2.03)
        front=u(.40,.658); back=u(.84,.986); toeY=u(.62,.79)
        topline=interp(x,[0,.056,.23,front-.12,front-.045,front,front+.045,back-.05,back,1],
            [.868,toeY+.025,toeY,u(.48,.66),u(.27,.46),u(.007,.125),.025,u(.01,.134),.12,.91])
        topline=np.maximum(topline,np.where(x>back,.64+.18*np.maximum(0,x-back)/max(.06,1-back),0))
        # A distinct rising arch on heeled boots. Block heels are also common.
        heel=u(.01,.16) if v>.31 else u(.16,.345)
        arch=interp(x,[0,.36,.58,.747,.82,1],[0,0,heel*.47,heel,0,0]); bottom=bottom-arch
        shoe=(y>topline)&(y<bottom)
    else:
        ratio=u(1.52,3.24) if v<.73 else u(1.06,1.94)
        # Flat sole, wedge, or raised heel under the ankle.
        heel=u(.01,.11) if v<.55 else u(.18,.52); strap=u(.036,.091)
        bottom=bottom-interp(x,[0,.32,.58,.77,.82,1],[0,0,heel*.51,heel,0,0])
        floor=interp(x,[0,.13,.33,.57,.79,.95,1],[.833,.805,.82,.76-heel*.46,.74-heel,.74-heel,.83-heel])
        shoe=(y<bottom)&(y>floor)
        # Straps rise from the footbed, leaving dark space above the sole.
        toe=u(.045,.18); vamp=u(.22,.40); join=u(.47,.66); ankle=u(.69,.88)
        shoe |= stripe(x,y,[(toe,.826),(toe+.035,.64),(vamp,u(.47,.59)),(vamp+.12,.80)],strap)
        if v<.18: # slide, with one wide band on the front of the foot
            shoe |= poly(x,y,[(toe,.80),(toe+.05,.50),(vamp,.38),(vamp+.14,.81)])
            ratio=u(2.05,3.62)
        else:
            shoe |= stripe(x,y,[(join-.11,.78),(join,u(.48,.66)),(ankle-.065,u(.20,.37)),(ankle,.10)],strap)
            shoe |= stripe(x,y,[(ankle-.087,.21),(ankle,.063),(.929,.093),(.951,.20),(.932,.70-heel*.48)],strap*u(.71,1.38))
            if v>.58:
                shoe |= poly(x,y,[(ankle+.015,.15),(.96,.09),(.991,.71-heel*.59),(ankle-.024,.75-heel)])
            if rng.random()<.48:
                shoe |= stripe(x,y,[(vamp,.53),(join+.02,.43),(.875,.71-heel*.71)],strap*.85)
    return shoe,ratio

# Code shared by the template compiler and predictor: identical canonical
# coordinate system, independent of each object's position in the frame.
GEOMETRY = '''
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
'''
# Expose the shared pure functions in the builder without reading any file.
exec(GEOMETRY)


def make_bank():
    banks=[]; attrs=[]
    for k in range(10):
        bank=[]; ars=[]
        # Equal numbers of independent designs per class; no measurement
        # across different designs is used to construct a template.
        for j in range(420):
            if k==1: mask,aspect=trousers()
            elif k==8: mask,aspect=bag()
            elif k in (5,7,9): mask,aspect=footwear(k)
            else: mask,aspect=upper(k)
            yy,xx=np.nonzero(mask)
            a=mask.astype(float)
            a=sample_box(a,yy.min(),yy.max(),xx.min(),xx.max(),48)
            a=pool(a,24)
            bank.append(features(a)); ars.append(np.log(aspect))
        banks.append(bank); attrs.append(ars)
    return dict(shapes=np.array(banks,dtype=np.float64), aspects=np.array(attrs,dtype=np.float64))

PREDICTOR = '''
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
'''

if __name__ == '__main__':
    dest=Path(sys.argv[1]); dest.mkdir(parents=True,exist_ok=True)
    (dest/'solution.py').write_text('import numpy as np\n'+GEOMETRY+PREDICTOR,encoding='utf-8')
    np.savez(dest/'constants.npz',**make_bank())
