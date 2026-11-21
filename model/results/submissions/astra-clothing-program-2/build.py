"""Build a dictionary of drawings from a parametric paper-doll pattern book.
All dimensions for garments are fractions of their length; for shoes they
are fractions of their sole length. No photos are used by the builder.
"""
import sys
from pathlib import Path
import numpy as np

rng = np.random.default_rng(9451)
u = rng.uniform

def choice(*options):
    return options[rng.integers(len(options))]

def profile(src):
    # Complete a symmetric polygon specified from the neck to the hem.
    return src + [(-x,y) for x,y in src[::-1]]

def mirror(p):
    return [(-x,y) for x,y in p]

def garment(k):
    # body height, width at bust, neck opening, shoulder slope, armscye,
    # waist shaping, hip ease, sleeve construction, hem drop, collar stand.
    style = u()
    if k == 0:
        chest = u(.29,.44); sh = chest*u(.77,1.02)
        sl = choice('short','short','short','short','vest','vest','cap','mid')
        if sl=='mid' and u()<.65: sl='short'
        if sl=='vest': chest *= u(.72,1.00); sh = chest*u(.57,.99)
        if sl in ('short','cap'): chest*=u(.83,1.0); sh=chest*u(.83,1.04)
        neck=sh*u(.38,.70); rise=u(.035,.12); dip=u(.002,.075)
        waist=chest*u(.80,1.03); hip=chest*u(.86,1.20); cuffhem=u(-.018,.04)
    elif k == 2:
        chest=u(.24,.38); sh=chest*u(.83,1.12); sl='long' if style<.90 else 'mid'
        neck=sh*u(.31,.60); rise=u(.055,.14); dip=u(.0,.047)
        waist=chest*u(.91,1.05); hip=chest*u(.84,1.06); cuffhem=u(-.012,.026)
    elif k==4:
        length=choice('hip','hip','hip','waist','thigh','thigh','knee')
        chest={'hip':u(.26,.37),'waist':u(.33,.46),'thigh':u(.23,.31),'knee':u(.20,.26)}[length]
        sh=chest*u(.78,1.04); sl='jacket'; neck=sh*u(.30,.58)
        rise=u(.045,.12); dip=u(0,.028); waist=chest*u(.86,1.04); hip=chest*u(.94,1.25)
        cuffhem=u(-.015,.036)
    elif k==6:
        chest=u(.23,.39); sh=chest*u(.70,1.08); sl=choice('long','long','mid','threequarter','cap','short','long','vest')
        if sl=='vest': sh*=u(.7,1.)
        neck=sh*u(.30,.66); rise=u(.04,.14); dip=u(0,.085)
        waist=chest*u(.76,1.10); hip=chest*u(.90,1.29); cuffhem=choice(u(.035,.125),u(.008,.050))
    else:
        chest=choice(u(.125,.22),u(.17,.255)); sh=chest*u(.75,1.04)
        sl=choice('vest','vest','vest','cap','short','short','dressmid','dresslong','dresslong')
        if sl=='vest': sh*=u(.62,1.)
        neck=sh*u(.35,.73); rise=u(.027,.08); dip=u(.0,.09); waist=chest*u(.66,1.02)
        hip=chest*choice(u(.95,1.36),u(1.32,2.08)); cuffhem=u(.00,.036)
    stand = (u(.025,.100) if k==4 else u(.008,.052) if k==6 and style<.65 else u(.00,.018))
    if k==2 and style<.10: stand=u(.03,.09)
    rise += stand
    pit=rise + chest*u(.52,.84); waisty=u(.39,.68) if k!=3 else u(.30,.49)
    if sl=='vest':
        dip=u(.012,.14) if k!=3 else u(.012,.10)
        neck=sh*u(.53,.91); rise = u(.00,.075)
    bod=profile([(0,dip),(neck*.76,dip*.48),(neck,0),(neck*1.10,stand),(sh,rise),
                 (sh*.45+chest*.55,rise+(pit-rise)*.55),(chest,pit),
                 (waist,waisty),(hip,1-cuffhem-.035),(hip*.994,1-cuffhem),
                 (hip*.57,1-.19*cuffhem),(0,1)])
    pieces=[bod]; # sleeve is a separate piece: overlaps are cloth, not holes.
    if sl!='vest':
        thick=chest*u(.30,.54)
        if sl=='long': end=u(.89,1.09); drift=u(.0,.16); tip=thick*u(.65,1.10); tilt=u(-.045,.045)
        elif sl=='jacket':
            end={'hip':u(.79,1.04),'waist':u(.93,1.13),'thigh':u(.68,.90),'knee':u(.52,.78)}[length]
            drift=u(-.015,.12); tip=thick*u(.75,1.13); tilt=u(-.03,.03)
        elif sl=='dresslong': end=u(.49,.77); drift=u(.00,.095); tip=thick*u(.69,1.08); tilt=u(-.025,.025)
        elif sl=='dressmid': end=u(.28,.48); drift=u(.03,.11); tip=thick*u(.85,1.35); tilt=u(.00,.06)
        elif sl=='mid' or sl=='threequarter': end=u(.45,.78); drift=u(.015,.16); tip=thick*u(.78,1.32); tilt=u(-.016,.06)
        else:
            end=pit+u(-.045,.1) if sl=='short' else rise+(pit-rise)*u(.50,.99)
            drift=chest*u(.10,.47); tip=thick*u(.70,1.22); tilt=u(.025,.105)*chest/.34
        # A coat, jumper or shirt can be photographed with its arms tucked
        # against its sides or hanging slightly apart; sleeve fullness at
        # the biceps must be allowed independently of the cuff position.
        if sl in ('long','jacket','threequarter','mid','dresslong','dressmid'):
            drift=u(-.045,.105); tip*=u(.77,1.10)
        cx=(sh+chest)*.5+drift + tip*.40
        cy=end; outer=cx+.5*tip; inner=cx-.5*tip
        if sl in ('long','jacket','threequarter','mid','dresslong','dressmid'):
            inner += u(-.03,.03); bow=u(.00,.05); elbow=u(.39,.69)
            arm=[(sh*.93,rise-.01),(sh,rise),(sh+.35*thick,rise+.066),
                 (sh*(1-elbow)+outer*elbow+bow,rise*(1-elbow)+end*elbow),
                 (outer,end-tilt),(inner,end),(chest*.52+inner*.48+(bow*.4),pit*.52+end*.48),
                 (chest*.96,pit),(sh*.89,rise+.05)]
        else:
            arm=[(sh*.93,rise-.008),(sh,rise),(sh*.55+outer*.45+.012,rise*.55+(end-tilt)*.45),
                 (outer,end-tilt),(inner,end),(chest*.96,pit),(sh*.87,rise+.025)]
        pieces.extend([arm,mirror(arm)])
    return pieces

def trousers():
    half=u(.155,.285); waist=half*u(.72,1.00); fork=u(.23,.43)
    leg=half*u(.31,.84); shift=half*u(-.22,.66); bend=half*u(-.20,.25)
    p=profile([(0,u(0,.028)),(waist,0),(half,.17),(half*u(.93,1.01),fork),
        (half+shift+bend,.64),(leg*.5+half*.6+shift,1),(half*.6+shift-leg*.5,1),
        (max(.002,half*.55+shift+bend-leg*.55),.65),(.004,fork),(0,fork)])
    return [p]

def interp(points,x):
    return np.interp(x,[a for a,b in points],[b for a,b in points])

def shoe(k):
    heel = u(.78,.95); toe=u(.08,.20)
    if k==7:
        h=choice(u(.28,.47),u(.30,.52),u(.32,.59)); tongue=u(.53,.73); back=h*u(.60,.96); lip=h*u(.12,.32)
        top=[(0,h*u(.09,.23)),(.06,h*u(.22,.40)),(toe+.10,h*u(.29,.55)),(tongue-.16,h*u(.58,.84)),
            (tongue-.035,h*.96),(tongue,h),(tongue+.045,h*.84),((tongue+heel)*.5,back-lip),
            (heel,back),(.984,back*u(.8,.98)),(1,h*.24)]
        bot=[(.99,.025),(.87,0),(.52,u(0,.023)),(.21,u(.0,.024)),(.052,u(.018,.067)),(0,h*.09)]
        return [top+bot]
    if k==9:
        h=choice(u(.39,.61),u(.46,.81),u(.59,1.03),u(.79,1.29)); start=u(.40,.66); tilt=u(-.09,.13); back=u(.91,1.0); toelev=u(.13,.24)
        lift=choice(u(.01,.10),u(.05,.23),u(.13,.34))
        top=[(0,.055),(.032,toelev*.75),(.12,toelev),(.28,toelev*1.12+lift*.27),(start-.055,.24+lift*.70),
             (start,.31+lift*.68),(start+tilt,h*u(.88,1)),(start+tilt+.055,h),(back,h*u(.85,1.06)),
             (back-.015,h*.69),(back-.029,h*.51),(1,max(.21,lift+.1)),(.982,.015)]
        bot=[(u(.76,.90),0),(u(.75,.86),lift),(.60,lift*.70),(.35,.02),(.15,0),(.037,.013)]
        return [top+bot]
    lift=choice(u(0,.055),u(.06,.24),u(.26,.56)); thickness=choice(u(.026,.067),u(.05,.14),u(.12,.21))
    yb=[(0,.034),(.09,.005),(.33,0),(.50,lift*.24),(.78,lift),(.94,lift*1.02),(1,lift+.02)]
    base=[(x,y+thickness) for x,y in yb]+yb[::-1]
    pieces=[base];
    if lift>.065:
        w=choice(u(.06,.16),u(.16,.32),u(.35,.65))
        pieces.append([(.95-w,lift*.71+thickness*.44),(.98,lift+thickness*.4),(.97,0),(.97-.63*w,0)])
    arch=u(.09,.22); ankle=u(.12,.34); height=lift+ankle+thickness
    def band(x0,x1,s0,s1,t):
        xm=x0*.35+x1*.65
        return [(x0,s0),(xm,s0*.25+s1*.75+.022),(x1,s1),(x1+t,s1-.007),(xm+t,s0*.25+s1*.75-.005),(x0+t,s0-.016)]
    pieces.append(band(u(.015,.13),u(.12,.33),thickness*.72,thickness+arch,u(.06,.22)))
    if u()<.5:
        pieces.append(band(u(.10,.30),u(.12,.40),thickness*.8,thickness+arch,u(.07,.24)))
    if u()<.76:
        pieces.append(band(u(.18,.37),u(.28,.57),thickness+lift*.04,thickness+arch+lift*.22,u(.04,.19)))
    if u()<.89:
        x0=u(.58,.84); x1=u(.63,.88); t=u(.036,.14)
        pieces.append(band(x0,x1,interp(yb,x0)+thickness*.75,height,t))
        pieces.append(band(.94,x1+.006,lift+thickness*.65,height,t*.64))
        if u()<.73:pieces.append(band(u(.29,.55),x1,thickness+lift*.14,height,u(.04,.13)))
    else:
        pieces.append(band(.40,.28,thickness,thickness+arch,u(.06,.19)))
    return pieces

def bag():
    h=choice(u(.47,.89),u(.65,1.02),u(.80,1.26)); top=u(.34,.55); bottom=u(.34,.53)
    drop=u(-.04,.11); side=u(.03,.13); corner=u(.035,.16)
    body=profile([(0,h-drop),(top*.77,h-.15*drop),(top,h),(max(bottom,top)+side*.3,h*.52),
                  (bottom,corner*h),(bottom-corner*.28,.025*h),(bottom-corner,0),(0,0)])
    pieces=[body];
    if u()<.81:
        rise=choice(u(.05,.21),u(.19,.55),u(.48,1.01)); width=u(.11,.41); thick=u(.018,.060)
        angle=np.linspace(0,np.pi,17); xx=width*np.cos(angle); yy=h-drop-.038+rise*np.sin(angle)**u(.49,1.50)
        handle=list(zip(xx,yy))+list(zip((xx*(1-thick/width))[::-1],(yy-thick)[::-1])); pieces.append(handle)
    return pieces

# Sample coverage at subpixel resolution then pool. Normalise each drawing's
# true bounding box, retaining its unnormalised aspect as a separate signal.
R=48
axis=(np.arange(R)+.5)/R
def draw(pieces,flipy,skew):
    arr=[np.array(p,dtype=float) for p in pieces]; comb=np.concatenate(arr)
    if flipy:
        for p in arr:p[:,1]*=-1
    for p in arr:p[:,0]+=skew*p[:,1]
    comb=np.concatenate(arr); lo=comb.min(0); wh=np.maximum(comb.max(0)-lo,.001)
    mask=np.zeros((R,R),bool)
    for p in arr:
        p=(p-lo)/wh; x,y=p.T; bx=np.roll(x,1); by=np.roll(y,1)
        use=(axis[:,None]>=np.minimum(y,by)) & (axis[:,None]<np.maximum(y,by)); dy=by-y
        inter=x+(axis[:,None]-y)*(bx-x)/np.where(dy==0,1,dy)
        # A scanline alternates outside/inside at successive edge crossings.
        crossings=np.clip(np.ceil(np.where(use,inter,2)*R-.5),0,R).astype(int)
        accum=np.zeros((R,R+1),np.int32); np.add.at(accum,(np.arange(R)[:,None],crossings),1)
        mask|=(np.cumsum(accum[:,:R],axis=1)%2==1)
    return mask.reshape(24,2,24,2).mean((1,3)),wh[0]/wh[1]

feature_code='''
def descriptor(mask, aspect):
    n=len(mask)
    patch=mask.reshape(n,12,2,12,2).mean(axis=(2,4))
    # Integrated cumulative occupancy gives subpixel estimates of the four
    # contours and suppresses isolated weak foreground responses.
    edge=np.clip((mask-.20)*1.7,0,1)
    contours=[]
    for a in (edge,edge[:,:,::-1],edge.transpose(0,2,1),edge.transpose(0,2,1)[:,:,::-1]):
        penetration=(1-np.clip(a.cumsum(axis=2),0,1)).mean(axis=2)
        contours.append(penetration.reshape(n,12,2).mean(axis=2))
    return np.rint(np.concatenate([patch.reshape(n,144)*25,contours[0]*42,contours[1]*42,contours[2]*47,contours[3]*40,np.clip(np.log(aspect),-3,3)[:,None]*185],axis=1)).astype(np.float32)
'''
# Use the exact same analytic representation for drawings and observations.
ns={'np':np}; exec(feature_code,ns); descriptor=ns['descriptor']

def book():
    blocks=[]
    for k in range(10):
        masks=[]; aspects=[]
        for i in range(3200):
            pieces=garment(k) if k in (0,2,3,4,6) else trousers() if k==1 else bag() if k==8 else shoe(k)
            m,a=draw(pieces,k in (5,7,8,9),u(-.042,.042) if k not in (5,7,9) else u(-.075,.075))
            masks.append(m); aspects.append(a)
        blocks.append(descriptor(np.array(masks),np.array(aspects)))
    return np.stack(blocks).astype(np.int16)

code='''import numpy as np
'''+feature_code+'''
book=CONSTANTS['drawings'].astype(np.float32)
lengths=(book*book).sum(axis=2)

def box(im):
    n=len(im); seq=np.arange(n)
    peak=im.max(axis=(1,2))[:,None,None]
    # A dark item still has an outline. Distinguish a weak but connected
    # contour from a stray antialiasing pixel in the margin.
    occ=np.clip((im-.014*peak-.004)/(.090*peak+.010),0,1)
    def extent(d):
        s=occ.sum(axis=d); alive=s>np.minimum(1.05,np.maximum(.46,s.max(axis=1,keepdims=True)*.049))
        l=alive.argmax(axis=1); r=27-alive[:,::-1].argmax(axis=1)
        return l,r
    left,right=extent(1); top,bot=extent(2)
    def sample(a,size):
        def coords(l,r):
            p=l[:,None]+((np.arange(size)+.5)/size)*(r-l+1)[:,None]-.5
            p=np.clip(p,0,27); low=p.astype(int); return low,np.minimum(low+1,27),p-low
        xl,xr,fx=coords(left,right); yt,yb,fy=coords(top,bot)
        def take(y,x): return a[seq[:,None,None],y[:,:,None],x[:,None,:]]
        fx=fx[:,None,:]; fy=fy[:,:,None]
        return (take(yt,xl)*(1-fx)+take(yt,xr)*fx)*(1-fy)+(take(yb,xl)*(1-fx)+take(yb,xr)*fx)*fy
    soft=sample(occ,24); pix=sample(im,32)
    return soft,pix,(right-left+1)/(bot-top+1)

def shapedist(f,k):
    # The quantised products and sums fit exactly in float32 integers.
    d=(f*f).sum(axis=1)[:,None]+lengths[k]-2*(f@book[k].T)
    # A small neighbourhood, rather than a single perfect drawing, should
    # account for the observation. This is a fixed geometric matching rule.
    best=np.partition(d,15,axis=1)[:,:16]
    return (.65*best.mean(axis=1)+.35*best.min(axis=1))/1024

def cap(a,l=0,h=1): return np.clip(a,l,h)

def apparel(im,mask):
    # Work in object coordinates. Measure coherent seams and texture in
    # the centre panel, where garment construction differs even when the
    # outline of a sweater, jacket and button-down is almost identical.
    p=im/(np.quantile(im,.80,axis=(1,2))[:,None,None]+.055)
    n=len(p); s=np.zeros((n,10))
    # Persistent central vertical edge or ridge: allow a gently inclined
    # placket and weak contrast (a zipper need not be black or white).
    d1=p[:,:,2:]-p[:,:,:-2]
    slow=(p[:,:,4:]-p[:,:,:-4])*.65
    line=np.abs(d1[:,:,1:-1]); longline=np.zeros(n); repeat=np.zeros(n)
    for tilt in (-2,0,2):
        slices=[]; signed=[]; buttons=[]
        for i,(lo,hi) in enumerate(((8,14),(14,20),(20,26),(26,30))):
            dx=int(round((i-1.5)*tilt/3))
            sl=slice(9+dx,19+dx)
            l=line[:,lo:hi,sl]; edge=d1[:,lo:hi,1:-1][:,:,sl]; wide=slow[:,lo:hi,sl]
            slices.append(l.mean(axis=1)); signed.append(.70*edge.mean(axis=1)+.30*wide.mean(axis=1))
            buttons.append(np.max(l,axis=1))
        mag=np.stack(slices,axis=1); signs=np.stack(signed,axis=1); b=np.stack(buttons,axis=1)
        v=.5*np.abs(signs.mean(axis=1))+.5*np.sort(mag,axis=1)[:,:2].mean(axis=1)
        # centre has stronger prior than a fold close to the side seam
        prior=1-np.abs(np.arange(10)-4.5)*.09
        longline=np.maximum(longline,(v*prior).max(axis=1))
        repeat=np.maximum(repeat,(np.sort(b,axis=1)[:,1:3].mean(axis=1)*prior).max(axis=1))
    flank=np.concatenate([line[:,9:29,5:10],line[:,9:29,18:23]],axis=2).mean(axis=(1,2))
    zipper=cap((longline-.014-.48*flank)/(.055+.53*flank),0,2.0)
    buttons=cap((repeat-.045-1.35*flank)/(.13+.55*flank),0,1.5)
    # Woven checks and all-over prints, unlike chest logos, continue across
    # the lower torso. Horizontal stripes alone are also common on knits.
    dx=np.abs(p[:,:,2:]-p[:,:,:-2]); dy=np.abs(p[:,2:]-p[:,:-2]); textures=[]
    for a,b in ((6,13),(13,20),(20,27)):
        textures.append(np.minimum(dx[:,a:b,7:23].mean((1,2)),dy[:,a:b,8:24].mean((1,2))))
    printed=cap((np.stack(textures,1).min(1)-.055)/.13,0,1.6)
    # A narrow peak in the top contour indicates a standing or folded
    # collar rather than the broad low arc of a crew neck.
    top=(1-cap(((mask-.18)*1.6).clip(0,1).cumsum(axis=1))).mean(axis=1)
    # Measure the neckline prominence at the crown instead of across
    # the thin outer tips of the shoulders. A lapel or hood projects high
    # above the shoulder; the centre of a scoop neck dips down.
    center=top[:,10:14].mean(1)
    shoulder=np.minimum(top[:,4:8].mean(1),top[:,16:20].mean(1))
    prominence=shoulder-center
    collar=cap((prominence-.026)/.105,0,1.8)
    scoop=cap((center-np.minimum(top[:,7:10].mean(1),top[:,14:17].mean(1))+.005)/.075)
    # Distinguish a contrasting inside collar (or lining) from a flat
    # crew neck with a symbol on the chest. Compare tapering central V
    # with the two neighbouring front panels at the same heights.
    vee=np.zeros(n); bib=np.zeros(n)
    for wid,height in ((5,7),(6,11),(4,9),(7,15)):
        inside=[]; outside=[]
        for y in range(2,height):
            r=max(1,int(wid*(1-y/height))); inside.append(p[:,y,16-r:16+r].mean(1)); outside.append((p[:,y,13-r:16-r].mean(1)+p[:,y,16+r:19+r].mean(1))*.5)
        a=np.stack(inside,1); b=np.stack(outside,1)
        vee=np.maximum(vee,(a-b).mean(1)); bib=np.maximum(bib,(b-a).mean(1))
    lining=cap((vee-.08)/.35); vneck=cap((bib-.12)/.38)
    # Localised chest motifs indicate a tee or sweatshirt; an all-over
    # print or fluted fabric is more typical of a blouse. Give the centre
    # closure a score relative to the other vertical folds, not the total
    # contrast in the image.
    upper=(dx[:,6:17,9:21].mean((1,2))+dy[:,6:17,10:22].mean((1,2)))*.5
    lower=(dx[:,19:27,9:21].mean((1,2))+dy[:,19:27,10:22].mean((1,2)))*.5
    logo=cap((upper-1.7*lower-.04)/.18)
    vertical=dx[:,12:28,6:24].mean((1,2)); horizontal=dy[:,12:28,7:25].mean((1,2))
    drape=cap((vertical-horizontal-.012)/(.038+horizontal*.6))
    stripe=cap((horizontal-vertical-.035)/(.09+vertical*.6))
    # Describe the yoke sector in a local system attached to the garment's
    # own shoulder tips. Its width near the top and collar pedestal are
    # much more informative than a single global silhouette distance.
    shoulderline=np.minimum(top[:,3:6].mean(1),top[:,18:21].mean(1))
    pedestal=cap((shoulderline-center-.085)/.135,0,1.5)
    crown=(mask[:,0:3,6:18].mean((1,2))-.5)*2
    # Rank centre-front line saliency against all off-centre columns.
    # Seek paired gutters (lapels on an open coat) as well as one zip.
    coherent=np.abs(d1[:,9:29,:].mean(axis=1))
    midline=coherent[:,12:18].max(1); shadow=np.concatenate([coherent[:,6:11],coherent[:,19:24]],1).mean(1)
    midzip=cap((midline-.9*shadow-.009)/(.045+.9*shadow),0,1.6)
    sides=cap((np.minimum(coherent[:,6:13].max(1),coherent[:,17:24].max(1))-.035)/.090)
    centrecloth=p[:,13:26,13:19].mean((1,2)); sidecloth=.5*(p[:,13:26,7:12].mean((1,2))+p[:,13:26,20:25].mean((1,2)))
    opening=sides*cap((np.abs(centrecloth-sidecloth)-.08)/.22)
    # Pointed collars / collars standing above the shoulder seam. Compare
    # the central crown to the neck-shoulder junction, *not* to the full
    # garment width (that also measures the sleeve slope). Use fractional
    # foreground coverage in the first few rows to handle quantised edges.
    neckedges=(top[:,7:10].mean(1)+top[:,14:17].mean(1))*.5
    stand=cap((neckedges-center+.002)/.080,0,1.9)
    high=mask[:,0:2,10:14].mean((1,2)); adj=(mask[:,0:2,6:10].mean((1,2))+mask[:,0:2,14:18].mean((1,2)))*.5
    raised=cap((high-adj-.035)/.4,0,1.6)
    tiltwall=cap((top[:,4:7].mean(1)+top[:,17:20].mean(1)-2*neckedges-.055)/.18)
    # A vertical placket is usually sharper than the low frequency folds
    # on either side, including on tone-on-tone jackets. Finite differences
    # of order two remove the shading ramp. Accumulate evidence in strips
    # to allow the seam to jog slightly left and right along its length.
    hp=p[:,:,1:-1]-(p[:,:,:-2]+p[:,:,2:])*.5
    lam=[]; base=[]
    for y0,y1 in ((7,12),(12,18),(18,24),(24,29)):
        v=np.abs(hp[:,y0:y1].mean(1))*.60+np.abs(hp[:,y0:y1]).mean(1)*.40
        lam.append(np.maximum(np.maximum(v[:,10:19],v[:,11:20]),v[:,12:21]))
        base.append(.5*(v[:,5:11].mean(1)+v[:,19:25].mean(1)))
    path=np.stack(lam,1); path=np.sort(path,axis=1)[:,0:3].mean(1)
    zipfine=(path*(1-.075*np.abs(np.arange(9)-4))).max(1)
    noise=np.stack(base,1).mean(1)
    fine=cap((zipfine-noise*1.25-.004)/(.020+noise*.78),0,1.8)
    s[:,4]+=1.35*zipper+.62*buttons+2.6*collar+1.8*pedestal+midzip+.8*opening+.90*lining+.7*vneck-1.5*logo-.52*drape-1.55
    s[:,4]+=1.6*stand+1.15*raised+.35*tiltwall+1.25*fine-.65
    s[:,2]+=-.45*stand-.5*raised-.45*fine+.35
    s[:,6]+=.5*stand+.45*fine
    # Differentiating a sunken neck from a lapel: look for a V-shaped
    # pair of edges meeting the placket. On a button-down, the light
    # triangular back of the collar ends high above the hem; an overcoat
    # often exposes a longer and broader swath of the layer underneath.
    shortv=np.zeros(n); lapel=np.zeros(n)
    for h in (6,9,16,21):
        for span in (4,7):
            samples=[]
            for y in range(2,h):
                r=max(1,int(span*(h-y)/(h-1))); delta=.5*(p[:,y,15-r]-p[:,y,17-r]+p[:,y,16+r]-p[:,y,14+r]); samples.append(delta)
            response=np.abs(np.stack(samples,1).mean(1))
            if h<10: shortv=np.maximum(shortv,response)
            else: lapel=np.maximum(lapel,response)
    tab=cap((shortv-1.4*lapel-.028)/.13)
    facing=cap((lapel-.06)/.17)
    s[:,6]+=1.45*tab+.8*printed+.45*zipper+.60*drape+.5*lining
    s[:,4]+=.80*facing-.60*tab-.52*printed
    # A deep pocket or button outline has lower-frequency contrast than
    # the stripes of a ribbed waistband. A crew neck has a wide opening
    # but no continuation from it down the centre panel.
    s[:,2]+=.55*scoop*cap(1-midzip)+.50*logo
    s[:,2]+=-.82*zipper-.34*buttons-.8*collar-.6*pedestal-.5*midzip+1.9*logo+.8*stripe-.7*drape-.8*printed+.58*scoop+.60*crown+.6
    s[:,6]+=.90*zipper+.65*buttons+3.1*printed+.7*lining+1.45*drape-1.55*logo+.4*collar+.5*midzip+1.05*opening-.52
    # Texture due to a graphic in the middle is not the same as a plaid
    # fabric: in a woven shirt the pattern continues into the shoulders
    # and across much of the lower half. A long smooth vertical fold alone
    # is not sufficient evidence against a jersey T-shirt.
    pattern=cap((np.minimum(np.minimum(dx[:,10:26,7:12].mean((1,2)),dx[:,10:26,18:23].mean((1,2))),dy[:,15:28,8:24].mean((1,2)))-.055)/.15)
    s[:,0]+=-.38*(zipper-.6)-.36*printed+2.05*logo+.75*scoop-.45*pattern-.15*lining
    s[:,6]+=.75*pattern
    # For garments with a knitted waistband, a horizontal line often runs
    # the breadth of the front close to the hem.
    hor=p[:,23:31,8:24]-p[:,21:29,8:24]
    welt=np.abs(hor.mean(axis=2)).max(axis=1)
    s[:,2]+=.7*cap((welt-.11)/.26)
    return s

def classify(im):
    soft,pix,aspect=box(im); f=descriptor(soft,aspect); rev=descriptor(soft[:,:,::-1],aspect)
    d=np.stack([shapedist(f,k) for k in range(10)],axis=1)
    for k in (5,7,9): d[:,k]=np.minimum(d[:,k],shapedist(rev,k))
    # Use a parser for the joints and proportions. Each block of classes
    # gets one silhouette score: details distinguish styles *within* the
    # block, not whether it is a long-sleeved item in the first place.
    side=cap((soft-.23)*2.1); left=(1-cap(side.cumsum(2))).sum(2)
    right=(1-cap(side[:,:,::-1].cumsum(2))).sum(2)
    widths=1-(left+right)/24
    hem=widths[:,18:22].mean(1); waist=widths[:,11:16].mean(1)
    shoulder=widths[:,3:8].mean(1); cuff=widths[:,7:12].mean(1)
    gaps=cap(widths-soft.mean(2)); gap=gaps[:,12:21].mean(1)
    drop=np.maximum(shoulder,cuff)-np.minimum(waist,hem)
    extension=cap((drop-.065)/.21); narrow=cap((.60-aspect)/.19)
    flare=cap((hem-waist-.025)/.14); pulledin=cap((shoulder-hem-.10)/.19)
    longarms=cap((hem+.25*waist-.26*shoulder-.68)/.28)+cap((gap-.032)/.13)
    tee=extension*.95+pulledin*.35-narrow*.40
    skirt=narrow+.70*flare-.75*cap((gap-.035)/.14)
    # Look for the two sleeve ends at the same height and inspect the
    # torso below them. A tank top has neither sleeves nor much skirt,
    # and has an open, often scooped neck between the shoulder straps.
    endstep=(widths[:,4:14]+widths[:,3:13]-widths[:,6:16]-widths[:,7:17])*.5
    sleeves=cap((endstep.max(1)-.053)/.15)*cap((.19-gap)/.15)
    topedge=(1-cap(side.cumsum(1))).sum(1)/24
    scoop=cap((topedge[:,9:15].mean(1)-np.minimum(topedge[:,4:8].mean(1),topedge[:,16:20].mean(1))-.012)/.105)
    topwidth=widths[:,2:5].mean(1)
    straightsides=cap((.17-np.abs(hem-waist))/.15); shortbody=cap((aspect*waist-.40)/.22)
    tank=(.30+.7*scoop)*cap((.92-topwidth)/.21)*straightsides*shortbody*cap((.14-gap)/.13)
    # On a dress with loose sleeves the overall photo may be quite wide,
    # but the waist of the dress still occupies little of its length.
    smallwaist=cap((.47-aspect*np.minimum(waist,soft[:,11:16].mean((1,2))+.04))/.19)
    # Estimate a waist/hip on the contiguous central cloth only, ignoring
    # space and cloth in dangling sleeves to the left and right of it.
    body=cap((soft-.12)*2.9)
    torso=(np.minimum.accumulate(body[:,:,12:],axis=2).sum(2)+np.minimum.accumulate(body[:,:,:12][:,:,::-1],axis=2).sum(2))/24
    midriff=np.minimum(torso[:,9:12].mean(1),torso[:,12:15].mean(1)); hip=torso[:,17:21].mean(1)
    # If a deep print actually breaks the central region, the raw extent
    # is safer than the connected width.
    midriff=np.maximum(midriff,waist*.47); hip=np.maximum(hip,hem*.47)
    longbody=cap((.48-aspect*midriff)/.17); sweep=cap((hip-midriff-.025)/.21)
    broad=cap((aspect*midriff-.46)/.19)
    split=cap((gap-.025)/.10); aLine=sweep*longbody*cap(1.18-.8*split)
    square=straightsides*broad
    skirt=skirt+.80*smallwaist-.7*tank
    longd=np.minimum(d[:,2],d[:,4]); detail=apparel(pix,soft)
    teeish=-d[:,0]+1.58*tee-.80*longarms-.75*skirt+1.65*tank+1.05*sleeves+.60*shortbody
    longish=-longd+1.40*(longarms-.70)-2.3*tee-.6*skirt-1.25*tank
    flowy=-d[:,6]-.35
    s=-d.copy()
    s[:,3]+=(2.55*skirt-.80*tee-1.1*longarms+2.4*aLine+1.35*longbody-.95*square)
    teeish+=.70*square-1.50*aLine-1.15*longbody
    # A curved shirttail, unstarched fabric that widens at the hips, and
    # narrow shoulders suggest a blouse; broad outerwear often continues
    # to or below the sleeve ends. Evaluate the hem only over body cloth.
    bottom=(1-cap(side[:,::-1].cumsum(1))).sum(1)/24
    curve=cap(((bottom[:,6:9].mean(1)+bottom[:,15:18].mean(1))*.5-bottom[:,10:14].mean(1)-.013)/.094)
    cuffheight=.5*(bottom[:,2:5].min(1)+bottom[:,19:22].min(1))-bottom[:,10:14].mean(1)
    longcoat=cap((cuffheight-.07)/.19)*cap((.91-aspect)/.26)*cap(longarms-.15)
    airyshirt=cap((hip-midriff-.025)/.18)*cap((.98-aspect)/.35)
    detail[:,4]+=.68*longcoat-.56*curve-.48*airyshirt
    detail[:,6]+=1.0*curve+.95*airyshirt-.40*longcoat
    detail[:,2]-=.55*curve+.68*airyshirt
    # Compare pullover, jacket and long-sleeved shirt. Treat outline as a
    # weak style cue; a change of fabric/fastening does not change sleeves.
    dr=np.stack([detail[:,2]-.23*(d[:,2]-longd),detail[:,4]-.23*(d[:,4]-longd),detail[:,6]-.12*cap(d[:,6]-longd,-5,10)-.55],1)
    dr=dr-dr.max(1,keepdims=True)
    s[:,2]=longish+dr[:,0]; s[:,4]=longish+dr[:,1]
    tdiff=detail[:,6]-detail[:,0]-2.0-.22*cap(d[:,6]-d[:,0],-8,8)-1.30*sleeves-.8*tank-.45*tee
    s[:,0]=teeish-np.maximum(tdiff,0)
    s[:,6]=np.maximum(teeish+np.minimum(tdiff,0),longish+dr[:,2])
    # Loose blouses and three-quarter sleeves need not match either long
    # or tee sleeves. Give shirt-pattern specificity its own path.
    s[:,6]=np.maximum(s[:,6],flowy-1.35+.9*cap(detail[:,6]-1.4,-1.0,1.0)-.6*tank-.65*sleeves)
    # Identify the long narrow slit between legs. It can veer off centre
    # (posed trousers) but should continue up from the bottom, and fabric
    # should lie on both sides of it, unlike a skirt with a single fold.
    lobe=cap((.46-soft[:,15:23,9:15].min(2).mean(1))/.4)*cap((soft[:,16:22,3:9].mean((1,2))-.32)/.4)*cap((soft[:,16:22,15:21].mean((1,2))-.32)/.4)
    flatwaist=cap((widths[:,1:3].mean(1)-.50)/.39)
    # The waistband is a continuous strip rather than two sleeve caps.
    # Smooth trousers can have adjoining legs but a substantial waistband
    # still contrasts with a dress neckline and the flared skirt below it.
    belt=cap((soft[:,0:3,6:18].mean((1,2))-.44)/.38)*flatwaist
    trouserline=cap((widths[:,5:9].mean(1)-widths[:,18:22].mean(1)+.03)/.20)
    s[:,1]+=2.25*lobe*(flatwaist+.35)+.70*(flatwaist-.5)*narrow+2.75*belt*(.25+trouserline)*narrow
    s[:,3]-=2.40*lobe+.80*belt*trouserline
    # Additional geometric tests which do not depend on the line drawing
    # style. Holes in the vamp / under the arch, open space below a bag
    # handle, and a step from the toe up to a boot shaft.
    def boundaries(m):
        v=cap((m-.15)*1.5)
        return (1-cap(v.cumsum(axis=1))).sum(axis=1),(1-cap(v[:,::-1].cumsum(axis=1))).sum(axis=1)
    upper,lower=boundaries(soft); flip=upper[:,3:10].mean(1)<upper[:,14:21].mean(1)
    roof=np.where(flip[:,None],upper[:,::-1],upper); floor=np.where(flip[:,None],lower[:,::-1],lower)
    closed=cap(1-(upper+lower)/24); hollow=(closed-soft.mean(axis=1)).clip(0,1)
    holes=hollow.mean(1); arch=floor[:,10:18].mean(1)-floor[:,20:23].mean(1)
    sided=roof[:,2:9].mean(1)-roof[:,16:22].mean(1)
    smooth=soft.reshape(len(soft),12,2,12,2).mean(axis=(2,4))
    missing=cap((holes-.03)/.19)
    heel=cap((arch-1.4)/4.5); shaft=cap((sided-6.2)/5)
    lowshaft=cap((2.35-aspect)/.88)
    deep=cap((np.maximum(roof[:,1:7].mean(1)-roof[:,8:14].mean(1),0)-4)/5)
    # Gap above the insole: seek background bracketed by strap material
    # horizontally or vertically. Unlike mere darkness, background near
    # an actual hole is typically close to zero, with bright strap edges.
    cloth=cap((soft-.20)*1.9); lr=np.minimum(np.maximum.accumulate(cloth,axis=2),np.maximum.accumulate(cloth[:,:,::-1],axis=2)[:,:,::-1]); ud=np.minimum(np.maximum.accumulate(cloth,axis=1),np.maximum.accumulate(cloth[:,::-1],axis=1)[:,::-1])
    through=(cap(np.maximum(lr,ud)-cloth)*cap(1-soft*1.24))[:,2:20,2:22]; void=cap((through.mean((1,2))-.057)/.19)
    shoearea=soft.mean((1,2)); airy=cap((.60-shoearea)/.17)
    s[:,5]+=3.5*missing+2.0*heel+1.70*airy+3.3*void
    s[:,7]-=1.2*void; s[:,9]-=1.2*void
    # A low trainer rises to the tongue and falls away behind it. Boots
    # have a more sustained plateau and often a clear heel breast.
    tongue=roof[:,19:23].mean(1)-roof[:,13:18].min(1)
    tongue=cap((tongue-1.2)/4.2); plateau=cap((2.5-roof[:,17:22].mean(1))/2.5)
    elongated=cap((aspect-1.72)/.75)
    s[:,7]+=1.35*elongated+1.50*tongue-3.5*missing-2.5*heel-1.25*shaft+.65*deep
    s[:,9]+=1.30*shaft+1.7*(lowshaft-.55)-3.5*missing+.60*heel-1.1*deep+1.15*plateau-.85*tongue
    # Even with a broad frame, textile handles leave a hole in the top
    # half. Non-handle bags can be recognised by the simple broad panel.
    holeside=(1-soft[:,2:11,8:16]).mean((1,2))
    panel=soft[:,14:21,5:19].mean((1,2)); sym=np.abs(smooth-smooth[:,:,::-1]).mean((1,2))
    s[:,8]+=2.8*cap((holeside-.35)/.45)*cap((panel-.5)/.36)+1.8*cap((aspect-1.02)/.37)*cap((.19-sym)/.14)
    # A handbag is sometimes perfectly rectangular, with just a narrow
    # strap poking out of one side. Test also for an arch flanked by its
    # two supports above a solid panel, not merely missing neck fabric.
    topvoid=(1-soft[:,2:10,9:15]).mean((1,2)); support=np.minimum(soft[:,2:10,1:10].max(2).mean(1),soft[:,2:10,14:23].max(2).mean(1))
    column=cap((soft[:,13:21].mean((1,2))-.77)/.20); handle=cap((topvoid-.48)/.47)*support*column
    rect=cap((widths[:,3:6].mean(1)-.76)/.20)*cap((hem-.81)/.17)*cap((.055-gap)/.055)
    s[:,8]+=2.25*handle+1.5*rect*cap((aspect-.92)/.34)
    # Very flat sandals can be almost just a sole and toe cap, with an
    # isolated rear strap. A shoe vamp instead rises fairly smoothly
    # from the toe to the lacing. Check for valleys after a local rise.
    valley=np.maximum.reduce([roof[:,i]-np.maximum(roof[:,max(i-4,0):i].min(1),roof[:,i+1:i+5].min(1)) for i in range(6,19)])
    openvamp=cap((valley-3.3)/5.2); pairrise=cap((roof[:,10:18].max(1)-roof[:,2:9].min(1)-3)/6)
    s[:,5]+=1.6*openvamp+.9*pairrise+1.7*cap((aspect-3.0)/1.0)*cap((.79-shoearea)/.18)
    s[:,7]-=1.35*openvamp; s[:,9]-=.55*openvamp
    return s.argmax(axis=1)

def predict(x):
    out=np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),96): out[start:start+96]=classify(x[start:start+96,0])
    return out
'''

out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
(out/'solution.py').write_text(code)
np.savez(out/'constants.npz',drawings=book())
