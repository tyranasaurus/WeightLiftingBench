import math
import numpy as np

# Soft truth values, including ramps with a decreasing right endpoint.
def unit(a):
    return np.clip(a, 0., 1.)

def ramp(a, lo, hi):
    return unit((a-lo)/(hi-lo))

def skincolor(r,g,b):
    # In gamma RGB a skin swatch follows a warm red-yellow direction.
    # Separate hue from the intensity and give saturation a broad plateau.
    p = r-g
    q = g-b
    amp = r-b
    hue = q/(np.maximum(amp,0.)+.018)
    sat = amp/(.13+.87*r)
    col = ramp(hue,-.30,.14)*ramp(hue,.83,.53)
    col *= ramp(sat,.045,.23)*ramp(sat,1.02,.74)
    col *= ramp(r,.095,.27)*ramp(p,-.002,.032)
    return col

def diffgrid(a, dy, dx):
    # One pixel differences, extended by repetition at the boundary.
    iy=np.clip(np.arange(96)+dy,0,95)
    ix=np.clip(np.arange(96)+dx,0,95)
    return a[:,iy,:][:,:,ix]-a

def prefix(a):
    return np.pad(a.cumsum(2).cumsum(1),((0,0),(1,0),(1,0)))

def fullbox(im,ht,wd):
    # Centered dense box sum, factored along the two axes.
    ay=np.arange(96)-round((ht-1)/2); ax=np.arange(96)-round((wd-1)/2)
    y0=np.clip(ay,0,95); y1=np.clip(ay+ht,y0+1,96)
    x0=np.clip(ax,0,95); x1=np.clip(ax+wd,x0+1,96)
    tmp=im[:,y1,:]-im[:,y0,:]
    return (tmp[:,:,x1]-tmp[:,:,x0])/((y1-y0)[:,None]*(x1-x0)).astype(np.float32)

def box(im,y,x,h,w,oy,ox,bh,bw):
    # Mean of a box in a regular grid of detection windows. Inputs and
    # offsets are relative to the face center, not to image coordinates.
    ht=max(1,round(h*bh)); wd=max(1,round(w*bw))
    by=y+round(oy*h-(ht-1)*.5); bx=x+round(ox*w-(wd-1)*.5)
    y0=np.clip(by,0,95); x0=np.clip(bx,0,95)
    y1=np.clip(by+ht,y0+1,96); x1=np.clip(bx+wd,x0+1,96)
    return (im[:,y1[:,None],x1]-im[:,y1[:,None],x0]+im[:,y0[:,None],x0]-im[:,y0[:,None],x1])/((y1-y0)[:,None]*(x1-x0)).astype(np.float32)

def sample(a,y,x,h,w,dy,dx):
    return a[:,np.clip(y+round(h*dy),0,95)[:,None],np.clip(x+round(w*dx),0,95)]

def colorstep(c1,c2):
    ma,p,q=c1; mb,r,s=c2
    return np.abs(ma-mb)+.55*(np.abs(p-r)+np.abs(q-s))

def trip(im,y,x,h,w,oy,ox,bh,bw):
    return tuple(box(ch,y,x,h,w,oy,ox,bh,bw) for ch in im)

def mean2(a,b):
    return tuple(.5*(u+v) for u,v in zip(a,b))

def sampled(a,dy,dx):
    if dx==0:
        return a[:,np.clip(np.arange(96)+dy,0,95),:]
    if dy==0:
        return a[:,:,np.clip(np.arange(96)+dx,0,95)]
    return a[:,np.clip(np.arange(96)+dy,0,95),:][:,:,np.clip(np.arange(96)+dx,0,95)]

def contours(lum,red,blue,grid,h):
    # Local gradient orientation evidence at the spatial bandwidth of
    # the proposed outline, normalized by the nearby texture energy.
    m=max(1,round(h*.10))
    ht=max(1,round(h*.16)); wd=max(1,round(h*.14))
    a=fullbox(lum,ht,wd); v=fullbox(red,ht,wd); b=fullbox(blue,ht,wd)
    def derivative(a):
        return sampled(a,0,m)-sampled(a,0,-m),sampled(a,m,0)-sampled(a,-m,0)
    dx,dy=derivative(a); ex,ey=derivative(v); fx,fy=derivative(b)
    mag=np.sqrt(dx*dx+dy*dy+.43*(ex*ex+ey*ey+fx*fx+fy*fy))
    pool=fullbox(prefix(mag),round(h*2.3),round(h*1.9))
    den=pool*1.40+.046
    def orient(nx,ny):
        res=np.sqrt((nx*dx+ny*dy)**2+.43*((nx*ex+ny*ey)**2+(nx*fx+ny*fy)**2))
        orth=np.sqrt((ny*dx-nx*dy)**2+.43*((ny*ex-nx*ey)**2+(ny*fx-nx*fy)**2))
        strength=unit((res-.45*orth)/den)
        mean=fullbox(prefix(strength),max(1,round(h*.38)),max(1,round(h*.34)))
        return .52*mean+.48*np.maximum.reduce([strength,sampled(strength,0,m),sampled(strength,0,-m),sampled(strength,m,0),sampled(strength,-m,0)])
    return (orient(1.,0.),orient(0.,1.),orient(.707,.707),orient(.707,-.707))

def outline(maps,y,x,h,w):
    # Unsigned edgels along a crown, temples, trapezius, deltoids and
    # upper arms. Balance the sides: a lone strong edge is not enough.
    def e(k,dy,dx):
        return sample(maps[k],y,x,h,w,dy,dx)
    def both(a,b):
        return .5*(a+b)-.23*np.abs(a-b)
    crest=.40*e(1,-.59,0)+.60*both(e(2,-.45,-.35),e(3,-.45,.35))
    temples=both(.52*e(0,-.15,-.52)+.48*e(0,.16,-.48),.52*e(0,-.15,.52)+.48*e(0,.16,.48))
    # The head ends at the neck or can taper into a visible jaw.
    jaw=both(e(3,.33,-.38),e(2,.33,.38))
    head=.36*crest+.49*temples+.15*jaw
    def shirt(sign,k,shift):
        sl=.24*e(k,.64+shift,sign*.48)+.33*e(1,.77+shift,sign*.90)+.20*e(k,.95+shift,sign*1.22)
        return sl+.14*e(0,1.30+shift,sign*1.40)+.09*e(0,1.76+shift,sign*1.44)
    # Shrugging, leaning and camera perspective change shoulder height.
    sh=both(np.maximum(shirt(-1,2,0),shirt(-1,2,.26)),np.maximum(shirt(1,3,0),shirt(1,3,.26)))
    # Compare with points just off the putative contour, *not* just an
    # absolute edge sum: clutter produces oriented edges everywhere.
    off=.25*(e(0,-.02,-1.05)+e(0,-.02,1.05)+e(0,-.12,0)+e(1,-1.07,0))
    flat=.25*(e(1,.47,-.94)+e(1,.47,.94)+e(1,1.41,-.68)+e(1,1.41,.68))
    hd=ramp(head-.32*off,.15,.68)
    shoulder=ramp(sh-.32*flat,.16,.67)
    return hd,shoulder

def binarysim(a,b):
    # Vectors for inside minus outside on the two sides of an object.
    # Opposing colors usually indicate an edge through (not around) it.
    u,v,t=a; x,y,z=b
    cross=u*x+v*y+t*z
    return unit((cross+.0015)/(.004+np.sqrt((u*u+v*v+t*t)*(x*x+y*y+z*z))))

def maskdiff(a,b):
    return tuple(u-v for u,v in zip(a,b))

def best(a):
    return a.max(axis=(1,2))

def skindistribution(mask):
    # Compare multi-resolution maps of plausible skin and its context.
    # Broad, warm scenes shouldn't score for the sheer number of boxes
    # they contain; human skin tends to be concentrated in just a few
    # compact regions, often higher in the picture than the clothing.
    si=prefix(mask)
    upper=mask[:,4:54,6:90].mean((1,2)); lower=mask[:,60:94,4:92].mean((1,2))
    rank=np.zeros(len(mask),np.float32)
    cnt=np.zeros_like(rank)
    for size in (3,6,12,24):
        core=fullbox(si,size,max(2,round(.74*size)))
        surround=fullbox(si,size*3,size*3)
        inner=np.minimum(core,ramp(core-surround,.06,.66))
        # A compact component has high response in both directions.
        barh=fullbox(si,size,3*size); barv=fullbox(si,3*size,max(2,round(.74*size)))
        enclosed=ramp(core-np.maximum(barh,barv),.035,.39)
        value=inner*(.50+.50*enclosed)
        cut=size//2
        value=value[:,cut:70,cut:96-cut]
        peak=value.max((1,2))
        rank=np.maximum(peak,rank)
        cnt+=ramp(value,.48,.86).mean((1,2))
    return .105*(rank-.65)+.050*ramp(upper-lower,.01,.14)-.066*ramp(lower-upper,.04,.20)-.047*ramp(cnt,.14,.48)

def correlation(im,y,x,h,w):
    # Sample a common grid in each proposed head window; the two grids
    # of displacements have fixed size, independent of the batch size.
    fy=(np.arange(12)+.5)*1.22/12-.60
    fx=(np.arange(12)+.5)*1.24/12-.62
    ht=max(1,round(h*1.22/12)); wd=max(1,round(w*1.24/12))
    by=y[:,None]+np.rint(h*fy-.5*(ht-1)).astype(int)
    bx=x[:,None]+np.rint(w*fx-.5*(wd-1)).astype(int)
    y0=np.clip(by,0,95)[:,None,:,None]; y1=np.clip(by+ht,by.clip(0,95)+1,96)[:,None,:,None]
    x0=np.clip(bx,0,95)[None,:,None,:]; x1=np.clip(bx+wd,bx.clip(0,95)+1,96)[None,:,None,:]
    patch=(im[:,y1,x1]-im[:,y0,x1]+im[:,y0,x0]-im[:,y1,x0])/((y1-y0)*(x1-x0)).astype(np.float32)
    patch=np.ascontiguousarray(patch.reshape(len(im),len(y)*len(x),12,12))
    patch-=patch.mean((2,3),keepdims=True)
    denom=np.sqrt((patch*patch).sum((2,3))+.075)
    flat=patch.reshape(len(im),-1,144)
    cor=(flat@CONSTANTS['face'])/denom[:,:,None]
    # Subtracting row/column marginals discounts simple horizontal
    # stripes or vertical light gradients which correlate spuriously.
    patch=patch-patch.mean(2,keepdims=True)-patch.mean(3,keepdims=True)
    rel=np.sqrt((patch*patch).sum((2,3))+.045)
    corr2=(patch.reshape(len(im),-1,144)@CONSTANTS['details'])/rel[:,:,None]
    score=(.56*cor+.44*corr2).max(2)
    return score.reshape(len(im),len(y),len(x))

def scalecheck(im,S,edge,y,x,h,w,scene,maps):
    yy=y[:,None]; xx=x[None,:]
    def s(dy,dx):
        return sample(S,y,x,h,w,dy,dx)
    def c(dy,dx,bh,bw):
        return trip(im,y,x,h,w,dy,dx,bh,bw)
    fy=.30*s(0,0)+.20*(s(.10,-.22)+s(.10,.22))+.15*(s(-.28,0)+s(.32,0))
    # Soft region immediately outside the cheeks and chin. Allow a
    # partial occlusion, hair over one cheek, or the face in profile.
    left=.65*s(0,-.73)+.35*s(.28,-.64)
    right=.65*s(0,.73)+.35*s(.28,.64)
    below=s(.91,0); above=s(-.76,0)
    surround=.30*(left+right)+.20*(above+below)
    island=ramp(fy-surround,.08,.72)
    separate=ramp(fy-.5*(left+right),.12,.70)
    bounded=.55*island+.3*separate+.15*ramp(fy-.5*(above+below),.10,.67)
    # Is the skin in the central part, rather than a warm corner of
    # the window? (For a side view one cheek alone is sufficient.)
    filled=.5*np.maximum(s(.08,-.22),s(.08,.22))+.5*s(.22,0)
    skin=.7*fy+.3*filled
    # How well does the color of the two cheeks agree?
    lc=c(.09,-.22,.21,.26); rc=c(.09,.22,.21,.26)
    cf=mean2(lc,rc)
    balance=ramp(colorstep(lc,rc),.27,.045)
    # Normalize facial contrasts against the local range to resist
    # mistaking a few tiny grey-level ripples for face structure.
    sd=box(edge[0],y,x,h,w,0,0,1,1)
    av=box(im[0],y,x,h,w,0,0,1,1)
    noise=.033+.38*np.sqrt(np.maximum(0.,sd-av*av))
    def light(dy,dx,bh,bw):
        return box(im[0],y,x,h,w,dy,dx,bh,bw)
    forehead=light(-.32,0,.16,.50)
    le=light(-.12,-.235,.14,.26); re=light(-.12,.235,.14,.26)
    eye=.5*(le+re)
    ridge=light(-.06,0,.19,.14)
    mouth=light(.268,0,.10,.38)
    chin=light(.395,0,.105,.40)
    eyes=ramp((cf[0]-eye)/noise,.15,1.35)*.50+ramp((forehead-eye)/noise,.12,1.3)*.32+ramp((ridge-eye)/noise,.12,1.15)*.18
    eyes*=.52+.48*ramp((np.maximum(lc[0]-le,rc[0]-re)*.15+np.minimum(lc[0]-le,rc[0]-re)*.85)/noise,-.32,.65)
    lips=ramp((.58*cf[0]+.42*chin-mouth)/noise,.10,.93)
    detail=eyes*(.73+.27*lips)
    if h<6:
        detail=detail*.65
    # Hair is frequently a dark cap above the face, or at its temples.
    hairtop=light(-.56,0,.28,.70)
    temple= .5*(light(-.20,-.49,.47,.20)+light(-.20,.49,.47,.20))
    cap=ramp((cf[0]-hairtop)/(.075+.21*cf[0]),.16,1.3)
    hair=.75*cap+.25*ramp((cf[0]-temple)/noise,.16,1.55)
    # A candidate head has edges on both sides, and should not just
    # straddle a single color boundary, e.g. grass meeting sand.
    cLeft=c(.01,-.79,.62,.33); cRight=c(.01,.79,.62,.33)
    fsep=.60*np.minimum(colorstep(lc,cLeft),colorstep(rc,cRight))+.40*np.maximum(colorstep(lc,cLeft),colorstep(rc,cRight))
    headedge=ramp(fsep,.045,.26)*(.76+.24*binarysim(maskdiff(lc,cLeft),maskdiff(rc,cRight)))
    # Clothing occupies an area wider than the head just below it.
    # Compare the center of a shirt with each shoulder, the space just
    # ABOVE each shoulder, and the scene outside the arm silhouettes.
    body=c(1.73,0,.78,1.08)
    bl=c(1.43,-.79,.49,.54); br=c(1.43,.79,.49,.54)
    al=c(.47,-.92,.57,.50); ar=c(.47,.92,.57,.50)
    ol=c(1.70,-1.70,.78,.52); ori=c(1.70,1.70,.78,.52)
    bodydiff=colorstep(body,cf)
    hchange=ramp(bodydiff,.067,.32)
    shl=colorstep(bl,al); shr=colorstep(br,ar)
    sh=.65*np.minimum(shl,shr)+.35*np.maximum(shl,shr)
    shcap=ramp(sh,.055,.28)
    bsl=colorstep(bl,ol); bsr=colorstep(br,ori)
    bwall=ramp(.7*np.minimum(bsl,bsr)+.3*np.maximum(bsl,bsr),.048,.27)
    bd=.5*(colorstep(bl,body)+colorstep(br,body))
    smooth=ramp(bd,.26,.035)
    similar=binarysim(maskdiff(bl,al),maskdiff(br,ar))
    shoulders=shcap*(.26+.40*smooth+.34*similar)
    silhouette=.60*shoulders+.40*bwall*(.4+.60*smooth)
    visible=ramp((95.-yy)/h,.65,1.85)*ramp((np.minimum(xx,95-xx)+w*.5)/w,.52,1.7)
    silhouette*=visible
    garment=hchange*(.42+.58*smooth)*ramp((95.-yy)/h,.40,1.52)
    # Hands and arms: look for warm islands at plausible displacements
    # from the head, NOT a common hue throughout the entire region.
    def arm(sign):
        mid=np.maximum(s(2.55,sign*1.20),s(1.87,sign*1.42))
        bent=np.maximum(s(2.30,sign*.70),s(1.32,sign*1.56))
        hand=np.maximum(mid,bent)
        back=.25*(s(1.40,sign*2.20)+s(2.15,sign*2.22)+s(2.76,sign*2.00)+s(2.15,0))
        return np.minimum(hand,ramp(hand-back,.10,.66))*ramp((95-yy)/h,1.1,2.35)
    armL=arm(-1); armR=arm(1)
    arms=.65*np.maximum(armL,armR)+.35*np.minimum(armL,armR)
    # An upright dark-or-light figure: similar trouser / leg colors
    # below a compact torso separated from a wider background. Cue
    # is optional; seated people and portraits need no visible legs.
    hip=c(3.08,0,.75,1.05)
    thigh=c(4.2,0,1.05,1.05)
    gl=c(4.10,-1.53,1.08,.62); gr=c(4.10,1.53,1.08,.62)
    ld=np.minimum(colorstep(thigh,gl),colorstep(thigh,gr))
    legs=ramp(ld,.060,.27)*(.35+.34*ramp(colorstep(hip,thigh),.25,.045)+.31*binarysim(maskdiff(thigh,gl),maskdiff(thigh,gr)))
    legs*=ramp((95.-yy)/h,3.,4.60)*ramp((np.minimum(xx,95-xx)+.36*w)/w,.6,1.55)
    hd,sh=outline(maps,y,x,h,w)
    shape=.55*hd+.45*sh
    # Test denser area coverage; a few individual warm pixels on busy
    # backgrounds should not mimic a whole face. Subtract skin on an
    # outer ring AND penalize an elliptical core made of two colors.
    def skinbox(dy,dx,bh,bw):
        return box(edge[1],y,x,h,w,dy,dx,bh,bw)
    core=.53*skinbox(0,0,.74,.69)+.47*skinbox(0,0,.57,.93)
    ring=.23*(skinbox(0,-.84,.77,.43)+skinbox(0,.84,.77,.43))+ .19*(skinbox(-.79,0,.40,.95)+skinbox(.91,0,.48,.95))+.04*(s(-.52,-.61)+s(-.52,.61)+s(.51,-.66)+s(.51,.66))
    blob=ramp(core-ring,.12,.71)
    # Very smooth material with skin hue can be wood, fences, food or
    # animal fur. Seek a second, chromatically consistent exposed patch
    # around an arm, rather than rewarding smooth warm backgrounds.
    def armmatch(sign):
        klist=[]
        for dy,dx in ((1.85,1.45),(2.6,1.18),(2.30,.74)):
            co=c(dy,sign*dx,.33,.41)
            # chroma comparisons compensate for the arm being in shade.
            rel=(co[0]+.22)/(cf[0]+.22)
            dist=np.abs(co[1]-cf[1]*rel)+np.abs(co[2]-cf[2]*rel)
            klist.append(s(dy,sign*dx)*ramp(dist,.16,.033))
        return np.maximum.reduce(klist)
    arms=arms*(.48+.52*np.maximum(armmatch(-1),armmatch(1)))
    skin=.65*core+.35*skin
    bounded=.43*bounded+.57*blob
    # Weak photographic prior: heads are less likely right at the
    # bottom, especially when the apparent head is tiny.
    center=unit(1.-abs(xx-47.5)/56.)
    prior=.044*(center-.48)-.21*ramp(yy,.63*96,.99*96)+.03*(1.-abs(yy-29)/62.)
    prior-=.050*ramp(8-h,0,5)
    prior=prior+scene[:,None,None]
    # Hand-specified evidence budgets for three arrangements: small
    # exposed head over clothes, detailed closer face, covered head
    # on a standing person. Each one needs multiple cues to fire.
    colored=.37*skin+.30*bounded+.11*headedge+.10*garment+.15*silhouette+.085*hair+.09*detail+.072*arms+.064*legs+.045*balance
    colored-=.946
    face=.32*detail+.095*lips+.12*balance+.11*hair+.09*headedge+.095*skin+.09*bounded+.085*garment+.10*silhouette+.035*arms
    face-=.862+.065*ramp(11-h,0,8)
    # In side/back views a head can be all hair, with no skin. Demand
    # a neck/shoulder transition and evidence from the lower body.
    darker=ramp((.5*(cLeft[0]+cRight[0])-cf[0])/(.12+.22*cf[0]),.22,1.2)
    solidhead=headedge*(.35+.35*balance+.30*darker)
    figure=.19*solidhead+.30*silhouette+.22*legs+.095*balance+.085*hchange+.08*arms+.06*bounded+.045*skin
    figure-=.83
    colored+=.12*(hd-.54)+.074*(sh-.54)*visible
    face+=.14*(hd-.52)+.06*(sh-.54)*visible
    figure+=.23*(shape-.56)
    portrait=np.full_like(face,-1.)
    if h>=11:
        match=correlation(im[0],y,x,h,w)
        headlike=ramp(match,.20,.68)
        face+=.29*(headlike-.59)
        colored+=.086*(headlike-.48)*detail
        portrait=.65*headlike+.11*headedge+.07*hair+.10*skin+.07*bounded+.09*detail+.095*shape+.04*garment-.886
        portrait-=.065*ramp(13-h,0,5)
    score=np.maximum(np.maximum(colored,face),np.maximum(figure,portrait))+prior
    # A single chance coincidence should carry less weight than a
    # geometrically stable local maximum. Spatial averaging is light
    # at the smallest scales, where the sampling stride is coarse.
    padded=np.pad(score,((0,0),(1,1),(1,1)),mode='edge')
    nbor=(padded[:,:-2,1:-1]+padded[:,2:,1:-1]+padded[:,1:-1,:-2]+padded[:,1:-1,2:])*.25
    stable=score*.8+nbor*.2+.022
    return stable

def scenecontext(lum,p,q,r,g,b,dx,dy):
    # Context alone is ambiguous. Bound the change in the detector
    # threshold and emphasize a few interpretable scene families.
    warmbright=ramp(p,.008,.08)*ramp(q,-.11,-.014)*ramp(lum,.13,.35)
    uniform=ramp(warmbright.mean((1,2)),.0015,.03)
    fieldpix=ramp(g-b,.04,.19)*ramp(g-r,-.045,.028)*ramp(g,.11,.34)
    # A stripe of closely mown grass, rather than green leaves.
    turf=fieldpix[:,45:92].mean((1,2))
    graingrass=((np.abs(dx)+np.abs(dy))*fieldpix)[:,45:92].mean((1,2))/(turf+.015)
    playing=ramp(turf,.11,.57)*ramp(graingrass,.17,.05)
    # Water below the horizon with white crests on or near the water.
    water=ramp(b-r,.035,.18)*ramp(g-r,-.01,.11)*ramp(b-g,-.10,.055)*ramp(lum,.12,.40)
    sea=water[:,50:].mean((1,2))
    foam=ramp(lum,.66,.86)*ramp(np.abs(p)+np.abs(q),.18,.04)
    seasport=ramp(sea,.12,.56)*ramp(foam[:,42:].mean((1,2)),.02,.18)
    snow=ramp(lum,.52,.81)*ramp(b-r,-.02,.055)*ramp(np.abs(g-b),.19,.04)
    snowfield=snow[:,45:].mean((1,2))
    winter=ramp(snowfield,.17,.61)
    # Hue-rich food and fleshy or wood-colored backgrounds can
    # generate lots of spurious warm blobs. Yellow vs pink is helpful.
    yellow=ramp(q,.04,.18)*ramp(q-p,-.014,.07)*ramp(r,.23,.62)
    red=ramp(p,.13,.32)*ramp(q,-.05,.04)*ramp(p/(.09+r),.31,.57)
    foodish=ramp((yellow+.55*red).mean((1,2)),.07,.31)
    white=ramp(lum,.71,.88)*ramp(np.abs(p)+np.abs(q),.16,.04)
    # Empty artificial interiors/vehicles often have long straight
    # edges: orientation and polarity surviving a long spatial average.
    def coherence(a,vertical):
        pp=prefix(a); ab=prefix(np.abs(a))
        ht,wd=(19,2) if vertical else (2,19)
        signed=np.abs(fullbox(pp,ht,wd)); unsigned=fullbox(ab,ht,wd)
        return (ramp(signed,.023,.095)*ramp(signed/(unsigned+.008),.38,.83)).mean((1,2))
    rigid=coherence(dx,True)+coherence(dy,False)
    interior=ramp(rigid,.15,.36)
    # High frequencies in the background vs the subject: a shallow
    # depth-of-field or smooth backdrop is common in action portraits.
    texture=np.abs(dx)+np.abs(dy)
    middle=texture[:,23:80,23:73].mean((1,2))
    side=(texture[:,:,0:16].mean((1,2))+texture[:,:,80:].mean((1,2)))*.5
    subject=ramp(middle/(.018+side),1.08,2.55)
    # A circular white rim with a warm/multicolored center often
    # belongs to a plate, not a portrait. Score only with multiple arcs.
    dish=np.zeros(len(lum),np.float32)
    yy=np.arange(96)[:,None]; xx=np.arange(96)[None,:]
    for cy,cx,ry,rx in ((50,48,41,43),(56,48,28,37),(56,48,33,45),(49,45,31,32),(61,49,33,42),(51,52,43,36)):
        radius=((yy-cy)/ry)**2+((xx-cx)/rx)**2
        ring=(radius>.71)&(radius<1.04); center=radius<.52
        rim=white[:,ring].mean(1)
        plate=ramp(rim,.23,.60)*ramp(rim-white[:,center].mean(1),.09,.40)
        stuff=(yellow+red)[:,center].mean(1)
        dish=np.maximum(dish,plate*ramp(stuff,.045,.25))
    return .14*playing+.08*seasport+.09*winter+.054*uniform+.040*subject-.09*interior-.093*foodish-.22*dish

def classify(a):
    r,g,b=a[:,0],a[:,1],a[:,2]
    lum=.27*r+.55*g+.18*b
    p=r-g; q=g-b
    mskin=skincolor(r,g,b)
    # Mild illuminant correction. Keep the original response: not
    # every scene has a grey average. All statistics here are per-photo.
    rl=r.mean((1,2),keepdims=True); gl=g.mean((1,2),keepdims=True); bl=b.mean((1,2),keepdims=True)
    neutral=(rl+2*gl+bl)*.25+.06
    rr=np.clip((neutral/(rl+.06))**.55,.83,1.18)
    gg=np.clip((neutral/(gl+.06))**.55,.86,1.14)
    bb=np.clip((neutral/(bl+.06))**.55,.82,1.22)
    mskin=np.maximum(mskin,.83*skincolor(r*rr,g*gg,b*bb))
    im=tuple(prefix(ch) for ch in (lum,p,q))
    si=prefix(mskin)
    edge=[prefix(lum*lum),si]
    grid=np.arange(96)
    # Small global corrections for broad areas of skin-tone material,
    # strong rectilinear man-made backgrounds, and snow/field sports.
    warm=mskin.mean((1,2))
    scene=-.070*ramp(warm,.29,.65)
    dx=diffgrid(lum,0,1); dy=diffgrid(lum,1,0)
    ax=np.abs(dx); ay=np.abs(dy)
    straight=(np.maximum(ax,ay)-np.minimum(ax,ay)).sum((1,2))/(.1+(ax+ay).sum((1,2)))
    scene-=.043*ramp(straight,.55,.72)
    scene+=scenecontext(lum,p,q,r,g,b,dx,dy)+skindistribution(mskin)
    # Very weak/strong overall color casts shift the odds of finding
    # a single warm patch by chance. Absence is weak negative evidence.
    scene-=.060*ramp(warm,.031,.002)
    high=np.full(len(a),-2.,np.float32)
    across=np.full((len(a),4,4),-2.,np.float32)
    for h,ws,stride in ((4,(2,4),2),(7,(4,7),2),(12,(6,12),3),(18,(9,17),4),(27,(14,25),4),(40,(20,37),6),(60,(30,54),6),(80,(47,),8)):
        y=np.arange(stride//2,96,stride); x=y
        maps=contours(im[0],im[1],im[2],grid,h)
        aspects=[]
        for w in ws:
            S=fullbox(si,max(1,round(h*.32)),max(1,round(w*.38)))
            aspects.append(scalecheck(im,S,edge,y,x,h,w,scene,maps))
        raw=.86*np.maximum.reduce(aspects)+.14*np.minimum.reduce(aspects)
        high=np.maximum(high,best(raw))
        # All strides divide 24, so these pooling cells partition the
        # same regions of the photo at each template size.
        tile=24//stride
        across=np.maximum(across,raw.reshape(len(a),4,tile,4,tile).max((2,4)))
    group=np.sort(across.reshape(len(a),16),axis=1)[:,-3]
    return (high+.044*ramp(group,-.12,.07)>0).astype(np.int64)

def predict(x):
    result=np.empty(len(x),np.int64)
    for start in range(0,len(x),12):
        batch=x[start:start+12]
        result[start:start+12]=classify(batch)
    return result
