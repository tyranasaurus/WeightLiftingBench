"""Construct a catalogue of geometric sewing patterns and accessory profiles.
The constructor takes no image input. All dimensions below are design dimensions.
"""
import math
import sys
from pathlib import Path
import numpy as np

# This code is shared by the renderer (one pattern at a time, or a stack of
# independent patterns) and the recognizer. All reductions are within an image.
geometry = '''import numpy as np

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
'''

# Categories and styles are separate: for instance a blouse needn't have a
# placket, and a short jacket needn't have the hem of an overcoat.
styles = [
 ('tee',0,1100),('tank',0,640),('captee',0,480),('longtee',0,300),
 ('jeans',1,920),('slacks',1,640),('shorts',1,320),
 ('jumper',2,1700),('boxknit',2,850),('highneck',2,600),('cardigan',2,400),
 ('sundress',3,900),('daydress',3,1250),('sleevedress',3,900),('sheath',3,640),
 ('jacket',4,1500),('overcoat',4,1000),('parka',4,750),
 ('flatstrap',5,1000),('heelstrap',5,1200),('slide',5,450),('wedge',5,450),
 ('buttonshirt',6,1300),('blouse',6,1200),('shortshirt',6,1000),('smock',6,600),
 ('trainer',7,1300),('hightop',7,550),
 ('purse',8,1250),('tote',8,800),('clutch',8,850),('rucksack',8,350),
 ('boot',9,1400),('heelboot',9,1000),('lowboot',9,750),
 ('jerseyfront',0,1800),('knitfront',2,1400),('shirtfront',6,1300),('coatfront',4,1400)]

# Low-discrepancy samples of the independent design dimensions. No fitted
# distribution or feedback from any render; every render stays independent.
primes=[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131]
class Dimensions:
    def __init__(self, index, family):
        self.index=index+1
        self.d=0
        self.family=family
    def __call__(self, a=0., b=1.):
        base=primes[self.d%len(primes)]
        k=self.index
        u=0.
        v=1.
        while k:
            v/=base
            u+=v*(k%base)
            k//=base
        self.d+=1
        u=(u + (self.d*.61803398875+self.family*.41421356237)%1)%1
        return a+(b-a)*u

def mirrored(side):
    return side+[(-x,y) for x,y in side[::-1]]

def garment(style,f):
    # Length, flat chest, hem/chest, shoulder/chest, sleeve length, collar
    # rise and neck half-opening. Units for lengths are garment centimetres.
    specs={
      'tee':((59,75),(43,56),(.88,1.05),(.76,.93),(14,23),(-1,2),(8,12)),
      'captee':((56,72),(40,51),(.9,1.12),(.75,.94),(5,14),(-2,1),(9,15)),
      'tank':((56,76),(35,47),(.88,1.18),(.55,.89),(0,0),(-4,0),(8,14)),
      'longtee':((59,74),(40,53),(.89,1.08),(.74,.92),(40,58),(-2,1),(9,14)),
      'jumper':((56,72),(43,55),(.83,1.0),(.75,.95),(52,64),(-2,2),(8,14)),
      'boxknit':((51,66),(47,62),(.78,.99),(.8,1.02),(43,56),(-2,2),(9,15)),
      'highneck':((56,72),(41,53),(.86,1.03),(.76,.94),(52,63),(4,12),(8,12)),
      'cardigan':((63,83),(43,57),(.9,1.09),(.76,.96),(49,61),(-1,3),(7,12)),
      'sundress':((83,132),(34,46),(1.07,2.0),(.54,.89),(0,0),(-4,0),(7,13)),
      'daydress':((82,114),(37,49),(1.03,1.76),(.73,.96),(5,26),(-2,2),(7,13)),
      'sleevedress':((83,116),(38,50),(1.00,1.65),(.76,.94),(29,62),(-2,3),(7,13)),
      'sheath':((81,118),(34,45),(.87,1.24),(.65,.97),(0,12),(-2,2),(7,13)),
      'jacket':((58,78),(47,60),(.90,1.06),(.73,.9),(52,65),(2,9),(7,12)),
      'overcoat':((79,106),(46,59),(1.02,1.35),(.75,.91),(52,65),(2,9),(7,12)),
      'parka':((66,88),(50,64),(.91,1.14),(.72,.91),(49,65),(4,12),(9,16)),
      'buttonshirt':((62,80),(43,58),(.91,1.06),(.73,.94),(49,62),(1.5,6),(6,10)),
      'blouse':((56,75),(41,56),(.89,1.19),(.72,1.0),(24,57),(-4,2),(9,16)),
      'shortshirt':((56,76),(43,57),(.89,1.13),(.73,.97),(9,25),(-2,5),(7,14)),
      'smock':((65,83),(40,55),(.98,1.33),(.65,.93),(0,19),(-3,2),(8,15))}
    L,C,hemratio,shoulderratio,sleeve,rise,neck=[f(*z) for z in specs[style]]
    dress=style in ('sundress','sheath','daydress','sleevedress')
    coat=style in ('jacket','parka','overcoat')
    knit=style in ('jumper','boxknit','highneck','cardigan')
    tank=style in ('tank','sundress') or (style=='sheath' and sleeve<5)
    hip=C*.5*hemratio
    W=C*.5
    S=W*shoulderratio
    neck=min(S*.86,neck)
    shoulderDrop=f(3.5,8.5) if not tank else f(0,4)
    pit=f(16,24) if not coat else f(21,29)
    pit=min(.42*L,pit)
    waist=f(.68,.99) if dress else f(.84,1.02)
    if coat or knit or style=='buttonshirt': waist=f(.94,1.04)
    waistY=f(.34,.47)*L if dress else f(.48,.7)*L
    if tank: neckdip=f(2,15)
    elif rise>3: neckdip=f(-1.5,2)
    else: neckdip=f(0,4.5)
    if style=='buttonshirt': hemcurve=f(3,9)
    elif knit: hemcurve=f(0,2.4)
    else: hemcurve=f(0,4.5)
    # Outline of the bodice: the back edge, not the front decoration, limits
    # the silhouette in the neck opening.
    side=[(0,-rise+neckdip),(neck*.65,-rise+.7*neckdip),(neck,-rise),
          (neck+(.3 if tank else 1.2),0),(S,shoulderDrop)]
    if tank: side.extend([(S*.97,shoulderDrop+pit*.32),(W*.86,pit*.79)])
    side += [(W,pit),(W*waist,waistY)]
    if dress and style=='sheath': side.append((max(hip, W*1.01),L*.60))
    if knit:
        side.extend([(max(hip*1.05,W*.94),L-f(5,9)),(hip,L-2)])
    side.extend([(hip,L-hemcurve),(hip*.85,L-.35*hemcurve),(hip*.48,L),(0,L+.15*hemcurve)])
    polygons=[mirrored(side)]
    if coat and rise>0:
        # A stand collar or hood is a raised, rounded piece, not a rectangular
        # post. Its sides roll down to the shoulder seam.
        shoulder=[(0,-rise+max(0.,neckdip)),(.62*neck,-rise),(.90*neck,-rise*.6),
                  (neck+1,0),(S,shoulderDrop)]
        polygons=[mirrored(shoulder+side[5:])]
    if tank: return polygons
    if sleeve<29:
        # On a short sleeve, the shoulder-to-outer-cuff distance IS the
        # specified sleeve length. The cuff extends inward from that point
        # (do not add half a cuff beyond the length measurement).
        downward=f(.57,.90)
        for sign in (-1,1):
            run=sleeve*f(.88,1.05)*math.sqrt(1-downward**2)
            X=S+run; Y=shoulderDrop+sleeve*downward+f(-1,1)
            span=f(12,18) if sleeve>9 else f(8,14)
            path=[(S-2,shoulderDrop),(S,shoulderDrop),(S+run*.55+.8,shoulderDrop+(Y-shoulderDrop)*.42),
                  (X,Y),(X-span*downward,Y+span*math.sqrt(1-downward**2)),(W-.6,pit),(S-3,shoulderDrop+2)]
            polygons.append([(sign*x,y) for x,y in path])
        return polygons
    # A hanging sleeve may overlap the body in projection, particularly
    # for a thick jacket on an invisible mannequin; it doesn't necessarily
    # leave two strips of background alongside the torso.
    spread=f(-.42,.17)*W if coat else f(-.24,.25)*W
    wrist=f(8,12) if knit else f(9,14)
    if coat: wrist=f(11,15)
    short=np.clip((35-sleeve)/19,0,1)
    # longer/higher armhole on jackets; a more horizontal cap on tees
    dx= (1-short)*(W-S+.51*wrist+spread) + short*sleeve*f(.52,.88)
    ang=math.atan2(dx,max(1.,sleeve))
    cuffY=shoulderDrop + sleeve*math.cos(ang)
    cuffW=(1-short)*wrist+short*f(13,20)
    outer=S+dx+.48*cuffW*math.cos(ang)
    inner=outer-cuffW*math.cos(ang)
    cuffTilt=cuffW*math.sin(ang)+f(-2,2)
    for sign in (-1,1):
        asym=f(-.07,.07)*sleeve
        endY=cuffY + f(-.035,.035)*sleeve
        endX=outer+asym
        middle=(.40*endX+.60*(W+f(2,6)))
        ytop=endY-cuffTilt*.43
        sleevePoly=[(S*.90,shoulderDrop*.86),(S,shoulderDrop),
            (S+(middle-S)*.57,shoulderDrop+.13*(ytop-shoulderDrop)),
            (middle,shoulderDrop+.37*(ytop-shoulderDrop)),
            (.55*endX+.45*middle+.7,shoulderDrop+.73*(ytop-shoulderDrop)),
            (endX,ytop),(inner+asym,endY+cuffTilt*.57),
            (.50*(inner+asym)+.5*W-.6,.51*endY+.49*pit),(W-.5,pit),
            (S*.82,shoulderDrop+2)]
        polygons.append([(sign*x,y) for x,y in sleevePoly])
    return polygons

def frontview(s,f):
    # An alternative, outline-first construction: draft the outer edge and
    # underarm in one piece. This models garments falling in a different
    # pose than the separate body and sleeve pieces in the cut patterns.
    if s=='jerseyfront':
        w=f(.28,.48); v=f(.46,.77); cuff=f(.19,.38); cuffspan=f(.085,.17)
        neck=f(.095,.18); dip=f(.005,.072); shoulder=w*f(.72,.96)
        cuffside=w/v; end=w*f(.90,1.08); bend=f(.0,.035)
        side=[(0,dip),(neck*.72,dip*.65),(neck,0),(shoulder,.044),
              (shoulder+.33*(cuffside-shoulder),.076),(cuffside,cuff),
              (max(w*.91,cuffside-cuffspan),cuff+f(.026,.081)),(w,cuff-f(.0,.06)),
              (w*f(.92,1.01),.58),(end,.97),(end*.92,1-bend*.55),(end*.38,1),(0,1)]
    else:
        iscoat=s=='coatfront'; woven=s=='shirtfront'
        shoulder=f(.24,.35) if iscoat else f(.225,.33)
        halfneck=f(.062,.123) if iscoat else f(.075,.148)
        rise=f(.08,.19) if iscoat else f(.04,.125)
        dip=f(-.012,.017) if iscoat else f(-.002,.05)
        torso=shoulder*f(.92,1.25)
        wrist=torso+f(.06,.17)
        wristwidth=f(.08,.14) if iscoat else f(.067,.132)
        inner=wrist-wristwidth
        bodyend=1 if iscoat or woven else f(.82,1.0)
        cuff=f(.71,.99) if iscoat else f(.68,1.01) if woven else f(.90,1.04)
        hem=torso*(f(.92,1.14) if iscoat or woven else f(.82,.99))
        roundness=f(.045,.12) if woven else f(.0,.039)
        top=[(0,dip),(.67*halfneck,dip-.008),(halfneck,0),(1.14*halfneck,.032),
             (shoulder*.89,rise*.76),(shoulder,rise)]
        outer=[(shoulder+.37*(wrist-shoulder),rise+.064),(shoulder+.65*(wrist-shoulder),.30),
                (wrist+.012,.71*cuff),(wrist,cuff-.012),(inner,cuff)]
        # A slit opens between the sleeve and body as the cuff moves away
        # from the hip. If not, the body and sleeve simply overlap.
        garmentbody=top+[(torso,.30),(torso*f(.92,1.02),.56),(hem,bodyend-roundness),
                         (hem*.76,bodyend-.24*roundness),(hem*.35,bodyend),(0,bodyend)]
        shapes=[mirrored(garmentbody)]
        for sign in (-1,1):
            drift=f(-.032,.032)
            edge=top[-2:]+outer+[(.6*inner+.4*torso,.64*cuff),(torso-.008,.27),(shoulder*.83,rise+.035)]
            shapes.append([(sign*(x+y*drift),y) for x,y in edge])
        return shapes
    return [mirrored(side)]

def trousers(s,f):
    if s=='shorts': L=f(44,78)
    else: L=f(90,112)
    waist=f(30,44)*.5
    hip=waist*f(1.06,1.33)
    crotch=f(23,37)
    hem=f(7,13) if s=='jeans' else f(10,18)
    gap=f(.4,6.5)
    if s=='shorts': hem=f(12,20); gap=f(.5,4)
    slant=f(-3,3)
    bottom=hem+gap
    side=[(0,f(0,2)),(waist*.86,1),(waist,0),(hip,.51*crotch),
          (max(hip,bottom)*f(.90,1.04),crotch),(.51*bottom+.49*hip-.8,.64*L),
          (bottom,L+slant*.15),(gap,L),(.6*gap,L*.70),(max(0,gap*.2),crotch+3),(0,crotch)]
    poly=mirrored(side)
    return [[(x+slant*y/L,y) for x,y in poly]]

def stroke(points,radius):
    p=np.array(points,dtype=float)
    delta=np.vstack([p[1]-p[0],p[2:]-p[:-2],p[-1]-p[-2]])
    normals=np.stack([-delta[:,1],delta[:,0]],axis=1)
    normals*=radius/np.maximum(1e-8,np.sqrt((normals*normals).sum(axis=1,keepdims=True)))
    return list(map(tuple,np.concatenate([p+normals,(p-normals)[::-1]])))

def bag(s,f):
    # Body width as unit. Handles are bent strips with an empty centre.
    if s=='clutch': h=f(.34,.86); handle=0
    elif s=='rucksack': h=f(.94,1.5); handle=f(.0,.17)
    elif s=='purse': h=f(.44,.9); handle=f(.17,.82)
    else: h=f(.62,1.18); handle=f(.12,.64)
    top=f(.70,1.03) if s!='tote' else f(.88,1.10)
    base=f(.73,1.0)
    roundness=f(.035,.19) if s!='rucksack' else f(.21,.43)
    if s=='clutch': roundness=f(.025,.22)
    lip=f(-.07,.05)
    poly=mirrored([(0,lip*h),(top*.34,lip*h*.5),(top*.5-roundness*.26,roundness*.12*h),
        (top*.50,roundness*h),(.50,.68*h),(base*.5+.015,h-roundness*.45*h),
        (base*.50-.03,h-.018*h),(base*.36,h),(0,h)])
    shapes=[poly]
    if handle:
        span=f(.16,.48) if s!='rucksack' else f(.06,.2)
        center=f(-.10,.10); lean=f(-.12,.12)
        t=np.linspace(0,np.pi,19); profile=f(.65,1.85)
        path=list(zip(center+span*np.cos(t)+lean*np.sin(t),.025-(handle+.025)*np.sin(t)**profile))
        shapes.append(stroke(path,f(.013,.037)))
    elif f()<.26:
        shapes.append(stroke([(-.43,0),(-.50,.03),(-.58,h*.36),(-.56,h*.81),(-.46,h*.94)],f(.012,.03)))
    return shapes

def footwear(s,f):
    # Coordinates here are fractions of heel-to-toe length, with altitude up.
    # Draw profiles from the ground upwards, then turn them into screen y.
    spring=f(.018,.083); toe=f(.11,.19); tilt=f(-.04,.05)
    if s in ('trainer','hightop'):
        h=f(.25,.44) if s=='trainer' else f(.43,.60)
        back=h*f(.76,1.01); tongue=f(.51,.69); low=back*f(.53,.84)
        shapes=[[(.00,.055+spring),(.018,spring*.63),(.13,spring*.20),(.33,0),
             (.66,f(-.008,.025)),(.94,.006),(.99,.027),(1,.10),(.97,back*.80),(.94,back),
             (.90,back*1.02),(.80,low),(.72,low*1.03),(tongue,h),(tongue-.04,h*f(.92,1.08)),
             (tongue-.12,h*.81),(.32,.73*toe+.19*h),(.14,toe),(.045,toe*.88)]]
    elif s in ('boot','lowboot','heelboot'):
        if s=='boot': height=f(.52,1.04); heel=f(.05,.19)
        elif s=='lowboot': height=f(.36,.61); heel=f(.025,.14)
        else: height=f(.57,1.10); heel=f(.18,.43)
        front=f(.42,.65); rear=f(.89,1.01); flare=f(-.035,.075); shaftTilt=f(-.11,.15)
        topangle=f(-.07,.07)
        heelFront=f(.68,.90) if s!='heelboot' else f(.72,.94)
        shapes=[[(0,.07+spring),(.02,spring),(.18,.01),(.37,0),(.55,heel*.50),
             (heelFront-.025,heel*.79),(heelFront-.01,.008),(.97,.004),(.99,heel),
             (.975,heel+.09),(rear-.024,heel+.16),(rear+flare+shaftTilt,height+topangle),
             ((front+rear)*.5+shaftTilt,height+topangle*.46-.012),(front-flare+shaftTilt,height),
             (front+.014,max(.24+heel*.6,height*.65)),(front-.045,toe+.10+heel*.4),
             (.37,toe+.042+heel*.22),(.20,toe+.005),(.085,toe),(.025,.81*toe)]]
    else:
        high=s in ('wedge','heelstrap'); slide=s=='slide'
        heel=f(.22,.62) if high else f(.018,.12)
        thick=f(.035,.090) if not s=='wedge' else f(.075,.17)
        fore=f(.24,.44); arch=f(.60,.80); rear=f(.91,.99)
        def sole(x):
            return float(np.interp(x,[0,.13,fore,arch,.94,1],[spring,.008,0,heel*.78,heel,heel*.98]))
        xs=[.00,.045,.15,fore,.51,arch,.92,1.0]
        bottom=[(x,sole(x)) for x in xs]
        if s=='wedge': bottom=[(0,spring),(.18,0),(.58,0),(.97,.006),(1,heel)]
        if s=='heelstrap':
            hf=f(.77,.94); bottom=[(0,spring),(.14,.013),(fore,0),(.55,sole(.55)),(hf-.055,sole(hf-.055)),(hf+.01,.002),(.97,.004),(1,heel)]
        shapes=[bottom+[(x,sole(x)+thick) for x in xs[::-1]]]
        def band(x,lift,width,tilt):
            y=sole(x)+thick
            return [(x-width*.62,y-.021),(x+width*.56,y-.019),
                 (x+width*.57+tilt,y+lift*.75),(x+tilt+width*.34,y+lift),
                 (x+tilt-width*.40,y+lift*1.06),(x+tilt-width*.53,y+lift*.64)]
        bx=f(.13,.37); bh=f(.08,.19) if not slide else f(.13,.23)
        strapWidth=f(.055,.18) if not slide else f(.19,.40)
        shapes.append(band(bx,bh,strapWidth,f(.01,.14)))
        if not slide:
            ah=f(.14,.34) if high else f(.085,.26)
            if f()<.52:
                shapes.append(stroke([(f(.39,.59),sole(.51)+.02),(f(.51,.67),sole(.60)+ah*.7),(.78,sole(.78)+ah),(.91,sole(.92)+ah*.77),(.97,sole(.96)+thick*.55)],f(.017,.040)))
            else:
                shapes.append(band(f(.72,.89),ah,f(.07,.16),f(-.11,.07)))
            if f()<.65:
                shapes.append(stroke([(bx+.02,sole(bx)+bh+.01),(.53,sole(.53)+.10),(.78,sole(.78)+ah*.83)],f(.012,.036)))
            if f()<.36:
                shapes.append(band(f(.38,.55),bh*.83,f(.04,.12),f(.03,.15)))
    return [[(x,-h+tilt*(x-.5)) for x,h in poly] for poly in shapes]

# Supersampled scan conversion. Each set of polygons is centred and scaled
# on its own, just like a product cropped for a catalogue page.
def render(polys):
    points=np.concatenate([np.asarray(p) for p in polys])
    low=points.min(axis=0); high=points.max(axis=0)
    middle=(low+high)*.5; scale=26.6*2/max(high-low)
    buffer=np.zeros((56,56),np.uint8)
    for poly in polys:
        p=(np.array(poly)-middle)*scale+28
        x,y=p.T; xn=np.roll(x,1); yn=np.roll(y,1)
        rows=np.arange(max(0,int(y.min())),min(56,int(y.max())+1)); h=rows[:,None]+.5
        valid=(y>h)!=(yn>h)
        cross=np.where(valid, x+(h-y)*(xn-x)/np.where(y!=yn,yn-y,1.),1.e3)
        cross.sort(axis=1)
        # Most garment pieces are y-monotone, needing only one pair.
        for k in range(0,cross.shape[1]-1,2):
            used=cross[:,k+1]<999
            if not used.any(): break
            scan=np.arange(56)+.5
            buffer[rows[used]] |= ((scan>=cross[used,k,None]) & (scan<cross[used,k+1,None]))
    return buffer.reshape(28,2,28,2).mean(axis=(1,3),dtype=np.float32)

def makebank():
    g={}
    exec(geometry,g)
    catalog={}
    for fam,(s,category,count) in enumerate(styles):
        vectors=[]
        for start in range(0,count,64):
            raw=[]
            for index in range(start,min(count,start+64)):
                f=Dimensions(index,fam)
                draw=(trousers if category==1 else bag if category==8 else footwear if category in (5,7,9) else garment)
                if s.endswith('front'): draw=frontview
                raw.append(render(draw(s,f)))
            masks=np.stack(raw)
            xx,yy,ar=g['framing'](masks)
            a=g['grid'](masks,xx,yy)
            vectors.append(g['describe'](a,ar))
        catalog[s]=np.concatenate(vectors,axis=0)
    return catalog

recognition = '''
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
'''

if __name__=='__main__':
    dest=Path(sys.argv[1]); dest.mkdir(parents=True,exist_ok=True)
    (dest/'solution.py').write_text(geometry+recognition)
    np.savez(dest/'constants.npz',**makebank())
