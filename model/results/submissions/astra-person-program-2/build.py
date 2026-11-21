"""Build the hand-specified person detector. No external inputs are used."""
from pathlib import Path
import sys

source = '''import numpy as np


def unit(z):
    return np.minimum(1., np.maximum(0., z))


def table(x):
    return np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0))).reshape(-1, x.shape[2])


def grid(w, h):
    stride = max(2, int(w*.17))
    x = np.arange(0, 96, stride, dtype=np.float32) + (stride-1)*.5
    y = x[(x > .30*h-1) & (x < 96-.49*h)]
    x = x[(x > .34*w-1) & (x < 96-.34*w)]
    x, y = np.meshgrid(x, y)
    return x.ravel(), y.ravel()


# A rectangle is specified relative to a head box, origin at its top centre.
# Store its four integral-image addresses on the search grid.
def plan(w, h, x, y, dx, dy, sx, sy):
    cx = x + dx*w + .5
    cy = y + (dy-.5)*h + .5
    x0 = np.clip(np.rint(cx-max(1., w*sx)*.5).astype(np.int32), 0, 95)
    x1 = np.maximum(x0+1, np.clip(np.rint(cx+max(1., w*sx)*.5).astype(np.int32), 1, 96))
    y0 = np.clip(np.rint(cy-max(1., h*sy)*.5).astype(np.int32), 0, 95)
    y1 = np.maximum(y0+1, np.clip(np.rint(cy+max(1., h*sy)*.5).astype(np.int32), 1, 96))
    norm = (1./((x1-x0)*(y1-y0))).astype(np.float32)
    return y1*97+x1, y1*97+x0, y0*97+x1, y0*97+x0, norm


def boxes(w, h):
    x, y = grid(w,h)
    layout = [(0,.64,.55,.48), (0,.19,.62,.27), (0,-.20,.65,.28),
              (-.74,.54,.40,.50), (.74,.54,.40,.50), (0,1.65,.96,.68),
              (-.85,1.70,.52,.60), (.85,1.70,.52,.60),
              (-1.72,1.75,.60,.75), (1.72,1.75,.60,.75),
              (0,2.80,1.20,.65), (-1.64,2.85,.62,.80), (1.64,2.85,.62,.80),
              (0,4.30,.90,.75), (-1.55,4.30,.75,.85), (1.55,4.30,.75,.85),
              (-.25,.49,.26,.14), (.25,.49,.26,.14), (0,.53,.15,.19),
              (-.23,.64,.28,.15),(.23,.64,.28,.15),(0,.33,.50,.17),
              (0,.81,.38,.10),(0,.91,.34,.11),(0,.72,.29,.10),
              (-.23,.32,.27,.16),(.23,.32,.27,.16),
              (0,.54,1.65,1.46),(-.91,.98,.49,.25),(.91,.98,.49,.25),
              (-.88,1.27,.47,.26),(.88,1.27,.47,.26)]
    plans = [plan(w,h,x,y,*params) for params in layout]
    return w,h,x,y,plans


search = [boxes(w,w*r) for w in (3.,4.5,6.5,9.5,14.,20.,29.,42.,62.,83.) for r in (1.1,1.82) if w*r<122]


def readbox(t, p):
    a,b,c,d,n = p
    return (t[a]-t[b]-t[c]+t[d])*n[:,None]


def dist(a,b):
    return abs(a[:,0]-b[:,0]) + .72*abs(a[:,1]-b[:,1]) + .56*abs(a[:,2]-b[:,2])


def softmin(a,b):
    return .80*np.minimum(a,b)+.20*np.maximum(a,b)


def agree(a,b):
    return .65*np.minimum(a,b)+.35*np.maximum(a,b)


def measure(t, spec, monochrome):
    w,h,x,y,p = spec
    face,hair,above,left,right,chest,sl,sr,bl,br,waist,wl,wr,leg,ll,lr = [readbox(t,k) for k in p[:16]]
    region=readbox(t,p[27]); sigma=np.sqrt(np.maximum(0., region[:,4]-region[:,0]**2))
    scale = .095+.72*sigma
    # Contrast is two-sided: a stripe from a window or a branch is not an isolated head.
    sideL=dist(face,left); sideR=dist(face,right)
    cap=unit(dist(hair,above)/scale)
    sides=unit(softmin(sideL,sideR)/scale)
    sidehair=unit(softmin(dist(hair,left),dist(hair,right))/scale)
    bgmatch=unit(1.-dist(left,right)/(.13+softmin(sideL,sideR)))
    skin=unit(face[:,3]); backskin=.40*np.minimum(left[:,3],right[:,3])+.60*np.maximum(left[:,3],right[:,3])
    nonskin=1.-unit(.44*backskin+.29*chest[:,3]+.14*above[:,3]+.13*hair[:,3])
    skinshape=unit((skin-.55*backskin-.18*chest[:,3]-.13*hair[:,3])/.69)
    darkhair=unit((face[:,0]-hair[:,0]+.018)/(.084+.38*sigma))
    capcolor=unit(dist(hair,face)/scale)
    # Clothing that widens at the shoulders, with contrasting background on both sides.
    shoulderL = unit((dist(sl,left)-.52*dist(sl,chest)) / scale)
    shoulderR = unit((dist(sr,right)-.52*dist(sr,chest)) / scale)
    shoulder=agree(shoulderL,shoulderR)
    torso=unit(softmin(dist(chest,bl),dist(chest,br))/scale)
    innerL=unit((dist(sl,bl)-.70*dist(sl,chest))/scale)
    innerR=unit((dist(sr,br)-.70*dist(sr,chest))/scale)
    fill=agree(innerL,innerR)
    below = dist(chest,face)
    collar=unit(below/scale)
    # Keep outline evidence distinct from just a change in colour at the horizon.
    taper=agree(unit((sideL-.44*dist(left,bl))/scale),unit((sideR-.44*dist(right,br))/scale))
    mid=unit(softmin(dist(waist,wl),dist(waist,wr))/scale)
    bottom=unit(softmin(dist(leg,ll),dist(leg,lr))/scale)
    mvis=unit((96-y)/h-1.8); bvis=unit((96-y)/h-3.15)
    groundmatch = unit(1.-dist(ll,lr)/(.11+softmin(dist(leg,ll),dist(leg,lr))))
    longbody=mid*mvis*.50+bottom*(.30+.20*groundmatch)*bvis
    body=.22*torso+.20*fill+.24*shoulder+.22*longbody+.12*taper
    body=body*unit((96-y)/h-.52)
    # The slope from neck to shoulder ought to separate shirt from backdrop.
    ul,ur,dl,dr=[readbox(t,k) for k in p[28:32]]
    slope=agree(unit((dist(dl,ul)-.55*dist(dl,chest))/scale),unit((dist(dr,ur)-.55*dist(dr,chest))/scale))
    headshape=.34*sides+.19*sidehair+.18*cap+.13*bgmatch+.16*taper
    # Eye sockets should occur as a pair. The bridge, brow and cheeks provide
    # three independent light references, so a single horizontal edge is insufficient.
    eyes=np.zeros_like(skin)
    mouth=np.zeros_like(skin)
    symmetry=np.zeros_like(skin)
    if w>=6.5:
        el,er,nose,cl,cr,brow,lip,chin,phil,fl,fr=[readbox(t,k) for k in p[16:27]]
        div=.035+.45*sigma
        l=unit((cl[:,0]-el[:,0])/div); r=unit((cr[:,0]-er[:,0])/div)
        browpair=softmin(unit((fl[:,0]-el[:,0])/div),unit((fr[:,0]-er[:,0])/div))
        pair=softmin(l,r)
        bridge=unit((nose[:,0]-np.maximum(el[:,0],er[:,0]))/div)
        symerr=.40*dist(el,er)+.60*dist(cl,cr)
        symmetry=unit(1.-symerr/(.085+.65*sigma))
        eyes=pair*(.35+.27*browpair+.23*bridge+.15*symmetry)
        mouth=unit((softmin(chin[:,0]-lip[:,0],phil[:,0]-lip[:,0])+.24*(lip[:,1]-chin[:,1]))/div)
    loc = -.095*unit((y-49)/37)-.045*unit((abs(x-47.5)-27)/20)
    # Each route below is a conjunction, expressed on a shared margin scale.
    chrom = 1.03*skin+.92*skinshape+.52*nonskin+.48*sides+.28*cap+.29*darkhair+.18*collar+.47*body+.67*eyes+.20*mouth
    chrom -= 3.60
    anatomical=1.85*headshape+1.76*body+.42*slope+.26*collar+.23*skin+.27*capcolor+.38*eyes-3.86
    portrait=1.75*eyes+.63*mouth+.33*symmetry+.45*headshape+.27*darkhair+.47*skin+.32*skinshape+.15*body-3.21
    if w<6.5:
        portrait=portrait-2.
    confidence=np.maximum(chrom,np.maximum(anatomical,portrait)) + loc
    if w<6.5:
        confidence-=.13
    if monochrome:
        confidence+=.12
    # Pool in wide spatial bins; a crowd of separate moderate detections is
    # stronger evidence than one clutter edge scoring highly at many scales.
    tile = np.minimum(2,(x/32).astype(np.int32))+3*np.minimum(1,(y/48).astype(np.int32))
    votes=np.full(6,-4.,np.float32)
    np.maximum.at(votes,tile,confidence)
    # Repeated face-and-shoulder patterns on the same scale. Use widely
    # separated centres (not adjacent responses to one head); allow perspective.
    if w<=20:
        pick=int(np.argmax(confidence)); dx=abs(x-x[pick]); dy=abs(y-y[pick]); spacing=1.5*w
        same=(dx>spacing)&(dy<1.5*h+.18*dx)&(dy<23.)
        others=np.where(same,confidence,-4.)
        k=int(np.argmax(others)); two=others[k]
        three=np.max(np.where(same&(abs(x-x[k])>spacing),confidence,-4.))
        repeat=.23*unit((two+.27)/.70)+.23*unit((three+.23)/.66)
        votes[tile[pick]]+=repeat
    return votes


# Strip geometry for a few articulated, clothed silhouettes. Opposing
# sides and different parts are verified separately, not just as a sum of edges.
def outlines():
    bank=[]
    for height in (20,30,44,63,90,128):
        for aspect in (.215,.315,.44):
            width=height*aspect
            step=max(2,round(width/7.5))
            xx,yy=np.meshgrid(np.arange(step*.5,96,step),np.arange(step*.5,96-height*.42,step))
            xpos=xx.ravel(); ypos=yy.ravel()
            poses=((0.,.32),(-.42,.67),(.42,.67)) if .30<aspect<.34 else ((0.,.32),)
            for lean,spread in poses:
                rings=[]
                for level,radius,hh in ((.084,.198,.093),(.195,.31,.046),(.30,.50,.143),(.46,.42,.135),(.605,.305,.12),(.747,.30+spread*.18,.14),(.902,.27+spread*.37,.13)):
                    ctr=xpos+(level-.13)*lean*width
                    # Outside strip, inside strip, opposite inside and outside.
                    gap=max(.033, .48/width)
                    thick=.18 if level>.14 else .12
                    outside=.24 if level>.14 else .18
                    rings.append([plan(width,height,ctr,ypos+.5*height,side*(radius+off),level,sw,hh) for side,off,sw in ((-1,gap+outside*.5,outside),(-1,-thick*.5,thick),(1,-thick*.5,thick),(1,gap+outside*.5,outside))])
                bank.append((height,width,xpos,ypos,lean,rings))
    return bank


figures=outlines()


def contour(t, geom):
    height,width,xpos,ypos,lean,rings=geom
    strengths=[]
    middles=[]
    outfits=[]
    separations=[]
    for j,ring in enumerate(rings):
        outerl,innerl,innerr,outerr=[readbox(t,k) for k in ring]
        dl=dist(outerl,innerl); dr=dist(outerr,innerr)
        inside=dist(innerl,innerr); outside=dist(outerl,outerr)
        texture=np.sqrt(np.maximum(0.,(innerl[:,4]-innerl[:,0]**2+innerr[:,4]-innerr[:,0]**2)*.5))
        divider=.11+.60*texture
        # If BOTH half-contours see the same transition from inside to
        # outside, it is more likely that this is an occluding object.
        pl=unit((dl-.18*inside)/divider); pr=unit((dr-.18*inside)/divider)
        sep=softmin(dl,dr)
        interior=unit(1.-inside/(.07+.8*sep)); background=unit(1.-outside/(.08+sep))
        strengths.append(agree(pl,pr)*(.53+.29*interior+.18*background))
        middles.append((innerl+innerr)*.5)
        outfits.append(interior)
        separations.append(sep)
    head,neck,shoulder,rib,hip,knee,ankle=strengths
    face,collar,shirt,hem,belt,trouser,foot=middles
    above=unit((96-ypos)/height-.56)
    shirtlike=unit(1.-dist(shirt,hem)/(.10+separations[2]))
    shirtcolor=unit(dist(face,shirt)/.16)
    shirtedge=unit(dist(trouser,shirt)/.17)
    skin=unit(face[:,3]-.38*shirt[:,3]); continuity=.58*shirtlike+.42*outfits[2]
    # Consistency of a whole articulated outline matters more than one very
    # strong segment; missing feet are allowed near the bottom of the frame.
    upper=.18*head+.08*neck+.26*shoulder+.20*rib+.17*hip+.11*softmin(head,shoulder)
    lower=(.58*knee+.42*ankle)*unit((96-ypos)/height+.22)
    score=3.8*upper+.94*lower+.41*continuity+.37*skin+.17*shirtcolor+.14*shirtedge+.14*softmin(shoulder,rib)-4.09
    pos=unit((abs(xpos-47.5)-24.)/23.)
    score-=.10*pos+.07*unit((ypos-47.)/28.)+.17*unit((22.-height)/7.)
    return score.max()


def facepatch(t):
    # Connected skin islands are candidate head proposals. This branch works
    # with pixel runs rather than a dense head grid and handles tilted profiles.
    y,rg,gb,skin=t
    smooth=np.pad(skin,1,mode='edge')
    smooth=.40*skin+.15*(smooth[1:-1,:-2]+smooth[1:-1,2:]+smooth[:-2,1:-1]+smooth[2:,1:-1])
    mask=smooth>.60
    # Union overlapping horizontal runs in consecutive rows (8-connected).
    edge=np.diff(np.pad(mask.astype(np.int8),((0,0),(1,1))),axis=1)
    rows,lo=np.nonzero(edge>0); hi=np.nonzero(edge<0)[1]
    if len(lo)==0:
        return -4.
    lo=lo.tolist(); hi=hi.tolist(); rows=rows.tolist()
    parent=list(range(len(lo)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    start=0
    for i in range(len(lo)):
        while start<i and (rows[start]<rows[i]-1 or (rows[start]==rows[i]-1 and hi[start]<lo[i])):
            start+=1
        k=start
        while k<i and rows[k]<rows[i] and lo[k]<=hi[i]:
            a=root(k); b=root(i)
            if a!=b:
                parent[b]=a
            k+=1
    ids=np.array([root(i) for i in range(len(lo))]); lo=np.array(lo); hi=np.array(hi); rows=np.array(rows); length=hi-lo
    n=len(lo)
    area=np.bincount(ids,weights=length,minlength=n)
    xl=np.full(n,96); xr=np.zeros(n,dtype=np.int32); yt=xl.copy(); yb=xr.copy()
    np.minimum.at(xl,ids,lo); np.maximum.at(xr,ids,hi); np.minimum.at(yt,ids,rows); np.maximum.at(yb,ids,rows+1)
    widths=xr-xl; heights=yb-yt
    ids2=np.flatnonzero((area>=6)&(widths>=2)&(heights>=3)&(widths<85)&(heights<91)&(area<.92*96*96))
    scores=[]
    for i in ids2:
        w=widths[i]; h=heights[i]; left=xl[i]; right=xr[i]; top=yt[i]; bot=yb[i]
        solidity=area[i]/(w*h); aspect=w/h
        if solidity<.29 or aspect<.18 or aspect>2.35:
            continue
        roi=(ids==i)
        # Average runs by height: an oval or chin tapers toward both ends.
        yy=rows[roi]-top
        line=np.bincount(yy, weights=length[roi],minlength=h)/w
        cut=max(1,round(h*.25)); waist=line[cut:h-cut].mean()
        taper=unit((waist-.5*(line[:cut].mean()+line[-cut:].mean())+.03)/.36)
        oval=unit((solidity-.28)/.37)*unit((1.13-solidity)/.30)
        ratio=unit((aspect-.20)/.29)*unit((2.05-aspect)/.83)
        skinmean=skin[top:bot,left:right].mean(); darkcap=y[max(0,top-max(1,h//3)):max(1,top),left:right].mean()
        patchy=y[top:bot,left:right]; hair=unit((patchy.mean()-darkcap+.01)/.15)
        sd=patchy.std()
        detail=unit((sd-.025)/.085)*unit((.32-sd)/.11)
        # Check for neutral or differently coloured surroundings and dark hair.
        yd=y[max(0,top-h//3):min(96,bot+h//3+1),max(0,left-w//2):min(96,right+w//2+1)]
        bg=skin[max(0,top-h//3):min(96,bot+h//3+1),max(0,left-w//2):min(96,right+w//2+1)]
        surround=(bg.sum()-skinmean*w*h)/max(1,bg.size-w*h)
        stand=unit((skinmean-surround-.11)/.53)
        pos=(top+bot)*.5
        # Exposed limbs below the head can corroborate an isolated small oval.
        ext=skin[min(95,bot+max(1,h//2)):min(96,bot+max(3,4*h)),max(0,left-w):min(96,right+w)]
        limbs=unit((ext.mean()-.035)/.13)*unit((.72-ext.mean())/.3)
        score=.86*ratio+.62*oval+.58*taper+.73*stand+.80*hair+.24*detail+.36*limbs+.24*skinmean-3.97
        score-=.13*unit((pos-53.)/28.)+.17*unit((5.-w)/3.)+.16*unit((4.-h)/2.)
        scores.append(score)
    if not scores:
        return -4.
    scores.sort()
    if len(scores)>1:
        return scores[-1]+.20*unit((scores[-2]+.35)/.70)
    return scores[-1]


def scene(y,rg,gb,skin):
    # Supporting evidence from the playing surface and vivid manufactured clothing.
    cool=unit((-rg-.015)/.12)*unit((-gb+.05)/.11)
    leafy=unit((-rg+.020)/.065)*unit((gb-.03)/.08)
    grass=unit((gb-.032)/.095)*unit((gb-rg-.035)/.095)
    dark=unit((.21-y)/.16)
    magenta=unit((rg-.10)/.20)*unit((.06-gb)/.15)
    red=unit((rg-.16)/.22)*unit((rg-1.4*gb-.10)/.20)
    royal=unit((-.11-gb)/.18)*unit((.14-abs(rg))/.12)
    colorful=np.maximum(royal,np.maximum(red,magenta))
    bright=colorful[15:82,8:88].mean()
    field=grass[49:].mean()
    water=cool[42:].mean()
    snow=unit((y[50:]-.64)/.22)*unit((.11-abs(rg[50:])-abs(gb[50:]))/.09)
    surface=unit((max(field,water,snow.mean())-.15)/.47)
    context=.16*surface+.23*unit((bright-.014)/.085)
    # Faces and exposed arms often supply a small high-quality skin area even
    # when their position or tilt defeats the upright geometry.
    sk=skin[6:89,4:92]; avg=sk.mean()
    context+=.16*unit((avg-.025)/.10)-.15*unit((avg-.36)/.25)
    # Structure tensor at a coarse scale: architecture has long uninterrupted
    # horizontal/vertical lines, unlike the contours of a foreground figure.
    dx=(y[1:-1,2:]-y[1:-1,:-2]); dy=(y[2:,1:-1]-y[:-2,1:-1])
    ax=abs(dx); ay=abs(dy); energy=ax+ay
    line=((ax>3.7*ay)|(ay>3.7*ax))
    rect=(energy*line).sum()/(energy.sum()+.001)
    context-=.26*unit((rect-.48)/.22)
    # Aggregate semantics across the picture: grass beside reddish infield
    # clay or uniform fencing, blue water with foam, snow against trees, and
    # coloured jerseys over the playing surface. A blue sky alone is not water.
    def lower(m):
        return .52*m[48:].mean()+.48*m[64:].mean()
    fieldlike=unit((field-.19)/.36)
    clay=unit((rg-.03)/.10)*unit((gb-.022)/.088)*unit((1.45*gb+.06-rg)/.10)
    boundary=grass[:52].mean(); infield=clay[42:].mean()
    turf=fieldlike*unit((.054-np.mean(abs(grass[54:,1:]-grass[54:,:-1])))/.043)
    fieldcolors=unit((max(min(infield,grass[18:67].mean()),min(clay[18:67].mean(),field))-.035)/.23)
    context+=.28*fieldcolors+.16*turf
    foam=unit((y-.64)/.20)*unit((.12-abs(rg)-abs(gb))/.095)
    ocean=unit((lower(cool)-.23)/.43)*unit((lower(foam)-.025)/.16)
    cold=snow.mean()
    ski=unit((cold-.32)/.43)*unit((bright+.7*dark[15:70].mean()-.021)/.092)
    context+=.30*ocean+.27*ski
    # A small vivid garment with a different background is common for a
    # player, surfer or skier. Do not reward an entirely red or blue field.
    colored=np.maximum(magenta,np.maximum(red,royal))
    block=colored.reshape(8,12,8,12).mean(axis=(1,3))
    peak=block[1:7,1:7].max(); diversity=unit((peak-colored.mean()-.18)/.44)
    context+=.14*surface*diversity+.085*diversity
    # A characteristic distribution of skin is compact highlights, not a
    # field of warm brown. A minority of small bright patches against other
    # colours is stronger evidence than the same colour covering the image.
    blocks=skin.reshape(12,8,12,8).mean(axis=(1,3))
    focus=blocks[1:10,1:-1].max()
    context+=.18*unit((focus-avg-.24)/.44)*unit((.44-avg)/.26)
    context-=.14*unit((avg-.25)/.34)
    # Texture orientation and scene composition. Horizontal bands of water
    # or pitch with a locally busy central region are more credible than a
    # uniformly busy canopy of grass, or the regular grid of a building.
    edg=(ax+ay).reshape(47,2,47,2).mean(axis=(1,3))
    lateral=.5*(edg[9:39,:8].mean()+edg[9:39,-8:].mean())
    central=edg[9:39,13:34].mean()
    focusedge=unit((central-lateral*.85)/(.06+lateral))
    context+=.12*surface*focusedge
    # Black-and-white reportage lacks chromatic cues; rely more on the shape path.
    monochrome=abs(rg).mean()+abs(gb).mean()<.007
    if monochrome:
        context+=.29
    return context


def single(im):
    r,g,b=im
    y=.299*r+.587*g+.114*b
    rg=r-g; gb=g-b
    # A mild illuminant correction, deliberately bounded so grass and sky do
    # not shift neutral pixels all the way to flesh colour.
    # Concave skin-colour locus in the red-green / red-blue plane.
    # Combine a daylight model with one modest scene-white correction. In
    # tungsten light, merely checking that R > G > B includes much of the room.
    white=unit((y-.37)/.41)*unit((.54-abs(rg)-abs(gb))/.31)+.018
    ur=np.clip((rg*white).sum()/white.sum(),-.12,.16)
    vr=np.clip((gb*white).sum()/white.sum(),-.13,.15)
    qr=np.clip(rg.mean()*.24+ur*.16,-.035,.046)
    qv=np.clip(gb.mean()*.24+vr*.16,-.035,.046)
    def flesh(u,v):
        rb=u+v
        chrom=unit((rb-.014)/(.047+.066*y))
        hue=u/(.006+np.maximum(0.,rb))
        chrom*=unit((hue-.34)/.21)*unit((1.035-hue)/.17)
        saturation=rb/(.085+y)
        chrom*=unit((1.06-saturation)/.32)*unit((.81-u/(.12+y))/.23)
        # A preference, not a hard exclusion, for soft rather than vivid colours.
        return chrom*(.74+.26*unit((.95-saturation)/.43))*unit((y+.015)/.095)
    skin=np.maximum(flesh(rg-qr,gb-qv),.86*flesh(rg-ur*.75,gb-vr*.75))
    mono=abs(rg).mean()+abs(gb).mean()<.007
    t=table(np.stack((y,rg,gb,skin,y*y),axis=2))
    evidence=np.full(6,-4.,np.float32)
    for spec in search:
        evidence=np.maximum(evidence,measure(t,spec,mono))
    ordered=np.sort(evidence)
    support=.20*unit((ordered[-2]+.33)/.68)+.09*unit((ordered[-3]+.24)/.64)
    headvote=ordered[-1]+support
    shape=max(contour(t,geom) for geom in figures)
    patch=facepatch((y,rg,gb,skin))
    # The three searches have different failure modes: let one strong result
    # dominate, but reward agreement of otherwise inconclusive routes.
    votes=sorted((headvote,shape,patch))
    return .73*votes[2]+.22*votes[1]+.05*votes[0]+scene(y,rg,gb,skin)>.44


def predict(x):
    return np.array([int(single(im)) for im in x], dtype=np.int32)
'''

folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
(folder / 'solution.py').write_text(source)
