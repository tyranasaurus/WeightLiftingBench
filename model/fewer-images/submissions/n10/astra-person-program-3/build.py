"""Data-free builder for a hand-specified sliding part detector."""
import sys
from pathlib import Path

program = '''import numpy as np
import math


def unit(x):
    return np.minimum(1., np.maximum(0., x))


def many(*args):
    a = args[0]
    for b in args[1:]:
        a = np.maximum(a, b)
    return a


def soft_skin(r, g, b):
    # Work in opponent coordinates. Scale the chroma tolerance with brightness;
    # allow a range along (rather than across) the melanated-skin locus.
    lum = .30*r + .58*g + .12*b
    exposure = .32 + .68*lum
    warm = (r-b)/exposure
    red = (r-g)/exposure
    hue = (red - .60*warm)/(.046 + .13*np.maximum(warm, 0.))
    hue = np.where(hue < 0, 1.18*hue, .84*hue)
    value = np.exp(-.5*hue*hue)
    value *= unit((warm-.030)/.155) * unit((.98-warm)/.25)
    value *= unit((red-.014)/.072) * unit((r-.035)/.17)
    return value


def photo_maps(pic):
    r,g,b = pic
    lum = .30*r + .58*g + .12*b
    # Limited illuminant compensation, retaining the uncorrected cue as well.
    rm,gm,bm = r.mean(),g.mean(),b.mean()
    neutral = (rm+gm+bm)/3.
    s = soft_skin(r,g,b)
    corr = []
    for v,m in ((r,rm),(g,gm),(b,bm)):
        gain = np.clip(((neutral+.12)/(m+.12))**.62,.84,1.18)
        corr.append(v*gain)
    skin = np.maximum(.97*s, .85*soft_skin(*corr))
    # Lower confidence in skin colour seen throughout an image (colour cast,
    # earth, food, furniture). A close face can still use internal detail.
    skin *= 1. - .22*unit((skin.mean()-.16)/.40)
    gray = unit(1.-np.mean(np.maximum.reduce((r,g,b))-np.minimum.reduce((r,g,b)))/.04)
    return (lum, .8*(r-g), .7*(b-g), skin), gray


def summed(a):
    out = np.zeros((len(a),97,97),dtype=np.float64)
    out[:,1:,1:] = a.cumsum(axis=2,dtype=np.float64).cumsum(axis=1)
    # Two edge-padded tables handle the two clipping conventions on a box
    # corner. Padding lets the densely sampled scales read contiguous slices.
    p=np.pad(out[:,:96],((0,0),(48,192),(0,0)),mode='edge')
    q=np.pad(out[:,1:],((0,0),(48,192),(0,0)),mode='edge')
    return out,p,q


def integrate(maps):
    lum, rg, bg, skin = maps
    return summed(np.array([lum,rg,bg,skin,lum*lum+.7*(rg*rg+bg*bg)]))


def make_grid(w,h):
    step = max(1,int(round(min(w,h)*.17)))
    yc = np.arange(1,96,step).reshape(-1,1)
    xc = np.arange(1,96,step).reshape(1,-1)
    return yc,xc


# Above a minimum image size, column and row offsets are identical for
# every photo. Store only geometry, never values from an input image.
geometry={}


def window(ii, yc, xc, dx, dy, wide, high, channels):
    wx,hy = max(1,int(round(wide))),max(1,int(round(high)))
    c,r = int(round(dx-.5*(wx-1))),int(round(dy-.5*(hy-1)))
    key=(xc.size,wx,hy,c,r)
    if key not in geometry:
        x0,x1 = np.clip(xc+c,0,95),np.clip(xc+c+wx,1,96)
        y0,y1 = np.clip(yc+r,0,95),np.clip(yc+r+hy,1,96)
        step=int(xc[0,1]-xc[0,0])
        # In the interior all blocks advance by fixed strides, which allows
        # fast slices instead of independent pixel indexing.
        def runs(a,b):
            a=a.reshape(-1)
            b=b.reshape(-1)
            ks=[0]
            for j in range(1,len(a)-1):
                if a[j]-a[j-1]!=a[j+1]-a[j] or b[j]-b[j-1]!=b[j+1]-b[j]:
                    ks.append(j)
            ks.append(len(a))
            res=[]
            for j,k in zip(ks,ks[1:]):
                mid=slice(j,k)
                def section(v):
                    stride=step if k-j>1 and v[j+1]!=v[j] else 0
                    return slice(int(v[j]),int(v[k-1]+1),max(1,stride))
                res.append((mid,section(a),section(b)))
            return res
        ystart=49+r
        yend=48+r+hy
        nrows=step*len(yc)
        yslices=(slice(ystart,ystart+nrows,step),slice(yend,yend+nrows,step))
        geom=(yslices,runs(x0,x1),1./((y1-y0)*(x1-x0)),(y1,x1,y0,x0))
        geometry[key]=geom
    yr,xr,recip,inds=geometry[key]
    y1,x1,y0,x0=inds
    if xc.size>=48:
        rows=ii[2][channels,yr[1],:]-ii[1][channels,yr[0],:]
    else:
        vals=ii[0][channels]
        rows=(vals[...,y1[:,0],:]-vals[...,y0[:,0],:])
    if xc.size>=48:
        dest=np.empty(rows.shape[:-1]+(xc.size,),dtype=np.float32)
        for mid,a,b in xr:
            dest[...,mid]=(rows[...,b]-rows[...,a])*recip[:,mid]
        return dest
    return ((rows[...,x1[0]]-rows[...,x0[0]])*recip).astype(np.float32)


def distance(a,b):
    d = a[:3]-b[:3]
    return np.sqrt(d[0]*d[0]+.7*(d[1]*d[1]+d[2]*d[2]))


def scatter(a):
    return np.sqrt(np.maximum(a[4] - a[0]*a[0]-.7*(a[1]*a[1]+a[2]*a[2]),0.))


def gentler(x):
    return x/(.085+x)


def multilateral(a,b):
    # Both sides should support a part, but one can be occluded.
    return .57*np.minimum(a,b)+.43*(a+b)*.5


def mixer(scores):
    # Mix the best complete explanation with evidence from the runner-up.
    hi = np.maximum(scores[0],scores[1])
    lo = np.minimum(scores[0],scores[1])
    for v in scores[2:]:
        lo = np.maximum(lo,np.minimum(hi,v))
        hi = np.maximum(hi,v)
    return hi + .12*np.clip(lo+.50,-.50,.70)


def scan(ii, h, w, gray):
    yc,xc=make_grid(w,h)
    def box(x,y,a,b,channels=slice(None)):
        return window(ii,yc,xc,x*w,y*h,a*w,b*h,channels)
    def col(x,y,a,b):
        return box(x,y,a,b,slice(0,3))
    def sk(x,y,a,b):
        return box(x,y,a,b,3)
    def light(x,y,a,b):
        return box(x,y,a,b,0)
    # Anchor is the center of an exposed face. Use a slightly smaller inner
    # block to avoid conflating the cheeks with hair, ears, and background.
    face = box(0,.02,.64,.70)
    fl = col(-.22,.12,.27,.33)
    fr = col(.22,.12,.27,.33)
    left = box(-.82,0,.43,.68)
    right = box(.82,0,.43,.68)
    top = box(0,-.78,.67,.34)
    lower = box(0,.87,.66,.42)
    crown = col(0,-.43,.64,.30)
    chin = col(0,.35,.45,.22)
    sideL,sideR = gentler(distance(fl,left)),gentler(distance(fr,right))
    sides = multilateral(sideL,sideR)
    below = gentler(distance(chin, lower))
    above = gentler(distance(face, top))
    topEdge = gentler(distance(crown,top))
    noise = scatter(face)
    fskin = face[3]
    skside = fskin - (.65*(left[3]+right[3])*.5+.35*np.maximum(left[3],right[3]))
    sklim = fskin - .5*(top[3]+lower[3])
    isolation = .67*skside+.33*sklim
    sym = 1.-unit(distance(fl,fr)/(.10+.55*noise))
    hair = unit((.5*(fl[0]+fr[0])-crown[0]+.005)/(.12+.30*noise))
    sidehair = unit((face[0] - np.maximum(left[0],right[0])+.005)/.19)
    hair = np.maximum(hair,.75*sidehair)

    # The torso lies below, with relatively coherent fabric and two side
    # boundaries. Test also at a small lateral offset for bent or tilted poses.
    def torso(off):
        core = box(off,1.75,1.27,1.06)
        lobe = col(off-.65,1.65,.42,.70)
        robe = col(off+.65,1.65,.42,.70)
        core2 = col(off,2.65,1.22,.66)
        lbg = col(off-1.54,1.78,.50,.91)
        rbg = col(off+1.54,1.78,.50,.91)
        surface = (distance(core,lobe)+distance(core,robe))*.34 + distance(core,core2)*.32
        distinct = multilateral(distance(lobe,lbg),distance(robe,rbg))
        incoat = scatter(core)
        body = unit((distinct-.43*surface-.24*incoat-.008)/.16)
        edgeL = gentler(distance(lobe,left))
        edgeR = gentler(distance(robe,right))
        spread = multilateral(edgeL,edgeR)
        shoulder = unit((spread-.6*gentler(surface)-.13)/.52)
        neck = gentler(distance(face,core))
        samebg = 1.-gentler(.5*(distance(lbg,left)+distance(rbg,right)))
        # Skin echoes at plausible hands and arms: bilateral or one clear arm,
        # absent from immediate surroundings and a different colour from shirt.
        arm1 = sk(off-1.10,2.20,.60,.98)
        arm2 = sk(off+1.10,2.20,.60,.98)
        arm = (.68*np.maximum(arm1,arm2)+.32*np.minimum(arm1,arm2)-.47*core[3]-.23*(left[3]+right[3])*.5)
        arm = unit((arm-.10)/.60)
        return body,shoulder,neck,arm,samebg
    body,shoulder,neck,arm,bg = torso(0.)
    # A lower garment may differ from a top; two legs can merge in the tiny
    # image. Look for coherent lower-body colour separated from both flanks.
    pants = box(0,4.30,1.05,1.15)
    pantL=col(-1.26,4.3,.55,1.10)
    pantR=col(1.26,4.3,.55,1.10)
    legs = unit((multilateral(distance(pants,pantL),distance(pants,pantR))-.42*scatter(pants)-.03)/.16)
    legs = legs*unit((96-yc-3.5*h)/h)
    bodyview = unit((96-yc-.75*h)/h)
    if h>26:
        bodyview = np.maximum(.15,bodyview)
    body = body*bodyview
    shoulder = shoulder*bodyview
    arm = arm*bodyview
    # Head and neck colour + shape. The max is local; an entire photograph
    # being a warm hue is not equivalent to a bounded patch of skin.
    skinhead = (1.65*fskin+1.08*isolation+1.00*sides+.31*above+.26*below+
                .36*hair+.18*sym+.55*body+.34*shoulder+.25*neck*bodyview+
                .32*arm+.13*legs - 3.47)
    # Tiny faces don't have to resolve the eyes, but a head-like highlight
    # only two pixels across needs stronger corroboration.
    skinhead -= .26*unit((4.-w)/2.) + .08*unit((7.-h)/3.)
    # Monochrome / back of head: a small head silhouette over a separate,
    # wider torso and (when visible) smaller elongated lower body.
    headhom = unit(1.-noise/(.075+.10*sides))
    back = (1.08*sides+.48*topEdge+.25*above+.17*sym+.26*headhom+
            .93*body+.77*shoulder+.42*neck*bodyview+.41*legs+.20*bg+
            .59*fskin+.40*arm+.18*hair-3.73+.11*gray)
    back -= .29*unit((4.-w)/2.)+.16*unit((h-25.)/16.)

    # Relative luminance layout of a frontal face. At sufficient resolution,
    # require the eye line to be darker *on both sides*, bounded by forehead,
    # cheeks and the nose bridge, with a distinct lower lip/chin structure.
    if w>=3.6 and h>=5:
        eyeL = light(-.233,-.146,.27,.14)
        eyeR = light(.233,-.146,.27,.14)
        browL = light(-.205,-.35,.29,.16)
        browR = light(.205,-.35,.29,.16)
        ckL = light(-.23,.075,.30,.18)
        ckR = light(.23,.075,.30,.18)
        bridge = light(0,-.08,.15,.23)
        mouth = light(0,.285,.41,.11)
        jaw = light(0,.427,.38,.14)
        nose = light(0,.135,.24,.16)
        eyecontrast = .044+.52*noise
        def eye(e,b,c):
            # Eyebrows may merge with the hair, esp. in low resolution.
            return unit((.65*(c-e)+.35*(b-e))/eyecontrast)
        eyes = multilateral(eye(eyeL,browL,ckL),eye(eyeR,browR,ckR))
        nosebridge = unit((bridge-.50*(eyeR+eyeL))/(.031+.42*noise))
        mouthline = unit((.53*jaw+.47*nose-mouth)/(.026+.35*noise))
        eyelevel = unit(1.-np.abs(eyeR-eyeL)/(.06+.88*noise))
        facepat = (1.24*eyes + .51*nosebridge + .49*mouthline+.22*eyelevel+
                   .27*sym+.45*sides +.24*above+.17*below+.34*hair+
                   .88*fskin+.55*isolation+.28*body+.24*shoulder+.15*arm-3.37+.14*gray)
        if w<6.5 or h<8:
            facepat -= .37
        # Guard against lattice patterns: any three dark horizontal lines are
        # easy to match in windows, roof tiles, and fences, without cheeks.
        facepat -= .35*(1.-fskin)*(1.-sides)+.18*(1.-sym)*(1.-fskin)
        # Features supply some support for a non-perfect skin candidate, too.
        skinhead += .20*eyes+.12*mouthline+.09*nosebridge-.13
    else:
        facepat = back - 1.
    # A one-sided edge must not be inferred from clamped pixels at a crop.
    margin = np.minimum(xc,95-xc)
    cut = .28*unit((.53*w-margin)/(.48*w)) + .19*unit((.45*h-yc)/(.4*h))
    cut = cut+.31*unit((.48*h-(95-yc))/(.40*h))
    # Smooth location prior, without ruling out peripheral people.
    loc = .18*((xc-47.5)/48.)**2+.12*((yc-36.)/60.)**2
    evidence = mixer((skinhead,back,facepat)).astype(np.float32)-cut-loc
    # Pool locally overlapping hits together. This prevents a single head at
    # a dozen scales being interpreted as a dozen people.
    stepmap = np.full((8,8),-8.,dtype=np.float32)
    # Separable maximum by coarse spatial bins, avoiding repeated scatter.
    ys = (yc[:,0]//12)
    xs = (xc[0]//12)
    yr = np.flatnonzero(np.r_[True, ys[1:]!=ys[:-1]])
    xr = np.flatnonzero(np.r_[True, xs[1:]!=xs[:-1]])
    pooled = np.maximum.reduceat(np.maximum.reduceat(evidence,yr,axis=0),xr,axis=1)
    stepmap[ys[yr,None],xs[xr]] = pooled
    return stepmap


def reduce2(a):
    n,h,w=a.shape
    return a.reshape(n,h//2,2,w//2,2).mean(axis=(2,4))


def blur(a):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    horiz=.25*(p[:,:,:-2]+2*p[:,:,1:-1]+p[:,:,2:])
    return .25*(horiz[:,:-2]+2*horiz[:,1:-1]+horiz[:,2:])


def derivatives(a):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    return (p[:,1:-1,2:]-p[:,1:-1,:-2])*.5,(p[:,2:,1:-1]-p[:,:-2,1:-1])*.5


# Oriented-boundary constellations. Coordinates are in head heights, with
# (0,0) at the head center. A chain is weighted per part, not per vertex.
def segments(points):
    result=[]
    for (x,y),(u,v) in zip(points,points[1:]):
        length=math.hypot(u-x,v-y)
        n=max(1,int(math.ceil(length/.32)))
        for i in range(n):
            t=(i+.5)/n
            result.append((x+(u-x)*t,y+(v-y)*t,u-x,v-y,length/n))
    return result


def build_shapes():
    # Head rim; shoulder slopes then upper arms. Normal orientation is
    # unsigned, since both dark-on-light and light-on-dark occur.
    rim=segments([(-.28,.36),(-.405,.07),(-.39,-.22),(-.26,-.44),(0,-.52),(.26,-.44),(.39,-.22),(.405,.07),(.28,.36)])
    # Ordered pairs of outer arm/torso profiles; legs are optional.
    shapes=[]
    for pose in range(4):
        if pose==0:
            left=[(-.20,.37),(-.24,.52),(-.76,.70),(-.95,.94),(-1.03,1.62),(-1.04,2.60),(-.87,2.96)]
            right=[(-x,y) for x,y in left]
            legs=[(-.80,2.90),(-.77,3.62),(-.63,4.86),(-.60,5.78)]
            rleg=[(-x,y) for x,y in legs]
        elif pose==1:
            left=[(-.18,.34),(-.27,.53),(-.59,.69),(-.71,1.17),(-.73,1.98),(-.62,2.78)]
            right=[(.19,.35),(.22,.53),(.52,.72),(.66,1.23),(.58,1.94),(.61,2.82)]
            legs=[(-.58,2.71),(-.58,3.47),(-.89,4.40),(-1.35,5.23)]
            rleg=[(.54,2.70),(.63,3.42),(.92,4.45),(1.26,5.16)]
        elif pose==2:
            left=[(-.22,.37),(-.26,.51),(-.72,.66),(-1.08,.96),(-1.36,1.34),(-1.19,1.72),(-.82,2.04)]
            right=[(-x,y) for x,y in left]
            legs=[(-.79,2.10),(-.78,2.74),(-.82,3.42),(-.93,4.40)]
            rleg=[(-x,y) for x,y in legs]
        else:
            left=[(-.19,.35),(-.22,.51),(-.58,.68),(-.64,1.13),(-.74,1.81),(-.67,2.48)]
            right=[(.19,.36),(.28,.54),(.73,.75),(1.02,1.14),(1.38,1.40),(1.24,1.70)]
            legs=[(-.65,2.45),(-.77,3.14),(-1.30,3.78),(-1.45,4.86)]
            rleg=[(.65,2.08),(.71,2.89),(1.05,3.67),(.74,4.68)]
        shapes.append([rim,segments(left),segments(right),segments(legs),segments(rleg)])
    return shapes


shapes=build_shapes()


def grad_maps(pic):
    # Oriented energy in luminance and opponent colour, with moderate
    # contrast normalization and finite smoothing for contour tolerance.
    gx,gy=derivatives(blur(pic))
    xx=(gx*gx).sum(axis=0)
    yy=(gy*gy).sum(axis=0)
    xy=(gx*gy).sum(axis=0)
    norm=np.sqrt(xx+yy+.000006)
    out=[]
    for i in range(6):
        ang=(i*math.pi)/6
        c,s=math.cos(ang),math.sin(ang)
        proj=np.maximum(0,xx*c*c+2*c*s*xy+yy*s*s)
        out.append(proj*proj/(norm*norm*norm))
    # Sharp local structures should outrank diffuse contrast in the scene.
    em=blur(np.array(out))
    k=np.pad(norm,3,mode='edge')
    integ=np.pad(k.cumsum(0).cumsum(1),((1,0),(1,0)))
    base=(integ[7:,7:]+integ[:-7,:-7]-integ[7:,:-7]-integ[:-7,7:])/49.
    em=em/(.022+.82*base[None]+.13*norm.mean())
    return np.minimum(em,2.).astype(np.float32)


def stencil(group, f, aspect, tilt):
    # Discretize the continuous boundary into weighted oriented taps once.
    taps={}
    for x,y,dx,dy,weight in group:
        px=(aspect*x+tilt*y)*f
        py=y*f
        ix,iy=int(round(px)),int(round(py))
        phi=math.atan2(aspect*dx+tilt*dy,-dy)
        orient=int(round(phi*6/math.pi))%6
        key=(orient,iy,ix)
        taps[key]=taps.get(key,0.)+weight*math.hypot(aspect*dx+tilt*dy,dy)/math.hypot(dx,dy)
    total=sum(taps.values())
    return [(i,y,x,np.float32(weight/total)) for (i,y,x),weight in taps.items()]


def shape_bank():
    bank=[]
    for f in (3.2,4.6,6.6):
        items=[]
        for aspect in (.68,1.10,1.63):
            for pose,shape in enumerate(shapes):
                for tilt in ((0.,) if pose in (0,2) else (-.15,.15)):
                    items.append((f,[stencil(p,f,aspect,tilt) for p in shape]))
        bank.append(items)
    return bank


bank=shape_bank()


def contours(pic,skin):
    best=np.full((8,8),-8.,dtype=np.float32)
    small=reduce2(pic)
    smallskin=reduce2(skin[None])[0]
    for level in range(3):
        pic2=small
        n=small.shape[-1]
        # At the last level large upper bodies can leave the frame.
        energy=grad_maps(pic2)
        pad=44
        padded=np.pad(energy,((0,0),(pad,pad),(pad,pad)))
        s=blur(smallskin[None])[0]
        y=np.arange(n,dtype=np.float32).reshape(-1,1)
        x=np.arange(n,dtype=np.float32).reshape(1,-1)
        loc=.13*((x+.5-n*.5)/(.5*n))**2+.09*((y+.5-n*.37)/(.64*n))**2
        def dot(group):
            res=np.zeros((n,n),dtype=np.float32)
            for i,dy,dx,weight in group:
                res+=weight*padded[i,pad+dy:pad+dy+n,pad+dx:pad+dx+n]
            return res
        local=np.full((n,n),-8.,dtype=np.float32)
        for items in bank:
            if items[0][0]>n*.39:
                continue
            for f,parts in items:
                rim,l,r,leg1,leg2=[dot(p) for p in parts]
                # Bound the impact of any one intense edge, and downweight
                # a contour if all its support is on just one side.
                sides=multilateral(l,r)
                legs=multilateral(leg1,leg2)
                visible=unit((n-y-2.8*f)/(1.8*f))
                silhouette=(.90*np.minimum(rim,1.5)+1.27*np.minimum(sides,1.5)+
                            .33*np.minimum(legs,1.45)*visible+.16*np.minimum(rim,sides)+
                            .29*s -2.34 -loc)
                local=np.maximum(local,silhouette)
        # Pool to the same coarse coordinate system as the colour parts.
        yy=(np.arange(n)*8//n)
        starts=np.flatnonzero(np.r_[True,yy[1:]!=yy[:-1]])
        pooled=np.maximum.reduceat(np.maximum.reduceat(local,starts,axis=0),starts,axis=1)
        best=np.maximum(best,pooled)
        if level<2:
            small=reduce2(small)
            smallskin=reduce2(smallskin[None])[0]
    return best


def components(mask):
    # Label horizontal runs, joining them if they touch in neighboring rows.
    height,width=mask.shape
    ids=np.zeros((height,width),dtype=np.int32)
    roots=[0]
    def root(k):
        while roots[k]!=k:
            roots[k]=roots[roots[k]]
            k=roots[k]
        return k
    padded=np.pad(mask,((0,0),(1,1)))
    switches=padded[:,1:]!=padded[:,:-1]
    for y,rr in enumerate(switches):
        points=np.flatnonzero(rr)
        for a,b in zip(points[::2],points[1::2]):
            links=ids[y-1,max(0,a-1):min(width,b+1)]
            first=int(links.max()) if y>0 else 0
            if first:
                k=root(first)
                for v in links[links!=first]:
                    if v:
                        roots[root(int(v))]=k
            else:
                k=len(roots)
                roots.append(k)
            ids[y,a:b]=k
    return np.array([root(k) for k in range(len(roots))])[ids]


def region_evidence(maps):
    # Is there a complete head-shaped island, with detached hands/arms of
    # a similar hue? This rejects hypotheses carved out of a larger expanse
    # of wood or earth merely by the bounding box filter.
    low=reduce2(np.array(maps))
    skin=low[3]
    best=np.zeros((8,8),dtype=np.float32)
    y,x=np.indices(skin.shape)
    for threshold in (.40,.68):
        ids=components(skin>threshold).ravel()
        area=np.bincount(ids)
        count=len(area)
        if count<2:
            continue
        xmin=np.full(count,48)
        xmax=np.zeros(count,dtype=np.int64)
        ymin=xmin.copy()
        ymax=xmax.copy()
        np.minimum.at(xmin,ids,x.ravel())
        np.maximum.at(xmax,ids,x.ravel())
        np.minimum.at(ymin,ids,y.ravel())
        np.maximum.at(ymax,ids,y.ravel())
        safe=np.maximum(area,1)
        def average(v):
            return np.bincount(ids,weights=v.ravel(),minlength=count)/safe
        xc,yc=average(x),average(y)
        w,h=xmax-xmin+1,ymax-ymin+1
        keep=np.flatnonzero((area>=2)&(w>0)&(h>0)&(area<1300)&(np.arange(count)>0))
        if not len(keep):
            continue
        p=keep
        w,h=w[p],h[p]
        cx,cy=xc[p],yc[p]
        ax,ay=average(x*x)[p]-cx*cx,average(y*y)[p]-cy*cy
        xy=average(x*y)[p]-cx*cy
        corr=np.abs(xy)/np.sqrt((ax+.25)*(ay+.25))
        fill=area[p]/(w*h)
        aspect=(w+.1)/(h+.1)
        compact=unit((fill-.31)/.38)*unit((aspect-.22)/.30)*unit((2.40-aspect)/1.)
        regular=unit((.99-fill)/.17+.28)*unit((.96-corr)/.6)
        strength=average(skin)[p]
        lum=average(low[0])[p]
        spread=np.sqrt(np.maximum(0,average(low[0]*low[0])[p]-lum*lum))
        chroma=np.arctan2(average(low[1])[p],-average(low[2])[p])
        dx=(cx[None,:]-cx[:,None])/np.maximum(2,w[:,None])
        dy=(cy[None,:]-cy[:,None])/np.maximum(2,h[:,None])
        ratio=area[p][None,:]/area[p][:,None]
        # A hand, forearm or lower leg has roughly comparable area; it
        # should be separated from the head by a neck and some clothing.
        partner=unit((dy-.48)/.72)*unit((6.2-dy)/2.3)*unit((3.25-np.abs(dx))/1.4)
        partner*=unit((ratio-.047)/.26)*unit((3.9-ratio)/2.)
        partner*=np.exp(-.5*((chroma[None,:]-chroma[:,None])/.32)**2)
        partner*=unit((.48-np.abs(lum[:,None]-lum[None,:]))/.28)
        support=partner.max(axis=1)
        two=np.minimum((partner*(dx>0)).max(axis=1),(partner*(dx<0)).max(axis=1))
        group=unit((area[p]-1)/9.)*(.61*compact+.39*regular)*unit((strength-.30)/.50)
        group*=1.-.48*unit((spread-.15)/.16)
        prior=unit((.38-skin.mean())/.28)
        value=group*(.26+.56*support+.18*two)*(.62+.38*prior)
        np.maximum.at(best,(np.minimum(7,(cy/6).astype(int)),np.minimum(7,(cx/6).astype(int))),value.astype(np.float32))
    return best


def context(pic,maps):
    r,g,b=pic
    # Sports are particularly common among depicted people: only give the
    # field cue weight for a visually fairly uniform grass-coloured surface.
    green=unit((g-r+.013)/.075)*unit((g-b-.025)/.115)*unit((.83-g)/.27)
    low=green[43:91]
    gx,gy=derivatives(maps[0][None])
    edge=np.sqrt(gx[0]*gx[0]+gy[0]*gy[0])
    turf=unit((low.mean()-.25)/.50)*unit((.093-(edge[43:91]*low).sum()/(low.sum()+1.))/.065)
    # Wide smooth blue-and-white lower areas suggest ski slopes or surf.
    snow=unit((np.minimum(np.minimum(r,g),b)-.58)/.28)*unit((b-r+.07)/.11)
    water=unit((b-r-.037)/.18)*unit((g-r-.018)/.11)*unit((g-.19)/.22)
    outside=unit((.95-np.mean(green[:24]))/.5)
    ski=unit((snow[46:].mean()-.36)/.45)*unit((water[:45].mean()-.10)/.45)
    surf=unit((water[46:].mean()-.31)/.52)*unit((snow[24:80].mean()-.035)/.18)
    # Repeated strong parallel horizontal / vertical lines are much more
    # consistent with architecture, interiors, or a vehicle than a portrait.
    ex,ey=np.abs(gx[0]),np.abs(gy[0])
    horizontal=np.maximum(0.,ey-1.8*ex)
    vertical=np.maximum(0.,ex-1.8*ey)
    hm=horizontal.mean(axis=1)
    vm=vertical.mean(axis=0)
    line=unit((np.mean(np.sort(hm)[-8:])-.034)/.085)+unit((np.mean(np.sort(vm)[-8:])-.032)/.083)
    warm_bg=unit((maps[3].mean()-.23)/.45)
    return .53*turf*outside+.33*ski+.29*surf-.13*line-.18*warm_bg


def classify(pic):
    maps,gray=photo_maps(pic)
    ii=integrate(maps)
    best=np.full((8,8),-8.,dtype=np.float32)
    for h in (3.6,5.2,7.6,11.,16.,23.,33.,48.,68.):
        for ratio in (.53,.82,1.23):
            score=scan(ii,h,h*ratio,gray)
            best=np.maximum(best,score)
    outline=contours(np.array(maps[:3]),maps[3])
    # Allow either a full silhouette or a good face to carry the decision;
    # increase confidence if the independent cues coincide spatially.
    together=np.maximum(best+.32*np.clip(outline+.22,-.55,.85),
                        1.13*outline+.29*np.clip(best+.28,-.85,.65)-.12)
    regions=region_evidence(maps)
    nearby=np.pad(regions,1)
    near=regions.copy()
    for j in range(3):
        for k in range(3):
            near=np.maximum(near,.67*nearby[j:j+8,k:k+8])
    together=together+.50*(near-.31)
    winner=np.argmax(together)
    row,column=winner//8,winner%8
    yy,xx=np.indices((8,8))
    far=np.maximum(np.abs(yy-row),np.abs(xx-column))>1
    first=together[row,column]
    second=np.max(together[far])
    score=first+.19*np.clip(second+.23,-.65,.76)+context(pic,maps)
    return int(score>.24)


def predict(x):
    return np.array([classify(pic) for pic in x],dtype=np.int64)
'''

folder=Path(sys.argv[1])
folder.mkdir(parents=True,exist_ok=True)
(folder/'solution.py').write_text(program, encoding='utf-8')
