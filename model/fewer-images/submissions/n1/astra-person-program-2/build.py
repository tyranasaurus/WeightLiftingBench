"""Build the hand-specified detector. No input files or model assets needed."""
from pathlib import Path
import sys

source = '''import numpy as np
import math


def sat(x):
    return np.clip(x, 0., 1.)


def ramp(x, a, b):
    return sat((x-a)/(b-a))


def blur(a):
    p = np.pad(a, ((0,0),)*(a.ndim-2)+((1,1),(1,1)), mode="edge")
    t = (p[..., :-2, :] + 2*p[..., 1:-1, :] + p[..., 2:, :])*.25
    return (t[..., :-2] + 2*t[..., 1:-1] + t[..., 2:])*.25


def half(a):
    return (a[..., ::2, ::2]+a[..., 1::2, ::2]+a[..., ::2, 1::2]+a[..., 1::2, 1::2])*.25


def skinmap(r,g,b):
    # A diffuse-reflectance locus, with a noise floor for shadow and an
    # extended low-saturation end for specular highlights on light skin.
    u = (r-g)/(r+g+.11)
    v = (g-b)/(g+b+.11)
    d = (v - .50*u + .012)/(.046+.18*np.maximum(u,0))
    return (np.exp(-.5*d*d)*ramp(u,.010,.069)*(1-ramp(u,.28,.44))
            *ramp(r,.055,.17)*(1-.50*ramp(v,.20,.32)))


def integral(a):
    n,m = a.shape[-2:]
    t = np.zeros(a.shape[:-2]+(n+1,m+1),np.float32)
    t[...,1:,1:] = a.cumsum(-1).cumsum(-2)
    return t


def offsets(a,b):
    i,j = int(math.floor(a+.5)), int(math.floor(b+.5))
    if j <= i:
        i = int(math.floor(.5*(a+b)))
        j = i+1
    return i,j


def rect(ii, x, y, a,b,c,d):
    # c,d exclusive; edge rectangles use only real pixels.
    h,w = ii.shape[-2:]
    l,r = np.clip(x+a,0,w-2), np.clip(x+c,1,w-1)
    t,v = np.clip(y+b,0,h-2), np.clip(y+d,1,h-1)
    return (ii[...,v,r]-ii[...,t,r]-ii[...,v,l]+ii[...,t,l])/((r-l)*(v-t)).astype(np.float32)


def nr(ii,x,y,w,h,l,t,r,b):
    a,c = offsets(w*l,w*r)
    d,e = offsets(h*t,h*b)
    return rect(ii,x,y,a,d,c,e)


def distance(a,b):
    return np.sqrt(np.sum((a-b)*(a-b),axis=0))


def atgrid(ch, yi, xi):
    return ch[...,yi[:,None],xi[None,:]]


def paint(canvas,xi,yi,s):
    # All hypothesis scales write to a common location map for peak pooling.
    gy,gx = np.ix_(yi//2,xi//2)
    canvas[gy,gx] = np.maximum(canvas[gy,gx],s)


def peaks(canvas, radius):
    z = canvas.copy()
    out = []
    for j in range(3):
        iy,ix = divmod(int(np.argmax(z)),z.shape[1])
        out.append(float(z[iy,ix]))
        z[max(0,iy-radius):iy+radius+1,max(0,ix-radius):ix+radius+1] = -3.
    return out


def makeplan():
    # Locations (in face widths/heights) on the internal features of a face.
    pts = [(0,-.28,.22,.12),(-.25,-.29,.20,.12),(.25,-.29,.20,.12),
           (-.235,-.065,.205,.095),(.235,-.065,.205,.095),
           (-.255,.095,.22,.13),(.255,.095,.22,.13),(0,.015,.105,.19),
           (0,.285,.26,.070),(0,.176,.24,.075),(0,.404,.20,.073)]
    plans = []
    for w in (6,9,14,21,32,47,63):
        for ratio in (1.04,1.64):
            h = w*ratio
            if h > 108:
                continue
            step = max(1,int(w*.075))
            xi = np.arange(max(0,int(.36*w)),min(96,97-int(.36*w)),step,dtype=np.intp)
            yi = np.arange(max(0,int(.38*h)),min(96,97-int(.37*h)),step,dtype=np.intp)
            grid = np.stack(np.broadcast_arrays(xi[None,:],yi[:,None]))
            for tilt in (-.23,0,.23):
                if w < 9 and tilt != 0:
                    continue
                # Rotate on the squashed image plane, keeping the rectangle
                # averages upright. Precompute the four summed-area indices.
                inds, areas = [], []
                for px,py,pw,ph in pts:
                    xx = px*w - tilt*py*h
                    yy = py*h + tilt*px*w
                    l,r = offsets(xx-.5*pw*w+.5,xx+.5*pw*w+.5)
                    t,b = offsets(yy-.5*ph*h+.5,yy+.5*ph*h+.5)
                    xlo = np.clip(grid[0]+l,0,95)
                    xhi = np.clip(grid[0]+r,1,96)
                    ylo = np.clip(grid[1]+t,0,95)
                    yhi = np.clip(grid[1]+b,1,96)
                    inds.append(np.stack([yhi*97+xhi,ylo*97+xhi,yhi*97+xlo,ylo*97+xlo]))
                    areas.append(1./((yhi-ylo)*(xhi-xlo)))
                plans.append((w,h,xi,yi,tilt,np.array(inds,dtype=np.int32),np.array(areas,np.float32)))
    return plans


faceplans = makeplan()


def faces(ii,iskin,icolor,prior,mono):
    flat = ii.ravel()
    facemap = np.full((48,48),-2.,np.float32)
    for w,h,xi,yi,tilt,ix,area in faceplans:
        x,y = xi[None,:], yi[:,None]
        if tilt <= -.22 or (w < 9 and tilt == 0):
            # The color / surround terms can be shared between nearby tilts.
            mid = nr(iskin,x,y,w,h,-.33,-.30,.33,.36)
            left = nr(iskin,x,y,w,h,-.79,-.25,-.50,.26)
            right = nr(iskin,x,y,w,h,.50,-.25,.79,.26)
            above = nr(iskin,x,y,w,h,-.35,-.80,.35,-.51)
            surround = (left+right+above)/3
            sk = ramp(mid,.17,.78)
            isolated = sat((mid-surround+.04)/.60)
            lcol = nr(icolor,x,y,w,h,-.33,.035,-.13,.20)
            rcol = nr(icolor,x,y,w,h,.13,.035,.33,.20)
            # Penalize an internal color boundary, not unilateral lighting.
            chdiff = abs((lcol[0]-lcol[1])-(rcol[0]-rcol[1])) + abs((lcol[2]-lcol[1])-(rcol[2]-rcol[1]))
            sim = 1-ramp(chdiff,.055,.25)
            top = nr(ii,x,y,w,h,-.29,-.58,.29,-.40)
            sides = (nr(ii,x,y,w,h,-.65,-.20,-.44,.26)+nr(ii,x,y,w,h,.44,-.20,.65,.26))*.5
            center = nr(ii,x,y,w,h,-.32,-.29,.32,.36)
            frame = .58*ramp(center-top,.005,.16)+.42*ramp(abs(center-sides),.018,.17)
            base = (.19*sk+.085*isolated + .052*sim + .068*frame
                    + .079*mono + atgrid(prior,yi,xi) - .033*(w < 10))
        v = (flat[ix[:,0]]-flat[ix[:,1]]-flat[ix[:,2]]+flat[ix[:,3]])*area
        fc,fl,fr,el,er,cl,cr,bridge,mouth,upper,chin = v
        # Eyes should be darker than *both* the forehead and the cheeks;
        # two independent local bars are much harder to match than one.
        den = (.046+.103*(fc+cl+cr)/3)
        def evidence(d):
            return sat((d/den+.015)/1.26)
        a = evidence(np.minimum(fl-el,cl-el)*.65+(.5*(fl+cl)-el)*.35)
        b = evidence(np.minimum(fr-er,cr-er)*.65+(.5*(fr+cr)-er)*.35)
        eyes = np.minimum(a,b)*.68 + (a+b)*.16
        link = evidence(bridge-(el+er)*.5)
        lips = evidence(np.minimum(upper-mouth,chin-mouth)*.55 + ((upper+chin)*.5-mouth)*.45)
        asym = abs((fl-el)-(fr-er)) + abs((cl-el)-(cr-er))
        balance = 1-ramp(asym,den*1.05,den*3.7)
        # Repeated stripes can make two eye-like bars but not a nose.
        internal = .276*eyes+.111*link+.104*lips+.045*balance+.050*np.minimum(link,eyes)
        s = base+internal
        # Resample on the common two-pixel grid (no repeated write indices).
        skip = 2 if xi[1]-xi[0] == 1 else 1
        paint(facemap,xi[::skip],yi[::skip],s[::skip,::skip])
    return facemap


def portraitplans():
    # Independent, analytic relief drawings on a face-relative grid. Their
    # coefficients are never combined to estimate a template or a threshold.
    u,v = np.meshgrid((np.arange(8)+.5)/8-.5,(np.arange(9)+.5)/9-.5)
    xx,yy = np.meshgrid(np.arange(48,dtype=np.int32)*2,np.arange(48,dtype=np.int32)*2)
    plans = []
    for w in (9,14,22,35,54,77):
        for a in (1.10,1.76):
            h = a*w
            if h>103:
                continue
            inds = []
            for x,y in zip(u.ravel(),v.ravel()):
                ix = np.clip(xx+round(x*w),0,95)
                iy = np.clip(yy+round(y*h),0,95)
                inds.append(ix+iy*96)
            banks = []
            for yaw in (0,.15,-.15):
                for tilt in (-.23,0,.23):
                    # Inverse rotation to draw a face tipped in the image.
                    c,s = math.cos(tilt),math.sin(tilt)
                    px,py = c*u+s*v*a,c*v-s*u/a
                    def spot(cx,cy,wx,wy):
                        return np.exp(-.5*((px-cx)/wx)**2-.5*((py-cy)/wy)**2)
                    shape = np.maximum(0,1-((px-.32*yaw)/.475)**2-((py+.015)/.60)**2)
                    for side in (-1,1):
                        z = .38 + .20*np.sqrt(shape)+.030*side*px
                        for d in (-1,1):
                            ex = yaw + .207*d*(1-1.1*abs(yaw))
                            z -= .170*spot(ex,-.072,.100,.057)
                            z -= .075*spot(ex,-.130,.108,.033)
                            z += (.027+.020*side*d)*spot(yaw*.6+.23*d,.085,.122,.095)
                        z += .078*spot(yaw*1.22,.032,.066,.120)
                        z -= .075*spot(yaw*1.32,.133,.100,.036)
                        z -= .145*spot(yaw*.80,.274,.142,.039)
                        z += .059*spot(yaw*.57,.380,.139,.049)
                        # Subtract this drawing's own row and column means
                        # as a spatial filter (not means across drawings).
                        # This suppresses lighting ramps and horizontal bars.
                        z = z-.79*z.mean(0,keepdims=True)-.84*z.mean(1,keepdims=True)
                        z = z-z.mean()
                        banks.append(z.ravel()/np.sqrt(np.sum(z*z)))
            plans.append((w,h,np.array(inds,dtype=np.int32).reshape(72,-1),np.array(banks,np.float32)))
    return plans


portraitbank = portraitplans()


def portraits(lum,iskin,prior,mono):
    xx = np.arange(0,96,2)[None,:]
    yy = xx.T
    bl1 = blur(lum)
    bl2 = blur(blur(bl1))
    bl3 = blur(blur(blur(bl2)))
    result = np.full((48,48),-2.,np.float32)
    for w,h,ids,bank in portraitbank:
        z = lum if w<=9 else (bl1 if w<=22 else (bl2 if w<=35 else bl3))
        q = z.ravel()[ids].reshape(9,8,-1)
        q = q-.79*q.mean(0,keepdims=True)-.84*q.mean(1,keepdims=True)
        q = q-q.mean((0,1),keepdims=True)
        v = q.reshape(72,-1)
        den = np.sqrt((v*v).sum(0)+72*.024**2)
        corr = ((bank @ v).max(0)/den).reshape(48,48)
        sk = nr(iskin,xx,yy,w,h,-.30,-.30,.30,.39)
        # Extrapolated image borders cannot count as the features of a face.
        valid = (xx>=.42*w)&(xx<96-.42*w)&(yy>=.42*h)&(yy<96-.42*h)
        score = corr+.157*ramp(sk,.18,.80)+.052*mono+prior[::2,::2]-.036*(w<=9)
        result = np.maximum(result,np.where(valid,score,-2.))
    return result


def edgefield(col):
    z = blur(col)
    p = np.pad(z,((0,0),(1,1),(1,1)),mode="edge")
    gx = p[:,1:-1,2:]-p[:,1:-1,:-2]
    gy = p[:,2:,1:-1]-p[:,:-2,1:-1]
    ex = np.sqrt((gx*gx).mean(0))
    ey = np.sqrt((gy*gy).mean(0))
    return ex,ey


def headparts(ii,iskin,prior,mono,globalwarm):
    canvas = np.zeros((48,48),np.float32)-2.
    bodmap = canvas.copy()
    cols = np.concatenate([ii,iskin[None]],axis=0)
    for w in (3.6,5.4,7.9,11.5,17.,25.,37.,53.):
        step = 2 if w<18 else 4
        xi = np.arange(0,96,step)
        yi = np.arange(0,96,step)
        x,y = xi[None,:],yi[:,None]
        for aspect in (1.18,1.85):
            h = w*aspect
            if h>82:
                continue
            def box(a,b,c,d):
                return nr(cols,x,y,w,h,a,b,c,d)
            center = box(-.32,-.26,.32,.32)
            # Background sampled on the two flanks and over the head.
            left = box(-.96,-.27,-.55,.28)
            right = box(.55,-.27,.96,.28)
            above = box(-.35,-.89,.35,-.54)
            cap = box(-.34,-.55,.34,-.29)
            under = box(-.37,.55,.37,.86)
            shirt = box(-.53,.99,.53,1.61)
            sc = sat(center[3]); sl = sat(left[3]); sr = sat(right[3]); st = sat(above[3])
            # Sort the three signed center/surround differences implicitly.
            sep = (sl+sr+st)/3
            worst = np.maximum(np.maximum(sl,sr),st)
            iso = sat((sc-.65*sep-.35*worst+.03)/.67)
            below = sat((sc - .35*under[3]-.65*shirt[3]+.07)/.70)
            lc,rc,uc = distance(center[:3],left[:3]),distance(center[:3],right[:3]),distance(center[:3],above[:3])
            sideoff = ramp(np.minimum(lc,rc)*.65+.35*(lc+rc)*.5,.052,.32)
            topoff = ramp(uc,.050,.33)
            bottomoff = ramp(distance(center[:3],shirt[:3]),.08,.39)
            capdark = ramp(center[0]-cap[0],-.014,.13)
            # The torso should widen below the head, as measured by a
            # five-column strip across the shoulders and upper garment.
            wingl = box(-.90,.97,-.48,1.64)
            wingr = box(.48,.97,.90,1.64)
            farl = box(-1.65,.89,-1.13,1.64)
            farr = box(1.13,.89,1.65,1.64)
            dl,dr = distance(wingl[:3],shirt[:3]),distance(wingr[:3],shirt[:3])
            vl,vr = distance(farl[:3],shirt[:3]),distance(farr[:3],shirt[:3])
            expand = .5*(ramp(distance(left[:3],wingl[:3])-dl*.60,.025,.28)+ramp(distance(right[:3],wingr[:3])-dr*.60,.025,.28))
            wide = .5*(ramp(vl-dl*.62,.035,.30)+ramp(vr-dr*.62,.035,.30))
            bg = 1-ramp(distance(left[:3],right[:3]),.12,.50)
            coherent = 1-ramp((dl+dr)*.5,.085,.38)
            # Reject a horizontal layer boundary that continues straight
            # through the presumed shoulders, using the outside below.
            bgcont = 1-ramp(.5*(distance(left[:3],farl[:3])+distance(right[:3],farr[:3])),.115,.52)
            torso = (.29*expand + .28*wide + .13*coherent + .13*bgcont + .17*np.minimum(wide,expand))
            visibility = sat((96-y-.48*h)/(h*1.2))
            torso = visibility*torso+(1-visibility)*.33
            p = atgrid(prior,yi,xi)
            # Skin separated on three sides, with a cap or an expanding
            # garment. Relative terms avoid treating a whole wood room as skin.
            ss = (.234*sc+.191*iso+.096*below+.090*sideoff+.044*topoff+.076*capdark
                  +.060*bottomoff+.178*torso+.031*bg+p - .020*(w<5) - .019*globalwarm)
            # Reduce responses on thick patches of chroma if there is no
            # separate head-sized component at all.
            ss = ss - .076*(1-iso)*(1-capdark)
            # Seek a second patch with similar chroma at plausible locations
            # for a hand or forearm. Subtract evidence from nearby background
            # so the mere presence of a large tan wall cannot satisfy this.
            extremities = []
            for side in (-1,1):
                hand = 0.
                for hx,hy,ww,hh in ((1.10,2.47,.27,.47),(1.38,1.54,.30,.32),(.46,2.16,.25,.35),(1.12,.38,.24,.34)):
                    cx = side*hx
                    a = box(cx-ww,hy-hh,cx+ww,hy+hh)
                    outside = nr(iskin,x,y,w,h,cx+side*.65-ww,hy-hh,cx+side*.65+ww,hy+hh)
                    coloragree = 1-ramp(distance(a[1:3],center[1:3]),.055,.235)
                    visible = (x+cx*w>=1)&(x+cx*w<95)&(y+hy*h<95)
                    # A true extremity is localized and similar to face color
                    # even when it is more strongly illuminated than the face.
                    m = (ramp(a[3],.18,.80)*ramp(a[3]-.60*outside-.30*shirt[3],.06,.49)
                         *coloragree*visible)
                    hand = np.maximum(hand,m)
                extremities.append(hand)
            flesh = .68*np.maximum(extremities[0],extremities[1])+.32*np.minimum(extremities[0],extremities[1])
            ss = ss+.090*ramp(sc,.25,.73)*(flesh-.24)*visibility
            paint(canvas,xi,yi,ss)
            # For hooded people / people from behind: a connected silhouette
            # with a head narrow relative to the body; test garment extent.
            if w <= 25:
                belly = box(-.48,1.75,.48,2.55)
                down = 1-ramp(distance(belly[:3],shirt[:3]),.10,.41)
                neckmatch = 1-ramp(distance(under[:3],shirt[:3]),.095,.39)
                headframe = .55*np.minimum(sideoff,topoff)+.45*(sideoff+topoff)*.5
                # The head is distinct on both sides; flanks above shoulders
                # should be similar to the background past the sleeve ends.
                silhouette = (.235*headframe + .241*torso + .087*bg + .086*bgcont + .084*down
                              + .055*neckmatch + .083*coherent + .058*sc + .048*bottomoff
                              + .036*mono+p-.04*(1-visibility)-.028*(w<5))
                paint(bodmap,xi,yi,silhouette)
    return canvas,bodmap


def neigh(a,dy,dx):
    h,w = a.shape[-2:]
    iy = np.clip(np.arange(h)+dy,0,h-1)
    ix = np.clip(np.arange(w)+dx,0,w-1)
    return a[...,iy[:,None],ix[None,:]]


def gradcols(rgb):
    p = np.pad(rgb,((0,0),(1,1),(1,1)),mode="edge")
    dx = (p[:,1:-1,2:]-p[:,1:-1,:-2])*.5
    dy = (p[:,2:,1:-1]-p[:,:-2,1:-1])*.5
    # Select the channel on which an edge is most visible.
    k = np.argmax(dx*dx+dy*dy,axis=0)[None]
    return np.take_along_axis(dx,k,axis=0)[0],np.take_along_axis(dy,k,axis=0)[0]


def contourparts():
    # Each entry: region, relative x, y (in person heights), tangent in
    # the normalized coordinate system. Independent analytic outlines.
    def outline(aspect,pose):
        if pose == 0: # standing / walking
            centerhead, neck, hip = .0,.0,.0
            segments = [[(-.24,.147),(-.66,.19),(-.84,.235)],
                        [(-.84,.235),(-.98,.355),(-1.03,.47)],
                        [(-.65,.285),(-.59,.43),(-.60,.525)],
                        [(-.60,.525),(-.57,.67),(-.50,.83),(-.55,.955)],
                        [(-.02,.60),(-.11,.725),(-.18,.88),(-.16,.95)]]
        elif pose == 1: # relaxed asymmetrical pose, one bent elbow
            centerhead, neck, hip = -.10,-.02,.08
            segments = [[(-.30,.145),(-.68,.185),(-.82,.22)],
                        [(-.82,.22),(-.93,.292),(-1.09,.362),(-.74,.427)],
                        [(-.60,.283),(-.54,.40),(-.58,.535)],
                        [(-.58,.535),(-.62,.68),(-.77,.80),(-.74,.952)],
                        [(-.03,.60),(-.12,.715),(-.38,.83),(-.41,.95)]]
        else: # running, split legs, offset head
            centerhead, neck, hip = .20,.12,-.04
            segments = [[(-.15,.145),(-.51,.168),(-.70,.20)],
                        [(-.70,.20),(-.92,.268),(-1.14,.335),(-1.44,.30)],
                        [(-.54,.285),(-.57,.39),(-.67,.505)],
                        [(-.67,.505),(-.86,.605),(-1.12,.72),(-1.61,.79),(-1.72,.93)],
                        [(-.10,.58),(-.52,.70),(-.76,.79),(-1.40,.84)]]
        arr = []
        # Relative x is expressed in upper-body half widths.
        for deg in (0,35,70,110,145,180,211,327):
            a = deg*math.pi/180
            rx,ry = .36*aspect,.067
            arr.append((0,(centerhead+.36*math.cos(a))*aspect,.080+ry*math.sin(a),-rx*math.sin(a),ry*math.cos(a)))
        for side in (1,-1):
            for j,seg in enumerate(segments):
                for k in range(len(seg)-1):
                    (x1,y1),(x2,y2) = seg[k],seg[k+1]
                    if side==-1 and pose>0:
                        # Other side of the body: unequal limb articulation.
                        if j==1:
                            x2 *= .91
                            x1 *= .91
                            y1 = .22+(y1-.22)*1.32
                            y2 = .22+(y2-.22)*1.32
                        if j>=3 and pose==2:
                            x1*=.65; x2*=.65
                            y1 = .51+(y1-.51)*.88
                            y2 = .51+(y2-.51)*.88
                    count = 2 if j in (1,2,3) else 1
                    for i in range(count):
                        t = (i+.5)/count
                        arr.append((1 if j<3 else 2,(x1*(1-t)+x2*t)*side*aspect,y1*(1-t)+y2*t,(x2-x1)*side*aspect,y2-y1))
        return arr
    programs = []
    for H in (31,45,66,96,136):
        # resolution of the gradient pyramid at this height
        stride = 2 if H <= 66 else 4
        for aspect in (.106,.177):
            for pose in (0,1,2):
                path = outline(aspect,pose)
                groups = []
                for part in range(3):
                    pts = []
                    for tag,x,y,tx,ty in path:
                        if tag == part:
                            orient = int(round(4*math.atan2(tx,-ty)/math.pi))%4
                            pts.append((orient,round((y-.08)*H/stride),round(x*H/stride)))
                    groups.append(tuple(pts))
                programs.append((H,stride,groups))
    return programs


contours = contourparts()


def contourmap(rgb,skin,prior):
    # Locally normalized oriented contour responses, not raw gradient
    # magnitude. Discount textures via competing orientations and context.
    canv = np.zeros((48,48),np.float32)-2.
    maps = []
    z = rgb
    for scale in (2,4):
        z = half(z)
        dx,dy = gradcols(z)
        mag = np.sqrt(dx*dx+dy*dy)
        den = .027+.58*blur(blur(mag))+.42*mag
        stack = []
        for k in range(4):
            a = k*math.pi/4
            u = abs(dx*math.cos(a)+dy*math.sin(a))
            v = abs(dy*math.cos(a)-dx*math.sin(a))
            val = sat((u-.54*v)/den)
            # Allow the silhouette to miss the exact sample by half a cell.
            val = np.maximum(val,.75*blur(val)+.25*np.maximum(neigh(val,0,1),neigh(val,0,-1)))
            stack.append(val)
        edge = np.stack(stack)
        maps.append((np.pad(edge,((0,0),(4,86),(46,46)),mode="constant",constant_values=.16),blur(blur(edge.mean(0)))))
    for H,stride,groups in contours:
        n = 96//stride
        ed,bg = maps[0 if stride==2 else 1]
        yi = np.arange(0,n,1 if stride==4 else 2)
        xi = yi
        st = 1 if stride==4 else 2
        values = []
        for pts in groups:
            v = 0.
            for ori,dy,dx in pts:
                top = 4+dy; left = 46+dx
                v = v+ed[ori,top:top+n:st,left:left+n:st]
            values.append(v/len(pts))
        hd,tor,leg = values
        # Legs need not be visible in portraits. More evidence is required
        # from the upper silhouette if the leg term is omitted.
        cont = .32*hd+.42*tor+.26*leg
        upper = .445*hd+.555*tor-.064
        cont = np.maximum(cont,upper)
        y,x = yi*stride,xi*stride
        context = atgrid(bg,yi,xi)
        sc = atgrid(skin,y,x)
        cont = cont-.31*context+.077*sc+atgrid(prior,y,x)*.7
        paint(canv,x,y,cont)
    return canv


def dilate(z,r):
    n,m = z.shape
    pad = np.pad(z,((0,0),(r,r)),mode="edge")
    v = np.maximum.reduce([pad[:,i:i+m] for i in range(2*r+1)])
    pad = np.pad(v,((r,r),(0,0)),mode="edge")
    return np.maximum.reduce([pad[i:i+n] for i in range(2*r+1)])


def scene(rgb,skin,ex,ey):
    r,g,b = rgb
    # Broad priors only: open playing field, water, snow, versus strongly
    # rectilinear interiors / structures. Avoid identifying green with grass
    # unless a substantial lower region is consistent with a field.
    field = ramp(g-r,.005,.077)*ramp(g-b,.02,.12)*ramp(g,.13,.39)
    turf = field[41:].mean()
    smoothfield = field[41:]*(1-ramp((ex+ey)[41:],.065,.21))
    ground = ramp(turf,.15,.57)*(.6+.4*ramp(smoothfield.mean(),.09,.40))
    waves = ramp(b-r,.035,.23)*ramp(g-r,.017,.15)*ramp(g,.21,.55)
    ocean = ramp(waves[43:].mean(),.16,.60)
    light = rgb.mean(0)
    snow = ramp(light,.67,.90)*(1-ramp(rgb.max(0)-rgb.min(0),.075,.24))
    snowfield = ramp(snow[48:].mean(),.29,.74)*ramp((b-r)[50:].mean(),-.01,.08)
    # Large axis-aligned edges and uniform horizontal bands are common in
    # photos of structures, vehicles and furnishings, less so on bodies.
    dx,dy = gradcols(blur(rgb))
    gm = np.sqrt(dx*dx+dy*dy)
    aligned = ((np.maximum(abs(dx),abs(dy))-.90*gm).clip(0)/(.1*gm+.0001)*gm).sum()/(gm.sum()+.1)
    lines = ramp(aligned,.48,.75)
    edge = ex+ey
    toptex = edge[:30].mean(); midtex=edge[27:80,18:78].mean()
    focus = ramp((midtex+.025)/(toptex+.025),1.15,2.7)
    # Evaluate central warm/skin occupancy relative to the photo, rather
    # than counting wood grain, sand, sunsets, etc. as exposed skin.
    cen = skin[12:78,18:78].mean()
    allskin = skin.mean()
    local = ramp(cen-allskin,-.005,.135)*ramp(cen,.045,.26)
    lip = ramp((r-g)/(r+g+.13),.13,.28) * ramp((b-g)/(b+g+.13),.05,.24)
    colorful = ramp(lip.mean(),.008,.07)
    # Long aligned runs favor buildings, furniture and vehicles over a
    # person. Pool a fraction of the strongest scanlines, not a single edge.
    vx = sat((abs(dx)-2.0*abs(dy))/.085)
    vy = sat((abs(dy)-2.0*abs(dx))/.085)
    spans = []
    for u in (vx,vy.T):
        f = np.pad(u,((0,0),(1,1)),mode="edge")
        u = np.maximum(u,.85*np.maximum(f[:,:-2],f[:,2:]))
        linescores = [u[i:i+32].mean(0) for i in (0,16,32,48,64)]
        spans.append(np.sort(np.concatenate(linescores))[-18:].mean())
    ruled = ramp((spans[0]+spans[1])*.5,.41,.81)
    texture = ramp(edge.mean(),.045,.14)
    dry = ramp(((r>b+.07)&(g>b+.04)&(g<r+.008)&(g>r-.09)).mean(),.28,.70)
    # A contextual cue for an arena: a colored playing surface in the
    # bottom of the photo below a noticeably more textured backdrop.
    stadium = ground*ramp((toptex+.03)/(edge[58:].mean()+.03),1.18,2.3)
    return (.035*ground+.046*ocean+.048*snowfield-.044*lines-.042*ruled +.019*(focus-.4)
            +.019*stadium-.014*texture-.012*dry+.052*local-.030*ramp(allskin,.34,.69)+.017*colorful)


def classify(pic):
    rgb = np.ascontiguousarray(pic)
    r,g,b = rgb
    # Fractional gray-world correction bounded to avoid neutralizing large
    # naturally colored subjects or introducing skin into monochrome photos.
    s = rgb.mean((1,2)); neutral = .26*s[0]+.51*s[1]+.23*s[2]
    gain = np.clip(((neutral+.09)/(s+.09))**.46,.84,1.19)
    adj = np.clip(gain[:,None,None]*rgb,0,1)
    sk0 = skinmap(r,g,b)
    sk1 = skinmap(adj[0],adj[1],adj[2])
    skin = blur(np.maximum(.88*sk0,sk1))
    # Chromaticity (less sensitive to shadow) plus intensity in the same
    # three-dimensional metric, for surround and garment comparisons.
    lum = .28*r+.54*g+.18*b
    col = np.stack((.90*lum,1.70*(r-g),1.45*(b-g)))
    ii = integral(rgb)
    isk = integral(skin)
    il = integral(lum)
    mono = 1-ramp((abs(r-g)+abs(g-b)).mean(),.009,.065)
    gy,gx = np.mgrid[:96,:96]
    prior = (.017 - .026*((gx-47.5)/48)**2 - .062*((gy-31)/61)**2).astype(np.float32)
    ex,ey = edgefield(rgb)
    globalwarm = ramp(skin.mean(),.30,.66)
    fm = faces(il,isk,ii,prior,mono)
    pm = portraits(lum,isk,prior,mono)
    hm,bm = headparts(integral(col),isk,prior,mono,globalwarm)
    cm = contourmap(rgb,skin,prior)
    # Require agreement at the same head location for the corroboration
    # bonus. Allow for the offset between hair, face, and silhouette centers.
    sh,sb,sf = dilate(hm,2),dilate(bm,2),dilate(fm,2)
    port = float((pm+.070*(ramp(sf,.59,.82)-.50)+.036*(ramp(sh,.63,.83)-.5)).max())
    face = float((fm+.038*(ramp(sh,.61,.81)-.55)+.030*(ramp(dilate(pm,1),.42,.69)-.50)).max())
    hs = peaks(hm+.034*(ramp(sf,.59,.80)-.42),6)
    bs = peaks(bm+.032*(ramp(dilate(cm,2),.43,.63)-.45),8)
    cs = peaks(cm+.063*(ramp(sb,.72,.91)-.45)+.029*(ramp(sh,.64,.84)-.45),7)
    ctx = scene(rgb,skin,ex,ey)
    # Evidence margins of different mechanisms, in comparable units. A
    # second spatially distinct candidate only contributes near threshold.
    mh = hs[0]-.788+.055*ramp(hs[1],.66,.83)+.017*ramp(hs[2],.70,.85)
    mf = max(face-.817,.67*(port-.777))
    mb = bs[0]-.910+.020*ramp(bs[1],.79,.92)
    mc = cs[0]-.635+.021*ramp(cs[1],.51,.65)
    ordered = sorted((mh,mf,mb,mc),reverse=True)
    margin = ordered[0] + .24*np.clip(ordered[1]+.014,-.093,.085) + .11*np.clip(ordered[2],-.08,.08) + ctx
    return int(margin > 0)


def predict(x):
    return np.array([classify(pic) for pic in x],dtype=np.int32)
'''

folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
(folder / 'solution.py').write_text(source)
