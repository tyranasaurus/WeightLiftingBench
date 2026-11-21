import math
import numpy as np


def unit(a):
    return np.minimum(1., np.maximum(0., a))


def pos(a):
    return np.maximum(0., a)


def box(a, width, height):
    # Edge-replicated moving average. Integer centre convention at even widths.
    if width == 1 and height == 1:
        return a
    l = width // 2
    t = height // 2
    b = np.pad(a, ((0, 0), (t, height-1-t), (l, width-1-l)), mode='edge')
    b = np.pad(b, ((0, 0), (1, 0), (1, 0)))
    b = b.cumsum(2, dtype=np.float32).cumsum(1, dtype=np.float32)
    return ((b[:, height:, width:] - b[:, :-height, width:]) -
            (b[:, height:, :-width] - b[:, :-height, :-width])) * (1./(width*height))


def skinfield(r, g, b, adjustment):
    # Chrominance relative to intensity with a noise floor in deep shadow.
    lum = .299*r + .587*g + .114*b
    dr = (r-g) / (.25 + lum) * adjustment
    db = (g-b) / (.25 + lum) * adjustment
    # Two opponent axes: orange strength and perpendicular distance from the
    # orange line. Very saturated red, magenta and yellow are outside it.
    amp = .84*dr + .49*db
    perp = db - .56*dr
    chroma = unit((amp-.016)/.086) * unit((.80-amp)/.29)
    chroma *= np.exp(-.5*(perp/(.048 + .27*pos(amp)))**2)
    return chroma * unit((lum-.024)/.085)


def edges(im):
    # Six unsigned orientations (unit vectors are normals, not tangents).
    p = np.pad(im, ((0, 0), (1, 1), (1, 1)), mode='edge')
    dx = ((p[:, :-2, 2:]+2*p[:, 1:-1, 2:]+p[:, 2:, 2:]) -
          (p[:, :-2, :-2]+2*p[:, 1:-1, :-2]+p[:, 2:, :-2]))*.125
    dy = ((p[:, 2:, :-2]+2*p[:, 2:, 1:-1]+p[:, 2:, 2:]) -
          (p[:, :-2, :-2]+2*p[:, :-2, 1:-1]+p[:, :-2, 2:]))*.125
    # Luminance usually has the strongest edge; permit isoluminant colours.
    energies = dx*dx + dy*dy
    which = energies.argmax(0)
    u = np.take_along_axis(dx, which[None], axis=0)[0]
    v = np.take_along_axis(dy, which[None], axis=0)[0]
    mag = np.sqrt(u*u + v*v)
    ori = (np.arctan2(v,u) * (6./math.pi)) % 6.
    bins = []
    for k in range(6):
        d = abs(ori-k)
        bins.append(mag * pos(1.-np.minimum(d,6.-d)))
    return np.stack(bins)


def make_contour(points):
    # Sample a hand-specified contour at midpoints of short line segments.
    q = []
    for k in range(len(points)-1):
        x,y = points[k]
        xx,yy = points[k+1]
        dx,dy = xx-x, yy-y
        count = max(1, int(math.ceil(math.hypot(dx,dy)/.45)))
        for j in range(count):
            a = (j+.5)/count
            q.append((x+dx*a,y+dy*a,dx,dy))
    return q


def mirror(q):
    return [(-x,y,-dx,dy) for x,y,dx,dy in q]


crown = make_contour([(-.50,.08),(-.51,-.13),(-.42,-.35),(-.23,-.49),
                      (0.,-.54),(.23,-.49),(.42,-.35),(.51,-.13),(.50,.08)])
neckside = make_contour([(-.51,.05),(-.47,.22),(-.32,.43),(-.32,.53),
                         (-.84,.64),(-1.08,.80)])
armline = make_contour([(-.84,.64),(-1.08,.80),(-1.22,1.19),(-1.32,1.67),(-1.31,2.12)])
legline = make_contour([(-.92,2.70),(-.89,3.38),(-.83,4.11),(-.82,4.89),(-.78,5.66)])


def contour_weights():
    groups = [crown, neckside, mirror(neckside), armline, mirror(armline), legline, mirror(legline)]
    bank = []
    for h,w in sizes:
        row = []
        for group in groups:
            stencil = []
            for x,y,dx,dy in group:
                ori = (math.atan2(dx*w,-dy*h)*6./math.pi) % 6.
                stencil.append((int(round(x*w)),int(round(y*h)),int(round(ori))%6))
            row.append(stencil)
        bank.append(row)
    return bank


sizes = []
for headheight in (3, 5, 8, 12, 18, 27, 40, 59):
    for aspect in (.57, .91, 1.35):
        sizes.append((headheight, max(2,int(round(headheight*aspect)))))
stencils = contour_weights()


def image_score(x):
    r,g,b = x
    lum = (.299*r + .587*g + .114*b)
    chrom = float(np.mean(np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)))
    grey = float(unit((.038-chrom)/.036))
    # Partial illuminant correction: never force the image's average to grey.
    means = x.mean(axis=(1,2))
    gains = np.clip(((means.mean()+.15)/(means+.15))**.52,.83,1.19)
    sk = np.maximum(skinfield(r,g,b,1.), .91*skinfield(r*gains[0],g*gains[1],b*gains[2],1.))
    rg = r-g
    bg = b-g
    tex = edges(np.stack((lum,.72*rg,.72*bg)))
    mag = tex.sum(axis=0)
    im = np.stack((lum,rg,bg,sk,sk*sk,mag))
    total_skin = float(sk.mean())
    strength = float(np.mean(sk*sk))

    # Context is deliberately weaker than a local detection. Compact areas
    # of skin are more informative than the same colour across the scene.
    inner = float(sk[12:86,12:84].mean())
    skin_prior = (1.-grey)*(.38*float(unit((strength-.005)/.095)) - .19
                             + .18*float(np.clip((inner-total_skin)/.09,-1.,1.))
                             - .15*float(unit((total_skin-.45)/.35)))
    # COCO sports fields, pools/surf and ski slopes are useful even at a
    # distance where an individual face has only two or three pixels.
    grass = unit((g-r+.014)/.085)*unit((g-b-.018)/.11)*unit((lum-.12)/.18)
    grass = grass * unit((.85-lum)/.20)
    snow = unit((lum-.59)/.20)*unit((.15-abs(r-b))/.10)*unit((b-r+.05)/.075)
    water = unit((b-r-.035)/.16)*unit((g-r-.012)/.12)*unit((.25-abs(b-g))/.14)
    lowerg = float(grass[38:].mean())
    upperg = float(grass[:32].mean())
    arena = float(unit((lowerg-.19)/.50))*float(unit((lowerg-upperg+.06)/.36))
    snowfield = float(unit((snow[47:].mean()-.28)/.48))
    sea = float(unit((water[38:].mean()-.27)/.51))
    context = skin_prior + .22*arena + .16*snowfield + .13*sea

    # Small supporting cue: long, straight upright or horizontal edges
    # occupying much of the picture more often indicate a building/interior.
    vlines = tex[0].mean(0)
    hlines = tex[3].mean(1)
    ruled = float(unit((max(vlines.max(),hlines.max())-.060)/.12))
    context -= .11*ruled
    # A uniform two-sided background and textured subject in the upper
    # middle is weak evidence that the photographer isolated a subject.
    centretex = float(mag[15:72,25:71].mean())
    sidetex = .5*float(mag[15:72,:20].mean()+mag[15:72,76:].mean())
    popout = (centretex-sidetex)/(.024+centretex+sidetex)
    context += .20*float(np.clip(popout,-.5,.75))
    warmmass = float(sk[9:83,10:86].mean())
    context -= .09*(1.-grey)*float(unit((.025-warmmass)/.025))

    maxgrid = np.full((12,12),-3.,np.float32)
    best = -3.
    cache = {}
    def filtered(a, wid, hei, key):
        wid = max(1,int(round(wid)))
        hei = max(1,int(round(hei)))
        tag = (key,wid,hei)
        if tag not in cache:
            cache[tag] = box(a,wid,hei)
        return cache[tag]

    for si in range(len(sizes)):
        h,w = sizes[si]
        step = max(2, int(round(min(w,h)*.17)))
        start = step//2
        xs = np.arange(start,96,step)
        ys = np.arange(start,96,step)
        # Cache integer lattice shifts within this aspect/scale only.
        xind = {}
        yind = {}
        def ix(k):
            if k not in xind:
                xind[k] = np.clip(xs+k,0,95)[None,:]
            return xind[k]
        def iy(k):
            if k not in yind:
                yind[k] = np.clip(ys+k,0,95)[:,None]
            return yind[k]
        def sample(a,dx,dy):
            return a[...,iy(int(round(dy*h))),ix(int(round(dx*w)))]

        small = filtered(im,.28*w,.21*h,0)
        core = filtered(im,.71*w,.74*h,0)
        body = filtered(im,1.10*w,.73*h,0)
        c = sample(core,0,.06)
        crownpix = sample(small,0,-.35)
        below = sample(body,0,1.08)
        lnear = sample(core,-.96,.03)
        rnear = sample(core,.96,.03)
        abv = sample(core,0,-.83)
        farl = sample(body,-1.92,1.12)
        farr = sample(body,1.92,1.12)

        def dist(a,b):
            return np.sqrt((a[0]-b[0])**2 + .50*((a[1]-b[1])**2+(a[2]-b[2])**2))
        def contrast(a,b):
            return unit((dist(a,b)-.025)/.25)
        def agree(a,b):
            return 1./(1.+(dist(a,b)/.125)**2)

        dl = contrast(c,lnear)
        dr = contrast(c,rnear)
        du = contrast(c,abv)
        isolate = (.46*np.minimum(dl,dr)+.24*np.maximum(dl,dr)+.30*du)
        sides = .65*np.minimum(contrast(below,farl),contrast(below,farr))+.35*np.maximum(contrast(below,farl),contrast(below,farr))
        shin = sample(body,0,2.02)
        bleft = sample(core,-.46,1.02)
        bright = sample(core,.46,1.02)
        shirt = (.4*agree(bleft,bright)+.30*agree(below,shin)+.30*sides)
        # Convex skin island and the skin-to-hair / skin-to-clothing transitions.
        fs = c[3]
        highskin = sample(small,0,.12)[3]
        facecolour = .72*fs + .28*highskin
        surround = (.34*lnear[3]+.34*rnear[3]+.18*abv[3]+.14*below[3])
        local = unit((fs-surround-.045)/.59)
        local *= .73+.27*unit((fs-np.maximum(lnear[3],rnear[3])+.02)/.55)
        cloth = unit((fs-below[3]+.025)/.70)
        hair = unit((c[0]-crownpix[0]+.01)/.26)*unit((fs-crownpix[3]+.23)/.65)
        pure = facecolour * (.46 + .34*local + .11*cloth + .09*hair)

        # Is there a pair of darker eye sockets with a lighter nose and
        # cheeks? Require both sockets, not just a strong one-sided edge.
        el = sample(small,-.225,-.079)
        er = sample(small,.225,-.079)
        cl = sample(small,-.225,.105)
        cr = sample(small,.225,.105)
        nose = sample(small,0,.025)
        brow = sample(small,0,-.25)
        mouth = sample(small,0,.245)
        chin = sample(small,0,.365)
        le = (.46*cl[0]+.32*nose[0]+.22*brow[0])-el[0]
        re = (.46*cr[0]+.32*nose[0]+.22*brow[0])-er[0]
        eye = unit((np.minimum(le,re)-.006)/.106)
        bridge = unit((nose[0]-.5*(el[0]+er[0])-.004)/.094)
        symmetry = .5*agree(el,er)+.5*agree(cl,cr)
        lips = unit((.40*chin[0]+.30*cl[0]+.30*cr[0]-mouth[0])/.09)
        eyes = eye * (.5+.25*bridge+.25*symmetry) * min(1.,max(0.,(w-2)/5.))
        facial = .66*eyes+.19*lips*unit(eye*2.)+.15*symmetry*eyes

        # Normal-direction edge support, with a broad local denominator and
        # contrast floor. A single straight line can't explain all groups.
        ew = max(1, int(round(w*.31)))
        eh = max(1, int(round(h*.235)))
        orient = filtered(tex,ew,eh,1)
        norm = filtered(mag[None],1.4*w,1.32*h,2)[0]
        tag = (3,ew,eh,max(1,int(round(1.4*w))),max(1,int(round(1.32*h))))
        if tag not in cache:
            signal = orient+.50*(np.roll(orient,1,axis=0)+np.roll(orient,-1,axis=0))
            cache[tag] = unit((signal/(.013+norm)-.17)/.99)
        oe = cache[tag]
        def trace(stencil):
            t = 0.
            for dx,dy,k in stencil:
                t = t + oe[k,iy(dy),ix(dx)]
            return t/len(stencil)
        st = stencils[si]
        top = trace(st[0])
        nl = trace(st[1])
        nr = trace(st[2])
        al = trace(st[3])
        ar = trace(st[4])
        upper = .36*top+.41*(.65*np.minimum(nl,nr)+.35*np.maximum(nl,nr))+.23*(.58*np.minimum(al,ar)+.42*np.maximum(al,ar))
        if h <= 27:
            ll, lr = trace(st[5]), trace(st[6])
            legs = .60*np.minimum(ll,lr)+.40*np.maximum(ll,lr)
            legs *= unit((96.-ys[:,None]-4.3*h)/(1.3*h))
        else:
            legs = 0.

        # Head-over-shirt colour geometry, including continuity of the
        # background past the narrowing at the neck and into the upper arms.
        lshould = sample(core,-.78,1.02)
        rshould = sample(core,.78,1.02)
        rl = unit((dist(lnear,below)-dist(lshould,below)+.005)/.20)
        rr = unit((dist(rnear,below)-dist(rshould,below)+.005)/.20)
        flare = .64*np.minimum(rl,rr)+.36*np.maximum(rl,rr)
        # Compare the left and right outer background with the narrow part
        # beside the head; never require them to be the same as each other.
        bgcontinuity = .5*agree(lnear,farl)+.5*agree(rnear,farr)
        transition = contrast(c,below)
        structure = (.22*isolate+.17*sides+.15*shirt+.23*flare+.13*transition+.10*bgcontinuity)

        # A second patch of skin at an anatomically plausible hand position.
        # A patch resembling the face in chrominance counts more, as does
        # an offset where a hand would be distinct from the shirt/background.
        hand = np.zeros_like(fs)
        for offx,offy in ((-1.25,2.40),(1.25,2.40),(-.85,1.70),(.85,1.70),
                          (-1.48,1.23),(1.48,1.23)):
            patch = sample(core,offx,offy)
            common = 1./(1.+((patch[1]-c[1])/.105)**2 + ((patch[2]-c[2])/.095)**2)
            hand = np.maximum(hand,common*unit((patch[3]-.5*below[3]-.20)/.58))
        hands = hand*facecolour*local

        # Scale and spatial priors are mild; cut-off heads are permitted,
        # but don't let replicated border pixels stand in for a whole body.
        xx = xs[None,:]
        yy = ys[:,None]
        sideborder = unit((.52*w-np.minimum(xx,95-xx))/(.52*w))
        low = unit((yy-72)/23.)
        border = .15*sideborder+.16*low+.08*unit((.36*h-yy)/(.36*h))
        missing = unit((yy+1.2*h-96.)/(1.2*h))
        tiny = unit((5.-min(w,h))/3.)
        prior = .068*(1.-abs(xx-47.5)/48.)+.035*(1.-abs(yy-35.)/60.)
        # Separate ways to justify a detection. Each score is a margin.
        coloured = (1.20*pure+.36*local+.19*hair*facecolour+.13*cloth*facecolour
                    +.32*facial+.30*upper+.32*structure+.18*hands+.09*legs
                    - 1.62 - .16*tiny-.11*missing)
        silhouette = (1.40*upper+.88*structure+.24*legs+.23*facial
                      +.30*pure+.10*hands - 1.47 - .11*tiny-.20*missing)
        # Body grouping without perfect pointwise alignment to a contour.
        dressed = (1.15*structure+.41*flare+.36*upper+.24*isolate+.33*pure
                   +.19*facial+.14*legs - 1.65 - .11*tiny-.22*missing)
        q = np.maximum(np.maximum(coloured,silhouette),dressed) + prior - border
        # Front-facing head branch for portraits too close for the body model.
        front = (1.13*facial+.35*symmetry+.45*pure+.31*local+.34*top
                 +.19*isolate+.17*hair*facecolour-1.46 + prior - .55*border)
        q = np.maximum(q,front)
        if h <= 18:
            # The dressed figure as two long colour regions: shirt above a
            # similar-coloured pair of legs. The two legs can lean apart.
            pants = sample(body,0,3.11)
            thighl = sample(core,-.44,3.61)
            thighr = sample(core,.44,3.61)
            pantmatch = .47*agree(thighl,thighr)+.265*(agree(pants,thighl)+agree(pants,thighr))
            shinl = sample(core,-.55,4.60)
            shinr = sample(core,.55,4.60)
            leftdiff = contrast(thighl,sample(body,-1.44,3.64))
            rightdiff = contrast(thighr,sample(body,1.44,3.64))
            legdiff = .61*np.minimum(leftdiff,rightdiff)+.39*np.maximum(leftdiff,rightdiff)
            pantcolor = (.37*pantmatch + .36*legdiff + .135*(agree(thighl,shinl)+agree(thighr,shinr)))
            vertical = (.73*structure+.41*upper+.32*legs+.75*pantcolor+.24*flare
                        +.25*isolate+.21*pure+.12*contrast(shin,pants)-1.81
                        +prior-border-.45*unit((yy+4.60*h-95.)/(1.40*h)))
            q = np.maximum(q,vertical)
        # A soft skin map can light up on wood, brick or sand. If the
        # purported face has no more redness than its surroundings, take
        # some of that cue back, without penalising a greyscale silhouette.
        excess = fs - .5*(np.minimum(lnear[3],rnear[3])+surround)
        q -= .13*facecolour*unit((.21-excess)/.21)
        # Scanning costs more false positives at very small scales: the
        # witness for a tiny coloured blob should also look like a body.
        q -= .13*tiny*(1.-.60*flare-.40*hands)
        best = max(best,float(q.max()))
        # Pool separately over 8-pixel cells for coarse corroboration.
        np.maximum.at(maxgrid,((ys//8)[:,None],(xs//8)[None,:]),q)
        if si%3 == 2:
            cache.clear()

    # Reward multiple strong detections far enough apart not to be the same
    # small head. Cap this term so that one face is sufficient by itself.
    j = int(maxgrid.argmax())
    gy,gx = j//12,j%12
    maxgrid[max(0,gy-2):min(12,gy+3),max(0,gx-2):min(12,gx+3)] = -3.
    extra = .12*float(unit((float(maxgrid.max())+.15)/.59))
    return best + .48*context + extra - .18


def predict(x):
    result = np.empty(len(x),dtype=np.int64)
    for i in range(len(x)):
        result[i] = int(image_score(x[i])>0.)
    return result
