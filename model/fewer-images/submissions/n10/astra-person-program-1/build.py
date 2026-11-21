from pathlib import Path
import sys

source = '''import math
import numpy as np


def unit(a):
    return np.clip(a, 0., 1.)


def ramp(a, low, high):
    return unit((a-low)/(high-low))


sizes = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
breaks = np.sqrt(np.array(sizes[1:]) * np.array(sizes[:-1])).astype(np.float32)
axis = np.arange(96)


def makegrid():
    # Anchor is the centre of the visible face, not the centre of the person.
    loc = []
    group = []
    last = 0
    for h in (2.9, 4.1, 5.8, 8.2, 11.6, 16.4, 23.2, 32.8, 46.4, 66., 88.):
        step = max(2, int(round(h*.18)))
        ys = np.arange(0, 96, step)
        xs = np.arange(0, 96, step)
        tile = max(14, int(h*1.2))
        for aspect in (.54, .80, 1.14):
            y, x = np.meshgrid(ys, xs, indexing='ij')
            x = x.ravel()
            y = y.ravel()
            loc.append(np.stack((x, y, np.full(x.size, h*aspect), np.full(x.size, h)), axis=1))
            group.append(last + (y//tile)*(1+95//tile) + x//tile)
        last += (1+95//tile)**2
    loc = np.concatenate(loc).astype(np.float32)
    group = np.concatenate(group)
    order = np.argsort(group, kind='stable')
    group = group[order]
    starts = np.concatenate((np.array([0]), np.where(group[1:]!=group[:-1])[0]+1))
    lens = np.diff(np.concatenate((starts, np.array([len(group)]))))
    return loc[order].T.copy(), starts, lens


anchors, starts, lens = makegrid()


def coarseindex(dx, dy, diameter):
    x, y, w, h = anchors
    i = np.clip(np.rint(x+dx*w).astype(np.int32), 0, 95)
    j = np.clip(np.rint(y+dy*h).astype(np.int32), 0, 95)
    level = np.searchsorted(breaks, diameter*np.sqrt(w*h)).astype(np.int32)
    return (level*96+j)*96+i


probes = [(0., .02, .66), (-.94, -.05, .47), (.94, -.05, .47),
          (0., -1.05, .54), (0., -.62, .38), (0., 1.85, .94),
          (-.82, 1.60, .70), (.82, 1.60, .70),
          (-1.04, .24, .59), (1.04, .24, .59),
          (-2.08, 1.75, .84), (2.08, 1.75, .84), (0., .14, .29),
          (0., -.23, .27), (0., -.05, .29)]
coarseplan = [coarseindex(x, y, d) for x, y, d in probes]


def field(im):
    r, g, b = im
    lum = .30*r+.58*g+.12*b
    a = (r-g)/(r+g+.055)
    v = (g-b)/(g+b+.055)
    # A weak adaptation to the colour of bright, nearly neutral surfaces.
    neutral = unit(1.-(np.abs(a)+np.abs(v))/.27) * lum**4
    mass = neutral.sum()+25.
    whitea = np.clip((neutral*a).sum()/mass, -.068, .068)
    whitev = np.clip((neutral*v).sum()/mass, -.085, .085)
    a = a-.80*whitea
    v = v-.80*whitev
    # Relative red excess and a correlated green-blue excess. Reduce
    # confidence in nearly neutral pixels, black pixels and saturated paint.
    red = ramp(a,.015,.074)*(1.-ramp(a,.235,.42))
    residual = v-.65*a
    hue = np.exp(-.5*(residual/np.where(residual>0., .080, .076))**2)
    skin = red*hue*ramp(lum, .038, .145)*(1.-.60*ramp(lum,.90,1.))
    feat = np.stack((lum,.68*(r-g),.55*(b-g),skin),axis=-1).astype(np.float32)
    return feat


def scalevolume(f):
    acc = np.zeros((97,97,4), np.float64)
    acc[1:,1:] = f.cumsum(axis=0,dtype=np.float64).cumsum(axis=1)
    vol = np.empty((len(sizes),96,96,4), np.float32)
    vol[0] = f
    for j in range(1,len(sizes)):
        k = sizes[j]
        lo = np.maximum(axis-(k-1)//2,0)
        hi = np.minimum(axis+1+k//2,96)
        dr = acc[hi]-acc[lo]
        area = (hi-lo)[:,None]*(hi-lo)[None,:]
        vol[j] = (dr[:,hi]-dr[:,lo])/area[:,:,None]
    return vol.reshape(-1,4)


def dist(a,b):
    return np.abs(a[...,0]-b[...,0])+np.abs(a[...,1]-b[...,1])+np.abs(a[...,2]-b[...,2])


def contrast(d):
    return d/(d+.105)


def selectbest(s):
    mx = np.maximum.reduceat(s, starts)
    ix = np.where(s == np.repeat(mx, lens))[0]
    return ix[np.searchsorted(ix, starts)]


def candidates(vol, abundance):
    f, l, r, up, cap, chest, sl, sr, al, ar, ol, oright, cheek, forehead, mid = [vol[p] for p in coarseplan]
    dc = contrast(dist(f,cap))
    dl = contrast(dist(f,l))
    dr = contrast(dist(f,r))
    du = contrast(dist(f,up))
    closed = .37*np.minimum(dl,dr)+.22*np.maximum(dl,dr)+.23*du+.18*contrast(dist(f,chest))
    surround = .29*(l[:,3]+r[:,3])+.22*up[:,3]+.20*chest[:,3]
    skin = .66*f[:,3]+.34*cheek[:,3]
    island = skin-.72*surround-.16*abundance
    hair = ramp(f[:,0]-cap[:,0],.005,.23)
    el = .55*dist(sl,al)+.45*dist(sl,ol)
    er = .55*dist(sr,ar)+.45*dist(sr,oright)
    nl = .064+.88*dist(sl,chest)
    nr = .064+.88*dist(sr,chest)
    tl = el/(el+nl)
    tr = er/(er+nr)
    torso = .55*np.minimum(tl,tr)+.45*(tl+tr)*.5
    eyes = ramp(.55*cheek[:,0]+.45*forehead[:,0]-mid[:,0],.005,.14)
    x,y,w,h = anchors
    vign = .026*(1.-np.abs(x-47.5)/48.)+.025*(1.-np.abs(y-35.)/62.)
    visible = ramp(99.-y, .5*h, 2.*h)
    top1 = .34*skin+.28*island+.14*closed+.11*hair+.10*torso+.03*eyes+vign
    top2 = .47*torso*visible+.28*closed+.10*dc+.08*skin+.07*eyes+vign
    top3 = .30*eyes+.21*hair+.20*closed+.12*skin+.09*island+.08*torso+vign
    return np.unique(np.concatenate((selectbest(top1), selectbest(top2), selectbest(top3))))


def geometry(v):
    return np.array(v,dtype=np.float32)


def points(a, probeset, tilt=0.):
    x,y,w,h = a
    dx,dy,d = probeset.T
    xx = x[:,None]+(dx+tilt*dy)*w[:,None]
    yy = y[:,None]+dy*h[:,None]
    diam = np.sqrt(w*h)[:,None]*d
    return xx, yy, diam


def sample(vol, xx, yy, diam):
    level = np.searchsorted(breaks, diam).astype(np.int32)
    # The even-width filters are centred half a pixel beyond the index.
    adjust = even[level]
    xx = np.clip(xx-adjust,0.,94.9999)
    yy = np.clip(yy-adjust,0.,94.9999)
    x0 = xx.astype(np.int32)
    y0 = yy.astype(np.int32)
    fx = (xx-x0).astype(np.float32)[...,None]
    fy = (yy-y0).astype(np.float32)[...,None]
    base = (level*96+y0)*96+x0
    p = vol[base]
    q = vol[base+96]
    return (1.-fy)*(p+fx*(vol[base+1]-p))+fy*(q+fx*(vol[base+97]-q))


even = geometry([.5 if s%2==0 else 0. for s in sizes])
faceprobes = geometry([
    (0.,-.29,.27), (-.23,.09,.28), (.23,.09,.28), (0.,.34,.24), (0.,0.,.57),
    (-.88,-.14,.41), (.88,-.14,.41), (-.81,.30,.38), (.81,.30,.38),
    (0.,-1.03,.47), (-.42,-.89,.37), (.42,-.89,.37), (0.,.99,.48),
    (0.,-.63,.39), (-.32,-.57,.31), (.32,-.57,.31),
    (-.49,-.23,.23), (.49,-.23,.23), (0.,.63,.27),
    (-.33,-.38,.22), (.33,-.38,.22), (-.33,.26,.24), (.33,.26,.24)])


def facialparts(vol, a, abund):
    f = sample(vol,*points(a,faceprobes)).transpose(1,0,2)
    forehead, cheekl, cheekr, chin, mid = f[:5]
    face = .21*forehead+.23*(cheekl+cheekr)+.11*chin+.22*mid
    spread = .38*dist(cheekl,cheekr)+.24*dist(forehead,mid)+.19*(dist(cheekl,mid)+dist(cheekr,mid))
    skin = .72*face[:,3]+.28*np.minimum(np.maximum(forehead[:,3],chin[:,3]), np.maximum(cheekl[:,3],cheekr[:,3]))
    # A face may be in profile; allow one of the side samples to be hair, neck or cheek.
    surrounds = [.54*f[5]+.46*f[7], .54*f[6]+.46*f[8], .48*f[9]+.26*(f[10]+f[11]), f[12]]
    left,right,above,below = surrounds
    surrounds2 = np.stack((left[:,3],right[:,3],above[:,3],below[:,3]))
    surr = .80*surrounds2.mean(axis=0)+.20*surrounds2.min(axis=0)
    island = ramp(skin-.76*surr-.16*abund,-.07,.68)
    def rim(outer):
        d = dist(face,outer)
        return d/(d+.065+spread*.80)
    ll,rr,tt,bb = [rim(s) for s in surrounds]
    ring = .24*np.minimum(ll,rr)+.16*(ll+rr)+.24*tt+.20*bb
    scalp = .48*f[13]+.26*(f[14]+f[15])
    temple = .5*(f[16]+f[17])
    dark = .65*(forehead[:,0]-scalp[:,0])+.35*(mid[:,0]-temple[:,0])
    hair = ramp(dark, .005,.20)
    # A cap of hair should have an upper boundary as well as a hairline.
    hair *= .65+.35*contrast(dist(scalp,above))
    continuous = unit(1.-spread/.26)
    skinshape = skin*(.68+.32*continuous)
    # Support for an uncoloured silhouette: the head has a boundary at
    # both temples and over the crown. Use the colour of the cap itself.
    head = .56*face+.44*scalp
    ld = contrast(.56*dist(left,head)+.44*dist(left,f[16]))
    rd = contrast(.56*dist(right,head)+.44*dist(right,f[17]))
    top = contrast(.7*dist(scalp,above)+.3*dist(head,above))
    outline = .46*np.minimum(ld,rd)+.20*np.maximum(ld,rd)+.34*top
    return face, skinshape, island, ring, hair, outline


# Face appearance bank. Each row is one separate analytical face, with
# its own normalisation. No image or population of images supplies weights.

def facebank():
    s = np.linspace(-.48,.48,13)
    yy,xx = np.meshgrid(s,s,indexing='ij')
    keep = (xx/.49)**2 + ((yy+.01)/.51)**2 < 1.
    x = xx[keep]
    y = yy[keep]
    probeset = np.stack((x,y,0.*x+.116),axis=1).astype(np.float32)
    # An orthogonal illumination basis, solely from the sample coordinates.
    x0 = x-x.mean()
    y0 = y-y.mean()
    basis = np.stack((x0/np.sqrt((x0*x0).sum()), y0/np.sqrt((y0*y0).sum()))).astype(np.float32)
    bank = []
    for angle in (-.29,0.,.29):
        for yaw in (-.11,0.,.11):
            for height in (-.19,-.095,.0):
                for spacing in (.20,.265):
                    for smile in (False,True):
                        xx = math.cos(angle)*x+math.sin(angle)*y*1.25
                        yy = -math.sin(angle)*x*.8+math.cos(angle)*y-height
                        def blob(u,v,sx,sy):
                            return np.exp(-.5*((xx-u)/sx)**2-.5*((yy-v)/sy)**2)
                        t = -.78*(blob(yaw-spacing,0.,.107,.055)+blob(yaw+spacing,0.,.107,.055))
                        t -= .46*(blob(yaw-spacing,-.067,.12,.034)+blob(yaw+spacing,-.067,.12,.034))
                        t += .25*blob(yaw*1.2,.068,.063,.12)
                        t += .24*(blob(yaw-.23,.157,.15,.079)+blob(yaw+.23,.157,.15,.079))
                        t += .25*blob(yaw,-.21,.23,.10)-.26*blob(1.4*yaw,.216,.084,.036)
                        t -= (.67 if smile else .42)*blob(yaw*1.18,.343,.16,.053 if smile else .025)
                        t += .17*blob(yaw,.437,.14,.062)
                        t = t-t.mean()
                        t -= (t@basis.T)@basis
                        bank.append(t/np.sqrt(np.sum(t*t)))
    return probeset, basis, np.array(bank,dtype=np.float32).T.copy()


facegrid, lightbasis, patterns = facebank()


def facestructure(vol, a):
    res = np.zeros(len(a[0]), np.float32)
    sel = np.where((a[3]>=5.5)&(a[2]>=4.3))[0]
    if len(sel):
        xx, yy, diam = points(a[:,sel], facegrid)
        patch = sample(vol,xx,yy,diam)[:,:,0].copy()
        patch -= patch.mean(axis=1,keepdims=True)
        patch -= (patch@lightbasis.T)@lightbasis
        norm = np.sqrt((patch*patch).sum(axis=1)+patch.shape[1]*.021**2)
        sim = (patch@patterns).max(axis=1)/norm
        # Very small heads cannot resolve independent eyes and a mouth.
        detail = .49+.51*ramp(np.minimum(a[2,sel],a[3,sel]),4.,13.)
        res[sel] = ramp(sim,.23,.76)*detail
    return res


bodygeom = geometry([
    (0.,1.46,.70), (0.,2.22,.82), (0.,2.89,.70),
    (-.70,1.40,.54), (.70,1.40,.54), (-.76,2.15,.59), (.76,2.15,.59),
    (-.74,2.78,.55), (.74,2.78,.55),
    (-1.04,.26,.54), (1.04,.26,.54), (-1.08,1.39,.51), (1.08,1.39,.51),
    (-2.06,1.47,.69), (2.06,1.47,.69), (-2.13,2.33,.77), (2.13,2.33,.77),
    (-.54,4.38,.68), (.54,4.38,.68), (-1.70,4.40,.82), (1.70,4.40,.82),
    (-1.39,2.38,.43), (1.39,2.38,.43), (-1.48,3.22,.48), (1.48,3.22,.48),
    (-1.34,4.02,.48), (1.34,4.02,.48), (-.66,2.91,.44), (.66,2.91,.44),
    (-2.16,2.10,.43), (2.16,2.10,.43), (-2.21,1.00,.40), (2.21,1.00,.40),
    (0.,6.10,1.0), (-2.02,6.10,.90), (2.02,6.10,.90)])


def boundary(d, clutter):
    # Strength both in absolute colour units and relative to texture inside.
    return .66*d/(d+.065+.98*clutter) + .34*ramp(d-.7*clutter,.015,.25)


def bodyparts(vol, a, face, abundance):
    x,y,w,h = a
    visible = ramp(96.-y,.61*h,2.15*h)
    legsvisible = ramp(96.-y,3.2*h,5.*h)
    best = np.zeros(len(x),np.float32)
    besthands = np.zeros(len(x),np.float32)
    tie = np.zeros(len(x),np.float32)
    for tilt, width in ((0.,1.),(-.32,1.06),(.32,1.06),(0.,.76),(0.,1.32)):
        xx, yy, d = points((x,y,w*width,h),bodygeom,tilt)
        b = sample(vol,xx,yy,d).transpose(1,0,2)
        chest = .38*b[0]+.42*b[1]+.20*b[2]
        axial = .5*(dist(b[0],b[1])+dist(b[1],b[2]))
        sides = []
        shoulders = []
        for j in (0,1):
            near = .44*b[3+j]+.38*b[5+j]+.18*b[7+j]
            outside = .51*dist(b[3+j],b[13+j])+.49*dist(b[5+j],b[15+j])
            inside = .37*dist(near,chest)+.35*dist(b[3+j],b[5+j])+.28*axial
            sides.append(boundary(outside,inside))
            cover = .62*dist(b[9+j],b[11+j])+.38*dist(b[9+j],b[3+j])
            dirt = .47*dist(b[3+j],b[11+j])+.33*dist(b[3+j],b[0])+.20*axial
            shoulders.append(boundary(cover,dirt))
        side = .65*np.minimum(sides[0],sides[1])+.35*np.maximum(sides[0],sides[1])
        shoulder = .62*np.minimum(shoulders[0],shoulders[1])+.38*np.maximum(shoulders[0],shoulders[1])
        ld = .5*(dist(b[17],b[19])+dist(b[18],b[20]))
        legs = boundary(ld,dist(b[17],b[18]))
        foot = boundary(.5*(dist(b[33],b[34])+dist(b[33],b[35])),dist(b[33],.5*(b[17]+b[18])))
        legs = .78*legs+.22*foot
        colchange = contrast(dist(face,chest))
        tor = .42*side+.40*shoulder+.12*np.minimum(side,shoulder)+.06*colchange
        tor += .095*(legs-.43)*legsvisible
        tor = visible*tor+(1.-visible)*.35
        # Look on the arms and in front of the shirt for smaller patches
        # of similar flesh, but not if they just continue the background.
        extrem = b[21:33]
        match = np.exp(-3.1*np.abs(extrem[:,:,1]-face[None,:,1])-3.6*np.abs(extrem[:,:,2]-face[None,:,2])-1.4*np.abs(extrem[:,:,0]-face[None,:,0]))
        hand = extrem[:,:,3]*match
        hv = unit((95.-yy[:,21:33].T)/np.maximum(h*.5,1.)+.5)
        hv *= unit((47.5-np.abs(xx[:,21:33].T-47.5))/np.maximum(w*.5,1.)+.5)
        hand *= hv
        lh = hand[::2].max(axis=0)
        rh = hand[1::2].max(axis=0)
        other = (.70*np.maximum(lh,rh)+.30*np.minimum(lh,rh))
        nuisance = .42*chest[:,3]+.31*abundance+.135*(b[13,:,3]+b[14,:,3])
        hand = ramp(other-.77*nuisance,.04,.70)*visible
        merit = tor+.08*hand
        better = merit>tie
        tie = np.maximum(merit,tie)
        best = np.where(better,tor,best)
        besthands = np.where(better,hand,besthands)
    return best,besthands,visible


# Short contour elements with a pair of samples straddling the expected
# boundary. Neighbouring normal derivatives should point to the same edge.
def arcplan(probeset):
    plan = []
    for x,y,nx,ny in probeset:
        for t in (-3.,-1.,1.,3.):
            plan.append((x+t*nx,y+t*ny,.19))
    return geometry(plan)


sidearc = [(.43,-.40,.135,-.055), (.53,-.13,.14,0.), (.49,.15,.14,.02), (.37,.38,.12,.065)]
leftarc = [(-x,y,-nx,ny) for x,y,nx,ny in sidearc]
headarc = arcplan(leftarc+sidearc+[(-.32,-.67,-.075,-.12),(-.11,-.77,-.023,-.13),(.11,-.77,.023,-.13),(.32,-.67,.075,-.12)])
sidearc = [(.68,.73,.071,-.16),(1.12,.91,.095,-.15),(1.40,1.11,.14,-.10),
           (1.54,1.46,.18,-.03),(1.60,1.86,.18,-.017),(1.59,2.27,.18,.014),(1.51,2.70,.18,.037)]
bodyarc = arcplan([(-x,y,-nx,ny) for x,y,nx,ny in sidearc]+sidearc)


def edgelinks(vol, a, plan, tilt):
    xx,yy,d = points(a,plan,tilt)
    patches = sample(vol,xx,yy,d).reshape(len(a[0]),-1,4,4)
    u,v,w,z = patches[:,:,0],patches[:,:,1],patches[:,:,2],patches[:,:,3]
    # Relative centre-surround derivative rejects both gradual ramps and
    # isolated flecks. A soft agreement factor also checks edge polarity.
    edge = w[:,:,:3]-v[:,:,:3]
    wide = z[:,:,:3]-u[:,:,:3]
    mag = np.abs(edge).sum(axis=2)
    core = boundary(mag,.5*(dist(u,v)+dist(w,z)))
    same = np.sum(np.minimum(np.abs(edge),np.maximum(np.sign(edge)*wide,0.)),axis=2)/(mag+.015)
    s = core*(.68+.32*same)
    xm = xx[:,1::4]
    ym = yy[:,1::4]
    s *= ((xm>=0.)&(xm<95.)&(ym>=0.)&(ym<95.))
    return s


def bestarc(vol, a, ishead):
    res = np.zeros(len(a[0]),np.float32)
    if ishead:
        for scale,shift in ((1.,0.),(.83,.015),(1.19,-.05)):
            x,y,w,h = a
            z = edgelinks(vol,(x,y+shift*h,w*scale,h*scale),headarc,0.)
            l = z[:,:4].mean(axis=1)
            r = z[:,4:8].mean(axis=1)
            t = z[:,8:].mean(axis=1)
            res = np.maximum(res,.30*np.minimum(l,r)+.22*np.maximum(l,r)+.31*t+.17*np.minimum(t,np.maximum(l,r)))
    else:
        for scale,tilt in ((1.,0.),(.79,0.),(1.27,0.),(1.,-.32),(1.,.32)):
            x,y,w,h = a
            z = edgelinks(vol,(x,y,w*scale,h),bodyarc,tilt)
            sh = .25*(z[:,:3].mean(axis=1)+z[:,7:10].mean(axis=1))+.50*np.minimum(z[:,:3].mean(axis=1),z[:,7:10].mean(axis=1))
            side = .25*(z[:,3:7].mean(axis=1)+z[:,10:].mean(axis=1))+.50*np.minimum(z[:,3:7].mean(axis=1),z[:,10:].mean(axis=1))
            res = np.maximum(res,.40*sh+.40*side+.20*np.minimum(sh,side))
    return res


def scene(im, f):
    r,g,b = im
    # A field, water or snow provides modest context for a small upright
    # subject. Do not treat a green, blue or white image as a detection.
    grass = ramp(g-r,-.015,.080)*ramp(g-b,.045,.18)*ramp(g,.16,.38)
    water = ramp(b-r,.04,.23)*ramp(g-r,.018,.12)
    snow = ramp(f[:,:,0],.66,.91)*(1.-ramp(r-b,.015,.12))
    sport = max(ramp(grass[43:].mean(),.19,.63),.85*ramp(water[45:].mean(),.31,.71),.70*ramp(snow[46:].mean(),.32,.76))
    chrom = np.maximum(np.maximum(r,g),b)-np.minimum(np.minimum(r,g),b)
    accents = ramp(chrom,.31,.64)*(1.-grass)*(1.-water)*(1.-f[:,:,3])
    accent = ramp(accents[17:83,8:88].mean(),.007,.051)
    return sport*(.55+.45*accent)


def classify(im):
    f = field(im)
    abundance = f[:,:,3].mean()
    vol = scalevolume(f)
    picks = candidates(vol,abundance)
    a = anchors[:,picks]
    face,skin,island,ring,hair,outline = facialparts(vol,a,abundance)
    eyes = facestructure(vol,a)
    torso,hands,visible = bodyparts(vol,a,face,abundance)
    x,y,w,h = a
    # Scoring paths can replace missing evidence: frontal facial structure,
    # exposed skin with hair/arms, or a largely achromatic clothed figure.
    color = .275*skin+.157*island+.108*ring+.099*hair+.177*torso+.091*hands+.093*eyes
    portrait = .303*eyes+.197*skin+.106*island+.127*ring+.104*hair+.128*torso+.035*hands
    figure = .395*torso+.225*outline+.097*ring+.080*hair+.085*skin+.058*hands+.060*eyes
    # An isolated single-colour head, e.g. a helmet or the back of the
    # head, needs unusually clear shoulders and flanks for this path.
    achrom = .595*torso+.32*outline+.085*ring
    core = np.maximum.reduce((color-.622,portrait-.605,figure-.658,achrom-.735))
    size = np.sqrt(w*h)
    tiny = 1.-ramp(size,2.5,8.)
    nearx = unit(1.-np.abs(x-47.5)/47.5)
    neary = unit(1.-np.abs(y-33.)/52.)
    prior = .029*(nearx-.40)+.041*(neary-.47)-.031*tiny
    prior -= .055*ramp(y,68.,94.)*ramp(21.-h,0.,16.)
    prior -= .032*ramp(abundance,.27,.68)
    prior += .025*scene(im,f)*visible
    initial = core+prior
    count = min(80,len(x))
    idx = np.argsort(initial,kind='stable')[-count:]
    edgeh = bestarc(vol,a[:,idx],True)
    edgeb = bestarc(vol,a[:,idx],False)
    # Contours verify the coarse region proposals. Do not require shoulder
    # or jawline evidence for a tightly cropped frontal portrait.
    extent = ramp(size[idx],2.8,7.)
    headshift = .085*(edgeh-.29)*(.44+.56*extent)
    bodyshift = .13*(edgeb-.30)*visible[idx]*(.44+.56*extent)
    appearance = np.maximum(color[idx]-.622,portrait[idx]-.605)+.55*headshift+.20*bodyshift
    silhouette = np.maximum(figure[idx]-.658,achrom[idx]-.735)+headshift+bodyshift
    core = np.maximum(appearance,silhouette)+prior[idx]
    x,y,w,h = a[:,idx]
    j = np.argmax(core)
    # A second spatially separate, comparably supported head helps a group
    # photograph; repeated windows on the same head don't count again.
    sep = ((x-x[j])/(.85*(w+w[j])+2.))**2+((y-y[j])/(.73*(h+h[j])+2.))**2
    second = np.max(np.where(sep>1.,core,-1.))
    evidence = core[j]+.028*ramp(second,-.085,.025)
    return int(evidence>0.)


def predict(x):
    return np.array([classify(im) for im in x],dtype=np.int64)
'''

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / 'solution.py').write_text(source)
