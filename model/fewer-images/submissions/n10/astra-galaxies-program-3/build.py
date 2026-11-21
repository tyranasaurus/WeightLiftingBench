import sys
from pathlib import Path

source = '''import numpy as np
import math

# The radial coordinate of an elliptical isophote has a particularly useful
# harmonic: 2 <log(r) exp(2 i theta)> = (1-q)/(1+q) exp(2 i PA).
pi = math.pi
angles = np.arange(160)*2*pi/160
ct = np.cos(angles)
st = np.sin(angles)
spin = np.exp(2j*angles)
rs = np.arange(161)*0.5
Y,X = np.mgrid[0:128,0:128]
edge = (X-63.5)**2 + (Y-63.5)**2 > 54**2
freq = np.fft.fftfreq(128)[:,None]**2 + np.fft.rfftfreq(128)[None,:]**2

def kernel(s):
    return np.exp(-2*pi*pi*s*s*freq)

small = kernel(0.85)
medium = kernel(1.7)
broad = kernel(3.2)

def clip(t):
    return np.clip(t,0.,1.)

def ramp(x,low,high):
    return clip((x-low)/(high-low))

def sample(im,x,y):
    x = np.clip(x,0.,126.99999)
    y = np.clip(y,0.,126.99999)
    i = x.astype(np.int32)
    j = y.astype(np.int32)
    a,b = x-i,y-j
    return (1-b)*((1-a)*im[j,i]+a*im[j,i+1])+b*((1-a)*im[j+1,i]+a*im[j+1,i+1])

def mad(im):
    b = im[edge]
    return max(0.0008,1.4826*np.median(np.abs(b-np.median(b))))

def nucleus(sm):
    # Prefer the catalog position unless a nearby, better resolved peak is evident.
    w = np.maximum(sm,0)*np.exp(-((X-63.5)**2+(Y-63.5)**2)/(2*8**2))
    j,i = np.unravel_index(np.argmax(w),w.shape)
    x,y = float(i),float(j)
    if w[j,i]<=0.0001:
        return 63.5,63.5
    for width in (3.,2.4,2.,2.,2.):
        rr = (X-x)**2+(Y-y)**2
        a = np.exp(-rr/(2*width**2))*np.maximum(sm-sm[j,i]*0.18,0)**2
        n = a.sum()+1e-18
        x,y = (a*X).sum()/n,(a*Y).sum()/n
    return x,y

def contours(s,t,cx,cy,noise,ns):
    xp = cx+ct[:,None]*rs
    yp = cy+st[:,None]*rs
    prof = sample(s,xp,yp)
    wide = sample(t,xp,yp)
    peak = max(prof[0,0],3*noise,0.001)
    levels = np.array([.82,.68,.54,.42,.32,.235,.17,.12,.084,.056,.036,.022])
    # Use the somewhat longer smoothing length only on the faint outskirts.
    mix = ramp(rs,4,12)*(.75*ramp(.30-prof/peak,.0,.23))
    prof = (1-mix)*prof+mix*wide
    rr = []
    for lev in levels:
        threshold = max(lev*peak,min(peak*.6,3.0*ns))
        below = prof<threshold
        hit = np.argmax(below,axis=1)
        hit = np.where(np.any(below,axis=1),hit,len(rs)-1)
        hit = np.maximum(hit,1)
        idx = np.arange(len(hit))
        prev = prof[idx,hit-1]
        frac = clip((prev-threshold)/np.maximum(prev-prof[idx,hit],.000001))
        r = np.maximum(.45,(hit-1+frac)*.5)
        # Reject narrow spikes in just a few rays, usually from a field source.
        r = np.median(np.stack([np.roll(r,k) for k in (-2,-1,0,1,2)]),axis=0)
        rr.append(r)
    rr = np.array(rr)
    lr = np.log(rr)
    z = 2*np.mean(lr*spin,axis=1)
    e = np.minimum(abs(z),.87)
    q = (1-e)/(1+e)
    angle = .5*np.angle(z)
    a = np.exp(lr.mean(axis=1))*(1+q)/(2*q)
    # Interpolate the same contours at equal eccentric anomaly, so diagnostics
    # for narrow galaxies don't privilege the part around the minor axis.
    dx,dy = ct[None,:],q[:,None]*st[None,:]
    direction = np.arctan2(dy,dx)+angle[:,None]
    ai = (direction*len(ct)/(2*pi))%len(ct)
    ji = ai.astype(int)%len(ct)
    f = ai-np.floor(ai)
    r = (1-f)*np.take_along_axis(rr,ji,axis=1)+f*np.take_along_axis(rr,(ji+1)%len(ct),axis=1)
    rad = r/np.sqrt(dx*dx+dy*dy)
    rad = rad/np.maximum(.1,np.median(rad,axis=1)[:,None])
    norm = np.log(np.clip(rad,.25,4))
    opposite = np.roll(norm,len(ct)//2,axis=1)
    odd = .5*abs(norm-opposite)
    asym = .55*np.mean(odd,axis=1)+.45*np.quantile(odd,.85,axis=1)
    box = 2*np.mean(norm*np.cos(4*angles),axis=1)
    # The frequency-2 estimate treats an ellipse with pointed ends as rounder
    # than its visible extent. Use the lengths of the four axial intercepts too.
    win = len(ct)//32
    def intercept(k):
        ids = (k+np.arange(-win,win+1))%len(ct)
        return np.median(r[:,ids],axis=1)
    major = .5*(intercept(0)+intercept(len(ct)//2))
    minor = .5*(intercept(len(ct)//4)+intercept(3*len(ct)//4))
    axial = np.clip(minor/np.maximum(.5,major),.05,1.)
    # One-sided protrusions occupy a minority of the circumference. Smooth
    # in angle before looking for an extended, unmatched tidal plume.
    left = norm-opposite
    left = sum(np.roll(left,k,axis=1) for k in range(-5,6))/11.
    plume = np.quantile(abs(left),.92,axis=1)
    # A tidal tail can occupy a narrow angle even though it is many seeing
    # disks long. The outer radial boundary should move in the same directions
    # if this is a mere displacement of the core, not a shell or tail.
    halo = np.mean(lr[-3:],axis=0)-np.mean(lr[4:7],axis=0)
    halo -= np.roll(halo,len(ct)//2)
    halo = sum(np.roll(halo,k) for k in range(-3,4))/7.
    growth = np.quantile(abs(halo),.965)
    return peak,z,q,angle,a,asym,box,axial,plume,growth

def diskframe(z,a):
    # Use a range of outer surface brightnesses for the disk projection.
    w = np.array([1,2,3,3,2],dtype=float)
    outer = (w*z[6:11]).sum()/w.sum()
    phi = .5*np.angle(outer)
    q = max(.085,(1-abs(outer))/(1+abs(outer)))
    size = np.clip(np.mean(a[7:10]),3.,78.)
    return outer,q,phi,size

logr = np.linspace(np.log(.075),np.log(1.55),61)
relative = np.exp(logr)
pa = np.arange(128)*2*pi/128
pc,ps = np.cos(pa),np.sin(pa)

def trimmed(a,f=.12):
    lo,hi = np.quantile(a,[f,1-f],axis=1)
    return np.clip(a,lo[:,None],hi[:,None])

def rings(rgb,lum,cx,cy,q,phi,size,noise):
    xp = size*(np.cos(phi)*pc-q*np.sin(phi)*ps)
    yp = size*(np.sin(phi)*pc+q*np.cos(phi)*ps)
    x = cx+relative[:,None]*xp
    y = cy+relative[:,None]*yp
    val = trimmed(sample(lum,x,y),.075)
    flux = np.mean(val,axis=1)
    coeff = np.fft.rfft(val,axis=1)[:,:11]/(len(pc)*.5)
    coeff = coeff/np.maximum(flux[:,None]+3*noise,4*noise)
    # Angular median colour profiles don't assign the colours of stars to a disk.
    red = np.median(sample(rgb[0],x,y),axis=1)
    blue = np.median(sample(rgb[2],x,y),axis=1)
    colour = (red-blue)/np.maximum(red+blue+2*noise,5*noise)
    return coeff,colour,flux

def region(low,high):
    return (relative>=low)&(relative<=high)

middle = region(.28,.85)
inner = region(.12,.32)
skirt = region(.65,1.15)

def meanmag(arr,mask,m):
    return np.mean(abs(arr[mask,m]))

# Log spiral filter bank: opposite windings, multiple arm counts and pitches.
# Correlate after subtracting the stationary angular pattern in each annulus
# packet. A bar or a misestimated inclination by itself has zero handedness.
slopes = np.array([.7,1.1,1.55,2.1,2.8,3.7,4.8,6.3,8.2])
waves = []
for mid,width in ((.42,.33),(.64,.32),(.84,.27),(.55,.50)):
    window = np.exp(-.5*(np.log(relative/mid)/width)**2)*region(.13,1.42)
    window = window/window.sum()
    group = []
    for m in (2,3,4):
        v = np.exp(1j*logr*slopes[:,None]*m)
        v = v-np.sum(window*v,axis=1)[:,None]
        group.append(v*window/np.maximum(.6,np.sqrt(np.sum(window*abs(v)**2,axis=1)))[:,None])
    waves.append(group)

def spirals(coeff):
    tight,loose,total,signal = 0.,0.,0.,0.
    for group in waves:
        for mi,m in enumerate((2,3,4)):
            v = group[mi]
            f = abs(v@coeff[:,m])
            b = abs(v.conj()@coeff[:,m])
            # Discount bilateral spectral power (a radial change in amplitude,
            # rather than a radial change of position angle).
            power = np.maximum(f,b)-.75*np.minimum(f,b)
            weight = 1. if m==2 else .85
            loose = max(loose,weight*np.max(power[:4]))
            tight = max(tight,weight*np.max(power[4:]))
            signal = max(signal,weight*np.max(np.abs(f-b)))
            total = max(total,weight*np.max(power))
    return loose,tight,signal

def armcurves(coeff):
    # Follow local maxima in a radially high-passed disk. Unlike a whole-ring
    # harmonic, a partial arc need not have a partner exactly pi radians away.
    c = coeff[:,1:9].copy()
    # stationary deviations from the projection and a slowly changing bulge
    smooth = np.zeros_like(c)
    wt = 0.
    for j in range(-10,11):
        w = np.exp(-.5*(j/5.)**2)
        smooth += w*c[np.clip(np.arange(len(c))+j,0,len(c)-1)]
        wt += w
    c -= smooth/wt
    m = np.arange(1,9)
    best = np.zeros((2,len(slopes)))
    for ctr in (.43,.65,.90):
        win = np.exp(-.5*(np.log(relative/ctr)/.26)**2)*region(.18,1.45)
        win /= win.sum()
        for idx,sgn in enumerate((-1,1)):
            # The phase is in radians per e-fold of radius; inverse transform
            # lets the arm segment start at any azimuth, not just a grid axis.
            v = np.exp(1j*sgn*slopes[:,None,None]*logr[None,:,None]*m)
            f = np.sum(v*c*win[None,:,None],axis=1)
            f = np.concatenate([np.zeros((len(slopes),1)),f],axis=1)
            az = np.fft.irfft(f,n=48)*24
            best[idx] = np.maximum(best[idx],np.max(az,axis=1))
    # A non-chiral residual responds similarly to both signs of pitch.
    power = np.max(best,axis=0)-.5*np.min(best,axis=0)
    return np.max(power[:5]),np.max(power[5:])

def debris(b,cx,cy,size,q,phi,peak,noise):
    # Inspect low surface brightness structure outside the main galaxy as
    # well as at its last isophote. Remove the symmetric background with a
    # half-turn. Bright compact sources have the wrong contrast and shape.
    dx,dy = X-cx,Y-cy
    u,v = dx*np.cos(phi)+dy*np.sin(phi),-dx*np.sin(phi)+dy*np.cos(phi)
    radius = np.sqrt(u*u+(v/q)**2)/size
    denom = max(.001,peak)
    low = max(noise*1.75,.009*denom)
    faint = ramp(b,low,2.2*low)*(1-ramp(b/denom,.13,.30))
    rev = sample(b,cx-dx,cy-dy)
    dif = b-rev
    # A centred ridge of debris can still have two-fold symmetry (a S warp),
    # hence measure bilateral off-axis extent separately in the shape code.
    excess = faint*ramp(dif/denom,.006,.042)*ramp(dif/(low+np.maximum(0,b)),.12,.60)
    excess *= ramp(radius,.62,1.02)*(1-ramp(radius,1.65,3.0))
    # Template for a stream: adjacent patches on a ray must both have diffuse
    # asymmetric light, not just a single compact companion. Allow a bend.
    d = np.linspace(.65,1.95,24)
    l = size/np.sqrt((ct*np.cos(phi)+st*np.sin(phi))**2+((-ct*np.sin(phi)+st*np.cos(phi))/q)**2)
    sx,sy = cx+ct[:,None]*l[:,None]*d,cy+st[:,None]*l[:,None]*d
    ex = sample(excess,sx,sy)
    signal = sample(b,sx,sy)
    # A tidal protrusion is contiguous with the target at low brightness.
    # Suppress diffuse light beyond a sky gap or beyond a second point peak.
    continuity = np.minimum.accumulate(signal,axis=1)
    ex *= ramp(continuity,low*.60,low*1.75)
    ex *= 1.-ramp((signal-continuity)/np.maximum(signal,low),.30,.73)
    ray = np.mean(ex,axis=1)
    ray = sum(np.roll(ray,k) for k in range(-3,4))/7.
    level = np.max(ray)
    area = np.sum(excess)/max(20.,pi*size*size*q)
    return ramp(level,.20,.59)*(.45+.55*ramp(area,.016,.09))

def companions(s,t,b,rgb,x,y,peak,a,q,phi,noise):
    # Candidate secondary nuclei must be peaked on the seeing scale, extended
    # beyond a point source, and significant relative to the primary galaxy.
    rr = (X-x)**2+(Y-y)**2
    refl = sample(t,2*x-X,2*y-Y)
    pairres = t-refl
    mask = (t>max(.10*peak,5*noise))&(rr>2.8**2)&(rr<58**2)
    # A local maximum in the diffuse light is sufficient, but not necessary:
    # closely blended nuclei can be resolved only in the high-frequency light.
    coreim = s-.7*b
    fine = mask&(coreim>.13*peak)&(rr<(a+4.)**2)
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)):
        mask &= t>=np.roll(t,(dy,dx),axis=(0,1))
        fine &= coreim>=np.roll(coreim,(dy,dx),axis=(0,1))
    ys,xs = np.where(mask|fine)
    order = np.argsort(-(t[ys,xs]/(1+rr[ys,xs]/(a+8)**2)))[:24]
    best = 0.
    rads = np.array([1.5,3.,4.5,6.,8.,11.,15.])
    ux,uy = ct[::4,None]*rads,st[::4,None]*rads
    centralcol = (sample(rgb[0],x,y)-sample(rgb[2],x,y))/(peak+.02)
    for j in order:
        xx,yy = xs[j],ys[j]
        dx,dy = xx-x,yy-y
        dist = np.hypot(dx,dy)
        rad = sample(s,xx+ux,yy+uy)
        prof = np.quantile(rad,.4,axis=0)
        loc = np.minimum.accumulate(prof)
        floor = max(0.,min(loc[-1],loc[3]))
        h = s[yy,xx]-floor
        if h<.09*peak:
            continue
        # Relative wing strength at radii outside the survey PSF.
        pr = (prof-floor)/max(.001,h)
        thick = .52*ramp(pr[1],.16,.48)+.48*ramp(pr[2],.045,.29)
        extend = np.interp(.14,pr[::-1].clip(0,1) if np.all(np.diff(pr)<=0) else (loc[::-1]-floor)/h,rads[::-1])
        lines = np.linspace(0,1,max(4,int(dist)+1))
        along = sample(t,x+dx*lines,y+dy*lines)
        valley = max(0.,along.min())
        prom = (t[yy,xx]-valley)/max(.004,t[yy,xx])
        direc = np.arctan2(dy,dx)-phi
        proj = a/np.sqrt(np.cos(direc)**2+(np.sin(direc)/q)**2)
        near = dist/(max(4.,.65*proj+.35*a*np.sqrt(q))+extend+2.)
        height = ramp(h/peak,.10,.55)
        delta = (rgb[0,yy,xx]-rgb[2,yy,xx])/(s[yy,xx]+.03)
        same = 1.-ramp(abs(centralcol-delta),.24,.78)
        strange = ramp(abs(delta),.66,1.35)
        # Separate a compact blue knot superposed on an arm from a second core.
        knot = ramp(centralcol-delta,.22,.7)*ramp(1.25-near,0,.7)
        score = (.32+.68*thick)*(.48+.52*height)*(.78+.22*same)*(1-.58*strange)*(1-.5*knot)
        score *= (.38+.62*ramp(prom,.035,.37))*(1.-ramp(near,.93,2.65))
        # A bridge and two separately resolved peaks are evidence of association.
        bridge = valley/max(.004,min(peak,t[yy,xx]))
        score += .13*ramp(bridge,.055,.32)*ramp(prom,.05,.27)*thick*height*(1-ramp(near,1.4,2.4))
        # A bright, resolved galaxy is still an unrelated field galaxy if the two
        # halos do not overlap. Use the flux between nuclei, compared to an
        # off-pair local sky, instead of just the distance between their peaks.
        halo = ramp(valley/max(peak*.018,2*noise),.65,2.8)
        halo = max(halo,ramp(bridge,.025,.13))
        overlap = 1-ramp(near,.7,1.6)
        connection = .15+.85*max(halo,overlap*.87)
        # Enforce a transverse size comparable to the target, not merely a
        # visible peak. Gaussian-like PSF cores have much smaller wing areas.
        objarea = np.sum(np.maximum(pr,0)*rads*np.gradient(rads))
        relarea = objarea/max(2.,.5*(a*.43)**2*q)
        mass = ramp(relarea*h/peak,.055,.39)
        colpen = 1.-.62*ramp(abs(delta-centralcol*0.7),.38,.92)
        score *= connection*(.46+.54*mass)*colpen
        # A companion is a distinct local concentration above the primary's
        # disk, not a maximum produced by knots in an otherwise bilateral arm.
        wp = np.maximum(0,pairres)*np.exp(-((X-xx)**2+(Y-yy)**2)/(2*max(2.,extend*.50)**2))
        isolated = pairres[yy,xx]/max(.004,t[yy,xx])
        extflux = wp.sum()/max(1.,peak*.25*pi*a*a*q)
        locallyred = (sample(rgb[0],xx,yy)-sample(rgb[2],xx,yy))/(h+floor+.01)
        armknot = ramp(centralcol-locallyred,.14,.48)*(1-ramp(near,.8,1.45))
        score *= (.45+.55*ramp(isolated,.14,.71))*(.65+.35*ramp(extflux,.065,.30))*(1.-.4*armknot)
        # Confirm a close pair by comparison with *the primary's core*, rather
        # than demanding a deep saddle in the shared, low-frequency envelope.
        hp = coreim[yy,xx]
        ridge = sample(coreim,x+dx*lines,y+dy*lines)
        hp0 = max(.01,ridge[0]); dip = (hp-ridge.min())/max(.01,hp)
        corepair = ramp(hp/hp0,.18,.72)*ramp(dip,.12,.70)*ramp(dist,3.,6.)
        corepair *= (1-ramp(near,.80,1.45))*ramp(bridge,.065,.34)*(.5+.5*thick)
        corepair *= (.48+.52*same)*(1-.55*armknot)*(1-.6*strange)
        # Measure the spectrum of just the core after subtracting its diffuse
        # pedestal. A blue HII knot on an old disk need not have the colour of
        # the disk, while genuine secondary bulges often resemble the primary.
        def corecolour(px,py):
            values=[]
            for channel in (rgb[0],rgb[2]):
                surround=np.median(sample(channel,px+ct[::4]*3.5,py+st[::4]*3.5))
                values.append(max(0.,sample(channel,px,py)-surround))
            return (values[0]-values[1])/(sum(values)+.025)
        cdiff = abs(corecolour(xx,yy)-corecolour(x,y))
        secondary = (.23+.77*ramp(isolated,.21,.74))*(1-.78*ramp(cdiff,.12,.53))
        # On a projected thin disk an aligned knot has another plausible
        # explanation. Require two resolved nuclei and a substantial dip.
        ondisk = (1-ramp(q,.24,.51))*(1-ramp(abs(np.sin(direc)),.10,.46))
        secondary *= 1.-.65*ondisk*(1-ramp(prom,.27,.60))
        corepair *= secondary*(.72+.28*mass)
        best = max(best,score,.94*corepair)
    return best

def edgewidth(sm,x,y,phi,size,q):
    # Vertical half-maximum thickness in cuts perpendicular to the major axis,
    # allowing for a warp or a slightly misplaced midplane.
    widths=[]
    height=[]
    for f in (0.,.3,-.3,.5,-.5,.68,-.68):
        v = np.arange(-48,49)*max(.25,min(.55,q*size/15.))
        px=x+np.cos(phi)*size*f-np.sin(phi)*v
        py=y+np.sin(phi)*size*f+np.cos(phi)*v
        p=sample(sm,px,py)
        center=np.argmax(np.where(abs(v)<max(2.,q*size*.55),p,-10.))
        h=max(.001,p[center])
        h2=h*.45
        l=np.where(p[:center+1]<h2)[0]
        r=np.where(p[center:]<h2)[0]
        il=int(l[-1]) if len(l) else 0
        ir=int(r[0]+center) if len(r) else len(v)-1
        vl=v[il]+(v[1]-v[0])*clip((h2-p[il])/max(.00001,p[min(il+1,len(v)-1)]-p[il]))
        vr=v[ir]-(v[1]-v[0])*clip((h2-p[ir])/max(.00001,p[max(ir-1,0)]-p[ir]))
        widths.append(max(.3,vr-vl))
        height.append(h)
    return widths[0]/max(.4,np.median(widths[1:])),np.median(height[3:5])/height[0]

def diskandbulge(s,rgb,cx,cy,phi,size,noise,peak):
    # Work in strips on opposite sides of the bulge. Test for a narrow,
    # aligned component and for a broader, differently coloured component
    # which peaks at the centre. The interval and scale are in disk units.
    u = np.linspace(0.,1.35,28)*size
    v = np.linspace(-1.,1.,81)*max(4.,size*.65)
    xx = np.cos(phi)*u[:,None]-np.sin(phi)*v
    yy = np.sin(phi)*u[:,None]+np.cos(phi)*v
    a = sample(s,cx+xx,cy+yy)
    b = sample(s,cx-xx,cy-yy)
    # The geometric mean suppresses other sources in just one wing.
    light = np.sqrt(np.maximum(a,0)*np.maximum(b,0))
    mid = len(v)//2
    core = np.mean(light[:,mid-1:mid+2],axis=1)
    def halfwidth(row,frac):
        p = row[mid:]+row[:mid+1][::-1]
        hit = np.argmax(p<frac*p[0]); hit = int(hit) if hit>0 else mid
        w = clip((frac*p[0]-p[hit])/max(.0001,p[hit-1]-p[hit]))
        return max(.25,(hit-w)*(v[1]-v[0]))
    side = (u>.43*size)&(u<1.08*size)&(core>max(4*noise,.055*peak))
    if not np.any(side):
        side = (u>.4*size)&(u<.65*size)
    wings = light[side]
    widths = np.array([halfwidth(row,.5) for row in wings])
    thickness = np.median(widths/np.maximum(2.,u[side]))
    bc = halfwidth(light[0],.40)
    bulgesize = bc/np.maximum(.4,np.median(widths))
    # Measure compactness and colour of the nucleus vs the midplane. Old red
    # bulges are common; a thin blue disk or central dust lane is distinct
    # from a single solid oval of the same aspect ratio.
    def both(im):
        return .5*(sample(im,cx+xx,cy+yy)+sample(im,cx-xx,cy-yy))
    r,g,blue = both(rgb[0]),both(rgb[1]),both(rgb[2])
    colour = (r-blue)/np.maximum(.01,r+blue+2*noise)
    centcol = np.mean(colour[:3,mid-2:mid+3]); wingcol = np.mean(colour[side,mid-2:mid+3])
    colorstep = centcol-wingcol
    shapepin = np.median(np.array([halfwidth(row,.23) for row in wings]))/size
    # Profile of the outer boundary: for any spheroid with similar ellipses,
    # h(u)/h(0) is sqrt(1-(u/L)**2), regardless of luminosity profile. In a
    # disk with a spheroidal bulge the sides pinch in well inside the ends.
    waist=[]; squash=[]
    for level in (.27,.16,.09,.055):
        cut = max(peak*level,3*noise)
        hit = np.where(core<cut)[0]
        il = int(hit[0]) if len(hit) else len(u)-1
        il = max(1,il)
        length = u[il-1]+(u[1]-u[0])*clip((core[il-1]-cut)/max(.0001,core[il-1]-core[il]))
        centerheight = halfwidth(light[0], min(.95,cut/max(.001,core[0])))
        vals=[]
        for k in range(1,len(u)):
            f = u[k]/max(1.,length)
            if f>.36 and f<.81 and core[k]>cut:
                hh = halfwidth(light[k],cut/core[k])
                vals.append(hh/(max(.5,centerheight)*np.sqrt(1-f*f)))
        waist.append(1.-np.median(vals) if len(vals) else 0.)
        squash.append(centerheight/max(2.,length))
    pinch = np.mean(waist[1:]); flat = np.mean(squash[1:])
    return thickness,bulgesize,colorstep,wingcol,shapepin,pinch,flat

# Analytic model comparison on radial contours. Each candidate model describes
# either one spheroid, a projected disk, or a disk plus central spheroid. No
# coefficient is estimated from an example and no image templates are stored.
modelrad = []
modelids = []
levels = np.array([.70,.48,.30,.17,.085])
modelangle = np.linspace(0,pi/2,15)
modelr = np.linspace(0,2.8,161)
mx = modelr[None,:]*np.cos(modelangle)[:,None]
my = modelr[None,:]*np.sin(modelangle)[:,None]

def modelcont(I):
    # Radial intercepts, normalized to the same semi-major diameter. Compare
    # shapes at fixed *relative* brightness, above the sky-noise regime.
    r=[]
    for f in levels:
        cut = I[0,0]*f
        k = np.maximum(1,np.argmax(I<cut,axis=1))
        k = np.where(I[:,-1]>=cut,len(modelr)-1,k)
        ids=np.arange(len(k)); hi=I[ids,k-1]; lo=I[ids,k]
        r.append(modelr[k-1]+(modelr[1]-modelr[0])*clip((hi-cut)/np.maximum(.00001,hi-lo)))
    r=np.array(r)
    return r/np.maximum(.001,r[:,0:1])

def addmodel(I,category):
    modelrad.append(np.log(np.maximum(.01,modelcont(I))))
    modelids.append(category)

for flattening in (.125,.175,.225,.275,.325,.375,.425,.475,.55,.65,.8):
    ell = np.sqrt(mx*mx+(my/flattening)**2)
    for power in (.65,1.,1.5,2.):
        # Central curvature includes the PSF instead of insisting on an
        # infinitely sharp cusp. A single spheroid has elliptical isophotes.
        S=np.exp(-2.1*(np.sqrt(ell*ell+.06**2)**power-.06**power))
        addmodel(S,0)
    # Projected exponential disks with thickness, with either exponential or
    # sech-like light falloff out of the plane. The projected radial profile
    # is shallower near the origin than the intrinsic exponential profile.
    for inc in (0.,.58,.85):
        R=np.sqrt(mx*mx+.13**2)-.13
        Z=np.sqrt((my/flattening)**2+.08**2)-.08
        diskdist=(R+Z)*(1-inc)+inc*np.sqrt(R*R+Z*Z)
        D=np.exp(-2.6*diskdist)
        addmodel(D,1)
        if flattening<.43:
            for h in (.16,.29,.48):
                for amp in (.42,1.,2.4):
                    for bq in (.55,.85):
                        Rb=np.sqrt(mx*mx+(my/bq)**2+.035**2)-.035
                        B=np.exp(-Rb/h)
                        addmodel((D+amp*B)/(1+amp),2)
modelrad=np.array(modelrad)
modelids=np.array(modelids)

def modelvote(s,cx,cy,phi,size,noise):
    r = np.linspace(0.,min(70.,1.65*size),121)
    # Fold all four quadrants; unrelated sources cannot dominate the median.
    profiles=[]
    for signx,signy in ((1,1),(-1,1),(1,-1),(-1,-1)):
        u = signx*np.cos(modelangle)[:,None]*r
        v = signy*np.sin(modelangle)[:,None]*r
        profiles.append(sample(s,cx+u*np.cos(phi)-v*np.sin(phi),cy+u*np.sin(phi)+v*np.cos(phi)))
    P=np.median(profiles,axis=0)
    radii=[]
    for level in levels:
        cut=max(3*noise,P[0,0]*level)
        k=np.maximum(1,np.argmax(P<cut,axis=1)); k=np.where(np.all(P>=cut,axis=1),len(r)-1,k)
        ids=np.arange(len(k)); hi=P[ids,k-1]; lo=P[ids,k]
        radii.append(np.maximum(.3,r[k-1]+(r[1]-r[0])*clip((hi-cut)/np.maximum(.00001,hi-lo))))
    radii=np.array(radii)
    rel=radii/np.maximum(.3,radii[:,0:1])
    # Outer contours are better resolved than the inner contours. In a very
    # small object, the PSF by itself makes the inner contours rounder, so
    # resolution limits the evidence from a two-component comparison.
    err = np.minimum(.35,abs(modelrad-np.log(rel)))
    wt = ramp(radii[:, -1],1.3,4.)*.75+.25
    loss = np.sum(np.mean(err,axis=2)*wt,axis=1)/wt.sum()
    single=np.min(loss[modelids==0]); flat=np.min(loss[modelids==1]); composite=np.min(loss[modelids==2])
    return single,flat,composite

def classify(img):
    rgb = img.astype(np.float64)
    rgb -= np.median(rgb[:,edge],axis=1)[:,None,None]
    ft = np.fft.rfft2(rgb,axes=(1,2))
    rgb = np.fft.irfft2(ft*small,s=(128,128),axes=(1,2))
    light = .3*ft[0]+.42*ft[1]+.28*ft[2]
    s = .3*rgb[0]+.42*rgb[1]+.28*rgb[2]
    t = np.fft.irfft2(light*medium,s=(128,128))
    b = np.fft.irfft2(light*broad,s=(128,128))
    s -= np.median(s[edge]); t -= np.median(t[edge]); b -= np.median(b[edge])
    noise = mad(s); ns = mad(t)
    cx,cy = nucleus(s)
    peak,z,qs,theta,a,lops,box,axial,plume,growth = contours(s,t,cx,cy,noise,ns)
    outer,q,phi,size = diskframe(z,a)
    c,col,flux = rings(rgb,s,cx,cy,q,phi,size,noise)
    loose,tight,hand = spirals(c)
    companion = companions(s,t,b,rgb,cx,cy,peak,size,q,phi,ns)
    colin = np.mean(col[inner]); colout = np.mean(col[middle|skirt]); gradient = colin-colout
    diam = size*2
    resolved = size*np.sqrt(q)
    ratio = np.mean(a[2:4])/max(1.,size)
    # Hand-assigned fuzzy predicates, with continuous transitions where class
    # definitions themselves don't specify an exact boundary.
    blue = ramp(-colout,-.065,.19)
    diskcolor = ramp(gradient,.045,.21)
    red = ramp(colout,.04,.22)
    big = ramp(resolved,5.,14.)
    discsize = ramp(resolved,7.,22.)
    ochange = np.mean(abs(z[7:10]-outer))
    strong = max(loose,tight)
    m1 = meanmag(c,middle,1)
    m3 = meanmag(c,middle,3)
    detail = np.mean(np.sqrt(np.sum(abs(c[middle,3:9])**2,axis=1)))
    outerodd = np.mean(lops[7:10]); innerodd=np.mean(lops[2:6])
    faintodd = np.mean(lops[9:12]); fainttwist=abs(z[-1]-z[6])
    # Compare the shape and direction of the putative bar to the outer disk.
    bar = 0.
    for start in (0,1,2,3):
        zz = z[start:start+3].mean()
        change = abs(zz-outer)
        alignment = (zz*np.conj(outer)).real/(abs(zz)*abs(outer)+.00001)
        # A circular bulge in an inclined disk looks elongated after deprojection,
        # but is not a bar. Its ellipticity merely grows towards that of the disk.
        elong = abs(zz)-abs(outer)*max(0.,alignment)
        coherence = np.mean(abs(z[start:start+3]-zz))
        v = ramp(change,.042,.18)*(.34+.66*ramp(elong,.005,.13))
        v *= (.35+.65*ramp(abs(zz),.055,.19))*(1-.48*ramp(coherence,.025,.1))
        v *= (.6+.4*ramp(a[start+1],2.,5.))
        bar = max(bar,v)
    midbar = abs(np.mean(c[region(.24,.56),2]))
    bar *= .78+.22*ramp(midbar,.07,.25)
    # A smoothly winding arm contributes one sign of angular frequency.
    # Power in higher angular modes also reveals fragmented spiral arms.
    arms = ramp(strong,.023,.135)*(.6+.4*ramp(hand,.012,.09))
    texture = ramp(detail,.065,.21)
    disk = (.26*big+.20*blue+.19*diskcolor+.17*discsize+.14*texture+.12*ramp(ratio,.37,.56))
    disk += .29*arms+.24*bar-.10*red
    # Account for resolution in deciding whether structure is actually visible.
    disk *= .71+.29*ramp(resolved,4,10)
    # Stronger irregularity than simple ellipticity changes: odd contours,
    # a one-sided halo or a strong bend localized to very low surface brightness.
    uneven = ramp(outerodd,.065,.25)
    centralodd = ramp(innerodd,.055,.22)
    tail = ramp(faintodd,.10,.32)*(.55+.45*ramp(faintodd-innerodd,.035,.16))
    chaotic = .32*uneven+.18*centralodd+.24*tail+.17*ramp(m1,.12,.43)+.09*ramp(fainttwist,.09,.32)
    chaotic += .12*tail*(1-diskcolor)+.08*centralodd*uneven
    chaotic -= .20*arms+.12*bar
    # Compare both the faint and bright shapes. Tidal distortions can be an
    # unmatched plume OR a bilateral warp (the latter has little odd power).
    outerplume = .5*(np.mean(plume[7:10])+np.median(plume[9:12]))
    crook = abs(z[4:7].mean()-z[9:12].mean())
    diffangle = abs(np.sin(np.angle(z[4:7].mean()*np.conj(z[9:12].mean()))))
    bend = ramp(crook,.085,.25)*ramp(diffangle,.21,.66)*ramp(abs(z[4:7].mean()),.05,.16)
    distort = ramp(outerplume,.27,.83)*(.64+.36*ramp(np.mean(plume[4:8]),.14,.43))
    disorder = (.49*distort+.19*bend+.19*ramp(outerodd,.06,.22)+.13*centralodd)
    disorder *= 1-.31*arms
    chaotic = max(chaotic,disorder)
    trail = debris(b,cx,cy,size,q,phi,peak,mad(b))
    gtail = ramp(growth,.29,.85)
    tidal = .47*trail+.21*gtail+.15*distort+.10*bend+.07*centralodd
    tidal *= 1.-.4*arms
    chaotic = max(chaotic,tidal)
    arcslow,arcfast = armcurves(c)
    # The apparent flattening of the main light distribution is more robust than
    # a full-frame second moment for a small target in a crowded patch of sky.
    qbody = .28*np.mean(qs[3:6])+.48*np.mean(qs[6:9])+.24*np.mean(qs[9:11])
    # Mild seeing correction to discriminate small genuinely flattened objects.
    qshape = max(.06,qbody-.5*(1-qbody)/max(1.,(size*qbody/2)**2))
    qmid = np.mean(qs[2:5]); pointy = np.mean(box[5:10])
    axialout = np.mean(axial[6:11]); axialmid = np.mean(axial[2:6])
    spread = .45*(np.mean(qs[1:5])-np.mean(qs[7:10]))+.55*(axialmid-axialout)
    needle = 1.-ramp(qshape,.20,.355)
    pointed = ramp(pointy,.026,.13)
    edgelike = needle+.23*pointed+.11*ramp(size,12.,30.)+.12*ramp(spread,.045,.20)+.07*blue
    # Decision by morphology, not by batch prevalence or nearest example.
    if companion > .57 and (companion > .75 or chaotic > .36 or companion > .65-.07*tail):
        return 1
    # If the main light body is accurately point symmetric, the classifier
    # should demand an attached trail, not merely a remote asymmetric contour.
    bodyodd = np.mean(lops[2:8])
    foreign = (1-ramp(bodyodd,.025,.085))*(1-ramp(trail,.24,.66))
    chaotic *= 1-.38*foreign
    if chaotic > .54 and not (edgelike>.82 and chaotic<.75):
        return 0
    # Edge-on disks have pointed outer isophotes and a shallow minor-axis
    # profile *away* from the centre. A bulge can conceal these features in
    # contours which are given equal weight over the whole object.
    width,shoulder = edgewidth(s,cx,cy,phi,size,q)
    enddisk = (1-ramp(axialout,.21,.43))*(.35+.27*ramp(width,1.,1.65)
               +.25*ramp(box[6:11].mean(),.015,.10)+.20*ramp(spread,.025,.15))
    thin,swelling,colstep,wingcol,offwidth,pinch,flataxis = diskandbulge(s,rgb,cx,cy,phi,size,noise,peak)
    # Require either an intrinsically very thin body, or the conjunction of
    # a narrow outer disk and the signature of a separate spheroidal component.
    narrow = .42*(1-ramp(thin,.16,.40))+.22*(1-ramp(offwidth,.20,.43))+.36*(1-ramp(axialout,.27,.54))
    twocomp = (.24*ramp(spread,.008,.15)+.22*ramp(swelling,1.05,1.75)+.17*ramp(colstep,.035,.18)
               +.19*pointed+.18*ramp(colin,-.025,.20))
    # For nearly edge-on systems, even a substantial projected halo shouldn't
    # cancel the evidence for a disk where the narrow waist is detected.
    spindle = (.44*(1-ramp(axialout,.25,.52))+.22*ramp(size,11.,28.)+.19*pointed+.15*ramp(spread,0.,.16))
    twin = narrow*(.55+.45*twocomp)+.22*twocomp+.11*spindle
    lens = ramp(pinch,.06,.34)*(.27+.24*ramp(colstep,.01,.17)+.22*ramp(spread,0.,.13)
                    +.15*(1-ramp(flataxis,.29,.52))+.12*ramp(size,8.,18.))
    bulgemodel = 0.
    if qshape<.61:
        one,plane,combo = modelvote(s,cx,cy,phi,size,noise)
        bulgemodel = ramp(one-combo,.018,.10)*(.53+.47*ramp(plane-combo,.005,.065))
        bulgemodel *= (1.-.65*ramp(combo,.085,.21))*(.58+.42*ramp(size*q,2.,5.))
    edgeprior = .34*(1.-ramp(flataxis,.32,.58))+.33*ramp(colin,-.04,.22)+.33*ramp(size,6.,18.)
    modelevidence = bulgemodel*(.48+.52*edgeprior)
    if (edgelike>.70 and qshape<.37) or enddisk>.43 or (qshape<.55 and (twin>.68 or lens>.38)) or modelevidence>.47:
        bulge = (.26*ramp(spread,.015,.155)+.20*ramp(width,.97,1.50)+.19*ramp(qmid,.165,.34)
                 +.12*ramp(colin,-.045,.18)+.14*(1.-ramp(shoulder,.26,.5))+.09*pointed)
        bulge = max(bulge, .46*twocomp+.29*ramp(qmid,.18,.40)+.25*ramp(colstep,.015,.17))
        bulge = max(bulge,.50*lens+.24*ramp(qmid,.16,.38)+.26*ramp(colin,-.06,.20))
        bulge = max(bulge,.58*bulgemodel+.25*ramp(qmid,.21,.43)+.17*ramp(colin,-.035,.21))
        return 9 if bulge>.52 else 8
    if qshape < .47:
        disk -= .14*(1.-ramp(qshape,.30,.47))
    # Isophotes with a common centre, ellipticity, and position angle describe
    # a spheroid even if its stellar population happens to be rather blue.
    variation = np.mean(abs(z[2:9]-np.median(z[2:9].real)-1j*np.median(z[2:9].imag)))
    regular = (1-ramp(variation,.012,.07))*(1-ramp(detail,.035,.12))*(1-ramp(outerodd,.035,.14))
    disk -= .23*regular*(1.-ramp(hand,.009,.06))
    if (disk > .55 or (bar>.73 and resolved>5.5 and qshape>.32)):
        if bar>.48 and (bar>.7 or strong<.14 or midbar>.16):
            return 5
        # Tightly wound arms overlap in projection and have less contrast;
        # broad, open arms maintain a slowly changing phase across the disk.
        pitch = (loose-tight)/(.04+loose+tight)
        openarms = (.34*ramp(pitch,-.19,.36)+.24*ramp(loose,.037,.18)+.13*texture
                    +.12*ramp(m1,.075,.27)+.10*blue+.07*ramp(outerodd,.05,.18))
        led = (arcslow-arcfast)/(.012+max(arcslow,arcfast))
        armcontrast = max(loose,arcslow)
        localarm = ramp(armcontrast,.025,.11)*(.42+.58*ramp(led,-.13,.36))
        return 7 if max(openarms,.65*localarm+.35*openarms)>.49 else 6
    if chaotic>.41 and (tail>.58 or centralodd>.62):
        return 0
    apparent = .4*qshape+.25*np.mean(qs[3:7])+.35*np.mean(axial[4:9])
    if apparent>.82:
        return 2
    if apparent<.50:
        return 4
    return 3

def predict(x):
    return np.array([classify(im) for im in x],dtype=np.int64)
'''

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / 'solution.py').write_text(source, encoding='utf-8')
