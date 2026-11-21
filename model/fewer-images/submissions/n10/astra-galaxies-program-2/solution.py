import math
import numpy as np

Y, X = np.indices((128, 128), dtype=np.float64)
D0 = (X-63.5)**2 + (Y-63.5)**2
SKY = D0 > 53**2
K2 = (np.fft.fftfreq(128)[:, None]**2 + np.fft.rfftfreq(128)[None, :]**2) * (2*math.pi**2)
K1 = np.exp(-1.15**2*K2)
K3 = np.exp(-2.7**2*K2)
ANG = np.arange(96)*2*math.pi/96
C = np.cos(ANG)
S = np.sin(ANG)
MODES = np.exp(1j*np.arange(1,9)[:,None]*ANG)
U = np.geomspace(.075, 1.85, 64)
LOGU = np.log(U)
EPS = 1.e-8


def unit(x):
    return np.clip(x, 0., 1.)


def ramp(x, lo, hi):
    return unit((x-lo)/(hi-lo))


def blurft(ft, kernel):
    return np.fft.irfft2(ft*kernel, s=(128,128), axes=(-2,-1))


def sample(a, x, y):
    x = np.clip(x, 0., 126.99999)
    y = np.clip(y, 0., 126.99999)
    j = x.astype(np.intp)
    i = y.astype(np.intp)
    dx, dy = x-j, y-i
    return (1-dy)*((1-dx)*a[i,j] + dx*a[i,j+1]) + dy*((1-dx)*a[i+1,j] + dx*a[i+1,j+1])


def reducew(a, w):
    return np.sum(a*w)/(EPS+np.sum(w))


def connected(b, iy, ix):
    # Run-length flood fill. Each horizontal run is visited once.
    hit = np.zeros(b.shape, dtype=bool)
    stack = [(iy,ix)]
    while stack:
        i,j = stack.pop()
        if hit[i,j] or not b[i,j]:
            continue
        l,r = j,j
        while l > 0 and b[i,l-1]:
            l -= 1
        while r < 127 and b[i,r+1]:
            r += 1
        hit[i,l:r+1] = True
        lo = max(0,l-1)
        hi = min(128,r+2)
        for h in (i-1,i+1):
            if 0 <= h < 128:
                val = b[h,lo:hi] & ~hit[h,lo:hi]
                if np.any(val):
                    for t in np.flatnonzero(val & ~np.r_[False,val[:-1]]):
                        stack.append((h,lo+int(t)))
    return hit


def geometry(w, dx, dy):
    tot = w.sum()+EPS
    xx = np.sum(w*dx*dx)/tot
    yy = np.sum(w*dy*dy)/tot
    xy = np.sum(w*dx*dy)/tot
    delta = math.hypot(xx-yy, 2*xy)
    pa = .5*math.atan2(2*xy, xx-yy)
    a2 = max(.3,(xx+yy+delta)/2)
    b2 = max(.25,(xx+yy-delta)/2)
    return 2*math.sqrt(a2), math.sqrt(min(1.,b2/a2)), pa


def isophotes(img, ref, noise, ix, iy, dx, dy, r2):
    # Use the fainter of antipodal pixels, with limited slack for texture.
    sym = np.minimum(img, ref) + .32*np.minimum(np.abs(img-ref), np.maximum(np.minimum(img,ref),0)+2*noise)
    peak = max(EPS, float(sym[iy,ix]))
    aa, qq, ang = [], [], []
    for f in (.055,.10,.18,.32,.50,.70):
        cut = max(2.8*noise, f*peak, .002)
        cut = min(cut,peak*.82)
        hit = connected((sym >= cut*.85) & (r2 < 60**2), iy, ix)
        w = hit*ramp(sym, cut*.85, 1.15*cut+EPS)
        a,q,p = geometry(w, dx, dy)
        aa.append(a)
        qq.append(q)
        ang.append(p)
    return np.array(aa),np.array(qq),np.array(ang)


def fieldseeing(img,raw,noise):
    maximum = (D0>24**2)&(X>6)&(Y>6)&(X<121)&(Y<121)&(img>max(.035,8*noise))
    for h,k in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(-1,1),(1,-1)):
        maximum &= img>np.roll(np.roll(img,h,0),k,1)
    y,x = np.nonzero(maximum)
    if len(x)<2:
        return 1.15
    top = np.argsort(img[y,x])[::-1][:32]
    x,y=x[top,None],y[top,None]
    circx,circy=C[::6],S[::6]
    base = np.median(sample(raw,x+5*circx,y+5*circy),axis=1)
    peak = np.maximum(.01,raw[y[:,0],x[:,0]]-base)
    v = (np.median(sample(raw,x+2*circx,y+2*circy),axis=1)-base)/peak
    sig = np.sqrt(-2./np.log(np.clip(v,.02,.8)))
    return float(np.clip(np.percentile(sig,25),.75,1.85))


def peaks(img, ref, pk, cx,cy, A,q,cp,sp, noise, seeing, rgb):
    # Is this bright neighbour extended, distinct from the primary, and close
    # on the scale of the two envelopes? Test its halo on several circles.
    dist = np.hypot(X-cx,Y-cy)
    maxima = (img >= np.roll(img,1,0)) & (img > np.roll(img,-1,0)) & (img >= np.roll(img,1,1)) & (img > np.roll(img,-1,1))
    for h,k in ((1,1),(-1,1),(1,-1),(-1,-1)):
        maxima &= img >= np.roll(np.roll(img,h,0),k,1)
    candidates = maxima & (dist>3) & (dist<53) & (img > max(4.5*noise,.07*pk,.012)) & (img-ref > max(3*noise,.035*pk))
    yy,xx = np.nonzero(candidates)
    if len(xx) == 0:
        return 0., np.ones_like(img), 0.
    order = np.argsort(((img-ref)[yy,xx])/(1+(dist[yy,xx]/18)**2))[::-1][:40]
    clean = np.ones_like(img)
    merge, near = 0., 0.
    rem = img-ref
    sc = C[::4]
    ss = S[::4]
    for k in order:
        x,y = xx[k],yy[k]
        d = dist[y,x]
        v = img[y,x]
        diff = rem[y,x]
        strength = (.3*v+.7*diff)/(pk+EPS)
        rad = np.array([2.2,3.6,6.,9.])[:,None]
        halo = np.maximum(0.,np.median(sample(rem,x+rad*sc,y+rad*ss),axis=1))/(diff+EPS)
        ext = .5*ramp(halo[1],.12,.45) + .3*ramp(halo[0],.43,.76) + .2*ramp(halo[2],.035,.25)
        ax = ((x-cx)*cp+(y-cy)*sp)/d
        ay = ((y-cy)*cp-(x-cx)*sp)/d
        pray = A/math.sqrt(ax**2+(ay/max(q,.13))**2+EPS)
        arad = 3.+6*unit(halo[1])+10*unit(halo[2])+8*unit(halo[3])
        sep = d/(4.+pray+arad)
        t = np.linspace(.10,.90,15)
        saddle = float(np.min(sample(img,cx+t*(x-cx),cy+t*(y-cy))))
        drop = (v-saddle)/(v+EPS)
        realpeak = .35+.65*ramp(drop,.02,.33)
        bright = ramp(math.log(max(EPS,strength)),math.log(.08),math.log(.43))
        close = 1.-ramp(sep,.80,1.75)
        # A physical pair needs two resolved cores with common envelope
        # light, not simply a second detection in the field. The PSF is
        # broader after our Gaussian filter (variances add).
        width2 = seeing**2+1.15**2
        stellar = math.exp(-3.6**2/(2*width2))
        resolvedcore = .65*ramp(halo[1]-stellar,.04,.29)+.35*ramp(halo[2],.026,.17)
        bridge = ramp((saddle-noise)/(min(pk,v)+EPS),.016,.19)
        touch = 1.-ramp(sep,.65,1.32)
        assoc = max(.12*touch+.88*bridge, .58*touch)*close
        fluxbal = strength*(arad/(A*math.sqrt(q)+EPS))**1.35
        nuclear = bright*(.50+.50*ramp(fluxbal,.07,.38))
        col = np.maximum(rgb[:,y,x],.004)
        nonstellar = 1.-.8*ramp(max(col)/(np.median(col)+EPS),1.65,3.3)
        # A diffuse, blue arm knot inside an existing disk is not a new
        # nucleus. Test for a compact central excess above the knot's halo
        # and compare core colours against the primary, not the sky.
        maincol = np.maximum(.004,rgb[:,int(round(cy)),int(round(cx))])
        colorcontrast = (col[2]-col[0])/(col[2]+col[0])-(maincol[2]-maincol[0])/(maincol[2]+maincol[0])
        flat = ramp(halo[0],.66,.94)
        knot = (1-ramp(d/(pray+EPS),.83,1.50))*ramp(colorcontrast,.04,.31)*(.42+.58*flat)
        score = nuclear*resolvedcore**.80*(.04+.96*assoc**.65)*realpeak*nonstellar*(1-.79*knot)
        merge = max(merge,score)
        near = max(near,bright*ext*close*realpeak)
        # Suppress point sources and detached objects for disk-arm analysis.
        # A resolved knot inside the disk can instead be part of an arm.
        outside = ramp(d/(pray+1.),.72,1.35)
        nuisance = max((1.-ext)*ramp(drop,.12,.5), outside*ramp(drop,.18,.65))
        if nuisance>.1:
            size = min(.45*d,max(2.7,(2.2+6*halo[1])*math.sqrt(1+math.log(1+diff/(.06*pk+noise+EPS)))))
            ds = (X-x)**2 + (Y-y)**2
            clean = np.minimum(clean,1.-nuisance*np.exp(-.5*(ds/(size*size))**2))
    return merge,clean,near


def rings(img,x,y,cover):
    values = sample(img,x,y)
    mid = np.median(values,axis=1,keepdims=True)
    return mid + cover*(values-mid)


def spiralbank(z, w, m):
    # DC-rejecting logarithmic spiral filters. A fixed ellipse with a wrong
    # inclination yields a stationary quadrupole, not a rotating quadrupole.
    w = w/(EPS+w.sum())
    z = z-np.sum(w*z)
    bestlow,besthigh = 0.,0.
    for pitch in (12.,17.,23.,31.,43.,58.):
        f = m/math.tan(math.radians(pitch))
        wave = np.exp(1j*f*LOGU)
        den = math.sqrt(max(.25,1.-abs(np.sum(w*wave))**2))
        res = max(abs(np.sum(z*w*wave)), abs(np.sum(z*w*wave.conj())))/den
        if pitch>=31:
            bestlow=max(bestlow,res)
        else:
            besthigh=max(besthigh,res)
    return bestlow,besthigh


def ellipseerrors(img,mask,cx,cy,A,aa,qq,pa,peak,noise):
    # Reconstruct concentric filled ellipses, one at each observed shell.
    # For an elliptical spheroid the square-radius interpolation of the
    # isophotes should be well behaved regardless of its radial profile.
    # A bar, dust lane, arm or tidal feature leaves contiguous residuals.
    ux,vy = X-cx,Y-cy
    shells = []
    for a,q,p in zip(aa,qq,pa):
        co,si=math.cos(p),math.sin(p)
        u,v = co*ux+si*vy,co*vy-si*ux
        shells.append(np.sqrt((u/max(a,1.))**2+(v/max(.8,a*q))**2))
    rad = np.array(shells)
    levels = np.array([.055,.10,.18,.32,.50,.70])*peak
    levels = np.minimum(np.maximum(levels,max(.002,2.8*noise)),.82*peak)
    expected = levels[0]*np.exp(-np.clip(2.8*(rad[0]-1),-2,30))
    for j in range(len(aa)-1):
        t = unit((1-rad[j])/(np.maximum(.004,rad[j+1]-rad[j])))
        value = levels[j]*(levels[j+1]/(levels[j]+EPS))**t
        expected = np.where(rad[j]<=1,value,expected)
    expected = np.where(rad[-1]<1,peak-(peak-levels[-1])*rad[-1]**2,expected)
    local = (1-ramp(rad[1],.9,1.48))*ramp(rad[1],.12,.4)*mask
    err = np.maximum(0.,abs(img-expected)-1.45*noise)/(.052*peak+np.maximum(0.,expected)+2*noise)
    # Median and a capped mean suppress isolated contaminating stars.
    trim = np.minimum(err,.75)
    val = reducew(trim,local)
    # Past the main body, estimate how close the outline is to a single
    # ellipse even if the signal is too faint to measure an arm ridge.
    outval = reducew(trim,local*ramp(rad[1],.50,1.0))
    return .55*val+.45*outval


def arcpitch(coeff,vis):
    # Unwrap only local angular differences, avoiding 2pi discontinuities.
    # For mode m of a logarithmic spiral d(phi_m)/d(log r)=m*cot(pitch).
    vec = []
    for m in (1,2,3):
        z = coeff[m-1]
        lo = z[:-6]
        hi = z[6:]
        phase = np.angle(hi*lo.conj())/((LOGU[6]-LOGU[0])*m)
        w = vis[3:-3]*np.exp(-.5*(np.log(U[3:-3]/.80)/.44)**2)
        w = w*ramp(np.minimum(abs(lo),abs(hi)),.035,.20)*ramp(abs(phase),.27,.9)
        # Opposite-sign twists cancel; a logarithmic arm has one chirality.
        turn = reducew(np.sign(phase),w)
        consistent = w*(1+np.sign(phase)*turn)*.5
        slope = reducew(abs(phase),consistent)
        evidence = abs(turn)*min(1.,np.sum(w)/9.)
        vec.append((evidence*(1. if m==2 else .63),slope))
    conf = sum(a for a,b in vec)
    slope = sum(a*b for a,b in vec)/(EPS+conf)
    return min(1.,conf),slope


def arcfilters(rgbp,dev,mid,noise,vis,A,q):
    # Unwrap the disk and shear by a sequence of signed logarithmic pitches.
    # A real arm straightens into a persistent maximum for only one sign.
    # Also test short arcs: galaxies need not have two full symmetric arms.
    contrast = []
    for j in (1,2):
        z = rgbp[j]
        mu = np.median(z,axis=1,keepdims=True)
        contrast.append(np.clip((z-mu)/(np.maximum(0,mu)+3*noise+.02*np.max(mid)+EPS),-1.1,1.5))
    dev = .50*dev+.20*contrast[0]+.30*contrast[1]
    # Remove angular DC, dipole and static ellipticity. Remove a broad
    # radial running average, which preserves tilted ridges, not straight
    # radial streaks. The units of the kernel are log radius.
    dev = dev-dev.mean(axis=1,keepdims=True)
    a1 = (dev @ MODES[0])/len(ANG)
    dev = dev - 2*(a1[:,None]*MODES[0].conj()).real
    kernel = np.exp(-.5*((LOGU[:,None]-LOGU[None,:])/.32)**2)
    dev = dev-(kernel/kernel.sum(axis=1,keepdims=True)) @ dev
    weighted = [vis*np.exp(-.5*(np.log(U/r)/.30)**2)*ramp(A*U*np.sqrt(q),2.,5.) for r in (.45,.80,1.12)]
    idx = np.arange(len(ANG))[None,:]
    ridx = np.arange(len(U))[:,None]
    result = []
    for pitch in (12.,18.,25.,35.,50.):
        vals = []
        for sign in (-1,1):
            spin = sign*LOGU/math.tan(math.radians(pitch))*len(ANG)/(2*math.pi)
            k = np.floor(spin).astype(np.intp)
            t = (spin-k)[:,None]
            k = (idx+k[:,None]) % len(ANG)
            arc = (1-t)*dev[ridx,k]+t*dev[ridx,(k+1)%len(ANG)]
            path = []
            for w in weighted:
                curve = np.sum(arc*w[:,None],axis=0)/(w.sum()+EPS)
                # Emphasise repeated pair/triplet responses rather than a
                # single bright HII region or a foreground star.
                curve = .5*curve+.25*np.roll(curve,2)+.25*np.roll(curve,-2)
                paired = .5*(curve+np.roll(curve,len(ANG)//2))
                path.append(.42*np.percentile(curve,91)+.58*np.percentile(paired,92))
            vals.append(.48*max(path)+.52*np.mean(path))
        result.append(max(vals)-.58*min(vals))
    return max(result[:3]),max(result[3:])


def tidalshape(broad,mask,cx,cy,cp,sp,A,q,pk,noise):
    # Shape asymmetry uses area rather than flux: it does not let a bright
    # symmetric nucleus drown out a diffuse tidal plume. Only connected
    # light counts, and peaks identified as unrelated are gated out.
    u = (X-cx)*cp+(Y-cy)*sp
    v = (Y-cy)*cp-(X-cx)*sp
    e2 = (u/A)**2+(v/(q*A))**2
    bound = ((u/(3.7*A+7))**2+(v/(2.0*A+7))**2<1.) & (D0<59**2)
    cenw = np.maximum(broad-2*noise,0)*mask*np.exp(-.5*e2)
    shiftx = np.clip(reducew(X-cx,cenw),-.20*A,.20*A)
    shifty = np.clip(reducew(Y-cy,cenw),-.20*A,.20*A)
    opposite = []
    for f in (0.,1.55):
        opposite.append((2*(cx+f*shiftx)-X,2*(cy+f*shifty)-Y))
    row,col = int(round(cy)),int(round(cx))
    measures = []
    for frac,sigma in ((.021,2.8),(.047,3.4),(.11,4.)):
        cut = max(frac*pk,sigma*noise)
        hit = connected((broad>cut*.86)&bound&(mask>.42),row,col)
        w = hit*ramp(broad,.86*cut,1.15*cut)
        norm = np.sum(w)+EPS
        area = []
        for x,y in opposite:
            rev = sample(w,x,y)
            pairmask = np.minimum(mask,sample(mask,x,y))*bound
            area.append(np.sum(abs(w-rev)*pairmask)/(EPS+np.sum((w+rev)*pairmask)))
        asym = min(area)
        # Small contours displaced by a fraction of a PSF, or contour
        # fluctuations of order sky noise, are weak evidence by themselves.
        asym *= ramp(norm,18.,95.)
        measures.append(asym)
    return .46*measures[0]+.34*measures[1]+.20*measures[2],measures[0]-measures[2]


def balancedasym(broad,mask,cx,cy,cp,sp,A,q,pk,noise):
    # Opposed halves compared about a luminosity centroid as well as the
    # peak: for a dust lane the brightest pixel need not be the centre.
    u = ((X-cx)*cp+(Y-cy)*sp)/A
    v = ((Y-cy)*cp-(X-cx)*sp)/(q*A)
    e2 = u*u+v*v
    w = np.maximum(broad-2*noise,0)*mask*np.exp(-.5*e2/.88**2)
    xx = np.clip(reducew(X-cx,w),-.23*A,.23*A)
    yy = np.clip(reducew(Y-cy,w),-.23*A,.23*A)
    asym = 1.
    for fac in (0.,1.,1.55):
        rx,ry = 2*(cx+fac*xx)-X,2*(cy+fac*yy)-Y
        rev = sample(broad,rx,ry)
        clean = mask*sample(mask,rx,ry)*(rx>=0)*(rx<=127)*(ry>=0)*(ry<=127)
        d = (np.maximum(broad,0)+np.maximum(rev,0))
        # Include a rounder and more extended area than the symmetric disk
        # for a bent or one-sided tidal stream. Restrict it to pixels which
        # are connected to the broad luminous body above the sky noise.
        gate = (u/2.1)**2+(v/(1.35+.85*q**-.35))**2 < 1.
        brighter = np.maximum(broad,rev)
        hit = connected(gate & (brighter>max(.008*pk,2.8*noise)),int(round(cy)),int(round(cx)))
        w = clean*hit*(.2+.8*ramp(e2,.05,.6))
        if np.sum(w)>1:
            num = np.sum(np.maximum(0.,abs(broad-rev)-1.3*noise)*w)
            den = np.sum(d*w)+EPS
            asym = min(asym,num/den)
        else:
            asym = 0.
    return asym


def analyse(frame):
    channels = frame.astype(np.float64)
    # Measure the field PSF on isolated field maxima in the unsmoothed image.
    # The lower-width envelope avoids using extended galaxies as stars.
    raw = .30*channels[0]+.45*channels[1]+.25*channels[2]
    raw -= np.median(raw[SKY])
    ft = np.fft.rfft2(channels, axes=(-2,-1))
    rgb = blurft(ft,K1)
    rgb -= np.median(rgb[:,SKY],axis=1)[:,None,None]
    img = .30*rgb[0] + .45*rgb[1] + .25*rgb[2]
    med = np.median(img[SKY])
    img -= med
    noise = max(.0015,1.4826*np.median(np.abs(img[SKY])))
    broad = blurft(.30*ft[0]+.45*ft[1]+.25*ft[2],K3)
    broad -= np.median(broad[SKY])
    target = (.64*img+.36*broad)*np.exp(-.5*D0/6.**2)
    iy,ix = np.unravel_index(np.argmax(target), target.shape)
    # Recenter locally on the nucleus, not on a distant bright star.
    aperture = (X-ix)**2+(Y-iy)**2<=3.6**2
    iy,ix = np.unravel_index(np.argmax(np.where(aperture,img,-10.)),img.shape)
    cy,cx = float(iy),float(ix)
    pk = max(.002,float(img[iy,ix]))
    for step in range(3):
        d = (X-cx)**2+(Y-cy)**2
        w = np.exp(-.5*d/2.35**2)*np.maximum(0,img-.30*pk)
        if w.sum()>EPS:
            cx = .85*reducew(X,w)+.15*ix
            cy = .85*reducew(Y,w)+.15*iy
    dx,dy = X-cx,Y-cy
    r2 = dx*dx+dy*dy
    ix,iy = int(round(cx)),int(round(cy))
    ref = sample(img,cx-dx,cy-dy)
    aa,qq,pa = isophotes(img,ref,noise,ix,iy,dx,dy,r2)
    A = float(np.clip(.65*aa[1]+.35*aa[2],3.,60.))
    zpa = np.sum((1-qq[:3])*np.exp(2j*pa[:3])*[.36,.4,.24])
    theta = .5*float(np.angle(zpa))
    cp,sp = math.cos(theta),math.sin(theta)
    # Use the observed projected shape for polar sampling, but partially
    # remove circular seeing from the apparent flattening for classification.
    q = np.clip(.37*qq[0]+.43*qq[1]+.20*qq[2],.12,1.)
    seeing = fieldseeing(img,raw,noise)
    seeing2 = seeing**2+1.15**2
    # Seeing increases the squared radius of a given fractional isophote
    # by 2 log(Ipeak/I) sigma^2 for a Gaussian, not by a fixed area.
    broadening = 2*np.log(1./np.array([.055,.10,.18,.32,.50,.70]))*seeing2
    broadening = np.minimum(broadening, .64*(aa*qq)**2)
    qiso = .20*qq+.80*np.sqrt(np.clip(((aa*qq)**2-broadening)/np.maximum(.1,aa**2-broadening),.006,1.))
    qshape = .2*qiso[0]+.35*qiso[1]+.30*qiso[2]+.15*qiso[3]
    qouter = .50*qiso[0]+.35*qiso[1]+.15*qiso[2]
    merge,mask,near = peaks(img,ref,pk,cx,cy,A,q,cp,sp,noise,seeing,rgb)
    rad = A*U[:,None]
    px = cx+rad*(cp*C-q*sp*S)
    py = cy+rad*(sp*C+q*cp*S)
    cover = sample(mask,px,py)
    cover *= (px>0)&(px<127)&(py>0)&(py<127)
    v = rings(img,px,py,cover)
    rgbp = np.array([rings(ch,px,py,cover) for ch in rgb])
    # Faint spiral arms can be farther out than the antipodal isophotes.
    # A circular aperture there is biased by field stars, so instead take
    # a high, but not extreme, angular quantile in an elliptical annulus.
    eu = np.geomspace(.55,2.9,26)
    ex = cx+A*eu[:,None]*(cp*C-q*sp*S)
    ey = cy+A*eu[:,None]*(sp*C+q*cp*S)
    ecov = sample(mask,ex,ey)
    ev = rings(img,ex,ey,ecov)
    ev = np.where((ex>0)&(ex<127)&(ey>0)&(ey<127),ev,0.)
    upper = np.quantile(ev,.82,axis=1)
    lower = np.quantile(ev,.30,axis=1)
    wings = (ev.shape[0]-1-np.argmax((upper>max(.042*pk,4*noise))[::-1])) if np.any(upper>max(.042*pk,4*noise)) else 0
    reach = A*eu[wings]
    armextent = reducew(np.clip((upper-lower-1.5*noise)/(.085*pk+2*noise),0,1.3),np.exp(-.5*(np.log(eu/1.3)/.25)**2))
    mid = np.median(v,axis=1)
    # Robust local contrast in elliptical annuli; floor the denominator
    # by the actual noise so that empty sky cannot act like a galaxy arm.
    scale = np.maximum(mid,0)+2.5*noise+.022*pk+.0015
    dev = np.clip((v-mid[:,None])/scale[:,None],-1.5,1.6)
    coeff = 2*(MODES @ dev.T)/len(ANG)
    vis = ramp(mid+.025*pk,noise,4.5*noise+.01*pk)*ramp(A*U,1.3,4.)
    wo = vis*np.exp(-.5*(np.log(U/.79)/.46)**2)
    wm = vis*np.exp(-.5*(np.log(U/.50)/.48)**2)
    wi = vis*np.exp(-.5*(np.log(U/.35)/.43)**2)
    far = vis*np.exp(-.5*(np.log(U/1.20)/.36)**2)
    # Large-scale lopsidedness, middle and outer, and non-elliptical texture.
    odd = np.sqrt(np.abs(coeff[0])**2+.55*np.abs(coeff[2])**2)
    oddout = .50*reducew(odd,wo)+.50*reducew(odd,far)
    dipole = .45*abs(reducew(coeff[0],wo))+.55*abs(reducew(coeff[0],far))
    smallscale = np.sqrt(np.sum(abs(coeff[2:7])**2,axis=0))
    texture = reducew(smallscale,wo)
    om = np.maximum(0,np.abs(coeff[1])-1.1*noise/(scale+EPS)/np.sqrt(2+A*U*q))
    outer2 = reducew(om,wo)
    logw = vis*ramp(U,.16,.28)*(1-ramp(U,1.10,1.6))
    wide,tight = spiralbank(coeff[1],logw,2)
    w3,t3 = spiralbank(coeff[2],logw,3)
    atight, awide = arcfilters(rgbp,dev,mid,noise,vis,A,q)
    # Stable inner quadrupole. A round bulge under deprojection has its
    # long axis along the deprojected minor axis, unlike a projected bar.
    bar = 0.
    for center in (.24,.35,.47,.59):
        wb = vis*np.exp(-.5*(np.log(U/center)/.28)**2)
        quad = reducew(coeff[1],wb)
        mod = abs(quad)
        align = float(quad.real)/(EPS+mod)
        bulgelike = ramp(-align,.48,.97)*ramp(1-q,.08,.46)
        coherence = mod/(reducew(abs(coeff[1]),wb)+EPS)
        bar = max(bar,mod*(1-.76*bulgelike)*coherence*ramp(center*A,1.1,3.7))
    # An elongated inner isophote relative to the outer disk is independent
    # support for a bar. Use spin-two ellipticity for angle differences.
    inner = (1-qq[3:5])/(1+qq[3:5])*np.exp(2j*(pa[3:5]-theta))
    outer = (1-q)/(1+q)
    diff = inner-outer
    elong = np.mean(np.abs(diff)*(.35+.65*ramp(np.abs(inner),.06,.20)))
    # Compare the direction and strength of the inner and outer red-light
    # quadrupoles directly. A bar typically has a constant position angle
    # and ends where a rounder disk or differently oriented arms take over.
    redring = rgbp[0]
    redmean = np.mean(redring,axis=1)
    redmode = (2/len(ANG))*(np.clip(redring/(np.maximum(redmean,0)[:,None]+2*noise+EPS),-1.,3.) @ MODES[1])
    barmorph = 0.
    for r in (.31,.47,.65):
        w1 = vis*np.exp(-.5*(np.log(U/r)/.30)**2)
        w2 = vis*np.exp(-.5*(np.log(U/(2*r))/.25)**2)
        za = reducew(redmode,w1)
        zb = reducew(redmode,w2)
        if abs(za)>.005:
            excessbar = abs(za)-(za.conjugate()*zb).real/abs(za)
            steady = abs(za)/(reducew(abs(redmode),w1)+EPS)
            notbulge = 1-ramp(-za.real/abs(za),.53,.98)*ramp(1-q,.06,.41)
            barmorph=max(barmorph,excessbar*steady**2*notbulge*ramp(r*A,1.6,4.2))
    # Colour from azimuthal estimates; sky-subtracted fractional red/blue
    # excess outside the bulge and at its centre.
    colors = np.median(rgbp,axis=2)
    redblue = (colors[2]-colors[0])/(np.maximum(colors[2]+colors[0],0.)+.035*pk+2*noise+EPS)
    blue = reducew(redblue,wm*.4+wo*.6)
    corecolor = reducew(redblue, np.exp(-.5*(U/.18)**2))
    colgrad = blue-corecolor
    # Additional outer-envelope asymmetry. Use an independently smoothed
    # image and keep the sign of the dipole across a range of faint radii.
    vv = rings(broad,px,py,cover)
    mm = np.median(vv,axis=1)
    bn = max(.0009,1.4826*np.median(np.abs(broad[SKY])))
    rr = np.clip((vv-mm[:,None])/(np.maximum(0,mm[:,None])+2*bn+.009*pk),-1.4,2.)
    fv = (2/len(ANG))*(MODES[0] @ rr.T)
    wf = np.exp(-.5*(np.log(U/1.35)/.35)**2)*ramp(mm,.6*bn,3*bn+.003*pk)
    faintdip = abs(reducew(fv,wf))
    ellipticerr = ellipseerrors(img,mask,cx,cy,A,aa,qq,pa,pk,noise)
    pitchconf,slope = arcpitch(coeff,vis)
    asym = balancedasym(broad,mask,cx,cy,cp,sp,A,q,pk,bn)
    outline, fadetail = tidalshape(broad,mask,cx,cy,cp,sp,A,q,pk,bn)
    # Compare inner and outer widths and surface-brightness concentration.
    # A deprojected fourth harmonic distinguishes pointed disks from ovals.
    # Compare quadrupoles between the red and blue light. A red stellar
    # bar in a blue disk is more elliptical (or differently oriented) in
    # red even when the grayscale quadrupole is partly cancelled by arms.
    rz,bz = rgbp[0],rgbp[2]
    chromquad = []
    for z in (rz,bz):
        mean = np.mean(z,axis=1)
        polar = np.clip(z/(np.maximum(mean,0)[:,None]+noise*3+EPS),-.5,3.)
        chromquad.append(2*(polar @ MODES[1])/len(ANG))
    colorbar = abs(reducew(chromquad[0]-chromquad[1],wi))
    qinner = .55*qq[4]+.45*qq[3]
    # Seeing alone rounds the core appreciably in a very narrow object.
    qnull = math.sqrt(qouter**2+(1-qouter**2)*min(.8,2.4**2/max(1.,aa[4]*aa[3])))
    swelling = qinner-qnull
    corewidth = aa[4]/max(aa[1],2.)
    pointed = reducew(coeff[3].real,wm)
    # Major-axis cut at native resolution. Extrapolate the exponential disk
    # inwards from two radii on both sides. Bulge excess should be bright
    # and relatively spheroidal; dust can obscure the equatorial midline.
    uline = np.linspace(-1.3,1.3,105)
    vline = np.linspace(-.75,.75,61)
    ux = A*uline[None,:]
    vy = A*q*vline[:,None]
    cut = sample(raw,cx+cp*ux-sp*vy,cy+sp*ux+cp*vy)
    cut = (cut+cut[::-1,::-1]+cut[:,::-1]+cut[::-1,:])*.25
    profile = np.mean(cut[abs(vline)<=max(.10,.90/(A*q))],axis=0)
    core = max(.005,np.max(profile[np.abs(uline)<.10]))
    h1 = max(.005,np.median(profile[(abs(uline)>.29)&(abs(uline)<.44)]))
    h2 = max(.002,np.median(profile[(abs(uline)>.62)&(abs(uline)<.82)]))
    excess = math.log(core*h2/h1**2)
    # A central layer sticking out above the local disk. For an exponential
    # vertical profile without a bulge the ratio should not change with x.
    off = (np.abs(vline)>.34)&(np.abs(vline)<.63)
    wing = cut[off].mean(axis=0)
    thickcore = np.median(wing[abs(uline)<.10])/core
    thickdisk = np.median(wing[(abs(uline)>.29)&(abs(uline)<.44)])/h1
    thickening = thickcore-thickdisk
    light = np.maximum(0,mid-2*noise)*U**2
    light = light/(EPS+np.sum(light))
    half = A*U[min(len(U)-1,np.searchsorted(np.cumsum(light),.5))]
    # Distinguish a resolved exponential disk from a single softened
    # spheroid. A shoulder is excess disk light outside a compact bulge.
    lm = np.log(np.maximum(mid,0)+2*noise+EPS)
    lmid = np.interp([.18,.38,.72,1.12],U,lm)
    shoulder = (lmid[0]-lmid[1])-(lmid[1]-lmid[2])*.20/.34
    # Pale compact nuclei can be white, but the red-green excess in the
    # inner disk is another indication of an older bulge population.
    old = reducew((colors[0]-colors[2])/(.015+np.maximum(0,colors[0]+colors[2])),wi)
    # Apparent size is evidence about whether the substructure is actually
    # resolved, not a proxy for physical galaxy size.
    resolved = ramp(math.log(A),math.log(6.),math.log(19.))
    areares = ramp(A*math.sqrt(q),5.,15.)
    # Membership evidence, specified in units of a soft transition width.
    # Each group has a gate (disk/elliptical, inclined/face-on, interaction)
    # and a separate within-group decision, so ambiguous subtypes do not
    # artificially reduce the evidence for their whole family.
    spiral = (.90*ramp(max(wide,tight),.027,.13)+.55*ramp(max(w3,t3),.018,.090)
              +.90*ramp(texture,.055,.22)+.42*ramp(outer2,.06,.25)
              +.90*areares+.72*ramp(blue,-.095,.13)+.53*ramp(colgrad,.025,.16)
              +.30*ramp(bar,.09,.29)-2.16)
    # Strong colour and a distinct extended disk can reveal tightly wrapped
    # arms even where the separate narrow ridges are not well resolved.
    spiral += .25*ramp(.44-corewidth,.0,.19)*ramp(colgrad,0.,.12)
    arc = max(atight,awide)
    extension = ramp(reach*math.sqrt(q),10.,24.)
    spiral += .78*ramp(arc,.012,.065)*(.35+.65*areares)+.60*extension*ramp(armextent,.12,.65)
    # A luminous extended blue disk around a distinctly red core is a
    # spiral cue even under seeing which smooths away the fine ridges.
    spiral += .72*extension*ramp(blue+.45*colgrad,-.035,.13)
    # A straight bar can be bright even when little of an arm is visible.
    spiral += .62*ramp(barmorph,.065,.20)*ramp(areares+extension,.35,1.3)
    diskcolour = ramp(colgrad,.04,.19)*ramp(blue,-.13,.08)
    spiral += .55*diskcolour*ramp(shoulder,-.05,.33)*extension
    # Smooth spheroids can have a colour gradient or an elliptical inner
    # light profile. If nested ellipses explain that light to within small
    # fractional residuals, such a profile alone is not a resolved bar.
    smooth = 1.-ramp(ellipticerr,.038,.14)
    smooth *= 1.-.70*ramp(arc,.03,.10)*extension
    spiral -= 1.25*smooth*(1.-.45*diskcolour)
    spiral += .58*ramp(ellipticerr,.13,.29)*ramp(A*math.sqrt(q)/seeing,5.,11.)
    spiral -= 2.5*(1-ramp(qouter,.23,.53))
    # Very small smooth elongated objects lack evidence of a resolved
    # midplane. Require a resolved length in PSF units or disk structure.
    disklen = half/(seeing+.15)
    diskresolved = ramp(disklen,3.,9.)
    isdisk = .63*ramp(pointed,.015,.13)+.60*ramp(colgrad,.05,.20)+.53*ramp(thickening,.035,.19)
    edge = (min(1.55,(.345-qouter)/.12)+1.20*(diskresolved-.58)+.56*ramp(swelling,.035,.18)
            +isdisk+.23*ramp(blue,-.01,.16)-1.10*(1.-diskresolved))
    edge -= .67*ramp(qshape,.43,.64)
    # A very flat disk has a pointed outline even when it is smooth. A
    # moderately flattened spheroid with excellent elliptical isophotes
    # and no extended midplane should stay in the smooth family instead.
    edge -= .70*smooth*(1-diskresolved)*ramp(qouter,.18,.33)
    # Relative strength of a central spheroid against a thin extended disk.
    bulge = (1.00*ramp(swelling,-.01,.17)+.70*ramp(.38-corewidth,-.03,.19)
             +1.40*ramp(excess,-.13,.60)+1.10*ramp(thickening,.01,.26)
             +.65*ramp(-blue,-.03,.16)+.82*ramp(old,.01,.19)
             +.45*ramp(colgrad,.055,.24)+.58*ramp(qouter,.16,.39)-2.35)
    # Disordered light differs from even, winding arms: weight a dipole
    # that persists into the faint envelope more than small-scale knots.
    disturbed = (1.3*ramp(dipole,.08,.31)+1.05*ramp(oddout,.14,.38)
                 +1.65*ramp(faintdip,.10,.43)+.62*ramp(near,.10,.59)
                 +.38*ramp(texture,.14,.33)-2.00)
    disturbed = .53*disturbed+2.65*ramp(asym,.065,.28)-.95
    disturbed -= .62*ramp(max(wide,tight),.045,.15)+.28*ramp(bar,.1,.3)
    # A merger needs two nuclei, but an unmistakable asymmetric
    # silhouette needs no companion. Carry their evidence into the final
    # comparison independently of how bright or resolved the disk is.
    tidal = (1.85*ramp(outline,.095,.29)+.80*ramp(fadetail,.035,.18)
             +.70*ramp(asym,.09,.26)+.48*ramp(faintdip,.15,.42)-1.15)
    tidal -= .55*ramp(max(wide,tight),.05,.17)*ramp(awide+atight,.04,.16)
    disturbed = max(tidal, .65*disturbed+.45*tidal)
    interaction = (merge-.52)*5.7
    interaction += .42*ramp(dipole,.09,.32)+.23*ramp(faintdip,.13,.40)
    # Force strong companion evidence to stand on its own: a merging pair
    # is not required to have a disordered primary.
    regular = max(0.,spiral,edge)
    # Disk structure can be present in a disturbed galaxy or a merger.
    # Do not give the regular alternative an unbounded advantage simply
    # by counting every detectible disk feature after a disturbance.
    conflict = .33*min(regular,1.8)
    categories = [disturbed+regular-conflict,interaction+regular-conflict,0.,spiral,edge]
    family = int(np.argmax(categories))
    if family==0:
        return 0
    if family==1:
        return 1
    if family==2:
        if qshape>.82:
            return 2
        if qshape<.435:
            return 4
        return 3
    if family==4:
        return 9 if bulge>0. else 8
    # Within the spiral family, compare linear bar support with rotating
    # patterns. Open arms are often high contrast and well separated.
    barvote = (bar-.19)/.13+.82*ramp(elong,.065,.22)-.42*ramp(max(wide,tight),.08,.22)
    loose = ((wide-.064)/.09+.62*np.clip((wide-tight)/.055,-1.,1.)
             +.72*ramp(outer2,.13,.32)+.42*ramp(oddout,.17,.36)
             +.48*ramp(blue,.015,.22)+.36*ramp(texture,.12,.30)-.65)
    barvote += .65*ramp(colorbar,.045,.19)+.38*ramp(elong,.07,.21)*ramp(colgrad,.04,.19)
    barvote += .92*ramp(barmorph,.08,.24)
    loose += .85*ramp(awide,.027,.12)+.60*np.clip((awide-atight)/.043,-1.,1.)
    loose += .50*ramp(armextent,.28,.87)*ramp(outer2,.07,.20)
    # A pitch above about 25 degrees is open. Use a phase measurement
    # only if there is signal of one handedness across multiple radii.
    loose += 1.10*pitchconf*np.clip((2.15-slope)/1.1,-1.,1.)
    barvote -= .50*smooth*(1.-extension)
    if barvote>max(0.,.60*loose):
        return 5
    if loose>0.:
        return 7
    return 6


def predict(x):
    return np.array([analyse(frame) for frame in x], dtype=np.int64)
