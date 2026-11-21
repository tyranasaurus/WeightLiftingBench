"""Write the self-contained morphology program. No data or assets needed."""
from pathlib import Path
import sys

SOURCE = '''import numpy as np
import math

# Fixed geometry and Gaussian transfer functions. Pixel centres are at integers.
line = np.arange(128, dtype=float)
ygrid, xgrid = np.meshgrid(line, line, indexing="ij")
wave2 = np.fft.fftfreq(128)[:, None]**2 + np.fft.rfftfreq(128)[None, :]**2

def kernel(width):
    return np.exp((-2 * np.pi**2 * width**2) * wave2)

kone, ktwo, kfour, keight = [kernel(s) for s in (1.0, 2.0, 4.0, 8.0)]
ang = np.arange(96) * (2*np.pi/96)
cang, sang = np.cos(ang), np.sin(ang)
rad = np.geomspace(0.25, 142, 122)
fracrad = np.geomspace(0.1, 1.65, 48)
step = np.log(fracrad[1]/fracrad[0])

def clip(x, a=0., b=1.):
    return np.minimum(np.maximum(x, a), b)

def ramp(x, a, b):
    return clip((x-a)/(b-a))

def blur(f, k):
    return np.fft.irfft2(f*k, s=(128,128), axes=(-2,-1))

def sample(im, x, y):
    x = clip(x, 0, 126.99999)
    y = clip(y, 0, 126.99999)
    i, j = x.astype(int), y.astype(int)
    a, b = x-i, y-j
    return (1-b)*((1-a)*im[j,i]+a*im[j,i+1]) + b*((1-a)*im[j+1,i]+a*im[j+1,i+1])

def dirs(q, pa):
    co, si = np.cos(pa), np.sin(pa)
    return co*cang-si*q*sang, si*cang+co*q*sang

def rays(im, cx, cy, r, q, pa):
    u,v = dirs(q,pa)
    return sample(im, cx+r[:,None]*u, cy+r[:,None]*v)

def border(im):
    return np.concatenate((im[:12,:].ravel(), im[-12:,:].ravel(), im[12:-12,:12].ravel(),im[12:-12,-12:].ravel()))

def noiseborder(im):
    lo,med,hi = np.quantile(border(im), [0.2,0.48,0.7])
    return med, max((hi-lo)/1.37,0.0015)

def eigshape(xx, xy, yy):
    d = np.hypot(xx-yy, 2*xy)
    return np.sqrt(np.maximum((xx+yy+d)*0.5, .01)), np.sqrt(np.maximum((xx+yy-d)*0.5, .01)), np.arctan2(2*xy,xx-yy)*.5

def contours(im, cx, cy, q, pa, lev):
    u,v = dirs(q, pa)
    xx, yy = cx+rad[:,None]*u, cy+rad[:,None]*v
    p = sample(im, xx, yy)
    p[(xx<0)|(xx>127)|(yy<0)|(yy>127)] = -1
    low = p[None,:,:] < lev[:,None,None]
    # Require a persistent crossing, not an isolated noise dip.
    under = low.copy()
    under[:,:-2] &= low[:,1:-1] | low[:,2:]
    under[:,-1] = True
    ix = np.argmax(under,axis=1)
    before = np.maximum(ix-1,0)
    cols = np.arange(96)[None,:]
    frac = clip((p[before,cols]-lev[:,None])/np.maximum(p[before,cols]-p[ix,cols],1.e-6))
    r = rad[before]*(1-frac)+frac*rad[ix]
    # Angular Hampel cap: another object intercepted by just a few rays.
    med = np.median(np.stack([np.roll(r,j,axis=1) for j in (-5,-3,0,3,5)]),axis=0)
    # Only cap an isolated spike when the local radial function is itself
    # smooth enough; an ellipse sampled in the wrong plane can have steep
    # but perfectly legitimate variations around its minor/major axes.
    neighbour = np.maximum(np.roll(r,1,axis=1),np.roll(r,-1,axis=1))
    r = np.minimum(r,np.maximum(1.13*med+.22,neighbour*1.06))
    den = np.mean(r**2,axis=1)+1.e-8
    dx = (2/3)*np.mean(u*r**3,axis=1)/den
    dy = (2/3)*np.mean(v*r**3,axis=1)/den
    xx = .5*np.mean(u*u*r**4,axis=1)/den-dx**2
    yy = .5*np.mean(v*v*r**4,axis=1)/den-dy**2
    xy = .5*np.mean(u*v*r**4,axis=1)/den-dx*dy
    a,b,theta = eigshape(xx,xy,yy)
    a,b = a*2,b*2
    # The area deficit relative to an ellipse of the same inertia.
    deficit = 1-q*den/np.maximum(a*b,.1)
    return a,b,theta,dx,dy,deficit

def anglemean(t,w):
    return .5*np.angle(np.sum(w*np.exp(2j*t)))

def nuclei(im, sm, broad, cc, cx, cy, peak, a, b, pa, sig):
    local = sm.copy()
    for i,j in ((1,0),(-1,0),(0,1),(0,-1),(-1,1),(1,-1),(-1,-1),(1,1)):
        local = np.maximum(local,np.roll(sm,(i,j),axis=(0,1)))
    ds = np.hypot(xgrid-cx,ygrid-cy)
    ys,xs = np.nonzero((sm>=local)&(sm>.19*peak+2*sig)&(ds>2.7)&(ds<49))
    pair = 0.
    mask = np.ones_like(im)
    corecolor = np.array([sample(c,cx,cy) for c in cc])
    corecolor = corecolor / max(np.sum(corecolor),.05)
    for x,y in zip(xs,ys):
        p = sm[y,x]
        d = ds[y,x]
        # Contrast of this local maximum above the saddle toward the nucleus.
        t = np.linspace(.07,.93,20)
        between = sample(im,cx+t*(x-cx),cy+t*(y-cy))
        top = im[y,x]
        saddle = max(0.,np.min(between))
        prom = (top-saddle)/max(top,.02)
        if prom < .065:
            continue
        # Compare the companion to the primary on several angular scales.
        halo = max(broad[y,x],0.)
        corehalo = max(sample(broad,cx,cy),.02)
        col = np.maximum(cc[:,y,x],0.)
        col = col / max(np.sum(col),.05)
        cd = np.linalg.norm(col-corecolor)
        # Relative light on the side AWAY from the primary tests whether
        # the peak is a resolved nucleus or an unrelated foreground star.
        ux,uy = (x-cx)/d,(y-cy)/d
        du,dv = ux*cang-uy*sang,uy*cang+ux*sang
        side = cang>-.2
        f3 = np.mean(sample(im,x+3*du[side],y+3*dv[side]))/max(top,.01)
        f5 = np.mean(sample(im,x+5*du[side],y+5*dv[side]))/max(top,.01)
        resolved = clip(.57*ramp(f3,.27,.67)+.43*ramp(f5,.13,.49))
        similarity = np.exp(-((cd/.22)**2))
        rel = np.sqrt(max(p/peak,.001)*max(halo/corehalo,.001))
        near = np.exp(-max(d-9.,0.)/max(1.35*a,10.))
        bridge = ramp(saddle/max(min(peak,top),.02),.06,.40)
        score = (.15+.85*ramp(rel,.14,.76))*(.24+.76*resolved)*(.60+.4*similarity)*(.69+.31*bridge)*near*ramp(prom,.07,.37)
        # A blue HII knot against its own disk can be resolved but does not
        # have the red central concentration of a second stellar nucleus.
        chroma = corecolor[0]-corecolor[2] - col[0]+col[2]
        score *= 1-.60*ramp(chroma,.045,.22)
        # A substantially brighter, already separated star cannot acquire
        # merger evidence solely from its absolute halo brightness.
        score *= 1-.70*ramp(rel,1.5,3.)*(1-bridge)
        pair = max(pair,score)
        if d>max(4.,.15*a) and ((resolved<.45 and prom>.22) or cd>.22 or (prom>.55 and d>.8*a)):
            # Exclude compact interlopers from disc photometry (not from
            # the preceding merger test). Size follows peak width.
            away = np.arange(1.,13.)[:,None]
            prof = np.mean(sample(im,x+away*du[side],y+away*dv[side]),axis=1)
            rr = min(9.,1.+np.sum(prof>.18*top))
            mask[(xgrid-x)**2+(ygrid-y)**2<rr**2]=0.
    return pair,mask

# Phase advance of a logarithmic spiral per unit log radius is cot(pitch).
slopes = np.array([.60,1.05,1.50,2.0,2.65,3.4,4.3,5.4,6.8,8.6])
slopes = np.concatenate((-slopes[::-1],slopes))
spiralwaves = np.exp(1j*np.arange(2,5)[:,None,None]*slopes[None,:,None]*np.log(fracrad)[None,None,:])

def spiraltest(harm, wt):
    tight, loose = 0.,0.
    # Remove the stationary quadrupole before testing for winding. Multiple
    # apertures allow arms to begin outside a bar, or fade before the edge.
    for lo,hi in ((.24,1.16),(.38,1.39),(.60,1.65)):
        w = wt*((fracrad>=lo)&(fracrad<=hi))
        if w.sum()<1.e-5:
            continue
        w = w/w.sum()
        for k in range(3):
            z = harm[:,k+2]
            z = z-np.sum(z*w)
            waves = spiralwaves[k]
            cov = np.sum(waves*(z*w)[None,:],axis=1)
            gain = np.sqrt(np.maximum(.20,1-abs(np.sum(waves*w,axis=1))**2))
            strength = abs(cov)/gain
            # Suppress standing waves (e.g. change of ellipticity).
            strength *= .35+.65*clip((strength-strength[::-1])/(strength+.018))
            if k>0:
                strength *= .88
            tight = max(tight,np.max(strength[abs(slopes)>=2.65]))
            loose = max(loose,np.max(strength[abs(slopes)<=2.0]))
    return tight,loose

def outskirts(broad, mask, cx, cy, scale, qo, pa, peak):
    # Divide the halo into approximately independent seeing-sized sectors.
    # A tail should be detected on adjoining radial shells and at a stable
    # azimuth, not just in a single bright off-centre pixel.
    r = scale*np.geomspace(.62,3.15,21)
    u,v = dirs(qo,pa)
    x,y = cx+r[:,None]*u,cy+r[:,None]*v
    s = sample(broad,x,y)
    ok = sample(mask,x,y)>.85
    bg, noise = noiseborder(broad)
    noise = max(noise,.002)
    s = np.maximum(s-bg,0.)
    other = np.roll(s,48,axis=1)
    ok &= np.roll(ok,48,axis=1)&(x>2)&(x<125)&(y>2)&(y<125)
    light = (s+other)/2
    odd = (s-other)*ok
    # Angular coherence rejects star residuals and background peaks.
    dip = abs(np.mean(odd*np.exp(1j*ang),axis=1))/(np.mean(light,axis=1)+noise*3+.01*peak)
    dip *= ramp(np.mean(ok,axis=1),.6,.92)
    # A localized plume is diluted in the global dipole. Also check for
    # unilateral emission extending over a radial chain of sectors.
    feat = np.maximum(odd-2.5*noise,0.)/(light+.075*peak+noise*4)
    chain = np.minimum(np.minimum(feat[:-4],feat[2:-2]),feat[4:])
    chain = sum(np.roll(chain,j,axis=1) for j in (-7,-4,0,4,7))/5
    return np.max((dip[:-4]+dip[2:-2]+dip[4:])/3),np.max(chain)

def edgeprofile(im,cx,cy,scale,pa,minor,peak):
    # Fold the light profile about each axis. Quantify how far an inclined
    # disk projects BEYOND the spheroid on the major axis. Fourfold folding
    # makes a chance neighbour on one side insufficient evidence.
    t = np.linspace(-1.6,1.6,97)
    z = np.linspace(-1.3,1.3,65)
    co,si = np.cos(pa),np.sin(pa)
    v = scale*t[None,:]
    u = max(minor,1.)*z[:,None]
    p = sample(im,cx+co*v-si*u,cy+si*v+co*u)
    fold = .5*(p+p[::-1])
    fold = .65*np.minimum(fold,fold[:,::-1])+.35*.5*(fold+fold[:,::-1])
    # Thickness above the midplane at constant fraction of LOCAL brightness.
    # A constant scale-height disk has little central rise in this quantity.
    mid = np.maximum(fold[len(z)//2],.012*peak)
    width = np.sum(clip(fold/mid[None,:]-.22),axis=0)
    center = np.mean(width[abs(t)<.13])
    wing = np.mean(width[(abs(t)>.45)&(abs(t)<.82)])
    pinch = center/max(wing,.1)
    # Compare silhouettes to a smooth ellipse at a faint isophote: the
    # disk+bulge system has narrower waists along an extended, thin disk.
    vals = []
    for lev in (.09,.14,.20):
        height = np.sum(clip((fold/peak-lev)*22+.5),axis=0)*.5
        length = np.sum(clip((mid/peak-lev)*22+.5))*.5
        tn = abs(t)/max(length*(t[1]-t[0]),.01)
        hm = max(np.max(height[abs(t)<.10]),.05)
        at = (tn>.52)&(tn<.82)
        if np.any(at):
            vals.append(np.mean(height[at]/hm/np.sqrt(1-tn[at]**2)))
    waist = 1.-np.mean(vals) if len(vals) else 0.
    return pinch,waist

def components(binary,cx,cy):
    # Run-length connected-component labelling with union by leftmost ID.
    # Only the component actually joined to the nucleus enters shape CAS.
    parent = []
    def root(v):
        while parent[v]!=v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    dy = np.diff(np.pad(binary,((0,0),(1,1))).astype(np.int8),axis=1)
    rows,left = np.nonzero(dy>0)
    right = np.nonzero(dy<0)[1]
    counts = np.concatenate(([0],np.cumsum(np.bincount(rows,minlength=128))))
    keep = -1
    for y in range(128):
        prior = counts[max(0,y-1)]
        for t in range(counts[y],counts[y+1]):
            l,r = left[t],right[t]
            parent.append(t)
            while prior<counts[y] and right[prior]<l:
                prior+=1
            k = prior
            while k<counts[y] and left[k]<=r:
                a,b = root(t),root(k)
                if a!=b:
                    parent[max(a,b)]=min(a,b)
                k+=1
            if y==int(round(cy)) and l<=int(round(cx))<r:
                keep = t
    out = np.zeros((128,128))
    if keep>=0:
        keep = root(keep)
        for t in range(len(left)):
            if root(t)==keep:
                out[rows[t],left[t]:right[t]]=1.
    return out

def segshape(sm,broad,mask,cx,cy,peak,scale,qo,pa):
    noise = noiseborder(sm)[1]
    # Limit chance connections far away from the target; ignore masked
    # points on both sides when comparing the shape to its rotation.
    co,si = np.cos(pa),np.sin(pa)
    dx,dy = xgrid-cx,ygrid-cy
    radius = np.hypot(co*dx+si*dy,(-si*dx+co*dy)/max(qo,.50))
    limit = (radius<max(2.6*scale,18.)) & (np.hypot(dx,dy)<61.)
    def asymmetry(obj):
        m = obj*mask
        area = np.sum(m)+1.e-6
        # Rotated mask allows a foreground star to be omitted on BOTH sides.
        asym = 1.
        for ox,oy in ((0.,0.),(.65,0.),(-.65,0.),(0.,.65),(0.,-.65)):
            xr,yr = cx+ox-dx,cy+oy-dy
            other = sample(obj,xr,yr)
            valid = sample(mask,xr,yr)*mask
            asym = min(asym,np.sum(abs(m-other)*valid)/(2*area))
        return asym
    cut = np.hypot(.047*peak,2.8*noise)
    obj = components((sm>cut)&limit,cx,cy)
    denom = max(obj.sum(),1.)
    distance = np.hypot(np.sum(obj*dx)/denom,np.sum(obj*dy)/denom)/max(np.sqrt(denom/np.pi),1.)
    sym = asymmetry(obj)
    # Compare the faint component inertia to the inner bright contours.
    mx = np.sum(obj*dx)/denom; my = np.sum(obj*dy)/denom
    a,b,t = eigshape(np.sum(obj*dx**2)/denom-mx**2,np.sum(obj*dx*dy)/denom-mx*my,np.sum(obj*dy**2)/denom-my**2)
    flatten = b/max(a,.1)
    return sym,distance,flatten

def classify(row):
    image = .32*row[0]+.43*row[1]+.25*row[2]
    f = np.fft.rfft2(image)
    im = blur(f,kone)
    sm = blur(f,ktwo)
    broad = blur(f,kfour)
    bg, sig = noiseborder(im)
    im,sm,broad = im-bg,sm-bg,broad-bg
    cc = blur(np.fft.rfft2(row,axes=(-2,-1)),kone)
    for c in cc:
        c -= np.median(border(c))
    # A small pointing prior, rather than the brightest pixel in the frame.
    dx = xgrid-63.5
    dy = ygrid-63.5
    loc = (np.maximum(sm,.0)+.4*np.maximum(broad,0.))*np.exp(-(dx**2+dy**2)/(2*5.**2))
    iy,ix = np.unravel_index(np.argmax(loc),loc.shape)
    if loc[iy,ix]<.0001:
        return 2
    cx,cy = float(ix),float(iy)
    # Subpixel centring on the nucleus is important for odd Fourier modes.
    for width in (4.,2.4,1.6,1.6):
        dx,dy = xgrid-cx,ygrid-cy
        w = np.exp(-(dx**2+dy**2)/(2*width**2))*np.maximum(im,0.)**2
        cx += np.sum(w*dx)/(np.sum(w)+1.e-10)
        cy += np.sum(w*dy)/(np.sum(w)+1.e-10)
    peak = max(sample(im,cx,cy),.015)
    lev = np.sqrt((peak*np.array([.72,.54,.38,.26,.17,.105,.065]))**2+(2.4*sig)**2)
    lev = np.minimum(lev,.91*peak)
    a,b,theta,ox,oy,deficit = contours(im,cx,cy,1.,0.,lev)
    q0 = clip(np.median(b[2:6]/a[2:6]),.1,1.)
    t0 = anglemean(theta[2:6],(1-b[2:6]/a[2:6])+.01)
    a,b,theta,ox,oy,deficit = contours(im,cx,cy,q0,t0,lev)
    # Bright contours can have a very different inclination and PA to the
    # outer disk. Retrace them in their own projected coordinate system.
    ain,bin,tin,oxn,oyn,din = contours(im,cx,cy,clip(np.median(b[:3]/a[:3]),.1,1.),anglemean(theta[:3],1.01-b[:3]/a[:3]),lev[:3])
    a[:3],b[:3],theta[:3],ox[:3],oy[:3],deficit[:3] = ain,bin,tin,oxn,oyn,din
    # Correct quadrature broadening of isophotal radii by the PSF and blur.
    smear = 2*np.log(peak/lev)*1.4**2
    ratio = clip(np.sqrt(np.maximum(b**2-smear,.40*b**2)/np.maximum(a**2-smear,.40*a**2)),.055,1.)
    q = np.median(ratio[2:6])
    qo = np.median(b[3:6]/a[3:6])
    pa = anglemean(theta[3:6],(1-b[3:6]/a[3:6])+.015)
    scale = max(a[5],2.)
    pair,mask = nuclei(im,sm,broad,cc,cx,cy,peak,scale,b[5],pa,sig)
    # Approximate the disk plane using the outer contours. The PSF is in
    # the observed image, so use observed (not deblurred) disc flattening.
    qo = clip(qo,.10,1.)
    u,v = dirs(qo,pa)
    r = fracrad[:,None]*scale
    xx,yy = cx+r*u,cy+r*v
    p = sample(im,xx,yy)
    m = (sample(mask,xx,yy)>.75)&(xx>1)&(xx<126)&(yy>1)&(yy<126)
    # Winsorise angular outliers; interpolate small masked arcs so that a
    # mask edge isn't interpreted as a spiral arm. No inter-image statistics.
    med = np.median(p,axis=1)
    p = np.where(m,p,med[:,None])
    med = np.median(p,axis=1)
    p = np.minimum(p,(np.quantile(p,.88,axis=1)*1.35+.035*peak+sig)[:,None])
    flux = p.mean(axis=1)
    norm = np.maximum(flux,0.)+.045*peak+2*sig
    harm = np.fft.rfft(p,axis=1)/48/norm[:,None]
    harm = harm[:,:11]
    # Downweight sky-dominated, undersampled and truncated rings.
    wt = ramp(flux/peak,.023,.105)*ramp(fracrad*scale,2.,4.)*ramp(m.mean(axis=1),.48,.90)
    def band(z,low,high):
        w = wt*((fracrad>=low)&(fracrad<=high))
        return np.sum(z*w)/(np.sum(w)+1.e-8)
    tight,loose = spiraltest(harm,wt)
    h = abs(harm)
    hspec = np.sqrt(np.sum(h[:,3:9]**2,axis=1))
    hp = band(hspec,.40,1.24)
    quad = band(h[:,2],.38,1.25)
    dip = band(h[:,1],.46,1.38)
    even = band(np.sqrt(h[:,2]**2+h[:,4]**2+h[:,6]**2),.46,1.38)
    # Measure asymmetry of faint isophotes separately. A tidal plume can
    # be visually dominant but contribute very little total flux.
    co,si = np.cos(pa),np.sin(pa)
    def offset(dx,dy):
        return np.hypot(co*dx+si*dy,(-si*dx+co*dy)/qo)
    shifts = offset(ox,oy)/np.maximum(a,1.)
    skew = np.median(shifts[3:])
    tail = np.mean(shifts[-2:])
    dshape = np.mean(deficit[-3:])
    # Coherent displacement of isophotes, as opposed to even spiral modes.
    disturb = .34*ramp(skew,.075,.25)+.23*ramp(tail,.105,.31)+.24*ramp(dip,.12,.39)+.12*ramp(dip/(even+.06),.80,1.8)+.07*ramp(dshape,.025,.13)
    # Disc colours are estimated away from the sky and foreground objects.
    rgb = np.stack([sample(c,xx,yy) for c in cc])
    rgb = np.median(rgb,axis=2)
    core = np.mean(rgb[:,(fracrad>=.10)&(fracrad<=.26)],axis=1)
    disc = np.mean(rgb[:,(fracrad>=.54)&(fracrad<=1.0)],axis=1)
    redin = (core[0]-core[2])/(max(core[0]+core[2],.02))
    redout = (disc[0]-disc[2])/(max(disc[0]+disc[2],.02))
    # Partial sky subtraction does not turn equal RGB gains into physical
    # fluxes; colour differences are used only as weak population cues.
    halo,plume = outskirts(broad,mask,cx,cy,scale,qo,pa,peak)
    tidal = max(ramp(halo,.18,.52),ramp(plume,.43,1.05))
    disturb = .70*disturb+.30*tidal
    # Blue passband unsharp-mask: remove the stationary m=0,1,2 modes of
    # each ellipse, then measure resolved fluctuations at adjacent scales.
    # A large smooth colour gradient should produce no arm texture here.
    bp = sample(cc[2],xx,yy)
    redp = sample(cc[0],xx,yy)
    bbase = np.median(bp,axis=1)
    rbase = np.median(redp,axis=1)
    bp = np.where(m,np.minimum(bp,(bbase*1.65+.08*peak)[:,None]),bbase[:,None])
    redp = np.where(m,np.minimum(redp,(rbase*1.65+.08*peak)[:,None]),rbase[:,None])
    bfour = np.fft.rfft(bp,axis=1)/48
    rfour = np.fft.rfft(redp,axis=1)/48
    denb = np.maximum(bbase+.045*peak+2*sig,.035*peak)
    denr = np.maximum(rbase+.045*peak+2*sig,.035*peak)
    bf = bfour[:,2:10]/denb[:,None]
    rf = rfour[:,2:10]/denr[:,None]
    # If the structure really exists, it should persist across radius AND
    # filter band. Cross-power does not mistake pixel noise for real arms.
    cross = np.real((bf[2:,1:]*rf[:-2,1:].conj()+rf[2:,1:]*bf[:-2,1:].conj())*.5).sum(axis=1)
    tx = np.sqrt(max(0.,band(np.pad(cross,(1,1)),.36,1.22)))
    # Band-limited pattern that turns in the SAME direction in multiple
    # passbands. Seeing smooths adjacent radii together, but does not give
    # sky noise a common winding sense in the blue and red filters.
    bharm = np.fft.rfft(bp,axis=1)/48/denb[:,None]
    rharm = np.fft.rfft(redp,axis=1)/48/denr[:,None]
    wchir = 0.
    for mode in (2,3,4):
        bz,rz = bharm[:,mode],rharm[:,mode]
        for low,high in ((.25,1.),(.45,1.4)):
            w = np.sqrt(wt[3:]*wt[:-3])*((fracrad[1:-2]>low)&(fracrad[1:-2]<high))
            den = np.sum(w)+1.e-8
            crosslag = .5*(bz[3:]*rz[:-3].conj()+rz[3:]*bz[:-3].conj())
            rot = abs(np.imag(np.sum(w*crosslag)/den))
            wchir = max(wchir,np.sqrt(rot))
    # Break in radial surface brightness. For a Sersic spheroid the
    # magnitude of the log-log slope increases monotonically. A disk arm
    # or ring frequently interrupts that progression, in particular in g.
    def bend(prof):
        prof = np.log(np.maximum(prof,.03*peak))
        grad = -(prof[6:]-prof[:-6])/(6*step)
        w = wt[3:-3]
        g = np.maximum(grad[:-6]-grad[6:],0.)
        ok = (fracrad[9:-3]<1.08)&(fracrad[3:-9]>.19)
        return np.max(g*w[6:]*w[:-6]*ok)
    shoulder = min(bend(bbase),bend(flux))
    # Sky noise in the redder passband and a seeing-limited core can wash
    # out a spiral in the broadband profile: use chromatic structure too.
    bluearms = band(np.sqrt(np.sum(abs(bf[:,1:])**2,axis=1)),.38,1.18)
    discblue = ramp(-redout,-.01,.24)
    bluer = ramp(redin-redout,.03,.21)
    diskpop = .60*discblue+.40*bluer
    # High fidelity rotational asymmetry in the luminous body. Weak noise
    # and neighbours should not cause a one-sided excess over several rings.
    kin = band(h[:,1],.28,.90)
    dipphase = abs(band(harm[:,1],.4,1.3))/(dip+.018)
    crooked = ramp(max(kin,.85*dip),.065,.24)*ramp(dipphase,.35,.84)
    oddmajor = ramp(dip/(quad+.045),.63,1.7)
    # Projected flatness of each inner ellipse and its excess elongation
    # relative to the outer disk. A bulge alone is ROUND in projection.
    qi = clip(b/a,.08,1.)
    ell = (1-qi)/(1+qi)*np.exp(2j*(theta-pa))
    outerell = np.median(ell[4:].real)+1j*np.median(ell[4:].imag)
    be = np.maximum(0.,np.abs(ell[:4]-outerell)-np.maximum(0.,abs(outerell)-abs(ell[:4])))
    be *= ramp(1-qi[:4],.06,.27)
    bar = np.max(np.minimum(be[:3],be[1:4]))
    # Is the inner elongation maintained over a range of radii?
    innerq = np.median(qi[:3])
    twist = abs(np.sin(theta[1]-pa))*(1-innerq)*(1-qo)
    barsignal = bar + .20*twist
    # Reserve independent evidence for a straight bar in the bright half:
    # constant orientation, mostly quadrupolar (not an off-centre clump).
    innerbar = min(be[0],be[1])*abs(np.cos(theta[0]-theta[1]))**4
    barbright = max(innerbar, min(be[1],be[2])*abs(np.cos(theta[1]-theta[2]))**4*.88)
    # In red light a bar normally remains elongated and straight. In an
    # open spiral, a quadrupole confined to the blue arms may mimic this in
    # white light, but the old central component is quite round.
    rpeak = max(sample(cc[0],cx,cy),.01)
    innerpa = anglemean(theta[:3],1.01-qi[:3])
    ra,rb,rt,rx,ry,rd = contours(cc[0],cx,cy,innerq,innerpa,np.array([.67,.48,.34])*rpeak)
    redq = np.median(rb/ra)
    redbar = (1-redq)*abs(np.cos(rt[1]-innerpa))**2
    # Deprojected radial fall-off. A large flattened blue envelope about a
    # red central component is disk evidence even if arm pitch is unresolved.
    # For Sersic law I=I0 exp(-k*r**(1/n)), two contours give a local n
    # from their log radius ratio. A disc is closer to n=1 than n=4.
    hl = np.sqrt(np.maximum(a*b,.1))
    slope = np.log(hl[5]/max(hl[2],.1))/max(np.log(np.log(lev[5]/peak)/np.log(lev[2]/peak)),.05)
    # Disk texture should also protrude above the smooth red light; use a
    # radial high-pass on g flux in the deprojected plane. Tight spirals
    # often give coherent circular shoulders rather than low angular modes.
    discprob = ramp(shoulder,.08,.43)*(.40+.60*diskpop)*ramp(scale*np.sqrt(qo),6.,15.)
    # Size and structure jointly distinguish a resolved disc from a compact
    # smooth spheroid. Strong non-axisymmetric light overrides size alone.
    discsize = scale*np.sqrt(qo)
    winding = max(tight,loose)
    spiral = (.28*ramp(discsize,7.,19.)+.22*ramp(hp,.075,.235)+.24*ramp(winding,.024,.116)+.11*ramp(quad,.064,.24)+.15*ramp(barsignal,.07,.23))
    spiral += .10*ramp(discsize,14.,27.)+.06*ramp(redin-redout,.055,.22)
    # A giant smooth spheroid can have the same diameter as a spiral. A
    # disk should also show excess outer light relative to a smooth fading
    # spheroid, colour stratification, or genuinely resolved structure.
    structure = max(ramp(hp,.055,.16),ramp(winding,.016,.08),ramp(barsignal,.07,.18))
    spiral -= .15*(1-structure)
    spiral += .08*discblue*ramp(discsize,7.,14.)
    # Independent disk evidence: spatially resolved light combined with
    # population change, resolved knots or a bar. This route is important
    # for tightly wound arms (nearly circular rings seen at small scale).
    resolved = ramp(discsize,5.,15.)
    diskspectrum = .25*resolved + .30*diskpop + .22*ramp(tx,.018,.100)+.13*ramp(bluearms,.045,.15)+.10*ramp(barbright,.035,.155)
    physical = max(ramp(wchir,.019,.065),ramp(shoulder,.18,.78),ramp(barbright,.07,.19))
    spiral = max(spiral,.70*diskspectrum*(.62+.38*physical))
    # Star-forming texture alone isn't specific to arm pitch; only its
    # coherent part should promote a blue smooth object to the disk branch.
    spiral += .11*physical*ramp(discsize,5.,12.)
    spiral += .12*discprob+.08*ramp(slope,1.25,.78)*resolved*diskpop
    if q<.53:
        spiral -= .12*(1-ramp(wchir,.034,.12))
    # Expect many more thin disks among highly resolved tapered objects
    # than among barely resolved, moderately flattened ellipses.
    thin = .61*ramp(q,.46,.19)+.27*ramp(scale,10.,28.)+.12*ramp(1-qi[5],.59,.80)
    thin += .10*ramp(q,.34,.20)*ramp(redin-redout,.035,.17)
    pinch,waist = edgeprofile(im,cx,cy,scale,pa,b[5],peak)
    rounder = np.median(qi[:3])-np.median(qi[4:6])
    qb = np.median(ratio[4:])
    # Check for a compact central component by three independent cues, so
    # neither a red colour nor a particular bulge orientation is required.
    bulgeshape = .40*ramp(waist,.05,.34)+.35*ramp(pinch,1.02,1.65)+.25*ramp(rounder,.01,.16)
    redcore = ramp(redin,-.012,.23)
    # The red band traces the old stellar bulge. If it projects higher
    # above the plane than the blue light, a mid-plane dust lane or a
    # superposed young disk can conceal a bulge in the total intensity.
    yoff = max(.4*b[3],1.25)
    px = np.array([cx-si*yoff,cx+si*yoff])
    py = np.array([cy+co*yoff,cy-co*yoff])
    rc = np.mean(sample(cc[0],px,py)); bc = np.mean(sample(cc[2],px,py))
    vertred = (rc-bc)/max(rc+bc,.02)
    conc = np.log(max((a[5]+.2)/(a[1]+.2),1.))
    concentration = ramp(conc,np.log(2.),np.log(4.5))
    stellarpop = .65*redcore+.35*ramp(vertred,.015,.27)
    bulge = .38*bulgeshape+.32*stellarpop+.17*concentration+.13*min(stellarpop,bulgeshape)
    # A prominent disk with its stellar component REDDER than the core
    # is not a red bulge. This can occur in blue, midplane-dominated disks.
    bulge += .12*ramp(redin-redout,.065,.24)-.10*ramp(discsize,10.,23.)*(1-concentration)
    thin += .22*bulge*ramp(qb,.52,.27)+.12*ramp(q-qb,.03,.13)
    # Razor-like extensions of a short disk are also visible in its faint
    # isophote. Do not require its bright core alone to be extremely flat.
    thin += .14*bulgeshape*ramp(qb,.53,.26)*ramp(redcore,.15,.72)
    sha,soff,faintq = segshape(sm,broad,mask,cx,cy,peak,scale,qo,pa)
    # Inspect the silhouette in the apparent disk plane. A thin disk sticks
    # out from the ellipse predominantly at the TWO major-axis ends, while
    # a featureless cigar-shaped ellipsoid should give an almost flat ring.
    uv = rad[:,None]*u
    vv = rad[:,None]*v
    ray = sample(sm,cx+uv,cy+vv)
    isocut = np.maximum(2.3*sig,peak*np.array([.08,.12,.18,.26]))
    under = ray[None,:,:]<isocut[:,None,None]
    under[:,-1,:]=True
    riso = rad[np.argmax(under,axis=1)]
    ecc4 = np.median(np.mean(riso*np.cos(4*ang),axis=1)/(np.mean(riso,axis=1)+.1))*2
    disky = ramp(ecc4,.018,.096)
    shortdisc = disky*ramp(qb,.51,.27)*ramp(stellarpop,.18,.70)
    thin += .22*shortdisc+.09*ramp(q-faintq,.035,.17)*ramp(qb,.55,.30)
    # Alternative bulge test: thick old component inside a narrow outer
    # disk, either through differential flattening or a disky silhouette.
    bulge += .12*disky*ramp(stellarpop,.18,.78)+.08*ramp(q-faintq,.04,.17)*stellarpop
    # Resolve interaction classes before deciding on a Hubble type.
    tails = .64*ramp(sha,.15,.42)+.36*ramp(soff,.07,.35)
    if pair>.32 and (q>.34 or pair>.5) and (disturb>.15 or pair>.52 or halo>.20 or tails>.45):
        return 1
    if pair>.16 and tails>.62 and pair+.4*disturb>.38 and q>.28:
        return 1
    disrupted = (.34*crooked+.23*oddmajor+.26*tidal+.17*ramp(skew,.04,.19))
    # A diffuse one-sided tail need not distort the bright core at all.
    if tails>.71 and (halo>.185 or skew>.10 or dip>.115) and (q>.40 or (tails>.9 and q>.27)) and (soff>.20 or oddmajor>.40):
        return 0
    if (disturb>.38 or (tidal>.65 and halo>.25 and dip>.10) or disrupted>.58) and (q>.37 or (disturb>.56 and q>.23)) and (disturb>.57 or dip>.71*even or halo>.31):
        return 0
    if thin>.52 and (qb<.42 or thin>.64):
        return 9 if bulge>.50 else 8
    # A small tidal extension need not have much flux. Give such an object
    # a separate route when the bright core is compact and the faint light
    # loses point symmetry; an ordinary spiral retains even arm structure.
    if (kin>.070 or skew>.063) and (crooked+.48*tails+.4*tidal>.89) and dip>.74*quad and (q>.40 or faintq>.45) and discsize<16.:
        return 0
    if spiral>.365 and q>.28:
        # As an independent pitch measure, resolve angular/radial gradient
        # components in the rectified disk. Repeated aligned edges are arm
        # ridges; the local angle of a ridge is the pitch even when the arm
        # is too short to trace a large change in Fourier phase.
        pill = (p-flux[:,None])/norm[:,None]
        dg = (np.roll(pill,-2,axis=1)-np.roll(pill,2,axis=1))/(4*2*np.pi/96)
        dr = np.gradient(pill,step,axis=0)
        w = wt[:,None]*((fracrad>.40)&(fracrad<1.45))[:,None]
        # Chirality coherence: logarithmic-spiral gradients have dr/dtheta
        # of the same sign on opposite sides, unlike ellipticity residuals.
        tanpitch = np.sqrt(np.sum(w*dg**2)/(np.sum(w*dr**2)+1.e-7))
        sgn = abs(np.sum(w*dg*dr))/(np.sqrt(np.sum(w*dr**2)*np.sum(w*dg**2))+1.e-7)
        # Coherent bars are intrinsically elongated in the luminous core;
        # an abrupt distortion confined to faint arms is not a bar.
        if barsignal>.112 and barbright>.080 and innerq<.81 and (redbar>.13 or barbright>.16) and (bar>.15 or bar>1.25*winding) and (q>.4 or twist>.08):
            return 5
        openness = (loose-tight)/(winding+.020)
        # Grand-design open arms make prominent low-order contrast even
        # under deprojection errors. Flocculent disks have higher harmonics.
        armcontrast = band(h[:,2]+.5*h[:,1],.55,1.45)
        outerphase = np.unwrap(np.angle(harm[:,2]))
        dphase = abs(np.gradient(outerphase,step))*.5
        paweight = wt*h[:,2]**2*((fracrad>.40)&(fracrad<1.45))
        phasepitch = np.sum(paweight*clip(dphase,.0,7.))/(np.sum(paweight)+1.e-6)
        ordered = ramp(armcontrast,.07,.23)
        # Opening an arm leaves low-surface-brightness space on either
        # side, in contrast to a tight arm on an almost continuous disk.
        armgap = band((np.quantile(p,.82,axis=1)-np.quantile(p,.18,axis=1))/norm,.55,1.35)
        loosevote = .20*ramp(openness,-.35,.28)+.20*ramp(tanpitch,.24,.62)+.18*ramp(phasepitch,4.0,1.6)*ordered+.20*ordered+.09*ramp(sgn,.16,.5)*ramp(tanpitch,.26,.52)+.13*ramp(armgap,.25,.68)
        return 7 if loosevote>.48 else 6
    if q>.79:
        return 2
    # When the minor axis is small, seeing and pixel integration bias the
    # axial ratio upward. Use both the bright and intermediate isophotes.
    smoothq = .65*q+.35*np.median(ratio[1:4])
    if smoothq<.50:
        return 4
    return 3

def predict(x):
    return np.array([classify(row) for row in x],dtype=np.int64)
'''

target = Path(sys.argv[1])
target.mkdir(parents=True, exist_ok=True)
(target / 'solution.py').write_text(SOURCE, encoding='utf-8')
