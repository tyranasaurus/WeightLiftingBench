"""Write the analytical classifier. No inputs other than the destination path."""
from pathlib import Path
import sys

solution = '''import numpy as np
import math

# Geometry in native pixel units. The even-sized thumbnail has its centre
# between pixels. Angles in the disk plane increase towards increasing y.
side = 128
ygrid, xgrid = np.indices((side, side), dtype=np.float64)
outside = (xgrid-63.5)**2 + (ygrid-63.5)**2 > 55**2
angle = np.arange(96) * (2*np.pi/96)
ca = np.cos(angle)
sa = np.sin(angle)
fx = np.fft.rfftfreq(side)
fy = np.fft.fftfreq(side)
freq = -2*np.pi**2 * (fy[:, None]**2 + fx**2)
scales = (0.85, 1.6, 3.2, 6.4)
filters = [np.exp(freq*s**2) for s in scales]
unitlog = np.linspace(0, 1, 32)
# Frequencies searched in log radius, parameterised by physical arm pitch.
pitches = np.array([-50,-40,-32,-26,-21,-17,-14,-11,-9,9,11,14,17,21,26,32,40,50])
slopes = 1/np.tan(pitches*np.pi/180)

def ramp(v, lo, hi):
    t = np.clip((v-lo)/(hi-lo), 0, 1)
    return t*t*(3-2*t)

def unit(v):
    return np.clip(v, 0, 1)

def blur(ft, i):
    return np.fft.irfft2(ft*filters[i], s=(side,side), axes=(0,1))

def noise(a):
    b = a[outside]
    return max(0.0008, (np.percentile(b,70)-np.percentile(b,30))/1.0488)

def sample(a,x,y):
    x = np.clip(x,0,126.99999)
    y = np.clip(y,0,126.99999)
    u = x.astype(np.int32)
    v = y.astype(np.int32)
    f = x-u
    g = y-v
    return ((1-g)*((1-f)*a[v,u]+f*a[v,u+1]) + g*((1-f)*a[v+1,u]+f*a[v+1,u+1]))

def ringgrid(cx,cy,theta,q,rs):
    p = rs[:,None] * ca
    t = (q*rs[:,None]) * sa
    c = np.cos(theta)
    s = np.sin(theta)
    return cx + c*p-s*t, cy + s*p+c*t

def moment(w,dx,dy,psf):
    total = np.sum(w)+1.e-12
    xx = np.sum(w*dx**2)/total
    yy = np.sum(w*dy**2)/total
    xy = np.sum(w*dx*dy)/total
    d = np.hypot(xx-yy,2*xy)
    big = max(0.5*(xx+yy+d)-psf**2, 1.)
    sml = max(0.5*(xx+yy-d)-psf**2, 0.6)
    return np.sqrt(big), np.clip(np.sqrt(sml/big), 0.07,1), 0.5*np.arctan2(2*xy,xx-yy)

def comp(mask, sy, sx):
    # Link all foreground runs that overlap in consecutive scanlines. Unlike
    # a circular crop this retains faint bridges but not detached field stars.
    padded = np.pad(mask.astype(np.int8), ((0,0),(1,1)))
    steps = np.diff(padded, axis=1)
    rows, starts = np.where(steps==1)
    ends = np.where(steps==-1)[1]-1
    nr = len(rows)
    tree = list(range(nr))
    def root(j):
        while tree[j]!=j:
            tree[j] = tree[tree[j]]
            j = tree[j]
        return j
    index = np.searchsorted(rows,np.arange(129))
    for y in range(1,128):
        k = index[y-1]
        for j in range(index[y],index[y+1]):
            while k < index[y] and ends[k]+1 < starts[j]:
                k += 1
            l = k
            while l < index[y] and starts[l]-1 <= ends[j]:
                tree[root(l)] = root(j)
                l += 1
    target = np.where((rows==sy)&(starts<=sx)&(ends>=sx))[0]
    res = np.zeros((128,128),dtype=bool)
    if len(target):
        key = root(target[0])
        for j in range(nr):
            if root(j)==key:
                res[rows[j],starts[j]:ends[j]+1] = True
    return res

def geom(co,im,ns,cx,cy,peak):
    dx,dy = xgrid-cx,ygrid-cy
    rr = dx**2+dy**2
    shapes=[]
    for frac in (0.065,0.14,0.30,0.55,0.75):
        cut = max(frac*peak,3*ns)
        m = co & (im>cut)
        r = np.clip(np.sqrt(m.sum()/np.pi),2.5,48)
        # First isolate the core's vicinity. Then use an elliptical window
        # of the isophote's own size to suppress filaments and interlopers.
        w = m * np.exp(-0.5*rr/(2.6*r)**2) * (0.75+0.25*unit(im/(cut+0.001)-1))
        a,q,t = moment(w,dx,dy,0.95)
        u,v = dx*np.cos(t)+dy*np.sin(t), dy*np.cos(t)-dx*np.sin(t)
        lim = (u/(2.8*a+2))**2+(v/(2.8*a*q+2))**2
        w = m * np.exp(-0.5*np.minimum(lim**2,80))
        a,q,t = moment(w,dx,dy,0.95)
        shapes.append((max(2,2*a),q,t))
    return shapes

def peakdata(im,fine,co,cx,cy,peak,rad,ns):
    local = im.copy()
    for i in (-1,0,1):
        for j in (-1,0,1):
            local = np.maximum(local,np.roll(np.roll(im,i,axis=0),j,axis=1))
    dist = np.hypot(xgrid-cx,ygrid-cy)
    ok = (im>=local) & (im>max(0.10*peak,5*ns)) & (dist>3.5) & (dist<57)
    ys,xs = np.where(ok)
    if not len(xs):
        return 0.
    best = 0.
    order = np.argsort(im[ys,xs])[::-1][:48]
    for h in order:
        y,x = ys[h],xs[h]
        d = dist[y,x]
        t = np.linspace(0,1,25)
        vals = sample(im,cx+(x-cx)*t,cy+(y-cy)*t)
        val = im[y,x]
        saddle = np.min(vals)
        prom = (val-saddle)/(val+0.001)
        flux = val/(peak+0.001)
        # Width of the secondary nucleus above its own bridge / pedestal.
        hh,ww = np.indices((15,15)) - 7
        patch = im[np.clip(y+hh,0,127),np.clip(x+ww,0,127)]
        broad = np.sqrt(np.sum((patch>saddle+(val-saddle)*0.48)&(hh**2+ww**2<=49))/np.pi)
        resolution = ramp(broad,1.9,4.0)
        sharp = val/(fine[y,x]+0.001)
        resolution = max(resolution,0.85*ramp(sharp,0.72,0.90))
        link = ramp(saddle/(min(peak,val)+0.001),0.012,0.16)
        prox = np.exp(-0.5*(d/(rad*1.18+3.5))**2)
        affinity = max(0.92*link*(0.72+0.28*float(co[y,x])),0.85*prox)*max(prox,link*0.75)
        vote = ramp(flux,0.15,0.65)*ramp(prom,0.13,0.55)*(0.20+0.80*resolution)*affinity
        best = max(best,vote)
    return best

def weighted(v,w):
    return np.sum(v*w)/(np.sum(w)+1.e-12)

def spectrum(norm, rw, rs, mode):
    # Matched filters for travelling angular waves. Search both handednesses.
    # An inclination error gives a standing m=2 wave, not a travelling one.
    c = np.fft.rfft(norm,axis=1)[:,mode] * (2/norm.shape[1])
    amp = np.abs(c)
    logs = np.log(rs)
    phase = np.exp(1j*(mode*slopes[:,None])*logs)
    best, signed, winding, pfound = 0., 0., 0., 22.
    for bottom,top in ((0,32),(5,32),(10,32),(2,25),(12,30)):
        sl = slice(bottom,top)
        w = rw[sl]
        z = c[sl]
        total = np.sum(w)+1.e-9
        if total<1.e-5:
            continue
        mean = np.sum(z*w)/total
        # Remove most of the standing wave; do not demand a full revolution.
        g = z-0.82*mean
        dot = np.abs(np.sum(phase[:,sl]*(w*g)[None,:],axis=1))/total
        # A changing radial ellipticity makes equal left- and right-handed
        # correlations. Only their imbalance is evidence of winding.
        opposite = dot[::-1]
        chirality = unit((dot-opposite)/(dot+opposite+0.015))
        coherent = dot/(weighted(np.abs(g),w)+0.017)
        pts = dot*(0.50+0.50*np.minimum(coherent,1.))*(0.57+0.43*np.sqrt(chirality))
        idx = np.argmax(pts)
        power = pts[idx]*min(1., (total/(top-bottom)+0.01)/0.26)
        if power>best:
            best = power
            pfound = abs(pitches[idx])
            signed = np.sign(pitches[idx])
            winding = coherent[idx]
    return best,signed,winding,pfound

def morphology(img):
    sky = np.median(img[:,outside],axis=1)
    colors = img.astype(np.float64)-sky[:,None,None]
    lum = 0.31*colors[0]+0.43*colors[1]+0.26*colors[2]
    ft = np.fft.rfft2(lum)
    sm = [blur(ft,i) for i in range(4)]
    for a in sm:
        a -= np.median(a[outside])
    fine,med,coarse,wide = sm
    nf,nm,nc,nw = [noise(a) for a in sm]
    mid = (xgrid-63.5)**2+(ygrid-63.5)**2
    prior = np.exp(-0.5*mid/5.0**2)
    centerstat = np.maximum(med,0)*prior + 0.24*np.maximum(coarse,0)*prior
    yi,xi = np.unravel_index(np.argmax(centerstat),centerstat.shape)
    if centerstat[yi,xi] < 0.0001:
        return 2
    cx,cy = float(xi),float(yi)
    for scale in (2.8,2.0,1.65):
        d2 = (xgrid-cx)**2+(ygrid-cy)**2
        w = np.exp(-d2/(2*scale**2))*np.maximum(fine-0.18*max(0.,fine[yi,xi]),0)**2
        if w.sum()>1.e-9:
            cx,cy = np.sum(xgrid*w)/np.sum(w),np.sum(ygrid*w)/np.sum(w)
    y0,x0 = int(round(cy)),int(round(cx))
    peak = max(float(sample(fine,cx,cy)),2*nf,0.009)
    seg = ((med>max(2.7*nm,peak*0.030)) | (coarse>max(3.0*nc,peak*0.031)))
    seg[y0,x0] = True
    co = comp(seg,y0,x0)
    if np.sum(co)<8:
        return 2
    shapes = geom(co,med,nm,cx,cy,max(med[y0,x0],0.012))
    # Radius and flattening of the disk: favour the two outer isophotes.
    ao,qo,to = shapes[0]
    am,qm,tm = shapes[1]
    ai,qi,ti = shapes[2]
    ah,qh,th = shapes[3]
    vec = (1-qo)*np.exp(2j*to)+0.72*(1-qm)*np.exp(2j*tm)+0.18*(1-qi)*np.exp(2j*ti)
    theta = 0.5*np.angle(vec)
    q = 0.53*qo+0.34*qm+0.13*qi
    qproj = np.clip(q,0.13,1.)
    dx,dy=xgrid-cx,ygrid-cy
    u=dx*np.cos(theta)+dy*np.sin(theta)
    v=dy*np.cos(theta)-dx*np.sin(theta)
    er=np.sqrt(u**2+(v/qproj)**2)
    # Outer size from connected light, downweighting remote tips of tidal tails.
    radius = np.clip(max(ao,1.12*am,1.40*ai),5.,64.)
    sel = (er<max(2.0*radius,15)) & co
    light = np.maximum(fine-max(0.8*nf,0.006*peak),0)*sel
    ir = np.minimum(np.floor(er*2).astype(np.int32),511)
    hist = np.bincount(ir.ravel(),weights=light.ravel(),minlength=512).cumsum()
    r20,r50,r80 = 0.5*np.searchsorted(hist,hist[-1]*np.array([0.2,0.5,0.8]))+0.25
    radius = np.clip(max(radius,1.15*r80),5.,64.)
    # A separate size in the sky plane for merger proximity.
    merge = peakdata(med,fine,co,cx,cy,max(med[y0,x0],0.012),np.sqrt(qproj)*radius,nm)
    # Mask other sources for colour and asymmetry. A faint isolated field star
    # should not make the central ellipse look irregular.
    footprint = co & (er<min(64.,1.4*radius+2))
    signal = np.maximum(fine,0)
    cweight = footprint * unit((signal-1.5*nf)/(0.09*peak+nf))
    fc = np.maximum(np.sum(colors*cweight,axis=(1,2)),0)
    color = np.clip((fc[2]-fc[0])/(fc[2]+fc[0]+1.e-6),-1,1)
    outer = cweight * (er>max(2.,0.70*r50))
    oc = np.maximum(np.sum(colors*outer,axis=(1,2)),0)
    ocolor = np.clip((oc[2]-oc[0])/(oc[2]+oc[0]+1.e-6),-1,1)
    blevel = ramp(0.4*color+0.6*ocolor,-0.19,0.13)
    def asym(a,ns):
        clean = np.maximum(a,0)*co
        rev = sample(clean,2*cx-xgrid,2*cy-ygrid)
        aperture = (er<1.25*radius+1) * (er>1.5)
        both = (clean+rev)*aperture
        dev = np.maximum(abs(clean-rev)-1.45*ns,0)*aperture
        return np.sum(dev)/(np.sum(both)+0.001)
    asymm = 0.55*asym(med,nm)+0.45*asym(fine,nf)
    # Shape asymmetry of the faint outskirts can reveal a stream or shell
    # even when the bright, centrally concentrated light is symmetric.
    envelope = comp(((coarse>max(3.0*nc,0.039*peak)) | (wide>max(4*nw,0.039*peak))) | ((xgrid==x0)&(ygrid==y0)),y0,x0)
    reverse = sample(envelope.astype(np.float64),2*cx-xgrid,2*cy-ygrid)
    envwin = np.hypot(dx,dy)<min(62.,1.62*radius+5)
    shapeasym = np.sum(abs(envelope.astype(np.float64)-reverse)*envwin)/(np.sum((envelope+reverse)*envwin)+1.)
    # Use a denoised, blue-sensitive projection for the arm map. Keep red
    # light in the mix to see old stellar bars and arms with little contrast.
    blueft = np.fft.rfft2(0.12*colors[0]+0.41*colors[1]+0.47*colors[2])
    bmap = blur(blueft,0)
    bmap -= np.median(bmap[outside])
    nb = noise(bmap)
    rstart = max(1.65,0.105*radius)
    rend = min(65.,max(7.,1.12*radius))
    rs = rstart * (rend/rstart)**unitlog
    def polar(dq,dt):
        xx,yy = ringgrid(cx,cy,dt,dq,rs)
        mask = sample(co.astype(np.float64),xx,yy)
        pv = sample(bmap,xx,yy)
        # Winsorise isolated peaks on a ring, but not broad arms.
        prof = np.median(pv,axis=1)
        high = np.percentile(pv,89,axis=1)
        high = np.maximum(high,prof)+np.maximum(0.05*peak,0.65*(high-prof))
        pv = np.minimum(pv,high[:,None])
        pv = np.where(mask>0.35,pv,np.minimum(pv,prof[:,None]))
        baseline = np.mean(pv,axis=1)
        den = np.maximum(baseline,0)+2.8*nb+0.010*peak
        norm = (pv-baseline[:,None])/den[:,None]
        norm = np.clip(norm,-1.5,2.3)
        rw = ramp(baseline/nb,0.95,4.2)*ramp(rs*dq,1.1,3.0)
        return norm,rw,baseline,den
    norm,rw,profile,den = polar(qproj,theta)
    def harmonics(m):
        return 2*np.fft.rfft(m,axis=1)/m.shape[1]
    h = harmonics(norm)
    # Odd and high-order modes quantify resolved non-elliptical structure.
    discw = rw*ramp(rs/radius,0.16,0.35)
    angular = np.sqrt(np.sum(abs(h[:,3:9])**2,axis=1))
    expected = nb/den*np.sqrt(np.minimum(1.,12/(np.maximum(rs,1)*np.sqrt(qproj))))
    texture = weighted(np.maximum(angular-1.05*expected,0),discw)
    lop = weighted(np.maximum(abs(h[:,1])-expected*0.6,0),discw)
    sp2,sgn,conf,pit2 = spectrum(norm,rw,rs,2)
    sp3,sgn3,conf3,pit3 = spectrum(norm,rw,rs,3)
    # Bypass modest errors in the projection for non-axisymmetric disks.
    for dq in (max(0.30,0.82*qproj),min(1.,qproj*1.2)):
        if abs(dq-qproj)<0.035 or qproj<0.27:
            continue
        n,w,base,dd = polar(dq,theta)
        s,g,k,p = spectrum(n,w,rs,2)
        if s*0.94>sp2:
            sp2,sgn,conf,pit2=s*0.94,g,k,p
        s,g,k,p = spectrum(n,w,rs,3)
        if s*0.94>sp3:
            sp3,sgn3,conf3,pit3=s*0.94,g,k,p
    spiral = max(sp2,0.90*sp3)
    pitch = pit2 if sp2>=0.90*sp3 else pit3
    # Flocculent spirals need not be a single log spiral. Measure the local
    # rotation of each angular harmonic between neighbouring annuli. Adjacent
    # arcs of the same handedness should accumulate even with changing pitch.
    modes = np.arange(2,7)[None,:]
    zz = h[:,2:7].copy()
    zz[:,0] -= 0.6*weighted(zz[:,0],discw)
    rot = zz[2:]*np.conj(zz[:-2])
    speed = np.angle(rot) / ((np.log(rs[2:]/rs[:-2]))[:,None]*modes)
    polarw = np.sqrt(discw[2:]*discw[:-2])[:,None]
    amp = np.sqrt(np.abs(rot))
    arcnoise = 0.8*nb/den[1:-1,None]*np.sqrt(np.minimum(1.,2.8/(rs[1:-1,None]*np.sqrt(qproj))))
    power = np.maximum(amp**2-arcnoise**2,0)*polarw
    good = ramp(abs(speed),0.35,1.3)*(1-0.75*ramp(abs(speed),7,12))
    cw = power*good
    pos = np.sum(cw*(speed>0))
    neg = np.sum(cw*(speed<0))
    arc = np.sqrt(abs(pos-neg)/(np.sum(polarw)+0.01))*abs(pos-neg)/(pos+neg+0.01)
    correct = speed*np.sign(pos-neg)>0
    arcweight = cw*correct
    localpitch = weighted(180/np.pi*np.arctan(1/np.maximum(0.03,abs(speed))),arcweight)
    pitchmix = 0.45*ramp(arc,0.08,0.20)
    pitch = pitch*(1-pitchmix)+localpitch*pitchmix
    spiral = max(spiral,0.88*arc)
    # Persistent quadrupole inside the disk, at a fixed position angle. A bar
    # needs a change between its own quadrupole and that of the outer disk.
    z=h[:,2]
    outerw=rw*ramp(rs/radius,0.57,0.77)
    outerz=weighted(z,outerw)
    bar=0.
    for low,high in ((0.11,0.34),(0.16,0.48),(0.23,0.60),(0.32,0.68)):
        wb = rw*((rs/radius>low)&(rs/radius<high))
        zbar = weighted(z,wb)
        strength = abs(zbar-0.45*outerz)
        phasecoh = abs(zbar)/(weighted(abs(z),wb)+0.014)
        bar=max(bar, strength*min(1.,phasecoh)**2*min(1.,wb.sum()/3.))
    # Isophote twist/elongation corroborates an inner bar without requiring
    # the disk to be face on. The complex form wraps position angles at pi.
    def ellip(qi,ti):
        return (1-qi)/(1+qi)*np.exp(2j*ti)
    eout = 0.55*ellip(qo,to)+0.45*ellip(qm,tm)
    eb = 0.45*ellip(qi,ti)+0.55*ellip(qh,th)
    bex = abs(eb-eout)*ramp(abs(eb),0.08,0.30)
    # Distinguish a real inner bar from a nearly spherical bulge inside a
    # flattened disk: the latter also looks like a bar *after* deprojection.
    elongated = max(0.,abs(eb)-abs(eout))
    crossed = abs(eb)*abs(np.sin(0.5*np.angle(eb*np.conj(eout))))
    barshape = ramp(max(elongated,crossed),0.018,0.14)
    barflag = ramp(bar+0.35*bex,0.14,0.38) * (0.40+0.60*max(barshape,ramp(bex,0.035,0.17)*barshape**0.4))
    # Photometric profile compared with a single exponential. Bulges raise
    # the centre relative to the outer extrapolation; a dust lane lowers it.
    cr = np.array([0.0,0.22,0.40,0.67,0.87])*radius
    xp,yp = ringgrid(cx,cy,theta,qproj,cr)
    cp = np.maximum(np.median(sample(med,xp,yp),axis=1),0.6*nm)+0.003*peak
    excess = np.log(cp[0]/cp[2])-0.40/0.47*np.log(cp[2]/cp[4])
    con = r80/max(r20,1.)
    brightbulge = 0.45*ramp(excess,-0.02,0.88)+0.55*ramp(con,3.1,5.0)
    # Minor-axis thickness at the nucleus, and away from it. Measure a
    # local half-maximum scaleheight and an absolute isophotal scaleheight.
    ts = np.linspace(-0.52,0.52,17)*radius
    vs = np.linspace(-1,1,81)*max(7.,min(36.,radius*max(0.42,qproj)))
    c,s = np.cos(theta),np.sin(theta)
    gx = cx+ts[:,None]*c-vs*s
    gy = cy+ts[:,None]*s+vs*c
    cuts = sample(fine,gx,gy) * sample(co.astype(np.float64),gx,gy)
    maxima = np.max(cuts,axis=1)
    half = np.sum(cuts>0.5*maxima[:,None],axis=1)*(vs[1]-vs[0])
    wmid = np.median(half[7:10])
    wings = 0.5*(half[1:5]+half[-5:-1][::-1])
    wwing = np.median(wings)
    swell = wmid/(wwing+0.35)
    contour = np.sum(cuts>max(0.17*peak,3*nf),axis=1)*(vs[1]-vs[0])
    # Extrapolate disk thickness along its major axis. A transparent slab
    # with a nearly exponential vertical profile widens roughly linearly.
    iso = 0.5*(contour[9:]+contour[7::-1])
    centralwidth = np.median(contour[7:10])
    extent = np.median(iso[5:8])
    extrap = np.median(iso[2:5]) + (np.median(iso[2:5])-extent)*(4./3.)
    bump = centralwidth/(max(extrap,0.55*centralwidth)+0.35)
    bulgecontour = ramp(bump,1.03,1.47)*ramp(extent,0.6,2.5)
    centralpeak = np.median(maxima[7:10])/(np.sqrt(max(0.001,np.median(maxima[2:5]))*max(0.001,np.median(maxima[-5:-2])))+0.001)
    # A symmetric red midplane relative to adjacent strips is a dust lane.
    lu = np.linspace(-0.46,0.46,25)*radius
    ht = max(1.3,0.28*qproj*radius)
    def stripe(k):
        xx = cx+lu*c-k*ht*s
        yy = cy+lu*s+k*ht*c
        br = sample(colors[2],xx,yy)
        rd = sample(colors[0],xx,yy)
        return np.median((rd-br)/(np.maximum(rd+br,0)+0.06))
    lane = stripe(0)-0.5*(stripe(-1)+stripe(1))
    # Independent lopsidedness and substructure tests. Use soft gates so a
    # bright companion or dusty edge-on galaxy is not classified by one cue.
    irr = ramp(asymm,0.12,0.36)
    clumps = ramp(texture,0.075,0.255)
    arm = ramp(spiral,0.053,0.20)
    vis = ramp(radius,9.,24.)
    flat = 0.55*qo+0.32*qm+0.13*qi
    thin = 1-ramp(flat,0.17,0.36)
    diskiness = (0.42*blevel + 0.20*clumps + 0.15*vis + 0.15*ramp(lane,0.025,0.17) + 0.08*(1-brightbulge))
    # Smooth spheroids generally have similar axis ratios on all isophotes;
    # an edge-on disk can be much flatter outside its central swelling.
    diskiness = max(diskiness,0.46*ramp(qh-qo,0.02,0.23)+0.22*vis+0.17*ramp(swell,1.23,1.85)+0.15*ramp(lane,0.02,0.15))
    edge = (1-ramp(flat,0.24,0.46))*(0.37+0.63*diskiness)+0.22*thin
    # Strong arms of an inclined spiral penalise the edge interpretation.
    edge -= 0.19*arm*ramp(flat,0.19,0.40)
    bulge = (0.20*ramp(swell,1.12,1.76)+0.19*ramp(qh-qo,-0.01,0.22)+0.16*brightbulge+0.13*ramp(centralpeak,1.6,3.8)+0.10*(1-blevel)+0.06*ramp(lane,0.03,0.14)+0.16*bulgecontour)
    # Spirals often have extended blue disks and a redder core. Use these
    # hints as supporting evidence, not a replacement for actual arm power.
    bluegrad = ramp(ocolor-color,0.012,0.095)
    supported = (0.42*arm+0.17*clumps+0.19*blevel+0.08*bluegrad+0.08*vis+0.06*irr)
    supported = max(supported,0.63*barflag+0.12*vis+0.13*blevel+0.12*clumps)
    # A resolved rotating mode with little power in higher modes is itself
    # evidence of a grand-design spiral, even in a red stellar population.
    supported = max(supported,0.77*arm+0.10*vis+0.08*blevel+0.05*clumps)
    # Below the angular resolution for a winding measurement, a blue disk
    # surrounding a red nucleus is useful (although weaker) spiral evidence.
    chromadisk = blevel*(0.25+0.75*bluegrad)
    supported = max(supported,0.48*chromadisk+0.14*vis+0.14*clumps+0.12*arm)
    disrupted = 0.51*irr+0.19*ramp(lop,0.09,0.39)+0.13*clumps+0.17*ramp(shapeasym,0.14,0.42)
    disrupted = max(disrupted,0.69*ramp(shapeasym,0.24,0.61)+0.31*irr)
    disrupted -= 0.20*arm + 0.14*thin
    if merge>0.38 and merge+0.35*irr>0.54:
        return 1
    if disrupted>0.54 and disrupted>0.82*edge and disrupted>0.86*supported-0.06:
        return 0
    if edge>0.49 and edge>supported*(0.67+0.34*ramp(flat,0.22,0.4)):
        return 9 if bulge>0.48 else 8
    if supported>0.46 and flat>0.18:
        if (barflag>0.42 and bar>0.90*spiral) or (barflag>0.70 and bar>0.66*spiral):
            return 5
        # Loose arms tend to be more contrasted, later type, and associated
        # with smaller bulges. Use pitch where the travelling wave is clear.
        loose = (0.24*ramp(sp2+0.6*sp3,0.10,0.32)+0.22*clumps+0.21*ramp(pitch,16,31)+0.13*blevel+0.11*(1-brightbulge)+0.09*irr)
        return 7 if loose>0.55 else 6
    # Unstructured and blue with an off-centre light distribution: common
    # appearance of an irregular not showing a well-defined spiral mode.
    if irr>0.46 and blevel>0.55 and lop>0.14:
        return 0
    # Volunteers judge roundness from the visible body, not the innermost
    # pixel (often round after convolution with the atmospheric PSF).
    smoothq = 0.26*qo+0.39*qm+0.27*qi+0.08*qh
    if smoothq>0.816:
        return 2
    if smoothq<0.425:
        return 4
    return 3

def predict(x):
    return np.array([morphology(row) for row in x],dtype=np.int64)
'''

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / 'solution.py').write_text(solution, encoding='utf-8')
