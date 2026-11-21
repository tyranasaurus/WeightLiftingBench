import numpy as np
import math

# Coordinates and a circular isophote sampler, independent of the input.
Y, X = np.indices((128,128), dtype=np.float64)
angle = np.arange(96) * (2*math.pi/96)
ca, sa = np.cos(angle), np.sin(angle)
radii = np.exp(np.linspace(0, math.log(84), 80))[:,None]
eps = 1.e-8


def limit(x):
    return np.clip(x, 0., 1.)


def ramp(x, low, high):
    return limit((x-low)/(high-low))


def blur(a, times=1):
    for it in range(times):
        for ax in (0,1):
            a = (6*a + 4*(np.roll(a,1,ax)+np.roll(a,-1,ax)) + np.roll(a,2,ax)+np.roll(a,-2,ax))*(1./16)
    return a


def sample(a,x,y):
    x = np.clip(x,0,126.99999)
    y = np.clip(y,0,126.99999)
    ix = x.astype(np.int32)
    iy = y.astype(np.int32)
    fx = x-ix
    fy = y-iy
    return (1-fy)*((1-fx)*a[iy,ix]+fx*a[iy,ix+1]) + fy*((1-fx)*a[iy+1,ix]+fx*a[iy+1,ix+1])


def localpeak(a):
    hood = np.maximum(np.roll(a,1,0),np.roll(a,-1,0))
    hood = np.maximum(a,hood)
    hood = np.maximum(hood,np.maximum(np.roll(hood,1,1),np.roll(hood,-1,1)))
    return a>=hood


def radius_at(profile, level):
    # Start inside the central light distribution. Require a sustained crossing.
    ids = np.nonzero((profile[:-2]<level)&(profile[1:-1]<level)&(profile[2:]<level))[0]
    if not len(ids):
        return 82.
    i = ids[0]
    if i == 0:
        return 1.
    f = limit((profile[i-1]-level)/(profile[i-1]-profile[i]+eps))
    return radii[i-1,0]*(1-f)+radii[i,0]*f


def detect(v, rgb, cx, cy, noise):
    # Local maxima are accepted as possible nuclei only if they have a saddle,
    # a resolved envelope and substantial flux compared with the main nucleus.
    peaks = localpeak(v) & (((X-cx)**2+(Y-cy)**2)<50**2)
    primary = max(.015,float(sample(v,cx,cy)))
    iy,ix = np.nonzero(peaks & (v>max(noise*4,.10*primary,.023)))
    order = np.argsort(-v[iy,ix], kind='stable')[:45]
    companion = 0.
    objects = []
    col0 = np.maximum(np.array([sample(chan,cx,cy) for chan in rgb]),0)
    col0 = col0/(np.sum(col0)+eps)
    for k in order:
        y,x = iy[k],ix[k]
        dx,dy = x-cx,y-cy
        d = math.hypot(dx,dy)
        if d < 2.6:
            continue
        h = v[y,x]
        r = np.array([2.,3.2,4.8,7.2])[:,None]
        tx,ty = ca[::4],sa[::4]
        rings = sample(v,x+r*tx,y+r*ty)
        # A local plane of diffuse host light must not inflate source width.
        line = np.linspace(.08,.92,20)
        valley = max(0.,sample(v,cx+line*dx,cy+line*dy).min())
        prom = limit(1-valley/(min(primary,h)+eps))
        surround = np.maximum(np.quantile(rings,.30,axis=1),0)
        back = min(valley, surround[-1]) * .85
        breadth = (surround[1]-back)/(h-back+eps)
        col = np.maximum(rgb[:,y,x],0)
        col = col/(np.sum(col)+eps)
        colmatch = np.sum(np.abs(col-col0))
        satellite = (ramp(breadth,.19,.58)*.85+.15) * (1-.75*ramp(colmatch,.17,.5))
        ratio = min(2., h/primary)
        # For close nuclei, a pair of peaks separated by a shallow valley is enough.
        evidence = satellite * ramp(prom,.018,.32) * ramp(ratio,.14,.55)
        evidence *= 1-.7*ramp(d,18.,44.)
        if d<18 and breadth>.38:
            evidence = max(evidence, (.8+.2*ramp(breadth,.38,.60))*ramp(prom,.025,.24)*ramp(ratio,.28,.75)*(1-.65*ramp(colmatch,.18,.47)))
        companion = max(companion,evidence)
        if prom>.17 and d>4.:
            objects.append((x,y,h,valley,breadth,colmatch,d))
    return companion, objects


def pair_envelope(v, cx, cy, aa, peak, noise, objects):
    # Two nearby extended sources can be unrelated. Require them to share
    # diffuse light, or be close compared with their visible extents.
    pair=0.
    for x,y,h,saddle,width,cd,d in objects:
        if d<4. or h<.12*peak:
            continue
        tx=(x-cx)/d
        ty=(y-cy)/d
        # Transverse slices on the secondary: avoid the primary's nucleus.
        t=np.arange(1,17)[:,None]
        offsets=np.array([-.7,0.,.7])[None,:]
        side1=sample(v,x+tx*offsets-ty*t,y+ty*offsets+tx*t)
        side2=sample(v,x+tx*offsets+ty*t,y+ty*offsets-tx*t)
        prof=np.minimum(side1.mean(1),side2.mean(1))
        size=1.+np.sum(np.minimum.accumulate(prof) > max(.12*h,noise*2.8))
        gap=d/(max(3.,aa[3])+size)
        bridge=saddle/(min(h,peak)+eps)
        connection=max(1-ramp(gap,.83,1.42),ramp(bridge,.035,.18))
        stellarity=ramp(width,.28,.55)
        color=1-.86*ramp(cd,.12,.40)
        ratio=ramp(h/peak,.18,.65)
        contrast=ramp(1-bridge,.10,.45)
        # A quiescent pair usually has two centrally concentrated nuclei, arms
        # more often have diffuse, blue knots. Require real separation of the
        # cores to reject local maxima inside a single flattened nucleus.
        purity=(1-.7*ramp(bridge,.52,.85))*(1-.4*ramp(peak/h,2.0,3.6))
        # Two bright objects separated by pure sky are not evidence of a
        # merger. A compact double core can share an unresolved envelope.
        purity*=max(1-ramp(d,9.,17.),.28+.72*ramp(bridge,.022,.14))
        if d/(size+eps)<.75:
            purity*=.65
        pair=max(pair,(.27+.73*stellarity)*color*ratio*connection*contrast*purity)
    return pair


def flood(binary, x, y):
    # Run-length flood. Adjacent rows may overlap diagonally.
    mask = np.zeros_like(binary)
    if not binary[y,x]:
        return mask
    work = [(x,y)]
    while len(work):
        x,y = work.pop()
        if mask[y,x]:
            continue
        lo=hi=x
        row=binary[y]
        while lo>0 and row[lo-1]:
            lo -= 1
        while hi<127 and row[hi+1]:
            hi += 1
        mask[y,lo:hi+1] = True
        for v in (y-1,y+1):
            if 0<=v<128:
                a = max(0,lo-1)
                b = min(128,hi+2)
                z = binary[v,a:b] & ~mask[v,a:b]
                for pos in np.flatnonzero(z & ~np.concatenate((np.array([False]),z[:-1]))):
                    work.append((a+pos,v))
    return mask


def isoshape(v,valid,cx,cy,peak,noise):
    # Shape from nested uniform-density contours. Doubling the area moment
    # gives the semiaxis length; diagonalize a 2x2 moment tensor analytically.
    shapes=[]
    px=int(np.clip(round(cx),0,127))
    py=int(np.clip(round(cy),0,127))
    for fraction in (.62,.40,.23,.13,.072):
        level = max(peak*fraction,2.6*noise,.012)
        m = flood((v>=level)&valid,px,py)
        yy,xx = np.nonzero(m)
        n = len(xx)
        if n<4:
            shapes.append((1.,1.,0.,cx,cy,1.))
            continue
        mx,my = xx.mean(),yy.mean()
        dx,dy = xx-mx,yy-my
        vx,vy,xy = np.mean(dx*dx),np.mean(dy*dy),np.mean(dx*dy)
        diff = math.hypot(vx-vy,2*xy)
        major = math.sqrt(max(1.,2*(vx+vy+diff)+1./3))
        minor = math.sqrt(max(.49,2*(vx+vy-diff)+1./3))
        fill = n/(math.pi*major*minor)
        shapes.append((major,minor,.5*math.atan2(2*xy,vx-vy),mx,my,fill))
    return np.array(shapes).T


def annmean(vec,r,l,h):
    w = (r>=l)&(r<=h)
    if not np.any(w):
        return vec[np.argmin(np.abs(r-.5*(l+h)))]
    return np.mean(vec[w],axis=0)


def outer_pattern(polar, vis, rr, scale, peak, noise):
    # In the faint outskirts a median isophote can exclude the arms entirely.
    # Trace the angular maxima after combining adjacent rings instead. Track
    # their phase change, thickness and radial reach relative to the host disk.
    r=rr/scale
    sm=(.25*np.roll(polar,1,0)+.5*polar+.25*np.roll(polar,-1,0))
    sm=(np.roll(sm,2,1)+2*np.roll(sm,1,1)+3*sm+2*np.roll(sm,-1,1)+np.roll(sm,-2,1))/9
    median=np.median(sm,axis=1)[:,None]
    residual=sm-median
    opposite=np.roll(residual,48,1)
    # Arcs extending on both sides of the disk (rather than a single star).
    sym=(residual+opposite)*.5
    positive=np.quantile(sym,.90,axis=1)
    floor=2.5*noise+.009*peak
    support=(r>.60)&(r<2.15)&(positive>floor)&(np.mean(vis,axis=1)>.78)
    norm=np.maximum(median[:,0],0)+4*noise+.015*peak
    modes=np.fft.rfft(np.clip(residual/norm[:,None],-1.5,1.5),axis=1)*(2./96)
    z=modes[:,2]
    s=np.abs(z)
    w=support*ramp(s,.04,.24)
    lag=4
    prod=z[lag:]*np.conj(z[:-lag]); w2=w[lag:]*w[:-lag]*ramp(abs(prod),.004,.04)
    # Angular turning per log radius equals cot(pitch). Straight bars and
    # ellipses have no such handedness. Retain coherent rotations only.
    dph=np.angle(prod)/(2*np.log(rr[lag:]/rr[:-lag]))
    turn=abs(np.sum(w2*dph))/(np.sum(w2)+eps)
    coher=abs(np.sum(w2*dph))/(np.sum(w2*abs(dph))+eps)
    openpitch=ramp(turn,.35,1.0)*(1-ramp(turn,2.15,3.8))*coher
    extent=annmean(ramp(positive/floor,.65,2.3)*ramp(positive/norm,.16,.52),r,1.05,1.72)
    strong=annmean(ramp(s,.09,.40),r,.85,1.30)
    odd=np.sqrt(abs(modes[:,1])**2+abs(modes[:,3])**2)
    tidal=annmean(ramp(odd,.13,.55)*ramp(np.quantile(residual,.9,axis=1)/floor,.55,1.7),r,.85,1.6)
    return extent,strong,openpitch,tidal


def disksection(light,cx,cy,cp,sp,major,minor,peak,noise):
    # In a smooth ellipse the contours stay parallel after scaling both
    # axes. A disk + bulge has a pointed outer contour and an excess of
    # central light when the radial disk profile is extrapolated inwards.
    u=np.linspace(-1.24,1.24,81)
    h=np.linspace(-1.5,1.5,101)[:,None]
    plane=sample(light,cx+cp*(u*major)-sp*(h*minor),cy+sp*(u*major)+cp*(h*minor))
    cut=np.mean(plane[abs(h[:,0])<.16],axis=0)
    central=max(np.max(cut[np.abs(u)<.16]),eps)
    prox=np.mean(cut[(abs(u)>.26)&(abs(u)<.42)])
    distal=np.mean(cut[(abs(u)>.55)&(abs(u)<.81)])
    # Forward extrapolate an exponential that passes through both strips.
    excess=np.log((central+.012)/(max(.0,prox)+.012))-np.log((max(0.,prox)+.012)/(max(.0,distal)+.012))
    # Correct a midplane displacement or mild warp by using column maxima.
    height=np.mean(plane>max(.15*peak,noise*3),axis=0)
    radius=abs(u)
    shoulder=height[(radius>.45)&(radius<.7)]
    centralWidth=np.mean(height[radius<.13])+eps
    pointed=1-np.mean(shoulder)/centralWidth
    # If the disk is very thin, the local PSF hides its true axial ratio.
    # Measure the tails in a narrow aligned strip at high signal-to-noise.
    return excess,pointed


def disk_shape(aa,bb,qq,theta,phi,cx,cy,light,peak,noise):
    # Flat wings versus central roundness are visible directly in isophote
    # silhouettes. Measure vertical chords in fractions of the local major
    # axis, averaging opposite sides so a knot cannot mimic a bulge.
    cp,sp=math.cos(phi),math.sin(phi)
    diamonds=[]
    for k,lev in ((2,.23),(3,.13),(4,.072)):
        ang=theta[k]
        cs,sn=math.cos(ang),math.sin(ang)
        u=np.linspace(-1.35,1.35,101)*aa[k]
        t=np.linspace(-1.8,1.8,91)[:,None]*bb[k]
        vv=sample(light,cx+cs*u-sn*t,cy+sn*u+cs*t)
        width=np.sum(vv>max(peak*lev,2.6*noise,.012),axis=0)
        top=width.max()
        if top<3 or np.sum(width>.1*top)<4:
            diamonds.append(0.)
            continue
        length=np.sum(width>top*.63)/(np.sum(width>top*.18)+eps)
        # ellipse: length ratio ~0.79; a rhombus: ~0.45.
        diamonds.append(1-ramp(length,.44,.79))
    diamond = .25*diamonds[0]+.42*diamonds[1]+.33*diamonds[2]
    # Distinguish central swelling from the ordinary inclination of a
    # uniformly thick disk by comparing its surface brightness profiles.
    t=np.linspace(-1,1,49)[:,None]*bb[3]
    u=aa[3]*np.array([0.,.38,-.38,.60,-.60])[None,:]
    vv=sample(light,cx+cp*u-sp*t,cy+sp*u+cp*t)
    denom=np.maximum(vv.max(0),.018)
    frac=np.maximum(vv,0)/denom[None,:]
    thick=np.sum(frac>.45,axis=0)
    swelling=thick[0]/(np.mean(thick[1:])+eps)
    # Exponential disks often remain disk-like far below the bright bulge.
    # Integrate in strips for greater sensitivity than a pixel threshold.
    t=np.linspace(-.22,.22,7)[:,None]*bb[3]
    u=np.arange(-72.,73.)[None,:]
    prof=sample(light,cx+cp*u-sp*t,cy+sp*u+cp*t).mean(0)
    sym = (prof[72:]+prof[72::-1])*.5
    sig = max(.008, 1.35*noise, .025*peak)
    cut=np.nonzero((sym[:-2]<sig)&(sym[1:-1]<sig)&(sym[2:]<sig))[0]
    winglen=float(cut[0]) if len(cut) else 70.
    return diamond,swelling,winglen


def ridges(polar,rr,scale,noise,peak):
    # Even a fairly axisymmetric disk has radial gradients. Remove its
    # radial light profile before measuring directions of individual arms.
    # A ridge with small pitch has its gradient primarily along log radius;
    # a loosely wrapped ridge has a larger azimuthal gradient component.
    smooth=(np.roll(polar,2,1)+4*np.roll(polar,1,1)+6*polar+4*np.roll(polar,-1,1)+np.roll(polar,-2,1))/16
    base=np.median(smooth,axis=1)[:,None]
    v=(smooth-base)/(np.maximum(0,base)+.035*peak+2.2*noise)
    v=np.clip(v,-1.4,1.4)
    da=(np.roll(v,-2,1)-np.roll(v,2,1))/(4*math.pi/48)
    dr=(v[4:]-v[:-4])/(np.log(rr[4:]/rr[:-4])[:,None])
    da=da[2:-2]; zone=(rr[2:-2]>.52*scale)&(rr[2:-2]<1.48*scale)
    g1=dr[zone]; g2=da[zone]
    w=limit((abs(v[2:-2][zone])-.025)/.19)
    tr=np.mean(w*(g1*g1+g2*g2))+eps
    slope=math.sqrt((np.mean(w*g2*g2)+eps)/(np.mean(w*g1*g1)+eps))
    handed=2*abs(np.mean(w*g1*g2))/tr
    arc=(np.mean(w*abs(g2))+.5*np.mean(w*abs(g1)))
    return slope,handed,arc


def population_shape(red,blue,rr,scale,peak,noise):
    # Subtract the disk's chromaticity rather than its brightness. Light
    # from an old bar stands out even against a much brighter round disk.
    r=rr/scale
    i=(r>.54)&(r<1.03)
    ratio=(max(0.,np.median(red[i]))+.012)/(max(0.,np.median(blue[i]))+.012)
    difference=red-blue*np.clip(ratio,.48,1.9)
    amplitude=max(.025*peak,np.median(difference[rr<=2.5]))
    w=np.maximum(0,np.minimum(difference,amplitude)-max(.09*amplitude,1.65*noise))
    w=w*(rr*rr*(r<.76))[:,None]
    wsum=np.sum(w)+eps
    second=w*(rr*rr)[:,None]
    quad=np.sum(second*np.exp(2j*angle))/(np.sum(second)+eps)
    # Third/fourth moment contamination from a single off-centre star.
    lop=abs(np.sum(w*np.exp(1j*angle)))/wsum
    rpop=math.sqrt(np.sum(second)/wsum)
    strength=ramp(amplitude/(peak+.02),.04,.22)
    ell=abs(quad)
    bar = ramp(ell,.15,.51)*ramp(rpop/scale,.085,.31)*strength*(1-ramp(lop,.24,.65))
    # Inner red component at high inclination: a genuine bulge is rounder
    # than the disk in the original sky plane, i.e. vertically elongated
    # in this deprojected coordinate system.
    lens=ramp(-quad.real,.06,.40)*(1-.5*ramp(lop,.24,.62))*strength
    return bar,lens


def morphology(image):
    image = np.asarray(image,dtype=np.float64)
    smooth = np.array([blur(chan) for chan in image])
    sky = (X-63.5)**2+(Y-63.5)**2>55**2
    backgrounds = np.median(smooth[:,sky],axis=1)
    rgb = smooth-backgrounds[:,None,None]
    light = .32*rgb[0]+.45*rgb[1]+.23*rgb[2]
    noise = max(.002,np.median(abs(light[sky]))*1.48)
    # A centrality prior prevents off-axis stars being chosen as the galaxy.
    v = blur(light,1)
    choose = v*np.exp(-((X-63.5)**2+(Y-63.5)**2)/(2*6.**2))
    py,px = np.unravel_index(np.argmax(choose),choose.shape)
    cx,cy=float(px),float(py)
    for it in range(6):
        w = np.maximum(v,0)**2 * np.exp(-((X-cx)**2+(Y-cy)**2)/(2*2.4**2))
        cx = .03*63.5+.97*(np.sum(w*X)+eps*63.5)/(np.sum(w)+eps)
        cy = .03*63.5+.97*(np.sum(w*Y)+eps*63.5)/(np.sum(w)+eps)
    peak = max(.018,float(sample(light,cx,cy)))
    companion, objects = detect(v,rgb,cx,cy,noise)
    valid = (X-cx)**2+(Y-cy)**2<62**2
    # Exclude detached peaks in the outer field (not small arm knots near the core).
    for x,y,h,saddle,width,cd,sep in objects:
        if sep>8. and (saddle<max(.03, .095*peak) or (width<.42 and cd>.23)):
            starR = min(.40*sep, max(3., 4.5+5*width))
            valid &= (X-x)**2+(Y-y)**2>starR**2
    aa,bb,theta,mx,my,fill = isoshape(light, valid,cx,cy,peak,noise)
    paired = pair_envelope(v,cx,cy,aa,peak,noise,objects)
    qq=bb/aa
    q = .10*qq[1]+.40*qq[2]+.35*qq[3]+.15*qq[4]
    # Choose the disk frame with a robust mean of doubled position angles.
    dirs = np.exp(2j*theta)
    weights = np.array([0.,.04,.22,.44,.30])*(1-qq)
    phi = .5*np.angle(np.sum(weights*dirs))
    diskq = max(.15,min(1.,.25*qq[2]+.45*qq[3]+.30*qq[4]))
    cp,sp=math.cos(phi),math.sin(phi)
    tx,ty = radii*(ca*cp-diskq*sa*sp), radii*(ca*sp+diskq*sa*cp)
    sx,sy=cx+tx,cy+ty
    polar = sample(light,sx,sy)
    red = sample(rgb[0],sx,sy)
    blue = sample(rgb[2],sx,sy)
    vis = (sample(valid.astype(float),sx,sy)>.6)&(sx>=1)&(sy>=1)&(sx<=126)&(sy<=126)
    # Use the opposite side to fill isolated contaminated sectors. Medians
    # provide a stable baseline even for a disk with a foreground star.
    med = np.median(polar,axis=1)[:,None]
    for ring in (polar,red,blue):
        ring[:] = np.where(vis,ring,np.where(np.roll(vis,48,axis=1),np.roll(ring,48,axis=1),np.median(ring,axis=1)[:,None]))
    med = np.median(polar,axis=1)[:,None]
    profile = med[:,0]
    rr = radii[:,0]
    scale = min(65.,max(4.,radius_at(profile,max(peak*.10,2.7*noise,.012))))
    r = rr/scale
    # Dynamically restrict ring statistics to visible galaxy light.
    baseline = np.maximum(med,0)+.032*peak+noise*2+eps
    res = (polar-med)/baseline
    res = np.clip(res,-1.25,1.25)
    coeff = np.fft.rfft(res,axis=1)[:,1:10]*(2./96)
    ab = np.abs(coeff)
    odd = np.sqrt(ab[:,0]**2+ab[:,2]**2+ab[:,4]**2)
    fine = np.sqrt(np.sum(ab[:,2:]**2,axis=1))
    a1 = annmean(odd,r,.30,.92)
    aout = annmean(odd,r,.80,1.32)
    acore = annmean(odd,r,.12,.55)
    # Distance between the nucleus and the outer contour centroid. An
    # elongated disk is more sensitive to shifts in its minor dimension.
    dx,dy=mx-cx,my-cy
    off = np.sqrt(((dx*cp+dy*sp)/aa)**2+((dy*cp-dx*sp)/np.maximum(bb,1.))**2)
    offcentre = .55*min(.6,off[3])+.25*min(.6,off[2])+.20*min(.6,off[4])
    # Colour and the colour gradient distinguish old spheroids, dusty
    # midplanes and blue star forming disks. Ratios cancel surface brightness.
    rd = np.median(red,axis=1)
    bl = np.median(blue,axis=1)
    redness = (rd-bl)/(np.maximum(rd+bl,0)+.018)
    corecol = annmean(redness,r,0,.25)
    diskcol = annmean(redness,r,.42,.95)
    gradcol = corecol-diskcol
    # Compare the fall-off in the outer wing to the radius at half height.
    # A concentrated bulge plus disk can have two separate slopes.
    half = radius_at(profile,.50*peak)
    quarter = radius_at(profile,.25*peak)
    ext = max(aa[2], quarter)
    wing = scale/max(1.,half)
    # Nonaxisymmetry of high spatial frequency after removing the first
    # two Fourier modes of each ring, and the winding two-arm component.
    texture = annmean(fine,r,.35,1.02)
    inner2 = annmean(coeff[:,1],r,.16,.44)
    outer2 = annmean(coeff[:,1],r,.60,.97)
    baramp = abs(inner2-outer2)
    # Logarithmic spirals have a linear phase versus log radius. Cancel
    # the constant component (ellipticity and bars) and compare chirality.
    zone = (r>.27)&(r<1.16)
    armstrength=0.
    tightness=0.
    chirality=0.
    if np.sum(zone)>4:
        lr = np.log(r[zone])
        win = np.sin(math.pi*(np.arange(len(lr))+.5)/len(lr))**.65
        for m in (2,3):
            z = coeff[zone,m-1]
            z = z-np.sum(win*z)/np.sum(win)
            pitch = np.array([9.,13.,18.,24.,31.,40.,52.])
            freq = m/np.tan(pitch*math.pi/180.)
            waves = np.exp(1j*freq[:,None]*lr[None,:])
            cw = abs(np.sum(waves*(z*win),axis=1))/np.sum(win)
            ccw = abs(np.sum(waves.conj()*(z*win),axis=1))/np.sum(win)
            amp = np.maximum(cw,ccw)
            balance = np.abs(cw-ccw)/(amp+.015)
            value=amp*(.58+.42*balance)*(1. if m==2 else .87)
            best=np.argmax(value)
            if value[best]>armstrength:
                armstrength = value[best]
                tightness = float(pitch[best])
                chirality = balance[best]
    twostruct = annmean(ab[:,1],r,.38,1.04)
    reach, outerarms, openpitch, outerodd = outer_pattern(polar,vis,rr,scale,peak,noise)
    diskexcess, point = disksection(light,cx,cy,cp,sp,aa[3],bb[3],peak,noise)
    diamond, swelling, winglen = disk_shape(aa,bb,qq,theta,phi,cx,cy,light,peak,noise)
    slope, handed, arc = ridges(polar,rr,scale,noise,peak)
    oldbar, oldbulge = population_shape(red,blue,rr,scale,peak,noise)
    radialflat=half/max(quarter,1.)
    # The vertical thickness of a classical bulge grows towards the midplane
    # centre. Interpolate minor-axis widths at a *fixed fraction of the local
    # intensity*. This differs from isophotal chord width, where a brighter
    # column appears thicker even in an infinitely long constant-width disk.
    offsets=np.linspace(-2.,2.,121)[:,None]*bb[3]
    along=np.array([0.,.34,-.34,.49,-.49,.62,-.62])[None,:]*aa[3]
    strips=sample(light,cx+cp*along-sp*offsets,cy+sp*along+cp*offsets)
    maxima=np.maximum(np.max(strips,axis=0),.01)
    widths=(np.sum(strips>.58*maxima,axis=0)+np.sum(strips>.32*maxima,axis=0))*.5
    relativewidth=widths[0]/max(1.,np.median(widths[1:]))
    deltaq=qq[1]-np.mean(qq[2:4])
    # Trace the signed first radial moment of the outskirts. Tidal debris
    # often pulls several successive rings in the same direction.
    taildir=annmean(coeff[:,0],r,.72,1.38)
    north=np.abs(taildir)/(annmean(ab[:,0],r,.72,1.38)+.01)
    tailsignal=north*ramp(abs(taildir),.06,.28)
    # Spheroids have a continuously declining brightness profile. Disks
    # are less centrally concentrated, even if their arms are unresolved.
    cumulative=np.cumsum(np.maximum(profile-2*noise,0)*rr**2*(rr<1.55*scale))
    rhalf=np.interp(cumulative[-1]*.5,cumulative,rr)
    r90=np.interp(cumulative[-1]*.90,cumulative,rr)
    concentration=r90/max(rhalf,1.)
    centerred=annmean(rd-bl,r,0.,.20)/(peak+.025)
    # Strong bars: a fixed-phase elongated central isophote in a disk
    # frame, and an abrupt transition to rounder or twisted isophotes.
    innerround = np.mean(qq[:2]); outerround = np.mean(qq[2:])
    change = abs(np.mean((1-qq[:2])*dirs[:2])-np.mean((1-qq[2:])*dirs[2:]))
    barell = max(0.,outerround-innerround)
    bar = .48*ramp(baramp,.10,.40)+.37*ramp(change,.090,.30)+.15*ramp(barell,.045,.18)
    structure = (.34*ramp(texture,.048,.18) + .35*ramp(armstrength,.023,.105)
                 +.18*ramp(gradcol,.065,.24)+.13*ramp(twostruct,.07,.23))
    # Axis ratio of the central luminous lens compared with that of the
    # disk, and central excess thickness above an edge-on exponential disk.
    bx = np.linspace(-.90,.90,61)
    by = np.linspace(-1.6,1.6,81)
    u = bx[None,:]*scale
    height = max(2.,diskq*scale)
    t = by[:,None]*height
    cut = sample(light,cx+cp*u-sp*t,cy+sp*u+cp*t)
    # Minor-axis second moments in a vertical slice, measured above sky.
    cut = np.maximum(cut-max(.018*peak,2*noise),0.)
    mass = np.sum(cut,axis=0)+eps
    centroid = np.sum(cut*by[:,None],axis=0)/mass
    thick = np.sqrt(np.sum(cut*(by[:,None]-centroid)**2,axis=0)/mass)
    centerthick = np.mean(thick[np.abs(bx)<.14])
    sidethick = np.mean(thick[(np.abs(bx)>.35)&(np.abs(bx)<.60)])
    swell = centerthick/(sidethick+.018)
    # The bulge is intrinsically compact and rounder than the disk;
    # a red nucleus and excess central light support this classification.
    bulge = (.32*ramp(swell,1.14,1.64)+.23*ramp(np.mean(qq[:2])-q,.035,.19)
             +.24*ramp(corecol,.045,.25)+.14*ramp(gradcol,.025,.15)
             +.07*ramp(wing,2.6,4.3))
    # Edge-on discrimination: disk flatness, pinched vertical profiles,
    # visible extent and stellar population, not position angle.
    flatten = 1-ramp(q,.20,.36)
    lens = .48*ramp(ext,8.,19.)+.26*flatten+.12*ramp(swell,1.22,1.58)+.14*(1-ramp(diskcol,-.14,.14))
    edge = (1-ramp(q,.30,.49))*(.45+.55*lens)
    edge = max(edge,.95*(1-ramp(q,.16,.27)))
    # A smooth cigar can be as elongated as a disk but has no extended
    # thin outer body. Correct part of the PSF bias for the roundness test.
    qstar = math.sqrt(max(.04,bb[2]**2-1.6**2)/max(.05,aa[2]**2-1.6**2))
    apparent = .82*q+.18*min(1.,qstar)
    # Separate small asymmetry typical of spiral arms from a strong
    # one-sided tidal tail, or a small peculiar galaxy with an offset core.
    asym = (.39*ramp(a1,.105,.38)+.22*ramp(aout,.13,.52)
            +.23*ramp(offcentre,.05,.24)+.16*ramp(acore,.10,.34))
    tidal = .65*asym+.14*ramp(1-fill[4],.04,.23)+.21*ramp(aout-a1,.09,.35)
    irregular = (.78*asym+.22*tidal) * (1-.34*ramp(armstrength,.045,.14)*chirality)
    # Fuzzy votes are organized as a decision tree. All branch comparisons
    # are individual morphological criteria; there are no population priors
    # inferred from examples.
    spiral = (.49*structure+.22*ramp(ext,7.,18.)+.13*ramp(-diskcol,-.09,.12)
              +.10*ramp(gradcol,.035,.19)+.06*ramp(scale,12.,30.))
    spiral = max(spiral, .48*bar+.25*ramp(ext,7.,17.)+.16*ramp(gradcol,.02,.18)+.11*ramp(texture,.04,.17))
    if paired>.68 or (companion>.64 and (asym>.49 or paired>.49)):
        return 1
    # For a tidal tail the outer light centroid moves steadily to one
    # side. A blue peculiar may be a similarly asymmetric knotty system
    # without the regular even-order phase progression of a spiral.
    wander=math.hypot(mx[4]-mx[1],my[4]-my[1])/(max(aa[4],3.))
    lopsided=max(irregular,.54*outerodd+.26*ramp(wander,.05,.23)+.20*ramp(a1,.085,.25))
    uneven=(.4*ramp(a1,.085,.26)+.30*ramp(acore,.065,.22)+.30*ramp(offcentre,.035,.16))
    peculiar=.56*uneven+.24*ramp(-corecol,.01,.20)+.20*(1-ramp(ext,8.,20.))
    compact=1-ramp(scale,11.,23.)
    strange = (.37*tailsignal+.22*ramp(offcentre,.035,.16)+.19*compact
               +.14*ramp(-centerred,.015,.28)+.08*ramp(acore,.06,.19))
    # Disturbance also shows up as an isophotal twist in the outer wings:
    # compare ellipse orientations weighted by their eccentricities, so
    # position-angle noise on a nearly circular isophote has no effect.
    twist=abs((1-qq[4])*dirs[4]-(1-qq[2])*dirs[2])
    crooked=(.34*ramp(twist,.11,.35)+.27*ramp(1-fill[4],.025,.17)
             +.23*tailsignal+.16*ramp(aout,.10,.34))
    if (lopsided>.51 or (peculiar>.59 and a1>.085) or strange>.51 or (crooked>.52 and north>.46)) and (q>.38 or lopsided>.62 or crooked>.60) and not (chirality>.64 and armstrength>.065 and a1<.25):
        return 0
    # Old bulges produce a warm midplane peak surrounded by a cooler,
    # thinner disk. A projected pure exponential does not peak as sharply.
    brightbulge=(.25*ramp(np.mean(qq[:2])-np.mean(qq[2:]),.005,.15)
                 +.22*ramp(corecol,-.015,.22)+.19*ramp(diskexcess,-.15,.27)
                 +.17*ramp(point,.21,.52)+.17*ramp(gradcol,.005,.12))
    lenticular=.24*ramp(scale,11,24)+.21*ramp(point,.23,.55)+.23*ramp(barell*-1+np.mean(qq[:2])-q,.02,.18)+.32*(1-ramp(diskq,.22,.42))
    # A thin disk bearing a small bulge makes a lens with pointed ends.
    # The old stars in the bulge are redder than the flat blue disk. An
    # ordinary featureless elliptical lacks both a pinch and a thin skirt.
    redlens = (.29*ramp(centerred,-.065,.30)+.20*diamond+.15*ramp(swelling,1.0,1.45)
               +.12*ramp(gradcol,-.015,.15)+.12*ramp(winglen/aa[1],2.4,4.6)
               +.12*ramp(diskexcess,-.17,.23))
    edgewing = (.30*ramp(winglen,14.,33.)+.22*diamond+.19*ramp(winglen/aa[2],1.55,2.8)
                +.15*ramp(swelling,.99,1.4)+.14*(1-ramp(diskq,.25,.48)))
    # Broad uniform high-luminance midplanes favour a bulgeless disk, even
    # if it contains a red dust lane. A true bulge is a separate compact
    # component. Compare its central extent to the faint disk's full length.
    midplane=ramp(aa[1],7.,18.)*(1-ramp(qq[1],.24,.43))
    centralcomponent=(.34*ramp(relativewidth,1.035,1.39)+.22*ramp(deltaq,.01,.14)
                      +.17*ramp(corecol,-.035,.16)+.16*ramp(point,.20,.52)
                      +.11*ramp(oldbulge,.08,.60))
    longflat=midplane*(1-ramp(relativewidth,1.04,1.29))*(1-ramp(deltaq,.025,.13))
    bulgeEvidence=max(redlens,brightbulge,.32+oldbulge*.51,centralcomponent)-.16*longflat
    if (edge>.64 and (apparent<.34 or spiral<.57) and (q<.31 or lens>.46)) or (diskq<.42 and brightbulge>.43 and lenticular>.47) or (diskq<.49 and edgewing>.48 and (redlens>.48 or ext>12)) or (diskq<.51 and centralcomponent>.60 and diamond>.24 and winglen>2.15*aa[2]):
        return 9 if bulgeEvidence>.50 or bulge-.1*longflat>.48 else 8
    # Test the outer disk for blue light as well as for measurable texture.
    # A small blue galaxy can still be featureless; a *resolved* blue disk
    # with a distinct warm centre provides independent evidence for arms.
    resolved=ramp(scale,8.,21.)
    twocolor=ramp(gradcol,.045,.21)*ramp(-diskcol,-.065,.14)
    diskvote=.34*twocolor+.28*resolved+.19*ramp(-diskcol,-.07,.12)+.19*max(outerarms,structure)
    # Conjunction, not a sum: a red spheroid's colour gradient or a big
    # angular diameter is insufficient on its own to signal spiral structure.
    population=.44*twocolor+.27*ramp(-diskcol,-.03,.14)+.29*(1-ramp(concentration,1.72,2.45))
    resolved_disk=resolved*population
    armvote=max(spiral,diskvote,.29*outerarms+.34*reach+.18*resolved+.19*ramp(armstrength,.015,.075),.28+.49*resolved_disk)
    bland=(1-ramp(texture,.045,.12))*(1-ramp(armstrength,.015,.06))*(1-ramp(arc,.08,.22))
    armvote-=.15*bland*ramp(diskcol,-.04,.14)
    # A cigar projects to an oval even in the wings; at high inclination
    # require direct evidence before identifying an arm through foreshortening.
    # Compare the visible disk to a centrally concentrated spheroid using
    # both light distribution and the spatial pattern of colour. In a
    # featureless spheroid the blue and red rings have little azimuthal
    # structure after removing the slowly changing mean radial colour.
    colorval=(red-blue)/(np.maximum(red+blue,.0)+.04*peak+2*noise)
    azcolor = annmean(np.quantile(colorval,.80,axis=1)-np.quantile(colorval,.20,axis=1),r,.35,.96)
    isquiet=(1-ramp(armstrength,.018,.07))*(1-ramp(texture,.052,.19))
    spheroid=(.38*ramp(corecol,.08,.26)+.24*ramp(concentration,1.90,2.49)
              +.18*(1-ramp(gradcol,.08,.24))+.20*(1-ramp(azcolor,.055,.19)))*isquiet
    armvote-=.22*spheroid
    colddisk=(.34*ramp(-diskcol,.0,.17)+.20*ramp(azcolor,.05,.19)
              +.23*ramp(gradcol,.055,.21)+.23*ramp(radialflat,.53,.74))
    # Colour locates candidate disk material, but doesn't prove it is an
    # arm. Couple a colour vote to a resolved non-radial pattern.
    traced=max(ramp(armstrength,.02,.086),ramp(texture,.055,.175),ramp(handed,.11,.39)*ramp(arc,.085,.25))
    armvote=max(armvote,.25+.49*colddisk*(.28+.72*traced)+.13*resolved*traced-.14*spheroid)
    armvote=max(armvote,.27+oldbar*(.38+.23*resolved)*ramp(change,.025,.16))
    if armvote>(.49+.17*(1-ramp(q,.35,.59))) and q>.27:
        # A bar may be strongest near its two ends, not in the innermost
        # isophote, where a luminous round nucleus dilutes the quadrupole.
        barscan=0.
        for l,h in ((.18,.38),(.28,.52),(.38,.67)):
            z=annmean(coeff[:,1],r,l,h)
            end=annmean(coeff[:,1],r,h*1.20,h*1.65)
            aligned=abs(z)/(annmean(ab[:,1],r,l,h)+.015)
            localbar=(.48*ramp(abs(z),.085,.31)+.29*ramp(abs(z-end),.10,.31)+.23*ramp(corecol-diskcol,.03,.18))*aligned
            barscan=max(barscan,localbar)
        if (bar>.49 and (abs(inner2)>.092 or barell>.13) and (bar>.66 or corecol>-.06)) or (barscan>.52 and corecol>-.035) or (oldbar>.42 and (barscan>.25 or change>.11)):
            return 5
        # Measure the arm *outside* the bright central part, emphasizing
        # high contrast arcs beyond the disk and their winding rate.
        loose = (.19*ramp(armstrength,.026,.12)+.12*ramp(texture,.09,.24)
                 +.14*ramp(tightness,18.,37.)*ramp(armstrength,.02,.08)
                 +.14*reach+.16*outerarms+.12*openpitch+.13*ramp(aout,.105,.34))
        # Combine a geometric pitch estimate from the structure tensor with
        # the phase measurement. A long young arm can have little residual
        # texture locally despite an open trajectory on the sky.
        pitchvote=ramp(slope,.24,.68)*(.58+.42*ramp(handed,.075,.32))
        outercontrast=annmean(np.quantile(res,.85,axis=1)-np.quantile(res,.15,axis=1),r,.64,1.20)
        m2dom=annmean(ab[:,1]/(.04+fine+ab[:,0]),r,.46,1.2)
        winding=(.30*pitchvote+.23*ramp(outercontrast,.15,.42)+.18*ramp(m2dom,.45,1.25)
                 +.14*ramp(arc,.095,.33)+.15*ramp(tightness,22.,42.)*ramp(armstrength,.015,.06))
        # A tightly wound multiring pattern must show the same handedness
        # in several radial ranges. Radial PSF blurring can raise the local
        # pitch estimate, so permit coherent logarithmic phase to override it.
        tight=ramp(handed,.14,.41)*chirality*(1-ramp(tightness,16.,30.))
        return 7 if loose>.38 or winding>.50+.13*tight else 6
    if apparent>.78:
        return 2
    # Allow for seeing: unresolved narrow cores are broadened in quadrature.
    intrins=math.sqrt(max(1.,bb[3]**2-2.4**2)/max(1.,aa[3]**2-2.4**2))
    if (.65*apparent+.35*intrins)>.52:
        return 3
    return 4


def predict(x):
    return np.array([morphology(image) for image in x],dtype=np.int64)
