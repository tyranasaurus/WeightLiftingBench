import numpy as np
import math

Y, X = np.indices((128, 128), dtype=np.float64)
border = np.maximum(abs(X-63.5), abs(Y-63.5)) > 53
ang = np.arange(96)*(2*np.pi/96)
cs, sn = np.cos(ang), np.sin(ang)
kx = np.fft.rfftfreq(128)
ky = np.fft.fftfreq(128)
k2 = (kx[None,:]**2 + ky[:,None]**2)*(2*np.pi**2)
scales = (1.05, 1.8, 2.8, 4.5, 7.0)
kernels = [np.exp(-k2*s*s) for s in scales]
tiny = 0.000001

def unit(v):
    return np.clip(v,0.,1.)

def ramp(v,a,b):
    return unit((v-a)/(b-a))

def sig(v):
    return 1/(1+np.exp(-np.clip(v,-40,40)))

def sample(z,x,y):
    x = np.clip(x,0,126.9999)
    y = np.clip(y,0,126.9999)
    i, j = x.astype(np.int32), y.astype(np.int32)
    s,t = x-i,y-j
    return (1-t)*((1-s)*z[j,i]+s*z[j,i+1])+t*((1-s)*z[j+1,i]+s*z[j+1,i+1])

def blur(z,k):
    return np.fft.irfft2(np.fft.rfft2(z)*kernels[k],s=(128,128))

def mad(z):
    return 1.4826*np.median(abs(z-np.median(z)))

def meanw(v,w):
    return np.sum(v*w)/(np.sum(w)+tiny)

def geom(w,dx,dy):
    a = np.sum(w*dx**2) + tiny
    b = np.sum(w*dy**2) + tiny
    c = np.sum(w*dx*dy)
    d = min(np.hypot(a-b,2*c),a+b-tiny)
    ratio = np.sqrt(max(a+b-d,tiny)/(a+b+d))
    return 0.5*np.arctan2(2*c,a-b), np.clip(ratio,.075,1.)

def iso(z,noise,cx,cy):
    # Trace the first outward crossing on each ray.  Moments of the
    # enclosed areas give the position angle and axis ratio of isophotes.
    rs = np.arange(0,63,.5)
    rays = sample(z,cx+cs[:,None]*rs,cy+sn[:,None]*rs)
    peak = max(np.median(rays[:,0]), tiny)
    floor = max(.004,2.8*noise)
    radii,qs,dirs,drifts,miss,support = [],[],[],[],[],[]
    for f in (.68,.44,.27,.16,.095,.05):
        level = max(f*peak, floor)
        below = rays < level
        first = np.argmax(below,axis=1)
        first = np.where(below.any(axis=1),first,len(rs)-1)
        k = np.maximum(0,first-1)
        aa = rays[np.arange(96),k]
        bb = rays[np.arange(96),first]
        rad = np.maximum(.65,rs[k]+.5*unit((aa-level)/np.maximum(aa-bb,tiny)))
        # Damp narrow spikes from overlapping field objects, without
        # requiring either bilateral symmetry or fixed contour curvature.
        rad = np.median(np.stack([np.roll(rad,h) for h in (-2,-1,0,1,2)]),axis=0)
        opp = np.roll(rad,48)
        clean = np.minimum(rad,1.35*opp+1)
        pa,q = geom(clean**4,cs,sn)
        sma = np.sqrt((clean**2).mean()/q)
        re = sma/np.sqrt(np.cos(ang-pa)**2+(np.sin(ang-pa)/q)**2)
        err = np.median(abs(np.log((rad+.4)/(re+.4))))
        off = 2./3*np.hypot(np.sum(rad**3*cs),np.sum(rad**3*sn))/(np.sum(rad**2)+tiny)
        radii.append(sma); qs.append(q); dirs.append(pa)
        drifts.append(off/(np.sqrt(q)*sma+.9)); miss.append(err)
        support.append(unit((f*peak/floor-.65)/.55))
    return tuple(np.asarray(v) for v in (radii,qs,dirs,drifts,miss,support))

def getpeaks(blurs, noises, cx,cy):
    # A companion must be resolved, stand out above the local saddle,
    # and be reasonably bright compared with the main galaxy.  Separately
    # identify PSF-like objects so they do not create spiral evidence.
    r = np.hypot(X-cx,Y-cy)
    pz = blurs[1]
    peak = max(sample(pz,cx,cy),.008)
    local = np.ones((128,128),dtype=np.bool_)
    for di,dj in ((0,1),(1,0),(1,1),(1,-1)):
        v = np.roll(pz,(di,dj),(0,1)); w = np.roll(pz,(-di,-dj),(0,1))
        local &= (pz>=v)&(pz>w)
    yy,xx = np.where(local & (pz>max(noises[1]*4,.043*peak)) & (r>3.2) & (r<70))
    if len(xx)>60:
        take = np.argsort(pz[yy,xx])[-60:]; yy,xx = yy[take],xx[take]
    objs = []
    for j,i in zip(yy,xx):
        dx,dy = i-cx,j-cy
        dist = np.hypot(dx,dy)
        u,v = dx/dist,dy/dist
        height = pz[j,i]
        # lower limit on local background estimated to the sides;
        # core width from the rate at which a Gaussian washes out
        # the peak, as well as the brightness distribution around it.
        outer = sample(pz,i+11*cs,j+11*sn)
        base = max(0.,np.percentile(outer,20))
        bump = max(height-base,tiny)
        growth = (blurs[2][j,i]-base)/bump
        resolved = ramp(growth,.61,.83)
        ring = sample(pz,i+5*cs,j+5*sn)
        broad = ramp((np.percentile(ring,40)-base)/bump,.15,.52)
        extended = .55*resolved+.45*broad
        tline = np.linspace(0,1,32)
        cut = sample(pz,cx+dx*tline,cy+dy*tline)
        saddle = np.min(cut)
        contrast = unit((min(peak,height)-saddle)/(min(peak,height)+tiny))
        # Compare light along the bridge with unrelated sky in its
        # vicinity. A linked pair is more persuasive than two peaks.
        tbridge = np.clip(tline[int(np.argmin(cut))],.25,.75)
        bx = cx+tbridge*dx; by=cy+tbridge*dy
        perp = np.clip(.28*dist,4,12)
        back = (sample(pz,bx+v*perp,by-u*perp)+sample(pz,bx-v*perp,by+u*perp))*.5
        bridge = max(0.,sample(pz,bx,by)-back)
        objs.append((i,j,dist,height/peak,contrast,extended,bridge/peak,saddle/peak,growth))
    return objs

def prepare(im):
    rgb = im.astype(np.float64)
    for ch in rgb:
        ch -= np.median(ch[border])
    lum = .31*rgb[0]+.42*rgb[1]+.27*rgb[2]
    ft = np.fft.rfft2(lum)
    z = [np.fft.irfft2(ft*kernel,s=(128,128)) for kernel in kernels]
    noises = []
    for v in z:
        v -= np.median(v[border]); noises.append(max(.0005,mad(v[border])))
    # Small search, guided by the known survey cutout centre.
    prior = np.exp(-((X-63.5)**2+(Y-63.5)**2)/(2*7**2))
    y,x = np.unravel_index(np.argmax(z[1]*prior),lum.shape)
    cx,cy = float(x),float(y)
    for width in (3.,2.,2.):
        dx,dy = X-cx,Y-cy
        w = np.maximum(z[0],0)*np.exp(-(dx**2+dy**2)/(2*width**2))
        cx += np.clip(meanw(dx,w),-1.,1.); cy += np.clip(meanw(dy,w),-1.,1.)
    if (cx-63.5)**2+(cy-63.5)**2 > 12**2:
        cx,cy=63.5,63.5
    return rgb,z,noises,cx,cy

def spiralwave(field,profile,ann,rr,q,noise,top,period):
    # Matched phase sums in log radius. Constant-pitch ridges have
    # theta = cot(pitch)*log(radius) + constant, modulo arm spacing.
    # Evaluate both chiralities and several contiguous radial spans.
    coeff = np.fft.rfft(field,axis=1)/field.shape[1]
    freq = np.array([0.,.22,.48,.75,1.05,1.40,1.80,2.3,2.85,3.6,4.5,5.8,7.5,9.5])
    freq = np.concatenate((-freq[:0:-1],freq))
    den = np.maximum(profile,0)+max(noise*3,.026*top,.003)
    lnr = np.log(rr)
    best=0.; bestk=0.; tw=0.; straight=0.; power=0.
    for m in (2,3,4):
        phase = np.exp(-1j*m*freq[:,None]*lnr)
        c = coeff[:,m]/den
        mod = np.abs(c)
        c = c*np.minimum(1.,.65/(mod+tiny))
        for lo,hi in ((.26,1.28),(.44,1.52),(.67,1.8)):
            window = ramp(lnr,np.log(lo*period)-.17,np.log(lo*period)+.14)*ramp(-lnr,-np.log(hi*period)-.16,-np.log(hi*period)+.14)
            w = ann*window*ramp(rr*q,1.3,3.2)
            sw = np.sum(w)+tiny
            response = abs(np.sum(phase*(c*w),axis=1))/sw
            flat = response[len(freq)//2]
            # An ellipse or bar leaks power from zero angular drift;
            # rewarding improvement over flat suppresses that leakage.
            gain = np.maximum(response-flat,0.)
            real = response*(.24+.76*unit(gain/(.45*response+tiny)))
            real = np.where(abs(freq)>=.48,real,0.)
            extent = np.sqrt(meanw((lnr-meanw(lnr,w))**2,w))
            real *= (.78 if m==3 else .68 if m==4 else 1.)*ramp(extent,.115,.27)*min(1.,sw/3)
            b = int(np.argmax(real)); val = real[b]
            if val>best:
                best=val; bestk=abs(freq[b]); tw=gain[b]
            straight=max(straight,flat*(1. if m==2 else .6))
            power=max(power,meanw(mod,w)*(.8 if m>2 else 1.))
    return best,bestk,tw,straight,power

def classify(im):
    rgb,z,noise,cx,cy = prepare(im)
    if z[1].max()<.004:
        return 2
    rad,qs,pas,drift,mis,supp=iso(z[0],noise[0],cx,cy)
    # The disk's inclination comes from the outer light, and the bar
    # test will compare this frame to that of the inner isophotes.
    wg = np.array([0.,.04,.13,.31,.38,.14])*(.12+.88*supp)
    q=float(meanw(qs,wg))
    st=wg*(1-qs+.025)
    pa = .5*np.arctan2(np.sum(st*np.sin(2*pas)),np.sum(st*np.cos(2*pas)))
    ca,sa=np.cos(pa),np.sin(pa)
    # Slight deconvolution only for the shape labels. The sampling
    # ellipse follows the observed (PSF-convolved) light.
    size = max(3.,rad[3]); qtrue=np.sqrt(max((q*size)**2-1.25**2,(.075*size)**2)/max(size**2-1.25**2,1.))
    objs=getpeaks(z,noise,cx,cy)
    veto=np.ones_like(z[0]); pair=0.; multiple=0.
    for i,j,d,rel,ctr,ext,br,sd,grow in objs:
        # A nucleus can be compact, but two stars without a common
        # envelope are not called a merger.
        near = 1/(1+(d/max(15.,1.6*size))**2)
        bridge = ramp(br,.009,.095)*ramp(sd,.035,.18)
        link = max(near,bridge*ramp(64-d,0,26))
        strength = ramp(np.log(max(rel,.001)),np.log(.085),np.log(.70))
        strength *= ramp(ctr,.08,.45)*ramp(d,3.3,7.)
        p = strength*(.20+.80*ext)*(.30+.70*link)
        p += .23*bridge*strength*(.25+.75*ext)
        pair=max(pair,p)
        multiple=max(multiple,strength*ext*link)
        # For morphology, mask isolated contaminants but keep most
        # of the connected, relatively faint star-forming knots.
        if (ext<.46 and ctr>.29 and rel>.13) or (ctr>.76 and sd<.12):
            width=np.clip(3.0+2.6*ext+1.0*np.log1p(rel),3.2,9.)
            dd=(X-i)**2+(Y-j)**2
            veto=np.minimum(veto,ramp(dd,width**2,(width+2.5)**2))
    # Area element is constant at fixed log radius up to a factor r^2.
    # Smooth polar profiles supply both a symmetric model and aperture
    # fluxes without including bright off-axis stars.
    rr=np.geomspace(.7,89.,76)
    px=cx+rr[:,None]*(cs*ca-sn*q*sa)
    py=cy+rr[:,None]*(cs*sa+sn*q*ca)
    ww=sample(veto,px,py)*((px>=.5)&(px<126.5)&(py>=.5)&(py<126.5))
    low = sample(z[0],px,py)
    med=np.median(low,axis=1)
    # Replace missing angles by the median; winsorize unrelated peaks
    # relative to the dispersion *on that ring*.
    scat=1.4826*np.median(abs(low-med[:,None]),axis=1)
    cap=med+np.maximum(3.8*scat, .48*np.maximum(med,0)+2.5*noise[0])
    fld=ww*np.minimum(low,cap[:,None])+(1-ww)*med[:,None]
    prof=fld.mean(axis=1)
    top=max(prof[0],.006)
    last=(prof<max(.018*top,2.2*noise[0],.003))&(rr>max(2.6,.75*size))
    limit=rr[int(np.argmax(last))] if np.any(last) else rr[-1]
    limit=np.clip(limit,max(3.,size),min(80.,max(8.,4.*size)))
    area=rr**2*(rr<limit)
    area[0]+=rr[0]**2/(2*np.log(rr[1]/rr[0]))
    light=area*np.maximum(prof,0.)
    cumul=np.cumsum(light); total=cumul[-1]+tiny
    r20,r50,r80=np.interp(np.array([.20,.50,.80])*total,cumul,rr)
    a=max(4.,r80)
    # The mean colours of disk and nucleus discriminate a blue stellar
    # disk from a smooth red spheroid. The r channel is less sensitive
    # to sky noise than the extremal bands in many of these cutouts.
    channels=[]
    for ch in rgb:
        c=sample(blur(ch,0),px,py)
        t=np.median(c,axis=1)
        half=2.7*np.median(abs(c-t[:,None]),axis=1)+noise[0]*2
        channels.append(np.mean(ww*np.clip(c,(t-half)[:,None],(t+half)[:,None])+(1-ww)*t[:,None],axis=1))
    red,green,blue=channels
    def colour(w):
        r,g,b = meanw(red,w),meanw(green,w),meanw(blue,w)
        return (b-r)/(max(r+g+b,.015)/1.5+.016)
    corecol=colour(area*(rr<max(2.,r50*.55)))
    diskcol=colour(area*(rr>max(1.7,r50*.82))*(rr<1.20*a))
    wholecol=colour(area)
    chrom=diskcol-corecol
    # Ring contrast: total, odd (180-degree residual), and higher
    # Fourier orders. The last are insensitive to small inclination
    # errors. Local asymmetry is measured in flux units, not pixels.
    norm=np.maximum(prof,0)+max(.018*top,noise[0]*2.5,.003)
    ann=ramp(prof,1.3*noise[0],5.5*noise[0])*ramp(rr*q,.85,2.3)
    ann*=ww.mean(axis=1)**2
    wd=ann*ramp(rr,.22*a,.50*a)*ramp(-rr,-1.52*a,-.99*a)
    residual=fld-prof[:,None]
    spec=np.fft.rfft(residual,axis=1)/len(cs)
    higher=np.sqrt(2*np.sum(abs(spec[:,3:11])**2,axis=1))
    hi=meanw(np.sqrt(np.maximum(higher**2-(.75*noise[0])**2,0))/norm,wd)
    odd=np.mean(abs(fld-np.roll(fld,48,axis=1)),axis=1)
    odd=np.sqrt(np.maximum(odd**2-(1.13*noise[0])**2,0))
    asym=meanw(odd/norm,wd)
    asymflux=meanw(odd/norm,light*ann)
    # Repeat the winding measurement in a bluer band mix. The red bulge
    # can dominate the luminance image even if the arms stand out in g.
    wave,k,turn,flat,mod=spiralwave(fld,prof,ann,rr,q,noise[0],top,a)
    short=blur(.64*rgb[2]+.36*rgb[1],0)
    ns=max(.0008,mad(short[border]))
    bf=sample(short,px,py)
    bm=np.median(bf,axis=1); bs=1.4826*np.median(abs(bf-bm[:,None]),axis=1)
    bf=ww*np.minimum(bf,(bm+np.maximum(3.8*bs,.48*np.maximum(bm,0)+2.5*ns))[:,None])+(1-ww)*bm[:,None]
    bp=bf.mean(axis=1)
    bw,bk,bt,bflat,bmod=spiralwave(bf,bp,ann*ramp(bp,1.4*ns,5*ns),rr,q,ns,max(bp[0],.005),a)
    if .86*bw>wave:
        wave,k,turn=.86*bw,bk,bt
    warp=meanw(mis[2:],wg[2:]); shift=meanw(drift[2:],wg[2:])
    # Disturbed galaxies can have a smooth-looking inner body but a
    # lopsided or displaced outer envelope. Use a broader matched PSF to
    # measure faint light, and mask *both* ends of each symmetry pair.
    tidalimg=z[3] if a>18 else z[2]
    tidalnoise=noise[3] if a>18 else noise[2]
    tf=sample(tidalimg,px,py); tm=np.median(tf,axis=1)
    tmask=np.minimum(ww,np.roll(ww,48,axis=1))
    tr=np.abs(tf-np.roll(tf,48,axis=1))
    # A sparse residual is usually a field object; an offset envelope
    # affects a substantial arc. Use the median over usable pairs.
    tord=np.argsort(tr,axis=1)
    tc=np.cumsum(np.take_along_axis(tmask,tord,axis=1),axis=1)
    ti=np.argmax(tc>=.5*tc[:,-1:],axis=1)
    tres=tr[np.arange(len(rr)),tord[np.arange(len(rr)),ti]]
    tres=np.sqrt(np.maximum(tres**2-(1.05*tidalnoise)**2,0))
    tw=ramp(rr,.70*a,1.01*a)*ramp(-rr,-2.17*a,-1.65*a)*ramp(tm,1.65*tidalnoise,4.5*tidalnoise)*(tc[:,-1]/len(cs))**2
    tailasym=meanw(tres/(np.maximum(tm,0)+2.5*tidalnoise+.003*top),tw)*min(1.,tw.sum()/3)
    # A bar must be elongated in the plane of the outer disk. Look for
    # a nearly fixed m=2 phase over the inner disk, with a drop in the
    # outer disk. This does not mistake a uniformly inclined disk for
    # a bar. A mild central twist alone is common in smooth spheroids.
    c2=spec[:,2]/norm
    inner=ann*ramp(rr,1.1,2.4)*ramp(rr,.10*a,.22*a)*ramp(-rr,-.73*a,-.47*a)
    outer=ann*ramp(rr,.56*a,.82*a)*ramp(-rr,-1.57*a,-1.13*a)
    bi=np.sum(c2*inner)/(inner.sum()+tiny)
    bo=np.sum(c2*outer)/(outer.sum()+tiny)
    coherent=abs(bi)/(meanw(abs(c2),inner)+tiny)
    # Distinct elongation or PA in isophotes, corroborating the Fourier
    # test. Use the quadrupole of the inner ellipse after deprojection.
    delta=pas-pa
    h=(np.cos(delta)**2+qs**2*np.sin(delta)**2)
    j=(np.sin(delta)**2+qs**2*np.cos(delta)**2)/q**2
    xy=(1-qs**2)*np.cos(delta)*np.sin(delta)/q
    e=np.sqrt((h-j)**2+4*xy**2)/(h+j)
    barshape=max(e[1],(e[0]+e[2])*.5)
    bar=abs(bi)*(.4+.6*coherent)*(.4+.6*ramp(abs(bi-bo),.02,.15))
    # An inner region rounder than an inclined disk is a bulge rather
    # than a bar aligned with the disk's minor axis after deprojection.
    rounder=ramp(qs[1]-q,.04,.23)*(1-ramp(abs(np.sin(2*(pas[1]-pa))),.25,.8))
    bar*=1-.82*rounder
    barshape*=1-.83*rounder
    # Disk vs. spheroid: blue light, an outward blue colour gradient,
    # high-frequency structure, and winding are independent cues.
    # A red smooth object can have appreciable m=2 ellipticity gradients.
    bluevote=ramp(diskcol,-.18,.27)
    bluecore=ramp(wholecol,-.18,.23)
    gradvote=ramp(chrom,.085,.38)
    text=ramp(hi,.075,.27)
    winding=ramp(wave,.024,.13)
    ellerr=ramp(warp,.058,.16)
    barvote=ramp(bar,.042,.15)*ramp(barshape,.25,.66)
    concentration=r80/max(r20,.8)
    diffuse=ramp(3.7-concentration,-.2,1.1)
    featured=sig(-3.00+1.95*bluevote+.70*bluecore+.84*gradvote+2.15*text+2.25*winding+.8*ellerr+1.65*barvote+.43*diffuse+.38*ramp(asym,.13,.39))
    # Gross departures from point symmetry and off-centre isophotes
    # indicate a disturbance even when no second nucleus is resolved.
    # Strong regular winding is counterevidence.
    tidal=ramp(tailasym,.24,.76)
    irregular=sig(-4.05+2.05*ramp(asymflux,.16,.57)+1.02*ramp(asym,.20,.65)+1.45*ramp(shift,.085,.32)+1.05*ramp(warp,.075,.24)+.60*ramp(hi,.15,.37)-1.05*winding+.45*ramp(multiple,.16,.65)+1.40*tidal)
    shell=ramp(abs(np.sin(2*(pas[1]-pa))),.30,.93)*ramp(1-qs[1],.13,.34)*ramp(1-q,.13,.34)
    irregular=max(irregular,sig(-4.25+2.45*tidal+1.75*ramp(shift,.075,.25)+1.40*shell+.80*ellerr-1.1*winding-.7*barvote))
    merging=sig(-4.65+8.2*pair+.90*ramp(asymflux,.19,.52)+.50*ramp(shift,.09,.32)+.68*tidal)
    # Edge-on disk geometry. Compare isophote flattening at different
    # brightness levels and thickness through the nucleus and on either
    # side. Bulgeless disks have roughly constant vertical scale height;
    # a bulge widens and rounds the centre and often introduces a redder
    # population. A central spike in the major-axis profile is another cue.
    def thickness(u):
        v=np.arange(-28.,28.01,.5)
        b=sample(z[0],cx+u*ca-v*sa,cy+u*sa+v*ca)
        b=np.maximum(b-max(noise[0]*1.4,.012*top),0)*sample(veto,cx+u*ca-v*sa,cy+u*sa+v*ca)
        s=np.cumsum(b)
        quant=np.interp(np.array([.17,.83])*(s[-1]+tiny),s,v)
        return max(.75,(quant[1]-quant[0])*.5)
    side=max(3.,.64*a)
    central=thickness(0.)
    outerthick=(thickness(side)+thickness(-side))*.5
    wid=central/max(.85,outerthick)
    # A dust lane is relatively blue-opaque, parallel to the major
    # axis and close to it, although not always centred on the nucleus.
    # A depression in the short-band vertical profiles, or a dark red
    # midplane persisting over many columns, supports an edge-on view.
    uu=np.concatenate((np.linspace(-.70,-.15,10),np.linspace(.15,.70,10)))*a
    vv=np.arange(-12.,12.01,.6)
    lx=cx+ca*uu[:,None]-sa*vv
    ly=cy+sa*uu[:,None]+ca*vv
    lv=sample(z[0],lx,ly); bv=sample(short,lx,ly)
    by=np.maximum(bv,0)/(np.maximum(lv,0)+.023*top+noise[0])
    lh=max(1.65,min(5.,central*.7)); bank=[]
    for off in (-.5*lh,0.,.5*lh):
        idx=int(np.argmin(abs(vv-off))); stride=int(np.round(lh/.6))
        shoulders=.5*(bv[:,idx-stride]+bv[:,idx+stride]); mid=bv[:,idx]
        notch=(shoulders-mid)/(np.maximum(shoulders,0)+3*ns+.025*top)
        colourdiff=.5*(by[:,idx-stride]+by[:,idx+stride])-by[:,idx]
        bank.append(.65*ramp(np.median(notch),.02,.24)+.62*ramp(np.median(colourdiff),.055,.23))
    dust=max(bank)*ramp(.57-q,0,.22)*ramp(a,6,14)
    # Side profiles of a projected exponential disk flatten at the
    # origin. Extrapolating the outer trend reveals excess nuclear light.
    def maj(t):
        u=max(1.3,a*t)
        return max(.003,.5*(sample(z[0],cx+ca*u,cy+sa*u)+sample(z[0],cx-ca*u,cy-sa*u)))
    excess=np.log(max(top,.004)*maj(.87)/maj(.435)**2)
    innerq=meanw(qs[:3],np.array([.25,.50,.25]))
    qdisk=min(qtrue,.55*qtrue+.45*qs[-2])
    longdisk=ramp(a,7,19)
    # Colour, a thin outer disk, and dust/asymmetry are particularly
    # useful where the cigar and edge-on categories overlap in q.
    edgecut=.270+.055*longdisk+.063*bluevote+.043*ramp(innerq-q,.055,.23)+.032*text+.015*gradvote+.105*dust-.040*winding*ramp(q,.25,.48)
    edge=sig((edgecut-qdisk)/.042)
    edge*=1-.45*irregular
    seeingbias=.65/(1+rad[1]*q)**2
    bulge=sig(-1.32+1.90*ramp(innerq-q-seeingbias,.048,.22)+1.65*ramp(wid,1.22,2.05)+1.05*ramp(excess,.13,.85)+.64*ramp(chrom,.095,.35)-.83*bluecore+.35*ramp(concentration,3.35,4.65))
    # A sharp bulge-and-disk signature makes a thick disk more plausible.
    edge=max(edge,.87*bulge*sig((.43-qdisk)/.037)*ramp(innerq-q,.09,.25))
    roundq=.62*qtrue+.38*meanw(qs[1:4],np.array([.24,.45,.31]))
    sphround=sig((roundq-.84)/.035)
    sphcigar=sig((.465-roundq)/.041)
    barred=sig(-2.65+3.85*barvote+1.10*ramp(barshape,.38,.72)+.50*ramp(q,.48,.78)-.55*rounder-.30*ramp(asym,.29,.62))
    # Tight arms have a larger phase slope on a log radius plot. Use
    # the colour and the prominence of the bulge as weak fallbacks
    # when spatial resolution or inclination makes pitch ambiguous.
    reliable=ramp(wave,.027,.105)*ramp(q,.25,.53)
    bypitch=sig((k-2.75)/.68)
    prior=sig(.10+.95*ramp(chrom,.08,.37)+.52*ramp(concentration,3.4,4.8)-.85*bluecore-.90*ramp(asym,.18,.46)+.49*ramp(.62-q,0,.33))
    tight=reliable*bypitch+(1-reliable)*prior
    # Soft hierarchical decisions. Flatten the subordinate probabilities
    # slightly so that a well-supported family is not rejected merely
    # because two adjacent subclasses are close.
    other=(1-merging)*(1-irregular)
    sm=other*(1-edge)*(1-featured)
    sp=other*(1-edge)*featured
    ed=other*edge
    vals=[(1-merging)*irregular,merging,sm*sphround**.80,sm*max(.001,1-sphround-sphcigar)**.80,sm*sphcigar**.80,sp*barred**.78,sp*((1-barred)*tight)**.78,sp*((1-barred)*(1-tight))**.78,ed*(1-bulge)**.82,ed*bulge**.82]
    return int(np.argmax(vals))

def predict(x):
    return np.array([classify(im) for im in x],dtype=np.int64)
