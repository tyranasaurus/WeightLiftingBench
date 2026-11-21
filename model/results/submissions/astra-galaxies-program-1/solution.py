import numpy as np
import math

# Pixel coordinates and spatial-frequency squared for Gaussian scale space.
Y,X=np.mgrid[:128,:128]
D=(X-63.5)**2+(Y-63.5)**2
skyzone=D>56**2
freq=(np.fft.fftfreq(128)[:,None]**2+np.fft.rfftfreq(128)[None,:]**2)*(-2*math.pi**2)
kernels=[np.exp(freq*s*s) for s in (1.15,2.2,4.5)]
theta=np.arange(128)*math.tau/128
cs=np.cos(theta)
sn=np.sin(theta)
second=np.exp(2j*theta)
rr=np.arange(0,85,0.5)[:,None]
lev=np.array([.8,.68,.55,.43,.32,.23,.16,.11,.075,.05,.033])
fr=np.geomspace(.07,1.5,52)
lf=np.log(fr)
# Log-polar derivatives along a bank of tangent directions. A zero-pitch
# template is a concentric annulus; here the templates track winding ridges.

def ridgefield(p,med,noise):
    # Compare successive annuli after rotating one in either direction.
    # The same galaxy inclination/spurious field objects can create a
    # stationary response. Subtract that term before counting winding.
    ff=np.fft.rfft(p,axis=1)*2/len(cs)
    modes=np.arange(2,11)
    v=ff[:,2:11]
    # Remove slowly varying quadrupole due to oval disk projection, while
    # retaining the radial phase changes of genuine arms.
    til=np.arange(len(v))
    smooth=np.exp(-.5*((til[:,None]-til)/4.5)**2)
    v=v-(smooth@v)/smooth.sum(axis=1)[:,None]
    eligible=(fr>.22)&(fr<1.04)&(med>2.3*noise)
    w=eligible*np.minimum(1,np.maximum(med,0)/(7*noise))
    sig=0.; best=0.; opened=0.; tight=0.
    for pitch in (10,15,21,29,40,55):
        bc=0.
        for hand in (-1,1):
            energy=0.; norm=0.
            for lag in (2,4,6):
                dt=lf[lag]-lf[0]
                rot=np.exp(1j*hand*modes*dt/math.tan(math.radians(pitch)))
                ww=w[lag:]*w[:-lag]
                prod=v[lag:]*v[:-lag].conj()
                # Improvement in signed covariance over a stationary ridge.
                term=np.real(np.sum(prod*(rot-.65),axis=1))
                energy+=np.sum(ww*term)
                norm+=np.sum(ww)
            val=math.sqrt(max(energy/(norm+1.e-6),0))
            bc=max(val,bc)
        if pitch<25: tight=max(tight,bc)
        else: opened=max(opened,bc)
    return max(tight,opened),tight,opened

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-50,50)))

def clip(z):
    return np.clip(z,0,1)

def sample(a,x,y):
    x=np.clip(x,0,126.9999)
    y=np.clip(y,0,126.9999)
    ix=x.astype(int)
    iy=y.astype(int)
    dx=x-ix
    dy=y-iy
    return ((1-dx)*a[iy,ix]+dx*a[iy,ix+1])*(1-dy)+((1-dx)*a[iy+1,ix]+dx*a[iy+1,ix+1])*dy

def rings(a,x,y):
    return sample(a,x+rr*cs,y+rr*sn)

def outlines(ray,levels):
    # First sustained crossing of each surface-brightness contour.
    stay=np.maximum(ray,np.vstack((ray[1:],ray[-1:])))
    stay[:2]=np.maximum(stay[:2],stay[2])
    low=stay[None,:,:]<levels[:,None,None]
    low[:,-1,:]=True
    k=np.maximum(np.argmax(low,axis=1),1)
    j=np.arange(len(cs))
    h=stay[k-1,j]
    l=stay[k,j]
    r=(k-1+np.clip((h-levels[:,None])/np.maximum(h-l,.00001),0,1))*.5
    r=np.maximum(r,.8)
    # Angular median rejects single rays that hit a field source.
    return np.median(np.stack([np.roll(r,i,axis=1) for i in (-2,-1,0,1,2)]),axis=0)

def ellipse(r):
    # The moment quadrupole has argument twice the position angle.
    # Use a polygon area moment for the axes. The radius-squared Jacobian
    # matters for needle-like galaxies, which occupy few circular rays.
    rsym=(r+np.roll(r,64,axis=1))/2
    lim=np.quantile(rsym,.96,axis=1)
    power=np.minimum(rsym,1.10*lim[:,None])**4
    ecc=np.mean(power*second,axis=1)/np.maximum(np.mean(power,axis=1),.01)
    e=np.minimum(np.abs(ecc),.99)
    ang=np.angle(ecc)/2
    q=np.sqrt((1-e)/(1+e))
    model=1/np.sqrt(np.cos(theta-ang[:,None])**2+(np.sin(theta-ang[:,None])/q[:,None])**2)
    a=np.median(rsym/model,axis=1)
    lograd=np.log(r)
    err=lograd-np.log(model*a[:,None])
    odd=(lograd-np.roll(lograd,64,axis=1))/2
    return a,q,ang,err,odd

def companion(s1,s2,s4,bands,cx,cy,peak,a,q,pa,bgnoise):
    # Maxima must have their own prominence, resolved wings, and a plausible
    # distance from the target in units of the two galaxies' visible sizes.
    peaks=s2>max(.085*peak,4*bgnoise)
    small=s1>max(.18*peak,5*bgnoise)
    for di,dj in ((1,0),(-1,0),(0,-1),(0,1),(1,1),(-1,1),(-1,-1),(1,-1)):
        peaks &= s2>=np.roll(s2,(di,dj),axis=(0,1))
        small &= s1>=np.roll(s1,(di,dj),axis=(0,1))
    peaks |= small
    dd=(X-cx)**2+(Y-cy)**2
    peaks &= (dd>4**2)&(D<62**2)
    yi,xi=np.nonzero(peaks)
    if not len(xi): return 0.
    # Cap work for densely populated fields.
    order=np.argsort(s2[yi,xi]/(1+dd[yi,xi]/400))[::-1][:18]
    best=0.
    for idx in order:
        x=xi[idx]; y=yi[idx]
        dist=math.sqrt(dd[y,x]); top=s1[y,x]
        if top < .10*peak: continue
        cp=math.cos(pa); sp=math.sin(pa)
        scale=a*dist/max(math.hypot((x-cx)*cp+(y-cy)*sp,((y-cy)*cp-(x-cx)*sp)/q),.1)
        path=np.linspace(.04,.96,24)
        line=sample(s1,x+(cx-x)*path,y+(cy-y)*path)
        valley=max(float(line.min()),0.)
        prom=(top-valley)/max(top,.01)
        if prom < .08: continue
        radii=np.arange(0,22,.65)[:,None]
        cut=sample(s1,x+radii*cs[::8],y+radii*sn[::8])
        lev2=max(.45*top,.5*valley+.35*top)
        below=cut<lev2; below[-1]=True
        rhalf=np.median(np.maximum(1,np.argmax(below,axis=0))-.5)*.65
        # Flux surviving a second broader smoothing discriminates resolved
        # companions from foreground stars and isolated star-forming knots.
        wing=(s4[y,x]-.25*valley)/(max(.01,top-.25*valley))
        resolved=sigmoid((rhalf-2.8)/.6+(wing-.32)*5)
        comparable=sigmoid((math.log(max(.01,top/peak))-math.log(.29))/.38)
        # Bright subknots deep inside the primary require stronger evidence.
        distinct=sigmoid((prom-.32)/.10)
        proximity=sigmoid((1.45-dist/(scale+2.8*rhalf+1))/.25)
        closecenter=sigmoid((41-dist)/9)
        blend=sigmoid((valley/max(min(top,peak),.01)-.045)/.035)
        # Apparent area is important: a second pinprick in a big elliptical
        # is rarely another galaxy of comparable luminosity. The outer
        # part of the candidate must also have a plausible stellar colour.
        def colourpatch(ch):
            p=sample(ch,x+radii*cs[::8],y+radii*sn[::8])
            return np.maximum(np.median(p[1:5]),0)
        cr=colourpatch(bands[0]); cb=colourpatch(bands[2]); cg=colourpatch(bands[1])
        hue=(cr-cb)/(cr+cb+2*bgnoise)
        colorscore=sigmoid((.52-abs(hue))/.075)*sigmoid((.30-abs((cr+cb-2*cg)/(cr+cb+2*cg+.03)))/.055)
        sizefrac=rhalf/(a*math.sqrt(q)+.1)
        relative=sigmoid((math.log(max(.005,(top/peak)*(4*sizefrac)**2))-math.log(.16))/.48)
        bound=sigmoid((1.05-dist/(scale+2.6*rhalf+1))/.16)
        # Two embedded nuclei can be too close for individual outer wings.
        # For that case look for an extended second colour-similar maximum
        # and connected light at a substantial fraction of both peaks.
        primary=np.maximum(0,np.median(bands[:,max(0,int(cy)-1):int(cy)+3,max(0,int(cx)-1):int(cx)+3],axis=(1,2)))
        phue=(primary[0]-primary[2])/(primary[0]+primary[2]+2*bgnoise)
        similar=sigmoid((.31-abs(phue-hue))/.09)
        bridge=sigmoid((valley/max(.02,min(peak,top))-.145)/.045)
        embedded=sigmoid((1.0-dist/(scale+1.5*rhalf))/.13)*bridge*similar*comparable*colorscore
        blended=embedded*sigmoid((prom-.22)/.075)*sigmoid((rhalf-1.9)/.5)*sigmoid((top/peak-.40)/.11)
        evidence=max(resolved*comparable*distinct*proximity*closecenter*(.39+.61*blend)*relative*bound*colorscore,blended)
        # A shared envelope is broad on both sides of the saddle, not just
        # a single bright pixel on the line of centres. Test at the saddle
        # and compare against light on the opposite side of the primary.
        t=float(path[np.argmin(line)]); bx=x+(cx-x)*t; by=y+(cy-y)*t
        nx=(y-cy)/dist; ny=(cx-x)/dist
        offsets=np.linspace(-1,1,9)*min(4.,.5*rhalf+1)
        across=np.maximum(sample(s2,bx+offsets*nx,by+offsets*ny),0)
        opposite=np.maximum(sample(s2,2*cx-bx+offsets*nx,2*cy-by+offsets*ny),0)
        shared=float(np.median(across)); excess=float(np.median(across-opposite))
        connect=sigmoid((shared/max(.035,min(peak,top))-.075)/.027)*sigmoid((shared/bgnoise-3)/1.2)
        tidal=sigmoid((excess-.008*peak)/(.012*peak+bgnoise))
        evidence*=.46+.36*connect+.18*tidal
        # A projected source on a bright round halo can satisfy a saddle
        # test merely because the halo is part of its sky background. On
        # subtracting the diametrically opposite patch, an actual companion
        # should still have wings. Test both extent and radial falloff.
        mirror=sample(s1,2*cx-x-radii*cs[::8],2*cy-y-radii*sn[::8])
        residual=cut-mirror
        profile=np.quantile(residual,.42,axis=1)
        amp=max(float(profile[0]),.03)
        rad=radii[:,0]
        corona=float(np.mean(profile[(rad>=4)&(rad<=8)]))/amp
        twonuc=clip((corona-.04)/.23)*sigmoid((amp/peak-.16)/.08)
        evidence*=.55+.45*twonuc
        best=max(best,evidence)
    return best

def spiralwave(z,w,m):
    # Correlate ring Fourier harmonics with both handednesses of log spirals.
    # Remove an unwinding (constant-phase) component due to inclination.
    maxsignal=0.; tight=0.; loose=0.
    for lo,hi in ((.19,.67),(.30,.96),(.44,1.27)):
        use=(fr>lo)&(fr<hi)
        t=lf[use]; v=z[use]; wt=w[use]; sw=wt.sum()+1.e-8
        v=v-np.sum(v*wt)/sw
        for pitch in (10,14,19,26,36,50):
            f=m/math.tan(math.radians(pitch))
            template=np.exp(1j*f*t)
            value=max(abs(np.sum(v*wt*template)),abs(np.sum(v*wt*template.conj())))/sw
            if pitch<=19: tight=max(tight,value)
            else: loose=max(loose,value)
            maxsignal=max(maxsignal,value)
    return maxsignal,tight,loose

def classify(im):
    # Survey sky estimated in the periphery, independently for each band.
    sky=np.median(im[:,::3,::3][:,skyzone[::3,::3]],axis=1)
    bands=im-sky[:,None,None]
    gray=.28*bands[0]+.44*bands[1]+.28*bands[2]
    spec=np.fft.rfft2(gray)
    s1,s2,s4=[np.fft.irfft2(spec*k,s=(128,128)) for k in kernels]
    base=np.median(s1[::2,::2][skyzone[::2,::2]])
    noise=max(.004,float(np.median(np.abs(s1[::2,::2][skyzone[::2,::2]]-base)))*1.48)
    s1-=base; s2-=base; s4-=base
    k=np.argmax(np.maximum(s2,0)*np.exp(-D/60)-.000001*D)
    cy,cx=divmod(int(k),128)
    # Subpixel centroid of the nucleus, not the centroid of the entire frame.
    for width in (3.,3.,4.):
        dd=(X-cx)**2+(Y-cy)**2
        w=np.exp(-dd/(2*width**2))*np.maximum(s1-.13*max(s1[int(cy),int(cx)],.02),0)**2
        sw=w.sum()+1.e-8
        cx=(np.sum(w*X)+1.e-8*cx)/sw; cy=(np.sum(w*Y)+1.e-8*cy)/sw
    peak=max(float(sample(s1,np.array(cx),np.array(cy))),noise*3,.015)
    ray=rings(s1,cx,cy)
    levels=np.maximum(lev*peak,2.4*noise)
    r=outlines(ray,levels)
    aa,qq,pa,err,odd=ellipse(r)
    mid=4
    outer=int(np.argmin(abs(levels/peak-max(.085,3*noise/peak))))
    outer=max(mid,outer)
    # Outer axis and flattening from adjacent isophotes, avoiding the sky.
    os=slice(max(mid,outer-1),outer+1)
    q=float(np.mean(qq[os])); a=max(4.,aa[outer]); phi=.5*np.angle(np.mean((1-qq[os])*np.exp(2j*pa[os])))
    co=math.cos(phi); si=math.sin(phi)
    # Deproject and unwrap the disk. Radius is a fraction of the visible
    # semimajor axis. Clamp foreshortening near edge-on (finite thickness).
    u=fr[:,None]*a*cs; v=fr[:,None]*a*max(q,.16)*sn
    xx=cx+co*u-si*v; yy=cy+si*u+co*v
    I=sample(s1,xx,yy)
    med=np.median(I,axis=1)
    den=np.maximum(med+1.8*noise,.035*peak)
    contrast=np.clip((I-med[:,None])/den[:,None],-.95,1.4)
    four=np.fft.rfft(contrast,axis=1)*2/len(cs)
    weight=clip((med-1.8*noise)/(.065*peak+noise))
    use=(fr>=.32)&(fr<=1.03)
    def average(v):
        return float(np.sum(v[use]*weight[use])/(np.sum(weight[use])+1.e-7))
    h2,t2,l2=spiralwave(four[:,2],weight,2)
    h3,t3,l3=spiralwave(four[:,3],weight,3)
    h4,t4,l4=spiralwave(four[:,4],weight,4)
    spiral=max(h2,h3,.9*h4)
    # Coherent nonelliptical structure, and unpaired/lopsided structure.
    texture=average(np.sqrt(np.sum(abs(four[:,3:10])**2,axis=1)))
    lop=average(abs(four[:,1])); asym=average(np.sqrt(np.sum(abs(four[:,1:10:2])**2,axis=1)))
    eccerr=np.median(np.abs(err),axis=1)
    asymiso=np.mean(np.abs(odd),axis=1)
    iso=float(np.mean(eccerr[mid:outer+1]))
    lopsided=float(np.mean(asymiso[mid:outer+1]))
    outerodd=float(np.max(asymiso[os]))
    # Disk colour compared with the nucleus. Use medians on rings to
    # suppress unrelated sky objects; include a noise pedestal in ratios.
    colors=[]
    for b in (0,2):
        col=sample(bands[b]-base,xx,yy)
        colors.append(np.maximum(np.median(col,axis=1),0))
    red,blue=colors
    # Star-forming ridges have much higher contrast in the blue band.
    # Filter the band at matched seeing before computing colour structure.
    bfield=np.fft.irfft2(np.fft.rfft2(bands[2])*kernels[0],s=(128,128))
    bp=sample(bfield,xx,yy); bmed=np.median(bp,axis=1)
    bnoise=max(noise,float(np.median(np.abs(bfield[::2,::2][skyzone[::2,::2]]-np.median(bfield[::2,::2][skyzone[::2,::2]]))))*1.48)
    bcontrast=np.clip((bp-bmed[:,None])/np.maximum(bmed[:,None]+2.8*bnoise,.04*peak),-.85,1.1)
    winding,windtight,windopen=ridgefield(.6*contrast+.4*bcontrast,med,noise)
    def colour(l,h):
        sel=(fr>=l)&(fr<=h)
        p=np.mean(red[sel]); b=np.mean(blue[sel]); return (p-b)/(p+b+3*noise)
    corecol=colour(.07,.22); diskcol=colour(.52,.92); grad=corecol-diskcol
    # Compare inner and outer shape after accounting for projection. Bars
    # are elongated in observed light as well as relative to the outer disk.
    bi=np.arange(1,max(3,mid+2))
    phase=pa[bi]-phi
    barq=qq[bi]
    tr=(np.cos(phase)**2+(barq*np.sin(phase))**2)*q+(np.sin(phase)**2+(barq*np.cos(phase))**2)/q
    ratio=2*barq/(tr+np.sqrt(np.maximum(tr**2-4*barq**2,0)))
    tiltpen=(1-q)*np.sin(phase)**2
    barext=aa[bi]/a
    bar=float(np.max((1-ratio)*clip((.95-barq)/.32)*clip((barext-.10)/.18)*clip((.78-barext)/.22)*(1-.8*tiltpen)))
    barstable=abs(np.mean((1-qq[1:mid+1])*np.exp(2j*pa[1:mid+1])))
    bar*=.75+.25*clip(barstable/.28)
    # A bar remains straight through several adjacent rings of the middle
    # disk; the quadrupole drops or changes angle beyond its ends. A
    # one-sided knot can give the same ellipticity, but not this pattern.
    m2=four[:,2]
    barlight=0.
    for low,high in ((.10,.34),(.17,.46),(.24,.60)):
        bsel=(fr>low)&(fr<high); esel=(fr>high*1.1)&(fr<min(1.2,high*2.1))
        bvec=np.mean(m2[bsel]); end=np.mean(m2[esel]); be=abs(bvec)
        stay=be/(np.mean(abs(m2[bsel]))+.015)
        change=abs(end-bvec)
        barlight=max(barlight,min(be,change*1.25)*stay*clip((barstable-.06)/.21))
    barvalue=bar+.42*min(.4,barlight)
    pair=companion(s1,s2,s4,bands,cx,cy,peak,a,q,phi,noise)
    # Apparent smooth shape and disk thicknesses. The galaxy has to be
    # resolved before structure and bulge tests become informative.
    qm=float(np.mean(qq[2:mid+1])); qi=float(np.mean(qq[:3])); qo=float(np.mean(qq[os]))
    qvis=.5*qm+.35*q+.15*qi
    half=aa[2]; size=a*math.sqrt(q)
    # Significance of nonelliptical structure in the bright body. Adjacent
    # angular Fourier modes in the faint outskirts also contain sky noise.
    # Measure a noise-corrected amplitude with roughly independent seeing
    # elements and give less credence to one deviant ring than several.
    nel=np.maximum(5,math.tau*a*fr*math.sqrt((1+q*q)/2)/3.5)
    noisepow=4/nel*(noise/den)**2
    harmonics=np.sqrt(np.maximum(np.sum(abs(four[:,3:10])**2,axis=1)-7*noisepow,0))
    body=(fr>.24)&(fr<.86)&(med>2.5*noise)
    structured=float(np.median(harmonics[body])) if np.any(body) else 0.
    # Face-on spirals frequently show a red core in a blue disk; a smooth
    # spheroid can have a mild colour gradient from its metallicity alone.
    diskcolour=clip((grad-.08)/.23)*clip((.16-diskcol)/.24)
    details=1.65*spiral+.83*structured+.35*iso
    resolved=math.log(max(size,2)/15)
    diskev=(details-.22)/.060+.85*resolved+1.6*diskcolour+.35*clip((-diskcol-.10)/.16)
    diskev+=2.8*clip((bar-.21)/.32)
    # An isophotally smooth spheroid stays smooth even when well resolved;
    # blue late-type disks often have visible radial breaks or rings.
    def radialmean(t,u):
        return max(0.,float(np.median(med[(fr>=t)&(fr<=u)])))
    p0=radialmean(.08,.16); p1=radialmean(.23,.36); p2=radialmean(.48,.61); p3=radialmean(.73,.89)
    shoulder=(p2+.02*peak)/(p1+.02*peak)-(p1+.02*peak)/(p0+.02*peak)
    diskev+=1.15*clip((shoulder-.025)/.19)*clip((grad-.015)/.15)
    quiet=clip((.11-structured)/.065)*clip((.085-spiral)/.05)*clip((.20-bar)/.1)
    diskev-=1.8*quiet
    # Gently tapered smooth ellipticals have centrally concentrated light
    # and broad wings. A nearly exponential disk has relatively more light
    # throughout its bright body: the 55%-peak radius is about 1/4 of the
    # 8%-peak radius in a pure exponential, before seeing convolution.
    bodyradius=aa[2]*math.sqrt(qq[2]); wingradius=max(size,1)
    bodyfrac=bodyradius/wingradius
    # Seeing-limited small objects can mimic low concentration. Do not
    # interpret a flat core there as evidence of a disk.
    bigbody=clip((bodyradius-3.8)/4.5)
    broad=bigbody*clip((bodyfrac-.20)/.14)
    # A detectable colour/brightness break from bulge to disk can identify
    # tightly wrapped arms even when the angular modulation is weak.
    radialred=colour(.21,.42)-colour(.69,1.0)
    twocol=max(0,grad-.035,.95*radialred-.035)
    diskpop=clip((.14-diskcol)/.22)
    photodisk=1.25*math.log(max(bodyradius,1.5)/5.5)+1.20*broad+3.4*twocol+1.20*diskpop+1.0*clip((shoulder-.015)/.20)-2.1
    # Calibrate winding against off-object sky from this observation, with
    # identical sampling, radial normalization, and both fields of view.
    noiseharm=[]
    for ox,oy in ((25,25),(102,25),(25,102),(102,102)):
        px=(xx-cx+ox)%127; py=(yy-cy+oy)%127
        patch=sample(s1,px,py); patch-=np.median(patch,axis=1)[:,None]
        patch=np.clip(patch,-2.5*noise,2.5*noise)/den[:,None]
        bpatch=sample(bfield,px,py); bpatch-=np.median(bpatch,axis=1)[:,None]
        bpatch=np.clip(bpatch,-2.5*bnoise,2.5*bnoise)/np.maximum(bmed[:,None]+2.8*bnoise,.04*peak)
        noiseharm.append(ridgefield(.60*patch+.40*bpatch,med,noise)[0])
    skywave=np.median(noiseharm)
    signal=math.sqrt(max(0,winding**2-1.6*skywave**2))
    armtest=(signal-.043)/.024+.45*diskpop+.40*broad+.45*resolved-.8
    # In a single smooth body, log radii vary smoothly with the log of the
    # isophote. Shoulders in this curve indicate a second (disk) component.
    obsr=aa*np.sqrt(qq)
    loggap=np.diff(np.log(obsr))/np.maximum(.03,np.diff(np.log(-np.log(lev))))
    smoothrad=float(np.median(abs(np.diff(loggap[1:outer])))) if outer>3 else 0.
    faintdisk=photodisk-.75*clip((.15-smoothrad)/.11)-.65*clip((.11-structured)/.085)-.65*clip((.14-twocol)/.12)
    diskev=max(diskev,faintdisk,armtest)
    diskev+=.58*clip((barlight-.11)/.18)
    # Measure angular pitch locally with a structure tensor on an unwrapped
    # disk. A logarithmic arm in (angle, log radius) is a straight ridge.
    # Remove the nonspiral background varying slowly in radius (ellipse,
    # bar, field stars) before taking derivatives.
    polar=.60*contrast+.40*bcontrast
    jr=np.arange(len(fr)); dyadic=jr[:,None]-jr
    def smoothradial(p,s):
        k=np.exp(-.5*(dyadic/s)**2); return (k@p)/k.sum(axis=1)[:,None]
    decomposition=np.fft.rfft(polar,axis=1)
    decomposition[:,:2]=0
    freqtheta=np.arange(decomposition.shape[1]); stokes=[]
    for radiusblur,angleblur in ((.8,.09),(1.4,.16)):
        pf=smoothradial(decomposition,radiusblur)*np.exp(-.5*(freqtheta*angleblur)**2)
        feature=np.fft.irfft(pf-smoothradial(pf,4.0),n=len(cs),axis=1)
        dr=np.gradient(feature,axis=0)/(lf[1]-lf[0]); dt=(np.roll(feature,-1,axis=1)-np.roll(feature,1,axis=1))/(2*math.tau/len(cs))
        select=(fr>.27)&(fr<1.1)&(med>2.8*noise)
        ww=weight[select,None]*np.minimum(3.,np.abs(feature[select])/(noise/den[select,None]+.02))
        drr=dr[select]; dtt=dt[select]; tot=np.sum(ww*(drr**2+dtt**2))+.0001
        xxmom=np.sum(ww*(drr**2-dtt**2))/tot
        xymom=2*np.sum(ww*drr*dtt)/tot
        stokes.append((xxmom,xymom))
    mom=np.mean(stokes,axis=0)
    chirality=abs(mom[1]); pitchang=math.atan2(chirality,mom[0])/2
    genuine=clip((chirality-.09)/.29)*clip((winding-skywave)/.047)
    diskev+=1.75*genuine

    # Tidal/irregular evidence: displacements of isophotes, asymmetric
    # arms, or a very blue and lumpy object lacking regular winding.
    disorder=(.45*lop+.35*asym+.65*lopsided+.24*outerodd)
    irregular=(disorder-.30)/.075+.75*clip((-diskcol-.04)/.15)-1.0*clip((spiral-.08)/.14)
    irregular+=.7*clip((lop-.15)/.18)-.5*clip((grad-.05)/.2)
    # The centre is much less likely to contain two overlapping objects
    # when even its outer isophotes remain point-symmetric and undistorted.
    merger=(pair-.46)/.12+1.5*clip((disorder-.08)/.27)-.35*quiet
    if merger>0 and pair>.22: return 1
    # A companion too close to separate photometrically often produces a
    # strongly lopsided, red, luminous double core. Do not interpret a blue
    # off-centre star-forming knot in the same way.
    if irregular>1 and pair>.12 and corecol>.08 and p1>.46*peak: return 1
    # Distinguish a thin disk with a bulge from a cigar-shaped smooth body
    # by looking explicitly at the light distribution along the major axis.
    # The inner shape includes a seeing correction, more important in small
    # objects. Avoid mistaking a compact round nucleus for a face-on galaxy.
    flat=float(np.min(qq[mid:outer+1])); lineq=.5*(flat+qo)
    halfq=qq[2]
    deconv=lambda t,z: math.sqrt(max((t*z)**2-2.0,1.0)/max(t*t-2.0,2.))
    qinner=deconv(half,halfq)
    slender=(.33-lineq)/.055+.80*clip((qm-lineq-.035)/.17)+.55*clip((a-18)/22)+.35*clip((-diskcol-.06)/.18)
    slender+=.6*clip((.27-qinner)/.13)
    # A disky isophote is pointed relative to an ellipse, with increased
    # flux in the axial directions (positive fourth harmonic in radius).
    disky=float(np.mean(2*np.mean(err[os]*np.cos(4*(theta-phi)),axis=1)))
    slender+=.65*clip((disky-.015)/.075)-.5*clip((.05-iso)/.04)*clip((.09-(qm-lineq))/.065)
    # Determine whether the body thickens at the nucleus. Compare vertical
    # widths above the local half-maximum on and off the central sightline.
    # A single spheroidal ellipse narrows more gently than a bulged disk.
    ax=a*np.array([-.61,-.45,-.30,-.13,0,.13,.30,.45,.61])[:,None]
    vert=np.linspace(-1,1,101)[None,:]*max(5,min(38,a*max(.38,1.6*q)))
    cuts=sample(s1,cx+ax*co-vert*si,cy+ax*si+vert*co)
    pk=np.max(cuts[:,36:65],axis=1)
    hwidth=np.sum(cuts>np.maximum(pk[:,None]*.5,2*noise),axis=1)*vert[0,-1]/100
    thick=(hwidth[3:6].mean()+.5)/(np.median(hwidth[[1,2,6,7]])+.5)
    radialpeak=pk[4]/max(.01,.5*(pk[1]+pk[-2]))
    # Non-elliptical pointed extensions: estimate disk aspect separately
    # from the radial Fourier shape. The 98th percentile is robust to a
    # single extended ray but still sees a very narrow major axis.
    ar=np.mean(r[os],axis=0)
    opta=np.mean(np.sort(.5*(ar+np.roll(ar,64)))[-8:])
    optb=np.median(ar[np.abs(np.sin(theta-phi))>.95]); pointq=optb/(opta+.001)
    edgekin=clip((.44-pointq)/.13)*( .7*clip((thick-1.12)/.32)+.7*clip((.37-pointq)/.12)+.85*clip((qm-pointq-.03)/.18) )
    slender+=edgekin
    # Project a high-pass image onto candidate disk midplanes through the
    # centre. Look for bilateral light extensions away from the bright bulge
    # that would not be resolved by an isophote-based thickness test alone.
    seg=np.linspace(.48,1.42,18)[None,:]*max(8,a)
    aoff=np.linspace(-.26,.26,9)[:,None]+phi
    dx=seg*np.cos(aoff); dy=seg*np.sin(aoff)
    transverse=np.maximum(2.,seg*.23)
    def lineflux(sgn,offset):
        return sample(s1,cx+sgn*dx-offset*transverse*np.sin(aoff),cy+sgn*dy+offset*transverse*np.cos(aoff))
    sidep=lineflux(1,0); sidem=lineflux(-1,0)
    along=np.minimum(sidep,sidem)
    away=(lineflux(1,1)+lineflux(1,-1)+lineflux(-1,1)+lineflux(-1,-1))*.25
    diskcontrast=np.clip((along-away)/(np.maximum(along,.03*peak)+2*noise),0,.95)
    narrow=float(np.max(np.mean(diskcontrast*clip((along-2*noise)/(.06*peak)),axis=1)))
    # Red regular old disk with tapered ends. Strong arms and position-angle
    # twists are less likely in an edge-on projection than a face-on disk.
    redold=clip((corecol+.01)/.19)
    needle=clip((.55-min(pointq,lineq,qvis))/.15)*clip((narrow-.035)/.15)
    slender+=needle*(1.6+.9*redold)+.8*clip((.45-qvis)/.12)*redold*clip((radialpeak-2)/2)
    if slender>0 and irregular<max(1.6,slender+1):
        bulge=.60*(qinner-lineq-.055)/.11+4.6*(corecol-.09)+2.2*(grad-.09)+.45*(qinner-.26)/.13
        bulge+=1.6*np.clip(math.log(max(.5,thick)/1.15),-.65,.85)+.60*np.clip(math.log(max(.5,radialpeak)/2.9),-1,1)
        bulge-=.85*clip((.045-corecol)/.13)
        # A thin exponential disk has a very similar on-axis and off-axis
        # colour. A localized central excess that is *both* bright and red
        # is better evidence for a bulge than red colour by itself.
        baraxis=seg*.39
        def diskchannel(b,sgn):
            dat=sample(b,cx+sgn*baraxis*co,cy+sgn*baraxis*si)
            return np.maximum(dat,0).mean()
        rval=.5*(diskchannel(bands[0],1)+diskchannel(bands[0],-1)); bval=.5*(diskchannel(bands[2],1)+diskchannel(bands[2],-1))
        linecolor=(rval-bval)/(rval+bval+noise)
        localgrad=corecol-linecolor
        bulge+=.75*clip((localgrad-.025)/.13)+.70*clip((corecol-.12)/.12)*clip((radialpeak-2.2)/2.0)
        return 9 if bulge>0 else 8
    # Low-contrast irregulars can have a nearly elliptical bright body.
    # Coherent off-centre isophotes and blue emission without a central
    # red bulge or the winding symmetry of a spiral are telltale signs.
    blueall=clip((-.045-corecol)/.18)*clip((.05-diskcol)/.17)
    ia=slice(1,mid+1)
    bodyodd=float(np.mean(asymiso[ia])); outershift=float(np.mean(abs(four[(fr>.50)&(fr<.97),1])))
    # Is the light distributed about a distinct bright nucleus? A flat
    # blue peak away from the centre of the outer body is common in
    # disturbed objects, but not in most grand-design spirals.
    nc=float(np.mean(med[(fr>.22)&(fr<.38)]))/peak
    imbalance=clip((bodyodd-.045)/.12)*clip((nc-.42)/.28)
    irregular+=1.65*blueall+.65*clip((outerodd-.13)/.16)-.55*clip((bar-.20)/.25)+1.4*imbalance
    irregular-=1.3*genuine+.6*clip((grad-.09)/.18)
    # Smooth but *very* blue objects with no distinct nucleus still favour
    # an irregular if the unpaired disk light dominates the winding signal.
    irregular+=1.0*blueall*clip((outershift-spiral)/.12)*clip((nc-.42)/.24)
    # Use the shift of successive isophote centroids, not just a large
    # scatter of irregular rays (also possible in quite normal spirals).
    # A normalized first moment of area is 2/3 times int r^3 dir / int r^2.
    radii2=np.minimum(r,1.25*np.quantile(r,.95,axis=1)[:,None])
    drift=2/3*np.mean(radii2**3*(cs+1j*sn),axis=1)/np.mean(radii2**2,axis=1)
    centralshift=float(np.median(abs(drift[2:6])/(obsr[2:6]+1)))
    outershift2=float(np.median(abs(drift[mid:outer+1]-drift[1])/(obsr[mid:outer+1]+1)))
    irregular+=1.65*clip((outershift2-.075)/.18)+1.0*clip((centralshift-.075)/.16)*clip((.11-corecol)/.19)
    irregular+=1.65*blueall*clip((.1-grad)/.17)-.4*clip((lineq-.48)/.25)*genuine
    if irregular>0 and (irregular>.85 or diskev<.6 or lop>1.8*spiral or blueall>.48): return 0
    if diskev<0:
        if qvis>.79: return 2
        # PSF bias is strongest for compact smooth objects. Use the
        # deblurred half-light shape for the cigar vs intermediate test.
        qshape=.32*qvis+.38*qinner+.30*pointq
        if qshape<.43: return 4
        return 3
    # Inner straightness, measured before deprojection. Bars with a bright
    # central bulge show their strongest elongation at intermediate levels.
    oval=(1-qq[2:7])*np.exp(2j*pa[2:7]); straight=abs(oval.sum())/(np.sum(abs(oval))+.01)
    qdrop=qq[0]-np.min(qq[2:7]); bscore=barvalue+.17*clip((qdrop-.065)/.22)*straight
    # Test the putative bar in its own image plane. A bar has a straight,
    # bilateral red ridge through the middle, whereas an elliptical bulge
    # with two arms begins to twist and is less elongated at high intensity.
    ib=int(np.argmin(qq[2:6]))+2
    orient=pa[ib]; bangle=abs(np.sum((1-qq[1:ib+1])*np.exp(2j*(pa[1:ib+1]-orient)))/(np.sum(1-qq[1:ib+1])+.001))
    binner=float(np.mean(1-qq[:3])); blength=aa[ib]/max(a,.1)
    strength=clip((binner-.13)/.23)*clip((blength-.11)/.2)+.6*clip((barlight-.13)/.2)
    bscore+=.07*strength-.12*clip((.27-binner)/.17)-.13*clip((.88-bangle)/.20)
    # Follow individual Fourier ridges in overlapping radial windows. The
    # effective pitch is the amount of rotation per log-radius interval,
    # not just the amplitude of the best-matching spiral template.
    phasevotes=[]; votew=[]
    for m in (2,3,4):
        coeff=np.fft.rfft(.5*contrast+.5*bcontrast,axis=1)[:,m]/(len(cs)*.5)
        for l,h in ((.27,.52),(.37,.72),(.51,.97),(.66,1.22)):
            j=(fr>l)&(fr<h)
            v=coeff[j]; step=3
            cross=v[step:]*v[:-step].conj()
            amp=float(np.median(abs(v)))
            cov=np.sum(cross)/(np.sum(abs(cross))+.00001)
            sweep=abs(np.angle(cov)); signed=abs(np.imag(cov))
            angle=math.atan2(m*step*(lf[1]-lf[0]),max(.03,sweep))
            w=amp**2*abs(cov)**3*clip((signed-.10)/.33)*clip((np.median(med[j])/noise-2.0)/4)
            phasevotes.append(np.clip((angle-math.radians(21))/math.radians(15),-1,1.3)); votew.append(w)
    phaseweight=sum(votew); localpitch=np.sum(np.array(phasevotes)*votew)/(phaseweight+.004)
    # Interarm gaps and sharp outward rises in the low-surface-brightness
    # disk make an open arm conspicuous even if very short or asymmetric.
    soft=smoothradial(.65*I+.35*bp,1.0)
    smoothang=np.fft.irfft(np.fft.rfft(soft,axis=1)*np.exp(-.5*(freqtheta*.095)**2),n=len(cs),axis=1)
    outerdisk=(fr[4:]>.44)&(fr[4:]<1.38)
    rise=(smoothang[4:]-smoothang[:-4]-.65*noise)/np.maximum(den[4:,None],.05*peak)
    armrise=float(np.quantile(np.maximum(rise[outerdisk],0),.93))
    z=(fr>.55)&(fr<1.15); rel=np.maximum(np.median(soft[z],axis=1),3*noise)
    armcontrast=float(np.median((np.quantile(soft[z],.80,axis=1)-np.quantile(soft[z],.20,axis=1)-1.6*noise)/rel))
    barpen=.11*clip((armrise-.04)/.15)*clip((.37-binner)/.22)+.06*clip((localpitch-.25)/.75)*clip((.40-binner)/.18)
    if bscore-barpen>.33 and barstable>.10 and (barlight>.07 or bar>.36 or qdrop>.16): return 5
    wound=max(t2,t3,.85*t4); openarm=max(l2,l3,.85*l4)
    loose=(openarm-wound)/.060+.66*(openarm-.083)/.070+.45*(structured-.12)/.12+.50*(grad-.12)/.18
    loose+=.80*(windopen-windtight)/.035+.65*clip((lop-.13)/.16)+.7*clip((.27-bodyfrac)/.12)
    loose+=genuine*np.clip((pitchang-math.radians(22))/math.radians(12),-1.45,1.45)
    # High arm-interarm contrast and a nearly detached core can be apparent
    # even when the outer winding is too broken for the phase correlation.
    ring2=abs(four[:,2]); armzone=(fr>.47)&(fr<1.13)
    knot=(float(np.median(ring2[armzone]))+structured)*clip((.38-bodyfrac)/.14)
    loose+=.85*clip((knot-.09)/.17)-.8*clip((broad-.45)/.45)
    outerevidence=1.3*localpitch+1.6*clip((armrise-.015)/.12)+1.0*clip((armcontrast-.25)/.43)
    # A low absolute surface-brightness disk with a small bright central
    # component is more likely to expose isolated open arm segments than
    # the high-surface-brightness, uniformly populated tight-arm pattern.
    shoulderlight=radialmean(.35,.60)/max(.015,p0)
    gap=clip((.42-shoulderlight)/.20)*clip((.35-bodyfrac)/.13)
    loose+=outerevidence+.85*gap
    return 7 if loose>0 else 6

def predict(x):
    return np.array([classify(im) for im in x],dtype=np.int64)
