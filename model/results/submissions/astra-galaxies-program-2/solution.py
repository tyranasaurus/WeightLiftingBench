import numpy as np
import math

# Image-plane and angular quadrature grids, independent of the observations.
Y,X=np.mgrid[:128,:128]
F2=np.fft.fftfreq(128)[:,None]**2+np.fft.rfftfreq(128)[None,:]**2

def lowpass(s):
    return np.exp(-2*(np.pi*s)**2*F2)

seeing=lowpass(1.35)
corescale=lowpass(2.8)
nb=128
ang=np.arange(nb)*(2*np.pi/nb)
ux=np.cos(ang)
uy=np.sin(ang)
phase2=np.exp(2j*ang)
rad=np.arange(0,81,.5)[:,None]
levels=np.array([.78,.60,.43,.30,.20,.13,.085,.055])
logR=np.linspace(np.log(.075),np.log(1.45),52)
unitR=np.exp(logR)
edge=(abs(X-63.5)>52)|(abs(Y-63.5)>52)

def restrict(x,a,b):
    return min(b,max(a,x))

def ramp(x,a,b):
    return restrict((x-a)/(b-a),0.,1.)

def sample(im,x,y):
    xx=np.clip(x,0,126.99999)
    yy=np.clip(y,0,126.99999)
    i=xx.astype(int)
    j=yy.astype(int)
    a=xx-i
    b=yy-j
    return (1-b)*(im[j,i]*(1-a)+im[j,i+1]*a)+b*(im[j+1,i]*(1-a)+im[j+1,i+1]*a)

def smoothang(x):
    return .5*x+.25*(np.roll(x,1,-1)+np.roll(x,-1,-1))

def iso(rays,threshold):
    # Interpolated first exit of each ray from the central isophote. A brief
    # dip (dust or noise) does not count as the edge of the object.
    low=rays<threshold
    cross=low[:-2]&low[1:-1]&low[2:]
    cross[-1]=True
    i=np.argmax(cross,axis=0)
    t=np.arange(nb)
    above=rays[np.maximum(0,i-1),t]
    frac=np.clip((above-threshold)/np.maximum(.00001,above-rays[i,t]),0,1)
    r=np.maximum(1.,.5*(i-1+frac))
    r=np.median(np.stack([np.roll(r,k) for k in [-2,-1,0,1,2]]),axis=0)
    lr=np.log(r)
    h=2*(lr-lr.mean()).dot(phase2)/nb
    f=min(abs(h),.88)
    q=(1-f)/(1+f)
    a=np.exp(lr.mean())/(1-f)
    pa=.5*np.angle(h)
    pred=a/np.sqrt(np.cos(ang-pa)**2+(np.sin(ang-pa)/q)**2)
    # Radial asymmetry and fractional departure from an elliptical contour.
    skew=np.mean(np.minimum(abs(lr-np.roll(lr,nb//2)),1.))
    err=np.mean(np.minimum(abs(lr-np.log(pred)),.6))
    return a,q,pa,skew,err

def centered(img):
    # Smooth peak with weak catalogue-position prior and a sub-pixel centroid.
    prior=np.exp(-((Y-63.5)**2+(X-63.5)**2)/(2*8**2))
    j,i=np.unravel_index(np.argmax(np.maximum(img,0)*prior),img.shape)
    cx=float(i); cy=float(j)
    peak=max(.001,img[j,i])
    for k in range(6):
        w=np.exp(-((X-cx)**2+(Y-cy)**2)/(2*3.**2))*np.maximum(img-.17*peak,0)**2
        mass=w.sum()
        if mass<1.e-12:
            break
        cx=(X*w).sum()/mass; cy=(Y*w).sum()/mass
    return cx,cy

def secondcore(light,broad,red,cx,cy,peak,aouter,noise):
    # Is there a separate resolved light concentration in the same envelope?
    high=(broad>max(.075*peak,4*noise))&((X-cx)**2+(Y-cy)**2<62**2)
    for dy,dx in [(0,1),(1,1),(1,0),(1,-1)]:
        v=np.roll(broad,(dy,dx),(0,1))
        high&=(broad>=v)&(broad>np.roll(broad,(-dy,-dx),(0,1)))
    js,ii=np.nonzero(high)
    best=0.
    p0=max(.01,sample(light,cx,cy))
    cw=np.maximum(0,light-.5*p0)*np.exp(-((X-cx)**2+(Y-cy)**2)/(2*9**2))
    primarywidth=np.sqrt(cw.sum()/p0+1.)
    for j,i in zip(js,ii):
        d=np.hypot(j-cy,i-cx)
        if d<4.5: continue
        t=np.linspace(0,1,41)
        track=sample(light,cx+(i-cx)*t,cy+(j-cy)*t)
        dip=np.min(track)
        central=track[0]; top=track[-1]
        if top<.115*p0 or dip>.91*min(central,top): continue
        # The angular median rejects the primary core and estimates the
        # companion's own half-prominence width (stars are seeing-limited).
        rs=np.linspace(0,15,31)[:,None]
        spokes=sample(light,i+rs*ux[None,::8],j+rs*uy[None,::8])
        prof=np.median(spokes,axis=1)
        hwhm=.5*np.sum(np.minimum.accumulate(prof)>(.54*top+.46*max(dip,0)))
        if hwhm<1.6: continue
        broadness=sample(broad,i,j)/max(.01,top)
        rel=min(top,p0)/max(.01,top,p0)
        between=max(0.,dip/max(.01,min(p0,top)))
        sep=d/(primarywidth+hwhm+.5)
        # Two isolated galaxies do not constitute a merger. Require a real
        # bridge above the local sky, or nuclei inside a common half-light core.
        close=1-ramp(sep,1.15,2.35)
        bridge=ramp(between,.06,.32)*ramp(dip/noise,2.6,6.)
        linked=max(close*.90,bridge*(1-.8*ramp(sep,1.9,3.6)))
        extent=.18+.82*max(ramp(hwhm,2.,4.),ramp(broadness,.64,.86))
        comparable=.25+.75*ramp(rel,.12,.55)
        saddle=.30+.70*ramp(1-between,.10,.52)
        redcore=ramp(sample(red,i,j)/max(.02,red[int(cy+.5),int(cx+.5)]),.14,.70)
        value=linked*extent*comparable*saddle*(.82+.18*redcore)
        best=max(best,value)
    return best

def spectrum(im,cx,cy,a,q,pa):
    ca=np.cos(pa); sa=np.sin(pa)
    xx=cx+(unitR*a)[:,None]*(ca*ux-sa*q*uy)
    yy=cy+(unitR*a)[:,None]*(sa*ux+ca*q*uy)
    return sample(im,xx,yy)

def modes(polar,noise,peak):
    med=np.median(polar,axis=1)
    # Clip isolated foreground sources, but retain moderate arm/interarm contrast.
    floor=max(.012*peak,noise*1.5,.001)
    contrast=(polar-med[:,None])/(np.maximum(med,0)[:,None]+floor)
    contrast=np.clip(contrast,-1.35,1.35)
    coeff=2*np.fft.rfft(contrast,axis=1)[:,1:9]/nb
    return med,coeff

def avg(z,w):
    return np.sum(z*w)/(1.e-12+np.sum(w))

def meanrange(z,left,right):
    return np.mean(z[(unitR>=left)&(unitR<right)],axis=0)

def winding(coeff,weight):
    # In an elliptical log-polar chart the m-th harmonic of a logarithmic
    # spiral rotates through m*cot(pitch)*log(r). Compare opposite windings
    # so static oval distortions have little effect on the pitch vote.
    rates=np.array([.66,1.,1.45,2.,2.75,3.7,5.,6.7,9.])
    loose=0.; tight=0.; wave=0.; spiral=0.
    for m in [1,2,3,4]:
        z=coeff[:,m-1]
        for lo,hi in [(.22,.84),(.35,1.15),(.50,1.36)]:
            w=weight*(unitR>lo)*(unitR<hi)
            total=w.sum()
            if total<1.e-6: continue
            w=w/total
            zc=z-np.sum(z*w)
            ph=np.exp(1j*m*rates[:,None]*logR)
            pos=abs(ph.dot(zc*w)); neg=abs(ph.conj().dot(zc*w))
            # Removing the mean suppresses slowly rotating waves. Correct
            # the response of the finite log-radius window, with a floor.
            resp=np.maximum(1-abs(ph.dot(w))**2,.36)
            s=(np.maximum(pos,neg)-.80*np.minimum(pos,neg))/np.sqrt(resp)
            s*=.68 if m==1 else (1. if m==2 else .88)
            wave=max(wave,np.max(s))
            spiral=max(spiral,np.max(abs(pos-neg)))
            loose=max(loose,np.max(s[:5]))
            tight=max(tight,np.max(s[5:]))
    return loose,tight,wave,spiral

def tails(rays,noise,peak):
    # Compare the radial reach in antipodal sectors, searching for *broad*
    # one-sided plumes, not the maximum of a set of noisy single-pixel rays.
    po=rays; radii=rad[:,0]
    power=0.; lopsided=0.
    for low in [.18,.075,.035]:
        t=max(2.7*noise,low*peak)
        blank=(po<t)&(np.roll(po,-1,axis=0)<t)&(np.roll(po,-2,axis=0)<t)
        blank[-3]=True
        r=np.maximum(1.,radii[np.argmax(blank,axis=0)])
        lr=np.log(r); skew=lr-np.roll(lr,nb//2)
        sm=sum(np.roll(skew,k) for k in [-7,-5,-3,-1,1,3,5,7])/8
        power=max(power,np.percentile(sm,93))
        lopsided=max(lopsided,abs(np.mean(lr*np.exp(1j*ang)))*2)
    return power,lopsided

def faintcontour(light,cx,cy,peak,noise,axes,qs,pas):
    # Probe diffuse light outside the high surface brightness contours with
    # a broad matched smoothing scale. Antipodal sector differences isolate
    # a tidal tail without requiring it to contribute much integrated flux.
    spread=restrict(.11*axes[5],2.8,6.)
    faint=np.fft.irfft2(np.fft.rfft2(light)*lowpass(spread),s=(128,128))
    faint-=np.median(faint[edge])
    sky=max(.0015,1.48*np.median(abs(faint[edge])))
    rays=smoothang(sample(faint,cx+rad*ux,cy+rad*uy))
    dis=0.; bend=0.
    normal=axes[5]/np.sqrt(np.cos(ang-pas[5])**2+(np.sin(ang-pas[5])/qs[5])**2)
    for frac in [.060,.025]:
        threshold=max(sky*2.6,frac*peak)
        below=rays<threshold
        gap=below[:-3]&below[1:-2]&below[2:-1]&below[3:]
        gap[-1]=True
        r=np.maximum(1.,.5*np.argmax(gap,axis=0))
        lr=np.log(r); dif=lr-np.roll(lr,nb//2)
        region=sum(np.roll(dif,k) for k in [-8,-6,-4,-2,0,2,4,6,8])/9
        # A tail usually extends one side rather than obliterating the other.
        dis=max(dis,np.percentile(region*np.minimum(1.,r/(normal+spread)),94))
        h=2*np.mean(lr*phase2)
        bend=max(bend,abs(h-(1-qs[5])/(1+qs[5])*np.exp(2j*pas[5])))
    return dis,bend

def outerwidth(rays,noise,peak,pa):
    # A round bulge can dominate almost all the angular quadrature in log r
    # even though there is a distinctly linear extension along the major axis.
    # Compare the actual two-sided diameter with the diameter *across* it.
    t=max(.075*peak,3.2*noise)
    blanks=(rays<t)&(np.roll(rays,-1,axis=0)<t)&(np.roll(rays,-2,axis=0)<t)
    blanks[-3]=True
    radius=np.maximum(1.,.5*np.argmax(blanks,axis=0))
    r=.5*(radius+np.roll(radius,nb//2))
    idx=int(np.round(pa*nb/(2*np.pi)))
    def sector(k):
        ids=(idx+k+np.arange(-3,4))%nb
        return np.median(r[ids])
    length=sector(0)
    thickness=sector(nb//4)
    return length,thickness/max(1.,length)

def ferrersbar(polar,rc,unit):
    # A projected Ferrers bar has flat, symmetric ridges which end at both
    # ansae. Compare this pattern to a rounded oval and to curving/open arms.
    # Estimate the bar axis separately on each candidate radial interval;
    # quadrupole angle in radians is half its Fourier phase, sign reversed.
    signal=0.; distortion=0.
    center=np.median(polar,axis=1)
    p=(polar-center[:,None])/(np.maximum(center,0)[:,None]+.025*np.max(center)+.003)
    p=np.clip(p,-1,1.)
    for end in [.38,.50,.64,.81]:
        band=(unit>.43*end)&(unit<end)
        z=rc[:,1]; avgz=np.mean(z[band]); phi=-.5*np.angle(avgz)
        sep=np.arccos(np.clip(np.cos(ang-phi),-1,1))
        # The two long sides must both be brighter than their angular flanks.
        ridge=[]
        for ds in [sep,np.pi-sep]:
            on=np.mean(p[band][:,ds<.22],axis=1)
            shoulder=np.mean(p[band][:,(ds>.58)&(ds<1.03)],axis=1)
            ridge.append(np.median(on-shoulder))
        straight=min(ridge); balance=max(0,straight)/(max(.005,max(ridge)))
        # Subtract quadrupole expected simply from the projection of a round
        # bulge in this deprojected chart: it elongates along the minor axis
        # and has a strength proportional to the radial fall-off of light.
        r0=np.mean(z[(unit>.13)&(unit<.4*end+.05)]); r1=np.mean(z[(unit>end)&(unit<end*1.43)])
        profile=abs(avgz-r1)
        coherence=abs(avgz)/(np.mean(abs(z[band]))+.002)
        fixed=1-ramp(np.max(np.abs(z[band]/(abs(z[band])+.005)-avgz/(abs(avgz)+.005))),.35,1.2)
        distal=np.real((avgz-r1)*np.conj(avgz))/(abs(avgz)+.003)
        v=straight*(.4+.6*ramp(balance,.33,.83))*(.48+.52*fixed)*(.60+.40*ramp(distal,.025,.16))
        signal=max(signal,v)
        distortion=max(distortion,profile*coherence)
    return signal,distortion

def arctest(polar,med,noise,peak,q,a):
    # Build a fractional high-pass map in log radius. An axisymmetric Sersic
    # component and static isophotal ellipticity then contribute little. In
    # this chart the gradient normal to an arm satisfies |Ir/It|=cot(pitch),
    # where r denotes log(radius) and t denotes azimuth in the disk plane.
    p=np.clip((polar-med[:,None])/(np.maximum(med,0)[:,None]+max(noise*2,.016*peak)), -1.,1.)
    p=smoothang(smoothang(p))
    inner=np.clip(np.arange(len(logR))-4,0,len(logR)-1)
    outer=np.clip(np.arange(len(logR))+4,0,len(logR)-1)
    dr=(p[outer]-p[inner])/(8*(logR[1]-logR[0]))
    dt=(np.roll(p,-3,1)-np.roll(p,3,1))/(6*2*np.pi/nb)
    weights=np.minimum(1,(np.maximum(med,0)/(6*noise))**2)[:,None]*((unitR>.38)&(unitR<1.22))[:,None]
    # Aggregate the structure tensor locally in radius and azimuth. Subtract
    # the isotropic component; alternating isolated clumps have no handedness.
    cross=dr*dt
    xx=dr**2; yy=dt**2
    handed=0.; total=1.e-12; opening=0.; strength=0.
    for zone in [(.38,.65),(.64,.9),(.88,1.20)]:
        w=weights*((unitR>zone[0])&(unitR<zone[1]))[:,None]
        for group in range(8):
            sl=slice(group*16,(group+1)*16)
            u=np.sum(w*xx[:,sl]); v=np.sum(w*yy[:,sl]); z=np.sum(w*cross[:,sl]);
            n=16*np.sum(w)+1.e-9
            oblique=abs(z)/np.sqrt(.0001*n*n+u*v)
            power=abs(z)/n*oblique
            power=min(power,.7)*ramp(oblique,.12,.70)
            pitch=.5*np.arctan2(2*abs(z),u-v)
            handed+=np.sign(z)*power; total+=power
            opening+=power*ramp(pitch,np.pi/11,np.pi/5)
            strength+=power
    coherence=abs(handed)/total
    return strength/24,coherence,opening/total

def classify(pic):
    sm=np.fft.irfft2(np.fft.rfft2(pic)*seeing,s=(128,128),axes=(-2,-1))
    sm-=np.median(sm[:,edge],axis=1)[:,None,None]
    red,g,b=sm
    light=.30*red+.42*g+.28*b
    broad=np.fft.irfft2(np.fft.rfft2(light)*corescale,s=(128,128))
    noise=max(.002,1.48*np.median(abs(light[edge])))
    cx,cy=centered(light)
    peak=max(.005,sample(light,cx,cy))
    rays=sample(light,cx+rad*ux,cy+rad*uy)
    rays=smoothang(rays)
    measures=np.array([iso(rays,max(v*peak,2.6*noise)) for v in levels])
    axes,qs,pas,askew,errors=measures.T
    aout=np.median(axes[4:7]); ain=axes[2]; half=.5*(axes[1]+axes[2])
    qin=np.median(qs[:3]); qmid=np.median(qs[2:5]); qout=np.median(qs[4:7]); qfar=np.median(qs[5:])
    # Use the outer disk to define the elliptical polar chart. Work with
    # double angles: a major axis and its reverse are the same axis.
    orient=np.sum((1-qs[4:7])*np.exp(2j*pas[4:7]))
    pa=.5*np.angle(orient)
    a=restrict(aout,4.,70.); q=restrict(qout,.07,1.)
    lp=spectrum(light,cx,cy,a,q,pa); bp=spectrum(b,cx,cy,a,q,pa); rp=spectrum(red,cx,cy,a,q,pa)
    med,co=modes(lp,noise,peak)
    bm,bc=modes(bp,noise*1.4,peak)
    rm,rc=modes(rp,noise*1.4,peak)
    # Blue stars outline the arms better than the old red population.
    co=.72*co+.28*bc
    w=np.minimum(1,np.maximum(med,0)/(6*noise))
    loose,tight,wave,chir=winding(co,w)
    # Amplitudes of centered odd modes, fine-scale modes and bar quadrupole.
    odd=np.sqrt(np.sum(abs(co[:,[0,2,4]])**2,axis=1))
    high=np.sqrt(np.sum(abs(co[:,2:])**2,axis=1))
    oddity=avg(odd,w*(unitR>.38)*(unitR<1.02))
    fine=avg(high,w*(unitR>.42)*(unitR<1.12))
    oddin=meanrange(abs(co[:,0]),.19,.55)
    sk=np.median(askew[2:6]); sko=np.median(askew[4:]); err=np.median(errors[3:7]);
    colcore=meanrange((rm-bm)/(np.maximum(rm+bm,0)+.015),.11,.38)
    colout=meanrange((rm-bm)/(np.maximum(rm+bm,0)+.025),.58,.95)
    gradient=colcore-colout
    comp=secondcore(light,broad,red,cx,cy,peak,aout,noise)
    # A bar remains at nearly fixed position angle across the inner disc,
    # has large deprojected ellipticity, and fades or turns at its ends.
    bar=0.; barcont=0.
    for left,right in [(.13,.40),(.20,.55),(.28,.69)]:
        mask=(unitR>left)&(unitR<right)
        z=.72*rc[mask,1]+.28*co[mask,1]
        mz=np.mean(z); amp=abs(mz)
        aligned=amp/(np.mean(abs(z))+.006)
        outer=meanrange(co[:,1],max(.65,right+.06),1.10)
        delta=abs(mz-outer)
        v=amp*(.45+.55*ramp(aligned,.7,.97))*(.65+.35*ramp(delta,.08,.3))
        if v>bar: bar=v; barcont=delta
    # Departures from bilateral symmetry are also measured at isophotes,
    # giving faint plumes some weight even when not bright in the blue band.
    offset=np.hypot(cx-63.5,cy-63.5)
    tail,onearm=tails(rays,noise,peak)
    if comp>.47 or (comp>.30 and (sk>.18 or oddity>.23 or offset>3 or (tail>.56 and onearm>.19))) or (comp>.20 and (sk>.33 or offset>4.5)):
        return 1
    # Blur rounds barely-resolved cores. Use several isophotes and a modest
    # quadrature correction for the seeing when classifying the overall shape.
    qraw=.28*qin+.45*qmid+.27*qout
    smq=np.sqrt(max(.025,(qmid*ain)**2-1.65**2)/max(1.,ain**2-1.65**2))
    shape=.72*qraw+.28*restrict(smq,.08,1.)
    # Edge-on disks: very thin projected outer envelope or long spindle.
    # At better resolution, a dust lane, boxiness and colour structure
    # can identify disks that would otherwise look like cigar ellipticals.
    longdisk=ramp(aout,13,30)
    slim=.55*qout+.30*qfar+.15*qmid
    # The bulge can round out the bright inner isophotes. Inspect a fainter
    # contour and compare vertical thickness near the axis and out on the disk.
    ca=np.cos(pa); sa=np.sin(pa)
    u=(a*unitR)[:,None]; v=np.linspace(0,max(6,.9*a),48)[None,:]
    sides=[]; vert=[]
    for sign in [-1.,1.]:
        def stripe(vv):
            return sample(light,cx+sign*u*ca-vv*sa,cy+sign*u*sa+vv*ca)
        top=.5*(stripe(v)+stripe(-v))
        vert.append(top)
        sides.append(np.sum(np.minimum.accumulate(top,axis=1)>.5*np.maximum(top[:,:1],5*noise),axis=1)*(v[0,1]-v[0,0]))
    hz=.5*(sides[0]+sides[1]); fat=meanrange(hz,.08,.25)/(meanrange(hz,.42,.70)+.1)
    tip=np.median(axes[5:]); tipq=.7*qfar+.3*slim
    ratio=(qin+.03)/(qfar+.03)
    edgecut=.28+.145*longdisk+.035*ramp(gradient,.02,.22)+.075*ramp(ratio,1.17,1.75)
    length,rimq=outerwidth(rays,noise,peak,pa)
    # The cross section off the midplane should contain a central lens-shaped
    # light excess over the adjacent disk, including in nearly dust-obscured cores.
    face=.5*(vert[0]+vert[1]); h=meanrange(hz,.42,.70)
    above=(v[0]>1.1*h)&(v[0]<max(3*h,1.1*h+2))
    excess=0.
    if np.any(above):
        lens=meanrange(np.mean(face[:,above],axis=1),.08,.24)
        sheet=meanrange(np.mean(face[:,above],axis=1),.43,.65)
        excess=(lens-sheet)/max(.01,peak)
    major=meanrange(face[:,0],.48,.68)/peak
    corelight=meanrange(med,.14,.25)/max(.01,meanrange(med,.50,.73))
    # Disks stay narrow far from the core. In an ellipse the HWHM of a
    # perpendicular slice tends instead to *increase* when the slice moves
    # off center. The rounded light embedded in the narrow slice is a bulge.
    thinwing=h/max(2.,length)
    isdisk=(thinwing<.133 and rimq<.52 and length>13.5 and (fat>.93 or rimq<.39))
    needle=(tipq<edgecut and tip>12) or isdisk or (rimq<.50 and length>19 and (fat>1.45 or ratio>1.28 or excess>.12))
    if needle and qmid<.67 and (sk<.37 or slim<.26):
        central=half/max(3.,aout)
        swell=.8*ramp(fat,1.08,1.6)+.43*ramp(ratio,1.02,1.60)+.48*ramp(qin,.22,.5)+.45*ramp(colcore,-.03,.16)+.42*ramp(central,.52,.28)+.50*ramp(slim,.15,.35)+.85*ramp(excess,.07,.23)+.36*ramp(major,.45,.23)
        # A bulge can be small in absolute terms; compare the color of
        # the mid-plane center to the tips, and the mid-plane radial profile
        # with a simple exponential disk extrapolated through the center.
        axial=.5*(lp[:,0]+lp[:,nb//2]); cxlight=meanrange(axial,.075,.18)
        body=max(.003,meanrange(axial,.37,.54)); tiplight=max(.002,meanrange(axial,.70,.89))
        nuc=np.log(max(.003,cxlight)*tiplight/(body*body))
        corered=meanrange((rm-bm)/(np.maximum(rm+bm,0)+.01),.08,.37)
        swell+=.68*ramp(nuc,-.12,.55)+.5*ramp(corered,-.07,.12)+.36*ramp(rimq,.14,.34)+.4*ramp(excess,.025,.11)
        # A young, very thin constant-thickness disk may have strong midplane
        # colour but no structural widening. Do not count dust as a bulge.
        swell-=.5*ramp(1.18-fat,0.,.40)*ramp(.31-rimq,0.,.14)+.25*ramp(-corered,.035,.22)
        return 9 if swell>1.70 else 8
    # A disturbed system has lopsided light on more than one spatial scale,
    # or a strongly displaced peak in an irregular, often star-forming body.
    arc,coherence,openness=arctest(lp,med,noise,peak,q,a)
    even=avg(abs(co[:,1]),w*(unitR>.38)*(unitR<1.02))
    bumpy=ramp(sk,.09,.33)+ramp(oddity,.09,.30)+.60*ramp(oddin,.06,.25)+.6*ramp(offset,1.4,4.5)+.45*ramp(sko,.13,.36)+.48*ramp(-colcore,.03,.22)
    # Tidal signatures need not include several bright clumps. One long
    # one-sided plume, a twisted inner body, or a strongly shifted photocentre
    # is evidence too. Penalize an ordinary two-armed density wave.
    tidal=bumpy+.85*ramp(tail,.27,.79)+.58*ramp(onearm,.10,.32)+.38*ramp(-colcore,.02,.19)-.48*ramp(even/(oddity+.015),.9,2.)
    plume,bend=faintcontour(light,cx,cy,peak,noise,axes,qs,pas)
    tidal+=.95*ramp(plume,.19,.66)+.40*ramp(bend,.065,.23)*ramp(tail,.19,.50)
    if tidal>2.35 and (sk>.10 or oddin>.10 or tail>.48 or plume>.47) and (oddity>.65*even or offset>2.7 or max(plume,tail)>.57) and (arc*coherence<.095 or oddity>.31):
        return 0
    # A distant pair of tidal extensions can remain point-symmetric (low
    # odd modes), but the outer axis swings away from an undisturbed core.
    twist=abs(np.sin(pas[6]-pas[2]))*np.sqrt((1-qs[6])*(1-qs[2]))
    if (plume>.52 and bend>.13 and err>.058 and chir<.09 and colcore<.12) or (twist>.32 and sko>.18 and fine<.18 and colcore<.06):
        return 0
    # A relative change in ellipticity alone misses a bar which is parallel
    # to the apparent disk major axis. Compare inner and outer quadrupole
    # vectors as well, computing ellipticity in isophotal (not light) units.
    vectors=(1-qs)/(1+qs)*np.exp(2j*pas)
    ez=np.mean(vectors[4:7]); iz=np.mean(vectors[1:4]);
    bariso=abs(iz-ez)*(1-.65*ramp(qin-qout,.0,.23))
    # A stellar bar is a fairly long *straight* feature in an oval red
    # component; in particular it may sit in a virtually featureless disk.
    # Track its phase, rising quadrupole amplitude and sudden end in the red.
    barline=0.
    for left,right in [(.15,.42),(.21,.56),(.27,.72),(.34,.85)]:
        zr=rc[(unitR>left)&(unitR<right),1]
        zh=meanrange(rc[:,1],.10,left+.05)
        ze=meanrange(rc[:,1],right,min(1.40,1.6*right))
        zm=np.mean(zr); const=abs(zm)/(np.mean(abs(zr))+.004)
        fixed=abs(zm)*ramp(const,.7,.97)
        end=abs(zm-ze); rise=abs(zm-zh)
        barline=max(barline,fixed*(.6+.4*ramp(end,.04,.15))*(.73+.27*ramp(rise,.015,.10)))
    barvote=max(bar/.143,bariso/.144,barline/.115)*(.76+.24*ramp(q,.3,.6))
    # An undistorted ellipse of *any* radial brightness profile has a
    # constant rank around its own isophote. Measure this directly, rather
    # than treating high angular harmonics in a misaligned chart as arms.
    contour=np.median(errors[1:6]); outererr=np.median(errors[4:])
    reg=1-max(ramp(contour,.025,.080),.8*ramp(sk,.05,.14),ramp(bariso,.05,.14),.6*ramp(outererr,.045,.13))
    # A winding arm can masquerade as a linear bar over a short radial
    # interval. Look for excess light of older (redder) stars along the bar,
    # a flat-topped quadrupole and two-sided straightness in the inner chart.
    zbar=meanrange(rc[:,1],.25,.6)
    rawcolor=(rp-bp)/(np.maximum(0.,rp+bp)+.04)
    chrom=meanrange(2*np.mean(rawcolor*np.exp(-2j*ang),axis=1),.25,.62)
    redbar=np.real(chrom*np.conj(zbar))/(abs(zbar)+.005)
    angle1=meanrange(rc[:,1],.16,.36); angle2=meanrange(rc[:,1],.39,.67)
    straight=np.real(angle1*np.conj(angle2))/(abs(angle1*angle2)+.00015)
    # Counter-arm pixels define a bright connected ridge on both sides,
    # unlike the many one-sided ridges of flocculent spiral structure.
    oddfrac=meanrange(abs(rc[:,0])+abs(rc[:,2]),.22,.63)/(abs(zbar)+.035)
    # Size is evidence of resolved disk structure, not sufficient by itself;
    # a red, large but completely featureless galaxy is still elliptical.
    sizevote=ramp(a*np.sqrt(q),7.,19.)
    resolved=ramp(a,9.,23.)
    arms=max(ramp(wave,.022,.10),ramp(fine,.049,.16),ramp(arc,.028,.19)*(.6+.4*coherence))
    disk=1.05*arms+.48*ramp(gradient,.045,.21)+.48*sizevote+.37*resolved+.36*ramp(err,.028,.11)+.38*ramp(barvote,.38,1.2)+.23*ramp(-colout,.02,.22)
    disk-=.53*reg*(1-.7*ramp(gradient,.08,.22))
    disk+=.24*ramp(corelight,2.6,1.7)*sizevote+.23*ramp(gradient,.08,.25)*resolved
    # Spiral disks have a distinct shallow envelope and are more resolved
    # than a seeing-dominated smooth core. Blue-light extent is useful when
    # a tightly wound pattern adds only a very low luminance contrast.
    # Typical disk/envelope transition: in an exponential the logarithm of
    # surface brightness is linear in r. A spheroid has a steep cusp and a
    # progressively shallower log-slope (de Vaucouleurs profile). Compare
    # three equally spaced annuli after suppressing light from companions.
    p0=meanrange(med,.18,.30); p1=meanrange(med,.50,.62); p2=meanrange(med,.82,.94)
    curve=np.log(max(.005,p0)*max(.005,p2)/(max(.005,p1)**2))
    disk+=.46*ramp(-curve,-.015,.30)*ramp(a*np.sqrt(q),7.,14.)-.36*ramp(curve,.13,.40)*reg
    bluehalo=meanrange(bm,.68,1.03)/max(.02,meanrange(bm,.09,.27))
    redhalo=meanrange(rm,.68,1.03)/max(.02,meanrange(rm,.09,.27))
    disk+=.40*ramp(bluehalo-redhalo,.035,.18)*sizevote
    disk-=.36*(1-resolved)*ramp(peak,.28,.66)*ramp(reg,.30,.78)
    if disk>1.31 or (barvote>.88 and resolved>.22 and reg<.57):
        # Pitch is a geometric quantity, but flocculent loose arms can have
        # little coherent Fourier amplitude. Their outer light is patchy and
        # their contrast crosses a broad range of radius on a single ray.
        ratio=loose/(tight+.006)
        opening=.72*ramp(ratio,.84,1.55)+.76*openness+.55*ramp(loose,.032,.13)+.64*ramp(oddity,.09,.31)+.44*ramp(fine,.069,.23)+.37*ramp(-colout,.01,.17)
        # Radial curvature of the two-armed phase: open arms maintain a
        # distinctive angular sector over much of the outer disk, while
        # tight arms pass a given sector swiftly as radius increases.
        z2=co[:,1]; select=(unitR>.46)&(unitR<1.20)
        lock=abs(np.mean(z2[select]))/(np.mean(abs(z2[select]))+.008)
        curled=np.mean(abs(z2[select]-np.mean(z2[select])))
        opening+=.48*ramp(lock,.38,.88)*ramp(curled,.03,.12)+.27*ramp(oddfrac,.5,1.4)-.21*ramp(corelight,2.5,4.4)
        intrinsic=(1-min(1,qin/qout))
        barquality=barvote*(.62+.38*ramp(straight,.30,.88))*(1-.40*ramp(oddfrac,.60,1.70))
        barquality*=1+.32*ramp(redbar,.004,.045)+.22*ramp(intrinsic,.14,.38)-.26*ramp(-redbar,.006,.05)
        ridge,turn=ferrersbar(rp,rc,unitR)
        # Barred galaxies get two votes: the shape of the model ridge itself
        # and elongation of the inner isophotes in the observed image. Requiring
        # these to agree prevents deprojected round bulges from becoming bars.
        innerflat=min(qs[2],qs[3],qs[4])
        # Projection stretches a spherical core along the reconstructed minor
        # axis. If a putative bar points there, test its intrinsic elongation.
        minor=max(0.,-np.real(zbar)/(abs(zbar)+.003))
        bulgealias=minor*ramp(qin-qout,.02,.24)*ramp(qin,.57,.90)
        barquality*=1-.68*bulgealias
        ridgematch=ramp(ridge,.025,.17)
        barquality*=.58+.52*ridgematch+.19*ramp(.73-innerflat,0,.32)
        if barquality>.76+.24*ramp(opening,1.25,2.5) and barquality>4.2*wave+.45 and barquality>3.*oddin+.45 and q>.29 and (barline>.075 or bariso>.125 or bar>.18) and (innerflat<.70 or redbar>.020 or ridge>.16):
            return 5
        # Loose arms also produce a large outer quadrupole *in their own
        # right*, with strong light contrast across each pair of arcs. A
        # tracer of recent star formation emphasizes these knots in blue.
        ampout=meanrange(abs(bc[:,1])+abs(bc[:,2]),.54,1.11)
        opening+=.40*ramp(ampout,.105,.34)+.28*ramp(-colcore,.04,.23)*ramp(fine,.05,.18)
        return 7 if opening>1.55 else 6
    if shape>.79: return 2
    if shape>.46: return 3
    return 4

def predict(x):
    return np.array([classify(p) for p in x],dtype=np.int64)
