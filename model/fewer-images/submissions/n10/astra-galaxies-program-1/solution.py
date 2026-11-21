import numpy as np
import math

Y, X = np.indices((128, 128), dtype=np.float64)
field = (X-63.5)**2 + (Y-63.5)**2
skyarea = field > 56**2
f2 = np.fft.fftfreq(128)[:,None]**2 + np.fft.rfftfreq(128)[None,:]**2

def bell(s):
    return np.exp((-2*np.pi**2*s*s)*f2)

beam = [bell(s) for s in (0.9, 1.5, 3.0, 6.0)]

def smooth(tf, k):
    return np.fft.irfft2(tf * beam[k], s=(128,128), axes=(-2,-1))

def ramp(a):
    return np.clip(a, 0., 1.)

def interp(im, x, y):
    x = np.clip(x, 0, 126.9999)
    y = np.clip(y, 0, 126.9999)
    i, j = x.astype(int), y.astype(int)
    a, b = x-i, y-j
    return ((1-a)*im[...,j,i]+a*im[...,j,i+1])*(1-b) + ((1-a)*im[...,j+1,i]+a*im[...,j+1,i+1])*b

def desky(x):
    offset = np.median(x[:,skyarea], axis=1)
    return x.astype(np.float64) - offset[:,None,None]

def dev(im):
    vals = im[skyarea]
    return 1.483*np.median(np.abs(vals-np.median(vals)))+0.001

def centre(lit, broad, noise):
    # Combine a soft position prior with the prominence of an extended source.
    prior = np.exp(-field/(2*6**2))
    merit = prior * np.maximum(broad,0)
    iy, ix = np.unravel_index(merit.argmax(), merit.shape)
    if broad[iy, ix] < noise:
        return 63.5,63.5
    cx,cy = float(ix),float(iy)
    for size in (4., 2.8, 2., 1.5, 1.5):
        w = np.maximum(lit,0)**2 * np.exp(-((X-cx)**2+(Y-cy)**2)/(2*size**2))
        total = w.sum()+1.e-15
        cx, cy = (w*X).sum()/total, (w*Y).sum()/total
    return cx, cy

def peakcolour(rgb,x,y):
    a=interp(rgb,x,y); total=max(.01,a[0]+a[2])
    return np.clip((a[0]-a[2])/total,-1.,1.)

def maxima(fine, mid, broad, cx,cy,noise,rgb):
    # Local maxima, non-maximum suppression, and the saddle on the line to the nucleus.
    # A separate, spatially resolved galaxy with connecting light has a high pair score.
    loc = np.ones(fine.shape,dtype=bool)
    for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        loc &= mid >= np.roll(np.roll(mid,dy,axis=0),dx,axis=1)
    rr = (X-cx)**2+(Y-cy)**2
    primary = max(interp(mid,cx,cy), 4*noise,0.01)
    cand = loc & (rr>3**2) & (rr<58**2) & (mid>max(4*noise,0.055*primary))
    js, ii = np.where(cand)
    order = np.argsort(-mid[js,ii])[:100]
    mask = np.ones(fine.shape)
    pair = 0.
    paircandidate=0.
    cwidth = np.clip(interp(broad,cx,cy)/(primary+.001),0,1)
    line = np.linspace(0,1,32)
    ringangle=np.arange(32)*np.pi/16
    cs,sn=np.cos(ringangle),np.sin(ringangle)
    primaryscale=max(3., np.sqrt(np.count_nonzero(component(mid>.15*primary,cx,cy))/np.pi))
    for n in order:
        j,i = js[n],ii[n]
        d = np.sqrt(rr[j,i]); high=mid[j,i]
        profile = interp(mid,cx+(i-cx)*line,cy+(j-cy)*line)
        low = max(0.,profile.min())
        prominence = high-low
        if prominence <= max(noise, .04*high):
            continue
        # Compare the dimensions of the secondary peak to a seeing-limited star.
        wide = broad[j,i]/(high+.001)
        near = fine[j,i]/(high+.001)
        # Remove the underlying pedestal before testing for excess over the PSF.
        pedestal=min(.8*high, .75*low)
        wr=(broad[j,i]-pedestal)/(high-pedestal+.001)
        width=np.sqrt(np.clip((9*wr-2.25)/max(.05,1-wr),.25,100.))
        width0=np.sqrt(np.clip((9*cwidth-2.25)/max(.05,1-cwidth),.25,100.))
        resolved=ramp((width-1.45)/1.45)*ramp((1.55-near)/.36)
        lumratio=(high-pedestal)*(width**2+2.25)/(primary*(width0**2+2.25))
        symmetric=min(lumratio,1./max(.001,lumratio))
        brightness=ramp(symmetric/.30)*ramp(high/(.32*primary))
        bridge=low/max(min(high,primary),.01)
        tight=np.exp(-.5*(d/(2.7*(width+width0)+1.))**2)
        affinity=tight*(.22+.78*ramp(bridge/.20))
        col=peakcolour(rgb,i,j); nucleus=peakcolour(rgb,cx,cy)
        starforming=ramp((nucleus-col-.10)/.23)*ramp((.16-col)/.20)
        brightness*=1.-.70*starforming*(1.-ramp(lumratio/.85))
        likelihood=1.4*brightness*affinity*ramp(prominence/(.35*high))*(.15+.85*resolved)
        # Curve of growth about the companion. Use the arc facing AWAY from the
        # target, so its host disk cannot masquerade as a secondary envelope.
        arc=(cs*(i-cx)+sn*(j-cy))>0
        rings=interp(mid,i+np.array([3.,5.,8.,12.])[:,None]*cs,j+np.array([3.,5.,8.,12.])[:,None]*sn)
        envelope=np.median(rings[:,arc],axis=1)
        wings=np.maximum(0.,(envelope[:3]-.75*envelope[3])/max(.01,high-.75*envelope[3]))
        # Gaussian PSF convolved with the detection filter. Extended sources retain
        # flux outside that core at several independent radii.
        extended=.55*ramp((wings[1]-.075)/.18)+.45*ramp((wings[0]-.36)/.24)
        wideflux=.5*(high+envelope[0])*9+.5*(envelope[0]+envelope[1])*16+.5*(envelope[1]+envelope[2])*39
        fluxratio=wideflux/(primary*primaryscale**2*.40+.001)
        scale2=np.sqrt(max(2.,wideflux/high))
        reach=d/(primaryscale+1.8*scale2)
        touching=np.exp(-.5*(reach/1.15)**2)
        bright=ramp((min(high/primary,primary/high)-.10)/.40)
        affinity2=touching*(.5+.5*ramp(bridge/.16))
        companion=(.32+.68*extended)*bright*(.45+.55*ramp(fluxratio/.38))*affinity2
        companion*=ramp(prominence/(.25*high))*(1-.65*starforming*(1-ramp(fluxratio/.64)))
        likelihood=max(likelihood,1.38*companion)
        paircandidate=max(paircandidate,likelihood)
        # Compare the saddle with the superposition of two noninteracting smooth
        # profiles, sampled on their antipodal sides. Also retain blends so close
        # that two overlapping cores are themselves evidence for an interaction.
        join=(line>=.2)&(line<=.8)
        f=line[join]; vx=i-cx; vy=j-cy
        opposite1=interp(mid,cx-f*vx,cy-f*vy)
        opposite2=interp(mid,i+(1-f)*vx,j+(1-f)*vy)
        excess=np.maximum(0.,profile[join]-np.maximum(opposite1,0)-np.maximum(opposite2,0))
        tide=ramp((np.median(excess)-noise)/(.10*min(primary,high)+2*noise))
        overlap=ramp((bridge-.07)/.29)*ramp((1.45-reach)/.7)
        likelihood*=.48+.52*max(tide,overlap)
        pair=max(pair,likelihood)
        # Mask the contaminant to about where its wings meet the primary profile.
        radius = np.clip(2.1*np.sqrt(max(.5,(9*wide-2.25)/max(.08,1-wide)))+1.5,3,16)
        radius = min(.65*d, radius*(1.-.55*ramp(bridge/.7)))
        dd = np.sqrt((X-i)**2+(Y-j)**2)
        knotlike = ramp((bridge-.30)/.45)*starforming*ramp((.5-lumratio)/.35)
        mask = np.minimum(mask, knotlike+(1-knotlike)*ramp((dd-.64*radius)/max(.36*radius,1.)))
    return mask,pair,paircandidate

def component(binary,cx,cy):
    # Eight-connected component through a small seeing disk at the target; span flood fill.
    h,w=binary.shape
    cx=int(np.clip(round(cx),0,w-1)); cy=int(np.clip(round(cy),0,h-1))
    todo=binary.copy()
    todo[max(0,cy-1):cy+2,max(0,cx-1):cx+2]=True
    output=np.zeros_like(todo)
    spans=[(cy,cx)]
    while spans:
        y,x=spans.pop()
        if not todo[y,x]:
            continue
        l=x
        while l>0 and todo[y,l-1]:
            l-=1
        r=x+1
        while r<w and todo[y,r]:
            r+=1
        todo[y,l:r]=False
        output[y,l:r]=True
        a=max(0,l-1); b=min(w,r+1)
        for yy in (y-1,y+1):
            if yy>=0 and yy<h:
                row=todo[yy,a:b]
                if row.any():
                    starts=np.where(row & np.concatenate(([True],~row[:-1])))[0]
                    for k in starts:
                        spans.append((yy,a+k))
    return output

def tensor(w,dx,dy,psf):
    W=w.sum()+1.e-10
    xx=(w*dx**2).sum()/W; yy=(w*dy**2).sum()/W; xy=(w*dx*dy).sum()/W
    delta=np.hypot(xx-yy,2*xy)
    major=max(.5,(xx+yy+delta)/2-psf**2)
    minor=max(.25,(xx+yy-delta)/2-psf**2)
    return np.clip(np.sqrt(minor/major),.065,1.), .5*np.arctan2(2*xy,xx-yy),np.sqrt(major)

def axes(sm,mask,cx,cy,noise,peak):
    dx,dy=X-cx,Y-cy
    levels=np.maximum(peak*np.array([.065,.14,.28,.50,.72]), noise*np.array([2.6,3.,4.,5.,6.]))
    qs=[]; ts=[]; lens=[]; drifts=[]; residuals=[]; boxes=[]
    for level in levels:
        seg = component((sm>level)&(mask>.25),cx,cy)
        # Antialias the isophote edge rather than count whole border pixels.
        w=seg*ramp((sm-level)/max(.14*level,noise))*mask
        q,t,rad=tensor(w,dx,dy,0.)
        ct,st=np.cos(t),np.sin(t)
        area=w.sum()+.0001
        xc,yc=(w*dx).sum()/area,(w*dy).sum()/area
        drifts.append(np.hypot(xc*ct+yc*st,(yc*ct-xc*st)/q)/(rad*2))
        # Fractional area of this isophote not accounted for by a smooth ellipse.
        u=(dx*ct+dy*st)/(2*rad); v=(dy*ct-dx*st)/(2*rad*q)
        er=np.sqrt(u*u+v*v)
        # Rolloff of the pixel-integrated ellipse across half a pixel at its boundary.
        roll=np.sqrt(u*u+(v/q)**2)/np.maximum(.05,er)/(2*rad)
        ew=ramp((1-er)/np.maximum(.01,roll)+.5)*mask
        residuals.append(np.abs(w-ew).sum()/max(area,ew.sum(),1.))
        phi=np.arctan2(v,u)
        boxes.append((w*np.cos(4*phi)*er**2).sum()/area)
        qs.append(q); ts.append(t); lens.append(2*rad)
    qs=np.array(qs); ts=np.array(ts); lens=np.array(lens)
    t=.5*np.angle(np.sum((1.-qs[:3]) * np.array([.32,.42,.26])*np.exp(2j*ts[:3])))
    q=np.median(qs[:3]); R=max(4.,.5*(lens[0]+1.25*lens[1]))
    return q,t,min(R,80.),qs,ts,lens,np.array(drifts),np.array(residuals),np.array(boxes)

angles=np.arange(96)*2*np.pi/96
ca,sa=np.cos(angles),np.sin(angles)
logscale=np.linspace(np.log(.10),np.log(1.16),36)
fr=np.exp(logscale)
pitches=np.array([8.,12.,16.,20.,25.,32.,41.,52.])
wind=1/np.tan(pitches*np.pi/180)

def angular(sm,mask,cx,cy,q,t,R,noise,peak):
    # Elliptical polar Fourier decomposition. Four orientations of an elongated
    # isophote are an even harmonic; spiral arms have phase which advances with log r.
    ct,st=np.cos(t),np.sin(t)
    rad=R*fr[:,None]
    dx=rad*(ca*ct-q*sa*st); dy=rad*(ca*st+q*sa*ct)
    px,py=cx+dx,cy+dy
    v=interp(sm,px,py); m=interp(mask,px,py)
    med=np.median(v,axis=1)
    opp=np.roll(v,48,axis=1); oppm=np.roll(m,48,axis=1)
    replacement=opp*oppm + (1-oppm)*med[:,None]
    v=v*m+(1-m)*replacement
    mean=v.mean(axis=1)
    scale=np.maximum(mean,0)+2.5*noise+.012*peak+.002
    sig=(v-mean[:,None])/scale[:,None]
    sig=np.clip(sig,-1.8,2.)
    F=np.fft.rfft(sig,axis=1)/len(angles)
    dr=logscale[1]-logscale[0]
    good=ramp((mean-1.6*noise)/(4*noise+.018*peak))*(.3+.7*ramp((m.mean(axis=1)-.6)/.3))
    radial=fr*R
    # Search coherent ridge segments (long and short) after removing the part
    # which is constant in azimuthal phase along radius (e.g., a projected ellipse).
    best=np.zeros(len(wind)); outer=np.zeros(len(wind)); innerbar=0.; barangle=0.
    for start,end in ((.22,.77),(.30,1.10),(.42,1.13),(.18,.54)):
        w=good*((fr>=start)&(fr<=end))*ramp((radial-1.5)/2.5)
        sw=w.sum()+1.e-5
        val=np.zeros(len(wind))
        for mode in (2,3,1):
            f=F[:,mode]; avg=(w*f).sum()/sw
            a=f-avg
            phase=mode*logscale[:,None]*wind[None,:]
            kernels=np.exp(1j*phase)
            kp=np.abs((w[:,None]*kernels).sum(axis=0))/sw
            # Shrink the correction for very open features degenerate with ovals.
            norm=sw*np.sqrt(np.maximum(.30,1-kp**2))
            plus=np.abs((w*a)@kernels)/norm
            minus=np.abs((w*a)@np.conj(kernels))/norm
            # Real radial changes in ellipticity project equally on both handednesses.
            # A genuine spiral ridge must prefer one of them.
            cf=np.maximum(plus,minus)-.65*np.minimum(plus,minus)
            if mode==2:
                val+=cf
            elif mode==3:
                val+=.42*cf
            else:
                val+=.22*cf
        val*=ramp(sw/6.)
        best=np.maximum(best,val)
        if start>=.30:
            outer=np.maximum(outer,val)
    # An elongated bar is a sustained fixed-phase inner quadrupole, preferably
    # different from the outer disk. This rejects a single inclined axisymmetric disk.
    for a,b in ((.14,.38),(.2,.49),(.27,.60)):
        w=good*((fr>=a)&(fr<=b))*ramp((radial-1.1)/1.8)
        z=(w*F[:,2]).sum()/(w.sum()+.001)
        u=good*(fr>b*1.35)*(fr<1.1)
        zouter=(u*F[:,2]).sum()/(u.sum()+.001)
        strength=min(abs(z),abs(z-.75*zouter))*ramp(w.sum()/4)
        if strength>innerbar:
            innerbar=strength; barangle=np.angle(z)
    # Subtract an approximate contribution of correlated sky noise in quadrature.
    sel=good*(fr>.28)*(fr<1.03)
    sel=sel/(sel.sum()+1.e-6)
    pairs=(sig-np.roll(sig,48,axis=1))*.5
    asym=np.sqrt(max(0.,(sel*(np.mean(np.minimum(pairs**2,1.),axis=1)-(.70*noise/scale)**2)).sum()))
    wav=F[:,2:7]
    detail=np.sqrt(max(0.,(sel*(2*(abs(wav[:,1:])**2).sum(axis=1)-.18*(noise/scale)**2)).sum()))
    pitchpower=(outer+.55*best)**4
    pitch=(pitchpower*pitches).sum()/(pitchpower.sum()+1.e-12)
    return max(best),pitch,innerbar,asym,detail

def tracearms(short,mask,cx,cy,q,t,R,peak,noise):
    # Follow individual bright spiral ridges (a Fourier quadrupole alone cannot
    # represent asymmetric arms). Along a logarithmic spiral compare the ridge
    # with equal-radius flanks, including its weaker half to reject point sources.
    a=np.arange(64)*2*np.pi/64
    s=np.linspace(np.log(.28),np.log(1.10),28)
    r=R*np.exp(s)[:,None]; c=np.cos(t); st=np.sin(t)
    px=cx+r*(c*np.cos(a)-q*st*np.sin(a)); py=cy+r*(st*np.cos(a)+q*c*np.sin(a))
    z=interp(short,px,py); m=interp(mask,px,py)
    med=np.median(z,axis=1,keepdims=True)
    z=m*z+(1-m)*med
    flanks=.5*(np.roll(z,4,axis=1)+np.roll(z,-4,axis=1))
    response=np.clip((z-flanks)/(np.maximum(med,0)+3.5*noise+.026*peak),-.70,.70)
    # Remove stationary angular residuals: a bar, disk ansae, and errors in the
    # viewing geometry are not spiral ridges.
    response-=.75*response.mean(axis=0)
    angleset=np.array([10.,15.,21.,28.,36.,47.]); signed=np.concatenate((angleset,-angleset))
    rate=1/np.tan(signed*np.pi/180)
    index=(np.arange(64)[None,None,:]+rate[:,None,None]*s[None,:,None]*64/(2*np.pi))%64
    lower=index.astype(int); frac=index-lower
    rows=np.arange(len(s))[None,:,None]
    val=response[rows,lower]*(1.-frac)+response[rows,(lower+1)%64]*frac
    power=np.zeros(len(signed))
    for k in (0,5,10):
        path=val[:,k:k+18]
        block=path.reshape(len(signed),3,6,64).mean(axis=2)
        evidence=.5*block.mean(axis=1)+.5*block.min(axis=1)
        positive=np.maximum(evidence,0)
        # Opposed arms support the same pitch, but don't require exact bilateral symmetry.
        evidence=positive+.45*np.roll(positive,32,axis=1)
        power=np.maximum(power,evidence.max(axis=1))
    paired=power.reshape(2,len(angleset))
    amps=paired.max(axis=0)-.4*paired.min(axis=0)
    mass=amps**5
    return (mass*angleset).sum()/(mass.sum()+1.e-12),amps.max()

def pitchangles(sm,mask,cx,cy,q,t,R,noise,peak):
    # Local tangent to residual light ridges in the deprojected disk. Derivatives
    # in (log r, theta) have equal metric factors. Subtract the radial profile
    # first, otherwise the gradient of a smooth circular disk implies zero pitch.
    n=64
    a=np.arange(n)*2*np.pi/n; c,s=np.cos(a),np.sin(a)
    sizes=np.exp(np.linspace(np.log(.22),np.log(1.1),25))
    dt=2*np.pi/n; dr=np.log(sizes[1]/sizes[0]); ct,st=np.cos(t),np.sin(t)
    rr=sizes[:,None]*R
    px=cx+rr*(ct*c-q*st*s); py=cy+rr*(st*c+q*ct*s)
    v=interp(sm,px,py); m=interp(mask,px,py)
    reflected=np.roll(v,n//2,axis=1)
    v=m*v+(1-m)*reflected
    level=v.mean(axis=1,keepdims=True)
    v=(v-level)/np.maximum(.02*peak+2*noise,level+2*noise)
    # Each localized orientation is computed from a structure tensor averaged
    # over a short arc and a small range in log radius, then cast as a pitch vote.
    gr=(v[2:]-v[:-2])/(2*dr)
    gt=(np.roll(v[1:-1],-1,axis=1)-np.roll(v[1:-1],1,axis=1))/(2*dt)
    tensors=[]
    for z in (gr*gr-gt*gt,2*gr*gt,gr*gr+gt*gt):
        temp=z+np.roll(z,-1,axis=1)+np.roll(z,1,axis=1)
        tensors.append((temp[:-2]+temp[1:-1]+temp[2:])/9.)
    vv,uv,total=tensors
    angle=.5*np.arctan2(uv,vv)
    aniso=np.hypot(vv,uv)
    intensity=np.maximum(level[2:-2],0.)
    # Lack of contrast or a masked source invalidates an orientation vote.
    w=ramp((aniso/(total+.001)-.4)/.45)*np.sqrt(aniso)*ramp((intensity-2*noise)/(5*noise+.025*peak))
    w*=m[2:-2]*m[1:-3]*m[3:-1]*ramp((v[2:-2]+.04)/.18)
    w*=((sizes[2:-2]>.37)&(sizes[2:-2]<1.05))[:,None]
    wt=w.sum()+.001
    # Positive and negative angles should not cancel to yield a 'tightly wound'
    # vote for pure noise. Use only their signed excess for evidence of arms.
    handedness=abs((w*np.sin(2*angle)).sum())/wt
    handedpitch=(w*np.clip(abs(angle)*180/np.pi,0,60)).sum()/wt
    return handedness,handedpitch

def linearbar(red,mask,cx,cy,q,t,R,qs,ts,signal,noise):
    # Search for a line whose two ends and flanks form a straight inner ridge.
    # Define the deprojection by the outermost bright disk, not by the bar itself.
    phasor=np.exp(2j*ts[:2])*(1-qs[:2]); t=.5*np.angle(2*phasor[0]+phasor[1])
    q=np.clip((2*qs[0]+qs[1])/3,.30,1.)
    theta=np.arange(36)*np.pi/36; c=np.cos(theta); s=np.sin(theta)
    ct,st=np.cos(t),np.sin(t)
    a=c*ct-s*q*st; b=c*st+s*q*ct; u=-s*ct-c*q*st; v=-s*st+c*q*ct
    found=0.
    for length in (.22,.34,.46,.58):
        L=length*R
        if L<2.6:
            continue
        along=L*np.array([.42,.66,.90,1.5,1.9])[:,None]
        # Bilateral evidence: a knot on just one side of the nucleus is not a bar.
        def both(al,ac):
            x=al*a+ac*u; y=al*b+ac*v
            p=interp(red,cx+x,cy+y); n=interp(red,cx-x,cy-y)
            mp=interp(mask,cx+x,cy+y); mn=interp(mask,cx-x,cy-y)
            return np.minimum(p*mp+n*(1-mp),n*mn+p*(1-mn))
        center=both(along,0.)
        # Compare with off-axis points at equal elliptical radius, not equal x:
        # otherwise a circular central concentration also produces a response.
        wing=.5*(both(along*.82,along*.57)+both(along*.82,-along*.57))
        relative=(center-wing)/(np.maximum(0.,center)+2.5*noise+.015*signal)
        base=.45*np.mean(relative[:3],axis=0)+.55*np.min(relative[:3],axis=0)
        termination=ramp((base-.50*np.mean(relative[3:],axis=0))/.17)
        evidence=base*(.50+.50*termination)
        # Below the minor-axis resolution limit line evidence is inconclusive.
        found=max(found,float(evidence.max())*ramp((L*q-1.2)/1.8))
    return found

def outertail(mid,mask,cx,cy,q,t,R,peak,noise):
    # Tidal debris can be too faint for an inner-isophote or texture measurement.
    # Integrate over radius first so that isolated noise peaks do not count, then
    # require a contiguous angular sector to extend past its diametrical opposite.
    c,s=np.cos(t),np.sin(t); r=np.linspace(.66,1.6,16)[:,None]*R
    px=cx+r*(c*ca-s*q*sa); py=cy+r*(s*ca+c*q*sa)
    z=interp(mid,px,py); m=interp(mask,px,py)
    opposite=np.roll(z,48,axis=1); valid=m*np.roll(m,48,axis=1)
    diff=(z-opposite)*valid
    clipped=np.clip(diff,-.18*peak,.18*peak)
    sector=clipped.mean(axis=0)
    sector=(sector+np.roll(sector,2)+np.roll(sector,-2)+np.roll(sector,4)+np.roll(sector,-4))/5
    base=np.maximum(z.mean(axis=0),0.)
    rel=sector/(base+.035*peak+2*noise)
    return np.max(rel)*ramp((np.max(sector)-1.5*noise)/(.03*peak+noise))

def weightedcolour(img, w):
    rgb=(img*w).sum(axis=(1,2))
    return (rgb[0]-rgb[2])/max(.001,rgb[0]+rgb[2])

def one(x):
    img=desky(x)
    spec=np.fft.rfft2(img,axes=(-2,-1))
    narrow=smooth(spec,0)
    # Broadband light: the display already gives significant weight to red light.
    gspec=.32*spec[0]+.44*spec[1]+.24*spec[2]
    fine=.32*narrow[0]+.44*narrow[1]+.24*narrow[2]
    mid=smooth(gspec,1); broad=smooth(gspec,2)
    noise=dev(fine)
    cx,cy=centre(fine,mid,noise)
    peak=max(interp(fine,cx,cy),3*noise,.02)
    mask,pair,paircandidate=maxima(fine,mid,broad,cx,cy,dev(mid),narrow)
    q,t,R,qs,ts,lens,drifts,resid,boxes=axes(fine,mask,cx,cy,noise,peak)
    ct,st=np.cos(t),np.sin(t)
    dx,dy=X-cx,Y-cy
    maj=dx*ct+dy*st; minor=-dx*st+dy*ct
    r=np.sqrt(maj**2+(minor/q)**2)/R
    nuc=(r<.26)*mask
    if nuc.sum()<2:
        nuc=(dx**2+dy**2<2.5**2)*mask
    ann=(r>.43)*(r<.97)*mask
    corecol=weightedcolour(narrow,nuc)
    diskcol=weightedcolour(narrow,ann)
    # Robust observed ratio of disk light to central brightness.
    disk=np.maximum(0.,fine-1.3*noise)
    light=(r<1.05)*mask*disk
    flux=light.sum()+1.e-6
    central=(light*(r<.32)).sum()/flux
    flatness=(np.sum(fine*ann)/(ann.sum()+.001))/peak
    arm,pitch,bar,odd,texture=angular(fine,mask,cx,cy,q,t,R,noise,peak)
    # Dust lanes and HII regions are more evident at short wavelengths. The
    # companion mask is from broadband light, not a per-band segmentation.
    short=.85*narrow[2]+.15*narrow[1]; ns=dev(short)
    shortpeak=max(interp(short,cx,cy),3*ns,.02)
    bluearm,bluepitch,bluebar,blueodd,bluetexture=angular(short,mask,cx,cy,q,t,R,ns,shortpeak)
    pitch=(pitch*arm+bluepitch*bluearm)/(arm+bluearm+.0001)
    arm=.55*arm+.45*max(arm,bluearm)
    handed,localpitch=pitchangles(.5*(fine+short),mask,cx,cy,q,t,R,.5*(ns+noise),peak)
    arcpitch,arcstrength=tracearms(short,mask,cx,cy,q,t,R,shortpeak,ns)
    # Colour of the envelope compared to the nucleus. A blue stellar population
    # in a broad shoulder around a compact central bulge is common in tight spirals.
    shoulder=ramp((flatness-.095)/.11)*ramp((corecol-diskcol-.04)/.16)
    blueregion=ramp((.06-diskcol)/.18)*ramp((.27-corecol)/.20)
    # Optional edge-on evidence: shape is diamond-like/thin instead of simply
    # elliptical; the inner isophotes can also be much rounder than the exterior.
    cross=(r<1.1)*mask*(np.abs(maj)>.36*R)*(np.abs(maj)<.80*R)
    z2=(cross*disk*minor**2).sum() / ((cross*disk).sum()+.001)
    thickness=np.sqrt(z2)/R
    # Measure the source in units of the seeing disk, not just the image width.
    # Bright, flat extended disks have a much larger half-peak area than cuspy
    # ellipticals even when the outermost visible isophotes subtend similar angles.
    resolved=np.sqrt(qs[3])*lens[3]
    aberr=.35*resid[0]+.40*resid[1]+.25*resid[2]
    diskiness=.5*(boxes[1]+boxes[2])
    roundcore=np.clip((qs[3]-q-.05)/.18,-1.,1.)
    projthin=np.clip((.32-q)/.10,-3.,2.)
    edge=projthin + .75*np.clip(np.log(R/22.),-1.,1.)+.60*roundcore
    edge+=.70*np.clip(diskiness/.065,-1.,1.)+.35*ramp((.08-thickness)/.045)
    edge+=.50*ramp((corecol-diskcol-.09)/.20)
    # Compare the mid-level isophotes with the outer disk after deprojection.
    # Under this affine transform an unbarred inclined disk becomes circular.
    cosa=np.cos(ts-t); sina=np.sin(ts-t)
    aa=cosa**2+(qs*sina)**2; bb=(sina**2+(qs*cosa)**2)/q**2
    ab=cosa*sina*(1-qs**2)/q
    deproj=(aa-bb+2j*ab)/(aa+bb+.00001)
    middle=.5*(deproj[2]+deproj[3]); innerell=abs(middle)
    barshape=innerell*ramp((1-qs[2])/.32)*ramp((lens[2]-3)/5)
    # A fixed phase at high surface brightness (the bar), terminated by a twist
    # or loss of ellipticity in the outer disk; not the circular seeing core of
    # a very inclined disk.
    # Bars trace the old stellar population. Require the central isophotes to
    # have coherent alignment, as opposed to the rapid twist of open spiral arms.
    redimg=.8*narrow[0]+.2*narrow[1]
    redpeak=max(.02,interp(redimg,cx,cy)); rednoise=dev(redimg)
    rb=[]; rpa=[]
    for lev in (.24,.40,.58):
        region=component((redimg>max(lev*redpeak,4*rednoise))&(mask>.35),cx,cy)
        ww=mask*region*ramp((redimg-lev*redpeak)/(.08*redpeak+rednoise))
        b,a,length=tensor(ww,maj,minor/q,0.)
        rb.append((1-b**2)/(1+b**2)); rpa.append(a)
    coherent=abs(np.mean(np.array(rb)*np.exp(2j*np.array(rpa))))
    redbar=coherent*ramp((1-qs[2])/.27)*ramp((lens[2]-2.)/4.)
    blueonly=ramp((bluearm-arm)/.06)
    barlike=(.56*redbar/.37+.26*barshape/.39+.18*bar/.085)*ramp((q-.23)/.17)
    barlike*=1-.22*blueonly
    barline=linearbar(redimg,mask,cx,cy,q,t,R,qs,ts,redpeak,rednoise)
    barlike=max(barlike,.70*barline/.20+.30*barlike)
    chrom=ramp((corecol-diskcol-.06)/.18)
    blue=ramp((.12-diskcol)/.24)
    symmetrybreak=arm/.055+.25*ramp((texture-.060)/.15)
    irregularoutline=ramp((aberr-.12)/.18)*ramp((q-.29)/.3)
    diskbody=np.clip(np.log(max(resolved,1.)/4.5),-1.5,1.5)
    spiral=symmetrybreak+.68*diskbody+.36*chrom+.24*blue+.45*irregularoutline-1.48
    spiral+=.42*ramp((barlike-.65)/.60) - .56*ramp((.50-q)/.22)
    spiral+=.45*shoulder+.42*blueregion+.60*ramp((handed-.18)/.38)
    if q>.43:
        spiral=max(spiral, (barlike-1.)+.45*diskbody+.55*shoulder)
    bluefeatures=ramp((bluearm-.035)/.065)*ramp((.15-diskcol)/.20)
    # Tidal tails and displacements of the outer optical centroid relative to
    # the nucleus, including the case of one long extension along the major axis.
    drift=.45*drifts[0]+.35*drifts[1]+.20*drifts[2]
    # If the eccentric red and blue isophotes wind with the same handedness, a
    # one-sided bright arm need not signal an interaction.
    disturb=max((odd-.18)/.09,(drift-.105)/.06)+.55*irregularoutline-.35*ramp(arm/.09)
    disturb-=.6*ramp((handed-.18)/.35)*bluefeatures
    tidal=outertail(mid,mask,cx,cy,q,t,R,peak,dev(mid))
    disturb=max(disturb,(tidal-.49)/.20-.65*bluefeatures)
    # In a lopsided host, a resolved companion may already have stripped most
    # of the bridge; in a smooth host insist on independent evidence of contact.
    merge=(max(pair,paircandidate*(.62+.38*ramp((disturb+.18)/1.5)))-.52)/.19
    if merge>0. and merge>max(disturb-1.,0.)*.30:
        return 1
    if disturb>0. and disturb>max(0.,edge-.5)*.5 and disturb>max(0.,spiral)*.56:
        return 0
    if edge>0. and (q<.40 or edge>spiral+.2):
        # Classical bulges are concentrated, rounder, redder, and brighter than
        # a pure edge-on disk, while a long blue spindle favours a bulgeless disk.
        bulge=(corecol-.14)/.16 + .65*(q-.20)/.10
        bulge+=.8*np.clip((qs[3]-q-.055)/.15,-1,1) + .73*np.clip((central-.34)/.12,-1,1)
        bulge+=.55*np.clip((diskcol-.035)/.16,-1,1)-.53*np.clip(np.log(R/25.),-1,1)
        return 9 if bulge>0 else 8
    if spiral>0.:
        if barlike>1. and (bar>.62*arm or redbar>.44 or barline>.21):
            return 5
        spectral=.55*localpitch+.45*pitch
        arcweight=.65*ramp(arcstrength/.10)
        looseness=((1-arcweight)*spectral+arcweight*arcpitch-24.)/10.
        looseness+=.55*np.clip((arm-.063)/.065,-1.,2.)+.40*np.clip((bluetexture-.10)/.12,-1.,2.)
        looseness+=.52*ramp((handed-.20)/.34)-.45*ramp((.05-arm)/.035)
        return 7 if looseness>0 else 6
    # Smooth projected spheroids: use the middle isophotes, avoiding both the
    # circularizing PSF in the core and faint companions at very low isophotes.
    smoothq=.38*qs[1]+.37*qs[2]+.25*qs[3]
    return 2 if smoothq>=.79 else (3 if smoothq>.46 else 4)

def predict(x):
    return np.array([one(image) for image in x],dtype=np.int64)
