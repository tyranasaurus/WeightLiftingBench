import numpy as np
import math

# The same angular samples are used for every source, independent of the sky PA.
tau = 2*np.pi
ang = tau*np.arange(128)/128
cs = np.cos(ang)
sn = np.sin(ang)
h2 = np.exp(2j*ang)
rays = np.arange(0,83,.5)[:,None]
rayx = rays*cs
rayy = rays*sn
ygrid,xgrid = np.mgrid[:128,:128]
dcenter = np.hypot(xgrid-63.5,ygrid-63.5)
sky = dcenter[::2,::2]>50
freq = np.fft.fftfreq(128)[:,None]**2+np.fft.rfftfreq(128)[None,:]**2
def gauss(s):
    return np.exp(-2*np.pi**2*s*s*freq)
small = gauss(1.05)
broad = gauss(2)
frac = np.array([.72,.50,.33,.21,.13,.08])
logr = np.linspace(np.log(.075),np.log(1.7),64)
rad = np.exp(logr)
wind = np.array([.75,1.05,1.45,2,2.8,4,5.6,8,11.3])
spins = np.concatenate((wind,-wind))
spiralwaves = np.exp(1j*np.arange(1,5)[:,None,None]*spins[None,:,None]*logr[None,None,:])

def clip(z):
    return np.clip(z,0,1)

def grade(z,low,high):
    return clip((z-low)/(high-low))

def sample(im,x,y):
    x=np.clip(x,0,126.9999)
    y=np.clip(y,0,126.9999)
    j=x.astype(np.int32)
    i=y.astype(np.int32)
    u=x-j
    v=y-i
    return (1-v)*((1-u)*im[...,i,j]+u*im[...,i,j+1])+v*((1-u)*im[...,i+1,j]+u*im[...,i+1,j+1])

def meanw(a,w):
    return np.sum(w*a)/(1.e-9+np.sum(w))

def smoothrings(a):
    return (np.roll(a,1,axis=0)+2*a+np.roll(a,-1,axis=0))*.25

def isoradii(im,cx,cy,levels):
    z = sample(im,cx+rayx,cy+rayy)
    # First outward crossing of each isophote, with subpixel interpolation.
    hit = z[None,:,:]<levels[:,None,None]
    hit[:,-1,:]=True
    k=hit.argmax(1)
    j=np.arange(len(cs))
    prev=z[np.maximum(0,k-1),j]
    val=z[k,j]
    r=.5*(np.maximum(0,k-1)+clip((prev-levels[:,None])/np.maximum(prev-val,.00001)))
    return np.maximum(.65,r)

def ellipse(r):
    # For an elliptical lamina the second harmonic of its angular area
    # density has modulus (a-b)/(a+b). Opposite rays suppress neighbours.
    back=np.roll(r,64,axis=1)
    sym=.7*np.minimum(r,back)+.3*np.sqrt(r*back)
    t=sym*sym
    for step in range(2):
        comp=(t*h2).sum(1)/t.sum(1)
        phi=np.angle(comp)*.5
        e=np.minimum(.91,abs(comp))
        q=(1-e)/(1+e)
        a=np.sqrt(t.mean(1)/q)
        cp=np.cos(ang-phi[:,None])
        sp=np.sin(ang-phi[:,None])
        model=a[:,None]/np.sqrt(cp*cp+(sp/q[:,None])**2)
        t=np.minimum(sym,model*1.26)**2
    return a,q,phi,model

def field(im,noise):
    x,y=63.5,63.5
    # Locate the bright core with a prior on the given cutout centre.
    merit=im*np.exp(-.5*(np.maximum(0,dcenter-3)/9)**2)
    i,j=np.unravel_index(np.argmax(merit),merit.shape)
    x=float(j); y=float(i)
    for step in range(4):
        ds=(xgrid-x)**2+(ygrid-y)**2
        w=np.maximum(0,im-.45*im[i,j])*np.exp(-ds/12)
        w*=ds<49
        x+=np.clip(meanw(xgrid-x,w),-1,1)
        y+=np.clip(meanw(ygrid-y,w),-1,1)
    return x,y

def neighbour(im,wide,colors,cx,cy,peak,noise,a,q,phi):
    # Look for two resolved cores with plausible relative luminosity,
    # separation and a shared envelope. A diffraction/seeing-sized isolated
    # maximum provides much weaker evidence than an extended one.
    dist=np.hypot(xgrid-cx,ygrid-cy)
    mask=(dist>3.2)&(dcenter<62)&(im>max(.12*peak,5*noise,.025))
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)):
        mask &= im>=np.roll(im,(dy,dx),axis=(0,1))
    yy,xx=np.nonzero(mask)
    if len(xx)==0:
        return 0.
    order=np.argsort((im[yy,xx]/(1+dist[yy,xx]/16)**1.7))
    best=0.
    primary=sample(colors,cx,cy)
    hue=(primary[0]-primary[2])/(.03+primary[1])
    for idx in order[-18:]:
        x=xx[idx]; y=yy[idx]; level=im[y,x]; d=dist[y,x]
        t=np.linspace(0,1,31)
        path=sample(im,x+t*(cx-x),y+t*(cy-y))
        valley=path.min()/max(.01,min(level,peak))
        # Radial wing excluding the half-plane facing the main galaxy.
        side=(cs*(x-cx)+sn*(y-cy))>-.2*d
        rs=np.array([2.,3.5,5.,8.,12.])[:,None]
        ring=sample(im,x+rs*cs[side],y+rs*sn[side])
        profile=np.maximum(0,np.median(ring,axis=1))/(level+.00001)
        extent=.5*(profile[1]+profile[2])
        res=grade(extent,.14,.55)
        col=colors[:,y,x]
        hue2=(col[0]-col[2])/(.03+max(0,col[1]))
        similar=1-grade(abs(hue2-hue),.35,1.3)
        blueknot=grade(-hue2,.18,.55)*grade(hue-hue2,.18,.65)
        a2=2+4*profile[1]+6*profile[2]+8*profile[3]
        bright=grade(level/(peak+.001),.14,.65)
        excess=grade(1-valley,.018,.32)
        # Even adjacent cores should produce a saddle unless strongly unequal.
        connected=grade(valley,.035,.42)
        rel=d/max(5.,.76*a+a2)
        close=1-grade(rel,.7,2.4)
        offcenter=grade(np.hypot(cx-63.5,cy-63.5),2,10)
        fluxratio=level/(peak+.001)*(a2/max(3,.68*a*np.sqrt(q)))**1.15
        evidence=(.28*bright+.25*res+.24*close+.12*connected+.08*similar+.13*grade(fluxratio,.12,.65)+.09*offcenter)
        evidence -= .37*(1-excess)+.16*blueknot+.25*(1-close)+.18*(1-res)*(1-connected)+.14*(1-bright)
        saddle=grade(1-valley,.04,.34)
        # Discount a knot on a nearly edge-on ridge and unrelated objects
        # with no detectable envelope bridging back to the primary.
        alignment=abs((x-cx)*np.cos(phi)+(y-cy)*np.sin(phi))/d
        evidence-=.24*grade(.49-q,.0,.23)*grade(alignment,.87,.98)*(1-.75*saddle)
        evidence-=.25*grade(.14-valley,0,.13)*(1-.70*offcenter)
        evidence-=.12*grade(d,14,32)*(1-offcenter)+.18*(1-saddle)+.10*(1-similar)
        evidence-=.22*grade(valley,.4,.9)*grade((hue-hue2),.1,.5)
        # A resolved peak should rise above its surroundings on both axes,
        # unlike a patch of clumpy disk. Contrast on a small circle is
        # measured relative to the saddle and the brightness of its host.
        ringcore=sample(im,x+3.5*cs,y+3.5*sn)
        prominence=(level-np.median(ringcore))/(min(peak,level)+.01)
        dx0=cx-63.5; dy0=cy-63.5
        bracket=grade(-(dx0*(x-63.5)+dy0*(y-63.5))/(d+.001),.4,3)
        inside=grade(valley,.12,.55)*grade(1-valley,.014,.19)*grade(prominence,.06,.21)
        evidence+=.13*bracket+.11*inside*similar+.065*res*bright*similar
        evidence-=.12*grade(.17-prominence,0,.13)*(1-bracket)
        # A bright object in the sky beyond the central source, even if
        # resolved, is weak evidence unless it distorts or overlaps it.
        independent=grade(d/max(a*.72+a2,5),.68,1.8)*grade(.28-valley,.0,.22)
        evidence-=.28*independent*(1-.85*max(offcenter,bracket))
        # Seeing-limited peak on an existing disk: compare its local
        # prominence across scales, not just the galaxy's shared envelope.
        wing=sample(im,x+6.5*cs,y+6.5*sn)
        ped=max(0,np.quantile(wing,.35))
        cwidth=(np.median(ringcore)-ped)/max(.01,level-ped)
        evidence-=.17*grade(.37-cwidth,0,.23)*(1-.72*bracket)
        if q<.40 and alignment>.92 and not bracket>.2:
            evidence-=.15*(1-grade(prominence,.24,.56))
        # If the primary's faint isophotes are undisturbed on the
        # companion side, a projected alignment is a better explanation.
        t=np.linspace(.22,.72,16)
        bridge=sample(wide,cx+t*(x-cx),cy+t*(y-cy))
        opposite=sample(wide,cx-t*(x-cx),cy-t*(y-cy))
        excesslight=np.median((bridge-opposite)/(np.maximum(.02,opposite)+.06*peak+2*noise))
        evidence-=.19*(1-grade(excesslight,.08,.8))*(1-offcenter)*(1-grade(d,5,8)*inside*.45)
        if excesslight>.38 and valley>.08:
            evidence+=.10*grade(prominence,.13,.36)*similar
        best=max(evidence,best)
    return best

def ridges(light,blue,noise,good):
    # Isotropic Gaussian Hessians in (log radius, azimuth): both axes
    # correspond locally to equal distances in the deprojected disk.
    # Negative principal curvature is transverse to a luminous ridge.
    base=.55*light+.45*np.minimum(blue,2*light+3*noise)
    z=np.log(np.maximum(.004,base+3*noise))
    z-=np.median(z,axis=1)[:,None]
    z=np.clip(z,-.9,1.)
    n=len(rad)
    ft=np.fft.rfft2(np.concatenate((z,z[::-1]),axis=0))
    fx=tau*np.fft.rfftfreq(len(cs),d=tau/len(cs))[None,:]
    fy=tau*np.fft.fftfreq(2*n,d=logr[1]-logr[0])[:,None]
    win=good[:,None]*grade(rad,.32,.58)[:,None]*(1-grade(rad,1.05,1.5))[:,None]
    weights=[]; steep=[]; curls=[]; strength=[]
    for sigma in (.105,.21):
        gft=-sigma**2*ft*np.exp(-.5*sigma**2*(fx*fx+fy*fy))
        xx=np.fft.irfft2(gft*fx**2,s=(2*n,len(cs)))[:n]
        yy=np.fft.irfft2(gft*fy**2,s=(2*n,len(cs)))[:n]
        xy=np.fft.irfft2(gft*fx*fy,s=(2*n,len(cs)))[:n]
        delta=np.hypot(xx-yy,2*xy)+.000001
        c=.5*(delta-xx-yy)
        other=.5*(delta+xx+yy)
        line=np.maximum(0,c-.65*abs(other))
        # cos(2*pitch) positive for circumferential arms.
        p=np.arccos(np.clip((xx-yy)/delta,-1,1))*.5
        deg=p*180/np.pi
        w=win*np.minimum(line,.22)**2*grade(.85-z,0,.3)*(1-grade(deg,56,79))
        weights.append(w.sum()+.00001)
        steep.append(np.sum(w*grade(deg,12,32)))
        curls.append(np.sum(w*np.sign(xy)*np.sin(2*p)))
        strength.append(np.sum(line*win)/(len(cs)*np.sum(win)+.00001))
    return sum(steep)/sum(weights),abs(sum(curls))/sum(weights),max(strength)

def contours(color,rr,a,q,phi,cx,cy,noise,peak):
    # Compare equal-brightness locations instead of selected apertures.
    # The test remains valid when ellipticity or the radial light law varies.
    # Estimate noise from the disagreement of neighbouring contours, keeping
    # only angular harmonics supported across a range of brightness.
    shapes=[]; hues=[]
    for j in range(len(a)):
        rel=ang-phi[j]
        model=a[j]/np.hypot(np.cos(rel),np.sin(rel)/q[j])
        rat=rr[j]/model
        # Angles running through neighbouring stars terminate too late.
        rat=np.clip(rat,.52,1.70)
        h=sample(color,cx+rr[j]*cs,cy+rr[j]*sn)
        hue=(h[2]-h[0])/(.08*peak+np.maximum(0,h.mean(0))+3*noise)
        shapes.append(np.fft.rfft(rat)/len(cs)*2)
        hues.append(np.fft.rfft(np.clip(hue,-1.5,1.5))/len(cs)*2)
    rs=np.array(shapes); hs=np.array(hues)
    # A coherent departure from the current ellipse in adjacent layers.
    shap=np.sqrt(np.maximum(0,np.sum((rs[2:5,3:10]*rs[3:6,3:10].conjugate()).real,axis=1))).mean()
    patch=np.sqrt(np.maximum(0,np.sum((hs[1:4,1:8]*hs[2:5,1:8].conjugate()).real,axis=1))).mean()
    return shap,patch

def armgeometry(light,wide,cx,cy,radius,axis,pa,weight,noise):
    # On an unwrapped log-polar surface, bright arm ridges are diagonal.
    # Subtract local radial background first, retaining the two sides of
    # each ridge. Measure ridge direction by the flux-gradient tensor.
    u=np.cos(pa)*cs-np.sin(pa)*axis*sn
    v=np.sin(pa)*cs+np.cos(pa)*axis*sn
    sm=sample(wide,cx+radius*rad[:,None]*u,cy+radius*rad[:,None]*v)
    z=.6*light+.4*sm
    z=np.log(np.maximum(.004,z+3*noise))
    z-=np.mean(z,axis=1)[:,None]
    z-=.25*(np.roll(z,4,axis=0)+np.roll(z,-4,axis=0)+np.roll(z,7,axis=0)+np.roll(z,-7,axis=0))
    dt=(np.roll(z,-2,axis=1)-np.roll(z,2,axis=1))/(4*tau/len(cs))
    dr=(np.roll(z,-1,axis=0)-np.roll(z,1,axis=0))/(2*(logr[1]-logr[0]))
    w=weight*grade(rad,.32,.53)*(1-grade(rad,1.0,1.25))
    cr=dt*dr
    # Limit influence of point source edges; use near-ridge pixels where
    # the arms contribute measurable light.
    wt=w[:,None]/(1+(dt**2+dr**2)/3)**.7
    pos=np.sum(np.maximum(0,cr)*wt); neg=np.sum(np.maximum(0,-cr)*wt)
    sign=1 if pos>neg else -1
    wt=wt*np.maximum(0,cr*sign)
    pitch=np.sum(np.arctan2(abs(dt),abs(dr)+.001)*wt)/(np.sum(wt)+.0001)
    chir=abs(pos-neg)/(pos+neg+.0001)
    return pitch*180/np.pi,chir

def thinprofile(im,cx,cy,a,q,pa):
    # Vertical half-light thickness and mid-plane brightness along the spine.
    # A spheroidal bulge gives a thickness excess over a pure disk.
    height=np.linspace(-1.6,1.6,81)[None,None,:]*max(2.,a*q)
    p=np.array([.0,.12,.29,.47,.68,.85])[:,None,None]*a*np.array([-1,1])[None,:,None]
    co=np.cos(pa); si=np.sin(pa)
    val=sample(im,cx+p*co-height*si,cy+p*si+height*co)
    val=np.maximum(0,val-np.quantile(val,.08,axis=2)[:,:,None])
    mass=val.sum(2)+.0001
    core=val.max(2)+.0001
    width=(mass/core).mean(1)
    # Width normalized by the interior/wing ratio for an elliptical
    # Gaussian (constant) or exponential (width increases outward).
    swelling=width[:2].mean()/max(.5,width[3:5].mean())
    mh=core.mean(1)
    pt=mh[0]/max(.01,mh[2]*(mh[2]/max(.01,mh[4]))**(.29/(.68-.29)))
    return swelling,pt

def morphology(raw):
    ft=np.fft.rfft2(raw,axes=(-2,-1))
    color=np.fft.irfft2(ft*small, s=(128,128),axes=(-2,-1))
    bg=np.median(color[:,::2,::2][:,sky],axis=1)
    color=color-bg[:,None,None]
    im=.30*color[0]+.42*color[1]+.28*color[2]
    wide=np.fft.irfft2((.30*ft[0]+.42*ft[1]+.28*ft[2])*broad,s=(128,128))-(.30*bg[0]+.42*bg[1]+.28*bg[2])
    noise=max(.0015,np.median(abs(im[::2,::2][sky]))*1.4826)
    ox,oy=field(wide,noise)
    cx,cy=ox,oy
    peak=max(.01,float(sample(im,cx,cy)))
    levels=np.minimum(.85*peak,np.maximum(frac*peak,2.8*noise))
    # Recentre on inner isophotes rather than an individual bright pixel.
    rr=isoradii(im,cx,cy,levels)
    a,q,phi,model=ellipse(rr)
    r=np.minimum(rr[:3],1.45*model[:3]); t=r**2
    dx=np.mean(2*(t*r*cs).sum(1)/(3*t.sum(1)))
    dy=np.mean(2*(t*r*sn).sum(1)/(3*t.sum(1)))
    k=min(1.,3/(np.hypot(dx,dy)+.001))
    cx+=k*dx; cy+=k*dy
    rr=isoradii(im,cx,cy,levels)
    a,q,phi,model=ellipse(rr)
    peak=max(.01,float(sample(im,cx,cy)))
    # Outer disk geometry as a complex ellipticity so angles can straddle pi.
    de=(1-q)/(1+q)*np.exp(2j*phi)
    disk=.20*de[3]+.48*de[4]+.32*de[5]
    pa=.5*np.angle(disk)
    axis=(1-abs(disk))/(1+abs(disk))
    radius=max(3.,min(72.,.65*a[4]+.35*a[5]))
    # Angular maps after deprojection of the outer envelope.
    co=np.cos(pa); si=np.sin(pa)
    u=co*cs-si*sn*axis; v=si*cs+co*sn*axis
    rho=radius*rad[:,None]
    polar=sample(color,cx+rho*u,cy+rho*v)
    light=.30*polar[0]+.42*polar[1]+.28*polar[2]
    med=np.median(light,axis=1)
    mad=np.median(abs(light-med[:,None]),axis=1)
    # Suppress isolated stars in annular statistics, but retain extended arms.
    symmetric=np.roll(light,64,axis=1)
    tailw=rad**1.5*grade(rad,.65,1.1)
    floor=np.maximum(.035*peak,3.5*noise)
    tide=np.maximum(light-np.maximum(0,symmetric)-1.6*noise,0)/(np.maximum(0,med)+floor)[:,None]
    # A tidal tail occupies contiguous angles at large radii: a low
    # quantile of the excess over a wedge ignores foreground points.
    th=np.minimum.reduce([np.roll(tide,k,axis=1) for k in range(-6,7,2)])
    tail=meanw(np.minimum(th.max(axis=1),3),tailw)
    cap=med+np.maximum(.72*np.maximum(med,0),4.5*mad+2*noise)
    light=np.minimum(light,cap[:,None])
    mu=light.mean(1)
    scaled=(light-mu[:,None])/np.maximum(mu+2*noise,.01)[:,None]
    modes=smoothrings(np.fft.rfft(scaled,axis=1)[:,1:11]*2/len(cs))
    amp=abs(modes)
    good=grade(mu/noise,1.8,6)*grade(rad*radius,1.8,5)
    inner=good*grade(rad,.12,.26)*(1-grade(rad,.43,.65))
    outer=good*grade(rad,.37,.62)*(1-grade(rad,.95,1.26))
    middle=good*grade(rad,.22,.4)*(1-grade(rad,.9,1.2))
    # Odd pattern and high-order angular structure, corrected for noise.
    scatter=amp**2
    nmode=(2*noise/np.maximum(.01,mu+2*noise))**2/np.maximum(4.,tau*radius*rad*np.sqrt(axis)/4)
    odd=np.sqrt(max(0,meanw(scatter[:,0]+scatter[:,2]+scatter[:,4]-3*nmode,middle)))
    lop=meanw(amp[:,0],outer)
    detail=np.sqrt(max(0,meanw(np.sum(scatter[:,2:8],axis=1)-6*nmode,outer)))
    sub=np.sqrt(max(0,meanw(np.sum(scatter[:,2:8],axis=1)-6*nmode,inner)))
    twist=meanw(abs(modes[:,1]-meanw(modes[:,1],middle)),middle)
    # Match logarithmic spirals and their mirror images in separate radial
    # windows. A bar or a fixed error in inclination favours neither hand.
    evidence=np.zeros((4,len(wind)))
    signed=np.zeros((4,len(wind)))
    for centre in (.47,.78):
        weight=good*np.exp(-.5*(np.log(rad/centre)/.34)**2)
        signal=modes[:,:4].T*weight
        response=abs(np.sum(signal[:,None,:]*spiralwaves,axis=2))/(weight.sum()+.001)
        plus=response[:,:len(wind)]; minus=response[:,len(wind):]
        signed+=plus-minus
        evidence+=abs(plus-minus)
    chiral=(evidence+abs(signed))*.25
    # Multi-arm consistency helps distinguish spiral patterns from lopsidedness.
    spectrum=np.sqrt(np.sum((chiral*np.array([.50,1,.85,.70])[:,None])**2,axis=0))
    winding=np.max(spectrum)
    pitch=sum(spectrum**3*np.log(wind))/(sum(spectrum**3)+.000001)
    # Coherent m=2 mode in the bright inner disk, discounted if the major
    # axis is just that of a spherical bulge projected back into the disk.
    bar=0.
    for centre in (.28,.40,.53):
        w=good*np.exp(-.5*(np.log(rad/centre)/.24)**2)
        m2=meanw(modes[:,1],w)
        coherent=abs(m2)/(meanw(amp[:,1],w)+.0001)
        b=abs(m2)*(.5+.5*coherent**2)
        if m2.real<0:
            b*=1-.72*grade(1-axis,.13,.57)*(-m2.real)/(abs(m2)+.00001)
        # Check that the high quadrupole does not simply continue to the edge.
        far=meanw(modes[:,1],outer)
        if abs(m2)>.001:
            b-=.28*max(0,(far*m2.conjugate()).real/abs(m2))
        bar=max(bar,b)
    # Colours of disk and bulge; cut bright contaminants before pooling.
    colors=np.median(polar,axis=2)
    diskweight=outer*np.maximum(mu,.01)**.3
    diskred=meanw(colors[0]-colors[2],diskweight)/(meanw(colors[1],diskweight)+.025)
    corew=(1-grade(rad,.15,.32))
    corered=meanw(colors[0]-colors[2],corew)/(max(0,meanw(colors[1],corew))+.04)
    gradient=corered-diskred
    # The area growth curve at fixed fractions of the core brightness.
    area=a*np.sqrt(q)
    flat=area[1]/max(1.,area[4])
    qbody=.25*q[1]+.4*q[2]+.35*q[3]
    qvis=.53*qbody+.47*axis
    # Shape of isophotes in their own elliptical coordinates: positive
    # cos(4 theta) means pointed, disky ends rather than an exact ellipse.
    diskiness=[]
    irregular=[]
    for j in (1,2,3,4):
        theta=(np.arctan2(sn*q[j],cs)+phi[j])%tau
        i=theta/tau*len(cs); p=i.astype(int)%len(cs); h=i-np.floor(i)
        r=(rr[j,p]*(1-h)+rr[j,(p+1)%len(cs)]*h)*np.sqrt(cs**2+(sn*q[j])**2)**-1/(a[j]+.001)
        r=np.clip(r,.4,1.8)
        diskiness.append(2*np.mean(r*np.cos(4*ang)))
        irregular.append(np.mean(abs(r-np.roll(r,64))))
    pointed=np.mean(diskiness)
    offset=np.mean(irregular)
    merger=neighbour(im,wide,color,ox,oy,peak,noise,radius,axis,pa)
    # Fingerprints of a stellar bar: stretch in the isophotes of the
    # red population in the inner disk, relative to the outer projection.
    rp=max(.01,float(sample(color[0],cx,cy)))
    rb=isoradii(color[0],cx,cy,np.maximum(np.array([.64,.48,.32])*rp,3*noise))
    ar,qr,pr,mr=ellipse(rb)
    bshape=0.
    for j in range(3):
        da=pr[j]-pa
        # Eigenvalue contrast after deprojecting this ellipse. Ignore
        # elongation along the projected minor axis of a tilted disk,
        # expected even for an axisymmetric central bulge.
        C=np.cos(da); S=np.sin(da); Q=qr[j]**2
        xx=C*C+Q*S*S; yy=(S*S+Q*C*C)/axis**2; xy=(1-Q)*S*C/axis
        em=np.hypot(xx-yy,2*xy)/(xx+yy)
        elong=1-np.sqrt((1-em)/(1+em))
        elong*=1-grade(qr[j],.72,.93)*.8
        elong*=1-.88*grade((yy-xx)/(np.hypot(xx-yy,2*xy)+.0001),.2,.98)*grade(1-axis,.08,.48)
        bshape=max(bshape,elong*grade(ar[j],2,4))
    coreblue=grade(-corered,-.02,.40)
    # Faint, unilateral extensions are measured in paired wedges around
    # several deep isophotes, rather than relative to a bright core only.
    wpeak=max(.02,float(sample(wide,cx,cy)))
    nw=max(.001,np.median(abs(wide[::2,::2][sky]))*1.4826)
    lowr=isoradii(wide,cx,cy,np.maximum(np.array([.17,.09,.05,.03])*wpeak,2.5*nw))
    lr=np.roll(lowr,64,axis=1)
    paired=(lowr-lr)/(lowr+lr)
    pairmed=np.mean(np.abs(paired),axis=1)
    # Difference from the low-level photocentre in units of isophotal radius.
    unilateral=np.mean(np.abs((paired*(cs+1j*sn)).mean(1)))*2
    tailshape=np.mean(pairmed)+.5*unilateral
    # A skewed envelope is tidal if asymmetric components dominate any
    # coherent two-armed pattern, not simply because all arms are bright.
    ac=abs(meanw(modes[:,0],outer))
    nonspiral=grade((ac+.06)/(winding+bar*.28+.05),1.2,3.2)
    degrees,hand=armgeometry(light,wide,cx,cy,radius,axis,pa,good,noise)
    shrink,bump=thinprofile(im,cx,cy,radius,axis,pa)
    # In blue light, star-forming arms may have high contrast even when
    # there is little variation in the sum of the three channels.
    bl=polar[2]; bm=colors[2]; dev=bl-bm[:,None]
    bs=np.median(abs(dev),axis=1)
    clipped=np.minimum(dev,np.maximum(.65*np.maximum(bm,0),4.5*bs+3*noise)[:,None])
    bc=smoothrings(np.fft.rfft(clipped,axis=1)[:,1:7]*2/len(cs))/np.maximum(.025,bm+3*noise)[:,None]
    bluew=outer*grade(bm/noise,3,9)
    structureB=np.sqrt(max(0,meanw(np.sum(abs(bc[:,2:])**2,axis=1)-8*nmode,bluew)))
    broadB=meanw(abs(bc[:,1]-meanw(bc[:,1],middle)),middle)
    misalign=abs(de[1]-de[4])
    # Profile curvature. Ignoring seeing, log(radius) grows as n times
    # log(log(peak/level)) for a Sersic profile. Extended wings and a
    # rapidly fading resolved disk are distinct even at the same shape.
    wingindex=np.log((area[5]+.01)/(area[2]+.01))/np.log(np.log(1/.08)/np.log(1/.33))
    # Check whether the bright disk continues outwards in an easily
    # visible segment of each spiral arm. An open two-arm pattern makes
    # its face-on envelope distinctly less circular than its nucleus.
    envelope=abs(de[4]-de[2])
    faintblue=grade(peak,.57,.25)*grade(corered,.10,-.22)
    # A very small apparent PSF-smeared core is not by itself evidence
    # for a bar, arm or bulge; consider the brightness of its disk too.
    resolved=grade(area[1],2.6,5.3)
    clear=grade(area[4],6.,15.)
    extended=.55*resolved+.45*clear
    warmcore=grade(corered,-.05,.48)
    tidal=grade(tail,.25,1.25)

    # Evidence comparisons are specified in human-interpretable units:
    # axis ratios, fractional Fourier amplitudes, relative colours and sizes.
    scale=np.log2(max(radius,3)/14.)
    blue=grade(-diskred,-.03,.48)
    red=grade(diskred,.04,.53)
    sf=grade(gradient,.02,.48)
    structured=grade(detail,.075,.26)
    arm=grade(winding,.038,.175)
    asym=grade(odd,.13,.52)
    lowconc=grade(flat,.38,.62)

    # Colour gradients can be introduced by the display stretch even in a
    # red featureless elliptical. Require blue *outer light*, or structure
    # above noise, before treating a gradient as stellar population evidence.
    pop=sf*(.20+.80*grade(.24-diskred,.0,.37))
    skyshape=grade(tailshape,.08,.30)
    # Compare coherent isophote ripples and population patches against
    # the axisymmetric early-type hypothesis. Also measure the physical
    # thickness of the bright lens, less affected by a small companion.
    shapeharm,huepatch=contours(color,rr,a,q,phi,cx,cy,noise,peak)
    armangle,hchir,ridge=ridges(light,polar[2],noise,good)
    nebula=grade(huepatch,.040,.18)
    ripples=grade(shapeharm,.034,.145)
    hand=max(hand,hchir)
    if merger>.86:
        return 1
    # Thin disks can have strong apparent asymmetry from midplane dust;
    # do not confuse that with tidal structure.
    edge=(.34-min(axis+.015,qvis))/.14+.95*grade(pointed,.012,.08)+.55*grade(qbody-axis,.035,.18)+.55*np.clip(scale,-1.3,1.6)+.19*blue+.22*structured
    swelling=grade(shrink,1.08,1.40)
    edge+=.60*swelling+.33*grade(bump,1.08,1.48)-.62*arm-.40*grade(hand,.18,.45)
    # Bulged edge-ons can have almost spheroidal inner contours. Test
    # whether their bright centre stands above the inward extrapolation
    # of the disk seen edge-on from each of its two sides.
    spine=sample(im,cx+np.array([-.65,-.40,-.22,.22,.40,.65])*radius*co,cy+np.array([-.65,-.40,-.22,.22,.40,.65])*radius*si)
    extrap=np.sqrt(np.prod(np.maximum(.01,spine[[1,4]])))*((np.prod(np.maximum(.01,spine[[1,4]]))/np.prod(np.maximum(.01,spine[[0,5]])))**.8)
    excessbulge=grade(peak/(extrap+.02),1.18,2.4)
    edge+=.81*excessbulge*grade(qbody-axis,-.02,.12)*grade(pointed,-.01,.05)+.43*warmcore*grade(pointed,.017,.075)*grade(radius,10,19)
    if edge>.47 and axis<.51 and qvis<.55:
        # A bulge swells the inner isophotes and usually has a warmer core.
        bulge=.43*grade(corered,.04,.55)+.24*grade(diskred,-.13,.50)+.23*grade(pointed,.018,.095)+.24*grade(qbody-axis,.015,.19)+.14*grade(gradient,.09,.6)-.20*grade(flat,.47,.72)
        bulge+=.29*swelling+.18*grade(bump,1.04,1.48)-.15*grade(1.-shrink,.08,.32)-.08*grade(blue,.20,.75)
        bulge+=.20*excessbulge-.09*grade(.46-qvis,.14,.28)*(1-excessbulge)
        return 9 if bulge>.55 else 8

    disturbed=(.35*grade(lop,.09,.47)+.21*grade(odd,.13,.49)+.17*grade(offset,.08,.32)+.12*grade(sub,.05,.23)+.19*coreblue+.07*blue+.23*tidal-.16*grade(warmcore*sf,.1,.6)-.12*arm)
    disturbed+=.19*skyshape+.15*nonspiral*grade(ac,.055,.19)+.21*coreblue*skyshape+.09*grade(.43-peak,.05,.26)*blue-.09*grade(hand,.17,.47)
    # A blue off-centre knot, low peak brightness and a wide colourless
    # tidal fan are common even if rotation itself is not strongly broken
    # in the bright inner isophotes. A red bulge argues for an intact disk.
    nocore=grade(-corered,-.14,.24)*grade(.66-peak,.02,.28)
    redcentre=grade(gradient,.07,.38)*grade(corered,-.12,.30)
    disturbed+=.16*nocore+.13*skyshape*(1-redcentre)+.09*nebula*coreblue+.12*grade(offset,.025,.12)*nonspiral-.10*redcentre*grade(hchir,.05,.34)
    displaced=grade(np.hypot(cx-63.5,cy-63.5),1.2,4.2)
    disturbed+=.22*displaced*max(skyshape,nocore)
    disturbed+=.19*faintblue*grade(tailshape,.04,.18)+.17*nocore*nonspiral*grade(ripples,.15,.65)
    if disturbed>.54:
        return 0
    spiral=(.23*arm+.19*structured+.12*grade(structureB,.08,.28)+.21*blue+.29*sf+.14*grade(twist,.025,.16)+.11*lowconc+.21*extended+.12*grade(bar,.115,.38)+.1*grade(bshape,.23,.53)+.08*asym-.09*red)
    spiral+=.30*(pop-sf)-.12*grade(red,.25,.82)*(1-structured)-.13*(1-extended)*(1-structured)+.08*grade(hand,.13,.37)*grade(detail,.05,.17)+.13*grade(bshape,.22,.49)*grade(sf,.2,.7)
    spiral+=.12*ripples+.13*nebula*max(blue,pop)+.14*grade(ridge,.015,.06)*grade(hchir,.06,.26)
    # If it is small and smooth at every contour, simple colour and size
    # alone should not displace the spheroid hypothesis. Blue disks in
    # contrast typically have a resolved half-peak *and* a colour gradient.
    compact=1-grade(area[1],2.8,5.4)
    quiet=(1-grade(max(shapeharm,detail*.6),.045,.14))*(1-grade(twist,.045,.17))
    spiral-=.18*quiet*(.5+.5*compact)+.06*compact*(1-grade(bshape,.18,.44))
    spiral+=.10*pop*grade(area[1],3.6,5.6)
    diskprofile=grade(wingindex,1.05,.57)
    spiral+=.13*(diskprofile-.5)*grade(area[1],2.3,4.0)+.12*grade(envelope,.045,.14)*blue+.10*grade(broadB,.06,.21)*grade(pop,.2,.75)
    spiral-=.12*quiet*grade(wingindex,.8,1.35)*grade(peak,.43,.7)
    if spiral<.41:
        if qvis>.814:
            return 2
        if qvis<.50:
            return 4
        return 3
    barflag=.49*grade(bshape,.19,.53)+.38*grade(bar,.085,.30)+.13*grade(misalign,.055,.22)+.08*sf-.12*grade(lop,.14,.46)-.06*grade(winding,.09,.22)
    # Weak bars have subtler ellipticity but may dominate the red core.
    # Require confirmation either by population or by a straight inner bar.
    straight=abs(meanw(modes[:,1],inner))/(meanw(amp[:,1],inner)+.001)
    barflag+=.12*grade(straight,.86,.99)*grade(bshape,.15,.38)+.085*warmcore*grade(misalign,.04,.15)
    # Inner red quadrupole at the half-light surface. Look inwards for
    # the continuation of the feature with the same position angle.
    ide=(1-qr)/(1+qr)*np.exp(2j*pr)
    manualbar=np.maximum(0,(abs(ide[0]-disk)+abs(ide[1]-disk))*.5-abs(ide[0]-ide[1])*.7)
    redbar=grade(manualbar,.045,.18)*grade(1-qr[1],.17,.42)
    barflag+=.15*redbar*(1-grade((disk*ide[1].conjugate()).real-abs(ide[1])**2,0,.035))*(1-grade(lop,.22,.50))
    if barflag>.44:
        return 5
    loose=(.23*grade(1.13-pitch,-.40,.55)+.22*grade(meanw(amp[:,1],outer),.08,.40)+.15*blue+.22*grade(lop,.06,.36)+.17*grade(detail,.08,.32)+.09*lowconc+.11*tidal+.07*coreblue-.10*red)
    trust=grade(hand,.06,.36)*grade(winding,.015,.11)
    loose+=trust*(.31*(grade(degrees,11,29)-.48))+.11*grade(ac,.06,.22)+.10*grade(meanw(amp[:,1],outer),.12,.42)-.08*grade(detail,.14,.32)*(1-grade(meanw(amp[:,1],outer),.08,.25))
    # Direct ridge angle is useful even without perfectly constant pitch;
    # for small, barely resolved disks the apparent angular winding and
    # radial prominence of the two principal arms are more reliable.
    windingangle=grade(armangle,.24,.62)
    bundle=grade(meanw(amp[:,1],outer)/(meanw(amp[:,2]+amp[:,3],outer)+.04),.73,2.9)
    loose+=.25*(windingangle-.45)*grade(ridge,.009,.045)+.10*(bundle-.40)+.06*grade(offset,.04,.19)*(1-grade(warmcore,.5,1))
    openarms=.13*grade(envelope,.045,.17)+.13*grade(lop,.08,.29)+.13*grade(degrees,15,38)*grade(hand,.04,.24)
    loose+=openarms-.11*(1-windingangle)*grade(winding,.04,.14)-.055
    return 7 if loose>.47 else 6

def predict(x):
    return np.array([morphology(image) for image in x],dtype=np.int64)
