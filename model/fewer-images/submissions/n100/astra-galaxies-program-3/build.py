import sys
from pathlib import Path

source = r'''import numpy as np

# All tables are sampling grids, Fourier bases or analytic filter responses.
pi = np.pi
sy, sx = np.indices((128, 128), dtype=float)
skyarea = (sx[::2, ::2]-63.5)**2 + (sy[::2, ::2]-63.5)**2 > 50**2
freq = np.fft.fftfreq(128)[:, None]**2 + np.fft.rfftfreq(128)[None, :]**2
def gaussian(s):
    return np.exp((-2*pi*pi*s*s)*freq)
fine = gaussian(1.1)
medium = gaussian(2.1)
wide = gaussian(3.4)
theta = 2*pi*np.arange(144)/144
ct = np.cos(theta)
st = np.sin(theta)
z2 = np.exp(2j*theta)
rad = np.arange(153)*0.5
rx = rad[:, None]*ct
ry = rad[:, None]*st
levels = np.array([.79,.63,.48,.35,.25,.175,.12,.08,.05,.03])
nearx = rx[:53, ::4]
neary = ry[:53, ::4]
logr = np.linspace(np.log(.13), np.log(1.28), 49)
er = np.exp(logr)[:, None]
wave = np.exp(-1j*theta[:,None]*np.arange(1,9))/len(theta)
slopes = np.array([.65,1.,1.45,2.,2.75,3.75,5.,6.5,8.5])
signed = np.concatenate((-slopes,slopes))
length = 29
window = np.sin(pi*(np.arange(length)+.5)/length)**1.4
window = window/window.sum()
spiralwave = np.exp(1j*np.arange(1,6)[:,None,None]*signed[None,:,None]*logr[None,None,:length]) * window

def sat(x):
    return np.clip(x,0.,1.)

def unit(x):
    return np.clip(x,-1.,1.)

def sample(a,x,y):
    x = np.clip(x,0.,126.99999)
    y = np.clip(y,0.,126.99999)
    ix=x.astype(int)
    iy=y.astype(int)
    dx=x-ix
    dy=y-iy
    return (a[...,iy,ix]*(1-dx)+a[...,iy,ix+1]*dx)*(1-dy)+(a[...,iy+1,ix]*(1-dx)+a[...,iy+1,ix+1]*dx)*dy

def rings(a):
    return (2*a + np.roll(a,1,axis=-1) + np.roll(a,-1,axis=-1))*.25

def crossing(p, t):
    # First downward crossing along each ray, with linear subpixel interpolation.
    below=p<t
    ix=np.where(np.any(below,axis=0),np.argmax(below,axis=0),len(p)-1)
    ix=np.maximum(1,ix)
    j=np.arange(p.shape[1])
    top=p[ix-1,j]
    frac=sat((top-t)/np.maximum(top-p[ix,j],.000001))
    return np.maximum(.45,.5*(ix-1+frac))

def antipode(x):
    return np.roll(x,x.shape[-1]//2,axis=-1)

def contourshape(r):
    # The second harmonic of log(radius) of an ellipse equals
    # (a-b)/(a+b) in amplitude. Reweight in ellipse angle to also
    # measure the narrow tips of highly inclined disks.
    lr=np.log(r)
    sym=(lr+antipode(lr))*.5
    v=2*np.mean((sym-sym.mean(axis=1)[:,None])*z2,axis=1)
    pa=np.angle(v)*.5
    q=np.clip((1-abs(v))/(1+abs(v)),.09,1.)
    # Start in the estimated reference ellipse, and apply the residual
    # second harmonic as an ellipse in those coordinates.
    co=np.cos(pa)[:,None]
    si=np.sin(pa)[:,None]
    dx=co*ct-si*q[:,None]*st
    dy=si*ct+co*q[:,None]*st
    h=np.mod(np.arctan2(dy,dx)*(len(ct)/(2*pi)),len(ct))
    j=h.astype(int)%len(ct)
    row=np.arange(len(q))[:,None]
    f=h-np.floor(h)
    lr2=sym[row,j]*(1-f)+sym[row,(j+1)%len(ct)]*f-.5*np.log(dx*dx+dy*dy)
    mid=np.median(lr2,axis=1)[:,None]
    v=2*np.mean(np.clip(lr2-mid,-.36,.36)*z2,axis=1)
    # Shape tensor of the residual ellipse, rotate and rescale to sky coordinates.
    v=v/np.maximum(1,abs(v)/.45)
    xx=1+abs(v)**2+2*v.real
    yy=q*q*(1+abs(v)**2-2*v.real)
    xy=2*q*v.imag
    spread=np.sqrt((xx-yy)**2+4*xy**2)
    pa=pa+.5*np.arctan2(2*xy,xx-yy)
    q=np.sqrt(np.maximum(.002,(xx+yy-spread)/(xx+yy+spread)))
    h=np.sqrt(np.cos(theta-pa[:,None])**2+(np.sin(theta-pa[:,None])/q[:,None])**2)
    residual=lr+np.log(h)
    size=np.exp(np.median(residual,axis=1))
    return q,pa,size,residual-np.log(size[:,None])

def center(g):
    weight=np.exp(-((sx-63.5)**2+(sy-63.5)**2)/(2*7**2))
    k=np.argmax(weight*np.maximum(g,0))
    y,x=divmod(int(k),128)
    if g[y,x]<.002:
        return 63.5,63.5
    x=float(x)
    y=float(y)
    for step in range(3):
        d=(sx-x)**2+(sy-y)**2
        w=np.maximum(0.,g-.43*sample(g,x,y))*np.exp(-d/(2*4**2))*(d<100)
        mass=w.sum()+1e-12
        x=x+np.sum(w*(sx-x))/mass
        y=y+np.sum(w*(sy-y))/mass
    return x,y

def neighbors(g,rgb,x,y,pk,profiles):
    # Local maxima compete as separate nuclei only if there is a saddle
    # and the second source has an extended envelope on its far side.
    dmap=(sx-x)**2+(sy-y)**2
    m=(g>max(.032,pk*.11)) & (dmap>3.5**2) & (dmap<59**2)
    for dy,dx in [(0,1),(1,0),(1,1),(-1,1)]:
        for sign in (-1,1):
            m &= g>=np.roll(g,(dy*sign,dx*sign),axis=(0,1))
    ys,xs=np.nonzero(m)
    if len(xs)>0:
        order=np.argsort(g[ys,xs]/(1+dmap[ys,xs]/18**2))[::-1][:32]
        ys,xs=ys[order],xs[order]
    mainring=crossing(profiles,.22*pk)
    mainrad=np.median(mainring)
    maincore=np.median(crossing(profiles,.5*pk))
    col=sample(rgb,x,y)
    col=col/max(.015,np.mean(col))
    pair=-5.
    bridge=-5.
    mask=np.zeros(g.shape,dtype=bool)
    for yy,xx in zip(ys,xs):
        d=np.sqrt(dmap[yy,xx])
        t=np.linspace(0.,1.,max(7,int(d*1.5)))
        path=sample(g,x+(xx-x)*t,y+(yy-y)*t)
        other=g[yy,xx]
        saddle=max(0.,path.min())
        prom=1-saddle/max(.002,min(other,pk))
        if prom<.045:
            continue
        outward=((xx-x)*ct[::4]+(yy-y)*st[::4])>-.08*d
        prof=sample(g,xx+nearx[:,outward],yy+neary[:,outward])
        # Far-side widths are much less affected by the primary's halo.
        half=crossing(prof,other*.50)
        wing=crossing(prof,other*.20)
        sz=np.median(wing)
        core=np.median(half)
        corecol=rgb[:,yy,xx]/max(.015,rgb[:,yy,xx].mean())
        cd=np.sqrt(np.mean((corecol-col)**2))
        relative=min(other,pk)/max(other,pk,.001)
        backwards=((xx-x)*ct+(yy-y)*st)<-.94*d
        toward=.65*np.median(mainring[backwards])+.35*mainrad
        proximity=d/(toward+sz+.2)
        over=saddle/max(.002,min(other,pk))
        light=other*sz*core/(pk*mainrad*maincore+.001)
        massratio=min(light,1/max(light,.001))
        red=corecol[0]-corecol[2]
        # A shared isophote and two comparable extended stellar systems.
        # Do not allow a bright remote galaxy to compensate for a blank gap.
        snucleus=(prom-.15)/.12
        common=(1.03-proximity)/.26+(over-.18)*1.4
        # A second old stellar nucleus is unlike a blue HII region. The
        # connected-envelope route is complemented by a close projected
        # pair route, but only for two distinctly extended, similar nuclei.
        ev=min(common,snucleus,(relative-.31)/.22,(massratio-.14)/.11,
               (sz-3.45)/1.6,(.38-cd)/.21,(red+.04)/.28)
        ev+=.44*unit((red-.10)/.38)+.28*unit((.16-cd)/.15)
        ev-=2.5*sat((.055-over)/.055)
        binary=min((1.48-proximity)/.35,(relative-.58)/.24,
                   (massratio-.27)/.18,(red-.15)/.23,(.19-cd)/.13,
                   (sz-4.0)/1.5,(mainrad-4.)/1.5,(prom-.22)/.16)
        ev=max(ev,binary)
        # Curved tidal bridges may have a dip on the line joining nuclei.
        # Inspect a narrow corridor between them, subtracting symmetric
        # light from the far sides of both galaxies. Reject diffuse arms
        # without a distinct second peak or resolved second core.
        if relative>.27 and massratio>.12 and d<3.6*(toward+sz) and prom>.32:
            frac=np.linspace(.24,.76,15)[None,:]
            perp=np.linspace(-1,1,15)[:,None]*min(6.,.17*d)
            row=xx-x; col=yy-y
            ddx=frac*row-perp*col/d
            ddy=frac*col+perp*row/d
            strip=sample(g,x+ddx,y+ddy)
            side=sample(g,x-ddx,y-ddy)+sample(g,2*xx-x-ddx,2*yy-y-ddy)
            extra=np.quantile(np.max(strip-np.maximum(0.,side),axis=0),.35)
            lightpath=np.quantile(np.max(strip,axis=0),.25)
            peakcontrast=prom*other
            tidal=min((extra/(.02+.04*pk)-.85)/.8,
                      (lightpath/(.012+.07*min(pk,other))-.95)/.65,
                      (massratio-.16)/.14,(relative-.31)/.24,
                      (peakcontrast/pk-.21)/.15,(sz-3.0)/1.7,
                      (.44-cd)/.24,(core-1.8)/1.0)
            bridge=max(bridge,tidal)
        # Below the shared isophote, favor a resolved color-consistent
        # nucleus rather than a knot on the outskirts of a spiral.
        ev-=.8*sat((.14-over)/.14)*sat((.2-red)/.3)
        pair=max(pair,ev)
        if (prom>.37 and (proximity>1. or core<2.15 or cd>.30)) or prom>.78:
            mr=min(d*.55,1.20*sz+.75,16.)
            mask |= (sx-xx)**2+(sy-yy)**2 < mr*mr
    return pair,bridge,mask

def despike(p,bad):
    # Repair contaminated angles by their diametric partners when available.
    opp=antipode(p)
    bopp=antipode(bad)
    return np.where(bad & (~bopp),opp,p)

def harmonics(bands,grey,x,y,a,q,pa,bad,noise):
    # Oversample in log-radius and deproject the disk before testing for
    # constant-pitch waves. Local subtraction rejects static oval structure.
    co,si=np.cos(pa),np.sin(pa)
    dx=a*er*(co*ct-si*q*st)
    dy=a*er*(si*ct+co*q*st)
    xx=x+dx
    yy=y+dy
    bm=sample(bad,xx,yy)>.35
    p=despike(sample(bands,xx,yy),bm)
    g=(p[0]+2*p[1]+p[2])*.25
    # Clip isolated foreground specks not caught by the local peak test.
    med=np.median(g,axis=1)
    ab=np.median(abs(g-med[:,None]),axis=1)
    hi=(med+.035+4.5*ab)[:,None]
    g=np.minimum(g,hi)
    means=np.mean(g,axis=1)
    den=np.maximum(.013+noise*1.5,means+.018+noise)
    waves=(g/den[:,None])@wave
    amp=abs(waves)
    # Separate lopsidedness and smaller irregularities from even arm patterns.
    inner= (er[:,0]>.32) & (er[:,0]<.88)
    out=(er[:,0]>.60) & (er[:,0]<1.16)
    lob=np.mean(amp[inner,0])
    lopout=np.mean(amp[out,0])
    texture=np.mean(np.sqrt(np.sum(amp[inner,2:7]**2,axis=1)))
    armcontrast=np.mean(amp[inner,1])
    # The radial velocity of an angular maximum measures winding directly.
    # In log polar coordinates an oblique ridge has opposite radial and
    # azimuthal derivatives on each flank. Measure local pitch even when
    # different arms cancel in the disk-wide Fourier coefficient.
    low=waves[:,:6]
    az=np.exp(1j*theta[None,:]*np.arange(1,7)[:,None])
    field=2*(low@az).real
    tang=2*((1j*np.arange(1,7)*low)@az).real
    radial=np.gradient(field,logr,axis=0)
    def integrate(arr):
        return np.mean([np.roll(arr,i,axis=1) for i in (-5,-3,0,3,5)],axis=0)[14:45]
    cross=integrate(tang*radial)
    slope=abs(cross)/(integrate(tang*tang)+.00007)
    motion=abs(cross)/(np.sqrt(integrate(tang*tang)*integrate(radial*radial))+.0002)
    w=abs(cross)*motion
    localpitch=(w/(1.+slope**2)).sum()/(w.sum()+.00001)
    # Remove the slowly varying non-spiral oval in each radial window.
    best=np.zeros((5,len(signed)))
    for off in (6,13,20):
        chunk=waves[off:off+length,:5].T
        chunk=chunk-np.sum(chunk*window,axis=1)[:,None]
        strength=abs((spiralwave*chunk[:,None,:]).sum(axis=2))
        best=np.maximum(best,strength)
    even=np.maximum(best[1,:len(slopes)],best[1,len(slopes):])
    many=np.maximum(best[2:,:len(slopes)],best[2:,len(slopes):]).max(axis=0)
    arm=even+.60*many
    # Angular swings in just the light-carrying part of the disk, using
    # adjacent complex coefficients instead of imposing a single pitch.
    c=waves[12:46,1]
    step=5
    corr=c[step:]*np.conj(c[:-step])
    speed=abs(np.angle(np.sum(corr)))/(2*(logr[step]-logr[0]))
    spec=arm.max()
    fast=arm[4:].max()
    slow=arm[:4].max()
    c3=np.maximum(best[2:,:len(slopes)],best[2:,len(slopes):]).max()
    def colorpart(lo,hi):
        use=(er[:,0]>=lo)&(er[:,0]<hi)
        # Use integrated ring light in each band; rare colored point sources
        # carry less weight after restricting to the interdecile range.
        z=p[:,use,:]
        z=np.minimum(z,np.quantile(z,.9,axis=2,keepdims=True))
        cols=z.mean(axis=(1,2))
        return (cols[0]-cols[2])/max(.03,cols[0]+cols[2])
    # A compact coherent quadrupole with a subsequent drop or phase
    # change is a bar, not a spiral wave. Weight in-plane elongation below.
    vec=waves[:,1]
    barpower=0.
    for j in (1,7,13,19):
        v=np.mean(vec[j:j+10]); ext=np.mean(vec[j+16:j+24]); mag=abs(v)
        alignment=max(0.,(ext*np.conj(v)).real/(mag+.00001))
        strength=min(mag,max(0.,mag-alignment))
        strength*=.4+.6*sat(1.+v.real/(mag+.00001))
        barpower=max(barpower,strength)
    return (lob,lopout,texture,armcontrast,spec,fast,slow,speed,c3,
            colorpart(.14,.32),colorpart(.5,.95),localpitch,barpower)

def outerlight(img,bad,x,y,a,q,pa,pk,noise):
    # Look farther out than the main body for a stream. Compare opposite
    # wedges, not opposite pixels: a stream can be both faint and curved.
    t=np.linspace(.5,2.7,52)[:,None]
    dx=a*t*(np.cos(pa)*ct-np.sin(pa)*q*st)
    dy=a*t*(np.sin(pa)*ct+np.cos(pa)*q*st)
    xx=x+dx
    yy=y+dy
    good=(xx>1)&(xx<126)&(yy>1)&(yy<126)
    good &= sample(bad,xx,yy)<.20
    p=sample(img,xx,yy)
    reference=.05*pk+2.8*noise
    diff=np.maximum(0.,p-antipode(p))*(good & antipode(good))
    valid=(p>2.5*noise+.012*pk) & good
    relative=diff/(np.maximum(.0,p)+reference)
    # Average neighboring directions to avoid isolated knots.
    smooth=np.mean(np.array([np.roll(relative,j,axis=1) for j in (-4,-2,0,2,4)]),axis=0)
    stream=smooth[12:35].mean(axis=0)
    broad=np.quantile(stream,.90)
    narrow=np.quantile(stream,.96)
    # Lopsided objects and tidal pairs sometimes have very long but
    # nearly point-symmetric tidal arcs. Inspect the angular filling factor.
    typical=np.median(p,axis=1)[:,None]
    rel2=np.maximum(p-typical,0)/(np.maximum(p,0)+reference)*valid
    extreme=np.quantile(np.mean(rel2[13:35],axis=0),.94)
    blip=np.mean(np.quantile(smooth[14:32],.94,axis=1))
    return .5*broad+.3*narrow+.2*blip,extreme

def diskdetail(g,mask,x,y,major,minor,pa,pk):
    # Cross-check smoothness using *local* isophote shape. For any profile
    # that depends only on elliptical radius, the intensity around each
    # contour is constant. Let the ellipse change from annulus to annulus.
    co=np.cos(pa)[:,None]
    si=np.sin(pa)[:,None]
    dx=major[:,None]*co*ct-minor[:,None]*si*st
    dy=major[:,None]*si*ct+minor[:,None]*co*st
    xx=x+dx; yy=y+dy
    m=sample(mask,xx,yy)>.20
    p=despike(sample(g,xx,yy),m)
    med=np.median(p,axis=1)
    norm=(p-med[:,None])/(np.maximum(.015,med[:,None])+.032*pk+.008)
    norm=np.clip(norm,-.85,.85)
    harmon=norm@wave
    # An offset nucleus, a tidal plume and well-balanced spiral arms
    # contribute to different angular modes after ellipse subtraction.
    smooth=np.mean(np.quantile(abs(norm[1:6]),.70,axis=1))
    detail=np.mean(np.sqrt(np.sum(abs(harmon[2:7,2:])**2,axis=1)))
    off=2*np.mean(abs(harmon[3:8,0]))
    return smooth,detail,off

def classify(im):
    F=np.fft.rfft2(im,axes=(-2,-1))
    amp=(F[0]+2*F[1]+F[2])*.25
    bands=np.fft.irfft2(F*fine,s=(128,128),axes=(-2,-1))
    bg=np.median(bands[:,::2,::2][:,skyarea],axis=1)
    bands=bands-bg[:,None,None]
    g=(bands[0]+2*bands[1]+bands[2])*.25
    g2=np.fft.irfft2(amp*medium,s=(128,128),axes=(-2,-1))
    g3=np.fft.irfft2(amp*wide,s=(128,128),axes=(-2,-1))
    g2-=np.median(g2[::2,::2][skyarea])
    g3-=np.median(g3[::2,::2][skyarea])
    noise=max(.001,-np.quantile(g2[::2,::2][skyarea],.16))
    x,y=center(g)
    pk=max(.006,sample(g,x,y))
    mainp=sample(g,x+rx,y+ry)
    pair,bridge,mask=neighbors(g,bands,x,y,pk,mainp)
    bm=sample(mask,x+rx,y+ry)>.4
    pf=rings(despike(mainp,bm))
    pm=rings(despike(sample(g2,x+rx,y+ry),bm))
    pw=rings(despike(sample(g3,x+rx,y+ry),bm))
    rr=[]
    for lev in levels:
        mix=sat((.19-lev)/.145)
        extra=.55*sat((.075-lev)/.05)
        prof=(1-mix)*pf+mix*((1-extra)*pm+extra*pw)
        rr.append(crossing(prof,lev*pk+noise*1.25*(1-lev)))
    rr=np.array(rr)
    q,pa,a,res=contourshape(rr)
    # Undo the isotropizing effect of smoothing in quadrature, allowing
    # more blur at the fainter contours (both larger kernel and more sigmas).
    blur=(1.25**2 + sat((.19-levels)/.145)*(2.1**2-1.1**2)+.55*sat((.075-levels)/.05)*(3.4**2-2.1**2)) * (-2*np.log(levels))
    aa=np.sqrt(np.maximum(a*a-blur,a*a*.54))
    bb=np.sqrt(np.maximum((a*q)**2-blur,(a*q)**2*.37))
    qc=np.clip(bb/aa,.045,1.)
    qin=np.median(qc[1:4])
    qout=np.median(qc[5:8])
    stream,filament=outerlight(g3,mask,x,y,max(4.,a[5]),max(.19,q[5]),pa[5],pk,noise)
    qshape=np.median(qc[2:7])
    qdisk=np.median(q[5:8])
    ang=np.angle(np.mean((1.05-q[5:8])*np.exp(2j*pa[5:8])))*.5
    size=max(3.,a[7])
    # Axis ratio and PA of each isophote relative to the outer disk.
    co=np.cos(pa-ang)
    si=np.sin(pa-ang)
    xx=co**2+(q*si)**2
    yy=(si**2+(q*co)**2)/qdisk**2
    xy=si*co*(1-q*q)/qdisk
    ell=(xx-yy+2j*xy)/(xx+yy+2*q/qdisk)
    barvec=np.mean(ell[1:5])
    elong=np.median(1-q[1:5])
    bar=abs(barvec)
    # Discount apparent minor-axis bars caused by deprojecting round bulges.
    bar-=.78*max(0.,-barvec.real)*(1-sat((elong-.28)/.35))
    bar*=.65+.35*sat((elong-.10)/.27)
    # Changes in PA without regard to which side of pi the estimates lie on.
    turn=np.max(abs(np.sin(pa[2:6]-ang))*(1-q[2:6]))
    # Diametric contour disagreement, including localized outer tidal streams.
    asym=abs(res-antipode(res))
    skew=np.mean(np.mean(np.minimum(asym,.95),axis=1)[3:8])
    skewout=np.mean(np.mean(np.minimum(asym,1.15),axis=1)[7:])
    tailness=np.mean(np.quantile(asym[7:],.9,axis=1))
    irregular=np.mean(np.quantile(abs(res[3:8]),.78,axis=1))
    lob,lopout,tex,h2,spir,fast,slow,speed,many,cin,cout,pitch,barpower=harmonics(bands,g,x,y,size,qdisk,ang,mask,noise)
    # Compare the contours in their own ellipse angles. The inverse slope
    # of a deformed isophote overweights its minor axis on the raw sky grid.
    dx=np.cos(pa)[:,None]*ct-np.sin(pa)[:,None]*q[:,None]*st
    dy=np.sin(pa)[:,None]*ct+np.cos(pa)[:,None]*q[:,None]*st
    phase=np.mod(np.arctan2(dy,dx)*len(ct)/(2*pi),len(ct))
    ix=phase.astype(int)%len(ct)
    iy=np.arange(len(pa))[:,None]
    delta=phase-np.floor(phase)
    rz=res[iy,ix]*(1-delta)+res[iy,(ix+1)%len(ct)]*delta
    # Boxy versus pointed contours (disky ellipses have a positive a4).
    pointiness=2*np.mean(np.clip(rz,-.4,.4)*np.cos(4*theta),axis=1)
    disky=np.median(pointiness[3:8])
    bvec=ell[1:6]
    dv=(np.maximum(0.,-bvec.real)*(.85-.70*sat((1-q[1:6]-.33)/.30)))
    barpeak=np.max((abs(bvec)-dv)*sat((1-q[1:6])/.36))
    bar=.55*bar+.45*barpeak
    smooth,detail,off=diskdetail(g,mask,x,y,a,a*q,pa,pk)
    # Orientation changes measured on ring *pairs*. This avoids declaring
    # an oval disk to be a bar when the outer arms alone change the PA.
    def coherence(v):
        return abs(v.mean())/(.006+np.mean(abs(v)))
    straight=coherence(ell[1:5])
    innerq=np.median(q[1:5])
    debar=(1-innerq)/(1+innerq)
    tipangle=np.max(abs(np.sin(pa[1:4]-pa[3]))*(1-q[1:4]))
    c=sample(bands,x,y)
    corecol=(c[0]-c[2])/max(.04,c[0]+c[2])
    warm=.45*corecol+.55*cin
    grad=warm-cout

    # Estimate disk thickness on a finer blur to retain thin spindles.
    # Edge-on disks have extended major-axis wings and a thicker central
    # component in the presence of a classical bulge.
    fr=np.array([crossing(pf,pk*t+noise*1.6) for t in (.30,.15,.075)])
    fq,fpa,fa,fres=contourshape(fr)
    diskq=np.median(fq[1:])
    ph=fpa[1]
    u=fa[1]*np.linspace(-1.,1.,41)
    v=np.arange(71)*.5
    dx=u[:,None]*np.cos(ph)-v[None,:]*np.sin(ph)
    dy=u[:,None]*np.sin(ph)+v[None,:]*np.cos(ph)
    upper=sample(g,x+dx,y+dy)
    lower=sample(g,x-dx[::-1],y-dy[::-1])
    profile=(upper+lower)*.5
    # Use thickness relative to the local ridge, not an absolute common
    # isophote, so a taper in disk brightness does not imitate a bulge.
    widths=crossing(profile.T,.5*np.maximum(.009,profile[:,0]))
    outerwidth=np.median(widths[[7,8,9,10,11,29,30,31,32,33]])
    swelling=np.median(widths[18:23])/(.1+outerwidth)
    major=profile[:,0]/pk
    centrality=1/max(.035,np.mean(major[[10,11,12,28,29,30]]))
    # A bulge and a disk can have quite similar half-light radii in a
    # projected image; inspect shape on the fine-blur wings as well.
    wingratio=fa[2]/max(2.,fa[0])

    # Two-nucleus systems have priority over single-object morphology.
    if pair>.14 or (bridge>-.05 and (tailness>.34 or skewout>.17)):
        return 1

    # Edge-on disks: require a long thin outer component; diskiness, size,
    # and a change of flattening at the bulge help in less extreme cases.
    flat=.42*qout+.31*qshape+.27*diskq
    edge=((.32-flat)/.080 + .92*unit((size-15)/9)
          +.64*unit((qin-qout-.035)/.14)+.45*unit((.29-diskq)/.10)
          +.40*unit((swelling-1.03)/.28)+.65*sat((.07-cout)/.19)
          +.80*unit((disky-.023)/.065))
    # Bulge light is rounder, redder and more concentrated than disk light.
    conc=aa[6]/max(aa[2],1.)
    bulge=(1.52*unit((warm-.072)/.14)+.65*unit((qin-qout-.035)/.13)
           +.45*unit((conc-2.25)/.75)+.48*unit((grad-.040)/.12)
           +1.20*unit((swelling-1.03)/.30)+.62*unit((disky-.040)/.075)
           +.45*unit((centrality-1.95)/.65)+.45*unit((wingratio-1.75)/.50)
           +1.0*unit((qin-.34)/.17)+.65*unit((warm+cout-.10)/.24))

    # Inner elongation compared to the outer disk indicates a bar. Winding,
    # residual small-scale light and blue disks indicate spiral structure.
    barscore=((bar-.105)/.064+.52*unit((turn-.08)/.10)+.30*unit((size-13)/9)
              +.65*unit((barpower-.038)/.047)+.30*unit((warm-.036)/.13)
              +.68*unit((debar-.15)/.10)+.60*unit((straight-.85)/.15)
              -.90*sat((tipangle-.045)/.11))
    structure=(.90*unit((tex-.042)/.030)+.85*unit((spir-.033)/.028)
               +1.20*unit((.055-cout)/.14)+.50*unit((grad-.075)/.13)
               +.60*unit((size-15)/8)+.72*unit((irregular-.067)/.065)
               -.90*sat((.45-qshape)/.16)-.74*sat((.045-tex)/.039)*sat((warm+.02)/.17))
    # Tidal damage can be a faint, narrow one-sided extension or gross
    # lopsidedness all the way in to the high surface-brightness region.
    disturb=(1.00*unit((skewout-.19)/.16)+.90*unit((tailness-.33)/.24)
             +.60*unit((skew-.12)/.12)+.75*unit((lopout-.10)/.10)
             +.30*unit((lob-.10)/.085)-.80*unit((spir-.052)/.048))
    disturb=.66*disturb+1.3*unit((stream-.16)/.13)+.62*unit((filament-.25)/.20)
    disturb+=.85*sat((tailness-.60)/.50)+.72*sat((skew-.30)/.25)
    # A strong tail alone can be diagnostic even if it emerges from an
    # undisturbed, nearly perfect elliptical body.
    tailev=(1.6*unit((stream-.165)/.135)+1.15*unit((filament-.21)/.17)
            +.78*unit((tailness-.29)/.23)-.68*unit((spir-.055)/.045)
            -.64*sat((.02-cout)/.15))
    # S-shaped tidal systems may have a straight or nearly smooth core
    # followed by a large bend in both arms. Look for a twist that would
    # leave the bright core ellipse, and which persists at very low light.
    twist=abs(np.sin(pa[7:]-pa[3]))
    oval=(1-q[3])*(1-q[7:])
    warp=np.mean(twist*np.sqrt(oval))
    tailpa=(1.20*unit((warp-.18)/.12)+.95*unit((filament-.18)/.15)
            +.70*unit((irregular-.055)/.050)+.42*unit((.085-smooth)/.07)
            -.85*sat((bar-.12)/.10)-.5*sat((spir-.055)/.05))
    one=(1.35*unit((lopout+.55*lob-.14)/.11)+.80*unit((off-.085)/.065)
         +.70*unit((skewout-.14)/.12)+.60*unit((.048-spir)/.038)
         +.60*unit((filament-.17)/.16))
    disturb=max(disturb,tailev,tailpa,one)
    if disturb>.95 and (edge<.55 or (disturb>2.65 and flat>.22)):
        return 0
    if edge>0.:
        return 9 if bulge>0. else 8
    if barscore>.0 and structure+min(2.,barscore)>.05 and qout>.30:
        return 5
    # Some well-resolved tight disks are virtually featureless at first
    # glance; unlike a spheroid their extended disk often has young stars.
    spiralcolor=(1.15*unit((.045-cout)/.13)+.95*unit((grad-.08)/.10)
                 +.55*unit((size-15)/7)+.55*unit((.045-warm)/.13)-.4)
    spiralcolor-=1.65*sat((.18-qshape)/.1)+.75*sat((.5-qshape)/.2)
    # A smooth ellipse can be blue or have a gradient. Only allow color to
    # supply the deciding vote if substructure or a disk break is visible.
    activity=.95*unit((detail-.033)/.027)+.70*unit((smooth-.090)/.068)
    spheroid=.9*sat((.10-smooth)/.07)+1.0*sat((.040-detail)/.027)
    spheroid+=.65*sat((.47-qshape)/.16)+.8*sat((warm-.14)/.13)
    featured=max(structure+.7*activity,spiralcolor+activity-.7)-spheroid
    if featured>0. and (qshape>.49 or (featured>1.35 and qout>.35)):
        # Open arms have high contrast organized at low winding frequency;
        # tightly wrapped disks can look nearly smooth at this resolution.
        loose=(.78*unit((slow-.032)/.032)+.78*unit((slow-fast)/.037)
               +1.00*unit((h2-.055)/.053)+.40*unit((2.10-speed)/1.5)
               +.50*unit((skew-.13)/.12)-.33*unit((many-.032)/.030)
               +.48*unit((irregular-.066)/.056)+.70*unit((pitch-.19)/.18)
               +.68*unit((cout+.02)/.12)+.40*unit((conc-2.30)/.75)
               +.4*unit((lopout-.092)/.08)+.48*unit((h2-1.3*tex)/.05))
        return 7 if loose>0. else 6
    if qshape>.81:
        return 2
    if qshape>.50:
        return 3
    return 4

def predict(x):
    return np.array([classify(image) for image in x],dtype=np.int64)
'''

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / 'solution.py').write_text(source, encoding='utf-8')
