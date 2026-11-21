import math
import numpy as np

# Image-plane coordinates, optical transfer functions, and polar quadrature.
Y, X = np.indices((128, 128), dtype=np.float64)
D2 = (X-63.5)**2 + (Y-63.5)**2
SKY = D2 > 52**2
FX = np.fft.rfftfreq(128)
FY = np.fft.fftfreq(128)
F2 = 2*np.pi**2 * (FX[None,:]**2 + FY[:,None]**2)
K = [np.exp(-s*s*F2) for s in (1.05, 2.15, 4.0)]
TH = 2*np.pi*np.arange(128)/128
CT = np.cos(TH)
ST = np.sin(TH)
STEP = .5
RAY = np.arange(0, 82, STEP)[:,None]
XRAY = RAY*CT
YRAY = RAY*ST
LEVEL = np.array([.70,.49,.32,.20,.12,.073,.043])
RAD = np.exp(np.linspace(np.log(.09),np.log(1.06),43))
LR = np.log(RAD)
WINDS = np.array([.85,1.15,1.5,1.9,2.35,2.9,3.6,4.5,5.6,7.,8.8])
FAMILIES = []
for lo,hi in ((.20,.68),(.27,.90),(.37,1.03),(.48,1.05)):
    sel = (RAD >= lo) & (RAD <= hi)
    zz = LR[sel]
    phase = np.arange(2,5)[:,None,None]*WINDS[None,:,None]*zz[None,None,:]
    FAMILIES.append((sel,np.exp(1j*phase)))


def clip(v,lo=0.,hi=1.):
    return np.minimum(np.maximum(v,lo),hi)


def trans(im, ker):
    return np.fft.irfft2(im*ker,s=(128,128),axes=(-2,-1))


def photgrid(x,y):
    ix = np.floor(clip(x,0,126.99999)).astype(np.int32)
    iy = np.floor(clip(y,0,126.99999)).astype(np.int32)
    dx = clip(x-ix)
    dy = clip(y-iy)
    keep = (x >= 0) & (y >= 0) & (x <= 127) & (y <= 127)
    return (iy,ix,(1-dy)*(1-dx)*keep,(1-dy)*dx*keep,dy*(1-dx)*keep,dy*dx*keep)


def project(im,grid):
    iy,ix,w,xw,yw,zw = grid
    return im[...,iy,ix]*w+im[...,iy,ix+1]*xw+im[...,iy+1,ix]*yw+im[...,iy+1,ix+1]*zw


def spread(v):
    return max(.0015,1.4826*np.median(np.abs(v-np.median(v))))


def light(col):
    return .30*col[0]+.41*col[1]+.29*col[2]


def locate(s,t):
    # The target, not necessarily the brightest source, is centered by the survey.
    w = (.60*s+.40*t)*np.exp(-.5*D2/8**2)
    yy,xx = np.unravel_index(np.argmax(w),w.shape)
    if D2[yy,xx] > 18**2:
        yy,xx = 64,64
    cx,cy = float(xx),float(yy)
    for n in range(4):
        d = (X-cx)**2+(Y-cy)**2
        a = np.maximum(s-.40*s[yy,xx],0)*np.exp(-d/(2*3.2**2))*(d < 9**2)
        v = a.sum()+1.e-12
        if v > 1.e-7:
            cx,cy = (a*X).sum()/v,(a*Y).sum()/v
    return cx,cy


def outlines(s,t,cx,cy,noise):
    grid = photgrid(cx+XRAY,cy+YRAY)
    v = project(s,grid)
    mix = .70*clip((RAY-5)/15)
    v = v*(1-mix)+mix*project(t,grid)
    top = max(.001,float(v[0,0]))
    # An isophote terminates at the first outward crossing, not at a field star.
    limits = np.minimum(top*.86,np.maximum(top*LEVEL,2.8*noise))
    hits = v[None,:,:] < limits[:,None,None]
    ids = np.argmax(hits,axis=1)
    ids = np.where(hits.any(axis=1),np.maximum(ids,1),len(RAY)-1)
    cols = np.arange(128)
    pre = v[ids-1,cols]
    post = v[ids,cols]
    r = STEP*(ids-1+clip((pre-limits[:,None])/np.maximum(pre-post,1.e-7)))
    r = np.median(np.stack([np.roll(r,j,axis=1) for j in (-2,-1,0,1,2)]),axis=0)
    return np.maximum(.25,r),top


def shape(r):
    # Integrals of x, y, xx, xy, yy over the interior of a polar curve.
    a = r*r
    nm = 1./np.maximum(a.sum(axis=1),1.e-7)
    dx = (2/3)*(a*r*CT).sum(axis=1)*nm
    dy = (2/3)*(a*r*ST).sum(axis=1)*nm
    xx = .5*(a*a*CT**2).sum(axis=1)*nm-dx**2
    yy = .5*(a*a*ST**2).sum(axis=1)*nm-dy**2
    xy = .5*(a*a*CT*ST).sum(axis=1)*nm-dx*dy
    diff = np.sqrt((xx-yy)**2+4*xy**2)
    major = np.sqrt(np.maximum(.0625,2*(xx+yy+diff)))
    minor = np.sqrt(np.maximum(.0625,2*(xx+yy-diff)))
    ang = .5*np.arctan2(2*xy,xx-yy)
    return major,minor,ang,dx,dy


def compass(ang,q,w):
    e = (1-q)/(1+q)
    z = np.sum(w*e*np.exp(2j*ang))/(np.sum(w)+1.e-10)
    v = min(.95,abs(z))
    return (1-v)/(1+v),.5*np.angle(z)


def isodetails(r,a,b,pa,dx,dy):
    # Re-express each contour in its own principal-axis coordinate system.
    # Positive cos(4 phi) means pointed/disky rather than oval/boxy.
    u = (r*CT-dx[:,None])*np.cos(pa)[:,None]+(r*ST-dy[:,None])*np.sin(pa)[:,None]
    v = -(r*CT-dx[:,None])*np.sin(pa)[:,None]+(r*ST-dy[:,None])*np.cos(pa)[:,None]
    u = u/a[:,None]
    v = v/b[:,None]
    arg = np.arctan2(v,u)
    rad = np.sqrt(u*u+v*v)
    w = np.abs(np.angle(np.exp(1j*(np.roll(arg,-1,1)-np.roll(arg,1,1)))))
    excess = clip(rad-.99,-.45,.45)
    v4 = 2*(excess*w*np.cos(4*arg)).sum(axis=1)/(w.sum(axis=1)+1.e-8)
    dev = np.sqrt((excess**2*w).sum(axis=1)/(w.sum(axis=1)+1.e-8))
    return v4,dev


def bandavg(v,w):
    return np.sum(v*w)/(np.sum(w)+1.e-9)


def sect(v,lo,hi,wt):
    return bandavg(v,wt*((RAD >= lo) & (RAD <= hi)))


def azimuth(image,cx,cy,rsize,q,pa):
    co,si = np.cos(pa),np.sin(pa)
    u,v = rsize*RAD[:,None]*CT,rsize*q*RAD[:,None]*ST
    return project(image,photgrid(cx+u*co-v*si,cy+u*si+v*co))


def winsor(v,noise):
    low,med,high = np.quantile(v,[.10,.5,.87],axis=1)
    high += .52*(high-med)+1.5*noise
    low -= .60*(med-low)+1.5*noise
    return clip(v,low[:,None],high[:,None])


def harmonics(az,noise,top):
    az = winsor(az,noise)
    m = az.mean(axis=1)
    denom = np.maximum(.8*noise,m)+1.5*noise+.009*top+.0005
    f = np.fft.rfft(az,axis=1)[:,1:13]*2/(az.shape[1]*denom[:,None])
    return f,denom,m


def winding(f,w):
    # Correlate radial Fourier coefficients with leading and trailing logarithmic
    # spirals. Static ovals give the same power at both handednesses.
    amp = np.zeros((2,len(WINDS)))
    for sel,templates in FAMILIES:
        wt = w[sel]
        if wt.sum() < .5:
            continue
        wt = wt/wt.sum()
        val = f[sel,1:4].T.copy()
        val -= (val*wt).sum(axis=1)[:,None]
        z = val[:,None,:]*wt
        plus = np.abs(np.sum(z*templates,axis=2))
        minus = np.abs(np.sum(z*templates.conj(),axis=2))
        # A smooth radial change in ellipticity needn't wind. Also require
        # evidence spread over the window rather than an individual bright knot.
        plus,minus = (plus+.001),(minus+.001)
        for sign,p,m in ((0,plus,minus),(1,minus,plus)):
            gain = clip((p-m)/p)
            vv = (p-.001)*np.sqrt(gain)
            amp[sign] += .25*np.sqrt((vv**2).sum(axis=0))
    ind = np.unravel_index(np.argmax(amp),amp.shape)
    row = amp[ind[0]]
    mask = clip((row/(row.max()+1.e-8)-.57)/.43)**2
    slope = (np.sum(mask*np.log(WINDS))+.0001*np.log(2.7))/(mask.sum()+.0001)
    return float(row.max()),np.exp(slope)


def thickness(s,cx,cy,pa,size,q,noise,top):
    # Fold about both axes and measure vertical half-width relative to each
    # column's OWN surface brightness (not a common isophote).
    us = np.array([0,.06,.12,.27,.36,.45,.54,.64])
    vs = np.linspace(0,max(4.,size*q*.95),64)
    u = size*us[:,None]
    v = vs[None,:]
    co,si = np.cos(pa),np.sin(pa)
    slices = []
    for h,k in ((1,1),(-1,1),(1,-1),(-1,-1)):
        slices.append(project(s,photgrid(cx+h*u*co-k*v*si,cy+h*u*si+k*v*co)))
    vprof = np.median(np.stack(slices),axis=0)
    val = np.maximum(vprof[:,0],vprof[:,:5].max(axis=1)*.93)
    widths = []
    for t in (.35,.60):
        lev = np.maximum(val*t,np.minimum(.8*val,1.8*noise))
        hit = vprof < lev[:,None]
        j = np.maximum(1,np.where(hit.any(axis=1),np.argmax(hit,axis=1),len(vs)-1))
        rows = np.arange(len(us))
        y1,y2 = vprof[rows,j-1],vprof[rows,j]
        widths.append((j-1+clip((y1-lev)/np.maximum(y1-y2,1.e-8)))*vs[1])
    thick = .5*(widths[0]+widths[1])
    swell = (np.mean(thick[:3])+.3)/(np.median(thick[3:7])+.3)
    centerfrac = np.median(val[3:6])/(val[0]+.01*top+1.e-7)
    return swell,centerfrac


def doubles(s,t,broad,col,cx,cy,pa,size,q,noise,top):
    # Look for a second resolved light maximum. A point source in an otherwise
    # empty patch of sky should not make a merger.
    ds = np.hypot(X-cx,Y-cy)
    dlim = min(56.,max(24.,size*1.7+10))
    mask = (ds>3.8) & (ds<dlim) & (t>max(noise*4.7,top*.115))
    for i,j in ((-1,0),(0,-1),(1,0),(0,1),(1,1),(-1,1),(-1,-1),(1,-1)):
        mask &= t >= np.roll(np.roll(t,i,0),j,1)
    ys,xs = np.nonzero(mask)
    if not len(ys):
        return -5.,0.
    order = np.argsort((t[ys,xs]+.5*broad[ys,xs])/(ds[ys,xs]+6)**.55)[::-1][:16]
    best,knots = -5.,0.
    co,si = np.cos(pa),np.sin(pa)
    for j in order:
        x,y = xs[j],ys[j]
        dd = ds[y,x]
        ux,uy = (x-cx)/dd,(y-cy)/dd
        line = np.linspace(0,1,max(6,int(dd)))
        seq = project(t,photgrid(cx+line*(x-cx),cy+line*(y-cy)))
        saddle = max(0.,np.min(seq))
        peak = max(.001,t[y,x])
        prom = clip(1-saddle/peak)
        if prom < .038:
            continue
        h = max(.001,s[y,x])
        # Check how much resolved envelope the secondary has AWAY from the
        # primary; an unresolved star has very little flux several PSF widths out.
        ang = np.linspace(-1.85,1.85,17)
        vx,vy = ux*np.cos(ang)-uy*np.sin(ang),uy*np.cos(ang)+ux*np.sin(ang)
        ring = np.array([np.median(project(s,photgrid(x+r*vx,y+r*vy))) for r in (3.,5.,8.)])
        extent = (.50*ring[0]+.34*ring[1]+.16*ring[2])/h
        relative = h/top
        bridge = saddle/(min(top,peak)+1.e-6)
        gap = saddle/(noise+.001)
        dist = dd*np.hypot((ux*co+uy*si),(-ux*si+uy*co)/max(.48,q))/(size+6)
        blue = (col[2,y,x]-col[0,y,x])/(.1+col[0,y,x]+col[2,y,x])
        pair = (.85*clip((relative-.20)/.62,-1,1.35)
                +1.1*clip((extent-.37)/.24,-1.5,1.35)
                +.8*clip((prom-.19)/.38,-.7,1.1)
                +.75*clip((bridge-.09)/.24,-.7,1.1)
                +.55*clip((gap-2.8)/3.,-1,1)
                +.48*clip((1.25-dist)/.62,-2,1)
                -.65*clip((blue-.09)/.15) - .65*clip((relative-1.85)/1.2))
        if gap < 2.5 and bridge < .13:
            pair -= 1.
        if prom < .13:
            pair -= .7
        if dd < 5.8 and extent < .46:
            pair -= .65
        best = max(best,pair)
        if dist < 1.1 and prom > .09:
            knots += min(1.,2*prom)*clip((relative-.09)/.19)
    return best,knots


def classify(im):
    ft = np.fft.rfft2(im,axes=(-2,-1))
    col = trans(ft,K[0])
    col -= np.median(col[:,SKY],axis=1)[:,None,None]
    s = light(col)
    lf = light(ft)
    t = trans(lf,K[1]); t -= np.median(t[SKY])
    broad = trans(lf,K[2]); broad -= np.median(broad[SKY])
    n1,n2 = spread(s[SKY]),spread(t[SKY])
    cx,cy = locate(s,t)
    r,top = outlines(s,t,cx,cy,n2)
    # A second centering pass using moderate isophotes avoids moving the center
    # to a bright star-forming knot or a random spot in a saturated nucleus.
    a,b,pa,dx,dy = shape(np.minimum(r,1.45*np.roll(r,64,1)+1.))
    cx += clip(.72*np.median(dx[1:4]),-2.4,2.4)
    cy += clip(.72*np.median(dy[1:4]),-2.4,2.4)
    r,top = outlines(s,t,cx,cy,n2)
    opp = np.roll(r,64,axis=1)
    balanced = np.minimum(r,1.35*opp+1.2)
    a,b,pa,dx,dy = shape(balanced)
    q = clip(b/a,.075,1.)
    qout,angle = compass(pa,q,np.array([0.,0.,.04,.12,.24,.36,.24]))
    qmid,am = compass(pa,q,np.array([.06,.17,.29,.29,.15,.04,0.]))
    qin,ai = compass(pa,q,np.array([.32,.38,.23,.07,0.,0.,0.]))
    # Use the faint edge for deprojection, but don't make the arm detector
    # hypersensitive to small angle errors in almost edge-on sources.
    size = clip(.22*a[-3]+.51*a[-2]+.27*a[-1],2.,78.)
    dq = max(.28,qout)
    v4,dev = isodetails(balanced,a,b,pa,dx,dy)
    pointed = np.median(v4[2:])
    asymiso = np.mean(np.abs(r-opp)/(r+opp+1.),axis=1)
    lops = .42*np.median(asymiso[1:4])+.58*np.median(asymiso[4:])
    warp = np.max(np.abs(np.sin(pa[1:5]-angle))*(1-q[1:5]))
    innerdiff = max(0.,qout-np.min(q[:5]))
    devouter = np.median(dev[3:])
    # Chromatic and morphological profiles use sky-subtracted light.
    az = azimuth(col,cx,cy,size,dq,angle)
    gz = light(az)
    f,den,mu = harmonics(gz,n1,top)
    # Suppress unrelated objects in azimuthal color averages as well.
    red = winsor(az[0],n1).mean(axis=1)
    blue = winsor(az[2],n1).mean(axis=1)
    clr = clip((blue-red)/(np.maximum(.05*top,red+blue)+.007),-1.,1.)
    quality = clip(mu/(n1*4+.012*top+.002),.01,1)*clip(RAD*size/3.,.1,1.)
    diskclr = sect(clr,.40,.91,quality)
    inclr = sect(clr,.09,.24,quality)
    colorgrad = diskclr-inclr
    # In very round bulges and in inclined discs asymmetric residuals are
    # different phenomena. Keep odd and higher order powers separately.
    noiseamp = n1/den
    hi = np.sqrt(np.maximum(0.,.5*np.sum(np.abs(f[:,2:])**2,axis=1)-.64*noiseamp**2))
    odd = np.sqrt(np.maximum(0.,.5*np.sum(np.abs(f[:,::2])**2,axis=1)-.40*noiseamp**2))
    fine = sect(hi,.27,.94,quality)
    uneven = sect(odd,.23,.91,quality)
    one = sect(np.abs(f[:,0]),.25,.96,quality)
    wind,pitch = winding(f,quality)
    # Strength and phase persistence of an inner two-fold pattern. Remove the
    # spurious bar due to deprojecting an intrinsically round central bulge.
    c2 = f[:,1]
    align = np.max(np.abs(np.sin(pa[:5]-angle))*(1-q[:5]))
    keepminor = clip((qout-.70)/.23+.05)+clip((align-.10)/.34)
    bar = 0.
    for lo,hiend in ((.12,.39),(.19,.52),(.27,.64)):
        coeff = sect(c2,lo,hiend,quality)
        strength = np.hypot(max(0.,coeff.real)+min(0.,coeff.real)*min(1.,keepminor),coeff.imag)
        outc = sect(c2,.69,.98,quality)
        strength -= .60*max(0.,(outc*coeff.conjugate()).real/(abs(coeff)+.001))
        strength *= clip((hiend*size-1.5)/4.5,.21,1.)
        bar = max(bar,strength)
    pair,knots = doubles(s,t,broad,col,cx,cy,angle,size,qout,n2,top)
    # Resolution-adjusted elongation and disky versus spheroidal evidence.
    # The isophotal size is a semi-major axis. Deconvolve only a small part
    # of the PSF since visual shape is the target rather than intrinsic shape.
    qvis = .56*qout+.36*qmid+.08*qin
    qvis = clip(qvis-.65*(1-qvis)/(max(1.3,b[3])**1.5),.07,1.)
    flat = .66*qout+.34*qmid
    flat -= min(.033,.15/(b[4]+.2))*(1-flat)
    extent = clip((size-14.)/14.,-1.,1.4)
    dustdisk = clip((diskclr+.07)/.20,-1.,1.5)
    edge = ((.302-flat)/.065+.62*extent+.48*clip(pointed/.085,-1,1.3)
            +.35*dustdisk+.32*clip((qin-qout-.04)/.18,-1,1.5)
            +.20*clip((fine-.06)/.08,-.8,1.2))
    # First distinguish a close pair from knots strung along a single disk.
    mergerline = 2.05+.70*clip((.40-flat)/.16)+.33*clip((wind-.09)/.09)
    if pair > mergerline-max(0.,min(.6,3.5*(lops-.10))):
        return 1
    if edge > 0. and flat < .445:
        swell,centerfrac = thickness(s,cx,cy,angle,size,max(.13,qout),n2,top)
        bulge = (1.18*clip((swell-1.17)/.31,-1.6,1.7)
                 +.80*clip((qin-qout-.048)/.14,-1.4,1.7)
                 +.59*clip((.49-centerfrac)/.26,-1.3,1.4)
                 +.28*clip((-inclr-.055)/.16,-1,1))
        return 9 if bulge > .05 else 8
    # Disturbance is mainly a breakdown of point symmetry, not simply the
    # presence of fine detail. A regular winding signal argues for a spiral.
    disorder = (.83*clip((lops-.117)/.079,-1.45,2.6)
                +.72*clip((uneven-.146)/.11,-1.35,2.3)
                +.47*clip((devouter-.071)/.063,-1.,2.)
                +.24*clip((one-.21)/.16,-1.,2.)
                +.32*clip(knots-.6,-.6,2.)
                -.65*clip((wind-.078)/.088,0,2.)
                -.42*clip((colorgrad-.16)/.16))
    if disorder > .98 or (lops > .225 and uneven > .19 and wind < .12):
        return 0
    # Evidence of resolved spiral structure: azimuthal modulation which
    # winds, higher harmonics, a blue outer population, or a bar+disk break.
    cblue = clip((diskclr+.028)/.105,-1.6,2.2)
    cgrad = clip((colorgrad-.092)/.12,-1.6,2.)
    texture = clip((fine-.068)/.072,-.94,2.1)
    turning = clip((wind-.061)/.066,-.92,2.5)
    barmorph = (.94*clip((bar-.168)/.14,-1.2,2.5)
                +.55*clip((innerdiff-.106)/.145,-.73,1.8)
                +.49*clip((warp-.095)/.185,-.52,1.8))
    structure = (.67*cblue+.46*cgrad+.48*texture+.69*turning
                 +.47*clip((size-12)/12,-.8,1.4)+.16*clip(knots,0,1.8)
                 +.20*clip((devouter-.055)/.06,-.8,1.2)+.64*clip(barmorph,0.,2.3))
    if structure > -.05 or barmorph > 1.64 or turning > 1.80:
        # Strong bars may be fairly smooth and red inside. Don't require a
        # clean winding estimate in order to recognize one.
        barvote = (barmorph+.22*clip((.5*qout+.5-qin-.22)/.3,-.8,1)
                   -.23*clip((fine-.11)/.12,0,1.4))
        if barvote > .75 and (bar > .14 or innerdiff > .245):
            return 5
        # Winding parameter is cot(pitch angle). Combine it with weaker Hubble
        # sequence cues when only a small part of an arm is resolved.
        reliability = clip((wind-.023)/.12,.0,1.)*clip((size*dq-3)/6,.22,1.)
        tight = (1.35*reliability*clip(np.log(pitch/2.65)/.72,-1.6,1.6)
                 +.44*clip((.10-fine)/.085,-1.35,1.2)
                 +.32*clip((colorgrad-.11)/.145,-1.,1.3)
                 +.34*clip((.19-qout+.40)/.24,-1.3,1.)
                 +.36*clip((.105-uneven)/.085,-1.6,1.1)
                 +.27*clip((size-17)/14,-1.,1.))
        return 6 if tight > .10 else 7
    if lops > .106 and uneven > .125 and (fine > .075 or devouter > .077):
        return 0
    if qvis > .815:
        return 2
    if qvis < .43:
        return 4
    return 3


def predict(x):
    return np.array([classify(im) for im in x],dtype=np.int64)
