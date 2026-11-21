"""Data-free morphology network. Only the output directory is accessed."""
import sys, json, math
from pathlib import Path
import numpy as np

out=Path(sys.argv[1]); out.mkdir(exist_ok=True,parents=True)
layers=[]; weights={}; dims={'image':(3,128,128)}
def layer(name,op,src,shape,**kw):
    inp=[src] if isinstance(src,str) else src
    layers.append(dict(name=name,op=op,inputs=inp,**kw)); dims[name]=tuple(shape); return name
def linear(name,src,w,b=None,spatial=False):
    w=np.asarray(w,np.float32); n=len(w)
    b=np.asarray(b,np.float32) if b is not None else np.zeros(n,np.float32)
    if spatial:
        w=w.reshape(n,dims[src][0],1,1); layer(name,'conv',src,(n,*dims[src][1:]),out=n,kernel=[1,1],stride=1,padding=0)
    else:
        w=w.reshape(n,-1);layer(name,'dense',src,(n,),out=n)
    weights[name+'.W']=w;weights[name+'.b']=b;return name
def reshape(name,x,sh):
    layers.append(dict(name=name,op='reshape',inputs=[x],shape=list(sh)));dims[name]=tuple(sh);return name
def pool(name,x,sy,sx,kind='avg',dy=None,dx=None):
    c,h,w=dims[x];dy=dy or sy;dx=dx or sx
    return layer(name,kind+'pool',x,(c,(h-sy)//dy+1,(w-sx)//dx+1),size=[sy,sx],stride=[dy,dx])
def plus(name,*ins):return layer(name,'add',list(ins),dims[ins[0]])

# Tiny compiler for affine expressions and positive parts. For fields it uses
# shared 1x1 convolutions; for scalars it uses ordinary affine layers.
class E:
    def __init__(self,d=None,b=0.):self.d=d or {};self.b=b
    def __add__(self,v):
        if not isinstance(v,E):return E(self.d,self.b+v)
        d=self.d.copy()
        for i,a in v.d.items():d[i]=d.get(i,0)+a
        return E(d,self.b+v.b)
    __radd__=__add__
    def __mul__(self,k):return E({i:a*k for i,a in self.d.items()},self.b*k)
    __rmul__=__mul__
    def __neg__(self):return self*-1.
    def __sub__(self,v):return self+(-v)
    def __rsub__(self,v):return -self+v
    def __truediv__(self,k):return self*(1./k)

def positive(x):return E({Node(x):1.})
class Node:
    def __init__(self,e):self.e=e;self.depth=1+max((i.depth if isinstance(i,Node) else 0 for i in e.d),default=0)
def maximum(x,y):return positive(x-y)+y
def minimum(x,y):return x-positive(x-y)
def absval(x):return positive(x)*2-x
def clip(x,l,h):return l+positive(x-l)-positive(x-h)
def unit(x):return clip(x,0,1)
def extrem(v,high=True):
    v=list(v);f=maximum if high else minimum
    while len(v)>1:v=[f(v[i],v[i+1]) for i in range(0,len(v)-1,2)]+(v[-1:] if len(v)%2 else [])
    return v[0]
def spline(x,xp,fp):
    slopes=np.diff(fp)/np.diff(xp)
    return fp[0]+sum((positive(x-t)*s for t,s in zip(xp,np.diff([0.,*slopes,0.]))),E())
def lg(x,lo,hi):
    xp=2.**np.arange(math.log2(lo),math.log2(hi)+.49,.5)
    return spline(x,xp,np.log2(xp))
def vec(name,spatial=False):
    n=dims[name][0] if spatial else math.prod(dims[name]);return [E({(name,i):1.}) for i in range(n)]
def compile_(name,seq,spatial=False):
    seq=list(seq);groups={};seen=set();addr={}
    def visit(e):
        for a in e.d:
            if isinstance(a,Node) and a not in seen:
                visit(a.e);seen.add(a);groups.setdefault(a.depth,[]).append(a)
    for e in seq:visit(e)
    def emit(prefix,es):
        tables={};b=np.float32([e.b for e in es]);n=len(es)
        for j,e in enumerate(es):
            for a,k in e.d.items():
                if k==0:continue
                s,i=addr[a] if isinstance(a,Node) else a
                if s not in tables:tables[s]=np.zeros((n,dims[s][0] if spatial else math.prod(dims[s])),np.float32)
                tables[s][j,i]+=k
        parts=[]
        for t,(s,w) in enumerate(tables.items()):
            parts.append(linear(prefix if len(tables)==1 else prefix+'p'+str(t),s,w,b if t==0 else None,spatial))
        return parts[0] if len(parts)==1 else plus(prefix,*parts)
    for d,ns in sorted(groups.items()):
        p=name+'h'+str(d);z=emit(p,[a.e for a in ns]);s=layer(p+'r','relu',z,dims[z]);addr.update({a:(s,i) for i,a in enumerate(ns)})
    return emit(name,seq)

# Gaussian aperture photometry on a uniform angular, roughly logarithmic
# radial mesh. Radii are in input pixel units, not catalog measurements.
small=pool('bin2','image',2,2)
lum=linear('light',small,[[.29,.47,.24]],spatial=True)
colour=linear('red_minus_blue',small,[[1,0,-1]],spatial=True)
y,x=np.mgrid[:64,:64]*2.-63.;A=48;H=A//2
radii=np.array([0.,2.,3.,4.,5.,6.,7.5,9.,11.,13.,15.5,18.5,22.,26.,31.,37.,44.,52.]);R=len(radii)
theta=2*np.pi*np.arange(A)/A

def aperture(cx,cy,sigma):
    d=(x-cx)**2+(y-cy)**2; g=np.exp(-d/(2*sigma**2));return (g/g.sum()).astype('f')
# Estimate each sky channel from several separated groups of sky apertures.
def sky_est(name,src):
    ang=np.arange(24)*2*np.pi/24; samples=[aperture(55*np.cos(t),55*np.sin(t),3.) for t in ang];v=vec(linear(name+'_phot',src,np.array(samples)))
    # Use the lower half of paired aperture means to reject stars; avoid
    # subtracting a minimum pixel (which would be dominated by shot noise).
    return compile_(name,[sum(minimum((v[i]+v[i+12])*.5,(v[i+6]+v[i+18])*.5) for i in range(6))/6])
sky=sky_est('sky',lum)
# A signed colour is not positive for all bright sources, use winsorization.
cs=vec(linear('skycolourphot',colour,np.array([aperture(55*np.cos(t),55*np.sin(t),3.) for t in theta[::4]])))
skyc=compile_('skycolour',[sum(clip(u,-.055,.055) for u in cs)/len(cs)])

def sample(name,src,bg,rads,angles):
    fs=np.array([aperture(r*np.cos(t),r*np.sin(t),max(1.25,r*.095)) for r in rads for t in angles]);
    p=linear(name+'Samples',src,fs);b=linear(name+'Sky',bg,-np.ones((len(fs),1)));return plus(name,p,b)
phot=sample('polar',lum,sky,radii,theta)
cp=sample('chromatic',colour,skyc,radii,theta[::4]);CA=A//4
# Central surface brightness. Dust lanes and slightly off-centre peaks should
# not make a visible outer disk look like sky.
pc=reshape('coregrid',phot,(1,R,A));maxang=pool('annularmaximum',pc,1,A,'max');avgang=pool('annularmean',pc,1,A)
vmax=vec(maxang);va=vec(avgang)
core=compile_('central_peak',[maximum(va[0],extrem(vmax[1:4])*.94)])

def map_rays(name,src,n):return reshape(name,src,(R,1,n))
I=vec(map_rays('raydata',phot,A),True)
# Reflect each ray through the galaxy centre. Even / odd components allow
# paired isophotes to reject projected unrelated sources.
f=reshape('halves',phot,(R*2,1,H));vf=vec(f,True)
sym=[];odd=[];low=[]
for j in range(R):
    a,b=vf[2*j:2*j+2];e=(a+b)*.5;o=absval(a-b)*.5;sym.append(e);odd.append(o);low.append(e-o)
def field(name,es):return compile_(name,es,True)
sy=field('even',sym);od=field('odd',odd);lo=field('paired',low)
# Isophotal widths. Gate by continuity towards the centre to avoid mistaking
# an isolated source farther out for part of the same isophote.
sc=vec(lo,True); ev=vec(sy,True)
cpH=reshape('peak_half',linear('repeatpeak',core,np.ones((H,1))),(1,1,H));pk=vec(cpH,True)[0]
th=[maximum(pk*.135,E(b=.036)),maximum(pk*.34,E(b=.05)),maximum(pk*.61,E(b=.065)),E(b=.085)]
iso=[]
for j in range(R):
    local=sc[j]*.78+ev[j]*.22
    if j>=3:
        m=minimum(local,sc[j-2]*.72+ev[j-2]*.28+.018)
        if j>=6:m=minimum(m,sc[j-4]*.65+ev[j-4]*.35+.03)
    else:m=local
    iso.append(m)
width=[]
# Trapezoidal integration of the soft inside-isophote indicator along a ray.
for t,w in zip(th,[.033,.045,.065,.04]):
    occ=[unit((m-t)/w+.5) for m in iso];width.append(sum((a+b)*(.5*dr) for a,b,dr in zip(occ,occ[1:],np.diff(radii))))
wid=field('radii_at_isophotes',width)
# Circularly smooth over only a few degrees, then take extrema over direction.
smo=np.zeros((4*H,4*H));
for j in range(4):
 for k in range(H):
    for d,h in [(-1,.25),(0,.5),(1,.25)]:smo[j*H+k,j*H+(k+d)%H]+=h
sw=reshape('smoothed_widths',linear('angular_smooth',wid,smo),(4,1,H))
major=vec(pool('semimajor',sw,1,H,'max'));negw=linear('negwidth',sw,-np.eye(4),spatial=True);minor=[-u for u in vec(pool('neg_semiminor',negw,1,H,'max'))]
# Measure the change in position angle of an isophote. Log-radius quadrupoles
# compare like with like across very different apparent sizes.
wv=vec(wid,True);lw=field('logwidth',[lg(a,.5,64) for a in wv]);lwf=vec(lw)
def harmonic(v,m,T):
    return [sum(u*f(m*t) for u,t in zip(v,T))* (2./len(T)) for f in (np.cos,np.sin)]
def amplitude(x,y): # octagonal approximation to Euclidean length
    a=absval(x);b=absval(y);return maximum(maximum(a,b),.70710678*(a+b))
quadr=[harmonic(lwf[j*H:(j+1)*H],2,theta[:H]) for j in range(4)]
# Photometric irregularity: odd power and radial outward rises. Compare a
# putative companion with its inward valley, penalizing unresolved sources.
vlo=vec(lo,True);vodd=vec(od,True);vev=vec(sy,True)
radial=field('radial_excess',[positive(I[j]-minimum(I[j-2],I[max(0,j-4)])) for j in range(3,R)])
rs=vec(radial,True);pack=field('rises_near_centre',[extrem([rs[j-3]*(1 if radii[j]<23 else max(.20,1-(radii[j]-23)/37)) for j in range(3,R-1)])]);comp=vec(pool('companion_contrast',pack,1,A,'max'))[0]
# Companion detector based on a resolved local maximum in 2-D, plus the
# surface brightness of the light joining it to the central galaxy.
# Cartesian grid uses half-pixel centering; sample spacing four input pixels.
grid=np.arange(-40,41,4);G=len(grid);centres=[(u,v) for v in grid for u in grid];D=len(centres)
# Difference of Gaussians requires an actual peak, not just outer-disk flux.
peaks=[];exts=[];bridges=[];ints=[]
for u,v in centres:
    narrow=aperture(u,v,1.65);mid=aperture(u,v,3.25);wide=aperture(u,v,6.5)
    peaks.append(mid-wide);exts.append(1.65*mid-.65*narrow);bridges.append(aperture(.5*u,.5*v,2.6));ints.append(mid)
peakmap=linear('resolved_peak_filters',lum,np.array(peaks+exts+bridges+ints));pm=reshape('neighbour_grid',peakmap,(4,G,G));vf=vec(pm,True)
psky=reshape('neighbour_skygrid',linear('neighbour_sky',sky,np.ones((D,1))),(1,G,G));bg=vec(psky,True)[0]
# Need a rise away from the centre. Compare to a halfway sample and to a
# sample on the same ray roughly one resolution element inward.
inward=linear('inside_peak',lum,np.array([aperture(u*max(.0,1-6/max(1,np.hypot(u,v))),v*max(.0,1-6/max(1,np.hypot(u,v))),2.15) for u,v in centres]));inside=vec(reshape('inside_grid',inward,(1,G,G)),True)[0]
strength=minimum(1.6*vf[0],minimum(vf[3]-minimum(inside,vf[2]),.85*(vf[1]-bg)))
# More credit for an extended peak joined by diffuse light.
strength+=.22*clip(vf[2]-bg,0,.22)
peakval=field('peak_strength',[strength]);pv=vec(peakval)
comp2=extrem([p*min(1.,19./max(1,np.hypot(u,v))) for p,(u,v) in zip(pv,centres) if 5<np.hypot(u,v)<43]);
# Angular Fourier photometry: odd modes measure lopsidedness; rotation of
# the even modes with radius distinguishes arms from a fixed inclined disk.
# Mask the non-central contribution on each ray using the preceding radii.
clean=[]
for j in range(R):
    m=vev[j]-.32*vodd[j]
    if j>=4:m=minimum(m,vev[j-3]+.035)
    clean.append(positive(m-.012))
cl=field('clean_even_light',clean);fv=vec(cl);T=theta[:H]
hr=[{m:harmonic(fv[j*H:(j+1)*H],m,T) for m in (2,4,6)} for j in range(R)]
hmean=[sum(fv[j*H:(j+1)*H])/H for j in range(R)]
# Colour of core and body. Aperture weighting downweights sky-dominated rings
# in both numerator and denominator; a signed log ratio is an opponent colour.
cv=vec(cp);clrs=[sum(cv[j*CA:(j+1)*CA])/CA for j in range(R)]
cen=vec(core)[0]

def opponent(c,i):
    # log2(red/blue), relative to a neutral source (weak regularization at sky)
    return lg(i+c*.5+.016,1/128,2)-lg(i-c*.5+.016,1/128,2)
corecol=opponent(sum(clrs[:4])/4,sum(va[:4])/4)
bodyix=[4,6,8,10]; bodycol=opponent(sum(clrs[j] for j in bodyix)/4,sum(va[j] for j in bodyix)/4)
outerix=[8,9,10,11,12];outercol=opponent(sum(clrs[j] for j in outerix)/5,sum(va[j] for j in outerix)/5)
# Relative power of angular departures. Normalize per-galaxy flux, not each
# ring separately (a ring consisting entirely of sky cannot provide arms).
jj=list(range(4,14));ww=np.array([radii[j]**.72 for j in jj]);ww/=ww.sum()
flux=sum(w*hmean[j] for w,j in zip(ww,jj)); normflux=lg(flux+.018,1/256,2)
def fraclog(x,off=.0015):return lg(x+off,1/1024,2)-normflux
a2=[amplitude(*h[2]) for h in hr];a4=[amplitude(*h[4]) for h in hr]
# Oppositely wound arms both count. Use the minimum odd-reflection residual
# over candidate major axes; any concentric elliptical profile has a common
# mirror axis in all annuli, regardless of radial falloff or ellipticity.
res=[]
for t in np.arange(18)*np.pi/36:
    def anis(j,m):
        c,s=hr[j][m];return absval(s*np.cos(m*t)-c*np.sin(m*t))
    res.append(sum(w*(anis(j,2)+.5*anis(j,4)+.25*anis(j,6)) for w,j in zip(ww,jj)))
mirror=fraclog(extrem(res,False))
# Nonmonotonic even light along rays and small-scale high-m structure.
rayrises=[positive(vlo[j]-minimum(vlo[j-2],vlo[max(1,j-4)])-.011) for j in range(5,14)]
rr=vec(pool('mean_even_rises',field('even_rises',rayrises),1,H));bump=fraclog(sum(w*u for w,u in zip(ww[1:],rr)))
# Relative odd light in an inner aperture and a wider one. Pair differences
# beyond an already sky-dominated ring receive less credit.
oduse=[minimum(vodd[j],positive(vev[max(1,j-3)])+.018) for j in range(R)]
oddmeans=vec(pool('annular_odd_mean',field('odd_light',oduse),1,H));
oddlog=lg(sum(w*oddmeans[j] for w,j in zip(ww,jj))+.002,1/512,2)-normflux
innodd=lg(sum(oddmeans[2:9])/7+.001,1/1024,2)-lg(sum(va[2:9])/7+.015,1/256,2)
# Coherence of m=2 along logarithmic spirals of different pitch. A tight
# spiral winds through a larger phase change per logarithmic radius.
wind=[]
for pitch in [0,12,18,26,38,55]:
    hands=[]
    for sgn in ([1,-1] if pitch else [1]):
        phase=2*sgn*(0 if not pitch else 1/np.tan(np.radians(pitch)))*np.log(radii[jj]/10)
        cx=sum(w*(hr[j][2][0]*np.cos(t)+hr[j][2][1]*np.sin(t)) for j,w,t in zip(jj,ww,phase));cy=sum(w*(hr[j][2][1]*np.cos(t)-hr[j][2][0]*np.sin(t)) for j,w,t in zip(jj,ww,phase));hands.append(amplitude(cx,cy))
    wind.append(extrem(hands))
# Remove stationary (inclination) mode by differencing annuli; measure the
# change in quadrupole direction/strength in fractional isophote units.
qshift=amplitude(quadr[0][0]-quadr[2][0],quadr[0][1]-quadr[2][1]);qshiftmid=amplitude(quadr[0][0]-quadr[1][0],quadr[0][1]-quadr[1][1]);
# Compare the central light distribution with a bank of elliptical profiles.
# Every row is ONE analytic template divided by its own norm. Aperture
# weighting limits the influence of neighbouring objects. Pooling chooses
# the best position angle, scale, index and flattening independently at test
# time. No image other than the one being classified is involved.
Q=np.array([1.,.84,.68,.54,.415,.31,.235,.17]); S=np.array([3.,4.25,6.,8.5,12.,17.,24.]); nt=16
# Crop and downsample before matching. The location is fixed by the task's
# centred cutout convention, not the position of any particular galaxy.
cw=20; sl=slice(32-cw,32+cw); xx=x[sl,sl];yy=y[sl,sl];ap=np.exp(-(xx**2+yy**2)/(2*17.**2));
# Direct weighted norm of the target via a piecewise-linear square.
# A continuous norm over direction, unlike a binary mask, remains sensitive
# to a faint close companion and to arms in an otherwise bright galaxy.
fw=np.eye(64*64,dtype=np.float32)[np.arange(64*64).reshape(64,64)[sl,sl].ravel()];crop=linear('central_crop',lum,fw);bgpatch=linear('crop_sky',sky,-np.ones((xx.size,1)));cent=reshape('patch',plus('crop_signal',crop,bgpatch),(1,*xx.shape));v=vec(cent,True)[0];cuts=np.array([-.25,-.08,-.04,-.02,0,.02,.04,.08,.12,.17,.24,.34,.48,.68,1.]);squared=field('light_squared',[spline(v,cuts,cuts**2)]);energy=vec(linear('weighted_energy',squared,ap.reshape(1,-1)))[0]
# Normalized correlation bank. An exponential scale length and a Gaussian
# standard deviation are each projected according to the intrinsic axis ratio,
# with seeing represented by an isotropic core added in quadrature.
templates=np.empty((len(Q),len(S),2,nt,xx.size),np.float32)
for a,q in enumerate(Q):
 for b,scale in enumerate(S):
    for k,t in enumerate(np.arange(nt)*np.pi/nt):
        X=xx*np.cos(t)+yy*np.sin(t);Y=yy*np.cos(t)-xx*np.sin(t)
        for h,n in enumerate([1.,.5]):
            psf=1.25 if h==0 else 1.65
            rho2=X**2/(scale**2+psf**2)+Y**2/((q*scale)**2+psf**2)
            g=np.exp(-((rho2+.075**2)**(.5/n))*(1 if h==0 else .5));templates[a,b,h,k]=(g*ap/np.sqrt(np.sum(g*g*ap))).ravel()
fits=linear('elliptical_profile',cent,templates.reshape(-1,xx.size));mp=reshape('model_grid',fits,(len(Q)*len(S),1,2*nt));ms=pool('best_profile',mp,1,2*nt,'max');scoregrid=reshape('model_scores',ms,(1,len(Q),len(S)));qmax=vec(pool('axisratio_scores',scoregrid,1,len(S),'max'));smax=vec(pool('scale_scores',scoregrid,len(Q),1,'max'));top=extrem(qmax)
# Within the shallow peak, interpolate the axis ratio on a log grid.
# Around its optimum, normalized Gaussian overlap has a quadratic loss in
# log axis ratio; compare the minima on the two sides of each interval.
def softcoord(vals,coordinates):
    coord=E(b=coordinates[0]);hi=lg(top,1/16,64)
    for k in range(1,len(vals)):
        right=lg(extrem(vals[k:]),1/16,64);left=lg(extrem(vals[:k]),1/16,64)
        coord+=(coordinates[k]-coordinates[k-1])*unit(.5+(right-left)*90.)
    return coord
model_e=softcoord(qmax,-np.log2(Q));model_s=softcoord(smax,np.log2(S))
# log norm minus log correlation. Around a good match, a small log gap is
# proportional to the fraction of energy not explained by the ellipse.
gap=.5*lg(energy,1/64,4096)-lg(top,1/16,64)
# Disk-plane photometry. Rather than reading a logarithmic spiral through
# the projected minor axis, sample in candidate deprojected coordinates and
# select the projection with the best elliptical-disk match. First reduce
# the sampled ellipse to Fourier coefficients; the angular sampling array
# itself does not become a parameter of a subsequent nonlinear stage.
qs=[0,2,3,4,5]; na=8; js=[3,4,5,6,7,8,9,10,11,12,13,14,15]; nj=len(js); nd=len(qs)*na
angfit=reshape('angles_per_model',fits,(len(Q),2*len(S),nt));byangle=vec(pool('anglescore',angfit,2*len(S),1,'max'))
projs=[(a,t) for a in qs for t in range(0,nt,nt//na)];candidate=[byangle[a*nt+t] for a,t in projs]
cbest=extrem(candidate);delta=compile_('projection_loss',[cbest-a for a in candidate]);dfield=vec(reshape('view_error',delta,(1,1,nd)),True)[0]
ty,tx=np.mgrid[:2*cw,:2*cw];ntheta=32;phis=np.arange(ntheta)*2*np.pi/ntheta
trig=np.stack([np.ones(ntheta),2*np.cos(phis),2*np.sin(phis),2*np.cos(2*phis),2*np.sin(2*phis),2*np.cos(3*phis),2*np.sin(3*phis),2*np.cos(4*phis),2*np.sin(4*phis)])/ntheta
mfilters=np.empty((nj,9,nd,xx.size),np.float32)
for p,(a,t) in enumerate(projs):
    t*=np.pi/nt
    for jr,j in enumerate(js):
        u=radii[j]*np.cos(phis);v=radii[j]*Q[a]*np.sin(phis);xs=u*np.cos(t)-v*np.sin(t);ys=u*np.sin(t)+v*np.cos(t);sigma=max(1.25,.075*radii[j]);
        ga=np.exp(-((xx[None]-xs[:,None,None])**2+(yy[None]-ys[:,None,None])**2)/(2*sigma**2)).reshape(ntheta,-1);ga/=ga.sum(axis=1,keepdims=True);mfilters[jr,:,p]=trig@ga
mh=linear('disk_plane_harmonics',cent,mfilters.reshape(-1,xx.size));fields=vec(reshape('disk_plane_views',mh,(nj*9,1,nd)),True)
pfield=vec(reshape('replicate_central_peak',linear('projection_peak',core,np.ones((nd,1))),(1,1,nd)),True)[0]
scale_field=vec(reshape('scale_per_view',linear('copy_scale',compile_('profile_size',[model_s]),np.ones((nd,1))),(1,1,nd)),True)[0]
# Normalize each ring by its own local light. Piecewise log-ratios avoid
# boosting quantization noise in very low surface brightness annuli.
mono=[];spir=[];oddamp=[];quadamp=[];barstrength=[];four=[]
for jr,j in enumerate(js):
    m,c1,s1,c2,s2,c3,s3,c4,s4=fields[9*jr:9*jr+9];lm=lg(maximum(m+.026,E(b=.025)),1/64,2);mono.append(m)
    quadamp.append(amplitude(c2,s2));oddamp.append(amplitude(c1,s1)+.6*amplitude(c3,s3));four.append(amplitude(c4,s4))
    # Use the phase of the doubled azimuth, not its value across the
    # projected image. Normalization by a common disk flux is done below.
    spir.append((c2,s2));barstrength.append(lg(quadamp[-1]+.005,1/256,2)-lm)
# Winding-mode evidence in two broad radial zones, with emphasis on the
# outskirts. Subtract the constant component of each radial wavelet; a
# straight bar should not accumulate tight-spiral evidence.
coefam=[];powers=[]
for idx in [[2,3,4,5,6,7,8],[5,6,7,8,9,10,11,12]]:
    w=np.hanning(len(idx)+2)[1:-1];w/=w.sum();mean=sum(wi*positive(mono[i]) for i,wi in zip(idx,w));den=lg(mean+.021,1/64,2)
    powers.append(lg(sum(wi*(quadamp[i]+four[i]*.5) for wi,i in zip(w,idx))+.003,1/512,2)-den)
    co=[]
    for pitch in [13,20,30,43,63]:
        hands=[]
        for sign in [-1,1]:
            z=np.exp(2j*sign/np.tan(np.radians(pitch))*np.log(radii[np.array(js)[idx]]));z=z-sum(z*w);n=np.sqrt(sum(abs(z)**2*w));z/=max(.35,n)
            u=sum(wi*(spir[i][0]*zz.real+spir[i][1]*zz.imag) for wi,i,zz in zip(w,idx,z));v=sum(wi*(spir[i][1]*zz.real-spir[i][0]*zz.imag) for wi,i,zz in zip(w,idx,z));hands.append(amplitude(u,v))
        co.append(extrem(hands))
    coefam.append([lg(extrem(cs)+.004,1/256,2)-den for cs in [co[:2],co[2:]]])
# A bar has inner m=2 power substantially in excess of a round PSF-blurred
# bulge and often decreases towards the arm-dominated region.
innerbar=extrem(barstrength[1:7]);arm=maximum(powers[0],powers[1]);tight=maximum(coefam[0][0],coefam[1][0]);loosew=maximum(coefam[0][1],coefam[1][1]);
# Odd structure in the blue disk and departures of the EVEN light from a
# smooth ellipse are allowed to vote independently.
wwd=np.array([1.,1.,1.,1.2,1.2,1.2,1.,1.,.7,.5,.35,.25,.2]);wwd/=wwd.sum();dmean=sum(w*m for w,m in zip(wwd,mono));odisk=lg(sum(w*o for w,o in zip(wwd,oddamp))+.004,1/256,2)-lg(dmean+.022,1/64,2)
viewfeats=[arm,tight,loosew,innerbar,odisk];selected=[]
for i,v in enumerate(viewfeats):
    # A maxout switch with large cost for suboptimal viewing geometry. In a
    # tie, two similar viewing geometries can supply supporting evidence.
    selected.append(v-48*dfield)
selected=vec(pool('selected_disk_features',field('view_gating',selected),1,nd,'max'))
# A multiscale difference-image test. Fixed Gaussian prefilters separate
# a true one-sided tail / offset nucleus from symmetric arms and curvature
# of an inclined disk. Use a soft relative surface-brightness floor so sky
# in an otherwise empty annulus is not interpreted as a feature.
rb=vec(reshape('radius_peak_repeat',linear('radius_repeat',core,np.ones((H,1))),(1,1,H)),True)[0]
wt=[radii[j]**1.1*np.exp(-.5*(radii[j]/15.)**2) for j in range(R)]; wt=np.array(wt)/sum(wt)
signed_flux=sum(w*positive(m-.02) for w,m in zip(wt,vev)); asymmetricflux=sum(w*positive(o-.015-.03*rb) for w,o in zip(wt,vodd))
# Clipping the paired odd component with the previous annulus suppresses
# spikes very far from the object; inner-heavy and outer-heavy evidence
# remain distinct diagnostics.
asym_in=sum(w*positive(o-.012-.02*rb) for w,o in zip(wt[:10],vodd[:10])); flux_in=sum(w*positive(m-.016) for w,m in zip(wt[:10],vev[:10]));
aa=vec(pool('CAS_means',field('CAS_rays',[signed_flux,asymmetricflux,asym_in,flux_in]),1,H));cas=lg(aa[1]+.001,1/1024,2)-lg(aa[0]+.008,1/256,2);casi=lg(aa[2]+.0005,1/2048,2)-lg(aa[3]+.004,1/256,2)
# Resolved central-body area and brightness; an inclined thin strip must
# keep a long contiguous midplane at a substantial fraction of peak light.
# Polar quadrature is in square pixels. A soft occupancy is used in place
# of pixel counting at an infinitely sharp isophote.
area_w=np.gradient(radii)*radii*2*np.pi
aparea=sum(area_w[j]*unit((vlo[j]-.082)/.04+.5) for j in range(R));barea=sum(area_w[j]*unit((vlo[j]-maximum(.35*rb,E(b=.067)))/.04+.5) for j in range(R));ag=vec(pool('mean_areas',field('projected_areas',[aparea,barea]),1,H));areal=.5*lg(ag[0]/np.pi+.5,.5,4096);areah=.5*lg(ag[1]/np.pi+.5,.5,4096)
# A ridge-valley-ridge test for a second extended nucleus in ANY direction,
# including pairs whose common midpoint (rather than a nucleus) is centred.
# Compare both peaks with the light along the line joining them. Several
# resolved scales reject isolated PSF-like peaks. The join must have excess
# diffuse light -- a mere projection elsewhere in the stamp isn't enough.
# Use a coarse square grid for pairs and local Gaussian averages for the
# subgrid samples along the path. A less central peak should be stronger to
# count equally. Only the set of coordinates below determines these masks.
coords=np.arange(-24,25,4);pairs=[]
for y1 in coords:
 for x1 in coords:
    if np.hypot(x1,y1)>13:continue
    for y2 in coords:
     for x2 in coords:
        d=np.hypot(x2-x1,y2-y1);r1=np.hypot(x1,y1);r2=np.hypot(x2,y2)
        if not (6<=d<=34 and r1<=r2 and (r1<6 or np.hypot(x1+x2,y1+y2)<.85*d)):continue
        pairs.append(((x1,y1),(x2,y2)))
# Pack the samples into a single reduced grid (integer input-pixel steps).
locs={}; gather=[]
for (x1,y1),(x2,y2) in pairs:
    addresses=[]
    for x_,y_,s_ in [(x1,y1,2),(x2,y2,2),(x1,y1,4),(x2,y2,4)]+[(round(x1+(x2-x1)*t),round(y1+(y2-y1)*t),2) for t in [.25,.5,.75]]:
        key=x_,y_,s_;addresses.append(locs.setdefault(key,len(locs)))
    gather.append(addresses)
fs=np.array([aperture(u,v,s)[sl,sl] for u,v,s in locs]);sam=linear('pair_apertures',cent,fs);allg=np.asarray(gather);ng=len(pairs)
# Route samples via separate dense maps for each field to limit im2col size.
pp=[]
for j in range(allg.shape[1]):
    w=np.zeros((ng,len(locs)),np.float32);w[np.arange(ng),allg[:,j]]=1.;s=linear('pair_sample'+str(j),sam,w);pp.append(vec(reshape('pair_f'+str(j),s,(1,1,ng)),True)[0])
p1,p2,b1,b2,v1,v2,v3=pp;bridge=minimum(v1,minimum(v2,v3));faint=minimum(p1,p2);broad=minimum(b1,b2);dip=positive(faint-bridge);extended=positive(broad-.39*faint);envelope=positive(bridge-.018)
# All terms have intensity units before division by the central peak.
pairval=minimum(dip,minimum(extended*2.1,envelope*2.8))
pairscore=vec(pool('close_double_max',field('close_double_strength',[pairval]),1,ng,'max'))[0];pairR=lg(pairscore+.004,1/256,2)-lg(cen+.05,1/64,2)
# Direct alternative-hypothesis comparison. Each continuous model below
# becomes exactly one weight row, using only its own aperture-weighted norm.
# Classes overlap intentionally: gains relative to a single smooth ellipse,
# not the raw response to the bright nucleus, indicate arms, tails, or a pair.
base3=pool('bin4',small,2,2);mono3=linear('bin4light',base3,[[.29,.47,.24]],spatial=True);side=24;ss=slice(4,28);yx=np.arange(-46.,47,4.);X,Y=np.meshgrid(yx,yx);AP=np.exp(-(X**2+Y**2)/(2*16.**2));tpf=[];counts=[]
sub=np.eye(1024,dtype=np.float32)[np.arange(1024).reshape(32,32)[ss,ss].ravel()];pat=plus('frame4',linear('crop4',mono3,sub),linear('sky4',sky,-np.ones((side**2,1))))
def save_template(g):
    # Neither normalization nor background subtraction ever involves another
    # generated image. The input has already had the measured sky removed.
    tpf.append((AP*g/np.sqrt((AP*g*g).sum())).astype('f').ravel())
def finish_grp():counts.append(len(tpf)-sum(counts))
for cls in ['simple','arms','bar','tail','double','flatdisk','thickcore']:
 for angle in np.arange(12 if cls not in ['tail','double'] else 20)*(np.pi/12 if cls not in ['tail','double'] else 2*np.pi/20):
    u=X*np.cos(angle)+Y*np.sin(angle);v=Y*np.cos(angle)-X*np.sin(angle)
    if cls=='simple':
        for sz in [3.,4.4,6.5,9.5,14.,21.]:
         for q in [.22,.31,.42,.55,.7,.85,1.]:
            r=u*u/(sz**2+2.5)+v*v/((sz*q)**2+2.5)
            for n in [.5,1.,1.6]:save_template(np.exp(-np.power(r+.008,.5/n)*(1 if n>=1 else .5)))
    elif cls in ('arms','bar'):
        for sz in [5.5,8.,12.,17.5]:
         for q in [.48,.72,.96]:
            r=np.sqrt(u**2+v**2/q**2+2.);a=np.arctan2(v/q,u);s=r/sz;disc=np.exp(-s);bul=np.exp(-.5*r*r/(2.3+.13*sz)**2);fade=1-np.exp(-s*s*2.)
            if cls=='arms':
                for pitch in [16,34]:
                 for hand in [1,-1]:
                  for phase in np.arange(6)*2*np.pi/6:
                    wave=np.cos(2*a+phase-hand*2./np.tan(np.radians(pitch))*np.log(.30+s))
                    save_template(disc*(1+.60*fade*wave)+.35*bul)
            else:
                for phi in np.arange(4)*np.pi/4:
                    xb=u*np.cos(phi)+v/q*np.sin(phi);yb=v/q*np.cos(phi)-u*np.sin(phi);barprof=np.exp(-.5*((xb/(1.20*sz))**2+(yb/(2.+.14*sz))**2));
                    for hand in [1,-1]:
                        wave=np.cos(2*(a-phi)-hand*4.*np.log(np.maximum(1.,s)/1.3));save_template(.82*disc*(1+.43*fade*wave)+.63*barprof+.23*bul)
    elif cls=='tail':
        for sz in [3.8,6.,9.,14.]:
         for q in [.4,.68,1.]:
          for bend in [-.65,.65]:
            # A one-sided, curved envelope around a relatively compact body.
            t=u/sz;z=v/(sz*q);warp=bend*np.maximum(t-.3,0)**1.4*.44
            main=np.exp(-np.sqrt(t*t+z*z+.02));wake=np.exp(-.5*((t-1.3)**2/1.5+(z-warp)**2/.55));save_template(main+.52*wake)
            # Off-centre star forming knot inside a diffuse lopsided body.
            knot=np.exp(-.5*((t+.48)**2+(z-.38)**2)/(.45+1./sz)**2);save_template(.82*main+.65*knot+.35*wake)
    elif cls=='double':
        for sc in [3.,5.,8.]:
         for sep in [1.7,3.1,5.4]:
          for shift in [0.,.50]:
           for ratio in [.48,1.]:
            dx=sep*sc;u1=u+shift*dx;u2=u1-dx;body=np.exp(-np.sqrt((u1/sc)**2+(v/(sc*.85))**2+.03));companion=np.exp(-np.sqrt((u2/(sc*.88))**2+(v/(sc*.82))**2+.03));envelope=np.exp(-.5*((u1-.48*dx)**2/(.6*dx+sc)**2+v*v/(sc*1.55)**2));save_template(body+ratio*companion+.13*envelope)
    else:
        for sc in [6.,9.,13.,19.]:
         for q in [.10,.17,.26]:
            # A thin exponential midplane, tapered ends, and optional thicker
            # spheroidal bulge; seeing keeps even an ultra-thin disk resolved.
            w=np.sqrt(2.4+(sc*q)**2);disk=np.exp(-.5*np.power(abs(u)/(sc*1.34),1.45)-.5*(v/w)**2)
            if cls=='flatdisk':save_template(disk)
            else:
                for h in [.52,1.2]:save_template(disk+h*np.exp(-.5*((u/(sc*.57))**2+(v/(sc*.34+1.))**2)))
 finish_grp()
templres=linear('morphological_hypotheses',pat,np.asarray(tpf));del tpf
# A max reduction per physical family; pad the last dimension with repeated
# rows only where necessary. Reduce via a tree of small pooling ops without
# allocating a one-hot routing matrix over the entire bank.
# Separate dense banks are more memory efficient than such a routing matrix.
allrows=weights.pop(templres+'.W');weights.pop(templres+'.b');layers.pop();del dims[templres];off=0;hyp=[]
for j,n in enumerate(counts):
    resp=linear('hypothesis'+str(j),pat,allrows[off:off+n]);off+=n
    hyp.append(vec(pool('best_hypothesis'+str(j),reshape('bank'+str(j),resp,(1,1,n)),1,n,'max'))[0])
hypl=[lg(v,1/16,32) for v in hyp];gain=[clip((v-hypl[0])*100,-6.,9.) for v in hypl[1:]]
# Global blue/diffuse light is especially informative for a disturbed disk,
# unlike a red interacting pair; matched colour differences reduce sky noise.
# Synthesize the scalar feature vector before applying the hand-written rules.
Lmaj=[lg(z,1,64) for z in major];Lmin=[lg(z,.7,64) for z in minor];elong=[a-b for a,b in zip(Lmaj,Lmin)]
compR=lg(comp+.006,1/256,1)-lg(cen+.05,1/64,2);compR2=lg(comp2+.006,1/256,1)-lg(cen+.065,1/64,2)
feature_names=['size','sizeabs','elong','elongmid','elonghi','extent','colour','bodycolour','outcolour','mirror','bump','odd','innerodd','comp','resolved','twist','twistmid','tight','loose','stationary','peak','area','logconc','arm4']
features=[Lmaj[0],Lmaj[3],elong[0],elong[1],elong[2],Lmaj[0]-Lmaj[2],corecol,bodycol,outercol,mirror,bump,oddlog,innodd,compR,compR2,qshift,qshiftmid,fraclog(extrem(wind[1:3])),fraclog(extrem(wind[3:])),fraclog(wind[0]),cen,(Lmaj[0]+Lmin[0])*.5,Lmaj[0]-Lmaj[1],fraclog(sum(w*a4[j] for w,j in zip(ww,jj)))]
feature_names+=['model_e','model_s','gap','diskarm','disktight','diskloose','diskbar','diskodd','cas','casi','areal','areah','pair','brightmaj'];features.extend([model_e,model_s,gap,*selected,cas,casi,areal,areah,pairR,Lmaj[1]]);feature_names+=['gainarm','gainbar','gaintail','gainpair','gaindisk','gainbulge'];features+=gain;feat=compile_('morphology',features);f=dict(zip(feature_names,vec(feat)))

# Fuzzy morphological decision tree. Inputs are in log2 pixel, axis-ratio,
# flux-ratio and colour units. Margins denote graded evidence, not votes
# estimated from sample frequencies.
size=f['size'];asz=f['sizeabs']; e=f['elong']*.61+f['elongmid']*.39; ei=f['elonghi']; col=f['colour']; bc=f['bodycolour'];oc=f['outcolour'];odd=f['odd'];io=f['innerodd'];mir=f['mirror'];bump=f['bump'];pc=f['comp'];rc=f['resolved'];tw=f['twist'];tm=f['twistmid'];
# Size is also morphological evidence: resolving arms and the vertical
# structure of a disk requires several resolution elements across the body.
visible=.64*size+.36*asz
red=clip(col,-1.5,1.5);bluebody=clip(-bc,-1,1);gradient=clip(col-oc,-.5,1.2)
e=.40*e+.60*f['model_e'];roundedge=-np.log2(.79); flatboundary=-np.log2(.54)
# Scores within each family: best member carries no penalty, so the number
# of morphologies in a family doesn't confer an advantage over other ones.
def relscores(ss):
    m=extrem(ss);return [s-m for s in ss]
smoothparts=relscores([2.8*(roundedge-e),E(),2.15*(e-flatboundary)])
# Bulge is signalled by a red concentrated central component that is rounder
# than the outer disk. A uniformly thin blue disk need not have a bulge.
bulge=1.17*red+.45*bc+.52*clip(e-ei,-1.2,1.2)+.43*(f['extent']-1.65)+.75*(f['peak']-.49)-.52
bulge+=.20*clip(f['gainbulge']-f['gaindisk']-.55,-2.,2.)+.45*(gradient-.5)
edgeparts=relscores([-bulge,bulge]);
# A bar changes inner ellipticity with a relatively coherent major axis;
# a loose unbarred arm makes stronger, longer coherent nonradial ripples.
bar=.84*clip(ei-f['elong'],-1.,1.2)+.67*clip(tw-.22,-.4,1.)+.44*red+.31*gradient+.32*clip(f['diskbar']+2.3,-1.1,1.2)-.14
loose=.38*clip(mir+3.5,-1.5,1.8)+.25*clip(bump+5.5,-1.6,1.8)+.50*clip(f['diskloose']-f['disktight'],-1.4,1.4)+.54*clip(f['diskloose']+3.4,-1.8,1.8)-.40*red+.18*gradient
bar+=.27*clip(f['gainbar']-f['gainarm'],-1.6,2.1)
loose+=.38*clip(f['diskodd']+2.8,-1.4,1.7)+.34*clip(3.9-f['brightmaj'],-1.,1.)
spiralparts=relscores([bar, -.53*loose-.2*bar,loose-.18*bar])
# Physical tests used for the top-level classes.
# Demand a *resolved disk* as well as structure. The radius of a circle of
# equal area and the semimajor axis of the bright body are less vulnerable
# to an isolated bright star than the maximum width of a faint isophote.
diskarea=.72*(f['areal']-4.0)+.48*(f['areah']-3.35)+.54*(visible-4.26)
structure=.38*clip(mir+3.4,-1.7,2.)+.51*gradient+.17*bluebody-.24*positive(red)
altarm=maximum(f['gainarm'],f['gainbar']);spiral=diskarea+structure+.44*clip(altarm-.75,-3.,3.)-.94*positive(e-.98)
edge=1.65*(e-1.5)+1.0*(f['brightmaj']-3.5)+.56*(asz-4.17)+.36*(f['model_s']-3.0)+.38*positive(e-2.)-.32*positive(mir+2.55)+.38*clip(maximum(f['gaindisk'],f['gainbulge'])-.65,-2.6,2.8)
# Monotonic symmetric light strongly disfavors both interactions. Require
# the two-nucleus candidate to be resolved or attached to detectable light.
interact=.79*clip(odd+2.85,-2.5,2.8)+.66*clip(io+3.35,-2.5,2.6)+.65*clip((f['gap']-.065)*7,-.5,1.7)
# A companion can be disconnected from a symmetric central galaxy. Use an
# alternative path which does not require strong asymmetry in the nucleus.
separated=1.42*clip(rc+2.55,-2.3,2.1)+.48*clip(pc+2.05,-2.,2.)+.48*red+.38*interact
# Fractions of one-sided residual light against the sky-subtracted central
# body. A locally bright double is especially significant when BOTH peaks
# are separated by a saddle, instead of merely contrasting with the sky.
merger=.95*clip(f['pair']+2.65,-2.8,2.5)+.60*clip(rc+2.35,-2.,2.)+.40*clip(f['cas']+3.2,-2.,2.5)+.46*red+.18*clip(f['diskodd']+2.4,-1.5,1.5)
peculiar=.48*clip(f['cas']+3.35,-2.4,2.7)+.58*clip(f['casi']+3.45,-2.4,2.7)+.38*clip(f['diskodd']+2.7,-1.5,2.)+.42*clip(mir+3.45,-1.3,1.8)+1.18*bluebody+.65*(.5-f['peak'])+.27*clip((f['gap']-.055)*8,-.5,1.7)-.15
# A high-contrast pair takes precedence over ordinary one-sidedness. A
# feature-rich two-armed disk is usually point-symmetric, not irregular.
merger=maximum(merger,separated-.2)+.54*clip(f['gainpair']-.60,-1.5,3.)
peculiar=.58*peculiar+.42*interact+.68*clip(f['gaintail']-.35,-1.8,3.)+.48*bluebody+.42*clip(f['diskodd']+2.85,-1.5,2.)-.24*positive(f['pair']+2.45)
# Strong evidence for a large underlying disk should not be annulled by
# a single projected source in its outskirts.
merger-=.37*positive(visible-4.3)+.35*positive(e-1.4)
peculiar-=.46*positive(e-1.2)
scores=[peculiar,merger,*smoothparts,*[spiral+s for s in spiralparts],*[edge+s for s in edgeparts]]
compile_('scores',scores)
(out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n');np.savez(out/'weights.npz',**weights)
