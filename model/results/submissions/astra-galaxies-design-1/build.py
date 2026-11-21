"""Construct a galaxy morphology instrument. No input files, example images or fitted coefficients."""
import json, math, sys
from pathlib import Path
import numpy as np

out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
layers=[]; weights={}
def layer(name,op,ins,**settings):
    layers.append(dict(name=name,op=op,inputs=ins if isinstance(ins,list) else [ins],**settings)); return name
def dense(name,src,W,b=0):
    W=np.asarray(W,'f').reshape(len(W),-1); weights[name+'.W']=W
    weights[name+'.b']=np.broadcast_to(b,(len(W),)).astype('f');return layer(name,'dense',src,out=len(W))
def conv(name,src,W,b=0,stride=1,padding=0):
    W=np.asarray(W,'f');weights[name+'.W']=W;weights[name+'.b']=np.broadcast_to(b,(len(W),)).astype('f')
    return layer(name,'conv',src,out=len(W),kernel=list(W.shape[2:]),stride=stride,padding=padding)
def relu(name,src):return layer(name,'relu',src)
def pool(name,src,s,method='avgpool',stride=None):
    return layer(name,method,src,size=list(s),stride=list(stride or s))

# Sampling on a log-polar grid: fine in the region around the nucleus, coarse in the outskirts.
# Antipodal pairing and phases of azimuthal harmonics then become fixed linear maps.
N=24; radii=np.array([2.,3.5,5.5,8.,11.5,16.,22.5,32.,45.]); R=len(radii)
ph=np.arange(N)*2*np.pi/N
pixel=pool('two_pixel_photometry','image',(2,2))
y,x=(np.indices((64,64))-31.5)*2
lum=np.array([.35,.4,.25]); col=np.array([1.,0.,-1.]); channels=[lum,col]
aps=[]
for v in channels:
    for r in radii:
        for a in ph:
            u=x*np.cos(a)+y*np.sin(a)-r; t=y*np.cos(a)-x*np.sin(a)
            g=np.exp(-.5*((u/(1+.075*r))**2+(t/(1+.10*r))**2));g/=g.sum()
            aps.append((v[:,None,None]*g).astype('f'))
# Local, per-frame sky estimate; don't assume the tone-mapped JPEG maps sky to zero.
rpix=np.hypot(x,y);sky=np.exp(-.5*((rpix-55)/6)**2);sky/=sky.sum()
for v in channels:aps.append(v[:,None,None]*sky)
phot=dense('polar_photometer',pixel,aps); del aps
npol=R*N; nraw=2*npol+2

def basis():return np.zeros(nraw)
def mask(i,j=0):
    m=basis();m[j*npol+i*N:j*npol+(i+1)*N]=1;return m

def skybasis(j):
    m=basis();m[2*npol+j]=1;return m
# Estimate the flux and colour in the central few seeing disks.
ref=(mask(0)+mask(1))/(2*N)-skybasis(0)
# Scalar feature arithmetic is compiled to dense/ReLU blocks. Instances of a Ref denote
# a physical observable or an intermediate hinge; Lin denotes just a weighted sum.
class Lin:
    def __init__(self,d=None,b=0):self.d=d or {};self.b=float(b)
    def __add__(self,o):
        o=aslin(o); d=self.d.copy()
        for k,v in o.d.items():
            d[k]=d.get(k,0)+v
            if abs(d[k])<1e-12:del d[k]
        return Lin(d,self.b+o.b)
    __radd__=__add__
    def __mul__(self,a):return Lin({k:v*a for k,v in self.d.items()},self.b*a)
    __rmul__=__mul__
    def __neg__(self):return self*(-1)
    def __sub__(self,o):return self+(-aslin(o))
    def __rsub__(self,o):return aslin(o)+(-self)
    def __truediv__(self,o):return self*(1./o)
def aslin(o):return o if isinstance(o,Lin) else Lin(b=o)
def unit(k):return Lin({k:1.})
exprs=[]; primitive=[]
def h(e):
    k=len(exprs);exprs.append(aslin(e));return unit(k)
def met():
    k=len(exprs);exprs.append(None);primitive.append(k);return unit(k)
def absval(e):return h(e)+h(-e)
def maximum(a,b):return b+h(a-b)
def minimum(a,b):return a-h(a-b)
def clip(e,lo,hi):return lo+h(e-lo)-h(e-hi)
def sat(e):return clip(e,0,1)
def gate(e,mid,width):return sat((e-mid)/width+.5)

def linrow(expr,idx):
    row=np.zeros(len(idx),'f')
    for j,a in expr.d.items():row[idx[j]]=a
    return row
# Central normalization by a piecewise-linear log. Ratios are represented by log differences;
# positive floors handle sky-dominated and nearly black apertures without diverging.
def ln(e,floor=0.002,top=3.):
    knots=[floor]
    while knots[-1]<top:knots.append(knots[-1]*1.55)
    s=0;result=math.log(floor)
    for a,b in zip(knots,knots[1:]):
        t=math.log(b/a)/(b-a);result=result+(t-s)*h(e-a);s=t
    return result

raw_metrics={}
def rawfeat(key,row):
    f=met();raw_metrics[next(iter(f.d))]=row;return f
peak=rawfeat('center',ref); peaklog=ln(peak)
# Polar coordinates after subtracting a sky and soft-capping very bright field objects.
# Use two hinges so clipping activates only above the estimated galaxy peak. Keep an
# uncapped branch for secondary-nucleus tests, since those peaks are themselves evidence.
ww=[]
for j in range(2):
    for i,r in enumerate(radii):
        for t in range(N):
            e=basis();e[j*npol+i*N+t]=1;e-=skybasis(j)
            # Keep signed colour via a luminance offset, removed after clipping.
            if j==1:
                l=basis();l[i*N+t]=1;l-=skybasis(0);e=.4*e+l
            ww.append(e)
ww=np.array(ww); caps=np.repeat([min(1.5,.85+4./r) for r in radii],N)
polarplus=relu('nonnegative_apertures',dense('sky_subtraction',phot,np.concatenate([ww,ww-np.tile(caps,2)[:,None]*ref]),np.r_[np.zeros(2*npol),np.full(2*npol,-.02)]))
ww2=np.concatenate([np.eye(2*npol),-np.eye(2*npol)],axis=1)
polar=dense('capped_photometry',polarplus,ww2)

# Azimuthal Fourier integrals. Integrate positive projections around a circle of phases
# to obtain amplitudes independent of position angle; no observed shape templates.
filters=[];pkeys={}; projections=[]
pp=2*npol
def recordkey(k,arr):
    q=met();idx=next(iter(q.d));pkeys[idx]=arr;return q

def ring(i,k=0,phase=0.,channel=0):
    a=np.zeros(pp);a[channel*npol+i*N:channel*npol+(i+1)*N]=np.cos(k*ph-phase)/N;return a

def components(a,b):
    ix=len(filters); n=8
    for t in range(n):filters.append(np.cos(2*np.pi*t/n)*a+np.sin(2*np.pi*t/n)*b)
    # Mean half-wave rectification is the amplitude divided by pi.
    q=met();projections.append((next(iter(q.d)),ix,n));return q

def amplitude(i,m):return components(ring(i,m),ring(i,m,np.pi/2))
D=[]; color=[]; A={m:[] for m in range(1,6)}; coloraxis=[]
for i in range(R):
    D.append(recordkey('mean',ring(i)))
    color.append(recordkey('red_minus_blue',2.5*(ring(i,channel=1)-ring(i))))
    for m in A:A[m].append(amplitude(i,m))
cart=lambda i,k:(ring(i,k),ring(i,k,np.pi/2))
# Phase twist of isophotes: two radial quadrupoles of the same long axis add coherently.
# Coarse- and fine-scale quadrupoles with misaligned phases are a bar / arms signature.
phasepair={}
for gap in (2,3,4):
    for i in range(R-gap):
        j=i+gap;fa,fb=cart(i,2);ga,gb=cart(j,2)
        phasepair[i,j]=components(.5*fa+ga,.5*fb+gb)
# Asymmetry (180-degree residual), with separately normalized radial bands.
AS=[]; lops=[]
for i in range(R):
    start=len(filters)
    for t in range(N):
        arr=np.zeros(pp);arr[i*N+t]=1.;arr[i*N+(t+N//2)%N]=-1.;filters.append(arr)
    q=met();projections.append((next(iter(q.d)),start,N));AS.append(q/math.pi*2) # mean |opposite - self|

polar_linear={};polar_linear.update(pkeys)
# Secondary peak contrast above an intervening saddle and above the nearby sky in other
# directions. Point responses go through an extended-source test on the image plane below.

def dense_dict(name,src,entries,columns):
    return dense(name,src,[entries.get(k,np.zeros(columns)) for k in primitive])

# True local maxima from a broad and a narrow Gaussian: x - max(neighbours) is positive
# only at a separated source; it is not large merely because a smooth profile slopes.
# Operate at four-pixel sampling on the field (small memory footprint, handles field stars).
coarse=pool('four_pixel_photometry',pixel,(2,2)); yy,xx=np.indices((5,5))-2
kern=[]
for sig in (.57,1.15):
    g=np.exp(-(xx**2+yy**2)/(2*sig**2));g/=g.sum();kern.append(lum[:,None,None]*g)
sm=conv('seeing_scales',coarse,kern,padding=2)
# Compare every pixel with neighbours in a circle with integer offsets. A local peak can
# lie between two pixels; use the lower of an opposite pair of contrasts only up to a
# small displacement tolerance (compare at more than one pixel distance).
# via per-direction translations on a cropped central field, projected by thin matrices.
# For compactness all offsets are sampled as 25x25 maps taken every 4 pixels.
peak_maps=[]
for s,sz,dist in [(0,7,2),(1,9,3)]:
    ker=[];cen=sz//2
    for k in range(8):
        dx=round(dist*np.cos(k*np.pi/4));dy=round(dist*np.sin(k*np.pi/4))
        a=np.zeros((2,sz,sz));a[s,cen+dy,cen+dx]=1.;a[s,cen,cen]-=1;ker.append(a)
    p=conv('surround_comparison_'+str(s),sm,ker);w=32-sz+1
    p=layer('directions_'+str(s),'reshape',p,shape=[1,8,w*w]);p=pool('strongest_neighbor_'+str(s),p,(8,1),'maxpool')
    p=layer('peak_field_'+str(s),'reshape',p,shape=[1,w,w]);p=relu('peak_prominence_'+str(s),conv('invert_contrast_'+str(s),p,[[[[-1.]]]]));peak_maps.append((p,w))
# Read out strongest secondary nucleus in annuli, preventing an unassociated bright object
# at the frame boundary from determining the classification. Fixed sparse selection maps.
peak_features=[]; peak_src=[]
for s,(p,w) in enumerate(peak_maps):
    Y,X=(np.indices((w,w))-(w-1)/2)*4;rr=np.hypot(Y,X)
    local=[]
    for lo,hi in [(0,6),(6,13),(13,23),(23,34)]:
        indices=np.flatnonzero((rr.ravel()>=lo)&(rr.ravel()<hi));local.append(indices)
    nn=max(map(len,local));wgt=np.zeros((4,nn,w*w),'f')
    for i,indices in enumerate(local):wgt[i,np.arange(len(indices)),indices]=1
    field=dense('annular_samples_'+str(s),p,wgt.reshape(4*nn,-1));field=layer('annulus_list_'+str(s),'reshape',field,shape=[4,1,nn]);field=pool('peak_by_annulus_'+str(s),field,(1,nn),'maxpool')
    fs=[met() for i in range(4)];peak_features.append(fs);peak_src.append((field,fs))

# Mean radial photometry, isophotal axis ratios from quadrupoles, and profile size.
# For S((x^2+y^2/q^2)^.5), a slightly flattened isophote produces a cos(2 phi)
# component with amplitude -(d S/d ln r)*ln(1/q)/2. The Fourier integral is half that.
# For strong flattening use the summed-light quadrupole: <cos 2phi>=(1-q)/(1+q)
# for luminosity with self-similar elliptical isophotes. Both estimates use smooth tapers.
lD=[ln(d+.003) for d in D]; rel=[z-peaklog for z in lD]

def bandmean(arr,ws):
    return sum(a*w for a,w in zip(arr,ws))/sum(ws)
def radw(s,low=0):return [r*r*math.exp(-.5*(r/s)**2)*(1-math.exp(-.5*(r/low)**2) if low else 1) for r in radii]
# Integrate r dr in approximately log-spaced rings. The taper's derivative changes
# Fourier ellipticity sensitivity. Calibrate analytically using the slope of the actual profile.
def ratio(x,y,floor=.001):return ln(x+floor)-ln(y+floor)
def ramp_ratio(x,y,lo,hi):return gate(ratio(x,y),math.log(math.sqrt(lo*hi)),math.log(hi/lo))
def posratio(x,y): # PWL approximation to a bounded positive ratio via its log
    z=ratio(x,y,.00035)
    knots=[.012,.032,.065,.11,.18,.27,.40,.58,.8,1.08,1.45,2.1,3.2];base=knots[0];s=0
    for a,b in zip(knots,knots[1:]):
        t=(b-a)/math.log(b/a);base=base+(t-s)*h(z-math.log(a));s=t
    return base

def ellip(scale,inner=False):
    ws=np.array(radw(scale));nrm=ws.sum();ws=ws/nrm
    if inner:ws=np.array([.08,.25,.35,.24,.08,0,0,0,0])
    # Adding vectors first, rather than their lengths, suppresses field objects and spiral arms.
    ax=np.zeros(pp);ay=np.zeros(pp)
    for i,w in enumerate(ws):
        u,v=cart(i,2);ax+=w*u;ay+=w*v
    amp=components(ax,ay)
    slope=0
    for i,w in enumerate(ws):
        lo=max(0,i-1);hi=min(R-1,i+1)
        slope+=w*(D[lo]-D[hi])/math.log(radii[hi]/radii[lo])
    # e here approximates ln(major/minor); at high flattening it is an effective value.
    return 4*posratio(amp,maximum(slope,.0008)),amp
inner,ia=ellip(7,True); mid,ma=ellip(10);outer,oa=ellip(17);large,la=ellip(26)
# Less slope-sensitive, coherent, flux-weighted angular flattening for highly inclined disks.
w=radw(21);total=bandmean(D,w);qfield=[]
for m in [2,4]:
    ax,ay=zip(*(cart(i,m) for i in range(R))); amp=components(np.average(ax,axis=0,weights=w),np.average(ay,axis=0,weights=w));qfield.append(posratio(amp,total))

def clr(ws):
    l=bandmean(D,ws);c=bandmean(color,ws);return (ln(l+.5*c+.006)-ln(l-.5*c+.006))
corecolor=clr([.4,.4,.2,0,0,0,0,0,0]);diskcolor=clr(radw(14,6));farcolor=clr(radw(22,9))
# A half-light-like scale: integrate the flux per radial annulus, using a Gaussian
# window to reject background sources. Estimate concentration with a surface brightness ratio.
size=ratio(bandmean(D,radw(15)),peak); extent=ratio(bandmean(D,radw(25)),peak)
# In a round ellipse, harmonic amplitudes of order >=3 are small. Colour gradients alone are
# not enough: use lopsidedness, fine arm contrast, and isophote twists too.
Wdisk=radw(15,6);Wout=radw(22,10);dq=bandmean(D,Wdisk);fq=bandmean(D,Wout)
asym=posratio(bandmean(AS,Wdisk),dq+.018*peak+.003);farasym=posratio(bandmean(AS,Wout),fq+.03*peak+.004)
odds=posratio(bandmean([a+.65*b for a,b in zip(A[3],A[5])],Wdisk),dq+.018*peak+.004)
# Inner asymmetry helps separate a paired nucleus from star-forming knots in the outskirts.
wi=[.05,.15,.27,.28,.17,.08,0,0,0];coreasym=posratio(bandmean(AS,wi),bandmean(D,wi)+.003)
# Contrast of opposite-sided spiral arms which wind relative to the inner disk or bar.
twist=[]
for gap in (2,3,4):
    w=np.array(radw(16,5))[gap:];nums=[];dens=[]
    for i in range(R-gap):
        j=i+gap;a=.5*A[2][i];b=A[2][j]; nums.append(a+b-phasepair[i,j]);dens.append(D[j]+.2*D[i])
    twist.append(posratio(bandmean(nums,w),bandmean(dens,w)+.003))
turn=(twist[0]+2*twist[1]+twist[2])/4
# Check for inner quadrupole that is diluted in the outer disk, characteristic of a bar.
barcontrast=inner-outer
# Quadrupole localized in mid-radii can be missed by cumulative shape moments. The
# normalization accounts for the local radial brightness gradient of a non-barred disk.
barbands=[]
for i in (2,3,4,5):
    ws=np.zeros(R);ws[i-1:i+2]=[.22,.56,.22];w2=ws
    amp=bandmean(A[2],ws);sl=sum(ws[j]*(D[j-1]-D[j+1])/math.log(radii[j+1]/radii[j-1]) for j in (i-1,i,i+1))
    barbands.append(4*posratio(amp,maximum(sl,.05*bandmean(D,ws)+.015*peak)))
localized=maximum(maximum(barbands[0],barbands[1]),maximum(barbands[2],barbands[3]))
# Interacting pairs: broad-scale off-centre peak prominence relative to the central
# surface brightness. Penalize the farthest candidates more for chance projections.
fine,broad=peak_features
nearcomp=maximum(broad[1],maximum(.86*broad[2],.52*broad[3]));sharpcomp=maximum(fine[1],maximum(.86*fine[2],.52*fine[3]))
companion=posratio(nearcomp,peak+.035);sharp=posratio(sharpcomp,peak+.035)
centercontrast=posratio(fine[0],peak+.025)

# Opposite-side common light. A spatially isolated foreground star need not invalidate
# the position angle or shape of an otherwise symmetric main galaxy.
I=np.eye(npol);flip=I.reshape(R,N,npol)[:,(np.arange(N)+N//2)%N,:].reshape(npol,npol)
zeros=np.zeros_like(I);lumonly=np.c_[I,zeros]
delta=relu('one_sided_excess',dense('antipodal_contrast',polar,np.c_[I-flip,zeros]))
commons=layer('common_light','add',[dense('brightness',polar,lumonly),dense('remove_unpaired_light',delta,-I)])
# Measure isophotes along many diameters. Two crossing radii per fractional isophote yield
# log semi-axis lengths by integrating the indicator I(r) > f*I(0) in d(log r).
# Smooth the crossing over a fraction of a magnitude to suppress noise, and use several
# isophotes so a central bulge can be separated from a thin, extended disk.
Wlog=np.diff(np.r_[np.log(radii[0])-.5*np.log(radii[1]/radii[0]),.5*(np.log(radii[:-1])+np.log(radii[1:])),np.log(radii[-1])+.5*np.log(radii[-1]/radii[-2])]);rmin=np.log(radii[0])-.5*np.log(radii[1]/radii[0])
frac=[.75,.48,.28,.15];nang=N//2;dire=[];ids=[]
thresholds=dense('fractional_isophotes',phot,np.stack([ref*f for f in frac]),np.array([.002,.003,.004,.005]))
for k,f in enumerate(frac):
    for a in range(nang):
        for i in range(R):dire.append(i*N+a);ids.append(k)
dire=np.array(dire);ids=np.array(ids);ns=len(dire);oneidx=np.arange(ns)
# Include some unpaired light for robustness to an imperfect catalogue centre.
wgt=np.zeros((ns,npol),'f');wgt[oneidx,dire]=.68
side=np.zeros((ns,pp),'f');side[oneidx,dire]=.16;side[oneidx,dire+N//2]=.16
thr=np.zeros((ns,len(frac)));thr[oneidx,ids]=-1
z=layer('crossing_margins','add',[dense('common_ray',commons,wgt),dense('both_rays',polar,side),dense('isophotal_level',thresholds,thr)])
# A PWL log of actual intensities would permit still more accurate relative thresholds.
# Here soft steps have fixed noise width and are bounded in low S/N regions.
width=np.repeat([.065,.05,.037,.027],nang*R);im=np.eye(ns)
z=relu('soft_crossing_hinges',dense('soft_crossing',z,np.r_[im,im],np.r_[width/2,-width/2]));
Wbins=np.zeros((len(frac)*nang,ns)); Wbins[np.arange(ns)//R,np.arange(ns)]=np.tile(Wlog,len(frac)*nang)/width
lengths=dense('log_isophote_radii',z,np.c_[Wbins,-Wbins],rmin)
# Use an orthogonal axis at the SAME phase as the major axis. This excludes the
# spurious apparent flattening that comparing an arbitrary bright and dark pixel creates.
proj=np.zeros((2,len(frac),nang,len(frac)*nang));O=np.eye(nang);O90=np.roll(O,nang//2,axis=0)
for k in range(len(frac)):
    proj[0,k,:,k*nang:(k+1)*nang]=O-O90;proj[1,k,:,k*nang:(k+1)*nang]=O
ml=dense('diametric_axes',lengths,proj.reshape(-1,len(frac)*nang));ml=layer('axis_candidates','reshape',ml,shape=[2*len(frac),1,nang]);ml=pool('major_axes',ml,(1,nang),'maxpool')
majorfeat=[met() for _ in range(2*len(frac))];peak_src.append((ml,majorfeat));iso=majorfeat[:len(frac)];maj=majorfeat[len(frac):]

# Angular signal in excess of a smooth ellipse. For an ellipse the annular m=2
# amplitude is proportional to the log radial brightness slope, at the SAME phase at
# adjacent radii. A complex Fourier difference removes this common oval response;
# compare the error for no rotation with both signs of the log-spiral phase advance.
# This is scale- and handedness-invariant, and does not require stored spiral pictures.
track={};phase_rates=[0,1/math.tan(math.radians(36)),1/math.tan(math.radians(18))];pairs=range(2,7)
for m in (2,3):
    for k,beta in enumerate(phase_rates):
        for sign in ((1,) if k==0 else (-1,1)):
            nums=[];dens=[]
            for i in pairs:
                j=i+1; a,b=cart(i,m);c,d=cart(j,m)
                # Compare the outer harmonic to the inner one propagated to the next
                # shell. Geometric attenuation is common in a disk brightness profile.
                f=.67;angle=sign*m*beta*math.log(radii[j]/radii[i]);co=np.cos(angle);si=np.sin(angle)
                nums.append(components(c-f*(co*a-si*b),d-f*(si*a+co*b)));dens.append(D[j]+f*D[i]);
            win=np.array(radw(17,7))[2:7];track[m,k,sign]=posratio(bandmean(nums,win),bandmean(dens,win)+.03*peak+.004)
rot=[]
for k in (1,2):
    costs=[.80*track[2,k,sign]+.20*track[3,k,sign] for sign in (-1,1)];rot.append(minimum(*costs))
fixed=.80*track[2,0,1]+.20*track[3,0,1];winding=clip(fixed-minimum(*rot),-.12,.12)
# Smooth galaxies fade monotonically away from the centre. An outer positive reversal,
# without an equivalent on the opposite side, is more diagnostic of a tidal tail than
# an ordinary shift of the photocentre. A symmetric reversal is evidence of an arm/ring.
rev_sides=[]
for i in range(2,R-1):
    rev_sides.append(I[(i+1)*N:(i+2)*N]-.87*I[i*N:(i+1)*N])
rev=np.concatenate(rev_sides);nr=len(rev);ri=np.eye(nr);rf=ri.reshape(-1,N,nr)[:,(np.arange(N)+N//2)%N,:].reshape(nr,nr)
z=relu('outward_rebrightening',dense('radial_differences',polar,np.c_[rev,np.zeros_like(rev)],-.003));z2=relu('unpaired_rebrightening',dense('reversal_symmetry',z,ri-rf))
ws=np.repeat(radw(18,7)[3:],N);ws=ws/ws.sum();ar=met();br=met();peak_src.extend([(dense('all_reversals',z,[ws]),[ar]),(dense('unpaired_reversals',z2,[ws]),[br])]);rden=bandmean(D[3:],radw(18,7)[3:])+.025*peak+.004
revnorm=posratio(ar,rden);tidaltail=posratio(br,rden)

# Smooth-ellipse null model. On log-radius coordinates, all rays through an elliptical
# spheroid have the same radial profile translated by log of their radius factor. The
# centering here is geometric (no training or fitted population parameters). A constant
# flattening or a change of axis ratio alone is weaker evidence than twisted isophotes.
# For each angular aperture compare the ridge at different surface brightnesses.
JJ=np.eye(nang)-np.ones((nang,nang))/nang;profile_masks=[]
for iaa,ibb in [(0,2),(1,3),(2,3)]:
    w=np.zeros((nang,4,nang));w[:,iaa,:]=JJ;w[:,ibb,:]=-JJ;profile_masks.extend(w.reshape(nang,-1))
xp=np.array(profile_masks);zp=relu('isophote_warp_absolute',dense('differential_isophotes',lengths,np.r_[xp,-xp]))
wc=np.tile(np.repeat(np.eye(3),nang,axis=1)/nang,2);mprof=[met() for _ in range(3)];peak_src.append((dense('profile_incoherence',zp,wc),mprof))
warp=.40*mprof[0]+.45*mprof[1]+.15*mprof[2]
# Outer profile and colour of the *same galaxy*, favoring antipodal light. Strong
# sources present on only one side don't masquerade as a blue disk around a smooth core.
colorcom=dense('colour_surface_brightness',polar,np.c_[zeros,I]);cdelta=relu('colour_one_side',dense('colour_asymmetry',colorcom,I-flip))
cm=layer('paired_colour','add',[colorcom,dense('subtract_colour_asymmetry',cdelta,-I)])
NEW=[]
for w0 in (radw(14,6),radw(22,10)):
    wv=np.repeat(w0,N);wv=wv/wv.sum();lumobs,colobs=met(),met()
    peak_src.extend([(dense('disk_paired_light_'+str(len(NEW)),commons,[wv]),[lumobs]),(dense('disk_paired_colour_'+str(len(NEW)),cm,[wv]),[colobs])]);cv=2.5*(colobs-lumobs)
    NEW.append(ln(lumobs+.5*cv+.004)-ln(lumobs-.5*cv+.004))
diskcolor=.75*NEW[0]+.25*diskcolor;farcolor=.8*NEW[1]+.2*farcolor
# A bulge protrudes from a thin disk: at substantial distances along the major axis
# the ridge is narrower than an extension of the central light. Conversely a disk of
# nearly uniform vertical thickness tends to have a flat-topped longitudinal profile.
# Measure major-axis concentration relative to an exponential profile. At two relative
# brightness levels f,g the radius ratio is log(1/f)/log(1/g), independent of scale.
long_conc=maj[3]-maj[1]-math.log(math.log(.15)/math.log(.48))
# Isolate low-surface-brightness one-sided streams. Track a positive radial excess
# against BOTH angular flanks, far outside the inner isophotes. An off-centre nucleus
# gives a compact ridge; long diffuse emission persists to the adjacent log-radius.
startr=3;pa=[];flanks=[]
for i in range(startr,R):
    for t in range(N):
        g=np.zeros(npol);g[i*N+np.array([(t-1)%N,t,(t+1)%N])]=[.25,.5,.25];pa.append(g)
        fl=np.zeros(npol);fl[i*N+np.array([(t-5)%N,(t-3)%N,(t+3)%N,(t+5)%N])]=.25;flanks.append(fl)
pa=np.array(pa);flanks=np.array(flanks);opp=pa.reshape(-1,N,npol)[:,(np.arange(N)+N//2)%N,:].reshape(-1,npol);pr=lambda A:np.c_[A,np.zeros_like(A)]
stream1=dense('flank_prominence',polar,pr(pa-flanks));stream2=relu('reject_equal_ridges',dense('opposing_ridge',polar,pr(opp-flanks)))
sz=len(pa);stream=relu('one_sided_stream',layer('stream_contrast','add',[stream1,dense('competing_ridge',stream2,-np.eye(sz))]));di=np.eye(sz)
# Average angularly, take the minimum along successive radial segments; no requirement
# that a tail is perfectly straight, as each probe is already angularly smoothed.
fwd=di[N:];back=di[:-N];sz2=len(fwd);streammin=layer('continuous_stream','add',[dense('outer_segment',stream,fwd),dense('radial_discontinuity',relu('break_hinge',dense('stream_break',stream,fwd-back)),-np.eye(sz2))]);
wtail=np.repeat(radw(23,12)[startr+1:],N);wtail=wtail/wtail.sum();stail=met();peak_src.append((dense('tail_flux',streammin,[wtail]),[stail]));tailstrength=posratio(stail,bandmean(D[startr+1:],radw(23,12)[startr+1:])+.015*peak+.003)

# Extended secondary nuclei: a multi-scale, many-direction second-difference operator.
# At a compact nucleus the curvature is negative along both axes. At a point on the
# flank of a single nucleus the radial curvature has the opposite sign. This test can
# resolve pairs closer than the first-maximum test and is invariant to a sloping pedestal.
fineim=conv('fine_galaxy_luminance',pixel,lum[None,:,None,None]);G=[];Yg,Xg=np.indices((15,15))-7
for width in (1.05,1.85,3.1):
    g=np.exp(-(Yg**2+Xg**2)/(2*width**2));g/=g.sum();G.append(g)
G=np.array(G);smallmaps=conv('nuclear_gaussians',fineim,G[:,None,:,:],stride=2,padding=7)
K=[]
for an in np.arange(4)*np.pi/4:
    dy,dx=np.rint([1.6*np.sin(an),1.6*np.cos(an)]).astype(int)
    a=np.zeros((3,5,5));a[0,2,2]=-1;a[0,2+dy,2+dx]+=.5;a[0,2-dy,2-dx]+=.5;K.append(a)
# Combine curvature with negative PSF evidence before a maximum is taken; broad blur
# dilutes an unresolved star much more than an extended galaxy. The soft cut is local.
z=conv('nuclear_curvatures',smallmaps,K);wd=28;z=layer('curvature_directions','reshape',z,shape=[1,4,wd*wd]);z=pool('weakest_curvature',z,(4,1),'maxpool');z=layer('curvature_plane','reshape',z,shape=[1,wd,wd])
K=np.zeros((2,3,5,5));K[0,:,2,2]=[1,-1.8,0];K[1,:,2,2]=[-.05,1,-1.5];p=relu('stellar_penalties',conv('stellar_psf_veto',smallmaps,K,[.008,.018]));z=relu('resolved_nuclei',layer('resolved_peak_height','add',[dense('negative_curvature',z,-np.eye(wd*wd)),dense('stellar_veto',p,np.c_[-1.4*np.eye(wd*wd),-.45*np.eye(wd*wd)])]));
# Curvature alone also fires on shoulders of the primary nucleus. Require a drop
# towards the primary for evidence of TWO nuclei. Measure the supporting pedestal
# midway between the candidates, and reject a detached star on black sky.
Yp,Xp=(np.indices((wd,wd))-(wd-1)/2)*4-1;rp=np.hypot(Yp,Xp)
MW=np.zeros((wd*wd,3,32,32));BW=np.zeros_like(MW);Cbase=np.zeros_like(MW)
for y0 in range(wd):
    for x0 in range(wd):
        j=y0*wd+x0;rr=rp[y0,x0];uy=Yp[y0,x0]/max(rr,.5);ux=Xp[y0,x0]/max(rr,.5)
        # Candidate minus a sample on the approach to the primary. A monotone
        # ellipse has nonpositive margin, regardless of its axis ratio or tilt.
        Cbase[j,0,y0+2,x0+2]=1; distance=min(10,.54*rr)/4
        def sample(arr,ch,yy,xx,val):
            yy=np.clip(yy,0,30.99999);xx=np.clip(xx,0,30.99999);i=int(yy);k=int(xx);v=yy-i;u=xx-k
            arr[j,ch,i:i+2,k:k+2]+=val*np.array([[(1-v)*(1-u),(1-v)*u],[v*(1-u),v*u]])
        sample(MW,0,y0+2-distance*uy,x0+2-distance*ux,-.98)
        sample(BW,2,y0+2-.5*rr/4*uy,x0+2-.5*rr/4*ux,1)
MW+=Cbase;marg=dense('inward_saddle',smallmaps,MW);base=relu('above_sky_bridge',dense('diffuse_bridge',smallmaps,BW,-.09));Id=np.eye(wd*wd)
# min(z, margin, bridge) via the identity min(a,b) = a - ReLU(a-b).
for arg,tag,fac in ((marg,'saddle',1.45),(base,'bridge',1.7)):
    z=layer('supported_by_'+tag,'add',[z,dense('remove_'+tag,relu('lack_'+tag,layer('missing_'+tag,'add',[z,dense('negative_'+tag,arg,-fac*Id)])),-Id)])
z=relu('paired_nucleus_candidates',z)
lists=[np.flatnonzero((rp>=lo)&(rp<hi)) for lo,hi in ((0,6),(6,13),(13,22),(22,33),(33,45))];nw=max(map(len,lists));ks=np.zeros((len(lists),nw,wd*wd))
for j,ix in enumerate(lists): ks[j,np.arange(len(ix)),ix]=1
p=dense('extended_peak_apertures',z,ks.reshape(-1,wd*wd));p=pool('extended_nucleus_maxima',layer('candidate_index','reshape',p,shape=[len(lists),1,nw]),(1,nw),'maxpool');nucs=[met() for _ in lists];peak_src.append((p,nucs))
double_nucleus=maximum(maximum(nucs[1],.83*nucs[2]),maximum(.56*nucs[3],.32*nucs[4]));resolved=posratio(double_nucleus,peak+.035)

# Count correlated light at reflected positions, discounted when the ray toward the
# centre passes through an actual gap. This distinguishes satellite/field objects from
# the asymmetric ENVELOPE of a centrally connected galaxy. Min-plus ray integration
# is a morphological opening, not a memorized shape mask. Use the radial excess of
# this opened image relative to its antipodal partner (a 180-degree rotation).
rows=np.zeros((R,1,N,npol));
for i in range(R):rows[i,0]=I[i*N:(i+1)*N]
opened=[dense('inner_connected_aperture',polar,pr(rows[0,0]))];eyeA=np.eye(N)
for i in range(1,R):
    bright=dense('ray_brightness_'+str(i),polar,pr(rows[i,0]));dif=layer('ray_gap_'+str(i),'add',[bright,dense('ray_allowance_'+str(i),opened[-1],-1.27*eyeA,-.008)]);
    opened.append(layer('connected_aperture_'+str(i),'add',[bright,dense('exclude_ray_gap_'+str(i),relu('positive_gap_'+str(i),dif),-eyeA)]))
win=radw(21,9);wa=np.array(win)/sum(win);oas=[];ols=[];os2=[]
for i in range(2,R):
    ir=eyeA-np.roll(eyeA,N//2,axis=0)
    u=relu('connected_rotational_residual_'+str(i),dense('connected_rotation_'+str(i),opened[i],ir));oas.append(dense('residual_flux_'+str(i),u,[np.ones(N)*2*wa[i]/N]));ols.append(dense('connected_flux_'+str(i),opened[i],[np.ones(N)*wa[i]/N]));
    # A one-sided extended envelope shows the same direction of excess over a
    # range of scales; winding spiral arms frequently change direction.
    os2.append(dense('dipole_'+str(i),opened[i],np.stack([np.cos(ph-a) for a in ph[::3]])*wa[i]/N))
oc=met();of=met();od=met();dip=relu('positive_dipole',layer('coherent_dipole','add',os2));peak_src.extend([(layer('connected_asymmetry','add',oas),[oc]),(layer('connected_disk_flux','add',ols),[of]),(dense('dipole_intensity',dip,[np.full(8,np.pi/8)]),[od])]);con_asym=posratio(oc,of+.018*peak+.001);con_dip=posratio(od,of+.018*peak+.001)

# Fuzzy morphological decision tree. Its 'votes' express necessary or supporting tests,
# not fitted likelihoods. Class boundaries are chosen in physical quantities like
# aspect, relative arm contrast, nucleus prominence and colour differences.
iso_inner=.5*(iso[0]+iso[1]);iso_outer=.45*iso[2]+.55*iso[3]
flatten=.30*mid+.20*outer+.12*inner+.38*(iso[1]+iso[2])/2
axis=.12*outer+.10*large+.25*iso[2]+.53*iso[3]
# A thin disk has a projected aspect exceeding about three and grows long in
# the major direction. A cigar-shaped spheroid is elongated but relatively compact.
edge=2.35*(axis-math.log(2.8))+1.35*(qfield[0]-.37)+.85*clip(maj[3]-math.log(17),-.75,.85)-.6*h(coreasym-.32)
# Less extended or purely elliptical bodies don't acquire a disk vote on aspect alone.
edge+=.4*clip(iso[3]-iso[1],-.7,.7)
# Multiple peaks having both a saddle and surrounding diffuse light are unlike arm knots.
merger=5.0*clip(companion,0,.65)+1.6*clip(sharp,0,.65)+1.2*clip(asym-.45,-.45,.75)+1.0*h(coreasym-.18)-.94
merger+=.7*h(diskcolor-.06)-.55*h(-.16-diskcolor)
# A chance field star can have a prominent peak but little extended shared light.
# Also require some independent asymmetry of the galaxy itself.
merger=minimum(merger,.85*(asym-.45)+1.2*(coreasym-.17)+2.2*(tidaltail-.065)+.65*(companion-.23))
# Structure vs smooth light. Smooth spheroids usually get redder, not bluer, as the
# stellar population ages. Arms generally add an extended blue disk and angular texture.
blue=clip(-diskcolor,-.55,.65);gradient=clip(corecolor-diskcolor,-.45,.65)
extended=clip(size+1.65,-1.3,1.35)
structured=3.25*blue+1.30*gradient+1.0*extended+2.6*clip(odds-.075,-.075,.28)+3.5*clip(turn-.018,-.018,.14)+.4*clip(asym-.22,-.22,.45)-.16
# Bar cue: not just a high m=2 signal, but ellipticity or position angle varying with
# radius. Give a round outer disk with a strong, long inner ridge the most support.
bar=1.00*clip(barcontrast,-1,1.3)+.88*clip(localized-outer,-.9,1.4)+3.1*clip(turn,0,.15)+.35*clip(inner-.4,-.4,.7)-.63
# Extra morphological votes from the radial and pitch measurements. Strong winding
# or a long bar in a more nearly round outer disk is positive evidence even for a
# reddish galaxy; extended monotonic light alone is not a spiral arm.
disk_major=clip(maj[3]-math.log(16),-1.,1.3)
bar_excess=.55*(iso[1]-iso[3])+.45*(iso[2]-iso[3]);bar+=.95*clip(bar_excess,-.7,.8)+.22*disk_major
structured+=.8*disk_major+.8*h(bar)+7.0*h(winding)+1.55*clip(revnorm-.10,-.10,.28)
# Positive pitch contrast favors the open-arm propagation rate.
pitch=rot[1]-rot[0]
# Quantify bulge strength with longitudinal concentration as well as central colour.
# A round PSF rounds off the centre of every thin disk, which should not alone
# count as a bulge; use red excess and concentration above a single exponential disk.
bulge=2.45*corecolor+.85*(centercontrast-.17)+.95*clip(long_conc,-.65,.65)+.22*clip(iso_outer-iso_inner,-1,1)-.30
edge+=.55*clip(iso[2]-.9,-.55,.8)+.65*clip(iso[3]-1,-.6,.9)+.40*(maj[2]-math.log(11))
# Profile mismatch in several isophotes is evidence against an ellipse. Blue but very
# small, nearly homologous spheroids can still look smooth. Conversely a symmetric
# bar with two arms can have very little odd-harmonic signal.
structured+=2.40*(warp-.13)+.7*h(bar)+.75*clip(maj[2]-math.log(12),-.8,.9)+1.45*h(revnorm-.10)
# Concentration indices measured along the major diameter are much less dependent on
# inclination than the mean intensity in a fixed circular aperture. Low concentration
# and a blue stellar population favor open arms. Bright, well resolved, coherent arms
# are evidence of winding structure even in the relatively late spirals.
loose=2.25*blue-1.15*corecolor-.94*clip(long_conc,-.8,.8)-.7*clip(centercontrast-.08,-.08,.35)+6.5*clip(pitch,-.14,.14)+1.7*(odds-.095)+.55*clip(asym-.35,-.35,.7)+.75*clip(.45-peak,-.65,.4)-.23
# A tail should persist over a range of radii. Do not label a symmetric high-contrast
# spiral or a distant bright star as a disturbed galaxy just for being asymmetric at
# the very outside of the aperture.
disturb=2.5*clip(tailstrength-.15,-.15,.45)+.6*clip(asym-.45,-.45,.8)+.7*clip(coreasym-.23,-.23,.7)+1.55*clip(tidaltail-.085,-.085,.28)-1.7*clip(resolved,0,.6)-.18
# A second resolved nucleus on the main galaxy is evidence of a merger even when the
# arms are faint. A sharp isolated field star with low surrounding curvature is not.
merger=maximum(merger,4.5*clip(resolved-.135,-.135,.55)+.95*clip(asym-.37,-.37,.8)+.5*(coreasym-.15)+.85*h(diskcolor-.05)-.35)
merger+=.65*clip(resolved-.09,-.09,.3)
# Connected dipole and antipodal excess probe the outer envelope. In a dipolar disk
# I(r,phi) = I0(r)(1+a cos(phi)), these estimators are a/2 and 4a/pi.
# Require evidence for a substantial persistent perturbation, not merely clumps.
disturb=maximum(disturb,2.35*(con_dip-.17)+1.15*(con_asym-.42)+.65*clip(tailstrength-.12,-.12,.35)+.55*clip(coreasym-.18,-.18,.6)-2.65*resolved)
# If the inner and outer isophotes both remain approximately elliptical, the galaxy
# should not be called a merger on the basis of a lone projected source. Resolved
# pairs near the nucleus bypass this veto through their saddle contrast.
merger-=.55*h(.32-con_asym)+.35*h(.20-coreasym)
# A compact centrally concentrated disk can have tight arms even when its disk
# is blue. Decouple diffuse star formation from a blue point-like bulge by comparing
# the flux on the innermost ray with an extrapolation of the outer exponential.
loose+=.55*clip(maj[0]-.5*(maj[1]+maj[2])+.65,-1.,1.)+.65*clip(.38-peak,-.6,.32)
smooth_flat=.65*flatten+.35*(.40*iso[1]+.60*iso[2]);cigar_boundary=math.log(2.05)
# Common score per morphological family; subclass competition does not increase the
# parent score (subtract the maximum), so adding a third subtype cannot bias the tree.
smooth2=.24-smooth_flat;smooth3=minimum(smooth_flat-.24,cigar_boundary-smooth_flat);smooth4=smooth_flat-cigar_boundary
smmax=maximum(smooth2,maximum(smooth3,smooth4));sm=-h(structured)*.95
sp=-h(-structured)*.95
sbar=bar;stight=minimum(-bar,-loose);sloose=minimum(-bar,loose);spmax=maximum(sbar,maximum(stight,sloose))
edgebase=edge; smoothpen=.4*h(edge+.1)
scores=[disturb,merger,sm+smooth2-smmax-smoothpen,sm+smooth3-smmax-smoothpen,sm+smooth4-smmax-smoothpen-.1,sp+sbar-spmax,sp+stight-spmax,sp+sloose-spmax,edgebase-h(bulge),edgebase-h(-bulge)]

# Emit feature branches only once all the primitive feature names have been assigned.
base1=dense_dict('nucleus_measurements',phot,raw_metrics,nraw)
base2=dense_dict('radial_profiles',polar,polar_linear,pp)
angular=relu('positive_harmonic_phases',dense('angular_filters',polar,filters));pridx={k:i for i,k in enumerate(primitive)}
mat=np.zeros((len(primitive),len(filters)),'f')
for k,idx,n in projections:mat[pridx[k],idx:idx+n]=math.pi/n
base3=dense('angular_invariants',angular,mat); bases=[base1,base2,base3]
for i,(src,fs) in enumerate(peak_src):
    W=np.zeros((len(primitive),len(fs)));
    for j,q in enumerate(fs):W[pridx[next(iter(q.d))],j]=1
    bases.append(dense('peak_features_'+str(i),src,W))
base=layer('measurements','add',bases)
# Schedule the hinge graph in levels. A skip edge only carries nodes actually read
# at that level. Splitting matrices by source avoids repeated identity-copy layers.
needed=set()
def visit(k):
    if k in needed:return
    needed.add(k)
    if exprs[k] is not None:
        for j in exprs[k].d:visit(j)
for s in scores:
    for j in s.d:visit(j)
level={k:0 for k in primitive}; group={0:primitive};srcs={0:base}
for k,e in enumerate(exprs):
    if e is not None and k in needed:
        t=1+max((level[j] for j in e.d),default=0);level[k]=t;group.setdefault(t,[]).append(k)
indices={t:{k:i for i,k in enumerate(ks)} for t,ks in group.items()}
def stage(name,expr):
    deps=sorted(set(level[j] for e in expr for j in e.d)); ss=[]
    for z,t in enumerate(deps):
        W=np.zeros((len(expr),len(group[t])),'f');b=np.array([e.b for e in expr]) if z==0 else 0
        for i,e in enumerate(expr):
            for j,val in e.d.items():
                if level[j]==t:W[i,indices[t][j]]=val
        ss.append(dense(name+'_from_'+str(t),srcs[t],W,b))
    return layer(name,'add',ss) if len(ss)>1 else ss[0]
for t in sorted(group):
    if t:srcs[t]=relu('hinges_'+str(t),stage('rule_stage_'+str(t),[exprs[k] for k in group[t]]))
stage('class_scores',list(map(aslin,scores)))
(out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n')
np.savez(out/'weights.npz',**weights)
