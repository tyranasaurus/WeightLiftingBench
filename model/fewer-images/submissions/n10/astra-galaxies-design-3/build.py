"""Analytic galaxy morphometry. This program takes no input other than the output path."""
import json, sys, math
from pathlib import Path
import numpy as np

out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
layers=[]; weights={}; F=np.float32

def layer(name,op,ins,**kw):
    layers.append(dict(name=name,op=op,inputs=ins if isinstance(ins,list) else [ins],**kw)); return name

def affine(name,src,w,b=0,cv=False):
    w=np.asarray(w,dtype=F); n=len(w)
    weights[name+'.W']=w
    weights[name+'.b']=np.broadcast_to(np.asarray(b,dtype=F),(n,)).copy()
    if cv: return layer(name,'conv',src,out=n,kernel=list(w.shape[-2:]),stride=1,padding=0)
    if w.size>80000000: raise ValueError('unreasonably large stage')
    return layer(name,'dense',src,out=n)

def reshape(name,x,shape): return layer(name,'reshape',x,shape=shape)
def pool(name,x,size,stride,mode):return layer(name,mode+'pool',x,size=size,stride=stride)
def relu(name,x): return layer(name,'relu',x)

# Work mostly in 2-pixel bins. Fixed luminance and two thresholds: sky subtraction,
# and an anticoincidence floor so colour measurements ignore the empty sky.
a=pool('bin2','image',[2,2],[2,2],'avg')
c=np.array([[.34,.42,.24],[1,0,0],[0,0,1]])
band=affine('bands',a,c[:,:,None,None],[-.14,-.12,-.12],True)
light=relu('above_sky',band)
y,x=(np.indices((64,64)) - 31.5)*2
r=np.hypot(x,y); ang=np.arctan2(y,x)

# Morphological reconstruction: keep flux only if connected to the central
# seed on superlevel sets. Max-pool propagates reachability and a ReLU
# implements min(proposed, available). This rejects stars and bridges only
# at the brightness of their saddle; the add restores the original peak.
# A compact luminance-only branch keeps the gates cheap.
yband=affine('luminance',light,np.array([[[[1]],[[0]],[[0]]]]),cv=True)
Y=reshape('luminance_plane',yband,[1,64,64])
# Modest preblurring suppresses noise paths. A 3x3 dilation preserves a thin
# nearly edge-on disk even when it passes diagonally through pixel bins.
def padded(name,src,K,pad):
    val=affine(name,src,np.array(K,F)[None,None],cv=True);layers[-1]['padding']=pad;return val
Y=padded('denoise',Y,np.outer([.2,.6,.2],[.2,.6,.2]),1); filtered=Y
# Seed by subtracting a spatial bias mask (obtained from a constant dense bias).
# There are no coordinate or intensity parameters extracted from examples.
seedv=affine('seed_mask',pool('scalar_seed',Y,[64,64],[64,64],'avg'),np.zeros((4096,1)),np.where(r.ravel()<5,0,-2))
seedv=reshape('seed_plane',seedv,[1,64,64]);C=relu('origin',layer('seed_origin','add',[Y,seedv]))
def component(src,available,sequence,st):
    for idx,half in enumerate(sequence):
        nm=st+str(idx);pad=padded('extent_'+nm,src,[[1]],half);dil=pool('spread_'+nm,pad,[2*half+1]*2,[1,1],'max')
        # min(Y,dil) = Y-relu(Y-dil)
        neg=affine('negative_spread_'+nm,dil,np.array([[[[-1]]]]),cv=True)
        gap=relu('unconnected_'+nm,layer('difference_'+nm,'add',[available,neg]));ng=affine('subtract_gap_'+nm,gap,np.array([[[[-1]]]]),cv=True);src=layer('component_'+nm,'add',[available,ng])
    return src
C=component(C,Y,[1,1,1,2,2,2,3,3,3],'primary')

# Use integral distances for a multiscale second-nucleus detector. Split the
# 2-D maximum into a 1-D max over sites *after* evaluating the gates. The
# following operations can be evaluated together at all sites by convolution.
def plus(name,*args):return layer(name,'add',list(args))
def neg(name,src):return affine(name,src,np.array([[[[-1]]]]),cv=True)
def dmin(name,X,Z):
    over=relu(name+'_relu',plus(name+'_dif',X,neg(name+'_neg',Z)));return plus(name,X,neg(name+'_sub',over))
def blurk(name,src,scale):
    # Approximate the heat kernel by a short walk with matched variance.
    # This avoids huge intermediate im2col tensors at evaluation time.
    n=max(1,math.ceil((scale/2)**2/1.25));v=(scale/2)**2/n
    if v<=.8:k=np.array([v/2,1-v,v/2]);pad=1
    else:
        h=(v-.5)/8;k=np.array([h,.25,.5-2*h,.25,h]);pad=2
    for j in range(n):src=padded(name+'_'+str(j),src,np.outer(k,k),pad)
    return src
# The output grid has 64x64 sites; max is evaluated in a Gaussian sky aperture.
def spatialmax(name,X):
    z=reshape(name+'_sites',X,[1,64,64]);z=pool(name+'_tiles',z,[8,8],[8,8],'max');return pool(name,z,[8,8],[8,8],'max')
coremaps=[];freemaps=[];zc_list=[]
for scale in (2.5,4.5):
    st='secondary_'+str(int(2*scale));Z=blurk(st+'_local',filtered,scale); off=int(round(1.25*scale)); compar=[]
    # Single-site comparisons are separable sparse shifts; implement as
    # vertical and horizontal neighboring-bin shifts in the smaller stages.
    for j,(dx,dy) in enumerate([(0,off),(0,-off),(off,0),(-off,0),(off,off),(-off,off),(off,-off),(-off,-off)]):
        shifted=Z; rem=off
        while rem:
            step=min(rem,2);k=np.zeros((2*step+1,2*step+1));k[step+int(np.sign(dy))*step,step+int(np.sign(dx))*step]=-1 if rem==step else 1
            shifted=padded(st+'_shift%d_%d'%(j,rem),shifted,k,step);rem-=step
        compar.append(plus(st+'_saddle'+str(j),Z,shifted))
    axisContrast=dmin(st+'_min',dmin(st+'_vertical',compar[0],compar[1]),dmin(st+'_horizontal',compar[2],compar[3]));diagContrast=dmin(st+'_diag_min',dmin(st+'_da',compar[4],compar[5]),dmin(st+'_db',compar[6],compar[7]));contrast=dmin(st+'_all',axisContrast,diagContrast)
    # Only count maxima connected to the nucleus. Mixed scale comparison makes
    # the contrast stable at sampled locations which are not exactly at a peak.
    connected=affine(st+'_link',C,np.array([[[[2.4]]]]),-.012,True)
    contrast=plus(st+'_prominence',contrast,affine(st+'_saddle_excess',plus(st+'_over_saddle',Z,neg(st+'_subtract_saddle',C)),np.array([[[[.70]]]]),cv=True))
    strength=dmin(st+'_evidence',contrast,connected);zc_list.append(Z)
    if len(zc_list)==2:
        # At an isolated source, the ratio of coarse/fine central flux is
        # (w^2+s_fine^2)/(w^2+s_coarse^2). Suppress pointlike companions.
        broad=plus('resolved_core',Z,affine('cancel_unresolved',zc_list[0],np.array([[[[-.69]]]]),cv=True));broad=affine('extended_peak',broad,np.array([[[[3.8]]]]),cv=True)
        free=dmin('resolved_prominence',contrast,broad)
    for far in (False,True):
        outer=36 if far else 30;inner=11 if far else 5.5
        cost=np.minimum(1.5,np.maximum(r-outer,0)*.021+np.maximum(inner-r,0)*.19)
        mask=affine(st+'_mask'+str(far),pool(st+'_dummy'+str(far),filtered,[64,64],[64,64],'avg'),np.zeros((4096,1)),-cost.ravel())
        mask=reshape(st+'_rad'+str(far),mask,[1,64,64]); coremaps.append(spatialmax(st+'_max'+str(far),plus(st+'_radial_peak'+str(far),mask,strength)))
        if len(zc_list)==2:freemaps.append(spatialmax('extended_secondary'+str(far),plus('nearby_extended'+str(far),mask,free)))
# Scale-adaptive angular derivative signature. The ridge response contains
# only flux above a local broad-band continuum. An ellipse produces two
# edge ridges as well: subsequently remove the radial m=2 component of fixed
# orientation. Clipped diffuse light is measured separately from the core.
lo=affine('diffuse_ceiling',C,np.array([[[[1]]]]),-.095,True);Dlight=plus('diffuse_isophote',C,neg('negative_excess',relu('excess_nucleus',lo)))
hp=relu('substructure_ridges',plus('bandpass',blurk('fine_band',filtered,1.6),neg('coarse_negative',blurk('coarse_band',filtered,4.5))))
# Each scalar is an affine expression in named hidden units. The compiler below
# emits dense/ReLU stages, sharing units and pruning already-consumed values.
class Expr:
    def __init__(self,t=None,b=0): self.t=t or {}; self.b=b
    def __add__(self,o):
        if not isinstance(o,Expr):return Expr(self.t,self.b+o)
        t=self.t.copy()
        for i,v in o.t.items():t[i]=t.get(i,0)+v
        return Expr({i:v for i,v in t.items() if v},self.b+o.b)
    __radd__=__add__
    def __mul__(self,z): return Expr({i:z*v for i,v in self.t.items()},self.b*z)
    __rmul__=__mul__
    def __neg__(self):return self*-1
    def __sub__(self,o):return self+(-o)
    def __rsub__(self,o):return -self+o
    def __truediv__(self,z):return self*(1/z)

sources={}; depth={}; nodes={}; seq=0

def variable(d=0):
    global seq
    i=seq;seq+=1;depth[i]=d
    return i,Expr({i:1})
def source(name,n):
    ids=[]; ret=[]
    for _ in range(n):
        i,e=variable();ids.append(i);ret.append(e)
    sources[name]=ids;return ret

def R(e):
    if not isinstance(e,Expr): return max(0,e)
    i,v=variable(1+max([depth[k] for k in e.t]+[0]));nodes[i]=e;return v

def clip(x,l=0,u=1):return l+R(x-l)-R(x-u)
def ramp(x,l,u):return clip((x-l)/(u-l))
def mx(x,y):return R(x-y)+y
def mn(x,y):return x-R(x-y)
def extreme(xs,fn):
    while len(xs)>1:xs=[fn(xs[i],xs[i+1]) if i+1<len(xs) else xs[i] for i in range(0,len(xs),2)]
    return xs[0]
def Max(*xs):return extreme(list(xs),mx)
def Min(*xs):return extreme(list(xs),mn)
def absx(x):return R(x)+R(-x)

def interp(x,points):
    # Piecewise linear, constant extrapolation.
    out=points[0][1]; m0=0
    for (a,b),(c,d) in zip(points,points[1:]):
        m=(d-b)/(c-a);out+=(m-m0)*R(x-a);m0=m
    return out-m0*R(x-points[-1][0])

def logx(x,lo=.002,hi=2):
    z=np.geomspace(lo,hi,max(4,round(math.log(hi/lo)/.38)+1));return interp(x,[(a,np.log(a)) for a in z])
def ratio(a,b,mxval=3,lo=.002,hi=2):
    z=np.geomspace(.001,mxval,22);return interp(logx(a,lo,hi)-logx(b,lo,hi),[(np.log(t),t) for t in z])

# Linear aperture photometry bank. Apertures are evaluated directly in physical
# input pixels. Polar samples consist of Gaussian spots, not images of objects.
masks=[[] for _ in range(6)]; groups=[[] for _ in masks]
def measure(w,ch=0):
    masks[ch].append(np.asarray(w,F).ravel());i,e=variable();groups[ch].append(i);return e

def gauss(s,cx=0,cy=0):
    w=np.exp(-((x-cx)**2+(y-cy)**2)/(2*s*s)); return w/w.sum()

def beam(s,ch=0):return measure(gauss(s),ch)

def moment(s,channel=0):
    w=gauss(s)*r*r/(2*s*s)
    T=measure(w,channel)
    X=measure(w*np.cos(2*ang),channel);Y=measure(w*np.sin(2*ang),channel)
    return T,X,Y

def mag(X,Y):
    # Uniformly spaced absolute projections approximate Euclidean length.
    return sum(absx(X*np.cos(th)+Y*np.sin(th)) for th in np.arange(4)*np.pi/4)*(np.pi/8)

# Iterated elliptical-aperture shape is approximated with a ladder of scales.
# For a Gaussian source with covariance C, windowed C'=(C^-1+s^-2 I)^-1.
# The correction below inverts this truncation, rather than assuming the outer
# isophotes of all galaxies have the same surface brightness.
S=[4,8,14,23]; shape={}; flux={}; ellip={}; moments={}
for s in S:
    flux[s]=beam(s,3); T,X,Y=moment(s,3);A=mag(X,Y);moments[s]=(T,X,Y,A)
    maj=(T+A); minor=R(T-A)
    # maj and minor are variance along the principal axes divided by s^2, times flux.
    # correction: q^2 = (minor/maj) * (1-maj/flux)/(1-minor/flux).
    # Relax inversion for Sérsic profiles with cores and wings, and for noise.
    lq=.5*(logx(minor,.00025)-logx(maj,.00025))
    lq+=.36*(logx(.22*flux[s]+R(flux[s]-maj),.001)-logx(.22*flux[s]+R(flux[s]-minor),.001))
    qs=np.linspace(.07,1,16);shape[s]=interp(lq,[(np.log(q),q) for q in qs]);ellip[s]=ratio(A,T,1,.0002)
q=.21*shape[4]+.39*shape[8]+.31*shape[14]+.09*shape[23]
qcore=.72*shape[4]+.28*shape[8]; qdisk=.65*shape[14]+.35*shape[23]
core=beam(2.5); wide=beam(18)
# Undo the known binning/denoising and seeing variance in the projected
# central second moments. The upper reliability floor represents unresolved
# minor axes. This matters especially for angularly small smooth spheroids.
shapeObserved=shape.copy()
for s in S:
    T,X,Y,A=moments[s]
    alpha=min(.28,2.5**2/s**2)
    logcor=.5*(logx(R(T-A-alpha*flux[s])+.0004,.00015)-logx(T-A+.0004,.00015)-logx(R(T+A-alpha*flux[s])+.0004,.00015)+logx(T+A+.0004,.00015))
    sq=np.linspace(.32,1,12);corr=interp(logcor,[(np.log(t),t) for t in sq]);
    # First-order product by interpolation of log lengths
    qs=np.linspace(.05,1,15);shape[s]=interp(logx(shapeObserved[s],.04,1)+logx(corr,.25,1),[(np.log(t),t) for t in qs])
q=.18*shape[4]+.40*shape[8]+.32*shape[14]+.10*shape[23]
qcore=.7*shape[4]+.3*shape[8];qdisk=.62*shape[14]+.38*shape[23]
# Azimuthal half reversals of the quadrupole: a misaligned elongated bar
# turns sharply into the rounder outer disk. Measure ellipticity of the wings
# at the same position angle, not just their circularized light concentration.
# Correct circularised size approximately for inclination (a second-order effect).
ext=ratio(flux[14],flux[4],.99)
area=ext+.048*(1-q)
red4=beam(4,1);blue4=beam(4,2)
red12=beam(12,1);blue12=beam(12,2)
colourCore=ratio(blue4+.015,red4+.015,2.6)
colour=ratio(blue12+.004,red12+.004,2.6)
# Colour difference in the disk relative to the nucleus and a red-core cue.
redwing=R(red12-.24*red4);bluewing=R(blue12-.24*blue4)
outercolour=ratio(bluewing+.008,redwing+.008,2.6)
colourgrad=outercolour-colourCore

# Compare paired polar samples at progressively larger radii. Minimise a relative
# 180-degree asymmetry over a small set of possible centres to allow astrometry
# and pixelisation errors. An isolated bright point is less important than a
# whole asymmetric region, hence the clipping of local contrasts.
angs=np.arange(16)*2*np.pi/16
radii=[4,6,9,13.5,20,30,44]
pol=[];ringflux=[]
for rad in radii:
    row=[measure(gauss(max(1.45,rad*.106),rad*np.cos(t),rad*np.sin(t)),3) for t in angs]
    pol.append(row);ringflux.append(sum(row)/16)

# Nonnegative polar harmonic amplitudes, phases, and lagged m=2 turn.
re=[];im=[];harm=[];angles=0
for i,rad in enumerate(radii):
    real=sum(v*np.cos(2*t) for v,t in zip(pol[i],angs))/8
    imag=sum(v*np.sin(2*t) for v,t in zip(pol[i],angs))/8
    re.append(real);im.append(imag);harm.append(mag(real,imag))

# Departure from both mirror axes of an inclined smooth disk. A bar alone or an
# ellipse alone is mirror symmetric; a winding spiral is not. Use *one* axis
# across radii. Clip contrasts relative to local light to limit companion stars.
mirror=[]
for sector in range(8):
    v=[]
    for i in range(1,6):
        a=pol[i]; dif=sum(absx(a[j]-a[(sector-j)%16]) for j in range(16))/16
        # offset the discrete orientation-grid error using a small amount of the quadrupole
        v.append(R(dif-.15*harm[i]))
    mirror.append(v)
mrweights=[.09,.20,.30,.27,.14]
mirrorraw=Min(*[sum(w*d for w,d in zip(mrweights,v)) for v in mirror]); den=sum(w*f for w,f in zip(mrweights,ringflux[1:6]))
mirrorRel=ratio(mirrorraw,den+.018,1.8,.0003)

asrows=[]; afrows=[]
for dx,dy in [(0,0),(1.6,0),(-1.6,0),(0,1.6),(0,-1.6),(1.4,1.4),(1.4,-1.4),(-1.4,1.4),(-1.4,-1.4)]:
    v=[];f=[]
    for i in [1,2,3,4,5]:
        rad=radii[i]; a=pol[i][::2] if dx==dy==0 else [measure(gauss(max(1.45,rad*.106),dx+rad*np.cos(t),dy+rad*np.sin(t)),3) for t in angs[::2]]
        A=sum(absx(a[j]-a[j+4]) for j in range(4))/4
        v.append(A);f.append(sum(a)/8)
    asrows.append(sum(w*d for w,d in zip(mrweights,v)));afrows.append(sum(w*d for w,d in zip(mrweights,f)))
asym=ratio(Min(*asrows),den+.016,2.4,.0003)

# Match phase changes between rings, independently of absolute position angle or
# winding direction. Cross-ring triangle deficits are zero for aligned ellipses.
dis=[]; opp=[]
for i,j in [(1,3),(2,4),(3,5),(1,4),(2,5)]:
    together=mag(re[i]+re[j],im[i]+im[j]); sep=mag(re[i]-re[j],im[i]-im[j]);
    dis.append(R(harm[i]+harm[j]-together));opp.append(R(sep-.6*together))
twist=ratio(sum(dis)/5,den+.015,1.5,.00015)
armOpp=ratio(sum(opp)/5,den+.015,1.5,.00015)
# m=2 core strength (bar), compared to shape of the outer disk.
bar=ratio((harm[1]+harm[2]+.5*harm[0])/2.5,(ringflux[1]+ringflux[2]+.5*ringflux[0])/2.5+.012,1.5)
barOuter=ratio((harm[3]+harm[4])/2,(ringflux[3]+ringflux[4])/2+.014,1.5)
# Round radial profiles are not the same as smooth profiles. Isolate steep local
# negative radial derivatives that turn positive again in the disk (arm crests).
bumps=[]
for i in range(1,6):
    bumps.append(sum(R(pol[i+1][j]-pol[i][j]-.009) for j in range(16))/16)
ripple=ratio(sum(w*v for w,v in zip(mrweights,bumps)),den+.01,1.8,.0002)

# Off-centre nucleus: Mexican-hat filters with an annular saddle test. The latter
# demands that the response be a peak in all directions rather than a bright bit
# of a monotonic disk. Maximise at inference (no peak coordinates in weights).
external=[]
for nm in coremaps+freemaps:
    i,e=variable();external.append((nm,i,e))
peaks=[external[j][2] for j in (0,2)];peaksfar=[external[j][2] for j in (1,3)]
nuc0=[measure(gauss(sz)-gauss(2.35*sz)) for sz in (2.5,4.5)]
# Companions should be extended: use evidence at two resolutions, and coherence
# with other asymmetry, rather than classifying every foreground star as a merger.
peak=.35*peaks[0]+.65*peaks[1]; peakRel=ratio(peak,.065+.45*core,2)
peakFar=.4*peaksfar[0]+.6*peaksfar[1]
nu=ratio(.5*(nuc0[0]+nuc0[1]),core+.03,1.5)

# Spatial structure on top of a smooth disk. Gaussian-Laplacian bright ridges and
# azimuthal butterflies (third and fourth harmonics). Circle/ellipse subtraction
# is done before rectification so bright smooth disks need not have large texture.
tex=[]; patches=[]
for i in [1,2,3,4,5]:
    rad=radii[i];a=pol[i]
    # A smooth incline can have high even angular harmonics. Restrict the utility
    # of the high-harmonic signal to low inclinations in the readout.
    v=sum(absx(a[j]-.5*(a[(j-2)%16]+a[(j+2)%16])-(re[i]*np.cos(2*angs[j])+im[i]*np.sin(2*angs[j]))) for j in range(16))/16
    tex.append(v)
    vals=[]
    for j,t in enumerate(angs):
        xx=rad*np.cos(t);yy=rad*np.sin(t); s=max(1.35,rad*.083)
        vals.append(R(measure(gauss(s,xx,yy)-gauss(s*2.5,xx,yy))-.009))
    patches.append(sum(vals)/16)
texture=ratio(sum(w*v for w,v in zip(mrweights,tex)),den+.018,2,.0003)
knots=ratio(sum(w*v for w,v in zip(mrweights,patches)),den+.018,1,.0002)

# A rotational elongation statistic for the inner and outer isophotes: annular
# ellipticity from angular flux is independent of absolute position angle.
# Absolute nuclear contrast and spread of the major axis distinguish thin disks
# with a rounded bulge from diffuse late-type needles, and from small ellipticals.
conc=ratio(core,flux[8]+.005,5)
# Scale-free ratio of central excess over the disk; excess concentration together
# with a rounder or redder core is evidence of a stellar bulge.
bulge=(.31*ramp(conc,1.6,2.9)+.25*ramp(qcore-qdisk,.025,.34)+
       .23*ramp(.99-colourCore,.06,.5)+.21*ramp(nu,.25,.57))

# Winding is a signed radial gradient of the polar Fourier phase. An affine
# quadrupole from projection alone increases or decreases on a fixed line in
# the (cos,sin) plane. Use triangle area of consecutive vectors via a small
# piecewise-linear product on normalized amplitudes, at inference only.
def signedfrac(v,den,lo=.0003):return ratio(R(v),den,1.65,lo)-ratio(R(-v),den,1.65,lo)
def prod(u,v):
    z=np.unique(np.r_[np.linspace(-3.4,3.4,35),np.arange(-.2,.201,.025)]);tab=[(t,t*t*.25) for t in z]
    return interp(u+v,tab)-interp(u-v,tab)
cr=[signedfrac(re[i],ringflux[i]+.022) for i in range(6)]
ci=[signedfrac(im[i],ringflux[i]+.022) for i in range(6)]
cross=[prod(cr[i],ci[i+1])-prod(ci[i],cr[i+1]) for i in range(1,5)]
winding=absx(.23*cross[0]+.33*cross[1]+.29*cross[2]+.15*cross[3]); winding=mx(winding,.55*Max(*(absx(v) for v in cross)))
# Size relative to the seeing disk, independent of sky fluctuations. For
# concentrated/truncated Sérsic wings use annular flux as well as the mean.
majorVar=ratio(moments[14][0]+moments[14][3],flux[14],1.5,.0003)
spread=ramp(majorVar,.12,.53)
outerFraction=ratio(ringflux[3]+ringflux[4],ringflux[0]+ringflux[1]+.018,1,.0004)
radialSize=ramp(outerFraction,.065,.35)
# Partner contrast against local saddle, relative to *nuclear contrast* rather
# than total central light. One of two nuclei can lie near the image centre.
double=ratio(peak,.035+.65*(nuc0[0]+nuc0[1])/2,2.5,.0005)
# Excursions in differential structure are stronger in coherent, broad arms
# than in tight flocculent spirals. Polar crest and winding are complementary.
# Log-polar matched filters for the *ridges*, not the nucleus. Their radial
# sine/cosine frequency is m/tan(pitch). Remove the stationary quadrupole
# from each analysis filter, so a straight ridge does not mimic a spiral.
# Separate signs cover both senses of winding. Normalize each sinusoid
# against its own radial aperture, never against examples.
pitchrows=[]
for s in (8,14):
    w=gauss(s)*(1-np.exp(-r*r/(2*(s*.55)**2)));w/=w.sum();normal=measure(w,5)
    bypitch=[]
    for pitch in (14,24,42):
        both=[]
        for n in (1,-1):
            z=np.exp(-1j*n*2/np.tan(np.deg2rad(pitch))*np.log(r/s));z-=np.sum(z*w)
            z*=np.exp(2j*ang);both.append(mag(measure(w*z.real,5),measure(w*z.imag,5)))
        bypitch.append(ratio(Max(*both),normal+.002,1.6,.00008))
    pitchrows.append(bypitch)
tightPitch,midPitch,loosePitch=[.47*a+.53*b for a,b in zip(*pitchrows)]
# Outer light axis ratio in the low-brightness envelope: this includes the
# disk but progressively suppresses the central spheroid. In the absence of
# high contrast, use the ordinary moments as a second, independent opinion.
outerQS=[]; lin=[];widths=[];envel=[]
skycut=relu('sky_pedestal_removed',affine('local_sky_cut',Dlight,np.array([[[[1]]]]),-.023,True))
for s in (10,18):
    T,X,Y=moment(s,4);A=mag(X,Y);lum=beam(s,4)
    ma=T+A;mi=R(T-A); lq=.5*(logx(mi+.00015,.00008)-logx(ma+.00015,.00008))
    lq+=.42*(logx(.15*lum+R(lum-ma),.0002)-logx(.15*lum+R(lum-mi),.0002))
    qs=np.linspace(.06,1,15);outerQS.append(interp(lq,[(np.log(v),v) for v in qs]))
    lin.append(ratio(ma,lum+.0006,2,.00012));widths.append(ratio(mi,lum+.0006,1,.00012))
    envel.append(lum)
outerQ=.43*outerQS[0]+.57*outerQS[1]
# Decompose the outer isophotes into a narrow linear strip and an oblate
# envelope. Apply a quadrupole Fourier projection at each candidate angle;
# the maximal orientation pools over a half turn. At the same angle compare
# the core to equidistant locations above the disk to look for a central bulge.
strips=[];centralPuff=[];thinW=[]
for a0 in np.arange(12)*np.pi/12:
    u=x*np.cos(a0)+y*np.sin(a0);v=-x*np.sin(a0)+y*np.cos(a0)
    g=np.exp(-.5*((abs(u)-14)/6.5)**2)
    h=g*np.exp(-.5*(v/2.6)**2);h/=h.sum()
    h0=g*np.exp(-.5*((abs(v)-6)/2.6)**2);h0/=h0.sum()
    ref=measure(h,3);side=measure(h0,3)
    # Ridge minus local background on either side. Max is gap magnitude.
    strips.append(ref-side);thinW.append(ref)
    h2=np.exp(-.5*((u/3.5)**2+((abs(v)-5)/2.4)**2));h2/=h2.sum()
    centralPuff.append(measure(h2,3)-side)
line=Max(*strips);linear=ratio(line,Max(*thinW)+.014,1.5,.00025)
# Gates choose only orientations which have a competitive thin disk. Compare
# height of light over the nucleus with light above the outer disk.
puff=ratio(Max(*[p-2.2*(line-z) for z,p in zip(strips,centralPuff)]),core+.012,1.2,.0003)
# Tail asymmetry: AGN or an off-centre nucleus can bias ordinary asymmetry;
# the odd angular harmonics of low-brightness isophotes at larger radius
# independently tell whether the light is lopsided or has a tidal tail.
odd=[];faint=[]
for s in (8,16,24):
    w=gauss(s)*r*r/(2*s*s);f=measure(w,4);faint.append(f)
    vals=[mag(measure(w*np.cos(m*ang),4),measure(w*np.sin(m*ang),4)) for m in (1,3)]
    odd.append(ratio(.66*vals[0]+.34*vals[1],f+.002,1,.0002))
lopsided=.3*odd[0]+.5*odd[1]+.2*odd[2]
# Direct ellipse model on isophote contours. The excursion area at fixed
# levels is insensitive to a bright slightly decentered nucleus, and central
# second moments remove the centroid shift. Handle unresolved minor axes via
# the known seeing covariance. Pool upper and lower levels (one traces the
# nucleus, another the disk). Smooth each threshold over a narrow intensity
# interval. Thresholds are in display RGB units, not estimated from a sample.
iso_q=[];iso_len=[];iso_width=[];iso_area=[];iso_offset=[];iso_planes=[]
for k,(level,transition) in enumerate([(.046,.035),(.15,.055),(.32,.07)]):
    nm='isophote%d'%k;v=relu(nm+'_positive',affine(nm+'_lower',C,np.array([[[[1]]]]),-level,True));ex=relu(nm+'_excess',affine(nm+'_upper',v,np.array([[[[1]]]]),-transition,True));bl=plus(nm,v,neg(nm+'_remove_excess',ex));iso_planes.append(bl);masks.append([]);groups.append([])
    ch=6+k;ap=(r<=38).astype(float)/math.pi/38**2*4/transition
    area0=measure(ap,ch);iso_area.append(area0);xx=measure(ap*x/20,ch);yy=measure(ap*y/20,ch)
    # Express distances in units of 20 arcseconds-on-input pixels.
    vx=signedfrac(xx,area0+.00008,.000001);vy=signedfrac(yy,area0+.00008,.000001)
    x2=ratio(measure(ap*(x/20)**2,ch),area0+.00008,2,.000002)-prod(vx,vx)
    y2=ratio(measure(ap*(y/20)**2,ch),area0+.00008,2,.000002)-prod(vy,vy)
    xy=signedfrac(measure(ap*x*y/400,ch),area0+.00008,.000001)-prod(vx,vy)
    e=mag(x2-y2,2*xy);major=R(.5*(x2+y2+e));minor=R(.5*(x2+y2-e));
    # No seeing correction on the outskirts. Correct only a portion at high
    # levels; one cannot distinguish a PSF from a microgalaxy with certainty.
    q2=ratio(R(minor-.003)+.002,R(major-.003)+.002,1,.0001)
    axis=interp(q2,[(t*t,t) for t in np.linspace(.06,1,18)])
    iso_q.append(axis);iso_len.append(major);iso_width.append(minor);iso_offset.append(mag(vx,vy));
# Low isophotes include the diffuse disk; nucleus may be below a higher
# isophote, so fall back continuously to the middle ellipse in this case.
isoq=.55*iso_q[0]+.45*iso_q[1];isomajor=iso_len[0]
# Very small contours are pixelised; regular weighted moments give the
# complement to contour features in that regime.
# Bar model: compare inner straightness with the outer body, in phase as
# well as amplitude. Normalize m=2 by the radial *gradient*; disk inclination
# otherwise produces a spurious rise of m=2 as the surface brightness falls.
def shear(i):
    dv=R(ringflux[i-1]-ringflux[i+1])/.81+.035*ringflux[i]+.012
    return signedfrac(re[i],dv),signedfrac(im[i],dv)
ex1,ey1=shear(1);ex2,ey2=shear(2);ex3,ey3=shear(3);ex4,ey4=shear(4)
biX=.50*(ex1+ex2);biY=.50*(ey1+ey2);boX=.55*ex3+.45*ex4;boY=.55*ey3+.45*ey4
barExcess=mag(biX-boX,biY-boY);innerStrength=mag(biX,biY);outerStrength=mag(boX,boY)
# A bar has long, relatively straight contours and a red stellar population.
# Measure the excess of the even central mode in red light over blue. The
# central bulge contributes to both but predominantly to the monopole.
barcolour=[]
for ch in (1,2):
    w=gauss(5.5)*r*r/(2*5.5**2);A=mag(measure(w*np.cos(2*ang),ch),measure(w*np.sin(2*ang),ch));barcolour.append(ratio(A,measure(w,ch)+.006,1,.0001))
barRed=barcolour[0]-barcolour[1]
# A foreground neighbour may have no bridge to the main galaxy. Distinguish
# a resolved companion from a tiny star by testing the saddle contrast on a
# wider smoothing scale and the local surface brightness between the two.
# Isomorphic polar symmetry about the centroid: a more faithful test of
# roundness than PSF deconvolution on a poorly resolved source. Compare radial
# widths at a ring of eight aperture angles; remove dipole contamination by
# folding antipodal pairs. The analytic mixture then gives *two* descriptions:
# measured (apparent) roundness, and intrinsic axis ratio for inclined disks.
qapp=.35*shapeObserved[4]+.40*shapeObserved[8]+.25*shapeObserved[14]
# Differential log-polar arm strength relative to its *own annular continuum*.
# Tight arms sum to little in a broad, low-frequency quadrupole; loose ones
# have a coherent nonstationary m=2 excess of either winding handedness.
lowAmp=[]
for sc in (7,13):
    w=gauss(sc)*(r/sc)**2;w/=w.sum(); osc=np.exp(1j*2*ang);mon=measure(w,3)
    vals=[]
    for pitch in (18,36,56):
        modes=[]
        for handed in [-1,1]:
            phase=np.exp(2j*handed*np.log(r/sc)/np.tan(np.deg2rad(pitch)));phase-=np.sum(w*phase)
            z=w*osc*phase;modes.append(mag(measure(z.real,3),measure(z.imag,3)))
        # Use chirality instead of just the modulus. Every mirror-symmetric
        # ellipse has equal energies in the two senses even at high inclination.
        vals.append(ratio(absx(modes[0]-modes[1]),mon+.012,1,.00006))
    lowAmp.append(vals)
lt,lo,vl=[.47*a+.53*b for a,b in zip(*lowAmp)];openChiral=mx(lo,vl);chiral=mx(openChiral,lt)
freePeak=.6*external[4][2]+.4*external[5][2];freePair=ratio(freePeak,.032+.58*nuc0[1],2.2,.0004)
# High concentration relative to second-moment extent is characteristic of
# smooth spheroids even when angularly large. Disk profiles occupy much more
# area at one-half of their nuclear intensity. Use the ratio of inner and
# outer Gaussian flux ratios as a curvature (Sersic-index) measurement.
halfcore=beam(1.6); f6=beam(6,3); f16=beam(16,3)
profile=logx(halfcore+.004)-logx(f6+.004)-.47*(logx(f6+.004)-logx(f16+.004))
# Morphology readout: establish fuzzy evidence terms, then implement a hierarchy
# with margins. The family ranking and the within-family choices are separate,
# so adding extra subclasses does not affect the class prior.
sizecue=.34*ramp(area,.17,.40)+.33*radialSize+.33*spread
bluecue=ramp(colour,.83,1.24)
bluedisk=ramp(outercolour,.93,1.31)
armcue=ramp(mirrorRel,.07,.36)
windcue=ramp(twist,.012,.15)
bumpcue=ramp(ripple,.016,.13)
rough=ramp(texture,.06,.26)
clump=ramp(knots,.011,.083)
openarms=(.29*ramp(twist,.025,.19)+.26*ramp(ripple,.035,.22)+.24*ramp(mirrorRel,.16,.66)+.11*ramp(armOpp,.06,.45)+.10*ramp(outercolour,1.,1.44))
barcue=(.30*ramp(bar,.16,.52)+.31*ramp(bar-barOuter,-.10,.28)+.19*ramp(shape[14]-shape[4],-.05,.24)+.13*ramp(colourgrad,.05,.5)+.07*ramp(winding,.009,.1))
openarms=.66*openarms+.23*ramp(winding,.010,.100)+.11*ramp(ripple,.012,.09)
face=ramp(q,.28,.62)
# For a flat star-forming disk greater size and a cool disk are also disk evidence,
# but are not spiral-arm evidence without the two-fold symmetric disk being tilted.
smoothDist=(.55*ramp(asym,.13,.53)+.25*ramp(mirrorRel,.12,.48)+.20*clump)
spiral=(.29*sizecue+.13*bluecue+.10*bluedisk+.16*armcue+.11*windcue+.10*bumpcue+.07*clump+.04*ramp(colourgrad,.08,.47))
# In a highly regular, extended blue disk the arms need not be very contrasty.
spiral+=.09*Min(sizecue,bluecue)+.05*Min(sizecue,barcue)
disturbed=(.39*ramp(asym,.19,.69)+.22*ramp(mirrorRel,.19,.67)+.18*ramp(ripple,.04,.22)+.12*clump+.09*rough)
merger=(.46*ramp(peakRel,.19,.72)+.20*ramp(peak,.047,.19)+.19*ramp(asym,.21,.78)+.15*ramp(peakFar,.04,.17))
# Thin projected axis, linear extent, and low central concentration or a disk-plus-
# bulge profile distinguish true edge-on disks from cigar-shaped smooth galaxies.
edgeQ=.60*qdisk+.40*q
edge=(.63*ramp(.52-edgeQ,0,.30)+.20*sizecue+.10*bluecue+.07*ramp(qcore-qdisk,.045,.30))
edge+=.15*Min(ramp(.40-edgeQ,0,.18),ramp(area,.135,.26))
# Re-centred smooth ellipticity boundary: round / intermediate / cigar shaped.
# A probability margin (not a normalized softmax) for each family.
sm=0
sp=3.4*(spiral-.44)-1.7*ramp(.39-q,0,.19)
ed=3.5*(edge-.565)
di=3.5*(disturbed-.48)-.63*sizecue-.40*windcue
me=3.7*(merger-.48)-.20*sizecue+.85*ramp(double,.32,1.25)
# An asymmetric compact blue object without a resolved disk is not a
# face-on spiral simply because it is blue; likewise a clearly resolved,
# symmetric extended disk with weak tightly wound arms can be a spiral.
compact=1-sizecue
di+=.75*Min(compact,ramp(asym,.14,.48))+.46*Min(compact,bluecue)-.15*windcue
sp+=.43*sizecue+.45*Min(sizecue,ramp(winding,.006,.1))-.40*Min(compact,ramp(asym,.16,.55))
ed+=.38*Min(spread,ramp(.49-edgeQ,0,.25))

def family_scores(base,raw):
    best=Max(*raw)
    return [base+v-best for v in raw]
# Relate classification to geometry in the diffuse plane. Prominent bulges
# can look round at half-light while being enclosed by an almost linear disk.
linecue=ramp(linear,.29,.70)
longcue=.50*ramp(lin[0],.42,.95)+.50*ramp(lin[1],.28,.79)
needle=Min(linecue,.60*ramp(.55-outerQ,0,.30)+.40*longcue)
ed=mx(ed,3.5*(needle-.53)+.42*(longcue-.5))
sp-=.96*Min(linecue,ramp(.53-outerQ,0,.24))
# The wing-to-nucleus colour gradient and central light above the midplane
# pick up bulges in addition to concentration. Reapply to the chosen disk.
bulge=.46*bulge+.24*ramp(puff,.19,.53)+.20*ramp(colourgrad,.04,.43)+.10*ramp(qcore-outerQ,.02,.26)
# Detect a prominent m=2 central bar even if the outer ring is also oval.
barSig=ramp((harm[0]+harm[1])*.5-.10*(ringflux[0]+ringflux[1])*.5,.002,.09)
barcue=.46*barcue+.24*barSig+.18*ramp(outerQ-qcore,-.02,.27)+.12*ramp(colourgrad,.04,.39)
openarms=.44*openarms+.26*ramp(loosePitch-tightPitch,-.05,.22)+.14*ramp(loosePitch,.10,.37)+.16*ramp(ripple,.008,.085)
tidal=ramp(lopsided,.075,.27)
di+=.95*tidal+.47*Min(tidal,compact)+.44*Min(compact,ramp(colourCore,.99,1.36))
me+=.44*Min(tidal,ramp(peak,.023,.14))
# Smooth labels describe the projected isophotes: bright nuclei should
# not turn a cigar into an intermediate spheroid. Diffuse image moments
# receive less weight for round systems, where the angle is unresolved.
qsmooth=.38*q+.20*outerQ+.42*isoq
isoEdge=Min(ramp(.515-iso_q[0],0,.245),ramp(isomajor,.048,.26)+.22*ramp(.36-iso_q[0],0,.14))
ed=mx(ed,3.0*(isoEdge-.45)+.28*longcue)
sp-=.80*Min(isoEdge,linecue);di-=.90*isoEdge;me-=.82*Min(isoEdge,1-ramp(peak,.025,.16))
# A low-isophote major semiaxis of a uniform ellipse is 2 std. This directly
# links the size gates to whether there is enough room for resolved arms.
resolved=ramp(isomajor,.06,.32);symdisk=Min(resolved,1-ramp(lopsided,.10,.30))
sp+=.44*Min(symdisk,ramp(q,.29,.55))-.26*Min(1-resolved,ramp(asym,.13,.5))
di+=.56*Min(1-resolved,ramp(asym,.16,.54))
bulge=.69*bulge+.18*ramp(iso_q[1]-iso_q[0],-.035,.25)+.13*ramp(.96-colourCore,.015,.34)
barModel=.32*ramp(innerStrength,.10,.35)+.33*ramp(barExcess,.10,.34)+.15*ramp(innerStrength-outerStrength,-.04,.19)+.11*ramp(barRed,-.025,.11)+.09*ramp(colourgrad,.06,.38)
barcue=.42*barcue+.58*barModel
# Prevent a distant ridge peak that fails the saddle test from defeating strong
# disky geometry. Subtract only for *bilaterally symmetric* disk candidates.
me-=.82*Min(1-ramp(asym,.13,.47),Max(isoEdge,Min(sizecue,face)))
# In an apparently round galaxy make the explicit distinction between
# 'round smooth' and 'intrinsically spherical'. The former is what is labeled.
qsmooth=.36*qsmooth+.64*qapp
# Smooth disks/ellipticals are nearly invariant under both point reflection
# and a mirror reflection. Size and blue colour alone are weak evidence; a
# shared absence of twist, bright ridges and asymmetry favours the smooth family.
regular=1-Max(ramp(asym,.115,.42),ramp(chiral,.009,.063),ramp(mirrorRel,.09,.32))
redOld=1-ramp(colour,.79,1.06)
sm+=.82*regular+.31*Min(redOld,1-ramp(mirrorRel,.11,.36))+.28*Min(regular,ramp(profile,-.03,.23))-.48
# Resolved light *length*, not circularized area, differentiates a small cigar
# from a disk seen edge-on. This also helps the case of a medium-sized thick bulge.
arealong=.58*ramp(isomajor,.045,.23)+.42*longcue
edgeDisk=Min(ramp(.62-iso_q[0],0,.31),arealong)
ed=mx(ed,3.3*(edgeDisk-.52)+.26*Min(ramp(.57-isoq,0,.27),ramp(linear,.21,.50)))
sp-=.54*Min(edgeDisk,ramp(.56-isoq,0,.26))
me=mx(me,3.6*(.62*ramp(freePair,.26,.95)+.38*ramp(freePeak,.031,.115)-.5)+.35*tidal)
# A prominent galaxy-galaxy pair can be asymmetric even when the two
# centres are far enough apart that a centre-only mask captures just one.
me+=.32*Min(ramp(peakRel,.16,.55),ramp(iso_offset[0],.06,.24))
# Coherent arms distinguish a spiral from compact irregular structure. A
# spiral with a smooth disk that is resolved only in the outer blue light
# is not expected to have a high-asymmetry *nucleus*.
rawOdd=ratio(sum(absx(pol[1][j]-pol[1][j+8]) for j in range(8))/8,ringflux[1]+.025,2,.0003)
di+=.59*Min(compact,ramp(rawOdd,.12,.55))+.40*Min(1-resolved,Min(bluecue,1-ramp(chiral,.02,.09)))
openarms=.52*openarms+.32*ramp(openChiral,.012,.09)+.16*ramp(openChiral-lt,-.004,.045)
# Readability of a bar is primarily a central-excess feature (not the
# outer winding). A single inclination with a red uniform disk should have
# neither excess in shear nor a transition in stellar colour.
barcue+=.12*Min(ramp(barExcess,.06,.24),ramp(colourgrad,.025,.35))
bulge+=.15*ramp(iso_q[2]-iso_q[0],.055,.32)+.11*ramp(qapp-iso_q[0],.125,.34)
smooth_scores=family_scores(sm,[3.2*(qsmooth-.816),0,3.6*(.49-qsmooth)])
spiral_scores=family_scores(sp,[2.0*(barcue-.41),0,2.2*(openarms-.36)])
edge_scores=family_scores(ed,[0,2.4*(bulge-.44)])
res=[di,me]+smooth_scores+spiral_scores+edge_scores

ids=[]; parts=[]; starts=[]
for ch,part in enumerate(masks):
    plane=([C,skycut,hp]+iso_planes)[ch-3] if ch>=3 else affine('filter_%d'%ch,light,np.eye(3)[ch:ch+1,:,None,None],cv=True)
    starts.append(len(ids));ids.extend(groups[ch]);parts.append(affine('photometry_%d'%ch,plane,part))
for nm,i,e in external:
    starts.append(len(ids));ids.append(i);parts.append(nm);groups.append([i])
src='photometry';sources[src]=ids;del masks

# Emit the scalar arithmetic as a ReLU network. Values not known to be positive
# are sent through relay layers as positive/negative pairs (photometry only).
used={i for e in res if isinstance(e,Expr) for i in e.t};todo=list(used)
while todo:
    i=todo.pop()
    for k in nodes.get(i,Expr()).t:
        if k not in used:todo.append(k);used.add(k)
D=max((depth[i] for i in used),default=0);last={}
for i in used:
    if i in nodes:
        for k in nodes[i].t:last[k]=max(last.get(k,0),depth[i])
for e in res:
    for k in e.t:last[k]=D+1
# Photometric vectors are much larger than the arithmetic stages. Fuse sparse
# first-use linear combinations with the original masks, and do not emit dense
# quadratic identity matrices for the raw first stage.
assert len(sources)==1
index={i:j for j,i in enumerate(ids)}; cur=src
for d in range(1,D+2):
    if d==D+1:
        expr=res;keys=[]
    else:
        keys=[i for i in sorted(used) if depth[i]==d];expr=[nodes[i] for i in keys]
        for i in sorted(used):
            if depth[i]<d and last[i]>d:
                keys.append(i);expr.append(Expr({i:1}))
                if depth[i]==0:keys.append(~i);expr.append(Expr({i:-1}))
    W=np.zeros((len(expr),len(index)),F);B=np.zeros(len(expr),F)
    for j,e in enumerate(expr):
        if not isinstance(e,Expr):B[j]=e;continue
        B[j]=e.b
        for i,v in e.t.items():
            W[j,index[i]]+=v
            if ~i in index:W[j,index[~i]]-=v
    name='morphology_%02d'%d if d<=D else 'class_scores'
    if d==1:
        contributions=[affine('integrate_%d'%ch,part,W[:,a:a+len(groups[ch])],B if ch==0 else 0) for ch,(part,a) in enumerate(zip(parts,starts))];cur=layer(name,'add',contributions)
    else:cur=affine(name,cur,W,B)
    if d<=D:cur=relu('hinges_%02d'%d,cur);index={i:j for j,i in enumerate(keys)}

(out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
np.savez(out/'weights.npz',**weights)
