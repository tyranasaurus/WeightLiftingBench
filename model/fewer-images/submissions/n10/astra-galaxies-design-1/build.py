"""Build an analytic photometry and morphology network. No input data or weights."""
import json
import math
import os
import sys
import numpy as np

D=sys.argv[1]
os.makedirs(D,exist_ok=True)
layers=[]
weights={}

def op(name,kind,src,**kw):
    layers.append(dict(name=name,op=kind,inputs=src if isinstance(src,list) else [src],**kw)); return name

def dense(name,src,W,b=0):
    W=np.asarray(W,'f'); weights[name+'.W']=W.reshape(len(W),-1); weights[name+'.b']=np.broadcast_to(b,(len(W),)).astype('f').copy()
    return op(name,'dense',src,out=len(W))

def conv(name,src,W,b=0,stride=1,padding=0):
    W=np.asarray(W,'f'); weights[name+'.W']=W; weights[name+'.b']=np.broadcast_to(b,(len(W),)).astype('f').copy(); k=W.shape[-1]
    return op(name,'conv',src,out=len(W),kernel=[k,k],stride=stride,padding=padding)

def pool(name,src,k,kind='avgpool',stride=None):
    s=stride or k; return op(name,kind,src,size=[k,k],stride=[s,s])

def reshape(name,src,sh):return op(name,'reshape',src,shape=sh)
def relu(name,src):return op(name,'relu',src)

# Downsample with proper area integration. Use two scales: a reference-frame
# map for measuring the centroid, and a finely sampled map which we translate.
p2=pool('bin2','image',2)
p4=pool('bin4',p2,2)
phot=np.array([.31,.47,.22]); chrom=np.array([1.,0.,-1.])
def proj(name,src,mat,b=0):return conv(name,src,np.array(mat)[:,:,None,None],b)
light=proj('guide_luminance',p2,[phot])
# Guide on contrast in excess of the sky. Gaussian apodization greatly
# reduces the chance of following a distant companion instead of the target.
zc=relu('guide_isophote',proj('guide_cut',light,[[1]],-.15))
# A scalar weighted first moment divided by total weight. The norm in the
# allowed layer set can implement division: encode (+/-scale,+/-value),
# for a bounded slope value/scale. A four-element norm then returns a sine
# of the angle; the inverse mapping is done by piecewise linear interpolation.

def ratio_layer(name,src,n,extent=1.,step=.001):
    # src contains interleaved numerators and denominators of n independent ratios.
    w=np.array([[1/extent,0],[-1/extent,0],[0,1],[0,-1.]])
    # dense rearrangement only used for small groups of photometric readouts
    ww=np.kron(np.eye(n),w)
    a=dense(name+'_quadratures',src,ww)
    a=reshape(name+'_angles',a,[n,1,4]); a=op(name+'_unit','norm',a)
    # For |ratio/extent| beyond 3.5, saturate, rather than extrapolate.
    knots=np.array([-3.5,-2.5,-1.75,-1.25,-1.,-.75,-.50,-.25,0.,.25,.5,.75,1,1.25,1.75,2.5,3.5]); z=np.sqrt(2)*knots/np.sqrt(1+knots**2)
    # Put channels in a spatial dimension, so a shared conv performs interpolation.
    take=np.zeros((n,4*n)); take[np.arange(n),np.arange(n)*4]=1
    a=dense(name+'_ordinate',a,take); a=reshape(name+'_ordinates',a,[1,1,n]); a=relu(name+'_hinges',proj(name+'_breakpoints',a,np.ones((len(z),1)),-z))
    slopes=np.diff(knots*extent)/np.diff(z); coeff=np.diff(np.r_[0,slopes,0]); a=proj(name+'_quotient',a,[coeff],knots[0]*extent)
    return reshape(name,a,[n])

yg,xg=(np.indices((64,64))+.5)*2-64
g=np.exp(-(xg*xg+yg*yg)/(2*7.**2)); g/=g.sum(); gain=300
cent0=dense('guide_moments',zc,[g*xg*gain,g*gain,g*yg*gain,g*gain],[0,.15,0,.15]); cent0=ratio_layer('coarse_centroid',cent0,2,extent=4.)
# Two more mean-shift iterations tighten the registration on the nucleus. We
# tabulate moments on a lattice of shifted circular apertures. Interpolation
# of the local estimates with a bounded-slope upper/lower envelope takes the
# value at the current guide coordinates, not an average of the guide grid.
cent=cent0
pts=np.arange(-12,13,3); coords=np.array([(xx,yy) for xx in pts for yy in pts]); Kc=len(coords)
for n,sigma in enumerate([4.0,2.9]):
    nm='recentering%d'%n; m=[]
    for xx,yy in coords:
        g=np.exp(-((xg-xx)**2+(yg-yy)**2)/(2*sigma**2)); g/=g.sum();g*=gain
        m.extend([g*(xg-xx),g,g*(yg-yy),g]);
    a=dense(nm+'_moments',zc,m,np.tile([0,.15,0,.15],Kc)); a=ratio_layer(nm+'_mean',a,Kc*2,3.); a=dense(nm+'_estimator',a,np.eye(Kc*2),coords.flatten());
    q=dense(nm+'_separations',cent,np.tile(np.r_[np.eye(2),-np.eye(2)],(Kc,1)),np.concatenate([np.r_[-pt,pt] for pt in coords]));q=relu(nm+'_unsigned',q)
    # The centroid operator is approximately 1-Lipschitz for a single peak.
    # Take both envelopes, so errors at points between grid nodes cancel.
    w=np.zeros((4*Kc,4*Kc)); b=np.zeros((4*Kc,2*Kc));
    for j in range(Kc):
        w[4*j:4*j+4,4*j:4*j+4]=-1; b[4*j:4*j+2,2*j:2*j+2]=np.eye(2); b[4*j+2:4*j+4,2*j:2*j+2]=-np.eye(2)
    v=op(nm+'_cones','add',[dense(nm+'_distance',q,w),dense(nm+'_heights',a,b)]);
    v=dense(nm+'_transpose',v,np.eye(4*Kc).reshape(Kc,4,-1).transpose(1,0,2).reshape(4*Kc,-1));v=reshape(nm+'_groups',v,[4,1,Kc]);v=op(nm+'_bounds','maxpool',v,size=[1,Kc],stride=[1,Kc]);cent=dense(nm+'_centre',v,[[.5,0,-.5,0],[0,.5,0,-.5]])
# Interpolate the shift within a central field. The formula clip(x-lo,0,h)
# distributes a bounded intensity among adjoining candidate translations;
# summing gated neighbours implements a monotone interpolation. Integer shifts
# use exactly the indicated neighbour. First horizontal, then vertical.
# Grids are in units of a two-pixel bin. Include margins for the soft shift.
N=32; M=4; full=N+2*M
def crop(src,side,new,ch,name):
    # A fixed gather expressed as a one-hot dense linear operation.
    ids=np.arange(ch*side*side).reshape(ch,side,side)[:,(side-new)//2:(side+new)//2,(side-new)//2:(side+new)//2].ravel()
    w=np.zeros((len(ids),ch*side*side),np.float32); w[np.arange(len(ids)),ids]=1
    return reshape(name+'_map',dense(name,src,w),[ch,new,new])

# Untranslated map of the centre; object is never scaled to a standard size.
# Crop the guide (one channel). Color photometry need not be resampled at every
# fine pixel and is computed in larger fixed apertures on the original sky.
field=crop(light,64,full,1,'field')
# Select among neighbouring shifts using the sums of clipped signed
# differences. For intensities in [0,1] this is exact at both endpoints
# and bounded and continuous in between.
def translate(src,axis):
    # Use a 1x1-style dense permutation (one sparse matrix per coordinate
    # direction would be huge). Use whole field fixed and shifts implemented
    # with one small square convolution on the map at the first step.
    nm='shift_'+('x' if axis==0 else 'y'); k=2*M+1
    w=np.zeros((1+2*(2*M),1,k,k),np.float32)
    def setv(row,j,v):
        if axis==0:w[row,0,M,j]+=v
        else:w[row,0,j,M]+=v
    setv(0,0,1)
    for j in range(2*M):
        setv(1+j,j+1,1); setv(1+j,j,-1); w[1+2*M+j]=-w[1+j]
    raw=conv(nm+'_finite_differences',src,w,padding=M); a=relu(nm+'_signed_parts',raw)
    # relu(d)-relu(d-gate), gate=clipped shifted centroid, from 0 to 1.
    w=np.zeros((2*(2*M),2)); w[:,axis]=.5
    pos=np.arange(-M,M); gate=dense(nm+'_intervals',cent,w,np.r_[-pos,-pos-1]); gate=relu(nm+'_fraction_knots',gate)
    # Use a sign-symmetric saturated difference for the fractional step:
    # integer steps are exact, and the approximation error stays local to
    # the two pixels on either side of the selected subpixel position.
    m=np.zeros((1+4*M,4*M))
    for j in range(2*M):
        for channel in [1+j,1+2*M+j]:m[channel,j]=-1;m[channel,j+2*M]=1
    bm=np.repeat(m,full*full,axis=0); gate=reshape(nm+'_gates',dense(nm+'_spatial_bias',gate,bm),[len(m),full,full]); cut=relu(nm+'_unused',op(nm+'_gated','add',[a,gate]))
    coeff=np.r_[0,np.ones(2*M),-np.ones(2*M)]; left=proj(nm+'_base',a,[coeff+np.r_[1,np.zeros(4*M)]]); right=proj(nm+'_increments',cut,[-coeff])
    return op(nm,'add',[left,right])
# Interpolate centroid motion; the two-stage translation captures much of the
# lopsided/offset distinction without searching over centres in template space.
field=translate(translate(field,0),1)
field=crop(field,full,N,1,'aligned')
# Light maps: suppress sky and distant sources. Use a mild threshold and a
# second one at a bright isophote to distinguish a compact bulge from a disc.
y,x=(np.indices((N,N))+.5)*2-N
r=np.hypot(x,y); theta=np.arctan2(y,x)
# Radial sky estimate in wide annulus, with upper-tail clipping to mute stars.
y4,x4=(np.indices((32,32))+.5)*4-64
r4=np.hypot(x4,y4)
raw4=proj('wide_intensity',p4,[phot]); overflow=relu('sky_excess',proj('sky_upper_cut',raw4,[[1]],-.135)); low=proj('sky_negative_excess',overflow,[[-1.]])
clipped=op('sky_winsorized','add',[low,raw4]); ann=(r4>44)&(r4<61); ann=ann/ann.sum()
sky=dense('background',clipped,[ann]); skyplane=reshape('sky_plane',dense('broadcast_background',sky,-np.ones((N*N,1))),[1,N,N]); flux=op('sky_subtracted','add',[field,skyplane]); iso=relu('isophotes',proj('light_thresholds',flux,[[1],[1],[1],[1]],[-.035,-.11,-.24,-.42]))

# Record linear measurements by source, pack them only at the end. Scalar
# readouts implement unit-independent ratios, magnitudes, clipping and rule
# composition, using only permitted linear, pooling, norm and ReLU blocks.
banks={}; expr=[]
class V:
    def __init__(self,i):self.i=i
    def __add__(self,a):return lin([self,a],[1,1])
    __radd__=__add__
    def __neg__(self):return lin([self],[-1])
    def __sub__(self,a):return self+(-a)
    def __rsub__(self,a):return a+(-self)
    def __mul__(self,a):return lin([self],[a])
    __rmul__=__mul__
    def __truediv__(self,a):return self*(1/a)
def var(kind,args=None,k=0):
    v=V(len(expr));expr.append((kind,args if args is not None else [],k));return v
def lin(items,coeff):
    bias=0; entries=[]
    for a,b in zip(items,coeff):
        if isinstance(a,V):entries.append((a.i,b))
        else:bias+=a*b
    return var('linear',entries,bias)
def pos(a):return var('relu',[(a.i,1)])
def ab(a):return pos(a)+pos(-a)
def mx(a,b):return b+pos(a-b)
def mn(a,b):return a-pos(a-b)
def clip(a,l=0,h=1):return l+pos(a-l)-pos(a-h)
def ramp(a,l,h):return clip((a-l)/(h-l))
def spline(a,xx,yy):
    slopes=np.diff(yy)/np.diff(xx); co=np.diff(np.r_[0,slopes,0]); return lin([pos(a-z) for z in xx],co)+yy[0]
def ratio(a,b,scale=1.):
    # compiler batches all ratios at each depth
    return var('ratio',[(a.i,1),(b.i,1)],scale)
def vecmag(a,b):
    # 8 equally spaced projections give a close Euclidean norm via L1.
    return sum(ab(a*math.cos(t)+b*math.sin(t)) for t in np.arange(6)*np.pi/6)*(np.pi/12)
def read(src,mask):
    v=var('input');banks.setdefault(src,[]).append((v.i,np.asarray(mask,'f').reshape(-1)));return v

def iso_read(mask,ch=0):
    w=np.zeros((4,N,N));w[ch]=mask;return read(iso,w)
def gauss(s):return np.exp(-r**2/(2*s*s))
def polar(s,a,b,order=3):return (r/s)**order*np.exp(-.5*order*(r/s)**2)*((np.cos(a*theta)) if b==0 else np.sin(a*theta))
def unit(m):return m/m.sum()
# Isophotal and angular shape. The second angular harmonic of a radial
# elliptical profile is (1-q)/(1+q), for minor/major axis ratio q. Finite
# apertures and seeing compress it, so outer rings and multiple levels help.
def ring2(s,ch):
    m=gauss(s)*(1-np.exp(-r**2/(2*3**2))); mass=iso_read(m,ch); u=iso_read(m*np.cos(2*theta),ch); v=iso_read(m*np.sin(2*theta),ch)
    return ratio(vecmag(u,v),mass+.045,.35)
ein=ring2(9,1); emid=ring2(16,0); eout=ring2(25,1); E=.22*ein+.42*emid+.36*eout
# Size: weighted mean radius inside Gaussian apertures, in pixels. Broad
# isophotal floor excludes the sea of sky pixels in the outskirts.
m=gauss(18); sz=ratio(iso_read(r*m),iso_read(m)+.03,10.)
m=gauss(26); szhi=ratio(iso_read(r*m,1),iso_read(m,1)+.03,10.)
size=.70*sz+.30*szhi
# Channel differences in core, disc and outer galaxy, weighted gently and
# background subtracted in each band. Colors are divided by aperture light.
def col(s,inner):
    m=np.exp(-r4**2/(2*s*s)); m*=1-np.exp(-r4**2/(2*inner**2)) if inner else 1
    m=unit(m)-ann
    lum=read(p4,phot[:,None,None]*m); rb=read(p4,chrom[:,None,None]*m)
    return ratio(rb,lum+.022,.45)
corecol=col(4,0); diskcol=col(10,4); outercol=col(17,6)
red=.49*corecol+.36*diskcol+.15*outercol
blue=ramp(-red,-.08,.22); old=ramp(red,.02,.30)
# Compactness as radial decline relative to the disc; a centrally peaky
# edge-on galaxy, particularly with a redder core, suggests a bulge.
core=read(flux,unit(gauss(2.3))); shoulder=read(flux,unit(gauss(8)*(1-gauss(3.5))))
compact=ratio(core-shoulder,core+.035,.6)
# Scale corrected by measuring shape along oriented major/minor axes. Max
# over all candidate major axes of the outer light; excess thickness at
# the centre relative to wings detects an edge-on bulge.
angles=np.arange(12)*np.pi/12
aniso=[]; thick=[]; midwidth=[]; disklen=[]; barcontrast=[]
for t in angles:
    X=x*np.cos(t)+y*np.sin(t); Y=y*np.cos(t)-x*np.sin(t)
    wing=np.exp(-X**2/(2*22**2))*(1-np.exp(-X**2/(2*7**2)))*np.exp(-Y**2/(2*6**2)); centre=np.exp(-X**2/(2*4**2)-Y**2/(2*7**2))
    def width(m):return ratio(iso_read(m*abs(Y),0),iso_read(m,0)+.045,3.)
    thick.append(width(centre)-width(wing)); midwidth.append(width(wing)); aniso.append(iso_read(wing))
# Soft max selection via closeness to max aperture light, without any learnt
# orientation or multiplication by data-dependent transform parameters.
def vmax(vals):
    while len(vals)>1: vals=[mx(vals[i],vals[i+1]) if i+1<len(vals) else vals[i] for i in range(0,len(vals),2)]
    return vals[0]
peakori=vmax(aniso)
bulge=vmax([a-1.8*(peakori-b) for a,b in zip(thick,aniso)])
# Deviation from 180-degree symmetry. Clip each pair at a radius-dependent
# surface brightness to give a star at large radius only bounded influence.
# The reversed flatten order is exactly the rotation by pi.
unitmat=np.eye(N*N,dtype='f'); diff=unitmat-unitmat[::-1]
sm=pool('symmetric_bin',field,2); small=N//2; P=small*small; dr=np.eye(P,dtype='f');dr-=dr[::-1].copy()
resid=relu('odd_half',dense('odd_light',sm,dr)); rad=np.hypot(*((np.indices((small,small))+.5)*4-N)); odd=[]
for s,bg in [(7,.012),(16,.014)]:
    g=np.exp(-rad**2/(2*s*s)); g1=g/g.sum(); a=read(resid,g1)*2; denom=read(flux,unit(gauss(s)))
    odd.append(ratio(pos(a-bg),denom+.03,.3))
asym=.44*odd[0]+.56*odd[1]
# Second nucleus: scale-space contrast and local peak prominence. Requiring
# decreases in every radial direction suppresses the bright flank of a single
# smooth galaxy. Pool over distance to the primary, after removing sky-level
# responses. Compare excess to the central surface brightness.
def dirfilter(nm,step,sg):
    tt=np.arange(8)*np.pi/4; K=int(np.ceil(step+2.3*sg)); yy,xx=np.indices((2*K+1,2*K+1))-K
    def g(a,b):return unit(np.exp(-((xx-a)**2+(yy-b)**2)/(2*sg*sg)))
    w=[g(0,0)-g(step*np.cos(t),step*np.sin(t)) for t in tt]; l=conv(nm+'_directions',field,-np.asarray(w)[:,None,:,:],padding=K)
    # Reduce across channels by treating all pixels like a row.
    a=reshape(nm+'_stack',l,[1,8,N*N]); a=op(nm+'_worst','maxpool',a,size=[8,1],stride=[8,1]);return reshape(nm+'_map',a,[1,N,N])
peaks=[]
for st,sg in [(2.5,.85),(4.,1.35)]:
    valley=dirfilter('nucleus_'+str(len(peaks)),st,sg); peak=proj('peak_sign_'+str(len(peaks)),valley,[[-1.]])
    # Permit bright nuclei further away, but demand progressively greater
    # angular extent farther out to reject unrelated field stars.
    penalty=np.maximum(0,(r-14)/65)+np.maximum(0,(7-r)/6)
    mask=dense('nucleus_radial_penalty_'+str(len(peaks)),sky,np.zeros((N*N,1)),-penalty.ravel()); mask=reshape('nucleus_mask_'+str(len(peaks)),mask,[1,N,N]); mapp=op('companion_candidates_'+str(len(peaks)),'add',[peak,mask]); top=pool('companion_peak_'+str(len(peaks)),mapp,N,'maxpool'); peaks.append(read(top,[1]))
comp=ratio(.4*peaks[0]+.6*peaks[1],core+.09,.35)
# Contour shape and texture after approximate deprojection: polar samples on
# several hypotheses for orientation and axis ratio. A Fourier transform in
# azimuth measures both amplitude and change of position angle with radius.
# Select the frame in which the light is most nearly circular (smallest even
# ellipticity), using broad overlapping isophotes. Higher harmonics are
# invariant to sky rotation; m=2 changing phase with log radius indicates
# arms, whereas a coherent m=2 confined to the inner disc indicates a bar.
# Fixed short stencils, tabulated as a dense layer on the central luminance.
# Each row is one analytic quadrature in projected elliptical coordinates.
rr=np.array([3.1,4.5,6.4,9.1,12.9,18.3,25.9]); orders=[0,1,2,3,4]; hyp=[(1.,0.)]+[(q,t) for q in [.68,.45] for t in np.arange(8)*np.pi/8]; H=len(hyp); R=len(rr); F=9
rows=[]
# At each ellipse accumulate periodic, equally spaced bilinear samples.
# Small radial blur across each ring helps suppress shot noise and JPEG grain.
az=np.arange(64)*2*np.pi/64
fourier=np.stack([np.ones_like(az)]+[f(k*az) for k in orders[1:] for f in (np.cos,np.sin)])/len(az)
for q,t in hyp:
    for radius in rr:
        w=np.zeros((F,N,N));
        for drho,p in [(-.13,.25),(0,.5),(.13,.25)]:
            X=radius*(1+drho)*np.cos(az); Y=radius*(1+drho)*q*np.sin(az); xx=(X*np.cos(t)-Y*np.sin(t)+N)/2-.5; yy=(X*np.sin(t)+Y*np.cos(t)+N)/2-.5
            i=np.floor(xx).astype(int); j=np.floor(yy).astype(int); sx=xx-i; sy=yy-j
            for dx,wx in [(0,1-sx),(1,sx)]:
                for dy,wy in [(0,1-sy),(1,sy)]:
                    for fi in range(F):np.add.at(w[fi],(j+dy,i+dx),p*fourier[fi]*wx*wy)
        rows.extend(w)
coeff=reshape('ring_fourier_cube',dense('ring_fourier',flux,rows),[H*R,F,1]);
# Reshape per-frame feature arrays as sites, via compact batched linear maps.
# Input order H x R x F -> W processes R*F for each of H positions. Since the
# API does not have transpose, do the reordering once with the mask bank.
prev=weights['ring_fourier.W'];weights['ring_fourier.W']=prev.reshape(H,R,F,-1).transpose(1,2,0,3).reshape(H*R*F,-1).copy(); coeff=reshape('harmonics',coeff,[R*F,1,H]);
# Linear forms on each ring/hypothesis, including first/second angular mode
# in the inner and outer ranges, a second radial difference of m=2 and arm
# coherence across neighbouring log radii. Absolute magnitude by projections.
forms=[]; form_names={}
def form(key,w):
    if key not in form_names:form_names[key]=len(forms);forms.append(w.flatten())
    return form_names[key]
def row(j,k=0):
    a=np.zeros((R,F));a[j,k]=1;return a
def cmode(j,k):return np.stack([row(j,2*k-1),row(j,2*k)])
# Scores are normalised by the azimuthal mean minus sky with a noise floor.
# Use a collection of radius windows instead of committing to an object size.
polar_defs=[]
for j in range(R-1):
    for k in (1,2,3,4):
        polar_defs.append(('m%d_%d'%(k,j),cmode(j,k),row(j),.025))
for j in range(1,R-1):
    polar_defs.append(('curve%d'%j,cmode(j,2)-.5*(cmode(j-1,2)+cmode(j+1,2)),row(j),.035))
    # Difference between ring harmonic and a local radial gradient prediction:
    # in a mildly elliptical smooth profile, the spurious m2 scales like the
    # slope dI/dlog(r). Curvature / cross-radius sign reversal rejects it.
    polar_defs.append(('twist%d'%j,cmode(j-1,2)-cmode(j+1,2),(.55*row(j-1)+.45*row(j+1)),.045))
# Broad m2 and azimuthal photometry for selection of deprojection hypothesis.
for group,js in [('shape',[1,2,3,4]),('outer',[3,4,5]),('bar',[0,1,2]),('disc',[2,3,4])]:
    z=sum(cmode(j,2) for j in js)/len(js); bright=sum(row(j) for j in js)/len(js); polar_defs.append((group,z,bright,.025))
# Batched approximate magnitude. Always include both signs; representation
# here is in surface brightness units, not synthesized feature exemplars.
for key,z,_,_ in polar_defs:
    for j,t in enumerate(np.arange(8)*np.pi/4):form((key,j),np.cos(t)*z[0]+np.sin(t)*z[1])
pr=relu('harmonic_halfwaves',proj('harmonic_projections',coeff,forms)); Mfeat=len(polar_defs)
w=np.zeros((Mfeat,len(forms)))
for i,(key,*_) in enumerate(polar_defs):
    for j in range(8): w[i,form_names[key,j]]=np.pi/8
am=proj('harmonic_magnitudes',pr,w); den=proj('ring_brightness',coeff,[d.flatten() for _,_,d,_ in polar_defs],[b for _,_,_,b in polar_defs]); den=relu('ring_light_floor',den)
# Normalise numerator/denominator per hypothesis and feature. Store as
# four signed lanes, using a compact channel interleaver (number of lanes
# is small enough to avoid a large dense photometric calculation).
L=Mfeat*H; am=reshape('amplitude_vector',am,[L]); den=reshape('surface_vector',den,[L]); ids=np.arange(L); w=np.zeros((L*4,L),np.float32);w[4*ids,ids]=5;w[4*ids+1,ids]=-5
u=dense('amplitude_lanes',am,w); w=np.zeros_like(w); w[4*ids+2,ids]=1;w[4*ids+3,ids]=-1
v=dense('surface_lanes',den,w); uv=op('ellipse_pair_sum','add',[u,v]); uv=reshape('ellipse_ratio_groups',uv,[L,1,4]); uv=op('ellipse_ratio_norm','norm',uv)
# Here sin(atan(5*m)) is a bounded contrast statistic in itself. Express
# it on a familiar fractional modulation scale by a short spline.
take=np.zeros((L,4*L),np.float32);take[ids,4*ids]=1; u=dense('ellipse_modulation',uv,take);u=reshape('modulations_by_position',u,[1,Mfeat,H]); knots=np.array([0,.08,.16,.26,.4,.65,1.]); knots_x=np.sqrt(2)*5*knots/np.sqrt(1+(5*knots)**2); sl=np.diff(knots)/np.diff(knots_x); coeffs=np.diff(np.r_[0,sl,0]); a=relu('modulation_hinges',proj('modulation_breaks',u,np.ones((len(knots),1)),-knots_x)); a=proj('fractional_harmonics',a,[coeffs]); a=reshape('ellipse_descriptors',a,[Mfeat,1,H]); idx={key:i for i,(key,*_) in enumerate(polar_defs)}
# Choose deprojection using the most nearly axisymmetric, radially averaged
# disc, not the orientation producing the strongest spiral signal. Preferences
# for a weak deprojection when an equally good fit is found regularise it.
fvec=np.zeros(Mfeat); fvec[idx['shape']]=.6;fvec[idx['outer']]=.4
fit=proj('deprojection_quality',a,[-fvec]); bias=np.array([.011*np.log(q) for q,t in hyp]); ff=dense('deprojection_prior',fit,np.eye(H),bias);ff=reshape('deprojection_values',ff,[1,1,H]); fm=op('best_deprojection','maxpool',ff,size=[1,H],stride=[1,H]); rep=reshape('best_deprojection_plane',dense('common_optimum',fm,np.ones((H,1))),[1,1,H]); fg=op('deprojection_deficit','add',[ff,proj('negative_optimum',rep,[[-1]])]);
# Shared score penalty for considering a shape far from the best deprojection.
fg=proj('frame_penalties',fg,[[2.8]]*Mfeat); gated=op('penalized_descriptors','add',[a,fg]); cross=op('selected_features','maxpool',gated,size=[1,H],stride=[1,H]);
def desc(key):return read(cross,np.eye(Mfeat)[idx[key]])
def mean(xs):return sum(xs)/len(xs)
def group(prefix,js):return mean([desc(prefix+str(j)) for j in js])
# Choose the annular range as a function of the galaxy light radius. Inner
# ring m=2 is the bar channel; outer m=2 alone need not indicate spirality.
# The high odd/even modes and radial highpass m2 indicate structure which a
# smooth elliptical isophote cannot produce in its preferred frame.
texture=group('m3_',[2,3,4])*.43+group('m4_',[2,3,4])*.34+group('curve',[2,3,4])*.23
outertex=group('m3_',[3,4,5])*.42+group('m4_',[3,4,5])*.28+group('curve',[3,4,5])*.30
arm=group('twist',[2,3,4]); innerbar=mx(desc('bar'),desc('disc')); bar=innerbar+.35*pos(desc('m2_2')-desc('outer'))
# Bar/disc comparison from the inner aperture vs outer in the optimal OUTER
# disc frame, not the best mean frame (which can rotate toward the bar).
fvec*=0; fvec[idx['outer']]=-1
of=proj('outer_disc_circularity',a,[fvec]); og=op('outer_disc_optimum','maxpool',of,size=[1,H],stride=[1,H]); og=reshape('outer_optimum_planes',dense('negative_outer_optimum',og,-np.ones((H,1))),[1,1,H]); delta=op('outer_disc_delta','add',[of,og]); vbar=np.zeros(Mfeat);vbar[idx['bar']]=.55;vbar[idx['disc']]=.45
ba=op('bar_in_disc','add',[proj('inner_bar_mean',a,[vbar]),proj('outer_disc_penalty',delta,[[2.]])]); ba=op('deprojected_bar','maxpool',ba,size=[1,H],stride=[1,H]); bar=.55*bar+.45*read(ba,[1]);
# Smooth disc versus bulge radial profile. If the central light is dominated
# by a compact bulge, the logarithmic radial slope steepens towards the
# centre relative to a disc. Also measure the size of the bright plateau.
concen=ratio(iso_read(gauss(18),3),iso_read(gauss(18),1)+.04,.25)
# Matched analytic surface-brightness profiles. Each row is one model of a
# sky-subtracted galaxy, windowed and normalized solely by its own norm.
# An n~2--4 spheroid with seeing blur is centrally concentrated; exponential
# discs with either a small nucleus or central depletion are much flatter.
# For edge-ons model a separable radial and vertical disc PLUS a round bulge.
# The discriminant is the template-family correlation difference, not the
# best absolute similarity. All continuous parameters are nuisance variables
# sampled on a fixed physical grid. No library of real images is involved.
prof_win=np.exp(-.5*(r/18.)**2); colours=[]
model_angles=np.arange(20)*np.pi/20
axis_grid=[.13,.22,.33,.46,.62,.80,1.]; radii=[2.6,3.6,5.,7.,10.,14.,19.]
def modbank(name,builder):
    filt=[]
    for rad in radii:
        for q in axis_grid:
            for t in model_angles:
                X=(x*np.cos(t)+y*np.sin(t))/rad; Y=(-x*np.sin(t)+y*np.cos(t))/rad
                P=builder(X,Y,q,rad); P=P*prof_win/(np.sqrt((P*P*prof_win).sum())+1e-15)
                filt.append(P.astype('f'))
    a=dense('profile_'+name,flux,filt); a=reshape('profile_grid_'+name,a,[1,len(radii)*len(axis_grid),len(model_angles)])
    a=op('profile_angle_'+name,'maxpool',a,size=[1,len(model_angles)],stride=[1,len(model_angles)]); a=reshape('profile_scale_'+name,a,[1,len(radii),len(axis_grid)])
    a=op('profile_q_'+name,'maxpool',a,size=[len(radii),1],stride=[len(radii),1]); top=op('profile_best_'+name,'maxpool',a,size=[1,len(axis_grid)],stride=[1,len(axis_grid)]);
    return read(top,[1]),[read(a,m) for m in np.eye(len(axis_grid))]
def er(X,Y,q,a): return np.sqrt(X**2+Y**2/(q*q+(1.15/a)**2)+(1.15/a)**2)
def smoothmod(X,Y,q,a):return np.exp(-2.7*(er(X,Y,q,a)+.05)**.52)
def expmod(X,Y,q,a):return np.exp(-1.1*er(X,Y,q,a))
def flatmod(X,Y,q,a):return np.exp(-.68*er(X,Y,q,a)**1.5)
def discmod(X,Y,q,a):return .74*np.exp(-.70*er(X,Y,q,a)**1.4)+.42*np.exp(-.5*(X*X+Y*Y)/(.21**2+(1.25/a)**2))
def edgeprofile(X,Y,q,a): return np.exp(-np.sqrt(X*X+.12))*np.exp(-.5*Y*Y/(.65*q**2+(1.1/a)**2))
def bulgeprofile(X,Y,q,a):return .75*edgeprofile(X,Y,q,a)+.80*np.exp(-2.9*np.sqrt(X*X+Y*Y/.72**2+(1.15/a)**2))
S,sq=modbank('spheroid',smoothmod); Xq,xq=modbank('exponential',expmod); Fq,fq=modbank('flat',flatmod); Dq,dq=modbank('disknucleus',discmod); T,tq=modbank('thin',edgeprofile); B,bq=modbank('inclinedbulge',bulgeprofile)
sph=mx(S,Xq); dm=mx(Dq,Fq); allmod=vmax([S,Xq,Fq,Dq,T,B]); profile_disc=ratio(dm-S,allmod+.05,.025); flatdisc=ratio(dm-Xq,allmod+.05,.025)
# Compare a very inclined model with the best single elliptical isophote;
# a bulge+thin outer disc does not have homologous inner/outer ellipses.
slender=vmax(tq[:3]+bq[:3]); core_bulge=ratio(vmax(bq[:3])-vmax(tq[:3]),allmod+.05,.025)
edge_adv=ratio(slender-sph,allmod+.05,.030); eopp=vmax([u for group in (sq,xq,fq,dq) for u in group[4:]]); thinprofile=ratio(slender-eopp,allmod+.05,.05)
# Geometry in the low-surface-brightness wings. The high central isophote
# mixes the bulge and disc; exclude it when measuring the outer flattening.
wingmask=gauss(18)*(1-gauss(6)); wmass=iso_read(wingmask); we=ratio(vecmag(iso_read(wingmask*np.cos(2*theta)),iso_read(wingmask*np.sin(2*theta))),wmass+.12,.55)
# Light-area radius: unlike a first moment, this scale is not shortened by
# a bright central bulge. Use a bounded indicator of detectably bright light.
area=iso_read(gauss(20),0)-iso_read(gauss(20),1); area=area*4/(.11-.035) # area in native square pixels
# Companion photometry on a wider field. Require a true secondary local
# maximum with resolved surrounding light, not just a bright neighbour.
# Place the peak and envelope tests on the SAME candidate before max pooling.
Nw=32; at=np.arange(Nw)*4+2-64; YY,XX=np.meshgrid(at,at,indexing='ij'); rw=np.hypot(YY,XX)
yk,xk=np.indices((11,11))-5; rrk=np.hypot(yk,xk)
def kg(x,y,s):return unit(np.exp(-((xk-x)**2+(yk-y)**2)/(2*s*s)))
inner=kg(0,0,.65); middle=kg(0,0,1.45); back=kg(0,0,3.5)
wide_tests=[inner-kg(2.3*np.cos(t),2.3*np.sin(t),.8) for t in np.arange(8)*np.pi/4]
# minimum of directional peak prominence and of two contrast measures.
wtest=np.stack(wide_tests+[1.3*(middle-back),2*(unit((rrk>1.15)*(rrk<2.75))-back)])
wr=conv('wide_peak_tests',raw4,-wtest[:,None,:,:],padding=5);wr=reshape('wide_peak_stack',wr,[1,len(wtest),Nw*Nw]);wr=op('wide_peak_weakest','maxpool',wr,size=[len(wtest),1],stride=[len(wtest),1]);wr=reshape('wide_peak_field',wr,[1,Nw,Nw]);
# Exact annular penalties are small for a close pair; beyond that the prior
# odds of chance alignment increase in proportion to available sky area.
radpen=np.maximum(0,7.5-rw)*.12+np.maximum(0,rw-16)*.0018+np.maximum(0,rw-33)*.006
wr=dense('peak_chance_alignment',wr,-np.eye(Nw*Nw),-radpen.ravel());wr=reshape('resolved_peaks',wr,[1,Nw,Nw]);wr=pool('bright_companion',wr,Nw,'maxpool'); widepeak=read(wr,[1]); widecomp=ratio(widepeak,core+.055,.15)
# Manual fuzzy decision rules. Scores model evidence for a hierarchy:
# single vs paired/lopsided; smooth vs featured; face-on vs edge-on; subtypes.
# Each group's winning subtype gets the group's score, other subtypes a
# margin penalty. These are fixed dimensionless rules, not a learned readout.
def logor(vals):return vmax(vals)
spd=ramp(size,5.5,13.5); tiny=ramp(-size,-7.8,-4.2); elongated=ramp(E,.32,.50)
structure=.45*ramp(texture,.028,.105)+.22*ramp(outertex,.03,.13)+.33*ramp(arm,.045,.18)
face=1-ramp(E,.29,.53)
spiral=-.71+1.17*spd+.79*structure+.43*blue+.23*ramp(bar,.07,.19)-.33*old-.8*elongated-.40*tiny
# Penalise extremely round tiny sources lacking any visible disc, even if
# they are blue. A large round disc with a smooth inner light profile can
# still be a tight spiral if it is blue and extended.
spiral-=.30*tiny*(1.)
smooth=.53+.23*tiny+.13*old-.36*structure-.24*spd-.17*blue
# A long thin disc (with considerably larger projected extent than a smooth
# cigar) strongly favours edge-on even for a red central bulge.
edge=-.48+1.32*ramp(E,.38,.64)+.71*ramp(size,6.8,14.5)+.18*blue-.50*ramp(-E,-.44,-.29)
# When light is exceptionally needle-like, size matters less.
edge+=.20*ramp(E,.61,.74)
# Merger needs another nucleus or strong, coherent inner two-sided substructure.
# Independently gate the secondary-peak evidence against minuscule sources and
# random unrelated companions in the annulus.
merger=-.63+1.43*ramp(comp,.10,.51)+.66*ramp(asym,.11,.58)+.28*ramp(odd[0],.15,.60)+.1*spd
# A large displacement of the light peak from the supplied target often
# indicates a tidal interaction or highly irregular object.
cenx=read(cent,[1,0]); ceny=read(cent,[0,1]); shift=vecmag(cenx,ceny)
disturbed=-.56+1.37*ramp(asym,.145,.54)+.54*ramp(odd[0],.12,.48)+.25*ramp(shift,2.4,6.5)-.33*ramp(comp,.22,.50)-.26*spd
# Soft subdivision of apparent axis ratio. Illustrative boundaries near q=.8
# and q=.5, allowing for aperture attenuation and atmospheric seeing.
s2=-pos(E-.103)*8;s3=-pos(.103-E)*8-pos(E-.318)*7;s4=-pos(.318-E)*7
# Large bulged edge-ons have a central broadening, central concentration,
# and an older, redder stellar population than the disc without a bulge.
bulge_ev=.34*ramp(bulge,-.10,1.05)+.27*ramp(red,-.05,.28)+.14*ramp(corecol-diskcol,.02,.32)+.15*ramp(compact,.45,.76)+.10*ramp(concen,.095,.34)
bul=2.1*(bulge_ev-.52)
# Margins between mechanistic models are often small: a correlation excess
# of .015 means a three-percent difference in explained light power. Treat
# differences in shape as evidence, not as direct ten-class predictions.
foot=ramp(area,230.,720.); broad=ramp(area,440.,1120.); smallarea=1-foot
pdisk=ramp(profile_disc,-.010,.045); plateau=ramp(flatdisc,-.008,.024)
elliptical=ramp(-profile_disc,-.005,.046)
spiral+=.72*foot+.27*broad+.47*(pdisk-.45)+.28*(plateau-.5)+.14*face
smooth+=.15*elliptical-.18*foot
outerthin=ramp(we,.46,.73); twocomp=ramp(edge_adv,-.011,.032); longthin=ramp(thinprofile,-.018,.055)
edge+=.52*outerthin+.43*twocomp+.29*longthin-.31+.17*broad
# The geometric bulge channel includes the relative evidence for a central
# spheroid on top of an edge-on disc. Color is useful but not obligatory.
bul=.80*bul+.80*(ramp(core_bulge,-.003,.021)-.40)+.32*(ramp(we-E,.045,.20)-.45)
merger=mx(merger,-.55+1.65*ramp(widecomp,.035,.19)+.42*ramp(asym,.09,.44)+.27*ramp(comp,.1,.4))
# Odd outer isophotes without a supported secondary core indicate tidal tails.
disturbed+=.32*ramp(odd[1]-odd[0],.04,.29)+.15*(1-elliptical)-.14*foot
# Inner quadrupole in deprojected frame signals a bar. Loose arms have high
# azimuthal modulation and change the two-fold pattern substantially across
# radii; tight arms resemble a smooth disc under coarse averaging.
bars=1.50*ramp(bar,.063,.225)+.20*ramp(arm,.08,.23)-.90-.26*ramp(outertex,.08,.18)
loose=1.08*ramp(outertex,.035,.14)+.48*ramp(arm,.07,.23)+.22*blue-.91
# A bar is a narrow, fairly straight inner component superposed on an
# extended face-on or inclined disc. Compare one- vs two-component shapes
# directly, marginalising over the bar and disc position angles. A redder
# centre than outer disc is supportive but never sufficient on its own.
barweights=[]
for radius in [4.,6.,9.,13.,18.]:
    for inc in [1.,.68,.45]:
        for alph in np.arange(12)*np.pi/12:
            X=(x*np.cos(alph)+y*np.sin(alph))/radius; Y=(-x*np.sin(alph)+y*np.cos(alph))/radius
            disk=np.exp(-.80*er(X,Y,inc,radius)**1.25)
            for rel in ([0.] if inc==1. else [0.,np.pi/4,np.pi/2,-np.pi/4]):
                u=X*np.cos(rel)+(Y/inc)*np.sin(rel); v=-X*np.sin(rel)+(Y/inc)*np.cos(rel)
                barbody=np.exp(-.5*((u/.87)**2+v*v/(.22**2+(1.2/radius)**2)))
                temp=.58*disk+.90*barbody+.15*np.exp(-.5*(X*X+Y*Y)/(.18**2+(1.1/radius)**2))
                barweights.append((temp*prof_win/np.sqrt(np.sum(temp*temp*prof_win))).astype('f'))
ba=dense('straight_bar_superposed',flux,barweights); ba=reshape('superposed_bar_views',ba,[1,1,len(barweights)]);ba=op('best_superposed_bar','maxpool',ba,size=[1,len(barweights)],stride=[1,len(barweights)]); barcorr=ratio(read(ba,[1])-mx(sph,dm),allmod+.05,.02)
# An explicit spiral-pitch transform in elliptical polar coordinates. Sum
# complex m=2 coefficients after undoing the rotation of a logarithmic arm
# at candidate pitch angles. Match the changing phase only (subtract the
# radial DC component), so uniformly elongated light is not a spiral. A
# coherent inner component can instead be a bar if it is stronger than the
# winding modes. The aperture amplitude uses the corresponding disc flux.
armforms=[]; armdens=[]; pitchset=[13.,21.,33.,48.,80.]; windows=[(1,2,3,4),(2,3,4,5),(3,4,5,6)]
for pitch in pitchset:
    for js in windows:
        denm=sum(row(j) for j in js)/len(js)
        for chir in [-1,1]:
            plan=np.exp(2j*chir*np.log(rr[list(js)]/rr[js[0]])/np.tan(np.deg2rad(pitch)))
            # This is the zero-DC part of a complex sinusoidal filter along
            # log r, not a statistic over multiple generated images.
            plan=plan-plan.sum()/len(plan); ng=np.sqrt(np.sum(abs(plan)**2)*len(js))+.00001;plan/=ng
            h1=np.zeros((R,F));h2=np.zeros_like(h1)
            for j,z in zip(js,plan):
                real,imag=cmode(j,2);h1+=z.real*real+z.imag*imag;h2+=z.real*imag-z.imag*real
            armforms+= [(h1*np.cos(t)+h2*np.sin(t)).flatten() for t in np.arange(8)*np.pi/4]; armdens.append(denm.flatten())
pr=relu('pitch_response_phases',proj('pitch_transform',coeff,armforms)); count=len(armdens); w=np.zeros((count,len(armforms)))
for i in range(count):w[i,i*8:i*8+8]=np.pi/8
pr=proj('pitch_amplitude',pr,w); de=proj('pitch_surface',coeff,armdens,.047)
# Compare filtered amplitude with several fractions of local mean light.
# Positive excess above the smaller threshold is a soft S/N-weighted arm
# modulation; the larger threshold only responds to conspicuous arms.
w=np.eye(count); exc=relu('arm_response',op('arm_contrast_threshold','add',[pr,proj('arm_continuum',de,-.028*w)]));
# For fixed surface contrast, an outer annulus consistent with sky should
# not vote. Max over handedness, radii and regularised deprojection.
fg2=proj('pitch_frame_regularization',of,[[.085]]*count); og2=proj('pitch_frame_reference',og,[[.085]]*count);exc=op('deprojected_arm_response','add',[exc,fg2,og2]); exc=reshape('pitch_groups',exc,[len(pitchset),1,len(windows)*2*H]);exc=op('pitch_peaks','maxpool',exc,size=[1,len(windows)*2*H],stride=[1,len(windows)*2*H]); pitchvals=[read(exc,ww) for ww in np.eye(len(pitchset))]; tight=pitchvals[0]; openarm=vmax(pitchvals[1:4]); anyarm=vmax(pitchvals[:4]);
# Quantify departure from light stratified on homologous ellipses. Compute
# robust, regularized annular RMS-like deviations across angular modes, not
# the residual relative to a raw profile library. Use ABSOLUTE surface
# contrast first so that extrapolating an almost invisible ring cannot drive
# this decision. Surface brightness is in fractions of the full JPEG range.
vfeat=np.zeros(Mfeat)
for j,f in [(2,.3),(3,.42),(4,.28)]:
    for k,c in [(1,.20),(3,.31),(4,.20)]:vfeat[idx['m%d_%d'%(k,j)]]+=c*f
    vfeat[idx['curve%d'%j]]+=.29*f
formal=proj('nonelliptical_residual','ellipse_descriptors',[-vfeat]);formal=op('smallest_contour_error','maxpool',formal,size=[1,H],stride=[1,H]); irreg=-read(formal,[1]);
# Sharp off-centre nuclei: require local prominence at the candidate plus
# galaxy surface light in its surroundings. Use the actual second strongest
# spatially distinct centre anywhere inside the aperture (including systems
# centered BETWEEN images). The maxima of opposite half-planes force peaks
# to lie on different sides along at least one direction. A soft dead zone
# around each cutting line eliminates adjacent pixels from the same peak.
pp='peak_sign_0'
# Retain a peak map defined on the aligned high-resolution field.
second=[]
for t in np.arange(6)*np.pi/6:
    X=x*np.cos(t)+y*np.sin(t)
    for off in [-7.,0.,7.]:
        b1=np.maximum(0,2.-X+off)*.075+np.maximum(0,r-18.)*.003
        b2=np.maximum(0,2.+X-off)*.075+np.maximum(0,r-18.)*.003
        second.extend([-b1.ravel(),-b2.ravel()])
# each dense numerator used just to copy with a spatial bias, so tile channel
# maps with 1x1 weights then inject biases from a scalar constant (zero W).
bs=np.asarray(second); nch=len(bs); pa=proj('separated_peak_maps',pp,[[1]]*nch); bias=dense('separating_lines',sky,np.zeros((bs.size,1)),bs.ravel()); pa=op('split_peaks','add',[pa,reshape('separating_planes',bias,[nch,N,N])]); pa=pool('half_plane_peaks',pa,N,'maxpool');pa=dense('negative_peak_pairs',pa,-np.eye(nch));pa=reshape('distinct_pairs',pa,[1,nch//2,2]);pa=op('weakest_of_pair','maxpool',pa,size=[1,2],stride=[1,2]);pa=proj('positive_pairs',pa,[[-1]]);pa=op('best_distinct_pair','maxpool',pa,size=[nch//2,1],stride=[nch//2,1]); twin=read(pa,[1]); twinfrac=ratio(twin,core+.065,.17)
# The hard branches use morphology relative to competitors instead of an
# unconditional disk bias: every smooth blue sphere/ellipse has a reasonable
# exponential fit. A disc-like fit on its own is not sufficient for spirals.
spiral-=.36*foot+.15+.21*pdisk
spiral+=.62*ramp(anyarm,.003,.019)+.32*ramp(irreg,.023,.073)
smooth+=.34*ramp(-irreg,-.044,-.015)+.12*elliptical
# Proper assessment of extension along the major axis independent of bulge.
# Major-axis tail beyond a circular aperture, versus light in a round core.
# Unlike weighted mean radius it does not vanish for a small bright bulge.
corearea=iso_read(gauss(5),0)-iso_read(gauss(5),1); disktail=ratio(area*(.11-.035)/4-corearea,corearea+.02,3.); extent=ramp(disktail,1.0,3.1)
bulgededge=.34+.83*ramp(we,.36,.62)+.44*extent+.46*ramp(core_bulge,-.004,.022)+.22*twocomp-.32*ramp(-E,-.44,-.26)
edge=mx(edge,bulgededge-.53); spiral-=.52*ramp(we,.40,.65); bul+=.44*ramp(core_bulge,-.005,.026)+.23*ramp(we-E,.02,.16)-.10
# Multi-nucleus evidence is decisive even in a smooth-looking common halo.
merger=mx(merger,-.27+1.68*ramp(twinfrac,.048,.245)+.45*ramp(widecomp,.03,.14)+.25*ramp(asym,.09,.37))
# A lopsided core, especially with a displaced photocentre, is distinct
# from anisotropic but concentric, smooth isophotes.
shift0=vecmag(read(cent0,[1,0]),read(cent0,[0,1])); displaced=vecmag(cenx-read(cent0,[1,0]),ceny-read(cent0,[0,1]));
disturbed+=.29*ramp(displaced,.75,3.2)+.38*ramp(odd[1],.17,.58)+.23*ramp(shift0,2.3,6.5)-.18*ramp(twinfrac,.07,.20)
bars+=1.2*(ramp(barcorr,-.010,.018)-.30)+.37*ramp(corecol-diskcol,.015,.29)+.40*ramp(bar,.035,.110)
loose+=.41*ramp(odd[1],.09,.34)+.21*ramp(-outercol,.01,.27)+.67*ramp(openarm,.004,.023)+.30*ramp(openarm-tight,-.001,.008)
# Complement the analytic features with an image-domain lack-of-fit measure.
# Given a unit norm template t, squared weighted reconstruction error after
# matching its brightness is ||I||^2 - <I,t>^2. We need no quadratic layer:
# norm(+/- weighted image, +/- reference) supplies the reciprocal RMS; a
# simpler stable construction uses a shared piecewise linear square spline
# and sqrt approximated on positive scales. All square-spline knots derive
# from evenly spaced intensities, not statistics of sample pixels.
knot=np.linspace(-.28,1.,49); knotslope=np.diff(knot**2)/np.diff(knot); qweights=np.diff(np.r_[0,knotslope,0]); qmap=relu('power_spline_knots',proj('power_cutpoints',flux,[[1]]*len(knot),-knot)); power=proj('light_squared',qmap,[qweights],knot[0]**2);
fluxpow=read(power,prof_win); grid=np.geomspace(.035,240.,75); fluxnorm=spline(fluxpow,grid,np.sqrt(grid)); match=mx(allmod,read(ba,[1])); error=ratio(fluxnorm-match,fluxnorm+.01,.06)
# Excess power in a close central aperture rather than in the field. Resolve
# distinct nuclei from point sources using THREE checks at the same pixel:
# a local maximum, a resolved bright envelope, and a bridge of light towards
# the target. The bridge test uses fixed geometric interpolation matrices.
# This avoids taking evidence from different stars to justify a single pair.
Nw=32; yy0,xx0=np.indices((Nw,Nw));
def interpgrid(xp,yp):
    z=np.zeros((Nw*Nw,Nw*Nw)); i=np.floor(xp).astype(int);j=np.floor(yp).astype(int);sx=xp-i;sy=yp-j
    for dx,wx in [(0,1-sx),(1,sx)]:
        for dy,wy in [(0,1-sy),(1,sy)]:z[np.arange(Nw*Nw),(np.clip(j+dy,0,Nw-1)*Nw+np.clip(i+dx,0,Nw-1)).flatten()]+=(wx*wy).flatten()
    return z
sk=reshape('wide_sky_plane',dense('wide_sky_bias',sky,-np.ones((Nw*Nw,1))),[1,Nw,Nw]); wideflux=op('wide_flux','add',[raw4,sk]);
bl=conv('bridge_seeing',wideflux,kg(0,0,.80)[None,None],padding=5); options=[]
for acoef in [.38,.62,.79]:
    xp=15.5+(xx0-15.5)*acoef;yp=15.5+(yy0-15.5)*acoef
    options.append(reshape('bridge_sample_'+str(len(options)),dense('bridge_kernel_'+str(len(options)),bl,-1.*interpgrid(xp,yp),.013),[1,1,Nw*Nw]))
# A broad source must be stronger in a near annulus than a stellar PSF of
# the same height. Also require a rise in that envelope relative to the sky.
rr1=unit(((rrk>=1.4)&(rrk<2.6)).astype(float));env=conv('resolved_envelope',wideflux,(.12*inner-1.30*rr1)[None,None],padding=5);options.append(reshape('envelope_test',env,[1,1,Nw*Nw]));
# Retain the directional local-maximum measure: this one is not satisfied by
# a diffuse gradient on a single centrally-concentrated galaxy.
options.append(reshape('prominence_test','wide_peak_field',[1,1,Nw*Nw]));
# Native max across the tests by the identity max(a,b)=b+ReLU(a-b).
z=options[0]
for jj,v in enumerate(options[1:]):
    dq=proj('negative_test%d'%jj,v,[[-1]]);dq=relu('excess_test%d'%jj,op('compare_test%d'%jj,'add',[z,dq]));z=op('worst_test%d'%jj,'add',[v,dq]);
wscore=dense('association_prior',z,-np.eye(Nw*Nw),-(np.maximum(0,7.-rw)*.14+np.maximum(0,rw-22.)*.0012+np.maximum(0,rw-40.)*.0035).ravel());wscore=reshape('associated_nuclei',wscore,[1,Nw,Nw]);wscore=pool('associated_secondary',wscore,Nw,'maxpool'); assocpeak=read(wscore,[1]); assoc=ratio(assocpeak,core+.08,.14)
# A visible resolved companion in a contiguous envelope is much stronger
# evidence than an arbitrary bright star. A blended double core may escape
# the bridge estimator but appear in both twin peaks and inner asymmetry.
merger=-.33+1.22*ramp(assoc,.018,.155)+.47*ramp(twinfrac,.064,.24)+.41*ramp(asym,.095,.42)+.22*ramp(odd[0],.14,.48)
merger+=.21*mn(ramp(widecomp,.07,.24),ramp(error,.035,.14))
# Directed outer tidal asymmetry: project the odd part only on large coherent
# modes in the central (rather than field-star dominated) radii. m=1 and m=3
# encode an asymmetric centre or asymmetric outer isophotes. We use their
# absolute contrast to reject spurious modes due to noisy near-black sky.
oddshape=group('m1_',[1,2,3]); oddtail=group('m1_',[3,4,5]);
disturbed= -.53+1.02*ramp(asym,.10,.43)+.57*ramp(oddtail,.09,.34)+.43*ramp(displaced,.42,2.50)+.35*ramp(oddshape,.055,.23)+.22*ramp(shift,2.1,5.8)-.30*ramp(assoc,.045,.16)-.13*broad
# Ellipticals have smooth light contours at all radii; do not confuse colour
# alone with an indication of spiral structure. For the subtype of a disc,
# explicitly test for a bar using isophotal twist: project the inner m2 on
# each trial orientation, subtracting the outer m2 in that direction. The
# modulus of this difference divided by inner light is a direct, phase-aware
# contrast; unlike subtracting scalar magnitudes it can detect a misaligned
# bar even when the outer disc is itself very elliptical.
forms=[]; denom=[]
for j,k in [(0,3),(1,3),(1,4),(2,4),(2,5),(3,5),(3,6)]:
    d=(.55*cmode(j,2)+.45*cmode(min(j+1,R-1),2)) - (.60*cmode(k,2)+.40*cmode(min(k+1,R-1),2));
    forms.extend([(np.cos(t)*d[0]+np.sin(t)*d[1]).flatten() for t in np.arange(8)*np.pi/4]);denom.append((.55*row(j)+.45*row(j+1)).flatten());
nd=len(denom);u=relu('bar_phase_rectified',proj('bar_phase_filter',coeff,forms));m=np.zeros((nd,nd*8));
for idx0 in range(nd):m[idx0,idx0*8:idx0*8+8]=np.pi/8
u=proj('bar_phase_strength',u,m);d=proj('bar_phase_continuum',coeff,np.asarray(denom)*-.072,-.002);u=op('bar_contrast_excess','add',[u,d]);u=op('bar_test_in_disc','add',[u,proj('bar_reference_frame',delta,[[.23]]*nd)]);u=op('bar_cross_frame','maxpool',u,size=[1,H],stride=[1,H]);u=op('bar_radial_aperture','maxpool',reshape('bar_windows',u,[1,1,nd]),size=[1,nd],stride=[1,nd]); straight=read(u,[1]);
# Core-to-disc colour differences alone would also mark the old bulge in
# some unbarred spirals. Require a non-circular central region along with
# the redder central population; a pure pitch signature favours loose arms.
bar_sig= .52*ramp(bar,.042,.142)+.42*ramp(straight,-.001,.020)+.41*ramp(barcorr,-.014,.014)+.22*ramp(corecol-diskcol,.014,.26)
loose_sig=.52*ramp(openarm,.003,.017)+.42*ramp(outertex,.025,.100)+.24*ramp(odd[1],.09,.38)+.24*ramp(-outercol,.005,.24)+.20*ramp(arm,.055,.17)
bars=bar_sig-.78;loose=loose_sig-.84
# A face-on disc with radial arms/disc-breaks, unlike a smooth spheroid,
# either has a measurable modulation or very extended blue isophotes.
spiral+=.21*ramp(error,.030,.13)+.29*ramp(straight,.001,.020)+.15*ramp(outertex,.03,.085)-.20*ramp(elliptical,.50,.9)
# Explicit track of the faint major axis past the bright centre. Compare
# light between narrow on-axis wedges and off-axis wedges, by angle, for two
# annuli. This rewards continued thin extensions beyond the central bulge
# without using the halo light from unrelated objects. Use both ends of the
# axis (min of opposing sectors) to ignore a single star or one-sided tail.
tailstrength=[]; ablength=[]; fullcent=read(flux,unit(gauss(3)))
for t in angles:
    X=x*np.cos(t)+y*np.sin(t);Y=y*np.cos(t)-x*np.sin(t);kern=np.exp(-.5*(Y/2.8)**2-.5*((abs(X)-15)/5.)**2); wing=np.exp(-.5*((abs(Y)-6)/2.5)**2-.5*((abs(X)-15)/5.)**2);
    q=[]
    for sign in [-1,1]:
        vv=unit(kern*(sign*X>0));ww=unit(wing*(sign*X>0));q.append(read(flux,vv-.60*ww));
    tailstrength.append(mn(q[0],q[1]));
length=ratio(vmax(tailstrength),fullcent+.052,.18); axial=ramp(length,.050,.235); thinwing=ramp(we,.36,.65)
# In the absence of a bilateral continuation of the major axis, a moderately
# elongated smooth source is more likely cigar-shaped than an edge-on disc.
edge-=.46*(1-axial)+.15*(1-longthin); edge+=.32*axial+.15*mn(thinwing,extent)
bul+=.39*(1-ramp(E,.42,.63))+.25*ramp(red,.02,.21)-.15*blue
# Scale from ellipse templates: q and width describe the same profile that
# gave the good correlation, not unrelated maxima of two sets of features.
# For any threshold Rmaj, compare best correlation on both sides; this is
# an evidence margin for major-axis size, conditional on profile family.
# An exponential has R_e=1.678*h; in the flat-disc family with exponent 1.5
# and coefficient .68 the half-light semi-major radius is about 1.35*h.
def extent_margin(name,fac):
    mask1=np.where(np.array(radii)*fac>=11,0.,-100.); mask2=np.where(np.array(radii)*fac<11,0.,-100.); vals=[]
    for j,mask in enumerate([mask1,mask2]):
        nm='extent_'+name+str(j); inp='profile_scale_'+name; v=dense(nm+'_masked',inp,np.eye(len(radii)*len(axis_grid)),np.repeat(mask,len(axis_grid)));v=reshape(nm+'_grid',v,[1,1,len(radii)*len(axis_grid)]);v=op(nm+'_max','maxpool',v,size=[1,len(radii)*len(axis_grid)],stride=[1,len(radii)*len(axis_grid)]);vals.append(read(v,[1]))
    return vals
exts=[extent_margin(name,f) for name,f in [('flat',1.35),('exponential',1.53),('disknucleus',1.23),('thin',1.02),('inclinedbulge',.90)]]; major_extent=ratio(vmax([p[0] for p in exts])-vmax([p[1] for p in exts]),allmod+.04,.035)
largeaxis=ramp(major_extent,-.007,.023)
# Shape change replacing aperture thickness alone: an entire galaxy that is
# well described by *one* thin elliptical profile shouldn't get credit for
# having a bulge and a disc unless it also has extended major-axis tails.
edge-=.72*(1-largeaxis)+.34*ramp(-edge_adv,-.002,.025);edge+=.35*largeaxis+.18*axial
spiral+=.22*largeaxis-.15*(1-largeaxis)
# Local asymmetry in the brightest, inner half of the galaxy, insensitive
# to sky normalization. Bound the leverage of any one star and compare
# compact inner, middle, outer regions separately. Recompute on two-pixel
# samples so narrow dust lanes are not removed by averaging 4x4 pixels.
fd=dense('rotational_difference_full',flux,diff); fd=reshape('rotational_difference',fd,[1,N,N]);fd=relu('positive_rotational_error',fd);u=relu('clipped_rotational_excess',proj('rotational_clipping',fd,[[1]],-.24));fd=op('robust_rotational_error','add',[fd,proj('neg_clipped_error',u,[[-1]])]);
rot=[]
for so,si in [(5.,0.),(10.,3.),(17.,6.)]:
    ww=unit(gauss(so)*(1-gauss(si)) if si else gauss(so));rrsym=read(fd,ww)*2; ll=read(flux,ww);rot.append(ratio(pos(rrsym-.012),ll+.035,.25))
innerodd=.60*rot[0]+.40*rot[1]; outerodd=.60*rot[1]+.40*rot[2]
# Compare orientation/odd patterns in the annuli to the amplitude of the
# central bulge. A symmetric two-armed spiral does not generate these modes;
# a displaced compact nucleus or a tidal tail does.
disturbed=mx(disturbed,-.22+.97*ramp(innerodd,.093,.35)+.56*ramp(outerodd,.14,.52)+.36*ramp(displaced,.32,2.2)+.24*ramp(shift0,1.8,5.3)+.21*ramp(oddshape,.06,.19)-.30*ramp(assoc,.035,.15)-.12*largeaxis)
# Galaxy pairs: brightness bridge alone misses deblended objects. A second
# rule accepts a separated, independently resolved nucleus when the other
# core is also bright. Lens-like smooth pairs need not have high asymmetry.
pair=mn(ramp(widecomp,.046,.166),ramp(twinfrac,.057,.195));merger=mx(merger,-.26+1.33*pair+.32*ramp(assoc,.023,.12)+.31*ramp(outerodd,.115,.48)+.20*ramp(displaced,.65,2.8))
# Colour/morphology coupling: paired angularly opposed blue arm knots with
# an intervening red/yellow elongation suggest a bar. Estimate an INNER colour
# quadrupole and compare it with concentric colour in the outer disc: a
# grey ellipse has no red-vs-blue shape contrast. Use raw fine-bin channels
# in a small central field and suppress the sky with zero-DC kernels.
Yc,Xc=(np.indices((64,64))+.5)*2-64; Rc=np.hypot(Xc,Yc);tc=np.arctan2(Yc,Xc); colorbar=[]; knotblue=[]
for rad0 in [5.5,9.,14.]:
    mask=(Rc/rad0)**2*np.exp(-(Rc/rad0)**2);mask=unit(mask); pal=[]
    for cang in [np.cos(2*tc),np.sin(2*tc)]:
        w0=mask*cang; w0-=w0.sum()*mask; rr=read(p2,np.array([1.,0.,-1.])[:,None,None]*w0);pal.append(rr)
    mcol=read(p2,phot[:,None,None]*mask);colorbar.append(ratio(vecmag(*pal),mcol-read(sky,[1])+.075,.12))
    w=unit(mask*(1-np.exp(-Rc**2/(2*4**2))));knotblue.append(read(p2,np.array([-1.,0.,1.])[:,None,None]*w))
color_shape=ramp(vmax(colorbar),.017,.089); barlead=ramp(bar,.026,.101); barredarms=ramp(straight,-.003,.011)
bars+=.29*mn(color_shape,barlead)+.23*mn(barredarms,ramp(barcorr,-.016,.004))+.14*ramp(arm,.030,.11)-.085
# Loose arms often have a blue outer disc, a pronounced ridge relative to the
# sky, and their light pattern reverses phase over a short logarithmic range.
# Tight arms can also look blue but have weak high-amplitude modulation.
loose+=.30*mn(ramp(openarm,.002,.012),ramp(outertex,.018,.070))+.15*ramp(outerodd,.12,.37)+.18*ramp(vmax(knotblue),-.012,.042)-.075
stype=[bars,lin([0],[1]),loose]; most=vmax(stype)
scores=[disturbed,merger,smooth+s2,smooth+s3,smooth+s4]+[spiral+u-most for u in stype]+[edge-pos(bul),edge-pos(-bul)]

# Compile the scalar measurement DSL. Combine consecutive affine maps and
# schedule each bank of parallel nonlinear operations topologically. Affine
# coefficients come solely from the formulas above.
from functools import lru_cache
@lru_cache(None)
def expand(i):
    typ,args,b=expr[i]
    if typ!='linear':return {i:1.},0.
    d={}; c=b
    for j,k in args:
        e,v=expand(j); c+=k*v
        for u,z in e.items():d[u]=d.get(u,0)+k*z
    return {u:z for u,z in d.items() if abs(z)>1.e-14},c
@lru_cache(None)
def depth(i):
    typ,args,k=expr[i]
    if typ=='input':return 0
    if typ=='linear':return max([depth(j) for j in expand(i)[0]]+[0])
    return max(depth(j) for j,b in args)+1
# Combine linear measurements taken from different spatial feature maps.
allids=[i for src,rows in banks.items() for i,w in rows]; dim=len(allids); loc={i:j for j,i in enumerate(allids)}; pieces=[]
for j,(src,rows) in enumerate(banks.items()):
    a=dense('measure_%02d'%j,src,[w for _,w in rows]); w=np.zeros((dim,len(rows)));w[[loc[i] for i,v in rows],np.arange(len(rows))]=1;pieces.append(dense('place_%02d'%j,a,w))
state=op('raw_observables','add',pieces)
def coefs(ids):
    w=np.zeros((len(ids),len(loc)));b=np.zeros(len(ids));
    for u,j in enumerate(ids):
        ds,b[u]=expand(j)
        for p,k in ds.items():w[u,loc[p]]=k
    return w,b
# Drop intermediate scalars after their last use; keeps the affine matrices
# small even when many independent deprojection statistics are assembled.
final_depth=max(depth(v.i) for v in scores)+1
last={}
for i,(kind,args,k) in enumerate(expr):
    if kind in ('relu','ratio'):
        for j,_ in args:
            for u in expand(j)[0]:last[u]=max(last.get(u,0),depth(i))
for v in scores:
    for u in expand(v.i)[0]:last[u]=final_depth
for level in range(1,final_depth):
    nm='rules_%02d'%level; pieces=[]; groups=[]
    keep=[i for i in loc if last.get(i,0)>level];
    if keep:pieces.append(dense(nm+'_retain',state,*coefs(keep)));groups.append(keep)
    for kind in ['relu','ratio']:
        group=[i for i,(k,_,_) in enumerate(expr) if k==kind and depth(i)==level];
        if not group:continue
        ids=[j for i in group for j,_ in expr[i][1]]; w,b=coefs(ids)
        if kind=='relu':a=relu(nm+'_relu',dense(nm+'_affine',state,w,b))
        else:
            scales=np.array([expr[i][2] for i in group]); w[::2]/=scales[:,None]; b[::2]/=scales
            a=dense(nm+'_pairs',state,w,b); a=ratio_layer(nm+'_normalize',a,len(group));a=dense(nm+'_units',a,np.diag(scales))
        groups.append(group);pieces.append(a)
    allids=sum(groups,[]);loc={i:j for j,i in enumerate(allids)}
    if len(groups)==1:state=pieces[0];continue
    pack=[]
    for j,(grp,src) in enumerate(zip(groups,pieces)):
        w=np.zeros((len(allids),len(grp)));w[[loc[i] for i in grp],np.arange(len(grp))]=1;pack.append(dense(nm+'_insert%d'%j,src,w))
    state=op(nm,'add',pack)
dense('galaxy_class',state,*coefs([v.i for v in scores]));
with open(os.path.join(D,'network.json'),'w') as f:json.dump({'layers':layers},f,separators=(',',':'))
np.savez(os.path.join(D,'weights.npz'),**weights)
