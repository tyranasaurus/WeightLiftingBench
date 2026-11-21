"""A fixed photometry, morphology and angular-harmonic network. No data inputs."""
import json
import math
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
L=[]; W={}; dim={'image':(3,128,128)}
F=np.float32

def node(name,op,ins,output_shape,**kw):
    L.append(dict(name=name,op=op,inputs=[ins] if isinstance(ins,str) else list(ins),**kw));dim[name]=output_shape;return name

def dense(name,src,w,b=0):
    w=np.asarray(w,F).reshape(-1,math.prod(dim[src])); W[name+'.W']=w; W[name+'.b']=np.broadcast_to(b,(len(w),)).astype(F)
    return node(name,'dense',src,(len(w),),out=len(w))

def conv(name,src,w,b=0,pad=0,stride=1):
    w=np.asarray(w,F);o,i,k,_=w.shape;c,h,n=dim[src];assert i==c
    W[name+'.W']=w; W[name+'.b']=np.broadcast_to(b,(o,)).astype(F)
    return node(name,'conv',src,(o,(h+2*pad-k)//stride+1,(n+2*pad-k)//stride+1),out=o,kernel=[k,k],padding=pad,stride=stride)

def reshape(name,src,shape):return node(name,'reshape',src,tuple(shape),shape=list(shape))
def plus(name,*src):return node(name,'add',list(src),dim[src[0]])
def rel(name,src):return node(name,'relu',src,dim[src])
def pool(name,src,k,st,op):
    c,h,w=dim[src];kh,kw=k;sh,sw=st
    return node(name,op,src,(c,(h-kh)//sh+1,(w-kw)//sw+1),size=list(k),stride=list(st))
def gather(name,src,indices):
    ids=np.asarray(indices).reshape(-1);w=np.zeros((len(ids),math.prod(dim[src])),F);w[np.arange(len(ids)),ids]=1
    return dense(name,src,w)

# The central field is sampled in 2x2 squares. The survey center, not a
# position from any example, is the origin of every filter.
bin2=pool('bin2','image',(2,2),(2,2),'avgpool')
n=36; nn=n*n; y,x=(np.indices((n,n))-(n-1)/2)*2; r=np.hypot(x,y)
idx=np.arange(64*64).reshape(64,64)[(64-n)//2:(64+n)//2,(64-n)//2:(64+n)//2].ravel()
luma=np.array([.32,.43,.25]);lum=conv('luminance',bin2,luma[None,:,None,None]);crop=reshape('field',gather('central_field',lum,idx),(1,n,n))
# Estimate sky from a wide annulus, clipping positive foreground contamination
# by sampling several disjoint angular sky sectors and taking their minimum.
Y,X=(np.indices((64,64))-31.5)*2;R=np.hypot(X,Y);ph=np.arctan2(Y,X)
sk=[]
for j in range(6):
    m=(R>42)&(R<62)&(np.cos(ph-j*np.pi/3)>np.cos(np.pi/6));sk.append(-m.astype(float)/m.sum())
sky=reshape('sky_sectors',dense('negative_sky',lum,sk),(1,1,len(sk)))
sky=pool('dark_sky',sky,(1,len(sk)),(1,len(sk)),'maxpool')
bg=reshape('sky_map',dense('repeat_sky',sky,np.ones((nn,1))),(1,n,n))
field=plus('sky_subtracted',crop,bg)
# Light smoothing to suppress pixel noise while retaining double nuclei.
t=np.arange(-2,3);g=np.exp(-.5*(t/.72)**2);g/=g.sum()
light=conv('seeing',field,np.outer(g,g)[None,None],pad=2)
# Keep above-sky flux. Additional, brighter isophotes give a robust shape
# estimate that is less affected by tidal tails and projected stars.
lev=conv('isophotes',light,np.ones((3,1,1,1)),[-.027,-.09,-.19]);lev=rel('isophotal_light',lev)

def single(name,src,j):
    w=np.zeros((1,dim[src][0],1,1));w[0,j]=1;return conv(name,src,w)
low=single('faint_light',lev,0);mid=single('body_light',lev,1)
# Elliptical exponential matched filters. Normalization is independent for
# every analytic template. A sky-subtracted, rotated family gives a soft
# photometric coordinate system for the angular measurements below.
qs=[.14,.215,.31,.43,.57,.72,.86,1.];scales=[2.5,3.7,5.4,7.9,11.6,17.,25.];nt=16
bank=[];compensated=[]
for q in qs:
    for s in scales:
        for j in range(nt):
            ang=j*np.pi/nt;u=x*np.cos(ang)+y*np.sin(ang);v=-x*np.sin(ang)+y*np.cos(ang)
            # Add the point-spread variance on both axes. Taper only the very
            # outskirts where the cutout is small relative to the template.
            rho=np.sqrt((u*u/(s*s+1.3**2))+(v*v/((q*s)**2+1.3**2)))
            m=np.exp(-rho)*np.exp(-.5*(r/32)**4)
            # Weighted zero-DC correlation gives a per-image goodness of
            # match via the Schwarz inequality; no population calibration.
            window=np.exp(-.5*(r/12)**2);window/=window.sum();std=m-(window*m).sum();compensated.append(window*std/np.sqrt((window*std**2).sum()))
            m-=m.mean();bank.append(m/np.sqrt((m*m).sum()))
shape=reshape('ellipse_grid',dense('ellipse_correlations',light,bank),(len(qs),len(scales),nt))
qfit=pool('axis_ratio_profile',shape,(len(scales),nt),(len(scales),nt),'maxpool')
sfit=pool('major_radius_profile',shape,(1,nt),(1,nt),'maxpool')
sfit=reshape('radius_by_flatness',sfit,(1,len(qs),len(scales)))
sfit=pool('major_scale',sfit,(len(qs),1),(len(qs),1),'maxpool')
geom=pool('shape_by_angle',shape,(len(scales),1),(len(scales),1),'maxpool')
best=reshape('ellipse_values',qfit,(1,1,len(qs)))
best=pool('matched_light',best,(1,len(qs)),(1,len(qs)),'maxpool')

# Linear measurements are collected in one matrix per input image field.
# The small scalar network below is emitted from symbolic affine expressions
# so every test in the decision tree has an explicit definition.
class E:
    def __init__(self,d=None,b=0):self.d=d or {};self.b=b
    def __add__(self,v):
        v=expr(v);d=self.d.copy()
        for k,a in v.d.items():d[k]=d.get(k,0)+a
        return E(d,self.b+v.b)
    __radd__=__add__
    def __neg__(self):return self*(-1)
    def __sub__(self,v):return self+(-expr(v))
    def __rsub__(self,v):return expr(v)+(-self)
    def __mul__(self,v):return E({k:a*v for k,a in self.d.items()},self.b*v)
    __rmul__=__mul__
    def __truediv__(self,v):return self*(1/v)

def expr(v):return v if isinstance(v,E) else E(b=float(v))
depth={};ops={};sources={};count=0

def raw(src,i=0):
    k=(src,int(i));depth[k]=0;sources[k]=k;return E({k:1})
def pos(a):
    global count
    a=expr(a);k=count; count+=1;depth[k]=1+max([depth[i] for i in a.d]+[0]);ops[k]=a;return E({k:1})
def absval(a):return pos(a)+pos(-a)
def max2(a,b):return pos(a-b)+b
def min2(a,b):return a-pos(a-b)
def vmax(a):
    a=list(a)
    while len(a)>1:a=[max2(*a[i:i+2]) if i+1<len(a) else a[i] for i in range(0,len(a),2)]
    return a[0]
def clip(a,lo=0,hi=1):return lo+pos(a-lo)-pos(a-hi)
def membership(x,h=.5):return clip(.5+x/(2*h))

masks={}
def measure(src,mask):
    name=src+'_apertures';M=masks.setdefault(src,[]);i=len(M);M.append(np.asarray(mask,F));return raw(name,i)

# RMS error of an analytic single-component profile. For the best-centered
# ellipse correlation h and aperture variance v, sqrt(1-h^2/v) measures
# departure from a smooth single galaxy. Implement directly in weighted
# squared-flux units. Squaring uses the secants of t^2 over a uniform grid.
ec=reshape('inner_match_grid',dense('inner_profile_match',light,compensated),(1,len(qs)*len(scales),nt));ec=pool('best_inner_match',ec,(len(qs)*len(scales),nt),(len(qs)*len(scales),nt),'maxpool')
knots=np.r_[0,np.arange(1,21)/25]; sq=rel('square_parts',conv('square_intervals',light,np.ones((len(knots),1,1,1)),-knots));sq=conv('square_light',sq,(np.r_[.04,np.full(len(knots)-1,.08)]).reshape(1,-1,1,1))
def sqr(z,step=.0075):return sum((step if i==0 else 2*step)*pos(z-i*step) for i in range(65))
smoothvar=measure(sq,window)-sqr(measure(light,window));varres=smoothvar-sqr(raw(ec))

def weight_rad(s):return np.exp(-.5*(r/s)**2)
def mean(src,mask):return measure(src,mask/mask.sum())
# Circular apertures and a compact central colour measurement, with a local
# wide sky reference in each band to cancel offsets in the JPEG sky.
def color_measure(mask):
    a=np.zeros((64,64));a.ravel()[idx]=mask.ravel()/mask.sum();skyband=(R>40)&(R<60);a-=skyband/skyband.sum()
    return [measure(bin2,np.eye(3)[j,:,None,None]*a) for j in range(3)]
coreR,coreG,coreB=color_measure(weight_rad(3.2)); envR,envG,envB=color_measure((1-weight_rad(5))*weight_rad(10))
central=mean(light,weight_rad(2.2));ap6=mean(light,weight_rad(6)); ap13=mean(light,weight_rad(13))

# Squared-light proxy for measuring moments without sky noise: piecewise
# linear integration through increasing brightness isophotes. The DC weights
# emphasize the central body over unrelated sources in the outer field.

def moment_mask(a):return np.stack([.25*a,.55*a,.40*a])

def moments(s):
    wt=weight_rad(s);norm=wt.sum();wt=wt/norm
    mass=measure(lev,moment_mask(wt)); xx=measure(lev,moment_mask(wt*r*r));
    cos2=measure(lev,moment_mask(wt*(x*x-y*y)));sin2=measure(lev,moment_mask(wt*(2*x*y)))
    # Piecewise linear Euclidean norm via a regular polygon.
    amp=sum(absval(cos2*np.cos(j*np.pi/6)+sin2*np.sin(j*np.pi/6)) for j in range(6))*(np.pi/12)
    return mass,xx,amp
mass,trace,anis=moments(9);mcore,trcore,ancore=moments(4.5)

# Extract the degree of departure from point symmetry, allowing subpixel
# centering differences. The central weighting discourages field sources.
# Reflecting a vector is just an index permutation (a fixed dense map).
rot=reshape('opposite_point',dense('reverse_field',low,np.eye(nn,dtype=F)[::-1]),(1,n,n))
negrot=conv('negative_opposite',rot,-np.ones((1,1,1,1)))
delta=plus('parity_difference',low,negrot);odd=rel('odd_light',delta)
# Pointwise intersection of antipodal flux suppresses unrelated field
# sources in estimates of the main body's symmetry and flattening.
paired=plus('symmetric_body',low,conv('remove_unpaired',odd,-np.ones((1,1,1,1))))
pshape=reshape('paired_grid',dense('paired_ellipses',paired,bank),(len(qs),len(scales),nt));pqfit=pool('paired_flatness',pshape,(len(scales),nt),(len(scales),nt),'maxpool');pbyangle=pool('paired_angle',pshape,(len(scales),1),(len(scales),1),'maxpool');pbest=pool('paired_match',reshape('paired_by_q',pqfit,(1,1,len(qs))),(1,len(qs)),(1,len(qs)),'maxpool')
# Estimate shape in several radial zones using a fractional *angular*
# quadrupole. For elliptical isophotes, the unweighted global value of
# |<exp(2i*theta)>| is (1-q)/(1+q), independently of radial profile.
# Sky rejection and inner/outer weighting regularize it in a finite field.
phi0=np.arctan2(y,x); radial_shape=[]
for w in [weight_rad(4),weight_rad(10),weight_rad(17)*(1-weight_rad(6)),weight_rad(27)*(1-weight_rad(10))]:
    # Mild flattening of the radial falloff makes outer disks visible.
    w=w/w.sum();f=measure(paired,w);ac=measure(paired,w*np.cos(2*phi0));asi=measure(paired,w*np.sin(2*phi0))
    radial_shape.append((ac,asi,f))
# Robust local difference: ignore displacement of one low-resolution pixel
# between a point and its antipode. Smooth ellipticals then have negligible
# positive residual even with mild centering errors.
rotmax=pool('opposite_neighborhood',reshape('opposite_image',rot,(1,n,n)),(3,3),(1,1),'maxpool')
inner_ids=np.arange(nn).reshape(n,n)[1:-1,1:-1].ravel();centerField=reshape('inner_field',gather('trim_field',low,inner_ids),(1,n-2,n-2))
negnear=conv('negative_local_opposite',rotmax,-np.ones((1,1,1,1)))
robodd=rel('unpaired_light',plus('tolerant_difference',centerField,negnear))
asym=[]
for s in [6.5,13,22]:
    wt=weight_rad(s); wt/=wt.sum()
    asym.append((measure(odd,wt)*2,measure(robodd,wt[1:-1,1:-1])*2,measure(low,wt)))
lopsided=[]
for rr0,sw in [(6,3),(12,4),(20,6)]:
    w=np.exp(-.5*((r-rr0)/sw)**2);w/=w.sum();f=measure(low,w);cx=measure(low,w*np.cos(phi0));cy=measure(low,w*np.sin(phi0));lopsided.append((f,cx,cy))
# Local peaks: positive contrast against every pair of diametrically
# opposite samples on a ring. A straight ridge has little such contrast.
# Contrast between two *pairs* versus a max across all directions makes
# detection tolerant of a faint bridge linking a pair of merging nuclei.
peak_fields=[]
for st in [2,4]:
    # Opposing samples along 4 diameters, at comparable radii.
    diag=max(1,round(st/math.sqrt(2))); ds=[(st,0),(diag,diag),(0,st),(-diag,diag)]
    fl=[]
    for dy,dx in ds:
        for lead in [.23,.77]:
            a=np.zeros((2*st+1,2*st+1));a[st,st]=-1;a[st+dy,st+dx]=lead;a[st-dy,st-dx]=1-lead;fl.append(a)
    a=conv('peak_directions'+str(st),light,np.array(fl)[:,None],pad=st)
    a=reshape('direction_stack'+str(st),a,(1,len(fl),nn));a=pool('weakest_curvature'+str(st),a,(len(fl),1),(len(fl),1),'maxpool');a=reshape('curvature_field'+str(st),a,(1,n,n));a=conv('peak_contrast'+str(st),a,-np.ones((1,1,1,1)));peak_fields.append(a)

def select_max(name,src,regions):
    # fixed lists of grid points, grouped for a maxpool; no measured locations
    ix=[np.flatnonzero(m) for m in regions];K=max(map(len,ix));idxs=[np.pad(a,(0,K-len(a)),mode='edge') for a in ix];t=gather(name+'_samples',src,idxs);t=reshape(name+'_regions',t,(len(ix),1,K));t=pool(name+'_maxima',t,(1,K),(1,K),'maxpool');return [raw(t,i) for i in range(len(ix))]
peaks=[]
for j,a in enumerate(peak_fields):
    # Also sample opposite sectors: a pair of peaks flanking the pointing
    # center indicates a merger even if there is no nucleus at the center.
    regs=[r<4.5,(r>=4.5)&(r<10),(r>=10)&(r<20),(r>=20)&(r<32)]
    regs += [(r>=3.5)&(r<16)&(np.cos(np.arctan2(y,x)-i*np.pi/4)>np.cos(np.pi/5.5)) for i in range(8)]
    peaks.append(select_max('peak'+str(j),a,regs))

# Isophotal diameters, independent of the exponential profile assumption.
# At several fractional levels of the central surface brightness, count
# the faint/body/core area and compute its central inertia ellipse. A
# second gate is required above sky. The steep saturating ramp approximates
# an indicator with a small transition band for numerical stability.
central_reference=mean(light,weight_rad(1.65)); gate=np.repeat((weight_rad(1.65)/weight_rad(1.65).sum()).reshape(1,-1),nn,axis=0)
refmap=reshape('central_brightness',dense('repeat_apex',light,gate),(1,n,n));fractions=np.array([.16,.34,.58])
gates=plus('above_isophote',conv('levels_light',paired,np.ones((3,1,1,1))),conv('fractional_levels',refmap,(-fractions).reshape(3,1,1,1)));tw=.018
occup=plus('inside_contours',rel('level_floor',conv('scale_levels',gates,np.eye(3)[:,:,None,None]/tw,.5)),conv('subtract_ceiling',rel('level_excess',conv('excess_levels',gates,np.eye(3)[:,:,None,None]/tw,-.5)),-np.eye(3)[:,:,None,None]));contour=[]
for j,s in enumerate([26,19,14]):
    w=weight_rad(s);wc=np.eye(3)[j,:,None,None]*w*4 # pixel area in original units
    area=measure(occup,wc);mom=measure(occup,wc*r*r);cx=measure(occup,wc*(x*x-y*y));cy=measure(occup,wc*2*x*y);contour.append((area,mom,cx,cy))

# A tidal tail can be faint in flux but long in angle and in area. Use
# the first angular moment of capped isophotal occupancy (not flux).
# Sky cut: the taper and paired core amplitude limit field contamination.
faint_area=rel('tail_threshold',conv('tail_occupancy_start',light,np.ones((1,1,1,1))/.075,-.035/.075));faint_area=plus('tail_occupancy',faint_area,conv('cap_tail_occupancy',rel('tail_saturation',conv('locate_tail_saturation',faint_area,np.ones((1,1,1,1)),-1)),-np.ones((1,1,1,1))));wt=weight_rad(18)*(1-weight_rad(5));wt/=wt.sum();tailmass=measure(faint_area,wt);tailx=measure(faint_area,wt*np.cos(phi0));taily=measure(faint_area,wt*np.sin(phi0))
# Two compact components: difference of Gaussian matched apertures for
# two nucleus scales, with a central-over-ring contrast to discount very
# small unrelated PSF-like sources. A max on each side of a dividing line
# measures the strength of a second source even if the brighter source is
# off center (the pointing is not necessarily the pair midpoint).
ng=8;nucpos=[(xx,yy) for yy in np.arange(-24,25,3) for xx in np.arange(-24,25,3) if xx**2+yy**2<24**2];nucpos=np.asarray(nucpos);nf=[];widths=[1.9,3.7]
for sw in widths:
    for xx,yy in nucpos:
        d2=(x-xx)**2+(y-yy)**2;wa=np.exp(-d2/(2*sw**2));wb=np.exp(-d2/(2*(sw*2.6)**2));nf.append(wa/wa.sum()-wb/wb.sum())
nuc=dense('nucleus_filter_bank',light,nf);np0=len(nucpos);sep=5.5;nsids=[]
# Minimum surface brightness along the path back to the target. A true
# close pair has a shared envelope; a randomly projected star usually has
# sky between it and the center. Compare to the local peak *at runtime*,
# not to a flux constant depending on distance or exposure.
bridge_filters=[];peak_filters=[]
for xx,yy in nucpos:
    for frac in [.30,.52,.72,.85]:
        m=np.exp(-.5*((x-frac*xx)**2+(y-frac*yy)**2)/1.8**2);bridge_filters.append(-m/m.sum())
    m=np.exp(-.5*((x-xx)**2+(y-yy)**2)/2**2);peak_filters.append(m/m.sum())
bridge=reshape('path_photometry',dense('negative_path_samples',light,bridge_filters),(np0,1,4));bridge=pool('least_path_brightness',bridge,(1,4),(1,4),'maxpool');pbright=dense('component_brightness',light,peak_filters);bridge=dense('negative_bridge',bridge,2*np.eye(np0));pbright=dense('bridge_minimum_required',pbright,.44*np.eye(np0),.014);gap=rel('disconnected_component',plus('path_deficit',bridge,pbright));gap=dense('detached_component_cost',gap,-2.0*np.tile(np.eye(np0),(2,1)))
for j in range(ng):
    a=j*np.pi/ng;proj=nucpos[:,0]*np.cos(a)+nucpos[:,1]*np.sin(a)
    for cut in [-4,0,4]:
        for sgn in [1,-1]:
            # Both sides must contain a local maximum. To prevent the
            # flanks of a *single* peak from counting twice, subtract a
            # broad positive patch displaced towards the separation line.
            # Select index pairs of displaced filters in build-time geometry.
            ids=np.flatnonzero(sgn*(proj-cut)>sep/2); nsids.append((sgn,a,cut,ids))
nsub=ng*3*2
# Sparse selection maps in this branch are implemented more economically
# one direction at a time by pooling on deterministic disk probes. Avoid
# materializing a halfplane-by-position-by-pixel matrix of zero weights.
# A nucleus is first qualified by contrast with the minimum increase toward
# its companions. Offset tests use a local maximum of the filter response.
nfmap=[]; npix=len(nucpos)
for h,sw in enumerate(widths):
    for i,pt in enumerate(nucpos):
        for j in range(8):
            delta=np.array([np.cos(j*np.pi/4),np.sin(j*np.pi/4)])*(3.5+sw/2);near=np.argmin(((nucpos-pt-delta)**2).sum(1));row=np.zeros((2,npix));row[h,i]=-.85;row[h,near]+=1;nfmap.append(row)
ncontrast=reshape('nucleus_neighbors',dense('local_nucleus_test',nuc,nfmap),(2,npix,8));ncontrast=pool('worst_nucleus_neighbor',ncontrast,(1,8),(1,8),'maxpool');ncontrast=reshape('nuc_unqualified',ncontrast,(1,1,2*npix));ncontrast=rel('nuc_shoulder',ncontrast);ncontrast=conv('penalize_nuclear_shoulders',ncontrast,-2.2*np.ones((1,1,1,1)));qualified=plus('qualified_nuclei',reshape('nuclear_peaks',nuc,(1,1,2*npix)),ncontrast)
# Max of min over two separated maxima, with the outer sky downweighted.
penalty=np.tile(np.maximum(np.hypot(*nucpos.T)-12,0)*.0025,2);qualified=plus('connected_nuclei',dense('nuclear_distance_cost',qualified,np.eye(2*npix),-penalty),gap)
nscores=[]
for h in range(2):
    # Fixed sparse 0/-1 mask instead of a full gather tensor per halfplane.
    gate=np.zeros((nsub,npix));
    for j,(_,_,_,ids) in enumerate(nsids):gate[j,ids]=1
    ni=gather('nuc_scale'+str(h),qualified,np.arange(npix)+npix*h);ni=reshape('nuc_one_scale'+str(h),ni,(1,1,npix));ni=conv('nuc_repeat'+str(h),ni,np.ones((nsub,1,1,1)));mask=dense('nuc_halfplane_mask'+str(h),sky,np.zeros((nsub*npix,1)),(gate.ravel()-1)*2);mask=reshape('nuc_mask_grid'+str(h),mask,(nsub,1,npix));ni=plus('nuc_masked'+str(h),ni,mask);ni=pool('nuc_halfplane_peak'+str(h),ni,(1,npix),(1,npix),'maxpool');ni=reshape('nuc_pairs'+str(h),ni,(ng*3,1,2));ni=conv('nuc_pair_sign'+str(h),ni,-np.eye(ng*3)[:,:,None,None]);ni=pool('weaker_of_two'+str(h),ni,(1,2),(1,2),'maxpool');ni=reshape('pair_strengths'+str(h),ni,(1,1,ng*3));ni=conv('secondary_strength'+str(h),ni,-np.ones((1,1,1,1)));ni=pool('best_pair'+str(h),ni,(1,ng*3),(1,ng*3),'maxpool');nscores.append(raw(ni))

# One-sided growth along radial rays is another type of asymmetry. A
# smooth centered ellipse fades along *every* ray. Nearby double nuclei
# and tidal knots, on the other hand, can brighten again away from the
# photocenter. The polar max tolerates an uncertain companion position.
ru=np.linspace(0,2*np.pi,40,endpoint=False); distances=[2,4.5,7.5,11.5,17,25,33]; ray_masks=[]
for radius in distances:
    for ang in ru:
        sep=(x-radius*np.cos(ang))**2+(y-radius*np.sin(ang))**2
        g=np.exp(-.5*sep/(1.4+.044*radius)**2);ray_masks.append(g/g.sum())
ray=dense('radial_profiles',light,ray_masks);nrays=len(ru); nd=len(distances)
growth_rules=[(i,j) for i in range(2,nd) for j in range(1,i) if i-j<=3];rw=[]
for i,j in growth_rules:
    for a in range(nrays):
        w=np.zeros((nd,nrays));w[i,a]=1;w[j,a]=-1;rw.append(w)
rr=reshape('radial_growth_grid',dense('radial_growth',ray,rw),(len(growth_rules),1,nrays));rr=pool('max_radial_growth',rr,(1,nrays),(1,nrays),'maxpool')
growth_near=vmax([raw(rr,k) for k,(i,j) in enumerate(growth_rules) if i<=4]);growth_far=vmax([raw(rr,k) for k,(i,j) in enumerate(growth_rules) if 4<=i<=5]);

# Tidal plumes: weak light on one side with a connected path to the main
# body. Require that the *opposite* side at the same separation is fainter.
# Pair comparison explicitly separates a tail from a large circular halo.
tr=[];tt=[]
for r0 in [7,12,19]:
    for a in range(nrays):
        ang=ru[a];u=x*np.cos(ang)+y*np.sin(ang);v=-x*np.sin(ang)+y*np.cos(ang);w=np.exp(-.5*((u-r0)/(.29*r0))**2-.5*(v/(1.7+.13*r0))**2);tr.append(w/w.sum());w=np.exp(-.5*((u-.56*r0)/(.2*r0))**2-.5*(v/(1.7+.07*r0))**2);tt.append(w/w.sum())
plume=dense('plume_probes',light,tr);base=dense('plume_bases',light,tt);k=len(tr);pi=np.arange(k).reshape(3,nrays);anti=np.roll(pi,nrays//2,axis=1).ravel();dpl=dense('unequal_plumes',plume,np.eye(k)-np.eye(k)[anti]);deficit=rel('plume_path_sky',plus('plume_continuity',dense('plume_requirement',plume,.57*np.eye(k),.008),dense('plume_inward',base,-np.eye(k))));plume=plus('attached_plume',dpl,dense('plume_gap_cost',deficit,-2.5*np.eye(k)));plume=reshape('plume_directions',plume,(3,1,nrays));plume=pool('unequal_extended_light',plume,(1,nrays),(1,nrays),'maxpool')
# Reconstruct a second-order polar isophote model and check the higher
# angular residual. Large smooth ellipticals can be flat, so subtract the
# smooth m=0,1,2 modes and a fourth-order term. Sky contamination is
# reduced by taking paired (180-degree symmetric) light before this step.
polar=dense('paired_polar_light',paired,ray_masks);hi=np.zeros((nd//1,nrays,nrays));angs=ru[:,None]-ru[None,:];
for i in range(nd):
    hi[i]=np.eye(nrays)-(1+2*np.cos(2*angs)+2*np.cos(4*angs)+2*np.cos(6*angs))/nrays
hm=np.zeros((nd,nrays,nd,nrays));
for i in range(nd):hm[i,:,i,:]=hi[i]
high=rel('positive_high_modes',dense('fine_arm_structure',polar,hm));high=reshape('fine_arm_rings',high,(nd,1,nrays));high=pool('fine_arm_energy',high,(1,nrays),(1,nrays),'avgpool')

# Number of sky-subtracted scale lengths along the major axis, and estimated
# axis ratio. Instead of fitting an image or using exponentials at inference,
# approximate an argmax with a compact piecewise-linear voting function.
# Compare each contiguous half of the parameter grid to its complement.
quality=raw(best);qresp=[raw(qfit,i) for i in range(len(qs))];sresp=[raw(sfit,i) for i in range(len(scales))]

def side_vote(arr,values,width,logarithmic=False):
    values=np.log(values) if logarithmic else np.array(values);total=expr(values[0]);
    for i in range(1,len(values)):
        d=vmax(arr[i:])-vmax(arr[:i]);total=total+(values[i]-values[i-1])*membership(d/width)
    return total
# Match scores have units of flux times the square root of aperture area.
# Sharpen the voting adaptively for small/faint objects by dividing the
# score gaps by total signal. The normalization operation implements scalar
# division by a positive reference as a ratio to its RMS, as described below.

# Radial surface-brightness changes along candidate major axes. Compare
# inner, middle and outer aspect separately: the extra round component
# in a bulged disk increases the center/wing ratio and thickens the center
# relative to a simple thin disk. Side apertures pair diametrically to
# reject an isolated field object. Pool over angle only after combining
# the evidence in each putative projected disk frame.
line_masks=[]
for j in range(nt):
    a=np.pi*j/nt;u0=x*np.cos(a)+y*np.sin(a);v0=-x*np.sin(a)+y*np.cos(a)
    for ucen,vcent,su,sv in [(7,0,2.6,1.7),(13,0,3.6,2.1),(19,0,4.1,2.4),(0,4.7,2.1,1.2),(10,4.7,2.8,1.3),(0,0,1.75,1.75)]:
        m=np.zeros_like(r)
        for sign in [-1,1]:
            for p in [-1,1]: m+=np.exp(-.5*((u0-sign*ucen)/su)**2-.5*((v0-p*vcent)/sv)**2)
        line_masks.append(m/m.sum())
line=dense('disk_spine',paired,line_masks);line_sel=[]
for j in range(nt):
    a=np.zeros((nt,6));a[j]=[.62,1.,.65,-.80,-.47,0];line_sel.append(a)
spine=dense('spine_rank',line,line_sel);spine=reshape('spine_angles',spine,(1,1,nt));spmax=pool('best_spine',spine,(1,nt),(1,nt),'maxpool');gaps=plus('spine_gap',reshape('per_angle_spine',spine,(nt,)),dense('best_spine_neg',spmax,-np.ones((nt,1))))
lf=[]
# Correct orientation gives the largest wing brightness relative to the
# nucleus. At that orientation also read central-to-wing colour/contrast.
for h,coef in enumerate([[0,1.,.6,0,0,0],[-.98,-.30,0,0,0,1],[-.30,0,0,1,-.7,0],[-.6,0,.85,0,0,0]]):
    ar=[]
    for j in range(nt):
        a=np.zeros((nt,6));a[j]=coef;ar.append(a)
    v=plus('spine_value'+str(h),dense('spine_apertures'+str(h),line,ar),dense('spine_orientation_cost'+str(h),gaps,2*np.eye(nt)));v=reshape('spine_readings'+str(h),v,(1,1,nt));v=pool('spine_stat'+str(h),v,(1,nt),(1,nt),'maxpool');lf.append(raw(v))

# Competing edge-on light profiles. A genuine bulge should improve the
# match more than merely changing the *radial* concentration at fixed q.
# Thus use the same two radial components in the control family, but lock
# their flattenings together. Normalize each template separately, using
# the same inner taper for all hypothesis tests. No template is ever
# compared to another template to calibrate a score or parameter.
eb=[];control=[];ns=12
for j in range(ns):
    a=j*np.pi/ns;u=x*np.cos(a)+y*np.sin(a);v=-x*np.sin(a)+y*np.cos(a)
    for s in [3.6,5.6,8.6,13.,20.]:
        for q in [.14,.235,.36,.52]:
            def prof(s,q):
                sig=1.6;h=np.hypot(u/np.hypot(s,sig),v/np.hypot(s*q,sig));soft=sig/max(1.6,2*s*q);return np.exp(soft-np.hypot(soft,h))
            disk=prof(s,q)
            def centered(m):
                z=m-(m*window).sum();return (z*window/np.sqrt((z*z*window).sum())).astype(F)
            control.append(centered(disk))
            for contraction in [.26,.48]:
                same=prof(s*contraction,q);rounder=prof(s*contraction,.78)
                for contrast in [.65,2.1]:
                    control.append(centered(disk+contrast*same));eb.append(centered(disk+contrast*rounder))

def best_profile(nm,templates):
    v=dense(nm+'_bank',light,templates);v=reshape(nm+'_grid',v,(1,ns,len(templates)//ns));v=pool(nm+'_scale',v,(1,len(templates)//ns),(1,len(templates)//ns),'maxpool');return raw(pool(nm+'_best',v,(ns,1),(ns,1),'maxpool'))
withbulge=best_profile('disk_and_bulge',eb);unified=best_profile('single_flatness',control)

# Dedicated angular channels: sample intensity on nested ellipses, retain
# the order-2 Fourier coefficient. For an axisymmetric inclined disk it is
# zero at its true flattening; spiral arms rotate its phase with log radius.
# For pitch p the derivative of the two-arm phase is 2*cot(p).
qplane=[.42,.63,.81,1.];na=8; radii=3.4*1.34**np.arange(9);nr=len(radii);nplane=len(qplane)*na
geomat=[];azim=[]; means=[]
for q in qplane:
    for ia in range(na):
        ang=np.pi*ia/na;u=x*np.cos(ang)+y*np.sin(ang);v=(-x*np.sin(ang)+y*np.cos(ang))/q
        phi=np.arctan2(v,u);ell=np.hypot(u,v);comp=np.stack([np.cos(2*phi),np.sin(2*phi)])
        for radius in radii:
            g=np.exp(-.5*((ell-radius)/max(1.25,.16*radius))**2);g/=g.sum()
            azim.extend(g*(comp-(g*comp).sum((1,2))[:,None,None]));means.append(g)
        # Correct geometry is the one whose correlation matches the overall
        # ellipse. Circular galaxies have no distinguished position angle.
        row=np.zeros((len(qs),nt));row[int(np.argmin(abs(np.array(qs)-q))),ia*nt//na]=1;geomat.append(row)
az=dense('quadrupoles',paired,azim);ann=dense('annular_flux',paired,means);plane_fit=dense('disk_projection',pbyangle,geomat)
# Share the same radial filters at each projection; convolution acts only at
# the feature-vector positions for that plane (1x1 kernels).
def plane_channels(name,src,k):
    # src vector interleaves features within each plane; move planes to the
    # spatial axis without a huge permutation using a small per-plane conv
    # alternative: original dense rows were ordered plane/radius/harmonic.
    # Fixed gather as a transpose (small for these scalar vectors).
    a=gather(name+'_transpose',src,np.arange(nplane*k).reshape(nplane,k).T);return reshape(name,a,(k,1,nplane))
az=plane_channels('angular_channels',az,nr*2);ann=plane_channels('ring_channels',ann,nr)

def plane_conv(name,src,w):return conv(name,src,np.asarray(w)[:,:,None,None])

def magnitudes(name,src,filters):
    # Sum absolute projections estimates the amplitude, independently of
    # spiral phase and hence invariant to the in-plane orientation.
    # filters: list of [2, input channels] giving complex response.
    w=np.array([np.cos(a)*h[0]+np.sin(a)*h[1] for h in filters for a in np.arange(8)*np.pi/4]);
    part=rel(name+'_phase_relu',plane_conv(name+'_phase',src,w));w=np.kron(np.eye(len(filters)),np.ones((1,8))*np.pi/8);return plane_conv(name+'_amplitude',part,w)

def twoarm_filter(weights):
    # convolution of a real/imag pair with a complex-valued radial filter
    w=np.asarray(weights);return np.stack([np.stack([w.real,-w.imag],1).ravel(),np.stack([w.imag,w.real],1).ravel()])

spirw=[];spirden=[];types=[]
for start in (0,1,2,3,4):
    ix=np.arange(start,start+5);tap=np.array([.55,.95,1.,.95,.55]);tap/=tap.sum()
    for pitch in (15,23,36,52):
        for chir in (1,-1):
            phase=2*chir*np.log(radii[ix]/radii[ix[2]])/np.tan(np.deg2rad(pitch));h=tap*np.exp(1j*phase);h-=tap*h.sum()
            w=np.zeros(nr,complex);w[ix]=h;spirw.append(twoarm_filter(w));w=np.zeros(nr);w[ix]=tap;spirden.append(w);types.append((start,pitch,chir))
sp=magnitudes('log_spirals',az,spirw);sden=plane_conv('spiral_continuum',ann,spirden)
# Bar: coherent m=2 in the central zone, followed by either a drop in
# amplitude or a different phase in the disk. Compare fractional rather
# than absolute Fourier amplitude so a luminous elliptical does not look
# barred just because its flux falls rapidly with radius.
barw=[];barden=[]
for start in [0,1,2,3]:
    w=np.zeros(nr);w[start:start+3]=[.25,.45,.3];barw.append(twoarm_filter(w));barden.append(w)
bar=magnitudes('straight_bar',az,barw);bd=plane_conv('bar_continuum',ann,barden)

# Vector fractional indicators in a plane. The following computes a / b
# using only affine maps and norm: the norm of [+a,-a,+b,-b] at that
# channel gives a/sqrt((a*a+b*b)/2 + eps). When a is attenuated relative
# to b this approximates division very accurately. Nonnegativity and a
# small floor on denominators make ratios stable in the noise.
# Implement scalar ratios together in a small photometric bank below;
# for the larger angular maps use homogeneous amplitude tests instead.

# Select the plane using a geometry cost. The mapping from loss of ellipse
# correlation to penalty is dimensionless: quality is divided by the
# matched light. A gate on absolute ring brightness suppresses sky noise.
# A single common reference for differences between spiral amplitudes.
def gated_modes(name,src,den,groups,threshold,amp_scale):
    k=dim[src][0];denpen=conv(name+'_flux_cost',den,(-threshold*np.eye(k))[:,:,None,None],-.0025)
    a=plus(name+'_excess',src,denpen)
    # Quality deficit measured independently for each plane. q<qplane[0]
    # still uses the closest available geometry.
    pg=reshape(name+'_corr',plane_fit,(1,1,nplane));bb=reshape(name+'_best',pbest,(1,1,1));bb=reshape(name+'_bcast',dense(name+'_broadcast',bb,-np.ones((nplane,1))),(1,1,nplane));diff=plus(name+'_fit_gap',pg,bb)
    cost=conv(name+'_geometry_cost',diff,(amp_scale*np.ones(k)).reshape(k,1,1,1));a=plus(name+'_plausible',a,cost)
    allm=[]
    for label,gp in groups:
        ids=np.array([i*nplane+j for i in gp for j in range(nplane)]);t=gather(name+'_'+label+'_entries',a,ids);t=reshape(name+'_'+label+'_grid',t,(1,1,len(ids)))
        # Hierarchical pooling, with factors of 8 planes each time. Scorer
        # stack cost remains small even on large input batches.
        t=pool(name+'_'+label+'_angles',t,(1,na),(1,na),'maxpool');t=pool(name+'_'+label+'_max',t,(1,len(ids)//na),(1,len(ids)//na),'maxpool');allm.append(raw(t))
    return allm
stight,sloose=gated_modes('winding',sp,sden,[('tight',[i for i,(_,p,_) in enumerate(types) if p<=23]),('loose',[i for i,(_,p,_) in enumerate(types) if p>23])],.035,.07)
barcue,=gated_modes('bar',bar,bd,[('coherent',range(4))],.12,.17)

# Counter-rotate the outer arm pattern back by its expected winding at
# each radius, and require a coherent direction over at least three
# adjacent rings. Flattening on its own can contribute mode 2 but can't
# account for a steadily revolving phase. An inter-ring difference removes
# the leading part of the radially constant projection quadrupole.
# A comparable *inner* straight quadrupole relative to an outer relaxed
# disk is evidence of a bar. Infer the outer plane separately to prevent
# a bar from defining its own deprojection and thereby hiding itself.
outerbank=[];ow=(1-weight_rad(6.5))*weight_rad(21);ow/=ow.sum()
for q in qplane:
    for ia in range(na):
        a=np.pi*ia/na;u=x*np.cos(a)+y*np.sin(a);v=-x*np.sin(a)+y*np.cos(a)
        for s in [5,9,15,23]:
            m=np.exp(-np.sqrt((u/s)**2+(v/(s*q))**2));dm=m-(m*ow).sum();outerbank.append(ow*dm/np.sqrt((ow*dm**2).sum()))
og=reshape('outer_grid',dense('disk_envelope',paired,outerbank),(nplane,1,4));og=pool('outer_plane',og,(1,4),(1,4),'maxpool');og=reshape('outer_planes',og,(1,1,nplane));ogmax=pool('best_outer_plane',og,(1,nplane),(1,nplane),'maxpool');ogminus=reshape('outer_best_map',dense('copy_outer_best',ogmax,-np.ones((nplane,1))),(1,1,nplane));ogap=plus('outer_deprojection_gap',og,ogminus)
# Bar channels measured against this independently inferred disk plane.
ob=plus('outer_frame_bar',bar,conv('outer_frame_geometry',ogap,1.8*np.ones((4,1,1,1))),plane_conv('outer_frame_light',bd,-.062*np.eye(4)));ob=reshape('outer_frame_candidates',ob,(1,4,nplane));ob=pool('outer_frame_bar_max',ob,(4,nplane),(4,nplane),'maxpool');outerbar=raw(ob)

# Scalar ratio bank. No statistics of an input batch are used; every
# normalization has four positions belonging to a *single* image/quantity.
ratio_specs=[]
def ratio(a,b,floor=.01,atten=.18):
    i=len(ratio_specs);ratio_specs.append((expr(a)*atten,expr(b)+floor,atten));return raw('photometric_ratios',i)

qnorm=[ratio(z,quality,.015,.12) for z in qresp];snorm=[ratio(z,quality,.015,.12) for z in sresp]
qraw=side_vote(qnorm,qs,.00085);logsize=side_vote(snorm,scales,.004,True)
pqnorm=[ratio(raw(pqfit,i),raw(pbest),.015,.12) for i in range(len(qs))];qpair=side_vote(pqnorm,qs,.0009);qhat=.60*qpair+.40*qraw
rad=[]
for ac,asi,f in radial_shape:rad.append([ratio(ac,f,.002),ratio(asi,f,.002)])
def length(v):return sum(absval(v[0]*np.cos(j*np.pi/6)+v[1]*np.sin(j*np.pi/6)) for j in range(6))*(np.pi/12)
radamp=[length(v) for v in rad]
# Difference as *vectors*: a change of PA is significant even if the
# amount of apparent flattening itself does not change.
bar_delta=length([rad[1][j]-rad[2][j] for j in range(2)])
outer_delta=length([rad[1][j]-rad[3][j] for j in range(2)])
ce=[];major2=[];minor2=[]
for area,mom,cx,cy in contour:
    a=length([cx,cy]);ce.append(ratio(a,mom,1.));major2.append(ratio((mom+a)/300,area,1.0,atten=.10)*600);minor2.append(ratio((mom-a)/300,area,1.0,atten=.10)*600)
# For a uniform elliptical isophote the major radius squared is
# 2*(trace + |quadrupole|)/area. Linear comparisons avoid square roots.
shape_core=ce[2]-.22*(ce[0]-ce[2]); outerflat=.68*ce[0]+.32*ce[1]

corecolor=ratio(coreR-coreB,(coreR+coreB)*.5,.015);envcolor=ratio(envR-envB,(envR+envB)*.5,.022)
corefrac=ratio(ap6,central,.016);widefrac=ratio(ap13,central,.018)
iso_concentration=ratio(contour[2][0],contour[0][0],1.)
# raw moment comparisons, expressed as eccentricity squared and radius
ell=ratio(anis,trace,.04,.13);ellcore=ratio(ancore,trcore,.025,.13)
A=[(ratio(t,f,.003),ratio(o,f,.003)) for t,o,f in asym]
lop=[length([ratio(cx,f,.007),ratio(cy,f,.007)]) for f,cx,cy in lopsided]
S=ratio(stight,ap6,.035);U=ratio(sloose,ap6,.035);B=ratio(barcue,ap6,.035)
growthN=ratio(growth_near,central,.07);growthF=ratio(growth_far,central,.07)
residual=ratio(30*varres,30*smoothvar,.003)
spinal=[ratio(h,central,.035) for h in lf];Bin=ratio(outerbar,ap6,.035)
extra_component=ratio(withbulge-unified,unified,.004)
plumes=[ratio(raw(plume,i),central,.06) for i in range(3)];fine=ratio((raw(high,2)+raw(high,3)+raw(high,4))*(2/3),ap6,.035)
secfrac=[ratio(z,central,.055) for z in nscores];tailmoment=length([ratio(tailx,tailmass,.03),ratio(taily,tailmass,.03)])
P=[]
for pk in peaks:
    root=pk[0];# analogous nucleus should carry a substantial fraction of the central surface brightness contrast
    P.append([ratio(z,central*.65+root*.35,.06) for z in pk]);

# Emit all photometric linear apertures now that they have been enumerated.
for src,M in masks.items():dense(src+'_apertures',src,M)

# Compile expressions in dependency order, sharing one dense matrix for
# all rows at each depth. The first phase only materializes intermediates
# required by the ratio bank. Layers retain all their named inputs.
known={k:v for k,v in sources.items() if v[0] not in ['photometric_ratios']}; serial=0

def emit_affine(name,rows):
    # Sum the partial projections from the different named branches.
    matrices={};bs=np.array([v.b for v in rows]);nodes=[]
    for j,v in enumerate(rows):
        for key,c in v.d.items():
            if c==0:continue
            src,i=known[key]
            if src not in matrices:matrices[src]=np.zeros((len(rows),math.prod(dim[src])),F)
            matrices[src][j,i]+=c
    for src,w in matrices.items():nodes.append(dense(name if len(matrices)==1 else name+'__'+src,src,w,bs if not nodes else 0))
    return nodes[0] if len(nodes)==1 else plus(name,*nodes)

def compile_for(rows):
    global serial
    needed=set()
    def include(k):
        if k in known or k in needed:return
        for parent in ops[k].d:include(parent)
        needed.add(k)
    for row in rows:
        for k in row.d:include(k)
    for h in sorted(set(depth[k] for k in needed)):
        keys=sorted(k for k in needed if depth[k]==h);serial+=1;nm='logic_'+str(serial);src=rel(nm,emit_affine(nm+'_affine',[ops[k] for k in keys]));known.update({k:(src,i) for i,k in enumerate(keys)})

rows=[v for a,b,_ in ratio_specs for v in [a,-a,b,-b]]; compile_for(rows);sr=emit_affine('paired_photometry',rows);sr=reshape('scalar_channels',sr,(len(ratio_specs),1,4));sr=node('unit_photometry','norm',sr,dim[sr]);sr=dense('photometric_ratios',sr,np.eye(len(rows),dtype=F)[np.arange(0,len(rows),4)]/np.array([np.sqrt(2)*d for _,_,d in ratio_specs])[:,None]);known.update({k:v for k,v in sources.items() if v[0]==sr})

# Hints in indicator units; all constants below express hand-selected
# morphological criteria. Positive group evidence takes a separate path
# from within-group discrimination, so the middle child of a three-way
# shape split is not penalized relative to a binary split.
# Combine model-based axis ratio with actual isophotal ellipticity
# (1-q^2)/(1+q^2), which is measured independently of radial profile.
flat_iso=membership(outerflat-.76,.135);thin=.47*membership(.34-qhat,.12)+.53*flat_iso
resolved=membership(major2[0]-16**2,150);extended=.5*membership(logsize-math.log(7),.5)+.5*resolved
edgeflat=membership(ell-.63,.18);roundcore=membership(ce[0]-ce[2]-.13,.14)
bluedisk=membership(-envcolor-.025,.29);redcore=membership(corecolor-.13,.29)
# Squashed plane responses indicate positive structure over the expected
# continuum m=2 response. Positive arm evidence distinguishes face-on disks
# from smooth systems without requiring a large blue colour excess.
arm=membership(max2(S,U)-.018,.027)
# A bar is more elongated than the outer disk and/or differs in PA.
# Compare the inner and outer angular vectors before measuring length. A
# projected axisymmetric disk usually becomes *more* elongated with radius.
barcontrast=.6*bar_delta+.4*outer_delta+.72*(radamp[1]-radamp[2]);
barred=.32*membership(B-.01,.045)+.44*membership(barcontrast-.085,.10)+.24*membership(shape_core-ce[0]-.01,.13)
outer_bar=membership(Bin-.053,.050);barred=.65*barred+.35*outer_bar
# Asymmetry can be caused by faint field sources: demand light close to the
# center, not only at the outermost aperture. Pairing tolerance handles
# small catalog offsets. Disturbances have a wide radial footprint.
asym_near=membership(.35*A[0][0]+.65*A[0][1]-.19,.14)
asym_body=membership(.25*A[1][0]+.75*A[1][1]-.27,.19)
asym_outer=membership(.15*A[2][0]+.85*A[2][1]-.37,.25)
# Compact paired peaks versus general tidal asymmetry.
peakscore=[];pairscore=[]
for h,pk in enumerate(P):
    # The longer baseline gives more contrast; a central smooth elliptical
    # can generate a shoulder near its center, so require small-scale peaks
    # too. Weak outer peaks are common in the unrelated sky population.
    ref=1 if h==0 else 1.65
    shoulder=vmax([pk[1]-.14*ref,pk[2]-.14*ref,pk[3]-.23*ref]);peakscore.append(membership(shoulder,.12*ref))
    oppos=vmax([min2(pk[4+i],pk[4+(i+4)%8]) for i in range(4)]);pairscore.append(membership(oppos-.13*ref,.1*ref))
second=.52*peakscore[0]+.48*peakscore[1]; dbl=.5*(pairscore[0]+pairscore[1]);
# Group scores have a common scale. Bright broad disks may have little
# colour contrast; an extended body plus a winding signal still suffices.
large=membership(logsize-math.log(8.5),.57); verylarge=membership(logsize-math.log(13),.55)
body=membership(widefrac-.215,.11); diffuse=membership(corefrac-.58,.18)
# A very thin but unresolved source can also be a smooth cigar. Classify
# flatness as an edge-on *disk* primarily when the major axis is resolved.
sizegate=membership(major2[0]-14**2,125); diskextent=membership(major2[0]-19**2,195)
# Disk evidence is conjunctive: resolving the diameter is necessary but
# not sufficient. This avoids calling large smooth ellipticals spirals.
# Outer isophotes of edge-ons stay thin even around a bright round bulge.
# The wings of a lenticular/bulged disk stay line-shaped below the
# brighter isophotes. This is not a property of an unresolved cigar.
wing=membership(radamp[3]-.56,.17);bulged_disk=min2(wing,max2(roundcore,membership((minor2[1]-minor2[0])-1,6)))
thin_wings=membership(spinal[0]-.34,.21); spine_excess=membership(spinal[1]-.36,.15); wing_disk=min2(wing,thin_wings)
edge=.86*thin+.56*flat_iso+.79*sizegate+.42*diskextent+.33*edgeflat+.20*large+.18*roundcore+.15*bluedisk+.48*bulged_disk+.55*wing_disk-.10*(1-thin_wings)-1.60
structure=max2(arm,.8*barred);spir=.60*extended+.39*body+.57*structure+.46*min2(extended,structure)+.38*bluedisk+.35*diskextent-1.12-.99*thin-.47*edgeflat
smooth=.71-.28*extended-.29*arm-.18*body-.17*bluedisk-.10*barred+.52*(1-sizegate)+.21*(1-structure)
rebright=membership(max2(growthN-.14,.8*growthF-.18),.14)
oddmode=membership(.22*lop[0]+.47*lop[1]+.31*lop[2]-.16,.125)
paired_nuc=membership(.48*secfrac[0]+.52*secfrac[1]-.16,.105)
# Invisible in ordinary moments but visible against the halo: score a
# positive imbalance relative to the peak, with a modest tolerance in
# the core to avoid reacting to a sub-pixel displacement.
pluminess=membership(vmax([plumes[0]-.29,plumes[1]-.185,plumes[2]-.115]),.115)
nonaxis=membership(fine-.020,.017)
nonell=membership(residual-.060,.054);tail_skew=membership(tailmoment-.17,.14)
# The residual measures departure from *any* centered exponential ellipse,
# so require an odd component as well before identifying it as disturbance.
# A pair of separate nuclei takes precedence over this asymmetry signal.
tidal=.26*asym_near+.24*asym_body+.19*oddmode+.18*tail_skew+.13*asym_outer
interact=max2(min2(paired_nuc,pluminess),min2(paired_nuc,asym_near));confirm=min2(paired_nuc,max2(rebright,dbl))
merger=.39*second+.25*dbl+.18*nonell+.15*redcore+.30*rebright+.60*paired_nuc+.90*interact+.66*confirm-1.28-.16*thin
dist=.84*tidal+.32*nonell+.39*min2(nonell,oddmode)+1.16*pluminess+.16*bluedisk-.70-.85*confirm-.19*thin
# Asymmetric residuals disfavor the single smooth model; point-symmetric
# departures (e.g., a bar) instead lend weaker support to a disk.
complex_disk=min2(max2(arm,barred),max2(nonaxis,nonell))
smooth=smooth-.18*min2(tidal,nonell)-.19*complex_disk
spir=spir+.26*complex_disk+.18*min2(nonaxis,extended)-.27*min2(tidal,nonell)-.16*(1-structure)
# Smooth galaxy flattening thresholds (major/minor axis photometry), and
# bulge evidence combining central reddening, concentration and rounding.
# The q thresholds refer to intrinsic (PSF-corrected) flattening.
s1=.70*membership(qhat-.80,.07)+.30*membership(.20-ce[1],.09); s2=.53*membership(.53-qhat,.07)+.47*membership(ce[1]-.56,.09);smid=min2(1-s1,1-s2)
# A prominent, rounder inner envelope compared with the thin outer disk
# is direct evidence of a bulge. A red disk alone is not sufficient.
rou=membership(.54-ce[2],.18);bulge=membership(.31*(redcore-.52)+.23*(roundcore-.48)+.14*(rou-.48)+.18*(.5-diffuse)+.12*(.5-bluedisk)+.20*(central-.34)+.22*(bulged_disk-.5),.22)
# Hubble sequence prior: tightly wound arms more often live in disks with
# a centrally concentrated old stellar population. Fractional isophote
# areas approximate the concentration without depending on angular size.
conc=membership(.225-iso_concentration,.125)
tight=membership((S-U)*9+.15*(conc-.5)+.18*(redcore-.50)+.16*(.47-bluedisk)+.12*(widefrac-.28)-.10*(nonaxis-.5)+.14*(.5-tidal),.14)
two_component=membership(extra_component-.0008,.0035)
bulge=membership(.41*(bulge-.49)+.14*(spine_excess-.46)+.10*(membership(spinal[2]-.10,.13)-.5)+.35*(two_component-.48)+.18*(redcore-bluedisk),.29)
edge=edge+.36*min2(two_component,wing)-.23*(1-wing)
spbar=membership(.86*(barred-.48)+.14*(redcore-.5)+.12*(B-max2(S,U)),.18)

def subclass(group,prefs):
    best=vmax(prefs);return [group+.85*(p-best) for p in prefs]
# A merger is a subtype of interaction, not an alternative to detecting
# tidal light. Thus a spectacular plume cannot by itself veto a merger.
# Diffuse, faint debris without a bright second condensation favors class 0;
# a double profile, a rebrightening or a compact, red off-axis condensation
# favors class 1. Use the signed evidence here; no template/class fitting.
interaction=max2(dist,merger)
pair_evidence=.26*paired_nuc+.22*second+.18*rebright+.13*dbl+.12*redcore+.09*asym_near
which_pair=membership(pair_evidence-.405,.16)
scores=subclass(interaction,[1-which_pair,which_pair])+subclass(smooth,[s1,smid,s2])+subclass(spir,[spbar,min2(1-spbar,tight),min2(1-spbar,1-tight)])+subclass(edge,[1-bulge,bulge]);compile_for(scores);emit_affine('morphology_scores',scores)
(out/'network.json').write_text(json.dumps({'layers':L},separators=(',',':')))
np.savez(out/'weights.npz',**W)
