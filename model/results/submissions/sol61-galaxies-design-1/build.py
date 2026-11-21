"""Data-free galaxy morphology network.

All images used in kernel construction below are mathematical functions on a
coordinate grid. No file is opened for reading, and no dataset is used.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

DT = np.float32
layers = []
weights = {}
shapes = {"image": (3, 128, 128)}


def node(name, op, inp, output_shape, **settings):
    layers.append(dict(name=name, op=op, inputs=[inp] if isinstance(inp, str) else inp, **settings))
    shapes[name] = tuple(output_shape)
    return name


def dense(name, inp, W, b=None):
    W = np.asarray(W, dtype=DT)
    assert W.shape[1] == math.prod(shapes[inp]), (name, W.shape, shapes[inp])
    weights[name + '.W'] = W
    weights[name + '.b'] = np.zeros(W.shape[0], DT) if b is None else np.broadcast_to(np.asarray(b, DT), (W.shape[0],)).copy()
    return node(name, 'dense', inp, (W.shape[0],), out=W.shape[0])


def conv(name, inp, W, b=None, padding=0):
    W = np.asarray(W, DT)
    o, c, k, _ = W.shape
    weights[name + '.W'] = W
    weights[name + '.b'] = np.zeros(o, DT) if b is None else np.broadcast_to(np.asarray(b, DT), (o,)).copy()
    _, h, w = shapes[inp]
    return node(name, 'conv', inp, (o, h+2*padding-k+1, w+2*padding-k+1), out=o, kernel=[k,k], stride=1, padding=padding)


def reshape(name, inp, shape):
    assert math.prod(shapes[inp]) == math.prod(shape)
    return node(name, 'reshape', inp, shape, shape=list(shape))


def relu(name, inp):
    return node(name, 'relu', inp, shapes[inp])


def pool(name, inp, size, stride=None, maximum=False):
    if stride is None:
        stride = size
    c,h,w = shapes[inp]
    a,b = size
    u,v = stride
    return node(name, 'maxpool' if maximum else 'avgpool', inp,
                (c,(h-a)//u+1,(w-b)//v+1), size=list(size), stride=list(stride))


def add(name, inputs):
    assert all(shapes[i] == shapes[inputs[0]] for i in inputs)
    return node(name, 'add', inputs, shapes[inputs[0]])


def maximum(name, inp):
    n = math.prod(shapes[inp])
    # Factor the reduction to avoid a very large pool window in the interpreter.
    a = min(256,math.isqrt(n))
    while a>1 and n%a:
        a-=1
    r = reshape(name+'_rows', inp, (1,a,n//a))
    r = pool(name+'_columns',r,(a,1),maximum=True)
    return pool(name,r,(1,n//a),maximum=True)


def combine(name, specs):
    """Concatenate small vectors, implemented with fixed scatter matrices."""
    sizes = [math.prod(shapes[s]) for s in specs]
    total = sum(sizes)
    parts = []
    start = 0
    for j,(s,n) in enumerate(zip(specs,sizes)):
        w = np.zeros((total,n), DT)
        w[start:start+n] = np.eye(n, dtype=DT)
        parts.append(dense(name+f'_part{j}', s, w))
        start += n
    return add(name, parts) if len(parts)>1 else parts[0]


def pwl_conv(name, inp, knots, vals):
    """Continuous linear spline for nonnegative scalar-valued feature maps."""
    knots = np.asarray(knots, float)
    vals = np.asarray(vals, float)
    slopes = np.diff(vals)/np.diff(knots)
    change = np.r_[slopes[0], np.diff(slopes), -slopes[-1]]
    h = conv(name+'_hinges', inp, np.ones((len(knots),1,1,1)), -knots)
    h = relu(name+'_positive', h)
    return conv(name, h, change.reshape(1,-1,1,1), vals[0])


def log_vector(name, inp, floor=0.0005):
    """Log(x+floor), with formula-generated logarithm spline knots."""
    n = math.prod(shapes[inp])
    knots = np.r_[0., floor*2.**np.arange(0,13)]
    vals = np.log(knots+floor)
    slopes = np.diff(vals)/np.diff(knots)
    changes = np.r_[slopes[0], np.diff(slopes), -slopes[-1]]
    W = np.zeros((n*len(knots),n), DT)
    B = np.zeros(n*len(knots), DT)
    for i in range(n):
        W[i*len(knots):(i+1)*len(knots),i] = 1
        B[i*len(knots):(i+1)*len(knots)] = -knots
    h = relu(name+'_relu', dense(name+'_hinges', inp, W, B))
    W = np.zeros((n,n*len(knots)), DT)
    for i in range(n):
        W[i,i*len(knots):(i+1)*len(knots)] = changes
    return dense(name, h, W, vals[0])


def clip_vector(name, inp, lo=-1., hi=1.):
    n = math.prod(shapes[inp])
    w = np.r_[np.eye(n),np.eye(n)]
    h = relu(name+'_relu', dense(name+'_hinges',inp,w,np.r_[np.full(n,-lo),np.full(n,-hi)]))
    return dense(name,h,np.c_[np.eye(n),-np.eye(n)],lo)


# ---------------------------------------------------------------------------
# Photometry and a centered, seeing-scale morphology image.
# ---------------------------------------------------------------------------
small = pool('seeing_pixels', 'image', (2,2))
gray = conv('luminance',small,np.array([.30,.45,.25]).reshape(1,3,1,1))
# A winsorized outer-field black-level estimate; it is an inference operation,
# not a number measured from the training set.
sky_h = conv('sky_clip_hinges',gray,np.ones((2,1,1,1)),[0.,-.12])
sky_h = relu('sky_clip_relu',sky_h)
sky_map = conv('sky_clip',sky_h,np.array([1.,-1.]).reshape(1,2,1,1))
y64,x64 = np.mgrid[:64,:64]
r64 = np.hypot((x64-31.5)*2,(y64-31.5)*2)
sky_w = ((r64>43)&(r64<61)).astype(float)
sky_w /= sky_w.sum()
sky = dense('sky_level',sky_map,sky_w.reshape(1,-1))
# Exact central 64 by 64 arc-pixel field, retaining two-pixel samples.
crop_w = np.zeros((1024,4096),DT)
for y in range(32):
    for x in range(32):
        crop_w[y*32+x,(y+16)*64+x+16] = 1
crop = dense('central_pixels',gray,crop_w)
bg = dense('central_black_level',sky,-np.ones((1024,1),DT),-.012)
crop = relu('above_sky',add('sky_subtracted',[crop,bg]))
crop = reshape('central_map',crop,(1,32,32))
knots = np.array([0.,.01,.025,.05,.10,.20,.40,.70,1.05])
comp = pwl_conv('dynamic_range',crop,knots,np.arcsinh(knots/.08)/np.arcsinh(1/.08))
yy,xx = np.mgrid[:32,:32]
X = (xx-15.5)*2
Y = (yy-15.5)*2
RR = np.hypot(X,Y)
def radial_weight(r):
    # Keep distant field objects from dominating a central-galaxy match.
    return np.exp(-(r/29.)**4)

taper = radial_weight(RR)
windowed = dense('central_window',comp,np.diag(taper.ravel()))
windowed = reshape('windowed_map',windowed,(1,32,32))
normed = node('contrast_normalized','norm',windowed,shapes[windowed])

# ---------------------------------------------------------------------------
# Formula-only smooth, disk, and disk+bulge models.
# Two by two quadrature and an approximate seeing PSF antialias every kernel.
# ---------------------------------------------------------------------------
OFF = [-.5,.5]
PSF = np.exp(-.5*(np.arange(-2,3)/.60)**2)
PSF /= PSF.sum()


def blur(a):
    p = np.pad(a,((0,0),(2,2)))
    b = sum(PSF[j]*p[:,j:j+32] for j in range(5))
    p = np.pad(b,((2,2),(0,0)))
    return sum(PSF[j]*p[j:j+32,:] for j in range(5))


def coords(q, theta, dx=0., dy=0.):
    c,s = math.cos(theta),math.sin(theta)
    x = (X+dx)*c+(Y+dy)*s
    y = (-(X+dx)*s+(Y+dy)*c)/q
    r = np.sqrt(x*x+y*y+1.e-12)
    return x,y,r,np.arctan2(y,x)


def sample(q, theta, fun):
    z = np.zeros((32,32),float)
    for dy in OFF:
        for dx in OFF:
            x,y,r,ph = coords(q,theta,dx,dy)
            z += fun(x,y,r,ph)/4
    return blur(z)*taper


def profile(q,theta,a,p):
    return sample(q,theta,lambda x,y,r,ph:np.exp(-(r/a)**p))


def unit(a):
    a = a.ravel().copy()
    a -= a.mean()
    return a/(np.linalg.norm(a)+1.e-14)


def row(a):
    return unit(a)/32.


angles = [j*math.pi/12 for j in range(12)]
scales = [2.,3.2,5.,7.5,11.,16.]
profiles = [.65,1.15,1.8]
q_values = [.13,.22,.32,.44,.58,.72,.86,1.]
groups = {k:[] for k in ('round','middle','cigar','thin','bulge','disk')}
for q in q_values:
    group = 'round' if q>=.8 else 'middle' if q>=.5 else 'cigar' if q>=.3 else 'thin'
    for theta in ([0.] if q==1. else angles):
        for a in scales:
            for p in profiles:
                v = row(profile(q,theta,a,p))
                groups[group].append(v)
                if q>=.44:
                    groups['disk'].append(v)
# Thin disks have a vertically confined stellar sheet. A bulge is circular,
# centrally concentrated light superposed on that sheet.
for q in [.12,.20,.30,.40]:
    for theta in angles:
        for a in [3.2,5.,7.5,11.,16.]:
            disk = sample(q,theta,lambda x,y,r,ph:
                          np.exp(-np.sqrt((x/a)**2+.25**2))/np.cosh(np.minimum(np.abs(y)/(1.5*a),50.))**2)
            if q<=.2:
                groups['thin'].append(row(disk))
            disk_flux = disk/(disk.sum()+1.e-14)
            for bulge_size in [.24,.45,.65]:
                bulge = profile(.85,theta,max(1.2,a*bulge_size),.85)
                bulge_flux = bulge/(bulge.sum()+1.e-14)
                for f in [.20,.40,.65]:
                    groups['bulge'].append(row((1-f)*disk_flux+f*bulge_flux))

# The same disk+bulge baselines used by the structured models must also be
# present among the smooth alternatives, so a bulge alone cannot imply arms.
def disk_bases(q,theta,a):
    disk1=profile(q,theta,a,.85)
    disk2=profile(q,theta,a,1.3)
    bulge=profile(.9,theta,max(1.2,.24*a),.9)
    disk1/=disk1.sum();disk2/=disk2.sum();bulge/=bulge.sum()
    return [disk1,disk2,.65*disk2+.35*bulge]

for q in [.44,.58,.72,.86,1.]:
    group='round' if q>=.8 else 'middle' if q>=.5 else 'cigar'
    for theta in ([0.] if q==1. else [j*math.pi/8 for j in range(8)]):
        for a in [3.2,5.,7.5,11.,16.]:
            for base in disk_bases(q,theta,a):
                groups[group].append(row(base))
                groups['disk'].append(row(base))
C = []
for label,bank in groups.items():
    response = dense('models_'+label,normed,np.asarray(bank,DT))
    C.append(maximum('match_'+label,response))
Cs = combine('shape_matches',C)
Cmax = maximum('best_shape_match',Cs)

# ---------------------------------------------------------------------------
# Complete analytic barred and spiral galaxies, compared with smooth models.
# Every kernel is normalized as a whole. This avoids amplifying near-zero,
# undersampled arm residuals. Model images are never taken from the data.
# ---------------------------------------------------------------------------
banks={'bar':[], 'tight':[], 'loose':[]}
for q in [.44,.58,.72,.86,1.]:
    for theta in ([0.] if q==1. else [j*math.pi/8 for j in range(8)]):
        for a in [3.2,5.,7.5,11.,16.]:
            bases=disk_bases(q,theta,a)
            for phase in [j*math.pi/8 for j in range(8)]:
                c,s=math.cos(phase),math.sin(phase)
                for length in [.6,1.,1.5]:
                    def barfun(x,y,r,ph):
                        u=x*c+y*s;v=-x*s+y*c
                        return np.exp(-.5*((u/(length*a))**4+(v/(.22*a))**2))
                    bar=sample(q,theta,barfun)
                    bar/=bar.sum()
                    for base in bases:
                        for f in [.12,.30]:
                            banks['bar'].append(row((1-f)*base+f*bar))
            for family,modes in [('tight',[(2,14.),(2,23.)]),
                                 ('loose',[(2,35.),(2,50.),(3,40.)])]:
                for m,pitch in modes:
                    alpha=m/math.tan(math.radians(pitch))
                    for hand in [-1.,1.]:
                        for phase in [j*math.pi/3 for j in range(6)]:
                            arm=sample(q,theta,lambda x,y,r,ph:
                                np.exp(-r/a)*(1-np.exp(-.5*(r/(.6*a))**4))*
                                (np.exp(1.4*np.cos(m*ph+hand*alpha*np.log(np.maximum(r/a,.12))-phase))-1.58))
                            normalization=sample(q,theta,lambda x,y,r,ph:np.exp(-r/a)).sum()
                            arm/=normalization
                            for base in bases:
                                for strength in [.35,.70]:
                                    banks[family].append(row(np.maximum(base+strength*arm,0.)))
H=[]
for family,bank in banks.items():
    z=dense('models_'+family,normed,np.asarray(bank,DT))
    z=maximum('strongest_'+family,z)
    correction=dense('shape_reference_'+family,Cmax,[[-1.]])
    z=add('conditional_'+family,[reshape('structure_scalar_'+family,z,(1,)),correction])
    H.append(z)
Hvec=combine('structure_strengths',H)
# Signed improvement is informative: an incompatible structured model should
# not be rescued by a colour prior alone.
Hpos=Hvec
armmax=maximum('strongest_arms',dense('arm_strengths',Hpos,[[0,1,0],[0,0,1]]))

# ---------------------------------------------------------------------------
# Rotation asymmetry at two physical apertures, allowing small centroid error.
# ---------------------------------------------------------------------------
asym_source = relu('asymmetry_noise_floor',conv('asymmetry_noise_subtraction',comp,np.ones((1,1,1,1)),-.025))
asymmap = pool('asymmetry_pixels',asym_source,(2,2))
y16,x16=np.mgrid[:16,:16]
r16=np.hypot((x16-7.5)*4,(y16-7.5)*4)
shifts=[(dy,dx) for dy in [-1,0,1] for dx in [-1,0,1]]
D=np.zeros((len(shifts)*2*256,256),DT)
for k,(dy,dx) in enumerate(shifts):
    for y in range(16):
        for x in range(16):
            j=y*16+x; yr=15-y+dy;xr=15-x+dx
            D[(2*k)*256+j,j]=1
            D[(2*k+1)*256+j,j]=-1
            if 0<=yr<16 and 0<=xr<16:
                D[(2*k)*256+j,yr*16+xr]=-1
                D[(2*k+1)*256+j,yr*16+xr]=1
symdiff=relu('absolute_opposite_differences',dense('opposite_differences',asymmap,D))
AF=[]
for j,sigma in enumerate([7.,14.]):
    w=np.exp(-.5*(r16/sigma)**2)*(1-np.exp(-.5*(r16/3.)**2));w/=w.sum()
    R=np.zeros((len(shifts),D.shape[0]),DT)
    for k in range(len(shifts)):
        R[k,2*k*256:(2*k+2)*256]=-np.tile(w.ravel(),2)
    allres=dense(f'asymmetry_residuals_{j}',symdiff,R)
    low=maximum(f'best_center_symmetry_{j}',allres)
    low=dense(f'asymmetry_{j}',low,[[-1.]])
    flux=dense(f'aperture_flux_{j}',asymmap,w.reshape(1,-1))
    AF.extend([low,flux])
AFvec=combine('asymmetry_photometry',AF)
AFlog=log_vector('asymmetry_logs',AFvec)
A=dense('log_asymmetry',AFlog,[[.65,-.65,.35,-.35]])

# ---------------------------------------------------------------------------
# Separate nuclei: compact positive curvature, central versus off-center.
# ---------------------------------------------------------------------------
grid=np.arange(-3,4)
gy,gx=np.meshgrid(grid,grid,indexing='ij')
g1=np.exp(-(gx*gx+gy*gy)/(2*.70**2));g1/=g1.sum()
g2=np.exp(-(gx*gx+gy*gy)/(2*1.7**2));g2/=g2.sum()
peaks=relu('nuclear_curvature',conv('nucleus_filter',crop,(g1-g2).reshape(1,1,7,7),padding=3))
peak_nodes=[]
for label,mask in [('central',np.exp(-.5*(RR/3.)**4)),
                   ('companion',(1-np.exp(-.5*(RR/4.)**4))*np.exp(-.5*(RR/17.)**4))]:
    masked=dense('nucleus_mask_'+label,peaks,np.diag(mask.ravel()))
    peak_nodes.append(maximum('nucleus_peak_'+label,masked))
# Same black-level-subtracted photometric units as the curvature image.
w=np.exp(-.5*(RR/12.)**2);w/=w.sum()
meanlight=dense('mean_central_light',crop,w.reshape(1,-1))
npvec=combine('nucleus_photometry',peak_nodes+[meanlight])
nplog=log_vector('nucleus_logs',npvec)
nuc_ratios=dense('nucleus_ratios',nplog,[[-1,1,0],[0,1,-1]])
nuc_ratios=clip_vector('bounded_nucleus_ratios',nuc_ratios,-2.,2.)

# ---------------------------------------------------------------------------
# Central and disk colour. Fluxes have an outer-field local sky subtraction.
# ---------------------------------------------------------------------------
colour_small=pool('colour_pixels','image',(4,4))
y32,x32=np.mgrid[:32,:32]
r32=np.hypot((x32-15.5)*4,(y32-15.5)*4)
outer=np.exp(-.5*((r32-48.)/8.)**2);outer/=outer.sum()
phot=[]
for sigma in [5.,13.]:
    w=np.exp(-.5*(r32/sigma)**2)
    if sigma>10:
        w*=1-np.exp(-.5*(r32/5.)**2)
    w/=w.sum()
    for ch in [0,2]:
        W=np.zeros((3,32,32),float);W[ch]=w-outer
        phot.append(W.ravel())
colour_flux=relu('positive_colour_flux',dense('colour_apertures',colour_small,phot))
colour_log=log_vector('colour_logs',colour_flux)
# Blue: +1 for a blue disk, -1 for a red old stellar population.
# Gradient: a red central bulge with a bluer surrounding stellar disk.
colour_vars=dense('physical_colours',colour_log,
                  [[0,0,-1/.6,1/.6],[1/.30,-1/.30,-1/.30,1/.30]], [.55/.6,0.])
colour_vars=clip_vector('bounded_colours',colour_vars)

# A bar has an elongated inner light distribution embedded in a less
# elongated disk. Compute dimensionless quadrupoles in two radial windows,
# using 12 analytic angular projections to approximate quadrupole magnitude.
quad=[]
phi=np.arctan2(Y,X)
for j,(radius,width) in enumerate([(4.,3.),(11.,5.)]):
    w=np.exp(-.5*((RR-radius)/width)**2)
    w/=w.sum()
    total=dense(f'quadrupole_flux_{j}',comp,w.reshape(1,-1))
    angular=[]
    for t in range(12):
        angular.append((w*np.cos(2*phi-t*math.pi/6)).ravel())
    angular=dense(f'quadrupole_directions_{j}',comp,angular)
    amplitude=maximum(f'quadrupole_amplitude_{j}',angular)
    qlog=log_vector(f'quadrupole_logs_{j}',combine(f'quadrupole_pair_{j}',[amplitude,total]))
    quad.append(dense(f'log_ellipticity_{j}',qlog,[[1.,-1.]]))
quadvec=combine('ellipticity_windows',quad)
bar_excess=dense('inner_bar_excess',quadvec,[[1.,-1.]],-math.log(1.5))
bar_excess=clip_vector('bounded_bar_excess',bar_excess,0.,1.)
inner_evidence=dense('absolute_bar_ellipticity',quadvec,[[1.,0.]],-math.log(.22))
inner_evidence=clip_vector('bounded_inner_ellipticity',inner_evidence,0.,1.)
bar_difference=add('bar_evidence_difference',[bar_excess,dense('negative_inner_evidence',inner_evidence,[[-1.]])])
bar_difference=relu('bar_evidence_excess',bar_difference)
bar_excess=add('supported_bar_excess',[bar_excess,dense('negative_bar_excess',bar_difference,[[-1.]])])
# Strong outer flattening is incompatible with a clearly seen face-on spiral.
outer_flat=dense('outer_flattening',quadvec,[[0.,1.]],-math.log(.45))
outer_flat=clip_vector('bounded_outer_flattening',outer_flat,0.,1.)

# ---------------------------------------------------------------------------
# Hand-written physical decision rules; no learned readout or empirical priors.
# ---------------------------------------------------------------------------
features=combine('morphology_features',[Cs,reshape('global_match_scalar',Cmax,(1,)),Hpos,
                                      reshape('arm_scalar',armmax,(1,)),A,nuc_ratios,colour_vars,bar_excess,outer_flat])
# Feature positions: six shape matches, global match, three structures,
# maximum arm strength, log asymmetry, two nucleus ratios, blue, bulge colour.
# 0 round,1 middle,2 cigar,3 thin,4 bulge,5 disk,6 best,
# 7 bar,8 tight,9 loose,10 arms,11 logA,12 off/center,13 off/mean,14 blue,15 gradient.
W=np.zeros((10,18),float);b=np.zeros(10,float)
# Smooth ellipsoidal populations.
for cls,shape in [(2,0),(3,1),(4,2)]:
    W[cls,shape]=25.;W[cls,6]=-25.
    W[cls,14]=-.45
    W[cls,7]=-.8
    b[cls]=.10
b[4]-=.05
# Thin stellar sheets, with or without a central bulge.
for cls,shape in [(8,3),(9,4)]:
    W[cls,shape]=25.;W[cls,6]=-25.
    W[cls,10]=-.25
    b[cls]=.05
W[8,14]=.20;W[8,15]=-.15
W[9,14]=-.25;W[9,15]=.15;b[9]=.10
# Bars and spiral winding. The three structure features already subtract
# the best smooth match: penalizing a structured galaxy a second time for
# not matching a smooth disk would incorrectly suppress strong structure.
W[5,7]=35.;W[5,14]=.10;W[5,16]=.60;W[5,17]=-1.2;b[5]=-.05
W[6,8]=35.;W[6,14]=.40;W[6,15]=.25;W[6,17]=-1.2;b[6]=.05
W[7,9]=35.;W[7,14]=.50;W[7,15]=-.15;W[7,17]=-1.2;b[7]=-.05
# Tidal disruption versus multiple compact nuclei.  A substantial shape
# mismatch adds independent evidence for disturbance.
W[0,11]=1.8;W[0,6]=-2.;W[0,13]=-.20;W[0,14]=.25;b[0]=-1.8*math.log(.22)+2*.8
W[1,11]=1.5;W[1,12]=1.0;W[1,13]=.65;W[1,14]=-.10;b[1]=-1.5*math.log(.22)+.35
dense('class_scores',features,W,b)

out=Path(sys.argv[1] if len(sys.argv)>1 else '.')
out.mkdir(parents=True,exist_ok=True)
(out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
np.savez(out/'weights.npz',**weights)
print('Built',len(layers),'layers;',sum(v.size for v in weights.values()),'float32 weights.')
