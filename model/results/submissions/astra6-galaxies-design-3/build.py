"""Data-free analytic galaxy morphology network.

No data are read. All templates are mathematical Sersic-like ellipses;
all feature maps and decision rules are specified below by hand.
"""
import sys
import json
from pathlib import Path
import numpy as np

DT = np.float32
layers, weights, dims = [], {}, {'image': (3, 128, 128)}


def layer(name, op, src, **kw):
    layers.append(dict(name=name, op=op, inputs=src if isinstance(src, list) else [src], **kw))
    return name


def dense(name, src, W, b=0):
    W = np.asarray(W, dtype=DT)
    weights[name+'.W'] = W
    weights[name+'.b'] = np.broadcast_to(np.asarray(b, dtype=DT), (len(W),)).copy()
    layer(name, 'dense', src, out=len(W))
    dims[name] = (len(W),)
    return name


def conv(name, src, W, b=0, padding=0):
    W = np.asarray(W, dtype=DT)
    weights[name+'.W'] = W
    weights[name+'.b'] = np.broadcast_to(np.asarray(b, dtype=DT), (len(W),)).copy()
    layer(name, 'conv', src, out=len(W), kernel=list(W.shape[-2:]), stride=1, padding=padding)
    return name


class E:
    def __init__(self, terms=None, bias=0.):
        self.terms = terms or {}
        self.bias = float(bias)
    def __add__(self, other):
        other = expr(other)
        t = self.terms.copy()
        for k,v in other.terms.items():
            t[k] = t.get(k, 0.)+v
        return E({k:v for k,v in t.items() if v}, self.bias+other.bias)
    __radd__ = __add__
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + -expr(other)
    def __rsub__(self, other):
        return expr(other) + -self
    def __mul__(self, c):
        return E({k:v*c for k,v in self.terms.items()}, self.bias*c)
    __rmul__ = __mul__
    def __truediv__(self, c):
        return self*(1/c)


def expr(v):
    return v if isinstance(v, E) else E(bias=v)


def refs(src, n=None):
    return [E({(src,i):1.}) for i in range(n if n is not None else int(np.prod(dims[src])))]


def affine(name, expressions, relu=False):
    es = [expr(e) for e in expressions]
    sources = sorted({s for e in es for s,i in e.terms})
    if not sources:
        sources = ['peak']
    parts = []
    for j, src in enumerate(sources):
        W = np.zeros((len(es), int(np.prod(dims[src]))), DT)
        for row,e in enumerate(es):
            for (s,i),v in e.terms.items():
                if s == src:
                    W[row,i] = v
        nm = name+'_p'+str(j)
        dense(nm, src, W, [e.bias for e in es] if j == 0 else 0)
        parts.append(nm)
    src = parts[0] if len(parts)==1 else layer(name+'_sum','add',parts)
    dims[src] = (len(es),)
    if relu:
        src = layer(name, 'relu', src)
        dims[src] = (len(es),)
    return refs(src)


def divide(name, numerators, denominators):
    # A norm dominated by +/- 256*d implements x/d without a learned layer.
    es = []
    for x,d in zip(numerators, denominators):
        es += [x, -expr(x), 256*expr(d), -256*expr(d)]
    packed = affine(name+'_pack', es)
    src = next(iter(packed[0].terms))[0]
    n = len(numerators)
    layer(name+'_shape', 'reshape', src, shape=[n,1,4])
    layer(name+'_norm', 'norm', name+'_shape')
    dims[name+'_norm'] = (n,1,4)
    W = np.zeros((n,4*n), DT)
    W[np.arange(n),4*np.arange(n)] = 256/np.sqrt(2)
    return refs(dense(name, name+'_norm', W))


def positive(name, xs):
    return affine(name, xs, relu=True)


def clip01(name, xs):
    p = positive(name+'_hinges', [e for x in xs for e in (expr(x),expr(x)-1)])
    return affine(name, [p[2*i]-p[2*i+1] for i in range(len(xs))])


def maximum(name, src, n, group):
    layer(name+'_shape','reshape',src,shape=[n,1,group])
    layer(name,'maxpool',name+'_shape',size=[1,group],stride=[1,group])
    dims[name]=(n,1,1)
    return refs(name,n)


def seeing(f):
    # A fixed, approximately two-original-pixel PSF, including binning and
    # the deliberate low-pass filter. This is not measured from any image.
    z=np.arange(7)-3
    g=np.exp(-z*z/(2*.95**2)); g/=g.sum()
    pad=np.pad(f,((3,3),(3,3)))
    v=sum(g[i]*pad[i:i+f.shape[0],:] for i in range(7))
    return sum(g[i]*v[:,i:i+f.shape[1]] for i in range(7))


# ---------- Low-pass colour images and a centred luminance field ----------
layer('bin2','avgpool','image',size=[2,2],stride=[2,2])
g = np.array([1,2,1],dtype=DT)/4
G = np.outer(g,g)
W = np.zeros((3,3,3,3),DT)
for c in range(3):
    W[c,c]=G
conv('colour_smooth','bin2',W,padding=1)
conv('luminance','colour_smooth',np.array([.28,.44,.28],DT).reshape(1,3,1,1))
N = 40
coords = np.arange(N)-(N-1)/2
y,x = np.meshgrid(coords,coords,indexing='ij')
r2 = x*x+y*y
r = np.sqrt(r2)
taper = np.exp(-.5*(r/18)**4)
W = np.zeros((N*N,64*64), DT)
for iy in range(N):
    for ix in range(N):
        W[iy*N+ix,(iy+12)*64+ix+12] = taper[iy,ix]
dense('central_field','luminance',W)
inds = np.flatnonzero((np.abs(x)<5)&(np.abs(y)<5))
W=np.zeros((len(inds),N*N),DT)
W[np.arange(len(inds)),inds]=1
dense('peak_samples','central_field',W)
peak=maximum('peak','peak_samples',1,len(inds))[0]
pix=refs('central_field')
clean=positive('sky_removed',[p-.018-.02*peak for p in pix])
clean_src=next(iter(clean[0].terms))[0]
dims[clean_src]=(N*N,)
layer('galaxy','reshape',clean_src,shape=[1,N,N])
dims['galaxy']=(1,N,N)
excess=positive('core_excess',[v-.30*peak for v in clean])
clipped=affine('disk_field',[a-b for a,b in zip(clean,excess)])
clip_src=next(iter(clipped[0].terms))[0]
layer('disk','reshape',clip_src,shape=[1,N,N])

# ---------- Analytic, rotation-marginalised shape templates ----------
# Discrete radii, flattenings and radial exponents are a physical model grid,
# not samples, centroids or statistics of any observed galaxies.
qs=np.array([.16,.26,.38,.52,.68,.84,1.0])
scales=[1.0,1.5,2.2,3.3,5.0,7.5,11.0]
powers=[.65,1.0,2.0]
angles=np.arange(24)*np.pi/24
models=[[],[]]
for q in qs:
    for a in scales:
        for p in powers:
            for angle in angles:
                u=np.cos(angle)*x+np.sin(angle)*y
                v=-np.sin(angle)*x+np.cos(angle)*y
                rr=np.sqrt(u*u+(v/q)**2+.04)
                f=seeing(np.exp(-((rr/a)**p)))
                f=f/f.max()
                f=np.maximum(f-.035,0)*taper
                for j,img in enumerate((f,np.minimum(f,.30))):
                    z=img-img.mean()
                    z=z/(np.sqrt(np.sum(z*z))*N)
                    models[j].append(z.ravel())
# Group the same analytic ellipses by half-maximum radius as well as q.
# Padding permits max pooling, without any large permutation matrix.
size_indices=[[],[]]
for si,s in enumerate(scales):
    for pi,p in enumerate(powers):
        group=0 if s*np.log(2)**(1/p)<2.5 else 1
        size_indices[group].extend(range((si*len(powers)+pi)*len(angles),(si*len(powers)+pi+1)*len(angles)))
size_width=max(map(len,size_indices))
shape_q=[]
fit=[]
for j,src in enumerate(['galaxy','disk']):
    prefix=['core_shape','disk_shape'][j]
    layer(prefix+'_norm','norm',src)
    M=np.zeros((len(qs)*2*size_width,N*N),DT)
    bias=np.full(len(M),-1.,DT)
    for qi in range(len(qs)):
        for group, ids in enumerate(size_indices):
            lo=(qi*2+group)*size_width
            inds=np.array(ids)+qi*len(scales)*len(powers)*len(angles)
            M[lo:lo+len(ids)]=np.array(models[j],DT)[inds]
            bias[lo:lo+len(ids)]=0
    dense(prefix+'_templates',prefix+'_norm',M,bias)
    sz=maximum(prefix+'_sizemax',prefix+'_templates',len(qs)*2,size_width)
    qm=maximum(prefix+'_qmax',prefix+'_sizemax',len(qs),2)
    best=maximum(prefix+'_best',prefix+'_qmax',1,len(qs))[0]
    rel=divide(prefix+'_relative',qm,[best+.00001]*len(qs))
    gates=positive(prefix+'_select',[1-1000*(1-v) for v in rel])
    qhat=divide(prefix+'_axisratio',[sum(float(q)*v for q,v in zip(qs,gates))],[sum(gates)+.00001])[0]
    shape_q.append(qhat)
    fit.append(best)
qcore,qdisk=shape_q

# Multi-component profiles: an exponential disk, compact round bulge, and
# optionally a projected bar or two logarithmic arms. Flux fractions are
# declared physical hypotheses, never estimates from observed images.
def unit(a):
    return a/(a.sum()+1e-20)


def render(disk, bulge, bf, extra=None, ef=0.):
    f=(1-bf-ef)*unit(disk)+bf*unit(bulge)
    if extra is not None:
        f+=ef*unit(extra)
    f=seeing(f)
    f=f/f.max()
    return np.maximum(f-.035,0)*taper


def model_bank(name, fs):
    raw, clipped=[],[]
    for f in fs:
        for dest,z in [(raw,f),(clipped,np.minimum(f,.30))]:
            z=z-z.mean()
            dest.append((z/(np.sqrt(np.sum(z*z))*N)).ravel())
    dense(name+'_core','core_shape_norm',.40*np.array(raw,DT))
    dense(name+'_disk','disk_shape_norm',.60*np.array(clipped,DT))
    layer(name,'add',[name+'_core',name+'_disk'])
    dims[name]=(len(raw),)
    return name,len(raw)


def components():
    for q in qs:
        for bf in [0.,.15,.35,.60]:
            for s in [1.8,2.7,4.,6.,9.,13.]:
                for angle in angles:
                    u=np.cos(angle)*x+np.sin(angle)*y
                    v=-np.sin(angle)*x+np.cos(angle)*y
                    rho=np.sqrt(u*u+(v/q)**2+.04)
                    disk=np.exp(-rho/s)
                    bq=np.sqrt(.5*(1+q*q))
                    rb=np.sqrt(u*u+(v/bq)**2+.04)
                    bulge=np.exp(-rb/(.20*s))
                    yield render(disk,bulge,bf)

src,n=model_bank('bulge_disk_models',components())
comp=maximum('component_groups',src,28,6*24)
comp_q=maximum('component_q','component_groups',7,4)
oldcore=refs('core_shape_qmax',7)
olddisk=refs('disk_shape_qmax',7)
qboth=affine('all_q_matches',[v for i in range(7) for v in (.4*oldcore[i]+.6*olddisk[i],comp_q[i])])
qsrc=next(iter(qboth[0].terms))[0]
allq=maximum('all_q',qsrc,7,2)
best_all=maximum('all_q_best','all_q',1,7)[0]
rel=divide('all_q_relative',allq,[best_all+.00001]*7)
gates=positive('all_q_select',[1-500*(1-v) for v in rel])
qdisk=divide('component_axisratio',[sum(float(q)*v for q,v in zip(qs,gates))],[sum(gates)+.00001])[0]
component_best=maximum('component_best','component_q',1,7)[0]


def bars():
    for q in [.38,.52,.68,.84,1.]:
        for s in [2.2,3.3,5.,7.5,11.]:
            for bf in [.05,.20]:
                for ef in [.30]:
                    for length in [.8,1.4]:
                        for phase in [0.,np.pi/4,np.pi/2]:
                            for angle in np.arange(16)*np.pi/16:
                                u=np.cos(angle)*x+np.sin(angle)*y
                                v=(-np.sin(angle)*x+np.cos(angle)*y)/q
                                rho=np.sqrt(u*u+v*v+.04)
                                disk=np.exp(-rho/s)
                                bulge=np.exp(-np.sqrt(r2+.04)/(.20*s))
                                ub=np.cos(phase)*u+np.sin(phase)*v
                                vb=-np.sin(phase)*u+np.cos(phase)*v
                                bar=np.exp(-.5*((ub/(length*s))**2+(vb/(.22*length*s))**2))
                                yield render(disk,bulge,bf,bar,ef)

src,n=model_bank('bar_models',bars())
bar_best=maximum('bar_best',src,1,n)[0]


def spirals():
    for q in [.38,.52,.68,.84,1.]:
        for s in [2.2,3.3,5.,7.5,11.]:
            for bf in [.10,.30]:
                for pitch in [12.,28.]:
                    for phase in [0.,np.pi/3,2*np.pi/3]:
                        for angle in np.arange(16)*np.pi/16:
                            u=np.cos(angle)*x+np.sin(angle)*y
                            v=(-np.sin(angle)*x+np.cos(angle)*y)/q
                            rho=np.sqrt(u*u+v*v+.04)
                            phi=np.arctan2(v,u)
                            disk=np.exp(-rho/s)
                            bulge=np.exp(-np.sqrt(r2+.04)/(.20*s))
                            winding=2*(phi-np.log(rho/s+.05)/np.tan(pitch*np.pi/180))+phase
                            arm=disk*(1+.85*np.cos(winding)*(1-np.exp(-rho*rho/(.6*s)**2)))
                            f=render(arm,bulge,bf)
                            yield f
                            yield f[:,::-1]  # both winding directions, no sky parity preference

src,n=model_bank('spiral_models',spirals())
spiral_best=maximum('spiral_best',src,1,n)[0]
smooth_best=.4*fit[0]+.6*fit[1]
model_evidence=clip01('model_evidence',[
    (component_best-smooth_best-.003)/.035,
    (bar_best-component_best-.003)/.045,
    (bar_best-smooth_best-.003)/.045,
    (spiral_best-smooth_best-.003)/.035,
    (spiral_best-component_best-.003)/.035,
])
model_bulge,bar_over_disk,bar_over_smooth,arms_over_smooth,arms_over_disk=model_evidence
bar_weak=positive('bar_template_min',[bar_over_disk-bar_over_smooth])[0]
bar_model=bar_over_disk-bar_weak
arm_weak=positive('arm_template_min',[arms_over_disk-arms_over_smooth])[0]
arm_model=arms_over_disk-arm_weak


def pairs():
    for s in [1.,1.7,2.8]:
        for separation in [3.,5.,8.,12.]:
            for relative in [.4,1.]:
                for origin in [0.,.5]:
                    for angle in np.arange(24)*2*np.pi/24:
                        dx,dy=separation*np.cos(angle),separation*np.sin(angle)
                        ra=np.sqrt((x+origin*dx)**2+(y+origin*dy)**2+.04)
                        rb=np.sqrt((x-(1-origin)*dx)**2+(y-(1-origin)*dy)**2+.04)
                        for profile in [.65,1.]:
                            for secondary_scale in [.65,1.]:
                                f=np.exp(-(ra/s)**profile)+relative*np.exp(-(rb/(s*secondary_scale))**profile)
                                f=seeing(f); f/=f.max()
                                yield np.maximum(f-.035,0)*taper

src,n=model_bank('pair_models',pairs())
pair_best=maximum('pair_best',src,1,n)[0]
pair_model=clip01('pair_evidence',[(pair_best-best_all-.008)/.06])[0]

# ---------- Photometric moments, concentration and asymmetry ----------
# All apertures below are analytic and centred on the specified image centre.
radii=[2.,3.5,6.,10.,16.]
W=[]
moment_keys=[]
for s in radii:
    ap=np.exp(-r2/(2*s*s))
    for key,z in [('flux',ap),('trace',ap*r2),('quadx',ap*(x*x-y*y)),('quady',ap*(2*x*y))]:
        moment_keys.append((s,key))
        W.append(z.ravel())
dense('moments','galaxy',W)
mom=dict(zip(moment_keys,refs('moments')))
num,den,ratio_names=[],[],[]
for s in radii:
    for key in ['quadx','quady']:
        num.append(mom[s,key]); den.append(mom[s,'trace']+.0001); ratio_names.append((s,key))
    num.append(mom[s,'trace']); den.append(mom[s,'flux']+.0001); ratio_names.append((s,'radius2'))
rat=dict(zip(ratio_names,divide('moment_ratios',num,den)))
oriented=[]
for s in radii:
    for theta in np.arange(16)*2*np.pi/16:
        oriented.append(np.cos(theta)*rat[s,'quadx']+np.sin(theta)*rat[s,'quady'])
ov=affine('oriented_ellipticity',oriented)
ovsrc=next(iter(ov[0].terms))[0]
ell=maximum('ellipticity',ovsrc,len(radii),16)
ell=dict(zip(radii,ell))
con=divide('concentrations', [mom[2.,'flux'],mom[3.5,'flux'],mom[6.,'flux']],
           [mom[6.,'flux']+.0001,mom[10.,'flux']+.0001,mom[16.,'flux']+.0001])
# Suppress the bright centre and measure the orientation of low-isophote light.
outer_ap=np.exp(-r2/(2*12**2))*(1-np.exp(-r2/(2*3**2)))
outerW=np.array([(outer_ap*z).ravel() for z in (r2,x*x-y*y,2*x*y,x,y,np.ones_like(x))])
dense('outer_moments','disk',outerW)
om=refs('outer_moments')
orr=divide('outer_ratios',[om[1],om[2]],[om[0]+.0001]*2)
od=affine('outer_directions',[np.cos(t)*orr[0]+np.sin(t)*orr[1] for t in np.arange(24)*2*np.pi/24])
odsrc=next(iter(od[0].terms))[0]
outer_e=maximum('outer_ellipticity',odsrc,1,24)[0]
# Isophotal light fractions require no hard segmentation of faint outskirts.
isoph=positive('isophotes',[p-t*peak for t in [.15,.40,.70] for p in pix])
issrc=next(iter(isoph[0].terms))[0]
W=np.zeros((3,3*N*N),DT)
for i in range(3):
    W[i,i*N*N:(i+1)*N*N]=np.exp(-r2/(2*10**2)).ravel()
dense('isophotal_flux',issrc,W)
iso=refs('isophotal_flux')
iso_ratio=divide('isophotal_ratios',[iso[1],iso[2]],[iso[0]+.0001]*2)
edge_evidence=clip01('outer_edge_evidence',[(outer_e-.55)/.30])[0]
# A minor/major ratio for Gaussian moments, with broad inclination bands.
outer_roundness=1-outer_e
outer_bar=clip01('outer_bar_evidence',[(ell[3.5]-outer_e-.08)/.25])[0]

W=np.zeros((N*N,N*N),DT)
for i in range(N*N):
    W[i,i]=1
    W[i,N*N-1-i]-=1
dense('rotation_difference','galaxy',W)
layer('positive_asymmetry','relu','rotation_difference')
aps=[np.exp(-r2/(2*s*s)).ravel() for s in [6.,10.,16.]]
dense('asymmetry_sums','positive_asymmetry',2*np.array(aps))
asym=divide('asymmetry',refs('asymmetry_sums'),[mom[s,'flux']+.0001 for s in [6.,10.,16.]])

# Outward increases in surface brightness violate smooth elliptical profiles.
# Bilinear sampling along a radial ray is a fixed linear map.
W=np.zeros((N*N,N*N),DT)
for iy in range(N):
    for ix in range(N):
        row=iy*N+ix
        W[row,row]=1
        sy=.76*y[iy,ix]+(N-1)/2
        sx=.76*x[iy,ix]+(N-1)/2
        iy0,ix0=int(np.floor(sy)),int(np.floor(sx))
        fy,fx=sy-iy0,sx-ix0
        for dy,wy in [(0,1-fy),(1,fy)]:
            for dx,wx in [(0,1-fx),(1,fx)]:
                W[row,(iy0+dy)*N+ix0+dx]-=wy*wx
dense('radial_difference','galaxy',W)
rad=positive('radial_increases',[v-.008*peak for v in refs('radial_difference')])
radsrc=next(iter(rad[0].terms))[0]
radaps=[np.exp(-r2/(2*s*s))*(1-np.exp(-r2/8)) for s in [6.,10.,16.]]
dense('radial_sums',radsrc,[v.ravel() for v in radaps])
dense('annular_flux','galaxy',[v.ravel() for v in radaps])
texture=divide('radial_texture',refs('radial_sums'),[v+.0001 for v in refs('annular_flux')])

# Nucleus/companion evidence at a scale somewhat broader than the survey PSF.
a=np.arange(5)-2
gg=np.exp(-a*a/(2*1.1**2)); gg=gg/gg.sum()
k=-np.outer(gg,gg); k[2,2]+=1
conv('nuclear_dog','galaxy',k.reshape(1,1,5,5),padding=2)
inds=np.flatnonzero((r>2.5)&(r<10.))
W=np.zeros((len(inds),N*N),DT)
W[np.arange(len(inds)),inds]=np.exp(-r.ravel()[inds]**2/(2*10**2))
dense('offcentre_samples','nuclear_dog',W)
off=maximum('offcentre_peak','offcentre_samples',1,len(inds))[0]
centre_kernel=np.exp(-r2/(2*1.0**2)); centre_kernel/=centre_kernel.sum()
dense('central_brightness','galaxy',centre_kernel.reshape(1,-1))
central=refs('central_brightness')[0]
peakrat=divide('peak_ratios',[off,central],[peak+.0001,peak+.0001])

# Colour contrast, with a sky-cancelling outer Gaussian for each aperture.
yc,xc=np.meshgrid(np.arange(64)-31.5,np.arange(64)-31.5,indexing='ij')
rc2=xc*xc+yc*yc
colour_kernels=[]
for s in [3.,6.,11.]:
    w=np.exp(-rc2/(2*s*s))
    sky=np.exp(-rc2/(2*23**2))-np.exp(-rc2/(2*17**2))
    w-=sky*(w.sum()/sky.sum())
    colour_kernels.append(w)
# An annular colour gives disk light more influence than a bright red nucleus.
w=np.exp(-rc2/(2*11**2))-np.exp(-rc2/(2*4**2))
sky=np.exp(-rc2/(2*23**2))-np.exp(-rc2/(2*17**2))
w-=sky*(w.sum()/sky.sum())
colour_kernels.append(w)
W=[]
for w in colour_kernels:
    for c in range(3):
        z=np.zeros((3,64,64),DT); z[c]=w; W.append(z.ravel())
dense('colour_integrals','colour_smooth',W)
col=refs('colour_integrals')
colp=positive('positive_colours',col)
redness=divide('red_blue_colour',[col[3*i]-col[3*i+2] for i in range(4)],
               [colp[3*i]+colp[3*i+2]+.03 for i in range(4)])

# ---------- Direct physical model evidence, conditioned on image size ----------
# A profile with a round nucleus and a flatter exponential envelope can have a
# nearly round total-light ellipse. Compare the full models rather than just
# interpolating the winning ellipse's q value.
# Samples used to marginalise shape are generated below, never read from data.
# Matrix-free group maxima of the component grid.
# component_groups is ordered by axis ratio, then bulge fraction.
ci=refs('component_groups',28)
crows=affine('physical_component_order',[
    ci[q*4+bf] for group in [
        [(q,b) for q in range(2) for b in [0,1]],
        [(q,b) for q in range(2) for b in [2,3]],
        [(q,b) for q in range(2,7) for b in [0,1]],
        [(q,b) for q in range(2,7) for b in [2,3]],
    ] for q,bf in group
])
# Groups have different sizes: use explicit max expressions via repeated ReLUs.
def maxexpr(name, es):
    values=list(es)
    depth=0
    while len(values)>1:
        a=values[0::2]; b=values[1::2]
        d=positive(name+'_'+str(depth),[x-y for x,y in zip(a,b)])
        values=[y+z for y,z in zip(b,d)]+(a[-1:] if len(a)>len(b) else [])
        depth+=1
    return values[0]
edge_disk=maxexpr('edgemodel',[ci[q*4+b] for q in range(2) for b in [0,1]])
edge_bulge=maxexpr('edgebulgemodel',[ci[q*4+b] for q in range(2) for b in [2,3]])
face_disk=maxexpr('facediskmodel',[ci[q*4+b] for q in range(2,7) for b in [0,1]])
face_bulge=maxexpr('facebulgemodel',[ci[q*4+b] for q in range(2,7) for b in [2,3]])
small_core=refs('core_shape_sizemax',14)
small_disk=refs('disk_shape_sizemax',14)
ss=[.4*a+.6*b for a,b in zip(small_core,small_disk)]
sm=[]
for sh,inds in enumerate([[5,6],[3,4],[0,1,2]]):
    for size in [0,1]:
        sm.append(maxexpr('smoothclass_'+str(sh)+'_'+str(size),[ss[2*q+size] for q in inds]))
small_best=maxexpr('smallbest',[ss[2*q] for q in range(7)])
large_best=maxexpr('largebest',[ss[2*q+1] for q in range(7)])
resolved=clip01('resolved',[(large_best-small_best+.015)/.06])[0]

# ---------- Hand-written continuous morphology rules ----------
# These dimensionless constants express qualitative morphology, not fitted
# class frequencies or quantities measured from training examples.
c_inner,c_middle,c_wide,c_disk=redness
features=clip01('morphology',[
    (c_middle+.01)/.20,               # warm stellar population
    (.04-c_middle)/.22,               # blue star-forming disk
    (c_inner+.01)/.20,                # warm central bulge
    (texture[1]-.018)/.10,            # non-monotone arms and clumps
    (asym[0]-.10)/.48,                # central lopsidedness
    (asym[1]-.13)/.55,                # wider lopsidedness
    (qdisk-qcore-.07)/.25,            # bar: core flatter than disk
    (qcore-qdisk-.06)/.30,            # round bulge in a flatter disk
    (ell[3.5]-ell[10.]-.025)/.25,     # independent inner bar evidence
    (.45-con[0])/.30,                 # spatially extended envelope
    (con[0]-.22)/.38,                # multi-scale central concentration
    (peakrat[0]-.065)/.22,            # separate compact nucleus
    (.78-peakrat[1])/.48,             # centre lies between nuclei
    (.96-fit[0])/.24,                # departure from smooth ellipses
])
red,blue,core_red,tex,a_inner,a_outer,bar,bulge,bar2,extended,compact,secondary,deficit,nonell=features
# A bar must not be merely a pair of separated compact nuclei.
bar_strength=clip01('bar_strength',[.35*bar+.15*bar2+.85*bar_model+.70*outer_bar-.5*secondary-.35*a_inner-.45*edge_evidence])[0]
# Fuzzy interval penalties in the projected minor/major axis ratio.
shape_pen=positive('shape_penalties',[
    .82-qdisk, qdisk-.82,
    .48-qdisk, qdisk-.80,
    .23-qdisk, qdisk-.45,
    qdisk-.27,
    .32-qdisk,
])
round_low,round_high,mid_low,mid_high,cigar_low,cigar_high,edge_high,face_low=shape_pen
smooth=.65*red+.35*core_red+.70*compact-.85*tex-.95*a_inner-.5*bar_strength-.25*bulge-.60*model_bulge-.50*arm_model-.80*edge_evidence
spiral=.45*blue+.55*tex+.35*extended+.35*bulge-.50*a_inner+.50*model_bulge+.50*arm_model-1.20*edge_evidence
scores=[
    -.05+1.65*a_inner+.40*a_outer+.50*tex+.30*blue+.25*nonell-.90*secondary-.35*red-.30*model_bulge-1.2*pair_model,
    -.10+2.25*secondary+1.25*deficit+1.20*a_inner+.20*nonell-.35*bar_strength+1.8*pair_model,
    .55+smooth-7*round_low-.10*extended,
    .50+smooth-6*mid_low-6*mid_high,
    .30+smooth-7*cigar_low-7*cigar_high-.30*blue,
    .10+spiral+1.55*bar_strength-.10*compact-8*face_low,
    .25+spiral+.35*core_red+.45*compact-.55*tex-.75*bar_strength-7*face_low,
    .05+spiral+.45*blue+.40*tex-.35*compact-.65*bar_strength-7*face_low,
    1.05+.65*blue+.30*extended-.55*bulge-.30*compact-.60*a_inner-.30*model_bulge-5*edge_high+.8*edge_evidence,
    1.10+.45*core_red+.80*bulge+.25*compact-.50*a_inner+.30*model_bulge-5*edge_high+.8*edge_evidence,
]
# Direct generative comparison compensates for uncertain hand-estimated
# axis ratios. The three smooth classes favour compact profiles; resolved
# multicomponent galaxies favour the disk/arm/bar models.
smooth_physical=[maxexpr('sm_evidence_'+str(i),[sm[2*i],sm[2*i+1]-.025]) for i in range(3)]
physical=[
    best_all-.035, pair_best-.012,
    smooth_physical[0],smooth_physical[1],smooth_physical[2],
    bar_best-.004,
    maxexpr('tight_physical',[face_bulge,face_disk-.01,spiral_best-.015]),
    maxexpr('loose_physical',[spiral_best,face_disk-.01]),
    edge_disk,edge_bulge,
]
# Shared max subtraction does not affect argmax; it keeps logits modest.
final=[.55*s+12*(m-best_all) for s,m in zip(scores,physical)]
# Compactness is primarily a smooth-versus-disk cue, not a colour cue.
for i in [2,3,4]:
    final[i]+=.30*(1-resolved)
for i in [5,6,7]:
    final[i]+=.30*resolved
# A straight central component is evidence for a bar only off edge-on.
final[5]+=.55*bar+.30*bar2-.50*edge_evidence
final[1]+=.35*pair_model
final[0]+=.35*nonell+.15*tex
final[8]+=.25*edge_evidence
final[9]+=.25*edge_evidence
affine('class_scores',final)

# Exact graph simplification: remove nodes not needed by the ten scores.
needed={layers[-1]['name']}
kept=[]
for L in reversed(layers):
    if L['name'] in needed:
        kept.append(L)
        needed.update(L['inputs'])
layers=list(reversed(kept))
weights={k:v for k,v in weights.items() if k.rsplit('.',1)[0] in needed}

out=Path(sys.argv[1] if len(sys.argv)>1 else '/app/out')
out.mkdir(parents=True,exist_ok=True)
(out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n')
np.savez(out/'weights.npz',**weights)
print('Wrote',len(layers),'layers and',sum(v.size for v in weights.values()),'float32 weights')
