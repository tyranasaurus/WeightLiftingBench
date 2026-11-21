"""Analytic photometry and morphology, compiled into affine/ReLU/pooling layers."""
import sys,json
from pathlib import Path
import numpy as np

out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
W={}; layers=[]
def op(name,typ,src,**kw):
    layers.append(dict(name=name,op=typ,inputs=src if isinstance(src,list) else [src],**kw)); return name
def affine(name,src,w,b=0,conv=False,**kw):
    w=np.asarray(w,np.float32); n=w.shape[0]; W[name+'.W']=w; W[name+'.b']=np.broadcast_to(np.float32(b),(n,)).copy()
    if conv: return op(name,'conv',src,kernel=list(w.shape[2:]),out=n,**kw)
    return op(name,'dense',src,out=n)
def relu(name,src):return op(name,'relu',src)
def reshape(name,src,shape):return op(name,'reshape',src,shape=shape)
def pool(name,typ,src,h,w):return op(name,typ,src,size=[h,w],stride=[h,w])

# Compile scalar expressions into vectors of simultaneous affine maps and hinges.
# An expression is a sparse linear combination of previously defined rectifiers
# (or input coordinates). Node -1 is the affine constant.
class E:
    def __init__(self,d): self.d=d
    def __add__(self,o):
        o=expr(o); d=self.d.copy()
        for k,v in o.d.items():
            d[k]=d.get(k,0.)+v
            if abs(d[k])<1e-14: del d[k]
        return E(d)
    __radd__=__add__
    def __neg__(self):return self*(-1.)
    def __sub__(self,o):return self+(-expr(o))
    def __rsub__(self,o):return expr(o)+(-self)
    def __mul__(self,a):return E({k:v*a for k,v in self.d.items()})
    __rmul__=__mul__
    def __truediv__(self,a):return self*(1./a)
def expr(o):return o if isinstance(o,E) else E({-1:float(o)})
class Circuit:
    def __init__(self,n):self.n=n;self.level=[0]*n;self.defs={};self.inp=[E({i:1.}) for i in range(n)]
    def R(self,e):
        e=expr(e); j=len(self.level); self.defs[j]=e; self.level.append(1+max([self.level[k] for k in e.d if k>=0]+[0]));return E({j:1.})
    def compile(self,name,src,outputs):
        needed=set(k for e in outputs for k in e.d if k>=0)
        for k in reversed(range(self.n,len(self.level))):
            if k in needed: needed.update(i for i in self.defs[k].d if i>=0)
        levels={0:(src,list(range(self.n)))}
        def linear(nm,rows):
            bias=[r.d.get(-1,0.) for r in rows];parts=[]
            for l,(node,keys) in levels.items():
                which={k:i for i,k in enumerate(keys)}; m=np.zeros((len(rows),len(keys)),np.float32)
                for j,r in enumerate(rows):
                    for k,v in r.d.items():
                        if k in which:m[j,which[k]]=v
                if np.any(m):parts.append(affine(nm+'_L'+str(l),node,m,bias if not parts else 0))
            return op(nm+'_sum','add',parts) if len(parts)>1 else parts[0]
        for l in range(1,max(self.level)+1):
            ids=[k for k in sorted(needed) if self.level[k]==l]
            if ids:levels[l]=(relu(name+'_relu'+str(l),linear(name+'_'+str(l),[self.defs[k] for k in ids])),ids)
        return linear(name+'_out',outputs)

# Work on 2x2 binned light; a convolution with symmetric zero margins in
# its kernel cheaply extracts the central square, without a large dense crop.
# Negative padding is not a layer option: use a sequence of valid 1-tap-offset
# 3x3 or 9x9 correlations only if needed. Here the photometric masks themselves
# are zero outside the aperture, so no crop at all is needed.
bin2=pool('bin2','avgpool','image',2,2)
# A 2x downsampled Gaussian-smoothed luminance field. Additional shallow channels
# are used for measuring colour. Gaussian smoothing reduces shot noise.
g=np.arange(-2,3); gg=np.exp(-(g[:,None]**2+g[None,:]**2)/(2*.67**2));gg/=gg.sum(); lum=np.array([.34,.44,.22]);
sm=affine('photometric_psf',bin2,(lum[:,None,None]*gg)[None],conv=True,padding=2,stride=1)
# Count light only above the faint sky floor; also cap the brightest light for
# measurements where a star or saturated core should not control the geometry.
skylevel=.105
fields=relu('positive_light',affine('isophotes',sm,np.ones((2,1,1,1)),[-skylevel,-.62],True))

# Scalar input banks: integrate a library of explicit aperture masks and spatial
# profiles on each source plane. Multiple banks are concatenated by zero embedding.
banks={}; sz=64
y,x=np.indices((sz,sz),dtype=float);x=(x+.5)*2-64;y=(y+.5)*2-64
r=np.hypot(x,y); theta=np.arctan2(y,x)
def measure(src,mask):
    if src not in banks:banks[src]=[]
    j=len(banks[src]); banks[src].append(np.asarray(mask,np.float32).ravel());return (src,j)
def light(m,cap=False):return measure(fields,np.stack([m,-m if cap else m*0]))
def band(m,coeff):return measure(bin2,np.array(coeff)[:,None,None]*m)
def unit(m):return m/m.sum()
def ga(s):return np.exp(-.5*(r/s)**2)
def smoothvec(s):return unit(ga(s))
def rmsmag(cs,sn,R):
    # Polygonal quadrature for sqrt(c^2+s^2), exact up to about one percent.
    return sum(R(np.cos(a)*cs+np.sin(a)*sn) for a in np.arange(12)*2*np.pi/12)*(np.pi/12)

# Compensated spin-2 moments for a nearly PSF-independent axial ratio. s is
# a Gaussian aperture dispersion in input pixels; trace and anisotropy have
# the same units. The r^2 factor gives the sky far from the aperture zero weight.
scales=[5.,8.,13.,20.,30.]; mt={}
for s in scales:
    h=unit(ga(s))*(r/s)**2
    mt[s]=[light(h,True),light(h*np.cos(2*theta),True),light(h*np.sin(2*theta),True),light(unit(ga(s)),True)]

# Compare the concentration of the green/red/blue light in circular apertures.
# All colour measurements are sky compensated: the same weight is subtracted
# from a diffuse surrounding annulus.
sky=unit(ga(32)-ga(23)); aperture={}
for s in [2.5,5.,9.,16.,24.]:
    h=smoothvec(s)-sky
    aperture[s]=[band(h,lum),band(h,[1,0,-1])]

# Fourier analysis in rings. A disk projected at an inclination can be very
# noncircular, but does not *wind*: its quadrupole has constant position angle.
# Whichever quadrupole the smooth projected envelope contributes cancels when
# comparing the phase or evaluating the cross products of adjacent rings.
rr=[3.5,5.,7.,10.,14.,20.,28.,39.]; rings={}
for a in rr:
    h=unit(np.exp(-.5*((r-a)/max(1.4,.19*a))**2))
    rings[a]=[light(h,True)]+[light(h*f(m*theta),True) for m in [1,2,3,4] for f in (np.cos,np.sin)]

# Polar contrasts with opposite directions are L1 measurements made in the
# rectifier circuit, not stored pixel exemplars. Average within a sector first
# so pixel noise doesn't masquerade as tidal structure.
sect={}
for a in [6.,10.,16.,24.]:
    rad=np.exp(-.5*((r-a)/(.23*a))**2); masks=[]
    for k in range(12):masks.append(unit(rad*np.exp(11*(np.cos(theta-k*np.pi/6)-1))))
    sect[a]=[light(masks[k]-masks[k+6],True) for k in range(6)]

# Search for ridges with positive curvature in BOTH dimensions. At a secondary
# nucleus all principal curvatures are negative; for a thin edge-on disk the
# longitudinal second derivative remains small or is positive. Filters of two
# widths distinguish extended companions from pixel-sized stars.
# Work on a 32x32 map (4 px spacing), centered on the same sky coordinate.
coarse=pool('bin4','avgpool',sm,2,2); yc,xc=np.indices((32,32));rc=np.hypot((xc-15.5)*4,(yc-15.5)*4)
blob={}
for t in [1.05,1.85]:
    ts=str(int(t*100)); k=int(np.ceil(t*2.8)); cy,cx=np.mgrid[-k:k+1,-k:k+1]; p=np.exp(-(cx**2+cy**2)/(2*t*t)); kernels=[]
    for a in np.arange(4)*np.pi/4:
        filt=p*(1-((cx*np.cos(a)+cy*np.sin(a))/t)**2);filt-=filt.sum()*p/p.sum();filt/=filt[filt>0].sum();kernels.append(-filt)
    hess=affine('curvature'+ts,coarse,np.array(kernels)[:,None],conv=True,padding=k)
    hess=reshape('hess_grid'+ts,hess,[1,4,1024]); mx=pool('least_curved'+ts,'maxpool',hess,4,1); pt=reshape('peak_grid'+ts,mx,[1,32,32]); pt=relu('convex_cores'+ts,affine('negcurv'+ts,pt,[[[[-1]]]],conv=True))
    # Pointwise maximum inside a smooth off-centre search aperture using a
    # log-free padded max: positive thresholded candidates contribute both
    # summed prominence and evidence of a bright, compact off-centre peak.
    for j,(inner,outer) in enumerate([(0,7),(6,22),(13,36)]):
        mask=(rc>=inner)&(rc<outer); blob[t,j]=measure(pt,mask if j else mask/mask.sum())
    # Hand-designed rectified levels emphasize a conspicuous secondary core.
    strong=relu('conspicuous'+ts,affine('core_threshold'+ts,pt,[[[[1]]]],[-.12],True)); blob[t,3]=measure(strong,(rc>=6)&(rc<25));blob[t,4]=measure(strong,(rc>=14)&(rc<35))

# Log-polar image. Average light in sectors rather than sampling a single
# pixel; equally spaced samples in log r turn logarithmic arms into lines.
# Shape and pitch are measured on a per-image basis, with no stored examples.
K=24; radii=3.5*2**(np.arange(14)/4); angles=np.arange(K)*2*np.pi/K
polar_masks=np.stack([unit(np.exp(-.5*((r-a)/max(1.3,.13*a))**2 + (np.cos(theta-phi)-1)/(.115**2+(1.15/a)**2))) for a in radii for phi in angles]); nr=len(radii)
polar0=affine('log_polar_photometry',sm,polar_masks.reshape(nr*K,-1)); polar=reshape('polar_map',polar0,[1,nr,K]);
# Ellipse estimator from the radius of several isophotes. Use an angular
# harmonic of log(radius) rather than the second moment of all the light so
# that unrelated sky objects don't dominate the minor eigenvalue.
iso={}; simple={}; radii_flux={}
for j,thr in enumerate([.18,.27,.39]):
    delta=.05
    cuts=relu('sector_hinges'+str(j),affine('sector_thresholds'+str(j),polar,np.ones((2,1,1,1)),[-thr,-thr-delta],True)); occ=affine('bright_sectors'+str(j),cuts,np.array([1/delta,-1/delta]).reshape(1,2,1,1),conv=True)
    # Integrating occupied rings in d(log r) gives log(r_iso/r_min).
    cs=np.ones((nr,1))*np.cos(angles*2)[None,:]/K;sn=np.ones((nr,1))*np.sin(angles*2)[None,:]/K
    iso[j]=[measure(occ,m*np.log(2)/4) for m in (cs,sn)]
    radii_flux[j]=measure(occ,np.broadcast_to((2*radii**2*np.log(2)/4)[:,None]/K,(nr,K)))
# Soft disk radii from monotonic light-profile crossings. Count a disk as
# resolved if both sides of a diameter brighten outwards; an elliptical with
# a smooth monotone profile cannot give a positive response on any such ray.
rev={}; rridge=np.arange(1,nr-2)
for step in [2,4]:
    mats=[];starts=list(range(2,nr-step))
    for j in starts:
        for k in range(K):
            m=np.zeros((nr,K));m[j+step,k]=1;m[j,k]=-1;mats.append(m.ravel())
    dif=affine('outward_diff'+str(step),polar0,mats,-.015);rise=relu('outward_rise'+str(step),dif)
    # min over opposite rays. A star on one side cannot imitate a pair of arms.
    m=np.eye(len(mats)).reshape(len(starts),K,-1);pair=m[:,:K//2];opp=m[:,K//2:]; side=relu('unpaired_rise'+str(step),affine('rise_difference'+str(step),rise,(pair-opp).reshape(-1,len(mats))));p1=affine('first_side'+str(step),rise,pair.reshape(-1,len(mats)));p2=affine('remove_unpaired'+str(step),side,-np.eye(len(starts)*K//2));both=op('paired_rise'+str(step),'add',[p1,p2]);
    for mkey,src0,cols in [('two',both,K//2),('one',rise,K)]:
        v=reshape('rise_map'+mkey+str(step),src0,[1,len(starts),cols]);v=pool('max_arm'+mkey+str(step),'maxpool',v,1,cols);v=pool('arm_coherence'+mkey+str(step),'avgpool',v,2,1);v=pool('arm_strength'+mkey+str(step),'maxpool',v,len(starts)//2,1);rev[step,mkey]=measure(v,[[[1]]])
# Normalize each ring's angular contrast by a stabilizing floor. norm on the
# duplicated ring with +epsilon and -epsilon copies has exactly the desired
# denominator sqrt(angular_variance+epsilon^2+norm_eps).
m=np.zeros((nr*3*K,nr*K));bias=np.zeros((nr,3,K));eps=.085
for j in range(nr):m[j*3*K:(j+1)*3*K,j*K:(j+1)*K]=np.tile(np.eye(K),(3,1));bias[j,1]=eps*np.sqrt(1.5);bias[j,2]=-eps*np.sqrt(1.5)
v=affine('polar_stabilizer',polar0,m,bias.ravel());v=reshape('ring_channels',v,[nr,3,K]);v=op('whiten_each_ring','norm',v)
L=5;starts=list(range(1,nr-L+1));armspec={}; pitches=[0.,14.,23.,36.,55.]
# Matched log-spiral phase filters. Remove the mean complex coefficient across
# the radial window so a plain inclined disk or central bar is not a spiral.
for mval in [2,3]:
    specs=[]
    for pitch in pitches:
        for handed in [-1,1]:
            omega=handed/np.tan(np.deg2rad(pitch)) if pitch else 0
            phase=np.exp(-1j*mval*omega*np.log(radii[:L]/radii[0]));
            if pitch:phase-=phase.mean()
            denom=np.sqrt(np.mean(abs(phase)**2));phase/=max(denom,.30)
            for j in starts:
                tmpl=phase[:,None]*np.exp(1j*mval*angles)[None,:]/(L*K)
                # max over phase orientations computes complex amplitude.
                for ang in np.arange(8)*np.pi/4:
                    z0=np.zeros((nr,3,K));z0[j:j+L,0,:]=(tmpl*np.exp(1j*ang)).real;specs.append(z0.ravel())
    v0=affine('spiral_quadrature'+str(mval),v,specs);v0=reshape('spiral_modes'+str(mval),v0,[len(pitches),len(starts)*2,8]);v0=pool('spiral_power'+str(mval),'maxpool',v0,len(starts)*2,8)
    for i,pitch in enumerate(pitches):armspec[mval,pitch]=measure(v0,np.eye(len(pitches))[i])
# Isophotal geometry with local sky removed on each image. Integrate occupied
# area and its moment tensor in a circular aperture. Bright isophotes isolate
# the stellar body; elliptical models allow an independent angular size estimate.
bg=affine('surrounding_sky',sm,sky.reshape(1,-1));bgmap=reshape('sky_grid',affine('subtract_sky',bg,-np.ones((sz*sz,1))),[1,sz,sz]);netlight=op('sky_subtracted','add',[sm,bgmap]);
levs=[.075,.14,.24];width=.06
cut=relu('sky_isophote_hinges',affine('sky_isophote_levels',netlight,np.ones((2*len(levs),1,1,1)),-np.ravel([(c,c+width) for c in levs]),True));mat=np.zeros((len(levs),len(levs)*2));
for j in range(len(levs)):mat[j,j*2:j*2+2]=[1/width,-1/width]
occupancy=affine('sky_corrected_occupancy',cut,mat[:,:,None,None],conv=True); contours={}
for j,c in enumerate(levs):
    for s in [11.,18.]:
        def cont(mask):
            mm=np.zeros((len(levs),sz,sz));mm[j]=mask;return measure(occupancy,mm)
        h=unit(ga(s));contours[c,s]=[cont(h),cont(h*(r/s)**2),cont(h*(r/s)**2*np.cos(2*theta)),cont(h*(r/s)**2*np.sin(2*theta))]

# Project onto the manifold of monotone, concentric elliptical isophotes.
# For each hypothesized axis ratio and orientation, sample elliptical rings in
# the central disk. Within-ring dispersion measures surface detail, regardless
# of the radial surface brightness law (exponential, Sersic, core + envelope...).
# Compute the disk-vs-core change in best-fitting ellipse as a separate bar cue.
angles_e=np.arange(10)*np.pi/10; qs=[.26,.40,.56,.72,.86,1.];er=3.5*2**(np.arange(12)/3);angpts=np.arange(16)*np.pi/8
em=[]; models=[]
for q0 in qs:
    for pa in (angles_e if q0<1 else [0.]):
        models.append((q0,pa));ux=x*np.cos(pa)+y*np.sin(pa);uy=-x*np.sin(pa)+y*np.cos(pa)
        # Average across a PSF-sized patch at each ellipse point, including a
        # small extra width increasing with radius to reject isolated shot noise.
        for a in er:
            for ang in angpts:
                h=np.exp(-.5*((ux-a*np.cos(ang))**2+(uy-a*q0*np.sin(ang))**2)/(1.3+.065*a)**2)
                # Input is already low pass. The central 40x40 binned square
                # suffices for this branch. Sampling matrix below is cropped.
                em.append(unit(h[12:-12,12:-12]).ravel().astype(np.float32))
# Extract the central square with a separable dense gather. A valid convolution
# with a single nonzero tap would also work, but would materialize large im2col.
cn=40;cropmat=np.zeros((cn*cn,sz*sz),np.float32);idx=np.arange(sz*sz).reshape(sz,sz)[12:-12,12:-12].ravel();cropmat[np.arange(cn*cn),idx]=1
cropped=affine('central_photometry',sm,cropmat);el=affine('ellipse_samples',cropped,em);del em
ne=len(models);na=len(angpts);nr_e=len(er);mp=reshape('all_ellipses',el,[ne,nr_e,na]);
# Average absolute paired residuals at 90 and 180 degrees, then divide in
# the final rule circuit by the brightness of the central galaxy. Probe each
# *ring mean* independently, by doing a shared pointwise map over angular
# positions treated as channels, and ellipse hypotheses as spatial locations.
# The dense map above uses angle-major rows here to allow channel mixing.
w_el=W['ellipse_samples.W'].reshape(ne,nr_e,na,-1).transpose(2,0,1,3).reshape(ne*nr_e*na,-1);W['ellipse_samples.W']=np.ascontiguousarray(w_el);del w_el
el0=reshape('elliptical_photometry_grid',el,[na,ne,nr_e]);I=np.eye(na);dev=I-np.ones_like(I)/na
# Even residual (twofold arms/bar), odd residual (lopsidedness, double core).
opp=np.roll(I,na//2,axis=1);d_even=.5*(I+opp)-np.ones_like(I)/na;d_odd=.5*(I-opp)
h=relu('ellipse_residuals',affine('decompose_ellipses',el0,np.concatenate([dev,d_even,d_odd])[:,:,None,None],conv=True));amean=np.zeros((3,3*na));
for j in range(3):amean[j,j*na:(j+1)*na]=2/na
mad=affine('angular_deviations',h,amean[:,:,None,None],conv=True)
# Take a common minimum over geometry for the central aperture, galaxy body,
# and outer disk separately. Free inclination minimizes false arms in smooth
# edge-on objects. Also compare face-on against the optimum in the inner disk.
err={};fits={};irange=list(range(nr_e));
for nm,center,wid in [('inner',6.,.50),('middle',12.,.50),('outer',21.,.49),('body',11.,.78)]:
    radial=unit(np.exp(-.5*(np.log(er/center)/wid)**2));
    # One short spatial dimension is radius; multiply by fixed radial weights
    # via dense layer with three input features per ellipse. Pool over models.
    nmats=np.zeros((3*ne,3,ne,nr_e),np.float32)
    for j in range(3):
        for i in range(ne):nmats[j*ne+i,j,i]=radial
    deviations=affine('model_errors_'+nm,mad,nmats.reshape(3*ne,-1));del nmats
    # Shape restriction: choose ellipse by angular residual relative to local
    # radial contrast. A small ellipse samples brighter regions than a round
    # one at the same major radius. This mildly favours round models, safely.
    for j in range(3):
        sgn=np.zeros((ne,3*ne),np.float32);sgn[:,j*ne:(j+1)*ne]=-np.eye(ne);nv=affine('neg_err_'+nm+str(j),deviations,sgn);nv=reshape('model_grid_'+nm+str(j),nv,[1,1,ne]);nv=pool('opt_model_'+nm+str(j),'maxpool',nv,1,ne);err[nm,j]=measure(nv,[[[-1]]])
    def subseterr(name,subset):
        n=len(subset);mm=np.zeros((n,3*ne));mm[np.arange(n),subset]=-1;v0=affine('select_'+nm+name,deviations,mm);v0=reshape('models_'+nm+name,v0,[1,1,n]);v0=pool('opt_'+nm+name,'maxpool',v0,1,n);fits[nm,name]=measure(v0,[[[-1]]])
    for tag,low,high in [('thin',0,.41),('intermediate',.55,.73),('round',.85,1.),('flat',0,.57)]:subseterr(tag,[i for i,(q0,pa) in enumerate(models) if low<=q0<=high])
# Generative morphology branch. Correlate a mean-removed, central, weighted
# image against noiseless analytic surface brightness profiles. Distinct bands
# of spatial frequencies: one for spiral detail and a second for the gross
# profile. All the trial models are generated here, not sampled from data.
nt=24; delta=3.; axis=(np.arange(nt)+.5-nt/2)*delta;yy,xx=np.meshgrid(axis,axis,indexing='ij');rt=np.hypot(xx,yy);bpweight=np.exp(-.25*(rt/15.)**2);nt2=nt*nt
# The image transform is exactly the same as the template transform, including
# edge handling. Dense matrices are just fixed separable filters + a taper.
def blurmat(sig):
    a=np.exp(-.5*((axis[:,None]-axis)/sig)**2);return a/a.sum(1,keepdims=True)
short=blurmat(3.2);long=blurmat(8.);apwt=bpweight.ravel();B1=np.eye(nt2)-np.kron(short,short);B2=np.eye(nt2)-np.kron(long,long)
T=(.60*B1+.40*B2)*apwt[:,None];broad=(np.eye(nt2)-np.broadcast_to(unit(1-bpweight**4).ravel(),(nt2,nt2)))*apwt[:,None]
# Binned photometry enters the template branch; the template PSF is analytic.
smpl=np.stack([unit(np.exp(-.5*((x-x0)**2+(y-y0)**2)/1.15**2)).ravel() for y0,x0 in zip(yy.ravel(),xx.ravel())]);sampled=affine('central_model_grid',sm,smpl);template_features={}
def model_input(name,T):
    a=affine(name+'_window',sampled,T);a=reshape(name+'_field',a,[1,nt,nt]);return op(name+'_normalization','norm',a)
fi=model_input('bandpass',T);bi=model_input('broadband',broad);P=blurmat(1.9)
# Remove mean and normalize to match the norm layer. The resulting dense
# filter is a correlation coefficient (not squared distance in pixel space).
# Blur/rebin template then highlight detail with same transform as the image.
def templ_norm(arr,typ):
    a=P@np.asarray(arr)@P.T
    if typ=='band':v=bpweight*(a-.60*(short@a@short.T)-.40*(long@a@long.T))
    else:v=bpweight*(a-np.sum(unit(1-bpweight**4)*a))
    v=v-v.mean();return (v.ravel()/(nt2*(v.std()+1e-8))).astype(np.float32)
def bankmax(name,arr,typ='band'):
    s=affine('template_'+name,fi if typ=='band' else bi,arr);s=reshape('views_'+name,s,[1,1,len(arr)]);s=pool('match_'+name,'maxpool',s,1,len(arr));template_features[name]=measure(s,[[[1]]])
# Projection hypotheses. Bars are modeled in the disk plane, not by drawing
# a bright line in an arbitrary sky direction. A softened exponential nucleus
# and finite arm widths avoid aliasing the templates at small radii.
for morph in ['plain','bar','tight','loose']:
    patterns=[]
    for q0 in [.52,.75,1.]:
        for pa in (np.arange(4)*np.pi/4 if q0<1 else [0.]):
            u=xx*np.cos(pa)+yy*np.sin(pa);v=(yy*np.cos(pa)-xx*np.sin(pa))/q0;d=np.hypot(u,v);az=np.arctan2(v,u)
            for h0 in [4.5,7.,10.,15.]:
                disk=np.exp(-d/h0);core=np.exp(-.5*(rt/(1.4+.10*h0))**2);d0=d/h0
                if morph=='plain':
                    for frac in [.06,.36,.82]:patterns.append(templ_norm(disk+frac*core,'band'))
                    continue
                if morph=='bar':
                    for phi in np.arange(10)*np.pi/10:
                        xb=u*np.cos(phi)+v*np.sin(phi);yb=v*np.cos(phi)-u*np.sin(phi)
                        for blen in [.85,1.55]:
                            # Flat-topped Ferrers light with a soft rectangular
                            # shape; this has more power near its ends than a
                            # geometrically similar exponential stellar disk.
                            barprof=np.exp(-.5*((xb/(blen*h0))**4+(yb/(.30*blen*h0))**2));patterns.append(templ_norm(disk+.22*core+.65*barprof,'band'))
                else:
                    for pitch in ([12.,21.] if morph=='tight' else [30.,45.]):
                        for sign in [-1,1]:
                            phase=2*(az-sign*np.log(np.maximum(.25,d0))/np.tan(np.deg2rad(pitch)));amplitude=.73*disk*(1-np.exp(-d0**2/.48));
                            for phi in np.arange(10)*np.pi/5:
                                arm=np.cos(phase-phi);patterns.append(templ_norm(disk+.27*core+amplitude*(.78*arm+.22*(2*arm**2-1)),'band'))
    bankmax(morph,patterns)
# Smooth elliptical, thin disk and central-spheroid-plus-thin-disk families.
# The difference of these last two matches explicitly tests the bulge rather
# than equating red colour with bulge light.
for morph in ['smooth','line','bulging']:
    patterns=[]
    for q0 in ([.33,.47,.62,.78,.90,1.] if morph=='smooth' else [.12,.20,.29]):
        for pa in (np.arange(16)*np.pi/16 if q0<1 else [0.]):
            u=xx*np.cos(pa)+yy*np.sin(pa);v=yy*np.cos(pa)-xx*np.sin(pa)
            for h0 in [3.2,5.,7.5,11.,17.]:
                if morph=='smooth':
                    d=np.hypot(u,v/q0)/h0
                    for n in [.8,1.55]:patterns.append(templ_norm(np.exp(-n*(d+.04)**(1/n)),'broad'))
                else:
                    disk=np.exp(-abs(u)/h0)/np.cosh(np.minimum(abs(v)/(h0*q0+.5),45))**2
                    if morph=='line':patterns.append(templ_norm(disk,'broad'))
                    else:
                        for frac in [.48,1.2]:patterns.append(templ_norm(disk+frac*np.exp(-np.sqrt((u/(.40*h0+1))**2+(v/(.32*h0+1))**2)),'broad'))
    bankmax(morph,patterns,'broad')
# A blue ridge in a disk is stronger evidence for young spiral arms than a
# ridge in a single broadband image. Use sky-subtracted colour differences,
# avoiding direct ratios between noisy outer-annulus colour channels.
chrom={};ringcol={}; greenex=[-.5,1.,-.5]
for s in [4.,8.,13.,21.]:
    h=unit(ga(s));chrom[s]=[band(h-sky,z) for z in [lum,[1.,0.,-1.],greenex]]
for a in [6.,10.,15.,22.]:
    h=unit(np.exp(-.5*((r-a)/(.27*a))**2));ringcol[a]=[band(h-sky,z) for z in [lum,[1.,0.,-1.],greenex]]
# Directional second derivatives tangential vs radial to the line of sight
# through the nucleus. A smooth monotone envelope has no strong local maximum
# along a central ray. Open and tightly wound arms have different inclinations
# relative to a ray. Compare neighbouring radial intervals on the polar field,
# adapting the radial sampling interval to the width of the feature.
# Direct two-nucleus templates: additive positive extended source, pair mass
# ratio, separation, and arbitrary sky orientation. Also include pairs with
# light centroid rather than dominant nucleus at the origin. No external atlas.
for tag in ['single_nucleus','double_nucleus']:
    pats=[]
    if tag=='single_nucleus':
        for a in [2.,3.8,6.,9.5]:
            for q0 in [.5,.75,1.]:
                for pa in (np.arange(6)*np.pi/6 if q0<1 else [0.]):
                    u=xx*np.cos(pa)+yy*np.sin(pa);v=yy*np.cos(pa)-xx*np.sin(pa);d=np.sqrt(u*u+(v/q0)**2)/a
                    for diskfrac in [.15,.6]:pats.append(templ_norm(np.exp(-.65*d*d)+diskfrac*np.exp(-.4*d),'band'))
    else:
        for a in [2.6,4.8]:
            for sep in [6.,10.,16.,24.,33.]:
                for pa in np.arange(16)*2*np.pi/16:
                    u=xx*np.cos(pa)+yy*np.sin(pa);v=yy*np.cos(pa)-xx*np.sin(pa)
                    for off in [0,.40]:
                        r1=((u+off*sep)/a)**2+(v/(.9*a))**2;r2=((u-(1-off)*sep)/a)**2+(v/a)**2
                        for ratio0 in [.42,.90]:pats.append(templ_norm(np.exp(-.5*r1)+ratio0*np.exp(-.5*r2)+.55*np.exp(-np.sqrt(r1)*.40)+.3*ratio0*np.exp(-np.sqrt(r2)*.40),'band'))
    bankmax(tag,pats)
# Shape of the *wings* of the central object. An edge-on galaxy with a round
# central bulge has a thin outer disk detected best after the core is masked;
# retain real flux rather than cap the weights or integrate sky-dominated pixels.
wing={}
for a in [9.,14.,21.]:
    h=unit(np.exp(-.5*((r-a)/(.32*a))**2));wing[a]=[measure(netlight,h),measure(netlight,h*np.cos(2*theta)),measure(netlight,h*np.sin(2*theta))]
# Local radial curvature in the galaxy itself. Compare each polar pixel to
# *linear interpolation in physical radius* on its ray. Decreasing exponentials
# are convex, but a luminous arm makes a concave ridge, even when the total
# surface brightness is still decreasing across the arm. Aggregate by both
# area and peak amplitude; average over neighboring angles before taking maxima.
radcurve={}
for step in [2,3]:
    ms=[];idxs=list(range(step,nr-step))
    for j in idxs:
        f=(radii[j]-radii[j-step])/(radii[j+step]-radii[j-step]);
        for k in range(K):
            m=np.zeros((nr,K));m[j,k]=1;m[j-step,k]=f-1;m[j+step,k]=-f;ms.append(m.ravel())
    rd=relu('radial_ridges'+str(step),affine('radial_curvature'+str(step),polar0,ms,-.008));rd=reshape('ridge_field'+str(step),rd,[1,len(idxs),K]);
    for key,typ in [('peak','maxpool'),('area','avgpool')]:
        ph=pool('ridge_in_rings'+key+str(step),typ,rd,1,K);ph=pool('ridge_in_annuli'+key+str(step),'avgpool',ph,2,1);ph=pool('visible_ridge'+key+str(step),'maxpool',ph,len(idxs)//2,1);radcurve[step,key]=measure(ph,[[[1]]])
# Absolute brightness of the strongest off-centre source within a few seeing
# widths of the galaxy. Different from the integral of all faint peaks.
central_peak={}
for key,lo,hi in [('near',3,7),('far',7,12)]:
    idx0=np.arange(nr*K).reshape(nr,K)[lo:hi].ravel();mat=np.eye(nr*K)[idx0];pk=affine('photometry_'+key,polar0,mat);pk=reshape('peaks_'+key,pk,[1,1,len(mat)]);pk=pool('brightest_'+key,'maxpool',pk,1,len(mat));central_peak[key]=measure(pk,[[[1]]])
centpix=measure(sm,smoothvec(2.5));skyval=measure(sm,sky)
# Make a compact ordered scalar vector. Keep dense layers small by only adding
# the final vector after each source has been integrated.
N=sum(len(v) for v in banks.values()); offsets={};beg=0;parts=[]
for i,(src,masks) in enumerate(banks.items()):
    offsets[src]=beg;part=affine('aperture_bank'+str(i),src,np.stack(masks));m=len(masks);pad=np.zeros((N,m));pad[beg:beg+m]=np.eye(m);parts.append(affine('slots'+str(i),part,pad));beg+=m
vec=op('measurements','add',parts);cir=Circuit(N);R=cir.R
def get(key):s,i=key;return cir.inp[offsets[s]+i]
def pos(x):return R(x)
def clamp(x,lo=0,hi=1):return lo+R(x-lo)-R(x-hi)
def ramp(x,lo,hi):return clamp((x-lo)/(hi-lo))
def ab(x):return R(x)+R(-x)
def maxi(a,b):return b+R(a-b)
def mini(a,b):return a-R(a-b)
def maximum(es):
    if len(es)==1:return es[0]
    return maximum([maxi(es[i],es[i+1]) for i in range(0,len(es)-1,2)]+([es[-1]] if len(es)%2 else []))
def approx(e,xs,ys):
    # Continuous linear spline, flat beyond the endpoints.
    slopes=np.diff(ys)/np.diff(xs);return ys[0]+sum(d*R(e-x) for x,d in zip(xs,np.diff(np.r_[0,slopes,0])))
cache_logs={}
def logp(e):
    key=tuple(sorted(e.d.items()));
    if key not in cache_logs:
        xs=2.**(np.arange(-11,4,.75));cache_logs[key]=approx(e,xs,np.log(xs))
    return cache_logs[key]
def ratio(a,b,eps=.008):
    xs=np.arange(-7.,2.25,.25);return approx(logp(a+eps*.015)-logp(b+eps),xs,np.exp(xs))
def signed_ratio(a,b,eps=.008):return ratio(R(a),b,eps)-ratio(R(-a),b,eps)
def times(a,b,limit=3):
    # Linear spline for the polarization identity. Used for dimensionless
    # colour/shape gates, not as a learned activation.
    xs=np.linspace(-limit,limit,2*int(limit*16)+1)
    return .25*(approx(a+b,xs,xs*xs)-approx(a-b,xs,xs*xs))
def mag(c,s):return rmsmag(c,s,R)

F={}; qvals=[]
for s in scales:
    tr,c,d,f=map(get,mt[s]);F[s]=f
    # Gaussian window and PSF circularize the ellipse. The Gaussian identity
    # T_ap/T_free follows by subtracting the window precision matrix. A mild
    # regularization is used at very low contrast and for small sources.
    mm=mag(c,d)
    # For a Gaussian, correcting the trace in (lambda1-lambda2)/(lambda1+lambda2)
    # entails subtracting 2*lambda1*lambda2/window_variance. In normalized
    # moments this denominator is T - (T^2-D^2)/2. Equivalently correct ratios
    # of the minor and major widths directly.
    maj=ratio(.5*(tr+mm),f,.002); minr=ratio(.5*(tr-mm),f,.002)
    # Work in log variances, and smoothly weaken the extrapolation as a
    # large source fills the aperture (nearly singular dewindowing).
    def lgvar(l):return logp(R(l-.45**2/(s*s))+0.001)-.82*logp(1.-clamp(l,0,.91))
    logq=clamp(.5*(lgvar(minr)-lgvar(maj)),-2.6,0)
    xs=np.arange(-2.75,.26,.25);qvals.append(approx(logq,xs,np.exp(xs)))
# Scale selection: measure small objects through a smaller aperture, and
# progressively enlarge it when appreciable surface brightness extends outside.
# Ratio F(2s)/F(s) tends to 1/4 for an unresolved source and 1 for a broad one.
grow=ratio(F[13.],F[5.],.003); big=ramp(grow,.27,.53); huge=ramp(grow,.5,.78)
qsmall=.20*qvals[0]+.55*qvals[1]+.25*qvals[2];qbig=.18*qvals[1]+.53*qvals[2]+.29*qvals[3];qhuge=.20*qvals[2]+.52*qvals[3]+.28*qvals[4]
qmom=qsmall+times(big,qbig-qsmall)+times(huge,qhuge-qbig)
# For an ellipse: <log r(theta) cos 2theta> = (1-q)/(2*(1+q)).
# A floor on polar radii from seeing biases q upward for very small objects;
# use the most informative isophote and retain the core moment estimate there.
qiso=[]
for j in iso:
    d=clamp(2*mag(*map(get,iso[j])),0,.88);qiso.append(ratio(1-d,1+d,.001))
qi=1-maximum([1-s for s in qiso]);q=.44*mini(qmom,qi)+.36*qi+.20*mini(qvals[1],qvals[2])
area_log=logp((get(radii_flux[0])+4.)/256.)+np.log(256.)
# Independent estimate of length, not area. A thin edge-on disk with the
# same angular area as a bright elliptical can be much more extended.
length=area_log-logp(q+.035)
longdisk=ramp(length,2*np.log(16),2*np.log(34))
# Shape of sky-subtracted, connected central light is reinforced at high
# isophotes to resist low surface-brightness noise and foreground stars.
qcont=[];contsize=[]
for c in levs:
    for s in [11.,18.]:
        f,tr,cc,ss=map(get,contours[c,s]);d=mag(cc,ss);l1=ratio(.5*(tr+d),f,.003);l2=ratio(.5*(tr-d),f,.003)
        # Isophotes are closer to uniform ellipses than Gaussians. Debias only
        # half the Gaussian window correction for a uniform disk.
        lq=.5*(logp(l2+.005)-logp(l1+.005))+.24*(logp(1-clamp(l1,0,.94))-logp(1-clamp(l2,0,.94)))
        xs=np.arange(-2.75,.26,.25);z=approx(lq,xs,np.exp(xs));qcont.append(z)
        if c==levs[0]:contsize.append(logp(f+.0005)+np.log(2*s*s)-logp(1-clamp(.58*f,0,.85)))
# For an undersampled ellipse, prefer the shape of the luminous body to the
# shape of a tiny bright core. A response consistent across cuts is reliable.
qis=(qcont[0]+qcont[1]+qcont[2]*2+qcont[3]*2+qcont[4]+qcont[5])/8
q=.55*qis+.25*mini(qis,q)+.20*mini(qvals[0],qvals[1]);area_log=.30*contsize[0]+.70*contsize[1];length=area_log-logp(q+.02);longdisk=ramp(length,2*np.log(14),2*np.log(32))
A={s:get(z[0]) for s,z in aperture.items()}; col={s:signed_ratio(get(z[1]),A[s],.018) for s,z in aperture.items()}
# A smooth optical elliptical usually has a similar warm stellar population
# throughout. Blue outskirts and a warm nucleus suggest a star-forming disk.
warm=.45*col[5.]+.55*col[9.]; blue=ramp(-warm,-.06,.20); red=ramp(warm,.04,.32)
grad=col[2.5]-col[16.]; diskcol=ramp(grad,.045,.245)
# Compare peakiness at fixed surface-brightness extent. A compact bulge embedded
# in a much more diffuse disk makes this log-profile convex across apertures.
sharp=logp(A[2.5]+.006)-logp(A[9.]+.006)
outer=logp(A[9.]+.006)-logp(A[24.]+.006)
conc=sharp-.66*outer
resolvable=ramp(grow,.23,.52)

# Angular power and phase. Normalize each ring to its own brightness, with
# a soft floor that discounts invisible outer rings instead of amplifying noise.
mean={}; normharm={}; amp={}
for a in rr:
    vals=list(map(get,rings[a]));mean[a]=vals[0]
    for m in [1,2,3,4]:
        c=vals[2*m-1];d=vals[2*m];amp[a,m]=ratio(mag(c,d),mean[a],.038)
        if m==2:normharm[a]=[signed_ratio(v,mean[a],.038) for v in [c,d]]

def cross(i,j):
    u,v=normharm[i];s,t=normharm[j];return times(u,t,1.8)-times(v,s,1.8)
# Winding has a sign, not just a difference in shape. Require consistency in
# adjacent rings but permit either handedness. A bar is coherent in the centre
# and loses its quadrupole or changes PA at its ends.
wind={};barparts=[];armcontrast=[]
for i in range(len(rr)-2):
    a,b,c=rr[i:i+3];z=cross(a,b)+cross(b,c)+.35*cross(a,c);wind[a]=ab(z)
    # Elongation of core in excess of both PSF anisotropy and disk inclination.
    inner=.5*(amp[a,2]+amp[b,2]);barparts.append(inner-.75*amp[c,2]-.25*amp[rr[min(i+3,7)],2]);armcontrast.append(amp[c,2]-amp[a,2])
winding=maximum([wind[a] for a in rr[1:-2]])
baramp=maximum(barparts[0:5]);arms2=maximum(armcontrast[1:]);
odd=maximum([.5*(amp[a,3]+amp[b,3]) for a,b in zip(rr[2:6],rr[3:7])]);
# m=4 also occurs for a very flat regular ellipse; discount it in such cases.
four=maximum([.5*(amp[a,4]+amp[b,4]) for a,b in zip(rr[2:6],rr[3:7])]);
# Asymmetry in inner and outer disk. Summing sector differences downweights a
# single unassociated source compared with a broad asymmetric envelope.
asym={}
for a,masks in sect.items():
    near=min(rr,key=lambda b:abs(b-a));asym[a]=ratio(sum(ab(get(k)) for k in masks)/6,mean[near],.075)
asymc=.45*asym[6.]+.55*asym[10.];asymo=.56*asym[16.]+.44*asym[24.]

# Photometric evidence variables on 0..1 scales.
warp=ramp(winding,.008,.070); pattern=ramp(odd,.047,.16);boxy=ramp(four-.14*R(.6-q),.055,.15)
clumps=ramp(asymo,.14,.52);lopsided=ramp(asymc,.16,.63)
arm=ramp(arms2,.05,.21);bar=ramp(baramp,.05,.19)
wide=ramp(grow,.25,.53);               # extent compared with the seeing/aperture
excess=ramp(conc,-.44,-.08)             # a cuspy center relative to scale
spiral=.23*warp+.16*pattern+.10*boxy+.10*arm+.12*clumps+.15*diskcol+.19*wide+.09*blue+.09*bar
# A second nucleus should be conspicuous away from the centre, convex on a
# sufficiently broad scale, and carry a non-negligible fraction of central light.
bb={k:get(v) for k,v in blob.items()}
peaks=[]
for t in [1.05,1.85]:
    center=bb[t,0];inner=bb[t,1];remote=bb[t,2];hot=bb[t,3];dist=bb[t,4];
    norm=.18+center
    p1=ramp(ratio(hot+.28*dist,norm,.02),.22,1.55)
    p2=ramp(ratio(inner+.3*remote,norm,.02),1.2,4.1)
    peaks.append(.68*p1+.32*p2)
merger=.44*peaks[0]+.56*peaks[1]
# Very long, cold disks with weak bulges vs bulging edge-on systems: local
# flattening changes with radius if there is a central spheroidal component.
bulgeround=ramp(qvals[0]-qvals[2],.025,.18)
bulge=.26*excess+.23*red+.16*bulgeround+.10*diskcol+.25*(1-longdisk)
paired=ramp(.60*get(rev[2,'two'])+.40*get(rev[4,'two']),.007,.07);single=ramp(get(rev[2,'one']),.02,.14)
tightpow=maximum([get(armspec[m,p]) for m in [2,3] for p in [14.,23.]])
openpow=maximum([get(armspec[m,p]) for m in [2,3] for p in [36.,55.]])
spwave=ramp(maxi(tightpow,openpow),.035,.20)
opening=ramp(openpow-tightpow,-.025,.065)
# A genuine spiral is a coherent *rotation* in the complex quadrupole, not
# just a change of modulus as is seen in inclined monotone ellipses.
structure=.26*spwave+.18*paired+.19*warp+.17*diskcol+.10*pattern+.10*boxy
resolved=ramp(area_log,2*np.log(8),2*np.log(18))
# Absolute residuals after discarding the best-case monotone ellipse geometry.
# Even light: true bisymmetric features, irrespective of projected inclination.
# Odd light: off-centering, tidal distortion and mergers. Discount the noise
# before dividing, so a faint (but otherwise smooth) object stays smooth.
resid={};modelerr={}
for nm,scale in [('inner',5.),('middle',9.),('outer',16.),('body',9.)]:
    for j in range(3):resid[nm,j]=ratio(R(get(err[nm,j])-.006),R(A[scale])+.06,.003)
    for tag in ['round','intermediate','thin','flat']:modelerr[nm,tag]=ratio(R(get(fits[nm,tag])-get(err[nm,0])),R(A[scale])+.04,.004)
spir= .35*resid['middle',1]+.40*resid['outer',1]+.25*resid['body',1]
spires=ramp(spir,.018,.104);chaos=ramp(.65*resid['body',2]+.35*resid['outer',2],.035,.22)
barshape=ramp(modelerr['inner','round']+.4*modelerr['middle','round']-.8*modelerr['outer','round'],.005,.095)
# A flat-body model that is substantially better than a moderately inclined
# model is direct evidence for an edge-on, even if a bright bulge contaminates
# its moment-based b/a. The converse is evidence that it is only inclined.
flatfit=ramp(modelerr['middle','intermediate']+modelerr['outer','intermediate']-modelerr['outer','thin'],.01,.15)
roundcore=ramp(modelerr['inner','thin']-modelerr['middle','thin'],.005,.085)
bulge=.26*red+.14*excess+.14*roundcore+.12*bulgeround+.09*diskcol+.25*(1-longdisk)
# Convert the visual predicates into class scores. A class competes against its
# closest alternative by morphology, not by a stored prior or example count.
flat=.70*ramp(.47-q,0,.22)+.30*flatfit; veryflat=ramp(.32-q,0,.16)
disky=.43*spires+.25*resolved+.14*ramp(grad,.065,.26)+.10*paired+.08*warp
smooth=1.22-.71*disky-.28*chaos-.18*resolved-.06*blue
# Smooth subclasses are separated by their apparent b/a, including the rare
# transition to a lenticular ('cigar') for non-razor-thin, smooth objects.
s2=smooth-3.5*R(.885-q);s3=smooth-3.5*R(q-.80)-3.6*R(.50-q)-.018;s4=smooth-.11-3.8*R(q-.42)-2.0*R(.34-q)
# Common disk score, with a distinct vote for a bar or otherwise pitch/
# star-formation / concentration indicators to distinguish Sa from Sc.
sp=.41+.97*disky+.16*resolved-.87*flat-.08*chaos
corebar=maximum([amp[a,2] for a in [3.5,5.,7.,10.]])
bs=.44*barshape+.26*bar+.16*ramp(corebar,.05,.18)+.10*diskcol+.04*red
# Resolved loose arms make azimuthal light variations on outer ellipses,
# whereas tightly wound arms average out. Activity and blue colour are useful
# but neither is necessary for a loose spiral.
loose=.29*spires+.17*opening+.15*paired+.14*blue+.11*chaos+.07*clumps+.07*warp
s5=sp+.88*(bs-.36);s6=sp-.49*R(bs-.37)-.43*R(loose-.36);s7=sp+.99*(loose-.41)-.45*R(bs-.38)
# A thin line on the sky distinguishes both edge-on classes even when almost
# no blue light is visible. Scale and colour help disambiguate E7/S0 galaxies.
edge=.42+1.15*flat+.23*veryflat+.27*longdisk+.06*blue+.05*diskcol-.85*R(q-.4)-.10*clumps
s8=edge+.70*(.51-bulge);s9=edge+.70*(bulge-.51)
# An offset knot generally has both large odd residual and a local maximum.
# A star on blank sky outside an otherwise regular galaxy does not meet the
# first condition in the body aperture. Very uneven systems without bright
# secondary nuclei favour the disturbed class.
s1=.20+.79*merger+.38*chaos+.37*lopsided+.13*clumps+.10*red+.11*single-.16*veryflat
s0=.24+.66*chaos+.36*lopsided+.20*clumps+.10*pattern+.16*flat+.19*blue-.46*merger-.13*paired
# Relative morphology likelihoods from the analytic forward models. Comparing
# improvements on a single image cancels the contribution from seeing, common
# bulge light, or unrelated sky pixels that neither template accounts for.
tf={k:get(v) for k,v in template_features.items()};tb=tf['bar'];tt=tf['tight'];tl=tf['loose'];tp=tf['plain'];ts=tf['smooth'];te=tf['line'];tg=tf['bulging']
barbetter=ramp(tb-maxi(tt,tl),-.015,.06);tightbetter=ramp(tt-tl,-.016,.045);loosebetter=ramp(tl-maxi(tt,tb),-.015,.055)
spgain=ramp(maximum([tb,tt,tl])-tp,.009,.072)
# Absolute residual light on the sky: a very regular smooth galaxy gives a
# broadband profile match near unity. Treat a poor match as evidence only
# together with actual lopsidedness or conspicuous core/arm structure.
regular=ramp(ts,.89,.984);irregular=ramp(1-ts,.055,.24)
edgegain=ramp(maxi(te,tg)-ts,-.016,.034)
linefit=ramp(maxi(te,tg),.89,.985)
# Fractions of isophotal elliptical light estimates in the compact core vs
# larger disk. Differentiate an elongated bright body in a *rounder* disk
# from an intrinsically flat galaxy at all radii.
qbody=mini(qis,qi);qcore=qvals[0];barredlight=ramp(qbody-qcore,-.015,.19)
# Smooth objects: use the moment of the core preferentially for near-round
# objects, avoiding centre shifts of faint outer isophotes.
qsmooth=.38*qmom+.36*qvals[0]+.26*qis
smb=1.21-.62*disky-.25*chaos-.13*resolved+.22*(regular-.5)-.13*spgain
s2=smb-3.65*R(.855-qsmooth);s3=smb-3.65*R(qsmooth-.79)-3.8*R(.48-qsmooth)-.016
s4=smb-.035-4*R(qsmooth-.445)-1.8*R(.30-qsmooth)-.24*longdisk
sp=.38+.61*disky+.19*resolved+.33*spgain+.13*spires-.75*flat-.10*chaos
bs=.27*barshape+.21*bar+.12*ramp(corebar,.045,.16)+.14*barredlight+.26*barbetter
loose=.31*spires+.15*paired+.14*blue+.10*chaos+.10*loosebetter+.08*opening+.07*spgain+.05*warp
s5=sp+.88*(bs-.31);s6=sp+.16*(tightbetter-.46)-.45*R(bs-.34)-.42*R(loose-.29);s7=sp+.93*(loose-.34)-.44*R(bs-.32)
edge=.34+.89*flat+.22*veryflat+.23*longdisk+.34*edgegain+.09*linefit-.83*R(q-.4)-.13*chaos
bgain=ramp(tg-te,-.002,.022);bulge=.30*red+.16*roundcore+.12*bulgeround+.09*excess+.12*(1-longdisk)+.21*bgain
s8=edge+.68*(.47-bulge);s9=edge+.68*(bulge-.47)
s1=.20+.87*merger+.47*chaos+.33*lopsided+.18*irregular+.12*clumps+.14*red+.10*single-.16*veryflat
s0=.23+.63*chaos+.37*lopsided+.17*clumps+.25*irregular+.15*flat+.23*blue-.46*merger-.16*paired
# Colour of disk after subtracting the colour expected from a single stellar
# population. Aperture colours in the ring are not contaminated by the central
# red bulge. Multiplication operates on normalized intensities, so a faint
# annulus is automatically downweighted rather than noise-amplified.
ccol=signed_ratio(get(chrom[4.][1]),get(chrom[4.][0]),.012);gcol=signed_ratio(get(chrom[4.][2]),get(chrom[4.][0]),.012)
colfeatures=[];ringblue=[]
for a in ringcol:
    l,rb,gg=map(get,ringcol[a]);cdiff=times(ccol,l)-rb;gdiff=gg-times(gcol,l)
    colfeatures.append(ramp(cdiff-.4*gdiff,.005,.047));ringblue.append(ramp(signed_ratio(-rb,l,.065),-.03,.23))
colarms=.68*maximum(colfeatures)+.32*(colfeatures[1]+colfeatures[2])/2
outerblue=.45*ringblue[1]+.35*ringblue[2]+.20*ringblue[3]
quad=[];direc=[]
for a in wing:
    l,u,v=map(get,wing[a]);quad.append(ratio(mag(u,v),R(l),.013));direc.append([u,v])
# Persistent axial signal at increasing radius, and minor axis thin compared
# with major: characteristic of edge-on disks as opposed to bars in a disk.
wingflat=ramp(.16*quad[0]+.43*quad[1]+.41*quad[2],.37,.68)
large=ramp(area_log,2*np.log(7),2*np.log(15));extended=ramp(length,2*np.log(12),2*np.log(25));
compact=ramp(.5*(sharp+outer),.96,1.5)
# The spiral-vs-elliptical decision uses visibility as well as perturbations:
# resolved disks often look smooth in the middle, and cannot be identified
# from goodness of the *central* profile match alone.
diskflag=.24*resolved+.18*colarms+.18*spires+.13*spgain+.10*paired+.08*warp+.05*outerblue+.04*arm
smb=1.16-.47*diskflag-.17*resolved-.21*chaos+.12*(regular-.5)-.12*colarms
s2=smb-3.65*R(.855-qsmooth);s3=smb-3.65*R(qsmooth-.79)-3.8*R(.48-qsmooth)-.016
s4=smb+.13-4.1*R(qsmooth-.405)-2.3*R(.30-qsmooth)-.19*longdisk-.12*colarms
sp=.46+.81*diskflag+.19*resolved+.11*colarms+.09*spires-.69*flat-.21*wingflat-.10*chaos
# Bar cue strongest if central bar contrasted with an outer (rounder) disk.
# A compact red nucleus, by itself, is not a bar. An inclined disk without a
# bar contributes a quadrupole with similar angle and shape *at all radii*.
bs=.28*barshape+.25*bar+.12*barredlight+.22*barbetter+.13*ramp(corebar,.055,.18)-.16*wingflat
# Wide arm/interarm contrasts can dominate a ring while looking weak in an
# integrated aperture. Take evidence from the more conspicuous arm diagnostic.
armcontrast=ramp(resid['outer',1],.012,.075)
loose=.25*armcontrast+.17*spires+.13*outerblue+.12*opening+.11*paired+.09*chaos+.08*loosebetter+.05*warp
s5=sp+.75*(bs-.32);s6=sp+.08*(tightbetter-.4)+.12*R(.45-loose)-.32*R(bs-.33);s7=sp+.85*(loose-.34)-.31*R(bs-.34)
# Compare evidence from the narrow wings and from the entire profile. The
# central bulge can dominate the latter. Small, smooth cigar-shaped galaxies
# tend to lack a razor-thin, resolved outer pair of lobes.
edge=.30+.67*flat+.43*wingflat+.27*longdisk+.18*edgegain+.11*linefit+.15*veryflat-.99*R(q-.44)-.12*chaos
bulge=.29*red+.12*roundcore+.12*bulgeround+.07*excess+.16*(1-longdisk)+.13*bgain+.11*colarms
s8=edge+.8*(.39-bulge);s9=edge+.8*(bulge-.39)
double=ramp(tf['double_nucleus']-maximum([tb,tt,tl,tf['single_nucleus']]),-.015,.090)
s1=.22+.63*merger+.41*double+.40*chaos+.28*lopsided+.15*irregular+.11*clumps+.15*red+.10*single-.20*veryflat
# An imbalanced blue stellar body or an off-centre tidal tail is more likely
# disturbed than an unobscured smooth central spheroid.
s0=.22+.56*chaos+.32*lopsided+.22*irregular+.25*outerblue+.16*blue+.13*clumps+.12*flat-.30*merger-.24*double-.12*paired
cvpeak=ramp((get(radcurve[2,'peak'])+get(radcurve[3,'peak']))/2,.016,.09);cvarea=ramp((get(radcurve[2,'area'])+get(radcurve[3,'area']))/2,.004,.026)
ridge=.55*cvarea+.45*cvpeak
smbias=-.19*ridge;s2+=smbias;s3+=smbias;s4+=.60*smbias
s5+=.13*ridge;s6+=.13*ridge;s7+=.16*ridge+.11*(cvarea-.35)
# Secondary core as bright as the primary in a sky-compensated comparison:
# require either convexity on multiple spatial scales or considerable imbalance.
offsetpk=ramp(ratio(get(central_peak['far'])-get(skyval),get(centpix)-get(skyval),.035),.46,1.18)
s1+=.22*mini(offsetpk,maxi(merger,lopsided))
cir.compile('morphology',vec,[s0,s1,s2,s3,s4,s5,s6,s7,s8,s9])
(out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n');np.savez(out/'weights.npz',**W)
