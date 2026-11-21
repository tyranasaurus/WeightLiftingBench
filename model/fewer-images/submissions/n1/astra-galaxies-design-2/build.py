"""Compile a hand-designed astronomical morphology measurement graph. No input files."""
import sys, json, math
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
layers=[]; weights={}; sizes={'image':(3,128,128)}

def op(name, kind, ins, result_shape, **kw):
    if isinstance(ins,str): ins=[ins]
    assert name not in sizes
    sizes[name]=tuple(result_shape)
    layers.append(dict(name=name,op=kind,inputs=ins,**kw));return name

def dense(name, inp, w, bias=0):
    w=np.asarray(w,'f'); w=w.reshape(-1,math.prod(sizes[inp])); n=w.shape[0]
    weights[name+'.W']=w;weights[name+'.b']=np.broadcast_to(bias,(n,)).astype('f')
    return op(name,'dense',inp,(n,),out=n)

def reshape(name,inp,sh):return op(name,'reshape',inp,sh,shape=list(sh))
def pool(name,inp,k,kind='avgpool',st=None):
    s=sizes[inp];st=st or k
    return op(name,kind,inp,(s[0],(s[1]-k[0])//st[0]+1,(s[2]-k[1])//st[1]+1),size=list(k),stride=list(st))
def conv(name,inp,w,b=0,p=0,st=1):
    w=np.array(w,'f');o,c,k,_=w.shape;s=sizes[inp]
    weights[name+'.W']=w;weights[name+'.b']=np.broadcast_to(b,(o,)).astype('f')
    return op(name,'conv',inp,(o,(s[1]+2*p-k)//st+1,(s[2]+2*p-k)//st+1),out=o,kernel=[k,k],stride=st,padding=p)
def relu(name,inp):return op(name,'relu',inp,sizes[inp])

def join(name, inputs):
    # Concatenation of small vectors, implemented with zero-padding projections.
    ns=[math.prod(sizes[i]) for i in inputs]; total=sum(ns); j=0; stages=[]
    for k,(i,n) in enumerate(zip(inputs,ns)):
        w=np.zeros((total,n),'f');w[j:j+n]=np.eye(n);j+=n
        stages.append(dense(name+'_part'+str(k),i,w))
    return op(name,'add',stages,(total,))

# All scales below are in original (128x128) pixel units unless indicated.
bin2=pool('photon_bins','image',(2,2))
y,x=np.mgrid[:64,:64]*2.-63
r=np.hypot(x,y);ang=np.arctan2(y,x)
def unit(w):return w/w.sum()
def gauss(rad,s):return np.exp(-.5*(rad/s)**2)
sky=unit(((r>51)&(r<66)).astype(float))
raw_lum=conv('white_light',bin2,np.array([.30,.40,.30]).reshape(1,3,1,1))
# Centered L2 correlations with a continuum of projected exponential profiles.
# Search over size, flattening, rotation, and two radial light profiles. Ignore
# shape degeneracy of tiny unresolved bulges by giving the wings some weight.
qs=[.12,.18,.25,.33,.43,.55,.68,.80,.91,1.0]
Nang=16
# keep original central intensity, downweight normalization of the outer field
W=[]
for q in qs:
    for a in np.arange(Nang)*np.pi/Nang:
        u=x*np.cos(a)+y*np.sin(a);v=y*np.cos(a)-x*np.sin(a)
        for scale in [3.,4.5,6.75,10.,15.,22.5,34.]:
            # Seeing/detector blur quadrature at the high-curvature minor axis.
            rad=np.sqrt(u*u/(scale**2+1.4**2)+v*v/((scale*q)**2+1.4**2))
            for p in [1.,1.65]:
                t=np.exp(-rad**p); taper=gauss(r,43)*(1-.68*gauss(r,3.5))
                t=t-(t*taper).sum()/taper.sum()
                W.append((t*taper/np.sqrt((t*t*taper).sum())).astype('f'))
corr=dense('ellipse_integrals',raw_lum,W);del W
corr=reshape('ellipse_grid',corr,(len(qs),Nang,14))
maxscale=pool('profile_search',corr,(1,14),'maxpool')
maxshape=pool('ellipse_peaks',maxscale,(Nang,1),'maxpool')

# Isophotal quadrupoles: two soft surface-brightness floors, several apodization
# radii. Angular-flux ellipticity is (1-q)/(1+q) for elliptical isophotes in
# a sufficiently large aperture. Bright inner isophotes give a bulge diagnostic.
# Thresholds refer to the image encoding, not to particular training examples.
levels=[.075,.16,.32,.54]
iso=conv('isophote_levels',raw_lum,np.ones((len(levels),1,1,1)), -np.array(levels))
iso=relu('isophote_light',iso)
# Also use a softly capped surface brightness to keep a foreground star from
# dominating faint isophotes.
radwin=[10.,17.,27.]
niso=len(levels)*len(radwin)
W=[]
for j,lv in enumerate(levels):
    for h in radwin:
        w=gauss(r,h)*(1-gauss(r,2.0))
        for f in [np.ones_like(r),np.cos(2*ang),np.sin(2*ang)]:
            a=np.zeros((len(levels),64,64),'f');a[j]=w*f
            if j<len(levels)-1: a[j+1]=-w*f
            W.append(a)
mom=dense('angular_light_integrals',iso,W);del W

# Photometric rings: color, concentration, and apparent emitting area. Nearby
# background galaxies are much less influential in the smaller apertures.
W=[]; phnames=[]
for key,mask in [('core',gauss(r,2.3)),('near',gauss(r,5.0)),('middle',gauss(r,10)*(1-gauss(r,4))),('outer',gauss(r,17)*(1-gauss(r,8))),('skirt',gauss(r,25)*(1-gauss(r,13)))]:
    for i,band in enumerate('rgb'):
        a=np.zeros((3,64,64),'f');a[i]=unit(mask)-sky;W.append(a);phnames.append(key+band)
phot=dense('aperture_photometry',bin2,W);del W

# Off-nuclear peaks. A secondary nucleus should stand above its *highest*
# directional local background, not just the mean (which also responds along
# a bar or edge-on disk). Grouped max over compass directions uses -ReLU(-.)
# equivalently via maxpool of the differences with the opposite sign.
# Use broad matched blobs as well to prefer extended nuclei over isolated stars.
sm=np.arange(-5,6)*2.;sy,sx=np.meshgrid(sm,sm,indexing='ij');W=[]
for scale,dist in [(1.8,6.),(3.,9.)]:
    center=unit(np.exp(-(sx*sx+sy*sy)/(2*scale**2)))
    for a in np.arange(8)*np.pi/4:
        neighbor=unit(np.exp(-((sx-dist*np.cos(a))**2+(sy-dist*np.sin(a))**2)/(2*scale**2)))
        W.append((neighbor-center)[None])
# central 44x44 grid via unpadded kernel followed by unpadded pooling if needed
pe=conv('nuclear_surrounds',raw_lum,W,st=2);del W
side=sizes[pe][1];pe=reshape('nuclear_direction_grid',pe,(2,8,side*side))
pe=pool('least_peak_contrast',pe,(8,1),'maxpool');pe=reshape('nuclear_field',pe,(2,side,side))
# Radially localized search with log-polar sample locations; bilinear interp at
# each, so do not use a giant dense diagonal spatial mask.
yp,xp=np.mgrid[:side,:side]*4.+(-63+10)
W=[]
for band in range(3):
    for scale in range(2):
        for a in np.arange(24)*2*np.pi/24:
            for d in [[4.,7.,10.],[10.,15.,20.],[20.,27.,35.]][band]:
                ww=np.maximum(0,1-np.abs(xp-d*np.cos(a))/4)*np.maximum(0,1-np.abs(yp-d*np.sin(a))/4)
                ar=np.zeros((2,side,side),'f');ar[scale]=-ww;W.append(ar)
peak=dense('companion_candidates',pe,W);del W
peak=reshape('companion_locations',peak,(6,1,72));peak=pool('companion_peaks',peak,(1,72),'maxpool')

# Build an approximately centered light field and suppress very faint sky
# fluctuations. Matched soft-annulus harmonic filters below use 2-pixel bins.
# Pad for a symmetric 32x32, 3-pixel-bin field by using resampling templates
# directly in the original 64-bin coordinates. Crop to cut large matmul size.
# Use 2x2 bins throughout the harmonic branch. Matrix view: a 1x1
# convolution can apply the same central-row selector to each column.
def central_cut(name,inp,n,cut):
    tmp=reshape(name+'_rows',inp,(n,n,1)); ww=np.eye(n,dtype='f')[cut:n-cut];m=n-2*cut
    tmp=conv(name+'_cut_rows',tmp,ww[:,:,None,None]); tmp=reshape(name+'_row_stack',tmp,(1,m,n))
    # Use shared scalar column selector via dense over rows implemented by a
    # convolution right after a flattened row stack would mix axes. Gather
    # row-stack columns by a linear matrix per row encoded as stride 1,
    # kernel=1 via reshaping isn't possible without transpose. Keep all columns.
    return tmp
# Suppress very low level sky features and cap intense interlopers.
# Luminosity field intentionally includes the reddish bulge rather than only
# the blue band, since blue arm regions can be redshifted or dust attenuated.
field=conv('polar_intensity_levels',raw_lum,np.ones((2,1,1,1)),[-.055,-.69]);field=relu('polar_emission',field)
field=conv('bounded_polar_light',field,np.array([1,-1]).reshape(1,2,1,1));field=central_cut('disk',field,64,8)
yf,xf=y[8:-8],x[8:-8]; rf=r[8:-8]; af=ang[8:-8]
# Interleaved mass and circular harmonic coefficients in logarithmic elliptical
# annuli. Project into the plane of trial disks; choose the geometry at runtime
# for which the outer annuli are most nearly axisymmetric on average.
geos=[(1.,0.)]+[(q,a) for q in [.77,.52,.32] for a in np.arange(8)*np.pi/8]
ringrs=3.5*np.power(1.41,np.arange(8)); nr=len(ringrs); nm=4
# components: total; cos2,sin2; cos1,sin1; cos3,sin3; cos4,sin4.
# Save pair-symmetric azimuthal surface-brightness samples in a separate bank:
# their third-moment sign distinguishes a narrow bright ridge (arm) from a
# broad stellar disk with a dust lane. Also retain fine azimuthal residuals.
W=[]; sectors=[]
for q,a in geos:
    u=xf*np.cos(a)+yf*np.sin(a);v=(yf*np.cos(a)-xf*np.sin(a))/q
    rr=np.maximum(.2,np.hypot(u,v)); t=np.arctan2(v,u)
    for rad in ringrs:
        b=unit(np.exp(-.5*(np.log(rr/rad)/.235)**2)*gauss(rf,46))
        W.append(b.astype('f'))
        for m in [2,1,3,4]:
            for fn in [np.cos,np.sin]:
                # remove small discrete-lattice DC leakage in each kernel
                f=fn(m*t);W.append((2*b*(f-(b*f).sum())).astype('f'))
        # At least one seeing element should fit within a sector. Taper the
        # angular resolution for small projected rings near the resolution limit.
        sharp=1/(.24**2+(1.6/(q*rad))**2)
        for az in np.arange(8)*np.pi/8:
            sector=unit(b*np.exp(.25*sharp*(np.cos(2*(t-az))-1)))
            sectors.append((sector-b).astype('f'))
harm=dense('disk_plane_fourier',field,W);del W
sector=dense('arm_interarm_samples',field,np.array(sectors).reshape(len(geos),8,8,-1).transpose(1,2,0,3));del sectors
sector=reshape('arm_interarm_grid',sector,(64,len(geos),1))

# Small-vector arithmetic compiler. Linear expressions are fused into the next
# ReLU bank. Supports products and division with explicit piecewise linear
# scalar functions, so the inference graph uses only the specified primitives.
# The piecewise functions are interpolants of mathematical identities.
class Expr:
    def __init__(self, t=None,b=0.):self.t={} if t is None else t;self.b=float(b)
    def __add__(a,b):
        b=ex(b);t=a.t.copy()
        for i,v in b.t.items():t[i]=t.get(i,0.)+v
        return Expr({k:v for k,v in t.items() if v},a.b+b.b)
    __radd__=__add__
    def __sub__(a,b):return a+(-ex(b))
    def __rsub__(a,b):return ex(b)+(-a)
    def __neg__(a):return a*(-1)
    def __mul__(a,b):return Expr({i:v*b for i,v in a.t.items()},a.b*b)
    __rmul__=__mul__
    def __truediv__(a,b):return a*(1/b)

def ex(x): return x if isinstance(x,Expr) else Expr(b=x)
class Arith:
    def __init__(self,pre):self.pre=pre;self.nodes=[];self.ext=[]
    def input(self,i):
        off=sum(math.prod(sizes[j]) for j in self.ext);self.ext.append(i)
        return [Expr({-(off+k+1):1}) for k in range(math.prod(sizes[i]))]
    def R(self,a):
        a=ex(a);self.nodes.append(a);return Expr({len(self.nodes)-1:1.})
    def clip(self,x,l=0,h=1):return l+self.R(x-l)-self.R(x-h)
    def ramp(self,x,l,h):return self.clip((x-l)/(h-l))
    def ab(self,x):return self.R(x)+self.R(-x)
    def hi(self,x,y):return self.R(x-y)+y
    def lo(self,x,y):return x-self.R(x-y)
    def mx(self,xs):
        while len(xs)>1:xs=[self.hi(xs[i],xs[i+1]) if i+1<len(xs) else xs[i] for i in range(0,len(xs),2)]
        return xs[0]
    def interp(self,x,knots,vals):
        slopes=np.diff(vals)/np.diff(knots)
        # Right-anchored hinges keep k*slope of log(k) near unity. Forward
        # hinges would subtract large multiples of x at the bright end.
        return vals[-1]+sum(self.R(k-x)*float(s) for k,s in zip(knots,np.diff(np.r_[0,slopes,0])))
    def log(self,x,lo,hi,n=25):
        k=np.geomspace(lo,hi,n);return self.interp(x,k,np.log(k))
    def ratio(self,x,y,lo,hi,n=24,cap=1.5):
        d=self.log(x,lo,hi,n)-self.log(y,lo,hi,n)
        k=np.linspace(-5,math.log(cap),23);return self.interp(d,k,np.exp(k))
    def mag(self,x,y):
        # octagon approximation to magnitude (3.9% worst relative error before correction)
        a=self.ab(x);b=self.ab(y);return .96*(self.hi(a,b)+.4142*self.lo(a,b))
    def finish(self,name,outputs):
        pre=self.pre; external=join(pre+'_inputs',self.ext) if len(self.ext)>1 else self.ext[0]
        n=math.prod(sizes[external]); depths={-i-1:0 for i in range(n)}
        for i,e in enumerate(self.nodes):depths[i]=1+max((depths[j] for j in e.t),default=0)
        maxd=max(depths.values()); uses={i:0 for i in depths}
        for i,e in enumerate(self.nodes):
            for j in e.t:uses[j]=max(uses[j],depths[i])
        outputs=list(map(ex,outputs))
        for e in outputs:
            for j in e.t:uses[j]=maxd+1
        src=external; positions={-i-1:(i,None) for i in range(n)};N=n
        def matrix(exprs):
            W=np.zeros((len(exprs),N),'f');b=[]
            for k,e in enumerate(exprs):
                b.append(e.b)
                for i,v in e.t.items():
                    p,m=positions[i];W[k,p]+=v
                    if m is not None:W[k,m]-=v
            return W,b
        for d in range(1,maxd+1):
            es=[];new={}
            for k in positions:
                if uses[k]>d:
                    pos=len(es);es.append(Expr({k:1}));neg=None
                    if k<0:neg=len(es);es.append(Expr({k:-1}))
                    new[k]=(pos,neg)
            for k,e in enumerate(self.nodes):
                if depths[k]==d and uses[k]>d:new[k]=(len(es),None);es.append(e)
            if not es:continue
            w,b=matrix(es);src=dense(pre+'_affine'+str(d),src,w,b);src=relu(pre+'_hinges'+str(d),src)
            N=len(es);positions=new
        w,b=matrix(outputs);return dense(name,src,w,b)

# For repeated computations over each of the disk planes, use shared 1x1
# convolutions on a (features, geometry, 1) layout. Dense input is permuted in
# weight-row order (no gigantic sparse transpose layer).
ng=len(geos); nf=nr*(1+2*nm)
w=weights[harm+'.W'];weights[harm+'.W']=w.reshape(ng,nf,-1).transpose(1,0,2).reshape(nf*ng,-1).copy();del w
harm=reshape('annuli_by_inclination',harm,(nf,ng,1))
# Operate with the same arithmetic compiler, then substitute pointwise convs.
def shared(pre,inp,make):
    # Use a temporary shape for compilation, not an actual extra network node.
    f,h,w=sizes[inp];fake=pre+'_vector';sizes[fake]=(f,);start=len(layers)
    T=Arith(pre);res=make(T,T.input(fake));last=T.finish(pre+'_summary',res); del sizes[fake]
    for layer in layers[start:]:
        n=layer['name'];op0=layer['op']
        if layer['inputs']==[fake]:layer['inputs']=[inp]
        sz=sizes[n][0]; sizes[n]=(sz,h,w)
        if op0=='dense':
            layer.update(op='conv',kernel=[1,1],padding=0,stride=1)
            weights[n+'.W']=weights[n+'.W'][:,:,None,None]
    return last

# Pair-opposed, narrow arms make peaks sharper than the interarm troughs and
# stronger bends in the azimuthal brightness around a ring. Residual against
# neighboring rings at the same disk azimuth responds to arms, not straight bars.
def profile_rules(A,z):
    w=[.08,.24,.70,1.25,1.6,1.8,1.55,.90]; skew=Expr(); ridge=Expr(); crossing=Expr()
    for j in range(nr):
        v=z[j*8:j*8+8];mx=A.mx(v);mn=-A.mx([-a for a in v]);skew+=(mx+mn)*w[j]
        ridge+=sum(A.ab(v[k]-.5*(v[(k-1)%8]+v[(k+1)%8])) for k in range(8))*(w[j]/8)
        if 0<j<nr-1:
            crossing+=sum(A.ab(v[k]-.5*(z[8*j-8+k]+z[8*j+8+k])) for k in range(8))*(w[j]/8)
    return [skew,ridge,crossing]
prof=shared('azimuthal_profiles',sector,profile_rules)
# Join by addition with channel offsets, as accomplished by two very small
# shared projections (not by duplicating the large spatial banks).
nf2=sizes[harm][0]+sizes[prof][0]; mat=np.eye(nf2,dtype='f')[:,:,None,None]
pa=conv('angular_channels',harm,mat[:,:nf]);pb=conv('profile_channels',prof,mat[:,nf:])
harm=op('disk_planes','add',[pa,pb],sizes[pa])

# Harmonic features use a stable contrast denominator: an annulus must have
# significant light relative to core and sky before its angular pattern counts.
# Vector phases yield spiral evidence from radial rotation of the m=2 pattern,
# without rotation/handedness dependence. m=1,3 detect lopsidedness and knots.
def plane_rules(A,vec):
    I=[]; M=[]; mag=[]
    for j in range(nr):
        s=vec[9*j:9*j+9];I.append(s[0]);M.append([(s[k],s[k+1]) for k in [1,3,5,7]])
        mag.append([A.mag(*pair) for pair in M[-1]])
    # Sum flux on a logarithmic radial grid, weighting somewhat toward the
    # outskirts while keeping field objects at large distances from dominating.
    rw=[.27,.52,.83,1.15,1.48,1.75,1.6,1.0]
    D=.055+sum(v*w for v,w in zip(I,rw))
    def av(xs,ww=rw):return sum(x*w for x,w in zip(xs,ww))
    def rat(v,den=D):return A.ratio(v,den,.001,12,26,2.)
    ratios=[rat(av([m[k] for m in mag])) for k in range(4)]
    # Coherent quadrupole relative to total quadrupole. A fixed ellipse has
    # nearly constant phase, while spiral light winds up radially. Coherent
    # sum downweights low-SNR phase measurements without angle division.
    def coh(idx,ww):return A.mag(av([m[idx][0] for m in M],ww),av([m[idx][1] for m in M],ww))
    coherent=rat(coh(0,rw))
    # coherent outer quadrupole is an indication that the deprojection is wrong.
    wo=[0.,0.,.12,.5,1.15,1.7,1.8,1.0]; wi=[.4,.9,1.25,1.0,.45,.1,0,0]
    dout=.04+av(I,wo); di=.05+av(I,wi)
    outer=rat(coh(0,wo),dout);inner=rat(coh(0,wi),di)
    # Match winding in log radius: rotating the two-fold coefficient by
    # -2*cot(pitch)*log(r) demodulates a logarithmic spiral. Use both signs.
    pitch=[]
    for slope in [1.35,2.5,4.4,6.9]: # inclination-corrected angular slopes = 2*cot(pitch)
        vals=[]
        for sign in [-1,1]:
            phase=sign*slope*np.log(ringrs/ringrs[0]);c=np.cos(phase);s=np.sin(phase)
            re=av([a[0][0]*cc-a[0][1]*ss for a,cc,ss in zip(M,c,s)])
            im=av([a[0][1]*cc+a[0][0]*ss for a,cc,ss in zip(M,c,s)])
            vals.append(A.mag(re,im))
        pitch.append(rat(A.hi(*vals)))
    # How much does ellipticity persist through the outer disk? Bars create an
    # inner, constant-phase quadrupole followed by a rapid fall or twist.
    # Look for this at more than one scale; subtract a soft faint-light floor.
    bump=[]
    for j in [1,2,3,4]:
        invec=[M[t][0] for t in [j-1,j,j+1]]
        v=A.mag(sum(z[0] for z in invec),sum(z[1] for z in invec));den=sum(I[j-1:j+2])+.10
        vo=A.mag(M[j+2][0][0]+M[j+3][0][0],M[j+2][0][1]+M[j+3][0][1]);do=I[j+2]+I[j+3]+.055
        bump.append(rat(v,den)-.70*rat(vo,do)-.6*A.R(.085-do))
    skew,ridge,crossing=vec[9*nr:]; skew=rat(A.R(skew))-rat(A.R(-skew))
    return [*ratios,coherent,outer,inner,*pitch,A.mx(bump),skew,rat(ridge),rat(crossing)]
planes=shared('polar_analysis',harm,plane_rules)
# Choose plane with a small coherent quadrupole and small m=4 projection
# artifact. Geometries with similar merit contribute comparably via soft gating.
# Implement winner-tolerant average using soft triangular support and ratios.
T=Arith('choose');v=T.input(planes);fnum=sizes[planes][0]
arr=[[v[k*ng+i] for k in range(fnum)] for i in range(ng)]
cost=[.56*f[5]+.24*f[4]+.20*f[0]+.11*f[3]+(.035*(1-q)) for f,(q,_) in zip(arr,geos)]
best=-T.mx([-a for a in cost])
# Min-cost envelopes (no averaging of unrelated projection angles). If an
# uncertain deprojection makes a feature implausibly large it is preferable to
# be conservative: positive evidence should survive several plausible views.
# Obtain approximate nearest plane response by both upper and lower envelopes.
polnames=['two','one','three','four','coherent','outerell','innerell','slow','open','medium','fast','barbump','skew','ridge','crossing']
pol=[]
for k in range(fnum):
    upper=T.mx([f[k]-4.5*(c-best) for c,f in zip(cost,arr)])
    lower=-T.mx([-f[k]-4.5*(c-best) for c,f in zip(cost,arr)])
    pol.append(.5*(upper+lower))
pol.append(arr[0][5]);polnames.append('skyell')
# Very thin deprojections magnify anisotropic PSF. Do not infer global axis
# ratio from the selected plane; use separate elliptical photometry instead.
polar=T.finish('selected_disk_plane',pol)

# Direct asymmetry and small-scale residuals. Photometrically normalize
# absolute difference across a half-turn; allow a small range of centroid
# offsets so that a subpixel targeting offset does not mimic a merger.
# A 2-pixel-bin polar reflection is implemented as dense differences on a
# compact set of paired interpolation footprints, not storage of exemplars.
W=[];aw=[]
for dx,dy in [(0,0),(-2.4,0),(2.4,0),(0,-2.4),(0,2.4)]:
    for rad in [4,7,11,16,23,32]:
        n=12
        for a in np.arange(n)*np.pi/n:
            ux=rad*np.cos(a);uy=rad*np.sin(a)
            sig=max(1.65,rad*.17)
            wp=unit(np.exp(-((xf-dx-ux)**2+(yf-dy-uy)**2)/(2*sig**2)))
            wn=unit(np.exp(-((xf-dx+ux)**2+(yf-dy+uy)**2)/(2*sig**2)))
            W.append((wp-wn).astype('f'));aw.append((wp+wn).astype('f'))
wd=np.array(W);wa=np.array(aw);del W,aw
w=np.concatenate([wd,-wd,wa],axis=0);del wd,wa
asym=dense('bisector_differences',field,w);del w
asym=relu('bisector_absolute_parts',asym)
# aggregate per candidate center before taking robust minimum
w=np.zeros((10,3,5,6,12),'f')
for i in range(5):
    rw=np.array([.5,.75,1.,1.3,1.5,1.2])[:,None]/12
    w[2*i,:2,i]=rw;w[2*i+1,2,i]=rw
asym=dense('half_turn_integrals',asym,w)

# CAS-style rotational asymmetry. A pi rotation reverses the flat raster.
# Sparse two-entry filters gather opposite samples, with exactly the same
# binning for the target and for estimation of the sky residual.
W=[];ca=[];idx=np.arange(-47,48,2.)
for yy in idx[idx<0]:
    for xx in idx:
        rr=np.hypot(xx,yy)
        if rr>47:continue
        w=np.zeros((64,64),'f');jy=int((yy+63)/2);ix=int((xx+63)/2)
        w[jy,ix]=1;w[63-jy,63-ix]=-1
        W.append(w);ca.append(rr)
casraw=dense('opposite_pixels',raw_lum,W);casabs=relu('positive_differences',casraw)
ca=np.array(ca); st=np.array([unit(gauss(ca,v)*(1-gauss(ca,2.5))) for v in [7,12,19]])
back=unit((ca>40).astype(float));W=st-back
# 2*ReLU(d)-d gives both signs with one nonlinear layer.
cas1=dense('rectified_apertures',casabs,2*W);cas2=dense('signed_apertures',casraw,-W)
casdif=op('background_corrected_residuals','add',[cas1,cas2],sizes[cas1])
casmass=dense('asymmetry_denominators',raw_lum,[2*(unit(gauss(r,v)*(1-gauss(r,2.5)))-sky) for v in [7,12,19]])
C0=Arith('cas');dd=C0.input(casdif);ds=C0.input(casmass)
cf=C0.finish('cas_asymmetry',[C0.ratio(C0.R(t),C0.R(u)+.025,.001,3,28,1.5) for t,u in zip(dd,ds)])

# A second check for a separated core: retain peak contrast on the full 2px
# lattice instead of blurring an off-axis unresolved peak between radial
# samples. Select only off-nuclear sites from a compact central square.
# Smaller compass kernels leave the central ~90px available for this search.
ny,nx=np.mgrid[-3:4,-3:4]*2.;w=[]
for sig,dist in [(1.4,4.5),(2.1,6.)]:
    cen=unit(np.exp(-(nx*nx+ny*ny)/(2*sig**2)))
    for a in np.arange(8)*np.pi/4:
        near=unit(np.exp(-((nx-dist*np.cos(a))**2+(ny-dist*np.sin(a))**2)/(2*sig**2)))
        w.append((near-cen)[None])
pp=conv('local_peak_test',raw_lum,w);ps=sizes[pp][1];pp=reshape('local_direction_grid',pp,(2,8,ps*ps))
pp=pool('local_least_contrast',pp,(8,1),'maxpool');pp=reshape('local_peak_maps',pp,(2,ps,ps))
# Implement masked spatial max with a fixed radial bias added to both maps.
# Positive inner exclusion also guards against a two-pixel astrometric error.
yb,xb=np.mgrid[:ps,:ps]*2.-(ps-1);rpeak=np.hypot(xb,yb)
const=dense('companion_mask',casdif,np.zeros((2*ps*ps,3)),np.tile((4*((rpeak<7)|(rpeak>32))).ravel(),2));const=reshape('companion_mask_map',const,(2,ps,ps))
pp=op('off_center_peak_maps','add',[pp,const],sizes[pp]);pp=conv('off_center_prominence',pp,(-np.eye(2))[:,:,None,None])
# Staged pooling avoids creating large im2col tensors at evaluation time.
pp=pool('off_center_pairs',pp,(2,2),'maxpool');pp=pool('off_center_max',pp,(ps//2,ps//2),'maxpool')

# Axis-ratio readout: interpolate location of the best correlation using a
# soft window about the maximum. Only smooth positive powers of analytically
# evaluated correlations are compared. Axis ratio near one is poorly resolved:
# the whole curvature of the peak is used rather than a hard winning index.
S=Arith('shape');cv=S.input(maxshape);mm=S.input(mom)
# Direct posterior-like soft peak widths, using a contrast-relative triangular
# kernel. Compute ratio after summation to avoid instability in faint sources.
ma=S.mx(cv);floor=.009*ma+.0002
tri=[S.R(floor-(ma-c)) for c in cv];total=sum(tri)+.00001
qfit=S.ratio(sum(q*t for q,t in zip(qs,tri)),total,1e-5,25,48,1.)
# A boundary correction to avoid soft interpolation pulling circular disks
# too far from q=1. Also use second-best wing information at the thin end.
qfit=qfit+.038*S.ramp(qfit,.76,.93)-.015*S.ramp(-qfit,-.26,-.14)
qmom=[]
for j in range(niso):
    mass,xx,xy=mm[3*j:3*j+3];ell=S.ratio(S.mag(xx,xy),mass+.08,.01,2000,47,1.)
    qmom.append(S.interp(ell,np.linspace(0,1,17),[(1-a)/(1+a) for a in np.linspace(0,1,17)]))
shnames=['qfit','qout','qmid','qinner','highmass','qfaint']
# q is biased high in circular apertures. Small aperture for bright inner
# isophotes limits contamination; outer low isophotes get a larger aperture.
sh=S.finish('ellipse_measurements',[qfit,.4*qmom[1]+.6*qmom[2],.5*qmom[4]+.5*qmom[5],qmom[7],mm[3*7],qmom[0]])

# Sum-light ratios in real colors, with a soft noise floor. A blue stellar disk,
# red bulge, and blue knots play different roles from a smooth yellow spheroid.
P=Arith('light');v=P.input(phot);p=dict(zip(phnames,v)); pe=P.input(peak); avv=P.input(asym); finepeak=P.input(pp)
L={k:sum(p[k+c]*w for c,w in zip('rgb',[.3,.4,.3])) for k in ['core','near','middle','outer','skirt']}
blend=lambda a,b,k: a*k+b*(1-k)
# Center-to-outer surface brightness ratios and relative color in apertures.
def color(k):
    # dimensionless opponent color. Taking log ratios removes intensity scale.
    return P.log(P.R(p[k+'b'])+.024,.012,1.3)-P.log(P.R(p[k+'r'])+.024,.012,1.3)
cn=color('near');cm=color('middle');co=color('outer')
cons=P.ratio(P.R(L['near'])+.01,P.R(L['middle'])+.01,.008,1.3,28,12.)
size=P.ratio(P.R(L['outer'])+.005,P.R(L['near'])+.01,.003,1.3,28,1.6)
size2=P.ratio(P.R(L['skirt'])+.005,P.R(L['middle'])+.01,.003,1.3,28,1.4)
cen=P.ratio(P.R(L['core'])+.01,P.R(L['near'])+.01,.003,1.3,28,4.)
turns=[P.ratio(avv[2*i],avv[2*i+1]+.09,.001,8,35,1.) for i in range(5)]
turn=-P.mx([-t-(.012 if i else 0.) for i,t in enumerate(turns)])
peaks=[P.ratio(P.R(v),P.R(L['core'])+.11,.001,1.5,26,2.) for v in pe]
lnames=['cc','cm','co','concentration','extent','wings','cusp','asym','corelum','outlum','midlum']+['pk'+str(i) for i in range(6)]
lnames+=['nucleus','broadernucleus']
fp=[P.ratio(P.R(v),P.R(L['core'])+.08,.001,1.5,26,2.) for v in finepeak]
light=P.finish('stellar_light_measurements',[cn,cm,co,cons,size,size2,cen,turn,L['core'],L['outer'],L['middle'],*peaks,*fp])

# Isophote thickness in silhouettes measured in elliptical rather than circular
# apertures. A bulge should raise the high-isophote thickness beyond what's
# explained by seeing at the measured major-axis size. Re-use the 2D ellipse
# bank, asking for orientation rather than selecting q/scale a second time.
K=12;orient=np.pi*np.arange(K)/K;W=[]
for a in orient:
    u=x*np.cos(a)+y*np.sin(a);v=y*np.cos(a)-x*np.sin(a)
    # Use wide major-axis support at low isophotes. More conservative window
    # in the core both attenuates unrelated stars and follows seeing which
    # often merges very bright central isophotes together.
    for j,(maj,minor) in enumerate([(27,10),(20,9),(13,7)]):
        aperture=np.exp(-.5*((u/maj)**2+(v/minor)**2));lv=j
        for f in [np.ones_like(r),u*u,v*v]:
            arr=np.zeros((len(levels),64,64),'f');arr[lv]=f*aperture;arr[lv+1]=-arr[lv];W.append(arr)
axraw=dense('strip_inertias',iso,np.array(W).reshape(K,9,-1).transpose(1,0,2));del W
axraw=reshape('strip_inertia_grid',axraw,(9,K,1))
def strip_rules(B,z):
    feat=[]
    for j in range(3):
        a,b,c=z[3*j:3*j+3]; lmass=B.log(a+.01,.01,500,39)
        lu=.5*(B.log(b+.01,.01,1e5,56)-lmass);lv=.5*(B.log(c+.01,.01,1e5,56)-lmass)
        feat.extend([lu,lv]);
        if j==2:feat.append(B.ramp(a,.20,2.0))
    return feat
strip=shared('strip_widths',axraw,strip_rules)
B=Arith('align');vv=B.input(strip);vs=[[vv[k*K+j] for k in range(7)] for j in range(K)]
# Infer the actual major axis consistently for all isophotes. The strip with
# the greatest major/minor contrast on prominent isophotes is well aligned.
crit=[.55*(u[0]-u[1])+.45*(u[2]-u[3]) for u in vs];best=B.mx(crit)
def sel(idx):return B.mx([u[idx]-5*(best-c) for c,u in zip(crit,vs)])
val=[sel(j) for j in range(7)];ua,va,ub,vb,uc,vc,brightness=val
qp=lambda d:B.interp(d,np.linspace(-3,0,21),np.exp(np.linspace(-3,0,21)))
stripnames=['qa','qb','qc','length','bulgew','thinhi']
# In a pure elliptical spheroid the RMS axis ratio changes only mildly with
# isophote. Disks with a bulge generally thicken on brighter isophotes.
ax=B.finish('major_axis_measurements',[qp(va-ua),qp(vb-ub),qp(vc-uc),ua,.55*(vc-uc-vb+ub)+.45*(vb-ub-va+ua),brightness])

# Residuals in the color of the stellar populations. A young arm can have
# noticeable g-z contrast even if overlaying a much brighter old disk. At
# fixed deprojected radius a smooth color gradient has zero angular Fourier
# modes. Compare the modes against the mean blue light including a small sky
# floor. Use the luminosity-only geometry cost, not a color-selected plane.
chroma=conv('young_minus_old',bin2,np.array([-.72,0.,1.]).reshape(1,3,1,1));chroma=central_cut('populations',chroma,64,8)
blueim=conv('blue_luminosity',bin2,np.array([0.,0.,1.]).reshape(1,3,1,1));blueim=central_cut('blue_disk',blueim,64,8)
# Collect pairs of rings into broad, overlapping radial bands for photon SNR.
# The existing Fourier filters are purely geometrical; linear combinations
# here are sums over apertures, not averages over example templates.
fw=weights['disk_plane_fourier.W'].reshape(nr,9,ng,-1)
bands=[(0,1,2),(2,3,4),(4,5,6),(5,6,7)];cbnames=[];cols=[];bw=[]
for ix in bands:
    ringbank=sum(fw[j] for j in ix)/len(ix)
    for k in [1,2,3,4,5,6]:cols.append(ringbank[k])
    bw.append(ringbank[0]-sky[8:-8].reshape(1,-1)/sky[8:-8].sum())
cmod=dense('chromatic_harmonics',chroma,cols);cbg=dense('local_blue_photometry',blueim,bw)
cmod=reshape('chromatic_grid',cmod,(6*len(bands),ng,1));cbg=reshape('blue_grid',cbg,(len(bands),ng,1))
cnf=7*len(bands); eye=np.eye(cnf,dtype='f')[:,:,None,None]
cmt=conv('chromatic_channels',cmod,eye[:,:6*len(bands)]);cat=conv('local_blue_channels',cbg,eye[:,6*len(bands):])
colorgrid=op('stellar_population_field','add',[cmt,cat],sizes[cmt])
def chromrules(B,z):
    mass=[B.R(t) for t in z[24:28]];w=[.55,1.,1.45,.8];den=.07+sum(a*b for a,b in zip(w,mass))
    def mode(k):return sum(w[j]*B.mag(z[j*6+k],z[j*6+k+1]) for j in range(4))
    return [B.ratio(mode(k),den,.001,6,28,1.8) for k in (0,2,4)]
colgrid=shared('color_patterns',colorgrid,chromrules)
T=Arith('adopt');cr=T.input(colgrid);pv=T.input(planes);arr=[[pv[k*ng+j] for k in range(fnum)] for j in range(ng)]
cs=[.56*f[5]+.24*f[4]+.20*f[0]+.11*f[3]+.035*(1-q) for f,(q,_) in zip(arr,geos)]
cp=[]
for k in range(3):
    v=cr[ng*k:ng*(k+1)];cp.append(.5*(T.mx([f-4.5*c for f,c in zip(v,cs)])-T.mx([-f-4.5*c for f,c in zip(v,cs)])))
chrom=T.finish('deprojected_color_patterns',cp)

# Final hand-written decision functions. Each morphology family has an evidence
# score in [roughly 0,1]. Its subtypes compete using only within-family clues.
A=Arith('morphology');get=lambda names,src:dict(zip(names,A.input(src)))
s=get(shnames,sh);l=get(lnames,light);h=get(polnames,polar);ax=get(stripnames,ax)
R=A.ramp;C=A.clip;abs_=A.ab;mx=A.mx;hi=A.hi;lo=A.lo
cas=A.input(cf); coltwo,colone,colthree=A.input(chrom)
# Fit dominates the shape of isolated symmetric sources. Angular moments add
# independent evidence on low isophotes, especially for a thin disk plus bulge.
q=.67*s['qfit']+.21*s['qmid']+.12*s['qout']
qd=.45*s['qfit']+.17*s['qout']+.15*s['qmid']+.12*ax['qa']+.11*ax['qb']
size=R(l['extent'],.11,.42); wing=R(l['wings'],.24,.63)
# Blue excess relative to the central population is characteristic of spiral
# arms. Using outer absolute color as well avoids mistaking reddened sky for
# an extremely blue disk.
blue=R(.58*l['cm']+.42*l['co'],-.24,.20)
grad=R(.55*l['cm']+.45*l['co']-l['cc'],.08,.40)
veryblue=R(l['cm'],-.04,.42)
# Rotation / angular substructure that cannot be explained by a tilted disk.
twist=R(h['two']-h['coherent'],.035,.23)
winding=R(hi(h['medium'],h['fast'])-.6*h['coherent'],.045,.20)
knots=R(.52*h['three']+.48*h['four'],.068,.25)
olop=R(h['one'],.13,.50)
irreg=R(l['asym'],.15,.44)
regbar=R(h['barbump'],.13,.44)
# Projection and seeing effects can cause spurious bar evidence in small, very
# flat galaxies. Taper its contribution there.
bar=regbar-.20*R(-q,-.42,-.22)-.10*(1-size)
ridge=R(h['ridge'],.023,.085);cross=R(h['crossing'],.020,.085)
structure=.25*twist+.17*winding+.17*knots+.21*C(bar)+.10*ridge+.10*cross
starforming=.50*blue+.32*grad+.18*veryblue
# Spiral evidence; featureless disk with warm, smooth light is not rewarded
# merely for being a slightly oblate ellipse.
spiral=.05+.31*starforming+.37*structure+.17*size+.10*wing+.09*lo(blue,grad)
spiral+=.10*lo(bar,grad)-.10*R(-q,-.36,-.18)
# Disturbed objects often have a small/ill-defined central peak or lopsided
# emission. An irregular which is smooth in color can still win by asymmetry.
asymm=.66*irreg+.34*olop
piec=R(.64*h['three']+.36*h['one'],.14,.47)
disturb=.04+.52*asymm+.18*piec+.12*veryblue+.09*(1-grad)+.09*R(-l['cusp'],-1.38,-1.)
disturb-=.12*lo(bar,grad)+.08*lo(winding,grad)
# Distinct second peaks at several distances, normalized to central brightness.
# Downweight distant companions unless there is asymmetric light in between.
close=hi(l['pk0'],l['pk1']); mid=hi(l['pk2'],l['pk3']); far=hi(l['pk4'],l['pk5'])
comp=mx([close-.07,mid,far-.085]); comp=R(comp,.06,.46)
merger=.02+.47*comp+.40*asymm+.13*piec+.055*size-.075*(1-irreg)
# Disk-vs-spheroid: an extremely narrow long axis is strong disk evidence.
# A somewhat thicker elongated system needs an extended disk, a bulge sticking
# out of the plane, a dust lane (reddening), or blue disk colors.
flat=R(-qd,-.42,-.16);superflat=R(-qd,-.27,-.14)
thick=R(s['qinner']-s['qout'],.035,.23)-.22*(1-R(s['highmass'],.35,3.2))
# Outer extent, length relative to width, inner spheroid, central excess.
edge=.015+.49*flat+.19*superflat+.145*size+.085*C(thick)+.11*starforming+.055*wing
elong=R(ax['length'],math.log(8),math.log(18))
edge+=.085*lo(flat,elong)+.055*lo(flat,wing)-.10*irreg
# Smooth class evidence is the reference; marginally blue small objects may be
# smooth if there is no spatial evidence of arms. Asphericity alone is neutral.
smooth=.655-.14*asymm-.085*structure-.09*starforming-.035*size
# Sufficiency rules (alternative explanations, rather than extra terms in
# the averages above). Compare to the SAME smooth hypothesis: each margin
# changes sign at a physically interpretable decision boundary. In particular,
# the original CAS asymmetry criterion of about 0.35 uses difference/flux;
# our paired measurement is difference/(flux in BOTH halves), hence the 2.
compact_asym=2*(.48*cas[0]+.34*cas[1]+.18*cas[2])
wide_asym=2*(.18*cas[0]+.37*cas[1]+.45*cas[2])
high_asym=.5*compact_asym+.5*wide_asym
both=lo(compact_asym,wide_asym); peak=hi(l['nucleus'],l['broadernucleus'])
merger=hi(merger,smooth+lo(.85*(wide_asym-.33),1.2*(peak-.20))+.10*lo(irreg,piec))
merger=hi(merger,smooth+lo(.55*(wide_asym-.19),1.05*(peak-.44)))
disturb=hi(disturb,smooth+.62*(both-.28)+.09*piec-.10*lo(grad,winding)-.08*R(peak,.18,.6))
# Strong flattened morphology can be diagnostic without *any* young stellar
# population. Limit this extra vote when faint-isophote elongation is absent.
qouter=.65*ax['qa']+.35*ax['qb']
edge=hi(edge,smooth+lo(1.8*(.295-qd)+.16*(elong-.4)+.13*C(thick),1.3*(.36-qouter)+.14*elong)-.13*asymm)
# Resolved arms, a bar with a surrounding disk, or a blue stellar disk can
# each be sufficient. A turn in isophote PA is evidence only if the nonzero
# winding frequency is appreciably stronger than its zero-frequency leakage.
wave=hi(h['open'],hi(h['medium'],h['fast']))
resolved=R(h['two']-h['coherent'],.025,.145)
wavewin=R(wave-.76*h['coherent'],.035,.145)
armvote=.47*resolved+.30*wavewin+.23*cross
colourvote=.63*blue+.37*grad
margin=mx([.46*(armvote-.49)+.11*(starforming-.35),
           .41*(colourvote-.51)+.14*(structure-.25)+.06*(wing-.5),
           .50*(bar-.48)+.09*(grad-.33),
           .33*(knots-.54)+.15*(blue-.49)+.09*(ridge-.35),
           .39*(R(coltwo+.65*colthree,.055,.245)-.50)+.12*(grad-.4)+.08*(resolved-.4)-.08*R(colone,.16,.40)])
spiral=hi(spiral,smooth+margin-.22*asymm-.22*R(-qd,-.35,-.19))
# Within-family Voronoi-like partitions; penalty 0 for the winning side.
# Smooth apparent axial ratio boundaries: round vs oval, oval vs cigar.
score=[disturb,merger,
       smooth-1.7*A.R(.806-q),
       smooth-1.7*(A.R(q-.806)+A.R(.425-q)),
       smooth-1.7*A.R(q-.425)]
# Impose slight separation even if an object lies exactly on a family boundary.
# Bar: straight persistent central quadrupole embedded in a rounder disk;
# additionally compare quadrupole in the deprojected core vs outer mean.
barprob=.55*C(bar)+.22*R(h['innerell']-.65*h['outerell'],.07,.36)+.15*R(h['slow']-.5*h['fast'],.04,.23)+.08*grad
# Loose spiral arms retain strong twofold contrast at lower radial frequencies,
# are relatively blue/clumpy and generally have a less concentrated bulge.
loose=.32*R(hi(h['slow'],h['open'])-hi(h['fast'],h['medium']),-.065,.11)+.16*twist+.12*knots+.13*veryblue+.09*R(-l['cusp'],-1.55,-1.10)+.05*olop+.13*R(h['skew'],.025,.14)
score.extend([spiral-.55*A.R(.52-barprob),spiral-.55*(A.R(barprob-.52)+A.R(loose-.53)),spiral-.55*(A.R(barprob-.52)+A.R(.53-loose))])
# Bulge: compare ellipticity on nested isophotes and central light excess.
# A sizeable bulge rounds the centre even when the outskirts are very thin.
bulge=.22*C(thick)+.22*lo(R(ax['bulgew'],.08,.4),ax['thinhi'])+.14*R(l['cusp'],1.24,1.67)+.10*R(l['concentration'],1.9,3.65)+.12*R(s['qinner'],.26,.58)+.10*R(-l['cc'],.05,.40)+.10*lo(ax['thinhi'],R(ax['qc'],.30,.59))
score.extend([edge-.50*A.R(bulge-.49),edge-.50*A.R(.49-bulge)])
A.finish('class_scores',score)
(out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n'); np.savez(out/'weights.npz',**weights)
