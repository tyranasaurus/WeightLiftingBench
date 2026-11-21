"""Build an analytic photometry / polar isophote network. No input files."""
import json
import math
from pathlib import Path
import sys
import numpy as np

outdir = Path(sys.argv[1]); outdir.mkdir(exist_ok=True, parents=True)
layers=[]; weights={}; shapes={'image':(3,128,128)}

def layer(name,op,inputs,result_shape,**kw):
    assert name not in shapes
    layers.append(dict(name=name,op=op,inputs=inputs,**kw)); shapes[name]=tuple(result_shape); return name

def dense(name,src,w,b=0.):
    w=np.asarray(w,np.float32).reshape(-1,math.prod(shapes[src])); no=len(w)
    weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(np.float32(b),(no,)).copy()
    return layer(name,'dense',[src],(no,),out=no)

def point(name,src,w,b=0.):
    ci,h,v=shapes[src]; w=np.asarray(w,np.float32).reshape(-1,ci,1,1); no=len(w)
    weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(np.float32(b),(no,)).copy()
    return layer(name,'conv',[src],(no,h,v),out=no,kernel=[1,1],stride=1,padding=0)

def reshape(name,x,sz):
    assert math.prod(shapes[x])==math.prod(sz);return layer(name,'reshape',[x],sz,shape=list(sz))

def add(name,*xs):
    assert len({shapes[x] for x in xs})==1;return layer(name,'add',list(xs),shapes[xs[0]])

def relu(name,x): return layer(name,'relu',[x],shapes[x])
def pool(name,x,h,w,what):
    c,a,b=shapes[x];return layer(name,what+'pool',[x],(c,a//h,b//w),size=[h,w],stride=[h,w])

def normalize(name,x): return layer(name,'norm',[x],shapes[x])

# Small compiler for fixed scalar linear combinations and rectifiers. Expressions
# can also denote channel-wise calculations shared between all polar angles.
class E:
    def __init__(self,t=(),b=0):self.t=dict(t);self.b=float(b)
    def __add__(self,o):
        o=expr(o); d=self.t.copy()
        for k,v in o.t.items():d[k]=d.get(k,0.)+v
        return E([(k,v) for k,v in d.items() if v!=0],self.b+o.b)
    __radd__=__add__
    def __neg__(self):return self*-1.
    def __sub__(self,o):return self+ -expr(o)
    def __rsub__(self,o):return expr(o)+ -self
    def __mul__(self,v):return E([(k,p*v) for k,p in self.t.items()],self.b*v)
    __rmul__=__mul__
    def __truediv__(self,v):return self*(1./v)

def expr(x):return x if isinstance(x,E) else E(b=x)
def refs(x):return [E([((x,j),1.)]) for j in range(shapes[x][0])]

def emit(name,es,rect=False):
    es=[expr(e) for e in es]; srcs=sorted({src for e in es for (src,i) in e.t});parts=[]
    assert srcs
    bs=[e.b for e in es]
    for j,src in enumerate(srcs):
        w=np.zeros((len(es),shapes[src][0]),np.float32)
        for i,e in enumerate(es):
            for (s,k),v in e.t.items():
                if s==src:w[i,k]=v
        nm=name if len(srcs)==1 and not rect else name+'_in'+str(j)
        parts.append((point if len(shapes[src])==3 else dense)(nm,src,w,bs if j==0 else 0))
    x=parts[0] if len(parts)==1 else add(name+'_sum' if rect else name,*parts)
    if rect:x=relu(name,x)
    return refs(x)

def hinges(name,es):return emit(name,es,True)
def absol(name,es):
    h=hinges(name,es+[-e for e in es]);n=len(es);return [h[i]+h[n+i] for i in range(n)]

def saturate(name,es,lo,hi):
    v=[(e-lo)/(hi-lo) for e in es]; h=hinges(name,v+[a-1 for a in v]);n=len(v)
    return [h[i]-h[n+i] for i in range(n)]

def maxima(name,groups):
    groups=[list(g) for g in groups]; stage=0
    while any(len(g)>1 for g in groups):
        ds=[g[i]-g[i+1] for g in groups for i in range(0,len(g)-1,2)]
        h=iter(hinges(name+str(stage),ds));stage+=1
        groups=[[g[i+1]+next(h) for i in range(0,len(g)-1,2)]+(g[-1:] if len(g)%2 else []) for g in groups]
    return [g[0] for g in groups]

def ratios(name,nums,dens):
    # One independent channel per quotient. Antipodal packing makes the mean
    # zero, while the small numerator coefficient makes the rms track d.
    arr=[];f=.025
    for n,d in zip(nums,dens):arr.extend([n*f,-n*f,d,-d])
    s=emit(name+'_pack',arr);x=next(iter(s[0].t))[0]; x=reshape(name+'_pairs',x,(len(nums),1,4))
    x=normalize(name+'_scale',x); x=reshape(name+'_flat',x,(len(arr),));v=refs(x)
    return [v[i*4]/(math.sqrt(2)*f) for i in range(len(nums))]

def norm2(name,pairs):
    # Mean absolute projection = 2/pi times the modulus; polygonal quadrature.
    dirs=np.arange(8)*np.pi/8;pr=[a*math.cos(t)+b*math.sin(t) for a,b in pairs for t in dirs];z=absol(name+'_proj',pr)
    return emit(name,[sum(z[i:i+8])*np.pi/16 for i in range(0,len(z),8)])

def spl(name,xs,knots,ys):
    k=np.asarray(knots);y=np.asarray(ys); slope=np.diff(y)/np.diff(k);coef=np.diff(np.r_[0.,slope,0.]);h=hinges(name+'_hinges',[x-t for x in xs for t in k]);return emit(name,[ys[0]+sum(c*u for c,u in zip(coef,h[i:i+len(k)])) for i in range(0,len(h),len(k))])

# Pixel coordinates in original pixels. Each basis function is a quadrature
# rule defined on the pixel lattice, not a sample drawn from any image.
small=pool('bin2','image',2,2,'avg'); lumw=np.array([.30,.45,.25]);lum=point('intensity',small,lumw)
v=(np.arange(64)+.5)*2-64; Y,X=np.meshgrid(v,v,indexing='ij'); rr=np.hypot(X,Y); ang=np.arctan2(Y,X)
def unit(a):return a/a.sum()
def gauss(cx,cy,s):return unit(np.exp(-((X-cx)**2+(Y-cy)**2)/(2*s*s)))
# Sky in eight broad peripheral sectors; use means of pairwise minima to
# suppress isolated bright neighbours. Additional central apertures for color.
sk=[unit(((rr>47)&(rr<64)&((ang+np.pi)%(2*np.pi)>=j*np.pi/4)&((ang+np.pi)%(2*np.pi)<(j+1)*np.pi/4)).astype(float)) for j in range(8)]
sig=[1.6,4.,9.,18.];phot=sk+[gauss(0,0,s) for s in sig];P=len(phot)
cw=np.zeros((3,P,3,64,64),'f');
for j in range(3):cw[j,:,j]=phot
photo=refs(dense('sky_and_apertures',small,cw)); sky=[];flux=[]
for j in range(3):
    a=photo[P*j:P*(j+1)]; diff=absol('sky_pair_diff'+str(j),[a[i]-a[i+4] for i in range(4)])
    sky.append((sum(a[:8])-sum(diff))/8);flux.append([z-sky[-1] for z in a[8:]])
sky=emit('sky',sky); flux=emit('net_apertures',sum(flux,[]));flux=[flux[i:i+4] for i in range(0,12,4)]
skylum=sum(z*c for z,c in zip(sky,lumw));total=[sum(c*q[i] for c,q in zip(lumw,flux)) for i in range(4)]
# Nonnegative core photometry with a sky-noise floor.
peak=hinges('central_light',[total[0]])[0]; scale=peak+.030
color=ratios('color_index',[flux[0][i]-flux[2][i] for i in range(4)],[total[i]+e for i,e in enumerate((.035,.025,.02,.015))])

A=36;theta=2*np.pi*np.arange(A)/A
radii=1.4*1.23**np.arange(19);R=len(radii);dl=np.log(1.23)
bounds=np.r_[0,np.sqrt(radii[:-1]*radii[1:]),radii[-1]*np.sqrt(1.23)];dr=np.diff(bounds)
# A Gaussian footprint grows slowly with radius, retaining enough transverse
# resolution for thin disks while averaging many sky pixels at large radii.
bank=np.array([gauss(r*math.cos(t),r*math.sin(t),math.hypot(.98,.067*r)) for r in radii for t in theta],np.float32)
sampled=dense('polar_samples',lum,bank);N=A*R
# Exact linear broadcast of sky and central intensity. A sign-symmetric
# rms packs one central reference with many much smaller polar measurements.
# This implements division by approximately the central surface brightness.
# Keep the spatial maps compact: positive and negative copies via a 1x1.
sampled=reshape('ray_grid',sampled,(1,R,A)); sr=refs(dense('sky_luma',next(iter(sky[0].t))[0],lumw))[0]
skyfield=dense('subtract_sky',next(iter(sr.t))[0],-np.ones((N,1),'f'));skyfield=reshape('sky_grid',skyfield,(1,R,A));net=add('polar_net',sampled,skyfield)
net=reshape('sample_list',net,(N,));ids=np.eye(N,dtype='f');p=dense('signed_samples',net,np.vstack([ids,-ids,np.zeros((2,N),'f')]))
sc=emit('peak_reference',[scale]);scsrc=next(iter(sc[0].t))[0];ref=np.zeros((2*N+2,1),'f');ref[-2:,0]=[8*np.sqrt(N),-8*np.sqrt(N)];s=dense('reference_pair',scsrc,ref)
x=add('relative_light_input',p,s);x=reshape('relative_light_channel',x,(1,2*N+2,1));x=normalize('relative_light_norm',x);x=reshape('relative_light_flat',x,(2*N+2,));x=dense('relative_light',x,np.pad(ids*8,((0,0),(0,N+2))));x=reshape('angle_rays',x,(R,A,1));raw=refs(x)
# Prefix minima along each ray measure light connected to the central galaxy.
# The scan is a sequence of parallel pairwise minima.
con=raw
for delta in (1,2,4,8,16):
    d=hinges('radial_drop'+str(delta),[con[j]-con[max(0,j-delta)] for j in range(R)])
    con=emit('connected'+str(delta),[v-h for v,h in zip(con,d)])

# Soft isophotes, from faint disk to bright core. Integrals of masks in r,
# log(r), r^2 are three different measurements of the same contour.
lev=np.array([.066,.125,.225,.39,.61,.79]); L=len(lev); radi=[];logradi=[];areas=[]
for i,t in enumerate(lev):
    m=saturate('iso'+str(i),con,.78*t,1.25*t)
    radi.append(sum(g*q for g,q in zip(dr,m)));areas.append(sum(g*q for g,q in zip(np.diff(bounds**2),m)))
    # Integral d(log r), including a softened innermost bin.
    ldr=np.diff(np.log(np.maximum(bounds,.7)))
    logradi.append(sum(g*q for g,q in zip(ldr,m)))
iso=emit('isophote_integrals',radi+logradi+areas);iso_src=next(iter(iso[0].t))[0]
# Angular Fourier quadrature on each contour. dc, dipole etc. are lengths,
# log lengths or squared lengths, depending on the measure being integrated.
def fourier(name,x,orders):
    C=shapes[x][0]; mat=[np.ones(A)/A]
    for m in orders:mat.extend([np.cos(theta*m)*2/A,np.sin(theta*m)*2/A]);
    w=np.einsum('ij,ka->ikja',np.eye(C),mat).reshape(C*len(mat),-1)
    f=refs(dense(name,x,w));return [f[i:i+len(mat)] for i in range(0,len(f),len(mat))]
ff=fourier('isophote_harmonics',iso_src,[1,2,3,4]);fl=ff[L:2*L];fr=ff[:L];fa=ff[2*L:]
meanR=[u[0] for u in fr];meanArea=[u[0] for u in fa];meanlog=[u[0] for u in fl]
ps=lambda z,i:(z[i],z[i+1]);subp=lambda a,b:(a[0]-b[0],a[1]-b[1]);addp=lambda a,b:(a[0]+b[0],a[1]+b[1]);mul=lambda a,c:(a[0]*c,a[1]*c)
# Area-weighted angular quadrupole has modulus (a-b)/(a+b) for an
# ellipse of any profile. The same result follows from the second harmonic
# of log R. Mix these two measurements to limit the impact of single rays.
area_amp=norm2('area_quadrupole',[ps(z,3) for z in fa]);area_e=ratios('area_ellipticity',area_amp,[2*z+.8 for z in meanArea]);lp=[ps(z,3) for z in fl];log_e=norm2('log_radius_ellipticity',lp)
ell=emit('contour_ellipticity',[.45*a+.55*b for a,b in zip(area_e,log_e)])
# Faint to bright contour axes and centroid displacements. In a translated
# family of concentric ellipses the opposite-ray radius difference is fixed,
# so differencing the dipoles cancels much of the geometric miscentering.
cent=norm2('center_shifts',[subp(ps(fr[i],1),ps(fr[4],1)) for i in (0,1,2)]+[ps(fr[0],1)])
asym=ratios('fractional_center_shifts',cent,[meanR[i]+1 for i in (0,1,2,0)])
odd=norm2('threefold_shape',[ps(fl[i],5) for i in (0,1,2,3)])
s4=norm2('fourfold_shape',[ps(fl[i],7) for i in (0,1,2)])
tw=norm2('isophote_shape_changes',[subp(lp[i],lp[j]) for i,j in ((0,2),(1,3),(1,4),(2,4),(0,1),(2,3),(3,5))]);
# Outward rises and compact secondary peaks. Suppress rises far outside
# the galaxy for texture; keep a separate, less strict companion branch.
ex=hinges('outward_excess',[u-c for u,c in zip(raw,con)])
cap=hinges('unconnected_excess',[e-(.75*c+.018) for e,c in zip(ex,con)])
clean=emit('bounded_arm_light',[c+e-k for c,e,k in zip(con,ex,cap)])
loglight=spl('log_disk_light',clean,[-.06,0,.035,.07,.14,.28,.55,1.,2.],np.log(np.array([.018,.075,.11,.145,.215,.355,.625,1.075,2.075])))
harm=fourier('disk_harmonics',next(iter(loglight[0].t))[0],[1,2,3]);
# Complex log-spiral quadrature, both handednesses. The difference of
# handed responses is zero for an inclined, untwisted elliptical profile.
spirals=[];indices=[]
for m in (2,3):
    for pitch in (16,24,38):
        for idx in (6,9,12,15):
            taper=np.exp(-.5*((np.arange(R)-idx)/2.45)**2);taper/=taper.sum(); om=m/np.tan(np.deg2rad(pitch));ph=(np.arange(R)-idx)*dl*om
            cw=np.cos(ph);sw=np.sin(ph);cw=(cw-taper@cw)*taper;sw=(sw-taper@sw)*taper
            for sign in (-1,1):
                co=0;si=0
                for j,z in enumerate(harm):
                    a,b=ps(z,2*m-1);co+=a*cw[j]-b*sw[j]*sign;si+=b*cw[j]+a*sw[j]*sign
                spirals.append((co,si))
            indices.append((m,pitch,idx))
sa=norm2('spiral_filter_modulus',spirals);sd=absol('spiral_handed_contrast',[a-b for a,b in zip(sa[::2],sa[1::2])]);
pitch2=maxima('best_pitch',[sd[k:k+4] for k in range(0,len(sd),4)]);tight,opened=maxima('pitch_families',[[pitch2[0],.85*pitch2[1],.8*pitch2[3]],[pitch2[2],.9*pitch2[1],.8*pitch2[5],.75*pitch2[4]]]);winding=maxima('winding',[[tight,opened]])[0]
# A second peak requires both an outward brightness increase and a valley
# towards the central object. A connection of diffuse light makes the peak
# more plausible as a physical companion, especially at large separation.
# Compare responses over neighboring angles as well as in individual rays.
comp=[];texture=[]
for j in range(4,R-1):
    dist=radii[j]; valley=con[j]; rising=ex[j];
    # subtract a distance-dependent penalty for chance projections
    comp.append(rising-0.004*max(0,dist-18));texture.append(ex[j]-cap[j])
cz=emit('candidate_peaks',comp+texture);czs=next(iter(cz[0].t))[0]
cp=refs(reshape('strongest_peak_per_ring',pool('azimuthal_peak',czs,A,1,'max'),(len(cz),)))
ca=refs(reshape('mean_peak_per_ring',pool('azimuthal_mean',czs,A,1,'avg'),(len(cz),)))
J=len(comp); companion,tex,texavg=maxima('peak_over_radii',[[cp[i]*.65+ca[i]*2.0 for i in range(J)],[cp[J+i] for i in range(J)],[ca[J+i] for i in range(J)]])
# Angular width of the feature matters: broad excess on a few adjacent rays
# is much less likely to be a chance foreground point source.
sm=np.array([(np.roll(np.eye(A),-1,axis=1)+np.eye(A)*2+np.roll(np.eye(A),1,axis=1))/4])[0]
w=np.einsum('ij,ab->iajb',np.eye(J),sm);smpeaks=dense('angularly_smoothed_peaks',czs,np.pad(w,((0,0),(0,0),(0,J),(0,0))).reshape(J*A,-1));smpeaks=reshape('smoothed_peak_grid',smpeaks,(J,A,1));sp=refs(reshape('broad_peaks',pool('peak_window',smpeaks,A,1,'max'),(J,)))
widecomp=maxima('broad_companion',[sp])[0]

# Edge-on dust lanes and off-plane central bulges. Oriented apertures are
# compared to a set of parallel strips. Only a direction with a genuine
# bright, elongated ridge can contribute much evidence for a dust lane.
orient=np.arange(12)*np.pi/12; stripe=[]; Nr=3
for j in range(Nr):
    h=(1.7,2.65,4.1)[j]; a=(12.,20.,32.)[j]
    for t in orient:
        u=X*np.cos(t)+Y*np.sin(t);v=-X*np.sin(t)+Y*np.cos(t)
        axial=np.exp(-.5*(u/a)**2)*(1-np.exp(-.5*(u/(a*.29))**2))
        axial*=1/(1+(u/(a*1.4))**8)
        def pair(offset,sig):
            return unit(axial*(np.exp(-.5*((v-offset)/sig)**2)+np.exp(-.5*((v+offset)/sig)**2)))
        axis=pair(0,max(1.,.45*h));either=pair(h,max(1.,.45*h));outerstrip=pair(2.9*h,h*.8)
        core=unit(np.exp(-.5*(u/(a*.24))**2)*(np.exp(-.5*((v-h)/max(1.,.55*h))**2)+np.exp(-.5*((v+h)/max(1.,.55*h))**2)))
        stripe.append([axis,either,outerstrip,core])
stripe=np.asarray(stripe);sr=dense('disk_cross_sections',lum,stripe);z=refs(sr);strc=point('opponent_color',small,[1.,0.,-1.]);co=refs(dense('lane_color_difference',strc,stripe[:,0]-stripe[:,1]));
# Normalize all of these flux measurements by the central light. As in the
# polar sampling stage, the sign-paired reference is deliberately dominant.
rawstrip=[]
for j in range(len(stripe)):
    axis,shoulder,out,core=z[4*j:4*j+4]
    rawstrip.extend([axis-out,shoulder-axis,core-shoulder,co[j]])
striprat=ratios('relative_disk_photometry',rawstrip,[scale]*len(rawstrip));striprat=[striprat[j:j+4] for j in range(0,len(rawstrip),4)]
ridges=maxima('disk_direction',[[r[0] for r in striprat[k:k+len(orient)]] for k in range(0,len(stripe),len(orient))]);
lanes=[];centralw=[]
for j,(ridge,dip,bul,chrom) in enumerate(striprat):
    # Wrong directions have weak brightness contrast across the plane.
    p=ridges[j//len(orient)]-ridge
    lanes.append(dip+.20*chrom-.82*p);centralw.append(bul-1.25*p)
lane,wing=maxima('aligned_lane_bulge',[lanes,centralw]);

# Blue excess in the outer disk is stronger evidence for star formation
# than a global blue color. Measure angular variations in red minus blue at
# fixed fractions of the local intensity. The first-order color estimate
# uses signed pairs; the flux floor attenuates the noisy sky.
# Use a coarser grid for the chromatic measurements. Their rms on annuli
# is approximated by a mean absolute deviation about the azimuthal mean.
cs=slice(4,R-1,2); caa=np.arange(A)[::2]; nc=len(caa);cind=np.arange(N).reshape(R,A)[cs][:,caa].ravel()
cb=bank[cind];cc=refs(dense('disk_color_samples',strc,cb));cg=refs(dense('disk_flux_samples',lum,cb));
colnet=hinges('positive_color_flux',[a-skylum for a in cg]);colrat=ratios('local_colors',[a-(sky[0]-sky[2]) for a in cc],[a+.040 for a in colnet]);
cl=len(colrat)//nc;cm=[sum(colrat[i:i+nc])/nc for i in range(0,len(colrat),nc)];
# Concentration of blue knots, particularly in a warm disk. An outer annulus
# by itself could be sky: gate each ring by its signal above the background.
local=absol('angular_color_variations',[c-cm[j//nc] for j,c in enumerate(colrat)]);
cd=[sum(local[i:i+nc])/nc for i in range(0,len(colrat),nc)]; mf=[sum(cg[i:i+nc])/nc-skylum for i in range(0,len(cg),nc)];
# A bounded significance penalty eliminates rings indistinguishable from sky.
lightgate=saturate('visible_ring',mf,.023,.10)
colscatter,bluegradient=maxima('color_knots',[[a-.47*(1-g) for a,g in zip(cd,lightgate)],[(color[1]-a)-.70*(1-g) for a,g in zip(cm,lightgate)]])

# Final fuzzy rules are in physical units (ellipticities, log radius,
# fractional contour deviations, fractional flux, and log contrast).
# Continuous switches here are bounded linear ramps.
class Switches:
    def __init__(self):self.spec=[]
    def put(self,key,x,lo,hi):self.spec.append((key,x,lo,hi))
    def compile(self,name):
        ds=[(x-lo)/(hi-lo) for k,x,lo,hi in self.spec];h=hinges(name,ds+[x-1 for x in ds]);return {k:h[i]-h[i+len(ds)] for i,(k,*_) in enumerate(self.spec)}
S=Switches();put=S.put
flat=.22*ell[0]+.37*ell[1]+.30*ell[2]+.11*ell[3]; inner=.5*ell[3]+.5*ell[4];outer=.45*ell[0]+.55*ell[1]
# An estimate of log semimajor axis. For an ideal ellipse mean(log R)
# is log(2ab/(a+b)); add -log(1-e) to obtain log(a).
logmajor=meanlog[1]+math.log(.7)+spl('major_axis_correction',[flat],[0,.2,.4,.55,.7,.82],[0,.223,.511,.799,1.204,1.715])[0]
logminor=meanlog[1]+math.log(.7)-.78*flat
put('size',logmajor,np.log(7),np.log(23));put('large',logmajor,np.log(13),np.log(34));put('tiny',-logmajor,-np.log(9),-np.log(4))
put('round',-flat,-.166,-.065);put('mid',flat,.077,.18);put('elong',flat,.29,.49)
put('thin',flat,.375,.595);put('verythin',flat,.515,.68);put('diskaxis',outer,.38,.59)
put('red',.55*color[0]+.45*color[1],.02,.42);put('blue',-.48*color[1]-.52*color[2],-.12,.20);put('redcore',color[0]-color[2],.018,.29)
put('diffuse',-peak,-.38,-.105);put('bright',peak,.21,.61)
put('bulgeshape',outer-inner,.025,.205);put('bulgeround',-inner,-.45,-.16)
put('concen',meanlog[1]-meanlog[4],.91,1.62);put('flatprof',meanlog[4]-meanlog[1],-1.19,-.65)
put('asym',.47*asym[0]+.38*asym[1]+.15*asym[2],.047,.235)
put('bigasym',.52*asym[0]+.48*asym[1],.18,.48);put('lopsided',asym[3],.15,.47)
put('odd',.38*odd[0]+.37*odd[1]+.25*odd[2],.026,.132)
put('oddcore',.55*odd[2]+.45*odd[3],.03,.145)
put('boxy',.4*s4[0]+.35*s4[1]+.25*s4[2]-.24*flat,.018,.12)
put('twist',.5*tw[0]+.5*tw[1],.047,.202);put('wind',winding,.031,.151)
put('tight',tight,.02,.134);put('open',opened,.027,.167)
put('texture',texavg,.0035,.035);put('arcs',tex,.043,.198)
put('comp',.45*companion+.55*widecomp,.16,.57);put('strongcomp',widecomp,.41,.90)
put('bar',.46*ell[3]+.36*ell[2]+.18*ell[4]-outer,.045,.225)
put('baraxis',.35*ell[4]+.40*ell[3]+.25*ell[2],.125,.345)
# Inner elongated isophotes followed by a change of PA or a rounder disk.
# If there is just a round bulge in an inclined disk, discount the change.
put('barturn',.56*tw[2]+.44*tw[1]-.82*(outer-inner),.065,.28)
put('quietcore',-tw[6],-.19,-.045)
put('thinphys',-logminor,-np.log(6.5),-np.log(3))
put('lane',lane,.003,.065);put('wing',wing,.20,.49)
put('patchcolor',colscatter,.043,.196);put('starforming',bluegradient,.092,.43)
F=S.compile('fuzzy_membership');v=F.__getitem__
# Evidence for visible arms. Use multiple cues; mere angular flattening is
# not evidence for arms, and sky noise matters more on small objects.
substructure=(.30*v('wind')+.18*v('odd')+.20*v('twist')+.17*v('texture')+.08*v('boxy')+.07*v('arcs'))
spEvidence=(.55*substructure+.225*v('size')+.085*v('large')+.055*v('blue')+.035*v('redcore')+.065*v('starforming')+.056*v('patchcolor')-.13*v('thin')-.065*v('tiny'))
barEvidence=(.42*v('bar')+.26*v('barturn')+.18*v('baraxis')+.08*v('quietcore')+.06*v('redcore'))
# Distinguish a stellar companion from one with significant associated
# diffuse asymmetric light. A clean two-core system is not required if the
# shape is strongly distorted and there is still a substantial companion.
mergeEvidence=(.48*v('comp')+.25*v('strongcomp')+.12*v('bigasym')+.09*v('lopsided')+.06*v('odd'))
irregular=(.31*v('bigasym')+.19*v('asym')+.17*v('oddcore')+.11*v('odd')+.13*v('diffuse')+.09*v('blue'))
# Ordered classes of apparent ellipse shape use intersecting logits, not a
# bank of images. Branch scores share the same gate within each family.
smooth=1.25-2.55*spEvidence-.78*v('thin')-.62*mergeEvidence-.36*irregular
spiral=2.55*spEvidence-.37-.82*v('verythin')-.32*mergeEvidence
edge=1.60*v('thin')+.40*v('verythin')+.27*v('diskaxis')+.20*v('size')+.16*v('lane')-.72-.34*mergeEvidence
bulge=.32*v('bulgeshape')+.15*v('bulgeround')+.14*v('concen')+.11*v('redcore')+.09*v('bright')+.18*v('wing')-.14*v('verythin')
# Open arms tend to have less central concentration and show more clumpy
# star formation; winding at the smaller pitch angles supports tight arms.
openarm=.37*(v('open')-v('tight'))+.21*v('odd')+.16*v('blue')+.15*v('flatprof')+.10*v('asym')+.12*v('patchcolor')-.12*v('red')-.20*flat
scores=[
    1.68*irregular+.28*v('tiny')+.13*v('asym')-.38-.51*mergeEvidence-.33*v('verythin'),
    1.86*mergeEvidence+.29*v('asym')+.13*v('size')-.26,
    smooth+.34*v('round')-.24*v('mid')-.53*v('elong'),
    smooth+.18*v('mid')-.27*v('round')-.45*v('elong'),
    smooth-.48+.91*v('elong')-.22*v('mid')-.13*v('large')-.18*v('verythin'),
    spiral+.88*barEvidence-.373,
    spiral+.14-.39*openarm-.26*barEvidence,
    spiral-.10+.50*openarm-.22*barEvidence,
    edge+.25-.52*bulge,
    edge-.18+.55*bulge,
]
emit('galaxy_scores',scores)
(outdir/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n');np.savez(outdir/'weights.npz',**weights)
