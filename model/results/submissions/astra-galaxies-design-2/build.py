"""A fixed isophotal photometer for Galaxy Zoo morphologies. No input files."""
import json, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
layers=[]; weights={}; shapes={'image':(3,128,128)}
def put(n,op,ins,sh,**kw):
    assert n not in shapes,n
    layers.append(dict(name=n,op=op,inputs=list(ins),**kw));shapes[n]=tuple(sh);return n
def dense(n,x,W,b=0):
    W=np.asarray(W,dtype='f').reshape(-1,int(np.prod(shapes[x])));o=len(W)
    weights[n+'.W']=W; weights[n+'.b']=np.broadcast_to(b,(o,)).astype('f').copy()
    return put(n,'dense',[x],(o,),out=o)
def point(n,x,W,b=0):
    c,h,w=shapes[x];W=np.array(W,dtype='f').reshape(-1,c); o=len(W)
    weights[n+'.W']=W[:,:,None,None];weights[n+'.b']=np.broadcast_to(b,(o,)).astype('f').copy()
    return put(n,'conv',[x],(o,h,w),out=o,kernel=[1,1],stride=1,padding=0)
def view(n,x,*sh):return put(n,'reshape',[x],sh,shape=list(sh))
def relu(n,x):return put(n,'relu',[x],shapes[x])
def add(n,*args):return put(n,'add',args,shapes[args[0]])
def norm(n,x):return put(n,'norm',[x],shapes[x])
def pool(n,x,h,w,op='maxpool'):
    c,H,W=shapes[x];return put(n,op,[x],(c,H//h,W//w),size=[h,w],stride=[h,w])

def unit(x):return x/np.sum(x)
def spatial_kernel(n,x,W,pad):
    W=np.array(W,dtype='f');o,c,k,_=W.shape;weights[n+'.W']=W;weights[n+'.b']=np.zeros(o,'f');return put(n,'conv',[x],(o,*shapes[x][1:]),out=o,kernel=[k,k],padding=pad,stride=1)
low=pool('bin_2x2','image',2,2,'avgpool')
lum=point('light',low,[[.32,.45,.23]])
Y,X=np.mgrid[:64,:64]*2-63.; rad=np.hypot(X,Y); theta=np.arctan2(Y,X)
# Antipodal residuals on a centered crop of the image separate an early
# type galaxy with a similar-size field star from a tidally disrupted disk.
# A constant small miscentering is not a tidal feature: minimize over a
# few plausible centering offsets in the Fourier/odd residual below.
# For the color field, do not use disk outskirts outside the galaxy:
# instead take the measurements along the same set of rays as the light.
A=48; angles=np.arange(A)*2*np.pi/A
r=np.array([0,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,10,11.5,13,14.5,16,18,20,22,24,26.5,29,32,35,38,41.5,45,49,53,57,61]); R=len(r)
# Band-specific sky, measured in angular tiles far from the target. The
# least contaminated tile is a good estimator in a field with bright stars.
tiles=[]
for a in np.arange(8)*np.pi/4:
    tiles.append(unit((rad>53)*(np.cos(theta-a)>np.cos(np.pi/8))))
def bands(kernels):
    return np.einsum('bc,shw->bschw',np.eye(3),kernels).reshape(3*len(kernels),-1)
sk=dense('sky_sectors',low,-bands(tiles));sk=pool('dark_sky',view('sky_shape',sk,3,1,8),1,8)
# Smooth radial probes. Fixed angular sampling, with a wider beam in the
# far outskirts so that individual bright pixels do not define a contour.
probes=[]
for d in r:
    for t in angles:
        dx=X-d*np.cos(t);dy=Y-d*np.sin(t);u=dx*np.cos(t)+dy*np.sin(t);v=dy*np.cos(t)-dx*np.sin(t)
        probes.append(unit(np.exp(-.5*(u*u/(1.25**2+(.04*d)**2)+v*v/(1.25**2+(.052*d)**2)))).ravel())
p0=dense('polar_apertures',lum,probes);del probes
sky_lum=dense('subtract_sky',sk,np.tile([.32,.45,.23],(R*A,1)))
p=view('polar_grid',add('sky_corrected',p0,sky_lum),R,A,1)
inside=point('inside_seeing_disc',p,np.eye(R)[:4]);pk=pool('central_peak',view('core_samples',inside,1,4,A),4,A)
# Normalization trick: include a pair of opposite, large reference values
# in a norm window. Subtraction of a zero-reference row cancels the mean.
# This computes flux / central_flux without a learned gain.
extra=3;gain=80.;E=np.eye(R+extra)[:,:R]
p_pad=point('flux_rows',p,E)
refpattern=np.tile(np.r_[np.zeros(R+1),gain,-gain][:,None],(1,A)).ravel()
pr=dense('reference_rows',pk,refpattern[:,None],refpattern*.006)
pp=add('photometric_window',p_pad,view('reference_grid',pr,R+3,A,1))
pp=norm('photometric_scale',view('single_aperture',pp,1,R+3,A));pp=view('rows_again',pp,R+3,A,1)
mat=E.T.copy();mat[:,R]=-1
f=point('relative_brightness',pp,mat*gain*np.sqrt(2/(R+3)))
# Running valley along each ray; crossing a dark gap is evidence that a
# distant peak does not belong to a central isophote.
base=f
for s in (1,2,4,8,16):
    I=np.eye(R);prev=I[np.maximum(np.arange(R)-s,0)]
    over=relu('above_valley_'+str(s),point('compare_inward_'+str(s),base,I-prev))
    base=add('inner_min_'+str(s),base,point('lower_'+str(s),over,-I))
delta=add('rise_after_valley',f,point('negative_valley',base,-np.eye(R)))
# A partial recovery is allowed (to bridge dust, arm gaps and noise), but
# not a detached bright object beyond a stretch of empty sky.
leak=point('connected_cap',base,-1.65*np.eye(R),-.018)
clipped=relu('detached_light',add('compare_cap',f,leak));connect=add('connected_light',f,point('subtract_detached',clipped,-np.eye(R)))
lev=np.array([.09,.15,.245,.365,.50,.655,.79]);L=len(lev)
# Integrate differentiable isophote indicators. Radii in input pixels.
sample=view('radial_samples',connect,1,R,A);dlog=.28
wid=lev*dlog+.012
cuts=np.r_[lev-wid/2,lev+wid/2]
h=relu('threshold_ramps',point('fractional_levels',sample,np.ones((2*L,1)),-cuts))
mask=point('isophotes',h,np.c_[np.diag(1/wid),-np.diag(1/wid)])
dr=np.r_[(r[1]-r[0])/2,(r[2:]-r[:-2])/2,(r[-1]-r[-2])/2]
M=np.kron(np.eye(L),dr[None,:]);length=point('contour_radius',view('level_by_radius',mask,L*R,A,1),M,.30)
# Smoothed log diameter removes a simple galaxy-centering error before
# ellipticity is measured. Opposite rays remain separate for lopsidedness.
M0=np.eye(A);smooth=(np.roll(M0,1,axis=1)+4*M0+np.roll(M0,-1,axis=1))/6
D=(smooth+np.roll(smooth,A//2,axis=1))/2
ang_transform=lambda n,x,B: view(n,dense(n+'_sum',x,np.kron(np.eye(L),B)),L,B.shape[0],1)
diameter=ang_transform('semi_diameter',length,D)

def elem_lut(n,x,knots,values):
    sh=shapes[x]; xx=view(n+'_flat',x,1,sh[0],int(np.prod(sh[1:])))
    m=np.diff(values)/np.diff(knots);h=relu(n+'_hinges',point(n+'_knots',xx,np.ones((len(m),1)),-knots[:-1]))
    return view(n,point(n+'_linear',h,np.diff(np.r_[0,m])[None,:],values[0]),*sh)
kn=np.geomspace(.3,72,20);ld=elem_lut('log_diameter',diameter,kn,np.log(kn));lr=elem_lut('log_radius',length,kn,np.log(kn))
# Faint extensions: absolute isophotes measured in a smoothed, sky-
# corrected image complement the fractional ones, which miss tidal tails
# around bright cores. Integration is capped at the first sufficiently
# dark gap, so separated objects contribute little.
val0=p
for ix in (1,2,4,8,16):
    I=np.eye(R);pr=I[np.maximum(np.arange(R)-ix,0)];dif=relu('above_sky_valley'+str(ix),point('abs_radial_diff'+str(ix),val0,I-pr));val0=add('sky_valley'+str(ix),val0,point('valley_subtract'+str(ix),dif,-I))
absolute=add('absolute_brightness',p,point('subtract_background_return',relu('beyond_dark_gap',add('dark_boundary',p,point('dark_gap',val0,-1.35*np.eye(R),-.007))),-np.eye(R)))
thresholds=np.array([.027,.05,.085]);width=np.array([.014,.019,.026]);absmap=view('absolute_grid',absolute,1,R,A)
absh=relu('faint_level_ramps',point('faint_levels',absmap,np.ones((6,1)),-np.r_[thresholds-width/2,thresholds+width/2]));absh=point('faint_isophotes',absh,np.c_[np.diag(1/width),-np.diag(1/width)])
absh=view('faint_level_radii',absh,3*R,A,1);abslength=point('faint_contour_radius',absh,np.kron(np.eye(3),dr[None,:]),.4);abslog=elem_lut('log_faint_contour',abslength,kn,np.log(kn))
Q=np.stack([np.ones(A)]+[2*fn(m*angles) for m in (1,2,3,4,5) for fn in (np.cos,np.sin)])/A
faintshape=dense('tidal_harmonics',abslog,np.kron(np.eye(3),Q))
# Signed polar residual. A smooth, star-like, azimuthally compact object
# should not dominate the integral, so measure connected-light residuals.
# Divide out brightness in each symmetric pair when testing asymmetry,
# but retain a separate flux-weighted measurement.
flip=np.roll(np.eye(A),A//2,axis=1);sw=view('for_antipodes',connect,1,R,A)
# Compute the transposition using a small shared matrix over the angles.
swapmat=dense('flip_angles',view('rays_as_vector',connect,1,R*A,1),np.kron(np.eye(R),np.eye(A)-flip));sw=view('odd_light',swapmat,1,R,A)
rp=relu('positive_odd_light',sw);rp=view('odd_radial',rp,R,A,1)
absstats=[]
for j,(lo,hi) in enumerate([(2,8),(6,16),(12,30),(22,46)]):
    u=unit(dr*r*((r>=lo)&(r<=hi)));t=point('odd_annulus'+str(j),rp,u[None]);absstats.append(pool('asymmetric_flux'+str(j),t,A,1,'avgpool'))
# Do adjacent radii have the same side of the galaxy in excess?
# The 1st Fourier mode will catch a broad one-sided tidal envelope even
# when all the direct light of the companion is excluded by the gap test.
# Local ridge residuals are better computed on the observed surface than
# on an interpolated isophote. A non-nuclear two-dimensional deposit of
# light in an *illuminated* portion of the field is evidence for clumps.
# Compute a bank of scale-normalized directional second derivatives.
cart=pool('bin_4x4',lum,2,2,'avgpool');YY,XX=np.mgrid[:32,:32]*4-62.;rr0=np.hypot(XX,YY)
smalls=[]
for sig,kk in [(1.05,3),(1.85,5)]:
    m=2*kk+1;yy,xx=np.mgrid[-kk:kk+1,-kk:kk+1];g=unit(np.exp(-.5*(xx**2+yy**2)/sig**2));B=[]
    for t in np.arange(8)*np.pi/8:
        z=((xx*np.cos(t)+yy*np.sin(t))/sig)**2-1;B.append(g*(z-np.sum(z*g)))
    B.append(g*.82) # brightness, including sky
    nm='hessian'+str(kk);z=spatial_kernel(nm,cart,np.array(B)[:,None],kk)
    # Both eigenvalues negative --> knot; one negative and one close to
    # zero --> ridge. The trace alone would pick up an ordinary profile.
    z0=point(nm+'_positive',z,np.eye(9)[:8]);z1=point(nm+'_negative',z,-np.eye(9)[:8]);
    # Transpose channels into pooling width without large dense maps.
    def chanmax(n,x,c):
        return view(n,pool(n+'_pool',view(n+'_reshape',x,1,c,32*32),c,1),1,32,32)
    big=chanmax(nm+'_max',z0,8); negsmall=chanmax(nm+'_minus_min',z1,8);lum0=point(nm+'_baseline',z,[[0]*8+[1]])
    # Local relative curvature: a noisy sky is excluded by thresholds and
    # signed smooth-background cancellation. Positive large eigenvalue
    # along with negative small eigenvalue indicates a saddle or bent arm.
    negbig=point(nm+'_negmax',big,[[-1]]);peakness=relu(nm+'_blobs',negbig)
    # Saddles (opposite principal curvatures), weighed toward a narrow
    # ridge and against the broad tangent of normal elliptical isophotes.
    saddle=relu(nm+'_saddles',add(nm+'_ridge_test',negsmall,point(nm+'_flank',big,[[-.55]]),point(nm+'_contrast_floor',lum0,[[-.10]],.004)))
    # Off-center excess over a local radial model. Compute curvature
    # perpendicular to the radius at each location. In a circular smooth
    # profile this is nonpositive. For spiral gaps/knots the tangential
    # curvature changes sign, although projection makes it imperfect.
    index=np.mod(np.rint((np.arctan2(YY,XX)+np.pi/2)*8/np.pi).astype(int),8)
    # Build only scalar statistics; after two soft thresholds, integrate
    # over several smoothly terminated annuli.
    tang=relu(nm+'_tangential_gaps',point(nm+'_gap_threshold',z,np.c_[np.eye(8),np.full(8,-.045)],-.001))
    for jj,(lo,hi) in enumerate([(4,13),(9,23),(18,40)]):
        aper=unit(np.exp(-.5*((rr0-(lo+hi)/2)/((hi-lo)/3))**2)*(rr0>lo*.7));
        for role,tmap in enumerate((peakness,saddle,tang)):
            W=(aper if role!=2 else aper[None]*np.eye(8)[index].transpose(2,0,1));smalls.append(dense(nm+'_patch_%d_%d'%(jj,role),tmap,W.ravel()[None,:]))
# Model-space arm filters. Elliptical polar quadrature deprojects a disk
# viewed at different inclinations. In these coordinates an oval with fixed
# position angle contributes a constant-phase m=2 mode, but a pair of arms
# winds that mode around the origin as log radius increases.
G=12; ph=np.arange(32)*2*np.pi/32; ring_r=np.geomspace(2.6,46,G); sheets=[]
for q,na in [(1,1),(.73,6),(.48,8),(.30,10)]:
    for a0 in np.arange(na)*np.pi/na:
        ring=[]
        for rr in ring_r:
            xx=rr*(np.cos(a0)*np.cos(ph)-q*np.sin(a0)*np.sin(ph)); yy=rr*(np.sin(a0)*np.cos(ph)+q*np.cos(a0)*np.sin(ph));bw=1.15+.036*rr
            beam=np.exp(-.5*((X[None]-xx[:,None,None])**2+(Y[None]-yy[:,None,None])**2)/bw**2);beam/=beam.sum((1,2),keepdims=True)
            ring.extend(np.einsum('mc,chw->mhw',np.stack([2*np.cos(2*ph),2*np.sin(2*ph)])/len(ph),beam))
        sheets.append(ring)
H=len(sheets); fm=dense('deprojected_ring_harmonics',lum,np.asarray(sheets).reshape(-1,64,64));del sheets
fm=view('disk_geometry',fm,H,G*2,1)
# Sum phase-matched adjacent bands, rejecting an untwisted radial mode.
# Opposite handedness is intrinsically paired; no learning of an angle.
slopes=[1/np.tan(np.deg2rad(a)) for a in (15,30,48)]; tw=[]; align=[]
for om in slopes:
    for start in range(G-4):
        logr=np.log(ring_r[start:start+5]); ww=unit(np.array([.6,1,1,1,.6]));ph0=2*om*(logr-logr.mean());basis=ww*(np.exp(1j*ph0)-np.sum(ww*np.exp(1j*ph0)))
        for hand in (1,-1):
            v=basis if hand==1 else basis.conj();
            for ph1 in np.arange(8)*np.pi/4:
                mat=np.zeros((G,2));h=v*np.exp(1j*ph1);mat[start:start+5]=np.stack([h.real,h.imag],1);tw.append(mat.ravel())
for start in range(G-4):
    # If the light distribution has been deprojected plausibly the
    # endpoints should not show a large systematic elongation.
    for ph1 in np.arange(8)*np.pi/4:
        mat=np.zeros((G,2));h=np.exp(1j*ph1);mat[start:start+5]=np.tile([h.real,h.imag],(5,1))/5;align.append(mat.ravel())
# Use shared 1x1 maps with candidate disk orientation as spatial axis.
fm=dense('transpose_ring_modes',fm,np.eye(H*G*2).reshape(H,G*2,-1).transpose(1,0,2).reshape(H*G*2,-1));fm=view('ring_mode_list',fm,G*2,H,1)
resp=point('spiral_correlators',fm,tw);Nstart=G-4;T=len(slopes)
unfold=pool('spiral_phase_invariant',view('spiral_phase_rows',resp,T*Nstart*2,8,H),8,1)
# A single straight bar or isolated off-center knot has identical scores
# for opposite handedness. A winding pair of arms breaks this reflection.
wa=np.eye(T*Nstart*2)[::2]-np.eye(T*Nstart*2)[1::2]
wa=point('handedness_difference',unfold,np.concatenate((wa,-wa)))
wa=pool('preferred_handedness',view('wind_sign',wa,2,T*Nstart,H),1,1);wa=view('handedness_vector',wa,2*T*Nstart,H,1)
we=point('handedness_evidence',wa,np.c_[np.eye(T*Nstart),np.zeros((T*Nstart,T*Nstart))]);wv=relu('positive_chirality',wa);chir=point('absolute_chirality',wv,np.tile(np.eye(T*Nstart),(1,2)))
chir=pool('pitch_asymmetry',view('chiral_bands',chir,T,Nstart,H),Nstart,H)
resp=pool('ignore_handedness',view('two_senses',unfold,T*Nstart,2,H),2,1)
ell=point('unwound_mode',fm,align);ell=pool('ellipse_phase_invariant',view('ellipse_phase',ell,Nstart,8,H),8,1)
resp=add('arm_model_advantage',resp,point('unwound_penalty',ell,np.tile(-.36*np.eye(Nstart),(T,1))))
armtest=pool('best_arm_pattern',view('pitch_bands',resp,T,Nstart,H),Nstart,H)
# Local-maximum prominence: measure a secondary peak against the highest
# of four surrounding half-annuli, so a monotonic shoulder of the central
# galaxy has weak response even if its Laplacian is high.
Yc,Xc=np.mgrid[-10:11,-10:11]*4.;xc=Xc.ravel();yc=Yc.ravel();rho=np.hypot(xc,yc);P=len(xc);peakmaps=[]
for scale in (2.6,5.0):
    probes=[]
    for xx,yy in zip(xc,yc):
        u=X-xx;v=Y-yy;d=np.hypot(u,v); gr=unit(np.exp(-.5*(d/scale)**2)); ring=np.exp(-.5*((d-2.5*scale)/(.66*scale))**2)
        for direction in np.arange(4)*np.pi/2:
            half=unit(ring*(.02+np.maximum(0,(u*np.cos(direction)+v*np.sin(direction))/(d+.01)))); probes.append((gr-half).ravel())
    resp0=dense('local_contrast_'+str(int(scale)),lum,probes)
    # min over directions = -max of negatives
    weights[resp0+'.W']*=-1
    x=pool('strongest_neighbor_'+str(int(scale)),view('nucleus_orientations_'+str(int(scale)),resp0,P,1,4),1,4);peakmaps.append(x)
# Require a saddle between two nuclei. A second maximum is a nucleus if
# it is bright on *two* smoothing scales; a very compact star usually is
# not. The weaker of a pair is captured by a fixed spatial exclusion mask.
peakmap=add('multiscale_nucleus',*peakmaps);mats=[]
# Companion-specific extension test: compare nested *background-subtracted*
# apertures. A star is much fainter after broadening; a galaxy retains an
# extended base. Correct the lowest directional contrast for this effect.
kern=[]
for xx,yy in zip(xc,yc):
    d2=(X-xx)**2+(Y-yy)**2;ap=lambda width:unit(np.exp(-.5*d2/width**2));narrow=ap(1.4);mid=ap(4.3);wide=ap(9.5)
    kern.extend([narrow-mid,mid-wide,wide-ap(17.)]);
ext=dense('nuclear_scales',lum,kern);ext=view('nuclear_scales_grid',ext,1,P,3)
# Express angular width as inequalities between the three radial samples.
Ext=dense('transpose_apertures',ext,np.eye(P*3).reshape(P,3,-1).transpose(1,0,2).reshape(P*3,-1));ext=view('nuclear_radius_grid',Ext,3,1,P)
star=relu('point_source_evidence',point('point_spread_test',ext,[[.40,-1,-.28]],-.002));
# Broad nuclei can have little local prominence in a binary galaxy, e.g.
# if the neighboring galaxy adds a gradient. Use the residual above the
# local envelope as well as curvature, and suppress overly compact peaks.
res0=point('extended_light_score',ext,[[.13,.45,.13]])
nc=view('nuclear_brightness_map',peakmap,1,1,P);nuccontrast=point('nucleus_contrast',nc,[[-.42]])
# The local envelope has an additional veto, not another additive vote:
# a point on a monotonic shoulder does not count as a nucleus, even if
# the shoulder is bright; a stellar spike needs a broad resolved skirt.
curveneg=relu('invalid_local_peak',point('peak_prominence_floor',nuccontrast,[[-1]],.004))
# Compare the lower envelope of several samples spaced around the
# candidate center with the local background at larger radius. This
# samples its skirt without the star's bright central pixels. Use the
# *second* lowest of four half-annuli to tolerate a flattened companion.
skirts=[]
for xx,yy in zip(xc,yc):
    dx=X-xx;dy=Y-yy;dd=np.hypot(dx,dy)
    for direction in np.arange(4)*np.pi/2:
        w=.008+np.maximum((dx*np.cos(direction)+dy*np.sin(direction))/(dd+.02),0)**2
        near=unit(w*np.exp(-.5*((dd-6.3)/1.8)**2));far0=unit(w*np.exp(-.5*((dd-13.)/3.2)**2));skirts.append((near-far0).ravel())
ss=dense('resolved_quadrants',lum,skirts);ss=view('skirt_vector',ss,1,P,4);ss=dense('quadrant_major',ss,np.eye(P*4).reshape(P,4,-1).transpose(1,0,2).reshape(P*4,-1));ss=view('outer_profile',ss,4,1,P)
# Opposite sides receiving positive flux suggests an extended component;
# a gradient from the primary alone cannot raise both sides of a pair.
I=np.eye(4);op=np.roll(I,2,axis=1)[:2];ee=point('skirt_test',ss,I[:2]-op);ee=relu('skirt_asymmetry',ee);ee=add('bilateral_halo',point('first_skirt',ss,I[:2]),point('subtract_onesided_skirt',ee,-np.eye(2)))
aa=point('select_skirt',ee,[[1,-1]]);ee=add('resolved_envelope',point('second_skirt',ee,[[0,1]]),relu('best_axis',aa))
diff=add('unresolved_test',point('peak_for_halo',nuccontrast,[[.22]]),point('halo_brightness',ee,[[-1]],-.003));penalty=relu('missing_skirt',diff)
nc=add('extended_crest',nuccontrast,res0,point('star_suppression',star,[[-2.8]]),point('slope_suppression',curveneg,[[-1.8]]),point('halo_suppression',penalty,[[-3.5]]))
# For a nearby sector peak, insist on a saddle on going back toward the
# presumed central object. The depression need not reach all the way to
# sky. This is a complementary test when the center falls on a nucleus.
m=[];dm=[]
for frac in (.35,.55,.72):
    for xx,yy in zip(xc,yc):
        fun=lambda f,s:unit(np.exp(-.5*((X-f*xx)**2+(Y-f*yy)**2)/s**2));m.append((fun(1,2.6)-fun(frac,2.6)).ravel());dm.append((fun(frac,4.0)-fun(1.45,6.5)).ravel())
val=dense('saddle_depth',lum,m);val=view('saddle_positions',val,3,1,P)
val=pool('best_saddle',view('saddle_matrix',val,1,3,P),3,1)
# Negative depth on all rays means this is merely a monotonic wing.
secmap=add('possible_secondary',nc,point('saddle_contribution',val,[[.44]]),point('no_saddle',relu('shoulder_not_peak',point('slope_test',val,[[-1]])),[[-1.6]]))
newpair=[]
for i,(lo,hi) in enumerate([(7.,25.),(16.,38.),(28.,52.)]):
    m=((rho>=lo)&(rho<hi));t=np.exp(-.5*(rho/(23+10*i))**2)*m;newpair.append(pool('extended_secondary%d'%i,view('extended_patch%d'%i,dense('distance_to_secondary%d'%i,secmap,np.diag(t)),1,1,P),1,P))
# A second, weaker maximum on opposite sides of a dividing line avoids
# assuming that the cutout center falls on one of the nuclei.
peakmap=view('clean_peak_vector',point('sign_of_extended_crest',nc,[[-2]]),P,1,1)
for rad0 in (9,17,27):
    # Two complementary half-planes with a gap. A pair separated by the
    # gap is supported only as strongly as its fainter component.
    M=[]
    for angle in np.arange(6)*np.pi/6:
        slide=xc*np.cos(angle)+yc*np.sin(angle)
        for offset in (0,.64*rad0,-.64*rad0):
            for side in (1,-1):
                ok=(side*(slide-offset)>=rad0/2)&(rho<rad0+17)
                M.append(np.where(ok,0.,-3.))
    M=np.array(M);h=view('nucleus_vector'+str(rad0),peakmap,1,1,P)
    # Position mask and radial taper, shared for all dividing lines.
    taper=np.exp(-.5*(rho/(17+rad0*.48))**2); wmat=np.tile(-.5*taper,(len(M),1));
    # A rectangular dense replication would waste memory. Use a 1x1
    # spatial broadcast and a position-dependent dense bias branch.
    raw=dense('nucleus_taper'+str(rad0),peakmap,np.diag(-.5*taper));raw=view('nucleus_map'+str(rad0),raw,1,1,P)
    raw=point('divide_field'+str(rad0),raw,np.ones((len(M),1)))
    ob=dense('partition_mask'+str(rad0),pk,np.zeros((M.size,1)),M.ravel());ob=view('partition'+str(rad0),ob,len(M),1,P)
    s=pool('half_peaks'+str(rad0),add('isolated_peaks'+str(rad0),raw,ob),1,P);s=point('pair_neg'+str(rad0),s,-np.eye(len(M)))
    s=pool('fainter_peak_neg'+str(rad0),view('nucleus_pairs'+str(rad0),s,len(M)//2,1,2),1,2);s=point('fainter_peak'+str(rad0),s,-np.eye(len(M)//2));s=pool('best_pair'+str(rad0),view('nuclei'+str(rad0),s,1,1,len(M)//2),1,len(M)//2);mats.append(s)
# Assemble a small named catalog of scalar intermediate measurements. All
# subsequent decisions use these quantities in physical units or ratios.
blocks=[]
def register(x,names):
    assert len(names)==np.prod(shapes[x]),(x,names,shapes[x]);blocks.append((x,names))
register(faintshape,['F%d_%s'%(j,s) for j in range(3) for s in ('mean','1c','1s','2c','2s','3c','3s','4c','4s','5c','5s')])
register(chir,['chir%d'%i for i in range(T)])
for i,t in enumerate(newpair):register(t,['extpair%d'%i])
register(pool('flux_at_center',point('zero_radius',f,np.eye(R)[:1]),A,1,'avgpool'),['centerfrac'])
for i,t in enumerate(absstats):register(t,['asym%d'%i])
for i,t in enumerate(smalls):register(t,['local%d'%i])
register(armtest,['pitch%d'%i for i in range(T)])
for i,v in enumerate(mats):register(v,['pair%d'%i])
register(pk,['peak']);register(sk,['negsky'+str(i) for i in range(3)])
# Band photometry in concentric annuli; Gaussian-like kernels avoid hard
# truncation of the disk and keep the central galaxy dominant.
rs=[2.5,5.5,10.,17.,28.];aps=[unit(np.exp(-.5*(rad/s)**2)*(1 if j==0 else rad**2)) for j,s in enumerate(rs)]
register(dense('multiband_photometry',low,bands(aps)),['band%d_%d'%(i,j) for i in range(3) for j in range(len(rs))])
B=np.stack([np.ones(A)]+[fn(m*angles)*2 for m in (2,4,6) for fn in (np.cos,np.sin)])/A
register(ang_transform('even_shape_modes',ld,B),['D%d_%s'%(j,s) for j in range(L) for s in ('mean','2c','2s','4c','4s','6c','6s')]);B=np.stack([fn(m*angles)*2 for m in (1,3,5) for fn in (np.cos,np.sin)])/A
register(ang_transform('odd_shape_modes',lr,B),['R%d_%s'%(j,s) for j in range(L) for s in ('1c','1s','3c','3s','5c','5s')]);register(pool('long_axis',diameter,A,1),['major%d'%j for j in range(L)])
register(pool('short_axis_neg',point('negate_diameters',diameter,-np.eye(L)),A,1),['negminor%d'%j for j in range(L)])
# A local reversal in the radial gradient is a powerful clue: combine
# its prominence, absolute height and the surface brightness of the
# bridge of light between the two peaks. Measure both compact crests
# and those that subtend several angular resolution elements.
# Skip the inner PSF; use moderate smoothing to reject isolated stars.
dir_ranges=[(5,23),(10,41),(18,60)]
def region_stats(n,x,B):
    # B has one scalar channel per radius; average adjacent angular rays.
    sm=(2*M0+np.roll(M0,1,axis=1)+np.roll(M0,-1,axis=1))/4
    use=np.flatnonzero((r>=B[0])&(r<=B[1]));mat=np.zeros((len(use),R));mat[np.arange(len(use)),use]=1
    # Average after maximising along radius, peak need not be at precisely
    # the same radius on adjacent rays.
    t=point(n+'_range',x,mat);t=view(n+'_transpose',t,1,len(use),A);t=pool(n+'_radial_peak',t,len(use),1)
    t=dense(n+'_azimuth',t,sm);t=view(n+'_rays',t,1,1,A)
    return pool(n,t,1,A)
# Penalties diminish evidence for isolated sources with no luminous bridge.
gap=relu('empty_sky_gap',point('bridge_test',base,-np.eye(R),.105))
crest=add('companion_response',delta,point('bridge_penalty',gap,-3.8*np.eye(R)))
for j,bound in enumerate(dir_ranges):register(region_stats('secondary_peak%d'%j,crest,bound),['secondary%d'%j])
# Smooth radial brightness test for concentration/dust; averages after
# removing disconnected sources, plus a high-order angular roughness cue.
for j,rr in enumerate([6.,12.,20.,32.]):
    u=unit(np.exp(-.5*((r-rr)/(.15*rr+1))**2)*dr);t=point('radial_flux%d'%j,connect,u[None,:]);register(pool('azimuthal_flux%d'%j,t,A,1,'avgpool'),['flux%d'%j]);register(pool('ridge_flux%d'%j,t,A,1),['ridge%d'%j])
# Evidence for arm/patch texture: substantial positive rebrightening over
# *many* rays, discounting any rays which have crossed into empty sky.
tex=relu('arm_crests',point('arm_threshold',crest,np.eye(R),-.025));z=region_stats('brightest_crest',tex,(7,44));
sel=(r>=7)&(r<=46);use=np.eye(R)[sel];t=point('arm_region',tex,use);t=pool('arm_max_along_ray',view('arm_scan',t,1,int(sel.sum()),A),int(sel.sum()),1);register(pool('distributed_arm_light',t,1,A,'avgpool'),['arm_rise'])

# Tiny algebra compiler. An expression is an affine form; the only
# nonlinear atoms are ReLUs. The compiler packs simultaneous scalar
# operations into layers and carries only still-needed measurements.
names=[n for _,keys in blocks for n in keys]; K=len(names);starts=[]; i=0
for n,keys in blocks:
    mat=np.zeros((K,len(keys)));mat[i:i+len(keys)]=np.eye(len(keys));starts.append(dense('record_'+str(i),n,mat));i+=len(keys)
cat=add('photometry_record',*starts)
class S:
    def __init__(self,d=None,b=0):self.d=d or {};self.b=float(b)
    def __add__(self,o):
        if not isinstance(o,S):return S(self.d,self.b+o)
        d=self.d.copy()
        for k,v in o.d.items():d[k]=d.get(k,0)+v
        return S({k:v for k,v in d.items() if v!=0},self.b+o.b)
    __radd__=__add__
    def __mul__(self,c):return S({k:v*c for k,v in self.d.items()},self.b*c)
    __rmul__=__mul__
    def __truediv__(self,c):return self*(1/c)
    def __neg__(self):return self*(-1)
    def __sub__(self,o):return self+(-o)
    def __rsub__(self,o):return o+(-self)
U={n:S({j:1}) for j,n in enumerate(names)}; nodes=[]
def pos(x):
    nodes.append(x);return S({K+len(nodes)-1:1})
def mx(a,b):return a+pos(b-a)
def mn(a,b):return a-pos(a-b)
def ab(x):return 2*pos(x)-x
def clip(x,a,b):return a+pos(x-a)-pos(x-b)
def ramp(x,a,b):return clip((x-a)/(b-a),0,1)
def lut(x,kn,val):
    sl=np.diff(val)/np.diff(kn);return float(val[0])+sum(float(v)*pos(x-float(k)) for k,v in zip(kn,np.diff(np.r_[0,sl,0])))
def hypot(x,y):
    # Angular quadrature of absolute projections of the vector.
    return sum(ab(x*np.cos(t)+y*np.sin(t)) for t in np.arange(8)*np.pi/8)*(np.pi/16)
def log(x):
    k=np.geomspace(.006,96,58);return lut(x,k,np.log(k))
def sq(x):
    k=np.linspace(-2,2,41);return lut(x,k,k*k)
def sqdiff(x,y):return sq(x)-sq(y)

def compile_scores(outcomes):
    # Lift initial signed inputs above zero when carrying through ReLUs.
    # The shift is removed exactly by the next affine layer.
    expr=nodes+outcomes;depth=[0]*K
    for x in nodes:depth.append(1+max((depth[j] for j in x.d),default=0))
    active=set(k for out in outcomes for k in out.d)
    for j in range(len(nodes)-1,-1,-1):
        if K+j in active:active.update(nodes[j].d)
    last=max(depth[j] for j in active)+1;life=[0]*len(depth)
    for idx,x in enumerate(expr):
        if idx<len(nodes) and K+idx not in active:continue
        d=depth[K+idx] if idx<len(nodes) else last
        for j in x.d:life[j]=max(life[j],d)
    cur=list(range(K));n=cat
    for d in range(1,last+1):
        passthru=[i for i in cur if life[i]>d];created=[i for i in range(K,len(depth)) if depth[i]==d and life[i]>d]
        rows=[S({i:1}) for i in passthru]+[nodes[i-K] for i in created] if d<last else outcomes
        shifts=[32. if i<K else 0. for i in passthru]+[0.]*len(created) if d<last else [0.]*len(rows)
        # Positive carries include major axis lengths; only the negated
        # minor axis is potentially below -32 (its absolute bound is 62).
        shifts=[128. if v else 0. for v in shifts]
        index={v:i for i,v in enumerate(cur)};W=np.zeros((len(rows),len(cur)));b=np.array([x.b for x in rows])+shifts
        for t,e in enumerate(rows):
            for j,c in e.d.items():
                W[t,index[j]]=c
                if d>1 and j<K:b[t]-=128*c
        n=dense('morphology_%02d'%d,n,W,b)
        if d<last:n=relu('decisions_%02d'%d,n)
        cur=passthru+created

# Interpret the catalog. For an ideal elliptical isophote, the second
# Fourier amplitude of log radius is (a-b)/(a+b), and the fourth is
# one half the square of the second (including the doubled phase).
E=[]; V=[]; lmaj=[]; ecc=[]; irregular=[]; dip=[]; odd=[]; logscale=[]
for j in range(L):
    v=lambda s:U['D%d_%s'%(j,s)]
    c,s=v('2c'),v('2s');V.append((c,s)); e=hypot(c,s);E.append(e)
    irregular.append(hypot(v('4c')-.5*sqdiff(c,s),v('4s')-.25*sqdiff(c+s,c-s)))
    mag=lambda m:hypot(U['R%d_%dc'%(j,m)],U['R%d_%ds'%(j,m)])
    dip.append(mag(1));odd.append(mag(3)+.55*mag(5));logscale.append(v('mean'))
    a=U['major%d'%j];lmaj.append(log(a));b=-U['negminor%d'%j];ecc.append(log(a)-log(b))

def change(j,k):return hypot(V[j][0]-V[k][0],V[j][1]-V[k][1])
# log major/minor. Two geometric estimators: extrema of a smoothed
# diameter curve, and its low harmonic (less sensitive to stray stars).
k=np.linspace(0,.88,24)
def logaxis(j):return .47*ecc[j]+.53*lut(E[j],k,np.log((1+k)/(1-k)))
axis=[logaxis(j) for j in range(L)]; elong=.20*axis[1]+.38*axis[2]+.30*axis[3]+.12*axis[4]
outer=.48*axis[1]+.35*axis[2]+.17*axis[0];inner=.50*axis[4]+.35*axis[5]+.15*axis[6]
major=.46*U['major1']+.35*U['major2']+.19*U['major0']
size=.55*logscale[1]+.30*logscale[2]+.15*logscale[0]
# Background-corrected color indices. Fractional color excess is more
# useful than absolute RGB in separating smooth old stars and blue arms.
cols=[];green=[]
for j in range(5):
    ch=[log(pos(U['band%d_%d'%(i,j)]+U['negsky%d'%i])+.014) for i in range(3)]
    cols.append(ch[0]-ch[2]);green.append(ch[1]-.5*(ch[0]+ch[2]))
blue=-.46*cols[1]-.40*cols[2]-.14*cols[3]; colorgrad=.72*(cols[0]-cols[2])+.28*(cols[1]-cols[3]);
# Nonellipticity, intrinsic lopsidedness and radial changes in position
# angle. A common displacement of all contours is weaker evidence than
# a distorted outer disk surrounding a well-centered core.
lop=.34*dip[0]+.38*dip[1]+.28*dip[2]; off=.40*dip[3]+.36*dip[4]+.24*dip[5]
rag=.27*irregular[0]+.32*irregular[1]+.26*irregular[2]+.15*irregular[3]
highodd=.31*odd[0]+.36*odd[1]+.23*odd[2]+.10*odd[3]
twist=.37*change(1,3)+.38*change(2,4)+.25*change(3,5)
barshift=.35*change(1,4)+.35*change(2,5)+.30*change(0,3)
bar_mid=mx(.6*E[2]+.4*E[3],.55*E[4]+.45*E[3]);bar_in=.55*E[3]+.45*E[4];bar_out=.58*E[0]+.42*E[1]
sec=mx(U['secondary0'],.95*U['secondary1']); far=U['secondary2']; arm=U['arm_rise'];
# Interpretable switches kept soft near a category boundary.
feat=(.95*ramp(lop,.06,.25)+1.25*ramp(highodd,.035,.16)+1.2*ramp(rag,.030,.135)
     +1.30*ramp(twist,.045,.20)+.9*ramp(arm,.015,.11)+.60*ramp(colorgrad,.16,.54)
     +.90*ramp(blue,-.06,.42)+.70*ramp(size,np.log(11),np.log(23)))
spiral=feat-2.2
# Flat disks: reliable evidence is long thin outer isophotes, together
# with either a central bulge or a long ridge of shallow surface brightness.
flat=.40*outer+.60*elong
ridge=.6*U['ridge2']+.4*U['ridge3']; leng=ramp(major,16,33)
edge=3.0*(flat-1.02)+.85*(leng-.45)+.30*ramp(ridge,.2,.52)-.60*ramp(highodd,.12,.30)
# A bright central spheroid changes the axis ratio towards the center
# and shortens the luminous ridge. Conversely a dust lane may dim the
# very center but leaves the bright elongated disk visible at large radii.
bulge=1.8*(outer-inner)+.80*(1-ramp(inner,1.12,1.8))-.90*ramp(ridge,.24,.59)-.65*(leng-.5)+.30*ramp(cols[0],.04,.47)-.32
cigar=1.0-2.6*ab(elong-1.01)-.95*leng-.65*ramp(outer-inner,.12,.55)-.6*ramp(blue,.02,.4)
# A second nucleus requires both a radial reversal and a luminous
# connection, or very high contrast. The same criteria at small radius
# also cover systems whose catalog center lies between two components.
merger= -2.4+3.6*ramp(sec,.13,.70)+1.8*ramp(far,.18,.65)+1.15*ramp(lop,.17,.53)+.85*ramp(off,.20,.58)+.42*ramp(highodd,.13,.34)-.40*ramp(blue,.05,.55)
# False companions in the extreme outskirts of an otherwise nearly
# perfect ellipse (a frequent projection on a crowded patch of sky).
merger-=.85*(1-ramp(lop+highodd+twist,.12,.39))
# Prefer a bar when inner twofold distortion is substantial but is not
# explained by the ellipticity and direction of the surrounding disk.
bar=1.20*ramp(bar_in,.13,.34)+.80*ramp(bar_mid-bar_out,-.06,.16)+1.28*ramp(barshift,.085,.32)+.45*ramp(colorgrad,.12,.50)-.62*ramp(lop,.17,.42)-1.82
# Tight spirals typically have a more regular, bulge-dominated image;
# open arms generate greater isophote twisting, color and radial contrast.
loose=.82*ramp(blue,-.10,.43)+.69*ramp(highodd,.04,.19)+.68*ramp(twist,.06,.25)+.48*ramp(arm,.018,.12)+.40*ramp(lop,.10,.33)-1.35
# Classes compete hierarchically, without letting a high arm score
# double-count structure in an edge-on disk or a merging pair.
smooth_base=.90-1.7*ramp(spiral,-.65,.85)
spiral_base=.02+1.7*ramp(spiral,-.65,1.15)-1.5*ramp(flat,.92,1.5)
edge_base=.32+clip(edge,-3,2.6)-.66*ramp(twist,.17,.46)
roundness=(elong-.25)*4.8
s2=smooth_base+clip(-roundness,-3,.85);s3=smooth_base+clip(roundness,-2,.65)-2.5*pos(elong-.86)
s4=smooth_base+cigar-.12
s5=spiral_base+clip(bar,-1.3,1.2);s6=spiral_base+clip(-loose,-1.,.84)-.48*pos(bar);s7=spiral_base+clip(loose,-1.,.96)-.40*pos(bar)
s8=edge_base+clip(-bulge,-.8,.8);s9=edge_base+clip(bulge,-.8,.8)
# Disturbed is reserved for strong irregularity for which an isolated
# organized spiral or a distinct secondary nucleus is a poor description.
s0=-1.25+1.2*ramp(lop,.17,.45)+1.2*ramp(highodd,.13,.36)+.73*ramp(off,.15,.45)+.62*ramp(blue,.05,.50)-.7*ramp(arm,.08,.24)-.48*ramp(bar_in,.16,.4)
# Compare scale-free winding signatures for the three families of pitch
# angle. Absolute arm light rejects a tiny target with a noisy halo.
peak=U['peak']; gainarm=[ramp(U['pitch%d'%j]-.016*peak,.001,.022) for j in range(T)]
strong=mx(gainarm[0],mx(gainarm[1],gainarm[2]));openarm=clip((U['pitch2']-U['pitch0'])*95,-1,1)
# Old stellar populations tend to be warm throughout a featureless
# spheroid. Young disk light can be conspicuous even when the arms
# themselves have little contrast or angular resolution.
cool=ramp(blue,-.23,.25);haloblue=ramp(-cols[2],-.15,.26);gradient=ramp(colorgrad,.06,.38)
photodisk=.72*cool+.60*haloblue+.80*gradient+.65*ramp(size,np.log(9),np.log(20))+.8*strong
structured=.55*spiral+photodisk-1.16
smooth_base=1.15-1.55*ramp(structured,-.30,.9)
spiral_base=.23+1.52*ramp(structured,-.12,1.05)-1.65*ramp(flat,.82,1.43)
# The very elongated, lens-like smooth type tends not to have a distinct
# embedded thin disk: distinguish it from the edge-on classes using the
# extent of the major axis, outer blue disk, and central thickening.
edge_base=.66+clip(edge+.72+.45*ramp(colorgrad,.12,.43),-3,2.7)-.5*ramp(twist,.17,.46)
bulge+=.30+.43*ramp(cols[1],.14,.52)-.40*ramp(inner,1.35,1.85)
s2=smooth_base+clip(-roundness,-3,.85);s3=smooth_base+clip(roundness,-2,.65)-3.4*pos(elong-.68)
s4=smooth_base+1.24-2.4*ab(elong-.98)-1.04*leng-.65*ramp(outer-inner,.12,.55)-.65*ramp(blue,.02,.4)
# The bar is an aligned central elongation embedded in a disk at a
# different inclination. Require the inner evidence, not just arm power.
bar=1.06*ramp(bar_in,.085,.28)+.88*ramp(bar_mid-bar_out,-.065,.115)+1.05*ramp(barshift,.06,.235)+.52*gradient-.55*ramp(lop,.17,.42)-1.67
loose=.88*ramp(blue,-.22,.30)+.56*ramp(highodd,.035,.17)+.60*ramp(twist,.045,.21)+.38*ramp(arm,.009,.09)+.34*ramp(lop,.09,.28)+.57*openarm+.35*haloblue-1.38
s5=spiral_base+clip(bar,-1.25,1.18);s6=spiral_base+clip(-loose,-1.,.78)-.43*pos(bar);s7=spiral_base+clip(loose,-1.,1.04)-.40*pos(bar)
s8=edge_base+clip(-bulge,-.86,.86);s9=edge_base+clip(bulge,-.86,.86)
# Joint evidence for a bright secondary peak and a trough in the
# surface-brightness field. Local peak tests omit the need for the two
# nuclei to be exactly collinear with the center pixel.
nuc=.66*U['pair0']+.25*U['pair1']+.09*U['pair2']; pair_sig=ramp(nuc-.038*peak,.002,.072)
nucleus=1.45*pair_sig+.63*ramp(nuc,.018,.084)
s1=-.48+nucleus+1.20*ramp(sec,.10,.49)+.77*ramp(far,.11,.51)+.90*ramp(lop,.10,.45)+.55*ramp(highodd,.09,.29)-.45*cool-.60*(1-ramp(lop+highodd+twist,.10,.36))
# Patchy blue objects may have lop-sided light even in the center,
# unlike spirals where most of the lopsidedness is in the outer arms.
s0=.01+1.26*ramp(lop,.10,.36)+1.16*ramp(highodd,.075,.30)+.83*ramp(off,.10,.40)+.75*cool-.57*ramp(arm,.08,.24)-.65*pair_sig-.50*ramp(bar_in,.14,.37)-.32*ramp(size,np.log(16),np.log(30))
# Dimensionless patch, arm-gap and odd-light strengths. Use each residual
# relative to the light in its annulus, not relative to the sky frame.
asym=[U['asym%d'%j] for j in range(4)];fl=[pos(U['flux%d'%j]) for j in range(4)]
aodd=[ramp(asym[j]-.06*fl[j],.003,.012+.016*(3-j)) for j in range(4)]
outsym=.16*aodd[0]+.24*aodd[1]+.35*aodd[2]+.25*aodd[3]
patch=[];gaps=[]
for jj in range(3):
    # fainter regions have lower thresholds for measurable structure.
    sky_floor=.007 if jj<2 else .005
    kk=jj*3
    # 2D knots that survive smoothing. Adjacent annuli vary smoothly in
    # ordinary disks, whereas spiral arms form sharp, high-contrast knots.
    knots=U['local%d'%kk]+.65*U['local%d'%(kk+9)]
    # Approximate conversion between relative and absolute photometry for
    # thresholding: broad red+green+blue apertures around the center.
    jj0=jj+1;band=sum((U['band%d_%d'%(i,jj0)]+U['negsky%d'%i]) for i in range(3))/3
    gaps.append(ramp(U['local%d'%(kk+2)]+U['local%d'%(kk+11)]-.018*pos(band),.00015,.0038))
    patch.append(ramp(knots-.018*pos(band),.0004,.0045))
substructure=.17*patch[0]+.39*patch[1]+.44*patch[2];gaplight=.20*gaps[0]+.44*gaps[1]+.36*gaps[2]
# Outer disk light of a blue hue, even around a red bulge, is a stronger
# disk indicator than hue at the centroid. Also use the absolute linear
# RGB difference to avoid diluting disk colors at low surface brightness.
oB=U['band2_2']+U['negsky2'];oR=U['band0_2']+U['negsky0'];oG=U['band1_2']+U['negsky1'];
blue_halo=.64*ramp(oB-.90*oR,-.004,.018)+.36*ramp(oB-.97*oG,-.004,.016)
# Broad bright disks with a slowly falling inner profile are distinct
# from a profile dominated by a compact, nearly unresolved spheroid.
concentration=logscale[1]-logscale[4]; disk_extent=ramp(major,13,28); innerextent=ramp(logscale[4],np.log(4),np.log(10))
physical=.70*blue_halo+.54*gaplight+.70*substructure+.42*outsym+.39*disk_extent+.36*innerextent
structured+=physical-.68
smooth_base=1.22-1.68*ramp(structured,-.28,.94)
spiral_base=.24+1.68*ramp(structured,-.10,1.17)-1.66*ramp(flat,.85,1.46)
s2=smooth_base+clip(-roundness,-3,.84);s3=smooth_base+clip(roundness,-2,.60)-3.4*pos(elong-.68)
s4=smooth_base+1.24-2.4*ab(elong-.98)-1.04*leng-.73*ramp(outer-inner,.12,.55)-.65*ramp(blue,.02,.4)
# An open arm is often patchier and bluer than a coherent tight wave.
loose+=.65*blue_halo+.57*substructure+.38*gaplight+.34*outsym-.56
bar+=.48*ramp(innerextent+bar_in, .38,1.02)-.53*blue_halo-.28*substructure+.12
s5=spiral_base+clip(bar,-1.24,1.12);s6=spiral_base+clip(-loose,-1,.84)-.46*pos(bar);s7=spiral_base+clip(loose,-1,1.13)-.43*pos(bar)
# Require extended light around a candidate pair; the azimuthal odd flux
# falling only in the empty, star-dominated sky is not equivalent to a
# broad asymmetric envelope around the target.
s1+=1.04*outsym+.36*substructure+.27*ramp(off,.06,.30)-.62+.56*ramp(sec,.48,1.0)-.28*ramp(flat,1.15,1.7)
s0+=.65*outsym+.51*substructure+.35*blue_halo+.29*ramp(off,.09,.3)-.70
# Very thin disks with nearly unchanged central flattening are less
# likely to conceal a bulge than thicker systems with the same colour.
bulge-=.44*ramp(flat,1.25,1.78)+.36*ramp(ridge,.22,.55)-.13
edge_base+=.43*ramp(major,20,37)-.12
s8=edge_base+clip(-bulge,-.86,.86);s9=edge_base+clip(bulge,-.86,.86)
# Bending of the center line with radius indicates a disturbed/tidal
# galaxy. Simple projection or a consistent centering error produces a
# roughly constant photocenter across isophotes, not a changing one.
kn=np.linspace(-2,4,25);dist=lut(logscale[1],kn,np.exp(kn));localdist=lut(logscale[4],kn,np.exp(kn))
def times(x,y):
    # bounded product via difference of two squares, scale-free inputs
    return .25*sqdiff(x+y,x-y)
shiftvec=[]
for mode in ['1c','1s']:
    shiftvec.append(times(clip(dist/24,0,1.4),clip(U['R1_'+mode],-.55,.55))-times(clip(localdist/24,0,1.4),clip(U['R4_'+mode],-.55,.55)))
bent=ramp(hypot(*shiftvec),.035,.18)
decentered=ramp(1-U['centerfrac'],.055,.30)
# Magnitude / color remain informative at fixed outline: a poorly
# resolved, diffuse, blue or recently disturbed dwarf need not exhibit
# well defined arms. Strong mirror-asymmetric winding favors a spiral.
cht=[ramp(U['chir%d'%j]-.006*peak,.0008,.016) for j in range(3)];winding=mx(cht[0],mx(cht[1],cht[2]))
chopen=clip((U['chir2']-U['chir0'])*95,-1,1)
brightness=ramp(peak,.24,.59);compact=1-ramp(major,13,25);orange=ramp(cols[0],.10,.53)
ex=.55*U['extpair0']+.30*U['extpair1']+.15*U['extpair2'];galpair=ramp(ex-.055*peak,.009,.11)
extended_pair=ramp(nuc-.060*peak,.008,.092)
# Hierarchical comparisons: a clear interacting pair takes precedence
# over the individual components' Hubble types; smooth, nearly convex
# isophotes with no extended second peak argue for a foreground star.
s1=.06+1.42*galpair+1.03*extended_pair+.62*ramp(sec,.08,.45)+.66*outsym+.48*bent+.36*decentered+.41*ramp(lop,.12,.37)+.25*orange-.64*(1-ramp(lop+highodd+twist,.06,.24))-.45*(1-ramp(ex+nuc,.05,.18))-.36*ramp(flat,1.4,1.9)
s0+=.55*compact+.68*(1-brightness)+.82*bent+.41*decentered+.40*(1-winding)-.31*orange-.80-.42*extended_pair
# Angular size and surface brightness complement arm contrast. Big,
# bright, uniformly coloured spirals can have very little contrast yet
# still subtend many resolution elements. A tiny blue fuzzy object is
# weak evidence for arms unless it has a recognizable winding mode.
structured+=.73*ramp(size,np.log(12),np.log(22))+.68*winding+.36*brightness+.48*blue_halo-.63-.45*compact
smooth_base=1.25-1.62*ramp(structured,-.24,.95)
spiral_base=.25+1.65*ramp(structured,-.1,1.17)-1.45*ramp(flat,.86,1.45)
s2=smooth_base+clip(-roundness,-3,.84);s3=smooth_base+clip(roundness,-2,.60)-3.8*pos(elong-.66)
s4=smooth_base+1.24-2.4*ab(elong-.96)-.98*leng-.65*ramp(outer-inner,.12,.55)-.56*ramp(blue,.01,.38)
# Detect a faint thin disk around a thicker bulge. Outer levels see the
# disk, inner levels the spheroid; compare both against the intermediate
# levels to discount uniformly cigar-shaped isophotes.
faintdisk=ramp(.56*axis[0]+.44*axis[1]-inner,.08,.55);thindisk=ramp(outer,.88,1.38)
edge_base+=.65*mn(faintdisk,thindisk)+.37*mn(ramp(major,13,25),thindisk)-.29*ramp(winding,.55,1)
bulge+=.45*faintdisk+.47*(1-ramp(major,16,31))-.38*leng-.20*ramp(flat,1.4,1.85)
s8=edge_base+clip(-bulge,-.90,.85);s9=edge_base+clip(bulge,-.90,.85)
# Tight arms and a prominent bar more often surround a luminous bulge;
# diffuse blue galaxies with an unbarred center favor the open-arm type.
loose+=.57*chopen+.57*(1-brightness)+.53*blue_halo-.49*orange-.16
bar+=.39*brightness+.31*orange-.28-.48*mn(1-brightness,blue_halo)
s5=spiral_base+clip(bar,-1.20,1.12);s6=spiral_base+clip(-loose,-1,.88)-.46*pos(bar);s7=spiral_base+clip(loose,-1,1.16)-.43*pos(bar)
# Use a broad, changing odd mode at faint isophotes to look for tidal
# debris. Small, isolated bright patches seldom deform multiple faint
# contours coherently. A consistent double spiral is strongly even.
ft=[];fd=[];fe=[];ftwist=[]
for j in range(3):
    F=lambda m:U['F%d_%s'%(j,m)]
    fd.append(hypot(F('1c'),F('1s')));ft.append(hypot(F('3c'),F('3s'))+.45*hypot(F('5c'),F('5s')))
    fe.append(hypot(F('2c'),F('2s')));ftwist.append(hypot(F('2c')-V[3][0],F('2s')-V[3][1]))
faintodd=.35*fd[0]+.45*fd[1]+.2*fd[2];faintkink=.30*ft[0]+.48*ft[1]+.22*ft[2]
tail=.49*ramp(faintodd,.085,.29)+.51*ramp(faintkink,.08,.24)
# Disturbed is a positive asymmetry call, not a fallback for normal but
# unresolved galaxies. Score a faint *uneven* envelope and a weak or
# disrupted core, versus an intact high surface brightness nucleus.
dwarf=mn(1-brightness,blue_halo);irreg=.36*tail+.25*bent+.23*outsym+.16*ramp(highodd,.055,.20)
s0=.10+1.13*irreg+1.16*tail+.52*bent+.56*dwarf+.48*(1-brightness)+.53*compact+.42*decentered-.52*winding-.45*extended_pair-.30*orange+.48*mn(irreg,compact)
# Extendedness is now tested locally. Demand high contrast from a
# symmetric system; if it has tidal structure, accept a fainter pair.
s1-=.22+.74*(1-mx(tail,outsym))+.44*mn(thindisk,1-bent)-.32*tail
# A bulging disk can have a small bright inner spheroid without being
# smooth/cigar-shaped. Include the shape of the absolute outer contour.
fedge=ramp(.58*fe[1]+.42*fe[2],.43,.62);diskex=ramp(U['F1_mean'],np.log(9),np.log(16))
edge_base+=.53*mn(fedge,diskex)+.4*mn(thindisk,ramp(.6*axis[1]+.4*axis[2]-inner,.15,.5))-.29*ramp(faintkink,.13,.35)
bulge+=.43*mn(1-leng,thindisk)+.37*faintdisk-.23*leng
s8=edge_base+clip(-bulge,-.9,.85);s9=edge_base+clip(bulge,-.9,.85)
# A reddened spheroid set in a relatively blue round disk is strong
# evidence for a face-on disk even with weak arms. Inner colour is a
# better population indicator than total brightness for loose vs tight.
faintsize=.55*U['F1_mean']+.45*U['F2_mean'];coherent=.62*winding+.38*strong
structured+=.53*mn(gradient,ramp(faintsize,np.log(9),np.log(19)))+.42*ramp(faintsize,np.log(13),np.log(22))+.34*ramp(.52*ftwist[1]+.48*ftwist[2],.09,.25)-.27
smooth_base=1.25-1.63*ramp(structured,-.24,.95)
spiral_base=.25+1.65*ramp(structured,-.1,1.17)-1.42*ramp(flat,.88,1.46)
s2=smooth_base+clip(-roundness,-3,.84);s3=smooth_base+clip(roundness,-2,.60)-3.8*pos(elong-.66)
s4=smooth_base+1.24-2.4*ab(elong-.96)-.98*leng-.65*ramp(outer-inner,.12,.55)-.56*ramp(blue,.01,.38)
# Spiral subtype evidence from odd/even power and the stellar population
# of the central galaxy: young blue cores and open, irregular outer arms
# vs an old central bar and a coherent narrow inner disk.
bluecore=ramp(-cols[0],-.15,.28);longbar=ramp(bar_in-bar_out,-.016,.14)
loose=.54*loose+.85*bluecore+.49*blue_halo+.41*(1-brightness)+.45*tail+.29*chopen-.70
bar=.69*bar+.68*longbar+.38*gradient+.42*orange-.63*bluecore-.30-.32*ramp(lop,.09,.28)
s5=spiral_base+clip(bar,-1.20,1.13);s6=spiral_base+clip(-loose,-1,.88)-.46*pos(bar);s7=spiral_base+clip(loose,-1,1.15)-.43*pos(bar)
compile_scores([s0,s1,s2,s3,s4,s5,s6,s7,s8,s9])
# Remove measurements with no downstream use (common subexpressions in
# explanatory branches need not be evaluated at inference time).
need={layers[-1]['name']}
for layer in reversed(layers):
    if layer['name'] in need:need.update(layer['inputs'])
layers=[l for l in layers if l['name'] in need];weights={k:v for k,v in weights.items() if k.rsplit('.',1)[0] in need}
(out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n');np.savez(out/'weights.npz',**weights)
