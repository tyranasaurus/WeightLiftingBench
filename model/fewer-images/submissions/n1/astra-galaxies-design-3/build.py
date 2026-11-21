"""Analytical photometry and morphology network. No input files or learned parameters."""
import json
import math
import sys
from pathlib import Path
import numpy as np

F=np.float32
layers=[]
weights={}
shapes={'image':(3,128,128)}
def layer(name,op,inputs,outshape,**kw):
    assert name not in shapes,name
    layers.append(dict(name=name,op=op,inputs=[inputs] if isinstance(inputs,str) else inputs,**kw))
    shapes[name]=tuple(outshape)
    return name
def dense(name,src,w,b=0):
    w=np.asarray(w,F).reshape(-1,math.prod(shapes[src])); n=len(w)
    weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(b,(n,)).astype(F).copy()
    return layer(name,'dense',src,(n,),out=n)
def conv(name,src,w,b=0,stride=1,pad=0):
    w=np.asarray(w,F); o,c,k,_=w.shape
    h,v=[(n+2*pad-k)//stride+1 for n in shapes[src][1:]]
    weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(b,(o,)).astype(F).copy()
    return layer(name,'conv',src,(o,h,v),out=o,kernel=[k,k],stride=stride,padding=pad)
def pool(name,src,op,sz,st):
    c,h,w=shapes[src]; a,b=sz; i,j=st
    return layer(name,op,src,(c,(h-a)//i+1,(w-b)//j+1),size=list(sz),stride=list(st))
def rel(name,src): return layer(name,'relu',src,shapes[src])
def reshape(name,src,*s): return layer(name,'reshape',src,s,shape=list(s))
def add(name,src): return layer(name,'add',src,shapes[src[0]])

# Scalar circuit compiler. Each register is an affine expression in existing
# vector ports; rectifiers and normalizers are emitted in batches by stage.
class E:
    def __init__(self,d=None,b=0.): self.d=d if d is not None else {}; self.b=float(b)
    def __add__(self,o):
        o=expr(o); d=self.d.copy()
        for k,c in o.d.items(): d[k]=d.get(k,0.)+c
        return E({k:c for k,c in d.items() if abs(c)>1e-12},self.b+o.b)
    __radd__=__add__
    def __mul__(self,c): return E({k:x*c for k,x in self.d.items()},self.b*c)
    __rmul__=__mul__
    def __neg__(self): return self*-1
    def __sub__(self,o): return self+(-expr(o))
    def __rsub__(self,o): return expr(o)+(-self)
    def __truediv__(self,c): return self*(1./c)
def expr(a): return a if isinstance(a,E) else E(b=a)
def reg(name,i): return E({(name,i):1})
def emit(name,xs):
    xs=list(map(expr,xs)); n=len(xs); mats={}
    for j,x in enumerate(xs):
        for (s,k),c in x.d.items():
            if s not in mats: mats[s]=np.zeros((n,math.prod(shapes[s])),F)
            mats[s][j,k]+=c
    parts=[]; bias=np.array([x.b for x in xs],F)
    for i,(s,w) in enumerate(mats.items()): parts.append(dense(name if len(mats)==1 else name+'_'+str(i),s,w,0 if i else bias))
    return parts[0] if len(parts)==1 else add(name,parts)
class Stage:
    def __init__(self,n): self.n=n; self.x=[]
    def pos(self,x):
        self.x.append(x); return reg(self.n,len(self.x)-1)
    def clip(self,x,lo=0,hi=1): return lo+self.pos(x-lo)-self.pos(x-hi)
    def s(self,x,mid,half): return self.clip((x-mid)/(2*half)+.5)
    def ab(self,x): return self.pos(x)+self.pos(-x)
    def cap(self,x,y): return x-self.pos(x-y)
    def mx(self,x,y): return y+self.pos(x-y)
    def sq(self,x,h,n):
        # Linear interpolation of x^2 on [-nh,nh]. Extra tail is linear.
        return h*self.ab(x)+2*h*sum(self.pos(x-i*h)+self.pos(-x-i*h) for i in range(1,n))
    def end(self): rel(self.n,emit(self.n+'_input',self.x))

def directions(name,pairs):
    # The mean of (x,-x,y,-y) is zero; the norm layer gives quadrature
    # direction cosines without having to estimate an angle or divide by flux.
    a=emit(name+'_pairs',[v for x,y in pairs for v in (x,-x,y,-y)])
    a=reshape(name+'_layout',a,len(pairs),1,4)
    a=layer(name,'norm',a,shapes[a]); s=math.sqrt(.5)
    return [(s*reg(a,4*i),s*reg(a,4*i+2)) for i in range(len(pairs))]

def to_features(name,x):
    a=emit(name,x); return [reg(a,i) for i in range(len(x))]

def mixed_contrasts(name,frameweights,values):
    # A shared 1x1 ReLU arithmetic circuit multiplies all gate/contrast
    # pairs pointwise, without expanding a large block-diagonal matrix.
    g=emit(name+'_gates',frameweights); v=emit(name+'_values',values); n=len(values)
    p=add(name+'_sum',[g,v]); m=add(name+'_diff',[g,dense(name+'_opposite',v,-np.eye(n))]);
    kp=[]; biases=[]; coeff=[]
    for i in range(9):
        for sg in (-1,1):
            kp.append(sg); biases.append(-i*.25); coeff.append(.25 if not i else .50)
    def square(src,tag):
        a=reshape(name+tag+'_layout',src,1,1,n)
        h=rel(name+tag+'_hinges',conv(name+tag+'_offsets',a,np.array(kp)[:,None,None,None],biases))
        return conv(name+tag+'_square',h,np.array(coeff).reshape(1,-1,1,1)*(1 if tag=='P' else -1)*.25)
    z=add(name+'_products',[square(p,'P'),square(m,'M')]); return [reg(z,i) for i in range(n)]

# First average the noise over 2x2 pixels. Use a local sky estimate at each
# aperture rather than normalizing the full photograph (which includes stars).
photo=pool('bin_pixels','image','avgpool',(2,2),(2,2))
white=conv('luminance',photo,np.array([.32,.40,.28]).reshape(1,3,1,1))
coarse=pool('bin_luminance',white,'avgpool',(2,2),(2,2))

def grid(n):
    a=((np.arange(n)+.5)/n-.5)*128
    return np.meshgrid(a,a)
def gauss(x,y,r): return np.exp(-.5*(x*x+y*y)/(r*r))
def unit(a): return a/(a.sum()+1e-30)
def sky(x,y,r): return unit(gauss(x,y,2*r)-gauss(x,y,r))

def clamps(x,y,r):
    # Gaussian-windowed moments are far less vulnerable to companions than
    # unwindowed moments. All kernels are defined in input pixel units.
    g=unit(gauss(x,y,r)); sk=sky(x,y,max(15,2*r))
    u=x/r; v=y/r
    def k(w): return w-sk*w.sum()
    return [k(g),k(g*u),k(g*v),k(g*(u*u+v*v)),k(g*(u*u-v*v)),k(g*2*u*v)]

scales=(4.,7.,12.,20.,32.)
x,y=grid(64)
kernels=[m for r in scales for m in clamps(x,y,r)]
mm=dense('windowed_photometry',white,kernels)
# f, x, y, trace, Q, U. Unit vector in (Q,U) encodes twice the
# position angle. A second normalization by f produces bounded moment ratios.
raw=[]
for j,r in enumerate(scales):
    f,mx,my,t,q,u=[reg(mm,6*j+k) for k in range(6)]
    raw.extend([(mx,f),(my,f),(t,f),(q,t),(u,t)])
    # A mass/flux ratio t/f crosses one when a circular Gaussian has sigma=r.
    # It works for a mixture of a disk and a concentrated bulge as well.
mrat=directions('flux_and_moment_ratios',raw)
# Undo the bounded ratio of a to b, s=a/sqrt(a^2+b^2).
# Piecewise-linear inverse; the last interval is continued linearly.
def invert(st,s,knots):
    x=np.array(knots,dtype=float); t=x/np.sqrt(1+x*x)
    slopes=np.diff(x)/np.diff(t); k=np.searchsorted(x,0.)
    # Represent with zero origin; positive and negative halves separately.
    line=slopes[0]*s
    for j in range(1,len(t)-1): line+=(slopes[j]-slopes[j-1])*(st.pos(s-t[j])-max(-t[j],0.))
    return line
st=Stage('invert_moment_coordinates')
moments=[]
for j in range(len(scales)):
    dx,dy,tr,q,u=[mrat[5*j+k][0] for k in range(5)]
    dx=invert(st,dx,[-1,-.6,-.3,0,.3,.6,1]); dy=invert(st,dy,[-1,-.6,-.3,0,.3,.6,1]); tr=invert(st,tr,[0,.25,.5,.75,1,1.25,1.5,2]);
    q=invert(st,q,[-1,-.8,-.55,-.3,0,.3,.55,.8,1]); u=invert(st,u,[-1,-.8,-.55,-.3,0,.3,.55,.8,1]);
    moments.append((dx,dy,tr,q,u))
st.end()

# Use a scale interpolation gate. A sigmoid in the trace-to-light ratio
# decides whether to continue to the next larger Gaussian aperture.
st=Stage('choose_aperture'); larger=[st.s(tr,1.,.19) for dx,dy,tr,q,u in moments[:-1]]; st.end()
# The dilation order is monotone for a single centrally peaked source. Prevent
# an unrelated companion in the outskirts from turning a larger aperture on.
for j in range(1,len(larger)):
    st=Stage('nested_aperture_'+str(j)); larger[j]=st.cap(larger[j-1],larger[j]); st.end()
gate=[a-b for a,b in zip([1]+larger,larger+[0])]
# The centroid corrections subtract the dipole before computing ellipticity.
# For q,u the normalization must be by the trace; use raw centered flux
# moments normalized in the outer readout instead of dividing Q/T twice.
st=Stage('centroid_quadratics'); cent=[]
for dx,dy,tr,q,u in moments:
    xx=st.sq(dx,.08,9); yy=st.sq(dy,.08,9); xy=(st.sq(dx+dy,.12,10)-st.sq(dx-dy,.12,10))*.5
    cent.append((xx+yy,xx-yy,xy))
st.end()
# At the matched scale, trace/f is near one. Correct Q/T using a scaled
# dipole, with a cap to avoid over-correcting a multi-core object.
st=Stage('central_second_moments'); corr=[]
for (dx,dy,tr,q,u),(sum2,diff2,twoxy) in zip(moments,cent):
    corr.append((q-st.clip(diff2,-.28,.28),u-st.clip(twoxy,-.28,.28),1-st.clip(sum2,0,.38)))
st.end()

def mag(st,x,y):
    # Polygonal approximation of Euclidean length, averaged absolute projections.
    return (np.pi/12)*sum(st.ab(x*np.cos(i*np.pi/6)+y*np.sin(i*np.pi/6)) for i in range(6))
st=Stage('quadrupole_amplitudes'); norms=[mag(st,q,u) for q,u,trace in corr]; st.end()
z=directions('centered_ellipticities',[(v,t) for v,(_,_,t) in zip(norms,corr)])
st=Stage('ellipticity_ratios'); ellipt=[invert(st,a,[0,.2,.4,.6,.8,1]) for a,b in z]; st.end()
# Piecewise-linear multiplier for [0,1] gate x signed value, with the
# polarization identity. This is exact at gate=0 and at gate=1.
def product(st,g,x):
    return (st.sq(g+x,.25,9)-st.sq(g-x,.25,9))*.25
st=Stage('scale_weighted_geometry'); ell=sum(product(st,g,e) for g,e in zip(gate,ellipt));
# The outer disk and the inner bulge are inspected separately.
ellout=sum(product(st,g,ellipt[min(j+1,4)]) for j,g in enumerate(gate)); ellin=sum(product(st,g,ellipt[max(j-1,0)]) for j,g in enumerate(gate));
sz=sum(g*np.log2(r) for g,r in zip(gate,scales)); st.end()

# Polar Fourier photometry. The angular means and Fourier components are
# computed independently for overlapping annuli. Phases of successive
# bisymmetric modes encode the winding of a two-arm pattern. The m=1 and
# m=3 modes, and an off-centre peak branch below, describe interactions.
radii=(3.,4.5,6.75,10.,15.,22.5,34.)
rr=np.hypot(x,y); theta=np.arctan2(y,x); rings=[]; base=[]
for r in radii:
    rad=unit(np.exp(-.5*((rr-r)/max(1.25,.25*r))**2)); bk=sky(x,y,max(15,2*r))
    rings.append(rad-bk); base.append(rad)
    for m in [1,2,3,4]:
        for w in [np.cos(m*theta),np.sin(m*theta)]:
            # Orthogonal to constant intensity, even on a pixelated ring.
            rings.append(2*rad*(w-(rad*w).sum()))
pr=dense('azimuthal_fourier_modes',white,rings)
# Color relative to local sky, with a broad annulus for better S/N. Merge
# neighboring apertures later, during circuit evaluation, not at build time.
ck=[]
for r in scales:
    g=unit(gauss(x,y,r)); bg=sky(x,y,max(16,2.2*r)); w=g-bg
    ck.extend([np.stack([w,0*w,-w]), np.stack([.65*w,.7*w,.65*w])])
cm=dense('aperture_colour_photometry',photo,ck)
col=directions('red_blue_fraction',[(reg(cm,2*j),reg(cm,2*j+1)) for j in range(5)])
# Soft color measure is positive for redder stellar populations.
st=Stage('select_colour'); red=sum(product(st,g,a) for g,(a,b) in zip(gate,col)); st.end()

# Difference-of-Gaussians for knot/peak detection; remove rings and smooth
# bulges by demanding a local maximum in several directions. Process only
# native bins of 4x4 at this stage to keep inference memory bounded.
def smoothing(name,src,sigma,k):
    u=np.arange(k)-k//2; a,b=np.meshgrid(u,u)
    return conv(name,src,unit(gauss(a,b,sigma)).reshape(1,1,k,k),pad=k//2)
smooth=smoothing('smooth_disk',coarse,.85,5)
# Four opposite-direction comparisons: min of several differences suppresses
# monotonic slopes, including the shoulders of a single bright nucleus.
def minimum_pair(name,src,n):
    w=np.zeros((n,n*2,1,1)); common=w.copy()
    for i in range(n): w[i,2*i,0,0]=1; w[i,2*i+1,0,0]=-1; common[i,2*i,0,0]=1
    a=conv(name+'_first',src,common); diff=rel(name+'_rectify',conv(name+'_difference',src,w));
    return add(name,[a,conv(name+'_subtract',diff,-np.eye(n).reshape(n,n,1,1))])
x32,y32=grid(32); r32=np.hypot(x32,y32)
peak=[]
for distance in (2,4):
    d=distance; k=2*d+1; kernels=[]
    for a,b in [(d,0),(-d,0),(0,d),(0,-d),(round(d*.707),round(d*.707)),(-round(d*.707),-round(d*.707)),(round(d*.707),-round(d*.707)),(-round(d*.707),round(d*.707))]:
        w=np.zeros((k,k)); w[d,d]=1; w[d+a,d+b]=-1; kernels.append(w)
    nm='peak_span_'+str(d); comp=conv(nm+'_sides',smooth,np.array(kernels)[:,None],pad=d)
    for n in (4,2,1): comp=minimum_pair(nm+'_min'+str(n),comp,n)
    comp=rel(nm+'_prominence',comp)
    # Sub-pixel galaxy centers can put the brightest pixel on any of the
    # central four. For companions, suppress this inner region explicitly.
    masks=[np.exp(-.5*(r32/6)**2), (1-np.exp(-.5*(r32/9)**4))*np.exp(-.5*(r32/27)**2)]
    # Apply radial taper, then max-reduce. With the allowed primitives this
    # pointwise map is a band of a dense matrix. Crop to the region of use.
    ids=np.flatnonzero(r32.ravel()<48); mat=np.zeros((2*len(ids),1024),F)
    for j,ms in enumerate(masks): mat[np.arange(len(ids))+j*len(ids),ids]=ms.ravel()[ids]
    p=dense(nm+'_taper',comp,mat); p=reshape(nm+'_regions',p,2,1,len(ids)); p=pool(nm+'_strongest',p,'maxpool',(1,len(ids)),(1,len(ids)))
    peak.append((reg(p,0),reg(p,1)))

# 180-degree residual after modest smoothing. Small displacements relative to
# the survey center are allowed by calculating asymmetry about nine trial
# centers. Sparse +/- sampling matrices act on a central crop. Radial masks
# attenuate stars far from the galaxy. Measuring at several scales decouples
# tidal structure from isolated peripheral contaminants.
# Work on a 24x24 inner crop; include center shifts of half of a 4-pixel bin.
A=[]
for j,(oy,ox) in enumerate([(0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]):
    yy,xx=np.mgrid[4:28,4:28]; y2=31-yy+oy; x2=31-xx+ox
    ids1=(yy*32+xx).ravel(); ids2=(y2*32+x2).ravel(); keep=ids1<ids2
    i1=ids1[keep]; i2=ids2[keep]; n=len(i1); w=np.zeros((2*n,1024),F); row=np.arange(n)
    w[row,i1]=1; w[row,i2]=-1; w[n:]=-w[:n]
    ab=rel('asym_abs_'+str(j),dense('paired_pixels_'+str(j),smooth,w)); ar=[]
    for r in scales:
        # Multiply the half-plane integral by two: each antipodal pair occurs
        # once. Normalize by twice the flux in the aperture later (CAS/2).
        mask=unit(gauss(x32-ox*2,y32-oy*2,r)); v=(mask.ravel()[i1]+mask.ravel()[i2])*.5
        ar.append(np.tile(v,2))
    aa=dense('asym_apertures_'+str(j),ab,ar); A.append([reg(aa,i) for i in range(5)])
while len(A)>1:
    st=Stage('asym_best_of_'+str(len(A))); A=[[st.cap(x,y) for x,y in zip(a,b)] for a,b in zip(A[::2],A[1::2])]+(A[-1:] if len(A)%2 else []); st.end()
# Counting fluctuations in the black sky as arms would bias measurements for
# faint targets. Subtract a small contrast allowance for smoothed shot noise.
st=Stage('debiased_asymmetry'); ad=[st.pos(a-.002) for a in A[0]]; st.end()
ar=directions('fractional_asymmetry',[(a,reg(mm,6*j)+.006) for j,a in enumerate(ad)])
st=Stage('select_asymmetry'); asym=sum(product(st,g,a) for g,(a,b) in zip(gate,ar)); outerasym=sum(product(st,g,ar[min(j+1,4)][0]) for j,g in enumerate(gate)); st.end()

# Local departures from a smooth elliptical profile. At the correct axis
# ratio and position angle, adjacent sectors on an ellipse should agree in
# brightness. Sample a bank of inclined frames, selected by the independently
# measured quadrupole, and analyze azimuthal Fourier coefficients in the disk
# frame. This prevents an inclined but otherwise featureless ellipse from
# looking like a bar or spiral. Pixel binning is included in annular widths.
angles=8
ratios=[1.,.73,.50,.31]
# Use circularized radius sqrt(ab) so disk inclination leaves the aperture
# scale approximately invariant. Only even position harmonics are needed to
# measure a bar or a spiral. Use a coarse grid for the large annuli, fine
# grid in the inner disk. Compute circular m=1,3 separately above.
frames=[(ratios[0],0.)]+[(q,i*np.pi/angles) for q in ratios[1:] for i in range(angles)]
# Sampling radii for the disk decomposition. Estimate a local radial slope
# as well: even a slightly imperfect deprojection leaves a constant phase
# mode proportional to that slope. Twisting with radius is better evidence.
fradii=np.geomspace(3.2,40.,11)
frame_inputs=[]
for band,indices in enumerate((range(5),range(5,len(fradii)))):
    n=64 if band==0 else 32; src=white if band==0 else coarse
    xx,yy=grid(n); rows=[]
    for j in indices:
        r=fradii[j]
        for q,t in frames:
            u=(xx*np.cos(t)+yy*np.sin(t))*np.sqrt(q)
            v=(-xx*np.sin(t)+yy*np.cos(t))/np.sqrt(q)
            rad=np.hypot(u,v); az=np.arctan2(v,u)
            win=unit(np.exp(-.5*((rad-r)/max(1.05,.13*r))**2)); bg=sky(xx,yy,max(16,2*r))
            rows.append(win-bg)
            for m in [2,4]:
                for trig in [np.cos,np.sin]:
                    h=trig(m*az); rows.append(2*win*(h-(win*h).sum()))
    frame_inputs.append(dense('disk_frame_projections_'+str(band),src,rows))

# Radial covariance ellipse under a matched Gaussian window: when its trace
# is one (in window units), e=(1-q)/(1+q) for a Gaussian galaxy. Give the
# outer aperture additional weight, since a bulge dilutes the quadrupole.
# The inclination and position angle gates blend neighboring disk frames.
vec=directions('isophote_directions',[(q,u) for q,u,t in corr]);
st=Stage('disk_direction'); vv=[]
for a in range(2): vv.append(sum(product(st,g,vec[min(4,j+1)][a]) for j,g in enumerate(gate)))
st.end()
pa=directions('outer_position_angle',[tuple(vv)])[0]
eproj=.66*ell+.34*ellout
# The outer Gaussian gives less circularization. Convert its excess
# ellipticity back toward the matched-scale value.
st=Stage('inclination_knots'); cuts=[]
knots=[(1-q)/(1+q) for q in ratios]
for lo,hi in zip(knots[:-1],knots[1:]): cuts.append(st.clip((eproj-lo)/(hi-lo)))
angular=[]
for i in range(angles):
    t=2*np.pi*i/angles; dot=np.cos(t)*pa[0]+np.sin(t)*pa[1]; knots=np.cos(np.linspace(2*np.pi/angles,0,5)); slope=0.; hat=0.
    for a,b in zip(knots[:-1],knots[1:]):
        s=.25/(b-a); hat+=(s-slope)*st.pos(dot-a); slope=s
    angular.append(hat)
st.end()
# Convert the raised-cosine lobes into approximately linear azimuth weights.
# Adjacent lobes sum to >1 at the midpoint; compensate in the next norm.
iw=[a-b for a,b in zip([1]+cuts,cuts+[0])]
st=Stage('oriented_disk_gates'); fg=[iw[0]]+[product(st,w,a) for w in iw[1:] for a in angular]; st.end()
# Instead of multiplying a small flux in each of many directions, use a
# local-switch cap. In the interior of a frame bin it reproduces the frame
# exactly, while interpolating shared radial structure at boundaries. Use
# flux fractions first, so switch multiplication has a fixed dynamic range.
# To avoid thousands of redundant normalizers, normalize AFTER interpolating
# by using the polarization identity with fine knots for flux. A simpler
# exact gate decomposition is possible here: use the circular frame as the
# base, and correct by only the *difference* from each tilted frame. To limit
# computation use hard piecewise-linear gates and smoothly blend the 2 or 4
# neighboring measurements.
# Per-frame normalized coefficients expressed as fractions of ring light.
# Isotropic noise protection in the reference flux keeps sky-only rings from
# dominating. Add a floor proportional to the nuclear brightness as well.
allmom=[]
for j,r in enumerate(fradii):
    band=int(j>=5); src=frame_inputs[band]; basei=(j-5*band)*len(frames)*5
    for i in range(len(frames)):
        idx=basei+i*5; light=reg(src,idx)+.012+.025*reg(mm,0)
        allmom.extend((reg(src,idx+k),light) for k in range(1,5))
# Compact implementation of the block-diagonal normalization. Using a dense
# embedding for thousands of affine terms would waste mostly zero weights.
def ring_fractions():
    pieces=[]
    for j in range(len(fradii)):
        part=directions('disk_contrasts_'+str(j),allmom[j*len(frames)*4:(j+1)*len(frames)*4]); pieces.append([a for a,b in part])
    return pieces
frac=ring_fractions()
# A smooth AND for bounded signed contrasts. Only two or three directions
# contribute significantly. Rotate m=2 and 4 back to the sky frame before
# combining: the trigonometric phase in disk coordinates is measured from
# that frame's line of nodes, not from a common zero.
fourier=[]
for j,part in enumerate(frac):
    w=mixed_contrasts('select_disk_'+str(j),[a for a in fg for _ in range(4)],part); stuff=[]
    for i in range(len(frames)):
        a=w[4*i:4*i+4]; t=frames[i][1]; rot=[]
        for h,m in [(0,2),(2,4)]:
            c=np.cos(m*t); s=np.sin(m*t); rot.extend([a[h]*c-a[h+1]*s,a[h]*s+a[h+1]*c])
        stuff.append(rot)
    fourier.append(to_features('deprojected_modes_'+str(j),[sum(row[k] for row in stuff) for k in range(4)]))

# Ring phases and fractional contrasts. Extract twisting and pitch from
# real and imaginary parts of z(r2)*conjugate(z(r1)). These are invariant
# to image rotation; the absolute handedness is taken only after combining
# neighboring radii, so random alternation is not mistaken for winding.
st=Stage('arm_strengths'); amplitudes=[[mag(st,*v[:2]),mag(st,*v[2:])] for v in fourier]; st.end()
def mul(st,x,y,h=.125): return .25*(st.sq(x+y,h,round(1.5/h))-st.sq(x-y,h,round(1.5/h)))
st=Stage('radial_phase_transport'); crosses=[]; dots=[]
for j in range(len(fradii)-1):
    x1,y1=fourier[j][:2]; x2,y2=fourier[j+1][:2]
    crosses.append(mul(st,x1,y2)-mul(st,y1,x2)); dots.append(mul(st,x1,x2)+mul(st,y1,y2))
st.end()
# The typical disk scale lies between the inner arms and the outer arms.
# Weight a small set of log-radius bands around the adaptive size. Express
# all normalized measures in the same roughly [0,1] interval.
def radial_weights(rs,r,rel,width):
    a=np.exp(-.5*(np.log(np.asarray(rs)/(r*rel))/width)**2); return a/a.sum()
st=Stage('structure_at_scale'); de=[]
for j,r in enumerate(scales):
    arm=radial_weights(fradii,r,1.9,.50); bar=radial_weights(fradii,r,.83,.40); transition=radial_weights(np.sqrt(np.array(fradii[:-1])*fradii[1:]),r,1.9,.50)
    wind=sum(t*w for t,w in zip(crosses,transition)); align=sum(t*w for t,w in zip(dots,transition))
    # Inner-to-outer change: a bar is a coherent high m=2 feature with a
    # fairly abrupt end; unbarred spirals twist and/or show a weaker core.
    bv=[sum(v[k]*w for v,w in zip(fourier,bar)) for k in range(2)]
    ov=[sum(v[k]*w for v,w in zip(fourier,arm)) for k in range(2)]
    de.append([sum(v[0]*w for v,w in zip(amplitudes,arm)),sum(v[1]*w for v,w in zip(amplitudes,arm)),mag(st,*bv),mag(st,bv[0]-ov[0],bv[1]-ov[1]),st.ab(wind),align,
               sum(st.ab(t)*w for t,w in zip(crosses,transition))])
st.end(); st=Stage('select_disk_structure'); arm,a4,bar,barchange,twist,align,anytwist=[sum(product(st,g,d[k]) for g,d in zip(gate,de)) for k in range(7)]; st.end()

# Demodulate the bisymmetric phase by both handednesses of logarithmic
# spirals. For pitch p, the Fourier phase advances at 2/tan(p) per e-fold
# in radius. A bar or an incorrectly deprojected ellipse has stationary
# phase, so remove a locally constant complex mode from each analyzing wave.
pitches=[12.,18.,25.,34.,46.]
st=Stage('logarithmic_spiral_correlations'); responses=[]
for j,r in enumerate(scales):
    w=radial_weights(fradii,r,1.9,.56); row=[]
    for p in pitches:
        for sign in (1,-1):
            phase=sign*2*np.log(fradii/r)/np.tan(np.deg2rad(p)); h=np.exp(-1j*phase); h=w*(h-np.dot(w,h))
            a=sum(hn.real*v[0]-hn.imag*v[1] for hn,v in zip(h,fourier)); b=sum(hn.imag*v[0]+hn.real*v[1] for hn,v in zip(h,fourier)); row.append(mag(st,a,b))
    responses.append(row)
st.end(); responses=[to_features('spiral_pitches_at_scale_'+str(j),v) for j,v in enumerate(responses)]
st=Stage('spiral_handedness_pool'); handed=[[st.mx(v[i],v[i+1]) for i in range(0,len(v),2)] for v in responses]; st.end()
st=Stage('spiral_pitch_pool'); bypitch=[(st.mx(v[0],v[1]),st.mx(v[3],v[4]),v[2]) for v in handed]; st.end()
st=Stage('pick_spiral_aperture'); tightmatch,loosematch,midmatch=[sum(product(st,g,v[k]) for g,v in zip(gate,bypitch)) for k in range(3)]; st.end()

# Circular odd modes: most useful in the outer optical galaxy. A bright
# one-sided tail and a second galaxy both contribute; peaks disambiguate.
oddpairs=[]
for i in range(len(radii)):
    for k in [1,2,5,6]: oddpairs.append((reg(pr,9*i+k),reg(pr,9*i)+.018))
oddfrac=directions('odd_mode_contrasts',oddpairs)
st=Stage('odd_mode_amplitudes'); odd=[]
for i in range(len(radii)):
    v=[oddfrac[4*i+j][0] for j in range(4)]; odd.append((mag(st,*v[:2]),mag(st,*v[2:])))
st.end(); st=Stage('select_odd_modes'); lop,knot=[sum(product(st,g,sum(v[k]*w for v,w in zip(odd,radial_weights(radii,r,1.8,.40)))) for r,g in zip(scales,gate)) for k in range(2)]; st.end()

# Surface brightness concentration and color help distinguish a smooth cigar
# from a dusty stellar disk, and a bulge from a flat or nearly uniform disk.
# Inner versus outer ellipticity gives another, color-independent bulge test.
cp=[]
for r in scales:
    outer=unit(gauss(x,y,r)); inner=unit(gauss(x,y,max(2.,r*.36))); bk=sky(x,y,max(15,2*r));
    cp.extend([inner-bk,outer-bk,unit(gauss(x,y,max(1.25,r*.16)))-bk]);
ct=dense('inner_outer_brightness',white,cp)
# Positive for centrally peaked profiles. Ratio inner-average / outer-average
# ~1.75 for a Gaussian evaluated at its matched scale; larger for a cusp.
zs=directions('concentration_index',[(reg(ct,3*i)-reg(ct,3*i+1),reg(ct,3*i)+reg(ct,3*i+1)+.008) for i in range(5)])
zc=directions('nuclear_cusp_index',[(reg(ct,3*i+2)-reg(ct,3*i),reg(ct,3*i+2)+reg(ct,3*i)+.008) for i in range(5)])
st=Stage('select_concentration'); concentration=sum(product(st,g,a) for g,(a,b) in zip(gate,zs)); cusp=sum(product(st,g,a) for g,(a,b) in zip(gate,zc)); st.end()
ps=directions('companion_contrasts',[(b,.04+a*.7+.14*reg(mm,0)) for a,b in peak]);
# Keep a relative-to-host test as well, since a single barely resolved star
# can be prominent on dark sky but contribute negligible light to the galaxy.
pp=directions('companion_to_host',[(b,.035+.40*reg(mm,0)) for a,b in peak]);

# Compact normalized readout. The hand-set boundaries are morphology rules,
# not parameters selected by running images. Use continuous, overlapping
# evidence so no single noisy threshold switches the entire classification.
keys=['ell','ellout','ellin','size','red','asym','outerasym','arm','a4','bar','barchange','twist','align','anytwist','lop','knot','concentration','peak','peakhost']
vals=[ell,ellout,ellin,sz,red,asym,outerasym,arm,a4,bar,barchange,twist,align,anytwist,lop,knot,concentration,.4*ps[0][0]+.6*ps[1][0],.4*pp[0][0]+.6*pp[1][0]]
keys+=['tight','loose','midpitch','cusp']; vals += [tightmatch,loosematch,midmatch,cusp]
features=dict(zip(keys,to_features('galaxy_observables',vals)))
f=features
st=Stage('morphology_evidence'); S=st.s
# Map measured quadrupoles to a thinness scale. For Gaussian disks under
# a matched window ell=(1-q)/(1+q); an outer window gives a value closer
# to (1-q^2)/(1+q^2). Use both, with their different circularization.
e=.70*f['ell']+.30*f['ellout']; flat=S(e,.57,.11); thin=S(e,.66,.09)
# Very small galaxies are often PSF dominated. Scale is used as a weak
# prior, never alone to claim an interaction or a spiral pattern.
big=S(f['size'],3.45,.64); resolved=S(f['size'],2.72,.48)
redness=S(f['red'],.125,.105); blue=1-redness
oddness=S(.66*f['asym']+.34*f['outerasym'],.125,.083); severe=S(.65*f['asym']+.35*f['outerasym'],.245,.11)
lopsided=S(f['lop'],.28,.17); patchy=S(f['knot'],.125,.09)
peakflag=S(f['peak'],.215,.15); hostflag=S(f['peakhost'],.19,.13)
twoarm=S(f['arm'],.155,.095); harmonic=S(f['a4'],.095,.060); turn=S(f['anytwist'],.017,.014)
barcore=S(f['bar'],.19,.095); barend=S(f['barchange'],.19,.105)
strongbar=S(f['bar'],.31,.10); twistabs=S(f['twist'],.018,.014)
# cos(2 delta PA) is positive for a slowly changing (loosely wound) pattern;
# a tight spiral passes through a large phase angle in a single radial step.
openphase=S(f['align']-.48*f['anytwist'],.002,.014)
pitchpref=S(f['loose']-f['tight'],.005,.052)
spiralwave=S(.34*f['tight']+.38*f['loose']+.28*f['midpitch'],.065,.045)
compact=S(f['concentration'],.325,.075); thickcore=S(f['ellout']-f['ellin'],.265,.13)
nuclear=S(f['cusp'],.092,.048); smallbulge=S(f['concentration'],.305,.065); flatcore=S(f['ellin'],.40,.13)
roundcut=S(e,.108,.053); cigarcut=S(e,.445,.085); st.end()

st=Stage('joint_morphology');
companion=st.cap(peakflag,hostflag)
# Arm contrast is stronger evidence when the residual actually changes in
# azimuth with radius, not merely when a smooth disk is highly inclined.
pattern=.19*twoarm+.18*harmonic+.15*turn+.12*patchy+.10*barend+.26*spiralwave
structured=st.clip((.62*pattern+.18*blue+.13*big+.07*resolved-.32)/.38)
barred=st.clip((.38*barcore+.27*barend+.16*strongbar+.11*twoarm+.08*big-.35)/.41)
disturbed=st.clip((.51*oddness+.19*severe+.15*lopsided+.15*patchy-.36)/.45)
bulge=.24*smallbulge+.20*thickcore+.19*redness+.11*(1-flatcore)+.06*resolved+.20*nuclear
# Flatness is the primary edge-on discriminator. Flat blue, extended
# systems need less flattening than smooth red objects without a disk.
edge=st.s(e+.062*blue+.040*big+.033*thickcore,.62,.090)
st.end()
st=Stage('galaxy_family_scores');
interact=st.clip(.64*disturbed+.26*companion+.14*lopsided+.09*severe-.10)
merger=st.clip(.49*companion+.29*disturbed+.12*hostflag+.10*severe)
# Smooth, spiral, edge-on and peculiar are competing families; subclasses
# share most of their score. Morphological evidence overrides the weak
# preference for spiral arms in bluer, better resolved targets.
spiral=st.clip(.76*structured+.13*twoarm+.10*big+.08*blue-.22*flat-.03)
smoothness=st.clip(1-.85*structured-.11*big-.06*blue+.13*flat)
barprob=st.clip(barred-.23*flat-.08*severe)
# Barred spirals have coherent inner bars; for unbarred spirals, more open
# arms tend to be clumpier, less concentrated and to twist less per annulus.
loose=st.clip(.43*pitchpref+.17*openphase+.08*twoarm+.10*blue+.09*patchy+.09*(1-compact)+.04*oddness)
st.end()
# Set each winning subclass's adjustment close to zero; a doubtful subclass
# shouldn't by itself rule out the parent family.
st=Stage('subclass_membership');
mid=st.cap(roundcut,1-cigarcut); bulgy=st.s(bulge,.52,.13); loosecut=st.s(loose,.54,.11)
strongmerge=st.cap(companion,disturbed); odd_no_pair=st.pos(disturbed-.53*companion)
st.end()
sm=.79*smoothness-.52*edge-.48*interact+.035
sp=.81*spiral-.52*edge-.39*interact+.02
ed=.98*edge-.21*disturbed-.04
pe=.88*interact+.12*severe-.12*edge
scores=[pe+.17*odd_no_pair-.28*companion-.035,
        .61*merger+.31*strongmerge+.19*interact-.13*edge-.085,
        sm-.39*roundcut,sm-.39*(1-mid),sm-.39*(1-cigarcut),
        sp+.22*(barprob-.5),sp-.16*barprob+.11*(.5-loosecut),sp-.16*barprob+.11*(loosecut-.5),
        ed+.15*(.5-bulgy),ed+.15*(bulgy-.5)]
emit('class_scores',scores)

def write(out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n')
    np.savez(out/'weights.npz',**weights)
if __name__=='__main__': write(Path(sys.argv[1]))
