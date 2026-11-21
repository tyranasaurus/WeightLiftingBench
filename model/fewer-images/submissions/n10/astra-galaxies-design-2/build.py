"""Hand-designed photometry graph; no file inputs."""
from pathlib import Path
import sys, json, math
import numpy as np

layers=[]; weights={}; counter=0

def op(opname,ins,name=None,**kw):
    global counter
    counter+=1
    name=name or ('%s_%03d'%(opname,counter))
    layers.append(dict(name=name,op=opname,inputs=[ins] if isinstance(ins,str) else ins,**kw))
    return name

def aff(ins,W,b=0,name=None):
    W=np.asarray(W,dtype='f'); n=W.shape[0]
    s=op('dense',ins,name,out=n)
    weights[s+'.W']=W.reshape(n,-1); weights[s+'.b']=np.broadcast_to(b,(n,)).astype('f')
    return s

def conv(ins,W,b=0,name=None):
    W=np.asarray(W,dtype='f'); n,_,k,_=W.shape
    s=op('conv',ins,name,kernel=[k,k],stride=1,padding=k//2,out=n)
    weights[s+'.W']=W; weights[s+'.b']=np.broadcast_to(b,(n,)).astype('f'); return s

def rect(x):return op('relu',x)
def pool(x,k,kind='maxpool'):
    return op(kind,x,size=list(k),stride=list(k))
def groupmax(x,n,k):return pool(op('reshape',x,shape=[n,1,k]),(1,k))
def grid(n,step=1):
    a=(np.arange(n)-(n-1)/2)*step
    return np.meshgrid(a,a)
def gauss(x,y,s):
    w=np.exp(-(x*x+y*y)/(2*s*s));return w/w.sum()

def search_sectors(src,masks):
    # Reuse the 64-way spatial max per coordinate subset. Matrices stay small
    # because disjoint narrow rings require only a sparse list of destinations.
    vals=[]
    for m in masks:
        ids=np.flatnonzero(m);w=np.zeros((len(ids),4096),'f');w[np.arange(len(ids)),ids]=1
        vals.append((groupmax(aff(src,w),1,len(ids)),1))
    return vals

def merge_columns(vecs,name):
    # Concatenate branches in disjoint coordinates using add.
    n=sum(k for _,k in vecs); ins=[];i=0
    for src,k in vecs:
        W=np.zeros((n,k),'f');W[i:i+k,:]=np.eye(k);ins.append(aff(src,W));i+=k
    return op('add',ins,name)

rgb=pool('image',(2,2),'avgpool')
y,x=np.mgrid[:64,:64]; x=(x-31.5)*2;y=(y-31.5)*2
rc=np.hypot(x,y); sky=np.exp(-((rc-51)/9)**2);sky/=sky.sum()
lum=np.array([.26,.45,.29]);gg=conv(rgb,lum.reshape(1,3,1,1),name='luminance')
# Estimate sky per image and remove it. A soft threshold reduces contributions from sky noise.
minus=aff(gg,[-sky],-.012,'sky_subtraction')
bmap=op('reshape',aff(minus,np.ones((64*64,1))),shape=[1,64,64])
signal=rect(op('add',[gg,bmap],'above_sky'))
# For fitting smooth shapes, slightly saturate at the bright end: very sharp nuclei and
# foreground stars should not dominate the importance of the extended isophotes.
cap=rect(conv(signal,np.ones((1,1,1,1)),-.53)); capped=op('add',[signal,conv(cap,-np.ones((1,1,1,1)))])
# Directly index a central stamp with a (fixed routing) dense map, then apply centering mask.
# Four times narrower in area than evaluating every template on the entire CCD.
ix=np.arange(64*64).reshape(64,64)[14:50,14:50].ravel();route=np.zeros((ix.size,64**2),'f');route[np.arange(ix.size),ix]=1
stamp=aff(capped,route,name='central_stamp');del route
X,Y=grid(36,2); r=np.hypot(X,Y)
ap=np.exp(-.5*(r/24)**2)  # fixed circular sky/confusion taper, applied to BOTH sides of correlation
phi=np.arange(16)*np.pi/16
scales=np.array([2.6,3.6,5,7,9.8,13.7,19.2])
qs=np.array([1,.84,.70,.57,.46,.36,.27,.20,.14])
templates=[]; quality=[]
for s in scales:
 for q in qs:
  for a in phi:
   U=X*np.cos(a)+Y*np.sin(a);V=Y*np.cos(a)-X*np.sin(a)
   rho=np.sqrt((U/s)**2+V**2/((s*q)**2+.9**2))
   t=np.exp(-np.sqrt(rho*rho+.28**2));den=np.sqrt((ap*t*t).sum());w=ap*t/den
   templates.append(w);quality.append([s,q,a])
sims=aff(stamp,np.stack(templates),name='ellipse_correlations');del templates
quality=np.asarray(quality)
# Select model parameters by maximizing correlation with only a small linear tilt.
# The finite difference of the support function gives the parameter of the best
# ellipse, smoothly interpolating between nearly indistinguishable fits.
param=[];paramnames=[]
for name,p,eps in [('q',quality[:,1],.035),('log_s',np.log2(quality[:,0]/7),.014)]:
 for sign in [1,-1]:
  pp=p*sign
  k=len(pp); t=aff(sims,np.diag(1+eps*pp).astype('f')); t=groupmax(t,1,k);param.append((t,1))
 paramnames.extend([name+'+',name+'-'])
bestsim=groupmax(sims,1,len(quality));param.append((bestsim,1));paramnames.append('sim')
# The residual of the best elliptical model is ||image||^2 - max(correlation)^2.
# Evaluate the norm with a piecewise linear approximation to squared pixel intensity.
knots=np.r_[np.linspace(0,.10,9),np.linspace(.125,.55,18),.60,.80,1.];slopes=knots[1:]+knots[:-1]; co=np.diff(np.r_[0,slopes]);
for name,signal_in in [('energy',capped)]:
 sq=rect(conv(signal_in,np.ones((len(co),1,1,1)),-knots[:-1]));sq=conv(sq,co.reshape(1,-1,1,1))
 mask=np.zeros((64,64));mask[14:50,14:50]=ap
 en=aff(sq,[mask]);param.append((en,1));paramnames.append(name)
# Multi-aperture photometry, color, and low-order Fourier modes of the isophotes.
# All aperture lengths below are expressed in input pixels, not binned pixels.
features=[];names=[]
def feature(src,keys):
 features.append((src,len(keys)));names.extend(keys)
feature(merge_columns(param,'model_measurements'),paramnames)
# Smooth-null test with alternative radial profiles. Evaluate in a more local
# aperture and subtract its DC component: a distant bright source should have
# little leverage in deciding whether the central galaxy is featureless.
# Use strided, binned photometry for this complementary, lower-resolution test.
xa,ya=grid(16,4); a2=np.exp(-.5*(np.hypot(xa,ya)/15)**2); aa=a2.sum()
loc=pool(capped,(2,2),'avgpool'); idx=np.arange(1024).reshape(32,32)[8:24,8:24].ravel()
wm=np.zeros((len(idx),1024),'f');wm[np.arange(len(idx)),idx]=1;local=aff(loc,wm)
def model(t):
 t=t-(a2*t).sum()/aa
 return a2*t/np.sqrt((a2*t*t).sum())
W=[]
for s in (3,4.5,6.75,10,15,22):
 for q0 in (1,.78,.59,.43,.29,.17):
  for angle in np.arange(12)*np.pi/12:
   u=xa*np.cos(angle)+ya*np.sin(angle);v=ya*np.cos(angle)-xa*np.sin(angle)
   rr=np.sqrt((u/s)**2+v**2/((s*q0)**2+1.3**2))
   for power in (.65,1.,1.65,2.5):
    W.append(model(np.exp(-(rr*rr+.3**2)**(power/2))))
feature(groupmax(aff(local,W,name='local_spheroids'),1,len(W)),['smoothsim'])
# A likelihood-ratio test between hand-generated classes of surface brightness
# distributions. The same aperture, mean removal and unit norm are used on every
# INDIVIDUAL template; no results or statistics of template evaluations are used.
def bank(key,gen):
 w=np.asarray([model(t) for t in gen],dtype='f')
 feature(groupmax(aff(local,w,name=key+'_bank'),1,len(w)),[key])
def galaxy(u,v,s,q,power=1):
 return np.exp(-(((u/s)**2+v*v/((s*q)**2+1.4**2)+.12)**(.5*power)))
def pair_models():
 for a in np.arange(24)*2*np.pi/24:
  u=xa*np.cos(a)+ya*np.sin(a);v=ya*np.cos(a)-xa*np.sin(a)
  for sep in (7,11,17,25,33):
   for s in (3.,5.5,9.):
    for l in (0.,.32,.50):
     for tilt in (0,np.pi/3):
      t=u+l*sep;c,sn=np.cos(tilt),np.sin(tilt)
      primary=galaxy(t*c+v*sn,v*c-t*sn,s,.70)
      secondary=galaxy(t-sep,v,.8*s,.85)
      for ratio in (.40,.85,1.40):yield primary+ratio*secondary
bank('double_core',pair_models())
# In a fair comparison with a pair hypothesis, allow a single nucleus to be
# displaced a little from the camera center as well. These inexpensive low-res
# tests implement the nuisance translation for an isolated spheroid.
def shifted_models():
 for a in np.arange(12)*np.pi/12:
  u=xa*np.cos(a)+ya*np.sin(a);v=ya*np.cos(a)-xa*np.sin(a)
  for s in (3,4.7,7.5,12,18):
   for q in (1.,.70,.45,.26):
    for off in np.arange(8)*np.pi/4:
     for mag in (1.3,3.):yield galaxy(u-mag*np.cos(off),v-mag*np.sin(off),s,q,1.2)
bank('offset_core',shifted_models())
# Deliberately one-sided structures, including bent tails, and an off-centered
# outer envelope. The core stays at the photometric origin.
def tidal_models():
 for a in np.arange(16)*2*np.pi/16:
  u=xa*np.cos(a)+ya*np.sin(a);v=ya*np.cos(a)-xa*np.sin(a)
  for s in (4,7.5,13):
   for q in (.35,.7,1.):
    core=galaxy(u,v,s,q)
    for frac in (.35,.65):
     for bend in (-1.,0.,1.):
      along=np.maximum(u/s,0);cross=v/s-bend*.20*along**1.7
      arm=np.exp(-((along-1.6)/1.3)**2/2-(cross/(.36+.22*q))**2/2)*np.clip(along,0,1)
      yield core+frac*arm
     yield core+frac*galaxy(u-s,v-.35*s,s*1.45,q*.85)
bank('tidal',tidal_models())
# Straight bar + disk; arm models on inclined disks for both spiral chiralities.
def disk_models(kind):
 for angle in np.arange(8)*np.pi/8:
  u=xa*np.cos(angle)+ya*np.sin(angle);v=ya*np.cos(angle)-xa*np.sin(angle)
  for s in (4,6.5,10,15):
   for q in (.48,.72,.96):
    U=u/s;V=v/(s*q);rad=np.hypot(U,V);ang=np.arctan2(V,U)
    disk=np.exp(-np.sqrt(rad*rad+.12)); bul=np.exp(-np.hypot(xa,ya)/(s*.38))
    for p in np.arange(8)*np.pi/8:
     if kind=='bar':
      xbar=U*np.cos(p)+V*np.sin(p);ybar=V*np.cos(p)-U*np.sin(p)
      for blen in (.85,1.45):
       bar=np.exp(-.5*((xbar/blen)**4+(ybar/(.25*blen+.10))**2))
       for amp in (.33,.75):yield disk+.22*bul+amp*bar
     else:
      pitch=np.deg2rad(16 if kind=='tight' else 34)
      for sign in (-1.,1.):
       phase=ang-p-sign*np.log(np.maximum(rad,.25))/np.tan(pitch)
       # Suppress unresolved windings and fade the arm out at the origin.
       strength=np.clip((rad-.35)/.65,0,1)*np.exp(-.5*(3/(np.maximum(rad,.25)*s*np.sin(pitch)))**2)
       for amp in (.45,.9):yield disk*(1+amp*strength*np.cos(2*phase))+.32*bul
for kind in ('bar','tight','loose'):bank('disk_'+kind,disk_models(kind))
# Edge-on disks with, and without, a separately resolved central spheroid.
def edge_models(with_bulge):
 for a in np.arange(32)*np.pi/32:
  u=xa*np.cos(a)+ya*np.sin(a);v=ya*np.cos(a)-xa*np.sin(a)
  for s in (6.,9.,14.,21.):
   for q in (.12,.20,.28):
    d=galaxy(u,v,s,q,1.4)
    if with_bulge:
     for amp in (.6,1.3,2.5):
      for bs in (.29,.50):yield d+amp*galaxy(u,v,s*bs,.84,1.15)
    else:yield d
bank('edgebulge',edge_models(True));bank('edgedisk',edge_models(False))
feature(aff(local,[a2/np.sqrt(aa)]),['localDC'])
# Sum of squares of binned pixels in the same aperture, for variance explained.
sq=op('reshape',local,shape=[1,16,16]);sq=rect(conv(sq,np.ones((len(co),1,1,1)),-knots[:-1]));sq=conv(sq,co.reshape(1,-1,1,1))
feature(aff(sq,[a2]),['localenergy'])
W=[];wn=[]
def phot(key,w):W.append(w);wn.append(key)
for s in [2,4,8,16,24]:phot('f'+str(s),gauss(x,y,s))
# Independent angular structure at multiple radii. Annular average has intensity units.
# Real and imaginary parts are retained, allowing measurement of position-angle twist.
theta=np.arctan2(y,x); rings=[4,7,11,17,25]
for i,rr in enumerate(rings):
 mask=np.exp(-.5*((rc-rr)/(.22*rr+.65))**2);mask/=mask.sum()
 phot('I'+str(i),mask)
 for m in (1,2,3,4):
  phot(f'C{i}:{m}',mask*np.cos(m*theta)*2);phot(f'S{i}:{m}',mask*np.sin(m*theta)*2)
feature(aff(signal,W,name='aperture_photometry'),wn)
# Cross sections of the core and disk measured at the inferred position angle.
# Weight lines by angular quadrupole rather than by the brightness of a single
# pixel. For an edge-on galaxy the outermost bright strip is aligned with its
# major axis; subtracting a wider perpendicular strip cancels the diffuse sky.
W=[];wn=[]
for ia,a in enumerate(np.arange(16)*np.pi/16):
 u=x*np.cos(a)+y*np.sin(a);v=y*np.cos(a)-x*np.sin(a)
 def mask(u0,v0,su,sv):
  t=np.exp(-.5*((np.abs(u)-u0)/su)**2-.5*((np.abs(v)-v0)/sv)**2);return t/t.sum()
 for name,w in [('strip',mask(17,0,7,2)-mask(17,9,7,3)),('mid',mask(12,0,4,2)),('wing',mask(12,4,4,1.8)),('pole',mask(0,4,2.5,1.8)),('eq',mask(4,0,1.8,2.5)),('length',mask(23,0,7,1.8))]:
  phot(name+str(ia),w)
feature(aff(signal,W,name='axial_sections'),wn)
W=[];wn=[]
for s in [3,8,15]:
 w=gauss(x,y,s)-sky
 for c in (0,1,2):
  v=np.zeros((3,64,64));v[c]=w;phot('rgb%d_%d'%(s,c),v)
feature(aff(rgb,W,name='colors'),wn)
# Color-dependent quadrupoles. A population of old, red stars in a bar can be
# much more elongated than the younger stars in the surrounding spiral disk.
W=[];wn=[]
for rr in [6,10,16]:
 ring=np.exp(-.5*((rc-rr)/(.29*rr))**2);ring/=ring.sum()
 for chan,c in [('r',0),('b',2)]:
  for lbl,mask in [('I',ring-sky),('C',ring*np.cos(2*theta)*2),('S',ring*np.sin(2*theta)*2)]:
   v=np.zeros((3,64,64));v[c]=mask;phot(chan+lbl+str(rr),v)
feature(aff(rgb,W,name='chromatic_quadrupoles'),wn)
# Detect an old red bar embedded in a younger blue disk, as distinct from a
# red, round core. Contrast the disk and core light in a fixed circular aperture
# and compare their best one-component ellipse to a two-component disk + bar.
# Smooth spheroids are included in the null over a range of concentrations.
W=[]; null=[]; sph=[];alternative=[]
wa=np.exp(-.5*(r/11)**2);sa=wa.sum()
def cmode(t):
 t=t-(t*wa).sum()/sa;return wa*t/np.sqrt((wa*t*t).sum())
for a in np.arange(12)*np.pi/12:
 u=X*np.cos(a)+Y*np.sin(a);v=Y*np.cos(a)-X*np.sin(a)
 for s in (2.8,4.3,6.5,10,15):
  for q0 in (1.,.80,.61,.43,.28):
   for power in (.65,1.,1.6):sph.append(cmode(galaxy(u,v,s,q0,power)))
  for qdisc in (.65,.9):
   disk=galaxy(u,v,s,qdisc); nucleus=galaxy(u,v,.38*s,1.)
   for p in np.arange(6)*np.pi/6:
    ub=(u*np.cos(p)+v/qdisc*np.sin(p))/s; vb=(v/qdisc*np.cos(p)-u*np.sin(p))/s
    b=np.exp(-.5*((ub/1.15)**4+(vb/.33)**2))
    for amp in (.4,.8):alternative.append(cmode(disk+amp*b+.30*nucleus))
# Color-enhanced image: a disk with the background color of young stars is
# partly cancelled, while the old bar light survives. No example colors used.
wc=conv(rgb,np.array([.80,.46,-.26]).reshape(1,3,1,1));idx=np.arange(4096).reshape(64,64)[14:50,14:50].ravel()
router=np.zeros((len(idx),4096),'f');router[np.arange(len(idx)),idx]=1;redstamp=aff(wc,router)
for label,ws in [('redbar',alternative),('rednull',sph)]:
 feature(groupmax(aff(redstamp,ws,name=label+'_shapes'),1,len(ws)),[label])
# Matched, radial tangential-curvature tests for the outer disk. Correlate the
# position angle of the quadrupole with a logarithmic winding law. Tightly and
# loosely wound arms differ in d(theta)/d(log radius), independent of overall
# rotation and multiplicative image size. Suppress a stationary inclined disk
# by subtracting the radial mean separately along each ray (analytic basis).
angles=np.arange(24)*2*np.pi/24
for si,s in enumerate([6,10,16]):
 ws=[];keys=[]
 for pitch in [14,22,34,50]:
  for sign in [-1,1]:
   for m in [2,3]:
    # Sample smooth, local Gaussian apertures along rays; the radial derivative
    # removes a non-winding angular modulation before coherently summing it.
    cc=np.zeros((36,36),complex)
    for angle in angles:
     for h in range(7):
      rad=s*(.60*1.28**h);width=max(1.5,.16*rad)
      mask=gauss(X-rad*np.cos(angle),Y-rad*np.sin(angle),width)
      ph=m*(angle-sign*np.log(rad/s)/np.tan(np.deg2rad(pitch)))
      # Suppress DC and fixed-angle elongation with a radial bandpass phase.
      phs=m*(angle-sign*np.log(.60*1.28**np.arange(7))/np.tan(np.deg2rad(pitch)))
      coeff=np.exp(1j*ph)-np.exp(1j*phs).mean()
      cc+=coeff*mask/(7*len(angles))
    ws.extend([cc.real,cc.imag]);keys.extend([f'wind{si}_{pitch}_{sign}_{m}_{d}' for d in 'cs'])
 feature(aff(stamp,ws,name='winding_scale'+str(si)),keys)
# Compact companion finder: a bandpass Mexican hat followed by an annular search.
# A shoulder on a monotone disk shouldn't count as a second local peak, so estimate
# the smaller of the contrasts to neighbors on the four cardinal axes (and diagonals).
xk,yk=grid(15,2)
K=[]
for sig,step in [(1.8,5.0),(3.,8.)]:
 center=gauss(xk,yk,sig); outer=gauss(xk,yk,3.0*sig)
 K.append((center-outer)[None]);
 for ang in np.arange(8)*np.pi/4:
  nbr=gauss(xk-step*np.cos(ang),yk-step*np.sin(ang),sig)
  K.append((center-nbr)[None])
peak=conv(gg,K,name='peak_comparison')
# Crop/stride by using fixed sample rows in dense readouts instead of large conv stacks.
# The minimum of oriented differences (slightly relaxed) rejects ridges on disks.
# Reorder: channels first in feature map, pool along orientation for pointwise max.
# pooling the negatives of all contrasts computes min; scale the symmetric contrast
# to ensure one small directional difference is tolerated at pixel sampling accuracy.
for j in range(2):
 T=np.zeros((8,18));T[:,9*j+1:9*j+9]=-np.eye(8)
 c=conv(peak,T.reshape(8,18,1,1));c=op('reshape',c,shape=[1,8,4096]);c=pool(c,(8,1));c=op('reshape',c,shape=[1,64,64]);
 T=np.zeros((1,18,1,1));T[0,9*j]=.18
 pk=op('add',[conv(peak,T),conv(c,-np.ones((1,1,1,1))*.82)])
 # direct routing selects a central circle and two search annuli
 for lab,lo,hi in [('nuc',0,4.5),('pair',5.5,15),('far',15,32)]:
  feature(search_sectors(pk,[(rc>=lo)&(rc<hi)])[0][0],[lab+str(j)])
# Divide the central field into paired opposite sectors. Compare a secondary
# peak with the particular peak reflected through the core, rather than just
# the strongest object anywhere. The symmetric part of a disk largely cancels.
for typ in ['knot','patch']:
 if typ=='knot':
  w=np.zeros((1,18,1,1));w[0,9]=1;pmap=conv(peak,w)
 else:pmap=conv(signal,gauss(xk,yk,3.0)[None,None])
 vals=[]
 for zone,(lo,hi) in enumerate([(6,12),(12,22),(22,34)]):
  for sect in range(8):
   angle=sect*np.pi/4
   mask=(rc>=lo)&(rc<hi)&(np.cos(theta-angle)>np.cos(np.pi/7))
   vals+=search_sectors(pmap,[mask])
 vec=merge_columns(vals,typ+'_sectors');T=np.eye(24);opp=np.arange(24).reshape(3,8);T-=np.eye(24)[np.roll(opp,4,axis=1).ravel()]
 feature(groupmax(aff(vec,T),3,8),[typ+str(z) for z in range(3)])
# A second test for paired nuclei. Contrast the candidate to its immediate
# neighborhood on all sides, and also require its light to be extended (the
# broad PSF-filter response shouldn't vanish relative to the narrow one).
# Both conditions must hold at the SAME position before spatial pooling.
ke,le=grid(15,2); g1=gauss(ke,le,1.35);g3=gauss(ke,le,3.5);g7=gauss(ke,le,7.)
multi=conv(gg,np.stack([g1-g7,g3-g7])[:,None]);extsrc=rect(conv(multi,np.array([-.40,1.]).reshape(1,2,1,1)))
w=np.zeros((1,18,1,1));w[0,9]=1;sharp=conv(peak,w);extent=conv(extsrc,np.ones((1,1,1,1))*3.5)
small=op('add',[sharp,conv(rect(op('add',[sharp,conv(extent,-np.ones((1,1,1,1)))])),-np.ones((1,1,1,1)))])
# min of the extended-source response and the actual local prominence.
small=op('add',[pk,conv(rect(op('add',[pk,conv(small,-np.ones((1,1,1,1)))])),-np.ones((1,1,1,1)))])
for k,(lo,hi) in enumerate([(5.6,12),(12,24),(24,36)]):
 feature(search_sectors(small,[(rc>lo)&(rc<=hi)])[0][0],['extendedpeak'+str(k)])
# A 180 degree asymmetry index as absolute bilateral differences. Central weighting
# reduces pollution by stars and neighboring galaxies far from the object.
xx,yy=grid(32,4);half=pool(signal,(2,2),'avgpool');ix=np.arange(1024);opp=1023-ix
A=np.zeros((1024,1024),'f');A[ix,ix]=1;A[ix,opp]=-1
v=rect(aff(half,A)); masks=[gauss(xx,yy,s)*2 for s in (5,10,18)]
feature(aff(v,masks),['as5','as10','as18'])

# A small arithmetic compiler: linear forms stay fused, ReLU breakpoints produce
# dense layers. No training or coefficient estimation. Only expressions participating
# in the final score are compiled.
class E:
 def __init__(self,terms={},b=0):self.t=terms;self.b=b
 def __add__(self,v):
  v=asE(v);t=self.t.copy()
  for k,x in v.t.items():t[k]=t.get(k,0)+x
  return E({k:x for k,x in t.items() if abs(x)>1e-12},self.b+v.b)
 __radd__=__add__
 def __neg__(self):return self*(-1)
 def __sub__(self,v):return self+(-asE(v))
 def __rsub__(self,v):return asE(v)+(-self)
 def __mul__(self,k):return E({n:x*k for n,x in self.t.items()},self.b*k)
 __rmul__=__mul__
 def __truediv__(self,k):return self*(1/k)

def asE(x):return x if isinstance(x,E) else E({},x)
raw=merge_columns(features,'observables');ids=len(names);defs={};D={k:E({i:1}) for i,k in enumerate(names)}
rect_cache={}
def R(x):
 global ids
 x=asE(x);key=(tuple(sorted(x.t.items())),x.b)
 if key not in rect_cache:
  defs[ids]=x;rect_cache[key]=E({ids:1});ids+=1
 return rect_cache[key]
def hi(x,y):return R(x-y)+y
def lo(x,y):return x-R(x-y)
def ramp(x,a=0,b=1):
 y=(x-a)/(b-a);return R(y)-R(y-1)
def clip(x,a,b):return a+(b-a)*ramp(x,a,b)
def pl(x,knots,values):
 slopes=np.diff(values)/np.diff(knots);return values[0]+sum(float(d)*R(x-float(k)) for k,d in zip(knots,np.diff(np.r_[0,slopes,0.])))
def log2(x,floor,ceil,sub=3):
 k=2**np.arange(np.log2(floor),np.log2(ceil)+1/sub,1/sub);return pl(x,k,np.log2(k))
def ex2(x,a,b):
 k=np.arange(a,b+.25,.25);return pl(x,k,2**k)
def div(x,y,a,b,f=.002):
 # Limited-range photometric ratio via a difference of per-image log measurements.
 return ex2(log2(x,f,64)-log2(y,f,64),a,b)
def length(a,b):
 # Mean absolute projection times pi/2 equals vector length in the angular integral.
 return (np.pi/8)*sum(R(a*np.cos(t)+b*np.sin(t))+R(-a*np.cos(t)-b*np.sin(t)) for t in np.arange(4)*np.pi/4)

# Resolve small fractional improvements over a null template without cancellation
# between separately approximated log or exponential functions near unity.
def gain(k,null,unit=.010):
 diff=D[k]-D[null]; den=D['smoothsim'];return (div(R(diff),den,-12,0,.00001)-div(R(-diff),den,-12,0,.00001))/unit
sim=D['sim']; q=clip(div((D['q+']-D['q-'])/.07,sim,-4,.25),.07,1)
# The log-scale can be negative, so take a ratio after shifting its range into R+.
ls=div((D['log_s+']-D['log_s-'])/.028+2*sim,sim,-2,2)-2
res=clip(1-ex2(2*log2(sim,.03125,64)-log2(D['energy'],.001,2048),-4,.25),0,1)
explained=ex2(2*log2(D['smoothsim'],.015625,64)-log2(D['localenergy'],.0005,2048),-7,.25)
nonDC=1-ex2(2*log2(D['localDC'],.015625,64)-log2(D['localenergy'],.0005,2048),-7,.25)
localres=div(hi(nonDC-explained,0),hi(nonDC,.0625),-8,0,.00025)
f2,f4,f8,f16,f24=[D['f'+str(s)] for s in (2,4,8,16,24)]
halo=div(f16,f4,-4,0);con=div(f2,f8,0,4)
blue=div(D['rgb8_2'],D['rgb8_0'],-3,2); bluecore=div(D['rgb3_2'],D['rgb3_0'],-3,2)
ase=div(.5*D['as5']+.5*D['as10'],.6*f4+.4*f8,-6,1)
asouter=div(D['as18'],f16+.012,-6,1)
# Angular moment amplitudes and twist (non-collinearity of m=2 across rings).
# A bar has an inner quadrupole even when the outer low-ellipticity disk has none.
# For spiral pattern, m=2 changes phase with radius, m>=3 can also be present.
A={}; twist=[];bar=[];isoph=[]
for i in range(len(rings)):
 I=D['I'+str(i)];
 for m in [1,2,3,4]:
  a,b=[D[f'{k}{i}:{m}'] for k in 'CS'];A[i,m]=length(a,b)
 isoph.append(div(A[i,2],I+.012,-6,1))
for i,j in [(0,2),(1,3),(2,4)]:
 ci,si,cj,sj=[D[f'{k}{n}:2'] for n in [i,j] for k in 'CS'];
 # Triangle inequality defect of the two vectors, cancels aligned ellipse exactly.
 torn=A[i,2]+A[j,2]-length(ci+cj,si+sj)
 twist.append(div(torn,D[f'I{j}']+.025,-7,1))
 bar.append(isoph[i]-.58*isoph[j]);
phitwist=hi(hi(*twist[:2]),twist[2]);
barstrength=hi(bar[0],hi(bar[1],bar[2]));
# Companion peak significance relative to both central contrast and absolute floor.
peaks=[]
for j in (0,1):
 denom=.6*D['nuc'+str(j)]+.35*f2+.04
 near=div(D['pair'+str(j)],denom,-6,2);far=div(D['far'+str(j)],denom,-6,2)
 peaks.append(hi(near,.73*far))
pair=hi(*peaks)
# Require evidence for an extra knot AND an envelope, both asymmetric and in
# the same annulus. This excludes simple bars, disks, and faint field stars.
companions=[]; patches=[]
for z,falloff in enumerate([1.,1.,.76]):
 knot=div(D['knot'+str(z)],.40*f2+.045,-6,2)
 patch=div(D['patch'+str(z)],f2+.050,-6,2)
 companions.append(falloff*lo(ramp(knot,.08,.42),ramp(patch,.08,.43)))
 patches.append(falloff*ramp(patch,.09,.44))
comp=hi(companions[0],hi(*companions[1:])); lopsided=hi(patches[0],hi(*patches[1:]))
chrom=[]
for rr in (6,10,16):
 mags=[div(length(D[c+'C'+str(rr)],D[c+'S'+str(rr)]),D[c+'I'+str(rr)]+.02,-6,1) for c in 'rb']
 chrom.append(mags[0]-mags[1]);
redbar=ramp(hi(chrom[0],hi(*chrom[1:])),.025,.17)
# Hand-built evidence rules. Scores share a family gate so that separating different
# elliptical shapes doesn't dilute the evidence for the elliptical family itself.
# Galaxy Zoo: smooth objects are commonly small, featureless, and red; resolved disks
# have extended light and often a blue disk around a brighter, redder core.
size=ramp(ls,-.65,1.1);ext=ramp(halo,.145,.38);bl=ramp(blue,.60,1.03)
structure=ramp(phitwist,.028,.23); mismatch=ramp(res,.045,.235)
bluegrad=ramp(blue-bluecore,0,.26)
spiral=.30*size+.24*ext+.18*bl+.13*bluegrad+.13*structure+.08*mismatch
spiral-=.11*ramp(ase,.25,.63)
edge=.61*ramp(.48-q,0,.29)+.31*ramp(ls,-.3,1.45)+.12*bl+.11*ramp(.38-q,0,.16)
edge-=.23*ramp(ase,.19,.65)
# Separate edge-on disks with a rounder, more concentrated nucleus from thin disks.
bulge=.39*ramp(con,1.8,4.1)+.32*ramp(isoph[2]-isoph[0],.14,.75)+.21*ramp(1-bluecore,.05,.57)+.12*ramp(q,.16,.36)
# Strong disturbance requires coherent broken point symmetry, not just elongation.
dist=.48*ramp(ase,.10,.52)+.23*ramp(asouter,.18,.75)+.23*mismatch+.09*bl
merg=.60*ramp(pair,.15,.74)+.23*ramp(ase,.10,.5)+.20*mismatch+.05*ramp(asouter,.2,.6)
# Disfavor apparently unrelated peripheral stars for symmetric central galaxies.
merg-=.15*(1-ramp(ase,.08,.29))
# A clean spheroid should be judged on shape, not simply surface area. In
# particular a spatially resolved but smooth disk is not evidence of arms.
clean=1-ramp(localres,.014,.080)
spiral-=.13*clean
# Very narrow disks remain edge-on even when the disk has a luminous bulge.
edge+=.15*ramp(.42-q,0,.17)*1.0+.06*ramp(ls,.6,1.5)
# The independent secondary-source and asymmetric-envelope tests are stronger
# evidence of interaction than a high residual alone. Mergers can be centered
# on the primary object, not necessarily the midpoint of a pair.
merg=hi(merg,.50*comp+.18*ramp(pair,.10,.45)+.19*ramp(localres,.05,.30)+.18*ramp(asouter,.20,.7))
dist=hi(dist,.36*ramp(ase,.12,.39)+.20*ramp(asouter,.19,.55)+.25*ramp(localres,.025,.19)+.14*lopsided+.09*bl)
# Radial lopsidedness tests for extended debris. Distinguish a torn, faint
# envelope from a second bright nucleus; the latter is separately tested above.
tail_mode=hi(div(A[2,1],D['I2']+.023,-6,1),div(A[3,1],D['I3']+.023,-6,1))
# Max-normalized within-family scores. Boundaries follow apparent axis ratio,
# coherent inner bar strength, and arm opening / strong outer structure.
def subclass(base,choices):
 best=choices[0]
 for c in choices[1:]:best=hi(best,c)
 return [base+c-best for c in choices]
# Test the physical alternatives against the centered elliptical null. A boost
# is reserved for an actual improvement in correlation, not merely a good match.
double=lo(gain('double_core','smoothsim'),gain('double_core','offset_core'));onearm=lo(gain('tidal','smoothsim'),gain('tidal','offset_core'));betterpair=gain('double_core','tidal')
diskev=[gain('disk_'+k,'smoothsim') for k in ['bar','tight','loose']];arms=hi(diskev[1],diskev[2]); diskevmax=hi(diskev[0],arms)
extra=biggest_peak=hi(D['extendedpeak0'],hi(D['extendedpeak1'],.72*D['extendedpeak2']))
second=ramp(div(extra,.38*f2+.035,-8,2),.075,.37)
# Translational robustness is most important when the presumed companion is not
# itself a resolved secondary peak. Otherwise the primary may well be offset.
double=.6*double+.4*gain('double_core','smoothsim')
merg=hi(merg,.23+.42*ramp(double,.30,2.9)+.17*ramp(betterpair,-.2,2.)+.15*comp+.19*second)
merg=hi(merg,.34*comp+.41*second+.23*ramp(double,.2,2.5)+.13*lopsided)
dist=hi(dist,.23+.39*ramp(onearm,.10,2.2)+.16*ramp(ase,.08,.35)+.12*lopsided+.08*ramp(-betterpair,-.3,1.5))
spiral+=.14*ramp(diskevmax,.10,1.1)
spiral-=.14*hi(ramp(double,1.2,5),ramp(onearm,1.1,4))
geomBulge=ramp(gain('edgebulge','edgedisk'),.18,2.1)
# The brightest opposed strip is a long disk candidate. Compare the off-plane
# light at its center to the off-plane light farther along the SAME strip.
def biggest(xs):
 while len(xs)>1:xs=[hi(xs[i],xs[i+1]) if i+1<len(xs) else xs[i] for i in range(0,len(xs),2)]
 return xs[0]
strips=[D['strip'+str(i)] for i in range(16)]; peakstrip=biggest(strips)
def aligned(tag):
 # Like the elliptical grid, use a directional derivative of a maximum to take
 # the flux in the orientation selected by a DIFFERENT filter.
 return (biggest([s+.09*D[tag+str(i)] for i,s in enumerate(strips)])-biggest([s-.09*D[tag+str(i)] for i,s in enumerate(strips)]))/.18
height=div(aligned('pole'),aligned('eq'),-4,1);outerheight=div(aligned('wing'),aligned('mid'),-4,1)
crossbulge=ramp(height-outerheight,-.02,.30);axlength=div(aligned('length'),f2,-6,.5)
bulge=.44*bulge+.19*geomBulge+.23*crossbulge+.17*ramp(1-bluecore,.04,.49)+.07*ramp(q,.20,.39)
edge+=.15*ramp(gain('edgebulge','smoothsim'),.15,1.8)+.22*ramp(axlength,.09,.32)-.12*(1-ramp(axlength,.032,.16))
smooth=.50-.13*ramp(ase,.16,.55)-.08*mismatch+.045*clean
symmetry=1-ramp(ase,.07,.27)
regular=lo(symmetry,clean);smooth+=.095*regular+.025*(1-ext)
dist=hi(dist,.17+.24*ramp(tail_mode,.14,.64)+.24*ramp(onearm,.08,1.9)+.17*ramp(asouter,.13,.58)+.14*lopsided+.10*bl-.15*second)
# A smooth, highly elliptical object with an extended major-axis strip is a
# plausible bulged disk, whereas a compact blue object need not be a spiral.
bulge+=.12*ramp(q,.17,.35)+.06*ramp(con,1.6,3.)
edge+=.085*lo(ramp(axlength,.055,.22),ramp(.49-q,0,.22))
# Persistent winding should not be dismissed as a clean elliptical profile,
# particularly when the tightly wound arms make a nearly round envelope.
winding={}
for pitch in [14,22,34,50]:
 items=[]
 for si,rr in enumerate([6,10,16]):
  for sign in [-1,1]:
   amps=[]
   for m in [2,3]:
    i=f'wind{si}_{pitch}_{sign}_{m}_';amps.append(length(D[i+'c'],D[i+'s']))
   items.append(div(hi(amps[0],.85*amps[1]),D['I'+str(si+1)]+.033,-8,0))
 winding[pitch]=biggest(items)
windtight=hi(winding[14],winding[22]);windloose=hi(winding[34],winding[50]);
pattern=ramp(hi(windtight,windloose),.045,.17)
spiral+=.13*pattern+.10*bluegrad
# Symmetric, diffuse disks do not require sharp arm knots. They usually show a
# broad area of blue light relative to both the old core and a red spheroid.
outerblue=div(D['rgb15_2'],D['rgb15_0'],-3,2)
spiral=hi(spiral,.17+.22*ext+.13*size+.21*ramp(outerblue,.66,1.09)+.19*bluegrad+.12*pattern-.14*ramp(ase,.19,.55))
# Smooth spheroids lack systematic wavelength-dependent morphology. Suppress
# spuriously inferred arms from tiny twists of a smooth elliptical light field.
spiral-=.12*lo(regular,1-ramp(blue-bluecore,.02,.20))+.05*(1-size)
spiral-=.12*ramp(ase,.16,.52)
# Correct for sampling in a nearly round isophote: the radial m=2 evidence
# distinguishes elongation from small directional biases in the best template.
roundness=q+.04*(1-ramp(isoph[1],.065,.25))
ell=subclass(smooth,[1.9*(roundness-.80),0.,1.9*(.53-q)])
barred=.75*ramp(barstrength,.07,.34)+.17*ramp(phitwist,.07,.26)+.44*redbar
# Loose arms emerge beyond the central concentration and often show strong m=2
# modes or clumps. Tight arms more nearly blend into an axisymmetric light profile.
outmode=div(A[3,3]+A[3,1],D['I3']+.025,-5,2)
loose=.38*structure+.22*ramp(outmode,.22,.76)+.22*bl+.18*ramp(res,.04,.22)
redshape=ramp(gain('redbar','rednull'),.05,1.3)
sp=subclass(spiral,[.37*(barred-.60)+.12*clip(gain('disk_bar','disk_tight'),-.6,1.)+.17*redshape,.36*(.52-loose),.36*(loose-.52)+.08*clip(gain('disk_loose','disk_tight'),-1,1)+.15*clip((windloose-windtight)/.085,-1,1)])
ed=subclass(edge,[.38*(.52-bulge),.38*(bulge-.52)])
scores=[dist-.08,merg-.04]+ell+sp+ed

# Schedule rectifier nodes by depth and preserve signed values through linear skips.
# Each stage reads only referenced older coordinates (not the entire prior history).
used=set();depth={i:0 for i in range(len(names))}
def visit(i):
 if i in depth:return depth[i]
 d=1+max((visit(j) for j in defs[i].t),default=0);used.add(i);depth[i]=d;return d
for f in scores:
 for i in f.t:visit(i)
bydepth={0:list(range(len(names)))}; src={0:raw}
for i in sorted(used):bydepth.setdefault(depth[i],[]).append(i)
def emit(exprs,nm):
 usedd=sorted({depth[j] for f in exprs for j in f.t});parts=[]
 for d in usedd:
  js=bydepth[d]; idx={j:k for k,j in enumerate(js)}
  # Many hinges differ only by the location of their breakpoint: compute the
  # common linear form once, then fan it out and add the separate biases.
  forms={}; ii=[]
  for f in exprs:
   key=tuple(sorted((idx[j],v) for j,v in f.t.items() if j in idx))
   if key not in forms:forms[key]=len(forms)
   ii.append(forms[key])
  b=np.array([f.b if d==usedd[0] else 0 for f in exprs]);dest=nm if len(usedd)==1 else None
  if len(forms)*(len(js)+len(ii))<.91*len(js)*len(ii):
   W=np.zeros((len(forms),len(js)),'f')
   for row,i in forms.items():
    for j,v in row:W[i,j]=v
   ux=aff(src[d],W);r=np.zeros((len(ii),len(forms)),'f');r[np.arange(len(ii)),ii]=1
   parts.append(aff(ux,r,b,name=dest))
  else:
   W=np.zeros((len(ii),len(js)),'f')
   for i,f in enumerate(exprs):
    for j,v in f.t.items():
     if j in idx:W[i,idx[j]]=v
   parts.append(aff(src[d],W,b,name=dest))
 return parts[0] if len(parts)==1 else op('add',parts,nm)
for d in sorted(bydepth)[1:]:src[d]=rect(emit([defs[i] for i in bydepth[d]],'logic_%d'%d))
emit(scores,'morphology_scores')
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
(out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':')))
np.savez(out/'weights.npz',**weights)
