"""A fixed photometric/morphology circuit. No inputs to the builder except its destination."""
import json, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
layers=[]; weights={}; D={}
def layer(name, op, ins, **kwargs):
    if isinstance(ins,str): ins=[ins]
    layers.append(dict(name=name,op=op,inputs=ins,**kwargs)); return name
def dense(name, inp, w, b=0):
    w=np.array(w,dtype='f').reshape(len(w),-1); D[name]=len(w)
    weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(np.asarray(b,dtype='f'),(len(w),)).copy()
    return layer(name,'dense',inp,out=len(w))
def conv(name,inp,w,b=0,stride=1,pad=0):
    w=np.asarray(w,dtype='f'); n=len(w); k=w.shape[-1]
    weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(np.asarray(b,dtype='f'),(n,)).copy()
    return layer(name,'conv',inp,out=n,kernel=[k,k],stride=stride,padding=pad)
def pool(name,inp,k,stride=None,op='avgpool'):
    if isinstance(k,int):k=[k,k]
    return layer(name,op,inp,size=k,stride=stride or k)
def maxgroups(name,inp,groups,per):
    sh=layer(name+'_bank','reshape',inp,shape=[groups,1,per]); mp=pool(name+'_maximum',sh,[1,per],op='maxpool')
    D[name]=groups; return layer(name,'reshape',mp,shape=[groups])

# Mild spatial integration. Three-band luminance emphasizes r and g, which
# have better angular resolution/less color speckle than the displayed z band.
rgb=pool('bin2','image',2)
lum=conv('luminance',rgb,np.array([.24,.48,.28]).reshape(1,3,1,1))
phot=conv('sky_clips',lum,np.ones((3,1,1,1)),[-.10,-.20,-.36]); phot=layer('isophotal_light','relu',phot)
# feature front ends produce vectors; scalar symbolic arithmetic below maps to
# a DAG of affine layers, ReLUs and skip connections.
class Expr:
    def __init__(self,c=0,terms=None):self.c=float(c);self.t=terms or {}
    def __add__(a,b):
        if not isinstance(b,Expr): b=Expr(b)
        d=a.t.copy()
        for k,v in b.t.items():d[k]=d.get(k,0)+v
        return Expr(a.c+b.c,{k:v for k,v in d.items() if v})
    __radd__=__add__
    def __neg__(a):return a*(-1)
    def __sub__(a,b):return a+(-b)
    def __rsub__(a,b):return b+(-a)
    def __mul__(a,b):return Expr(a.c*b,{k:v*b for k,v in a.t.items()})
    __rmul__=__mul__
    def __truediv__(a,b):return a*(1/float(b))
front=[]; rs=[]; depths=[]
def var(i):return Expr(terms={i:1.})
def features(name):
    ids=[]
    for i in range(D[name]):
        ids.append(var(len(depths)));depths.append(0);front.append((name,i))
    return ids
def pos(e):
    depth=1+max((depths[i] for i in e.t),default=0); i=len(depths);rs.append((i,depth,e));depths.append(depth);return var(i)
def clip(e,a,b):return a+pos(e-a)-pos(e-b)
def mx(a,b):return pos(a-b)+b
def mn(a,b):return a-pos(a-b)
def absval(a):return pos(a)+pos(-a)
def mean(x):return sum(x)/len(x)
def ratio(a,b,lim=(0.,1.),levels=16,floor=.0001):
    # Uniform bank of comparators for a/b. Its output is the integral of
    # 1[a > t*b] in t, restricted to the specified interval.
    lo,hi=lim;step=(hi-lo)/levels; total=Expr(lo)
    for j in range(levels):
        t=lo+(j+.5)*step; total+=step*clip((a-t*(b+floor))*600+.5,0,1)
    return total

def grid(n):
    u=(np.arange(n)+.5-n/2)*(128/n);return np.meshgrid(u,u)
x,y=grid(64); r=np.hypot(x,y); r2=r*r; th=np.arctan2(y,x)
def gauss(s):return np.exp(-.5*r2/s**2)
def unit(a):return a/a.sum()
# Aperture moments of excess light. The soft isophote suppresses sky and
# background sources; concentric scales separate spheroid, bar and thin disk.
rows=[]; momkeys=[]
def photrow(key,w,frac):
    momkeys.append(key);rows.append(np.einsum('c,yx->cyx',frac,w))
for idx,s in enumerate([5,9,16,26]):
    a=unit(gauss(s));rad=r2/s**2; f=[.25,.75,0] if s<12 else [.63,.37,0]
    for what,arr in [('L',1),('T',rad),('Q',(x*x-y*y)/s**2),('U',2*x*y/s**2)]:
        photrow((idx,what),a*arr,f)
# Effective area (in original pixels) within soft isophotes.
for idx,t in enumerate([.1,.2,.36]):
    photrow(('area',idx),4*gauss(16),np.eye(3)[idx]);photrow(('core',idx),unit(gauss(2)),np.eye(3)[idx]);
fm=features(dense('aperture_moments',phot,rows));moms=dict(zip(momkeys,fm))
e=[]; phase=[]
for idx,s in enumerate([5,9,16,26]):
    q,u,tr=[moms[idx,v] for v in ['Q','U','T']];
    # norm of the quadrupole via angular quadrature of |projection|.
    amps=[absval(q*np.cos(a)+u*np.sin(a)) for a in np.arange(6)*np.pi/6];amp=mean(amps)*(np.pi/2)
    e.append(ratio(amp,tr,(0,1),32,.00015))
    phase.append((q,u,tr))
# Effective light area divided by central SB: a rough size independent of flux.
area=ratio(moms['area',1]/400,moms['core',1]+.02,(0,2.5),32)*400
# Broad-band colors with local sky measured outside the source aperture.
rows=[]
for s in [3,7,14,22]:
    w=unit(gauss(s))-unit(np.exp(-.5*((r-53)/5)**2)); rows.extend([w[None]*v[:,None,None] for v in np.eye(3)])
colorphot=features(dense('sky_subtracted_colors',rgb,rows));warm=[]
for i in range(4):
    R,G,B=colorphot[3*i:3*i+3];warm.append(ratio(R-B,R*.45+G*.3+B*.25+.01,(-.8,1),36))

# Bilateral lopsidedness on a polar lattice. Maximal symmetry over a tiny
# neighborhood makes this tolerant of subpixel errors in catalog astrometry.
# A far ring is allowed to contribute less than a near one (field objects).
proj=[]; pairweights=[]; numoffs=5
for dx,dy in [(0,0),(2,0),(-2,0),(0,2),(0,-2)]:
    for rr,n,dr in [(4,8,4),(8,12,5),(14,16,7),(22,20,9),(33,24,13)]:
        for a in np.arange(n//2)*2*np.pi/n:
            vx,vy=rr*np.cos(a),rr*np.sin(a);s=1.3+rr*.085
            ap=[unit(np.exp(-((x-dx-sign*vx)**2+(y-dy-sign*vy)**2)/(2*s*s))) for sign in [1,-1]]
            proj.extend([ap[0]-ap[1],ap[1]-ap[0],ap[0]+ap[1]-2*unit(np.exp(-.5*((r-52)/5)**2))]);pairweights.append((rr,rr*dr/n*np.exp(-.5*(rr/16)**2)))
asymproj=dense('opposite_sectors',lum,proj,np.tile([-.006,-.006,-.015],len(pairweights))); apositive=layer('sector_excess','relu',asymproj)
npair=len(pairweights)//numoffs; rows=[]
for i in range(numoffs):
    for v in [[1,1,0],[0,0,1]]:
        co=np.zeros((len(pairweights),3));w=np.array([w for _,w in pairweights[i*npair:(i+1)*npair]]);co[i*npair:(i+1)*npair]=unit(w)[:,None]*v;rows.append(co)
af=features(dense('bilateral_flux',apositive,rows));asym=ratio(af[0],af[1]+.012,(0,.85),24)
for i in range(1,numoffs):asym=mn(asym,ratio(af[2*i],af[2*i+1]+.012,(0,.85),24)+.012)
# A compact nucleus away from the center is evidence for an interaction. Test
# for a radial valley toward the primary, not just flux in the primary's wing.
# Test both equidistant and intermediate saddles to cover unequal pairs.
rows=[]; offsets=[]; info=[]
for rr,n in [(5,12),(8,16),(12,24),(18,32),(26,40),(36,48)]:
    for a in np.arange(n)*2*np.pi/n:
        xx,yy=rr*np.cos(a),rr*np.sin(a)
        def loc(f,s):return unit(np.exp(-((x-xx*f)**2+(y-yy*f)**2)/(2*s*s)))
        # a nucleus should be bright, locally compact, and there should be
        # a dip between it and the origin. Red light highlights old stars.
        s=min(2.2,rr*.22);peak=loc(1,s);saddle=loc(.5,s*.83);broad=loc(1,5.5);sky=unit(np.exp(-.5*((r-54)/5)**2));
        # one common response and a penalty if the object is a tiny PSF
        narrow=loc(1,1.2); ext=loc(1,3.5)
        rows.extend([peak-saddle,peak-broad,peak-sky,ext-.50*narrow-.50*sky]);offsets.append(rr);info.append((xx,yy))
redlum=conv('old_stellar_light',rgb,np.array([.58,.34,.08]).reshape(1,3,1,1))
pairraw=dense('secondary_peak_tests',redlum,rows); npeak=len(offsets)
# Use repeated 1x1 filters to combine tests at each sample.
pstack=layer('peak_layout','reshape',pairraw,shape=[1,npeak,4]);pstack=layer('peak_vector','reshape',pstack,shape=[npeak*4])
def perlocation(name,src,k,co,bias=0):
    co=np.asarray(co); nn,k0=co.shape;assert k==k0
    return dense(name,src,np.kron(np.eye(npeak,dtype='f'),co),np.tile(np.broadcast_to(np.asarray(bias),(nn,)),npeak))
pn=perlocation('nucleus_contrasts',pstack,4,[[1,0,0,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,-1]], [0,0,.07,.030]);pr=layer('positive_nucleus_contrasts','relu',pn)
# Penalize absence of a saddle; prevent strong unrelated field stars from
# winning unless they are close, extended and have comparable luminosity.
wp=[];bp=[]
for j,rr in enumerate(offsets):
    w=np.zeros((npeak,4));w[j]=[1.,-.3,-.6,-2.0];wp.append(w);bp.append(-.0025*max(0,rr-8))
peaks=dense('paired_nucleus_votes',pr,wp,bp);pair=features(maxgroups('second_nucleus',peaks,1,npeak))[0]

# Compensated elliptical Gaussian matched filters. Each is calculated by an
# analytic function, normalized independently. Max correlation supplies a
# second axis-ratio estimator less biased by circular weighting/truncation.
# Soft selection among q values also yields a scale estimate (using the same
# shape/size grid transposed below).
sm=pool('bin_matched',lum,2); X,Y=grid(32);RR=np.hypot(X,Y);sky=unit(np.exp(-.5*((RR-53)/5)**2)); qs=[.15,.22,.31,.42,.56,.72,.89,1.]; sigmas=[3.,4.5,6.5,9.,13.,18.,25.];nt=16
filters=[]
for q in qs:
    for s in sigmas:
        for a in np.arange(nt)*np.pi/nt:
            A=X*np.cos(a)+Y*np.sin(a);B=Y*np.cos(a)-X*np.sin(a)
            p=np.exp(-.5*((A/s)**2+(B/(s*q))**2)); p=p/(np.sqrt((p*p).sum())+1.e-20); filters.append(p-sky*p.sum())
match=dense('ellipse_templates',sm,filters); ma=maxgroups('ellipse_angles',match,len(qs)*len(sigmas),nt)
shapebank=features(maxgroups('ellipse_widths',ma,len(qs),len(sigmas)))
# Peak position by summing the indicators that the matched filter response is
# still increasing as q increases. The response is unimodal in the one-ellipse case.
qhat=Expr(qs[0]);
for j in range(1,len(qs)):
    qhat+=(qs[j]-qs[j-1])*clip(.5+(shapebank[j]-shapebank[j-1])*65,0,1)
# Azimuthal gradients in tilted disk coordinates. Straight features on a
# deprojected circle have slope 0; logarithmic arms have a slope of either
# handedness. Compare local residual orientations, not a global arm template,
# since arms are frequently broken or branch. Precompute analytic polar masks.
# Independent evidence from the inner/outer quadrupoles detects a bar which
# ends inside a disk of quite different shape/orientation.
rows=[];pt=[]; cfg=[]
for rr,sw in [(4,1.5),(7,2.),(11,2.7),(16,3.6),(23,4.8),(32,6.)]:
    rad=unit(np.exp(-.5*((r-rr)/sw)**2));
    for m in [0,2,3,4]:
        if m==0:rows.append(rad-unit(np.exp(-.5*((r-53)/5)**2)));pt.append((rr,0,'l'))
        else:
            for tag,arr in [('c',np.cos(m*th)),('s',np.sin(m*th))]:rows.append(rad*(arr-(rad*arr).sum()));pt.append((rr,m,tag))
rf=dict(zip(pt,features(dense('ring_harmonics',lum,rows))));
# differences of normalized quadrupole vectors at different radii (isophote
# twist, or a strong change of ellipticity). Preserve harmonic sign/phase.
ringvec={};quad={}
for rr in [4,7,11,16,23,32]:
    den=rf[rr,0,'l']+.028
    V=[ratio(rf[rr,2,t],den,(-.9,.9),48) for t in ['c','s']];ringvec[rr]=V
    quad[rr]=mean([absval(V[0]*np.cos(a)+V[1]*np.sin(a)) for a in np.arange(4)*np.pi/4])*(np.pi/2)
def vdiff(i,j):
    u=ringvec[i][0]-ringvec[j][0];v=ringvec[i][1]-ringvec[j][1];return mean([absval(u*np.cos(a)+v*np.sin(a)) for a in np.arange(4)*np.pi/4])*(np.pi/2)
# A bar makes early isophotes more elongated than outer isophotes or
# misaligned with them, not simply more circular than them as in a bulge.
bars=[]
for i,j in [(4,11),(7,16),(11,23),(16,32)]:
    bars.append(.58*vdiff(i,j)+.55*(quad[i]-quad[j])+.22*quad[i]-.7*pos(.055-rf[j,0,'l']))
bar=mx(mx(bars[0],bars[1]),mx(bars[2],bars[3]))
# Spiral-arm roughness: exploit the local curvature of surface brightness.
# A smooth ellipse has predominantly one bright maximum. A spiral can have
# many local ridges against a locally concave (inter-arm) background. Test
# a peak against both sides along 4 axes; apply an inner hole to reject cores.
def roughness():
    rows=[];we=[];br=[];pts=[]
    for rr,n in [(5,12),(9,16),(14,24),(21,32),(30,40)]:
        for ang in np.arange(n)*2*np.pi/n:
            xx,yy=rr*np.cos(ang),rr*np.sin(ang);s=1.15+rr*.04;d=2.5+rr*.09
            def loc(a,b):return unit(np.exp(-((x-a)**2+(y-b)**2)/(2*s*s)))
            c=loc(xx,yy);pts.append((rr,ang));
            # Along the outward radial direction the surface brightness of a
            # smooth elliptical galaxy decreases, whatever its PA or q.
            u,v=np.cos(ang),np.sin(ang)
            k1=loc(xx-u*d,yy-v*d); k2=loc(xx+u*d,yy+v*d)
            rows.extend([c-k1,k2-c,c-loc(xx-u*2*d,yy-v*2*d)])
            we.append(rr*(4 if rr<12 else 8)/n*np.exp(-.5*(rr/17)**2));br.append(c)
    raw=dense('radial_reversals',lum,rows,np.tile([-.007,-.007,-.009],len(we)));rev=layer('rising_profile','relu',raw)
    num=features(dense('positive_variations',rev,[np.einsum('i,j->ij',unit(np.asarray(we)),[.50,.35,.15])]))[0]
    radialmean=np.einsum('i,ijk->jk',unit(np.asarray(we)),br)-unit(np.exp(-.5*((r-53)/5)**2));den=features(dense('disk_light',lum,[radialmean]))[0];return ratio(num,den+.015,(0,.45),32)
rough=roughness()

# Central isophotal chord lengths: integrate a soft occupancy outwards on
# each ray. Requiring light halfway along the same ray removes disconnected
# stars. Taking the perpendicular chord at the longest diameter suppresses
# spurious roundness from the seeing disk or background.
angs=np.arange(32)*np.pi/16; radii=np.arange(1,51,2); rows=[]
for a in angs:
    for d in radii:
        s=1.+.035*d
        rows.append(unit(np.exp(-.5*((x-np.cos(a)*d)**2+(y-np.sin(a)*d)**2)/s**2)))
ray=dense('ray_photometry',lum,rows)
# two levels, one referenced to the central aperture, one to a faint
# absolute surface brightness above sky. Sample them via soft clipping.
core,sky0=features(dense('nuclear_reference',lum,[unit(gauss(2)),unit(np.exp(-.5*((r-53)/5)**2))]))
rayvars=features(ray); chordqs=[]; lens=[]; raybars=[]; irregularities=[]; minors=[]; diameters=[]; major_angles=[]; allchords=[]; contourwarp=[]
for frac,delta in [(.15,.035),(.5,.015)]:
    chords=[]; rag=[]
    for ai,a in enumerate(angs):
        p=rayvars[ai*len(radii):(ai+1)*len(radii)];cut=sky0+frac*(core-sky0)+delta
        masks=[clip((v-cut)*30+.5,0,1) for v in p];
        # Same-ray connectivity downweights unrelated sources; guard at two
        # upstream positions. Two points suffice for visible plateau widths.
        masks=[mn(v,mn(masks[int(i*.50)],masks[int(i*.77)])) if i>2 else v for i,v in enumerate(masks)]
        chords.append(2*sum(masks))
    diam=[(chords[i]+chords[i+16])*.5 for i in range(16)]
    def extremum(items):
        if len(items)==1:return items[0]
        return extremum([mx(a,b) for a,b in zip(items[::2],items[1::2])]+(items[-1:] if len(items)%2 else []))
    major=extremum(diam); minor=-extremum([-v for v in diam]);lens.append(major);chordqs.append(ratio(minor,major,(0,1),32,.02))
    allchords.append(chords); minors.append(minor); diameters.append(diam)
    # First-order effect of translating an ellipse: the odd part of its
    # squared radial intersections divided by the even part squared is a
    # first harmonic. Estimate the two translation terms from the entire
    # contour instead of confusing a centroid error with a tidal tail.
    odd=[chords[i]-chords[i+16] for i in range(16)];
    # For offsets << disk radius, the normalized difference is a dipole. A
    # soft denominator handles very small/noisy contours.
    g=[ratio(v,d+2,(-1.,1.),24,.01) for v,d in zip(odd,diam)]
    A=2*mean([v*np.cos(i*np.pi/16) for i,v in enumerate(g)]);B=2*mean([v*np.sin(i*np.pi/16) for i,v in enumerate(g)])
    contourwarp.append(mean([absval(v-A*np.cos(i*np.pi/16)-B*np.sin(i*np.pi/16)) for i,v in enumerate(g)]))
    scale=clip(major*.075,.55,1.6); major_angles.append([clip(1-(major-v)*1.2,0,1) for v in diam])
    # A left-right contour length discrepancy is very diagnostic of tails,
    # but only when it cannot be explained by a small centering error.
    rag=mean([pos(absval(chords[i]-chords[i+16])-2.) for i in range(16)]);irregularities.append(ratio(rag,mean(diam)+1,(0,1),32))

# Compare an image to its reflection in a candidate major axis. Elliptical
# systems have both reflection and half-turn symmetry; a two-armed spiral
# can have half-turn symmetry yet no good reflection plane. Average residues
# within annuli so that blank sky never has the weight of thousands of pixels.
M=32; rradii=[4,7,11,16,23]; an=np.arange(M)*2*np.pi/M; fields=[]; lightrows=[]
for R in rradii:
    vals=[]
    for a in an:
        vals.append(unit(np.exp(-.5*((x-R*np.cos(a))**2+(y-R*np.sin(a))**2)/(1+R*.06)**2)))
    fields.extend(vals); lightrows.append(np.mean(vals,axis=0)-unit(np.exp(-.5*((r-53)/5)**2)))
polar=dense('polar_disk_samples',lum,fields); light=features(dense('annular_brightness',lum,lightrows));nring=len(rradii)
mat=[];bins=[]
for off in range(M//2):
    # The two axes of an ellipse are interchangeable in this metric. Count
    # both perpendicular reflections, which also suppresses the asymmetry of
    # a central displacement along one of the axes.
    for ir,R in enumerate(rradii):
        start=len(mat)
        for add in [0,M//2]:
            for i in range(M):
                j=(off+add-i)%M
                if j<=i:continue
                w=np.zeros((nring,M),'f');w[ir,i]=1;w[ir,j]=-1;mat.extend([w,-w]);
        bins.append((start,len(mat)))
diff=dense('reflected_sectors',polar,mat,-.010);diff=layer('reflection_discrepancy','relu',diff)
w=np.zeros((len(bins),len(mat)),'f')
for i,(a,b) in enumerate(bins):w[i,a:b]=2/(b-a)
refl=features(dense('reflection_errors',diff,w));co=[.10,.24,.28,.25,.13];den=sum(c*l for c,l in zip(co,light))+.025
numer=[sum(c*v for c,v in zip(co,refl[j*nring:(j+1)*nring])) for j in range(M//2)];mirror=ratio(-extremum([-v for v in numer]),den,(0,.85),32)
# Disk texture also causes residuals relative to a monotone profile. Compare
# small positive and negative azimuthal high-pass (the 2nd harmonic accounts
# for most of the elongation of a smooth, moderately inclined ellipse).
mat=[]
for i,R in enumerate(rradii):
    for j in range(M):
        a=an-an[j];w=np.zeros((nring,M));w[i]=-(1+2*np.cos(a)+2*np.cos(2*a)+2*np.cos(3*a)+2*np.cos(4*a))/M;w[i,j]+=1;mat.extend([w,-w])
hp=layer('resolved_knots','relu',dense('azimuthal_highpass',polar,mat,-.008)); weights_ring=np.repeat(np.asarray(co)/(M),M*2)
knots=ratio(features(dense('fine_structure_flux',hp,[weights_ring]))[0],den,(0,.4),32)
# Central profile curvature: a broad, nearly uniform disk enclosing a small
# bulge has more light at intermediate radii than a simple spheroid. Use
# average enclosed-radii rather than diameter alone for the size evidence.
radmean=[mean(d) for d in diameters];relative_core=ratio(radmean[1],radmean[0]+.5,(0,1),32)
# Local disk aspect at each station on the major axis. Subtract the minor-
# axis width well outside the nucleus from the central width; this makes a
# bulge detector largely independent of the red color of the disk itself.
# Include extent along the axis, since a round PSF should not count as a bulge.
wp=[];cent=[];arms=[];axisdir=[]
for a in np.arange(16)*np.pi/16:
    u,v=np.cos(a),np.sin(a); arr=[]
    def spot(along,across):return unit(np.exp(-.5*((x-(u*along-v*across))**2+(y-(v*along+u*across))**2)/(1.35**2)))
    for dt in [0,8,14]:
        for off in [0,3,6]:
            arr.append(sum(spot(s1*dt,s2*off) for s1 in [-1,1] for s2 in [-1,1])/4-unit(np.exp(-.5*((r-53)/5)**2)))
    wp.extend(arr)
ag=features(dense('disk_cross_sections',lum,wp));bulgev=[]
for i in range(16):
    c0,c3,c6,a0,a3,a6,b0,b3,b6=ag[9*i:9*i+9]
    # use q vs the same PSF both on the center and away from it
    wmid=ratio(.65*c3+.35*c6,c0+.015,(0,1),32); wfar=ratio(.50*(a3+b3)*.65+.50*(a6+b6)*.35,.5*(a0+b0)+.02,(0,1),32)
    bulgev.append(wmid-wfar-.30*(1-major_angles[0][i]))
bulge_width=extremum(bulgev)

# Project a family of face-on disks into the measured plane. Within each
# projected disk sample the phase of the m=2 harmonic on a log-radius grid.
# An inclined ellipse or straight bar has stationary phase (zero imaginary
# cross-spectrum); the angular phase of logarithmic arms advances linearly
# with log radius. Detect both senses of winding without fitting templates.
dq=[1.,.64,.40]; angles_d=np.arange(8)*np.pi/8; radii_d=4.0*1.40**np.arange(7); cfg=[];rows=[]
for dratio in dq:
    for ia,a in enumerate(angles_d[:1] if dratio==1 else angles_d):
        u=x*np.cos(a)+y*np.sin(a);v=(-x*np.sin(a)+y*np.cos(a))/dratio;rd=np.hypot(u,v);ad=np.arctan2(v,u)
        for k,rad in enumerate(radii_d):
            win=unit(np.exp(-.5*((rd-rad)/(.8+rad*.105))**2));
            for ff in [np.cos(2*ad),np.sin(2*ad)]:rows.append(win*(ff-(win*ff).sum()))
        cfg.append((dratio,ia))
fourier=dense('inclined_spiral_harmonics',lum,rows); z=features(fourier)
# Coherent sums for two log-spiral pitches over overlapping radial windows.
# Opposite handedness is its own null control: a steady ellipse or bar gives
# equal responses at left and right pitch and does not look like a spiral.
rows=[];descriptor=[]; ncoeff=len(radii_d)*2
for j,(_,ia) in enumerate(cfg):
    for start in [0,2]:
        for pitch in [17.,33.]:
            for hand in [-1,1]:
                kvec=hand*2/np.tan(np.deg2rad(pitch))*np.log(1.4)*np.arange(5)
                # Subtract the DC response on each radius segment analytically,
                # so a fixed elliptical component does not add to the arm.
                w=np.exp(1j*kvec); w-=w.mean(); w/=5
                for ang in np.arange(8)*np.pi/4:
                    p=np.zeros((len(cfg),len(radii_d),2)); f=w*np.exp(1j*ang);p[j,start:start+5]=np.stack([f.real,f.imag],axis=-1);rows.append(p)
sp=dense('log_spiral_pitch',fourier,rows); votes=features(maxgroups('spiral_phase',sp,len(rows)//8,8));running=[];tights=[];opens=[]
# Favor projected disks that agree with the outer contours, but keep several
# alternatives active for galaxies with weakly determined position angles.
for j,(dratio,ia) in enumerate(cfg):
    pr=.020*pos(absval(chordqs[0]-dratio)-.12)
    if dratio<1:pr+=.018*(1-major_angles[0][2*ia])
    tight=[];wide=[]
    for k in range(2):
        tL,tR,oL,oR=votes[8*j+4*k:8*j+4*k+4]
        tight.append(absval(tL-tR)-pr);wide.append(absval(oL-oR)-pr)
    tights.append(mx(*tight));opens.append(mx(*wide))
wt=extremum(tights);wo=extremum(opens);winding=ratio(mx(wt,wo),.7*light[2]+.3*light[1]+.045,(0,.38),36)
pitchsign=ratio(wo-wt,wo+wt+.006,(-1,1),32)
# Even an elongated disk with no obvious arms can be a smooth cigar. Determine
# diskiness from outer-to-inner diameter, area, and profile concentration.
thindisk=clip((lens[0]-2.1*lens[1]-1.0)/4,-1.3,1.7)

# Compare to reflection/rotation after removing the *directional derivative*
# caused by a modest center error. A shift only produces odd m=1 to first
# order on circular rings of a round source. Odd higher harmonics and strong
# changes of the dipole across radii identify tails/knots and multiple centers;
# normal spirals are primarily even and a shifted elliptical has fixed phase.
rows=[]
for ir,rad in enumerate(rradii):
    for m in [1,3,5]:
        for tag in [np.cos,np.sin]:
            w=np.zeros((len(rradii),M)); w[ir]=tag(m*an)/M; rows.append(w)
od=features(dense('odd_disk_multipoles',polar,rows));skew=[];lops=[]
for i in range(len(rradii)):
    a0,b0,a,b,c,d=od[6*i:6*i+6]; lops.append([a0,b0]);skew.append((absval(a)+absval(b))+.6*(absval(c)+absval(d)))
m3=ratio(sum(k*v for k,v in zip(co,skew)),den,(0,.8),32)
# Same dipole in the inner and outer ring, with larger normalized amplitude
# at intermediate radius for a small pointing offset; opposite directions
# or a large *outer* unbalanced component indicate a genuinely distorted disk.
tailodd=ratio(absval(lops[3][0]-.6*lops[1][0])+absval(lops[3][1]-.6*lops[1][1]),.08+light[1]+light[3],(0,.8),32)
# Identify bars in red (old stellar) light, independently of blue arms.
# Angular gradient normalizes a harmonic against the *local radial slope*,
# not annular flux. An unbarred ellipse has nearly fixed isophote ellipticity
# in this representation, independent of its precise radial light profile.
rr=[4,7,11,16,23]; rows=[]
for R in rr:
    sw=1+R*.09; ann=unit(np.exp(-.5*((r-R)/sw)**2));gi=unit(np.exp(-.5*((r-R*.68)/sw)**2));go=unit(np.exp(-.5*((r-R*1.4)/sw)**2));
    rows.append(gi-go)
    for f in [np.cos(2*th),np.sin(2*th)]:rows.append(ann*(f-(ann*f).sum()))
brm=features(dense('bar_isophote_tensors',redlum,rows));tens=[];ell=[]
for i,R in enumerate(rr):
    slope,c,s=brm[3*i:3*i+3];v=[ratio(c,.05+pos(slope),(-.9,.9),36),ratio(s,.05+pos(slope),(-.9,.9),36)];tens.append(v)
    ell.append(mean([absval(v[0]*np.cos(a)+v[1]*np.sin(a)) for a in np.arange(4)*np.pi/4])*np.pi/2)
bar_end=[]
for i,j in [(0,2),(1,3),(2,4)]:
    d=[tens[i][l]-tens[j][l] for l in [0,1]];dt=mean([absval(d[0]*np.cos(a)+d[1]*np.sin(a)) for a in np.arange(4)*np.pi/4])*np.pi/2
    bar_end.append(.65*dt+.70*(ell[i]-ell[j])+.25*ell[i]-.8*pos(.055-light[j]))
redbar=extremum(bar_end)

# A second nucleus must stick up on BOTH sides of the saddle. Search along
# short line segments with the center either near an endpoint or between
# the endpoints (the catalog can center a pair on its joint photocenter).
# A broad-aperture test rejects PSF-like point sources, and a dark gap is
# weaker evidence of interaction than a saddle inside a luminous envelope.
rows=[]; separate=[];sky=unit(np.exp(-.5*((r-53)/5)**2))
for R,N in [(4,8),(7,12),(10,16),(15,24),(21,32),(28,40),(38,48)]:
    for a in np.arange(N)*2*np.pi/N:
        for t in [0.,-.55,-1.]:
            if t==-1 and a>=np.pi:continue
            def at(f,s):return unit(np.exp(-.5*((x-f*R*np.cos(a))**2+(y-f*R*np.sin(a))**2)/s**2))
            p1,p2=at(t,1.6),at(1,1.6);v=at((t+1)*.5,1.45);wide1,wide2=at(t,4),at(1,4)
            # min(P1,P2)-valley = (P1-valley)-relu(P1-P2).
            rows.extend([p1-v,v-p1,p1-p2,.43*(p1-sky)-(wide1-sky),.43*(p2-sky)-(wide2-sky),sky-v]);separate.append(R*(1-t))
nd=len(separate);b=np.tile([0,0,0,.008,.008,.055],nd);pp=layer('two_lobe_contrasts','relu',dense('two_lobe_photometry',redlum,rows,b))
w=np.zeros((nd,nd,6),'f')
for j in range(nd):w[j,j]=[1,-1,-1,-2.,-2.,-.85]
pp=dense('lobe_saddle_depths',pp,w,[-.0018*max(0,d-12) for d in separate]);realpair=features(maxgroups('resolved_pair',pp,1,nd))[0]

# For a contour of an ellipse 1/r(theta)^2 has ONLY a constant and a
# 2nd angular harmonic. Test the residual shape at both surface brightness
# levels. Use antipodal diameters (first-order translation invariance) and a
# softened inverse-square to limit pixel discretization at the smallest radii.
def invsq(d):
    xp=np.array([0,1.5,2.5,3.5,4.5,6.,8.,10.,13.,17.,23.,32.,50.]);yp=100/(xp+1.5)**2;sl=np.diff(yp)/np.diff(xp)
    return yp[0]+sl[0]*d+sum((sl[i]-sl[i-1])*pos(d-xp[i]) for i in range(1,len(sl)))
ringerrors=[];quadvec=[]
for diam in diameters:
    v=[invsq(d) for d in diam];dc0=mean(v);uc=np.cos(2*angs[:16]);us=np.sin(2*angs[:16]);qc=2*mean([t*u for t,u in zip(v,uc)]);qs0=2*mean([t*u for t,u in zip(v,us)])
    quadvec.append([ratio(qc,dc0+.001,(-1,1),48),ratio(qs0,dc0+.001,(-1,1),48)])
    residual=mean([absval(t-dc0-u*qc-z*qs0) for t,u,z in zip(v,uc,us)])
    ringerrors.append(ratio(residual,dc0+.001,(0,.6),32))
contour_twist=mean([absval((quadvec[1][0]-quadvec[0][0])*np.cos(a)+(quadvec[1][1]-quadvec[0][1])*np.sin(a)) for a in np.arange(4)*np.pi/4])*np.pi/2

# Rule heads. Units: q is minor/major; e is quadrupole/trace; color is
# sky-subtracted red-minus-blue normalized by light. Score evidence in
# comparable units, then use a small conditional tree to avoid swamping
# the family decision with confidence about a within-family subclass.
# Invert e -> q in a Gaussian-weighted moment. A finite window rounds a shape,
# hence compare moments most heavily in the best illuminated aperture.
def tof(e):
    knots=[0,.18,.38,.55,.72,.84,.93,1];vals=[np.sqrt((1-a)/(1+a)) for a in knots];sl=np.diff(vals)/np.diff(knots)
    return vals[0]+sl[0]*e+sum((sl[i]-sl[i-1])*pos(e-knots[i]) for i in range(1,len(sl)))
qmom=tof(.62*e[1]+.38*e[2]);q=clip(.38*qhat+.15*(qmom-.045)+.37*chordqs[0]+.10*chordqs[1],.08,1)
col=.55*warm[1]+.45*warm[2];blue=clip((.12-col)*2.6,-1.3,1.4)
# Diameter distinguishes a narrow lenticular with no visible extended disk
# from an edge-on spiral. Dilute the significance of size in nearby objects.
size=clip((lens[0]-18)/8,-1.7,2.2); clump=clip((rough-.045)*10,-.45,1.6)
# Separate feature families with signed scores.
edged=clip((.33-q)*23,-5,3.5)+.74*size+.38*blue-.30*pos(clump)+.20*clip((e[2]-e[0]-.20)*5,-1,1)
edgecolor=.65*warm[0]+.35*warm[1];bulge=3.15*(edgecolor-.27)+1.3*(e[2]-e[0]-.30)+.65*clip((moms['core',2]-.25)*3,-1,1)+4*(bulge_width-.08)
# An edge-on disk with a bright round bulge has a larger core axis ratio;
# do not use a single global width to exclude these objects.
outerthin=clip((.33-chordqs[0])*18,-3,3); edged=mx(edged,.87*outerthin+.84*size+.32*clip(bulge,-1,1)+.5*clip((chordqs[1]-chordqs[0]-.04)*5,-1,1))
texture=clip((mirror-.075)*10,-.7,2)+.68*clip((knots-.018)*18,-.36,1.5)
face=clip((q-.38)*5,-1,1); profilesize=clip((radmean[0]-12)/5,-1.8,2.4)
spiral=1.25*blue+.70*profilesize+.15*size+.43*clump+.45*clip((bar-.15)*5,-.6,1.2)+.84*texture+.60*clip((relative_core-.52)*4,-1,1)-.15-1.5*pos(-face)
barvote=(bar-.19)*9+.75*clip((warm[0]-.12)*3,-1,1.3)+.7*clip((warm[0]-warm[2]-.07)*5,-1,1)+.20*size+.40*clip((chordqs[0]-chordqs[1]-.04)*4,-1,1)
loose=.77*blue+.62*clump+.20*size-.75*clip((moms['core',2]-.14)*4,-.75,1)+.72*clip((mirror-.12)*5,-.8,1.5)-.38
# Demand mutual evidence of distinct nuclei and significant disturbed flux:
# an unrelated field star is paired only if a substantial fraction of the
# extended light is lopsided as well. For major disturbances red pairs are
# often old stellar populations whereas blue tails have fewer compact knots.
interaction=clip((asym-.14)*9,-1.2,2.3)+2.8*irregularities[0]
merge=mn(clip((pair-.08)*22,-3.4,5), interaction+.85)+.72*interaction+.52*clip((col-.14)*3,-1,1)-.35
merge=mx(merge,1.45*interaction+1.55*clip((col-.16)*4,-1.3,1.5)-1.1)
dist=(asym-.12)*11+.50*clump+.50*blue+.52*texture-.2*pos(size)+3.6*irregularities[0]-.22
# Smooth subtypes: transition points in projected axis ratio.
smooth=[(q-.80)*9,Expr(0),(.57-q)*10];smoothmax=mx(mx(smooth[0],smooth[1]),smooth[2]);smooth=[z-smoothmax for z in smooth]
# Combine arm and stationary-ellipse evidence. High eccentricity makes any
# angular high-pass unreliable; require additional evidence to call it spiral.
wind=clip((winding-.028)*27,-.85,2.5)
flat=clip((.61-q)*6,0,1.5); spiral+=.80*wind-1.30*flat+.32*clip((bar-.16)*7,-.8,1.2)+.23
barvote+=.32+.60*clip((warm[0]-warm[2]-.06)*4,-1,1)-.45*pos(wind)+.37*clip((q-chordqs[1]-.015)*6,-1,1)
loose+=1.02+1.35*pitchsign+.73*pos(wind)+.60*clip((warm[0]-.18)*4,-1,1)*(-1)+.65*clip((relative_core-.55)*5,-1,1)
# Blue late disks are often diffuse through the center, as distinct from
# the bright, smooth, central light common in tightly wound spirals.
loose+=.52*clip((.60-core)*4,-1.5,1.5)
merge-=.58*pos((.63-q)*2.8)+.15;dist+=.40*clip((.12-col)*4,-1,1)+.60*clip((.13-asym)*(-4),-.6,1.5)+.25
edged+=.48*thindisk+.30+.60*clip((.38-chordqs[0])*6,-1,1)
bulge+=-.68+.85*clip((edgecolor-.16)*3,-1.2,1.3)+.55*clip((warm[0]-warm[2]-.055)*5,-1.2,1.2)
# Smooth early-type objects are usually red all the way out. In an Sa/Sb a
# relatively small bulge can dominate the aperture even if the disk is blue.
# Disk colour with that bulge explicitly removed gives an independent test.
rows=[]
for s1,s2 in [(4,10),(7,16)]:
    ann=unit(gauss(s2)*(1-gauss(s1))**2)-unit(np.exp(-.5*((r-50)/5)**2))
    for v in np.eye(3):rows.append(v[:,None,None]*ann)
dc=features(dense('disk_only_colors',rgb,rows));dcol=[]
for i in [0,3]:
    R,G,B=dc[i:i+3];dcol.append(ratio(R-B,.3*R+.4*G+.3*B+.025,(-.8,1.),36))
bluedisk=clip((.10-.65*dcol[0]-.35*dcol[1])*4,-1.4,1.8)
# Likewise suppress an aspect-ratio-only classification of a smooth *blue*
# feature-containing source. The disk must be resolved, not just a blue PSF.
spiral+=.76*bluedisk+.68*clip((warm[0]-dcol[0]-.05)*4,-.9,1.1)+.35*pos(profilesize)+.16
spiral=mx(spiral,1.25*blue+.90*bluedisk+.73*profilesize+.80*clump+.52*texture-1.25*flat-.45)
# Emphasize color, bulge prominence, and non-monotone radial arms for subtype;
# an amplitude ratio without absolute detection is weak evidence about pitch.
barvote=.66*barvote+3.6*(redbar-.17)+.58*clip((warm[0]-.24)*3,-1,1)-.60*pos(clump)-.38*pos(bluedisk)
loose=.63*loose+.65*clump-.65+.47*clip((.65-core)*5,-1.4,1.4)+.48*clip((.12-warm[0])*3,-1,1)
# Compact warm core + faint elongated envelope -> edge-on bulged rather than
# smooth cigar; a shallow, blue central profile instead favors a pure disk.
wing=clip((chordqs[1]-chordqs[0]-.08)*7,-1.3,1.7)
edged+=.73*wing+.55*clip((lens[0]-16)/6,-1.5,1.6)+.32*pos((.46-chordqs[0])*4)
bulge=1.45*(edgecolor-.235)+1.6*(warm[0]-dcol[0]-.04)+1.95*(core-.63)+.85*wing+1.6*(bulge_width-.10)+.48*(e[2]-e[0]-.26)
# Weak asymmetry in a singly-peaked red galaxy is more likely to be an
# astrometric offset; high odd multipoles and a warped outer contour support
# the disturbed hypothesis even for an isolated galaxy with a central bulge.
dist+=.70*bluedisk+6.*(contourwarp[0]-.099)+6.*(m3-.065)+3.7*(tailodd-.045)-.35*flat-.28
merge+=.55*clip((pair-.09)*9,-1.2,1.8)+.75*clip((col-.10)*3,-1.5,1.3)+3*(tailodd-.035)-.23
# Safeguards: a narrow, well-described ellipse lacks positive spiral-arm
# evidence even when a linear bar filter responds strongly. Conversely a
# blue galaxy with irregular isophotes and a twist need not be very large.
nonellipse=clip((ringerrors[0]-.082)*8,-.60,1.45); twist=clip((contour_twist-.16)*4,-.64,1.4)
spiral+=.70*nonellipse+.25*twist-.45-1.7*flat-.42*pos((.60-chordqs[0])*4)
dist+=.60*nonellipse+.45*pos(twist)+.64*pos((asym-.13)*5)+.25*bluedisk+.24
merge=.45*merge+.55*clip((realpair-.059)*25,-2.2,5)+.24*interaction+.22*clip((col-.1)*4,-1,1)
# A strong two-arm signal with a faint center is sufficient for the late
# spiral class; bars instead require coherent inner isophote elongation.
loose+=.48+.40*pos(clump)+.74*clip((.66-core)*5,-.8,1.4)+.30*pos(nonellipse)
barvote-=.35*pos(nonellipse)+.32*pos(bluedisk);barvote+=.28*pos(twist)
# In an edge-on bulged galaxy a faint linear disk can be hidden underneath
# a dominant compact nucleus. Look at major axis extent and the change in
# contour flattening, not just the minor/major of the overlaid components.
embedded=clip((.48-chordqs[0])*10,-2.4,1.8)+.93*wing+.83*size+.62*thindisk+.38*clip((edgecolor-.23)*3,-1,1)-.72
edged=mx(edged,embedded)+.28*pos(bulge);bulge+=.35*clip((warm[0]-.22)*4,-1.2,1.4)
spirals=[barvote,Expr(0),loose];smx=mx(mx(spirals[0],spirals[1]),spirals[2]);spirals=[spiral+z-smx for z in spirals]
# Merging and disturbed objects can themselves have high spiral or disk
# evidence. Blend rather than requiring a separate purely exclusive family.
merge=merge+.25*pos(spiral);dist=dist+.24*pos(spiral)+.32*pos(edged)
scores=[dist,merge]+smooth+spirals+[edged-pos(bulge),edged-pos(-bulge)]

# Lower affine expressions to groups of dense layers and adds grouped by
# depth, retaining genuine branches instead of allocating image-sized dense
# identity matrices. Every weight here is from the formulas above.
locations={};front_i=0
for j,d in enumerate(depths):
    if d==0:locations[j]=front[front_i];front_i+=1
# Omit unused analytic gates; split wide independent blocks to avoid storing
# millions of structural zeros for unrelated measurements in the same matrix.
needed=set().union(*(z.t for z in scores));bydepth={}
for i,depth,expr in reversed(rs):
    if i in needed:
        needed.update(expr.t);bydepth.setdefault(depth,[]).append((i,expr))
def compile_group(prefix,exprs):
    bysrc={};b=np.array([e.c for e in exprs],dtype='f')
    for row,ex in enumerate(exprs):
        for i,v in ex.t.items():
            src,col=locations[i]
            if src not in bysrc:bysrc[src]=np.zeros((len(exprs),D[src]),'f')
            bysrc[src][row,col]+=v
    chunks=[]
    for j,(src,mat) in enumerate(bysrc.items()):chunks.append(dense(prefix+'_from_'+src,src,mat,b if not j else 0))
    n=chunks[0] if len(chunks)==1 else layer(prefix+'_add','add',chunks);D[n]=len(exprs);return n
for d,items in sorted(bydepth.items()):
    items=sorted(items,key=lambda pair:pair[0]);chunk=384
    for start in range(0,len(items),chunk):
        part=items[start:start+chunk];tag='%02d_%02d'%(d,start//chunk)
        n=compile_group('logic'+tag,[z for _,z in part]);act=layer('gate'+tag,'relu',n);D[act]=len(part)
        for k,(i,_) in enumerate(part):locations[i]=(act,k)
last=compile_group('class_scores',scores)
needed={last};kept=[]
for l in reversed(layers):
    if l['name'] in needed:kept.append(l);needed.update(l['inputs'])
layers=kept[::-1];weights={k:v for k,v in weights.items() if k.rsplit('.',1)[0] in needed}
(out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n');np.savez(out/'weights.npz',**weights)
