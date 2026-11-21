"""Analytic weights for a multiscale, part-based person detector.

All coordinates below are in generic object coordinates (or in feature-map
cells). No pictures, pretrained parameters, or auxiliary files are used.
"""
import json
import sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
L = []
W = {}

def layer(name, op, *ins, **kw):
    L.append(dict(name=name, op=op, inputs=list(ins), **kw))
    return name

def conv(name, x, w, b=0, stride=1, padding=0):
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 2: w = w[:, :, None, None]
    W[name+'.W'] = w
    W[name+'.b'] = np.broadcast_to(b,(len(w),)).astype(np.float32).copy()
    return layer(name, 'conv', x, out=len(w), kernel=list(w.shape[2:]), stride=stride, padding=padding)

def dense(name,x,w,b=0):
    w = np.asarray(w, np.float32)
    W[name+'.W'] = w
    W[name+'.b'] = np.broadcast_to(b,(len(w),)).astype(np.float32).copy()
    return layer(name,'dense',x,out=len(w))

def pool(name,x,mode,k,s):
    if isinstance(k,int): k=[k,k]
    if isinstance(s,int): s=[s,s]
    return layer(name,mode+'pool',x,size=k,stride=s)

def rel(name,x): return layer(name,'relu',x)
def add(name,*xs): return layer(name,'add',*xs)

def scalar_row(n, d):
    r=np.zeros(n)
    for i,v in d.items(): r[i]=v
    return r

# Color-opponent coordinates and soft half-space tests. Skin is not a
# brightness band: allow a range of intensities along a reddish-orange ray.
r,g,b=np.eye(3)
y=.30*r+.58*g+.12*b
v=r-g; u=g-b
wcolor=[]; bcolor=[]
def test(w, bias=0):
    i=len(wcolor);wcolor.append(w);bcolor.append(bias);return i
# Common penalties for low saturation, darkness, high chroma, cast.
lowred=test(-v+.038*y,.013)
toored=test(v-.60*y,-.045)
blue=test(-u-.16*v,-.007)
yellow=test(u-1.31*v,-.014)
dark=test(-y,.078)
# More specific, rosy complexion relative to tan / wood / ground hues.
palered=test(-v+.071*y,.021)
golden=test(u-.92*v,-.007)
# Positive evidence of strongly chromatic clothing, especially cool hues
# and vivid red. These are soft memberships, not RGB lookup tables.
royal=test(b-.66*g-.34*r,.0)
scarlet=test(r-.68*g-.32*b,-.14)
magenta=test(.53*r+.47*b-g,-.047)
sward=test(g-.63*r-.37*b,-.02)
cyansky=test(.59*b+.41*g-r,-.045)
yellows=test(.5*r+.5*g-b,-.085)
redgreen=test(v); greenred=test(-v)
greenblue=test(u); bluegreen=test(-u)
white=test(y,.0)
redblue=test(r-b); bluered=test(b-r)
# Very dark grey must not be identified as brown just because the
# absolute color differences are small.
achromatic=test(-v,.019)
# A spatially averaged pixel at stride two, before color tests.
base=pool('two_pixel_means','image','avg',2,2)
h=rel('opponent_hinges',conv('color_planes',base,wcolor,bcolor))
# raw maps 0=skin 1=rosy skin 2=luma 3=vivid color 4=green 5=blue sky
# 6=warm color 7=R-G 8=G-B, 9=short-wave (blue) 10=vivid red
D=len(wcolor)
rawD=[{lowred:-32,toored:-12,blue:-26,yellow:-28,dark:-17,achromatic:-43},
      {palered:-31,toored:-12,blue:-28,golden:-28,dark:-17,achromatic:-43},
      {white:1}, {royal:3.7,scarlet:2.0,magenta:3.6},
      {sward:4.5},{cyansky:4},{yellows:3},
      {redgreen:1,greenred:-1},{greenblue:1,bluegreen:-1},
      {royal:4},{scarlet:2.6}]
raw_bias=[1,1,0,0,0,0,0,0,0,0,0]
# Clip the two complexion maps below at zero, preserve other (nonnegative)
# maps; color axes used for contours below come directly from RGB instead.
raw=rel('material_membership',conv('materials',h,[scalar_row(D,d) for d in rawD],raw_bias))
NR=len(rawD)
# Compact face input: mix broad and narrow chromaticity gates, plus luma.
parts=conv('complexion_and_light',raw,[scalar_row(NR,{0:.6,1:.4}),scalar_row(NR,{2:1})])
# Include intermediate octave for spatial scale coverage without a huge bank
# of large kernels at full resolution.
scales={2:(parts,48,2)}
scales[6]=(pool('six_pixel_means',parts,'avg',3,3),16,6)
scales[4]=(pool('four_pixel_means',parts,'avg',2,2),24,4)
scales[8]=(pool('eight_pixel_means',scales[4][0],'avg',2,2),12,8)
scales[12]=(pool('twelve_pixel_means',scales[6][0],'avg',2,2),8,12)

# Each kernel contrasts (i) the area inside an elliptical face with both
# flanks and a collar, (ii) the upper face with a hair/cap area above it,
# (iii) brighter cheeks/nose with the eye sockets, and sometimes mouth.
# Generate area masks with subcell quadrature, so small ovals change smoothly.
def region(k,x0,y0,rx,ry,angle=0):
    a=np.deg2rad(angle); c=np.cos(a); s=np.sin(a)
    iy,ix=np.mgrid[:k,:k]-k//2
    A=np.zeros((k,k))
    for dx in [-.4,-.2,0,.2,.4]:
        for dy in [-.4,-.2,0,.2,.4]:
            X=c*(ix+dx)+s*(iy+dy); Y=-s*(ix+dx)+c*(iy+dy)
            z=((X-x0)/rx)**2+((Y-y0)/ry)**2
            A+=np.maximum(0,1-z)**.65
    return A/(A.sum()+1e-16)

def gauss(k,x0,y0,rx,ry,angle=0):
    iy,ix=np.mgrid[:k,:k]-k//2
    a=np.deg2rad(angle); c=np.cos(a); s=np.sin(a)
    X=c*ix+s*iy; Y=-s*ix+c*iy
    # Cell aperture places a lower bound on the bandwidth.
    A=np.exp(-.5*((X-x0)**2/(rx*rx+.095)+(Y-y0)**2/(ry*ry+.095)))
    return A/A.sum()

face_scores=[]
face_regions=[]
for q,(inp,N,pix) in scales.items():
    k=9 if pix<=6 else 7
    bank=[]; bias=[]
    for wx,hy,tilt in [(2.4,3.25,0),(3.9,4.6,0),(2.8,5.3,-12),(2.8,5.3,12),(4.9,3.85,0)]:
        def G(x,y,w,h):return gauss(k,x*wx,y*hy,w*wx,h*hy,tilt)
        def R(x,y,w,h):return region(k,x*wx,y*hy,w*wx,h*hy,tilt)
        core=R(0,0,.50,.48)
        sides=.5*(G(-.70,-.01,.17,.28)+G(.70,-.01,.17,.28))
        collar=G(0,.78,.33,.19); hair=G(0,-.57,.31,.115)
        eyes=.5*(G(-.235,-.14,.10,.069)+G(.235,-.14,.10,.069))
        cheeks=.5*(G(-.18,.085,.14,.10)+G(.18,.085,.14,.10))
        brow=G(0,-.36,.22,.10); mouth=G(0,.28,.19,.065)
        nose=G(0,-.055,.095,.18)
        face_light=.45*cheeks+.24*brow+.18*nose+.13*G(0,.40,.20,.072)
        features=face_light-.76*eyes-.24*mouth
        crown=core-hair
        # A stronger local complexion contrast is needed on wood or another
        # large warm background. Dark hair is useful but not compulsory.
        bank.append(np.stack([.99*core-.45*sides-.23*collar-.19*hair,
                              .46*crown+.78*features]));bias.append(0)
        if wx>=3.8:
            # Monochrome/low-saturation face: favor eye-pair and cheeks over
            # a single top/bottom contrast. This detector peaks on larger faces.
            bank.append(np.stack([.37*core-.17*sides-.10*collar,
                                  .48*crown+1.72*features]));bias.append(.04)
    name='face'+str(pix); stride=2 if pix==2 else 1
    hits=conv(name,inp,bank,bias,stride,k//2)
    outN=(N-1)//stride+1
    # Low-resolution local and image-wide maxima; modest center preference.
    # First pool into a coarse lattice. Pooling over the shape variants is
    # implemented as a second spatial max after reordering by reshape.
    def max_channels(name, maps, count, side):
        t=layer(name+'_rows','reshape',maps,shape=[1,count,side*side]);t=pool(name+'_choose',t,'max',[count,1],[count,1]);return layer(name+'_map','reshape',t,shape=[1,side,side])
    m=max_channels(name+'_variants',hits,len(bank),outN)
    cells=pool(name+'_districts',m,'max',outN//4,outN//4)
    # grids in this design are exactly divisible by four
    pos=np.zeros((1,1,4,4))
    pos[0,0,:,:]=np.array([[.12,.064,.064,.12],[.055,0,0,.055],[.065,.015,.015,.065],[.16,.105,.105,.16]])
    # Add the positional prior via a small dense, rather than zero-padding a
    # conv over the image (which would make fake edges at image boundaries).
    t=dense(name+'_prior',cells,np.eye(16),-pos.reshape(16));t=layer(name+'_grid','reshape',t,shape=[1,4,4])
    face_regions.append(t)
    peak=pool(name+'_peak',t,'max',4,4);
    face_scores.append(peak)

# Directional color edges for a HOG-like silhouette detector. Use color
# opponent gradients in addition to light: sportswear can be isoluminant.
# A contrast-limiting hinge later makes one long, very hard edge count less
# than a set of correctly placed moderate ones.
dr=np.array([[1,2,1]])/4
kw=[]; dirs=4
for ang in np.arange(dirs)*np.pi/dirs:
    # project a smoothed gradient; orientation refers to the contour NORMAL
    dx=np.zeros((3,3)); dx[:,0]=-dr;dx[:,2]=dr
    dy=dx.T
    kernel=dx*np.cos(ang)+dy*np.sin(ang)
    for axis in [y, .58*(r-b)]:
        for sign in [1,-1]:kw.append(sign*axis[:,None,None]*kernel)
ed=rel('edge_halfwaves',conv('directional_differences',base,kw,padding=0))
# 46x46 -> pad just at the later small cells, avoiding fake input boundaries
mix=np.zeros((dirs,len(kw)))
for i in range(dirs):mix[i,4*i:4*i+4]=[1,1,.65,.65]
ed=conv('edge_strength',ed,mix)
# A HOG channel measures the excess amplitude in this orientation above
# the competing orientation. Produces a sparse directional representation.
A=np.eye(dirs)*1.35-np.roll(np.eye(dirs),2,axis=1)*.73-(np.roll(np.eye(dirs),1,axis=1)+np.roll(np.eye(dirs),-1,axis=1))*.15
ed=conv('select_orientations',ed,np.concatenate([A,A]),[0]*4+[-.17]*4)
ed=rel('oriented_hinges',ed)
ed=conv('bounded_contours',ed,np.concatenate([np.eye(4),-.82*np.eye(4)],axis=1)*6)

# Generic silhouettes drawn from ellipses and tapered limb segments.
# Make orientation masks by analytically differentiating a smooth occupancy
# image of the union and integrating its gradients into feature cells.
# The hip/hand/foot endpoints below describe poses, NOT image coordinates.
def person_outline(X,Y,pose):
    # in units of standing height, head centered just above shoulders
    def ellipse(x0,y0,rx,ry,theta=0):
        c=np.cos(theta);s=np.sin(theta); x=X-x0;y=Y-y0
        return (1-np.sqrt(((x*c+y*s)/rx)**2+((y*c-x*s)/ry)**2))*min(rx,ry)
    def limb(p,q,r0,r1):
        dx=q[0]-p[0];dy=q[1]-p[1]
        t=np.clip(((X-p[0])*dx+(Y-p[1])*dy)/(dx*dx+dy*dy),0,1)
        return r0*(1-t)+r1*t-np.hypot(X-p[0]-dx*t,Y-p[1]-dy*t)
    shift= .034 if pose in ('stride','throw') else 0
    trunk=ellipse(0,-.118,.116,.200,shift*3)
    items=[trunk,ellipse(shift,-.406,.068,.089,-.08),limb((shift,-.34),(0,-.268),.039,.052),ellipse(-.004,.055,.100,.083)]
    if pose=='stand':
        arms=[[(-.092,-.242),(-.132,-.095),(-.139,.042)],[(.092,-.242),(.134,-.081),(.13,.054)]]
        legs=[[(-.053,.063),(-.065,.253),(-.084,.465)],[(.051,.063),(.069,.262),(.081,.459)]]
    elif pose=='stride':
        arms=[[(-.085,-.245),(-.171,-.11),(-.227,-.16)],[(.094,-.245),(.187,-.174),(.125,-.106)]]
        legs=[[(-.041,.069),(-.145,.239),(-.233,.405)],[(.047,.056),(.174,.213),(.112,.408)]]
    elif pose=='throw':
        arms=[[(-.077,-.240),(-.174,-.32),(-.228,-.415)],[(.093,-.237),(.208,-.198),(.308,-.23)]]
        legs=[[(-.047,.056),(-.144,.235),(-.229,.425)],[(.060,.058),(.141,.24),(.175,.443)]]
    else:
        arms=[[(-.088,-.242),(-.126,-.125),(-.058,-.029)],[(.087,-.242),(.122,-.127),(.041,-.063)]]
        legs=[[(-.05,.057),(-.177,.118),(-.189,.30)],[(.056,.057),(.203,.104),(.181,.318)]]
    for pts in arms: items.extend([limb(pts[0],pts[1],.036,.028),limb(pts[1],pts[2],.027,.020)])
    for pts in legs:items.extend([limb(pts[0],pts[1],.058,.036),limb(pts[1],pts[2],.035,.023),ellipse(pts[2][0]+.012,pts[2][1]+.013,.044,.020)])
    return np.maximum.reduce(items)

def contour_kernel(k,height,aspect,pose,mirror):
    ss=12
    axis=(np.arange(k*ss)+.5)/ss-k/2
    Y,X=np.meshgrid(axis,axis,indexing='ij'); Z=person_outline(X/(aspect*height)*mirror,Y/height,pose)
    z=np.tanh(Z/.021); dy,dx=np.gradient(z); strength=np.hypot(dx,dy)
    phi=np.arctan2(dy,dx); masks=[]
    for a in np.arange(4)*np.pi/4:
        masks.append((strength*np.maximum(0,1-np.arccos(np.clip(np.abs(np.cos(phi-a)),0,1))/(np.pi/4))).reshape(k,ss,k,ss).mean((1,3)))
    masks=np.array(masks)
    # Enforce a contribution from the head/shoulders as well as long legs.
    rows=np.arange(k)-k//2
    for group,budget in [(rows<-.23*height,.28),((rows>=-.23*height)&(rows<.10*height),.36),(rows>=.10*height,.36)]:
        masks[:,group]*=budget/(masks[:,group].sum()+1e-9)
    # Adjacent backgrounds with the same orientations (poles, repetitive
    # texture) lower the match. Same-location wrong orientations do too.
    bg=.5*(np.roll(masks,2,axis=2)+np.roll(masks,-2,axis=2))
    return masks-.55*bg-.38*np.roll(masks,2,axis=0)

body_scores=[]
for label,step,n in [('mid',2,23),('far',1,46),('near',4,11)]:
    inp=pool(label+'_hog_cells',ed,'avg',step,step) if step!=1 else ed
    k=11
    K=[]
    for height,aspect,pose in [(8,1,'stand'),(8.8,1.55,'stand'),(6.1,1.35,'stand'),(9,.8,'stand'),(8,1.2,'stride'),(8,1.45,'throw'),(8.5,1.20,'sit')]:
        for mirror in ([1,-1] if pose in ('stride','throw') else [1]): K.append(contour_kernel(k,height,aspect,pose,mirror))
    # Valid correlation near image boundaries can still vote for truncated
    # figures, as all orientation coordinates vanish outside the picture.
    stride=2 if step==1 else 1
    t=conv(label+'_figures',inp,K,0,stride,k//2); N=(n-1)//stride+1
    t=max_channels(label+'_pose',t,len(K),N)
    t=pool(label+'_silhouette_regions',t,'max',N//3,N//3)
    t=dense(label+'_centered_silhouette',t,np.eye(9),[-.046,-.021,-.046,-.022,0,-.022,-.055,-.031,-.055]);t=layer(label+'_body_grid','reshape',t,shape=[1,3,3]);body_scores.append(pool(label+'_strongest_body',t,'max',3,3))

# Relational figure cue: a head-sized compact skin patch on a torso that
# differs from both flanks in luma and/or chromaticity. This also works on
# rear views with very little visible skin, if the clothing boundary is good.
figure_maps=[]
upper_scores=[]
for pix,step in [(2,1),(4,2),(6,3),(8,4)]:
    N=48//step; prefix='relation'+str(pix)
    skin=scales[pix][0]
    rgb=base if step==1 else pool(prefix+'_rgb',base,'avg',step,step)
    # Head at upper center of a window. A single shared head contrast and
    # multiple center-surround contrasts for the upper and lower torso.
    k=9;c=k//2; stride=2 if step==1 else 1
    def G(x,y,wx,hy):return gauss(k,x,y,wx,hy)
    head=G(0,-2.15,.49,.59); neck=G(0,-.99,.48,.45)
    hair=G(0,-3.21,.55,.37); hf=.5*(G(-1.53,-2.17,.49,.65)+G(1.53,-2.17,.49,.65)); chin=G(0,-.37,.7,.53)
    sk=np.stack([.84*head+.17*neck-.41*hf-.20*hair-.22*chin,.43*(head-hair)])
    sk=conv(prefix+'_head',skin,[sk],0,stride,c)
    ks=[]
    for flip in [-1,1]:
        center=G(0,.62,.62,1.28); side=G(2.45*flip,.63,.63,1.22)
        for axis in [y,.74*(r-b),.70*(g-.55*r-.45*b)]:
            ker=axis[:,None,None]*(center-side)
            ks.extend([ker,-ker])
    # A second score recognizes head-and-shoulder silhouettes without any
    # skin cue. A dark oval above the torso has two sides and a top; the
    # neck-to-shoulder step should have the SAME contrast polarity as the
    # side of the shirt. Otherwise the response is probably a pole/stripe.
    # Use a valid convolution for the silhouette, so padding cannot supply
    # the required background on the sides of a bright object.
    if pix<=6:
        sil=[]
        for sign in [-1,1]:
            for width in [1,1.5]:
                hd=G(0,-2.12,.46*width,.61)
                ht=G(0,-3.93,.63*width,.38)
                temples=.5*(G(-1.39*width,-2.19,.36,.63)+G(1.39*width,-2.19,.36,.63))
                trunk=G(0,.90,.65*width,1.14)
                flanks=.5*(G(-2.2*width,1.0,.42,1.02)+G(2.2*width,1.0,.42,1.02))
                upper_arm=.5*(G(-.96*width,.11,.33,.55)+G(.96*width,.11,.33,.55))
                over_arm=.5*(G(-.96*width,-1.37,.31,.43)+G(.96*width,-1.37,.31,.43))
                sil.extend([y[:,None,None]*(ht-hd),y[:,None,None]*(temples-hd),sign*y[:,None,None]*(trunk-flanks),sign*y[:,None,None]*(upper_arm-over_arm)])
        sh=conv(prefix+'_outline_tests',rgb,sil,0,stride,0)
        # Hinge representation of bounded tests, not a plain linear Haar
        # classifier: no single implausibly strong edge wins the test.
        sh=rel(prefix+'_outline_ramp',conv(prefix+'_outline_bands',sh,np.tile(np.eye(16),(2,1)),[.015]*16+[-.21]*16));K=np.zeros((4,32))
        for i in range(4):K[i,4*i:4*i+4]=[.20,.31,.27,.22]
        K[:,16:]=-K[:,:16];sh=conv(prefix+'_outline_agreement',sh,K*4)
        n=(N-k)//stride+1;sh=max_channels(prefix+'_select_outline',sh,4,n)
        # Avoid the maximum over an entire cluttered image having the same
        # weight as strong evidence in multiple smaller neighborhoods.
        sh=pool(prefix+'_regional_outlines',sh,'max',n-2*(n//3),n//3)
        sh=dense(prefix+'_summarize_outlines',sh,np.eye(9),[-.12,-.055,-.12,-.06,0,-.06,-.14,-.08,-.14]);sh=layer(prefix+'_outline_grid','reshape',sh,shape=[1,3,3]);upper_scores.append(pool(prefix+'_peak_outline',sh,'max',3,3))
    tc=rel(prefix+'_half',conv(prefix+'_cloth_edges',rgb,ks,0,stride,c))
    mat=np.zeros((4,12));mat[0,:6]=1;mat[1,6:]=1;mat[2,:6]=1;mat[2,6:]=-1;mat[3]=-mat[2]
    tc=rel(prefix+'_bilateral',conv(prefix+'_left_right',tc,mat,[-.0,-.0,0,0]))
    # Half the sum minus half the absolute difference is the smaller
    # response. Reject strong contours on only one side of the torso.
    tc=conv(prefix+'_weaker_boundary',tc,[[.5,.5,-.5,-.5]])
    # Saturated clothing should not suffice on its own.
    cap=rel(prefix+'_excess',conv(prefix+'_cap',tc,[[1]],[-.44])); cap=conv(prefix+'_subtract_excess',cap,[[-1]])
    tc=add(prefix+'_limited',tc,cap)
    tc=conv(prefix+'_support',tc,[[.79]])
    f=add(prefix+'_combined',sk,tc); n=(N-1)//stride+1
    f=pool(prefix+'_quadrants',f,'max',n//4,n//4)
    f=dense(prefix+'_centered',f,np.eye(16),[-.13,-.075,-.075,-.13,-.058,0,0,-.058,-.07,-.022,-.022,-.07,-.16,-.11,-.11,-.16]);f=layer(prefix+'_grid','reshape',f,shape=[1,4,4]);figure_maps.append(pool(prefix+'_peak',f,'max',4,4))

# Readout utilities: pack scalar branches along a channel axis by adding
# one-hot embeddings; the final response can then pool across scales.
def pack(name,maps):
    d=len(maps);out=[]
    for i,m in enumerate(maps):out.append(conv(name+'_slot'+str(i),m,np.eye(d)[:,i:i+1]))
    return add(name,*out)

def max_across(name,items):
    z=pack(name+'_slots',items); z=layer(name+'_list','reshape',z,shape=[1,len(items),1]);return pool(name+'_max',z,'max',[len(items),1],[len(items),1])

bestface=max_across('face_scale',face_scores)
bestfig=max_across('joint_scale',figure_maps)
bestbody=max_across('contour_scale',body_scores)

# Context. The distributions of skin-like color in face/hand-sized patches
# and of vivid clothing in the middle of a scene are useful tie-breakers.
# Broad warm surfaces should NOT constitute positive evidence of people.
coarse=pool('material_tiles',raw,'avg',8,8)
# weighted averages for objects near the middle, and wider background.
gy,gx=np.mgrid[:6,:6]; ctr=np.exp(-.5*(((gx-2.5)/1.36)**2+((gy-2.56)/1.48)**2));ctr/=ctr.sum()
bg=1-np.exp(-.5*(((gx-2.5)/1.50)**2+((gy-2.55)/1.50)**2));bg/=bg.sum()
allpos=np.ones((6,6))/36
upper=np.exp(-.5*((gy-.7)/.95)**2);upper/=upper.sum();lower=upper[::-1]

def spatial(ch,dist):
    out=np.zeros((NR,6,6))
    for i,c in ch.items():out[i]=c*dist
    return out
skinmix={0:.55,1:.45}
scene=np.array([spatial(skinmix,ctr),spatial(skinmix,bg),spatial({3:1},ctr),spatial({3:1},bg),spatial({4:1},lower),spatial({5:1},upper),spatial({6:1},bg),spatial({9:1},ctr-bg),spatial({10:1},ctr-bg)])
scene=dense('scene_composition',coarse,scene.reshape(len(scene),-1)); scene=layer('scene_vector','reshape',scene,shape=[9,1,1])
# Smoothly capped/triangular preferences. Favor small and moderate,
# localized skin coverage over a warm wash. Green-field / sky evidence is
# weak: increasing the field area ought not by itself to imply a person.
H=[];HB=[];out=[]
def hinge(co,bias,gain):
    H.append(scalar_row(9,co));HB.append(bias);out.append(gain)
hinge({0:1},0,1.3);hinge({0:1},-.13,-1.3)
hinge({0:1,1:-1},0,.90);hinge({0:1,1:-1},-.25,-.90)
hinge({1:1},-.13,-.70);hinge({1:1},-.51,.70)
hinge({2:1},-.053,.68);hinge({2:1},-.33,-.68)
hinge({2:1,3:-1},0,.81);hinge({2:1,3:-1},-.21,-.81)
hinge({3:1,2:-1},0,-.33);hinge({3:1,2:-1},-.20,.33)
hinge({4:1},-.055,.20);hinge({4:1},-.25,-.20)
hinge({7:1},0,.48);hinge({7:1},-.20,-.48)
hinge({8:1},0,.36);hinge({8:1},-.20,-.36)
hinge({6:1},-.14,-.26);hinge({6:1},-.48,.26)
scene=rel('context_hinges',conv('context_rules',scene,H,HB));context=conv('context_vote',scene,[out],-.16)

# Spatial context independent of skin: continuous open sky, a smooth field
# or snow behind a more colorful subject, and a crowd of small multicolored
# details are typical of action photos. A continuous yellow-brown wall, a
# large tan foreground, and large straight architectural contours are not.
# Use broad masks for these clues, never treating a scene as decisive.
mn=pool('context_patch_means',base,'avg',3,3)
# Centered residual in a 3x3 neighborhood. Rectify both signs.
low=np.ones((3,3))/9;hp=-low;hp[1,1]+=1
filt=[]
for a in [y, r-g, b-g]:
    for s in [1,-1]:filt.append(s*a[:,None,None]*hp)
texture=rel('patch_detail',conv('patch_contrasts',mn,filt,padding=0))
texture=conv('detail_bands',texture,[[1,1,0,0,0,0],[0,0,1,1,1,1]])
tiles=pool('detail_regions',texture,'avg',2,2)
Y7,X7=np.mgrid[:7,:7];central=np.exp(-.5*(((X7-3)/1.3)**2+((Y7-3)/1.6)**2));central/=central.sum()
back=1-central/central.max();back/=back.sum()
# A restricted vicinity makes small red/pink/blue items much more relevant
# than a blue sky. Crowd-like colored texture has little relation to skin.
detail=dense('subject_detail',tiles,[np.stack([central-back,central*0]).ravel(),np.stack([central*0,central]).ravel(),np.stack([central*0,back]).ravel()]);detail=layer('detail_vector','reshape',detail,shape=[3,1,1])
# Extra color context rules work on coarse regional proportions rather than
# on a color histogram: sky above green turf, brightly dressed figures
# against snow/water, multiple cool/red colors in the same region.
cc=conv('event_colors',raw,[scalar_row(NR,{i:1}) for i in [2,4,5,6,9,10,0,1]])
cc=pool('event_tiles',cc,'avg',8,8)
snow=rel('bright_ground',conv('bright_surface',cc,[[1,-.25,-.20,-.22,-.19,-.27,-.065,-.065]],[-.72]))
snow=dense('lower_bright_area',snow,[lower.ravel()]);snow=layer('snow_fraction','reshape',snow,shape=[1,1,1])
# Select for both red and blue local accents while retaining a linear
# component. Their co-occurrence is less typical of vegetation or furniture.
event=dense('event_layout',cc,np.array([np.stack([dist if j==i else np.zeros_like(dist) for j in range(8)]) for i,dist in [(1,lower),(2,upper),(4,ctr),(5,ctr),(4,bg),(5,bg),(1,ctr),(3,lower),(6,lower),(7,upper)]]).reshape(10,-1));event=layer('event_vector','reshape',event,shape=[10,1,1])
V=[];B=[];G=[]
def ev(co,bias,gain):V.append(scalar_row(10,co));B.append(bias);G.append(gain)
ev({0:1},-.045,.51);ev({0:1},-.22,-.51)
ev({1:1},-.052,.16);ev({1:1},-.23,-.16)
ev({2:1},-.063,.48);ev({2:1},-.25,-.48)
ev({3:1},-.041,.42);ev({3:1},-.24,-.42)
ev({2:1,4:-1},.0,.28);ev({2:1,4:-1},-.22,-.28)
ev({3:1,5:-1},.0,.20);ev({3:1,5:-1},-.22,-.20)
ev({8:1,9:-1},-.16,-.28);ev({8:1,9:-1},-.58,.28)
ev({7:1},-.18,-.35);ev({7:1},-.50,.35)
event=rel('event_hinges',conv('event_gates',event,V,B));event=conv('event_vote',event,[G],-.075)
detail=conv('detail_vote',detail,[[.65,1.5,.95]],[-.061]);snow=conv('snow_vote',snow,[[.82]])
context=add('scene_support',context,detail,event,snow)
bestupper=max_across('bust_scale',upper_scores)

# Full-face confirmation. Here the face parts are tested separately, rather
# than adding them as image filters. This matters because an extremely dark
# area above a bright rectangle is not by itself good evidence. Each Haar
# comparison advances a bounded ramp. Above the top of the ramp, making a
# part still darker does not compensate for other parts that are missing.
confirmed=[]
for pix in [2,4,6,8]:
    inp,N,_=scales[pix]
    k=11 if pix<=4 else 9
    filters=[];lowerb=[];upperb=[];weightsets=[]
    for width,height,angle in [(4.8,5.9,0),(6.4,7.2,-12),(6.4,7.2,12),(4.05,7.0,0),(6.8,5.6,0)]:
        def G(X,Y,R,H):return gauss(k,X*width,Y*height,R*width,H*height,angle)
        zero=np.zeros((k,k)); face=region(k,0,0,.45*width,.47*height,angle)
        cheeks=G(0,.083,.28,.112); forehead=G(0,-.32,.22,.076);chin=G(0,.378,.16,.075);lip=G(0,.233,.135,.048);bridge=G(0,-.094,.078,.108)
        eyeL=G(-.218,-.135,.100,.052);eyeR=G(.218,-.135,.100,.052)
        eyes=.5*(eyeL+eyeR)
        left=G(-.60,-.024,.12,.21);right=G(.60,-.024,.12,.21);top=G(0,-.505,.265,.075);below=G(0,.68,.32,.13)
        terms=[(zero,.44*forehead+.56*G(-.218,.055,.12,.073)-eyeL,.008,.085,.115),
               (zero,.44*forehead+.56*G(.218,.055,.12,.073)-eyeR,.008,.085,.115),
               (zero,bridge-eyes,.008,.079,.087),
               (zero,.61*cheeks+.39*chin-lip,.005,.063,.080),
               (zero,forehead-top,.015,.18,.080),
               (zero,face-.5*(left+right),.005,.18,.035),
               (face,zero,.08,.66,.130),
               (face-.36*left-.36*right-.28*below,zero,.02,.47,.180),
               (.5*(G(-.19,.067,.12,.11)+G(.19,.067,.12,.11))-.55*top-.45*below,zero,.025,.59,.090),
               (zero,cheeks-eyes,.005,.089,.088)]
        for sk,br,lo,hi,weight in terms:
            filters.append(np.stack([sk,br])/(hi-lo));lowerb.append(-lo/(hi-lo));upperb.append(-hi/(hi-lo))
        weightsets.append([e[-1] for e in terms])
    prefix='anatomy'+str(pix); stride=2 if pix==2 else 1
    # Valid windows: crop out unsupported tests at the edge instead of
    # interpreting the image boundary as a dark hairline.
    t=conv(prefix+'_measure',inp,filters,stride=stride);C=len(filters)
    t=conv(prefix+'_limits',t,np.tile(np.eye(C),(2,1)),lowerb+upperb);t=rel(prefix+'_hinges',t)
    sel=np.zeros((len(weightsets),2*C));q=0
    for i,ww in enumerate(weightsets):sel[i,q:q+len(ww)]=ww;q+=len(ww)
    sel[:,C:]=-sel[:,:C];t=conv(prefix+'_agreement',t,sel)
    n=(N-k)//stride+1;t=max_channels(prefix+'_aspects',t,len(weightsets),n)
    m=3 if n>=6 else 2;s=n//m
    t=pool(prefix+'_areas',t,'max',n-(m-1)*s,s)
    pos=np.array([[-.095,-.047,-.095],[-.050,0,-.050],[-.12,-.07,-.12]]) if m==3 else np.zeros((m,m))
    t=dense(prefix+'_composition',t,np.eye(m*m),pos.ravel());t=layer(prefix+'_grid','reshape',t,shape=[1,m,m]);confirmed.append(pool(prefix+'_peak',t,'max',m,m))
confirmed=max_across('anatomical_face',confirmed)
# Multiple face-sized peaks in distinct regions are useful for groups and
# people at a table. Exclude weaker votes rather than counting every patch.
fw=rel('neighboring_face_excess',conv('regional_face_test',add('face_neighborhoods',*face_regions),[[1/len(face_regions)]],[-.43]));fv=pool('group_neighborhood_mean',fw,'avg',4,4);fv=conv('group_vote',fv,[[.95]],[-.045])
context=add('all_context',context,fv)
# Vote in units of match quality. The decision level is for a partial
# figure, not a textbook face AND a complete visible body. Reference levels
# are specified per cue: e.g. half of the unit complexion contrast, or a
# half-saturated dark-head/shoulder response. No weights depend on checks.
partscores=pack('part_scores',[bestface,bestfig,bestbody,bestupper,context,confirmed]);P=np.eye(6)
M=[P[0],P[1],P[2],P[2],P[3],P[4],-P[4],P[0],P[1],P[5],P[5]]
t=rel('vote_hinges',conv('vote_terms',partscores,M,[0,0,0,-.44,0,0,0,-.91,-1.0,0,-.72]));
gains=[.37,.40,.40,-.30,.29,1,-1,.22,.22,.86,.56]
reference=.37*.55+.40*.60+.40*.24+.29*.54+.86*.66
# Context vote is allowed to make the difference for marginal shapes.
t=dense('class_scores',t,[[0]*len(gains),gains],[0,-reference])
(OUT/'network.json').write_text(json.dumps({'layers':L},indent=1)+'\n')
np.savez(OUT/'weights.npz',**W)
