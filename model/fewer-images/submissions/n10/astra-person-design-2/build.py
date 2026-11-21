"""Compile geometric and photometric person detectors. No input files or fitted parameters."""
import sys
import json
from pathlib import Path
import numpy as np

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
weights, layers = {}, []
shapes = {'image': (3, 96, 96)}


def layer(name, op, ins, output_shape, **settings):
    assert name not in shapes
    shapes[name] = tuple(output_shape)
    layers.append(dict(name=name, op=op, inputs=ins, **settings))
    return name


def conv(name, src, w, b=0, stride=1, pad=None):
    w = np.array(w, dtype=np.float32)
    if w.ndim == 2:
        w = w[:, :, None, None]
    o, c, k, k2 = w.shape
    C, H, W = shapes[src]
    assert C == c and k == k2
    p = k // 2 if pad is None else pad
    weights[name+'.W'] = w
    weights[name+'.b'] = np.broadcast_to(b, (o,)).astype(np.float32)
    return layer(name, 'conv', [src], (o,(H+2*p-k)//stride+1,(W+2*p-k)//stride+1),
                 out=o, kernel=[k,k], stride=stride, padding=p)


def dense(name, src, w, b=0):
    w = np.asarray(w, dtype=np.float32)
    n, k = w.shape
    assert k == np.prod(shapes[src])
    weights[name+'.W'] = w
    weights[name+'.b'] = np.broadcast_to(b,(n,)).astype(np.float32)
    return layer(name,'dense',[src],(n,),out=n)


def relu(name, src):
    return layer(name, 'relu', [src], shapes[src])


def plus(name, *srcs):
    assert all(shapes[s] == shapes[srcs[0]] for s in srcs)
    return layer(name, 'add', list(srcs), shapes[srcs[0]])


def pool(name, src, k, s, mode='avg'):
    c,h,w = shapes[src]
    return layer(name,mode+'pool',[src],(c,(h-k)//s+1,(w-k)//s+1),size=[k,k],stride=[s,s])


def decisions(name, src, group_count, copies):
    # Conv responses are stored group-major; reduce alternatives at each position.
    c,h,w = shapes[src]
    assert c == group_count*copies
    rr = layer(name+'_alternatives','reshape',[src],(group_count,copies,h*w),shape=[group_count,copies,h*w])
    mm = layer(name+'_best','maxpool',[rr],(group_count,1,h*w),size=[copies,1],stride=[copies,1])
    return layer(name,'reshape',[mm],(group_count,h,w),shape=[group_count,h,w])


# Work on a two-pixel cell grid. The color rules are piecewise-linear membership
# functions in linear combinations of R,G,B; their biases are widths in RGB units.
rgb = pool('color_cells','image',2,2)
R,G,B = np.eye(3)
Y = .299*R+.587*G+.114*B
Cr = .713*(R-Y)
Cb = .564*(B-Y)
# Each triple is an inequality a.rgb >= b with a soft transition of width d.
# The first set gates on hue, not brightness; the second uses relative channel
# differences and allows a cooler white balance. Small Cr at low light is allowed.
rules = [
    [(Cr,.025,.022),(-Cr,-.19,.06),(-Cb,.006,.029),(Cb,-.185,.065),
     (G-B+.12*(R-G),-.006,.045),(R-G-.20*(G-B),.003,.043),
     (.86*G+.052*R-B,-.21,.09),(Y,.061,.054),(.62*G-R,-.48,.13)],
    [(R-.99*G,.015,.050),(G-.91*B,.0,.052),(.92*G+.12*R-B,-.025,.055),
     (G-.46*R,-.042,.075),(B-.38*R,-.05,.071),(R-B,.031,.047),
     (.80*R-1.80*G+B,-.01,.070),(Y,.054,.049),(-Cr,-.175,.060)]]
wr=[]; br=[]; groups=[]
for rule in rules:
    indices=[]
    for a,b,d in rule:
        indices.append(len(wr)); wr.append(-a/d); br.append(b/d)
    groups.append(indices)
cs=relu('color_violations',conv('color_halfplanes',rgb,wr,br))
w=np.zeros((2,len(wr)))
for j,idx in enumerate(groups): w[j,idx]=-1
members=relu('warm_memberships',conv('warm_intersections',cs,w,1))
skin=conv('complexion',members,[[.62,.38]])

# Correct for a lamp or a sunset turning much of an entire image warm. This
# is a per-input scene penalty, not a statistic used to make the weights.
terms=[]
cover=pool('global_complexion',skin,48,48)
covergate=relu('widespread_warmth',dense('global_hinges',cover,[[1],[1],[1]],[-.09,-.21,-.47]))
terms.append(dense('color_context_vote',covergate,[[-.56,-.47,.59]]))

# All spatial filters below are finite area sums. Gaussians have unit mass
# individually so response is a difference in local *averages*, independent of scale.
def grid(k):
    yy,xx=np.mgrid[:k,:k]-(k-1)/2
    return xx,yy


def blob(X,Y,x,y,sx,sy,tilt=0):
    xp=X-x-tilt*(Y-y); yp=Y-y
    a=np.exp(-.5*((xp/sx)**2+(yp/sy)**2))
    return a/a.sum()


# Outputs are in units of votes relative to a decision boundary. Reduce the
# search over locations with both the best location and the mean of positive
# detections in separate spatial tiles (rather than the mean over background).
def evidence(name, src, threshold, area_weight, top_weight):
    c,h,w=shapes[src]
    assert c==1 and h==w
    tiles=pool(name+'_tile_max',src,h//4,h//4,'max')
    t=shapes[tiles][1]
    gy,gx=np.mgrid[:t,:t]; u=(gx+.5)/t; v=(gy+.5)/t
    # The prior is weak. A face can be at the very edge of a cropped picture.
    prior=-.08*((u-.5)/.5)**2-.065*((v-.40)/.6)**2
    # This is a bias-only map (implemented as a dense on one scalar).
    bm=dense(name+'_composition_bias',cover,np.zeros((t*t,1)),prior.ravel())
    bm=layer(name+'_composition','reshape',[bm],(1,t,t),shape=[1,t,t])
    tiles=plus(name+'_locations',tiles,bm)
    mx=pool(name+'_peak',tiles,t,t,'max')
    best=dense(name+'_peak_vote',mx,[[top_weight]],[-top_weight*threshold])
    good=relu(name+'_confident',conv(name+'_tile_margin',tiles,[[1]],[-(threshold-.14)]))
    broad=dense(name+'_area_vote',good,np.full((1,t*t),area_weight/(t*t)))
    return plus(name+'_vote',best,broad)


# A small color-opponent edge bank. Fold both signs with ReLU so boundaries
# between clothes and background need not have a prescribed polarity. Half of
# each orientation is its own response; half competes against the perpendicular
# orientation and unstructured texture. A saturated edge cannot dominate a pose.
thetas=np.arange(4)*np.pi/4

def edges(name, src):
    X,Ygrid=grid(3); ws=[]
    # The three channels emphasize luminance; two lower-weight opponent
    # measurements preserve an isoluminant clothing/background boundary.
    for th in thetas:
        dx,dy=np.cos(th),np.sin(th)
        der=(X*dx+Ygrid*dy)*np.exp(-.5*((X*dx+Ygrid*dy)**2/.82**2+(-X*dy+Ygrid*dx)**2/.71**2))
        der=der/der.clip(min=0).sum()
        for col in [Y, R-G, (R+G)*.5-B]:
            for sign in [1,-1]: ws.append(sign*col[:,None,None]*der[None])
    # Use a valid convolution and add zero boundary in the following step,
    # rather than detecting the black padding as a strong real outline.
    raw=relu(name+'_unsigned',conv(name+'_derivatives',src,ws,pad=0))
    mix=np.zeros((4,24))
    for i in range(4): mix[i,6*i:6*i+6]=[1,1,.23,.23,.21,.21]
    mag=conv(name+'_magnitudes',raw,mix)
    orient=np.eye(4)*1.17-np.roll(np.eye(4),2,axis=1)*.46-.045
    hinges=relu(name+'_edge_ramps',conv(name+'_selectivity',mag,np.vstack([orient,orient]),np.r_[np.full(4,-.013),np.full(4,-.205)]))
    ww=np.zeros((4,8,3,3))
    ww[:,:,1,1]=np.hstack([np.eye(4),-np.eye(4)])/.192
    return conv(name+'_edges',hinges,ww,pad=2)


# Grid pyramid: heads roughly a handful of pixels to nearly the whole frame.
colorp=[rgb,pool('color_medium',rgb,2,2),pool('color_large',rgb,4,4)]
skinp=[skin,pool('skin_medium',skin,2,2),pool('skin_large',skin,4,4)]
edgep=[edges('mid',colorp[1]),edges('coarse',colorp[2])]
# normalized local eye/mouth/crown contrasts are capped via linear splines
# before combining. No single dark stripe suffices for a face match.
facemaps=[]
for lev,(col,sk) in enumerate(zip(colorp,skinp)):
    pref='face'+str(lev); stride=2 if lev==0 else 1
    k=9; X,Z=grid(k)
    widths=[(2.45,3.2),(3.55,4.35),(4.25,5.9),(5.35,5.35),(3.05,5.65)]
    if lev==0: widths=widths[:3]
    ns=len(widths); masks=[]; lights=[]; details=[]
    for fw,fh in widths:
        def f(x,y,sx,sy): return blob(X,Z,x*fw,y*fh,max(.43,sx*fw),max(.42,sy*fh))
        inner=f(0,.02,.245,.25)
        outside=.33*(f(-.80,0,.24,.40)+f(.80,0,.24,.40))+.24*f(0,-.76,.41,.23)+.10*f(0,.86,.48,.25)
        # The below-head surround has low weight: the neck is often also skin.
        masks.append(1.07*inner-.59*outside)
        cheek=f(0,.08,.255,.123)
        brow=.5*(f(-.234,-.15,.14,.067)+f(.234,-.15,.14,.067))
        forehead=f(0,-.342,.24,.115)
        mouth=f(0,.29,.215,.065)
        chin=f(0,.415,.205,.080)
        browridge=f(0,-.14,.087,.138)
        crown=f(0,-.55,.35,.15)
        temples=(f(-.51,-.10,.12,.285)+f(.51,-.10,.12,.285))*.5
        lights.append(.72*inner-.27*crown-.22*temples-.23*f(0,-.30,.50,.42))
        details.extend([.60*cheek+.40*forehead-brow,
                        .7*cheek+.3*chin-mouth,
                        .68*cheek+.32*forehead-.59*crown-.41*temples,
                        browridge-brow])
    lum=conv(pref+'_gray',col,[Y])
    # align the luminance DC at image boundaries to a neutral grey, by
    # centering it before using zero-padded kernels.
    lum0=conv(pref+'_center_light',lum,[[1]],[-.43])
    warm=conv(pref+'_skin_oval',sk,np.array(masks)[:,None],stride=stride)
    phot=conv(pref+'_luminance_model',lum0,np.array(details+lights)[:,None],stride=stride)
    m=4*ns+ns
    slopes=np.tile([4.1,3.,2.65,2.6],ns).tolist()+[1.25]*ns
    # A positive and a negative hinge limit contrast to a finite contribution.
    diag=np.diag(slopes)
    cap=np.tile([.75,.48,.70,.53],ns).tolist()+[.48]*ns
    base=np.tile([.082,.043,.035,.034],ns).tolist()+[.042]*ns
    delta=relu(pref+'_bounded_ramps',conv(pref+'_contrast_ramps',phot,np.vstack([diag,diag]),np.r_[base,np.array(base)-cap]))
    m1=np.zeros((ns,m))
    for j in range(ns):
        m1[j,4*j:4*j+4]=[.88,.40,.46,.26]
        m1[j,4*ns+j]=.17
    photo=conv(pref+'_features',delta,np.hstack([m1,-m1]))
    # Combine shape and complexion before reducing scale. Small unresolved
    # faces may have a skin spot under hair but little internal detail.
    raw=plus(pref+'_color_and_detail',warm,photo)
    # Border truncation removes evidence in the surround. Penalize sampling
    # outside the image by the same order of magnitude, especially for portraits.
    h=shapes[raw][1]; yy,xx=np.mgrid[:h,:h]; dist=np.minimum.reduce([xx,yy,h-1-xx,h-1-yy])*stride
    penalties=np.array([-.19*np.maximum(0,1-dist/(.77*max(fw,fh))) for fw,fh in widths])
    pen=dense(pref+'_border_bias',cover,np.zeros((ns*h*h,1)),penalties.ravel())
    pen=layer(pref+'_border','reshape',[pen],(ns,h,h),shape=[ns,h,h])
    raw=plus(pref+'_corrected',raw,pen)
    best=decisions(pref+'_any_aspect',raw,1,ns)
    facemaps.append(best)

# For a sufficiently resolved face, require *both* eye sockets and test an
# independent mouth and nose-bridge contrast. This route can fire in B&W.
landmarkmaps=[]
for lev,(col,sk) in enumerate(zip(colorp,skinp)):
    pre='landmark'+str(lev); stride=2 if lev==0 else 1; X,Z=grid(11)
    filters=[]; warm=[]
    for fw,fh in [(4.5,6.4),(6.4,7.4),(5.9,9.5),(8.4,8.1)]:
        def reg(x,y,sx,sy): return blob(X,Z,fw*x,fh*y,max(.41,fw*sx),max(.41,fh*sy))
        cheeks=[reg(s*.234,.089,.103,.094) for s in [-1,1]]
        eyes=[reg(s*.227,-.148,.097,.052) for s in [-1,1]]
        for j,s in enumerate([-1,1]):
            filters.append(.60*cheeks[j]+.40*reg(s*.211,-.345,.13,.097)-eyes[j])
        nose=reg(0,-.090,.073,.123)
        filters.extend([.32*(cheeks[0]+cheeks[1])+.36*reg(0,.398,.17,.056)-reg(0,.263,.163,.049),
                        .5*(cheeks[0]+cheeks[1])-.60*reg(0,-.56,.34,.13)-.20*(reg(-.54,-.07,.108,.26)+reg(.54,-.07,.108,.26)),
                        nose-.5*(eyes[0]+eyes[1]),cheeks[0]-cheeks[1]])
        oval=reg(0,.01,.27,.27)
        warm.append(.26*oval-.12*(.5*(reg(.83,0,.25,.34)+reg(-.83,0,.25,.34))))
    n=len(warm); q=6
    lum=conv(pre+'_gray',col,[Y],[-.43]); contrast=conv(pre+'_patches',lum,np.array(filters)[:,None],stride=stride)
    # Collapse two eye contrasts into a single minimum; limit each feature
    # before voting, and discount a large left/right cheek mismatch.
    trans=np.zeros((5*n,q*n)); ad=np.zeros((3*n,q*n));
    for j in range(n):
        trans[5*j,6*j]=1
        for i in range(3): trans[5*j+1+i,6*j+2+i]=1
        ad[3*j,6*j:6*j+2]=[1,-1]; ad[3*j+1,6*j+5]=1; ad[3*j+2,6*j+5]=-1
    pen=relu(pre+'_asymmetry',conv(pre+'_compare',contrast,ad))
    reduce=np.zeros((5*n,3*n))
    for j in range(n): reduce[5*j,3*j]=-1; reduce[5*j+4,3*j+1:3*j+3]=1
    local=plus(pre+'_landmarks',conv(pre+'_features',contrast,trans),conv(pre+'_balance',pen,reduce))
    caps=np.tile([.163,.135,.21,.142,.19],n)
    v=relu(pre+'_contrast_caps',conv(pre+'_thresholds',local,np.vstack([np.eye(5*n),np.eye(5*n)]),np.r_[np.zeros(5*n),-caps]))
    proj=np.zeros((n,5*n))
    for j in range(n): proj[j,5*j:5*j+5]=[2.85,1.48,.92,1.41,-.79]
    raw=plus(pre+'_eye_pair_and_mouth',conv(pre+'_match',v,np.hstack([proj,-proj])),conv(pre+'_optional_color',sk,np.array(warm)[:,None],stride=stride))
    landmarkmaps.append(decisions(pre+'_any',raw,1,n))

# A truly small head: it may be only a few cells of color. Compare the center
# with a surrounding U rather than trying to resolve the eyes at this scale.
X,Z=grid(5)
sm=lambda x,y,sx,sy:blob(X,Z,x,y,sx,sy)
inside=sm(0,0,.66,.78)
around=(sm(-1.8,0,.57,.97)+sm(1.8,0,.57,.97)+sm(0,-1.9,.87,.55))/3
spot=conv('small_head_color',skin,[(inside-.66*around)[None]],stride=2)
smallgray=conv('small_head_light',rgb,[(Y[:,None,None]*(inside-.55*sm(0,-1.4,.83,.51)-.45*around))],stride=2)
spotlight=relu('small_head_relief',conv('small_head_relief_ramps',smallgray,[[1],[1]],[-.025,-.24]))
spotlight=conv('small_head_limited_relief',spotlight,[[.9,-.9]])
small=plus('small_head',spot,spotlight)

# A hand or forearm near a plausible face is more informative than another
# warm pixel at an arbitrary location. Match both sides separately, and give
# raised, lowered and outstretched arms separate endpoint sectors. A negative
# ring and center-to-wrist sample suppress broad monochromatic objects.
limbmaps=[]
for lev,(col,sk) in enumerate(zip(colorp,skinp)):
    pf='limbs'+str(lev); stride=2 if lev==0 else 1
    X,Z=grid(13); fw,fh=3.0,3.75
    def reg(cx,cy,sx,sy): return blob(X,Z,cx*fw,cy*fh,sx*fw,sy*fh)
    face=reg(0,0,.26,.255)
    face_near=.36*(reg(-.84,.02,.23,.33)+reg(.84,.02,.23,.33))+.28*reg(0,-.78,.4,.21)
    bank=[.77*face-.45*face_near]; light=face-.66*reg(0,-.48,.37,.14)-.17*(reg(-.55,-.03,.15,.26)+reg(.55,-.03,.15,.26))
    for side in [-1,1]:
        for cx,cy in [(1.02,1.46),(1.4,.80),(1.36,.08),(.66,1.37)]:
            cx*=side
            wrist=reg(cx,cy,.245,.27)
            near=(reg(cx-.60,cy,.22,.32)+reg(cx+.60,cy,.22,.32)+reg(cx,cy+.58,.30,.18))/3
            middle=reg(.51*cx,.55*cy,.21,.19)
            bank.append(wrist-.42*near-.23*middle)
    maps=conv(pf+'_head_and_wrists',sk,np.array(bank)[:,None],stride=stride)
    wr=conv(pf+'_wrists',maps,np.eye(9)[1:]); best=decisions(pf+'_best_wrist',wr,1,8)
    face=conv(pf+'_head',maps,np.eye(9)[:1]); lum=conv(pf+'_gray',col,[Y],[-.43])
    relief=conv(pf+'_relief',lum,[light[None]],stride=stride)
    rr=relu(pf+'_relief_ramps',conv(pf+'_dark_hair',relief,[[1],[1]],[-.015,-.24]))
    face=plus(pf+'_head_match',face,conv(pf+'_bounded_relief',rr,[[.9,-.9]]))
    # Half credit from a single part is not enough. Most of the vote demands both.
    gap=plus(pf+'_excess',face,conv(pf+'_negative',best,[[-1]]))
    joint=plus(pf+'_association',conv(pf+'_head_weight',face,[[1.16]]),
               conv(pf+'_arm_weight',best,[[.32]]),conv(pf+'_agreement',relu(pf+'_missing_arm',gap),[[-.56]]))
    limbmaps.append(joint)

# A paired-contour HOG model. Each polygon segment deposits mass in normal-
# orientation bins, at fuzzy positions around the idealized contour. Parts
# have their own budgets, so long straight legs don't drown the shoulders.
# The side, central and lower portions are contrasted with adjacent translated
# copies, suppressing evenly textured surfaces. The templates are independent.
def segment_density(X,Z,p,q,sigma):
    ux,uy=p; vx,vy=q; dx=vx-ux; dy=vy-uy
    t=np.clip(((X-ux)*dx+(Z-uy)*dy)/(dx*dx+dy*dy+1e-10),0,1)
    return np.exp(-.5*((X-ux-dx*t)**2+(Z-uy-dy*t)**2)/sigma**2)


def segments(points): return list(zip(points[:-1],points[1:]))


def contour(X,Z,segs,sigma):
    maps=np.zeros((4,)+X.shape)
    for p,q in segs:
        dx,dy=np.subtract(q,p); ln=np.hypot(dx,dy)
        if ln < 1e-5: continue
        normal=np.arctan2(dy,dx)+np.pi/2
        votes=np.maximum(0,np.cos(2*(thetas-normal)))**2
        votes/=votes.sum()
        g=segment_density(X,Z,p,q,sigma)
        maps+=ln*votes[:,None,None]*g/max(1e-12,g.sum())
    return maps/maps.sum()


def shifted(a,dx,dy):
    n=a.shape[1]; padded=np.pad(a,((0,0),(abs(dy),abs(dy)),(abs(dx),abs(dx))))
    ii=abs(dy)-dy; jj=abs(dx)-dx
    return padded[:,ii:ii+n,jj:jj+n]


def with_background(c):
    # Surround positions have no preferred gradient sign either.
    neg=(shifted(c,-2,0)+shifted(c,2,0)+shifted(c,0,-2)+shifted(c,0,2))
    return c-.64*neg/neg.sum()


figmaps=[]; joined=[]
for lev,(edge,sk) in enumerate(zip(edgep,skinp[1:])):
    pre='figure'+str(lev); k=11; X,Z=grid(k)
    pose=[(2.45,3.7,0),(3.7,4.7,0),(2.85,3.6,.28),(2.85,3.6,-.28),(1.8,2.65,0),(4.75,5.6,0)]
    ws=[]; hs=[]; scalp=[]; scalp_gray=[]
    for hw,hh,lean in pose:
        def pt(u,v): return (u*hw+lean*v*hh,v*hh-1.75)
        def line(pts): return segments([pt(u,v) for u,v in pts])
        # (x,y) in head widths and heights; y=0 is center of head.
        skull=line([(-.30,.48),(-.48,.12),(-.46,-.26),(-.30,-.45),(0,-.53),(.30,-.45),(.46,-.26),(.48,.12),(.30,.48)])
        left=[(-.28,.33),(-.29,.54),(-.58,.60),(-.96,.72),(-1.10,.95),(-1.18,1.29)]
        shoulders=line(left)+line([(-u,v) for u,v in left]);
        neck=line([(-.79,.77),(-.46,.62),(-.28,.57),(-.25,.40)])+line([(.79,.77),(.46,.62),(.28,.57),(.25,.40)])
        ch=contour(X,Z,skull,.51); cs=contour(X,Z,shoulders,.58); cn=contour(X,Z,neck,.57)
        ws.extend([with_background(ch),with_background(.78*cs+.22*cn)])
        x,y=pt(0,.07)
        inn=blob(X,Z,x,y,.245*hw,.255*hh,lean)
        out=.34*(blob(X,Z,x-.83*hw,y,.24*hw,.39*hh,lean)+blob(X,Z,x+.83*hw,y,.24*hw,.39*hh,lean))+.32*blob(X,Z,*pt(0,-.80),.43*hw,.21*hh,lean)
        scalp.append(inn-.66*out)
        scalp_gray.append(inn-.58*blob(X,Z,*pt(0,-.48),.34*hw,.14*hh,lean)-.21*(blob(X,Z,*pt(-.55,-.07),.16*hw,.26*hh,lean)+blob(X,Z,*pt(.55,-.07),.16*hw,.26*hh,lean)))
    n=len(pose)
    pair=conv(pre+'_pair_boundaries',edge,ws)
    # Piecewise linear approximation to conjunction: integrate both parts,
    # with a penalty when the weaker of two contours is missing.
    wt=np.zeros((2*n,2*n));
    for j in range(n): wt[2*j,2*j:2*j+2]=[1,-1]; wt[2*j+1,2*j:2*j+2]=[-1,1]
    diff=relu(pre+'_part_mismatch',conv(pre+'_part_difference',pair,wt))
    wt=np.zeros((n,2*n))
    for j in range(n): wt[j,2*j:2*j+2]=[.54,.46]
    outline=conv(pre+'_outline_mean',pair,wt)
    outline=plus(pre+'_upper_outline',outline,conv(pre+'_unpaired_penalty',diff,-.38*(wt!=0)))
    scalp=conv(pre+'_face_patch',sk,np.asarray(scalp)[:,None])
    gray=conv(pre+'_head_intensity',colorp[lev+1],[Y],[-.43])
    relief=conv(pre+'_head_relief',gray,np.array(scalp_gray)[:,None])
    ramp=relu(pre+'_head_ramps',conv(pre+'_head_response',relief,np.vstack([np.eye(n),np.eye(n)]),np.r_[np.full(n,-.024),np.full(n,-.28)]))
    relief=conv(pre+'_head_shading',ramp,.97*np.hstack([np.eye(n),-np.eye(n)]))
    combo=plus(pre+'_matched_skin_and_body',scalp,relief,conv(pre+'_shape_contribution',outline,1.46*np.eye(n)))
    joined.append(decisions(pre+'_joint_any',combo,1,n))
    figmaps.append(decisions(pre+'_silhouette_any',outline,1,n))

# Full or three-quarter standing figures. Here the width/height alternatives
# explicitly accommodate the squashing of the original rectangular photos.
# Pair a closed head/shoulder cap with torso edges, then a split-leg pattern.
fullmaps=[]; fullcolor=[]
for lev,(edge,sk) in enumerate(zip(edgep,skinp[1:])):
    pre='standing'+str(lev); X,Z=grid(11); banks=[]; faces=[]
    for width,height,split,lean in [(2.35,8.8,.30,0),(3.35,9.3,.37,0),(2.25,8.3,.68,.19),(2.25,8.3,.68,-.19),(4.15,9.35,.34,0)]:
        def pt(u,v): return ((u*width)+(v-.46)*height*lean,(v-.5)*height)
        def path(points): return segments([pt(u,v) for u,v in points])
        side=[(-.11,.17),(-.22,.095),(-.17,.023),(0,0),(.17,.023),(.22,.095),(.11,.17)]
        cap=path(side)
        body=[]; legs=[]
        for s in [-1,1]:
            body+=path([(s*.12,.15),(s*.16,.19),(s*.46,.23),(s*.54,.37),(s*.49,.54),(s*.31,.58)])
            legs+=path([(s*.37,.49),(s*.35,.64),(s*(split+.13),.92),(s*(split+.12),.98),(s*(split-.10),.98),(s*(split-.12),.86),(s*.04,.59)])
        banks.extend([with_background(.43*contour(X,Z,cap,.48)+.57*contour(X,Z,body,.50)),with_background(contour(X,Z,legs,.51))])
        x,y=pt(0,.1); f=blob(X,Z,x,y,max(.45,.16*width),.54)
        ss=.40*(blob(X,Z,x-1.4,y,.54,.68)+blob(X,Z,x+1.4,y,.54,.68))+.20*blob(X,Z,x,y+2.,.7,.65)
        faces.append(f-.68*ss)
    n=len(faces); response=conv(pre+'_body_and_legs',edge,banks)
    a=np.zeros((n,2*n)); d=np.zeros_like(a)
    for i in range(n): a[i,2*i:2*i+2]=[.61,.39]; d[i,2*i:2*i+2]=[1,-1]
    ab=relu(pre+'_mismatch',conv(pre+'_differences',response,np.vstack([d,-d])))
    both=plus(pre+'_whole',conv(pre+'_parts',response,a),conv(pre+'_cost',ab,-.25*np.hstack([np.eye(n),np.eye(n)])))
    fullmaps.append(decisions(pre+'_any',both,1,n))
    joint=plus(pre+'_head_and_clothing',conv(pre+'_skin',sk,np.array(faces)[:,None]),conv(pre+'_body',both,np.eye(n)*1.68))
    fullcolor.append(decisions(pre+'_any_with_head',joint,1,n))

# Fixed decision margins, specified in the units of the individual filters.
# First combine neighboring resolution levels in the same frame of reference.
def resolutions(name, maps):
    h=min(shapes[m][1] for m in maps); mlist=[]
    for i,m in enumerate(maps):
        s=shapes[m][1]//h; assert shapes[m][1]==s*h
        if s>1: m=pool(name+'_resolution'+str(i),m,s,s,'max')
        mlist.append(m)
    a=mlist[0]
    for j,b in enumerate(mlist[1:]):
        d=plus(name+'_diff'+str(j),a,conv(name+'_negative'+str(j),b,[[-1]]))
        a=plus(name+'_max'+str(j),relu(name+'_positive_diff'+str(j),d),b)
    return a

# Each entry says how much an excellent detection in this family should
# exceed an ordinary non-match. Take the strongest vote and also ask the
# other cue families for a small amount of corroboration.
votes=[evidence('facial',resolutions('face_search',facemaps),1.42,.68,1.03),
       evidence('bust',resolutions('bust_search',joined),1.12,.49,1.02),
       evidence('small_figure',resolutions('small_figure_search',fullcolor),1.12,.46,1.00),
       evidence('upper_outline',resolutions('bust_edges',figmaps),.34,.73,2.40),
       evidence('lower_outline',resolutions('entire_outline',fullmaps),.35,.72,2.34),
       evidence('unresolved',small,.96,.51,.64),
       evidence('exposed_parts',resolutions('limb_search',limbmaps),1.18,.45,1.0),
       evidence('eye_pair',resolutions('landmark_search',landmarkmaps),.89,.55,1.02)]
# max in score space; also aggregate a shallow vote sharing the decision
# among detailed face, clothed figure and tiny-head evidence.
a=votes[0]
for i,b in enumerate(votes[1:]):
    d=plus('compare'+str(i),a,dense('negative_vote'+str(i),b,[[-1]]))
    a=plus('best_vote'+str(i),b,relu('improvement'+str(i),d))
terms.append(dense('strongest_cue',a,[[.78]]))
for i,(v,coef) in enumerate(zip(votes,[.10,.08,.04,.03,.037,.023,.03,.027])):
    terms.append(dense('corroborate'+str(i),v,[[coef]]))

# Weak context: fields of green with a short, saturated turf-like texture
# below sky are common for sports. Weight only a little; trees and grass
# are not evidence comparable to the detection of a face.
coarse=colorp[-1]
ctx=relu('outdoor_color_ramps',conv('outdoor_colors',coarse,[G-R,G-R,G-B,G-B,B-R,B-R,B-G,B-G],[-.010,-.10,-.04,-.16,-.045,-.19,.007,-.11]))
arr=np.zeros((2,8)); arr[0,:4]=[5.0,-5.0,4.2,-4.2]; arr[1,4:]=[3.9,-3.9,3.9,-3.9]
ctx=conv('field_and_sky',ctx,arr)
_,h,w=shapes[ctx]; wt=np.zeros((2,2,h,w)); wt[0,0,h//2:]=2/(h*w); wt[1,1,:h//3]=3/(h*w)
ctx=dense('field_layout',ctx,wt.reshape(2,-1))
ctx=relu('field_and_horizon',dense('field_test',ctx,[[1,.38],[1,.38]],[-.27,-.80]))
terms.append(dense('sports_context_vote',ctx,[[.11,-.11]]))

final=plus('person_margin',*terms)
dense('scores',final,[[0],[1]],[0,0])
(out_dir/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
np.savez(out_dir/'weights.npz',**weights)
