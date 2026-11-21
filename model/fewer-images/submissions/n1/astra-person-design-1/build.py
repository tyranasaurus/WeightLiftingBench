"""Construct a person-presence network from hand-written rules. No data inputs."""
import json
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
layers, weights = [], {}

def layer(name, op, inputs, **kwargs):
    layers.append(dict(name=name, op=op, inputs=[inputs] if isinstance(inputs,str) else inputs, **kwargs))
    return name

def conv(name, src, w, b=0., stride=1, pad=0):
    w = np.asarray(w, np.float32)
    if w.ndim == 2: w = w[:,:,None,None]
    n, _, k, _ = w.shape
    weights[name+'.W'] = w
    weights[name+'.b'] = np.broadcast_to(b,(n,)).astype(np.float32).copy()
    return layer(name,'conv',src,out=n,kernel=[k,k],stride=stride,padding=pad)

def dense(name,src,w,b=0.):
    w=np.atleast_2d(w).astype(np.float32)
    weights[name+'.W']=w
    weights[name+'.b']=np.broadcast_to(b,(len(w),)).astype(np.float32).copy()
    return layer(name,'dense',src,out=len(w))

def pos(name,src): return layer(name,'relu',src)
def add(name,*src): return layer(name,'add',list(src))
def pool(name,src,k,s=None,maximum=False):
    if s is None: s=k
    return layer(name,'maxpool' if maximum else 'avgpool',src,size=[k,k],stride=[s,s])

def gauss(x,y,cx,cy,sx,sy):
    z=-.5*(((x-cx)/sx)**2+((y-cy)/sy)**2)
    a=np.exp(z-z.max())
    return a/a.sum()

def coords(k):
    y,x=np.mgrid[:k,:k]-(k-1)/2
    return x,y

# Average locally rather than assigning a color from a single (possibly noisy)
# pixel. Build a soft polyhedron in chroma/intensity space. All six costs are
# distances beyond a plane, not distances to an exemplar color.
im2=pool('half_rgb','image',2)
R,G,B=np.eye(3)
d=R-G; e=G-B
luma=.28*R+.56*G+.16*B
cast=conv('cast',im2,[[1,-1,0],[0,1,-1]],stride=1)
# Rows include the intercept the violation would have at RGB=0.
planes=[(-d+.085*G,.019, .048), (d-.46*G,-.15,.16),
        (-e-.28*d,.004,.05), (e-1.03*d-.058*G,-.014,.070),
        (-R,.125,.10), (d+e-.70*R,-.025,.13)]
wc=np.array([w/t for w,b,t in planes]); bc=np.array([b/t for w,b,t in planes])
# Apply a fraction of a gray-world correction outside a chroma dead zone.
castmean=pool('mean_cast',pool('block_cast',cast,8),6)
# Reduce the warm-axis shift on warm pictures: preserve some actual skin hue.
idcast=np.eye(2)
c=conv('cast_deadzone',castmean,np.vstack([idcast,-idcast]),[-.042,-.032,-.030,-.030])
c=pos('cast_tails',c)
wbr=np.repeat((-.38*np.array([[1,0,-1,0],[0,0,0,0],[0,-1,0,1]])),48*48,axis=0)
illum=layer('illuminant','reshape',dense('illuminant_field',c,wbr),shape=[3,48,48])
white=add('adapted_half',im2,illum)
err=pos('hue_costs',conv('hue_planes',white,wc,bc))
skin=pos('skin',conv('skin_margin',err,[[-1]*len(planes)],1))

# Average color statistics also preserve evidence for hands etc. when a clean
# head/body relation is not visible. Temper the benefit of further area: a whole
# wall in the right hue is much weaker evidence than a number of distinct parts.
area=pool('warm_fraction',skin,48)
skin_global=dense('area_gain',pos('area_segments',dense('area_knots',area,[[1],[1],[1],[1]],[-.005,-.045,-.13,-.31])),[[3.3,-2.5,-.9,-.3]],-.063)

Y=conv('half_light',im2,[luma])
# Edge normals in four directions. Use the interior of the image only for scene
# statistics; the outline of the frame is not a person or an architectural line.
x,y=coords(3)
edge_w=[]
for t in np.arange(4)*np.pi/4:
    a=(np.cos(t)*x+np.sin(t)*y)*np.exp(-.65*(x*x+y*y)); a/=a[a>0].sum()
    edge_w.extend([a,-a])
q=pos('edge_signs',conv('light_differences',Y,np.array(edge_w)[:,None],stride=2,pad=1))
# A clipped edge vote cannot dominate several missing figure parts on its own.
Wmag=np.repeat(np.eye(4),2,axis=1)
mag=conv('normal_magnitude',q,Wmag)
# center-weighted composition statistic, includes a penalty for strong long
# horizontal/vertical edges (buildings, interiors, large machines).
u,v=coords(24); layout=np.exp(-((u/8.)**2+((v+.8)/9.)**2)); layout[0,:]=0; layout[:,0]=0
interior=np.ones((24,24)); interior[[0,-1],:]=0; interior[:,[0,-1]]=0
spatial=layout/layout.sum()-interior/interior.sum()
wc_scene=np.array([1.3*spatial+q0*interior/interior.sum() for q0 in [-.65,.65,-.65,.65]])
scene=dense('composition',mag,wc_scene.ravel())
# Direction-selective HOG-like responses: subtract competing orientations, but
# don't insist on high selectivity at every point of a small rough contour.
# Light-to-dark polarity does not matter.
w_or=np.eye(4)-.64/4*np.ones((4,4))
w_or=np.vstack([w_or/.12,(w_or/.12)])
oresp=pos('edge_ramps',conv('edge_gate',mag,w_or,[-.055]*4+[-1.055]*4))
orient=conv('contours',oresp,np.hstack([np.eye(4),-np.eye(4)]))

# Sliding groups of weak tests. The kernels are field averages, each normalized
# within that very region. Kernels that measure a difference compare two unit-
# integral fields. They are specified by geometry, not by sampling photographs.
patch_scores=[]
def summarize(tag,margin,side,peak,distributed,body=False):
    # Location prior is mild. Allow a head at the edge, but prefer a response in
    # the upper/middle of the frame to something at the very bottom.
    y,x=coords(side); x=x/(side/2); y=y/(side/2)
    prior=.04*(1-x*x)-.070*np.maximum(y-.12,0)-.012
    if body: prior=.035*(1-x*x)-.015
    bias=dense(tag+'_position_bias',pool(tag+'_unit_source',margin,side),np.zeros((side*side,1)),prior.ravel())
    bias=layer(tag+'_position','reshape',bias,shape=[1,side,side])
    s=pos(tag+'_positive',add(tag+'_located',margin,bias))
    # Use a global maximum plus a much smaller vote for corroborating responses
    # in distinct regions. Nested pooling bounds workspace on the largest maps.
    if side==24: bs=6
    elif side==16: bs=4
    elif side==12: bs=3
    elif side==8: bs=2
    elif side==6: bs=2
    else: bs=side
    p=pool(tag+'_regional',s,bs,maximum=True)
    pmax=pool(tag+'_peak',p,side//bs,maximum=True) if side!=bs else p
    pavg=pool(tag+'_support',p,side//bs) if side!=bs else p
    sp=add(tag+'_read',dense(tag+'_best',pmax,[[peak]]),dense(tag+'_coverage',pavg,[[distributed]]))
    patch_scores.append(sp)
    return sp

# High and low resolutions share the same body plan. The smallest scale measures
# a few-pixel head directly at half resolution, without an extra downsampling.
for tag,step,side,headw,headh,K in [
        ('distant',2,24,1.8,2.5,9),('small',3,16,3.1,4.2,13),
        ('middle',4,12,3.0,4.,11),('near',6,8,4.0,5.1,11),
        ('large',8,6,5.1,6.1,11)]:
    src=skin if step<=3 else pool(tag+'_skin_scale',skin,step//2)
    lum=Y if step<=3 else pool(tag+'_light_scale',Y,step//2)
    ss=3 if step==3 else 2
    # Different tests vote for a warm oval and against a long stripe, floor or wall.
    x,y=coords(K)
    F=lambda a,b,c,d: gauss(x,y,a,b,c,d)
    center=F(0,0,max(.52,.29*headw),max(.55,.29*headh))
    left=F(-.87*headw,0,.31*headw,.40*headh); right=left[:,::-1]
    top=F(0,-.78*headh,.47*headw,.24*headh)
    above=F(0,-.60*headh,.35*headw,.18*headh)
    bottom=F(0,1.05*headh,.52*headw,.34*headh)
    # Three directions of boundary evidence, independent and softly saturated.
    # Read central occupancy as well: a warm ellipse amid a background which is
    # warm everywhere must not be rewarded merely for having the right color.
    kr=np.array([center,center-left,center-right,center-top,center-bottom])
    reg=conv(tag+'_regions',src,kr[:,None],stride=ss,pad=K//2)
    # Luminance tests measure a hairline, left and right sides differing from
    # head, and a collar beneath the neck. Signs on the latter tests can vary.
    proj=[center-above,center-left,center-right,center-bottom]
    # Coarse scales: 12/2=6 etc.; center-aligned pooling and odd kernels.
    light=conv(tag+'_contrast',lum,np.array(proj)[:,None],stride=ss,pad=K//2)
    # For color the two side margins should BOTH contribute, not a single strong
    # transition. Clipping each before summation implements that soft conjunction.
    a=np.diag([1,1,1,1,1]); knots=np.array([.13,.025,.025,.02,.035]); caps=np.array([.72,.50,.50,.48,.50]);
    rr=pos(tag+'_region_gates',conv(tag+'_region_thresholds',reg,np.vstack([a,a]),np.r_[-knots,-knots-caps]))
    coef=np.array([.38,.31,.31,.25,.24]); color=conv(tag+'_bounded_color',rr,[np.r_[coef,-coef]])
    # Exposure-independent sign for side/collar contrasts; specifically look for
    # a darker cap above the head. Bound each response to prevent high-contrast
    # lines from earning as much evidence as a complete head.
    a=np.vstack([np.eye(4),-np.eye(4)[1:]])
    knots=np.array([.020,.022,.022,.025,.022,.022,.025]); caps=np.array([.16,.22,.22,.22,.22,.22,.22])
    pp=pos(tag+'_contrast_gates',conv(tag+'_contrast_thresholds',light,np.vstack([a,a]),np.r_[-knots,-knots-caps]))
    coef=np.array([.75,.27,.27,.37,.27,.27,.37]); features=conv(tag+'_bounded_contrast',pp,[np.r_[coef,-coef]])
    # Simultaneously look for a torso below the head. An inner shirt sample
    # should differ in RGB from neighbors on both sides, more than two samples
    # within the shirt differ from one another. Absolute RGB differences also
    # catch equal-luminance clothing. Penalize off-frame tests further below.
    rgb=im2 if step<=3 else pool(tag+'_rgb_scale',im2,step//2)
    # All clothing tests are on the candidate lattice. This allows a wider
    # receptive field without a large convolution on the original grid.
    rgb=pool(tag+'_garment_grid',rgb,ss)
    kw=9 if tag in ('distant','small','middle') else 11
    xx,yy=coords(kw); GD=lambda a,b,c,d:gauss(xx,yy,a,b,c,d)
    h=headh/ss; w=headw/ss
    # garment at a slightly deeper location than the chin. Two symmetric flank
    # samples plus an interior variation cost (texture is not a silhouette).
    mid=GD(0,1.75*h,max(.52,.48*w),max(.55,.55*h));
    le=GD(-2.1*w,1.75*h,max(.52,.45*w),max(.55,.62*h)); ri=le[:,::-1]
    high=GD(0,1.18*h,max(.50,.45*w),max(.36,.30*h)); low=GD(0,2.32*h,max(.50,.45*w),max(.36,.30*h))
    # Three differences of slightly decorrelated colors: luma is most stable;
    # chroma is best for colorful clothing and is less sensitive to lighting.
    axes=np.array([luma,.85*(R-B),.9*(G-.5*(R+B))]);
    t=np.array([c[:,None,None]*v for v in (mid-le,mid-ri,high-low) for c in axes]);
    # In the interior padding cannot matter. Toward a border, damp positives
    # by reducing the clothing bonus at those candidate locations.
    contrasts=conv(tag+'_shirt_differences',rgb,np.vstack([t,-t]),pad=kw//2)
    absdiff=pos(tag+'_shirt_abs',contrasts)
    score=np.array([[1.9,1.,1.,0,0,0,-1.15,-.65,-.65],[0,0,0,1.9,1.,1.,-1.15,-.65,-.65]])
    summed=conv(tag+'_shirt_sides',absdiff,np.hstack([score,score]),-.035)
    sides=pos(tag+'_shirt_gates',conv(tag+'_shirt_knots',summed,np.vstack([np.eye(2),np.eye(2)]),[0,0,-.35,-.35]))
    tshirt=conv(tag+'_shirt_bonus',sides,[[.29,.29,-.29,-.29]])
    sy,sx=np.mgrid[:side,:side]; xr=sx*ss; yr=sy*ss
    # Directly account for missing context. A head at the bottom is unlikely
    # to have shirt evidence, but still has valid skin and eye evidence.
    edge=np.maximum(0,1-np.minimum(xr,side*ss-1-xr)/(.95*headw))
    cut=np.maximum(0,1-(side*ss-1-yr)/(2.9*headh))
    topcut=np.maximum(0,1-yr/(.8*headh))
    penalty=.16*edge+.22*cut+.095*topcut
    threshold=.758 if tag=='distant' else .715 if tag=='small' else .685
    prior=np.exp(-((sx-(side-1)/2)/(.45*side))**2-((sy-.38*(side-1))/(.5*side))**2)
    thresh=dense(tag+'_level',pool(tag+'_context_source',color,side),np.zeros((side*side,1)),(-threshold-penalty+.045*prior).ravel())
    response=add(tag+'_candidate',color,features,tshirt,layer(tag+'_level_map','reshape',thresh,shape=[1,side,side]))
    summarize(tag,response,side,.90,.25)

# A face-pattern branch which doesn't require a skin-colored surround or even
# a color image. Eye sockets are usually darker than cheeks and brow, bridge of
# nose is brighter than eye sockets, tip of nose brighter than mouth, etc.
# A single horizontal dark stripe should fail the tests *between* the eyes.
face_scores=[]
for tag,down,side,hw,hh in [('face8',1,24,4.0,5.3),('face15',2,12,4.1,5.6),('face27',3,8,4.9,6.0),('face42',4,6,5.9,7.1)]:
    k=9 if down<=2 else 11
    lum=Y if down==1 else pool(tag+'_light',Y,down)
    sk=skin if down==1 else pool(tag+'_hue',skin,down)
    x,y=coords(k)
    alltests=[]; cols=[]
    # Try level faces as well as two modest head tilts on the more resolved
    # scales. Larger templates also have sufficient resolution for mouth cues.
    angles=[0] if down==1 else [0,.22,-.22]
    for angle in angles:
        X=(np.cos(angle)*x+np.sin(angle)*y)/hw
        Z=(-np.sin(angle)*x+np.cos(angle)*y)/hh
        f=lambda a,b,c,d:gauss(X,Z,a,b,c,d)
        eyes=[f(s*.222,-.090,.105,.066) for s in (-1,1)]
        cheeks=[f(s*.235,.105,.124,.090) for s in (-1,1)]
        brows=[f(s*.17,-.29,.14,.085) for s in (-1,1)]
        bridge=f(0,-.064,.078,.089); nose=f(0,.13,.096,.083); mouth=f(0,.272,.163,.055); chin=f(0,.384,.152,.057)
        # Rows individually normalized by the regions defining the comparison.
        tests=[cheeks[0]-eyes[0],cheeks[1]-eyes[1],brows[0]-eyes[0],brows[1]-eyes[1],
               bridge-eyes[0],bridge-eyes[1],nose-mouth,chin-mouth,eyes[1]-eyes[0]]
        alltests+=tests
        inside=f(0,.045,.24,.28); left=f(-.73,0,.20,.28); right=f(.73,0,.20,.28)
        cols += [inside,inside-.5*(left+right)]
    light=conv(tag+'_haar',lum,np.array(alltests)[:,None],stride=2,pad=k//2)
    col=conv(tag+'_pigment',sk,np.array(cols)[:,None],stride=2,pad=k//2)
    n=len(angles)
    # Each test can give limited support; in particular both eye sockets must be
    # supported rather than one shadow generating a large sum of correlations.
    thresholds=np.array([.015,.015,.012,.012,.009,.009,.012,.008,.040]); caps=np.array([.090,.090,.085,.085,.068,.068,.085,.063,.13]);
    a=np.vstack([np.eye(9),np.eye(9),-np.eye(9)[-1]])
    ramps=pos(tag+'_parts',conv(tag+'_part_gates',light,np.kron(np.eye(n),a),np.tile(np.r_[-thresholds,-thresholds-caps,-.04],n)))
    co=np.array([1.35,1.35,1.0,1.0,1.22,1.22,1.10,.65,-.65]); co=np.r_[co,-co,-.65]
    lightvote=conv(tag+'_part_vote',ramps,np.kron(np.eye(n),co[None]))
    colg=pos(tag+'_skin_gate',conv(tag+'_skin_bound',col,np.kron(np.eye(n),np.array([[1,0],[0,1],[0,-1]])),np.tile([-.10,0,-.05],n)))
    pigment=conv(tag+'_skin_vote',colg,np.kron(np.eye(n),np.array([[.27,.10,-.07]])))
    face=add(tag+'_mixture',lightvote,pigment)
    # Reduce alternative orientations with maxpool over a reshaped angle axis.
    if n>1:
        a=layer(tag+'_hypotheses','reshape',face,shape=[1,n,side*side])
        a=layer(tag+'_angle_max','maxpool',a,size=[n,1],stride=[n,1])
        face=layer(tag+'_angle_map','reshape',a,shape=[1,side,side])
    # Padding can fabricate a cheek at the boundary; require eye/nose/mouth
    # inside the frame. At larger scales a response covers more pixels and there
    # are fewer locations for accidental coincidences.
    sy,sx=np.mgrid[:side,:side]; x=2*sx; y=2*sy; h=2*side
    clipped=np.maximum(0,1-np.minimum(x,h-1-x)/(.48*hw))+np.maximum(0,1-np.minimum(y,h-1-y)/(.46*hh))
    threshold={1:.716,2:.667,3:.623,4:.59}[down]
    bias=dense(tag+'_level',pool(tag+'_source',face,side),np.zeros((side*side,1)),(-threshold-.28*clipped).ravel())
    s=add(tag+'_raw',face,layer(tag+'_level_map','reshape',bias,shape=[1,side,side]))
    face_scores.append(summarize(tag,s,side,.96,.10))

# Contour templates for figures with poorly visible or turned-away faces.
# Pose is specified as outline segments with an expected edge normal. Balance
# head/shoulders, torso, and legs separately so the longest part doesn't dictate
# the answer. The negative surround tests compare an outline to nearby texture.
body_scores=[]
for tag,down,side,height in [('figure24',1,12,6.9),('figure45',2,12,6.2),('figure70',2,12,9.0)]:
    src=orient if down==1 else pool(tag+'_scale',orient,down)
    st=2 if down==1 else 1
    k=int(2*np.ceil(height*.54)+1); x,y=coords(k); sigma=.49 if height<8 else .58
    bank=[]; masks=[]
    # dimensionless outline coordinates: the torso flares at the shoulders;
    # stride, lateral lean and aspect ratio vary across templates.
    for wide,lean,stride in [(1,0,.065),(1.38,0,.11),(.79,.085,.16),(1.15,-.10,.23)]:
        X=lambda p:(height*(p[0]*wide+lean*(p[1]+.42)),height*p[1]);
        groups=[]
        for group in range(3):
            oriented=np.zeros((4,k,k)); surround=np.zeros((k,k))
            for sign in [-1,1]:
                if group==0:
                    pts=[(0,-.49),(.051,-.485),(.068,-.425),(.050,-.348),(.071,-.320),(.15,-.284),(.167,-.237)]
                elif group==1:
                    pts=[(.164,-.244),(.188,-.12),(.185,-.031),(.138,.052),(.126,.126)]
                else:
                    pts=[(.112,.068),(.094+stride*.28,.27),(.062+stride,.463),(.104+stride,.484)]
                for p0,p1 in zip(pts,pts[1:]):
                    a,b=X((sign*p0[0],p0[1])),X((sign*p1[0],p1[1])); dx,dy=b[0]-a[0],b[1]-a[1]; le=np.hypot(dx,dy)
                    nx,ny=dy/le,-dx/le; theta=np.arctan2(ny,nx)
                    bins=np.maximum(0,1-np.abs(((np.arange(4)*np.pi/4-theta+np.pi/2)%np.pi)-np.pi/2)/(np.pi/4)); bins/=bins.sum()
                    for t0 in (.15,.5,.85):
                        cx=a[0]+t0*dx; cy=a[1]+t0*dy
                        oriented+=le/3*bins[:,None,None]*gauss(x,y,cx,cy,sigma,sigma)
                        surround+=le/3*gauss(x,y,cx+sign*1.8,cy,1.0,.90)
            oriented/=oriented.sum(); surround/=surround.sum()
            # Compare to local nontarget orientations AND a spatial surround.
            w=oriented-.195*oriented.sum(0)[None]-.105*surround[None]
            groups.append(w)
        bank.extend(groups)
        # Test for a patch of skin at the head. It only assists a shape match;
        # no skin is required for a strong contour match.
        cx,cy=X((0,-.407)); a=gauss(x,y,cx,cy,.50,.58); b=.5*(gauss(x,y,cx-1.5,cy,.6,.7)+gauss(x,y,cx+1.5,cy,.6,.7)); masks.append(1.38*a-.38*b)
    responses=conv(tag+'_outline',src,np.array(bank),stride=st,pad=k//2)
    n=4
    # Approximate a robust average over body parts; edges of an isolated pole
    # are not three body parts. A long straight pair is mainly vertical, so the
    # curved head/shoulder and articulated leg parts have independent weight.
    eye=np.eye(3*n); g=pos(tag+'_shape_gates',conv(tag+'_shape_bounds',responses,np.vstack([eye,eye]),np.r_[np.zeros(3*n),-np.tile([.30,.40,.34],n)]))
    mix=np.kron(np.eye(n),np.array([[1.30,1.00,1.20]])); shape=conv(tag+'_geometry',g,np.hstack([mix,-mix]))
    srcskin=pool(tag+'_hue_scale',skin,2*down)
    votes=conv(tag+'_head',srcskin,np.array(masks)[:,None],stride=st,pad=k//2)
    bonus=conv(tag+'_head_vote',pos(tag+'_head_gate',votes),.33*np.eye(n))
    votes=add(tag+'_match',shape,bonus)
    a=layer(tag+'_poses','reshape',votes,shape=[1,n,side*side]); a=layer(tag+'_pose_max','maxpool',a,size=[n,1],stride=[n,1]); a=layer(tag+'_map','reshape',a,shape=[1,side,side])
    sy,sx=np.mgrid[:side,:side]; ub=np.minimum(st*sx,st*side-1-st*sx); vb=np.minimum(st*sy,st*side-1-st*sy)
    penalty=.12*np.maximum(0,1-ub/(height*.19))+.16*np.maximum(0,1-vb/(height*.46))
    bias=dense(tag+'_level',pool(tag+'_source',a,side),np.zeros((side*side,1)),(-.58-penalty).ravel())
    body_scores.append(summarize(tag,add(tag+'_raw',a,layer(tag+'_level_map','reshape',bias,shape=[1,side,side])),side,.71,.13,body=True))

# Compress overlapping scale votes: reward agreement but don't count the same
# face found at three sizes as three people. Read peak and average separately.
def family(tag,items,alpha,beta):
    n=len(items); copies=[]
    for i,v in enumerate(items):
        w=np.zeros((n,1)); w[i]=1
        copies.append(dense(tag+'_lane'+str(i),v,w))
    joined=add(tag+'_values',*copies)
    col=layer(tag+'_column','reshape',joined,shape=[1,n,1]); pk=layer(tag+'_strongest','maxpool',col,size=[n,1],stride=[n,1])
    return add(tag+'_vote',dense(tag+'_max_gain',pk,[[alpha]]),dense(tag+'_mean_gain',joined,np.full((1,n),beta/n)))
heads=family('head_scales',patch_scores[:5],.93,.33)
faces=family('face_scales',face_scores,.93,.27)
bodies=family('body_scales',body_scores,.85,.23)

# Multiple small skin regions with a repeatable vertical relationship can
# distinguish people in action from isolated wooden, sandy or furry objects.
# Test a head-sized center-surround blob; allow an exposed forearm or hand to
# occur anywhere in either of two flank regions, and lower legs farther down.
# Use only the best CO-LOCATED configuration, not a count of all orange pixels.
limb_votes=[]
for tag,ds,size,diam in [('limb_small',2,24,1.45),('limb_medium',3,16,1.9),('limb_near',4,12,2.65)]:
    m=pool(tag+'_scale',skin,ds)
    k=7; x,y=coords(k); f=lambda a,b,c,d:gauss(x,y,a,b,c,d)
    c=f(0,0,max(.40,.23*diam),max(.44,.31*diam))
    lr=.5*(f(-1.05*diam,0,.52,.75)+f(1.05*diam,0,.52,.75)); ud=.5*(f(0,-1.06*diam,.64,.60)+f(0,1.06*diam,.64,.60))
    b=conv(tag+'_compactness',m,np.array([1.45*c-.45*lr,1.45*c-.45*ud,c-.5*(ud+lr)])[:,None],pad=k//2)
    slope=np.eye(3); bands=pos(tag+'_blob_bounds',conv(tag+'_blob_knots',b,np.vstack([slope,slope]),[-.15,-.15,-.05,-.90,-.90,-.64]))
    blob=conv(tag+'_blobs',bands,[[.37,.37,.64,-.37,-.37,-.64]])
    # Give strong discount to a pure hue edge that has no end in one direction.
    # Partially pool the limb locations before looking below the head. Border
    # padding for a nonnegative map can represent lack of supporting evidence.
    kernel=int(np.ceil(7*diam)); kernel+=1-kernel%2
    xo,yo=coords(kernel); tests=[]
    for xs,ys in [(-1.6,2.45),(1.6,2.45),(-.76,4.5),(.76,4.5)]:
        # Soft deformation via local max, followed by spatial averaging. Offset
        # to account for the valid 3x3 max window starting at its output index.
        tests.append(gauss(xo,yo,xs*diam-1,ys*diam-1,.50*diam,.66*diam))
    soft=pool(tag+'_limb_tolerance',blob,3,1,maximum=True)
    # Place the valid (size-2)^2 map at the top-left of a zero-extended grid.
    z=np.zeros((1,1,3,3)); z[0,0,2,2]=1
    soft=conv(tag+'_limb_grid',soft,z,pad=2)
    distant=conv(tag+'_sites',soft,np.array(tests)[:,None],pad=kernel//2)
    # Non-additive votes: once one site is strong it cannot compensate for
    # lack of a head by growing still stronger. A single arm is sufficient.
    z=np.eye(4); gates=pos(tag+'_site_gates',conv(tag+'_site_knots',distant,np.vstack([z,z]),[-.21]*4+[-.66]*4))
    bonuses=conv(tag+'_limb_support',gates,[[.37,.37,.20,.20,-.37,-.37,-.20,-.20]])
    h=conv(tag+'_head_strength',blob,[[.60]],-.72)
    s=add(tag+'_association',h,bonuses)
    limb_votes.append(summarize(tag,s,size,.47,.15))
limbs=family('limb_scales',limb_votes,.95,.17)

# Additional scene cues for outdoor activities. Grass in the lower frame, sky
# in the upper frame, etc., are intentionally quite weak cues in isolation.
rgb12=pool('coarse_rgb',im2,4)
colors=pos('scene_colors',conv('scene_color_planes',rgb12,[(G-R)*2.5,(G-B)*2.,(B-R)*2.,(G+B-2*R),.8*R+.2*B-G],[-.018,-.026,-.06,-.03,-.075]))
u,v=coords(12); field=(.4+.6*np.clip(v/5+.45,0,1)); field=field/field.sum(); sky=(.25+np.clip(-v/5,0,1)); sky=sky/sky.sum()
color_ctx=dense('field_context',colors,np.array([.28*field,.13*field,.17*sky,.10*sky,.08*np.ones_like(v)/144]).ravel(),-.027)
# Long, axis-aligned runs are usually architectural or mechanical, whereas
# people have mostly short boundaries that change direction. Require a run to
# remain strong after averaging ALONG its edge tangent.
line=np.zeros((4,4,9,9))
for j in range(4):
    if j==0: line[j,j,:,4]=1/9
    if j==2: line[j,j,4,:]=1/9
    if j==1: line[j,j]=np.fliplr(np.eye(9))/9
    if j==3: line[j,j]=np.eye(9)/9
stripe=pos('straight_runs',conv('long_lines',orient,line,-.26,stride=2))
runs=dense('linear_context',pool('line_density',stripe,8),[[-1.8,.64,-1.8,.64]])
# Detect spatially extensive, low-texture green turf with some foreground
# contrast. Max of green vegetation costs is inappropriate (an isolated tree
# isn't a field), so integrate coverage over the lower part of the frame.
field_hinges=pos('turf_costs',conv('turf_planes',rgb12,[R-G,B-G,.12*R-G,-G],[.025,.045,.03,.17]))
turf=pos('turf_pixel',conv('turf_membership',field_hinges,[[-16,-11,-6,-6]],1.0))
energy=pool('patch_edge_energy',mag,2)
rough=conv('turf_roughness',energy,[[1.8]*4],-.18)
carpet=pos('unbroken_grass',add('grass_contrast',turf,conv('roughness_cost',pos('roughness',rough),[[-1]])))
xx,yy=coords(12); lawn=1/(1+np.exp(-.85*(yy+1))); lawn=lawn/lawn.sum()
coverage=dense('grass_coverage',carpet,lawn.ravel(),-.15)
fg=dense('foreground_detail',mag,np.broadcast_to(1.7*layout/layout.sum(),(4,24,24)).ravel(),-.20)
# Fuzzy AND: clamp a + b - |a-b| to make a sports-scene bonus contingent on
# both the field coverage and material in front of that field.
a=pos('grass_extent',coverage); b=pos('field_detail',fg)
excess=pos('grass_without_detail',add('grass_excess',a,dense('inverted_detail',b,[[-1]])))
sport=add('playing_field',dense('grass_gain',a,[[.37]]),dense('grass_limit',excess,[[-.37]]))
# Limit context in both directions; the local part detectors should decide.
ctx=add('context_unbounded',scene,color_ctx,runs,sport)
ctxp=pos('context_limit_gates',dense('context_limit_planes',ctx,[[1],[-1]],[-.16,-.13]))
ctx=add('context',ctx,dense('context_bend',ctxp,[[-.80,.80]]))
# Person margin; no-person is the reference with a constant zero score.
score=add('person_evidence',heads,faces,bodies,limbs,skin_global,ctx)
dense('class_scores',score,[[0],[1]],[0,-.33])
(out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
np.savez(out/'weights.npz',**weights)
