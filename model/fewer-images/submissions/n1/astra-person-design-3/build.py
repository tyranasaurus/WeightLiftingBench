"""Hand-constructed RGB photo classifier. Usage: python3 build.py OUTPUT_DIR

No inputs to the construction other than the output directory. Units used to
specify the geometry of features are pixels on a particular pyramid level.
"""
from pathlib import Path
import json
import sys
import numpy as np

layers = []
weights = {}
shapes = {'image': (3, 96, 96)}


def layer(name, op, inputs, shape, **params):
    if isinstance(inputs, str):
        inputs = [inputs]
    assert name not in shapes
    layers.append(dict(name=name, op=op, inputs=inputs, **params))
    shapes[name] = tuple(shape)
    return name


def conv(name, src, w, b=0, s=1, p=0):
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 2:
        w = w[:, :, None, None]
    n, c, k, k2 = w.shape
    ci, h, l = shapes[src]
    assert c == ci and k == k2
    weights[name+'.W'] = w
    weights[name+'.b'] = np.broadcast_to(np.asarray(b, dtype=np.float32), (n,)).copy()
    return layer(name, 'conv', src, (n, (h+2*p-k)//s+1, (l+2*p-k)//s+1),
                 out=n, kernel=[k,k], stride=s, padding=p)


def rel(name, x):
    return layer(name, 'relu', x, shapes[x])


def add(name, xs):
    assert len(set(shapes[x] for x in xs)) == 1
    return layer(name, 'add', xs, shapes[xs[0]])


def pool(name, src, n, s=None, maximum=False):
    if s is None: s = n
    c, h, l = shapes[src]
    return layer(name, 'maxpool' if maximum else 'avgpool', src,
                 (c, (h-n)//s+1, (l-n)//s+1), size=[n,n], stride=[s,s])


def dense(name, src, w, b=0):
    w = np.asarray(w, np.float32).reshape(-1, int(np.prod(shapes[src])))
    weights[name+'.W'] = w
    weights[name+'.b'] = np.broadcast_to(np.asarray(b, dtype=np.float32), (len(w),)).copy()
    return layer(name, 'dense', src, (len(w),), out=len(w))


def grid(k):
    y, x = np.mgrid[:k, :k].astype(float)-(k-1)/2
    return x,y


def unit(g):
    return g/g.sum()


def gauss(x,y,cx,cy,sx,sy):
    return unit(np.exp(-.5*((x-cx)/sx)**2-.5*((y-cy)/sy)**2))


# Soft intersection of color half-spaces. Each hinge is a violation of a
# bound, not a signed response that can cancel violations of other bounds.
# Using a constant term and a luminance-dependent term handles shadows and
# highlights without the divisions in a chromaticity representation.
r,g,b = np.eye(3)
Y = .29*r + .58*g + .13*b
u, v = r-g, g-b
rgb = pool('rgb_antialias', 'image', 2)
planes = []

def bound(coef, const, tolerance):
    # coef.RGB + const should be positive inside the locus
    planes.append((-coef/tolerance, -const/tolerance))

bound(u-.044*Y, -.009, .036)
bound(-u+.51*Y, .035, .087)
bound(v-.003*Y, -.002, .047)
bound(-v+.37*Y, .035, .085)
bound(v-.26*u, .018, .040)
bound(-v+1.37*u, .015, .060)
bound(Y, -.085, .075)
bound(-Y, .95, .05)
# Weak-chroma colors (particularly achromatic pixels) should not produce
# a strong face response simply by lying near the hue wedge's vertex.
bound(u+.40*v-.068*Y, -.022, .032)
# A second broader hue region accommodates pinkish skin and color casts.
bound(u-.027*Y, -.008, .037)
bound(v+.055*Y, .012, .047)
bound(v-.13*u, .039, .045)
bound(-v+1.85*u, .018, .07)
bound(u+.28*v-.038*Y, -.020, .035)
P = rel('tint_violations', conv('tint_bounds', rgb, [a for a,z in planes], [z for a,z in planes]))
cw = np.zeros((2,len(planes)))
cw[0,:9] = -1
cw[1,[1,3,6,7,9,10,11,12,13]] = -1
sm = rel('tint_memberships', conv('tint_margin', P, cw, 1))
# Neutral gray must give essentially zero even for the relaxed locus.
S = conv('warm_skin', sm, [[.76,.24]])
L = conv('luminance', rgb, [Y])

# Local unsigned orientations, with a soft noise floor, gain control and
# saturation. The pair of hinges realizes clip((abs(derivative)-lo)/width).
# At a contour with tangent (tx,ty) the gradient direction is its normal.
angles = np.arange(4)*np.pi/4

def oriented(name, src):
    x,y=grid(3)
    env=np.exp(-.5*(x*x+y*y)/.8**2)
    bank=[]
    for a in angles:
        q=(x*np.cos(a)+y*np.sin(a))*env
        q=q/np.maximum(q,0).sum()
        bank.extend([q,-q])
    d=rel(name+'_rect', conv(name+'_derivative',src,np.array(bank)[:,None],p=1))
    # An oriented full-wave response less the projection onto its orthogonal
    # orientation. Suppresses the unconditional score of highly textured areas.
    w=np.zeros((8,8)); biases=[]
    for i in range(4):
        t=np.zeros(8); t[2*i:2*i+2]=1
        j=(i+2)%4; t[2*j:2*j+2]=-.42
        w[2*i]=w[2*i+1]=t/.16
        biases.extend([-.019/.16,-1-.019/.16])
    h=rel(name+'_clip',conv(name+'_directed',d,w,biases))
    return conv(name+'_orientations',h,np.repeat(np.eye(4),2,axis=1)*np.tile([1,-1],4))


def scalar_clip(name, src, lo, hi):
    width=hi-lo
    h=rel(name+'_hinges',conv(name+'_bounds',src,[[1/width],[1/width]],[-lo/width,-hi/width]))
    return conv(name,h,[[1,-1]])


# Common readout: the strongest object hypothesis matters most; use regions
# so that several moderate responses at different places can also vote. Cap
# the extremes, since a single small accidental match should not dominate.
votes=[]

def regions(name, src, central_penalty=.10, cap=1.55):
    c,h,w=shapes[src]
    # Tile the map into four or nine roughly equal regions. Include both
    # borders, allowing a little overlap when an axis isn't evenly divisible.
    nr = 3 if h>=9 else 2
    step=h//nr
    win=h-(nr-1)*step
    z=pool(name+'_regions',src,win,step,True)
    assert shapes[z][1:]==(nr,nr)
    # Allow each pose to compete with each other pose at the same location.
    # A channel reduction is done by reinterpreting the tensor's axes.
    if c>1:
        z=reshape(name+'_axes',z,(1,c,nr*nr))
        z=layer(name+'_poses','maxpool',z,(1,1,nr*nr),size=[c,1],stride=[c,1])
        z=reshape(name+'_tile_map',z,(1,nr,nr))
    # Mild center preference but no prohibition on border/corner people.
    ii,jj=grid(nr); spatial=-central_penalty*(ii**2+jj**2)/max(1,2*((nr-1)/2)**2)
    z=dense(name+'_prior',z,np.eye(nr*nr),spatial.ravel())
    z=reshape(name+'_grid',z,(1,nr,nr))
    top=pool(name+'_strongest',z,nr,maximum=True)
    pos=rel(name+'_credible_tiles',conv(name+'_center_tiles',z,[[1]],-.47))
    support=pool(name+'_support',pos,nr)
    support=conv(name+'_support_gain',support,[[.27]])
    vote=add(name+'_candidate',[top,support])
    capped=scalar_clip(name+'_bounded',vote,-.45,cap)
    return conv(name+'_vote',capped,[[cap+.45]],-.45)


def reshape(name, src, shp):
    # layer() has a positional shape argument; write the reshape setting too.
    layer(name,'reshape',src,shp)
    layers[-1]['shape']=list(shp)
    return name


# Each layout samples a face center, a nearby annulus, forehead/hair, eyes,
# cheeks and regions at either side of the face. Every region is normalized
# separately, so changing scale does not change its DC response.
def face(name, sk, lum, radius, aspect, stride=1, fine=1, ambiguity=0):
    rx, ry=radius, radius*aspect
    # ellipse with approx horizontal diameter 2*radius
    k=2*int(np.ceil(max(1.80*rx,1.47*ry)))+1
    x,y=grid(k)
    def G(a,b,c,d): return gauss(x,y,a*rx,b*ry,c*rx,d*ry)
    core=G(0,.02,.59,.57)
    left=G(-1.38,-.10,.33,.57); right=G(1.38,-.10,.33,.57)
    top=G(0,-1.30,.58,.31)
    hem=G(0,1.33,.62,.37)
    surround=.31*(left+right)+.24*top+.14*hem
    # Difference-of-regions localizes colored islands, rather than voting for
    # all of a wall of similar color. Hair and eye components are independent.
    sc=np.stack([core, core-surround, core-left,core-right,core-top])
    A=conv(name+'_skin_layout',sk,sc[:,None],s=stride,p=k//2)
    eyes=(G(-.39,-.21,.23,.145)+G(.39,-.21,.23,.145))*.5
    cheeks=(G(-.40,.20,.32,.23)+G(.40,.20,.32,.23))*.5
    forehead=G(0,-.61,.41,.19)
    bridge=G(0,-.13,.15,.28)
    eyesur=.42*cheeks+.35*forehead+.23*bridge
    hair=G(0,-.99,.66,.22)
    le=G(-.39,-.21,.22,.15); re=G(.39,-.21,.22,.15)
    mouth=G(0,.53,.35,.13)
    lip_surr=.47*G(0,.32,.37,.14)+.53*G(0,.73,.32,.17)
    lc=np.stack([core-hair,eyesur-eyes,le-re,lip_surr-mouth,core-.5*left-.5*right])
    B=conv(name+'_value_layout',lum,lc[:,None],s=stride,p=k//2)
    if fine!=1:
        # At the larger scales resolve the thin eyelids and lips using a
        # finer luminance map than the one needed for the skin outline.
        # Average analytic weights over each finer pixel's footprint and
        # align the centers to the coarser cells (including half-pixel shift).
        n=k*fine
        gx,gy=grid(n)
        raw=np.zeros((5,n,n))
        for dx in [-.25,.25]:
            for dy in [-.25,.25]:
                gx1,gy1=(gx+dx)/fine,(gy+dy)/fine
                def H(a,b,c,d):
                    return gauss(gx1,gy1,a*rx,b*ry,c*rx,d*ry)
                eye_l=H(-.39,-.21,.192,.112); eye_r=H(.39,-.21,.192,.112)
                check=.5*(H(-.40,.19,.28,.17)+H(.40,.19,.28,.17))
                smooth=H(0,.04,.56,.52)
                raw+=np.stack([smooth-H(0,-.91,.60,.22),
                        .43*check+.29*H(0,-.53,.39,.17)+.28*H(0,-.12,.145,.25)-.5*(eye_l+eye_r),
                        eye_l-eye_r,
                        .48*H(0,.32,.34,.11)+.52*H(0,.74,.32,.13)-H(0,.53,.32,.084),
                        smooth-.5*(H(-1.38,-.10,.33,.57)+H(1.38,-.10,.33,.57))])*.25
        # The sharp template handles well-resolved frontal faces. A little
        # of the broader template preserves tolerance for shifted features.
        wb=conv(name+'_fine_layout',L,raw[:,None]*.81,s=fine*stride,p=(k//2)*fine)
        coarse=conv(name+'_broad_detail',B,np.eye(5)*.19)
        B=add(name+'_merged_details',[wb,coarse])
    # Bounded soft tests: a very high contrast single edge cannot substitute
    # for several facial parts. Signed differences are in original RGB units.
    clips=[(.008,.20),(.004,.116),(.01,.13),(.002,.088),(.006,.24)]
    w=np.zeros((12,5)); bias=np.zeros(12)
    for i,(lo,hi) in enumerate(clips):
        w[2*i:2*i+2,i]=1/(hi-lo); bias[2*i:2*i+2]=[-lo/(hi-lo),-hi/(hi-lo)]
    lo,hi=clips[2]; w[10:12,2]=-1/(hi-lo);bias[10:12]=[-lo/(hi-lo),-hi/(hi-lo)]
    bh=rel(name+'_photo_tests',conv(name+'_photo_bounds',B,w,bias))
    reduce=np.zeros((5,12))
    for i in range(5): reduce[i,2*i:2*i+2]=[1,-1]
    reduce[2,10:12]=[1,-1]
    # Four scoring rules: a colored oval with a plausible rim; facial detail
    # with skin; grayscale frontal face; tiny backlit/half-profile head.
    # (The strongest side is not used in isolation; broad color transitions
    # should have much less support than an island.)
    sw=np.array([[.51,.91,0,0,.09], [.43,.43,0,0,.08],
                 [.055,.10,0,0,0], [.32,.85,0,0,.19]])
    photo=np.array([[.24,.16,-.035,.065,.075], [.28,.48,-.16,.23,.095],
                    [.34,.58,-.37,.36,.19], [.41,.075,0,.015,.13]])
    # Vertical bilateral flank agreement: penalize a failure on either side.
    # This also rejects skin against the picture boundary if it extends inward
    # uniformly or has no real top margin.
    misses=rel(name+'_flank_shortfalls',conv(name+'_flank_tests',A,
                  [[0,0,-1,0,0],[0,0,0,-1,0],[0,0,0,0,-1]], [.23,.23,.16]))
    penal=np.array([[-.25,-.25,-.14],[-.13,-.13,-.07],[0,0,0],[-.20,-.20,-.09]])
    # Penalize a large kernel centered too close to the picture edge. This is
    # an analytic spatial prior (not a measured distribution of face centers).
    base=conv(name+'_skin_score', A,sw,np.array([-.34,-.32,-.46,-.34])-ambiguity)
    vis=conv(name+'_features_score',bh,photo@reduce)
    fl=conv(name+'_flank_score',misses,penal)
    det=add(name+'_detection',[base,vis,fl])
    c,h,w=shapes[det]
    # Ignore hypotheses whose face center is outside the useful image. Use a
    # valid 1x1-equivalent selection via a small identity center kernel.
    border=max(1,int(np.ceil(.67*rx/stride)))
    select=np.zeros((c,c,2*border+1,2*border+1)); select[np.arange(c),np.arange(c),border,border]=1
    det=conv(name+'_interior',det,select)
    return regions(name,det,.12)


# Contour templates. A curve contributes votes for its local normal, with
# spatial and angular tolerance. Templates are constructed independently
# from lists of vertices, not by comparing several rendered appearances.
def path_kernel(k, parts, blur):
    x,y=grid(k); kernel=np.zeros((4,k,k))
    for points, importance in parts:
        tmp=np.zeros_like(kernel)
        for (x0,y0),(x1,y1) in zip(points,points[1:]):
            dx=x1-x0; dy=y1-y0
            length=np.hypot(dx,dy)
            if length<1e-8: continue
            # Distribute a segment between adjacent orientation bins.
            th=(np.arctan2(dy,dx)+np.pi/2)%np.pi
            f=4*th/np.pi; j=int(np.floor(f)); u=f-j
            n=max(1,int(np.ceil(length*2)))
            for i in range(n):
                t=(i+.5)/n
                blob=gauss(x,y,x0+dx*t,y0+dy*t,blur,blur)*length/n
                tmp[j%4]+=(1-u)*blob; tmp[(j+1)%4]+=u*blob
        kernel += importance*tmp/tmp.sum()
    return kernel


def circ(cx,cy,rx,ry,start,end):
    t=np.linspace(start*np.pi/180,end*np.pi/180,16)
    return list(zip(cx+rx*np.cos(t),cy+ry*np.sin(t)))


def mirror(path): return [(-x,y) for x,y in path]


def upper(name, e, sk, size, broad, stride):
    # Head -> neck indentation -> downward slope toward the shoulders. Parts
    # on opposite sides must both match, not just a long straight contour.
    r=size; ry=1.27*r; cy=-1.0*r
    k=2*int(np.ceil(3.45*r))+1
    x,y=grid(k)
    crown=circ(0,cy,r,ry,198,342)
    cheek=circ(0,cy,r,ry, -23,55)
    jaw=[(.58*r,cy+.81*ry),(.49*r,cy+1.15*ry)]
    shoulder=[(.50*r,cy+1.12*ry),(1.02*r,cy+1.37*ry),(broad*r,cy+1.63*ry),
              ((broad+.24)*r,cy+2.02*ry),((broad+.40)*r,cy+2.62*ry)]
    parts=[[(crown,1)],[(cheek,.54),(jaw,.12),(shoulder,.34)],
           [(mirror(cheek),.54),(mirror(jaw),.12),(mirror(shoulder),.34)],
           [(shoulder,1)],[(mirror(shoulder),1)]]
    ks=[path_kernel(k,group,max(.45,.19*r)) for group in parts]
    # Competing edge energy in the head/torso's neighborhood.
    surround=gauss(x,y,0,0,2*r,1.90*r)
    ks.append(np.tile(surround[None]/4,(4,1,1)))
    A=conv(name+'_contours',e,ks,s=stride,p=k//2)
    # Penalize missing parts and nonselective texture. The offset reflects
    # the fact that real contours can have gaps (hair, clothing, occlusion).
    base=np.array([[.45,.72,.72,.28,.28,-2.0], [.34,.48,.48,.20,.20,-1.45]])
    main=conv(name+'_outline_score',A,base,[-.26,-.32])
    misses=rel(name+'_missing',conv(name+'_part_floors',A,
                         [[0,-1,0,0,0,.48],[0,0,-1,0,0,.48],[-1,0,0,0,0,.36]], [.14,.14,.080]))
    pen=conv(name+'_agreement',misses,[[-.85,-.85,-.40],[-.40,-.40,-.18]])
    core=gauss(x,y,0,cy+.12*ry,.58*r,.59*ry)
    ext=(gauss(x,y,-1.54*r,cy,.34*r,.69*ry)+gauss(x,y,1.54*r,cy,.34*r,.69*ry))*.34
    ext+=.21*gauss(x,y,0,cy-1.34*ry,.65*r,.35*ry)+.11*gauss(x,y,0,cy+2*ry,.92*r,.49*ry)
    color=conv(name+'_face_color',sk,np.stack([.20*core-.17*ext, .95*core-.62*ext])[:,None],s=stride,p=k//2)
    det=add(name+'_detection',[main,pen,color])
    # Exclude padding-induced outlines. Allow a cropped torso below the head.
    border=max(1,int(round(r/stride)))
    pick=np.zeros((2,2,2*border+1,2*border+1));pick[[0,1],[0,1],border,border]=1
    return regions(name,conv(name+'_interior',det,pick),.10)


# Body outline: the width of a head is smaller than the shoulders; the trunk
# extends several head radii downwards. Models with separated and close legs
# allow walking as well as standing and different image aspect ratios.
def body(name,e,sk):
    k=13; x,y=grid(k); bank=[]; colors=[]
    for breadth,split in [(.79,.36),(1.12,.48),(1.28,1.35),(.85,1.40)]:
        r=.85*breadth; cy=-4.33
        crown=circ(0,cy,r,1.03,186,354)
        side=circ(0,cy,r,1.03,-28,57)
        sh=[(.50*r,-3.47),(.53*breadth,-3.24),(1.52*breadth,-2.88),
            (1.79*breadth,-1.76),(1.89*breadth,-.30)]
        torso=[(1.26*breadth,-1.92),(1.13*breadth,-.72),(1.11*breadth,.48)]
        leg=[(1.06*breadth,.56),((.90+.37*split)*breadth,2.65),((.69+.75*split)*breadth,4.94)]
        inner=[(.06*breadth,1.09),(.27*split*breadth,2.92),(.65*split*breadth,4.98)]
        headpart=[(crown,.40),(side,.30),(mirror(side),.30)]
        lpart=[(sh,.54),(torso,.12),(leg,.22),(inner,.12)]
        rpart=[(mirror(q),a) for q,a in lpart]
        for group in [headpart,lpart,rpart]: bank.append(path_kernel(k,group,.52))
        co=gauss(x,y,0,cy+.17,.58*breadth,.62)
        ex=.37*gauss(x,y,-1.82*breadth,cy,.47*breadth,.85)+.37*gauss(x,y,1.82*breadth,cy,.47*breadth,.85)
        ex+=.26*gauss(x,y,0,-1.62,1.0*breadth,.76)
        colors.append(.75*co-.59*ex)
    bank.append(np.tile(gauss(x,y,0,-.4,2.5,4.2)[None]/4,(4,1,1)))
    A=conv(name+'_contours',e,bank,p=k//2)
    w=np.zeros((4,13)); m=np.zeros((8,13))
    for i in range(4):
        w[i,3*i:3*i+3]=[.54,1.02,1.02];w[i,-1]=-2.10
        for j in range(2): m[2*i+j,3*i+1+j]=-1;m[2*i+j,-1]=.45
    main=conv(name+'_figure_score',A,w,-.29)
    missing=rel(name+'_missing',conv(name+'_side_floors',A,m,.11))
    pen=conv(name+'_agreement',missing,np.repeat(-.90*np.eye(4),2,1))
    color=conv(name+'_head_colors',sk,np.array(colors)[:,None],p=k//2)
    det=add(name+'_detection',[main,pen,color])
    crop=np.zeros((4,4,3,3));crop[np.arange(4),np.arange(4),1,1]=1
    return regions(name,conv(name+'_interior',det,crop),.12)


# The face pyramid covers about 4 to 53 original pixels in width, with
# different aspect ratios; the two smallest levels capture very little
# detail and have many search locations, so require stronger evidence.
Ls={1:L}; Ss={1:S}
for f in [2,3,4,6]:
    Ls[f]=pool('Y_scale'+str(f),L,f)
    Ss[f]=pool('skin_scale'+str(f),S,f)
F=[]
for i,(f,r,aspect,step) in enumerate([(1,1.12,1.33,1),(1,1.82,1.24,2),
                         (2,1.42,1.52,1),(2,2.12,1.16,1),(3,2.20,1.48,1),
                         (4,2.30,1.19,1),(6,2.20,1.28,1)]):
    F.append(face('face'+str(i),Ss[f],Ls[f],r,aspect,step,f,
                  [.28,.17,.14,.075,.025,0,0][i]))
E={}
for f in [1,2,3,4]: E[f]=oriented('edge'+str(f),Ls[f])
U=[]
for i,(f,r,broad,step) in enumerate([(1,1.28,2.22,2),(2,1.20,2.30,1),
                  (2,1.89,2.03,1),(3,1.80,2.25,1),(4,2.02,1.99,1)]):
    U.append(upper('bust'+str(i),E[f],Ss[f],r,broad,step))
B=[body('figure'+str(f),E[f],Ss[f]) for f in [2,3,4]]

# Complement unpolarized contours with a signed segmentation test: the two
# sides of a shirt should both differ from their respective backgrounds in
# the *same* direction. Rectifying the weaker side makes a straight edge
# alone insufficient. Repeat for the head, and allow a different sign there.
def clothing(name, f):
    k=13; x,y=grid(k); stride=2 if f==1 else 1
    ga=lambda a,b,c,d: gauss(x,y,a,b,c,d)
    hp=ga(0,-3.91,.58,.71)
    torso=.52*ga(0,-1.70,.95,.80)+.48*ga(0,-.32,.84,.83)
    flanks=[torso-ga(i*2.62,-1.13,.62,1.09) for i in [-1,1]]
    edge=.40*ga(-1.62,-3.88,.43,.83)+.40*ga(1.62,-3.88,.43,.83)+.20*ga(0,-5.52,.64,.39)
    flanks.extend([hp-edge, hp-torso])
    # Bring a vertical pattern of contrast across the shoulder line into the
    # torso score. It is the absolute color difference that matters, not
    # white clothes vs dark clothes.
    ups=.5*(ga(-1.23,-3.25,.40,.41)+ga(1.23,-3.25,.40,.41))
    sh=.5*(ga(-1.23,-1.94,.42,.63)+ga(1.23,-1.94,.42,.63))
    flanks.append(sh-ups)
    d=conv(name+'_differences',Ls[f],np.array(flanks)[:,None],s=stride,p=k//2)
    # Get abs(left-right) and all remaining signed components, then express
    # min(l,r)=(l+r-abs(l-r))/2 and min(-l,-r) similarly.
    err=rel(name+'_asymmetry',conv(name+'_difference',d,[[1,-1,0,0,0],[-1,1,0,0,0]]))
    pair=conv(name+'_bilateral',d,[[.5,.5,0,0,0],[-.5,-.5,0,0,0]])
    pair=add(name+'_weaker_side',[pair,conv(name+'_subtract_asymmetry',err,[[-.5,-.5],[-.5,-.5]])])
    def bounded_abs(src, inds, limits, tag):
        ww=[];bb=[]; c=shapes[src][0]
        for j,(lo,hi) in zip(inds,limits):
            for sign in [1,-1]:
                a=np.eye(c)[j]*sign/(hi-lo);ww.extend([a,a]);bb.extend([-lo/(hi-lo),-hi/(hi-lo)])
        rr=rel(name+'_'+tag+'_hinges',conv(name+'_'+tag+'_limits',src,ww,bb))
        return conv(name+'_'+tag,rr,np.kron(np.eye(len(inds)),[[1,-1,1,-1]]))
    # For pair, use only its POSITIVE differences, since its negative ones
    # do not imply there is bilateral contrast. The two polarities compete.
    hh=rel(name+'_shirt_hinges',conv(name+'_shirt_limits',pair,np.repeat(np.eye(2),2,0)/.23,
                                     np.tile([-.025/.23,-1-.025/.23],2)))
    shirt=conv(name+'_shirt',hh,[[.50,-.50,.50,-.50]])
    v=bounded_abs(d,[2,3,4],[(.015,.225),(.018,.25),(.014,.22)],'part_contrast')
    vc=conv(name+'_part_votes',v,[[.27,.19,.18]],-.33)
    col=conv(name+'_face_color',Ss[f],(.64*hp-.43*edge)[None,None],s=stride,p=k//2)
    # The body's lower end isn't needed in this branch: a table or occluder
    # often hides it. Test the outline at the shoulder/neck level instead.
    e=E[f]; fn=ga(-1.53,-1.68,.50,.82)+ga(1.53,-1.68,.50,.82)
    hn=ga(-.91,-3.94,.37,.67)+ga(.91,-3.94,.37,.67)
    bg=ga(0,-1.74,2.35,2.41)
    ek=np.tile((-bg*.46/4)[None],(4,1,1));ek[0]+=.14*fn+.10*hn
    ek[2]+=.16*ga(0,-5.03,.64,.37)
    ev=conv(name+'_outline',e,ek[None],s=stride,p=k//2)
    sc=add(name+'_joint',[shirt,vc,col,ev]); cutoff=2 if f<=3 else 1
    cut=np.zeros((1,1,1+2*cutoff,1+2*cutoff));cut[0,0,cutoff,cutoff]=1
    return regions(name,conv(name+'_interior',sc,cut),.11)

C=[clothing('clothed'+str(f),f) for f in [1,2,3,4]]


def maximum(name, xs):
    n=len(xs)
    vec=add(name+'_collect',[conv(name+'_place'+str(i),v,np.eye(n)[:,i:i+1]) for i,v in enumerate(xs)])
    row=reshape(name+'_row',vec,(1,1,n))
    return layer(name,'maxpool',row,(1,1,1),size=[1,n],stride=[1,1])


bestF=maximum('best_face',F); bestU=maximum('best_bust',U); bestB=maximum('best_figure',B)
# Scale consistency: the best face and one of its neighbor scales being
# strong is more informative than one isolated spike. Use the family max,
# the family mean, and the max across families as complementary votes.
def meanvote(name,vs): return conv(name,add(name+'_sum',vs),[[1/len(vs)]])
meanF=meanvote('face_scale_support',F)
meanU=meanvote('bust_scale_support',U)
meanB=meanvote('figure_scale_support',B)

# Part association in the same place, unlike the global votes above. Search
# for a localized head above a shoulder outline and allow an optional second
# exposed area (hands, forearms) farther down from the head. Interior skin
# alone is not enough: the candidate should contrast with the area to its
# sides and/or show internal facial detail.
linked=[]
for i,(fi,ui,a,step,radius) in enumerate([(0,0,1,2,1.12),(2,1,2,1,1.42),
                     (3,2,2,1,2.12),(4,3,3,1,2.20),(5,4,4,1,2.30)]):
    name='association'+str(i)
    face_sc='face'+str(fi)+'_detection'; upper_sc='bust'+str(ui)+'_detection'
    cs,h,w=shapes[upper_sc]; bs=[1.28,1.20,1.89,1.80,2.02][i]
    # Look upwards from the neck, allowing uncertainty in the head center.
    # The Gaussian sits on a grid of face scores (one cell per pyramid pixel).
    size=2*int(np.ceil(bs+1))+1
    x,y=grid(size)
    fw=gauss(x,y,0,-bs,max(.47,.23*bs),max(.45,.20*bs))
    # The particular shape/layout hypotheses here are kept separate until
    # after they have accumulated local support from the shoulder detector.
    head=conv(name+'_above_shoulder',face_sc,np.stack([np.eye(4)[c,:,None,None]*fw*.65 for c in [0,1,2]]),s=step,p=size//2)
    bust=conv(name+'_shoulders',upper_sc,[[.31,.15],[.30,.16],[.46,.07]])
    # Hand search is imprecise, so its contribution is only a bonus. Each
    # side independently searches a neighborhood; use broad envelopes so
    # this also responds to an uncovered forearm in a short-sleeved shirt.
    n=2*int(np.ceil(3.2*radius))+1; xx,yy=grid(n)
    paint=[]
    for side in [-1,1]:
        hand=gauss(xx,yy,side*2.06*radius,2.29*radius,.67*radius,.80*radius)
        around=.48*gauss(xx,yy,side*2.83*radius,2.18*radius,.79*radius,1.2*radius)
        around+=.52*gauss(xx,yy,side*.60*radius,1.97*radius,.72*radius,.9*radius)
        paint.append(hand-.72*around)
    loc=rel(name+'_arm_evidence',conv(name+'_arm_color',Ss[a],np.array(paint)[:,None],-.08,s=step,p=n//2))
    arms=conv(name+'_exposed_arms',loc,[[.20,.20],[.20,.20],[.18,.18]])
    sc=add(name+'_local',[head,bust,arms])
    border=max(1,int(round(radius/step))); crop=np.zeros((3,3,2*border+1,2*border+1))
    crop[np.arange(3),np.arange(3),border,border]=1
    linked.append(regions(name,conv(name+'_interior',sc,crop),.10))
best_link=maximum('best_agreement',linked)

# An additional low-frequency cue: concentration of skin in the upper and
# middle interior, compared with the whole picture. Absolute area alone
# would give an excessive vote for a wood floor or a color cast.
ss=Ss[4]; sz=shapes[ss][1]; xx,yy=grid(sz); X=xx*2/sz; V=yy*2/sz
center=unit(np.exp(-.5*((X/.58)**2+((V+.15)/.59)**2)))
whole=np.ones_like(center)/center.size
concentration=reshape('skin_center_scalar',dense('skin_center_contrast',ss,[(center-whole).ravel()]),(1,1,1))
localized=scalar_clip('centered_skin',concentration,-.04,.17)
# A soft cap prevents a pervasive warm cast from guaranteeing a positive.
area=pool('area',ss,sz)
area_positive=scalar_clip('skin_present',area,.015,.17)
area_excess=scalar_clip('skin_pervasive',area,.25,.65)
# Interior vs margins of the image: composed figure photographs often have
# higher interior gradient energy, whereas a uniform geometric scene does
# not. This is deliberately a small supporting vote, not a primary detector.
eh=shapes[E[2]][1]; X,V=grid(eh); X=X*2/eh; V=V*2/eh
center=unit(np.exp(-.5*((X/.48)**2+((V+.04)/.65)**2)))
edgeW=np.tile((center-1/(eh*eh))[None]/4,(4,1,1))
salience=reshape('composition_scalar',dense('interior_detail',E[2],edgeW.ravel()),(1,1,1))
central_detail=scalar_clip('centered_detail',salience,-.035,.072)

# All detector families use dimensionless margins. Retain strong single
# detections as well as some consensus between distinct families.
bestC=maximum('best_clothed',C)
best=maximum('best_object',[bestF,bestU,bestB,best_link,bestC])
terms=[(best,.285),(bestF,.275),(bestU,.115),(bestB,.08),(best_link,.14),(bestC,.105),
       (meanF,.060),(meanU,.035),(meanB,.020),
       (localized,.12),(area_positive,.067),(area_excess,-.17),
       (central_detail,.068)]
raw=add('evidence',[conv('term'+str(i),t,[[gain]]) for i,(t,gain) in enumerate(terms)])
dense('class_scores',raw,[[0],[1]],[0,-1.155])

out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
(out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
np.savez(out/'weights.npz',**weights)
