"""Write a hand-designed, data-independent person detector. Usage: build.py OUTPUT"""
import json
import sys
from pathlib import Path
import numpy as np

layers = []
weights = {}
shapes = {'image': (3,96,96)}

def put(name,op,src,outshape,**kw):
    layers.append(dict(name=name, op=op, inputs=src if isinstance(src,list) else [src], **kw))
    shapes[name]=outshape
    return name

def conv(name,src,w,b=0,p=0,s=1):
    w=np.asarray(w,dtype=np.float32)
    if w.ndim==2: w=w[:,:,None,None]
    out,c,k,k2=w.shape
    assert c==shapes[src][0] and k==k2
    h,v=shapes[src][1:]
    weights[name+'.W']=w
    weights[name+'.b']=np.zeros(out,np.float32)+np.asarray(b,np.float32)
    return put(name,'conv',src,(out,(h+2*p-k)//s+1,(v+2*p-k)//s+1),out=out,kernel=[k,k],stride=s,padding=p)

def dense(name,src,w,b=0):
    w=np.asarray(w,np.float32).reshape(-1,int(np.prod(shapes[src])))
    weights[name+'.W']=w
    weights[name+'.b']=np.zeros(len(w),np.float32)+np.asarray(b,np.float32)
    return put(name,'dense',src,(len(w),),out=len(w))

def rel(name,src): return put(name,'relu',src,shapes[src])

def add(name,*src):
    assert all(shapes[x]==shapes[src[0]] for x in src)
    return put(name,'add',list(src),shapes[src[0]])

def pool(name,src,k,s,how='avg'):
    if isinstance(k,int): k=k,k
    if isinstance(s,int): s=s,s
    c,h,w=shapes[src]
    return put(name,how+'pool',src,(c,(h-k[0])//s[0]+1,(w-k[1])//s[1]+1),size=list(k),stride=list(s))

def reshape(name,src,*shape):
    assert np.prod(shape)==np.prod(shapes[src])
    return put(name,'reshape',src,shape,shape=list(shape))

def maxchan(name,src):
    c,h,w=shapes[src]
    x=reshape(name+'_stack',src,1,c,h*w)
    x=pool(name+'_max',x,(c,1),(c,1),'max')
    return reshape(name,x,1,h,w)

def ceilrect(name,src,cap):
    hi=rel(name+'_excess',conv(name+'_offset',src,[[1]],-cap))
    return add(name,src,conv(name+'_minus',hi,[[-1]]))

def squash(name,src,gain=1):
    # Bounded signed contrast, +/- 1.
    c=shapes[src][0]
    a=rel(name+'_ramps',conv(name+'_intervals',src,gain*np.vstack([np.eye(c),np.eye(c)]),[1]*c+[-1]*c))
    return conv(name,a,np.hstack([np.eye(c),-np.eye(c)]),-1)

def taper(n,ay,ax):
    y,x=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n,indexing='ij')
    return 1-ay*(2*y-1)**2-ax*(2*x-1)**2

rgb_lum=np.array([.28,.57,.15])
base=pool('rgb_half','image',2,2)
# Chroma coordinates r, g, b are gamma-encoded. Use soft half-plane
# intersections, not a threshold on red alone. Graded boundaries let pale,
# partly shaded, and dark skin contribute. Very low light has little chroma.
r,g,b=np.eye(3)
xrg=r-g; xgb=g-b
planes=[(xrg-.028*g,-.024,.032),       # a neutral grey cannot clear both ramps
        (r-b-.045*g,-.027,.045),       # require actual warm chroma, not just brightness
        (xgb+.04*g,.002,.070),         # reject pink/purple, retaining rosy skin
        (.88*xrg-xgb,.018,.080),       # green-yellow side
        (1.8*g-r,.035,.145),           # heavily saturated red/orange side
        (1.92*b-g,.055,.14),           # heavily saturated yellow side
        (r,-.058,.080),                # reject nearly black
        (r+g,-.25,.18)]
loss=rel('skin_outside',conv('skin_halfplanes',base,[-v/s for v,c,s in planes],[-c/s for v,c,s in planes]))
skin=rel('skin',conv('skin_intersection',loss,-np.ones((1,len(planes))),1))
lum=conv('lightness',base,[rgb_lum])
raw={48:base}; skinp={48:skin}; lum_p={48:lum}
for n in (24,16,12,8):
    k=48//n
    raw[n]=pool('rgb_'+str(n),base,k,k)
    skinp[n]=pool('skin_'+str(n),skin,k,k)
    lum_p[n]=conv('luma_'+str(n),raw[n],[rgb_lum])

# An oriented edge field with a bounded response. Subtracting local clutter
# from the edge agreements makes an isolated contour different from texture.
angles=np.arange(4)*np.pi/4
Y,X=np.mgrid[-1:2,-1:2]
grad=np.array([(X*np.cos(a)+Y*np.sin(a))*np.exp(-.66*(X**2+Y**2)) for a in angles])
grad/=np.maximum(grad,0).sum((1,2))[:,None,None]

def edgefield(src,n):
    # Add only interior edges: the image border is not a contour.
    a=conv('direction_'+str(n),src,np.vstack([grad,-grad])[:,None],p=0)
    z=rel('half_edges_'+str(n),a)
    # Approximate strongest orientation via subtraction of the orthogonal edge.
    m=np.eye(4)-.56*np.roll(np.eye(4),2,axis=1)
    h=rel('oriented_'+str(n),conv('edge_select_'+str(n),z,np.hstack([m,m])))
    excess=rel('edge_saturation_'+str(n),conv('edge_over_'+str(n),h,6*np.eye(4),-1))
    return add('edge_'+str(n),conv('edge_strength_'+str(n),h,6*np.eye(4)),conv('edge_subtract_'+str(n),excess,-np.eye(4)))

# Skin islands. Each center is surrounded by several separately tested regions;
# this prevents a single long warm/cool boundary looking like a complete head.
# Dimensions are measured in feature-map cells, not taken from any photo.
def skin_heads(n):
    prefix='head'+str(n)+'_'
    s=skinp[n]; l=lum_p[n]
    k=9; mid=k//2
    yy,xx=np.mgrid[-mid:mid+1,-mid:mid+1]
    def ga(x,y,sx,sy):
        q=np.exp(-.5*((xx-x)**2/sx**2+(yy-y)**2/sy**2));return q/q.sum()
    variants=[(.73,.96),(1.23,1.34),(1.12,1.88)] if n!=48 else [(.76,.94),(1.22,1.42)]
    sw=[]; lw=[]; direct=[]; off=[]
    # One score per variant. Rectified terms penalize missing evidence, so
    # strong hair contrast cannot make up for a face consisting of a flat wall.
    for sx,sy in variants:
        core=ga(0,.12*sy,sx*.74,sy*.66)
        hi=ga(0,-1.75*sy,sx*.96,.55*sy)
        lo=ga(0,1.76*sy,sx*1.2,.53*sy)
        left=ga(-1.88*sx,-.22*sy,.58*sx,sy*.82)
        right=left[:,::-1]
        surround=.29*(left+right)+.22*hi+.20*lo
        sw.extend([core,core-surround,core-left,core-right,core-hi,core-lo])
        # cap/hair often darker than forehead and cheeks.
        cheek=ga(0,.18*sy,.75*sx,.67*sy)
        crown=ga(0,-1.14*sy,.93*sx,.40*sy)
        lw.extend([cheek-crown,cheek-ga(-1.35*sx,0,.48*sx,.75*sy),cheek-ga(1.35*sx,0,.48*sx,.75*sy)])
    count=len(variants)
    cs=conv(prefix+'skin_samples',s,np.array(sw)[:,None],p=mid)
    hl=conv(prefix+'light_samples',l,np.array(lw)[:,None],p=mid)
    # Compute bounded hair/shadow evidence on all scales.
    hair=squash(prefix+'hair',hl,5.0)
    mh=np.zeros((count,count*3)); ms=np.zeros((count,count*6))
    for i in range(count):
        ms[i,6*i:6*i+6]=[.19,.46,0,0,0,0]
        mh[i,3*i:3*i+3]=[.095,.025,.025]
    # Penalize absent skin/adjacent warm surfaces by hinges. This also caps
    # how much an extremely strong difference on just one side helps.
    lim=np.tile([.63,.45,.43,.43,.41,.37],count)
    pen=rel(prefix+'deficits',conv(prefix+'minimums',cs,-np.eye(6*count),lim))
    mp=np.zeros_like(ms)
    for i in range(count):mp[i,6*i:6*i+6]=[-.45,-.22,-.16,-.16,-.12,-.12]
    sc=add(prefix+'variants',conv(prefix+'core_rule',cs,ms),conv(prefix+'penalties',pen,mp),conv(prefix+'hair_rule',hair,mh))
    # Small heads may appear anywhere. A face cut by the outer border has
    # less reliable surround evidence, so discount the artificial padding.
    top=maxchan(prefix+'best_shape',sc)
    top=rel(prefix+'positive',top)
    # Dense position-wise addition implemented on a compact (pooled) grid.
    top=pool(prefix+'nearby',top,n//8,n//8,'max')
    return top

heads=[skin_heads(n) for n in (48,24,16,8)]

# Frontal face appearance: distinct checks for both eye sockets, cheeks,
# nose bridge, mouth and the darker frame above and beside a face. Unlike a
# single correlation, these checks cannot all be satisfied by one dark stripe.
def face_pattern(n):
    pre='face'+str(n)+'_'; k=9; yy,xx=np.mgrid[-4:5,-4:5]
    def bell(x,y,sx,sy):
        a=np.exp(-.5*(((xx-x)/sx)**2+((yy-y)/sy)**2));return a/a.sum()
    allw=[]; skinw=[]
    for dx,dy in [(1.65,2.25),(2.28,2.18),(1.65,3.05)]:
        eye1=bell(-.54*dx,-.32*dy,.26*dx,.17*dy); eye2=eye1[:,::-1]
        lcheek=bell(-.52*dx,.21*dy,.33*dx,.20*dy); rcheek=lcheek[:,::-1]
        brow=bell(0,-.82*dy,.6*dx,.23*dy)
        nose=bell(0,-.13*dy,.22*dx,.30*dy)
        mouth=bell(0,.58*dy,.44*dx,.165*dy)
        chin=bell(0,.87*dy,.44*dx,.16*dy)
        eyes=.5*(eye1+eye2); cheeks=.5*(lcheek+rcheek)
        allw.extend([lcheek-eye1,rcheek-eye2,brow-eyes,nose-eyes,.68*cheeks+.32*chin-mouth,
                     .6*brow+.4*cheeks-bell(0,-1.32*dy,.90*dx,.26*dy),
                     cheeks-.5*(bell(-1.34*dx,.13*dy,.35*dx,.67*dy)+bell(1.34*dx,.13*dy,.35*dx,.67*dy))])
        cen=.7*cheeks+.3*brow
        ring=.34*(bell(-1.43*dx,0,.32*dx,.77*dy)+bell(1.43*dx,0,.32*dx,.77*dy))+.32*bell(0,-1.43*dy,.82*dx,.29*dy)
        skinw.extend([cen,cen-ring])
    # For a genuine facial configuration several small differences should
    # have the expected sign; cap each so a single black edge cannot dominate.
    z=conv(pre+'parts',lum_p[n],np.array(allw)[:,None],p=4)
    z=squash(pre+'bounded',z,13)
    ss=conv(pre+'colour',skinp[n],np.array(skinw)[:,None],p=4)
    wl=np.zeros((3,21)); ws=np.zeros((3,6))
    for i in range(3):
        wl[i,7*i:7*i+7]=[.12,.12,.105,.13,.12,.045,.045]
        ws[i,2*i:2*i+2]=[.15,.13]
    v=add(pre+'evidence',conv(pre+'constellation',z,wl),conv(pre+'complexion',ss,ws))
    v=rel(pre+'positive',maxchan(pre+'best',v))
    v=pool(pre+'nearby',v,n//8,n//8,'max') if n%8==0 else pool(pre+'nearby',v,3,3,'max')
    return v

faces=[face_pattern(n) for n in (48,24,16,8)]

# Shape-only and colour-contrast person templates. Continuous parametric
# contours for the head, shoulders, torso, and legs. Each contour is its own
# template. Spatial weights are evaluated analytically at the filter centers.
def contour_template(spec,k):
    aspect,pose,side,slant=spec
    mid=(k-1)/2
    yy,xx=np.mgrid[-mid:mid+1,-mid:mid+1]
    field=np.zeros((4,k,k))
    def seg(a,b,weight,width=.68):
        x1,y1=a; x2,y2=b
        # Shear + horizontal stretch, to allow bent bodies and squashed photos.
        tx=lambda x,y: (aspect*x+slant*y)*side
        x1,x2=tx(x1,y1),tx(x2,y2)
        vx,vy=x2-x1,y2-y1; lens=vx*vx+vy*vy
        if lens<1.e-8:return
        u=np.clip(((xx-x1)*vx+(yy-y1)*vy)/lens,0,1)
        d2=(xx-x1-u*vx)**2+(yy-y1-u*vy)**2
        q=np.exp(-.5*d2/width**2); q/=q.sum()
        # Squared angular selectivity of unsigned gradient orientation.
        phi=np.arctan2(vx,-vy)
        al=np.maximum(0,np.cos(2*(angles-phi)))**2
        field[:]+=weight*np.sqrt(lens)*(al/al.sum())[:,None,None]*q
    if pose=='bust':
        head=(0,-1.12,1.42,2.03)
        paths=[([(-1.04,.13),(-.94,.85),(-2.93,1.58),(-3.72,2.78),(-4.06,4.2)],.95)]
    else:
        head=(-.10 if pose=='stride' else 0,-3.19,.63,.85)
        paths=[([(-.44,-2.42),(-1.23,-2.08),(-1.54,-.88),(-1.68,.05)],.95),
               ([(-.95,-1.15),(-.84,.29),(-.88,.89)],.75)]
    hx,hy,rx,ry=head
    # Full head oval, with a slightly weaker chin / neck junction.
    a=np.linspace(0,2*np.pi,17)
    for u,v in zip(a[:-1],a[1:]):
        seg((hx+rx*np.cos(u),hy+ry*np.sin(u)),(hx+rx*np.cos(v),hy+ry*np.sin(v)),1.12 if np.sin((u+v)/2)<.65 else .43,.59)
    for path,w in paths:
        for flip in (-1,1):
            pts=[(flip*x,y) for x,y in path]
            for u,v in zip(pts,pts[1:]):seg(u,v,w)
    if pose!='bust':
        for flip in (-1,1):
            knee=1.23 if pose=='stride' else .62
            foot=1.93 if pose=='stride' else .72
            for start,delta in [(.86,.30),(.12,-.24)]:
                path=[(flip*start,.72),(flip*(knee+delta),2.46),(flip*(foot+delta*.72),4.03)]
                for u,v in zip(path,path[1:]):seg(u,v,.8,.65)
    # Center-surround contour evidence, normalized for this template alone.
    pos=field/field.sum()
    bg=np.exp(-.5*((xx/3.0)**2+(yy/3.85)**2));bg/=bg.sum()
    return 1.62*pos-.54*pos.sum(0)[None,:,:]/4-.93*bg[None,:,:]/4

# Appearance-independent torso contrast. The comparison planes are selected
# once by hand and have both signs; no garment or background colour is assumed.
planes_rgb=np.array([rgb_lum,.9*r-.72*g-.18*b, .85*b-.58*g-.27*r])

def people(n,ed):
    pre='figure'+str(n)+'_'; k=9; mid=4; yy,xx=np.mgrid[-4:5,-4:5]
    def g2(x,y,sx,sy):
        a=np.exp(-.5*((xx-x)**2/sx**2+(yy-y)**2/sy**2));return a/a.sum()
    specs=[(.9,'stand',1,0),(1.34,'stand',1,0),(1.,'stride',1,.12),(1.,'stride',1,-.12),(.91,'bust',1,0),(1.48,'bust',1,0)]
    filt=np.array([contour_template(sp,k) for sp in specs]); count=len(specs)
    # ed is on the one-cell inset grid. Use padding 3 for RGB & skin, 4 for ed.
    ec=conv(pre+'contours',ed,filt,p=mid)
    cf=[]; sf=[]
    for aspect,pose,side,slant in specs:
        if pose=='bust':
            cy=2.88; w=1.88*aspect; h=1.08; hx=0; hy=-1.15; rx=1.14*aspect; ry=1.63
        else:
            cy=-.91; w=.85*aspect; h=1.28; hx=-3.15*slant;hy=-3.15;rx=.64*aspect;ry=.75
        inside=g2(slant*cy,cy,w*.68,h*.71)
        outside1=g2(-2.24*w+slant*cy,cy,.66*w,h*.86)
        outside2=g2(2.24*w+slant*cy,cy,.66*w,h*.86)
        for sidebg in (outside1,outside2):
            for plane in planes_rgb:cf.append(plane[:,None,None]*(inside-sidebg))
        center=g2(hx,hy,rx*.73,ry*.68)
        adj=.37*(g2(hx-2.48*rx,hy,.70*rx,ry)+g2(hx+2.48*rx,hy,.70*rx,ry))+.26*inside
        sf.append(.52*center+.48*(center-adj))
    rgb=conv(pre+'garment',raw[n],cf,p=3)
    # Both flanks must differ from torso. Sum of absolute contrasts = L1
    # opponent-colour distance, softly clipped to give low-contrast edges a say.
    # Share left/right measurements with the two possible lean directions.
    c=shapes[rgb][0]
    ap=rel(pre+'contrast_pos',rgb)
    distw=np.zeros((count*2,c))
    for i in range(2*count):distw[i,3*i:3*i+3]=[1,.65,.65]
    # |x|=2 relu(x)-x. Reduce immediately so that signed copies aren't needed.
    dist=add(pre+'distance',conv(pre+'dist_positive',ap,2*distw),conv(pre+'dist_signed',rgb,-distw))
    # min(a,b)=(a+b)/2-(relu(a-b)+relu(b-a))/2
    diff=np.zeros((count,c//3)); mean=np.copy(diff)
    for i in range(count):
        diff[i,2*i:2*i+2]=[1,-1]; mean[i,2*i:2*i+2]=[1,0]
    difp=rel(pre+'flank_difference',conv(pre+'left_minus_right',dist,diff))
    low=add(pre+'both_flanks',conv(pre+'left',dist,mean),conv(pre+'minus_diff',difp,-np.eye(count)))
    ez=rel(pre+'bright_limit',conv(pre+'over',low,np.eye(count),-.32))
    garment=add(pre+'bounded_contrast',low,conv(pre+'subtract_over',ez,-np.eye(count)))
    sk=conv(pre+'exposed_head',skinp[n],np.array(sf)[:,None],p=3)
    # Complementary score: a clothed silhouette with or without a skin island.
    combined=add(pre+'votes',ec,conv(pre+'contrast_vote',garment,.79*np.eye(count)),conv(pre+'skin_vote',sk,.26*np.eye(count)))
    best=rel(pre+'positive',maxchan(pre+'best',combined))
    # No clipping of human bodies to a single spatial bin: take the best match
    # anywhere and a second summary for multiple detections.
    siz=shapes[best][1]; glob=pool(pre+'anywhere',best,siz,siz,'max')
    q=pool(pre+'quadrants',best,siz//2,siz//2,'max')
    return glob, pool(pre+'multiple',q,2,2)

edges={n:edgefield(lum_p[n],n) for n in (24,16,12)}
bodies=[people(n,edges[n]) for n in (24,16,12)]

# Readout: combine multiple scales without requiring several separately
# strong detections. Spatial bins give a small prior against border artefacts.
scalar_terms=[]
def term(src,w):
    assert int(np.prod(shapes[src]))==1
    if len(shapes[src])!=3: src=reshape(src+'_map',src,1,1,1)
    scalar_terms.append((src,w))

def mx(name,ls):
    # Encode scalars/maps as channels for a max; add disjoint channels.
    c=len(ls)
    stack=add(name+'_bank',*[conv(name+'_ch'+str(i),v,np.eye(c)[:,i:i+1]) for i,v in enumerate(ls)])
    return maxchan(name,stack)

def summarise(name,ls,weights2):
    m=mx(name+'_any_scale',ls)
    # Same explicit position priors at all scales. This is a multiplication
    # by a fixed positive diagonal, not an image-dependent attention map.
    a=taper(8,.085,.13); a[0,:]*=.91; a[-1,:]*=.87; a[:,0]*=.94; a[:,-1]*=.94
    a=dense(name+'_prior',m,np.diag(a.reshape(-1)))
    a=reshape(name+'_located',a,1,8,8)
    peak=pool(name+'_strongest',a,8,8,'max')
    co=pool(name+'_regional',a,4,4,'max'); co=pool(name+'_coverage',co,2,2)
    term(peak,weights2[0]);term(co,weights2[1]);return a
sh=summarise('head_summary',heads,(1.22,.33))
fa=summarise('face_summary',faces,(.83,.21))
# Same-location agreement instead of requiring a skin coloured face in every
# photo. When head and internal features agree it is stronger evidence.
sa=add('local_agreement',sh,fa)
term(pool('best_agreement',sa,8,8,'max'),.53)
term(mx('body_max',[t[0] for t in bodies]),1.12)
term(mx('body_coverage',[t[1] for t in bodies]),.38)

# Scene cues. Long rectilinear edges and many warm pixels all over a photo
# are ambiguous/support other explanations (rooms, furniture, food, fur).
# Colorful compact garments are especially common in street/sports photos.
# These global cues get much smaller votes than localized shape matches.
pre='context_'; sr=raw[24]
# Piecewise-linear chromatic saliency. Reject green foliage and cyan sky.
cl=np.array([r-g,r-b,b-r,b-g, .52*(b+r)-g, r-g, b-g, g-b, g-r])
chrom=rel('colour_threshold',conv('colour_halfplanes',sr,cl,[-.16,-.16,-.12,-.12,-.18,-.33,-.29,-.1,-.04]))
# min red-green/red-blue, and min blue-red/blue-green
cd=conv('colour_min_parts',chrom,[[1,-1,0,0,0,0,0,0,0],[0,0,1,-1,0,0,0,0,0]])
cp=rel('colour_min_slacks',cd)
sat=add('vivid_cloth',conv('colour_warm_cool',chrom,[[1,0,1,0,.5,.17,.17,0,0]]),conv('colour_min_subtract',cp,[[-1,-1]]))
# Contrast with nearby environment is more useful than a big uniform sky.
sat6=pool('cloth_regions',sat,4,4)
csur=np.full((1,1,3,3),-1/8.);csur[0,0,1,1]=1
contrast=rel('cloth_islands',conv('cloth_surround',sat6,csur))
term(pool('cloth_peak',contrast,4,4,'max'),.59)
term(pool('cloth_amount',sat,24,24),.98)
# Very large fractions of skin colour without its characteristic geometry
# often come from warm interiors, food or animal coats.
sm=pool('skin_amount',skin,48,48)
term(rel('widespread_skin',conv('skin_extent',sm,[[1]],-.17)),-.72)
# Axis alignment in both directions; average sustained, oriented edges over
# modest line segments, rectifying away weak or isolated texture responses.
e=edges[24]
edv=conv('vertical_edge',e,[[1,0,0,0]])
edh=conv('horizontal_edge',e,[[0,0,1,0]])
for ix,(field,hw) in enumerate([(edv,(5,1)),(edh,(1,5))]):
    line=pool('lines'+str(ix),field,hw,(1,1))
    strong=rel('sustained'+str(ix),conv('line_threshold'+str(ix),line,[[1]],-.37))
    av=pool('line_density'+str(ix),strong,shapes[strong][1:],shapes[strong][1:])
    term(av,-1.18)
# Fine-scale edge orientation: balance diagonal structure against straight
# horizontal/vertical room and vehicle edges.
emean=pool('orientation_histogram',e,22,22)
term(conv('oblique_detail',emean,[[-.44,.44,-.44,.44]]),1)

# A separate head-anchored detector. Keep colour comparisons signed and
# local until after testing garment homogeneity and both background flanks.
# The key configuration is warm face above contrasting cloth, with upper
# shoulders going outwards on both sides. Also allow a hat above the face.
def dressed(n):
    pre='dressed'+str(n)+'_'
    yy,xx=np.mgrid[-6:7,-6:7]
    def bell(x,y,sx,sy):
        t=np.exp(-.5*(((xx-x)/sx)**2+((yy-y)/sy)**2));return t/t.sum()
    geo=[(1.,.99),(.84,1.43),(1.43,1.30)]
    cnt=len(geo); colour=[]; warmth=[]; shading=[]
    for w,h in geo:
        dx,dy=0,-2.08*h
        face=bell(dx,dy,.62*w,.73*h)
        left=bell(-1.93*w,dy,.55*w,1.00*h);right=left[:,::-1]
        top=bell(0,dy-1.73*h,.97*w,.50*h)
        under=bell(0,dy+2.12*h,1.3*w,.64*h)
        coat=bell(0,dy+4.19*h,1.05*w,1.04*h)
        cl=bell(-3.37*w,dy+3.70*h,.79*w,1.16*h);cr=cl[:,::-1]
        for mask in (face-coat,coat-cl,coat-cr,
                     bell(-.82*w,dy+4.19*h,.59*w,.82*h)-bell(.82*w,dy+4.19*h,.59*w,.82*h),
                     face-left,face-right):
            for plane in planes_rgb:colour.append(plane[:,None,None]*mask)
        warmth.extend([face, face-.28*(left+right)-.22*top-.22*coat,face-left,face-right,face-coat])
        shading.extend([face-bell(0,dy-1.04*h,.85*w,.43*h),face-.5*(left+right)])
    # Valid search windows only: otherwise the outer black padding looks like
    # perfect chromatic contrast at a shoulder! The other detectors handle
    # truncated bodies and large faces near the edge.
    # Leave two positions at each end for the halo when the image is small.
    pad=0 if n>=16 else 2
    step=2 if n==48 else 1
    z=conv(pre+'part_differences',raw[n],colour,p=pad,s=step)
    ac=rel(pre+'above_zero',z)
    dw=np.zeros((6*cnt,18*cnt))
    for i in range(6*cnt):dw[i,3*i:3*i+3]=[1,.60,.60]
    dis=add(pre+'opponent_distances',conv(pre+'absolutes',ac,2*dw),conv(pre+'negative',z,-dw))
    # Soft limits prevent one high-contrast edge from claiming to be all parts.
    saturation=np.tile([.29,.29,.29,.20,.24,.24],cnt)
    excess=rel(pre+'distance_excess',conv(pre+'boundaries',dis,np.eye(6*cnt),-saturation))
    bounded=add(pre+'bounded_distances',dis,conv(pre+'remove_excess',excess,-np.eye(6*cnt)))
    sam=conv(pre+'warmth',skinp[n],np.array(warmth)[:,None],p=pad,s=step)
    shades=squash(pre+'cap',conv(pre+'shading',lum_p[n],np.array(shading)[:,None],p=pad,s=step),6.5)
    wd,ws,wh=[np.zeros((cnt,m*cnt)) for m in (6,5,2)]
    for i in range(cnt):
        wd[i,6*i:6*i+6]=[.53,.57,.57,-.58,.16,.16]
        ws[i,5*i:5*i+5]=[.18,.25,0,0,.09]
        wh[i,2*i:2*i+2]=[.095,.035]
    wc=np.zeros_like(ws);bad=np.tile([.68,.39,.29,.29,.38],cnt)
    for i in range(cnt):wc[i,5*i:5*i+5]=[-.46,-.21,-.14,-.14,-.18]
    fail=rel(pre+'missing_parts',conv(pre+'min_warmth',sam,-np.eye(5*cnt),bad))
    # Left and right shoulders: compute response to an outward step of the
    # garment silhouette immediately below head level, independently of hue.
    wt=[]
    for w,h in geo:
        for sign in (-1,1):
            along=bell(sign*1.4*w,-.32*h,.58*w,.55*h)
            # At the left shoulder outward normal is upper-left; on the
            # right it is upper-right. Map alignment to the four gradient bins.
            ca=np.array([.08,.20,.47,.25]) if sign<0 else np.array([.08,.25,.47,.20])
            ca[1 if sign<0 else 3]+=.23;ca[3 if sign<0 else 1]-=.18
            wt.append(ca[:,None,None]*along-.21*along[None,:,:]/4)
    # The edge field excludes one border cell. Convolving a cropped spatial
    # filter gives the same center alignment as the RGB comparisons.
    ew=np.array(wt)[:,:,1:-1,1:-1]
    ed=edges[n] if n in edges else edgefield(lum_p[n],n)
    shoulders=conv(pre+'shoulders',ed,ew,p=pad,s=step)
    gv=np.zeros((cnt,2*cnt))
    for i in range(cnt):gv[i,2*i:2*i+2]=[.13,.13]
    score=add(pre+'scores',conv(pre+'wardrobe',bounded,wd),conv(pre+'head',sam,ws),
              conv(pre+'missing_vote',fail,wc),conv(pre+'shadow',shades,wh),conv(pre+'outline_vote',shoulders,gv))
    peak=maxchan(pre+'shape',score);sz=shapes[peak][1]
    return pool(pre+'best',peak,sz,sz,'max')

joint=mx('head_on_garment',[dressed(n) for n in (48,24,16,12)])
term(joint,1.32)

# A sports field gives useful context for a small or occluded player, but
# vegetation by itself is not evidence. Check that the grass occupies a
# coherent ground plane (as opposed to a canopy above the horizon), and allow
# for either yellowish or bluish green. Snow/water cues likewise include a
# contrasting central foreground object and count for less than local parts.
lin=conv('green_ground_planes',raw[16],[[ -1,1,0],[0,1,-1],[1,0,-1]],[-.012,-.038,.04])
neg=rel('lawn_violations',conv('lawn_minimum',lin,-np.diag([14,14,7]),[.30,.43,0]))
gr=rel('lawn',conv('lawn_intersection',neg,[[-1,-1,-1]],1))
y,x=np.mgrid[0:16,0:16]; y=(y+.5)/16;x=(x+.5)/16
lo=np.exp(-.5*((y-.73)/.19)**2);top=np.exp(-.5*((y-.17)/.14)**2)
field=dense('ground_plane',gr,(lo/lo.sum()-.68*top/top.sum()).ravel())
term(rel('field_context',field),.31)
# Bright snow with blue shadows. Grey sky is not sufficient; the cue must
# come from the lower part of the photo.
whc=conv('snow_planes',raw[16],[rgb_lum,b-r,g-r],[-.65,.037,.046])
whv=rel('snow_violations',conv('snow_boundaries',whc,-np.diag([7,13,11]),[.8,.45,.35]))
snow=rel('pale_ground',conv('snow_intersection',whv,[[-1,-1,-1]],1))
term(rel('snow_context',dense('snow_plane',snow,(lo/lo.sum()-.42*top/top.sum()).ravel())),.28)

# Dark or back-facing people needn't have any visible warm face pixels. A
# narrow, upright, articulated silhouette against a coherent background is a
# stronger alternative than simply declaring that a well-contrasted patch is
# a body. Independently test the signs at the head, shoulders, hips and legs,
# and test the negative space just outside the neck (above the shoulders).
def backlit(n):
    pre='backlit'+str(n)+'_'; size=13; mid=6
    yy,xx=np.mgrid[-mid:mid+1,-mid:mid+1]
    def at(x,y,w,h):
        q=np.exp(-.5*(((xx-x)/w)**2+((yy-y)/h)**2));return q/q.sum()
    samples=[]; spec=[]
    for wide,step_out,lean in [(1.,.64,0),(1.35,.64,0),(.90,1.47,.12),(.90,1.47,-.12)]:
        comb=[[] for _ in range(6)]
        for s in (-1,1):
            # Head is narrower than shoulders. Above shoulders, two windows
            # to either side should return to background colour.
            comb[0].append(at(-3.87*lean,-3.87,.53*wide,.65)-at(s*1.69*wide-3.87*lean,-3.87,.47*wide,.79))
            comb[1].append(at(-3.95*lean,-3.95,.60*wide,.64)-at(-5.70*lean,-5.64,.85*wide,.44))
            comb[2].append(at(s*.92*wide-lean,-1.0,.53*wide,.80)-at(s*2.95*wide-lean,-1.0,.65*wide,.92))
            comb[3].append(at(.1*lean,.1,.85*wide,1.15)-at(s*2.61*wide,.14,.60*wide,1.07))
            comb[4].append(at(s*step_out*wide+3.12*lean,3.12,.54*wide,1.13)-at(s*(step_out+1.75)*wide+3.12*lean,3.12,.65*wide,1.1))
            comb[5].append(at(s*1.11*wide-lean,-1.2,.45*wide,.66)-at(s*1.22*wide-3.65*lean,-3.65,.41*wide,.58))
        for i in range(6):
            # sum the two halves only within each part (not across poses).
            samples.append((comb[i][0]+comb[i][1])*.5)
        spec.append((wide,step_out,lean))
    # Both bright-on-dark and dark-on-bright. Smoothly cap very high contrast
    # so a single bright upper edge won't overwhelm several incorrect parts.
    l=conv(pre+'comparisons',lum_p[n],np.array(samples)[:,None],s=2 if n==48 else 1)
    c=len(samples); sgns=[1,-1];cl=[];wb=[]
    for sgn in sgns:
        for i in range(len(spec)):
            for j in range(6):
                row=np.zeros(c);row[6*i+j]=sgn;cl.append(row/.22)
    cl=np.array(cl);planes=len(cl)
    # clamp each signed contrast to [-0.45,1], then penalize absent parts.
    lv=conv(pre+'contrasts',l,cl)
    ramp=rel(pre+'ramps',conv(pre+'levels',lv,np.vstack([np.eye(planes),np.eye(planes),-np.eye(planes)]),[.45]*planes+[-1.0]*planes+[.30]*planes))
    wm=np.zeros((planes//6,3*planes))
    for i in range(planes//6):
        sl=slice(6*i,6*i+6)
        w=np.array([.25,.08,.23,.15,.14,.15]);wm[i,sl]=w
        wm[i,planes+6*i:planes+6*i+6]=-w
        wm[i,2*planes+6*i:2*planes+6*i+6]=[-.21,-.05,-.16,-.11,-.05,-.12]
    sc=conv(pre+'figure_rule',ramp,wm,-.45)
    sc=maxchan(pre+'pose',sc);dim=shapes[sc][1]
    pos=taper(dim,.22,.33)
    sc=reshape(pre+'spatial',dense(pre+'prior',sc,np.diag(pos.ravel())),1,dim,dim)
    return pool(pre+'anywhere',sc,dim,dim,'max')

sil=mx('backlit_body',[backlit(n) for n in (48,24,16)])
term(sil,.40)

# Strongly saturated dyestuffs are more suggestive of clothing than ordinary
# warm food / wood or blue-green skies. Require a blob, not one vivid pixel.
Y=raw[24]; pc=np.array([r-g,r-b,b-g,b-r,r-g,b-g]);off=np.array([.12,.14,.12,.17,.13,.11])
viol=rel('dye_halfplanes',conv('dye_margins',Y,-pc/.16,1+off/.16))
wd=np.zeros((3,6))
for i in range(3):wd[i,2*i:2*i+2]=[-1,-1]
dye=rel('dye_membership',conv('dye_intersection',viol,wd,1))
region=pool('dye_regions',dye,4,2)
# Center minus halo. Spatial taper downweights the sky along the picture edge.
dwm=np.zeros((3,3,3,3))
for i in range(3):dwm[i,i,:,:]=-1/8;dwm[i,i,1,1]=1.17
blob=conv('dye_isolation',region,dwm);blob=maxchan('dye_hue',blob)
dim=shapes[blob][1];prior=taper(dim,.19,.12)
blob=reshape('dye_spatial',dense('dye_location',blob,np.diag(prior.ravel())),1,dim,dim)
term(pool('dyed_garment',blob,dim,dim,'max'),.74)
area=pool('dyed_area',dye,24,24);term(conv('dye_fraction',area,[[.88,1.18,1.28]]),.83)
# Food on a plate is a common case of a warm oval against a cold surround.
# Look for a fairly continuous pale rim around a textured, warm interior.
L=raw[16];a=conv('porcelain_planes',L,[rgb_lum,r-g,g-b,b-r],[-.62,0,0,0]);
wp=np.array([[-7,0,0,0],[0,11,0,0],[0,-11,0,0],[0,0,11,0],[0,0,-11,0],[0,0,0,11],[0,0,0,-11]])
v=rel('porcelain_impurities',conv('porcelain_limits',a,wp,[.75,-.57,-.57,-.57,-.57,-.71,-.71]))
porcelain=rel('light_ceramic',conv('porcelain',v,-np.ones((1,7)),1))
yy,xx=np.mgrid[-5:6,-5:6]; circles=[]; centers=[]
for rx,ry in [(4,3.3),(3.2,4),(4.65,4.45)]:
    rad=np.sqrt((xx/rx)**2+(yy/ry)**2); ang=np.arctan2(yy/ry,xx/rx)
    inside=np.exp(-.5*(rad/.44)**2);inside/=inside.sum();centers.append(inside)
    for theta in (0,np.pi/2,np.pi,3*np.pi/2):
        qr=np.exp(-.5*((rad-.96)/.135)**2+2.6*(np.cos(ang-theta)-1));qr/=qr.sum();circles.append(qr)
cr=conv('plate_arcs',porcelain,np.array(circles)[:,None]); ce=conv('meal',skinp[16],np.array(centers)[:,None]);
ag=np.zeros((3,12));
for i in range(3):ag[i,4*i:4*i+4]=.25
lack=rel('broken_arcs',conv('arc_requirements',cr,-np.eye(12),.42))
pl=rel('plate',add('rim_and_meal',conv('full_rim',cr,ag,-.62),conv('borders_missing',lack,-1.40*ag),conv('warm_center',ce,.53*np.eye(3))))
pl=maxchan('plate_shape',pl);term(pool('dinner_plate',pl,6,6,'max'),-.64)
# Very decorrelated chromaticity (not merely low saturation) is especially
# common when photographers deliberately choose monochrome for portraits.
abs_=np.array([r-g,g-r,b-g,g-b]);cv=pool('chroma_area',rel('absolute_chroma',conv('chroma_sign',raw[16],abs_)),16,16)
term(rel('monochrome',conv('neutral_frame',cv,[[-76,-76,-76,-76]],1)),.24)

# Expected support under the no-person hypothesis, expressed in the units
# of the detectors. All coefficients and offsets are hand specified.
offset=(1.22*.46+.33*.34 + .83*.59+.21*.48 + .53*.86 + 1.12*.46 + .38*.365 + .035 + 1.32*.79 + .075 + .40*.72 + .74*.26 + .025)
# Baseline appearance is ambiguous: agreement of several branches is
# generally necessary to exceed this threshold.
v=add('combined_score',*[conv('vote_'+str(i),src,[[w]]) for i,(src,w) in enumerate(scalar_terms)])
dense('class_scores',v,[[0.],[1.]],[0.,-offset])

path=Path(sys.argv[1]);path.mkdir(parents=True,exist_ok=True)
(path/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
np.savez(path/'weights.npz',**weights)
