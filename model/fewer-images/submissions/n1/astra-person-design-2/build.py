"""A hand-specified, multiresolution part detector. No data inputs."""
import json
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
layers = []; weights = {}


def op(name, what, inputs, **kw):
    if isinstance(inputs, str):
        inputs = [inputs]
    layers.append(dict(name=name, op=what, inputs=inputs, **kw))
    return name


def conv(n, x, w, b=0., stride=1, pad=None):
    w = np.asarray(w, np.float32)
    if w.ndim == 2:
        w = w[:, :, None, None]
    o, i, k, _ = w.shape
    weights[n+'.W'] = w
    weights[n+'.b'] = np.broadcast_to(b, (o,)).astype(np.float32).copy()
    return op(n, 'conv', x, out=o, kernel=[k,k], stride=stride,
              padding=k//2 if pad is None else pad)


def dense(n, x, w, b=0.):
    w = np.atleast_2d(w).astype(np.float32)
    weights[n+'.W'] = w
    weights[n+'.b'] = np.broadcast_to(b,(len(w),)).astype(np.float32).copy()
    return op(n,'dense',x,out=len(w))


def pool(n, x, k, s=None, how='avg'):
    if isinstance(k,int): k=(k,k)
    if s is None: s=k
    if isinstance(s,int): s=(s,s)
    return op(n,how+'pool',x,size=list(k),stride=list(s))


def relu(n,x): return op(n,'relu',x)
def add(n,*xs): return op(n,'add',list(xs))
def reshape(n,x,*shape): return op(n,'reshape',x,shape=list(shape))
def point(n,x,w,b=0): return conv(n,x,w,b)

def chmax(n,x,c,m):
    a=reshape(n+'_as_strip',x,1,c,m*m)
    a=pool(n+'_max',a,(c,1),how='max')
    return reshape(n,a,1,m,m)


def pointclip(n, x, w, b=0., hi=1.):
    # clamp(Wx+b, 0, hi), with one pair of rectifiers per output
    w=np.atleast_2d(w); c=len(w)
    b=np.broadcast_to(b,(c,))
    a=point(n+'_hinges',x,np.concatenate([w,w]),np.r_[b,b-hi])
    return point(n,relu(n+'_positive',a),np.c_[np.eye(c),-np.eye(c)])


def absolute(n,x,c):
    a=point(n+'_both',x,np.r_[np.eye(c),-np.eye(c)])
    return point(n,relu(n+'_above0',a),np.c_[np.eye(c),np.eye(c)])


# Measurements are made on a dyadic pyramid. First level integrates 2x2 pixels
# before measuring chromaticity so as not to treat JPEG color fringes as parts.
rgb={48:pool('half_rgb','image',2)}
for m in (24,12,6): rgb[m]=pool('rgb'+str(m),rgb[2*m],2)
meanrgb=pool('mean_rgb',rgb[6],6)
# A partial gray-world correction, bounded to avoid undoing real scene colors.
adj=pointclip('light_cast',meanrgb,[[-.27,.27,0],[0,.27,-.27]],
              [.04+.27*.025,.04-.27*.015],.08)
w=np.zeros((3,48,48,2)); w[0,:,:,0]=1; w[2,:,:,1]=1
cast=reshape('per_image_offset',dense('extend_cast',adj,w.reshape(-1,2),
        np.repeat([-.04,0.,-.04],48*48)),3,48,48)
white=add('partly_balanced',rgb[48],cast)
R=np.array([1.,0.,0.]); G=np.array([0.,1.,0.]); B=np.array([0.,0.,1.])
Y=.30*R+.56*G+.14*B
u=R-G; v=G-B
# A soft convex color region. Each hinge measures a violation of a different
# face-pigment constraint. These are inequalities on RGB, not a skin palette.
constraints=[
    (.145*Y-u, .010, .050),         # insufficient red over green
    (.18*Y-(R-B), .005, .068),      # insufficient red over blue
    (-v-.28*u, -.009, .045),       # too magenta
    (v-1.07*u, -.014, .067),       # yellows, straw and foliage
    (u-.64*Y, -.022, .085),        # pure reds and very saturated oranges
    ((R-B)-1.05*Y, -.030, .12),
    (-Y,.10,.095),                 # near black: little reliable chroma
]
cc=point('pigment_halfspaces',white,[t[0]/t[2] for t in constraints],
         [t[1]/t[2] for t in constraints])
rp=relu('pigment_violations',cc)
s=relu('skin_affinity',point('pigment_margin',rp,-np.ones((1,len(constraints))),1.))
# Three planes: skin affinity, midgray-centered luminosity, and image support.
# The latter lets templates discount samples whose surround is off the image.
lum=point('luminance_and_domain',rgb[48],np.array([[0,0,0],Y,[0,0,0]]),[0,-.46,1])
surface={48:add('surface48',lum,point('skin_plane',s,[[1],[0],[0]]))}
for m in (24,12,6): surface[m]=pool('surface'+str(m),surface[m*2],2)


def grid(r):
    y,x=np.mgrid[-r:r+1,-r:r+1]; return x.astype(float),y.astype(float)

def unit(a): return a/(a.sum()+1.e-30)

def bell(x,y,cx,cy,sx,sy):
    return unit(np.exp(-.5*(((x-cx)/sx)**2+((y-cy)/sy)**2)))


def face_bank(m, heights, r, step):
    # The positive and negative lobes are sampled integrals of elementary
    # elliptical regions in face coordinates. Subpixel integration matters at
    # the smallest head sizes. A separate set of lobes describes each aspect.
    x,y=grid(r); kernels=[]; offsets=[]; types=[]
    for h in heights:
        for aspect in (.59,.90):
            wid=aspect*h
            def area(a,b,sx,sy):
                return bell(x,y,a*wid,b*h,np.hypot(sx*wid,.30),np.hypot(sy*h,.30))
            def sides(a,b,sx,sy): return .5*(area(-a,b,sx,sy)+area(a,b,sx,sy))
            core=.30*area(0,-.24,.22,.13)+.50*sides(.17,.11,.18,.13)+.20*area(0,.31,.15,.105)
            outside=(.49*sides(.83,0,.20,.27)+.20*area(0,-.70,.32,.15)+
                     .19*sides(.61,.55,.23,.18)+.12*area(0,.86,.35,.17))
            hair=.53*area(0,-.55,.30,.14)+.47*sides(.48,-.26,.12,.22)
            face_light=.68*sides(.19,.12,.15,.12)+.32*area(0,-.31,.19,.105)
            def eyelobe(a):
                return (.69*area(a,.09,.14,.094)+.31*area(a*.85,-.32,.14,.072)
                        -area(a,-.135,.139,.066))
            # A bright bridge and mid-forehead, with darker temples and a
            # mouth under the nose. Independent of a horizontal eye stripe.
            relief=(.50*area(0,-.04,.105,.17)+.24*area(0,.14,.14,.085)
                    +.26*area(0,-.30,.19,.09)-.55*sides(.393,.015,.12,.21)
                    -.25*area(0,.251,.19,.064)-.20*sides(.22,-.135,.11,.08))
            z=np.zeros((3,2*r+1,2*r+1)); idx=len(kernels)
            def put(c, mask, bias=0, support=0):
                t=z.copy(); t[c]=mask
                if support: t[2]+=support*outside
                kernels.append(t); offsets.append(bias-support)
            put(0,core,support=.46); put(0,outside)
            # Saturating contrast measurements (reused both above and below
            # the upper hinge; reconstructed after the ReLU).
            for lobes,lo,width in [(face_light-hair,.008,.185),
                                   (eyelobe(-.225),.004,.103),
                                   (eyelobe(.225),.004,.103),
                                   (relief,.005,.096)]:
                for bound in (0,1): put(1,lobes/width,-lo/width-bound,support=.40)
            types.append((idx,h))
    pref='face'+str(m); size=m//step
    a=relu(pref+'_rect',conv(pref+'_lobes',surface[m],kernels,offsets,stride=step))
    rows=[]; bs=[]
    for i,h in types:
        pixels=h*(96/m)
        # Small chromatic speckles must pass a higher bar. Conversely the
        # eye measurements are only trusted once separated at this resolution.
        small=np.clip((13.-pixels)/9.,0.,1.)
        for coeff,bias in [([1.12,-.78,.265,.127,.127,.135],-1.102-.123*small),
                           ([.25,-.17,.177,.367,.367,.402],-1.127-.22*small-.055*(h<4.0))]:
            w=np.zeros(len(kernels)); w[i:i+2]=coeff[:2]
            for j,val in enumerate(coeff[2:]): w[i+2+j*2:i+4+j*2]=[val,-val]
            rows.append(w); bs.append(bias)
    a=point(pref+'_part_rules',a,rows,bs)
    return chmax(pref+'_best',a,len(rows),size)


# 5 to 62 pixels in head height; the largest filters can describe a close-up
# without assuming that a whole person is in the frame.
faces={48:face_bank(48,[2.7,4.7],5,2),
       24:face_bank(24,[3.5,5.25],6,1),
       12:face_bank(12,[3.75,5.65,7.8],7,1)}


def edge_map(m):
    # Directional, clipped absolute derivatives. Integration over a cell
    # tolerates a small displacement of a sleeve or the side of a head.
    a=np.array([1,2,1])/4; d=np.array([-1,0,1]); dx=np.outer(a,d); dy=dx.T
    ks=[]; bs=[]
    for ang in np.arange(4)*np.pi/4:
        g=(np.cos(ang)*dx+np.sin(ang)*dy)[None,:,:]*Y[:,None,None]
        for sg,th in [(1,.015),(-1,.015),(1,.215),(-1,.215)]:
            ks.append(sg*g); bs.append(-th)
    n='edge'+str(m)
    # Centering the gray plane reduces the size of artificial border steps.
    # The domain measurement in the part detector discounts border responses.
    centered=point(n+'_gray',rgb[m*2],[Y],[-.46])
    ks=np.asarray(ks).sum(axis=1,keepdims=True)
    z=relu(n+'_rect',conv(n+'_derivs',centered,ks,bs))
    w=np.kron(np.eye(4),np.array([[5,5,-5,-5]]))
    return pool(n+'_cells',point(n+'_capped',z,w),2)


def torso_bank(m, widthlist, radius):
    pre='figure'+str(m); edge=edge_map(m); x,y=grid(radius)
    angles=np.arange(4)*np.pi/4
    outlines=[]; hands=[]; contrasts=[]; ids=[]
    centered_rgb=point(pre+'_midgray_padding',rgb[m],np.eye(3),-.46)
    for width in widthlist:
        # The tall and short forms account for squashing the full photo to
        # a square, and for the displacement of shoulders in a seated pose.
        for elong in (.82,1.30):
            height=width*elong
            def loc(a,b,ax,ay):
                return bell(x,y,a*width,b*height,np.hypot(.35,ax*width),np.hypot(.35,ay*height))
            # Upright contour in head-relative coordinates. Integrals of
            # tangents to line segments give the expected unsigned directions.
            templ=np.zeros((4,*x.shape)); support=np.zeros_like(x)
            def line(x1,y1,x2,y2,portion):
                ax,ay,bx,by=x1*width,y1*height,x2*width,y2*height
                vx,vy=bx-ax,by-ay; v2=vx*vx+vy*vy
                t=np.clip(((x-ax)*vx+(y-ay)*vy)/v2,0,1)
                ds=(x-ax-t*vx)**2+(y-ay-t*vy)**2
                sigma=np.hypot(.44,.080*width)
                mask=unit(np.exp(-.5*ds/(sigma*sigma)))
                phi=np.arctan2(vx,-vy)
                # Interpolate normals between adjacent orientation channels.
                diff=np.abs(((angles-phi+np.pi/2)%np.pi)-np.pi/2)
                weights=np.maximum(1-diff/(np.pi/4),0.)
                directional=weights-.72*np.roll(weights,2)
                templ[:]+=portion*directional[:,None,None]*mask
                support[:]+=portion*mask
            for flip in [-1,1]:
                def seg(x1,y1,x2,y2,f):line(x1*flip,y1,x2*flip,y2,.5*f)
                seg(.02,-.33,.18,-.23,.09)
                seg(.19,-.20,.195,.12,.14)
                seg(.19,.11,.12,.28,.055)
                seg(.135,.28,.46,.47,.23)
                seg(.46,.47,.535,.84,.195)
                seg(.535,.84,.50,1.30,.165)
                seg(.34,1.10,.30,1.63,.125)
            # Subtract the same directional histogram spread over adjacent
            # cells; a wall of vertical stripes is not a human silhouette.
            bg=.54*loc(0,.46,.64,.59)+.46*loc(0,1.08,.65,.59)
            templ-=templ.sum(axis=(1,2),keepdims=True)*bg
            outlines.append(templ)
            # A weak contextual cue: exposed extremities beside a clothed
            # torso, rather than a torso and its surround all of one pigment.
            hand=(.50*loc(-.58,1.33,.21,.31)+.50*loc(.58,1.33,.21,.31)
                  -.52*loc(0,.83,.23,.25)-.24*loc(-.97,1.33,.19,.30)-.24*loc(.97,1.33,.19,.30))
            hands.append(np.stack([.19*hand,np.zeros_like(hand),.20*bg]))
            # Compare the garment to its sides in lightness and two opponent
            # colors. Check vertical consistency within the garment as well.
            chest=loc(0,.82,.24,.27)
            for mask in [chest-loc(-.83,.86,.16,.32),chest-loc(.83,.86,.16,.32),
                         loc(0,.62,.20,.14)-loc(0,1.13,.23,.17)]:
                for axis in [Y,u,v]: contrasts.append(axis[:,None,None]*mask)
            ids.append((width,elong))
    c=len(ids)
    line_q=conv(pre+'_outlines',edge,outlines)
    skin_q=conv(pre+'_extremities_and_support',surface[m],hands,-.20)
    cc=conv(pre+'_shirt_differences',centered_rgb,contrasts)
    ab=absolute(pre+'_distances',cc,len(contrasts))
    w=np.kron(np.eye(3*c),[[4.,4.0,3.0]])
    dist=pointclip(pre+'_bounded_distances',ab,w)
    # Per-side figure/ground differences. Use the weaker side as well as
    # the mean. A single edge (e.g. a door jamb) isn't enough by itself.
    delta=np.zeros((c,3*c))
    for i in range(c): delta[i,3*i:3*i+2]=[1,-1]
    lr=absolute(pre+'_side_asymmetry',point(pre+'_lr',dist,delta),c)
    w=np.kron(np.eye(c),[[.5,.5,-.45]])
    col_q=add(pre+'_shirt_shape',point(pre+'_bounded_contrast',dist,w),point(pre+'_bilateral',lr,-.34*np.eye(c)))
    # Each contribution is in units of bounded local contrast. Keep the
    # signed score so an incoherent torso can count against a marginal face.
    return chmax(pre+'_best',add(pre+'_evidence',
                point(pre+'_geometry_vote',line_q,1.28*np.eye(c),-.155),
                point(pre+'_clothing_vote',col_q,.52*np.eye(c),-.16),skin_q),c,m)


bodies={24:torso_bank(24,[2.15,3.55,4.9],7),
        12:torso_bank(12,[2.95,4.6],6),
        6:torso_bank(6,[2.9,4.5],5)}


def to6(n,x,m): return x if m==6 else pool(n,x,m//6,how='max')

regions=[]
for i,(facelevel,bodylevel) in enumerate([(48,24),(24,12),(12,6)]):
    f=faces[facelevel]; m=min(24,facelevel); n='joint'+str(i)
    if m>bodylevel: paired=pool(n+'_locate_head',f,m//bodylevel,how='max')
    else: paired=f
    q=bodies[bodylevel]
    # Associate each head with a body at that scale, in that position.
    combined=add(n+'_associated',paired,point(n+'_body_bonus',q,[[.57]],.012))
    regions.append(to6(n+'_local',combined,bodylevel))
    regions.append(to6(n+'_head_alone',point(n+'_unpaired',f,[[1.]],-.078),m))
    # Color/outline evidence can justify a detection without visible skin, but
    # then the torso must be much more distinctive than for a paired head.
    # Discount the stand-alone score: a narrow piece of strongly contrasting
    # architecture can resemble a torso, especially at the finest scale.
    # For the smallest bodies the head spans very few cells, so the outline
    # evidence should not be regarded as conclusive without corroboration.
    torso_only=point(n+'_clothed',q,[[.91]],[-.405,-.369,-.362][i])
    regions.append(to6(n+'_body_alone',torso_only,bodylevel))

# Mild scene prior for outdoor action pictures. A green playing surface or
# pale snow is useful context but cannot supply enough evidence on its own.
q=rgb[12]
fieldplanes=point('playing_surface_tests',q,[10*(G-R),10*(G-B),Y*5,(R-G)*8,(G-B)*8,
                                           (G-R)*8,(B-G)*8],[-.02,-.13,-3.45,0.,0.,0.,0.])
a=relu('playing_surface_rect',fieldplanes)
# min(green-red,green-blue) as a soft AND; require both differences, not just
# green-blue (which is high in warmly illuminated indoor photographs).
A=pointclip('grass_ramps',a,[[1,0,0,0,0,0,0],[0,1,0,0,0,0,0]])
D=relu('grass_disagree',point('grass_delta',A,[[1,-1]]))
green=add('grass',point('grass_left',A,[[1,0]]),point('grass_sub',D,[[-1]]))
snow=pointclip('snow',a,[[0,0,1,-.8,-.8,-.8,-.8]])
priorw=np.zeros((12,12)); priorw[5:11]=1/(6*12)
fieldprior=dense('field_context',green,.14*priorw.reshape(1,-1),-.011)
snowprior=dense('snow_context',snow,.075*priorw.reshape(1,-1),-.008)
# Suppress a sea of similar skin-colored material; retain chromatic evidence
# if it is concentrated where faces normally occur in the composition.
six=surface[6]; xx,yy=np.meshgrid((np.arange(6)+.5)/6,(np.arange(6)+.5)/6)
center=np.exp(-.5*(((xx-.5)/.27)**2+((yy-.37)/.245)**2)); center=unit(center)
w=np.zeros((1,3,6,6)); w[0,0]=.36*(center-1/36)-.16/36
skinprior=dense('skin_distribution',six,w.reshape(1,-1),.013)

# Reduce hypotheses of different sizes at a common spatial resolution.
n=len(regions)
emb=[]
for i,a in enumerate(regions):
    w=np.zeros((n,1)); w[i]=1; emb.append(point('vote_slot'+str(i),a,w))
hyp=chmax('regional_winner',add('all_regional_votes',*emb),n,6)
# Edge heads are allowed. Apply a small composition prior, especially to
# the bottom corners, where a detached face-sized patch is less persuasive.
location=.047-.063*(np.abs(xx-.5)/.5)**1.6-.135*(np.maximum(yy-.48,0)/.52)**1.3-.022*(np.maximum(.23-yy,0)/.23)
loc=reshape('localized_evidence',dense('composition',hyp,np.eye(36),location.ravel()),1,6,6)
peak=reshape('peak_evidence',pool('best_location',loc,6,how='max'),1)
# Small corroboration from distinct regions, not a requirement for crowds.
blocks=pool('nearby_winners',loc,2,how='max')
cor=relu('partial_corroboration',point('partial_margin',blocks,[[1.]],.16))
more=dense('repeated_evidence',cor,np.full((1,9),.34/9),-.017)
margin=add('photo_margin',peak,more,fieldprior,snowprior,skinprior)
dense('class_scores',margin,[[-.5],[.5]])

(out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
np.savez(out/'weights.npz',**weights)
