"""A data-free, multiscale part detector for the presence of people."""
import json
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
layers, weights, shape = [], {}, {'image': (3,96,96)}

def layer(n, op, inp, sh, **opts):
    layers.append(dict(name=n,op=op,inputs=[inp] if isinstance(inp,str) else inp,**opts)); shape[n]=tuple(sh); return n

def conv(n, inp, W, b=0, stride=1, pad=None):
    W=np.array(W,dtype='f'); W=W[:,:,None,None] if W.ndim==2 else W
    o,c,k,_=W.shape; assert c==shape[inp][0]; p=k//2 if pad is None else pad
    weights[n+'.W']=W; weights[n+'.b']=np.broadcast_to(b,(o,)).astype('f').copy()
    return layer(n,'conv',inp,(o,)+tuple((j+2*p-k)//stride+1 for j in shape[inp][1:]),out=o,kernel=[k,k],stride=stride,padding=p)

def dense(n, inp, W, b=0):
    W=np.asarray(W,dtype='f').reshape(-1,int(np.prod(shape[inp]))); o=len(W)
    weights[n+'.W']=W; weights[n+'.b']=np.broadcast_to(b,(o,)).astype('f').copy()
    return layer(n,'dense',inp,(o,),out=o)

def rel(n, inp): return layer(n,'relu',inp,shape[inp])
def reshape(n, inp, sh): return layer(n,'reshape',inp,sh,shape=list(sh))
def add(n, *ins):
    assert all(shape[t]==shape[ins[0]] for t in ins); return layer(n,'add',list(ins),shape[ins[0]])
def pool(n, inp, k, mode='avg', st=None):
    ks=(k,k) if isinstance(k,int) else tuple(k); ss=ks if st is None else (st,st)
    return layer(n,mode+'pool',inp,(shape[inp][0],)+tuple((s-v)//t+1 for s,v,t in zip(shape[inp][1:],ks,ss)),size=list(ks),stride=list(ss))

def points(k):
    y,x=np.mgrid[:k,:k]-(k-1)/2; return x,y

def gaussian(x,y,X,Y,sx,sy):
    a=np.exp(-.5*(((x-X)/sx)**2+((y-Y)/sy)**2)); return a/max(1.e-20,a.sum())

def av(a):return a/(a.sum()+1e-20)

def form_bank(fs):
    # Each row is a linear combination of named input maps.
    return np.stack([np.stack(row) for row in fs])

# Photometric front end, in display RGB. A weak gray-world correction in the
# chroma plane handles illumination, but cannot turn gray into saturated flesh.
rgb=pool('two_pixel_mean','image',2)
mean=pool('light_estimate',rgb,48)
r,g,b=np.eye(3); lum=.28*r+.54*g+.18*b; d=r-g; e=g-b
light=rel('cast_excess',dense('cast_halfplanes',mean,[d,-d,e,-e,d,-d,e,-e],[-.035,-.025,-.025,-.025,-.19,-.12,-.17,-.14]))
fix=np.array([[1,-1,0,0,-1,1,0,0],[0,0,0,0,0,0,0,0],[0,0,-1,1,0,0,1,-1]])*(-.42)
shift=reshape('illuminant',dense('broadcast_cast',light,np.repeat(fix,48*48,axis=0)),(3,48,48))
color=add('balanced_color',rgb,shift)
# Distance to a wedge in (R-G,G-B,G); use ramps rather than a hard mask.
# The low-chroma edge rises with brightness; the two hue sides reject pure
# yellow, magenta and bright red. An additional stricter ramp favours rosy skin.
wrong=[(.065*g-d,.023,19), (d-.64*g,-.055,7), (e-1.04*d,-.026,14),
       (-e-.24*d,-.027,13), (-g,.085,8), (d-.40*g,-.085,5),
       (e-.72*d,-.026,11), (.14*g-d,.038,13), (d-.32*g,-.055,5)]
viol=rel('color_distances',conv('skin_halfspaces',color,[v*t for v,c,t in wrong],[c*t for v,c,t in wrong]))
sc=rel('skin_membership',conv('soft_skin_cones',viol,[[-1,-1,-1,-1,-1,-1,0,0,0],[-1,-1,-1,-1,-1,-1,-1,-1,-1]],1))
# A two-plane feature pyramid: face colour and luminance. Separate branches
# for subtle facial shading and for colour-independent silhouettes below.
f48=add('skin_and_light',conv('skin_plane',sc,[[.48,.52],[0,0]]),conv('value_plane',rgb,[[0,0,0],lum]))
features={48:f48}; colors={48:rgb}
for s in (24,12,6):
    features[s]=pool('appearance_%d'%s,features[2*s],2); colors[s]=pool('rgb_%d'%s,colors[s*2],2)

# Spatial prior used AFTER testing each template: large heads can be above
# centre, small ones can be almost anywhere. Do not crop the photo.
zero=dense('homogeneous',mean,[[0,0,0]],1)
def location(n,h,kind):
    yy,xx=np.mgrid[:h,:h]; u=(xx+.5)/h; v=(yy+.5)/h
    side=np.maximum(0,abs(u-.5)-.30)/.2
    if kind=='face': cost=.09*side+.22*np.maximum(0,v-.59)/.41+.055*np.maximum(0,.12-v)/.12
    else: cost=.075*side+.14*np.maximum(0,v-.74)/.26+.07*np.maximum(0,.18-v)/.18
    return reshape(n,dense(n+'_values',zero,-cost.reshape(-1,1)),(1,h,h))
locs={}
def localprior(h,kind):
    if (h,kind) not in locs:locs[h,kind]=location(kind+'_position_%d'%h,h,kind)
    return locs[h,kind]

peaks={'face':[],'bust':[],'detail':[],'figure':[]}; coverage={'face':[],'figure':[]}
def peak(n,x,kind,prior='face',area=False):
    h=shape[x][1]
    x=add(n+'_located',x,localprior(h,prior)); peaks[kind].append(pool(n+'_best',x,h,'max'))
    if area:
        # Max over a few large cells, then average them: rewards more than a
        # single accidental match without requiring several separate people.
        m=pool(n+'_regions',x,h//3,'max'); coverage[kind].append(pool(n+'_support',m,3))

def face(n,s,w,k,st,angle=0):
    # Face-relative coordinates (one unit = one face width). Render soft
    # ellipses/rings, not bitmaps. A small tilt models heads in action scenes.
    x,y=points(k); a=np.deg2rad(angle); X=(x*np.cos(a)+y*np.sin(a))/w; Y=(-x*np.sin(a)+y*np.cos(a))/w
    z=x*0.; fs=[]
    def blob(dx,dy,rx,ry):return gaussian(X,Y,dx,dy,rx,ry)
    def ell(dx,dy,rx,ry):
        dist=((X-dx)/rx)**2+((Y-dy)/ry)**2
        return av(1/(1+np.exp(np.clip((dist-1)*3,-70,70))))
    center=ell(0,.035,.43,.55)
    # radial surround split in two directions, to distinguish a bounded
    # patch from the edge of a table/wall. Include the underside of the chin.
    rr=np.sqrt((X/.61)**2+((Y-.015)/.80)**2); ring=np.exp(-.5*((rr-1.18)/.23)**2)
    left=av(ring*(X<0)*(1+.35*(Y<.25))); right=av(ring*(X>0)*(1+.35*(Y<.25)))
    rim=(left+right)/2
    fs.extend([(center,z),(rim,z),(left-right,z)])
    # Cheeks brighter than the eye sockets, a highlight on the nose bridge,
    # and a comparatively dark arc of hair ABOVE the cheeks.
    cheeks=[blob(i*.23,.13,.13,.125) for i in (-1,1)]
    eyes=[blob(i*.225,-.145,.105,.075) for i in (-1,1)]
    brow=blob(0,-.35,.235,.09); nose=blob(0,-.08,.08,.12)
    mouth=blob(0,.32,.16,.062); chin=blob(0,.47,.16,.075)
    for j in (0,1): fs.append((z,.71*cheeks[j]+.18*brow+.11*nose-eyes[j]))
    cap=.54*blob(0,-.67,.33,.14)+.23*blob(-.43,-.28,.12,.20)+.23*blob(.43,-.28,.12,.20)
    fs.extend([(z,.5*(cheeks[0]+cheeks[1])-cap), (z,.49*(cheeks[0]+cheeks[1])/2+.51*chin-mouth)])
    # In upright figures the next area down is often the shirt rather than
    # more skin. Also sample either side of the neck above the shoulder slope.
    shirt=ell(0,1.22,.73,.34); sides=.5*(blob(-.85,.34,.20,.24)+blob(.85,.34,.20,.24))
    fs.extend([(shirt,z),(z,shirt-sides)])
    f=conv(n+'_probes',features[s],form_bank(fs),stride=st)
    E=np.eye(len(fs)); skin,rim,asym,el,er,hair,lip,torso,shirt_edge=E
    # Bounded votes prevent a high-contrast non-face edge from dominating.
    expr=[(el+er,0), (el+er,-.30), (el-er,0),(er-el,0),
          (hair,-.015),(hair,-.35),(lip,-.01),(lip,-.16),
          (asym,0),(-asym,0),(shirt_edge,0),(-shirt_edge,0),(shirt_edge,-.32),(-shirt_edge,-.32),
          (skin,0),(rim,0),(torso,0)]
    h=rel(n+'_positive',conv(n+'_responses',f,[v for v,_ in expr],[b for _,b in expr])); I=np.eye(len(expr));
    eyescore=I[0]-I[1]-.48*(I[2]+I[3]); hairscore=I[4]-I[5]; lipscore=I[6]-I[7];
    wd=eyescore*.82+hairscore*.51+lipscore*.31
    ws=1.05*I[14]-.63*I[15]-.12*(I[8]+I[9])-.085*I[16]
    we=(I[10]+I[11]-I[12]-I[13])*.24
    score=conv(n+'_head',h,[ws+wd+we],-.98 if s==48 else -.93)
    peak(n,score,'face',area=True)
    # A secondary detector requiring considerable internal facial contrast.
    # It doesn't require skin, so it can also help on monochrome photographs.
    if w>=4.5:
        det=conv(n+'_internal',h,[eyescore*2.45+hairscore*.52+lipscore*1.35+ws*.35+we*.7],-1.05)
        peak(n+'_detail',det,'detail')

for nm,s,w,k,st,tilt in [
    ('small',48,2.1,7,1,0), ('far',48,3.55,9,2,0),
    ('half',24,2.7,7,1,0), ('near',24,4.7,11,1,0),
    ('medium',12,3.35,9,1,0), ('large',12,5.35,13,1,0),
    ('portrait',6,4.35,9,1,0), ('inclined_l',12,4.4,11,1,-20),('inclined_r',12,4.4,11,1,20),
    ('eyes',24,7.7,15,1,0),('close_eyes',12,8.0,15,1,0)]:
    face(nm,s,w,k,st,tilt)

# Contrast on opposite sides of a silhouette. The kernels integrate soft
# strips along its contour; separate the left and right, top and bottom, so
# a single large straight line cannot furnish all the evidence for a person.
def silhouette(n,s,wide,st,pose):
    if pose=='bust':
        # in units of head width, origin halfway from face to collarbone
        parts=[ [(-.51,-.61,-1,0),(-.49,-.91,-.94,-.34),(-.38,-1.11,-.72,-.70)],
                [(-.30,-1.23,-.48,-.88),(0,-1.30,0,-1)],
                [(-.48,.00,-.87,.48),(-.39,.16,-.82,.57)],
                [(-.72,.37,-.40,-.92),(-1.00,.50,-.50,-.87),(-1.22,.68,-.61,-.79)],
                [(-1.32,.91,-.96,-.24),(-1.40,1.21,-.98,-.19),(-1.43,1.50,-1,-.13)] ]
        # parts 0-2 head sides, crown and inward turn below jaw; 3-4 shoulder and sleeve
        k=2*int(np.ceil(1.97*wide))+1
    else:
        # whole upright outline in units of shoulder width; the two feet can
        # be separate. The upper taper is important for rejecting tree trunks.
        parts=[ [(-.19,-1.58,-1,0),(-.19,-1.74,-.93,-.36)],
                [(0,-1.91,0,-1),(-.13,-1.86,-.57,-.82)],
                [(-.37,-1.33,-.58,-.81),(-.49,-1.17,-.98,-.18)],
                [(-.56,-.83,-.99,-.16),(-.57,-.55,-1,0),(-.50,-.27,-.96,.26)],
                [(-.37,.08,-1,0),(-.36,.42,-1,0),(-.36,.71,-1,0)],
                [(-.35,1.05,-1,0),(-.35,1.34,-1,0),(-.34,1.59,-1,0)] ]
        k=2*int(np.ceil(2.32*wide))+1
    x,y=points(k); fs=[]
    def mark(X,Y,sig):return gaussian(x,y,X*wide,Y*wide,sig*wide,sig*wide)
    for region in parts:
        for mirror in [-1,1]:
            field=0*x.astype(float)
            for xx,yy,nx,ny in region:
                off=.27 if pose=='bust' else .205
                field+=mark(mirror*(xx-off*nx),yy-off*ny,.165)-mark(mirror*(xx+off*nx),yy+off*ny,.19)
            fs.append(field/len(region))
    # Luma probes and lower-frequency chromatic contrasts in the same layout
    # (skin/hat/hair need not share the colour of the clothing).
    luma=features[s]
    F=[(0*z,z) for z in fs]; raw=conv(n+'_outline',luma,form_bank(F),stride=st)
    N=len(fs); I=np.eye(N); a,b,c,d,e2=[I[2*j:2*j+2].mean(axis=0) for j in range(5)]
    if pose=='bust': head=.54*a+.33*b+.13*c; body=.65*d+.35*e2
    else: head=.55*a+.45*b; body=.36*c+.31*d+.21*e2+.12*I[10:12].mean(axis=0)
    rows=[head,body];
    # Average absolute deviations from the group contrast: partial disagreement
    # is allowed (e.g. a head against a horizon), systematic mismatch isn't.
    for j,unit in enumerate(I):
        group=head if j<(6 if pose=='bust' else 4) else body
        rows.append(unit-group)
    row=np.array(rows); ab=rel(n+'_absolute',conv(n+'_differences',raw,np.concatenate([row,-row])))
    A=np.eye(len(row)*2); ar=A[:len(row)]+A[len(row):]
    mg=6 if pose=='bust' else 4
    hc=ar[0]-.72*ar[2:2+mg].mean(axis=0); bc=ar[1]-.87*ar[2+mg:].mean(axis=0)
    # Soft minimum over head and torso contrast after allowing polarity to change.
    rh=rel(n+'_contrast',conv(n+'_strengths',ab,[hc,bc,hc-bc,bc-hc,hc,bc],[0,0,0,0,-.36,-.40]))
    score=conv(n+'_joint',rh,[[2.35,2.5,-1.10,-1.13,-1.45,-1.48]],-.85 if pose=='bust' else -.76)
    # Local flesh-colour support in the head can make a weak outline credible.
    X,Y=x/wide,y/wide; cen=-.61 if pose=='bust' else -1.61; rx=.29 if pose=='bust' else .145; ry=rx*1.24
    face=gaussian(X,Y,0,cen,rx,ry); outer=.5*(gaussian(X,Y,rx*3,cen,rx*.85,ry)+gaussian(X,Y,-rx*3,cen,rx*.85,ry))
    skin=conv(n+'_flesh',luma,form_bank([(.62*face-.42*outer,0*x)]),stride=st)
    peak(n,add(n+'_response',score,skin),'bust' if pose=='bust' else 'figure','figure',area=pose!='bust')

for nm,s,w,st in [('bust_tiny',24,1.23,1),('bust_small',24,2.05,1),('bust_med',12,2.20,1),('bust_lg',12,3.45,1),('bust_xl',6,2.80,1)]:
    silhouette(nm,s,w,st,'bust')
for nm,s,w in [('walker',24,1.65),('pedestrian',24,2.7),('standing',12,2.18),('nearby',12,3.30)]:
    silhouette(nm,s,w,1,'standing')

# A part constellation: a rounded patch with skin chroma sits ABOVE a pair
# of shoulders. Detect the torso as a *colour difference*, not just a bright
# or dark region: center vs both lateral flanks, consistent upper and lower
# shirt colour, and lower than the backdrop above either shoulder. Couple
# the two maps before taking their maxima; unrelated peaks cannot cooperate.
peaks['clothed']=[]; coverage['clothed']=[]
chromas={s:conv('opponents_%d'%s,colors[s],[lum,r-g,b-(r+g)/2]) for s in (24,12,6)}
def dressed(n,s,w,lean):
    k=int(np.ceil(2.85*w))*2+1; x,y=points(k); X=x/w; Y=y/w
    # Origin at top of chest, offset of the head increases in proportion to
    # tilt. A shared shirt will still register even if the pants don't match.
    def blob(xx,yy,rx,ry):return gaussian(X,Y,xx+lean*yy,yy,rx,ry)
    mid=blob(0,.68,.48,.60); upper=blob(0,.18,.43,.27); lower=blob(0,1.28,.47,.38)
    sideL=.5*blob(-1.65,.11,.35,.50)+.5*blob(-1.74,1.25,.36,.48)
    sideR=.5*blob(1.65,.11,.35,.50)+.5*blob(1.74,1.25,.36,.48)
    above=.5*blob(-.91,-.93,.24,.29)+.5*blob(.91,-.93,.24,.29)
    across=blob(-.50,.74,.22,.45)-blob(.50,.74,.22,.45)
    outside=(sideL+sideR)/2
    ww=[mid-outside, upper-above, .5*(sideL-sideR),upper-lower,across]; T=np.eye(3)
    fw=[t[:,None,None]*f[None] for f in ww for t in T]
    dmap=conv(n+'_shirt_samples',chromas[s],fw)
    eye=np.eye(len(fw)); dd=rel(n+'_shirt_rect',conv(n+'_unsigned',dmap,np.concatenate((eye,-eye))))
    A=(np.eye(2*len(fw))[:len(fw)]+np.eye(2*len(fw))[len(fw):]).reshape(5,3,-1)
    combined=.90*A[:,0]+.62*A[:,1]+.52*A[:,2]
    v=.72*combined[0]+.28*combined[1]-.47*combined[2]-.18*combined[3]-.20*combined[4]
    nice=rel(n+'_fabric',conv(n+'_shirt_separation',dd,[v,v],[0,-.33]))
    cloth=conv(n+'_uniform',nice,[[1.13,-.89]])
    # Patch quality, bottom-heavy oval for the face, surrounded by non-skin
    # *and* a shirt. Flanking colour differences larger than the center/rim
    # contrast are a wall edge, not a localized head.
    fc=blob(0,-1.02,.27,.35); L=blob(-.78,-1.06,.23,.34); R=blob(.78,-1.06,.23,.34)
    top=blob(0,-1.84,.33,.22); chin=blob(0,-.13,.38,.23); bg=.32*(L+R)+.22*top+.14*chin
    zz=0*x; fs=[(fc,zz),(bg,zz),(L-R,zz),(mid,zz),(zz,fc-top)]
    fm=conv(n+'_head_samples',features[s],form_bank(fs)); I=np.eye(5); hr=rel(n+'_patch',conv(n+'_head_rect',fm,[I[0],I[1],I[2],-I[2],I[3],I[4],I[4]],[0,0,0,0,0,-.02,-.27]));
    head=conv(n+'_bounded_head',hr,[[.85,-.56,-.14,-.14,-.15,.46,-.46]])
    score=add(n+'_in_clothes',head,cloth)
    peak(n,conv(n+'_clothed_margin',score,[[1]],-.78),'clothed',area=True)
for nm,s,w,ln in [('littleshirt',24,.95,0),('shirt',24,1.55,0),('jumper',24,2.5,0),('blouse',12,1.9,0),('coat',12,2.9,0),('closecoat',6,2.4,0),('tiltedcoat1',12,1.75,-.43),('tiltedcoat2',12,1.75,.43)]:
    dressed(nm,s,w,ln)

# Spectral composition of the scene. Axial structure which persists in long
# straight lines suggests buildings/rooms, independently of colour. People
# doing outdoor activities often break up a relatively featureless field;
# compare middle and lateral texture rather than simply counting edges.
scene_more=[]
for s in (48,24):
    x,y=points(3); kernel=[]
    for a in np.arange(8)*np.pi/8:
        n=x*np.cos(a)+y*np.sin(a); tang=-x*np.sin(a)+y*np.cos(a)
        def directional(dx):return av(np.exp(-.5*((n-dx)/.52)**2-.5*(tang/.82)**2))
        kernel.append(directional(.88)-directional(-.88))
    kernel=np.array(kernel); kernel=np.concatenate((kernel,-kernel)); kernel=(kernel[:,None,:,:]*lum[None,:,None,None]);
    n='frequency_%d'%s; grad=rel(n+'_energy',conv(n+'_filters',colors[s],kernel,pad=0));
    # Aggregate all directions; small correction per direction for discrete-grid sampling.
    ang=np.arange(8)*np.pi/8; axial=np.cos(4*ang); df=np.tile((axial-axial.mean())/8,2)
    # An axis-minus-diagonal vote is independent of total contrast in expectation.
    orient=conv(n+'_summary',grad,[df,np.full(16,1/8)]); m=pool(n+'_global',orient,shape[orient][1]);
    scene_more.append(dense(n+'_axis_vote',m,[[-7.8,0]]))
    h=shape[orient][1]; yy,xx=np.mgrid[:h,:h]/(h-1); focus=np.exp(-.5*((xx-.5)/.255)**2-.5*((yy-.46)/.33)**2)
    periph=(np.abs(xx-.5)>.32).astype(float); delta=av(focus)-av(periph)
    scene_more.append(dense(n+'_focus',orient,np.stack((0*delta,1.35*delta)).reshape(1,-1)))

# Suppress common false alarms: long axial edges of rooms/vehicles, or warm
# colour over the entire scene. Conversely, modest support for a coloured
# jersey and for ground resembling a sports field or snow.
S=24; inp=colors[S]; Y=conv('gray',inp,[lum]); x,y=points(7); z=x*0.
# Oriented linear edge segments at multiple positions; aligned strips give
# very high responses on long straight light/dark borders.
def edge(t):
    a=np.deg2rad(t); n=x*np.cos(a)+y*np.sin(a); tang=-x*np.sin(a)+y*np.cos(a)
    def strip(q):return av(np.exp(-.5*((n-q)/.67)**2-.5*(tang/2.3)**2))
    return strip(.91)-strip(-.91)
E=np.array([edge(t) for t in (0,90,45,135)]); E=np.concatenate((E,-E)); grad=rel('straight_edge_excess',conv('line_segments',Y,E[:,None],-.075,pad=0)); grad=pool('line_mean',grad,shape[grad][1]); axes=dense('architectural_lines',grad,[[-3.8,-3.8,3.8,3.8]*2])
# Soft colour indicators used only for scene context. All defined in RGB,
# using two comparisons to competitors for a saturated primary.
col_rules=[(r-g,.12),(r-b,.12),(b-r,.10),(b-g,.06),(g-r,.015),(g-b,.075),(g-b,.17),(r-g,-.13),(b-g,-.04),(lum,.73)]
I=np.eye(len(col_rules)); q=rel('context_halfplanes',conv('colour_tests',inp,[w for w,_ in col_rules],[-v for _,v in col_rules]));
# max(0,min(u,v)) via (u+v-|u-v|)/2
v=np.array([I[0],I[1],I[0]-I[1],I[1]-I[0],I[2],I[3],I[2]-I[3],I[3]-I[2],I[4],I[5],I[4]-I[5],I[5]-I[4],I[9]])
h=rel('colour_comparisons',conv('primary_comparison',q,v)); v=np.zeros((4,len(v)))
for t in range(3):v[t,4*t:4*t+4]=[.5,.5,-.5,-.5]
v[3,-1]=1
p=conv('red_blue_grass_snow',h,v); blocks=pool('scene_layout',p,6); sheet=np.zeros((4,4,4))
sheet[0,1:3,:]=.75/8; sheet[1,1:3,:]=1.55/8; sheet[2,2:,:]=1.55/8; sheet[3,2:,:]=.48/8
context=dense('scene_vote',blocks,sheet.reshape(1,-1),-.045)
occ=pool('warm_fraction',sc,48); wood=rel('large_warm_areas',dense('warm_excess',occ,[[.55,.45]],-.11)); warm=dense('colour_specificity',wood,[[-.49]])

# Per-family pooling (best match plus broad support over size); then allow a
# single strong family to carry the decision. Scores are heuristic margins.
def collect(n,nodes):
    ws=[]
    for i,t in enumerate(nodes):
        w=np.zeros((len(nodes),1)); w[i,0]=1; ws.append(dense(n+str(i),t,w))
    return add(n+'_stack',*ws) if len(ws)>1 else ws[0]
def integrate(n,nodes,best,avg):
    u=collect(n,nodes); a=dense(n+'_avg',u,np.full((1,len(nodes)),avg/len(nodes)))
    mx=pool(n+'_max',reshape(n+'_bank',u,(1,len(nodes),1)),(len(nodes),1),'max'); b=dense(n+'_vote',mx,[[best]])
    return add(n+'_sum',a,b)
responses=[]
for kind,pk,avw in [('face',.77,.23),('bust',.70,.30),('figure',.70,.30),('detail',.74,.26),('clothed',.72,.28)]:
    responses.append(integrate(kind+'_sizes',peaks[kind],pk,avw))
joint=collect('families',responses)
# Competing parts (OR). A missing cue is NOT counter-evidence: e.g. a
# detected head needn't be accompanied by a detectable full standing outline.
# Reference matches: 85% skin membership with 1/5 membership outside,
# mild eye/hair shading; or roughly 15--20% contrast across a whole outline.
# Internal-feature-only matches need more contrast because corners resemble eyes.
parts=dense('margins',joint,np.eye(5),[.18,.12,.10,.12,0]);
familybest=pool('most_convincing',reshape('alternatives',parts,(1,5,1)),(5,1),'max')
s2=dense('disjunction',familybest,[[.60]])
weak=rel('weak_matches',dense('weak_thresholds',parts,np.eye(5),.20))
s1=dense('agreement',weak,[[.12,.09,.09,.04,.12]],-.052)
paired=dense('paired_parts',parts,[[.04,.02,.02,0,.32]])
spatial=[]
for kind,mult in [('face',.12),('figure',.08)]:
    rep=integrate(kind+'_repetition',coverage[kind],.60,.40)
    support=rel(kind+'_recurrent',dense(kind+'_recurrent_threshold',rep,[[1]],.48))
    spatial.append(dense(kind+'_recurrent_vote',support,[[mult]]))
margin=add('person_margin',s1,s2,paired,*spatial,axes,context,warm,*scene_more); final=dense('class_scores',margin,[[-.5],[.5]])
(out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n'); np.savez(out/'weights.npz',**weights)
