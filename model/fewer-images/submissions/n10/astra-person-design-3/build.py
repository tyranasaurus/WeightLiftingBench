"""Construct a person detector from analytic filters and hand-written rules only."""
import json
import math
import sys
from pathlib import Path
import numpy as np

layers, weights = [], {}
sizes = {'image': (3, 96, 96)}

def layer(name, op, src, shape, **kw):
    assert name not in sizes
    layers.append(dict(name=name, op=op, inputs=[src] if isinstance(src,str) else src, **kw))
    sizes[name] = tuple(shape)
    return name

def conv(name, src, w, b=0, step=1, pad=0):
    w = np.asarray(w, np.float32)
    if w.ndim == 2: w = w[:,:,None,None]
    n,c,k,_ = w.shape
    assert sizes[src][0] == c
    weights[name+'.W'] = w
    weights[name+'.b'] = np.broadcast_to(b,(n,)).astype(np.float32).copy()
    _,h,x = sizes[src]
    return layer(name,'conv',src,(n, (h+2*pad-k)//step+1, (x+2*pad-k)//step+1),out=n,kernel=[k,k],stride=step,padding=pad)

def dense(name,src,w,b=0):
    w = np.array(w, np.float32).reshape((-1, math.prod(sizes[src])))
    weights[name+'.W']=w
    weights[name+'.b']=np.broadcast_to(b,(len(w),)).astype(np.float32).copy()
    return layer(name,'dense',src,(len(w),),out=len(w))

def pool(name,src,k,stride=None,op='avgpool'):
    if stride is None: stride=k
    c,h,w=sizes[src]
    return layer(name,op,src,(c,(h-k)//stride+1,(w-k)//stride+1),size=[k,k],stride=[stride,stride])

def rel(name,src): return layer(name,'relu',src,sizes[src])
def add(name,*src):
    assert all(sizes[x]==sizes[src[0]] for x in src)
    return layer(name,'add',list(src),sizes[src[0]])
def shape(name,src,dims):
    assert math.prod(dims)==math.prod(sizes[src])
    assert name not in sizes
    layers.append({'name':name,'op':'reshape','inputs':[src],'shape':list(dims)})
    sizes[name]=tuple(dims)
    return name

def channel_max(name,src):
    c,h,w=sizes[src]
    a=shape(name+'_stack',src,(1,c,h*w))
    a=layer(name+'_reduce','maxpool',a,(1,1,h*w),size=[c,1],stride=[1,1])
    return shape(name,a,(1,h,w))

# Opponent-colour wedges. Values in a wedge have the relative excess of red
# over green, and green over blue, typical of diffuse reflection from skin.
# Boundary costs are measured in RGB units, not learned chrominance bins.
ycoef=np.array([.299,.587,.114]); dr=np.array([1.,-1.,0]); db=np.array([0,1.,-1.])
hw=[]; hb=[]; clauses=[]
def hinge(v,b):
    hw.append(v); hb.append(b); return len(hw)-1

def wedge(amin,amax,bmin,bmax,lower,upper,params):
    # lower/upper give oblique sides of the wedge in (R-G,G-B).
    sides=[(-dr+amin[0]*ycoef,amin[1]),(dr-amax[0]*ycoef,-amax[1]),
           (-db+bmin[0]*ycoef,bmin[1]),(db-bmax[0]*ycoef,-bmax[1]),
           (-db+lower[0]*dr,lower[1]),(db-upper[0]*dr,-upper[1]),
           (-ycoef,.075),(np.array([1.,0.,0.]),-1.)]
    clauses.append([(hinge(v,b),g) for (v,b),g in zip(sides,params)])

wedge((.10,.014),(.46,.055),(.027,.004),(.31,.045),(.09,-.008),(1.05,.012),[38,15,30,18,23,26,16,5])
wedge((.064,.009),(.61,.080),(-.009,0),(.42,.04),(-.045,-.008),(1.48,.023),[42,15,26,17,24,22,16,5])
lum=hinge(ycoef,0)
rgb=pool('two_pixel_rgb','image',2)
# A modest, bounded grey-world correction. It offsets a fraction of a
# whole-frame lighting cast, and cannot turn one hue into an arbitrary one.
light=pool('ambient_light',rgb,48)
light_w=np.stack([.47*dr,.47*dr,.47*db,.47*db]); offsets=np.array([.044,-.044,.040,-.040])-.47*np.array([.017,.017,.008,.008])
light=rel('ambient_ramps',conv('ambient_bounds',light,light_w,offsets))
transform=np.array([[-1,1,0,0],[0,0,0,0],[0,0,1,-1]])
correction=dense('ambient_planes',light,np.repeat(transform,48*48,axis=0),np.repeat([.044,0,-.040],48*48))
corrected=add('neutralized_light',rgb,shape('ambient_rgb',correction,(3,48,48)))
f=rel('chromatic_violations',conv('opponent_halfspaces',corrected,hw,hb))
w=np.zeros((3,len(hw)))
for i,terms in enumerate(clauses):
    for j,g in terms: w[i,j]=-g
w[2,lum]=1
col=rel('pixel_memberships',conv('colour_rules',f,w,[1,1,0]))
# A permissive colour match always counts less than a strict one. Subsequent
# filters use channels (warm-skin membership, luminance).
base=conv('face_material_and_light',col,[[.73,.27,0],[0,0,1]])

# Oriented changes of brightness, with averages along the contour tangent.
yg,xg=np.mgrid[-1:2,-1:2]
D=[]
for theta in [0,math.pi/4,math.pi/2,3*math.pi/4]:
    kernel=(xg*math.cos(theta)+yg*math.sin(theta))*np.exp(-(xg*xg+yg*yg)/(2*.82**2))
    kernel=kernel / np.maximum(kernel,0).sum()
    for sign in [1,-1]: D.append(np.stack((np.zeros_like(kernel),sign*kernel)))
e0=rel('one_sided_edges',conv('oriented_differences',base,D,pad=1))
w=np.zeros((4,8));
for j in range(4): w[j,2*j:2*j+2]=1
edge=conv('unsigned_edges',e0,w)

# Probability-like scores for a face at one position. These templates are
# differences of normalized Gaussians, so their weights have a direct
# interpretation as differences of regional mean luminance or colour match.
def unit(a): return a/(a.sum()+1.e-24)
def grid(k):
    a=np.arange(k)-(k-1)/2
    return np.meshgrid(a,a)

def scalar_peak(name,src):
    assert sizes[src][0]==1
    return pool(name,src,sizes[src][1],op='maxpool')

def spatial_penalty(name,src,amount):
    # A weak prior against accidental fragments far from the subject area;
    # not a crop: high-scoring off-centre candidates remain eligible.
    _,n,_=sizes[src]
    u=(np.arange(n)+.5)/n
    x,y=np.meshgrid(u,u)
    prior=-amount*(.7*(abs(x-.5)*2)**2 + .4*(abs(y-.42)*2)**2 + .7*np.maximum((y-.64)/.36,0)**1.5)
    a=dense(name+'_coordinates',light,np.zeros((n*n,4)),prior.ravel())
    return add(name,src,shape(name+'_map',a,(1,n,n)))

def face_scale(tag,src,level):
    k=7; X,Y=grid(k)
    # Generic squashed frontal, oblique and profile faces. Width and height
    # are given in this pyramid level's pixels (not original image pixels).
    specs=[]
    for hx,hy in [(2.35,3.25),(3.25,3.85),(4.7,4.05),(3.9,5.45)]:
        specs.append((hx,hy,0,0))
        if hy>3.5:
            for tilt in [-.24,.24]: specs.append((hx,hy,tilt,0))
    # Three-quarter view: one half of the hair/head region is more visible.
    for shift in [-.18,.18]: specs.append((3.65,4.75,0,shift))
    templates=[]; skin_mix=[]
    for fw,fh,t,shift in specs:
        u=(math.cos(t)*X+math.sin(t)*Y)/fw
        v=(-math.sin(t)*X+math.cos(t)*Y)/fh
        def g(cx,cy,sx,sy): return unit(np.exp(-.5*(((u-cx)/sx)**2+((v-cy)/sy)**2)))
        core=g(0,.025,.225,.246)
        left=g(-.70,0,.19,.32); right=g(.70,0,.19,.32)
        crown=g(shift*.6,-.63,.35,.155)
        shirt=g(0,.90,.46,.20)
        surround=.24*(left+right)+.25*crown+.27*shirt
        # The halo is fairly close, so broad material and lighting regions
        # cancel over most of the image.
        colourblob=core-.91*surround
        cheeks=.5*(g(-.21-shift*.3,.115,.13,.15)+g(.21-shift*.3,.115,.13,.15))
        brow=.5*(g(-.225-shift,-.115,.116,.071)+g(.225-shift,-.115,.116,.071))
        forehead=g(-shift,-.315,.17,.125)
        nose=g(-shift,.006,.096,.14)
        mouth=g(-shift,.29,.162,.071)
        chin=g(-shift,.365,.16,.097)
        eyes=.36*forehead+.39*cheeks+.25*nose-brow
        mouthband=.48*chin+.52*cheeks-mouth
        hair=core-(.53*crown+.235*g(-.46+shift,-.18,.13,.23)+.235*g(.46+shift,-.18,.13,.23))
        internal=.78*eyes+.22*mouthband
        facecontrast=.47*internal+.53*hair
        templates.append(np.stack((1.36*colourblob+.135*core,1.22*facecontrast)))
        skin_mix.append(0)
        # Stronger luminance structure is needed when colour is not reliable.
        # Kept as a separate expert so the skin response isn't compulsory.
        if not t and not shift and fh>=3.75:
            templates.append(np.stack((.51*colourblob+.09*core,2.90*(.68*internal+.32*hair))))
            skin_mix.append(1)
    stride=2 if level==0 else 1
    b=np.array([-.33 if m==1 else -.85 for m in skin_mix])
    response=conv(tag+'_head_bank',src,templates,b,step=stride,pad=3)
    a=channel_max(tag+'_best_head',response)
    a=spatial_penalty(tag+'_local_head',a,.095)
    return a

# Each pyramid is made by averaging the original response maps, rather than
# rescaling any collection of templates. The uneven ratios improve coverage.
levels=[(1,base),(2,None),(3,None),(5,None),(8,None)]
face_maps=[]; facepeaks=[]; facevotes=[]
for i,(s,src) in enumerate(levels):
    if src is None: src=pool('pyramid_'+str(s),base,s)
    tag='s'+str(s)
    a=face_scale(tag,src,i)
    face_maps.append(a)
    peak=scalar_peak(tag+'_peak_head',a); facepeaks.append(peak)
    # A sum over coarse cells gives repeated detections (e.g. a group photo)
    # a small bonus, without the many near-identical peaks from one face.
    n=sizes[a][1]; block=max(1,n//4)
    cell=pool(tag+'_head_cells',a,block,op='maxpool')
    active=rel(tag+'_head_votes',conv(tag+'_vote_gate',cell,[[1]],.14))
    facevotes.append(pool(tag+'_mean_vote',active,sizes[active][1]))

# Extra frontal-face route. Multiple contrasts must agree at the SAME
# position and scale: pair of eyes below forehead, eyes above cheeks, bridge
# between eyes, face within hair/cheek outline, lips between cheeks and chin.
# Different channels saturate separately: a black stripe alone cannot
# substitute for the entire configuration. Applied at moderate resolution.
fronts=[]
for s in [1,2,4,7]:
    src=base if s==1 else (('pyramid_'+str(s)) if s in [2,3,5,8] else pool('portrait_pyramid_'+str(s),base,s))
    k=9; X,Y=grid(k); ws=[]; bs=[]; groups=[]
    for fw,fh,angle in [(4.15,5.45,0),(5.7,6.2,0),(4.55,7.3,0),(5.1,6.4,-.19),(5.1,6.4,.19)]:
        u=(X*np.cos(angle)+Y*np.sin(angle))/fw; v=(-X*np.sin(angle)+Y*np.cos(angle))/fh
        def g(x,y,rx,ry): return unit(np.exp(-.5*((u-x)**2/rx**2+(v-y)**2/ry**2)))
        eye=[g(q*.21,-.105,.094,.058) for q in [-1,1]]
        brow=sum(eye)*.5
        core=g(0,.045,.233,.265); forehead=g(0,-.302,.185,.092)
        cheek=[g(q*.20,.105,.120,.105) for q in [-1,1]]
        bridge=g(0,-.055,.079,.09)
        lips=g(0,.273,.144,.046); chin=g(0,.379,.132,.080)
        crown=g(0,-.61,.31,.12)
        outline=.34*crown+.33*g(-.535,-.10,.119,.27)+.33*g(.535,-.10,.119,.27)
        halo=.24*crown+.24*g(-.66,0,.17,.27)+.24*g(.66,0,.17,.27)+.28*g(0,.75,.39,.15)
        # (skin, luminance, full contrast, relative vote). A pair of ReLUs
        # represents a truncated ramp, including a little negative evidence.
        items=[(0,forehead-brow,.118,.16),(0,cheek[0]-eye[0],.12,.14),
               (0,cheek[1]-eye[1],.12,.14),(0,bridge-brow,.092,.11),
               (0,.47*chin+.53*(cheek[0]+cheek[1])*.5-lips,.078,.13),
               (0,core-outline,.18,.09),(core-.84*halo,0,.65,.23)]
        idx=[]
        for colf,lumf,div,vote in items:
            filt=np.stack((colf+0*X,lumf+0*Y))/div
            ws.extend((filt,filt)); bs.extend((.18,-.82)); idx.append(vote)
        groups.append(idx)
    # Stride two suffices here because the tested face is larger.
    a=rel('portrait_ramps_'+str(s),conv('portrait_tests_'+str(s),src,ws,bs,step=2 if s<=2 else 1,pad=4))
    sw=np.zeros((len(groups),len(bs))); j=0
    for i,votes in enumerate(groups):
        for v in votes: sw[i,j:j+2]=[v,-v];j+=2
    a=conv('portrait_agreement_'+str(s),a,sw,-.73)
    a=channel_max('portrait_hypothesis_'+str(s),a)
    a=spatial_penalty('portrait_prior_'+str(s),a,.052)
    fronts.append(scalar_peak('portrait_peak_'+str(s),a))

# Contour grammar for a torso or standing figure. Segments are quadrature-
# free oriented edge-energy stencils. There is no assumption that clothes
# are dark (or bright). The subtraction of the local orientation background
# rejects highly textured patches and rewards alignment, not raw edge power.
def element_grid(k):
    X,Y=grid(k)
    return X,Y, np.zeros((4,k,k),dtype=float)

def segment_field(X,Y,segments):
    a=np.zeros((4,)+X.shape); mass=0.
    for x1,y1,x2,y2,gain,blur in segments:
        dx,dy=x2-x1,y2-y1
        le=math.hypot(dx,dy); t=np.clip(((X-x1)*dx+(Y-y1)*dy)/le**2,0,1)
        dist=(X-x1-t*dx)**2+(Y-y1-t*dy)**2
        prof=unit(np.exp(-.5*dist/blur**2))
        theta=math.atan2(dx,-dy)
        # Interpolate between the two bracketing unsigned gradient bins.
        bin=(theta*4/math.pi)%4; j=int(bin); f=bin-j
        a[j]+=prof*gain*(1-f); a[(j+1)%4]+=prof*gain*f; mass+=gain
    return a/mass

def subtract_background(f,k):
    X,Y=grid(k)
    broad=unit(np.exp(-.5*((X/(k*.34))**2+(Y/(k*.38))**2)))
    for j in range(4): f[j]-=f[j].sum()*broad
    return f

# For every unsigned-edge template there is a separate material filter
# at exactly the same centre. This enforces a plausible head/body relation.
def body_bank(s,edges,mats):
    tag='body'+str(s); k=9; X,Y,empty=element_grid(k); es=[]; ms=[]; bs=[]
    # Coordinate system: anchor around the upper chest. The head tapers
    # into a narrower neck, shoulder slopes down/out, then torso sides.
    for width,height in [(1.,1.),(.76,1.06),(1.34,.88)]:
        for motion in [0,-.22,.22]:
            def tr(x,y): return ((width*x+motion*(.3-y)/2),height*y)
            def pt(p): return tr(*p)
            def seg(p,q,w,b=.45): return (*pt(p),*pt(q),w,b)
            def g(x,y,rx,ry):
                cx,cy=tr(x,y)
                return unit(np.exp(-.5*((X-cx)**2/(rx*width)**2+(Y-cy)**2/(ry*height)**2)))
            e=[]
            e += [seg((-.52,-3.32),(.52,-3.32),.85),seg((-.59,-3.18),(-.67,-2.25),.8),seg((.59,-3.18),(.67,-2.25),.8)]
            for sign in [-1,1]:
                def P(x,y): return (sign*x,y)
                e += [seg(P(.47,-1.90),P(1.70,-1.31),1.25),seg(P(1.72,-1.30),P(2.05,.55),1.05),
                      seg(P(1.26,.40),P(1.15,2.37),.55)]
            f=subtract_background(segment_field(X,Y,e),k)
            core=g(0,-2.55,.49,.64); noggin=core-.27*g(-1.75,-2.65,.52,.78)-.27*g(1.75,-2.65,.52,.78)-.29*g(0,-.19,.96,.85)-.09*g(0,-4,.65,.39)
            es.append(2.80*f);ms.append(np.stack((1.02*noggin,.45*(core-.56*g(0,-3.55,.67,.48)-.22*g(-.76,-2.75,.36,.67)-.22*g(.76,-2.75,.36,.67)))))
            bs.append(-.68)
            # Shape-only view for a hood, helmet, back of a head, etc.
            if motion==0:
                es.append(6.2*f);ms.append(np.stack((.28*noggin,np.zeros_like(core))));bs.append(-.29)
    a=conv(tag+'_contours',edges,es,bs,pad=k//2)
    b=conv(tag+'_materials',mats,ms,pad=k//2)
    a=channel_max(tag+'_best',add(tag+'_aligned',a,b))
    a=spatial_penalty(tag+'_position',a,.12)
    return scalar_peak(tag+'_peak',a)

bodies=[]
for s in [2,3,5,8]:
    er=pool('edge_pyramid_'+str(s),edge,s)
    bodies.append(body_bank(s,er,'pyramid_'+str(s)))

# Additional contour-only templates at small scales actually include legs.
# A signed local contrast test (both polarities) complements the outline.
walks=[]
for s in [2,3,5]:
    k=11; X,Y=grid(k); es=[]; ms=[]; bs=[]
    for breadth in [.77,1.18]:
        for walking in [-1,0,1]:
            for lean in ([0] if not walking else [-.19,.19]):
                def tr(x,y): return (x*breadth+lean*(y+1),y)
                def sg(p,q,w,blur=.42): return (*tr(*p),*tr(*q),w,blur)
                def g(x,y,rx,ry):
                    cx,cy=tr(x,y)
                    return unit(np.exp(-.5*(((X-cx)/(rx*breadth))**2+((Y-cy)/ry)**2)))
                e=[sg((-.34,-4.0),(.34,-4.0),.48)]
                for q in [-1,1]:
                    legspread=.35+q*walking*.38
                    e.extend([sg((q*.42,-3.88),(q*.47,-3.15),.51),sg((q*.33,-2.96),(q*1.04,-2.58),.8),
                              sg((q*1.04,-2.58),(q*1.29,-.82),.86),sg((q*.8,-.2),(q*.74,.88),.41),
                              sg((q*.76,.88),(q*(.69+legspread),2.28),.83),sg((q*(.69+legspread),2.28),(q*(.7+2*legspread),4.03),1.)])
                f=subtract_background(segment_field(X,Y,e),k)
                face=g(0,-3.49,.38,.5); shoulders=g(0,-1.81,.6,.77)
                env=.5*g(-2,-1.78,.5,.85)+.5*g(2,-1.78,.5,.85)
                h=face-.32*g(-1.34,-3.5,.46,.62)-.32*g(1.34,-3.5,.46,.62)-.28*shoulders
                for polarity in [-1,1]:
                    es.append(3.65*f); ms.append(np.stack((.58*h,.48*polarity*(shoulders-env))));bs.append(-.53)
    a=conv('stride_contours_'+str(s),'edge_pyramid_'+str(s),es,bs,pad=5)
    a=add('stride_fusion_'+str(s),a,conv('stride_colour_'+str(s),'pyramid_'+str(s),ms,pad=5))
    a=channel_max('stride_pose_'+str(s),a)
    a=spatial_penalty('stride_position_'+str(s),a,.09)
    walks.append(scalar_peak('stride_peak_'+str(s),a))

# Global context. Evidence from the proportion of skin-coloured material
# is useful only in moderation: a uniformly beige room is not a person.
# Prefer concentrated material in the central/upper image to a cast that
# also covers the entire periphery. All statistics here occur at inference.
m=pool('pooled_material',col,6)
X,Y=np.meshgrid((np.arange(8)+.5)/8,(np.arange(8)+.5)/8)
subject=unit(np.exp(-.5*(((X-.50)/.245)**2+((Y-.45)/.255)**2)))
outer=unit(np.clip(((X-.5)/.43)**2+((Y-.48)/.50)**2,0,1))
wa=np.zeros((5,3,8,8));wa[0,0]=subject;wa[1,0]=outer;wa[2,1]=subject;wa[3,0]=1/64;wa[4,1]=1/64
cs=dense('material_distribution',m,wa)
cw=[];cb=[];co=[]
def scalar_ramp(a,b,lo,hi,gain):
    """g * clamp((a.x+b-lo)/(hi-lo),0,1)"""
    for threshold,sign in [(lo,1),(hi,-1)]: cw.append(np.array(a)/(hi-lo)); cb.append((b-threshold)/(hi-lo)); co.append(sign*gain)
scalar_ramp([1,-.61,0,0,0],0,.012,.17,.21)
scalar_ramp([0,0,1,0,-.70],0,.01,.18,.085)
scalar_ramp([0,0,0,1,0],0,.006,.095,.12)
scalar_ramp([0,0,0,1,0],0,.26,.60,-.13)
scalar_ramp([0,0,0,0,1],0,.44,.75,-.075)
act=rel('material_ranges',dense('material_knots',cs,cw,cb))
colour_context=dense('material_context',act,[co],-.16)

# Dark hair or clothes interposed in front of bright grass, water, snow or
# an indoor wall create a compact vertical interruption. Require contrast
# against BOTH sides, avoiding a single shadow boundary. Works for either
# polarity and for people facing away. Additional apex/chest width cue.
solids=[]
for s in [2,3,5,8]:
    k=9; X,Y=grid(k); ws=[];bs=[];votes=[]
    for width in [.8,1.25]:
        def g(x,y,rx,ry): return unit(np.exp(-.5*(((X-width*x)/(rx*width))**2+((Y-y)/ry)**2)))
        chest=g(0,-.1,.80,.94); head=g(0,-2.66,.45,.63); mid=g(0,1.65,.72,.89)
        terms=[]
        for sign in [-1,1]:
            for region,side,gap,wt in [(head,g(sign*1.18,-2.70,.38,.66),.19,.18),
             (chest,g(sign*2.19,-.05,.51,1.1),.24,.19),(mid,g(sign*1.93,1.54,.49,.86),.19,.08)]:
                terms.append((side-region,gap,wt))
        # Taper above shoulders, with a compact cap silhouetted against the background.
        terms.append((g(0,-3.93,.59,.28)-head,.18,.10))
        skin=head-.48*g(0,-.41,.86,.8)-.26*g(-1.3,-2.58,.47,.72)-.26*g(1.3,-2.58,.47,.72)
        for polarity in [-1,1]:
            start=len(ws); coeff=[]
            for filt,gap,wt in terms:
                pair=np.stack((0*filt,polarity*filt/gap))
                ws.extend([pair,pair]); bs.extend([.10,-.90]);coeff.extend([wt,-wt])
            pair=np.stack((skin/.63,0*skin));ws.extend([pair,pair]);bs.extend([.0,-1.]);coeff.extend([.22,-.22]);votes.append((start,coeff))
    a=rel('silhouette_ramps_'+str(s),conv('silhouette_tests_'+str(s),'pyramid_'+str(s),ws,bs,pad=4))
    w=np.zeros((len(votes),len(ws)))
    for i,(j,v) in enumerate(votes): w[i,j:j+len(v)]=v
    a=conv('silhouette_agreement_'+str(s),a,w,-.75)
    a=channel_max('silhouette_choice_'+str(s),a)
    a=spatial_penalty('silhouette_position_'+str(s),a,.09)
    solids.append(scalar_peak('silhouette_peak_'+str(s),a))

# Combine scalars without concatenation. All offsets and calibrations are
# design constants; clipped responses let several cues corroborate one another.
def pack(name,seq):
    arr=[]
    for i,src in enumerate(seq):
        w=np.zeros((len(seq),1));w[i,0]=1
        arr.append(conv(name+'_slot'+str(i),src,w))
    return add(name,*arr)

def family(name,seq,center,scale):
    v=pack(name+'_scales',seq)
    peak=channel_max(name+'_maximum',v)
    return dense(name,peak,[[scale]],-center*scale)

# Scores from the head bank are offsets above a moderate skin-island
# threshold (or a high internal contrast threshold in its grey route).
head=family('head_evidence',facepeaks,.065,1.0)
front=family('portrait_evidence',fronts,-.030,1.7)
body=family('shoulder_evidence',bodies,.048,1.0)
walk=family('figure_evidence',walks,.026,1.1)
solid=family('silhouette_evidence',solids,.0,1.35)
vv=pack('votes_by_scale',facevotes)
group=dense('repetition',channel_max('strongest_group',vv),[[.90]],-.028)
vs=pack('detector_evidence',[shape('vector_'+n,n,(1,1,1)) for n in [head,front,body,walk,solid]])
# Soft-OR: a weighted consensus plus the strongest interpretation. Use a
# bounded floor/ceiling for each cue, to avoid one flat channel dominating.
lo=-.29;hi=.51
w=np.r_[np.eye(5),np.eye(5)]; b=[-lo]*5+[-hi]*5
v=rel('expert_ramps',conv('expert_limits',vs,w,b));v=conv('bounded_experts',v,np.c_[np.eye(5),-np.eye(5)],[lo]*5)
cons=dense('weighted_consensus',v,[[.46,.23,.20,.055,.055]])
# The shape-only and portrait experts are corroborating cues. Their max
# should count slightly less than an excellent skin/structure detection.
z=conv('best_expert_inputs',v,np.diag([1,.82,.89,.74,.74]),[0,-.024,-.01,-.03,-.03])
best=dense('salient_detection',channel_max('most_salient',z),[[.50]])
logit=add('person_logit',cons,best,group,colour_context)
dense('class_scores',logit,[[-.5],[.5]],[.105,-.105])

if __name__=='__main__':
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n')
    np.savez(out/'weights.npz',**weights)
