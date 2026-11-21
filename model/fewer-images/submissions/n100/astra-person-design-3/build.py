"""Data-independent, analytic multiscale part detector. No input files."""
import sys, json, math
from pathlib import Path
import numpy as np

outdir = Path(sys.argv[1]); outdir.mkdir(parents=True, exist_ok=True)
layers=[]; weights={}; dims={'image':(3,96,96)}; votes=[]

def layer(name,op,inputs,output_shape,**kw):
    if isinstance(inputs,str): inputs=[inputs]
    assert name not in dims
    layers.append(dict(name=name,op=op,inputs=inputs,**kw)); dims[name]=output_shape
    return name

def conv(name,src,w,b=0,p=0,s=1):
    w=np.array(w,np.float32)
    if w.ndim==2: w=w[:,:,None,None]
    n,c,k,_=w.shape; C,h,v=dims[src]; assert c==C
    weights[name+'.W']=w; weights[name+'.b']=np.zeros(n,np.float32)+np.array(b,np.float32)
    return layer(name,'conv',src,(n,(h+2*p-k)//s+1,(v+2*p-k)//s+1),out=n,kernel=[k,k],padding=p,stride=s)

def dense(name,src,w,b=0):
    w=np.array(w,np.float32).reshape(-1,math.prod(dims[src])); n=len(w)
    weights[name+'.W']=w; weights[name+'.b']=np.zeros(n,np.float32)+np.array(b,np.float32)
    return layer(name,'dense',src,(n,),out=n)

def rel(name,src): return layer(name,'relu',src,dims[src])
def pool(name,src,kind,k,s):
    c,h,w=dims[src]; return layer(name,kind+'pool',src,(c,(h-k)//s+1,(w-k)//s+1),size=[k,k],stride=[s,s])
def add(name,*src):
    assert all(dims[a]==dims[src[0]] for a in src)
    return layer(name,'add',list(src),dims[src[0]])
def aff(name,src,A,b=0): return conv(name,src,A,b)
def pw(name,src,a,b): return aff(name,src,[a],b)
def halve(src): return pool(src+'_half',src,'avg',2,2)

def grid(k,aa=1):
    t=(np.arange(k*aa)+.5)/aa-k/2
    return np.meshgrid(t,t)
def unit(a): return a/max(a.sum(),1e-25)
def uni(k,rx,ry,dx=0,dy=0,tilt=0,l=0):
    X,Y=grid(k,5); x=X-dx;y=Y-dy; shape=(((x-tilt*y)/rx)**2+(y/ry)**2)**.5
    z=((shape<=1)&(shape>=l)).reshape(k,5,k,5).mean((1,3))
    return unit(z)
def gauss(k,rx,ry,dx=0,dy=0):
    x,y=grid(k);return unit(np.exp(-.5*(((x-dx)/rx)**2+((y-dy)/ry)**2)))
def dim_vote(name,src,values,b=0):
    weights_list=np.array(values).reshape((1,-1)); z=dense(name,src,weights_list,b);votes.append(z)

# Pixelwise skin membership in an oblique wedge of the RGB cube. Use raw
# chroma and a mild gray-world correction. Hinge penalties have broad ramps.
base=pool('antialias','image','avg',2,2)
rg=np.array([1.,-1,0]);gb=np.array([0.,1,-1]);red=np.array([1.,0,0]);lum=np.array([.27,.56,.17])
# f=b+w.rgb is positive outside the desired gamut.
planes=[(-rg+.020*red,.012,18), (rg-.38*red,-.035,10),
        (-gb-.135*rg,.004,13), (gb-1.22*rg,-.024,15),
        (-(rg+gb)+.075*red,.028,12), (rg+gb-.69*red,-.04,7),
        (-red,.13,5)]
W=[a*g for a,b,g in planes]; B=[b*g for a,b,g in planes]
raw=conv('gamut_planes',base,W,B)
mu=pool('white_balance',base,'avg',48,48)
# The following reshape broadcasts per-image offsets using a small dense map.
# Do not broadcast via add across different shapes (not a network primitive).
chromatic=np.eye(3)-np.ones((3,1))@lum[None,:]
wb=aff('illuminant_projection',mu,-.64*np.array(W)@chromatic)
# Avoid a large dense broadcast by replicating the three independent color
# offsets at a coarse grid, then constant-extension convolution is unnecessary:
# instead do the channel adjustment via exact global stencil on the planes.
# Broadcast an offset per plane using a channel-major dense then reshape.
n=len(W); sz=48
wbgrid=layer('illuminant_map','reshape',dense('replicate_illuminant',wb,np.repeat(np.eye(n),sz*sz,axis=0)),(n,sz,sz),shape=[n,sz,sz])
h1=rel('raw_penalties',raw); h2=rel('balanced_penalties',add('balanced_planes',raw,wbgrid))
a=rel('native_skin',pw('native_evidence',h1,-np.ones(n),1))
b=rel('balanced_skin',pw('balanced_evidence',h2,-np.ones(n),1))
skin=add('skin',aff('native_weight',a,[[.63]]),aff('balanced_weight',b,[[.37]]))

# Luminance and color gradients: positive fine horizontal, vertical and both
# diagonal derivatives, plus a chromatic component for the contour measures.
k=3; X,Y=grid(k); g=np.exp(-.5*(X*X+Y*Y)/.75**2)
kernels=[]
for v in (lum,rg,gb):
    for u,t in ((1,0),(0,1),(2**-.5,2**-.5),(2**-.5,-2**-.5)):
        f=(u*X+t*Y)*g;f/=f[f>0].sum()
        kernels.extend([v[:,None,None]*f,-v[:,None,None]*f])
# Replicate no exterior: border gradients are not evidence, so use valid
# differences with the input's naturally darker surroundings in later stencils.
grad=rel('rectified_derivatives',conv('image_derivatives',base,kernels,p=1))
proj=np.zeros((4,24));
for j in range(4):
    for i,w in enumerate([1,.28,.22]):proj[j,8*i+2*j:8*i+2*j+2]=w
ori=aff('contour_energy',grad,proj)
luma=aff('luminance',base,[lum]); texture=aff('unsigned_contrast',ori,[[.5,.5,0,0]])
# Merge in a compact set of channels (skin, luminance, local contrast, and
# support mask). The support channel distinguishes boundaries from evidence.
F=add('part_features',aff('map_skin',skin,[[1],[0],[0],[0]],[0,0,0,1]),
      aff('map_luma',luma,[[0],[1],[0],[0]]),aff('map_contrast',texture,[[0],[0],[1],[0]]))
factors={1:F,2:halve(F)};factors[4]=halve(factors[2])

# Inspect facial regions at multiple scales. Contrast the core with each
# side separately: a straight interface should not look like a face. A
# second conjunction checks eye-band, hair and mouth/cheek contrasts.
def faces(src,key,rx,ry,k,small=False):
    p=k//2; name='f'+key; w=[]; bs=[]
    def register(terms,offset=0):
        f=np.zeros((4,k,k));
        for ch,m,d in terms:f[ch]+=d*m
        w.append(f);bs.append(offset)
    core=uni(k,rx*.92,ry*.88); top=uni(k,rx*.86,ry*.40,0,-1.20*ry)
    left=uni(k,rx*.48,ry*.80,-1.29*rx,-.09*ry);right=left[:,::-1]
    bot=uni(k,rx*.85,ry*.43,0,1.25*ry)
    ring=.28*(left+right)+.24*top+.20*bot
    register([(0,core,1)]) # central skin
    register([(0,core,1),(0,ring,-.93),(3,core,.52),(3,ring,.42)],-.94)
    for m in (left,right,top,bot):
        register([(0,m,1),(0,core,-1)],.15)
    eyes=.5*(gauss(k,max(.30,rx*.23),max(.26,ry*.12),dx=-rx*.43,dy=-ry*.24)+gauss(k,max(.30,rx*.23),max(.26,ry*.12),dx=rx*.43,dy=-ry*.24))
    cheeks=.5*(gauss(k,max(.38,rx*.27),max(.26,ry*.14),dx=-rx*.42,dy=ry*.12)+gauss(k,max(.38,rx*.27),max(.26,ry*.14),dx=rx*.42,dy=ry*.12))
    brow=gauss(k,max(.35,rx*.36),max(.27,.16*ry),0,-.63*ry)
    mouth=gauss(k,max(.34,rx*.37),max(.27,ry*.14),0,.50*ry)
    nose=gauss(k,max(.30,rx*.21),max(.27,ry*.19),0,.15*ry)
    for pos,neg,basebias,fac in [(cheeks,eyes,.022,1),(brow,eyes,.015,1),(cheeks,mouth,.010,1),(core,top,.025,1),(nose,eyes,.018,1)]:
        register([(1,pos,-fac),(1,neg,fac),(3,pos,-.4),(3,neg,.4)],basebias)
        register([(1,pos,fac),(1,neg,-fac),(3,pos,.4),(3,neg,-.4)],-basebias)
    # cost for fairly smooth neighborhood interiors; very high detail is not
    # characteristic of skin. Do not require an absence of facial features.
    register([(2,core,1)],-.115)
    register([(3,core,-1)],.95)
    h=rel(name+'parts',conv(name+'scan',src,w,bs,p=p))
    # Both branches are in units where a skin-filled face on a completely
    # different background is about one. Photometric bonuses are bounded.
    # An asymmetric hinge rewards modest contrast, not black/white stripes.
    A=[]; B=[]
    for a in [0,1]:
        co=np.zeros(len(w));co[0]=.34 if a==0 else .29;co[1]=.72 if a==0 else .28
        co[2:6]=-np.array([.27,.27,.20,.18])*(1.0 if a==0 else .35)
        negs=np.array([1.3,.78,.62,.32,.48])*(.43 if a==0 or small else 1)
        co[6:16:2]=-negs; co[7:16:2]=negs*(.34 if a==0 else .49)
        co[16]=-1.15;co[17]=-1.0
        # A larger scan supports internal facial contrasts; the small-scale
        # branch principally looks for a head-sized isolated skin island.
        A.append(co);B.append((0 if a==0 else (.0 if small else .25)))
    score=aff(name+'match',h,A,B)
    # Modest photographic placement prior, additive in the matching score.
    size=dims[src][1];x,y=grid(size); x/=size*.5;y/=size*.5
    bias=-.082*x**4-.11*y**4-.065*np.maximum(y,0)**2
    # Negative offset only; spatially uniform constant map from dense(1).
    position=layer(name+'position','reshape',dense(name+'prior',mu,np.zeros((2*size**2,3)),np.tile(bias.ravel(),2)),(2,size,size),shape=[2,size,size])
    score=add(name+'located',score,position)
    return pool(name+'best',score,'max',size,size)

facep=[]
for i,(f,rx,ry,k) in enumerate([(1,.94,1.18,5),(1,1.60,1.93,7),(1,2.34,3.05,11),
                                   (2,1.74,2.2,9),(2,2.75,3.45,13),(2,3.75,4.6,15),
                                   (4,2.65,3.4,11),(4,4.0,5.0,15)]):
    facep.append(faces(factors[f],str(i),rx,ry,k,i<2))

# A pedestrian seen from the front/back commonly has a small cap above a
# wider shoulder line, shirt against two flanks, hips and narrower legs.
# Use separately rectified part contrasts; each contour only has to resemble
# one of the layouts, and neither skin nor a particular clothing color is
# mandatory. Stencils are differences of hand-specified anatomical regions.
bodyp=[]
def body(src,sk,key,squat=1):
    stride=2 if dims[src][1]==48 else 1
    m=dims[src][1]//stride; k=13;p=6; name='body'+key
    W=[];B=[]; arrangements=[]; sw=[]; sb=[]
    def pair(A,C):
        # Ignore luminance changes due just to padding by centering RGB.
        indices=[]
        for color in [lum,rg,gb]:
            t=color[:,None,None]*(A-C);i=len(W);W.extend([t,-t]);B.extend([0,0]);indices.extend([i,i+1])
        return indices
    for q,(width,lean) in enumerate([(1,0),(.79,.33)]):
        # Body axis runs downward; anchor is the shoulder girdle.
        w=width*squat; hx=1.02*w; head=uni(k,hx,1.26,dx=-lean*2,dy=-2.63)
        shirt=uni(k,1.93*w,2.15,0,.15,tilt=lean*.43)
        pants=uni(k,1.16*w,1.62,dx=lean,dy=4.10,tilt=lean*.55)
        parts=[]
        for s in [-1,1]:
            flank=uni(k,.83*w,2.02,dx=s*2.74*w,dy=.2,tilt=lean*.43)
            shs=uni(k,.65*w,1.06,dx=s*1.05*w,dy=-.44)
            sho=uni(k,.70*w,.90,dx=s*1.64*w-lean,dy=-2.56)
            hside=uni(k,.79*w,1.20,dx=s*1.85*w-lean*2,dy=-2.66)
            ps=uni(k,.87*w,1.50,dx=s*2.02*w+lean,dy=4.23,tilt=lean*.55)
            parts += [pair(shirt,flank),pair(shs,sho),pair(head,hside),pair(pants,ps)]
        # background agreement across object; clothing interior agreement;
        # broad high-contrast boundaries alone lack a collar or a head.
        parts.append(pair(uni(k,.93*w,1.85,-2.77*w,.1),uni(k,.93*w,1.85,2.77*w,.1)))
        parts.append(pair(uni(k,.67*w,1.45,-.83*w,-.09),uni(k,.67*w,1.45,.83*w,-.09)))
        parts.append(pair(uni(k,.81*w,1.21,dx=-lean,dy=-2.63),uni(k,1.15*w,1.25,dy=.12)))
        arrangements.append(parts)
        ring=.32*uni(k,hx*.78,1.35,-lean*2-hx*1.88,-2.6)+.32*uni(k,hx*.78,1.35,-lean*2+hx*1.88,-2.6)+.20*uni(k,hx,.72,-lean*2,-4.72)+.16*shirt
        sw.append([(0,head-ring,1),(0,head,.13),(3,head,.44),(3,ring,.36)]);sb.append(-.80)
    # Zero-centered color so artificial borders cannot act as bright edges.
    xc=aff(name+'center',src,np.eye(3),[-.5,-.5,-.5]);h=rel(name+'responses',conv(name+'diffs',xc,W,B,p=p,s=stride))
    # Saturated absolute distances and a shortfall below a desired contour
    # strength. A two-sided match is much better than one huge edge.
    A=[];bs=[]
    for parts in arrangements:
        for ind in parts:
            a=np.zeros(len(W));a[ind]=[1,1,.52,.52,.40,.40]
            A.extend([a,-a]);bs.extend([0,.105])
    dd=rel(name+'dists',aff(name+'dist',h,A,bs))
    A=[]
    pr=np.array([.18,.12,.125,.075]*2)
    for j in range(len(arrangements)):
        a=np.zeros(len(bs));i=j*22
        # Capped evidence via a constant minus missing contrast, with a
        # small residual preference for a well marked two-sided outline.
        a[i:i+16:2]=pr*.72; a[i+1:i+16:2]=-pr*5.7
        a[i+16]=-.43; a[i+18]=-.72; a[i+20]=.19
        A.append(a)
    s1=aff(name+'silhouette',dd,A,.60)
    w=np.zeros((len(sw),4,k,k))
    for j,terms in enumerate(sw):
        for ch,reg,coef in terms:w[j,ch]+=coef*reg
    s2=rel(name+'skinev',conv(name+'head',sk,w,sb,p=p,s=stride))
    sc=add(name+'joint',s1,aff(name+'headweight',s2,.44*np.eye(len(sw))))
    # Suppress boundary hallucinations by requiring support for head, shirt
    # and the patch on either side. Upper body need not have visible legs.
    masks=[]
    for j in range(len(sw)):
        x,y=grid(m); xx=x/(m*.5); yy=y/(m*.5)
        bias=-.055*(xx**4+yy**4)-.03*np.maximum(yy,0)**2
        bias-=.095*np.maximum(0,(abs(x)-m/2+3))/3+.14*np.maximum(0,4-(y+m/2))/4
        masks.append(bias)
    pri=layer(name+'position','reshape',dense(name+'prior',mu,np.zeros((len(sw)*m*m,3)),np.array(masks).ravel()),(len(sw),m,m),shape=[len(sw),m,m])
    z=add(name+'located',sc,pri)
    return pool(name+'max',z,'max',m,m)

# Scale space sampled at 2, 4 and 6 input pixels per feature cell.
for j,(q,sq) in enumerate([(2,.91),(4,.95),(6,1.06)]):
    if q==2:rgb=base;ff=F
    elif q==4:rgb=halve(base);ff=factors[2]
    else:rgb=pool('rgb_six',base,'avg',3,3);ff=pool('feat_six',F,'avg',3,3)
    bodyp.append(body(rgb,ff,str(j),sq))

# An orientation-field silhouette matcher. Rather than correlate with a
# drawing's colors, compare unsigned edge orientations to the *normals* of
# its silhouette. Normalize each pose stencil by its own boundary only.
# Dividing the outline into parts avoids a single vertical bar dominating.
contourp=[]
def segment_dist(x,y,a,b):
    vx=b[0]-a[0];vy=b[1]-a[1];t=np.clip(((x-a[0])*vx+(y-a[1])*vy)/(vx*vx+vy*vy),0,1)
    return np.sqrt((x-a[0]-t*vx)**2+(y-a[1]-t*vy)**2)
def silhouette(lk,bend,width):
    x,y=grid(lk,6);x=x/width
    head=(np.sqrt((x/.84)**2+((y+4.30)/1.08)**2)-1)*.9
    neck=segment_dist(x,y,(0,-3.5),(0,-2.5))-.42
    torso=(np.sqrt((x/1.40)**2+((y+.98)/2.13)**2)-1)*1.45
    d=np.minimum(np.minimum(head,neck),torso)
    for side in [-1,1]:
        sh=(side*1.08,-2.22);el=(side*(1.82+bend*.25),-.28-bend*.57);hand=(side*(1.78+bend*.90),1.39-bend*1.1)
        hip=(side*.66,.61);knee=(side*(.66+bend*.82),2.99);ankle=(side*(.70+bend*1.45),5.39)
        for a,b,r in [(sh,el,.43),(el,hand,.33),(hip,knee,.64),(knee,ankle,.44)]:
            d=np.minimum(d,segment_dist(x,y,a,b)-r)
    # Orientation from the gradient of the implicit outline; match the
    # nearest pair of derivative axes with a soft orientation assignment.
    dy,dx=np.gradient(d);ang=np.arctan2(dy,dx)
    bins=np.array([0,np.pi/2,np.pi/4,-np.pi/4]);basis=np.maximum(0,np.cos(2*(ang[None]-bins[:,None,None])))**3
    basis/=np.maximum(1.e-9,basis.sum(0))
    shell=np.exp(-.5*(d/.40)**2)
    W=[]
    for mask in [y<-3.16,(y>=-3.16)&(y<-1.52), (y>=-1.52)&(y<1.28)&(x<0), (y>=-1.52)&(y<1.28)&(x>=0),y>=1.28]:
        band=unit(shell*mask); weighted=band[None]*(1.75*basis-.25)
        # Sidebands of the outline: penalize oriented energy nearby but
        # off the boundary. Compare to neighbors of the predicted contour.
        surround=unit(np.exp(-.5*((np.abs(d)-1.30)/.48)**2)*mask)
        weighted-=.19*surround[None]
        W.append(weighted.reshape(4,lk,6,lk,6).sum((2,4)))
    return W
for j,(oe,st) in enumerate([(ori,2),(pool('ori4',ori,'avg',2,2),1),(pool('ori6',ori,'avg',3,3),1)]):
    k=15; profiles=[silhouette(k,bd,wd) for bd,wd in [(.25,.87),(.92,1.13)]]
    resp=conv('outline_scan'+str(j),oe,np.concatenate(profiles),s=st,p=k//2)
    A=[];B=[]
    for i in range(10):
        a=np.eye(10)[i];A.extend([a,-a]);B.extend([0,.036])
    sr=rel('outline_clipped'+str(j),aff('outline_test'+str(j),resp,A,B))
    A=np.zeros((2,20)); coeff=np.array([.24,.23,.15,.15,.23]);
    for i in range(2):
        A[i,10*i:10*(i+1):2]=coeff*1.8;A[i,10*i+1:10*(i+1):2]=-coeff*4.4
    joint=aff('outline_parts'+str(j),sr,A,.036*4.4)
    # Axis evidence alone is common on fixtures. Confirm that the model's
    # head position is warmer than the two flanks and the clothing below.
    src=[F,factors[2],'feat_six'][j];gate=np.zeros((1,4,k,k))
    hc=uni(k,1.10,1.28,0,-4.20);fs=.34*(uni(k,1.02,1.35,-2.21,-4.18)+uni(k,1.02,1.35,2.21,-4.18))+.32*uni(k,1.36,1.62,0,-.81)
    gate[0,0]=hc-.78*fs;gate[0,3]=.55*hc+.45*fs
    ge=rel('outline_support'+str(j),conv('outline_head'+str(j),src,gate,-1,p=k//2,s=st));boost=aff('outline_confirmation'+str(j),ge,[[.23],[.23]],-.11)
    joint=add('supported_outline'+str(j),joint,boost)
    m=dims[joint][1];contourp.append(pool('best_outline'+str(j),joint,'max',m,m))

# Extra face branch: use signed internal features with a two-sided test at
# the eyes. Forehead, cheeks and nose bridge should be lighter, and the
# mouth should sit below. No skin color is necessary in this branch.
eyepeaks=[]
for j,(f,rx,ry) in enumerate([(1,2.6,3.5),(2,2.45,3.35),(2,3.75,4.8),(4,3.0,4.0)]):
    src=factors[f];k=2*math.ceil(1.44*ry)+1; filters=[];bs=[]
    def blob(x,y,wx,wy):return gauss(k,max(.30,wx*rx),max(.26,wy*ry),x*rx,y*ry)
    le=blob(-.44,-.19,.22,.13);re=le[:,::-1];nose=blob(0,-.01,.15,.24)
    mouth=blob(0,.53,.32,.10);chin=blob(0,.77,.36,.14)
    pairs=[]
    for side,eye in [(-1,le),(1,re)]:
        pairs.extend([(blob(.44*side,.11,.22,.13),eye), (blob(.37*side,-.50,.22,.17),eye), (nose,eye)])
    pairs.extend([(blob(0,.26,.32,.14),mouth),(chin,mouth)])
    co=uni(k,.88*rx,.87*ry);rr=uni(k,rx*1.46,ry*1.33,l=.81)
    hair=blob(0,-1.04,.55,.17)
    for pos,neg in pairs+[(co,hair)]:
        mat=np.zeros((4,k,k));mat[1]=pos-neg;mat[3]=-.5*mat[1];filters.extend([mat,-mat]);bs.extend([0,.045])
    mat=np.zeros((4,k,k));mat[0]=co-.77*rr;mat[3]=.57*co+.43*rr;filters.append(mat);bs.append(-1.)
    mat=np.zeros((4,k,k));mat[3]=-co;filters.append(mat);bs.append(.97)
    fe=rel('eye_parts'+str(j),conv('eye_scan'+str(j),src,filters,bs,p=k//2))
    A=np.zeros(len(bs));pc=np.array([.17,.135,.08,.17,.135,.08,.095,.085,.05]);
    A[:18:2]=pc*.80;A[1:18:2]=-pc*4.5;A[18]=.33;A[19]=-.44
    obj=aff('face_geometry'+str(j),fe,[A],.045*4.5-.07)
    m=dims[obj][1];eyepeaks.append(pool('face_geometry_best'+str(j),obj,'max',m,m))

# Nonparametric color isolation. Faces and hands can be locally warm even
# under a color cast where their absolute RGB falls outside the gamut.
# In red-vs-cyan and pink-vs-yellow axes, compare each ellipse separately
# to its surroundings, and veto monotonically increasing color ramps.
# Also ask whether sparse skin down and to one side could belong to hands.
humanpeaks=[]
for j,(f,rx,ry) in enumerate([(1,1.10,1.40),(1,1.94,2.5),(2,2.05,2.6),(2,3.05,3.8)]):
    src=factors[f]; rgb=base if f==1 else base+'_half'; k=2*math.ceil(2.25*ry)+1; st=2 if f==1 else 1
    name='relative'+str(j); core=uni(k,.90*rx,.83*ry); mats=[]; b0=[]
    for dx,dy,ax,ay in [(-1.38,-.05,.52,.84),(1.38,-.05,.52,.84),(0,-1.27,.89,.46),(0,1.27,.89,.46)]:
        s=uni(k,ax*rx,ay*ry,dx*rx,dy*ry);delta=core-s
        for axis,gain in [(rg,1.),(.80*rg-.72*gb,.76)]:
            A=axis[:,None,None]*delta*gain;mats.extend([A,-A]);b0.extend([0,.045])
    h=rel(name+'hinge',conv(name+'gamutcontrast',rgb,mats,b0,s=st,p=k//2))
    p=np.repeat([.28,.28,.20,.24],2)*np.tile([.66,.34],4)
    a=np.zeros(len(mats));a[::2]=p*1.6;a[1::2]=-p*6.4
    color=aff(name+'blob',h,[a],.045*6.4)
    # Credibility rises if the candidate lies in the skin gamut and is not
    # an exceptionally saturated color. All constants in opponent units.
    mat=np.zeros((4,k,k));mat[0]=core*.32;mat[3]=core*.25
    sk=conv(name+'gamut',src,[mat],-.25,s=st,p=k//2)
    clr=np.zeros((4,3,k,k));bias=[]
    for i,(axis,threshold) in enumerate([(rg,.28),(gb,.21),(-gb,.047),(-rg,-.025)]):
        clr[i]=axis[:,None,None]*core;bias.append(-threshold)
    excess=rel(name+'penalties',conv(name+'objectcolor',rgb,clr,bias,s=st,p=k//2));pen=aff(name+'penalty',excess,[[-3.1,-1.9,-2.9,-1.5]])
    score=add(name+'isolated',color,sk,pen);m=dims[score][1]
    humanpeaks.append(pool(name+'max',score,'max',m,m))

# Look for two *separated* patches, a face and an extremity, with a torso
# sized gap in between. This is a truncated-part model: the displacements
# are angular sectors, not a particular assumed pose. Using a neighborhood
# peak permits an arm to end anywhere in the sector. A scene filled with
# wood won't produce positive blob responses in the first place.
companions=[]
for j,(f,rad) in enumerate([(1,1.25),(2,1.39),(2,2.05),(4,1.72)]):
    src=factors[f];k=2*math.ceil(2.25*rad)+1; r=k//2
    c=uni(k,rad,1.16*rad); s=uni(k,rad*2.06,rad*2.24,l=.68)
    mat=np.zeros((4,k,k));mat[0]=c-.87*s;mat[3]=.48*c+.36*s
    blob=rel('companion_blob'+str(j),conv('companion_filter'+str(j),src,[mat],-.84,p=r))
    reach=2*math.ceil(4.4*rad)+1;X,Y=grid(reach)
    # Shift and maxpool grid before scoring the pair: allows modest sweeps
    # in both limb length and arm orientation. Max requires no dilation bias.
    z=pool('companion_flex'+str(j),blob,'max',3,1);sel=np.zeros((9,1,reach,reach));mx=reach//2
    sel[0,0,mx,mx]=1
    for i,(dx,dy) in enumerate([(-2.35,2.7),(2.35,2.7),(-1.90,3.95),(1.90,3.95),(-2.85,.85),(2.85,.85),(-3.7,2.2),(3.7,2.2)]):
        sel[i+1,0,mx+round(dy*rad),mx+round(dx*rad)]=1
    shifted=conv('skin_part_offsets'+str(j),z,sel,p=mx)
    # min(head, hand) = head - ReLU(head - hand). Keep some evidence for
    # the head since a hand is typically smaller and less well resolved.
    A=np.zeros((9,9));A[0,0]=1
    for i in range(8):A[i+1,0]=1;A[i+1,i+1]=-1
    penalties=rel('skin_part_mismatch'+str(j),aff('skin_part_test'+str(j),shifted,A));A=np.zeros((8,9));A[:,0]=1;A[:,1:]=-.68*np.eye(8)
    pairs=aff('skin_part_joint'+str(j),penalties,A);m=dims[pairs][1]; companions.append(pool('skin_pair_peak'+str(j),pairs,'max',m,m))

# Scene context. Degree to which edges follow a Manhattan-world layout,
# and where in the photograph those edges and saturated colors occur.
# Components are energy in excess of other directions, not all edges.
az=rel('angular_selectivity',aff('angle_tests',ori,[[1,-.34,-.34,-.34],[-.34,1,-.34,-.34],[-.34,-.34,1,-.34],[-.34,-.34,-.34,1]]))
coarse=pool('contour_layout',az,'avg',8,8)
X,Y=grid(6);mid=unit(np.exp(-.5*((X/1.35)**2+(Y/1.65)**2)));outer=unit(1-np.exp(-.5*((X/1.6)**2+(Y/1.8)**2)))
# Central texture/evidence benefits from peripheral negative context. A
# small preference for diagonal/curved over perfectly axial contours.
E=np.stack([(mid-outer)*1.05-.48/36,(mid-outer)*1.0-.43/36,(mid-outer)*.65+.40/36,(mid-outer)*.65+.40/36]);dim_vote('layout_vote',coarse,E)
# Estimate long straight edges: continuous horizontal/vertical boundaries
# in rooms, furniture and vehicles, unlike the shorter boundaries of limbs.
# Densely packed architecture should count against a flimsy local match.
te=pool('direction_cells',az,'avg',2,2);linef=np.zeros((4,4,7,7));
for i in range(4):
    for t in range(7):
        yy,xx=[(t,3),(3,t),(t,6-t),(t,t)][i];linef[i,i,yy,xx]=1/7
line=rel('long_contours',conv('straight_contours',te,linef,-.018,p=3));lineavg=pool('line_density',line,'avg',24,24)
dim_vote('lines_vote',lineavg,[-.9,-.85,.40,.40],.012)
# Bright, strongly colored clothing/accessories against scene background.
# A red/orange object alone is not enough to trigger the decision.
colors=np.array([[1,-.70,-.30],[.55,-1,.45],[-.57,-.43,1],[.46,.54,-1],[-.5,1,-.5]])
hi=rel('color_excess',aff('color_planes',base,colors,[-.19,-.11,-.16,-.23,-.15]));hc=pool('color_composition',hi,'avg',8,8)
v=np.stack([w*(mid-.42*outer) for w in [.32,.60,.40,-.04,.15]]);dim_vote('color_vote',hc,v)
# Skin color is much less informative when lots of the surrounding scene
# has that hue. Use a center-vs-periphery vote, not global warm-pixel count.
sp=pool('skin_composition',skin,'avg',8,8);dim_vote('skin_context',sp,.16*mid-.23*outer)

# Look for a compact, upright foreground interruption in a comparatively
# simple field. Examples of the general photographic situation include a
# player against turf or a rider in snow. Compare local edge density, not
# a specific background color. Two flanks must both be less busy.
figs=[]
for j,(oe,tsx,tsy) in enumerate([(texture,2.0,3.9),(halve(texture),2.0,4.1),(pool('texture_six',texture,'avg',3,3),2.25,4.2)]):
    k=17;center=uni(k,tsx,tsy);shapes=[uni(k,tsx*.81,tsy*1.1,i*tsx*2.2,0) for i in [-1,1]]
    bank=[center-side for side in shapes];bank.extend([center, .5*(shapes[0]+shapes[1])]);s=2 if j==0 else 1
    q=conv('foreground_profile'+str(j),oe,np.array(bank)[:,None],p=k//2,s=s)
    # Need textured figure and clear flanks on both sides. Use the lesser
    # of the two contrasts, penalizing complex surround in either case.
    e=rel('foreground_test'+str(j),aff('foreground_features'+str(j),q,[[1,0,0,0],[1,-1,0,0],[0,0,-1,0],[0,0,0,1]],[0,0,.085,-.060]))
    p=aff('foreground_match'+str(j),e,[[1,-.88,-.90,-.62]])
    m=dims[p][1];x,y=grid(m);x/=m/2;y/=m/2; prior=-.085*x**4-.14*y**4
    bp=layer('field_placement'+str(j),'reshape',dense('field_prior'+str(j),mu,np.zeros((m*m,3)),prior.ravel()),(1,m,m),shape=[1,m,m]);p=add('field_localized'+str(j),p,bp)
    figs.append(pool('foreground_best'+str(j),p,'max',m,m))

# A nearly horizontal color field is typical of an outdoor playing area,
# water, or snow. Look for smooth flanks matching *each other* at several
# heights, a different colored figure between them, and a change on going
# from its head to its torso. Symmetrically placed windows repeated across
# a building don't satisfy the different-from-flanks test.
salpeaks=[]
for j,(src,sk) in enumerate([(base,F),(base+'_half',factors[2]),('rgb_six','feat_six')]):
    k=15;m0=dims[src][1];st=2 if j==0 else 1;z=[];groups=[];name='subject'+str(j)
    def cd(a,b):
        g=[]
        for v in [lum,rg,gb]:
            t=v[:,None,None]*(a-b);i=len(z);z.extend([t,-t]);g.extend([i,i+1])
        groups.append(g)
    # On several rows compare the background on either side and compare the
    # object with that background. Use coarse means for pose tolerance.
    for ry,y,wx in [(1.32,-3.4,1.09),(1.7,-.1,1.72),(1.40,3.2,1.14)]:
        cc=uni(k,wx,ry,0,y);ll=uni(k,1.40,ry,-3.90,y);rr=uni(k,1.4,ry,3.90,y)
        cd(cc,(ll+rr)*.5);cd(ll,rr)
    ct=uni(k,1.7,1.75,0,-.15);cd(ct,uni(k,1.15,1.34,0,-3.42));cd(ct,uni(k,1.12,1.42,0,3.55))
    h=rel(name+'abs',conv(name+'contrasts',src,z,p=k//2,s=st));A=np.zeros((len(groups),len(z)))
    for i,g in enumerate(groups):A[i,g]=[1,1,.55,.55,.49,.49]
    d=aff(name+'distances',h,A);A=np.zeros((10,8));bb=np.zeros(10)
    for i in range(3):
        A[i,2*i]=1;A[i,2*i+1]=-1.13
        A[i+3]=-A[i];bb[i+3]=.14
    A[6,6]=1;A[7,7]=1;A[8,1:6:2]=1/3;A[9,1:6:2]=-1/3;bb[9]=.13
    h=rel(name+'gate',aff(name+'compare',d,A,bb));s1=aff(name+'figure',h,[[.13,.19,.12,-.43,-.73,-.38,.10,.08,-.38,.45]],.14)
    # A busy flank is unlikely to be a playing field even if its mean
    # happens to match the other flank. Attach heads as a modest bonus.
    w=np.zeros((1,4,k,k));cc=uni(k,1.03,1.4,0,-3.41);fs=.5*(uni(k,1.4,2.7,-3.9,0)+uni(k,1.4,2.7,3.9,0));w[0,0]=.14*(cc-fs);w[0,2]=-.86*fs;w[0,3]=.31*fs+.19*cc
    extra=conv(name+'support',sk,w,-.50,p=k//2,s=st);a=add(name+'evidence',s1,extra);m=dims[a][1]
    salpeaks.append(pool(name+'peak',a,'max',m,m))

# Pool across sizes, avoiding dependence on any single spatial cell. Each
# map maximum stays in an explicit channel until scale pooling.
def stack(name,sources):
    C=sum(math.prod(dims[s]) for s in sources);i=0;zs=[]
    for j,s in enumerate(sources):
        c=math.prod(dims[s]);a=np.zeros((C,c));a[i:i+c]=np.eye(c);i+=c
        zs.append(dense(name+'embed'+str(j),s,a))
    return add(name+'values',*zs)
facevec=stack('face',facep);bvec=stack('posture',bodyp)
# reshape multi-channel maxima so that scale is a pooling dimension.
fh=layer('all_scales','reshape',facevec,(1,len(facep),2),shape=[1,len(facep),2]);fh=layer('face_maxima','maxpool',fh,(1,1,2),size=[len(facep),1],stride=[1,1])
bh=layer('all_poses','reshape',bvec,(1,1,len(bodyp)*2),shape=[1,1,len(bodyp)*2]);bh=layer('pose_maximum','maxpool',bh,(1,1,1),size=[1,len(bodyp)*2],stride=[1,1])
# Treat face evidence as the primary observation; pose helps particularly
# for distant/bundled-up subjects without much exposed skin.
face_margin=dense('face_margin',fh,[[.67,.33]],-.67*.68-.33*.57)
pose_margin=dense('pose_margin',bh,[[1]],-.78)
# An isolated strong anatomical cue can be enough, e.g. a backlit figure
# with no visible face. Convex combination of mean evidence and best cue.
margin=stack('alternative',[face_margin,pose_margin]);m2=layer('margin_map','reshape',margin,(1,1,2),shape=[1,1,2])
best=layer('best_explanation','maxpool',m2,(1,1,1),size=[1,2],stride=[1,1])
dim_vote('either_vote',best,[.33]);dim_vote('joint_vote',margin,[.56,.27])
# Broad peaks in scale space are more believable than accidents at a
# single size. Low-weight consensus as opposed to another detection gate.
dim_vote('face_scale_consensus',facevec,np.tile([.105,.045],len(facep))/len(facep),-.105*.61-.045*.57)
dim_vote('body_scale_consensus',bvec,np.ones(6)*.085/6,-.085*.65)
def summarize_parts(name,parts,center,a,b):
    s=stack(name,parts);n=dims[s][0];r=layer(name+'map','reshape',s,(1,1,n),shape=[1,1,n]);m=layer(name+'peak','maxpool',r,(1,1,1),size=[1,n],stride=[1,1])
    dim_vote(name+'detection',m,[a],-center*a);dim_vote(name+'agreement',s,np.full(n,b/n),-b*(center-.043))
summarize_parts('portrait',eyepeaks,.37,.17,.04)
summarize_parts('oriented_body',contourp,.255,.22,.050)
summarize_parts('chromatic_head',humanpeaks,.56,.24,.055)
summarize_parts('associated_skin',companions,.60,.22,.035)
summarize_parts('field_figure',figs,.157,.64,.23)
summarize_parts('color_figure',salpeaks,.31,.43,.14)
score=add('evidence_total',*votes);dense('class_scores',score,[[0],[1]])
(outdir/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':'))+'\n')
np.savez(outdir/'weights.npz',**weights)
