"""Hand-built person detector. No input files, exemplars or fitted parameters."""
import json, sys, math
from pathlib import Path
import numpy as np

outdir=Path(sys.argv[1]); outdir.mkdir(parents=True,exist_ok=True)
L=[]; W={}; sizes={'image':96}; chans={'image':3}

def layer(n,op,src,side,channels,**kw):
    src=[src] if isinstance(src,str) else src
    L.append(dict(name=n,op=op,inputs=src,**kw)); sizes[n]=side; chans[n]=channels
    return n

def conv(n,src,f,b=0,s=1,p=0):
    f=np.asarray(f,dtype='f'); o,i,k,_=f.shape
    assert chans[src]==i
    W[n+'.W']=f; W[n+'.b']=np.broadcast_to(np.array(b,dtype='f'),(o,)).copy()
    return layer(n,'conv',src,(sizes[src]+2*p-k)//s+1,o,out=o,kernel=[k,k],stride=s,padding=p)

def mix(n,src,rows,b=0):
    return conv(n,src,np.asarray(rows)[:,:,None,None],b)

def rel(n,src): return layer(n,'relu',src,sizes[src],chans[src])
def add(n,*src):
    assert len({(sizes[t],chans[t]) for t in src})==1
    return layer(n,'add',list(src),sizes[src[0]],chans[src[0]])
def pool(n,src,k,s=None,mode='avg'):
    if s is None: s=k
    return layer(n,mode+'pool',src,(sizes[src]-k)//s+1,chans[src],size=[k,k],stride=[s,s])
def dense(n,src,rows,b=0):
    rows=np.array(rows,dtype='f'); o=rows.shape[0]
    W[n+'.W']=rows.reshape(o,-1); W[n+'.b']=np.broadcast_to(np.array(b,dtype='f'),(o,)).copy()
    return layer(n,'dense',src,1,o,out=o)

def grid(k):
    t=np.arange(k)-(k-1)/2
    return np.meshgrid(t,t)

def gauss(x,y,sx,sy,ux=0,uy=0):
    v=np.exp(-.5*((x-ux)/sx)**2-.5*((y-uy)/sy)**2)
    return v/(v.sum()+1e-30)

def disk(x,y,rx,ry,ux=0,uy=0):
    r=np.sqrt(((x-ux)/rx)**2+((y-uy)/ry)**2)
    d=np.clip(.5+(1-r)*min(rx,ry),0,1)
    return d/(d.sum()+1e-15)

def gaussian_conv(n,src,k,sigma,s=1,p=None):
    x,y=grid(k); g=gauss(x,y,sigma,sigma); c=chans[src]
    return conv(n,src,np.eye(c)[:,:,None,None]*g,0,s,k//2 if p is None else p)

# Summary heads are global maxima with a mild photographic framing prior.
summaries=[]
def peak(n,src,edge=.10,portrait=False):
    # Work in blocks both to save memory and to allow edge subjects at a small cost.
    side=sizes[src]; k=next((i for i in (6,5,4,3,2) if side%i==0 and side//i>=3),1)
    a=pool(n+'_tiles',src,k,mode='max') if k!=1 else src
    s=sizes[a]; c=chans[a]
    x,y=grid(s); x=x*2/s; y=y*2/s
    pen=edge*(.80*x*x+.65*(y+(.32 if portrait else .12))**2)
    # The last reduction pools all spatial positions AND variant channels.
    v=layer(n+'_flat','reshape',a,1,c*s*s,shape=[c*s*s,1,1]); m=c*s*s
    # Add spatial offsets with a zero projection from a single constant scalar.
    # This avoids a large identity matrix at the finer resolutions.
    z=mix(n+'_prior',constant,np.zeros((m,1)),np.tile(-pen.ravel(),c))
    v=add(n+'_biased',v,z)
    v=layer(n+'_column','reshape',v,1,1,shape=[1,m,1])
    v=layer(n+'_max','maxpool',v,1,1,size=[m,1],stride=[1,1])
    return v

small=pool('antialias','image',2)
meanRGB=pool('meanRGB',pool('rgb_8',small,6),8)
constant=mix('one',meanRGB,[[0,0,0]],1.)
# Symmetric, bounded partial grey-world adjustment. Keep some absolute colour.
rg=np.array([1.,-1,0]); gb=np.array([0.,1,-1]); white=np.array([.28,.56,.16]); I=np.eye(3)
raw=mix('global_cast',meanRGB,[rg,gb,-rg,-gb],[-.025,-.025,-.025,-.025])
g=rel('cast_excess',raw)
co=np.array([[-1,0,1,0],[0,0,0,0],[0,1,0,-1.]])*.48
corr=mix('small_correction',g,co)
# Broadcast the three offsets to the image using a rank-three dense layer.
s=sizes[small]; corr=dense('apply_cast',corr,np.repeat(I,s*s,axis=0))
corr=layer('cast_map','reshape',corr,s,3,shape=[3,s,s])
rgb=add('adapted',small,corr)

# Soft convex chromaticity regions. Test red excess relative to luminance,
# and blue-yellow position relative to red excess. Two overlapping cones
# deal with warm/pink illumination without treating all brown as flesh.
rr=[]; bb=[]; ends=[]
def halfspace(row,b,slope): rr.append(row*slope);bb.append(b*slope)
for kind in ['usual','warm','cool']:
    if kind=='usual':
        low=.092; hi=.76; ylo=-.31; yhi=1.20; off=.005; strength=1.
    elif kind=='warm':
        low=.135; hi=.87; ylo=.25; yhi=1.70; off=.006; strength=.80
    else:
        low=.048; hi=.50; ylo=-.67; yhi=.70; off=.006; strength=.75
    halfspace(low*white-rg,off,40)
    halfspace(rg-hi*white,-.013,18)
    halfspace(ylo*rg-gb,-.011,33)
    halfspace(gb-yhi*rg,-.014,30)
    halfspace(-white,.090,14)
    ends.append(len(rr))
h=rel('chroma_violations',mix('chroma_planes',rgb,rr,bb))
cc=np.zeros((3,len(rr)))
for i in range(3):cc[i,i*5:i*5+5]=-1
h=rel('chroma_support',mix('colour_cones',h,cc,[1.,.83,.75]))
skin=mix('tissue',h,[[.66,.17,.17]])
gray=mix('intensity',small,[white]);
# Face-colour and luminance planes to process on the same lattice.
two=add('tissue_and_luma',mix('first_plane',skin,[[1],[0]]),mix('second_plane',gray,[[0],[1]]))

# Signed and unsigned locally normalized gradients. Contrast normalization is
# bounded so nearly flat regions do not turn into very strong edges.
x,y=grid(3); g=np.exp(-(x*x+y*y)/1.45); thetas=np.arange(4)*np.pi/4
f=[]
for t in thetas:
    d=(np.cos(t)*x+np.sin(t)*y)*g; d/=np.maximum(d,0).sum(); f.extend([d,-d])
der=rel('oriented_halfwaves',conv('compass',gray,np.array(f)[:,None],p=1))
A=np.repeat(np.eye(4),2,axis=1)
e=mix('unsigned_edges',der,A)
# Piecewise linear compression of the gradient magnitude: approximate a
# saturating contrast response; no division by small edge magnitudes.
e2=rel('strong_edges',mix('contrast_knee',e,np.eye(4),-.15))
e=add('compressed_edges',e,mix('soft_limit',e2,-.80*np.eye(4)))
e=gaussian_conv('edge_spread',e,3,.68)
# Subtract orientation-independent edge energy. This directly rewards the
# orientation of a contour rather than just the amount of clutter there.
e=mix('oriented_contrast',e,np.eye(4)-.78*np.ones((4,4))/4)

# Sparse spatially independent diagnostic evidence for architecture: long,
# straight, axis-aligned edges are common in rooms, signs and buildings.
line_heads=[]
for j,fac in enumerate([1,2]):
    src=der if fac==1 else pool('coarse_edge_directions',der,2)
    k=9; filt=[]
    for channel in [0,1,4,5]:
        v=np.zeros((8,k,k)); horiz=channel>=4
        line=np.zeros((k,k)); line[:,k//2]=1./k
        # Require the edge not to be one of many parallel textured bands.
        line[:,[1,-2]]-=.14/k
        v[channel]=line.T if horiz else line
        v[channel^1]=-.60*(np.maximum(v[channel],0))
        filt.append(v)
    m=conv('straight_runs'+str(j),src,filt,-.075,2,0)
    m=rel('aligned_energy'+str(j),m)
    m=mix('sum_straights'+str(j),m,[np.ones(4)])
    m=pool('mean_runs'+str(j),m,sizes[m]);line_heads.append(m)

# Facial constellation masks. At the smallest octave use simple top/side
# surround tests; at larger octaves also resolve brows, nose and mouth.
# Coordinates below are ratios of a hypothetical face's radii, not pixels
# read from photographs. All contrasts pool a positive and negative region.
face_peaks=[]
for octv,(sub,shrink,face_radii) in enumerate([
        (1,1,[(1.18,1.65),(1.85,2.48),(2.7,3.0)]),
        (2,1,[(1.75,2.6),(2.6,3.35),(3.4,3.85)]),
        (4,1,[(2.0,2.7),(3.1,3.65),(3.9,4.8)]),
        (6,1,[(2.2,3.0),(3.5,4.25)])]):
    prefix='face'+str(octv)+'_'
    src=two if sub==1 else pool(prefix+'down',two,sub)
    # Pad with zero evidence. Deduct missing-context bonus near frame later.
    kernels=[]; B=[]; groups=[]; k=11 if octv>0 else 9; x,y=grid(k)
    for rx,ry in face_radii:
        members=[]
        def part(skinmask=None,lum=None,b=0):
            z=np.zeros((2,k,k));
            if skinmask is not None:z[0]=skinmask
            if lum is not None:z[1]=lum
            kernels.append(z);B.append(b);members.append(len(B)-1)
        inside=disk(x,y,rx*.86,ry*.83)
        outerx=.42*rx; outery=.59*ry
        l=gauss(x,y,outerx,outery,-rx*1.58,0);r=gauss(x,y,outerx,outery,rx*1.58,0)
        top=gauss(x,y,rx*.73,ry*.38,0,-ry*1.47)
        bottom=gauss(x,y,rx*.88,ry*.45,0,ry*1.79)
        crown=gauss(x,y,.71*rx,.35*ry,0,-ry*1.02)
        cheeks=(gauss(x,y,.28*rx,.27*ry,-.47*rx,.14*ry)+gauss(x,y,.28*rx,.27*ry,.47*rx,.14*ry))/2
        forehead=gauss(x,y,.44*rx,.22*ry,0,-.58*ry)
        eyes=(gauss(x,y,.23*rx,max(.31,.17*ry),-.44*rx,-.24*ry)+gauss(x,y,.23*rx,max(.31,.17*ry),.44*rx,-.24*ry))/2
        mouth=gauss(x,y,.37*rx,max(.32,.13*ry),0,.46*ry)
        nose=gauss(x,y,.23*rx,.24*ry,0,.03*ry)
        # Representation stores positive contributions (or penalties) with
        # separate limits, so a single giant edge can't simulate a face.
        part(inside)
        part(.65*(l+r)/2+.35*top)
        part(bottom)
        part(lum=(.60*inside+.40*cheeks)-crown,b=-.012)
        part(lum=(.48*forehead+.39*cheeks+.13*nose)-eyes,b=-.008)
        part(lum=(.6*cheeks+.4*nose)-mouth,b=-.008)
        part(lum=inside-.5*(l+r),b=-.02)
        part(lum=.5*(l+r)-inside,b=-.02)
        groups.append(members)
    a=rel(prefix+'response',conv(prefix+'parts',src,kernels,B,s=2 if sub==1 else 1,p=k//2))
    tr=np.tile([1.,1.,1.,.22,.125,.105,.23,.23],len(groups))
    b=rel(prefix+'past_cap',mix(prefix+'caps',a,np.eye(len(B)),-tr))
    a=add(prefix+'bounded',a,mix(prefix+'excess',b,-np.eye(len(B))))
    rows=[]; biases=[]
    for members in groups:
        for coef,bias in [([1.17,-.65,-.17,.85,1.40,.65,.31,.28],-.02),
                          ([.73,-.36,-.09,.88,2.6,1.6,.49,.37],-.08)]:
            row=np.zeros(len(B));row[members]=coef;rows.append(row);biases.append(bias)
    m=mix(prefix+'models',a,rows,biases)
    face_peaks.append(peak(prefix+'best',m,.14,True))

# HOG-like body-outline templates made from articulated capsules on a
# canonical coordinate grid. Each pose mask is used individually. No
# quantities are estimated from collections of templates. The positive
# template is a blurred normal field on the figure boundary; negative
# samples alongside it penalize clutter and an outline in the wrong place.

def figure(k,height,aspect,pose,phase,kind):
    # normalized stature: top = 0, feet = 1. Centre template at mid-height.
    d=4
    tt=(np.arange(k*d)+.5)/d-k/2
    xx,yy=np.meshgrid(tt,tt); X=xx/(height*aspect); Y=yy/height+.5
    def ellipse(cx,cy,rx,ry):return (1-np.sqrt(((X-cx)/rx)**2+((Y-cy)/ry)**2))*min(rx,ry)
    def capsule(a,b,r1,r2):
        dx=b[0]-a[0];dy=b[1]-a[1]
        u=np.clip(((X-a[0])*dx+(Y-a[1])*dy)/(dx*dx+dy*dy),0,1)
        return r1+(r2-r1)*u-np.hypot(X-a[0]-u*dx,Y-a[1]-u*dy)
    tilt=0.; hips=(0,.48); sh=(0,.215); head=(0,.083)
    knee=[(-.065,.73),(.065,.73)]; ankle=[(-.068,.966),(.071,.966)]
    elbow=[(-.15,.365),(.15,.365)]; hand=[(-.144,.505),(.14,.505)]
    if pose=='walking':
        head=(.035,.09); sh=(.02,.22);knee=[(-.12,.72),(.15,.70)];ankle=[(-.235,.95),(.132,.96)]
        elbow=[(-.155,.33),(.135,.34)];hand=[(-.128,.47),(.237,.30)]
    elif pose=='sport':
        head=(.095,.10);sh=(.068,.235);knee=[(-.15,.67),(.175,.66)];ankle=[(-.31,.85),(.21,.94)]
        elbow=[(-.11,.275),(.225,.30)];hand=[(-.25,.21),(.31,.20)]
    elif pose=='seated':
        head=(0,.094);sh=(0,.24);hips=(-.026,.54);knee=[(-.12,.64),(.18,.61)];ankle=[(-.16,.92),(.16,.89)]
        elbow=[(-.16,.42),(.15,.43)];hand=[(-.06,.51),(.047,.49)]
    elif pose=='upper':
        # Bust template. Stature here represents the visible upper body.
        head=(0,.185);sh=(0,.50);hips=(0,1.08)
        elbow=[(-.35,.93),(.35,.93)];hand=[(-.34,1.22),(.34,1.22)]
    scale=2.05 if pose=='upper' else 1.
    h=ellipse(*head,.061*scale,.080*scale)
    neck=capsule((head[0],head[1]+.05*scale),sh,.040*scale,.046*scale)
    dist=np.maximum(h,neck);torso=capsule(sh,hips,.112*scale,.098*scale)
    # Compress the round top of the torso into the flat shoulder line.
    torso=np.minimum(torso,Y-sh[1]+.040*scale);dist=np.maximum(dist,torso)
    for idx,sign in enumerate([-1,1]):
        shoulder=(sh[0]+sign*.104*scale,sh[1]+.023*scale)
        dist=np.maximum(dist,capsule(shoulder,elbow[idx],.042*scale,.033*scale));dist=np.maximum(dist,capsule(elbow[idx],hand[idx],.035*scale,.026*scale))
        if pose!='upper':
            hip=(hips[0]+.054*sign,hips[1]);dist=np.maximum(dist,capsule(hip,knee[idx],.057,.042));dist=np.maximum(dist,capsule(knee[idx],ankle[idx],.042,.027))
    mask=np.clip(.5+dist*height*d*.9,0,1)
    if phase:mask=mask[:,::-1]
    gy,gx=np.gradient(mask)
    # For a truncated upper-body view, suppress any edges near the bottom.
    if pose=='upper':
        taper=np.clip((1.-Y)/.22,0,1);gx*=taper;gy*=taper
    ss=np.hypot(gx,gy); orient=np.arctan2(gy,gx); F=[]
    for t in thetas:
        v=ss*np.cos(t-orient)**8
        F.append(v.reshape(k,d,k,d).sum((1,3)))
    F=np.array(F);F/=max(F.sum(),1e-12)
    # suppress general texture with a local surrounding contour-energy cost
    sur=np.pad(F,((0,0),(2,2),(2,2)));bg=sum(sur[:,2+i:2+i+k,2+j:2+j+k] for i,j in [(-2,0),(2,0),(0,-2),(0,2)])*.25
    # Colour masks for a second, region-based reading of this same pose.
    # Compare the subject against the left AND the right, not just against
    # a single ring (which would fire at the edge of every large object).
    Xm=-X if phase else X
    band=np.clip(.5+dist*height*d,0,1)
    # Distance units are fractions of stature. The masks leave a margin
    # at the contour, where the reduced-resolution image mixes colours.
    inner=band*np.clip(.65+dist*height*.7,0,1)
    near=np.exp(-.5*((dist+.11*scale)/(.059*scale))**2)*(1-band)
    def unit(a):
        a=a.reshape(k,d,k,d).mean((1,3));return a/(a.sum()+1e-12)
    def section(cy,dy,cx):
        vertical=np.exp(-.5*((Y-cy)/dy)**2)
        left=unit(vertical*near*(Xm<cx));right=unit(vertical*near*(Xm>=cx))
        centre=unit(vertical*inner)
        return centre,left,right
    sections=[section(sh[1]+.08*scale,.087*scale,sh[0]),
              section(hips[1]+(.16 if pose!='upper' else -.30),.092*scale,hips[0])]
    return F-.40*bg,sections

# A companion head-position feature is combined before maximizing. It
# rewards an isolated warm spot above the torso but isn't required for a
# strong achromatic silhouette. Central context checks favour similar
# colours to either side, e.g. a person against grass, snow or a wall.
body_peaks=[]; region_peaks=[]
for octave,(sub,k,hgt) in enumerate([(1,9,6.7),(2,9,6.6),(3,9,6.9),(4,11,8.3),(6,11,8.6)]):
    p='body'+str(octave)+'_'
    o=e if sub==1 else pool(p+'edge_grid',e,sub)
    # e has two-pixel cells. For the first octave, use a larger stride in
    # the feature grids instead: a 2.5% box usually spans many pixels.
    if sub==1:
        o=pool(p+'edge_half',e,2);sub=2;hgt=4.55;k=7
    cm=pool(p+'colour_grid',rgb,sub)
    sm=pool(p+'warm_grid',skin,sub)
    x,y=grid(k); fl=[]; hues=[]; groups=[]; poses=[]; outfit=[]
    for pose in ('upright','walking','sport','seated','upper'):
        for aspect in ([.80,1.40] if pose in ('upright','upper','walking') else [1.02]):
            for mirror in ([False,True] if pose in ('sport','walking','seated') else [False]):
                ff,reg=figure(k,hgt,aspect,pose,mirror,None)
                fl.append(ff);outfit.append(reg);poses.append(pose)
                srcx= -.095 if mirror else .095
                hx=(srcx if pose=='sport' else srcx*.37 if pose=='walking' else 0)*hgt*aspect
                upper=pose=='upper';hy=(-.32 if upper else -.41)*hgt
                rx=max(.37,hgt*aspect*(.118 if upper else .068));ry=max(.42,hgt*(.158 if upper else .084))
                hf=gauss(x,y,rx*.8,ry*.72,hx,hy)
                lf=gauss(x,y,max(.5,rx*.7),max(.45,ry*.87),hx-rx*2.4,hy)
                rf=gauss(x,y,max(.5,rx*.7),max(.45,ry*.87),hx+rx*2.4,hy)
                cf=gauss(x,y,rx*1.25,ry*.91,hx,hy+2.7*ry)
                hues.append(hf-.46*(lf+rf)/2-.25*cf)
                # centre-surround contrast at shirt and pants; left/right
                # background disparity discounts simple single step edges.
                rad=(.26 if upper else .13)*hgt*aspect
                ty=(.14 if upper else -.15)*hgt
                center=gauss(x,y,max(.42,.55*rad),hgt*.11,0,ty)
                left=gauss(x,y,max(.40,rad*.52),hgt*.13,-1.68*rad,ty)
                right=gauss(x,y,max(.40,rad*.52),hgt*.13,1.68*rad,ty)
                groups.append((center,left,right))
    step=1
    shape=conv(p+'silhouettes',o,fl,0,step,k//2)
    hue=conv(p+'head_spots',sm,np.array(hues)[:,None],0,step,k//2)
    # Reuse chest masks among mirror pairs and among poses of like build.
    fields=[]; bgmap={}; ids=[]
    for c,l,r in groups:
        key=(c.tobytes(),l.tobytes(),r.tobytes())
        if key not in bgmap:
            bgmap[key]=len(fields)
            for mask in [c-(l+r)/2,(l-r)]:
                for basis in [white,rg,.5*(rg+gb)]:
                    for sign in [1,-1]:fields.append(sign*basis[:,None,None]*mask)
        ids.append(bgmap[key])
    col=rel(p+'colour_steps',conv(p+'torso_surround',cm,fields,p=k//2))
    rows=np.zeros((len(fl),len(fields)))
    for i,j in enumerate(ids):
        q=np.array([1,1,.48,.48,.34,.34]);rows[i,j:j+6]=q;rows[i,j+6:j+12]=-.60*q
    pop=rel(p+'two_sided_contrast',mix(p+'contrast',col,rows))
    # Bound the size of each kind of evidence: a lone black stripe against
    # white should not win just by being of very high contrast.
    n=len(fl); comp=[]
    for inp,gain,cap,suffix in [(shape,12.,.053,'contour'),(hue,.58,.68,'colour'),(pop,1.18,.39,'contrast')]:
        # Negative orientation matches and head-surround differences are
        # informative too; use an upper, rather than two-sided, clip.
        excess=rel(p+suffix+'_past',mix(p+suffix+'_knee',inp,np.eye(n),-cap))
        comp.extend([mix(p+suffix+'_vote',inp,np.eye(n)*gain),mix(p+suffix+'_deduct',excess,-np.eye(n)*gain)])
    bodymap=add(p+'match',*comp)
    body_peaks.append(peak(p+'best',bodymap,.20))
    if octave>=1:
        # Pool shirt and trouser contrasts independently. Use the identity
        # |a-(b+c)/2|-|b-c|/2: for scalar colours this gives distance to
        # the nearer background if both contrasts have the same sign.
        f=[]
        for pair in outfit:
            for center,left,right in pair:
                for axis in [white,rg+.78*gb]:
                    for mask in [center-.5*(left+right),.5*(left-right)]:
                        f.extend([axis[:,None,None]*mask,-axis[:,None,None]*mask])
        a=rel(p+'region_halfwaves',conv(p+'region_masks',cm,f,s=1,p=k//2))
        coef=np.array([1,1,-1,-1,.62,.62,-.62,-.62])
        mat=np.kron(np.eye(2*n),coef[None,:])
        a=rel(p+'separate_garments',mix(p+'bg_discount',a,mat))
        overflow=rel(p+'garment_excess',mix(p+'garment_knee',a,np.eye(2*n),-.285))
        mat2=np.kron(np.eye(n),[[2.00,1.62]])
        m=add(p+'dressed_figure',mix(p+'garments',a,mat2),mix(p+'cap_garments',overflow,-mat2),
              mix(p+'coloured_head',hue,.39*np.eye(n)),mix(p+'boundaries',shape,6.5*np.eye(n)))
        region_peaks.append(peak(p+'best_outfit',m,.23))

# Close-up portraits: combine positive evidence for each of the eyes, the
# mouth, the brows and the facial surround, not just their summed contrast.
# Bilateral constraints prevent a single high-contrast corner scoring well.
portrait_peaks=[]
for j,sub in enumerate([2,4,6]):
    src=pool('portrait_grid'+str(j),two,sub); k=11;x,y=grid(k)
    kernels=[]; groups=[]
    for rx,ry in [(2.25,3.05),(3.45,4.25),(2.42,4.65)]:
        gp=[]
        for tilt in [0,-.18,.18]:
            u=x*np.cos(tilt)+y*np.sin(tilt);v=-x*np.sin(tilt)+y*np.cos(tilt)
            def patch(cx,cy,rx1,ry1):return gauss(u,v,max(.33,rx1*rx),max(.31,ry1*ry),cx*rx,cy*ry)
            gm=[]
            def fw(lm=None,sm=None):
                a=np.zeros((2,k,k));
                if lm is not None:a[1]=lm
                if sm is not None:a[0]=sm
                kernels.append(a);gm.append(len(kernels)-1)
            for sign in [-1,1]:
                brow=patch(sign*.43,-.25,.23,.13)
                cheek=patch(sign*.46,.15,.27,.20);temple=patch(sign*.49,-.59,.28,.20);bridge=patch(0,-.16,.15,.25)
                fw(.40*temple+.44*cheek+.16*bridge-brow)
            fw(.54*patch(0,.05,.26,.22)+.46*patch(0,.73,.34,.18)-patch(0,.42,.39,.12))
            fw(patch(0,.06,.57,.60)-patch(0,-1,.68,.28))
            fw(sm=disk(u,v,rx*.88,ry*.87)-.26*(patch(-1.47,0,.42,.53)+patch(1.47,0,.42,.53))-.20*patch(0,-1.4,.66,.35))
            groups.append(gm)
    pref='portrait'+str(j)+'_'
    a=conv(pref+'parts',src,kernels,p=k//2)
    # Winsorize eye, mouth and hair responses independently. Negative part
    # responses also count; dark spots in random clutter rarely agree.
    cap=np.tile([.16,.16,.13,.21,.74],len(groups))
    m=rel(pref+'overflow',mix(pref+'cap',a,np.eye(len(kernels)),-cap))
    rows=np.zeros((len(groups),len(kernels)))
    for i,idx in enumerate(groups):rows[i,idx]=[1.90,1.90,1.78,.88,.54]
    pm=add(pref+'combined',mix(pref+'vote',a,rows),mix(pref+'remove',m,-rows))
    portrait_peaks.append(peak(pref+'best',pm,.18,True))

# Area of warm material by itself is ambiguous (wood, bread, sand, fur).
# Give only a modest vote to the excess in the upper centre over the rest
# of the picture, and to the contrast between warm blobs and a surround.
sk=pool('colour_regions',skin,4);side=sizes[sk];x,y=grid(side)
inside=gauss(x,y,side*.25,side*.29,0,-side*.13); flat=np.ones_like(x)/x.size
warmstat=dense('warm_location',sk,[(inside-.88*flat).ravel()])
# National sports fields, beaches, ski slopes and pools are common settings
# for people: the useful property is a coherent, relatively smooth ground
# with isolated contrasting subjects, not merely green or blue pixels.
# Coherent ground is only a weak prior.
field=pool('ground_grid',rgb,4)
rgbs=[[-1,1,0],[0,1,-1],[-1,0,1],[0,-1,1]]
fld=rel('outdoor_ground',mix('ground_colours',field,rgbs,[.008,-.02,-.035,-.02]))
x,y=grid(sizes[fld]);ground=gauss(x,y,4.5,2.7,0,3.2)
fieldstat=dense('ground_prior',fld,[(np.array([.38,.15,.20,.12])[:,None,None]*ground).ravel()])

# Rule-table readout. A strong detection at any scale can win. Use a small
# additional vote for repeatability across nearby scales, and permit one
# especially convincing route to compensate for absence of another cue.
def across(n,ls):
    # Pack independent scalar branches and pool them along a spatial axis.
    T=len(ls); eye=np.eye(T)
    ins=[mix(n+'_entry'+str(i),v,eye[:,i,None]) for i,v in enumerate(ls)]
    a=add(n+'_scales',*ins);a=layer(n+'_list','reshape',a,1,1,shape=[1,T,1])
    mx=layer(n+'_strongest','maxpool',a,1,1,size=[T,1],stride=[1,1]);av=layer(n+'_agreement','avgpool',a,1,1,size=[T,1],stride=[1,1])
    return mx,av
fm,fa=across('face',face_peaks);bm,ba=across('body',body_peaks);pm,pa=across('portrait',portrait_peaks)
# All reference levels below are hand set on the scales defined above.
# Face votes around unity describe a light oval, surrounded by different
# material; body votes need both outline and centre-surround support.
rm,ra=across('outfit',region_peaks)
sc=[]
def vote(n,src,g,b=0):
    a=dense(n,src,[[g]],[b]);sc.append(a);return a
vote('face_decision',fm,1.78,-1.78*.93);vote('repeated_face',fa,.68,-.68*.62)
vote('body_decision',bm,1.14,-1.14*.77);vote('repeated_body',ba,.45,-.45*.57)
vote('outfit_decision',rm,.95,-.95*.96);vote('repeated_outfit',ra,.31,-.31*.71)
vote('portrait_decision',pm,.95,-.95*.83);vote('repeated_portrait',pa,.30,-.30*.57)
vote('warm_decision',warmstat,1.44);vote('ground_decision',fieldstat,.85)
for i,m in enumerate(line_heads):vote('architectural_penalty'+str(i),m,-3.9,.050)
a=add('person_evidence',*sc);dense('scores',a,[[0.],[1.]])
(outdir/'network.json').write_text(json.dumps({'layers':L},separators=(',',':'))+'\n')
np.savez(outdir/'weights.npz',**W)
