"""Construct a photo detector from color inequalities and geometric part models."""
import sys
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

D = Path(sys.argv[1]); D.mkdir(parents=True, exist_ok=True)
net = []; weights = {}; dims = {'image': (3,96,96)}

def node(tag, op, inputs, shape, **settings):
    name = tag+'_'+str(len(net)); net.append(dict(name=name,op=op,inputs=inputs,**settings)); dims[name]=shape; return name

def conv(x, w, b=0, s=1, p=None, tag='filter'):
    w=np.asarray(w,dtype='f'); w=w[:,:,None,None] if w.ndim==2 else w
    o,i,k,_=w.shape; p=k//2 if p is None else p
    c,h,v=dims[x]; assert i==c
    name=node(tag,'conv',[x],(o,(h+2*p-k)//s+1,(v+2*p-k)//s+1),out=o,kernel=[k,k],stride=s,padding=p)
    weights[name+'.W']=w; weights[name+'.b']=np.broadcast_to(b,(o,)).astype('f'); return name

def dense(x,w,b=0,tag='linear'):
    w=np.array(w,dtype='f'); o,i=w.shape; assert i==np.prod(dims[x]); n=node(tag,'dense',[x],(o,),out=o); weights[n+'.W']=w; weights[n+'.b']=np.broadcast_to(b,(o,)).astype('f');return n

def relu(x):return node('positive','relu',[x],dims[x])
def add(*x):
    assert all(dims[t]==dims[x[0]] for t in x); return node('sum','add',list(x),dims[x[0]])
def reshape(x,sh):
    name='axes_'+str(len(net));net.append(dict(name=name,op='reshape',inputs=[x],shape=list(sh)));assert np.prod(sh)==np.prod(dims[x]);dims[name]=tuple(sh);return name

def pool(x, a, stride=None, op='avg'):
    a=(a,a) if isinstance(a,int) else tuple(a); stride=a if stride is None else (stride,stride) if isinstance(stride,int) else stride
    c,h,w=dims[x];return node(op,op+'pool',[x],(c,(h-a[0])//stride[0]+1,(w-a[1])//stride[1]+1),size=list(a),stride=list(stride))

def channels_max(x):
    c,h,w=dims[x];return reshape(pool(reshape(x,[1,c,h*w]),(c,1),op='max'),[1,h,w]) if c>1 else x

def cat(xs):
    # Embed each part into its own coordinates before summing.
    if len(xs)==1:return xs[0]
    total=sum(dims[x][0] for x in xs); start=0; branches=[]
    for x in xs:
        c=dims[x][0]; branches.append(conv(x,np.eye(total,c,-start),tag='pack')); start+=c
    return add(*branches)

def unit(x):
    # 1 - relu(1-relu(x)), i.e. bound each cue independently.
    I=np.eye(dims[x][0]);return conv(relu(conv(relu(x),-I,1,tag='ceiling')),-I,1,tag='bounded')

def project(x,coefs,tag='readout'):
    return conv(x,np.array(coefs,ndmin=2),tag=tag)

def grid(k):return np.mgrid[:k,:k]-(k//2)
def gauss(y,x,cy,cx,sy,sx):return np.exp(-.5*(((y-cy)/sy)**2+((x-cx)/sx)**2))
def normal(a):return a/max(1.e-12,np.sum(a))
def expandmap(x, scalar):
    # Spatial log prior, independent of image values. The low-resolution grid
    # indexes a candidate's position, not an object's position in any photo.
    h,w=dims[x][1:]; y,z=np.mgrid[:h,:w]; xx=(z+.5)/w; yy=(y+.5)/h
    prior= .045-.085*((xx-.5)/.5)**4-.16*((yy-.40)/.60)**4
    return add(x,reshape(dense(scalar,np.zeros((w*h,1)),prior.ravel(),tag='location'),(1,h,w)))

rgb='image'; mono=conv(rgb,[[.299,.587,.114]],tag='lightness')
small=pool(rgb,2); slow=pool(small,2)
# Soft skin locus. These half-spaces implement ranges for R-G and G-B,
# including dim, lightly pigmented, and pinkish appearances. Penalties grow
# continuously, rather than snapping pixels into a binary lookup table.
R,G,B=np.eye(3); rg=R-G; gb=G-B
planes=[(.11*G-rg,.029,17),(.08*G-rg-.25*gb,.024,8),
        (gb-1.17*rg,-.028,16),(-gb-.38*rg,-.018,18),
        (rg-.50*G,-.090,10),(gb-.38*G,-.050,8),(-R,.14,8)]
skin=relu(conv(relu(conv(small,[v*t for v,b,t in planes],[b*t for v,b,t in planes],tag='skin_chroma')),
               -np.ones((1,len(planes))),1,tag='skin_membership'))
scalar=pool(skin,48)
skinp={2:skin}; rgbp={1:rgb,2:small,4:slow}; lum={1:mono}
for d in (4,8,16): skinp[d]=pool(skinp[d//2],2)
for d in (8,16):rgbp[d]=pool(rgbp[d//2],2)
for d in (2,4,8,16):lum[d]=pool(lum[d//2],2)

# Two-dimensional matched eye/eyebrow, nose, forehead and mouth constellations.
# Distances below are fractions of a generic face width and height. A score
# must be supported by several distinct contrasts, not one intense edge.

def detail(d):
    # Patches of increasing absolute size. Downsample only after correlating
    # the finer image so that blurred eyes 1-2 pixels wide are still visible.
    s=2 if d<=4 else 1; k=13; ys,xs=grid(k); W=[]; V=[]; a=[]; biases=[]
    profiles=[(4.8,7.3,0),(7.2,8.6,0),(8.,7.,0),(6.4,8.3,.20),(6.4,8.3,-.20)]
    for fw,fh,tilt in profiles:
        x=(xs*np.cos(tilt)+ys*np.sin(tilt))/fw; y=(ys*np.cos(tilt)-xs*np.sin(tilt))/fh
        def g(cx,cy,sx,sy):return normal(gauss(y,x,cy,cx,sy,sx))
        eyes=[g(l*.22,-.13,.097,.061) for l in (-1,1)]
        cheeks=[g(l*.225,.09,.11,.09) for l in (-1,1)]
        brows=[g(l*.20,-.32,.12,.07) for l in (-1,1)]
        bridge=g(0,-.07,.074,.11); nose=g(0,.085,.095,.085); chin=g(0,.35,.15,.075); mouth=g(0,.225,.17,.055)
        hair=g(0,-.56,.29,.11); e=np.mean(eyes,axis=0); c=np.mean(cheeks,axis=0); f=np.mean(brows,axis=0)
        # Independent bounded votes at each location, including signed
        # penalties for strongly wrong features and unlike left/right cheeks.
        feat=[(cheeks[0]-eyes[0],.125,.04,.135),(cheeks[1]-eyes[1],.125,.04,.135),
              (brows[0]-eyes[0],.12,.02,.10),(brows[1]-eyes[1],.12,.02,.10),
              (bridge-e,.11,.02,.10),(.60*nose+.4*chin-mouth,.105,.03,.095),
              (.65*c+.35*f-hair,.20,.02,.085),
              (cheeks[0]-cheeks[1],.18,-.30,-.065),(cheeks[1]-cheeks[0],.18,-.30,-.065),
              (eyes[0]-eyes[1],.20,-.40,-.04),(eyes[1]-eyes[0],.20,-.40,-.04)]
        for v,sc,b,wt in feat:W.append(v/sc);biases.append(b)
        a.append([it[3] for it in feat]);
        # Both eyes should be genuine indentations, with lighter skin both
        # above and below. A sharp horizontal step is not an eye band.
        V.extend([cheeks[0]-eyes[0],brows[0]-eyes[0],cheeks[1]-eyes[1],brows[1]-eyes[1],
                  bridge-eyes[0],bridge-eyes[1],nose-mouth,chin-mouth])
    light=conv(lum[d],np.asarray(W)[:,None],biases,s=s,tag='face_contrasts')
    # ReLU difference implements sum_i w_i clip(x_i,0,1) using two
    # branches; combine before accumulating evidence for the other scales.
    def votes(x,ww):
        return project(x,ww)
    n=len(profiles); m=len(a[0]); M=np.zeros((n,n*m))
    for j in range(n):M[j,j*m:(j+1)*m]=a[j]
    texture=add(votes(relu(light),M),votes(relu(conv(light,np.eye(n*m),-1,tag='saturation')),-M))
    # First take pairwise minima, then the minimum across the two eyes.
    # Keep these comparisons local to each pose; evidence from two different
    # hypothesized rotations must not supply the two sides of the same eye.
    def pairmins(x,n):
        I=np.eye(n); A=np.kron(I,[[1,0]]);T=np.kron(I,[[1,-1]])
        return add(project(x,A),project(relu(project(x,T)),-I))
    pairs=pairmins(conv(lum[d],np.asarray(V)[:,None],s=s,tag='feature_indentations'),4*n)
    pair_weights=np.zeros((n,4*n)); eye_weights=np.zeros((2*n,4*n));
    for j in range(n):
        pair_weights[j,4*j+2:4*j+4]=[.31,.23];eye_weights[2*j:2*j+2,4*j:4*j+2]=np.eye(2)
    valleys=add(project(unit(conv(pairs,9*np.eye(4*n),-.045,tag='darker_features')),pair_weights),
                project(unit(conv(pairmins(project(pairs,eye_weights),n),9*np.eye(n),-.025,tag='paired_eyes')),.46*np.eye(n)))
    texture=add(project(texture,.80*np.eye(n)),project(valleys,.33*np.eye(n)))
    # Coarser skin plane has aligned receptive fields to within half a pixel.
    sd=max(d,2); ss=s if d>=2 else 1
    k2=9 if d==1 else 13; yy,xx=grid(k2); C=[]
    for fw,fh,tilt in profiles:
        x=(xx*np.cos(tilt)+yy*np.sin(tilt)+(.24 if d==1 else 0))/(fw*d/sd); y=(yy*np.cos(tilt)-xx*np.sin(tilt)+(.24 if d==1 else 0))/(fh*d/sd)
        face=normal(gauss(y,x,.02,0,.22,.29)); sides=normal(gauss(y,x,-.05,-.65,.32,.16)+gauss(y,x,-.05,.65,.32,.16));cap=normal(gauss(y,x,-.60,0,.14,.29))
        C.append(.43*face-.09*sides-.07*cap)
    pigment=conv(skinp[sd],np.asarray(C)[:,None],s=ss,tag='face_skin_support')
    combo=add(texture,pigment)
    # Suppress truncated constellations with missing eyes etc. Zero padding
    # must not fabricate a dark eye at an image edge.
    n,h,w=dims[combo]; yy,xx=np.mgrid[:h,:w]; margin=np.minimum(np.minimum(xx,w-1-xx)*s,np.minimum(yy,h-1-yy)*s)
    border=np.maximum(0,1-margin/3.5)*.56
    combo=add(channels_max(combo),reshape(dense(scalar,np.zeros((h*w,1)),-border.ravel(),tag='face_extent'),(1,h,w)))
    return expandmap(combo,scalar)

# Oriented contour channels. Subtracting a fraction of the orthogonal
# energy suppresses edges that are not aligned with the hypothesized outline.
@lru_cache(None)
def oriented(d):
    y,x=grid(3); g=gauss(y,x,0,0,.76,.76); fx=x*g; fx/=fx[fx>0].sum(); fy=fx.T
    angles=np.arange(4)*np.pi/4; F=np.array([np.cos(t)*fx+np.sin(t)*fy for t in angles]);
    v=relu(conv(lum[d],np.r_[F,-F][:,None],tag='oriented_derivatives')); eye=np.eye(4)
    M=eye-.65*np.roll(eye,2,axis=0); return unit(conv(v,6*np.c_[M,M],-.045,tag='edge_selectivity'))

# Smooth silhouette contour masks, with independently scored left/right
# sides. Positive mass follows a drawing of a head and clothed upper body;
# negative mass samples an adjacent strip and subtracts general edge clutter.
def silhouette_masks(k,r,t):
    y,x=grid(k); accum=[]
    for side in (-1,1):
        # half-width of head = r, t is vertical length unit; origin at face.
        pts=[(.04,-1.38),(.65,-1.25),(.96,-.82),(1.,-.28),(.82,.26),(.55,.66),
             (.57,.86),(1.16,1.01),(1.95,1.25),(2.23,1.62),(2.51,2.62),(2.60,3.35)]
        parts=[]
        for ia,ib in [(0,5),(5,len(pts)-1)]:
            mask=np.zeros((4,k,k)); base=np.zeros_like(mask)
            for i in range(ia,ib):
                xa,ya=pts[i]; xb,yb=pts[i+1]; xa*=r*side; xb*=r*side; ya*=t; yb*=t
                dx=xb-xa;dy=yb-ya;ll=np.hypot(dx,dy); nx,ny=dy/ll,-dx/ll
                ang=np.arctan2(ny,nx); ori=np.maximum(0,np.cos(2*(ang-np.arange(4)*np.pi/4)))**2; ori/=ori.sum()
                for u in np.linspace(.0625,.9375,8):
                    cx=xa+u*dx; cy=ya+u*dy; w=ll/8
                    mask+=w*ori[:,None,None]*normal(gauss(y,x,cy,cx,.60,.60))
                    for off in [-1.65,1.65]:base+=.5*w*ori[:,None,None]*normal(gauss(y,x,cy+ny*off,cx+nx*off,.77,.77))
            parts.append(normal(mask)-.64*normal(base))
        accum+=parts
    return accum

def body(d):
    # Each hypothesis is anchored on a face, whether or not looking at camera.
    # Small projected figures need not have resolvable eyes. Look for: warm
    # oval; distinct cap; head/neck discontinuities; contrast and coherent
    # clothing below; oriented silhouette instead of a horizontal strip.
    k=11; y,x=grid(k); stride=2 if d==2 else 1; shp=[(.80,1.02),(1.28,1.27),(1.59,1.62)]
    skin_filters=[]; light_filters=[]; rgb_filters=[]; contours=[]
    for r,t in shp:
        def g(xx,yy,sx,sy):return normal(gauss(y,x,yy*t,xx*r,sy*t,sx*r))
        fa=g(0,0,.56,.64); cap=g(0,-1.38,.66,.37)
        sides=(g(-1.58,-.11,.44,.72)+g(1.58,-.11,.44,.72))/2
        chest=g(0,2.67,1.26,.69); bg=(g(-2.98,2.43,.47,.72)+g(2.98,2.43,.47,.72))/2
        upper=g(0,-2.28,1.,.43); low=g(0,.30,.54,.48); upperface=g(0,-.42,.53,.40)
        skin_filters.extend([fa,fa-sides,fa-cap,fa-chest,fa-upper]);light_filters.extend([low-cap,fa-sides,chest-bg,fa-chest,low-upperface,sides-upper]);
        # Opponent-color contrast detects colored shirts even at equal luma.
        for spec in [(rg+.45*gb)*1.1,gb+.30*rg]:rgb_filters.append(spec[:,None,None]*(chest-bg))
        contours.extend(silhouette_masks(k,r,t))
    n=len(shp)
    # Enclose the head in distinct material at three different sides. A
    # single straight boundary should not act like a small isolated face.
    ring=[]; lobes=[]
    for r,t in shp:
        def g(xx,yy,sx,sy):return normal(gauss(y,x,yy*t,xx*r,sy*t,sx*r))
        fa=g(0,0,.53,.61); lower=g(0,.22,.55,.44)
        for cx,cy,sx,sy in [(-1.55,-.03,.44,.66),(1.55,-.03,.44,.66),(0,-1.65,.64,.40),(0,2.35,1.05,.63)]:
            ring.append(fa-g(cx,cy,sx,sy))
        for cx,cy,sx,sy in [(-.80,.11,.28,.42),(.80,.11,.28,.42),(0,-.94,.42,.30)]:
            lobes.append(lower-g(cx,cy,sx,sy))
    def make(src,f,tag):
        w=np.asarray(f);return conv(src,w[:,None] if w.ndim==3 else w,s=stride,tag=tag)
    sk=make(skinp[d],skin_filters,'head_and_neck_color'); l=make(lum[d],light_filters,'hair_and_garment'); co=make(rgbp[d],rgb_filters,'garment_chroma'); edge=make(oriented(d),contours,'upper_body_outline')
    def aggregate(x, rules, tag):
        # Each cue is specified by (source, inverse range, bias, vote).
        m=dims[x][0]//n; U=np.zeros((len(rules)*n,m*n)); v=[]; M=np.zeros((n,n*len(rules)))
        for i in range(n):
            for j,(idx,gain,b,wt) in enumerate(rules):
                row=i*len(rules)+j; U[row,i*m+idx]=gain; v.append(b); M[i,row]=wt
        raw=conv(x,U,v,tag=tag); return project(unit(raw),M)
    skins=aggregate(sk,[(0,1.32,-.13,.43),(1,1.45,0,.275),(2,1.50,0,.13),(3,1.35,0,.11),(4,1.35,0,.085)],'localized_skin')
    arrangements=aggregate(l,[(0,5.1,-.03,.14),(0,-5.5,-.12,.035),(1,5.,-.03,.055),(1,-5.,-.03,.055),
                              (2,5,-.03,.075),(2,-5,-.03,.075),(3,5,-.05,.085),(3,-5,-.05,.085),
                              (4,6,-.08,.045)],'part_contrast')
    garb=aggregate(co,[(j,sg*7,-.06,.039) for j in (0,1) for sg in (1,-1)],'clothing_color')
    outlines=aggregate(edge,[(i,3.0,-.13,.077 if i%2==0 else .09) for i in range(4)],'silhouette_vote')
    def soft_min(x, group, slope):
        # Average minus a penalty on each component below the minimum
        # expected support. This tolerates a single partly occluded side.
        mat=np.kron(np.eye(n),np.ones((1,group))/group)
        return add(project(x,mat),conv(relu(conv(x,-np.eye(n*group),.46)), -slope*mat,tag='missing_sides'))
    enc=unit(conv(make(skinp[d],ring,'head_surround_sectors'),1.6*np.eye(n*4),-.07,tag='skin_separation'))
    brown=unit(conv(make(lum[d],lobes,'hair_cap_sectors'),6.5*np.eye(n*3),-.04,tag='hair_over_skin'))
    localized=soft_min(enc,4,1.8); hair=soft_min(brown,3,.9)
    # Reinterpret the old skin vote as a local part rather than letting a
    # single highly saturated area swamp the other structural parts.
    prior=add(skins,arrangements,garb,outlines)
    score=add(project(prior,.66*np.eye(n)),project(localized,.35*np.eye(n)),project(hair,.16*np.eye(n)),conv(outlines,.54*np.eye(n),.045))
    n,h,w=dims[score]; yy,xx=np.mgrid[:h,:w]; b=np.zeros((h,w)); dist=np.minimum(xx,w-1-xx)*stride
    b-=.20*np.maximum(0,1-dist/2.5); b-=.22*np.maximum(0,1-yy*stride/2.5);b-=.18*np.maximum(0,1-(h-1-yy)*stride/3.5)
    score=add(channels_max(score),reshape(dense(scalar,np.zeros((h*w,1)),b.ravel(),tag='head_extent'),(1,h,w)))
    return expandmap(score,scalar)

# A second upright-figure cue does not depend on skin. Compare the figure's
# interior colors to the scene on BOTH sides, at head, shirt, and trouser
# heights. Penalize high internal texture and long lines extending past head.
def figure(d):
    k=15; y,x=grid(k); stride=2 if d==2 else 1
    vals=[]; sk=[]; shapes=[(.86,1.09),(1.19,1.14),(1.45,1.35)];n=len(shapes)
    for w,h in shapes:
        def g(cx,cy,sx,sy):return normal(gauss(y,x,cy*h,cx*w,sy*h,sx*w))
        for cy,inside,outside,sy in [(-3.74,.56,1.67,.68),(-.66,1.02,2.70,1.),(2.88,.91,2.26,1.24)]:
            for s in [-1,1]:vals.append(g(s*.18,cy,inside,sy)-g(s*outside,cy,.55,sy*1.10))
        vals+= [g(0,-3.74,.56,.64)-g(0,-5.53,.70,.50),g(0,-.63,1.0,1.)-g(0,2.88,.91,1.),
                g(-2.6,-.70,.54,.94)-g(2.6,-.70,.54,.94),g(-.70,-.68,.47,.9)-g(.7,-.68,.47,.9),
                g(-2.4,2.40,.60,1.)-g(2.4,2.4,.60,1.),g(0,-.66,1.02,1.)-g(0,-3.74,.56,.68)]
        face=g(0,-3.74,.58,.68);sk.append( .16*face+.105*(face-(g(-1.74,-3.74,.55,.70)+g(1.74,-3.74,.55,.70))/2) )
    # Color distances: sum absolute luma and chroma differences. Averaging
    # colors before differencing discounts distracting fine texture.
    nch=2; color=conv(rgbp[d],[[.28,.56,.16],[.5,.12,-.62]],tag='figure_color_axes'); v=np.array(vals);K=len(v); W=np.zeros((K*nch,nch,k,k));
    for i in range(nch):W[i::nch,i]=v
    raw=conv(color,W,s=stride,tag='figure_parts'); I=np.eye(K*nch)
    # abs(x) = 2 relu(x)-x
    a=add(project(relu(raw),2*I),project(raw,-I)); A=np.zeros((K,K*nch))
    for j in range(K): A[j,j*nch:j*nch+2]=[4.9,4.4]
    energy=unit(conv(a,A,-.12,tag='part_distance'))
    m=K//n
    # Compute bilateral minimum instead of allowing one side of a step edge
    # to supply both contour votes. min(l,r)=l-relu(l-r).
    T=np.zeros((3*n,K)); M=np.zeros((n,K)); P=np.zeros((n,3*n));
    for i in range(n):
        for j,wt in enumerate([.24,.36,.20]):
            T[i*3+j,i*m+j*2:i*m+j*2+2]=[1,-1];M[i,i*m+j*2]=wt;P[i,i*3+j]=-wt
        M[i,i*m+6:i*m+12]=[.12,.11,-.13,-.115,-.09,.085]
    evidence=add(project(energy,M),project(relu(project(energy,T)),P),conv(skinp[d],np.array(sk)[:,None],s=stride,tag='figure_head'))
    n,h,w=dims[evidence]; yy,xx=np.mgrid[:h,:w]; dist=np.minimum(xx,w-1-xx)*stride
    b=-.49*np.maximum(0,1-dist/3)-.68*np.maximum(0,1-yy*stride/5)-.32*np.maximum(0,1-(h-1-yy)*stride/4)
    centered=add(evidence,reshape(dense(scalar,np.zeros((n*h*w,1)),np.tile(b.ravel(),n),tag='figure_extent'),(n,h,w)))
    # Put the response at the head, not the window center. This lets other
    # parts vote at the same site instead of anywhere in the image.
    move=np.zeros((n,n,11,11))
    for i,(_,t) in enumerate(shapes):move[i,i,5+int(round(3.74*t/stride)),5]=1
    return expandmap(channels_max(conv(centered,move,tag='align_head')),scalar)

# A faceless figure branch. At small scale the face can be just two or
# three pixels; the differing widths of the head, shirt and legs often
# survive. Probe cross-sections on each side separately, looking both for a
# step and for a relatively continuous region on either side of the step.
def dressed(d):
    k=17; y,x=grid(k); s=2 if d==2 else 1
    sizes=[(.78,1.05),(1.08,1.15),(1.48,1.32)]; n=len(sizes); bank=[]; contours=[]; fac=[]
    for w,h in sizes:
        def g(dx,dy,sx,sy):return normal(gauss(y,x,dy*h,dx*w,sy*h,sx*w))
        A=g(0,-4.12,.51,.59); B=g(0,-1.25,.84,.88); C=g(0,1.88,.72,.8)
        for side in [-1,1]:
            for cy,in_x,out_x,out_sd,sy in [(-4.12,.13,1.60,.44,.62),(-1.25,.54,2.62,.49,.92),(1.87,.41,2.16,.48,.92)]:
                bank.append(g(in_x*side,cy,.48,sy)-g(out_x*side,cy,out_sd,sy))
            # The side of the head and outer edge of the shoulder need not
            # differ from the shirt in the same direction of contrast.
            bank.append(g(1.23*side,-1.74,.33,.51)-g(1.23*side,-3.51,.35,.45))
            bank.append(g(.75*side,.25,.52,.44)-g(2.37*side,.25,.39,.48))
        bank.extend([A-g(0,-5.73,.59,.46),B-C,A-B,
                     g(-2.46,-.45,.65,.71)-g(2.46,-.45,.65,.71),
                     g(-.70,-1.25,.40,.80)-g(.70,-1.25,.40,.80),
                     g(0,4.08,.65,.78)-g(0,5.98,.82,.47),
                     g(0,-1.74,.84,.56)-g(0,-.41,.84,.56)])
        # Clothing has oriented contours extending over several rows. A
        # modest slant on either side models non-rigid arms and trousers.
        for side in [-1,1]:
            for cx,cy,sy,tilt,orientation in [(1.64,-1.28,1.01,.13,0),(1.18,2.31,1.48,-.15,0),
                                             (1.05,-2.63,.62,.88,1 if side==-1 else 3)]:
                gx=x-side*(cx*w+tilt*(y-cy*h)); gy=y-cy*h
                a=normal(gauss(gy,gx,0,0,sy*h,.63)); b=normal(gauss(gy,abs(gx),0,1.65,sy*h,.64));
                W=np.zeros((4,k,k));W[orientation]=a-.57*b;contours.append(W)
        fac.append(.13*A+.07*(A-.5*(g(-1.55,-4.12,.49,.67)+g(1.55,-4.12,.49,.67))))
    K=len(bank); m=K//n; v=np.asarray(bank); W=np.zeros((2*K,2,k,k));W[::2,0]=v;W[1::2,1]=v
    raw=conv(conv(rgbp[d],[[.28,.56,.16],[.55,.17,-.72]],tag='cloth_color_axes'),W,s=s,tag='cloth_cross_sections')
    # Nonnegative distance in a two-dimensional photometric space.
    A=np.zeros((K,2*K)); A[np.arange(K),np.arange(K)*2]=4.9;A[np.arange(K),np.arange(K)*2+1]=4.6
    dist=unit(add(conv(relu(raw),2*A,-.09,tag='rectified_distance'),project(raw,-A)))
    M=np.zeros((n,K)); limits=[(0,.16),(1,.21),(2,.19),(3,.095),(4,.095)]
    T=np.zeros((n*len(limits),K)); P=np.zeros((n,n*len(limits)))
    for i in range(n):
        for j,(z,wt) in enumerate(limits):
            off=i*m+z;T[i*5+j,off]=1;T[i*5+j,off+5]=-1;M[i,off]=wt;P[i,i*5+j]=-wt
        M[i,i*m+10:i*m+17]=[.085,.085,.055,-.115,-.15,.040,-.060]
    parts=add(project(dist,M),project(relu(project(dist,T)),P))
    E=conv(oriented(d),np.array(contours),s=s,tag='long_cloth_contours')
    segment=unit(conv(E,3.3*np.eye(6*n),-.18,tag='cloth_edge_support'))
    Q=np.kron(np.eye(n),np.array([[.075,.085,.055,.075,.085,.055]]))
    val=add(parts,project(segment,Q),conv(skinp[d],np.array(fac)[:,None],s=s,tag='optional_exposed_head'))
    _,h,w=dims[val];yy,xx=np.mgrid[:h,:w];b=-.45*np.maximum(0,1-np.minimum(xx,w-1-xx)*s/3)-.65*np.maximum(0,1-yy*s/5.7)-.38*np.maximum(0,1-(h-1-yy)*s/5)
    return expandmap(add(channels_max(val),reshape(dense(scalar,np.zeros((h*w,1)),b.ravel(),tag='clothing_extent'),(1,h,w))),scalar)

def summarize(maps,tag,center):
    # Pool maxima at two extents. Broad or repeated agreement is useful, but
    # a lone, highly distinctive face in a corner should suffice as well.
    blocks=[]
    for x in maps:
        h=dims[x][1]; blocks.append(pool(x,h//6,op='max') if h>6 else x)
    b=channels_max(cat(blocks)); peaks=pool(b,6,op='max')
    quadrant=pool(pool(b,3,op='max'),2)
    mass=pool(relu(conv(b,[[1]],-center,tag=tag)),6)
    return [peaks,quadrant,mass]

face_maps=[detail(d) for d in (1,2,4,8)]
head_maps=[body(d) for d in (2,4,8,16)]
figure_maps=[figure(d) for d in (2,4,8)]

def match_grid(x,h):
    q=dims[x][1]//h
    return x if q==1 else pool(x,q,op='max')

# A part may be a couple of pixels off, but all three must agree about
# the site and approximate scale. Subtract the strongest single vote: a
# featureless skin patch or stripe standing on its own is not a detection.
def agree(x,y,z):
    h=min(dims[m][1] for m in [x,y,z]); raw=cat([match_grid(m,h) for m in [x,y,z]])
    v=conv(raw,np.diag([1.25,.98,1.1]),[-.12,-.085,-.12],tag='same_site_parts')
    strongest=channels_max(v); return add(project(v,[[.37,.43,.29]]),project(strongest,[[-.17]]))
joined=[agree(face_maps[i],head_maps[i],figure_maps[min(i,2)]) for i in range(4)]
faces=summarize(face_maps,'face_clusters',.64)
heads=summarize(head_maps,'head_clusters',.79)
people=summarize(figure_maps,'figure_clusters',.70)
matched=summarize(joined,'local_agreement',.65)
# Absolute skin coverage is unreliable (wood, earth, food). Count localized
# skin in excess of its own neighborhood rather than rewarding a warm cast.
s=skinp[4]; y,x=grid(5); surround=normal(1-np.exp(-.5*(x*x+y*y)/1.3**2)); f=-surround;f[2,2]+=1
spots=pool(relu(conv(s,f[None,None],-.13,tag='warm_islands')),24)
# Scene evidence: bright pigment isolated against a less colorful field is
# common in sport/clothing; long level fixtures and empty earth tones are
# not. These weak votes cannot by themselves trigger a detection.
s2=rgbp[4]; warm=np.array([1,-.54,-.46]); blue=np.array([-.35,-.65,1]);pink=np.array([.52,-1,.48]);
pig=unit(conv(s2,5*np.array([warm,blue,pink]),[-1.05,-.70,-.68],tag='saturated_color'))
y,x=grid(5); center=normal(gauss(y,x,0,0,.78,.78)); ring=normal(1-gauss(y,x,0,0,1.5,1.5)); delta=center-ring
W=np.zeros((3,3,5,5));
for j in range(3):W[j,j]=delta
pops=pool(channels_max(relu(conv(pig,W,-.16,tag='isolated_bright_garment'))),24,op='max')
# Rectilinear structure: doors, wall/cabinet intersections, furniture.
# Compare a line to the local background; stripes or vegetation alone give
# little sustained correctly oriented energy along a long straight path.
edge=oriented(2); k=11; y,x=grid(k); E=np.zeros((2,4,k,k));
for i,(idx,dx,dy) in enumerate([(0,x,y),(2,y,x)]):
    line=normal(np.exp(-.5*(dx/.48)**2)*np.exp(-.5*(dy/3.8)**2)); offset=normal(np.exp(-.5*((abs(dx)-2)/.70)**2)*np.exp(-.5*(dy/3.8)**2)); E[i,idx]=line-.40*offset
rect=pool(relu(conv(edge,E,-.43,s=2,tag='long_architectural_edges')),24)
# Soft contextual vote for water/snow/playing fields. Require both a field
# of the right color and a contrasting island reaching above a structured
# lower half: a solid blue sky alone is not evidence of a person.
C=np.array([[-.4,.72,-.32],[-.75,.24,.51],[-.20,-.68,.88],[.70,.04,-.74]])
ctx=unit(conv(rgbp[8],C*np.array([12,10,11,6])[:,None],[-.42,-.45,-.36,-.50],tag='field_colors'))
low=np.array([.10,.10,.20,.26,.33,.45,.65,.90,1.,1.,1.,.95]);low=low/low.sum(); P=np.zeros((4,4,12,12))
for i in range(4):P[i,i]=np.broadcast_to(low[:,None]/12,(12,12))
fields=unit(reshape(dense(ctx,P.reshape(4,-1)*2.15,[-.32,-.29,-.31,-.36],tag='playing_surfaces'),(4,1,1)))
# A saturated, sharply localized shirt or bathing suit seen against a large
# continuous natural background is more diagnostic than the surface alone.
fields=reshape(fields,(4,1,1)); nat=channels_max(conv(fields,[[1,0,0,0],[0,1,0,0],[0,0,.82,0]],tag='field_kind'))
sports=relu(conv(cat([nat,pops]),[[.92,1.05]],-.85,tag='outdoor_clothing'))
street_clothes=summarize([dressed(d) for d in (2,4,8)],'clothed_people',.68)
allcues=cat(faces+heads+people+[spots]+matched+[pops,rect]+street_clothes+[sports]);
# Each branch gets a weak vote by default and a strong vote only above its
# nominal evidence floor. Maxima, quadrant support and island density are
# deliberately on different scales.
M=np.zeros((6,20)); M[0,:3]=[.80,.27,.32]; M[1,3:6]=[.77,.28,.31];M[2,6:9]=[.74,.31,.26];M[3,10:13]=[.85,.25,.31];M[4,9]=1.35;M[4,13:16]=[.105,-1.25,-1.55]; M[5,16:19]=[.78,.28,.31];M[4,19]=.28
score=conv(allcues,M,[-.86,-1.14,-1.02,-.86,-.042,-1.01],tag='expert_margins')
# Lack of frontal eyes or a visible lower body should not veto a group
# photograph or a frame-filling portrait.
combined=add(conv(score,[[.25,.35,.05,.66,1.,.16]],tag='agreement'),
             conv(relu(score),[[.87,.96,.20,.98,0,.84]],tag='confident_votes'))
logits=dense(combined,[[0],[1]],[0,-.060],tag='class_scores')
(D/'network.json').write_text(json.dumps({'layers':net},separators=(',',':'))+'\n')
np.savez(D/'weights.npz',**weights)
