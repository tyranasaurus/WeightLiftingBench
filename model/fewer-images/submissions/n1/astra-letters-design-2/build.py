"""Build an analytic pose-library network. No input data, external models or assets.

Lengths in the skeleton are relative to the palm width. Positive y is distal,
positive z is out of the palm, and positive x is towards the thumb. The camera
is orthographic. Numerical constants specify anatomy, sign articulation, camera,
illumination, or fixed image filters, not measurements on a dataset.
"""
import json, math, sys
from pathlib import Path
import numpy as np

ABC = 'ABCDEFGHIKLMNOPQRSTUVWXY'
D = np.pi/180

def unit(a):
    return a / max(np.linalg.norm(a), 1.e-12)

def rotations(turn, tip, roll):
    a,b,c=np.array([turn,tip,roll])*D
    ca,cb,cc=np.cos([a,b,c]); sa,sb,sc=np.sin([a,b,c])
    return (np.array([[cc,-sc,0],[sc,cc,0],[0,0,1]]) @
            np.array([[1,0,0],[0,cb,-sb],[0,sb,cb]]) @
            np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]]))

# A pose is four independently articulated three-link chains and a thumb.
# Angles are (MCP flexion, PIP flexion, DIP flexion), in degrees. Lateral
# slopes specify ab/adduction. Fixed presets are anatomical instructions.
straight=(4,3,2)
fist=(78,95,64)
hook=(19,91,65)
curved=(29,39,28)

def skeleton(sign, rng, form):
    digit=[fist]*4               # index through little
    spread=[0.,0.,0.,0.]
    if sign=='B': digit=[straight]*4; spread=[-1,-1,0,1]
    if sign=='C': digit=[(27,39,24),(31,42,22),(33,40,25),(37,43,26)]; spread=[2,0,-1,-3]
    if sign=='D': digit=[straight,(45,64,34),(48,75,39),(55,76,45)]
    if sign=='E': digit=[hook]*4
    if sign=='F': digit=[(53,64,28),straight,straight,straight]; spread=[-3,9,0,-12]
    if sign=='G': digit[0]=straight
    if sign=='H': digit[:2]=[straight,straight]; spread[:2]=[-1,1]
    if sign in 'IY': digit[3]=straight; spread[3]=-15 if sign=='Y' else -10
    if sign in 'KP': digit=[straight,(54,8,3),fist,fist]; spread[:2]=[13,-9]
    if sign=='L': digit[0]=straight; spread[0]=-2
    if sign in 'MN': digit=[(69,83,54) if i<(3 if sign=='M' else 2) else fist for i in range(4)]
    if sign=='O': digit=[(34,57,36),(35,64,36),(38,69,38),(43,73,42)]; spread=[-7,0,3,5]
    if sign=='Q': digit[0]=(6,13,6)
    if sign=='R': digit[:2]=[(3,12,2),(5,4,1)]; spread[:2]=[-17,9]
    if sign=='T': digit[0]=(52,94,52)
    if sign=='U': digit[:2]=[straight,straight]; spread[:2]=[-1,1]
    if sign=='V': digit[:2]=[straight,straight]; spread[:2]=[16,-13]
    if sign=='W': digit[:3]=[straight]*3; spread[:3]=[18,1,-17]
    if sign=='X': digit[0]=(9,63,44); spread[0]=3
    # The carpal end is narrower, and the little-finger knuckle is lower.
    xj=np.array([.303,.097,-.114,-.310]); yj=np.array([.43,.49,.45,.338]); zj=np.array([.015,.005,-.003,.023]);
    palmwidth=rng.uniform(.92,1.1); palmlength=rng.uniform(.91,1.10)
    length=rng.uniform(.92,1.10); thickness=rng.uniform(.94,1.09)*palmwidth
    xj*=palmwidth; yj*=palmlength
    lengths=np.array([[.344,.216,.166],[.382,.245,.177],[.357,.230,.169],[.277,.171,.137]])*length
    widths=np.array([.111,.115,.107,.094])*thickness
    uj=rng.uniform(-1,1)
    # primitives: (start, end, base radius, end radius, type)
    seg=[]
    fingers=[]
    for i in range(4):
        p=np.array([xj[i],yj[i],zj[i]])
        pnts=[p]
        flex=0.; sd=spread[i]+rng.normal(0,2.7)
        dlen=lengths[i]*rng.uniform(.95,1.05); rad=widths[i]*np.array([1.03,.95,.84,.74])
        for j in range(3):
            nominal=digit[i][j]; noise=(9. if nominal>15 else 3.5)
            flex+=nominal + noise*(.50*uj+.50*rng.uniform(-1,1))
            sway=sd*D
            v=np.array([np.cos(flex*D)*np.sin(sway),np.cos(flex*D)*np.cos(sway),np.sin(flex*D)])
            q=p+v*dlen[j]; seg.append((p,q,rad[j],rad[j+1],0)); p=q; pnts.append(p)
        fingers.append(pnts)
    # Thumb opposition uses the broad thenar base and an independently placed
    # two-link distal chain. Targets on the fingers enforce anatomical contacts.
    base=np.array([.205,-.255,.075]); root=np.array([.348,-.093,.126]);
    # The points below are local anatomical landmarks, not image positions.
    if sign in 'ALY':
        if sign=='A': a,b,t=[.407,.00,.145],[.438,.247,.172],[.444,.490,.175]
        else: a,b,t=[.446,-.075,.120],[.670,.013,.130],[.887,.085,.147]
    elif sign=='G':
        a,b,t=[.405,-.007,.164],[.467,.235,.236],[.508,.55,.274]
    elif sign=='Q':
        a,b,t=[.430,-.041,.165],[.570,.142,.235],[.618,.337,.278]
    elif sign in 'CFDO':
        if sign=='C':
            a,b,t=[.446,-.078,.163],[.490,-.025,.437],[.453,.049,.663]
        else:
            tip=np.array(fingers[0 if sign in 'FO' else 1][-1]); t=tip+[.035,-.058,-.011]
            a=np.array([.400,-.010,.158]); b=(a+t)*.5+[.11,-.025,.06]
    elif sign in 'KP':
        a,b,t=[.382,-.004,.167],[.288,.218,.303],[.169,.483,.335]
    elif sign=='B': a,b,t=[.349,-.052,.139],[.204,.100,.246],[-.061,.163,.229]
    elif sign=='E': a,b,t=[.321,-.079,.153],[.136,-.018,.225],[-.121,-.001,.211]
    elif sign in 'MNT':
        n={'T':1,'N':2,'M':3}[sign]; gap=.5*(xj[n-1]+xj[n]);
        a=np.array([.348,-.019,.146]); t=np.array([gap,.424 if sign=='T' else .300,.316]); b=(t+a)*.5+[-.018,.014,-.032]
    elif sign=='W':
        a,b,t=[.321,-.019,.145],[.077,.125,.254],[-.208,.189,.283]
    elif sign=='S': a,b,t=[.338,.028,.224],[.190,.265,.455],[-.051,.305,.447]
    else: a,b,t=[.347,-.018,.154],[.176,.098,.292],[-.076,.157,.327]
    line=np.array([base,root,a,b,t]); line[:,0]*=palmwidth; line[:,1]*=palmlength
    if sign not in 'FDO': line[3:]+=rng.normal(0,.023,(2,3))
    r=np.array([.153,.157,.14,.116,.094])*thickness
    for i in range(4): seg.append((line[i],line[i+1],r[i],r[i+1],1))
    # Projection of a smooth palm cross-section, plus opponens mound, heel,
    # and an elliptical forearm. Render all geometry with the same z-buffer.
    blobs=[]
    def ell(x,y,z,rx,ry,rz,kind=0):
        blobs.append((np.array([x*palmwidth,y*palmlength,z]),np.array([rx*palmwidth,ry*palmlength,rz*thickness]),kind))
    ell(0,.038,0,.454,.529,.142)
    ell(.015,.237,0,.457,.303,.132)
    ell(.232,-.115,.051,.202,.316,.140)
    ell(-.151,-.144,.021,.231,.287,.137)
    return seg,blobs,palmwidth,palmlength

# Principal ways the anatomical axes are presented to a viewer. Rotating the
# wrist in depth (rather than mirroring the image) gives foreshortening,
# occlusion and, for lateral signs, a radial/ulnar profile.
def camera(k,rng,form):
    yaw,pitch,roll={
        'A':(20,8,5),'B':(8,1,-2),'C':(84,7,-15),'D':(38,4,-4),
        'E':(14,7,4),'F':(24,4,1),'G':(74,6,-66),'H':(64,8,-67),
        'I':(11,5,5),'K':(41,1,11),'L':(8,4,1),'M':(21,10,-3),
        'N':(24,10,-3),'O':(76,7,-17),'P':(-71,30,-140),'Q':(72,1,144),
        'R':(18,1,2),'S':(21,12,1),'T':(25,8,-5),'U':(16,1,2),
        'V':(13,1,0),'W':(6,1,1),'X':(44,2,-3),'Y':(8,3,0),
    }[k]
    # Some signs have alternative views, particularly when made sideways,
    # or when the thumb hides behind the hand. Sample the two lateral views
    # of a pointing finger, including views of the dorsum of the hand.
    if k in 'GHPQ':
        if k=='G' and form%4==0: yaw,pitch,roll=(22,-11,-70)
        if k=='H' and form%3!=0: yaw,pitch,roll=(135,-8,-62)
        if k=='P':
            if form%3==0: yaw,pitch,roll=(95,-16,145)
            elif form%3==1: yaw,pitch,roll=(-112,38,122)
            else: yaw,pitch,roll=(-50,59,111)
        if k=='Q':
            if form%3==0: yaw,pitch,roll=(102,7,143)
            elif form%3==1: yaw,pitch,roll=(23,13,137)
            else: yaw,pitch,roll=(66,-12,142)
    if k=='K' and form%2==0: yaw=77
    if k in 'STMN' and form%4==0: yaw+=37
    if k=='X' and form%4==0: yaw=79
    yaw+=rng.normal(0,15 if k not in 'GHPQ' else 20)
    pitch+=rng.normal(0,11)
    roll+=rng.normal(0,8 if k not in 'GHPQ' else 12)
    return rotations(yaw,pitch,roll)

# Rendering resolution before antialiasing. The small fixed raster and the
# local primitive bounding boxes keep the entire build inexpensive.
HH=56
X=np.arange(HH)[None,:]+.5
Y=np.arange(HH)[:,None]+.5

def box(x0,x1,y0,y1):
    i,j=max(0,int(np.floor(x0))),min(HH,int(np.ceil(x1)))
    k,l=max(0,int(np.floor(y0))),min(HH,int(np.ceil(y1)))
    if i>=j or k>=l: return None
    return (slice(k,l),slice(i,j))

def render(sign,rng,form):
    seg,blobs,pw,pl=skeleton(sign,rng,form)
    R=camera(sign,rng,form)
    # Convert camera up into raster down.
    R[1]*=-1
    s0=np.array([s[0] for s in seg]) @ R.T; s1=np.array([s[1] for s in seg]) @ R.T
    r0=np.array([s[2] for s in seg]); r1=np.array([s[3] for s in seg]);
    bc=np.array([b[0] for b in blobs]) @ R.T; radii=np.array([b[1] for b in blobs])
    S=(R[None]*radii[:,None,:]**2) @ R.T
    br=np.sqrt(np.diagonal(S,axis1=1,axis2=2))
    # A photographic crop should include the entire pose and a little of the
    # wrist; this is a per-render camera transform, not a pooled statistic.
    heel=np.array([0,-.68*pl,0]) @ R.T
    lo=np.min(np.vstack([s0-r0[:,None],s1-r1[:,None],bc-br,heel[None]-[.16,.11,0]]),axis=0)
    hi=np.max(np.vstack([s0+r0[:,None],s1+r1[:,None],bc+br,heel[None]+[.16,.11,0]]),axis=0)
    scale=HH*rng.uniform(.815,1.055)/np.max((hi-lo)[:2])
    off=(HH/2)-(hi+lo)[:2]*.5*scale + HH*rng.normal(0,.035,2)
    off3=np.r_[off,0.]
    dep=np.full((HH,HH),-1.e6); intensity=np.zeros((HH,HH));
    # Signs are identified partly by internal shading. Model an upper or
    # upper-side light, ambient fill, and shallow anatomical creases.
    light=unit(np.array([rng.uniform(-.98,.65),rng.uniform(-1.25,-.36),rng.uniform(.40,1.3)]))
    amb=rng.uniform(.18,.39); diffuse=rng.uniform(.42,.65)
    bg=rng.uniform(.52,.85) if form%6!=0 else rng.uniform(.24,.57)
    albedo=rng.uniform(.77,1.03)
    def put(sl,z,nx,ny,nz,ok,material=1.):
        hit=ok & (z>dep[sl]); dep[sl]=np.where(hit,z,dep[sl]);
        lit=(nx*light[0]+ny*light[1]+nz*light[2]); shade=amb+diffuse*(.18+.82*np.maximum(0.,lit))
        intensity[sl]=np.where(hit,albedo*shade*material,intensity[sl])
    # Orthographic ray / ellipsoid intersection, in block matrix form.
    def draw_ell(c,S):
        ex,ey=np.sqrt([S[0,0],S[1,1]]); sl=box(c[0]-ex,c[0]+ex,c[1]-ey,c[1]+ey)
        if sl is None: return
        i,j=sl; u=X[:,j]-c[0]; v=Y[i]-c[1];
        A,B,E=S[0,0],S[0,1],S[1,1]; det=A*E-B*B
        h00,h01,h11=E/det,-B/det,A/det
        au=h00*u+h01*v; av=h01*u+h11*v
        a,b=S[2,:2]; slopex=a*h00+b*h01; slopey=a*h01+b*h11
        q=max(1.e-8,S[2,2]-a*slopex-b*slopey)
        d=1-u*au-v*av; top=np.sqrt(np.maximum(0,q*d)); z=c[2]+slopex*u+slopey*v+top
        nx=au*q-slopex*top; ny=av*q-slopey*top; nz=top
        norm=1/np.sqrt(np.maximum(1.e-12,nx*nx+ny*ny+nz*nz)); put(sl,z,nx*norm,ny*norm,nz*norm,d>0)
    def draw_seg(a,b,ra,rb,kind):
        sl=box(min(a[0]-ra,b[0]-rb),max(a[0]+ra,b[0]+rb),min(a[1]-ra,b[1]-rb),max(a[1]+ra,b[1]+rb))
        if sl is None: return
        i,j=sl; dx,dy,dz=b-a; u=X[:,j]-a[0]; v=Y[i]-a[1]; ll=dx*dx+dy*dy; dr=rb-ra
        # A swept sphere. Move the closest point along the centerline towards
        # the camera-facing tangent of the sloped cylinder, then cap the ends.
        t0=(u*dx+v*dy)/max(.000001,ll); rt=ra+dr*np.clip(t0,0,1)
        d2=(u-t0*dx)**2+(v-t0*dy)**2
        if ll>.00001:
            t=np.clip(t0+np.sqrt(np.maximum(0,rt*rt-d2)/max(ll*(ll+dz*dz),1.e-9))*dz,0,1)
        else: t=float(dz>0)
        rad=ra+dr*t; xx=u-t*dx; yy=v-t*dy; dz2=rad*rad-xx*xx-yy*yy; zz=np.sqrt(np.maximum(0,dz2));
        z=a[2]+t*dz+zz; mat=1.
        if kind==0: mat=.97+.03*np.sin(np.pi*t)
        put(sl,z,xx/rad,yy/rad,zz/rad,dz2>0,mat)
    # The forearm need not point along the fingers: wrist flexion and ulnar
    # deviation are especially pronounced when spelling P, Q, G and H.
    wrist=np.array([-.018*pw,-.355*pl,-.022]) @ R.T
    out=unit((np.array([0,-1,-.20]) @ R.T)[:2]); down=np.array([0,1.]);
    if sign in 'GH': side=unit(np.array([-1.,rng.uniform(.30,1.05)])); out=unit(.36*out+.64*side)
    elif sign in 'PQ':
        side=unit(np.array([rng.uniform(.7,1.2),rng.uniform(-.25,.80)])); out=unit(.25*out+.75*side)
    else: out=unit(.36*out+.64*down+[rng.uniform(-.28,.22),0])
    a=wrist*scale+off3; b=a+np.r_[out*2.2,-.60]*scale
    draw_seg(a,b,scale*.245*pw,scale*.34*pw,2)
    for c,ss in zip(bc,S): draw_ell(c*scale+off3,ss*scale**2)
    for a,b,ra,rb,s in zip(s0,s1,r0,r1,seg): draw_seg(a*scale+off3,b*scale+off3,ra*scale,rb*scale,s[4])
    # A short-range occlusion term shades grooves between touching digits.
    mask=dep>-1.e5; d=np.where(mask,dep,-1000.)
    occ=np.zeros_like(dep)
    for delta in [(0,3),(0,-3),(3,0),(-3,0)]:
        sh=np.roll(d,shift=delta,axis=(0,1));
        if delta[0]>0: sh[:delta[0]]=-1000.
        if delta[0]<0: sh[delta[0]:]=-1000.
        if delta[1]>0: sh[:,:delta[1]]=-1000.
        if delta[1]<0: sh[:,delta[1]:]=-1000.
        occ+=np.clip((sh-d-1.8)/4.5,0,1)
    im=np.where(mask,intensity*(1.-.105*occ),bg)
    return im.reshape(28,2,28,2).mean(axis=(1,3))

# Fixed image descriptor. Derivatives have zero DC; small local windows
# integrate oriented contrast, and one-sided soft clipping limits outliers.
def kernels():
    y,x=np.mgrid[-2:3,-2:3]; sig=.85; gauss=np.exp(-.5*(x*x+y*y)/sig**2); gauss/=gauss.sum()
    ret=[]
    for angle in np.arange(8)*np.pi/4:
        k=gauss*(np.cos(angle)*x+np.sin(angle)*y); ret.append(k/max(np.sum(np.maximum(k,0)),1.e-9))
    surround=np.exp(-.5*(x*x+y*y)/1.5**2); surround/=surround.sum()
    k=gauss-surround; k/=np.maximum(k,0).sum()
    ret.extend([k,-k]); return np.array(ret)

TAPER=np.array([.35,.78,1.,1.,1.,1.,1.,1.,1.,1.,.78,.35])
TAPER=np.outer(TAPER,TAPER)
TAPER[9:]*=np.array([.98,.91,.80])[:,None]
KT=kernels()
KTFLAT=KT.reshape(10,25).T.copy()

def descriptors(im):
    # Single-render processing only. Return one high-dimensional template,
    # here split into its oriented and low-frequency parts for two dense ops.
    images=(im-im.mean())/np.sqrt(im.var()+1.e-5)
    patches=np.lib.stride_tricks.sliding_window_view(images,(5,5)).reshape(-1,25)
    f=(patches @ KTFLAT).reshape(24,24,10).transpose(2,0,1)
    f=np.clip(f-.055,0,1.15)
    f=f.reshape(10,12,2,12,2).mean((2,4));
    # A triangular spatial lowpass softly votes to adjacent cells. Outside
    # the valid gradient grid zero votes are supplied, exactly as in conv.
    p=np.pad(f,((0,0),(1,1),(1,1)))
    f=(p[:,1:-1,1:-1]*4 + (p[:,:-2,1:-1]+p[:,2:,1:-1]+p[:,1:-1,:-2]+p[:,1:-1,2:])*2
       +p[:,:-2,:-2]+p[:,2:,:-2]+p[:,:-2,2:]+p[:,2:,2:])/16.
    f=f*TAPER; f-=f.mean(axis=(1,2),keepdims=True)*TAPER/(TAPER.mean())
    # Angular-selective HOG-like channels + low frequency normalized grey.
    f[8:]*=.88
    f=f.reshape(-1); f/=max(np.linalg.norm(f),1e-8)
    grey=images.reshape(14,2,14,2).mean((1,3)); grey=np.clip(grey,-1.7,1.7)
    taper=np.pad(TAPER,1,constant_values=.17); grey=(grey-(grey*taper).sum()/taper.sum())*taper
    grey=grey.reshape(-1); grey/=max(np.linalg.norm(grey),1e-8)
    return f,grey

# Assemble the neural network. The image side of a match has data-independent
# normalization (per input image), not parameters estimated from examples.
def build(dst):
    weights={}; layers=[]
    def layer(name,op,ins,**kw):
        layers.append(dict(name=name,op=op,inputs=[ins] if isinstance(ins,str) else ins,**kw)); return name
    def conv(name,ins,w,b=None,pad=0):
        w=np.asarray(w,np.float32); weights[name+'.W']=w; weights[name+'.b']=np.zeros(w.shape[0],np.float32) if b is None else np.broadcast_to(np.asarray(b,np.float32),w.shape[0]).copy();
        return layer(name,'conv',ins,out=w.shape[0],kernel=list(w.shape[2:]),padding=pad,stride=1)
    def dense(name,ins,w,b=None):
        weights[name+'.W']=np.asarray(w,np.float32); weights[name+'.b']=np.zeros(len(w),np.float32) if b is None else np.asarray(b,np.float32)
        return layer(name,'dense',ins,out=len(w))
    def clip(name,ins,n,lo,hi):
        # clip(x,lo,hi)-lo = relu(x-lo)-relu(x-hi)
        I=np.eye(n); a=conv(name+'_cutoffs',ins,np.concatenate([I,I])[:,:,None,None],np.repeat([-lo,-hi],n)); a=layer(name+'_hinges','relu',a)
        return conv(name+'_limited',a,np.concatenate([I,-I],axis=1)[:,:,None,None])
    def pool2(name,ins):return layer(name,'avgpool',ins,size=[2,2],stride=[2,2])
    u=layer('contrast','norm','image'); e=conv('oriented',u,KT[:,None]); e=clip('grad',e,10,.055,1.205)
    e=pool2('cells',e); k=np.outer([1,2,1],[1,2,1])/16.; e=conv('vote',e,np.eye(10)[:,:,None,None]*k,pad=1)
    # Normalize across position and orientation together; matching rows are
    # zero-sum per orientation, so subtracting a grand mean has no effect.
    e=layer('one_histogram','reshape',e,shape=[1,10,144]); e=layer('grad_energy','norm',e)
    g=pool2('albedo_cells',u); g=clip('brightness',g,1,-1.7,1.7); g=layer('albedo_energy','norm',g)
    n=640  # evenly many independent poses per sign; no ranking or filtering.
    we=np.empty((24*n,1440),np.float32); wg=np.empty((24*n,196),np.float32)
    for c,sign in enumerate(ABC):
        rng=np.random.default_rng(9407+1009*c)
        for j in range(n):
            im=render(sign,rng,j); fe,fg=descriptors(im)
            we[c*n+j]=fe*(.76/math.sqrt(1440)); wg[c*n+j]=fg*(.24/14)
    h=dense('pose_grad_match',e,we); a=dense('pose_albedo_match',g,wg); s=layer('pose_similarity','add',[h,a]);
    # Max within eight independent sub-libraries, then a fixed blend of their
    # mean and the global max. Reduces sensitivity to an accidental match.
    s=layer('group_by_sign','reshape',s,shape=[24,8,n//8]); s=layer('best_in_group','maxpool',s,size=[1,n//8],stride=[1,n//8]);
    m=layer('consensus','avgpool',s,size=[8,1],stride=[8,1]); b=layer('best_pose','maxpool',s,size=[8,1],stride=[8,1]);
    m=dense('consensus_weight',m,np.eye(24)*.59); b=dense('best_weight',b,np.eye(24)*.41); layer('scores','add',[m,b]);
    dst.mkdir(parents=True,exist_ok=True)
    (dst/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
    np.savez(dst/'weights.npz',**weights)

if __name__=='__main__': build(Path(sys.argv[1]))
