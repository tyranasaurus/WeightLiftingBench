"""A small articulated hand, an orthographic renderer, and fixed template matching.

All distances below are in arbitrary anatomical units (palm breadth about 4).
X points from little finger to thumb, Y from wrist to fingertips, Z out of palm.
The pose descriptions are angles at the knuckles, not photo coordinates.
"""
import json
import sys
from pathlib import Path
import numpy as np

ABC = 'ABCDEFGHIKLMNOPQRSTUVWXY'
BATCH = 32
POSES = 832                  # independently sampled 3-D hands per letter
OFFSETS = 5                  # nearby crop centers per pose
T = POSES * OFFSETS
DEG = np.pi / 180


def mat(axis, degrees):
    a = np.asarray(degrees)*DEG
    co, si = np.cos(a), np.sin(a)
    m = np.zeros(a.shape+(3,3))
    j,k=(axis+1)%3,(axis+2)%3
    m[...,axis,axis]=1
    m[...,j,j]=m[...,k,k]=co
    m[...,j,k]=-si; m[...,k,j]=si
    return m


def batchcol(x):
    return np.asarray(x)[...,None]


def unit(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-10)


def geometry(letter, n, rng):
    U = lambda a,b: rng.uniform(a,b,n)
    def vec(x,y,z): return np.stack(np.broadcast_arrays(x,y,z,np.empty(n))[:3],axis=-1)
    def mix(a,b,t):return a+(b-a)*t[:,None]
    # Broad person-to-person variation; correlated thickness of all digits.
    width = U(.90,1.10); length = U(.91,1.08); thick = U(.87,1.14)
    fingerlen=U(.92,1.08) * length
    segments=[]; ellips=[]; nails=[]
    def ell(c,r,rot=None,material='skin'):
        if rot is None: rot=np.broadcast_to(np.eye(3),(n,3,3))
        ellips.append((c,r,rot,material))
    def bone(a,b,r1,r2,material='skin'):
        segments.append((a,b,r1,r2,material))
    # Ovoid palm/thenar and the heel joining the forearm.
    ell(vec(0,.35*length,-.06),vec(2.13*width,2.13*length,.76*thick))
    ell(vec(.92*width,-.02,.36*thick),vec(.92*width,1.48*length,.52*thick), mat(2,U(-22,-10)))
    ell(vec(-.96*width,-.16,.24*thick),vec(.82*width,1.40*length,.47*thick),mat(2,U(2,12)))
    # Abduction is separate from curling. Angles from the metacarpal axis;
    # triples below are MCP, PIP and DIP flexion in degrees.
    shut=(87,88,60); straight=(1,3,2); soft=(60,80,53)
    flex=[shut]*4; spread=[1,0,-4,-10]
    thumb='over'; yaw=U(-18,20); pitch=U(-10,13); roll=U(-12,12)
    radial=U(-.32,.32)       # direction from forearm to hand before projection
    if letter=='A':
        thumb='alongside'; flex=[(85,94,66)]*4; yaw=U(-10,26)
    elif letter=='B':
        flex=[straight]*4;spread=[-3,-1,2,5];thumb='across';yaw=U(-12,23)
    elif letter=='C':
        flex=[(39,40,25),(40,42,25),(44,45,28),(46,50,30)];thumb='lowerC';yaw=U(59,90);pitch=U(-15,5); roll=U(-23,8)
    elif letter=='D':
        flex=[straight,(59,75,58),(64,81,55),(67,84,60)];thumb='middlepad';yaw=U(3,40)
    elif letter=='E':
        flex=[(56,104,64),(56,106,64),(54,103,65),(54,102,65)];thumb='under';yaw=U(-13,24)
    elif letter=='F':
        flex=[(44,63,44),straight,(1,8,5),(5,9,7)];spread=[12,-3,-12,-25];thumb='indexpad';yaw=U(8,40);roll=U(-8,18)
    elif letter=='G':
        flex=[straight,shut,shut,shut]; thumb='parallel'; yaw=U(-80,-43);pitch=U(-4,15);roll=U(-105,-79);radial=U(-2.8,-.9)
    elif letter=='H':
        flex=[straight,straight,shut,shut];spread=[-3,1,0,-6];thumb='over';yaw=U(-35,10);pitch=U(-22,16);roll=U(-108,-79);radial=U(-3,-1.2)
    elif letter=='I':
        flex=[shut,shut,shut,straight]; spread[3]=U(-22,-6);thumb='over'
    elif letter=='K':
        # The thumb is between the two long fingers, not laid over a fist.
        flex=[straight,(48,9,6),shut,shut];spread=[6,-14,-4,-10];thumb='between'; yaw=U(-89,-54);roll=U(6,29)
    elif letter=='L':
        flex=[straight,shut,shut,shut]; spread[0]=U(-6,5);thumb='open';yaw=U(-18,16);roll=U(-8,18)
    elif letter in 'MN':
        flex=[(73,92,57)]*(3 if letter=='M' else 2)+[(91,95,62)]*(1 if letter=='M' else 2);thumb=letter;yaw=U(-18,33)
    elif letter=='O':
        flex=[(53,64,50),(51,66,55),(56,66,56),(58,69,53)];thumb='indexpad';yaw=U(54,88);roll=U(-21,10)
    elif letter=='P':
        flex=[straight,(68,6,3),shut,shut];spread=[8,-13,-3,-6];thumb='between';yaw=U(-30,1);pitch=U(27,57);roll=U(62,98);radial=U(-.3,.5)
    elif letter=='Q':
        flex=[straight,shut,shut,shut];thumb='parallel';yaw=U(23,65);pitch=U(24,61);roll=U(152,202);radial=U(-1.0,.8)
    elif letter=='R':
        flex=[(6,6,2),(1,4,2),shut,shut];spread=[-15,10,-4,-9];thumb='across';yaw=U(-26,23)
    elif letter=='S':
        thumb='over';yaw=U(-30,33)
    elif letter=='T':
        flex=[(75,91,60),shut,shut,shut];thumb='T';yaw=U(-18,31)
    elif letter=='U':
        flex=[straight,straight,shut,shut];spread=[-4,3,0,-10];thumb='across';yaw=U(-24,28)
    elif letter=='V':
        flex=[straight,straight,shut,shut];spread=[U(10,19),U(-20,-9),-4,-10];thumb='across';yaw=U(-20,23)
    elif letter=='W':
        flex=[straight,straight,straight,shut];spread=[U(12,22),0,U(-24,-13),-10];thumb='little';yaw=U(-17,23)
    elif letter=='X':
        flex=[(24,84,45),shut,shut,shut];thumb='over';yaw=U(21,65);pitch=U(-18,8)
    elif letter=='Y':
        flex=[shut,shut,shut,straight];spread[3]=U(-34,-16);thumb='open';yaw=U(-15,18);roll=U(-14,13)
    # Horizontal G, H and P require radial/ulnar wrist deviation, rather
    # than rotating a straight raised arm as a unit. H brings the two long
    # fingers into profile; Q tips the metacarpus over the lowered digits.
    side_pose=letter in 'GHPTQ'
    if letter=='G':
        # The thumb side is uppermost for G: palm toward the signer, wrist
        # flexed down from the horizontal index (unlike H's paired fingers).
        flex[0]=(U(2,21),3,2); yaw=U(123,183);roll=U(-101,-73)
    if letter=='H':
        flex[:2]=[(U(6,29),3,2)]*2;yaw=U(60,111);roll=U(-100,-58)
    if letter=='P':
        yaw=U(-35,-4);pitch=U(33,65);roll=U(51,99)
    if letter=='Q':
        # Flex the wrist away from the raised forearm: index and thumb
        # point toward the ground and the back of the hand is visible.
        flex[0]=(U(47,79),4,3);yaw=U(-33,24);pitch=U(61,92);roll=U(-26,29)
    # T: include the side view with the index loosely curled over the thumb,
    # as well as the compact front-facing fist.
    if letter=='T':
        alt=U(0,1)<.75
        yaw=np.where(alt,U(109,157),yaw);roll=np.where(alt,U(-107,-78),roll)
        a,b,c=flex[0]; flex[0]=tuple(np.where(alt,ex,cur) for cur,ex in zip((a,b,c),(19,6,4)))
    if letter=='X':
        pitch=U(-24,14);yaw=U(39,89);flex[0]=(U(1,25),U(83,113),U(42,73))
    if letter=='K':
        # View the two long fingers edge-on. The middle finger projects
        # diagonally forward while the side of the index stays upright.
        yaw=U(-105,-69);roll=U(3,28);flex[1]=(U(31,61),8,6)
    if letter=='A':
        # A is distinct from S because the thumb stays OUTSIDE the fist.
        roll=U(-31,11);pitch=U(-20,14)
    if letter in 'EMSNT':
        # E has knuckled fingertips on top of the thumb; M and N have a
        # sloping row of fingers wrapped OVER a buried thumb. An S has the
        # thumb most superficial, with its tip beside the middle knuckle.
        if letter=='E':flex=[(U(37,59),U(97,124),U(58,82)) for _ in range(4)]
        if letter=='M':yaw=U(12,53);roll=U(-36,7);pitch=U(-19,17)
        if letter=='N':yaw=U(24,67);roll=U(-39,4);pitch=U(-15,29)
        if letter=='S':yaw=U(23,75);roll=U(-33,10);pitch=U(-24,16)
    points=[]
    for f,(x,y,ln,r) in enumerate(zip([1.38,.47,-.46,-1.34],[1.94,2.09,1.98,1.61],[3.35,3.73,3.46,2.72],[.48,.49,.466,.414])):
        r=r*thick*U(.95,1.05); base=vec(x*width,y*length,-.015)
        ab=(spread[f]+U(-3.2,3.2))*DEG
        a,b,c=flex[f]; bend=U(-6,7); a=a+bend+U(-4,4);b=b+U(-5,6);c=c+U(-6,6)
        # A more cylindrical upper palm where the four knuckles meet it;
        # blend in a raised metacarpal pad at the base of each finger.
        ell(base+vec(0,-.48,-.02),vec(r*1.08,.91*length,r*1.03),mat(2,-ab/DEG*.45))
        sizes=ln*fingerlen*U(.955,1.045)
        # Each phalanx has a slightly narrower middle and a rounded knuckle.
        p=[base]; radii=[1,.94,.84,.73]
        for j,(angle,fr) in enumerate(zip([a,a+b,a+b+c],[.455,.305,.24])):
            t=angle*DEG
            direction=vec(np.cos(t)*np.sin(ab),np.cos(t)*np.cos(ab),np.sin(t))
            q=p[-1]+direction*(sizes*fr)[:,None]
            bone(p[-1],q,r*radii[j],r*radii[j+1]);p.append(q)
        # Small, pale ellipsoid on the dorsal side of the distal phalanx.
        sidevec=vec(np.cos(ab),-np.sin(ab),0)
        back=unit(np.cross(direction,sidevec)); rot=np.stack((sidevec,direction,back),axis=-1)
        ell(mix(p[-2],p[-1],np.full(n,.64))+back*(r*.68)[:,None],vec(r*.59,sizes*.097,r*.15),rot,'nail')
        points.append(p)
    # Thumb bends obliquely from its own low, opposable basal joint.
    base=vec(1.30*width,-.38*length,.26*thick)
    if thumb=='alongside': coords=[(1.94,.63,.41),(2.02,1.70,.53),(1.98,2.63,.58)]
    elif thumb=='lowerC': coords=[(2.05,.05,.92),(2.30,.18,1.95),(2.31,.60,2.78)]
    elif thumb=='open': coords=[(2.26,.02,.45),(3.23,.39,.55),(4.06,.88,.60)]
    elif thumb=='parallel':
        if letter=='Q':coords=[(2.06,.28,1.05),(2.26,.92,1.94),(2.15,1.62,2.72)]
        else:coords=[(1.94,.74,.77),(1.86,1.85,.81),(1.78,2.87,.77)]
    elif thumb in ('middlepad','indexpad'):
        f=int(thumb=='middlepad'); end=points[f][-1]+vec(.16,-.25,.04)
        mid=mix(base,end,np.full(n,.45))+vec(.78,-.39,.04)
        distal=mix(mid,end,np.full(n,.58))+vec(.18,-.10,.17)
        coords=None
    elif thumb=='under':coords=[(1.67,.48,.94),(.79,.90,1.29),(-.18,.96,1.34)]
    elif thumb in ('M','N','T'):
        f={'T':0,'N':1,'M':2}[thumb]
        tipx=(.91,-.01,-.88)[f]; coords=[(1.55,.56,1.0),((tipx+1.6)/2,1.56,1.38),(tipx,2.49,1.59)]
    elif thumb=='little':coords=[(1.36,.65,1.09),(.36,1.22,1.40),(-.76,1.54,1.77)]
    elif thumb=='between':coords=[(1.52,.80,.92),(1.02,1.97,1.22),(.84,2.97,1.36)]
    elif thumb=='across':coords=[(1.48,.63,.89),(.53,1.13,1.35),(-.43,1.16,1.41)]
    else: coords=[(1.67,.60,1.05),(1.06,1.55,1.89),(.11,2.04,2.03)]
    if coords is not None:
        shiftx,shifty,shiftz=U(-.24,.24),U(-.25,.25),U(-.22,.22)
        mid,distal,end=[vec(x*width+shiftx*i/3,y*length+shifty*i/3,z*thick+shiftz*i/3) for i,(x,y,z) in enumerate(coords,1)]
    else:
        mid+=vec(U(-.18,.18),U(-.20,.20),U(-.14,.14))
    if letter=='S':
        end+=vec(-.23,.31,.30);distal+=vec(.02,.18,.29)
    if letter=='T':
        distal=np.where(alt[:,None],vec(2.28,1.27,1.13),distal)
        end=np.where(alt[:,None],vec(2.83,1.81,1.22),end)
    a=base
    for b,r,s in [(mid,.63,.55),(distal,.55,.48),(end,.48,.4)]:
        bone(a,b,r*thick,s*thick);a=b
    wrist=vec(.0,-1.05*length,-.12)
    d=unit(vec(radial,-1,U(-.47,.28)))
    proximal=wrist+4.7*d; far=wrist+16*d
    rotation=mat(2,roll)@mat(0,pitch)@mat(1,yaw)
    if side_pose:
        # Draw the elbow separately from the hand: for a downward-pointing
        # hand a raised forearm is foreshortened and partly hidden by it.
        screen_d=unit(vec(U(-1.25,.36),U(-2,-.42),U(-1.3,-.1)))
        if letter=='Q':screen_d=unit(vec(U(-.9,.7),U(.45,1.45),U(-2.7,-.45)))
        if letter=='P':screen_d=unit(vec(U(.1,1.25),U(-2,-.4),U(-1.9,-.4)))
        if letter=='T':screen_d=np.where(alt[:,None],unit(vec(U(-.8,-.1),-1,U(-.9,-.1))), (rotation@d[...,None])[...,0])
        d=(rotation.transpose(0,2,1)@screen_d[...,None])[...,0]
        proximal=wrist+d*U(2.8,6.0)[:,None];far=wrist+d*16
    # Overlapping tapered surfaces make the forearm and optional cuff.
    bone(wrist,far,1.27*width,1.91*width,'arm')
    bone(proximal,far,1.53*width,2.10*width,'sleeve')
    return ellips,segments,rotation


def raster(letter,n,rng):
    ell,seg,R=geometry(letter,n,rng)
    def xf(a):return (R@a[...,None])[...,0]
    es=[];ss=[];lo=np.full((n,2),1e5);hi=-lo
    def bounds(low,high):
        np.minimum(lo,low,out=lo);np.maximum(hi,high,out=hi)
    for c,rad,rot,material in ell:
        c=xf(c);axes=(R@rot)*rad[:,None,:]; ex=np.linalg.norm(axes[:,:2],axis=-1)
        bounds(c[:,:2]-ex,c[:,:2]+ex)
        # World-space quadratic form of the ellipsoid.
        q=(R@rot)/rad[:,None,:]; es.append((c,q@q.transpose(0,2,1),material))
    for a,b,r,s,material in seg:
        a=xf(a); b=xf(b); ss.append((a,b,r,s,material))
        if material=='skin':
            bounds(np.minimum(a[:,:2]-r[:,None],b[:,:2]-s[:,None]),np.maximum(a[:,:2]+r[:,None],b[:,:2]+s[:,None]))
    U=lambda a,b:rng.uniform(a,b,n)
    # Frame the projected hand, leaving room for the arm, with independent crop jitter.
    extent=(hi-lo).max(1)*U(1.00,1.48)
    ctr=(hi+lo)*.5
    wrist=ss[-2][0][:,:2]; along=wrist-ctr
    ctr+=along/np.maximum(np.linalg.norm(along,axis=1,keepdims=True),.01)*(extent*U(.035,.16))[:,None]
    ctr+=extent[:,None]*rng.uniform(-.038,.038,(n,2))
    size=48
    def plane(v):return v[:,None,None]
    grid=(np.arange(size,dtype=np.float32)+.5)/size-.5
    px=(plane(ctr[:,0])+grid[None,None,:]*plane(extent)).astype('f')
    py=(plane(ctr[:,1])-grid[None,:,None]*plane(extent)).astype('f')
    shape=(n,size,size)
    depth=np.full(shape,-1e4,'f'); intensity=np.zeros(shape,'f'); skin=U(.46,.77); bg=U(.63,.85)
    light=unit(np.stack([U(-1.05,.35),U(.2,1.05),U(.53,1.5)],axis=-1)); lx,ly,lz=[plane(z).astype('f') for z in light.T]
    key=plane(U(.23,.47)*skin).astype('f'); ambient=plane(skin).astype('f')-key*.67
    sleeve=plane(U(.06,.39)).astype('f'); sleeve_on=(U(0,1)<.60)[:,None,None]
    def paint(z,nx,ny,nz,ok,material):
        use=ok & (z>depth)
        if material=='sleeve':use &= sleeve_on
        dot=nx*lx+ny*ly+nz*lz
        # Convex highlights and grazing-angle darkening reveal the joints.
        ink=ambient+key*(np.maximum(dot,-.35)+.26*(nz-.65))
        if material=='nail':ink+=.045
        if material=='sleeve':ink=sleeve+.06*dot
        np.copyto(intensity,ink,where=use);np.copyto(depth,z,where=use)
    for a,b,r,s,material in ss:
        # Intersection of a ray with the union of spheres along a phalanx;
        # varying radius gives taper. The constant-radius case is analytic.
        x=px-plane(a[:,0]).astype('f');y=py-plane(a[:,1]).astype('f')
        dx,dy,dz=[plane(t).astype('f') for t in (b-a).T]
        rr=plane(r).astype('f');dr=plane(s-r).astype('f')
        den=np.maximum(dx*dx+dy*dy,.000001); line=(x*dx+y*dy)/den
        ex=x-line*dx;ey=y-line*dy
        rc=rr+np.clip(line,0,1)*dr
        v=np.maximum(rc*rc-ex*ex-ey*ey,0)
        t=np.clip(line+dz*np.sqrt(v/(den*(den+dz*dz))),0,1)
        ex=x-t*dx;ey=y-t*dy;rad=rr+t*dr
        val=rad*rad-ex*ex-ey*ey; ez=np.sqrt(np.maximum(val,0))
        paint(plane(a[:,2]).astype('f')+t*dz+ez,ex/rad,ey/rad,ez/rad,val>0,material)
    for c,q,material in es:
        x=px-plane(c[:,0]).astype('f');y=py-plane(c[:,1]).astype('f')
        A,B,C,D,E,F=[plane(q[:,i,j]).astype('f') for i,j in ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))]
        l=C*x+E*y; discr=l*l-F*(A*x*x+2*B*x*y+D*y*y-1)
        z=(-l+np.sqrt(np.maximum(discr,0)))/F
        nx=A*x+B*y+C*z;ny=B*x+D*y+E*z;nz=l+F*z;inv=1/np.sqrt(np.maximum(nx*nx+ny*ny+nz*nz,1e-10))
        paint(plane(c[:,2]).astype('f')+z,nx*inv,ny*inv,nz*inv,discr>0,material)
    present=depth>-1e3
    # A narrow contact shadow on the occluded side of a depth discontinuity.
    shadow=np.zeros(shape,'f')
    for ax,step in ((1,1),(1,-1),(2,1),(2,-1)):
        other=np.roll(depth,step,axis=ax); oc=np.clip((other-depth-.18)*.83,0,1)
        if ax==1:oc[:,0 if step==1 else -1,:]=0
        else:oc[:,:,0 if step==1 else -1]=0
        shadow=np.maximum(shadow,oc)
    intensity-=.12*shadow
    # Broad soft shading down the side of the hand, represented by a
    # low-frequency incident-light gradient in addition to local normals.
    intensity+=plane(U(-.17,-.035))*grid[None,None,:]+plane(U(-.08,.045))*grid[None,:,None]
    # Smooth ambient gradient independent of the hand shape.
    back=plane(bg)+plane(U(-.09,.09))*grid[None,None,:]+plane(U(-.10,.10))*grid[None,:,None]
    im=np.where(present,intensity,back).astype('f')
    return im


def angle_basis(ch):
    # A triangular voting bin: retain a signed projection only when it is
    # appreciable relative to the cross-axis projection. The last filter
    # (a center-surround filter) is independent of orientation.
    w=np.eye(2*ch,dtype='f')
    for i in range(2*ch):
        if i%ch<4:
            j=(i%ch+2)%4
            w[i,[j,j+ch]]=-.60
    return w


def kernels():
    # Four axes of oriented contrast, and a Mexican hat sensitive to ridges.
    y,x=np.mgrid[-2:3,-2:3]; g=np.exp(-(x*x+y*y)/(2*.95**2))
    bank=[]
    for i in range(4):
        a=i*np.pi/4; w=(x*np.cos(a)+y*np.sin(a))*g; w-=w.mean();w/=np.abs(w).sum()*.5;bank.append(w)
    w=np.exp(-(x*x+y*y)/(2*.68**2))-.0;w=w/w.sum()-g/g.sum();w/=np.abs(w).sum()*.5;bank.append(w)
    return np.asarray(bank,dtype='f')


def unit_channel(x):
    z=x-x.mean(axis=(-2,-1),keepdims=True)
    return z/np.sqrt(np.mean(z*z,axis=(-2,-1),keepdims=True)+1e-5)


def crop(im,dx,dy):
    n,S,_=im.shape
    # Bilinear quadrature at four subpixels for each output pixel.
    grid=(np.arange(56,dtype='f')+.5)*S/56-.5
    xx=np.clip(grid[None,:]+dx[:,None]*S,0,S-1.001);yy=np.clip(grid[None,:]+dy[:,None]*S,0,S-1.001)
    xi=xx.astype(np.int32);yi=yy.astype(np.int32);fx=(xx-xi)[:,None,:];fy=(yy-yi)[:,:,None]; b=np.arange(n)[:,None,None]
    xi=xi[:,None,:];yi=yi[:,:,None]
    a=(im[b,yi,xi]*(1-fx)+im[b,yi,xi+1]*fx)*(1-fy)+(im[b,yi+1,xi]*(1-fx)+im[b,yi+1,xi+1]*fx)*fy
    return a.astype('f').reshape(n,28,2,28,2).mean((2,4))


def cell(x,h):
    n,c,w,_=x.shape
    return x.reshape(n,c,w//h,h,w//h,h).mean((3,5))


def templates(images,K):
    # No reductions over template index, here or in the build loop.
    im=unit_channel(images); n=len(im)
    resp=np.zeros((n,len(K),24,24),'f')
    for y in range(5):
        for x in range(5):resp+=K[None,:,y,x,None,None]*im[:,None,y:y+24,x:x+24]
    def finish(a):
        a=unit_channel(a);return a.reshape(n,-1)
    ch=len(K)
    rect=np.maximum(np.concatenate((resp,-resp),axis=1),0)
    for i in range(2*ch):
        if i%ch<4: rect[:,i]=np.maximum(rect[:,i]-.60*np.abs(resp[:,(i%ch+2)%4]),0)
    # Contrast saturation keeps a sharp garment boundary from dwarfing a
    # dim fingertip. Threshold is in standardized image contrast units.
    rect=np.minimum(rect,1.05)
    hist=cell(rect,3)
    signed=finish(hist);unsigned=finish(hist[:,:ch]+hist[:,ch:]);coarse=finish(cell(im[:,None],4))
    # A second, finer orientation grid captures small internal ridges and
    # fingertip separations without the spatial averaging of a larger cell.
    fine=np.abs(resp[:,:4]);fine=np.maximum(fine-.60*fine[:,[2,3,0,1]],0)
    fine=np.minimum(fine,1.05)
    fine=finish(cell(fine,2))
    # Scale factors compensate analytically for the number of coordinates.
    # A radial spatial taper gives the peripheral background less influence.
    def weighted(a,q,weight):
        t=(np.arange(q)+.5)/q-.5
        mask=(1-.75*(t[:,None]**2+t**2)).astype('f')
        a=a.reshape(n,-1,q,q)*mask
        a=a-a.mean(axis=(2,3),keepdims=True)
        a*=1/np.sqrt(np.mean(a*a,axis=(2,3),keepdims=True)+1e-5)
        return a.reshape(n,-1)*(weight/a[0].size)
    return weighted(signed,8,.32),weighted(unsigned,8,.42),weighted(coarse,7,.18),weighted(fine,12,.20)


def build(folder):
    folder.mkdir(exist_ok=True,parents=True)
    K=kernels(); C=len(K); layers=[];weights={}
    def op(name,kind,inp,**kw):
        layers.append(dict(name=name,op=kind,inputs=inp if isinstance(inp,list) else [inp],**kw));return name
    def linear(name,kind,inp,w,**kw):
        weights[name+'.W']=w.astype('f',copy=False);weights[name+'.b']=np.zeros(len(w),'f')
        return op(name,kind,inp,out=len(w),**kw)
    im=op('exposure','norm','image')
    st=linear('differentials','conv',im,np.concatenate((K,-K))[:,None],kernel=[5,5],padding=0,stride=1)
    st=op('half_waves','relu',st)
    st=linear('angle_bins','conv',st,angle_basis(C)[:,:,None,None],kernel=[1,1],padding=0,stride=1)
    st=op('oriented_votes','relu',st)
    # Apply min(x, ceiling) via ceiling - ReLU(ceiling - x).
    # Working on pooled maps would be cheaper, but would let a single very
    # strong edge hide several weak finger edges in the same cell.
    neg=-np.eye(C*2,dtype='f')[:,:,None,None]
    st=linear('headroom','conv',st,neg,kernel=[1,1],padding=0,stride=1);weights['headroom.b'][:]=1.05
    st=op('bounded_headroom','relu',st)
    st=op('mean_headroom','avgpool',st,size=[3,3],stride=[3,3])
    st=linear('signed_cells','conv',st,neg,kernel=[1,1],padding=0,stride=1);weights['signed_cells.b'][:]=1.05
    def poolnorm(pre,inp,q):
        return op(pre+'_norm','norm',op(pre+'_cells','avgpool',inp,size=[q,q],stride=[q,q]))
    signed=op('signed_norm','norm',st)
    # Sum the two half-waves of each filter after pooling (same as mean |r|).
    u=linear('both_polarities','conv','signed_cells',np.tile(np.eye(C),(1,2))[:,:,None,None],kernel=[1,1],padding=0,stride=1)
    unsigned=op('unsigned_norm','norm',u);coarse=poolnorm('tone',im,4)
    # The fine grid sees the same *individual* template as the coarse grid.
    # Use a subset with one offset per pose to bound the extra bank's size.
    # The location is shared with the coarse descriptor for that template.
    fine=op('fine_headroom','avgpool','bounded_headroom',size=[2,2],stride=[2,2])
    select=np.tile(np.eye(C,dtype='f')[:4],(1,2))[:,:,None,None]
    fine=linear('fine_axes','conv',fine,-select,kernel=[1,1],padding=0,stride=1);weights['fine_axes.b'][:]=2.10
    fine=op('fine_norm','norm',fine)
    sizes=[2*C*64,C*64,49];banks=[np.empty((24*T,d),'f') for d in sizes]
    fine_bank=np.empty((24*POSES,4*144),'f')
    for k,letter in enumerate(ABC):
        rng=np.random.default_rng(92041+1009*k)
        for start in range(0,POSES,BATCH):
            n=min(BATCH,POSES-start);im48=raster(letter,n,rng)
            alpha=rng.uniform(0,np.pi*.5,n); dist=rng.uniform(.048,.091,n)
            for j in range(OFFSETS):
                dx=dist*np.cos(alpha+j*np.pi/2) if j else np.zeros(n)
                dy=dist*np.sin(alpha+j*np.pi/2) if j else np.zeros(n)
                v=templates(crop(im48,dx,dy),K)
                sl=slice(k*T+start*OFFSETS+j,k*T+(start+n)*OFFSETS+j,OFFSETS)
                for arr,val in zip(banks,v[:3]):arr[sl]=val
                if j==0:fine_bank[k*POSES+start:k*POSES+start+n]=v[3]
    contributions=[linear('match_'+key,'dense',src,w) for key,src,w in zip(('signed','axis','tone'),(signed,unsigned,coarse),banks)]
    scores=op('correlations','add',contributions)
    scores=op('pose_grid','reshape',scores,shape=[24,POSES,OFFSETS])
    scores=op('best_alignment','maxpool',scores,size=[1,OFFSETS],stride=[1,OFFSETS])
    # Consensus between independent sub-banks; all pooling here is inference,
    # not a computation on images used to select any weight or hyperparameter.
    scores=op('best_pose','maxpool',scores,size=[POSES//4,1],stride=[POSES//4,1])
    scores=op('vote','avgpool',scores,size=[4,1],stride=[4,1])
    fine=linear('match_fine','dense',fine,fine_bank)
    fine=op('fine_grid','reshape',fine,shape=[24,POSES,1]);fine=op('fine_best','maxpool',fine,size=[POSES//4,1],stride=[POSES//4,1])
    fine=op('fine_vote','avgpool',fine,size=[4,1],stride=[4,1]);scores=op('combined','add',[scores,fine]);op('letters','reshape',scores,shape=[24])
    (folder/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(folder/'weights.npz',**weights)


if __name__=='__main__':build(Path(sys.argv[1]))
