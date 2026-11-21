"""Data-free ASL classifier. All anatomical coordinates are in approximate cm.

The only objects used as class-dependent weights are separate, procedurally
posed hand templates. There is no example set, optimization or calibration.
"""
import json
import sys
from pathlib import Path
import numpy as np

ABC = 'ABCDEFGHIKLMNOPQRSTUVWXY'
D = np.pi / 180
# Reproducibility: each class and pose has its own random stream.
SEED = 742619
COUNT = 2048


def basis(ax, ay, az):
    # Pitch, yaw, and clockwise roll, in that order. Y is up, Z toward eye.
    a,b,c=np.array([ax,ay,az])*D
    ca,sa,cb,sb,cc,sc=np.cos(a),np.sin(a),np.cos(b),np.sin(b),np.cos(c),np.sin(c)
    return np.array([[cc,sc,0],[-sc,cc,0],[0,0,1.]]) @ np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]]) @ np.array([[1.,0,0],[0,ca,-sa],[0,sa,ca]])


def hand(letter, rng):
    u=rng.uniform
    # Order: index, middle, ring, little. Angles at MCP, PIP and DIP.
    fist=np.array([67.,87.,56.]); straight=np.array([-3.,6.,3.])
    bend=np.tile(fist,(4,1)); fan=np.array([3.,0.,-2.,-5.])
    thumb='over'; pitch=u(-13,17); yaw=u(-24,33); roll=u(-16,18)
    if letter=='A': thumb='along'
    elif letter=='B':
        bend[:]=straight; fan[:]=[-2,0,2,3]; thumb='base'
    elif letter=='C':
        bend[:]=[28,41,24]; fan[:]=[-4,-2,1,4]; thumb='open'; yaw=u(58,102); roll+=u(0,18)
    elif letter=='D':
        bend[:]=[39,65,45]; bend[0]=straight; thumb='meet'; yaw=u(12,69)
    elif letter=='E':
        bend[:]=[18,104,66]; thumb='under'
    elif letter=='F':
        bend[:]=straight; bend[0]=[33,68,39]; fan[:]=[-6,5,-4,-15]; thumb='ok'; yaw=u(-15,47)
    elif letter=='G':
        thumb='parallel'
        if u()<.55:
            bend[0]=[u(38,78),5,1]; yaw=u(66,106); roll=u(28,57)
        else:
            bend[0]=[u(0,28),4,1]; yaw=u(91,167); roll=u(65,95)
    elif letter=='H':
        bend[:2]=straight+[u(0,25),0,0]; fan[:2]=[-3,0]; yaw=u(85,183); roll=u(62,99)
    elif letter=='I':
        bend[3]=straight; fan[3]=u(-16,-3)
    elif letter=='K':
        bend[:2]=straight; bend[1]=[u(25,65),6,2]; fan[:2]=[8,-13]; thumb='between'; yaw=u(-25,33)
    elif letter=='L':
        bend[0]=straight; thumb='out'; fan[0]=u(-8,8)
    elif letter in 'MNT':
        cover={'M':3,'N':2,'T':1}[letter]
        bend[:cover]=[u(35,56),83,56]; thumb='tuck'+str(cover); yaw=u(-24,49)
    elif letter=='O':
        bend[:]=[40,65,42]; fan[:]=[-9,-3,3,10]; thumb='meet'; yaw=u(51,111); roll+=u(1,17)
    elif letter=='P':
        bend[:2]=straight; bend[1]=[u(52,92),8,2]; fan[:2]=[6,-10]; thumb='between'; yaw=u(-126,-24); roll=u(-79,-32); pitch=u(-10,36)
    elif letter=='Q':
        bend[0]=[u(5,56),6,3]; thumb='parallel'; yaw=u(-137,-51); roll=u(-161,-104); pitch=u(-15,21)
    elif letter=='R':
        bend[:2]=straight; bend[1,0]+=u(6,18); fan[:2]=[u(-17,-8),u(11,21)]
    elif letter=='S': thumb='over'
    elif letter=='U':
        bend[:2]=straight; fan[:2]=[-4,1]
    elif letter=='V':
        bend[:2]=straight; fan[:2]=[u(10,19),u(-20,-10)]
    elif letter=='W':
        bend[:3]=straight; fan[:3]=[u(12,24),0,u(-23,-12)]; thumb='little'
    elif letter=='X':
        bend[0]=[u(3,29),u(51,85),u(26,53)]; yaw=u(-17,79); thumb='base'
    elif letter=='Y':
        bend[3]=straight; fan[3]=u(-30,-12); thumb='out'

    # A more tightly flexed knuckle draws the fingertips back over the palm.
    # Finger extension and thumb opposition couple to one another in the ASL
    # fist family, not just to the camera orientation. Vary these jointly.
    if letter in 'ANTMS':
        bend[:,0]+=u(-5,18); bend[:,1]+=u(-6,12)
    if letter in 'MNT':
        bend[:cover]=[u(47,72),u(76,101),u(47,67)]
    if letter=='E': bend[:,0]+=u(-10,13)
    if letter=='Q' and u()<.58:
        # Alternative wrist posture: palm upright, both pointing rays bend
        # down and forward at the knuckle, viewed from the thumb side.
        pitch=u(12,52); yaw=u(-100,-57); roll=u(-24,22); bend[0]=[u(82,111),u(5,19),6]
    if letter=='P' and u()<.52:
        pitch=u(7,37); yaw=u(-96,-43); roll=u(-56,-17); bend[0]=[u(8,39),5,3]; bend[1]=[u(72,104),10,7]
    if letter=='K':
        bend[1,0]+=u(-14,21); fan[1]+=u(-7,9); yaw+=u(-14,20)
    if letter=='F': fan[1:]+=np.array([1,0,-1])*u(-5,7)
    if letter=='X': bend[0,1]+=u(-12,16); yaw+=u(-8,18)

    # All four MCPs lie on a shallow transverse arch. The little MCP is
    # shorter, and slightly closer to the palm, than the middle one.
    root=np.array([[2.60,3.03,0.05],[.86,3.38,-.16],[-.91,3.12,-.04],[-2.52,2.34,.26]])
    lengths=np.array([[3.55,2.22,1.65],[3.95,2.51,1.77],[3.75,2.36,1.65],[2.86,1.76,1.41]])
    lengths*=u(.91,1.10)*rng.uniform(.96,1.04,(4,1)); thickness=u(.90,1.11)
    girth=np.array([[.98,.91,.79,.65],[1.01,.92,.83,.67],[.93,.85,.77,.62],[.81,.75,.66,.56]])*thickness
    bend+=rng.uniform(-7,7,(1,3))+rng.uniform(-4,4,(4,3)); fan+=u(-4,4)+rng.uniform(-2,2,4)
    angles=bend.cumsum(1)*D; lateral=fan[:,None]*D
    delta=np.stack([np.cos(angles)*np.sin(lateral),np.cos(angles)*np.cos(lateral),np.sin(angles)],axis=-1)*lengths[:,:,None]
    fingers=np.concatenate([root[:,None,:],root[:,None,:]+delta.cumsum(1)],axis=1)

    # Thumb: CMC originates near the wrist, MCP at the end of its mobile
    # metacarpal. Opposed poses place the thumb in front of the finger pads.
    cmc=[1.88,-2.38,.21]
    thumb_paths={
        'along':([3.31,.08,1.49],[3.45,2.35,1.77],[3.22,4.31,1.86]),
        'over':([2.62,.10,2.33],[.69,1.69,4.06],[-1.42,2.54,4.11]),
        'base':([2.56,-.23,1.84],[.77,.72,2.42],[-1.44,1.38,2.32]),
        'under':([2.60,-.17,1.93],[.94,1.22,3.34],[-1.36,1.43,3.17]),
        'out':([4.12,-.87,1.10],[5.95,.34,1.38],[7.98,1.18,1.50]),
        'open':([2.95,-.70,1.73],[2.97,.12,3.89],[2.68,1.50,5.55]),
        'parallel':([3.18,-.45,1.77],[3.85,1.32,3.10],[4.06,3.07,4.17]),
        'between':([2.78,-.22,2.14],[1.69,2.02,3.05],[.96,4.43,3.00]),
        'little':([2.03,-.05,2.03],[.01,1.41,2.54],[-2.22,1.63,2.66]),
        'tuck1':([3.05,-.04,1.35],[2.43,2.28,2.05],[1.79,4.44,2.44]),
        'tuck2':([2.55,-.09,1.59],[1.32,2.09,2.08],[.05,3.93,2.59]),
        'tuck3':([1.95,-.06,1.49],[.28,1.53,1.89],[-1.66,3.11,2.58])}
    if thumb in ('meet','ok'):
        dest=fingers[1 if thumb=='meet' else 0,-1].copy(); dest[1]-=.56; dest[0]+=.24
        if thumb=='meet': dest[0]+=.15
        mcp=np.array([2.92,-.52,1.82]); ip=.50*mcp+.50*dest+[.47,-.48,.30]
        t=np.array([cmc,mcp,ip,dest])
    else: t=np.array([cmc,*thumb_paths[thumb]])
    if thumb=='parallel':
        # Thumb parallel to but separated from the index (G and Q).
        t[2:]=t[1]+np.array([.31,.66])[:,None]*(fingers[0,-1]-fingers[0,0])+np.array([.70,-.28,.36])
    if thumb=='over': t[2:,2]+=[.50,1.10]
    if thumb=='along': t[2:,0]+=.42; t[-1,1]+=u(.12,1.35)
    if thumb.startswith('tuck'):
        n=int(thumb[-1]); t[-1,2]+=u(.65,1.71); t[-1,1]+=u(.00,1.35) if n==1 else u(-.22,.57)
    t[1:]+=rng.uniform(-.28,.28,(3,3)); t[1:,0]+=u(-.45,.45); t[1:,1]+=u(-.51,.51)
    if thumb=='out': t[1:,1]+=np.linspace(.3,1,3)*u(-1.5,1.9)
    digits=np.concatenate([fingers,t[None]],axis=0); radii=np.vstack([girth,np.array([1.27,1.12,.91,.76])*thickness]);
    # Palm and pads are ellipsoids with their own local tilt. The navicular
    # palm and thenar pad protrude in front; the dorsum is relatively flat.
    pads=[([0,-.15,-.02],[3.66,4.50,1.30],0),
          ([0,1.93,-.12],[3.42,2.06,1.24],-4),
          ([1.69,-1.00,.67],[1.62,2.97,1.01],20),
          ([-1.83,-.83,.48],[1.44,2.88,.85],-11)]
    stretch=np.array([u(.92,1.09),u(.94,1.06),u(.90,1.10)])
    cam=basis(pitch,yaw,roll); mat=cam*stretch
    return digits@mat.T,radii,pads,mat,cam


def render(letter,rng):
    digits,rad,pads,mat,cam=hand(letter,rng); u=rng.uniform
    # Fit the *generated* hand, not the arm, loosely inside a square camera.
    lo=(digits[:,:,:2]-rad[:,:,None]).min(axis=(0,1)); hi=(digits[:,:,:2]+rad[:,:,None]).max(axis=(0,1))
    for c,axes,a in pads:
        E=mat@basis(0,0,a)@np.diag(axes); xy=(mat@c)[:2]; sz=np.sqrt((E[:2]**2).sum(1)); lo=np.minimum(lo,xy-sz); hi=np.maximum(hi,xy+sz)
    extent=hi-lo
    # Crops can cut off a fingertip or include some forearm. Use independent
    # scales, mild decentering, and continuous camera variation.
    unit=max(extent)/u(.63,1.07); mid=(lo+hi)*.5
    # A convolutional match handles the last few pixels of decentering.
    mid[0]+=unit*u(-.115,.115); mid[1]-=unit*u(-.07,.21)
    res=56; pix=unit/res; xx=(np.arange(res)+.5-res/2)*pix+mid[0]; yy=(res/2-.5-np.arange(res))*pix+mid[1]
    # Using a 2x raster makes the silhouettes and finger separations
    # antialiased without any image library or external asset.
    zz=np.full((res,res),-999.,np.float32); image=np.empty_like(zz)
    bg=u(.42,.89) if u()<.83 else u(.17,.44); image.fill(bg)
    skin=u(.62,.94); ambient=u(.26,.57); key=np.array([u(-1.0,.35),u(.16,1.1),u(.35,1.13)]); key/=np.linalg.norm(key)
    def draw(depth,nx,ny,nz,sl,albedo=1.):
        buf=zz[sl]; mask=depth>buf
        diffuse=np.maximum(0.,nx*key[0]+ny*key[1]+nz*key[2]); lum=skin*albedo*(ambient+(1-ambient)*diffuse)
        np.copyto(image[sl],lum,where=mask,casting='unsafe'); np.copyto(buf,depth,where=mask,casting='unsafe')
    def window(xlo,xhi,ylo,yhi):
        ix0=max(0,int(np.floor((xlo-xx[0])/pix))); ix1=min(res,int(np.ceil((xhi-xx[0])/pix))+1)
        iy0=max(0,int(np.floor((yy[0]-yhi)/pix))); iy1=min(res,int(np.ceil((yy[0]-ylo)/pix))+1)
        if ix1<=ix0 or iy1<=iy0:return None
        return (slice(iy0,iy1),slice(ix0,ix1)),xx[None,ix0:ix1],yy[iy0:iy1,None]
    def ellipsoid(center,E,color):
        M=E@E.T; ax,ay=np.sqrt([M[0,0],M[1,1]]); cx,cy,cz=center
        w=window(cx-ax,cx+ax,cy-ay,cy+ay)
        if w is None:return
        sl,x,y=w; x=x-cx; y=y-cy; A=np.linalg.inv(M)
        linear=A[2,0]*x+A[2,1]*y; disc=linear**2-A[2,2]*(A[0,0]*x*x+2*A[0,1]*x*y+A[1,1]*y*y-1)
        z=(np.sqrt(np.maximum(disc,0))-linear)/A[2,2]; nx=A[0,0]*x+A[0,1]*y+A[0,2]*z; ny=A[1,0]*x+A[1,1]*y+A[1,2]*z; nz=linear+A[2,2]*z
        inv=1./np.sqrt(nx*nx+ny*ny+nz*nz+1e-12); draw(np.where(disc>=0,z+cz,-1000.),nx*inv,ny*inv,nz*inv,sl,color)
    def capsule(a,b,r0,r1,color,crease=False):
        # Approximate a swept cone with spherical ends. At each pixel, find
        # the frontmost sphere along the sweep, allowing for view inclination.
        x0,y0,z0=a; x1,y1,z1=b; dx,dy,dz=b-a; dr=r1-r0
        w=window(min(x0-r0,x1-r1),max(x0+r0,x1+r1),min(y0-r0,y1-r1),max(y0+r0,y1+r1))
        if w is None:return
        sl,x,y=w; rx=x-x0; ry=y-y0; l2=dx*dx+dy*dy; k=max(l2-dr*dr,.00001)
        dot=rx*dx+ry*dy+r0*dr; t0=dot/k
        height=r0*r0-rx*rx-ry*ry+dot*t0
        t=np.clip(t0+dz*np.sqrt(np.maximum(height,0)/(k*(k+dz*dz))),0,1)
        if l2<dr*dr+.00001: t=np.full_like(t,1. if dz+dr>0 else 0.)
        r=r0+t*dr; nx=rx-t*dx; ny=ry-t*dy; h2=r*r-nx*nx-ny*ny; nz=np.sqrt(np.maximum(h2,0)); dep=z0+t*dz+nz
        if crease:
            # Very shallow transverse flexion crease near the base of a phalanx.
            color=color*(1-.13*np.exp(-((t-.13)/.065)**2)*(t>0)*(t<1))
        draw(np.where(h2>=0,dep,-1000.),nx/r,ny/r,nz/r,sl,color)
    # Forearm orientation tracks the hand partly; the wrist joint can bend
    # independently. This also avoids drawing free-floating inverted hands.
    wrist=mat@np.array([-.12,-3.0,-.07]); arm=cam@np.array([u(-.22,.22),-1.,u(-.22,.22)])
    arm[1]=min(-.35,arm[1]); arm[0]=arm[0]*.70+u(-.32,.32); arm[2]=min(0.,arm[2])-.20; arm/=np.linalg.norm(arm)
    far=wrist+arm*unit*1.65
    capsule(wrist,far,2.14,2.67,u(.84,.99))
    for i,(c,axes,a) in enumerate(pads):ellipsoid(mat@c,mat@basis(0,0,a)@np.diag(axes),1 if i==0 else .99)
    for finger in range(5):
        tint=u(.94,1.02)
        for bone in range(3):
            capsule(digits[finger,bone],digits[finger,bone+1],rad[finger,bone],rad[finger,bone+1],tint*(1-.018*bone),bone>0)
    # Small self-occlusion shadows: sample toward the light on this rendering's
    # depth buffer. This is a lighting calculation, not a recognition test.
    shadow=np.zeros_like(image); valid=zz>-900
    for distance in (2.,4.,7.):
        sx=int(np.rint(key[0]*distance)); sy=int(np.rint(-key[1]*distance))
        height=np.roll(zz,(-sy,-sx),(0,1))-zz-distance*pix*key[2]
        dark=np.clip((height-.30)/1.10,0.,1.)
        if sx>0: dark[:,-sx:]=0
        if sx<0: dark[:,:-sx]=0
        if sy>0: dark[-sy:]=0
        if sy<0: dark[:-sy]=0
        shadow+=dark
    image*=1-u(.10,.34)/3*shadow*valid
    return image.reshape(28,2,28,2).mean((1,3))


# Descriptor shared between the procedural template encoder and the network.
# Four unsigned orientation bins, two signed derivatives, center-surround.
def kernels():
    y,x=np.mgrid[-2:3,-2:3]; g=np.exp(-(x*x+y*y)/(2*.90**2)); g/=g.sum()
    dx=x*g/(x*x*g).sum(); dy=dx.T
    dog=g-np.exp(-(x*x+y*y)/5.8)/np.exp(-(x*x+y*y)/5.8).sum()
    dog/=np.abs(dog).sum()/2
    return np.array([dx,dy,(dx+dy)/np.sqrt(2),(dx-dy)/np.sqrt(2),dog],np.float32)


def frontend():
    layers=[]; weights={}
    def layer(name,op,src,**kw):layers.append(dict(name=name,op=op,inputs=[src],**kw));return name
    def conv(name,src,W,b):
        weights[name+'.W']=np.asarray(W,dtype='f'); weights[name+'.b']=np.asarray(b,dtype='f'); k=W.shape[-1]
        return layer(name,'conv',src,out=W.shape[0],kernel=[k,k],stride=1,padding=0)
    src=layer('exposure','norm','image'); ks=kernels(); K=np.concatenate([ks,-ks])[:,None]
    a=conv('contrast',src,K,np.zeros(10)); a=layer('half_wave','relu',a)
    # Soft triangular steerable orientation bins, including a little energy
    # from neighboring orientations. Keep signed contrasts for light/shadow.
    mix=np.zeros((8,10),'f'); orth=[1,0,3,2]
    for j in range(4):
        mix[j,j]=mix[j,j+5]=1.; mix[j,orth[j]]=mix[j,orth[j]+5]=-.78
    mix[4,0]=1; mix[4,5]=-1; mix[5,1]=1; mix[5,6]=-1; mix[6,4]=1; mix[7,9]=1
    # Clip high contrasts so a background seam cannot swamp a finger edge.
    # For the two signed channels clip symmetrically to [-t,t].
    limits=np.array([.72]*4+[.89,.89,.93,.93],'f'); lower=np.array([0]*4+[-.89,-.89,0,0],'f')
    b=conv('orientation',a,np.concatenate([mix,mix])[:,:,None,None],np.r_[-lower,-limits]); b=layer('orientation_gate','relu',b)
    identity=np.eye(8,dtype='f'); b=conv('bounded',b,np.c_[identity,-identity][:,:,None,None],lower)
    b=layer('spatial_bins','avgpool',b,size=[4,4],stride=[2,2]); b=layer('contrast_maps','norm',b)
    # A bounded, zero-centered local-contrast channel gives intact, flat
    # background regions low evidence without requiring a skin threshold.
    fg=conv('broad_contrast',src,np.full((1,1,5,5),1/25,dtype='f'),[0.])
    fg=conv('broad_bounds',fg,np.ones((2,1,1,1),'f'),[1.,-1.]); fg=layer('broad_gate','relu',fg)
    fg=conv('broad_bounded',fg,np.array([1.,-1.]).reshape(1,2,1,1),[-1.]); fg=layer('broad_bins','avgpool',fg,size=[4,4],stride=[2,2]); fg=layer('broad_map','norm',fg)
    def concatenate_maps():
        # Block-identity 1x1 projections implement channel concatenation.
        left=np.zeros((9,8,1,1),'f'); left[:8,:,0,0]=np.eye(8); right=np.zeros((9,1,1,1),'f'); right[-1]=1.
        l=conv('gradient_slots',b,left,np.zeros(9)); r=conv('shading_slot',fg,right,np.zeros(9)); layers.append(dict(name='mixed_maps',op='add',inputs=[l,r]))
    concatenate_maps()
    return layers,weights,ks,mix,limits,lower


def encode(images,ks,mix,hi,lo):
    # Every reduction involving pixel data is strictly within ONE template.
    def norm(a,axes):
        c=a-a.mean(axes,keepdims=True); return c/np.sqrt((c*c).mean(axes,keepdims=True)+1e-5)
    im=norm(images,(1,2)); win=np.lib.stride_tricks.sliding_window_view(im,(5,5),(1,2)); v=win.reshape(-1,25)@ks.reshape(5,25).T
    v=np.maximum(np.c_[v,-v],0)@mix.T; v=np.clip(v,lo,hi).reshape(-1,24,24,8)
    cells=np.lib.stride_tricks.sliding_window_view(v,(4,4),(1,2))[:,::2,::2].mean((-1,-2)); desc=norm(cells,(1,2)).transpose(0,3,1,2)
    # Mostly polarity-independent oriented shape, with secondary signed
    # gradient and center/surround cues. Values chosen by design, not by test.
    broad=win.mean((-1,-2)).clip(-1.,1.)
    broad=np.lib.stride_tricks.sliding_window_view(broad,(4,4),(1,2))[:,::2,::2].mean((-1,-2)); broad=norm(broad,(1,2))
    desc=np.concatenate([desc,broad[:,None]],axis=1)
    gains=np.array([.166]*4+[.079,.079,.053,.053,.072],dtype='f')
    # The receptive fields overlap the outer margin but the matching kernel
    # itself leaves one feature cell free for searching in each direction.
    desc=norm(desc[:,:,1:-1,1:-1],(2,3))
    return desc*(gains[None,:,None,None]/81.)


def main(folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    layers,weights,ks,mix,hi,lo=frontend(); nbank=4; assert COUNT%nbank==0
    desc=np.empty((len(ABC)*COUNT,9,9,9),'f'); batch=32
    for j,letter in enumerate(ABC):
        rng=np.random.Generator(np.random.PCG64(SEED+1009*j))
        for start in range(0,COUNT,batch):
            n=min(batch,COUNT-start); images=np.stack([render(letter,rng) for _ in range(n)])
            desc[j*COUNT+start:j*COUNT+start+n]=encode(images,ks,mix,hi,lo)
    # Correlation in a 3x3 search window aligns each template; the spatial
    # stride of the earlier pooling layer is two input pixels per step.
    # The reshape makes translations an axis of the pool without duplicating
    # any template weights.
    for j in range(24):
        tag='letter_'+ABC[j]; weights[tag+'.W']=desc[j*COUNT:(j+1)*COUNT]; weights[tag+'.b']=np.zeros(COUNT,'f')
        layers+=[dict(name=tag,op='conv',inputs=['mixed_maps'],out=COUNT,kernel=[9,9],stride=1,padding=0),dict(name=tag+'_group',op='reshape',inputs=[tag],shape=[nbank,COUNT//nbank,9])]
        # Pairwise max stages avoid a very large temporary for pooling.
        # Count is a power of two. First reduce the pose dimension in chunks.
        g=tag+'_group'
        for k in (8,8,COUNT//nbank//64):
            nm=g+'_reduce'; layers.append(dict(name=nm,op='maxpool',inputs=[g],size=[k,1],stride=[k,1]));g=nm
        layers+= [dict(name=tag+'_aligned',op='maxpool',inputs=[g],size=[1,9],stride=[1,9]),dict(name=tag+'_groups',op='reshape',inputs=[tag+'_aligned'],shape=[1,1,nbank]),dict(name=tag+'_score',op='avgpool',inputs=[tag+'_groups'],size=[1,nbank],stride=[1,nbank]),dict(name=tag+'_slot',op='dense',inputs=[tag+'_score'],out=24)]
        slot=np.zeros((24,1),'f');slot[j]=1.;weights[tag+'_slot.W']=slot;weights[tag+'_slot.b']=np.zeros(24,'f')
    layers.append(dict(name='letter',op='add',inputs=['letter_'+c+'_slot' for c in ABC]))
    (folder/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n'); np.savez(folder/'weights.npz',**weights)


if __name__=='__main__':main(sys.argv[1])
