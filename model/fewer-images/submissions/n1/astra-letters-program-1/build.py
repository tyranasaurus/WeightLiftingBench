"""Data-free builder for an analysis-by-synthesis static fingerspelling recognizer."""
import math
import sys
from pathlib import Path
import numpy as np

# Kept as source strings so that the sandboxed file has no dependencies.
FEATURES = '''import numpy as np


def smooth(v):
    a = np.pad(v, ((0,0),(1,1),(1,1)), mode='edge')
    b = (a[:,:,:-2]+a[:,:,2:]+2*a[:,:,1:-1])*.25
    return (b[:,:-2]+b[:,2:]+2*b[:,1:-1])*.25


def describe(pic):
    v = (.5*pic + .5*smooth(pic)).astype(np.float32)
    a = np.pad(v, ((0,0),(1,1),(1,1)), mode='edge')
    dx = (a[:,1:-1,2:]-a[:,1:-1,:-2])*.5
    dy = (a[:,2:,1:-1]-a[:,:-2,1:-1])*.5
    mag = np.sqrt(dx*dx+dy*dy+1.e-12)
    # Soft saturation preserves weak interior creases without letting noise
    # dominate the flat part of the frame.
    avg = mag.mean(axis=(1,2),keepdims=True)
    amp = np.sqrt(mag/(mag+.9*avg+.004))*mag/(mag+.44*avg+.003)
    theta = np.arctan2(dy,dx)
    hog = []
    for j in range(6):
        c = np.maximum(1-np.abs((theta*(6/np.pi)-j+3)%6-3),0)
        hog.append(smooth(smooth(amp*c)))
    # Linear inhibition of the other orientations improves selectivity but
    # allows a weaker match with an edge of slightly different orientation.
    h = np.stack(hog,axis=-1)
    h = h-.11*np.sum(h,axis=-1,keepdims=True)
    mass = smooth(smooth(amp))
    # Additional channel: interiors that differ from the background plane.
    bg = backdiff(v)
    return np.concatenate((h,(mass-mass.mean(axis=(1,2),keepdims=True))[:,:,:,None]*.48,bg[:,:,:,None]*.30),axis=-1).astype(np.float32)


def backdiff(v):
    # Use robust upper-side samples, where the hand least often fills both
    # borders. Interpolate lighting from left to right, rejecting a bad side.
    l = np.median(v[:,:19,:2].reshape(len(v),-1),axis=1)[:,None,None]
    r = np.median(v[:,:19,-2:].reshape(len(v),-1),axis=1)[:,None,None]
    weight = np.linspace(0,1,28,dtype=np.float32)[None,None,:]
    d = np.abs(v-(l+(r-l)*weight))
    d = d/(d+.65*np.mean(d,axis=(1,2),keepdims=True)+.017)
    return d-np.mean(d,axis=(1,2),keepdims=True)


def unit(v):
    return v/np.sqrt(np.sum(v*v,axis=-1,keepdims=True)+1.e-10)


def coarse(h):
    # Seven-by-seven, 4-pixel cells.
    v = h.reshape(len(h),7,4,7,4,8).mean(axis=(2,4))
    return unit(v.reshape(len(h),-1))
'''

PREDICT = '''

def predict(x):
    out = np.empty(len(x),np.int32)
    if len(x)==0:
        return out
    # Fixed-size GEMM even for single-image calls; a row's answer must not
    # depend on the shape-dependent floating point path of a BLAS library.
    bank = CONSTANTS['coarse']
    fine = CONSTANTS['fine']
    count = len(bank)//24
    keep = 12
    batch = 16
    offsets = ((0,0),(-3,0),(3,0),(0,-3),(0,3))
    rows = len(offsets)
    for start in range(0,len(x),batch):
        m = min(batch,len(x)-start)
        planes = describe(x[start:start+m,0])
        border = np.pad(planes,((0,0),(3,3),(3,3),(0,0)),mode='edge')
        q = np.zeros((batch,rows,392),np.float32)
        fi = np.empty((m,rows,14,14,8),np.float32)
        for v,(dy,dx) in enumerate(offsets):
            part = border[:,3+dy:31+dy,3+dx:31+dx]
            q[:m,v] = coarse(part)
            fi[:,v] = part.reshape(m,14,2,14,2,8).mean(axis=(2,4))
        sim = (q.reshape(batch*rows,392)@bank.T).reshape(batch,rows,24,count)
        for k in range(m):
            s = sim[k]
            s[1:] -= .012
            # Choose a single best alignment for each individual template.
            simk = s.max(axis=0)
            ix = np.argpartition(simk,-keep,axis=1)[:,-keep:]
            ids = ix+np.arange(24)[:,None]*count
            view = s.reshape(rows,-1)[:,ids].argmax(axis=0).reshape(-1)
            ts = fine[ids.reshape(-1)].astype(np.float32)*(1/127.)
            ce = np.einsum('nhwc,nhwc->nhw',ts,ts).reshape(len(ts),7,2,7,2).sum(axis=(2,4))
            # For the shortlisted poses, allow individual palm and finger
            # patches to move slightly relative to the global alignment.
            pad = np.pad(fi[k],((0,0),(1,1),(1,1),(0,0)),mode='edge')
            central = np.empty_like(ce)
            best = np.empty_like(ce)
            for v in range(rows):
                mask = np.flatnonzero(view==v)
                if len(mask)==0:
                    continue
                sub = ts[mask]
                energy = ce[mask]+.045
                for oy,ox in ((0,0),(-1,0),(1,0),(0,1),(0,-1)):
                    part = pad[v,1+oy:15+oy,1+ox:15+ox]
                    ue = np.sum(part*part,axis=2).reshape(7,2,7,2).sum(axis=(1,3))
                    dot = np.einsum('nhwc,hwc->nhw',sub,part).reshape(len(sub),7,2,7,2).sum(axis=(2,4))
                    patch = dot/np.sqrt(energy*(ue+.045))
                    if oy==0 and ox==0:
                        best[mask] = patch
                        central[mask] = patch
                    else:
                        best[mask] = np.maximum(best[mask],patch-.09)
            weight = ce/(ce+.10)+.14
            refined = np.sum(weight*(.64*best+.36*central),axis=(1,2))/np.sum(weight,axis=(1,2))
            score = .52*np.take_along_axis(simk,ix,axis=1)+.48*refined.reshape(24,keep)
            top = np.partition(score,-3,axis=1)[:,-3:]
            out[start+k] = np.argmax(.73*top.max(axis=1)+.27*top.mean(axis=1))
    return out
'''

# All hand dimensions below are in anatomical units (adult palm width ~4).
# Positive y: toward fingertips; +x: radial (thumb) side; +z: palmar.
letters='ABCDEFGHIKLMNOPQRSTUVWXY'
rad=np.pi/180

def camera(ax,ay,az):
    p,t,r=np.array((ax,ay,az))*rad
    a,b=np.cos(p),np.sin(p); c,d=np.cos(t),np.sin(t); e,f=np.cos(r),np.sin(r)
    return np.array([[e*c,e*d*b-f*a,e*d*a+f*b],[f*c,f*d*b+e*a,f*d*a-e*b],[-d,c*b,c*a]])

def geometry(ch,j,rng):
    # Joint angles (degrees) flex toward the palm. Curled fingers are
    # independent phalanges, so their knuckles remain visible edge-on.
    U=rng.uniform
    splay=np.array([5.,1.,-3.,-8.]); curl=np.tile([62.,87.,48.],(4,1))
    ext=[2.,3.,2.]; lax=[17.,15.,8.]
    th=[(1.83,1.75,1.03),(1.78,2.60,1.56),(.64,2.85,1.87)]
    yaw,pitch,roll=9.,-6.,-3.
    if ch=='A':
        th=[(1.97,2.00,.50),(2.09,2.98,.53),(2.11,3.83,.48)]
    elif ch=='B':
        curl[:]=ext;splay[:]=[0.,-.5,-1.,-2.5]
        th=[(1.90,1.7,1.0),(1.03,2.49,1.21),(-.16,2.65,1.19)]
    elif ch in 'CO':
        # Profile of the arch facing the thumb. The C varies in openness;
        # in O, the distal pad of the index meets the thumb.
        curl[:]=[30,45,34] if ch=='C' else [40,56,38]
        curl[:,0]+=np.array([0.,1.,3.,5.]);splay[:]=[0,-1,-2,-4]
        th=[(1.69,1.30,1.27),(1.60,1.73,2.50),(1.42,2.28,3.18)] if ch=='C' else [(1.77,1.57,1.23),(1.67,2.30,2.25),(1.49,2.84,2.94)]
        yaw=68.;roll=-7.;pitch=0
    elif ch in 'DF':
        if ch=='D':
            curl[:]=[45.,60.,36.];curl[0]=ext
            th=[(1.85,1.78,1.24),(1.33,2.51,2.10),(.47,2.96,2.58)]
            yaw=38.;splay[0]=2.
        else:
            curl[:]=ext;curl[0]=[49.,63.,38.];splay[:]=[8.,-2.,-12.,-25.]
            th=[(2.18,1.78,1.13),(2.34,2.47,2.05),(1.97,2.95,2.72)]
            yaw=21.
    elif ch=='E':
        curl[:]=[30.,100.,64.]; splay[:]=[3,0,-2,-5]
        th=[(1.88,1.62,.92),(.94,1.92,1.34),(-.19,2.19,1.47)]
    elif ch in 'GH':
        curl[0]=ext
        if ch=='H':curl[1]=ext;splay[:2]=[1.,0.]
        th=[(1.8,1.35,1.49),(1.78,2.32,2.17),(1.65,3.42,2.49)]
        yaw=65.; roll=-78.; pitch=-3.
    elif ch in 'IY':
        curl[3]=ext;splay[3]=-17. if ch=='I' else -29.
        if ch=='Y':th=[(2.02,1.40,.67),(2.94,2.03,.86),(3.77,2.75,.89)]
        yaw=3.;roll=-4.
    elif ch in 'KP':
        curl[0]=ext;curl[1]=[52.,5.,2.];splay[:2]=[4.,-12.]
        th=[(1.9,1.85,1.01),(1.34,2.83,1.39),(.60,3.54,1.62)]
        if ch=='K': yaw=37.;roll=6.
        else: yaw=59.;roll=124.;pitch=1.
    elif ch=='L':
        curl[0]=ext;splay[0]=2.;yaw=0
        th=[(2.04,1.47,.60),(3.28,1.74,.71),(4.36,1.93,.72)]
    elif ch in 'MNST':
        if ch=='S':
            th=[(2.02,1.97,.96),(1.28,2.97,1.58),(.13,3.10,1.85)]
        else:
            gap={'M':-1.02,'N':.03,'T':1.10}[ch]
            curl[:]=[45 if ch!='T' else 50,83,52]
            th=[(1.74,1.79,.8),((gap+1.20)/2,2.92,1.30),(gap,3.85,1.41)]
            if ch=='T':curl[0]=[19.,95.,61.];curl[1]=[57.,89.,54.]
            if ch in 'MN':curl[:3 if ch=='M' else 2,0]-=9.
            yaw=18.
    elif ch=='Q':
        curl[0]=lax;splay[0]=8.
        th=[(2.08,1.32,1.08),(2.48,2.03,1.89),(2.59,3.04,2.41)]
        yaw=58.;pitch=5.;roll=153.
    elif ch in 'RUVW':
        curl[:2]=ext
        if ch=='R':
            splay[:2]=[-13.,11.];curl[0]=[11.,8.,2.];curl[1]=[2.,3.,3.]
        if ch=='U': splay[:2]=[-1.,0.]
        if ch=='V': splay[:2]=[16.,-14.]
        if ch=='W': curl[2]=ext;splay[:3]=[21.,0.,-19.]
        th=[(1.86,1.78,1.11),(.95,2.42,1.52),(-.18 if ch!='W' else -.83,2.65,1.74)]
        roll=-5.
    elif ch=='X':
        curl[0]=[9.,64.,52.];splay[0]=4.;yaw=58.
    # Shared camera variability, not placements from example pictures.
    yaw+=U(-24,24); pitch+=U(-17,17);roll+=U(-16,16)
    if ch in 'GPQH':roll+=U(-17,17)
    if ch in 'GH':
        # Alternate projection: index sideways, folded fingers and thumb
        # lower in the picture than the index. The back of the hand may show.
        if j%4<3:yaw+=115.
        pitch+=U(-25,20)
        if ch=='H':th=[(1.85,1.75,.94),(1.55,2.42,1.41),(.68,2.72,1.73)]
    if ch=='P':
        # K with the wrist flexed: the index extends outward and the
        # middle finger points toward the ground, with strong foreshortening.
        if j%3!=0:roll-=U(88,143);pitch+=U(55,100);yaw-=U(35,70)
    if ch=='Q' and j%3!=0:pitch+=U(15,55);yaw+=U(0,30)
    if ch=='K' and j%2==0:yaw+=U(-90,-45)
    if ch in 'CTOX' and j%5==0:yaw+=U(17,39)
    if ch in 'MNESTA' and j%3==0:pitch-=U(10,27);yaw+=U(-12,15)
    if ch in 'CFO' and j%3==0:pitch+=U(10,30)
    cam=camera(pitch,yaw,roll)
    # Each ellipsoid is defined by a centre and a basis of radii.
    parts=[]
    def bulb(c,axes,albedo=1.):parts.append((np.array(c,dtype=float),np.array(axes,dtype=float),albedo))
    def egg(c,r,a=1.):bulb(c,np.diag(r),a)
    def capsule(p,q,r0,r1,aspect=1.):
        # Consecutive rounded pads overlap, giving a soft crease at each joint.
        p=np.array(p);q=np.array(q);d=q-p;l=np.linalg.norm(d);n=d/l
        v=np.cross(n,[0,0,1.]);v/=max(np.linalg.norm(v),.001);w=np.cross(n,v)
        basis=np.stack([r0*v,r0*aspect*w,(l*.5+r1)*n],axis=1)
        bulb((p+q)*.5,basis)
    girth=U(.93,1.09)
    egg((0,1.43,-.04),(2.03*girth,2.52,.69))
    egg((.00,2.65,.00),(1.99*girth,1.34,.70))
    egg((1.04,1.18,.29),(.94,1.68,.65));egg((-.99,1.13,.13),(.96,1.76,.63))
    base_x=np.array([1.41,.475,-.47,-1.35])*girth
    base_y=[3.05,3.25,3.13,2.76]; length=np.array([3.48,3.83,3.56,2.81])*U(.92,1.09)
    width=np.array([.48,.50,.47,.41])*girth
    flexnoise=U(-6,6)
    for k in range(4):
        p=np.array([base_x[k],base_y[k],.09]);lng=length[k]*U(.97,1.035);w=width[k]
        al=splay[k]+U(-2.4,2.4); bend=np.cumsum(curl[k]+[flexnoise+U(-4,4),U(-6,6),U(-5,5)])*rad
        if ch in 'BUH' and k<2:al=splay[k]+U(-1.2,1.2)
        for m,(f,wr) in enumerate(zip((.48,.30,.22),(1.,.92,.82))):
            b=bend[m];q=p+lng*f*np.array([np.sin(al*rad)*np.cos(b),np.cos(al*rad)*np.cos(b),np.sin(b)])
            capsule(p,q,w*wr,.66*w*wr,.92);p=q
    p=np.array([1.07,.68,.32]);tn=U(-.12,.12,3)
    for m,q in enumerate(th):
        q=np.array(q)+tn*(m+1)*.55
        capsule(p,q,[.72,.57,.49][m]*girth,[.36,.35,.38][m],.92);p=q
    # Project the hand. The forearm direction is independent of the finger
    # axes at the wrist (especially for the horizontally pointing letters).
    parts=[(cam@c,cam@b,a) for c,b,a in parts]
    # The wrist enters from below, or obliquely for lateral signs.
    side=ch in 'GHPQ'
    wa=U(16,58) if ch in 'GH' else U(-55,-18) if side else U(-20,16)
    wrist=cam@np.array([.0,-.34,-.06]);direction=np.array([-np.sin(wa*rad),-np.cos(wa*rad),-.16]);direction/=np.linalg.norm(direction)
    b1=np.array([direction[1],-direction[0],0]);b1/=np.linalg.norm(b1);b2=np.cross(direction,b1)
    boundary=np.array([(c[0]-np.linalg.norm(b[0]),c[0]+np.linalg.norm(b[0]),c[1]-np.linalg.norm(b[1]),c[1]+np.linalg.norm(b[1])) for c,b,a in parts]);x1=boundary[:,0].min();x2=boundary[:,1].max();y1=boundary[:,2].min();y2=boundary[:,3].max()
    parts.append((wrist+direction*3.40,np.stack((b1*1.31*girth,b2*.80,direction*4.5),axis=1),.98))
    cuffdist=U(1.75,4.4)
    if j%3==0:parts.append((wrist+direction*(cuffdist+3.8),np.stack((b1*1.62*girth,b2*.98,direction*4.2),axis=1),U(.32,.77)))
    return parts,(x1,x2,y1,y2)


def render(ch,j,rng,grid):
    U=rng.uniform
    parts,(x1,x2,y1,y2)=geometry(ch,j,rng)
    # Frame relative to the object's own anatomical envelope.
    span=max(x2-x1,y2-y1); zoom=U(.64,1.15); sc=span/(28*zoom)
    cx=(x1+x2)*.5+U(-.085,.085)*span
    top=y2+U(-.05,.25)*span
    ratio=U(.90,1.11)
    xx=cx+(grid-14)*sc*ratio;yy=top-grid*sc
    zbuf=np.full((len(grid),len(grid)),-100.,np.float32)
    bg=U(.24,.78); image=np.full(zbuf.shape,bg,np.float32)
    az=U(-.85,.55);el=U(.15,.94);light=np.array([az,el,U(.33,.91)]);light/=np.linalg.norm(light)
    ambient=U(.30,.54); diffuse=U(.31,.55);edge=U(.05,.21);skin=U(.81,1.10)
    for c,axes,alb in parts:
        # Analytic orthographic ray intersection. Smaller projected boxes
        # keep work proportional to object area, rather than primitive count.
        ex=np.linalg.norm(axes[0]);ey=np.linalg.norm(axes[1]);
        xlo=int(np.searchsorted(xx,c[0]-ex));xhi=int(np.searchsorted(xx,c[0]+ex,'right'))
        ylo=int(np.searchsorted(-yy,-c[1]-ey));yhi=int(np.searchsorted(-yy,-c[1]+ey,'right'))
        if xhi<=xlo or yhi<=ylo:continue
        inv=np.linalg.inv(axes@axes.T);ix=xx[None,xlo:xhi]-c[0];iy=yy[ylo:yhi,None]-c[1];p=inv[0,2]*ix+inv[1,2]*iy
        d=p*p-inv[2,2]*(inv[0,0]*ix*ix+2*inv[0,1]*ix*iy+inv[1,1]*iy*iy-1)
        iz=(-p+np.sqrt(np.maximum(d,0)))/inv[2,2];z=iz+c[2];tile=zbuf[ylo:yhi,xlo:xhi];mask=(d>0)&(z>tile)
        nx=inv[0,0]*ix+inv[0,1]*iy+inv[0,2]*iz;ny=inv[0,1]*ix+inv[1,1]*iy+inv[1,2]*iz;nz=p+inv[2,2]*iz
        den=np.sqrt(nx*nx+ny*ny+nz*nz+1.e-12);dot=(nx*light[0]+ny*light[1]+nz*light[2])/den
        shade=alb*skin*(ambient+diffuse*np.maximum(dot,-.25)+edge*nz/den)
        np.copyto(tile,z,where=mask);np.copyto(image[ylo:yhi,xlo:xhi],shade,where=mask)
    # Contact and occlusion are darker immediately behind an adjacent pad.
    for dy,dx in ((-1,0),(0,-1)):
        front=np.roll(zbuf,(dy,dx),(0,1));image-=.054*np.clip((front-zbuf-.17)*1.7,0,1)*(zbuf>-99)
    return image.reshape(28,2,28,2).mean(axis=(1,3))


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(FEATURES+PREDICT)
    scope={};exec(FEATURES,scope)
    rng=np.random.default_rng(314159)
    n=2048;grid=(np.arange(56)+.5)*.5
    small=np.empty((24*n,392),np.float32);detail=np.empty((24*n,14,14,8),np.int8)
    for i,ch in enumerate(letters):
        for j in range(0,n,32):
            # No scoring, cross-image statistics or class aggregation. One
            # independently coded template per rendering, in original order.
            pics=np.stack([render(ch,k,rng,grid) for k in range(j,j+32)])
            h=scope['describe'](pics);a=i*n+j;b=a+len(pics)
            small[a:b]=scope['coarse'](h)
            fine=h.reshape(len(h),14,2,14,2,8).mean(axis=(2,4))
            detail[a:b]=np.rint(np.clip(fine,-1,1)*127).astype(np.int8)
    np.savez(out/'constants.npz',coarse=small,fine=detail)

if __name__=='__main__':main()
