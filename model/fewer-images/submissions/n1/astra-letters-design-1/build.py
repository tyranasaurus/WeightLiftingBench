"""Compile an analytic hand-and-camera model into a correlation network.

No assets. Coordinates in the articulation functions are in cm, referred to
an upright right palm, x toward the thumb, y up, z toward the observer.
"""
import json
import math
import os
import sys
import numpy as np

LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'
NPOSE = 2048
NGROUP = 8
rng = np.random.default_rng(697421)
rad = np.deg2rad

def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1.e-10)

def rx(a):
    c,s=np.cos(rad(a)),np.sin(rad(a))
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def ry(a):
    c,s=np.cos(rad(a)),np.sin(rad(a))
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rz(a):
    c,s=np.cos(rad(a)),np.sin(rad(a))
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

class Hand:
    # A list of ellipsoids and tapered, round-ended segments. All geometry,
    # including finger joints hidden from the camera, is specified in 3-D.
    def __init__(self):
        self.parts=[]
    def oval(self,p,r,basis=None,tone=1,group=0,crop=True):
        if basis is None: basis=np.eye(3)
        self.parts.append(('o',np.array(p),basis*np.array(r),tone,group,crop))
    def link(self,a,b,r,s,tone=1,group=0,crop=True):
        self.parts.append(('l',np.array(a),np.array(b),r,s,tone,group,crop))
    def digit(self,points,radii,group):
        # The middle cushion of each phalanx bulges a little; this and the
        # differences at the joint create transverse relief on flexed digits.
        for j,(a,b) in enumerate(zip(points,points[1:])):
            a,b=np.asarray(a),np.asarray(b)
            r,s=radii[j:j+2]
            self.link(a,b,r,s,1+.014*j,group)
            t=b-a; along=unit(t)
            side=unit(np.cross(along,np.array([1.,0,0])))
            if np.linalg.norm(side)<.5:side=unit(np.cross(along,[0,1,0]))
            ortho=np.cross(along,side)
            self.oval(.48*a+.52*b,[.51*(r+s),.51*(r+s),.53*np.linalg.norm(t)+.18*s],np.column_stack([side,ortho,along]),1+.014*j,group)

def articulation(k,variant):
    h=Hand()
    # Index through little finger; ordinary adult hand proportions.
    bases=np.array([[2.85,3.05,.05],[.91,3.35,.02],[-1.00,3.10,.05],[-2.76,2.32,.22]])
    lengths=np.array([[3.85,2.38,1.85],[4.36,2.63,1.95],[4.08,2.52,1.83],[3.12,1.86,1.60]])
    widths=np.array([.98,1.00,.94,.81])*rng.uniform(.94,1.07)
    lengths *= rng.uniform(.94,1.07,size=(4,1))*rng.uniform(.94,1.07)
    bases[:,0]*=rng.uniform(.97,1.05)
    h.oval([-.05,-.45,-.02],[3.90,4.95,1.30])
    h.oval([.02,1.88,-.06],[3.91,2.34,1.18])
    h.oval([1.75,-1.14,.62],[1.78,3.06,1.04],rz(-14),1.03)
    h.oval([-2.15,-.92,.53],[1.48,3.13,.84],rz(10),.97)

    # Flexion angles are relative at the metacarpophalangeal, proximal
    # interphalangeal and distal interphalangeal joints.
    tight=np.array([72.,94.,58.]); loose=np.array([52.,90.,64.]); straight=np.array([1.,3.,3.])
    angles=np.tile(tight,(4,1)); spreads=np.array([0.,0.,0.,0.]); ext=[]
    if k=='B': ext=[0,1,2,3]; spreads[:]=[0,0,1,1]
    if k=='C': angles[:]=[22,43,33]; spreads[:]=[2,0,-1,-3]
    if k=='D': ext=[0]; angles[1:]=[[38,66,48],[40,70,47],[36,71,54]]
    if k=='E': angles[:]=[30,103,76]
    if k=='F': ext=[1,2,3]; spreads[:]=[0,1,-6,-17]; angles[0]=[29,68,52]
    if k in 'GH': ext=[0] if k=='G' else [0,1]
    if k=='I': ext=[3]; spreads[3]=-8
    if k=='K': ext=[0,1]; spreads[:2]=[7,-15]
    if k=='L': ext=[0]; spreads[0]=2
    if k in 'MN':
        # M covers the distal thumb with three fingers, N with two. The
        # remaining digits close into the palm rather than around the thumb.
        ncover=3 if k=='M' else 2
        angles[:ncover]=[52,85,66]; angles[ncover:]=[82,100,63]
        spreads[ncover-1:ncover+1]=[1.5,-3.]
    if k=='O': angles[:]=[38,59,44]
    if k=='P': ext=[0,1]; spreads[:2]=[11,-22]
    if k=='Q': ext=[0]
    if k=='R': ext=[0,1]; spreads[:2]=[-15,12]
    if k=='S': angles[:]=[76,101,70]
    if k=='T': angles[:]=loose; angles[0]=[44,85,63];spreads[:2]=[7,-3]
    if k=='U': ext=[0,1]; spreads[:2]=[-1,1]
    if k=='V': ext=[0,1]; half=rng.uniform(9,17); spreads[:2]=[half,-half]
    if k=='W': ext=[0,1,2]; fan=rng.uniform(11,20);spreads[:3]=[fan,0,-fan]
    if k=='X': angles[0]=[2,78,45]
    if k=='Y': ext=[3]; spreads[3]=-22
    if k=='A': angles[:]=[75,93,63]
    if ext: angles[ext]=straight
    if k in 'KP': angles[1]=[rng.uniform(28,61),6,5]
    if k=='R': angles[0]=[-5,4,7]; angles[1]=[6,-1,3]
    if k in 'BQGUHLIVWY':
        # Straight fingers are not always locked.
        for i in ext: angles[i,0]+=rng.uniform(-6,8)
    spreads+=rng.normal(0,1.8,4)
    angles+=rng.normal(0,3.5,(4,3))+rng.normal(0,3,(1,3))
    allpts=[]
    for i in range(4):
        positions=[bases[i]]; bend=0; spread=rad(spreads[i]); axis=np.array([np.sin(spread),np.cos(spread),0])
        for a,l in zip(angles[i], lengths[i]):
            bend+=rad(a)
            direction=axis*np.cos(bend)+np.array([0,0,1])*np.sin(bend)
            positions.append(positions[-1]+l*direction)
        allpts.append(positions)
        h.digit(positions, widths[i]*np.array([1.01,.96,.85,.77]),i+1)
        h.link(bases[i]+[0,-1.75,-.12],bases[i],widths[i]*1.08,widths[i])
    # Thumb opposition. Special contacts refer to the moving fingers, not to
    # locations in a camera frame.
    p0=np.array([2.00,-2.92,.62]); p1=np.array([3.50,-.05,1.08]); p2=np.array([2.05,1.36,1.96]); p3=np.array([.26,1.66,2.13])
    if k=='A': p1,p2,p3=([3.65,-.17,.78],[4.11,1.95,1],[4.15,3.86,.98])
    if k=='B': p1,p2,p3=([3.18,-.17,1.26],[1.56,1.66,1.79],[-.14,2.36,1.75])
    if k=='C': p1,p2,p3=([3.60,-1.13,2.12],[3.90,-.86,4.31],[3.62,-.10,6.21])
    if k in 'DFO':
        idx={'D':1,'F':0,'O':0}[k]; goal=allpts[idx][-1]
        p3=goal+np.array([.17,-.87,-.12]);p2=p3+np.array([.59,-1.62,-.62]);p1=np.array([3.32,-.80,2.14])
        if k=='D':p2=p3+np.array([.95,-1.31,-.48])
    if k=='E': p1,p2,p3=([3.26,-.70,1.37],[1.64,-.02,2.17],[-.57,.18,2.40])
    if k in 'GLQY':
        if k=='L':p1,p2,p3=([3.94,-1.58,1.00],[6.01,-.85,1.05],[8.06,-.48,.97])
        if k=='Y':p1,p2,p3=([3.78,-1.08,.64],[5.47,.44,.77],[7.11,1.88,.83])
        if k in 'GQ':p1,p2,p3=([3.84,-1.15,1.99],[4.39,.60,3.44],[4.58,2.68,4.01])
    if k in 'KP': p1,p2,p3=([3.18,-.06,1.58],[2.43,2.14,2.08],[1.58,4.30,2.57])
    if k in 'MNT':
        slot={'M':-2,'N':0,'T':1.95}[k]
        p1,p2,p3=([2.43,-.05,1.62],[(2+slot)/2,1.65,2.22],[slot,3.43,2.62])
    if k=='S': p1,p2,p3=([3.48,.40,2.06],[2.27,2.07,3.20],[.33,2.49,3.89])
    forvis=[p0,np.asarray(p1)+rng.normal(0,.16,3),np.asarray(p2)+rng.normal(0,.19,3),np.asarray(p3)+rng.normal(0,.23,3)]
    h.digit(forvis,np.array([1.33,1.14,.95,.84]),5)
    return h

def camera(k,v):
    # Palm facing camera: right thumb is at viewer's right. Yaw toward the
    # radial edge shows the C/O loop in profile. Roll is positive CCW.
    yaw,pitch,roll=8.,6.,0.
    ys,ps,rs=17,11,12
    if k in 'AEMNST': yaw=18; pitch=13; ys=23; ps=17;rs=16
    if k in 'CO': yaw=75;pitch=5;roll=-13;ys=13;ps=13;rs=11
    if k in 'DF':yaw=25;ys=20
    if k in 'GHPQ':
        yaw=55;ys=23;pitch=2;ps=18
        if k=='G':roll=-71
        if k=='H':roll=-65
        if k in 'GH':
            # Look across the extended finger(s), with the palm facing
            # sideways or back and the index on top rather than underneath.
            yaw=126 if v%3 else 85; pitch=-13;ys=30;ps=19
        if k=='P':roll=64 if v%4 else 118; yaw=-20;pitch=60;ps=21;rs=19
        if k=='Q':roll=136; yaw=(-68 if v%3 else 48);rs=18
    if k in 'KRU':yaw=9;ys=20
    if k=='X':yaw=47;ys=22;roll=-5
    # T may also be made in a wrist-pronated fist, showing the projecting
    # thumb between the first two knuckles in near profile.
    if k=='T' and v%3==0: yaw=66;pitch=39;roll=-42;ys=20;ps=18;rs=20
    yaw+=np.clip(rng.normal(),-2,2)*ys; pitch+=np.clip(rng.normal(),-2,2)*ps;roll+=np.clip(rng.normal(),-2,2)*rs
    return rz(roll)@rx(pitch)@ry(yaw)

SS=2
# Supersampling grid lives in final image units; y increases down the image.
qy,qx=np.mgrid[:28*SS,:28*SS]; qx=(qx+.5)/SS;qy=(qy+.5)/SS

def render(k,v):
    hand=articulation(k,v); proj=camera(k,v);proj[1]*=-1
    proj=proj@np.diag([rng.uniform(.95,1.07),rng.uniform(.97,1.03),rng.uniform(.94,1.05)])
    parts=[]; lows=[];highs=[]
    def project(part):
        if part[0]=='l':
            _,a,b,r,s,col,g,crop=part
            a=proj@a;b=proj@b
            lo=np.minimum(a[:2]-r,b[:2]-s);hi=np.maximum(a[:2]+r,b[:2]+s)
            parts.append(('l',a,b,r,s,col,g))
        else:
            _,p,axes,col,g,crop=part
            p=proj@p;axes=proj@axes
            sz=np.sqrt((axes**2).sum(axis=1));lo=p[:2]-sz[:2];hi=p[:2]+sz[:2]
            parts.append(('o',p,np.linalg.inv(axes@axes.T),col,g))
        if crop: lows.append(lo);highs.append(hi)
    for part in hand.parts:project(part)
    wrist=np.array([0,-4.17,0]); wcam=proj@wrist
    # The forearm can bend at the wrist, independently of the fingers. Its
    # width increases away from the wrist; only the hand determines the crop.
    aim=np.array([rng.normal(-.11,.20),1.,-.32])
    if k in 'GH':aim=np.array([-.83+rng.normal(0,.21),.81,-.3])
    if k=='Q':aim=np.array([1.,rng.uniform(-.55,.38),-.24])
    if k=='P':aim=np.array([rng.uniform(-.35,.6),1.,-.26])
    end=wcam+22*unit(aim); r=rng.uniform(2.05,2.63)
    # flatten forearm slightly in depth and let it pass behind the palm
    wcam[2]-=.27;parts.insert(0,('l',wcam,end,r,1.55*r,.90,6))
    if rng.random()<.28:
        seam=rng.uniform(.20,.39); start=wcam+seam*(end-wcam)
        parts.insert(1,('l',start,end,r*1.23,r*1.70,rng.uniform(.49,.87),7))
    low=np.min(lows,axis=0);high=np.max(highs,axis=0);span=high-low
    scale= rng.uniform(21.0,27.7)/max(span)
    ctr=(low+high)/2
    ctr+=(np.array([rng.normal(0,1.47), rng.normal(1.05,1.18)]))/scale
    xx=((qx-14)/scale+ctr[0]).reshape(-1); yy=((qy-14)/scale+ctr[1]).reshape(-1)
    zbuf=np.full_like(xx,-1.e4);lum=np.zeros_like(xx);ids=np.full(xx.shape,-1,np.int8)
    light=unit(np.array([rng.uniform(-1.30,.25),rng.uniform(-1.15,-.30),rng.uniform(.30,1.0)]))
    for part in parts:
        if part[0]=='l':
            _,a,b,r,s,col,grp=part
            dx,dy,dz=b-a
            # Orthographic ray/segment nearest distance. For a varying radius
            # use the radius at the closest projected point on the centerline.
            ex=xx-a[0];ey=yy-a[1];den=dx*dx+dy*dy
            t0=np.clip((ex*dx+ey*dy)/(den+1.e-9),0.,1.)
            rr=r+(s-r)*t0; u=ex-t0*dx; w=ey-t0*dy
            d=rr*rr-u*u-w*w
            active=np.flatnonzero(d>0)
            if not active.size:continue
            # The frontmost intersection is on the cylinder or an end cap.
            # Maximize z(t) along the projected segment, clamping the solution.
            en=ex[active];eyy=ey[active]
            tt=(en*dx+eyy*dy)/(den+1.e-10)
            perp=(en-tt*dx)**2+(eyy-tt*dy)**2
            delta=np.sqrt(np.maximum(0,(rr[active]**2-perp)/(den+1.e-9))) * dz/np.sqrt(den+dz*dz+1.e-9)
            t=np.clip(tt+delta,0,1)
            ux=en-t*dx;uy=eyy-t*dy; rr=r+(s-r)*t
            uz=np.sqrt(np.maximum(0,rr*rr-ux*ux-uy*uy));z=a[2]+t*dz+uz
            intensity=(ux*light[0]+uy*light[1]+uz*light[2])/rr
        else:
            _,p,F,col,grp=part
            ex=xx-p[0];ey=yy-p[1];a=F[2,2];b=F[0,2]*ex+F[1,2]*ey
            c=F[0,0]*ex*ex+2*F[0,1]*ex*ey+F[1,1]*ey*ey-1
            d=b*b-a*c;active=np.flatnonzero(d>0)
            if not active.size:continue
            ez=(np.sqrt(d[active])-b[active])/a;z=p[2]+ez
            n=unit(np.stack((ex[active],ey[active],ez),axis=-1)@F);intensity=n@light
        visible=z>zbuf[active]; ii=active[visible]
        zbuf[ii]=z[visible];ids[ii]=grp;lum[ii]=col*(.45+.55*np.clip(intensity[visible],-.5,1))
    # Relative albedo and lighting are nuisance variables. Soft relief shadows
    # at depth breaks help preserve finger seams even under diffuse light.
    shp=qx.shape; z=zbuf.reshape(shp);mask=ids.reshape(shp)>=0
    shadow=np.zeros(shp)
    for dx,dy in [(SS,0),(-SS,0),(0,SS),(0,-SS)]:
        other=np.roll(z,(dy,dx),axis=(0,1)); shadow+=np.clip((other-z-.28)/1.65,0,1)
    fg=lum.reshape(shp)-.095*shadow
    # The reference background has no scene objects. Choose reflectance on
    # either side of the range of shaded skin, and some similar reflectances.
    bg=[.97,.23,.84,.66,.91,.35,.76,.58][v%8]+rng.uniform(-.06,.06)
    image=np.where(mask,fg,bg)
    return image.reshape(28,SS,28,SS).mean(axis=(1,3))

# Shared fixed feature operator. Half-wave rectified gradient and curvature
# filters, local gradient compression, then broad overlapping spatial bins.
def filters():
    y,x=np.mgrid[-2:3,-2:3]; g=np.exp(-(x*x+y*y)/(2*.83**2)); fs=[]
    for j in range(8):
        a=j*np.pi/4; u=x*np.cos(a)+y*np.sin(a); f=g*u;fs.append(f/np.sum(abs(f)))
    for j in range(4):
        a=j*np.pi/4; u=x*np.cos(a)+y*np.sin(a); f=g*(1-(u/.93)**2);f-=g*f.sum()/g.sum(); f/=abs(f).sum();fs.extend([f,-f])
    return np.array(fs)

FILT=filters(); C=len(FILT)

def metric(desc):
    # Mix each gradient bin with the opposite bin. Leave polarity partly
    # intact for relief, while giving a contour evidence for either polarity.
    # The other eight channels are already curvature (bar/groove) features.
    result=desc.copy(); result[:8]=.75*desc[:8]+.25*np.roll(desc[:8],4,axis=0)
    return result

def template(im):
    im=(im-im.mean())/np.sqrt(im.var()+1.e-5)
    win=np.lib.stride_tricks.sliding_window_view(im,(5,5))
    ans=(FILT.reshape(C,25) @ win.reshape(24*24,25).T).reshape(C,24,24)
    # A linear-ReLU spline approximates a saturating contrast response.
    pos=np.maximum(ans,0)-.68*np.maximum(ans-.22,0)-.25*np.maximum(ans-.68,0)
    # Overlapping 4x4 windows pool at stride two into an 11x11 grid.
    # One more smoothing stage produces a 10x10 grid.
    pooled=np.lib.stride_tricks.sliding_window_view(pos,(4,4),axis=(1,2))[:,::2,::2].mean(axis=(-1,-2))
    desc=np.lib.stride_tricks.sliding_window_view(pooled,(2,2),axis=(1,2)).mean(axis=(-1,-2))
    # Center separately in each orientation, then normalize the entire
    # descriptor. Every number here depends only on this one template.
    desc-=desc.mean(axis=(1,2),keepdims=True)
    desc=metric(desc)
    return metric(desc).reshape(-1)/np.sqrt(np.sum(desc**2)+1.e-9)

def main(path):
    os.makedirs(path,exist_ok=True);layers=[];weights={}
    def layer(name,op,inp,**kw):
        layers.append(dict(name=name,op=op,inputs=inp if isinstance(inp,list) else [inp],**kw));return name
    def linear(name,op,inp,W,b,**kw):
        weights[name+'.W']=np.asarray(W,dtype=np.float32);weights[name+'.b']=np.asarray(b,dtype=np.float32)
        return layer(name,op,inp,out=W.shape[0],**kw)
    n=layer('exposure','norm','image')
    # Three copies of each filter implement the spline's three hinges.
    n=linear('oriented','conv',n,np.tile(FILT[:,None],(3,1,1,1)),np.repeat([0.,-.22,-.68],C),kernel=[5,5],stride=1,padding=0)
    n=layer('half_waves','relu',n)
    mix=np.concatenate([np.eye(C),-.68*np.eye(C),-.25*np.eye(C)],axis=1)[:,:,None,None]
    n=linear('compress','conv',n,mix,np.zeros(C),kernel=[1,1],padding=0,stride=1)
    n=layer('bins','avgpool',n,size=[4,4],stride=[2,2]);n=layer('overlap','avgpool',n,size=[2,2],stride=[1,1])
    bank=np.empty((len(LETTERS)*NPOSE,C*100),np.float32)
    for j,k in enumerate(LETTERS):
        for v in range(NPOSE):bank[j*NPOSE+v]=template(render(k,v))
    n=linear('compare','dense',n,bank,np.zeros(bank.shape[0]))
    # Group maxima soften the maximum against accidental matches to clutter.
    # Only query scores are pooled; generated descriptors are never pooled.
    n=layer('pose_groups','reshape',n,shape=[24,NGROUP,NPOSE//NGROUP]);n=layer('group_winners','maxpool',n,size=[1,NPOSE//NGROUP],stride=[1,NPOSE//NGROUP]);m=layer('best_pose','maxpool',n,size=[NGROUP,1],stride=[NGROUP,1]);a=layer('support','avgpool',n,size=[NGROUP,1],stride=[NGROUP,1]);n=layer('evidence','add',[m,a]);layer('class_scores','reshape',n,shape=[24])
    with open(os.path.join(path,'network.json'),'w') as f:json.dump({'layers':layers},f,indent=1)
    np.savez(os.path.join(path,'weights.npz'),**weights)

if __name__=='__main__':main(sys.argv[1])
