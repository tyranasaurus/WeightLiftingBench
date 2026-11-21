"""Construct independent posed-hand templates. No data files are used by this builder."""
import sys
from pathlib import Path
import numpy as np

ABC = 'ABCDEFGHIKLMNOPQRSTUVWXY'
# Anatomy in palm-width units. +x is radial, +y distal, +z palmar.
rad = np.pi / 180
rng = np.random.default_rng(74931)


def vec(*v):
    return np.array(v, dtype=float)


def rx(a):
    c, s = np.cos(a*rad), np.sin(a*rad)
    return np.array(((1,0,0),(0,c,-s),(0,s,c)))


def ry(a):
    c, s = np.cos(a*rad), np.sin(a*rad)
    return np.array(((c,0,s),(0,1,0),(-s,0,c)))


def rz(a):
    c, s = np.cos(a*rad), np.sin(a*rad)
    return np.array(((c,-s,0),(s,c,0),(0,0,1)))


def drawparams(letter, k):
    normal = rng.normal
    # A flexion triplet holds metacarpophalangeal, proximal, distal angles.
    straight = np.array((1.,3.,3.)); curled = np.array((74.,96.,64.))
    hook = np.array((32.,91.,68.)); flat = np.array((71.,101.,37.))
    flex = np.tile(curled, (4,1)); spread = np.array((1.,1.,1.,1.))
    yaw, pitch, roll = 13.,-12.,-5.
    # Thumb path starting at the thenar eminence, then MCP, IP, tip.
    thumb = [[.30,.14,.02],[.47,.48,.26],[.30,.65,.46],[.07,.75,.48]]
    if letter=='A':
        flex[:] = [64,100,62]; thumb = [[.29,.15,0],[.52,.48,.13],[.55,.77,.14],[.54,1.01,.15]]
        yaw,pitch,roll=9,-20,-12
    if letter=='B':
        flex[:]=straight; spread[:]=[0,0,0,-1]; yaw,pitch,roll=9,-4,-3
    if letter in 'CO':
        # Four fingers arch toward the thumb; a side view exposes the aperture.
        flex[:]=[32,37,24] if letter=='C' else [47,55,33]
        flex += np.arange(4)[:,None]*np.array([3.,1.,0.]); yaw,pitch,roll=70,-5,-9
        thumb = [[.31,.13,.02],[.56,.41,.29],[.61,.59,.59],[.53,.81,.75]]
        if letter=='O':thumb[-1]=[.39,.84,.80]
    if letter=='D':
        flex[0]=straight; flex[1:]=[68,66,39]; spread[0]=0
        thumb=[ [.29,.14,.05],[.48,.53,.29],[.38,.78,.53],[.15,.91,.55]]
        yaw,pitch,roll=24,-5,-2
    if letter=='E':
        flex[:]=[38,103,66]; thumb=[ [.27,.15,0],[.46,.41,.22],[.23,.51,.36],[-.05,.54,.38]]
        yaw,pitch,roll=8,-15,-7
    if letter=='F':
        flex[:]=straight; flex[0]=[52,55,39]; spread[:]=[14,8,-3,-15]
        thumb=[ [.30,.15,.04],[.52,.47,.24],[.56,.74,.47],[.46,.95,.66]]
        yaw,pitch,roll=38,-11,5
    if letter=='G' or letter=='H':
        flex[0]=straight
        if letter=='H': flex[1]=straight; spread[1]=-1
        roll=-79; yaw=22 if letter=='G' else 50; pitch=-6
        thumb=[ [.27,.14,0],[.43,.47,.17],[.41,.76,.34],[.35,1.04,.39]]
    if letter=='I' or letter=='Y':
        flex[3]=straight; spread[3]=-19 if letter=='Y' else -11
        if letter=='Y':thumb=[ [.28,.12,0],[.59,.36,.09],[.81,.54,.10],[1.04,.71,.11]]
        yaw,pitch,roll=11,-15,-7
    if letter in 'KP':
        flex[0]=straight; flex[1]=[49,5,0]; spread[0]=1; spread[1]=-14
        thumb=[ [.27,.14,.03],[.39,.49,.24],[.22,.79,.38],[.11,.98,.38]]
        yaw,pitch,roll=-31,-16,8
        if letter=='P':yaw,pitch,roll=50,12,125
    if letter=='L':
        flex[0]=straight; thumb=[ [.28,.12,0],[.61,.38,.05],[.87,.51,.05],[1.13,.60,.07]]
        yaw,pitch,roll=5,-4,0
    if letter in 'MNT':
        n={'M':3,'N':2,'T':1}[letter]; flex[:]=flat
        flex[:n]=[51,95,49]; flex[n:]=[81,94,69]
        thumb=[ [.29,.15,0],[.41,.45,.20],[.40-.15*n,.69,.29],[.455-.244*n,.98,.31]]
        yaw,pitch,roll=16,-24,-15
        if letter=='T' and k%5!=0:
            # A side-on presentation: the index is hooked over the upright thumb.
            yaw,pitch,roll=80,-18,-11; flex[0]=[72,50,49]
    if letter=='Q':
        flex[0]=[23,20,4]; thumb=[[.26,.14,0],[.53,.46,.25],[.59,.72,.49],[.57,.93,.74]]
        yaw,pitch,roll=54,20,165
    if letter=='R':
        flex[:2]=straight; flex[0]=[8,6,0]; spread[:2]=[-14,7]; yaw,pitch,roll=16,-9,5
    if letter=='S':
        flex[:]=[74,102,57]; thumb=[[.28,.14,.02],[.42,.47,.27],[.29,.74,.45],[.07,.95,.43]]
        yaw,pitch,roll=15,-24,-16
    if letter=='U' or letter=='V' or letter=='W':
        flex[:2]=straight; spread[:2]=[0,0] if letter=='U' else [12,-13]
        if letter=='W':flex[2]=straight; spread[:3]=[18,0,-17]
        yaw,pitch,roll=8,-6,0
    if letter=='X':
        flex[0]=[10,67,52]; thumb=[[.28,.14,0],[.43,.5,.2],[.28,.69,.42],[.08,.79,.43]]
        yaw,pitch,roll=50,-12,-3
    # Side-facing pointing signs often show the dorsum; the folded digits
    # then lie below the pointing index, not above it. Use both rotations of
    # the forearm, as well as a lateral view of the stacked fingers in H.
    if letter=='G' and k%4!=0:yaw=154
    if letter=='H':yaw=(106,135,160,65)[k%4]
    if letter=='T' and k%5!=0:
        # Alternative T convention, with the index crossing a vertical thumb.
        yaw,pitch,roll=83,-6,-5
        flex[0]=[87,6,8];thumb=[[.28,.14,0],[.46,.49,.08],[.46,.87,.13],[.46,1.16,.17]]
    if letter=='F':
        # Viewed obliquely from the thumb side, index and thumb form a
        # loop; the three raised digits may fan behind it in depth.
        yaw=52;spread[1:]=[15,-6,-25];flex[1:]=[[10,8,6],[1,7,5],[-10,5,7]]
    if letter=='C' and k%2==0:flex[:]=[23,35,19]
    if letter=='O' and k%2==0:flex[:]=[36,58,38]
    if letter=='P' and k%3!=0:yaw,pitch,roll=-53,17,111
    if letter=='Q' and k%3!=0:yaw,pitch,roll=110,15,157
    if letter in 'MNS' and k%2==0:
        # Pronating and flexing the wrist exposes the backs of the
        # proximal phalanges and steepens the arc of the knuckles.
        yaw+=19;pitch+=21;roll-=10
    if letter=='E' and k%2==0:flex[:]=[18,103,78]
    if k>=336:
        if letter in 'MNS' and k%4==0:
            yaw+=24;pitch+=12
        if letter=='E':
            # Counterpose the thumb along the palm beneath the tucked nails.
            thumb=[[.28,.15,.03],[.44,.47,.20],[.20,.57,.32],[-.09,.60,.34]]
            if k%2==0:flex[:]=[26,106,66]
        if letter=='A' and k%3==0:flex[:]=[76,104,72]
        if letter in 'DRU':
            # The raised digits stay together in U; in D a single digit
            # is free, and in R the index crosses the middle digit in depth.
            if letter=='R':spread[0],spread[1]=-18,8;flex[0]=[16,1,1];flex[1]=[-2,3,7]
            if letter=='D':flex[1:]=[52,74,49];yaw-=17
            if letter=='U':yaw+=17
        if letter=='K':
            flex[1]=[37,6,1];spread[1]=-19;yaw=24 if k%2==0 else -24
    if letter=='S':
        thumb=[[.28,.14,.03],[.46,.53,.34],[.22,.79,.58],[-.045,.88,.58]]
    if letter=='P' and k%4!=0:
        # Rotate K downward, with the extended middle finger flexing
        # perpendicular to the index. The forearm need not enter on the index axis.
        flex[1]=[80,4,3];spread[1]=-11;yaw,pitch,roll=68,-10,115
    if letter=='X' and k%3!=0:
        flex[0]=[3,78,49];yaw,pitch,roll=73,-4,-4
    if letter=='O' and k%3!=0:
        yaw,pitch,roll=86,-8,-7;flex[:]=[32,62,51]
    if letter in 'MN':
        # Close the palm over the thumb. The crossing fingers stop on the
        # thumb pad, while their neighbours can curl tightly into the palm.
        count=3 if letter=='M' else 2
        flex[:count]=[45,83,44];flex[count:]=[86,104,65]
        spread[:count]-=4;thumb[2][2]=.34;thumb[3][2]=.44
    # Low-discrepancy changes in viewing angle, with independent variation in the joints.
    def wave(a):return 2*((k*a+.5)%1)-1
    yaw += (26 if letter not in 'CO' else 23)*wave(.61803398875)
    pitch += 18*wave(.41421356237); roll += (20 if letter in 'GHPQ' else 15)*wave(.73205080757)
    flex+=np.clip(normal(0,1,(4,3)),-2,2)*np.where(flex>24,6.5,2.5)
    spread+=np.clip(normal(0,1,4),-2,2)*np.where(np.array([i<4 and letter=='B' for i in range(4)]),.6,2.0)
    if letter=='U':spread[0]=spread[1]+normal(0,1.)
    t=np.array(thumb); t[1:]+=normal(0,.027,(3,3)); t[1:,1]+=normal(0,.042)
    return flex, spread, t, rz(roll)@rx(pitch)@ry(yaw)


def model(letter,k):
    flex,spread,thumb,view=drawparams(letter,k)
    normal=rng.normal
    fat=np.clip(normal(1,.065),.83,1.2); height=np.clip(normal(1,.07),.82,1.18)
    digits=np.clip(normal(1,.06),.86,1.14); width=np.clip(normal(1,.07),.83,1.17)
    # Surface represented by convex quadric pieces; store as center and local shape matrix.
    pieces=[];landmarks=[]
    def ell(center,axes,orientation=None,shade=1):
        if orientation is None:orientation=np.eye(3)
        pieces.append((np.asarray(center),orientation*np.array(axes)[None,:],shade))
    ell([-.015,.50,-.045],[.50,.673,.184*fat])
    ell([-.019,.70,-.007],[.485,.405,.163*fat])
    # Thenar and hypothenar pads, slanted along the life line.
    ell([.227,.415,.079],[.221,.409,.158*fat],rz(-17))
    ell([-.275,.466,.026],[.167,.414,.14*fat],rz(12))
    def bone(a,b,r1,r2,depth=1,shade=1):
        a,b=np.asarray(a),np.asarray(b); u=b-a; length=np.linalg.norm(u);u=u/length
        v=np.cross(u,np.array([0,0,1.]) if abs(u[2])<.92 else np.array([1.,0,0]));v/=np.linalg.norm(v)
        w=np.cross(v,u); ori=np.stack([v,u,w],axis=1)
        # A smoothly varying tube with rounded ends, not a chain of bare bones.
        ell((a+b)*.5,[(r1+r2)*.518,(length*.5+min(r1,r2)*.65),(r1+r2)*.5*depth],ori,shade)
        ell(a,[r1,r1*.92,r1*depth],ori,shade)
        ell(b,[r2,r2*.95,r2*depth],ori,shade)
    roots=np.array([[.337,.964,-.001],[.103,1.015,.003],[-.131,.975,-.015],[-.342,.86,-.031]])
    lengths=np.array([[.421,.264,.200],[.467,.298,.217],[.433,.278,.204],[.34,.211,.168]])
    # Alternate skeletal constructions for signs where moving towards the
    # viewer matters as much as motion in the picture plane. All bearings
    # are relative to the hand, never pixel positions. In this model each
    # vector is a projected phalanx direction and a foreshortening factor.
    projected={}
    if k>=336 and letter in 'PQTGXHCO':
        if letter in 'GH':
            direction=0 if letter=='G' else rng.uniform(-7,9)
            projected[0]=([direction+6,direction+2,direction-2],[1.,1.,1.])
            if letter=='H':projected[1]=([direction+2,direction-2,direction-5],[.95,.96,.93])
            # Carpal flexion brings the hand over a horizontal forearm.
            view=rz(-35+rng.uniform(-16,16))@ry(rng.uniform(138,184))
        if letter=='T':
            view=rz(rng.uniform(-18,13))@rx(-9)@ry(164+rng.uniform(-20,18))
            pivot=roots[0];direction=rng.uniform(-13,10)
            projected[0]=([direction-3,direction,direction+5],[.99,.98,.98])
            thumb=np.array([[.28,.14,.01],[.42,.51,.22],[.35,.86,.35],[.29,1.18,.36]])
        if letter=='P':
            view=rz(58+rng.uniform(-14,16))@rx(22)@ry(-37)
            direction=160+rng.uniform(-11,12);pivot=roots[0]
            projected[0]=([direction+2,direction,direction-2],[1.,.99,.99])
            projected[1]=([-103,-98,-92],[.85,.82,.81])
        if letter=='Q':
            view=rz(28+rng.uniform(-14,19))@rx(-75+rng.uniform(-12,12))@ry(143+rng.uniform(-20,20))
            pivot=np.zeros(3);direction=-112+rng.uniform(-13,13)
            projected[0]=([direction-6,direction,direction+9],[.94,.96,.97])
            # The second prong of Q is the thumb, not a second long finger.
            pp=view@thumb[0];newthumb=[thumb[0]]
            for ang,dist in zip((-54,-68,-78),(.39,.29,.25)):
                a=(ang+rng.uniform(-9,9))*rad;pp=pp+vec(np.cos(a)*dist,np.sin(a)*dist,.10)
                newthumb.append(view.T@pp)
            thumb=np.array(newthumb)
        if letter=='X':
            view=rz(rng.uniform(-18,5))@ry(rng.uniform(25,58));pivot=roots[0]
            rot=rng.uniform(-14,14);projected[0]=(np.array([83,41,-45])+rot,[.99,.98,.86])
        if letter in 'CO':
            view=rz(rng.uniform(-12,0))@ry(rng.uniform(63,88));pivot=roots[0]
            for j in range(4):
                a=np.array([60,13,-36] if letter=='C' else [56,0,-62])+rng.uniform(-9,9)
                projected[j]=(a,[.94,.95,.96])
            # Oppose the thumb to the arched fingertips, leaving C open.
            pp=view@thumb[0];newthumb=[thumb[0]]
            for ang,dist in zip((20,37,56) if letter=='C' else (17,44,83),(.41,.28,.25)):
                a=(ang+rng.uniform(-5,5))*rad;pp=pp+vec(np.cos(a)*dist,np.sin(a)*dist,.035)
                newthumb.append(view.T@pp)
            thumb=np.array(newthumb)
    # The rest arch of the knuckles and a very gentle convergence in a fist.
    for j in range(4):
        p=roots[j].copy(); ang=np.cumsum(flex[j])*rad; ab=spread[j]*rad
        taper=([.126,.119,.106,.096] if j==0 else [.132,.122,.111,.100] if j==1 else [.120,.112,.099,.092] if j==2 else [.104,.096,.086,.078])
        leng=lengths[j]*digits*np.clip(normal(1,.024),.93,1.07)
        for e in range(3):
            u=vec(np.sin(ab)*np.cos(ang[e]),np.cos(ab)*np.cos(ang[e]),np.sin(ang[e]))
            if j in projected:
                angles,cosines=projected[j];angle=angles[e]*rad;co=cosines[e]
                u=view.T@vec(co*np.cos(angle),co*np.sin(angle),np.sqrt(max(0,1-co*co)))
            p2=p+u*leng[e];bone(p,p2,taper[e]*fat,taper[e+1]*fat,.90,1.0-.015*e);p=p2
            if e!=1:landmarks.append(p.copy())
            if e==2:
                # Distal dorsal nail plate, on the opposite side from the pad.
                dorsal=vec(np.sin(ab)*np.sin(ang[e]),np.cos(ab)*np.sin(ang[e]),-np.cos(ang[e]))
                tangent=vec(np.cos(ab),-np.sin(ab),0)
                if j in projected:
                    dorsal=view.T@np.array([0.,0.,-1.]);dorsal-=u*(dorsal@u);dorsal/=np.linalg.norm(dorsal);tangent=np.cross(u,dorsal)
                ell(p-u*.065+dorsal*taper[3]*fat*.78,[taper[3]*.68,.086,.024],np.stack([tangent,u,dorsal],axis=1),1.16)
    landmarks += [thumb[3],thumb[2]]
    for i in range(3):bone(thumb[i],thumb[i+1],(.183,.145,.123)[i]*fat,(.145,.123,.112)[i]*fat,.91)
    transform=view@np.diag([width,height,1.]); transformed=[]
    for c,b,s in pieces:
        c=transform@c; b=transform@b; transformed.append((c,b@b.T,s))
    # Frame the photographed hand including some wrist, independently of all other poses.
    xy=np.array([t[0][:2] for t in transformed]); r=np.sqrt(np.array([np.diag(t[1])[:2] for t in transformed]))
    lo=(xy-r).min(0);hi=(xy+r).max(0)
    wrist=view@vec(0,-.27*height,-.045);lo=np.minimum(lo,wrist[:2]);hi=np.maximum(hi,wrist[:2])
    center=(lo+hi)*.5;span=max(hi-lo);side=span*np.clip(normal(1.20,.10),.98,1.45)
    center+=span*normal(0,.048,2)
    # The forearm can approach the sign from below even when the wrist is flexed.
    wrist0=vec(-.04,-.105,-.055); axis=np.array([0.,-1.,0.]); world=view@axis
    if letter in 'GH':world=vec(-.87,-.51,-.05)
    elif letter=='P':world=vec(.92,-.38,-.07)
    elif letter=='Q':world=vec(-.38,-.78,-.50)
    else:world=vec(-.24+normal(0,.12),-1.,-.17)
    u=view.T@world;u/=np.linalg.norm(u);wrist1=wrist0+u*.58;wrist2=wrist0+u*2.2
    pieces=[]
    bone(wrist0,wrist1,.30*fat,.29*fat,.77,.96);bone(wrist1,wrist2,.29*fat,.37*fat,.79,.94)
    for c,b,s in pieces:
        c=transform@c;b=transform@b;transformed.append((c,b@b.T,s))
    lm=np.stack(landmarks)@transform.T
    lm=(lm[:,:2]-center)*np.array([28.,-28.])/side+13.5
    return transformed,center,side,lm


def render(letter,k,res=44):
    parts,center,side,lm=model(letter,k)
    unit=side/res; xs=center[0]+unit*(np.arange(res)+.5-res/2);ys=center[1]-unit*(np.arange(res)+.5-res/2)
    zbuf=np.full((res,res),-10.); gray=np.zeros((res,res))
    light=vec(-.77+rng.normal(0,.24),.64+rng.normal(0,.24),.60+rng.normal(0,.20));light/=np.linalg.norm(light)
    bg=np.clip(rng.normal(.64,.17),.23,.94); albedo=np.clip(rng.normal(.74,.10),.48,.94)
    for c,q,s in parts:
        a,b,d=q[0,0],q[0,1],q[1,1]; det=a*d-b*b;ix,ixy,iy=d/det,-b/det,a/det
        rl,rt=np.sqrt(a),np.sqrt(d)
        j0=max(0,int(np.ceil((c[0]-rl-xs[0])/unit)));j1=min(res,1+int((c[0]+rl-xs[0])/unit))
        i0=max(0,int(np.ceil((ys[0]-c[1]-rt)/unit)));i1=min(res,1+int((ys[0]-c[1]+rt)/unit))
        if i0>=i1 or j0>=j1:continue
        dx=xs[None,j0:j1]-c[0];dy=ys[i0:i1,None]-c[1]; gx=ix*dx+ixy*dy;gy=ixy*dx+iy*dy
        v=1-dx*gx-dy*gy;p=q[2,0]*ix+q[2,1]*ixy;t=q[2,0]*ixy+q[2,1]*iy
        h=np.sqrt(max(1.e-12,q[2,2]-p*q[2,0]-t*q[2,1]));sv=np.sqrt(np.maximum(v,0))
        z=c[2]+p*dx+t*dy+h*sv
        dest=zbuf[i0:i1,j0:j1];mask=(v>0)&(z>dest)
        nx=h*gx-p*sv;ny=h*gy-t*sv;nz=sv
        lum=(nx*light[0]+ny*light[1]+nz*light[2])/np.sqrt(np.maximum(1.e-10,nx*nx+ny*ny+nz*nz))
        g=gray[i0:i1,j0:j1];g[mask]=((.32+.63*np.maximum(0,lum)+.10*lum)*albedo*s)[mask];dest[mask]=z[mask]
    fg=zbuf>-9
    # Contact shadow at depth discontinuities and in the webbing of the hand.
    zp=np.pad(zbuf,2,mode='edge'); shade=np.zeros_like(gray)
    for dy,dx in ((0,-2),(-2,0),(-1,-1),(1,1)):
        other=zp[2+dy:2+dy+res,2+dx:2+dx+res];shade+=np.clip((other-zbuf-.045)*1.9,0,1)
    gray[fg]*=(1-.095*shade[fg]); gray[~fg]=bg
    # Area resampling, performed on this render alone.
    grid=np.linspace(0,res,29);i=np.arange(res)[None,:]; w=np.maximum(0,np.minimum(grid[1:,None],i+1)-np.maximum(grid[:-1,None],i))*28/res
    high=gray.copy(); high[~fg]=max(.67,bg)
    return (w@gray@w.T).astype(np.float32), (w@fg.astype(float)@w.T).astype(np.float32), (w@high@w.T).astype(np.float32), lm


descriptors = '''import numpy as np

def smooth(a):
    v = np.pad(a, ((0,0),(1,1),(0,0)), mode="edge")
    v = (v[:,:-2] + 2*v[:,1:-1] + v[:,2:]) * .25
    w = np.pad(v, ((0,0),(0,0),(1,1)), mode="edge")
    return (w[:,:,:-2] + 2*w[:,:,1:-1] + w[:,:,2:]) * .25

def diff(a):
    q=np.pad(a,((0,0),(1,1),(1,1)),mode="edge")
    return (q[:,1:-1,2:]-q[:,1:-1,:-2])*.5,(q[:,2:,1:-1]-q[:,:-2,1:-1])*.5

def pooling(n):
    t=(np.arange(28,dtype=np.float32)+.5)*(n/28)-.5
    p=np.maximum(1-np.abs(t-np.arange(n,dtype=np.float32)[:,None]),0)
    return p/p.sum(1,keepdims=True)

pool7=pooling(7)
pool10=pooling(10)

def cells(v,p):
    return (p@v@p.T).reshape(len(v),-1)

def norm(v):
    return v/np.sqrt((v*v).sum(1,keepdims=True)+1.e-9)

def foreground(a):
    # Background is often a slowly changing wall. Interpolate from both sides
    # on each scan line and combine the residual with local texture evidence.
    v=smooth(a)
    t=(np.arange(28,dtype=np.float32)/27)[None,None,:]
    l=v[:,:,:2].mean(2,keepdims=True);r=v[:,:,-2:].mean(2,keepdims=True)
    # Large jumps between consecutive rows need not describe the background.
    u=v[:,:3,:4].mean((1,2),keepdims=True);w=v[:,:3,-4:].mean((1,2),keepdims=True)
    level=u*(1-t)+w*t
    dl=l-u;dr=r-w
    dl=np.where(np.abs(dl)<np.abs(dr),dl,np.clip(dl,dr-.11,dr+.11))
    dr=np.where(np.abs(dr)<np.abs(l-u),dr,np.clip(dr,l-u-.11,l-u+.11))
    back=level+dl*(1-t)+dr*t
    gx,gy=diff(v); mag=np.sqrt(gx*gx+gy*gy)
    texture=smooth(smooth(mag)); err=np.abs(v-back)
    signal=(err+.85*texture)/( .045+ .20*np.abs(l-r))
    mask=1-np.exp(-signal*signal*.53)
    tex=1-np.exp(-(texture/.034)**2)
    photometric=smooth(.71*mask+.29*tex)
    # Minimax barrier paths from the image frame. A patch that differs in
    # brightness from the distant wall can still be reached along a smooth
    # path, while the inside of a hand is often enclosed by a contrast edge.
    wall=np.ones_like(a)*.22
    wall[:,0,:]=0;wall[:,-1,:]=0;wall[:,:,0]=0;wall[:,:,-1]=0
    cost=(.65*a+.35*v)
    cx=np.abs(cost[:,:,1:]-cost[:,:,:-1]);cy=np.abs(cost[:,1:,:]-cost[:,:-1,:])
    for it in range(24):
        new=wall.copy()
        new[:,1:,:]=np.minimum(new[:,1:,:],np.maximum(cy,wall[:,:-1,:]))
        new[:,:-1,:]=np.minimum(new[:,:-1,:],np.maximum(cy,wall[:,1:,:]))
        new[:,:,1:]=np.minimum(new[:,:,1:],np.maximum(cx,wall[:,:,:-1]))
        new[:,:,:-1]=np.minimum(new[:,:,:-1],np.maximum(cx,wall[:,:,1:]))
        wall=new
    barrier=1-np.exp(-(wall/.031)**1.55)
    return .70*photometric+.30*smooth(barrier)

def features(a,m):
    a=np.asarray(a,dtype=np.float32);m=np.asarray(m,dtype=np.float32)
    # Compressive oriented energy and unsigned orientation channels are
    # insensitive to a reversal of contrast at the silhouette.
    s=smooth(a);gx,gy=diff(.6*s+.4*a); mag=np.sqrt(gx*gx+gy*gy+1.e-11)
    strength=mag/(.045+mag); power=np.sqrt(mag+1.e-8)*strength
    hog=[]
    for i in range(6):
        angle=i*np.pi/6
        cs=np.abs(gx*np.float32(np.cos(angle))+gy*np.float32(np.sin(angle)))/mag
        hog.append(cells(power*np.maximum(0,cs-.76)**2,pool7))
    h=norm(np.sqrt(np.maximum(0,np.concatenate(hog,1))+1.e-8))
    # Retain spatial variation, rather than correlating constant histograms.
    h=norm(h-.60*h.mean(1,keepdims=True))
    energy=cells(power,pool10);en=norm(energy-energy.mean(1,keepdims=True))
    mask=cells(m,pool10);mask=norm(mask-mask.mean(1,keepdims=True))
    # Bandpass shading makes the crease between digits useful, even if
    # parts of the hand have the same intensity as the wall.
    blur=s
    for j in range(5):blur=smooth(blur)
    band=s-blur;band=band/(.045+np.abs(band));bp=norm(cells(band,pool10))
    return np.concatenate((h*.70,en*.41,mask*.50,bp*.30),axis=1).astype(np.float32)
'''

finecode = '''
def curvefield(a):
    # Soft oriented chamfer fields. Each channel asks if an edge of the
    # specified normal passes nearby, irrespective of unrelated edges elsewhere.
    gx,gy=diff(.78*smooth(a)+.22*a);mag=np.sqrt(gx*gx+gy*gy+1.e-12)
    response=mag/(mag+.023)
    channels=[]
    for i in range(6):
        angle=i*np.pi/6
        direction=np.abs(gx*np.float32(np.cos(angle))+gy*np.float32(np.sin(angle)))/mag
        e=response*direction**6; base=e
        for j in range(2):
            p=np.pad(e,((0,0),(1,1),(1,1)),mode="edge")
            e=np.maximum(e,np.maximum(np.maximum(p[:,:-2,1:-1],p[:,2:,1:-1]),np.maximum(p[:,1:-1,:-2],p[:,1:-1,2:]))*.75)
        e=.75*e+.25*base
        channels.append(e.reshape(len(e),-1))
    return np.concatenate(channels,1)
'''

localcode='''
def localcontrast(a):
    s=.6*smooth(a)+.4*a
    p=np.pad(s,((0,0),(2,2),(2,2)),mode="edge")
    pair=[]
    # Signed ordinal comparisons at two radii, along both cardinal and
    # diagonal directions. A concave knuckle, a fingertip and the edge of
    # a thumb crossing a finger have different patterns of these signs.
    for radius in (1,2):
        for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
            x=dx*radius;y=dy*radius
            left=p[:,2-y:30-y,2-x:30-x];right=p[:,2+y:30+y,2+x:30+x]
            linear=right-left;curve=s-(left+right)*.5
            pair.append((linear/(.055+np.abs(linear))).reshape(len(s),-1))
            pair.append((curve/(.032+np.abs(curve))).reshape(len(s),-1))
    return np.concatenate(pair,axis=1)
'''

def ordinal(image,lm,field):
    # A per-landmark winner in each region chooses one directional first
    # difference and one second difference. Each render gets its own probes.
    v=field.reshape(16,28,28);indexes=[];vals=[]
    for lx,ly in lm:
        cx=int(np.rint(lx));cy=int(np.rint(ly))
        for offy,offx in ((-1,-1),(-1,1),(1,-1),(1,1)):
            x=np.clip(cx+offx,1,26);y=np.clip(cy+offy,1,26)
            for parity in (0,1):
                tile=v[parity::2,y-1:y+2,x-1:x+2];idx=np.abs(tile).argmax();d,rem=divmod(idx,9);b,a=divmod(rem,3)
                indexes.append(((2*d+parity)*28+y+b-1)*28+x+a-1);vals.append(tile[d,b,a])
    val=np.array(vals);return indexes,np.round(32767*val/(np.abs(val).sum()+1.e-8)).astype(np.int16)

# A fixed spatial allocation of directed edge probes prevents a long, high
# contrast wrist border from drowning out the shorter edges in the fingers.
def probes(intensity,alpha,high):
    pts=[];weights=[]
    for source,amount in ((alpha,.59),(high,.41)):
        p=np.pad(source,1,mode='edge');gx=(p[1:-1,2:]-p[1:-1,:-2])*.5;gy=(p[2:,1:-1]-p[:-2,1:-1])*.5
        mag=np.hypot(gx,gy);ang=np.mod(np.rint(np.arctan2(gy,gx)*(6/np.pi)).astype(int),6)
        if source is high:
            # The contour has its own probes. Carry the internal shading here.
            mag=mag*np.minimum(1,alpha**3+.10)
        ix=[];wt=[]
        for y in range(0,28,4):
            for x in range(0,28,4):
                cell=mag[y:y+4,x:x+4];v,u=divmod(cell.argmax(),4);v+=y;u+=x
                ix.append((ang[v,u]*28+v)*28+u);wt.append(mag[v,u]**.76)
        weight=np.array(wt);pts.extend(ix);weights.extend(amount*weight/(weight.sum()+1.e-10))
    return np.array(pts,np.int16),np.round(32767*np.array(weights)).astype(np.int16)

inference = '''
bank=CONSTANTS["templates"].astype(np.float32)*(1/32767.)
poses=bank.shape[1]
width=bank.shape[2]
bank=bank.reshape(-1,width).T.copy()
probe=CONSTANTS["probes"].reshape(24*poses,-1)
weight=CONSTANTS["weights"].reshape(24*poses,-1).astype(np.float32)*(1/32767.)
land=CONSTANTS["land"].reshape(24*poses,-1)
ordinalweight=CONSTANTS["ordinal"].reshape(24*poses,-1).astype(np.float32)*(1/32767.)

def shift(a,dy,dx):
    if dx==0 and dy==0:return a
    ypos=np.clip(np.arange(28)+dy,0,27);xpos=np.clip(np.arange(28)+dx,0,27)
    return a[:,ypos[:,None],xpos[None,:]]

def predict(x):
    labels=np.empty(len(x),dtype=np.int32)
    # Fixed matrix geometry, including for an incomplete batch, so BLAS
    # uses the same accumulation path for every query.
    for start in range(0,len(x),32):
        n=min(32,len(x)-start);a=np.zeros((32,28,28),np.float32);a[:n]=x[start:start+n,0]
        m=foreground(a)
        best=np.full((32,24,4),-10.,dtype=np.float32)
        offsets=np.arange(24)[None,:,None]*poses
        # Compute the translation once in the edge field, not separately for
        # each candidate. The directed score is calculated at full resolution.
        field=curvefield(a).reshape(32,6,28,28)
        lf=localcontrast(a).reshape(32,16,28,28)
        for dy,dx in ((0,0),(0,-2),(0,2),(-2,0),(2,0),(-2,-2),(-2,2),(2,-2),(2,2)):
            f=features(shift(a,dy,dx),shift(m,dy,dx));s=(f@bank).reshape(32,24,poses)
            idx=np.argpartition(s,-6,axis=2)[:,:,-6:];coarse=np.take_along_axis(s,idx,axis=2)
            idx=idx+offsets;ef=shift(field.reshape(-1,28,28),dy,dx).reshape(32,-1)
            hit=ef[np.arange(32)[:,None,None,None],probe[idx]]
            contour=(hit*weight[idx]).sum(3)
            detail=shift(lf.reshape(-1,28,28),dy,dx).reshape(32,-1)
            d=detail[np.arange(32)[:,None,None,None],land[idx]]
            anatomical=(d*ordinalweight[idx]).sum(3)
            candidate=.49*coarse+.36*contour+.15*anatomical - .0015*(abs(dx)+abs(dy))
            best=np.partition(np.concatenate((best,candidate),2),-4,axis=2)[:,:,-4:]
        score=.55*best.max(2)+.45*best.mean(2)
        labels[start:start+n]=score[:n].argmax(1)
    return labels
'''


def main():
    path=Path(sys.argv[1]);path.mkdir(parents=True,exist_ok=True)
    (path/'solution.py').write_text(descriptors+finecode+localcode+inference)
    context={};exec(descriptors+localcode,context)
    templates=[];points=[];weights=[];land=[];ow=[]
    for letter in ABC:
        tmp=[]
        for j in range(36):
            images=[]; masks=[]; landmarks=[]
            for i in range(24):
                k=j*24+i
                image,alpha,high,lm=render(letter,k);images.append(image);masks.append(alpha);landmarks.append(lm)
                pi,pw=probes(image,alpha,high);points.append(pi);weights.append(pw)
            for img,lm,v in zip(images,landmarks,context['localcontrast'](np.stack(images))):
                li,lw=ordinal(img,lm,v);land.append(li);ow.append(lw)
            # Match both the ideal occupancy and the uncertain occupancy recovered from this rendering alone.
            a=np.array(images);m=.59*np.array(masks)+.41*context['foreground'](a)
            tmp.append(np.round(32767*context['features'](a,m)).astype(np.int16))
        templates.append(np.concatenate(tmp))
    np.savez(path/'constants.npz',templates=np.stack(templates),probes=np.stack(points),weights=np.stack(weights),land=np.array(land,np.int16),ordinal=np.stack(ow))

if __name__=='__main__':main()
