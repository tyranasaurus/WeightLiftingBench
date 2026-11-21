"""Data-free ASL hand renderer and matching program builder."""
import math
import sys
from pathlib import Path
import numpy as np

LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'
rad = np.deg2rad

def basis(yaw, pitch, roll):
    a,b,c = rad([yaw,pitch,roll]); u,v,w = np.cos([a,b,c]); i,j,k = np.sin([a,b,c])
    return np.array([[w,k,0],[-k,w,0],[0,0,1]]) @ np.array([[1,0,0],[0,v,-j],[0,j,v]]) @ np.array([[u,0,i],[0,1,0],[-i,0,u]])


def pose(letter, rng):
    U = rng.uniform
    N = lambda: np.clip(rng.normal(),-2.25,2.25)
    # Hand coordinates: y points from wrist to middle knuckle, x toward the thumb,
    # and z out of the palm. Dimensions are in units of roughly half a palm width.
    curled = np.array([65.,151.,218.]); straight = np.array([0.,4.,9.])
    aims = np.repeat(curled[None,:],4,axis=0)
    splay = np.array([-3.,0.,2.,5.]); twist = np.zeros(4); skew = np.zeros(4)
    baseyaw = 0.; dyaw = 17.; pitch = 0.; roll = 0.; droll=9.
    tmode = 'fold'; wrist = np.array([-.15,-1.,-.12])
    if letter in 'AMNSE':
        if letter == 'A': tmode='side'; baseyaw=5.; pitch=-7.
        if letter == 'E':
            aims[:]=[34,126,190]; tmode='below'; baseyaw=7.; pitch=3.
        if letter in 'MN':
            tmode='underM' if letter=='M' else 'underN'; pitch=2.; baseyaw=25.; roll=7.
            # The three (two for N) fingers covering the thumb cannot close as
            # tightly as the remaining fingers. Allow the knuckles to rise.
            ncover=3 if letter=='M' else 2
            aims[:ncover]=[39,115,174]; aims[ncover:]=[80,163,224]
            # Ulnar deviation when reaching over the thumb; for N the thumb
            # can show between the two raised digits and the two closed ones.
            if letter=='N': baseyaw=11.; roll=5.;splay[1]=6.;splay[2]=-7.
        if letter=='S': tmode='over'; pitch=2.; baseyaw=17.; roll=6.
    if letter == 'B':
        aims[:]=straight; splay[:]=[0.,0.,1.,3.]; tmode='below'; baseyaw=-13.; pitch=-3.
    if letter in 'CO':
        baseyaw=80.; dyaw=13.; roll=12.; pitch=3.; tmode=letter
        aims[:]=[18,62,98] if letter=='C' else [36,88,132]
        splay*=.4
    if letter == 'D':
        aims[:]=[55,130,177]; aims[0]=straight; tmode='touch3'; baseyaw=-7.; roll=1.; pitch=-3.
    if letter=='F':
        aims[:]=straight+[3,14,25]; aims[0]=[32,99,157]; splay[:]=[5.,19.,2.,-15.]; tmode='okay'; baseyaw=49.; dyaw=23.; roll=10.
    if letter in 'ILXY':
        if letter in 'IY': aims[3]=straight; splay[3]=-13.; roll=4.; baseyaw=5.
        if letter in 'LY': tmode='out'
        if letter=='L': aims[0]=straight; splay[0]=0.; roll=-3.; baseyaw=2.
        if letter=='X': aims[0]=[6.,74.,145.]; baseyaw=50.; dyaw=24.; roll=7.
    if letter in 'RUVW':
        aims[:2]=straight; pitch=-2.; baseyaw=-17.; roll=-5.
        if letter=='R':
            splay[:2]=[-13.,6.]; aims[1]=[-1.,8.,23.]; skew[1]=-5.; twist[1]=15.
        if letter=='U': splay[:2]=[-1.,3.]
        if letter in 'VW': splay[:2]=[17.,-10.]
        if letter=='W': splay[:3]=[18.,1.,-17.];aims[2]=straight; baseyaw=-11.
    if letter in 'KP':
        aims[0]=straight; aims[1]=[59.,64.,70.]; splay[:2]=[3.,-5.]; tmode='middle';baseyaw=-32.;pitch=-11.;roll=-3.
        if letter=='P': baseyaw=-68.;pitch=48.;roll=-48.; dyaw=18.; aims[1]=[69.,78.,87.];
    if letter in 'GHTQ':
        if letter in 'GQ':
            aims[0]=straight; splay[0]=0.; tmode='parallel';baseyaw=144.; roll=55.; pitch=19.; aims[0]=[26.,32.,36.];
            if letter=='Q':
                # Flex the index over a dorsiflexed wrist. The hand's long axis
                # points out of the image, not straight down from an inverted wrist.
                baseyaw=96.;pitch=44.;roll=77.;aims[0]=[59.,66.,77.]; tmode='downpinch';
        if letter=='H':
            aims[:2]=[24.,29.,34.]; splay[:2]=[-1.,4.]; baseyaw=123.; roll=55.;pitch=19.;
        if letter=='T':
            # End-on view of a closed hand with thumb protruding by the index.
            aims[:]=[65.,145.,201.]; aims[0]=[72.,87.,105.]; tmode='underT'; baseyaw=74.; roll=0.; pitch=10.;
    yaw=baseyaw+dyaw*N(); pitch+=N()*10.;roll+=droll*N(); R=basis(yaw,pitch,roll)
    if letter in 'PGHQ':
        # The forearm descends from the wrist even when the hand is flexed or
        # pronated. Specify its direction in camera space for these signs.
        wcam={'P':[.40,-.85,-.42],'G':[-.73,-.76,-.32],'H':[-.74,-.74,-.34],'Q':[-.35,-.82,-.67]}[letter]
        wrist=R.T@np.array(wcam)
    spread=U(.75,1.20); flexcommon=N()*5.; yawf=N()*2.
    fingerlen=U(.90,1.12); breadth=U(.90,1.1); girth=U(.89,1.10); palmheight=U(.91,1.11)
    caps=[]; ell=[]
    def ball(c,axes,ang=0):
        ell.append((np.array(c),np.array(axes),basis(0,0,ang)))
    ball([0,.85,.0],[.84*breadth,1.03*palmheight,.32*girth]);
    ball([.34*breadth,.72,.13],[.425*breadth,.73*palmheight,.29*girth],17)
    ball([-.38*breadth,.82,.073],[.38*breadth,.77*palmheight,.24*girth],-11)
    # A broad metacarpal pad supports all four knuckles; the crease at its base
    # and the thicker thenar pad break up the otherwise flat palm.
    ball([0,1.40*palmheight,.035],[.81*breadth,.43*palmheight,.31*girth],4)
    roots=[];tips=[]; digitpoints=[]
    for f in range(4):
        px=[.58,.19,-.21,-.57][f]*breadth;py=[1.58,1.65,1.58,1.41][f]*palmheight
        p=np.array([px,py,.045]); roots.append(p); points=[p];
        lens=np.array([.45,.30,.25])*[1.49,1.65,1.54,1.22][f]*fingerlen*U(.96,1.04)
        fr=[.208,.216,.204,.173][f]*girth
        angles=aims[f]+flexcommon+N()*4. + np.array([N()*3.,N()*5.,N()*6.]);
        # Curl stays within a digit's sagittal plane, which can fan sideways.
        fan=rad(splay[f]*spread + yawf + N()*(1.1 if letter in 'BURH' else 3.)); axial=rad(twist[f]+N()*3.)
        for j,(length,theta) in enumerate(zip(lens,rad(angles))):
            vx=np.sin(fan)*np.cos(theta)+np.cos(fan)*np.sin(axial)*np.sin(theta)
            vy=np.cos(fan)*np.cos(theta)-np.sin(fan)*np.sin(axial)*np.sin(theta)
            v=np.array([vx,vy,np.cos(axial)*np.sin(theta)]);q=p+length*v
            caps.append((p,q,fr*[.99,.91,.82][j],fr*[.92,.85,.70][j],f,j))
            p=q; points.append(p)
        tips.append(p);digitpoints.append(points)
    # Thumb carpometacarpal, metacarpophalangeal, interphalangeal and tip.
    root=[.43*breadth,.43*palmheight,.10]
    if tmode=='side': th=[[.78,.98,.24],[.86,1.49,.34],[.86,1.91,.32]]
    elif tmode in ('below','fold','over'):
        th=({'below':[[.70,1.02,.34],[.36,1.26,.57],[-.22,1.27,.56]],
             'fold':[[.68,1.08,.37],[.28,1.40,.68],[-.14,1.60,.65]],
             'over':[[.67,1.12,.48],[.22,1.44,.87],[-.27,1.66,.92]]})[tmode]
    elif tmode in ('underM','underN','underT'):
        end={'underM':-.47,'underN':-.04,'underT':.35}[tmode]
        th=[[.64,1.01,.39],[(end+.57)/2,1.40,.59],[end,1.72,.73]]
        if tmode=='underT': th=[[.68,1.08,.38],[.49,1.74,.48],[end,2.43,.44]]
    elif tmode=='out': th=[[.93,.76,.23],[1.40,1.04,.28],[1.85,1.24,.31]]
    elif tmode=='parallel': th=[[.90,.95,.62],[.96,1.48,.95],[.95,1.99,1.04]]
    elif tmode=='downpinch': th=[[.81,.89,.53],[.91,1.21,.96],[.93,1.52,1.43]]
    elif tmode=='C': th=[[.76,.82,.50],[.75,1.10,.99],[.70,1.47,1.37]]
    elif tmode in ('touch3','okay','O'):
        end=tips[0 if tmode=='okay' else 1].copy(); end[1]-=.09;end[2]-=.02
        if tmode=='touch3': th=[[.70,1.08,.50],[.62,1.48,.88],end]
        elif tmode=='okay':
            # Abduct the base of the thumb to leave the characteristic aperture.
            th=[[1.02,1.04,.36],[1.10,1.56,.74],end]
        else: th=[[.78,.96,.53],[.66,1.31,.99],end]
    elif tmode=='middle': th=[[.66,1.04,.39],[.46,1.49,.65],[.34,1.98,.62]]
    tp=np.array([root]+th);tp[1:]+=rng.normal(0,.060,(3,3));tp[1:,0]*=breadth; tp[1:,1]*=palmheight
    for j in range(3): caps.append((tp[j],tp[j+1],girth*[.29,.231,.198][j],girth*[.24,.204,.158][j],4,j))
    # Forearm may bend independently at the wrist.
    wrist=wrist+np.array([U(-.23,.22),0,U(-.17,.17)]); wrist=wrist/np.linalg.norm(wrist)
    if letter in 'GHTX': wrist=R.T@np.array([-.62+N()*.15,-.87,-.35]); wrist/=np.linalg.norm(wrist)
    if letter=='Q' and rng.random()<.6:
        # An alternative common production: a bent-index precision grip, seen
        # dorsally. The two jaws point down on either side of the forearm.
        R=basis(92+N()*15,46+N()*12,35+N()*15)
        wrist=R.T@np.array([-.20,-.89,-.41]);wrist/=np.linalg.norm(wrist)
        def camera_chain(root,segments):
            pts=[root]
            for v in segments:pts.append(pts[-1]+R.T@(np.array(v)*U(.91,1.09)))
            return pts
        pp=camera_chain(roots[0],[(.32,-.40,.42),(-.05,-.44,.17),(-.16,-.33,-.03)])
        tp=camera_chain(np.array([.50,.52,.14]),[(.18,.28,.44),(-.11,-.40,.29),(-.27,-.33,.10)])
        for j in range(3):
            caps[j]=(pp[j],pp[j+1],caps[j][2],caps[j][3],0,j)
            caps[-3+j]=(tp[j],tp[j+1],caps[-3+j][2],caps[-3+j][3],4,j)
    wristroot=np.array([-.04,.25,-.06]);
    # Object coordinates have depth in camera direction after rotation.
    def projell(c,axes,orient):
        return R@c, (R@orient)*axes
    bodies=[projell(*e) for e in ell]; bones=[(R@p,R@q,a,b,k,j) for p,q,a,b,k,j in caps]
    # Smooth flaring forearm with an elliptical cross section.
    v=wrist; ux=np.array([1.,0,0]); uz=np.cross(v,ux);uz/=np.linalg.norm(uz);ux=np.cross(uz,v);orient=np.stack([ux,v,uz],axis=1)
    fore=[projell(wristroot+wrist*d,[w*breadth,h,.34*girth],orient) for d,w,h in [(0.63,.54,1.23),(2.10,.63,1.75),(3.58,.77,1.60)]]
    sample=[]
    for c,m in bodies:
        half=np.sqrt((m*m).sum(1)); sample.extend([c-half,c+half]);
    for p,q,a,b,k,j in bones:sample.extend([p-a,p+a,q-b,q+b]);
    armlen=U(.32,1.06) if letter not in 'PQ' else U(.08,.42)
    cut=R@(wristroot+wrist*armlen); sample.extend([cut+(R@ux)*.18,cut-(R@ux)*.18]); bounds=np.array(sample)[:,:2];lo=bounds.min(0); hi=bounds.max(0)
    side=max(hi-lo)*U(.91,1.17);center=(hi+lo)*.5+np.array([N(),N()])*side*.063
    center[1]+=side*.035
    return bodies+fore,bones,center,side


def render(letter,rng,res=56,aux=False):
    bodies,bones,c,s=pose(letter,rng); h=res
    step=s/h; xs=(np.arange(h)+.5-h/2)*step+c[0];ys=(h/2-np.arange(h)-.5)*step+c[1];
    z=np.full((h,h),-20.);alb=np.ones((h,h));
    def patch(lx,ly,hx,hy):
        a,b=xs.searchsorted([lx,hx],side='left'); l,r=(-ys).searchsorted([-hy,-ly],side='left'); return slice(l,r),slice(a,b)
    def apply(box,zp,al):
        zz=z[box];keep=zp>zz; np.maximum(zz,zp,out=zz); np.copyto(alb[box],al,where=keep)
    for pos,m in bodies:
        widths=np.sqrt((m[:2]*m[:2]).sum(1)); box=patch(*(pos[:2]-widths),*(pos[:2]+widths)); sy,sx=box
        dx=xs[None,sx]-pos[0];dy=ys[sy,None]-pos[1];
        cov=m@m.T; a,b,cc=cov[0,0],cov[0,1],cov[1,1]; det=a*cc-b*b;aa=cc/det;bb=-b/det;dd=a/det
        ux=aa*dx+bb*dy;uy=bb*dx+dd*dy;norm=1-dx*ux-dy*uy
        p,q=cov[2,:2];height=cov[2,2]-(p*p*aa+2*p*q*bb+q*q*dd)
        top=pos[2]+p*ux+q*uy+np.sqrt(np.maximum(0,norm*height));apply(box,np.where(norm>=0,top,-20),1.)
    for p,q,r0,r1,k,j in bones:
        box=patch(*(np.minimum(p[:2]-r0,q[:2]-r1)),*(np.maximum(p[:2]+r0,q[:2]+r1)));sy,sx=box
        dx=xs[None,sx]-p[0];dy=ys[sy,None]-p[1]; vx,vy,vz=q-p;den=vx*vx+vy*vy+1e-8
        foot=(dx*vx+dy*vy)/den; t=np.clip(foot,0,1);rr=r0+(r1-r0)*t
        cross=dx*vy-dy*vx; off=np.sqrt(np.maximum(0,rr*rr-cross*cross/den));
        t=np.clip(foot+vz*off/np.sqrt(den*(den+vz*vz)),0,1); rr=r0+(r1-r0)*t
        disc=rr*rr-(dx-t*vx)**2-(dy-t*vy)**2;top=p[2]+t*vz+np.sqrt(np.maximum(disc,0));
        # Crease and tip details, softer on the finger pads than on the joints.
        al=1.-.066*np.exp(-((t-.90)/.09)**2) if j<2 else 1.+.042*np.exp(-((t-.73)/.26)**2)
        apply(box,np.where(disc>=0,top,-20),al)
    fg=z>-15; depth=np.where(fg,z,0); padded=np.pad(z,1,mode='edge'); near=lambda a: np.where(a>-15,a,depth-.13)
    left,right,up,down=[near(a) for a in (padded[1:-1,:-2],padded[1:-1,2:],padded[:-2,1:-1],padded[2:,1:-1])]
    gx=(right-left)/(-2*step);gy=(up-down)/(-2*step);norm=np.sqrt(1+gx*gx+gy*gy)
    # Model both direct light and bounced light. This also models the larger
    # brightness change across a curved hand, beyond the fine digit creases.
    phi=rng.uniform(-.12,.94); lx=-np.cos(phi)*(1 if rng.random()<.91 else -1);ly=np.sin(phi);lz=rng.uniform(.18,.85)
    light=(gx*lx+gy*ly+lz)/(norm*np.sqrt(1+lz*lz)); bright=rng.uniform(.38,.62); contrast=rng.uniform(.30,.53)
    amb=(lx*(xs-c[0])[None,:]+ly*(ys-c[1])[:,None])/s
    shade=(bright+contrast*light+rng.uniform(.12,.40)*amb)*alb
    # Cast short shadows where one digit overlaps another in projection.
    pad=np.pad(z,5,mode='edge');occ=0.
    for d in (2,5):
        oy=int(round(ly*d));ox=int(round(lx*d)); blocker=pad[5-oy:5-oy+h,5+ox:5+ox+h];occ+=np.clip((blocker-z-step*d*lz)*3.,0,1)
    shade-=.11*occ;bg=rng.uniform(.56,.9) if rng.random()<.92 else rng.uniform(.27,.55)
    bg=bg+rng.normal(0,.07)*(xs-c[0])[None,:]/s+rng.normal(0,.12)*(ys-c[1])[:,None]/s
    ans=np.where(fg,shade,bg); image=np.clip(ans.reshape(28,res//28,28,res//28).mean((1,3)),0,1).astype(np.float32)
    if aux:
        coverage=fg.reshape(28,res//28,28,res//28).mean((1,3)).astype(np.float32)
        return image,coverage
    return image


# This block is shared verbatim by the builder (for analytic renderings) and
# the predictor. It never needs a stored photograph or data-derived statistic.
FEATURES = '''import numpy as np

def blur(a):
    p = np.pad(a,((0,0),(1,1),(1,1)),mode="edge")
    b=(p[:,:-2]+2*p[:,1:-1]+p[:,2:])*.25
    return (b[:,:,:-2]+2*b[:,:,1:-1]+b[:,:,2:])*.25

def pooled(a,n):
    # Bilinear spatial voting, equivalent to a triangular cell window.
    pts=(np.arange(n)+.5)*28/n-.5
    w=np.maximum(1-np.abs(pts[:,None]-np.arange(28))/(28/n),0).astype(np.float32)
    w/=w.sum(1,keepdims=True)
    return w @ a @ w.T

def vector(a):
    return a.reshape(len(a),-1)

def unit(a):
    return a / np.sqrt(np.sum(a*a,axis=1,keepdims=True)+1e-10)

def describe(a):
    smooth=blur(a); grad=.55*a+.45*smooth
    pad = np.pad(grad,((0,0),(1,1),(1,1)),mode="edge")
    dx=(pad[:,1:-1,2:]-pad[:,1:-1,:-2]); dy=pad[:,2:,1:-1]-pad[:,:-2,1:-1]
    mag=np.sqrt(dx*dx+dy*dy+1e-10)
    # Clip rare high-contrast texture boundaries. Keep smooth regions weak.
    ref=np.sqrt(np.mean(mag*mag,axis=(1,2),keepdims=True))+.008
    strength=np.minimum(mag/ref,2.5)**.75
    dx/=mag;dy/=mag
    ax=np.linspace(-1,1,28,dtype=np.float32)
    weight=1-.30*np.abs(ax)[None,:]**3-.21*((ax[:,None]+1)*.5)**4
    strength*=weight
    # Separate fine, directed finger detail from large silhouettes. These
    # oriented even/odd filters respond to ridges as well as to step edges.
    p=np.pad(a,((0,0),(3,3),(3,3)),mode="edge")
    def shift(oy,ox): return p[:,3+oy:31+oy,3+ox:31+ox]
    fingers=[]
    for oy,ox in ((0,1),(1,0),(1,1),(1,-1)):
        t1=shift(oy,ox);t2=shift(-oy,-ox)
        d1=(t1-t2); d2=t1+t2-2*a
        # Average ridge energy along the hypothetical long axis of the digit.
        # The principal fine-scale response depends on width, not skin tone.
        sig=np.minimum(2.5,np.sqrt(.42*d1*d1+1.6*d2*d2)/(ref+.01))
        pp=np.pad(sig,((0,0),(2,2),(2,2)),mode="edge")
        ridge=(2*sig+pp[:,2+ox:30+ox,2-oy:30-oy]+pp[:,2-ox:30-ox,2+oy:30+oy])*.25
        fingers.append(vector(pooled(ridge*weight,7)))
    orient=[]; signed=[]
    for i in range(4):
        phi=i*np.pi/4; dot=dx*np.cos(phi)+dy*np.sin(phi)
        ch=(strength*np.maximum(np.abs(dot)-.57,0)).astype(np.float32)
        orient.append(ch);signed.append(vector(pooled(ch*np.sign(dot),9)))
    fine=[];coarse=[]
    for ch in orient:
        fine.append(vector(pooled(ch,10)));coarse.append(vector(pooled(ch,6)))
    # Soft detection of relief: departures from a locally smooth background.
    # A photo's background may be darker OR lighter than the skin.
    wide=smooth
    for k in range(8):wide=blur(wide)
    hp=np.clip((smooth-wide)/(ref+.02),-2,2)
    dark=pooled(np.maximum(-hp,0)*weight,9);lite=pooled(np.maximum(hp,0)*weight,9)
    edge=pooled(strength,8)
    # Propagate confidence in the background from the outside edges. Penalize
    # paths crossing high gradients, not the absolute color at the endpoint:
    # this can follow continuous changes of illumination across a wall.
    barrier=np.minimum(1,(mag/(ref*1.20))**1.6)
    distance=np.ones_like(a)
    distance[:,0,:]=barrier[:,0,:];distance[:,:,0]=barrier[:,:,0];distance[:,:,-1]=barrier[:,:,-1]
    for k in range(16):
        p=np.pad(distance,((0,0),(1,1),(1,1)),mode="edge")
        neigh=np.minimum(np.minimum(p[:,1:-1,2:],p[:,1:-1,:-2]),np.minimum(p[:,2:,1:-1],p[:,:-2,1:-1]))
        distance=np.minimum(distance,np.maximum(neigh,barrier))
    support=pooled((.75*distance+.25*np.minimum(1,strength))*weight,8)
    def contrast(b):
        return unit(b-.85*np.mean(b,axis=1,keepdims=True))
    h=contrast(np.concatenate(fine,axis=1));g=contrast(np.concatenate(coarse,axis=1));sg=unit(np.concatenate(signed,axis=1))
    bands=np.concatenate([vector(dark),vector(lite)],axis=1)
    ridgevec=contrast(np.concatenate(fingers,axis=1))
    return np.concatenate((.50*h,.29*g,.37*sg,.31*contrast(vector(edge)),.34*contrast(vector(support)),.38*contrast(bands),.34*ridgevec),axis=1).astype(np.float32)
'''

LOCAL = '''
def keyplanes(a):
    b=blur(a); c=blur(b); w=c
    for k in range(5): w=blur(w)
    p=np.pad(.5*(a+b),((0,0),(1,1),(1,1)),mode="edge")
    dx=p[:,1:-1,2:]-p[:,1:-1,:-2];dy=p[:,2:,1:-1]-p[:,:-2,1:-1]
    m=np.sqrt(dx*dx+dy*dy+.000016); ref=np.sqrt(np.mean(m*m,axis=(1,2),keepdims=True))+.01
    volume=[]
    for i in range(4):
        angle=i*np.pi/4;g=(dx*np.cos(angle)+dy*np.sin(angle)).astype(np.float32)
        volume.append(blur(np.maximum(np.abs(g)-.58*m,0)/ref))
    volume.append(blur(np.maximum(w-b,0)/ref));volume.append(blur(np.maximum(b-w,0)/ref))
    volume=np.stack(volume,axis=1)
    return np.minimum(volume,1.6)

def localfield(planes):
    # Endpoint validation: soft maximum in a small neighborhood tolerates
    # foreshortening and articulation error between nearby poses.
    q=planes.shape
    p=np.pad(planes,((0,0),(0,0),(1,1),(1,1)),mode="edge")
    z=np.maximum.reduce([p[:,:,j:j+28,i:i+28] for i,j in ((0,1),(1,0),(1,1),(1,2),(2,1))])
    z=.55*planes+.45*z
    z=z/(z+.18)
    return z.reshape(q[0],-1)
'''

PREDICT = '''
def predict(x):
    ans=np.empty(len(x),dtype=np.int64)
    # Padding makes the matrix product and all reductions identical for a
    # given image even when its position in the caller's batch changes.
    count=64
    for start in range(0,len(x),count):
        a=x[start:start+count,0];n=len(a); block=np.zeros((count,28,28),dtype=np.float32);block[:n]=a
        d=describe(block); scores=[]; planes=keyplanes(block);field=localfield(planes)
        quiet=np.max(planes[:,:4],axis=1)+.7*np.maximum(planes[:,4],planes[:,5]);quiet=(quiet/(quiet+.23)).reshape(count,784)
        align=CONSTANTS["align"]; poses=CONSTANTS["points"]
        lean=CONSTANTS["lean"]; penalty=CONSTANTS["penalty"]
        rows=np.arange(count)[:,None,None]
        for label in range(24):
            bank=CONSTANTS[str(label)]
            sim=d @ bank
            ids=np.argpartition(sim,-20,axis=1)[:,-20:]
            kp=poses[label][ids]; bg=CONSTANTS["spaces"][label][ids];base=np.take_along_axis(sim,ids,axis=1)
            # A small global translation, plus independent small motions of the
            # upper and lower articulations, are nuisance variables.
            best=np.full(base.shape,-2,dtype=np.float32)
            for j in range(len(align)):
                up=field[rows,align[j][kp[:,:,:32]]].mean(2)
                dn=field[rows,align[j][kp[:,:,32:]]].mean(2)
                if j==0: fixedup=up;fixeddn=dn
                v=.5*(np.maximum(up,fixedup-lean[j])+np.maximum(dn,fixeddn-lean[j]))-penalty[j]
                stray=quiet[rows,align[j][bg]]
                # Permit partial occlusion of the nominal background.
                v-=.37*np.mean(np.minimum(stray,.60),axis=2)
                best=np.maximum(best,v)
            merits=base+.70*best
            top=np.partition(merits,-4,axis=1)[:,-4:]
            scores.append(.45*np.max(top,axis=1)+.55*np.mean(top,axis=1))
        ans[start:start+n]=np.argmax(np.stack(scores,axis=1),axis=1)[:n]
    return ans
'''


def keypoints(planes,cov):
    # Keep separate sets of non-overlapping landmarks for the digit and palm
    # ends. Some landmarks describe the outer contour, others the wrinkles and
    # in-between-finger edges. All points come from the analytic pose only.
    n=len(planes); p=np.pad(cov,((0,0),(1,1),(1,1)),mode='edge')
    dx=p[:,1:-1,2:]-p[:,1:-1,:-2];dy=p[:,2:,1:-1]-p[:,:-2,1:-1];m=np.hypot(dx,dy)
    edge=np.stack([np.maximum(0,np.abs(dx*np.cos(i*np.pi/4)+dy*np.sin(i*np.pi/4))-.55*m) for i in range(4)],axis=1)
    score=planes.copy();score[:,:4]+=.55*edge
    score[:,:,:,0]*=.2;score[:,:,:,-1]*=.2;score[:,:,:1]*=.2;score[:,:,-1:]*=.2
    chan=score.argmax(1);strength=np.max(score,1);out=[];row=np.arange(n)
    yy,xx=np.ogrid[:28,:28];
    for half in range(2):
        v=strength.copy(); v[:,14:] *= half;v[:,:14]*=1-half
        for j in range(32):
            idx=np.argmax(v.reshape(n,784),axis=1);y=idx//28;x=idx%28;out.append(idx+chan[row,y,x]*784)
            dist=(yy-y[:,None,None])**2+(xx-x[:,None,None])**2
            v*=np.minimum(1,dist/6)
    return np.stack(out,axis=1).astype(np.uint16)


def stability(planes,cov):
    # Negative evidence: the predicted background is usually smooth. Test a
    # spread of empty places, so a locally good match to just one finger does
    # not ignore an extra finger in a V, W, or Y. Avoid the crop boundaries.
    y,x=np.mgrid[:28,:28];y=y[None];x=x[None];bg=(1-cov)**2
    buff=np.pad(bg,((0,0),(1,1),(1,1)),mode='edge')
    confidence=sum(buff[:,a:a+28,b:b+28] for a in range(3) for b in range(3))/9
    envelope=(.1+np.sin(np.pi*(x+.5)/28)**.8)*(.1+np.sin(np.pi*(y+.5)/28)**.65)
    fg=cov*(.15+planes[:,4]+planes[:,5]);v=bg*(confidence**2)*envelope
    # Static samples near the outside of the shape carry more information
    # than locations on a remote blank wall.
    q=np.pad(cov,((0,0),(5,5),(5,5)),mode='edge'); near=sum(q[:,a:a+28,b:b+28] for a in (0,5,10) for b in (0,5,10))/9
    v*=.1+near
    out=[]
    for j in range(24):
        idx=np.argmax(v.reshape(len(v),784),axis=1);out.append(idx)
        dist=(x-(idx%28)[:,None,None])**2+(y-(idx//28)[:,None,None])**2
        v*=np.minimum(1,dist/20)
    return np.stack(out,axis=1).astype(np.uint16)


def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'solution.py').write_text(FEATURES+LOCAL+PREDICT)
    f={};exec(FEATURES+LOCAL,f);bank={};landmarks=[];spaces=[]
    u=np.arange(6*784);ch=u//784;y=u//28%28;x=u%28
    shifts=[(0,0),(-2,0),(2,0),(0,-2),(0,2),(-1,-1),(-1,1),(1,-1),(1,1)]
    bank['align']=np.stack([ch*784+np.clip(y+dy,0,27)*28+np.clip(x+dx,0,27) for dy,dx in shifts]).astype(np.int32)
    # Small motions of only one region cost more than coherent translation.
    bank['lean']=np.array([0.0]+[.018]*8,np.float32);bank['penalty']=np.array([0.0]+[.006]*8,np.float32)
    for k,letter in enumerate(LETTERS):
        rng=np.random.default_rng(57017+k);desc=[];points=[];empty=[]
        for start in range(52):
            drawings,masks=map(np.stack,zip(*(render(letter,rng,aux=True) for j in range(64))))
            desc.append(f['describe'](drawings));planes=f['keyplanes'](drawings)
            points.append(keypoints(planes,masks));empty.append(stability(planes,masks))
        bank[str(k)]=np.ascontiguousarray(np.concatenate(desc,axis=0).T)
        landmarks.append(np.concatenate(points,axis=0));spaces.append(np.concatenate(empty,axis=0))
    bank['points']=np.stack(landmarks);bank['spaces']=np.stack(spaces)
    np.savez(out/'constants.npz',**bank)

if __name__=='__main__':build(sys.argv[1])
