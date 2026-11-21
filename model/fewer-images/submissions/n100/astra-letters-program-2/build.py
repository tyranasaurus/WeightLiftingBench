"""Build the template library from forward kinematics and an orthographic renderer.
All lengths below are in anatomical units (approximately centimetres), never input pixels.
"""
import math
import sys
from pathlib import Path
import numpy as np

SOLUTION = '''import numpy as np


def neighbor(a, d, axis):
    n = a.shape[axis]
    return np.take(a, np.clip(np.arange(n) + d, 0, n-1), axis=axis)


def smooth(a):
    for axis in (1, 2):
        a = (neighbor(a, -1, axis) + 2*a + neighbor(a, 1, axis)) * .25
    return a


def normrows(a):
    a = a.reshape(len(a), -1)
    return a / np.sqrt((a*a).sum(axis=1, keepdims=True) + 1e-10)


def describe(a):
    # Unsigned gradients: the hand/background contrast can change sign.
    a = smooth(a)
    gx = (neighbor(a, 1, 2)-neighbor(a, -1, 2)) * .5
    gy = (neighbor(a, 1, 1)-neighbor(a, -1, 1)) * .5
    mag = np.sqrt(gx*gx + gy*gy + 1e-12)
    n = len(a)
    # A smooth illumination ramp is not a contour. Remove that background
    # response before any compression, and cap strong clutter edges locally.
    rms = np.sqrt(np.mean(mag*mag, axis=(1,2), keepdims=True) + 1e-8)
    residual = np.maximum(0, mag / rms - .27)
    weight = np.minimum(residual, 2.8)
    field = []
    for k in range(6):
        t = k * (np.pi/6)
        proj = np.abs((np.cos(t)*gx + np.sin(t)*gy) / mag)
        field.append((weight * np.maximum(0, 1 - (1-proj)*7.5)).astype(np.float32))
    field = np.stack(field, axis=3)
    # Four-pixel cells, softly overlap their immediate neighbors.
    f = smooth(field).reshape(n,7,4,7,4,6).mean(axis=(2,4))
    density = f.sum(axis=-1, keepdims=True)
    loc = .22 + smooth(density)
    hog = f / np.sqrt(loc)
    # Separately preserve orientation selectivity and occupancy. Uniform dirt
    # producing energy in every orientation should not resemble a finger.
    selective = hog - hog.mean(axis=3, keepdims=True)
    def center(v): return v-v.mean(axis=(1,2),keepdims=True)
    edge = normrows(center(hog))
    occ = normrows(center(density / np.sqrt(loc)))
    orient = normrows(selective)
    desc = np.concatenate([edge*.70, occ*.55, orient*.45], axis=1)
    # Piecewise smooth hand-surface shading, without a broad illumination ramp.
    broad = a
    for stride in (1, 2, 3):
        for axis in (1, 2):
            broad = (neighbor(broad,-stride,axis)+2*broad+neighbor(broad,stride,axis))*.25
    hp = a-broad
    texture = np.maximum(-2.5, np.minimum(2.5, hp/(2*rms+.006)))
    texture = smooth(texture).reshape(n,14,2,14,2).mean(axis=(2,4))
    signed = normrows(texture)
    # Difference of local averages responds to the two flanks of a narrow digit,
    # even if one of those flanks has almost the same brightness as the wall.
    fine=[]
    for dy, dx in ((0,2),(2,0),(1,1),(1,-1)):
        ridge=np.abs(a-.5*(shifted(a,dy,dx)+shifted(a,-dy,-dx))) / (rms+.004)
        ridge=np.clip(ridge-.22,0,2.5)
        fine.append(smooth(ridge).reshape(n,14,2,14,2).mean(axis=(2,4)))
    bar=normrows(center(np.stack(fine,axis=3)))
    # Hypothesize a backdrop varying smoothly between the two side margins.
    # If a margin contains a strong edge, favor the less textured margin.
    def border(v):
        for stride in (1,2,3):
            v=(neighbor(v,-stride,1)+2*v+neighbor(v,stride,1))*.25
        return v
    left=border(np.median(a[:,:,:3],axis=2,keepdims=True))
    right=border(np.median(a[:,:,-3:],axis=2,keepdims=True))
    ltex=border(mag[:,:,:3].mean(axis=2,keepdims=True))
    rtex=border(mag[:,:,-3:].mean(axis=2,keepdims=True))
    pos=np.linspace(.06,.94,28,dtype=np.float32)[None,None,:]
    wl=(1-pos)/(ltex/rms+.24)**2
    wr=pos/(rtex/rms+.24)**2
    wall=(wl*left+wr*right)/(wl+wr)
    mat=np.minimum(1,np.abs(a-wall)/(2.4*rms+.015))
    coverage=smooth(mat).reshape(n,14,2,14,2).mean(axis=(2,4))
    matte=normrows(center(coverage))
    desc=np.concatenate([desc*.70, signed*.37, bar*.36, matte*.47],axis=1)
    # Emphasize the articulated hand, not the sleeve, and retain a separate
    # orientation signature restricted to the finger region. Weighting after
    # centering does not introduce an outside brightness reference.
    def focus(h,w):
        y,x=np.mgrid[0:h,0:w]
        y=(y+.5)/h; x=(x+.5)/w
        return (.22+.78*np.exp(-.5*((y-.33)/.27)**2-.5*((x-.50)/.40)**2)).astype(np.float32)
    fmask=focus(7,7)[None,:,:,None]
    upper=normrows(fmask*center(hog))
    dmask=focus(14,14)[None,:,:,None]
    upperridge=normrows(dmask*center(np.stack(fine,axis=3)))
    desc=np.concatenate([desc*.83, upper*.39, upperridge*.39],axis=1)
    # Integer fixed point lets every dot product be exactly representable in float32.
    return np.rint(desc*1536).astype(np.int16)


def shifted(a, y, x):
    return neighbor(neighbor(a, y, 1), x, 2)


def predict(x):
    bank = CONSTANTS['shape'].astype(np.float32)
    coarse=np.ascontiguousarray(np.concatenate([bank[:,:833],bank[:,1617:2107]],axis=1))
    nmodel = bank.shape[0] // 24
    out = np.zeros(len(x), dtype=np.int64)
    for start in range(0, len(x), 40):
        a = x[start:start+40,0]
        views = []
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                views.append(describe(shifted(a, dy, dx)))
        q = np.stack(views, axis=1).reshape(-1,bank.shape[1]).astype(np.float32)
        # First align the entire hand rigidly. Use the inexpensive pooled
        # gradients, matte and shading to shortlist individual render/shift
        # pairs; only those need the full-resolution comparison.
        cols=list(range(833))+list(range(1617,2107))
        matrix=q[:,cols]@coarse.T
        allscore=matrix.reshape(len(a),9,24,nmodel).transpose(0,2,1,3).reshape(len(a),24,-1)
        nshort=48
        ids=np.argpartition(allscore,9*nmodel-nshort,axis=2)[:,:,-nshort:]
        tiny=np.array([[0,0],[-1,0],[1,0],[0,-1],[0,1]])
        for j in range(len(a)):
            templates=ids[j]%nmodel+np.arange(24)[:,None]*nmodel
            aligns=ids[j]//nmodel
            row=q[j*9:j*9+9]
            resc=(row[aligns]*bank[templates]).sum(axis=2)
            which=np.argpartition(resc,nshort-6,axis=1)[:,-6:]
            letter=np.arange(24)[:,None]
            indices=templates[letter,which].ravel()
            bestv=aligns[letter,which].ravel()
            globalmatch=resc[letter,which].ravel()/(1536.*1536.)
            def patchscore(first,last,side,channels,off):
                # Soft quadrants overlap at the palm. Moving one quadrant by
                # one fine cell allows small changes in digit length/spread.
                t=bank[indices,first:last].reshape(-1,side,side,channels)
                image=q[9*j:9*j+9,first:last].reshape(9,side,side,channels)
                h=(side+1)//2+1
                centergrid=np.arange(side)
                total=np.zeros(len(t))
                for d in (0,side-h):
                    for e in (0,side-h):
                        sub=t[:,d:d+h,e:e+h]
                        best=np.full(len(t),-1.)
                        for dy,dx in tiny:
                            yy=np.clip(centergrid[d:d+h]+dy,0,side-1)
                            xx=np.clip(centergrid[e:e+h]+dx,0,side-1)
                            match=image[:,yy[:,None],xx[None,:]][bestv]
                            dot=(sub*match).sum(axis=(1,2,3))
                            den=np.sqrt((sub*sub).sum(axis=(1,2,3))*(match*match).sum(axis=(1,2,3))+1.)
                            best=np.maximum(best,dot/den-off*(abs(dy)+abs(dx)))
                        total+=best*.25
                return total
            # Fine-scale second derivative and shading maps share a 14x14 grid.
            micro=patchscore(833,1617,14,4,.045)
            light=patchscore(637,833,14,1,.05)
            local=.57*micro+.43*light
            sc=(globalmatch*.73+local*.27).reshape(24,-1)
            out[start+j]=(.70*sc.max(axis=1)+.30*sc.mean(axis=1)).argmax()
    return out
'''

# Unit vector math. Palm frame: x toward thumb, y to fingertips, z palm-normal.
def unit(v):
    v=np.asarray(v,float)
    return v / max(1e-9,np.linalg.norm(v))

def rmat(axis,degree):
    a=math.radians(degree); c,s=math.cos(a),math.sin(a)
    if axis==0: return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis==1: return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# Tapered components are capped with rounded ends, palms are ellipsoids.
def model(letter, r):
    balls=[]; bones=[]
    def ball(pos, axes, radii, tone=0, group=0):
        balls.append((np.array(pos,dtype=float),np.array(axes,dtype=float)@np.diag(radii),tone,group))
    def oval(p,rad,tone=0,group=0,tilt=0):
        ball(p,rmat(2,tilt),rad,tone,group)
    def bone(a,b,r1,r2,tone,group):
        bones.append((np.array(a,dtype=float), np.array(b,dtype=float),r1,r2,tone,group))
    # Muscular heel, metacarpals, thenar pad and hypothenar pad.
    oval((0,.12,-.03),(3.83,4.8,1.38))
    oval((.05,2.02,-.06),(3.95,2.78,1.14))
    oval((1.52,-.70,.68),(1.96,3.05,1.07),.012,0,-17)
    oval((-1.74,-.44,.54),(1.68,3.5,.91),-.018,0,9)
    # Wrist runs into the heel. Long arm bones don't influence the crop.
    wrist=np.array([-.25,-3.43,-.18])
    armslant = r.uniform(-.14,.10)
    if letter in 'COD': armslant -= .12
    armdir=unit((armslant,-1,-.11))
    # Finger states specify MCP, PIP and DIP joint flexion. The spread angle is
    # in the plane of the palm, positive toward the thumb.
    fisted=[81,99,64]; curled=[50,95,76]; straight=[-2,4,2]
    flex=np.tile(fisted,(4,1)).astype(float); spread=np.array([2,0,-2,-4],float)
    def finger(k,angles=straight,s=0): flex[k]=angles;spread[k]=s
    if letter=='A':
        flex[:]=[73,101,72]
    elif letter=='B':
        for k,s in enumerate([-.6,-.2,.3,.9]): finger(k,[-2,4,1],s)
    elif letter in 'CO':
        # Curved fingers curl towards the opposed thumb. Rotate into profile below.
        for k in range(4):
            finger(k,[25+k*3,43+k,31] if letter=='C' else [35+k*2,61,40],-3+k*2)
    elif letter=='D':
        finger(0,straight,2)
        flex[1:]=[49,69,43]
    elif letter=='E':
        flex[:]=[24,111,64]
    elif letter=='F':
        finger(0,[45,80,43],11)
        for k,s in [(1,5),(2,-5),(3,-21)]: finger(k,[-5,6,3],s)
    elif letter=='G': finger(0,straight,5)
    elif letter=='H':
        finger(0,straight,0); finger(1,straight,0)
    elif letter=='I': finger(3,straight,-11)
    elif letter in 'KP':
        finger(0,straight,8)
        finger(1,[34 if letter=='K' else 51,5,1],-24)
    elif letter=='L': finger(0,straight,3)
    elif letter in 'MN':
        # The fingers over the thumb run down its slope, with a bend at
        # the PIP and the tips visible past the distal pad of the thumb.
        for k in range(3 if letter=='M' else 2): finger(k,[37+k*6,102,54],-1)
    elif letter=='Q': finger(0,[-3,7,4],4)
    elif letter=='R':
        finger(0,[4,1,0],-12); finger(1,[-4,4,2],10)
    elif letter=='S': flex[:]=[82,103,61]
    elif letter=='T':
        # Side-on, index and middle point along the hand while the thumb rises.
        # The ordinary tucked-thumb version is also included.
        if r.random()<.90:
            finger(0,[13,7,3],4); finger(1,[16,9,2],0)
    elif letter=='U':
        finger(0,straight,-.5); finger(1,straight,.5)
    elif letter=='V':
        finger(0,straight,14); finger(1,straight,-11)
    elif letter=='W':
        finger(0,straight,17); finger(1,straight,0); finger(2,straight,-15)
    elif letter=='X': finger(0,[2,65,57],2)
    elif letter=='Y': finger(3,straight,-28)

    base=np.array([[2.73,3.03,-.10],[.94,3.57,.03],[-.91,3.28,.07],[-2.63,2.49,.0]])
    lengths=np.array([[3.9,2.42,1.92],[4.43,2.76,2.07],[4.05,2.56,1.9],[3.08,1.96,1.63]])
    radii=[1.03,1.03,.96,.82]
    fattiness=r.uniform(.91,1.11); tall=r.uniform(.92,1.1)
    # Correlated variation avoids implausible finger crossings, but allows
    # closed signs to vary from a loose curve to a tight ball. MCP flexion in
    # a fist changes how far the proximal knuckles protrude above the palm.
    tight=r.uniform(-15,14); hook=r.uniform(-12,13)
    if letter in 'AEMNS':
        flex[:,0]+=tight; flex[:,1]+=hook
        if letter=='S': flex[:,0]+=r.uniform(-13,1)
    if letter in 'VWFY': spread*=r.uniform(.75,1.50)
    if letter=='O': flex[:,0]+=r.uniform(-7,8);flex[:,1]+=r.uniform(-14,6)
    if letter=='Q': flex[0,0]+=r.uniform(0,23)
    fingers=[]
    for k in range(4):
        p=base[k].copy(); angle=0; path=[p]
        dim=lengths[k]*tall*r.uniform(.96,1.04)
        width=radii[k]*fattiness*r.uniform(.96,1.04)
        rr=width*np.array([1,.88,.80,.69]); angles=flex[k]+r.uniform(-5,5,3)
        if np.all(flex[k]<20): angles=r.uniform(-2.5,2.5,3)+flex[k]
        s=np.deg2rad(spread[k]+r.uniform(-3,3)); plane=unit([np.sin(s),np.cos(s),0])
        for j in range(3):
            angle+=np.deg2rad(angles[j]); direction=np.cos(angle)*plane+np.sin(angle)*np.array([0,0,1])
            p1=p+dim[j]*direction; bone(p,p1,rr[j],rr[j+1],-.018*j,k+1); path.append(p1); p=p1
        fingers.append(path)

    # Thumb CMC, MCP, IP and tip; opposition moves its distal pad toward other fingers.
    t0=np.array([2.12,-1.83,.13]); t1=np.array([3.52,.14,1.07]);
    t2=np.array([2.38,1.92,2.25]); t3=np.array([.18,2.57,2.86]);
    if letter in 'AI':
        t1=[3.84,.04,.5];t2=[4.0,2.46,1.5];t3=[3.67,4.53,1.73]
    elif letter=='B':
        t1=[3.45,-.04,.92];t2=[1.82,.73,2.03];t3=[-.46,1.31,2.04]
    elif letter=='C':
        t1=[3.22,-.74,1.84];t2=[3.72,.22,4.57];t3=[3.44,1.47,6.23]
    elif letter in 'DOF':
        k={'D':1,'O':0,'F':0}[letter]; goal=fingers[k][-1]+np.array([.0,-.60,.05]);
        t1=np.array([3.72,-.22,1.33]);t2=t1+.53*(goal-t1)+np.array([.78,-.22,.55]); t3=goal
    elif letter=='E':
        t1=[3.38,-.07,1.35];t2=[1.68,1.06,2.54];t3=[-.68,1.52,2.35]
    elif letter=='G':
        t1=[3.75,.39,.81];t2=[4.05,2.61,1.82];t3=[3.93,4.99,2.08]
    elif letter=='H':
        t1=[3.8,-.01,.54];t2=[3.57,2.11,1.45];t3=[2.58,3.83,1.98]
    elif letter in 'KP':
        t1=[3.52,.11,1.22];t2=[2.26,2.81,1.94];t3=[1.65,5.28,2.08]
    elif letter in 'LQY':
        t1=[3.76,-.35,.56];t2=[5.98,.40,1.20];t3=[8.10,.83,1.54]
        if letter=='Q': t2=[5.08,1.66,2.09];t3=[5.61,3.33,3.31]
        if letter in 'LY':
            # Abduction and elevation of the thumb are independent of pinky spread.
            swivel=rmat(2,r.uniform(7,34))
            t2=t0+swivel@(t2-t0);t3=t0+swivel@(t3-t0)
    elif letter in 'MN':
        t1=[3.3,.11,.92];t2=[1.22,2.28,1.84]; t3=[-1.97 if letter=='M' else -.41,3.33,1.73]
    elif letter=='S':
        t1=[3.61,.42,1.72];t2=[1.85,2.59,3.93];t3=[-.54,3.54,4.29]
    elif letter=='T':
        if flex[0,0]<30:
            t1=[3.69,-.31,.99];t2=[5.84,.32,1.27];t3=[7.4,1.32,1.25]
        else:
            t2=[2.36,2.70,2.22];t3=[1.82,4.51,2.22]
    elif letter=='W':
        t2=[1.60,1.89,2.46]; t3=[-1.28,2.36,2.76]
    elif letter in 'RUVX':
        t2=[1.47,1.77,2.73]; t3=[-.80,2.56,3.01]
    path=np.array([t0,t1,t2,t3],float); path[1:]+=r.uniform(-.28,.28,(3,3)); rr=np.array([1.44,1.24,.99,.82])*fattiness
    for j in range(3): bone(path[j],path[j+1],rr[j],rr[j+1],-.014*j,5)

    # Rigid rotation is shared by the fingers, palm and their surface normals.
    yaw=12; pitch=10; roll=-5; yawr=23; pitchr=19; rollr=13
    if letter in 'CO':
        yaw=69; yawr=18; pitch=1; roll=-14;rollr=17
    elif letter=='D': yaw=21; pitch=9; roll=-2
    elif letter=='F': yaw=36; pitch=7; roll=-5
    elif letter in 'GHT' and (letter!='T' or flex[0,0]<30):
        # Turn the thumb side UP when aiming right (back/ulnar-side view).
        # A palm-forward hand rotated right instead puts its index underneath.
        roll=-81; yaw=145 if letter=='G' else 114; yawr=26; pitch=2; rollr=16
        if letter=='T': yaw=167; yawr=21; roll=-91
    elif letter in 'KP':
        yaw=-33; pitch=12; yawr=27; roll=3; rollr=14
        if letter=='P': roll=73; yaw=101; yawr=30; pitch=-14; pitchr=21
    elif letter=='Q':
        roll=154; yaw=108; yawr=39; pitch=-23; pitchr=25;rollr=26
    elif letter in 'MNS':
        yaw=28; pitch=28; roll=-13; rollr=22;pitchr=25;yawr=33
    elif letter=='X': yaw=70; yawr=25; pitch=8;roll=-7
    elif letter=='Y': yaw=0; roll=-3
    elif letter=='R': yaw=20; yawr=24; pitch=16;roll=-4;rollr=18
    elif letter=='A': pitch=20;roll=-7
    if letter=='O': yaw=94; yawr=15
    if letter=='C': yaw=88; yawr=16
    if letter=='P': roll=79; yaw=29; yawr=35; pitch=-15;pitchr=29
    if letter=='Q':
        # Fingers point to the floor with the wrist bent forward over them.
        # Look along the back of the hand rather than up at the palm.
        yaw=35; yawr=52; pitch=34; pitchr=24;roll=168;rollr=22
    if letter=='H': yaw=151; yawr=28
    if letter=='K': yaw=-44; yawr=24; pitch=20; pitchr=16
    if letter in 'CO': roll-=r.uniform(0,16)
    if letter=='S':
        # A fist flexed towards the forearm, thumb wrapped over the fingers.
        yaw=r.uniform(-14,29); yawr=16;pitch=28; pitchr=29
    if letter=='G' and r.random()<.5: yaw=r.uniform(113,155)
    if letter=='O': pitchr=25; yawr=20
    if letter=='U': yaw=18; yawr=18
    if letter=='R': yaw=24; yawr=16
    if letter=='Y': roll=-2; rollr=18; yawr=16
    R=rmat(2,roll+r.uniform(-rollr,rollr))@rmat(0,pitch+r.uniform(-pitchr,pitchr))@rmat(1,yaw+r.uniform(-yawr,yawr))
    R=R@np.diag([r.uniform(.92,1.08),r.uniform(.94,1.06),1])
    if letter in 'GHT':
        # A sideways pointing arm presents a flexed wrist. Bend the fingers
        # onto the pointing axis while leaving the palm and wrist oblique.
        if letter!='T' or flex[0,0]<30:
            turn=r.uniform(37,69) if letter=='T' else r.uniform(5,52)
            Rr=rmat(2,turn)@R; undo=np.linalg.inv(Rr)@rmat(2,-turn)@Rr
            bent=[]
            for a,b,ra,rb,tone,g in bones:
                if 1<=g<=4 and flex[g-1,0]<30:
                    p=base[g-1];a=p+undo@(a-p);b=p+undo@(b-p)
                elif g==5 and letter=='T':
                    # Thumb rises upward from the upper side of the palm.
                    u=np.linalg.inv(Rr)@rmat(2,-.7*turn)@Rr;p=path[0];a=p+u@(a-p);b=p+u@(b-p)
                bent.append((a,b,ra,rb,tone,g))
            bones=bent; R=Rr
    # Place the forearm behind the heel. In a downward/sideward sign the wrist
    # is bent; the forearm is not simply collinear with the metacarpals.
    armroot=R@wrist; d=unit(R@armdir)
    if letter in 'PGHQT' and (letter!='T' or flex[0,0]<30):
        dd= {'G':[-1,-.54,-.30], 'H':[-1,-.48,-.20], 'T':[-.83,-.78,-.2], 'P':[.87,-.60,-.35], 'Q':[.40,-1,-.45]}[letter]
        d=unit(np.array(dd)+r.uniform(-.30,.30,3))
        if letter=='Q':
            # The wrist is at the upper/rear side of the hanging digits;
            # foreshorten the palm, with the forearm continuing downward.
            tilt=r.uniform(.36,.72)
            palm=np.linalg.inv(R)@np.diag([1.,tilt,1.])@R
            balls=[(palm@p,palm@axes,tone,g) for p,axes,tone,g in balls]
            result=[]
            for a,b,ra,rb,tone,g in bones:
                met=base[g-1] if 1<=g<=4 else t0
                move=palm@met-met; result.append((a+move,b+move,ra,rb,tone,g))
            bones=result;armroot=R@(palm@wrist)
            d=unit(np.array([r.uniform(.0,.52),-1,-.35]))
    return balls,bones,R,armroot,d


def render(letter, r, hi=56):
    balls,bones,R,root,d=model(letter,r)
    ellipsoids=[]; capsules=[]; bounds=[]
    for p,A,tone,g in balls:
        p=R@p; A=R@A
        delta=np.sqrt((A*A).sum(axis=1)); bounds.extend([p-delta,p+delta]); ellipsoids.append((p,A,tone,g))
    for a,b,ra,rb,tone,g in bones:
        a=R@a; b=R@b
        for p,v in [(a,ra),(b,rb)]: bounds.extend([p-v,p+v])
        capsules.append((a,b,ra,rb,tone,g))
    pts=np.array(bounds)[:,:2]; low=pts.min(axis=0); high=pts.max(axis=0)
    # A square crop containing the hand, with variable margins and a little wrist.
    ha=high-low; side=max(ha)*r.uniform(.93,1.39)
    middle=(low+high)*.5 + side*(r.uniform(-.065,.065,2)+[0,-.046]);
    step=side/hi; xx=middle[0]+(np.arange(hi)-hi/2+.5)*step; yy=middle[1]-(np.arange(hi)-hi/2+.5)*step
    zz=np.full((hi,hi),-99.); tones=np.zeros_like(zz);groups=np.full((hi,hi),-1,int)
    def paste(patch, tone,g, xin,yin):
        dest=zz[yin,xin]; better=patch>dest
        np.maximum(patch,dest,out=dest); np.copyto(tones[yin,xin],tone,where=better); np.copyto(groups[yin,xin],g,where=better)
    def limits(l,h):
        i0=max(0,int(math.floor((l[0]-xx[0])/step)));i1=min(hi,int(math.ceil((h[0]-xx[0])/step))+1)
        j0=max(0,int(math.floor((yy[0]-h[1])/step)));j1=min(hi,int(math.ceil((yy[0]-l[1])/step))+1)
        return slice(i0,max(i0,i1)),slice(j0,max(j0,j1))
    for p,A,tone,g in ellipsoids:
        delta=np.linalg.norm(A,axis=1); xs,ys=limits(p-delta,p+delta)
        # The projection of an ellipsoid is an ellipse; solve the ray quadratic.
        C=A@A.T; S=C[:2,:2]; det=S[0,0]*S[1,1]-S[0,1]**2
        M=np.array([[S[1,1],-S[0,1]],[-S[0,1],S[0,0]]])/det
        v=C[2,:2]@M; h=np.sqrt(max(.00001,C[2,2]-v@C[:2,2]))
        X=xx[xs][None,:]-p[0];Y=yy[ys][:,None]-p[1]
        dist=1-M[0,0]*X*X-2*M[1,0]*X*Y-M[1,1]*Y*Y
        z=p[2]+v[0]*X+v[1]*Y+h*np.sqrt(np.maximum(0,dist))
        paste(np.where(dist>0,z,-100),tone,g,xs,ys)
    capsules.append((root,root+18*d,2.54,3.36,-.03,0))
    for a,b,ra,rb,tone,g in capsules:
        xs,ys=limits(np.minimum(a-ra,b-rb),np.maximum(a+ra,b+rb)); X=xx[xs][None,:]-a[0];Y=yy[ys][:,None]-a[1]
        # Upper envelope of a linearly tapered family of spheres. At each
        # ray, squared sphere height is a concave quadratic in bone parameter.
        v=b-a; dr=rb-ra; A=v[0]**2+v[1]**2-dr*dr; B=X*v[0]+Y*v[1]+ra*dr; C=ra*ra-X*X-Y*Y
        if A>.0001:
            t0=B/A; H=np.maximum(0,C+B*t0);t=np.clip(t0+v[2]*np.sqrt(H/(A*(A+v[2]**2))),0,1)
            h=C+(2*B-A*t)*t; z=np.where(h>0,a[2]+v[2]*t+np.sqrt(np.maximum(0,h)),-100)
        else:
            h1=C+2*B-A
            z=np.maximum(np.where(C>0,a[2]+np.sqrt(np.maximum(0,C)),-100),np.where(h1>0,b[2]+np.sqrt(np.maximum(0,h1)),-100))
        paste(z,tone,g,xs,ys)
    # Shade the visible surface. Depth steps create crease shadows, while the
    # smooth inside of an individual digit has conventional diffuse shading.
    mask=zz>-98; center=np.where(mask,zz,0)
    def nb(dy,dx): return np.roll(center,(dy,dx),(0,1))
    def same(dy,dx): return np.roll(mask,(dy,dx),(0,1))
    def change(dy,dx):
        dif=nb(dy,dx)-center
        return np.where(same(dy,dx),np.clip(dif,-2.8*step,2.8*step),-1.7*step)
    gx=(change(0,-1)-change(0,1))/(2*step);gy=(change(1,0)-change(-1,0))/(2*step)
    light=unit([r.uniform(-1.3,-.20) if r.random()<.85 else r.uniform(.2,1),r.uniform(.22,1.2),r.uniform(.35,.95)])
    dot=np.maximum(-.5,(light[2]-gx*light[0]-gy*light[1])/np.sqrt(1+gx*gx+gy*gy))
    shadow=np.zeros_like(zz)
    for dy,dx in [(0,1),(0,-1),(1,0),(-1,0),(2,1)]:
        gap=nb(dy,dx)-center; shadow+=np.where(same(dy,dx),np.clip((gap-step*.6)/1.8,0,1),0)
    # Ambient occlusion also delineates fingers at places where neighboring
    # surfaces meet at nearly the same depth. A grazing contour has less fill.
    seam=np.zeros_like(zz)
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        seam+=same(dy,dx) & (groups != np.roll(groups,(dy,dx),(0,1)))
    grazing=1-1/np.sqrt(1+gx*gx+gy*gy)
    skin=.48 + .42*dot + tones - .074*shadow - .050*seam - .105*grazing
    bg=r.uniform(.42,.84) if r.random()<.77 else r.uniform(.17,.47)
    # A wall in directional light need not be constant across the crop.
    ramp=np.linspace(-.5,.5,hi)
    wall=bg+r.uniform(-.10,.10)*ramp[None,:]+r.uniform(-.25,.25)*ramp[:,None]
    pic=np.where(mask,skin,wall)
    return pic.reshape(28,hi//28,28,hi//28).mean(axis=(1,3)).astype(np.float32)


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(SOLUTION)
    funcs={};exec(SOLUTION,funcs)
    result=[]
    for k,letter in enumerate('ABCDEFGHIKLMNOPQRSTUVWXY'):
        r=np.random.default_rng(1701+k)
        for packet in range(40):
            x=np.stack([render(letter,r) for j in range(32)])
            result.append(funcs['describe'](x))
    np.savez(out/'constants.npz',shape=np.concatenate(result,axis=0))

if __name__=='__main__': main()
