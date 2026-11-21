"""Build a dictionary of independent analytic hand renders. No input files."""
import math
import sys
from pathlib import Path
import numpy as np

# This extractor runs both on one rendered view and on one observed image.
features = '''import numpy as np

def blur(a, n):
    for k in range(n):
        p = np.pad(a, ((0,0),(0,0),(1,1)), mode='edge')
        q = (p[:,:,:-2] + p[:,:,2:] + 2*a)*0.25
        p = np.pad(q, ((0,0),(1,1),(0,0)), mode='edge')
        a = (p[:,:-2] + p[:,2:] + 2*q)*0.25
    return a

def derivatives(a):
    p = np.pad(a, ((0,0),(1,1),(1,1)), mode='edge')
    return (p[:,1:-1,2:]-p[:,1:-1,:-2])*0.5, (p[:,2:,1:-1]-p[:,:-2,1:-1])*0.5

def sample(a, xs, ys):
    n,h,w = a.shape
    xs = np.clip(xs, 0, w-1.0001)
    ys = np.clip(ys, 0, h-1.0001)
    i = xs.astype(np.int32)
    j = ys.astype(np.int32)
    fx = xs-i
    fy = ys-j
    r = np.arange(n)[:,None,None]
    return ((1-fy)*((1-fx)*a[r,j,i]+fx*a[r,j,i+1]) + fy*((1-fx)*a[r,j+1,i]+fx*a[r,j+1,i+1])).astype(np.float32)

def normalize(v):
    return v / np.sqrt(0.000001+(v*v).sum(1,keepdims=True))

def featuremaps(gray, mask):
    # Undirected orientations for contours of either polarity, and signed
    # bandpass for dark creases between digits. Bound the response of strong
    # edges so clothing cannot dominate merely through higher contrast.
    ax,ay = derivatives(blur(gray,1))
    mag = np.sqrt(ax*ax+ay*ay+0.00000001)
    gain = 1 / (0.019 + 0.6*blur(mag,4) + 0.4*mag)
    mx,my = derivatives(blur(mask,1))
    mm = np.sqrt(mx*mx+my*my+0.00000001)
    channels = []
    for cs,sn in [(1,0),(0.70710678,0.70710678),(0,1),(-0.70710678,0.70710678)]:
        z = np.maximum(0,(ax*cs+ay*sn)**2/mag-0.28*mag)*gain
        edge = np.maximum(0,(mx*cs+my*sn)**2/mm-0.28*mm)
        channels.append(blur(z+0.9*edge,2))
    base = blur(gray,1)
    band = base-blur(base,10)
    adjust = np.maximum(0.022,blur(abs(band),5))
    band = np.clip(band/adjust,-2,2)
    return channels+[blur(mask,1),band]

def extract(maps,xx,yy):
    f = np.stack([sample(a,xx,yy) for a in maps],axis=3)
    return f

def describe(f):
    # A local descriptor for hand shape, independent of other windows. The
    # patch Hann-like weighting reduces contamination from the background.
    n,h,w,c=f.shape
    wx=1-.6*(2*(np.arange(w,dtype=np.float32)+.5)/w-1)**4
    wy=1-.6*(2*(np.arange(h,dtype=np.float32)+.5)/h-1)**4
    wt=wx[None,None,:,None]*wy[None,:,None,None]
    edge=f[:,:,:,:4]; edge=(edge-0.82*edge.mean((1,2),keepdims=True))*wt
    sh=f[:,:,:,4:5];sh=(sh-sh.mean((1,2),keepdims=True))*wt
    band=f[:,:,:,5:6]*wt
    return np.concatenate((normalize(edge.reshape(n,-1)),.76*normalize(sh.reshape(n,-1)),.55*normalize(band.reshape(n,-1))),1)
'''

runtime = '''
def observed(a):
    # Smooth background field inferred from the two upper corners, independently
    # per photograph. The cap prevents a dark object at one side dominating it.
    top = blur(a,2)
    l = np.median(top[:,:4,:4].reshape(len(a),-1),axis=1)
    r = np.median(top[:,:4,-4:].reshape(len(a),-1),axis=1)
    l,r = np.maximum(l,r-0.16),np.maximum(r,l-0.16)
    row = np.arange(28, dtype=np.float32)[None,:,None]/27
    col = np.arange(28, dtype=np.float32)[None,None,:]/27
    bg = l[:,None,None]*(1-col)+r[:,None,None]*col
    s = blur(a,1)
    # Estimate gentle vertical change separately on each side. Avoid carrying
    # a sleeve or a person at the boundary across the entire photograph.
    lside = np.median(s[:,:,:3],axis=2)[:,:,None]
    rside = np.median(s[:,:,-3:],axis=2)[:,:,None]
    drift = np.clip(lside-l[:,None,None],-.13,.13)*(1-col)+np.clip(rside-r[:,None,None],-.13,.13)*col
    residual = s-bg-drift
    gx,gy = derivatives(s)
    texture = blur(np.sqrt(gx*gx+gy*gy),3)
    # Photographic backdrops are smoother than the fingers. Use both signs
    # of contrast and also retain very dark regions next to a pale hand.
    dark = (-residual + .055*row - .035)/.115
    light = (residual - .055*row - .060)/.14
    rough = (texture-.022)/.066
    strength = np.clip(np.maximum(np.maximum(dark,light),rough),0,1)
    strength = blur(strength,2)
    # Close narrow dark/light bands inside the hand without closing large
    # gaps such as the space between two separated fingers.
    p = np.pad(strength,((0,0),(1,1),(1,1)),mode='edge')
    lifted = np.maximum.reduce([p[:,i:i+28,j:j+28] for i,j in [(0,1),(1,0),(1,1),(2,1),(1,2)]])
    p = np.pad(lifted,((0,0),(1,1),(1,1)),mode='edge')
    strength = np.minimum.reduce([p[:,i:i+28,j:j+28] for i,j in [(0,1),(1,0),(1,1),(2,1),(1,2)]])
    return strength

def localmatch(maps, windows, cells, model):
    # Only compare homologous patches within a single pose: left/right,
    # distal/proximal. Allow each patch one cell of relative movement.
    h = len(windows)
    pos = cells[None,None,None,:]
    xx = windows[:,:,0,None,None] + windows[:,:,2,None,None]*pos
    yy = windows[:,:,1,None,None] + windows[:,:,2,None,None]*pos.transpose(0,1,3,2)
    xx=xx+np.zeros_like(yy);yy=yy+np.zeros_like(xx)
    xx=xx.reshape(h,-1,len(cells));yy=yy.reshape(h,-1,len(cells))
    f=extract(maps,xx,yy).reshape(h,24,len(cells),len(cells),6)
    ps=[]
    # Use a grip patch and two patches spanning the upper digits. They
    # overlap slightly so a thin finger is not lost at a partition edge.
    for y,x in [(2,2),(2,8),(8,5)]:
        best=np.full((h,24),-4,np.float32)
        for dy,dx in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
            q=f[:,:,y+dy:y+dy+8,x+dx:x+dx+8,:].reshape(h*24,8,8,6)
            d=describe(q).reshape(h,24,-1)
            best=np.maximum(best,(d*model[:,:,len(ps)]).sum(2))
        ps.append(best)
    return (ps[0]+ps[1]+ps[2])/3

def predict(x):
    n = len(x)
    result = np.empty(n, dtype=np.int32)
    bank = CONSTANTS['poses']
    # Fixed block size, including the final partial block, ensures identical
    # floating point operations for any image order or call size.
    block = 64
    coord = (np.arange(14,dtype=np.float32)*2+0.5)-13.5
    gridx = coord[None,None,:]
    gridy = coord[None,:,None]
    for start in range(0,n,block):
        size = min(n-start,block)
        a = np.zeros((block,28,28),np.float32)
        a[:size] = x[start:start+size,0]
        m = observed(a)
        maps = featuremaps(a,m)
        score = np.full((block,24),-100,np.float32)
        windows = np.zeros((block,24,3),np.float32)
        index = np.zeros((block,24),np.int32)
        # Search modest position changes as well as a little extra zoom.
        for zoom in (1.0,0.83,1.17):
            for dx,dy in [(0,0),(-2,-1),(2,-1),(-1.5,1.5),(1.5,1.5)]:
                # Border replication would extend a tip or a wrist. This is
                # suitable for a wrist; the top and sides have little weight.
                xx = 13.5+gridx*zoom+dx
                yy = 13.5+gridy*zoom+dy
                feat = describe(extract(maps,xx,yy))
                sim = (feat @ bank.T).reshape(block,24,-1)
                best=sim.argmax(2)
                fit=np.take_along_axis(sim,best[:,:,None],axis=2)[:,:,0]
                better=fit>score
                windows[better]=np.array([13.5+dx,13.5+dy,zoom],np.float32)
                index=np.where(better,best,index)
                score=np.maximum(score,fit)
        ids=index+np.arange(24)[None,:]*(len(bank)//24)
        patch=localmatch(maps,windows,np.arange(18,dtype=np.float32)*2-17,CONSTANTS['parts'][ids])
        result[start:start+size] = (score+0.85*patch)[:size].argmax(1)
    return result
'''

# ---- Articulated geometry; units are fractions of a palm half-width.
# Joint-angle conventions: 0=straight, positive flexion toward the palm;
# positive abduction moves radially, i.e. toward the thumb. Coordinates are
# hand-fixed, not photographic coordinates.
rad = math.pi/180
rng = np.random.RandomState(42119)
def uniform(a,b): return rng.uniform(a,b)
def v3(x,y,z): return np.array([x,y,z],dtype=float)
def mat(axis,angle):
    c,s=math.cos(angle*rad),math.sin(angle*rad)
    if axis=='x':return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis=='y':return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def model(letter):
    # Primitives are generalized capsules (segments with ellipsoidal ends).
    bodies=[]; phalanges=[]
    def part(ls,a,b,r1,r2=None,ratio=(1,1,1),material=0):
        ls.append((np.asarray(a),np.asarray(b),r1,r1 if r2 is None else r2,np.array(ratio),material))
    slim=uniform(.88,1.1); length=uniform(.92,1.13)
    # Palm, thenar and hypothenar pads, carpus and tapered forearm.
    part(bodies,(0,-.72,-.07),(-.025,.60,-.055),.78,.92,(1.07,.87,.44))
    part(bodies,(-.05,-1.08,-.07),(0,-.68,-.07),.57,.77,(1,.78,.55))
    part(bodies,(.30,-.69,.11),(.55,-.01,.12),.41,.42,(1,.98,.60))
    part(bodies,(-.40,-.62,.03),(-.63,.38,.05),.34,.30,(.94,1,.73))
    armlean = uniform(-.28,.19)
    part(bodies,(-.05,-1.05,-.08),(-.08+armlean,-2.75,-.20),.57,.65,(1,.95,.70))
    part(bodies,(-.08+armlean,-2.7,-.20),(-.12+2.4*armlean,-5.8,-.45),.65,.85,(1,1,.77))
    # Floating metacarpophalangeal joints: the knuckle line arches and the
    # little finger is set lower and has a shorter proximal phalanx.
    xs=np.array([-.72,-.275,.18,.615])*uniform(.96,1.05)
    ys=np.array([.56,.84,.94,.85]) * length
    zbase=[-.015,.01,.025,.02]
    lens=np.array([1.19,1.62,1.77,1.59])*length
    widths=np.array([.196,.227,.244,.23])*slim
    spread=np.array([-8.,-3.,1.,6.]); straight=(3,4,3)
    folded=(81,90,58); hooked=(7,92,66)
    angles=[folded]*4
    thumb='wrap'; thumbAt=1; curlnoise=7
    yaw=uniform(-13,20);pitch=uniform(-10,18);roll=uniform(-12,12)
    if letter=='A':
        angles=[(67,99,65)]*4;thumb='side';yaw+=uniform(-12,14)
    elif letter=='B':
        angles=[straight]*4;spread=np.array([3,1,-1,-3.]);thumb='flat'
    elif letter in 'CO':
        a=uniform(18,40) if letter=='C' else uniform(29,52)
        b=uniform(34,55) if letter=='C' else uniform(46,68)
        angles=[(a+i*1.5,b,34) for i in [3,2,1,0]]
        spread=np.array([3,1,0,-2.]);thumb='arc' if letter=='C' else 'contact'
        thumbAt=3;yaw=uniform(48,88);pitch=uniform(-12,18);roll=uniform(-20,2)
    elif letter=='D':
        angles=[(54,69,54)]*3+[straight];thumb='contact';thumbAt=2
        yaw+=uniform(-5,30);spread[3]=uniform(-9,5)
    elif letter=='E':
        angles=[(26,100,65)]*4;thumb='shelf';yaw+=uniform(-14,12)
    elif letter=='F':
        angles=[straight,straight,straight,(36,73,55)];spread=np.array([-24,-9,13,8.]);thumb='contact';thumbAt=3
        yaw=uniform(20,75);roll+=uniform(-10,4)
    elif letter in 'GHT':
        # Rotate the palm as well as the extended digits. Side views see
        # the G pinch almost edge-on; H shows the backs of the two digits.
        angles[3]=straight
        if letter=='H':angles[2]=straight
        spread[2:]=[0,0];thumb='sideways' if letter=='G' else 'wrap' if letter=='H' else 'upbehind'
        roll=uniform(-97,-67);yaw=uniform(-53,42);pitch=uniform(-14,32)
        if letter=='H':yaw=uniform(-146,-64)
        if letter=='T':roll=uniform(-108,-79);yaw=uniform(-49,20)
        # The foreshortened hypothenar edge can lie behind the pinch, not
        # necessarily directly opposite the fingertips.
        if letter in 'GT' and uniform(0,1)<.5:
            roll+=30; yaw=uniform(-81,-43);angles[3]=(uniform(38,67),5,0)
    elif letter=='I':angles[0]=straight;thumb='wrap';spread[0]=uniform(-17,-5)
    elif letter=='K':
        angles[2]=(uniform(30,62),7,5);angles[3]=straight
        spread[2:]=[-15,12];thumb='between';yaw=uniform(-53,22)
    elif letter=='L':angles[3]=straight;thumb='out';spread[3]=1
    elif letter in 'MN':
        n=3 if letter=='M' else 2
        angles=[(81,95,60)]*(4-n)+[(49,83,47)]*n
        thumb='under';thumbAt=4-n;yaw+=uniform(-15,9)
    elif letter=='P':
        angles[2]=(uniform(43,78),7,4);angles[3]=straight;spread[2:]=[-19,8]
        thumb='between';yaw=uniform(-77,-42);roll=uniform(77,117);pitch=uniform(-15,16)
    elif letter=='Q':
        angles[3]=straight;thumb='q';yaw=uniform(-58,10);roll=uniform(136,177);pitch=uniform(12,51)
    elif letter=='R':
        angles[2]=(9,7,6);angles[3]=(0,6,4)
        spread[2:]=[uniform(8,13),uniform(-13,-7)];thumb='wrap';thumbAt=1
    elif letter=='S':
        angles=[(78,95,61)]*4;thumb='over'
    elif letter=='U':
        angles[2:]=[straight]*2;spread[2:]=[4,1];thumb='wrap'
    elif letter=='V':
        angles[2:]=[straight]*2;spread[2:]=[uniform(-23,-9),uniform(11,23)];thumb='wrap'
    elif letter=='W':
        angles[1:]=[straight]*3;spread[1:]=[uniform(-25,-12),-2,uniform(13,26)];thumb='hold';thumbAt=0
    elif letter=='X':
        angles[3]=(uniform(-15,12),uniform(64,96),uniform(31,58));thumb='wrap';yaw=uniform(22,84)
    elif letter=='Y':
        angles[0]=straight;spread[0]=uniform(-40,-17);thumb='out';
    else:raise ValueError(letter)
    tips=[]; joints=[]
    def chain(points,radii):
        for i in range(len(points)-1):
            part(phalanges,points[i],points[i+1],radii[i],radii[i+1],material=(1 if i==len(points)-2 else 0))
    for k in range(4):
        p=v3(xs[k],ys[k],zbase[k]);points=[p]
        a=(spread[k]+uniform(-3,3))*rad
        flex=0; rr=widths[k]*uniform(.96,1.04)
        for link,(fract,angle) in enumerate(zip([.475,.298,.227],angles[k])):
            flex+=(angle+uniform(-curlnoise,curlnoise))*rad
            direction=v3(math.sin(a)*math.cos(flex),math.cos(a)*math.cos(flex),math.sin(flex))
            p=p+lens[k]*fract*direction;points.append(p)
        chain(points,rr*np.array([1.03,1,.885,.755]));tips.append(points[-1]);joints.append(points)
    # Opposable thumb with large metacarpal fat pad and two shorter segments.
    # Endpoints chosen relative to the joint or intervening space they touch.
    p=v3(.46,-.58,.10); q=v3(.79,-.11,.22);r=v3(.67,.24,.52);t=v3(.0,.54,.74)
    if thumb=='side':q=v3(.82,-.07,.16);r=v3(.96,.60,.24);t=v3(.92,1.10,.33)
    if thumb=='flat':r=v3(.44,.12,.47);t=v3(-.22,.35,.46)
    if thumb=='shelf':r=v3(.49,.09,.64);t=v3(-.38,.11,.65)
    if thumb=='wrap':r=v3(.53,.22,.54);t=np.array(joints[1][-1])+v3(.10,-.03,.16)
    if thumb=='over':r=v3(.48,.49,.91);t=v3(-.26,1.01,1.01)
    if thumb=='under':
        q=v3(.76,-.08,.19);r=v3(.36,.19,.48);t=v3(xs[thumbAt]-.22,.55,.60)
    if thumb=='hold':r=v3(.29,.17,.50);t=np.array(tips[0])+v3(.05,-.04,.15)
    if thumb=='out' or thumb=='q':
        a=uniform(19,45)*rad
        q=v3(.95,-.33,.23);direction=v3(math.cos(a),math.sin(a),uniform(-.12,.27));r=q+.62*direction;t=r+.55*direction+v3(0,.055,.01)
        if thumb=='q':q=v3(.86,-.16,.33);r=v3(1.02,.48,.43);t=v3(1.09,1.09,.48)
    if thumb in ('arc','contact'):
        q=v3(.91,-.12,.38);r=v3(.92,.08,.99)
        if thumb=='contact':t=tips[thumbAt].copy();r=(r+t)*.52+v3(.06,-.14,.06)
        else:t=v3(.94,.48,1.66);r=v3(.96,.03,1.08)
    if thumb=='between':q=v3(.88,-.06,.24);r=v3(.64,.59,.44);t=v3(.44,1.23,.50)
    if thumb=='sideways':q=v3(.89,.06,.32);r=v3(.95,.73,.56);t=v3(.90,1.32,.67)
    if thumb=='upbehind':q=v3(.91,.03,.17);r=v3(1.18,.57,.16);t=v3(1.39,1.01,.20)
    r+=rng.uniform(-.07,.07,3);t+=rng.uniform(-.07,.07,3)
    chain([p,q,r,t],slim*np.array([.363,.302,.253,.203]))
    # The forearm bends independently from the hand, particularly in signs
    # presented horizontally or downwards. Apply this distal to the wrist.
    rot=mat('z',roll)@mat('x',pitch)@mat('y',yaw)
    return bodies,phalanges,rot

resolution=60
grid=np.arange(resolution,dtype=float)+.5

def render(letter):
    bodies,digits,rot=model(letter)
    # Enclose just the hand (not the forearm) when aiming the camera.
    # This calculation and every subsequent normalization is for THIS view.
    main=bodies[:4]+digits
    lo=np.full(2,np.inf);hi=-lo
    for a,b,ra,rb,ratio,material in main:
        ex=np.sqrt(((rot[:2]*ratio)**2).sum(1))
        for p,r in [(a,ra),(b,rb)]:
            c=(rot@p)[:2];lo=np.minimum(lo,c-r*ex);hi=np.maximum(hi,c+r*ex)
    horizontal=letter in 'GHTP'; down=letter=='Q'
    # Tighter framing for more compact poses is not necessary: allow the
    # wrist and a portion of the forearm to lie in the square as well.
    span=hi-lo
    fill=uniform(.69,.96) if not horizontal and not down else uniform(.72,1.03)
    scale=resolution*fill/max(span)
    ox=resolution*.5-scale*(lo[0]+hi[0])*.5+uniform(-.035,.035)*resolution
    oy=resolution*uniform(.015,.12)+scale*hi[1]
    if horizontal or down:
        ox+=resolution*(uniform(.02,.20) if letter=='P' or down else uniform(-.19,-.015))
        oy+=resolution*uniform(.035,.16)
    # Side-lit matte surface with gentle ambient occlusion. Shape and light
    # parameters are sampled before drawing any pixels.
    side=-1 if uniform(0,1)<.84 else 1
    light=v3(side*uniform(.56,.95),uniform(.15,.78),uniform(.09,.68));light/=np.linalg.norm(light)
    skin=uniform(.40,.68);bg=uniform(.50,.85);directional=uniform(.31,.52)
    depth=np.full((resolution,resolution),-100.0);lit=np.full_like(depth,bg)
    for seq,(a,b,ra,rb,ratio,material) in enumerate(bodies+digits):
        # Exact orthographic intersection with an ellipsoid swept along a
        # line, approximating its taper by interpolating endpoint radii.
        a=rot@a; b=rot@b
        axisrot=rot
        if seq in (4,5):
            # Flex the wrist without twisting the hand sign. The elbow hangs
            # below the hand, while pointing gestures can enter from a side.
            pivot=rot@bodies[4][0]
            if seq==4:
                if down: bend=uniform(-158,-98)
                elif letter=='P': bend=uniform(-64,-14)
                elif horizontal: bend=uniform(19,75)
                else: bend=uniform(-25,9)
            turn=mat('z',bend);a=pivot+turn@(a-pivot);b=pivot+turn@(b-pivot);axisrot=turn@rot
        # proximal metacarpal and arm are anisotropic; use the joint's metric
        M=(axisrot*(1/(ratio*ratio)))@axisrot.T
        ext=np.sqrt(((axisrot[:2]*ratio)**2).sum(1))
        left,bot=np.minimum(a[:2]-ra*ext,b[:2]-rb*ext)
        right,top=np.maximum(a[:2]+ra*ext,b[:2]+rb*ext)
        x0=max(0,int(np.floor(ox+left*scale-1)));x1=min(resolution,int(np.ceil(ox+right*scale+1)))
        y0=max(0,int(np.floor(oy-top*scale-1)));y1=min(resolution,int(np.ceil(oy-bot*scale+1)))
        if x1<=x0 or y1<=y0:continue
        u=(grid[None,x0:x1]-ox)/scale-a[0]; v=(oy-grid[y0:y1,None])/scale-a[1]
        # Eliminate z in the ellipsoid metric to project its footprint.
        D=b-a; zz=M[2,2];xz=M[0,2]/zz;yz=M[1,2]/zz
        aa=M[0,0]-M[0,2]*xz;bb=M[0,1]-M[0,2]*yz;cc=M[1,1]-M[1,2]*yz
        ku=aa*D[0]+bb*D[1]; kv=bb*D[0]+cc*D[1]; den=ku*D[0]+kv*D[1]
        frac=np.clip((ku*u+kv*v)/(den+1e-9),0,1)
        du=u-frac*D[0];dv=v-frac*D[1];r=ra+(rb-ra)*frac
        cap=r*r-aa*du*du-2*bb*du*dv-cc*dv*dv
        # Move the point of tangency toward the nearer endpoint when a
        # finger is strongly foreshortened (cylinder rather than flat ribbon).
        heightSlope=D[2]+xz*D[0]+yz*D[1]
        adjust=heightSlope*np.sqrt(np.maximum(0,cap)/(den*(den/zz+heightSlope*heightSlope)+1e-10))
        frac=np.clip(frac+adjust,0,1);du=u-frac*D[0];dv=v-frac*D[1];r=ra+(rb-ra)*frac
        cap=r*r-aa*du*du-2*bb*du*dv-cc*dv*dv
        dz=-xz*du-yz*dv+np.sqrt(np.maximum(cap,0)/zz)
        z=a[2]+D[2]*frac+dz
        old=depth[y0:y1,x0:x1]; keep=(cap>0)&(z>old)
        nx=M[0,0]*du+M[0,1]*dv+M[0,2]*dz;ny=M[1,0]*du+M[1,1]*dv+M[1,2]*dz;nz=M[2,0]*du+M[2,1]*dv+zz*dz
        dot=(nx*light[0]+ny*light[1]+nz*light[2])/(np.sqrt(nx*nx+ny*ny+nz*nz)+1e-8)
        color=skin+directional*(.67*dot+.33*np.maximum(dot,0))
        old[keep]=z[keep];lit[y0:y1,x0:x1][keep]=color[keep]
    inside=depth>-90
    # Nearby, visibly higher skin darkens recessed skin and clarifies overlap.
    occ=np.zeros_like(lit);z=np.pad(depth,2,mode='edge')
    for i,j in [(0,2),(1,0),(2,4),(4,2),(0,0),(4,4)]:
        occ+=np.clip((z[i:i+resolution,j:j+resolution]-depth)*1.8,0,1)
    lit=np.clip(lit-.042*occ*inside,0,1)
    # Area sample through a separable pixel footprint (60 -> 28).
    image=shrink@lit@shrink.T; mask=shrink@inside.astype(float)@shrink.T
    return image.astype(np.float32),mask.astype(np.float32)

edges=np.linspace(0,resolution,29)
shrink=np.maximum(0,np.minimum(edges[1:,None],grid[None,:]+.5)-np.maximum(edges[:-1,None],grid[None,:]-.5))*(28/resolution)

def build(out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(features+runtime)
    env={};exec(features,env);extract=env['extract'];describe=env['describe']
    letters='ABCDEFGHIKLMNOPQRSTUVWXY'
    # Fixed number of independent poses, not mined or filtered by score.
    poses=640
    bank=np.empty((len(letters)*poses,14*14*6),np.float32)
    patches=np.empty((len(bank),3,8*8*6),np.float32)
    nodes=np.arange(18,dtype=np.float32)*2-3.5
    for c,letter in enumerate(letters):
        for begin in range(0,poses,32):
            ims,ms=zip(*(render(letter) for i in range(32)))
            maps=env['featuremaps'](np.stack(ims),np.stack(ms))
            f=extract(maps,nodes[None,None,:],nodes[None,:,None])
            sl=slice(c*poses+begin,c*poses+begin+32)
            bank[sl]=describe(f[:,2:16,2:16,:])
            for j,(y,x) in enumerate([(2,2),(2,8),(8,5)]):
                patches[sl,j]=describe(f[:,y:y+8,x:x+8,:])
    np.savez(out/'constants.npz',poses=bank,parts=patches)

if __name__=='__main__':build(Path(sys.argv[1]))
