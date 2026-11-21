"""Build a library of invented hand poses; no input files or external assets."""
import os, sys
import numpy as np

# Grayscale and edge measurements shared by the procedural renderer and recognizer.
common = '''import numpy as np

def surround(a):
    p = np.pad(a, ((0,0),(1,1),(1,1)), mode='edge')
    h = p[:,:,:-2] + 2*p[:,:,1:-1] + p[:,:,2:]
    return (h[:,:-2] + 2*h[:,1:-1] + h[:,2:]) * .0625

def evidence(im):
    # How deep is each pixel inside an island of fine-scale detail? A calm
    # background remains connected to the frame; even a smooth palm has an edge.
    sm = surround(im)
    dx = np.pad(sm, ((0,0),(0,0),(1,1)),mode='edge')
    dy = np.pad(sm, ((0,0),(1,1),(0,0)),mode='edge')
    dx = dx[:,:,2:] - dx[:,:,:-2]
    dy = dy[:,2:] - dy[:,:-2]
    mag = np.sqrt(dx*dx+dy*dy)
    strong = np.maximum(.014, mag.max((1,2),keepdims=True)*.36)
    c = .075*np.minimum(1.3,mag/strong)**1.5
    # Horizontal and vertical frame scans give an estimate of the outer envelope.
    a = np.cumsum(c,axis=2); b = np.cumsum(c[:,:,::-1],axis=2)[:,:,::-1]
    t = np.cumsum(c,axis=1)
    env = np.minimum(np.minimum(a/.105,b/.105),t/.095)
    # Also allow reaching the frame on a curved path (soft flood fill).
    cost = .002 + .65*(np.minimum(1.3,mag/strong)**1.8)
    cost[:,0,:] *= .5
    d = np.zeros_like(c)+2
    d[:,0] = cost[:,0]; d[:,:,0]=cost[:,:,0]; d[:,:,-1]=cost[:,:,-1]
    for j in range(18):
        d[:,1:] = np.minimum(d[:,1:], d[:,:-1]+cost[:,1:]); d[:,:-1] = np.minimum(d[:,:-1],d[:,1:]+cost[:,:-1]); d[:,:,1:] = np.minimum(d[:,:,1:],d[:,:,:-1]+cost[:,:,1:]); d[:,:,:-1] = np.minimum(d[:,:,:-1],d[:,:,1:]+cost[:,:,:-1])
    hull=np.minimum(1,env)*.26 + .74*np.minimum(1,d/.59)
    # A companion cue for holes and narrow finger gaps. A candidate stretch
    # of background should resemble a nearby parallel stretch reaching the
    # image border, in both brightness and smoothness. Work per scan line so
    # that a gradual vertical lighting gradient doesn't create a fake hand.
    left=np.median(sm[:,:,:3],axis=2,keepdims=True); right=np.median(sm[:,:,-3:],axis=2,keepdims=True)
    u=(np.arange(28,dtype=np.float32)/27)[None,None,:]
    bg=(1-u)*left+u*right
    detail=np.minimum(1,mag/(.5*strong+.001)); dev=np.abs(sm-bg)
    fl=1-np.exp(-2.7*(dev/(.042+.55*strong))**2-1.5*detail**2)
    agree=np.exp(-np.abs(left-right)/(.062+.32*strong))
    return hull*(1-.47*agree)+fl*.47*agree

def edgechan(im):
    sm = .45*im+.55*surround(im)
    xp=np.pad(sm,((0,0),(0,0),(1,1)),mode='edge'); yp=np.pad(sm,((0,0),(1,1),(0,0)),mode='edge')
    dx=xp[:,:,2:]-xp[:,:,:-2]; dy=yp[:,2:]-yp[:,:-2]
    norm=np.sqrt(dx*dx+dy*dy)+.0001
    amp=np.minimum(1.25,(norm/(.025+np.mean(norm,(1,2),keepdims=True)*1.4))**.75)
    out=[]
    for i in range(4):
        v=(dx*np.cos(i*np.pi/4)+dy*np.sin(i*np.pi/4))/norm
        out.append((v**4*amp).astype(np.float32))
    return out

def pooled(a,n):
    # Triangular interpolation between spatial histogram cells.
    f=np.maximum(0, 1-abs((np.arange(28,dtype=np.float32)[None,:]+.5)*n/28-(np.arange(n,dtype=np.float32)[:,None]+.5)))
    f=f/f.sum(1,keepdims=True)
    return (f @ a @ f.T).reshape(len(a),-1)

def unit(a,center):
    a = a-center*a.mean(1,keepdims=True)
    return a / np.sqrt(.00001+np.sum(a*a,1,keepdims=True))

def features(im,mask):
    n=len(im); seg=unit(pooled(mask,10),1)
    weight=np.array([.82,1,1,1,1,1,1,1,.93,.79,.64,.49],np.float32)[None,:,None]
    edges=[pooled(e,12).reshape(n,12,12)*weight for e in edgechan(im)]
    edges=np.stack(edges,3)
    # Equalize inter-finger valleys and the outline, which otherwise
    # dominates all inner edges. Include a separate, signed crease channel.
    edges = edges/(.11+np.sum(edges**2,axis=3,keepdims=True))**.29
    a=surround(im); b=surround(surround(surround(a))); crease=(a-b)*(.2+.8*mask)
    crease=np.tanh(crease/(.012+np.mean(np.abs(crease),(1,2),keepdims=True)))
    detail=pooled(crease,13)
    # An elongated narrow ridge has similar intensities along its length and
    # contrasts with one or both of its sides. Preserve its direction and
    # location even when a more prominent edge is a wall or a sleeve.
    ridges=[];pad=np.pad(a,((0,0),(3,3),(3,3)),mode='edge')
    for v,h in ((2,0),(2,2),(0,2),(2,-2)):
        plus=pad[:,3+v:31+v,3+h:31+h];minus=pad[:,3-v:31-v,3-h:31-h]
        nex=pad[:,3+h:31+h,3-v:31-v];prev=pad[:,3-h:31-h,3+v:31+v]
        thin=np.abs(2*a-nex-prev)-.55*np.abs(2*a-plus-minus)
        thin=np.clip(thin/(.045+np.mean(np.abs(thin),(1,2),keepdims=True)),0,2.5)
        ridges.append(pooled(thin,9))
    ridge=unit(np.stack(ridges,2).reshape(n,-1),.77)
    return np.concatenate([.61*seg,.76*unit(edges.reshape(n,-1),.68),.32*unit(detail,1),.355*ridge],1).astype(np.float32)
'''

runtime = '''
def cleanbox(mask):
    # Crop by robust one-sided extents of a connected hand, ignoring loose marks.
    n=len(mask)
    p=np.maximum(0,mask-.42)**1.25
    y,x=np.indices((28,28),dtype=np.float32)
    # Some photos have wallpaper or furniture touching the border. Reduce its
    # influence, but never assume the hand is restricted to a fixed window.
    central=.25+.75*np.exp(-((x-13.5)/11)**4)
    p *= central
    sm=surround(p)
    main=np.maximum(0,sm-.085)
    w=main.sum((1,2),keepdims=True)
    cx=(main*x).sum((1,2),keepdims=True)/(w+.0001)
    focus=np.exp(-.5*((x-cx)/8)**4)
    p=main*(.18+.82*focus)
    # Locate the top and the two sides above the end of the forearm.
    def bound(profile,q):
        s=profile.cumsum(1); return np.argmax(s>=s[:,-1:] * q,axis=1)
    v=p.sum(2); top=bound(v,.014)-1.
    last=bound(v,.995)+1.
    h=np.maximum(9,last-top)
    upper=(y[None,:,:]-top[:,None,None])/h[:,None,None]
    hp=(p*np.clip((1.06-upper)*3,.18,1)).sum(1)
    left=bound(hp,.018)-1.; right=bound(hp,.982)+1.
    mid=(left+right)*.5; width=right-left
    return mid.astype(np.float32),top.astype(np.float32),h.astype(np.float32),width.astype(np.float32)

def sample(a,x,y):
    x=np.clip(x,0,27); y=np.clip(y,0,27)
    ix=x.astype(int); iy=y.astype(int); ix1=np.minimum(27,ix+1); iy1=np.minimum(27,iy+1)
    b=np.arange(len(a))[:,None,None]; fx=x-ix; fy=y-iy
    return ((a[b,iy,ix]*(1-fx)+a[b,iy,ix1]*fx)*(1-fy)+(a[b,iy1,ix]*(1-fx)+a[b,iy1,ix1]*fx)*fy).astype(np.float32)

def contour(ext):
    # Fourier magnitudes of soft row extents are blind to vertical displacement.
    # Retain relative left/right phase via center and width as separate signals.
    n=len(ext); m=ext-ext.min((1,2),keepdims=True); m/=np.maximum(.0001,m.max((1,2),keepdims=True))
    pos=np.arange(m.shape[2],dtype=np.float32)[None,None,:]
    def bound(a,coord):
        p=np.maximum(0,a-.28)**1.1
        den=p.sum(2);ok=np.minimum(1,den/.82)
        c=p.cumsum(2); left=np.argmax(c>=den[:,:,None]*.058,axis=2)
        right=np.argmax(c>=den[:,:,None]*.942,axis=2)
        width=(right-left+.7)*ok;center=np.sum(p*coord,axis=2)/(den+.0001)
        mid=(center*ok).sum(1,keepdims=True)/(ok.sum(1,keepdims=True)+.001)
        return width,(center-mid)*ok
    w,b=bound(m,pos);h,a=bound(m.transpose(0,2,1),pos)
    def spectra(w):
        v=np.abs(np.fft.rfft(w,axis=1)[:,1:9]);return unit(np.sqrt(v+.001),0)
    return np.concatenate([spectra(w),spectra(b),spectra(h),spectra(a)],1).astype(np.float32)*.5

outlines=[contour(m[:,:100].reshape(len(m),10,10)) for m in CONSTANTS['models']]

def scores(im):
    seg=evidence(im); mid,top,h,width=cleanbox(seg)
    y,x=np.indices((28,28),dtype=np.float32)
    poses=[]; shapes=[]
    # Canonical square, parameterized primarily by the top of the hand and
    # the width of its finger span. Include a conservative, less cropped view.
    for tight,shift,zoom,above in ((1.,0.,1.,.045),(.0,0.,1.,0.),(.9,-.055,.85,.04),(.9,.055,.85,.04),(.75,0.,.77,.025),(.8,0.,1.13,.095)):
        side=np.maximum(np.maximum(h,width*1.08),15.)
        side=(side*tight+27*(1-tight))*zoom
        cx=mid*tight+13.5*(1-tight)+side*shift
        ty=top*tight-side*above
        xx=cx[:,None,None]+(x-13.5)*side[:,None,None]/27
        yy=ty[:,None,None]+y*side[:,None,None]/27
        vec=features(sample(im,xx,yy),sample(seg,xx,yy))
        poses.append(vec); shapes.append(contour(vec[:,:100].reshape(len(im),10,10)))
    inp=np.concatenate(poses); descriptors=np.concatenate(shapes)
    out=[]
    for i in range(24):
        m=CONSTANTS['models'][i]
        # Each comparison is a direct dot product with a mathematical pose,
        # never with a training photograph. Keep multiple supporting poses.
        sim=inp @ m.T + .21*(descriptors @ outlines[i].T)
        vals=np.partition(sim,sim.shape[1]-6,axis=1)[:,-6:]
        val=.6*vals.max(1)+.4*vals.mean(1)
        out.append(val.reshape(len(poses),len(im)).max(0))
    return np.stack(out,1)

def predict(x):
    ans=np.empty(len(x),np.int64)
    for i in range(0,len(x),32):
        block=x[i:i+32,0]
        # Fix the matrix multiplication dimensions even for the final block.
        a=np.zeros((32,28,28),np.float32); a[:len(block)]=block
        ans[i:i+32]=scores(a)[:len(block)].argmax(1)
    return ans
'''

# Synthetic hand coordinate system in palm widths: +y towards the fingertips,
# +x towards the thumb, +z towards a viewer facing the palm.
rng = np.random.default_rng(271828)
letters = 'ABCDEFGHIKLMNOPQRSTUVWXY'

def uni(a,b):return rng.uniform(a,b)
def rot(axis,deg):
    c=np.cos(np.deg2rad(deg));s=np.sin(np.deg2rad(deg))
    if axis==0:return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis==1:return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def anatomy(letter):
    L=letter
    fat=uni(.92,1.09); span=uni(.92,1.07); digit=uni(.92,1.09)
    # poses are MCP flexion, PIP flexion, DIP flexion, and lateral abduction
    closed=(76,91,51,0); straight=(2,4,2,0); bent=(64,88,46,0)
    state=[closed]*4
    thumb='cross'; yaw=uni(-15,24); pitch=uni(-10,13); roll=uni(-11,11)
    if L=='A':thumb='alongside';yaw+=uni(-9,25)
    # T also has a lateral handshape in which the middle finger traverses the
    # raised thumb; its long, straight segment is important in profile.
    elif L=='T' and uni(0,1)<.9:
        state[0]=(uni(3,23),uni(4,40),uni(2,26),6); thumb='letterT'
        yaw=uni(-123,-66);pitch=uni(-16,26);roll=uni(-85,-72)
    elif L=='B':state=[straight]*4;thumb='transverse'
    elif L in 'CO':
        meet=L=='O'; angle=uni(23,42) if not meet else uni(18,40)
        state=[(angle+1.5*j,uni(26,47) if not meet else uni(42,61),uni(18,36),0) for j in range(4)]
        thumb='circular' if meet else 'openArc'; yaw=uni(73,111);pitch+=uni(-7,11);roll+=uni(-21,4)
    elif L=='D':
        state=[straight,(58,60,44,-3),(62,65,45,0),(68,68,47,0)];thumb='middleTip';yaw+=25
    elif L=='E':state=[(uni(16,42),uni(89,112),uni(46,72),0)]*4;thumb='under';yaw+=uni(-15,12);pitch+=uni(-30,-7)
    elif L=='F':
        state=[(38,67,45,6),(0,4,4,8),(0,3,3,-5),(0,3,3,-23)];thumb='indexTip';yaw=uni(38,75);roll+=uni(-19,0)
    elif L in 'GH':
        state[0]=straight;state[1]=(4,6,1,-2) if L=='H' else closed
        thumb='parallel' if L=='G' else 'cross'
        yaw=uni(-176,-98) if L=='H' else uni(-100,-54);pitch=uni(-15,21);roll=uni(-94,-67)
    elif L=='I':state[3]=(0,4,2,-8);thumb='cross'
    elif L in 'KP':
        state[0]=(-3,4,4,uni(3,12));state[1]=(uni(31,67),uni(0,13),4,uni(-16,-6))
        thumb='between';yaw=uni(-64,-17) if L=='K' else uni(100,164)
        if L=='P':roll=uni(53,99);pitch=uni(6,43)
    elif L=='L':state[0]=straight;thumb='spread';yaw=uni(-15,15)
    elif L in 'MN':
        num=3 if L=='M' else 2
        for j in range(num): state[j]=(uni(26,58),uni(66,87),uni(35,58),0)
        thumb='tuck'+str(num);yaw=uni(8,43);roll+=uni(-12,3)
    elif L=='Q':
        state[0]=(uni(0,22),uni(7,25),4,6);thumb='pinchDown';yaw=uni(100,175);roll=uni(151,210);pitch=uni(18,60)
    elif L=='R':state[0]=(5,3,1,-10);state[1]=(-3,5,9,12);thumb='cross';yaw+=12
    elif L=='S':thumb='acrossKnuckles';yaw+=uni(-19,21);pitch+=uni(-24,6)
    elif L=='T':thumb='tuck1';state[0]=(40,83,48,8)
    elif L=='U':state[0]=straight;state[1]=(1,4,2,1);thumb='cross'
    elif L=='V':state[0]=(2,3,2,uni(10,22));state[1]=(0,4,2,uni(-18,-10));thumb='cross'
    elif L=='W':state[:3]=[(0,4,2,uni(12,23)),(0,3,2,0),(1,4,1,uni(-23,-12))];thumb='little'
    elif L=='X':state[0]=(uni(-3,18),uni(69,96),uni(42,78),0);thumb='cross';yaw=uni(32,67)
    elif L=='Y':state[3]=(1,2,3,uni(-32,-15));thumb='spread'
    paths=[];ells=[]
    def ball(c,r,tilt=0):ells.append((np.array(c),np.array(r),rot(2,tilt)))
    def line(points,rads):paths.append((np.asarray(points),np.asarray(rads)*fat))
    ball((0,-.06,0),(.505*span,.635,.195*fat),-3)
    ball((.225,-.24,.082),(.252,.43,.131*fat),-18)
    ball((-.245,-.13,.054),(.218,.452,.123*fat),11)
    tips=[]
    for j,(a,b,c,s) in enumerate(state):
        # Relative lengths and radii of the phalanges, index to little finger.
        x=[.34,.113,-.125,-.34][j]*span
        root=np.array([x,[.34,.405,.329,.198][j],.002])
        ln=np.array([.43,.285,.222])*[1,1.09,1.015,.8][j]*digit*uni(.97,1.03)
        rad=[.114,.117,.111,.099][j]; points=[root]
        bend=np.cumsum(np.array([a,b,c])+rng.normal(0,[4,4,3])); s+=uni(-3,3)
        for k in range(3):
            ang=np.deg2rad(bend[k]); sp=np.deg2rad(s)
            vec=np.array([np.sin(sp)*np.cos(ang),np.cos(sp)*np.cos(ang),np.sin(ang)])
            points.append(points[-1]+ln[k]*vec)
        line(points,rad*np.array([1.05,1,.89,.785]));tips.append(points[-1]);
        # Rounded heel of each finger in the palm.
        ball((x,.234-(j==3)*.10,-.008),(.145,.25,.16),s*.3)
    root=np.array([.265,-.35,.06]); base=np.array([.48,-.08,.16]); joint=np.array([.38,.14,.38]); tip=np.array([.02,.18,.46])
    if thumb=='spread':base=[.59,-.11,.09];joint=[.83,.025,.15];tip=[1.09,.14,.18]
    elif thumb=='alongside':base=[.46,-.01,.15];joint=[.475,.32,.225];tip=[.45,.59,.22]
    elif thumb=='acrossKnuckles':base=[.46,.009,.225];joint=[.26,.30,.43];tip=[-.065,.365,.49]
    elif thumb=='transverse':base=[.408,-.10,.23];joint=[.19,.031,.29];tip=[-.15,.106,.26]
    elif thumb=='under':base=[.405,-.055,.24];joint=[.14,.115,.405];tip=[-.19,.143,.48]
    elif thumb in ('letterT','pinchDown'):
        if thumb=='letterT':base=[.46,-.032,.21];joint=[.48,.14,.44];tip=[.49,.34,.62]
        else:base=[.54,-.02,.21];joint=[.58,.34,.31];tip=[.60,.68,.41]
    elif thumb.startswith('tuck'):
        n=int(thumb[-1]);target=(.34-.226*(n-.48))*span; base=[.435,-.035,.21];joint=[.5*target+.23,.146,.40];tip=[target,.40+(n==1)*.12,.40]
    elif thumb in ('circular','openArc','middleTip','indexTip'):
        j=1 if thumb=='middleTip' else 0
        target=tips[j].copy();target[1]-=.045 if thumb!='openArc' else uni(.43,.68)
        target[2]-=.025
        base=np.array([.47,-.11,.22]); joint=base*.42+target*.58+[.055,-.145,.105];tip=target
    elif thumb=='parallel':base=[.50,-.04,.18];joint=[.55,.31,.30];tip=[.55,.65,.33]
    elif thumb=='between':base=[.43,-.08,.21];joint=[.28,.23,.36];tip=[.19,.53,.45]
    elif thumb=='little':base=[.42,-.085,.198];joint=[.125,.095,.31];tip=[-.23,.20,.34]
    line([root,base,np.array(joint)+rng.normal(0,.025,3),np.array(tip)+rng.normal(0,.025,3)],[.169,.155,.122,.095])
    view=rot(2,roll)@rot(0,pitch)@rot(1,yaw)
    return paths,ells,view

def motif(L):
    # A second, view-centered construction. These are drawings in local palm
    # widths: joint positions are specified relative to one another, never
    # relative to image pixels. The ray tracer is shared with the 3-D model.
    paths=[];ells=[]
    def chain(p,r): paths.append((np.array(p),np.array(r)))
    def oval(x,y,z,a,b,c,ang):ells.append((np.array([x,y,z]),np.array([a,b,c]),rot(2,ang)))
    def tube(p,r,zs):chain([(*xy,z) for xy,z in zip(p,zs)],r)
    def stem(a,b,c,r=.11):tube([a,b,c],[r*1.12,r,r*.82],[.03,.16,.22])
    def digit(a,b,c,d,r=.11,z=.35):tube([a,b,c,d],[r*1.1,r,r*.94,r*.83],[.06,z,z+.06,z+.095])
    def fan(x,y,length,angle,r):
        q=np.deg2rad(angle);v=np.array([np.sin(q),np.cos(q)])*length; o=np.array([x,y]);w=np.array([np.cos(q),-np.sin(q)])*uni(-.069,.069)
        tube([o,o+.46*v,o+.77*v+w*.45,o+v+w],[1.09*r,r,r*.88,r*.75],[.05,.10,.105,.12])
    roll=uni(-15,15); pitch=uni(-14,16);yaw=uni(-18,18)
    if L in 'GHT':
        # The distal direction is sideways but the forearm is downward.
        oval(0,0,0,.43,.52,.24,-24);oval(-.15,-.22,.04,.285,.39,.25,-25)
        # The last three fingers turn over the end of the palm.
        for j in range(3 if L!='H' else 2):
            y=.21-j*.20
            digit((-.24,y-.11),(.20,y+.06),(.35,y-.035),(.155,y-.205),.119-j*.006,.255)
        stem((-.20,.27),(.39,.37),(.85,.36),.125)
        if L=='H':stem((-.16,.12),(.44,.18),(uni(.76,.96),.19),.123)
        if L=='G':
            # Thumb partly hidden behind the pointing index.
            tube([(-.30,-.036),(-.26,.32),(.13,.424),(.49,.41)],[.15,.13,.114,.094],[-.1,-.12,-.11,-.09]);
        elif L=='T':stem((-.09,.20),(-.11,.47),(-.12,.70),.125)
        else:digit((-.25,-.30),(-.27,.01),(-.04,.115),(.17,.10),.13,.32)
        roll=uni(-6,19)
    elif L in 'PQ':
        if L=='P':
            oval(.09,-.05,.01,.46,.37,.24,-23);oval(.24,-.21,.02,.30,.39,.22,25)
            digit((.29,-.26),(.01,-.27),(-.16,-.22),(.035,-.19),.134,.19)
            digit((.24,-.135),(-.115,-.12),(-.26,-.12),(-.08,-.065),.124,.24)
            stem((.06,.11),(-.40,.34),(-.91,.48),.118)
            digit((.017,.09),(-.19,-.039),(-.345,-.23),(-.43,-.39),.12,.34)
            stem((.25,-.047),(-.02,.035),(-.23,.005),.132)
            roll=uni(-17,23)
        else:
            oval(.04,.12,.03,.428,.48,.22,13)
            for j in range(3):
                x=-.33+j*.203;digit((x,.29),(x-.065,-.01),(x-.12,-.19),(x-.18,-.27),.115,.16)
            digit((.24,.185),(.34,-.20),(.35,-.50),(.37,-.72),.125,.21)
            digit((-.02,.34),(-.20,.04),(-.33,-.28),(-.36,-.48),.14,.28)
            roll=uni(-22,16)
    elif L in 'CO':
        oval(-.07,-.06,0,.31,.59,.195,-20)
        for j in range(4):
            ii=3-j;dx=-ii*.042;dy=-ii*.045
            x0,y0=(-.055+dx,.23+dy)
            if L=='C':c=(.578+dx,.73+dy);d=(.71+dx,.624+dy)
            else:c=(.55+dx,.64+dy);d=(.64+dx,.437+dy)
            tube([(x0,y0),(.16+dx,.63+dy),(.365+dx,.75+dy),c,d],[.15,.135,.125,.108,.088],[.105-j*.028,.13-j*.02,.14-j*.02,.15-j*.019,.155-j*.019])
        tip=(.699,uni(.20,.34)) if L=='C' else (.644,.44)
        digit((-.07,-.325),(.23,-.21),(.49,-.049),tip,.144,.275)
        roll=uni(-18,17);yaw=uni(-23,29)
    else:
        oval(0,-.077,0,.474,.58,.197,0);oval(.226,-.21,.067,.237,.373,.132,-14)
        across=[.328,.108,-.117,-.321];yp=[.302,.351,.295,.18];radii=[.117,.121,.113,.099];lengths=[.93,1.025,.965,.774]
        ext={};arc={}; short={}
        if L=='B':ext={0:0,1:0,2:0,3:0}
        if L in 'DL':ext={0:uni(-5,6)}
        if L=='K':ext={0:uni(-2,10),1:uni(-39,-15)};short[1]=uni(.40,.73)
        if L=='F':ext={1:uni(9,27),2:uni(-6,4),3:uni(-29,-10)}
        if L in 'IY':ext={3:uni(-29,-6) if L=='Y' else uni(-13,1)}
        if L=='V':ext={0:uni(10,20),1:uni(-21,-9)}
        if L=='W':ext={0:uni(11,21),1:uni(-4,4),2:uni(-27,-12)}
        if L=='U':ext={0:uni(-3,3),1:uni(-2,3)}
        if L=='R':ext={0:uni(-16,-9),1:uni(9,18)}
        if L=='D':arc={1,2,3}
        if L=='F':arc={0}
        if L=='X':arc={0}
        tips=[]
        for j,(x,y,r) in reversed(list(enumerate(zip(across,yp,radii)))):
            if j in ext:
                if L in 'DKRU': x*=uni(.55,.86)
                fan(x,y,lengths[j]*short.get(j,1)*uni(.89,1.115),ext[j],r)
            elif j in arc:
                if L=='F':
                    end=(x+.39,y+.34);digit((x,y),(x+.253,y+.51),(x+.423,y+.44),end,r,.275);tips.append(end)
                elif L=='X':
                    end=(x+.355,y+.408);digit((x,y),(x+.082,y+.620),(x+.32,y+.585),end,r,.30)
                else:
                    end=(x+.13,y+.08); digit((x,y),(x+.033,y+.283),(x+.175,y+.21),end,r,.29);tips.insert(0,end)
            else:
                # Exposed length and direction of the folded distal phalanges:
                # fingernails face out in E; in M/N fingers drape over a thumb.
                high=.14;bottom=-.17;dx=.017
                if L=='E':high=.284;bottom=.06;dx=-.026
                if L in 'MN' and j<(3 if L=='M' else 2):high=.278;bottom=-.253;dx=-.067
                if L=='A':high=.17;bottom=-.20;dx=-.033
                digit((x,y),(x+dx*.1,y+high),(x+dx*.6,y+high-.121),(x+dx,y+bottom),r,.32+uni(-.04,.04))
        base=(.425,-.18);joint=(.30,.065);end=(.025,.091)
        if L in 'LY':
            fan(.32,-.18,uni(.78,.94),uni(47,72) if L=='Y' else uni(61,88),.144)
        else:
            if L=='A':joint=(.495,.19);end=(.485,.51)
            if L=='B':joint=(.20,.011);end=(-.164,.135)
            if L=='D':joint=(.38,.145);end=tips[0]
            if L=='F':base=(.42,-.19);joint=(.596,.045);end=tips[0]
            if L=='E':joint=(.19,.046);end=(-.183,.068)
            if L=='K':joint=(.25,.15);end=(.11,.54)
            if L=='S':joint=(.266,.255);end=(-.049,.30)
            if L=='M':joint=(.12,-.066);end=(-.238,.007)
            if L=='N':joint=(.18,-.035);end=(-.076,.080)
            if L=='W':joint=(.19,.035);end=(-.212,.145)
            z=.44 if L not in 'MN' else .24
            digit((.265,-.35),base,joint,end,.137,z)
        if L in 'MNDXR': roll+=uni(-16,3)
        if L=='F': roll+=uni(-17,-1)
    view=rot(2,roll)@rot(0,pitch)@rot(1,yaw)
    return paths,ells,view

def render(L,drawing=False):
    paths,ells,view=motif(L) if drawing else anatomy(L)
    def proj(a):return np.asarray(a)@view.T
    paths=[(proj(p),r) for p,r in paths];ells=[(proj(p),r,view@o) for p,r,o in ells]
    pts=np.concatenate([np.concatenate([p[:,:2]-r[:,None],p[:,:2]+r[:,None]]) for p,r in paths]+[np.stack([p[:2]-np.sqrt(((o*r)**2).sum(1))[:2],p[:2]+np.sqrt(((o*r)**2).sum(1))[:2]]) for p,r,o in ells]);mn=pts.min(0);mx=pts.max(0)
    # Attach a forearm at an angle appropriate to the wrist, not at the
    # finger axis of a hand that is flexed down at the wrist (P/Q).
    wrist=proj([0,-.52,-.045]); behind=proj([0,-.68,-.065]);
    if drawing and L in 'GHT':wrist=proj([-.19,-.33,-.03])
    if drawing and L=='Q':wrist=proj([-.08,.34,-.19])
    if drawing and L=='P':wrist=proj([.28,-.21,-.08])
    if L in 'GHT':direction=np.array([uni(-1,-.42),uni(-1,-.62),.02]);
    elif L=='P':direction=np.array([uni(.51,1),uni(-1,-.25),-.22]);
    elif L=='Q':direction=np.array([uni(-.92,.16),uni(-1,-.09),-1.]);
    else:direction=behind-wrist+np.array([uni(-.058,.058),-.03,uni(-.038,.038)])
    direction=direction/np.linalg.norm(direction);end=wrist+direction*5
    # The crop is based on the hand, not the full length of the arm.
    tall=mx[1]-mn[1];wide=mx[0]-mn[0];side=max(tall*uni(1.04,1.27),wide*uni(1.04,1.23));side*=uni(.94,1.10)
    # Loose photographic cropping leaves some arm visible. For a horizontal
    # hand the spare room in the square is shared above and below the hand.
    side*=uni(1.,1.30)
    top=mx[1]+side*uni(-.03,.09)+max(0,side-tall)*uni(.02,.46)
    cx=(mn[0]+mx[0])*.5+side*uni(-.11,.11)
    grid=36;u=(np.arange(grid)+.5)/grid;xx=(cx+(u[None,:]-.5)*side).astype(np.float32); yy=(top-u[:,None]*side).astype(np.float32)
    inf=-100.;z=np.full((grid,grid),inf,dtype=np.float32); shade=np.zeros_like(z)
    light=np.array([uni(-.9,.1),uni(.35,.99),uni(.28,.89)]);light/=np.linalg.norm(light)
    def paint(d,nx,ny,nz,keep):
        visible=keep&(d>z); v=nx*light[0]+ny*light[1]+nz*light[2];shade[visible]=v[visible];z[visible]=d[visible]
    def tube(p,r):
        for j in range(len(p)-1):
            start=p[j];v=p[j+1]-start
            rx=xx-start[0];ry=yy-start[1];len2=v[0]**2+v[1]**2+1.e-9
            center=(rx*v[0]+ry*v[1])/len2; t=np.clip(center,0,1); radius=r[j]+(r[j+1]-r[j])*t
            perp=(rx*v[1]-ry*v[0])**2/len2
            # Frontmost cross section of the swept sphere, taking z slope into account.
            t=np.clip(center+v[2]*np.sqrt(np.maximum(0,radius**2-perp)/(len2*(len2+v[2]**2))),0,1)
            nx=rx-t*v[0];ny=ry-t*v[1];radius=r[j]+(r[j+1]-r[j])*t;nz2=radius**2-nx**2-ny**2;nz=np.sqrt(np.maximum(0,nz2));d=start[2]+t*v[2]+nz
            paint(d,nx/radius,ny/radius,nz/radius,nz2>=0)
    tube(np.stack([wrist,end]),np.array([.265,.425])*uni(.93,1.08))
    for c,r,o in ells:
        q=(o*(1/r**2))@o.T;dx=xx-c[0];dy=yy-c[1];b=q[2,0]*dx+q[2,1]*dy;cc=q[0,0]*dx**2+2*q[0,1]*dx*dy+q[1,1]*dy**2-1;disc=b*b-q[2,2]*cc
        dz=(-b+np.sqrt(np.maximum(0,disc)))/q[2,2]; nx=q[0,0]*dx+q[0,1]*dy+q[0,2]*dz;ny=q[1,0]*dx+q[1,1]*dy+q[1,2]*dz;nz=b+q[2,2]*dz;nrm=np.sqrt(nx**2+ny**2+nz**2)+1.e-7;paint(c[2]+dz,nx/nrm,ny/nrm,nz/nrm,disc>=0)
    for p,r in paths:tube(p,r)
    mask=z>inf
    # A little contact shadow where a nearer finger occludes the palm behind it.
    zz=np.where(mask,z,0); zp=np.pad(zz,1,mode='edge');mp=np.pad(mask,1,mode='edge');occ=0
    for j,k in [(0,1),(1,0),(2,1),(1,2)]:occ+=np.clip((zp[j:j+grid,k:k+grid]-zz-.035)*4,0,1)*mp[j:j+grid,k:k+grid]
    # Fingers cast soft shadows on more distant surfaces, especially in the
    # recess between a bent finger and the thumb. Trace towards the light.
    longocc=np.zeros_like(occ); zp=np.pad(zz,8,mode='edge');mp=np.pad(mask,8,mode='edge')
    length=max(.4,np.hypot(light[0],light[1]));
    for distance in (2,3,5,7):
        j=8-int(round(distance*light[1]/length));k=8+int(round(distance*light[0]/length));
        front=zp[j:j+grid,k:k+grid]-zz-distance*side/grid*light[2]/length
        longocc=np.maximum(longocc,np.clip(front*3.5,0,1)*mp[j:j+grid,k:k+grid])
    skin=uni(.42,.62)+shade*uni(.19,.35)-.074*occ-uni(.13,.28)*longocc
    bg=uni(.44,.88);im=np.where(mask,skin,bg);return im.astype(np.float32),mask.astype(np.float32)

def resize(a):
    p=(np.arange(28)+.5)*a.shape[-1]/28-.5;u=p.astype(int);v=np.minimum(u+1,a.shape[-1]-1);f=(p-u).astype(np.float32);h=a[:,:,u]*(1-f)+a[:,:,v]*f;return h[:,u,:]*(1-f[:,None])+h[:,v,:]*f[:,None]

def build(folder):
    os.makedirs(folder,exist_ok=True)
    with open(os.path.join(folder,'solution.py'),'w') as f:f.write(common+runtime)
    ns={'CONSTANTS':{'models':()}};exec(common+runtime,ns)
    models=[]
    for L in letters:
        bank=[]
        for j in range(60):
            images,masks=zip(*(render(L,drawing=(j%3!=0)) for k in range(48)))
            images=resize(np.stack(images));masks=resize(np.stack(masks))
            masks=.47*ns['evidence'](images)+.53*ns['surround'](masks)
            if j%4!=0:
                mid,top,h,width=ns['cleanbox'](masks);y,x=np.indices((28,28),dtype=np.float32)
                side=np.maximum(np.maximum(h,width*1.08),15.)
                side=side*rng.uniform(.89,1.10,48);mid=mid+side*rng.uniform(-.039,.039,48)
                xx=mid[:,None,None]+(x-13.5)*side[:,None,None]/27
                yy=(top-side*rng.uniform(.012,.086,48))[:,None,None]+y*side[:,None,None]/27
                images=ns['sample'](images,xx,yy);masks=ns['sample'](masks,xx,yy)
            bank.append(ns['features'](images,masks))
        models.append(np.concatenate(bank));print('built',L,flush=True)
    np.savez(os.path.join(folder,'constants.npz'),models=np.stack(models))

if __name__=='__main__':build(sys.argv[1])
