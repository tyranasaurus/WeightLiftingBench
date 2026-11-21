"""Build one template per independent rendering of an articulated hand.

All geometry is in a palm-local rest frame (length unit: palm width).
No part of this build uses example photos or example descriptors.
"""
import sys
from pathlib import Path
import numpy as np

# Shared image descriptor. Positive/negative polarity is not reliable at the
# silhouette; use unoriented gradients and a separate texture-occupancy field.
FEATURES = '''import numpy as np


def smooth(a):
    l = np.concatenate((a[:, :, :1], a[:, :, :-1]), axis=2)
    r = np.concatenate((a[:, :, 1:], a[:, :, -1:]), axis=2)
    b = (l + 2*a + r)*0.25
    t = np.concatenate((b[:, :1], b[:, :-1]), axis=1)
    d = np.concatenate((b[:, 1:], b[:, -1:]), axis=1)
    return (t + 2*b + d)*0.25


def descriptor(a, occupancy):
    a = a.astype(np.float32)
    b = 0.5*(a + smooth(a))
    dx = np.concatenate((b[:,:,1:],b[:,:,-1:]),axis=2) - np.concatenate((b[:,:,:1],b[:,:,:-1]),axis=2)
    dy = np.concatenate((b[:,1:],b[:,-1:]),axis=1) - np.concatenate((b[:,:1],b[:,:-1]),axis=1)
    mag = np.sqrt(dx*dx+dy*dy+1e-12)
    noise = np.quantile(mag.reshape(len(a),-1),0.25,axis=1).astype(np.float32)[:,None,None]
    magc = np.maximum(mag-0.75*noise,0)
    energy = np.sqrt(magc)*np.minimum(magc/(noise+0.012),1.0)
    # Soft, unsigned bins with an angular width of one bin. Pool locally;
    # truncate local amplitudes to limit the effect of cast shadows.
    angle = np.arctan2(dy,dx)*np.float32(5/np.pi)
    bins = []
    broad = smooth(smooth(smooth(magc)))
    strength = np.minimum(2.5,magc / (0.04 + noise + broad))
    for k in range(5):
        d = np.abs((angle-k+2.5)%5-2.5)
        field = np.maximum(1-d,0)*np.sqrt(energy*strength)
        bins.append(smooth(field))
    occupancy = smooth(occupancy.astype(np.float32))
    # Signed skin relief: compare a point with neighbouring skin, not with
    # the background. Cast shadows between fingers tend to be dark whatever
    # the polarity of the silhouette against the wall.
    acc=b*occupancy; weight=occupancy
    for step in range(7):
        acc=smooth(acc); weight=smooth(weight)
    tone=(b-acc/(weight+1e-6))*occupancy
    # Enclosed negative space. On each ray ask if skin is present before a
    # finger-length distance; a hole has skin in even its weakest direction.
    h,w=a.shape[1:]
    pad=np.pad(occupancy,((0,0),(8,8),(8,8)),mode='constant')
    rays=[]
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
        cover=np.zeros_like(occupancy)
        for r in (1,2,3,4,6,8):
            rr=int(0.72*r+0.5) if dy*dx else r
            oy=8+dy*rr; ox=8+dx*rr
            cover=np.maximum(cover,pad[:,oy:oy+h,ox:ox+w])
        rays.append(cover)
    enclosure=np.min(np.stack(rays),axis=0)
    hole=np.maximum(enclosure-occupancy,0)
    return np.concatenate((np.stack(bins,axis=3),occupancy[:,:,:,None],contourfield(occupancy),tone[:,:,:,None],hole[:,:,:,None]),axis=3)


def contourfield(m):
    h = smooth(m)
    dx = np.concatenate((h[:,:,1:],h[:,:,-1:]),axis=2) - np.concatenate((h[:,:,:1],h[:,:,:-1]),axis=2)
    dy = np.concatenate((h[:,1:],h[:,-1:]),axis=1) - np.concatenate((h[:,:1],h[:,:-1]),axis=1)
    return np.stack((np.maximum(dx,0),np.maximum(-dx,0),np.maximum(dy,0),np.maximum(-dy,0)),axis=3)


def cellbank(n,g):
    step = n/g
    c = (np.arange(g)+0.5)*step-0.5
    w = np.maximum(1-np.abs(np.arange(n)[None,:]-c[:,None])/step,0)
    return (w/w.sum(axis=1,keepdims=True)).astype(np.float32)


def pooldesc(field,g):
    n,h,w,c = field.shape
    bank = cellbank(h,g)
    temp = (bank @ field.reshape(n,h,w*c)).reshape(n,g,w,c).transpose(0,1,3,2)
    v = (temp @ bank.T).transpose(0,1,3,2)
    edge = v[:,:,:,:5]
    # Local HOG normalization with a floor relative to this image's energy.
    s = np.sum(edge*edge,axis=3,keepdims=True)
    floor = 0.50*s.mean(axis=(1,2),keepdims=True)+1e-8
    edge = edge/(s+floor)**0.33
    edge = edge.reshape(n,-1)
    edge = edge/np.sqrt(np.sum(edge*edge,axis=1,keepdims=True)+1e-9)
    occ = v[:,:,:,5].reshape(n,-1)
    occ = occ/np.sqrt(np.sum(occ*occ,axis=1,keepdims=True)+1e-9)
    contour=v[:,:,:,6:10].reshape(n,-1)
    contour=contour / np.sqrt(np.sum(contour*contour,axis=1,keepdims=True)+1e-9)
    # Concatenated groups with a fixed ratio of energy. Boundary normals
    # distinguish a gap between fingers from a peak of similar orientation.
    tone=v[:,:,:,10].reshape(n,-1)
    tone=tone / np.sqrt(np.sum(tone*tone,axis=1,keepdims=True)+.015*g*g*.015)
    hole=v[:,:,:,11].reshape(n,-1)
    # A little hole evidence at every one of many edge troughs is not as
    # compelling as a window with roughly its own inscribed disk.
    hole=np.maximum(hole-.08,0)
    hole=hole / np.sqrt(np.sum(hole*hole,axis=1,keepdims=True)+(.060*g)**2)
    return np.concatenate((edge*.72,occ*.45,contour*.43,tone*.235,hole*.25),axis=1)
'''

INFERENCE = '''

def foreground(a):
    # A border pixel lying in a smooth patch may explain other pixels as
    # background. Extrapolate its intensity using its local first derivatives.
    # Several depths into the border accommodate frames around a photo.
    b=smooth(a)
    dx=(np.roll(b,-1,axis=2)-np.roll(b,1,axis=2))*0.5
    dy=(np.roll(b,-1,axis=1)-np.roll(b,1,axis=1))*0.5
    dx[:,:,[0,-1]]=0; dy[:,[0,-1],:]=0
    # Certificate of local planarity: approximation error in a neighbourhood.
    tex=smooth(np.abs(a-b)+1.5*(np.abs(dx-smooth(smooth(dx)))+np.abs(dy-smooth(smooth(dy)))))
    dx=smooth(dx); dy=smooth(dy)
    v,u=np.mgrid[:28,:28].astype(np.float32)
    best=np.full_like(a,10)
    noise=np.quantile(tex.reshape(len(a),-1),.25,axis=1).astype(np.float32)[:,None,None]
    for c in (1,4,23,26):
        for t in range(2,27,4):
            for y,z in ((c,t),(t,c)):
                du=u-z; dv=v-y
                dist=np.sqrt(du*du+dv*dv)
                err=np.abs(b-b[:,y:y+1,z:z+1]-np.clip(dx[:,y:y+1,z:z+1],-.018,.018)*du-np.clip(dy[:,y:y+1,z:z+1],-.018,.018)*dv)
                # Smooth-field extrapolation becomes less certain further from
                # the seed. Cost of explaining texture as a plane is local.
                cost=err/(.025+noise+.0025*dist)+tex[:,y:y+1,z:z+1]/(.012+1.8*noise)+.027*dist
                best=np.minimum(best,cost)
    residual=(best-.9)*.58+(tex-noise)/(.036+1.7*noise)*.51
    m=np.clip(residual,0,1)
    # Diffuse confidence within patches of similar intensity, not across
    # strong boundaries. Per-image, no sharing amongst the batch.
    ref=b
    wt=[]
    for axis,shift in ((1,-1),(1,1),(2,-1),(2,1)):
        d=np.abs(ref-np.roll(ref,shift,axis=axis))
        w=np.exp(-d/(.030+noise))
        if axis==1: w[:,0 if shift==1 else -1]=0
        else: w[:,:,0 if shift==1 else -1]=0
        wt.append(w)
    init=m.copy(); denom=1.38+sum(wt)
    for it in range(8):
        m=(m+.38*init+wt[0]*np.roll(m,-1,axis=1)+wt[1]*np.roll(m,1,axis=1)+wt[2]*np.roll(m,-1,axis=2)+wt[3]*np.roll(m,1,axis=2))/denom
    # A low-texture region can run from the background into a hole in an
    # O or F, but it cannot run through several strong digit boundaries.
    # Compute a soft minimax path barrier to the image border. Smooth skin
    # enclosed by the contour should not be explained by an unrelated seed.
    barrier=np.full_like(b,1)
    barrier[:,0]=0;barrier[:,-1]=0;barrier[:,:,0]=0;barrier[:,:,-1]=0
    difx=np.abs(b[:,:,1:]-b[:,:,:-1]);dify=np.abs(b[:,1:]-b[:,:-1])
    for it in range(20):
        left=np.maximum(barrier[:,:,:-1],difx); right=np.maximum(barrier[:,:,1:],difx)
        up=np.maximum(barrier[:,:-1],dify); down=np.maximum(barrier[:,1:],dify)
        barrier[:,:,1:]=np.minimum(barrier[:,:,1:],left);barrier[:,:,:-1]=np.minimum(barrier[:,:,:-1],right)
        barrier[:,1:]=np.minimum(barrier[:,1:],up);barrier[:,:-1]=np.minimum(barrier[:,:-1],down)
    seal=np.clip((barrier-.005-noise*.5)/(.055+noise*1.5),0,1)
    return m+(1-m)*seal*.6*np.minimum(1,best*.8)


def warps(a, geometries):
    n,h,w=a.shape
    gy,gx=np.mgrid[:h,:w].astype(np.float32)
    stack=[]
    for sx,sy,dx,dy in geometries:
        u=(gx-(w-1)/2)*sx+(w-1)/2+dx
        v=(gy-(h-1)/2)*sy+(h-1)/2+dy
        u=np.clip(u,0,w-1); v=np.clip(v,0,h-1)
        j=u.astype(np.int32); i=v.astype(np.int32)
        fu=(u-j).astype(np.float32); fv=(v-i).astype(np.float32)
        jj=np.minimum(j+1,w-1); ii=np.minimum(i+1,h-1)
        stack.append((1-fv)*((1-fu)*a[:,i,j]+fu*a[:,i,jj])+fv*((1-fu)*a[:,ii,j]+fu*a[:,ii,jj]))
    return np.stack(stack,axis=1).reshape(-1,h,w)


def verify(template, image, outside):
    # Recompute normalized inner products using each candidate's own spatial
    # support. Edges well outside its silhouette can be background objects.
    # A soft support is important: an absent finger still counts against a
    # match, even though the model does not exactly align every joint.
    cells=144
    p=template[:,:,:5*cells].reshape(24,-1,cells,5)
    q=image[:,:,:5*cells].reshape(24,-1,cells,5)
    pm=template[:,:,5*cells:6*cells]; qm=image[:,:,5*cells:6*cells]
    m=pm/(np.max(pm,axis=2,keepdims=True)+1e-8)
    w=.22+.78*np.minimum(1,m*2.5)
    def corr(p,q,weight):
        num=(weight*np.einsum('abch,abch->abc',p,q)).sum(axis=2)
        aa=(weight*np.einsum('abch,abch->abc',p,p)).sum(axis=2)
        bb=(weight*np.einsum('abch,abch->abc',q,q)).sum(axis=2)
        return num/np.sqrt(aa*bb+1e-10)
    texture=corr(p,q,w)
    pp=template[:,:,6*cells:10*cells].reshape(24,-1,cells,4); qq=image[:,:,6*cells:10*cells].reshape(24,-1,cells,4)
    w=.4+.6*np.minimum(1,m*4)
    boundary=corr(pp,qq,w)
    # Foreground agreement, with less penalty for an explained background
    # patch at the edge of the camera frame than one in the hand interior.
    pm=pm-(pm*outside).sum(axis=2,keepdims=True)/outside.sum()
    qm=qm-(qm*outside).sum(axis=2,keepdims=True)/outside.sum()
    region=(pm*qm*outside).sum(axis=2)/np.sqrt((pm*pm*outside).sum(axis=2)*(qm*qm*outside).sum(axis=2)+1e-10)
    shadow=(template[:,:,10*cells:11*cells]*image[:,:,10*cells:11*cells]).sum(axis=2)/(.235**2)
    ht=template[:,:,11*cells:];hq=image[:,:,11*cells:]
    vacancy=1-np.sum((ht-hq)**2,axis=2)/(.25**2)
    return .44*texture+.21*boundary+.20*region+.085*shadow+.065*vacancy


coarse = CONSTANTS['coarse'].astype(np.float32) / 16384
fine = CONSTANTS['fine']


def predict(x):
    n=len(x)
    pred=np.empty(n,np.int64)
    if n==0:
        return pred
    translations=[(1,1,0,0),(1,1,2.4,0),(1,1,-2.4,0),(1,1,0,2.2),(1,1,0,-2.2),
                  (0.85,0.88,-1.5,-1.5),(0.85,0.88,1.5,1.5),(0.85,0.88,-1.5,1.5),(0.85,0.88,1.5,-1.5),
                  (1.16,1.14,0,-1),(1.16,1.14,0,1.5)]
    g=len(translations)
    count=len(coarse)//24
    row,col=np.mgrid[:12,:12].astype(np.float32)
    outside=(.4+.6*np.exp(-((row-5.5)**2+(col-5.5)**2)/42)).reshape(1,1,-1)
    for start in range(0,n,8):
        b=x[start:start+8,0]
        nb=len(b)
        # Fixed matrix dimensions, irrespective of how the caller split the batch.
        if nb<8:
            b=np.concatenate((b,np.repeat(b[-1:],8-nb,axis=0)),axis=0)
        m=foreground(b)
        f=descriptor(warps(b,translations),warps(m,translations))
        small=pooldesc(f,6)
        large=pooldesc(f,12).reshape(8,g,-1)
        match=(small @ coarse.T).reshape(8,g,24,count)
        for i in range(nb):
            # Shortlist pose/registration pairs separately for each letter.
            # Use the fine descriptor for verification before pooling poses.
            s=match[i].transpose(1,0,2).reshape(24,g*count)
            choices=np.argpartition(s,-18,axis=1)[:,-18:]
            pose=choices%count+np.arange(24)[:,None]*count
            query=choices//count
            tf=fine[pose].astype(np.float32)/16384
            high=verify(tf,large[i,query],outside)
            low=np.take_along_axis(s,choices,axis=1)
            both=0.20*low+0.80*high
            best=np.sort(both,axis=1)[:,-3:]
            answer=(2*best[:,2]+best[:,1]+best[:,0])*0.25
            pred[start+i]=np.argmax(answer)
    return pred
'''

# Build-time-only geometry.
from math import sin, cos

def deg(a):
    return np.deg2rad(a)

def v3(x,y,z):
    return np.array([x,y,z],dtype=float)

class Hand:
    def __init__(self,letter,rng):
        self.bones=[]
        self.letter=letter
        self.rng=rng
        self.plen=rng.uniform(0.95,1.21)
        self.variant=rng.uniform()
        self.pth=rng.uniform(.22,.28)
        self.wrist=v3(0,-self.plen*.5,0)
        self.mcp=[v3(x,self.plen*.5+offset,z) for x,offset,z in zip([.341,.111,-.122,-.338],[0,.065,.015,-.095],[.010,0,.014,.051])]
        self.frad=np.array([.116,.12,.112,.095])*rng.uniform(.95,1.09)
        self.flen=np.array([.96,1.04,.98,.79])*rng.uniform(.93,1.07)*rng.uniform(.97,1.03,4)
        self.thumbroot=v3(.25,-self.plen*.31,.08)
        self.tip=[]
        self.pip=[]
        self.dip=[]
        self.makepose()

    def segment(self,p,q,r0,r1,kind):
        self.bones.append((np.array(p),np.array(q),r0,r1,kind))

    def finger(self,i,az,flex):
        p=self.mcp[i]
        a=deg(az+self.rng.uniform(-2.3,2.3))
        r=self.frad[i]
        for j,(theta,frac,f0,f1) in enumerate(zip(flex,[.46,.31,.23],[1,.94,.86],[.95,.87,.80])):
            t=deg(theta + self.rng.uniform(-3.2,3.2))
            length=self.flen[i]*frac
            d=v3(sin(a)*cos(t),cos(a)*cos(t),sin(t))
            q=p+length*d
            self.segment(p,q,r*f0,r*f1,2+j)
            if j==0: self.pip.append(q)
            if j==1: self.dip.append(q)
            p=q
        self.tip.append(p)

    def thumb(self,points):
        p=self.thumbroot
        radii=[.193,.158,.132,.109]
        jitter=self.rng.uniform(-.024,.024,(3,3))
        for j,q in enumerate(points):
            q=np.asarray(q)+jitter[j]
            self.segment(p,q,radii[j],radii[j+1],5+j)
            p=q

    def makepose(self):
        l=self.letter
        # Cumulative elevation relative to the palm plane at each phalanx.
        # A straight digit extends along +y; flexion brings it toward +z,
        # then back towards the heel of the palm.
        curl=self.rng.uniform(-9,9)
        closed=np.array([84,166,240])+curl
        straight=np.array([4,6,8])+self.rng.uniform(-5,4)
        az=[3,0,-2,-6]
        angles=[closed.copy() for i in range(4)]
        if l=='A': angles=[np.array([78,165,237])+curl for i in range(4)]
        if l=='B':
            angles=[straight.copy() for i in range(4)]
            az=[-1,0,1,2]
        if l in 'CODF':
            # Fingers curve around the negative space in C and O; D and F
            # instead bring a subset of fingertips to the opposed thumb.
            base={'C':[22,66,102], 'O':[38,89,131], 'D':[54,99,143], 'F':[53,109,151]}[l]
            angles=[np.array(base)+self.rng.uniform(-7,7)+i*2 for i in range(4)]
            if l=='D': angles[0]=straight+self.rng.uniform(-3,2)
            if l=='F':
                for j in [1,2,3]: angles[j]=straight+self.rng.uniform(-2,7)
                spread=self.rng.uniform(2,12); az=[0,spread*1.3,0,-spread*1.4]
        if l=='E': angles=[np.array([42,149,211])+curl for i in range(4)]
        if l in 'GHLKPX': angles[0]=straight.copy()
        if l in 'GH':
            angles[0]=np.array([0,0,1])+self.rng.uniform(-4,3)
            az[0]=0
            if l=='H':
                angles[1]=angles[0]+self.rng.uniform(-3,4)
                az[1]=az[0]+self.rng.uniform(-2,2)
        if l in 'IY': angles[3]=straight.copy(); az[3]=-self.rng.uniform(5,18)
        if l in 'KP':
            angles[0]=straight.copy()
            angles[1]=np.array([45,50,52])+self.rng.uniform(-11,11)
            az[:2]=[8,-9]
        if l=='T':
            # The thumb protrudes across the folded fingers. Also include the
            # lateral presentation with a pointing digit and raised thumb.
            az[0]=0
        if l=='X': angles[0]=np.array([5,80,151])+self.rng.uniform(-6,6)
        if l=='Q': angles[0]=straight.copy()
        if l in 'MNT':
            # Fingers draped over the thumb are less deeply flexed.
            cover={'M':3,'N':2,'T':1}[l]
            for i in range(cover): angles[i]=np.array([52,142,190])+curl+i*2
        if l=='R':
            angles[0]=straight+1; angles[1]=straight+12
            az[:2]=[-self.rng.uniform(6,14),self.rng.uniform(6,14)]
        if l=='S': angles=[np.array([69,162,230])+curl for i in range(4)]
        if l in 'UVW':
            angles[0]=straight; angles[1]=straight+self.rng.uniform(-3,4)
            s=self.rng.uniform(8,17)
            if l=='V': az[:2]=[s,-s*.64]
            if l=='U': az[:2]=[-1,0]
            if l=='W':
                angles[2]=straight+self.rng.uniform(-2,4)
                az[:3]=[s,1,-s]
        if l=='T' and self.variant>.2: angles[0]=straight
        # When a thumb lies under curled fingers, contact pushes the covering
        # fingers up. Flexion advances across the knuckle row; the uncovered
        # digits are folded closer to the distal palmar crease.
        if l in 'MN':
            cover=3 if l=='M' else 2
            for i in range(4):
                if i<cover:
                    angles[i]=np.array([43+4*i,134+4*i,202])+curl
                else:
                    angles[i]=np.array([98,183,241])+curl
        if l=='S':
            # Opposed thumb over the flexed index and middle fingers.
            angles=[np.array([86,173,220])+curl for j in range(4)]
        if l=='X' and self.variant<.55:
            # Flex the interphalangeal joints without making a full fist at
            # the MCP joint. Some signers curl the tip towards the palm.
            angles[0]=np.array([9,56,131])+curl
        if l in 'GH' and self.rng.random()<.5:
            angles[0]+=self.rng.uniform(0,11)
        for i in range(4): self.finger(i,az[i],angles[i])
        h=self.plen; mc=self.mcp; tips=self.tip
        # Thumb opposition and abduction are specified by joint landmarks
        # relative to the palm or to the finger contacted, never in image space.
        if l=='A':
            end=v3(.50,.57*h,.17); pts=[(.47,-.02,.16),(.53,.24,.20),end]
        elif l=='B': pts=[(.43,-.05,.25),(.19,.16,.34),(-.085,.26,.30)]
        elif l=='C':
            spread=self.rng.uniform(.85,1.13)
            pts=[(.55,-.06,.22),(.67,.11,.48),(.72,.24*spread,.70*spread)]
        elif l in 'ODF':
            if l=='O': target=tips[0]*.6+tips[1]*.4
            elif l=='D': target=tips[1]
            else: target=tips[0]
            target=target+v3(.016,-.055,-.045)
            pts=[(.52,-.05,.21),target*.61+v3(.26,-.08,.055),target]
        elif l=='E': pts=[(.45,-.05,.24),(.18,.13,.40),(-.15,.15,.42)]
        elif l in 'GQ':
            # Index and thumb point along nearly parallel paths.
            pts=[(.55,-.03,.16),(.62,.29,.26),(.62,.59,.31)]
            if l=='Q': pts=[(.56,-.08,.30),(.70,.19,.51),(.73,.47,.65)]
        elif l=='H': pts=[(.46,-.07,.24),(.32,.15,.44),(.12,.30,.51)]
        elif l in 'KP':
            target=mc[1]+(self.pip[1]-mc[1])*.88
            pts=[(.52,-.01,.21),(.40,.29,.35),target+v3(.06,0,.06)]
        elif l in 'LY':
            rise=self.rng.uniform(-.05,.34) if l=='L' else self.rng.uniform(.29,.60)
            pts=[(.58,-.14,.10),(.86,-.05+rise*.48,.15),(1.10,-.02+rise,.17)]
        elif l=='T' and self.variant>.2:
            pts=[(.54,-.08,.26),(.68,.07,.48),(.78,.18,.70)]
        elif l in 'MNT':
            gap={'M':2,'N':1,'T':0}[l]; xx=(mc[gap][0]+mc[gap+1][0])/2
            pts=[(.49,-.08,.18),((xx+.45)*.5,.21,.29),(xx,.54*h+.03,.32)]
        elif l=='S':
            # Cross over the index to the radial edge of the ring finger.
            pts=[(.49,.02,.34),(.22,.35,.58),(-.13,.49,.61)]
        elif l=='W': pts=[(.44,-.07,.25),(.16,.11,.39),self.tip[3]+v3(.025,-.025,.045)]
        else: pts=[(.47,-.06,.25),(.32,.20,.43),(.035,.31,.47)]
        self.thumb(pts)


def camera(letter,rng,variant):
    # Camera yaw interpolates between the palm view and the thumb-side or
    # little-finger-side view. In profile, palmar flexion projects right.
    # Roll follows the direction that the ASL letter is presented.
    az,dp,rot=0.,4.,0.
    if letter in 'AIMSNTY': az=14.; dp=11.; rot=-5.
    if letter in 'MNS': az=25.; dp=18.; rot=-10.
    if letter=='E': az=10.; dp=13.; rot=-4.
    if letter=='B': az=4.; dp=3.; rot=0.
    if letter in 'CO': az=70.; dp=12.;rot=-9.
    if letter=='D': az=26.;dp=6.;rot=1.
    if letter=='F': az=42.;dp=6.;rot=-5.
    if letter in 'GH':
        # View the sideways index finger from the dorsum, not the palm:
        # standing the hand on its ulnar side puts the index above the fist.
        az=152. if letter=='G' else 136.; dp=6.; rot=-83.
    if letter in 'KR': az=-27.;dp=3.;rot=6.
    if letter=='K': az=-54.; rot=2.; dp=-41.
    if letter=='L': az=3.;dp=4.;rot=2.
    if letter=='P': az=63.;dp=50.;rot=68.
    if letter=='Q': az=-86.;dp=35.;rot=154.
    if letter=='T': az=-157.;dp=8.;rot=-86.
    if letter in 'UVW': az=6.; dp=4.;rot=3.
    if letter=='X': az=49.;dp=6.;rot=-5.
    if letter in 'CO' and variant<.65: az=90.;dp=-3.
    if letter=='X' and variant<.65: az=105.;dp=-5.
    if letter in 'MNS' and variant<.78: az=-24.;dp=-4.;rot=-13.
    if letter=='K' and variant<.55: az=-48.;dp=-18.;rot=5.
    if letter=='T':
        if variant>.2: az=-70.;dp=-4.;rot=-96.
        else: az=2.;dp=6.;rot=-10.
    if letter=='H' and variant<.6: az=164.;dp=-7.;rot=-82.
    a=deg(az+rng.uniform(-24,24)); t=deg(dp+rng.uniform(-15,14)); b=deg(rot+rng.uniform(-14,14))
    # Map to (horizontal, vertical up, depth toward viewer).
    yaw=np.array([[cos(a),0,sin(a)],[0,1,0],[-sin(a),0,cos(a)]])
    pitch=np.array([[1,0,0],[0,cos(t),-sin(t)],[0,sin(t),cos(t)]])
    roll=np.array([[cos(b),-sin(b),0],[sin(b),cos(b),0],[0,0,1]])
    return roll@pitch@yaw


def render(letter,rng,n):
    hand=Hand(letter,rng)
    cam=camera(letter,rng,hand.variant)
    plen=hand.plen
    # Ellipsoids for the heel, thenar and hypothenar pads, and distal palm.
    parts=[((0,-.015,0),(.483,plen*.54,hand.pth)),
           ((0,plen*.25,-.005),(.492,plen*.29,hand.pth*.9)),
           ((.224,-plen*.18,.062),(.260,plen*.33,hand.pth*.85)),
           ((-.255,-plen*.105,.045),(.225,plen*.37,hand.pth*.78))]
    radii=np.array([p[1] for p in parts]); cen=np.array([p[0] for p in parts])@cam.T
    screenr=np.sqrt(radii**2 @ (cam**2).T)
    lo=(cen-screenr).min(axis=0); hi=(cen+screenr).max(axis=0)
    bones=[]
    for lp,lq,r,s,kind in hand.bones:
        p,q=cam@lp,cam@lq
        lo=np.minimum(lo,np.minimum(p-r,q-s));hi=np.maximum(hi,np.maximum(p+r,q+s))
        bones.append((p,q,r,s,kind))
    # A camera is framed on the whole gesture, not on an individual joint.
    # Include a little forearm before choosing the crop.
    wrist=cam@hand.wrist
    if letter in 'GHT': armangle=rng.uniform(-59,-27)
    elif letter=='P': armangle=rng.uniform(40,82)
    elif letter=='Q': armangle=rng.uniform(105,200)
    else: armangle=rng.uniform(-22,16)
    arm=v3(sin(deg(armangle)),-cos(deg(armangle)),-rng.uniform(.10,.30))
    fore= wrist + arm*rng.uniform(.12,.64)
    lo[:2]=np.minimum(lo[:2],fore[:2]-.17); hi[:2]=np.maximum(hi[:2],fore[:2]+.17)
    size=max(hi[0]-lo[0],hi[1]-lo[1])*rng.uniform(.84,1.18)
    center=.5*(hi+lo)
    center[:2]+=size*rng.uniform(-.049,.049,2)
    px=size/n*rng.uniform(.95,1.06);py=size/n
    u=(np.arange(n)-(n-1)*.5)*px+center[0]
    v=((n-1)*.5-np.arange(n))*py+center[1]
    # Supersampled ray casting. Each primitive contributes the closest
    # surface depth and a surface normal, with soft anatomical shading.
    zbuf=np.full((n,n),-25.,float)
    image=np.zeros((n,n),float)
    lid=rng.uniform(-.6,.1); elev=rng.uniform(.45,.92)
    light=np.array([lid,elev,rng.uniform(.45,.9)]);light/=np.linalg.norm(light)
    albedo=rng.uniform(.40,.76)
    def patch(xmin,xmax,ymin,ymax):
        x0=max(0,int(np.searchsorted(u,xmin)));x1=min(n,int(np.searchsorted(u,xmax,side='right')))
        y0=max(0,int(np.searchsorted(-v,-ymax)));y1=min(n,int(np.searchsorted(-v,-ymin,side='right')))
        return (slice(y0,y1),slice(x0,x1)),u[None,x0:x1],v[y0:y1,None]
    def draw(sl,z,inside,nx,ny,nz,tone,crease=1.):
        buf=zbuf[sl]; win=inside & (z>buf)
        illumination=nx*light[0]+ny*light[1]+nz*light[2]
        brightness=tone*(.63+.44*illumination+.07*np.maximum(illumination,0)**8)*crease
        np.copyto(buf,z,where=win);np.copyto(image[sl],brightness,where=win)
    for c,rad,r2 in zip(cen,radii,screenr):
        sl,x,y=patch(c[0]-r2[0],c[0]+r2[0],c[1]-r2[1],c[1]+r2[1]);x=x-c[0];y=y-c[1]
        Q=(cam*(1/rad**2))@cam.T
        bb=(Q[0,2]*x+Q[1,2]*y)/Q[2,2]
        det=bb**2+(1-Q[0,0]*x*x-2*Q[0,1]*x*y-Q[1,1]*y*y)/Q[2,2]
        z=np.sqrt(np.maximum(det,0))-bb
        nx=Q[0,0]*x+Q[0,1]*y+Q[0,2]*z; ny=Q[1,0]*x+Q[1,1]*y+Q[1,2]*z; nz=Q[2,0]*x+Q[2,1]*y+Q[2,2]*z
        norm=np.sqrt(nx**2+ny**2+nz**2+1e-12)
        draw(sl,z+c[2],det>=0,nx/norm,ny/norm,nz/norm,albedo)
    sleeve=rng.random()<.25
    wriststart=cam@v3(0,-plen*.37,-.04)
    join=wrist+arm*.26
    bones.append((wriststart,join,.317,.305,-1))
    end=join+arm*4.
    bones.append((join,end,.305,.51,-1))
    if sleeve:
        cuff=join+arm*rng.uniform(.25,.92)
        bones.append((cuff,end,.37,.55,-2))
    for p,q,r,s,kind in bones:
        R=max(r,s)*1.003
        sl,x,y=patch(min(p[0],q[0])-R,max(p[0],q[0])+R,min(p[1],q[1])-R,max(p[1],q[1])+R)
        x=x-p[0];y=y-p[1]
        dx,dy,dz=q-p
        den=dx*dx+dy*dy+1e-14
        mid=(x*dx+y*dy)/den
        dline=(x*dy-y*dx)**2/den
        rad=r+(s-r)*np.clip(mid,0,1)
        # Front intersection of a ray with a swept sphere; clipping the
        # tangent location to the segment gives spherical end caps.
        alpha=np.clip(mid+dz*np.sqrt(np.maximum(rad*rad-dline,0)/(den*(den+dz*dz))),0,1)
        rad=r+(s-r)*alpha
        nx=(x-alpha*dx)/rad;ny=(y-alpha*dy)/rad
        nz2=1-nx*nx-ny*ny;nz=np.sqrt(np.maximum(nz2,0))
        crease=1.
        if 2<=kind<=7:
            # The flexion creases wrap around the joint, with stronger
            # creases at the interphalangeal joints than at the fingertip.
            crease=1-.19*np.exp(-((alpha-.085)/.092)**2)*(mid>=0)*(mid<=1)*(kind not in (2,5))
        draw(sl,p[2]+alpha*dz+nz*rad,nz2>=0,nx,ny,nz,albedo if kind!=-2 else albedo*.33,crease)
    mask=zbuf>-20
    # Ambient occlusion from the depth discontinuities between skin surfaces.
    ambient=np.zeros_like(image)
    for dy,dx in [(0,2),(-2,0),(1,-2),(-1,1)]:
        other=np.roll(zbuf,(dy,dx),axis=(0,1))
        ambient+=np.clip((other-zbuf-.038)*1.5,0,.16)
    image*=1-.94*ambient
    bg=rng.uniform(.38,.84) + np.linspace(-1,1,n)[None,:]*rng.uniform(-.065,.065) + np.linspace(-1,1,n)[:,None]*rng.uniform(-.07,.07)
    image=np.where(mask,image,bg)
    a=image.reshape(28,n//28,28,n//28).mean(axis=(1,3))
    m=mask.reshape(28,n//28,28,n//28).mean(axis=(1,3))
    return a,m


def build(output):
    output.mkdir(parents=True,exist_ok=True)
    source=FEATURES+INFERENCE
    (output/'solution.py').write_text(source)
    scope={};exec(FEATURES,scope)
    letters='ABCDEFGHIKLMNOPQRSTUVWXY'
    poses=1024
    codes={name:np.empty((len(letters)*poses,side*side*12),np.int16) for name,side in [('coarse',6),('fine',12)]}
    for k,l in enumerate(letters):
        rng=np.random.default_rng(7379+191*k)
        for j in range(0,poses,32):
            a=[];m=[]
            for i in range(32):
                im,mask=render(l,rng,56);a.append(im);m.append(mask)
            # Each image is encoded on its own. The batch dimension here is
            # only to vectorize the arithmetic, not to compute any statistic.
            f=scope['descriptor'](np.array(a),np.array(m))
            for name,g in [('coarse',6),('fine',12)]:
                v=scope['pooldesc'](f,g)
                codes[name][k*poses+j:k*poses+j+32]=np.rint(16384*v).astype(np.int16)
        print('rendered',l,flush=True)
    np.savez(output/'constants.npz',**codes)

if __name__=='__main__':
    build(Path(sys.argv[1]))
