"""Data-independent ASL pose renderer and compilation to a correlation network.

The unit of length in the model is approximately one palm width. x is toward
 the index side of a frontal right palm, y toward the fingertips, z toward
 the observer. A pose describes anatomical joints, not image pixels.
"""
import sys, json, math
from pathlib import Path
import numpy as np

LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'
RAD = math.pi/180

def normalize(a):
    return a / max(1e-12, float(np.sqrt((a*a).sum())))

def rot(i, a):
    a *= RAD; c,s = math.cos(a), math.sin(a)
    R=np.eye(3); j,k=(i+1)%3,(i+2)%3
    R[j,j]=R[k,k]=c; R[j,k]=-s; R[k,j]=s
    return R

def hand(letter, r):
    """Return ellipsoids and tapered phalanges in a common local frame."""
    u=r.uniform
    width=u(.89,1.12); length=u(.90,1.12); thick=u(.88,1.13)
    F=u(.92,1.09)*length**.4
    # An alternate off-axis presentation of G uses a palm tilted toward
    # the viewer and a flexed index MCP; H keeps two shafts side by side.
    foreshortened=(letter in 'GH' and r.random()<.5)
    objects=[]
    def ell(c,ra,ry=0,par=2,tone=1):
        objects.append(('e',np.array(c),np.array(ra),rot(2,ry),par,tone))
    def tube(p,q,a,b,flat=.88,tone=1):
        objects.append(('t',np.array(p),np.array(q),a,b,flat,tone))
    # A blunt, slightly asymmetric palm, with a heel, thumb mound and wrist.
    ell((0,-.015,0),(.49*width,.57*length,.208*thick),-3,2.85)
    ell((.245*width,-.14*length,.113*thick),(.219*width,.393*length,.135*thick),-22,2.2)
    ell((-.25*width,-.10*length,.075*thick),(.193*width,.345*length,.128*thick),12,2.3)
    wristX=u(-.13,.09); wx=.30*width*u(.94,1.09)
    # Long ellipse-section forearm (and occasional cuff), behind palm.
    wrist=np.array((-.025,-.32*length,-.02))
    fore=np.array((wristX,-2.05,-.07))
    tube(wrist,fore,wx,wx*1.21,.67)
    if r.random()<.34:
        t=u(.23,.45); stop=(1-t)*wrist+t*fore
        tube(stop,fore,wx*1.19,wx*1.45,.76,u(.20,.60))
    # Finger indices run little, ring, middle, index. Flexion angles are
    # from straight at MCP/PIP/DIP, independently of sideways abduction.
    fingers=[]; bases=[]
    for j,(xx,yy,L,rr) in enumerate(zip((-.36,-.145,.095,.323),(.286,.386,.427,.365),(.71,.933,1.,.923),(.095,.111,.116,.108))):
        base=np.array([xx*width,yy*length,-.008 if j in (0,3) else .014]); bases.append(base)
        flex=np.array([64.,84.,57.]); abd=(-7.,-3.,1.,5.)[j]
        straight=False
        if letter=='B' or (letter=='F' and j<3):
            straight=True; abd=((-4.,-2.,0.,1.5) if letter=='B' else (-25.,-9.,10.,0.))[j]
        elif letter in 'DLX' and j==3:
            straight=letter!='X'; abd=u(-5,8); flex=np.array([5.,61.,57.])
        elif letter in 'IY' and j==0:
            straight=True; abd=u(-22,-6) if letter=='Y' else u(-18,-3)
        elif letter in 'UKRVPW' and j in (2,3):
            straight=True
            abd={'U':(2.,0.),'V':(-16.,14.),'R':(11.,-11.),'W':(0.,24.),'K':(-16.,9.),'P':(-16.,9.)}[letter][j-2]
        elif letter=='W' and j==1: straight=True; abd=-20
        elif letter in 'GH' and (j==3 or (j==2 and letter=='H')):
            straight=True; abd=1
        elif letter=='Q' and j==3:
            straight=True; abd=2
        if straight: flex=np.array([1.,3.,2.])+r.normal(0,2,3)
        if letter in 'CO':
            flex=np.array([34.,38.,25.]) if letter=='C' else np.array([47.,58.,38.]); abd=(j-1.5)*1.4
        if (letter=='D' and j<3) or (letter=='F' and j==3):
            flex=np.array([48.,63.,44.]); abd=abd*.6
        if letter=='R' and j==2: flex=np.array([12.,-8.,2.])
        if letter in 'KP' and j==2: flex=np.array([u(39,81),7.,4.]);abd=u(-9,6)
        if letter in 'UW' and j>=2: flex+=u(-6,6)
        if letter in 'GH' and foreshortened and (j==3 or (j==2 and letter=='H')):
            flex=np.array([u(65,91),3.,2.]);abd=0.
        if letter=='X' and j==3: flex=np.array([u(-4,22),u(65,108),u(43,76)])
        if letter=='E': flex=np.array([u(4,26),u(103,127),u(55,74)])
        if letter=='A': flex=np.array([52.,101.,65.])
        if letter in 'MNT':
            pinch={'M':(1,2,3),'N':(2,3),'T':(3,)}[letter]
            flex=np.array([48.,78.,46.] if j in pinch else [70.,95.,54.])
        if letter in 'MN' and j in pinch:
            flex=np.array([u(22,39),u(63,92),u(30,46)]); abd-=u(3,9)
        if letter=='S': flex=np.array([71.,90.,47.])
        if letter=='Q' and j==3: flex=np.array([13.,10.,4.])
        if letter=='T' and j==3: flex=np.array([75.,12.,9.]);abd=0
        if letter=='T' and j==2: flex=np.array([63.,42.,36.]);abd=0
        flex+=r.normal(0, 5 if not straight else 2.0, 3); abd+=u(-2.5,2.5)
        bend=0.; pts=[base]; cur=base; rad=rr*width**.5*thick**.5*u(.96,1.05)
        for frac,turn in zip((.447,.306,.247),flex):
            bend+=turn*RAD; a=abd*RAD
            v=np.array([math.sin(a)*math.cos(bend),math.cos(a)*math.cos(bend),math.sin(bend)])
            cur=cur+v*(L*F*frac); pts.append(cur)
        fingers.append(pts)
        # A groove is left between neighboring digits. Model the joint pads
        # as slightly more prominent than the shafts.
        for k,(a,b) in enumerate(zip(pts,pts[1:])):
            aa=(1,.96,.86)[k]*rad; bb=(.97,.88,.74)[k]*rad
            tube(a,b,aa,bb,.86, (.97,1.,1.02)[k])
    # Thumb carpometacarpal, MCP, IP and tip. Choose free, adducted,
    # opposed, tucked, overlaid, or contacting a specific finger.
    root=np.array((.31*width,-.27*length,.04*thick))
    p=np.array([.48,-.01,.15]); q=np.array([.52,.25,.23]); t=np.array([.52,.51,.27])
    if letter=='A':
        p=np.array([.49,-.00,.13]);q=np.array([.52,.28,.18]);t=np.array([.54,.53,.23])
    elif letter in 'LY':
        slope=u(.20,.75);p=np.array([.53,-.06,.08]);q=p+np.array([.27,slope*.27,.04]);t=q+np.array([.27,slope*.27,.015]);
    elif letter in 'CO':
        p=np.array([.44,-.024,.31]);q=np.array([.45,.062,.57]);t=np.array([.42,.21,.80])
        if letter=='O':
            t=fingers[3][-1]+[.016,-.04,.015];q=t+[.015,-.225,-.09];p=(q+root)/2+[.055,0,.04]
    elif letter in 'DF':
        t=fingers[2 if letter=='D' else 3][-1]+[.015,-.045,.015]
        q=t+[.055,-.17,-.105];p=(q+root)/2+[.10,.01,.02]
    elif letter in 'KP':
        p=np.array([.46,-.035,.22]);q=np.array([.31,.27,.33]);t=np.array([.14,.57,.37]);
    elif letter=='Q':
        p=np.array([.55,-.005,.18]);q=p+[.21,.21,.16];t=q+[.17,.20,.14]
    elif letter=='G':
        p=np.array([.46,.025,.17]);q=np.array([.48,.35,.24]);t=np.array([.47,.66,.26])
    elif letter in 'MNT':
        j={'M':0,'N':1,'T':2}[letter];xx=(bases[j][0]+bases[j+1][0])*.5
        t=np.array([xx,.47+.12*(letter=='T'),.39]);q=t+[.14,-.16,-.025];p=(root+q)/2+[.055,.00,.06]
    elif letter=='S':
        p=np.array([.45,.04,.30]);q=np.array([.18,.32,.44]);t=np.array([-.095,.42,.46])
    elif letter=='E':
        p=np.array([.45,-.05,.22]);q=np.array([.15,.02,.32]);t=np.array([-.11,.073,.31]);
    else:
        # Folded over the palm, securing the other fingers (the group
        # differs between a W, an I, and a two-finger sign).
        end={'B':(-.245,.12,.29),'I':(-.015,.35,.42),'W':(-.35,.34,.44),'X':(-.115,.26,.39),'H':(-.13,.27,.42)}.get(letter,(-.15,.32,.43))
        t=np.array(end); q=(np.array([.48,.13,.33])+t)*.5+[0,.035,.055]; p=np.array([.45,-.018,.25])
    p,q,t=[a+r.normal(0,.028,3) for a in (p,q,t)]
    for a,b,r1,r2 in ((root,p,.163,.15),(p,q,.145,.125),(q,t,.125,.095)):
        tube(a,b,r1*thick,r2*thick,.86)
    # Orientation families: curling into z looks like a C from the side;
    # G and H point sideways; the wrist is supinated and lowered for Q.
    yaw=u(-21,21); pitch=u(-13,13); roll=u(-14,14)
    if letter in 'CO': yaw=u(44,92); roll+=u(-7,0);pitch+=u(-9,13)
    elif letter=='F': yaw=u(19,61);roll+=u(-4,12)
    elif letter=='D': yaw=u(7,47)
    elif letter=='X': yaw=u(39,85);pitch+=u(0,12)
    elif letter=='U':yaw=u(-45,24)
    elif letter=='R':yaw=u(-51,35)
    elif letter=='B':yaw=u(-29,27)
    elif letter in 'GHT':
        if letter in 'GH':
            yaw=u(-90,-44);roll=-u(63,101);pitch=u(-19,15)
        else:
            # T can also be presented with palm turned in and fingers
            # horizontal, emphasizing the projecting thumb.
            if r.random()<.90: yaw=u(53,91);roll=u(-18,9);pitch=u(-15,8)
    elif letter in 'KP':
        yaw=u(8,64); pitch=u(-16,18)
        if letter=='P':roll=u(58,127);yaw=u(-63,-12);pitch=u(-35,13)
    elif letter=='Q':
        yaw=u(19,70);pitch=u(32,66);roll=u(147,208)
    elif letter in 'MNES': yaw=u(-43,43)
    elif letter=='L':yaw=u(-20,32)
    R=rot(2,roll)@rot(0,pitch)@rot(1,yaw)
    # In side-on signs, MCP flexion is independent of wrist flexion. Rotate
    # after bending the finger, rather than rotating a whole upright sign.
    if letter in 'GH' and foreshortened:
        R=rot(2,u(-31,10))@rot(0,u(-9,16))@rot(1,u(48,84) if letter=='G' else u(25,70))
    elif letter in 'GH':
        view=u(135,173) if letter=='G' else u(25,82)
        roll=-u(24,88);pitch=u(-25,15)
        # Elevate the index side toward the upper edge of the fist.
        R=rot(2,roll)@rot(0,pitch)@rot(1,view)
        bend=u(4,55) if letter=='G' else u(1,36)
        for j in ([3] if letter=='G' else [2,3]):
            for k in range(3):
                ix=len(objects)-15+3*j+k
                typ,p,q,a,b,f,tone=objects[ix]
                bendR=rot(0,bend);o=bases[j]
                objects[ix]=(typ,o+bendR@(p-o),o+bendR@(q-o),a,b,f,tone)
    elif letter in 'KP':
        yaw=u(-49,13);pitch=u(-34,-6);roll=u(-9,17) if letter=='K' else u(67,119)
        if letter=='P': pitch+=u(24,46)
        R=rot(2,roll)@rot(0,pitch)@rot(1,yaw)
    elif letter=='Q':
        # Thumb and index point towards the floor; flexion at the wrist
        # leaves the arm coming from below, not down from above the hand.
        R=rot(2,u(-40,15))@rot(0,u(-197,-145))@rot(1,u(-67,-17))
    # In camera space, articulate forearm at the wrist. It can be much
    # more nearly vertical than the hand for the bent-wrist signs.
    if letter in 'GHPTQ' or r.random()<.45:
        lo,hi={'G':(-78,-29),'H':(-75,-24),'P':(54,113),'Q':(-24,15),'T':(-32,4)}.get(letter,(-24,15))
        an=u(lo,hi)*RAD;dz=u(-.82,-.12)
        v=normalize(np.array([math.sin(an),-math.cos(an),dz]));v=R.T@v
        for ix in range(3,len(objects)-15):
            typ,p,q,a,b,f,tone=objects[ix]
            objects[ix]=(typ,wrist+v*np.linalg.norm(p-wrist),wrist+v*np.linalg.norm(q-wrist),a,b,f,tone)
    return objects,R

# Simple z-buffer renderer. Mesh construction and projection are independent
# of the resolution, so thin fingers can be positioned at subpixel offsets.
def render(letter,r, resolution=44):
    objects,R=hand(letter,r)
    H=resolution
    # Bounds only of this pose, without its long forearm. A loose square
    # camera crop includes some wrist and space above the fingertips.
    projected=[]; lo=np.full(2,999.);hi=-lo
    for obj in objects:
        typ=obj[0];p=R@obj[1]
        if typ=='e':
            _,_,q,i,j,tone=obj
            A=R@i@np.diag(q); ext=np.linalg.norm(A,axis=1)[:2]; start=p[:2]-ext;stop=p[:2]+ext
            desc=(typ,p,A,0,j,tone)
        else:
            _,_,q,i,j,k,tone=obj
            q=R@q; start=np.minimum(p[:2]-i,q[:2]-j);stop=np.maximum(p[:2]+i,q[:2]+j)
            desc=(typ,p,q,i,j,k,tone)
        projected.append(desc)
        if typ=='e' or np.linalg.norm(q-p)<1.25:
            lo=np.minimum(lo,start);hi=np.maximum(hi,stop)
    arm0,arm1=projected[3][1:3]
    bottom=(arm0+.28*(arm1-arm0))[:2]
    lo=np.minimum(lo,bottom);hi=np.maximum(hi,bottom)
    span=max(hi-lo)*r.uniform(.85,1.18)
    # Aim between the center of the gesture and center of the wrist/palm,
    # accounting for crops in which the extended fingers meet the edge.
    anchor=(R@np.array([0.,.06,.07]))[:2]
    mid=(hi+lo)/2;mid=mid+r.uniform(.0,.34)*(anchor-mid)
    cx,cy=mid+r.normal(0,.062,2)*span
    step=span/H
    x=cx+(np.arange(H)+.5-H/2)*step;y=cy-(np.arange(H)+.5-H/2)*step
    depth=np.full((H,H),-5.); toneMap=np.ones((H,H)); shades=np.zeros((H,H))
    light=normalize(np.array([r.uniform(-.94,-.28) if r.random()<.85 else r.uniform(.25,.85),r.uniform(.25,.85),r.uniform(.30,.85)]))
    def patch(start,stop):
        l=max(0,int(math.floor((start[0]-x[0])/step)));rr=min(H,int(math.ceil((stop[0]-x[0])/step))+1)
        t=max(0,int(math.floor((y[0]-stop[1])/step)));b=min(H,int(math.ceil((y[0]-start[1])/step))+1)
        if l>=rr or t>=b:return None
        return (slice(t,b),slice(l,rr)),x[None,l:rr],y[t:b,None]
    for ob in projected:
        if ob[0]=='e':
            _,p,A,_,power,tone=ob
            # A quadric ellipsoid. The anatomical palm is blunt enough to
            # warrant a little extra width at the finger bases.
            ext=np.linalg.norm(A,axis=1)[:2];region=patch(p[:2]-ext,p[:2]+ext)
            if region is None:continue
            sl,X,Y=region; dx=X-p[0];dy=Y-p[1]
            inv=np.linalg.inv(A@A.T); B=inv[2,0]*dx+inv[2,1]*dy
            C=inv[0,0]*dx*dx+2*inv[0,1]*dx*dy+inv[1,1]*dy*dy-1
            disc=B*B-inv[2,2]*C; dz=(-B+np.sqrt(np.maximum(0,disc)))/inv[2,2]
            nx=inv[0,0]*dx+inv[0,1]*dy+inv[0,2]*dz;ny=inv[1,0]*dx+inv[1,1]*dy+inv[1,2]*dz;nz=B+inv[2,2]*dz
            zz=p[2]+dz;ok=disc>0
        else:
            _,p,q,a,b,flat,tone=ob;rad=max(a,b);region=patch(np.minimum(p[:2],q[:2])-rad,np.maximum(p[:2],q[:2])+rad)
            if region is None:continue
            sl,X,Y=region; dx=X-p[0];dy=Y-p[1];vx,vy,vz=q-p
            ll=vx*vx+vy*vy+1.e-10
            # Front surface of the swept rounded segment, including the
            # shift of its visible axial position on tipped phalanges.
            t0=(dx*vx+dy*vy)/ll; dist2=(dx-t0*vx)**2+(dy-t0*vy)**2
            rr=a+(b-a)*np.clip(t0,0,1); dzax=vz/max(flat,.001)
            t=np.clip(t0+dzax*np.sqrt(np.maximum(0.,(rr**2-dist2)/(ll*(ll+dzax*dzax)))),0,1)
            nx=dx-t*vx;ny=dy-t*vy; rr=a+(b-a)*t
            d=rr**2-nx**2-ny**2;h=np.sqrt(np.maximum(0.,d));nz=h/flat
            zz=p[2]+t*vz+flat*h;ok=d>0
        mask=ok&(zz>depth[sl]); depth[sl][mask]=zz[mask];toneMap[sl][mask]=tone
        ln=np.sqrt(nx**2+ny**2+nz**2+1.e-10)
        lam=(light[0]*nx+light[1]*ny+light[2]*nz)/ln
        shades[sl][mask]=lam[mask]
    # A local horizon darkens the back surface at finger/palm occlusions.
    fg=depth>-4; shadows=np.zeros_like(depth); dpx=np.pad(depth,8,mode='edge')
    for ox,oy in ((2,0),(-2,0),(0,2),(0,-2)):
        neighbor=dpx[8+oy:8+oy+H,8+ox:8+ox+H]
        shadows+=np.clip((neighbor-depth-.045)/.29,0,1)*.25
    # Projecting phalanges cast a shadow on the palm and between the
    # knuckles; sample the horizon on the incoming-light side.
    cast=np.zeros_like(depth)
    for n in (1,2,4,6,8):
        ox=int(round(n*light[0]));oy=-int(round(n*light[1]));d=dpx[8+oy:8+oy+H,8+ox:8+ox+H]-depth
        cast=np.maximum(cast,np.clip((d-n*step*light[2]-.025)/(.13+n*.006),0,1))
    skin=r.uniform(.46,.66);rel=r.uniform(.21,.35)
    bg=skin+r.uniform(-.03,.26) if r.random()<.81 else skin-r.uniform(.07,.24)
    # The wrist is useful as a direction cue, but an off-frame forearm
    # should not be necessary to recognize a sign. Fade the arm silhouette
    # out beyond the heel of the palm in the more tightly focused templates.
    pigment=toneMap*(skin+rel*shades-.19*shadows-r.uniform(.07,.22)*cast)
    if r.random()<.60:
        d=(arm1-arm0)[:2];length=np.linalg.norm(d);d/=max(1.e-5,length)
        s=(x[None,:]-arm0[0])*d[0]+(y[:,None]-arm0[1])*d[1]
        pigment=bg+(pigment-bg)*np.clip(1.45-1.55*s,0,1)
    out=np.where(fg,pigment,bg)
    return np.clip(out,0,1)

# Feature compiler. Every row of the dictionary is generated from a single
# render. No class-level (or dictionary-level) normalization or calibration.
N=28
K=4
EPS=1e-5
Y,X=np.mgrid[-2:3,-2:3]
SMOOTH=np.exp(-(X*X+Y*Y)/(2*.86**2));SMOOTH/=SMOOTH.sum()
angles=np.arange(K)*np.pi/K
BASE=np.array([(X*np.cos(a)+Y*np.sin(a))*SMOOTH for a in angles]);BASE/=np.abs(BASE).sum((1,2),keepdims=True)
NARROW=np.exp(-(X*X+Y*Y)/(2*.76**2));NARROW/=NARROW.sum()
RIDGE=np.array([(((X*np.cos(a)+Y*np.sin(a))/.76)**2-1)*NARROW for a in angles]); RIDGE-=RIDGE.mean((1,2),keepdims=True);RIDGE/=np.abs(RIDGE).sum((1,2),keepdims=True)
# Center-surround and half-wave rectified ridge detectors encode finger
# ridges, valleys and tips in addition to the direction of the outline.
SUR=np.exp(-(X*X+Y*Y)/(2*1.55**2));SUR/=SUR.sum();SUR[2,2]-=1.;SUR/=np.abs(SUR).sum()
FILTERS=np.concatenate((BASE,-BASE,RIDGE,-RIDGE,SUR[None]),0)
C=len(FILTERS)
E=np.eye(K);Z=np.zeros_like(E)
MIX=np.zeros((4*K+1,C));MIX[:K,:2*K]=np.c_[E,E];MIX[K:2*K,:2*K]=np.c_[E,-E];MIX[2*K:,2*K:]=np.eye(2*K+1)
GAINS=np.repeat([1.,.71,.51,.51],K).tolist()+[.48]
GAINS=np.array(GAINS)
# Invert the perspective of the feature grid: each cell is a channel, its
# histogram bins become positions. This implements HOG-style normalization
# with the network's spatial norm. Two constant bins supply a contrast
# floor; sign pairs ensure that the normalizer has identically zero mean.
LOCAL=np.vstack((MIX,-MIX,np.zeros((2,C))))
LB=np.array([0.]*(2*C)+[.26,-.26])
CELLS=12*12

def unit_hist(h):
    den=np.sqrt(((h*h).sum(0)+.26**2)/(C+1)+EPS)
    return h/den


def resized(im):
    # Bilinear resample after a small low pass to model finite pixel area.
    imp=np.pad(im,1,mode='edge');im=(4*im+imp[0:-2,1:-1]+imp[2:,1:-1]+imp[1:-1,0:-2]+imp[1:-1,2:])*.125
    H=im.shape[0];x=(np.arange(N)+.5)*H/N-.5;a=np.floor(x).astype(int);w=x-a
    im=im[a,:]*(1-w[:,None])+im[a+1,:]*w[:,None]
    return im[:,a]*(1-w)+im[:,a+1]*w

def feats(im):
    im=(im-im.mean())/np.sqrt(im.var()+EPS)
    # Use valid filtering, suppressing artificial edges at the crop boundary.
    win=np.lib.stride_tricks.sliding_window_view(im,(5,5))
    g=np.maximum(0,(win.reshape(24*24,25)@FILTERS.reshape(C,25).T).T)
    g-=.61*np.maximum(g-.28,0)+.23*np.maximum(g-.76,0)
    raw=(MIX@g).reshape(C,12,2,12,2).mean((2,4))
    h=unit_hist(raw)*GAINS[:,None,None]
    coarse=raw[:2*K].reshape(2*K,6,2,6,2).mean((2,4))*GAINS[:2*K,None,None]
    # A unit-vector correlation in each of two complementary spaces.
    # Local-normalized fine features are good for digit shape, while the
    # coarse unnormalized energy map preserves the distribution of contrast.
    def unit_map(m):
        m=m-m.mean((1,2),keepdims=True)
        return normalize(m).ravel()
    return unit_map(h),unit_map(coarse)


# Number of independent camera/shape/light hypotheses per sign, arranged in
# small groups for a max then average reduction, less brittle than one argmax.
GROUPS=4; PER=384

def build(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True); layers=[];weights={}
    def layer(name,op,inputs,**kw):
        layers.append(dict(name=name,op=op,inputs=inputs if isinstance(inputs,list) else [inputs],**kw));return name
    def lin(name,op,inp,w,b,**kw):
        weights[name+'.W']=np.asarray(w,dtype=np.float32);weights[name+'.b']=np.asarray(b,dtype=np.float32)
        return layer(name,op,inp,out=w.shape[0],**kw)
    def conv(name,inp,w,b,pad=0):
        k=w.shape[-1];return lin(name,'conv',inp,w,b,kernel=[k,k],stride=1,padding=pad)
    im=layer('contrast','norm','image')
    v=conv('steerable_ramps',im,np.tile(FILTERS,(3,1,1))[:,None],np.repeat([0,-.28,-.76],C))
    v=layer('positive','relu',v)
    v=layer('cell_histograms','avgpool',v,size=[2,2],stride=[2,2])
    w=np.concatenate([t*LOCAL for t in (1.,-.61,-.23)],axis=1)[:,:,None,None]
    v=conv('moments',v,w,LB)
    # Read each 12x12 feature map with a one-hot filter bank, making cells
    # the channels and histogram bins the spatial axis for local norm.
    D=len(LB);S=12
    v=layer('stack_maps','reshape',v,shape=[1,D*S,S]);
    v=lin('cells_to_channels','conv',v,np.eye(CELLS).reshape(CELLS,1,S,S),np.zeros(CELLS),kernel=[S,S],stride=S,padding=0)
    v=layer('local_norm','norm',v)
    side=math.isqrt(D);assert side**2==D
    v=layer('stack_histograms','reshape',v,shape=[1,CELLS*side,side]);
    v=lin('take_features','conv',v,(GAINS[:,None]*np.eye(D)[:C]).reshape(C,1,side,side),np.zeros(C),kernel=[side,side],stride=side,padding=0)
    # Remove DC directions at inference too, with a fixed linear projection
    # followed by a joint norm for cosine distance.
    def unit_map(name,src,nch,npos):
        # Flatten each map as a channel-by-bin matrix. For channel centering
        # subtract its within-map average, broadcasting by an analytical
        # dense-on-positions convolution (shared across feature channels).
        side=math.isqrt(npos);src=layer(name+'_layout','reshape',src,shape=[1,nch*side,side]);
        vec=np.eye(npos)-1./npos
        src=lin(name+'_dc','conv',src,vec.reshape(npos,1,side,side),np.zeros(npos),kernel=[side,side],stride=side,padding=0)
        src=layer(name+'_stripes','reshape',src,shape=[1,npos,nch]);src=layer(name+'_norm','norm',src)
        return src
    v=unit_map('fine',v,C,CELLS)
    wc=np.eye(D)[:2*K]*GAINS[:2*K,None]
    c=conv('coarse_edges','moments',wc[:,:,None,None],np.zeros(2*K));c=layer('coarse_cells','avgpool',c,size=[2,2],stride=[2,2]);c=unit_map('coarse',c,2*K,36)
    count=len(LETTERS)*GROUPS*PER
    bank=np.empty((count,C*CELLS),np.float32);wide=np.empty((count,2*K*36),np.float32);idx=0
    for letter in LETTERS:
        for n in range(GROUPS*PER):
            seed=78277+1000003*LETTERS.index(letter)+n*127
            r=np.random.default_rng(seed)
            a,b=feats(resized(render(letter,r)))
            bank[idx]=a.reshape(C,CELLS).T.ravel()*(.61/math.sqrt(C*CELLS))
            wide[idx]=b.reshape(2*K,36).T.ravel()*(.39/math.sqrt(2*K*36));idx+=1
    v=lin('finger_match','dense',v,bank,np.zeros(count));c=lin('outline_match','dense',c,wide,np.zeros(count));v=layer('pose_matches','add',[v,c])
    v=layer('by_pose','reshape',v,shape=[24,GROUPS,PER]);v=layer('best_views','maxpool',v,size=[1,PER],stride=[1,PER]);v=layer('letter_evidence','avgpool',v,size=[GROUPS,1],stride=[GROUPS,1]);v=layer('letters','reshape',v,shape=[24])
    (path/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
    np.savez(path/'weights.npz',**weights)

if __name__=='__main__':build(sys.argv[1])
