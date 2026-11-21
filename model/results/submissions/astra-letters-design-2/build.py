"""Construct signed-gradient match filters from a procedural model of a signing hand.

Model coordinates are in palm half-widths, with the wrist at the origin,
positive v towards the four knuckles and positive u towards the thumb.
No assets, data, optimized parameters or dependencies other than numpy.
"""
import sys, json
from pathlib import Path
import numpy as np

ABC = 'ABCDEFGHIKLMNOPQRSTUVWXY'
R = np.random.default_rng(11813)
VIEWS = 640
GRID = 40
XI, YI = np.meshgrid(np.arange(GRID, dtype='f') + .5 - GRID/2,
                     GRID/2 - .5 - np.arange(GRID, dtype='f'))

def U(a,b): return R.uniform(a,b)
def vec(a): return np.asarray(a,dtype='f')

class Hand:
    def __init__(self):
        self.parts=[]
    def part(self,path,radii,depths,material=0.,flat=1.,frame=True):
        path=vec(path); n=len(path)
        rad=np.broadcast_to(vec(radii),(n,)).copy()
        dep=np.broadcast_to(vec(depths),(n,)).copy()
        self.parts.append((path,rad,dep,material,flat,frame))
    def arm(self,wrist,root,width,thick=1.):
        w=vec(wrist); d=vec(root)-w; d=d/np.linalg.norm(d)
        self.part([w+9*d, w+1.8*d, w], [width*1.42,width*1.1,width],[-.65,-.50,-.32],-.05,thick,False)
    def pad(self,wrist,knuckle,width=1.):
        w=vec(wrist);k=vec(knuckle);d=k-w
        self.part([w,w+.32*d,w+.72*d,k],vec([.60,.78,.90,.88])*width,
                  [-.30,-.23,-.18,-.21],-.02,.68)
    def digit(self,points,radius=.22,z=.26,tone=0.,taper=None,flat=1.):
        n=len(points)
        if taper is None: taper=np.linspace(1.,.77,n)
        self.part(points,vec(taper)*radius,z,tone,flat)

def front(c):
    h=Hand()
    h.arm((-.05,.23),(U(-.85,.10),-2),.62)
    h.pad((-.04,.30),(0,1.53))
    # Pinky to index, with the normal anatomical arch of MCP joints.
    ux=np.array([-.70,-.255,.218,.66])+R.uniform(-.035,.035,4)
    uy=np.array([1.68,1.94,2.03,1.92])+R.uniform(-.05,.05,4)
    lengths=np.array([1.46,1.82,2.02,1.90])*U(.89,1.12)
    widths=np.array([.204,.233,.241,.226])*U(.95,1.08)
    # Change the whole hand, not just extended fingers, when seen in oblique view.
    en={'B':[0,1,2,3], 'D':[3], 'F':[0,1,2], 'I':[0], 'K':[2,3],
        'L':[3], 'R':[2,3], 'U':[2,3], 'V':[2,3], 'W':[1,2,3],
        'X':[3], 'Y':[0]}.get(c,[])
    # Directions of the four fingers in the fan (angles away from upright).
    tilt={'B':[-5,-2,0,-2],'I':[-9,0,0,0], 'Y':[-24,0,0,0],
          'D':[0,0,0,-6],'K':[0,0,-26,-2], 'L':[0,0,0,0],
          'U':[0,0,-3,-4],'R':[0,0,8,-15],'V':[0,0,-18,15],
          'W':[0,-19,0,19],'F':[-29,-7,20,0]}.get(c,[0]*4)
    kshort=U(.40,.88)
    fold=U(.58,.83)
    for i,(x,y) in enumerate(zip(ux,uy)):
        z=0.03; r=widths[i]; L=lengths[i]; a=np.deg2rad(tilt[i]+U(-3,3));dx=np.sin(a)*L;dy=np.cos(a)*L
        if i in en and c!='X':
            if c=='K' and i==2: dy*=kshort; dx*=U(.9,1.45)
            if c=='R':
                # The middle crosses in front of the index.
                bend=.10 if i==2 else 0.
                h.digit([(x,y),(x+dx*.55+bend,y+dy*.55),(x+dx,y+dy)],r,[z,.1+(i==2)*.23,.15+(i==2)*.25]);continue
            if c=='F': # diverging fingers viewed partly from the ulnar edge
                dx+=U(0,.28)
            h.digit([(x,y),(x+.55*dx+U(-.06,.06),y+.54*dy),(x+dx,y+dy)],r,[z,-.015,.02],U(-.03,.03));continue
        if c=='X' and i==3:
            # In side view MCP flexion projects the proximal bone up and back,
            # the middle bone across and the distal phalanx back down.
            h.digit([(x,y),(x-.36,y+.74),(x-.52,y+1.22),(x-.31,y+1.55),(x+.23,y+1.68),(x+.63,y+1.65),(x+.72,y+1.40)],r,[0,.04,.09,.15,.21,.26,.35]);continue
        if c=='F' and i==3:
            # Pad of the flexed index and abducted thumb meet in an O.
            h.digit([(x,y),(x+.42,y+.41),(x+.83,y+.44),(x+1.14,y+.25)],r,[.02,.12,.21,.28]);continue
        if c=='D':
            y+=.12*i; xx=.13+i*.06
            h.digit([(x,y),(x+.07,y+.53),(x+xx,y+.64),(x+xx+.12,y+.35)],r,[.01,.06,.19,.43]);continue
        # MCP -> proximal knuckle -> folded fingertip. Overlapping depth layers
        # make shadows between adjacent digits without painting the creases.
        lift=fold
        if c=='B': lift=.35
        if c in 'E': lift+=.23
        if c in 'MN': lift+=.13
        top=y+lift
        lean=U(-.16,-.035)
        if c in 'MN': lean-=.10
        backx=x+lean
        drop={'A':.49,'S':.55,'E':.78,'M':.83,'N':.84}.get(c,.61)
        end=top-drop+U(-.06,.06)
        if c=='M' and i==0: end+=.16
        if c=='N' and i<2:end+=.12
        if c=='Y' or c=='L':end-=.10
        h.digit([(x,y-.10),(backx,top-.11),(backx+.035,top),(x+.10,end)],r,
                [-.05,.04,.22,.43],U(-.04,.04),[1.05,1.,1.,.89])
    # For the compact fists, flexion rotates the fingertip towards the palm.
    # Thumb-under M and N cover exactly three and two digits respectively;
    # these digits lie in front of the thumb, not behind it.
    if c=='X':
        # A tightly closed base leaves a visible cleft below the index hook.
        # Lower the bent knuckles relative to the extended proximal phalanx.
        for j in range(3):
            pp,rr,zz,*_=h.parts[2+j];pp[1:,1]-=.32;pp[1:,0]+=.10
        hook=h.parts[5][0];hook[1:,1]+=.22;hook[2:,0]-=.16
    if c in 'MN':
        covered=3 if c=='M' else 2
        for j in range(4-covered,4):
            ps,rs,z,*_=h.parts[2+j]; ps[-1,0]+=.14;ps[-1,1]-=.21; z[-1]+=.29
        # The protruding thumb tip is visible in the interdigital space,
        # although most of its shaft is occluded.  Pinky folds further into
        # the palm to expose the three fingers crossing it in an M.
        for j in range(4-covered):
            ps,rs,z,*_=h.parts[2+j];ps[1:,1]-=.30 if c=='M' else .20
            z[1:]-=.19
        if c=='M':
            for j in range(1,4):
                ps,rs,z,*_=h.parts[2+j];ps[-1,0]+=.20;ps[-1,1]+=.12;ps[1:3,0]-=.12
    # Thumb CMC to distal phalanx.  Thumb depth is distinct from long fingers.
    if c in 'LY':
        ang=np.deg2rad(U(14,38) if c=='Y' else U(2,22));p=vec([.64,.90]); v=vec([np.cos(ang),np.sin(ang)])
        h.digit([(.15,.38),p,p+v*.80+vec([0,.22]),p+v*1.76+vec([0,.22])],.34,[-.14,.04,.12,.12],taper=[1.62,1.35,.84,.63])
    elif c=='F':
        h.digit([(.18,.44),(.94,.94),(1.44,1.25),(1.72,1.67),(ux[3]+1.14,uy[3]+.25)],.32,[-.16,.03,.11,.21,.29],taper=[1.60,1.17,.91,.76,.69])
    else:
        tip={'A':(.92,2.64),'E':(-.35,1.42),'S':(-.52,2.17),'M':(-.56,1.98),
             'N':(-.06,2.03),'B':(-.48,1.53),'D':(-.11,2.48),'R':(-.32,2.01),
             'U':(-.38,2.10),'V':(-.42,2.13),'W':(-.54,2.16),'I':(-.15,2.57),
             'K':(.20,2.91),'X':(-.44,2.22)}[c]
        tx,ty=tip;tx+=U(-.09,.09);ty+=U(-.12,.12)
        px=.96 if c=='A' else .83
        mid=(.61*px+.39*tx, .36*1.32+.64*ty+.09)
        ztip=1.02 if c in 'ES' else .73
        if c=='E': mid=(.42,1.24);px=.94
        thumb_mid_z=.41
        if c=='S':
            # Adducted and opposed to the fist. A thumb pointing mostly
            # into the camera is foreshortened; in a frontal view it crosses
            # the other digits. Vary how much of its distal segment is seen.
            cross=U(0,1); px=.87;mid=(.74-.54*cross,1.94+.16*cross)
            tx=.68-1.32*cross;ty=2.24+U(-.12,.16);ztip=1.06;thumb_mid_z=.67+.20*cross
        if c in 'MN': ztip=.25
        h.digit([(.13,.39),(.63,.93),(px,1.44),mid,(tx,ty)],.33,
                [-.13,.07,.22,thumb_mid_z,ztip],taper=[1.59,1.43,1.02,.90,.71]);
    # Three-quarter perspective: the radial side is closer to the camera,
    # especially when the index or the thumb makes the discriminative shape.
    yaw={'R':U(.35,.73),'K':U(.29,.73),'X':U(.38,.74),'D':U(.28,.78),
         'N':U(.37,.83),'M':U(.20,.68),'U':U(.1,.57),'F':U(.28,.77)}.get(c,U(-.07,.38))
    for ps,rs,z,*_ in h.parts:
        xx=ps[:,0].copy();ps[:,0]=np.cos(yaw)*xx+.48*np.sin(yaw)*z
        z+=.67*np.sin(yaw)*xx
    return h

def round_hand(c):
    # The palm is viewed almost edge-on in C and O.  Curved fingers make the
    # upper arc; the opposable thumb is the lower arc, touching only in O.
    h=Hand(); h.arm((-.40,.20),(-1.7,-2),.60)
    h.part([(-.40,.27),(-.34,.80),(-.26,1.32),(-.13,1.90)],
           [.60,.70,.69,.60],[-.27,-.20,-.15,-.14],-.04,.73)
    closed=c=='O'; size=U(.92,1.08);bend=U(-.16,.16)
    for i in range(4):
        # A sheaf of superimposed arcs viewed from the radial side.
        o=(i-3)*.095
        points=[(-.18+o,1.73+o*.40),(-.05+o,2.37+o),(.31+o,2.85+o+bend),
                (.80+o,2.97+o+bend),(1.15+o,2.79+o)]
        if closed:points+=[(1.44+o,2.43+o),(1.48+o,2.26+o)]
        else:points+=[(1.35+o,2.70+o*.8+bend)]
        points=[(x*size,y) for x,y in points]; h.digit(points,.245+o*.1,-.42+i*.18, taper=np.linspace(1.25,.78,len(points)))
    if closed:path=[(-.48,.37),(.10,.77),(.84,1.10),(1.35,1.57),(1.50,2.23)]
    else:path=[(-.48,.37),(.12,.79),(.66,1.06),(1.17,1.19),(1.68,1.55+U(-.18,.18))]
    h.digit(path,.36,[-.17,.03,.13,.21,.23],taper=[1.62,1.37,1.01,.79,.65]);return h

def side(c):
    h=Hand()
    if c in 'GH':
        # Index points right; the other extended digit in H runs parallel.
        h.arm((-.86,.02),(-2.4,-1.8),.57)
        h.part([(-.86,.04),(-.65,.58),(-.42,1.03),(-.30,1.39)],
               [.58,.72,.78,.76],[-.30,-.23,-.20,-.17],-.03,.66)
        num=3 if c=='G' else 2
        for j in range(num):
            y=.35+.39*j; dx=U(-.07,.07)
            h.digit([(-.40,y),(.19,y-.10),(.57+dx,y-.03),(.51+dx,y+.22)],.259,[-.03,.11,.24,.31],taper=[1.25,1.04,1.,.8])
        a=U(.03,.39) if c=='G' else U(-.09,.09)
        y=1.57
        if c=='H':h.digit([(-.55,y-.20),(.30,y-.24),(1.39,y-.22+a),(2.20,y-.22+a*2)],.262,[.04,.08,.10,.12]); y+=.12
        h.digit([(-.60,y),(.35,y+.07),(1.37,y+.10+a),(2.29,y+.10+2*a)],.245,[.02,.10,.17,.20],taper=[1.22,1.06,.90,.76])
        # thumb lies along the folded fingers, or parallel behind index
        h.digit([(-.68,.70),(-.27,1.21),(.15,1.48),(.50,1.53)],.28,[-.04,.13,.20,.21],taper=[1.50,1.12,.91,.75]);return h
    if c=='T':
        # Edge-on fist with the thumb emerging between bent digits.
        h.arm((-.54,-.04),(-1.,-2.),.57)
        h.part([(-.53,.05),(-.42,.59),(-.37,1.24),(-.28,1.57)],[.58,.72,.71,.60],[-.27,-.19,-.14,-.13],-.02,.75)
        for j in range(3):
            h.digit([(-.64,.77+j*.27),(-.10,.78+j*.28),(.22,.87+j*.27),(.15,1.06+j*.27)],.23,[-.06,.06,.18,.24]);
        h.digit([(-.54,.79),(-.26,1.35),(-.19,1.97),(-.24,2.69)],.31,[-.04,.07,.18,.15],taper=[1.4,1.02,.83,.67]);
        h.digit([(-.65,1.65),(-.07,1.86),(.49,1.86),(.94,1.76),(1.35,1.77+U(-.12,.12))],.26,[.0,.12,.21,.25,.27]);return h
    # In the downward K/G configurations, the back of the palm forms an arch
    # from a bent wrist; distal segments point left and/or down, not up.
    if c=='P':
        h.arm((.83,.17),(2.23,-1.55),.60)
        h.part([(.82,.23),(.60,.69),(.37,1.02),(-.12,1.14)],[.60,.72,.75,.69],[-.25,-.22,-.18,-.18],-.02,.78)
        for j in range(2):h.digit([(.44,.87-j*.34),(-.08,.80-j*.29),(-.32,.59-j*.25),(-.11,.46-j*.25)],.26,[-.03,.03,.14,.26])
        a=U(-.15,.24);h.digit([(.24,1.39),(-.55,1.56),(-1.31,1.58+a),(-2.16,1.54+a*2)],.24,[0,.04,.08,.10],taper=[1.24,1.05,.88,.74]);
        h.digit([(-.33,1.18),(-.77,.77),(-1.00,.03),(-1.18,-.79+U(-.18,.18))],.235,[-.04,.03,.11,.14]);
        h.digit([(.59,.50),(.24,.46),(-.28,.71),(-.73,.84)],.32,[-.10,.11,.24,.30],taper=[1.45,1.14,.86,.69]);return h
    h.arm((.65,.12),(1.42,-2),.61)
    h.part([(.65,.20),(.53,.69),(.24,1.02),(-.26,1.27)],[.59,.73,.77,.71],[-.29,-.22,-.16,-.14],-.04,.76)
    for j in range(3):
        h.digit([(.24-j*.14,.91+j*.12),(-.23-j*.14,.62+j*.12),(-.44-j*.16,.37+j*.12),(-.19-j*.15,.25+j*.12)],.24,[-.09,.01,.11,.24])
    h.digit([(-.16,1.42),(-.70,1.21),(-1.13,.64),(-1.48,-.08),(-1.75,-.55)],.24,[-.06,.02,.10,.11,.13]);
    h.digit([(.35,1.25),(-.03,1.44),(-.54,1.21),(-.78,.67),(-.88,.04)],.31,[-.16,.03,.17,.22,.24],taper=[1.26,1.04,.91,.80,.65]);return h

def draw(c,ambience):
    h=round_hand(c) if c in 'CO' else side(c) if c in 'GHPTQ' else front(c)
    # Metric -> image plane: vary hand proportions, aspect from foreshortening,
    # wrist roll, finger size relative to crop, illumination and partial crop.
    sx=U(.84,1.08); sy=U(.95,1.05)
    if c in 'DRKUIXMN':sx*=U(.84,1.06)
    if c=='F':sx*=U(.90,1.13)
    if c=='I':sx*=U(.95,1.16)
    if c in 'CDOX':sx*=U(1.05,1.34)
    if c in 'GHQ': sy*=U(1.12,1.38)
    roll=U(-.23,.21)
    if c in 'AMNSIY':roll+=.095  # natural radial tilt around wrist
    if c=='K':roll-=.20
    if c in 'MNSX':roll+=U(.03,.30)
    if c in 'GH':roll=U(-.22,.18)
    cs=np.cos(roll);sn=np.sin(roll)
    mat=np.array([[sx*cs,-sx*sn],[sy*sn,sy*cs]],dtype='f')
    if c in 'GHPTQ':
        # In profile, knuckles are stacked around the edge of the palm; small
        # changes in forearm pronation expose different amounts of each digit.
        mat[0,1]+=U(-.16,.16); mat[1,0]+=U(-.17,.17)
    # Front-facing dorsum or palm, rounded so that even a similarly coloured
    # background produces boundary edges from the surface normal.
    lx=U(-.90,-.25);ly=U(.07,.57);lz=U(.58,.96)
    bg=U(.47,.86);skin=U(.37,.61); light=U(.34,.54)
    if R.random()<.12:lx*=-1
    dims=[]
    for ps,rs,ds,tone,flat,fr in h.parts:
        if fr:
            pp=ps@mat;dims.append([*(pp-rs[:,None]).min(0),*(pp+rs[:,None]).max(0)])
    dim=vec(dims);lo=dim[:,:2].min(0);hi=dim[:,2:].max(0);extent=hi-lo
    sc=U(21.,32.0)/max(extent);cy=U(-2.1,2.8)
    cen=(lo+hi)*.5; transform=mat*sc
    # centre the unoccluded hand with a small allowance for the entering arm
    centerpix=vec([U(-2.85,2.85),cy])-cen*sc
    zbuf=np.full(XI.shape,-40.,dtype='f'); img=np.full(XI.shape,bg,dtype='f')
    for ps,rs,ds,tone,flat,fr in h.parts:
        pp=ps@transform+centerpix; rad=rs*sc*(sx+sy)*.5
        # Shade the front (visible) surface of tapered ellipsoidal capsules.
        for j in range(len(ps)-1):
            a,b=pp[j:j+2]; d=b-a; l2=max(.021,float(d@d));r1,r2=rad[j:j+2]; rmax=max(r1,r2)+.6
            x0=max(0,int(np.floor(min(a[0],b[0])+GRID*.5-.5-rmax)));x1=min(GRID,int(np.ceil(max(a[0],b[0])+GRID*.5+.5+rmax)))
            y0=max(0,int(np.floor(GRID*.5-.5-max(a[1],b[1])-rmax)));y1=min(GRID,int(np.ceil(GRID*.5+.5-min(a[1],b[1])+rmax)))
            if x0>=x1 or y0>=y1:continue
            s=np.s_[y0:y1,x0:x1];x=XI[s]-a[0];y=YI[s]-a[1]; t=np.clip((x*d[0]+y*d[1])/l2,0,1)
            rx=x-d[0]*t; ry=y-d[1]*t;r=r1+t*(r2-r1);dist=np.sqrt(rx*rx+ry*ry+1e-9);nr=dist/r
            nz=np.sqrt(np.maximum(1-nr*nr,0)); depth=(ds[j]+t*(ds[j+1]-ds[j]))*sc+nz*r*flat
            good=(nr<=1)&(depth>zbuf[s]);
            nx=rx/np.maximum(r,dist+.001); ny=ry/np.maximum(r,dist+.001)
            v=skin+tone+light*(lx*nx+ly*ny+lz*nz-.51)-.10*(1-nz)**2
            zbuf[s]=np.where(good,depth,zbuf[s]);img[s]=np.where(good,v,img[s])
    # Mild contact shadows cast by raised fingers onto the palm (height-field
    # approximation).  No cast shadow on distant background.
    behind=np.roll(zbuf, (1,round(-lx*2.0)),axis=(0,1))-zbuf
    img-=.14*np.clip((behind-.22*sc)/(.4*sc),0,1)*(zbuf>-39)
    # A small antialiasing kernel reduces unrealistic subpixel stair-steps.
    img=.5*img+.125*(np.roll(img,1,0)+np.roll(img,-1,0)+np.roll(img,1,1)+np.roll(img,-1,1))
    silhouette=(zbuf>-39).astype('f')
    # Composite the same shape over a bank of analytically specified changes
    # of background exposure. These five settings are never sampled from photos.
    img+=ambience*(1-silhouette)
    return img, silhouette

# A quadrature-like set of rectified derivatives. Use opposite polarities
# separately but allow model evidence of either polarity at a silhouette.
NANG=4
angles=np.arange(NANG)*np.pi/NANG
yy,xx=np.mgrid[-2:3,-2:3]; g=np.exp(-(xx**2+yy**2)/(2*.90**2))
der=np.array([(np.cos(a)*xx+np.sin(a)*yy)*g for a in angles],dtype='f');der/=np.abs(der).sum(axis=(1,2),keepdims=True)*.5
der=np.concatenate([der,-der],0)
center=np.exp(-(xx**2+yy**2)/(2*.60**2)); surround=np.exp(-(xx**2+yy**2)/(2*1.55**2)); blob=center/center.sum()-surround/surround.sum();blob/=np.maximum(blob,0).sum()
der=np.concatenate([der,blob[None],-blob[None]],0).astype('f')
F=len(der)

def correlate(a,filters):
    return np.lib.stride_tricks.sliding_window_view(a,filters.shape[-2:],axis=(-2,-1)).reshape(-1,filters.shape[-1]**2)@filters.reshape(len(filters),-1).T

def features(im,mask):
    ims=np.var(im[6:-6,6:-6])+.0001
    # Compute a gentle local contrast-retinex field at the same locations
    # as the edge channels, to capture the valleys under bent fingers.
    hann=np.hanning(11)[1:-1].astype('f'); hann/=hann.sum()
    blank=np.pad(im,4,mode='reflect');smooth=np.lib.stride_tricks.sliding_window_view(blank,9,axis=1)@hann
    smooth=np.lib.stride_tricks.sliding_window_view(smooth,9,axis=0)@hann
    local=((im-smooth)/np.sqrt(ims))[2:-2,2:-2];D=correlate(im,der).reshape(GRID-4,GRID-4,F).transpose(2,0,1)/np.sqrt(ims)
    f=np.maximum(np.concatenate([D,local[None],-local[None]],0),0);f-=np.maximum(f-.92,0)*.82
    # Mix a small amount of reversed polarity around a silhouette, whose
    # contrast may be reversed by a background change.
    p=np.lib.stride_tricks.sliding_window_view(mask,(5,5));border=(p.max((2,3))-p.min((2,3)))
    mix=.10+.20*border
    f[:8]=(1.-mix*.5)*f[:8]+mix*.5*np.roll(f[:8],NANG,axis=0)
    g=np.lib.stride_tricks.sliding_window_view(f,(3,3),axis=(-2,-1)).max(axis=(-2,-1))
    f=.50*f[:,1:-1,1:-1]+.50*g
    return np.lib.stride_tricks.sliding_window_view(f,(2,2),axis=(-2,-1)).mean((-2,-1))

def build(out):
    out.mkdir(exist_ok=True,parents=True); layers=[];w={}
    def layer(name,op,inp,**kw):layers.append(dict(name=name,op=op,inputs=[inp] if isinstance(inp,str) else inp,**kw));return name
    def conv(name,inp,fo,b):
        w[name+'.W']=fo.astype('f',copy=False);w[name+'.b']=vec(b);return layer(name,'conv',inp,out=len(fo),kernel=list(fo.shape[-2:]),padding=0,stride=1)
    def dense(name,inp,fo,b):
        w[name+'.W']=fo.astype('f',copy=False);w[name+'.b']=vec(b);return layer(name,'dense',inp,out=len(fo))
    def pool(n,op,inp,sz,st):return layer(n,op,inp,size=sz,stride=st)
    x=layer('exposure','norm','image')
    # Exact reflect padding implemented in the permitted linear operations.
    # No invented dark border and no need to discard fingertips at the top.
    pix=np.pad(np.arange(28**2).reshape(28,28),5,mode='reflect').ravel()
    pad=np.zeros((len(pix),28**2),dtype='f');pad[np.arange(len(pix)),pix]=1
    x=dense('reflect_at_frame',x,pad,np.zeros(len(pix)));x=layer('expanded_frame','reshape',x,shape=[1,38,38]);nf=F+2
    kernel=np.zeros((nf,1,9,9),'f');kernel[:F,0,2:-2,2:-2]=der;hann=np.hanning(11)[1:-1];hann/=hann.sum();kernel[F,0]=-np.outer(hann,hann);kernel[F,0,4,4]+=1;kernel[F+1]=-kernel[F]
    x=conv('compass_spot_and_shade',x,kernel,np.zeros(nf));x=layer('edge','relu',x)
    t=conv('strong',x,np.eye(nf,dtype='f')[:,:,None,None],np.full(nf,-.92));t=layer('excess','relu',t);t=conv('attenuate',t,-.82*np.eye(nf,dtype='f')[:,:,None,None],np.zeros(nf));x=layer('bounded','add',[x,t]);t=pool('nearby','maxpool',x,[3,3],[1,1]);ident=np.eye(nf,dtype='f');center=np.zeros((nf,nf,3,3),'f');center[:,:,1,1]=ident;x=conv('at_anchor',x,center,np.zeros(nf));x=layer('local_evidence','add',[x,t]);x=pool('cells','avgpool',x,[2,2],[2,2]);
    # Share the same matching kernel at each of nine relative translations.
    # Reflecting the cell field first keeps the receptive field consistent.
    wsize=nf*14*14;parts=2
    order=np.pad(np.arange(wsize).reshape(nf,14,14),((0,0),(1,1),(1,1)),mode='reflect').ravel();perm=np.zeros((len(order),wsize),'f');perm[np.arange(len(order)),order]=1
    x=dense('reflect_cells',x,perm,np.zeros(len(order)));x=layer('larger_cell_field','reshape',x,shape=[nf,16,16]);
    fil=np.empty((24,VIEWS,parts,wsize),'f')
    for ci,c in enumerate(ABC):
        for k in range(VIEWS):
            im,mask=draw(c,[-.24,-.09,0.,.09,.17][k%5]); f=features(im,mask)
            # suppress normal shared part of orientations, retain some total
            # activity evidence. Penalize activity outside the predicted hand,
            # and reduce dependence on wrist position at the base of the frame.
            oy=(GRID-28)//2-3;ox=oy
            v=f[:,oy:oy+28:2,ox:ox+28:2].copy()
            # An oriented edge must win against competing orientations at
            # the same location. A center-surround event must have the
            # correct polarity (bright phalanx vs dark inter-finger gap).
            wv=v.copy(); wv[:8]-=.72*v[:8].mean(0);wv[8:10]-=.31*v[8:10].mean(0);wv[10:] *= 1.1
            wv-=wv.mean(axis=(1,2),keepdims=True)*.95
            focus=np.linspace(1.15,.84,14,dtype='f') if c not in 'PQ' else np.ones(14,'f')
            wv*=focus[None,:,None];wv[:,-1]*=.78;wv[:,[0,-1]]*=.93
            wv/=np.sqrt((wv**2).sum()+.000001)
            upper=np.clip((9-np.arange(14))/5.,.15,.85)
            fil[ci,k,0]=(wv*upper[None,:,None]).ravel()
            fil[ci,k,1]=(wv*(1-upper[None,:,None])).ravel()
    # Score small independent pose groups so no maxpool needs to make a
    # temporary copy of the full 9-position activation bank at inference.
    scores=[];groups=8
    for g in range(groups):
        s=str(g+1);n=VIEWS//groups;fg=fil[:,g*n:(g+1)*n].reshape(-1,nf,14,14)
        t=conv('pose_agreement_'+s,x,fg,np.zeros(len(fg),dtype='f'))
        t=pool('align_parts_'+s,'maxpool',t,[3,3],[3,3]);t=layer('letter_and_view_'+s,'reshape',t,shape=[24,n,parts]);t=pool('part_agreement_'+s,'avgpool',t,[1,parts],[1,parts]);scores.append(pool('best_view_'+s,'maxpool',t,[n,1],[n,1]))
    t=layer('view_consensus','add',scores);
    layer('letter_scores','reshape',t,shape=[24]);
    (out/'network.json').write_text(json.dumps(dict(layers=layers),indent=1)+'\n');np.savez(out/'weights.npz',**w)

if __name__=='__main__':build(Path(sys.argv[1]))
