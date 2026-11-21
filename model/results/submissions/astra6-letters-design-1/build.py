"""Data-free ASL recognizer made from analytic filters and anatomical diagrams.

No inputs other than the destination directory are read.  Geometry below is
specified in units of palm width, not in image coordinates.  All hypotheses
are retained; there is no training, fitting, or data-dependent selection.
"""
import json
import sys
from pathlib import Path
import numpy as np

LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'

class Hand:
    def __init__(self):
        self.parts = []
        self.lines = []
        self.forearm_parts = []
        self.holes = []
    def hole(self, center, radii):
        self.holes.append((tuple(center),tuple(radii)))
    def capsule(self, a, b, r):
        self.parts.append(('capsule', tuple(a), tuple(b), float(r)))
    def ellipse(self, center, radii, angle=0):
        self.parts.append(('ellipse', tuple(center), tuple(radii), float(angle)))
    def stroke(self, points, strength=.38):
        for a, b in zip(points[:-1], points[1:]):
            self.lines.append((tuple(a), tuple(b), float(strength)))
    def finger(self, points, radius=.115, taper=.85, seams=True):
        for j, (a, b) in enumerate(zip(points[:-1], points[1:])):
            self.capsule(a, b, radius * (1-(1-taper)*j/max(1,len(points)-2)))
        if seams and len(points) > 2:
            a, p, b = map(np.asarray, (points[0],points[1],points[-1]))
            tangent = b-a
            tangent /= np.sqrt(np.sum(tangent*tangent)) + 1e-10
            normal = np.array([-tangent[1], tangent[0]])
            self.stroke([p-normal*radius*.65, p+normal*radius*.65], .24)
    def wrist(self, start=(0,.55), end=(0,1.35), width=.30):
        # The forearm continues outside a hand-centered crop. Its arbitrarily
        # drawn endpoint must not determine the hand's size or centering.
        self.forearm_parts.append(len(self.parts))
        direction=np.asarray(end)-np.asarray(start)
        direction=direction/(np.sqrt(np.sum(direction*direction))+1e-12)
        self.capsule(start,np.asarray(start)+direction*5.0,width)


def anatomical_hand(letter, variant):
    """Small set of front/oblique hand diagrams, from ASL articulation rules.

    In the canonical palm view, little finger is left and thumb is right.
    Ellipses represent palm/curl masses; capsules are phalanges.  Lines mark
    visible interphalangeal folds, curled-finger divisions, and thumb creases.
    The two variants change joint flexion and finger spread, not the label.
    """
    h = Hand()
    spread = .90 if variant == 0 else 1.10
    flex = -.04 if variant == 0 else .04
    xfinger = [-.39, -.14, .12, .36]
    lengths = [.77, 1.03, 1.13, 1.00]
    def palm():
        h.ellipse((0,.20),(.48,.58))
        h.wrist()
    def curled(fingers=(0,1,2,3), low=.03):
        for i in fingers:
            x = xfinger[i]
            top = -.44 - .10*np.sin(np.pi*(i+.5)/4)
            h.finger([(x,top),(x-.015, -.24),(x-.025,low)], .132, .95, False)
            h.stroke([(x-.09, -.10),(x, low+.045),(x+.085, -.10)], .42)
        for i in list(fingers)[:-1]:
            if i+1 in fingers:
                x = (xfinger[i]+xfinger[i+1])/2
                h.stroke([(x,-.43),(x,-.16)], .34)
    def extended(i, tip=None, bend=0):
        x = xfinger[i]
        if tip is None:
            tip=(x*spread,-.28-lengths[i])
        mid=((x+tip[0])/2+bend,(-.25+tip[1])/2)
        h.finger([(x,-.24),mid,tip], .107 if i==0 else .116)
    def thumb_cross(high=False):
        end=(-.19,-.14 if high else .03)
        h.finger([(.43,.37),(.22,.17),end], .142, .9, False)
        h.stroke([(.36,.20),(.18,.02),(-.14,-.12 if high else -.08)], .53)

    if letter in 'CO':
        # Profile view: a curved four-finger block opposing the thumb.
        h.ellipse((-.32,.16),(.29,.50),-.22)
        h.wrist((-.40,.46),(-.53,1.23),.27)
        upper=[(-.48,-.12),(-.40,-.51),(-.13,-.73),(.24,-.72),(.57,-.55)]
        if letter == 'O':
            upper += [(.66,-.27),(.54,-.02)]
            thumb=[(-.29,.41),(.08,.33),(.44,.12),(.54,-.02)]
        else:
            upper += [(.64,-.39+flex)]
            thumb=[(-.29,.41),(.10,.25),(.56,.08-flex)]
        h.finger(upper,.175,.88,False)
        h.finger(thumb,.135,.92,False)
        h.stroke([(-.38,-.23),(-.12,-.49),(.26,-.48),(.50,-.36)],.23)
        h.stroke([(-.26,.23),(-.06,.09),(.28,.02)],.29)
    elif letter in 'GH':
        # Sideways index; H additionally extends the middle finger.
        h.ellipse((-.43,.19),(.38,.43),-.25)
        h.wrist((-.55,.45),(-.67,1.13),.28)
        for j in range(3 if letter=='G' else 2):
            y=.08+j*.20
            h.ellipse((-.20,y),(.24,.14),.15)
            h.stroke([(-.27,y-.08),(-.03,y-.04)],.38)
        h.finger([(-.35,-.16),(.29,-.25),(1.02,-.27+flex)],.117)
        if letter=='H':
            h.finger([(-.31,.08),(.30,.03),(1.00,.02-flex)],.122)
            h.finger([(-.60,.26),(-.27,.09),(-.08,-.08)],.15)
        else:
            h.finger([(-.58,.25),(-.12,-.02),(.57,-.04-flex)],.131)
        h.stroke([(-.53,.37),(-.27,.15),(-.08,.17)],.35)
    elif letter == 'P':
        # K articulation with the middle finger pointing down.
        h.ellipse((.24,.04),(.45,.32),.30)
        h.wrist((.53,.23),(.72,1.04),.28)
        h.finger([(.22,-.17),(-.40,-.33),(-1.03,-.40+flex)],.12)
        h.finger([(.04,.0),(-.32,.31),(-.53,.73)],.13)
        h.finger([(.46,.17),(.05,.28),(-.24,.43)],.14)
        h.ellipse((.33,.25),(.27,.22),.5)
        h.stroke([(.45,.01),(.17,.04),(-.02,.25)],.4)
    elif letter == 'Q':
        # G articulation directed down, with a bent wrist.
        h.ellipse((-.05,-.26),(.39,.36),-.25)
        h.wrist((-.34,-.03),(-.50,1.04),.27)
        h.finger([(.22,-.32),(.43,.15),(.57,.81)],.122)
        h.finger([(-.18,-.19),(-.23,.22),(-.39,.58+flex)],.136)
        h.ellipse((-.08,-.18),(.26,.25),.2)
        h.stroke([(-.22,-.35),(-.02,-.24),(.17,-.13)],.4)
    else:
        palm()
        if letter == 'A':
            curled()
            h.finger([(.39,.42),(.52,-.03),(.52,-.79+flex)],.137,.90)
            h.stroke([(.37,.28),(.37,-.23)],.5)
        elif letter == 'B':
            for i in range(4): extended(i, (xfinger[i]*.90,-.28-lengths[i]))
            thumb_cross()
            for i in range(3):
                x=(xfinger[i]+xfinger[i+1])*.45
                h.stroke([(x,-.33),(x,-.91)],.4)
        elif letter == 'D':
            extended(3, (.36,-1.37))
            for i in range(3):
                h.finger([(xfinger[i],-.30),(xfinger[i]-.06,-.03),(0,.12)],.12)
            h.finger([(.43,.36),(.26,.12),(0,.12)],.14)
            h.hole((-.035,-.015),(.13,.105))
            h.stroke([(-.26,-.10),(-.11,.04),(.10,.07),(.24,-.02)],.54)
        elif letter == 'E':
            curled(low=.16)
            h.finger([(.43,.37),(.05,.34),(-.31,.29)],.135,.9,False)
            h.stroke([(-.36,.16),(0,.22),(.30,.15)],.55)
        elif letter == 'F':
            for i in (0,1,2):
                extended(i,(xfinger[i]*1.24,-.25-lengths[i]))
            h.finger([(.35,-.31),(.60,-.21),(.64,.05),(.41,.22)],.113,.92,False)
            h.finger([(.34,.44),(.23,.21),(.41,.22)],.136,.90,False)
            h.hole((.43,.015),(.12,.115))
            h.stroke([(.15,.20),(.10,.02),(.25,-.10)],.38)
        elif letter == 'I':
            curled((1,2,3))
            extended(0,(-.48*spread,-1.15))
            thumb_cross()
        elif letter == 'K':
            curled((0,1))
            extended(3,(.47,-1.35))
            extended(2,(-.49*spread,-.76+flex))
            h.finger([(.43,.35),(.22,-.09),(.09,-.63)],.145,.85)
            h.stroke([(.33,.14),(.09,-.25)],.5)
        elif letter == 'L':
            curled((0,1,2))
            extended(3,(.34,-1.36))
            h.finger([(.36,.39),(.80,.15),(1.21,.03+flex)],.143,.83)
            h.stroke([(.30,.27),(.46,.05)],.38)
        elif letter in 'MN':
            # Tucked thumb; different numbers of fingers cover the thumb.
            covered = 3 if letter=='M' else 2
            h.finger([(.43,.29),(.10,.02),(-.12 if covered==2 else -.33,.06)],.145,.9,False)
            curled(low=.18 if letter=='M' else .07)
            h.stroke([(-.38,.30),(-.10,.19),(.17,.22)],.38)
            h.stroke([(-.22 if letter=='M' else .02,.15),(-.24 if letter=='M' else .0,.37)],.48)
        elif letter == 'R':
            curled((0,1))
            extended(3,(-.03,-1.26),-.03)
            extended(2,(.35,-1.40),.02)
            thumb_cross()
            h.stroke([(.12,-.53),(.24,-.88)],.38)
        elif letter == 'S':
            curled()
            thumb_cross(high=True)
        elif letter == 'T':
            curled((0,1,2,3),low=.10)
            h.finger([(.43,.35),(.23,.03),(.11,-.73)],.145,.85)
            h.stroke([(.03,-.46),(.03,-.08),(.23,.10)],.46)
        elif letter in 'UVW':
            folded=(0,) if letter=='W' else (0,1)
            curled(folded)
            if letter=='U':
                extended(2,(.10,-1.43)); extended(3,(.35,-1.31))
                h.stroke([(.23,-.31),(.23,-1.15)],.43)
            else:
                extended(2,(-.20*spread,-1.39))
                extended(3,(.64*spread,-1.22))
                if letter=='W': extended(1,(-.62*spread,-1.13))
            thumb_cross()
        elif letter == 'X':
            curled((0,1,2))
            h.finger([(.35,-.24),(.30,-.79),(.60,-.87),(.70,-.67+flex)],.119,.87)
            thumb_cross()
            h.stroke([(.34,-.56),(.51,-.62)],.4)
        elif letter == 'Y':
            curled((1,2,3))
            extended(0,(-.67*spread,-1.08))
            h.finger([(.38,.31),(.81,-.01),(1.12,-.36-flex)],.138,.83)
            h.stroke([(.29,.19),(.39,-.03)],.33)
        else:
            raise ValueError(letter)
        # Main palm crease, deliberately weak relative to outline/fingers.
        h.stroke([(-.31,.29),(-.06,.37),(.19,.30)],.15)
    if letter=='R':
        # Interlaced fingers touch rather than leave a V-shaped gap.
        h.capsule((.21,-.77),(.20,-1.22),.10)
        h.stroke([(.06,-.90),(.29,-1.11)],.55)
    if letter in 'BST':
        # Front/oblique views of the right hand's palm face the observer.
        # The canonical construction above uses the back-view digit order.
        h.parts=[(kind,(-p[0],p[1]),(-q[0],q[1]) if kind=='capsule' else q,
                  r if kind=='capsule' else -r) for kind,p,q,r in h.parts]
        h.lines=[((-p[0],p[1]),(-q[0],q[1]),strength) for p,q,strength in h.lines]
        h.holes=[((-p[0],p[1]),q) for p,q in h.holes]
    return h


def distance_segment(x,y,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    t=np.clip(((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy+1e-12),0,1)
    return np.sqrt((x-a[0]-t*dx)**2+(y-a[1]-t*dy)**2)


def bounds(h):
    bb=[]
    for i,(kind,a,b,r) in enumerate(h.parts):
        if i in h.forearm_parts:
            continue
        if kind=='capsule':
            bb.extend([(a[0]-r,a[1]-r),(a[0]+r,a[1]+r),(b[0]-r,b[1]-r),(b[0]+r,b[1]+r)])
        else:
            c,s=np.cos(r),np.sin(r)
            rx=np.sqrt((b[0]*c)**2+(b[1]*s)**2)
            ry=np.sqrt((b[0]*s)**2+(b[1]*c)**2)
            bb.extend([(a[0]-rx,a[1]-ry),(a[0]+rx,a[1]+ry)])
    bb=np.array(bb)
    return bb.min(0), bb.max(0)

# Continuous evaluation grids: four samples per eventual 2x2 descriptor cell.
Y,X=np.mgrid[0:28,0:28].astype(np.float64)
ANGLES=np.arange(4)*np.pi/4


def diagram_features(h, scale, angle, mirror, shift, aspect):
    lo,hi=bounds(h)
    center=(lo+hi)/2
    unit=scale/max(hi-lo)
    a=np.deg2rad(angle); c,s=np.cos(a),np.sin(a)
    u=(X-13.5-shift[0])/unit; v=(Y-13.5-shift[1])/unit
    x=(c*u+s*v)*mirror/aspect+center[0]
    y=(-s*u+c*v)+center[1]
    d=np.full(X.shape,100.0)
    for kind,p,q,r in h.parts:
        if kind=='capsule':
            dd=distance_segment(x,y,p,q)-r
        else:
            ca,sa=np.cos(r),np.sin(r)
            ex=ca*(x-p[0])+sa*(y-p[1]); ey=-sa*(x-p[0])+ca*(y-p[1])
            # First-order signed distance to ellipse, exact at the boundary.
            z=np.sqrt((ex/q[0])**2+(ey/q[1])**2+1e-12)
            den=np.sqrt((ex/q[0]**2)**2+(ey/q[1]**2)**2+1e-12)/z
            dd=(z-1)/np.maximum(den,1/max(q))
        d=np.minimum(d,dd)
    for p,q in h.holes:
        ex=x-p[0]; ey=y-p[1]
        z=np.sqrt((ex/q[0])**2+(ey/q[1])**2+1e-12)
        den=np.sqrt((ex/q[0]**2)**2+(ey/q[1]**2)**2+1e-12)/z
        dd=(z-1)/np.maximum(den,1/max(q))
        d=np.maximum(d,-dd)
    dp=d*unit
    gy,gx=np.gradient(dp)
    gn=np.sqrt(gx*gx+gy*gy+1e-9)
    gx/=gn; gy/=gn
    edge=np.exp(-.5*(dp/1.0)**2)
    feat=[]
    for theta in ANGLES:
        co,si=np.cos(theta),np.sin(theta)
        directional=np.maximum(np.abs(gx*co+gy*si)-.65*np.abs(-gx*si+gy*co),0)
        f=edge*directional
        for p,q,strength in h.lines:
            dist=distance_segment(x,y,p,q)*unit
            tx,ty=q[0]-p[0],q[1]-p[1]
            ln=np.hypot(tx,ty)+1e-12
            nx,ny=-ty/ln*mirror/aspect,tx/ln
            nn=np.hypot(nx,ny)+1e-12
            nx,ny=nx/nn,ny/nn
            nx,ny=c*nx-s*ny,s*nx+c*ny
            direct=max(abs(nx*co+ny*si)-.65*abs(-nx*si+ny*co),0)
            f+=strength*direct*np.exp(-.5*(dist/.85)**2)
        # Only centers at distance >=2 from the image boundary are observed
        # by the valid 5x5 derivative bank.
        f=f[2:26,2:26].reshape(12,2,12,2).mean(axis=(1,3))
        feat.append(f)
    feat=np.array(feat)
    feat=feat-feat.mean()
    feat=feat/(np.sqrt(np.sum(feat*feat))+1e-12)/np.sqrt(feat.size)
    # A weak foreground-support cue.  Remove constant and planar illumination
    # analytically on the fixed grid, not by regression on any photographs.
    mask=1/(1+np.exp(np.clip(dp/.70,-40,40)))
    mask=mask.reshape(14,2,14,2).mean(axis=(1,3))
    yy,xx=np.mgrid[0:14,0:14].astype(float); xx-=6.5; yy-=6.5
    mask-=mask.mean()
    mask-=np.sum(mask*xx)/np.sum(xx*xx)*xx
    mask-=np.sum(mask*yy)/np.sum(yy*yy)*yy
    mask/=np.sqrt(np.sum(mask*mask))+1e-12
    mask*=.22/14
    return feat.astype(np.float32).ravel(),mask.astype(np.float32).ravel()


class Network:
    def __init__(self): self.layers=[]; self.weights={}
    def op(self,name,op,inputs,**settings):
        self.layers.append(dict(name=name,op=op,inputs=[inputs] if isinstance(inputs,str) else inputs,**settings))
        return name
    def conv(self,name,source,w,b=None,padding=0):
        w=np.asarray(w,dtype=np.float32)
        self.weights[name+'.W']=w
        self.weights[name+'.b']=np.zeros(w.shape[0],np.float32) if b is None else np.asarray(b,np.float32)
        return self.op(name,'conv',source,out=w.shape[0],kernel=[w.shape[2],w.shape[3]],stride=1,padding=padding)
    def dense(self,name,source,w,b=None):
        w=np.asarray(w,dtype=np.float32)
        self.weights[name+'.W']=w
        self.weights[name+'.b']=np.zeros(w.shape[0],np.float32) if b is None else np.asarray(b,np.float32)
        return self.op(name,'dense',source,out=w.shape[0])


def build(destination):
    n=Network()
    n.op('contrast','norm','image')
    yy,xx=np.mgrid[-2:3,-2:3].astype(float)
    g=np.exp(-(xx*xx+yy*yy)/(2*.85**2))
    filters=[]
    for theta in ANGLES:
        f=(xx*np.cos(theta)+yy*np.sin(theta))*g
        f/=np.sum(xx*xx*g)
        filters.extend([f,-f])
    n.conv('derivatives','contrast',np.array(filters)[:,None])
    n.op('signed_edges','relu','derivatives')
    w=np.zeros((4,8,1,1))
    for i in range(4):
        w[i,2*i:2*i+2]=1
        j=(i+2)%4
        w[i,2*j:2*j+2]=-.65
    n.conv('orientation_competition','signed_edges',w,[-.025]*4)
    n.op('oriented_edges','relu','orientation_competition')
    w=np.zeros((8,4,1,1)); b=np.zeros(8)
    for i in range(4):
        w[2*i:2*i+2,i]=1; b[2*i+1]=-.40
    n.conv('edge_cap_components','oriented_edges',w,b)
    n.op('edge_cap_relu','relu','edge_cap_components')
    w=np.zeros((4,8,1,1))
    for i in range(4): w[i,2*i]=1; w[i,2*i+1]=-1
    n.conv('bounded_edges','edge_cap_relu',w)
    n.op('edge_cells','avgpool','bounded_edges',size=[2,2],stride=[2,2])
    n.op('edge_plane','reshape','edge_cells',shape=[1,48,12])
    n.op('edge_normalization','norm','edge_plane')
    n.op('intensity_cells','avgpool','contrast',size=[2,2],stride=[2,2])

    # Exhaustive, data-independent nuisance grid. No hypothesis is selected or
    # adjusted using data. Scale is the diagram's bounding-square side length.
    scales=(21.,25.,29.)
    angles=(-30.,-15.,0.,15.,30.)
    aspects=(.70,1.0)
    shifts=((-2.,-2.),(0.,-2.),(2.,-2.),(-2.,0.),(0.,0.),(2.,0.),(-2.,2.),(0.,2.),(2.,2.))
    mirrors=(1,-1)
    per_class=2*len(scales)*len(angles)*len(mirrors)*len(shifts)*len(aspects)
    count=24*per_class
    edge_weights=np.empty((count,576),np.float32)
    mask_weights=np.empty((count,196),np.float32)
    pose_prior=np.empty(count,np.float32)
    k=0
    for letter in LETTERS:
        for variant in range(2):
            hand=anatomical_hand(letter,variant)
            for scale in scales:
                for angle in angles:
                    for mirror in mirrors:
                        for aspect in aspects:
                            for shift in shifts:
                                edge_weights[k],mask_weights[k]=diagram_features(hand,scale,angle,mirror,shift,aspect)
                                # A fixed prior, not an estimated class frequency:
                                # retain the opposite handedness, mildly disfavor it.
                                pose_prior[k]=-.02 if mirror==-1 else 0.
                                k+=1
        print('Built geometric hypotheses for',letter,flush=True)
    n.dense('outline_hypotheses','edge_normalization',edge_weights,pose_prior)
    n.dense('foreground_hypotheses','intensity_cells',mask_weights)
    n.op('foreground_plane','reshape','foreground_hypotheses',shape=[1,count,1])
    n.conv('foreground_polarities','foreground_plane',np.array([1.,-1.]).reshape(2,1,1,1))
    n.op('foreground_polarity_relu','relu','foreground_polarities')
    n.conv('foreground_absolute','foreground_polarity_relu',np.ones((1,2,1,1)))
    n.op('foreground_scores','reshape','foreground_absolute',shape=[count])
    n.op('pose_scores','add',['outline_hypotheses','foreground_scores'])
    n.op('class_pose_grid','reshape','pose_scores',shape=[24,per_class,1])
    n.op('best_anatomical_pose','maxpool','class_pose_grid',size=[per_class,1],stride=[per_class,1])
    n.op('class_scores','reshape','best_anatomical_pose',shape=[24])
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'network.json').write_text(json.dumps({'layers':n.layers},indent=2)+'\n')
    np.savez(destination/'weights.npz',**n.weights)
    print('Total weights:',sum(v.size for v in n.weights.values()),flush=True)

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('usage: python3 build.py OUTPUT_DIRECTORY')
    build(sys.argv[1])
