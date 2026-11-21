"""Data-free ASL classifier. All geometry below is a manually specified hand model.

No input files are opened. Model coordinates describe generic anatomy, not any
photograph. A fixed seed supplies anatomical variation, never a fitted value.
"""
import sys
import json
from pathlib import Path
import numpy as np

LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'


class Hand:
    def __init__(self):
        self.parts = []
        self.lines = []

    def ellipse(self, x, y, rx, ry, angle=0):
        self.parts.append(('ellipse', (x, y, rx, ry, angle)))

    def bone(self, a, b, r):
        self.parts.append(('bone', (*a, *b, r)))

    def digit(self, points, radius, crease=True):
        for k, (a, b) in enumerate(zip(points[:-1], points[1:])):
            r = radius * (1 - .09*k)
            self.bone(a, b, r)
            # A weak, analytically placed cylinder-side shadow describes finger
            # relief even when two adjacent fingers share one silhouette.
            av,bv=np.array(a,float),np.array(b,float)
            v=bv-av
            n=np.array([-v[1],v[0]])/(np.linalg.norm(v)+1e-9)
            self.line(av+.68*r*n,bv+.68*r*n,.021,.16)
        if crease:
            for k in range(1, len(points)-1):
                p = np.array(points[k], float)
                v = np.array(points[k+1], float)-np.array(points[k-1], float)
                v = np.array([-v[1], v[0]]) / (np.linalg.norm(v)+1e-9)
                self.line(p-.65*radius*v, p+.65*radius*v, .022, .30)

    def line(self, a, b, r=.025, strength=.45):
        self.lines.append((*a, *b, r, strength))


def make_hand(letter, variant):
    """Generic right-hand palm view; x goes toward thumb, y toward fingertips."""
    h = Hand()
    rng = np.random.default_rng(8100 + variant)
    width = 1.0 + rng.uniform(-.08,.08)
    length = 1.0 + rng.uniform(-.07,.07)
    # Three generic viewpoints and four degrees of finger separation/bulk.
    shape_variant = variant % 4
    view_variant = (variant % 12) // 4
    h.lighting = variant // 12
    h.light_side = 1 if shape_variant % 2 == 0 else -1
    spread = [0, -.08, .08, .03][shape_variant]
    thick = [.090,.105,.080,.095][shape_variant]
    # A wrist is part of every hand, but its direction depends on the sign.
    wrist_end = (-.08,-1.10)
    if letter in 'CO': wrist_end = (-.35,-1.02)
    if letter in 'GH': wrist_end = (-.83,-.65)
    if letter in 'PQ': wrist_end = (.42,-.94)
    h.bone(wrist_end, (0,-.35), .24)
    h.ellipse(0,-.12,.40,.43)
    bases=[(-.29,.10),(-.10,.18),(.105,.20),(.29,.13)]
    lengths=[.58,.78,.88,.77]

    def extended(i, angle=0, curl=0):
        x,y = bases[i]
        a = angle + spread*(i-1.5)
        l = lengths[i]*length
        # Fingers pressed together in B and U are also represented: their
        # silhouettes are a single narrow fan rather than separated rods.
        if letter == 'B':
            x *= [.70,.88,1.02,.78][shape_variant]
            l *= [1.02,.92,1.04,.98][shape_variant]
            a = -.01 + spread*.25*(i-1.5)
        if letter == 'U':
            x = .20 + (x-.20)*[.62,.80,1.,.72][shape_variant]
            a = .01 + spread*.20*(i-2.5)
        if letter in 'VW' and shape_variant == 2:
            a *= 1.65
        if letter in 'DL' and shape_variant == 3:
            a -= .22
        # The two distal phalanges can lean without rotating the entire palm.
        p1=(x + np.sin(a)*l*.48, y+np.cos(a)*l*.48)
        p2=(p1[0]+np.sin(a+curl)*l*.30,p1[1]+np.cos(a+curl)*l*.30)
        p3=(p2[0]+np.sin(a+curl*1.6)*l*.22,p2[1]+np.cos(a+curl*1.6)*l*.22)
        h.digit([(x,y),p1,p2,p3],thick)

    def folded(i, top=.43, inward=0):
        x,y = bases[i]
        x += inward
        if letter in 'AEMNST':
            top -= [.02,.10,0,.065][shape_variant]
        else:
            top -= [.07,.15,.025,.11][shape_variant]
        # Flexed fingers present rounded proximal knuckles and short nail pads.
        h.ellipse(x,top-.06,thick*1.02,.185, -.10*(i-1.5))
        h.ellipse(x+.015,top-.19,thick*.90,.112, -.10*(i-1.5))
        h.line((x-thick*.64,top-.20),(x+thick*.65,top-.22),.020,.48)
        if i < 3:
            h.line((x+thick*.96,top-.02),(x+thick*.95,top-.25),.019,.36)

    def folded_others(active):
        for i in range(4):
            if i not in active:
                folded(i, .40 + .035*(1.5-abs(i-1.5)))

    def thumb_folded(high=0):
        h.digit([(.34,-.22),(.39,-.01+high),(.16,.09+high),(-.05,.10+high)],.116,False)
        h.line((.30,-.13+high),(.07,.01+high),.022,.34)

    def thumb_out(angle=.12):
        h.digit([(.29,-.18),(.56,-.05),(.83,.03+angle)],.105)

    if letter == 'A':
        for i in range(4): folded(i,.49-.045*abs(i-1.5))
        h.digit([(.34,-.22),(.48,.04),(.48,.37)],.12)
        h.line((.32,-.10),(.34,.31),.025,.56)
    elif letter == 'B':
        for i in range(4): extended(i,[-.025,0,0,-.035][i],.015)
        thumb_folded(-.06)
        h.line((-.25,.02),(.22,-.10),.025,.18)
    elif letter in 'CO':
        # A side view: four fingers form one curved bank; opposing thumb is low.
        h.ellipse(-.11,.03,.32,.43,-.12)
        gap = [.25,.32,.20,.28][shape_variant] if letter == 'C' else .03
        # A C's top bank bends less than the closed O bank.
        tip_y = .62 if letter == 'C' else .51
        h.digit([(-.31,.17),(-.21,.55),(.07,.76),(.39,.70),(.50,tip_y)],.145)
        h.digit([(-.23,.09),(-.10,.40),(.17,.59),(.42,.53)],.110)
        h.digit([(.00,-.36),(.28,-.24),(.48,.04),(.50,.51-gap*2)],.125)
        h.line((-.18,.39),(.18,.56),.025,.34)
        h.line((.01,-.13),(.25,-.04),.022,.23)
    elif letter == 'D':
        folded_others([3]); extended(3,-.08,.015)
        h.digit([(.35,-.23),(.52,-.01),(.32,.23),(.10,.27)],.110)
        h.line((.12,.09),(.29,.14),.023,.43)
    elif letter == 'E':
        for i in range(4):
            folded(i,.53-.05*abs(i-1.5))
            x=bases[i][0]
            h.line((x-.06,.23),(x+.07,.22),.030,.72)
        h.digit([(.36,-.22),(.36,-.03),(.10,-.02),(-.19,.01)],.105,False)
        h.line((-.27,.09),(.29,.08),.034,.65)
    elif letter == 'F':
        for i in range(3): extended(i,[-.34,-.16,.02][i],.06)
        # Index curls to meet thumb, leaving a small, genuine hole.
        h.digit([(.29,.13),(.44,.38),(.59,.29),(.53,.08),(.37,.03)],.092)
        h.digit([(.28,-.19),(.53,-.08),(.53,.08)],.105)
    elif letter == 'G':
        h.ellipse(-.06,.06,.34,.34)
        for i in range(3): folded(i,.30-.035*i)
        h.digit([(.04,.26),(.41,.47),(.83,.56),(1.08,.58)],.095)
        h.digit([(.17,-.17),(.49,.04),(.83,.20)],.105)
        h.line((.10,.15),(.40,.25),.025,.36)
    elif letter == 'H':
        h.ellipse(-.04,.00,.36,.34)
        for i in range(2): folded(i,.28-.04*i)
        for j in range(2):
            dy=j*.205
            h.digit([(.02,.07+dy),(.43,.18+dy),(.85,.22+dy),(1.10,.23+dy)],.087)
        h.digit([(.23,-.19),(.45,-.01),(.20,.04)],.12)
    elif letter == 'I':
        folded_others([0]); extended(0,-.18,.035); thumb_folded()
    elif letter == 'K':
        folded_others([2,3])
        extended(2,-.33,.02); extended(3,.26,-.10)
        h.digit([(.32,-.20),(.24,.12),(.23,.53)],.109)
    elif letter == 'L':
        folded_others([3]); extended(3,-.12,.00); thumb_out(.12)
    elif letter in 'MNT':
        for i in range(4): folded(i,.46-.04*abs(i-1.5))
        # Thumb tip protrudes through the appropriate adjacent-finger gap.
        # Pinky/ring, ring/middle, middle/index respectively.
        slot={'M':-.20,'N':.00,'T':.20}[letter]
        h.digit([(.34,-.23),(.22,-.02),(slot,.26),(slot,.44)],.087)
        h.line((slot-.075,.26),(slot+.07,.29),.021,.58)
        for i in range({'M':3,'N':2,'T':1}[letter]):
            x=.29-i*.20
            h.line((x+.055,.13),(x+.05,.40),.019,.52)
    elif letter == 'P':
        # K tipped downward: the wrist is above the flexed palm and the middle
        # finger points toward the floor.  Model this viewpoint directly.
        h.parts=[];h.lines=[]
        h.bone((-.40,.94),(-.15,.45),.24)
        h.ellipse(-.03,.31,.34,.31,-.25)
        h.ellipse(-.24,.24,.13,.21,-.4)
        h.ellipse(-.07,.16,.12,.20,-.4)
        h.digit([(.13,.40),(.51,.40),(.91,.28)],.096)
        h.digit([(.15,.18),(.33,-.24),(.36,-.66)],.098)
        h.digit([(.22,.21),(.49,.04),(.56,-.15)],.110)
        h.line((.25,.10),(.43,.03),.025,.40)
    elif letter == 'Q':
        # G directed down. Its long opposed index and thumb must remain
        # visible below the compact upper fist, rather than inside the palm.
        h.parts=[];h.lines=[]
        h.bone((.25,1.02),(.02,.43),.24)
        h.ellipse(.00,.29,.32,.32)
        h.ellipse(-.21,.36,.12,.19,-.3)
        h.ellipse(-.07,.30,.12,.18,-.3)
        h.ellipse(.08,.27,.11,.18,-.3)
        h.digit([(.24,.24),(.25,-.20),(.11,-.70)],.100)
        h.digit([(-.10,.08),(-.28,-.23),(-.21,-.57)],.108)
    elif letter == 'R':
        folded_others([2,3])
        h.digit([(.10,.20),(.20,.63),(.39,1.05)],.093)
        h.digit([(.29,.13),(.15,.62),(.09,.98)],.092)
        h.line((.13,.58),(.28,.70),.027,.50)
        thumb_folded(-.08)
    elif letter == 'S':
        for i in range(4): folded(i,.49-.05*abs(i-1.5))
        h.digit([(.35,-.22),(.42,.03),(.13,.22),(-.14,.19)],.122,False)
        h.line((-.14,.10),(.12,.13),.026,.55)
    elif letter == 'U':
        folded_others([2,3]); extended(2,.025,.02); extended(3,-.045,.02)
        thumb_folded(-.07)
    elif letter == 'V':
        folded_others([2,3]); extended(2,-.28,.015); extended(3,.31,.015)
        thumb_folded(-.07)
    elif letter == 'W':
        folded_others([1,2,3])
        extended(1,-.34,.025); extended(2,.01,.015); extended(3,.35,.02)
        thumb_folded(-.10)
    elif letter == 'X':
        folded_others([3])
        curl_y = [.59,.69,.54,.64][shape_variant]
        h.digit([(.29,.13),(.31,curl_y),(.16,curl_y+.13),(.02,curl_y+.03),(.02,curl_y-.19)],.103)
        thumb_folded(-.05)
    elif letter == 'Y':
        folded_others([0]); extended(0,-.42,.04); thumb_out(.20)
    else:
        raise ValueError(letter)

    # Generic superficial palm folds, intentionally weaker than the silhouette.
    if letter not in 'GHOPQ':
        h.line((-.27,-.03),(.13,-.15),.022,.15)
        h.line((.13,-.15),(.28,-.35),.020,.13)
    # Foreshortening and a mild camera shear are generic alternative views.
    width *= [1.0,.79,1.09][view_variant]
    shear = [0,-.17,.17][view_variant]
    def transform_point(x,y):
        return x*width + shear*y, y
    # Anatomical variations are independent of labels and contain no data.
    for j,(typ,v) in enumerate(h.parts):
        if typ=='bone':
            a,b,c,d,r=v
            h.parts[j]=(typ,(*transform_point(a,b),*transform_point(c,d),r*np.sqrt(width)))
        else:
            x,y,rx,ry,a=v
            h.parts[j]=(typ,(*transform_point(x,y),rx*width,ry,a))
    h.lines=[(*transform_point(a,b),*transform_point(c,d),r,s) for a,b,c,d,r,s in h.lines]
    return h


def capsule_distance(X,Y,x1,y1,x2,y2,r):
    dx,dy=x2-x1,y2-y1
    t=np.clip(((X-x1)*dx+(Y-y1)*dy)/(dx*dx+dy*dy+1e-12),0,1)
    return np.sqrt((X-x1-t*dx)**2+(Y-y1-t*dy)**2)-r


def field(h,angle,scale,mirror, resolution=52):
    # Bounds depend only on the analytic anatomy, not on a photograph.
    bounds=[]
    # A hand crop encloses the digits and palm, not the full forearm.  The
    # forearm still renders, continuing naturally through the crop boundary.
    for typ,v in h.parts[1:]:
        if typ=='bone':
            a,b,c,d,r=v;bounds.extend([(a-r,b-r),(a+r,b+r),(c-r,d-r),(c+r,d+r)])
        else:
            x,y,rx,ry,a=v;rr=max(rx,ry);bounds.extend([(x-rr,y-rr),(x+rr,y+rr)])
    bounds=np.array(bounds)
    lo,hi=bounds.min(0),bounds.max(0)
    center=(lo+hi)/2
    extent=max(hi-lo)
    # The crop is a square containing hand and wrist, with a modest margin.
    # Match a 26-pixel feature field within the original 28-pixel frame.
    pp=(resolution/extent)*.92*scale
    yy,xx=np.mgrid[0:resolution,0:resolution]
    x=(xx-(resolution-1)/2)/pp
    y=((resolution-1)/2-yy)/pp
    a=np.deg2rad(angle)
    X=(np.cos(a)*x+np.sin(a)*y)*mirror+center[0]
    Y=-np.sin(a)*x+np.cos(a)*y+center[1]
    D=np.full_like(X,99.)
    surface=np.zeros_like(X)
    for typ,v in h.parts:
        if typ=='bone':
            d=capsule_distance(X,Y,*v)
            x1,y1,x2,y2,r=v
            dx,dy=x2-x1,y2-y1
            t=np.clip(((X-x1)*dx+(Y-y1)*dy)/(dx*dx+dy*dy+1e-12),0,1)
            u=(X-x1-t*dx)/r
            w=(Y-y1-t*dy)/r
            radial=np.sqrt(u*u+w*w)
        else:
            x0,y0,rx,ry,ang=v
            u=((X-x0)*np.cos(ang)+(Y-y0)*np.sin(ang))/rx
            w=(-(X-x0)*np.sin(ang)+(Y-y0)*np.cos(ang))/ry
            radial=np.sqrt(u*u+w*w)
            d=(radial-1)*min(rx,ry)
        D=np.minimum(D,d)
        # A front-lit rounded solid, not a learned texture.  The slight lateral
        # light makes finger valleys visible while keeping silhouettes primary.
        z=np.sqrt(np.maximum(1-radial*radial,0))
        albedo=np.clip(.55+.45*z+.12*h.light_side*u+.06*w,.25,1.10)
        surface=np.where(radial<1,np.maximum(surface,albedo),surface)
    # Smooth antialiasing and separately drawn anatomical creases.
    mask=1/(1+np.exp(np.clip(D*pp/.65,-60,60)))
    shade=np.zeros_like(mask)
    for x1,y1,x2,y2,r,s in h.lines:
        d=capsule_distance(X,Y,x1,y1,x2,y2,r)
        shade=np.maximum(shade,s*np.exp(-np.maximum(d,0)**2/(2*(.60/pp)**2)))
    light_strength = 0 if h.lighting == 0 else .72
    relief=mask*(1-light_strength+light_strength*np.maximum(surface,.45))*(1-.65*shade)
    return relief,mask


def template(h,angle,scale,mirror):
    # Analytic templates are represented on a 2x supersampled 26-pixel field.
    f,m=field(h,angle,scale,mirror)
    # Derivatives approximate the model's contour normals and palm/finger folds.
    gy,gx=np.gradient(f)
    dirs=[0,np.pi/4,np.pi/2,3*np.pi/4]
    e=[]
    for a in dirs:
        para=np.abs(gx*np.cos(a)+gy*np.sin(a))
        perp=np.abs(-gx*np.sin(a)+gy*np.cos(a))
        q=np.maximum(para-.5*perp,0)
        # A little positional tolerance is part of the geometric edge model.
        qp=np.pad(q,((1,1),(1,1)))
        q=np.maximum.reduce([qp[dy:dy+52,dx:dx+52] for dy in range(3) for dx in range(3)])
        q=q.reshape(13,4,13,4).mean((1,3))
        e.append(q)
    e=np.array(e)
    # Weak inhibition where the geometric model predicts no edge.
    e=e-.60*e.mean()
    # Every channel is spatially standardized in the actual network.  Construct
    # the same zero-DC model response, then give every candidate equal energy.
    spatial=np.linspace(1.05,.75,13)[:,None]
    e*=spatial
    e-=e.mean(axis=(1,2),keepdims=True)
    e/=np.sqrt((e*e).sum()+1e-12)
    raw=f.reshape(13,4,13,4).mean((1,3))
    raw-=raw.mean();raw*=spatial
    raw-=raw.mean()
    raw/=np.sqrt((raw*raw).sum()+1e-12)
    signed=np.array([gx.reshape(13,4,13,4).mean((1,3)),
                     gy.reshape(13,4,13,4).mean((1,3))])
    signed*=spatial
    signed-=signed.mean(axis=(1,2),keepdims=True)
    signed/=np.sqrt((signed*signed).sum()+1e-12)
    return e,raw,signed


def build(out):
    layers=[]; weights={}
    def op(name,typ,ins,**kw):
        layers.append(dict(name=name,op=typ,inputs=[ins] if isinstance(ins,str) else ins,**kw))
        return name
    def conv(name,ins,W,b=None,padding=0):
        W=np.asarray(W,np.float32)
        weights[name+'.W']=W
        weights[name+'.b']=np.zeros(W.shape[0],np.float32) if b is None else np.asarray(b,np.float32)
        return op(name,'conv',ins,out=W.shape[0],kernel=list(W.shape[-2:]),stride=1,padding=padding)
    op('contrast','norm','image')
    # Four unsigned Sobel projections, sharpened by orthogonal inhibition.
    sx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)/8
    sy=sx.T
    kernels=[]
    for a in [0,np.pi/4,np.pi/2,3*np.pi/4]:
        k=np.cos(a)*sx+np.sin(a)*sy
        kernels.extend([k,-k])
    conv('gradients','contrast',np.array(kernels)[:,None],padding=1)
    op('half_waves','relu','gradients')
    W=np.zeros((4,8,1,1),np.float32)
    for i in range(4):
        W[i,2*i:2*i+2,0,0]=1
        j=(i+2)%4;W[i,2*j:2*j+2,0,0]=-.5
    conv('oriented','half_waves',W)
    op('edges','relu','oriented')
    # Drop the boundary where zero-padding creates an artificial image edge;
    # average rather than dilate so narrow finger gaps survive at 28 pixels.
    W=np.zeros((4,4,3,3),np.float32)
    for i in range(4): W[i,i,1,1]=1
    conv('edge_tolerance','edges',W)
    op('edge_grid','avgpool','edge_tolerance',size=[2,2],stride=[2,2])
    W=np.zeros((7,4,1,1),np.float32)
    for i in range(4): W[i,i,0,0]=1
    op('edge_standardization','norm','edge_grid')
    conv('edge_slots','edge_standardization',W)
    g=np.array([1,2,1],float);g=np.outer(g,g)/16
    conv('smooth_intensity','contrast',g[None,None])
    op('intensity_grid','avgpool','smooth_intensity',size=[2,2],stride=[2,2])
    op('intensity_standardization','norm','intensity_grid')
    W=np.zeros((7,1,1,1),np.float32);W[4,0,0,0]=1
    conv('intensity_slot','intensity_standardization',W)
    conv('signed_gradients','contrast',np.array([sx,sy])[:,None])
    op('signed_grid','avgpool','signed_gradients',size=[2,2],stride=[2,2])
    op('signed_standardization','norm','signed_grid')
    W=np.zeros((7,2,1,1),np.float32)
    W[5,0,0,0]=1;W[6,1,0,0]=1
    conv('signed_slots','signed_standardization',W)
    op('features','add',['edge_slots','intensity_slot','signed_slots'])
    # Each candidate is a symbolic sign pose, not an example from the dataset.
    rotations=[-40.,-24.,-12.,0.,12.,24.,40.]
    scales=[.78,.96,1.14,1.32]
    nvar=24*len(rotations)*len(scales)*2*2
    W=np.empty((24*nvar,7,13,13),np.float32)
    k=0
    for letter in LETTERS:
        for variant in range(24):
            h=make_hand(letter,variant)
            for angle in rotations:
                for scale in scales:
                    for mirror in [1,-1]:
                        e,r,signed=template(h,angle,scale,mirror)
                        for polarity in [1,-1]:
                            W[k,:4]=e
                            W[k,4]=.45*polarity*r
                            W[k,5:]=.65*polarity*signed
                            k+=1
    conv('pose_matches','features',W,padding=2)
    op('translation_invariance','maxpool','pose_matches',size=[5,5],stride=[5,5])
    op('sign_pose_table','reshape','translation_invariance',shape=[24,nvar,1])
    op('best_symbolic_pose','maxpool','sign_pose_table',size=[nvar,1],stride=[nvar,1])
    op('scores','reshape','best_symbolic_pose',shape=[24])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**weights)
    print('Wrote',len(layers),'layers,',sum(v.size for v in weights.values()),'weights.')


if __name__=='__main__':
    build(sys.argv[1])
