"""Build a purely hand-specified, multi-cue photo recognizer. No input files."""
import sys,json,math
from pathlib import Path
import numpy as np

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
layers=[]; weights={}
def op(name,kind,src,**kw):
    layers.append(dict(name=name,op=kind,inputs=src if isinstance(src,list) else [src],**kw));return name
def linear(name,src,W,b=0,kind='dense',**kw):
    W=np.array(W,dtype='f');weights[name+'.W']=W;weights[name+'.b']=np.broadcast_to(np.asarray(b,'f'),(len(W),)).copy()
    return op(name,kind,src,out=len(W),**kw)
def conv(name,src,W,b=0,pad=0,stride=1):return linear(name,src,W,b,'conv',kernel=list(W.shape[-2:]),padding=pad,stride=stride)
def avg(name,src,n,stride=None):return op(name,'avgpool',src,size=[n,n],stride=[stride or n]*2)
def mp(name,src,size,stride):return op(name,'maxpool',src,size=size,stride=stride)
def norm(name,src):return op(name,'norm',src)
def relu(name,src):return op(name,'relu',src)
def rs(name,src,*shape):return op(name,'reshape',src,shape=list(shape))
def add(name,*src):return op(name,'add',list(src))
ln=lambda x: np.sqrt(np.square(x).sum())
LUM=np.array([.30,.53,.17]); lum=conv('light','image',LUM[None,:,None,None]); light=norm('relative_light',lum)

# Draw generic projections of objects in arbitrary object-space units. A list of
# inked parts is rasterized at build time, not obtained from example pictures.
class Drawing:
    def __init__(self):self.shapes=[]
    def e(self,x,y,w,h,v,a=0):self.shapes.append(('e',(x,y,w,h,a),v)) # semiaxes
    def p(self,points,v):self.shapes.append(('p',points,v))
    def r(self,x,y,xx,yy,v):self.p([(x,y),(xx,y),(xx,yy),(x,yy)],v)
    def line(self,x,y,xx,yy,w,v):
        l=math.hypot(xx-x,yy-y); dx=(yy-y)/l*w/2;dy=-(xx-x)/l*w/2
        self.p([(x+dx,y+dy),(xx+dx,yy+dy),(xx-dx,yy-dy),(x-dx,y-dy)],v)
    def field(self,x,y):
        z=np.zeros_like(x); cover=np.zeros_like(x,dtype=bool)
        for kind,pts,v in self.shapes:
            if kind=='e':
                cx,cy,w,h,a=pts;a=math.radians(a);xx=x-cx;yy=y-cy
                mask=((xx*math.cos(a)+yy*math.sin(a))/w)**2+((yy*math.cos(a)-xx*math.sin(a))/h)**2<=1
            else:
                mask=np.zeros_like(cover);x1,y1=pts[-1]
                for x2,y2 in pts:
                    if y2!=y1:mask^=((y1>y)!=(y2>y)) & (x < x1+(x2-x1)*(y-y1)/(y2-y1))
                    x1,y1=x2,y2
            z=np.where(mask,v,z);cover|=mask
        return z,cover

# Paint levels represent brightness relative to the surroundings, not colors
# sampled from photographs. Wheels and glass are generally darker than paint.
def airplane(i,v):
    d=Drawing();p=d.p;e=d.e
    if i==0: # looking up at the planform: swept wings and tailplane
        p([(-.13,-.52),(-.12,-.13),(-1,.24),(-.98,.37),(-.11,.15),(-.07,.57),(-.34,.73),(-.32,.79),(0,.71),(.32,.79),(.34,.73),(.07,.57),(.11,.15),(.98,.37),(1,.24),(.12,-.13),(.11,-.52),(0,-.78)],v)
    elif i==1: # profile, fin on the left, wing seen obliquely
        p([(-1,-.06),(-.89,-.1),(-.97,-.47),(-.83,-.46),(-.57,-.1),(.61,-.13),(.83,-.06),(1,.04),(.83,.13),(-.46,.15),(-.95,.06)],v)
        p([(-.05,.03),(.32,.01),(-.23,.44),(-.48,.47)],v*.95)
    elif i==2: # head on, wings taper, nacelles below
        p([(-1,-.12),(-.2,-.025),(.2,-.025),(1,-.12),(.92,-.045),(.13,.13),(-.13,.13),(-.92,-.045)],v)
        e(0,.012,.17,.19,v);d.line(0,0,0,-.48,.05,v)
        for s in [-1,1]:e(s*.42,.105,.092,.085,v)
    else: # three-quarter swept wing
        p([(-.94,.32),(-.35,.03),(-.85,-.29),(-.68,-.31),(-.03,-.12),(.74,-.4),(.94,-.35),(.76,-.2),(.22,.02),(.77,.48),(.58,.49),(-.10,.15),(-.68,.40)],v)
        p([(-.63,.24),(-.81,-.04),(-.9,-.01),(-.83,.34)],v)
    return d

def car(i,v):
    d=Drawing();p=d.p;e=d.e;r=d.r
    if i in (0,1,2):
        # low chassis, glazed greenhouse, paired tires on the road
        front=.98; shift=[0,.18,-.09][i]; roof=[-.53,-.49,-.58][i]; rear=[-.52,-.57,-.68][i]; win= -.65 if v>0 else .15
        p([(-.96,-.03),(-.83,-.16),(rear,-.18),(-.31+shift,roof),(.18+shift,roof),(.56,-.19),(.87,-.13),(front,.01),(.95,.28),(-.92,.3)],v)
        p([(rear+.08,-.16),(-.26+shift,roof+.07),(.16+shift,roof+.07),(.48,-.16)],win)
        d.line(-.035+shift,roof+.05,-.04+shift,-.16,.04,v)
        p([(-.95,.21),(.96,.21),(.88,.33),(-.88,.33)],-.60)
        for x in [-.60,.60]:
            e(x,.26,.185,.21,-1.0);e(x,.26,.09,.11,.24)
        if i==1:p([(.58,-.12),(.91,-.09),(.97,.03),(.62,.055)],v*.6+.22)
    elif i==3: # frontal or rear view, windshield, grille, lamps
        p([(-.9,.39),(-.94,-.02),(-.77,-.17),(-.58,-.56),(-.42,-.62),(.45,-.62),(.61,-.55),(.77,-.17),(.94,-.02),(.9,.39)],v)
        p([(-.68,-.16),(-.51,-.5),(.49,-.5),(.68,-.16)],-.75 if v>0 else .24)
        r(-.46,.16,.45,.32,-.84)
        for s in (-1,1):e(s*.68,.075,.18,.077,.75);r(s*.69-.12,.38,s*.69+.12,.5,-.90)
    else: # three-quarter view: projecting hood and near wheel
        p([(-1,.02),(-.83,-.15),(-.53,-.46),(-.21,-.57),(.32,-.53),(.60,-.16),(.94,.03),(.96,.29),(.50,.46),(-.82,.21)],v)
        p([(-.73,-.15),(-.49,-.40),(-.26,-.47),(-.29,-.11)],-.65)
        p([(-.21,-.47),(.28,-.45),(.49,-.16),(-.22,-.09)],-.65 if v>0 else .26)
        p([(.32,.22),(.91,.12),(.89,.25),(.34,.37)],-.61)
        for x,y in [(-.73,.16),(.06,.32)]:e(x,y,.13,.19,-1);e(x,y,.055,.09,.22)
    return d

def truck(i,v):
    d=Drawing();p=d.p;e=d.e;r=d.r
    if i in (0,1,2): # box lorry, semi or tipper
        ht=[-.57,-.66,-.39][i];r(-.99,ht,.32,.20,v)
        p([(.34,.23),(.34,-.37),(.67,-.36),(.83,-.16),(.94,-.10),(.95,.22)],v*.5+.10)
        p([(.43,-.31),(.64,-.31),(.75,-.12),(.43,-.12)],-.81)
        r(-.93,.21,.93,.31,-.78)
        for x in [-.75,-.47,.70]:e(x,.27,.12,.16,-1.0);e(x,.27,.055,.065,.15)
    elif i==3: # tall cab nearly head-on
        p([(-.67,.47),(-.69,-.51),(-.5,-.7),(.51,-.7),(.68,-.50),(.67,.47)],v)
        r(-.54,-.40,.54,-.06,-.85 if v>0 else .22);r(-.38,.13,.38,.39,-.75)
        for s in [-1,1]:r(s*.48-.095,.24,s*.48+.095,.34,.76);r(s*.5-.12,.44,s*.5+.12,.62,-.96)
    else: # perspective on the box side and cab front
        p([(-.94,-.41),(.23,-.73),(.25,.26),(-.93,.09)],v)
        p([(.23,-.73),(.88,-.57),(.9,.24),(.25,.31)],v*.75)
        p([(.16,-.37),(.35,-.53),(.91,-.42),(.97,.44),(.26,.54),(.12,.34)],v*.55+.06)
        p([(.32,-.35),(.84,-.33),(.89,-.03),(.32,-.06)],-.8);r(.4,.16,.83,.36,-.65)
        for x,y in [(-.78,.13),(-.52,.16),(.16,.37)]:e(x,y,.09,.14,-.98)
    return d

def ship(i,v):
    d=Drawing();p=d.p;r=d.r
    if i==0: # sailing boat
        p([(-.92,.24),(.95,.19),(.66,.41),(-.50,.43)],v);d.line(.06,.24,.06,-.91,.03,-.5)
        p([(.00,-.78),(-.70,.15),(.00,.12)],.86);p([(.11,-.63),(.14,.12),(.73,.11)],.77)
    elif i in [1,2]: # cargo ship or ferry; the water cuts off the bottom
        p([(-1,.02),(-.64,.10),(.63,.10),(1,-.02),(.77,.33),(-.75,.33)],v)
        if i==1:r(-.54,-.22,-.25,.09,.77);r(-.48,-.36,-.34,-.21,.51);r(-.08,-.07,.62,.10,.32)
        else:
            p([(-.74,.09),(-.59,-.18),(.44,-.20),(.72,.09)],.78);r(-.49,-.29,.31,-.17,.65);r(-.20,-.47,-.07,-.27,-.56)
            r(-.52,-.13,.43,-.08,-.39)
    elif i==3: # speedboat
        p([(-.97,.04),(.53,-.06),(1,-.24),(.74,.14),(-.49,.33),(-.93,.20)],v)
        p([(-.52,.02),(-.35,-.23),(.02,-.30),(.31,-.09)],-.66)
    else:
        p([(-.77,.02),(.70,.05),(.29,.36),(-.20,.43)],v);r(-.41,-.25,.35,.04,.76);r(-.23,-.48,.20,-.24,.69);d.line(0,-.45,.02,-.81,.035,-.4)
    return d

def bird(i,v):
    d=Drawing();e=d.e;p=d.p
    if i in [0,1,2]: # bird on perch, upright or swimming
        lean=[-27,-61,0][i];e(-.09,.05,.57,.31,v,lean)
        hx,hy=[(.37,-.27),(.17,-.48),(.43,-.18)][i];e(hx,hy,.225,.224,v);p([(hx+.12,hy-.055),(hx+.51,hy+.02),(hx+.12,hy+.08)],v)
        p([(-.32,.07),(-.89,.28 if i!=1 else .65),(-.78,.48 if i!=1 else .74),(-.15,.29)],v)
        e(-.14,.03,.36,.16,v*.56,lean)
        e(hx+.076,hy-.085,.04,.04,-.9)
    elif i==3: # airborne, curved wings rather than a rigid cruciform
        p([(-.99,-.42),(-.66,-.38),(-.24,-.21),(0,.04),(.26,-.22),(.72,-.36),(1,-.32),(.61,-.20),(.25,.03),(.03,.24),(-.15,.18),(-.38,-.06),(-.72,-.27)],v)
    else: # long neck on an aquatic bird
        e(-.17,.18,.62,.27,v);d.line(.30,.17,.40,-.46,.14,v);e(.46,-.46,.15,.117,v);p([(.53,-.48),(.83,-.43),(.53,-.40)],v)
    return d

def quadruped(k,i,v):
    d=Drawing();e=d.e;p=d.p;line=d.line;horse=k==7;dog=k==5;cat=k==3
    if k==4 and i in (1,3): # grazing profile or a buck looking at the camera
        if i==1:
            e(-.15,-.18,.54,.235,v,3);line(.21,-.19,.57,.32,.18,v);e(.65,.36,.21,.105,v,32);e(.54,.22,.066,.11,v,-26)
            for xx in [-.52,-.32,.14,.30]:line(xx,-.04,xx+.055,.62,.06,v)
            p([(-.61,-.26),(-.76,-.39),(-.63,-.09)],v)
        else:
            e(0,.35,.265,.40,v);e(0,-.10,.218,.32,v);e(0,.08,.126,.13,v*.65);e(0,.13,.085,.05,-.84)
            for s in [-1,1]:
                e(s*.24,-.29,.175,.075,v,-s*30);e(s*.11,-.15,.041,.033,-.86)
                line(s*.095,-.33,s*.25,-.62,.032,v);line(s*.25,-.62,s*.47,-.91,.031,v);line(s*.29,-.69,s*.23,-.94,.025,v)
                line(s*.13,.52,s*.15,.95,.058,v)
        return d
    if i in ((0,1) if (dog or cat) else (0,1,2)): # lateral projection, legs displaced by the gait
        ys=-.17; bw=.52 if dog or cat else .59;bh=.28 if not cat else .24
        e(-.14,ys,bw,bh,v)
        if horse or k==4:
            p([(.11,-.36),(.41,-.77),(.57,-.69),(.39,-.13),(.18,.015)],v);e(.56,-.65,.23,.13,v,26 if horse else 8);e(.70,-.57,.136,.095,v,28)
            for x in [.42,.57]:p([(x-.068,-.73),(x-.025,-.97),(x+.07,-.76)],v)
            if horse:p([(-.64,-.32),(-.79,-.20),(-.89,.32),(-.75,.24),(-.56,-.16)],v*.86-.13)
            else:p([(-.65,-.25),(-.82,-.36),(-.66,-.04)],v)
        else:
            e(.43,-.33,.215,.22,v);e(.61,-.27,.16,.094,v)
            if cat:p([(.32,-.45),(.29,-.67),(.49,-.5)],v)
            else:e(.33,-.37,.09,.18,v*.7-.24,12)
            line(-.61,-.17,-.89,-.47,.086,v)
        top=.015;end=.7 if horse or k==4 else .47
        for num,x in enumerate([-.54,-.34,.15,.32]):
            q=([-.10,.06,-.04,.10] if i==0 else [-.33,.24,-.20,.25] if i==1 else [-.12,-.22,.20,.01])[num]; knee=x+q*.72+(-.05 if num<2 else .045)
            line(x,top-.12,knee,end*.55,.108 if horse else .081,v);line(knee,end*.55,x+q,end,.062,v)
    elif i==3 and (horse or k==4): # frontal animal standing, long face and two legs
        e(0,-.02,.34,.45,v);e(0,-.43,.23,.33,v*.7);e(0,-.29,.143,.18,v*.7-.21)
        for s in [-1,1]:
            e(s*.24,-.71,.092,.18,v,s*23);e(s*.138,-.5,.042,.045,-.86);line(s*.19,.19,s*.2,.83,.09,v)
    else: # head study; cats have pointed ears, dogs hanging ears and a marked muzzle
        if k in [3,5]:
            if i==2: # seated, head over shoulders and forepaws
                e(0,.34,.50,.62,v);e(.22,.6,.37,.30,v*.80)
                hy=-.36;e(0,hy,.38,.38,v)
                for s in [-1,1]:
                    if cat:p([(s*.35,hy-.10),(s*.33,hy-.51),(s*.06,hy-.26)],v)
                    else:e(s*.345,hy,.15,.33,v*.54-.23,s*12)
                    e(s*.17,hy-.047,.068,.041,-.94);line(s*.13,.26,s*.16,.84,.092,v*.9)
                e(0,hy+.12,.095 if dog else .054,.064,-.93);return d
            if i==3 and dog: # profile; brow, ear, bridge of muzzle, nose
                e(-.13,-.08,.51,.51,v);e(-.49,-.02,.21,.44,v*.50-.42,16);e(.42,.08,.40,.22,v)
                e(.72,.025,.11,.108,-.95);e(.18,-.235,.086,.077,-.93);e(-.20,.56,.41,.53,v);return d
            if i==3 and cat: # three-quarter face, narrower far cheek
                e(-.09,.025,.59,.59,v);p([(-.66,-.11),(-.67,-.82),(-.22,-.46)],v);p([(.08,-.47),(.51,-.78),(.44,-.17)],v)
                e(-.21,-.12,.117,.068,-.95);e(.276,-.144,.070,.053,-.92);e(.20,.12,.072,.055,-.90);e(.14,.25,.14,.10,v*.65+.18);return d
            if cat:
                p([(-.70,-.19),(-.67,-.87),(-.21,-.54),(.22,-.54),(.68,-.87),(.70,-.17)],v)
                for s in [-1,1]:p([(s*.58,-.68),(s*.55,-.36),(s*.32,-.43)],v*.52)
            else:
                for s in [-1,1]:e(s*.65,-.13,.23,.49,v*.38-.50,s*(-14))
            e(0,-.014,.66 if cat else .57,.57 if cat else .68,v)
            for s in [-1,1]:
                e(s*.265,-.14,.124 if cat else .102,.069,-.9);e(s*.25,-.16,.034,.023,.60)
                e(s*.112,.224,.15,.14,.36 if v>0 else -.10)
            e(0,.137,.072 if cat else .155,.065 if cat else .10,-.91)
        else: # large horse/deer head in profile, long rectangular nose
            p([(-.74,.85),(-.5,-.30),(-.23,-.67),(.16,-.63),(.79,.13),(.69,.37),(.42,.41),(-.02,.05),(-.22,.78)],v)
            p([(-.36,-.54),(-.47,-.97),(-.18,-.66),(.0,-.95),(.048,-.58)],v);e(.02,-.36,.064,.063,-.88);e(.60,.22,.065,.048,-.76)
    return d

def frog(i,v):
    d=Drawing();e=d.e;p=d.p
    if i in [0,1,2]:
        # crouching body and folded back legs, flat head with raised eye
        e(-.14,.1,.58,.35,v,-12);e(.36,-.072,.41,.235,v,-8);e(.36,-.25,.14,.145,v);e(.38,-.27,.075,.078,-.9)
        e(-.39,.20,.38,.21,v*.60+.07,-25);d.line(-.62,.32,-.25,.46,.107,v)
        d.line(.20,.10,.34,.43,.09,v);d.line(.34,.43,.64,.42,.06,v)
        if i==1:d.line(-.64,.27,-.89,.51,.074,v)
    else:
        e(0,.16,.7,.39,v);e(0,-.08,.6,.30,v)
        for s in [-1,1]:e(s*.36,-.29,.175,.176,v);e(s*.36,-.31,.101,.075,-.89);e(s*.62,.27,.27,.20,v*.76,s*25)
        d.line(-.41,.05,.41,.05,.04,v*.4-.38)
    return d

def drawing(k,i,v):
    if k in [3,4,5,7]:return quadruped(k,i,v)
    return {0:airplane,1:car,2:bird,6:frog,8:ship,9:truck}[k](i,v)

# Correlate blurred, locally contrasted drawings and their oriented boundaries
# against a normalized light field and four soft orientation energy fields.
# Translation is part of the weight-sharing implicit in template construction.
def translated(q,dy,dx):
    r=np.zeros_like(q);a,b=q.shape[-2:];r[...,max(0,dy):min(a,a+dy),max(0,dx):min(b,b+dx)]=q[...,max(0,-dy):min(a,a-dy),max(0,-dx):min(b,b-dx)];return r
small=avg('small_photo',light,2)
Y,X=np.mgrid[-1:2,-1:2];kern=[]
for a in np.arange(4)*np.pi/4:
    g=(X*np.cos(a)+Y*np.sin(a))*np.exp(-(X*X+Y*Y)/1.4);g/=abs(g).sum()/2
    kern.extend([g,-g])
diff=conv('slope',light,np.array(kern)[:,None],stride=2)
slope=relu('rectified_slope',diff); di=np.repeat(np.eye(4),2,axis=1)
orient=conv('orientation',slope,di[:,:,None,None]);orientN=norm('relative_orientation',orient)

def blur(x,s):
    # Small fixed separable Gaussian with constant (edge-extension) borders.
    n=max(1,int(s*2.8+.5));g=np.exp(-np.arange(-n,n+1)**2/(2*s*s));g/=g.sum()
    for ax in [0,1]:
        x=np.swapaxes(x,ax,0);pad=np.pad(x,((n,n),(0,0)),mode='edge');x=sum(v*pad[j:j+x.shape[0]] for j,v in enumerate(g));x=np.swapaxes(x,ax,0)
    return x

def geom(k,i,j):
    # Product grid of aspect, scale, azimuth/elevation, and field of view.
    ii=j%3; jj=(j//3)%2; t=j//6
    w=([17,24,31] if k not in (2,6) else [13,21,29])[ii]
    if i>=3 and k in (3,4,5,6,7):w=[17,26,35][ii]
    if k==7 and i<3:w=[21,28,34][ii]
    if k==4 and i<3:w=[14,21,28][ii]
    h=w*([.82,1.12][jj]); a=0
    if k==0:a=([0,28,60,90,-25,-60] if i==0 else [-19,-7,6,18,-31,34])[j%6]
    elif k in [1,8,9]:a=[-7,0,6][(j//2)%3]
    elif k==2:a=[-18,0,17][(j//2)%3]
    else:a=[-9,0,9][(j//2)%3]
    # Reflections and contrast are both represented for each pose.
    return w/2,h/2,a,[-1,1][t%2],[-.68,.70][t//2]

def unit(x):return x/max(ln(x),1.e-6)
yy,xx=np.mgrid[0:64,0:64];xx=(xx+.5)/2-16;yy=(yy+.5)/2-16
Ts=[];Os=[];Ms=[];nvar=5*24*9;per_class_parts=[]
for k in range(10):
    for i in range(5):
        for j in range(24):
            w,h,a,flip,v=geom(k,i,j);a=math.radians(a);c=math.cos(a);s=math.sin(a)
            z,mask=drawing(k,i,v).field(flip*(xx*c+yy*s)/w,(yy*c-xx*s)/h)
            # Soft luminance templates emphasize both within-object layout and contour.
            b=blur(z,.95);g=b.reshape(16,4,16,4).mean((1,3));tlow=g-blur(g,2.4);tlow-=tlow.mean()
            tw=.60*unit(tlow)+.40*unit(g-g.mean());tw=unit(tw)
            # Oriented contour templates have a zero-mean background and mildly
            # subtract competing directions at the same location.
            gy,gx=np.gradient(b);ors=[]
            for a1 in np.arange(4)*np.pi/4:
                u=abs(gx*np.cos(a1)+gy*np.sin(a1));u=blur(u,1.8)[2:62:4,2:62:4];ors.append(u)
            ori=np.array(ors);ori=ori-.52*ori.mean(0);ori-=ori.mean((1,2),keepdims=True);ori=unit(ori)
            if j//6<2: # just geometry, independent of paint polarity
                frac=mask.reshape(16,4,16,4).mean((1,3));proximity=blur(frac,1.9)
                fg=frac/(frac.sum()+1.e-6);bg=(1-frac)*proximity;bg=bg/(bg.sum()+1.e-6)
                ms=fg-bg
                for mdy in [-2,0,2]:
                    for mdx in [-2,0,2]:
                        mm=translated(ms,mdy,mdx);mm-=mm.mean();Ms.append(mm.ravel())
            # For the nine translated views, translation clips evidence outside
            # the image, rather than wrapping the far border into the template.
            for dy in [-2,0,2]:
                for dx in [-2,0,2]:
                    T=translated(tw,dy,dx);T-=T.mean();Ts.append(unit(T).ravel());O=translated(ori,dy,dx);O-=O.mean((1,2),keepdims=True);Os.append(unit(O).ravel())
# The maps have 16x16 and 15x15 samples respectively. Normalize by square
# root of the area; the scores then have a correlation-coefficient scale.
A=linear('model_shading',small,np.array(Ts,'f')*(.68/16));del Ts
B=linear('model_contour',orientN,np.array(Os,'f')*(.32/(15*2)));del Os
ab=avg('chromatic_grid','image',2);channels=[]
for tag,axis in [('leaf',[-.55,1,-.45]),('cool',[-.68,-.32,1]),('russet',[1,-.76,-.24])]:
    q=conv(tag+'_axis',ab,np.array(axis)[None,:,None,None]);rho=norm(tag+'_unit',q)
    # A raw component suppresses unimportant chromatic noise in neutral scenes;
    # a relative component handles global color casts and tinted illumination.
    mix=add(tag+'_opponent',conv(tag+'_raw',q,np.array([5.0]).reshape(1,1,1,1)),conv(tag+'_relative',rho,np.array([.51]).reshape(1,1,1,1)))
    channels.append(linear(tag+'_in_outline',mix,np.asarray(Ms,'f')))
# These are sums of opponent-channel contrasts inside vs beside a particular
# outline, not image-wide color votes. Opposing cues need not be present for
# a gray-scale photo to get a good match.
color_rules=np.array([[-.14,-.62,.04],[-.45,-.12,.22],[-.42,-.12,.19],[-.30,-.13,.18],[-.55,-.25,.37],[-.32,-.16,.34],[.59,-.25,.06],[-.48,-.24,.38],[-.14,-.64,.09],[-.39,-.15,.21]])
# Channel-scaling is done as a 1x1 conv, sharing over all object poses.
cc=[]
for jj,ch in enumerate(channels):
    node=rs('outline_field_'+str(jj),ch,10,5,108)
    cc.append(conv('color_expectation_'+str(jj),node,np.diag(color_rules[:,jj])[:,:,None,None]*.105))
color=add('outline_color_agreement',*cc);color=rs('chroma_polarity_grid',color,50,1,108)
# Double the alternatives along the contrast-polarity dimension, representing
# dark and light object paint with the same underlying color-mask response.
stack=np.zeros((100,50));stack[np.arange(100),np.repeat(np.arange(50),2)]=1
# Segment by departure from the scene flanking the subject. At each height
# estimate two possible backgrounds from broad bands near the left and right
# borders, smooth in height, and ask if the pixel could belong to *either*.
# This cancels gradients such as sky above water while retaining a boat.
tt=np.arange(16);yy0,xx0=np.mgrid[:16,:16];eye=np.eye(256).reshape(256,16,16);bgweights=[]
for cx0 in [1.,14.]:
    sm=np.exp(-.5*(((yy0[None]-yy0.ravel()[:,None,None])/1.65)**2+((xx0[None]-cx0)/1.4)**2));sm/=sm.sum((1,2),keepdims=True);bgweights.append((eye-sm).reshape(256,256))
# Treat channels as positions in a 1x1 convolution so the spatial operator
# is shared across R, G, and B, without filling a huge dense matrix with zeros.
t0=rs('channels_as_rows',ab,3,16,16)
# Interleave positions by systematic fixed rows; dense subtracts each of the
# six smoothly varying flanks from its corresponding color plane.
ww=np.zeros((6,256,3,256),'f')
for j in range(2):
    for c0 in range(3):ww[3*j+c0,:,c0,:]=bgweights[j]
bg=rs('flank_residuals',linear('subtract_flanks',t0,ww.reshape(1536,768)),6,16,16);ww=np.zeros((12,6,1,1));colour_axes=[3*LUM,2.8*np.array([1,-1,0]),2.8*np.array([-.5,-.5,1])]
for k0 in range(2):
    for c0,ax0 in enumerate(colour_axes):
        for ph,sign in enumerate([1,-1]):ww[6*k0+2*c0+ph,3*k0:3*k0+3,0,0]=ax0*sign
opp=relu('foreground_differences',conv('flank_opponents',bg,ww));ww=np.zeros((2,12,1,1));ww[0,:6]=1;ww[1,6:]=1
res=conv('difference_from_flanks',opp,ww);better=relu('prefer_right_background',conv('which_flank',res,np.array([1,-1]).reshape(1,2,1,1)));sal=add('foreground_margin',conv('left_novelty',res,np.array([1,0]).reshape(1,2,1,1)),conv('subtract_background_match',better,np.array([-1.]).reshape(1,1,1,1)))
# Keep shape evidence bounded and smooth for very strong color contrasts.
clip=relu('foreground_overflow',conv('overly_strong_contrast',sal,np.array([1.]).reshape(1,1,1,1),-1.35));sal=add('foreground',sal,conv('limit_contrast',clip,np.array([-1.]).reshape(1,1,1,1)));salf=norm('relative_foreground',sal)
sw=np.asarray(Ms,'f');sw=sw/np.maximum(np.sqrt((sw*sw).sum(1,keepdims=True)),1.e-6);sal_match=linear('segmented_silhouettes',salf,sw*(.22/16));sal_match=rs('figure_polarity_grid',sal_match,50,1,108)
color=add('colored_figures',color,sal_match);color=conv('paint_polarities',color,stack[:,:,None,None]);color=rs('colored_poses',color,10800)
match=add('model_agreement',A,B,color);match=rs('pose_grid',match,10,5,nvar//5)
each=mp('best_per_view',match,[1,nvar//5],[1,nvar//5]);best=mp('best_view',each,[5,1],[5,1]);best=rs('best_model',best,10)
meanpose=op('typical_view','avgpool',each,size=[5,1],stride=[5,1]);meanpose=rs('typical_model',meanpose,10)
model=add('shape_evidence',linear('top_shape',best,np.eye(10)*7),linear('support_shapes',meanpose,np.eye(10)*2.5))

# Piecewise-linear color categories. A pixel's membership falls off as it
# violates the indicated RGB half spaces. Thresholds are in ordinary 0..1 RGB.
R=np.eye(3)[0];G=np.eye(3)[1];B=np.eye(3)[2]
colors={
 'blue':[(B-R,.145,.18),(B-G,.038,.115)],
 'leaf':[(G-R,.075,.135),(G-B,.14,.172)],
 'olive':[(G-B,.18,.16),(G-R,-.018,.115),(R-G,-.025,.15)],
 'warm':[(R-G,.14,.16),(G-B,.08,.16)],
 'tan':[(R-G,.077,.11),(R-B,.18,.19),(G-B,.045,.13)],
 'red':[(R-G,.32,.20),(R-B,.25,.17)],
 'paintblue':[(B-R,.31,.19),(B-G,.19,.15)],
 'bright':[(LUM,.86,.28)], 'black':[(-LUM,-.10,.20)],
 'white':[(LUM,.83,.28)], 'grey':[], 'muted':[],
 'pink':[(R-G,.16,.12),(B-G,.08,.10)],
}
for ch in ['white','grey','muted']:
 tol,soft={'white':(.085,.16),'grey':(.055,.14),'muted':(.12,.18)}[ch]
 for axis in [R-G,R-B,G-B]:colors[ch].extend([(axis,-tol,soft),(-axis,-tol,soft)])
bounds=[]; spans=[]; bias=[]
for cname,conditions in colors.items():
    inds=[]
    for ax,level,soft in conditions:
        inds.append(len(bounds));bounds.append(-ax/soft);bias.append(level/soft)
    spans.append(inds)
clr=avg('local_color','image',2);viol=relu('color_violations',conv('color_boundaries',clr,np.array(bounds)[:,:,None,None],bias));un=np.zeros((len(spans),len(bounds)))
for i,idx in enumerate(spans):un[i,idx]=-1
cmap=relu('color_membership',conv('soft_intersections',viol,un[:,:,None,None],1))

# Set up named spatial averages. Only those used by a hand-authored rule
# are emitted. A weighted mean or a difference of means is just a dense row.
fields={}
def field(name,node,ch,nc,side):fields[name]=node,ch,nc,side
for j,k in enumerate(colors):field(k,cmap,j,len(colors),16)
field('value',avg('area_light',lum,2),0,1,16)

def areas(n,region):
    y,x=np.mgrid[:n,:n];x=(x+.5)/n;y=(y+.5)/n
    ctr=np.exp(-((x-.5)/.285)**4-((y-.51)/.26)**4);rim=1-np.exp(-((x-.5)/.39)**6-((y-.50)/.365)**6)
    arr={'all':np.ones_like(x),'ctr':ctr,'rim':rim,'top':(y<.31),'bot':(y>.72),
         'ends':((x<.22)|(x>.78)),'corn':rim*((y<.28)|(y>.73)),
         'shoulder':(abs(x-.5)>.29)*(y>.18)*(y<.66),
         'belt':np.exp(-((y-.55)/.16)**4),'lower':np.exp(-((y-.71)/.15)**4)*np.exp(-((x-.5)/.37)**6)}[region]
    return arr/arr.sum()

# Fine texture versus lengthwise-averaged long, straight boundaries. Responses
# here are contrast normalized within each picture, not from dataset moments.
def oriented_kernel(sz,angle,longwidth,shortwidth):
    y,x=np.mgrid[:sz,:sz]-(sz-1)/2;c=math.cos(angle);s=math.sin(angle);r=x*c+y*s;t=y*c-x*s
    f=r*np.exp(-.5*((r/shortwidth)**2+(t/longwidth)**2));return f/(abs(f).sum()/2)

def edge_features():
    kernels=[];nms=[]
    for length,sig,tag in [(2.65,.88,'long'),(.95,.72,'short')]:
        for j in range(4):
            d=oriented_kernel(7,j*np.pi/4,length,sig);kernels.extend([d,-d]);nms.append(f'{tag}{j}')
    # Isotropic DoG measures stippling, grass, scales and textured fur. Uniform
    # image areas and smooth shading give almost no response at this scale.
    y,x=np.mgrid[-3:4,-3:4];g=np.exp(-(x*x+y*y)/(2*1.03**2));g/=g.sum();d=-g;d[3,3]+=1;kernels.extend([d,-d]);nms.append('micro')
    rr=relu('edge_halfwaves',conv('spatial_derivatives',light,np.array(kernels)[:,None],stride=2));ne=len(nms)
    ed=conv('edge_strength',rr,np.repeat(np.eye(ne),2,axis=1)[:,:,None,None])
    for j,k in enumerate(nms):field(k,ed,j,ne,13)
    # A local smoothness indicator distinguishes featureless sky/backdrop from
    # grass/trees, without confusing oscillatory texture with a DC color.
    M=np.zeros((3,ne));M[0,4:8]=-.82;M[0,-1]=-2.0;M[1,4:8]=.45;M[2,-1]=3.0
    flags=relu('surface_features',conv('surface_thresholds',ed,M[:,:,None,None],[1.0,-.32,-.20]));
    for j,k in enumerate(['flat','texture','grain']):field(k,flags,j,3,13)
edge_features()

# For very long horizontal edges compute separate left/right and center
# contrasts; a strong waterline, windowsill or road can extend for many pixels.
# Use the fixed relative intensities on 2x2 neighborhoods for these features.
line_w=[];line_groups=[];ly,lx=np.mgrid[:16,:16]+.5
for part in ['across','side','low','vert']:
    group=[]
    for axis in ([0] if part!='vert' else [1]):
        U,V=(lx,ly) if axis==0 else (ly,lx)
        for pos in (np.arange(2,14,.75) if part!='low' else np.arange(8,14,.75)):
            for sl in ([-.1,0,.1] if part!='side' else [0]):
                z=(V-pos-sl*(U-8));vert=np.exp(-.5*((z-.8)/.65)**2)-np.exp(-.5*((z+.8)/.65)**2)
                horiz=np.ones_like(lx) if part!='side' else (np.abs(U-8)>3.5)
                H=vert*horiz;H-=H.mean();H=2*H/abs(H).sum();group.extend([len(line_w),len(line_w)+1]);line_w.extend([H.ravel(),-H.ravel()])
    line_groups.append(group)
lines=linear('line_contrasts',small,line_w)
# Pack selected responses as one map per type and pool over positions/orientations.
ng=max(map(len,line_groups));S=np.zeros((ng*4,len(line_w)))
for i,ids in enumerate(line_groups):S[np.arange(ng)+ng*i,np.resize(ids,ng)]=1
lgrid=rs('line_search_grid',linear('line_groups',lines,S),4,1,ng);linefacts=mp('longest_edges',lgrid,[1,ng],[1,ng]);
for j,s in enumerate(['hline','waterline','baseline','vline']):field(s,linefacts,j,4,1)

# Separate, deformable landmark evidence. Each part is scored against its
# local surround. Combine parts with an approximation of their minimum, so a
# single very dark mark cannot masquerade as two wheels or a pair of eyes.
# Evaluated by dense layers over the per-photo normalized small luminance map.
partw=[];layout=[];partnames=['tires','running','feline','canine','gaze','haunch']
y,x=np.mgrid[:16,:16];x=2*x+1;y=2*y+1
def gauss(a,b,sx,sy):
    m=np.exp(-.5*(((x-a)/sx)**2+((y-b)/sy)**2));return m/max(m.sum(),1.e-12)
def point(a,b,sx,sy):return gauss(a,b,sx*2.1,sy*2.1)-gauss(a,b,sx,sy)
for tag in partnames:
    group=[]
    for cx in [11.5,14.5,17.5,20.5]:
        for cy in ([17,20.5,24] if tag in ('tires','running') else [9,12.5,16,19]):
            for scale in ([.73,1.,1.3] if tag!='gaze' else [.70,1.,1.35]):
                for lean in [-.12,.12]:
                    def loc(ax,ay):return cx+scale*ax,cy+scale*(ay+lean*ax)
                    def blob(ax,ay,sx,sy):return point(*loc(ax,ay),scale*sx,scale*sy)
                    def spot(ax,ay,sx,sy):return gauss(*loc(ax,ay),scale*sx,scale*sy)
                    if tag in ('tires','running'):
                        apart=6.8 if tag=='tires' else 9.;sep=apart
                        a,b=[blob(t,0,1.28,1.55) for t in (-sep,sep)]
                        # Light ground below the dark underside, lighter paint above.
                        c=(spot(0,3.10,4.3,1.05)+spot(0,-3.5,4.5,1.1))/2-spot(0,.4,4.4,.9)
                    elif tag in ('feline','canine','gaze'):
                        sep=3.7 if tag!='canine' else 4.2
                        a,b=[blob(t,0,1.15,.90) for t in (-sep,sep)];dy={'feline':3.25,'canine':5.4,'gaze':4.4}[tag]
                        c=blob(0,dy,1.1 if tag!='canine' else 1.65,1.0)
                        if tag=='gaze':c=.5*(a+b)
                    else: # a crouching frog's prominent dark eye over its lighter mouth
                        a=blob(2.3,-2.7,.98,.89);b=spot(1.7,1.0,1.8,1.0)-spot(3.4,-2.4,1.4,1.)
                        c=spot(-1.6,4.,3.1,1.3)-spot(-.9,1.4,3.,.70)
                    group.append(len(partw)//3);partw.extend([a.ravel(),b.ravel(),c.ravel()])
    layout.append(group)
# A clipped positive part is computed with two ReLUs. Every part must
# contribute for the arithmetic mean to reach the strong-detection range.
dp=linear('part_contrast',small,partw);nr=len(partw)//3
# Implement clipping as x - relu(x - cap); the signed response already
# penalizes an absent or reversed landmark, so no lower clip is needed.
dg=rs('part_triplets',dp,1,nr,3);cl=relu('excess_part_contrast',conv('part_caps',dg,np.ones((1,1,1,1)),-.85));part=add('bounded_parts',dg,conv('remove_excess',cl,-np.ones((1,1,1,1))))
parts=op('landmark_layout','avgpool',part,size=[1,3],stride=[1,3])
ng=max(map(len,layout));pick=np.zeros((len(layout)*ng,nr))
for j,ids in enumerate(layout):pick[j*ng+np.arange(ng),np.resize(ids,ng)]=1
pg=rs('landmark_poses',linear('part_groups',parts,pick.astype('f')),len(layout),1,ng);pbest=mp('landmark_evidence',pg,[1,ng],[1,ng]);
for j,s in enumerate(partnames):field(s,pbest,j,len(partnames),1)

# Bilateral symmetry of a head or frontal vehicle, tolerant of its center
# being offset. Averages of absolute differences are linear in half waves.
sy=[];sm=[]
for center in [5.5,6.5,7.5,8.5,9.5]:
    tmp=[]
    for row in range(3,13,2):
        for d in [1.5,3.5,5.5]:
            t=np.zeros((16,16));a=int(center-d);b=int(center+d);t[row:row+2,a:a+1]=.5;t[row:row+2,b:b+1]=-.5;tmp.extend([len(sy),len(sy)+1]);sy.extend([t.ravel(),-t.ravel()])
    sm.append(tmp)
ssd=relu('mirror_differences',linear('mirror_pairs',small,sy));m=np.zeros((5,len(sy)))
for i,ids in enumerate(sm):m[i,ids]=-2/len(ids)
syms=rs('mirrored_windows',linear('mirror_agreement',ssd,m,1),1,1,5);sym=mp('bilateral',syms,[1,5],[1,5]);field('sym',sym,0,1,1)

# Regional material/layout models. Each named material has a soft assignment
# cost (a sum of one-sided color/texture violations). E.g. tarmac is neutral
# and not so heavily textured; sky can be blue OR uniformly pale gray. Use the
# lowest cost compatible material in each region, not independent positive
# votes for both 'blue' and 'smooth' everywhere in the frame.
# Signed DoG/lengthwise edges here use the same photo contrast normalization.
sem=np.zeros((3,9));sem[0,-1]=1.;sem[1,2]=1;sem[1,[0,1,3]]=[-.54,-.23,-.23];sem[2,[0,2]]=.5;sem[2,[1,3]]=-.5
sloc=conv('local_structure','edge_strength',sem[:,:,None,None]);sloc=avg('material_structure',sloc,2,2)
cavg=avg('material_color',cmap,2,2);vavg=avg('material_light',lum,4)
features=list(colors)+['value','micro','horizontal','orthogonal']
# Every tuple asks for a minimum or maximum property with a tolerance/weight.
# The evaluator uses a 6x6 grid in the interior (texture filters have valid
# padding). The color grid spans the whole image; take its middle cells here.
rawparts=[];nc=len(features)
for z0,ch0,n0,rng0,offset0 in [('material_color',len(colors),8,0,1),('material_light',1,8,len(colors),1),('material_structure',3,6,len(colors)+1,0)]:
    if offset0:
        W=np.zeros((nc,ch0,3,3));W[range(rng0,rng0+ch0),range(ch0),1,1]=1
    else:
        W=np.zeros((nc,ch0,1,1));W[range(rng0,rng0+ch0),range(ch0),0,0]=1
    rawparts.append(conv('grid_'+z0,z0,W))
mr=add('local_ingredients',*rawparts)
mat={
 'sky': [('blue','>',.53,1.8),('value','>',.51,1.2),('micro','<',.10,2.0)],
 'cloud':[('white','>',.44,1.65),('value','>',.60,1.4),('micro','<',.105,1.7)],
 'mist':[('grey','>',.69,1.6),('value','>',.48,2.),('micro','<',.067,2.5)],
 'water':[('blue','>',.44,1.6),('value','<',.71,1.0),('horizontal','>',-.025,.7),('warm','<',.27,.85)],
 'greywater':[('muted','>',.70,1.7),('horizontal','>',.06,1.18),('micro','<',.19,1.1),('value','>',.23,.7)],
 'grass':[('leaf','>',.37,1.8),('olive','>',.20,.72),('blue','<',.13,1.)],
 'hay':[('olive','>',.45,1.7),('tan','>',.23,1.),('value','>',.33,.92)],
 'brush':[('leaf','>',.24,1.6),('micro','>',.13,1.9),('value','<',.53,1.4)],
 'leafblur':[('leaf','>',.37,1.8),('micro','<',.10,2.5)],
 'wood':[('tan','>',.46,1.6),('warm','>',.21,1.2),('micro','>',.11,1.6)],
 'stone':[('muted','>',.70,1.4),('micro','>',.18,2.15),('value','<',.57,.8)],
 'shade':[('value','<',.29,2.1),('micro','<',.16,1.5)],
 'road':[('grey','>',.69,1.7),('micro','<',.12,1.85),('value','>',.19,1.1),('value','<',.65,.96)],
 'concrete':[('muted','>',.75,1.32),('micro','<',.13,1.65),('value','>',.56,1.4)],
 'street':[('muted','>',.66,1.30),('orthogonal','>',.052,1.35),('micro','>',.12,1.1)],
 'paint':[('red','>',.52,1.5),('micro','<',.17,.9)],
 'lacquer':[('paintblue','>',.44,1.58),('micro','<',.17,.90)],
 'metal':[('grey','>',.70,1.4),('horizontal','>',.08,.9)],
 'pale':[('white','>',.40,1.6),('value','>',.64,1.0)],
 'rubber':[('black','>',.60,1.6),('value','<',.26,1.5)],
 'brown':[('tan','>',.50,1.37),('warm','>',.32,1.24),('value','<',.67,.8)],
 'cream':[('tan','>',.28,1.76),('value','>',.57,1.45)],
 'moss':[('leaf','>',.44,1.63),('micro','>',.14,1.35)],
 'fur':[('muted','>',.60,1.35),('micro','>',.13,1.6),('horizontal','<',.13,1.25)],
 'room':[('warm','>',.25,1.5),('value','>',.28,.7),('micro','<',.16,1.3)],
 'soft':[('micro','<',.085,2.1),('horizontal','<',.09,1.1),('orthogonal','<',.05,1.3)],
}
kk=[];bb=[];ids=[]
for constraints in mat.values():
    indices=[]
    for ch,cmp,val,mult in constraints:
        t=1 if cmp=='<' else -1;row=np.zeros(nc);row[features.index(ch)]=t*mult;indices.append(len(kk));kk.append(row);bb.append(-t*val*mult)
    ids.append(indices)
viol=relu('material_violations',conv('material_tests',mr,np.array(kk)[:,:,None,None],bb));W=np.zeros((len(mat),len(kk)))
for i,indices in enumerate(ids):W[i,indices]=-1
matmap=conv('material_quality',viol,W[:,:,None,None]);matnames=list(mat)
sets={
 'air':'sky cloud mist', 'sea':'water greywater', 'meadow':'grass hay', 'forest':'brush wood grass',
 'plant':'grass leafblur brush', 'ground':'wood moss stone', 'pavement':'road concrete', 'town':'street concrete shade',
 'finish':'paint lacquer metal pale', 'dark':'rubber shade', 'fawn':'brown cream', 'coat':'brown rubber cream',
 'pelt':'fur brown cream pale', 'pet':'fur cream brown', 'interior':'room shade concrete', 'perch':'leafblur sky soft',
 'greenfrog':'moss grass', 'dryfrog':'wood stone brown', 'birdcoat':'brown cream fur', 'neutral':'metal pale',
 'white':'pale', 'brown':'brown', 'glass':'metal shade', 'tree':'brush shade wood', 'fur':'fur', 'cream':'cream'
}
nchoice=max(len(v.split()) for v in sets.values());block=np.zeros((len(sets)*nchoice,len(mat)))
for j,v in enumerate(sets.values()):
    ix=[matnames.index(n) for n in v.split()];block[j*nchoice+np.arange(nchoice),np.resize(ix,nchoice)]=1
choices=conv('compatible_materials',matmap,block[:,:,None,None]);packed=rs('material_options',choices,len(sets),nchoice,36);mmax=mp('most_plausible_material',packed,[nchoice,1],[nchoice,1]);mm=rs('material_grid',mmax,len(sets),6,6)
# Describe where the materials are *expected*: hollow horizon/field scenes,
# central islands surrounded by sky, tall boxes, heads occupying the frame.
# X = exterior sides, U = high, L = low, C = central, M = belt, N = lower
# contact region. Generic regions contain no traced object silhouettes.
gy,gx=np.mgrid[:6,:6];gy=(gy+.5)/6;gx=(gx+.5)/6
region={
 'U':(gy<.34),'L':(gy>.66),'X':(abs(gx-.5)>.29),'C':np.exp(-((gx-.5)/.25)**4-((gy-.49)/.24)**4),
 'M':np.exp(-((gx-.5)/.36)**4-((gy-.53)/.12)**4),'N':np.exp(-((gx-.5)/.36)**4-((gy-.69)/.10)**4),
 'O':(abs(gx-.5)>.30)|(abs(gy-.5)>.35),'T':np.exp(-((gx-.50)/.31)**4-((gy-.31)/.12)**4),
 'H':np.exp(-((gx-.5)/.20)**4-((gy-.40)/.19)**4),'J':np.exp(-((gx-.5)/.23)**4-((gy-.70)/.20)**4)}
scene_drawings=[
 ['O:air C:neutral M:dark', 'U:air L:meadow M:neutral','U:air L:pavement C:neutral','O:air C:dark','U:air X:perch L:air M:neutral'],
 ['U:town L:pavement M:finish N:dark T:glass','U:tree L:pavement M:finish N:dark','L:pavement X:town C:glass M:finish N:dark','U:air L:pavement T:glass C:finish N:dark','U:town L:pavement M:dark T:glass'],
 ['O:perch C:birdcoat','O:plant C:birdcoat L:forest','U:air X:air C:dark','O:perch C:finish L:forest','O:sea C:birdcoat U:perch'],
 ['X:interior H:pelt J:pelt','X:plant H:pelt J:pelt','O:interior C:fur J:fur','O:interior C:cream J:white','X:ground H:fur J:dark'],
 ['O:forest C:fawn L:meadow','X:forest U:tree L:ground C:fawn','O:meadow C:fawn','U:tree X:ground L:ground H:fawn','U:forest L:meadow C:brown'],
 ['X:interior H:coat J:pet','X:meadow H:coat J:pet','O:interior H:white J:white','X:ground H:brown J:coat','O:pavement C:coat H:pet'],
 ['O:ground C:dryfrog','O:plant C:greenfrog','O:dark C:greenfrog','O:ground C:greenfrog','O:plant C:dryfrog'],
 ['U:air L:meadow M:coat','U:tree L:meadow C:brown','U:air L:ground C:coat','O:meadow C:dark H:brown','U:forest L:meadow C:coat'],
 ['U:air L:sea M:dark','U:air L:sea M:white','U:air X:sea L:sea C:neutral','U:forest L:sea M:white','U:town L:sea M:dark'],
 ['U:air L:pavement T:finish C:finish N:dark','U:town L:pavement C:finish N:dark','U:air L:pavement T:white C:glass J:finish','U:tree L:pavement T:neutral C:neutral','U:town L:pavement T:glass J:finish'],
];rw=[]
for descriptions in scene_drawings:
    for text in descriptions:
        W=np.zeros((len(sets),6,6));tot=0
        for part in text.split():
            rg,material=part.split(':');area=region[rg];factor={'O':2,'C':1.6,'M':1.45,'U':1.25,'L':1.4}.get(rg,1.)
            W[list(sets).index(material)]+=factor*area/area.sum();tot+=factor
        rw.append((W/tot).ravel())
lay=rs('scene_hypotheses',linear('layout_compatibility',mm,rw),10,1,5);layouts=mp('best_scene',lay,[1,5],[1,5]);layouts=rs('layout_agreement',layouts,10)

# Tires have a dark arc at road level, often with a hub of brighter metal.
# Search radial part pairs separately from the signed drawing match; require
# both wheels, their shadow line and some car body above the line. The truck
# layout has smaller tires relative to the body and a tall, flatter upper box.
rimw=[];rimix=[];rimcoeff=[];radparts={}
def savepart(v):
    v=np.asarray(v,'f');key=v.tobytes()
    if key not in radparts:radparts[key]=len(rimw);rimw.append(v.ravel())
    return radparts[key]
for typ in ['car','truck']:
    group=[]
    for center in [12,15.5,19]:
        for level in [17.5,20.5,23.5,26]:
            for rad in [1.7,2.35,3.0]:
                sep=rad*(2.84 if typ=='car' else 4.05)
                for skew in [-.13,.13]:
                    ids=[];B1=0
                    for side in [-1,1]:
                        cx=center+side*sep;cy=level+side*skew*sep
                        rr=np.sqrt(((x-cx)/(.91*rad))**2+((y-cy)/rad)**2);ring=np.exp(-.5*((rr-.84)/.32)**2);ring/=ring.sum()
                        border=np.exp(-.5*((rr-1.56)/.35)**2);border/=max(border.sum(),1.e-6);hub=gauss(cx,cy,.43*rad,.45*rad)
                        ids.append(savepart(.62*border+.38*hub-ring))
                    def gp(dx,dy,wx,hy):return gauss(center+dx,level+dy,wx,hy)
                    # The edge between dark underbody and road is upward in
                    # intensity when travelling down the photo.
                    ids.append(savepart(gp(0,1.03*rad,sep*.65,rad*.41)-gp(0,-.23*rad,sep*.65,rad*.4)))
                    ids.append(savepart(gp(0,-1.55*rad,sep*.64,rad*.52)-gp(0,-.24*rad,sep*.64,rad*.40)))
                    height=3.4 if typ=='car' else 5.0
                    ids.append(savepart(gp(0,-2.25*rad,sep*.48,.53*rad)-gp(0,-height*rad,sep*.45,.51*rad)))
                    group.append(ids)
    rimix.append(group)
wv=linear('round_parts',small,rimw);nwr=len(rimw);rg=rs('rim_response_grid',wv,1,1,nwr);ww=np.ones((1,1,1,1));over=relu('rim_clip',conv('rim_clip_level',rg,ww,-.98));rp=add('bounded_rims',rg,conv('rim_clip_excess',over,-ww));ids=np.array(rimix).reshape(-1,5);W=np.zeros((len(ids),nwr))
for i,v in enumerate(ids):np.add.at(W[i],v,[.30,.30,.15,.125,.125])
axles=rs('axle_poses',linear('wheel_chassis',rp,W),2,1,len(ids)//2);axles=mp('axle_agreement',axles,[1,len(ids)//2],[1,len(ids)//2]);
for i,tag in enumerate(['caraxle','truckaxle']):field(tag,axles,i,2,1)

# Symbolic arithmetic for the rulebook. Nothing here estimates coefficients.
class E:
    def __init__(self,items=None,c=0):self.d=items or {};self.c=c
    def __add__(self,b):
        if not isinstance(b,E):return E(self.d,self.c+b)
        d=self.d.copy()
        for k,v in b.d.items():d[k]=d.get(k,0)+v
        return E(d,self.c+b.c)
    __radd__=__add__
    def __mul__(self,v):return E({k:x*v for k,x in self.d.items()},self.c*v)
    __rmul__=__mul__
    def __sub__(self,b):return self+(-1*b)
    def __rsub__(self,b):return b+(-1*self)
    def __neg__(self):return self*(-1)
def f(ch,r='all'):return E({(ch,r):1})
H=[]
def hinge(e):H.append(e);return E({len(H)-1:1})
def cap(e,lo,hi):return (hinge(e-lo)-hinge(e-hi))*(1/(hi-lo))
# A bounded response means that an especially saturated lawn or sky does not
# count as several independent pieces of evidence. The class weights below
# are in additive evidence units on the same scale as the contour matches.
sections=[]
def section():sections.append([E() for k in range(10)])
section()
def vote(e,*w):
    assert len(w)==10
    for k,v in enumerate(w):sections[-1][k]=sections[-1][k]+v*e
blue=lambda r: f('blue',r);grass=lambda r:f('leaf',r)+.38*f('olive',r);warm=lambda r: f('warm',r)+.28*f('tan',r)
green=cap(grass('all'),.04,.56); lawn=cap(grass('bot'),.06,.66);canopy=cap(grass('rim'),.08,.59); greenobj=cap(grass('ctr')-.45*grass('rim'),.06,.48)
#                                 air   car  bird   cat  deer   dog  frog horse  ship truck
vote(green,                      -.85,-.75, .54, -.48,1.00, -.33, .85, .42,-1.18, -.85)
vote(lawn,                       -.08,-.44, .06, -.34, .70, -.15, .02, .83, -.93, -.30)
vote(canopy,                     -.38,-.32, .50, -.18, .58, -.08, .24, .27, -.64, -.32)
vote(greenobj,                   -.42,-.31, .23, -.45, .03, -.41,1.16,-.33, -.64, -.39)
aqua=cap(blue('all'),.035,.55);wet=cap(blue('bot'),.055,.56);sky=cap(blue('top'),.055,.59)
vote(aqua,                        .93, .01, .19, -.52,-.82, -.56,-.75,-.54,  .76, .03)
vote(wet,                         .70,-.39, .06, -.19,-.57, -.46,-.48,-.61, 1.47,-.40)
vote(sky,                        1.00, .00, .22, -.45,-.37, -.38,-.61, .08,  .57, .58)
# Cool sky above a lawn suggests a horse/pasture, plane on runway, or lorry;
# cool sea below a pale horizon suggests a ship; blue above AND below: flight.
vote(cap(blue('top')+grass('bot'),.65,1.3),.26,-.05,.02,-.25,.20,-.20,-.58,.76,-.16,.32)
vote(cap(blue('bot')-blue('top'),.02,.5),-.32,-.23,.02,.0,-.26,-.13,-.20,-.4,.80,-.32)
vote(cap(blue('rim')-blue('ctr'),.05,.5),.48,-.03,.18,-.1,-.18,-.1,-.16,-.13,.27,-.07)
brown=cap(warm('all'),.11,.72);tanctr=cap(warm('ctr')-.51*warm('rim'),.06,.48)
vote(brown,                      -.98,-.39, .14, .35, .52, .55, .51, .31, -.81, -.51)
vote(tanctr,                     -.40,-.19, .02, .12, .50, .55, .04, .66, -.41, -.30)
# Bright primary colors and achromatic metal in a central band.
paint=cap(f('red','ctr')+.78*f('paintblue','ctr')+.6*f('pink','ctr'),.045,.37)
vote(paint,                      -.08,1.05, .28, -.28,-.67, -.39, .15,-.52, -.06, .69)
vote(cap(f('grey','all'),.38,.88), .23, .45,-.10, .29,-.47, .04,-.46,-.33, .59, .44)
vote(cap(f('white','top'),.30,.80), .73, .01,-.03,-.13,-.32,-.17,-.61, .27, .63, .64)
vote(cap(f('white','bot'),.24,.75), .64, .28, .07, .24,-.64, .31,-.51,-.46, .25,-.06)
vote(cap(f('white','ctr')-.55*f('white','rim'),.09,.50), .21, .12,-.02, .24,-.20, .39,-.47,-.15, .13, .56)
vote(cap(f('value','top')-f('value','bot'),.08,.42), .08,-.03,-.20,-.32, .03,-.25,-.49, .29, .69, .65)
vote(cap(f('value','bot')-f('value','top'),.05,.35),-.06, .54,-.03, .11,-.05, .20, .29,-.09,-.48,-.23)
vote(cap(f('black','all'),.14,.51), -.42,-.03,-.06, .12, .03, .13, .62,-.05,-.43,-.15)
vote(cap(f('value','rim')-f('value','ctr'),.05,.33), .64, .31,.22,.01,.03,.05,-.45,.27,.10,-.08)
section()
# Roughness and orientation. Negative values of cartesian-vs-oblique energy
# are evidence of irregular biological contours; long horizontal structure
# is shared by waterlines, decks, coachwork and trailers.
hor=f('long2');ver=f('long0');ob=(f('long1')+f('long3'))*.5
axis=cap(.5*(ver+hor)-ob,-.025,.115);hdom=cap(hor-ver,.025,.38);vdom=cap(ver-hor,.025,.35)
rect=cap((hor+ver)-1.55*ob,.12,.52)
vote(axis,                       -.07, .65,-.23,-.12,-.19,-.16,-.41,-.12, .20, .89)
vote(rect,                       -.05, .47,-.13,-.14,-.15,-.18,-.30,-.09, .29, .58)
vote(hdom,                        .11, .62,-.29,-.27,-.36,-.35,-.27,-.09, .99, .34)
vote(vdom,                       -.33,-.43, .08, .08, .26, .19, .05, .09,-.63, .21)
section()
smooth=cap(f('flat','all'),.15,.66);calmrim=cap(f('flat','rim'),.22,.76);skyclear=cap(f('flat','top'),.34,.81);calmbot=cap(f('flat','bot'),.32,.76)
vote(smooth,                      .93,-.21, .63,-.13,-.34,-.17,-.53,-.20, .07,-.32)
vote(calmrim,                     .76,-.01, .59,-.08,-.28,-.09,-.63,-.12, .24,-.19)
vote(skyclear,                    .80,-.35, .16,-.35,-.20,-.31,-.52, .10, .65, .43)
vote(calmbot,                     .59, .37, .28,-.10,-.45,-.05,-.50,-.32,-.10,-.07)
vote(cap(f('flat','top')-f('flat','bot'),.12,.53),-.06,-.42,-.04,-.32, .01,-.22,-.46, .32, .85, .32)
vote(cap(f('texture','ctr')-f('texture','rim'),.07,.55),.35,.19,.54,-.12,-.19,-.11,-.24,-.12,.00,-.12)
# Grass, speckled amphibian skin and foliage have higher fine frequencies
# than the smoothly shaded surfaces of vehicles or the sky.
fine=cap(f('micro'),.10,.38);finebg=cap(f('micro','rim'),.09,.32)
vote(fine,                       -.65,-.28, .09, .20, .39, .13, .79, .12,-.19,-.23)
vote(finebg,                     -.30,-.04,-.18, .03, .42, .02, .46, .13,-.15, .00)
vote(cap(f('hline'),.82,2.15),   -.05, .44,-.32,-.28,-.26,-.29,-.28,-.07, .72, .51)
vote(cap(f('waterline')-.48*f('vline'),.40,1.62),.07,.19,-.28,-.37,-.21,-.37,-.30,-.06,.91,.33)
vote(cap(f('baseline')-.48*f('vline'),.41,1.75),-.08,.70,-.25,-.32,-.20,-.35,-.23,-.07,.52,.36)
vote(cap(f('vline')-.72*f('hline'),.30,1.5),-.29,-.21,.10,.11,.15,.13,-.06,.03,-.49,.48)
section()
# Local object parts and bilateral correspondence.
vote(cap(f('tires'),.29,.74),-.19,1.18,-.18,-.13,-.06,-.14,-.17,.00,-.21,.49)
vote(cap(f('running'),.33,.77),-.16,.53,-.18,-.16,.02,-.14,-.12,.08,-.16,.67)
vote(cap(f('canine'),.34,.77),-.11,-.09,-.05,.45,-.06,1.00,.04,.04,-.09,-.09)
vote(cap(f('feline'),.34,.77),-.10,-.08,-.02,.93,-.09,.33,.00,-.05,-.13,-.10)
vote(cap(f('haunch'),.37,.79),-.09,-.08,.21,-.04,.0,-.02,.48,-.03,-.06,-.08)
vote(cap(f('sym'),.45,.84),.20,.18,.05,.65,.08,.68,-.05,-.08,-.09,.24)
section()
# Rim evidence and a low silhouette work particularly well together. Use
# the axle separation and body height as soft alternatives, not fixed boxes.
vote(cap(f('caraxle')-.22*f('truckaxle'),.21,.68),-.26,1.55,-.19,-.23,-.12,-.13,-.17,-.03,-.22,.09)
vote(cap(f('truckaxle')-.38*f('caraxle'),.22,.59),-.19,.08,-.16,-.18,-.11,-.15,-.16,-.06,-.16,1.16)
section()
# Pairwise co-occurrence rules using raw measurements. Conjunction via a
# rectified sum: individually modest cues count when simultaneously present.
vote(cap(grass('rim')+f('flat','rim'),.62,1.15),-.17,-.19,1.13,-.16,.12,-.14,-.13,-.06,-.24,-.24)
vote(cap(grass('bot')+warm('ctr')-.40*warm('rim'),.61,1.30),-.19,-.10,-.13,-.10,.40,.07,-.04,.70,-.42,-.13)
vote(cap(grass('all')+f('micro','ctr')*1.3,.58,1.07),-.20,-.15,-.03,-.13,.19,-.17,.82,-.13,-.32,-.18)
vote(cap(f('grey','ctr')+f('sym')*.65,.80,1.32),-.06,.03,-.04,.70,-.17,.12,-.21,-.22,-.01,.03)
vote(cap(f('tan','ctr')+.64*f('sym'),.68,1.20),-.11,-.06,-.13,.01,-.02,.70,-.02,.06,-.11,-.1)
vote(cap(blue('bot')+.56*f('flat','top'),.49,1.04),.17,-.09,-.02,-.09,-.14,-.14,-.19,-.20,.78,-.09)

# Some priors are implicit in the number of positive rules: center each
# bounded rule halfway between absence and presence, with sparser cues
# centered nearer absence. These reference points describe the semantics
# of each detector, not its activation frequency on a collection of images.
# Do this cue by cue below using analytical reference signals: an unsaturated
# gray/tan outdoor snapshot with weak grass, some structure, no salient parts.
ref={'blue':.10,'leaf':.10,'olive':.09,'warm':.14,'tan':.18,'red':.038,'paintblue':.024,'pink':.027,'grey':.55,'white':.15,'black':.15,'value':.46,'long0':.42,'long2':.46,'long1':.42,'long3':.42,'flat':.32,'micro':.16,'texture':.42,'hline':1.2,'waterline':1.22,'baseline':1.1,'vline':1.06,'sym':.54,'tires':.49,'running':.49,'canine':.49,'feline':.49,'haunch':.50,'caraxle':.42,'truckaxle':.42}
def reference(k):
    ch,r=k;v=ref[ch]
    if (ch,r)==('value','top'):v+=.07
    if (ch,r)==('value','ctr'):v-=.035
    if (ch,r) in [('blue','top'),('white','top')]:v+=.06
    if (ch,r)==('flat','top'):v+=.09
    if (ch,r)==('flat','rim'):v+=.06
    if (ch,r)==('leaf','bot'):v+=.05
    if (ch,r) in [('red','ctr'),('warm','ctr')]:v+=.025
    return v
hv=[max(0,e.c+sum(v*reference(k) for k,v in e.d.items())) for e in H]
score=[]
for sub in sections:
    for e in sub:score.append(e-(e.c+sum(v*hv[k] for k,v in e.d.items())))
# Convert the symbolic evidence computation into ordinary allowed layers.
keys=sorted(set(k for e in H for k in e.d));n=len(keys);pieces={}
for i,(ch,r) in enumerate(keys):
    name,ic,nc,side=fields[ch]
    if name not in pieces:pieces[name]=np.zeros((n,nc,side,side),'f')
    pieces[name][i,ic]=areas(side,r)
measure=[linear('measure_'+s,s,W.reshape(n,-1)) for s,W in pieces.items()];base=add('scene_measurements',*measure)
U=np.zeros((len(H),n));b=[]
for i,e in enumerate(H):
    b.append(e.c)
    for k,v in e.d.items():U[i,keys.index(k)]=v
rules=relu('rule_responses',linear('rule_conditions',base,U,b));U=np.zeros((len(score),len(H)));b=[]
for i,e in enumerate(score):
    b.append(e.c)
    for k,v in e.d.items():U[i,k]=v
# Saturation within each cue family prevents counting strongly correlated
# facts ('blue', 'blue sky', 'blue lower surround') as independent evidence.
# Scale by how much testimony the family itself contains. Caps are evidence
# budgets, equal for all classes, rather than corrections to class priors.
families=linear('cue_families',rules,U,b);limits=np.repeat([2.15,.79,1.75,1.06,1.1,1.06],10);d=len(limits)
bounds=relu('family_bounds',linear('family_limits',families,np.r_[np.eye(d),np.eye(d)],np.r_[limits,-limits]));combine=np.tile(np.eye(10),len(sections));scene=linear('scene_evidence',bounds,np.c_[combine,-combine],-combine@limits)
# An explicitly hierarchical readout: record whether it looks manufactured
# or animate, then discriminate members using the degrees of freedom *within*
# each family. A grassy habitat for instance shouldn't decide deer over horse
# nearly as strongly as it decides deer over ship.
transport=[0,1,8,9];living=[2,3,4,5,6,7];groupmat=np.zeros((10,10))
for ids in [transport,living]:groupmat[np.ix_(ids,ids)]=1/len(ids)
inside=np.eye(10)-groupmat
final=add('scores',linear('habitat_testimony',scene,.73*groupmat+.56*inside),linear('geometry_testimony',model,1.73*inside+1.32*groupmat),linear('layout_testimony',layouts,4.2*np.eye(10)))
(out/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n');np.savez(out/'weights.npz',**weights)

