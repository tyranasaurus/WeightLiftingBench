"""A data-free model of the faces and legends of German road signs.
Only the Python standard library and numpy are needed. There are no asset files.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

F = np.float32
layers = []
weights = {}


def node(name, op, sources, **args):
    layers.append(dict(name=name, op=op, inputs=sources if isinstance(sources,list) else [sources], **args))
    return name


def conv(name, inp, w, b=None, padding=0):
    w = np.asarray(w, dtype=F)
    if w.ndim == 2: w = w[:,:,None,None]
    n, _, k, _ = w.shape
    weights[name+'.W'] = w
    weights[name+'.b'] = np.broadcast_to(np.asarray(0 if b is None else b, dtype=F), (n,)).copy()
    return node(name, 'conv', inp, out=n, kernel=[k,k], stride=1, padding=padding)


def dense(name, inp, w):
    w = np.asarray(w,dtype=F).reshape(len(w),-1)
    weights[name+'.W'] = w
    weights[name+'.b'] = np.zeros(len(w),F)
    return node(name,'dense',inp,out=len(w))


def crop(name, inp, c,h,w, y,x, hh,ww):
    # Just select elements; unlike a padded convolution this doesn't look outside the patch.
    idx=np.arange(c*h*w).reshape(c,h,w)[:,y:y+hh,x:x+ww].reshape(-1)
    m=np.zeros((len(idx),c*h*w),F); m[np.arange(len(idx)),idx]=1
    return node(name,'reshape',dense(name+'_select',inp,m),shape=[c,hh,ww])


def pool(name, inp, op, h,w, sh,sw):
    return node(name,op,inp,size=[h,w],stride=[sh,sw])


# The patterns below are drawn in normalized sign coordinates. x is right, y is down;
# outer edges of a frontal sign are approximately +/-1. All primitive coordinates
# refer to this drawing board, not to any photographed sign.
OUT,WHITE,RED,BLACK,BLUE,YELLOW,GRAY,GREEN = range(8)
# An ideal reflectance chart, not sampled colours. The second chart is just a set
# of opponent-colour coefficients for distinguishing hue from neutral shading.
LUM=np.array([.38,.96,.21,.035,.22,.81,.53,.37])
PAL_R=np.array([0,0,1,0,-.36,.27,0,-.45])
PAL_B=np.array([0,0,-.33,0,1,-.50,0,-.28])
PAL_Y=np.array([0,0,.22,0,-.55,1,0,.10])

def polygon(x,y,pts):
    inside=np.zeros(x.shape,bool)
    x0,y0=pts[-1]
    for x1,y1 in pts:
        if y1!=y0:
            inside ^= ((y1>y)!=(y0>y)) & (x < x1+(x0-x1)*(y-y1)/(y0-y1))
        x0,y0=x1,y1
    return inside


def polyline(x,y,pts,width):
    m=np.zeros(x.shape,bool)
    for (a,b),(c,d) in zip(pts,pts[1:]):
        u=c-a;v=d-b
        t=np.clip(((x-a)*u+(y-b)*v)/(u*u+v*v+1.e-15),0,1)
        m |= (x-a-u*t)**2+(y-b-v*t)**2 < (width*.5)**2
    return m


def bezier(a,b,c,d,n=8):
    t=np.linspace(0,1,n+1)[:,None];z=1-t
    return z**3*np.array(a)+3*t*z*z*np.array(b)+3*z*t*t*np.array(c)+t**3*np.array(d)


def ellipse(x,y,a,b,rx,ry):
    return ((x-a)/rx)**2+((y-b)/ry)**2<=1


class Ink:
    def __init__(self,x,y,img): self.x=x;self.y=y;self.img=img
    def paint(self,m,col): self.img[m]=col
    def poly(self,p,c=BLACK): self.paint(polygon(self.x,self.y,p),c)
    def line(self,p,w,c=BLACK): self.paint(polyline(self.x,self.y,p,w),c)
    def arc(self,a,b,c,d,w,col=BLACK): self.line(bezier(a,b,c,d),w,col)
    def dot(self,a,b,rx,ry=None,c=BLACK): self.paint(ellipse(self.x,self.y,a,b,rx,rx if ry is None else ry),c)
    def box(self,x0,y0,x1,y1,c=BLACK): self.paint((self.x>=x0)&(self.x<=x1)&(self.y>=y0)&(self.y<=y1),c)
    def local(self,cx,cy,sx,sy): return Ink((self.x-cx)/sx,(self.y-cy)/sy,self.img)


def numerals(p,s,variant):
    # Hand-lettered open terminals, rounded bowls, flat-ended DIN-like strokes.
    # Coordinates of each stroke are in a unit-square character cell.
    heights=[.85,.79,.91,.81,.88,.83]
    widths=[.43,.44,.39,.49,.45,.405] if len(s)==2 else [.355,.36,.33,.385,.355,.33]
    h=heights[variant%6];w=widths[variant%6]
    gap=(.09,.115,.10,.075,.10,.085)[variant%6]
    cells=[w*(.67 if ch=='1' else 1.) for ch in s]
    left=-(sum(cells)+gap*(len(s)-1))/2
    for i,ch in enumerate(s):
        q=p.local(left, -.5*h, cells[i],h)
        # Draw in physical coordinates relative to glyph cell for nearly uniform line weight.
        thick=h*(.136,.148,.12,.148,.15,.13)[variant%6]
        def path(a):
            pts=np.asarray(a)*[cells[i],h]+[left,-h/2]
            p.line(pts,thick)
        def curve(a,b,c,d):return bezier(a,b,c,d)
        def contour(seq):path(np.concatenate(seq))
        if ch=='0':
            contour([curve((.5,.065),(.18,.065),(.115,.19),(.115,.48)),
                     curve((.115,.48),(.115,.78),(.14,.935),(.5,.935)),
                     curve((.5,.935),(.85,.935),(.885,.77),(.885,.48)),
                     curve((.885,.48),(.885,.20),(.83,.065),(.5,.065))])
        elif ch=='1':path([(.09,.26),(.67,.075),(.67,.935)])
        elif ch=='2':
            contour([curve((.11,.245),(.16,-.005),(.85,.015),(.89,.245)),
                     curve((.89,.245),(.95,.43),(.58,.56),(.12,.92)),np.array([(.12,.92),(.90,.92)])])
        elif ch=='3':
            contour([curve((.135,.185),(.41,-.045),(.90,.065),(.87,.255)),
                     curve((.87,.255),(.86,.39),(.71,.455),(.47,.475))]);
            contour([curve((.47,.475),(.76,.46),(.915,.555),(.90,.705)),
                     curve((.90,.705),(.87,.955),(.41,1.015),(.11,.845))])
        elif ch=='5':
            path([(.86,.075),(.195,.075),(.155,.465)])
            contour([curve((.155,.465),(.65,.275),(.915,.46),(.895,.68)),
                     curve((.895,.68),(.89,.965),(.40,1.015),(.115,.84))])
        elif ch=='6':
            contour([curve((.73,.06),(.42,.30),(.125,.46),(.12,.66)),
                     curve((.12,.66),(.07,1.01),(.90,1.045),(.89,.70)),
                     curve((.89,.70),(.90,.42),(.43,.385),(.17,.57))])
        elif ch=='7':path([(.09,.075),(.90,.075),(.34,.935)])
        elif ch=='8':
            for ya,yb,rx in [(.06,.475,.35),(.465,.94,.395)]:
                t=np.linspace(0,2*np.pi,28)
                path(np.stack([.5+rx*np.cos(t),(ya+yb)/2+(yb-ya)/2*np.sin(t)],1))
        left+=cells[i]+gap


def car_front(p,col=BLACK):
    p.poly([(-.94,.82),(-.95,-.03),(-.78,-.24),(-.59,-.82),(-.40,-.93),(.47,-.93),(.64,-.82),(.82,-.23),(.96,-.03),(.96,.82),(.68,.82),(.67,.99),(.43,.99),(.40,.80),(-.45,.80),(-.47,.99),(-.72,.99),(-.73,.80)],col)
    p.poly([(-.65,-.24),(-.45,-.68),(.46,-.68),(.66,-.24)],WHITE)


def lorry(p,col=BLACK,front=False):
    if front:
        p.box(-.85,-1,.78,.74,col);p.box(-.72,.60,-.45,.98,col);p.box(.40,.60,.68,.98,col)
        p.box(-.68,-.05,.62,.16,WHITE)
    else:
        p.box(-.96,-.72,.31,.44,col);p.poly([(.36,-.49),(.70,-.49),(.97,-.08),(.97,.55),(-.95,.55),(-.95,.36),(.36,.36)],col)
        p.poly([(.49,-.36),(.65,-.36),(.81,-.11),(.49,-.11)],WHITE)
        p.dot(-.59,.55,.20,.32,col);p.dot(.64,.55,.20,.32,col)


def cars(p,truck=False):
    if truck: lorry(p.local(-.255,-.085,.255,.38),RED,True)
    else: car_front(p.local(-.272,-.018,.252,.218),RED)
    car_front(p.local(.255,.082,.234,.205))


def walk(p, col=BLACK):
    p.dot(.07,-.80,.143,.18,col)
    p.line([(.035,-.52),(-.055,-.20),(-.12,.09)],.23,col)
    p.line([(.00,-.42),(.16,-.22),(.40,-.17)],.094,col)
    p.line([(-.04,-.40),(-.24,-.24),(-.34,.03)],.098,col)
    p.line([(-.095,.06),(.04,.45),(.30,.79)],.148,col)
    p.line([(-.08,.055),(-.20,.44),(-.46,.70)],.154,col)


def warnings(p,k,v):
    # The icon is in the bottom, wide part of a triangular face.
    if k==18:
        p.poly([(-.083,-.47),(.083,-.47),(.05,.26),(-.05,.26)])
        p.dot(0,.44,.09)
    elif k in (19,20,21):
        q=p.local(0,0,1 if k==20 or (k==21 and v%3==2) else -1,1)
        if k==21:
            q.line([(-.19,.55),(-.19,.27),(.14,.015),(.14,-.14),(-.10,-.32)],.155)
        else:
            q.line([(-.22,.54),(-.22,.24)],.16)
            q.arc((-.22,.25),(-.22,-.06),(-.057,-.14),(.215,-.14),.16)
            q.line([(.215,-.14),(.27,-.14)],.16)
    elif k==22:
        p.poly([(-.62,.53),(-.53,.46),(-.46,.30),(-.37,.235),(-.25,.25),(-.15,.39),(-.08,.41),(.015,.29),(.09,.18),(.21,.15),(.31,.205),(.45,.42),(.59,.51),(.62,.57),(-.62,.57)])
    elif k==23:
        q=p.local(.09,-.095,.245,.25)
        # skidding car, tilted a little with respect to its sinuous tracks
        z=Ink(q.x+.25*q.y,q.y-.20*q.x,q.img);car_front(z)
        for dx in [-.16,.16]:
            p.arc((dx+.06,.105),(dx-.40,.24),(dx+.24,.32),(dx-.12,.47),.058)
            p.arc((dx-.12,.47),(dx-.19,.50),(dx-.25,.52),(dx-.28,.57),.058)
    elif k==24:
        p.line([(-.215,.55),(-.215,-.22)],.091)
        p.line([(.26,.55),(.26,.26),(.115,.015),(.115,-.22)],.097)
    elif k==25:
        p.dot(.02,-.27,.10);p.line([(-.02,-.12),(-.205,.065),(-.21,.20)],.17)
        p.line([(-.18,.145),(-.34,.36),(-.45,.54)],.108)
        p.line([(-.19,.14),(-.08,.32),(-.12,.54)],.115)
        p.line([(-.04,-.07),(.025,.09),(.18,.15)],.089)
        p.line([(-.18,.01),(-.12,.16),(.015,.18)],.075)
        p.line([(-.29,-.08),(.42,.47)],.038)
        p.poly([(.18,.55),(.36,.34),(.44,.32),(.63,.55)])
    elif k==26:
        p.poly([(-.123,-.27),(-.123,.45),(-.075,.52),(.077,.52),(.123,.45),(.123,-.27),(.073,-.35),(-.073,-.35)])
        for yy,c in [(-.191,RED),(.082,YELLOW),(.35,GREEN)]:p.dot(0,yy,.084,c=c)
    elif k==27:walk(p.local(0,.12,.65,.49))
    elif k==28:
        # Two running children holding hands. The larger one leads.
        q=p.local(0,.10,-1 if v%3!=2 else 1,1)
        q.dot(.175,-.31,.090);q.dot(-.218,-.11,.079)
        q.line([(.135,-.18),(.07,-.01),(.105,.09)],.14)
        q.line([(.13,-.15),(.29,-.025),(.41,-.09)],.07)
        q.line([(.11,-.14),(-.012,-.025),(-.12,.008)],.07)
        q.line([(.10,.08),(.29,.20),(.42,.36)],.087)
        q.line([(.10,.08),(-.025,.27),(-.135,.32)],.09)
        q.poly([(-.24,-.01),(-.175,-.015),(-.135,.22),(-.33,.22)])
        q.line([(-.21,.025),(-.105,.005),(-.045,-.02)],.062)
        q.line([(-.23,.02),(-.33,.12),(-.43,.085)],.058)
        q.line([(-.245,.20),(-.33,.37),(-.43,.39)],.07);q.line([(-.21,.20),(-.14,.30),(-.09,.41)],.07)
    elif k==29:
        for a in [-.31,.315]:
            p.dot(a,.32,.188);p.dot(a,.32,.139,c=WHITE)
        p.line([(-.31,.32),(-.15,.05),(.01,.32),(-.31,.32)],.044)
        p.line([(.01,.32),(.225,.055),(-.15,.05),(-.195,-.05)],.044)
        p.line([(.315,.32),(.194,-.07),(.125,-.09)],.052)
        p.dot(.005,-.27,.081);p.line([(-.035,-.15),(-.18,-.01),(-.01,.105),(-.018,.27)],.082)
        p.line([(-.025,-.13),(.078,-.03),(.20,-.034)],.057)
    elif k==30:
        for ang in np.arange(6)*np.pi/3-np.pi/2:
            c,s=math.cos(ang),math.sin(ang);q=Ink(p.x*c+(p.y-.135)*s,-p.x*s+(p.y-.135)*c,p.img)
            q.line([(0,0),(.43,0)],.049)
            for r in [.17,.285]:
                for dy in [-1,1]:q.line([(r,0),(r+.105,dy*.105)],.042)
    elif k==31:
        # Leaping deer, head facing left; fine detail is less certain than the pose.
        q=p.local(0,.12,(-1 if v%3==2 else 1),1)
        q.poly([(-.42,-.19),(-.38,-.26),(-.26,-.26),(-.19,-.15),(.10,-.09),(.27,-.01),(.42,.12),(.44,.22),(.33,.24),(.17,.12),(-.04,.11),(-.22,.04),(-.28,-.12)])
        q.line([(-.16,.025),(-.24,.16),(-.42,.18),(-.49,.31)],.058)
        q.line([(-.105,.065),(-.13,.21),(-.34,.27),(-.37,.36)],.049)
        q.line([(.30,.135),(.22,.28),(.44,.34),(.50,.43)],.068)
        q.line([(.36,.18),(.35,.29),(.53,.36)],.048)
        q.line([(-.28,-.22),(-.20,-.33),(-.22,-.42)],.034);q.line([(-.255,-.25),(-.35,-.35),(-.36,-.44)],.028)


def arrow(p, kind, mirror, variant):
    q=p.local(0,0,-1 if mirror else 1,1)
    if kind=='keep':
        # Rotate the straight-ahead arrow clockwise by 135 degrees.
        c=-2**-.5;s=2**-.5
        q=Ink(c*q.x+s*q.y,-s*q.x+c*q.y,q.img)
        q.poly([(-.147,.67),(.147,.67),(.147,-.076),(.38,-.076),(0,-.72),(-.38,-.076),(-.147,-.076)])
    elif kind=='up':
        q.poly([(-.105,.68),(.105,.68),(.105,-.23),(.37,-.035),(.435,-.14),(0,-.70),(-.435,-.14),(-.37,-.035),(-.105,-.23)])
    elif kind=='turn':
        q.line([(-.35,.65),(-.35,.18)],.20,WHITE)
        q.arc((-.35,.18),(-.35,-.22),(-.16,-.30),(.22,-.30),.20,WHITE)
        q.poly([(.13,-.62),(.63,-.30),(.13,.02)],WHITE)
    elif kind=='both':
        q.poly([(-.13,.68),(.075,.68),(.075,-.29),(.275,-.29),(-.032,-.70),(-.35,-.29),(-.13,-.29)],WHITE)
        q.poly([(.015,.18),(.37,.18),(.37,.365),(.71,.075),(.37,-.225),(.37,-.035),(.015,-.035)],WHITE)


def roundabout(p):
    # Three tangential arrows around a central blue hole, counter-clockwise.
    for a in [np.pi/2,np.pi/2+2*np.pi/3,np.pi/2+4*np.pi/3]:
        t=np.linspace(a,a-1.44,16);p.line(np.stack([.50*np.cos(t),.50*np.sin(t)],1),.125,WHITE)
        b=t[-1];n=np.array([np.cos(b),np.sin(b)]);d=np.array([np.sin(b),-np.cos(b)])
        o=.50*n;p.poly([o+.18*n-.055*d,o+.25*d,o-.18*n-.055*d],WHITE)


def stopword(p):
    for n,ch in enumerate('STOP'):
        q=p.local((n-1.5)*.347,0,.14,.24)
        if ch=='S':q.line([(.80,-.79),(.47,-.92),(-.42,-.92),(-.82,-.65),(-.82,-.27),(-.43,-.07),(.51,.10),(.82,.33),(.82,.65),(.40,.91),(-.44,.91),(-.81,.75)],.29,WHITE)
        if ch=='T':q.line([(-.84,-.89),(.84,-.89)],.29,WHITE);q.line([(0,-.84),(0,.90)],.32,WHITE)
        if ch=='O':q.line([(-.81,-.57),(-.48,-.90),(.43,-.90),(.81,-.57),(.81,.57),(.46,.90),(-.48,.90),(-.81,.57),(-.81,-.57)],.29,WHITE)
        if ch=='P':q.line([(-.65,.89),(-.65,-.89),(.41,-.89),(.79,-.57),(.79,-.19),(.43,.06),(-.65,.06)],.30,WHITE)


SPEED={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}
END={6,32,41,42}

def grp(k):
    if k in END:return 'ended'
    if k<=10 or k in (15,16):return 'ring'
    if k==11 or 18<=k<=31:return 'warn'
    if 33<=k<=40:return 'blue'
    return {12:'diamond',13:'yield',14:'stop',17:'entry'}[k]


def triangle(x,y,t):
    # t is border inset; approximate a rounded triangular plate by a Minkowski sum.
    r=.083
    top=-.994+2*t;bot=.984-t;side=.984*(bot-top)/1.978
    u=[(0,top+2*r),(side-1.7*r,bot-r),(-side+1.7*r,bot-r)]
    return polygon(x,y,u)|polyline(x,y,u+u[:1],2*r)


def plate(x,y,k,v, symbol=True):
    fam=grp(k);img=np.zeros(x.shape,np.uint8);p=Ink(x,y,img)
    rad=x*x+y*y
    if fam in ('ring','ended','entry','blue'):
        img[rad<=1]=WHITE
        if fam=='ring':
            img[rad<.978**2]=RED;img[rad<( .782,.806,.752)[v%3]**2]=WHITE
        elif fam=='entry':img[rad<=.975**2]=RED
        elif fam=='blue':img[rad<=.973**2]=BLUE
        else:img[rad<=.98**2]=GRAY;img[rad<=.945**2]=WHITE
    elif fam in ('warn','yield'):
        yy=-y if fam=='yield' else y
        img[triangle(x,yy,0)]=WHITE;img[triangle(x,yy,.018)]=RED
        img[triangle(x,yy,(.16,.188,.142)[v%3] if fam=='warn' else (.184,.215,.154)[v%3])]=WHITE
    elif fam=='diamond':
        r=np.abs(x)+np.abs(y);img[r<=1.03]=BLACK;img[r<.99]=WHITE;img[r<.695]=YELLOW
    elif fam=='stop':
        r=np.maximum.reduce([np.abs(x),np.abs(y),(np.abs(x)+np.abs(y)) / math.sqrt(2)])
        img[r<1]=WHITE;img[r<.933]=RED
    if not symbol:return img
    # Different part sizes, sharing a common coordinate frame for face and part.
    factor=(1,.91,1.07,1.0,.95,1.06)[v%6]
    q=p.local(0,(0,-.025,.022,.014,-.030,.005)[v%6],factor,factor)
    if k in SPEED:numerals(p,SPEED[k],v)
    if k in (9,10,41,42): cars(q,k in (10,42))
    if k==16:lorry(q.local(0,-.01,.64,.38))
    if k==11:
        q.poly([(-.074,.56),(.074,.56),(.074,.17),(.38,.17),(.38,.086),(.074,.086),(.074,-.29),(-.074,-.29),(-.074,.086),(-.38,.086),(-.38,.17),(-.074,.17)])
        q.poly([(-.145,-.22),(0,-.44),(.145,-.22)])
    if k==14: stopword(q)
    if k==17:q.box(-.72,-.153,.72,.153,WHITE)
    if 18<=k<=31:warnings(q,k,v)
    if 33<=k<=39:
        if k in (35,38,39):
            # These polygons, unlike the curved arrow, use the foreground as default ink.
            old=q.img.copy();arrow(q, 'up' if k==35 else 'keep',k==39,v);q.img[q.img!=old]=WHITE
        else:arrow(q,'turn' if k in (33,34) else 'both',k in (34,37),v)
    if k==40:roundabout(q)
    if k in END:
        img[(img==BLACK)|(img==RED)]=GRAY
        # Five parallel cancelling strokes form a wide rising diagonal band.
        t=(x+y*(1,.88,1.12)[v%3])/math.sqrt(2)
        for d in np.arange(-2,3)*.095:
            img[(np.abs(t-d)<.032)&(rad<.973**2)]=BLACK
    return img


# Photometric front end. Approximate log(.02 + channel) with linear splines.
# It increases the resolution of differences in dark/shaded faces without making
# bright background sky determine the scale of symbols on the sign.
def front_end():
    knots=np.array([0,.035,.10,.24,.52,1.0]); slopes=np.diff(np.log(.02+knots))/np.diff(knots)
    hinge=conv('light_breaks','image',np.repeat(np.eye(3),len(slopes),axis=0),-np.tile(knots[:-1],3))
    pos=node('light_positive','relu',hinge)
    mat=np.zeros((3,3*len(slopes)),F)
    for j in range(3):mat[j,j*len(slopes):(j+1)*len(slopes)]=np.diff(np.r_[0,slopes])
    light=conv('log_light',pos,mat)
    lum=conv('luminance',light,[[.29,.51,.20]])
    # A neutral pixel has zero opponent response at every brightness. Favour a
    # red-green axis for red, and a blue-red axis for blue (camera shadows can be blue).
    hue=conv('opponents',light,[[1,-.80,-.20],[-.75,-.25,1],[.49,.51,-1]])
    # Contextual opponent patterns should retain their sign when the chroma is weak.
    # Floors on the per-channel std are implemented by scaling before 'norm'.
    hc=conv('hue_floor',hue,.019*np.eye(3));hn=node('hue_pattern','norm',hc)
    sign=conv('joint_letters',light,[[.27,.52,.21]])
    # Colour-specific interior intensity: the red and blue rings should both
    # contrast with the white centre, and black text should contrast with white.
    lc=conv('joint_floor',sign,[[.14]]);ln=node('face_intensity','norm',lc)
    # A weak high-pass component removes broad shadows across the interior. Padding
    # is never used by the icon matcher, since that part is cropped out afterwards.
    u=np.arange(-3,4);g=np.exp(-u**2/(2*1.55**2));g=g/g.sum()
    high=-np.outer(g,g);high[3,3]+=1
    hp=conv('local_ink',lum,high[None,None],padding=3)
    return lum,hn,ln,hp


def spread(name, sources, matrices):
    return node(name,'add',[conv(name+'_'+str(n),s,m) for n,(s,m) in enumerate(zip(sources,matrices))])


def local_normalize(inp,name,hh,ww,y0,x0,h,w,gain):
    patch=crop(name+'_area',inp,1,hh,ww,y0,x0,h,w)
    patch=conv(name+'_floor',patch,[[gain]])
    return node(name,'norm',patch)

def sg(a,sigma):
    if sigma==0:return a
    r=max(1,int(math.ceil(2.5*sigma)));u=np.arange(-r,r+1);w=np.exp(-.5*(u/sigma)**2);w/=w.sum()
    p=np.pad(a,((0,0),(r,r)),mode='edge');f=sum(v*p[:,i:i+a.shape[1]] for i,v in enumerate(w))
    p=np.pad(f,((r,r),(0,0)),mode='edge');return sum(v*p[i:i+a.shape[0]] for i,v in enumerate(w))


def ssavg(a,ss):
    h,w=a.shape;return a.reshape(h//ss,ss,w//ss,ss).mean((1,3))


def normalize_component(arr,mask=None):
    # Exactly one channel of one template. Remove DC locally, and use L2 rather
    # than L1 to prevent a tiny coincidental pixel match outweighing large icons.
    if mask is None:r=arr-arr.mean()
    else:r=mask*(arr-(mask*arr).sum()/max(1.e-10,mask.sum()))
    n=np.linalg.norm(r)
    return r/n if n>1.e-5 else np.zeros_like(r)


# The pose library is a deterministic quadrature of plausible sign poses, not
# sampled or selected using photographs. Center translations are handled below.
# Entries are relative width, relative height, rotation (deg), horizontal shear,
# vertical foreshortening gradient, and icon/font style.
POSE=[(1,1,0,0,0,0),(.94,.98,-3,-.035,.04,1),(1.02,1.01,3,.025,-.055,2),
      (.91,1.025,2,.045,.08,3),(1.035,.946,-2,-.04,-.09,4),(.984,1.,0,0,0,5)]
RADII=[10.15,11.8,13.4,14.65]
DX=[-1.65,0,1.65];DY=[-1.65,0,1.65]
# Adjacent poses use interleaved subpixel locations for the fine-detail branch.
def offsets(v):
    phase=[(0,0),(-.35,-.45),(.45,.35),(-.4,.35),(.35,-.45),(0,0)][v]
    return [(dx+phase[0],dy+phase[1]) for dy in DY for dx in DX]
S=3;H=44
axis=(np.arange(H*S)+.5)/S-H/2
PX,PY=np.meshgrid(axis,axis)

def coordinates(radius,pose):
    sx,sy,deg,shear,persp,v=pose
    c=math.cos(math.radians(deg));s=math.sin(math.radians(deg))
    y=(-s*PX+c*PY)/(radius*sy)
    x=((c*PX+s*PY)/(radius*sx)-shear*y)/(1+persp*y)
    return x,y


def translate_samples(a,n,dx,dy,step=1):
    # Sample the canonical antialiased drawing on the input pixel/cell grid.
    pts=(np.arange(n)+.5-n/2)*step+(H-1)/2
    xs=pts-dx;ys=pts-dy
    x0=np.floor(xs).astype(int);y0=np.floor(ys).astype(int);fx=xs-x0;fy=ys-y0
    a0=a[y0[:,None],x0[None,:]]*(1-fx)+a[y0[:,None],(x0+1)[None,:]]*fx
    a1=a[(y0+1)[:,None],x0[None,:]]*(1-fx)+a[(y0+1)[:,None],(x0+1)[None,:]]*fx
    return a0*(1-fy[:,None])+a1*fy[:,None]


def interior(x,y,k):
    r=np.sqrt(x*x+y*y);fam=grp(k)
    if fam=='warn':return triangle(x,y,.223)
    if fam=='yield':return triangle(x,-y,.248)
    if fam=='ring':return r<.735
    if fam=='blue':return r<.834
    if fam=='diamond':return (abs(x)+abs(y))<1.04
    if fam=='stop':return (np.maximum(abs(x),abs(y))<.80)&((abs(x)+abs(y))<1.22)
    return r<.85


def face_mask(x,y,k):
    # Outside the plate the colour of the world is unknown. Include only a narrow
    # neighbourhood of the perimeter (not corners filled with sky, trees, etc.).
    f=grp(k)
    if f in ('ring','blue','entry','ended'):return x*x+y*y<1.155**2
    if f in ('warn','yield'):return triangle(x,y if f=='warn' else -y,-.092)
    if f=='diamond':return abs(x)+abs(y)<1.19
    return (np.maximum(abs(x),abs(y))<1.12)&((abs(x)+abs(y))<1.57)


def face_template(ink,face,dx,dy,k,v):
    # The backdrop has no specified reflectance. Only use a modest outer contrast.
    n=16;fam=grp(k)
    def at(a):return translate_samples(a,n,dx,dy,2.)
    m=at(face)
    # antialiasing and the scorer's 2x2 average on the image side
    lm,r,b,y=map(at,ink)
    ls=normalize_component(lm,m)
    cols=[normalize_component(a,m) for a in (r,b,y)]
    # Group colour weights specified from the meaning of the sign. Include the
    # intensity of the face even for strongly coloured signs, for colour casts.
    h={ 'ring':(.80,.20,0), 'warn':(.80,.20,0), 'yield':(.80,.20,0),
        'blue':(.26,.74,0), 'diamond':(.12,.10,.78), 'stop':(.80,.20,0),
        'entry':(.80,.20,0), 'ended':(0,0,0)}[fam]
    if fam=='ended':
        # cancellation signs have NO red ring. Counterevidence is a formulaic
        # surround filter, not a statistic of the other classes' templates.
        t=(np.arange(n)+.5-n/2)*2;xx,yy=np.meshgrid(t-dx,t-dy)
        rr=np.sqrt(xx*xx+yy*yy);rad=v
        ring=np.exp(-.5*((rr-.89*rad)/(.10*rad))**2);red=-normalize_component(ring,m)
        return np.stack([.88*ls,.29*red,0*red,0*red])/np.sqrt(m.sum().clip(1))
    return np.stack([.56*ls]+[.53*coef*a for coef,a in zip(h,cols)])/np.sqrt(m.sum().clip(1))


def whole_drawing(k,radius,pose):
    x,y=coordinates(radius,pose);v=pose[-1];d=plate(x,y,k,v)
    fm=ssavg(face_mask(x,y,k),S)
    # Blurring here is spatial filtering of this drawing alone.
    inks=[sg(ssavg(chart[d],S),.76) for chart in (LUM,PAL_R,PAL_B,PAL_Y)]
    return [face_template(inks,fm,dx,dy,k,radius) for dy in DY for dx in DX]


def make_legend_maps(lum,hp):
    maps=[lum,hp];crops=[]
    for num,src in enumerate(maps):
        core=crop('core'+str(num),src,1,32,32,6,6,20,20)
        glob=node('context'+str(num),'norm',conv('contrast_floor'+str(num),core,[[.16 if num==0 else .25]]))
        crops.append(glob)
        # The narrow pictograms don't occupy the whole 20x20 patch. Compute a
        # second std over the most likely symbol region, rather than normalizing
        # the symbol against the large contrast of an irrelevant edge.
        loc=local_normalize(core,'inner'+str(num),20,20,6,4,12,12,.12 if num==0 else .16)
        w=np.zeros((400,144),F);i=np.arange(400).reshape(20,20)[6:18,4:16].ravel()
        # A small raised-cosine transition avoids introducing artificial ink edges.
        t=np.arange(12)+.5;rim=np.minimum(1,np.minimum(t,12-t)/1.8)
        w[i,np.arange(144)]=(.44*(rim[:,None]*rim)).ravel()
        lift=node('inner_map'+str(num),'reshape',dense('inner_scatter'+str(num),loc,w),shape=[1,20,20])
        keep=np.eye(400,dtype=F);keep[i,i]-=w[i,np.arange(144)]
        outer=node('outer_map'+str(num),'reshape',dense('context_weight'+str(num),glob,keep),shape=[1,20,20])
        maps[num]=node('blended_detail'+str(num),'add',[lift,outer])
    spread('legend_contrast',crops,[[[1],[0]],[[0],[1]]])
    return spread('legend_maps',maps,[[[1],[0]],[[0],[1]]])


def detail_drawing(k,radius,pose):
    x,y=coordinates(radius,pose);v=pose[-1];d=plate(x,y,k,v)
    a=sg(ssavg(LUM[d],S),(.50,.72,.57,.41,.65,.57)[v]);mask=sg(ssavg(interior(x,y,k),S),.32)
    sharp=a-sg(a,1.55)
    ret=[]
    for dx,dy in offsets(v):
            m=translate_samples(mask,20,dx,dy)
            t=translate_samples(a,20,dx,dy);e=translate_samples(sharp,20,dx,dy)
            # Each input norm is over a 20x20 square; scale the masked dot product
            # to the same units, independently of sign size and the size of the mask.
            div=1/np.sqrt(max(25,m.sum()))
            if k in (13,15):
                # A blank interior is detected by absence of local contrast below.
                ret.append(np.zeros((2,20,20),F));continue
            if k==12:
                # Yellow/white contrast varies with the illuminant. The full-face
                # colour branch is a much more specific predictor for this sign.
                div*=.54
            ret.append(np.stack([.53*normalize_component(t,m),.47*normalize_component(e,m)])*div)
    return ret


def from_templates(src,name,fn,radii,which=range(43)):
    # Keep a single contiguous float32 array rather than a Python list of banks.
    rows=[];classes=list(which)
    for k in classes:
        for r in radii:
            for pose in POSE:
                rows.extend(np.asarray(fn(k,r,pose),dtype=F))
    c=len(classes); nr=len(radii); p=len(POSE); shifts=len(DX)*len(DY)
    det=dense(name,src,rows)
    grid=node(name+'_poses','reshape',det,shape=[c,nr,p*shifts]);return grid


def contour_maps(lum):
    # Polarity-sensitive derivative filters. Pool nearby contour energy before
    # normalizing, so a displaced contour is different from a missing contour.
    z=np.arange(-2,3);x,y=np.meshgrid(z,z);g=np.exp(-.5*(x*x+y*y)/.9**2)
    filters=[]
    for th in (0,np.pi/2,np.pi/4,-np.pi/4):
        a=g*(x*np.cos(th)+y*np.sin(th));a/=np.sum(abs(a))
        filters.extend([a,-a])
    filters=np.array(filters)
    # Valid convolution skips the two-pixel margin around the photograph; this
    # prevents an artificial black frame from entering the orientation maps.
    e=conv('directional_edges',lum,filters[:,None])
    e=pool('edge_cells',node('edge_positive','relu',e),'avgpool',2,2,2,2)
    e=conv('edge_floor',e,np.eye(8)*.085)
    return node('edge_contrast','norm',e),filters


def contour_drawing(k,radius,pose,filters):
    x,y=coordinates(radius,pose);v=pose[-1];d=plate(x,y,k,v)
    # Calculate oriented descriptors on this drawing only. Different templates
    # never share a statistic, not even within the same speed limit.
    a=sg(ssavg(LUM[d],S),.55);ap=np.pad(a,2,mode='edge')
    vec=[]
    for f in filters:
        z=sum(f[i,j]*ap[i:i+H,j:j+H] for i in range(5) for j in range(5))
        vec.append(sg(np.maximum(z,0),.8))
    field=sg(ssavg(face_mask(x,y,k),S),.75)
    inner=sg(ssavg(interior(x,y,k),S),.75)
    ret=[]
    for dx,dy in offsets(v):
        m=translate_samples(field,14,dx,dy,2);mi=translate_samples(inner,14,dx,dy,2)
        # Centre the descriptor within each polarity bin. Most of the mask is on
        # the legend, but retaining the boundary makes its position identifiable.
        m=.35*m+.65*mi;vecs=np.stack([translate_samples(z,14,dx,dy,2) for z in vec]);sm=m.sum()
        vecs=(vecs-(vecs*m).sum((1,2),keepdims=True)/max(1.,sm))*m
        vecs/=max(1.e-7,np.linalg.norm(vecs))*math.sqrt(8*max(1.,(m*m).sum()))
        ret.append(vecs)
    return ret


def chroma_support(hue):
    # Soft tests for absolute hue (as opposed to matching a zero-mean pattern).
    # Saturating them prevents an intensely coloured background from deciding the
    # family by itself. This part doesn't depend on any synthetic image.
    # A yellow ink reflects BOTH red and green; a red stop sign is not yellow
    # just because its blue channel is low. min(R,G)-B can be implemented by
    # subtracting the positive part of R-G from R-B.
    ab=conv('red_green_difference','log_light',[[1,-1,0]]);ap=node('red_over_green','relu',ab)
    base=conv('semantic_axes','log_light',[[1,-.8,-.2],[-.75,-.25,1],[1,0,-1]])
    minus=conv('yellow_correction',ap,[[0],[0],[-1]])
    hue=node('semantic_chroma','add',[base,minus]);mat=np.repeat(np.eye(3),2,axis=0)
    sig=conv('chroma_hinges',hue,mat,[0,-1.05,0,-1.05,0,-1.05]);r=node('chroma_positive','relu',sig)
    h=conv('chroma_amount',r,[[1,-1,0,0,0,0],[0,0,1,-1,0,0],[0,0,0,0,1,-1]])
    h=pool('chroma_cells',h,'avgpool',2,2,2,2)
    u=(np.arange(16)+.5-8)*2/12.5;x,y=np.meshgrid(u,u);rr=np.sqrt(x*x+y*y)
    z=[]
    for k in range(43):
        f=grp(k); w=np.zeros((3,16,16))
        if f in ('ring','warn','yield'):
            if f=='ring':a=np.exp(-.5*((rr-.86)/.17)**2)
            else:
                yy=y if f=='warn' else -y
                # Distances to the two sloped borders and the bottom border.
                dd=np.minimum((yy+.87-1.86*np.abs(x))/2.11,.87-yy)
                a=np.exp(-.5*(dd/.16)**2)*(yy<1.12)*(yy>-1.17)*(np.abs(x)<1.08)
            a/=a.sum();w[0]=a*.17;w[1]=-a*.067
        if f in ('blue','entry','stop','diamond','ended'):
            a=np.exp(-.5*(rr/(.47 if f=='diamond' else .58))**4);a/=a.sum()
            if f=='blue':w[1]=a*.24;w[0]=-a*.13
            elif f=='diamond':w[2]=a*.37;w[1]=-a*.08
            elif f=='ended':w[0]=-a*.20;w[1]=-a*.08
            else:w[0]=a*.235;w[1]=-a*.12
        z.append(w)
    return dense('hue_rules',h,z)


def build(dest):
    dest.mkdir(parents=True,exist_ok=True)
    lum,hn,ln,hp=front_end()
    f=spread('face_maps',[ln,hn],[[[1],[0],[0],[0]],[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]])
    fs=pool('face_cells',f,'avgpool',2,2,2,2)
    # Separate per-channel centering/contrast normalization restricted to the
    # central patch prevents a bright sky or a shadowy border suppressing text.
    dn=make_legend_maps(lum,hp)
    full=from_templates(fs,'sign_faces',whole_drawing,RADII)
    part=from_templates(dn,'sign_legends',detail_drawing,RADII)
    cm,filters=contour_maps(lum)
    ec=from_templates(cm,'sign_contours',lambda k,r,p:contour_drawing(k,r,p,filters),RADII)
    # Do not require perfect registration of the icon and face. Condition on the
    # same radius, but allow a small independent displacement and style change.
    count=len(POSE)*len(DX)*len(DY)
    fb=pool('face_pose_max',full,'maxpool',1,count,1,count)
    pb=pool('legend_pose_max',part,'maxpool',1,count,1,count)
    # Scales set by the type of sign. Families with featureless interiors must
    # use the whole face, including its contrast against the world.
    gs=[];gp=[];bias=[]
    for k in range(43):
        f=grp(k)
        a,b=(.81,.55) if f=='ring' else (.77,.64)
        if f=='warn':a,b=.81,.55
        if f=='ended':a,b=.64,.76
        if k in (12,13,15,17,14):
            a,b={12:(1.30,.25),13:(1.38,0),15:(1.17,0),17:(.96,.55),14:(1.03,.45)}[k]
        gs.append(a);gp.append(b);bias.append(0.)
    fg=conv('coat_evidence',fb,np.diag(gs));pg=conv('legend_evidence',pb,np.diag(np.maximum(0,np.array(gp)-.25)))
    # Treat the edge matcher as a complementary view. Part of its score must
    # agree on both scale AND pose with the intensity matcher, rather than
    # accepting three unrelated best explanations of a cluttered crop.
    joint=node('registered_views','add',[part,ec]);jb=pool('registered_pose_max',joint,'maxpool',1,count,1,count)
    eg=conv('oriented_evidence',jb,np.diag([.28 if k not in (13,15) else .38 for k in range(43)]))
    combined=node('within_size','add',[fg,pg,eg]);best=pool('best_size',combined,'maxpool',len(RADII),1,len(RADII),1)
    main=node('shape_scores','reshape',best,shape=[43]);color=chroma_support('opponents')
    extras=local_rules(hp,lum)
    node('class_scores','add',[main,color,extras]);(dest/'network.json').write_text(json.dumps({'layers':layers},indent=1)+'\n')
    np.savez(dest/'weights.npz',**weights)


def local_rules(hp,lum):
    # Magnitude of high-pass luminance, relative to the local printing contrast.
    # No-vehicles has nothing printed at all on its white central field. A high-
    # contrast bit of text in the middle is evidence against it, even if the
    # outer red annulus alone matches perfectly.
    u=(np.arange(20)+.5-10);x,y=np.meshgrid(u,u);r=np.sqrt(x*x+y*y)
    # Extract one normalized channel from the shared crop, then take |x|.
    t=conv('ink_signs','legend_contrast',[[0,1],[0,-1]])
    t=node('ink_energy','relu',t);m=np.zeros((43,2,20,20))
    center=np.exp(-.5*(r/4.5)**4);center/=center.sum()
    m[15,:,:,:]=-.67*center;m[13,:,:,:]=-.15*center
    # STOP text forms a band of alternating light/dark strokes. In contrast,
    # no-entry has a continuous white bar, and few edges along its centre line.
    return dense('empty_face_test',t,m)


if __name__=='__main__':build(Path(sys.argv[1]))
