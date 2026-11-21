"""A data-free sign recognizer.  All coordinates are design coordinates, not samples.
Run: python3 build.py DIRECTORY. Only writes the two answer files.
"""
import sys, json, math
from pathlib import Path
import numpy as np

layers=[]; weights={}
def node(name,op,x,**kw):
    layers.append(dict(name=name,op=op,inputs=[x] if isinstance(x,str) else x,**kw));return name
def aff(name,x,W,b=0,conv=False,stride=1,padding=0):
    W=np.asarray(W,dtype=np.float32); o=W.shape[0]
    if not conv: W=W.reshape(o,-1)
    weights[name+'.W']=W;weights[name+'.b']=np.broadcast_to(np.float32(b),(o,)).copy()
    return node(name,'conv' if conv else 'dense',x,out=o,**(dict(kernel=list(W.shape[-2:]),stride=stride,padding=padding) if conv else {}))
def lin(name,x,W,b=0):return aff(name,x,np.asarray(W)[:,:,None,None],b,True)
def norm(name,x):return node(name,'norm',x)
def shape(name,x,*s):return node(name,'reshape',x,shape=list(s))
def pool(name,x,h,w,op='maxpool',sy=None,sx=None):
    return node(name,op,x,size=[h,w],stride=[sy or h,sx or w])
def relu(name,x):return node(name,'relu',x)
def add(name,*xs):return node(name,'add',list(xs))

def window(name,img,k):
    # A one-kernel valid crop using a delta; kernel size 33-k.
    d=33-k;W=np.zeros((1,1,d,d));W[0,0,d//2,d//2]=1
    return aff(name,img,W,conv=True)

def coordinate(n):
    t=np.arange(n,dtype=np.float64)+.5-n/2
    return np.meshgrid(t,t)

def lunit(a):return a/(np.sqrt((a*a).sum())+1e-12)

# Smooth pixel footprint for shape primitives. (x,y) are in the canonical
# pictogram coordinate system; +/-1 is the bounding radius of a typical glyph.
class Art:
    def __init__(self,x,y,e=.09):self.x=x;self.y=y;self.e=e;self.a=np.zeros_like(x)
    def paint(self,d,v=1):
        w=np.clip(.5+d/self.e,0,1); self.a=self.a+(v-self.a)*w
    def ell(self,x,y,rx,ry=None,v=1):
        if ry is None:ry=rx
        self.paint((1-np.hypot((self.x-x)/rx,(self.y-y)/ry))*min(rx,ry),v)
    def rect(self,x1,y1,x2,y2,v=1):
        self.paint(np.minimum(np.minimum(self.x-x1,x2-self.x),np.minimum(self.y-y1,y2-self.y)),v)
    def path(self,pts,w,v=1):
        for (x,y),(a,b) in zip(pts,pts[1:]):
            dx=a-x;dy=b-y
            t=np.clip(((self.x-x)*dx+(self.y-y)*dy)/(dx*dx+dy*dy+1e-30),0,1)
            self.paint(w/2-np.hypot(self.x-x-dx*t,self.y-y-dy*t),v)
    def poly(self,p,v=1):
        inside=np.zeros_like(self.x,dtype=bool);ds=np.full_like(self.x,1e5)
        for (x,y),(u,t) in zip(p,p[1:]+p[:1]):
            dx=u-x;dy=t-y
            if abs(dy)>1e-10:inside^=((y>self.y)!=(t>self.y))&(self.x<x+(self.y-y)*dx/dy)
            s=np.clip(((self.x-x)*dx+(self.y-y)*dy)/(dx*dx+dy*dy+1e-30),0,1)
            ds=np.minimum(ds,np.hypot(self.x-x-dx*s,self.y-y-dy*s))
        self.paint(np.where(inside,ds,-ds),v)

def bez(p,q,r,s,n=8):
    t=np.linspace(0,1,n)[:,None];u=1-t
    return list(u**3*np.array(p)+3*u*u*t*np.array(q)+3*u*t*t*np.array(r)+t**3*np.array(s))

def rounded_loop(a,cx,cy,rx,ry,stroke,power=2.35):
    # Rounded rectangular counters, as in DIN road lettering.
    x=abs(a.x-cx); y=abs(a.y-cy)
    d=(1-((x/rx)**power+(y/ry)**power)**(1/power))*min(rx,ry)
    # Digit cells stretch a unit x less than a unit y; compensate so a
    # closed counter has the same physical stroke width as a straight stem.
    ir=rx-stroke*1.36;ih=ry-stroke*.78
    di=(((x/ir)**power+(y/ih)**power)**(1/power)-1)*min(ir,ih)
    a.paint(np.minimum(d,di))

def digit(k,x,y,style,e):
    # Coordinates here span one digit cell, width=height=2 before stretching.
    a=Art(x,y,e); w=.33 if style==0 else .39
    if k=='0':rounded_loop(a,0,0,.95,1,w*1.14,2.8 if style==0 else 2.4)
    elif k=='1':
        a.path([(-.62,-.53),(.10,-.91),(.27,-.91),(.27,.91)],w*1.28)
    elif k=='2':
        a.path(bez((-.71,-.49),(-.64,-1.12),(.73,-1.10),(.72,-.48))+bez((.72,-.48),(.78,-.08),(.24,.14),(-.73,.79)),w)
        a.rect(-.88,.68,.90,.97)
    elif k=='3':
        a.path(bez((-.70,-.60),(-.38,-1.06),(.70,-.98),(.70,-.51))+bez((.70,-.51),(.70,-.20),(.43,-.04),(-.11,-.05)),w)
        a.path(bez((-.12,-.05),(.51,-.11),(.83,.14),(.75,.48))+bez((.75,.48),(.68,.98),(-.42,1.01),(-.76,.62)),w)
    elif k=='5':
        a.rect(-.71,-.98,.85,-.69);a.path([(-.62,-.79),(-.67,-.03)],w)
        a.path(bez((-.68,-.02),(-.15,-.43),(.80,-.34),(.75,.29))+bez((.75,.29),(.85,.99),(-.32,1.11),(-.75,.62)),w)
    elif k=='6':
        a.path(bez((.36,-.93),(.08,-.60),(-.43,-.25),(-.66,.12)),w*1.05)
        rounded_loop(a,-.01,.40,.90,.59,w*.92)
    elif k=='7':a.path([(-.75,-.84),(.77,-.84),(-.25,.90)],w)
    elif k=='8':
        rounded_loop(a,0,-.46,.80,.54,w*.85,2.1);rounded_loop(a,0,.43,.92,.57,w*.90,2.1)
    return a.a

speed={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}

def warning(k,x,y,e,style):
    if k in (27,28,29,31) and style:x=-x
    if k==23 and style:x=-x
    if k==28:x=x/.98;y=(y-.10)/.88
    if k==29:x=x/1.06;y=(y-.27)/.72
    a=Art(x,y,e);m=-1 if k in (19,) else 1
    if k==11: # thick priority stem and thinner side street arms
        a.poly([(-.26,1),(.26,1),(.26,.32),(.76,.54),(.76,.20),(.26,-.15),(.26,-.58),(0,-1),(-.26,-.58),(-.26,-.15),(-.76,.20),(-.76,.54),(-.26,.32)])
    elif k==18:
        a.poly([(-.24,-.93),(0,-1),(.24,-.93),(.13,.40),(-.13,.40)]);a.ell(0,.76,.20,.21)
    elif k in (19,20):
        if style:
            a=Art(x*m,y,e)
            a.poly([(-.54,.98),(-.54,-.13),(-.48,-.43),(-.32,-.68),(-.05,-.80),(.77,-.80),(.77,-.40),(.08,-.40),(-.10,-.31),(-.17,-.10),(-.17,.98)])
            return a.a
        # Single continuous right-angle curve, flat caps rather than a
        # separate vertical and horizontal stroke (left bend is the mirror).
        a=Art(x*m,y,e);w=.52 if style else .41
        a.path([(-.38,.80),(-.38,-.08)]+bez((-.38,-.08),(-.38,-.59),(-.18,-.61),(.51,-.61)),w)
        a.rect(.35,-.61-w/2,.74,-.61+w/2);a.rect(-.38-w/2,.70,-.38+w/2,1)
    elif k==21:
        a.rect(.21,.35,.62,1);a.path(bez((.42,.46),(.42,-.06),(-.39,.20),(-.39,-.38)),.40)
        a.rect(-.59,-.95,-.19,-.35)
    elif k==22:
        a.poly([(-1,.84),(1,.84),(.89,.73),(.78,.50),(.59,.32),(.42,.28),(.23,.34),(0,.58),(-.19,.40),(-.39,.30),(-.59,.32),(-.74,.46),(-.87,.72)])
    elif k==23:
        if style:
            # Alternate slippery-road symbol, rear of car shown as a strong
            # solid silhouette with two narrow winding tire marks.
            a=Art(x-.16*y,(y+.13*x),e)
            a.poly([(-.53,-.77),(.48,-.77),(.58,-.60),(.67,-.37),(.67,-.13),(.53,-.05),(.53,.06),(.35,.06),(.35,-.06),(-.40,-.06),(-.40,.04),(-.59,.04),(-.59,-.09),(-.69,-.16),(-.69,-.42)])
            for xx in [-.49,.44]:a.path(bez((xx,.08),(xx+.45,.44),(xx-.59,.48),(xx-.33,.92)),.105)
            return a.a
        # A listing car and its two winding skid tracks.
        b=Art(x+.19*(y+.24),y-.21*x,e);b.poly([(-.65,-.78),(.43,-.78),(.60,-.45),(.76,-.32),(.76,-.05),(-.75,-.05),(-.75,-.40),(-.65,-.47)])
        b.rect(-.54,-.67,.37,-.48,.1);b.rect(-.62,-.11,-.39,.06);b.rect(.39,-.11,.62,.06);a.a=b.a
        for xx in [-.39,.50]:a.path(bez((xx,.08),(xx+.44,.49),(xx-.65,.45),(xx-.34,.86)),.14)
    elif k==24:
        a.path([(-.47,-.93),(-.47,.96)],.22);a.path([(.18,-.93),(.18,-.31),(.51,.30),(.51,.96)],.23)
    elif k==25:
        a.ell(-.09,-.73,.15);a.path([(-.28,-.47),(-.53,-.04),(-.45,.21)],.32);a.path([(-.4,-.38),(-.20,-.19),(-.18,-.02)],.15)
        a.path([(-.48,.13),(-.62,.49),(-.8,.89)],.22);a.path([(-.4,.16),(-.18,.43),(-.25,.86)],.23)
        a.path([(-.6,-.19),(.47,.63)],.095);a.poly([(.05,.93),(.2,.68),(.39,.59),(.6,.37),(.77,.51),(1,.93)])
    elif k==26:
        for yy,v in [(-.69,.68),(0,.33),(.69,.80)]:a.ell(0,yy,.27,.27,v)
        if style:a.rect(-.29,-.89,-.23,.88,.5);a.rect(.23,-.89,.29,.88,.5)
    elif k==27:
        a.ell(.13,-.77,.16);a.path([(.07,-.48),(-.16,.07)],.31);a.path([(-.02,-.38),(-.30,-.26),(-.48,.09)],.12);a.path([(.02,-.38),(.22,-.08),(.48,.04)],.12)
        a.path([(-.11,.05),(-.26,.51),(-.58,.84)],.20);a.path([(-.06,.07),(.23,.40),(.42,.88)],.22)
    elif k==28:
        a.ell(-.18,-.66,.15);a.path([(-.18,-.40),(-.25,.21)],.32);a.path([(-.26,-.26),(-.55,-.08),(-.68,.2)],.13);a.path([(-.14,-.25),(.08,.04),(.28,.11)],.15)
        a.path([(-.27,.2),(-.43,.57),(-.63,.89)],.19);a.path([(-.23,.20),(-.07,.48),(.08,.81)],.19)
        a.ell(.48,-.28,.12);a.poly([(.34,-.08),(.56,-.05),(.75,.47),(.28,.47)]);a.path([(.38,.4),(.30,.86)],.15);a.path([(.52,.4),(.76,.78)],.14);a.path([(.39,-.02),(.17,.11)],.12)
    elif k==29:
        if style:
            # Side elevation of a bicycle, including the riderless emblem.
            for xx in [-.59,.59]:
                a.ell(xx,.38,.392,.49);a.ell(xx,.38,.265,.332,0)
            a.path([(-.59,.38),(-.25,-.25),(.02,.37),(-.59,.38)],.12)
            a.path([(.02,.37),(.38,-.34),(.59,.38)],.12);a.path([(-.25,-.25),(.38,-.34)],.10)
            a.path([(-.23,-.20),(-.35,-.49)],.12);a.rect(-.51,-.54,-.18,-.43)
            a.path([(.38,-.33),(.29,-.61),(.53,-.67)],.11)
            return a.a
        for xx in [-.60,.61]:
            a.ell(xx,.52,.34,.36);a.ell(xx,.52,.23,.25,0)
        a.path([(-.60,.52),(-.23,.06),(.04,.49),(-.60,.52)],.095);a.path([(.04,.49),(.42,-.02),(.61,.53)],.12)
        a.ell(.15,-.68,.14);a.path([(.03,-.47),(-.32,-.24),(-.37,-.08)],.21);a.path([(-.31,-.06),(.06,.13),(-.07,.51)],.16);a.path([(-.01,-.43),(.17,-.20),(.46,-.16)],.12)
    elif k==30:
        for ang in np.arange(6)*np.pi/3+(np.pi/2 if style else 0):
            d=np.array([np.cos(ang),np.sin(ang)]);p=np.array([-d[1],d[0]]);a.path([(0,0),d*.93],.19)
            for s in [-1,1]:a.path([d*.45,d*.72+p*.26*s],.16)
    elif k==31:
        if style:
            # Leaping stag: oblique trunk; forelegs stretched forward and
            # lowered head, bent hind legs stretched away from the haunches.
            a=Art(x,y-.36*x,e)
            a.ell(.05,.00,.54,.25);a.poly([(-.53,-.50),(-.30,-.48),(-.12,-.13),(.2,.02),(-.36,.12),(-.54,-.22),(-.76,-.27),(-.78,-.38)])
            for dx in [-.07,.08]:
                a.path([(-.46+dx,-.39),(-.49+dx,-.63),(-.72+dx,-.85)],.09);a.path([(-.53+dx,-.61),(-.39+dx,-.80)],.08)
                a.path([(-.30+dx,.05),(-.51+dx,.29),(-.81+dx,.57)],.13)
                a.path([(.45+dx,.06),(.48+dx,.34),(.86+dx,.54)],.16)
            return a.a
        a.poly([(-.74,-.48),(-.48,-.19),(-.29,-.12),(.38,-.12),(.67,.06),(.78,.09),(.69,.28),(.33,.31),(-.21,.13),(-.50,.10),(-.67,-.25),(-.82,-.25),(-.92,-.37)])
        a.path([(-.58,-.36),(-.60,-.64),(-.79,-.83)],.085);a.path([(-.60,-.62),(-.42,-.79)],.09)
        for pts in [[(-.28,.03),(-.41,.33),(-.81,.49)], [(-.28,.06),(-.38,.42),(-.57,.65)],[(.41,.20),(.56,.49),(.92,.72)],[(.43,.2),(.39,.54),(.66,.74)]]:a.path(pts,.11)
    return a.a

warn=[11]+list(range(18,32)); blue=list(range(33,41));ends=[6,32,41,42]

def mandatory(k,x,y,e,style):
    a=Art(x,y,e)
    if k in [33,34,36,37]:
        a=Art(x*(-1 if k in [34,37] else 1),y,e)
        if k in [33,34]:
            a.rect(-.56,.06,-.23,.95);a.path(bez((-.395,.16),(-.395,-.39),(-.19,-.43),(.28,-.43)),.33);a.rect(.11,-.595,.36,-.265)
            a.poly([(.20,-.98),(.91,-.44),(.20,.13)])
        else:
            a.rect(-.42,-.50,-.13,.96);a.poly([(-.74,-.37),(-.275,-1),(.21,-.37)])
            a.path(bez((-.27,.47),(-.25,-.03),(-.06,-.13),(.36,-.13)),.28);a.poly([(.25,-.60),(.90,-.13),(.25,.35)])
    elif k in [35,38,39]:
        if k!=35:
            # rotate an up arrow into the lower right/left quadrant
            sg=1 if k==38 else -1; u=(x*sg-y)/np.sqrt(2); v=-(x*sg+y)/np.sqrt(2);a=Art(u,v,e)
        a.poly([(-.17,1),(.17,1),(.17,-.20),(.59,-.20),(0,-1),(-.59,-.20),(-.17,-.20)])
    elif k==40:
        for start in np.arange(3)*2*np.pi/3-.4:
            t=start-np.linspace(0,1.42,12);a.path(list(zip(.68*np.cos(t),.68*np.sin(t))),.23)
            d=np.array([np.sin(t[-1]),-np.cos(t[-1])]);p=np.array([-d[1],d[0]]);c=.68*np.array([np.cos(t[-1]),np.sin(t[-1])]);a.poly([c+d*.27,c-d*.15+p*.32,c-d*.15-p*.32])
    return a.a

def bans(k,x,y,e,style):
    a=Art(x,y,e)
    if k in [9,10,41,42]:
        for cx,v,dy in [(-.49,.62 if k<41 else .6,.12),(.47,1 if k<41 else .62,-.10)]:
            if cx<0 and k in [10,42]:
                a.rect(-.92,-.68,-.08,.49,v);a.rect(-.82,.43,-.18,.62,v);continue
            a.poly([(cx-.43,.30+dy),(cx-.43,-.05+dy),(cx-.34,-.15+dy),(cx-.25,-.44+dy),(cx+.22,-.44+dy),(cx+.34,-.16+dy),(cx+.43,-.03+dy),(cx+.43,.30+dy)],v)
            a.poly([(cx-.22,-.35+dy),(cx+.18,-.35+dy),(cx+.28,-.13+dy),(cx-.29,-.13+dy)],0)
            a.rect(cx-.37,dy+.23,cx-.23,dy+.44,v);a.rect(cx+.23,dy+.23,cx+.37,dy+.44,v)
    if k==16:
        a.rect(-.37,-.48,.94,.29);a.poly([(-.94,.31),(-.94,-.05),(-.67,-.34),(-.46,-.34),(-.46,.17),(-.32,.17),(-.32,.32)])
        a.rect(-.84,-.04,-.56,.04,0);a.ell(-.63,.32,.18);a.ell(.58,.32,.18)
    return a.a

# Normalized polygon interior distance for sign faces. The coordinates of the
# bounding box are -1..1; positive distance denotes the inside.
def insidepoly(x,y,pts):
    r=np.full_like(x,1e4)
    for (a,b),(c,d) in zip(pts,pts[1:]+pts[:1]):
        r=np.minimum(r,((c-a)*(y-b)-(d-b)*(x-a))/math.hypot(c-a,d-b))
    return r

def outer(k,x,y,R):
    if k in warn or k==13:
        if k==13:y=-y
        d=insidepoly(x,y,[(0,-1),(1,.98),(-1,.98)])
    elif k==12:d=(1-abs(x)-abs(y))/np.sqrt(2)
    elif k==14:
        t=(np.arange(8)+.5)*np.pi/4;d=insidepoly(x,y,list(zip(np.cos(t)/np.cos(np.pi/8),np.sin(t)/np.cos(np.pi/8))))
    else:d=1-np.hypot(x,y)
    f=lambda t:np.clip(.5+(d-t)*R,0,1)
    obj=f(0); red=np.zeros_like(x); bl=np.zeros_like(x);yellow=np.zeros_like(x)
    lum=obj*.67   # values relative to a gray surround, then locally DC removed
    if k in warn or k==13 or k in speed or k in [9,10,15,16]:
        red=f(.028)-f(.18 if k in warn or k==13 else .206);lum-=.73*red
    elif k==12:
        yellow=f(.145);lum-=.20*yellow
    elif k in blue:
        bl=f(.045);lum-=.78*bl
        ink=mandatory(k,x/.73,y/.73,.13,0);lum+=ink*.90;bl-=ink*.85
    elif k in [14,17]:
        red=f(.055);lum-=.72*red
        if k==17:ink=np.clip(.5+(.73-abs(x))*R,0,1)*np.clip(.5+(.17-abs(y))*R,0,1)
        else:
            ink=np.zeros_like(x)
            for i,letter in enumerate('STOP'):
                aa=Art((x-(i-1.5)*.36)/.15,y/.32,.25)
                if letter=='S':aa.path([(.67,-.85),(-.44,-.85),(-.73,-.58),(-.64,-.14),(.52,.10),(.70,.57),(.46,.87),(-.67,.87)],.37)
                elif letter=='T':aa.rect(-.94,-.99,.94,-.73);aa.rect(-.19,-.80,.19,.96)
                elif letter=='O':rounded_loop(aa,0,0,.95,1,.32)
                else:aa.rect(-.85,-.97,-.48,.98);aa.path([(-.58,-.82),(.33,-.82),(.64,-.63),(.64,-.23),(.32,.00),(-.56,.00)],.34)
                ink=np.maximum(ink,aa.a)
        red-=ink*.93;lum+=ink*.96
    elif k in ends:
        band=np.clip(.5+(.205-abs(x+y)/np.sqrt(2))*R,0,1)*obj
        lum-=band*.78
    return (d,obj,lum,red,bl,yellow)

def prep():
    # Physiologically motivated opponent channels. Log-like contrast reduces the
    # effect of several stops of changing exposure without ratios of small RGBs.
    knees=[0,.03,.09,.23,.5];slopes=[8,5,2.7,1.35,.72]
    W=np.repeat(np.eye(3),len(knees),axis=0)
    steps=relu('exposure_ramps',lin('exposure_knots','image',W,-np.tile(knees,3)))
    W=np.zeros((3,3*len(knees)))
    for i in range(3):W[i,i*len(knees):(i+1)*len(knees)]=np.diff([0]+slopes)
    rgb=lin('companded_RGB',steps,W)
    W=[[.20,.53,.27],[1,-.67,-.33],[-.67,-.33,1],[.45,.55,-1]]
    feat=lin('opponents',rgb,W);nfeat=norm('adapted_opponents',feat)
    # Retain information about the absolute strength of chroma too.
    absf=lin('unscaled_opponents',feat,np.diag([0,.85,.85,.70]));f=add('face_channels',nfeat,absf)
    # Exposure-normalized luma; remove some of a broad illumination ramp.
    pen=lin('ink_luminance',nfeat,[[1,0,0,0]])
    return f,pen,feat

def posecoords(x,y,rad,dx,dy,aspect,ang):
    x=x-dx;y=y-dy;t=ang*np.pi/180
    return (x*np.cos(t)+y*np.sin(t))/(rad*aspect),(-x*np.sin(t)+y*np.cos(t))/rad

# Scores for family membership; pooling across size, position, slant. One face
# per weight row. Empty signboard for symbol-bearing families (ink matched below).
family_ids=[15,11,12,13,14,17,32]+blue
fam_of={k:(15 if k in speed or k in [9,10,15,16] else 11 if k in warn else 32 if k in ends else k) for k in range(43)}

def gaussblur(a,s):
    r=math.ceil(s*3);q=np.arange(-r,r+1);g=np.exp(-q*q/(2*s*s));g/=g.sum()
    for ax in (0,1):
        p=np.pad(a,[(r,r) if i==ax else (0,0) for i in (0,1)],mode='edge');a=sum(w*np.take(p,np.arange(a.shape[ax])+i,axis=ax) for i,w in enumerate(g))
    return a

def acfilter(ink,mask,s=.50):
    # One drawing supplies one matched filter. Only the drawing's own local
    # mean and energy enter its normalization; no inter-drawing statistics.
    a=gaussblur(ink,s);a=a-(a*mask).sum()/(mask.sum()+1e-12)
    return lunit(a*mask)

def interior_mask(k,x,y,toe=.10):
    if k in warn:d=insidepoly(x,y,[(0,-1),(1,.98),(-1,.98)])-.18
    else:d=.795-np.hypot(x,y)
    return np.clip(d/toe+.25,0,1)

def project(name,p,keys,gain,bias=0):
    W=np.zeros((43,len(keys)));b=np.zeros(43)
    for j,k in enumerate(keys):W[k,j]=gain;b[k]=bias
    return aff(name,p,W,b)

def replicate(name,arr,classes,which):
    # Channels=poses, width=source classes. Convolution applies the exact same
    # class-routing table at every pose without a huge block diagonal matrix.
    W=np.zeros((len(classes),len(which)))
    for i,c in enumerate(classes):W[i,which.index(fam_of[c])]=1
    return lin(name,arr,W)

def allposes():
    return [(r,a,b,asp,ang) for r in [9.0,10.7,12.5,14.3,15.8]
            for a in [-2.4,-.8,.8,2.4] for b in [-2.4,-.8,.8,2.4]
            for asp,ang in [(1,0),(.89,-6),(1.09,6)]]

def jointfamily(f,P):
    x,y=coordinate(16);W=[]
    for k in family_ids:
        for r,dx,dy,asp,t in P:
            u,v=posecoords(x,y,r/2,dx/2,dy/2,asp,t)
            d,obj,lum,rd,bl,yl=outer(k,u,v,r/2)
            mask=np.clip(1.55+d*r/2,0,1)
            if k in [15,11,13]:
                rim=.18 if k in [11,13] else .206
                # Use the two sides of the border; do not require blank white
                # all the way through the middle of a symbol-bearing sign.
                if k!=13:mask*=.10+.90*np.clip(1.8-(d-rim)*r/2,0,1)
            def ac(a):return lunit((a-(a*mask).sum()/(mask.sum()+1e-10))*mask)
            z=0*lum
            if k in [15,11,13]:w=[.61*ac(lum),.56*ac(rd),z,z]
            elif k==12:w=[.53*ac(lum),z,z,.70*ac(yl)]
            elif k in blue:w=[.70*ac(lum),z,.51*ac(bl),z]
            elif k in [14,17]:w=[.71*ac(lum),.54*ac(rd),z,z]
            else:w=[1.23*ac(lum),z,z,z]
            W.append(w)
    h=pool('face_antialias',f,2,2,'avgpool');z=aff('rim_matches',h,np.array(W)/8)
    rims=shape('rim_by_pose',z,len(family_ids),1,len(P))
    return rims

def smoothink(pen,feat):
    # The red channel is unusually useful for black printing on red-bordered
    # signs: red enamel reflects much more red than black enamel. Combine the
    # red-luminance and achromatic responses before testing for dark lines.
    red=lin('red_sensitive',feat,[[1,.67,0,0]])
    nred=norm('red_adapted',red)
    ink=add('print_luminance',pen,nred)
    # Difference of Gaussians and positive part. Asymmetric blur used only on
    # the input; rectification suppresses negative contrast from bright glare.
    x,y=coordinate(9);g=np.exp(-(x*x+y*y)/(2*2.05**2));g/=g.sum()
    g2=np.exp(-(x*x+y*y)/(2*.54**2));g2/=g2.sum()
    return relu('ink_on_light',aff('local_dark_lines',ink,(g-g2)[None,None],conv=True,padding=4))

def glyph(k,u,v,style):
    if k in speed:
        text=speed[k];three=len(text)==3
        # The 1 in a three-digit roundel is narrower. Center the visual text
        # (rather than centering the advance boxes of a monospaced font).
        centers=([-.32,.32] if not three else [-.49,-.07,.40]);widths=([.265,.265] if not three else [.15,.20,.21]);hh=.41 if not three else .39
        ink=np.zeros_like(u)
        for c,xc,rx in zip(text,centers,widths):ink=np.maximum(ink,digit(c,(u-xc)/rx,v/hh,style,.20))
        return ink
    if k in warn:
        sx=.58 if k in [22,23,25,28,29,31] else .47
        return warning(k,u/sx,(v-.29)/.50,.13,style)
    if k in [9,10,16]:return bans(k,u/.68,v/.66,.13,style)
    if k in blue:return mandatory(k,u/.73,v/.73,.13,style)
    if k==6:return np.maximum(digit('8',(u+.32)/.26,v/.41,style,.2),digit('0',(u-.32)/.26,v/.41,style,.2))
    if k in [41,42]:return bans(k,u/.68,v/.66,.13,style)
    return np.zeros_like(u)

def local_mask(k,u,v):
    m=interior_mask(k,u,v)
    if k in ends:m=m*np.clip((abs(u+v)/np.sqrt(2)-.215)/.08,0,1)
    return m

def normalized_patch(n,name,img):return norm(name+'_adapt',window(name+'_crop',img,n))

def native(name,ks,img,ink,fan,P,fam):
    # Simultaneously normalize separate interior regions in the photographed
    # image. No interpolation of the input. Each channel to norm contains one
    # candidate region. A few points repeat so every channel has equal length.
    sub=[j for j,p in enumerate(P) if p[3]==1 and p[0]<15];H=len(sub)
    n=24;x,y=coordinate(n);regions=[];coords=[]
    for j in sub:
        r,dx,dy,_,_=P[j];u=(x-dx)/r;v=(y-dy)/r;coords.append((u,v,r))
        if fam==11:d=insidepoly(u,v,[(0,-1),(1,.98),(-1,.98)])-.19
        else:d=.78-np.hypot(u,v)
        if fam==32:d=np.minimum(d,abs(u+v)/np.sqrt(2)-.20)
        regions.append(np.flatnonzero(d>0))
    # Stratified quadrature, at most 256 sites (one site's worth of work for
    # each cell on a 16x16 surface). Sites refer to original pixel centers.
    L=min(256,max(map(len,regions)));gath=np.zeros((H,L,n*n),np.float32);allidx=[]
    for h,ids in enumerate(regions):
        ids=ids[(np.arange(L)*len(ids))//L];allidx.append(ids);gath[h,np.arange(L),ids]=1
    def gather(tag,img):
        z=aff(name+tag+'_sample',img,gath.reshape(-1,n*n));return norm(name+tag+'_norm',shape(name+tag+'_layout',z,H,1,L))
    # Independently normalize the signed and rectified bandpass measurements.
    s1=gather('_intensity',img);s2=gather('_lines',ink)
    alts=[(sc,asp,sty,off) for sc in [.85,1,1.16] for asp,sty,off in [(.88,0,0),(1.10,1,0),(1,0,-.07),(1,1,.07)]]
    V=len(alts);B=len(ks);stride=B*V
    # Dense matrices are block diagonal in the pose index. Use multiple
    # chunks to avoid paying for quadratically many zero blocks.
    outs=[];chunksize=2
    def featblock(tag,i,s):
        # Select a small consecutive group of pose channels.
        ww=np.eye(H)[i:i+chunksize];return lin(name+tag+'_select'+str(i),s,ww)
    for i in range(0,H,chunksize):
        m=min(chunksize,H-i);wa=np.zeros((m,stride,m,L),np.float32);wb=np.zeros_like(wa)
        for q in range(m):
            h=q+i;u,v,r=coords[h];ids=allidx[h];cy=.29 if fam==11 else 0
            for b,k in enumerate(ks):
                for j,(scale,asp,sty,off) in enumerate(alts):
                    G=glyph(k,u/(scale*asp),(v-cy-off)/scale+cy,sty);G=gaussblur(G,.53)
                    if k in (15,32):continue
                    g=G.ravel()[ids];D=np.maximum(G-gaussblur(G,2.05),0).ravel()[ids]
                    wa[q,b*V+j,q]=-lunit(g-g.mean())/math.sqrt(L)
                    wb[q,b*V+j,q]=lunit(D-D.mean())/math.sqrt(L)
        sa=featblock('_I',i,s1);sb=featblock('_D',i,s2)
        p=aff(name+'_template_I'+str(i),sa,wa.reshape(m*stride,-1));d=aff(name+'_template_D'+str(i),sb,wb.reshape(m*stride,-1)*.45)
        c=add(name+'_NCC'+str(i),p,d);c=shape(name+'_layout'+str(i),c,m,B,V);c=pool(name+'_style'+str(i),c,1,V)
        # Expand this group into a common H x B table, then add disjoint rows.
        outs.append(lin(name+'_place'+str(i),c,np.eye(H)[:,i:i+m]))
    basic=add(name+'_local_NCC',*outs)
    if 15 in ks or 32 in ks:
        k=15 if 15 in ks else 32
        w=-10*gath.mean(axis=1);null=aff(name+'_empty',ink,w,.65);null=shape(name+'_empty_as_row',null,1,H,1)
        q=np.zeros((B,1));q[ks.index(k)]=1;null=lin(name+'_empty_class',null,q)
        # native NCC is H x B; transpose the small B x H null field.
        w=np.eye(B*H).reshape(B,H,B*H).transpose(1,0,2).reshape(B*H,-1)
        null=shape(name+'_empty_map',aff(name+'_empty_transpose',null,w),H,B,1);basic=add(name+'_all_contents',basic,null)
    # An incompatible rim geometry disqualifies a symbol hypothesis. There are
    # also oblique rim alternatives at each location; take their maximum.
    w=np.zeros((1,len(family_ids)));w[0,family_ids.index(fam)]=1
    rim=lin(name+'_border',fan,w);w=np.eye(len(P))[np.asarray(sub)[:,None]+np.arange(3)]
    rim=shape(name+'_border_angles',aff(name+'_pick_border',rim,w.reshape(-1,len(P))),H,1,3);rim=pool(name+'_best_angle',rim,1,3)
    rim=shape(name+'_border_row',rim,1,H,1);rim=lin(name+'_border_per_class',rim,np.ones((B,1))*.77)
    w=np.eye(B*H).reshape(B,H,B*H).transpose(1,0,2).reshape(B*H,-1);rim=shape(name+'_border_map',aff(name+'_border_transpose',rim,w),H,B,1)
    det=add(name+'_simultaneous',rim,basic);return pool(name+'_result',shape(name+'_pose_layout',det,1,H,B),H,1)

def jointpart(name,ks,P,view,lineview,fan,n,step):
    x,y=coordinate(n); W=[];D=[];nv=len(P)
    for k in ks:
        for r,dx,dy,asp,ang in P:
            u,v=posecoords(x,y,r,dx,dy,asp,ang);mask=local_mask(k,u,v)
            for scale,style in step:
                cy=.29 if k in warn else 0
                ink=glyph(k,u/scale,(v-cy)/scale+cy,style)
                if k in (15,32):
                    W.append(np.zeros_like(mask));D.append(-mask/max(mask.sum(),1)*10)
                else:
                    sign=1 if k in blue else -1
                    W.append(sign*acfilter(ink,mask,.48));g=gaussblur(ink,.53);wide=gaussblur(ink,2.05)
                    D.append(acfilter(np.maximum(g-wide,0),mask,.3))
    a=aff(name+'_luma',view,W);b=aff(name+'_print',lineview,D)
    a=shape(name+'_luma_layout',a,1,len(ks),nv*len(step));b=shape(name+'_print_layout',b,1,len(ks),nv*len(step))
    ratio=(.36 if ks[0] not in blue else 0.)
    ab=add(name+'_agreement',lin(name+'_luma_gain',a,[[.070]]),lin(name+'_print_gain',b,[[ratio*.070]]))
    ab=pool(name+'_font_pool',ab,1,len(step));ab=shape(name+'_by_pose',ab,len(ks),1,nv)
    rim=replicate(name+'_rim',fan,ks,family_ids)
    joint=add(name+'_joint',rim,ab)
    return pool(name+'_best',joint,1,nv)

def digit_parts(form,img):
    # A compositional segmentation: the same last zero is common to the
    # entire family. Normalize the other digit in a local reading cell. A
    # dark stripe outside this cell neither hides a faint digit nor helps it.
    name='character_'+str(form);ks=[0,1,2,3,4,5] if form==2 else [7,8]
    x,y=coordinate(20);P=[(h,asp,dx,dy) for h in [3.25,4.15,5.1,6.1] for asp in [.86,1.08] for dx in [-1.5,0,1.5] for dy in [-1.5,0,1.5]]
    H=len(P);tset=[(t,style,ang) for t,style in [(-.13,0),(0,0),(.13,0),(0,1)] for ang in [-4.5,4.5]]
    regions=[]
    for hh,asp,dx,dy in P:
        # Set the reading window slightly wider than a digit and include
        # whitespace immediately above and below its baseline/cap height.
        cx=dx-(.76 if form==2 else .13)*hh*asp;rx=hh*(.73 if form==2 else .48)*asp
        regions.append(np.flatnonzero((abs(x-cx)<rx*1.16)&(abs(y-dy)<hh*1.24)))
    L=min(100,max(map(len,regions)));I=np.zeros((H,L,400),np.float32);ids=[]
    for h,idx in enumerate(regions):
        idx=idx[np.arange(L)*len(idx)//L];I[h,np.arange(L),idx]=1;ids.append(idx)
    a=aff(name+'_sample',img,I.reshape(-1,400));a=norm(name+'_contrast',shape(name+'_cells',a,H,1,L))
    outs=[];B=len(ks);V=len(tset)
    for i in range(0,H,3):
        m=min(3,H-i);w=np.zeros((m,B,V,m,L),np.float32)
        for q in range(m):
            hh,asp,dx,dy=P[q+i];idx=ids[q+i];rx=hh*(.65 if form==2 else .45)*asp
            cx=dx-hh*(.76 if form==2 else .13)*asp
            for b,k in enumerate(ks):
                ch=speed[k][0 if form==2 else 1]
                for j,(off,sty,ang) in enumerate(tset):
                    u,v=posecoords(x,y,hh,cx+off*hh,dy,1,ang)
                    art=digit(ch,u/(rx/hh),v,sty,.20);g=gaussblur(art,.43).ravel()[idx]
                    w[q,b,j,q]=-lunit(g-g.mean())/math.sqrt(L)
        sel=lin(name+'_at'+str(i),a,np.eye(H)[i:i+m]);d=aff(name+'_bank'+str(i),sel,w.reshape(m*B*V,-1));d=shape(name+'_grid'+str(i),d,m,B,V)
        d=pool(name+'_beststyle'+str(i),d,1,V)
        # Context in the rest of the printed speed confirms the reading cell.
        # Evaluate it on a separate contrast-adapted image, with the cell
        # itself masked off so none of the target strokes are reused here.
        C=[]
        for q in range(m):
            hh,asp,dx,dy=P[q+i];u=(x-dx)/(hh*asp);v=(y-dy)/hh
            ctx=digit('0',(u-(.76 if form==2 else 1.05))/(.65 if form==2 else .48),v,0,.2)
            if form==3:ctx=np.maximum(ctx,digit('1',(u+1.25)/.37,v,0,.2))
            mask=np.clip(1.55-abs(v),0,1)*np.clip(2.4-abs(u),0,1)
            mask*=np.clip((abs(u+(.76 if form==2 else .13))-(.72 if form==2 else .50))/.20,0,1)
            C.append(-acfilter(ctx,mask,.44))
        ctx=aff(name+'_neighbor'+str(i),img,np.array(C)*.080);ctx=shape(name+'_neighbor_row'+str(i),ctx,1,1,m)
        # Tile m contexts along B columns (only m*B scalar weights).
        mat=np.eye(m).repeat(B,axis=0);ctx=shape(name+'_common'+str(i),aff(name+'_context'+str(i),ctx,mat),m,B,1)
        z=add(name+'_joint'+str(i),d,ctx);z=pool(name+'_bestpose'+str(i),shape(name+'_posegrid'+str(i),z,1,m,B),m,1)
        outs.append(z)
    table=[]
    for i,z in enumerate(outs):
        z=shape(name+'_block'+str(i),z,1,1,B);table.append(lin(name+'_deposit'+str(i),z,np.eye(len(outs))[:,i:i+1]))
    z=add(name+'_table',*table);return pool(name+'_read',shape(name+'_all'+str(form),z,1,len(outs),B),len(outs),1),ks

def create():
    f,pen,feat=prep();ink=smoothink(pen,feat);P=allposes();family=jointfamily(f,P)
    v=normalized_patch(22,'symbol',pen);line=normalized_patch(22,'stroke',ink)
    # Separate banks, shared pose coordinates. The gains below are expressed in
    # units of normalized center-patch contrast, not class frequencies.
    nums=list(speed)+[9,10,16,15];s1=jointpart('roundel',nums,P,v,line,family,22,[(.91,0),(1.04,0),(1.16,1)])
    s2=jointpart('warning',warn,P,v,line,family,22,[(.88,0),(1.04,0),(1.19,1)])
    s3=jointpart('end',ends,P,v,line,family,22,[(.93,0),(1.09,1)])
    s4=jointpart('arrow',blue,P,v,line,family,22,[(.93,0),(1.07,1)])
    red=norm('print_channel',lin('print_channel_raw',feat,[[1,.93,-.09,0]]))
    circleview=window('native_ink_view',red,24);iv=window('native_stroke_view',ink,24)
    detail=[]
    for name,keys,fam in [('n_digits',nums,15),('n_warning',warn,11),('n_end',ends,32)]:
        det=native(name,keys,circleview,iv,family,P,fam)
        vals=shape('read_'+name,det,1,1,len(keys));peak=pool('read_'+name+'_max',vals,1,len(keys))
        detail.append(project('read_'+name+'_positive',det,keys,1.10));coef=np.zeros((43,1));coef[keys]=-1.10
        detail.append(aff('read_'+name+'_neutral',peak,coef))
    digview=normalized_patch(20,'character',red)
    for form in [2,3]:
        det,keys=digit_parts(form,digview);tag='char_vote'+str(form);vals=shape(tag+'_row',det,1,1,len(keys));peak=pool(tag+'_max',vals,1,len(keys))
        detail.append(project(tag,det,keys,1.45));coef=np.zeros((43,1));coef[keys]=-1.45
        detail.append(aff(tag+'_neutral',peak,coef))
    ks=[12,13,14,17];r=replicate('unlettered_shapes',family,ks,family_ids);s5=pool('unlettered_best',r,1,len(P))
    # Group scores as a family gate plus within-family evidence. Do not pay
    # each family again merely for having more foreground pixels. Compare its
    # symbol alternatives relative to the BEST one. This cancels scale and
    # contrast variations between families in an input-dependent manner.
    terms=[]
    for i,(s,keys) in enumerate([(s1,nums),(s2,warn),(s3,ends),(s4,blue),(s5,ks)]):
        h='group'+str(i);terms.append(project(h+'_joint',s,keys,1))
        if i<4:
            gm=pool(h+'_maximum',shape(h+'_class_layout',s,1,len(keys),1),len(keys),1)
            shared=replicate(h+'_family',family,keys,family_ids);shared=pool(h+'_best_rim',shared,1,len(P));shared=pool(h+'_gate',shape(h+'_rim_layout',shared,1,len(keys),1),len(keys),1)
            A=np.zeros((43,1));A[keys,0]=.76
            terms.append(aff(h+'_remove_common',gm,-A));terms.append(aff(h+'_restore_gate',shared,A))
    add('class_scores',*terms,*detail)

if __name__=='__main__':
    create();out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps(dict(layers=layers),indent=1)+'\n');np.savez(out/'weights.npz',**weights)
