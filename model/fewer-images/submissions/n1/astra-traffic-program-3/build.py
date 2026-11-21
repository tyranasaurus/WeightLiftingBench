"""Construct every filter and reference drawing from geometry, without file inputs."""
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
arrays = {}

# Signed distance, positive inside. Coordinates run left to right and top to bottom.
def clip(x): return np.clip(x,0,1)
def tri(x,y):
    # Round the corners of an offset triangle (Minkowski sum with a disk).
    a,b,c=(0.,-.83),(.86,.89),(-.86,.89)
    dist=np.minimum.reduce([segdist(x,y,a,b),segdist(x,y,b,c),segdist(x,y,c,a)])
    inside=(y<=b[1]) & (y>=2*abs(x)+a[1])
    return .105+np.where(inside,dist,-dist)
def segdist(x,y,a,b):
    u,v=b[0]-a[0],b[1]-a[1]; t=clip(((x-a[0])*u+(y-a[1])*v)/(u*u+v*v+1e-12)); return np.hypot(x-a[0]-t*u,y-a[1]-t*v)
def shape(f,x,y):
    if f==1:return tri(x,y)
    if f==3:return tri(x,-y)
    if f==4:return (1.04-abs(x)-abs(y))*.7071
    if f==5:return np.minimum(1-np.maximum(abs(x),abs(y)),(1.4142-abs(x)-abs(y))*.7071)
    return 1-np.hypot(x,y)
# f = red circle, warning triangle, blue disc, yield, priority, stop, no entry, cancellation
# Four independent measures per pose: pigment, light/dark, outline, unwanted pigment.
def kernel(f,x,y,aa):
    d=shape(f,x,y)
    def s(t):return clip(.5+t/aa)
    def area(lo,hi):return s(d-lo)-s(d-hi)
    def unit(a):return a / max(.001,a.sum())
    def u(lo,hi):return unit(area(lo,hi))
    outside=u(-.27,-.06); face=s(d-.20); body=s(d-.035); ring=u(.015,.18)
    # Avoid relying on the unknown pictogram for the estimate of paper colour.
    if f in (0,1,3):
        w=unit(face*(.35+.65*s(.42-d)))
        if f==3:w=unit(face)
        pigment=ring-.54*w-.19*outside
        light=w-ring
        edge=.55*u(-.065,.07)+.45*u(.12,.235)-.24*outside
        reject=unit(face)
    elif f==2:
        blue=unit(body*(.38+.62*s(.49-d))); w=unit(s(d-.34))
        pigment=blue-.33*outside
        light=.60*(w-u(.08,.31))
        edge=u(-.08,.09)-.25*outside
        reject=unit(body)
    elif f==4:
        yellow=unit(s(d-.20)); white=u(.028,.146)
        pigment=yellow-.43*white-.16*outside
        light=white-yellow
        edge=.55*u(-.07,.07)+.45*u(.12,.21)-.2*outside
        reject=yellow
    elif f in (5,6):
        stripe=s((.165 if f==6 else .222)-abs(y))*s((.72 if f==6 else .81)-abs(x))
        ink=unit(stripe); paint=unit(s(d-.07)*(1-stripe)*s(abs(y)-(.19 if f==6 else .25)))
        pigment=paint-(.55 if f==6 else .19)*ink-.19*outside
        light=ink-paint
        edge=u(-.06,.085)-.2*outside
        reject=unit(body)
    else:
        stripe=abs((x+y)*.7071); black=unit(body*s(.195-stripe)); paper=unit(body*s(stripe-.27))
        pigment=0*paper
        light=paper-black
        edge=.52*u(-.07,.08)+.48*unit(body*s(.075-abs(stripe-.20)))-.18*outside
        reject=unit(body)
    return np.stack([pigment,light,edge,reject]).astype(np.float32)

def poses(f,stage):
    ret=[]
    if stage==0:
        for cx in (-3.6,-1.8,0,1.8,3.6):
          for cy in (-3.6,-1.8,0,1.8,3.6):
            for rad in (8.8,10.5,12.3,14.1,15.8):
              for aspect in (.83,1,1.19):
                ret.append((cx+15.5,cy+15.5,rad*aspect**.5,rad/aspect**.5,0.,0.))
    else:
        for cx in (-.075,0,.075):
          for cy in (-.075,0,.075):
            for ax,ay in ((1,1),(.92,.92),(1.08,1.08),(.90,1.04),(1.085,.955)):
              for sx,sy in ((0,0),(.13,-.05),(-.13,.05),(0,.13),(0,-.13)):
                ret.append((15.5+12*cx,15.5+12*cy,12*ax,12*ay,sx,sy))
    return ret

def locbank(f,stage):
    yy,xx=np.mgrid[:32,:32]; filters=[]; p=poses(f,stage)
    for cx,cy,rx,ry,sx,sy in p:
        a,b=(xx-cx)/rx,(yy-cy)/ry
        x,y=(a-sx*b)/(1-sx*sy),(b-sy*a)/(1-sx*sy)
        filters.append(kernel(f,x,y,1.38/min(rx,ry)))
    tag=str(f)+str(stage)
    arrays['k'+tag]=np.stack(filters,axis=-1).reshape(4,1024,len(p)).astype(np.float16)
    arrays['p'+tag]=np.array(p,np.float32)

# A small vector sketch language, used for sign interiors only.
class Pen:
    def __init__(self,x,y):self.x=x; self.y=y; self.im=np.zeros_like(x,dtype=float)
    def paint(self,m,col):self.im=np.where(m,col,self.im)
    def path(self,pts,w,col=1):
        for a,b in zip(pts,pts[1:]):self.paint(segdist(self.x,self.y,a,b)<w/2,col)
    def poly(self,pts,col=1):
        inside=np.zeros(self.x.shape,bool)
        for a,b in zip(pts,pts[1:]+pts[:1]):
            if a[1]!=b[1]:inside^=((self.y>a[1])!=(self.y>b[1]))&(self.x<(b[0]-a[0])*(self.y-a[1])/(b[1]-a[1])+a[0])
        self.paint(inside,col)
    def oval(self,x,y,rx,ry,col=1):self.paint(((self.x-x)/rx)**2+((self.y-y)/ry)**2<1,col)
    def arc(self,x,y,rx,ry,a,b,w,col=1):
        ang=np.linspace(a,b,25)*np.pi/180;self.path(list(zip(x+rx*np.cos(ang),y+ry*np.sin(ang))),w,col)
    def rect(self,x1,y1,x2,y2,col=1):self.paint((self.x>x1)&(self.x<x2)&(self.y>y1)&(self.y<y2),col)

# Digit centre lines in a character cell: rounded DIN-like numerals with open counters.
def digit(p,n,x,y,w,h,weight,style):
    oldx,oldy=p.x,p.y;p.x=(oldx-x)/w;p.y=(oldy-y)/h
    # Work in a system whose height is one; keep physical stroke width isotropic.
    aspect=w/h;p.x=p.x*aspect
    def line(pts,sw=1):p.path([(a*aspect,b) for a,b in pts],weight*sw)
    def curve(points):
        pts=[np.array(points[0])];start=pts[0]
        for k in range(1,len(points),3):
            a,b,c=map(np.array,points[k:k+3]);t=np.linspace(0,1,9)[1:,None];pts.extend((1-t)**3*start+3*(1-t)**2*t*a+3*(1-t)*t*t*b+t**3*c);start=c
        line(pts)
    if n=='0':
        curve([(.5,.065),(.78,.065),(.90,.17),(.90,.33),(.91,.43),(.91,.57),(.90,.68),(.90,.85),(.78,.935),(.5,.935),(.22,.935),(.10,.85),(.10,.68),(.09,.57),(.09,.43),(.10,.33),(.10,.17),(.22,.065),(.5,.065)])
    elif n=='1':line([(.18,.24),(.63,.065),(.63,.94)])
    elif n=='2':
        curve([(.11,.28),(.15,.09),(.35,.055),(.54,.065),(.9,.065),(.98,.28),(.84,.42),(.71,.55),(.27,.76),(.105,.925)]);line([(.105,.925),(.92,.925)])
    elif n=='3':
        curve([(.11,.23),(.17,.075),(.34,.05),(.56,.065),(.95,.065),(.99,.36),(.70,.445),(.64,.47),(.52,.475),(.40,.475)]);curve([(.40,.475),(.75,.45),(.92,.56),(.92,.71),(.92,.98),(.27,1.025),(.10,.77)])
    elif n=='5':
        line([(.87,.07),(.18,.07),(.145,.48)]);curve([(.145,.48),(.35,.36),(.71,.375),(.84,.51),(.97,.64),(.965,.84),(.73,.91),(.48,1.0),(.18,.92),(.095,.78)])
    elif n=='6':
        if style%2:curve([(.85,.19),(.70,.03),(.33,.025),(.20,.20),(.10,.36),(.09,.58),(.105,.69)])
        else:line([(.69,.065),(.38,.32),(.18,.51),(.11,.64)])
        curve([(.105,.68),(.09,.46),(.34,.405),(.55,.42),(.80,.43),(.92,.54),(.92,.685),(.93,.86),(.75,.94),(.52,.94),(.25,.94),(.105,.85),(.105,.68)])
    elif n=='7':line([(.06,.07),(.91,.07),(.34,.945)])
    elif n=='8':
        for cy,rx,ry in ((.27,.365,.208),(.704,.406,.232)):
            t=np.linspace(0,2*np.pi,34);line(list(zip(.50+rx*np.cos(t),cy+ry*np.sin(t))))
    p.x,p.y=oldx,oldy

def number(p,txt,weight,style):
    # Natural bounding box for a string, later scaled to the desired sign-face size.
    gap=.12 if style<2 else .19
    widths=[(.32 if k=='1' else .55) for k in txt];total=sum(widths)+gap*(len(txt)-1)
    a,b=p.x,p.y;p.x=(a+.5)*total;p.y=b+.5
    x=0
    for n,w in zip(txt,widths):digit(p,n,x,0,w,1,weight,style);x+=w+gap
    p.x,p.y=a,b

def car(p,cx,cy,w,h,col):
    def pts(lst):return [(cx+w*a,cy+h*b) for a,b in lst]
    p.poly(pts([(-.49,.35),(-.5,-.02),(-.39,-.12),(-.28,-.43),(-.16,-.48),(.21,-.48),(.31,-.41),(.405,-.12),(.50,-.02),(.49,.35)]),col)
    for s in (-1,1):p.rect(cx+w*(s*.335-.09),cy+.24*h,cx+w*(s*.335+.09),cy+.49*h,col)
    p.poly(pts([(-.30,-.12),(-.21,-.34),(.21,-.34),(.30,-.12)]),.04)

def truck(p,mirror=1):
    a=p.x;p.x=p.x*mirror
    p.rect(-.10,-.42,.49,.26);p.poly([(-.49,.26),(-.49,-.07),(-.33,-.29),(-.115,-.29),(-.115,.25),(.45,.25),(.45,.34),(-.49,.34)])
    p.poly([(-.426,-.08),(-.31,-.225),(-.22,-.225),(-.22,-.08)],0)
    for i in (-.31,.26):p.oval(i,.32,.104,.145)
    p.x=a

def arrow(p,points,tip,base,wing):
    p.path(points,.16);end=points[-1]; vx,vy=tip[0]-base[0],tip[1]-base[1]; s=np.hypot(vx,vy);nx,ny=-vy/s,vx/s
    p.poly([tip,(base[0]+wing*nx,base[1]+wing*ny),(base[0]-wing*nx,base[1]-wing*ny)])

def chevron(p,base,tip,side,thick):
    dx,dy=tip[0]-base[0],tip[1]-base[1];le=np.hypot(dx,dy);nx,ny=-dy/le,dx/le
    p.poly([tip,(base[0]+nx*side,base[1]+ny*side),(base[0]+nx*side-dx*thick,base[1]+ny*side-dy*thick),(tip[0]-dx*thick,tip[1]-dy*thick),(base[0]-nx*side-dx*thick,base[1]-ny*side-dy*thick),(base[0]-nx*side,base[1]-ny*side)])

def figure(p,x,y,s,mirror=1,run=False):
    # head, leaning torso, separated arms and walking legs
    a,b=p.x,p.y;p.x=(a-x)/(s*mirror);p.y=(b-y)/s
    p.oval(.06,-.40,.085,.093);p.path([(.031,-.25),(-.05,.020)],.142)
    p.path([(.00,-.20),(.13,-.06),(.253,-.035)],.068);p.path([(.0,-.21),(-.13,-.15),(-.22,-.02)],.067)
    p.path([(-.035,.01),(.10,.18),(.18,.44)],.086);p.path([(-.055,.015),(-.14,.22),(-.29,.42 if not run else .29)],.085)
    p.x,p.y=a,b

def corner(p,st):
    if st==1:p.path([(-.32,.49),(-.32,.04),(-.27,-.10),(.39,-.40)],.24)
    else:
        t=np.linspace(0,np.pi/2,17);r=.28 if st==0 else .52
        p.rect(-.45,-.24+r,-.20,.5);p.path([(-.325+r*(1-np.cos(a)),-.24+r-r*np.sin(a)) for a in t],.25);p.rect(-.325+r,-.365,.49,-.115)

def icon(p,c,style):
    # Each drawing is authored in its own nominal unit box, not sampled from a photo.
    if c in (0,1,2,3,4,5,7,8):number(p,{0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[c],(.13,.165,.12,.155)[style%4],style)
    elif c in (9,10):
        if c==9:car(p,-.257,-.04,.46,.77,.76 if style%2==0 else .97)
        else:
            p.rect(-.49,-.48,-.054,.24,.80);p.rect(-.47,.21,-.075,.395,.80);p.rect(-.445,.27,-.36,.47,.8);p.rect(-.176,.27,-.10,.47,.80)
        car(p,.258,.07 if c==10 else -.04,.435,.65 if c==10 else .77,1.)
    elif c==16:truck(p,1 if style%2==0 else -1)
    elif c==11:
        if style%2==0:
            p.poly([(-.14,.49),(.14,.49),(.14,.015),(.34,.015),(0,-.51),(-.34,.015),(-.14,.015)]);p.rect(-.50,.18,.50,.265)
        else:
            p.poly([(-.17,.49),(.17,.49),(.17,-.17),(0,-.50),(-.17,-.17)]);p.rect(-.50,.02,.50,.14)
    elif c==18:
        p.poly([(-.24,-.49),(.24,-.49),(.12,.18),(-.12,.18)]);p.oval(0,.365,.20,.10)
    elif c in (19,20):
        if c==19:p.x=-p.x
        corner(p,style)
    elif c==21:
        if style>=2:p.x=-p.x
        pts=[(.24,.49),(.24,.22),(.22,.13),(.16,.07),(-.10,-.03),(-.18,-.10),(-.20,-.21),(-.20,-.48)]
        if style%2:pts=[(.28,.49),(.28,.10),(.22,-.003),(-.15,-.003),(-.21,-.08),(-.21,-.48)]
        p.path(pts,.22)
    elif c==22:
        p.rect(-.50,.30,.50,.44);p.oval(-.218,.31,.24,.52);p.oval(.227,.31,.227,.57);p.rect(-.6,.44,.6,.9,0)
    elif c==23:
        # Rear of skidding car above the pair of sinuous tyre tracks.
        a,b=p.x,p.y;ang=-.26 if style%2==0 else .22;p.x=np.cos(ang)*a+np.sin(ang)*(b+.225);p.y=-np.sin(ang)*a+np.cos(ang)*(b+.225);car(p,.02,0,.53,.43,1);p.x,p.y=a,b
        for dx in (-.17,.16):p.path([(dx+.02,-.04),(dx-.10,.06),(dx-.14,.13),(dx+.12,.27),(dx+.15,.33),(dx-.08,.48)],.052)
    elif c==24:
        p.path([(-.29,.49),(-.29,-.49)],.135);p.path([(.34,.49),(.34,.14),(.29,.035),(.08,-.16),(.06,-.49)],.138)
    elif c==25:
        if style%2:p.x=-p.x
        p.oval(-.05,-.40,.081,.085);p.path([(-.14,-.29),(-.292,-.095),(-.305,-.015)],.151);p.path([(-.30,-.01),(-.405,.225),(-.46,.45)],.105);p.path([(-.27,-.005),(-.17,.147),(-.13,.44)],.108)
        p.path([(-.12,-.235),(-.012,-.068),(.105,-.034)],.072);p.path([(-.21,-.207),(-.157,-.09),(-.03,-.042)],.070);p.path([(-.28,-.21),(.32,.32)],.038);p.poly([(.04,.46),(.18,.26),(.29,.16),(.35,.16),(.51,.46)])
    elif c==26:
        if style%2:p.rect(-.37,-.48,.37,.49)
        for y,v in ((-.326,.78),(0,.57),(.33,.84)):p.oval(0,y,.37,.145,v)
    elif c==27:figure(p,0,.0,1.,-1 if style%2 else 1)
    elif c==28:
        figure(p,-.16,-.02,.93,-1,True);figure(p,.215,.14,.65,-1,True);p.path([(-.05,-.07),(.10,.03),(.16,-.01)],.062)
    elif c==29:
        for x in (-.294,.294):p.arc(x,.251,.181,.209,0,360,.049)
        p.path([(-.294,.25),(-.137,-.05),(.057,.243),(-.294,.25),(-.02,.254),(.19,-.11),(.294,.25)],.047)
        p.oval(.041,-.408,.078,.080);p.path([(-.01,-.294),(-.19,-.087),(-.016,.042),(-.083,.24)],.094);p.path([(-.01,-.266),(.08,-.113),(.203,-.089)],.062)
    elif c==30:
        for t in np.arange(6)*np.pi/3+np.pi/2:
            v=np.array([np.cos(t),np.sin(t)]);n=np.array([-v[1],v[0]]);p.path([(0,0),tuple(.475*v)],.055)
            for k in (.215,.345):
                for sign in (-1,1):p.path([tuple(v*k),tuple(v*(k+.10)+sign*.115*n)],.048)
    elif c==31:
        if style%2:p.x=-p.x
        p.poly([(-.39,-.075),(-.26,-.16),(-.05,-.19),(.14,-.22),(.26,-.37),(.31,-.44),(.38,-.415),(.47,-.33),(.47,-.286),(.352,-.30),(.277,-.09),(.13,.015),(-.13,.074),(-.289,.03),(-.37,.075)])
        p.path([(-.28,.01),(-.34,.18),(-.475,.31)],.060);p.path([(-.2,.01),(-.27,.21),(-.18,.4)],.052);p.path([(.135,-.02),(.27,.108),(.38,.285)],.053);p.path([(.05,.012),(.14,.157),(.10,.385)],.048);p.path([(.32,-.382),(.27,-.49),(.214,-.52)],.033)
    elif c in (33,34,35,36,37,38,39):
        if c in (34,37,39):p.x=-p.x
        if c in (33,34):
            p.path([(-.26,.46),(-.26,.035),(-.239,-.074),(-.18,-.16),(-.095,-.21),(.015,-.235),(.29,-.235)],.167)
            chevron(p,(.19,-.235),(.49,-.235),.246,.55)
        elif c in (35,36,37):
            cx=0 if c==35 else -.17
            p.path([(cx,.48),(cx,-.37)],.16);chevron(p,(cx,-.18),(cx,-.5),.24 if c!=35 else .35,.52)
            if c!=35:
                p.path([(cx,.20),(-.09,.107),(-.01,.026),(.083,-.005),(.35,-.005)],.144);chevron(p,(.23,-.005),(.51,-.005),.215,.53)
        else:
            p.path([(-.425,-.428),(.338,.34)],.195);chevron(p,(.22,.22),(.48,.48),.364,.67)
    elif c==40:
        for a in (0,120,240):
            angle=a+30*(style%2);p.arc(0,0,.35,.35,angle+3,angle+93,.14);t=(angle+3)*np.pi/180;v=np.array([np.sin(t),-np.cos(t)]);n=np.array([np.cos(t),np.sin(t)]);base=.325*n
            p.poly([tuple(base+.15*v),tuple(base-.035*v+.137*n),tuple(base-.035*v-.137*n)])
    elif c in (6,32,41,42):
        if c==6:
            ox,oy=p.x,p.y;p.x=ox/1.19;p.y=oy/.98;number(p,'80',.14,style);p.im*=.57;p.x,p.y=ox,oy
        elif c in (41,42):
            ox,oy=p.x,p.y;p.x=ox/1.32;p.y=oy/(.79 if c==42 else .60);icon(p,9 if c==41 else 10,style);p.im*=.67;p.x,p.y=ox,oy
        stripe=(p.x+p.y)*.7071
        if style%2:p.paint(abs(stripe)<.20,1.)
        else:p.paint((abs(stripe)<.22)&((stripe+.23)%.094<.064),1.)
    elif c==14:
        # Compact block STOP, negative (light lettering on paint).
        for x,letter in zip((-.38,-.128,.12,.37),'STOP'):
            l=x-.098;r=x+.098;t=-.46;b=.46
            if letter=='S':p.path([(r,t+.10),(r-.035,t),(l+.035,t),(l,t+.13),(l,-.1),(r,.06),(r,b-.12),(r-.03,b),(l+.03,b),(l,b-.11)],.073)
            if letter=='T':p.path([(l,t),(r,t)],.086);p.path([(x,t),(x,b)],.085)
            if letter=='O':p.path([(x-.054,t),(x+.054,t),(r,t+.11),(r,b-.11),(x+.054,b),(x-.054,b),(l,b-.11),(l,t+.11),(x-.054,t)],.076)
            if letter=='P':p.path([(l,b),(l,t),(x+.055,t),(r,t+.11),(r,-.09),(x+.055,.025),(l,.025)],.078)
    elif c==17:p.rect(-.50,-.50,.50,.50)

# Generate each glyph once in a large vector viewport, then generate individual
# scale/width variants by area sampling that same drawing. No between-image statistics.
def sample(a,x,y):
    x=np.clip(x,0,a.shape[1]-1.001);y=np.clip(y,0,a.shape[0]-1.001);i=x.astype(int);j=y.astype(int);fx=x-i;fy=y-j
    return (a[j,i]*(1-fx)+a[j,i+1]*fx)*(1-fy)+(a[j+1,i]*(1-fx)+a[j+1,i+1]*fx)*fy

def smooth(a):
    if a.ndim==3:return np.stack([smooth(layer) for layer in a])
    b=np.pad(a,((1,1),(1,1)),mode='edge');return (b[:-2,:-2]+b[:-2,2:]+b[2:,:-2]+b[2:,2:]+2*(b[1:-1,:-2]+b[1:-1,2:]+b[:-2,1:-1]+b[2:,1:-1])+4*b[1:-1,1:-1])/16

def sizes(c):
    if c in (0,1,2,3,4,5,7,8):
        for w in ((1.10,1.25,1.39) if c in (7,8) else (.91,1.06,1.20,1.32)):
            for h in (.78,.91,1.03):yield w,h,0
    elif c in (9,10,16):
        for w in (1.08,1.24,1.39):
            for h in ((.73,.92,1.05) if c in (10,16) else (.53,.64,.78)):yield w,h,0
    elif c in (6,32,41,42):yield 1,1,0
    elif c==17:
        for w in (1.37,1.57):
            for h in (.31,.41):yield w,h,0
    elif c==14:
        for w in (1.49,1.68):
            for h in (.39,.55,.68):yield w,h,0
    elif 33<=c<=40:
        for w in ((.78,.98,1.17) if c==35 else (1.07,1.28,1.46)):
            for h in (1.17,1.38,1.57):yield w,h,0
    else:
        w,h={11:(.88,.96),18:(.59,1.05),19:(.83,.80),20:(.83,.80),21:(.67,.98),22:(1.04,.38),23:(.78,.94),24:(.76,.86),25:(1.13,.91),26:(.49,1.12),27:(1.03,1.06),28:(1.03,.86),29:(1.09,.85),30:(.90,.91),31:(1.11,.80)}[c]
        for ax in (.85,1.,1.15):
            for ay in (.84,1.,1.15):
                height=h*ay;yield w*ax,height,.78-height/2

groups=[[0,1,2,3,4,5,7,8,9,10,16],[11]+list(range(18,32)),list(range(33,41)),[],[],[14],[17],[6,32,41,42]]

# Pool signed gradient orientations on a 5 x 5 grid in an aligned glyph window.
def hog(z):
    g=smooth(z);a=np.pad(g,((0,0),(1,1),(1,1)),mode='edge');dx=a[:,1:-1,2:]-a[:,1:-1,:-2];dy=a[:,2:,1:-1]-a[:,:-2,1:-1];mag=np.sqrt(dx*dx+dy*dy)
    feat=[]
    for w in (dx,dy,.7071*(dx+dy),.7071*(dx-dy)):
        for sign in (1,-1):
            response=np.maximum(sign*w-.58*mag,0);response=smooth(smooth(response));feat.append(response.reshape(-1,5,6,5,6).mean((2,4)))
    v=np.stack(feat,axis=3);ss=(v*v).sum(3,keepdims=True);v=v/(ss+.45*ss.mean((1,2),keepdims=True)+.000002)**.30;v=v.reshape(-1,200)
    return v/(np.sqrt((v*v).sum(1,keepdims=True))+.00001)

def tight_template(p):
    # Canonicalise a *single* sketch by its own foreground distribution.
    ax=(np.arange(201)-100)/100
    def span(w):
        cs=np.cumsum(w);return np.interp([.025,.975],cs/(cs[-1]+1.e-12),ax)
    l,r=span(p.im.sum(0));t,b=span(p.im.sum(1));y,x=(np.mgrid[:90,:90]+.5-45)/45
    z=sample(p.im,100+100*((l+r)/2+x*.63*(r-l)),100+100*((b+t)/2+y*.63*(b-t))).reshape(30,3,30,3).mean((1,3));z=.74*z+.26*smooth(z)
    def unit(u):
        u=u-u.mean();return u/(np.sqrt((u*u).sum())+.0001)
    hp=z-smooth(smooth(smooth(smooth(smooth(z)))))
    return np.stack([unit(z),unit(hp),unit(smooth(smooth(z)))]),hog(z[None])[0],z

def glyphbank(f):
    n=28;y,x=(np.mgrid[:n,:n]+.5)*2/n-1;d=shape(f,x,y)
    margin={0:.205,1:.135,2:.075,5:.09,6:.10,7:.095}.get(f,.10)
    mask=clip((d-margin)/.08+.5);arrays['m'+str(f)]=mask.astype(np.float32)
    if not groups[f]:return
    sy,sx=(np.mgrid[:n*3,:n*3]+.5)*2/(n*3)-1
    gy,gx=(np.mgrid[:201,:201]-100)/100
    bank=[];ids=[];tb=[];hb=[];ti=[];sil=[];dtab=[]
    for c in groups[f]:
        for st in range(4 if c in (0,1,2,3,4,5,7,8,21) else (3 if c in (19,20) else (2 if c not in (18,22,24,28,29,30,14,17) else 1))):
            p=Pen(gx.copy(),gy.copy());icon(p,c,st)
            # Reframe each individual pictogram from its actual stroke bounds,
            # keeping the hand-specified physical dimensions independent of the
            # drawing viewport. A falling road, bicycle or animal need not fill
            # its authoring cell symmetrically.
            if f==1 and c not in (18,26):
                ys,xs=np.nonzero(p.im>.03)
                if len(xs):
                    xl,xr=xs.min(),xs.max();yt,yb=ys.min(),ys.max()
                    # Maintain natural narrow stroke aspect for single people,
                    # but distribute the families' wider multi-part figures.
                    if c not in (27,):p.im=sample(p.im,(xl+xr)/2+gx*(xr-xl),(yt+yb)/2+gy*(yb-yt))
            if f in (0,1,2,5,6):
                tz,hz,zz=tight_template(p);tb.append(tz);hb.append(hz);ti.append(c)
                if f==1:
                    fore=clip((zz-.23)/.59);sil.append(fore/(fore.sum()+1e-7));ny,nx=np.mgrid[:30,:30];dist=np.full((30,30),400.)
                    for iy,ix in zip(*np.nonzero(zz>.43)):dist=np.minimum(dist,(ny-iy)**2+(nx-ix)**2)
                    dtab.append(1/(1+dist/10))
            for w,h,dy in sizes(c):
                img=sample(p.im,sx/w*100+100,(sy-dy)/h*100+100).reshape(n,3,n,3).mean((1,3))
                img=.71*img+.29*smooth(img)
                slow=smooth(smooth(smooth(smooth(img))))
                vec=[]
                for g in (img,img-slow):
                    z=(g-(mask*g).sum()/mask.sum())*np.sqrt(mask);vec.append(z/(np.sqrt((z*z).sum())+.0001))
                bank.append(np.stack(vec));ids.append(c)
    arrays['b'+str(f)]=np.stack(bank,axis=-1).reshape(2,n*n,len(ids)).astype(np.float32)
    arrays['i'+str(f)]=np.array(ids,np.int32)
    if sil:
        arrays['stencil']=np.stack(sil,axis=-1).reshape(900,-1).astype(np.float32);arrays['distance']=np.stack(dtab,axis=-1).reshape(900,-1).astype(np.float32)
    if tb:
        arrays['h'+str(f)]=np.stack(hb,axis=-1).astype(np.float32)
        arrays['t'+str(f)]=np.stack(tb,axis=-1).reshape(3,900,len(ti)).astype(np.float32);arrays['j'+str(f)]=np.array(ti,np.int32)

def profiles(f):
    y,x=(np.mgrid[:32,:32]-15.5)/12;d=shape(f,x,y)
    def band(lo,hi):return clip((d-lo)*10+.5)*clip((hi-d)*10+.5)
    def norm(w):return w/max(1e-12,w.sum())
    outer=band(-.27,-.075);ring=band(.025,.165);face=clip((d-.23)*10+.5)
    if f in (0,1,3):
        # Compare paint and paper separately in each sector of the perimeter.
        origin=.29 if f==1 else (-.29 if f==3 else 0)
        a=np.arctan2(y-origin,x);paper=band(.25,.47);kernels=[]
        for phi in np.arange(8)*np.pi/4:
            sec=clip((np.cos(a-phi)-np.cos(np.pi/3.8))*12)
            kernels.append([norm(ring*sec)-.66*norm(paper*sec)-.12*norm(outer*sec),norm(paper*sec)-norm(ring*sec),norm(ring*sec),norm(paper*sec)])
    else:
        # Here the angular distribution is less important than a uniform ground.
        kernels=[];a=np.arctan2(y,x);inside=clip((d-.08)*8+.5)
        for phi in np.arange(8)*np.pi/4:
            sec=clip((np.cos(a-phi)-np.cos(np.pi/3.8))*12)
            if f==2:w=inside*(.35+.65*band(.09,.43));v=outer
            elif f==4:w=face;v=band(.018,.155)
            elif f in (5,6):w=inside*clip((abs(y)-.25)*9+.5);v=outer
            else:w=inside;v=outer
            kernels.append([norm(w*sec)-.27*norm(v*sec),norm(v*sec)-norm(w*sec),norm(w*sec),norm(v*sec)])
    arrays['pro'+str(f)]=np.array(kernels,np.float32).transpose(1,2,3,0).reshape(4,1024,8).copy()

for fam in range(8):
    locbank(fam,0);locbank(fam,1);glyphbank(fam);profiles(fam)

# Independent single-character masks. These are glyph drawings, not extracted patches.
yc,xc=(np.mgrid[:201,:201]-100)/100
chars=[];charh=[];charids=[]
for ch in '01235678':
    for st in range(4):
        pen=Pen(xc.copy(),yc.copy());number(pen,ch,(.115,.14,.16,.18)[st],st)
        tz,hz,zz=tight_template(pen);chars.append(tz);charh.append(hz);charids.append(int(ch))
arrays['dh']=np.stack(charh,axis=-1).astype(np.float32)
arrays['digits']=np.stack(chars,axis=-1).reshape(3,900,len(chars)).astype(np.float32)
arrays['didx']=np.array(charids,np.int32)

SOURCE = r'''import numpy as np

# Convert the small template banks once. None of the following state changes between calls.
kernels = [[CONSTANTS['k'+str(f)+str(s)].astype(np.float32) for s in range(2)] for f in range(8)]
poses = [[CONSTANTS['p'+str(f)+str(s)] for s in range(2)] for f in range(8)]
families = [[0,1,2,3,4,5,7,8,9,10,16],[11]+list(range(18,32)),list(range(33,41)),[13],[12],[14],[17],[6,32,41,42]]

def central(a):return a[:,6:26,6:26].reshape(len(a),400)
def percentile(a,q):return np.percentile(a,q,axis=1).astype(np.float32)[:,None,None]
def cropfit(v,a,b):return np.minimum(np.maximum(v,a),b)
def smooth(a):
    b=np.pad(a,((0,0),(1,1),(1,1)),mode='edge')
    return (b[:,:-2,:-2]+b[:,:-2,2:]+b[:,2:,:-2]+b[:,2:,2:]+2*(b[:,1:-1,:-2]+b[:,1:-1,2:]+b[:,:-2,1:-1]+b[:,2:,1:-1])+4*b[:,1:-1,1:-1])/16

def hog(z):
    g=smooth(z);a=np.pad(g,((0,0),(1,1),(1,1)),mode='edge');dx=a[:,1:-1,2:]-a[:,1:-1,:-2];dy=a[:,2:,1:-1]-a[:,:-2,1:-1];mag=np.sqrt(dx*dx+dy*dy)
    feat=[]
    for w in (dx,dy,.7071*(dx+dy),.7071*(dx-dy)):
        for sign in (1,-1):
            response=np.maximum(sign*w-.58*mag,0);response=smooth(smooth(response));feat.append(response.reshape(-1,5,6,5,6).mean((2,4)))
    v=np.stack(feat,axis=3);ss=(v*v).sum(3,keepdims=True);v=v/(ss+.45*ss.mean((1,2),keepdims=True)+.000002)**.30;v=v.reshape(-1,200)
    return v/(np.sqrt((v*v).sum(1,keepdims=True))+.00001)

def features(batch):
    r,g,b=[batch[:,k] for k in range(3)]
    brightness=.29*r+.53*g+.18*b
    floor=.006+.29*percentile(central(brightness),55)
    up=percentile(central(brightness),85)+.012
    def tone(a):
        raw=np.log1p(a/floor);res=[]
        for v in (raw,a):
            q0=percentile(central(v),8);q1=percentile(central(v),89)
            res.append(cropfit((v-q0)/(q1-q0+.014),-.38,1.42))
        return .60*res[0]+.40*res[1]
    l=tone(brightness);gg=tone(g);rr=tone(r)
    offs=.006+.033*up
    def chrom(a,bb):return np.log((a+offs)/(bb+offs))
    def paint(v):
        base=percentile(central(v),30)
        relative=v-cropfit(base,-.38,.55)
        return cropfit(.71*relative+.29*v,-.48,1.45)
    red=paint(np.minimum(chrom(r,g),chrom(r,b)+.15))
    blue=paint(np.minimum(chrom(b,r),1.5*chrom(b,g)+.20))
    yellow=paint(np.minimum(chrom(r,b),chrom(g,b)+.17))
    saturate=np.maximum.reduce([red,blue,yellow,np.zeros_like(red)])
    edge=0
    for a,weight in ((l,.64),(red,.20),(blue,.16)):
        z=np.pad(a,((0,0),(1,1),(1,1)),mode='edge');dx=z[:,1:-1,2:]-z[:,1:-1,:-2];dy=z[:,2:,1:-1]-z[:,:-2,1:-1]
        edge=edge+weight*np.minimum(1.6,np.sqrt(dx*dx+dy*dy))
    return [red,blue,yellow,saturate,l,gg,rr,edge,r,g,b]

def groupmaps(m,f):
    red,blue,yellow,sat,l,gg,rr,edge=m[:8]
    if f==2:return [blue,.64*rr+.36*l,edge,red]
    if f==4:return [yellow,l,edge,blue]
    if f==7:return [sat,l,edge,sat]
    return [red,.71*gg+.29*l,edge,blue]

# Hold the BLAS row dimension fixed, even at a batch boundary. This keeps the
# same arithmetic path for a given example in different calling contexts.
def product(a,b):
    nr=a.shape[0];n=((nr+31)//32)*32
    if nr==n:return np.ascontiguousarray(a)@b
    t=np.zeros((n,a.shape[1]),np.float32);t[:nr]=a
    return (t@b)[:nr]

def location(m,f,stage):
    k=kernels[f][stage];p=poses[f][stage]
    def dot(j):return product(m[j].reshape(len(m[j]),1024),k[j])
    color=dot(0) if f!=7 else 0
    lig=dot(1);edge=dot(2)
    # Evidence for ink/paint, for the pigment boundary, and for the sign outline.
    if f in (0,1,3):v=.75*color+.64*lig+.39*edge
    elif f==2:v=.88*color+.62*lig+.54*edge+.02
    elif f==4:v=.83*color+.67*lig+.52*edge
    elif f==5:v=.85*color+.71*lig+.50*edge-.025
    elif f==6:v=.83*color+.70*lig+.50*edge-.025
    else:v=.95*lig+.47*edge-.41*dot(3)+.045
    if stage==0:
        v=v-.002*((p[:,0]-15.5)**2+(p[:,1]-15.5)**2)-.008*np.abs(p[:,2]-12.5)
    else:
        v=v-.014*(abs(p[:,4])+abs(p[:,5]))-.003*(abs(p[:,0]-15.5)+abs(p[:,1]-15.5))
    idx=v.argmax(1);return v[np.arange(len(idx)),idx],p[idx]

def coordinates(p,x,y,s=1):
    x=x*s;y=y*s
    return (p[:,0,None]+p[:,2,None]*(x+p[:,4,None]*y),p[:,1,None]+p[:,3,None]*(y+p[:,5,None]*x))

def sampler(x,y):
    x=cropfit(x,0,30.99999);y=cropfit(y,0,30.99999);i=x.astype(np.int32);j=y.astype(np.int32);fx=(x-i).astype(np.float32);fy=(y-j).astype(np.float32)
    return i+j*32+np.arange(len(i),dtype=np.int32)[:,None]*1024,fx,fy

def sample(a,where):
    ij,fx,fy=where;v=a.reshape(-1);return (v[ij]*(1-fx)+v[ij+1]*fx)*(1-fy)+(v[ij+32]*(1-fx)+v[ij+33]*fx)*fy

y32,x32=np.mgrid[:32,:32].astype(np.float32);x32=(x32.ravel()-15.5)/12;y32=(y32.ravel()-15.5)/12
yg,xg=np.mgrid[:28,:28].astype(np.float32);xg=(xg.ravel()-13.5)/14;yg=(yg.ravel()-13.5)/14

def refine(m,f,p):
    w=sampler(*coordinates(p,x32,y32));mm=[sample(v,w).reshape(len(p),32,32) for v in m]
    val,q=location(mm,f,1)
    # Compose coarse diagonal transform with the interior affine transform.
    pose=q.copy();pose[:,0]=p[:,0]+(q[:,0]-15.5)*p[:,2]/12;pose[:,1]=p[:,1]+(q[:,1]-15.5)*p[:,3]/12
    pose[:,2:4]=q[:,2:4]*p[:,2:4]/12
    return val,pose

# Small residual displacements account for variations in centring of the printed symbol.
shifts=[(a,b,t) for a in (-.057,0,.057) for b in (-.057,0,.057) for t in (0,)]
shifts+= [(0,0,-.095),(0,0,.095),(-.095,0,0),(.095,0,0),(0,.103,0),(0,-.103,0)]
shiftx=np.concatenate([xg-t*yg+a for a,b,t in shifts]);shifty=np.concatenate([yg+t*xg+b for a,b,t in shifts]);ns=len(shifts)

# The second alignment does not assume a particular printed symbol size. Bound
# the foreground by the marginal distribution of contrast within the sign face.
y30,x30=np.mgrid[:30,:30].astype(np.float32);x30=(x30.ravel()-14.5)/15;y30=(y30.ravel()-14.5)/15
fine=[(u,v,0,0) for u,v in ((1.,1.),(.87,.87),(1.14,1.14),(.88,1.10),(1.10,.88))]+[(1,1,a,b) for a,b in ((.12,0),(-.12,0),(0,-.12),(0,.12))]
finex=np.concatenate([x30*u+a for u,v,a,b in fine]);finey=np.concatenate([y30*v+b for u,v,a,b in fine]);nf=len(fine)

def extent(mass):
    count=len(mass);cs=np.cumsum(mass,axis=1);total=cs[:,-1:]
    def quantile(q):
        j=np.argmax(cs>=q*total,axis=1);massj=mass[np.arange(count),j];rel=(q*total[:,0]-cs[np.arange(count),j])/np.maximum(massj,.00001)+1
        return (j.astype(np.float32)+cropfit(rel,0,1))*2/28-1
    return quantile(.025),quantile(.975)

def letters(a,p,ink,whole):
    # Parse a two- or three-character word; optimize cuts only within the likely
    # inter-character gaps. The last zero provides a consistency check.
    cnt=len(p);row=np.arange(cnt);left,right,top,bot=whole;wid=right-left
    grid=xg.reshape(28,28)[0];ink=ink.reshape(cnt,28,28);hist=ink.sum(1)
    def gap(frac):
        center=left+wid*frac;dist=abs(grid[None,:]-center[:,None])/(wid[:,None]+.02)
        local=hist/(.10+hist.max(1,keepdims=True))+.85*dist+10*(dist>.125)
        return grid[local.argmin(1)]
    def letter(lo,hi):
        region=(grid[None,:]>=lo[:,None])&(grid[None,:]<hi[:,None]);mass=ink*region[:,None,:];l,r=extent(mass.sum(1));t,b=extent(mass.sum(2));cx=.5*(l+r);cy=.22*(t+b)+.28*(bot+top)
        rx=.63*cropfit(r-l,.075,.64);ry=.63*cropfit(.62*(b-t)+.38*(bot-top),.26,1.20)
        w=sampler(*coordinates(p,cx[:,None]+rx[:,None]*finex,cy[:,None]+ry[:,None]*finey));im=sample(a,w).reshape(cnt*nf,30,30)
        def unit(z):
            z=z.reshape(-1,900);z=z-z.mean(1,keepdims=True);return z/np.sqrt((z*z).sum(1,keepdims=True)+.045)
        g=smooth(smooth(im));slow=smooth(smooth(smooth(g)));db=CONSTANTS['digits'];ids=CONSTANTS['didx']
        v=.52*product(unit(im),db[0])+.34*product(unit(im-slow),db[1])+.14*product(unit(g),db[2]);oh=product(hog(im),CONSTANTS['dh']);v=(.75*v+.25*(1.6*oh-.60)).reshape(cnt,nf,len(ids)).max(1)
        return dict((k,v[:,ids==k].max(1)) for k in (0,1,2,3,5,6,7,8))
    mid=gap(.50);ll=letter(left-.06,mid);rr=letter(mid,right+.06)
    double=[.65*ll[n]+.35*rr[0] for n in (2,3,5,6,7,8)]
    i=gap(.265);j=gap(.638);la=letter(left-.06,i);lb=letter(i,j);lc=letter(j,right+.06)
    triple=[.27*la[1]+.47*lb[n]+.26*lc[0] for n in (0,2)]
    return np.stack(double+triple,axis=1)

def foreground(a,m,f,p,mask):
    cnt=len(p);w=sampler(*coordinates(p,xg,yg));z=sample(a,w);rd=sample(m[0],w);bl=sample(m[1],w)
    if f in (0,1): ink=z-.26*np.maximum(rd,0)
    else:ink=z
    core=(mask.ravel()>.80);m0=ink[:,core];bg=np.percentile(m0,20,axis=1).astype(np.float32)[:,None];fg=np.percentile(m0,94,axis=1).astype(np.float32)[:,None]
    ink=cropfit((ink-bg)/(.015+fg-bg),0,1.5)
    if f in (0,1):
        # More conservative frame exclusion for segmentation than for matching.
        safe=(mask>.91)
        if f==0:safe=safe&(xg*xg+yg*yg<.725**2)
        else:safe=safe&(yg<.80)&(yg>2*np.abs(xg)-.58)
    else:safe=mask>.85
    bank=CONSTANTS['t'+str(f)];ids=CONSTANTS['j'+str(f)];best=None;ocr=None
    for th in (.28,.52):
        mass=(cropfit((ink-th)*2.7,0,1)*safe).reshape(cnt,28,28);left,right=extent(mass.sum(1));top,bot=extent(mass.sum(2))
        if f==0 and ocr is None:ocr=letters(a,p,mass,(left,right,top,bot))
        cx=(left+right)*.5;cy=(top+bot)*.5;rx=.63*cropfit(right-left,.12,1.72);ry=.63*cropfit(bot-top,.17,1.72)
        sx=cx[:,None]+rx[:,None]*finex;sy=cy[:,None]+ry[:,None]*finey
        w2=sampler(*coordinates(p,sx,sy));image=sample(a,w2).reshape(cnt*nf,30,30)
        def unit(z):
            z=z.reshape(-1,900);z=z-z.mean(1,keepdims=True);return z/np.sqrt((z*z).sum(1,keepdims=True)+.035)
        g=smooth(smooth(image));slow=smooth(smooth(smooth(g)));v=.51*product(unit(image),bank[0])+.28*product(unit(image-slow),bank[1])+.21*product(unit(g),bank[2]);oh=product(hog(image),CONSTANTS['h'+str(f)]);v=.67*v+.33*(1.6*oh-.60)
        if f==1:
            # Symmetric soft overlap: account for ink not explained by the sketch
            # as well as sketch strokes missing from the photograph.
            lo=np.repeat(bg,nf,axis=0);hi=np.repeat(fg,nf,axis=0);inked=cropfit(((image.reshape(-1,900)-lo)/(.025+hi-lo)-.25)*1.55,0,1).reshape(-1,30,30);pi=inked.reshape(-1,900);prec=product(pi/(pi.sum(1,keepdims=True)+.01),CONSTANTS['distance'])
            wide=np.pad(inked,((0,0),(2,2),(2,2)),mode='edge');near=np.maximum.reduce([wide[:,i:i+30,j:j+30] for i,j in ((0,2),(2,0),(4,2),(2,4),(1,1),(3,3),(3,1),(1,3),(2,2))]);rec=product((.25*inked+.75*near).reshape(-1,900),CONSTANTS['stencil']);chamfer=2*prec*rec/(prec+rec+.00001)
            v=.75*v+.25*(1.9*chamfer-.90)
        v=v.reshape(cnt,nf,len(ids)).max(1)
        best=v if best is None else np.maximum(best,v)
    return np.stack([best[:,ids==c].max(1) for c in families[f]],axis=1),ocr

def localtone(a,idx):
    h=a[:,idx];lo=np.percentile(h,7,axis=1).astype(np.float32)[:,None];hi=np.percentile(h,86,axis=1).astype(np.float32)[:,None];med=np.percentile(h,50,axis=1).astype(np.float32)[:,None]
    # Differences within a dark sign should not be compressed by a bright sky.
    lin=cropfit((a-lo)/(hi-lo+.006),-.28,1.30);floor=.004+.32*med
    l0=np.log(lo+floor);log=cropfit((np.log(a+floor)-l0)/(np.log(hi+floor)-l0+.028),-.28,1.30)
    return .58*log+.42*lin

def canvas(m,f,p):
    w=sampler(*coordinates(p,x32,y32));raw=[sample(z,w) for z in m[8:11]]
    mask=CONSTANTS['m'+str(f)].reshape(28,28);ii=np.clip(np.rint(x32*14+13.5).astype(np.int32),0,27);jj=np.clip(np.rint(y32*14+13.5).astype(np.int32),0,27);inside=(mask[jj,ii]>.59)&(abs(x32)<1)&(abs(y32)<1)
    lum=.30*raw[0]+.52*raw[1]+.18*raw[2]
    local=[sample(z,w).reshape(-1,32,32) for z in m[:8]]
    for i,ch in ((4,lum),(5,raw[1]),(6,raw[0])):local[i]=.88*localtone(ch,inside).reshape(-1,32,32)+.12*local[i]
    pp=np.zeros_like(p);pp[:,:2]=15.5;pp[:,2:4]=12
    # A uniform blank face may have a strong illumination gradient; test fine
    # structure relative to its *own red rim*, rather than absolute brightness.
    L=(.65*lum+.35*raw[0]).reshape(-1,32,32);lm=smooth(smooth(L));rad=np.sqrt(x32*x32+y32*y32);inn=rad<.63;rim=(rad>.835)&(rad<.955)
    hp=(smooth(L)-smooth(smooth(smooth(lm)))).reshape(-1,1024);swing=np.percentile(hp[:,inn],95,axis=1)-np.percentile(hp[:,inn],5,axis=1)
    val=L.reshape(-1,1024);paper=np.percentile(val[:,inn],80,axis=1);paint=np.median(val[:,rim],axis=1);activity=(swing/(.004+.15*paper+abs(paper-paint))).astype(np.float32)
    return local,pp,activity

def scoresymbol(m,f,p,activity):
    red,blue,yellow,sat,l,gg,rr,edge=m[:8];cnt=len(p)
    mask=CONSTANTS['m'+str(f)].reshape(1,-1);mw=np.sqrt(mask);den=mask.sum()
    # Invert dark figures on paper; for blue or solid red plates seek light figures.
    if f in (0,1):
        a=-(.40*l+.38*rr+.22*gg)
        # A red outline is not black ink. Keep it from becoming a false stroke.
        a=a+.22*np.maximum(red,0)
    elif f==2:a=.48*rr+.52*l-.32*blue
    elif f in (5,6):a=.64*gg+.36*l-.22*red
    else:a=-l
    if f in (0,1,2):
        # Subtract a coarse reflected background. Outside the paper use its typical
        # light tone, not the road or sky behind the sign.
        interior=(mask.reshape(28,28)[np.clip(np.rint(y32*14+13.5).astype(np.int32),0,27),np.clip(np.rint(x32*14+13.5).astype(np.int32),0,27)]>.65)&(abs(x32)<1)&(abs(y32)<1)
        paper=np.percentile(a.reshape(cnt,1024)[:,interior],20,axis=1).astype(np.float32)[:,None,None];field=np.where(interior.reshape(1,32,32),a,paper)
        # Integral image box mean; padding carries the paper level.
        acc=np.pad(field,((0,0),(4,5),(4,5)),mode='edge').cumsum(1).cumsum(2);low=(acc[:,9:,9:]+acc[:,:-9,:-9]-acc[:,9:,:-9]-acc[:,:-9,9:])/81
        a=a-.45*(low-paper)
    q=sampler(*coordinates(p,shiftx,shifty));raw=sample(a,q).reshape(cnt*ns,28,28)
    # Subtract a broad, within-photo background estimate; preserve weak features in deep shade.
    x=raw.reshape(-1,784);midx=(mask.ravel()>.60)
    v0=np.percentile(x[:,midx],10,axis=1).astype(np.float32)[:,None];v1=np.percentile(x[:,midx],93,axis=1).astype(np.float32)[:,None]
    width=v1-v0+.014;x=cropfit(x,v0-.30*width,v1+.27*width);raw=x.reshape(-1,28,28)
    slow=smooth(smooth(smooth(smooth(raw))));h=(raw-slow).reshape(-1,784)
    def normal(z,floor):
        z=(z-(z*mask).sum(1,keepdims=True)/den)*mw
        return z/np.sqrt((z*z).sum(1,keepdims=True)+floor)
    bank=CONSTANTS['b'+str(f)];idx=CONSTANTS['i'+str(f)]
    sc=.76*product(normal(x,.020),bank[0])+.24*product(normal(h,.008),bank[1]);sc=sc.reshape(cnt,ns,len(idx)).max(1)
    direct=np.stack([sc[:,idx==c].max(1) for c in families[f]],axis=1)
    if f!=7:
        tight,ocr=foreground(a,m,f,p,mask)
        direct=.49*direct+.37*tight+.14*np.maximum(direct,tight)
        if f==0:direct[:,:8]=.70*direct[:,:8]+.30*ocr
    ans=[direct[:,i] for i in range(len(families[f]))]
    if f==0:
        # Blank face: the body has no lettering or pictogram. Measure its internal
        # contrast relative to the surrounding ring rather than the camera exposure.
        w=sampler(*coordinates(p,xg,yg));z=sample(a,w);rad=np.sqrt(xg*xg+yg*yg);inner=z[:,rad<.64];outer=z[:,(rad>.845)&(rad<.97)]
        qlo=np.percentile(inner,20,axis=1);hi=np.percentile(inner,93,axis=1);contrast=hi-qlo
        border=np.maximum(.04,outer.mean(1)-qlo);blank=.84-.62*contrast/(.13+border)-.18*contrast
        blank=.35*blank+.65*(.83-1.75*activity)
        # A featureless face should not have a convincing non-blank matched symbol.
        blank=blank-.65*np.maximum(0,direct.max(1)-.40);ans.append(cropfit(blank,-.25,.77))
        # The left vehicle of an overtaking-ban sign is painted red, not black.
        z=sample(red,w);left=(xg<-.07)&(xg>-.56)&(abs(yg)<.24);right=(xg>.07)&(xg<.56)&(abs(yg)<.24)
        bonus=cropfit(z[:,left].mean(1)-z[:,right].mean(1),-.30,1.0)
        ans[8]=ans[8]+.19*bonus;ans[9]=ans[9]+.19*bonus
    return np.stack(ans,axis=1)

def support(local,f):
    # A chance patch of the right colour should not substitute for a complete rim.
    co,li,ed,re=groupmaps(local,f);k=CONSTANTS['pro'+str(f)];n=len(co);color=product(co.reshape(n,1024),k[0]);absolute=product(co.reshape(n,1024),k[2]);lig=product(li.reshape(n,1024),k[1]);level=product(li.reshape(n,1024),k[2])
    if f in (0,1,3):ev=.70*color+.57*lig
    elif f==2:ev=.85*color
    elif f==4:ev=.79*color+.60*lig
    elif f in (5,6):ev=.85*color
    else:return -.20*np.median(absolute,axis=1)
    ordered=np.sort(ev,axis=1)
    return .20*ordered[:,1:4].mean(1)+.075*ordered[:,4:7].mean(1)-.045

def process(batch):
    m=features(batch);count=len(batch);base=np.zeros((count,8),np.float32);pos=[]
    for f in range(8):
        v,p=location(groupmaps(m,f),f,0);base[:,f]=v;pos.append(p)
    # Hierarchical verification: shortlist face shapes, then consider their interiors.
    order=np.argsort(-base,axis=1);result=np.full((count,43),-100.,np.float32)
    for f in range(8):
        sel=np.nonzero((order[:,:4]==f).any(1))[0]
        for start in range(0,len(sel),16):
            take=sel[start:start+16];s=[v[take] for v in m];cm=groupmaps(s,f);v,p=refine(cm,f,pos[f][take]);geo=.42*base[take,f]+.58*v
            orig=s;originalpose=p;s,p,activity=canvas(s,f,p);geo=geo+support(s,f)
            if f in (3,4):
                # Both have a nearly featureless, uniform face.
                w=sampler(*coordinates(originalpose,xg,yg));z=sample(orig[4],w);inside=(abs(xg)+abs(yg)<.58) if f==4 else ((yg>-.65)&(yg<.32-1.94*abs(xg)))
                iz=z[:,inside];span=np.percentile(iz,90,axis=1)-np.percentile(iz,10,axis=1)
                match=cropfit(.61-.51*span,.07,.61)[:,None]
            else:match=scoresymbol(s,f,p,activity)
            cls=families[f]+([15] if f==0 else [])
            # Template NCC is a second, independent witness in addition to the frame.
            result[np.ix_(take,cls)]=geo[:,None]+.86*(match-.50)
    return result.argmax(1)

def predict(x):
    result=np.empty(len(x),dtype=np.int64)
    for i in range(0,len(x),32):result[i:i+32]=process(x[i:i+32])
    return result
'''

np.savez(out/'constants.npz',**arrays)
(out/'solution.py').write_text(SOURCE)
