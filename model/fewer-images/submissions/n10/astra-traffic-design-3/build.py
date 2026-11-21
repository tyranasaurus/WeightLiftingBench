"""Construct a pose-search and pictogram-reading circuit. No data or font files."""
import sys, json, math
from pathlib import Path
import numpy as np

F=np.float32
# Coordinates in the sign drawings: the centre is (0,0), the bounding box is [-1,1]^2.
# This little vector pen supports filled polygons/ellipses, round strokes and quadratic arcs.
class Pen:
    def __init__(self,x,y):
        self.x=x; self.y=y; self.a=np.zeros_like(x)
    def rule(self,mask,v): self.a[mask]=v
    def ellipse(self,x,y,rx,ry,v=1):self.rule(((self.x-x)/rx)**2+((self.y-y)/ry)**2<=1,v)
    def dot(self,x,y,r,v=1): self.ellipse(x,y,r,r,v)
    def poly(self,ps,v=1):
        mask=np.zeros(self.x.shape,bool); x=self.x;y=self.y
        for (a,b),(c,d) in zip(ps,ps[1:]+ps[:1]):
            if d!=b: mask ^= ((b>y)!=(d>y)) & (x<(c-a)*(y-b)/(d-b)+a)
        self.rule(mask,v)
    def line(self,pts,width,v=1):
        for (a,b),(c,d) in zip(pts,pts[1:]):
            u=c-a; w=d-b
            t=np.clip(((self.x-a)*u+(self.y-b)*w)/(u*u+w*w+1e-15),0.,1.)
            self.rule((self.x-a-u*t)**2+(self.y-b-w*t)**2<=(width/2)**2,v)
    def curve(self,a,b,c,width,v=1): self.line(quad(a,b,c),width,v)

def quad(a,b,c):
    return [tuple((1-t)**2*i+2*(1-t)*t*j+t*t*k for i,j,k in zip(a,b,c)) for t in np.linspace(0,1,10)]

def rim(x,y,f,t=1):
    if f in ('warn','yield'):
        yy=y if f=='warn' else -y
        # triangle vertices (0,-1), (-1,1), (1,1); inset about the centroid
        return (yy<=1/3+t*2/3)&(2*np.abs(x)-yy<=t*4/3-1/3)
    if f=='diamond': return abs(x)+abs(y)<t
    if f=='stop':return (np.maximum(abs(x),abs(y))<t)&(abs(x)+abs(y)<t*math.sqrt(2))
    return x*x+y*y<t*t

def family(k):
    if k==11 or 18<=k<=31:return 'warn'
    if k==12:return 'diamond'
    if k==13:return 'yield'
    if k==14:return 'stop'
    if k==17:return 'entry'
    if 33<=k<=40:return 'blue'
    if k in (6,32,41,42):return 'end'
    return 'red'

# DIN-style, monoline numerals. Paths below are in a digit's own bounding box.
def digit(p,ch,variant=0):
    w=(.16,.19,.18,.22)[variant%4]
    if ch=='0':
        # DIN counter: parallel stems and rounded top/bottom, in addition
        # to the regular oval variant. Width and height have different units.
        if variant%4 in (1,3):
            center=np.clip(p.y,.29,.71);p.rule(((p.x-.5)/.50)**2+((p.y-center)/.29)**2<1,1);p.rule(((p.x-.5)/.30)**2+((p.y-center)/.177)**2<1,0)
        else:p.ellipse(.5,.5,.50,.50);p.ellipse(.5,.5,.29,.365,0)
    if ch=='1':
        p.poly([(.07,.225),(.54,.015),(.78,.015),(.78,.99),(.48,.99),(.48,.18),(.07,.345)])
    if ch=='2':
        p.line(quad((.12,.24),(.15,.045),(.51,.065))+quad((.51,.065),(.93,.065),(.87,.32))[1:]+[(.14,.916),(.91,.916)],w)
    if ch=='3':
        p.line(quad((.15,.18),(.32,.015),(.67,.095))+quad((.67,.095),(1.1,.30),(.51,.47))[1:]+quad((.51,.47),(.95,.48),(.87,.75))[1:]+quad((.87,.75),(.73,1.085),(.12,.82))[1:],w)
    if ch=='5':
        p.line([(.83,.085),(.24,.085),(.18,.47)]+quad((.18,.47),(.50,.29),(.75,.46))[1:]+quad((.75,.46),(1.04,.70),(.73,.875))[1:]+quad((.73,.875),(.44,1.045),(.12,.81))[1:],w)
    if ch=='6':
        if variant%2:p.line(quad((.76,.12),(.28,-.04),(.17,.55)),w)
        else:p.line(quad((.70,.06),(.25,.35),(.17,.60)),w)
        p.ellipse(.505,.678,.404,.316);p.ellipse(.505,.678,.214,.18,0)
    if ch=='7':p.line([(.09,.08),(.90,.08),(.38,.94)],w)
    if ch=='8':
        p.ellipse(.5,.272,.396,.269);p.ellipse(.5,.723,.455,.277);p.ellipse(.5,.270,.202,.133,0);p.ellipse(.5,.721,.250,.143,0)

def numerals(x,y,text,style):
    h=(.73,.82,.68,.79)[style%4] if len(text)==2 else (.82,.88,.74,.85)[style%4]
    width=h*(.59,.56,.65,.64)[style%4] if len(text)==2 else h*(.50,.45,.55,.51)[style%4]
    gap=h*(.13,.14,.115,.10)[style%4]; widths=[width*(.80 if c=='1' else 1) for c in text];left=-(sum(widths)+(len(text)-1)*gap)/2
    arr=np.zeros_like(x)
    for c,d in zip(text,widths):
        p=Pen((x-left)/d,(y+.5*h)/h);digit(p,c,style);arr=np.maximum(arr,p.a);left+=d+gap
    return arr

def car(p,cx,cy,w,h,v=1):
    def tr(pts):return [(cx+x*w,cy+y*h) for x,y in pts]
    p.poly(tr([(-.49,-.07),(-.37,-.22),(-.30,-.48),(.26,-.48),(.39,-.18),(.5,-.05),(.5,.34),(.40,.34),(.40,.49),(.23,.49),(.23,.34),(-.25,.34),(-.25,.49),(-.42,.49),(-.42,.33),(-.49,.33)]),v)
    p.poly(tr([(-.33,-.13),(-.23,-.37),(.20,-.37),(.30,-.13)]),0)

def truck(p,cx,cy,w,h,v=1,side=True):
    def tr(pts):return [(cx+x*w,cy+y*h) for x,y in pts]
    if not side:
        p.poly(tr([(-.45,-.56),(.44,-.56),(.44,.38),(.33,.50),(-.34,.50),(-.47,.31)]),v)
        p.poly(tr([(-.36,-.01),(.34,-.01),(.34,.16),(-.36,.16)]),0);return
    p.poly(tr([(-.50,-.01),(-.41,-.33),(-.20,-.33),(-.20,-.48),(.49,-.48),(.49,.23),(.41,.23),(.37,.32),(-.50,.32)]),v)
    p.poly(tr([(-.44,-.015),(-.365,-.24),(-.25,-.24),(-.25,-.015)]),0)
    for a in (-.33,.23,.38):p.ellipse(cx+a*w,cy+.31*h,.10*h,.18*h,v)

def walker(p,cx,cy,s,flip=1,skirt=False):
    def pt(x,y):return (cx+flip*s*x,cy+s*y)
    def ln(points,w):p.line([pt(x,y) for x,y in points],w*s)
    p.dot(*pt(.09,-.46),.105*s);ln([(.04,-.31),(-.06,.04)],.18)
    ln([(.005,-.23),(.22,-.05),(.31,-.10)],.075);ln([(.0,-.24),(-.18,-.09),(-.30,-.06)],.08)
    if skirt:p.poly([pt(-.05,-.22),pt(-.23,.14),pt(.08,.14)])
    ln([(-.06,.03),(-.14,.28),(-.30,.53)],.105);ln([(-.06,.04),(.12,.30),(.25,.50)],.112)

def stopword(p):
    for j,c in enumerate('STOP'):
        a=Pen((p.x-(-.79+.40*j))/.34,(p.y+.25)/.50)
        if c=='S': a.line(quad((.87,.15),(.29,-.08),(.13,.19))+quad((.13,.19),(-.08,.39),(.53,.50))[1:]+quad((.53,.50),(1.1,.62),(.82,.83))[1:]+quad((.82,.83),(.55,1.05),(.12,.83))[1:],.21)
        elif c=='T':a.line([(.08,.10),(.92,.10)],.19);a.line([(.50,.10),(.50,.93)],.23)
        elif c=='O':digit(a,'0')
        else:a.line([(.13,.93),(.13,.09),(.54,.09)]+quad((.54,.09),(1.2,.15),(.78,.45))[1:]+[(.13,.47)],.21)
        p.a=np.maximum(p.a,a.a)

def symbol(k,x,y,style):
    # One drawing per candidate. Style indices select prescribed variations, not statistics.
    p=Pen(x,y);s=style%4
    if k in (0,1,2,3,4,5,7,8):return numerals(x,y,{0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[k],s)
    if k in (9,10,16,41,42):
        if k==16:truck(p,0,.01,1.38,.66)
        else:
            if k in (9,41):car(p,-.36,.02,.60,.42,.75 if k==9 else 1)
            else:truck(p,-.34,-.045,.65,.66,.75 if k==10 else 1,False)
            car(p,.34,.055,.58,.40)
    if k==11:p.poly([(-.13,.69),(.13,.69),(.13,.32),(.34,.32),(.34,.16),(.13,.16),(.13,-.04),(.21,-.04),(0,-.40),(-.21,-.04),(-.13,-.04),(-.13,.16),(-.34,.16),(-.34,.32),(-.13,.32)])
    if k==12:p.rule(abs(x)+abs(y)<.71,1)
    if k==14:stopword(p)
    if k==17:p.rule((abs(x)<.74)&(abs(y)<.19),1)
    if k==18:
        p.poly([(-.083,-.36),(.083,-.36),(.055,.43),(-.055,.43)]);p.dot(0,.598,.093)
    if k in (19,20):
        a=1 if k==19 else -1
        pts=[(.20,.65),(.20,.32)]+quad((.20,.32),(.20,-.005),(-.09,-.005))[1:]+[(-.25,-.005)]
        p.line([(a*b,c) for b,c in pts],.205);p.rule((y>.63)|(a*x<-.25),0)
    if k==21:
        p.line([(.20,.62),(.20,.24)]+quad((.20,.24),(.20,.09),(.068,.09))[1:]+[(-.08,.09)]+quad((-.08,.09),(-.16,.09),(-.16,-.01))[1:]+[(-.16,-.28)],.205);p.rule((y<-.28)|(y>.62),0)
    if k==22:
        p.poly([(-.64,.62),(-.57,.52),(-.47,.51),(-.37,.38),(-.25,.35),(-.09,.49),(.02,.48),(.14,.30),(.25,.29),(.40,.45),(.54,.5),(.65,.62)])
    if k==23:
        # A skidding car over a pair of meandering tracks. Also include the
        # profile version: a long body with a cabin over a pair of wheels.
        c=Pen(x+.16*(y+.10),y)
        if s%2:car(c,0,-.085,.63,.37)
        else:
            c.poly([(-.39,-.09),(-.29,-.12),(-.16,-.23),(.035,-.23),(.16,-.13),(.29,-.10),(.35,.02),(.30,.095),(-.34,.095)]);c.dot(-.21,.087,.077);c.dot(.20,.087,.077)
        p.a=c.a
        p.line(quad((-.12,.21),(-.16,.36),(.25,.32))+quad((.25,.32),(.49,.40),(-.15,.51))[1:]+[(-.35,.61)],.065)
        p.line(quad((.18,.14),(.13,.26),(.43,.27))+quad((.43,.27),(.65,.42),(.20,.51))[1:]+[(.02,.60)],.061)
    if k==24:
        p.line([(-.205,.62),(-.205,-.24)],.095);p.line([(.255,.62),(.255,.24),(.092,.01),(.092,-.24)],.106)
    if k==25:
        p=Pen(-x,y)
        p.dot(.075,-.19,.098);p.line([(.10,-.03),(.28,.19),(.29,.35)],.145);p.line([(.24,.17),(.10,.36),(.04,.62)],.115);p.line([(.27,.24),(.44,.43),(.52,.60)],.115);p.line([(.10,.0),(-.02,.17),(-.17,.20)],.072);p.line([(-.39,.51),(.21,.03)],.048);p.poly([(-.63,.64),(-.51,.50),(-.38,.42),(-.23,.53),(-.14,.64)])
    if k==26:
        p.line([(0,-.255),(0,.52)],.27,.70)
        for cy,col in [(-.21,.48),(.11,.24),(.45,.67)]:p.dot(0,cy,.114,col)
    if k==27:walker(p,0,.16,.83,-1)
    if k==28:
        walker(p,.15,.105,.95,1,True);walker(p,-.22,.30,.61,1);p.line([(.14,.02),(-.03,.14),(-.19,.16)],.07)
        if s%2:
            p.line([(-.20,.20),(-.31,.45),(-.27,.64)],.108);p.poly([(-.32,.16),(-.20,.12),(-.17,.42),(-.36,.42)]);p.poly([(.08,-.09),(.22,-.11),(.26,.19),(-.012,.18)])
    if k==29:
        # Both rider-with-bicycle and the standalone bicycle pictogram.
        for cx in (-.34,.34):p.dot(cx,.40,.245);p.dot(cx,.40,.181,0)
        p.line([(-.33,.43),(-.13,.10),(.06,.43),(-.33,.43)],.058);p.line([(.06,.43),(.25,.07),(.32,.43)],.059)
        if s%2:
            p.dot(.075,-.24,.093);p.line([(.03,-.13),(-.15,.035),(.055,.15),(-.02,.34)],.103);p.line([(.015,-.105),(.14,.022),(.25,.012)],.065)
        else:
            p.line([(-.21,-.01),(-.07,-.01)],.065);p.line([(-.15,-.01),(.045,.40),(.23,.04),(-.10,.13)],.054);p.line([(.34,.40),(.20,-.105),(.10,-.11)],.052)
    if k==30:
        for j in range(6):
            a=j*math.pi/3+(math.pi/2 if s%2==0 else 0);u=np.cos(a);v=np.sin(a);cx=0;cy=.20
            p.line([(cx,cy),(cx+.436*u,cy+.436*v)],.064)
            for t in (.19,.302):
                for d in (-1,1):p.line([(t*u,cy+t*v),((t+.081)*u+d*.077*v,cy+(t+.081)*v-d*.077*u)],.046)
    if k==31:
        p.poly([(-.47,.05),(-.34,-.06),(-.25,-.22),(-.23,-.06),(-.10,.04),(.13,.12),(.28,.24),(.30,.39),(.47,.57),(.43,.62),(.17,.45),(.08,.32),(-.13,.24),(-.26,.16),(-.40,.13)]);p.line([(-.20,.16),(-.25,.34),(-.36,.41)],.062);p.line([(-.12,.21),(-.12,.36),(-.20,.42)],.055);p.line([(.21,.32),(.16,.47),(.30,.59)],.062)
    if k in (33,34):
        q=lambda a,b:(a*(1 if k==33 else -1),b)
        if s%2==0:
            pts=[(-.24,.68),(-.24,-.06)]+quad((-.24,-.06),(-.24,-.37),(.16,-.37))[1:]+[(.45,-.37)]
            p.line([q(*t) for t in pts],.18);p.poly([q(*t) for t in [(.20,-.71),(.59,-.37),(.20,-.04),(.27,-.32)]])
        else:
            p.line([q(-.63,.09),q(.42,.09)],.23);p.poly([q(*t) for t in [(.018,-.38),(.65,.09),(.018,.53),(.13,.14)]])
    if k in (35,36,37):
        p.poly([(-.105,.73),(.105,.73),(.105,-.19),(.37,-.13),(0,-.76),(-.37,-.13),(-.105,-.19)])
        if k!=35:
            a=1 if k==36 else -1
            p.poly([(a*b,c) for b,c in [(.0,.25),(.39,.25),(.34,.51),(.76,.17),(.34,-.18),(.39,.065),(.0,.065)]])
    if k in (38,39):
        a=1 if k==38 else -1
        # Local axes parallel and perpendicular to the diagonal shaft
        p.poly([(a*(t+v)*.7071,(t-v)*.7071) for t,v in [(-.72,-.11),(.10,-.11),(-.04,-.43),(.72,0),(-.04,.43),(.10,.11),(-.72,.11)]])
    if k==40:
        for j in range(3):
            a=j*2*math.pi/3-math.pi/2
            ang=np.linspace(a+.60,a-.57,13);p.line([(.50*math.cos(b),.50*math.sin(b)) for b in ang],.136)
            th=ang[-1];cx=.50*np.cos(th);cy=.50*np.sin(th);u=np.sin(th);v=-np.cos(th)
            p.poly([(cx+.28*u,cy+.28*v),(cx-.12*u+.20*v,cy-.12*v-.20*u),(cx-.12*u-.20*v,cy-.12*v+.20*u)])
    if k in (6,32,41,42):
        if k==6:p.a=numerals(x,y,'80',s)*.77
        d=(x+y)/1.4142; p.rule((abs(d)<(.17,.20,.15,.18)[s]) & (rim(x,y,'end',.97)),1)
    return p.a

# Network description helpers
layers=[];weights={}
def layer(name,op,ins,**kw):
    layers.append(dict(name=name,op=op,inputs=[ins] if isinstance(ins,str) else ins,**kw));return name
def affine(name,op,inp,w,b=None,stride=1,padding=0):
    w=np.asarray(w,dtype=F);weights[name+'.W']=w;weights[name+'.b']=np.zeros(len(w),F) if b is None else np.asarray(b,F)
    return layer(name,op,inp,out=len(w),**(dict(kernel=list(w.shape[-2:]),stride=stride,padding=padding) if op=='conv' else {}))
def dense(n,i,w,b=None):return affine(n,'dense',i,w,b)
def point(n,i,w,b=None):return affine(n,'conv',i,np.asarray(w)[:,:,None,None],b)
def pool(n,i,h,w,op='max'):return layer(n,op+'pool',i,size=[h,w],stride=[h,w])
def reshape(n,i,*sh):return layer(n,'reshape',i,shape=list(sh))

def clip(n,i,c,lim):
    # symmetric saturation via two shifted ReLUs
    a=point(n+'Ramp',i,np.concatenate([np.eye(c),np.eye(c)],axis=0),[lim]*c+[-lim]*c);a=layer(n+'Rect','relu',a)
    return point(n,a,np.concatenate([np.eye(c),-np.eye(c)],axis=1),[-lim]*c)

def features():
    # Suppress uneven illumination BEFORE pose matching. The spatial high pass is
    # renormalized: a weak shadowed numeral can still have well-defined strokes.
    comp=point('photometry','image',[[.25,.54,.21],[1,-.70,-.30],[-.8,-.2,1],[.42,.58,-1]])
    z=layer('exposure','norm',comp); z=clip('robust',z,4,2.7)
    g=point('gray',z,[[1,0,0,0]])
    q=np.arange(-4,5); ga=np.exp(-q*q/(2*2.25**2)); ga/=ga.sum();ker=-np.outer(ga,ga);ker[4,4]+=1
    hp=affine('detail','conv',g,ker[None,None],padding=4);hn=layer('detailGain','norm',hp);hn=clip('detailClip',hn,1,2.4)
    # Parallel 1D piecewise-linear log intensities to improve shadow invariance.
    th=np.array([0.,.016,.04,.09,.18,.36,.62,1.]); ramps=[];bs=[]
    for c in range(3):
        for t in th[:-1]:ramps.append(np.eye(3)[c]);bs.append(-t)
    a=point('tonalHinges','image',ramps,bs);a=layer('tonalRect','relu',a)
    slopes=np.diff(np.log(th+.024))/np.diff(th);dl=np.diff(np.r_[0,slopes]);mix=np.array([np.kron(row,dl) for row in [[.25,.54,.21],[1,-.7,-.3],[-.8,-.2,1],[.42,.58,-1]]]);a=point('logOpponents',a,mix);a=layer('logGain','norm',a);a=clip('logClip',a,4,2.7)
    h=point('logGray',a,[[1,0,0,0]]);h=affine('logDetail','conv',h,ker[None,None],padding=4);h=layer('logDetailGain','norm',h);h=clip('logDetailClip',h,1,2.4)
    # A 5-plane stack: gray, red, blue, yellow and centre-surround gray.
    def embed(n,i,w):return point(n,i,w)
    m=np.zeros((5,4));m[:4,:]=np.eye(4);mh=np.array([[0,0,0,0,1.]]).T
    return layer('planes','add',[embed('linBase',z,m*.53),embed('logBase',a,m*.47),embed('linEdge',hn,mh*.56),embed('logEdge',h,mh*.44)])

N=28
SIZES=[(8.5,8.5),(10,10),(11.5,11.5),(13,13),(14.5,14.5),(15.6,15.6),(10.5,12),(12,10.5),(12.5,14),(14,12.5)]

# Rasterization for an individual template: anti-aliased area integrals and
# independent contrast normalization. No drawing is compared with any other.
def grid(side=24,sub=3):
    t=((np.arange(side*sub)+.5)/(side*sub)*2-1)*1.08
    return np.meshgrid(t,t)
def down(v):return v.reshape(N,3,N,3).mean((1,3))
X,Y=grid(N)
def template(z,m):
    a=down(np.asarray(z,float)*m);w=down(m.astype(float)); ss=w.sum()
    if ss<1e-7:return np.zeros_like(a)
    a-=w*(a.sum()/ss);l=np.sqrt(np.sum(a*a/(w+1e-9))*ss)
    return a/(l+1e-12)

def frame(f):
    x,y=X,Y; inside=rim(x,y,f);m=np.ones_like(x,bool);a=np.zeros((5,N,N));
    if f in ('red','warn','yield'):
        inner=rim(x,y,f,.775 if f=='red' else .735);ring=inside&~inner
        # A margin outside the shape provides a reference but background far
        # from the sign carries little evidence. Use the shape's own support.
        m=rim(x,y,f,1.145);g=.20+.80*inner+.06*(~inside);r=ring.astype(float)
        a[0]=template(g,m)*.52;a[1]=template(r,m)*.51;a[4]=template(g,m)*.23
    elif f=='diamond':
        yellow=rim(x,y,f,.71);white=inside&~yellow;g=.22+.77*white+.50*yellow;a[0]=template(g,m)*.42;a[3]=template(yellow,m)*.68;a[4]=template(g,m)*.18
    elif f in ('blue','stop','entry'):
        inner=rim(x,y,f,.94);g=.39-.21*inner+.53*(inside&~inner);c=inside.astype(float);m=rim(x,y,f,1.23)
        if f=='stop':g=np.maximum(g,.86*symbol(14,x,y,0))
        if f=='entry':g=np.maximum(g,.94*symbol(17,x,y,0));c*=1-symbol(17,x,y,0)
        a[0]=template(g,m)*(.17 if f=='blue' else .37);a[2 if f=='blue' else 1]=template(c,m)*(.91 if f=='blue' else .76);a[4]=template(g,m)*.13
    else:
        c=inside.astype(float);a[0]=template(c,m)*.62;a[4]=template(c,m)*.22
        # Cancellation signs are achromatic; the bands are dark, not red/blue.
        a[1]=template(inside&~rim(x,y,f,.77),rim(x,y,f,1.145))*(-.18)
    return a

def glyph(k,s,orientation=False):
    f=family(k);x,y=X,Y
    # Interior-only match. Match non-symbol pixels too, so e.g. an extra
    # arrow branch counts against a straight-ahead arrow.
    zx=(1.,.96,1.075,1.0)[s%4];zy=(1.,1.055,.96,.94)[s%4];ang=(0.,.025,-.035,.0)[s%4];yc=(0,.032,-.018,-.03)[s%4]
    if f not in ('warn','blue','end') and k not in (9,10,16):zx=zy=1.;ang=yc=0.
    xx=(x+ang*y)/zx;yy=(y-yc-ang*x)/zy
    if f=='warn':
        # Library of typographic sizes, placements and small tilts within
        # the triangular field; each pose is a separate, complete symbol.
        scx,scy,dv,tilt=[(1,1,0,0),(.84,.81,.04,0),(1.15,1.13,-.04,0),(.94,1,.13,0),(1,.94,-.13,0),(1.15,.86,0,.075),(.88,1.13,0,-.075),(1.02,1.04,.04,.10)][s%8]
        xx=(x-tilt*(y-.2))/scx;yy=(y-.20-dv+tilt*x)/scy+.20
    if k in (0,1,2,3,4,5,7,8) and s>=4:
        xx=x/(1.09 if s%2==0 else .92);yy=y/(1.12 if s%3 else .90)
    v=symbol(k,xx,yy,s);a=np.zeros((5,N,N))
    if f=='warn':m=rim(x,y,f,.723); col=-v
    elif f=='blue':m=rim(x,y,f,.86);col=v
    elif f=='end':m=rim(x,y,f,.91);col=-v
    else:m=rim(x,y,f,.745);col=-v
    if k in (14,17):col=v
    if k in (12,13,15):return a
    if k in (0,1,2,3,4,5,7,8):
        # Within a speed-limit sign most of the white disc need not be used
        # a second time; the zero at right should not drown the telling digit.
        win=m&(abs(y)<.53)&(abs(x)<(.75 if k in (7,8) else .65))
        m=win.astype(float)*np.where(x<(.20 if k in (7,8) else -.015),1.0,.56)
    if orientation=='local':
        mm=down(m.astype(float));raw=down(col*m);raw/=np.maximum(mm,.00001);weights=[]
        for cy in (-.04,.44):
            for cx in (-.24,.24):
                # Match separate sectors of the icon. Each sector is an
                # independent template window, never a cross-drawing mean.
                zone=mm*down(np.exp(-.5*((x-cx)/.29)**2-.5*((y-cy)/.32)**2));mass=zone.sum();dc=(raw*zone).sum()/mass;w=(raw-dc)*zone
                weights.append(w/np.sqrt((w*w/(zone+1e-8)).sum()*mass+1e-7))
        return weights
    if orientation:
        # An unsigned oriented-edge descriptor of this drawing alone. Pool
        # votes over a small area, so exact stroke boundaries need not align.
        v=down(col);gy,gx=np.gradient(v);planes=[]
        for theta in np.arange(4)*np.pi/4:
            du=gx*np.cos(theta)+gy*np.sin(theta);dv=gy*np.cos(theta)-gx*np.sin(theta)
            z=np.maximum(abs(du)-.50*abs(dv),0)
            g=np.array([1,4,6,4,1])/16
            for axis in (0,1):z=np.apply_along_axis(lambda l:np.convolve(l,g,'same'),axis,z)
            planes.append(z)
        z=np.array(planes);mm=down(m.astype(float));z-=((z*mm).sum((1,2))/mm.sum())[:,None,None];z*=mm
        return z/np.sqrt((z*z/(mm+1e-8)).sum()*mm.sum()+1e-10)
    # Split the readout into a broad silhouette match and a bandpass stroke
    # match. Both filters come from this one glyph. For the second, ignore
    # the surrounding rim and look for nearby pairs of opposite contrast.
    broad=template(col,m);a[0]=broad*.46
    q=np.arange(-4,5);g=np.exp(-q*q/(2*1.28**2));g/=g.sum()
    blur=np.apply_along_axis(lambda z:np.convolve(z,g,'same'),0,broad)
    blur=np.apply_along_axis(lambda z:np.convolve(z,g,'same'),1,blur)
    mm=down(m.astype(float));hh=(broad-blur)*mm;hh-=mm*hh.sum()/mm.sum();hn=np.sqrt((hh**2/(mm+1e-9)).sum()*mm.sum())
    a[4]=hh/(hn+1e-12)*.54
    if k in (9,10):a[1]=template(v*(xx<0),m)*.19
    if k==26:
        a[1]=template(((xx/.13)**2+((yy+.21)/.13)**2<1).astype(float),m)*.12
    if f=='blue':a[2]=template(-v,m)*.11
    return a

# Compose fixed bilinear scale changes into each filter separately. Translation
# sharing then comes for free from convolution over the 5-by-5 search grid.

def interpolation(q,n=32):
    q=np.clip(q,0,n-1.);j=np.floor(q).astype(int);d=q-j
    a=np.zeros((len(q),n),F);a[np.arange(len(q)),j]=1-d;a[np.arange(len(q)),np.minimum(n-1,j+1)]+=d;return a

def make_network():
    inp=features()
    FAM=8;S=8;K=43
    fams=['red','warn','yield','diamond','stop','entry','blue','end']
    # Additional channels used for color and an L1 estimate of local detail.
    a=point('polarities',inp,[[0,0,0,0,1],[0,0,0,0,-1]]);a=layer('halfWaves','relu',a);a=point('energy',a,[[1,1]])
    co=affine('colourAbsolute','conv','tonalRect',weights['logOpponents.W'][[1,2,3]])
    co=clip('colourAmount',co,3,1.4)
    # Common frame/energy and actual colour correlation, one per family.
    framebank=np.asarray([frame(f) for f in fams]); iconbank=np.asarray([glyph(k,s) for k in range(K) for s in range(S)])
    W=[];CW=[]
    for k in range(K):
        f=family(k);m=rim(X,Y,f,.67 if f!='blue' else .83);e=down(m.astype(float));e/=e.sum()
        W.append(e);mc=np.zeros((3,N,N));ring=down((rim(X,Y,f,.98)&~rim(X,Y,f,.77)).astype(float));ring/=ring.sum();inner=down(rim(X,Y,f,.86).astype(float));inner/=inner.sum()
        if f in ('red','warn','yield'):mc[0]=.30*ring
        if f=='blue':mc[1]=.52*inner
        if f in ('entry','stop'):mc[0]=.51*inner
        if f=='diamond':mc[2]=.48*inner
        if f=='end':mc[0]=-.14*ring;mc[1]=-.06*inner
        CW.append(mc)
    stride=1;padding=2;G=5;L=G*G
    # Compute the five by five position grid using one im2col per bank. Split
    # grayscale symbol responses from colour frames to avoid redundant flops.
    base=point('ink',inp,[[.68,0,0,0,.32],[.15,0,0,0,.85]])
    # Additional localized shape probes: test that both ends of the
    # cancellation stripe cross the disc edge, outside the text region.
    d=(X+Y)/np.sqrt(2);ax=(X-Y)/np.sqrt(2);slash=(abs(d)<.17)&(abs(ax)>.60)&(abs(ax)<.93);ms=(abs(ax)>.60)&(abs(ax)<.93)&(abs(d)<.38)&rim(X,Y,'end',.96)
    cancel=template(-slash.astype(float),ms)
    # Check stripe endpoints in both luminance and high-pass channels.
    cancelbank=np.stack([cancel*.56,cancel*.44])[None]
    # Read under the diagonal independently of detecting it. For the blank
    # cancellation look for lack of high-frequency contrast off the stripe.
    endids=[6,32,41,42];endbank=[];em=(abs(X+Y)/np.sqrt(2)>.32)&rim(X,Y,'end',.78)&(abs(Y)<.57)
    for k in endids:
        for st in range(4):
            if k==6:v=numerals(X,Y,'80',st)
            elif k in (41,42):v=symbol(9 if k==41 else 10,X,Y,st)
            else:v=np.zeros_like(X)
            endbank.append(template(-v,em))
    endbank=np.asarray(endbank);ee=down(em.astype(float));ee/=ee.sum()
    # Cost of ink in the central field (not the local high-pass response of
    # the red ring). A genuinely blank disc should win if its interior has
    # negligible fine contrast.
    cm=(X*X+Y*Y<.50**2);bm=down(cm.astype(float));bm/=bm.sum()
    clrids=[9,10,26,33,34,35,36,37,38,39,40]
    csbank=np.stack([glyph(k,0)[[1,2]] for k in clrids]);cplanes=point('colouredInk',inp,[[0,1,0,0,0],[0,0,1,0,0]])
    # The input counterpart uses analytic Sobel filters and rectifiers. It is
    # evaluated on the input only at inference. Four unsigned bins distinguish
    # e.g. the two straight road edges from a bend, or wheels from a mound.
    orientids=[0,1,2,3,4,5,7,8,11]+list(range(18,32))+list(range(33,41))
    obank=np.asarray([glyph(k,st,True) for k in orientids for st in (0,2,5,6)])
    gx=np.outer([1,2,1],[-1,0,1])/8;gy=gx.T;dw=np.array([np.cos(t)*gx+np.sin(t)*gy for t in np.arange(4)*np.pi/4]);gray=point('contourSource',inp,[[.75,0,0,0,.25]]);der=affine('derivatives','conv',gray,np.concatenate([dw,-dw])[:,None],padding=1);der=layer('slopeSign','relu',der);der=point('unsignedSlope',der,np.concatenate([np.eye(4),np.eye(4)],1));ob=point('orientationBins',der,np.eye(4)-.5*np.roll(np.eye(4),2,axis=1));ob=layer('positiveVotes','relu',ob);ob=layer('edgeGain','norm',ob);ob=clip('softOrientation',ob,4,2.5)
    # A dedicated OCR reader: orthographic counters/loops are finer than the
    # coarse background correlation. Laplacian-of-Gaussian filtering removes
    # the large white field and preserves both sides of narrow black stems.
    ocrids=[0,1,2,3,4,5,7,8];oc=[]
    for k in ocrids:
        text={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}[k]
        for std in range(4):
            for stretch in (.90,1.,1.10):
                for dx,dy in [(0,0),(.5,0),(0,.5),(.5,.5)]:
                    # Fixed half-pixel phases in sign coordinates, prescribed
                    # for a canonical diameter of 24 pixels. Their response
                    # maximum is computed only on the incoming photograph.
                    v=-numerals((X-dx/12)/stretch,Y-dy/12,text,std);mask=(abs(Y)<.51)&(abs(X)<(.73 if k>=7 else .63));mask=mask*np.where(X<(.20 if k>=7 else 0),1.,.56)
                    oc.append(template(v,mask))
    oc=np.asarray(oc);t=np.arange(-2,3);ga=np.exp(-t*t/(2*1.04**2));ga/=ga.sum();lap=-np.outer(ga,ga);lap[2,2]+=1
    sharp=affine('fineStrokes','conv',gray,lap[None,None],padding=2);sharp=layer('fineGain','norm',sharp);sharp=clip('fineClip',sharp,1,2.6)
    warnids=[11]+list(range(18,32));localbank=np.array([glyph(k,st,'local') for k in warnids for st in (0,1,2,6)]).reshape(-1,N,N);localbase=point('sectorInk',inp,[[.55,0,0,0,.45]])
    cls_scales=[]; fam_scales=[]
    for si,(rx,ry) in enumerate(SIZES):
        suf=str(si);t=((np.arange(N)+.5)/N*2-1)*1.08;wx=interpolation(15.5+rx*t);wy=interpolation(15.5+ry*t)
        cv=lambda nm,v,w:affine(nm+suf,'conv',v,w,stride=stride,padding=padding)
        frame_s=cv('frame',inp,(wy.T@framebank)@wx)
        col_s=cv('colour',co,(wy.T@np.asarray(CW))@wx);energy=cv('texture',a,((wy.T@np.asarray(W))@wx)[:,None]);
        sym_s=cv('symbol',base,(wy.T@iconbank[:,[0,4]])@wx);sym_s=reshape('styleGrid'+suf,sym_s,K,S,L);sym_s=pool('bestStyle'+suf,sym_s,S,1);sym_s=reshape('styles'+suf,sym_s,K,G,G)
        colink=cv('colourSymbol',cplanes,(wy.T@csbank)@wx);colink=point('whichColour'+suf,colink,np.eye(K)[:,clrids]);sym_s=layer('colourAndInk'+suf,'add',[sym_s,colink]);cancel_s=cv('stripeEnds',base,(wy.T@cancelbank)@wx);cancel_s=point('whichEnds'+suf,cancel_s,[[.55 if family(k)=='end' else 0] for k in range(K)])
        ev=cv('cancelledEmblem',base,(wy.T@np.stack([endbank*.65,endbank*.35],1))@wx);ev=reshape('emblemStyleAxis'+suf,ev,4,4,L);ev=pool('emblemStyle'+suf,ev,4,1);ev=reshape('emblemGrid'+suf,ev,4,G,G);ev=point('emblemVote'+suf,ev,.77*np.eye(K)[:,endids]);
        ce=cv('fieldEnergy',a,((wy.T@np.stack([ee,bm]))@wx)[:,None]);ww=np.zeros((K,2));bb=np.zeros(K);ww[32,0]=-.68;bb[32]=.48;ww[15,1]=-.85;bb[15]=.42;ce=point('emptyField'+suf,ce,ww,bb)
        # Match the independently normalized unsigned-orientation templates.
        ow=(wy.T@obank)@wx;ori=cv('edgeTemplate',ob,ow);ori=reshape('edgeAxis'+suf,ori,len(orientids),4,L);ori=pool('edgeStyle'+suf,ori,4,1);ori=reshape('edgeGrid'+suf,ori,len(orientids),G,G);ori=point('edgeVote'+suf,ori,.23*np.eye(K)[:,orientids]);
        # Fine OCR filters have no support near the rim, so evaluate with a
        # smaller valid convolution, with the same output centre positions.
        ow=(wy.T@oc)@wx;ow=ow[:,2:-2,2:-2];rd=affine('readDigits'+suf,'conv',sharp,ow[:,None],stride=1);rd=reshape('digitsAxis'+suf,rd,len(ocrids),48,L);rd=pool('digitsMatch'+suf,rd,48,1);rd=reshape('digitsGrid'+suf,rd,len(ocrids),G,G);
        # Contrastive OCR at inference only: evidence shared by every digit
        # reader is not evidence for a speed limit over a truck pictogram.
        rd=point('digitsVote'+suf,rd,.43*np.eye(K)[:,ocrids]@(np.eye(len(ocrids))-np.ones((len(ocrids),len(ocrids)))/len(ocrids)))
        # Soft part alignment and an AND-like demand that several distinct
        # sectors agree. Upper-limit individual part votes so a single stroke
        # cannot entirely dominate the decision.
        lw=((wy.T@localbank)@wx)[:,2:-2,2:-2];lo=affine('partLookup'+suf,'conv',localbase,lw[:,None]);lo=reshape('partChannels'+suf,lo,1,len(warnids)*16,L);lp=point('partHinge'+suf,lo,[[1]],[ -.70]);lp=layer('partSaturate'+suf,'relu',lp);lp=point('negateExcess'+suf,lp,[[-1]]);lo=layer('partVotes'+suf,'add',[lo,lp]);lo=reshape('partAxis'+suf,lo,len(warnids)*4,4,L);lo=pool('partAgree'+suf,lo,4,1,'avg');lo=reshape('partStyles'+suf,lo,len(warnids),4,L);lo=pool('partStyleVote'+suf,lo,4,1);lo=reshape('partGrid'+suf,lo,len(warnids),G,G);lo=point('distributedInk'+suf,lo,.72*np.eye(K)[:,warnids]);
        # Mapping of common outer shape responses to per-class hypotheses.
        mp=np.eye(FAM)[[fams.index(family(k)) for k in range(K)]];fr=point('shapeMap'+suf,frame_s,mp);fr=layer('shapeColour'+suf,'add',[fr,col_s]);
        # Discount weak pose estimates. The interior can sit off-centre inside
        # the frame, but the frame-grid match still limits the displacement.
        sy=affine('padInk'+suf,'conv',sym_s,np.eye(K)[:,:,None,None],padding=1);sy=layer('inkNearby'+suf,'maxpool',sy,size=[3,3],stride=[1,1]);flex=point('rigidFraction'+suf,sym_s,np.eye(K)*.45);sy=point('flexFraction'+suf,sy,np.eye(K)*.55);sy=layer('inkMatch'+suf,'add',[flex,sy]);
        # Evidence of actual interior strokes, distinct from the coloured rim.
        # If inside is almost featureless, all nonblank correlations are near
        # zero. A blank symbol gets an explicit likelihood in that case.
        diag=np.array([-.34 if k not in (12,13,15) else (-1.0 if k!=12 else -.12) for k in range(K)]);en=point('textureCost'+suf,energy,np.diag(diag));
        bias=np.array([.38 if k in (13,15) else (.09 if k==12 else 0) for k in range(K)]);sy=point('symbolVote'+suf,sy,np.diag([.70 if k in (14,17) else .91 for k in range(K)]),bias)
        r=layer('joint'+suf,'add',[sy,fr,en,cancel_s,ev,ce,ori,rd,lo]);cls_scales.append(pool('poseMax'+suf,r,G,G))
        fam_scales.append(pool('frameMax'+suf,fr,G,G))
    def maxscales(n,rs):
        U=len(rs);ss=[]
        for i,r in enumerate(rs):
            w=np.zeros((K,U,K),F);w[:,i]=np.eye(K);ss.append(dense(n+'place'+str(i),r,w.reshape(-1,K)))
        r=layer(n+'collect','add',ss);r=reshape(n+'axis',r,K,1,U);return pool(n+'max',r,1,U)
    joint=maxscales('joint',cls_scales);frameout=maxscales('shape',fam_scales);frameout=point('shapeConfidence',frameout,np.eye(K)*.25);r=layer('classEvidence','add',[joint,frameout]);reshape('scores',r,K)

if __name__=='__main__':
    target=Path(sys.argv[1]);target.mkdir(exist_ok=True,parents=True);make_network();(target/'network.json').write_text(json.dumps({'layers':layers},indent=1));np.savez(target/'weights.npz',**weights)
