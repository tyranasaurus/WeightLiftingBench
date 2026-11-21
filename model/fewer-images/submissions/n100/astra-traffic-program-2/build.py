"""Data-free construction of hand-drawn traffic sign templates and recognizer."""
import math
from pathlib import Path
import sys
import numpy as np

# Draw pictograms with dimensionless vector primitives on a supersampled canvas.
ns = 192
axis = (np.arange(ns)+.5)*2/ns-1
X,Y=np.meshgrid(axis,axis)
class Pen:
    def __init__(self, x=X, y=Y):
        self.x,self.y=x,y
        self.a=np.zeros(x.shape)
    def ellipse(self,cx,cy,rx,ry,col=1):
        self.a[((self.x-cx)/rx)**2+((self.y-cy)/ry)**2<=1]=col
    def poly(self,pts,col=1):
        m=np.zeros(self.x.shape,bool)
        for (a,b),(c,d) in zip(pts,pts[1:]+pts[:1]):
            if d!=b:
                m ^= ((b>self.y)!=(d>self.y)) & (self.x < (c-a)*(self.y-b)/(d-b)+a)
        self.a[m]=col
    def box(self,x0,y0,x1,y1,col=1):
        self.poly([(x0,y0),(x0,y1),(x1,y1),(x1,y0)],col)
    def line(self,pts,w,col=1):
        asp=getattr(self,'aspect',1.)
        for (a,b),(c,d) in zip(pts,pts[1:]):
            dx,dy=c-a,d-b
            t=np.clip(((self.x-a)*dx*asp**2+(self.y-b)*dy)/(dx*dx*asp**2+dy*dy+1e-12),0,1)
            self.a[((self.x-a-dx*t)*asp)**2+(self.y-b-dy*t)**2<=(w/2)**2]=col
    def curve(self,pts,w,col=1):
        # Cubic Beziers, optionally chained end-to-end
        path=[pts[0]]
        for i in range(0,len(pts)-1,3):
            a,b,c,d=np.asarray(pts[i:i+4]);
            for t in np.linspace(0,1,13)[1:]:
                s=1-t; path.append(tuple(a*s**3+3*b*t*s*s+3*c*s*t*t+d*t**3))
        self.line(path,w,col)

def digit(p,ch,style):
    # Monoline DIN-like figures. The pen coordinates of each figure run from 0 to 1.
    p.aspect=(.28 if ch=='1' else .52)
    w=.102 if style==0 else .084
    if ch=='0':
        p.curve([(.5,.07),(.83,.07),(.9,.19),(.9,.4),(.91,.61),(.96,.93),(.5,.93),(.08,.93),(.085,.75),(.085,.53),(.085,.30),(.035,.07),(.5,.07)],w)
    if ch=='1':
        p.line([(.1,.23),(.5,.065),(.5,.925)],w)
        if style: p.line([(.19,.925),(.80,.925)],w*.8)
    if ch=='2':
        p.curve([(.09,.27),(.1,.035),(.85,-.03),(.89,.265),(.92,.405),(.70,.51),(.53,.60),(.39,.69),(.19,.80),(.105,.925)],w)
        p.line([(.10,.925),(.92,.925)],w)
    if ch=='3':
        p.curve([(.10,.225),(.25,.015),(.88,.02),(.88,.27),(.88,.422),(.70,.475),(.43,.478)],w)
        p.curve([(.43,.478),(.79,.478),(.94,.568),(.93,.71),(.94,.965),(.30,1.02),(.08,.795)],w)
    if ch=='5':
        p.line([(.88,.082),(.17,.082),(.125,.488)],w)
        p.curve([(.125,.488),(.38,.343),(.92,.36),(.92,.62),(.97,.985),(.32,1.025),(.085,.802)],w)
    if ch=='6':
        if style==0:
            p.curve([(.68,.06),(.53,.21),(.17,.50),(.12,.64)],w)
        else:
            p.curve([(.86,.15),(.46,-.07),(.07,.16),(.10,.68)],w)
        p.curve([(.10,.67),(.10,.40),(.89,.353),(.90,.673),(.92,1.017),(.07,1.019),(.10,.67)],w)
    if ch=='7':
        p.line([(.06,.084),(.915,.084),(.32,.935)],w)
        if style: p.line([(.06,.081),(.06,.22)],w*.8)
    if ch=='8':
        if style==0:
            p.ellipse(.5,.263,.471,.251);p.ellipse(.5,.263,.258,.144,0)
            p.ellipse(.5,.711,.497,.28);p.ellipse(.5,.711,.280,.165,0)
        else:
            p.curve([(.5,.48),(.02,.46),(-.03,.07),(.49,.07),(1.02,.07),(.98,.46),(.5,.48)],w)
            p.curve([(.5,.48),(-.03,.49),(-.08,.93),(.49,.93),(1.06,.93),(1.04,.51),(.5,.48)],w)

def numerals(s,style):
    h=.84 if len(s)==2 else .78
    w=h*(.51 if style==0 else .54)
    gap=.105 if len(s)==2 else .071
    widths=[w*(.55 if c=='1' else 1) for c in s]
    pos=-(sum(widths)+gap*(len(s)-1))/2
    out=0*X
    for c,wd in zip(s,widths):
        p=Pen((X-pos)/wd,(Y+h/2)/h)
        digit(p,c,style)
        out=np.maximum(out,p.a); pos+=wd+gap
    return out

def car(p,x,y,w,h,col=1):
    q=Pen((p.x-x)/w,(p.y-y)/h); q.a=p.a
    q.poly([(-.48,.35),(-.5,.02),(-.38,-.14),(-.27,-.45),(-.17,-.5),(.22,-.5),(.29,-.45),(.40,-.13),(.5,.0),(.48,.35)],col)
    q.poly([(-.295,-.13),(-.22,-.39),(.20,-.39),(.30,-.13)],0)
    for s in [-1,1]: q.box(s*.36-.07,.25,s*.36+.07,.50,col)

def truck(p,x,y,w,h,col=1,mirror=1):
    q=Pen((p.x-x)*mirror/w,(p.y-y)/h); q.a=p.a
    q.box(-.12,-.49,.50,.215,col)
    q.poly([(-.49,.25),(-.49,-.04),(-.33,-.30),(-.16,-.30),(-.16,.15),(.49,.15),(.49,.28),(-.49,.28)],col)
    q.poly([(-.40,-.04),(-.30,-.225),(-.224,-.225),(-.224,-.04)],0)
    for z in [-.32,.31]: q.ellipse(z,.29,.098,.205,col)

def vehicle(cl,sty):
    p=Pen()
    if cl in (9,41):
        car(p,-.30,.045,.55,.38, .67 if cl==9 else 1); car(p,.30,-.035,.55,.38)
    if cl in (10,42):
        truck(p,-.28,-.05,.63,.43, .70 if cl==10 else 1); car(p,.36,.10,.46,.33)
        if sty:
            p=Pen();truck(p,-.24,-.145,.75,.41,.73 if cl==10 else 1);car(p,.22,.15,.53,.35)
    if cl==16: truck(p,0,0,1.22,.54)
    return p.a

def arrow(p,x,y,dx,dy,w,l,head,wing):
    d=math.hypot(dx,dy); u,v=dx/d,dy/d
    pts=[(-l,-w/2),(-head,-w/2),(-head,-wing),(0,0),(-head,wing),(-head,w/2),(-l,w/2)]
    p.poly([(x+a*u-b*v,y+a*v+b*u) for a,b in pts])

def blue(cl,sty):
    s=-1 if cl in (34,37,39) else 1
    p=Pen(X*s,Y)
    if cl in (33,34):
        p.curve([(-.28,.61),(-.28,.41),(-.28,.20),(-.28,.03),(-.27,-.26),(-.08,-.36),(.23,-.36)],.18)
        arrow(p,.62,-.36,1,0,.18,.48,.34,.30)
    if cl==35: arrow(p,0,-.73,0,-1,.20,1.42,.47,.37)
    if cl in (36,37):
        arrow(p,-.22,-.73,0,-1,.17,1.43,.39,.27)
        if sty: p.curve([(-.22,.42),(-.22,.17),(-.01,.14),(.25,.14)],.16)
        else: p.line([(-.21,.14),(.28,.14)],.17)
        arrow(p,.72,.14,1,0,.18,.52,.38,.28)
    if cl in (38,39):
        if sty==0: arrow(p,.51,.51,1,1,.19,1.45,.50,.36)
        else:
            p.poly([(-.57,-.42),(-.42,-.57),(.35,.20),(.35,-.12),(.53,-.12),(.53,.53),(-.12,.53),(-.12,.35),(.19,.35)])
    if cl==40:
        shift=0 if sty==0 else 60
        for ang in [0,120,240]:
            a=math.radians(ang+shift); h=Pen(X*math.cos(a)+Y*math.sin(a),-X*math.sin(a)+Y*math.cos(a));h.a=p.a
            h.curve([(.59,.17),(.71,-.21),(.38,-.61),(.055,-.61)],.125)
            arrow(h,-.215,-.61,-1,0,.14,.32,.28,.19)
    return p.a

def person(p,x,y,s,facing=1,skirt=False,lean=.08):
    q=Pen((p.x-x)*facing/s,(p.y-y)/s);q.a=p.a
    q.ellipse(lean+.015,-.44,.070,.081)
    q.poly([(lean-.055,-.347),(lean+.067,-.34),(.07,-.04),(.02,.065),(-.095,.04),(-.065,-.18)])
    q.line([(lean+.03,-.28),(.16,-.125),(.30,-.19)],.053)
    q.line([(lean-.01,-.29),(-.15,-.16),(-.22,-.02)],.054)
    if skirt: q.poly([(-.046,-.19),(.064,-.17),(.145,.10),(-.14,.10)])
    q.line([(.03,.005),(.16,.20),(.25,.40)],.084)
    q.line([(-.045,.012),(-.11,.24),(-.28,.40)],.080)
    q.line([(.24,.40),(.31,.40)],.057);q.line([(-.28,.40),(-.22,.42)],.055)

def warning(cl,sty):
    p=Pen()
    if cl==11:
        p.poly([(-.09,.65),(.095,.65),(.095,.41),(.34,.47),(.34,.385),(.102,.24),(.102,.105),(.197,.105),(0,-.37),(-.20,.105),(-.10,.105),(-.10,.24),(-.34,.385),(-.34,.47),(-.09,.41)])
        if sty: p.box(-.10,-.08,.1,.65)
    if cl==18:
        p.poly([(-.087,-.36),(.087,-.36),(.045,.385),(-.045,.385)]);p.ellipse(0,.54,.087,.096)
    if cl in (19,20):
        p=Pen(X*(1 if cl==19 else -1),Y)
        p.curve([(.23,.585),(.23,.48),(.23,.36),(.22,.25),(.21,.065),(.09,-.012),(-.19,-.012)],.17)
        p.box(.145,.49,.315,.64);p.box(-.23,-.097,-.15,.073)
    if cl==21:
        p=Pen(X,Y)
        if sty==0:p.curve([(.19,.61),(.19,.50),(.19,.45),(.19,.38),(.19,.18),(-.16,.40),(-.16,.19),(-.16,.07),(-.16,-.07),(-.16,-.17)],.145)
        else:p.line([(.19,.60),(.19,.30),(.14,.24),(-.12,.24),(-.16,.19),(-.16,-.19)],.157)
    if cl==22:
        p.poly([(-.53,.6),(-.50,.55),(-.41,.53),(-.34,.405),(-.28,.363),(-.22,.362),(-.16,.398),(-.085,.47),(0,.48),(.075,.452),(.15,.373),(.23,.354),(.32,.39),(.43,.53),(.52,.54),(.55,.61)])
    if cl==23:
        car(p,-.045,.10,.56,.44)
        for k in [-.20,.16]:p.curve([(k,.28),(k-.15,.40),(k-.28,.405),(k-.13,.49),(k+.03,.54),(k-.12,.59),(k-.16,.65)],.052)
    if cl==24:
        p.box(-.205,-.14,-.105,.63)
        p.line([(.225,.60),(.225,.39),(.10,.15),(.095,-.10)],.11)
    if cl==25:
        q=Pen();q.ellipse(.04,-.17,.088,.092)
        q.poly([(-.019,-.097),(.096,-.066),(.062,.035),(-.152,.188),(-.205,.30),(-.31,.259),(-.26,.07)])
        q.line([(-.23,.22),(-.30,.38),(-.42,.59)],.09);q.line([(-.20,.215),(-.106,.38),(-.095,.60)],.089)
        q.line([(-.05,.0),(.016,.174),(.142,.19)],.058)
        q.line([(-.30,.104),(.18,.355),(.34,.51)],.037)
        q.poly([(.075,.61),(.14,.51),(.21,.486),(.295,.393),(.35,.395),(.46,.514),(.56,.60)])
        p=q
    if cl==26:
        p.box(-.14,-.29,.14,.59);p.ellipse(0,-.28,.14,.11);p.ellipse(0,.58,.14,.10)
        for y,c in [(-.197,.66),(.146,.27),(.477,.64)]:p.ellipse(0,y,.106,.114,c if sty==0 else .33)
    if cl==27:person(p,0,.20,1.04,-1 if sty else 1)
    if cl==28:
        person(p,.125,.17,1.00, -1,True); person(p,-.245,.33,.65,-1)
    if cl==29:
        for x in [-.292,.301]:p.ellipse(x,.415,.193,.193);p.ellipse(x,.415,.136,.136,0)
        p.line([(-.29,.415),(-.155,.14),(.02,.415),(.20,.163),(-.155,.14)],.041)
        p.line([(.30,.415),(.14,.058),(.21,.025)],.043)
        p.ellipse(.095,-.125,.088,.091)
        p.line([(.020,-.038),(-.145,.111),(.0,.205),(-.08,.378)],.087)
        p.line([(.02,-.036),(.09,.075),(.192,.09)],.063)
        if sty:p.a=p.a[:,::-1].copy()
    if cl==30:
        for k in range(6):
            a=math.radians(k*60-90);u,v=math.cos(a),math.sin(a);xc,yc=0,.23
            p.line([(xc,yc),(xc+.426*u,yc+.426*v)],.054)
            for sign in [-1,1]:
                p.line([(xc+.21*u,yc+.21*v),(xc+.31*u+sign*.112*v,yc+.31*v-sign*.112*u)],.042)
                if sty==0:p.line([(xc+.31*u,yc+.31*v),(xc+.386*u+sign*.080*v,yc+.386*v-sign*.080*u)],.039)
    if cl==31:
        p=Pen(X*(1 if sty==0 else -1),Y)
        p.poly([(-.40,.20),(-.31,.14),(-.25,.18),(.11,.23),(.23,.27),(.32,.32),(.30,.41),(.13,.405),(-.12,.31),(-.20,.32),(-.315,.40),(-.34,.34),(-.29,.246),(-.40,.25)])
        p.line([(-.13,.29),(-.21,.40),(-.39,.49)],.060);p.line([(-.14,.30),(-.247,.34),(-.42,.36)],.047)
        p.line([(.24,.34),(.35,.45),(.49,.50)],.065);p.line([(.20,.35),(.22,.47),(.41,.55)],.045)
        p.line([(-.29,.21),(-.30,.07),(-.36,.005)],.032);p.line([(-.30,.13),(-.22,.06),(-.23,.005)],.029)
    # Additional conventional drawings: signal lenses without a housing, running
    # children holding hands, a bare bicycle, and the arched body of a leaping deer.
    if cl==26:
        p=Pen()
        for y,c in [(-.145,.76),(.17,.46),(.485,.93)]:p.ellipse(0,y,.138,.143,c)
    if cl==28:
        p=Pen();q=Pen(X*(1 if sty==0 else -1),Y);p=q
        q.ellipse(.09,-.135,.082,.089);q.ellipse(-.245,.115,.068,.075)
        q.poly([(.04,-.055),(.16,-.050),(.20,.16),(.12,.26),(.0,.24),(.065,.10)])
        q.line([(.085,.004),(-.052,.173),(-.173,.211)],.070);q.line([(.142,.0),(.256,.145),(.321,.22)],.059)
        q.line([(.081,.195),(.157,.333),(.347,.502)],.095);q.line([(.095,.203),(.04,.366),(-.055,.566)],.089)
        q.poly([(-.276,.18),(-.189,.173),(-.177,.327),(-.109,.403),(-.316,.409),(-.267,.274)])
        q.line([(-.213,.216),(-.130,.143),(-.05,.155)],.045);q.line([(-.258,.215),(-.341,.296),(-.386,.322)],.048)
        q.line([(-.239,.371),(-.292,.486),(-.365,.563)],.063);q.line([(-.196,.367),(-.192,.468),(-.117,.555)],.065)
    if cl==29 and sty==0:
        p=Pen(X,Y)
        for x in [-.295,.297]:p.ellipse(x,.39,.202,.215);p.ellipse(x,.39,.147,.158,0)
        p.line([(-.295,.39),(-.139,.085),(.0,.394),(.167,.095),(-.139,.085)],.056)
        p.line([(-.295,.39),(.00,.394)],.049);p.line([(.297,.39),(.122,-.006),(.207,-.043)],.059)
        p.line([(-.188,.014),(-.069,.014)],.063);p.line([(-.149,.015),(-.139,.085)],.051)
    if cl==31 and sty==0:
        p=Pen(X,Y)
        p.poly([(-.30,.16),(-.24,.035),(-.19,.00),(-.133,.068),(-.06,.11),(.078,.128),(.195,.179),(.243,.271),(.233,.359),(.153,.323),(.041,.270),(-.076,.249),(-.177,.23),(-.245,.191)])
        p.line([(-.238,.111),(-.281,.028),(-.351,.027)],.067)
        p.line([(-.244,.047),(-.242,-.064),(-.292,-.15)],.027);p.line([(-.245,-.013),(-.201,-.089)],.027)
        p.line([(-.133,.197),(-.263,.251),(-.329,.395)],.064);p.line([(-.129,.200),(-.260,.185),(-.393,.293)],.049)
        p.line([(.18,.26),(.237,.391),(.437,.509)],.069);p.line([(.180,.282),(.148,.418),(.344,.55)],.049)
    if cl==31 and sty==1:
        # The dominant silhouette is the long back, outstretched forelegs and
        # antlered head; keep these connected so they survive downsampling.
        p=Pen();p.poly([(-.345,.015),(-.285,-.060),(-.215,-.036),(-.145,.100),(-.04,.126),(.088,.166),(.199,.218),(.264,.323),(.248,.398),(.155,.373),(.054,.30),(-.072,.285),(-.170,.236),(-.218,.166),(-.264,.122),(-.341,.115),(-.377,.081)])
        p.line([(-.263,.048),(-.258,-.115),(-.329,-.230)],.047);p.line([(-.263,-.056),(-.193,-.145)],.034)
        p.line([(-.130,.220),(-.238,.303),(-.402,.350)],.070);p.line([(-.108,.238),(-.209,.338),(-.330,.441)],.049)
        p.line([(.200,.308),(.290,.421),(.448,.530)],.074);p.line([(.186,.316),(.213,.451),(.372,.568)],.052)
    if cl==23:
        p=Pen();q=Pen(X+.12*Y,Y-.22*X);q.a=p.a
        car(q,0,.10,.53,.46)
        for k in [-.17,.16]:p.curve([(k,.29),(k+.12,.43),(k-.25,.398),(k-.18,.51),(k-.13,.58),(k-.03,.585),(k-.06,.626)],.054)
    return p.a

def stoptext():
    p=Pen(); letters=[(-.54,'S'),(-.27,'T'),(.02,'O'),(.31,'P')]
    for x,ch in letters:
        q=Pen((X-x)/.23,(Y+.21)/.42);q.a=p.a
        if ch=='S':q.curve([(.90,.17),(.70,-.10),(-.14,.05),(.13,.30),(.25,.50),(.92,.48),(.89,.73),(.90,1.03),(.18,1.04),(.04,.83)],.145)
        if ch=='T':q.line([(.04,.06),(.96,.06)],.15);q.line([(.5,.06),(.5,.94)],.16)
        if ch=='O':digit(q,'0',0)
        if ch=='P':q.line([(.10,.95),(.10,.07),(.56,.07)],.16);q.curve([(.55,.07),(1.04,.04),(1.09,.55),(.57,.54)],.16);q.line([(.58,.54),(.12,.54)],.15)
    return p.a

nums={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120'}
groups=[[0,1,2,3,4,5,7,8,9,10,15,16],[11]+list(range(18,32)),list(range(33,41)),[6,32,41,42]]

def icon(c,sty):
    if c in nums:return numerals(nums[c],sty)
    if c in (9,10,16,41,42):return vehicle(c,sty)
    if c in (11,*range(18,32)):return warning(c,sty)
    if c in range(33,41):return blue(c,sty)
    if c==6:return numerals('80',sty)
    return 0*X

def sample(a,x,y):
    xx=np.clip((x+1)*ns/2-.5,0,ns-1.00001); yy=np.clip((y+1)*ns/2-.5,0,ns-1.00001)
    i=xx.astype(int);j=yy.astype(int);f=xx-i;g=yy-j
    return ((1-f)*a[j,i]+f*a[j,i+1])*(1-g)+((1-f)*a[j+1,i]+f*a[j+1,i+1])*g

def grid(n,rect):
    x0,x1,y0,y1=rect
    return np.meshgrid(x0+(np.arange(n)+.5)*(x1-x0)/n,y0+(np.arange(n)+.5)*(y1-y0)/n)
rects=[(-.77,.77,-.55,.55),(-.63,.63,-.44,.74),(-.83,.83,-.83,.83),(-.79,.79,-.59,.59)]

def supp(x,y,g):
    if g==0:return np.clip((.805-np.hypot(x,y))*16,0,1)
    if g==1:return np.clip(((y+.57)*.53+.016-np.abs(x))*16,0,1)*np.clip((.75-y)*18,0,1)
    if g==2:return np.clip((.895-np.hypot(x,y))*16,0,1)
    return np.clip((.86-np.hypot(x,y))*16,0,1)

def norm(a,mask):
    a=(a-(a*mask).sum()/mask.sum())*np.sqrt(mask)
    return a.ravel()/max(np.linalg.norm(a),1e-6)

def boxblur(a):
    p=np.pad(a,1,mode='edge');return (p[1:-1,1:-1]*4+(p[2:,1:-1]+p[:-2,1:-1]+p[1:-1,2:]+p[1:-1,:-2])*2+p[:-2,:-2]+p[2:,2:]+p[:-2,2:]+p[2:,:-2])/16

def vectors():
    out={}
    for g,classes in enumerate(groups):
        n=20; xx,yy=grid(n*3,rects[g]); x,y=grid(n,rects[g]); mask=supp(x,y,g)
        out['mask'+str(g)]=mask.astype('f'); bank=[]; cut=[0]; penalty=[]
        for cl in classes:
            if cl in [15,32]:cut.append(len(bank));continue
            for sty in (0,1):
                a=icon(cl,sty)
                for sx in (.83,1.,1.17):
                    for sy in (.80,1.,1.20):
                        for du in (-.048,0,.048):
                            for dv in (-.064,0,.064):
                                cy=.22 if g==1 else 0
                                z=sample(a,(xx-du)/sx, (yy-dv-cy)/sy+cy).reshape(n,3,n,3).mean((1,3))
                                # Cancellation stripe is ignored when correlating the residual pictogram.
                                v=boxblur(z)*.4+z*.6
                                filt=mask if g!=3 else mask*np.clip((np.abs(x+y)-.20)*9,0,1)
                                f1=norm(v,filt)
                                coarse=boxblur(boxblur(v));f2=norm(coarse,filt)
                                bank.append(np.r_[f1*.70,f2*.30]);penalty.append(.012*(abs(sx-1)+abs(sy-1))*4+.004*(abs(du)/.048+abs(dv)/.064))
            cut.append(len(bank))
        out['bank'+str(g)]=np.asarray(bank,dtype='f');out['cuts'+str(g)]=np.array(cut);out['cost'+str(g)]=np.array(penalty,dtype='f')
        if g==0:
            # Recover the shape channel of each individual template; local
            # re-normalizations ignore the common final zero in speed limits.
            parts=[];q=np.maximum(np.sqrt(mask),.001)
            masks=[mask*soft(-.07,.025,z-x) *soft(.62,.43,abs(y)) for z in (0,.18)]
            masks[1]=mask*soft(-.43,-.265,x)*soft(.28,.15,x)*soft(.53,.39,abs(y))
            for cl,i,j in zip(classes,cut[:-1],cut[1:]):
                if cl not in nums:continue
                m=masks[int(len(nums[cl])==3)]
                for row in bank[i:j]:
                    im=row[:n*n].reshape(n,n)/q;coarse=boxblur(boxblur(im));parts.append(np.r_[.76*norm(im,m),.24*norm(coarse,m)])
            out['speedparts']=np.array(parts,'f');out['wordmasks']=np.array(masks,'f')
    return out

# Coarse localization uses paired interior/border and inside/outside contrasts.
# The signed distance is positive inside. Normalize by effective inradius for the triangle.
def distance(x,y,f):
    if f in (0,2,5,7):return 1-np.hypot(x,y)
    if f in (1,4):
        u=y if f==1 else -y
        return np.minimum((1-u),((u+1)*.5-abs(x))*.8944)
    if f==3:return (1-abs(x)-abs(y))*.7071
    return np.minimum(np.minimum(1-abs(x),1-abs(y)),(1.4142-abs(x)-abs(y))*.7071)

def soft(lo,hi,x):return np.clip((x-lo)/(hi-lo),0,1)
def meanmask(m):return m/max(.001,m.sum())

def localizers():
    n=16; xx,yy=np.meshgrid((np.arange(n)+.5)*2-.5,(np.arange(n)+.5)*2-.5)
    banks=[];params=[]; offsets=[0]; penalties=[]
    for f in range(8):
        # 0 ring, 1 warning, 2 blue disk, 3 diamond, 4 yield, 5 cancellation, 6 stop, 7 entry
        thick={0:.176,1:.131,2:.075,3:.14,4:.15,5:.07,6:.07,7:.07}[f]
        for cx in (13.5,15.5,17.5):
            for cy in (13.5,15.5,17.5):
                for rx in (8.8,10.7,12.6,14.5):
                    for aspect in (.84,1,1.18):
                        ry=rx*aspect; x=(xx-cx)/rx;y=(yy-cy)/ry;d=distance(x,y,f)
                        s=.12
                        outer=soft(-s,s,d);ins=soft(-s,s,d-thick)
                        ring=meanmask(outer-ins);inside=meanmask(ins)
                        outside=meanmask(np.exp(-((d+.12)/.13)**2)*soft(-.02,.12,-d))
                        # color/red-chroma and achromatic contrast targets
                        if f in (0,1,4):
                            col=ring-inside*.75-outside*.25
                            lum=inside-ring
                        elif f==3:
                            col=inside-.7*ring-.3*outside;lum=ring-inside
                        elif f==5:
                            stripe=meanmask(ins*np.exp(-((x+y)/.18)**4)); wh=meanmask(ins*soft(.20,.37,abs(x+y))); col=inside-ring;lum=.65*(wh-stripe)+.35*(wh-outside)
                        else:
                            # Blue, stop, no entry all have a saturated ground and a white detail.
                            if f==7:white=meanmask(ins*soft(-.05,.05,.17-abs(y))*soft(-.06,.06,.75-abs(x)))
                            elif f==6:white=meanmask(ins*soft(-.03,.06,.21-abs(y))*soft(-.1,.1,.64-abs(x)))
                            else:white=inside
                            col=inside-.63*outside-.37*ring
                            if f in (6,7):col+=.48*(inside-white)
                            lum=(ring-inside)*(.50 if f==2 else .25)+(white-inside)*(.85 if f==7 else .50)
                        banks.append(np.r_[col.ravel(),lum.ravel()]);params.append([cx,cy,rx,ry]);penalties.append(.012*((cx-15.5)**2+(cy-15.5)**2)/4+.013*((rx-12.4)/2.5)**2+.012*abs(aspect-1)*5)
        offsets.append(len(params))
    return {'local':np.asarray(banks,'f'),'params':np.asarray(params,'f'),'offsets':np.array(offsets),'prior':np.array(penalties,'f')}

solution = r'''import numpy as np

classes=[[0,1,2,3,4,5,7,8,9,10,15,16],[11]+list(range(18,32)),list(range(33,41)),[6,32,41,42]]
rects=[(-.77,.77,-.55,.55),(-.63,.63,-.44,.74),(-.83,.83,-.83,.83),(-.79,.79,-.59,.59)]

def grid(n,rect):
    x0,x1,y0,y1=rect
    return np.meshgrid((x0+(np.arange(n,dtype=np.float32)+.5)*(x1-x0)/n),(y0+(np.arange(n,dtype=np.float32)+.5)*(y1-y0)/n))

def soft(a,b,x):return np.clip((x-a)/(b-a),0,1)

def colors(img):
    r,g,b=img[:,0],img[:,1],img[:,2]
    intensity=.35*r+.45*g+.20*b
    # Mild dynamic compression preserves differences in shadow, without amplifying sensor zeros.
    floor=.025*np.mean(intensity,axis=(1,2),keepdims=True)+.004
    lum=np.sqrt(intensity+floor);lo=np.quantile(lum,.07,axis=(1,2)).astype(np.float32)[:,None,None];hi=np.quantile(lum,.93,axis=(1,2)).astype(np.float32)[:,None,None]
    lum=np.clip((lum-lo)/(hi-lo+.011),-.25,1.25)
    red=np.clip((r-.52*g-.48*b)/(r+.52*g+.48*b+floor)*2.6,-1,1)
    blue=np.clip((b-.65*r-.35*g)/(b+.65*r+.35*g+floor)*3.1,-1,1)
    yellow=np.clip((.52*r+.48*g-b)/(.52*r+.48*g+b+floor)*2.9,-1,1)
    return np.stack((red,blue,yellow,lum),axis=1)

def dist(x,y,f):
    if f in (0,2,5,7):return 1-np.sqrt(x*x+y*y)
    if f in (1,4):
        u=y if f==1 else -y
        return np.minimum(1-u,((u+1)*.5-np.abs(x))*.8944)
    if f==3:return (1-np.abs(x)-np.abs(y))*.7071
    return np.minimum(np.minimum(1-np.abs(x),1-np.abs(y)),(1.4142-np.abs(x)-np.abs(y))*.7071)

def pull(img,cfg,x,y):
    # cfg is (center x,y, half width,height, horizontal shear)
    cx,cy,rx,ry,sh=cfg.T
    xx=np.clip(cx[:,None,None]+rx[:,None,None]*x+sh[:,None,None]*y,0,30.9999)
    yy=np.clip(cy[:,None,None]+ry[:,None,None]*y,0,30.9999)
    i=xx.astype(np.int32);j=yy.astype(np.int32);fx=(xx-i).astype(np.float32);fy=(yy-j).astype(np.float32)
    t=np.arange(len(img))[:,None,None]
    return ((1-fx)*img[t,j,i]+fx*img[t,j,i+1])*(1-fy)+((1-fx)*img[t,j+1,i]+fx*img[t,j+1,i+1])*fy

def dotbank(a,b):
    # Fixed GEMM dimensions even in a short last batch, for order/batch independence.
    n=len(a);p=np.zeros((32,a.shape[1]),np.float32);p[:n]=a
    return (p@b.T)[:n]

def localize(ch):
    n=len(ch);small=ch.reshape(n,4,16,2,16,2).mean((3,5)).reshape(n,4,256)
    opts=[];vals=[];off=CONSTANTS['offsets']; bank=CONSTANTS['local']
    for f,k in enumerate((0,0,1,2,0,0,0,0)):
        col=small[:,k];lum=small[:,3];gain=(.79,.80,.96,1.02,.88,.04,1.04,1.04)[f]
        # Outer boundaries also provide information when red has lost its saturation.
        a=np.concatenate((col*gain,lum*(.79,.83,.48,.53,.90,1.12,.57,.94)[f]),axis=1)
        a=dotbank(a,bank[off[f]:off[f+1]])-CONSTANTS['prior'][off[f]:off[f+1]]
        kmax=a.argmax(1);par=CONSTANTS['params'][off[f]+kmax]
        opts.append(np.column_stack((par,np.zeros(n,np.float32))));vals.append(a[np.arange(n),kmax])
    return np.stack(opts,axis=1),np.stack(vals,axis=1)

def blur(a):
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='edge');return (p[:,1:-1,1:-1]*4+(p[:,2:,1:-1]+p[:,:-2,1:-1]+p[:,1:-1,2:]+p[:,1:-1,:-2])*2+p[:,:-2,:-2]+p[:,2:,2:]+p[:,:-2,2:]+p[:,2:,:-2])/16

def avg(a,m):return np.sum(a*m,axis=(-2,-1))/(.00001+np.sum(m,axis=(-2,-1)))

def normalize(a,mask):
    a=(a-avg(a,mask)[:,None,None])*np.sqrt(mask)
    v=a.reshape(len(a),-1);return v/np.sqrt(.001+(v*v).sum(1,keepdims=True))

def confighyp(cfg,f):
    # Small coordinate search around a coarse shape pose. Equal-size hypotheses for every input.
    p=cfg[:,None,:]+CONSTANTS['delta'][None,:,:];n=len(p)
    return p.reshape(n,-1,5)

def refine(ch,cfg,f):
    # Match the sign border to pairs of samples on its two sides.
    pts=CONSTANTS['probes'+str(f)];p=confighyp(cfg,f);n,h,j=p.shape
    cx,cy,rx,ry,sh=p.transpose(2,0,1)[:,:,:,None];u,v=pts.T
    xx=np.clip(cx+rx*u+sh*v,0,30.9999);yy=np.clip(cy+ry*v,0,30.9999)
    i=xx.astype(np.int32);j=yy.astype(np.int32);fx=(xx-i).astype(np.float32);fy=(yy-j).astype(np.float32)
    im=ch[:,0] if f in (0,1,4,6,7) else ch[:,1 if f==2 else 2]
    if f in (0,1,4):im=.87*im-.85*ch[:,3]
    if f in (2,3,6,7):im=.74*im-.46*ch[:,3]
    if f==5:im=ch[:,3]
    t=np.arange(n)[:,None,None];z=(((1-fx)*im[t,j,i]+fx*im[t,j,i+1])*(1-fy)+((1-fx)*im[t,j+1,i]+fx*im[t,j+1,i+1])*fy).reshape(n,h,3,-1)
    if f in (0,1,4):d=z[:,:,1]-.75*z[:,:,2]-.25*z[:,:,0]
    elif f==5:d=z[:,:,2]-z[:,:,0]
    else:d=z[:,:,2]-.60*z[:,:,1]-.40*z[:,:,0]
    score=np.clip(d,-.65,1.45).mean(-1)-CONSTANTS['dcost']
    return p[np.arange(n),score.argmax(1)]

def borderpose(ch,cfg,f):
    # Dense photometric likelihood; multiple boundary sectors must agree on the
    # pose, unlike an overall mean that can be dominated by just one bright edge.
    extra=CONSTANTS['fine'];p=cfg[:,None,:]+extra[None,:,:];n,h,j=p.shape
    pts=CONSTANTS['probes'+str(f)];u,v=pts.T;cx,cy,rx,ry,sh=p.transpose(2,0,1)[:,:,:,None]
    xx=np.clip(cx+rx*u+sh*v,0,30.9999);yy=np.clip(cy+ry*v,0,30.9999);i=xx.astype(np.int32);j=yy.astype(np.int32);fx=(xx-i).astype(np.float32);fy=(yy-j).astype(np.float32);t=np.arange(n)[:,None,None]
    im=ch[:,0]*.76-ch[:,3]
    z=(((1-fx)*im[t,j,i]+fx*im[t,j,i+1])*(1-fy)+((1-fx)*im[t,j+1,i]+fx*im[t,j+1,i+1])*fy).reshape(n,h,3,-1)
    inner=z[:,:,1]-z[:,:,2];outer=z[:,:,1]-z[:,:,0];v=.74*np.clip(inner,-.5,1.20)+.26*np.clip(outer,-.5,1.20)
    v=v.reshape(n,h,-1,3).mean(3);return p[np.arange(n),(v.mean(2)-.28*v.std(2)-CONSTANTS['finecost']).argmax(1)]

def inspect(ch,cfg,f):
    # Absolute color evidence is weak in coloured daylight, so also use the color *change* at the edge.
    xx,yy=grid(28,(-1.18,1.18,-1.18,1.18)); d=dist(xx,yy,f)
    k=(0,0,1,2,0,0,0,0)[f];c=pull(ch[:,k],cfg,xx,yy);l=pull(ch[:,3],cfg,xx,yy)
    th=(.18,.13,.08,.145,.15,.07,.07,.07)[f]
    ring=soft(-.035,.035,d)*(1-soft(th-.04,th+.04,d)); inner=soft(th,th+.07,d);outside=soft(-.19,-.10,d)*(1-soft(-.06,.005,d))
    cr,ci,co=avg(c,ring),avg(c,inner),avg(c,outside);lr,li,lo=avg(l,ring),avg(l,inner),avg(l,outside)
    if f in (0,1,4):
        if f==0:sec=[(xx>0,yy>0),(xx<0,yy>0),(xx<0,yy<0),(xx>0,yy<0)]
        else:
            vv=yy if f==1 else -yy;sec=[(vv<.68,xx<0),(vv<.68,xx>0),(vv>.75,np.abs(xx)<.82)]
        edges=np.stack([avg(c-l,ring*a*b) for a,b in sec],axis=1);mixed=ci-li
        reliable=np.clip(edges-mixed[:,None],-.8,1.6)
        return .70*(cr-ci)+.21*(cr-co)+.27*cr+.82*(li-lr)+.09*(li-lo)-.29*reliable.std(1)+.09*np.min(reliable,axis=1)
    if f==3:
        return .76*(ci-cr)+.26*(ci-co)+.40*ci+.78*(lr-li)+.20*(lr-lo)
    if f==5:
        stripe=inner*soft(.27,.13,np.abs(xx+yy));wh=inner*soft(.26,.41,np.abs(xx+yy))
        diag=avg(l,wh)-avg(l,stripe)
        # Look at the two remote ends of the band separately from the symbol it cancels.
        a=np.abs(yy)>.30;ends=avg(l,wh*a)-avg(l,stripe*a);anti=inner*soft(.26,.13,np.abs(xx-yy))
        selectivity=avg(l,anti*a)-avg(l,stripe*a)
        return 1.23*diag+.55*ends+.39*selectivity+.31*(avg(l,wh)-lo)-.34*np.abs(cr-ci)-.24*np.abs(cr)
    if f==2:
        # Color under a white arrow may be weak; look for blue in multiple
        # quadrants, rather than treating external sky as part of the disc.
        field=inner*(d>.22);parts=np.stack([avg(c,field*(xx*s>0)*(yy*t>0)) for s,t in ((1,1),(-1,1),(1,-1),(-1,-1))],1)
        return .61*(ci-cr)+.36*(ci-co)+.50*ci+.52*(lr-li)+.13*(lr-lo)+.23*(parts.min(1)-.16)
    text=(np.abs(yy)<(.20 if f==6 else .16))*(np.abs(xx)<(.59 if f==6 else .72))
    field=inner*(1-soft(.15,.29,.36-np.abs(yy)))
    cf=avg(c,field);lf=avg(l,field);ct=avg(c,text);lt=avg(l,text)
    abscontrast=(lt-lf)+.55*(cf-ct)
    # Along the middle of an entry bar, red-normalized brightness is
    # uniformly high. STOP instead has ink gaps right through its middle.
    center=(np.abs(yy)<.102)&(np.abs(xx)<.55);signal=l-.55*c
    qlo=np.quantile(signal[:,center],.17,axis=1);qhi=np.quantile(signal[:,center],.86,axis=1);fld=lf-.55*cf
    continuity=(qlo-fld)/(.065+np.maximum(.0,qhi-fld))
    if f==6:
        # STOP lettering has interleaved vertical strokes, not a continuous bar.
        u,v=grid(24,(-.74,.74,-.35,.35));txt=pull(ch[:,3]-.36*ch[:,0],cfg,u,v);pat=CONSTANTS['stopt'];corr=(normalize(txt,np.ones_like(pat))*normalize(blur(pat[None,:,:]),np.ones_like(pat))).sum(1)
        fieldtest=(d>.135)&(np.abs(yy)>.28);fieldred=np.quantile(c[:,fieldtest],.30,axis=1)
        # A speed-limit numeral is dark ink on pale ground; STOP has the
        # opposite polarity. Verify it in the central field, not at the rim.
        quiet=(d>.18)&(np.abs(yy)>.30)&(np.abs(yy)<.69);gap=avg(signal,quiet);lightletters=qhi-gap;darkletters=gap-qlo
        polarity=np.clip((lightletters-darkletters)/(.07+np.maximum(.0,qhi-qlo)),-1,1)
        return .42*(cf-cr)+.30*(cf-co)+.39*cf+.44*(lr-lf)+.36*abscontrast+.69*(corr-.20)+.36*(fieldred-.10)-1.05*np.maximum(0,continuity-.37)+.30*(polarity-.34)
    return .28*(cf-cr)+.28*(cf-co)+.45*cf+.27*(lr-lf)+.85*abscontrast+.54*(np.clip(continuity,-.5,1)-.42)

def icons(ch,cfg,g):
    xx,yy=grid(20,rects[g]);mask=CONSTANTS['mask'+str(g)]
    lum=pull(ch[:,3],cfg,xx,yy);red=pull(ch[:,0],cfg,xx,yy);blue=pull(ch[:,1],cfg,xx,yy)
    if g==2: ink=lum-.44*blue
    else:ink=-lum+(.14*red if g==0 else 0)
    if g==1:
        # Extrapolate local white paper inward. Ignore the red frame and shadows
        # outside the triangle when estimating how dark a pictogram stroke is.
        m=(mask>.35);p=np.pad(np.where(m,lum,-2),((0,0),(3,3),(3,3)),mode='edge');white=np.maximum.reduce([p[:,i:i+20,j:j+20] for i in (0,3,6) for j in (0,3,6)])
        high=np.quantile(lum[:,m],.90,axis=1).astype(np.float32)[:,None,None];ink=ink+.61*np.clip(white,high-.55,high+.06)
    if g==3:mask=mask*soft(.20,.312,np.abs(xx+yy))
    flat=avg(ink,mask);sd=np.sqrt(avg((ink-flat[:,None,None])**2,mask))
    contrast=sd/(.074+sd)
    z=ink; v=normalize(z,mask);w=normalize(blur(blur(z)),mask)
    a=dotbank(np.concatenate((v,w),axis=1),CONSTANTS['bank'+str(g)])-CONSTANTS['cost'+str(g)]
    cut=CONSTANTS['cuts'+str(g)];scores=[]
    for j,cl in enumerate(classes[g]):
        if cut[j]==cut[j+1]:s=np.maximum(-.25,1-sd/(.075 if g==0 else .074))
        else:s=a[:,cut[j]:cut[j+1]].max(axis=1)*(contrast*.41+.64)
        scores.append(s)
    sc=np.stack(scores,axis=1)
    if g==1:
        # Central printed ink provides a second alignment estimate, independent
        # of the red triangle. Fit a box by trimmed 1D ink mass at inference.
        valid=(mask>.8);bl=np.quantile(ink[:,valid],.20,axis=1).astype(np.float32);bh=np.quantile(ink[:,valid],.91,axis=1).astype(np.float32)
        weight=soft(.25,.70,(ink-bl[:,None,None])/(.022+bh[:,None,None]-bl[:,None,None]))*valid
        def span(m,z):
            c=np.cumsum(m,1);a=c[:,-1:];lo=(c<.036*a).sum(1).clip(0,19);hi=(c<.964*a).sum(1).clip(0,19);return z[lo],z[hi]
        xl,xr=span(weight.sum(1),xx[0]);yt,yb=span(weight.sum(2),yy[:,0]);xmid=(xl+xr)*.5;ymid=(yt+yb)*.5;height=yb-yt+.057;width=xr-xl+.06
        u,v=grid(24,(-.08,1.08,-.08,1.08));ax=xl[:,None,None]+(xr-xl+.001)[:,None,None]*u;ay=yt[:,None,None]+(yb-yt+.001)[:,None,None]*v
        tight=-pull(ch[:,3],cfg,ax,ay);m=np.ones((24,24),np.float32);tt=np.concatenate((normalize(tight,m),normalize(blur(blur(tight)),m)),1)
        da=dotbank(tt,CONSTANTS['details'])-CONSTANTS['detailcost'];ds=CONSTANTS['dsizes'];local=np.stack([da[:,a:b].max(1) for a,b in zip(ds[:-1],ds[1:])],1)
        sc=.75*sc+.25*local
        # At low resolution topology is lost, but gross glyph proportions remain informative.
        for j,cl in enumerate(classes[1]):
            # Physical aspect ranges from a very narrow exclamation mark to twin broad humps.
            ratio={11:.70,18:.23,19:.88,20:.88,21:.60,22:2.7,23:1.05,24:.59,25:1.23,26:.34,27:.66,28:1.03,29:1.56,30:1.04,31:1.35}[cl]
            aspect=np.log((width+.06)/(height+.06)/ratio);sc[:,j]-=.13*np.minimum(np.abs(aspect),1.7)
        wc=normalize(ink,mask); sm=blur(blur(ink));k=classes[1].index(31)
        # The two double bends are upright strokes joined by a transverse bar;
        # the deer's back and legs instead follow a continuous oblique axis.
        perp=(yy-ymid[:,None,None])-.70*(xx-xmid[:,None,None]);diagonal=np.exp(-((perp/.15)**2));diagmass=(weight*diagonal).sum((1,2))/(.01+weight.sum((1,2)))
        # A smooth oblique body has diagonal gradients concentrated on its
        # long edges. A double bend instead contains mostly vertical and
        # horizontal edges, even when its bounding rectangle is oblique.
        z=blur(ink);dx=z[:,1:-1,2:]-z[:,1:-1,:-2];dy=z[:,2:,1:-1]-z[:,:-2,1:-1];mm=valid[1:-1,1:-1]*(np.abs(xx[1:-1,1:-1])<.46)*(yy[1:-1,1:-1]<.54)
        tensor=(-2*dx*dy*mm).sum((1,2))/(.014+((dx*dx+dy*dy)*mm).sum((1,2)))
        sc[:,k]+=.40*(diagmass-.54)+.38*(tensor-.12);sc[:,classes[1].index(21)]-=.16*np.maximum(0,tensor-.23)
    if g==2:
        # Roundabout arrows encircle an empty center; all other blue signs
        # contain a shaft passing through or near the middle.
        r=np.hypot(xx,yy);core=avg(ink,r<.28);edge=avg(ink,(r>.46)&(r<.72));pool=(r<.81)
        dynamic=np.quantile(ink[:,pool],.90,axis=1)-np.quantile(ink[:,pool],.20,axis=1)
        hollow=np.clip((edge-core)/(.045+dynamic),-.65,.65)
        sc[:,-1]+=.58*hollow
    if g==0:
        sc=readinterior(ch,cfg,sc)
    if g==3:
        # Residual ink, counted only well away from the diagonal and edge.
        use=(np.hypot(xx,yy)<.70)&(np.abs(xx+yy)>.32)
        bg=(np.hypot(xx,yy)<.79)&(np.abs(xx+yy)>.35)
        white=np.quantile(lum[:,bg],.8,axis=1).astype(np.float32)
        q=np.maximum(0,white[:,None,None]-lum)
        hor=(np.abs(yy)<.27)&(np.abs(xx)>.20)&(np.abs(xx)<.68)&use
        str1=np.quantile(q[:,use],.87,axis=1);str2=np.quantile(q[:,hor],.80,axis=1)
        # Stripes remain even on a blank cancellation sign. Measure the
        # remaining ink with respect to stripe contrast in the same image.
        stripe=(np.abs(xx+yy)<.145)&(np.hypot(xx,yy)<.75);depth=white-avg(lum,stripe)
        strength=(.52*str1+.48*str2)/(.045+np.maximum(.03,depth))
        # Smooth broad illumination across the disc is not a printed vehicle.
        p=np.pad(lum,((0,0),(2,2),(2,2)),mode='edge');peak=np.maximum.reduce([p[:,i:i+20,j:j+20] for i in (0,2,4) for j in (0,2,4)])
        fine=np.quantile((peak-lum)[:,hor],.82,axis=1)/(.048+np.maximum(.03,depth))
        sc[:,1]=np.clip(1.03-1.32*(.50*strength+.50*fine),-.30,.85)
    return sc

def inkmap(ch,cfg):
    xx,yy=grid(32,(-.84,.84,-.84,.84));l=pull(ch[:,3],cfg,xx,yy);r=pull(ch[:,0],cfg,xx,yy);v=l-.14*r
    mask=np.hypot(xx,yy)<.68
    side=(np.abs(xx)<.46)&(np.abs(yy)>.46)&(np.hypot(xx,yy)<.72)
    white=np.quantile(v[:,mask],.84,axis=1).astype(np.float32)
    # Limit smooth illumination gradients by estimating the local bright paper.
    p=np.pad(v,((0,0),(5,5),(5,5)),mode='edge');a=np.maximum.reduce([p[:,i:i+32,j:j+32] for i in (1,5,9) for j in (1,5,9)])
    ink=np.maximum(0,.78*a+.22*white[:,None,None]-v)
    profile=(np.abs(xx)<.56)&(np.abs(yy)<.41)
    amp=np.quantile(ink[:,profile],.87,axis=1).astype(np.float32)
    paper=np.quantile(ink[:,side],.7,axis=1).astype(np.float32)
    return ink,xx,yy,amp,paper

def readinterior(ch,cfg,sc):
    ink,x,y,amp,paper=inkmap(ch,cfg);n=len(cfg)
    # Segmentation treats the number as a printed word. Its overall extent is
    # estimated from ink mass; each candidate reading must explain all its glyphs.
    usable=(np.hypot(x,y)<.728)&(np.abs(y)<.53)
    pix=soft(.24,.69,(ink-paper[:,None,None]*.55)/(.032+amp[:,None,None]))*usable
    pix*=soft(.17,.36,(ink+.005)/(.04+amp[:,None,None]))
    px=pix.sum(1); py=pix.sum(2)
    def interval(m,axis,lo,hi):
        cum=np.cumsum(m,axis=1);z=cum[:,-1:];u=(cum<z*lo).sum(axis=1).clip(0,31);v=(cum<z*hi).sum(axis=1).clip(0,31)
        return axis[u]-.023,axis[v]+.023
    xl,xr=interval(px,x[0],.021,.979);yt,yb=interval(py,y[:,0],.023,.977)
    cy=np.clip((yt+yb)*.5,-.13,.13);h=np.clip(yb-yt,.54,1.02);cx=np.clip((xl+xr)*.5,-.14,.14);total=np.clip(xr-xl,.61,1.49)
    # Crop using sign-relative coordinates; independent OCR metric validates template matching.
    u,v=grid(18,(-.11,1.11,-.08,1.08));src=ink
    def readat(pos,width):
        par=np.column_stack(((15.5+(cx+pos-total*.5)*32/1.68).astype(np.float32),(15.5+(cy-h*.5)*32/1.68).astype(np.float32),width*32/1.68,h*32/1.68,np.zeros(n,np.float32)))
        im=pull(src,par,u,v);co=blur(blur(im));m=np.ones((18,18),np.float32)
        f=np.concatenate((normalize(im,m),normalize(co,m)),1);a=dotbank(f,CONSTANTS['ocrbank']);cut=CONSTANTS['ocrcuts']
        part=np.concatenate([normalize(im[:,i:i+6,j:j+6],m[:6,:6]) for i in (0,6,12) for j in (0,6,12)],1);a=.64*a+.36*dotbank(part,CONSTANTS['ocrparts'])
        return np.stack([a[:,i:j].max(1) for i,j in zip(cut[:-1],cut[1:])],1)
    w=total*.44;first=readat(total*0,w);last=readat(total-w,w)
    first3=readat(total*0,total*.23);mid=readat(total*.295,total*.317);last3=readat(total*.687,total*.313)
    two=.62*first[:,[2,3,4,5,6,7]]+.38*last[:,0:1];three=.32*first3[:,1:2]+.40*mid[:,[0,2]]+.28*last3[:,0:1]
    ocr=np.concatenate((two,three),1)
    sc[:,:8]=.65*sc[:,:8]+.35*ocr
    # A blank disc should not turn into a number just because the outer ring
    # leaks into the sampled window. Compare the central stroke contrast with
    # the contrast in the white margin and the sign rim.
    flat=np.abs(y)<.42;central=flat&(np.abs(x)<.55);cap=(np.hypot(x,y)<.69)&(np.abs(y)>.40)
    # Use a contrast ratio, not a threshold on absolute brightness. Shadowed
    # numerals can have very little global contrast yet still form clear strokes.
    raw=pull(ch[:,3],cfg,x,y);qhi=np.quantile(raw[:,central],.86,axis=1);qlo=np.quantile(raw[:,central],.13,axis=1)
    back=avg(raw,cap);core=np.quantile(raw[:,central],.25,axis=1)
    rimx,rimy=grid(30,(-1.,1.,-1.,1.));rr0=np.hypot(rimx,rimy);edge=avg(pull(ch[:,3],cfg,rimx,rimy),(rr0>.845)&(rr0<.97));ringdepth=np.maximum(.055,back-edge)
    signal=(.70*np.maximum(0,amp-.36*paper)+.30*np.maximum(0,back-core))/(.04+ringdepth)
    sc[:,10]=np.minimum(.786,1.05-2.8*signal)-.22*np.maximum(0,sc[:,:8].max(1)-.57)
    # A black silhouette, two black / red cars, and printed numerals also differ in hue.
    left=(x<-.07)&(x>-.58)&(np.abs(y)<.19);right=(x>.06)&(x<.59)&(np.abs(y)<.19)
    rr=pull(ch[:,0],cfg,x,y);warm=avg(rr,left)-avg(rr,right)
    sc[:,8:10]+=.20*np.maximum(0,warm[:,None]);sc[:,11]-=.22*np.maximum(0,warm)
    # A projection-based foreground detector is an independent source of scale
    # for the whole-word templates. It helps if the red ring is not concentric
    # with the word (for instance after a perspective distortion).
    u,v=grid(20,rects[0]);m=CONSTANTS['mask0'];b=CONSTANTS['bank0'];cost=CONSTANTS['cost0'];cut=CONSTANTS['cuts0']
    for length,j0,j1 in ((1.02,0,6),(1.17,6,8)):
        ww=np.clip(total/length,.71,1.34);hh=h/(.84 if j0==0 else .78)
        par=np.column_stack((15.5+cx*32/1.68,15.5+cy*32/1.68,ww*32/1.68,hh*32/1.68,np.zeros(n,np.float32)))
        word=pull(ink,par,u,v);wordfeat=np.concatenate((normalize(word,m),normalize(blur(blur(word)),m)),axis=1)
        mat=dotbank(wordfeat,b[cut[j0]:cut[j1]])-cost[cut[j0]:cut[j1]]
        res=np.stack([mat[:,cut[j]-cut[j0]:cut[j+1]-cut[j0]].max(1) for j in range(j0,j1)],1)
        sc[:,j0:j1]=.55*sc[:,j0:j1]+.25*res+.20*np.maximum(res,sc[:,j0:j1])
    # Condition on the number of glyphs before inspecting the distinctive
    # part of the word. A leading 1 should not be penalized for a bad fit
    # of the last zero shared by both three-digit classes.
    for length,j0,j1,ind in ((1.02,0,6,0),(1.17,6,8,1)):
        mask=CONSTANTS['wordmasks'][ind];partbank=CONSTANTS['speedparts'][cut[j0]:cut[j1]]
        baseword=-pull(ch[:,3],cfg,u,v);ww=np.clip(total/length,.71,1.34);hh=h/(.84 if ind==0 else .78)
        par=np.column_stack((15.5+cx*32/1.68,15.5+cy*32/1.68,ww*32/1.68,hh*32/1.68,np.zeros(n,np.float32)))
        aligned=pull(ink,par,u,v);cor=[]
        for wim in (baseword,aligned):
            feat=np.concatenate((normalize(wim,mask),normalize(blur(blur(wim)),mask)),axis=1)
            # Pair a local and a whole-word likelihood at the very same pose:
            # the zero provides an anchor, not a free extra template shift.
            wf=np.concatenate((normalize(wim,m),normalize(blur(blur(wim)),m)),axis=1)
            s=.56*dotbank(feat,partbank)+.44*dotbank(wf,b[cut[j0]:cut[j1]])-cost[cut[j0]:cut[j1]]
            cor.append(np.stack([s[:,cut[j]-cut[j0]:cut[j+1]-cut[j0]].max(1) for j in range(j0,j1)],1))
        res=.46*cor[0]+.39*cor[1]+.15*np.maximum(cor[0],cor[1]);sc[:,j0:j1]=.60*sc[:,j0:j1]+.40*res+.35*(res-res.max(1,keepdims=True))
    return sc

def classify(img):
    n=len(img);ch=colors(img);opts,vals=localize(ch);allscore=np.full((n,43),-10,np.float32)
    bases=[];poses=[]
    for f in range(8):
        cfg=refine(ch,opts[:,f],f)
        if f in (0,1,4):cfg=borderpose(ch,cfg,f)
        poses.append(cfg);bases.append(inspect(ch,cfg,f))
    base=np.stack(bases,axis=1)
    for g,f in enumerate((0,1,2,5)):
        sc=icons(ch,poses[f],g);allscore[:,classes[g]]=base[:,f:f+1]+.89*(sc-.56)
    for f,c in ((3,12),(4,13),(6,14),(7,17)):allscore[:,c]=base[:,f]
    return allscore.argmax(1)

def predict(x):
    ans=np.empty(len(x),np.int64)
    for i in range(0,len(x),32):ans[i:i+32]=classify(x[i:i+32])
    return ans
'''

def refiners():
    arr=[]; costs=[]
    for dx in [-1.,0.,1.]:
        for dy in [-1.,0.,1.]:
            for dr in [-.85,0.,.85]:
                for dh in [-.85,0,.85]:
                    for sh in [-1.25,0.,1.25]:
                        arr.append([dx,dy,dr,dh,sh]);costs.append(.006*(abs(dx)+abs(dy))+.005*(abs(dr)+abs(dh)+abs(sh)))
    fine=[];fc=[]
    for x in (-.55,0,.55):
        for y in (-.55,0,.55):
            for w in (-.6,0,.6):
                for h in (-.6,0,.6):
                    fine.append([x,y,w,h,0]);fc.append(.004*(abs(x)+abs(y)+abs(w)+abs(h)))
    ans={'delta':np.array(arr,'f'),'dcost':np.array(costs,'f'),'fine':np.array(fine,'f'),'finecost':np.array(fc,'f')}
    for f in range(8):
        if f in (0,2,5,7):
            t=np.arange(36)*(2*np.pi/36); pts=np.stack((np.cos(t),np.sin(t)),axis=1); center=np.array([0.,0.])
        else:
            if f in (1,4):verts=[(0,-1),(-1,1),(1,1)];center=np.array([0.,.33]);
            elif f==3:verts=[(0,-1),(-1,0),(0,1),(1,0)];center=np.array([0.,0.]);
            else:verts=[(-.41,-1),(.41,-1),(1,-.41),(1,.41),(.41,1),(-.41,1),(-1,.41),(-1,-.41)];center=np.array([0.,0.]);
            t=(np.arange(9)+.5)/9;pts=np.concatenate([np.array(v)[None,:]*(1-t[:,None])+np.array(w)[None,:]*t[:,None] for v,w in zip(verts,verts[1:]+verts[:1])]);
            if f==4:pts[:,1]*=-1;center[1]*=-1
        scales=(1.09,.906,.736) if f==0 else ((1.12,.886,.73) if f in (1,4) else (1.13,.96,.74))
        ans['probes'+str(f)]=np.concatenate([center+(pts-center)*s for s in scales]).astype('f')
    return ans

def detailbank():
    # Each tight template is normalized independently, ignoring the white
    # margins that otherwise give partial glyph matches too much influence.
    x,y=grid(24*3,(-.08,1.08,-.08,1.08));bank=[];groupsizes=[0];cost=[]
    for cl in groups[1]:
        for sty in (0,1):
            a=icon(cl,sty);density=a.sum();cs=a.sum(0).cumsum();rs=a.sum(1).cumsum();left,right=[axis[np.searchsorted(cs,density*t)] for t in (.036,.964)];top,bot=[axis[np.searchsorted(rs,density*t)] for t in (.036,.964)]
            for sx in (.90,1.0,1.11):
                for sy in (.88,1.0,1.14):
                    for ox in (-.057,0,.057):
                        for oy in (-.064,0,.064):
                            zz=sample(a,((x-.5-ox)/sx+.5)*(right-left)+left,((y-.5-oy)/sy+.5)*(bot-top)+top).reshape(24,3,24,3).mean((1,3));m=np.ones_like(zz)
                            bank.append(np.r_[.72*norm(zz,m),.28*norm(boxblur(boxblur(zz)),m)]);cost.append(.048*(abs(sx-1)+abs(sy-1)+abs(ox)+abs(oy)))
        groupsizes.append(len(bank))
    return {'details':np.array(bank,'f'),'dsizes':np.array(groupsizes),'detailcost':np.array(cost,'f')}

def ocrbank():
    x,y=grid(18*3,(-.11,1.11,-.08,1.08)); bank=[]; cut=[0];part=[]
    for c in '01235678':
        for sty in (0,1):
            for wide in (.87,1,1.13):
                for ox in (-.075,0,.075):
                    for oy in (-.048,0,.048):
                        if c=='1':wide1=1.00*wide
                        else:wide1=wide
                        p=Pen((x-.5-ox)/wide1+.5,y-oy);digit(p,c,sty)
                        z=p.a.reshape(18,3,18,3).mean((1,3));m=np.ones_like(z)
                        bank.append(np.r_[.76*norm(z,m),.24*norm(boxblur(boxblur(z)),m)])
                        zz=.68*z+.32*boxblur(z);part.append(np.concatenate([norm(zz[i:i+6,j:j+6],m[:6,:6])/9 for i in (0,6,12) for j in (0,6,12)]))
        cut.append(len(bank))
    return {'ocrbank':np.array(bank,'f'),'ocrparts':np.array(part,'f'),'ocrcuts':np.array(cut),'stopt':sample(stoptext(),*grid(24,(-.74,.74,-.35,.35))).astype('f')}

def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True)
    (out/'solution.py').write_text(solution);constants=vectors();constants.update(localizers());constants.update(refiners());constants.update(ocrbank());constants.update(detailbank());np.savez(out/'constants.npz',**constants)
if __name__=='__main__':main()
