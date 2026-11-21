"""Data-free construction of traffic sign geometry and the recognition program."""
import math
from pathlib import Path
import sys
import numpy as np

# Distances on a canonical sign with horizontal/vertical outer radii of one.
def ellipse(x,y,a,b,w,h):
    return ((x-a)/w)**2+((y-b)/h)**2<=1

def polygon(x,y,points):
    out=np.zeros_like(x,dtype=bool)
    for (a,b),(c,d) in zip(points,points[1:]+points[:1]):
        if b!=d: out ^= ((y>b)!=(y>d)) & (x<(c-a)*(y-b)/(d-b)+a)
    return out

def line(x,y,coords,width):
    p=coords[0]; mask=np.zeros_like(x,dtype=bool)
    for q in coords[1:]:
        dx=q[0]-p[0];dy=q[1]-p[1];t=np.clip(((x-p[0])*dx+(y-p[1])*dy)/(dx*dx+dy*dy+1e-20),0,1)
        mask |= (x-p[0]-t*dx)**2+(y-p[1]-t*dy)**2<=(width/2)**2;p=q
    return mask

def bezier(start,*curves):
    ans=[start];p=np.array(start)
    for c1,c2,end in curves:
        c=np.array(c1);d=np.array(c2);e=np.array(end)
        for t in np.arange(1,11)/10: ans.append(tuple(p*(1-t)**3+3*c*t*(1-t)**2+3*d*t*t*(1-t)+e*t**3))
        p=e
    return ans

def borderdist(f,x,y):
    if f in ('ring','blue','end','bar'): return np.sqrt(x*x+y*y)
    if f=='diamond': return abs(x)+abs(y)
    if f=='stop': return np.maximum(np.maximum(abs(x),abs(y)),(abs(x)+abs(y))/math.sqrt(2))
    if f=='yield':y=-y
    # round off the tips of a regular warning triangle
    g=np.maximum(y,2*abs(x)-y)
    return np.maximum(.96*g, np.maximum((abs(x)-.04)*1.06,(abs(y)-.04)*1.06))

families=['ring','tri','yield','blue','end','stop','bar','diamond']

def canonical(f,x,y):
    d=borderdist(f,x,y);dd=(np.maximum(-y,2*abs(x)+y) if f=='yield' else np.maximum(y,2*abs(x)-y)) if f in ('tri','yield') else d
    inside=d<1
    color=np.zeros_like(x)+.50
    lum=np.zeros_like(x)+.49
    if f in ('ring','end','tri','yield'):
        if f in ('tri','yield'):
            yy=y if f=='tri' else -y
            dd=np.maximum((yy-1/3)/(2/3),(2*abs(x)-yy+1/3)/(4/3))
        bound=.79 if f=='ring' else .97 if f=='end' else .77
        white=(dd<bound) & inside
        color[inside]=1;color[white]=0;lum[inside]=.10;lum[white]=.92
        if f=='end':lum[inside]=.92;color[inside]=0
        quiet=dd<( .65 if f=='ring' else .58 if f=='tri' else .92 if f=='end' else 0)
        if f in ('ring','tri','end'):lum[quiet]=.67 if f!='end' else .59
    if f in ('blue','stop','bar'):
        color[inside]=1;lum[inside]=.14
        if f in ('blue','stop'):
            edge=(d>.925) & inside;lum[edge]=.84;color[edge]=0
        if f=='bar':
            stripe=(abs(y)<.155)&(abs(x)<.83)&inside;lum[stripe]=.98;color[stripe]=0
        if f=='stop':
            band=(abs(y)<.36)&(abs(x)<.81);lum[band]=.62;color[band]=.35
        if f=='blue':
            quiet=d<.67;lum[quiet]=.32;color[quiet]=.77
    if f=='diamond':
        color[inside]=0;lum[inside]=1;yellow=d<.72;color[yellow]=1;lum[yellow]=.65
    return lum,color

def wholebank():
    # Coarse bank: search shape, horizontal/vertical scale and translation.
    yy,xx=np.mgrid[:32,:32];offs=np.array([-.27,.27]);xx=xx[:,:,None,None]+offs[None,None,:,None]+np.zeros((1,1,1,2));yy=yy[:,:,None,None]+offs[None,None,None,:]+np.zeros((1,1,2,1))
    poses=[];dl=[];dc=[]
    for f in families:
        fm=[];lc=[];cc=[]
        for r in (8,10.5,13,15):
            for aspect in (.85,1,1.17):
                for cy in (13,15.5,18):
                    for cx in (13,15.5,18):
                        w=r*aspect**.5;h=r/aspect**.5
                        x=(xx-cx)/w;y=(yy-cy)/h
                        l,c=canonical(f,x,y)
                        def unit(a):
                            a=a.mean((2,3));a=a-a.mean();return (a/(np.sqrt(np.sum(a*a))+1e-8)).ravel()
                        fm.append((cx,cy,w,h));lc.append(unit(l));cc.append(unit(c))
        poses.append(fm);dl.append(lc);dc.append(cc)
    return dict(pose=np.array(poses,np.float32),light=np.array(dl,np.float32),color=np.array(dc,np.float32))

# Digit outlines are pen paths specified in the character's own coordinate frame.
def digit(n,x,y,weight,style):
    X=x*.53;Y=y;stroke=weight
    def path(points):return line(X,Y,[(a*.53,b) for a,b in points],stroke)
    def curve(p,*parts):return path(bezier(p,*parts))
    if n==0:
        # DIN numerals have fairly straight sides and rounded ends.
        p=2.45 if style==0 else 2.9
        outer=(abs(X)/.51)**p+(abs(Y)/1.)**p<=1
        inner=(abs(X)/(.51-stroke))**p+(abs(Y)/(1.-stroke))**p<=1
        return outer & ~inner
    if n==1:return path([(-.67,-.55),(.24,-.87),(.24,.88)])
    if n==2:
        return curve((-.75,-.54),((-.69,-1.04),(.74,-1.04),(.73,-.43)),((.77,-.10),(.38,.09),(-.74,.78))) | path([(-.74,.78),(-.74,.87),(.79,.87)])
    if n==3:
        return curve((-.73,-.60),((-.34,-1.03),(.73,-.96),(.68,-.47)),((.68,-.18),(.39,-.04),(-.16,-.04))) | curve((-.17,-.04),((.52,-.07),(.84,.17),(.74,.50)),((.69,.99),(-.44,1.04),(-.77,.59)))
    if n==5:
        return path([(.69,-.88),(-.57,-.88),(-.65,-.04)]) | curve((-.64,-.04),((.06,-.38),(.78,-.23),(.75,.32)),((.86,.99),(-.45,1.05),(-.77,.62)))
    if n==6:
        ring=curve((-.70,.36),((-.79,-.19),(.66,-.33),(.74,.30)),((.89,.99),(-.69,1.12),(-.70,.36)))
        if style==0:return ring|curve((-.70,.36),((-.80,.01),(-.05,-.52),(.38,-.90)))
        return ring|curve((-.69,.38),((-.76,-.41),(-.49,-.98),(.48,-.85)))
    if n==7:
        return path([(-.73,-.72),(-.73,-.88),(.75,-.88),(-.39,.89)])
    if n==8:
        out=np.zeros_like(x,dtype=bool)
        for cy,h,a in [(-.47,.53,.48),(.45,.55,.51)]:
            out|=ellipse(X,Y,0,cy,a,h)&~ellipse(X,Y,0,cy,a-stroke,h-stroke)
        return out

def digitgroup(x,y,num,style,thick):
    seq=[int(c) for c in str(num)]
    h=.46 if len(seq)==2 else .44
    w=h*(.51 if len(seq)==2 else (.46 if style else .42))
    widths=[2*w*(.68 if d==1 else 1) for d in seq];gap=(.085 if len(seq)==2 else .060)
    total=sum(widths)+gap*(len(seq)-1);left=-total/2;centers=[]
    for n,wd in zip(seq,widths):
        centers.append(left+wd*.5+(.16*w if n==1 else 0));left+=wd+gap
    if style:w*=.94
    out=np.zeros_like(x,dtype=bool)
    for i,n in enumerate(seq):out|=digit(n,(x-centers[i])/w,y/h,thick,style)
    return out

def car(x,y,a,b,sx,sy):
    # Rear elevation, with windscreen, wide body and wheels.
    X=(x-a)/sx;Y=(y-b)/sy
    p=polygon(X,Y,[(-.94,.56),(-.94,-.02),(-.76,-.21),(-.56,-.74),(.55,-.74),(.77,-.22),(.95,-.05),(.95,.56),(.80,.56),(.80,.82),(.49,.82),(.49,.56),(-.50,.56),(-.50,.82),(-.80,.82),(-.80,.56)])
    return p & ~polygon(X,Y,[(-.60,-.17),(-.44,-.57),(.43,-.57),(.62,-.17)])

def lorryfront(x,y,a,b,sx,sy):
    X=(x-a)/sx;Y=(y-b)/sy
    return ((abs(X)<.96)&(Y>-.97)&(Y<.74))|((abs(X)>.52)&(abs(X)<.88)&(Y<.97)&(Y>.5))

def vehicles(k,x,y,style):
    truck=k in (10,42);end=k in (41,42)
    gap=.30;top=-.13 if truck else 0;sy=.32 if truck else .22
    left=lorryfront(x,y,-gap,top,.265,sy) if truck else car(x,y,-gap,-.01,.265,sy)
    right=car(x,y,gap,.07 if truck else -.01,.262,.22)
    if style: right=car(x,y,gap,.12 if truck else .03,.24,.20)
    return (left*(.53 if end else .64)+right*(.68 if end else 1.)).clip(0,1)

def sideTruck(x,y):
    return ((x>-.16)&(x<.64)&(y>-.32)&(y<.11)) | polygon(x,y,[(-.63,.24),(-.63,-.03),(-.46,-.21),(-.22,-.21),(-.22,.115),(.61,.115),(.61,.21),(-.19,.21),(-.2,.25)]) | ellipse(x,y,-.44,.22,.135,.135) | ellipse(x,y,.39,.19,.135,.135)

def arrows(k,x,y,style):
    if k in (34,37,39): x=-x
    t=1 if style==0 else .83
    if k in (38,39):
        u=(x+y)/math.sqrt(2);v=(y-x)/math.sqrt(2)
        return polygon(u,v,[(-.66,-.105*t),(.24,-.105*t),(-.06,-.42*t),(.20,-.42*t),(.70,0),(.20,.42*t),(-.06,.42*t),(.24,.105*t),(-.66,.105*t)])
    if k in (33,34):
        return line(x,y,bezier((-.39,.63),((-.39,.06),(-.44,-.34),(.20,-.34)),((.26,-.34),(.32,-.34),(.40,-.34))),.17*t) | polygon(x,y,[(.14,-.70),(.64,-.34),(.14,.04),(.14,-.16),(.32,-.34),(.14,-.51)])
    def straight(a):
        return polygon(x-a,y,[(-.093*t,.66),(.093*t,.66),(.093*t,-.27),(.36*t,-.02),(.36*t,-.27),(0,-.73),(-.36*t,-.27),(-.36*t,-.02),(-.093*t,-.27)])
    if k==35:return straight(0)
    if k in (36,37):
        return straight(-.23) | polygon(x,y,[(-.24,.12),(.24,.12),(.03,-.085),(.27,-.085),(.65,.21),(.27,.52),(.03,.52),(.24,.30),(-.24,.30)])
    out=np.zeros_like(x,dtype=bool)
    for j in range(3):
        rot=(j*120+(15 if style else 0))*math.pi/180
        # Three anticlockwise arrows along a circular line.
        X=x*np.cos(rot)-y*np.sin(rot);Y=x*np.sin(rot)+y*np.cos(rot)
        theta=np.arctan2(Y,X);radius=np.hypot(X,Y)
        out |= (radius<.62)&(radius>.47)&(theta>-.96)&(theta<.31)
        a=-.95;u=X*np.cos(a)+Y*np.sin(a)-.545;v=-X*np.sin(a)+Y*np.cos(a)
        out |= polygon(u,v,[(-.23,.12),(0,-.20),(.23,.12),(.19,.21),(0,.05),(-.19,.21)])
    return out

def warning(k,x,y,style):
    def p(points):return polygon(x,y,points)
    def l(points,w):return line(x,y,points,w)
    def c(a,b,r):return ellipse(x,y,a,b,r,r)
    if k==11:
        if style:return p([(-.142,.45),(.142,.45),(.142,.067),(.31,.067),(.31,-.041),(.142,-.041),(.142,-.19),(0,-.42),(-.142,-.19),(-.142,-.041),(-.31,-.041),(-.31,.067),(-.142,.067)])
        return p([(-.13,.50),(.13,.50),(.13,.05),(.36,.05),(.13,-.06),(.13,-.20),(0,-.46),(-.13,-.20),(-.13,-.06),(-.36,.05),(-.13,.05)])
    if k==18:return p([(-.084,-.45),(.084,-.45),(.063,.14),(.033,.29),(-.035,.29),(-.063,.14)])|c(0,.46,.077)
    if k in (19,20,21):
        if k==20:x=-x
        if k!=21:
            a=bezier((.20,.53),((.23,.19),(.21,-.02),(-.12,-.12)))
            return line(x,y,a,.135) | polygon(x,y,[(-.04,-.28),(-.29,-.12),(-.05,.035)])
        a=bezier((.19,.54),((.22,.17),(.11,.03),(-.08,.10)),((-.28,.16),(-.20,-.08),(-.13,-.21)))
        return line(x,y,a,.13)|polygon(x,y,[(-.29,-.15),(-.10,-.32),(-.04,-.07)])
    if k==22:
        profile=[(-.59,.52),(-.49,.48),(-.41,.33),(-.33,.27),(-.25,.27),(-.15,.38),(-.09,.44),(0,.46),(.10,.40),(.18,.30),(.27,.26),(.35,.26),(.45,.38),(.51,.47),(.60,.52),(.60,.55),(-.59,.55)]
        return p(profile)
    if k==23:
        # Skidding car over sinuous tracks.
        ang=-.29;X=x*math.cos(ang)+y*math.sin(ang);Y=-x*math.sin(ang)+y*math.cos(ang)
        body=car(X,Y,-.01,-.04,.22,.185)
        if style:
            body=polygon(X,Y,[(-.34,-.02),(-.31,-.12),(-.21,-.14),(-.13,-.24),(.09,-.24),(.20,-.14),(.32,-.12),(.34,-.01),(.29,.055),(-.27,.055)]) | ellipse(X,Y,-.20,.055,.065,.073) | ellipse(X,Y,.21,.055,.065,.073)
        for a in [-.14,.14]: body |=l(bezier((a,.10),((a+.34,.21),(a-.40,.29),(a-.16,.39)),((a+.24,.50),(a+.05,.52),(a-.03,.55))),.043)
        return body
    if k==24:
        return l([(-.22,-.28),(-.22,.54)],.096) | l([(.055,-.28),(.055,-.08),(.22,.16),(.22,.54)],.096)
    if k==25:
        x=-x
        mound=p([(-.59,.56),(-.46,.42),(-.36,.39),(-.26,.31),(-.14,.38),(-.03,.52),(.02,.56)])
        return mound | c(.125,-.218,.085)|l([(.1,-.11),(.24,-.04),(.18,.12)],.14)|l([(.19,.11),(.28,.27),(.32,.52)],.083)|l([(.17,.12),(.08,.32),(-.01,.53)],.077)|l([(.10,-.075),(.025,.08),(-.10,.14)],.067)|l([(-.26,.42),(.20,-.04)],.032)
    if k==26:
        q=np.zeros_like(x)
        for a,v in [(-.30,.86),(0,.40),(.30,.70)]:q+=ellipse(x,y,0,a,.136,.136)*v
        if style:q=np.maximum(q,.40*((abs(x)<.116)&(y>-.18)&(y<.38)))
        return q
    if k==27:
        if style:x=-x
        return c(.075,-.26,.078)|l([(.045,-.135),(-.057,.10)],.115)|l([(.055,-.125),(.09,.023),(.233,.121)],.061)|l([(-.004,-.12),(-.099,-.02),(-.183,.092)],.055)|l([(-.052,.10),(.015,.28),(.20,.49)],.088)|l([(-.055,.10),(-.15,.30),(-.22,.5)],.084)
    if k==28:
        adult=c(.085,-.21,.083)|l([(.054,-.07),(.04,.18)],.127)|l([(.04,.16),(.14,.33),(.24,.50)],.083)|l([(.01,.16),(-.04,.38),(-.16,.49)],.072)|l([(.04,-.04),(.18,.09),(.25,.10)],.064)|l([(.03,-.02),(-.07,.045),(-.17,.00)],.053)
        child=c(-.24,-.035,.065)|p([(-.25,.03),(-.18,.04),(-.12,.30),(-.34,.30)])|l([(-.245,.27),(-.30,.38),(-.40,.48)],.068)|l([(-.19,.27),(-.17,.48)],.067)|l([(-.2,.06),(-.13,.00),(-.07,.04)],.046)
        return adult|child
    if k==29:
        if style:x=-x
        wheels=(c(-.29,.355,.198)&~c(-.29,.355,.146))|(c(.29,.355,.198)&~c(.29,.355,.146))
        if style: return wheels|l([(-.29,.355),(-.12,.025),(.025,.355),(-.29,.355),(.19,.073),(.29,.355)],.052)|l([(-.12,.025),(-.17,-.055),(-.24,-.055)],.055)|l([(.19,.073),(.12,-.12),(.24,-.12)],.055)
        return wheels|l([(-.29,.36),(-.12,.055),(.05,.35),(-.29,.36),(.20,.09),(.29,.36)],.042)|l([(-.12,.055),(-.18,-.02)],.044)|l([(.18,.10),(.13,-.03),(.22,-.05)],.04)|c(.04,-.25,.071)|l([(.00,-.14),(-.14,.00),(.02,.12),(-.01,.32)],.076)|l([(.015,-.12),(.11,-.026),(.18,-.006)],.05)
    if k==30:
        y=y+.17
        out=np.zeros_like(x,dtype=bool);size=.40
        for j in range(6):
            angle=(j*60-90+(30 if style else 0))*math.pi/180;u=x*math.cos(angle)+(y-.17)*math.sin(angle);v=-x*math.sin(angle)+(y-.17)*math.cos(angle)
            out |=line(u,v,[(0,0),(size,0)],.057)
            for a in (.19,.29):
                out |=line(u,v,[(a,0),(a+.088,.088)],.046)|line(u,v,[(a,0),(a+.088,-.088)],.046)
        return out
    if k==31:
        body=p([(-.23,.035),(-.10,.06),(.09,.13),(.27,.145),(.33,.21),(.28,.30),(.04,.26),(-.14,.20),(-.22,.15),(-.26,.00),(-.37,-.04),(-.40,-.105),(-.25,-.105),(-.20,-.18),(-.18,-.09)])
        return body|l([(.24,.24),(.34,.35),(.49,.44)],.048)|l([(.24,.23),(.24,.37),(.37,.51)],.044)|l([(-.10,.16),(-.24,.3),(-.36,.25)],.042)|l([(-.11,.17),(-.15,.32),(-.30,.39)],.043)|l([(-.22,-.10),(-.16,-.24),(-.25,-.32)],.028)

classfam=[0]*6+[4,0,0,0,0,1,7,2,5,0,0,6]+[1]*14+[4]+[3]*8+[4,4]

def icon(k,x,y,style=0,weight=.225):
    if k<=8:
        num=(20,30,50,60,70,80,80,100,120)[k];out=digitgroup(x,y,num,style,weight).astype(float)
        if k==6:out*=.65
    elif k in (9,10,41,42):out=vehicles(k,x,y,style)
    elif k==16:out=sideTruck(x,y)
    elif k in (11,*range(18,32)):
        if k==29 and style:y=y+.13
        if k in (19,20,21,23,25,27,28,29,31):y=y+.12
        if k==22:y=y+.14
        out=warning(k,x,y,style)
    elif 33<=k<=40:out=arrows(k,x,y,style)
    else:out=np.zeros_like(x)
    if k in (6,32,41,42):
        u=(x+y*(.8 if style else 1))/math.sqrt(2)
        stripes=(abs(u)<(.19 if style else .15)) & ((np.mod(u+.015,.049)<.032) if style==0 else True)
        out=np.maximum(out,.95*stripes)
    return out.astype(float)

def stoptext(x,y):
    a=.175;b=.32;txt=np.zeros_like(x,dtype=bool)
    for i,c in enumerate('STOP'):
        X=x-(i-1.5)*.40
        if c=='S':
            pts=bezier((.83,-.63),((.43,-1.09),(-.83,-1.00),(-.80,-.48)),((-.84,-.12),(-.53,-.09),(.02,.02)),((1.21,.19),(.94,.79),(.34,.87)),((-.28,1.04),(-.70,.80),(-.87,.66)))
            txt |= line(X,y,[(u*a,v*b) for u,v in pts],.075)
        if c=='T':txt |=line(X,y,[(-a,-b),(a,-b)],.075)|line(X,y,[(0,-b),(0,b)],.08)
        if c=='O':txt |=((abs(X)/a)**3+(abs(y)/b)**3<1)&((abs(X)/(a-.072))**3+(abs(y)/(b-.072))**3>1)
        if c=='P':txt |=line(X,y,[(-a*.75,b),(-a*.75,-b),(.01,-b)],.078)|line(X,y,bezier((.01,-b),((a*1.2,-b),(a*1.2,.01),(.015,.01))),.080)|line(X,y,[(.015,.01),(-a*.68,.01)],.075)
    return txt.astype(float)

def signpaint(k,x,y,style):
    f=classfam[k];name=families[f];d=borderdist(name,x,y);ins=d<1
    l=np.zeros_like(x);c=np.zeros_like(x)
    if f in (0,1,2,4):
        d2=d
        if f in (1,2):
            yy=y if f==1 else -y;d2=np.maximum((yy-1/3)/(2/3),(2*abs(x)-yy+1/3)/(4/3))
        w=(d2<( (.77 if style==0 else .82) if f==0 else (.75 if style==0 else .79))) if f!=4 else ins
        ink=icon(k,x,y-(1/3 if f==1 else 0),style,.23)
        l=np.where(w,.96-.84*ink,.24);c=np.where(w,0.,1.)
        if f==0 and k in (9,10):c=np.maximum(c,(x<0)*ink*1.4)
    elif f==7:
        l=np.where(d<.71,.73,1);c=(d<.71)*1.
    else:
        if f==5:ink=stoptext(x,y)
        elif f==6:ink=((abs(y)<.155)&(abs(x)<.82)).astype(float)
        else:ink=icon(k,x,y,style)
        ink=np.maximum(ink,(d>(.93 if f==5 else .963))*1.)
        l=.22+.74*ink;c=1.-ink
    return l,c,ins

def digbank():
    n=24;v=((np.arange(n*3)+.5)/(n*3)*2-1);x,y=np.meshgrid(v*1.25,v*1.18);fs=[];keys=[]
    for k in (0,1,2,3,5,6,7,8):
        offset=len(fs)
        for thick in (.185,.255,.31):
            for style in (0,1):
                for w,h in ((1,1),(.87,1.05),(1.13,.99),(1.02,.88)):
                    for skew in (0,-.15,.15):
                        b=digit(k,(x-skew*y)/w,y/h,thick,style).astype(float).reshape(n,3,n,3).mean((1,3));b-=b.mean()
                        fs.append((b/(np.sqrt(np.sum(b*b))+1e-8)).ravel())
        keys.append((k,len(fs)-offset))
    return dict(digits=np.array(fs,np.float32),dkey=np.array(keys,np.int32))

def appearance():
    n=24;ss=3;g=((np.arange(n*ss)+.5)/(n*ss)*2-1)*1.01;xx,yy=np.meshgrid(g,g)
    v=((np.arange(n)+.5)/n*2-1)*1.01;x,y=np.meshgrid(v,v);out={}
    for f in range(8):
        m=np.clip((1.008-borderdist(families[f],x,y))/.067,0,1);ms=m.sum();out['am'+str(f)]=m.astype(np.float32)
        def feat(v):
            v=v-(v*m).sum()/ms;return (v*m/(np.sqrt((m*v*v).sum())+1e-7)).ravel()
        lu=[];co=[];keys=[]
        def avg(v):return v.reshape(n,ss,n,ss).mean((1,3))
        for k in range(43):
            if classfam[k]!=f:continue
            count=0
            for style in range(2):
                # Pre-render once so the alignment bank only does small array operations.
                v0=(np.arange(208)+.5)/80-1.30;gx,gy=np.meshgrid(v0,v0);L,C,I=signpaint(k,gx,gy,style)
                for s in (.95,1,1.065):
                    for dx,dy in ((0,0),(-.05,0),(.05,0),(0,-.05),(0,.05)):
                        ix=np.clip(np.floor(((xx-dx)/s+1.3)*80).astype(int),0,207);iy=np.clip(np.floor(((yy-dy)/s+1.3)*80).astype(int),0,207);ins=I[iy,ix]
                        # Extrapolate the canonical border to the support boundary; do
                        # not predict the colour of a scene behind a sign.
                        lu.append(feat(avg(np.where(ins,L[iy,ix],.50))));co.append(feat(avg(np.where(ins,C[iy,ix],.45))));count+=1
            keys.append((k,count))
        out['al'+str(f)]=np.array(lu,np.float32);out['ac'+str(f)]=np.array(co,np.float32);out['ak'+str(f)]=np.array(keys,np.int32)
    return out

# Inner template features: compare a rectified patch at several plausible motif sizes,
# with a small allowance for displaced text, shear, and the stroke weight of the type.
def igrid(f,n):
    v=(np.arange(n)+.5)*2/n-1
    x,y=np.meshgrid(v,v)
    if f==1:return x*.70,y*.62+.17
    if f==3 or f==4:return x*.83,y*.83
    return x*.71,y*.58

def maskweight(f,x,y):
    if f==1:
        d=np.maximum((y-1/3)/(2/3),(2*abs(x)-y+1/3)/(4/3));return np.clip((.81-d)/.20,0,1)
    else:
        radius=.79 if f==0 else .90
        return np.clip((radius-np.hypot(x,y))/.12,0,1)

def gradfeat(a,m):
    # Eight-by-eight signed gradient histograms, with overlapping filters
    # followed by spatial pooling.  The descriptor is computed per template.
    v=np.pad(a,1,mode='edge');gx=(v[1:-1,2:]-v[1:-1,:-2])*m*m;gy=(v[2:,1:-1]-v[:-2,1:-1])*m*m
    channels=[gx,gy,(gx+gy)*.7071,(gx-gy)*.7071]
    b=np.stack([np.maximum(u*t,0).reshape(8,3,8,3).mean((1,3)) for u in channels for t in (-1,1)],axis=-1)
    v=np.pad(b,((1,1),(1,1),(0,0)));b=.60*b+.10*(v[1:-1,2:]+v[1:-1,:-2]+v[2:,1:-1]+v[:-2,1:-1]);e=np.sum(b*b,axis=2,keepdims=True);b=b/(e+.28*e.mean()+1e-12)**.25
    return (b/(np.sqrt(np.sum(b*b))+1e-8)).ravel()

def inners():
    out={};n=24
    for f in (0,1,3,4):
        xx,yy=igrid(f,n*3);x,y=igrid(f,n);m=maskweight(f,x,y);ms=m.sum();features=[];keys=[];grads=[]
        for k in range(43):
            if classfam[k]!=f or k==15:continue
            base=[]
            for style in (0,1):
                for thick in ((.19,.26) if k<=8 else (.225,)):
                    # Render once at high resolution, then resample multiple transforms.
                    g=(np.arange(192)+.5)/80-1.2;gx,gy=np.meshgrid(g,g);bmp=icon(k,gx,gy,style,thick)
                    sizes=((1,1),(.86,.85),(1.14,1.13),(.84,1.03),(1.11,.91))
                    if f==1:sizes+=((1.25,1.25),(1.12,1.30),(.80,.80),(1.30,1.02))
                    if f==0:sizes+=((1,.80),(.77,1.10),(1.26,1),(1,1.25))
                    positions=[(u,v) for u in (-.066,0,.066) for v in (-.08,0,.08)]
                    if f==1:positions.extend([(0,-.16),(0,.16)])
                    for sx,sy in sizes:
                        for dx,dy in positions:
                            for skew in ((-.16,0,.16) if dx==0 else (0,)):
                                anchor=1/3 if f==1 else 0
                                ix=np.clip(np.floor(((xx-dx-skew*(yy-anchor))/sx+1.2)*80).astype(int),0,191);iy=np.clip(np.floor((((yy-dy-anchor)/sy)+1.2)*80).astype(int),0,191)
                                a=bmp[iy,ix].reshape(n,3,n,3).mean((1,3)).astype(np.float32)
                                # Mild symmetric smoothing for antialiasing mismatch.
                                u=np.pad(a,1,mode='edge');a=.64*a+.09*(u[1:-1,:-2]+u[1:-1,2:]+u[:-2,1:-1]+u[2:,1:-1]);a=a-(a*m).sum()/ms
                                base.append((a*m/(np.sqrt((a*a*m).sum())+1e-8)).ravel());grads.append(gradfeat(a,m))
            features.extend(base);keys.append((k,len(base)))
        out['hg'+str(f)]=np.array(grads,np.float32)
        out['in'+str(f)]=np.ascontiguousarray(np.array(features,np.float32));out['keys'+str(f)]=np.array(keys,np.int32)
    return out

solution=r'''import numpy as np
import math

poses=CONSTANTS['pose']
outerL=CONSTANTS['light']
outerC=CONSTANTS['color']
famlist=[0]*6+[4,0,0,0,0,1,7,2,5,0,0,6]+[1]*14+[4]+[3]*8+[4,4]

def normall(a):
    a=a.reshape(len(a),-1)
    a=a-a.mean(axis=1,keepdims=True)
    return a/(np.sqrt(np.sum(a*a,axis=1,keepdims=True))+.0001)

def chroma(a):
    # Channel ratios reduce sensitivity to irradiance. Subtract a per-image
    # neutral estimate to adapt the opponent channels to the illuminant.
    r,g,b=a[:,0],a[:,1],a[:,2]
    red=r-.70*g-.30*b;blue=b-.78*r-.22*g;yellow=(r+g)*.50-b
    v=(r+g+b)/3+.023
    red,blue,yellow=red/v,blue/v,yellow/v
    def normc(c):
        baseline=np.quantile(c,.25,axis=(1,2)).astype(np.float32)[:,None,None]
        return np.clip((c-.65*baseline-.035)/.55,0,1.5)
    return normc(red),normc(blue),normc(yellow)

def grids(f,n):
    v=(np.arange(n,dtype=np.float32)+.5)*2/n-1;x,y=np.meshgrid(v,v)
    if f==1: x,y=x*.70,y*.62+.17
    elif f==3 or f==4: x,y=x*.83,y*.83
    else:x,y=x*.71,y*.58
    if f==1:m=np.clip((.81-np.maximum((y-1/3)/(2/3),(2*abs(x)-y+1/3)/(4/3)))/.20,0,1)
    else:m=np.clip(((.79 if f==0 else .90)-np.sqrt(x*x+y*y))/.12,0,1)
    return x,y,m

insidegrids=[grids(f,24) for f in range(8)]

def sample(img,x,y):
    x=np.clip(x,0,30.9999);y=np.clip(y,0,30.9999);ix=x.astype(np.int32);iy=y.astype(np.int32);u=x-ix;v=y-iy
    k=np.arange(len(img)).reshape((-1,)+(1,)*(x.ndim-1))
    return ((1-u)*((1-v)*img[k,iy,ix]+v*img[k,iy+1,ix])+u*((1-v)*img[k,iy,ix+1]+v*img[k,iy+1,ix+1])).astype(np.float32)

def boundary(f,theta):
    x=np.cos(theta);y=np.sin(theta)
    if f==1 or f==2:
        s=1 if f==1 else -1
        d=np.maximum(s*y,2*abs(x)-s*y)
        # clipped vertices; use separate distance to the base/oblique edges below
        return x/d,y/d
    if f==7:d=abs(x)+abs(y)
    elif f==5:d=np.maximum(np.maximum(abs(x),abs(y)),(abs(x)+abs(y))/math.sqrt(2))
    else:d=np.sqrt(x*x+y*y)
    return x/d,y/d

# Uniform arc-length sampling, afocal contrast across the INSIDE of the coloured
# border. This is much less sensitive to the background than the exterior edge.
def rimpoints(f):
    if f in (1,2):
        s=1 if f==1 else -1
        pts=[(0,-1*s),(1,1*s),(-1,1*s)]; groups=[]
        for j in range(3):
            a,b=pts[j];c,d=pts[(j+1)%3];t=np.linspace(.09,.91,12)
            groups.extend(zip(a+(c-a)*t,b+(d-b)*t))
        xy=np.array(groups);x,y=xy.T
    else:
        x,y=boundary(f,(np.arange(40)+.5)*2*math.pi/40)
    # In a triangle, scaling must be about its incenter rather than its box center.
    y0=(.32 if f==1 else -.32 if f==2 else 0)
    return x.astype(np.float32), (y-y0).astype(np.float32), y0
rimgrids=[rimpoints(f) for f in range(8)]

def contrast(v):
    return np.mean(np.clip(v,-1,1),axis=-1)

def geomvalue(img,col,f,pose):
    cx,cy,w,h,sk,sl=[pose[:,:,j,None] for j in range(6)];x,y,y0=rimgrids[f]
    def at(z,a):
        X=x*a;Y=y*a+y0
        return sample(z,cx+w*(X+sk*Y),cy+h*(Y+sl*X))
    if f in (0,1,2):
        # all signs have an approximately uniform coloured rim
        inner=.73 if f==0 else .65
        mid=.89 if f==0 else .865
        lr=at(img,mid);li=at(img,inner);cr=at(col,mid);ci=at(col,inner)
        diff=contrast((li-lr)*1.4)+.53*contrast(cr-ci)
        return diff+.13*contrast(cr)-.12*np.std(np.clip(li-lr,-.7,.7),axis=-1)
    if f==6:
        # Follow both long edges of the white bar, not just the outside disc.
        xx=np.linspace(-.64,.64,15,dtype=np.float32)
        def band(a,b):return sample(a,cx+w*(xx+sk*b),cy+h*(b+sl*xx))
        li=band(img,0);lu=band(img,-.26);ld=band(img,.26);cc=band(col,.26)+band(col,-.26)-2*band(col,0)
        return .70*contrast(li-.5*(lu+ld))+.20*contrast(cc)+.25*contrast(at(col,.86)-at(col,1.08))
    if f==4:
        li=at(img,.86);le=at(img,1.075)
        return .65*contrast((li-le)*1.6)+.27*np.mean(np.minimum(1,abs(li-le)*1.8),axis=-1)+.20*contrast(li)
    if f==7:
        delta=at(col,.57)-at(col,.89);li=at(img,.87);le=at(img,1.10)
        return .75*contrast(delta)+.4*contrast(li-le)+.22*contrast(li-at(img,.57))
    # Coloured disc or octagon with white narrow edge; outside may be anything.
    lc=at(img,.85);cc=at(col,.85);ce=at(col,1.09);le=at(img,1.08)
    wh=at(img,.969)-lc
    return .62*contrast(cc-ce)+.38*contrast(le-lc)+.36*contrast(wh)+.24*contrast(cc)

def refiner(img,col,f,p):
    p=np.pad(p,((0,0),(0,2)));best=geomvalue(img,col,f,p[:,None,:])[:,0]
    base=p.copy()
    for step in (1.3,.68):
        for dims in ((0,1),(2,3),(4,5),(0,1),(2,3),(4,5)):
            a,b=dims;opts=np.array([(u,v) for u in (-1,0,1) for v in (-1,0,1)],np.float32)*step
            cand=np.repeat(p[:,None,:],9,axis=1)
            mult=.075 if a==4 else 1
            cand[:,:,a]+=opts[:,0]*mult;cand[:,:,b]+=opts[:,1]*mult
            score=geomvalue(img,col,f,cand)
            score-=.0025*((cand[:,:,0]-15.5)**2+(cand[:,:,1]-15.5)**2)
            score-=.30*(cand[:,:,4]**2+cand[:,:,5]**2)+.002*((cand[:,:,2]-base[:,None,2])**2+(cand[:,:,3]-base[:,None,3])**2)
            ii=np.argmax(score,axis=1);p=cand[np.arange(len(p)),ii];best=score[np.arange(len(p)),ii]
    return p,best

def crop(a,p,f):
    x,y,m=insidegrids[f];cx,cy,w,h,s,t=[p[:,j,None,None] for j in range(6)]
    return sample(a,cx+w*(x+s*y),cy+h*(y+t*x))

def histgrad(a,m):
    v=np.pad(a,((0,0),(1,1),(1,1)),mode='edge');gx=(v[:,1:-1,2:]-v[:,1:-1,:-2])*m*m;gy=(v[:,2:,1:-1]-v[:,:-2,1:-1])*m*m
    channels=[gx,gy,(gx+gy)*.7071,(gx-gy)*.7071];n=len(a)
    b=np.stack([np.maximum(u*t,0).reshape(n,8,3,8,3).mean((2,4)) for u in channels for t in (-1,1)],axis=-1)
    v=np.pad(b,((0,0),(1,1),(1,1),(0,0)));b=.60*b+.10*(v[:,1:-1,2:]+v[:,1:-1,:-2]+v[:,2:,1:-1]+v[:,:-2,1:-1]);e=np.sum(b*b,axis=3,keepdims=True);b=b/(e+.28*e.mean((1,2,3),keepdims=True)+1e-12)**.25
    return (b/(np.sqrt(np.sum(b*b,axis=(1,2,3),keepdims=True))+1e-8)).reshape(n,-1)

def interior(img,red,blue,f,p):
    x,y,m=insidegrids[f];ms=m.sum();v=crop(img,p,f)
    if f==3:v=v-.20*crop(blue,p,f)
    v=v*(1 if f==3 else -1)
    if f==0:v=v+.18*crop(red,p,f)
    v=v-(v*m).sum((1,2),keepdims=True)/ms
    # A bounded contrast stretch keeps highlights and deep shadow edges from
    # dominating and improves the match of saturated or partially faded symbols.
    dev=np.sqrt((v*v*m).sum((1,2),keepdims=True)/ms)+.008
    v=np.clip(v/dev,-2.6,2.8)
    v=v-(v*m).sum((1,2),keepdims=True)/ms
    v=v/np.sqrt((v*v*m).sum((1,2),keepdims=True)+.008)
    feats=CONSTANTS['in'+str(f)];keys=CONSTANTS['keys'+str(f)];q=v.reshape(len(p),-1)@feats.T
    hq=histgrad(v,m)@CONSTANTS['hg'+str(f)].T
    q=.68*q+.32*(1.50*hq-.50)
    start=0;ans=np.full((len(p),43),-3,np.float32)
    for k,n in keys:
        ans[:,k]=q[:,start:start+n].max(axis=1);start+=n
    if f==0:
        ans[:,15]=np.clip(.90-3.8*dev[:,0,0],-.15,.85)
    return ans

def checkpose(img,col,f,p):
    # Make a final local comparison by the whole-sign appearance. Border evidence
    # alone can put a thin, high-contrast rim slightly inside the sign boundary.
    n=len(p);delta=np.array([(0,0,1,1),(-.8,0,1,1),(.8,0,1,1),(0,-.8,1,1),(0,.8,1,1),(0,0,.91,.91),(0,0,1.09,1.09),(0,0,.94,1.06),(0,0,1.06,.94)],np.float32)
    cand=np.repeat(p[:,None,:],len(delta),axis=1);cand[:,:,:2]+=delta[None,:,:2];cand[:,:,2:4]*=delta[None,:,2:]
    g=geomvalue(img,col,f,cand)
    g-=.0015*((cand[:,:,0]-15.5)**2+(cand[:,:,1]-15.5)**2)
    ix=np.repeat(np.arange(n),len(delta));cm=cand.reshape(-1,6);q=apmatch(img[ix],col[ix],f,cm).max(1).reshape(n,-1)
    merit=.66*q+.34*g;best=merit.argmax(1);return cand[np.arange(n),best]

def apmatch(lum,col,f,p):
    v=(np.arange(24,dtype=np.float32)+.5)/12-1;x,y=np.meshgrid(v*1.01,v*1.01);cx,cy,w,h,s,t=[p[:,j,None,None] for j in range(6)]
    X=cx+w*(x+s*y);Y=cy+h*(y+t*x);l=sample(lum,X,Y);c=sample(col,X,Y)
    m=CONSTANTS['am'+str(f)];ms=m.sum();kc=(c*m).sum((1,2))/ms
    # A plane of lighting can cross an entire sign. Local subtraction gives a
    # second view of its structure without making a bad shadow decisive.
    def comp(a,key,floor):
        a=a-(a*m).sum((1,2),keepdims=True)/ms;sd=np.sqrt((a*a*m).sum((1,2),keepdims=True)/ms)
        a=np.clip(a/(sd+floor),-2.6,2.6);a=a-(a*m).sum((1,2),keepdims=True)/ms
        a=a/np.sqrt((a*a*m).sum((1,2),keepdims=True)+.001)
        return a.reshape(len(a),-1)@CONSTANTS[key+str(f)].T,sd[:,0,0]
    q,ld=comp(l,'al',.02)
    if f!=4:
        qc,cd=comp(c,'ac',.03);rc=np.minimum(1,cd*5)[:,None]
        q=q+.57*rc*qc
        if f in (3,5,6,7):q+=.26*np.minimum(1,kc*2)[:,None]-.13
    else:q-=.32*kc[:,None]
    ans=np.full((len(p),43),-3,np.float32);st=0
    for k,n in CONSTANTS['ak'+str(f)]:ans[:,k]=q[:,st:st+n].max(1);st+=n
    return ans

def readnumber(lum,p):
    # Read each candidate at high resolution, demanding a common type size and
    # baseline. In particular, a 120 must explain three glyphs, not just the 20.
    n=len(p);v=(np.arange(24,dtype=np.float32)+.5)/12-1;gx,gy=np.meshgrid(v*1.25,v*1.18)
    cx,cy,w,h,s,t=[p[:,j,None,None,None] for j in range(6)];out=np.zeros((n,43),np.float32)
    ds=CONSTANTS['digits'];keys=CONSTANTS['dkey'];first=[2,3,5,6,7,8]
    for length in (2,3):
        scores=[[] for j in range(8)]
        for scale in (.84,1,1.17):
            for width in ((.86,1.06) if length==2 else (.83,1.03,1.21)):
                H=.455*scale;W=(.235 if length==2 else .185)*width;gap=(.085 if length==2 else .062)*width
                A=W+.5*gap
                centers=(-A,A) if length==2 else (-1.68*W-gap,-.32*W,1.68*W+gap)
                loc=[(c,dx,dy) for c in centers for dx in (-.042,.042) for dy in (-.055,.055)]
                xp=np.array([a+b for a,b,d in loc],np.float32)[:,None,None]+W*gx;yp=np.array([d for a,b,d in loc],np.float32)[:,None,None]+H*gy
                patch=-sample(lum,cx+w*(xp+s*yp),cy+h*(yp+t*xp));patch=patch-patch.mean((2,3),keepdims=True)
                patch=patch/(np.sqrt(np.sum(patch*patch,axis=(2,3),keepdims=True))+.045)
                q=patch.reshape(-1,576)@ds.T;st=0;sc=[]
                for d,cnt in keys:
                    sc.append(q[:,st:st+cnt].max(1).reshape(n,length,4));st+=cnt
                if length==2:
                    for j,fd in enumerate(first):
                        di=(2,3,4,5,6,7)[j]
                        val=(.57*sc[di][:,0]+.43*sc[0][:,1]).max(1)
                        scores[j].append(val)
                else:
                    for j,mid in ((6,0),(7,2)):
                        val=(.31*sc[1][:,0]+.37*sc[mid][:,1]+.32*sc[0][:,2]).max(1)
                        scores[j].append(val)
        for j,arr in enumerate(scores):
            if len(arr)>0:out[:,j if length==2 else j+1]=np.max(np.stack(arr),0)
    return out

def flattenlight(u):
    v=np.pad(u,((0,0),(3,3),(3,3)),mode='edge');v=np.pad(v,((0,0),(1,0),(1,0)));v=v.cumsum(1).cumsum(2)
    smooth=(v[:,7:,7:]-v[:,:-7,7:]-v[:,7:,:-7]+v[:,:-7,:-7])/49
    return u-.73*smooth

def scorebatch(a):
    n=len(a);red,blue,yellow=chroma(a);cols=(red,red,red,blue,red,red,red,yellow)
    rgb=.27*a[:,0]+.57*a[:,1]+.16*a[:,2]
    # Compress luminance mildly, then scale from this frame's central range.
    lum=np.sqrt(rgb+.012)
    lo,hi=np.quantile(lum[:,4:28,4:28],[.08,.92],axis=(1,2)).astype(np.float32)
    lum=np.clip((lum-lo[:,None,None])/(.036+hi[:,None,None]-lo[:,None,None]),-.8,1.65)
    l=normall(lum);nc=[normall(c) for c in cols];vals=[];seeds=[]
    for f in range(8):
        g=l@outerL[f].T
        if f!=4:
            c=cols[f];strength=np.minimum(1,c.std((1,2))*4.5)[:,None]
            g+=.76*strength*(nc[f]@outerC[f].T)
        g-=.002*((poses[f,:,0]-15.5)**2+(poses[f,:,1]-15.5)**2)
        # Whole-frame correlation is biased by clutter and parking signs in the
        # margin. Rerank candidate locations by edge polarity along the perimeter.
        candidates=np.argsort(g,axis=1)[:,-28:];cand=poses[f,candidates];trial=np.pad(cand,((0,0),(0,0),(0,2)))
        edge=geomvalue(lum,cols[f],f,trial);local=np.take_along_axis(g,candidates,axis=1)+.68*edge
        j=candidates[np.arange(n),local.argmax(1)];seeds.append(poses[f,j]);vals.append(g[np.arange(n),j])
    out=np.full((n,43),-8,np.float32)
    detail=flattenlight(lum)
    for f in range(8):
        p,geom=refiner(lum,cols[f],f,seeds[f]);base=.28*vals[f]+.26*geom
        p=checkpose(lum,cols[f],f,p)
        whole=apmatch(lum,cols[f],f,p)
        if f in (0,1,3,4):
            ii=interior(.53*lum+.47*detail,red,blue,f,p)
            if f==1:
                X,Y,M=insidegrids[1];R,G,B=[crop(a[:,ch],p,1) for ch in (0,1,2)]
                opp=(R-.65*G-.35*B)/(.04+(R+G+B)/3)
                region=np.exp(-((X/.15)**2+((Y-.03)/.16)**2));low=np.exp(-((X/.15)**2+((Y-.61)/.16)**2));flank=np.exp(-(((abs(X)-.23)/.10)**2+((Y-.27)/.28)**2))
                rv=(opp*region).sum((1,2))/region.sum();gv=(opp*low).sum((1,2))/low.sum();fv=(opp*flank).sum((1,2))/flank.sum()
                val=np.minimum(rv-gv,rv-fv+.035)
                ii[:,26]+=.72*np.clip(val,-.23,.78)-.07
                ii[:,26]+=.54*np.clip(np.minimum(rv-fv,fv-gv),-.22,.45)
            if f==0:
                digits=readnumber(.45*lum+.55*detail,p)
                for j in (0,1,2,3,4,5,7,8):ii[:,j]=.48*ii[:,j]+.52*digits[:,j]
                # A blank disc lacks local edges. Uniformity alone is ambiguous
                # if a candidate pose has locked onto a patch of background.
                vv=crop(lum,p,f);X,Y,M=insidegrids[0];mask=np.hypot(X,Y)<.65
                ctr=vv[:,mask];dr=np.quantile(ctr,.88,axis=1)-np.quantile(ctr,.06,axis=1)
                ii[:,15]=np.clip(.86-2.1*dr,-.20,.86)
                # Count separate columns of ink by projecting the central band.
                # An extra narrow '1' creates a third dark group and second gap;
                # this check does not depend on the detailed curve of the '2'.
                xx=np.linspace(-.75,.75,40,dtype=np.float32);yy=np.linspace(-.30,.30,13,dtype=np.float32);xx,yy=np.meshgrid(xx,yy)
                cx,cy,w,h,s,t=[p[:,j,None,None] for j in range(6)]
                lineimg=sample(lum,cx+w*(xx+s*yy),cy+h*(yy+t*xx));bright=np.quantile(lineimg,.85,axis=(1,2)).astype(np.float32)[:,None,None]
                ink=np.maximum(bright-lineimg,0);prof=ink.mean(1);noise=.045+prof.std(1)
                # Marginal left stem: compare to the gap just to its right, and
                # require ink in the middle character as well as the final zero.
                evidence=np.full(n,-2,np.float32)
                for shift in (-2,0,2):
                    for sep in (8,10,12):
                        mid=20+shift;left=mid-sep;gap=left+sep//2
                        v1=prof[:,left-2:left+2].max(1);v2=prof[:,gap-2:gap+2].min(1);v3=prof[:,mid-2:mid+3].max(1)
                        evidence=np.maximum(evidence,np.minimum(v1-v2,v3-v2)/noise)
                factor=np.clip((evidence-.90)*.095,-.09,.10);ii[:,7]+=factor;ii[:,8]+=factor
            if f==4:
                # A cancellation is achromatic and carries a strong positive-slope slash.
                bx,by,ww=insidegrids[f];cx=crop(red,p,f);cb=crop(blue,p,f)
                cv=(ww*(cx+.65*cb)).sum((1,2))/ww.sum();base-=.70*cv
                base+=.24
            for j in range(43):
                if famlist[j]==f:out[:,j]=base+.83*ii[:,j]+.74*whole[:,j]
        else:
            j=(13 if f==2 else 14 if f==5 else 17 if f==6 else 12)
            out[:,j]=base+1.13*whole[:,j]+.12
    return out.argmax(axis=1)

def predict(x):
    # Fixed internal block size, padding the last block to keep BLAS calls identical.
    result=np.empty(len(x),np.int64);size=32
    for begin in range(0,len(x),size):
        a=x[begin:begin+size];n=len(a)
        if n<size:a=np.concatenate((a,np.repeat(a[:1],size-n,axis=0)),axis=0)
        result[begin:begin+n]=scorebatch(a)[:n]
    return result
'''

def main(dest):
    d=Path(dest);d.mkdir(parents=True,exist_ok=True)
    data=wholebank();data.update(inners());data.update(appearance());data.update(digbank());np.savez(d/'constants.npz',**data)
    (d/'solution.py').write_text(solution)

if __name__=='__main__':main(sys.argv[1])
