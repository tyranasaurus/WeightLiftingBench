# All glyphs are built from paths in a typographic unit square. Coordinates
# describe the drawing, not coordinates in a video frame.
import numpy as np
import sys
from pathlib import Path

def bez(p,a,b,q):
    p,a,b,q=map(np.asarray,(p,a,b,q)); t=np.linspace(0,1,19)[:,None]
    return (1-t)**3*p+3*t*(1-t)**2*a+3*t*t*(1-t)*b+t**3*q

class Canvas:
    def __init__(self,x,y):
        self.x=x; self.y=y; self.a=np.zeros(np.broadcast_shapes(x.shape,y.shape))
    def dot(self,x,y,rx,ry=None,v=1):
        if ry is None: ry=rx
        self.a[((self.x-x)/rx)**2+((self.y-y)/ry)**2<=1]=v
    def path(self,points,w,v=1):
        for (a,b),(c,d) in zip(points,points[1:]):
            dx=c-a; dy=d-b; t=np.clip(((self.x-a)*dx+(self.y-b)*dy)/(dx*dx+dy*dy+1e-20),0,1)
            self.a[(self.x-a-dx*t)**2+(self.y-b-dy*t)**2<=(w/2)**2]=v
    def poly(self,*points,v=1):
        inside=np.zeros(self.a.shape,bool)
        for (x,y),(u,w) in zip(points,points[1:]+points[:1]):
            if y!=w: inside^=((self.y>=y)!=(self.y>=w))&(self.x<(u-x)*(self.y-y)/(w-y)+x)
        self.a[inside]=v
    def box(self,x,y,u,w,v=1):
        self.a[(self.x>=x)&(self.x<=u)&(self.y>=y)&(self.y<=w)]=v

def digit(n,x,y,style):
    # Stroke coordinates have the same scale on both axes (cap height one).
    w=.56; s=(.105,.135,.115)[style]; x=x*w
    c=Canvas(x,y); h=s/2; a=h; b=w-h; m=w/2
    def curve(p,*pieces):
        q=[p]
        for u,v,z in pieces: q.extend(bez(q[-1],u,v,z)[1:])
        c.path(q,s)
    if n=='0':
        r=(.18,.15,.25)[style]; z=b-m
        curve((m,h),((m+z*.62,h),(b,h+r*.30),(b,h+r)),((b,.35),(b,.65),(b,1-h-r)),((b,1-h-r*.3),(m+z*.62,1-h),(m,1-h)),((m-z*.62,1-h),(a,1-h-r*.30),(a,1-h-r)),((a,.65),(a,.35),(a,h+r)),((a,h+r*.3),(m-z*.62,h),(m,h)))
    if n=='1':
        c.path([(a+.02,.25),(m+.06,h),(m+.06,1-h)],s)
        c.box(m+.06-h,1-s,m+.06+h,1)
        if style==1:c.box(a,.995-s,b,.995)
    if n=='2':
        curve((a,.255),((a,.0),(b,-.055),(b,.25)),((b,.39),(m+.07,.47),(m-.015,.59)))
        c.path([(m-.015,.59),(a,.91),(a,1-h),(b,1-h)],s)
        c.box(a-h,1-s,b+h,1)
    if n=='3':
        curve((a,.205),((.105,.005),(b,-.045),(b,.25)),((b,.39),(m+.1,.47),(m-.045,.473)),((b,.47),(b+.017,.61),(b,.745)),((b,1.045),(.085,1.025),(a,.818)))
    if n=='5':
        c.box(.055,0,b+h,s); c.path([(.09,h),(.08,.475)],s)
        curve((.08,.475),((.18,.339),(b,.33),(b,.58)),((b,.65),(b,.71),(b,.757)),((b,1.04),(.095,1.035),(a,.815)))
    if n=='6':
        if style==2:
            curve((b-.025,.13),((m+.07,-.095),(a,.10),(a,.42)),((a,.5),(a,.56),(a,.69)))
        else:
            c.path([(m+.09,.018),(a+.055,.465),(a+.005,.602)],s)
        curve((a,.675),((a,.376),(b,.343),(b,.64)),((b,.70),(b,.71),(b,.735)),((b,1.04),(a,1.045),(a,.72)),((a,.7),(a,.685),(a,.675)))
    if n=='7':
        c.box(0,0,w,s); c.path([(b,s*.85),(.15,1-h)],s)
    if n=='8':
        for cy,rx,ry in [(.251,m-h-.016,.251-h),(.733,m-h,.267-h)]:
            t=np.linspace(0,2*np.pi,61); c.path(list(zip(m+rx*np.cos(t),cy+ry*np.sin(t))),s)
    return c.a

def digits(text,x,y,style,spacing):
    trim=(.26,.12,.0)[style] if text.startswith('1') else 0
    x=x*(len(text)+(len(text)-1)*spacing-trim)
    out=np.zeros(np.broadcast_shapes(x.shape,y.shape))
    for j,n in enumerate(text):out=np.maximum(out,digit(n,x-j*(1+spacing)+(trim if j else 0),y,style))
    return out

def car(c,x,y,w,h,v=1):
    def pt(a,b):return (x+a*w,y+b*h)
    c.poly(*[pt(a,b) for a,b in [(0,.46),(.1,.37),(.25,.08),(.75,.08),(.9,.37),(1,.46),(1,.84),(0,.84)]],v=v)
    c.box(*pt(.09,.77),*pt(.23,1),v);c.box(*pt(.77,.77),*pt(.91,1),v)

def person(c,x,y,w,h,rev=False):
    def p(a,b):return (x+(1-a if rev else a)*w,y+b*h)
    c.dot(*p(.56,.095),.095*h)
    for pts,sz in [([(.55,.25),(.46,.52)],.147),([(.5,.30),(.34,.45),(.15,.53)],.070),([(.53,.29),(.71,.44),(.86,.49)],.072),([(.46,.52),(.29,.73),(.10,.98)],.096),([(.48,.53),(.57,.73),(.87,.95)],.102)]:
        c.path([p(a,b) for a,b in pts],sz*h)

def glyph(k,x,y,style):
    c=Canvas(x,y); st=style%3
    if k in [0,1,2,3,4,5,7,8]:
        return digits(['20','30','50','60','70','80','','100','120'][k],x,y,st,(.17,.23,.12)[style//3%3])
    if k in [9,10,41,42]:
        truck=k in [10,42]; end=k>=41
        if truck:
            c.box(.005,.025,.42,.81,.85 if end else .76);c.box(.035,.78,.395,1,.85 if end else .76)
            car(c,.49,.45,.50,.53,1)
        else:
            car(c,0,.14,.46,.85,.85 if end else .76);car(c,.52,.115,.48,.85,1)
        # Red paint's gray-level contrast changes dramatically with light.
        if style==1 and not end:c.a[x<.48]*=.40
    if k==16:
        c.box(.33,.0,.99,.74);c.poly((.025,.48),(.13,.23),(.31,.23),(.31,.78),(.015,.78));c.box(.11,.70,.93,.84)
        c.poly((.09,.48),(.16,.29),(.25,.29),(.25,.48),v=0)
        c.dot(.2,.82,.083,.17); c.dot(.76,.82,.083,.17)
    if k==11:
        c.poly((.35,1),(.35,.66),(0,.66),(0,.52),(.35,.52),(.35,.31),(.17,.31),(.50,0),(.83,.31),(.65,.31),(.65,.52),(1,.52),(1,.66),(.65,.66),(.65,1))
    if k==18:
        c.poly((.02,.09),(.18,.015),(.77,.015),(.98,.09),(.67,.73),(.33,.73)); c.dot(.5,.9,.48,.095)
    if k in (19,20,21):
        if k==20:c=Canvas(1-x,y)
        if k==21:
            pts=[(.79,.99),(.79,.62),(.26,.44),(.26,.02)]
            c.path(pts,.255); c.box(.66,.79,.92,1); c.box(.13,0,.39,.18)
        else:
            q=bez((.83,.92),(.83,.36),(.76,.18),(.13,.17)); c.path(q,.29); c.box(.685,.75,.975,1);c.box(.02,.025,.2,.315)
    if k==22:
        c.dot(.29,.98,.29,.88);c.dot(.72,.99,.28,.89);c.box(.01,.84,.99,1);c.a[y>1]=0
    if k==23:
        # A skidding car, with two winding tyre tracks below the wheels.
        u=(x-.50)*1.35+.5; v=y+(x-.5)*.18
        b=Canvas(u,v);car(b,.08,.02,.84,.43);c.a=b.a
        for d in [-.20,.19]:
            q=bez((.5+d,.40),(.75+d,.55),(.05+d,.61),(.41+d,.77));q2=bez(q[-1],(.75+d,.89),(.49+d,.92),(.34+d,1.0));c.path(list(q)+list(q2),.072)
    if k==24:
        c.path([(.11,.035),(.11,.98)],.13);q=bez((.60,.10),(.6,.47),(.90,.4),(.90,.78));c.path([(.60,.01)]+list(q)+[(.90,.98)],.14)
    if k==25:
        c.poly((.48,.98),(.61,.77),(.74,.66),(.81,.69),(.99,.98)); c.dot(.52,.105,.071)
        c.path([(.43,.23),(.32,.41),(.29,.55)],.14);c.path([(.28,.52),(.18,.68),(.07,.97)],.09);c.path([(.30,.52),(.38,.68),(.34,.97)],.09);c.path([(.44,.25),(.50,.44),(.42,.53)],.073);c.path([(.4,.3),(.27,.4),(.19,.42)],.066);c.path([(.16,.37),(.74,.83)],.034)
    if k==26:
        for j,strength in enumerate((.91,.63,.82)):
            c.dot(.5,.165+j*.33,.47,.157,strength)
    if k==27: person(c,.02,.01,.96,.96,st==2)
    if k==28:
        person(c,.36,.01,.63,.91,True);person(c,.01,.32,.56,.64,True)
    if k==29:
        for a in [.23,.77]:
            c.dot(a,.73,.225,.255);c.dot(a,.73,.161,.186,0)
        c.path([(.22,.72),(.39,.38),(.51,.71),(.22,.72),(.56,.47),(.66,.40),(.77,.72)],.046);c.dot(.61,.10,.08);c.path([(.54,.23),(.41,.38),(.53,.46),(.49,.66)],.084);c.path([(.55,.24),(.59,.35),(.70,.36)],.055)
    if k==30:
        for i in range(6):
            th=np.pi/2+i*np.pi/3;d=np.array([np.cos(th),np.sin(th)]);n=np.array([-d[1],d[0]]);o=np.array([.5,.5]);c.path([o,o+d*.48],.062)
            for sz in [.26,.38]:
                for sg in [-1,1]:c.path([o+d*(sz-.09),o+d*sz+sg*n*.105],.051)
    if k==31:
        c.poly((.11,.45),(.21,.33),(.32,.29),(.35,.07),(.43,.03),(.43,.25),(.56,.4),(.82,.49),(.9,.36),(.91,.58),(.81,.66),(.52,.59),(.42,.54),(.34,.40),(.23,.47));c.path([(.48,.49),(.29,.66),(.07,.70)],.064);c.path([(.49,.55),(.39,.72),(.28,.75)],.058);c.path([(.79,.62),(.87,.78),(.97,.91)],.062);c.path([(.76,.61),(.75,.81),(.87,.96)],.056)
    if k in (33,34,36,37):
        if k in (34,37):c=Canvas(1-x,y)
        cy=.28 if k in (33,34) else .59
        stem=.15 if k in (33,34) else .2; th=.14 if st==0 else .115
        q=bez((stem,cy+.42),(stem,cy+.09),(stem+.12,cy),(stem+.40,cy));c.path(list(q)+[(.76,cy)],th);c.box(stem-th/2,cy+.4,stem+th/2,1)
        c.poly((.59,cy-.24),(.71,cy-.24),(.995,cy),(.71,cy+.24),(.59,cy+.24),(.72,cy+th/2),(.72,cy-th/2))
        if k in (36,37):
            c.box(stem-th/2,.2,stem+th/2,1);c.poly((stem,0),(stem+.21,.32),(stem+.17,.36),(stem,.23),(stem-.19,.37),(stem-.21,.32))
    if k==35:
        c.box(.5-.10,.225,.5+.10,1); c.poly((.5,0),(1,.37),(.93,.42),(.5,.265),(.07,.42),(0,.37))
    if k in (38,39):
        if k==39:c=Canvas(1-x,y)
        th=.10 if st==0 else .075;c.poly((0,th*1.5),(th*1.5,0),(.79,.63),(.74,.28),(.85,.28),(1,1),(.27,.87),(.27,.76),(.65,.79))
    if k==40:
        for j in range(3):
            t=np.linspace(.15,1.82,26)+j*2*np.pi/3;pts=np.stack((.5+.37*np.cos(t),.5-.37*np.sin(t)),1);c.path(pts,.09)
            p=pts[-1];f=np.array([-np.sin(t[-1]),-np.cos(t[-1])]);n=np.array([f[1],-f[0]]);c.poly(p+.10*f,p-.13*f+.15*n,p-.13*f-.15*n)
    if k==14:
        # Four narrow uppercase sans serif letters.
        for j,ch in enumerate('STOP'):
            u=(x-j*.254)*4.4;c2=Canvas(u,y)
            if ch=='S':
                q=bez((.88,.21),(.85,-.045),(.08,-.01),(.1,.26));q2=bez(q[-1],(.1,.49),(.90,.46),(.9,.73));q3=bez(q2[-1],(.94,1.01),(.12,1.07),(.07,.79));c2.path(list(q)+list(q2)+list(q3),.155)
            if ch=='T':c2.box(0,.02,1,.14);c2.box(.40,.03,.61,.98)
            if ch=='O':
                c2.dot(.50,.5,.50,.51);c2.dot(.50,.5,.29,.39,0)
            if ch=='P':
                c2.dot(.41,.30,.55,.29);c2.dot(.44,.30,.3,.17,0);c2.box(.04,.03,.24,.98)
            c.a=np.maximum(c.a,c2.a)
    if k==17:c.box(0,0,1,1)
    if k==11:
        # The priority at an intersection symbol is a broad through road
        # crossing a narrow side road, rather than a mandatory-direction arrow.
        c=Canvas(x,y);c.poly((.28,1),(.28,.61),(.02,.61),(.02,.43),(.28,.43),(.28,.25),(.50,.015),(.72,.25),(.72,.43),(.98,.43),(.98,.61),(.72,.61),(.72,1))
    if k==27 and st==1:
        # Older pedestrian pictograms use a coat and relatively broad limbs.
        c.dot(.52,.36,.14,.19);c.poly((.43,.43),(.58,.48),(.56,.59),(.37,.57));c.path([(.46,.54),(.31,.77),(.14,.95)],.14);c.path([(.50,.55),(.6,.73),(.84,.93)],.13)
    if k==21 and st==1:
        c=Canvas(x,y);c.box(.025,0,.30,.55);c.poly((.03,.38),(.97,.30),(.97,.56),(.03,.64));c.box(.70,.4,.97,1)
    if k==23 and st==2:
        c=Canvas(x,y);c.poly((.09,.07),(.79,.12),(.85,.19),(.83,.35),(.70,.42),(.08,.34));
        for dx in (-.20,.22):
            t=np.linspace(0,1,26);c.path(list(zip(.49+dx+.16*np.sin(2*np.pi*t+.7),.34+.64*t)),.034)
    if k==29 and st==2:
        # The bicycle itself (no rider): two equal wheels, diamond frame,
        # saddle and handlebars.
        c=Canvas(x,y)
        for cx in (.23,.77):c.dot(cx,.64,.219,.32);c.dot(cx,.64,.163,.24,0)
        c.path([(.23,.64),(.38,.28),(.51,.65),(.23,.64),(.42,.37),(.65,.28),(.77,.64)],.047);c.path([(.37,.29),(.34,.17),(.28,.17)],.04);c.path([(.65,.29),(.62,.10),(.72,.09)],.042)
    if k in (36,37) and st==1:
        c=Canvas(1-x if k==37 else x,y)
        c.box(.18,.16,.30,.98);c.poly((.24,0),(.48,.27),(.40,.30),(.24,.19),(.06,.31),(0,.27));
        q=bez((.24,.79),(.28,.54),(.40,.51),(.71,.51));c.path(q,.12);c.poly((.58,.31),(.69,.31),(.99,.51),(.68,.75),(.58,.75),(.72,.55),(.72,.47))
    if st==1:
        if k==11:
            c.box(.29,.27,.71,1);c.poly((.06,.41),(.5,0),(.94,.41));c.box(0,.52,1,.73)
        if k==18:
            c.box(.08,.04,.92,.36);c.poly((.08,.35),(.92,.35),(.67,.77),(.33,.77))
        if k==24:
            c=Canvas(x,y);c.box(.06,0,.22,1);p=bez((.55,.10),(.54,.40),(.90,.47),(.90,.76));c.path([(.55,0)]+list(p)+[(.9,1)],.18)
        if k==23:
            c=Canvas(x,y);car(c,.10,.025,.80,.33);c.box(.15,.07,.85,.19)
            for dx in (-.19,.22):
                t=np.linspace(0,1,30);c.path(list(zip(.5+dx+.18*np.sin(2*np.pi*t),.34+.64*t)),.085)
        if k==28:
            c=Canvas(x,y);person(c,.41,0,.59,.99,False);person(c,.025,.28,.57,.70,False)
        if k==29:
            c.path([(.32,.35),(.50,.34),(.60,.24)],.08);c.path([(.45,.45),(.31,.6),(.37,.71)],.08)
        if k==31:
            c.dot(.62,.47,.25,.13);c.path([(.28,.34),(.24,.17),(.32,.05)],.082);c.path([(.26,.25),(.17,.16)],.045);c.path([(.27,.19),(.20,.1)],.04)
        if k==40:
            c=Canvas(x,y)
            for j in range(3):
                t=np.linspace(.27,1.9,30)+np.pi/2+j*2*np.pi/3;pts=np.stack((.5+.385*np.cos(t),.5-.385*np.sin(t)),1);c.path(pts,.095);p=pts[-1];f=np.array([-np.sin(t[-1]),-np.cos(t[-1])]);n=np.array([f[1],-f[0]]);c.poly(p+.11*f,p-.13*f+.16*n,p-.13*f-.16*n)
        if k==26:
            for j,strength in enumerate((.8,.38,.8)):c.dot(.5,.15+.35*j,.485,.142,strength)
    return c.a

# Canonical icon boxes in units of sign radius: (width,height,vertical centre).
# Their precise size and placement are nuisance parameters in the matcher.
boxes={0:(1.15,.9,0),1:(1.15,.9,0),2:(1.15,.9,0),3:(1.15,.9,0),4:(1.15,.9,0),5:(1.15,.9,0),7:(1.30,.90,0),8:(1.30,.90,0),9:(1.33,.43,.12),10:(1.32,.75,.04),11:(.87,1.02,.28),14:(1.67,.63,0),16:(1.36,.66,.02),17:(1.76,.40,0),18:(.20,1.17,.22),19:(.63,.84,.30),20:(.63,.84,.30),21:(.64,1.05,.24),22:(1.12,.36,.52),23:(.85,.97,.31),24:(.72,1.03,.25),25:(1.28,1.0,.28),26:(.31,1.13,.22),27:(.78,.99,.25),28:(1.04,.97,.29),29:(1.12,.89,.36),30:(.89,.99,.27),31:(.99,.9,.29),33:(1.22,1.21,-.02),34:(1.22,1.21,-.02),35:(.82,1.58,0),36:(1.38,1.50,0),37:(1.38,1.50,0),38:(1.18,1.18,0),39:(1.18,1.18,0),40:(1.62,1.62,0),41:(1.35,.43,.10),42:(1.32,.75,.04)}

def group(k):
    if k in (6,32,41,42):return 6
    if k==12:return 3
    if k==13:return 2
    if k in (14,17):return 5
    if k==11 or 18<=k<=31:return 1
    if 33<=k<=40:return 4
    return 0

# High-resolution vector raster, reused when sampling different optical blurs,
# layouts, and inclinations of this same individual template.
def makeicon(k,st):
    t=(np.arange(160)+.5)/160; x,y=np.meshgrid(t,t)
    return glyph(k,x,y,st)

def interp(a,xf,yf):
    h,w=a.shape; x=xf*w-.5;y=yf*h-.5;u=np.floor(x).astype(int);v=np.floor(y).astype(int);f=x-u;g=y-v
    def at(u,v):return a[np.clip(v,0,h-1),np.clip(u,0,w-1)]*((u>=0)&(u<w)&(v>=0)&(v<h))
    return (at(u,v)*(1-f)+at(u+1,v)*f)*(1-g)+(at(u,v+1)*(1-f)+at(u+1,v+1)*f)*g

# Each entry stores only an individual drawing. All means and norms used in
# constructing a filter are spatial reductions of that drawing alone.
def iconbanks():
    data={}
    for g in (0,1,4,5,6):
        bx,by,cy,side,mask=window(g)
        step=2*bx/side;ystep=2*by/side
        ts=(np.arange(side*3)+.5)/(side*3)*2-1
        x,y=np.meshgrid(ts*bx,ts*by+cy);filt=[];labels=[];pri=[]
        for k in range(43):
            if k==15 or group(k)!=g:continue
            if k in (6,32): basek=5
            else: basek=k
            w,h,oy=boxes[basek];styles=range(3) if k in [0,1,2,3,4,5,6,7,8,23,27,29] else (0,1) if (33<=k<=40 or k in (9,10,11,18,21,24,26,28,31,41,42)) else (0,)
            for st in styles:
                a=makeicon(basek,st)
                for sx in ((.64,.82,1.,1.18) if g==1 else (.85,1.,1.15)):
                    for sy in ((.67,.85,1.,1.15) if g==1 else (.85,1.,1.15)):
                        for angle in (-.095,0.,.095):
                            xx=x+angle*(y-oy); yy=y-angle*x
                            # Condensing a warning pictogram typically moves its centre
                            # down: its feet still stand on the same baseline.
                            baseline=.5*h*(1-sy) if g==1 else 0
                            im=interp(a,xx/(w*sx)+.5,(yy-oy-baseline)/(h*sy)+.5)
                            if k in (6,32,41,42):
                                d=(xx+yy)/np.sqrt(2);stripe=np.abs(d)<.20
                                if k==32:im=stripe.astype(float)
                                else:im=np.maximum(.7*im,stripe)
                            f=im.reshape(side,3,side,3).mean((1,3));
                            # Pixel aperture / modest point spread; both sharp and
                            # blurred structure remain in the matched pattern.
                            pad=np.pad(f,1,mode='edge');f=(4*f+pad[:-2,1:-1]+pad[2:,1:-1]+pad[1:-1,:-2]+pad[1:-1,2:])/8
                            f=f[mask];f-=f.mean();f/=max(np.linalg.norm(f),1e-7)
                            filt.append(f);labels.append(k);pri.append(.009*(abs(sx-1)+abs(sy-1))/.15+.006*abs(angle)/.095)
        data['icons'+str(g)]=np.asarray(filt,dtype=np.float32);data['labels'+str(g)]=np.array(labels);data['cost'+str(g)]=np.array(pri,np.float32)
    return data

def window(g):
    if g==0: bx,by,cy,s=.77,.61,.01,25
    if g==1: bx,by,cy,s=.76,.69,.18,24
    if g==4 or g==6: bx,by,cy,s=.99,.99,0,26
    if g==5: bx,by,cy,s=.98,.6,0,25
    t=(np.arange(s)+.5)*2/s-1;x,y=np.meshgrid(t*bx,t*by+cy)
    if g==1:mask=(y<.78)&(y>2.03*np.abs(x)-.71)
    else:mask=x*x+y*y<({0:.79,4:.91,5:.91,6:.91}[g])**2
    return bx,by,cy,s,mask

def outlines():
    data={}; yy,xx=np.mgrid[:32,:32]
    poses=[(cx,cy,rx,ry) for cx in (12.5,14,15.5,17,18.5) for cy in (12.5,14,15.5,17,18.5) for rx in (8.25,9.75,11.25,12.75,14.25,15.75) for ry in (8.25,9.75,11.25,12.75,14.25,15.75) if .62<rx/ry<1.61]
    data['poses']=np.array(poses,np.float32)
    for g in range(7):
        co=[]; lum=[]; ed=[]
        for cx,cy,rx,ry in poses:
            x=(xx-cx)/rx;y=(yy-cy)/ry
            rad=np.sqrt(x*x+y*y); d=rad
            if g in (1,2):
                t=y if g==1 else -y
                # Polygonal gauge about the incenter, with a rounded apex.
                d=np.maximum((t-1/3)*1.5,1.5*np.abs(x)-.75*(t-1/3));d=np.maximum(d,(-t-.27)/.72)
            if g==3:d=np.abs(x)+np.abs(y)
            def inside(r):return np.clip(.5+(r-d)/.095,0,1)
            def between(lo,hi):return inside(hi)-inside(lo)
            def unit(m):return m/(m.sum()+1e-9)
            outer=unit(between(1.035,1.24)); edge=unit(np.exp(-((d-1)/.09)**2)); pale=unit(inside(.73));paint=unit(between(.76,.965))
            if g==0:pale=unit(inside(.745)*(.1+np.minimum(1,(np.abs(y)/.48)**3)))
            if g in (1,2):
                paint=unit(between(.71,.956));pale=unit(between(.43,.68)+.22*inside(.43));edge=.55*edge+.45*unit(np.exp(-((d-.70)/.085)**2))
            if g==0:edge=.60*edge+.40*unit(np.exp(-((d-.76)/.075)**2))
            if g in (0,1,2):
                chrom=.55*(paint-.84*pale-.16*outer)+.30*paint
                light=.64*(pale-paint)+.10*(pale-outer)
            if g==3:
                paint=unit(inside(.66));pale=unit(between(.75,.94));chrom=.57*(paint-.85*pale-.15*outer)+.36*paint;light=.41*(pale-paint)+.21*(pale-outer);edge=.5*edge+.5*unit(np.exp(-((d-.69)/.1)**2))
            if g==4:
                paint=unit(between(.60,.91)+.16*inside(.60));inner=unit(inside(.56));rim=unit(between(.96,1.055));chrom=.41*(paint-outer)+.53*paint+.16*(paint-inner);light=.24*(rim-paint)+.14*(outer-paint)+.14*(inner-paint)
            if g==5:
                paint=unit(inside(.90)*np.clip((np.abs(y)-.24)*7,0,1));pale=unit(inside(.82)*np.clip((.235-np.abs(y))*10,0,1));chrom=.43*(paint-.60*pale-.40*outer)+.46*paint;light=.48*(pale-paint)+.10*(outer-paint)
            if g==6:
                sd=np.abs(x+y)*.7071;pale=unit(inside(.94)*np.clip((sd-.29)*8,0,1)*(.35+np.minimum(1,np.abs(y)*1.8)));shade=unit(inside(.96)*np.clip((.21-sd)*8,0,1));chrom=-.48*pale;light=.70*(pale-shade)+.30*(pale-outer);edge=.7*edge+.3*unit(inside(.95)*np.exp(-((sd-.21)/.085)**2))
            co.append(chrom.ravel());lum.append(light.ravel());ed.append(edge.ravel())
        # Compact storage of these formula-generated spatial weights.
        data['shape'+str(g)]=np.concatenate([co,lum,ed],axis=1).T.astype(np.float32)
    return data

def gradientpool(g):
    bx,by,cy,s,m=window(g); ix=np.arange(m.sum()); ids=np.zeros(m.shape,int);ids[m]=ix
    inner=m.copy();inner[1:,:]&=m[:-1,:];inner[:-1,:]&=m[1:,:];inner[:,1:]&=m[:,:-1];inner[:,:-1]&=m[:,1:];inner[[0,-1]]=False;inner[:,[0,-1]]=False
    iy,ix=np.where(inner); edges=np.array([ids[iy,ix-1],ids[iy,ix+1],ids[iy-1,ix],ids[iy+1,ix]])
    cols=np.linspace(2,s-3,6);p=[]
    for y in cols:
        for x in cols:
            f=np.exp(-((ix-x)**2+(iy-y)**2)/10.);p.append(f)
    return edges,np.array(p,np.float32).T.copy()

def histograms(a,edges,pool):
    l,r,t,b=edges;dx=a[:,r]-a[:,l];dy=a[:,b]-a[:,t];angle=np.array([0,.78539816,1.57079633,2.35619449],np.float32)
    mag=np.sqrt(dx*dx+dy*dy);hs=[]
    for theta in angle:
        p=dx*np.cos(theta)+dy*np.sin(theta);hs.extend([np.maximum(0,p-.5*mag)@pool,np.maximum(0,-p-.5*mag)@pool])
    z=np.concatenate(hs,axis=1);z/=np.maximum(1e-8,z.sum(1,keepdims=True));z=np.sqrt(z);return z

def addhist(data):
    for g in (0,1,4,5,6):
        e,p=gradientpool(g);key=str(g);data['hp'+key]=p;data['he'+key]=e;data['hist'+key]=histograms(data['icons'+key],e,p)

def components(data):
    # Individual numerals, without the common zero dominating the readout.
    g=0;bx,by,cy,s,m=window(g);ts=(np.arange(s)+.5)*2/s-1;x,y=np.meshgrid(ts*bx,ts*by+cy);x=x[m];y=y[m]
    mat=data['icons0'];labs=data['labels0'];res=[]
    for xr in (-.33,.13):
        w=1/(1+np.exp((x-xr-.28)/.045))/(1+np.exp((-x+xr-.28)/.045))
        f=mat-(mat*w).sum(1,keepdims=True)/w.sum();f=f*w;f/=np.maximum(1e-8,np.sqrt((f*f).sum(1,keepdims=True)));res.append(f.astype(np.float32))
    data['digitparts']=np.stack(res)
    for g in (0,1,4):
        bx,by,cy,s,m=window(g);ts=(np.arange(s)+.5)*2/s-1;u,v=np.meshgrid(ts*bx,ts*by+cy);u=u[m];v=v[m]
        weights=[];mats=[]
        # Compare half-patterns on each side of a natural inter-symbol space.
        # Local centering prevents a shadow, or one black blob, from hiding a
        # missing stroke in the other half of the drawing.
        regions={0:[(-.47,0,.26,.49),(.19,0,.39,.49)],1:[(0,-.015,.60,.29),(0,.48,.62,.26)],4:[(0,-.38,.71,.37),(0,.38,.71,.37)]}[g]
        for cx,cy,wx,wy in regions:
            w=np.exp(-.5*((u-cx)/wx)**4-.5*((v-cy)/wy)**4);w=w.astype(np.float32);mat=data['icons'+str(g)];f=(mat-(mat*w).sum(1,keepdims=True)/w.sum())*np.sqrt(w)
            # A constant patch is not evidence either for or against the sign.
            f/=np.maximum(.07,np.sqrt((f*f).sum(1,keepdims=True)));mats.append(f);weights.append(w)
        data['local'+str(g)]=np.stack(mats);data['regions'+str(g)]=np.stack(weights)

def ocrbank(data):
    ts=(np.arange(96)+.5)/96;x,y=np.meshgrid(ts*1.36-.18,ts*1.24-.12);mats=[];ids=[]
    for k in (0,1,2,3,5,6,7,8):
        for st in range(3):
            for a in (-.08,0,.08):
                z=digit(str(k),x+a*(y-.5),y,st).reshape(16,6,16,6).mean((1,3));p=np.pad(z,1,mode='edge');z=(4*z+p[1:-1,:-2]+p[1:-1,2:]+p[:-2,1:-1]+p[2:,1:-1])/8;z-=z.mean();z/=np.maximum(.00001,np.linalg.norm(z));mats.append(z.ravel());ids.append(k)
    data['ocr']=np.array(mats,np.float32);data['ocrid']=np.array(ids)

solution_source = "import numpy as np\n\nposebank=CONSTANTS['poses']\nshapeweights=[CONSTANTS['shape'+str(g)].T.copy() for g in range(7)]\n\ndef product(a,b):\n    # Force the same GEMM tile dimensions and strides even when pruning\n    # leaves a partial batch. Each dot product is still over one image only.\n    n=len(a);out=np.empty((n,b.shape[1]),np.float32);chunk=64;p=np.zeros((chunk,a.shape[1]),np.float32)\n    for i in range(0,n,chunk):\n        k=min(n-i,chunk);p[:k]=a[i:i+k];out[i:i+k]=(p@b)[:k]\n    return out\nspatialprior=(.018*((posebank[:,0]-15.5)**2+(posebank[:,1]-15.5)**2)/9+.012*((posebank[:,2]-12)**2+(posebank[:,3]-12)**2)/16+.013*((posebank[:,2]-posebank[:,3])/3)**2).astype(np.float32)\n\ndef grid(g):\n    if g==0: bx,by,cy,s=.77,.61,.01,25\n    if g==1: bx,by,cy,s=.76,.69,.18,24\n    if g==4 or g==6: bx,by,cy,s=.99,.99,0,26\n    if g==5: bx,by,cy,s=.98,.6,0,25\n    t=(np.arange(s)+.5)*2/s-1;x,y=np.meshgrid(t*bx,t*by+cy)\n    if g==1:mask=(y<.78)&(y>2.03*np.abs(x)-.71)\n    else:mask=x*x+y*y<({0:.79,4:.91,5:.91,6:.91}[g])**2\n    return x[mask].astype(np.float32),y[mask].astype(np.float32)\n\nicongroups=(0,1,4,5,6)\ngrids={g:grid(g) for g in icongroups}\nclasses={g:np.unique(CONSTANTS['labels'+str(g)]) for g in icongroups}\nstarts={g:np.searchsorted(CONSTANTS['labels'+str(g)],classes[g]) for g in icongroups}\nshifts=np.array([(u,v,s,t) for s,t in [(1,1),(.94,.94),(1.06,1.06)] for u in (-1.05,0,1.05) for v in (-1.05,0,1.05)],np.float32)\nshiftcost=(.012*(shifts[:,0]**2+shifts[:,1]**2)+.13*abs(shifts[:,2]-1)).astype(np.float32)\n\ndef sample(im,x,y):\n    n=im.shape[0];xx=np.clip(x,0,30.9999);yy=np.clip(y,0,30.9999);u=xx.astype(np.int32);v=yy.astype(np.int32);xx=xx-u;yy=yy-v\n    i=np.arange(n)[:,None,None];return (((1-xx)*im[i,v,u]+xx*im[i,v,u+1])*(1-yy)+((1-xx)*im[i,v+1,u]+xx*im[i,v+1,u+1])*yy).astype(np.float32)\n\ndef innerhist(a,g):\n    e=CONSTANTS['he'+str(g)];pool=CONSTANTS['hp'+str(g)];dx=a[:,e[1]]-a[:,e[0]];dy=a[:,e[3]]-a[:,e[2]];mag=np.sqrt(dx*dx+dy*dy);hs=[]\n    for theta in (0,.78539816,1.57079633,2.35619449):\n        p=dx*np.float32(np.cos(theta))+dy*np.float32(np.sin(theta));hs.extend([product(np.maximum(0,p-.5*mag),pool),product(np.maximum(0,-p-.5*mag),pool)])\n    z=np.concatenate(hs,axis=1);z/=np.maximum(1e-8,z.sum(1,keepdims=True));return np.sqrt(z)\n\ndef interior(g,lumin,poses,noise):\n    n=len(lumin);u,v=grids[g];p=poses[:,None,:]+np.zeros((1,len(shifts),1),np.float32)\n    p[:,:,0:2]+=shifts[:,0:2];p[:,:,2:4]*=shifts[:,2:4]\n    x=p[:,:,0:1]+p[:,:,2:3]*u;y=p[:,:,1:2]+p[:,:,3:4]*v\n    a=sample(lumin,x,y);mean=a.mean(2,keepdims=True);a-=mean\n    # Remove a smooth brightness gradient so a shadow need not look like ink.\n    for t in (u,v):\n        z=t-t.mean();a-=.80*(a*z).sum(2,keepdims=True)/(z*z).sum()*z\n    std=np.sqrt((a*a).mean(2,keepdims=True));a=np.maximum(np.minimum(a,2.85*std),-2.85*std);a-=a.mean(2,keepdims=True)\n    if g not in (4,5):a=-a\n    norm=np.sqrt((a*a).sum(2,keepdims=True));a/=np.maximum(norm,.00001)\n    mat=a.reshape(n*len(shifts),-1);b=product(mat,CONSTANTS['icons'+str(g)].T)\n    # For numbers, an extra local correlation compares the leading digit and\n    # reduces interference from the zero shared between almost all limits.\n    if g==0:\n        for j,xr in enumerate((-.33,.13)):\n            w=1/(1+np.exp((u-xr-.28)/.045))/(1+np.exp((-u+xr-.28)/.045));f=(mat-(mat*w).sum(1,keepdims=True)/w.sum())*w;f/=np.maximum(.00001,np.sqrt((f*f).sum(1,keepdims=True)))\n            ab=starts[0][0] if j==0 else starts[0][6];end=starts[0][6] if j==0 else starts[0][8]\n            b[:,ab:end]=.76*b[:,ab:end]+.24*product(f,CONSTANTS['digitparts'][j,ab:end].T)\n    if g in (0,1,4):\n        part=np.zeros_like(b)\n        for j,w in enumerate(CONSTANTS['regions'+str(g)]):\n            f=(mat-(mat*w).sum(1,keepdims=True)/w.sum())*np.sqrt(w);f/=np.maximum(.07,np.sqrt((f*f).sum(1,keepdims=True)));part+=product(f,CONSTANTS['local'+str(g)][j].T)\n        b=.68*b+.16*part\n    hog=product(innerhist(mat,g),CONSTANTS['hist'+str(g)].T)\n    b=.77*b+.23*(hog*2-1);b-=CONSTANTS['cost'+str(g)]; corr=np.maximum.reduceat(b,starts[g],axis=1).reshape(n,len(shifts),-1)\n    # Quantisation noise is local to the sign face: the brightness of the\n    # surrounding sky need not predict the readability of a number.\n    strength=std[:,:,0]/(noise[:,None]+.004)\n    reliab=strength/(strength+.65);result=((corr-.23)*reliab[:,:,None]-shiftcost[None,:,None]).max(axis=1)\n    if g==0:\n        # Test blankness on a central disk, away from any painted border.\n        mid=(u*u+v*v<.57**2);blankstd=(norm[:,4,0]/np.sqrt(a.shape[2]))*np.std(a[:,4,mid],axis=1)*np.sqrt(a.shape[2]);flat=blankstd/(noise+.004)\n        blank=.40/(1+flat*flat*.30)-.06\n        result=np.column_stack((result,blank))\n    return result\n\ndef quant(a,q):\n    n=a.shape[0];v=a.reshape(n,-1);index=int((v.shape[1]-1)*q);return np.partition(v,index,axis=1)[:,index,None,None]\n\ndef batch(images):\n    r=images[:,0];g=images[:,1];b=images[:,2];n=len(r)\n    raw=.27*r+.53*g+.20*b\n    lumin=np.arcsinh(raw/.17);centre=lumin[:,5:27,5:27];lo=quant(centre,.09);hi=quant(centre,.91)\n    gray=np.clip((lumin-lo)/(hi-lo+.054),-.42,1.38)\n    dx=np.empty_like(gray);dy=np.empty_like(gray);dx[:,:,1:-1]=(gray[:,:,2:]-gray[:,:,:-2])*.5;dx[:,:,0]=0;dx[:,:,-1]=0;dy[:,1:-1,:]=(gray[:,2:,:]-gray[:,:-2,:])*.5;dy[:,0,:]=0;dy[:,-1,:]=0\n    edge=np.minimum(.9,np.sqrt(dx*dx+dy*dy)*2.5)\n    red=(r-.64*g-.36*b)/(r+.64*g+.36*b+.04)\n    blue=(b-.58*r-.42*g)/(b+.58*r+.42*g+.04)\n    yellow=(.65*r+.35*g-b)/(.65*r+.35*g+b+.045)-.63*np.maximum(red,0)\n    # Relative chroma catches a red ring even with a colour cast. Absolute\n    # chroma prevents turning arbitrary gray variations into saturated paint.\n    def colour(t,level):\n        base=quant(t[:,5:27,5:27],.18);return np.clip((t-.58*base-.025)/level,-.32,1.3)\n    red=colour(red,.33);blue=colour(blue,.30);yellow=colour(yellow,.36)\n    # Estimate a noise floor from diagonal high-pass differences. Using a\n    # central region keeps the background separate from the sign material.\n    hp=lumin[:,11:21,11:21]+lumin[:,10:20,10:20]-lumin[:,10:20,11:21]-lumin[:,11:21,10:20];noise=.40*quant(np.abs(hp),.4)[:,0,0]+.012/(.24+quant(raw[:,11:21,11:21],.6)[:,0,0]*4)\n    arr=np.zeros((n,7),np.float32);ps=np.zeros((n,7,4),np.float32)\n    for j in range(7):\n        cm=red\n        if j==3:cm=yellow\n        if j==4:cm=blue\n        if j==6:cm=np.maximum(0,np.maximum(red,blue))\n        feat=np.concatenate([cm.reshape(n,-1),gray.reshape(n,-1),edge.reshape(n,-1)],axis=1)\n        # Multiply a fixed number of rows each time, independent of batch size.\n        val=feat@shapeweights[j].T-spatialprior\n        best=val.argmax(1);ps[:,j]=refinepose(val,best,posebank[best]);arr[:,j]=val[np.arange(n),best]\n    # Family hypotheses compete with each other after the internal symbol has\n    # been read. Ambiguous borders are not a hard veto on an excellent symbol.\n    score=np.full((n,43),-4,np.float32)\n    score[:,12]=arr[:,3]+.32;score[:,13]=arr[:,2]+.32\n    # Yield has a plain inner white field, not a symbol. Look for internal\n    # high-frequency structure using points strictly inside its white triangle.\n    u,v=np.meshgrid(np.linspace(-.6,.6,12,dtype=np.float32),np.linspace(-.69,.32,12,dtype=np.float32));m=v<.53-2*abs(u);u=u[m];v=v[m];p=ps[:,2];sx=p[:,0,None]+p[:,2,None]*u;sy=p[:,1,None]+p[:,3,None]*v\n    edges=sample(edge,sx[:,None,:],sy[:,None,:])[:,0].mean(axis=1);score[:,13]-=.78*edges\n    for j in icongroups:\n        # Rule out only very poor outline/colour matches before expensive OCR.\n        ii=np.where(arr[:,j]>arr.max(1)-.78)[0]\n        if len(ii)==0:continue\n        if j in (4,5): fg=np.arcsinh((.47*r[ii]+.5*g[ii]+.03*b[ii])/.17)\n        else:fg=lumin[ii]\n        if j==0 or j==1:fg=np.arcsinh((.24*r[ii]+.55*g[ii]+.21*b[ii])/.18)\n        contrast=noise[ii]; labels=classes[j]\n        if j==0:labels=np.append(labels,15)\n        for begin in range(0,len(ii),8):\n            sl=slice(begin,begin+8);idx=ii[sl]\n            read=interior(j,fg[sl],ps[idx,j],contrast[sl])\n            if j==0:\n                words=independentdigits(fg[sl],ps[idx,j],contrast[sl]);whole=read[:,:8].copy();separate=.58*whole+.42*words\n                # The component read is a second way to recognize the same\n                # text, not a requirement that each tiny numeral be legible.\n                read[:,:8]=np.maximum(separate,whole-.025)\n            score[idx[:,None],labels]=arr[idx,j,None]+read\n    colouredsymbols(images,ps,noise,score)\n    return score.argmax(axis=1)\n\ndef predict(x):\n    out=np.empty(len(x),np.int64)\n    for i in range(0,len(x),32):\n        # Identical GEMM panel sizes even for a singleton call.\n        k=min(len(x)-i,32);a=np.zeros((32,3,32,32),np.float32);a[:k]=x[i:i+k];out[i:i+k]=batch(a)[:k]\n    return out\n\ndef region(im,p,ux,uy,wx,wy):\n    # Mean over a small elliptical area; support points are a deterministic\n    # quadrature rule. There is no dependence on other inputs in the batch.\n    t=np.array([0,.70,-.70,0,0,.50,.50,-.50,-.50],np.float32);s=np.array([0,0,0,.70,-.70,.50,-.50,.50,-.50],np.float32)\n    x=p[:,0,None]+p[:,2,None]*(ux+wx*t);y=p[:,1,None]+p[:,3,None]*(uy+wy*s)\n    return sample(im,x[:,None,:],y[:,None,:])[:,0].mean(1)\n\ndef colouredsymbols(images,ps,noise,score):\n    r,g,b=images[:,0],images[:,1],images[:,2]\n    # Paint hypotheses specific to the sign face: a red vehicle should be\n    # visible on the left, even if its gray value matches the black vehicle.\n    redness=np.log((r+.025)/(.66*g+.34*b+.025));p=ps[:,0]\n    left=region(redness,p,-.33,.10,.16,.14);right=region(redness,p,.33,.10,.16,.14);paper=region(redness,p,0,-.36,.26,.10)\n    # Log chroma can also change with brightness through the camera response.\n    # Compare two objects of similar brightness as well as the local paper,\n    # and require colour to be localized rather than spilling in from a rim.\n    outside=region(redness,p,-.33,-.36,.15,.09);car=np.clip(np.minimum(left-right,left-.5*(outside+paper))/.28,-.4,1.15);score[:,9]+=.17*car;score[:,10]+=.17*car\n    # Red, amber, green arranged vertically distinguish signals from a black\n    # exclamation mark. Relative colour rejects a uniform yellow cast.\n    p=ps[:,1];xg=np.log((g+.025)/(r+.025));bottom=region(xg,p,0,.57,.10,.10);top=region(xg,p,0,-.13,.09,.10);white=.5*(region(xg,p,-.26,.53,.09,.09)+region(xg,p,.26,.53,.09,.09));tgreen=bottom-white\n    amber=np.log((.50*r+.50*g+.025)/(b+.025));gold=region(amber,p,0,.21,.10,.1)-.5*(region(amber,p,-.22,.24,.06,.07)+region(amber,p,.22,.24,.06,.07))\n    chrom=np.clip(np.minimum(bottom-top,2*tgreen+.35*np.maximum(gold,0))/.32,0,1.3);score[:,26]+=.26*chrom-.12;score[:,18]+=.055*(1-np.minimum(chrom,1))\n\ndef refinepose(val,best,ps):\n    # Sub-grid pose estimate from the local response surface. Keep widely\n    # separated maxima (e.g. another sign above this one) out of the centroid.\n    d=posebank[None,:,:]-ps[:,None,:];near=(abs(d).max(axis=2)<1.56);v=val-val[np.arange(len(val)),best,None]\n    weight=near*np.exp(np.maximum(v*19,-40));return (weight@posebank)/weight.sum(axis=1,keepdims=True)\n\ndef independentdigits(fg,poses,noise):\n    # Read a word as two or three independent glyphs on a common baseline.\n    # This admits condensed type without mistaking narrow digit spacing for\n    # thin strokes in a wide character. Intensities are normalized per glyph.\n    n=len(fg);digits=(0,1,2,3,5,6,7,8);ts=(np.arange(16,dtype=np.float32)+.5)/16;u,v=np.meshgrid(ts*1.36-.68,ts*1.24-.62);u=u.ravel();v=v.ravel()\n    ans=[]\n    for length in (2,3):\n        layout=np.array([(pitch,dx,dy,h) for pitch in ((.51,.61,.71) if length==2 else (.315,.39,.465,.54)) for dx in (-.075,0,.075) for dy in (-.065,.035) for h in (.77,.95,1.08)],np.float32);pitch,dx,dy,h=layout.T\n        per=[];strengths=[]\n        for j in range(length):\n            px=dx+(j-(length-1)/2)*pitch;w=pitch*.83\n            if length==3:px=px+(.12 if j==0 else -.04)*pitch\n            a=sample(fg,poses[:,None,0:1]+poses[:,None,2:3]*(px[:,None]+w[:,None]*u),poses[:,None,1:2]+poses[:,None,3:4]*(dy[:,None]+h[:,None]*v));a=a.mean(axis=2,keepdims=True)-a\n            for t in (u,v):a-=.55*(a*t).sum(axis=2,keepdims=True)/(t*t).sum()*t\n            std=np.sqrt((a*a).mean(axis=2));a=np.clip(a,-2.8*std[:,:,None],2.8*std[:,:,None]);a-=a.mean(axis=2,keepdims=True);norm=np.sqrt((a*a).sum(axis=2,keepdims=True));a/=np.maximum(norm,.00001)\n            z=product(a.reshape(-1,len(u)),CONSTANTS['ocr'].T).reshape(n,len(px),8,9).max(3);rel=std/(std+.7*noise[:,None]+.004);per.append((z-.25)*rel[:,:,None]);strengths.append(std)\n        for number in (('20','30','50','60','70','80') if length==2 else ('100','120')):\n            cs=np.stack([per[j][:,:,digits.index(int(ch))] for j,ch in enumerate(number)]);val=.70*cs.mean(0)+.30*cs.min(0)\n            if length==3:\n                # Beyond contrast alone, use the repeated shape in '100'.\n                # Two clipped or defocused zeroes may still match each other\n                # closely, even when neither matches an ideal unblurred zero.\n                xx=poses[:,None,0:1]+poses[:,None,2:3]*((dx+pitch*.46)[:,None]+(pitch*.83)[:,None]*u);yy=poses[:,None,1:2]+poses[:,None,3:4]*(dy[:,None]+h[:,None]*v)\n                off=poses[:,None,2:3]*pitch[:,None]*.5;left=sample(fg,xx-off,yy);right=sample(fg,xx+off,yy);left-=left.mean(2,keepdims=True);right-=right.mean(2,keepdims=True)\n                repeat=(left*right).sum(2)/np.sqrt(np.maximum(1e-10,(left*left).sum(2)*(right*right).sum(2)));val+=.22*(repeat-.35)*(1 if number=='100' else -.35)\n            # Large pose corrections cost a little, but are sometimes needed\n            # with asymmetric glyph side bearings or a sign seen at an angle.\n            val-=.09*(abs(dx)+abs(dy));ans.append(val.max(1))\n    return np.stack(ans,1)\n"

def main():
    folder=Path(sys.argv[1]);folder.mkdir(parents=True,exist_ok=True)
    drawings=iconbanks();addhist(drawings);components(drawings);ocrbank(drawings);drawings.update(outlines())
    np.savez(folder/'constants.npz',**drawings)
    (folder/'solution.py').write_text(solution_source)

if __name__=='__main__':main()
