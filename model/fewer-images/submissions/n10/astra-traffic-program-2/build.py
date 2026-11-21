# Geometry for individual sign templates. Units are sign half-widths/heights.
import numpy as np
import math, sys
from pathlib import Path

# Material indices: neutral surround, white enamel, red, blue, yellow, black,
# gray cancellation print, dark stripe, green signal.
O,W,R,B,Y,K,H,D,G=range(9)
light=np.array([.48,.97,.24,.21,.74,.07,.54,.24,.42])
resolution=128
grid=(np.arange(resolution)+.5)*2.4/resolution-1.2
X,Z=np.meshgrid(grid,grid)

def poly(points, color):
    inside=np.zeros(X.shape,bool)
    a,b=points[-1]
    for c,d in points:
        if b!=d:
            inside ^= ((Z>b)!=(Z>d)) & (X<(c-a)*(Z-b)/(d-b)+a)
        a,b=c,d
    canvas[inside]=color

def ellipse(x,y,a,b,color):
    canvas[((X-x)/a)**2+((Z-y)/b)**2<1]=color

def line(points,width,color=K):
    for (a,b),(c,d) in zip(points,points[1:]):
        dx,dy=c-a,d-b
        t=np.clip(((X-a)*dx+(Z-b)*dy)/(dx*dx+dy*dy+1e-12),0,1)
        canvas[(X-a-t*dx)**2+(Z-b-t*dy)**2<(width/2)**2]=color

def curve(pts):
    out=[pts[0]]
    for i in range(1,len(pts),3):
        a,b,c,d=np.array(pts[i-1:i+3]); t=np.arange(1,10)[:,None]/9
        out.extend((1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t**2*c+t**3*d)
    return out

def rect(x1,y1,x2,y2,col=K):
    canvas[(X>=x1)&(X<=x2)&(Z>=y1)&(Z<=y2)]=col

def group(c):
    if c in (6,32,41,42): return 6
    if c in (11,*range(18,32)):return 1
    if c in (12,13,14):return c-10
    if c==17:return 5
    if c>=33:return 7
    return 0

def margin(g,x,y):
    # Approximate signed distance, positive inward. All shapes touch +/- 1.
    if g==1 or g==3:
        z=y if g==1 else -y
        # Slightly round off the three pointed triangle corners.
        return np.minimum.reduce([1-z,(z+1.12-2.13*abs(x))/2.35, .999+z, 1.005-abs(x)])
    if g==2:return (1.03-abs(x)-abs(y))*.707
    if g==4:return np.minimum(np.minimum(1-abs(x),1-abs(y)),(1.416-abs(x)-abs(y))*.707)
    return 1-np.sqrt(x*x+y*y)

def digit(ch,cx,h,w,thick,style):
    # Centre-line construction of the DIN-like numerals, with explicit open
    # forms for 6 and 7. Stroke coordinates refer to the individual type body.
    paths=[]
    if ch=='0':
        paths=[curve([(.5,.04),(.78,.04),(.92,.16),(.92,.34),(.92,.45),(.92,.57),(.92,.67),(.92,.88),(.79,.96),(.49,.96),(.19,.96),(.08,.81),(.08,.66),(.08,.54),(.08,.43),(.08,.34),(.08,.14),(.24,.04),(.5,.04)])]
    if ch=='1':paths=[[ (.12,.22),(.57,.035),(.57,.96) ]]
    if ch=='2':paths=[curve([(.07,.24),(.12,-.025),(.8,-.04),(.9,.18),(1.04,.37),(.72,.49),(.51,.63)])+[(.08,.95),(.96,.95)]]
    if ch=='3':paths=[curve([(.07,.18),(.26,-.05),(.86,.005),(.89,.22),(.94,.40),(.73,.47),(.41,.475)]),curve([(.4,.475),(.77,.45),(.98,.59),(.92,.76),(.89,1.01),(.27,1.04),(.06,.83)])]
    if ch=='5':paths=[[ (.89,.055),(.19,.055),(.12,.48) ]+curve([(.12,.48),(.35,.32),(.87,.37),(.92,.58),(1.04,.89),(.54,1.11),(.08,.84)])]
    if ch=='6':
        p=curve([(.15,.62),(.21,.43),(.83,.4),(.91,.64),(1.05,.91),(.56,1.08),(.22,.89),(-.02,.76),(.10,.55),(.23,.42)])
        paths=[[(.7,.026),(.23,.42)]+p[::-1]]
        if style==2:paths=[curve([(.86,.12),(.37,-.16),(.05,.26),(.12,.65)])+p]
    if ch=='7':paths=[[ (.04,.055),(.94,.055),(.34,.965) ]]
    if ch=='8':paths=[curve([(.5,.04),(.98,.04),(1.02,.38),(.61,.46),(.30,.54),(.05,.55),(.08,.74),(.09,1.03),(.91,1.03),(.93,.74),(.95,.55),(.69,.51),(.42,.45),(.0,.36),(.05,.04),(.5,.04)])]
    for p in paths:
        line([(cx+(x-.5)*w,(y-.5)*h) for x,y in p],thick)

def auto(cx,cy,s,col,truck=False):
    def pg(points):poly([(cx+s*x,cy+s*y) for x,y in points],col)
    if truck:
        pg([(-.87,-.86),(.68,-.86),(.68,.38),(.88,.38),(.88,.72),(-.87,.72)])
    else:
        pg([(-1,.43),(-.96,-.03),(-.76,-.19),(-.56,-.61),(.52,-.61),(.75,-.21),(.97,-.04),(1,.44)])
        poly([(cx+s*x,cy+s*y) for x,y in [(-.59,-.20),(-.42,-.46),(.40,-.46),(.56,-.20)]],W)
    for x in (-.65,.64):rect(cx+s*(x-.13),cy+.35*s,cx+s*(x+.13),cy+.69*s,col)

def arrowline(points,width,tip,spread,depth,solid=False):
    line(points,width,W)
    x,y=points[-1]; dx,dy=tip; l=np.hypot(dx,dy); dx/=l;dy/=l
    a=(x-dx*depth-dy*spread,y-dy*depth+dx*spread); b=(x-dx*depth+dy*spread,y-dy*depth-dx*spread)
    if solid:poly([a,(x+dx*.045,y+dy*.045),b],W)
    else:line([a,(x,y),b],width,W)

def pict(c,st):
    if c in (0,1,2,3,4,5,6,7,8):
        text=['20','30','50','60','70','80','80','100','120'][c]; n=len(text)
        w=(.37 if n==2 else .297)*[1,.93,1.05][st]; h=[.78,.82,.74][st]; stride=w+(.147 if n==2 else .105)
        if c==6:
            # The cancellation sign has no red annulus to take up room.
            h*=[1.17,1.33,1.40][st];w*=[1.06,1.16,1.22][st];stride*=1.13
        for i,ch in enumerate(text):digit(ch,(i-(n-1)*.5)*stride,h,w,([.100,.118,.085][st])*(.86 if n==3 else 1),st)
        if c==6:canvas[canvas==K]=H
        # On narrow three-digit indications the '1' has a narrower advance
        # than a full-width numeral in some versions of the lettering.
        if n==3 and st!=2:
            canvas[canvas==K]=W
            for ch,cx,ww in zip(text,[-.46,-.095,.352],[.205,.33,.33]):digit(ch,cx,h,ww,[.087,.103][st],st)
    if c in (9,10,41,42):
        tr=c in (10,42)
        auto(-.32,-.09 if tr else .04,.266,R if c<41 else H,tr)
        auto(.315,.125 if tr else .04,.265,K if c<41 else H)
        if tr:
            rect(-.56,-.40,-.11,.04,R if c<41 else H)
    if c==16:
        rect(-.135,-.27,.56,.145); poly([(-.55,.16),(-.55,-.055),(-.39,-.23),(-.17,-.23),(-.17,.13),(.48,.13),(.48,.2),(-.5,.2)],K)
        poly([(-.48,-.045),(-.37,-.16),(-.25,-.16),(-.25,-.045)],W)
        for x in (-.37,.32):ellipse(x,.18,.099,.099,K)
    if c==11:
        poly([(-.14,.69),(.14,.69),(.14,.35),(.37,.36),(.32,.23),(.14,.15),(.14,.0),(.235,.0),(0,-.42),(-.235,.0),(-.14,.0),(-.14,.15),(-.32,.23),(-.37,.36),(-.14,.35)],K)
    if c==18:
        poly([(-.067,-.42),(.067,-.42),(.053,.38),(-.042,.38)],K); ellipse(0,.57,.084,.088,K)
    if c in (19,20):
        flip=1 if c==19 else -1
        # A straight approach and a quarter-turn, with squared-off ends.
        cy=[.12,.0,-.095][st];rr=[.20,.235,.18][st];edge=.225
        p=[(edge,.64),(edge,cy+rr)]+[(edge-rr+rr*math.cos(t),cy+rr-rr*math.sin(t)) for t in np.linspace(0,math.pi/2,12)]+[(-.26,cy)]
        line([(flip*x,y) for x,y in p],.164)
        canvas[(canvas==K)&((Z>.64)|(flip*X<-.26))]=W
    if c==21:
        p=curve([(-.14,-.24),(-.15,-.06),(-.18,.12),(-.02,.21),(.10,.28),(.20,.18),(.24,.32),(.27,.40),(.25,.53),(.25,.62)])
        if st==0:line(p,.16)
        else:
            # Two opposing quarter-turns separated by a straight road.
            ax,ay=[(-.16,-.26),(-.19,-.19)][st-1];bx=.19;mid=[.21,.17][st-1];r=[.054,.095][st-1]
            path=[(ax,ay),(ax,mid-r)]+[(ax+r+r*np.cos(t),mid-r+r*np.sin(t)) for t in np.linspace(np.pi,np.pi/2,10)]+[(bx-r,mid)] + [(bx-r+r*np.cos(t),mid+r+r*np.sin(t)) for t in np.linspace(-np.pi/2,0,10)]+[(bx,.64)]
            line(path,.162);canvas[(canvas==K)&((Z<ay)|(Z>.64))]=W
    if c==22:
        p=curve([(-.51,.61),(-.42,.59),(-.40,.33),(-.23,.36),(-.12,.37),(-.10,.48),(-.02,.48),(.04,.48),(.10,.32),(.23,.35),(.40,.33),(.41,.59),(.53,.61)])
        poly(list(p)+[(.53,.65),(-.51,.65)],K)
    if c==23 and st==0:
        # Side/front view variants of the slippery-road warning symbol.
        def drift(x,y):return (x-.055,y+.12+.19*x)
        poly([drift(x,y) for x,y in [(-.34,.105),(-.33,-.045),(-.24,-.067),(-.18,-.18),(.03,-.18),(.155,-.062),(.31,-.033),(.34,.089),(.29,.125)]],K)
        for xx in [-.20,.19]:
            cx,cy=drift(xx,.105);ellipse(cx,cy,.062,.062,K)
        for dx in [-.21,.13]:
            line(curve([(dx,.24),(dx-.28,.33),(dx+.28,.40),(dx,.49),(dx-.22,.55),(dx-.12,.59),(dx-.08,.62)]),.038)
    if c==23 and st!=0:
        # Skidding car, tilted body and winding tire marks.
        sl=[.34,-.33,.18][st];stretch=[1,1.16,1.25][st]
        def skew(x,y):return ((x*.94-y*sl)*stretch,x*sl+y*.94+.015)
        poly([skew(x,y) for x,y in [(-.26,.12),(-.25,-.08),(-.14,-.25),(.16,-.25),(.24,-.09),(.27,.13)]],K)
        poly([skew(x,y) for x,y in [(-.17,-.09),(-.10,-.19),(.11,-.19),(.17,-.09)]],W)
        for x in (-.16,.15):
            a,b=skew(x,.12); line(curve([(a,b),(a+.14,b+.12),(a+.13,b+.16),(a-.01,b+.22),(a-.43,b+.33),(a-.06,b+.40),(a-.07,b+.51)]),.052)
    if c==24:
        rect(-.22,-.19,-.115,.66);line(curve([(.055,-.16),(.058,.11),(.23,.15),(.23,.38)])+[(.23,.62)],.105)
    if c in (25,27,28):
        if c==25:
            ellipse(-.055,-.17,.077,.088,K);line([(-.11,-.07),(-.26,.15),(-.27,.28)],.14)
            line([(-.24,.22),(-.36,.41),(-.42,.64)],.085);line([(-.23,.24),(-.10,.41),(-.13,.60)],.081)
            line([(-.13,-.015),(-.08,.15),(.07,.22)],.063); line([(-.27,.09),(.31,.55)],.040)
            poly([(.00,.65),(.17,.48),(.26,.41),(.36,.45),(.55,.65)],K)
        else:
            if c==27:people=[(0,0,1)]
            else:
                # Older child leading a smaller one by the hand; both in motion.
                people=[[(.18,.00,.91),(-.25,.21,.62)],[(.14,.01,.89),(-.24,.22,.65)],[(-.11,.00,.84),(.22,.25,.54)]][st]
            for ox,oy,s in people:
                def body(p,w):line([(ox+s*x,oy+s*y) for x,y in p],w*s)
                ellipse(ox+.042*s,oy-.22*s,.075*s,.083*s,K)
                body([(.018,-.09),(-.075,.20)],.13);body([(-.05,.18),(.08,.4),(.23,.61)],.091);body([(-.048,.19),(-.16,.42),(-.31,.61)],.086)
                body([(-.005,-.065),(.09,.105),(.23,.17)],.064);body([(-.04,-.05),(-.20,.09),(-.245,.23)],.061)
    if c==26:
        poly([(-.118,-.35),(0,-.415),(.118,-.35),(.118,.60),(0,.666),(-.118,.60)],K)
        for y,col in [(-.215,R),(.105,Y),(.445,G)]:ellipse(0,y,.103,.134,col)
    if c==29:
        for x in (-.32,.30):
            ellipse(x,.43,.191,.192,K);ellipse(x,.43,.143,.144,W)
        line([(-.32,.43),(-.13,.19),(.02,.43),(-.32,.43)],.04);line([(.02,.43),(.20,.14),(.30,.43)],.04)
        ellipse(.049,-.17,.075,.077,K);line([(.005,-.06),(-.13,.095),(.02,.19),(-.006,.37)],.071);line([(.005,-.055),(.12,.08),(.224,.065)],.052)
        if st!=0:
            # The cycle symbol may depict a bare, diamond-frame bicycle.
            canvas[(canvas==K)&(Z<.12)]=W
            line([(-.15,.175),(-.19,.055)],.047);line([(-.26,.057),(-.124,.057)],.036)
            line([(.19,.17),(.14,-.044),(.046,-.065)],.042)
    if c==30:
        for i in range(6):
            a=(i+.5)*np.pi/3; dx,dy=np.cos(a),np.sin(a); rad=.36; cy=.27
            line([(0,cy),(dx*rad,cy+dy*rad)],.044)
            for s in (-1,1):line([(.19*dx,cy+.19*dy),(.29*dx+s*.095*dy,cy+.29*dy-s*.095*dx)],.039)
    if c==31:
        poly([(-.43,.16),(-.33,.19),(-.23,.12),(-.06,.15),(.14,.17),(.23,.05),(.27,-.08),(.35,-.12),(.41,-.065),(.47,-.025),(.36,.014),(.33,.04),(.29,.25),(.16,.31),(-.06,.30),(-.22,.28),(-.32,.38),(-.42,.51),(-.49,.51),(-.38,.32)],K)
        for p in [[(-.2,.24),(-.23,.38),(-.05,.51)],[(.19,.24),(.27,.38),(.38,.38)],[(.13,.28),(.13,.42),(.28,.51)],[(.29,-.064),(.20,-.18),(.22,-.25)]]:line(p,.042)
        # Galloping silhouette: also allow a leaping deer, with the
        # neck high and front and rear hooves extended in opposite directions.
        if st!=0:
            canvas[canvas==K]=W
            poly([(-.36,.14),(-.29,.08),(-.12,.05),(.06,.08),(.18,.045),(.225,-.16),(.285,-.21),(.34,-.145),(.40,-.12),(.38,-.075),(.29,-.076),(.265,.06),(.23,.25),(.12,.29),(-.085,.245),(-.24,.25),(-.30,.33),(-.40,.37),(-.48,.42),(-.52,.41),(-.43,.29),(-.345,.22)],K)
            for limb in [[(.16,.21),(.26,.27),(.39,.39),(.45,.40)],[(.10,.22),(.14,.34),(.33,.42)],[(-.20,.20),(-.25,.31),(-.37,.39),(-.44,.53)],[(-.19,.21),(-.20,.39),(-.27,.49)]]:line(limb,.048)
            for antler in [[(.27,-.16),(.20,-.22),(.15,-.31)],[(.22,-.22),(.215,-.32)],[(.26,-.14),(.30,-.27)]]:line(antler,.033)
        if st==1:
            # Deer viewed side-on in mid leap: raised head and ears, deep
            # chest, tapered body, tucked forelegs and trailing hind legs.
            canvas[canvas==K]=W
            poly([(-.46,.01),(-.36,-.005),(-.28,-.08),(-.26,-.17),(-.22,-.09),(-.16,-.14),(-.17,-.032),(-.18,.04),(-.09,.104),(.154,.17),(.28,.235),(.32,.29),(.365,.25),(.35,.33),(.305,.37),(.246,.355),(.135,.28),(-.055,.253),(-.135,.24),(-.195,.20),(-.24,.119),(-.293,.052),(-.38,.067),(-.45,.047)],K)
            for leg in [[(-.145,.206),(-.232,.29),(-.353,.255),(-.393,.291)],[(-.085,.204),(-.152,.37),(-.24,.361)],[(.208,.249),(.25,.41),(.406,.515),(.471,.55)],[(.263,.27),(.343,.41),(.488,.454)]]:line(leg,.045)
            for branch in [[(-.257,-.093),(-.21,-.192),(-.205,-.285)],[(-.21,-.192),(-.14,-.225)]]:line(branch,.028)
        animal=(canvas==K);shoulder=[-.15,.06,.28][st];mirror=[-1,1,-1][st];scaleY=[1.17,1.12,1.12][st]
        u=mirror*(X*np.cos(shoulder)+(Z-.23)*np.sin(shoulder));v=(-X*np.sin(shoulder)+(Z-.23)*np.cos(shoulder))/scaleY+.23
        ix=np.clip(((u/2.4+.5)*resolution).astype(int),0,resolution-1);iy=np.clip(((v/2.4+.5)*resolution).astype(int),0,resolution-1)
        canvas[animal]=W;canvas[animal[iy,ix]]=K
    if c==14:
        # Condensed lettering on the red octagon.
        w=.25; centers=[-.51,-.18,.15,.48]
        letters=[ [curve([(.86,.13),(.12,-.16),(-.12,.27),(.28,.405),(.4,.48),(.73,.49),(.82,.60),(1.07,.9),(.54,1.18),(.02,.9)])], [[(0,.04),(1,.04)],[(.5,.04),(.5,.99)]], [curve([(.5,0),(.1,0),(.08,.18),(.08,.31),(.08,.47),(.08,.64),(.08,.75),(.08,1.06),(.94,1.09),(.94,.75),(.94,.59),(.94,.45),(.94,.30),(.94,.15),(.90,0),(.5,0)])], [[(.1,1),(.1,.035)]+curve([(.1,.035),(1.14,-.12),(1.19,.62),(.1,.51)])] ]
        for cx,paths in zip(centers,letters):
            for p in paths:line([(cx+(x-.5)*w,(y-.5)*.62) for x,y in p],.075,W)
    if c==17:rect(-.77,-.14,.77,.14,W)
    if c in (33,34):
        s=1 if c==33 else -1
        p=[(-.34,.60),(-.34,.07)]+list(curve([(-.34,.07),(-.34,-.13),(-.18,-.28),(.04,-.28)]))+[(.48,-.28)]
        arrowline([(s*x,y) for x,y in p],.155,(s,0),.22,.22,st==2)
    if c in (35,36,37):
        ox=0 if c==35 else (-.19 if c==36 else .19)
        arrowline([(ox,.65),(ox,-.62)],.19,(0,-1),.30,.31,st!=1)
        if c!=35:
            s=1 if c==36 else -1
            p=curve([(ox,.50),(ox,.31),(ox+s*.19,.13),(ox+s*.37,.13)])+[(.53*s,.13)]
            arrowline(p,.155,(s,0),.195,.20,st!=1)
    if c in (38,39):
        s=1 if c==38 else -1
        arrowline([(-.44*s,-.47),(.43*s,.43)],.17,(s,1),.29,.31,st==2)
    if c==40:
        for i in range(3):
            a=np.pi*(i*2/3+[.085,.22,.37][st]); ts=np.linspace(a,a-[1.52,1.65,1.42][st],15);r=[.62,.66,.57][st]
            arrowline([(r*np.cos(t),r*np.sin(t)) for t in ts],.123,(np.sin(ts[-1]),-np.cos(ts[-1])),.16,.17,st==1)

def art(c,st):
    global canvas
    canvas=np.full(X.shape,O,dtype=np.int8); g=group(c); d=margin(g,X,Z)
    canvas[d>0]=W
    if g in (0,1,3):
        canvas[d>(.018 if g==0 else .022)]=R
        canvas[d>([.192,.167,.213][st] if g==0 else [.150,.132,.168][st])]=W
    elif g==2:canvas[d>.175]=Y
    elif g in (4,5,7):canvas[d>(.038 if g==4 else .018)]=B if g==7 else R
    pict(c,st)
    if g==6:
        # Multiple close black diagonal lines optically merge on small signs.
        band=(X+.0+(.78 if st!=1 else .91)*Z)
        sel=(d>.010)&(abs(band)<[.207,.24,.163][st]);canvas[sel]=D;canvas[sel&(np.mod(band+.26,.084)<.025)]=H
    if c==6 and st==0:canvas[(canvas==H)&(abs(X+.78*Z)>.21)]=K
    return canvas

labels=[c for c in range(43)]

# Compiling a template only ever uses that one drawing.
const={}

def unit(v,eps=1e-10):return v/(np.sqrt(np.sum(v*v))+eps)
def getpix(a,x,y):
    x=np.clip((x/2.4+.5)*resolution-.5,0,resolution-1.001); y=np.clip((y/2.4+.5)*resolution-.5,0,resolution-1.001)
    j=x.astype(int); i=y.astype(int); u=x-j;v=y-i
    return (1-v)*((1-u)*a[i,j]+u*a[i,j+1])+v*((1-u)*a[i+1,j]+u*a[i+1,j+1])

def blurplane(a,s):
    for axis in (0,1):
        y=np.arange(-int(3*s+.5),int(3*s+.5)+1); k=np.exp(-.5*(y/s)**2);k/=k.sum()
        a=sum(w*np.roll(a,v,axis) for w,v in zip(k,y))
    return a

def removeplane(v,x,y):
    v=v-v.mean();xx=x-x.mean();yy=y-y.mean()
    return v-xx*((v*xx).sum()/((xx*xx).sum()+1e-8))-yy*((v*yy).sum()/((yy*yy).sum()+1e-8))

def shapefilters():
    # Inward/outward unit transition over an edge broadened by pixel sampling.
    pos=np.arange(32)-15.5; px,py=np.meshgrid(pos,pos); params=[]; fs=[]; fam=[]; scale=[]; strong=[]
    for g in range(8):
        for r in [7.4,8.8,10.3,11.9,13.5,15.1]:
            for ratio in [.89,1,1.12]:
                rx,ry=r*ratio,r/ratio
                for dx in [-2,0,2]:
                    for dy in [-2,0,2]:
                        x=(px-dx)/rx;y=(py-dy)/ry;d=margin(g,x,y); sw=.64/r
                        def step(t):return np.clip(.5+t/(2*sw),0,1)
                        face=step(d);support=step(d+.205);gap=support-face
                        def contrast(a,b):
                            return a/(a.sum()+1e-8)-b/(b.sum()+1e-8)
                        if g in (0,1,3):
                            th=.19 if g==0 else .153
                            ring=step(d-.018)-step(d-th); interior=step(d-th-.03)
                            # Look preferentially just inside the red border.
                            inner=interior*(.20+.80*np.exp(-((d-th-.12)/.20)**2))
                            chrom=contrast(ring,.65*inner/(inner.sum()+1e-8)+.35*gap/(gap.sum()+1e-8))
                            lum=contrast(inner,ring)
                            a0=0
                        elif g==2:
                            core=step(d-.183); border=step(d-.005)-step(d-.17)
                            chrom=contrast(core,.65*border/(border.sum()+1e-8)+.35*gap/(gap.sum()+1e-8));lum=contrast(border,core);a0=.55
                        elif g in (4,5,7):
                            if g==5:
                                bar=np.clip((.79-abs(x))/sw+.5,0,1)*np.clip((.148-abs(y))/sw+.5,0,1); main=face*(1-bar);neutral=bar
                            else:
                                neutral=step(d)-step(d-.042);main=step(d-.045)
                                if g==4:main*=1-.85*np.exp(-((y/.34)**4))*np.exp(-((x/.72)**8))
                                if g==7:main*=.48+.52*np.clip((.47-d)*3,0,1)
                            chrom=contrast(main,.75*gap/(gap.sum()+1e-8)+.25*neutral/(neutral.sum()+1e-8));lum=contrast(neutral,main);a0=.68
                        else:
                            stripe=np.clip((.214-abs(x+.83*y))/(2*sw)+.5,0,1)*face; bare=face-stripe
                            lum=contrast(stripe*0+bare,stripe);chrom=contrast(face,gap);a0=0
                        fs.append([chrom.ravel(),lum.ravel()]); params.append([dx+15.5,dy+15.5,rx,ry,0,0]);fam.append(g);scale.append(r);strong.append(a0)
    const['shapes']=np.array(fs,np.float32);const['pose']=np.array(params,np.float32)

shapefilters()

# Fine, local, edge-aligned registration: oriented sample points are on the
# ideal border. The sample just outside it and the one inside form a detector.
for g in range(8):
    if g in (0,5,6,7):
        ang=np.arange(48)*np.pi/24;nx=np.cos(ang);ny=np.sin(ang); bx=nx;by=ny
    elif g in (2,4):
        if g==2:corners=[(0,-1),(1,0),(0,1),(-1,0)]
        else:corners=[(-.414,-1),(.414,-1),(1,-.414),(1,.414),(.414,1),(-.414,1),(-1,.414),(-1,-.414)]
        mat=[]
        for (a,b),(c,d) in zip(corners,corners[1:]+corners[:1]):
            vx=c-a;vy=d-b;l=np.hypot(vx,vy)
            for t in np.linspace(.08,.92,8):mat.append([a+vx*t,b+vy*t,vy/l,-vx/l])
        bx,by,nx,ny=np.array(mat).T
    else:
        corners=[(0,-1.12),(1,1),(-1,1)];mat=[]
        for (a,b),(c,d) in zip(corners,corners[1:]+corners[:1]):
            vx=c-a;vy=d-b;l=np.hypot(vx,vy)
            for t in np.linspace(.10,.90,16):mat.append([a+vx*t,b+vy*t,vy/l,-vx/l])
        bx,by,nx,ny=np.array(mat).T
        if g==3:by=-by;ny=-ny
    const['edge'+str(g)]=np.array([bx,by,nx,ny],np.float32)

for g in range(8):
    cs=[c for c in labels if group(c)==g];v=(np.arange(30)+.5)/15-1; xx,yy=np.meshgrid(v,v);d=margin(g,xx,yy)
    mask=d>({0:.255,1:.195,7:.13,6:.105}.get(g,.085));
    if g==0:mask &= (abs(yy)<.58)
    xx=xx[mask];yy=yy[mask];const['points'+str(g)]=np.array([xx,yy],np.float32)
    if len(cs)==1:continue
    allfs=[]; cid=[]; vpen=[]; en=[]; directional=[]; rgbs=[]
    for c in cs:
        for st in range(3):
            image=art(c,st); intensity=light[image];red=(image==R)*1.; yellow=(image==Y)*1.
            broad=(1.55,2.95) if g in (0,6) else (1.65,2.80)
            for ib,s in enumerate(broad):
                lum=blurplane(intensity,s);color=blurplane(red if c!=26 else yellow+red,s); gy,gx=np.gradient(lum); tx=gx*resolution/2.4;ty=gy*resolution/2.4
                for sx,sy in [(1,1),(.87,1.02),(1.13,1.02),(1,.86),(1,1.16)]:
                    # Enamel outside the inner patch is intentionally not sampled.
                    x=xx/sx;y=(yy-(.30 if g==1 else 0))/sy+(.30 if g==1 else 0)
                    if g in (0,1):
                        # Sampling shouldn't cause the red frame to appear in a pictogram.
                        base=light[W]; sym=np.where(image==O,base,intensity)
                        if g==0:
                            sym=np.where((image==R)&(X*X+Z*Z>.42),base,sym)
                        else:sym=np.where(margin(g,X,Z)<.18,base,sym)
                        if sx==1 and sy==1:
                            ls=blurplane(sym,s);gs,hs=np.gradient(ls);hs=hs*resolution/2.4;gs=gs*resolution/2.4
                        b=getpix(ls,x,y);bx=getpix(hs,x,y)/sx;by=getpix(gs,x,y)/sy
                    else:
                        b=getpix(lum,x,y);bx=getpix(tx,x,y)/sx;by=getpix(ty,x,y)/sy
                    b=removeplane(b,xx,yy);b=np.clip(b,-.80,.80); be=float(np.sqrt(np.mean(b*b)))
                    allfs.append(unit(b) if be>.001 else np.zeros_like(b));en.append(be)
                    gg=np.stack([bx,by],0); mag=np.sqrt(bx*bx+by*by); gg=gg/np.sqrt(mag+.32)
                    directional.append(unit(gg.ravel()) if be>.001 else np.zeros(gg.size));
                    col=getpix(color,x,y);rgbs.append(unit(removeplane(col,xx,yy)) if c in (9,10,26) else np.zeros_like(col))
                    cid.append(c);vpen.append(.007*(sx!=1 or sy!=1)+.004*ib)
    key=str(g);const['gray'+key]=np.array(allfs,np.float32);const['grad'+key]=np.array(directional,np.float32);const['col'+key]=np.array(rgbs,np.float32);const['energy'+key]=np.array(en,np.float32);const['cid'+key]=np.array(cid);const['pen'+key]=np.array(vpen,np.float32)

for g in range(8):
    q=(np.arange(32)+.5)/16-1;xx,yy=np.meshgrid(q,q);sel=margin(g,xx,yy)>.029; x=xx[sel];y=yy[sel];key=str(g)
    const['whole'+key]=np.array([x,y],np.float32); lf=[];cf=[];ids=[]
    for c in [c for c in labels if group(c)==g]:
        for st in range(3):
            im=art(c,st); a=light[im];ch=im==(B if g==7 else Y if g==2 else R);a=np.where(im==O,W*.9,a)
            for s in [1.75,3.1]:
                b=removeplane(getpix(blurplane(a,s),x,y),x,y);lf.append(unit(b))
                u=getpix(blurplane(ch*1.,s),x,y); u=removeplane(u,x,y); cf.append(unit(u) if np.std(u)>.09 else np.zeros_like(u));ids.append(c)
    const['wf'+key]=np.array(lf,np.float32);const['wc'+key]=np.array(cf,np.float32);const['wi'+key]=np.array(ids)

# Integrate each warning pictogram in narrow directional strips. They record
# where ink is distributed, complementary to alignment of its detailed edges.
key='1';xx,yy=const['points'+key];gray=const['gray'+key]; strips=[]
for angle in [0,np.pi*.5,np.pi*.25,-np.pi*.25]:
    u=xx*np.cos(angle)+yy*np.sin(angle);filters=[]
    for t in np.linspace(u.min(),u.max(),9):
        f=np.exp(-.5*((u-t)/.071)**2);filters.append(f/(f.sum()+1e-8))
    proj=np.array(filters,dtype=np.float32);strips.append(proj)
    f=gray@proj.T; f=f-f.mean(1,keepdims=True);const['stripTemplate'+str(len(strips))]=f/(np.sqrt(np.sum(f*f,axis=1,keepdims=True))+.012)
const['strips']=np.stack(strips)

# Parametric textureless disc detector, independent of any particular symbol.
# The absolute value of the radial derivative is used for neutral signs since
# their background can be darker OR lighter than the foreground.
yy,xx=np.mgrid[:32,:32];mats=[]
for cx,cy,rx,ry,tx,ty in const['pose'][:len(const['pose'])//8]:
    xxn=(xx-cx)/rx;yyn=(yy-cy)/ry;r=np.sqrt(xxn**2+yyn**2);theta=np.arctan2(yyn,xxn)
    edge=np.exp(-.5*((r-.99)/(.78/min(rx,ry)))**2);edge/=edge.sum()
    mat=[]
    for phi in [0,np.pi/4,np.pi/2,3*np.pi/4]:
        dist=np.arccos(np.clip(abs(np.cos(theta-phi)),0,1)); mat.append((edge*np.maximum(0,1-dist/(np.pi/4))).ravel())
    mats.append(mat)
const['discRim']=np.array(mats,np.float32)

# Additional partial-template comparisons concentrate on defining marks. A numeral
# in a string is distinguished by what is to its left as well as by its shape.
# For cancellation signs the common diagonal is an occluder, not a pictogram.
for g in [0,6]:
    key=str(g);x,y=const['points'+key];a=const['gray'+key];ids=const['cid'+key]
    if g==0:
        windows=[(x<.14),(x<-.23)&(abs(y)<.48),(abs(x)<.22)&(abs(y)<.49)]
    else:windows=[(abs(x+.82*y)>.37)&(x*x+y*y<.63)&(abs(y)<.56)]
    for i,keep in enumerate(windows):
        h=a[:,keep];h=h-h.mean(1,keepdims=True)
        # Linear detrending on this patch uses only the one template.
        xx=x[keep]-x[keep].mean();yy=y[keep]-y[keep].mean();yy=yy-((xx*yy).sum()/(xx*xx).sum())*xx
        for q in (xx,yy):h=h-((h*q).sum(1,keepdims=True)/(q*q).sum())*q
        norm=np.sqrt((h*h).sum(1,keepdims=True));h=h/np.maximum(norm,.025)
        const['patch'+key+str(i)]=h.astype(np.float32);const['select'+key+str(i)]=keep


solution = '''import numpy as np

families=[[0,1,2,3,4,5,7,8,9,10,15,16],[11]+list(range(18,32)),[12],[13],[14],[17],[6,32,41,42],list(range(33,41))]

def product(a,b):
    # Fixed row tiles even when the caller changes the batching.
    a=np.ascontiguousarray(a,dtype=np.float32); n=len(a);k=32
    padded=np.zeros(((n+k-1)//k*k,a.shape[1]),np.float32);padded[:n]=a
    return (padded.reshape(-1,k,a.shape[1])@b.T).reshape(-1,b.shape[0])[:n]

def smooth(im):
    a=np.pad(im,((0,0),(1,1),(1,1)),mode='edge'); m=a[:,1:-1]; b=m*.68+(a[:,:-2]+a[:,2:])*.16
    return b[:,:,1:-1]*.68+(b[:,:,:-2]+b[:,:,2:])*.16

def center(im):
    m=im[:,5:27,5:27].reshape(len(im),-1);q=np.quantile(m,[.08,.50,.92],axis=1).astype(np.float32)[:,:,None,None]
    return np.clip((im-q[1])/(q[2]-q[0]+.032),-1.6,1.6)

def features(x):
    r,g,b=x[:,0],x[:,1],x[:,2]; a=.28*r+.50*g+.22*b
    # Square root keeps detail in the shadows without greatly amplifying black noise.
    lum=np.sqrt(a+.008); lum=center(lum)
    v=.012+.36*r+.42*g+.22*b; red=np.clip((r-.64*g-.36*b)/v,-1.3,1.55);blue=np.clip((b-.62*r-.38*g)/v,-1.45,1.55);yellow=np.clip((.48*r+.52*g-b)/v,-1.4,1.55)
    return lum,red,blue,yellow

def rimscore(lum):
    ap=np.pad(smooth(lum),((0,0),(1,1),(1,1)),mode='edge');gx=ap[:,1:-1,2:]-ap[:,1:-1,:-2];gy=ap[:,2:,1:-1]-ap[:,:-2,1:-1]
    mats=CONSTANTS['discRim'];s=0
    for h,v in enumerate([gx,(gx+gy)*.707,gy,(gy-gx)*.707]):
        s+=product(np.minimum(abs(v),1.).reshape(len(lum),-1),mats[:,h])
    return s

def coarse(lum,red,blue,yellow):
    m=len(lum);n=CONSTANTS['shapes'].shape[0]//8; pos=[]; scores=[]
    for g in range(8):
        f=CONSTANTS['shapes'][g*n:(g+1)*n]; col=blue if g==7 else yellow if g==2 else red
        a=product(col.reshape(m,-1),f[:,0]); l=product(lum.reshape(m,-1),f[:,1]);
        if g in (0,1,3):s=.81*a+.71*l
        elif g==2:s=1.15*a+.41*l
        elif g==6:s=.82*l-.15*abs(a)+.94*rimscore(lum)
        else:s=(1.14 if g==7 else .99)*a+(.30 if g==7 else .49)*l
        p=CONSTANTS['pose'][g*n:(g+1)*n]; s=s-.015*((p[:,0]-15.5)**2+(p[:,1]-15.5)**2)/9-.032*((p[:,2]-11.9)/5)**2-.025*((p[:,3]-p[:,2])/4)**2
        ind=s.argmax(1);scores.append(s[np.arange(m),ind]);pos.append(p[ind]);
    return np.stack(scores,axis=1),np.stack(pos,axis=1)

def interp(im,x,y):
    n=len(im); x=np.clip(x,0,30.9999);y=np.clip(y,0,30.9999);a=x.astype(np.int32);b=y.astype(np.int32);s=x-a;t=y-b
    s=s.astype(np.float32);t=t.astype(np.float32);i=b*32+a+np.arange(n,dtype=np.int32).reshape((n,)+(1,)*(a.ndim-1))*1024;v=im.reshape(-1)
    return (1-t)*((1-s)*v[i]+s*v[i+1])+t*((1-s)*v[i+32]+s*v[i+33])

def positions(p,x,y):
    return p[:,:,0,None]+p[:,:,2,None]*(x+p[:,:,4,None]*y),p[:,:,1,None]+p[:,:,3,None]*(y+p[:,:,5,None]*x)

def posescore(p,img,lum,g):
    e=CONSTANTS['edge'+str(g)]; a,b=e[:2];nx,ny=e[2:]; v=.19 if g==0 else .152 if g in (1,3) else .18
    def val(d,im):
        x,y=positions(p,a-d*nx,b-d*ny);return interp(im,x,y)
    if g in (0,1,3,2):
        ring=val(v*.53,img); inner=val(v+.095,img);out=val(-.115,img)
        diff=ring-inner;outer=ring-out
        s=np.mean(np.clip(diff,-.44,1.45)+.26*np.clip(outer,-.52,1.15),axis=-1)
    else:
        inner=val(.105,img);out=val(-.095,img); outerl=val(-.078,lum);innerl=val(.084,lum)
        s=np.mean(.85*np.clip(inner-out,-.66,1.35)+.52*np.minimum(abs(innerl-outerl),.95),axis=-1)
        if g==6:
            t=np.linspace(-.64,.64,21,dtype=np.float32);vx=-.64*t;vy=.77*t
            def across(w):
                j,i=positions(p,vx+.77*w,vy+.64*w);return interp(lum,j,i)
            s+=.72*np.mean(np.clip((across(.29)+across(-.29))*.5-across(0),-.42,1.25),axis=-1)+.60*np.mean(np.minimum(abs(innerl-outerl),.95),axis=-1)
    prior=.014*((p[:,:,0]-15.5)**2+(p[:,:,1]-15.5)**2)/7+.025*((p[:,:,2]-p[:,:,3])/4)**2+.020*((p[:,:,2]+p[:,:,3]-24)/10)**2+.11*(abs(p[:,:,4])+abs(p[:,:,5]))
    return s-prior

def refine(p,col,l,g):
    if g in (0,1,3):sig=.76*col-.72*l
    elif g==2:sig=-col+.42*l
    elif g==6:sig=l*.22
    else:sig=col
    cand=np.array([[dx,dy,ax,ay,tx,ty] for dx,dy,ax,ay,tx,ty in [(0,0,0,0,0,0),(-1,0,0,0,0,0),(1,0,0,0,0,0),(0,-1,0,0,0,0),(0,1,0,0,0,0),(0,0,-1,0,0,0),(0,0,1,0,0,0),(0,0,0,-1,0,0),(0,0,0,1,0,0),(0,0,-1,-1,0,0),(0,0,1,1,0,0),(0,0,0,0,.105,-.105),(0,0,0,0,-.105,.105),(0,0,0,0,.115,0),(0,0,0,0,-.115,0)]],np.float32)
    start=p;p=p.copy();ii=np.arange(len(p));
    for step in (1.,.80,.55,.40):
        pp=p[:,None,:]+step*cand[None];s=posescore(pp,sig,l,g)
        s-=.024*(((pp[:,:,:4]-start[:,None,:4])**2).sum(axis=2)/6)
        idx=s.argmax(1);p=pp[ii,idx]
    return p

def normal(v,dust=0.):return v/(np.sqrt(np.sum(v*v,axis=-1,keepdims=True)+dust)+1e-6)

def detrend(v,x,y):
    v=v-v.mean(-1,keepdims=True);x=x-x.mean();y=y-y.mean()
    return v-(v*x).sum(-1,keepdims=True)*x/((x*x).sum()+1e-7)-(v*y).sum(-1,keepdims=True)*y/((y*y).sum()+1e-7)

def fullscore(p,lum,col,g):
    k=str(g); x,y=CONSTANTS['whole'+k]; pp=p[:,None];j,i=positions(pp,x,y);v=interp(lum,j,i)[:,0];h=interp(col,j,i)[:,0]
    z=product(normal(detrend(v,x,y),len(x)*.025**2),CONSTANTS['wf'+k]); f=product(normal(detrend(h,x,y),len(x)*.095**2),CONSTANTS['wc'+k]); ids=CONSTANTS['wi'+k]
    z=.66*z+(.22 if g!=6 else 0)*f
    if g==2:
        # The outer diamond is white, its middle yellow. They can have nearly the same brightness.
        z=.38*product(normal(detrend(v,x,y),len(x)*.045**2),CONSTANTS['wf'+k])+.53*f
    return np.stack([np.max(z[:,ids==c],1) for c in families[g]],axis=1)

def symbols(p,lum,clr,g):
    key=str(g); x,y=CONSTANTS['points'+key];n=len(p);m=len(x)
    dx=np.array([0,-1,1,0,0,-1,1,-1,1],np.float32);dy=np.array([0,0,0,-1,1,-1,-1,1,1],np.float32)
    pp=np.broadcast_to(p[:,None,:],(n,len(dx),6)).copy();pp[:,:,0]+=.073*p[:,2,None]*dx;pp[:,:,1]+=.073*p[:,3,None]*dy
    u,v=positions(pp,x,y); z=interp(lum,u,v); zn=detrend(z,x,y);rms=np.sqrt(np.mean(zn*zn,axis=-1));main=normal(zn,m*.022**2)
    # Orientations measured in sign coordinates; a one-pixel-scale symmetric derivative.
    ax=.06*p[:,2,None,None];ay=.06*p[:,3,None,None];gx=interp(lum,u+ax,v)-interp(lum,u-ax,v);gy=interp(lum,u,v+ay)-interp(lum,u,v-ay)
    scale=(gx*gx+gy*gy+.0012)**.25;vec=normal(np.concatenate([gx/scale,gy/scale],axis=-1),m*.012**2)
    ids=CONSTANTS['cid'+key];f=product(main.reshape(-1,m),CONSTANTS['gray'+key]); f=.74*f+.26*product(vec.reshape(-1,2*m),CONSTANTS['grad'+key]);f=f.reshape(n,len(dx),-1)
    if g==1:
        agg=0
        for h in range(4):
            zv=product(main.reshape(-1,m),CONSTANTS['strips'][h]);zv-=zv.mean(1,keepdims=True)
            agg+=product(normal(zv,.012**2),CONSTANTS['stripTemplate'+str(h+1)])
        f=.80*f+.20*agg.reshape(n,len(dx),-1)/4
    if g in (0,6):
        patches=[]
        for h in range(3 if g==0 else 1):
            keep=CONSTANTS['select'+key+str(h)];px=x[keep];py=y[keep];px=px-px.mean();py=py-py.mean();py=py-(px*py).sum()*px/(px*px).sum();vm=detrend(z[:,:,keep],px,py); count=len(px)
            if g==6:unused=np.sqrt(np.mean(vm*vm,axis=-1))
            patches.append(product(normal(vm,count*.020**2).reshape(-1,count),CONSTANTS['patch'+key+str(h)]).reshape(n,len(dx),-1))
        if g==0:
            sel=(ids<9);f[:,:,sel]=.64*f[:,:,sel]+.19*patches[0][:,:,sel]+.085*patches[1][:,:,sel]+.085*patches[2][:,:,sel]
        else:
            sel=(ids!=32);f[:,:,sel]=.52*f[:,:,sel]+.48*patches[0][:,:,sel];f[:,:,ids==32]=.52*f[:,:,ids==32]+.48*np.exp(-unused[:,:,None]/.12)
    if g in (0,1):
        cm=detrend(interp(clr,u,v),x,y); cw=product(normal(cm,m*.14**2).reshape(-1,m),CONSTANTS['col'+key]).reshape(n,len(dx),-1)
        f+=.13*np.maximum(cw,0)
    f-=CONSTANTS['pen'+key][None,None,:]+.011*(abs(dx)+abs(dy))[None,:,None]; scores=[]
    for c in families[g]:
        if c!=15:score=np.max(f[:,:,ids==c],axis=(1,2))
        else:
            # No vehicles: no symbol against the otherwise identical enamel.
            # Estimate the local noise floor from the unprinted top and bottom
            # of the pale centre. This avoids using a global exposure cutoff.
            wing=zn[:,0,abs(y)>.455]; noise=np.median(abs(wing-np.median(wing,axis=1,keepdims=True)),axis=1)*1.4826
            observed=np.quantile(zn[:,0],.85,axis=1)-np.quantile(zn[:,0],.055,axis=1)
            score=.82*np.exp(-np.maximum(0,observed-noise*.6)/.21)
            score-=.46*np.maximum(0,np.max(f[:,:,ids!=15],axis=(1,2))-.30)
        scores.append(score)
    return np.stack(scores,axis=1)

def classify(im):
    lum,red,blue,yellow=features(im);s,ps=coarse(lum,red,blue,yellow);n=len(im);result=np.full((n,43),-20,np.float32)
    cutoff=np.sort(s,axis=1)[:,-4]; sl=smooth(lum);ink=smooth(center(np.sqrt(.70*im[:,0]+.20*im[:,1]+.10*im[:,2]+.006)))
    for g in range(8):
        which=np.nonzero(s[:,g]>=cutoff)[0]
        if len(which)==0:continue
        col=(blue if g==7 else yellow if g==2 else red)[which];l=sl[which];p=refine(ps[which,g],col,l,g)
        # Compare the aligned outline against its immediate neighbourhood.
        # A locally plausible but backwards/upside-down boundary should not
        # win only because it encloses an untextured patch of road or sky.
        dx=p[:,0]-15.5;dy=p[:,1]-15.5
        linefeat=.76*col-.72*l if g in (0,1,3) else (-col+.42*l if g==2 else l*.22 if g==6 else col)
        verified=posescore(p[:,None],linefeat,l,g)[:,0]
        w=fullscore(p,l,col,g);t=s[which,g,None]+.28*verified[:,None]
        if len(families[g])>1:
            sym=(ink[which]*.62+l*.38) if g in (0,1,6) else l
            sy=symbols(p,sym,col,g);total=.69*t+.83*w+.87*sy
            if g in (0,1,6,7):
                # Foreshortening, blur and changes in icon layout need not
                # affect ink and frame in the same way. Rectify again at
                # nearby poses and retain the best joint agreement.
                offset=np.array([[0,-.08,.90,.92],[0,.08,1.10,1.08],[-.065,0,1,1],[.065,0,1,1],[0,0,.82,1.11],[0,0,1.14,.87]],np.float32)
                for off in offset:
                    alt=p.copy();alt[:,0]+=p[:,2]*off[0];alt[:,1]+=p[:,3]*off[1];alt[:,2]*=off[2];alt[:,3]*=off[3]
                    local=symbols(alt,sym,col,g)
                    # Do not move the no-ink test onto an arbitrary blank patch.
                    if g==0:local[:,families[g].index(15)]=sy[:,families[g].index(15)]
                    localw=fullscore(alt,l,col,g) if g==6 else w
                    total=np.maximum(total,.69*t+.83*localw+.87*local-.037)
        else:total=.69*t+1.55*w+.19
        if g in (1,3):
            # A triangle has three sides, not just one strong horizontal bar.
            e=CONSTANTS['edge'+str(g)]; pp=p[:,None]
            def sidelevel(d):
                j,i=positions(pp,e[0]-d*e[2],e[1]-d*e[3]);return interp(linefeat,j,i)[:,0]
            side=(sidelevel(.081)-sidelevel(.239)).reshape(len(p),3,-1).mean(2)
            total+=.36*(np.min(side,axis=1)-side.mean(1))[:,None]
        if g in (1,3):
            # An upright and an inverted warning triangle cannot both explain
            # the same coloured contour. Verify vertex/flat-base orientation
            # by comparing the correct boundary to its vertical reflection.
            opposite=3 if g==1 else 1
            anti=posescore(p[:,None],linefeat,l,opposite)[:,0]
            total+=.28*np.clip(verified-anti,-.65,.52)[:,None]
        if g==3:
            # The wide red side of a yield sign is ABOVE its point. Test the
            # red cap against the lower centre, in addition to the border.
            xx=np.linspace(-.44,.44,17,dtype=np.float32)
            j,i=positions(p[:,None],xx,-.919+xx*0);redtop=interp(col,j,i).mean(axis=(1,2))
            j,i=positions(p[:,None],xx*.48,.16+xx*0);inner=interp(col,j,i).mean(axis=(1,2))
            total+=.26*np.clip(redtop-inner-.25,-.75,.22)[:,None]
        if g==6:
            # Cancelled bans retain both the white disc and the central stripe.
            # Normalise partial-ink agreement independently from those two.
            total=np.maximum(total,.57*t+1.10*w+.70*sy)
        result[which[:,None],np.array(families[g])[None,:]]=total
    return result.argmax(1)

def predict(x):
    output=np.empty(len(x),np.int64)
    for j in range(0,len(x),32):output[j:j+32]=classify(x[j:j+32])
    return output
'''
target=Path(sys.argv[1])
target.mkdir(parents=True,exist_ok=True)
(target/'solution.py').write_text(solution)
np.savez(target/'constants.npz',**const)
