# Construction of individual antialiased pictogram templates, in a local [-1,1] frame.
# A pencil is a polyline with circular nib; piece() is a filled polygon.
import math, sys
from pathlib import Path
import numpy as np

S=24
Q=4

def piece(x,y,pts):
    inside=np.zeros(x.shape,bool)
    a,b=pts[-1]
    for c,d in pts:
        if b!=d:
            inside ^= ((b>y)!=(d>y)) & (x < a+(y-b)*((c-a)/(d-b)))
        a,b=c,d
    return inside

def pencil(x,y,pts,width):
    a,b=pts[0]; mask=np.zeros(x.shape,bool)
    for c,d in pts[1:]:
        u,v=c-a,d-b
        t=np.clip(((x-a)*u+(y-b)*v)/(u*u+v*v+1e-15),0,1)
        mask |= (x-a-u*t)**2+(y-b-v*t)**2 < (width*.5)**2
        a,b=c,d
    return mask

def oval(x,y,cx,cy,rx,ry):
    return ((x-cx)/rx)**2+((y-cy)/ry)**2 <= 1

def bezier(pts):
    # Cubic path specified by initial vertex, followed by control/control/end triples.
    out=[pts[0]]
    for j in range(1,len(pts),3):
        for t in np.linspace(0,1,10)[1:]:
            s=1-t
            out.append(tuple(s**3*pts[j-1][k]+3*s*s*t*pts[j][k]+3*s*t*t*pts[j+1][k]+t**3*pts[j+2][k] for k in (0,1)))
    return out

# Pen paths for rounded sans serif highway numerals. Unit coordinates in a character cell.
def numeral(x,y,d,style):
    w=.18 if style==1 else (.245 if style==2 else .208)
    # draw in physical units (taller than wide), keeping the pen nearly circular
    def path(p,c=False):
        if c:p=bezier(p)
        return pencil(x,y*1.57,[(a,b*1.57) for a,b in p],w)
    def ring(cy,rx,ry,n):
        dx=abs(x-.5);dy=abs(y-cy)
        return ((dx/rx)**n+(dy/ry)**n<=1)&((dx/(rx-w))**n+(dy/(ry-w/1.57))**n>=1)
    if d=='0':return ring(.5,.485,.50,2.9 if style!=1 else 2.25)
    if d=='8':return ring(.267,.432,.272,2.15)|ring(.73,.485,.272,2.3)
    if d=='1':return path([(.20,.219),(.54,.074),(.54,.932)])
    if d=='2':return path([(.10,.264),(.10,-.018),(.89,-.018),(.89,.256),(.90,.415),(.72,.49),(.55,.609),(.38,.726),(.21,.825),(.105,.931)],True)|path([(.105,.931),(.91,.931)])
    if d=='3':return path([(.098,.225),(.22,-.016),(.90,.009),(.874,.268),(.86,.408),(.718,.49),(.41,.49)],True)|path([(.43,.485),(.74,.469),(.91,.55),(.91,.704),(.92,.99),(.284,1.035),(.09,.822)],True)
    if d=='5':return path([(.854,.066),(.195,.066),(.153,.469)])|path([(.153,.471),(.41,.337),(.91,.372),(.90,.64),(.938,1.001),(.29,1.046),(.098,.841)],True)
    if d=='6':
        if style==2:return ring(.709,.47,.295,2.12)|path([(.68,.07),(.48,.224),(.20,.492),(.098,.67)],True)
        arch=[(.825,.12),(.53,-.05),(.13,.10),(.119,.48),(.09,.621),(.11,.735),(.13,.77)]
        if style==1:arch=[(.647,.067),(.49,.175),(.23,.392),(.129,.575),(.087,.656),(.102,.736),(.14,.79)]
        return ring(.697,.465,.30,2.25)|path(arch,True)
    if d=='7':return path([(.083,.068),(.905,.068),(.314,.93)])
    raise ValueError(d)

speeds={0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}

def digits(x,y,text,style):
    n=len(text); h=2.; w=(1.46 if n==2 else 1.15); gap=.17
    widths=[w*(.64 if c=='1' else 1) for c in text]
    left=-(sum(widths)+gap*(n-1))/2
    out=np.zeros(x.shape,bool)
    for wd,d in zip(widths,text):
        out |= numeral((x-left)/wd,(y+1)/h,d,style); left+=wd+gap
    return out

def arrow(x,y,k,style):
    if k in (34,37,39):x=-x
    p=lambda a:piece(x,y,a)
    l=lambda a,w:pencil(x,y,a,w)
    if k in (38,39):
        # Rotation of a straight arrow into the lower right quadrant.
        x,y=(x-y)*.707107,-(x+y)*.707107
        return p([(-.175,.87),(.175,.87),(.175,-.14),(.58,-.14),(0,-.98),(-.58,-.14),(-.175,-.14)])
    if k in (33,34):
        if style==1:y=y+.10
        if style==2:y=y-.14
        path=bezier([(-.49,.80),(-.49,.26),(-.55,-.28),(.10,-.28)])
        return l(path+[(.28,-.28)],.34)|p([(.14,-.78),(.94,-.28),(.14,.245)])|p([(-.66,.57),(-.32,.57),(-.32,.94),(-.66,.94)])
    vert=p([(-.145,.95),(.145,.95),(.145,-.41),(-.145,-.41)])
    up=p([(-.495,-.27),(0,-.97),(.49,-.27),(.29,-.22),(0,-.56),(-.29,-.22)]) if style==1 else p([(-.49,-.30),(0,-.98),(.49,-.30),(.32,-.19),(0,-.50),(-.32,-.19)])
    if style==2: up=p([(-.49,-.28),(0,-.97),(.49,-.28)])
    if k==35:return vert|up|p([(-.14,-.67),(.14,-.67),(.14,-.39),(-.14,-.39)])
    if k in (36,37):
        if style==1:
            shaft=p([(-.22,-.58),(.06,-.58),(.06,.93),(-.22,.93)])
            branch=p([(-.11,-.03),(.45,-.03),(.45,-.30),(.99,.12),(.45,.56),(.45,.27),(-.11,.27)])
            top=p([(-.55,-.31),(-.08,-.98),(.39,-.31),(.23,-.17),(-.08,-.52),(-.39,-.17)])
            return shaft|branch|top
        return vert|up|p([(-.13,-.66),(.13,-.66),(.13,-.39),(-.13,-.39)])|p([(0,-.01),(.49,-.01),(.49,-.31),(.95,.126),(.49,.54),(.49,.265),(0,.265)])
    # Three arrows that circulate counterclockwise, tips tangent to the circle.
    out=np.zeros(x.shape,bool)
    for off in (0,120,240):
        off+=12*style
        t=np.deg2rad(np.linspace(74+off,0+off,16)); px=.736*np.cos(t); py=.736*np.sin(t)
        out|=l(list(zip(px,py)),.186)
        t=t[-1]; u,v=np.sin(t),-np.cos(t); a,b=px[-1],py[-1]
        out|=p([(a+.31*u,b+.31*v),(a-.09*u+.24*v,b-.09*v-.24*u),(a-.09*u-.24*v,b-.09*v+.24*u)])
    return out

# Silhouettes for the regulatory signs.
def frontcar(x,y):
    return piece(x,y,[(-.95,.12),(-.9,-.16),(-.74,-.22),(-.53,-.73),(.53,-.73),(.74,-.22),(.91,-.16),(.95,.13),(.95,.57),(.80,.60),(.80,.79),(.51,.79),(.50,.58),(-.5,.58),(-.51,.79),(-.80,.79),(-.80,.59),(-.95,.57)]) & ~piece(x,y,[(-.56,-.19),(-.423,-.561),(.425,-.561),(.558,-.19)])

def passing(x,y,truck,style):
    if truck:
        a,b=(x+.88)/.70,(y+.10)/.88
        red=piece(a,b,[(-.94,-.88),(.87,-.88),(.87,.19),(.79,.24),(.79,.76),(.54,.76),(.48,.57),(-.53,.57),(-.6,.77),(-.82,.77),(-.82,.2),(-.94,.2)])
        black=frontcar((x-.77)/.73,(y-.23)/.78)
    else:
        red=frontcar((x+.83)/.75,(y-.01)/.79); black=frontcar((x-.82)/.75,(y-.01)/.79)
    return red,black

def lorry(x,y,style):
    if style==1:x=-x
    return piece(x,y,[(-1.54,.45),(-1.54,-.03),(-1.12,-.61),(-.64,-.61),(-.64,-.81),(1.49,-.81),(1.49,.43),(1.24,.46),(-1.32,.46)])|oval(x,y,-1.04,.48,.255,.29)|oval(x,y,.97,.48,.25,.29)

# Local coordinates for warning symbols: x is centered on the vertical axis, y on
# the symbol, rather than on the bounding box of the surrounding warning triangle.
def warning(x,y,k,style):
    p=lambda pts:piece(x,y,pts)
    l=lambda pts,w:pencil(x,y,pts,w)
    e=lambda a,b,c,d:oval(x,y,a,b,c,d)
    z=np.zeros(x.shape,bool)
    if k==11:
        u=(.16,.20,.23)[style]; v=(.47,.57,.64)[style]
        return p([(-u,.96),(u,.96),(u,.09),(v,.09),(0,-.99),(-v,.09),(-u,.09)])
    if k==18:
        return p([(-.16,-.96),(.16,-.96),(.103,.43),(-.103,.43)])|e(0,.79,.155,.17)
    if k in (19,20):
        if k==20:x=-x
        if style!=0:
            t=.12 if style==1 else .37
            path=bezier([(.34,.96),(.34,.43),(.43,-.11),(-.11,-.20),(-.26,-.23),(-.47,-.22),(-.57,-.22)])
            return l(path,.33)|p([(-.42,-.22-t),(-.87,-.22),(-.42,-.22+t)])
        path=bezier([(.27,.91),(.27,.49),(.34,-.04),(-.04,-.16),(-.17,-.22),(-.30,-.22),(-.43,-.22)])
        return l(path,.34)|p([(-.32,-.68),(-.94,-.21),(-.32,.19)])|p([(.10,.59),(.44,.59),(.44,.99),(.10,.99)])
    if k==21:
        if style==2:x=-x
        path=bezier([(.37,.91),(.37,.60),(.47,.11),(.10,.07),(-.05,.05),(-.19,.12),(-.20,-.12),(-.21,-.31),(-.20,-.39),(-.20,-.49)])
        return l(path,.32)|p([(-.58,-.37),(-.20,-.99),(.19,-.37)])
    if k==22:
        if style==1:return p([(-1.5,.7),(-1.34,.67),(-1.04,.43),(-.65,.17),(-.30,.04),(0,0),(.3,.04),(.65,.17),(1.04,.43),(1.34,.67),(1.5,.7),(1.5,.92),(-1.5,.92)])
        return p([(-1.52,.74),(-1.31,.65),(-1.09,.32),(-.89,.18),(-.67,.16),(-.47,.23),(-.19,.60),(0,.66),(.16,.58),(.45,.24),(.66,.16),(.91,.18),(1.10,.33),(1.35,.66),(1.54,.74),(1.54,.93),(-1.52,.93)])
    if k==23:
        # Skidding automobile above two serpentine tyre tracks.
        c,s=np.cos(.43),np.sin(.43); a=x*.96;b=y+.33
        car=frontcar((c*a+s*b)/.64,(-s*a+c*b)/.71)
        for sh in (-.36,.36):
            path=bezier([(sh-.19,-.03),(sh+.43,.11),(sh+.51,.30),(sh+.04,.44),(sh-.64,.67),(sh-.69,.79),(sh-.18,.98)])
            z|=l(path,.13)
        return car|z
    if k==24:
        return p([(-.69,-.9),(-.44,-.9),(-.44,.95),(-.69,.95)])|p([(.06,-.9),(.31,-.9),(.32,-.37),(.38,-.20),(.66,.15),(.70,.30),(.70,.95),(.45,.95),(.45,.38),(.41,.25),(.14,-.095),(.065,-.28)])
    if k==25:
        if style==2:x=-x
        heap=p([(.06,.92),(.29,.67),(.44,.62),(.63,.33),(.82,.28),(1.06,.44),(1.37,.88),(1.55,.95)])
        return heap|e(-.09,-.67,.20,.20)|l([(-.27,-.46),(-.55,-.12),(-.71,.18)],.34)|l([(-.62,.09),(-.76,.51),(-1.02,.93)],.222)|l([(-.56,.08),(-.36,.40),(-.52,.89)],.20)|l([(-.34,-.36),(-.21,-.02),(.12,.24)],.165)|l([(-.91,-.29),(.86,.78)],.083)
    if k==26:
        main=p([(-.29,-.97),(.29,-.97),(.29,.93),(-.29,.93)])
        if style==1:return e(0,-.61,.25,.26)|e(0,0,.25,.26)|e(0,.61,.25,.26)
        if style==2:return main*.85-.52*(e(0,-.61,.215,.23)|e(0,0,.215,.23)|e(0,.61,.215,.23))
        return main
    if k==27 or k==28:
        def person(a,b,sz,lean):
            # a walking person, seen in profile
            return (e(a+.12*sz*lean,b-.76*sz,.17*sz,.19*sz)|l([(a+.05*sz*lean,b-.49*sz),(a-.18*sz*lean,b+.17*sz)],.305*sz)|l([(a-.14*sz*lean,b+.12*sz),(a-.35*sz*lean,b+.51*sz),(a-.69*sz*lean,b+.88*sz)],.19*sz)|l([(a-.14*sz*lean,b+.13*sz),(a+.21*sz*lean,b+.46*sz),(a+.45*sz*lean,b+.89*sz)],.185*sz)|l([(a+.00*sz*lean,b-.38*sz),(a+.28*sz*lean,b-.09*sz),(a+.58*sz*lean,b+.03*sz)],.146*sz)|l([(a-.05*sz*lean,b-.37*sz),(a-.40*sz*lean,b-.21*sz),(a-.57*sz*lean,b+.11*sz)],.135*sz))
        if k==27:return person(.02,.0,1.04,1 if style==2 else -1)
        return person(.41,-.03,1.04,-1)|person(-.58,.30,.66,-1)
    if k==29:
        for a in (-.80,.77):z|=e(a,.45,.50,.50)&~e(a,.45,.391,.39)
        return z|l([(-.80,.45),(-.41,-.28),(.01,.45),(-.80,.45)],.099)|l([(.01,.45),(.49,-.32),(-.35,-.14)],.106)|l([(.77,.45),(.44,-.47),(.62,-.49)],.114)|l([(-.51,-.29),(-.25,-.29)],.13)|l([(-.05,-.33),(-.35,-.56),(.0,-.88)],.223)|e(.16,-.91,.16,.17)|l([(-.21,-.55),(.10,-.34),(.47,-.39)],.132)
    if k==30:
        for t in np.arange(6)*np.pi/3-np.pi/2+style*np.pi/12:
            a,b=np.cos(t),np.sin(t);z|=l([(0,0),(.98*a,.98*b)],.106)
            for r in (.42,.71):
                for s in (-1,1):z|=l([(r*a,r*b),((r+.19)*a+s*.17*b,(r+.19)*b-s*.17*a)],.086)
        return z
    if k==31:
        if style==2:x=-x
        return p([(-.94,-.35),(-.78,-.45),(-.70,-.65),(-.49,-.62),(-.30,-.37),(.34,-.15),(.71,-.13),(.95,.06),(.82,.23),(.47,.25),(.15,.18),(-.30,.06),(-.55,-.19),(-.64,-.30),(-.82,-.28)])|l([(.80,.11),(.89,.40),(1.25,.68),(1.35,.88)],.135)|l([(.68,.18),(.60,.40),(.92,.68)],.113)|l([(-.30,-.02),(-.37,.21),(-.69,.36),(-.78,.65)],.13)|l([(-.53,-.46),(-.39,-.78),(-.61,-.98)],.08)|l([(-.39,-.78),(-.28,-.89)],.065)
    raise ValueError(k)

# Hand-specified alternatives for the shapes vary stroke, aspect and alignment.
tri=[11]+list(range(18,32))
blues=list(range(33,41))
# Region sampled for local matching (scale relative to half width/height of sign).
# x spans [-1,1] in every crop; inside our drawing x is expanded for wide symbols.
def template_bank(classes,group):
    xy=(np.arange(S*Q)+.5)/(S*Q)*2-1
    xx,yy=np.meshgrid(xy,xy)
    allv=[]; labs=[]; offsets=[]
    for k in classes:
        for style in range(3):
            for scale,aspect in [(1,1),(.84,1),(1.15,1),(.97,.80),(1,1.18),(.84,1.16),(1.14,.85)]:
                for xoff,yoff,tilt in [(0,0,0),(-.079,0,0),(.079,0,0),(0,-.105,0),(0,.105,0),(-.055,-.069,.10),(.055,.069,-.10)]:
                    # For each pose render independently. Geometric transforms are
                    # sampled from a fine master of *this particular symbol* below.
                    xx0=(xx-xoff-tilt*yy)/scale/aspect; yy0=(yy-yoff)/scale
                    if group=='circ':
                        # Physical ROI .68 by .50, typical digit height .77 and width 1.10 in sign-radius units
                        x=xx0*1.98;y=yy0*1.32
                    elif group=='warn':x=xx0*1.85;y=yy0*1.20
                    else:x=xx0*1.09;y=yy0*1.09
                    if group=='blue' and k in (38,39,40):
                        theta=(style-1)*.16; x,y=x*np.cos(theta)+y*np.sin(theta),y*np.cos(theta)-x*np.sin(theta)
                    if group=='warn' and k in (23,27,28,29,30,31):
                        theta=(style-1)*.11;x,y=x*np.cos(theta)+y*np.sin(theta),y*np.cos(theta)-x*np.sin(theta)
                    if len(offsets)==0 or offsets[-1]!=(k,style):
                        # Rasterize once at much higher resolution than the final template.
                        w=320; a=(np.arange(w)+.5)/w*6-3; ax,ay=np.meshgrid(a,a)
                        if k in speeds: ink=digits(ax,ay,speeds[k],style)
                        elif k in (9,10):
                            red,black=passing(ax,ay,k==10,style);ink=red*.68+black*1.
                        elif k==16:ink=lorry(ax,ay,style)
                        elif group=='warn':ink=warning(ax,ay,k,style)
                        else:ink=arrow(ax,ay,k,style)
                        master=ink.astype(np.float32);offsets.append((k,style))
                    ix=np.clip(((x+3)*w/6).astype(int),0,w-1);iy=np.clip(((y+3)*w/6).astype(int),0,w-1)
                    shape=master[iy,ix].reshape(S,Q,S,Q).mean(axis=(1,3));allv.append(shape);labs.append(k)
    return np.array(allv,np.float32),np.array(labs,np.int32)

RUNTIME=r'''import numpy as np

# A coarse generative detector estimates the sign frame. Coordinates in these
# tables are independent of the images: sectors, strips and quadrature points.
fkind=[0,1,2,3,4,5,6,7,8]

def curve(kind,n):
    if kind==2: pts=[(0,-1),(1,1),(-1,1)]
    elif kind==3:pts=[(-1,-1),(1,-1),(0,1)]
    elif kind==4:pts=[(0,-1),(1,0),(0,1),(-1,0)]
    elif kind==5:pts=[(-.415,-1),(.415,-1),(1,-.415),(1,.415),(.415,1),(-.415,1),(-1,.415),(-1,-.415)]
    else:
        t=(np.arange(n)+.5)*2*np.pi/n
        return np.stack([np.cos(t),np.sin(t)],1)
    p=np.array(pts);s=(np.arange(n)+.5)*len(p)/n;j=s.astype(int);f=s-j
    return p[j]*(1-f[:,None])+p[(j+1)%len(p)]*f[:,None]

def weightspec(kind):
    q=curve(kind,36); m=np.array([0, .3333 if kind==2 else (-.3333 if kind==3 else 0)])
    def contours(sc):return np.concatenate([m+(q-m)*s for s in sc])
    a=contours([.81,.92]); b=contours([.61]); c=contours([.955]); d=contours([1.115]); inner=contours([.49,.68])
    def strip(ar,br):
        xs=np.linspace(-ar,ar,10);ys=np.array(br);xx,yy=np.meshgrid(xs,ys);return np.stack([xx.ravel(),yy.ravel()],1)
    def mean(p,ch,w):return [(ch,p,w)]
    def dif(p,q,ch,w):return mean(p,ch,w)+mean(q,ch,-w)
    # feature channels: redness, blueness, yellowness, red-green luminance,
    # green luminance, and gradient magnitude. Equal weights within each stencil.
    if kind in (0,2,3):
        # White annulus just inside the red margin; center may contain a symbol.
        w=mean(a,0,.66)+dif(a,b,0,.64)+dif(a,d,0,.24)+dif(b,a,4,.70)+dif(b,d,3,.16)+mean(c,5,.10)
        if kind==3:w+=mean(inner,4,.09)+dif(inner,a,4,.13)
        return w
    if kind==1:
        a=contours([.72,.87]);b=contours([.39,.58]);return mean(a,1,.92)+dif(a,d,1,.46)+mean(b,1,.22)+dif(contours([.89]),d,3,.16)+mean(c,5,.19)
    if kind==4:
        gold=contours([.25,.50,.67]); white=contours([.81,.91]);return mean(gold,2,.76)+dif(gold,white,2,.63)+dif(white,gold,4,.27)+dif(white,d,3,.35)+mean(c,5,.14)
    if kind in (5,6):
        red=contours([.65,.81]);border=contours([.91,.98]);mid=strip(.63,[-.085,.03,.116]); sides=strip(.58,[-.55,-.38,.38,.54]);
        w=mean(red,0,.77)+dif(red,d,0,.39)+mean(sides,0,.33)+dif(border,red,4,.19)+dif(mid,sides,4,.19)+mean(c,5,.10)
        if kind==6:w+=dif(mid,sides,4,.55)+dif(sides,mid,0,.29)
        else:w+=mean(red,0,.04)
        return w
    # End of restriction: a dark descending slash (in Cartesian coordinates)
    # on a light, usually unsaturated circular face. Some variants have a
    # sheaf of thin parallel lines and some have pale grey pictograms.
    r=np.linspace(-.76,.76,23); slash=np.stack([r*.707,-r*.707],1)
    u=np.array([.707,.707]);mid=np.concatenate([slash+s*u for s in [-.088,0,.088]])
    side=np.concatenate([slash+s*u for s in [-.38,.38,-.53,.53]])
    body=contours([.59,.79,.90]);w=dif(side,mid,3, .76 if kind==7 else .45)+dif(body,d,3,.48)+mean(c,5,.18)
    w+=mean(body,0,-.24)+mean(body,1,-.22)+mean(side,4,.16)
    return w

# Bilinear-splat the quadrature stencil onto a 16x16 grid for the fast search.
def makegeom():
    output=[]; poses=[]
    for k in fkind:
        specs=weightspec(k);allw=[];allp=[]
        for rad in (9.5,11.2,13.1,15.15):
            for dx,dy in [(0,0),(-1.65,0),(1.65,0),(0,-1.65),(0,1.65),(-1.35,-1.35),(1.35,-1.35),(-1.35,1.35),(1.35,1.35)]:
                for aspect,shear in [(1,0),(.84,0),(1.16,0),(1,-.13),(1,.13)]:
                    rx=rad*aspect**.5;ry=rad/aspect**.5;cx=15.5+dx;cy=15.5+dy
                    if k==2 or k==3:ry*=.96
                    allp.append([cx,cy,rx,ry,shear,0]);w=np.zeros((6,16,16),np.float32)
                    for ch,p,gain in specs:
                        u=np.clip((cx+rx*(p[:,0]+shear*p[:,1])-.5)*.5,0,14.9999);v=np.clip((cy+ry*p[:,1]-.5)*.5,0,14.9999)
                        i=u.astype(int);j=v.astype(int);u=u-i;v=v-j
                        for a in (0,1):
                            for b in (0,1):np.add.at(w[ch],(j+a,i+b),(gain/len(p))*(v if a else 1-v)*(u if b else 1-u))
                    allw.append(w.ravel())
        output.append(np.array(allw));poses.append(np.array(allp,np.float32))
    return np.concatenate(output),np.array(poses)

def smooth(z):
    a=np.pad(z,[(0,0),(1,1),(1,1)],mode='edge')
    b=(a[:,:-2]+2*a[:,1:-1]+a[:,2:])*.25
    return (b[:,:,:-2]+2*b[:,:,1:-1]+b[:,:,2:])*.25

def sortpart(x,lo,hi):
    a=x.reshape(len(x),-1); ll=int(a.shape[1]*lo);hh=int(a.shape[1]*hi);a=np.partition(a,(ll,hh),axis=1);return a[:,ll,None,None],a[:,hh,None,None]

def relative(x,lo=.07,hi=.93,eps=.04):
    l,h=sortpart(x,lo,hi);return np.clip((x-l)/(h-l+eps),-.44,1.44)

def channels(x):
    r,g,b=x[:,0],x[:,1],x[:,2];v=(r*.285+g*.535+b*.18)
    # Log ratios suppress shadows. Also contrast-center each opponent channel
    # on the crop to suppress a global colour cast without removing red fill.
    lr=np.log(r+.019);lg=np.log(g+.019);lb=np.log(b+.019)
    r1=lr-.72*lg-.28*lb; b1=.79*(lb-lr)+.21*(lb-lg);y1=.52*lr+.48*lg-lb-.37*abs(lr-lg)
    def opp(z,gain):
        l,h=sortpart(z[:,4:28,4:28],.12,.90)
        return np.clip(.52*np.clip((z-l)/(h-l+.21),-.50,1.45)+.48*np.clip(z*gain,-.55,1.45),-.45,1.45)
    red=opp(r1,1.6);blue=opp(b1,1.6);gold=opp(y1,1.7)
    # Use the central part of the image as a second exposure meter: a bright
    # sky or a deep shadow outside the sign needn't set its contrast scale.
    def dual(z):
        lo,hi=sortpart(z[:,5:27,5:27],.06,.94)
        return .62*np.clip((z-lo)/(hi-lo+.055),-.50,1.50)+.38*relative(z)
    lum=dual(np.log(v+.018));gl=dual(np.log(g+.016));gx,gy=grad(lum);edge=np.minimum(np.sqrt(gx*gx+gy*gy)*2.5,1.5)
    # The low red response of a blue reflector remains useful when its blue
    # channel is saturated; it also preserves contrast of black-on-white print.
    rl=dual(lr)
    return np.stack([red,blue,gold,lum,gl,edge,np.clip(r1*1.7,-1.4,1.4),np.clip(b1*1.7,-1.4,1.4),rl],axis=1).astype(np.float32)

def grad(x):
    a=np.pad(x,[(0,0),(1,1),(1,1)],mode='edge')
    return (a[:,1:-1,2:]-a[:,1:-1,:-2])*.5,(a[:,2:,1:-1]-a[:,:-2,1:-1])*.5

def coarse(x):
    n=len(x);vec=x[:,:6].reshape(n,6,16,2,16,2).mean(axis=(3,5)).reshape(n,-1)
    a=vec@CONSTANTS['geom'].T;a=a.reshape(n,len(fkind),-1)
    pp=CONSTANTS['poses'];j=np.argmax(a,axis=2)
    return a.max(axis=2),pp[np.arange(len(fkind))[None,:],j]

def bilinear(x,u,v):
    # x: B,H,W; u,v: B,... same output spatial dimensions
    u=np.clip(u,0,30.99999);v=np.clip(v,0,30.99999);i=u.astype(np.int32);j=v.astype(np.int32);u=u-i;v=v-j
    row=np.arange(len(x)).reshape([len(x)]+[1]*(u.ndim-1));a=x[row,j,i];b=x[row,j,i+1];c=x[row,j+1,i];d=x[row,j+1,i+1]
    return ((a*(1-u)+b*u)*(1-v)+(c*(1-u)+d*u)*v).astype(np.float32)

def uv(p,X,Y):
    shape=[len(p)]+[1]*X.ndim;c,d,w,h,s,t=[p[:,j].reshape(shape) for j in range(6)]
    return c+w*(X+s*Y),d+h*(Y+t*X)

def view(x,p,X,Y):
    u,v=uv(p,X,Y);return bilinear(x,u,v)

def refine(ch,k,p):
    # Coordinate search of edge/color stencils at full resolution. Use two
    # rounds of small alternatives rather than a free bounding box extraction.
    n=len(ch);spec=SPEC[k];base=p.copy()
    for steps in ([(0,0,0,0,0,0),(-.90,0,0,0,0,0),(.90,0,0,0,0,0),(0,-.90,0,0,0,0),(0,.90,0,0,0,0),(0,0,-.75,-.75,0,0),(0,0,.75,.75,0,0)],[(0,0,0,0,0,0),(0,0,-.65,.65,0,0),(0,0,.65,-.65,0,0),(0,0,0,0,-.08,0),(0,0,0,0,.08,0),(0,0,0,0,0,-.085),(0,0,0,0,0,.085)]):
        off=np.array(steps,np.float32);ps=(p[:,None,:]+off[None,:,:]);sc=np.zeros((n,len(off)),np.float32)
        a=ps[:,:,:,None];cx,cy,rx,ry,sh,sl=[a[:,:,j] for j in range(6)]
        for j,pts,wt in spec:
            X,Y=pts[:,0],pts[:,1];u=cx+rx*(X+sh*Y);v=cy+ry*(Y+sl*X)
            sc+=(bilinear(ch[:,j],u,v)*wt).sum(axis=-1)
        p=ps[np.arange(n),sc.argmax(1)]
    return p,sc.max(axis=1)

# Consolidate the quadrature points by channel once, without using image data.
SPEC=[]
for kind in fkind:
    spec=weightspec(kind);pieces=[]
    for chan in range(6):
        pair=[(q,w) for ci,q,w in spec if ci==chan];
        if pair:pieces.append((chan,np.concatenate([q for q,w in pair]).astype(np.float32),np.concatenate([np.full(len(q),w/len(q),np.float32) for q,w in pair])))
    SPEC.append(pieces)

S=24
axis=(np.arange(S,dtype=np.float32)+.5)*2/S-1
X,Y=np.meshgrid(axis,axis)

def units(a):
    a=a.reshape(len(a),-1); return a/np.sqrt((a*a).sum(axis=1,keepdims=True)+1e-7)

def innerfeatures(ink,group):
    # Normalized correlation in the sign interior, plus local center-surround
    # filters and pooled directed edges. The same map is used for a single
    # template and for a rectified photo; there is no population normalization.
    if group=='part':mask=np.clip((1.16-(X*.80)**2-(Y*.71)**2)*4,0,1)
    elif group=='circ':mask=np.clip((1.13-(X*.89)**2-(Y*.76)**2)*5,0,1)
    elif group=='warn':mask=np.clip((1.38+.47*Y-abs(X)*1.46)*5,0,1)
    else:mask=np.clip((1.62-X**2-Y**2)*4,0,1)
    z=smooth(ink); mean=(z*mask).sum(axis=(1,2),keepdims=True)/mask.sum()
    z=(z-mean)*mask
    detail=z;low=smooth(smooth(smooth(smooth(z)))); dog=(detail-low)*mask
    a,b=grad(z); ed=[]
    for c,d in [(1,0),(0,1),(.707,.707),(.707,-.707)]:
        r=(a*c+b*d)*mask
        for sg in (1,-1):
            s=np.maximum(sg*r,0).reshape(len(z),6,4,6,4).mean(axis=(2,4));ed.append(s)
    edge=np.stack(ed,1);edge=edge-edge.mean(axis=(2,3),keepdims=True)
    def pool2(a):return a.reshape(len(z),12,2,12,2).mean(axis=(2,4))
    return np.concatenate([units(pool2(z))*.50,units(pool2(dog))*.25,units(edge)*.25],axis=1).astype(np.float32)

circ=[0,1,2,3,4,5,7,8,9,10,16];warn=[11]+list(range(18,32));bluecls=list(range(33,41))

def pictogram(ink,group,classes):
    if group=='part':mask=((X*.81)**2+(Y*.75)**2<1)
    elif group=='circ': mask=((X*.87)**2+(Y*.80)**2<.95)
    elif group=='warn':mask=(abs(X)<.79+.45*Y)
    else:mask=(X**2+Y**2<1.32)
    l,h=sortpart(ink[:,mask],.10,.93);amp=h-l
    # Soft saturation reduces the influence of small, extremely bright specular highlights.
    z=np.clip((ink-l)/(amp+.024),-.24,1.19)
    f=innerfeatures(z,group);v=dotstable(f,CONSTANTS[group]);v=v.reshape(len(ink),len(classes),-1).max(axis=2)
    return v,amp[:,0,0]

def dotstable(a,w):
    n=len(a); q=(-n)%16
    if q:a=np.concatenate([a,np.zeros((q,a.shape[1]),np.float32)])
    return np.concatenate([a[i:i+16]@w.T for i in range(0,len(a),16)])[:n]

def region(a,mask):return np.mean(a[:,mask],axis=1)

def cropchannels(ch,p,idx,X,Y):
    u,v=uv(p,X,Y);return [bilinear(ch[:,j],u,v) for j in idx]

def weightedquant(q,axis,frac):
    q=np.cumsum(q,axis=1);target=q[:,-1]*frac
    j=np.minimum((q<target[:,None]).sum(axis=1),len(axis)-1)
    return axis[j]

def framedigit(ink):
    # Separate the ink from the white field in a two-dimensional window,
    # then align the printed word by its ink projection rather than assuming
    # a particular German sign diameter, spacing, or font size.
    mask=(X*X+.70*Y*Y<1.12);l,h=sortpart(ink[:,mask],.21,.91);thr=l+.47*(h-l)
    inkpos=np.minimum(np.maximum(ink-thr,0),1.*(h-l)+.005)*mask
    px=inkpos.sum(axis=1);py=inkpos.sum(axis=2);ll=weightedquant(px,axis,.027);rr=weightedquant(px,axis,.973);tt=weightedquant(py,axis,.037);bb=weightedquant(py,axis,.963)
    # Quantiles are at pixel centers; account for the truncated ends of the strokes.
    cx=np.clip((rr+ll)*.5,-.20,.20);cy=np.clip((tt+bb)*.5,-.26,.26)
    w=np.clip((rr-ll)*.5+.070,.54,.99);h=np.clip((bb-tt)*.5+.085,.51,.98)
    return cx,cy,w,h

def childpose(p,cx,cy,w,h):
    q=p.copy();q[:,0]+=p[:,2]*(cx+p[:,4]*cy);q[:,1]+=p[:,3]*(cy+p[:,5]*cx);q[:,2]*=w;q[:,3]*=h
    q[:,4]*=h/w;q[:,5]*=w/h
    return q

def glyphread(ch,p,cx,cy,hx,hy):
    q=childpose(p,cx,cy,hx,hy);u,v=cropchannels(ch,q,[3,4],X,Y)
    return pictogram(-.70*u-.30*v,'part',range(7))[0]

def circscore(ch,p):
    l,g,r=cropchannels(ch,p,[3,4,0],.68*X,.50*Y);ink=-.70*l-.30*g
    s,amp=pictogram(ink,'circ',circ)
    # Recognize the leading character independently of the trailing zero.
    # In a three-character limit it is the middle character that disambiguates.
    cx,cy,w,h=framedigit(ink)
    a,b=cropchannels(ch,childpose(p,.68*cx,.5*cy,.68*w/.82,.50*h/.76),[3,4],X,Y)
    # The rectified crop above already has its physical half extents in the pose.
    order,aa=pictogram(-.70*a-.30*b,'circ',circ)
    s=np.maximum(s,.17*s+.83*order-.015)
    part=glyphread(ch,p,.68*(cx-.52*w),.50*cy,.68*w*.61,.50*h*1.22)
    mid=glyphread(ch,p,.68*(cx-.13*w),.50*cy,.68*w*.445,.50*h*1.22)
    s[:,:6]+=.58*(part[:,1:7]-np.mean(part[:,1:7],axis=1,keepdims=True))
    s[:,6]+=.44*(mid[:,0]-mid[:,1]);s[:,7]+=.44*(mid[:,1]-mid[:,0])
    rr=region(r,(X<-.09)&(X>-.86)&(abs(Y)<.47))-region(r,(X>.08)&(X<.89)&(abs(Y)<.47))
    s[:,8:10]+=np.clip(rr[:,None],-.19,.53)*.32
    strength=np.clip((amp-.09)/.44,0,1);s*=.67+.33*strength[:,None]
    val=np.max(s,axis=1);c=np.array(circ)[np.argmax(s,axis=1)]
    # Empty center: compare the darkest of the central pixels with the white
    # field, relative to the outer border contrast. The percentile excludes specks.
    a,b=sortpart(ink[:,(X*X+.80*Y*Y)<.88],.12,.95);rng=(b-a).ravel();blank=np.clip(.66-1.21*rng,.04,.68)
    return np.where(blank>val,15,c),np.maximum(blank,val)

def figcenter(ink,mask,qx,qy,xmax,ymax):
    a,b=sortpart(ink[:,mask],.25,.92);w=np.clip((ink-a)/(b-a+.034)-.44,0,.66)*mask
    px=w.sum(axis=1);py=w.sum(axis=2)
    cx=np.clip((weightedquant(px,axis,qx)+weightedquant(px,axis,1-qx))*.5,-xmax,xmax)
    cy=np.clip((weightedquant(py,axis,qy)+weightedquant(py,axis,1-qy))*.5,-ymax,ymax)
    return cx,cy

def warnscore(ch,p):
    l,g,r,b=cropchannels(ch,p,[3,4,0,1],.641*X,.47*Y+.27);ink=-.65*l-.35*g
    s,amp=pictogram(ink,'warn',warn)
    mask=(abs(X)<.69+Y*.34)&(abs(Y)<.9)
    dx,dy=figcenter(ink,mask,.04,.04,.22,.22);p2=childpose(p,dx*.641,(dy-.06)*.47,1,1)
    ll,gg=cropchannels(ch,p2,[3,4],.641*X,.47*Y+.27);s2,aa=pictogram(-.65*ll-.35*gg,'warn',warn);s=np.maximum(s,s2-.016)
    # The slippery-road sign and bicycle have important structure below a
    # broad midsection; conversely the junction arrow has a single narrow stem.
    a,h=sortpart(ink[:,mask],.15,.91);w=np.clip((ink-a)/(h-a+.025)-.34,0,.77)*mask
    center=region(w,(abs(X)<.13)&(Y>.40)&(Y<.80));shoulder=region(w,(abs(X)>.20)&(abs(X)<.55)&(Y>.40)&(Y<.80))
    s[:,0]+=.13*(center-shoulder-.22)
    # Traffic signals have coloured lights along their vertical center line.
    spot=(abs(X)<.19); upper=region(r,spot&(Y<-.22)&(Y>-.82)); sides=region(r,(abs(X)>.24)&(abs(X)<.43)&(abs(Y)<.39));bot=region(b,spot&(Y>.12)&(Y<.82))-region(b,(abs(X)>.27)&(abs(X)<.46)&(Y>.12)&(Y<.82))
    s[:,warn.index(26)]+=.24*np.clip(upper-sides,-.20,.70)+.12*np.clip(bot,-.14,.60)
    return np.array(warn)[np.argmax(s,axis=1)],np.max(s,axis=1)

def bluescore(ch,p):
    l,b,r=cropchannels(ch,p,[3,1,8],.78*X,.78*Y);ink=.40*l+.49*r-.28*b;s,amp=pictogram(ink,'blue',bluecls)
    dx,dy=figcenter(ink,X*X+Y*Y<1.32,.025,.025,.22,.22);q=childpose(p,dx*.78,dy*.78,1,1)
    l,b,r=cropchannels(ch,q,[3,1,8],.78*X,.78*Y);s2,aa=pictogram(.40*l+.49*r-.28*b,'blue',bluecls);s=np.maximum(s,s2-.012)
    return np.array(bluecls)[np.argmax(s,axis=1)],np.max(s,axis=1)

def ends(ch,p):
    # Measure residual ink on either side of the cancellation band. Fit only
    # the visible parts of several cancelled-symbol models, not the stripe itself.
    l,g=cropchannels(ch,p,[3,4],.68*X,.50*Y);ink=-.8*l-.2*g; bs=[]
    diag=(.68*X+.50*Y)*.707; along=(.68*X-.5*Y)*.707
    for i in range(3):
        good=(abs(diag)>(.15+.088*i))&((.68*X)**2+(.5*Y)**2<.54)
        low,high=sortpart(ink[:,good],.12,.91);v=np.clip((ink-low)/(high-low+.03),-.2,1.25);v=v[:,good];v=units(v-v.mean(axis=1,keepdims=True))
        match=dotstable(v,CONSTANTS['end'+str(i)]).reshape(len(ch),3,-1).max(axis=2)
        amp=(high-low).ravel();match*=np.clip(amp[:,None]/.28,.2,1)
        bs.append(match)
    score=np.maximum.reduce(bs);best=score.argmax(1);sv=score.max(axis=1)
    return np.where(sv>.43,np.array([6,41,42])[best],32),sv

def verifyface(ch,p,k):
    # Regional constraints that are poor linear filters: uniformity around a
    # contour, sign of figure/ground contrast, and presence of the STOP word.
    q=curve(k,48).astype(np.float32);m=np.array([0,.3333 if k==2 else (-.3333 if k==3 else 0)],np.float32)
    def sample(scale,j):
        a=m+(q-m)*scale
        return view(ch[:,j],p,a[:,0],a[:,1])
    def robust(a):return np.sort(a.reshape(len(ch),8,-1).mean(axis=2),axis=1)[:,1:7].mean(axis=1)
    def corr(a,b):
        a=a-a.mean(axis=1,keepdims=True);b=b-b.mean(axis=1,keepdims=True)
        return (a*b).mean(axis=1)/np.sqrt(((a*a).mean(axis=1)+.015)*((b*b).mean(axis=1)+.015))
    if k in (0,2,3):
        red=sample(.86,0)-sample(.58,0);white=sample(.58,4)-sample(.86,4)
        # A white face becomes brighter as it gets less red. A black numeral
        # stays dark in *all* RGB channels, unlike the red surround.
        support=robust(.51*red+.64*white)
        return .42*np.clip(support,-.50,1)+.29*robust(sample(.86,6))
    if k==1:
        fill=sample(.73,1);l,b=cropchannels(ch,p,[3,1],X*.72,Y*.72)
        mask=X*X+Y*Y<1.34
        return .22*robust(fill)-.17*corr(l[:,mask],b[:,mask])+.37*robust(sample(.77,7))
    if k in (5,6):
        l,r=cropchannels(ch,p,[4,0],X*.75,Y*.75)
        row=abs(Y)<.23;flank=(abs(Y)>.4)&(abs(Y)<.75)&(abs(X)<.77)
        stroke=region(l,row)-region(l,flank);side=region(r,flank); br=sample(.72,0)
        # STOP has white letter strokes interleaved with red within the word;
        # no-entry is white across an entire horizontal bar. Both are red above/below.
        top=np.clip(side-.27,-.8,.8); redring=robust(br)
        chrom=.36*robust(sample(.69,6));band=l[:,(abs(Y)<.21)&(abs(X)<.83)].reshape(len(l),np.count_nonzero(abs(axis)<.21),-1).mean(axis=1)
        a,b=sortpart(band,.10,.90);letters=(b-a).ravel();lowbar=(a.ravel()-region(l,flank))
        if k==5:
            a,b=sortpart(l[:,row],.23,.87)
            return .37*top+.21*np.clip((b-a).ravel()-.19,-.4,.8)+.17*np.clip(stroke,-.6,.6)+.19*(redring-.30)+.42*letters+chrom
        return .42*top+.27*np.clip(stroke-.15,-.8,.8)+.15*(redring-.30)+.42*lowbar-.26*letters+chrom
    if k in (7,8):
        l,r,b=cropchannels(ch,p,[3,0,1],X*.73,Y*.73);diag=(X+Y)*.707; along=(X-Y)*.707
        side=(abs(diag)>.53)&(X*X+Y*Y<1.40); bar=(abs(diag)<.19)&(abs(along)<.99)
        diff=region(l,side)-region(l,bar); edge=robust(sample(.83,3)-sample(1.13,3))
        chrom=robust(sample(.83,0))+robust(sample(.83,1))
        # Test for a continuous stripe: in each of a sequence of tiles
        # along the slash the two sides should be lighter than the middle.
        t=np.linspace(-.79,.79,16,dtype=np.float32);z=np.linspace(-.12,.12,5,dtype=np.float32);along,across=np.meshgrid(t,z)
        def strip(off):return view(ch[:,3],p,(along+across+off)*.707,(across+off-along)*.707).mean(axis=1)
        center=strip(0);away=(strip(.39)+strip(-.39))*.5
        diffq=np.sort(away-center,axis=1)[:,2:12].mean(axis=1)
        tint=abs(robust(sample(.78,6)))+abs(robust(sample(.78,7)))
        return .44*np.clip(diff,-.65,1.2)+.24*np.clip(edge,-.6,1)-.19*np.clip(chrom,-.2,1.25)+.52*diffq+.20-.29*np.minimum(tint,1.4)
    return .27*robust(sample(.51,2)-sample(.87,2))

familyclass=[15,35,18,13,12,14,17,32,32]
def block(x):
    ch=channels(x);sc,ps=coarse(ch)
    # Structural likelihoods. Downweight the no-entry stripe for comparing
    # different outer shapes; retain the stripe as strong internal evidence.
    sc[:,6]*=.79;sc[:,3]*=.95
    out=np.array(familyclass)[None,:]+np.zeros((len(x),9),np.int32)
    best=sc.max(axis=1);base=sc.copy();sc-=1
    for k in fkind:
        take=np.flatnonzero((base[:,k]>best-.90)&((k!=8)|(base[:,8]>base[:,7]+.015)))
        if len(take)==0:continue
        sub=ch[take];p,hs=refine(sub,k,ps[take,k])
        if k==6:hs*=.79
        if k==3:hs*=.95
        sc[take,k]=.32*base[take,k]+.68*hs+verifyface(sub,p,k)
        if k==0:c,s=circscore(sub,p)
        elif k==1:c,s=bluescore(sub,p)
        elif k==2:c,s=warnscore(sub,p)
        elif k in (7,8):c,s=ends(sub,p)
        else:continue
        out[take,k]=c
        if k in (0,1,2):sc[take,k]+=.63*(s-.44)
    return out[np.arange(len(x)),sc.argmax(axis=1)]

def predict(x):
    # Always use the same internal dimensions, including on a short final
    # block, so that BLAS reduction order cannot depend on the caller's batch.
    res=np.empty(len(x),np.int32)
    for i in range(0,len(x),48):
        a=x[i:i+48]; n=len(a)
        if n<48:a=np.concatenate([a,np.repeat(a[-1:],48-n,axis=0)])
        res[i:i+n]=block(a)[:n]
    return res
'''

def main(out):
    out.mkdir(exist_ok=True,parents=True)
    (out/'solution.py').write_text(RUNTIME)
    runtime={}; exec(RUNTIME,runtime)
    geom,poses=runtime['makegeom'](); const={'geom':geom,'poses':poses}
    for key,cls in [('circ',runtime['circ']),('warn',runtime['warn']),('blue',runtime['bluecls'])]:
        bank,labs=template_bank(cls,key)
        f=np.concatenate([runtime['innerfeatures'](bank[j:j+96],key) for j in range(0,len(bank),96)])
        # Runtime feature vectors carry additive correlation weights; templates are unit length in each block.
        f[:,:144]/=.50;f[:,144:288]/=.25; f[:,288:]/=.25
        const[key]=f
        if key=='circ':
            ink=bank[np.isin(labs,[5,9,10])]; X,Y=runtime['X'],runtime['Y']; diag=(.68*X+.50*Y)*.707
            for i in range(3):
                good=(abs(diag)>(.15+.088*i))&((.68*X)**2+(.5*Y)**2<.54)
                v=runtime['smooth'](ink)[:,good];const['end'+str(i)]=runtime['units'](v-v.mean(axis=1,keepdims=True))
    parts=[];u=(np.arange(S*Q)+.5)/(S*Q)*2-1; xx,yy=np.meshgrid(u,u)
    for d in '0235678':
        for style in range(3):
            w=320; a=(np.arange(w)+.5)/w*2-.5;ax,ay=np.meshgrid(a,a);master=numeral(ax,ay,d,style)
            for sx,sy in [(.76,.78),(.88,.78),(.67,.78),(.78,.90),(.78,.65),(.68,.66),(.90,.92)]:
                for dx in [-.18,0,.18]:
                    for dy in [-.13,0,.13]:
                        ix=np.clip(((.5+(xx-dx)/sx*.5+.5)*w/2).astype(int),0,w-1);iy=np.clip(((.5+(yy-dy)/sy*.5+.5)*w/2).astype(int),0,w-1)
                        parts.append(master[iy,ix].reshape(S,Q,S,Q).mean(axis=(1,3)))
    parts=np.array(parts,np.float32);f=np.concatenate([runtime['innerfeatures'](parts[j:j+100],'part') for j in range(0,len(parts),100)]);f[:,:144]*=2;f[:,144:]*=4;const['part']=f
    np.savez(out/'constants.npz',**const)

if __name__=='__main__':main(Path(sys.argv[1]))
