"""Construct a library of independent views of hand-modeled objects. No data inputs."""
import math
import sys
from pathlib import Path
import numpy as np

# Model coordinates: longitudinal (nose positive), transverse, up.
# Colors below describe materials in daylight, not measured image colors.
def col(r,g,b): return np.array((r,g,b),dtype=float)
def choose(items): return items[R.integers(len(items))]
def u(a,b): return R.uniform(a,b)
def prob(p): return R.random()<p
def vary(c,s=.12): return np.clip(np.asarray(c)*u(1-s,1+s)+R.normal(0,s*.16,3),.015,1)
white=col(.88,.88,.85); black=col(.075,.072,.07); chrome=col(.63,.66,.68)
red=col(.73,.12,.09); blue=col(.13,.29,.56); green=col(.24,.38,.15); yellow=col(.94,.67,.18)
buff=col(.73,.56,.38); umber=col(.38,.23,.14); slate=col(.37,.40,.41); cream=col(.89,.80,.65)
paints=[white,white,black,chrome,chrome,red,blue,slate,col(.32,.16,.20),col(.16,.33,.28),yellow]
glasscolors=[col(.15,.22,.25),col(.33,.44,.49),col(.47,.56,.62),col(.12,.15,.16)]

def axis(d):
    d=np.array(d,dtype=float); return d/(np.linalg.norm(d)+1e-15)

def ell(p,axes,c,frame=None):
    f=np.eye(3) if frame is None else frame
    shapes.append(('e',np.asarray(p),f @ np.diag(axes),np.asarray(c)))

def link(a,b,w,h,c,ext=1.09):
    a,b=np.array(a),np.array(b); d=axis(b-a); side=axis(np.cross([0,0,1],d)) if abs(d[2])<.999 else np.array([0.,1.,0.]); up=np.cross(d,side)
    ell(.5*(a+b),(.5*ext*np.linalg.norm(b-a),w,h),c,np.stack([d,side,up],1))

def poly(points,c): shapes.append(('p',np.array(points,dtype=float),np.asarray(c)))

def loft(sections,c,top=None):
    # Solid with trapezoidal transverse cross sections (x, bottom, top, halfwidth bottom/top).
    rings=[[(x,-w,b),(x,w,b),(x,v,t),(x,-v,t)] for x,b,t,w,v in sections]
    poly(rings[0],c*.84); poly(rings[-1],c*.92)
    for a,b in zip(rings,rings[1:]):
        for j in range(4):
            k=(j+1)%4; poly([a[j],a[k],b[k],b[j]], (top if top is not None else c) if j==2 else c*(.56 if j==0 else 1))

def box(x1,x2,y1,y2,z1,z2,c):
    a=[(x1,y1,z1),(x2,y1,z1),(x2,y2,z1),(x1,y2,z1)]; b=[(x,y,z2) for x,y,z in a]; poly(a,c*.6); poly(b,c)
    for j in range(4):
        k=(j+1)%4; poly([a[j],a[k],b[k],b[j]],c)

def wheel(x,y,r,z=None):
    if z is None: z=r
    ell((x,y,z),(r,.087,r),black*.74)
    s=np.sign(y); ell((x,y+s*.067,z),(r*.60,.025,r*.60),vary(chrome,.3)); ell((x,y+s*.089,z),(r*.16,.008,r*.16),slate)

def car():
    paint=vary(choose(paints)); glass=vary(choose(glasscolors)); van=prob(.18); w=u(.39,.49); r=u(.18,.225); deck=u(.42,.57); roof=deck+u(.25,.46)+(van*.14); front=u(.18,.38); rear=u(-.68,-.32)
    loft([(-1,.16,deck-.065,.90*w,.93*w),(-.78,.13,deck,w,w),(.52,.13,deck,w,.98*w),(1,.19,deck-u(.05,.15),w*.9,w*.82)],paint)
    a=-.91 if van else -.71; b=.81 if van else .61
    loft([(a,deck-.05,deck+.012,w*.95,w*.94),(rear,deck-.02,roof,w*.97,w*.79),(front,deck-.02,roof-.013,w*.97,w*.79),(b,deck-.04,deck+.012,w*.92,w*.90)],glass,paint)
    for s in [-1,1]:
        # Pillars, door sill and mirrors.
        for xx in [rear,(rear+front)*.5,front]: link((xx,s*.97*w,deck),(xx,s*.79*w,roof),.018,.024,paint)
        link((a,s*w*.97,deck),(b,s*w*.94,deck),.015,.017,chrome)
        ell((front+.12,s*(w+.02),deck+.07),(.069,.056,.041),paint)
        for xx in [-.64,.635]: wheel(xx,s*w*.96,r)
        box(.976,1.013,s*.64*w-.075,s*.64*w+.075,deck-.14,deck-.07,white)
        box(-1.015,-.99,s*.70*w-.04,s*.70*w+.04,deck-.14,deck-.065,red)
    box(.998,1.007,-.18,.18,.215,.28,black); box(1.006,1.01,-.071,.071,.206,.253,cream)

def truck():
    c=vary(choose([white,white,blue,red,yellow,green])); d=vary(choose([white,white,chrome,slate,red,blue,c])); glass=vary(choose(glasscolors)); w=u(.29,.39); z=u(.92,1.14); cab=u(.72,1.00); r=u(.146,.19); bonnet=prob(.34)
    box(-1.12,.95,-w*.70,w*.70,.18,.31,black)
    box(.38,.78 if bonnet else 1.02,-w,w,.28,cab,c)
    front=.785 if bonnet else 1.027
    box(front-.002,front+.01,-w*.92,w*.92,cab*.66,cab*.95,glass)
    if bonnet: box(.77,1.08,-w*.85,w*.85,.29,.59,c)
    for s in [-1,1]:
        box(.45,front-.045,s*w-.004,s*w+.004,cab*.65,cab*.94,glass)
        box(1.017,1.09,s*.74*w-.045,s*.74*w+.045,.38,.45,white)
        wheel(.80,s*w,r)
        for x in ([-.92,-.60,.14] if prob(.52) else [-.78,.12]): wheel(x,s*w,r)
    box(1.027,1.085,-.15,.15,.35,.485,slate)
    if prob(.18):
        # Tipper body with an overhanging lip.
        loft([(-1.14,.41,z,w*.85,w),(-.89,.34,z,w*.9,w),(.21,.36,z,w*.9,w),(.37,z-.18,z,w,w)],d)
    else:
        box(-1.15,.29,-w*1.04,w*1.04,.35,z,d)
        if prob(.56):
            stripe=vary(choose([blue,red,green,buff,chrome])); h=u(.52,.80)
            for s in [-1,1]: box(-1.14,.28,s*w*1.04-.003,s*w*1.04+.003,h,h+u(.03,.11),stripe)

def plane():
    c=vary(choose([white,white,chrome,slate,cream,blue])); d=vary(choose([c,blue,white,red,slate])); jet=prob(.72); r=u(.075,.145); z=r*.1
    ell((0,0,0),(1.04,r,r*1.05),c); sweep=u(.22,.63) if jet else u(0,.14); span=u(.79,1.15)
    for s in [-1,1]:
        poly([(.31,s*.03,z),(.31-sweep,s*span,z+.065),(.16-sweep,s*span,z+.064),(-.32,s*.04,z)],c*.85)
        poly([(-.60,s*.02,z+.03),(-.85,s*.40,z+.095),(-.98,s*.39,z+.096),(-.90,s*.01,z+.044)],d)
        if jet: ell((.035,s*.40,-.063),(.18,.067,.069),c*.81)
        else: link((.09,s*.45,z-.13),(.1,s*.49,z+.02),.009,.009,slate)
        for j in range(7 if jet else 2):
            ell((.54-j*.145,s*r*.95,.018),(.026 if jet else .07,.008,.025 if jet else .042),slate*.65)
    poly([(-.52,0,.04),(-.84,0,u(.36,.49)),(-.98,0,.47),(-.94,0,.015)],d)
    ell((.77,0,.057),(.142,.74*r,.042),slate*.64)
    if not jet: link((1.015,0,-.26),(1.015,0,.26),.014,.016,black)

def ship():
    w=u(.23,.36); h=u(.18,.36); hull=vary(choose([black,slate,blue,white,white,red,umber])); cabin=vary(choose([white,white,cream,slate])); win=vary(choose(glasscolors)); kind=R.integers(5)
    loft([(-1.01,0,h+.04,w*.60,w*.82),(-.82,-.09,h,w*.76,w),(.54,-.09,h,w*.76,w),(.87,-.015,h+.027,w*.45,w*.61),(1.10,h*.88,h+.072,.014,.014)],hull,cabin*.77)
    if kind==0:
        top=u(1.,1.75); x=u(-.15,.15); link((x,0,h),(x,0,top),.011,.011,umber)
        poly([(x-.017,0,top),(x-.76,.065,h+.08),(x-.018,0,h+.066)],cabin)
        poly([(x+.045,0,top*.84),(.82,0,h+.075),(x+.035,-.041,h+.19)],vary(choose([cream,white,red,blue])))
    else:
        mid=-.55 if kind==1 else u(-.13,.06); L=.29 if kind==1 else u(.45,.73); levels=3 if kind==3 else (1 if kind==4 else 2)
        for j in range(levels):
            a=mid-L+.095*j; b=mid+L-.055*j; W=w*(.82-.08*j); base=h+.124*j
            box(a,b,-W,W,base,base+.126,cabin)
            for s in [-1,1]:
                if kind==4: box(a+.035,b-.035,s*W-.003,s*W+.003,base+.045,base+.107,win)
                else:
                    for xx in np.linspace(a+.059,b-.052,5): box(xx-.033,xx+.033,s*W-.003,s*W+.003,base+.045,base+.097,win)
        if kind==1:
            for xx in [-.09,.23,.55]: box(xx-.14,xx+.13,-.74*w,.74*w,h,h+u(.1,.22),vary(choose([red,umber,slate,blue])))
        else: box(mid-.14,mid+.017,-w*.28,w*.28,h+.12*levels,h+.12*levels+.16,choose([black,red,slate]))
        link((mid+.11,0,h+.12*levels),(mid+.11,0,h+.12*levels+u(.22,.66)),.012,.012,slate)

def spoteye(p,r,c=black): ell(p,(r,r*.66,r),c)

def bird():
    palette=choose([(umber,buff,cream),(slate,black,white),(green,green,yellow),(yellow,umber,yellow),(blue,slate,buff),(white,chrome,white),(black,black,slate),(red,umber,red)])
    c,d,h=[vary(t,.24) for t in palette]; flying=prob(.31); water=prob(.16); tilt=u(.05,.36) if not flying else u(-.10,.14)
    link((-.44,0,-tilt),(.39,0,tilt),.27,.32,c,1.12); hx=u(.30,.53); hz=tilt+u(.16,.31)
    if water: hx+=.19; link((.22,0,.02),(hx,0,hz),.144,.16,c)
    ell((hx,0,hz),(.239,.217,.232),h); beak=choose([yellow,umber,black,buff]); tip=hx+u(.36,.62)
    for s in [-1,1]:
        poly([(hx+.16,-s*.118,hz+.001),(hx+.16,s*.116,hz+.037),(tip,0,hz-.023)],beak)
        poly([(hx+.16,s*.116,hz+.037),(hx+.16,s*.066,hz-.073),(tip,0,hz-.023)],beak*.77)
        spoteye((hx+.101,s*.183,hz+.066),.038)
        if flying:
            zz=u(.14,.8); poly([(.25,s*.12,.03),(.33,s*.62,zz*.69),(-.26,s*1.26,zz),(-.42,s*.75,zz*.60),(-.33,s*.12,.0)],d)
        else:
            link((-.55,s*.17,-tilt*.87),(.23,s*.19,.14+tilt*.44),.105,.21,d)
            foot=-.43-tilt; link((-.075,s*.109,-.15),(-.105,s*.112,foot),.02,.025,beak); link((-.20,s*.111,foot),(.04,s*.122,foot-.015),.02,.014,beak)
    poly([(-.25,-.08,-tilt*.75),(-u(.75,1.09),-.153,-tilt*1.55),(-.91,.147,-tilt*1.55),(-.25,.08,-tilt*.75)],d)
    return flying

def hoofed(horse,portrait):
    c=vary(choose(([umber,umber,buff,black,cream,white,col(.48,.245,.13)] if horse else [buff,umber,col(.48,.36,.23),col(.58,.43,.29)])),.22)
    mane=vary(choose([black,umber,c*.58]) if horse else c*.81); belly=c*1.12; leg=u(.68,.90) if horse else u(.84,1.10); H=leg+.20; body=.32 if horse else .246
    ell((-.07,0,H),(.756,.259,body),c); ell((-.50,0,H+.005),(.32,.261,body*1.01),c)
    high=u(.47,.78); hx=u(.67,.91); hz=H+high; nod=u(-.28,.18); graze=(not portrait) and prob(.21)
    if graze: hx=.91; hz=H-.41; nod=-.25
    dst=(hx,0,hz); link((.30,0,H-.12),dst,.20 if horse else .14,.23 if horse else .16,c,1.19)
    if horse: link((.34,0,H+.23),(hx-.09,0,hz+.09),.089,.10,mane,1.08)
    nose=(hx+.30,0,hz-.22+nod) if horse else (hx+.26,0,hz-.108+nod*.50)
    ell(dst,(.20,.153,.212) if horse else (.165,.129,.15),c); link(dst,nose,.13 if horse else .106,.166 if horse else .112,c,1.25)
    ell(nose,(.124,.127,.117) if horse else (.084,.096,.078),c*.70 if horse else black)
    for s in [-1,1]:
        link((hx-.074,s*.072,hz+.072),(hx-.12,s*(.138 if horse else .245),hz+(.36 if horse else .28)),.066,.067,c)
        spoteye((hx+.086,s*(.137 if horse else .118),hz+.04),.041 if horse else .032)
        if not horse and prob(.42):
            a=(hx-.06,s*.082,hz+.10); b=(hx-.28,s*.32,hz+.56); link(a,b,.025,.025,buff*.74)
            link((hx-.17,s*.21,hz+.36),(hx+.016,s*.33,hz+.56),.017,.017,buff*.75)
            link(b,(hx-.19,s*.40,hz+.71),.014,.014,buff*.72)
        for fore in [False,True]:
            x=.39 if fore else -.51; y=s*.147; thick=.092 if horse else .062; dx=u(-.19,.19); a=(x,y,H-.05); b=(x+(-.03 if fore else .14)+dx*.3,y*1.10,leg*.53); d=(x+dx+(.015 if fore else -.071),y*1.12,.084)
            link(a,b,thick*1.15,thick*1.50,c); link(b,d,thick*.65,thick*.72,c*.90); ell((d[0]+.013,d[1],.05),(.076 if horse else .048,thick*.65,.058),black if horse else c*.53)
    if horse:
        link((-.72,0,H+.15),(-.90,0,H-.31),.094,.11,mane); link((-.90,0,H-.28),(-1.0,.02,H-u(.64,.87)),.079,.103,mane)
        if not portrait and prob(.19):
            ell((-.04,0,H+.25),(.23,.275,.109),umber*.55); link((-.08,0,H+.29),(-.014,0,H+.83),.125,.112,vary(choose([white,red,slate,blue]))); ell((0,0,H+.94),(.106,.091,.125),buff); ell((-.009,0,H+1.02),(.121,.103,.073),black)
    else: link((-.72,0,H+.10),(-.84,0,H-.11),.06,.075,cream)
    return np.array(dst)

def pet(cat,portrait):
    c=vary(choose(([buff,slate,chrome,white,black,umber,col(.76,.51,.27)] if cat else [buff,umber,cream,white,black,col(.63,.37,.17),slate])),.2)
    second=vary(choose([cream,white,c*1.14,c*.71])); pattern=choose(['plain','patched','striped' if cat else 'saddle']); ears=c*choose([.65,.9,1.01]); sit=prob(.48); H=.53 if sit else u(.53,.81); hx=.38 if sit else .55; hz=H+(.51 if sit else .25); fluffy=u(.90,1.21)
    if sit: link((-.26,0,.28),(.20,0,H+.24),.27*fluffy,.32*fluffy,c,1.35)
    else: ell((-.059,0,H),(.63,.25*fluffy,.29*fluffy),c)
    for s in [-1,1]:
        for fore in [False,True]:
            x=hx-.18 if fore else -.43; a=(x,s*.154,H-.026); d=(x+u(-.085,.085),s*.17,.09); b=(x+(.23 if sit and not fore else .07),s*.183,.27)
            link(a,b,.090,.117,c); link(b,d,.07,.072,second if pattern=='patched' else c); ell((d[0]+.05,d[1],.062),(.131,.077,.066),second if pattern!='plain' else c)
    root=(-.48 if sit else -.62,0,H-.05); tip=(-.94,.10,H+u(-.40,.47)); bp=(-.85,.04,H-.14); link(root,bp,.073,.079,c); link(bp,tip,.064,.067,c if cat else second)
    neck=(hx-.13,0,H-.02); link(neck,(hx,0,hz),.197*fluffy,.202,c,1.2)
    if pattern!='plain':
        ell((-.26,0,H+.15),(.21,.248*fluffy,.156),second if pattern=='patched' else c*.61)
    hr=u(.225,.29) if cat else u(.21,.32); w=hr*u(.91,1.08); cx=hx; cz=hz; ec=choose([col(.58,.63,.27),col(.58,.49,.23),green]) if cat else umber*.65
    ell((cx,0,cz),(hr*.93,w,hr),c); muzzle=hr*(u(.38,.63) if cat else u(.56,1.13)); forward=cx+hr*.61
    ell((forward,0,cz-hr*.29),(muzzle,w*.61,hr*(.43 if cat else .46)),second)
    nx=forward+muzzle*.93; ell((nx,0,cz-hr*.16),(hr*.125,w*(.20 if cat else .34),hr*(.12 if cat else .22)),choose([umber,black,col(.66,.42,.42)]) if cat else black)
    drop=not cat and prob(.68)
    for s in [-1,1]:
        if drop:
            link((cx-.09,s*w*.71,cz+hr*.64),(cx-.079,s*w*1.26,cz-hr*u(.51,1.31)),hr*.40,hr*.53,ears)
        else:
            a=(cx-hr*.07,s*w*.30,cz+hr*.70); b=(cx-hr*.20,s*w*.96,cz+hr*.37); apex=(cx-hr*.36,s*w*u(.72,1.07),cz+hr*(1.49 if cat else 1.74)); back=(cx-hr*.65,s*w*.64,cz+hr*.57)
            poly([a,b,apex],ears); poly([a,apex,back],ears*.8); poly([b,back,apex],c); mid=(np.array(a)+b+apex)/3; tri=[mid+.62*(np.array(p)-mid)+[.003,0,0] for p in [a,b,apex]]; poly(tri,.52*c+.48*buff)
        ey=(cx+hr*.676,s*w*.65,cz+hr*.18); rr=hr*(.20 if cat else .164)
        ell(ey,(rr*.78,rr,rr*.88),ec); ell((ey[0]+rr*.63,ey[1]+s*rr*.29,ey[2]),(rr*.28,rr*(.25 if cat else .61),rr*.72),black)
        if pattern=='striped':
            for v in [.04,.30,.56]: link((cx+v*hr,s*.46*w,cz+.85*hr),(cx+(v+.07)*hr,s*.88*w,cz+.40*hr),hr*.069,hr*.069,c*.49)
    return np.array((cx,0,cz)),hr

def frog():
    c=vary(choose([green,col(.40,.48,.16),umber,buff,col(.21,.25,.12),col(.25,.57,.18)]),.25); belly=vary(choose([cream,buff,c*1.16])); spot=c*u(.34,.68)
    ell((-.12,0,.27),(.51,.335,.252),c); ell((.33,0,.31),(.36,.33,.213),c); ell((.48,0,.245),(.245,.296,.099),belly)
    for s in [-1,1]:
        ell((.36,s*.218,.46),(.135,.132,.136),c); ell((.409,s*.287,.487),(.087,.079,.084),choose([buff,yellow,black])); ell((.436,s*.335,.494),(.063,.041,.031),black)
        link((-.37,s*.21,.23),(-.063,s*.46,.164),.148,.15,c); link((-.055,s*.46,.158),(-.61,s*.466,.071),.084,.080,c)
        link((.211,s*.22,.29),(.034,s*.40,.089),.067,.077,c); link((.034,s*.40,.097),(.34,s*.47,.042),.055,.055,c)
        for j in range(3): link((.29,s*.455,.048),(.48-.077*j,s*(.43+.07*j),.024),.016,.016,c)
    if prob(.68):
        for j in range(16):
            x=u(-.51,.25); y=u(-.23,.23); surf=np.sqrt(max(0,1-((x+.12)/.51)**2-(y/.335)**2)); r=u(.024,.071); ell((x,y,.27+.247*surf),(r,r*.85,.022),spot)

# Orthographic z-buffer renderer. Ellipsoids use the analytic front ray intersection;
# polygon depth is interpolated in the screen plane. One image at a time.
S=64
YY,XX=np.mgrid[:S,:S].astype(float); X=(XX+.5)/S; Y=(YY+.5)/S

def noise(nx,ny):
    xx=X*(nx-1); yy=Y*(ny-1); ix=xx.astype(int); iy=yy.astype(int); fx=xx-ix; fy=yy-iy; z=R.normal(size=(ny,nx))
    return (1-fy)*((1-fx)*z[iy,ix]+fx*z[iy,ix+1])+fy*((1-fx)*z[iy+1,ix]+fx*z[iy+1,ix+1])

def material(kind):
    sky=[col(.41,.66,.85),col(.59,.72,.80),col(.78,.80,.80),col(.82,.85,.89),col(.31,.53,.77),col(.92,.90,.85)]
    grass=[col(.33,.46,.22),col(.42,.46,.24),col(.55,.56,.30),col(.29,.39,.19),col(.30,.46,.25),col(.49,.40,.26)]
    woods=[col(.19,.28,.14),col(.25,.30,.17),col(.31,.31,.18),col(.43,.38,.27),col(.23,.33,.22)]
    earth=[buff,umber,col(.47,.43,.36),col(.35,.30,.27),col(.54,.46,.29)]
    water=[col(.36,.48,.55),col(.23,.40,.52),col(.41,.51,.50),col(.38,.51,.64),col(.50,.57,.62),col(.17,.29,.37),col(.42,.45,.35)]
    room=[cream,slate,buff,umber,white,col(.48,.34,.30),col(.31,.37,.42),col(.31,.29,.25)]
    city=[chrome,slate,buff,cream,col(.34,.37,.38),col(.64,.53,.45)]
    road=[col(.41,.41,.40),col(.36,.36,.37),col(.55,.54,.51),col(.64,.62,.59),col(.31,.32,.32)]
    bg=vary(choose({'sky':sky,'grass':grass,'woods':woods,'earth':earth,'water':water,'room':room,'city':city,'road':road}[kind]),.19)
    if kind=='sky': t=u(.003,.048)*noise(6,4)+u(-.05,.13)*(Y-.4); ch=0.
    elif kind=='water': t=u(.02,.13)*noise(13,61)+.046*noise(5,17)+.036*noise(4,5); ch=.018
    elif kind=='road': t=.04*noise(4,4)+u(.008,.051)*noise(59,54); ch=.008
    elif kind=='city' or kind=='room':
        t=.086*noise(3,4)+u(.026,.105)*noise(6,6)+.019*noise(32,32); ch=.028
        for j in range(R.integers(2,9)):
            xx=u(-.3,.95); yy=u(-.48,.88); t+=u(-.20,.18)*((X>xx)&(X<xx+u(.08,.65))&(Y>yy)&(Y<yy+u(.13,.94)))
    else:
        tex=u(.36,1.18); t=tex*(.10*noise(5,6)+.075*noise(12,14)+.055*noise(31,29)+.065*noise(62,63)); ch=.039*tex
        if kind=='woods': t+=u(.04,.14)*noise(18,3)
        if kind=='grass': t+=.028*(Y-.51)*noise(37,61)
    return np.clip(bg+t[:,:,None]+(ch*noise(8,7)[:,:,None]*np.array([1,-.30,-.73]) if ch else 0),0,1)

def scene(k,air,grazing=False):
    if k==0:
        bottom=choose(['road','grass','earth']) if not air else 'sky'; mid=choose(['woods','city']); top='sky'; horizon=u(.39,.71)
    elif k==8:
        bottom='water'; mid=choose(['woods','city','city']); top='sky'; horizon=u(.24,.60)
    elif k in [1,9]:
        bottom=choose(['road','road','road','earth','grass']); mid=choose(['city','city','woods']); top='sky'; horizon=u(.31,.66)
    elif k==2:
        bottom=choose(['woods','woods','grass','water','earth','sky']); mid='woods'; top=choose(['sky','woods','grass']); horizon=u(-.14,.71)
        if air and prob(.72): bottom=top='sky'
    elif k in [3,5]:
        bottom=choose(['room','room','room','earth','grass'] if k==3 else ['room','grass','grass','road','earth','room']); top=choose(['city','woods','sky']) if bottom!='room' else 'room'; mid=choose(['city','woods']); horizon=u(-.15,.61)
    elif k in [4,7]:
        bottom=choose(['grass','grass','earth','woods'] if k==4 else ['grass','grass','earth','grass']); top=choose(['sky','woods','woods','grass'] if k==4 else ['sky','sky','woods','city']); mid=choose(['woods','woods','city']) if k==7 else 'woods'; horizon=u(.08,.53)
    else:
        bottom=choose(['earth','woods','grass','earth','water']); mid='woods'; top=choose([bottom,'woods','grass','earth']); horizon=u(-.2,.45)
    img=material(bottom); slope=u(-.067,.067); hill=Y+slope*(X-.5)
    if top!=bottom:
        usemid=top=='sky' and prob(.87 if k in [1,9,4,7] else .37); thick=u(.035,.41) if usemid else 0
        img=np.where((hill<horizon-thick)[:,:,None],material(top),img)
        if usemid:
            wav=(.015 if mid=='city' else .053)*noise(9,2); middle=(hill<horizon)&(hill+wav>horizon-thick-.03); img=np.where(middle[:,:,None],material(mid),img)
    return img,bottom

def paint(scene,P,center,scale,tx,ty,light):
    depth=np.full((S,S),-100.); pix=scene.copy(); coords=[]; positions=[]
    for shape in shapes:
        if shape[0]=='e':
            q=P@(shape[1]-center); a=P@shape[2]; rad=np.sqrt((a*a).sum(1)); coords.append((q[:2]-rad[:2],q[:2]+rad[:2])); positions.append((q,a))
        else:
            vs=(shape[1]-center)@P.T; coords.append((vs.min(0)[:2],vs.max(0)[:2])); positions.append(vs)
    lo,hi=np.array(coords).min(0)[0],np.array(coords).max(0)[1]; mid=(lo+hi)/2
    if scale[1] is None: sc=scale[0]*S/max(hi-lo); origin=mid
    else: sc=S*scale[0]/scale[1]; origin=np.zeros(2)
    off=np.array([tx,ty])*S-sc*origin
    for shape,(low,high),v in zip(shapes,coords,positions):
        xmin,ymin=np.maximum(0,np.ceil(low*sc+off).astype(int)); xmax,ymax=np.minimum(S-1,np.floor(high*sc+off).astype(int));
        if xmax<xmin or ymax<ymin: continue
        sl=np.s_[ymin:ymax+1,xmin:xmax+1]; U=(XX[sl]-off[0])/sc; V=(YY[sl]-off[1])/sc; c=shape[-1]
        if shape[0]=='e':
            q,a=v; m=a@a.T; det=m[0,0]*m[1,1]-m[0,1]**2; i0=m[1,1]/det; i1=-m[0,1]/det; i2=m[0,0]/det
            du=U-q[0]; dv=V-q[1]; gu=i0*du+i1*dv; gv=i1*du+i2*dv; e=1-du*gu-dv*gv; mask=e>=0
            hu=m[2,0]*i0+m[2,1]*i1; hv=m[2,0]*i1+m[2,1]*i2; dz=math.sqrt(max(1e-14,m[2,2]-hu*m[0,2]-hv*m[1,2])); t=np.sqrt(np.maximum(e,.00001)); z=q[2]+hu*du+hv*dv+dz*t
            nx=dz*gu-hu*t; ny=dz*gv-hv*t; dot=(nx*light[0]+ny*light[1]+t*light[2])/np.sqrt(nx*nx+ny*ny+t*t); bright=.74+.32*dot
        else:
            n=np.cross(v[1]-v[0],v[2]-v[0]);
            if abs(n[2])<1.e-10: continue
            z=v[0,2]-(n[0]*(U-v[0,0])+n[1]*(V-v[0,1]))/n[2]; pos=np.ones(U.shape,bool); neg=pos.copy()
            for a,b in zip(v,np.roll(v,-1,axis=0)):
                side=(b[0]-a[0])*(V-a[1])-(b[1]-a[1])*(U-a[0]); pos&=side>=-1.e-8; neg&=side<=1.e-8
            mask=pos|neg; bright=.76+.25*np.dot(n,light)*np.sign(n[2])/np.linalg.norm(n)
        hit=mask&(z>depth[sl]); depth[sl][hit]=z[hit]; pix[sl][hit]=bright[hit,None]*c if shape[0]=='e' else bright*c
    return pix,depth>-99

def render(k):
    global shapes
    shapes=[]; center=np.zeros(3); crop=None; air=False; portrait=False; elevation=u(0,.43); az=u(-math.pi,math.pi); roll=u(-.085,.085); extent=u(.64,1.05); texture=.016; ox=u(-.068,.068); oy=u(-.080,.080)
    if k==0:
        plane(); air=prob(.70); elevation=u(-.92,.53) if air else u(-.06,.25); roll=u(-.59,.60) if air else roll; extent=u(.53,1.08)
    elif k==1: car(); elevation=u(.03,.54); extent=u(.73,1.26)
    elif k==9: truck(); elevation=u(-.055,.42); extent=u(.72,1.24)
    elif k==8: ship(); elevation=u(.012,.45); extent=u(.54,1.13); oy+=.053
    elif k==2:
        air=bird(); roll=u(-.42,.42) if air else u(-.22,.22); extent=u(.47,1.18); texture=.035
    elif k in [4,7]:
        portrait=prob(.15 if k==7 else .23); center=hoofed(k==7,portrait); texture=.036; extent=u(.55,1.05)
        if portrait: crop=u(.75,1.47); extent=1.; oy-=.05
    elif k in [3,5]:
        portrait=prob(.48 if k==3 else .42); center,hr=pet(k==3,portrait); texture=u(.044,.095); elevation=u(-.10,.75)
        if portrait: crop=hr*u(2.45,4.35); extent=1.; az=math.pi*.5+u(-.89,.89); roll=u(-.25,.25)
        elif prob(.50): az=math.pi*.5+u(-1.35,1.35)
    else: frog(); elevation=u(.12,1.20); extent=u(.65,1.26); texture=.053
    if k in [0,1,8,9] and not (k==0 and air) and prob(.80): az=u(-1.18,1.18)+choose([0.,math.pi])
    azbasis=np.array([np.cos(az),np.sin(az),0]); front=np.array([np.sin(az)*np.cos(elevation),-np.cos(az)*np.cos(elevation),np.sin(elevation)]); down=np.cross(azbasis,front)
    P=np.array([np.cos(roll)*azbasis+np.sin(roll)*down,-np.sin(roll)*azbasis+np.cos(roll)*down,front]); background,bottom=scene(k,air)
    pix,mask=paint(background,P,center,(extent,crop),.5+ox,.5+oy,axis([u(-.8,.8),u(-.95,-.33),u(.28,.90)]))
    if k in [1,9] and mask.any():
        # Soft ambient shadow directly beneath the vehicle.
        yy,xx=np.nonzero(mask); cx=.5*(xx.max()+xx.min()); cy=yy.max()-1; shadow=np.exp(-2*((XX-cx)/(.51*(xx.max()-xx.min())+1))**2-2*((YY-cy)/u(2.3,6.5))**2); pix*=1-(~mask)[:,:,None]*shadow[:,:,None]*u(.15,.50)
    tex=texture*(.75*noise(42,49)+.65*noise(19,20)+.57*noise(7,7)); pix+=mask[:,:,None]*tex[:,:,None]
    pix=pix*u(.80,1.15)*R.uniform(.93,1.07,size=3)+u(-.035,.065); pix=np.clip(pix,0,1).reshape(32,2,32,2,3).mean((1,3))
    if prob(.16):
        grey=np.sum(pix*np.array([.28,.55,.17]),axis=2,keepdims=True); pix=grey+(pix-grey)*u(0,.49)
    return pix.transpose(2,0,1).astype(np.float32)

# The following pure numpy description is shared by builder and predictor. It
# transforms a view independently; none of its normalizers use the batch axis.
features=r'''import numpy as np

def smooth(x):
    h=np.concatenate((x[:,:,:1],x,x[:,:,-1:]),2)
    h=(h[:,:,:-2]+2*h[:,:,1:-1]+h[:,:,2:])*.25
    v=np.concatenate((h[:,:1],h,h[:,-1:]),1)
    return (v[:,:-2]+2*v[:,1:-1]+v[:,2:])*.25

def grid(x,n):
    m=x.shape[1]//n
    return x.reshape(len(x),n,m,n,m).mean((2,4))

def unit(x):
    x=x.reshape(len(x),-1)
    return x/(np.sqrt(np.sum(x*x,axis=1,keepdims=True))+.00001)

def dif(x):
    h=np.concatenate((x[:,:,:1],x,x[:,:,-1:]),2); v=np.concatenate((x[:,:1],x,x[:,-1:]),1)
    return .5*(h[:,:,2:]-h[:,:,:-2]), .5*(v[:,2:]-v[:,:-2])

def fuzzy(x): return np.clip(x,0,1)

def imageparts(x):
    r,g,b=x[:,0],x[:,1],x[:,2]; lum=.28*r+.55*g+.17*b
    return r,g,b,lum

def describe(x):
    r,g,b,y=imageparts(x); sm=smooth(y); dx,dy=dif(sm)
    mag=np.sqrt(dx*dx+dy*dy)+.000001
    # Centered, unsigned orientation channels. The flat angular response is
    # removed so texture alone cannot mimic a directional boundary.
    cells=[]; contours=[]
    for j in range(6):
        angle=j*(np.pi/6); pr=np.abs(dx*np.cos(angle)+dy*np.sin(angle))/mag
        cells.append(grid(smooth((np.maximum(pr-.77,0)/.23-.18)*np.sqrt(mag)*mag/(mag+.012)),8))
        # Keep the prominent, persistent structures in a second edge map, for
        # comparisons less sensitive to incidental textures on an object.
        contours.append(grid(smooth((pr**4-.375)*mag),8))
    hog=np.stack(cells,3); strength=np.sqrt((hog*hog).sum(3,keepdims=True))
    hog=hog/np.sqrt(strength+.023)
    coarse=np.stack([grid(smooth(v),4) for v in contours],3)
    central=hog[:,1:7,1:7]
    gap=grid(sm,8); ga=gap-gap.mean((1,2),keepdims=True); detail=gap-smooth(smooth(gap)); energy=grid(np.sqrt(mag),8); energy-=energy.mean((1,2),keepdims=True)
    # Chromatic adaptation with a floor: do not amplify noise in shadows.
    rg=(r-g)/(.23+y); gb=(g-b)/(.23+y); sat=x.max(1)-x.min(1)
    veget=fuzzy(.41-rg*5)*fuzzy(gb*5+.13)*fuzzy(g*6)
    sky=fuzzy(.24-3.7*gb)*fuzzy(.48-3.5*rg)
    warm=fuzzy(rg*3+.13)*fuzzy(gb*3.7+.24)
    neutral=fuzzy(1.2-sat/(y+.22)*4.5)
    red=fuzzy((r-np.maximum(g,b)-.075)*5); golden=fuzzy(gb*3.5)*fuzzy(rg*3+.46)
    # A coarse map of soft material/color memberships; RBF tones distinguish
    # asphalt, white overcast sky and deep shadows without a hard label.
    maps=[veget,sky,warm,neutral*.76,red,golden*.67]
    for t in [.14,.47,.82]: maps.append(np.exp(-((y-t)/.235)**2)*.71)
    color=np.stack([grid(smooth(v),4) for v in maps],3)
    # These fixed reference levels are not statistics of a collection of views.
    color-=np.array([.18,.18,.20,.28,.05,.13,.15,.25,.15],np.float32)
    return np.concatenate([unit(hog)*.23**.5,unit(central)*.16**.5,unit(coarse)*.21**.5,unit(ga)*.17**.5,unit(detail)*.14**.5,unit(energy)*.09**.5],1).astype(np.float32),unit(color).astype(np.float32)
'''

runtime=r'''
def context(x):
    r,g,b,y=imageparts(x); dx,dy=dif(smooth(y)); norm=np.sqrt(dx*dx+dy*dy)+.000001
    def avg(t): return t.mean((1,2))
    def upper(t): return avg(t[:,:9])
    def lower(t): return avg(t[:,24:])
    def middle(t): return avg(t[:,8:25,6:26])
    def surround(t): return (upper(t)+lower(t)+.5*(avg(t[:,:,0:5])+avg(t[:,:,-5:])))/3
    def squash(t): return np.clip(t,-1,1)
    rg=(r-g)/(.22+y); gb=(g-b)/(.22+y); sat=x.max(1)-x.min(1)
    leafy=fuzzy(.46-5.5*rg)*fuzzy(6.5*gb+.11)*fuzzy(7*g)
    blue=fuzzy(.22-5*gb)*fuzzy(.55-4*rg)
    earth=fuzzy(.14+3.8*rg)*fuzzy(.28+4.2*gb)
    cool=fuzzy(.59-2.8*rg-1.9*gb); warm=fuzzy(2.4*rg+2.3*gb+.12)
    pale=fuzzy((y-.52)*3)*fuzzy(1.2-3.5*sat); dark=fuzzy((.31-y)*4.7)
    hi=upper(blue); lo=lower(blue); leaf=surround(leafy); dirt=surround(earth); field=lower(leafy); cold=surround(cool)
    # Smooth expanses and horizontally stratified backgrounds (sky, sea, road).
    topgrad=upper(norm); botgrad=lower(norm); centergrad=middle(norm); smoothsky=np.exp(-topgrad/.025)*fuzzy(upper(y)*1.6)*fuzzy(upper(cool)+upper(pale))
    cleanbelow=np.exp(-botgrad/.034); bothclean=np.exp(-(topgrad+botgrad)/.046)
    whiteabove=upper(pale); skyabove=fuzzy(hi*1.4+.78*whiteabove)*(.45+.55*np.exp(-topgrad/.046))
    def horiz(dx,dy):
        ax=avg(abs(dx)); ay=avg(abs(dy)); return squash(3*(ay-ax)/(.016+ax+ay))
    horizont=horiz(dx,dy); basehoriz=horiz(dx[:,18:],dy[:,18:]); vert=-horiz(dx[:,15:29,4:28],dy[:,15:29,4:28]);
    # Local orientation coherence separates straight panels from fur or foliage.
    tensor=np.sqrt(grid(dx*dx-dy*dy,8)**2+grid(2*dx*dy,8)**2)/(grid(norm*norm,8)+.00012)
    cohere=squash((avg(tensor)-.48)*3.2)
    align=avg((abs(dx*dx-dy*dy)-abs(2*dx*dy))/(norm+.004))/(avg(norm)+.01); square=squash(align*4)
    red=fuzzy((r-np.maximum(g,b)-.10)*5); flash=middle(red)+middle(fuzzy((b-np.maximum(r,g)-.12)*5));
    # Bottom-vs-middle contrast of horizontal gradients (shadow under chassis).
    strip=grid(y,8); stripavg=strip[:,:,1:7].mean(2); shadow=fuzzy(((stripavg[:,6:8].mean(1)-stripavg[:,4:7].min(1))-.035)*4)
    neutral=fuzzy(1.15-4.*sat/(y+.22)); pavement=lower(neutral)*cleanbelow
    animal=1.22*leaf+.72*dirt-.68*cold-.47*skyabove-.46*square-.27*cohere-.24*horizont
    vehicle=-.83*leaf-.53*dirt+.41*cold+.46*square+.21*cohere+.20*horizont
    midwarm=middle(warm); middark=middle(dark); texture=fuzzy((centergrad-.027)*13); brownground=lower(earth)
    skyexp=smoothsky; water=lo+.40*lower(cool); forest=upper(leafy); centralLeaf=middle(leafy)
    animal*=.8; vehicle*=.8
    sc=[vehicle+.98*bothclean+.81*hi+.40*lo+.49*skyexp-.56*horizont-.28*square-.22*shadow-.24,
        vehicle+.67*pavement+.46*shadow+.44*cleanbelow+.35*flash+.20*middark-.37*skyexp-.36*lo-.18,
        .52*animal+.54*bothclean+.32*centralLeaf+.32*forest+.39*hi+.18*flash-.21*square-.10,
        .62*animal+.36*midwarm+.23*texture-.62*field-.28*forest-.42*skyabove-.30*cold+.24*surround(neutral)+.09,
        animal+.55*leaf+.35*brownground+.39*midwarm+.27*forest+.20*vert-.28*flash-.12,
        .70*animal+.30*midwarm+.18*field+.19*texture-.40*skyabove-.26*centralLeaf+.10,
        animal+.79*centralLeaf+.49*brownground+.27*texture-.62*skyabove-.22*vert-.31*cleanbelow,
        animal+.67*field+.50*skyabove+.29*vert+.29*middark-.22*centralLeaf-.10,
        vehicle+.96*water+.68*horizont+.49*basehoriz+.42*skyabove-.38*square-.25*shadow-.20,
        vehicle+.32*pavement+.40*square+.27*skyabove+.25*flash+.31*shadow-.34*lo-.15]
    return np.stack(sc,1)

def parts(x):
    r,g,b,y=imageparts(x)
    # Matched features on the query only: dark wheels in an arch, eyes and nose,
    # thin legs in front of a field, and straight boundaries of flat panels.
    sm=smooth(y); dx,dy=dif(sm); n=len(y)
    def moved(a,iy,ix): return np.roll(a,(iy,ix),axis=(1,2))
    def mx(a): return a.reshape(n,-1).max(1)
    def av(a): return a.mean((1,2))
    def signed(a): return np.clip(a,-1,1)
    def sample(a,iy,ix): return moved(a,-iy,-ix)
    def stripe(a,iy,ix):
        v=a
        for shift in [-4,-2,2,4]: v=v+moved(a,iy*shift,ix*shift)
        return abs(v)*.20
    # Compare horizontal, vertical and diagonal supporting lines, not just
    # individual pixel gradients. Allocate evidence for right-angled bodies.
    horizontal=stripe(dy,0,1)[:,4:28,5:-5]; vertical=stripe(dx,1,0)[:,4:28,5:-5]
    diag=(stripe((dx-dy)*.707,1,1)[:,4:28,5:-5]+stripe((dx+dy)*.707,1,-1)[:,4:28,5:-5])*.5
    def energy(a): return np.sqrt(av(a*a*a*a)+1.e-12)**.5
    hv=energy(horizontal); vv=energy(vertical); dv=energy(diag); ortho=signed(3*((hv+vv)*.5-dv)/(.027+dv+hv*.3))
    tall=signed(4*(energy(vertical[:,:13])-dv)/(.048+hv+vv)); base=signed(3*(energy(horizontal[:,10:])-dv)/(.039+hv+vv))
    # Matched wheel pairs: a dark curved tire and a possible bright hub.
    # Corners and straight shadow boundaries do not enclose a disk.
    wheel=np.zeros(n,np.float32); eyes=np.zeros(n,np.float32)
    for radius in [2,3,4]:
        off=radius+1; dg=max(1,int(round(off*.707))); inside=max(1,radius-1)
        ring=(sample(sm,0,inside)+sample(sm,0,-inside)+sample(sm,-inside,0)+sample(sm,inside,0))*.25
        corners=[sample(sm,j,i) for i,j in [(off,0),(-off,0),(0,-off),(0,off),(dg,-dg),(-dg,-dg),(dg,dg),(-dg,dg)]]
        arc=sum(corners)/8; px=np.minimum(corners[0],corners[1]); py=np.minimum(corners[2],corners[3]); rim=.65*(arc-ring)+.30*(np.minimum(px,py)-ring)+.20*np.maximum(sm-ring,0)
        rim=np.maximum(0,rim)*(.55+.45*fuzzy((.60-ring)*2.5)); bestx=np.zeros(n,np.float32)
        for gap in [3*radius,4*radius,5*radius,6*radius]:
            if gap>23: continue
            for ddy in [-2,0,2]:
                other=sample(rim,ddy,gap); a=rim[:,16:28,off:32-gap-off]; q=other[:,16:28,off:32-gap-off]
                if a.shape[2]>0: bestx=np.maximum(bestx,mx(.83*np.minimum(a,q)+.17*np.maximum(a,q)))
        wheel=np.maximum(wheel,bestx)
        # A bilateral dark eye pair; look for a dark nose on the middle axis,
        # but allow a contrasting short muzzle or forehead to support one eye.
        blob=np.maximum(0,arc-sm+.2*(py-sm)); ew=radius
        for gap in [4*ew,6*ew]:
            if gap>18: continue
            left=blob; right=sample(blob,0,gap); nose=.5*sample(blob,gap//2,gap//2)+.5*sample(blob,3*gap//4,gap//2)
            pair=np.minimum(left,right)+.18*np.minimum((left+right)*.5,nose)
            eyes=np.maximum(eyes,mx(pair[:,5:19,off:32-gap-off]))
    wheel=signed((wheel-.095)*6); eyes=signed((eyes-.105)*3.7)
    # Stripes representing long, narrow legs, typically against lighter grass.
    darkline=.5*(sample(sm,0,2)+sample(sm,0,-2))-sm; leg=np.maximum(0,(darkline+sample(darkline,2,0)+sample(darkline,-2,0))/3)
    legs=np.zeros(n,np.float32)
    for gap in [4,7,10,14]: legs=np.maximum(legs,mx(np.minimum(leg,sample(leg,0,gap))[:,16:28,3:30-gap]))
    legs=signed((legs-.072)*6.)
    # Local roughness and blue open space around the subject. Frogs tend to
    # occupy the texture right to the margins, unlike perched silhouettes.
    rough=abs(y-sm); border=.5*(av(rough[:,:6])+av(rough[:,26:])); grain=signed((border-.024)*19.)
    contrast=av(abs(sm[:,6:26,6:26]-sm[:,6:26,6:26].mean((1,2),keepdims=True)))
    cornersmooth=np.exp(-border/.019); central=signed((contrast-.105)*7); similar=np.zeros(n,np.float32)
    for shift in [-3,0,3]:
        rhs=sample(sm[:,:,::-1],0,shift); diff=av(abs(sm-rhs)[:,6:24,6:26]); similar=np.maximum(similar,np.exp(-diff/(.027+.55*contrast)))
    sym=signed((similar-.34)*2.9); face=eyes*(.78+.25*sym)+.20*sym
    # Achromatic manufactured colors and the conspicuous yellow-green of
    # many amphibians are useful when the background occupies little area.
    sat=x.max(1)-x.min(1); lime=av((fuzzy((g-r+.022)*7)*fuzzy((g-b-.052)*6))[:,8:25,6:26]);
    skycolor=fuzzy(.35+(b-r)*6)*fuzzy(.50+(b-g)*6); skycolor=av(skycolor[:,:10]);
    man=ortho*.28-grain*.25; natural=-.25*ortho+.22*grain
    return np.stack([-.22*ortho-.26*base-.30*grain-.13*legs+.13*central,
        man+.83*wheel+.30*base-.17*tall+.15*central,
        -.28*ortho+.32*cornersmooth+.16*central-.18*legs-.19*base,
        natural+.44*face+.13*central-.15*legs,
        natural+.46*legs-.26*face-.10*sym,
        natural+.33*face+.17*legs,
        natural+.65*lime+.13*face+.27*grain-.25*legs-.12*central,
        natural+.58*legs+.09*tall-.14*face+.13*central,
        -.10*ortho+.42*base-.15*wheel+.16*skycolor,
        man+.47*wheel+.46*tall+.32*ortho+.12*base],1)

def match(x):
    cues=context(x)+parts(x)
    shape,color=describe(x); flipshape,flipcolor=describe(x[:,:,:,::-1].copy())
    # Either handedness is a valid view. Pair colors with the correct mirror.
    sh=shape@CONSTANTS['shape'].T; shf=flipshape@CONSTANTS['shape'].T
    co=color@CONSTANTS['color'].T; cof=flipcolor@CONSTANTS['color'].T
    def best(a,b):
        vals=np.maximum(a,b).reshape(len(x),10,-1); p=np.partition(vals,-5,axis=2)[:,:,-5:]
        return .50*p.max(2)+.50*p.mean(2)
    s=best(.79*sh+.21*co,.79*shf+.21*cof)
    c=best(.31*sh+.69*co,.31*shf+.69*cof)
    return 15*s+6*c+1.22*cues

def predict(x):
    out=np.empty(len(x),np.int64)
    for i in range(0,len(x),48):
        n=min(48,len(x)-i); z=np.zeros((48,3,32,32),np.float32); z[:n]=x[i:i+n]; out[i:i+n]=match(z)[:n].argmax(1)
    return out
'''

def build(dest):
    dest.mkdir(exist_ok=True,parents=True); (dest/'solution.py').write_text(features+runtime)
    tools={}; exec(features,tools); describe=tools['describe']; shapeset=[]; colorset=[]
    global R
    for k in range(10):
        R=np.random.default_rng(18731+259*k)
        for j in range(96):
            # Stack only for parallel arithmetic: every operation in describe
            # retains its leading (view) axis until returning its own template.
            a,b=describe(np.stack([render(k) for i in range(32)])); shapeset.append(a); colorset.append(b)
        print('Rendered class',k,flush=True)
    np.savez(dest/'constants.npz', shape=np.concatenate(shapeset),color=np.concatenate(colorset))

if __name__=='__main__': build(Path(sys.argv[1]))
