"""Build a pictorial-model classifier. No input files; all dimensions are design coordinates."""
import sys, json, math
from pathlib import Path
import numpy as np

D = Path(sys.argv[1]); D.mkdir(parents=True,exist_ok=True)
rng = np.random.default_rng(713045)
u = rng.uniform

def pick(*v): return v[int(rng.integers(len(v)))]
def coin(p=.5): return u()<p
def col(hexstr): return np.array(tuple(bytes.fromhex(hexstr)),dtype=float)/255
def paint(*colors): return col(pick(*colors))*u(.83,1.12)
Y,X = np.mgrid[:32,:32].astype(float)/16 - 31/32
BLACK = col('202126'); WHITE = col('ece8da'); GLASS=col('566978')
SKY=('7caacb','b0cddd','6597c4','b8c0c6','d6d9d8','8395a5','5596cf','e0d9c8')
WATER=('446c83','617c8b','81929b','395e79','437e90','717e75','698aa7','8a8b86')
LEAF=('748047','58633e','364c35','797454','4f5b37','a3a26b','736349','778560','928467')
GRASS=('7a8846','9b9a5c','82925b','727c3a','b6a56a','596843','857c50')
EARTH=('867856','665743','99846d','a39985','735f4a','626450','b39b7a')
ROAD=('626063','797a7c','8a8986','9d958a','656370','b4aca1')
ROOM=('ac9988','c8c2b4','8f8b8a','6f655c','635143','a3937e','b9b9b3','998681','716961')
COAT=('81624a','78533b','9e7953','b7a188','d1c6b0','574d43','423e39','ac8258','867e75','e3d9c7')
METAL=('e4e3dd','babcc0','8f989e','393e42','353942','8f3a33','ae3732','385772','567693','e4d8bd','b1afb1','546459')

def noise_field(scale):
    # A small continuous patch of irregular illumination; no image assets.
    g=rng.uniform(-1,1,(scale+1,scale+1)); q=(np.arange(32)+.5)*scale/32; i=q.astype(int); t=q-i
    a=g[i[:,None],i]*(1-t)+g[i[:,None],i+1]*t
    b=g[i[:,None]+1,i]*(1-t)+g[i[:,None]+1,i+1]*t
    return a*(1-t[:,None])+b*t[:,None]

class Canvas:
    def __init__(self,k):
        self.k=k
        scenes=[('air','air','air','air','airport','airport'),('road','road','road','street'),('woods','woods','woods','meadow','air','waterbird'),('room','room','room','woods','garden'),('woods','woods','garden','meadow'),('room','room','garden','meadow','woods'),('woods','garden','mud','mud','woods'),('meadow','meadow','garden','woods','arena'),('marine',)*6,('road','street','road','street')]
        s=pick(*scenes[k]); self.scene=s
        top=paint(*SKY); bot=paint(*GRASS); horizon=u(-.3,.4); rough=.022; flat=False
        if s=='air': bot=top*u(.9,1.2)+u(0,.085); horizon=u(-.5,.85); flat=True; rough=u(.012,.09)
        if s=='airport': bot=paint(*pick(ROAD,GRASS)); horizon=u(.05,.64)
        if s in ('marine','waterbird'): bot=paint(*WATER); horizon=u(-.63,.3); rough=.019
        if s in ('road','street'): bot=paint(*ROAD); horizon=u(-.22,.35); rough=.05
        if s=='meadow': horizon=u(-.83,-.18); rough=.08
        if s in ('woods','garden','mud'):
            top=paint(*pick(LEAF,LEAF,EARTH)); bot=paint(*({'woods':LEAF,'garden':GRASS,'mud':EARTH}[s])); rough=u(.1,.24); horizon=u(-.8,.5); flat=True
        if s in ('room','arena'): top=paint(*ROOM); bot=paint(*pick(ROOM,EARTH)); flat=coin(.55); rough=u(.045,.16)
        a=noise_field(3); b=noise_field(9); c=noise_field(18)
        ridge = horizon+u(-.06,.06)*X+(0 if s in ('marine','air') else a*.14)
        blend=np.clip(.5+(Y-ridge)*(u(.8,2.6) if flat else u(7,35)),0,1)
        self.im=top+(bot-top)*blend[:,:,None]; self.x=X; self.y=Y
        if s in ('street','road','meadow','airport') and coin(.84):
            far=paint(*pick(LEAF,ROOM) if s in ('street','road') else LEAF)
            skyline=ridge-u(.08,.48)-noise_field(6)*u(.07,.24)
            mask=np.clip((Y-skyline)*18+.5,0,1)*(1-blend)
            self.im+=(far-self.im)*mask[:,:,None]
            if s in ('street','road') and coin():
                for _ in range(int(rng.integers(1,5))):
                    x=u(-1,1); w=u(.16,.6); y=horizon-u(.2,.65)
                    self.poly([(x,y),(x+w,y),(x+w,horizon),(x,horizon)],paint(*ROOM))
        n=rough*(a+.63*b+.32*c)
        if s in ('marine','waterbird'): n+=blend*(.04*np.sin(Y*u(30,65)+2*a)+.055*c)
        if s in ('woods','garden','mud'): n+=.048*rng.uniform(-1,1,(32,32))
        self.im+=n[:,:,None]*np.array([1,.97,.84]); self.grain=noise_field(10)
    def pose(self,sx,sy,angle,dx,dy):
        c,s=np.cos(angle),np.sin(angle); a=X-dx; b=Y-dy
        self.x=(a*c+b*s)/sx*pick(-1,1); self.y=(b*c-a*s)/sy
    def put(self,alpha,color): self.im+=(color-self.im)*np.clip(alpha,0,1)[...,None]
    def ellipse(self,x,y,w,h,color,ang=0,shade=.12):
        a=self.x-x; b=self.y-y; c,s=np.cos(ang),np.sin(ang)
        t=(a*c+b*s)/w; v=(b*c-a*s)/h; d=t*t+v*v
        # Directional light and rounded cross-section.
        q=np.clip(.5+(1-d)*min(w,h)*9,0,1)
        fill=color*(1-shade*(v*.65+np.minimum(d,1)*.72))[...,None] if shade else color
        self.put(q,fill)
    def poly(self,pts,color):
        # Scan conversion, also works for concave silhouettes.
        q=np.zeros((32,32),dtype=bool); x1,y1=pts[-1]
        for x2,y2 in pts:
            if abs(y2-y1)>1e-7: q^=((self.y>y1)!=(self.y>y2))&(self.x<(x2-x1)*(self.y-y1)/(y2-y1)+x1)
            x1,y1=x2,y2
        self.put(q,color)
    def seg(self,x1,y1,x2,y2,w,color):
        dx=x2-x1; dy=y2-y1
        f=np.clip(((self.x-x1)*dx+(self.y-y1)*dy)/(dx*dx+dy*dy+1e-8),0,1)
        d=np.hypot(self.x-x1-f*dx,self.y-y1-f*dy); self.put(.5+(w-d)*16,color)
    def box(self,x,y,w,h,color): self.poly([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],color)
    def finish(self): return np.clip(self.im,0,1)

def airplane(g):
    e,p,l=g.ellipse,g.poly,g.seg; m=paint('dadde0','f0ebe0','a1a8aa','737e84','4e5960','b7b6ac','495763'); trim=paint(*METAL)
    if coin(.23): m*=u(.32,.61)
    view=pick('side','side','quarter','quarter','front','below'); tilt=u(-.32,.22) if view!='below' else u(-1.45,1.45)
    sc=u(.62,1.2); g.pose(sc,sc*u(.83,1.25),tilt,u(-.12,.12),u(-.17,.16))
    if view in ('side','quarter'):
        a=u(.2,.63) if view=='quarter' else u(.02,.22)
        p([(-.58,.0),(-.83,-.34),(-.69,-.33),(-.36,-.04)],trim)
        p([(-.08,.01),(-.48,-a),(-.41,-a),(.24,-.02)],m*.76)
        p([(-.89,-.025),(.52,-.1),(.76,-.068),(.94,.005),(.9,.05),(.36,.098),(-.59,.065)],m)
        p([(-.72,.01),(-.91,.14),(-.79,.16),(-.45,.05)],m*.8)
        p([(-.04,.026),(-.48,a+.045),(-.32,a+.027),(.29,.044)],m*.74)
        if coin(.75):
            e(.055,a*.43+.09,.13,.053,m*.52); l(-.51,-.003,.50,-.017,.017,trim*.66)
            p([(.57,-.085),(.72,-.068),(.79,-.037),(.55,-.044)],GLASS*.65)
    elif view=='front':
        p([(-1,-.13),(-.8,-.07),(0,.035),(.81,-.07),(1,-.13),(.81,.005),(0,.14),(-.83,.008)],m*.85)
        p([(-.045,0),(-.018,-.47),(.038,-.48),(.05,.05)],trim); l(-.32,-.12,.32,-.12,.032,m)
        for x in [-.44,.44]: e(x,.09,.095,.087,BLACK)
        e(0,.028,.16,.19,m); e(0,-.013,.10,.035,GLASS*.56)
    else:
        p([(-.09,-.25),(-.86,.24),(-.84,.39),(-.06,.1),(.08,.1),(.84,.39),(.86,.24),(.09,-.25)],m*.89)
        p([(0,.43),(-.31,.68),(-.29,.72),(0,.62),(.3,.72),(.32,.67)],trim); e(0,-.01,.105,.78,m)
        for x in (-.33,.33): e(x,-.035,.043,.13,m*.72)

def vehicle(g,truck):
    e,p,l,b=g.ellipse,g.poly,g.seg,g.box
    sc=u(.69,1.28); g.pose(sc,sc*u(.84,1.16),u(-.095,.095),u(-.11,.11),u(-.14,.13))
    m=paint(*METAL); glass=paint('43535d','627582','2a3036','8d9da4','6b7379'); tire=BLACK*u(.65,1.23); light=WHITE*u(.8,1)
    front=coin(.22 if truck else .20); z=u(.09,.37)
    if front:
        a=u(.45,.69); top=-u(.55,.8) if truck else -u(.26,.47); base=u(.27,.47)
        for x in [-a*.8,a*.8]: e(x,base-.01,.105,.14,tire)
        p([(-a,0),(-a*.78,top),(-a*.67,top-.045),(a*.67,top-.045),(a*.8,top),(a,.02),(a,base),(-a,base)],m)
        p([(-a*.9,-.01),(-a*.72,top+.057),(a*.72,top+.057),(a*.88,-.01)],glass)
        if truck: l(0,top+.05,0,-.02,.017,m); b(-a*.56,.09,1.12*a,.17,tire)
        else: p([(-a,-.01),(a,-.01),(a*.93,.17),(-a*.95,.17)],m*1.12)
        l(-a*.91,base-.028,a*.91,base-.028,.04,m*.54)
        for x in [-a*.81,a*.81]: e(x,.14,.09,.049,light)
        b(-.095,base-.09,.19,.055,light*.77); return
    base=u(.22,.38); wx=.14 if truck else u(.13,.195); end=.88; near=end-z
    e(-.02,base+.085,.91,.105,tire*.9)
    if not truck:
        roof=-u(.28,.46); belt=-u(.015,.13); back=-u(.32,.58); fore=u(.17,.40)
        p([(-.91,belt+.055),(back-.18,belt),(back+.04,roof+.025),(back+.18,roof),(fore-.12,roof+.015),(fore+.17,belt),(near,belt+.035),(end,belt+.13),(end,base-.035),(near,base),(-.87,base)],m)
        p([(back-.105,belt),(back+.078,roof+.066),(fore-.16,roof+.062),(fore+.045,belt)],glass)
        l((fore+back)/2,roof+.04,(fore+back)/2+.02,belt,.025,m)
        p([(fore-.12,roof+.032),(fore-.12+z*.6,roof+.069),(fore+.19+z*.61,belt+.014),(fore+.13,belt)],glass*1.12)
        p([(near,belt+.078),(end,belt+.13),(end,base-.042),(near,base)],m*.78)
        l(-.80,base-.075,near-.09,base-.056,.018,m*.62); b(near+.02,belt+.15,z*.61,.05,light)
        wheels=[-.57,near-.07]
    else:
        roof=-u(.42,.72); cargo=paint('e8e6dc','d5d4ca','adada9','908d81','904f39','757c85','54728b','ded9c7','68715d'); cut=u(.16,.38)
        subtype=pick('box','box','box','tanker','bed','bus')
        b(-.87,base-.10,1.65,.115,tire)
        if subtype=='tanker': e((cut-.92)/2,(roof+.07)/2,(cut+.92)/2,(.08-roof)/2,cargo)
        else:
            ry=roof if subtype!='bed' else roof*.32
            p([(-.94,ry+.05),(-.82,ry),(cut,ry),(cut,.12),(-.92,.14)],cargo)
            l(-.90,.11,cut-.01,.10,.02,cargo*.6)
            if subtype=='bus': b(-.88,ry+.11,cut+.87,.19,glass)
            if coin(.34): l(-.89,(ry+.1)/2,cut-.035,(ry+.1)/2,.035,m*.83)
        cab=roof+u(.04,.22) if subtype!='bed' else roof
        p([(cut+.023,cab),(near-.13,cab),(near+.027,-.105),(end,-.05),(end,base-.035),(cut+.023,base-.013)],m)
        p([(cut+.06,cab+.07),(near-.13,cab+.072),(near-.04,-.09),(cut+.06,-.105)],glass)
        p([(near-.11,cab+.068),(near+z*.35,cab+.10),(end-.02,-.096),(near+.024,-.089)],glass*1.13)
        p([(near+.025,-.08),(end,-.065),(end,base-.04),(near+.04,base)],m*.78)
        b(near+.044,.045,z*.64,.098,tire*.96); wheels=[-.66,-.39,near-.04] if coin(.66) else [-.63,near-.04]
    for x in wheels:
        e(x,base,wx*(1-.48*z),wx,tire); e(x,base,wx*.49,wx*.49,m*.54+.14)

def ship(g):
    e,p,l,b=g.ellipse,g.poly,g.seg,g.box; sc=u(.65,1.26); g.pose(sc,sc*u(.78,1.23),u(-.14,.14),u(-.14,.14),u(-.14,.13))
    hull=paint('3e4445','453e3a','563e37','c9cbc6','e6e5da','6c7a82','3d5166'); deck=paint('e8e9dd','cdd1cb','c1bfb7','9d9383','d7d6d2'); kind=pick('sail','sail','cruise','navy','freighter','freighter','launch','fishing'); h=u(.09,.20); tip=u(.68,.96)
    if kind=='sail':
        mid=u(-.22,.14); mast=-u(.58,.97); l(mid,h,mid,mast,.014,hull*.7)
        p([(mid-.025,mast+.04),(mid-.058,h-.08),(-.62,h-.038)],deck)
        p([(mid+.018,mast+.16),(mid+.04,h-.09),(tip*.8,h-.023)],deck*.88); h*=.87
    elif kind in ('cruise','navy','fishing','launch'):
        n=3 if kind in ('cruise','navy') else 1; left=-u(.46,.72); right=u(.13,.48); height=u(.095,.19)
        for j in range(n):
            y=h-(j+1)*height; p([(left,y+.02),(right-.11,y),(right,y+height),(left-.02,y+height)],deck*(1-.04*j)); l(left+.035,y+.064,right-.095,y+.064,.023,GLASS*.7); left+=.10;right-=.075
        x=pick(left+.04,right-.12); b(x,y-.16,.092,.18,hull*.86); l(x+.13,y+.04,x+.13,y-.30,.014,deck*.6)
    else:
        x=pick(-.67,.35); b(x,h-.28,.23,.31,deck); b(x-.02,h-.29,.26,.075,GLASS)
        for j in range(4):
            xx=-.55+j*.235
            if abs(xx-x)>.16: b(xx,h-u(.095,.21),.225,.22,paint('925a46','7b7463','576877','73594a'))
    p([(-.88,h-.025),(tip,h-.056),(tip-.13,h+.108),(tip-.28,h+.16),(-.71,h+.16)],hull)
    l(-.72,h+.156,tip-.27,h+.156,.027,hull*.66); l(-.72,h+.193,tip-.24,h+.182,.019,WHITE*.86)

def ungulate(g,horse):
    p,e,l=g.poly,g.ellipse,g.seg; s=u(.65,1.23); g.pose(s,s*u(.9,1.22),u(-.14,.14),u(-.14,.14),u(-.13,.12))
    c=paint(*('805038','996242','634537','4a342c','a2744e','c0aca0','ddd5c4','483d34') if horse else ('947751','ae8c66','967b5b','816e53','b29876','78644f')); mane=c*pick(.30,.38,.55,.7); hoof=c*.41
    if coin(.19 if horse else .24):
        # A closer view of the neck and head, with ears high against the background.
        e(-.06,.63,.44,.90,c,-.15,.23); e(.08,-.105,.33,.54,c,-.24,.21)
        for a in (-1,1):
            ex=.06+a*.24; angle=a*(.22 if horse else .82); e(ex,-.62,.078 if horse else .105,.245,c,angle); e(ex,-.65,.037,.14,c*.60,angle); e(.09+a*.237,-.20,.049,.039,BLACK)
        e(.19,.11,.202,.34,c*.94,-.20);e(.22,.34,.20,.13,c*.68)
        for x in [.11,.32]:e(x,.343,.039,.028,hoof)
        if horse:e(-.004,-.48,.16,.23,mane,.44)
        return
    trot=u(-.36,.39) if coin(.42) else u(-.14,.14); thick=.057 if horse else .036
    for i,(x,shift) in enumerate([(-.43,-.11),(.31,-.105),(-.49,.025),(.31,.07)]):
        far=i<2; color=c*(.66 if far else .92); ox=trot*(1 if i in [0,3] else -1)+shift
        knee=x+ox*.53+(-.085 if x<0 else .018); end=x+ox; y=u(.60,.76)-.05*far
        l(x,.09,knee,.37,thick*1.15,color); l(knee,.37,end,y,thick*.68,color); l(end-.012,y,end+.032,y+.002,thick*.85,hoof)
    e(-.11,-.015,.55,.24 if horse else .21,c,-.028,.25); e(-.42,.014,.23,.25 if horse else .21,c,.12,.21)
    neck=u(-.20,.16); e(.39,-.246,.16 if horse else .11,.385,c,.51+neck,.22)
    hx=.54; hy=-.48; e(hx,hy,.17,.19 if horse else .142,c,-.21,.19); e(hx+.141,hy+.077,.23 if horse else .17,.099,c,.53,.2)
    e(hx+.306,hy+.165,.078,.064,c*.59,.45);e(hx+.066,hy-.027,.034,.031,BLACK)
    for j in [-1,1]:e(hx+j*.085,hy-.145,.044 if horse else .058,.119,c,j*.46)
    if horse:
        l(.21,-.085,.453,-.526,.056,mane);e(-.672,.14,.084,.285,mane,.31,.2)
    else:
        e(-.64,-.008,.068,.117,c*1.23,-.7)
        if coin(.42):
            for xx,sg in [(hx-.09,-1),(hx+.03,1)]:
                l(xx,hy-.10,xx+sg*.11,hy-.37,.023,c*.69);l(xx+sg*.055,hy-.24,xx+sg*.2,hy-.35,.018,c*.69);l(xx+sg*.11,hy-.36,xx+sg*.22,hy-.48,.018,c*.69)

def pet(g,cat):
    p,e,l=g.poly,g.ellipse,g.seg; s=u(.71,1.30);g.pose(s,s*u(.86,1.16),u(-.24,.24),u(-.17,.17),u(-.14,.12))
    c=paint(*COAT); patch=paint(*COAT) if coin(.36) else c*u(.55,.88); muzzle=WHITE*u(.7,1) if coin(.43) else c*1.18; eyes=paint('8e985a','919a75','bc9958') if cat else paint('5a4731','382e25'); nose=paint('ab8074','755c55','4a4342') if cat else BLACK
    full=coin(.40); sit=coin(.57); prof= u(-.08,.13) if not full else u(.02,.14)
    if full:
        if sit:
            e(-.06,.19,.31,.48,c,-.11,.19);e(-.24,.38,.23,.27,c,-.3);e(.13,.36,.089,.29,c,.04);e(.12,.62,.127,.071,muzzle)
            l(-.35,.48,-.58,.29,.067 if cat else .091,patch);hx=.052;hy=-.22; f=.58
        else:
            e(-.13,.012,.50,.232,c,.035,.19)
            for x,dx in [(-.48,-.08),(-.32,.10),(.15,-.055),(.27,.095)]:l(x,.12,x+dx,.48,.050,c*.91);e(x+dx+.037,.48,.092,.045,c)
            l(-.55,.04,-.82,-.16,.069,patch);l(-.82,-.16,-.86,-.40,.050,patch);hx=.38;hy=-.21; f=.50;prof+=.14
    else:
        e(-.02,.74,.54,.60,c,-.20,.25);hx=0;hy=-.033; f=u(.96,1.32)
    def E(x,y,w,h,color,a=0,sh=.13):e(hx+f*x,hy+f*y,w*f,h*f,color,a,sh)
    def P(pts,color):p([(hx+f*x,hy+f*y) for x,y in pts],color)
    ear=pick('drop','drop','drop','prick') if not cat else 'prick'; wid=.43 if cat else .40
    for sign in [-1,1]:
        if ear=='prick':
            P([(sign*.11,-.25),(sign*.408,-(.57 if cat else .66)),(sign*.39,-.038)],c*.91)
            P([(sign*.20,-.26),(sign*.377,-(.48 if cat else .54)),(sign*.33,-.14)],patch*.7+.06)
        else:E(sign*.38,-.027,.145,.32,patch,-sign*.19,.27)
    E(0,0,wid,.362 if cat else .42,c,-.035,.22)
    if not cat and coin(.56): E(-.18,-.10,.16,.20,patch,.25)
    if cat and coin(.65):
        for a in [-1,0,1]:E(a*.08,-.23,.019,.088,c*.55,a*.24)
    for a in [-1,1]:
        x=prof+a*.183; E(x,-.076,.083 if cat else .064,.047,BLACK,a*(-.15)); E(x,-.076,.061 if cat else .038,.032,eyes,-a*.15); E(x,-.076,.019,.029,BLACK);E(x-.015,-.09,.012,.011,WHITE)
    if cat:
        for a in [-1,1]:E(prof+a*.081,.124,.109,.081,muzzle)
        P([(prof-.057,.075),(prof+.056,.075),(prof,.13)],nose)
    else:
        E(prof,.14,.213,.159,muzzle,-prof,.16)
        if coin(.46):E(prof+.006,.24,.136,.073,BLACK);E(prof+.032,.29,.079,.074,col('a6726a'))
        E(prof+.015,.096,.106,.066,nose)

def bird(g):
    p,e,l=g.poly,g.ellipse,g.seg; sc=u(.55,1.20); g.pose(sc,sc*u(.88,1.22),u(-.32,.32),u(-.18,.18),u(-.17,.14))
    c=paint('6b6252','554d44','a49272','a8a69d','d4cfbf','e0e0d6','4d5560','6a7770','626c93','a47a46','819151'); wing=c*u(.52,.89); bill=paint('b59a62','443b32','50473b','cc9a43'); flight=g.scene=='air' or coin(.20)
    if flight:
        high=u(.34,.8); p([(-.22,.09),(-.50,-high*.30),(-.79,-high),(-.38,-high*.81),(.17,-.025)],wing)
        p([(.12,.04),(-.23,.36),(-.64,high*.83),(-.49,.25),(-.34,.01)],c*.81); p([(-.23,-.024),(-.76,.064),(-.80,.19),(-.18,.14)],wing)
        e(.013,.034,.37,.124,c,-.13);e(.32,-.037,.134,.11,c);p([(.41,-.06),(.60,-.019),(.41,.003)],bill)
    else:
        water=g.scene=='waterbird' or coin(.18); hy=-.14 if water else -.32; hx=.27
        if not water:
            y=u(.38,.57); l(-1,y+u(-.23,.23),.94,y+.01,u(.024,.069),paint(*EARTH)*.72)
            for x in [-.11,.11]:l(x,.15,x-.025,y,.018,bill);l(x-.052,y+.002,x+.044,y,.014,bill)
        p([(-.25,.005),(-.78,.25 if water else .39),(-.74,.35 if water else .50),(-.09,.18)],wing)
        e(-.015,.025,.40 if water else .30,.21 if water else .33,c,.05 if water else .62,.25); e(-.13,.06,.31,.134,wing,-.29 if water else -.58,.24)
        hc=paint('354437','3c4144','5b5144','a28a6b') if coin(.44) else c;e(hx,hy,.17,.163,hc,-.15,.21);p([(hx+.12,hy-.015),(hx+.37,hy+.031),(hx+.127,hy+.071)],bill); e(hx+.077,hy-.042,.027,.028,BLACK)

def frog(g):
    p,e,l=g.poly,g.ellipse,g.seg; sc=u(.64,1.31);g.pose(sc,sc*u(.84,1.23),u(-.43,.43),u(-.14,.14),u(-.13,.13)); c=paint('8a8e48','788340','6a7847','716944','928358','685c46','9da05d','51924e','557541','93866a'); mark=c*u(.42,.70)
    front=coin(.48)
    if front:
        for a in [-1,1]:
            e(a*.39,.13,.244,.275,c,a*-.75,.28);l(a*.32,.07,a*.54,.36,.074,c*.88);l(a*.54,.36,a*.72,.37,.058,c);l(a*.56,.36,a*.68,.47,.024,c)
        e(0,.014,.45,.33,c,.02,.24); e(0,.075,.424,.179,c*1.19)
        for x in [-.245,.245]:e(x,-.194,.137,.135,c); e(x,-.212,.078,.077,paint('a79654','847649'));e(x,-.215,.057,.031,BLACK);e(x-.02,-.23,.016,.015,WHITE)
        l(-.31,.085,0,.128,.019,mark);l(0,.128,.31,.084,.019,mark)
    else:
        e(-.12,.05,.411,.255,c,-.22,.28);e(-.37,.184,.261,.143,c*.9,-.58,.24);e(-.33,.286,.32,.078,c,-.04);l(.14,.083,.258,.32,.062,c);l(.255,.32,.52,.33,.034,c)
        e(.22,-.084,.31,.185,c,-.063,.18); e(.224,-.226,.122,.103,c);e(.244,-.242,.075,.063,paint('b3a269','8a7d59'));e(.25,-.244,.055,.030,BLACK);l(.12,-.00,.49,-.06,.014,mark)
    if coin(.72):
        for _ in range(9):
            x=u(-.26,.19);y=u(-.115,.088);a=u(.021,.06);e(x,y,a,a*.70,mark,shade=.0)

DRAW=[airplane,lambda g:vehicle(g,False),bird,lambda g:pet(g,True),lambda g:ungulate(g,False),lambda g:pet(g,False),frog,lambda g:ungulate(g,True),ship,lambda g:vehicle(g,True)]

# Network builder primitives.
L=[]; W={}
def node(name,op,source,**kw):
    L.append(dict(name=name,op=op,inputs=source if isinstance(source,list) else [source],**kw)); return name

def affine(name,op,src,w,b=None,**kw):
    w=np.asarray(w,dtype='f'); W[name+'.W']=w; W[name+'.b']=np.zeros(len(w),'f') if b is None else np.asarray(b,'f'); return node(name,op,src,out=len(w),**kw)
def conv(name,src,w,b=None,stride=1,pad=0):
    w=np.asarray(w); w=w[:,:,None,None] if w.ndim==2 else w
    return affine(name,'conv',src,w,b,kernel=[w.shape[-1]]*2,stride=stride,padding=pad)
def dense(name,src,w,b=None):return affine(name,'dense',src,w,b)
def pool(name,src,size,stride=None,m=False):return node(name,'maxpool' if m else 'avgpool',src,size=list(size),stride=list(stride or size))
def rel(name,src):return node(name,'relu',src)
def norm(name,src):return node(name,'norm',src)
def shape(name,src,s):return node(name,'reshape',src,shape=list(s))
def summ(name,*src):return node(name,'add',list(src))

RGB=np.array([.28,.56,.16]); AX=np.array([RGB,[1,0,-1],[-.5,1,-.5]])
EPS=np.sqrt(1e-5); floors=np.array([.055,.058,.035]); AXS=AX*(EPS/floors)[:,None]
half=pool('half','image',(2,2)); small=pool('thumbnail',half,(2,2)); low=norm('layout_norm',conv('opponent_axes',small,AXS))

def smooth_kernel(n,s):
    i=np.arange(n)-(n-1)/2; y,x=np.meshgrid(i,i,indexing='ij'); w=np.exp(-.5*(x*x+y*y)/(s*s));return w/w.sum()
BP=smooth_kernel(5,.48)-smooth_kernel(5,1.50); bm=BP[None,None,:,:]*RGB[None,:,None,None]*(EPS/.031)
middle=norm('detail_norm',conv('center_surround',half,bm)); gray=conv('brightness','image',RGB[None,:]); angles=np.arange(4)*np.pi/4

def edges(n,across,along):
    y,x=np.mgrid[:n,:n]-(n-1)/2; bank=[]
    for t in angles:
        a=x*np.cos(t)+y*np.sin(t); b=y*np.cos(t)-x*np.sin(t);f=a*np.exp(-.5*((a/across)**2+(b/along)**2)); f=f/f.clip(0).sum();bank.append(f)
    return np.array(bank)
G=edges(4,.78,1.05); E8=np.concatenate([G,-G])[:,None]; sel=np.eye(4)-.68*np.roll(np.eye(4),2,axis=0); SEL=np.tile(sel,(1,2)); signed=conv('local_derivatives',gray,E8,stride=2); edges_pos=rel('local_halfwave',signed)
votes=rel('direction_votes',conv('opponent_directions',edges_pos,SEL)); hog=pool('gradient_cells',votes,(3,3),(2,2)); hogflat=shape('all_orientations',hog,(1,28,7)); hogfeat=norm('gradient_norm',conv('gradient_floor',hogflat,[[EPS/.026]]))

# A single multichannel template, independently normalized by feature band.
# No calculations ever compare different pictures or use batch statistics.
from numpy.lib.stride_tricks import sliding_window_view as windows
def features(im):
    halfim=im.reshape(16,2,16,2,3).mean((1,3)); thumb=halfim.reshape(8,2,8,2,3).mean((1,3))
    coarse=(thumb@AX.T).transpose(2,0,1); coarse-=coarse.mean((1,2),keepdims=True); coarse/=np.sqrt(np.mean(coarse**2,(1,2))+floors**2)[:,None,None]
    def standard(f,floor):
        f=f-f.mean();return f/np.sqrt(np.mean(f*f)+floor**2)
    band=standard(windows(halfim@RGB,(5,5)).reshape(12,12,25)@BP.ravel(),.031)
    d=windows(im@RGB,(4,4))[::2,::2].reshape(15,15,16)@G.reshape(4,16).T; d=np.abs(d);d=np.maximum(d-.68*np.roll(d,2,axis=2),0)
    cells=windows(d,(3,3),axis=(0,1))[::2,::2].mean((-2,-1)).transpose(2,0,1); h=standard(cells,.026)
    return ((coarse*(np.array([.20,.095,.055])/64)[:,None,None]).ravel(),(.29/144*band).ravel(),(.36/196*h).ravel())

GROUPS=6; PER=256; N=10*GROUPS*PER
banks=[np.empty((N,n),np.float32) for n in [192,144,196]]
for k,draw in enumerate(DRAW):
    for j in range(GROUPS*PER):
        g=Canvas(k);draw(g);f=features(g.finish()); row=k*GROUPS*PER+j
        for bank,feat in zip(banks,f):bank[row]=feat
projs=[dense(n,x,w) for n,x,w in zip(('layout_match','detail_match','contour_match'),(low,middle,hogfeat),banks)]
match=shape('pose_groups',summ('individual_matches',*projs),(10,GROUPS,PER)); best=pool('best_in_group',match,(1,PER),m=True); av=pool('repeatable_match',best,(GROUPS,1)); mx=pool('best_pose',best,(GROUPS,1),m=True)
object_score=summ('shape_evidence',dense('match_consensus',av,4.6*np.eye(10)),dense('peak_match',mx,2.1*np.eye(10)))

# Per-pixel soft color tests. Each margin is an inequality in RGB space;
# the conjunction penalizes failures, and is bounded between zero and one.
R=np.array([1.,0,0]); V=np.array([0.,1,0]); B=np.array([0.,0,1]); I=RGB
cool=B-R; warm=R-B; leaf=V-.56*R-.44*B
quiet=[( V-R,-.065,.11),(R-V,-.065,.11),(B-V,-.065,.11),(V-B,-.065,.11),(B-R,-.10,.14),(R-B,-.10,.14)]
colorspec={
 'blue':[(cool,.115,.135),(B-V,.038,.075),(I,.15,.18)],
 'pale':[(cool,.075,.12),(B-V,.010,.065),(I,.58,.22)],
 'plant':[(leaf,.056,.068),(V-R,-.006,.075),(V-B,.064,.085)],
 'olive':[(V-B,.10,.10),(leaf,.037,.063),(V-R,-.065,.10),(I,.11,.16)],
 'tan':[(R-V,.066,.095),(V-B,.053,.078),(warm,.135,.15)],
 'brown':[(R-V,.049,.085),(V-B,.025,.08),(-I,-.43,.17)],
 'neutral':quiet,
 'white':quiet+[(I,.71,.19)],
 'light':[(I,.69,.23),(B-R,-.105,.16),(B-V,-.080,.12)],
 'dark':[(-I,-.22,.15)],
 'red':[(R-V,.22,.16),(R-B,.20,.16)],
 'aqua':[(B-R,.093,.12),(V-R,.056,.09),(V-B,-.115,.12)],
}
cm=[];cb=[];owner=[]
for ci,(name,terms) in enumerate(colorspec.items()):
    for axis,target,width in terms:cm.append(-axis/width);cb.append(target/width);owner.append(ci)
colors=rel('color_deficits',conv('color_halfspaces',half,cm,cb)); w=np.zeros((len(colorspec),len(owner)));w[owner,np.arange(len(owner))]=-1
colors=rel('soft_colors',conv('color_conjunctions',colors,w,np.ones(len(colorspec)))); colors=pool('color_regions',colors,(2,2))

# A compact language for spatial rule tables. Every axis and weight below is
# from a visual hypothesis, not an empirical class profile.
ty,tx=np.mgrid[:8,:8]/7*2-1
mask={'all':np.ones((8,8)),'top':ty<-.46,'bottom':ty>.46,'center':(abs(tx)<.62)&(abs(ty)<.6),'side':abs(tx)>.53,'edge':(abs(tx)>.62)|(abs(ty)>.62),'middle':abs(ty)<.40,'ground':(ty>.05)&(ty<.72),'highside':(ty<-.10)&(abs(tx)>.54),'base':(ty>.55)&(abs(tx)<.72),'lowcorner':(ty>.41)&(abs(tx)>.53)}
mask={a:z/z.sum() for a,z in mask.items()}
def field(keys,code):
    v=np.zeros((len(keys),8,8))
    for key,region,gain in code: v[keys.index(key)]+=mask[region]*gain
    return v.ravel()
class ColorVotes:
    def __init__(self): self.keys=list(colorspec); self.table=np.zeros((10,8*8*len(self.keys)))
    def vote(self,key,reg,coeff): self.table+=np.outer(coeff,field(self.keys,[(key,reg,1)]))
cv=ColorVotes()
# Class order: plane, car, bird, cat, deer, dog, frog, horse, ship, truck.
cv.vote('blue','top',     [.87,.03,.18,-.33,-.28,-.35,-.42,.01,.38,.06])
cv.vote('pale','top',     [.51,.10,.01,-.32,-.12,-.25,-.50,.04,.36,.18])
cv.vote('blue','bottom',  [.98,-.18,.15,-.12,-.55,-.26,-.47,-.50,.95,-.20])
cv.vote('aqua','bottom',  [.21,-.19,.08,-.21,-.22,-.20,-.03,-.32,.75,-.21])
cv.vote('light','top',    [.47,.12,-.10,-.26,-.18,-.20,-.36,.01,.43,.17])
cv.vote('light','bottom', [.04,.13,-.14,.11,-.25,.13,-.31,-.13,.11,.08])
cv.vote('white','center', [.14,.12,.04,.12,-.34,.19,-.45,-.07,.06,.39])
cv.vote('plant','top',    [-.77,-.25,.65,-.25,.65,-.12,.69,.25,-.72,-.27])
cv.vote('plant','bottom', [-.58,-.59,.12,-.18,.72,.15,.57,.92,-.83,-.51])
cv.vote('plant','center', [-.31,-.40,.31,-.24,.07,-.30,1.02,-.30,-.49,-.36])
cv.vote('olive','center', [-.29,-.24,.13,.02,.26,-.07,.58,.04,-.35,-.20])
cv.vote('tan','center',   [-.43,-.28,.07,.22,.58,.28,.32,.62,-.37,-.16])
cv.vote('brown','center', [-.19,-.13,.05,.17,.32,.22,.26,.32,-.25,-.20])
cv.vote('tan','edge',     [-.35,-.27,.00,.27,.27,.25,.39,.05,-.34,-.16])
cv.vote('neutral','all',  [.15,.38,-.13,.09,-.30,.03,-.39,-.23,.14,.24])
cv.vote('neutral','bottom',[.00,.42,-.12,-.06,-.25,-.08,-.32,-.31,.05,.34])
cv.vote('red','center',   [-.18,.66,.07,-.23,-.38,-.19,-.31,-.25,-.03,.55])
cv.vote('dark','center',  [-.04,.12,.14,.12,-.04,.12,-.11,.07,-.02,-.01])
cv.vote('dark','top',     [-.42,-.07,.13,.17,.14,.17,.25,.01,-.31,.01])
cv.vote('red','edge',     [-.19,.03,-.08,.28,-.28,.35,-.15,-.02,-.12,.05])
# Aggregate conjunctions: surface on which a different-colored object sits.
clauses=[
 ([('plant','bottom',1),('tan','center',.7)],.66, [-.20,-.28,.1,-.20,.96,.10,.14,1.25,-.36,-.18]),
 ([('plant','center',1),('olive','center',.45),('plant','side',.20)],.62,[-.16,-.18,.18,-.21,.05,-.15,.94,-.16,-.14,-.2]),
 ([('plant','top',.8),('plant','bottom',.8),('plant','center',-.9)],.38,[-.25,-.22,.55,-.12,.87,.13,-.25,.64,-.28,-.16]),
 ([('blue','bottom',.9),('blue','top',.8),('aqua','bottom',.5)],.79,[.96,-.20,.27,-.15,-.18,-.18,-.28,-.21,.58,-.23]),
 ([('blue','top',.85),('light','top',.46),('plant','bottom',.70)],.73,[.22,-.11,.13,-.27,.38,-.15,-.35,.90,-.13,.05]),
 ([('blue','bottom',.9),('aqua','bottom',.85),('light','top',.70)],.84,[.23,-.14,.04,-.17,-.3,-.15,-.26,-.26,1.42,-.11]),
 ([('white','center',.76),('red','center',.8),('neutral','bottom',1)],.93,[-.03,.54,-.18,-.14,-.21,-.16,-.24,-.20,.04,.77]),
 ([('tan','edge',.9),('neutral','edge',.25),('plant','edge',-.6),('light','top',-.3)],.39,[-.19,-.16,.02,.56,.01,.56,.13,-.19,-.28,-.15]),
 ([('neutral','top',.65),('neutral','bottom',.75),('light','top',.65),('dark','center',.42)],1.21,[.43,.18,-.14,-.19,-.19,-.17,-.32,-.16,.37,.26]),
 ([('blue','edge',.8),('blue','center',-.68),('pale','edge',.65)],.31,[.65,-.15,.25,-.14,-.15,-.15,-.17,-.19,.26,-.16]),
]
conditions=rel('scene_conditions',dense('scene_logic',colors,[field(cv.keys,rule) for rule,t,s in clauses],[-t for rule,t,s in clauses]))
# Center each vote across hypotheses: common-mode evidence does not affect argmax.
scene_score=summ('color_evidence',dense('region_votes',colors,cv.table-cv.table.mean(0)),dense('conditional_votes',conditions,np.array([s for rule,t,s in clauses]).T))

# Oriented contour and texture logic. Direction index 0 has a horizontal
# gradient and responds to a vertical boundary; index 2 is the converse.
# Normalize the axes jointly so a straight horizontal line is not confused
# with a high-energy patch of disordered fur. Keep a second, absolute path.
pos=conv('unsigned_edges',edges_pos,np.tile(np.eye(4),(1,2))); absolute=pool('local_roughness',pos,(2,2),(2,2)); peak=pool('prominent_edges',pos,(2,2),(2,2),m=True)
shared=shape('joint_axes',pos,(1,60,15)); scg=norm('joint_norm',conv('edge_floor',shared,[[EPS/.040]])); scg=shape('relative_edges',scg,(4,15,15)); scg=pool('relative_cells',scg,(3,3),(2,2))
# Rectify edge strength above a fixed visible contrast. Saturating second
# hinge means that one extreme discontinuity cannot dominate a whole region.
texture=rel('roughness_hinges',conv('roughness_thresholds',absolute,np.concatenate([np.eye(4)]*2)*10,[-.33]*4+[-1.68]*4));texture=conv('bounded_roughness',texture,np.concatenate([np.eye(4),-np.eye(4)],axis=1))
sy,sx=np.mgrid[:7,:7]/3-1
reg7={'all':np.ones((7,7)),'edge':(abs(sx)>.64)|(abs(sy)>.64),'side':abs(sx)>.64,'core':(abs(sx)<.67)&(abs(sy)<.55),'top':sy<-.45,'bottom':sy>.44,'belt':abs(sy)<.41,'end':(sy>.21)&(abs(sx)>.61),'below':sy>.1}
reg7={k:o/o.sum() for k,o in reg7.items()}; H=np.array([0.,0.,1.,0.]);T=np.array([1.,0.,0.,0.]); Di=np.array([0.,.5,0.,.5]);Avg=np.ones(4)/4

def area(reg,axis):return (axis[:,None,None]*reg7[reg]).ravel()
def texture_table(terms):
    return sum(np.outer(np.array(s),area(reg,axis)) for reg,axis,s in terms)
texterms=[
 ('top',Avg,[-.74,.10,.13,.01,.48,.01,.59,.20,-.52,.07]),
 ('bottom',Avg,[-.77,-.48,.18,.08,.48,.11,.66,.40,.02,-.37]),
 ('side',Avg,[-.46,-.09,.10,.01,.22,.03,.52,.12,-.31,-.14]),
 ('core',Avg,[.09,.02,.02,.16,-.12,.04,.20,-.12,-.08,-.02]),
]
relterms=[('all',H-Di,[.13,.40,-.17,-.27,-.12,-.19,-.12,-.15,.61,.45]),
 ('all',T-Di,[-.16,.13,-.04,-.07,.08,-.06,-.23,.05,-.08,.60]),
 ('core',H-T,[.21,.17,-.11,-.22,-.12,-.17,.06,-.09,.29,-.14]),
 ('core',Di-Avg,[.46,-.20,.32,.15,-.02,.09,.13,-.01,-.21,-.19]),
 ('below',H-Di,[-.06,.32,-.19,-.16,-.10,-.09,-.02,-.14,.53,.18]),
 ('below',T-Di,[-.09,-.08,-.01,-.03,.16,-.08,-.11,.38,-.20,.18]),
 ('belt',Avg,[.36,.21,.04,-.21,-.11,-.14,-.09,-.02,.28,.09]),
]
# Long, sharp lines. A hull meeting water, a windshield, a truck bed or
# roof usually has longer coherent contours than an irregular animal body.
row=pool('horizontal_coherence',pos,(1,6),(1,3)); line=pool('long_horizontal',row,(15,4),(1,1),m=True)
column=pool('vertical_coherence',pos,(5,1),(3,1)); post=pool('long_vertical',column,(4,15),(1,1),m=True)
linet=dense('long_line_votes',line,np.outer([.49,.95,-.49,-.50,-.41,-.56,-.38,-.27,1.52,1.14],3*H-1.5*Di-1.5*T))
postt=dense('upright_votes',post,np.outer([-.58,.14,-.13,-.08,.27,-.06,-.24,.31,-.10,.84],2.8*T-1.5*Di-1.3*H))
# A bus/truck body often has both a vertical side and a long horizontal
# top edge. ReLU(a) + ReLU(b) - ReLU(a-b) is not needed here: the exact
# minimum of the two normalized positive axis advantages is a-ReLU(a-b).
zrow=dense('horizontal_advantage',line,[(H-.68*Di-.32*T)*5]);zcol=dense('vertical_advantage',post,[(T-.68*Di-.32*H)*5]);diff=summ('axis_difference',zrow,dense('negative_upright',zcol,[[-1]]));box=summ('right_angles',zrow,dense('negative_excess',rel('dominant_horizontal',diff),[[-1]]));box=rel('boxlike',box)
boxscore=dense('rectilinearity_votes',box,np.array([-.26,.38,-.39,-.30,-.15,-.26,-.35,-.10,.01,.93])[:,None])
# Wheels: dark annulus below bodywork, with a hub and a lighter road
# outside. Pair nearby wheel candidates at several discrete wheelbases.
wy,wx=np.mgrid[-4:5,-4:5]; rr=np.hypot(wx,wy); ring=[]
for rad in [1.65,2.50]:
    def ga(r,s):return np.exp(-.5*((rr-r)/s)**2)
    def unit(a):return a/a.sum()
    ff=.75*unit(ga(rad+1.48,.68))+.25*unit(ga(0,.64))-unit(ga(rad,.67)); ring.append(ff)
whe=rel('dark_annuli',conv('wheel_hubs',gray,np.array(ring)[:,None]*7,[-.32,-.32],stride=2)); ws=pool('wheel_locations',whe,(2,2),(1,1),m=True)
# 32 -> 12 -> 11 sites. Tolerate oblique side views and perspective.
pairs=[]
for rowid in range(4,10):
    for dx in [3,5,7,9]:
        for left in range(0,11-dx):
            for dy in [-1,0,1]:
                for r in range(2):
                    v=np.zeros((2,11,11)); v[r,rowid,left]=.5; v[r,rowid+dy,left+dx]=.5; pairs.append(v.ravel())
pair=dense('wheelbase_tests',ws,pairs);pair=shape('wheelbase_list',pair,(1,1,len(pairs)));pair=pool('paired_wheels',pair,(1,len(pairs)),m=True)
pair=rel('convincing_wheels',dense('wheel_salience',pair,[[1]],[-.49])); wheelvote=dense('wheel_votes',pair,np.array([-.12,.62,-.09,-.07,-.07,-.06,-.04,-.06,-.08,.47])[:,None])
# Relative and absolute evidence have complementary invariances.
structure_score=summ('structure_evidence',dense('surface_votes',texture,texture_table(texterms)),dense('directional_votes',scg,texture_table(relterms)),linet,postt,boxscore,wheelvote)

node('scores','add',[object_score,scene_score,structure_score])
(D/'network.json').write_text(json.dumps({'layers':L},indent=2)+'\n'); np.savez(D/'weights.npz',**W)
