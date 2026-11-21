"""Build a hand-designed CIFAR-10 network.  No input files are read.

All object templates are individual analytic drawings.  Every template is
normalised using only itself.  All other constants are hand-chosen formulas.
"""
import sys, json
import numpy as np

OUT = sys.argv[1]
rng = np.random.default_rng(731942)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
T = 24
TR = 12

# ---------- tiny analytic drawing language ----------
class Canvas:
    def __init__(self, n, top, bottom):
        self.n = n
        q = np.linspace(-1, 1, n, dtype=np.float64)
        self.X, self.Y = np.meshgrid(q, q)
        t = ((self.Y + 1) / 2)[..., None]
        self.a = np.array(top)[None, None, :] * (1-t) + np.array(bottom)[None, None, :] * t
        self.a = np.broadcast_to(self.a, (n, n, 3)).copy()
        self.pose = (0., 0., 1., 0., 1.)
    def set_pose(self, cx=0., cy=0., scale=1., angle=0., flip=1.):
        self.pose = (cx, cy, scale, angle, flip)
    def pt(self, p):
        cx, cy, s, a, f = self.pose
        x, y = p[0]*f, p[1]
        co, si = np.cos(a), np.sin(a)
        return (cx+s*(co*x-si*y), cy+s*(si*x+co*y))
    def inv(self):
        cx, cy, s, a, f = self.pose
        x, y = self.X-cx, self.Y-cy
        co, si = np.cos(a), np.sin(a)
        return (f*(co*x+si*y)/s, (-si*x+co*y)/s)
    def paint(self, alpha, color):
        alpha = np.clip(alpha, 0, 1)[..., None]
        self.a = self.a*(1-alpha) + np.asarray(color)[None, None, :]*alpha
    def ellipse(self, c, r, color, angle=0.):
        x, y = self.inv()
        co, si = np.cos(angle), np.sin(angle)
        u = co*(x-c[0])+si*(y-c[1]); v = -si*(x-c[0])+co*(y-c[1])
        d = (1-np.sqrt((u/r[0])**2+(v/r[1])**2+1e-12))*min(r)*self.pose[2]
        self.paint(.5+d*self.n/2, color)
    def circle(self, c, r, color):
        self.ellipse(c, (r, r), color)
    def poly(self, points, color):
        p = np.array([self.pt(v) for v in points])
        x, y = self.X, self.Y
        inside = np.zeros_like(x, dtype=bool)
        dist = np.full_like(x, 10.)
        for i in range(len(p)):
            a, b = p[i], p[(i+1)%len(p)]
            vx, vy = b-a
            den = vx*vx+vy*vy+1e-12
            t = np.clip(((x-a[0])*vx+(y-a[1])*vy)/den, 0, 1)
            dist = np.minimum(dist, np.hypot(x-a[0]-t*vx, y-a[1]-t*vy))
            if abs(vy) > 1e-12:
                cross = ((a[1]>y)!=(b[1]>y)) & (x < (b[0]-a[0])*(y-a[1])/(vy+1e-30)+a[0])
                inside ^= cross
        d = np.where(inside, dist, -dist)
        self.paint(.5+d*self.n/2, color)
    def line(self, a, b, width, color):
        a, b = np.array(self.pt(a)), np.array(self.pt(b))
        vx, vy = b-a; den = vx*vx+vy*vy+1e-12
        t = np.clip(((self.X-a[0])*vx+(self.Y-a[1])*vy)/den, 0, 1)
        d = width*self.pose[2]-np.hypot(self.X-a[0]-t*vx, self.Y-a[1]-t*vy)
        self.paint(.5+d*self.n/2, color)
    def rect(self, x0, y0, x1, y1, color):
        self.poly([(x0,y0),(x1,y0),(x1,y1),(x0,y1)], color)

WHITE=(.91,.91,.87); BLACK=(.035,.04,.045); GRAY=(.42,.45,.47)
GLASS=(.20,.34,.43); BROWN=(.43,.25,.12); TAN=(.64,.45,.25)
GREEN=(.18,.40,.15); DARKGREEN=(.08,.22,.09); BLUE=(.15,.40,.72)


def background(c, n, j):
    # Broad, generic settings, not reproductions of any photograph.
    skies=[(.40,.65,.88),(.62,.76,.88),(.80,.80,.75),(.28,.48,.75)]
    if c==0:
        top=skies[j%4]; bot=(.52,.57,.55) if j%3==0 else (.55,.70,.83)
    elif c in (1,9):
        top=[(.48,.65,.76),(.52,.61,.50),(.65,.65,.61)][j%3]; bot=(.28,.29,.29)
    elif c==8:
        top=skies[j%4]; bot=[(.08,.32,.48),(.12,.42,.55),(.20,.48,.57)][j%3]
    elif c in (4,7):
        top=[(.38,.56,.38),(.60,.68,.51),(.40,.50,.34)][j%3]; bot=[(.28,.40,.15),(.46,.42,.25),(.24,.34,.13)][j%3]
    elif c==6:
        top=[(.23,.43,.18),(.18,.37,.22),(.33,.48,.20)][j%3]; bot=[(.16,.32,.12),(.12,.29,.20),(.26,.39,.12)][j%3]
    elif c in (2,):
        top=[skies[j%4],(.30,.47,.28),(.17,.37,.43)][j%3]; bot=[(.30,.43,.22),(.20,.35,.15),(.17,.31,.27)][j%3]
    else:
        top=[(.49,.45,.37),(.40,.53,.35),(.60,.57,.50),(.30,.39,.31)][j%4]
        bot=[(.31,.26,.20),(.26,.39,.18),(.43,.38,.31),(.22,.28,.21)][j%4]
    d=Canvas(n,top,bot)
    # A simple horizon, cloud, wake, or road is part of this individual scene.
    if c in (1,9):
        d.rect(-1,.45,1,1,(.27+.025*(j%3),.28+.025*(j%3),.28+.025*(j%3)))
        d.line((-1,.73),(1,.73),.012,(.65,.63,.52))
    elif c==8:
        for yy in (.53,.72,.87):
            d.line((-1,yy),(1,yy+.025),.009,(.35,.60,.65))
    elif c==0 and j%4==1:
        d.ellipse((-.45,-.55),(.38,.10),(.85,.86,.83))
        d.ellipse((.55,-.30),(.28,.08),(.88,.88,.85))
    return d


def airplane(d,j):
    body=[WHITE,(.72,.76,.78),(.86,.82,.69),(.60,.65,.70)][j%4]
    accent=[(.75,.08,.06),(.08,.25,.65),(.85,.55,.08),(.15,.45,.25)][j%4]
    v=j%3
    if v==0: # side view
        d.ellipse((0,0),(.78,.13),body)
        d.poly([(.60,-.08),(.91,-.015),(.60,.07)],body)
        d.poly([(-.15,-.05),(.18,-.62),(.37,-.59),(.18,-.025)],body)
        d.poly([(-.10,.035),(.25,.48),(.42,.44),(.19,.02)],body)
        d.poly([(-.66,-.06),(-.77,-.43),(-.57,-.40),(-.40,-.04)],accent)
        d.poly([(-.70,.015),(-.88,.24),(-.60,.17),(-.42,.03)],body)
        d.rect(-.70,-.025,.58,.015,accent)
        for x in (-.35,-.18,-.01,.16,.33): d.circle((x,-.055),.025,GLASS)
    elif v==1: # front view
        d.ellipse((0,.05),(.15,.68),body)
        d.poly([(-.10,-.15),(-.93,.22),(-.88,.36),(-.06,.12)],body)
        d.poly([(.10,-.15),(.93,.22),(.88,.36),(.06,.12)],body)
        d.poly([(-.07,.43),(-.45,.68),(-.42,.76),(0,.59)],body)
        d.poly([(.07,.43),(.45,.68),(.42,.76),(0,.59)],body)
        d.ellipse((0,-.36),(.12,.17),GLASS)
        d.rect(-.13,.30,.13,.39,accent)
        d.circle((0,-.65),.07,GRAY)
    else: # top / banking view
        d.ellipse((0,0),(.13,.79),body,angle=.12)
        d.poly([(-.08,-.12),(-.82,.43),(-.72,.56),(.02,.15)],body)
        d.poly([(.08,-.12),(.74,.23),(.72,.39),(-.02,.15)],body)
        d.poly([(-.06,.49),(-.42,.78),(-.33,.83),(.02,.62)],accent)
        d.poly([(.06,.49),(.38,.73),(.33,.81),(-.02,.62)],body)
        d.ellipse((0,-.48),(.10,.17),GLASS)


def car(d,j,truck=False):
    body=[(.72,.07,.045),(.08,.31,.65),(.82,.82,.78),(.12,.48,.23),(.78,.56,.08),(.25,.25,.28)][j%6]
    if truck:
        if j%3!=0:
            d.rect(-.88,-.48,.28,.30,[(.78,.78,.70),(.65,.68,.70),(.74,.69,.54)][j%3])
            d.poly([(.28,-.22),(.56,-.22),(.78,.04),(.82,.31),(.28,.31)],body)
            d.poly([(.39,-.15),(.55,-.15),(.68,.03),(.39,.03)],GLASS)
            for x in (-.61,-.35,.52): d.circle((x,.32),.17,BLACK); d.circle((x,.32),.075,GRAY)
            d.rect(-.84,.12,.25,.20,[accent for accent in [(.55,.12,.08),(.15,.35,.55),(.35,.43,.25)]][j%3])
        else:
            d.poly([(-.62,-.58),(.62,-.58),(.78,.42),(-.78,.42)],body)
            d.rect(-.49,-.43,.49,-.05,GLASS)
            d.rect(-.42,.08,.42,.28,GRAY)
            d.rect(-.62,.31,.62,.40,BLACK)
            for x in (-.59,.59): d.circle((x,.38),.17,BLACK)
            for x in (-.43,.43): d.rect(x-.10,.03,x+.10,.14,WHITE)
        return
    if j%4!=0: # side
        d.poly([(-.84,.05),(-.70,-.17),(-.30,-.24),(-.10,-.49),(.34,-.49),(.63,-.18),(.82,-.08),(.80,.25),(-.79,.25)],body)
        d.poly([(-.20,-.25),(-.06,-.42),(.13,-.42),(.13,-.23)],GLASS)
        d.poly([(.19,-.42),(.32,-.40),(.53,-.21),(.19,-.21)],GLASS)
        d.line((-.72,-.03),(.68,-.03),.018,(.75,.75,.68))
        for x in (-.52,.51): d.circle((x,.25),.18,BLACK); d.circle((x,.25),.085,GRAY)
        d.rect(.65,-.04,.79,.055,(.90,.85,.59))
    else: # front
        d.poly([(-.65,-.20),(-.40,-.53),(.40,-.53),(.65,-.20),(.78,.39),(-.78,.39)],body)
        d.poly([(-.35,-.44),(.35,-.44),(.50,-.18),(-.50,-.18)],GLASS)
        d.rect(-.56,.10,.56,.26,GRAY)
        for x in (-.48,.48): d.ellipse((x,-.015),(.15,.09),WHITE)
        for x in (-.61,.61): d.circle((x,.36),.16,BLACK)


def bird(d,j):
    feather=[(.40,.25,.12),(.20,.25,.20),(.66,.52,.28),(.16,.35,.48),(.72,.68,.52),(.32,.16,.10)][j%6]
    light=(min(feather[0]+.25,.9),min(feather[1]+.25,.9),min(feather[2]+.20,.9))
    if j%3!=0:
        d.ellipse((-.05,.04),(.47,.30),feather,angle=-.25)
        d.circle((.37,-.25),.22,feather)
        d.poly([(.52,-.30),(.88,-.20),(.53,-.12)],(.82,.55,.12))
        d.poly([(-.35,.02),(-.86,.31),(-.48,-.13)],feather)
        d.ellipse((-.12,-.015),(.35,.19),light,angle=-.35)
        d.circle((.43,-.30),.032,BLACK)
        d.line((.05,.26),(.02,.57),.025,(.45,.28,.12)); d.line((.25,.23),(.29,.55),.025,(.45,.28,.12))
        d.line((.02,.55),(-.10,.61),.018,(.45,.28,.12)); d.line((.29,.54),(.43,.58),.018,(.45,.28,.12))
    else:
        d.ellipse((0,.10),(.14,.47),feather)
        d.circle((0,-.35),.17,feather)
        d.poly([(-.06,-.18),(-.90,-.55),(-.62,.12),(-.12,.27)],feather)
        d.poly([(.06,-.18),(.90,-.55),(.62,.12),(.12,.27)],feather)
        d.poly([(-.09,.42),(-.28,.78),(0,.61),(.28,.78),(.09,.42)],light)
        d.poly([(-.06,-.40),(-.30,-.34),(-.06,-.29)],(.82,.55,.12))
        d.circle((-.07,-.40),.025,BLACK)


def cat(d,j):
    fur=[(.48,.31,.17),(.32,.32,.30),(.72,.62,.45),(.15,.15,.14),(.70,.70,.65)][j%5]
    light=(min(fur[0]+.22,.9),min(fur[1]+.22,.9),min(fur[2]+.20,.9))
    if j%3!=1: # frontal face and seated chest
        d.ellipse((0,.38),(.40,.52),fur)
        d.poly([(-.43,-.30),(-.38,-.82),(-.08,-.48)],fur)
        d.poly([(.43,-.30),(.38,-.82),(.08,-.48)],fur)
        d.poly([(-.34,-.43),(-.34,-.68),(-.17,-.47)],(.65,.39,.36))
        d.poly([(.34,-.43),(.34,-.68),(.17,-.47)],(.65,.39,.36))
        d.ellipse((0,-.18),(.46,.40),fur)
        d.ellipse((0,.02),(.25,.18),light)
        for x in (-.19,.19):
            d.ellipse((x,-.23),(.095,.065),(.55,.67,.25)); d.ellipse((x,-.23),(.018,.060),BLACK)
        d.poly([(-.055,-.07),(.055,-.07),(0,.015)],(.64,.35,.34))
        for yy in (-.02,.06):
            d.line((-.09,yy),(-.63,yy-.07),.009,light); d.line((.09,yy),(.63,yy-.07),.009,light)
        d.line((.27,.55),(.68,.32),.075,fur); d.line((.68,.32),(.59,.04),.075,fur)
    else: # standing side view
        d.ellipse((-.10,.05),(.56,.27),fur)
        d.circle((.43,-.20),.25,fur)
        d.poly([(.26,-.35),(.28,-.67),(.43,-.42)],fur); d.poly([(.46,-.42),(.61,-.65),(.64,-.31)],fur)
        d.poly([(-.48,.13),(-.86,-.18),(-.72,-.33),(-.30,.02)],fur)
        for x in (-.40,-.12,.25,.43): d.line((x,.18),(x-.04,.66),.055,fur)
        d.circle((.51,-.23),.028,BLACK)


def deer(d,j,horse=False):
    fur=[(.43,.25,.12),(.55,.35,.18),(.32,.22,.14),(.70,.56,.37),(.82,.80,.69)][j%5] if not horse else [(.42,.22,.10),(.62,.36,.16),(.25,.20,.17),(.78,.76,.67),(.48,.42,.34)][j%5]
    dark=tuple(v*.55 for v in fur)
    if j%4!=0:
        d.ellipse((-.13,.02),(.53,.29),fur)
        for x in (-.48,-.26,.18,.36):
            d.line((x,.17),(x+.03,.70),.040,fur); d.line((x+.03,.70),(x+.12,.76),.025,dark)
        d.line((.22,-.05),(.43,-.54),.16,fur)
        d.ellipse((.49,-.57),(.25,.13),fur,angle=.15)
        d.ellipse((.67,-.53),(.15,.075),dark)
        d.poly([(.36,-.64),(.24,-.87),(.44,-.70)],fur); d.poly([(.50,-.68),(.59,-.88),(.60,-.62)],fur)
        d.circle((.54,-.61),.025,BLACK)
        d.poly([(-.59,-.05),(-.82,-.24),(-.67,.06)],dark if horse else WHITE)
        if horse:
            d.line((.27,-.20),(.39,-.61),.065,dark)
            d.line((-.61,-.02),(-.84,.45),.065,dark)
        else:
            for sgn in (-1,1):
                x=.44+sgn*.07
                d.line((x,-.67),(x+sgn*.06,-.94),.022,dark)
                d.line((x+sgn*.04,-.82),(x+sgn*.18,-.98),.018,dark)
                d.line((x+sgn*.05,-.88),(x-sgn*.06,-1.02),.016,dark)
    else:
        d.ellipse((0,.43),(.34,.52),fur)
        d.ellipse((0,-.24),(.28,.47),fur)
        d.ellipse((0,.02),(.19,.24),dark)
        d.poly([(-.20,-.48),(-.60,-.66),(-.31,-.28)],fur); d.poly([(.20,-.48),(.60,-.66),(.31,-.28)],fur)
        for x in (-.12,.12): d.circle((x,-.30),.032,BLACK)
        if horse:
            d.line((-.18,-.52),(-.04,.25),.075,dark)
        else:
            for sgn in (-1,1):
                d.line((sgn*.15,-.52),(sgn*.25,-.94),.025,dark)
                d.line((sgn*.22,-.78),(sgn*.43,-.95),.020,dark)


def dog(d,j):
    fur=[(.57,.37,.18),(.77,.69,.50),(.28,.22,.17),(.73,.73,.67),(.45,.27,.12)][j%5]
    dark=tuple(v*.58 for v in fur)
    if j%3!=1:
        d.ellipse((-.12,.08),(.53,.30),fur)
        for x in (-.43,-.18,.20,.39): d.line((x,.23),(x,.67),.065,fur)
        d.circle((.40,-.20),.29,fur)
        d.ellipse((.61,-.08),(.25,.15),tuple(min(v+.12,.9) for v in fur))
        d.ellipse((.26,-.23),(.14,.31),dark,angle=-.25)
        d.circle((.76,-.10),.065,BLACK); d.circle((.48,-.27),.035,BLACK)
        d.line((-.57,-.01),(-.84,-.40),.075,fur)
    else:
        d.ellipse((0,.40),(.43,.52),fur)
        d.ellipse((0,-.19),(.44,.42),fur)
        d.ellipse((-.36,-.18),(.17,.38),dark,angle=.20); d.ellipse((.36,-.18),(.17,.38),dark,angle=-.20)
        d.ellipse((0,.01),(.28,.20),tuple(min(v+.13,.9) for v in fur))
        d.circle((0,-.055),.075,BLACK)
        for x in (-.17,.17): d.circle((x,-.26),.040,BLACK)


def frog(d,j):
    skin=[(.20,.48,.12),(.35,.55,.15),(.18,.38,.19),(.46,.48,.17)][j%4]
    light=(min(skin[0]+.18,.85),min(skin[1]+.20,.9),min(skin[2]+.12,.8))
    d.ellipse((0,.22),(.55,.45),skin)
    d.ellipse((-.48,.40),(.35,.18),skin,angle=-.35); d.ellipse((.48,.40),(.35,.18),skin,angle=.35)
    d.ellipse((0,-.16),(.48,.33),skin)
    for x in (-.27,.27):
        d.circle((x,-.40),.18,skin); d.circle((x,-.43),.115,WHITE); d.circle((x,-.44),.060,BLACK)
    d.ellipse((0,-.05),(.30,.16),light)
    d.line((-.28,-.06),(0,.05),.016,DARKGREEN); d.line((0,.05),(.28,-.06),.016,DARKGREEN)
    for x in (-.52,.52):
        d.line((x,.42),(x*1.35,.65),.065,skin)
        for dx in (-.10,0,.10): d.line((x*1.35,.65),(x*1.35+dx,.72),.022,skin)


def ship(d,j):
    hull=[(.65,.08,.05),(.12,.20,.29),(.75,.75,.68),(.16,.35,.42)][j%4]
    if j%3!=1:
        d.poly([(-.88,.10),(.88,.10),(.58,.48),(-.59,.48)],hull)
        d.rect(-.57,-.12,.48,.10,WHITE)
        d.rect(-.28,-.36,.28,-.12,WHITE)
        d.rect(-.20,-.29,.20,-.22,GLASS)
        d.line((0,-.36),(0,-.82),.025,GRAY)
        d.line((0,-.70),(.37,-.43),.018,GRAY)
        d.poly([(.02,-.77),(.30,-.65),(.02,-.59)],(.75,.12,.07))
        d.line((-.75,.55),(.72,.55),.025,(.65,.78,.79))
    else:
        d.poly([(-.63,-.05),(.63,-.05),(.35,.66),(-.35,.66)],hull)
        d.poly([(-.45,-.10),(-.28,-.52),(.28,-.52),(.45,-.10)],WHITE)
        d.rect(-.25,-.41,.25,-.28,GLASS)
        d.line((0,-.52),(0,-.85),.022,GRAY)
        d.ellipse((0,.64),(.72,.10),(.50,.70,.73))


def scene(c,n,j,foreground=False):
    d=Canvas(n,(.5,.5,.5),(.5,.5,.5)) if foreground else background(c,n,j)
    d.set_pose(float(rng.uniform(-.10,.10)),float(rng.uniform(-.06,.10)),float(rng.uniform(.82,1.04)),float(rng.uniform(-.10,.10)),-1 if (j//2)%2 else 1)
    if c==0: airplane(d,j)
    elif c==1: car(d,j,False)
    elif c==2: bird(d,j)
    elif c==3: cat(d,j)
    elif c==4: deer(d,j,False)
    elif c==5: dog(d,j)
    elif c==6: frog(d,j)
    elif c==7: deer(d,j,True)
    elif c==8: ship(d,j)
    else: car(d,j,True)
    return np.clip(d.a,0,1).astype(np.float64)


def blur(a,sigma):
    r=max(1,int(2.5*sigma)); q=np.arange(-r,r+1)
    g=np.exp(-q*q/(2*sigma*sigma)); g/=g.sum()
    p=np.pad(a,((r,r),(r,r)) if a.ndim==2 else ((r,r),(r,r),(0,0)),mode='reflect')
    z=np.zeros_like(a)
    for i,v in enumerate(g): z += v*p[i:i+a.shape[0],r:r+a.shape[1]]
    p=np.pad(z,((r,r),(r,r)) if a.ndim==2 else ((r,r),(r,r),(0,0)),mode='reflect')
    z=np.zeros_like(a)
    for i,v in enumerate(g): z += v*p[r:r+a.shape[0],i:i+a.shape[1]]
    return z


def unit_template(a,per_channel=False):
    # Only this individual drawing contributes to its normalization.
    a=np.array(a,dtype=np.float64,copy=True)
    if per_channel and a.ndim==3: a -= a.mean(axis=(0,1),keepdims=True)
    else: a -= a.mean()
    a /= np.sqrt(np.sum(a*a)+1e-10)
    return a.astype(np.float32)


def shape_template(c,k,j):
    a=scene(c,k,j,True)
    g=a@np.array([.299,.587,.114])
    mode=j%3
    if mode==0: g=g-blur(g,1.0)
    elif mode==1: g=blur(g,.65)
    else: g=blur(g,1.15)
    return unit_template(g)


def raw_template(c,k,j):
    return unit_template(blur(scene(c,k,j),.65),True)


def scene_template(c,j):
    a=scene(c,32,j)
    a=a.reshape(8,4,8,4,3).mean(axis=(1,3))
    # Cosine-like colour layout matching; retain the drawing's absolute colour.
    a=a/np.sqrt(np.sum(a*a)+1e-10)
    return a.astype(np.float32)

SOBEL=np.array([
 [[-1,0,1],[-2,0,2],[-1,0,1]],
 [[-1,-2,-1],[0,0,0],[1,2,1]],
 [[-2,-1,0],[-1,0,1],[0,1,2]],
 [[0,1,2],[-1,0,1],[-2,-1,0]]],np.float32)/4

def edge_template(c,j):
    # All operations here are on this one drawing alone.
    a=scene(c,32,j)@np.array([.299,.587,.114])
    a=(a-a.mean())/np.sqrt(np.mean((a-a.mean())**2)+1e-5)
    p=np.pad(a,((1,1),(1,1)))
    f=[]
    for w in SOBEL:
        z=sum(w[dy,dx]*p[dy:dy+32,dx:dx+32] for dy in range(3) for dx in range(3))
        f.append(np.abs(z))
    f=np.stack(f).reshape(4,8,4,8,4).mean(axis=(2,4))
    return unit_template(f)

layers=[]; weights={}
def conv(name,inp,W,b=None,stride=1,pad=0):
    W=np.asarray(W,dtype=np.float32)
    layers.append({'name':name,'op':'conv','inputs':[inp],'out':len(W),'kernel':[W.shape[-1],W.shape[-1]],'stride':stride,'padding':pad})
    weights[name+'.W']=W
    weights[name+'.b']=np.zeros(len(W),np.float32) if b is None else np.asarray(b,np.float32)
    return name

def relu(name,inp):
    layers.append({'name':name,'op':'relu','inputs':[inp]}); return name

def pool(name,op,inp,size,stride=None):
    layers.append({'name':name,'op':op,'inputs':[inp],'size':list(size),'stride':list(size if stride is None else stride)}); return name

def reshape(name,inp,shape):
    layers.append({'name':name,'op':'reshape','inputs':[inp],'shape':list(shape)}); return name

def dense(name,inp,W,b=None):
    W=np.asarray(W,dtype=np.float32)
    layers.append({'name':name,'op':'dense','inputs':[inp],'out':len(W)})
    weights[name+'.W']=W; weights[name+'.b']=np.zeros(len(W),np.float32) if b is None else np.asarray(b,np.float32)
    return name

def add(name,ins):
    layers.append({'name':name,'op':'add','inputs':list(ins)}); return name

def diag(name,inp,v):
    return dense(name,inp,np.eye(10,dtype=np.float32)*v)

conv('gray','image',np.array([[[[.299]],[[.587]],[[.114]]]],np.float32))
layers.append({'name':'gray_n','op':'norm','inputs':['gray']})
layers.append({'name':'color_n','op':'norm','inputs':['image']})

# Class scores from max/mean peak responses of independently drawn templates.
def template_branch(prefix,inp,templates,k,stride,count,gain):
    conv(prefix,inp,templates,stride=stride)
    side=(32-k)//stride+1
    if side>1:
        # Standardise each filter's response map within each image.  This
        # removes arbitrary differences in template/background energy.
        layers.append({'name':prefix+'_n','op':'norm','inputs':[prefix]})
        g=pool(prefix+'_peak','maxpool',prefix+'_n',(side,side))
        q=reshape(prefix+'_groups',g,(10,count,1))
    else:
        # With one position, standardise within the template family instead.
        q0=reshape(prefix+'_flat',prefix,(1,10,count))
        layers.append({'name':prefix+'_n','op':'norm','inputs':[prefix+'_flat']})
        q=reshape(prefix+'_groups',prefix+'_n',(10,count,1))
    mx=pool(prefix+'_max','maxpool',q,(count,1))
    av=pool(prefix+'_mean','avgpool',q,(count,1))
    mx=reshape(prefix+'_max10',mx,(10,)); av=reshape(prefix+'_mean10',av,(10,))
    av=diag(prefix+'_mean_gain',av,.35)
    z=add(prefix+'_evidence',[mx,av])
    return diag(prefix+'_gain',z,gain)

shape_scores=[]
for k,st in ((12,2),(20,2),(28,2),(32,1)):
    W=np.stack([shape_template(c,k,j) for c in range(10) for j in range(T)])[:,None,:,:]
    shape_scores.append(template_branch('shape%d'%k,'gray_n',W,k,st,T,1.0))
shape_sum=add('shape_sum',shape_scores)
shape_score=diag('shape_score',shape_sum,.10)

raw_scores=[]
for k,st in ((16,2),(32,1)):
    W=np.stack([raw_template(c,k,j) for c in range(10) for j in range(TR)]).transpose(0,3,1,2)
    raw_scores.append(template_branch('raw%d'%k,'image',W,k,st,TR,1.0))
raw_score_sum=add('raw_score_sum',raw_scores)
raw_score=diag('raw_score',raw_score_sum,.15)

small=pool('small','avgpool','image',(4,4))
SW=np.stack([scene_template(c,j).transpose(2,0,1).ravel() for c in range(10) for j in range(T)])
dense('scene',small,SW)
sg=reshape('scene_groups','scene',(10,T,1))
sm=pool('scene_max','maxpool',sg,(T,1))
sa=pool('scene_mean','avgpool',sg,(T,1))
sm=reshape('scene_max10',sm,(10,)); sa=reshape('scene_mean10',sa,(10,))
sa=diag('scene_mean_gain',sa,.35)
se=add('scene_evidence',[sm,sa])
# Standardise across classes, preserving their order within each image.
sz=reshape('scene_3d',se,(1,1,10))
layers.append({'name':'scene_n','op':'norm','inputs':['scene_3d']})
sz=reshape('scene_10','scene_n',(10,))
scene_score=diag('scene_score',sz,.25)

# Soft nearest-colour histograms.  For palette colour p, the logit is
# 2*p dot x - |p|^2.  Its maximum differs from negative squared distance
# only by |x|^2, which is shared by all palette entries.  Subtracting the
# maximum and applying a fixed margin gives a soft winner without squares.
palette=np.array([
 (.03,.03,.03),(.16,.17,.17),(.38,.40,.40),(.65,.66,.64),(.90,.90,.86),
 (.06,.14,.28),(.10,.32,.62),(.43,.67,.86),(.20,.62,.68),(.08,.32,.33),
 (.07,.20,.08),(.18,.40,.12),(.38,.55,.18),(.37,.38,.16),
 (.20,.11,.06),(.43,.25,.12),(.65,.45,.25),(.83,.74,.55),
 (.32,.07,.05),(.72,.10,.06),(.88,.45,.08),(.85,.75,.16),
 (.35,.18,.43),(.75,.40,.48),(.38,.48,.55),(.60,.55,.43)],np.float32)
P=len(palette); margin=.10
PW=2*palette[:,:,None,None]
pb=-np.sum(palette*palette,axis=1)
conv('palette_logits','image',PW,pb)
pf=reshape('palette_flat','palette_logits',(1,P,1024))
pm=pool('palette_max','maxpool',pf,(P,1))
pm=reshape('palette_maxmap',pm,(1,32,32))
conv('palette_negmax','palette_maxmap',-np.ones((P,1,1,1),np.float32),np.full(P,margin))
add('palette_margin',['palette_logits','palette_negmax'])
relu('palette_winner','palette_margin')
pool('palette_hist','avgpool','palette_winner',(4,4))

# Hand-written colour distributions, expressed as relative probabilities.
# Indices above: blacks/greys 0..4, blues 5..9, greens 10..13,
# browns 14..17, reds/yellows 18..21, purple/pink 22..23.
base_tables=[
 {7:5,4:3,3:3,2:2,6:2,24:2,1:1},
 {0:2,1:3,2:3,3:3,4:2,6:2,19:2,11:1,7:1,25:1},
 {14:2,15:3,16:3,17:1,11:2,10:1,7:1,4:1,1:1,2:1},
 {0:1,1:2,2:2,3:1,4:1,15:2,16:3,17:2,25:2,11:1},
 {15:4,16:4,14:2,17:1,11:3,12:2,10:1,25:1},
 {15:3,16:4,17:3,4:2,1:1,14:1,25:2,11:1,3:1},
 {10:2,11:5,12:4,13:2,15:2,16:1,14:1,4:.5,21:.5},
 {15:4,16:3,14:2,17:1,4:1,11:3,12:2,25:1,1:1},
 {6:3,7:4,8:3,9:3,4:3,3:2,2:2,1:1,19:1,24:2},
 {0:2,1:3,2:3,3:3,4:3,6:2,19:2,20:1,11:1,25:1}
]
region_tables=[
 {'top':{7:4,6:2},'bottom':{1:2,2:2,3:2},'center':{4:2,3:2}},
 {'bottom':{0:4,1:3,2:2},'center':{3:2,4:2,6:2,19:2}},
 {'center':{14:2,15:2,16:2},'top':{7:2,11:2}},
 {'center':{1:2,2:2,15:2,16:2,17:2}},
 {'center':{14:2,15:3,16:3},'bottom':{11:2,12:2}},
 {'center':{15:2,16:3,17:3,4:2},'bottom':{11:2}},
 {'center':{10:2,11:4,12:3,13:2}},
 {'center':{14:2,15:3,16:3},'bottom':{11:2,12:2}},
 {'top':{7:5,6:2},'bottom':{6:3,8:4,9:4},'center':{3:2,4:3,19:2}},
 {'bottom':{0:4,1:3,2:2},'center':{3:2,4:3,6:2,19:2}}
]
y,x=np.mgrid[0:8,0:8]; x=(x-3.5)/3.5; y=(y-3.5)/3.5
regions={'center':np.exp(-(x*x+y*y)/.45),
         'top':np.exp(-((y+.85)/.45)**2),
         'bottom':np.exp(-((y-.85)/.45)**2)}
CW=np.zeros((10,P,8,8),np.float32)
for c in range(10):
    base=np.full(P,.03,np.float64)
    for p,v in base_tables[c].items(): base[p]=v
    base/=base.sum()
    factors=np.ones((P,8,8),np.float64)
    for reg,items in region_tables[c].items():
        for p,v in items.items(): factors[p]*=1+(v-1)*regions[reg]
    prob=base[:,None,None]*factors
    prob=prob**3
    prob/=prob.sum(axis=0,keepdims=True)
    # Each hand-written layout is its own centred, unit-norm template.
    t=prob.ravel(); t-=t.mean(); t/=np.sqrt(np.sum(t*t)+1e-12)
    CW[c]=t.reshape(P,8,8)/margin
dense('colour_linear','palette_hist',CW.reshape(10,-1))
cz=reshape('colour_3d','colour_linear',(1,1,10))
layers.append({'name':'colour_n','op':'norm','inputs':['colour_3d']})
cz=reshape('colour_10','colour_n',(10,))
colour_score=diag('colour_score',cz,1.0)

# Coarse directed-edge layout, with individually generated scene templates.
EW=SOBEL[:,None,:,:].astype(np.float32)
conv('edges','gray_n',EW,stride=1,pad=1)
ep=relu('edges_pos','edges')
conv('edges_neg','gray_n',-EW,stride=1,pad=1)
en=relu('edges_neg_relu','edges_neg')
ea=add('edges_abs',[ep,en])
pool('edge_grid','avgpool','edges_abs',(4,4))
ET=np.stack([edge_template(c,j).ravel() for c in range(10) for j in range(T)])
dense('edge_templates','edge_grid',ET)
eg=reshape('edge_groups','edge_templates',(10,T,1))
em=pool('edge_max','maxpool',eg,(T,1))
ea=pool('edge_mean','avgpool',eg,(T,1))
em=reshape('edge_max10',em,(10,)); ea=reshape('edge_mean10',ea,(10,))
ea=diag('edge_mean_gain',ea,.35)
ee=add('edge_evidence',[em,ea])
ez=reshape('edge_3d',ee,(1,1,10))
layers.append({'name':'edge_n','op':'norm','inputs':['edge_3d']})
ez=reshape('edge_10','edge_n',(10,))
edge_score=diag('edge_score',ez,.5)

add('scores',['shape_score','raw_score','scene_score','colour_score','edge_score'])
# Rename by making the final layer the actual last layer (it already is).
with open(OUT+'/network.json','w') as f: json.dump({'layers':layers},f,separators=(',',':'))
np.savez(OUT+'/weights.npz',**weights)
print('wrote',len(layers),'layers and',sum(v.size for v in weights.values()),'weights')
