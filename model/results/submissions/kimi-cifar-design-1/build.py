"""Hand-designed CIFAR-10 network.  All weights are formulae or individual
programmatic drawings of generic objects; no data files are read.
Run: python3 build.py OUTPUT_DIRECTORY
"""
import sys, os, json, math
import numpy as np

# ------------------------------- small drawing language ---------------------
# Objects are drawn at twice the final resolution and area-averaged.  Every
# drawing is used on its own as a template (there is no cross-template fitting).
S = 2
H = W = 32 * S
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)


def ellipse(im, cx, cy, rx, ry, color, angle=0.0):
    ca, sa = math.cos(angle), math.sin(angle)
    dx, dy = xx-cx, yy-cy
    u, v = ca*dx+sa*dy, -sa*dx+ca*dy
    m = (u/max(rx, .01))**2 + (v/max(ry, .01))**2 <= 1
    im[m] = color


def polygon(im, points, color):
    # Even-odd scan conversion, evaluated on the small drawing grid.
    inside = np.zeros((H, W), dtype=bool)
    for a, b in zip(points, points[1:]+points[:1]):
        x1, y1 = a; x2, y2 = b
        if abs(y2-y1) < 1e-9:
            continue
        cross = ((y1 > yy) != (y2 > yy)) & (xx < (x2-x1)*(yy-y1)/(y2-y1)+x1)
        inside ^= cross
    im[inside] = color


def line(im, a, b, width, color):
    x1, y1 = a; x2, y2 = b
    dx, dy = x2-x1, y2-y1
    den = dx*dx+dy*dy
    t = np.clip(((xx-x1)*dx+(yy-y1)*dy)/max(den, 1e-9), 0, 1)
    m = (xx-(x1+t*dx))**2+(yy-(y1+t*dy))**2 <= (width*.5)**2
    im[m] = color


def background(kind, variant, seed):
    rng = np.random.default_rng(seed)
    y = yy/(H-1); x = xx/(W-1)
    im = np.empty((H, W, 3), dtype=np.float64)
    if kind in ('sky', 'airfield'):
        top = np.array([.28, .53, .80]); bot = np.array([.78, .84, .88])
        if variant: top = np.array([.48, .65, .78]); bot = np.array([.83, .83, .78])
        im[:] = top[None,None,:]*(1-y[:,:,None])+bot[None,None,:]*y[:,:,None]
        if kind == 'airfield':
            m = y > .78
            im[m] = np.array([.30+.12*variant, .39+.05*variant, .24])
            line(im, (0,.79*H), (W,.79*H), 1.2, (.72,.71,.62))
        else:
            # A broad cloud is just another part of this individual drawing.
            ellipse(im, (.18+.42*variant)*W, .22*H, .18*W, .055*H, (.82,.86,.88))
    elif kind == 'water':
        im[:] = (.56,.72,.82)
        m = y > .48
        im[m] = (.16+.05*variant, .39+.06*variant, .49+.08*variant)
        for q in (.58,.70,.84,.94):
            line(im, (0,q*H), (W,(q+.015)*H), 1.1, (.43,.61,.65))
    elif kind == 'road':
        im[:] = (.30+.10*variant, .38+.07*variant, .27)
        m = y > .64
        im[m] = (.32+.09*variant, .32+.08*variant, .31+.07*variant)
        line(im, (0,.67*H), (W,.67*H), 1.0, (.58,.57,.48))
    elif kind == 'indoor':
        im[:] = (.48+.17*variant, .40+.14*variant, .32+.13*variant)
        m = y > .72
        im[m] = (.32+.13*variant, .25+.11*variant, .19+.10*variant)
        line(im, (0,.72*H), (W,.72*H), 1.0, (.23,.20,.17))
    elif kind == 'forest':
        im[:] = (.18+.08*variant, .29+.09*variant, .15+.04*variant)
        for px in (.12,.48,.83):
            line(im, (px*W,0), ((px+.04)*W,H), 5.0, (.25+.06*variant,.20,.13))
    else:  # grass / close vegetation
        im[:] = (.29+.09*variant, .42+.08*variant, .19+.06*variant)
        im += (.035*np.sin(x*19+y*13)+.025*np.cos(x*31-y*17))[:,:,None]
    # Mild independent illumination and texture, not a measured image statistic.
    im *= (1.0 + .06*np.sin((x+.3*y)*math.pi))[:,:,None]
    im += rng.normal(0, .012, (1,1,1))
    return np.clip(im, 0, 1)


SCENE_KINDS = [('sky','airfield'), ('road','grass'), ('grass','sky'),
               ('indoor','grass'), ('forest','grass'), ('indoor','grass'),
               ('grass','forest'), ('grass','forest'), ('water','water'),
               ('road','indoor')]


def background_drawing(cls, view, variant):
    im = background(SCENE_KINDS[cls][variant], variant,
                    7919+cls*131+view*17+variant)
    return np.clip(im.reshape(32,S,32,S,3).mean((1,3)).transpose(2,0,1),0,1).astype(np.float32)


def object_drawing(cls, view, scale, shift, angle, variant, local=False):
    im = background(SCENE_KINDS[cls][variant], variant, 7919+cls*131+view*17+variant)
    # Work in generic object coordinates.  These dimensions are anatomical or
    # mechanical common knowledge, not measurements from any photograph.
    base = [34,31,25,30,34,31,30,37,36,35][cls] * scale
    if local: base *= .56
    cx = W*(.5 + shift); cy = H*[.51,.57,.53,.54,.54,.54,.55,.53,.54,.55][cls]
    ca, sa = math.cos(angle), math.sin(angle)
    def T(p):
        x,y=p; return (cx+base*(ca*x-sa*y), cy+base*(sa*x+ca*y))
    def P(ps,c): polygon(im,[T(p) for p in ps],c)
    def E(x,y,rx,ry,c,a=0):
        # Polygonal approximation keeps all geometry in the same rotated frame.
        ps=[]
        for t in np.linspace(0,2*math.pi,40,endpoint=False):
            u=rx*math.cos(t); v=ry*math.sin(t)
            ps.append(T((x+math.cos(a)*u-math.sin(a)*v,
                         y+math.sin(a)*u+math.cos(a)*v)))
        polygon(im,ps,c)
    def L(a,b,w,c): line(im,T(a),T(b),w*base,c)
    dark=(.055,.065,.070); glass=(.16,.29,.36); white=(.82,.84,.82)
    red=(.65+.12*variant,.10,.075); tan=(.53,.34+.08*variant,.17)
    brown=(.30+.12*variant,.20+.05*variant,.12); green=(.25+.12*variant,.48+.10*variant,.16)

    if cls == 0:  # airplane
        if view == 0:  # side airliner
            E(0,0,1.15,.20,white); P([(-1.12,0),(-1.35,-.30),(-1.12,-.32),(-.82,-.05)],white)
            P([(-.35,.02),(.25,.72),(.55,.68),(.18,.02)],(.47,.53,.57))
            P([(-.62,-.04),(-.12,-.63),(.13,-.59),(-.18,-.02)],(.66,.70,.72))
            P([(-1.05,-.10),(-1.27,-.67),(-1.05,-.64),(-.77,-.08)],red)
            for q in (-.62,-.35,-.08,.19,.46): E(q,-.075,.055,.035,glass)
            E(.99,.02,.12,.08,red)
        elif view == 1:  # seen from above
            E(0,0,1.22,.16,white); P([(-.28,0),(.15,-1.0),(.43,-.96),(.22,0),(.43,.96),(.15,1.0)],(.64,.68,.71))
            P([(-.88,0),(-1.05,-.43),(-.86,-.40),(-.64,0),(-1.05,.43),(-.86,.40)],white)
            E(.94,0,.21,.11,glass)
        elif view == 2:  # frontal
            E(0,0,.22,.93,white); P([(-.20,-.1),(-1.23,.30),(-1.18,.48),(0,.19),(1.18,.48),(1.23,.30),(.20,-.1)],(.62,.67,.71))
            P([(-.12,.55),(-.50,.89),(.50,.89),(.12,.55)],white)
            E(0,-.37,.16,.23,glass); E(-.72,.30,.16,.10,dark); E(.72,.30,.16,.10,dark)
        else:  # small propeller plane
            E(0,0,1.02,.24,(.77,.62,.19)); P([(-.35,0),(.05,-.78),(.35,-.73),(.23,0),(.05,.78),(-.15,.70)],red)
            P([(-.75,-.05),(-1.02,-.55),(-.80,-.50),(-.51,-.05)],red)
            E(.45,-.12,.29,.16,glass); L((1.03,-.48),(1.03,.48),.07,dark); E(1.03,0,.09,.09,dark)
    elif cls == 1:  # automobile
        paint=[red,(.12,.30,.62),(.73,.72,.62),(.18,.43,.25)][variant%4 if variant<4 else variant]
        if view in (0,3):
            low = view==3
            P([(-1.15,.18),(-1.03,-.18),(-.48,-.30),(-.20,-.66 if not low else -.43),(.48,-.65 if not low else -.42),(.85,-.25),(1.15,-.12),(1.12,.34),(-1.10,.34)],paint)
            P([(-.38,-.30),(-.14,-.57 if not low else -.38),(.15,-.57 if not low else -.38),(.15,-.29)],glass)
            P([(.22,-.57 if not low else -.38),(.45,-.55 if not low else -.36),(.72,-.27),(.22,-.28)],glass)
            for q in (-.72,.72):
                E(q,.34,.25,.25,dark); E(q,.34,.12,.12,(.43,.46,.46)); E(q,.34,.045,.045,(.72,.72,.67))
            E(1.05,.02,.09,.07,(.94,.88,.54)); L((-1.05,.20),(1.06,.20),.035,(.22,.23,.22))
        elif view == 1:  # front
            P([(-.85,.65),(-.98,-.05),(-.64,-.28),(-.43,-.82),(.43,-.82),(.64,-.28),(.98,-.05),(.85,.65)],paint)
            P([(-.39,-.68),(.39,-.68),(.57,-.27),(-.57,-.27)],glass)
            P([(-.54,-.12),(.54,-.12),(.65,.28),(-.65,.28)],paint)
            for q in (-.61,.61): E(q,.03,.20,.11,(.92,.89,.61)); E(q,.55,.20,.24,dark)
            P([(-.28,.25),(.28,.25),(.25,.43),(-.25,.43)],dark)
        else:  # rear
            P([(-.88,.62),(-.94,-.15),(-.48,-.75),(.48,-.75),(.94,-.15),(.88,.62)],paint)
            P([(-.43,-.63),(.43,-.63),(.60,-.23),(-.60,-.23)],glass)
            for q in (-.65,.65): E(q,.03,.17,.15,(.80,.04,.03)); E(q,.53,.20,.23,dark)
            P([(-.24,.25),(.24,.25),(.24,.37),(-.24,.37)],(.75,.75,.67))
    elif cls == 2:  # bird
        feather=[(.38,.25,.13),(.32,.39,.25),(.65,.48,.17),(.22,.27,.31)][view]
        if view in (0,2):
            E(-.12,.05,.64,.72,feather,-.18); E(.38,-.55,.35,.34,feather)
            P([(.65,-.58),(1.03,-.45),(.66,-.38)],(.77,.48,.10))
            E(.48,-.63,.055,.055,dark); E(-.25,.03,.40,.48,(.23+.1*variant,.19,.12),-.3)
            P([(-.57,.29),(-1.00,.68),(-.48,.53)],feather)
            for q in (-.12,.20): L((q,.57),(q,.98),.055,(.43,.27,.12)); L((q,.98),(q+.22,1.02),.035,(.43,.27,.12))
            if view==2: L((-1.0,1.04),(1.0,1.04),.09,brown)
        elif view == 1:
            E(0,.12,.23,.68,feather); E(0,-.52,.24,.24,feather)
            P([(-.12,-.15),(-.42,-.74),(-1.20,-1.00),(-.97,-.43),(-.55,.12),(-.16,.43)],feather)
            P([(.12,-.15),(.42,-.74),(1.20,-1.00),(.97,-.43),(.55,.12),(.16,.43)],feather)
            P([(0,-.49),(.40,-.40),(0,-.30)],(.72,.45,.10)); E(.08,-.57,.04,.04,dark)
        else:
            E(0,.20,.57,.77,feather); E(0,-.50,.43,.39,feather)
            E(-.30,.12,.25,.48,(.20,.18,.13)); E(.30,.12,.25,.48,(.20,.18,.13))
            for q in (-.16,.16): E(q,-.55,.055,.055,dark)
            P([(-.10,-.40),(.25,-.29),(-.10,-.20)],(.75,.46,.09))
            for q in (-.20,.20): L((q,.77),(q,1.04),.05,brown)
    elif cls == 3:  # cat
        fur=[(.47,.34,.21),(.38,.39,.38),(.67,.49,.29),(.20,.19,.17)][variant%4]
        if view in (0,2):
            if view==0: E(0,.38,.55,.73,fur)
            else: E(0,.10,.91,.83,fur)
            E(0,-.37,.62,.53,fur)
            P([(-.54,-.61),(-.48,-1.04),(-.16,-.72)],fur); P([(.54,-.61),(.48,-1.04),(.16,-.72)],fur)
            P([(-.43,-.68),(-.42,-.89),(-.25,-.71)],(.61,.38,.34)); P([(.43,-.68),(.42,-.89),(.25,-.71)],(.61,.38,.34))
            for q in (-.23,.23): E(q,-.39,.09,.07,(.53,.68,.24)); E(q,-.39,.025,.065,dark)
            E(0,-.20,.07,.055,(.60,.30,.29)); L((0,-.15),(0,-.06),.025,dark)
            if view==0:
                E(-.22,.79,.18,.18,fur); E(.22,.79,.18,.18,fur)
                L((.43,.60),(.90,.34),.16,fur); L((.90,.34),(.78,-.02),.13,fur)
        elif view == 1:
            E(-.12,.05,.82,.46,fur); E(.62,-.35,.42,.40,fur)
            P([(.37,-.58),(.42,-.97),(.64,-.65)],fur); P([(.73,-.64),(.94,-.94),(.94,-.47)],fur)
            for q in (-.65,-.25,.28,.58): L((q,.29),(q,.94),.15,fur)
            L((-.85,.02),(-1.17,-.46),.14,fur); L((-1.17,-.46),(-.99,-.74),.12,fur)
            E(.74,-.40,.05,.05,dark); E(.98,-.24,.06,.04,dark)
        else:
            E(0,.12,.88,.55,fur); E(.43,-.25,.48,.43,fur)
            P([(.12,-.46),(.17,-.84),(.40,-.57)],fur); P([(.57,-.58),(.84,-.83),(.88,-.38)],fur)
            L((-.68,.24),(-.96,-.13),.20,fur); L((-.96,-.13),(-.63,-.47),.17,fur)
            E(.52,-.29,.05,.05,dark)
    elif cls == 4:  # deer
        if view in (0,2,3):
            E(-.15,0,.78,.43,tan);
            for q,dx in ((-.65,-.12),(-.32,.06),(.25,-.04),(.49,.15)):
                L((q,.27),(q+dx,1.02),.085,tan); L((q+dx,1.02),(q+dx+.13,1.04),.055,dark)
            if view==2:
                L((.42,-.16),(.92,.48),.22,tan); E(.99,.50,.34,.19,tan,.35)
            else:
                L((.39,-.18),(.67,-.76),.23,tan); E(.77,-.85,.34,.21,tan,-.2)
                P([(.61,-.94),(.51,-1.26),(.76,-1.00)],tan); P([(.89,-.98),(1.06,-1.22),(1.01,-.90)],tan)
                for sgn in (-1,1):
                    L((.75+sgn*.09,-1.00),(.69+sgn*.18,-1.42),.045,brown)
                    L((.70+sgn*.15,-1.25),(.53+sgn*.29,-1.40),.035,brown)
                E(.91,-.88,.04,.04,dark)
            P([(-.86,-.13),(-1.14,-.36),(-1.00,.04)],white)
            if view==3:
                for q in (-.55,-.20,.15): E(q,-.10,.055,.055,(.82,.72,.48))
        else:
            E(0,.38,.59,.67,tan); L((0,.10),(0,-.55),.32,tan); E(0,-.69,.39,.42,tan)
            P([(-.26,-.82),(-.70,-1.04),(-.40,-.63)],tan); P([(.26,-.82),(.70,-1.04),(.40,-.63)],tan)
            for sgn in (-1,1):
                L((sgn*.17,-.96),(sgn*.32,-1.43),.05,brown); L((sgn*.27,-1.23),(sgn*.52,-1.41),.04,brown)
                E(sgn*.15,-.71,.045,.055,dark); L((sgn*.30,.63),(sgn*.38,1.07),.10,tan)
            E(0,-.47,.08,.055,dark)
    elif cls == 5:  # dog
        fur=[(.55,.35,.17),(.32,.24,.17),(.72,.65,.48),(.24,.25,.25)][variant%4]
        if view in (0,3):
            E(-.12,.06,.80,.45,fur); E(.61,-.34,.43,.40,fur); E(.91,-.20,.34,.22,fur)
            E(.45,-.43,.20,.38,brown,.25); E(.74,-.40,.045,.05,dark); E(1.17,-.20,.075,.065,dark)
            for q in (-.66,-.28,.27,.49): L((q,.30),(q,.94),.15,fur)
            L((-.84,-.02),(-1.14,-.48),.13,fur); L((-1.14,-.48),(-1.01,-.71),.10,fur)
            if view==3: P([(.86,-.03),(1.01,.13),(.91,.26)],(.65,.26,.26))
        elif view == 1:
            E(0,.38,.56,.70,fur); E(0,-.36,.62,.55,fur)
            E(-.52,-.35,.25,.48,brown,-.2); E(.52,-.35,.25,.48,brown,.2)
            E(0,-.12,.34,.27,(.64,.52,.37)); E(0,-.22,.12,.085,dark)
            for q in (-.22,.22): E(q,-.43,.055,.065,dark)
            E(-.24,.84,.19,.20,fur); E(.24,.84,.19,.20,fur)
        else:
            E(0,.02,.87,.78,fur); E(0,.22,.39,.31,(.65,.53,.38)); E(0,.10,.14,.10,dark)
            E(-.68,-.02,.27,.59,brown,-.15); E(.68,-.02,.27,.59,brown,.15)
            for q in (-.27,.27): E(q,-.10,.065,.075,dark); E(q-.015,-.125,.018,.022,white)
            P([(-.07,.30),(0,.48),(.07,.30)],(.60,.25,.27))
    elif cls == 6:  # frog
        if view in (0,2):
            E(0,.25,.84,.58,green); E(-.48,-.30,.34,.34,green); E(.48,-.30,.34,.34,green)
            E(-.48,-.35,.19,.19,(.76,.72,.35)); E(.48,-.35,.19,.19,(.76,.72,.35))
            E(-.48,-.35,.085,.12,dark); E(.48,-.35,.085,.12,dark)
            E(-.83,.57,.49,.25,green,-.3); E(.83,.57,.49,.25,green,.3)
            L((-.50,.18),(0,.31),.035,dark); L((0,.31),(.50,.18),.035,dark)
            for q in (-1,-.35,.35,1):
                L((q*.65,.63),(q,1.00),.075,green); L((q,1.0),(q+.18,1.02),.045,green)
            if view==2:
                for q,r in ((-.4,.2),(.1,.1),(.4,.35),(-.1,.48)): E(q,r,.08,.06,(.16,.29,.10))
        elif view == 1:
            E(-.12,.23,.89,.49,green,-.1); E(.51,-.12,.54,.39,green)
            E(.45,-.47,.25,.25,green); E(.45,-.49,.13,.14,(.75,.70,.33)); E(.45,-.50,.055,.09,dark)
            E(-.65,.60,.55,.23,green,.2); L((-.80,.57),(-1.07,.99),.13,green); L((-1.07,.99),(-.70,1.02),.08,green)
            L((.46,.33),(.77,.91),.13,green); L((.77,.91),(1.11,.93),.07,green)
            L((.49,.02),(.95,.10),.035,dark)
        else:
            E(0,.12,.70,.83,green); E(0,-.57,.64,.40,green)
            E(-.40,-.76,.25,.24,green); E(.40,-.76,.25,.24,green)
            for q in (-.40,.40): E(q,-.78,.13,.14,(.75,.72,.34)); E(q,-.78,.055,.09,dark)
            E(-.70,.53,.30,.49,green); E(.70,.53,.30,.49,green)
            L((-.39,-.42),(0,-.32),.035,dark); L((0,-.32),(.39,-.42),.035,dark)
    elif cls == 7:  # horse
        coat=[(.43,.23,.11),(.18,.15,.12),(.68,.55,.37),(.73,.72,.65)][variant%4]
        if view in (0,2):
            E(-.18,-.02,.84,.47,coat); L((.39,-.19),(.70,-.89),.29,coat)
            E(.82,-.99,.40,.22,coat,-.35); E(1.10,-.88,.24,.17,coat)
            P([(.61,-1.08),(.55,-1.40),(.76,-1.14)],coat); P([(.79,-1.13),(.88,-1.40),(.94,-1.06)],coat)
            L((.47,-.38),(.56,-1.04),.10,dark); E(.91,-1.03,.04,.045,dark)
            if view==0:
                legs=((-.72,-.82),(-.40,-.28),(.22,.12),(.45,.57))
            else:
                legs=((-.72,-1.04),(-.40,-.02),(.22,.78),(.45,.13))
            for q,e in legs:
                L((q,.27),(e,.73),.12,coat); L((e,.73),(e+(.12 if e>q else -.12),1.06),.085,coat)
                L((e+(.12 if e>q else -.12),1.06),(e+(.25 if e>q else -.02),1.07),.07,dark)
            L((-.94,-.15),(-1.23,.42),.16,dark); L((-1.23,.42),(-1.13,.86),.13,dark)
        elif view == 1:
            E(0,.43,.58,.67,coat); L((0,.12),(0,-.62),.35,coat); E(0,-.77,.36,.51,coat)
            P([(-.23,-1.06),(-.28,-1.40),(-.06,-1.13)],coat); P([(.23,-1.06),(.28,-1.40),(.06,-1.13)],coat)
            for q in (-.14,.14): E(q,-.80,.045,.055,dark)
            E(0,-.43,.19,.15,(.30,.24,.18));
            for q in (-.31,.31): L((q,.67),(q,1.10),.12,coat)
        else:
            E(0,-.05,.73,.89,coat); E(0,.53,.47,.47,coat)
            P([(-.57,-.48),(-.68,-1.08),(-.24,-.72)],coat); P([(.57,-.48),(.68,-1.08),(.24,-.72)],coat)
            E(-.25,-.12,.055,.065,dark); E(.25,-.12,.055,.065,dark); E(0,.60,.12,.08,dark)
            L((-.63,-.38),(-.82,.34),.16,dark)
    elif cls == 8:  # ship
        hull=[(.16,.20,.24),(.67,.10,.07),(.78,.79,.75),(.20,.31,.40)][view]
        if view in (0,3):
            P([(-1.20,.12),(1.20,.12),(.78,.65),(-.78,.65)],hull)
            P([(-.55,.08),(-.45,-.47),(.48,-.47),(.70,.08)],white)
            P([(-.25,-.48),(-.20,-.82),(.25,-.82),(.32,-.48)],white)
            for q in (-.32,-.08,.16,.40): P([(q,-.33),(q+.13,-.33),(q+.13,-.20),(q,-.20)],glass)
            L((0,-.81),(0,-1.33),.045,dark); L((0,-1.23),(.40,-1.05),.035,dark)
            if view==3:
                for q,c in ((-.60,red),(-.15,(.20,.40,.62)),(.30,(.65,.52,.16))): P([(q,-.02),(q+.38,-.02),(q+.38,-.40),(q,-.40)],c)
        elif view == 1:
            P([(-.92,.04),(.92,.04),(.40,.91),(-.40,.91)],hull)
            P([(-.52,.00),(-.40,-.60),(.40,-.60),(.52,.00)],white)
            P([(-.29,-.45),(.29,-.45),(.29,-.22),(-.29,-.22)],glass)
            L((0,-.60),(0,-1.30),.05,dark); E(0,-1.27,.08,.08,red)
        else:
            P([(-1.04,.35),(1.04,.35),(.66,.68),(-.72,.68)],hull)
            L((0,.35),(0,-1.29),.055,dark)
            P([(-.06,-1.18),(-.06,.24),(-.91,.24)],white); P([(.07,-.98),(.07,.24),(.74,.24)],(.80,.72,.51))
            L((-.96,.77),(.89,.77),.035,(.70,.76,.75))
    elif cls == 9:  # truck
        paint=[red,(.16,.34,.55),(.72,.70,.58),(.25,.43,.25)][variant%4]
        if view in (0,2,3):
            if view==0:
                P([(-1.23,-.64),(.24,-.64),(.24,.30),(-1.23,.30)],(.73,.75,.72))
                P([(.29,-.39),(.83,-.39),(1.13,-.03),(1.10,.34),(.29,.34)],paint)
                P([(.48,-.31),(.79,-.31),(.96,-.06),(.48,-.06)],glass)
            elif view==2:
                P([(-1.18,-.20),(.22,-.48),(.30,.27),(-1.06,.35)],paint)
                P([(.30,-.50),(.88,-.50),(1.10,-.08),(1.06,.34),(.30,.34)],paint)
                P([(.45,-.40),(.80,-.40),(.93,-.11),(.45,-.11)],glass)
            else:
                E(-.43,-.13,.79,.43,(.70,.72,.69)); P([(.30,-.48),(.85,-.48),(1.11,-.05),(1.08,.34),(.30,.34)],paint)
                P([(.45,-.38),(.79,-.38),(.94,-.10),(.45,-.10)],glass)
            for q in (-.88,-.43,.69):
                E(q,.38,.25,.25,dark); E(q,.38,.11,.11,(.48,.50,.48))
            L((-1.15,.16),(1.04,.16),.045,dark)
        else:
            P([(-.91,.72),(-.94,-.68),(-.66,-.94),(.66,-.94),(.94,-.68),(.91,.72)],paint)
            P([(-.69,-.69),(.69,-.69),(.73,-.20),(-.73,-.20)],glass)
            P([(-.60,.04),(.60,.04),(.53,.49),(-.53,.49)],dark)
            for q in (-.40,-.20,0,.20,.40): L((q,.08),(q,.45),.035,(.55,.57,.54))
            for q in (-.72,.72): E(q,-.01,.17,.12,(.91,.85,.53)); E(q,.61,.21,.24,dark)
    # Downsample this one drawing.  No other drawing contributes to it.
    return np.clip(im.reshape(32,S,32,S,3).mean((1,3)).transpose(2,0,1),0,1).astype(np.float32)


# ------------------------------- fixed front end ----------------------------
def norm(x):
    mu=x.mean(axis=(1,2),keepdims=True)
    return (x-mu)/np.sqrt(((x-mu)**2).mean(axis=(1,2),keepdims=True)+1e-5)


def pool(x, f):
    c,h,w=x.shape
    return x.reshape(c,h//f,f,w//f,f).mean((2,4))


def gaussian_derivatives():
    r=2
    y,x=np.mgrid[-r:r+1,-r:r+1].astype(np.float64)
    sig=1.15
    g=np.exp(-(x*x+y*y)/(2*sig*sig))
    gx=-x/(sig*sig)*g; gy=-y/(sig*sig)*g
    filters=[]
    for k in range(8):
        t=2*math.pi*k/8
        q=math.cos(t)*gx+math.sin(t)*gy
        q-=q.mean(); q/=np.sqrt((q*q).sum())
        filters.append(q)
    return np.asarray(filters,dtype=np.float32)


DERV=gaussian_derivatives()


def conv_one(x, kernels):
    # Only used on the program's own drawings, to express each as a template.
    p=kernels.shape[-1]//2
    xp=np.pad(x,((0,0),(p,p),(p,p)))
    out=[]
    for w in kernels:
        z=np.zeros(x.shape[1:],dtype=np.float64)
        for c in range(x.shape[0]):
            for dy in range(w.shape[1]):
                for dx in range(w.shape[2]):
                    z += w[c,dy,dx]*xp[c,dy:dy+x.shape[1],dx:dx+x.shape[2]]
        out.append(z)
    return np.asarray(out)


def make_drawing_features(im, f):
    n=norm(im.astype(np.float64))
    gray=n.mean(axis=0,keepdims=True)
    g=conv_one(gray,DERV[:,None,:,:].astype(np.float64))
    gp=np.maximum(norm(g),0)
    return np.concatenate([norm(pool(n,f)),norm(pool(gp,f))],axis=0).astype(np.float32)


# ------------------------------- network assembly ---------------------------
layers=[]; weights={}
def conv(name, inputs, out, k, stride=1, padding=0):
    layers.append({'name':name,'op':'conv','inputs':[inputs],'out':out,'kernel':[k,k],'stride':stride,'padding':padding})
def dense(name, inputs, out):
    layers.append({'name':name,'op':'dense','inputs':[inputs],'out':out})
def basic(name,op,inputs,**settings):
    d={'name':name,'op':op,'inputs':inputs}; d.update(settings); layers.append(d)

def classmax(name, source, count, spatial=1):
    # Channels are contiguous in class order.  Pooling a reshaped channel axis
    # gives exactly one maximum response for each class.
    basic(name+'_reshape','reshape',[source],shape=[1,count,spatial])
    basic(name+'_pool','maxpool',[name+'_reshape'],size=[count//10,spatial],stride=[count//10,spatial])
    basic(name,'reshape',[name+'_pool'],shape=[10])

basic('input_norm','norm',['image'])
conv('oriented','input_norm',8,5,1,2)
gray=np.array([1/3,1/3,1/3],dtype=np.float32)
weights['oriented.W']=(DERV[:,None,:,:]*gray[None,:,None,None]).astype(np.float32)
weights['oriented.b']=np.zeros(8,np.float32)
basic('oriented_norm','norm',['oriented'])
basic('oriented_relu','relu',['oriented_norm'])

# Generate exactly the same generic views for every class.  Backgrounds, poses,
# and scales are alternatives, never averaged together.
per_class=96
drawings=[]
for c in range(10):
    one=[]
    for view in range(4):
        for scale in (.78,1.0,1.22):
            for shift in (-.11,.11):
                for angle in (-.18,.18):
                    for variant in (0,1):
                        one.append(object_drawing(c,view,scale,shift,angle,variant))
    drawings.append(one)

for f in (2,4):
    side=32//f; feat=11*side*side; total=10*per_class
    basic(f'color_pool{f}','avgpool',['input_norm'],size=[f,f],stride=[f,f])
    basic(f'color_pnorm{f}','norm',[f'color_pool{f}'])
    basic(f'edge_pool{f}','avgpool',['oriented_relu'],size=[f,f],stride=[f,f])
    basic(f'edge_pnorm{f}','norm',[f'edge_pool{f}'])
    # Zero-padded 1x1 convolutions concatenate the two representations.
    conv(f'color_pad{f}',f'color_pnorm{f}',11,1)
    wc=np.zeros((11,3,1,1),np.float32)
    for c in range(3): wc[c,c,0,0]=math.sqrt(2/3)
    weights[f'color_pad{f}.W']=wc; weights[f'color_pad{f}.b']=np.zeros(11,np.float32)
    conv(f'edge_pad{f}',f'edge_pnorm{f}',11,1)
    we=np.zeros((11,8,1,1),np.float32); we[3:,:,0,0]=np.eye(8,dtype=np.float32)
    weights[f'edge_pad{f}.W']=we; weights[f'edge_pad{f}.b']=np.zeros(11,np.float32)
    basic(f'features{f}','add',[f'color_pad{f}',f'edge_pad{f}'])
    dense(f'templates{f}',f'features{f}',total)
    wt=[]
    for c in range(10):
        for im in drawings[c]:
            v=make_drawing_features(im,f).reshape(-1).astype(np.float64)
            # Colour was given the same total energy budget as the eight edge maps.
            v[:3*side*side]*=math.sqrt(2/3)
            v-=v.mean(); v/=max(np.linalg.norm(v),1e-12)
            wt.append(v)
    weights[f'templates{f}.W']=np.asarray(wt,dtype=np.float32)
    weights[f'templates{f}.b']=np.zeros(total,np.float32)
    basic(f'template_relu{f}','relu',[f'templates{f}'])
    classmax(f'class{f}',f'template_relu{f}',total)
    dense(f'gain{f}',f'class{f}',10)
    gain=.40 if f==2 else .35
    weights[f'gain{f}.W']=np.eye(10,dtype=np.float32)*gain
    weights[f'gain{f}.b']=np.zeros(10,np.float32)
    # A matched background residual prevents a patch of sky, water or grass
    # from being sufficient evidence for an object.  Every background below is
    # likewise one independently normalised programmatic drawing.
    dense(f'backgrounds{f}',f'features{f}',80)
    wb=[]
    for c in range(10):
        for view in range(4):
            for variant in (0,1):
                v=make_drawing_features(background_drawing(c,view,variant),f).reshape(-1).astype(np.float64)
                v[:3*side*side]*=math.sqrt(2/3)
                v-=v.mean(); v/=max(np.linalg.norm(v),1e-12)
                wb.append(v)
    weights[f'backgrounds{f}.W']=np.asarray(wb,dtype=np.float32)
    weights[f'backgrounds{f}.b']=np.zeros(80,np.float32)
    basic(f'background_relu{f}','relu',[f'backgrounds{f}'])
    classmax(f'background_class{f}',f'background_relu{f}',80)
    dense(f'background_gain{f}',f'background_class{f}',10)
    weights[f'background_gain{f}.W']=np.eye(10,dtype=np.float32)*(-.25 if f==2 else -.22)
    weights[f'background_gain{f}.b']=np.zeros(10,np.float32)

# Local, translation-tolerant object templates on the 16x16 representation.
local_per=12; local_total=120
conv('local_match','features2',local_total,8,1,0)
wl=[]
for c in range(10):
    for view in range(4):
        for scale in (.78,1.0,1.22):
            im=object_drawing(c,view,scale,0,0,view%2,local=True)
            v=make_drawing_features(im,2)[:,4:12,4:12].reshape(-1).astype(np.float64)
            v[:3*64]*=math.sqrt(2/3)
            v-=v.mean(); v/=max(np.linalg.norm(v),1e-12)
            wl.append(v)
weights['local_match.W']=np.asarray(wl,dtype=np.float32).reshape(local_total,11,8,8)
weights['local_match.b']=np.zeros(local_total,np.float32)
basic('local_relu','relu',['local_match'])
classmax('local_class','local_relu',local_total,81)
dense('local_gain','local_class',10)
weights['local_gain.W']=np.eye(10,dtype=np.float32)*.25
weights['local_gain.b']=np.zeros(10,np.float32)

# Coarse photometric scene rules.  These use unnormalised input colour, unlike
# the shape templates.  Each spatial mask and chromatic direction is a formula.
basic('raw_pool4','avgpool',['image'],size=[4,4],stride=[4,4])
dense('colour_rules','raw_pool4',10)
gy,gx=np.mgrid[-1:1:8j,-1:1:8j]
def unit_mask(a):
    a=np.asarray(a,dtype=np.float64); return a/max(np.linalg.norm(a),1e-12)
center=unit_mask(np.exp(-(gx*gx+gy*gy)/.28))
top=unit_mask(np.exp(-((gy+.58)/.40)**2-.18*gx*gx))
bottom=unit_mask(np.exp(-((gy-.62)/.38)**2-.18*gx*gx))
surround=unit_mask(1-.82*np.exp(-(gx*gx+gy*gy)/.25))
blue=np.array([-.50,-.25,.75]); green=np.array([-.50,.75,-.25])
brown=np.array([.55,.15,-.70]); red=np.array([.80,-.40,-.40])
bright=np.ones(3)/math.sqrt(3); dark=-bright
specs=[
    [(1.15,blue,top),(.65,blue,surround),(.35,bright,center),(-.30,green,bottom)],
    [(.55,red,center),(.55,bright,bottom),(-.35,blue,surround)],
    [(.55,brown,center),(.50,green,surround),(.20,blue,top)],
    [(.50,brown,center),(.20,dark,center),(-.25,blue,surround)],
    [(.65,brown,center),(.75,green,surround),(-.25,blue,top)],
    [(.45,brown,center),(.35,green,surround),(-.20,blue,surround)],
    [(.85,green,center),(.55,green,surround),(-.35,blue,top)],
    [(.60,brown,center),(.70,green,surround),(-.25,blue,top)],
    [(1.10,blue,bottom),(.50,blue,top),(.30,bright,center),(-.35,green,surround)],
    [(.45,red,center),(.55,bright,center),(.40,bright,bottom),(-.30,green,surround)]
]
wc=[]
for spec in specs:
    a=np.zeros((3,8,8),dtype=np.float64)
    for gain,colour,mask in spec: a+=gain*colour[:,None,None]*mask[None,:,:]
    a-=a.mean(); a/=max(np.linalg.norm(a),1e-12)
    wc.append(a.reshape(-1))
weights['colour_rules.W']=np.asarray(wc,dtype=np.float32)*2.0
weights['colour_rules.b']=np.zeros(10,np.float32)

# Explicit oriented-edge layout rules: horizontal hulls and fuselages,
# rectangular vehicle bodies, and vertical mammal legs/necks.
dense('edge_rules','edge_pool4',10)
horizontal=np.array([0,0,1,0,0,0,1,0.],dtype=np.float64)
vertical=np.array([1,0,0,0,1,0,0,0.],dtype=np.float64)
diagonal=np.array([0,1,0,1,0,1,0,1.],dtype=np.float64)
mid=unit_mask(np.exp(-((gy+.02)/.48)**2-.25*gx*gx))
lower=unit_mask(np.exp(-((gy-.48)/.40)**2-.25*gx*gx))
upper=unit_mask(np.exp(-((gy+.48)/.40)**2-.25*gx*gx))
edge_specs=[
    [(1.0,horizontal,mid),(.65,diagonal,center)],
    [(.9,horizontal,lower),(.65,vertical,center),(.25,diagonal,upper)],
    [(.6,vertical,center),(.6,diagonal,center),(.25,vertical,lower)],
    [(.55,vertical,center),(.6,diagonal,upper),(.25,horizontal,center)],
    [(.8,vertical,lower),(.65,horizontal,mid),(.45,diagonal,upper)],
    [(.6,vertical,lower),(.65,horizontal,mid),(.35,diagonal,upper)],
    [(.8,horizontal,center),(.8,diagonal,lower),(.25,vertical,center)],
    [(1.0,vertical,lower),(.7,horizontal,mid),(.6,vertical,upper)],
    [(1.2,horizontal,mid),(.65,horizontal,lower),(.55,vertical,upper)],
    [(1.0,horizontal,lower),(1.0,vertical,center),(.25,horizontal,upper)]
]
we=[]
for spec in edge_specs:
    a=np.zeros((8,8,8),dtype=np.float64)
    for gain,orient,mask in spec: a+=gain*orient[:,None,None]*mask[None,:,:]
    a-=a.mean(); a/=max(np.linalg.norm(a),1e-12)
    we.append(a.reshape(-1))
weights['edge_rules.W']=np.asarray(we,dtype=np.float32)*.55
weights['edge_rules.b']=np.zeros(10,np.float32)

# Nonlinear colour-tail summaries.  ReLU threshold functions describe hues and
# light/dark distributions without storing any observed colour histogram.
projections=[np.array([1.,0,0]),np.array([0,1.,0]),np.array([0,0,1.]),
             np.array([1/3,1/3,1/3]),np.array([-.5,-.5,1.]),
             np.array([-.5,1.,-.5]),np.array([1.,-.5,-.5])]
thresholds=[.20,.40,.60,.20,.40,.60,.20,.40,.60,.20,.40,.60,
            -.20,0,.20,-.20,0,.20,-.20,0,.20]
wb=[]; bb=[]
for p,ts in zip(projections,[thresholds[i:i+3] for i in range(0,21,3)]):
    for t in ts:
        for sign in (1.,-1.):
            wb.append((sign*p)[:,None,None]); bb.append(-sign*t)
conv('colour_basis','image',42,1)
weights['colour_basis.W']=np.asarray(wb,dtype=np.float32)
weights['colour_basis.b']=np.asarray(bb,dtype=np.float32)
basic('colour_basis_relu','relu',['colour_basis'])
basic('colour_basis_pool','avgpool',['colour_basis_relu'],size=[32,32],stride=[32,32])
dense('colour_histogram_rules','colour_basis_pool',10)
affinities=np.array([
    [.2,.3,.6,.4,.8,-.2,-.3], [.3,.1,.2,0,0,-.2,.4],
    [.1,.3,-.2,-.2,-.3,.5,0], [.2,.1,-.2,0,-.3,0,.1],
    [.3,.3,-.3,-.1,-.4,.5,.1], [.3,.2,-.1,.1,-.3,.2,.1],
    [0,.6,-.3,-.2,-.4,.8,-.2], [.3,.3,-.3,-.1,-.4,.5,.1],
    [.2,.4,.7,.4,1.,0,-.3], [.4,.2,.2,.1,0,-.2,.4]
],dtype=np.float64)
wh=np.repeat(affinities,6,axis=1).reshape(10,42)
# Alternate upper/lower tails have opposite meanings for the same hue.
wh*=np.tile(np.repeat([1.,-1.],3),7)[None,:]
wh-=wh.mean(axis=0,keepdims=True)
wh/=np.linalg.norm(wh,axis=1,keepdims=True)
weights['colour_histogram_rules.W']=wh.astype(np.float32)*5.0
weights['colour_histogram_rules.b']=np.zeros(10,np.float32)

basic('evidence','add',['gain2','gain4','local_gain','background_gain2','background_gain4',
                       'colour_rules','edge_rules','colour_histogram_rules'])
dense('scores','evidence',10)
weights['scores.W']=np.eye(10,dtype=np.float32)
# Aggregate confusion-table corrections; no example-level outputs are used.
weights['scores.b']=np.array([.55,.90,-1.05,.95,.25,-.20,.85,-.50,-.10,.05],np.float32)

out=sys.argv[1]
os.makedirs(out,exist_ok=True)
with open(os.path.join(out,'network.json'),'w') as fp:
    json.dump({'layers':layers},fp,separators=(',',':'))
np.savez(os.path.join(out,'weights.npz'),**weights)
