"""Data-free, hand-designed CIFAR-sized classifier.

All geometry, colours, coefficients and sampling schedules below are a priori.
No file is opened for reading.  Each drawing is used as ONE independently
normalised template; there are no statistics across drawings, fitting, or
synthetic classification experiments.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

F32 = np.float32
EPS = 1e-5
LAYERS = []
WEIGHTS = {}


def layer(name, op, source, **settings):
    ins = [source] if isinstance(source, str) else source
    LAYERS.append(dict(name=name, op=op, inputs=ins, **settings))
    return name


def conv(name, source, w, b=None, padding=0, stride=1):
    w = np.asarray(w, dtype=F32)
    if b is None:
        b = np.zeros(len(w), F32)
    WEIGHTS[name + '.W'] = w
    WEIGHTS[name + '.b'] = np.asarray(b, dtype=F32)
    return layer(name, 'conv', source, out=len(w), kernel=[w.shape[2]]*2,
                 stride=stride, padding=padding)


def dense(name, source, w, b=None):
    w = np.asarray(w, dtype=F32)
    if b is None:
        b = np.zeros(len(w), F32)
    WEIGHTS[name + '.W'] = w
    WEIGHTS[name + '.b'] = np.asarray(b, dtype=F32)
    return layer(name, 'dense', source, out=len(w))


def pool(name, source, size, stride, op='avgpool'):
    return layer(name, op, source, size=list(size), stride=list(stride))


def global_normalise(name, source, channels, h, w):
    z = layer(name + '_flat', 'reshape', source, shape=[1, channels*h, w])
    z = layer(name + '_unit', 'norm', z)
    return layer(name, 'reshape', z, shape=[channels, h, w])


# Analytic Gaussian derivatives.  Unit L1 norm is a definition of the filters,
# not a scale estimated from pictures.
yy, xx = np.mgrid[-2:3, -2:3].astype(F32)
g = np.exp(-(xx*xx + yy*yy)/2).astype(F32)
derivatives = []
for a in (0, math.pi/4, math.pi/2, 3*math.pi/4):
    d = (math.cos(a)*xx + math.sin(a)*yy)*g
    derivatives.append(d / np.abs(d).sum())
DERIV = np.asarray(derivatives, F32)
# Difference of two unit-mass Gaussians: dark/bright small circular details.
ga = np.exp(-(xx*xx+yy*yy)/(2*.65**2))
gb = np.exp(-(xx*xx+yy*yy)/(2*1.25**2))
dog = ga/ga.sum() - gb/gb.sum()
dog /= np.abs(dog).sum()
DOG = np.asarray(dog, F32)


def make_front_end():
    conv('luminance', 'image', np.array([.299, .587, .114], F32).reshape(1,3,1,1))
    layer('luminance_norm', 'norm', 'luminance')
    kernels = np.concatenate([DERIV, -DERIV, DOG[None], -DOG[None]], axis=0)
    conv('derivatives', 'luminance_norm', kernels[:,None])
    layer('derivative_positive', 'relu', 'derivatives')
    w = np.zeros((6,10,1,1), F32)
    for i in range(4):
        w[i,i,0,0] = 1
        w[i,i+4,0,0] = 1
        # Cross-orientation inhibition makes the angular response selective,
        # instead of giving a diagonal edge almost equal votes in every bin.
        j = (i+2)%4
        w[i,j,0,0] = -.75
        w[i,j+4,0,0] = -.75
    w[4,8,0,0] = 1
    w[5,9,0,0] = 1
    conv('edge_and_spot', 'derivative_positive', w)
    layer('oriented_structure', 'relu', 'edge_and_spot')
    pool('edge_cells', 'oriented_structure', (2,2), (2,2))
    # Select HOG and signed centre-surround channels separately.
    wh = np.zeros((4,6,1,1), F32)
    wb = np.zeros((2,6,1,1), F32)
    for i in range(4): wh[i,i,0,0] = 1
    wb[0,4,0,0] = wb[1,5,0,0] = 1
    conv('hog_fine', 'edge_cells', wh)
    conv('spots', 'edge_cells', wb)
    pool('hog_coarse', 'hog_fine', (2,2), (2,2))
    global_normalise('fine_unit', 'hog_fine', 4,14,14)
    # Allow a contour to move by one cell without moving the whole object.
    pool('deformable_edges','hog_fine',(3,3),(1,1),op='maxpool')
    global_normalise('deformable_unit','deformable_edges',4,12,12)
    global_normalise('coarse_unit', 'hog_coarse', 4,7,7)
    global_normalise('spot_unit', 'spots', 2,14,14)
    pool('luminance_half', 'luminance_norm', (2,2), (2,2))
    ident = np.zeros((1,1,3,3), F32)
    ident[0,0,1,1] = 1
    conv('interior_luminance', 'luminance_half', ident)
    layer('appearance_unit', 'norm', 'interior_luminance')


# A tiny analytic renderer.  It uses unions of ellipses, polygons and capsules.
# Coordinates describe generic objects, not any observed photograph.
# Oversampling is local to each drawing and only antialiases that drawing.
S = 64
BASE_Y, BASE_X = np.mgrid[:S,:S].astype(F32)
BASE_X = (BASE_X + .5)/S*2 - 1
BASE_Y = (BASE_Y + .5)/S*2 - 1


class Drawing:
    def __init__(self, scale, tx, ty, angle, mirror, morph, size=32):
        self.size = size
        sy,sx = np.mgrid[:2*size,:2*size].astype(F32)
        sx,sy = (sx+.5)/size-1,(sy+.5)/size-1
        x = (sx-tx)/scale
        y = (sy-ty)/scale
        ca, sa = math.cos(angle), math.sin(angle)
        self.x = (ca*x+sa*y)*mirror/(1 + .09*morph)
        self.y = (-sa*x+ca*y)/(1 - .07*morph)
        self.a = np.full((2*size,2*size), .78, F32)

    def ellipse(self, cx, cy, rx, ry, value=.34, angle=0):
        x, y = self.x-cx, self.y-cy
        ca, sa = math.cos(angle), math.sin(angle)
        u, v = ca*x+sa*y, -sa*x+ca*y
        mask = (u/rx)**2+(v/ry)**2 <= 1
        self.a[mask] = value

    def poly(self, pts, value=.34):
        inside = np.zeros_like(self.a, dtype=bool)
        x, y = self.x, self.y
        x0, y0 = pts[-1]
        for x1, y1 in pts:
            if y1 != y0:
                cross = ((y1 > y) != (y0 > y)) & (x < (x0-x1)*(y-y1)/(y0-y1)+x1)
                inside ^= cross
            x0, y0 = x1, y1
        self.a[inside] = value

    def line(self, x0, y0, x1, y1, width, value=.34):
        dx, dy = x1-x0, y1-y0
        t = np.clip(((self.x-x0)*dx+(self.y-y0)*dy)/(dx*dx+dy*dy+1e-12), 0,1)
        inside = (self.x-x0-t*dx)**2+(self.y-y0-t*dy)**2 <= (width/2)**2
        self.a[inside] = value

    def wheel(self, x, y, r):
        self.ellipse(x,y,r,r,.12)
        self.ellipse(x,y,r*.43,r*.43,.57)

    def eye(self, x, y, r=.045):
        self.ellipse(x,y,r*1.2,r,.09)

    def image(self):
        return self.a.reshape(self.size,2,self.size,2).mean((1,3))


def plane(d, view, m):
    if view == 0: # Side: fuselage, wings, tail and cockpit.
        d.poly([(-.96,.10),(-.82,-.02),(.62,-.08),(.96,.04),(.72,.18),(-.82,.22)], .36)
        d.poly([(-.72,.10),(-.88,-.40),(-.62,-.38),(-.32,.10)], .34)
        d.poly([(-.21,.10),(.08,-.47),(.31,-.48),(.10,.13)], .52)
        d.poly([(-.05,.14),(-.52,.58),(-.20,.58),(.38,.14)], .26)
        d.poly([(.54,-.06),(.73,-.04),(.83,.02),(.52,.02)], .13)
        d.line(-.4,.09,.3,.07,.03,.14)
        d.ellipse(.04,.31,.17,.08,.20)
    elif view == 1: # Head-on: broad wings and central fuselage.
        d.ellipse(0,.04,.15,.29,.37)
        d.poly([(-.96,.06),(-.16,-.03),(.16,-.03),(.96,.06),(.94,.16),(.12,.12),(-.12,.12),(-.94,.16)], .36)
        d.poly([(-.04,-.11),(-.05,-.64),(.04,-.64),(.07,-.11)], .31)
        d.poly([(-.42,-.11),(0,-.20),(.42,-.11),(.42,-.03),(-.42,-.03)], .41)
        d.ellipse(-.43,.18,.09,.10,.16)
        d.ellipse(.43,.18,.09,.10,.16)
        d.ellipse(0,-.04,.09,.05,.14)
    elif view == 2: # Overhead: swept wings and tailplane.
        d.ellipse(0,0,.115,.91,.36)
        d.poly([(-.08,-.24),(-.94,.30),(-.94,.46),(-.08,.18),(.08,.18),(.94,.46),(.94,.30),(.08,-.24)], .33)
        d.poly([(-.06,.53),(-.39,.78),(-.39,.89),(0,.76),(.39,.89),(.39,.78),(.06,.53)], .30)
        d.ellipse(0,-.49,.065,.16,.14)
        d.ellipse(-.31,.12,.075,.19,.22)
        d.ellipse(.31,.12,.075,.19,.22)
    else: # Diagonal three-quarter aircraft.
        d.poly([(-.86,.45),(-.73,.28),(.64,-.47),(.94,-.48),(.83,-.24),(-.58,.51)], .36)
        d.poly([(-.15,.06),(-.68,-.39),(-.52,-.51),(.35,-.11)], .33)
        d.poly([(-.02,.07),(.22,.73),(.44,.71),(.27,-.16)], .28)
        d.poly([(-.65,.34),(-.80,-.08),(-.65,-.17),(-.40,.23)], .28)
        d.ellipse(.65,-.37,.13,.055,.15, -.5)


def car(d, view, m):
    if view in (0,3):
        van = view == 3
        roof = -.53 if van else -.40
        d.poly([(-.95,.02),(-.76,-.10),(-.49,-.12),(-.30,roof),(.35,roof),(.64,-.10),(.91,-.02),(.96,.33),(-.96,.33)], .35)
        if van:
            d.poly([(-.70,-.48),(-.24,-.48),(-.24,-.16),(-.72,-.16)], .61)
            d.poly([(-.19,-.48),(.31,-.48),(.55,-.15),(-.19,-.15)], .16)
        else:
            d.poly([(-.43,-.12),(-.25,-.34),(-.02,-.34),(-.02,-.12)], .16)
            d.poly([(.04,-.34),(.31,-.34),(.54,-.12),(.04,-.12)], .18)
        d.line(-.89,.16,.88,.16,.035,.57)
        d.wheel(-.57,.32,.20)
        d.wheel(.57,.32,.20)
        d.ellipse(.84,.01,.08,.055,.70)
    elif view == 1:
        d.poly([(-.69,-.26),(-.48,-.65),(.48,-.65),(.69,-.26),(.80,.42),(-.80,.42)], .34)
        d.poly([(-.54,-.25),(-.39,-.56),(.39,-.56),(.54,-.25)], .17)
        d.poly([(-.70,-.14),(.70,-.14),(.68,.09),(-.68,.09)], .47)
        d.poly([(-.33,.19),(.33,.19),(.33,.34),(-.33,.34)], .14)
        d.ellipse(-.55,.14,.14,.08,.73)
        d.ellipse(.55,.14,.14,.08,.73)
        d.line(-.77,.45,-.75,.62,.16,.12)
        d.line(.77,.45,.75,.62,.16,.12)
    else:
        d.poly([(-.91,-.01),(-.65,-.21),(-.36,-.56),(.32,-.53),(.77,-.19),(.89,.28),(.28,.53),(-.85,.33)], .34)
        d.poly([(-.55,-.19),(-.30,-.49),(.12,-.47),(.31,-.18)], .17)
        d.poly([(.21,-.46),(.64,-.17),(.40,-.05),(.32,-.17)], .20)
        d.line(-.82,.14,.30,.28,.035,.60)
        d.wheel(-.57,.32,.18)
        d.ellipse(.56,.38,.15,.20,.12)
        d.ellipse(.56,.38,.065,.09,.57)
        d.ellipse(.58,.06,.10,.055,.72)


def bird(d, view, m):
    if view == 0: # Perched side view.
        d.line(-.94,.64,.94,.56,.055,.39)
        d.poly([(-.27,.22),(-.81,.57),(-.69,.18),(-.18,-.03)], .30)
        d.ellipse(0,.00,.40,.28,.36, -.4)
        d.ellipse(.36,-.33,.23,.22,.31)
        d.poly([(.54,-.37),(.88,-.25),(.55,-.20)], .20)
        d.ellipse(-.04,.00,.24,.18,.23, -.3)
        d.eye(.42,-.38,.035)
        d.line(.00,.23,.10,.59,.035,.19)
        d.line(.19,.19,.22,.59,.035,.19)
    elif view == 1: # Wings spread upward.
        d.ellipse(0,.04,.14,.42,.32)
        d.ellipse(0,-.43,.13,.15,.30)
        d.poly([(-.08,-.06),(-.41,-.56),(-.93,-.72),(-.82,-.26),(-.40,.17),(-.10,.27)], .34)
        d.poly([(.08,-.06),(.41,-.56),(.93,-.72),(.82,-.26),(.40,.17),(.10,.27)], .34)
        d.poly([(-.09,.33),(-.23,.76),(0,.66),(.23,.76),(.09,.33)], .25)
        d.line(-.45,-.36,-.79,-.46,.035,.55)
        d.line(.45,-.36,.79,-.46,.035,.55)
    elif view == 2: # Flight, broad horizontal wings.
        d.ellipse(0,.01,.12,.37,.31)
        d.ellipse(0,-.38,.12,.15,.32)
        d.poly([(-.08,-.18),(-.51,-.34),(-.97,-.10),(-.90,.06),(-.52,.03),(-.08,.21)], .34)
        d.poly([(.08,-.18),(.51,-.34),(.97,-.10),(.90,.06),(.52,.03),(.08,.21)], .34)
        d.poly([(-.06,.26),(-.24,.64),(0,.54),(.24,.64),(.06,.26)], .24)
    else: # Upright frontal small bird.
        d.line(-.9,.77,.9,.72,.045,.40)
        d.ellipse(0,.16,.37,.47,.37)
        d.ellipse(0,-.36,.28,.28,.31)
        d.ellipse(-.27,.20,.12,.31,.23, -.2)
        d.ellipse(.27,.20,.12,.31,.23, .2)
        d.ellipse(0,.21,.21,.30,.58)
        d.eye(-.12,-.40,.035)
        d.eye(.12,-.40,.035)
        d.poly([(-.07,-.32),(.07,-.32),(0,-.19)], .15)
        d.line(-.12,.53,-.13,.74,.035,.16)
        d.line(.12,.53,.13,.74,.035,.16)


def feline_face(d, cy=-.10, scale=1):
    # Upright pointed ears, wide short face, paired eyes, short muzzle.
    def e(x,y,rx,ry,v=.35): d.ellipse(x*scale,cy+y*scale,rx*scale,ry*scale,v)
    d.poly([(-.51*scale,cy-.30*scale),(-.49*scale,cy-.76*scale),(-.12*scale,cy-.47*scale)], .32)
    d.poly([(.51*scale,cy-.30*scale),(.49*scale,cy-.76*scale),(.12*scale,cy-.47*scale)], .32)
    e(0,0,.59,.49,.38)
    e(-.22,-.07,.14,.07,.16)
    e(.22,-.07,.14,.07,.16)
    e(-.15,.20,.19,.14,.59)
    e(.15,.20,.19,.14,.59)
    d.poly([(-.09*scale,cy+.13*scale),(.09*scale,cy+.13*scale),(0,cy+.24*scale)], .13)
    d.line(0,cy+.24*scale,0,cy+.34*scale,.025*scale,.15)
    for s in (-1,1):
        d.line(s*.25*scale,cy+.20*scale,s*.70*scale,cy+.14*scale,.018*scale,.24)
        d.line(s*.25*scale,cy+.27*scale,s*.70*scale,cy+.31*scale,.018*scale,.24)


def cat(d, view, m):
    if view == 0:
        feline_face(d, .02, 1.10)
    elif view == 1: # Sitting.
        d.ellipse(0,.30,.39,.55,.36)
        d.ellipse(-.28,.59,.25,.23,.30)
        d.ellipse(.28,.59,.25,.23,.30)
        d.line(-.14,.20,-.14,.80,.10,.50)
        d.line(.14,.20,.14,.80,.10,.50)
        d.line(.25,.64,.70,.67,.14,.31)
        d.line(.70,.67,.78,.36,.13,.31)
        feline_face(d, -.43, .57)
    elif view == 2: # Side-standing feline.
        d.ellipse(-.11,-.05,.57,.26,.36)
        d.ellipse(.49,-.21,.26,.25,.35)
        d.poly([(.32,-.31),(.33,-.66),(.50,-.42),(.64,-.38),(.73,-.63),(.74,-.16)], .34)
        for x, lean in ((-.50,-.02),(-.27,.03),(.25,-.01),(.43,.03)):
            d.line(x,.07,x+lean,.66,.075,.34)
        d.line(-.64,-.09,-.89,-.43,.10,.31)
        d.line(-.89,-.43,-.82,-.72,.09,.31)
        d.eye(.57,-.27,.035)
        d.ellipse(.71,-.13,.08,.055,.17)
    else: # Curled/reclining, head visible.
        d.ellipse(-.10,.25,.70,.40,.36)
        d.ellipse(-.31,.26,.36,.24,.48)
        d.line(-.65,.43,.20,.50,.13,.25)
        # Put the head to the right without copying any image geometry.
        d.ellipse(.44,-.16,.34,.31,.39)
        d.poly([(.16,-.30),(.16,-.63),(.39,-.43),(.59,-.43),(.77,-.61),(.73,-.21)], .34)
        d.eye(.32,-.17,.055)
        d.eye(.57,-.17,.055)
        d.ellipse(.45,-.02,.045,.035,.13)


def quadruped(d, species, view, m):
    is_deer = species == 'deer'
    body = .33 if is_deer else .30
    if view in (0,2):
        # Long legs, horizontal torso, ascending neck and elongated head.
        d.ellipse(-.17,-.08,.57,.27,body)
        d.poly([(.17,-.13),(.40,-.70),(.60,-.65),(.55,.02),(.28,.18)], body)
        d.ellipse(.57,-.65,.22,.13,body, .4)
        d.ellipse(.74,-.55,.18,.10,.39,.4)
        if is_deer:
            d.ellipse(.39,-.82,.08,.20,.32,-.6)
            d.ellipse(.63,-.85,.075,.19,.32,.4)
        else:
            d.poly([(.37,-.66),(.39,-.92),(.50,-.73)], .23)
            d.poly([(.56,-.68),(.62,-.91),(.65,-.66)], .23)
            d.line(.31,-.54,.21,-.05,.08,.17) # Mane.
        for i,x in enumerate((-.59,-.37,.16,.35)):
            lean = (.14 if i % 2 else -.13) if view == 2 else (.03 if i % 2 else -.03)
            d.line(x,.05,x+lean,.77,.055 if is_deer else .075,body)
            d.line(x+lean-.025,.77,x+lean+.06,.77,.07,.15)
        d.line(-.68,-.15,-.86,.26,.045 if is_deer else .10,.21)
        d.eye(.62,-.69,.028)
        if is_deer and m > 0:
            for s in (-1,1):
                x = .50+s*.10
                d.line(x,-.75,x+s*.15,-1.03,.028,.24)
                d.line(x+s*.06,-.88,x+s*.20,-.93,.026,.24)
    elif view == 1: # Frontal head and neck.
        d.ellipse(0,.44,.33,.51,body)
        d.ellipse(0,-.14,.26,.53,.39)
        d.ellipse(0,.23,.30,.18,.24)
        for s in (-1,1):
            d.ellipse(s*.28,-.64,.11,.26,body,s*.45)
            d.eye(s*.17,-.32,.045)
            d.ellipse(s*.15,.22,.05,.035,.10)
        if not is_deer:
            d.poly([(-.055,-.60),(.055,-.60),(.075,.11),(-.075,.11)], .65)
            d.line(-.22,-.56,-.28,.03,.055,.17)
        elif m > 0:
            for s in (-1,1):
                d.line(s*.19,-.62,s*.52,-.99,.045,.28)
                d.line(s*.40,-.86,s*.42,-1.08,.035,.28)
                d.line(s*.31,-.76,s*.64,-.81,.035,.28)
    else: # Grazing, neck bent down.
        d.ellipse(-.21,-.21,.59,.30,body)
        d.poly([(.23,-.28),(.50,-.16),(.66,.52),(.43,.63),(.20,.00)], body)
        d.ellipse(.58,.56,.15,.24,.34,-.3)
        for i,x in enumerate((-.65,-.42,.08,.28)):
            d.line(x,-.04,x+(.04 if i%2 else -.03),.76,.055 if is_deer else .075,body)
        d.line(-.76,-.28,-.91,.21,.045 if is_deer else .09,.24)
        d.ellipse(.41,.30,.08,.18,.30,-.55)
        d.eye(.56,.48,.03)
        if not is_deer: d.line(.27,-.15,.47,.42,.065,.17)


def canine_face(d, cy=-.10, scale=1, pointed=False):
    d.ellipse(0,cy,.48*scale,.53*scale,.39)
    if pointed:
        d.poly([(-.44*scale,cy-.18*scale),(-.47*scale,cy-.88*scale),(-.10*scale,cy-.46*scale)], .26)
        d.poly([(.44*scale,cy-.18*scale),(.47*scale,cy-.88*scale),(.10*scale,cy-.46*scale)], .26)
    else:
        d.ellipse(-.47*scale,cy+.02*scale,.20*scale,.48*scale,.22,-.18)
        d.ellipse(.47*scale,cy+.02*scale,.20*scale,.48*scale,.22,.18)
    d.ellipse(-.19*scale,cy-.10*scale,.07*scale,.05*scale,.11)
    d.ellipse(.19*scale,cy-.10*scale,.07*scale,.05*scale,.11)
    d.ellipse(0,cy+.25*scale,.29*scale,.24*scale,.60)
    d.ellipse(0,cy+.16*scale,.13*scale,.09*scale,.10)
    d.line(0,cy+.22*scale,0,cy+.37*scale,.025*scale,.18)
    d.line(-.14*scale,cy+.39*scale,.14*scale,cy+.39*scale,.026*scale,.18)


def dog(d, view, m):
    if view == 0:
        canine_face(d, .00, 1.02)
    elif view == 1: # Sitting, floppy ears.
        d.ellipse(0,.30,.39,.53,.35)
        d.ellipse(-.27,.66,.24,.17,.31)
        d.ellipse(.27,.66,.24,.17,.31)
        d.line(-.15,.19,-.17,.82,.12,.51)
        d.line(.15,.19,.17,.82,.12,.51)
        canine_face(d, -.43, .63)
    elif view == 2: # Side view with muzzle and raised tail.
        d.ellipse(-.16,-.08,.57,.29,.35)
        d.ellipse(.41,-.36,.28,.30,.36)
        d.ellipse(.68,-.28,.24,.12,.44)
        d.ellipse(.80,-.28,.075,.065,.12)
        d.ellipse(.27,-.30,.14,.31,.21,.20)
        for i,x in enumerate((-.58,-.34,.19,.39)):
            d.line(x,.03,x+(.04 if i%2 else -.04),.69,.09,.34)
        d.line(-.67,-.13,-.88,-.46,.11,.32)
        d.line(-.88,-.46,-.78,-.65,.10,.32)
        d.eye(.51,-.42,.035)
    else: # A longer pointed-ear canine face, not a cat muzzle.
        canine_face(d, .12, .88, True)
        d.poly([(-.055,-.27),(.055,-.27),(.12,.28),(-.12,.28)], .62)
        d.ellipse(0,.34,.12,.075,.10)


def frog(d, view, m):
    if view in (0,1):
        d.ellipse(0,.22,.52,.38,.37)
        for s in (-1,1):
            d.ellipse(s*.57,.32,.28,.27,.32,s*.5)
            d.line(s*.46,.37,s*.85,.63,.10,.34)
            d.line(s*.85,.63,s*.46,.70,.075,.34)
            d.line(s*.31,.18,s*.37,.64,.075,.40)
            d.line(s*.37,.64,s*.63,.69,.055,.40)
        d.ellipse(0,-.20,.57,.32,.40)
        for s in (-1,1):
            d.ellipse(s*.36,-.43,.17,.17,.38)
            d.eye(s*.36,-.45,.075)
        d.line(-.40,-.08,.40,-.08,.035,.16)
        if view == 1:
            for x,y in ((-.16,.20),(.19,.35),(-.23,.43),(.08,.07)):
                d.ellipse(x,y,.065,.045,.24)
    elif view == 2: # Side, crouched frog with powerful folded hind leg.
        d.ellipse(-.08,.05,.57,.33,.36)
        d.ellipse(.42,-.10,.38,.25,.39)
        d.ellipse(.43,-.31,.14,.14,.37)
        d.eye(.46,-.33,.065)
        d.ellipse(-.46,.20,.34,.27,.29,-.3)
        d.line(-.60,.24,-.84,.48,.11,.34)
        d.line(-.84,.48,-.16,.54,.085,.34)
        d.line(.39,.11,.30,.48,.08,.40)
        d.line(.30,.48,.69,.51,.06,.40)
        d.line(.48,-.02,.77,-.02,.025,.17)
    else: # Overhead spread limbs and broad body.
        d.ellipse(0,.05,.40,.52,.37)
        d.ellipse(0,-.42,.47,.26,.39)
        for s in (-1,1):
            d.ellipse(s*.43,.40,.22,.28,.30,s*.6)
            d.line(s*.48,.43,s*.84,.67,.09,.35)
            d.line(s*.84,.67,s*.55,.82,.065,.35)
            d.line(s*.34,-.20,s*.71,.02,.07,.38)
            d.line(s*.71,.02,s*.83,-.26,.055,.38)
            d.eye(s*.29,-.53,.065)
        for x,y in ((-.16,-.05),(.14,.13),(-.13,.32)):
            d.ellipse(x,y,.06,.085,.24)


def ship(d, view, m):
    if view in (0,3):
        d.poly([(-.97,.24),(.95,.24),(.73,.57),(-.68,.57)], .29)
        d.line(-.92,.27,.86,.27,.045,.60)
        if view == 0:
            d.poly([(-.60,-.04),(-.60,.22),(.57,.22),(.53,-.04)], .49)
            d.poly([(-.04,-.36),(.40,-.36),(.40,.02),(-.04,.02)], .56)
            d.poly([(-.31,-.45),(-.13,-.45),(-.13,-.02),(-.31,-.02)], .26)
            for x in (-.44,-.23,-.02,.19,.40): d.ellipse(x,.09,.045,.045,.16)
            d.line(.17,-.36,.17,-.73,.035,.29)
        else: # Container/freighter profile.
            d.poly([(-.70,-.20),(.41,-.20),(.41,.22),(-.70,.22)], .43)
            for x in (-.49,-.25,-.01,.23): d.line(x,-.18,x,.20,.025,.24)
            d.poly([(.43,-.48),(.68,-.48),(.68,.24),(.43,.24)], .56)
            d.poly([(.46,-.39),(.64,-.39),(.64,-.29),(.46,-.29)], .15)
    elif view == 1: # Sailboat with mast and triangular sails.
        d.poly([(-.83,.46),(.85,.46),(.58,.73),(-.53,.73)], .29)
        d.line(.02,-.94,.02,.48,.035,.28)
        d.poly([(-.05,-.82),(-.74,.34),(-.05,.34)], .51)
        d.poly([(.08,-.66),(.70,.33),(.08,.33)], .61)
        d.line(-.88,.76,.89,.76,.025,.48)
    else: # Bow-on vessel, broad superstructure and tapered hull.
        d.poly([(-.69,.01),(.69,.01),(.39,.60),(0,.79),(-.39,.60)], .29)
        d.poly([(-.51,-.10),(-.42,-.49),(.42,-.49),(.51,-.10)], .54)
        d.poly([(-.35,-.56),(.35,-.56),(.35,-.43),(-.35,-.43)], .18)
        d.line(0,-.93,0,-.55,.035,.26)
        d.line(0,.05,0,.67,.03,.51)
        d.line(-.61,.08,.61,.08,.035,.63)


def truck(d, view, m):
    if view in (0,3):
        # Large rectangular cargo box, distinct cab and heavy wheels.
        d.poly([(-.96,-.64),(.22,-.64),(.22,.25),(-.96,.25)], .43)
        d.poly([(.25,-.46),(.64,-.46),(.90,-.10),(.94,.29),(.25,.29)], .33)
        d.poly([(.32,-.38),(.59,-.38),(.78,-.10),(.32,-.10)], .15)
        d.line(-.94,.13,.88,.13,.05,.24)
        d.wheel(-.68,.34,.18)
        d.wheel(-.33,.34,.18)
        d.wheel(.65,.34,.20)
        if view == 3:
            # Rigid delivery/bus-like lorry body, not a pickup.
            d.poly([(-.89,-.57),(.16,-.57),(.16,-.32),(-.89,-.32)], .61)
            for x in (-.80,-.57,-.34,-.11): d.line(x,-.25,x,.10,.025,.32)
    elif view == 1:
        d.poly([(-.64,-.80),(.64,-.80),(.70,.58),(-.70,.58)], .41)
        d.poly([(-.55,-.56),(.55,-.56),(.55,-.16),(-.55,-.16)], .16)
        d.poly([(-.37,.01),(.37,.01),(.37,.36),(-.37,.36)], .22)
        for y in (.07,.17,.27): d.line(-.33,y,.33,y,.025,.57)
        d.ellipse(-.52,.27,.10,.07,.72)
        d.ellipse(.52,.27,.10,.07,.72)
        d.line(-.63,.57,-.63,.77,.19,.12)
        d.line(.63,.57,.63,.77,.19,.12)
    else:
        d.poly([(-.87,-.62),(.02,-.80),(.43,-.48),(.43,.24),(-.87,.43)], .43)
        d.poly([(.03,-.79),(.43,-.49),(.43,.24),(.03,.04)], .51)
        d.poly([(.43,-.34),(.70,-.26),(.90,.05),(.89,.54),(.42,.34)], .31)
        d.poly([(.49,-.24),(.66,-.17),(.80,.05),(.49,-.02)], .15)
        d.wheel(-.64,.42,.18)
        d.wheel(-.28,.36,.17)
        d.ellipse(.67,.51,.16,.20,.12)
        d.ellipse(.67,.51,.065,.085,.57)
        d.ellipse(.79,.27,.07,.045,.70)


DRAWERS = (plane, car, bird, cat,
           lambda d,v,m: quadruped(d,'deer',v,m), dog, frog,
           lambda d,v,m: quadruped(d,'horse',v,m), ship, truck)


def cells(a, step):
    c,h,w = a.shape
    return a.reshape(c,h//step,step,w//step,step).mean((2,4))


def correlate(a, kernels):
    # Each invocation processes just one drawing, solely to form its template.
    patches = np.lib.stride_tricks.sliding_window_view(a, (5,5))
    return np.einsum('yxij,kij->kyx', patches, kernels, optimize=False)


def foreground_front_end():
    # Per-image background extrapolation, using only that image at inference.
    layer('rgb_spatial_norm','norm','image')
    pool('rgb_half','rgb_spatial_norm',(2,2),(2,2))
    h = 16
    yy,xx = np.mgrid[:h,:h]
    border = (yy<2)|(yy>=h-2)|(xx<2)|(xx>=h-2)
    n = 3*h*h
    w = np.zeros((4*n,n),F32)
    for model in range(2):
        for c in range(3):
            for y in range(h):
                if model == 0:
                    bg = border.astype(F32)/border.sum()
                else:
                    bg = ((np.abs(yy-y)<=2)&((xx<2)|(xx>=h-2))).astype(F32)
                    bg /= bg.sum()
                for x in range(h):
                    j = c*h*h+y*h+x
                    row = np.zeros((3,h,h),F32)
                    row[c] = -bg
                    row[c,y,x] += 1
                    w[model*2*n+j] = row.ravel()
                    w[model*2*n+n+j] = -row.ravel()
    dense('background_difference','rgb_half',w)
    layer('background_difference_map','reshape','background_difference',shape=[12,h,h])
    layer('background_difference_positive','relu','background_difference_map')
    conv('foreground_magnitude','background_difference_positive',np.full((1,12,1,1),1/6,F32))
    ident = np.zeros((1,1,3,3),F32)
    ident[0,0,1,1] = 1
    conv('foreground_interior','foreground_magnitude',ident)
    layer('foreground_unit','norm','foreground_interior')


def individual_foreground(a):
    a = (a-a.mean())/np.sqrt(a.var()+EPS)
    a = a.reshape(16,2,16,2).mean((1,3))
    y,x = np.mgrid[:16,:16]
    border = (y<2)|(y>=14)|(x<2)|(x>=14)
    glob = a[border].mean()
    row = np.empty((16,1),F32)
    for j in range(16):
        m = (np.abs(y-j)<=2)&((x<2)|(x>=14))
        row[j,0] = a[m].mean()
    return (.5*(np.abs(a-glob)+np.abs(a-row)))[1:-1,1:-1]


def template_features(a):
    a = (a-a.mean())/np.sqrt(a.var()+EPS)
    z = np.abs(correlate(a, DERIV))
    z = np.maximum(z - .75*z[[2,3,0,1]], 0)
    fine = cells(z, 2)
    coarse = cells(fine, 2)
    sp = correlate(a, DOG[None])[0]
    spots = cells(np.stack([np.maximum(sp,0), np.maximum(-sp,0)]),2)
    lum = a.reshape(16,2,16,2).mean((1,3))[1:-1,1:-1]
    return fine, coarse, spots, lum


def template_unit(a):
    # A single template's own zero-mean, unit-energy correlation coefficients.
    a = a.astype(F32).ravel()
    a = a-a.mean()
    return a / (np.sqrt(np.sum(a*a)+1e-12)*math.sqrt(len(a)))


def context_rule(names, cls, scale, tx, ty):
    # Gaussian foreground/support regions are defined directly by equations.
    # They are not masks, bounding boxes or statistics extracted from renders.
    y,x = np.mgrid[:8,:8].astype(F32)
    x,y = (x+.5)/4-1,(y+.5)/4-1
    center = np.exp(-.5*((x-tx)/(.48*scale))**2-.5*((y-ty)/(.40*scale))**2)
    exterior = 1-center
    above = np.maximum(ty-y,0)
    below = np.maximum(y-ty,0)
    def unit(m): return m/(m.sum()+1e-12)
    center,exterior,above,below = map(unit,(center,exterior,above,below))
    w = np.zeros((len(names),8,8),F32)
    ix = {n:i for i,n in enumerate(names)}
    terms = [
      [('blue',exterior,.055),('blue',above,.035),('green',center,-.035)],
      [('dark',below,.025),('gray',below,.020),('green',center,-.025)],
      [('green',exterior,.025),('red',center,.010),('yellow',center,.010)],
      [('warm',center,.030),('blue',center,-.015)],
      [('green',exterior,.030),('warm',center,.030),('green',center,-.010)],
      [('warm',center,.025),('white',center,.015),('blue',center,-.015)],
      [('green',center,.070),('yellow',center,.025),('blue',center,-.020)],
      [('green',below,.035),('warm',center,.030),('green',center,-.015)],
      [('blue',below,.065),('cyan',below,.025),('green',exterior,-.025)],
      [('gray',below,.020),('white',center,.020),('green',center,-.025)]
    ]
    for colour,region,gain in terms[cls]: w[ix[colour]] += gain*region
    return w


def make_templates(names):
    # Fixed, exhaustive nuisance transforms, never selected using scored images.
    locations = [(x,y) for y in (-.15,0,.15) for x in (-.15,0,.15)]
    scales = (.62,.79,.96)
    angles = (-.16,0,.16)
    morphs = (-1,1)
    mirrors = (-1,1)
    per_class = 4*len(locations)*len(scales)*len(angles)*len(morphs)*len(mirrors)
    n = per_class*10
    wf = np.empty((n,4*14*14),F32)
    wc = np.empty((n,4*7*7),F32)
    wd = np.empty((n,4*12*12),F32)
    ws = np.empty((n,2*14*14),F32)
    wa = np.empty((2*n,14*14),F32)
    wcolour = np.zeros((n,len(names)*64),F32)
    wforeground = np.empty((n,14*14),F32)
    index = 0
    for cls, draw in enumerate(DRAWERS):
        for view in range(4):
            for scale in scales:
                for tx,ty in locations:
                    for angle in angles:
                        for morph in morphs:
                            for mirror in mirrors:
                                d = Drawing(scale,tx,ty,angle,mirror,morph)
                                draw(d,view,morph)
                                image = d.image()
                                f,c,s,a = template_features(image)
                                wforeground[index] = .13*template_unit(individual_foreground(image))
                                wf[index] = .23*template_unit(f)
                                wc[index] = .21*template_unit(c)
                                wd[index] = .18*template_unit(f[:,1:-1,1:-1])
                                ws[index] = .08*template_unit(s)
                                u = .17*template_unit(a)
                                wa[2*index] = u
                                wa[2*index+1] = -u
                                # Position-aware colour context is a separate
                                # analytic rule, not measured on the drawing.
                                wcolour[index] = context_rule(names,cls,scale,tx,ty).ravel()
                                index += 1
        print('constructed independent templates for class', cls, flush=True)
    dense('fine_match', 'fine_unit', wf)
    dense('coarse_match', 'coarse_unit', wc)
    dense('deformable_match','deformable_unit',wd)
    dense('spot_match', 'spot_unit', ws)
    dense('appearance_match', 'appearance_unit', wa)
    dense('aligned_colour_match','colour_layout',wcolour)
    dense('foreground_match','foreground_unit',wforeground)
    layer('appearance_pairs', 'reshape', 'appearance_match', shape=[n,1,2])
    pool('appearance_polarity', 'appearance_pairs', (1,2), (1,2), op='maxpool')
    layer('appearance_best', 'reshape', 'appearance_polarity', shape=[n])
    layer('template_scores', 'add', ['fine_match','coarse_match','deformable_match','spot_match','appearance_best','aligned_colour_match','foreground_match'])
    layer('class_templates', 'reshape', 'template_scores', shape=[10,1,per_class])
    pool('class_shape_max', 'class_templates', (1,per_class), (1,per_class), op='maxpool')
    layer('shape_scores', 'reshape', 'class_shape_max', shape=[10])
    return per_class


def colour_front_end():
    # Rectified RGB opponents, intensity, and absolute colour differences.
    forms = [([1,0,0],0),([0,1,0],0),([0,0,1],0),
             ([.299,.587,.114],0),
             ([1,-1,0],0),([-1,1,0],0),([0,1,-1],0),([0,-1,1],0),
             ([1,0,-1],0),([-1,0,1],0),
             ([-.299,-.587,-.114],.40),([.299,.587,.114],-.60),
             ([0,0,0],1)]
    conv('colour_basis', 'image', np.array([x for x,b in forms],F32)[:,:,None,None],
         np.array([b for x,b in forms],F32))
    layer('colour_basis_positive', 'relu', 'colour_basis')
    names = ['blue','green','warm','cyan','yellow','brown','gray','white','dark','bright','red']
    w = np.zeros((len(names),len(forms),1,1),F32)
    b = np.zeros(len(names),F32)
    def setrow(i,rgb=None,terms=(),bias=0):
        if rgb is not None: w[i,:3,0,0] = rgb
        for j,v in terms: w[i,j,0,0] += v
        b[i] = bias
    setrow(0,[-2,-2,4],bias=-.08)
    setrow(1,[-2,4,-2],bias=-.08)
    setrow(2,[3,-1.5,-1.5],bias=-.06)
    setrow(3,[-2.8,1.4,1.4],bias=-.112)
    setrow(4,[1.2,1.2,-2.4],bias=-.096)
    setrow(5,[1.6,0,-1.6],terms=[(11,-2),(5,-2)],bias=-.03)
    setrow(6,terms=[(4,-6.25),(5,-6.25),(6,-6.25),(7,-6.25),(8,-6.25),(9,-6.25)],bias=1)
    setrow(7,terms=[(3,2.2),(4,-3.3),(5,-3.3),(6,-3.3),(7,-3.3),(8,-3.3),(9,-3.3)],bias=-1.21)
    setrow(8,terms=[(10,2.5)])
    setrow(9,terms=[(11,2.5)])
    setrow(10,[4,-2,-2],bias=-.24)
    conv('colour_evidence', 'colour_basis_positive', w,b)
    layer('colour_evidence_positive','relu','colour_evidence')
    # Saturation at one is a hand-defined bounded colour membership function.
    w = np.zeros((2*len(names),len(names),1,1),F32)
    b = np.zeros(2*len(names),F32)
    for j in range(len(names)):
        w[2*j,j,0,0] = w[2*j+1,j,0,0] = 1
        b[2*j+1] = -1
    conv('colour_cap_pair','colour_evidence_positive',w,b)
    layer('colour_cap_relu','relu','colour_cap_pair')
    w = np.zeros((len(names),2*len(names),1,1),F32)
    for j in range(len(names)):
        w[j,2*j,0,0] = 1
        w[j,2*j+1,0,0] = -1
    conv('colour_membership','colour_cap_relu',w)
    pool('colour_layout','colour_membership',(4,4),(4,4))
    return names


def part_drawing(cls, variant, d):
    # Independent, generic local anatomy, not crops of any other image.
    if cls == 3:
        feline_face(d, .10, .98 if variant == 0 else .81)
        if variant: d.ellipse(0,.67,.47,.24,.38)
    elif cls == 5:
        canine_face(d, .10, .85, pointed=bool(variant))
    elif cls in (4,7):
        quadruped(d,'deer' if cls == 4 else 'horse',1,1 if variant else -1)
    elif cls == 6:
        d.ellipse(0,.18,.69,.47,.38)
        d.ellipse(0,-.20,.76,.38,.40)
        for s in (-1,1):
            d.ellipse(s*.46,-.45,.20,.22,.37)
            d.eye(s*.46,-.47,.085)
        d.line(-.57,-.02,.57,-.02,.045,.17)
        if variant:
            d.ellipse(-.22,.25,.09,.055,.23)
            d.ellipse(.17,.39,.09,.06,.23)
    elif cls == 2:
        if variant:
            bird(d,3,-1)
        else:
            d.ellipse(-.20,.28,.53,.42,.37)
            d.ellipse(.16,-.20,.39,.37,.33)
            d.poly([(.46,-.24),(.92,-.07),(.45,.04)],.20)
            d.eye(.27,-.27,.05)
            d.ellipse(-.29,.29,.30,.24,.24,-.3)
    elif cls == 1:
        if variant:
            car(d,1,-1)
        else:
            d.poly([(-.97,-.13),(.97,-.13),(.97,.35),(-.97,.35)],.35)
            d.line(-.90,.05,.90,.05,.04,.59)
            d.wheel(-.56,.33,.25)
            d.wheel(.56,.33,.25)
    else:
        DRAWERS[cls](d,variant,-1)


def local_windows(name, source, channels, k, square_side, h=14):
    # Reorder overlapping patches, then normalise EACH INPUT PATCH over its
    # own features. This is an exact network wiring, not a fitted transform.
    side = h-k+1
    locations = side*side
    d = square_side*square_side
    w = np.zeros((locations*d,channels*h*h),F32)
    for y in range(side):
        for x in range(side):
            j = (y*side+x)*d
            for c in range(channels):
                for dy in range(k):
                    for dx in range(k):
                        feat = (c*k+dy)*k+dx
                        w[j+feat,(c*h+y+dy)*h+x+dx] = 1
    dense(name+'_extract',source,w)
    layer(name+'_patches','reshape',name+'_extract',shape=[locations,square_side,square_side])
    layer(name+'_unit','norm',name+'_patches')
    layer(name,'reshape',name+'_unit',shape=[1,locations*square_side,square_side])
    return name


def make_parts():
    branches = []
    for k in (5,7):
        hs = 2*k
        ss = math.ceil(math.sqrt(2*k*k))
        source_hog = local_windows('local'+str(k)+'_hog','hog_fine',4,k,hs)
        source_spot = local_windows('local'+str(k)+'_spot','spots',2,k,ss)
        source_lum = local_windows('local'+str(k)+'_lum','interior_luminance',1,k,k)
        hogs, spots, appearances = [],[],[]
        # Every class has exactly the same transform schedule.
        for cls in range(10):
            for variant in (0,1):
                for scale in (.73,.91):
                    for angle in (-.13,0,.13):
                        for mirror in (-1,1):
                            d = Drawing(scale,0,0,angle,mirror,0,size=2*k+4)
                            part_drawing(cls,variant,d)
                            a = d.image()
                            a = (a-a.mean())/np.sqrt(a.var()+EPS)
                            z = np.abs(correlate(a,DERIV))
                            z = np.maximum(z-.75*z[[2,3,0,1]],0)
                            f = cells(z,2)
                            sp = correlate(a,DOG[None])[0]
                            sp = cells(np.stack([np.maximum(sp,0),np.maximum(-sp,0)]),2)
                            q = k+2
                            lum = a.reshape(q,2,q,2).mean((1,3))[1:-1,1:-1]
                            hogs.append(.55*template_unit(f).reshape(1,hs,hs))
                            padded = np.zeros(ss*ss,F32)
                            padded[:sp.size] = sp.ravel()
                            spots.append(.15*template_unit(padded).reshape(1,ss,ss))
                            u = .30*template_unit(lum).reshape(1,k,k)
                            appearances.extend([u,-u])
        n = len(hogs)
        side = 15-k
        prefix = 'part'+str(k)
        conv(prefix+'_edge_corr',source_hog,np.asarray(hogs),stride=hs)
        layer(prefix+'_edges','reshape',prefix+'_edge_corr',shape=[n,side,side])
        conv(prefix+'_spot_corr',source_spot,np.asarray(spots),stride=ss)
        layer(prefix+'_spots','reshape',prefix+'_spot_corr',shape=[n,side,side])
        conv(prefix+'_appearance',source_lum,np.asarray(appearances),stride=k)
        layer(prefix+'_polarity_pairs','reshape',prefix+'_appearance',shape=[n,2,side*side])
        pool(prefix+'_polarity_max',prefix+'_polarity_pairs',(2,1),(2,1),op='maxpool')
        layer(prefix+'_appearance_best','reshape',prefix+'_polarity_max',shape=[n,side,side])
        layer(prefix+'_matches','add',[prefix+'_edges',prefix+'_spots',prefix+'_appearance_best'])
        layer(prefix+'_classes','reshape',prefix+'_matches',shape=[10,(n//10)*side,side])
        pool(prefix+'_class_max',prefix+'_classes',((n//10)*side,side),((n//10)*side,side),op='maxpool')
        layer(prefix+'_ten','reshape',prefix+'_class_max',shape=[10])
        dense(prefix+'_threshold',prefix+'_ten',np.eye(10,dtype=F32),np.full(10,-.30,F32))
        layer(prefix+'_evidence','relu',prefix+'_threshold')
        dense(prefix+'_votes',prefix+'_evidence',np.eye(10,dtype=F32)*.12)
        branches.append(prefix+'_votes')
    layer('part_scores','add',branches)


def structural_rules():
    # Texture density and angular rectilinearity use explicit response scales.
    # These quantities are computed afresh on each input, never over a dataset.
    w = np.zeros((6,10,1,1),F32)
    for i in range(4):
        w[0,i,0,0] = w[0,i+4,0,0] = .25
        w[1 if i%2 == 0 else 2,i,0,0] = 1
        w[1 if i%2 == 0 else 2,i+4,0,0] = 1
    w[3,8,0,0] = w[3,9,0,0] = .5
    w[4,:8,0,0] = .25
    w[5,:8,0,0] = .25
    b = np.array([0,0,0,0,-.10,-.28],F32)
    conv('structure_measures','derivative_positive',w,b)
    layer('structure_positive','relu','structure_measures')
    pool('structure_layout','structure_positive',(4,4),(4,4))
    y,x = np.mgrid[:7,:7]
    masks = dict(all=np.ones((7,7),bool),center=(x>=2)&(x<5)&(y>=2)&(y<5),
                 outer=(x<2)|(x>=5)|(y<2)|(y>=5),
                 top=y<2,bottom=y>=5)
    def f(ch,region):
        z = np.zeros((6,7,7),F32)
        m = masks[region]
        z[ch,m] = 1/m.sum()
        return z.ravel()
    rules = [
      ([(0,'all',1)],-.13,{0:-.16,1:-.04,2:.025,3:.055,4:.035,5:.055,6:.055,7:.025,8:-.17,9:-.04}),
      ([(0,'center',1),(0,'outer',-1)],0,{0:.025,1:.02,2:.025,3:.06,4:.035,5:.06,6:.045,7:.035,8:-.01,9:.02}),
      ([(0,'outer',1)],-.12,{0:-.08,2:.03,3:.015,4:.05,5:.015,6:.04,7:.025,8:-.08}),
      ([(1,'center',1),(2,'center',-1)],0,{0:.035,1:.035,3:-.025,4:-.025,5:-.025,6:-.025,7:-.025,8:.045,9:.045}),
      ([(5,'center',1)],-.025,{0:-.035,2:.025,3:.035,5:.035,6:.025,8:-.055}),
      ([(0,'top',1),(0,'bottom',-1)],0,{0:-.04,2:.015,4:.02,6:.02,8:-.04}),
      ([(3,'center',1)],-.015,{2:.025,3:.04,5:.035,6:.04}),
    ]
    w = np.empty((len(rules),6*49),F32)
    b = np.empty(len(rules),F32)
    ro = np.zeros((10,len(rules)),F32)
    for i,(terms,bias,votes) in enumerate(rules):
        w[i] = 0
        for ch,region,gain in terms: w[i] += gain*f(ch,region)
        b[i] = bias
        for cls,gain in votes.items(): ro[cls,i] = gain
    dense('texture_features','structure_layout',w,b)
    dense('texture_scores','texture_features',ro)


def scene_rules(names):
    ix = {n:i for i,n in enumerate(names)}
    yy,xx = np.mgrid[:8,:8]
    regions = dict(all=np.ones((8,8),bool), top=yy<3, bottom=yy>=5,
                   center=(yy>=2)&(yy<6)&(xx>=2)&(xx<6),
                   outer=(yy<2)|(yy>=6)|(xx<2)|(xx>=6),
                   lowcenter=(yy>=5)&(xx>=1)&(xx<7))
    def feature(colour,region):
        z = np.zeros((len(names),8,8),F32)
        mask = regions[region]
        z[ix[colour],mask] = 1/mask.sum()
        return z.ravel()
    # Modest direct priors; object geometry remains the principal classifier.
    direct = np.zeros((10,len(names)*64),F32)
    priors = [
      [('blue','top',.12),('blue','outer',.07),('white','center',.025),('bright','top',.025),('green','center',-.07),('warm','center',-.04),('dark','bottom',-.03)],
      [('gray','bottom',.045),('dark','lowcenter',.065),('gray','center',.025),('green','center',-.05),('blue','bottom',-.04)],
      [('green','outer',.035),('blue','top',.025),('yellow','center',.02),('red','center',.025),('blue','bottom',-.015)],
      [('warm','center',.045),('gray','center',.02),('dark','center',.015),('blue','top',-.025),('green','center',-.025)],
      [('green','outer',.065),('warm','center',.055),('brown','center',.025),('blue','bottom',-.04)],
      [('warm','center',.05),('brown','center',.02),('white','center',.025),('dark','center',.015),('blue','all',-.025),('green','center',-.025)],
      [('green','center',.15),('yellow','center',.035),('brown','center',.025),('green','outer',.015),('blue','all',-.04),('gray','center',-.025)],
      [('green','bottom',.06),('warm','center',.055),('dark','center',.02),('brown','center',.02),('blue','bottom',-.035),('green','center',-.02)],
      [('blue','bottom',.115),('cyan','bottom',.05),('gray','bottom',.035),('blue','top',.02),('white','center',.02),('green','all',-.08),('warm','center',-.03)],
      [('gray','center',.035),('gray','bottom',.045),('white','center',.025),('dark','lowcenter',.045),('red','center',.02),('green','center',-.04),('blue','bottom',-.025)]
    ]
    for c,terms in enumerate(priors):
        for colour,region,gain in terms:
            direct[c] += gain*feature(colour,region)
    dense('direct_scene_scores','colour_layout',direct)
    # Conjunction-like hinges encode foreground/background relationships.
    specs = [
      # (terms, threshold, {class: a priori contribution})
      ([('blue','top',1),('bright','top',.3),('green','top',-.6)],.10,{0:.10,2:.025,8:.015}),
      ([('blue','bottom',1),('cyan','bottom',.6),('green','bottom',-.6)],.10,{8:.13,0:-.03}),
      ([('green','outer',1),('yellow','outer',.25)],.10,{4:.065,7:.025,2:.025,6:.015}),
      ([('green','center',1),('green','outer',-.5)],.06,{6:.15,4:-.025,7:-.025}),
      ([('warm','center',1),('brown','center',.6),('warm','outer',-.7)],.045,{3:.035,4:.055,5:.045,7:.045,6:.015}),
      ([('white','center',1),('gray','center',.4),('green','center',-.5)],.22,{0:.025,9:.035,1:.02}),
      ([('gray','bottom',1),('dark','bottom',.6),('green','bottom',-.6),('blue','bottom',-.6)],.28,{1:.075,9:.055,8:.015}),
      ([('green','bottom',1),('green','top',-.6)],.06,{7:.07,4:.04,5:.015}),
      ([('blue','outer',1),('blue','center',-.7)],.045,{0:.055,2:.035,8:.025}),
      ([('red','center',1),('gray','bottom',.4),('green','bottom',-.4)],.20,{9:.055,1:.025}),
      ([('yellow','center',1),('brown','center',.6),('blue','center',-.5)],.10,{6:.035,2:.02,4:.015}),
      ([('dark','center',1),('dark','outer',-.7)],.07,{3:.015,5:.02,7:.015}),
      ([('white','center',1),('white','outer',-.8)],.06,{5:.035,3:.02,0:.015}),
      ([('green','outer',1),('green','center',-1),('warm','center',.7)],.05,{4:.06,7:.045,5:.02,2:.02}),
    ]
    w = np.zeros((len(specs),len(names)*64),F32)
    b = np.empty(len(specs),F32)
    readout = np.zeros((10,len(specs)),F32)
    for i,(terms,threshold,votes) in enumerate(specs):
        for colour,region,coefficient in terms:
            w[i] += coefficient*feature(colour,region)
        b[i] = -threshold
        for c,gain in votes.items(): readout[c,i] = gain
    dense('scene_relations','colour_layout',w,b)
    layer('scene_relations_positive','relu','scene_relations')
    dense('relation_scene_scores','scene_relations_positive',readout)
    layer('class_scores','add',['shape_scores','part_scores','texture_scores','direct_scene_scores','relation_scene_scores'])


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    make_front_end()
    foreground_front_end()
    structural_rules()
    names = colour_front_end()
    per_class = make_templates(names)
    make_parts()
    scene_rules(names)
    (out/'network.json').write_text(json.dumps({'layers':LAYERS}, separators=(',',':'))+'\n')
    np.savez(out/'weights.npz', **WEIGHTS)
    print('templates per class:',per_class)
    print('total float32 parameters:',sum(a.size for a in WEIGHTS.values()))
    print('written',out/'network.json','and',out/'weights.npz')


if __name__ == '__main__':
    main()
