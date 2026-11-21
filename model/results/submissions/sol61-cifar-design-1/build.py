#!/usr/bin/env python3
"""Build an entirely hand-specified, data-free small-photo classifier.

No input files are opened.  Shapes below are schematic geometric constructions,
not observations, tracings, prototypes, or descriptors of any dataset images.
There is no optimizer, statistical estimation, parameter search, or training.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

F32 = np.float32
LUMA = np.array([0.299, 0.587, 0.114], dtype=F32)
COLORS = ['red', 'green', 'blue', 'warm', 'white', 'dark',
          'neutral', 'light', 'saturation']


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def op(self, name, op, inputs, **settings):
        self.layers.append(dict(name=name, op=op, inputs=list(inputs), **settings))
        return name

    def linear(self, name, source, W, b=None, conv=False, stride=1, padding=0):
        W = np.asarray(W, dtype=F32)
        if b is None:
            b = np.zeros(W.shape[0], dtype=F32)
        self.weights[name + '.W'] = W
        self.weights[name + '.b'] = np.asarray(b, dtype=F32)
        if conv:
            return self.op(name, 'conv', [source], out=len(W),
                           kernel=list(W.shape[-2:]), stride=stride, padding=padding)
        return self.op(name, 'dense', [source], out=len(W))

    def relu(self, name, source):
        return self.op(name, 'relu', [source])

    def pool(self, name, source, size, stride=None, maximum=False):
        return self.op(name, 'maxpool' if maximum else 'avgpool', [source],
                       size=list(size), stride=list(stride or size))


def gauss_kernel(sigma, radius):
    q = np.arange(-radius, radius + 1, dtype=np.float64)
    a = np.exp(-q * q / (2 * sigma * sigma))
    return a / a.sum()


def blur(a, sigma):
    """Formulaic smoothing of a code-drawn geometric shape, not input data."""
    radius = int(math.ceil(3 * sigma))
    g = gauss_kernel(sigma, radius)
    for axis in (0, 1):
        pads = [(0, 0), (0, 0)]
        pads[axis] = (radius, radius)
        p = np.pad(a, pads, mode='edge')
        a = sum(g[j] * np.take(p, np.arange(a.shape[axis]) + j, axis=axis)
                for j in range(len(g)))
    return a


# A shape is a union of ellipses, polygons, and thick line segments.  'detail'
# primitives change its schematic tone but not the foreground occupancy mask.
class Shape:
    def __init__(self):
        self.parts = []
        self.details = []
        self.points = []

    def ellipse(self, x, y, rx, ry, angle=0, tone=1, detail=False):
        item = ('e', (x, y, rx, ry, angle), tone)
        (self.details if detail else self.parts).append(item)
        return self

    def poly(self, xy, tone=1, detail=False):
        (self.details if detail else self.parts).append(('p', tuple(xy), tone))
        return self

    def line(self, xy, width=0.04, tone=1, detail=False):
        for a, b in zip(xy[:-1], xy[1:]):
            (self.details if detail else self.parts).append(('l', (a, b, width), tone))
        return self

    def rect(self, x0, y0, x1, y1, tone=1, detail=False):
        return self.poly([(x0,y0), (x1,y0), (x1,y1), (x0,y1)], tone, detail)

    def spot(self, x, y, r=0.05, strength=1, scale=0):
        self.ellipse(x, y, r, r, tone=0.05, detail=True)
        self.points.append((x, y, strength, scale))
        return self

    def wheel(self, x, y, r=0.17):
        self.ellipse(x, y, r, r, tone=0.15)
        self.ellipse(x, y, r * 0.45, r * 0.45, tone=0.7, detail=True)
        self.points.append((x, y, 1, 1))
        return self


def inside(kind, args, x, y):
    if kind == 'e':
        cx, cy, rx, ry, angle = args
        c, s = math.cos(angle), math.sin(angle)
        u, v = x-cx, y-cy
        return ((c*u+s*v)/rx)**2 + ((-s*u+c*v)/ry)**2 <= 1
    if kind == 'l':
        (ax, ay), (bx, by), width = args
        dx, dy = bx-ax, by-ay
        t = np.clip(((x-ax)*dx+(y-ay)*dy) / (dx*dx+dy*dy+1e-12), 0, 1)
        return (x-ax-t*dx)**2+(y-ay-t*dy)**2 <= (width/2)**2
    hit = np.zeros(x.shape, dtype=bool)
    for a, b in zip(args, args[1:] + args[:1]):
        ax, ay = a
        bx, by = b
        if abs(by-ay) < 1e-12:
            continue
        hit ^= ((ay > y) != (by > y)) & (x < (bx-ax)*(y-ay)/(by-ay)+ax)
    return hit


def render(shape, scale, dx, dy, angle, mirror):
    """Rasterize only a hand-authored geometric construction, at 2x resolution."""
    q = (np.arange(64) + 0.5) / 2 - 16
    xx, yy = np.meshgrid(q-dx, q-dy)
    c, s = math.cos(angle), math.sin(angle)
    x, y = (c*xx+s*yy)/scale, (-s*xx+c*yy)/scale
    x = x * mirror
    mask = np.zeros(x.shape, dtype=bool)
    tone = np.zeros(x.shape, dtype=np.float64)
    for kind, args, value in shape.parts:
        m = inside(kind, args, x, y)
        mask |= m
        tone[m] = value
    for kind, args, value in shape.details:
        m = inside(kind, args, x, y) & mask
        tone[m] = value
    mask = mask.astype(np.float64).reshape(32,2,32,2).mean((1,3))
    tone = tone.reshape(32,2,32,2).mean((1,3))
    # Expected unsigned edge orientation of the schematic's contours.
    # Central derivatives and a little smoothing give anti-aliased normals.
    t = blur(tone, 0.65)
    p = np.pad(t, 1, mode='edge')
    gx = (p[1:-1,2:] - p[1:-1,:-2]) * 0.5
    gy = (p[2:,1:-1] - p[:-2,1:-1]) * 0.5
    directions = [gx, (gx+gy)/math.sqrt(2), gy, (gy-gx)/math.sqrt(2)]
    absolute = np.abs(np.stack(directions))
    hog = np.maximum(absolute - np.roll(absolute, 2, axis=0), 0)
    hog = hog.reshape(4,16,2,16,2).mean((2,4))
    occ16 = mask.reshape(16,2,16,2).mean((1,3))
    occ8 = mask.reshape(8,4,8,4).mean((1,3))
    blobs = np.zeros((2,16,16), dtype=np.float64)
    grid_y, grid_x = np.mgrid[:16,:16]
    for px, py, strength, which in shape.points:
        px *= mirror
        tx = scale*(c*px-s*py)+dx+16
        ty = scale*(s*px+c*py)+dy+16
        bx, by = (tx-1)/2, (ty-1)/2
        if -0.5 <= bx <= 15.5 and -0.5 <= by <= 15.5:
            g = np.exp(-((grid_x-bx)**2+(grid_y-by)**2)/(2*0.7**2))
            blobs[which] += strength*g/(g.sum()+1e-12)
    return hog, occ16, occ8, blobs


def airplane_models():
    models = []
    # Airliner, side view: tapered tube, vertical tail, swept main wing.
    a = Shape().poly([(-.98,.02),(-.80,-.10),(.72,-.10),(.98,.04),(.76,.14),(-.82,.14)])
    a.poly([(-.79,-.08),(-.84,-.58),(-.63,-.55),(-.40,-.06)])
    a.poly([(-.18,.02),(.15,.04),(.53,.50),(.30,.50)])
    a.poly([(-.75,.06),(-.57,.06),(-.44,.30),(-.65,.28)])
    a.line([(-.53,-.025),(.66,-.025)], .035, .2, True)
    a.ellipse(.31,.24,.16,.08,tone=.35)
    models.append(a)
    # Compact swept fighter profile.
    a = Shape().poly([(-.98,.08),(-.52,-.08),(.42,-.12),(.98,.05),(.35,.18),(-.70,.16)])
    a.poly([(-.72,-.03),(-.68,-.52),(-.44,-.50),(-.22,.02)])
    a.poly([(-.18,.03),(.18,.03),(.61,.51),(-.01,.26)])
    a.ellipse(.31,-.10,.24,.07,tone=.15,detail=True)
    models.append(a)
    # Overhead view with wings normal to the fuselage.
    a = Shape().ellipse(0,0,.13,.98)
    a.poly([(-.12,-.23),(-.94,.24),(-.94,.40),(-.08,.20),(.08,.20),(.94,.40),(.94,.24),(.12,-.23)])
    a.poly([(-.08,.58),(-.43,.81),(-.43,.92),(.43,.92),(.43,.81),(.08,.58)])
    a.ellipse(0,-.61,.075,.16,tone=.2,detail=True)
    models.append(a)
    # Head-on aircraft: broad wings, central nose and tail.
    a = Shape().poly([(-.98,.12),(-.98,.19),(-.18,.13),(.18,.13),(.98,.19),(.98,.12),(.13,-.01),(-.13,-.01)])
    a.ellipse(0,.04,.19,.21).poly([(-.05,-.10),(-.06,-.62),(.06,-.62),(.05,-.10)])
    a.ellipse(-.43,.22,.10,.11).ellipse(.43,.22,.10,.11)
    a.ellipse(0,-.015,.12,.045,tone=.1,detail=True)
    models.append(a)
    # Diagonal top view, constructed by rotating a long fuselage and swept wings.
    a = Shape().ellipse(0,0,.11,.97, angle=-.62)
    a.poly([(-.16,-.14),(-.87,-.23),(-.90,-.08),(-.08,.18),(.18,.16),(.68,.76),(.82,.70),(.18,-.12)])
    a.poly([(.34,.48),(.04,.63),(.10,.77),(.55,.72),(.68,.38),(.56,.32)])
    models.append(a)
    # Delta-wing planform.
    a = Shape().poly([(0,-.97),(-.13,-.23),(-.92,.69),(-.15,.55),(-.09,.95),(.09,.95),(.15,.55),(.92,.69),(.13,-.23)])
    a.ellipse(0,-.40,.06,.20,tone=.2,detail=True)
    models.append(a)
    return models


def car_models():
    models = []
    a = Shape().poly([(-.96,.03),(-.72,-.12),(-.40,-.47),(.34,-.47),(.69,-.13),(.94,-.02),(.94,.35),(-.94,.35)])
    a.poly([(-.58,-.14),(-.34,-.40),(.25,-.40),(.53,-.14)], .15, True)
    a.line([(-.05,-.41),(-.05,.28)], .025,.4,True)
    a.wheel(-.59,.32).wheel(.60,.32)
    a.line([(-.9,.15),(.9,.15)], .025,.7,True)
    models.append(a)
    # Front view: windscreen above bonnet, lamps and low bumper.
    a = Shape().poly([(-.78,.68),(-.81,.03),(-.59,-.63),(.59,-.63),(.81,.03),(.78,.68)])
    a.poly([(-.56,-.53),(.56,-.53),(.66,-.06),(-.66,-.06)], .15, True)
    a.rect(-.65,.11,-.33,.23,.1,True).rect(.33,.11,.65,.23,.1,True)
    a.rect(-.36,.30,.36,.46,.15,True).line([(-.69,.56),(.69,.56)],.05,.2,True)
    a.wheel(-.68,.63,.14).wheel(.68,.63,.14)
    models.append(a)
    # Rear hatchback, relatively upright glazing.
    a = Shape().poly([(-.74,.64),(-.76,-.04),(-.56,-.66),(.56,-.66),(.76,-.04),(.74,.64)])
    a.rect(-.53,-.52,.53,-.13,.15,True)
    a.rect(-.68,.10,-.50,.37,.2,True).rect(.50,.10,.68,.37,.2,True)
    a.rect(-.24,.31,.24,.45,.3,True)
    a.wheel(-.65,.63,.14).wheel(.65,.63,.14)
    models.append(a)
    # Three-quarter view with a sloping bonnet and perspective windscreen.
    a = Shape().poly([(-.96,-.03),(-.53,-.25),(-.24,-.66),(.46,-.49),(.74,-.02),(.94,.16),(.82,.56),(-.84,.38)])
    a.poly([(-.45,-.25),(-.19,-.57),(.38,-.43),(.58,-.08)], .15, True)
    a.line([(-.91,.08),(.70,.28),(.90,.18)],.035,.35,True)
    a.wheel(-.60,.33,.19).wheel(.60,.46,.19)
    a.rect(.45,.15,.72,.25,.15,True)
    models.append(a)
    # Van, viewed from the front.
    a = Shape().rect(-.68,-.78,.68,.64)
    a.rect(-.58,-.64,.58,-.14,.12,True).line([(0,-.63),(0,-.15)],.025,.65,True)
    a.rect(-.56,.17,-.28,.28,.12,True).rect(.28,.17,.56,.28,.12,True)
    a.rect(-.3,.34,.3,.49,.15,True)
    a.wheel(-.60,.62,.14).wheel(.60,.62,.14)
    models.append(a)
    # SUV side elevation.
    a = Shape().poly([(-.97,-.02),(-.69,-.57),(.46,-.57),(.66,-.16),(.96,-.07),(.96,.36),(-.96,.36)])
    a.poly([(-.59,-.46),(.34,-.46),(.50,-.16),(-.72,-.16)],.12,True)
    a.line([(-.22,-.45),(-.22,.30)],.035,.55,True)
    a.wheel(-.61,.35,.20).wheel(.61,.35,.20)
    models.append(a)
    return models


def truck_models():
    models = []
    a = Shape().rect(-.98,-.65,.35,.22)
    a.poly([(.39,-.26),(.73,-.26),(.96,.02),(.96,.36),(.36,.36)])
    a.poly([(.47,-.18),(.69,-.18),(.85,.01),(.47,.01)],.15,True)
    for x in [-.70,-.44,.51,.78]: a.wheel(x,.34,.15)
    a.line([(-.92,.10),(.27,.10)],.03,.6,True)
    models.append(a)
    a = Shape().rect(-.70,-.93,.70,.67)
    a.rect(-.60,-.65,.60,-.13,.10,True).line([(0,-.64),(0,-.14)],.03,.7,True)
    a.rect(-.48,.05,.48,.37,.22,True)
    for y in [.12,.22,.31]: a.line([(-.43,y),(.43,y)],.025,.65,True)
    a.rect(-.61,.44,-.36,.54,.1,True).rect(.36,.44,.61,.54,.1,True)
    a.wheel(-.61,.66,.15).wheel(.61,.66,.15)
    models.append(a)
    a = Shape().poly([(-.97,-.57),(.20,-.84),(.66,-.56),(.66,.22),(-.84,.49),(-.97,.31)])
    a.poly([(.25,-.43),(.69,-.39),(.93,-.07),(.93,.47),(.34,.53)])
    a.poly([(.34,-.33),(.64,-.29),(.79,-.07),(.35,-.02)],.12,True)
    a.line([(.20,-.80),(.20,.27)],.035,.35,True)
    a.wheel(-.60,.43,.17).wheel(.57,.50,.17)
    models.append(a)
    a = Shape().poly([(-.98,-.48),(.25,-.48),(.13,.10),(-.86,.10)])
    a.rect(.25,-.35,.76,.30).rect(.74,-.06,.96,.30)
    a.rect(.34,-.26,.68,-.04,.12,True)
    a.wheel(-.64,.28,.22).wheel(-.18,.28,.22).wheel(.67,.28,.22)
    models.append(a)
    a = Shape().rect(-.97,-.62,.93,.32)
    a.rect(.48,-.50,.85,-.12,.12,True)
    a.line([(.38,-.56),(.38,.26)],.035,.35,True)
    for x in [-.70,-.42,.64]: a.wheel(x,.33,.18)
    a.line([(-.90,.12),(.30,.12)],.03,.6,True)
    models.append(a)
    a = Shape().poly([(-.76,-.82),(.41,-.82),(.85,-.47),(.85,.48),(-.65,.65),(-.76,.44)])
    a.poly([(-.57,-.63),(.28,-.63),(.47,-.31),(-.57,-.31)],.12,True)
    a.line([(.43,-.72),(.43,.48)],.04,.3,True)
    a.rect(-.52,.13,.21,.33,.2,True)
    a.wheel(-.51,.61,.17).wheel(.65,.47,.17)
    models.append(a)
    return models


def ship_models():
    models = []
    a = Shape().poly([(-.98,.27),(.98,.27),(.76,.62),(-.72,.62)])
    a.rect(-.65,-.10,.60,.26).rect(-.38,-.32,.33,-.10)
    a.rect(-.16,-.58,.04,-.30)
    a.line([(-.59,.02),(.54,.02)],.05,.15,True)
    a.line([(-.87,.39),(.83,.39)],.04,.2,True)
    models.append(a)
    a = Shape().poly([(-.85,.59),(.91,.59),(.61,.85),(-.56,.85)])
    a.line([(.07,-.96),(.07,.59)],.055)
    a.poly([(-.02,-.88),(-.02,.43),(-.79,.43)])
    a.poly([(.16,-.62),(.16,.44),(.79,.44)])
    a.line([(-.70,.48),(.73,.48)],.025)
    models.append(a)
    a = Shape().poly([(-.97,.20),(.98,.12),(.65,.57),(-.75,.57)])
    a.poly([(-.43,.18),(-.24,-.21),(.31,-.21),(.52,.16)])
    a.poly([(-.29,.09),(-.16,-.13),(.24,-.13),(.37,.08)],.15,True)
    a.line([(-.82,.34),(.72,.29)],.035,.2,True)
    models.append(a)
    a = Shape().poly([(-.98,.19),(.98,.19),(.77,.58),(-.79,.58)])
    for x,y in [(-.76,-.15),(-.38,-.15),(.00,-.15)]:
        a.rect(x,y,x+.32,.18)
        a.line([(x+.08,y+.02),(x+.08,.12)],.025,.5,True)
    a.rect(.46,-.46,.75,.17).rect(.39,-.15,.85,.02)
    a.line([(-.53,-.17),(-.53,-.64),(-.13,-.35)],.035)
    models.append(a)
    a = Shape().poly([(-.62,.10),(.62,.10),(.34,.78),(0,.96),(-.34,.78)])
    a.rect(-.47,-.57,.47,.11).rect(-.30,-.77,.30,-.57)
    a.rect(-.37,-.44,.37,-.19,.15,True)
    a.line([(0,-.95),(0,-.74)],.04)
    a.line([(-.53,.27),(.53,.27)],.035,.2,True)
    models.append(a)
    a = Shape().poly([(-.98,.25),(.98,.25),(.79,.54),(-.77,.54)])
    a.rect(-.47,.02,.45,.25).rect(-.13,-.30,.22,.02)
    a.line([(.03,-.74),(.03,-.28)],.04)
    a.line([(-.27,-.50),(.27,-.50)],.035)
    a.line([(-.55,-.01),(-.77,-.14)],.05)
    a.line([(.52,.11),(.80,.00)],.05)
    a.line([(-.91,.36),(.86,.36)],.035,.3,True)
    models.append(a)
    return models


def bird_models():
    models = []
    a = Shape().ellipse(-.02,.12,.38,.60,angle=.2).ellipse(.17,-.48,.28,.28)
    a.poly([(.38,-.51),(.69,-.41),(.38,-.35)])
    a.poly([(-.26,.39),(-.49,.91),(-.13,.69)])
    a.ellipse(-.10,.14,.24,.39,angle=.2,tone=.4,detail=True)
    a.line([(-.07,.58),(-.09,.83),(-.25,.87)],.035)
    a.line([(.12,.58),(.14,.82),(.29,.87)],.035)
    a.spot(.23,-.53,.045)
    models.append(a)
    a = Shape().ellipse(-.12,.08,.60,.34,angle=-.1).ellipse(.44,-.25,.28,.29)
    a.poly([(.65,-.29),(.96,-.20),(.67,-.11)])
    a.poly([(-.58,.02),(-.98,-.23),(-.88,.21),(-.52,.27)])
    a.ellipse(-.15,.04,.39,.23,angle=-.2,tone=.4,detail=True)
    for x in [-.10,.16]: a.line([(x,.35),(x-.04,.75),(x+.12,.78)],.035)
    a.spot(.49,-.32,.045)
    models.append(a)
    a = Shape().ellipse(0,.07,.15,.53)
    a.poly([(-.08,-.13),(-.61,-.60),(-.97,-.73),(-.65,-.13),(-.15,.25)])
    a.poly([(.08,-.13),(.61,-.60),(.97,-.73),(.65,-.13),(.15,.25)])
    a.ellipse(0,-.46,.15,.18).poly([(.1,-.51),(.30,-.42),(.1,-.38)])
    a.poly([(-.08,.39),(-.23,.76),(.23,.76),(.08,.39)])
    models.append(a)
    a = Shape().ellipse(-.10,.55,.65,.69).ellipse(-.06,-.34,.49,.51)
    a.poly([(.32,-.43),(.96,-.18),(.33,-.10)])
    a.ellipse(-.27,.60,.32,.46,tone=.5,detail=True).spot(.12,-.43,.075)
    models.append(a)
    a = Shape().ellipse(-.22,.35,.60,.36)
    a.line([(.22,.24),(.39,-.09),(.33,-.48),(.51,-.67)],.16)
    a.ellipse(.54,-.68,.22,.17).poly([(.69,-.70),(.98,-.62),(.69,-.59)])
    a.poly([(-.67,.28),(-.94,.08),(-.85,.48)])
    a.line([(-.24,.62),(-.27,.92)],.025).line([(.05,.61),(.05,.92)],.025)
    a.spot(.58,-.72,.035)
    models.append(a)
    a = Shape().ellipse(-.13,.15,.60,.42,angle=-.2).ellipse(.43,-.27,.29,.30)
    a.poly([(.67,-.27),(.98,-.14),(.67,-.09)])
    a.poly([(-.58,.16),(-.96,-.49),(-.67,-.57),(-.35,.18)])
    a.ellipse(-.09,.13,.41,.24,angle=-.3,tone=.45,detail=True).spot(.48,-.33,.05)
    models.append(a)
    return models


def cat_face(wide=1.0):
    a = Shape().ellipse(0,.05,.69*wide,.63)
    a.poly([(-.61*wide,-.18),(-.58*wide,-.91),(-.17*wide,-.48)])
    a.poly([(.61*wide,-.18),(.58*wide,-.91),(.17*wide,-.48)])
    a.poly([(-.48*wide,-.35),(-.49*wide,-.73),(-.29*wide,-.46)],.3,True)
    a.poly([(.48*wide,-.35),(.49*wide,-.73),(.29*wide,-.46)],.3,True)
    a.ellipse(-.28*wide,-.11,.16,.065,tone=.08,detail=True)
    a.ellipse(.28*wide,-.11,.16,.065,tone=.08,detail=True)
    a.points += [(-.28*wide,-.11,1,0),(.28*wide,-.11,1,0)]
    a.poly([(-.10,.20),(.10,.20),(0,.31)],.05,True)
    a.line([(0,.31),(0,.42),(-.18,.48)],.025,.15,True)
    a.line([(0,.42),(.18,.48)],.025,.15,True)
    for s in [-1,1]:
        a.line([(s*.19,.30),(s*.62,.24)],.018,.25,True)
        a.line([(s*.20,.36),(s*.64,.43)],.018,.25,True)
    return a


def cat_models():
    models = [cat_face()]
    a = Shape().ellipse(0,.36,.43,.62).ellipse(0,-.48,.37,.35)
    a.poly([(-.32,-.57),(-.31,-.95),(-.08,-.74)])
    a.poly([(.32,-.57),(.31,-.95),(.08,-.74)])
    a.line([(-.21,.28),(-.22,.86)],.14).line([(.21,.28),(.22,.86)],.14)
    a.line([(.30,.80),(.71,.80),(.77,.49),(.61,.33)],.12)
    a.spot(-.14,-.51,.045).spot(.14,-.51,.045).spot(0,-.34,.04)
    models.append(a)
    a = Shape().ellipse(-.10,.0,.62,.28).ellipse(.61,-.28,.28,.28)
    a.poly([(.40,-.39),(.43,-.72),(.63,-.48)]).poly([(.63,-.47),(.79,-.70),(.82,-.31)])
    for x in [-.53,-.25,.30,.51]: a.line([(x,.16),(x-.04,.68)],.09)
    a.line([(-.64,-.02),(-.85,-.28),(-.83,-.66),(-.70,-.79)],.09)
    a.spot(.71,-.30,.04)
    models.append(a)
    a = Shape().ellipse(-.18,.25,.75,.40).ellipse(.59,.02,.33,.33)
    a.poly([(.34,-.08),(.35,-.45),(.57,-.24)]).poly([(.62,-.27),(.87,-.44),(.87,.00)])
    a.ellipse(.55,.55,.33,.10).ellipse(-.60,.56,.31,.09)
    a.spot(.52,.00,.04).spot(.73,.00,.04).spot(.66,.16,.035)
    models.append(a)
    a = cat_face(.82)
    a.ellipse(-.31,.52,.35,.40,tone=.65)
    models.append(a)
    a = Shape().ellipse(-.11,.17,.80,.59).ellipse(.40,-.03,.38,.37)
    a.poly([(.12,-.16),(.13,-.54),(.40,-.28)]).poly([(.50,-.28),(.75,-.43),(.74,-.02)])
    a.line([(-.67,.05),(-.66,.49),(-.27,.61),(.01,.47)],.12,.3,True)
    a.line([(.25,-.04),(.43,-.01)],.025,.1,True).line([(.52,-.01),(.67,-.04)],.025,.1,True)
    models.append(a)
    return models


def dog_models():
    models = []
    a = Shape().ellipse(0,-.05,.52,.66)
    a.ellipse(-.53,-.04,.23,.55,angle=-.15).ellipse(.53,-.04,.23,.55,angle=.15)
    a.ellipse(0,.36,.32,.30,tone=.65)
    a.spot(-.23,-.20,.065).spot(.23,-.20,.065).spot(0,.28,.10)
    a.line([(0,.38),(0,.51),(-.19,.56)],.025,.1,True)
    a.line([(0,.51),(.19,.56)],.025,.1,True)
    models.append(a)
    a = Shape().ellipse(0,.02,.53,.66)
    a.poly([(-.47,-.24),(-.49,-.96),(-.11,-.49)])
    a.poly([(.47,-.24),(.49,-.96),(.11,-.49)])
    a.ellipse(0,.38,.27,.36,tone=.55)
    a.spot(-.24,-.13,.06).spot(.24,-.13,.06).spot(0,.36,.105)
    models.append(a)
    a = Shape().ellipse(-.16,.00,.61,.33).ellipse(.47,-.34,.28,.35)
    a.ellipse(.73,-.20,.28,.14).ellipse(.36,-.19,.13,.30,tone=.65)
    for x in [-.61,-.32,.22,.46]: a.line([(x,.18),(x-.02,.75)],.105)
    a.line([(-.68,.02),(-.89,-.22),(-.91,-.57)],.10)
    a.spot(.55,-.41,.04).spot(.94,-.21,.05)
    models.append(a)
    a = Shape().ellipse(0,.31,.45,.65).ellipse(0,-.49,.39,.35)
    a.ellipse(-.37,-.37,.15,.33).ellipse(.37,-.37,.15,.33)
    a.line([(-.24,.31),(-.25,.90)],.16).line([(.24,.31),(.25,.90)],.16)
    a.ellipse(0,-.28,.23,.18,tone=.6)
    a.spot(-.16,-.52,.04).spot(.16,-.52,.04).spot(0,-.26,.07)
    models.append(a)
    a = Shape().ellipse(-.14,-.12,.55,.68).ellipse(.32,.31,.49,.25)
    a.ellipse(-.53,-.01,.25,.58,tone=.7).ellipse(.07,-.04,.20,.52,tone=.75)
    a.spot(.16,-.26,.06).spot(.75,.27,.085)
    a.line([(.14,.44),(.63,.45)],.035,.1,True)
    models.append(a)
    a = Shape().ellipse(0,.00,.70,.60)
    a.ellipse(-.62,-.24,.24,.34,angle=-.4).ellipse(.62,-.24,.24,.34,angle=.4)
    a.ellipse(0,.32,.43,.29,tone=.65)
    a.spot(-.30,-.14,.065).spot(.30,-.14,.065).spot(0,.22,.13)
    a.line([(-.42,.03),(-.18,.12)],.03,.3,True).line([(.42,.03),(.18,.12)],.03,.3,True)
    models.append(a)
    return models


def ungulate_side(horse=False, grazing=False, antlers=False, running=False):
    a = Shape().ellipse(-.15,-.02,.61,.35 if horse else .29)
    if grazing:
        a.poly([(.29,-.18),(.54,-.06),(.75,.55),(.57,.69),(.33,.23)])
        a.ellipse(.71,.59,.25,.14,angle=.55)
    else:
        a.poly([(.24,-.20),(.43,-.67),(.63,-.73),(.70,-.24),(.44,.16)])
        a.ellipse(.68,-.60,.29 if horse else .24,.16,angle=.35)
        a.ellipse(.51,-.84,.055,.16,angle=-.15)
        a.ellipse(.68,-.84,.055 if horse else .075,.16,angle=.35)
        a.spot(.70,-.64,.035)
    if running:
        paths = [[(-.56,.14),(-.89,.43),(-.97,.75)],
                 [(-.35,.19),(-.14,.50),(.08,.63)],
                 [(.21,.17),(.59,.35),(.90,.17)],
                 [(.41,.10),(.65,.44),(.88,.65)]]
    else:
        paths = [[(x,.13),(x+off,.51),(x+off*.8,.93)]
                 for x,off in [(-.58,-.03),(-.34,.07),(.22,-.04),(.40,.07)]]
    for path in paths: a.line(path,.085 if horse else .048)
    if horse:
        a.line([(-.71,-.07),(-.91,.18),(-.94,.61)],.13)
        if not grazing:
            a.line([(.33,-.19),(.42,-.67)],.075,.3,True)
    else:
        a.poly([(-.66,-.13),(-.88,-.29),(-.80,-.02)])
    if antlers and not grazing:
        for sign in [-1,1]:
            x = .56+sign*.07
            a.line([(x,-.78),(x+sign*.07,-1.02),(x+sign*.20,-1.10)],.025)
            a.line([(x+sign*.05,-.96),(x+sign*.16,-.94)],.02)
    return a


def deer_models():
    models = [ungulate_side(), ungulate_side(grazing=True), ungulate_side(antlers=True)]
    a = Shape().ellipse(0,.23,.31,.54).rect(-.12,-.52,.12,.10)
    a.ellipse(0,-.60,.20,.29)
    a.ellipse(-.28,-.68,.21,.09,angle=.4).ellipse(.28,-.68,.21,.09,angle=-.4)
    a.line([(-.20,.48),(-.20,.97)],.055).line([(.20,.48),(.20,.97)],.055)
    a.spot(-.095,-.65,.035).spot(.095,-.65,.035).spot(0,-.38,.06)
    models.append(a)
    a = Shape().ellipse(0,.03,.29,.61)
    a.ellipse(-.43,-.48,.30,.13,angle=.3).ellipse(.43,-.48,.30,.13,angle=-.3)
    a.spot(-.15,-.21,.045).spot(.15,-.21,.045).spot(0,.53,.09)
    for s in [-1,1]:
        a.line([(s*.13,-.44),(s*.20,-.80),(s*.39,-.99)],.035)
        a.line([(s*.19,-.74),(s*.08,-.89)],.025)
    models.append(a)
    a = Shape().ellipse(-.17,.17,.42,.56)
    a.poly([(.01,-.02),(.23,-.55),(.41,-.58),(.36,.19)])
    a.ellipse(.45,-.59,.25,.14,angle=-.2)
    a.ellipse(.33,-.78,.07,.17,angle=-.4).ellipse(.51,-.78,.07,.17,angle=.3)
    for x in [-.40,-.13,.13]: a.line([(x,.48),(x-.02,.98)],.055)
    a.ellipse(-.26,.39,.18,.22,tone=.5,detail=True).spot(.52,-.64,.035)
    models.append(a)
    return models


def horse_models():
    models = [ungulate_side(horse=True), ungulate_side(horse=True,running=True),
              ungulate_side(horse=True,grazing=True)]
    a = Shape().ellipse(0,.23,.41,.56).rect(-.18,-.50,.18,.12)
    a.ellipse(0,-.49,.24,.40)
    a.ellipse(-.16,-.89,.05,.18).ellipse(.16,-.89,.05,.18)
    for x in [-.27,.27]: a.line([(x,.48),(x,.98)],.085)
    a.spot(-.13,-.56,.035).spot(.13,-.56,.035)
    a.ellipse(0,-.22,.16,.13,tone=.4,detail=True)
    models.append(a)
    a = Shape().ellipse(0,.02,.34,.76)
    a.ellipse(-.22,-.82,.065,.20,angle=-.1).ellipse(.22,-.82,.065,.20,angle=.1)
    a.ellipse(0,.49,.33,.24,tone=.65)
    a.line([(-.26,-.56),(-.41,-.11),(-.38,.66)],.13,.3)
    a.spot(-.20,-.24,.045).spot(.20,-.24,.045)
    a.spot(-.14,.53,.055).spot(.14,.53,.055)
    models.append(a)
    a = Shape().ellipse(-.27,.12,.50,.44)
    a.poly([(.06,-.07),(.23,-.60),(.44,-.67),(.51,-.20),(.37,.27)])
    a.ellipse(.51,-.57,.29,.17,angle=.4)
    a.ellipse(.35,-.81,.05,.16).ellipse(.52,-.80,.05,.16)
    for x in [-.57,-.27,.13,.35]: a.line([(x,.30),(x-.04,.94)],.08)
    a.line([(-.69,.10),(-.86,.37),(-.87,.75)],.14)
    a.spot(.56,-.62,.035)
    models.append(a)
    return models


def frog_models():
    models = []
    a = Shape().ellipse(0,.24,.62,.44).ellipse(0,-.11,.70,.35)
    for s in [-1,1]:
        a.ellipse(s*.44,-.39,.22,.23)
        a.ellipse(s*.70,.38,.28,.34,angle=s*.3)
        a.line([(s*.44,.11),(s*.51,.65),(s*.78,.70)],.11)
        a.spot(s*.44,-.43,.095)
    a.line([(-.49,.04),(0,.13),(.49,.04)],.03,.1,True)
    models.append(a)
    a = Shape().ellipse(-.18,.14,.61,.43).ellipse(.45,-.12,.42,.29)
    a.ellipse(.48,-.35,.19,.18).ellipse(-.60,.44,.34,.29,angle=-.3)
    a.line([(-.72,.36),(-.12,.58),(-.48,.71)],.12,.5)
    a.line([(.47,.08),(.66,.59),(.94,.64)],.10)
    a.spot(.51,-.36,.08).line([(.37,.06),(.83,.04)],.03,.1,True)
    models.append(a)
    a = Shape().ellipse(0,.16,.78,.59)
    for s in [-1,1]:
        a.ellipse(s*.49,-.34,.18,.19).ellipse(s*.70,.51,.28,.27)
        a.spot(s*.49,-.36,.075)
    a.line([(-.58,.00),(0,.11),(.58,.00)],.035,.15,True)
    for x,y in [(-.34,.19),(.20,.32),(-.09,.46),(.41,.14),(-.50,.40)]:
        a.ellipse(x,y,.045,.045,tone=.35,detail=True)
    models.append(a)
    a = Shape().ellipse(0,.04,.42,.59).ellipse(0,-.47,.49,.29)
    for s in [-1,1]:
        a.line([(s*.30,-.23),(s*.75,-.12),(s*.89,-.43)],.11)
        a.line([(s*.30,.29),(s*.76,.61),(s*.37,.91)],.17)
        a.line([(s*.37,.91),(s*.70,.96)],.07)
        a.spot(s*.30,-.53,.075)
    models.append(a)
    a = Shape().ellipse(0,.17,.88,.58)
    for s in [-1,1]:
        a.ellipse(s*.52,-.36,.26,.29).spot(s*.52,-.39,.13)
    a.line([(-.66,.18),(0,.29),(.66,.18)],.045,.1,True)
    a.ellipse(0,-.01,.25,.07,tone=.6,detail=True)
    models.append(a)
    a = Shape().ellipse(-.12,.13,.63,.32).ellipse(.50,-.03,.39,.25)
    a.ellipse(.56,-.26,.16,.17).spot(.58,-.27,.065)
    a.line([(-.51,.15),(-.94,.51),(-.30,.62)],.16)
    a.line([(.36,.16),(.64,.57),(.93,.57)],.10)
    a.line([(.43,.10),(.82,.10)],.025,.1,True)
    models.append(a)
    return models


def frontend(n):
    lum = n.linear('luminance', 'image', LUMA[None,:,None,None], conv=True)
    z = n.op('contrast_normalization', 'norm', [lum])
    gray16 = n.pool('normalized_gray_16', z, (2,2))
    # Signed Sobel derivatives, followed by an unsigned, four-bin steerable HOG.
    d = np.array([-1,0,1],dtype=F32)/2
    s = np.array([1,2,1],dtype=F32)/4
    gx, gy = np.outer(s,d), np.outer(d,s)
    kernels = [gx,(gx+gy)/math.sqrt(2),gy,(gy-gx)/math.sqrt(2)]
    W = np.stack(kernels+[-a for a in kernels])[:,None]
    grad = n.linear('signed_derivatives', z, W, conv=True, padding=1)
    grad = n.relu('positive_derivatives', grad)
    A = np.zeros((4,8,1,1),F32)
    for i in range(4): A[i,i,0,0] = A[i,i+4,0,0] = 1
    absolute = n.linear('absolute_derivatives', grad, A, conv=True)
    A = np.eye(4,dtype=F32)-np.roll(np.eye(4,dtype=F32),2,axis=1)
    hog = n.linear('orientation_selectivity', absolute, A[:,:,None,None], conv=True)
    hog = n.relu('oriented_edges', hog)
    hog16 = n.pool('raw_edge_cells_16', hog, (2,2))
    hog8 = n.pool('raw_edge_cells_8', hog16, (2,2))
    # Standardized descriptors make comparison a correlation, not a vote
    # for whichever object schematic happens to have the shortest outline.
    def normalize_descriptor(name, source, side):
        a = n.op(name+'_layout','reshape',[source],shape=[1,2*side,2*side])
        a = n.op(name+'_normalization','norm',[a])
        return n.op(name,'reshape',[a],shape=[4,side,side])
    match16 = normalize_descriptor('standardized_edges_16',hog16,16)
    match8 = normalize_descriptor('standardized_edges_8',hog8,8)
    # Circular dark-center / light-surround filters for pupils, noses and wheels.
    q = np.arange(-4,5,dtype=np.float64)
    yy,xx = np.meshgrid(q,q,indexing='ij')
    B = []
    for inner,outer in [(.75,1.65),(1.3,2.7)]:
        a = np.exp(-(xx*xx+yy*yy)/(2*inner*inner)); a /= a.sum()
        b = np.exp(-(xx*xx+yy*yy)/(2*outer*outer)); b /= b.sum()
        if inner>1:
            # A wheel is a dark tyre around a brighter hub, not a dark
            # solid disc. Its outer surround is also often brighter.
            radius = np.sqrt(xx*xx+yy*yy)
            ring = np.exp(-.5*((radius-2.1)/.65)**2)
            ring /= ring.sum()
            b = .5*a+.5*b-ring
        else:
            b = b-a
        b /= np.maximum(b,0).sum()
        B.append(b)
    blobs = n.linear('circular_contrast', z, np.array(B)[:,None], conv=True,
                     stride=2,padding=4)
    blobs = n.relu('dark_round_features', blobs)
    # Long straight edges: averages along a line suppress unstructured texture.
    v = np.zeros((5,5),dtype=F32)
    v[:,1] = -.05; v[:,3] = .05
    h = v.T
    lines = n.linear('long_signed_edges', gray16, np.array([v,h,-v,-h])[:,None],
                     conv=True,padding=2)
    lines = n.relu('long_positive_edges', lines)
    A = np.array([[1,0,1,0],[0,1,0,1]],dtype=F32)[:,:,None,None]
    lines = n.linear('straight_edge_cells', lines, A,conv=True)
    # Absolute RGB differences enable saturation and neutrality tests.
    A = np.zeros((9,3,1,1),F32)
    A[:3,:,0,0] = np.eye(3)
    pairs = [(0,1),(0,2),(1,2)]
    for j,(i,k) in enumerate(pairs):
        A[3+2*j,i,0,0]=1; A[3+2*j,k,0,0]=-1
        A[4+2*j,i,0,0]=-1; A[4+2*j,k,0,0]=1
    rgb = n.linear('rgb_and_opponency', 'image', A,conv=True)
    rgb = n.relu('rgb_and_positive_opponency',rgb)
    # A colour name is an intersection of simple physical RGB conditions.
    # For example, red must exceed green, rather than merely exceed the
    # average of green and blue (which would also fire strongly on brown).
    conditions = [
        [([5,-5,0],-.60),([4,0,0],-1.20)],
        [([-6,6,0],-.10),([0,6,-6],-.10)],
        [([-5,0,5],-.10),([0,-4,4],.10)],
        [([4,0,-4],-.20),([0,4,-4],.20)],
        [([4,0,0],-2.40),([0,4,0],-2.40),([0,0,4],-2.40)],
        [((-3*LUMA).tolist(),1.20)],
        [([0,0,0, -3,-3,-3,-3,-3,-3],1)],
        [(LUMA.tolist(),0)],
        [([0,0,0, 1.5,1.5,1.5,1.5,1.5,1.5],0)],
    ]
    C = np.zeros((27,9),F32)
    b = np.zeros(27,F32)
    for j,clauses in enumerate(conditions):
        for p in range(3):
            coefficients,offset = clauses[min(p,len(clauses)-1)]
            C[j*3+p,:len(coefficients)]=coefficients
            b[j*3+p]=offset
    C = np.concatenate([C,C],axis=0)
    bias = np.concatenate([b,b-1])
    colors = n.linear('color_thresholds','rgb_and_positive_opponency',C[:,:,None,None],bias,conv=True)
    colors = n.relu('color_ramps',colors)
    A = np.concatenate([-np.eye(27),np.eye(27)],axis=1)
    colors = n.linear('negative_bounded_color_clauses',colors,A[:,:,None,None],conv=True)
    colors = n.op('color_clause_layout','reshape',[colors],shape=[9,3,1024])
    colors = n.pool('color_clause_intersections',colors,(3,1),maximum=True)
    colors = n.op('color_image_layout','reshape',[colors],shape=[9,32,32])
    colors = n.linear('bounded_color_memberships',colors,-np.eye(9)[:,:,None,None],conv=True)
    colors8 = n.pool('color_cells_8',colors,(4,4))
    # Brightness and colour-opponent foreground maps. Instance normalization
    # is applied at inference, not estimated from any collection of images.
    P = np.zeros((4,9),F32)
    P[0,:3]=LUMA
    P[1,:3]=[1,-1,0]
    P[2,:3]=[-.5,-.5,1]
    P[3,3:]=.5
    appearance = n.linear('foreground_opponent_maps','rgb_and_positive_opponency',P[:,:,None,None],conv=True)
    appearance = n.pool('foreground_maps_16',appearance,(2,2))
    appearance = n.op('normalized_foreground_maps','norm',[appearance])
    return hog16, gray16, blobs, lines, colors8, match16, match8, appearance


def normalized_region(size, which):
    y,x = np.mgrid[:size,:size]
    u,v = (x+.5)/size,(y+.5)/size
    regions = {
        'all':np.ones((size,size),bool),
        'top':v<.375,
        'bottom':v>.625,
        'center':(u>.25)&(u<.75)&(v>.25)&(v<.75),
        'border':(u<.25)|(u>.75)|(v<.25)|(v>.75),
        'lower_center':(u>.125)&(u<.875)&(v>.5)&(v<.875),
        'upper_center':(u>.25)&(u<.75)&(v>.125)&(v<.5),
    }
    a = regions[which].astype(np.float64)
    return a/a.sum()


class Cue:
    def __init__(self, name, bias=0):
        self.name = name
        self.c = np.zeros((9,8,8),F32)
        self.h = np.zeros((4,16,16),F32)
        self.l = np.zeros((2,16,16),F32)
        self.bias = bias

    def color(self, name, region='all', weight=1):
        self.c[COLORS.index(name)] += weight*normalized_region(8,region)
        return self

    def edge(self, kind='all', region='all', weight=1):
        directions = {'all':[0,1,2,3], 'horizontal':[2], 'vertical':[0],
                      'axes':[0,2], 'diagonal':[1,3]}[kind]
        for j in directions: self.h[j] += weight*normalized_region(16,region)
        return self

    def line(self, kind, region='all',weight=1):
        self.l[0 if kind=='vertical' else 1] += weight*normalized_region(16,region)
        return self


def context_scores(n, hog, lines, colors):
    # Each hidden unit is a named, manually specified piecewise-linear cue.
    cues = []
    def add(c):
        cues.append(c)
    add(Cue('water',-.04).color('blue','bottom').color('green','bottom',-.35))
    add(Cue('blue_below',-.025).color('blue','bottom').color('blue','top',-.7))
    add(Cue('open_sky',-.12).color('blue','top').color('white','top',.7).edge('all','top',-.7))
    add(Cue('blue_scene',-.08).color('blue').edge('all','all',-.15))
    add(Cue('white_background',-.10).color('white','border').edge('all','border',-.4))
    add(Cue('green_object',-.02).color('green','center').color('green','border',-.65))
    add(Cue('vegetation',-.05).color('green'))
    add(Cue('grass',-.025).color('green','bottom').color('green','top',-.35))
    add(Cue('green_background',-.02).color('green','border').color('green','center',-.45))
    add(Cue('warm_object',-.025).color('warm','center').color('warm','border',-.5))
    add(Cue('warm_scene',-.07).color('warm'))
    add(Cue('red_body',-.035).color('red','center').color('red','border',-.65))
    add(Cue('blue_body',-.025).color('blue','center').color('blue','border',-.65))
    add(Cue('white_body',-.035).color('white','center').color('white','border',-.5))
    add(Cue('dark_body',-.025).color('dark','center').color('dark','border',-.5))
    add(Cue('neutral_body',-.08).color('neutral','center').color('neutral','border',-.55))
    add(Cue('saturated_body',-.06).color('saturation','center').color('saturation','border',-.65))
    add(Cue('central_texture',-.015).edge('all','center').edge('all','border',-.65))
    add(Cue('background_texture',-.015).edge('all','border').edge('all','center',-.65))
    add(Cue('water_lines',-.005).line('horizontal','bottom').line('vertical','bottom',-.6))
    add(Cue('legs',-.005).edge('vertical','lower_center').edge('horizontal','lower_center',-.6).edge('diagonal','lower_center',-.4))
    add(Cue('flat_body',-.005).edge('horizontal','center').edge('vertical','center',-.5).edge('diagonal','center',-.4))
    add(Cue('box_edges',-.01).edge('axes','center').edge('diagonal','center',-.9))
    add(Cue('oblique_edges',-.01).edge('diagonal','center').edge('axes','center',-.65))
    add(Cue('long_lines',-.005).line('horizontal','center').line('vertical','center').edge('all','center',-.12))
    add(Cue('dark_road',-.04).color('dark','bottom').color('green','bottom',-.4))
    add(Cue('smooth_green_skin',-.03).color('green','center',1.5).color('green','border',-.6).edge('all','center',-.35))
    add(Cue('smooth_olive_skin',-.07).color('warm','center',.5).color('green','center').color('light','center',.06).edge('all','center',-.4))
    add(Cue('grazing_animal',-.06).color('warm','center',.75).color('green','bottom',.65).edge('vertical','lower_center',.8).edge('horizontal','lower_center',-.4))
    add(Cue('small_wildlife',-.02).color('green','border',.35).edge('all','center').edge('all','border',-.8))
    add(Cue('sky_silhouette',-.03).color('white','border',.8).color('blue','border',.5).color('dark','center',.2).edge('all','border',-.5))
    add(Cue('upright_vehicle',-.02).line('vertical','center').edge('vertical','center',.4).color('warm','center',-.15).color('green','center',-.2))
    c = n.linear('context_color_terms',colors,np.stack([q.c.ravel() for q in cues]),[q.bias for q in cues])
    h = n.linear('context_edge_terms',hog,np.stack([q.h.ravel() for q in cues]))
    l = n.linear('context_line_terms',lines,np.stack([q.l.ravel() for q in cues]))
    a = n.op('context_terms','add',[c,h,l])
    a = n.relu('context_evidence',a)
    # Convert a collection of weak, nonnegative cues into a comparable
    # image-specific evidence pattern. This is an inference operation;
    # no population mean, frequency, or class prior is measured or stored.
    a = n.op('context_evidence_layout','reshape',[a],shape=[1,1,len(cues)])
    a = n.op('context_evidence_standardization','norm',[a])
    a = n.relu('distinctive_context_evidence',a)
    a = n.op('context_evidence_vector','reshape',[a],shape=[len(cues)])
    # Rule-table entries express qualitative habitat/material expectations.
    # Their strengths are chosen by hand, never estimated from labels.
    rules = [
        {'open_sky':.55,'blue_scene':.30,'white_background':.40,'water':-.25,
         'vegetation':-.25,'warm_object':-.18,'central_texture':-.15,'flat_body':.15,'long_lines':.10},
        {'red_body':.26,'blue_body':.20,'neutral_body':.12,'box_edges':.22,
         'flat_body':.24,'long_lines':.23,'dark_road':.12,'green_object':-.25,'central_texture':-.10},
        {'green_background':.18,'saturated_body':.13,'dark_body':.07,'oblique_edges':.12,
         'box_edges':-.12,'water':-.08,'long_lines':-.10},
        {'warm_object':.25,'warm_scene':.13,'central_texture':.15,'neutral_body':.09,
         'oblique_edges':.07,'grass':-.18,'open_sky':-.17,'water':-.25,'green_object':-.13},
        {'green_background':.28,'grass':.20,'warm_object':.24,'legs':.40,
         'background_texture':.16,'green_object':-.14,'water':-.18,'blue_scene':-.13},
        {'warm_object':.27,'warm_scene':.08,'central_texture':.15,'neutral_body':.13,
         'dark_body':.07,'water':-.23,'open_sky':-.16,'green_object':-.14},
        {'green_object':.70,'vegetation':.21,'warm_scene':.08,'water':-.15,
         'open_sky':-.22,'box_edges':-.12,'legs':-.25,'long_lines':-.15},
        {'grass':.30,'green_background':.16,'warm_object':.23,'legs':.36,'dark_body':.09,
         'background_texture':-.06,'water':-.20,'blue_scene':-.12},
        {'water':.75,'blue_below':.45,'blue_scene':.18,'water_lines':.35,'open_sky':.12,
         'green_object':-.23,'warm_object':-.13,'vegetation':-.12},
        {'white_body':.28,'neutral_body':.15,'red_body':.13,'blue_body':.08,'box_edges':.28,
         'long_lines':.30,'open_sky':.08,'green_object':-.22,'central_texture':-.08},
    ]
    # Conjunction cues distinguish colour on an object from its habitat.
    rules[0].update(sky_silhouette=.40)
    rules[1].update(upright_vehicle=.12,smooth_green_skin=-.15)
    rules[2].update(small_wildlife=.45,green_background=.40,sky_silhouette=.05)
    rules[3].update(smooth_green_skin=-.30,grazing_animal=-.15)
    rules[4].update(grazing_animal=.55,green_background=.45)
    rules[5].update(smooth_green_skin=-.25)
    rules[6].update(smooth_green_skin=1.2,smooth_olive_skin=.7,green_object=.9,vegetation=.35)
    rules[7].update(grazing_animal=.55)
    rules[8].update(sky_silhouette=.05)
    rules[9].update(upright_vehicle=.55,smooth_green_skin=-.2)
    W = np.zeros((10,len(cues)),F32)
    for k,rule in enumerate(rules):
        for j,q in enumerate(cues): W[k,j]=rule.get(q.name,0)
    return n.linear('habitat_and_material_scores',a,.16*W)


def shape_bank(n, hog, coarse_hog, gray, blobs, colors, appearance):
    all_models = [airplane_models(),car_models(),bird_models(),cat_models(),
                  deer_models(),dog_models(),frog_models(),horse_models(),
                  ship_models(),truck_models()]
    assert all(len(models)==6 for models in all_models)
    # Equal numbers of poses for every class.  Mirror, small rotations, sizes,
    # and displacements are geometric invariances, not a parameter search.
    positions = [(0,0),(-4,-4),(4,-4),(-4,4),(4,4)]
    scales = [8.0,11.0,14.0,18.0]
    angles = [-.18,0,.18]
    mirrors = [-1,1]
    per_class = 6*len(positions)*len(scales)*len(angles)*len(mirrors)
    total = 10*per_class
    W_h = np.empty((total,4*16*16),F32)
    W_coarse = np.empty((total,4*8*8),F32)
    W_segment = np.empty((total,16*16),F32)
    W_g = np.empty((total*2,16*16),F32)
    W_b = np.zeros((total,2*16*16),F32)
    W_c = np.zeros((total,9*8*8),F32)
    bias = np.empty(total,F32)
    color_rules = [
        {'blue':(-.015,.055),'white':(.015,.035),'neutral':(.025,0),'green':(-.065,0),'warm':(-.025,0)},
        {'red':(.065,-.025),'blue':(.045,-.015),'neutral':(.035,0),'green':(-.045,0)},
        {'warm':(.035,-.010),'red':(.020,-.005),'green':(0,.025)},
        {'warm':(.040,-.010),'neutral':(.025,0),'dark':(.015,0),'green':(-.025,0),'blue':(-.020,0)},
        {'warm':(.065,-.020),'green':(-.015,.035),'blue':(-.020,0)},
        {'warm':(.050,-.015),'neutral':(.025,0),'dark':(.020,0),'green':(-.020,0),'blue':(-.025,0)},
        {'green':(.150,-.090),'warm':(.035,0),'blue':(-.035,0)},
        {'warm':(.055,-.015),'green':(-.010,.035),'dark':(.020,0)},
        {'blue':(-.015,.055),'white':(.040,0),'green':(-.040,0)},
        {'white':(.075,-.020),'neutral':(.035,0),'red':(.030,-.010),'green':(-.045,0)},
    ]
    row = 0
    for k,models in enumerate(all_models):
        print('Constructing geometric templates for class',k,flush=True)
        for model in models:
            for scale in scales:
                for dx,dy in positions:
                    for angle in angles:
                        for mirror in mirrors:
                            e,m16,m8,b = render(model,scale,dx,dy,angle,mirror)
                            # Orientation selectivity rejects isotropic clutter.
                            # A spatially/orientationally uniform edge field has
                            # exactly zero dot product with these weights.
                            e /= e.sum()+1e-12
                            coverage = e.sum(0,keepdims=True)
                            # Mean-centered, unit-length template correlation:
                            # normalization uses only the synthetic drawing.
                            fine = e-e.mean()
                            coarse = e.reshape(4,8,2,8,2).mean((2,4))
                            coarse -= coarse.mean()
                            W_h[row] = (.60*fine/(np.linalg.norm(fine)+1e-12)/math.sqrt(fine.size)).ravel()
                            W_coarse[row] = (.40*coarse/(np.linalg.norm(coarse)+1e-12)/math.sqrt(coarse.size)).ravel()
                            # A silhouette correlation penalizes occupied
                            # pixels outside the expected object as well as
                            # rewarding the expected body, head and limbs.
                            region = blur(m16,.45)
                            region -= region.mean()
                            W_segment[row] = (region/(np.linalg.norm(region)+1e-12)/16).ravel()
                            # Foreground/background luminance contrast is
                            # polarity invariant; max(+v,-v) is implemented
                            # below with a two-element max pool.
                            fg = m16/(m16.sum()+1e-12)
                            bg = (1-m16)/(np.sum(1-m16)+1e-12)
                            wg = .105*(fg-bg)
                            W_g[2*row] = wg.ravel()
                            W_g[2*row+1] = -wg.ravel()
                            if b.sum()>0:
                                b /= b.sum()
                                # Roundness evidence must exceed the image's
                                # average blob response, not merely be present.
                                for j in range(2):
                                    mass=b[j].sum()
                                    b[j] -= mass/(16*16)
                                W_b[row] = (.065*b).ravel()
                            fg8 = m8/(m8.sum()+1e-12)
                            bg8 = (1-m8)/(np.sum(1-m8)+1e-12)
                            wc = np.zeros((9,8,8),F32)
                            for color,(fcoef,bcoef) in color_rules[k].items():
                                wc[COLORS.index(color)] = fcoef*fg8+bcoef*bg8
                            W_c[row] = wc.ravel()
                            # Small contours have higher accidental-match
                            # variance. This analytic, label-free penalty is
                            # determined only by the geometric template size.
                            effective = 1/(np.sum(coverage*coverage)+1e-12)
                            bias[row] = -.12/math.sqrt(effective)
                            row += 1
    assert row==total
    a = n.linear('geometric_edge_matches',hog,W_h,bias)
    coarse = n.linear('coarse_geometric_matches',coarse_hog,W_coarse)
    a = n.op('multiscale_geometric_matches','add',[a,coarse])
    # One shared matrix compares each of four input-derived foreground maps
    # to each silhouette, with no image-dependent numbers in the weights.
    projection = np.zeros((total,4,4,256),F32)
    for channel in range(4): projection[:,channel,channel]=W_segment
    seg = n.linear('silhouette_correlations',appearance,projection.reshape(total*4,1024))
    # |v| = relu(2v)-v. Pointwise operations on a flattened single
    # channel avoid any quadratic diagonal matrix of weights.
    flat = n.op('correlation_flat','reshape',[seg],shape=[1,1,total*4])
    twice = n.linear('twice_correlation',flat,np.full((1,1,1,1),2,F32),conv=True)
    twice = n.relu('positive_twice_correlation',twice)
    minus = n.linear('negative_correlation_flat',flat,-np.ones((1,1,1,1),F32),conv=True)
    absseg = n.op('absolute_correlation_flat','add',[twice,minus])
    absseg = n.op('absolute_correlation_rows','reshape',[absseg],shape=[total,1,4])
    absseg = n.pool('best_foreground_channel',absseg,(1,4),maximum=True)
    absseg = n.op('foreground_match_vector','reshape',[absseg],shape=[1,1,total])
    absseg = n.linear('foreground_match_strength',absseg,np.full((1,1,1,1),.30,F32),conv=True)
    absseg = n.op('foreground_match_scores','reshape',[absseg],shape=[total])
    a = n.op('contours_and_foregrounds','add',[a,absseg])
    g = n.linear('silhouette_contrast_pairs',gray,W_g)
    g = n.op('silhouette_pair_layout','reshape',[g],shape=[total,1,2])
    g = n.pool('polarity_invariant_contrast',g,(1,2),maximum=True)
    g = n.op('silhouette_contrast','reshape',[g],shape=[total])
    b = n.linear('wheel_eye_nose_matches',blobs,W_b)
    c = n.linear('shape_conditioned_colors',colors,W_c)
    a = n.op('pose_evidence','add',[a,g,b,c])
    a = n.op('class_pose_layout','reshape',[a],shape=[10,1,per_class])
    a = n.pool('best_geometric_explanation',a,(1,per_class),maximum=True)
    return n.op('geometric_class_scores','reshape',[a],shape=[10])


def main():
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python3 build.py OUTPUT_DIRECTORY')
    out = Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=True)
    n = Network()
    hog,gray,blobs,lines,colors,match16,match8,appearance = frontend(n)
    context = context_scores(n,hog,lines,colors)
    geometry = shape_bank(n,match16,match8,gray,blobs,colors,appearance)
    n.op('class_scores','add',[geometry,context])
    count = sum(w.size for w in n.weights.values())
    assert count <= 120_000_000
    assert all(np.all(np.isfinite(w)) for w in n.weights.values())
    (out/'network.json').write_text(json.dumps({'layers':n.layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**n.weights)
    print('Wrote',len(n.layers),'layers and',count,'float32 weights.',flush=True)


if __name__=='__main__':
    main()
