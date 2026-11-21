#!/usr/bin/env python3
"""Data-free, formula-built classifier. No file is opened for reading.

The numerical choices below are hand-written geometric and photometric priors,
not estimates from a dataset. There is no optimizer, learning loop, random search,
external model, image bank, or dependence on a previous run.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def layer(self, name, op, inputs, **settings):
        if isinstance(inputs, str):
            inputs = [inputs]
        self.layers.append(dict(name=name, op=op, inputs=inputs, **settings))
        return name

    def conv(self, name, source, w, b=None, padding=0, stride=1):
        w = np.asarray(w, dtype=np.float32)
        if b is None:
            b = np.zeros(w.shape[0], np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.asarray(b, dtype=np.float32)
        return self.layer(name, 'conv', source, out=w.shape[0], kernel=list(w.shape[2:]),
                          stride=stride, padding=padding)

    def dense(self, name, source, w, b=None):
        w = np.asarray(w, dtype=np.float32)
        if b is None:
            b = np.zeros(w.shape[0], np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.asarray(b, dtype=np.float32)
        return self.layer(name, 'dense', source, out=w.shape[0])

    def relu(self, name, source):
        return self.layer(name, 'relu', source)

    def pool(self, name, source, size, stride=None, maximum=False):
        if isinstance(size, int):
            size = [size, size]
        if stride is None:
            stride = size
        if isinstance(stride, int):
            stride = [stride, stride]
        return self.layer(name, 'maxpool' if maximum else 'avgpool', source,
                          size=list(size), stride=list(stride))


LUMA = np.array([0.30, 0.59, 0.11], dtype=np.float64)


def derivatives():
    # Four unsigned orientations, with positive and negative phases later.
    dx = np.outer([1., 2., 1.], [-1., 0., 1.]) / 8.
    dy = dx.T
    return np.array([dx, (dx + dy) / math.sqrt(2), dy,
                     (dy - dx) / math.sqrt(2)])


def add_edges(n, source, channels, prefix, long=False):
    if long:
        y, x = np.mgrid[-3:4, -3:4]
        kernels = []
        for a in [0, math.pi/4, math.pi/2, 3*math.pi/4]:
            u, v = x*math.cos(a) + y*math.sin(a), -x*math.sin(a) + y*math.cos(a)
            k = u * np.exp(-0.5*((u/0.8)**2 + (v/2.0)**2))
            k -= k.mean()
            k /= np.abs(k).sum()
            kernels.append(k)
        kernels = np.array(kernels)
    else:
        kernels = derivatives()
    ksize = kernels.shape[-1]
    w = np.zeros((4*2*channels, channels, ksize, ksize))
    for a in range(4):
        for c in range(channels):
            for s, sign in enumerate([1., -1.]):
                w[(a*channels+c)*2+s, c] = sign*kernels[a]
    # Derivatives are evaluated only on complete, observed neighborhoods.
    # Padding the *energy* with zero afterwards cannot create a false image
    # frame, unlike differentiating a photograph against a black exterior.
    signed = n.conv(prefix + '_signed', source, w, padding=0)
    positive = n.relu(prefix + '_phases', signed)
    combine = np.zeros((4, 8*channels, 1, 1))
    for a in range(4):
        combine[a, a*2*channels:(a+1)*2*channels, 0, 0] = 1./channels
    return n.conv(prefix + '_energy', positive, combine, padding=ksize//2)


def add_colors(n, source):
    # Soft intersections of RGB half-spaces. Six constraints per color.
    # All intensities and tolerances are specified in the physical [0,1] range.
    R, G, B = np.eye(3)
    lum = LUMA
    constant = (np.zeros(3), 1.)
    def greater(v, threshold, width):
        return (v/width, -threshold/width)
    def less(v, threshold, width):
        return (-v/width, threshold/width)
    neutral = [greater(R-G, -.18, .18), less(R-G, .18, .18),
               greater(R-B, -.18, .18), less(R-B, .18, .18),
               greater(G-B, -.18, .18), less(G-B, .18, .18)]
    definitions = {
        'blue': [greater(B-R, .015, .18), greater(B-G, -.025, .16),
                 greater(lum, .15, .30)],
        'green': [greater(G-R, .005, .14), greater(G-B, .015, .18),
                  greater(lum, .10, .30)],
        'warm': [greater(R-B, .035, .20), greater(G-B, .005, .18),
                 greater(R-G, -.025, .20)],
        'red': [greater(R-G, .06, .24), greater(R-B, .045, .24),
                greater(R, .20, .40)],
        'neutral': neutral,
        'dark': [less(lum, .40, .30)],
        'bright': [greater(lum, .60, .30)],
        'pale': [greater(lum, .50, .35), greater(R-B, -.16, .20),
                 less(R-B, .16, .20), greater(R-G, -.16, .20),
                 less(R-G, .16, .20)],
        'yellow': [greater(R-B, .10, .24), greater(G-B, .08, .24),
                   greater(lum, .30, .35)],
        'brown': [greater(R-B, .035, .18), greater(G-B, .005, .16),
                  greater(R-G, -.02, .20), less(lum, .65, .35)],
    }
    names = list(definitions)
    rows, bias = [], []
    for name in names:
        criteria = definitions[name]
        # Min(1, constraints), followed by max(0, .), is membership.
        # Neutral has six constraints already, all <=1 jointly.
        criteria = criteria + [constant]*(6-len(criteria))
        for w, b in criteria:
            rows.append(-w)
            bias.append(-b)
    affine = n.conv('color_constraints', source, np.array(rows)[:, :, None, None], bias)
    arranged = n.layer('color_constraint_groups', 'reshape', affine, shape=[len(names), 6, 256])
    worst = n.pool('color_worst_constraint', arranged, [6, 1], maximum=True)
    maps = n.layer('color_map_shape', 'reshape', worst, shape=[len(names), 16, 16])
    maps = n.conv('color_minimum', maps, -np.eye(len(names))[:, :, None, None])
    return n.relu('color_memberships', maps), names


# ---- Generic mathematical object models -----------------------------------
# A coordinate here is a parameter of a generic ellipse, segment, or polygon.
# None is a measurement, outline, crop, or trace of any supplied photograph.
class Drawing:
    def __init__(self, x, y, softness):
        self.x, self.y, self.softness = x, y, softness
        self.mask = np.zeros_like(x)
        self.ink = np.zeros_like(x)

    def ellipse(self, cx, cy, rx, ry, angle=0):
        a = math.radians(angle)
        x, y = self.x-cx, self.y-cy
        u, v = x*math.cos(a)+y*math.sin(a), -x*math.sin(a)+y*math.cos(a)
        distance = (np.sqrt((u/rx)**2+(v/ry)**2)-1)*min(rx, ry)
        return np.clip(.5-distance/self.softness, 0., 1.)

    def polygon(self, vertices):
        v = np.asarray(vertices, dtype=np.float64)
        following = np.roll(v, -1, axis=0)
        orientation = np.sign(np.sum(v[:, 0]*following[:, 1]-following[:, 0]*v[:, 1]))
        d = np.full_like(self.x, np.inf)
        for p, q in zip(v, following):
            dx, dy = q-p
            signed = orientation*(dx*(self.y-p[1])-dy*(self.x-p[0])) / math.hypot(dx, dy)
            d = np.minimum(d, signed)
        return np.clip(.5+d/self.softness, 0., 1.)

    def line(self, ax, ay, bx, by, radius):
        dx, dy = bx-ax, by-ay
        t = np.clip(((self.x-ax)*dx+(self.y-ay)*dy)/(dx*dx+dy*dy+1e-12), 0., 1.)
        d = np.sqrt((self.x-ax-t*dx)**2+(self.y-ay-t*dy)**2)-radius
        return np.clip(.5-d/self.softness, 0., 1.)

    def rect(self, x0, y0, x1, y1):
        return self.polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])

    def fill(self, *parts):
        for part in parts:
            self.mask = np.maximum(self.mask, part)

    def dark(self, part, amount=1.):
        self.ink -= amount*part

    def light(self, part, amount=1.):
        self.ink += amount*part


def animal_side(d, kind, pose):
    # A torso, a rising neck, a head, four legs, ears and a tail.
    slender = kind == 'deer'
    small = kind in ('cat', 'dog')
    if small:
        d.fill(d.ellipse(-.08, .02, .51, .24), d.line(.31,-.02,.49,-.30,.15),
               d.ellipse(.53,-.32,.22,.22))
        if kind == 'cat':
            d.fill(d.polygon([(.38,-.44),(.39,-.72),(.55,-.47)]),
                   d.polygon([(.57,-.47),(.70,-.66),(.70,-.36)]),
                   d.line(-.52,.02,-.77,-.30,.065))
        else:
            d.fill(d.ellipse(.39,-.24,.13,.23, -25), d.ellipse(.73,-.24,.17,.09),
                   d.line(-.54,.02,-.78,-.13,.065))
        for x, lean in [(-.44,-.03),(-.23,.07),(.17,-.02),(.34,.08)]:
            d.fill(d.line(x,.15,x+lean,.65,.055))
        d.dark(d.ellipse(.60,-.36,.035,.035), .7)
        d.dark(d.ellipse(.80,-.24,.045,.04), .8)
        return
    body_rx, body_ry = (.44,.22) if slender else (.53,.28)
    d.fill(d.ellipse(-.13,-.04,body_rx,body_ry),
           d.line(.21,-.10,.39,-.53,.10 if slender else .16),
           d.ellipse(.45,-.59,.18,.10 if slender else .14, 18))
    for x, lean in [(-.47,-.04),(-.25,.07),(.12,-.04),(.30,.06)]:
        if pose == 1:
            lean += .18 if x < 0 else -.17
        d.fill(d.line(x,.11,x+lean,.72,.033 if slender else .052))
        d.fill(d.line(x+lean,.72,x+lean+.055,.73,.035 if slender else .047))
    d.fill(d.ellipse(.36,-.76,.055,.13,-20), d.ellipse(.51,-.75,.05,.11,25))
    if slender:
        d.fill(d.line(-.53,-.07,-.68,-.17,.04))
        if pose == 2:
            for shift in [-.06,.06]:
                d.fill(d.line(.39+shift,-.68,.34+shift,-.94,.018),
                       d.line(.35+shift,-.85,.25+shift,-.90,.015),
                       d.line(.36+shift,-.82,.45+shift,-.90,.015))
    else:
        d.fill(d.line(-.59,-.10,-.68,.39,.065))
        d.dark(d.line(.23,-.24,.32,-.64,.04), .6)
    d.dark(d.ellipse(.51,-.62,.026,.026), .8)


def face(d, kind, model):
    cat = kind == 'cat'
    # Six views are generic face, bust, sitting, side, reclining, profile.
    if model == 3:
        animal_side(d, kind, 0)
        return
    if model == 4:
        d.fill(d.ellipse(-.16,.26,.60,.35), d.ellipse(.43,-.08,.31,.30))
        if cat:
            d.fill(d.polygon([(.19,-.23),(.24,-.65),(.44,-.32)]),
                   d.polygon([(.49,-.34),(.68,-.59),(.72,-.10)]))
        else:
            d.fill(d.ellipse(.24,-.05,.16,.32,-25), d.ellipse(.67,-.04,.13,.28,20))
        d.dark(d.ellipse(.34,-.10,.045,.045))
        d.dark(d.ellipse(.54,-.10,.045,.045))
        d.dark(d.ellipse(.46,.06,.06,.04))
        return
    if model == 5:
        d.fill(d.ellipse(-.08,.02,.46,.58), d.ellipse(.38,.13,.27,.18))
        if cat:
            d.fill(d.polygon([(-.38,-.31),(-.31,-.89),(-.07,-.47)]),
                   d.polygon([(.05,-.48),(.27,-.78),(.31,-.27)]))
        else:
            d.fill(d.ellipse(-.40,.02,.20,.46,-10))
        d.dark(d.ellipse(.16,-.13,.055,.04))
        d.dark(d.ellipse(.60,.08,.07,.055))
        return
    cy = -.28 if model in (1,2) else 0.
    scale = .72 if model in (1,2) else 1.
    def e(x,y,rx,ry,angle=0):
        return d.ellipse(x*scale,cy+y*scale,rx*scale,ry*scale,angle)
    if model in (1,2):
        d.fill(d.ellipse(0,.40,.43 if model==1 else .36,.45))
        d.fill(d.line(-.20,.36,-.20,.83,.085),d.line(.20,.36,.20,.83,.085))
    d.fill(e(0,0,.56 if cat else .48,.53 if cat else .58))
    if cat:
        d.fill(d.polygon([(-.52*scale,cy-.20*scale),(-.48*scale,cy-.84*scale),(-.14*scale,cy-.44*scale)]),
               d.polygon([(.14*scale,cy-.44*scale),(.48*scale,cy-.84*scale),(.52*scale,cy-.20*scale)]))
        d.dark(e(-.23,-.08,.10,.045), .9)
        d.dark(e(.23,-.08,.10,.045), .9)
        d.dark(e(0,.18,.065,.05), .8)
        d.light(e(-.11,.25,.13,.10), .3)
        d.light(e(.11,.25,.13,.10), .3)
    else:
        if model == 2:
            d.fill(d.polygon([(-.45,cy-.10),(-.41,cy-.65),(-.15,cy-.33)]),
                   d.polygon([(.15,cy-.33),(.41,cy-.65),(.45,cy-.10)]))
        else:
            d.fill(e(-.50,-.02,.20,.43,-12),e(.50,-.02,.20,.43,12))
        d.dark(e(-.21,-.12,.065,.055), .9)
        d.dark(e(.21,-.12,.065,.055), .9)
        d.light(e(0,.25,.28,.25), .3)
        d.dark(e(0,.18,.12,.085))
        d.dark(e(0,.40,.15,.035), .5)


def draw_object(k, model, x, y, softness):
    d = Drawing(x,y,softness)
    if k == 0:  # aircraft: top, side, front, swept wings, oblique top, jet side
        if model in (0,3,4):
            d.fill(d.ellipse(0,0,.09,.87),
                   d.polygon([(-.09,-.20),(-.86,.25),(-.80,.43),(0,.15),(.80,.43),(.86,.25),(.09,-.20)]))
            # The wing polygon is split into convex panels, avoiding concavity.
            d.fill(d.polygon([(-.05,-.20),(-.88,.23),(-.80,.40),(-.03,.15)]),
                   d.polygon([(.05,-.20),(.88,.23),(.80,.40),(.03,.15)]),
                   d.polygon([(-.04,.45),(-.36,.72),(-.34,.80),(.34,.80),(.36,.72),(.04,.45)]))
            d.dark(d.ellipse(0,-.54,.045,.14), .5)
        elif model in (1,5):
            d.fill(d.ellipse(0,.03,.88,.105),
                   d.polygon([(-.72,-.01),(-.69,-.45),(-.49,-.02)]),
                   d.polygon([(-.11,.06),(-.34,.50),(.31,.09)]),
                   d.polygon([(-.12,-.02),(.22,-.38),(.35,.01)]))
            for xx in [-.35,-.20,-.05,.10,.25,.40]:
                d.dark(d.ellipse(xx,0,.022,.024), .7)
        else:
            d.fill(d.ellipse(0,.05,.15,.19), d.line(-.91,.13,.91,.13,.045),
                   d.polygon([(-.035,.02),(0,-.59),(.035,.02)]),
                   d.ellipse(-.38,.20,.07,.09),d.ellipse(.38,.20,.07,.09))
            d.dark(d.ellipse(0,-.02,.095,.035), .7)
    elif k in (1,9):
        truck = k == 9
        if model in (0,1,4):
            if truck:
                d.fill(d.rect(-.83,-.57,.22,.25),
                       d.polygon([(.22,-.30),(.57,-.30),(.81,-.02),(.81,.30),(.22,.30)]))
                d.dark(d.polygon([(.32,-.24),(.55,-.24),(.69,-.04),(.32,-.04)]), .7)
                d.dark(d.line(.20,-.52,.20,.25,.018), .35)
            else:
                d.fill(d.ellipse(0,.13,.82,.21),
                       d.polygon([(-.56,.04),(-.34,-.42),(.28,-.42),(.61,.04)]))
                d.dark(d.polygon([(-.40,-.04),(-.27,-.34),(.20,-.34),(.43,-.04)]), .65)
                d.light(d.line(0,-.32,.02,-.04,.02), .6)
            for xx in [-.53,.53]:
                wheel = d.ellipse(xx,.29,.15,.16)
                d.fill(wheel)
                d.dark(wheel)
                d.light(d.ellipse(xx,.29,.055,.055), .5)
            d.light(d.ellipse(.75,.08,.055,.04), .5)
            if model == 4:
                d.dark(d.line(-.72,.10,.60,.10,.025), .25)
        else:
            height = .70 if truck else .43
            d.fill(d.polygon([(-.65,-height*.42),(-.47,-height),(.47,-height),(.65,-height*.42),(.69,.40),(-.69,.40)]),
                   d.rect(-.65,.26,-.43,.61),d.rect(.43,.26,.65,.61))
            d.dark(d.polygon([(-.48,-height*.40),(-.36,-height*.86),(.36,-height*.86),(.48,-height*.40)]), .8)
            d.dark(d.rect(-.39,.20,.39,.32), .65)
            d.dark(d.rect(-.65,.38,-.43,.60), 1.)
            d.dark(d.rect(.43,.38,.65,.60), 1.)
            for xx in [-.48,.48]:
                d.light(d.ellipse(xx,.07,.11,.06), .6)
            if truck:
                d.dark(d.line(-.42,.05,.42,.05,.018), .6)
    elif k == 2:
        if model in (0,1,2):
            d.fill(d.ellipse(-.07,.04,.31,.43,-25 if model!=2 else 0),
                   d.ellipse(.19,-.40,.20,.20),
                   d.polygon([(.35,-.43),(.60,-.35),(.34,-.31)]),
                   d.polygon([(-.24,.24),(-.61,.64),(-.26,.51)]))
            for xx in [-.05,.13]:
                d.fill(d.line(xx,.40,xx+.025,.70,.023),d.line(xx-.08,.72,xx+.12,.72,.018))
            d.dark(d.ellipse(.25,-.43,.035,.035))
            d.dark(d.ellipse(-.12,.03,.16,.27,-25), .30)
            if model == 2:
                d.fill(d.line(.15,-.37,.04,-.76,.055),d.ellipse(.04,-.78,.10,.12))
        elif model in (3,4):
            d.fill(d.ellipse(0,0,.13,.43),
                   d.polygon([(-.03,-.14),(-.86,-.42),(-.68,.15),(-.08,.22)]),
                   d.polygon([(.03,-.14),(.86,-.42),(.68,.15),(.08,.22)]),
                   d.polygon([(-.05,.24),(-.22,.61),(.22,.61),(.05,.24)]))
        else:
            d.fill(d.ellipse(-.08,.12,.54,.30),d.line(.30,.03,.42,-.56,.08),
                   d.ellipse(.43,-.62,.14,.13),d.polygon([(.51,-.65),(.75,-.58),(.50,-.54)]))
            d.dark(d.ellipse(.47,-.65,.025,.025))
    elif k in (3,5):
        face(d, 'cat' if k==3 else 'dog', model)
    elif k in (4,7):
        kind = 'deer' if k==4 else 'horse'
        if model < 4:
            animal_side(d, kind, model%3)
        else:
            # Frontal ungulate, with narrow head, long neck, upright ears.
            d.fill(d.ellipse(0,.12,.30,.33),d.line(0,.02,0,-.44,.14),
                   d.ellipse(0,-.48,.16,.24),
                   d.ellipse(-.14,-.74,.065,.16,-25),d.ellipse(.14,-.74,.065,.16,25))
            for xx in [-.19,.19]:
                d.fill(d.line(xx,.30,xx,.83,.04 if k==4 else .06))
            d.dark(d.ellipse(-.085,-.51,.025,.03), .8)
            d.dark(d.ellipse(.085,-.51,.025,.03), .8)
            d.dark(d.ellipse(0,-.31,.09,.05), .6)
    elif k == 6:
        if model in (0,1,2):
            d.fill(d.ellipse(0,.17,.55,.33),d.ellipse(0,-.06,.42,.25),
                   d.ellipse(-.27,-.28,.13,.14),d.ellipse(.27,-.28,.13,.14),
                   d.ellipse(-.58,.27,.25,.18,-30),d.ellipse(.58,.27,.25,.18,30),
                   d.line(-.25,.12,-.38,.55,.07),d.line(.25,.12,.38,.55,.07))
            d.dark(d.ellipse(-.27,-.29,.055,.06))
            d.dark(d.ellipse(.27,-.29,.055,.06))
            d.dark(d.line(-.27,.02,.27,.02,.025), .6)
        elif model in (3,4):
            d.fill(d.ellipse(-.14,.06,.48,.34),d.ellipse(.35,-.07,.30,.24),
                   d.ellipse(.35,-.27,.12,.12),d.ellipse(-.51,.30,.27,.16,-30),
                   d.line(.27,.08,.41,.50,.065),d.line(-.31,.20,-.68,.52,.055))
            d.dark(d.ellipse(.38,-.29,.047,.047))
            d.dark(d.line(.38,.02,.62,.03,.02), .6)
        else:
            d.fill(d.ellipse(0,.02,.36,.54),d.ellipse(0,-.40,.37,.22))
            for sign in [-1,1]:
                d.fill(d.line(sign*.22,-.20,sign*.69,.01,.07),
                       d.line(sign*.24,.27,sign*.64,.57,.11),
                       d.line(sign*.64,.57,sign*.71,.24,.05))
                d.dark(d.ellipse(sign*.23,-.48,.055,.06))
    elif k == 8:
        if model in (0,1,2,4):
            d.fill(d.polygon([(-.88,.10),(.86,.10),(.61,.43),(-.63,.43)]))
            if model == 1:
                d.fill(d.line(0,.11,0,-.82,.025),
                       d.polygon([(-.04,-.75),(-.04,.03),(-.63,.03)]),
                       d.polygon([(.05,-.59),(.05,.03),(.55,.03)]))
            else:
                d.fill(d.rect(-.35,-.17,.38,.10),d.rect(-.14,-.39,.25,-.17),
                       d.line(.04,-.39,.04,-.75,.022))
                d.dark(d.rect(-.27,-.13,.29,-.06), .6)
                if model == 4:
                    d.fill(d.rect(-.68,-.10,-.39,.10),d.rect(.40,-.07,.68,.10))
            d.dark(d.line(-.66,.25,.66,.25,.023), .4)
        else:
            d.fill(d.polygon([(-.53,.01),(0,.65),(.53,.01),(.39,-.10),(-.39,-.10)]),
                   d.rect(-.28,-.40,.28,-.04),d.rect(-.18,-.61,.18,-.40),
                   d.line(0,-.61,0,-.90,.022))
            d.dark(d.rect(-.21,-.35,.21,-.22), .7)
    return d.mask, d.ink


def soften_energy(e):
    # Fixed binomial neighborhood for tolerance to subpixel boundary placement.
    p = np.pad(e, ((0,0),(1,1),(1,1)), mode='constant')
    out = np.zeros_like(e)
    weights = [1.,2.,1.]
    for y in range(3):
        for x in range(3):
            out += weights[y]*weights[x]*p[:,y:y+16,x:x+16]/16.
    return out


def contour_formula(m):
    # Analytic-model derivative stencil, not a descriptor of any photograph.
    out = []
    for kernel in derivatives():
        v = np.zeros((16,16))
        for y in range(3):
            for x in range(3):
                v[1:-1,1:-1] += kernel[y,x]*m[y:y+14,x:x+14]
        out.append(np.abs(v))
    return np.array(out)


def correlation_weights(a):
    a = a-a.mean()
    denominator = np.sqrt(np.sum(a*a)*a.size)
    return a/max(denominator, 1e-12)


def add_shape_bank(n, gray_normal, edge_normal, direction_normal, colors, color_names, microtexture):
    # Equal numbers of poses, scales, placements and contrast phases per class.
    positions = [(0.,0.),(-.14,0.),(.14,0.),(0.,-.12),(0.,.12)]
    scales = [.65,.85,1.05]
    yy, xx = np.mgrid[:64,:64]
    xx, yy = (xx+.5)/32.-1., (yy+.5)/32.-1.
    gray_rows, edge_rows, direction_rows, material_rows, texture_rows = [], [], [], [], []
    # Generic foreground material and contrast priors, in the color order
    # supplied by add_colors. These are semantic rules, not color prototypes.
    foreground = np.zeros((10,len(color_names)))
    contrast = np.zeros_like(foreground)
    def set_material(k, foreground_values, contrast_values):
        for name,value in foreground_values.items():
            foreground[k,color_names.index(name)] = value
        for name,value in contrast_values.items():
            contrast[k,color_names.index(name)] = value
    set_material(0,{'neutral':.10,'pale':.10},{'blue':-.40,'green':-.30})
    set_material(1,{'red':.30,'neutral':.10},{'red':.50,'green':-.30,'blue':-.25})
    set_material(2,{'warm':.15,'red':.10},{'green':-.15,'warm':.20,'red':.25})
    set_material(3,{'neutral':.15,'warm':.15},{'warm':.20,'green':-.15,'blue':-.15})
    set_material(4,{'warm':.35,'brown':.15},{'warm':.35,'green':-.40})
    set_material(5,{'warm':.25,'brown':.15},{'warm':.25,'green':-.20,'blue':-.15})
    set_material(6,{'green':.60,'brown':.30,'yellow':.15},{'green':.15,'brown':.15,'blue':-.20})
    set_material(7,{'warm':.25,'brown':.10,'dark':.10},{'warm':.30,'dark':.30,'green':-.35})
    set_material(8,{'neutral':.10,'pale':.10},{'blue':-.55,'green':-.25})
    set_material(9,{'neutral':.15,'pale':.10,'red':.20},{'red':.30,'green':-.30,'blue':-.25})
    for k in range(10):
        for model in range(6):
            for scale in scales:
                for tx,ty in positions:
                    for mirror in [-1.,1.]:
                        x, y = (xx-tx)*mirror/scale, (yy-ty)/scale
                        # Small out-of-plane effects are represented by simple
                        # aspect changes, and aircraft/birds additionally tilt.
                        angle = 0.
                        if k == 0:
                            angle = [0.,-12.,0.,25.,-30.,12.][model]
                        elif k == 2:
                            angle = [0.,12.,0.,-18.,18.,0.][model]
                        if model == 1:
                            y = y/1.12
                        elif model == 4:
                            y = y/.86
                        a = math.radians(angle)
                        u, v = x*math.cos(a)+y*math.sin(a), -x*math.sin(a)+y*math.cos(a)
                        mask, ink = draw_object(k, model, u, v, .045/scale)
                        ink *= mask
                        ink -= mask*ink.sum()/max(mask.sum(),1e-12)
                        m = mask.reshape(16,4,16,4).mean((1,3))
                        ink = ink.reshape(16,4,16,4).mean((1,3))
                        silhouette_w = correlation_weights(m)
                        ink_w = correlation_weights(ink)
                        energy = contour_formula(m) + .60*contour_formula(ink)
                        # Fur/feather/skin texture may occur inside an outline.
                        texture_prior = [0.,0.,.018,.035,.015,.035,.025,.015,0.,0.][k]
                        energy += texture_prior*m[None]
                        energy = soften_energy(energy)
                        energy_w = correlation_weights(energy)
                        direction_w = correlation_weights(energy-energy.mean(axis=0,keepdims=True))
                        # Foreground/surround averages are computed from the
                        # *formula mask*, never from labeled image statistics.
                        inner = m/max(m.sum(),1e-12)
                        outer = (1.-m)/max((1.-m).sum(),1e-12)
                        material_w = foreground[k,:,None,None]*inner + contrast[k,:,None,None]*(inner-outer)
                        m8 = m.reshape(8,2,8,2).mean((1,3))
                        inner8 = m8/max(m8.sum(),1e-12)
                        outer8 = (1.-m8)/max((1.-m8).sum(),1e-12)
                        # Amphibian skin often has internal fine mottling;
                        # rigid vehicle panels favor the opposite relation.
                        micro_coefficient = [-.15,-.30,.10,.25,.05,.25,.80,.05,-.20,-.30][k]
                        micro_w = 8.*micro_coefficient*(inner8-outer8)
                        for polarity in [-1.,1.]:
                            gray_rows.append((.35*polarity*silhouette_w + .25*ink_w).ravel())
                            edge_rows.append((.90*energy_w).ravel())
                            direction_rows.append((1.35*direction_w).ravel())
                            material_rows.append(material_w.ravel())
                            texture_rows.append(micro_w.ravel())
    g = n.dense('shape_luminance_matches', gray_normal, gray_rows)
    e = n.dense('shape_contour_matches', edge_normal, edge_rows)
    d = n.dense('shape_orientation_matches', direction_normal, direction_rows)
    c = n.dense('shape_material_matches', colors, material_rows)
    t = n.dense('shape_texture_matches', microtexture, texture_rows)
    matches = n.layer('shape_matches', 'add', [g,e,d,c,t])
    count = len(gray_rows)//10
    grouped = n.layer('shape_class_groups', 'reshape', matches, shape=[10,count,1])
    best = n.pool('shape_pose_invariance', grouped, [count,1], maximum=True)
    return n.layer('shape_evidence', 'reshape', best, shape=[10])


def add_parts(n, gray_normal):
    """Conjunctions of local contrast primitives, entirely from geometry.

    Whole-object correlation can be high because of a single broad silhouette.
    A minimum over three independently centered parts requires all of them.
    """
    yy,xx = np.mgrid[:16,:16]
    xx,yy = (xx+.5)/8.-1.,(yy+.5)/8.-1.
    rows = []
    positions = [(0.,0.),(-.14,0.),(.14,0.),(0.,-.14),(0.,.14)]
    for k in range(10):
        for model in range(2):
            for scale in [.65,.85,1.05]:
                for tx,ty in positions:
                    for mirror in [-1.,1.]:
                        x,y = (xx-tx)*mirror/scale,(yy-ty)/scale
                        def blob(cx,cy,rx,ry,angle=0.):
                            a = math.radians(angle)
                            u = (x-cx)*math.cos(a)+(y-cy)*math.sin(a)
                            v = -(x-cx)*math.sin(a)+(y-cy)*math.cos(a)
                            inner = np.exp(-.5*((u/rx)**2+(v/ry)**2))
                            outer = np.exp(-.5*((u/(2.1*rx))**2+(v/(2.1*ry))**2))
                            inner /= max(inner.sum(),1e-12)
                            outer /= max(outer.sum(),1e-12)
                            result = outer-inner
                            return result/max(np.abs(result).sum(),1e-12)
                        if k == 0:
                            if model == 0:
                                parts = [blob(0,0,.07,.55),blob(-.36,.05,.25,.065,-20),blob(.36,.05,.25,.065,20)]
                            else:
                                parts = [blob(0,.05,.60,.06),blob(-.57,-.20,.045,.20,-15),blob(0,.24,.07,.25,35)]
                        elif k in (1,9):
                            tire_y = .30
                            half_width = .52 if model == 0 else .40
                            roof = -.22 if k==1 else -.48
                            parts = [blob(-half_width,tire_y,.075,.085),blob(half_width,tire_y,.075,.085),
                                     blob(0,roof,.30,.10 if k==1 else .16)]
                        elif k == 2:
                            parts = [blob(.27,-.43,.038,.038),blob(.49,-.33,.11,.035),
                                     blob(-.09,.06,.16,.30,-25 if model==0 else 0)]
                        elif k == 3:
                            eye_y = -.10 if model==0 else -.23
                            parts = [blob(-.22,eye_y,.07,.035),blob(.22,eye_y,.07,.035),blob(0,eye_y+.25,.045,.04)]
                        elif k == 5:
                            eye_y = -.13 if model==0 else -.26
                            parts = [blob(-.20,eye_y,.05,.045),blob(.20,eye_y,.05,.045),blob(0,eye_y+.33,.10,.075)]
                        elif k in (4,7):
                            leg_width = .026 if k==4 else .045
                            lean = 0 if model==0 else 12
                            parts = [blob(-.36,.41,leg_width,.25,lean),blob(.22,.41,leg_width,.25,-lean),
                                     blob(.34,-.36,.065 if k==4 else .10,.23,18)]
                        elif k == 6:
                            parts = [blob(-.26,-.26,.05,.055),blob(.26,-.26,.05,.055),
                                     blob(0,.02,.25,.025 if model==0 else .045)]
                        elif k == 8:
                            parts = [blob(0,.27,.50,.08),blob(.04,-.40,.025,.30),
                                     blob(0,-.11,.21,.06 if model==0 else .13)]
                        for polarity in [-1.,1.]:
                            # Eyes, noses and tires are usually dark; external
                            # limbs, wings, hulls and bodywork can have either sign.
                            for j,part in enumerate(parts):
                                fixed_dark = k in (3,5,6) or (k in (1,9) and j<2) or (k==2 and j==0)
                                rows.append((part if fixed_dark else polarity*part).ravel())
    # Negative responses let maxpool implement minimum without a new operator.
    responses = n.dense('local_part_contrasts',gray_normal,-np.array(rows))
    conjunctions = len(rows)//3
    groups = n.layer('local_part_groups','reshape',responses,shape=[conjunctions,3,1])
    weakest = n.pool('weakest_object_part',groups,[3,1],maximum=True)
    weakest = n.layer('part_class_layout','reshape',weakest,shape=[10,conjunctions//10,1])
    evidence = n.conv('part_sign',weakest,-np.eye(10)[:,:,None,None])
    evidence = n.relu('present_part_conjunctions',evidence)
    best = n.pool('part_pose_invariance',evidence,[conjunctions//10,1],maximum=True)
    return n.layer('part_evidence','reshape',best,shape=[10])


# ---- Human-written scene and material rules --------------------------------
def regions(size):
    yy,xx = np.mgrid[:size,:size]
    x,y = (xx+.5)/size, (yy+.5)/size
    center = np.exp(-.5*(((x-.5)/.24)**2+((y-.5)/.26)**2))
    surround = 1.-center
    # A gentle margin eliminates artificial zero-padding edge responses.
    margin = np.minimum(np.minimum(x,1-x),np.minimum(y,1-y))
    margin = np.clip(margin/.10,0.,1.)
    parts = {
        'all': np.ones_like(x),
        'center': center,
        'surround': surround,
        'top': np.exp(-.5*((y-.20)/.18)**2),
        'bottom': np.exp(-.5*((y-.80)/.18)**2),
        'upper': np.exp(-.5*(((x-.5)/.27)**2+((y-.34)/.20)**2)),
        'lower': np.exp(-.5*(((x-.5)/.27)**2+((y-.66)/.20)**2)),
    }
    return {key:(v*margin)/(v*margin).sum() for key,v in parts.items()}


def add_summaries(n, sources):
    specs = []
    for family, source, channels, size, amplitude in sources:
        for channel in channels:
            for region in regions(size):
                specs.append(f'{family}.{channel}.{region}')
    index = {name:i for i,name in enumerate(specs)}
    outputs = []
    for family, source, channels, size, amplitude in sources:
        w = np.zeros((len(specs),len(channels),size,size),np.float32)
        for c,channel in enumerate(channels):
            for region,mask in regions(size).items():
                w[index[f'{family}.{channel}.{region}'],c] = amplitude*mask
        outputs.append(n.dense('summarize_'+family, source, w.reshape(len(specs),-1)))
    return n.layer('scene_features','add',outputs), index


def add_scene_rules(n, summaries, index, shape, parts):
    F = len(index)
    def feature(family, channel, region='all'):
        w = np.zeros(F)
        w[index[f'{family}.{channel}.{region}']] = 1.
        return w
    def color(channel, region='all'):
        return feature('color',channel,region)
    def coarse(region='all'):
        return (feature('edge','x',region)+feature('edge','y',region))*.5
    def fine(region='all'):
        return (feature('fine','x',region)+feature('fine','y',region))*.5
    def long(region='all'):
        return (feature('long','x',region)+feature('long','y',region))*.5
    def horizontal(family, region):
        return feature(family,'y',region)-feature(family,'x',region)

    # Scores are in fixed heuristic evidence units, not fitted probabilities.
    base = np.zeros((10,F))
    def assign(expression, coefficients):
        for k,coefficient in enumerate(coefficients):
            base[k] += coefficient*expression

    # Vegetation and soil are more characteristic of wildlife backgrounds.
    assign(color('green','surround'), [-.65,-.50,.65,.00,1.15,.05,.55,.65,-.80,-.45])
    assign(color('warm','surround'), [-.40,-.25,.30,.20,.45,.30,.40,.25,-.45,-.20])
    # Blue at the bottom is water evidence; blue above is also sky evidence.
    assign(color('blue','top'), [.95,.10,.30,-.15,-.05,-.15,-.35,.05,.65,.15])
    assign(color('blue','bottom'), [.40,-.15,.10,-.25,-.45,-.25,-.40,-.35,1.25,-.10])
    assign(color('pale','surround'), [.55,.10,-.05,.15,-.20,.05,-.35,-.05,.35,.20])
    # Material/color of the central foreground, without memorized prototypes.
    assign(color('green','center'), [-.25,-.35,.30,-.20,.05,-.15,1.05,.00,-.40,-.30])
    assign(color('warm','center'), [-.30,-.20,.20,.45,.55,.70,.25,.60,-.30,-.20])
    assign(color('brown','center'), [-.20,-.15,.15,.25,.40,.45,.20,.45,-.25,-.15])
    assign(color('red','center')-.45*color('red','surround'), [.05,.85,.15,-.05,-.15,-.10,-.10,-.10,.05,.65])
    assign(color('neutral','center'), [.05,.20,-.10,.25,-.05,.00,-.25,.00,.10,.35])
    assign(color('dark','center')-.5*color('dark','surround'), [.00,.10,.15,.25,.00,.15,-.10,.35,.00,.05])
    # Fine structure throughout the background suggests outdoor habitat.
    assign(fine('surround'), [-.55,-.10,.25,.05,.35,.05,.20,.15,-.35,-.10])
    # Fine central texture supports coat/skin/feathers rather than body panels.
    assign(fine('center')-.50*coarse('center'), [-.20,-.30,.10,.35,.10,.35,.25,.10,-.20,-.25])
    assign(long('center')-.65*fine('center'), [.15,.65,-.15,-.10,-.15,-.10,-.20,-.10,.30,.75])
    assign(horizontal('long','lower'), [.15,.35,-.10,-.10,-.15,-.10,.10,-.10,.65,.25])
    assign(horizontal('fine','bottom'), [.00,.05,-.10,-.10,-.15,-.10,.10,-.10,.45,.05])
    # Relative vertical placement matters: a car's windows/roof lie below a
    # truck cab's roof. A body low in frame differs from a tall central box.
    low_body = long('lower')-long('upper')
    assign(low_body, [.10,.85,-.10,-.10,-.15,-.10,.15,-.10,.25,-.60])
    assign(horizontal('edge','center'), [.15,.60,-.10,-.10,-.20,-.10,.20,-.15,.20,-.25])
    # Fine isolated skin detail and squat contour versus low-texture panels.
    roughness = feature('micro','roughness','center')-.65*feature('micro','roughness','surround')
    assign(roughness, [-.15,-.25,.05,.15,.00,.15,.65,.00,-.15,-.25])

    rule_w, rule_b, rule_out = [],[],[]
    def rule(expression, threshold, outputs):
        rule_w.append(expression)
        rule_b.append(-threshold)
        rule_out.append(outputs)

    # Fuzzy conjunctions: a rule only contributes when combined evidence clears
    # its stated, manually chosen threshold. No threshold is data-estimated.
    sky = .85*color('blue','surround')+.65*color('pale','surround')-.65*fine('surround')
    rule(sky,.10,[1.10,-.10,.10,-.25,-.25,-.25,-.30,-.25,.30,-.10])
    water = color('blue','bottom')+.40*color('blue','surround')+.50*horizontal('fine','bottom')-.45*color('green','bottom')
    rule(water,.15,[-.10,-.25,-.10,-.20,-.25,-.20,-.20,-.25,1.10,-.20])
    grass_brown = color('green','surround')+.65*color('warm','center')+.35*color('brown','center')
    rule(grass_brown,.40,[-.25,-.25,.15,-.10,.75,.10,.20,.65,-.30,-.25])
    green_subject = color('green','center')-.65*color('green','surround')+.20*color('yellow','center')
    rule(green_subject,.10,[-.10,-.15,.20,-.10,-.10,-.10,1.25,-.10,-.15,-.15])
    green_field = color('green','surround')-.45*color('green','center')
    rule(green_field,.10,[-.10,-.05,.20,-.05,.55,.00,-.25,.40,-.15,-.05])
    brown_fur = .70*color('warm','center')+.40*color('brown','center')+.40*fine('center')-.50*color('green','surround')-.40*color('blue','surround')
    rule(brown_fur,.25,[-.20,-.20,.05,.40,.00,.65,.15,.10,-.20,-.20])
    neutral_fur = .70*color('neutral','center')+.50*fine('center')-.45*long('center')-.40*color('blue','surround')-.30*color('green','surround')
    rule(neutral_fur,.20,[-.10,-.10,.00,.60,-.10,.25,-.10,-.05,-.10,-.05])
    paint = color('red','center')-.55*color('red','surround')+.50*long('center')-.35*fine('center')
    rule(paint,.15,[.00,.75,.10,-.10,-.20,-.10,-.20,-.10,.05,.55])
    # Long, coherent straight boundaries support man-made objects.
    rigid = long('center')-.70*fine('center')+.25*color('neutral','center')
    rule(rigid,.15,[.15,.55,-.15,-.15,-.15,-.15,-.15,-.15,.20,.65])
    # A dark substantial animal against grass is more horse-like than deer-like.
    dark_animal = color('dark','center')+.55*color('green','surround')-.45*color('dark','surround')
    rule(dark_animal,.30,[-.15,-.10,.05,.05,-.05,.10,-.10,.65,-.15,-.10])

    linear = n.dense('scene_linear_evidence', summaries, base)
    hidden = n.dense('scene_rule_tests', summaries, rule_w, rule_b)
    hidden = n.relu('scene_rules', hidden)
    fuzzy = n.dense('scene_rule_evidence', hidden, np.array(rule_out).T)
    # Equal geometric confidence scale and equal class priors.
    geometric = n.dense('geometric_evidence', shape, np.eye(10)*1.65)
    local = n.dense('local_geometric_evidence', parts, np.eye(10)*1.25)
    return n.layer('class_scores', 'add', [linear,fuzzy,geometric,local])


def build(destination):
    n = Network()
    rgb16 = n.pool('rgb16', 'image', 2)
    gray16 = n.conv('luminance16', rgb16, LUMA[None,:,None,None])
    gray_normal = n.layer('contrast_normalized_luminance', 'norm', gray16)
    colors, color_names = add_colors(n, rgb16)
    edges = add_edges(n,rgb16,3,'color_edges')
    smooth_w = np.zeros((4,4,3,3))
    for c in range(4):
        smooth_w[c,c] = np.outer([1.,2.,1.],[1.,2.,1.])/16.
    smooth = n.conv('tolerant_contours', edges, smooth_w, padding=1)
    arranged = n.layer('joint_orientation_shape','reshape',smooth,shape=[1,64,16])
    normalized = n.layer('contour_contrast_normalization','norm',arranged)
    edge_normal = n.layer('normalized_contours','reshape',normalized,shape=[4,16,16])
    direction = n.conv('orientation_selectivity',smooth,(np.eye(4)-np.ones((4,4))/4.)[:,:,None,None])
    direction = n.layer('orientation_selectivity_shape','reshape',direction,shape=[1,64,16])
    direction = n.layer('orientation_selectivity_norm','norm',direction)
    direction = n.layer('orientation_selectivity_grid','reshape',direction,shape=[4,16,16])
    long_edges = add_edges(n,gray16,1,'long_edges',long=True)

    gray32 = n.conv('luminance32','image',LUMA[None,:,None,None])
    fine_raw = add_edges(n,gray32,1,'fine_edges')
    fine = n.pool('fine_texture_grid',fine_raw,4)
    # High-frequency center/surround response is another fixed material cue.
    laplace = -np.ones((3,3))/8.
    laplace[1,1] = 1.
    hp = n.conv('microtexture_signed',gray32,np.stack([laplace,-laplace])[:,None])
    hp = n.relu('microtexture_phases',hp)
    hp = n.conv('microtexture_magnitude',hp,np.ones((1,2,1,1)),padding=1)
    hp = n.pool('microtexture_grid',hp,4)

    shape = add_shape_bank(n,gray_normal,edge_normal,direction,colors,color_names,hp)
    parts = add_parts(n,gray_normal)
    sources = [
        ('color',colors,color_names,16,1.),
        ('edge',edges,['x','d1','y','d2'],16,8.),
        ('fine',fine,['x','d1','y','d2'],8,8.),
        ('long',long_edges,['x','d1','y','d2'],16,8.),
        ('micro',hp,['roughness'],8,8.),
        ('light',gray16,['luma'],16,1.),
    ]
    summaries,index = add_summaries(n,sources)
    add_scene_rules(n,summaries,index,shape,parts)
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'network.json').open('w') as f:
        json.dump({'layers':n.layers},f,indent=2)
        f.write('\n')
    np.savez(destination/'weights.npz',**n.weights)
    print('Wrote',len(n.layers),'layers and',sum(w.size for w in n.weights.values()),'float32 weights.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 build.py NEW_FOLDER')
    build(sys.argv[1])
