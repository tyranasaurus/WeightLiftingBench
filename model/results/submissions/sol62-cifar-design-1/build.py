#!/usr/bin/env python3
"""Data-free compiler for a fixed scene-and-shape classifier.

All spatial models below are drawings in abstract object coordinates. The
builder never opens a data file. At inference all computations are ordinary
convolution, pooling, affine maps and ReLUs.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


class Net:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def layer(self, name, op, source, **settings):
        inputs = [source] if isinstance(source, str) else source
        self.layers.append(dict(name=name, op=op, inputs=inputs, **settings))
        return name

    def affine(self, name, source, w, b=None, stride=1, padding=0):
        w = np.asarray(w, np.float32)
        n = len(w)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = (np.zeros(n, np.float32) if b is None
                                     else np.asarray(b, np.float32))
        if w.ndim == 4:
            k = w.shape[-1]
            return self.layer(name, 'conv', source, out=n, kernel=[k, k],
                              stride=stride, padding=padding)
        return self.layer(name, 'dense', source, out=n)

    def pool(self, name, op, source, size, stride):
        return self.layer(name, op, source, size=list(size), stride=list(stride))


class Expr:
    """Sparse affine expression in one feature bank."""
    def __init__(self, terms=None, bias=0.):
        self.terms = {} if terms is None else terms
        self.bias = bias

    def __add__(self, rhs):
        if not isinstance(rhs, Expr):
            return Expr(self.terms, self.bias + rhs)
        out = self.terms.copy()
        for k, v in rhs.terms.items():
            out[k] = out.get(k, 0.) + v
        return Expr(out, self.bias + rhs.bias)

    __radd__ = __add__

    def __mul__(self, scale):
        return Expr({k: scale*v for k, v in self.terms.items()}, scale*self.bias)

    __rmul__ = __mul__

    def __sub__(self, rhs):
        return self + (-1. * rhs)

    def __rsub__(self, lhs):
        return (-1. * self) + lhs

    def __truediv__(self, scale):
        return self * (1. / scale)


def fields(names):
    return {name: Expr({i: 1.}) for i, name in enumerate(names)}


def compile_affine(net, name, src, size, rows, conv):
    w = np.zeros((len(rows), size), np.float32)
    b = np.zeros(len(rows), np.float32)
    for i, e in enumerate(rows):
        if not isinstance(e, Expr):
            e = Expr(bias=e)
        for j, v in e.terms.items():
            w[i, j] = v
        b[i] = e.bias
    return net.affine(name, src, w[:, :, None, None] if conv else w, b)


class Hinges:
    """Pack scalar piecewise-linear programs into pairs of affine layers."""
    def __init__(self):
        self.in_rows = []
        self.out_rows = {}

    def pos(self, e):
        idx = len(self.in_rows)
        self.in_rows.append(e)
        return Expr({idx: 1.})

    def copy(self, e):
        return self.pos(e) - self.pos(-1. * e)

    def ramp(self, e, low, high):
        t = (e-low)/(high-low)
        return self.pos(t)-self.pos(t-1.)

    def minimum(self, a, b):
        return self.copy(a) - self.pos(a-b)

    def emit(self, net, name, src, size, spatial):
        a = compile_affine(net, name+'_affine', src, size, self.in_rows, spatial)
        r = net.layer(name+'_hinges', 'relu', a)
        y = compile_affine(net, name, r, len(self.in_rows), list(self.out_rows.values()), spatial)
        return y, fields(self.out_rows)


def grid(k):
    a = np.arange(k, dtype=np.float64) - (k-1)/2
    return np.meshgrid(a, a, indexing='xy')


def gauss(k, sx, sy=None):
    if sy is None:
        sy = sx
    x, y = grid(k)
    a = np.exp(-.5*((x/sx)**2 + (y/sy)**2))
    return a/a.sum()


def odd_derivative(k, theta, sx, sy=None):
    x, y = grid(k)
    a = gauss(k, sx, sy) * (x*math.cos(theta)+y*math.sin(theta))
    return 2*a/np.abs(a).sum()


def image_fields(net):
    # A fine derivative bank measures microtexture before downsampling.
    lum = np.array([.25, .55, .20])
    chroma = (np.array([1., -1., 0.]), np.array([.5, .5, -1.]))
    wf, nf = [], []
    for j, c in enumerate((lum, *chroma)):
        for i in range(2):
            k = odd_derivative(3, i*math.pi/2, .8)
            for s in (1, -1):
                wf.append(s*c[:, None, None]*k)
                nf.append(f'm{j}_{i}_{s}')
    k = np.full((3, 3), -.125)
    k[1, 1] = 1.
    for s in (1, -1):
        wf.append(s*lum[:, None, None]*k)
        nf.append(f'grain_{s}')
    f = net.affine('micro_derivatives', 'image', wf, padding=1)
    f = net.layer('micro_abs_halves', 'relu', f)
    f = net.pool('micro_cells', 'avgpool', f, (2, 2), (2, 2))
    q = net.pool('rgb_cells', 'avgpool', 'image', (2, 2), (2, 2))

    # Coarse derivatives, elongated bars, and symmetric center-surrounds.
    wb, nb = [], []
    delta = np.zeros((7, 7)); delta[3, 3] = 1.
    for name, c in zip('rgb', np.eye(3)):
        wb.append(c[:, None, None]*delta)
        nb.append(name)
    for i in range(4):
        wb.append(lum[:, None, None]*odd_derivative(7, i*math.pi/4, .81))
        nb.append(f'd{i}')
    for i, (sx, sy) in enumerate(((.68, 2.0), (2.0, .68))):
        wb.append(lum[:, None, None]*odd_derivative(7, i*math.pi/2, sx, sy))
        nb.append(f'line{i}')
    for name, k in (
        ('spot', gauss(7, .48)-gauss(7, 1.07)),
        ('blob', gauss(7, .79)-gauss(7, 1.70)),
    ):
        wb.append(lum[:, None, None]*k*(2/np.abs(k).sum()))
        nb.append(name)
    nq, nm = len(nb), len(nf)
    base = net.affine('coarse_derivatives', q, np.concatenate([wb, np.zeros((nm, 3, 7, 7))]), padding=3)
    embed = np.zeros((nq+nm, nm, 1, 1)); embed[nq:, :, 0, 0] = np.eye(nm)
    mic = net.affine('embed_micro_cells', f, embed)
    mixed = net.layer('mixed_derivatives', 'add', [base, mic])
    d = fields(nb + nf)
    r, g, b = (d[k] for k in 'rgb')
    l = .25*r + .55*g + .20*b

    h = Hinges()
    o = h.out_rows
    o['luma'] = h.copy(l)
    o['dark'] = h.ramp(-1*l, -.45, -.12)
    o['pale_luma'] = h.ramp(l, .47, .83)
    o['white_luma'] = h.ramp(l, .68, .95)
    o['mid_light'] = h.ramp(l, .17, .49)
    o['rg_pos'] = h.pos(r-g)
    o['rg_neg'] = h.pos(g-r)
    o['bg_pos'] = h.pos(b-g)
    o['bg_neg'] = h.pos(g-b)
    o['rb_abs'] = h.pos(r-b) + h.pos(b-r)
    o['blue'] = h.ramp(.83*b+.17*g-r, .015, .24)
    o['teal'] = h.ramp(.52*g+.48*b-r, .017, .20)
    o['blue_green_gap'] = h.ramp(g-b, .025, .20)
    o['green'] = h.ramp(g-r, .012, .13)
    o['earth'] = h.ramp(.52*r+.48*g-b, .025, .23)
    o['grass_ok'] = 1.-h.ramp(r-g, .030, .16)
    o['red'] = h.ramp(r-.63*g-.37*b, .085, .42)
    o['gold'] = h.ramp(.32*r+.68*g-b, .08, .34)
    o['maroon'] = h.ramp(r-g, .065, .30)
    o['grain'] = h.ramp(d['grain_1']+d['grain_-1'], .009, .095)
    for i in range(3):
        band = sum(d[f'm{i}_{j}_{s}'] for j in range(2) for s in (1, -1))
        o[f'micro{i}'] = h.ramp(.5*band, .008 if i else .018, .16 if i else .25)
    for i in range(4):
        o[f'edge{i}'] = h.ramp(d[f'd{i}'], .033, .38) + h.ramp(-1*d[f'd{i}'], .033, .38)
    for i in range(2):
        x = d[f'line{i}']
        o[f'bar{i}'] = h.ramp(x, .045, .34) + h.ramp(-1*x, .045, .34)
    for k in ('spot', 'blob'):
        o[k+'_dark'] = h.ramp(-1*d[k], .045 if k == 'spot' else .055, .35)
        o[k+'_light'] = h.ramp(d[k], .045 if k == 'spot' else .055, .35)
    src, v = h.emit(net, 'contrast_and_palette', mixed, len(d), spatial=True)

    h = Hinges(); o = h.out_rows
    sat = .5*(v['rg_pos']+v['rg_neg']+v['bg_pos']+v['bg_neg']+v['rb_abs'])
    edge = sum(v[f'edge{i}'] for i in range(4)) * .25
    busy = .61*v['grain'] + .39*v['micro0']
    o['luma'] = h.pos(v['luma'])
    o['edge'] = h.pos(edge)
    o['busy'] = h.pos(busy)
    o['smooth'] = 1. - h.ramp(.55*busy+.45*edge, .07, .32)
    o['grainy'] = h.ramp(v['grain'] + .45*(v['micro1']+v['micro2']) - .60*edge, .09, .54)
    o['white'] = h.minimum(v['white_luma'], 1.-h.ramp(sat, .095, .28))
    o['pale'] = h.minimum(v['pale_luma'], 1.-h.ramp(sat, .085, .30))
    o['blue'] = h.pos(v['blue'])
    o['cool'] = h.ramp(.65*v['blue']+.35*v['teal']-.34*v['blue_green_gap'], .03, .77)
    o['sky'] = h.ramp(.83*v['blue']+.17*v['teal']+.43*v['mid_light']-.80*busy-.30*v['blue_green_gap'], .32, 1.12)
    o['water'] = h.ramp(.53*v['blue']+.47*v['teal']+.13*v['mid_light']-.35*v['white_luma']-.37*v['blue_green_gap'], .11, .72)
    o['greens'] = h.ramp(.69*v['green']+.31*v['earth']+.28*v['grass_ok']-.31*v['dark'], .31, .88)
    o['olive'] = h.ramp(.66*v['earth']+.34*v['grass_ok']-.21*v['teal']-.12*v['dark'], .31, .91)
    o['bark'] = h.ramp(.58*v['earth']+.30*v['dark']+.15*v['rg_pos']-.30*v['blue'], .20, .71)
    o['warm'] = h.ramp(.60*v['earth']+.22*v['rg_pos']+.18*v['mid_light']-.31*v['green'], .18, .68)
    o['rust'] = h.ramp(.56*v['red']+.34*v['maroon']+.10*v['gold'], .04, .74)
    o['vivid'] = h.ramp(sat, .19, .52)
    o['neutral'] = 1.-h.ramp(sat, .055, .23)
    o['shadow'] = h.pos(v['dark'])
    # A repeated geometric edge is strongest when the elongated derivative
    # agrees with the small-scale derivative, not merely when the tile is busy.
    for i in range(2):
        o[f'bar{i}'] = h.minimum(v[f'bar{i}'], .17+.95*v[f'edge{2*i}'])
    o['axis'] = h.ramp(.51*(v['bar0']+v['bar1'])-.30*(v['edge1']+v['edge3']), .016, .39)
    o['corner'] = h.ramp(v['bar0']+v['bar1']-.56, 0, .86)
    o['spot_dark'] = h.minimum(.22+.87*v['dark'], v['spot_dark'])
    o['spot_light'] = h.minimum(.27+.84*v['pale_luma'], v['spot_light'])
    o['blob_dark'] = h.minimum(.23+.88*v['dark'], v['blob_dark'])
    o['blob_light'] = h.minimum(.33+.77*v['pale_luma'], v['blob_light'])
    for i in range(4):
        o[f'edge{i}'] = h.pos(v[f'edge{i}'])
    return h.emit(net, 'object_and_scene_fields', src, len(v), spatial=True)


# Geometry for the scene branch is ordinal: sky is up, surfaces down,
# objects usually in the middle, and left and right are interchangeable.
def scene_summary(net, source, fmap):
    names = list(fmap)
    y, x = np.mgrid[:16, :16]
    a, b = (x-7.5)/8, (y-7.5)/8
    rim_x = np.clip((abs(a)-.43)*2.6, 0, 1)
    rim_y = np.clip((abs(b)-.46)*2.6, 0, 1)
    zones = dict(all=np.ones_like(a), top=np.clip((.03-b)*1.9, 0, 1),
                 bot=np.clip((b-.02)*1.9, 0, 1),
                 mid=np.exp(-.5*((a/.48)**2+((b-.06)/.40)**2)),
                 core=np.exp(-.5*((a/.31)**2+(b/.29)**2)),
                 rim=.70*rim_x+.30*rim_y,
                 high_side=np.clip((.09-b)*1.8, 0, 1)*(.28+.72*rim_x),
                 low_side=np.clip((b+.06)*1.9, 0, 1)*(.35+.65*rim_x),
                 band=np.exp(-.5*((b-.29)/.26)**2)*np.exp(-.5*(a/.67)**2))
    requests = {
        'luma': 'top bot mid rim', 'sky': 'top bot high_side low_side',
        'cool': 'top bot mid rim', 'water': 'top bot low_side mid',
        'pale': 'top bot rim mid', 'white': 'mid rim bot',
        'greens': 'top bot mid rim', 'olive': 'top bot mid rim',
        'bark': 'top bot mid rim', 'warm': 'mid rim bot',
        'rust': 'mid rim', 'vivid': 'mid rim', 'neutral': 'mid rim bot',
        'shadow': 'mid rim bot band', 'smooth': 'top bot rim mid',
        'busy': 'top bot mid rim', 'grainy': 'top bot mid rim core',
        'edge': 'mid rim core', 'axis': 'mid rim band', 'corner': 'mid rim',
        'bar0': 'mid rim band', 'bar1': 'mid rim band',
        'spot_dark': 'mid band', 'spot_light': 'mid band',
        'blob_dark': 'mid band', 'blob_light': 'mid band',
    }
    rows, out_names = [], []
    for name, areas in requests.items():
        i = names.index(name)
        for area in areas.split():
            region = zones[area].copy()
            if name in {'sky', 'smooth', 'busy', 'grainy', 'edge', 'axis', 'corner',
                        'bar0', 'bar1', 'spot_dark', 'spot_light', 'blob_dark', 'blob_light'}:
                # Ignore the artificial gradients at the zero-padded border.
                for ax in range(2):
                    trim = 2 if name.startswith(('bar', 'corner', 'axis')) else 1
                    region = np.swapaxes(region, 0, ax)
                    region[:trim] = region[-trim:] = 0
                    region = np.swapaxes(region, 0, ax)
            z = np.zeros((len(names), 16, 16), np.float32)
            z[i] = region / region.sum()
            rows.append(z.ravel())
            out_names.append(name+'_'+area)
    return net.affine('regional_scene_means', source, np.stack(rows)), out_names


# An anatomical/engineering stencil. E = ellipse (angle in degrees), P =
# polygon, L = a tapered limb, S = a semantic patch used as an additional cue.
# Nominal coordinates refer to the object, not to positions in any photo.
class Stencil:
    def __init__(self):
        self.shapes = []
        self.marks = []

    def E(self, x, y, a, b, deg=0):
        self.shapes.append(('e', (x, y, a, b, deg)))

    def P(self, *points):
        self.shapes.append(('p', points))

    def L(self, x, y, u, v, w1, w2):
        length = math.hypot(u-x, v-y)
        a, b = -(v-y)/length, (u-x)/length
        self.P((x+a*w1, y+b*w1), (u+a*w2, v+b*w2),
               (u-a*w2, v-b*w2), (x-a*w1, y-b*w1))

    def S(self, field, weight, x, y, a, b, deg=0):
        self.marks.append((field, weight, (x, y, a, b, deg)))

    def eyes(self, y, sep, w, x=0., frog=False):
        for side in (-1, 1):
            self.S('spot_dark', w*(.52 if frog else .73), x+side*sep, y, .15, .11)
            self.S('blob_dark', w*(.48 if frog else .27), x+side*sep, y+.015, .14, .14)

    def tire(self, x, y, a):
        self.E(x, y, a, a*1.04)
        self.S('shadow', .040, x, y, .69*a, .73*a)
        self.S('blob_dark', .090, x, y, .92*a, .83*a)
        self.S('luma', -.080, x, y, .88*a, .85*a)
        self.S('luma', .080, x, y-1.90*a, 1.12*a, .72*a)


def airplane(p):
    s = Stencil()
    if p in (0, 1):                       # airliner broadside / light aircraft
        s.E(-.02, -.02, .94, .13 if p == 0 else .17, -3)
        s.P((.57,.015),(.90,-.39),(.77,-.42),(.39,-.01))
        s.P((-.14,.00),(.17,.00),(.51,.39),(.34,.41),(-.25,.11))
        s.P((-.20,.00),(.12,-.09),(.34,-.25),(.22,-.25),(-.38,-.03))
        if p == 0:
            s.E(.02,.15,.17,.095)
        else:
            s.E(-.36,-.08,.21,.11)
        s.S('edge1', .21, .55, -.07, .43, .21)
    elif p in (2, 3):                     # approaching on spread wings
        slope = .08 if p == 2 else .21
        s.P((-.99,-slope-.04),(-.27,-.09),(.27,-.09),(.99,slope-.025),
            (.98,slope+.05),(.19,.055),(-.19,.055),(-.98,-slope+.05))
        s.E(0.,-.01,.22,.16)
        s.P((-.08,-.06),(-.06,-.42),(.035,-.42),(.12,-.08))
        for side in (-1,1):
            s.E(side*.37, .075+side*slope*.35,.104,.086)
            s.S('spot_dark', .095, side*.37, .087+side*slope*.35,.15,.13)
        s.S('bar1', .22, 0., .00, .69, .20)
    else:                                  # underside of a swept-wing aircraft
        s.E(.02,-.08,.115,.86)
        s.P((-.08,-.37),(-.99,.28),(-.93,.44),(-.10,.23),
            (.11,.23),(.93,.44),(.99,.28),(.08,-.37))
        s.P((-.06,.48),(-.34,.72),(-.35,.79),(.0,.71),(.35,.79),(.34,.72),(.06,.48))
        s.S('edge1', .13, -.48,.14,.47,.19,-30)
        s.S('edge3', .13, .48,.14,.47,.19,30)
    return s


def automobile(p):
    s = Stencil()
    if p < 3:                             # side, incl. hatchback and van
        top = (-.53, -.62, -.73)[p]
        s.P((-.98,.20),(-.97,-.10),(-.61,-.22),(-.33,top),(.43,top+.045),
            ((.78 if p<2 else .88),-.22),(.97,-.15),(.98,.20),(.77,.26),(-.88,.26))
        s.P((-.95,.14),(.91,.14),(.88,.30),(-.88,.30))
        for x in (-.62, .63): s.tire(x,.25,(.18,.20,.19)[p])
        s.S('shadow', .080, .06, (top-.21)/2, .41,.16)
        s.S('luma', -.095, .05, (top-.21)/2,.44,.14)
        s.S('luma', .095, 0.,.075,.57,.11)
    else:                                  # front or front-quarter
        shift = 0 if p == 3 else .19
        s.P((-.96,.45),(-.91,-.08),(-.65+shift,-.20),(-.49+shift,-.63),
            (.51+shift,-.67),(.75+shift,-.26),(.91,-.18),(.96,.38),(.72,.51),(-.76,.51))
        for x in (-.70, .72): s.tire(x,.43,.14)
        s.S('shadow', .11, shift, -.37,.46,.20)
        s.S('luma', -.11, shift,-.38,.43,.13)
        s.S('luma', .11, 0.,-.095,.53,.12)
        for x in (-.60,.60): s.S('spot_light', .085, x,.11,.22,.10)
        s.S('bar1', .14, -.08,.14,.63,.17)
    return s


def bird(p):
    s = Stencil()
    if p < 3:                              # upright, horizontal, long-necked
        if p == 0:
            s.E(-.06,.03,.39,.57,-26); s.E(.20,-.48,.23,.23)
            s.P((.34,-.56),(.59,-.47),(.34,-.41))
            s.P((-.19,.14),(-.70,.69),(-.83,.61),(-.36,-.015))
            s.L(.01,.30,.00,.58,.075,.035); s.L(.17,.31,.17,.60,.06,.033)
            s.S('spot_dark', .15,.27,-.51,.13,.11)
        elif p == 1:
            s.E(.00,.09,.65,.29,-9); s.E(.51,-.18,.23,.25,-20)
            s.P((.64,-.25),(.95,-.21),(.70,-.12))
            s.P((-.48,-.02),(-.94,-.14),(-.80,.14),(-.45,.24))
            s.L(.10,.24,.15,.52,.06,.038)
            s.S('spot_dark', .15,.56,-.24,.12,.11)
        else:
            s.E(-.14,.25,.65,.33,-8); s.L(.34,.20,.53,-.48,.16,.10)
            s.E(.54,-.54,.17,.14,12); s.P((.61,-.62),(.86,-.53),(.65,-.49))
            s.L(.08,.45,.08,.83,.06,.038); s.L(-.18,.45,-.22,.82,.06,.035)
            s.S('spot_dark', .12,.56,-.59,.11,.11)
        s.S('luma', -.075, -.30,.19,.42,.25)
        s.S('luma', .075, -.13,.01,.43,.22)
    else:                                   # flying, swept/wheeling poses
        v = .26 if p == 3 else -.36
        s.E(.03,.02,.23,.48,22); s.E(.14,-.37,.13,.17)
        s.P((-.18,-.23),(-.61,-.54),(-.98,-.79),(-.88,-.26),(-.54,.04),(-.13,.17))
        s.P((.12,-.13),(.53,v-.21),(.97,v-.19),(.81,v+.06),(.53,v+.17),(.09,.18))
        s.P((-.13,.21),(-.33,.61),(-.03,.68),(.04,.24))
    return s


def cat(p):
    s = Stencil()
    if p < 2:                              # forward face, narrow or broad
        w = .59 if p == 0 else .76
        s.E(0.,.08,w,.58)
        for t in (-1,1):
            s.P((t*.18,-.40),(t*(w-.09),-.89),(t*(w+.07),-.27),(t*(w-.06),-.09))
            s.S('edge0', .092, t*(w-.07),-.49,.15,.33,t*10)
        s.eyes(.014,w*.48,.115)
        s.S('spot_dark', .118,0.,.255,.17,.13)
        s.S('luma', -.080,0.,.25,.15,.10)
        s.S('luma', .080,0.,.37,.31,.11)
        s.S('grainy', .115,0.,-.24,w*.66,.20)
    else:
        if p == 2:                         # seated
            s.E(-.19,.37,.49,.58,-18); s.E(.22,-.29,.34,.34)
            s.P((-.01,-.39),(.005,-.85),(.25,-.50)); s.P((.24,-.49),(.51,-.76),(.52,-.32))
            s.E(.19,.58,.18,.27); s.E(-.55,.64,.30,.12,18)
            s.S('spot_dark', .15,.32,-.32,.21,.13)
        elif p == 3:                       # walking
            s.E(-.09,.09,.65,.29,-9); s.E(.62,-.16,.25,.26)
            s.P((.45,-.23),(.49,-.58),(.69,-.29)); s.P((.62,-.29),(.80,-.53),(.83,-.20))
            s.L(-.62,.04,-.94,-.50,.104,.064); s.L(-.47,.17,-.57,.58,.109,.073)
            s.L(.31,.17,.45,.54,.105,.073); s.L(.53,.12,.63,.57,.089,.062)
            s.S('spot_dark', .15,.70,-.13,.12,.11)
        else:                               # crouching/lying
            s.E(-.26,.23,.68,.38,-5); s.E(.49,-.03,.32,.31)
            s.P((.20,-.14),(.30,-.51),(.47,-.21)); s.P((.48,-.18),(.70,-.50),(.77,-.04))
            s.E(.53,.40,.38,.11); s.S('spot_dark', .14,.57,-.062,.19,.11)
    return s


def deer(p):
    s = Stencil()
    if p < 3:
        s.E(-.18,.035,.63,.285,-3)
        s.E(.35,-.18,.27,.18,-61)
        s.E(.55,-.38,.28,.13,9 if p != 2 else 23)
        s.E(.48,-.52,.18,.065,-50); s.E(.61,-.50,.16,.060,-18)
        for x, ankle in ((-.60,-.65),(-.39,-.32),(.18,.15),(.37,.39)):
            s.L(x,.16,x+.035,.43,.080,.05)
            s.L(x+.035,.43,ankle,.85 if p==0 else .77,.051,.039)
        s.L(-.73,.015,-.94,-.07,.09,.042)
        if p == 1:                         # antler branches
            for x, t in ((.44,-1),(.62,1)):
                s.L(x,-.47,x+t*.11,-.83,.035,.029)
                for z in (-1,1): s.L(x+t*.07,-.67,x+t*.12+z*.14,-.89,.030,.020)
        s.S('warm', .10,-.14,.023,.46,.20)
        s.S('shadow', .067,-.17,.07,.46,.20)
        s.S('luma', -.10,-.20,.07,.48,.18)
        s.S('luma', .10,-.17,.54,.25,.085)
    else:                                   # ears/antlers framing a narrow face
        s.E(0., .25, .27,.55)
        s.P((-.22,-.17),(-.74,-.36),(-.65,-.56),(-.42,-.47),(-.12,-.30))
        s.P((.22,-.17),(.74,-.36),(.65,-.56),(.42,-.47),(.12,-.30))
        s.E(0.,.53,.19,.22)
        if p == 4:
            for t in (-1,1):
                s.L(t*.25,-.17,t*.59,-.77,.055,.039)
                s.L(t*.42,-.46,t*.85,-.68,.04,.023)
                s.L(t*.48,-.55,t*.39,-.92,.039,.023)
                s.L(t*.57,-.72,t*.74,-.93,.032,.020)
        s.eyes(.02,.19,.095)
    return s


def dog(p):
    s = Stencil()
    if p < 3:                               # frontal face and muzzle
        w = .62 if p != 1 else .69
        s.E(0.,-.11,w,.58)
        s.E(0.,.38,.35 if p != 1 else .48,.34)
        for t in (-1,1):
            if p == 2:
                s.P((t*.18,-.48),(t*.63,-.86),(t*.76,-.40),(t*.58,-.11))
            else:
                s.E(t*(w-.07),-.12,.21,.54,t*(14 if p == 0 else -3))
            s.S('shadow', .072,t*(w-.07),-.14,.20,.29)
        s.eyes(-.13,.30,.107)
        s.S('blob_dark', .27,0.,.32 if p==0 else .26,.24,.17)
        s.S('luma', -.12,0.,.32,.20,.13)
        s.S('luma', .06,-.29,.34,.17,.14)
        s.S('luma', .06,.29,.34,.17,.14)
    else:
        if p == 3:                          # standing canine
            s.E(-.18,.07,.64,.30,-6); s.E(.56,-.31,.26,.29,-19)
            s.E(.76,-.24,.22,.12,13); s.E(.44,-.27,.123,.25,-15)
            s.L(-.69,.02,-.94,-.46,.09,.057)
            for x, d in ((-.59,-.05),(-.41,.09),(.17,.00),(.39,.07)):
                s.L(x,.18,x+d,.65,.11,.057)
            s.S('spot_dark', .14,.85,-.26,.13,.11)
        else:                               # seated
            s.E(-.06,.37,.43,.54,-10); s.E(.16,-.32,.36,.33)
            s.E(.39,-.21,.28,.14,15); s.E(.015,-.25,.135,.28,-24)
            s.E(.26,.64,.14,.22); s.E(-.04,.67,.139,.21)
            s.S('spot_dark', .15,.53,-.21,.14,.12)
    return s


def frog(p):
    s = Stencil()
    if p < 3:                              # squat / crouch, eyes on high points
        dx = (0., .26, .39)[p]
        s.E(-.11,.14,.65,.41,(-2,-15,-20)[p])
        s.E(dx,-.12,.52 if p==0 else .41,.245,-4)
        for t in (-1,1):
            s.E(dx+t*.29,-.295+(t*.055 if p else 0),.148,.144)
            s.E(t*.55,.25,.27,.21,t*-31)
            s.L(t*.41,.37,t*.67,.57,.11,.080)
            s.L(dx+t*.23,.01,dx+t*.30,.44,.103,.062)
        s.eyes(-.289,.294,.147,x=dx,frog=True)
        s.S('olive', .14,-.07,.07,.52,.24)
        s.S('luma', -.072,dx,.13,.39,.10)
        s.S('luma', .072,dx,-.07,.40,.11)
    else:                                   # viewed from above, limbs splayed
        s.E(0.,.12,.32,.52,(p-3)*22); s.E(.0,-.34,.33,.24)
        for t in (-1,1):
            s.E(t*.38,.33,.18,.32,t*28); s.L(t*.27,-.20,t*.67,-.41,.106,.066)
            s.L(t*.67,-.41,t*.78,-.13,.065,.035); s.L(t*.44,.45,t*.77,.69,.092,.057)
        s.eyes(-.40,.23,.118,frog=True)
        s.S('olive', .19,0.,.12,.32,.47)
    return s


def horse(p):
    s = Stencil()
    if p < 3:
        s.E(-.25,.015,.665,.333,-2)
        s.E(.33,-.18,.43,.231, -61 if p!=2 else -43)
        s.E(.59,-.45,.30,.146, 23 if p!=2 else 43)
        s.E(.46,-.605,.119,.075,-64)
        s.E(.56,-.61,.124,.078,-65)
        s.L(-.81,.0,-.92,.24,.134,.080); s.L(-.92,.24,-.98,.48,.081,.051)
        for i, (x, ankle) in enumerate(((-.66,-.68),(-.43,-.34),(.11,.11),(.31,.38))):
            if p == 1 and i in (1,2): ankle += .23 if i==2 else -.20
            s.L(x,.17,x-.025,.45,.115,.070)
            s.L(x-.025,.45,ankle,.85 if i in (0,2) else .76,.066,.051)
        s.S('shadow', .12, .23,-.35,.22,.18,-60)
        s.S('luma', -.12, -.13,.08,.45,.19)
        s.S('luma', .12, -.15,.56,.24,.12)
    else:                                   # long face with a substantial muzzle
        if p == 3:
            s.E(0.,.10,.31,.62); s.E(0.,.59,.30,.27)
            s.E(-.23,-.57,.085,.225,-25); s.E(.23,-.57,.085,.225,25)
            for t in (-1,1): s.S('spot_dark', .095,t*.235,.51,.13,.115)
            s.eyes(-.18,.22,.097)
        else:
            s.E(-.26,.27,.41,.64,-21); s.E(.24,-.24,.57,.235,23)
            s.E(.69,-.01,.252,.19,26); s.E(-.23,-.59,.092,.215,-25)
            s.E(-.08,-.63,.088,.188,-8); s.S('spot_dark', .15,.69,-.047,.17,.12)
            s.S('shadow', .10,-.39,-.08,.15,.42,-28)
    return s


def ship(p):
    s = Stencil()
    if p in (0, 1, 2):
        s.P((-.97,.18),(.97,.09),(.79,.39),(-.67,.46),(-.88,.33))
        if p==2:                            # sailboat
            s.P((-.11,.17),(-.095,-.90),(-.67,.15)); s.P((.016,.12),(.012,-.63),(.52,.11))
        else:
            x = -.36 if p == 0 else .05
            s.P((x-.34,.20),(x-.30,-.15),(x+.27,-.16),(x+.44,.16))
            s.P((x-.22,-.12),(x-.17,-.38),(x+.12,-.40),(x+.24,-.14))
            s.L(x-.09,-.38,x-.095,-.66,.077,.032)
            s.S('pale', .16, x,-.11,.37,.21)
        s.S('shadow', .091,0.,.32,.57,.12)
        s.S('bar1', .17,0.,.24,.75,.16)
    else:                                   # angled bow, broad cabin
        s.P((-.84,-.11),(.65,-.29),(.94,-.04),(.62,.42),(.14,.64),(-.56,.37))
        s.P((-.60,-.11),(-.47,-.42),(.29,-.54),(.57,-.25),(.52,-.11))
        s.P((-.29,-.38),(-.25,-.63),(.20,-.69),(.29,-.49))
        s.L(-.13,-.52,-.16,-.87,.068,.032)
        s.S('pale', .21,-.04,-.36,.46,.22)
        s.S('shadow', .095,.00,.31,.49,.22)
        s.S('water', .11,-.63,.58,.27,.15)
        s.S('water', .11,.63,.62,.27,.15)
    return s


def truck(p):
    s = Stencil()
    if p<3:                                 # box trailer, rig, dump bed
        roof = (-.66,-.59,-.50)[p]
        s.P((-.96,.27),(-.95,roof),(.30,roof-.055),(.33,.22))
        s.P((.27,.27),(.27,-.48),(.63,-.48),(.78,-.21),(.94,-.10),(.94,.30))
        if p == 2:
            s.P((-.97,-.54),(.31,-.44),(.18,-.19),(-.82,-.15))
        for x in ((-.63,-.26,.66) if p != 1 else (-.73,-.49,.68)):
            s.tire(x,.27,.154)
        s.S('bar1', .136,-.4,roof,.54,.14)
        s.S('bar0', .138,-.86,-.21,.13,.40)
        s.S('shadow', .11,.49,-.25,.19,.19)
    else:                                    # square cab, head-on or quarter view
        x = 0 if p == 3 else -.20
        s.P((x-.59,.55),(x-.64,-.61),(x-.44,-.77),(x+.45,-.78),(x+.66,-.60),(x+.65,.49))
        if p == 4:
            s.P((.25,-.65),(.93,-.43),(.92,.21),(.31,.43))
        for side in (-1,1):
            s.tire(x+side*.49,.49,.148)
            s.S('spot_light', .065,x+side*.43,.23,.18,.14)
            s.S('bar0', .084,x+side*.54,-.19,.13,.37)
        s.S('shadow', .11,x,-.36,.46,.19)
        s.S('luma', -.094,x,-.33,.43,.14)
        s.S('luma', .094,x,-.045,.48,.10)
        s.S('bar1', .186,x,.12,.49,.28)
    return s


PROTOTYPES = [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]


def rotate(x, y, theta):
    c, s = math.cos(theta), math.sin(theta)
    return c*x+s*y, -s*x+c*y


def ellipse(x, y, args):
    u, v, a, b, angle = args
    m, n = rotate(x-u, y-v, math.radians(angle))
    return (m/a)**2+(n/b)**2


def polygon(x, y, points):
    inside = np.zeros(x.shape, dtype=bool)
    for (u, v), (r, s) in zip(points, points[1:]+points[:1]):
        if abs(s-v) < 1e-10:
            continue
        inside ^= ((v>y) != (s>y)) & (x < (r-u)*(y-v)/(s-v)+u)
    return inside


def blur(a, sigma):
    if sigma == 0:
        return a
    n = max(1, math.ceil(2.5*sigma))
    b = np.arange(-n,n+1)
    w = np.exp(-.5*(b/sigma)**2); w /= w.sum()
    for axis in (0,1):
        q = a.swapaxes(axis,-1)
        q = np.pad(q, [(0,0)]*(q.ndim-1)+[(n,n)])
        a = sum(t*q[...,i:i+a.shape[axis]] for i,t in enumerate(w)).swapaxes(axis,-1)
    return a


def normalized(a, aperture):
    a = a*aperture
    return a/max(.003, a.sum())


def draw_template(model, scale, dx, dy, flip, angle, channels):
    sub = 4
    x, y = grid(16*sub)
    x, y = (x/(8*sub)-dx)/scale, (y/(8*sub)-dy)/scale
    x, y = rotate(x,y,math.radians(angle)); x = flip*x
    inside = np.zeros_like(x, dtype=bool)
    for kind, args in model.shapes:
        inside |= ellipse(x, y, args) <= 1. if kind == 'e' else polygon(x, y, args)
    sy, sx = np.gradient(blur(inside.astype(np.float64), .95))
    mag = np.hypot(sy,sx)
    align = np.stack([np.abs(sx*math.cos(j*math.pi/4)+sy*math.sin(j*math.pi/4))
                      for j in range(4)])
    assign = align**7 / (1e-40+(align**7).sum(axis=0))
    def cells(a):
        return a.reshape(*a.shape[:-2], 16, sub, 16, sub).mean(axis=(-3,-1))
    edge = cells(assign*mag)
    ap = np.ones((16,16))
    ap[[0,-1]] = ap[:,[0,-1]] = 0
    ap[[1,-2]] *= .45; ap[:,[1,-2]] *= .45
    near = np.stack([blur(a,.57) for a in edge]) * ap
    near /= max(.003,near.sum())
    broad = normalized(blur(mag.reshape(16,sub,16,sub).mean(axis=(1,3)),1.57), ap)
    core = normalized(blur(cells(inside.astype(np.float64)), 1.08), ap)
    w = np.zeros((len(channels),16,16),np.float32)
    for j in range(4):
        w[channels[f'edge{j}']] = near[j]-.17*near[(j+2)%4]
    w[channels['edge']] = -.57*broad-.15*core
    for kind, strength, args in model.marks:
        soft = cells(np.exp(-.70*ellipse(x,y,args)))
        center = normalized(soft, ap)
        if kind in ('spot_dark','spot_light','blob_dark','blob_light','bar0','bar1','edge0','edge1','edge3','grainy'):
            # Prefer a localized part to the same response everywhere.
            center = center - .47*normalized(blur(soft,1.65), ap)
        w[channels[kind]] += strength*center
    return w.ravel()


def shape_summary(net, source, fmap):
    names = list(fmap)
    wanted = 'edge0 edge1 edge2 edge3 edge bar0 bar1 spot_dark spot_light blob_dark blob_light shadow luma warm grainy olive water pale'.split()
    channels = {n:i for i,n in enumerate(wanted)}
    w = np.zeros((len(wanted),len(names),1,1),np.float32)
    for k,i in channels.items(): w[i,names.index(k),0,0] = 1.
    compact = net.affine('shape_input_planes', source, w)
    size = len(wanted)*16*16

    # Comparable pose banks for all classes. Maxima over overlapping nuisance
    # transformations give tolerance to pose, crop, object scale and location.
    # The banks are produced from geometry, not from rendered training samples.
    scales = (.51, .73, .98)
    shifts = ((0.,.04),(-.19,.045),(.19,.045),(0.,-.17),(0.,.23))
    warps = ((1,0),(-1,0),(1,14),(-1,-14))
    pose_count = len(scales)*len(shifts)*len(warps)
    rows = np.empty((10*5*pose_count,size),np.float32)
    bas = np.empty(len(rows),np.float32)
    idx = 0
    nms = 'plane auto bird cat deer dog frog horse ship truck'.split()
    for i, fn in enumerate(PROTOTYPES):
        for p in range(5):
            model = fn(p)
            for z in scales:
                for dx,dy in shifts:
                    for flip, theta in warps:
                        # Reduce the crop of large laterally translated poses.
                        dx_eff = dx*(1 if z < .9 else .68)
                        rows[idx] = draw_template(model,z,dx_eff,dy,flip,theta,channels)
                        bas[idx] = -(.028 if z < .6 else 0.)-.023*(abs(dx)+abs(dy))
                        idx += 1
    match = net.affine('deformable_stencil_bank', compact, rows, bas)
    m = net.layer('deformations_by_pose', 'reshape', match, shape=[1,50,pose_count])
    m = net.pool('best_deformation_per_pose', 'maxpool', m, (1,pose_count), (1,pose_count))
    m = net.layer('poses_by_class', 'reshape', m, shape=[1,10,5])
    best = net.pool('best_pose_per_class', 'maxpool', m, (1,5), (1,5))
    mean = net.pool('mean_pose_per_class', 'avgpool', m, (1,5), (1,5))
    return best, mean, nms


def articulated_parts(net, source, fmap):
    # A complementary bank tests relations between *distinct* parts. A face
    # must supply two eyes AND the right structure below or above them; one
    # strong dark spot cannot stand in for the whole face. Tests travel together
    # in scale and translation but don't require the full silhouette to match.
    keys = 'edge0 edge1 edge2 edge3 bar0 bar1 spot_dark blob_dark luma olive shadow'.split()
    take = np.zeros((len(keys), len(fmap), 1, 1), np.float32)
    for i,k in enumerate(keys): take[i,list(fmap).index(k),0,0] = 1.
    src = net.affine('part_input_planes', source, take)
    specs = 'pointed_face muzzle_face raised_eyes jointed_legs low_wheels tall_cab'.split()
    locs = [(x,y,z,t) for z in (.40,.61,.85) for x in (-.23,0.,.23)
            for y in (-.24,-.035,.18) for t in (-8,0,8)]
    xg, yg = grid(16)
    ap = np.ones((16,16)); ap[[0,-1]] = ap[:,[0,-1]] = 0.
    mtx = np.empty((3,len(specs),len(locs),len(keys)*256), np.float32)
    for k, spec in enumerate(specs):
        for j,(dx,dy,z,t) in enumerate(locs):
            x,y = rotate((xg/8-dx)/z, (yg/8-dy)/z, math.radians(t))
            zero = np.zeros((len(keys),16,16), np.float32)

            def g(key,u,v,w,h):
                q = zero.copy()
                q[keys.index(key)] = normalized(np.exp(-.5*(((x-u)/w)**2+((y-v)/h)**2)),ap)
                return q

            def dot(u,v,w=.13,h=.11):
                return .58*g('spot_dark',u,v,w,h)+.42*g('blob_dark',u,v,w,h)

            def eye(u,v):
                return (dot(u,v)-.57*dot(u,v-.24,.16,.14)-.31*dot(u*.38,v,.16,.13)
                        +.35*(g('luma',u,v-.24,.17,.11)-g('luma',u,v,.13,.10)))

            if k < 3:
                gap, y_eye = ((.40,-.08),(.32,-.19),(.49,-.28))[k]
                a, b = eye(-gap,y_eye), eye(gap,y_eye)
                if k == 0:
                    ears = (g('edge1',-.59,-.61,.18,.25)+g('edge3',.59,-.61,.18,.25)
                            +.82*g('edge0',-.70,-.55,.16,.25)+.82*g('edge0',.70,-.55,.16,.25))
                    ears -= 1.54*g('edge0',0.,-.47,.28,.18)
                    c = .23*ears+.43*(dot(0.,.22)-.61*dot(0.,-.06,.21,.15))
                elif k == 1:
                    c = (.77*g('blob_dark',0.,.34,.20,.15)-.51*g('blob_dark',0.,.02,.24,.14)
                         +.61*(g('luma',-.34,.36,.16,.17)*.5+g('luma',.34,.36,.16,.17)*.5
                                -g('luma',0.,.34,.18,.13)))
                else:
                    c = (.64*g('bar1',0.,.12,.49,.11)-.47*g('bar1',0.,-.53,.45,.17)
                         +.24*g('olive',0.,.20,.57,.23)+.24*g('shadow',0.,.16,.41,.15))
            elif k == 3:
                # A belly over two regions of vertical edges, with space
                # between the legs rather than a solid rectangular front.
                a = (g('bar0',-.55,.47,.19,.21)-.28*g('bar0',-.09,.55,.22,.18)
                     +.24*(g('luma',-.25,.52,.17,.16)-g('luma',-.55,.51,.13,.17)))
                b = (g('bar0',.48,.47,.18,.21)-.28*g('bar0',.01,.55,.22,.18)
                     +.24*(g('luma',.17,.52,.17,.16)-g('luma',.48,.51,.13,.17)))
                c = (.6*g('edge2',0.,.18,.47,.15)+.35*g('edge2',0.,-.28,.43,.12)
                     -.38*g('edge2',0.,.60,.29,.12))
            else:
                gap = .65 if k == 4 else .49
                y_wheel = .37 if k == 4 else .49
                def wheel(u):
                    return (.63*g('blob_dark',u,y_wheel,.17,.15)+.29*g('shadow',u,y_wheel,.16,.14)
                            -.35*g('blob_dark',u,y_wheel-.32,.21,.14)
                            +.55*(g('luma',u,y_wheel-.27,.132,.11)-g('luma',u,y_wheel,.14,.15)))
                a,b = wheel(-gap),wheel(gap)
                if k == 4:
                    c = .71*g('bar1',0.,.11,.62,.18)+.29*g('bar1',0.,-.32,.55,.18)-.28*g('bar1',0.,.67,.52,.17)
                else:
                    c = (.35*g('bar0',-.61,-.16,.14,.37)+.35*g('bar0',.61,-.16,.14,.37)
                         +.37*g('bar1',0.,-.54,.48,.12)+.37*g('bar1',0.,.03,.47,.12)
                         -.28*g('bar0',0.,-.25,.21,.38))
            mtx[:,k,j] = np.stack([a.ravel(),b.ravel(),c.ravel()])
    m = net.affine('spatial_part_tests',src,mtx.reshape(-1,len(keys)*256))
    m = net.layer('local_part_triplets','reshape',m,shape=[3,len(specs),len(locs)])
    s = fields(('first','second','third'))
    h = Hinges()
    h.out_rows['pair'] = h.minimum(s['first'],s['second'])
    h.out_rows['average'] = h.copy((s['first']+s['second']+s['third'])/3)
    h.out_rows['third'] = h.copy(s['third'])
    p, f = h.emit(net,'form_part_pairs',m,3,spatial=True)
    h = Hinges()
    h.out_rows['agreement'] = .70*h.minimum(f['pair'],f['third'])+.30*h.copy(f['average'])
    p, _ = h.emit(net,'require_all_parts',p,len(f),spatial=True)
    p = net.pool('best_local_part_joint_match','maxpool',p,(1,len(locs)),(1,len(locs)))
    return p, specs


def readout(net, scene, scene_names, best, mean, classes, parts, part_names):
    # Put scene, whole-object and articulated part evidence in a common vector.
    ns, nk, na = len(scene_names), len(classes), len(part_names)
    width = ns+nk+na
    w = np.zeros((width,ns),np.float32); w[:ns] = np.eye(ns)
    a = net.affine('embed_scene_summary',scene,w)
    w = np.zeros((width,nk),np.float32); w[ns:ns+nk] = np.eye(nk)
    b = net.affine('embed_best_pose',best,.82*w)
    c = net.affine('embed_pose_support',mean,.18*w)
    w = np.zeros((width,na),np.float32); w[ns+nk:] = np.eye(na)
    d = net.affine('embed_part_relations',parts,w)
    src = net.layer('combined_object_evidence','add',[a,b,c,d])
    e = fields(scene_names+classes+part_names)
    h = Hinges()
    o = h.out_rows
    sky_top = .64*e['sky_top']+.36*e['sky_high_side']
    sky_bottom = .49*e['sky_bot']+.51*e['sky_low_side']
    water_bottom = .64*e['water_bot']+.36*e['water_low_side']
    foliage = .53*e['greens_rim']+.47*e['olive_rim']
    grass = .55*e['greens_bot']+.45*e['olive_bot']
    earthy = .44*e['warm_mid']+.32*e['olive_mid']+.24*e['bark_mid']
    smooth = .60*e['smooth_rim']+.40*e['smooth_top']
    pale_rim = .45*e['pale_top']+.34*e['pale_rim']+.21*e['pale_bot']
    edge_gap = e['edge_mid']-e['edge_rim']
    grain = .68*e['grainy_mid']+.32*e['grainy_core']
    body_tone = e['luma_mid']-e['luma_rim']

    # Soft logical evidence. All cutoffs are dimensionless hand-chosen
    # thresholds on color memberships, relative positions and edge responses.
    o['open_sky'] = h.ramp(.51*sky_top+.49*sky_bottom-.24*foliage+.08*e['smooth_bot'], .10,.54)
    o['upper_sky'] = h.ramp(.82*sky_top+.18*e['cool_top']-.12*foliage,.10,.56)
    o['all_around_sky'] = h.ramp(.76*sky_bottom+.24*e['smooth_bot']-.35*water_bottom+.20*e['pale_bot'], .13,.54)
    o['sea'] = h.ramp(water_bottom+.74*(e['water_bot']-e['water_top'])+.32*(e['luma_top']-e['luma_bot'])
                     -.62*grass-.18*e['warm_bot'],.10,.62)
    o['blue_below'] = h.ramp(.78*water_bottom+.22*e['cool_bot']-.26*sky_bottom-.32*grass,.13,.62)
    o['vegetation'] = h.ramp(.65*foliage+.35*grass,.085,.47)
    o['meadow'] = h.ramp(.80*grass+.20*e['pale_top']-.19*e['grainy_bot'],.10,.52)
    o['woodland'] = h.ramp(.42*foliage+.33*e['bark_rim']+.25*e['grainy_rim']-.19*e['pale_rim'],.10,.46)
    o['muddy'] = h.ramp(.40*earthy+.30*e['bark_rim']+.25*e['grainy_rim']+.19*e['shadow_rim']-.25*sky_top,.13,.51)
    o['pond_green'] = h.ramp(.50*e['olive_mid']+.26*e['greens_mid']+.27*e['bark_mid']-.23*e['pale_rim'], .10,.56)
    o['fur'] = h.ramp(.58*grain+.42*earthy-.19*e['axis_mid'],.125,.46)
    o['tawny'] = h.ramp(.57*e['warm_mid']+.30*e['warm_rim']+.18*e['rust_mid']-.23*e['greens_mid'],.10,.52)
    o['small_subject'] = h.ramp(.47*smooth+.47*edge_gap+.14*(e['busy_mid']-e['busy_rim']),.17,.58)
    o['studio'] = h.ramp(.65*pale_rim+.35*smooth-.44*foliage,.17,.65)
    o['bleached_rim'] = h.ramp(pale_rim-.25*e['pale_mid'],.06,.52)
    o['underbody'] = h.ramp(.70*e['shadow_band']+.37*(e['shadow_band']-e['shadow_rim'])
                            +.36*e['blob_dark_band']+.32*e['bar1_band'],.18,.61)
    o['specular'] = h.ramp(.69*e['spot_light_mid']+.68*e['blob_light_mid']+.30*e['white_mid'],.10,.46)
    o['white_subject'] = h.ramp(.84*e['white_mid']+.41*(e['white_mid']-e['white_rim'])+.18*body_tone,.015,.37)
    o['road'] = h.ramp(.33*e['neutral_bot']+.25*e['smooth_bot']+.26*e['shadow_bot']+.38*e['bar1_band']
                      -.58*grass-.49*water_bottom,.12,.65)
    o['long_verticals'] = h.ramp(.69*e['bar0_mid']+.31*e['bar0_band']+.42*(e['bar0_mid']-e['bar0_rim']),.09,.36)
    o['long_horizontals'] = h.ramp(.65*e['bar1_mid']+.35*e['bar1_band']+.29*(e['bar1_mid']-e['bar1_rim']),.09,.37)
    o['rectilinear'] = h.ramp(.75*e['axis_mid']+.40*e['corner_mid']+.43*(e['axis_mid']-e['axis_rim']),.037,.31)
    o['paint'] = h.ramp(.62*e['vivid_mid']+.28*e['rust_mid']+.36*(e['vivid_mid']-e['vivid_rim'])-.28*e['olive_mid'],.01,.46)
    o['soft_edges'] = h.ramp(.61*grain+.31*e['busy_mid']-.44*e['axis_mid'],.08,.42)
    o['luminance_horizon'] = h.ramp(e['luma_top']-e['luma_bot'],.045,.42)
    o['dark_ground'] = h.ramp(.65*e['shadow_bot']+.35*e['bark_bot']-.32*e['sky_bot'],.18,.73)
    o['dark_subject'] = h.ramp(.50*e['shadow_mid']-.26*e['shadow_rim']-.45*body_tone,.035,.41)
    o['ambient_rough'] = h.ramp(.70*e['grainy_rim']+.30*e['busy_rim'],.13,.45)
    for k in classes:
        o[k] = h.copy(e[k])
    for k in part_names:
        o[k] = h.ramp(e[k], .03, .31)
    f, d = h.emit(net,'scene_and_pose_evidence',src,width,spatial=False)

    # Mix related cues before the decision so a single weak proxy need never
    # act as a hard gate. Some class evidence is intentionally shared.
    h = Hinges(); o = h.out_rows
    for k in d:
        o[k] = h.copy(d[k])
    o['marine_layout'] = h.minimum(.08+d['sea'], .23+.76*d['blue_below']+.22*d['luminance_horizon'])
    o['land_machine'] = h.ramp(.39*d['road']+.33*d['rectilinear']+.21*d['underbody']+.17*d['paint']
                               -.30*d['woodland']-.29*d['sea'],.14,.84)
    o['blocky_vehicle'] = h.minimum(.13+d['long_verticals'], .055+d['long_horizontals'])
    o['grass_animal'] = h.minimum(d['meadow']+.14, .29+.73*d['vegetation'])
    o['fur_not_field'] = h.ramp(.57*d['fur']+.28*d['soft_edges']-.28*d['vegetation']-.32*d['rectilinear'],.038,.61)
    o['blended_wildlife'] = h.minimum(.08+d['woodland'], .08+d['muddy']+.20*d['vegetation'])
    o['perch_scene'] = h.minimum(d['small_subject']+.12, .47+.45*d['vegetation']-.23*d['road'])
    f, v = h.emit(net,'combine_soft_priors',f,len(d),spatial=False)

    # Rows express relative evidence, not calibrated probabilities. The first
    # term is the match for the class, mixed with its broad kind of setting.
    t = v
    sky, air, sea = t['upper_sky'], t['open_sky'], t['marine_layout']
    veg, field, woods = t['vegetation'], t['grass_animal'], t['woodland']
    road, machine = t['road'], t['land_machine']
    rect, block = t['rectilinear'], t['blocky_vehicle']
    studio, tiny = t['studio'], t['small_subject']
    fur, wild = t['fur_not_field'], t['blended_wildlife']
    scores = [
        3.40*t['plane']+1.59*air+.67*sky+.54*t['all_around_sky']+.67*studio+.19*tiny
        -.96*sea-.52*veg-.25*block-.27*t['muddy']-.28,
        3.40*t['auto']+1.28*machine+.50*road+.54*t['underbody']+.51*studio+.35*t['specular']
        +.19*t['paint']+.42*t['low_wheels']-.19*t['tall_cab']-.44*block-.29*sky-.49*sea-.72,
        3.50*t['bird']+1.02*t['perch_scene']+.40*tiny+.31*veg+.32*sky+.23*t['tawny']
        -.48*machine-.53*sea-.36*t['pond_green']-.22*t['white_subject']-.35,
        3.53*t['cat']+.83*fur+.42*t['tawny']+.41*t['soft_edges']+.22*t['ambient_rough']+.29*t['dark_subject']
        +.67*t['pointed_face']-.19*t['muzzle_face']-.32*field-.58*machine-.41*sky-.64*sea-.36,
        3.61*t['deer']+.65*woods+.52*veg+.53*wild+.25*t['ambient_rough']+.22*tiny+.19*t['tawny']
        +.24*t['jointed_legs']-.59*machine-.45*t['white_subject']-.29*sea-.46,
        3.41*t['dog']+.68*fur+.56*t['white_subject']+.46*t['tawny']+.31*t['soft_edges']+.22*t['bleached_rim']
        +.70*t['muzzle_face']-.12*t['pointed_face']-.55*machine-.31*woods-.44*sky-.56*sea-.43,
        3.43*t['frog']+.93*t['pond_green']+.69*t['muddy']+.38*wild+.31*t['dark_ground']+.14*t['ambient_rough']
        +.42*t['raised_eyes']-.56*machine-.70*air-.38*sea-.53,
        3.57*t['horse']+1.02*field+.40*veg+.38*t['dark_subject']+.23*t['white_subject']+.18*t['tawny']
        +.48*t['jointed_legs']-.55*machine-.29*t['muddy']-.37*sea-.41,
        3.40*t['ship']+1.85*sea+.93*t['blue_below']+.60*t['luminance_horizon']+.36*sky+.34*studio
        +.28*t['long_horizontals']-.66*veg-.44*t['all_around_sky']-.61,
        3.27*t['truck']+1.03*machine+1.02*block+.41*rect+.42*t['paint']+.21*road+.31*sky
        +.54*t['tall_cab']+.14*t['low_wheels']-.38*sea-.40*veg-.91,
    ]
    compile_affine(net,'class_scores',f,len(v),scores,False)


def main(target):
    net = Net()
    field_map, ch = image_fields(net)
    scene, names = scene_summary(net,field_map,ch)
    best, mean, classes = shape_summary(net,field_map,ch)
    parts, part_names = articulated_parts(net,field_map,ch)
    readout(net,scene,names,best,mean,classes,parts,part_names)
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':net.layers},indent=2)+'\n', encoding='utf-8')
    np.savez(out/'weights.npz',**net.weights)


if __name__ == '__main__':
    main(sys.argv[1])
