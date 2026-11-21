"""Build a data-free, geometric CIFAR-size object recognition network.

No files are read. All coordinates below describe generic drawings, not images.
There is no optimization, sampling from data, or learned readout. Normalizing a
DRAWN TEMPLATE is just an analytic matched-filter construction.

Usage: python3 build.py OUTPUT_DIRECTORY
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


LAYERS = []
WEIGHTS = {}


def op(name, kind, source, **settings):
    inputs = [source] if isinstance(source, str) else source
    LAYERS.append(dict(name=name, op=kind, inputs=inputs, **settings))
    return name


def affine(name, source, W, b=None, convolution=False, stride=1, padding=0):
    W = np.asarray(W, dtype=np.float32)
    b = np.zeros(W.shape[0], np.float32) if b is None else np.asarray(b, np.float32)
    WEIGHTS[name + '.W'] = W
    WEIGHTS[name + '.b'] = b
    if convolution:
        return op(name, 'conv', source, out=len(W), kernel=list(W.shape[-2:]),
                  stride=stride, padding=padding)
    return op(name, 'dense', source, out=len(W))


def pointwise(name, source, W, b=None):
    return affine(name, source, np.asarray(W)[:, :, None, None], b, convolution=True)


def blur(a):
    """Binomial smoothing of a drawing, with zero extension."""
    p = np.pad(a, 1)
    result = np.zeros_like(a)
    k = (1., 2., 1.)
    for y in range(3):
        for x in range(3):
            result += k[y] * k[x] * p[y:y+a.shape[0], x:x+a.shape[1]] / 16.
    return result


def valid_filter(a, k):
    h, w = a.shape
    n = len(k)
    result = np.zeros((h-n+1, w-n+1), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            result += k[y, x] * a[y:y+h-n+1, x:x+w-n+1]
    return result


SOBEL_X = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.
SOBEL_Y = SOBEL_X.T
DIRECTIONS = [SOBEL_X, SOBEL_Y, (SOBEL_X+SOBEL_Y)/math.sqrt(2.),
              (SOBEL_X-SOBEL_Y)/math.sqrt(2.)]
BINOMIAL = np.outer([1., 2., 1.], [1., 2., 1.]) / 16.


class Drawing:
    """Supersampled unions of ellipses, polygons, and rounded segments."""
    def __init__(self, scale, cx, cy, angle, flip):
        q = (np.arange(64, dtype=np.float64) + .5) / 32. - 1.
        X, Y = np.meshgrid(q-cx, q-cy)
        c, s = math.cos(angle), math.sin(angle)
        self.x = (c*X+s*Y) / scale * flip
        self.y = (-s*X+c*Y) / scale
        self.mask = np.zeros((64, 64), np.float64)
        self.dark = np.zeros_like(self.mask)

    def put(self, a, dark=False, strength=1.):
        target = self.dark if dark else self.mask
        np.maximum(target, np.asarray(a, np.float64)*strength, out=target)

    def ellipse(self, x, y, rx, ry, angle=0., dark=False, strength=1.):
        c, s = math.cos(angle), math.sin(angle)
        u, v = self.x-x, self.y-y
        self.put(((c*u+s*v)/rx)**2+((-s*u+c*v)/ry)**2 <= 1., dark, strength)

    def poly(self, points, dark=False, strength=1.):
        inside = np.zeros(self.x.shape, bool)
        xj, yj = points[-1]
        for xi, yi in points:
            if yi != yj:
                inside ^= ((yi > self.y) != (yj > self.y)) & (
                    self.x < (xj-xi)*(self.y-yi)/(yj-yi)+xi)
            xj, yj = xi, yi
        self.put(inside, dark, strength)

    def line(self, a, b, radius, dark=False, strength=1.):
        ax, ay = a
        bx, by = b
        t = np.clip(((self.x-ax)*(bx-ax)+(self.y-ay)*(by-ay)) /
                    ((bx-ax)**2+(by-ay)**2), 0., 1.)
        self.put((self.x-ax-t*(bx-ax))**2+(self.y-ay-t*(by-ay))**2 <= radius**2,
                 dark, strength)

    def finish(self):
        return tuple(a.reshape(16, 4, 16, 4).mean(axis=(1, 3))
                     for a in (self.mask, self.dark))


def wheels(d, positions, radius):
    for x, y in positions:
        d.ellipse(x, y, radius, radius)
        d.ellipse(x, y, radius*.85, radius*.85, dark=True)


def face_marks(d, x, y, spacing, nose_y, eye=.055, nose=.075):
    for sign in (-1., 1.):
        d.ellipse(x+sign*spacing, y, eye, eye*.8, dark=True)
    d.ellipse(x, nose_y, nose, nose*.7, dark=True)


def cat_ears(d, x, y, spread, size):
    for sign in (-1., 1.):
        xx = x+sign*spread
        d.poly([(xx-size*.65, y+size*.4), (xx+size*.65, y+size*.4),
                (xx+sign*size*.25, y-size)])


def draw_airplane(d, model):
    if model == 0:  # Overhead fuselage, swept wings, tailplane.
        d.ellipse(0., -.02, .095, .85)
        d.poly([(-.08, -.38), (-.89, .17), (-.89, .32), (-.06, .1),
                (.06, .1), (.89, .32), (.89, .17), (.08, -.38)])
        d.poly([(-.04, .52), (-.32, .72), (-.32, .8), (0., .7),
                (.32, .8), (.32, .72), (.04, .52)])
    elif model == 1:  # Side view with swept fin and projecting wing.
        d.ellipse(.02, .03, .89, .115)
        d.poly([(-.77, .02), (-.72, -.4), (-.57, -.4), (-.36, .02)])
        d.poly([(-.12, .04), (.26, .4), (.46, .42), (.16, .01)])
        d.poly([(-.66, .06), (-.9, .24), (-.66, .22), (-.4, .03)])
    else:  # Frontal silhouette: wing bar and central fuselage.
        d.ellipse(0., .03, .15, .14)
        d.poly([(-.94, -.12), (-.94, -.045), (-.13, .12),
                (.13, .12), (.94, -.045), (.94, -.12), (0., .025)])
        d.poly([(-.055, .02), (-.025, -.37), (.025, -.37), (.055, .02)])
        for x in (-.34, .34):
            d.ellipse(x, .1, .085, .07)


def draw_car(d, model):
    if model == 0:
        d.poly([(-.9, -.02), (-.73, -.2), (-.42, -.22), (-.22, -.47),
                (.37, -.47), (.64, -.2), (.89, -.1), (.91, .24), (-.91, .24)])
        wheels(d, [(-.57, .25), (.58, .25)], .17)
        d.poly([(-.35, -.23), (-.17, -.41), (.31, -.41), (.48, -.23)],
               dark=True, strength=.5)
    elif model == 1:
        d.poly([(-.67, .3), (-.69, -.03), (-.48, -.52), (.48, -.52),
                (.69, -.03), (.67, .3)])
        wheels(d, [(-.56, .31), (.56, .31)], .13)
        d.poly([(-.44, -.43), (.44, -.43), (.54, -.16), (-.54, -.16)],
               dark=True, strength=.55)
        d.line((-.33, .19), (.33, .19), .05, dark=True, strength=.4)
    else:  # A three-quarter silhouette, not a perspective fit to a photograph.
        d.poly([(-.87, .19), (-.84, -.12), (-.4, -.3), (-.15, -.57),
                (.4, -.51), (.73, -.17), (.86, -.02), (.8, .32), (-.35, .4)])
        wheels(d, [(-.58, .26), (.57, .28)], .16)
        d.poly([(-.28, -.3), (-.1, -.49), (.35, -.43), (.56, -.2)],
               dark=True, strength=.5)


def draw_truck(d, model):
    if model == 0:
        d.poly([(-.92, -.64), (.3, -.64), (.3, .23), (-.92, .23)])
        d.poly([(.29, -.38), (.64, -.38), (.88, -.12), (.93, .26), (.29, .26)])
        wheels(d, [(-.65, .28), (-.35, .28), (.65, .28)], .145)
        d.poly([(.39, -.3), (.61, -.3), (.77, -.12), (.39, -.12)], dark=True)
    elif model == 1:
        d.poly([(-.61, -.78), (.61, -.78), (.67, .4), (-.67, .4)])
        d.poly([(-.51, -.4), (.51, -.4), (.54, -.05), (-.54, -.05)],
               dark=True, strength=.7)
        wheels(d, [(-.56, .42), (.56, .42)], .14)
        d.line((-.36, .23), (.36, .23), .06, dark=True, strength=.4)
    else:
        d.poly([(-.9, -.55), (.13, -.74), (.46, -.51), (.44, .22), (-.9, .35)])
        d.poly([(.43, -.38), (.7, -.3), (.88, -.01), (.83, .39), (.41, .25)])
        wheels(d, [(-.58, .35), (.66, .38)], .145)
        d.poly([(.49, -.3), (.66, -.24), (.76, -.04), (.48, -.09)], dark=True)


def draw_bird(d, model):
    if model == 0:
        d.ellipse(-.1, .08, .42, .28, -.4)
        d.ellipse(.32, -.25, .22, .2)
        d.poly([(.49, -.28), (.77, -.21), (.48, -.16)])
        d.poly([(-.39, .11), (-.85, .35), (-.7, .48), (-.27, .27)])
        d.line((-.04, .29), (-.08, .54), .025)
        d.line((.14, .25), (.18, .51), .025)
        d.ellipse(.38, -.29, .038, .038, dark=True)
    elif model == 1:
        d.ellipse(-.02, .09, .32, .49, .25)
        d.ellipse(.12, -.4, .23, .22)
        d.poly([(.29, -.44), (.54, -.33), (.29, -.29)])
        d.poly([(-.22, .35), (-.39, .77), (-.16, .7), (.07, .38)])
        d.line((-.04, .49), (.03, .68), .025)
        d.ellipse(.18, -.44, .04, .04, dark=True)
    else:
        d.ellipse(0., .02, .14, .42)
        d.ellipse(0., -.37, .14, .14)
        d.poly([(-.08, -.14), (-.5, -.52), (-.92, -.68), (-.64, -.12),
                (-.22, .23)])
        d.poly([(.08, -.14), (.5, -.52), (.92, -.68), (.64, -.12), (.22, .23)])
        d.poly([(-.09, .26), (-.27, .57), (.27, .57), (.09, .26)])


def quadruped(d, species, grazing=False):
    # Relative anatomical proportions are specified from general knowledge.
    if species == 'deer':
        body_y, body_ry, leg, neck_width = -.08, .235, .71, .09
    elif species == 'horse':
        body_y, body_ry, leg, neck_width = -.06, .31, .72, .16
    elif species == 'dog':
        body_y, body_ry, leg, neck_width = .015, .235, .54, .13
    else:
        body_y, body_ry, leg, neck_width = .035, .22, .47, .13
    d.ellipse(-.12, body_y, .54, body_ry)
    for x, slant in [(-.48, -.06), (-.32, .09), (.22, -.02), (.36, .12)]:
        knee = (x+slant, (leg+body_y)*.53)
        d.line((x, body_y+.1), knee, .045 if species == 'deer' else .06)
        d.line(knee, (x+slant*.5, leg), .03 if species == 'deer' else .045)
    if species in ('deer', 'horse'):
        if grazing:
            d.line((.31, -.09), (.59, .27), neck_width)
            d.ellipse(.68, .41, .12, .26, -.3)
            d.ellipse(.67, .37, .035, .035, dark=True)
        else:
            d.line((.31, -.12), (.49, -.55), neck_width)
            d.ellipse(.65, -.57, .25 if species == 'horse' else .21, .12, .4)
            d.ellipse(.52, -.75, .055, .15, -.2)
            d.ellipse(.63, -.74, .045, .12, .2)
            d.ellipse(.64, -.61, .035, .035, dark=True)
        if species == 'horse':
            d.line((-.59, -.06), (-.76, .4), .09)
        else:
            d.line((-.58, -.06), (-.75, -.22), .045)
    elif species == 'dog':
        d.line((.29, -.06), (.47, -.33), neck_width)
        d.ellipse(.54, -.3, .22, .2)
        d.ellipse(.74, -.23, .2, .1)
        d.ellipse(.45, -.25, .10, .21, -.25)
        d.line((-.6, .02), (-.88, -.31), .06)
        d.ellipse(.63, -.35, .035, .035, dark=True)
        d.ellipse(.9, -.24, .05, .05, dark=True)
    else:
        d.ellipse(.49, -.21, .23, .21)
        cat_ears(d, .49, -.35, .14, .15)
        d.line((-.6, .03), (-.84, -.25), .05)
        d.line((-.84, -.25), (-.79, -.5), .05)
        d.ellipse(.58, -.23, .04, .035, dark=True)


def draw_cat(d, model):
    if model == 0:
        d.ellipse(0., .02, .61, .53)
        cat_ears(d, 0., -.35, .42, .37)
        face_marks(d, 0., -.06, .24, .25, .065, .075)
    elif model == 1:
        quadruped(d, 'cat')
    else:
        d.ellipse(0., .27, .37, .52)
        d.ellipse(0., -.38, .36, .31)
        cat_ears(d, 0., -.61, .25, .23)
        d.ellipse(-.21, .68, .15, .08)
        d.ellipse(.21, .68, .15, .08)
        d.line((.3, .5), (.61, .62), .065)
        face_marks(d, 0., -.4, .15, -.21, .045, .05)


def dog_head(d, x, y, size):
    d.ellipse(x, y, size*.64, size*.64)
    d.ellipse(x, y+size*.35, size*.39, size*.36)
    for s in (-1., 1.):
        d.ellipse(x+s*size*.57, y+size*.01, size*.22, size*.49, -s*.22)
    face_marks(d, x, y-size*.08, size*.26, y+size*.45, size*.065, size*.115)


def draw_dog(d, model):
    if model == 0:
        dog_head(d, 0., -.1, .86)
    elif model == 1:
        quadruped(d, 'dog')
    else:
        d.ellipse(0., .26, .39, .5)
        dog_head(d, 0., -.4, .53)
        for s in (-1., 1.):
            d.line((s*.23, .22), (s*.25, .7), .08)
            d.ellipse(s*.25, .73, .15, .09)


def draw_deer(d, model):
    if model == 0:
        quadruped(d, 'deer')
    elif model == 1:
        d.ellipse(0., .03, .29, .41)
        for x in (-.19, .19):
            d.line((x, .26), (x, .79), .035)
        d.ellipse(0., -.39, .15, .31)
        d.ellipse(-.24, -.61, .18, .075, .55)
        d.ellipse(.24, -.61, .18, .075, -.55)
        face_marks(d, 0., -.46, .095, -.2, .033, .045)
    else:
        d.ellipse(0., .02, .27, .58)
        d.ellipse(0., .44, .2, .22)
        d.ellipse(-.39, -.43, .28, .12, .45)
        d.ellipse(.39, -.43, .28, .12, -.45)
        face_marks(d, 0., -.15, .18, .48, .05, .07)
        for s in (-1., 1.):
            d.line((s*.15, -.49), (s*.31, -.87), .025)
            d.line((s*.25, -.7), (s*.48, -.81), .02)


def draw_horse(d, model):
    if model == 0:
        quadruped(d, 'horse')
    elif model == 1:
        d.ellipse(0., .07, .32, .64)
        d.ellipse(0., .51, .31, .23)
        d.ellipse(-.19, -.64, .09, .24, -.12)
        d.ellipse(.19, -.64, .09, .24, .12)
        face_marks(d, 0., -.19, .24, .54, .055, .09)
        for x in (-.17, .17):
            d.ellipse(x, .52, .055, .08, dark=True)
    else:
        quadruped(d, 'horse', grazing=True)


def draw_frog(d, model):
    if model == 0:
        d.ellipse(0., .13, .47, .45)
        d.ellipse(0., -.22, .53, .3)
        for s in (-1., 1.):
            d.ellipse(s*.34, -.46, .16, .17)
            d.ellipse(s*.34, -.48, .07, .08, dark=True)
            d.ellipse(s*.52, .32, .23, .37, s*.55)
            d.line((s*.25, -.04), (s*.66, .53), .065)
            d.line((s*.66, .53), (s*.82, .5), .045)
    elif model == 1:
        d.ellipse(0., .17, .49, .31)
        d.ellipse(0., -.13, .62, .3)
        for s in (-1., 1.):
            d.ellipse(s*.37, -.34, .17, .16)
            d.ellipse(s*.37, -.37, .075, .07, dark=True)
            d.poly([(s*.34, .08), (s*.84, .3), (s*.68, .58), (s*.3, .32)])
            d.line((s*.22, .04), (s*.41, .49), .07)
    else:
        d.ellipse(-.16, .06, .52, .32)
        d.ellipse(.36, -.1, .34, .23)
        d.ellipse(.38, -.3, .13, .13)
        d.ellipse(.41, -.32, .06, .06, dark=True)
        d.ellipse(-.47, .29, .32, .19, -.3)
        d.line((-.62, .28), (-.82, .48), .07)
        d.line((.29, .07), (.56, .44), .065)
        d.line((.56, .44), (.82, .45), .04)


def draw_ship(d, model):
    if model == 0:
        d.poly([(-.93, .16), (.93, .16), (.7, .45), (-.7, .45)])
        d.poly([(-.45, -.12), (.56, -.12), (.56, .15), (-.45, .15)])
        d.poly([(-.18, -.39), (.28, -.39), (.28, -.1), (-.18, -.1)])
        d.line((.05, -.34), (.05, -.8), .025)
        d.line((-.16, -.65), (.28, -.65), .02)
        d.line((-.31, -.01), (.43, -.01), .035, dark=True, strength=.45)
    elif model == 1:
        d.poly([(-.78, .33), (.8, .33), (.57, .55), (-.54, .55)])
        d.line((.06, .32), (.06, -.88), .025)
        d.poly([(.01, -.81), (-.65, .23), (.01, .23)])
        d.poly([(.12, -.69), (.66, .24), (.12, .24)])
    else:
        d.poly([(-.59, .04), (.59, .04), (.38, .51), (0., .72), (-.38, .51)])
        d.poly([(-.41, -.48), (.41, -.48), (.48, .04), (-.48, .04)])
        d.poly([(-.27, -.66), (.27, -.66), (.27, -.48), (-.27, -.48)])
        d.line((0., -.6), (0., -.91), .025)
        d.line((-.29, -.27), (.29, -.27), .055, dark=True, strength=.6)


DRAWERS = [draw_airplane, draw_car, draw_bird, draw_cat, draw_deer,
           draw_dog, draw_frog, draw_horse, draw_ship, draw_truck]


# Color feature names, all bounded in [0,1]. They are soft predicates, not
# estimates of class-conditional distributions.
COLOR = ['light', 'dark', 'white', 'neutral', 'blue', 'green', 'warm', 'red', 'saturated']
CI = {name: i for i, name in enumerate(COLOR)}


def color_frontend():
    # Preserve RGB, and expose positive and negative opponent differences.
    base = np.zeros((9, 3))
    base[:3] = np.eye(3)
    for j, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        base[3+2*j, a], base[3+2*j, b] = 1., -1.
        base[4+2*j] = -base[3+2*j]
    pointwise('color_basis', 'small_image', base)
    op('color_rectified', 'relu', 'color_basis')
    W = np.zeros((len(COLOR), 9))
    b = np.zeros(len(COLOR))
    lum = np.array([.299, .587, .114])
    W[CI['light'], :3] = lum
    W[CI['dark'], :3] = -lum/.4
    b[CI['dark']] = .45/.4
    W[CI['white'], :3] = lum/.35
    W[CI['white'], 3:] = -1./.35
    b[CI['white']] = -.55/.35
    W[CI['neutral'], 3:] = -1./.30
    b[CI['neutral']] = 1.
    # Blue exceeds both red and green; green exceeds both red and blue.
    W[CI['blue'], :3] = np.array([-.5, -.5, 1.])/.20
    W[CI['blue'], 3:5] = -.5/.20
    b[CI['blue']] = -.01/.20
    W[CI['green'], :3] = np.array([-.5, 1., -.5])/.16
    W[CI['green'], 5:7] = -.5/.16
    b[CI['green']] = -.01/.16
    # Warm/yellow/brown: min(red,green) exceeds blue.
    W[CI['warm'], :3] = np.array([.5, .5, -1.])/.20
    W[CI['warm'], 3:5] = -.5/.20
    b[CI['warm']] = -.02/.20
    W[CI['red'], :3] = np.array([1., -.5, -.5])/.25
    W[CI['red'], 7:9] = -.5/.25
    b[CI['red']] = -.02/.25
    W[CI['saturated'], 3:] = 1./.5
    # clip(z,0,1) = relu(z)-relu(z-1).
    pointwise('color_thresholds', 'color_rectified', np.concatenate([W, W]),
              np.concatenate([b, b-1.]))
    op('color_hinges', 'relu', 'color_thresholds')
    pointwise('color_membership', 'color_hinges',
              np.concatenate([np.eye(len(COLOR)), -np.eye(len(COLOR))], axis=1))
    op('coarse_color', 'avgpool', 'color_membership', size=[2, 2], stride=[2, 2])


# Small, deliberately weak semantic compatibility terms. In particular,
# animals are not required to have a single coat color.
FG_RULES = [
    dict(neutral=.10, white=.07, blue=-.08, green=-.18, warm=-.08, red=-.04),
    dict(neutral=.06, red=.04, green=-.12, warm=-.04),
    dict(saturated=.07, warm=.03, red=.04, green=.02, blue=.02),
    dict(neutral=.07, warm=.07, green=-.15, blue=-.10, red=-.04),
    dict(warm=.12, green=-.16, blue=-.10, red=-.05),
    dict(warm=.10, neutral=.04, green=-.14, blue=-.10, red=-.04),
    dict(green=.22, warm=.04, blue=-.12, neutral=-.02),
    dict(warm=.12, white=.03, green=-.16, blue=-.10),
    dict(neutral=.06, white=.06, green=-.16, warm=-.08, red=-.02),
    dict(neutral=.09, white=.09, green=-.14, warm=-.06, blue=-.04),
]
BG_RULES = [
    dict(blue=.14, neutral=.03, green=-.08),
    dict(blue=-.04, green=-.03),
    dict(green=.06, blue=.04),
    dict(blue=-.05, green=-.04),
    dict(green=.09, blue=-.02),
    dict(blue=-.04),
    dict(green=.03, blue=-.03),
    dict(green=.08, blue=-.02),
    dict(green=-.10),
    dict(blue=-.04, green=-.03),
]


def color_template(k, mask):
    m = mask.reshape(8, 2, 8, 2).mean(axis=(1, 3))
    inside = m / max(m.sum(), 1e-8)
    outside = (1.-m) / max((1.-m).sum(), 1e-8)
    W = np.zeros((len(COLOR), 8, 8))
    for name, weight in FG_RULES[k].items():
        W[CI[name]] += weight*inside
    for name, weight in BG_RULES[k].items():
        W[CI[name]] += weight*outside
    # Generic habitat layouts: water below vessels, grass below ungulates,
    # and relatively neutral road surfaces below wheeled vehicles.
    yy = (np.arange(8)+.5)/8.
    lower = np.broadcast_to(np.clip((yy[:, None]-.4)/.4, 0., 1.), (8, 8))
    upper = 1.-lower
    for region, rules in [
        (lower, {0: dict(blue=-.02), 1: dict(neutral=.04),
                 4: dict(green=.05), 7: dict(green=.06),
                 8: dict(blue=.21), 9: dict(neutral=.04)}.get(k, {})),
        (upper, {8: dict(blue=.04)}.get(k, {})),
    ]:
        area = region*(1.-m)
        area /= max(area.sum(), 1e-8)
        for name, weight in rules.items():
            W[CI[name]] += weight*area
    return W.ravel()


def template_bank():
    masks, contours, details, colors = [], [], [], []
    # These are nuisance transformations of generic drawings, not examples
    # selected by evaluation. Every class has the same 486 transformations.
    for k, draw in enumerate(DRAWERS):
        for model in range(3):
            for scale in (.65, .9, 1.15):
                for cy in (-.16, .04, .24):
                    for cx in (-.18, 0., .18):
                        for degrees in (-18., 0., 18.):
                            for flip in (-1., 1.):
                                d = Drawing(scale, cx, cy, math.radians(degrees), flip)
                                draw(d, model)
                                mask, dark = d.finish()
                                # Correlation with an ideal constant-colored silhouette.
                                z = mask-mask.mean()
                                z /= max(np.sqrt(np.mean(z*z)), .035)*256.
                                masks.append(z)
                                # Oriented, polarity-independent outline correlations.
                                h = np.stack([valid_filter(np.abs(valid_filter(mask, g)), BINOMIAL)
                                              for g in DIRECTIONS])
                                h -= h.mean(axis=(1, 2), keepdims=True)
                                h /= np.maximum(np.sqrt(np.mean(h*h, axis=(1, 2), keepdims=True)), .004)
                                contours.append((.55*h/(4.*12.*12.)).ravel())
                                # Local surround-minus-center detects dark eyes/wheels.
                                p = blur(blur(dark))-dark
                                p -= p.mean()
                                denom = np.maximum(p, 0.).sum()
                                if denom > 1e-5:
                                    p *= .065/denom
                                else:
                                    p[:] = 0.
                                details.append(p.ravel())
                                colors.append(color_template(k, mask))
    return tuple(np.asarray(a, np.float32) for a in (masks, contours, details, colors))


def frontend():
    op('small_image', 'avgpool', 'image', size=[2, 2], stride=[2, 2])
    # Achromatic and two fixed, biologically motivated opponent coordinates.
    pointwise('opponent_coordinates', 'small_image',
              [[.299, .587, .114], [1., -1., 0.], [-.5, -.5, 1.]])
    op('contrast_normalized', 'norm', 'opponent_coordinates')
    pointwise('weighted_contrast', 'contrast_normalized', np.diag([1., .8, .8]))
    pointwise('gray_contrast', 'contrast_normalized', [[1., 0., 0.]])
    # Stack color planes vertically so a shared square convolution applies
    # the identical silhouette filter to each opponent coordinate.
    op('stacked_planes', 'reshape', 'weighted_contrast', shape=[1, 48, 16])
    signed = np.stack([s*g for g in DIRECTIONS for s in (1., -1.)])[:, None]
    affine('signed_derivatives', 'gray_contrast', signed, convolution=True)
    op('rectified_derivatives', 'relu', 'signed_derivatives')
    W = np.zeros((4, 8))
    for j in range(4):
        W[j, 2*j:2*j+2] = 1.
    pointwise('edge_magnitudes', 'rectified_derivatives', W)
    W = np.zeros((4, 4, 3, 3))
    for j in range(4):
        W[j, j] = BINOMIAL
    affine('soft_contours', 'edge_magnitudes', W, convolution=True)
    op('normalized_contours', 'norm', 'soft_contours')
    color_frontend()


def scene_structure():
    # Coherent long edges favor manufactured forms; fine, incoherent texture
    # is weaker evidence for them. These are signed derivative formulas.
    horizontal = np.zeros((7, 7))
    horizontal[2, :] = -1./14.
    horizontal[4, :] = 1./14.
    signed = np.stack([horizontal, -horizontal, horizontal.T, -horizontal.T])[:, None]
    affine('long_signed_edges', 'gray_contrast', signed, convolution=True)
    op('long_edge_rectification', 'relu', 'long_signed_edges')
    pointwise('long_edge_magnitudes', 'long_edge_rectification', [[1, 1, 0, 0], [0, 0, 1, 1]])
    op('long_edge_regions', 'avgpool', 'long_edge_magnitudes', size=[5, 5], stride=[5, 5])
    op('short_edge_regions', 'avgpool', 'edge_magnitudes', size=[7, 7], stride=[7, 7])
    longW = np.zeros((10, 2, 2, 2))
    shortW = np.zeros((10, 4, 2, 2))
    for k, strength in enumerate([.12, .22, -.07, -.04, -.04, -.04, -.03, -.04, .14, .26]):
        longW[k] = strength/8.
        shortW[k, :2] = -.45*strength/8.
    # A vessel's waterline is specifically a lower horizontal contour.
    longW[8, 0, 1, :] += .09/2.
    # Clear upper surroundings support aircraft and ships, not rough foliage.
    shortW[0, :, 0, :] -= .06/8.
    shortW[8, :, 0, :] -= .03/8.
    affine('long_structure_scores', 'long_edge_regions', longW.reshape(10, -1))
    affine('short_structure_scores', 'short_edge_regions', shortW.reshape(10, -1))


def build(destination):
    frontend()
    masks, contours, details, colors = template_bank()
    T = len(masks)
    assert T == 10*486
    # Interleaved positive and negative filters allow either object polarity.
    shapeW = np.stack([masks, -masks], axis=1).reshape(2*T, 1, 16, 16)*.65
    affine('silhouette_correlations', 'stacked_planes', shapeW,
           convolution=True, stride=16)
    op('silhouette_alternatives', 'reshape', 'silhouette_correlations', shape=[1, T, 6])
    op('best_color_contrast', 'maxpool', 'silhouette_alternatives', size=[1, 6], stride=[1, 6])
    op('silhouette_scores', 'reshape', 'best_color_contrast', shape=[T])
    affine('contour_scores', 'normalized_contours', contours)
    affine('landmark_scores', 'gray_contrast', details)
    affine('semantic_color_scores', 'coarse_color', colors)
    op('drawing_scores', 'add', ['silhouette_scores', 'contour_scores',
                               'landmark_scores', 'semantic_color_scores'])
    op('class_pose_groups', 'reshape', 'drawing_scores', shape=[10, 1, 486])
    op('best_pose_per_class', 'maxpool', 'class_pose_groups', size=[1, 486], stride=[1, 486])
    op('object_scores', 'reshape', 'best_pose_per_class', shape=[10])
    scene_structure()
    op('scores', 'add', ['object_scores', 'long_structure_scores', 'short_structure_scores'])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/'network.json').open('w') as f:
        json.dump({'layers': LAYERS}, f, indent=2)
        f.write('\n')
    np.savez(destination/'weights.npz', **WEIGHTS)
    print('Wrote', len(LAYERS), 'layers and', sum(w.size for w in WEIGHTS.values()), 'float32 weights.')


if __name__ == '__main__':
    build(sys.argv[1])
