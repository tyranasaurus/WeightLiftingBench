#!/usr/bin/env python3
"""Data-free, hand-designed CIFAR-sized object recognizer.

There is no optimizer, data reader, label statistic, or learned coefficient here.
Shapes are analytic drawings in dimensionless object coordinates, not traces.
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
        self.shapes = {'image': (3, 32, 32)}

    def op(self, name, op, source, **kw):
        ins = [source] if isinstance(source, str) else source
        self.layers.append(dict(name=name, op=op, inputs=ins, **kw))
        s = self.shapes[ins[0]]
        if op == 'conv':
            k = kw['kernel'][0]
            st, p = kw.get('stride', 1), kw.get('padding', 0)
            shape = (kw['out'], (s[1]+2*p-k)//st+1, (s[2]+2*p-k)//st+1)
        elif op == 'dense':
            shape = (kw['out'],)
        elif op in ('avgpool', 'maxpool'):
            shape = (s[0], (s[1]-kw['size'][0])//kw['stride'][0]+1,
                     (s[2]-kw['size'][1])//kw['stride'][1]+1)
        elif op == 'reshape':
            shape = tuple(kw['shape'])
        else:
            shape = s
        self.shapes[name] = shape
        return name

    def conv(self, name, source, W, b=None, stride=1, padding=0):
        W = np.asarray(W, dtype=np.float32)
        self.weights[name+'.W'] = W
        self.weights[name+'.b'] = np.zeros(W.shape[0], np.float32) if b is None else np.asarray(b, np.float32)
        return self.op(name, 'conv', source, out=W.shape[0], kernel=list(W.shape[-2:]), stride=stride, padding=padding)

    def dense(self, name, source, W, b=None):
        W = np.asarray(W, dtype=np.float32)
        self.weights[name+'.W'] = W
        self.weights[name+'.b'] = np.zeros(W.shape[0], np.float32) if b is None else np.asarray(b, np.float32)
        return self.op(name, 'dense', source, out=W.shape[0])

    def relu(self, name, source):
        return self.op(name, 'relu', source)

    def pool(self, name, source, size, stride=None, kind='avg'):
        return self.op(name, kind+'pool', source, size=list(size), stride=list(size if stride is None else stride))


def gaussian_kernel(size, sigma):
    t = np.arange(size, dtype=np.float64) - (size-1)/2
    g = np.exp(-t*t/(2*sigma*sigma))
    return g/g.sum()


def blur(a, sigma):
    """Small deterministic separable Gaussian, used only to define shape weights."""
    r = int(math.ceil(3*sigma))
    g = gaussian_kernel(2*r+1, sigma)
    out = np.zeros_like(a, dtype=np.float64)
    p = np.pad(a, ((r, r), (0, 0)))
    for i, q in enumerate(g):
        out += q*p[i:i+a.shape[0]]
    p = np.pad(out, ((0, 0), (r, r)))
    out = np.zeros_like(a, dtype=np.float64)
    for i, q in enumerate(g):
        out += q*p[:, i:i+a.shape[1]]
    return out


class Drawing:
    """A union of ellipses, polygons and capsules, rasterized analytically."""
    def __init__(self, size=13, scale=1, angle=0, mirror=False):
        self.size = size
        n = size*4
        q = ((np.arange(n)+.5)/n*2-1)*1.2/scale
        xx, yy = np.meshgrid(q, q)
        c, s = math.cos(angle), math.sin(angle)
        self.x = c*xx+s*yy
        self.y = -s*xx+c*yy
        if mirror:
            self.x = -self.x
        self.mask = np.zeros((n, n), dtype=np.float64)
        self.paint = self.mask.copy()

    def put(self, test, value=.75, inside=False):
        if inside:
            test = test & (self.mask > 0)
        else:
            self.mask[test] = 1
        self.paint[test] = value

    def ellipse(self, x, y, rx, ry, value=.75, angle=0, inside=False):
        c, s = math.cos(angle), math.sin(angle)
        dx, dy = self.x-x, self.y-y
        xx, yy = c*dx+s*dy, -s*dx+c*dy
        self.put((xx/rx)**2+(yy/ry)**2 <= 1, value, inside)

    def polygon(self, points, value=.75, inside=False):
        x, y = self.x, self.y
        test = np.zeros_like(x, dtype=bool)
        previous = points[-1]
        for current in points:
            x0, y0 = previous
            x1, y1 = current
            if y1 != y0:
                test ^= ((y0 > y) != (y1 > y)) & (x < (x1-x0)*(y-y0)/(y1-y0)+x0)
            previous = current
        self.put(test, value, inside)

    def rect(self, x0, y0, x1, y1, value=.75, inside=False):
        self.put((self.x >= x0) & (self.x <= x1) & (self.y >= y0) & (self.y <= y1), value, inside)

    def line(self, x0, y0, x1, y1, width, value=.75, inside=False):
        t = np.clip(((self.x-x0)*(x1-x0)+(self.y-y0)*(y1-y0))/((x1-x0)**2+(y1-y0)**2+1e-12), 0, 1)
        self.put((self.x-(x0+t*(x1-x0)))**2+(self.y-(y0+t*(y1-y0)))**2 <= width**2, value, inside)

    def finish(self):
        k = self.size
        return (self.mask.reshape(k, 4, k, 4).mean((1, 3)),
                self.paint.reshape(k, 4, k, 4).mean((1, 3)))


def car(d, view):
    if view in (0, 1):
        d.polygon([(-.96,-.05),(-.80,-.20),(-.45,-.25),(-.25,-.62),(.35,-.62),(.67,-.26),(.92,-.15),(.97,.27),(-.97,.27)])
        d.polygon([(-.41,-.25),(-.20,-.55),(.28,-.55),(.55,-.25)], .18, True)
        d.line(.05,-.53,.06,-.23,.025,.65,True)
        for x in (-.59,.60):
            d.ellipse(x,.26,.21,.22,.12)
            d.ellipse(x,.26,.095,.10,.5,inside=True)
        d.rect(-.91,-.02,-.74,.06,.95,True)
        d.rect(.75,-.02,.91,.08,.95,True)
    elif view == 2:
        d.polygon([(-.78,.53),(-.85,-.13),(-.55,-.73),(.55,-.73),(.85,-.13),(.78,.53)])
        d.polygon([(-.65,-.12),(-.45,-.62),(.45,-.62),(.65,-.12)], .16, True)
        d.rect(-.72,.16,-.35,.32,.97,True)
        d.rect(.35,.16,.72,.32,.97,True)
        d.rect(-.31,.22,.31,.40,.18,True)
        d.rect(-.77,.45,-.55,.65,.12)
        d.rect(.55,.45,.77,.65,.12)
    else:
        d.polygon([(-.96,.24),(-.89,-.18),(-.47,-.43),(-.26,-.76),(.32,-.70),(.67,-.33),(.87,-.13),(.86,.42),(.14,.62)])
        d.polygon([(-.45,-.37),(-.20,-.65),(.25,-.60),(.50,-.30)],.17,True)
        d.polygon([(.34,-.57),(.59,-.29),(.12,-.05),(-.04,-.36)],.22,True)
        d.ellipse(-.61,.22,.16,.23,.10)
        d.ellipse(.54,.43,.18,.24,.10)
        d.line(.22,.22,.76,.02,.045,.97,True)


def truck(d, view):
    if view in (0, 1):
        d.rect(-.98,-.78,.34,.24,.82)
        d.polygon([(.36,-.46),(.73,-.46),(.96,-.15),(.96,.27),(.36,.27)], .6)
        d.polygon([(.44,-.38),(.68,-.38),(.83,-.16),(.44,-.16)],.16,True)
        d.line(-.94,.27,.93,.27,.055,.18)
        for x in (-.73,-.39,.68):
            d.ellipse(x,.34,.17,.19,.10)
            d.ellipse(x,.34,.07,.08,.48,inside=True)
        d.line(-.25,-.72,-.25,.18,.012,.57,True)
    elif view == 2:
        d.rect(-.65,-.94,.65,.61,.8)
        d.rect(-.55,-.69,.55,-.20,.17,True)
        d.line(0,-.68,0,-.21,.025,.65,True)
        d.rect(-.45,.04,.45,.40,.22,True)
        for y in (.11,.22,.33):
            d.line(-.43,y,.43,y,.022,.55,True)
        d.rect(-.63,.42,-.39,.55,.99,True)
        d.rect(.39,.42,.63,.55,.99,True)
        d.rect(-.67,.59,-.44,.80,.12)
        d.rect(.44,.59,.67,.80,.12)
    else:
        d.polygon([(-.98,-.58),(-.16,-.94),(.53,-.62),(.53,.19),(-.22,.49),(-.98,.17)],.82)
        d.polygon([(.06,-.27),(.61,-.47),(.91,-.16),(.91,.50),(.35,.75),(.03,.36)],.62)
        d.polygon([(.29,-.24),(.59,-.35),(.78,-.13),(.43,.01)],.16,True)
        d.rect(.13,-.12,.27,.10,.22,True)
        for x,y in [(-.65,.30),(-.33,.45),(.60,.64)]:
            d.ellipse(x,y,.13,.19,.10)


def airplane(d, view):
    if view == 0:
        d.ellipse(0,-.04,.11,.99,.8)
        d.polygon([(-.10,-.33),(-1.03,.21),(-.98,.34),(-.08,.11),(.08,.11),(.98,.34),(1.03,.21),(.10,-.33)],.76)
        d.polygon([(-.07,.57),(-.46,.83),(-.43,.93),(0,.82),(.43,.93),(.46,.83),(.07,.57)],.65)
        d.ellipse(0,-.72,.065,.13,.22,inside=True)
    elif view in (1, 2):
        d.ellipse(0,.02,1.02,.105,.83)
        d.polygon([(-.87,0),(-.86,-.44),(-.68,-.43),(-.48,0)],.7)
        d.polygon([(-.14,-.02),(.18,-.03),(-.46,.48),(-.68,.45)],.72)
        d.polygon([(-.15,-.03),(.11,-.04),(-.39,-.24),(-.60,-.22)],.65)
        d.ellipse(.76,-.015,.13,.045,.18,inside=True)
        d.ellipse(-.23,.17,.16,.075,.44)
        d.line(-.66,.02,.64,.02,.015,.35,True)
    else:
        d.line(-1.05,.03,1.05,.03,.045,.82)
        d.ellipse(0,.06,.17,.18,.75)
        d.line(0,-.11,0,-.51,.045,.65)
        d.line(-.35,-.27,.35,-.27,.035,.67)
        for x in (-.43,.43):
            d.ellipse(x,.13,.105,.11,.3)


def bird(d, view):
    if view in (0,1):
        d.ellipse(-.12,.16,.42,.57,.65,angle=-.35)
        d.ellipse(.19,-.43,.25,.25,.75)
        d.polygon([(.37,-.45),(.70,-.32),(.37,-.27)],.45)
        d.polygon([(-.32,.32),(-.82,.75),(-.61,.83),(-.07,.47)],.43)
        d.ellipse(-.17,.17,.25,.36,.40,angle=-.4,inside=True)
        d.ellipse(.25,-.47,.045,.045,.1,inside=True)
        for x in (-.17,.08):
            d.line(x,.61,x-.03,.97,.023,.40)
            d.line(x-.12,.97,x+.10,.97,.02,.4)
    elif view == 2:
        d.ellipse(-.17,.42,.59,.30,.7)
        d.line(.25,.35,.38,-.50,.075,.66)
        d.ellipse(.47,-.58,.22,.16,.8)
        d.polygon([(.62,-.61),(1.0,-.51),(.61,-.47)],.4)
        d.line(-.18,.64,-.20,1.0,.022,.35)
        d.line(.10,.64,.13,1.0,.022,.35)
    else:
        d.ellipse(.05,0,.36,.12,.65)
        d.polygon([(-.20,-.02),(-.88,-.76),(-.64,-.82),(.20,-.04)],.75)
        d.polygon([(-.14,.03),(-.85,.68),(-.59,.78),(.23,.05)],.7)
        d.ellipse(.39,-.04,.15,.12,.72)
        d.polygon([(.51,-.08),(.75,-.02),(.51,.02)],.4)
        d.polygon([(-.23,-.06),(-.71,-.22),(-.69,.19),(-.23,.06)],.5)


def cat(d, view):
    if view in (0,1):
        wide = 1.0 if view == 0 else 1.12
        d.ellipse(0,.11,.65*wide,.65,.73)
        d.polygon([(-.59,-.12),(-.60,-.92),(-.16,-.49)],.68)
        d.polygon([(.59,-.12),(.60,-.92),(.16,-.49)],.68)
        d.polygon([(-.49,-.41),(-.51,-.75),(-.30,-.47)],.35,True)
        d.polygon([(.49,-.41),(.51,-.75),(.30,-.47)],.35,True)
        for x in (-.28,.28):
            d.ellipse(x,-.04,.14,.085,.24,angle=-math.copysign(.12,x),inside=True)
            d.ellipse(x,-.04,.035,.075,.07,inside=True)
        d.ellipse(-.14,.33,.22,.17,.90,inside=True)
        d.ellipse(.14,.33,.22,.17,.90,inside=True)
        d.polygon([(-.10,.20),(.10,.20),(0,.32)],.1,True)
        d.line(0,.31,0,.43,.016,.15,True)
    else:
        d.ellipse(-.14,.13,.60,.30,.7)
        d.ellipse(.48,-.20,.28,.27,.75)
        d.polygon([(.25,-.29),(.24,-.68),(.49,-.40)],.65)
        d.polygon([(.51,-.38),(.67,-.68),(.71,-.18)],.67)
        for x in (-.47,-.28,.29,.50):
            d.line(x,.25,x+.02,.78,.055,.6)
        d.line(-.65,.16,-.92,-.12,.055,.65)
        d.line(-.92,-.12,-.95,-.64,.05,.65)
        d.ellipse(.57,-.24,.04,.04,.12,inside=True)


def dog(d, view):
    if view in (0,1):
        d.ellipse(0,-.04,.55,.68,.72)
        if view == 0:
            for x in (-.54,.54):
                d.ellipse(x,-.15,.24,.53,.4,angle=-math.copysign(.2,x))
        else:
            d.polygon([(-.52,-.28),(-.55,-1.0),(-.17,-.57)],.48)
            d.polygon([(.52,-.28),(.55,-1.0),(.17,-.57)],.48)
        d.ellipse(0,.33,.35,.34,.9,inside=True)
        for x in (-.25,.25):
            d.ellipse(x,-.16,.09,.075,.1,inside=True)
        d.ellipse(0,.20,.15,.11,.08,inside=True)
        d.line(0,.28,0,.47,.02,.18,True)
        d.line(-.17,.48,.17,.48,.022,.24,True)
    else:
        d.ellipse(-.10,.05,.60,.31,.68)
        d.ellipse(.48,-.27,.28,.31,.72)
        d.ellipse(.73,-.17,.25,.14,.62)
        d.ellipse(.34,-.21,.14,.26,.35,angle=-.3)
        for x in (-.47,-.27,.32,.51):
            d.line(x,.20,x-.025,.85,.065,.62)
        d.line(-.65,.02,-.98,-.32,.065,.65)
        d.ellipse(.92,-.17,.05,.06,.12,inside=True)
        d.ellipse(.56,-.34,.04,.04,.1,inside=True)


def ungulate(d, view, horse=False):
    if view in (0,1):
        d.ellipse(-.15,-.05,.63,.34 if horse else .27,.72)
        if horse:
            d.polygon([(.28,-.10),(.44,-.71),(.66,-.87),(.82,-.62),(.63,.04)],.67)
            d.ellipse(.74,-.64,.17,.29,.63,angle=-.25)
            d.line(-.71,-.12,-.88,.57,.065,.35)
            legwidth=.052
        else:
            d.polygon([(.30,-.12),(.52,-.72),(.67,-.72),(.56,.07)],.7)
            d.ellipse(.69,-.74,.25,.13,.76,angle=.13)
            d.ellipse(.53,-.91,.08,.18,.64,angle=-.5)
            d.ellipse(.71,-.93,.07,.17,.64,angle=.4)
            d.line(-.72,-.12,-.88,-.28,.035,.84)
            legwidth=.033
        for j,x in enumerate((-.54,-.34,.26,.44)):
            foot=x+(.07 if j%2 else -.06)
            d.line(x,.12,x-.015,.48,legwidth,.66)
            d.line(x-.015,.48,foot,.94,legwidth*.8,.6)
            d.line(foot-.035,.95,foot+.045,.95,legwidth,.23)
        if horse:
            d.ellipse(.60,-.96,.045,.16,.45,angle=-.1)
            d.ellipse(.76,-.92,.042,.14,.45,angle=.1)
        d.ellipse(.81,-.75 if not horse else -.67,.027,.028,.1,inside=True)
    elif view == 2:
        # Long equine muzzle versus the shorter, narrow deer face.
        d.ellipse(0,.02,.34 if horse else .31,.77 if horse else .52,.72)
        d.ellipse(0,.62 if horse else .36,.30,.24,.55,inside=True)
        for x in (-.22,.22):
            d.ellipse(x,-.25 if horse else -.15,.055,.05,.10,inside=True)
            d.ellipse(x,-.83,.085,.28,.60,angle=-math.copysign(.25,x))
        for x in (-.14,.14):
            d.ellipse(x,.63 if horse else .39,.055,.042,.13,inside=True)
        if not horse:
            for sign in (-1,1):
                d.line(sign*.15,-.50,sign*.35,-.98,.025,.45)
                d.line(sign*.27,-.79,sign*.56,-.91,.02,.45)
    else:
        d.ellipse(0,.11,.39,.50,.7)
        d.ellipse(.05,-.43,.22,.42,.72)
        d.ellipse(.09,-.62,.20,.27,.66)
        for x in (-.24,.24):
            d.line(x,.37,x,.98,.05 if horse else .033,.58)
        for x in (-.07,.23):
            d.ellipse(x,-.92,.055,.19,.57)
        if not horse:
            for sign in (-1,1):
                d.line(sign*.13,-.78,sign*.42,-1.08,.022,.45)


def frog(d, view):
    if view in (0,3):
        d.ellipse(0,.18,.54,.47,.65)
        for x in (-.61,.61):
            d.ellipse(x,.35,.33,.29,.65,angle=-math.copysign(.4,x))
            d.line(x,.53,x*1.45,.68,.065,.62)
        d.ellipse(0,-.22,.64,.36,.74)
        for x in (-.39,.39):
            d.ellipse(x,-.51,.16,.16,.76)
            d.ellipse(x,-.53,.075,.085,.12,inside=True)
        d.line(-.45,-.10,.45,-.10,.025,.20,True)
        for sign in (-1,1):
            d.line(sign*.36,.06,sign*.42,.63,.055,.60)
            d.line(sign*.42,.63,sign*.62,.76,.05,.58)
        if view == 3:
            d.line(-.62,.35,-1.05,-.10,.075,.6)
            d.line(.62,.35,1.05,-.10,.075,.6)
    else:
        d.ellipse(-.20,.12,.64,.42,.67)
        d.ellipse(.47,-.12,.42,.27,.74)
        d.ellipse(.48,-.37,.15,.15,.78)
        d.ellipse(.50,-.39,.065,.07,.1,inside=True)
        d.ellipse(-.52,.39,.38,.23,.6,angle=-.4)
        d.line(-.53,.52,-.94,.64,.075,.58)
        d.line(.49,.09,.33,.62,.055,.61)
        d.line(.33,.62,.74,.68,.05,.62)
        d.line(.46,.02,.85,.01,.022,.2,True)


def ship(d, view):
    if view in (0,1):
        d.polygon([(-1.03,.22),(.98,.22),(.70,.57),(-.80,.57)],.64)
        d.rect(-.48,-.12,.52,.22,.88)
        d.rect(-.24,-.43,.26,-.12,.9)
        d.line(.02,-.43,.02,-.92,.025,.55)
        d.rect(-.10,-.64,.08,-.42,.35)
        d.line(-.40,.00,.44,.00,.035,.22,True)
        d.line(-.95,.32,.84,.32,.035,.22,True)
    elif view == 2:
        d.polygon([(-.84,.48),(.89,.48),(.65,.71),(-.63,.71)],.60)
        d.line(.06,-1.02,.06,.48,.025,.5)
        d.polygon([(-.03,-.93),(-.77,.36),(-.03,.36)],.9)
        d.polygon([(.14,-.66),(.75,.35),(.14,.35)],.8)
    else:
        d.polygon([(-1.05,.22),(1.04,.22),(.77,.58),(-.91,.58)],.5)
        d.rect(-.79,-.15,.68,.22,.9)
        d.rect(-.62,-.37,.49,-.15,.88)
        d.rect(-.30,-.55,.15,-.37,.83)
        d.rect(-.12,-.78,.04,-.55,.35)
        for y in (-.22,-.02,.17):
            d.line(-.65,y,.53,y,.025,.2,True)


def geometric_filter(cls, view, scale, angle):
    d = Drawing(scale=scale, angle=angle, mirror=(view == 1 or (cls in (3,5,6) and view == 3)))
    if cls == 0:
        airplane(d, view)
    elif cls == 1:
        car(d, view)
    elif cls == 2:
        bird(d, view)
    elif cls == 3:
        cat(d, view)
    elif cls == 4:
        ungulate(d, view, False)
    elif cls == 5:
        dog(d, view)
    elif cls == 6:
        frog(d, view)
    elif cls == 7:
        ungulate(d, view, True)
    elif cls == 8:
        ship(d, view)
    else:
        truck(d, view)
    mask, paint = d.finish()
    # Outer boundaries and internal parts both contribute. There is no feature
    # extraction from any photograph: these derivatives are of our own formulas.
    paint = blur(paint, .45)
    gy, gx = np.gradient(paint)
    magnitude = np.hypot(gx, gy)
    theta = np.mod(np.arctan2(gy,gx), math.pi)*4/math.pi
    base = np.floor(theta).astype(int) % 4
    fraction = theta-np.floor(theta)
    contour = np.sqrt(magnitude)
    contour /= contour.sum()+1e-12
    W = np.zeros((5,13,13), np.float64)
    for j in range(4):
        orientation = (base == j)*(1-fraction) + ((base+1)%4 == j)*fraction
        # Subtract the orientation-averaged response so random clutter is not
        # rewarded merely for containing many edges.
        W[j] = 1.2*blur(contour*(orientation-.26), .45)
    fg = blur(mask,.55)
    ring = np.maximum(blur(mask,1.6)-fg,0)
    # A balanced center/surround contrast, with zero DC response.
    W[4] = .10*(fg/(fg.sum()+1e-12)-ring/(ring.sum()+1e-12))
    return W


def add_color_maps(net, source):
    # Unit-interval ramps. Color differences are in ordinary RGB units.
    L = np.array([.299,.587,.114])
    primitives = []
    def ramp(name, a, b):
        primitives.append((name, np.asarray(a,dtype=float),float(b)))
    ramp('light',L,0)
    ramp('g_r',np.array([-1,1,0])/.11,-.005/.11)
    ramp('g_b',np.array([0,1,-1])/.11,-.005/.11)
    ramp('b_r',np.array([-1,0,1])/.12,-.005/.12)
    ramp('b_g',np.array([0,-1,1])/.12,.025/.12)
    ramp('cyan_g',np.array([-1,1,0])/.10,-.005/.10)
    ramp('cyan_b',np.array([-1,0,1])/.14,-.005/.14)
    ramp('warm',np.array([1,0,-1])/.20,-.025/.20)
    ramp('tan_g',np.array([0,1,-1])/.14,-.005/.14)
    ramp('red',np.array([1,-.5,-.5])/.20,-.045/.20)
    ramp('dark',-L/.30,.42/.30)
    ramp('black',-L/.16,.21/.16)
    ramp('white',L/.30,-.55/.30)
    ramp('yellow',np.array([.5,.5,-1])/.16,-.04/.16)
    for name,a in [('rg',[1,-1,0]),('gb',[0,1,-1]),('br',[-1,0,1])]:
        ramp(name+'p',a,0)
        ramp(name+'m',-np.array(a),0)
    P = {name:i for i,(name,a,b) in enumerate(primitives)}
    W = np.zeros((len(P)*2,3,1,1))
    b = np.zeros(len(P)*2)
    for j,(name,a,c) in enumerate(primitives):
        W[2*j,:,0,0]=W[2*j+1,:,0,0]=a
        b[2*j],b[2*j+1]=c,c-1
    h = net.conv('color_ramp_bounds',source,W,b)
    h = net.relu('color_ramp_relu',h)
    colors=[]
    def color(name, terms, bias=0):
        colors.append((name,terms,bias))
    color('light',{'light':1})
    color('leaf',{'g_r':1/1.4,'g_b':1/1.4},-.6/1.4)
    color('sky',{'b_r':1/1.5,'b_g':1/1.5},-.5/1.5)
    color('water',{'cyan_g':1/1.6,'cyan_b':1/1.6},-.4/1.6)
    color('warm',{'warm':1})
    color('tan',{'warm':1/1.65,'tan_g':1/1.65,'red':-.8/1.65},-.35/1.65)
    color('red',{'red':1})
    color('dark',{'dark':1})
    color('black',{'black':1})
    color('white',{'white':1})
    color('yellow',{'yellow':1})
    diffs={n:1 for n in ('rgp','rgm','gbp','gbm','brp','brm')}
    color('neutral',{n:-.5/.16 for n in diffs},.18/.16)
    color('saturated',{n:.5/.4 for n in diffs})
    W = np.zeros((len(colors)*2,len(P)*2,1,1))
    b = np.zeros(len(colors)*2)
    for j,(name,terms,c) in enumerate(colors):
        for key,v in terms.items():
            W[2*j,2*P[key],0,0] += v
            W[2*j,2*P[key]+1,0,0] -= v
        W[2*j+1] = W[2*j]
        b[2*j],b[2*j+1] = c,c-1
    h=net.conv('semantic_color_bounds',h,W,b)
    h=net.relu('semantic_color_relu',h)
    W=np.zeros((len(colors),len(colors)*2,1,1))
    for j in range(len(colors)):
        W[j,2*j,0,0],W[j,2*j+1,0,0]=1,-1
    h=net.conv('semantic_colors',h,W)
    return h,[c[0] for c in colors]


def add_edges(net, gray):
    # Signed derivatives, center-surround detail, and elongated straight edges.
    t=np.arange(9,dtype=float)-4
    g=gaussian_kernel(9,.9)
    dg=t*g
    dg/=dg[dg>0].sum()
    kx=np.outer(g,dg)
    ky=kx.T
    kernels=[kx,ky,(kx+ky)/math.sqrt(2),(ky-kx)/math.sqrt(2)]
    for small,large in ((.6,1.35),(1.15,2.4)):
        gs,gl=gaussian_kernel(9,small),gaussian_kernel(9,large)
        k=np.outer(gs,gs)-np.outer(gl,gl)
        k/=k[k>0].sum()
        kernels.append(k)
    along=gaussian_kernel(9,2.5)
    kernels.extend([np.outer(dg,along),np.outer(along,dg)])
    W=np.stack([sign*k for k in kernels for sign in (1,-1)])[:,None]
    h=net.conv('analytic_derivatives',gray,W,padding=4)
    h=net.relu('signed_edge_relu',h)
    W=np.zeros((8,16,1,1))
    for j in range(8):
        W[j,2*j:2*j+2,0,0]=1
    h=net.conv('unsigned_edges',h,W)
    h=net.pool('edge_cells',h,(2,2))
    return h,['vertical','horizontal','diagonal_down','diagonal_up','fine','coarse','bar_h','bar_v']


def spatial_regions(size=16):
    y,x=np.meshgrid((np.arange(size)+.5)/size,(np.arange(size)+.5)/size,indexing='ij')
    center=np.exp(-.5*(((x-.5)/.23)**2+((y-.5)/.25)**2))
    regions={
        'all':np.ones_like(x),
        'top':(y<.34).astype(float),
        'bottom':(y>.66).astype(float),
        'center':center,
        'upper':np.exp(-.5*(((x-.5)/.23)**2+((y-.30)/.16)**2)),
        'lower':np.exp(-.5*(((x-.5)/.26)**2+((y-.70)/.16)**2)),
        'background':1-center/center.max(),
        'wide':np.exp(-.5*(((x-.5)/.35)**2+((y-.57)/.18)**2)),
    }
    return {name:m/m.sum() for name,m in regions.items()}


def add_context(net, colors, color_names, edges, edge_names):
    regions=spatial_regions()
    specifications=[(c,r) for c in color_names+edge_names for r in regions]
    index={c+'.'+r:j for j,(c,r) in enumerate(specifications)}
    N=len(index)
    sources=[]
    for source,names,layer in [(colors,color_names,'color_regions'),(edges,edge_names,'edge_regions')]:
        W=np.zeros((N,len(names),16,16))
        for k,c in enumerate(names):
            for r,mask in regions.items():
                W[index[c+'.'+r],k]=mask
        sources.append(net.dense(layer,source,W.reshape(N,-1)))
    features=net.op('regional_features','add',sources)
    # Four alternative scene explanations per class. The maximum, rather than
    # their sum, avoids counting correlated color cues as independent evidence.
    rules=[]
    def rule(c, terms, bias=0):
        rules.append((c,terms,bias))
    rule(0,{'sky.background':.85,'leaf.all':-.25,'warm.all':-.15})
    rule(0,{'white.background':.65,'leaf.all':-.45,'warm.all':-.20,'coarse.background':-.15},.05)
    rule(0,{'sky.top':.80,'sky.bottom':-.30,'leaf.bottom':-.20})
    rule(0,{'sky.all':.70,'fine.background':-.50})
    rule(1,{'neutral.all':.30,'dark.bottom':.35,'red.center':.15,'leaf.all':-.30,'water.bottom':-.20},.05)
    rule(1,{'red.center':.60,'dark.bottom':.25,'leaf.all':-.30,'water.bottom':-.20})
    rule(1,{'bar_h.wide':.60,'bar_v.lower':.30,'leaf.all':-.30},-.12)
    rule(1,{'white.background':.35,'dark.bottom':.30,'leaf.all':-.20,'water.bottom':-.20})
    rule(2,{'leaf.background':.55,'sky.background':.20,'dark.all':-.10},.03)
    rule(2,{'sky.background':.55,'warm.center':.10,'water.bottom':-.20})
    rule(2,{'tan.center':.35,'leaf.all':.20,'fine.center':.10})
    rule(2,{'leaf.upper':.45,'leaf.lower':-.10},.10)
    rule(3,{'tan.center':.55,'fine.center':.22,'leaf.background':-.35,'sky.background':-.30},.05)
    rule(3,{'neutral.center':.40,'fine.center':.20,'dark.center':.15,'leaf.background':-.30,'sky.background':-.30},.03)
    rule(3,{'dark.center':.40,'warm.center':.10,'leaf.background':-.30,'sky.background':-.25},.10)
    rule(3,{'white.center':.35,'tan.center':.20,'leaf.background':-.25,'sky.background':-.20},.10)
    rule(4,{'leaf.bottom':.55,'tan.center':.40,'dark.center':-.15,'sky.bottom':-.15})
    rule(4,{'leaf.background':.45,'tan.center':.40,'coarse.center':.10})
    rule(4,{'tan.center':.65,'leaf.all':.10,'sky.all':-.25,'neutral.all':-.20})
    rule(4,{'leaf.bottom':.50,'bar_v.lower':.20,'leaf.top':-.20})
    rule(5,{'tan.center':.45,'white.center':.15,'leaf.background':-.25,'sky.background':-.25},.08)
    rule(5,{'neutral.center':.25,'dark.center':.30,'fine.center':.12,'leaf.background':-.25,'sky.background':-.25},.07)
    rule(5,{'white.center':.50,'dark.center':.15,'leaf.background':-.25,'sky.background':-.20},.06)
    rule(5,{'warm.center':.35,'dark.center':.25,'leaf.background':-.20,'sky.background':-.20},.08)
    rule(6,{'leaf.all':.80,'tan.all':.15,'sky.all':-.30,'white.all':-.15})
    rule(6,{'tan.all':.50,'fine.all':.35,'sky.all':-.30,'white.all':-.15})
    rule(6,{'leaf.center':.80,'leaf.background':-.20,'dark.center':.10})
    rule(6,{'leaf.bottom':.60,'dark.center':.20,'sky.all':-.20})
    rule(7,{'leaf.bottom':.50,'warm.center':.30,'dark.center':.25,'sky.bottom':-.20})
    rule(7,{'leaf.background':.35,'warm.center':.35,'dark.center':.30})
    rule(7,{'warm.center':.40,'dark.center':.30,'sky.all':-.20},.05)
    rule(7,{'leaf.bottom':.45,'bar_v.lower':.30,'leaf.top':-.25})
    rule(8,{'water.bottom':.90,'sky.top':.20,'leaf.all':-.30,'warm.all':-.20})
    rule(8,{'sky.all':.75,'white.center':.25,'leaf.all':-.30,'warm.all':-.20})
    rule(8,{'bar_h.lower':.65,'white.center':.25,'water.bottom':.30,'leaf.all':-.20},-.15)
    rule(8,{'neutral.all':.35,'dark.bottom':.30,'white.center':.20,'leaf.all':-.30,'warm.all':-.25})
    rule(9,{'neutral.center':.40,'white.upper':.30,'dark.bottom':.20,'leaf.all':-.25,'water.bottom':-.20})
    rule(9,{'bar_v.center':.55,'bar_h.center':.35,'white.center':.25,'leaf.all':-.20},-.12)
    rule(9,{'red.center':.40,'dark.bottom':.30,'neutral.all':.10,'leaf.all':-.30})
    rule(9,{'white.background':.40,'dark.bottom':.25,'neutral.center':.15,'leaf.all':-.25,'water.bottom':-.20})
    assert len(rules)==40
    W=np.zeros((40,N))
    b=np.zeros(40)
    for j,(cls,terms,bias) in enumerate(rules):
        assert cls==j//4
        for name,v in terms.items():
            W[j,index[name]]=v
        b[j]=bias
    h=net.dense('scene_rules',features,W,b)
    h=net.relu('scene_support',h)
    h=net.op('scene_rule_groups','reshape',h,shape=[10,4,1])
    h=net.pool('best_scene_explanation',h,(4,1),kind='max')
    return net.op('scene_vector','reshape',h,shape=[10])


def build(destination):
    net=Network()
    gray=net.conv('luminance','image',np.array([.299,.587,.114]).reshape(1,3,1,1))
    gray=net.op('contrast_normalization','norm',gray)
    edges,edge_names=add_edges(net,gray)
    rgb16=net.pool('color_cells','image',(2,2))
    colors,color_names=add_color_maps(net,rgb16)
    scene=add_context(net,colors,color_names,edges,edge_names)
    # Pack HOG and signed luminance into a common five-channel map, without a
    # concatenate primitive: disjoint channel embeddings followed by addition.
    W=np.zeros((5,8,1,1))
    # Geometry bins advance by 45 degrees; derivative outputs list x, y, x+y, y-x.
    for j,k in enumerate((0,2,1,3)):
        W[j,k,0,0]=1
    edge_pack=net.conv('geometry_edge_channels',edges,W)
    gray16=net.pool('luminance_cells',gray,(2,2))
    W=np.zeros((5,1,1,1));W[4,0,0,0]=1
    gray_pack=net.conv('geometry_contrast_channel',gray16,W)
    geometry=net.op('geometry_channels','add',[edge_pack,gray_pack])
    templates=[]
    for cls in range(10):
        degrees=30 if cls==0 else (15 if cls==2 else (5 if cls==8 else 8))
        for view in range(4):
            for scale in (.72,1.0,1.2):
                for angle in (-degrees,0,degrees):
                    W=geometric_filter(cls,view,scale,math.radians(angle))
                    for polarity in (1,-1):
                        V=W.copy()
                        V[4]*=polarity
                        templates.append(V)
    W=np.asarray(templates,dtype=np.float32)
    h=net.conv('analytic_object_models',geometry,W,padding=1)
    h=net.pool('translation_invariance',h,(6,6),kind='max')
    h=net.op('object_view_groups','reshape',h,shape=[10,72,1])
    h=net.pool('view_scale_polarity_invariance',h,(72,1),kind='max')
    shape=net.op('geometry_vector','reshape',h,shape=[10])
    scene=net.dense('context_scores',scene,np.eye(10)*.8)
    shape=net.dense('shape_scores',shape,np.eye(10)*3)
    net.op('class_scores','add',[scene,shape])
    total=sum(a.size for a in net.weights.values())
    assert total<=120_000_000
    assert net.shapes['class_scores']==(10,)
    assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in net.weights.values())
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'network.json').open('w') as f:
        json.dump({'layers':net.layers},f,indent=2)
        f.write('\n')
    np.savez(destination/'weights.npz',**net.weights)
    print('Wrote %d layers and %s fixed float32 weights.'%(len(net.layers),format(total,',')))


if __name__=='__main__':
    build(sys.argv[1])
