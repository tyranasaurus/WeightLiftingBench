"""Data-free construction of a visual-parts and scene-context network.

All distances below are design units on a canonical diagram, not measurements
of photographs. This compiler only writes its two output files.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


layers, weights = [], {}


def op(name, kind, *inputs, **settings):
    layers.append(dict(name=name, op=kind, inputs=list(inputs), **settings))
    return name


def conv(name, src, w, b=None, stride=1, pad=0):
    o, _, k, _ = w.shape
    weights[name + '.W'] = np.asarray(w, dtype=np.float32)
    weights[name + '.b'] = np.zeros(o, 'f') if b is None else np.asarray(b, 'f')
    return op(name, 'conv', src, out=o, kernel=[k, k], stride=stride, padding=pad)


def dense(name, src, w, b=None):
    o = len(w)
    weights[name + '.W'] = np.asarray(w, 'f')
    weights[name + '.b'] = np.zeros(o, 'f') if b is None else np.asarray(b, 'f')
    return op(name, 'dense', src, out=o)


def relu(name, src):
    return op(name, 'relu', src)


def pool(name, kind, src, size, stride):
    if isinstance(size, int):
        size = (size, size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return op(name, kind, src, size=list(size), stride=list(stride))


def linear(rows, keys):
    w = np.zeros((len(rows), len(keys)), 'f')
    b = np.zeros(len(rows), 'f')
    idx = {x: i for i, x in enumerate(keys)}
    for i, (c, d) in enumerate(rows):
        b[i] = c
        for key, v in d.items():
            w[i, idx[key]] += v
    return w, b


def affmap(name, src, keys, rows):
    w, b = linear(list(rows.values()), keys)
    return conv(name, src, w[:, :, None, None], b), list(rows)


def ramps(name, src, keys, rules):
    """Parallel affine ramps with the exact saturation r(z)-r(z-1)."""
    rows = {}
    for key, (base, direction) in rules.items():
        rows[key + '+'] = base, direction
        rows[key + '-'] = base - 1, direction
    layer, channels = affmap(name + '_affine', src, keys, rows)
    return affmap(name, relu(name + '_positive', layer), channels,
                  {k: (0, {k + '+': 1, k + '-': -1}) for k in rules})


def ramp(key, low, high):
    return -low / (high - low), {key: 1 / (high - low)}


def mixed(terms, low, high):
    return -low / (high - low), {k: v / (high - low) for k, v in terms.items()}


# A downsampled RGB image and named intensity/opponent channels.
rgb = pool('half_rgb', 'avgpool', 'image', 2, 2)
L = np.array([.299, .587, .114])
A = np.array([1., -1., 0.])
B = np.array([-.5, -.5, 1.])
base_w = np.stack([*np.eye(3), L, A, B])
color0 = conv('luma_opponent', rgb, base_w[:, :, None, None])
color_keys = ['r', 'g', 'b', 'l', 'a', 'c']

# Slope in the four compass directions, for luma and for two color axes.
# The average across the edge suppresses single-pixel noise.
Gx = np.outer([.25, .50, .25], [-.5, 0., .5])
Gy = Gx.T
angles = np.arange(4) * (np.pi / 4)
filters = []
dkeys = []
for band in ('l', 'a', 'c'):
    for i, t in enumerate(angles):
        f = math.cos(t) * Gx + math.sin(t) * Gy
        for sign in (1, -1):
            row = np.zeros((6, 3, 3))
            row[color_keys.index(band)] = f * sign
            filters.append(row)
            dkeys.append(f'{band}{i}{sign:+d}')
deriv = relu('positive_slopes', conv('signed_slopes', color0, np.array(filters), pad=1))
magrows = {f'{c}{i}': (0, {f'{c}{i}+1': 1, f'{c}{i}-1': 1})
           for c in ('l', 'a', 'c') for i in range(4)}
mag, mk = affmap('slope_magnitudes', deriv, dkeys, magrows)
edge_at, ekeys = ramps('orientation_selectivity', mag, mk, {
    **{f'e{i}': mixed({f'l{i}': 1., f'l{(i+2)%4}': -.52,
                      f'a{i}': .32, f'a{(i+2)%4}': -.16,
                      f'c{i}': .32, f'c{(i+2)%4}': -.16}, .012, .145)
       for i in range(4)},
    'detail': mixed({**{f'l{i}': .30 for i in range(4)},
                     **{f'a{i}': .075 for i in range(4)},
                     **{f'c{i}': .075 for i in range(4)}}, .018, .158),
    'mild': mixed({f'l{i}': -.30 for i in range(4)}, -.115, -.014),
})

# Unaveraged Haar details capture hairs, feathers, foliage and gravel, rather
# than simply the contrast of the main silhouette at the coarser scale.
haar = np.array([[[1., -1.], [1., -1.]], [[1., 1.], [-1., -1.]],
                 [[1., -1.], [-1., 1.]]]) / 2
hf_w = np.stack([s * band[:, None, None] * h for band in (L, .42*A, .42*B)
                 for h in haar for s in (-1, 1)])
hf = relu('fine_absolute_halves', conv('fine_haar', 'image', hf_w, stride=2))
hf_map, hf_keys = ramps('fine_texture', hf, list(range(len(hf_w))), {
    'fine': mixed({i: (.39 if i % 6 < 4 else .30) for i in range(len(hf_w))}, .015, .13),
})

# Per-pixel color evidence; conjunctions make hue predicates robust to
# brightness. Both chromatic and light neutral sky / dark water are possible.
color, ck = ramps('color_memberships', color0, color_keys, {
    'b_over_r': mixed({'b': 1, 'r': -1}, .014, .19),
    'b_near_g': mixed({'b': 1, 'g': -1}, -.09, .020),
    'light': ramp('l', .43, .79),
    'verylight': ramp('l', .68, .95),
    'dark': ramp('l', .53, .12),
    'black': ramp('l', .30, .025),
    'g_over_r': mixed({'g': 1, 'r': -1}, -.002, .13),
    'g_near_r': mixed({'g': 1, 'r': -1}, -.115, .021),
    'g_over_b': mixed({'g': 1, 'b': -1}, .024, .19),
    'yellow': mixed({'r': .5, 'g': .5, 'b': -1}, .043, .23),
    'warm': mixed({'r': 1, 'g': -1}, .027, .20),
    'warm_hi': mixed({'r': 1, 'g': -1}, .16, .44),
    'not_red': mixed({'r': -1, 'g': 1}, -.27, -.07),
    'nonblack': ramp('l', .10, .32),
    'nonwhite': ramp('l', .95, .56),
    'abs_rg_hi': mixed({'r': 1, 'g': -1}, .023, .14),
    'abs_rg_lo': mixed({'g': 1, 'r': -1}, .023, .14),
    'abs_bg_hi': mixed({'b': 1, 'g': -1}, .023, .14),
    'abs_bg_lo': mixed({'g': 1, 'b': -1}, .023, .14),
})
color2, c2k = ramps('semantic_colors', color, ck, {
    'blue': mixed({'b_over_r': .64, 'b_near_g': .36}, .33, .88),
    'green': mixed({'g_over_r': .54, 'g_over_b': .46}, .29, .87),
    'olive': mixed({'g_near_r': .40, 'g_over_b': .31, 'yellow': .29}, .36, .90),
    'earth': mixed({'warm': .39, 'yellow': .34, 'nonblack': .15, 'not_red': .12}, .36, .85),
    'neutral': mixed({k: -1. for k in ('abs_rg_hi', 'abs_rg_lo', 'abs_bg_hi', 'abs_bg_lo')}, -.60, -.04),
    'red': mixed({'warm_hi': .70, 'yellow': .30}, .26, .86),
    'light': ramp('light', 0, 1), 'white': ramp('verylight', 0, 1),
    'dark': ramp('dark', 0, 1), 'black': ramp('black', 0, 1),
    'mid': mixed({'nonblack': .5, 'nonwhite': .5}, .52, .93),
})

# Part detectors. The positive half of the specified contrast is preserved;
# a signed two-slot intensity detector permits ordinary gray or black tires.
def grid(k):
    z = np.arange(k, dtype=float) - (k - 1) / 2
    return np.meshgrid(z, z)


def balanced(p, n):
    return p / np.sum(p) - n / np.sum(n)


def gauss(x, y, cx, cy, sx, sy):
    return np.exp(-.5 * (((x-cx)/sx)**2 + ((y-cy)/sy)**2))


x, y = grid(7)
part_w = []
part_keys = []
for r in (.83, 1.35, 1.93):
    rr = np.hypot(x, y)
    tire = np.exp(-.5*(rr/(r*.66+.23))**4)
    surround = np.exp(-.5*((rr-(r+.8))/.62)**2)
    surround *= (.55 + .45*(y < .5))
    hub = gauss(x, y, 0, 0, r*.38, r*.38)
    tread = np.exp(-.5*((rr-r)/(.32+.17*r))**2)
    part_w.extend((balanced(surround, tire), balanced(.78*hub+.22*surround, tread)))
    part_keys.extend((f'tire{len(part_keys)//2}', f'rim{len(part_keys)//2}'))
for scale in (.8, 1.25, 1.75):
    spot = gauss(x, y, 0, 0, .83*scale, .61*scale)
    ring = gauss(x, y, 0, .1, 1.55*scale, 1.35*scale)
    part_w.append(balanced(ring, spot))
    part_keys.append(f'darkspot{len(part_keys)-6}')
for width in (.9, 1.5):
    core = gauss(x,y,0,0,width,.6)
    outside = gauss(x,y,0,-1.5,width, .55)+gauss(x,y,0,1.5,width,.55)
    part_w.append(balanced(core, outside))
    part_keys.append(f'brightbar{len(part_keys)-9}')
# Lower pad used inside the maps is avoided by the pooling regions.
parts_raw = conv('radial_and_bar_contrasts', rgb, np.stack(part_w)[:,None,:,:]*L[None,:,None,None], pad=3)
part, pk = ramps('primitive_parts', parts_raw, part_keys, {
    k: ramp(k, .055 if k.startswith('rim') else .040,
            .29 if k.startswith(('tire','darkspot')) else .23) for k in part_keys
})

# Sparse spatial relation filters on the primitive maps. One row per proposed
# part layout; ramps above the minimum useful amount implement a soft AND.
# The readout uses lower pairs for wheels and upper pairs for eyes, independently.
pair_w, pair_b, pair_keys = [], [], []
x, y = grid(9)
for j, gap in enumerate((2.0, 3.0, 4.0)):
    s = min(j, 2)
    for slant in (-.72, 0., .72):
        left = gauss(x, y, -gap, -slant, .48, .48)
        right = gauss(x, y, gap, slant, .48, .48)
        left /= left.sum(); right /= right.sum()
        w = np.zeros((len(pk), 9, 9))
        w[pk.index(f'tire{s}')] = .42*(left+right)
        w[pk.index(f'rim{s}')] = .18*(left+right)
        cup = gauss(x, y, 0, -1.8, 1.2, .58)
        w[pk.index('brightbar1')] = .21*cup/cup.sum()
        pair_w.append(w); pair_b.append(-.43); pair_keys.append(('wheels', j, slant))
for j, gap in enumerate((1.35, 2.2, 3.0)):
    blob = gauss(x, y, 0, 1.65+gap*.26, .65, .52)
    blob /= blob.sum()
    eyes = gauss(x, y, -gap, -.5, .48, .48)+gauss(x, y, gap, -.5, .48, .48)
    eyes /= eyes.sum()
    w = np.zeros((len(pk), 9, 9))
    w[pk.index(f'darkspot{min(j+1,2)}')] = .61*blob
    w[pk.index(f'darkspot{max(j-1,0)}')] = .49*eyes
    pair_w.append(w); pair_b.append(-.37); pair_keys.append(('muzzle', j, 0.))
relations = relu('part_agreement', conv('local_parts_geometry', part, np.stack(pair_w), pair_b, pad=4))

# Shape sketches in canonical coordinates. These are symbolic outlines, not
# pixel arrays. Coordinates are in arbitrary units around a centered origin.
def P(*pts):
    return ('p', pts)


def E(cx, cy, rx, ry, tilt=0):
    return ('e', (cx, cy, rx, ry, tilt))


def stroke(x0, y0, x1, y1, width):
    return E((x0+x1)/2, (y0+y1)/2, math.hypot(x1-x0, y1-y0)*.55,
             width, math.atan2(y1-y0, x1-x0))


def equine(stag=False, lowered=False, facing=False):
    if facing:
        return [E(0, .40, 1.75, 2.1), E(0, -2.0, .97, 1.48),
                P((-.75,-2.65),(-1.8,-3.8),(-.55,-3.35)),
                P((.75,-2.65),(1.8,-3.8),(.55,-3.35)),
                stroke(-.9,1.,-1.04,4.45,.28), stroke(.9,1.,1.04,4.45,.28)]
    body = E(-.72, .15, 3.05 if not stag else 2.7, 1.27 if not stag else 1.12)
    if lowered:
        neck = stroke(1.6,-.1,3.28,1.65,.74 if not stag else .50)
        head = E(3.66,2.05,1.10,.49,.9)
    else:
        neck = stroke(1.24,.18,2.54,-2.18,.80 if not stag else .53)
        head = E(3.00,-2.48,1.14 if not stag else .92,.57,.49)
    tail = stroke(-3.35,.07,-4.34,2.1,.45 if not stag else .24)
    legs = [stroke(-2.60,.70,-2.98,3.60,.30 if not stag else .22),
            stroke(-1.94,.72,-1.60,3.12,.27 if not stag else .18),
            stroke(1.32,.57,1.55,3.56,.30 if not stag else .21),
            stroke(.82,.67,.65,3.27,.24 if not stag else .16)]
    ears = [] if lowered else [stroke(2.47,-2.57,2.18,-3.34,.24),
                               stroke(2.84,-2.65,2.82,-3.53,.20)]
    if stag and not lowered:
        ears += [stroke(2.53,-2.76,2.63,-4.3,.14), stroke(2.61,-3.48,3.25,-4.01,.13)]
    return [body, neck, head, tail, *legs, *ears]


shapes = {
    'plane': [
        [P((-5,.38),(-4.9,-.37),(-3.98,-.35),(-4.52,-2.14),(-3.78,-2.15),(-2.46,-.42),
            (3.40,-.47),(4.79,.04),(4.96,.37),(3.31,.73),(.95,.76),(-1.50,1.90),(-2.57,1.93),(-.96,.67),(-4.45,.65))],
        [E(0,-.10,.60,4.40), P((-.4,-1.35),(-4.9,1.64),(-4.89,2.28),(-.25,.87),
            (.25,.87),(4.89,2.28),(4.9,1.64),(.4,-1.35)),
         P((-.55,2.5),(-2.01,3.82),(-1.95,4.23),(0,3.8),(1.95,4.23),(2.01,3.82),(.55,2.5))],
        [E(0,.0,1.06,1.49), P((-5.05,-.42),(-1.05,.12),(1.05,.12),(5.05,-.42),(4.74,.26),(1.09,.8),(-1.09,.8),(-4.74,.26)),
         E(-2.01,.68,.65,.64), E(2.01,.68,.65,.64), E(0,-1.16,.27,1.6)],
        [E(.50,.04,4.66,.59,-.36), P((-.1,-.2),(-3.28,-3.10),(-2.24,-3.30),(1.8,-.62),
            (-.57,3.28),(-1.68,3.38),(-.41,.35)), P((-4.2,1.70),(-4.71,.15),(-3.12,1.18),(-3.08,1.72))],
    ],
    'car': [
        [P((-4.9,1.46),(-4.70,.04),(-2.96,-.43),(-1.85,-1.76),(1.12,-1.96),(2.78,-.67),
            (4.32,-.39),(4.89,.07),(4.9,1.46),(3.70,1.64),(-3.93,1.64)), E(-3.11,1.41,.9,.96),E(3.02,1.41,.9,.96)],
        [P((-4.49,1.50),(-4.59,.15),(-3.65,-.51),(-2.36,-2.20),(.46,-2.24),(2.06,-1.52),
            (4.68,-.34),(4.91,1.18),(2.94,2.28),(-.45,2.17)), E(-3.0,1.36,.75,.98), E(2.92,1.55,.80,1.01)],
        [P((-4.06,2.20),(-4.52,.66),(-3.32,-.43),(-2.44,-2.14),(2.44,-2.14),(3.32,-.43),(4.52,.66),(4.06,2.20)),
         E(-3.23,1.78,.70,.94),E(3.23,1.78,.70,.94)],
        [P((-4.8,1.2),(-4.6,-.04),(-2.81,-.67),(-1.59,-1.65),(1.08,-1.85),(2.62,-.60),
            (4.5,-.24),(4.90,.50),(4.9,1.37),(3.98,1.65),(-3.77,1.68)),E(-3.08,1.49,.82,.86),E(3.34,1.49,.86,.88)],
    ],
    'bird': [
        [E(-.42,1.03,1.65,2.58,.60), E(1.06,-1.63,1.13,1.12), P((1.83,-1.92),(3.4,-1.52),(1.82,-1.30)),
         P((-1.75,.86),(-4.35,3.64),(-3.05,3.64),(-.74,2.1))],
        [E(-.57,1.12,2.75,1.48,.04),E(1.6,-.89,1.02,1.22),P((2.30,-1.35),(3.66,-1.07),(2.35,-.72)),
         P((-2.0,.59),(-4.68,-.12),(-3.70,1.40),(-1.71,1.76))],
        [E(0,-.41,.71,2.05),P((-.3,-.73),(-2.40,-1.30),(-4.86,-3.56),(-4.37,-1.27),(-2.76,.75),(0,1.29),
                         (2.76,.75),(4.37,-1.27),(4.86,-3.56),(2.40,-1.30),(.3,-.73))],
        [E(-.12,1.02,1.56,2.32,.20), E(.36,-1.62,.87,1.30),P((.82,-2.06),(2.44,-2.28),(.94,-1.65)),
         stroke(-.77,1.4,-2.6,4.14,.52)],
    ],
    'cat': [
        [E(0,.40,3.46,2.99),P((-3.20,-.26),(-2.65,-3.93),(-.61,-2.10)),P((3.20,-.26),(2.65,-3.93),(.61,-2.10)), E(0,2.4,2.12,1.75)],
        [E(-.56,1.25,2.89,2.14),E(1.31,-.67,1.70,1.50),P((-.32,-.9),(.12,-3.10),(1.15,-1.98)),
         P((2.27,-.98),(2.69,-2.96),(3.08,-.82)),stroke(-2.62,1.17,-4.13,.01,.37),stroke(-4.13,.01,-4.20,-1.22,.32)],
        [E(0,2.1,2.03,2.21), E(0,-.87,1.77,1.69), P((-1.62,-.87),(-1.50,-3.38),(-.35,-2.2)),
         P((1.62,-.87),(1.50,-3.38),(.35,-2.2)), E(-1.07,3.46,.62,.80),E(1.07,3.46,.62,.80)],
        [E(0,.39,2.85,2.75),P((-2.80,-.5),(-2.36,-3.76),(-.74,-2.06)),
         P((2.80,-.5),(2.36,-3.76),(.74,-2.06)), E(0,2.42,1.74,1.16)],
    ],
    'deer': [equine(stag=True), equine(stag=True,lowered=True), equine(stag=True,facing=True),
             [E(-.5,.19,2.56,1.08), E(1.9,-.77,1.47,.67,-.65), E(3.33,-1.67,.82,.49,.25),
              stroke(2.89,-1.78,2.57,-2.74,.22), stroke(3.20,-1.79,3.42,-2.74,.18),
              stroke(-2.04,.65,-3.,2.94,.23),stroke(-2.40,.68,-2.04,3.15,.19),
              stroke(.89,.58,1.71,3.0,.21), stroke(.63,.60,.28,2.88,.19)]],
    'dog': [
        [E(0,-.48,2.54,2.44),E(-2.33,-.02,1.09,2.48,-.20),E(2.33,-.02,1.09,2.48,.20), E(0,1.15,1.79,1.42), E(0,3.06,2.12,1.89)],
        [E(0,-.42,2.1,2.01),P((-2.40,-.26),(-3.00,-3.56),(-1.34,-2.25),(-.55,-1.62)),
         P((2.40,-.26),(3.00,-3.56),(1.34,-2.25),(.55,-1.62)), E(0,1.32,1.51,1.61), E(0,3.30,1.98,1.64)],
        [E(-.89,.88,2.74,1.29), E(1.89,-.85,1.3,1.37),E(2.85,-.52,1.21,.67), E(1.25,-.46,.67,1.41),
         stroke(-2.61,1.08,-2.92,3.60,.44),stroke(-1.97,1.26,-1.58,3.37,.36),
         stroke(.81,1.15,1.08,3.60,.46), stroke(1.59,1.04,1.66,3.31,.35), stroke(-3.09,.47,-4.51,-.74,.34)],
        [E(-.06,.09,2.92,2.81), E(-2.24,-.13,.91,2.30,-.33),E(2.24,-.13,.91,2.30,.33), E(0,1.45,1.97,1.49)],
    ],
    'frog': [
        [E(0,-.6,2.93,1.63), E(-1.90,-1.62,.82,.71), E(1.90,-1.62,.82,.71), E(0,.85,2.16,1.92),
         E(-2.65,.82,1.15,1.51,.53),E(2.65,.82,1.15,1.51,-.53),stroke(-1.61,.8,-2.42,2.94,.31),stroke(1.61,.8,2.42,2.94,.31)],
        [E(-.50,.23,3.12,1.70,-.21),E(2.04,-.69,1.67,1.20,.25), E(1.66,-1.48,.76,.62),
         E(-2.42,.36,1.37,1.39),stroke(-2.85,.53,-4.13,1.68,.39),stroke(2.12,.01,2.1,2.25,.34),stroke(2.07,2.1,3.47,2.40,.30)],
        [E(0,.66,3.40,2.20),E(.2,-.72,2.78,1.27),E(-1.42,-1.76,.76,.65),E(1.78,-1.76,.83,.63),
         E(-2.44,1.12,1.24,1.15),E(2.44,1.12,1.24,1.15)],
        [E(0,.12,1.61,2.27), E(.16,-1.39,1.78,1.02), E(-1.18,-1.9,.66,.68),E(1.24,-1.9,.66,.68),
         stroke(-.7,0,-3.22,1.13,.41),stroke(-3.22,1.13,-3.74,2.29,.33),
         stroke(.8,1.45,3.22,2.79,.57),stroke(3.22,2.79,1.69,3.84,.36)],
    ],
    'horse': [equine(), equine(lowered=True), equine(facing=True),
              [E(-.46,.23,3.09,1.27,-.10), stroke(1.13,.13,2.76,-2.35,.81), E(3.14,-2.45,1.05,.58,.55),
               stroke(2.56,-2.68,2.49,-3.58,.25), stroke(-2.97,.0,-4.65,1.15,.39),
               stroke(-2.36,.79,-3.81,2.96,.33),stroke(-1.88,.94,-1.23,3.39,.28),
               stroke(1.37,.70,2.58,2.73,.33),stroke(.84,.72,.37,3.15,.26)]],
    'ship': [
        [P((-5.04,.26),(4.95,.26),(4.25,1.93),(-3.90,2.13)), P((-2.92,.36),(-2.92,-.69),(-2.17,-.69),(-2.17,-1.52),
           (.97,-1.52),(.97,-.73),(2.77,-.73),(2.77,.36)),stroke(-.65,-1.4,-.65,-3.47,.18), E(.60,-1.7,.3,.80)],
        [P((-4.98,.08),(4.91,.4),(2.59,1.72),(-3.99,1.72)), P((-2.33,.25),(-1.81,-1.21),(1.96,-1.21),(2.74,.29)),
         P((-1.48,-1.10),(-1.40,-1.88),(.46,-1.88),(.54,-1.10)),stroke(-.84,-1.7,-.84,-3.06,.22)],
        [P((-3.82,-.25),(3.82,-.25),(2.27,2.83),(-2.27,2.83)), P((-2.85,.05),(-2.29,-1.61),(2.29,-1.61),(2.85,.05)),
         E(0,-2.16,.31,1.29)],
        [P((-4.79,1.10),(4.79,1.10),(3.87,2.25),(-3.69,2.25)), P((-.05,-3.90),(-.04,.89),(-3.58,.89)),
         P((.29,-3.47),(3.50,.80),(.28,.80))],
    ],
    'truck': [
        [P((-4.75,-2.90),(1.95,-2.90),(1.95,1.80),(-4.75,1.80)), P((1.77,-1.35),(3.65,-1.35),(4.73,-.35),
           (4.97,1.13),(4.57,1.92),(1.77,1.92)),E(-3.35,1.88,.86,1.0),E(-1.71,1.88,.81,1.),E(3.58,1.88,.87,1.0)],
        [P((-4.72,-2.93),(-.68,-2.56),(.60,-1.83),(3.62,-1.75),(4.60,-.32),(4.89,1.71),(3.16,2.80),
           (.20,2.63),(-4.71,1.32)),E(-3.25,1.40,.65,.87),E(.34,1.99,.78,1.13),E(3.28,2.09,.81,1.04)],
        [P((-2.99,-2.95),(2.99,-2.95),(3.64,-1.94),(3.64,2.45),(-3.64,2.45),(-3.64,-1.94)),
         E(-2.67,2.20,.77,1.05),E(2.67,2.20,.77,1.05)],
        [P((-4.65,-2.65),(4.53,-2.50),(4.66,1.68),(-4.69,1.83)),
         E(-3.38,1.77,.82,.93), E(-1.58,1.77,.82,.93), E(3.44,1.75,.81,.93)],
    ],
}


def inside(x, y, primitives):
    m = np.zeros_like(x, dtype=bool)
    for kind, data in primitives:
        if kind == 'e':
            cx, cy, rx, ry, theta = data
            xx, yy = x-cx, y-cy
            co, si = math.cos(theta), math.sin(theta)
            m |= ((co*xx+si*yy)/rx)**2 + ((co*yy-si*xx)/ry)**2 < 1
        else:
            q = np.zeros_like(m)
            for (ax,ay), (bx,by) in zip(data, data[1:]+data[:1]):
                if by != ay:
                    q ^= ((ay > y) != (by > y)) & (x < (bx-ax)*(y-ay)/(by-ay)+ax)
            m |= q
    return m


def blur(m, sigma):
    radius = math.ceil(3*sigma)
    z = np.arange(-radius, radius+1)
    g = np.exp(-.5*(z/sigma)**2); g /= g.sum()
    def along(a):
        return np.convolve(a,g,'full')[radius:radius+len(a)]
    return np.apply_along_axis(along, 0, np.apply_along_axis(along, 1, m))


def edge_stencil(primitives, scale, mirror, rot=0.):
    # Rasterize above the actual resolution to avoid aliasing thin legs, struts
    # and diagonals. Edges are distributed between nearby orientation bands.
    k, sub = 12, 9
    xx, yy = grid(k*sub)
    xx /= sub; yy /= sub
    co, si = math.cos(rot), math.sin(rot)
    x = (co*xx+si*yy)*mirror/scale
    y = (co*yy-si*xx)/scale
    m = inside(x,y,primitives).astype(float)
    gy, gx = np.gradient(blur(m, 2.1))
    strength = np.hypot(gx,gy)
    theta = np.arctan2(gy,gx)
    # The polarity of the diagonal axes is in image coordinates (y down).
    orient = np.stack([np.abs(np.cos(theta-a))**10 for a in angles])
    orient /= orient.sum(axis=0)
    a = (orient*strength).reshape(4,k,sub,k,sub).sum(axis=(2,4))
    # Subtract nearby parallel-edge support and weak global background support.
    # Uniform clutter in any one orientation then gives exactly zero response.
    near = np.stack([blur(w,1.30) for w in a]); wide = np.stack([blur(w,2.65) for w in a])
    mass = a.sum(axis=(1,2),keepdims=True)
    near *= mass/np.maximum(near.sum(axis=(1,2),keepdims=True),1e-12)
    wide *= mass/np.maximum(wide.sum(axis=(1,2),keepdims=True),1e-12)
    return (a - .62*near - .38*wide)/max(a.sum(), 1e-12)


classes = list(shapes)
orientation = conv('four_edge_orientations', edge_at, np.eye(len(ekeys), dtype='f')[:4,:,None,None])
shape_w = []
for c in classes:
    for model in shapes[c]:
        for scale in (.68,.91,1.12):
            for mirror in (-1,1):
                shape_w.append(edge_stencil(model, scale, mirror))
                if c == 'plane':
                    # Bank also covers the steep oblique alignment of a plane.
                    shape_w.append(edge_stencil(model, scale, mirror, mirror*.45))
                else:
                    # Most objects are upright, but modest roll is common.
                    shape_w.append(edge_stencil(model, scale, mirror, mirror*.15))

geom = conv('outline_agreement', orientation, np.stack(shape_w), stride=2)
n_geom = len(shape_w)//len(classes)
geom_grid = op('outline_hypotheses', 'reshape', geom, shape=[10,n_geom,9])
geom_max = pool('best_outlines', 'maxpool', geom_grid, (n_geom,9), (1,1))

# A second geometry test is polarity-sensitive. The central surface and
# the immediate surroundings are compared on four color axes. Agreement
# across the silhouette is more useful than arbitrary textured edges;
# the outer belt ignores very remote background or framing.
def silhouette_stencil(primitives, scale, mirror):
    k, sub = 12, 9
    x,y = grid(k*sub)
    m = inside(x*mirror/(sub*scale), y/(sub*scale), primitives)
    m = m.reshape(k,sub,k,sub).mean(axis=(1,3))
    inner = m * m
    border = (1-m) * blur(m,1.15)
    return balanced(inner, border)


chroma_axes = [L, .55*A, .66*B, np.array([-.36,.72,-.36])]
sil_w = []
for c in classes:
    for model in shapes[c][:3]:
        for scale in (.79,1.06):
            for mirror in (-1,1):
                stencil = silhouette_stencil(model,scale,mirror)
                sil_w.extend(band[:,None,None]*stencil for band in chroma_axes)
contrast = conv('silhouette_color_separation', rgb, np.stack(sil_w), stride=2)
sc = len(sil_w)
abs_w = np.concatenate([np.eye(sc,dtype='f'),-np.eye(sc,dtype='f')])[:,:,None,None]
abs_map = relu('silhouette_color_polarity', conv('two_contrast_polarities',contrast,abs_w))
nh = sc // (10*len(chroma_axes))
coherent_w = np.zeros((10*nh,sc*2,1,1),'f')
for i in range(10*nh):
    j = i*len(chroma_axes)
    coherent_w[i,j:j+len(chroma_axes),0,0] = [.60,.14,.16,.10]
    coherent_w[i,sc+j:sc+j+len(chroma_axes),0,0] = [.60,.14,.16,.10]
coherent = conv('silhouette_surface_contrast', abs_map, coherent_w)
coherent_grid = op('surface_hypotheses','reshape',coherent,shape=[10,nh,9])
coherent_max = pool('best_surface_shapes','maxpool',coherent_grid,(nh,9),(1,1))

# Oriented rectilinearity within the object: a long isolated straight line
# should get support along its length, not merely one strong edge pixel.
rect_w = []
for orient in (0, 2):
    for k_len in (3, 5, 7):
        w = np.zeros((4,7,7))
        a = slice(3-k_len//2,4+k_len//2)
        if orient == 0:
            w[orient,a,3] = 1/k_len
            w[orient,a,2] = w[orient,a,4] = -.15/k_len
        else:
            w[orient,3,a] = 1/k_len
            w[orient,2,a] = w[orient,4,a] = -.15/k_len
        rect_w.append(w)
line_field = relu('straight_edge_support', conv('line_segments', orientation, np.stack(rect_w), np.array([- .29, -.24, -.19]*2), pad=3))

# Global track: the two halves of the image are useful for the sky/water
# distinction, as are the surroundings versus the center of the picture.
# Fixed window averages are analytic masks, with the outermost cell excluded
# for gradient-derived measurements to avoid zero-padding artifacts.
regs = {}
x,y = grid(16)

def ellipse(cx,cy,sx,sy):
    return gauss(x,y,cx,cy,sx,sy)


def window(a,b,c,d):
    return ((x>=a)&(x<=b)&(y>=c)&(y<=d)).astype(float)


regs['all'] = window(-6.5,6.5,-6.5,6.5)
regs['top'] = window(-6.5,6.5,-6.5,-3.5)
regs['bottom'] = window(-6.5,6.5,3.5,6.5)
regs['center'] = ellipse(0,0,2.60,2.40)*regs['all']
regs['shoulder'] = ellipse(0,-1.9,3.55,2.0)*regs['all']
regs['lower'] = ellipse(0,2.5,3.45,1.62)*regs['all']
regs['rim'] = regs['all']*np.minimum(1, np.maximum(0, (np.hypot(x,y)-3.2)/3.2))
regs['sides'] = (window(-6.5,-4.5,-5.5,5.5)+window(4.5,6.5,-5.5,5.5))
regs['topedge'] = window(-6.5,6.5,-6.5,-4.5)
regs['floor'] = window(-6.5,6.5,4.5,6.5)
regs['middleband'] = window(-6.5,6.5,-2.5,2.5)
regs['lowerband'] = window(-6.5,6.5,.5,4.5)
regs = {k: a/a.sum() for k,a in regs.items()}

# The descriptor vector is packed from independent branches by zero-filling
# their other columns and summing. No descriptor is saved from any photo.
branches = []

def stats(src, keys, request):
    rows = {}
    for key, windows in request.items():
        for win in windows:
            arr = np.zeros((len(keys),16,16),'f'); arr[keys.index(key)] = regs[win]
            rows[f'{key}_{win}'] = arr.ravel()
    branches.append((src, len(keys)*256, rows))


stats(color2,c2k,{
    'blue': ['top','bottom','center','rim','sides','all'],
    'green': ['top','bottom','center','rim','all'],
    'olive': ['top','bottom','center','rim','all'],
    'earth': ['top','bottom','center','rim','all'],
    'neutral': ['top','bottom','center','all'],
    'light': ['top','bottom','center','rim','all'],
    'white': ['top','bottom','center','rim','all'],
    'dark': ['top','bottom','center','rim','all'],
    'black': ['top','bottom','center','rim','all'],
    'mid': ['top','bottom','all'],
    'red': ['center','rim','all'],
})
stats(edge_at, ekeys, {
    'e0':['top','bottom','center','shoulder','lower','rim','all'], 'e2':['top','bottom','center','middleband','lowerband','rim','all'],
    'e1':['center','all'], 'e3':['center','all'],
    'detail':['top','bottom','center','rim','all','sides'],
    'mild':['top','bottom','center','rim','all','sides'],
})
stats(hf_map,hf_keys,{'fine':['top','bottom','center','rim','all']})
stats(line_field,list(range(6)), {i:['center','shoulder','middleband','lowerband','all'] for i in range(6)})
branches.append((geom_max,10,{f'shape_{k}': np.eye(10,dtype='f')[i] for i,k in enumerate(classes)}))
branches.append((coherent_max,10,{f'surface_{k}': np.eye(10,dtype='f')[i] for i,k in enumerate(classes)}))

# Keep only wheel configurations near the lower object, and face features
# near the middle. Zeroed channels make a fixed rectangular max-pool possible.
pair_scaled = conv('relation_junctions', relations, np.eye(len(pair_keys), dtype='f')[:,:,None,None])
# These gather windows use one convolution as a fixed crop so that padding on
# the local part bank cannot by itself be mistaken for a tire or eye.
def crop(name,src,c,y0,x0,h,w):
    # Pull a rectangular region through a thin spatial selector followed by
    # rectangular pooling: for these even grids square crops are sufficient.
    assert h == w and y0 == x0 and y0*2+h == 16
    k = 2*y0+1
    mat = np.zeros((c,c,k,k),'f'); mat[np.arange(c),np.arange(c),y0,x0] = 1
    return conv(name,src,mat)


crop_rel = crop('central_part_layouts', pair_scaled, len(pair_keys), 3,3,10,10)
# Reshape is not a transpose: the slots below appear in ordinary CHW order.
wheel_sel = np.zeros((9,len(pair_keys),1,1),'f'); wheel_sel[np.arange(9),np.arange(9),0,0] = 1
wheel_field = conv('wheel_pair_channels',crop_rel,wheel_sel)
# For wheels, select lower anchors with a kernel that also provides a moderate
# tolerance for vertical position before the global pooling.
wfold = np.zeros((9,9,5,5),'f'); wfold[np.arange(9),np.arange(9),4,2] = 1
lower_wheels = conv('lower_wheel_anchors',wheel_field,wfold)
wheel_cap = pool('best_lower_wheels','maxpool',lower_wheels,6,6)
face_fold = np.zeros((3,len(pair_keys),3,3),'f'); face_fold[np.arange(3),9+np.arange(3),1,1] = 1
face_map = conv('center_face_anchors',crop_rel,face_fold)
face_cap = pool('best_face_parts','maxpool',face_map,8,8)
branches.append((wheel_cap,9,{f'wheel{i}': np.eye(9,dtype='f')[i] for i in range(9)}))
branches.append((face_cap,3,{f'face{i}': np.eye(3,dtype='f')[i] for i in range(3)}))


def pack(branch_list):
    keys = [k for _,_,rows in branch_list for k in rows]
    projections = []
    n = 0
    for src,dim,rows in branch_list:
        w = np.zeros((len(keys),dim),'f')
        w[n:n+len(rows)] = np.stack(list(rows.values()))
        projections.append(dense('describe_'+src,src,w))
        n += len(rows)
    return op('image_descriptors','add',*projections), keys


descriptors, descriptor_keys = pack(branches)


def bank_ramps(name,src,old_keys,rules):
    rows = {k: (0,{k:1}) for k in old_keys}
    for k,(a,b) in rules.items():
        rows[k+'+'] = a,b
        rows[k+'-'] = a-1,b
    w,b = linear(list(rows.values()),old_keys)
    layer = relu(name+'_positive',dense(name+'_conditions',src,w,b))
    use = {**{k:(0,{k:1}) for k in old_keys}, **{k:(0,{k+'+':1,k+'-':-1}) for k in rules}}
    w,b = linear(list(use.values()),list(rows))
    return dense(name,layer,w,b),list(use)


# Evidence calibration ranges are fractions of a unit membership or a filter's
# unit response. No distribution estimates or class counts enter this readout.
evidence, keys = bank_ramps('scene_and_object_evidence',descriptors,descriptor_keys,{
    'sky': mixed({'blue_top':.52,'light_top':.22,'mild_top':.26},.30,.72),
    'plain_top': mixed({'mild_top':.52,'light_top':.31,'neutral_top':.17},.50,.90),
    'blue_ground': mixed({'blue_bottom':.57,'mild_bottom':.28,'light_bottom':.15},.28,.70),
    'open_cool': mixed({'blue_bottom':.38,'blue_top':.24,'mild_rim':.27,'mild_sides':.11},.31,.76),
    'still_ground': mixed({'mild_bottom':.63,'fine_bottom':-.37},.15,.59),
    'water_bottom': mixed({'blue_bottom':.29,'mild_bottom':.35,'fine_bottom':-.23,'dark_bottom':.20,
                           'green_bottom':-.24,'olive_bottom':-.14,'red_all':-.20},.16,.52),
    'waterline': mixed({'light_top':.49,'light_bottom':-.49,'dark_bottom':.34,'mild_bottom':.23,
                        'olive_bottom':-.32,'green_bottom':-.15},.12,.49),
    'sky_diff': mixed({'blue_top':.47,'blue_bottom':-.35,'light_top':.28,'light_bottom':-.30,
                       'green_bottom':-.19,'olive_bottom':-.13,'mild_bottom':.12},.045,.43),
    'grass': mixed({'olive_bottom':.55,'green_bottom':.60,'earth_bottom':.10},.21,.63),
    'leafy': mixed({'green_rim':.77,'olive_rim':.49,'blue_rim':-.23},.13,.58),
    'woodland': mixed({'olive_rim':.45,'earth_rim':.23,'green_rim':.31,'fine_rim':.19,
                       'blue_rim':-.25,'light_rim':-.14},.10,.48),
    'earthy': mixed({'earth_all':.44,'olive_all':.38,'green_all':.13,'blue_all':-.22,'black_all':.09},.096,.51),
    'green_body': mixed({'green_center':.68,'olive_center':.53,'blue_center':-.24},.16,.69),
    'brown_body': mixed({'earth_center':.66,'olive_center':.15,'blue_center':-.24},.12,.51),
    'no_sky': mixed({'blue_top':-.43,'mild_top':-.25,'white_top':-.32,'light_top':-.18,'dark_top':.20},-.49,.04),
    'bright_top': mixed({'light_top':.62,'light_bottom':-.30,'mild_top':.28},.24,.69),
    'sparse_subject': mixed({'detail_center':.62,'detail_rim':-.50,'fine_center':.22,'fine_rim':-.20},.01,.32),
    'rough_scene': mixed({'fine_rim':.48,'detail_rim':.41,'mild_rim':-.16},.12,.62),
    'rough_body': mixed({'fine_center':.50,'detail_center':.40,'mild_center':-.10},.16,.73),
    'camouflage': mixed({'fine_rim':.37,'fine_center':.27,'detail_rim':.23,'mild_rim':-.13},.12,.57),
    'body_separation': mixed({'dark_center':.52,'dark_rim':-.45,'light_rim':.35,'light_center':-.33},.025,.41),
    'bright_body': mixed({'white_center':.57,'white_rim':-.26,'light_center':.33,'light_rim':-.21},.03,.51),
    'warm_indoor': mixed({'earth_rim':.35,'earth_center':.21,'blue_rim':-.23,'green_rim':-.23,
                           'mild_rim':.20,'dark_rim':.12},.07,.45),
    'asphalt': mixed({'neutral_bottom':.45,'mid_bottom':.29,'mild_bottom':.20,'blue_bottom':-.20,
                       'green_bottom':-.29,'olive_bottom':-.26},.25,.80),
    'ground_shadows': mixed({'black_bottom':.43,'black_center':.31,'dark_bottom':.30,
                             'black_rim':-.24},.09,.50),
    'rails': mixed({'3_middleband':.27,'4_middleband':.34,'5_middleband':.43,
                    '3_lowerband':.20,'4_lowerband':.24,'5_lowerband':.32},.035,.36),
    'uprights': mixed({'0_shoulder':.24,'1_shoulder':.29,'2_shoulder':.42,
                       '0_center':.13,'1_center':.16,'2_center':.18},.019,.27),
    'angled': mixed({'e1_center':.46,'e3_center':.46,'e0_center':-.17,'e2_center':-.17},.015,.225),
    'side_vertical': mixed({'e0_lower':.74,'e0_bottom':.26,'e2_lowerband':-.33},.032,.32),
    'flat_hull': mixed({'e2_middleband':.7,'e2_lowerband':.30,'e0_center':-.25,
                       'e2_top':-.16,'e2_bottom':-.20},.035,.30),
    'body_color': mixed({'red_center':.72,'red_rim':-.22,'neutral_bottom':.17},.06,.53),
    'wheel_pair': mixed({f'wheel{i}':1/3 for i in range(9)},.035,.60),
    'face': mixed({'face0':.60,'face1':.66,'face2':.74},.055,.55),
    **{f'outline_{c}': mixed({f'shape_{c}':.76,f'surface_{c}':.24}, .030, .226)
       for c in classes},
})

# Soft combinations are competitive when multiple independent cues agree.
# For example, a horizontal line in a room is not enough evidence for a ship.
composition, kk = bank_ramps('combined_evidence',evidence,keys,{
    'road_form': mixed({'asphalt':.27,'rails':.24,'wheel_pair':.26,'body_color':.11,
                        'uprights':.13,'leafy':-.15,'water_bottom':-.08},.155,.70),
    'car_form': mixed({'outline_car':.35,'wheel_pair':.28,'asphalt':.24,'body_color':.18,
                        'outline_truck':-.11,'uprights':-.07},.14,.72),
    'truck_form': mixed({'outline_truck':.31,'uprights':.31,'rails':.24,'bright_body':.10,
                          'asphalt':.16,'outline_car':-.08},.20,.78),
    'air_form': mixed({'sky':.31,'plain_top':.13,'open_cool':.31,'outline_plane':.22,
                        'waterline':-.14,'asphalt':-.11},.19,.75),
    'ship_form': mixed({'water_bottom':.32,'waterline':.22,'flat_hull':.20,'sky':.13,
                         'outline_ship':.24,'leafy':-.18,'blue_ground':-.09},.21,.82),
    'pasture': mixed({'grass':.58,'sky':.20,'bright_top':.20,'rough_scene':-.07},.16,.77),
    'deer_form': mixed({'woodland':.27,'camouflage':.18,'brown_body':.17,'outline_deer':.25,
                         'side_vertical':.20,'outline_horse':-.11},.19,.79),
    'horse_form': mixed({'grass':.33,'outline_horse':.32,'side_vertical':.16,'body_separation':.13,
                          'sky':.15,'outline_deer':-.09},.19,.77),
    'bird_form': mixed({'outline_bird':.31,'sparse_subject':.26,'leafy':.21,'still_ground':.10,
                         'woodland':.10,'wheel_pair':-.15},.14,.73),
    'frog_form': mixed({'earthy':.28,'camouflage':.23,'green_body':.26,'no_sky':.22,'outline_frog':.20,
                         'body_separation':-.13,'face':-.05},.24,.85),
    'cat_form': mixed({'outline_cat':.35,'warm_indoor':.27,'rough_body':.13,'sparse_subject':.16,
                        'face':.12,'grass':-.19,'outline_dog':-.07},.19,.75),
    'dog_form': mixed({'outline_dog':.28,'face':.30,'body_separation':.19,'rough_body':.15,
                        'warm_indoor':.14,'outline_cat':-.09},.18,.75),
})

# Competing hypotheses. Support from the same physical cue is not repeatedly
# counted at full strength: most weight is on the combined evidence above.
readout = {
    'plane': (-.18, {'air_form':1.10,'outline_plane':.32,'open_cool':.30,'plain_top':.22,
                     'angled':.17,'sparse_subject':.11,'grass':-.29,'road_form':-.18,'ship_form':-.14,
                     'warm_indoor':-.18,'rough_scene':-.11}),
    'car': (-.10, {'car_form':.90,'road_form':.56,'outline_car':.25,'wheel_pair':.22,'body_color':.23,
                   'truck_form':-.18,'grass':-.20,'water_bottom':-.15,'woodland':-.14}),
    'bird': (.00, {'bird_form':.86,'outline_bird':.36,'sparse_subject':.25,'leafy':.21,
                   'still_ground':.08,'grass':-.16,'air_form':-.18,'road_form':-.26,'frog_form':-.19,
                   'face':-.13,'rough_scene':-.08}),
    'cat': (.055, {'cat_form':.99,'outline_cat':.36,'warm_indoor':.19,'rough_body':.16,
                    'sparse_subject':.10,'grass':-.27,'open_cool':-.18,'ship_form':-.16,'dog_form':-.12,
                    'road_form':-.20,'leafy':-.08}),
    'deer': (.00, {'deer_form':.94,'outline_deer':.40,'woodland':.28,'brown_body':.13,
                   'side_vertical':.13,'warm_indoor':-.15,'frog_form':-.23,'horse_form':-.13,
                   'road_form':-.23,'ship_form':-.17}),
    'dog': (.04, {'dog_form':.99,'outline_dog':.30,'face':.24,'body_separation':.20,'warm_indoor':.18,
                   'rough_body':.12,'open_cool':-.21,'frog_form':-.23,'cat_form':-.10,'road_form':-.19}),
    'frog': (-.06, {'frog_form':1.02,'outline_frog':.29,'green_body':.26,'camouflage':.23,
                    'earthy':.19,'no_sky':.14,'pasture':-.19,'open_cool':-.24,'road_form':-.20,
                    'face':-.12,'side_vertical':-.09}),
    'horse': (-.07, {'horse_form':.93,'outline_horse':.38,'pasture':.42,'side_vertical':.16,
                     'body_separation':.14,'woodland':-.09,'warm_indoor':-.20,'ship_form':-.19,
                     'road_form':-.23,'frog_form':-.12}),
    'ship': (-.15, {'ship_form':1.02,'outline_ship':.30,'water_bottom':.33,'waterline':.23,
                    'sky_diff':.16,'flat_hull':.16,'grass':-.30,'road_form':-.23,'frog_form':-.20}),
    'truck': (-.105, {'truck_form':.99,'road_form':.34,'outline_truck':.32,'uprights':.29,
                      'rails':.15,'bright_body':.13,'car_form':-.16,'grass':-.19,'woodland':-.18,
                      'ship_form':-.16}),
}
w,b = linear([readout[c] for c in classes],kk)
dense('class_scores',composition,w,b)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(out/'weights.npz', **weights)


if __name__ == '__main__':
    main()
