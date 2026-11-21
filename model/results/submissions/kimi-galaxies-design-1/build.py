"""Build a hand-designed, rotation-invariant galaxy morphology network.
No data are read: every kernel and every decision threshold is analytic or
chosen from general galaxy morphology.  See method.md for the feature layout.
"""
import json
import sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
OUT.mkdir(parents=True, exist_ok=True)
layers = []
weights = {}

def layer(name, op, inputs, **settings):
    d = {'name': name, 'op': op, 'inputs': inputs}
    d.update(settings)
    layers.append(d)

def put(name, W, b=None):
    weights[name + '.W'] = np.asarray(W, dtype=np.float32)
    if b is None:
        b = np.zeros(W.shape[0], dtype=np.float32)
    weights[name + '.b'] = np.asarray(b, dtype=np.float32)

def gaussian(n, sigma):
    y, x = np.mgrid[:n, :n] - (n - 1) / 2
    g = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    return g / g.sum()

# Normalize color and luminance independently of exposure.  Rectified
# differences of thresholded luminance make three brightness-clipped maps.
layer('p2', 'avgpool', ['image'], size=[2, 2], stride=[2, 2])
layer('zn', 'norm', ['p2'])
layer('lum', 'conv', ['zn'], out=1, kernel=[1, 1], stride=1, padding=0)
put('lum', np.ones((1, 3, 1, 1)) / 3)
layer('ln', 'norm', ['lum'])
layer('thresholds', 'conv', ['ln'], out=4, kernel=[1, 1], stride=1, padding=0)
put('thresholds', np.ones((4, 1, 1, 1)), [0., -.5, -1.5, -3.])
layer('tr', 'relu', ['thresholds'])
W = np.zeros((3, 4, 1, 1))
W[0, 0], W[0, 3] = 1, -1       # clipped at 0 and 3 standard deviations
W[1, 1], W[1, 3] = 1, -1       # a faint-light map
W[2, 2], W[2, 3] = 1, -1       # a core-light map
layer('clips', 'conv', ['tr'], out=3, kernel=[1, 1], stride=1, padding=0)
put('clips', W)

# The first three maps are Gaussian-smoothed light.  The last three are
# positive local contrast (delta minus Gaussian).  ReLU also suppresses sky.
g = gaussian(5, 1.25)
h = gaussian(5, .8) - gaussian(5, 2.2)
W = np.zeros((6, 3, 5, 5))
for c in range(3):
    W[c, c] = g
    W[c+3, c] = h
layer('local', 'conv', ['clips'], out=6, kernel=[5, 5], stride=1, padding=2)
put('local', W)
layer('local_relu', 'relu', ['local'])
layer('fmaps', 'avgpool', ['local_relu'], size=[2, 2], stride=[2, 2])
layer('cmaps', 'avgpool', ['zn'], size=[2, 2], stride=[2, 2])

# Feature indices.  Radial sums are normalized Gaussian apertures.  Angular
# features are maxima over 16 phases of cos(m*theta-phase), hence approximate
# Fourier amplitudes and do not depend on the orientation of the galaxy.
radii = [1.5, 3., 5., 8., 12., 18.]
cradii = [2., 4., 7., 12.]
R = 6 * len(radii)
C = 3 * len(cradii)
start_ang = R + C
ang = []
for c in (1, 4):
    for r in (1, 2, 3, 4):
        for m in (1, 2, 3, 4):
            ang.append((c, r, m))
for r in (1, 2):
    ang.append((2, r, 2))
for r in (2, 3, 4):
    for m in (5, 6):
        ang.append((4, r, m))
F = start_ang + len(ang)
ang_index = {v: start_ang+i for i, v in enumerate(ang)}

def radial(c, r):
    return c*len(radii) + r

def color(c, r):
    return R + c*len(cradii) + r

y, x = np.mgrid[:32, :32] - 15.5
rho = np.sqrt(x*x + y*y)
theta = np.arctan2(y, x)

def aperture(sigma):
    a = np.exp(-(rho/sigma)**2 / 2)
    # A gentle outer cutoff prevents frame-edge stars entering the largest
    # apertures while leaving the central source untouched.
    a *= np.clip((29-rho)/4, 0, 1)
    return a / a.sum()

W = np.zeros((F, 6, 32, 32))
for c in range(6):
    for r, sigma in enumerate(radii):
        W[radial(c, r), c] = aperture(sigma)
layer('radial', 'dense', ['fmaps'], out=F)
put('radial', W.reshape(F, -1))

W = np.zeros((F, 3, 32, 32))
for c in range(3):
    for r, sigma in enumerate(cradii):
        W[color(c, r), c] = aperture(sigma)
layer('color', 'dense', ['cmaps'], out=F)
put('color', W.reshape(F, -1))

W = np.zeros((F*16, 6, 32, 32), dtype=np.float64)
for i, (c, r, m) in enumerate(ang):
    a = aperture(radii[r])
    for p in range(16):
        phase = 2*np.pi*p/16
        W[(start_ang+i)*16+p, c] = a * np.cos(m*theta-phase)
layer('angular', 'dense', ['fmaps'], out=F*16)
put('angular', W.reshape(F*16, -1))
layer('angular_reshape', 'reshape', ['angular'], shape=[F, 16, 1])
layer('angular_max', 'maxpool', ['angular_reshape'], size=[16, 1], stride=[16, 1])
layer('angular_flat', 'reshape', ['angular_max'], shape=[F])
layer('features', 'add', ['radial', 'color', 'angular_flat'])

# A readout made of manually specified morphology prototypes.  Each prototype
# is a small set of desired dimensionless ratios.  Since no division primitive
# exists, |A/M-t| is represented by relu(A-t*M)+relu(t*M-A).  Thus decisions
# remain invariant to overall brightness and, to a useful approximation, size.
rules = []
def term(weights_dict):
    rules.append(dict(weights_dict))
    return F + len(rules) - 1

def abs_terms(weights_dict):
    return term(weights_dict), term({k: -v for k, v in weights_dict.items()})

metric_specs = {}
def metric(name, c, r, m, base_c=None, base_r=None):
    key = (name, r)
    if key in metric_specs:
        return metric_specs[key]
    base_c = c if base_c is None else base_c
    base_r = r if base_r is None else base_r
    M = radial(base_c, base_r)
    if name == 'E':
        d = {ang_index[(c, r, 2)]: 1., M: 0.}
    elif name == 'A':
        d = {ang_index[(c, r, 1)]: 1., M: 0.}
    elif name == 'D':
        d = {ang_index[(c, r, 1)]: 1., ang_index[(c, r, 3)]: .7,
             ang_index[(c, r, 4)]: .5, M: 0.}
    elif name == 'U2':
        d = {ang_index[(c, r, 2)]: 1., radial(c, r): 0.}
    elif name == 'U4':
        d = {ang_index[(c, r, 4)]: 1., radial(c, r): 0.}
    else:
        raise ValueError(name)
    metric_specs[key] = d
    return d

# target ratios for ellipticity E, irregular angular structure D, dipole A,
# two-arm/four-arm contrast U2/U4, central quadrupole B (a bar/disc measure),
# and Q, the fraction of faint-map light remaining in the core-light map.
prototypes = {
    0: {'E': .18, 'D': .20, 'A': .18, 'U2': .22, 'B': .15},
    1: {'E': .24, 'D': .17, 'A': .30, 'U2': .25, 'B': .25},
    2: {'E': .00, 'D': .00, 'A': .00, 'U2': .00, 'B': .03},
    3: {'E': .18, 'D': .00, 'A': .00, 'U2': .00, 'B': .18},
    4: {'E': .43, 'D': .01, 'A': .00, 'U2': .05, 'B': .43, 'Q': .45},
    5: {'E': .13, 'D': .08, 'A': .03, 'U2': .25, 'B': .32},
    6: {'E': .09, 'D': .05, 'A': .02, 'U2': .13, 'B': .08},
    7: {'E': .13, 'D': .17, 'A': .04, 'U2': .34, 'B': .12},
    8: {'E': .50, 'D': .11, 'A': .03, 'U2': .20, 'B': .50, 'Q': .25},
    9: {'E': .43, 'D': .07, 'A': .02, 'U2': .12, 'B': .42, 'Q': .60},
}
penalty = {'E': 5., 'D': 5., 'A': 3., 'U2': 1.2, 'U4': 1.2,
           'B': 2., 'Q': 1.5}
SW = np.zeros((10, F + 600), dtype=np.float64)
# The identity features occupy the first F hidden units.  Extra rule units are
# addressed sparsely; unused trailing columns remain zero.
for cls, proto in prototypes.items():
    SW[cls, radial(1, 3)] = 1.2
    if cls in (0, 1):
        SW[cls, radial(1, 3)] = .8
    elif cls in (5, 6, 7):
        SW[cls, radial(1, 3)] = .65
    for name, target in proto.items():
        rs = (1, 2) if name == 'B' else (2, 3)
        for r in rs:
            if name == 'B':
                d = {ang_index[(2, r, 2)]: 1.}
                base = radial(2, r)
            elif name == 'Q':
                d = {radial(2, r): 1.}
                base = radial(1, r)
            else:
                if name == 'U4':
                    d = metric(name, 4, r, 4)
                elif name in ('D', 'U2'):
                    d = metric(name, 4, r, 2)
                else:
                    d = metric(name, 1, r, 2)
                d = dict(d)
                base = radial(4 if name in ('D', 'U2', 'U4') else 1, r)
            d[base] = d.get(base, 0.) - target
            plus, minus = abs_terms(d)
            SW[cls, plus] -= penalty[name]/2
            SW[cls, minus] -= penalty[name]/2

# Spiral and irregular classes require positive, non-ellipsoidal angular
# structure rather than winning merely by being closest to a noisy smooth map.
for cls, dbonus, ubonus in ((0, 3., 0.), (1, 3., 0.),
                            (5, 5., 1.), (6, 6., 1.), (7, 5., 2.)):
    for r in (2, 3):
        for m, v in ((1, 1.), (3, .7), (4, .5)):
            SW[cls, ang_index[(4, r, m)]] += dbonus*v/2
        SW[cls, ang_index[(4, r, 2)]] += ubonus/2

# A bar changes its quadrupole phase/amplitude relative to the outer disc.
# Penalize the same bar-contrast measurement for barred and unbarred classes.
for cls, target, amount in ((2, 0., 1.5), (3, 0., 1.5),
                            (5, .15, 3.), (6, 0., 2.), (7, 0., 2.)):
    d = {ang_index[(2, 1, 2)]: 1., radial(2, 1): -target,
         ang_index[(1, 3, 2)]: -1.}
    # The outer term must be put on the common faint-map scale.  Its raw
    # coefficient is one; the small core/outer exposure mismatch is absorbed
    # in the deliberately soft target and penalty.
    plus, minus = abs_terms(d)
    SW[cls, plus] -= amount
    SW[cls, minus] -= amount

# A small central colour preference separates old smooth/bulge light from
# blue discs without overwhelming the morphology distances.
red = {color(0, 1): .30, color(2, 1): -.30}
for cls, amount in ((2, .15), (3, .12), (4, .10), (9, .20), (7, -.06), (8, -.12)):
    for j, v in red.items():
        SW[cls, j] += amount*v

# Include the negative parts of signed color projections.
negative_color = {}
for j in range(R, R+C):
    negative_color[j] = term({j: -1.})
for cls, amount in ((2, .6), (3, .5), (4, .4), (9, .8), (7, -.25), (8, -.5)):
    for j, v in red.items():
        SW[cls, negative_color[j]] -= amount*v
H = F + len(rules)
SW = SW[:, :H]
W = np.zeros((H, F), dtype=np.float64)
W[:F, :F] = np.eye(F)
for i, d in enumerate(rules):
    for j, v in d.items():
        W[F+i, j] = v
layer('rules', 'dense', ['features'], out=H)
put('rules', W)
layer('rule_relu', 'relu', ['rules'])
layer('scores', 'dense', ['rule_relu'], out=10)
put('scores', SW)

(OUT/'network.json').write_text(json.dumps({'layers': layers}, indent=2))
np.savez(OUT/'weights.npz', **weights)
print('wrote', len(layers), 'layers,', sum(v.size for v in weights.values()), 'weights,', F, 'features,', H, 'rule units')
