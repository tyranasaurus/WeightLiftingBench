"""Data-free analytic galaxy morphology network.

All parameters below are geometric, photometric, or manually specified rules.
There is no input-data access, learned parameter, optimization, or randomness.
"""
import json
import sys
from pathlib import Path
import numpy as np

DT = np.float32
layers = []
weights = {}
shapes = {'image': (3, 128, 128)}


def layer(name, op, inputs, output_shape, **settings):
    layers.append(dict(name=name, op=op, inputs=list(inputs), **settings))
    shapes[name] = tuple(output_shape)
    return name


def dense(name, src, matrix, bias=None):
    matrix = np.asarray(matrix, dtype=DT)
    n = matrix.shape[0]
    assert matrix.shape[1] == int(np.prod(shapes[src]))
    weights[name + '.W'] = matrix
    weights[name + '.b'] = np.zeros(n, DT) if bias is None else np.broadcast_to(np.asarray(bias, DT), (n,)).copy()
    return layer(name, 'dense', [src], (n,), out=n)


def conv(name, src, matrix, bias=None, padding=0):
    matrix = np.asarray(matrix, DT)
    out, inc, k, _ = matrix.shape
    assert inc == shapes[src][0]
    weights[name + '.W'] = matrix
    weights[name + '.b'] = np.zeros(out, DT) if bias is None else np.broadcast_to(np.asarray(bias, DT), (out,)).copy()
    h, w = shapes[src][1:]
    return layer(name, 'conv', [src], (out, h + 2 * padding - k + 1, w + 2 * padding - k + 1), out=out, kernel=[k, k], stride=1, padding=padding)


def point(name, src, coeff, bias=0):
    return conv(name, src, np.asarray(coeff, DT).reshape(-1, shapes[src][0], 1, 1), bias)


def reshape(name, src, shape):
    return layer(name, 'reshape', [src], shape, shape=list(shape))


def relu(name, src):
    return layer(name, 'relu', [src], shapes[src])


def norm(name, src):
    return layer(name, 'norm', [src], shapes[src])


def add(name, *srcs):
    return layer(name, 'add', srcs, shapes[srcs[0]])


def pool(name, src, size, stride=None, maximum=False):
    if stride is None:
        stride = size
    c, h, w = shapes[src]
    out = (c, (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1)
    return layer(name, 'maxpool' if maximum else 'avgpool', [src], out, size=list(size), stride=list(stride))


def gaussian_kernel(sigma, radius):
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-t * t / (2 * sigma * sigma))
    g /= g.sum()
    return np.outer(g, g)


# Photometry: bin two pixels, smooth with a known approximate seeing kernel,
# subtract a distant sky annulus, and retain the central 64 original pixels.
rgb = pool('bin2', 'image', (2, 2))
g = gaussian_kernel(.65, 1)
gray = conv('luminance', rgb, np.array([.30, .45, .25])[None, :, None, None] * g[None, None], padding=1)
y64, x64 = np.mgrid[:64, :64].astype(np.float64)
x64 -= 31.5
y64 -= 31.5
r64 = np.hypot(x64, y64)
sky = ((r64 >= 23) & (r64 < 30)).astype(np.float64)
sky /= sky.sum()
central = np.arange(4096).reshape(64, 64)[16:48, 16:48].ravel()
W = np.broadcast_to(-sky.ravel(), (1024, 4096)).copy()
W[np.arange(1024), central] += 1
raw = dense('central_sky_subtraction', gray, W, -.008)
raw = reshape('central_image', raw, (1, 32, 32))
raw = relu('positive_signal', raw)
clip = point('saturation_hinges', raw, [1, 1], [0, -.70])
clip = relu('saturation_relu', clip)
flux = point('bounded_signal', clip, [[1, -1]])

# A tolerant 180-degree symmetry intersection suppresses unrelated stars.
# The 3x3 tolerance also avoids interpreting subpixel centering as disturbance.
flipW = np.zeros((1024, 1024), DT)
flipW[np.arange(1024), np.arange(1023, -1, -1)] = 1
flipped = dense('rotate_half_turn', flux, flipW)
flipped = reshape('rotated_image', flipped, (1, 32, 32))
flipped = pool('rotation_tolerance', flipped, (3, 3), (1, 1), maximum=True)
identity = np.zeros((1, 1, 3, 3), DT)
identity[0, 0, 1, 1] = 1
original = conv('inner_image', flux, identity)
negative_flip = point('negative_rotated', flipped, [-1])
unmatched = relu('unmatched_light', add('symmetry_difference', original, negative_flip))
symmetric = add('symmetric_light', original, point('negative_unmatched', unmatched, [-1]))

# Smoothness in a nucleus-excluding aperture.
blur = conv('diffuse_light', flux, gaussian_kernel(1.25, 3)[None, None], padding=3)
blur = conv('inner_diffuse', blur, identity)
highpass = relu('positive_fine_structure', add('fine_structure', original, point('negative_diffuse', blur, [-1])))
# Outer 180-degree asymmetry is measured after smoothing, without the dilation
# used above for nuisance-star rejection. Excluding the nucleus limits centroid
# errors; smoothing limits the noise contribution.
reverse_inner = np.zeros((900, 900), DT)
reverse_inner[np.arange(900), np.arange(899, -1, -1)] = -1
reverse_blur = reshape('reverse_diffuse_image', dense('negative_reverse_diffuse', blur, reverse_inner), (1, 30, 30))
outer_asymmetry = relu('outer_asymmetric_light', add('outer_symmetry_difference', blur, reverse_blur))

Y, X = np.mgrid[:30, :30].astype(np.float64)
X -= 14.5
Y -= 14.5
R = np.hypot(X, Y)
taper = np.exp(-R * R / (2 * 12.0 ** 2))
# A fixed square-root stretch equalizes Poisson-dominated shot noise and gives
# faint outer structure influence. Slopes are exact secants of sqrt, not fitted.
def root_stretch(name, src):
    knots = np.array([0, .002, .008, .032, .128, .512, 1.0])
    slopes = np.diff(np.sqrt(knots)) / np.diff(knots)
    h = point(name + '_knots', src, np.ones((len(slopes), 1)), -knots[:-1])
    h = relu(name + '_hinges', h)
    coeff = np.r_[slopes[0], np.diff(slopes)]
    return point(name + '_root', h, coeff[None])

# Increasing by ten makes the mandated norm epsilon negligible for faint light.
taperW = np.diag((10 * taper).ravel()).astype(DT)
symnorm = dense('central_window_symmetric', root_stretch('symmetric_stretch', symmetric), taperW)
symnorm = norm('normalize_symmetric', reshape('window_symmetric_image', symnorm, (1, 30, 30)))
rawnorm = dense('central_window_original', root_stretch('original_stretch', original), taperW)
rawnorm = norm('normalize_original', reshape('window_original_image', rawnorm, (1, 30, 30)))

# Templates are rendered only from the following generic physical equations.
# Coordinates are in the binned image, not measurements of any photograph.
psf = gaussian_kernel(.85, 2)

def blur_template(a):
    padded = np.pad(a, 2)
    return sum(psf[j, i] * padded[j:j + 30, i:i + 30] for j in range(5) for i in range(5))


def unit_flux(a):
    return a / max(float(a.sum()), 1e-12)


def plane(q, angle, x=X, y=Y):
    u = x * np.cos(angle) + y * np.sin(angle)
    v = (-x * np.sin(angle) + y * np.cos(angle)) / q
    return u, v


def sersic(radius, half_radius, n):
    return np.exp(-(2 * n - 1 / 3) * (radius / half_radius) ** (1 / n))


def ellipse(q, angle, scale, n=1, cx=0, cy=0):
    # Subpixel quadrature is prescribed, not estimated from input images.
    result = np.zeros_like(X)
    for dy in (-1 / 3, 0, 1 / 3):
        for dx in (-1 / 3, 0, 1 / 3):
            u, v = plane(q, angle, X - cx + dx, Y - cy + dy)
            result += sersic(np.hypot(u, v), scale, n)
    return unit_flux(blur_template(result / 9))


def disk(q, angle, h):
    return ellipse(q, angle, h * (2 - 1 / 3), 1)


def bulge(h):
    return ellipse(1, 0, max(.35, .35 * h), 2)


def normalized_template(t):
    t = np.sqrt(np.maximum(t, 0)) * taper
    t = t - t.mean()
    return (t / max(float(np.sqrt(np.sum(t * t))), 1e-12) / 30).ravel().astype(DT)


banks = {}

def bank(name, templates, src=symnorm):
    W = np.asarray([normalized_template(t) for t in templates], DT)
    projection = dense(name + '_correlations', src, W)
    a = reshape(name + '_grid', projection, (1, 1, len(W)))
    a = pool(name + '_best', a, (1, len(W)), maximum=True)
    banks[name] = a
    return a


angles = np.arange(12) * np.pi / 12
radii = [1.5, 2.2, 3.3, 5.0, 7.5, 11.0]
for name, qs in [('round', [1.0, .88]), ('oval', [.74, .60, .48]), ('cigar', [.36, .27])]:
    models = []
    for q in qs:
        for a in ([0] if q == 1 else angles):
            for r in radii:
                for n in [1.0, 2.5]:
                    models.append(ellipse(q, a, r, n))
    bank(name, models)

for name, fractions in [('thin', [0, .03]), ('thin_bulge', [.15, .35, .60, .80])]:
    models = []
    for q in [.10, .18, .26, .36]:
        for a in angles:
            for h in [2.0, 3.0, 4.5, 6.5, 9.0]:
                d, b = disk(q, a, h), bulge(h)
                for f in fractions:
                    models.append((1 - f) * d + f * b)
    bank(name, models)

models = []
for q in [.40, .55, .70, .85, 1.0]:
    for a in ([0] if q == 1 else angles):
        for h in [1.2, 1.8, 2.7, 4.0, 6.0, 9.0]:
            d, b = disk(q, a, h), bulge(h)
            for f in [0, .20, .40]:
                models.append((1 - f) * d + f * b)
bank('disk', models)

geometries = [(q, a) for q in [1.0, .75, .50] for a in ([0] if q == 1 else np.arange(6) * np.pi / 6)]
hs = [2.0, 3.0, 4.5, 6.5]
models = []
for q, a in geometries:
    u, v = plane(q, a)
    for h in hs:
        d, b = disk(q, a, h), bulge(h)
        for phase in np.arange(8) * np.pi / 8:
            bu = u * np.cos(phase) + v * np.sin(phase)
            bv = -u * np.sin(phase) + v * np.cos(phase)
            for length in [1.5, 2.5, 3.5]:
                # A finite Ferrers bar has a comparatively flat central profile
                # and visible ends, unlike an exponential ellipsoidal bulge.
                bar_radius2 = (bu / (length * h)) ** 2 + (bv / (.25 * length * h)) ** 2
                bar = np.maximum(1 - bar_radius2, 0) ** 2
                bar = unit_flux(blur_template(bar))
                for f in [.20, .40, .60]:
                    models.append((.90 - f) * d + .10 * b + f * bar)
bank('bar', models)
# A null bank on the identical disk geometry grid prevents scale/inclination
# sampling differences from being mistaken for positive bar evidence.
bar_nulls = []
for q, a in geometries:
    for h in hs:
        d, b = disk(q, a, h), bulge(h)
        for f in [0, .10, .30, .50]:
            bar_nulls.append((1 - f) * d + f * b)
bank('bar_null', bar_nulls)

for name, pitches in [('tight', [12, 20]), ('loose', [32, 45])]:
    models = []
    for q, a in geometries:
        u, v = plane(q, a)
        r = np.hypot(u, v)
        theta = np.arctan2(v, u)
        for h in hs:
            b = bulge(h)
            rho = r / h
            arm_window = (1 - np.exp(-(rho / .60) ** 4)) * np.exp(-(rho / 4) ** 2)
            for pitch in pitches:
                for handedness in [-1, 1]:
                    winding = 2 * handedness / np.tan(np.deg2rad(pitch))
                    for phase in np.arange(6) * 2 * np.pi / 6:
                        modulation = np.cos(2 * theta - winding * np.log(rho + .01) + phase)
                        spiral = np.exp(-rho) * (1 + .65 * arm_window * modulation)
                        spiral = unit_flux(blur_template(spiral))
                        for f in [.12, .35]:
                            models.append((1 - f) * spiral + f * b)
    bank(name, models)

# Two analytic nuclei and envelopes: compare their best explanation with a
# single smooth object, rather than treating every foreground star as a merger.
models = []
for q in [.20, .35, .55, .75, 1.0]:
    for a in ([0] if q == 1 else angles):
        for r in radii:
            for n in [1.0, 2.5]:
                models.append(ellipse(q, a, r, n))
bank('single', models, rawnorm)
models = []
for a in np.arange(16) * 2 * np.pi / 16:
    for separation in [3.0, 5.0, 8.0, 12.0]:
        for size in [.8, 1.3, 2.1, 3.3]:
            for q in [.65, 1.0]:
                for layout in [0.0, .5]:
                    ca = -layout * separation
                    cb = (1 - layout) * separation
                    first = ellipse(q, a, size, 2, ca * np.cos(a), ca * np.sin(a))
                    second = ellipse(q, a, size, 2, cb * np.cos(a), cb * np.sin(a))
                    for ratio in [.4, .7, 1.0]:
                        models.append((first + ratio * second) / (1 + ratio))
bank('pair', models, rawnorm)

# Bars must improve on every non-barred smooth shape, including an edge-on
# disk. Otherwise an unmodelled inclined disk can masquerade as a Ferrers bar.
null_families = ['bar_null', 'disk', 'round', 'oval', 'cigar', 'thin', 'thin_bulge']
null_parts = []
for j, family in enumerate(null_families):
    column = np.zeros((len(null_families), 1), DT)
    column[j, 0] = 1
    null_parts.append(dense('bar_null_alternative_' + str(j), banks[family], column))
null_vector = add('bar_null_alternatives', *null_parts)
null_grid = reshape('bar_null_shapes', null_vector, (1, 1, len(null_families)))
banks['bar_null'] = pool('best_nonbarred_explanation', null_grid, (1, len(null_families)), maximum=True)

# Assemble interpretable ratios.  norm on [D,-D,e*N,-e*N] implements
# N/sqrt(D^2+e^2*N^2+epsilon), with e=.05. No reciprocal approximation is
# estimated from data, and numerator/denominator pairs have fixed definitions.
ratio_names = []
ratio_terms = []
ratio_bias = []


def aperture(sigma):
    a = np.exp(-R * R / (2 * sigma ** 2))
    return a / a.sum()


def ratio(name, numerator, denominator, denominator_floor=.002):
    ratio_names.append(name)
    ratio_terms.append((numerator, denominator))
    ratio_bias.append(denominator_floor)

for sigma in [6.0, 10.0]:
    a = aperture(sigma).ravel()
    ratio('asym' + str(int(sigma)), {unmatched: a}, {original: a})
ring = np.exp(-R * R / (2 * 7 ** 2)) * (1 - np.exp(-(R / 3) ** 4))
ring /= ring.sum()
ratio('texture', {highpass: ring.ravel()}, {original: ring.ravel()})
for sigma in [3.0, 6.0, 10.0]:
    a = aperture(sigma)
    den = (a * R * R / (sigma * sigma)).ravel()
    for j, theta in enumerate(np.arange(12) * np.pi / 12):
        num = a * ((X * X - Y * Y) * np.cos(2 * theta) + 2 * X * Y * np.sin(2 * theta)) / (sigma * sigma)
        ratio('quadrupole%d_%d' % (sigma, j), {symmetric: num.ravel()}, {symmetric: den})

for name, a in [('core_colour', np.exp(-r64 ** 2 / (2 * 3 ** 2))),
                ('outer_colour', np.exp(-r64 ** 2 / (2 * 7 ** 2)) * (1 - np.exp(-(r64 / 3) ** 4)))]:
    a /= a.sum()
    a -= sky
    num = np.zeros((3, 64, 64))
    den = np.zeros_like(num)
    num[0], num[2] = a, -a
    den[0], den[2] = a, a
    ratio(name, {rgb: num.ravel()}, {rgb: den.ravel()}, .015)

outer_weight = np.exp(-R * R / (2 * 9 ** 2)) * (1 - np.exp(-(R / 3.5) ** 4))
outer_weight /= outer_weight.sum()
ratio('outer_asymmetry', {outer_asymmetry: outer_weight.ravel()}, {blur: outer_weight.ravel()})

nr = len(ratio_names)
by_source = {}
bias = np.zeros(nr * 4, DT)
for j, ((nums, dens), floor) in enumerate(zip(ratio_terms, ratio_bias)):
    bias[4 * j] = 8 * floor
    bias[4 * j + 1] = -8 * floor
    for src, coeff in nums.items():
        if src not in by_source:
            by_source[src] = np.zeros((nr * 4, int(np.prod(shapes[src]))), DT)
        by_source[src][4 * j + 2] += .4 * coeff
        by_source[src][4 * j + 3] -= .4 * coeff
    for src, coeff in dens.items():
        if src not in by_source:
            by_source[src] = np.zeros((nr * 4, int(np.prod(shapes[src]))), DT)
        by_source[src][4 * j] += 8 * coeff
        by_source[src][4 * j + 1] -= 8 * coeff
parts = [dense('ratio_inputs_' + str(i), src, W, bias if i == 0 else None) for i, (src, W) in enumerate(by_source.items())]
ratios = add('ratio_inputs', *parts)
ratios = norm('ratio_normalization', reshape('ratio_channels', ratios, (nr, 1, 4)))
W = np.zeros((nr, nr * 4), DT)
W[np.arange(nr), np.arange(nr) * 4 + 2] = np.sqrt(.5) / .05
ratios = dense('dimensionless_measurements', ratios, W)
ri = {n: i for i, n in enumerate(ratio_names)}
W = np.zeros((36, nr), DT)
for j in range(36):
    W[j, 3 + j] = 1
quad = dense('oriented_ellipticity', ratios, W)
quad = reshape('ellipticity_angles', quad, (3, 1, 12))
quad = pool('rotation_invariant_ellipticity', quad, (1, 12), maximum=True)

# Named scalar feature vector, assembled without a concatenation primitive.
feature_names = list(banks) + ['asym6', 'asym10', 'texture', 'core_colour', 'outer_colour', 'ellipticity3', 'ellipticity6', 'ellipticity10', 'outer_asymmetry']
fi = {n: i for i, n in enumerate(feature_names)}
nf = len(fi)
parts = []
for name, src in banks.items():
    W = np.zeros((nf, 1), DT)
    W[fi[name], 0] = 1
    parts.append(dense('feature_' + name, src, W))
W = np.zeros((nf, nr), DT)
for name in ['asym6', 'asym10', 'texture', 'core_colour', 'outer_colour', 'outer_asymmetry']:
    W[fi[name], ri[name]] = 1
parts.append(dense('feature_ratios', ratios, W))
W = np.zeros((nf, 3), DT)
for i, name in enumerate(['ellipticity3', 'ellipticity6', 'ellipticity10']):
    W[fi[name], i] = 1
parts.append(dense('feature_ellipticity', quad, W))
features = add('morphology_features', *parts)

# Hand-written, piecewise-linear morphology rules. Coefficients express rules,
# not empirical frequencies: old red populations favor smooth types; an inner
# quadrupole excess favors a bar; multiple distinct cores favor mergers.
hinge_names = ['colour_positive', 'colour_upper', 'texture_positive', 'texture_upper', 'bar_excess', 'pair_excess', 'bar_gain', 'bulge_gain', 'tight_gain', 'loose_gain', 'not_edge', 'not_thin', 'not_round', 'not_flat', 'too_thin', 'not_smooth_round', 'blue_population', 'bulge_shape', 'bar_shape', 'outer_disturbance', 'bulge_cap', 'asymmetry_cap', 'red_core', 'red_core_cap', 'flat_bar_penalty']
H = np.zeros((len(hinge_names), nf), DT)
hb = np.zeros(len(hinge_names), DT)
for row in [0, 1]:
    H[row, fi['core_colour']] = .4
    H[row, fi['outer_colour']] = .6
hb[1] = -.4
H[2, fi['texture']] = H[3, fi['texture']] = 8
hb[2], hb[3] = -.4, -1.2
H[4, fi['ellipticity3']] = 2
H[4, fi['ellipticity10']] = -2
hb[4] = -.20
H[5, fi['pair']] = 20
H[5, fi['single']] = -20
for j, family, baseline in [(6, 'bar', 'bar_null'), (7, 'thin_bulge', 'thin'), (8, 'tight', 'disk'), (9, 'loose', 'disk')]:
    H[j, fi[family]] = 50
    H[j, fi[baseline]] = -50
# Quadrupole e=(a^2-b^2)/(a^2+b^2) supplies shape constraints, independently
# of how successfully a small round bulge matches a bright nucleus.
for row, offset in [(10, .43), (11, .60)]:
    H[row, fi['ellipticity10']] = -1
    hb[row] = offset
H[12, fi['ellipticity6']] = 1
hb[12] = -.25
H[13, fi['ellipticity6']] = -1
hb[13] = .45
H[14, fi['ellipticity10']] = 1
hb[14] = -.65
H[15, fi['round']] = 1
H[15, fi['disk']] = -1
H[16, fi['core_colour']] = -.4
H[16, fi['outer_colour']] = -.6
hb[16] = .10
H[17, fi['ellipticity10']] = 1
H[17, fi['ellipticity3']] = -1
hb[17] = -.12
H[18, fi['ellipticity3']] = 1
H[18, fi['ellipticity10']] = -1
hb[18] = -.05
H[19, fi['outer_asymmetry']] = 1
hb[19] = -.18
H[20] = H[17]
hb[20] = hb[17] - .15
H[21, fi['outer_asymmetry']] = 1
hb[21] = -.30
H[22, fi['core_colour']] = 1
H[23, fi['core_colour']] = 1
hb[23] = -.30
H[24, fi['ellipticity10']] = 1
hb[24] = -.45
hinges = relu('rule_hinges', dense('rule_conditions', features, H, hb))

W = np.zeros((10, nf), DT)
b = np.zeros(10, DT)
# Subtract a common baseline from ordinary classes by using the disk score;
# disturbance/merger scores receive the same baseline in their formulas.
for cls, family in [(2, 'round'), (3, 'oval'), (4, 'cigar'), (5, 'bar'), (6, 'tight'), (7, 'loose'), (8, 'thin'), (9, 'thin_bulge')]:
    W[cls, fi[family]] += 20
    W[cls, fi['disk']] -= 20
for cls in [2, 3, 4]:
    b[cls] -= 2 * .12
for cls in [5, 6, 7]:
    b[cls] += 2 * .12
b[5] -= .15
# Flat smooth ellipses should not explain a disk plus a separate round bulge.
b[4] -= .25
# Inner elongation favors a bar only when it exceeds the outer disk elongation.
W[0, fi['asym6']] = 8
W[0, fi['asym10']] = 3
W[0, fi['single']] = -5
b[0] = 2.5
W[0, fi['outer_asymmetry']] = 3
# Colour is only weak auxiliary evidence: a red disk can be bulgeless too.
b[9] -= .20
b[8] += .20
W[1, fi['pair']] = 40
W[1, fi['single']] = -40
W[1, fi['asym6']] = 5
W[1, fi['asym10']] = 2
W[1, fi['outer_asymmetry']] = 2
b[1] = -.5
base_scores = dense('morphology_scores', features, W, b)
W = np.zeros((10, len(hinge_names)), DT)
for cls in [2, 3, 4]:
    W[cls, 0], W[cls, 1] = 2, -2
    W[cls, 2], W[cls, 3] = -.5, .5
for cls in [5, 6, 7]:
    W[cls, 0], W[cls, 1] = -2, 2
    W[cls, 2], W[cls, 3] = .5, -.5
W[5, 4] = 1.5
W[5, 6] = 2
# A bulge-template improvement is meaningful only inside the edge-on family;
# the raw preference is already present, so it is not awarded a second time.
W[9, 7] = 0
W[6, 8] = 1
W[7, 9] = 1
W[9, 10] = -12
W[8, 11] = -12
W[2, 12] = -5
W[4, 13] = -5
for cls in [5, 6, 7]:
    W[cls, 14] = -6
    W[cls, 15] = -15
for cls in [2, 3, 4]:
    W[cls, 16] = -5
W[5, 16] = 3
W[6, 16] = 3
W[7, 16] = 3
W[9, 17], W[9, 20] = 2, -2
W[8, 17], W[8, 20] = -1, 1
W[5, 18] = 6
W[5, 24] = -4
W[0, 21] = -3
W[9, 22], W[9, 23] = 1, -1
W[8, 22], W[8, 23] = -1, 1
W[0, 5] = -1
adjustments = dense('physical_rule_scores', hinges, W)
add('class_scores', base_scores, adjustments)

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / 'network.json').write_text(json.dumps({'layers': layers}, indent=2) + '\n')
np.savez(out / 'weights.npz', **weights)
print('Wrote', len(layers), 'layers;', sum(w.size for w in weights.values()), 'float32 parameters')
