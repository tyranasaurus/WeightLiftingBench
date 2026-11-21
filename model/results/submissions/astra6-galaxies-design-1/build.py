"""Data-free, analytic galaxy morphology network.

Only Python's standard library and numpy are used.  No files are read.  All
filters are formulas, all selection operations are inference-time network
operations, and all classification rules are explicitly specified below.
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
        self.shapes = {'image': (3, 128, 128)}

    def layer(self, name, op, inputs, output_shape, **settings):
        self.layers.append(dict(name=name, op=op, inputs=inputs, **settings))
        self.shapes[name] = tuple(output_shape)
        return name

    def dense(self, name, source, w, b=None):
        w = np.asarray(w, dtype=np.float32)
        assert w.ndim == 2 and w.shape[1] == math.prod(self.shapes[source])
        b = np.zeros(w.shape[0], np.float32) if b is None else np.asarray(b, np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = b
        return self.layer(name, 'dense', [source], (len(b),), out=len(b))

    def conv(self, name, source, w, b=None, padding=0):
        w = np.asarray(w, np.float32)
        o, c, k, _ = w.shape
        assert c == self.shapes[source][0]
        b = np.zeros(o, np.float32) if b is None else np.asarray(b, np.float32)
        self.weights[name + '.W'], self.weights[name + '.b'] = w, b
        _, h, width = self.shapes[source]
        return self.layer(name, 'conv', [source], (o, h+2*padding-k+1, width+2*padding-k+1),
                          out=o, kernel=[k, k], stride=1, padding=padding)

    def reshape(self, name, source, shape):
        assert math.prod(shape) == math.prod(self.shapes[source])
        return self.layer(name, 'reshape', [source], shape, shape=list(shape))

    def relu(self, name, source):
        return self.layer(name, 'relu', [source], self.shapes[source])

    def norm(self, name, source):
        return self.layer(name, 'norm', [source], self.shapes[source])

    def pool(self, name, source, size, stride=None, maximum=True):
        stride = size if stride is None else stride
        c, h, w = self.shapes[source]
        shape = (c, (h-size[0])//stride[0]+1, (w-size[1])//stride[1]+1)
        return self.layer(name, 'maxpool' if maximum else 'avgpool', [source], shape,
                          size=list(size), stride=list(stride))

    def join(self, name, sources):
        # Concatenation is implemented using the specified dense and add ops.
        lengths = [math.prod(self.shapes[s]) for s in sources]
        total = sum(lengths)
        pieces = []
        offset = 0
        for i, (s, n) in enumerate(zip(sources, lengths)):
            w = np.zeros((total, n), np.float32)
            w[offset:offset+n] = np.eye(n, dtype=np.float32)
            pieces.append(self.dense(name + '_part' + str(i), s, w))
            offset += n
        return self.layer(name, 'add', pieces, (total,))

    def ratios(self, name, source, groups):
        """Normalize paired numerators using +/-128 times their denominator.

        Pairing makes the normalization mean exactly zero.  The reference
        dominates even the radius numerator, avoiding size/shape coupling.
        This approximates division without learned parameters.
        groups contains (denominator row, list of numerator rows).
        """
        n = math.prod(self.shapes[source])
        width = 2 + 2 * max(len(nums) for den, nums in groups)
        w = np.zeros((len(groups)*width, n), np.float32)
        extract = []
        for g, (den, nums) in enumerate(groups):
            start = g*width
            w[start, den], w[start+1, den] = 128., -128.
            for j, num in enumerate(nums):
                w[start+2+2*j, num], w[start+3+2*j, num] = 1., -1.
                extract.append(start+2+2*j)
        z = self.dense(name+'_pairs', source, w)
        z = self.reshape(name+'_shape', z, (len(groups), 1, width))
        z = self.norm(name+'_norm', z)
        w = np.zeros((len(extract), len(groups)*width), np.float32)
        w[np.arange(len(extract)), extract] = 128.*math.sqrt(2./width)
        return self.dense(name, z, w)


net = Network()
pooled = net.pool('two_pixel_average', 'image', (2, 2), maximum=False)
# Estimate the sky only from the current input, outside the central target.
# This is an inference operation; no training-set moments enter the builder.
y64, x64 = np.meshgrid(np.arange(64)-31.5, np.arange(64)-31.5, indexing='ij')
r64 = x64*x64+y64*y64
sky_mask = ((r64 > 24.**2) & (r64 < 31.**2)).astype(float)
sky_mask /= sky_mask.sum()
w = np.zeros((3, 3, 64, 64), np.float32)
for i in range(3):
    w[i, i] = sky_mask
sky = net.dense('input_sky_estimate', pooled, w.reshape(3, -1))
w = np.zeros((3*64*64, 3), np.float32)
for i in range(3):
    w[i*4096:(i+1)*4096, i] = -1.
sky = net.dense('negative_sky_field', sky, w)
sky = net.reshape('sky_field_image', sky, (3, 64, 64))
pooled = net.layer('sky_subtracted_rgb', 'add', [pooled, sky], (3, 64, 64))
gray = net.conv('luminance', pooled, np.array([.25, .5, .25]).reshape(1, 3, 1, 1))
# The object is specified to be at the frame centre.  A 64-pixel central field
# still contains the galaxy while suppressing most unrelated field sources.
crop = np.zeros((32*32, 64*64), np.float32)
for y in range(32):
    for x in range(32):
        crop[y*32+x, (y+16)*64+x+16] = 1.
gray = net.dense('central_field', gray, crop)
gray = net.reshape('central_image', gray, (1, 32, 32))
signal = net.conv('sky_floor', gray, np.ones((1, 1, 1, 1)), [-.005])
signal = net.relu('positive_light', signal)

# Piecewise-linear x**0.7: modest dynamic-range compression lets the disk
# contribute alongside the nucleus.  Knots are generic radiometric values.
knots = np.array([0., .01, .03, .07, .15, .3, .6, 1.])
values = knots**.7
slopes = np.diff(values)/np.diff(knots)
coeff = np.r_[slopes[0], np.diff(slopes)]
z = net.conv('compression_knots', signal, np.ones((len(coeff), 1, 1, 1)), -knots[:-1])
z = net.relu('compression_hinges', z)
compressed = net.conv('compressed_light', z, coeff.reshape(1, -1, 1, 1))

coords = np.arange(32, dtype=np.float64)-15.5
yy, xx = np.meshgrid(coords, coords, indexing='ij')
r2 = xx*xx+yy*yy
outer_window = np.exp(-.5*(r2/100.)**2)
core_window = np.exp(-r2/(2*3.5**2))


def blur_array(a):
    # Fixed optical/pixel blur, not a measured point-spread function.
    k = np.array([1., 6., 1.])/8.
    p = np.pad(a, 1)
    return sum(k[i]*k[j]*p[i:i+32, j:j+32] for i in range(3) for j in range(3))


def analytic_bank(name, source, window, radii, indices, temperature):
    """Analytic Sersic ellipse projections, maximized over position angle.

    The network, rather than the builder, selects the applicable hypotheses
    for each new input.  Profiles are not taken from, or optimized on, data.
    """
    ratios = [.15, .25, .4, .55, .7, .85, 1.]
    angles = np.arange(12)*np.pi/12.
    descriptors, filters = [], []
    for radius in radii:
        for q in ratios:
            for index in indices:
                descriptors.append((q, radius, index))
                bn = 2.*index-1./3.+.009876/index
                for angle in angles:
                    c, s = np.cos(angle), np.sin(angle)
                    profile = np.zeros((32, 32))
                    # Subpixel area integration is analytic quadrature.
                    for dx in [-.25, .25]:
                        for dy in [-.25, .25]:
                            u = (xx+dx)*c+(yy+dy)*s
                            v = -(xx+dx)*s+(yy+dy)*c
                            radius_map = np.sqrt(u*u+(v/q)**2)
                            profile += np.exp(-bn*(radius_map/radius)**(1./index))*.25
                    profile = blur_array(profile)*window
                    profile -= profile.mean()
                    profile /= max(np.linalg.norm(profile), 1e-12)
                    filters.append(profile.ravel()/32.)
    masked = net.dense(name+'_aperture', source, np.diag(window.ravel()))
    masked = net.reshape(name+'_image', masked, (1, 32, 32))
    normalized = net.norm(name+'_contrast', masked)
    scores = net.dense(name+'_projections', normalized, np.array(filters))
    count = len(descriptors)
    scores = net.reshape(name+'_angle_grid', scores, (count, 1, 12))
    scores = net.pool(name+'_best_angle', scores, (1, 12))
    best = net.reshape(name+'_hypothesis_grid', scores, (1, count, 1))
    best = net.pool(name+'_best_match', best, (count, 1))
    # A narrow triangular near-maximum window avoids brittle quantization.
    both = net.join(name+'_scores_and_best', [scores, best])
    w = np.zeros((count, count+1), np.float32)
    w[np.arange(count), np.arange(count)] = 1./temperature
    w[:, -1] = -1./temperature
    gates = net.dense(name+'_near_maximum', both, w, np.ones(count))
    gates = net.relu(name+'_hypothesis_weights', gates)
    ds = np.array(descriptors)
    w = np.stack([np.ones(count), ds[:, 0], ds[:, 1], ds[:, 2]])
    summary = net.dense(name+'_weighted_parameters', gates, w)
    summary = net.ratios(name+'_parameters', summary, [(0, [1, 2, 3])])
    return net.join(name+'_summary', [summary, best])

outer = analytic_bank('outer_shape', compressed, outer_window,
                      [1., 1.5, 2.25, 3.375, 5.0625, 7.59375, 11.390625],
                      [.65, 1., 2.], .003)
core = analytic_bank('core_shape', signal, core_window,
                     [.7, 1., 1.5, 2.25, 3.375, 5.0625], [.8, 1.5], .003)

# Foreground-resistant colour photometry: identical sky floors in R and B,
# with a central and an annular aperture.  Positive R-B means a red population.
w = np.zeros((2, 3, 1, 1), np.float32)
w[0, 0, 0, 0] = w[1, 2, 0, 0] = 1.
colors = net.conv('red_blue_sky_subtraction', pooled, w, [-.005, -.005])
colors = net.relu('red_blue_light', colors)
y64, x64 = np.meshgrid(np.arange(64)-31.5, np.arange(64)-31.5, indexing='ij')
r64 = x64*x64+y64*y64
color_masks = [np.exp(-r64/(2*1.5**2)),
               np.exp(-r64/(2*6.**2))-np.exp(-r64/(2*2.5**2)),
               np.exp(-r64/(2*6.**2))]
w = np.zeros((6, 2, 64, 64), np.float32)
for i, mask in enumerate(color_masks):
    w[2*i, 0] = w[2*i, 1] = mask
    w[2*i+1, 0], w[2*i+1, 1] = mask, -mask
colors = net.dense('colour_integrals', colors, w.reshape(6, -1))
colors = net.ratios('colour_indices', colors, [(0, [1]), (2, [3]), (4, [5])])

# Rotational asymmetry, allowing small offsets of the centre.  Positive and
# negative residuals are retained, so the aperture sum is truly absolute.
k = np.array([1., 2., 1.])/4.
smoothed = net.conv('symmetry_smoothing', compressed, np.outer(k, k).reshape(1, 1, 3, 3), padding=1)
shifts = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
asym_w = np.zeros((len(shifts)*2*1024, 1024), np.float32)
for i, (dy, dx) in enumerate(shifts):
    for y in range(32):
        for x in range(32):
            row = (2*i)*1024+y*32+x
            ry, rx = 31-y+dy, 31-x+dx
            asym_w[row, y*32+x] = 1.
            if 0 <= ry < 32 and 0 <= rx < 32:
                asym_w[row, ry*32+rx] -= 1.
            asym_w[row+1024] = -asym_w[row]
asym = net.dense('rotated_differences', smoothed, asym_w)
asym = net.relu('absolute_rotation_residuals', asym)
asym_masks = [np.exp(-r2/(2*s*s)) for s in [3., 6., 10.]]
w = np.zeros((3*len(shifts), len(shifts)*2*1024), np.float32)
for s, mask in enumerate(asym_masks):
    for i in range(len(shifts)):
        # Divide by 2 to use the usual rotational-asymmetry convention.
        w[s*len(shifts)+i, 2*i*1024:(2*i+2)*1024] = -np.tile(mask.ravel(), 2)*.5
asym = net.dense('negative_asymmetry_integrals', asym, w)
asym = net.reshape('asymmetry_offset_grid', asym, (3, 1, len(shifts)))
asym = net.pool('minimum_offset_asymmetry', asym, (1, len(shifts)))
flux = net.dense('aperture_fluxes', smoothed, np.array([m.ravel() for m in asym_masks]))
joined = net.join('asymmetry_flux', [flux, asym])
asym = net.ratios('negative_asymmetry', joined, [(0, [3]), (1, [4]), (2, [5])])
asym = net.dense('asymmetry', asym, -np.eye(3))

# Fine-scale positive residuals outside the nucleus indicate knots and arms.
k = np.array([1., 4., 6., 4., 1.])/16.
w = -np.outer(k, k)
w[2, 2] += 1.
texture = net.conv('small_scale_residual', compressed, w.reshape(1, 1, 5, 5), padding=2)
texture = net.relu('positive_small_scale_residual', texture)
texture_masks = [m*(1.-np.exp(-r2/(2*2.**2))) for m in asym_masks]
texture = net.dense('texture_integrals', texture, np.array([m.ravel() for m in texture_masks]))
joined = net.join('texture_flux', [flux, texture])
texture = net.ratios('texture_fraction', joined, [(0, [3]), (1, [4]), (2, [5])])

# Direct quadrupole moments provide an independent flattened-disk and bar
# measurement.  The maximum over analytic angles approximates vector length.
scales = [2., 3.5, 6., 10.]
moment_rows = []
for scale in scales:
    a = np.exp(-r2/(2*scale**2))
    moment_rows.extend([(r2*a).ravel(), ((xx*xx-yy*yy)*a).ravel(), (2*xx*yy*a).ravel()])
moments = net.dense('quadrupole_integrals', signal, np.array(moment_rows))
moments = net.ratios('quadrupole_components', moments,
                     [(3*i, [3*i+1, 3*i+2]) for i in range(len(scales))])
w = np.zeros((len(scales)*16, len(scales)*2), np.float32)
for i in range(len(scales)):
    for j in range(16):
        phi = 2*np.pi*j/16
        w[i*16+j, i*2:i*2+2] = [np.cos(phi), np.sin(phi)]
moments = net.dense('quadrupole_directions', moments, w)
moments = net.reshape('quadrupole_direction_grid', moments, (len(scales), 1, 16))
moments = net.pool('ellipticity_moments', moments, (1, 16))

# Radial disagreement of quadrupole vectors detects twisted/lopsided light.
w = np.zeros((16, 8), np.float32)
for j in range(16):
    phi = 2*np.pi*j/16
    direction = np.array([np.cos(phi), np.sin(phi)])
    w[j, 2:4] = direction
    w[j, 6:8] = -direction
incoherence = net.dense('radial_orientation_difference', 'quadrupole_components', w)
incoherence = net.reshape('radial_orientation_grid', incoherence, (1, 1, 16))
incoherence = net.pool('radial_incoherence', incoherence, (1, 16))

# Bars have three components: a round bulge, an elongated central bar, and a
# broader disk.  The analytic model bank contains no observed galaxy shapes.
bar_filters = []
for radius in [2., 3.5, 6., 9.]:
    for disk_q in [.55, .8, 1.]:
        for bar_fraction in [.15, .35]:
            for relative_angle in [0., np.pi/4, np.pi/2]:
                for angle in np.arange(12)*np.pi/12:
                    u = xx*np.cos(angle)+yy*np.sin(angle)
                    v = -xx*np.sin(angle)+yy*np.cos(angle)
                    disk = np.exp(-np.sqrt(u*u+(v/disk_q)**2+.16)/radius)
                    disk /= disk.sum()
                    a = angle+relative_angle
                    u = xx*np.cos(a)+yy*np.sin(a)
                    v = -xx*np.sin(a)+yy*np.cos(a)
                    rho = (u/(radius*.85))**2+(v/(radius*.18))**2
                    bar = np.exp(-.5*rho)
                    bar /= bar.sum()
                    bulge = np.exp(-np.sqrt(r2+.16)/(.3*radius))
                    bulge /= bulge.sum()
                    light = (.9-bar_fraction)*disk+bar_fraction*bar+.1*bulge
                    p = blur_array(light)**.7*outer_window
                    p -= p.mean()
                    p /= max(np.linalg.norm(p), 1e-12)
                    bar_filters.append(p.ravel()/32.)
bars = net.dense('bar_disk_projections', 'outer_shape_contrast', np.array(bar_filters))
bars = net.reshape('bar_disk_grid', bars, (1, 1, len(bar_filters)))
bars = net.pool('bar_disk_match', bars, (1, len(bar_filters)))

# Two-core hypotheses: pairs of analytic exponential nuclei.  They can win
# only if two separated maxima explain the light better than one ellipse.
pair_filters = []
for separation in [3., 5., 8., 12.]:
    for width in [.8, 1.4, 2.2]:
        for brightness in [.5, 1.]:
            for centering in [0., .5]:
                for angle in np.arange(12)*np.pi/12:
                    u = xx*np.cos(angle)+yy*np.sin(angle)
                    v = -xx*np.sin(angle)+yy*np.cos(angle)
                    a = np.sqrt((u-separation*(.5-centering))**2+v*v+.16)
                    b = np.sqrt((u+separation*(.5+centering))**2+v*v+.16)
                    p = (np.exp(-a/width)+brightness*np.exp(-b/width))**.7
                    p = blur_array(p)*outer_window
                    p -= p.mean()
                    p /= max(np.linalg.norm(p), 1e-12)
                    pair_filters.append(p.ravel()/32.)
pairs = net.dense('double_nucleus_projections', 'outer_shape_contrast', np.array(pair_filters))
pairs = net.reshape('double_nucleus_grid', pairs, (1, 1, len(pair_filters)))
pairs = net.pool('double_nucleus_match', pairs, (1, len(pair_filters)))

# Fourier-Mellin spiral filters.  p = m/tan(pitch) for two-armed logarithmic
# spirals.  Both handednesses and all phases are represented.  Subtracting the
# Gaussian log-radius DC term suppresses a straight bar's response.
spiral_rows = []
spiral_denoms = []
phi = np.arctan2(yy, xx)
rr = np.sqrt(r2)
for pitch in [15., 35.]:
    for radius in [3., 5., 8., 12.]:
        t = np.log(rr/radius)
        width = .4
        mask = np.exp(-t*t/(2*width**2))/np.maximum(r2, 1.)
        mask *= (1.-np.exp(-r2/8.))*outer_window
        mask /= mask.sum()
        spiral_denoms.append(mask.ravel())
        p = 2./np.tan(np.deg2rad(pitch))
        dc = np.exp(-.5*(p*width)**2)
        for sign in [-1., 1.]:
            for phase in np.arange(8)*2*np.pi/8:
                f = (np.cos(2*phi+sign*p*t+phase)-dc*np.cos(2*phi+phase))*mask
                spiral_rows.append(f.ravel())
spiral_num = net.dense('logarithmic_spiral_projections', signal, np.array(spiral_rows))
spiral_num = net.reshape('spiral_phase_grid', spiral_num, (8, 1, 16))
spiral_num = net.pool('spiral_phase_maximum', spiral_num, (1, 16))
spiral_den = net.dense('spiral_annular_flux', signal, np.array(spiral_denoms))
spiral = net.join('spiral_flux_and_response', [spiral_den, spiral_num])
spiral = net.ratios('spiral_contrast', spiral, [(i, [i+8]) for i in range(8)])
spiral = net.reshape('spiral_scale_grid', spiral, (2, 1, 4))
spiral = net.pool('spiral_pitch_responses', spiral, (1, 4))

# Feature order is fixed by construction, not obtained from data.
features = net.join('morphology', [outer, core, colors, asym, texture, moments, pairs, spiral, bars, incoherence])
feature_names = ['q', 'radius', 'sersic', 'regularity',
                 'core_q', 'core_radius', 'core_sersic', 'core_regularity',
                 'core_red', 'outer_red', 'total_red',
                 'asym_small', 'asym_middle', 'asym_large',
                 'texture_small', 'texture_middle', 'texture_large',
                 'e_core', 'e_middle', 'e_outer', 'e_far',
                 'pair_match', 'tight_spiral', 'loose_spiral',
                 'bar_match', 'radial_incoherence']
F = {key: i for i, key in enumerate(feature_names)}

# Explicit continuous decision rules.  Constants express generic morphology:
# nearly circular spheroids, flattened disks, red bulges, blue arms, a central
# bar more elongated than its surrounding disk, and broken point symmetry.
hinges = []
def hinge(name, terms, bias):
    row = np.zeros(len(F), np.float32)
    for key, value in terms.items():
        row[F[key]] = value
    hinges.append((name, row, bias))

# V-shaped penalties about the centres of three smooth-axis-ratio categories.
for label, target in [('round', .97), ('oval', .68), ('cigar', .34)]:
    hinge(label+'_plus', {'q': 1}, -target)
    hinge(label+'_minus', {'q': -1}, target)
hinge('not_edge', {'q': 1}, -.34)
hinge('moment_not_edge', {'e_outer': -1}, .36)
hinge('round_not_edge', {'q': 1}, -.55)
hinge('too_flat_for_spiral', {'q': -1}, .43)
hinge('large_cigar', {'radius': 1}, -3.)
hinge('low_regularity', {'regularity': -1}, .9)
hinge('asym_excess', {'asym_middle': 1}, -.08)
hinge('texture_excess', {'texture_middle': 1}, -.035)
hinge('bar', {'q': 1., 'core_q': -1.}, -.06)
hinge('central_bar', {'e_middle': 1., 'e_outer': -.6}, -.07)
hinge('central_rounding', {'e_outer': 1., 'e_core': -1.}, -.25)
hinge('pair_gain', {'pair_match': 1., 'regularity': -1.}, -.01)
hinge('bar_gain', {'bar_match': 1., 'regularity': -1.}, -.005)
hinge('radial_disagreement', {'radial_incoherence': 1.}, -.15)
hinge('very_blue', {'outer_red': -1.}, -.05)
hinge('compact', {'radius': -1.}, 3.)

h = net.dense('rule_thresholds', features,
              np.array([v[1] for v in hinges]), np.array([v[2] for v in hinges]))
h = net.relu('rule_hinges', h)
all_features = net.join('features_and_rules', [features, h])
H = {name: len(F)+i for i, (name, _, _) in enumerate(hinges)}
I = dict(F, **H)
readout = np.zeros((10, len(F)+len(H)), np.float32)
bias = np.zeros(10, np.float32)
def rule(cls, constant, terms):
    bias[cls] += constant
    for key, value in terms.items():
        readout[cls, I[key]] += value

# Smooth spheroids: old stellar colours and little non-axisymmetric texture.
for cls, shape in [(2, 'round'), (3, 'oval'), (4, 'cigar')]:
    rule(cls, .58, {'outer_red': 1.9, 'core_red': .35,
                    'low_regularity': -1., 'asym_excess': -1.2,
                    'texture_excess': -1.0, 'pair_gain': -2., 'bar_gain': -4.,
                    shape+'_plus': -1.8, shape+'_minus': -1.8})
rule(4, 0., {'large_cigar': -.12})

# Spiral disks share colour/structure evidence; their subtypes differ in bar
# ellipticity, central stellar population, and radial light concentration.
for cls in [5, 6, 7]:
    rule(cls, .52, {'outer_red': -1.6, 'core_red': -.20,
                    'too_flat_for_spiral': -4., 'asym_excess': -.7,
                    'texture_excess': .5, 'pair_gain': -1.})
rule(5, -.08, {'bar': 3., 'central_bar': 2.0, 'bar_gain': 8., 'outer_red': 1.})
# A compact, symmetric, concentrated bulge favours tightly wound arms.
# Broad logarithmic waves also respond to simple ellipticity, so their role
# is deliberately subordinate to concentration and stellar-population cues.
rule(6, -.24, {'core_red': 1.1, 'sersic': .18, 'tight_spiral': .3})
rule(7, .08, {'core_red': -1.1, 'sersic': -.18, 'loose_spiral': .15})

# Thin disks are separated by a central round, old-stellar-population bulge.
for cls in [8, 9]:
    rule(cls, .95, {'not_edge': -5., 'moment_not_edge': -1.,
                    'asym_excess': -.6, 'round_not_edge': -3.})
rule(8, .18, {'core_red': -1.2, 'central_rounding': -.7})
rule(9, -.18, {'core_red': 1.2, 'central_rounding': .7})

# Disturbances and interactions violate point symmetry.  Red multiple nuclei
# favour an interaction; blue patchy emission favours a disturbed galaxy.
rule(0, -.20, {'asym_middle': 3.0, 'outer_red': -.9,
               'core_red': -.4, 'texture_excess': .8, 'low_regularity': 1.,
               'radial_disagreement': 1.2})
rule(1, -.10, {'asym_middle': 2.5, 'outer_red': .9,
               'core_red': .4, 'asym_small': .3, 'low_regularity': .7, 'pair_gain': 6.})
net.dense('class_scores', all_features, readout, bias)

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
with (out/'network.json').open('w') as f:
    json.dump({'layers': net.layers}, f, indent=2)
np.savez(out/'weights.npz', **net.weights)
print('Built', len(net.layers), 'layers and', sum(w.size for w in net.weights.values()), 'analytic weights.')
