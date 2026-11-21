"""Build a data-free, analytical person-presence network.

All coefficients below are colour-space formulae, geometric templates, or
hand-chosen fuzzy-logic margins. This program never opens an input file, never
uses a random generator, and never optimizes any parameter.
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

    def op(self, name, op, source, **settings):
        inputs = [source] if isinstance(source, str) else source
        self.layers.append(dict(name=name, op=op, inputs=inputs, **settings))
        return name

    def conv(self, name, source, w, b=None, stride=1, padding=0):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            w = w[:, :, None, None]
        if b is None:
            b = np.zeros(w.shape[0], dtype=np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.asarray(b, dtype=np.float32)
        return self.op(name, 'conv', source, out=w.shape[0],
                       kernel=list(w.shape[-2:]), stride=stride, padding=padding)

    def dense(self, name, source, w, b):
        w = np.asarray(w, dtype=np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.asarray(b, dtype=np.float32)
        return self.op(name, 'dense', source, out=w.shape[0])

    def relu(self, name, source):
        return self.op(name, 'relu', source)

    def pool(self, name, source, size, stride=None, maximum=False):
        if isinstance(size, int):
            size = [size, size]
        if stride is None:
            stride = size
        if isinstance(stride, int):
            stride = [stride, stride]
        return self.op(name, 'maxpool' if maximum else 'avgpool', source,
                       size=size, stride=stride)

    def maximum(self, name, source, channels, side):
        z = self.pool(name + '_space', source, side, maximum=True)
        z = self.op(name + '_axis', 'reshape', z, shape=[1, 1, channels])
        z = self.pool(name + '_types', z, [1, channels], maximum=True)
        return self.op(name, 'reshape', z, shape=[1])


def gaussian(k, cx, cy, sx, sy):
    """Pixel-integrated Gaussian: coordinates describe generic anatomy only."""
    yy, xx = np.mgrid[:k, :k].astype(np.float64)
    xx -= k // 2
    yy -= k // 2
    a = np.zeros((k, k), dtype=np.float64)
    for dy in (-0.375, -0.125, 0.125, 0.375):
        for dx in (-0.375, -0.125, 0.125, 0.375):
            a += np.exp(-0.5 * (((xx + dx - cx) / sx) ** 2
                                + ((yy + dy - cy) / sy) ** 2))
    return a / a.sum()


def face_branch(net, source, side, width, aspect, name):
    """Soft conjunction of compact skin, hair/eyes, and shoulder contrast."""
    height = width * aspect
    radius = int(math.ceil(max(2.7 * width, 2.05 * height)))
    k = 2 * radius + 1
    step = 1 if width < 2 else 2
    out_side = (side - 1) // step + 1

    def g(x, y, sx, sy):
        return gaussian(k, x * width, y * height, sx * width, sy * height)

    core = g(0, 0.03, .27, .29)
    left = g(-.88, .02, .25, .32)
    right = g(.88, .02, .25, .32)
    halo = .4 * left + .4 * right + .2 * g(0, -.94, .4, .23)
    hair = g(0, -.66, .38, .16)
    shirt = g(0, 1.25, .61, .32)
    shirt_left = g(-1.65, 1.25, .34, .34)
    shirt_right = g(1.65, 1.25, .34, .34)
    eye_pair = .5 * (g(-.20, -.15, .115, .085)
                     + g(.20, -.15, .115, .085))
    cheeks = .5 * (g(-.20, .13, .15, .13) + g(.20, .13, .15, .13))
    forehead = g(0, -.35, .27, .12)
    eye_left = (.45 * g(-.20, -.35, .16, .12)
                + .55 * g(-.20, .13, .15, .13)
                - g(-.20, -.15, .115, .085))
    eye_right = (.45 * g(.20, -.35, .16, .12)
                 + .55 * g(.20, .13, .15, .13)
                 - g(.20, -.15, .115, .085))
    eye_contrast = .5 * (eye_left + eye_right)
    nose_contrast = g(0, -.02, .09, .16) - eye_pair
    balance_x = g(-.20, .04, .17, .27) - g(.20, .04, .17, .27)
    balance_y = g(0, -.20, .25, .14) - g(0, .25, .25, .14)

    kernels, biases, indices = [], [], {}

    def unit(label, channel, mask, bias=0):
        a = np.zeros((3, k, k), dtype=np.float64)
        a[channel] = mask
        indices[label] = len(kernels)
        kernels.append(a)
        biases.append(bias)

    def absolute(label, channel, mask):
        unit(label + '+', channel, mask)
        unit(label + '-', channel, -mask)

    unit('core', 0, core)
    # Preserve signed contrasts through two ReLU channels.
    absolute('compact', 0, core - halo)
    absolute('clothes', 0, core - shirt)
    unit('hair', 1, core - hair)
    unit('hair_cap', 1, core - hair, -1.0)
    unit('eyes', 1, eye_contrast)
    unit('eyes_cap', 1, eye_contrast, -.35)
    unit('nose', 1, nose_contrast)
    unit('nose_cap', 1, nose_contrast, -.35)
    absolute('eyes_lr', 1, eye_left - eye_right)
    unit('support', 2, halo)
    for label, mask in [('face_l', core - left), ('face_r', core - right),
                        ('face_lr', left - right),
                        ('body_l', shirt - shirt_left),
                        ('body_r', shirt - shirt_right),
                        ('body_lr', shirt_left - shirt_right)]:
        absolute(label, 1, mask)
    absolute('balance_x', 0, balance_x)
    absolute('balance_y', 0, balance_y)

    z = net.conv(name + '_measure', source, kernels, biases, step, radius)
    z = net.relu(name + '_rectify', z)
    # Six interpretable channels: base, hair, eyes, nose, bilateral head edge,
    # and bilateral clothing edge. Opposite sides must agree in contrast sign:
    # (|c-l| + |c-r| - |l-r|)/2 = distance(c, interval(l,r)).
    w = np.zeros((6, len(kernels)), dtype=np.float64)
    w[0, indices['core']] = 1.1
    for label, gain in [('compact', .9), ('clothes', .2)]:
        w[0, indices[label + '+']] = gain
        w[0, indices[label + '-']] = -gain
    for label, gain in [('balance_x', -.35), ('balance_y', -.15)]:
        w[0, indices[label + '+']] = gain
        w[0, indices[label + '-']] = gain
    for row, label, cap in [(1, 'hair', 1.), (2, 'eyes', .35), (3, 'nose', .35)]:
        w[row, indices[label]] = 1 / cap
        w[row, indices[label + '_cap']] = -1 / cap
    for row, prefix in [(4, 'face'), (5, 'body')]:
        for suffix, gain in [('_l', .5 / .7), ('_r', .5 / .7), ('_lr', -.5 / .7)]:
            for sign in ['+', '-']:
                w[row, indices[prefix + suffix + sign]] = gain
    # Missing halo pixels are unknown, not evidence of a skin boundary.
    w[0, indices['support']] = .9
    b = np.array([-.9, 0., 0., 0., 0., 0.])
    # Eye response is the weaker of left and right eye contrasts, rather than
    # allowing one unrelated dark mark to stand for both eyes.
    for sign in ['+', '-']:
        w[2, indices['eyes_lr' + sign]] = -.5 / .35
    z = net.conv(name + '_evidence', z, w, b)
    # Clipping stops one very strong edge from replacing the other evidence.
    w = np.zeros((12, 6), dtype=np.float64)
    b = np.zeros(12)
    for i in range(6):
        w[2 * i, i] = 1
        w[2 * i + 1, i] = 1
        b[2 * i + 1] = -1
    # Channel zero is a signed base, not a clipped probability.
    w[1, 0] = -1
    b[1] = 0
    z = net.conv(name + '_limit', z, w, b)
    z = net.relu(name + '_limited', z)
    w = np.zeros((1, 12), dtype=np.float64)
    w[0, :2] = [1, -1]
    for i, gain in enumerate([.22, .12, .06 if width >= 2 else .0, .08, .16], 1):
        w[0, 2 * i:2 * i + 2] = [gain, -gain]
    colour_vote = net.conv(name + '_colour_vote', z, w, [-1.4])

    # A compact warm blob is not sufficient: require a face-like combination.
    # All caps are dimensionless semantic evidence, each in [0,1].
    rules = np.array([[1., .7, 0., .2, 1.],    # hair plus wider clothing
                      [.5, 1., .2, 1., .2],  # face, eyes and lateral edges
                      [.2, 1., 1., .2, .1]]) # two eyes plus a central nose
    w = np.zeros((3, 12))
    for row in range(3):
        for i in range(5):
            w[row, 2 * (i + 1):2 * (i + 1) + 2] = [rules[row, i], -rules[row, i]]
    shape = net.conv(name + '_structure_rules', z, w, [-.75, -.8, -.9])
    # Elementwise maximum over the three alternative anatomical explanations.
    w = np.array([[1., -1., 0.], [0., 1., 0.], [0., -1., 0.],
                  [0., 0., 1.], [0., 0., -1.]])
    shape = net.conv(name + '_rule_compare_1', shape, w)
    shape = net.relu(name + '_rule_relu_1', shape)
    w = np.array([[1., 1., -1., -1., 1.],
                  [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])
    shape = net.conv(name + '_rule_compare_2', shape, w)
    shape = net.relu(name + '_rule_relu_2', shape)
    shape = net.conv(name + '_structure_vote', shape, [[1., 1., -1.]], [.35])
    # min(colour, structure) = colour - relu(colour - structure).
    negative_shape = net.conv(name + '_negative_structure', shape, [[-1.]])
    excess = net.op(name + '_colour_excess', 'add', [colour_vote, negative_shape])
    excess = net.relu(name + '_unconfirmed_colour', excess)
    excess = net.conv(name + '_structure_veto', excess, [[-1.]])
    vote = net.op(name + '_vote', 'add', [colour_vote, excess])
    return net.maximum(name + '_max', vote, 1, out_side)


def achromatic_face(net, source, side, width, name):
    """A conservative frontal-face route independent of skin chromaticity.

    Each anatomical part is mandatory. The masks are generic Gaussian regions,
    not photographs, and are evaluated at enough resolution to separate eyes.
    """
    height = 1.35 * width
    radius = int(math.ceil(max(1.8 * width, 1.6 * height)))
    k = 2 * radius + 1
    step = 2 if width <= 6 else 3
    out_side = (side - 1) // step + 1
    gray = net.conv(name + '_light', source, [[0., 1., 0.], [0., 0., 1.]])

    def g(x, y, sx, sy):
        return gaussian(k, x * width, y * height, sx * width, sy * height)

    eyes = [g(x, -.16, .09, .055) for x in [-.22, .22]]
    cheeks = [g(x, .12, .12, .11) for x in [-.22, .22]]
    forehead = [g(x, -.35, .13, .09) for x in [-.22, .22]]
    core = g(0, 0, .28, .28)
    left, right = g(-.8, 0, .17, .3), g(.8, 0, .17, .3)
    shirt = g(0, 1.0, .5, .20)
    shirt_l, shirt_r = g(-1.25, 1.0, .2, .2), g(1.25, 1.0, .2, .2)
    masks = [.5 * forehead[0] + .5 * cheeks[0] - eyes[0],
             .5 * forehead[1] + .5 * cheeks[1] - eyes[1],
             g(0, -.06, .075, .12) - .5 * (eyes[0] + eyes[1]),
             .25 * (cheeks[0] + cheeks[1]) + .5 * g(0, .41, .18, .06)
             - g(0, .28, .18, .045),
             core - left, core - right, left - right,
             shirt - shirt_l, shirt - shirt_r, shirt_l - shirt_r,
             eyes[0] - eyes[1], cheeks[0] - cheeks[1]]
    w = np.zeros((2 * len(masks) + 1, 2, k, k))
    for i, mask in enumerate(masks):
        w[2 * i, 0] = mask
        w[2 * i + 1, 0] = -mask
    w[-1, 1] = core
    measured = net.conv(name + '_regions', gray, w, stride=step, padding=radius)
    measured = net.relu(name + '_region_rectify', measured)
    # Mandatory margins for two eyes, bridge of nose, mouth, bilateral head
    # boundaries, broader shoulders, and sufficient real-image support.
    w = np.zeros((9, 25))
    for i in range(4):
        w[i, 2 * i:2 * i + 2] = [4., -4.]
    for row, start in [(4, 4), (5, 7)]:
        for offset, sign in [(0, 1.), (1, 1.), (2, -1.)]:
            w[row, 2 * (start + offset):2 * (start + offset) + 2] = 2. * sign
    w[6, 24] = 4.
    w[7, 20:22] = -4.
    w[8, 22:24] = -4.
    margins = net.conv(name + '_requirements', measured, w,
                       -4. * np.array([.06, .06, .06, .035, .10, .08, .95, -.10, -.15]))
    # Signed pairwise minima implemented exactly with linear maps and ReLU.
    count, stage = 9, 0
    while count > 1:
        pairs, remainder = count // 2, count % 2
        w = np.zeros((pairs * 3 + remainder * 2, count))
        for j in range(pairs):
            w[3 * j, 2 * j:2 * j + 2] = [1., -1.]
            w[3 * j + 1, 2 * j] = 1.
            w[3 * j + 2, 2 * j] = -1.
        if remainder:
            w[-2, -1], w[-1, -1] = 1., -1.
        margins = net.conv(name + f'_min_compare_{stage}', margins, w)
        margins = net.relu(name + f'_min_relu_{stage}', margins)
        w = np.zeros((pairs + remainder, pairs * 3 + remainder * 2))
        for j in range(pairs):
            w[j, 3 * j:3 * j + 3] = [-1., 1., -1.]
        if remainder:
            w[-1, -2:] = [1., -1.]
        margins = net.conv(name + f'_min_value_{stage}', margins, w)
        count, stage = pairs + remainder, stage + 1
    vote = net.conv(name + '_vote', margins, [[1.]], [.35])
    best = net.maximum(name + '_max', vote, 1, out_side)
    # Only use this weaker, colour-free route when colour information is absent.
    minus_gate = net.dense(name + '_negative_gate', 'monochrome_gate', [[-1.]], [0.])
    diff = net.op(name + '_gate_difference', 'add', [best, minus_gate])
    diff = net.relu(name + '_gate_relu', diff)
    diff = net.dense(name + '_gate_subtract', diff, [[-1.]], [0.])
    return net.op(name + '_gated_vote', 'add', [best, diff])


def silhouette(k, height, aspect, pose):
    """Continuous generic head, trapezoidal torso, arms and separated legs."""
    yy, xx = np.mgrid[:k, :k].astype(np.float64)
    x = (xx - k // 2) / (height * aspect)
    y = (yy - k // 2) / height

    def capsule(x1, y1, x2, y2, radius):
        t = np.clip(((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1))
                    / ((x2 - x1) ** 2 + (y2 - y1) ** 2), 0, 1)
        d = np.sqrt((x - x1 - t * (x2 - x1)) ** 2
                    + (y - y1 - t * (y2 - y1)) ** 2)
        return 1 / (1 + np.exp(np.clip((d - radius) * height * 3, -60, 60)))

    head_distance = np.sqrt((x / .078) ** 2 + ((y + .36) / .092) ** 2)
    head = 1 / (1 + np.exp(np.clip((head_distance - 1) * height * .25, -60, 60)))
    torso_width = .155 - .035 * np.clip((y + .22) / .34, 0, 1)
    torso_d = np.maximum.reduce([np.abs(x) - torso_width, -.23 - y, y - .13])
    torso = 1 / (1 + np.exp(np.clip(torso_d * height * 3, -60, 60)))
    parts = [head, torso, capsule(0, -.28, 0, -.19, .045)]
    for sign in (-1, 1):
        arm_end = (.21, .12) if pose == 0 else (.28, -.03)
        parts.append(capsule(sign * .15, -.18, sign * arm_end[0], arm_end[1], .04))
        foot_x = .075 if pose == 0 else .18
        parts.append(capsule(sign * .065, .10, sign * foot_x, .45, .047))
    return np.maximum.reduce(parts), y


def body_branch(net, source, side, name):
    # Absolute directional Gaussian/Sobel-like derivatives of normalized light.
    sobel_x = np.array([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    directions = [sobel_x, (sobel_x + sobel_x.T) / math.sqrt(2),
                  sobel_x.T, (-sobel_x + sobel_x.T) / math.sqrt(2)]
    w = np.zeros((8, 3, 3, 3))
    for i, d in enumerate(directions):
        w[2 * i, 1] = d
        w[2 * i + 1, 1] = -d
    z = net.conv(name + '_gradient', source, w, padding=1)
    z = net.relu(name + '_signed_edges', z)
    w = np.zeros((4, 8))
    for i in range(4):
        w[i, 2 * i:2 * i + 2] = 1
        j = (i + 2) % 4
        w[i, 2 * j:2 * j + 2] -= .75
    z = net.conv(name + '_orientation', z, w)
    z = net.relu(name + '_oriented_edges', z)
    z = net.pool(name + '_tolerant_edges', z, 2, 1, maximum=True)

    k, height = 21, 17.
    filters, skin_filters = [], []
    models = [(.85, 0), (1.2, 0), (1.0, 1)]
    for aspect, pose in models:
        s, yy = silhouette(k, height, aspect, pose)
        gy, gx = np.gradient(s)
        strength = np.sqrt(gx * gx + gy * gy)
        theta = np.arctan2(gy, gx)
        gates = [yy < -.24, (yy >= -.24) & (yy < .12), yy >= .12]
        for gate in gates:
            edge = strength * gate
            edge /= edge.sum() + 1e-12
            filt = np.stack([(np.cos(theta - j * math.pi / 4) ** 2 - .58) * edge
                             for j in range(4)])
            filters.append(filt)
        # Generic head skin provides optional, not mandatory, corroboration.
        sw = np.zeros((3, k, k))
        sw[0] = gaussian(k, .5, -.36 * height + .5,
                          .055 * height * aspect, .065 * height)
        skin_filters.append(sw)
    shape = net.conv(name + '_parts', z, filters, stride=2, padding=k // 2)
    skin = net.conv(name + '_head_colour', source, skin_filters, stride=2, padding=k // 2)
    n = len(models)
    w = np.zeros((n * 6, n * 3))
    b = np.zeros(n * 6)
    for m in range(n):
        for p, threshold in enumerate([.06, .05, .03]):
            w[m * 6 + p, m * 3 + p] = 1
            w[m * 6 + p + 3, m * 3 + p] = -1
            b[m * 6 + p + 3] = threshold
    shape = net.conv(name + '_part_margins', shape, w, b)
    shape = net.relu(name + '_part_rectify', shape)
    w = np.zeros((n, n * 6))
    for m in range(n):
        w[m, m * 6:m * 6 + 3] = 2
        w[m, m * 6 + 3:m * 6 + 6] = -3
    shape_vote = net.conv(name + '_shape_vote', shape, w, np.full(n, -.65))
    skin = net.conv(name + '_colour_vote', skin, .25 * np.eye(n))
    z = net.op(name + '_raw_vote', 'add', [shape_vote, skin])
    # Exact conjunction: the weakest of head, torso and legs limits the vote.
    parts = net.conv(name + '_semantic_parts', name + '_parts',
                     5. * np.eye(n * 3), np.full(n * 3, -.25))
    w = np.zeros((n * 3, n * 3))
    for m in range(n):
        w[m * 3, m * 3] = 1
        w[m * 3, m * 3 + 1] = -1
        w[m * 3 + 1, m * 3] = 1
        w[m * 3 + 2, m * 3] = -1
    diff = net.conv(name + '_minimum_compare_1', parts, w)
    diff = net.relu(name + '_minimum_relu_1', diff)
    w = np.zeros((n, n * 3))
    for m in range(n):
        w[m, m * 3:m * 3 + 3] = [-1., 1., -1.]
    first_min = net.conv(name + '_first_minimum', diff, w)
    w = np.zeros((n, n * 3))
    for m in range(n):
        w[m, m * 3 + 2] = -1
    minus_legs = net.conv(name + '_negative_legs', parts, w)
    diff = net.op(name + '_minimum_compare_2', 'add', [first_min, minus_legs])
    diff = net.relu(name + '_minimum_relu_2', diff)
    diff = net.conv(name + '_minimum_subtract_2', diff, -np.eye(n))
    weakest = net.op(name + '_weakest_part', 'add', [first_min, diff])
    negative = net.conv(name + '_negative_weakest', weakest, -np.eye(n))
    diff = net.op(name + '_conjunction_compare', 'add', [z, negative])
    diff = net.relu(name + '_conjunction_relu', diff)
    diff = net.conv(name + '_conjunction_subtract', diff, -np.eye(n))
    z = net.op(name + '_vote', 'add', [z, diff])
    out_side = side // 2
    return net.maximum(name + '_max', z, n, out_side)


def build(destination):
    net = Network()
    rgb = net.pool('antialias', 'image', 2)
    chroma = net.conv('channel_differences', rgb,
                      [[1., -1., 0.], [-1., 1., 0.], [0., 1., -1.], [0., -1., 1.]])
    chroma = net.relu('absolute_channel_differences', chroma)
    chroma = net.pool('average_chromaticity', chroma, 48)
    net.dense('monochrome_gate', chroma, [[-4., -4., -4., -4.]], [.43])
    luminance = net.conv('luminance', rgb, [[.299, .587, .114]])
    luminance = net.op('contrast_normalize', 'norm', luminance)
    # Standard full-range BT.601 chroma axes, with soft, broad skin-colour
    # inequalities. These ranges are general colour knowledge, not estimates.
    cr = np.array([.5, -.418688, -.081312])
    cb = np.array([-.168736, -.331264, .5])
    rg = np.array([1., -1., 0.])
    gb = np.array([0., 1., -1.])
    constraints = [(-cr, .025, 40.), (cr, -.18, 15.),
                   (-cb, -.17, 15.), (cb, .012, 30.),
                   (-rg, .015, 20.), (-gb, -.02, 10.),
                   (-(cr + .6 * cb), -.005, 20.)]
    w = np.stack([a * gain for a, bias, gain in constraints])
    b = np.array([bias * gain for a, bias, gain in constraints])
    skin = net.conv('skin_inequalities', rgb, w, b)
    skin = net.relu('skin_violations', skin)
    skin = net.conv('skin_confidence', skin, -np.ones((1, len(constraints))), [1.])
    skin = net.relu('skin', skin)
    skin_embed = net.conv('skin_axis', skin, [[1.], [0.], [0.]], [0., 0., 1.])
    light_embed = net.conv('light_axis', luminance, [[0.], [1.], [0.]])
    features = net.op('colour_and_light', 'add', [skin_embed, light_embed])

    maxima = []
    head_models = [(1.6, 1.35), (2.35, 1.35), (3.25, 1.0), (3.25, 1.6)]
    side = 48
    for level in range(4):
        if level:
            features = net.pool('pyramid_' + str(level), features, 2)
            side //= 2
        for j, (width, aspect) in enumerate(head_models):
            maxima.append(face_branch(net, features, side, width, aspect,
                                       f'head_{level}_{j}'))
        if side >= 12:
            maxima.append(body_branch(net, features, side, f'body_{level}'))
        if level == 0:
            for j, width in enumerate([4., 6., 9., 13.]):
                maxima.append(achromatic_face(net, features, side, width,
                                              f'achromatic_{j}'))

    # A weak area veto discourages an entire wooden/food-coloured scene, while
    # retaining local maxima for several people or an off-centre person.
    area = net.pool('skin_area', skin, 48)
    area = net.dense('excess_skin_area', area, [[1.]], [-.4])
    area = net.relu('area_penalty', area)

    # Stack scalar branches using disjoint linear embeddings and addition.
    embedded = []
    for i, scalar in enumerate(maxima):
        w = np.zeros((len(maxima), 1))
        w[i, 0] = 1
        embedded.append(net.dense(f'vote_axis_{i}', scalar, w, np.zeros(len(maxima))))
    votes = net.op('all_votes', 'add', embedded)
    votes = net.op('votes_axis', 'reshape', votes, shape=[1, 1, len(maxima)])
    best = net.pool('best_evidence', votes, [1, len(maxima)], maximum=True)
    best = net.dense('presence_margin', best, [[1.]], [-.35])
    penalty = net.dense('area_cost', area, [[-.8]], [0.])
    decision = net.op('decision', 'add', [best, penalty])
    net.dense('scores', decision, [[0.], [1.]], [0., 0.])

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'network.json').open('w') as f:
        json.dump({'layers': net.layers}, f, indent=2)
        f.write('\n')
    np.savez(destination / 'weights.npz', **net.weights)
    print(f'Wrote {len(net.layers)} layers, '
          f'{sum(a.size for a in net.weights.values()):,} float32 weights.')


if __name__ == '__main__':
    build(sys.argv[1])
