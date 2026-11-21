"""Data-free, hand-designed clothing recognizer.

No input files are opened. Every silhouette is an analytic drawing using
ordinary garment proportions; no examples, descriptors, learned parameters,
or dataset statistics are embedded here. Randomness samples the explicitly
specified geometric nuisance ranges, with a fixed seed.
"""
import json
import sys
from pathlib import Path
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def layer(self, name, op, source, **settings):
        self.layers.append(dict(name=name, op=op,
                                inputs=source if isinstance(source, list) else [source],
                                **settings))
        return name

    def conv(self, name, source, w, b, padding=0):
        w = np.asarray(w, dtype=np.float32)
        self.layer(name, 'conv', source, out=len(w), kernel=list(w.shape[-2:]),
                   stride=1, padding=padding)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.asarray(b, dtype=np.float32)
        return name

    def dense(self, name, source, w, b):
        w = np.asarray(w, dtype=np.float32)
        self.layer(name, 'dense', source, out=len(w))
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.asarray(b, dtype=np.float32)
        return name


def polygon(points):
    return ('polygon', np.asarray(points, dtype=np.float64))


def ellipse(cx, cy, rx, ry, power=2):
    return ('ellipse', (cx, cy, rx, ry, power))


def box(x0, y0, x1, y1):
    return polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def band(a, b, thickness):
    a, b = np.array(a), np.array(b)
    d = b - a
    v = np.array([-d[1], d[0]]) * (thickness / (2 * np.linalg.norm(d)))
    return polygon([a + v, b + v, b - v, a - v])


def bounds(p):
    if p[0] == 'polygon':
        a = p[1]
        return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
    cx, cy, rx, ry, _ = p[1]
    return cx-rx, cy-ry, cx+rx, cy+ry


def inside(p, x, y):
    if p[0] == 'ellipse':
        cx, cy, rx, ry, power = p[1]
        return np.abs((x-cx)/rx)**power + np.abs((y-cy)/ry)**power <= 1
    vertices = p[1]
    result = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=bool)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        if abs(a[1]-b[1]) < 1e-12:
            continue
        cross = ((a[1] > y) != (b[1] > y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
        result ^= cross
    return result


def garment(k, r):
    """A torso and separate sleeve polygons, in arbitrary physical units."""
    u = r.uniform
    positive, negative = [], []
    if k == 0:
        length = u(.85, 1.1)
        shoulder, pit, hem = u(.25, .37), u(.23, .34), u(.24, .36)
        style = 'tank' if r.random() < .23 else 'short'
        cuff_y = u(.25, .43)
        extension, cuff_width = u(.09, .22), u(.10, .17)
        neck_w, neck_h = u(.09, .16), u(.07, .19)
    elif k == 2:
        length = u(.90, 1.12)
        shoulder, pit, hem = u(.31, .40), u(.28, .36), u(.27, .36)
        style = 'long'
        cuff_y = u(.87, 1.04)
        extension, cuff_width = u(.13, .23), u(.10, .17)
        neck_w, neck_h = u(.075, .14), u(.035, .12)
    elif k == 4:
        length = u(.90, 1.43)
        shoulder, pit, hem = u(.30, .41), u(.27, .37), u(.27, .42)
        style = 'long' if r.random() < .93 else 'tank'
        cuff_y = u(.91, 1.08)
        extension, cuff_width = u(.08, .24), u(.10, .16)
        neck_w, neck_h = u(.075, .15), u(.08, .26)
    else:
        length = u(.98, 1.20)
        shoulder, pit, hem = u(.29, .37), u(.25, .33), u(.24, .33)
        style = 'long' if r.random() < .73 else 'short'
        cuff_y = u(.87, 1.05) if style == 'long' else u(.32, .47)
        extension, cuff_width = u(.12, .22), u(.075, .125)
        neck_w, neck_h = u(.07, .14), u(.08, .23)
    if style == 'tank':
        shoulder *= .8
        pit *= .88
        hem *= .9
    if k == 6:
        hem_points = [(-hem, length-.09), (-hem*.65, length-.025),
                      (0, length), (hem*.65, length-.025), (hem, length-.09)]
    else:
        hem_points = [(-hem, length), (hem, length)]
    positive.append(polygon([(-neck_w, 0), (-shoulder, .08), (-pit, .40)] +
                            hem_points + [(pit, .40), (shoulder, .08), (neck_w, 0)]))
    if style != 'tank':
        for sign in [-1, 1]:
            if style == 'long':
                cuff_x = shoulder + extension*r.uniform(.20, 1.10)
                p = [(shoulder*.9, .06), (shoulder+extension, .28),
                     (cuff_x, cuff_y),
                     (cuff_x-cuff_width, cuff_y+.015), (pit*.94, .37)]
            else:
                p = [(shoulder*.9, .06), (shoulder+extension, cuff_y-.13),
                     (shoulder+extension-.065, cuff_y), (pit*.94, .34)]
            positive.append(polygon([(sign*x, y) for x, y in p]))
    if k in (4, 6) and r.random() < .55:
        negative.append(polygon([(-neck_w, -.03), (neck_w, -.03), (0, neck_h)]))
    else:
        negative.append(ellipse(0, -.015, neck_w, neck_h))
    if k == 2 and r.random() < .16:
        positive.append(box(-neck_w*.85, -.08, neck_w*.85, .04))
        negative = [ellipse(0, -.09, neck_w*.55, .04)]
    if k == 4 and r.random() < .32:
        opening = u(.035, .09)
        negative.append(polygon([(0, .22), (-opening, length), (opening, length)]))
    return positive, negative


def trousers(r):
    u = r.uniform
    waist, hip = u(.16, .23), u(.18, .26)
    outer, gap, crotch = u(.14, .25), u(.025, .10), u(.29, .47)
    lean = u(-.035, .035)
    p = [(-waist, 0), (waist, 0), (hip, .16), (outer+lean, 1),
         (gap+lean, 1), (0, crotch), (-gap+lean, 1), (-outer+lean, 1), (-hip, .16)]
    return [polygon(p)], []


def dress(r):
    u = r.uniform
    shoulder, bust = u(.16, .24), u(.16, .23)
    waist, bottom, wy = u(.105, .18), u(.19, .38), u(.32, .48)
    neck = u(.06, .11)
    positive = [polygon([(-neck, 0), (-shoulder, .05), (-bust, .20),
                         (-waist, wy), (-bottom, 1), (bottom, 1), (waist, wy),
                         (bust, .20), (shoulder, .05), (neck, 0)])]
    choice = r.random()
    if choice > .4:
        long = choice > .80
        endy = u(.56, .73) if long else u(.20, .30)
        ex = u(.08, .14)
        for sign in [-1, 1]:
            p = [(shoulder*.9, .04), (shoulder+ex, endy-.07),
                 (shoulder+ex-.065, endy), (bust*.9, .20)]
            positive.append(polygon([(sign*x, y) for x, y in p]))
    negative = [ellipse(0, -.015, neck, u(.06, .14))]
    return positive, negative


def bag(r):
    u = r.uniform
    positive, negative = [], []
    mode = r.random()
    if mode < .18:
        # Backpacks: upright rounded body and small carrying loop.
        width = u(.30, .43)
        positive.append(ellipse(0, .5, width, .5, u(2.6, 4.0)))
        positive.append(ellipse(0, .015, .11, .08))
        negative.append(ellipse(0, -.005, .07, .045))
    else:
        h = u(.47, .84)
        if mode < .35:
            positive.append(ellipse(0, h/2, .5, h/2, u(2.7, 6.0)))
        else:
            top = u(.34, .49)
            bottom = u(.36, .50)
            positive.append(polygon([(-top, 0), (top, 0), (bottom, h*.85),
                                     (bottom*.87, h), (-bottom*.87, h), (-bottom, h*.85)]))
        if mode >= .30:
            rx, ry, thickness = u(.15, .32), u(.14, .40), u(.025, .065)
            # Only the free upper part of the handle is a hole.
            positive.append(ellipse(0, 0, rx, ry))
            negative.append(ellipse(0, -.025, rx-thickness, max(.035, ry-thickness)))
            positive.append(box(-rx, 0, rx, min(h, .12)))
    return positive, negative


def footwear(k, r):
    u = r.uniform
    positive, negative = [], []
    if k == 7:
        toe, rise, heel = u(.12, .22), u(.29, .45), u(.23, .38)
        positive.append(polygon([(0, -.045), (.025, -toe*.70), (.12, -toe),
            (.34, -toe*1.12), (.56, -rise), (.68, -rise*.95), (.78, -heel*.75),
            (.94, -heel), (.99, -heel*.85), (1, -.04), (.88, 0), (.12, 0)]))
    elif k == 9:
        height, shaft_l, shaft_r = u(.48, .94), u(.49, .69), u(.89, 1.0)
        toe = u(.12, .20)
        heel = u(0, .19)
        positive.append(polygon([(0, -.04), (.025, -toe*.75), (.12, -toe),
            (.33, -toe*1.05), (shaft_l-.08, -.29), (shaft_l, -height),
            (shaft_r, -height*u(.91, 1.02)), (shaft_r+.015, -.10-heel),
            (.97, -heel), (.91, 0), (.80, 0), (.76, -heel*.70),
            (.44, -.035), (.12, 0)]))
        if heel > .10:
            negative.append(polygon([(.45, .02), (.76, .02), (.76, -heel*.64), (.54, -.04)]))
    else:
        rise = 0 if r.random() < .50 else u(.15, .37)
        # A narrow footbed, possibly raised at the heel.
        positive.append(polygon([(0, -.025), (.06, -.075), (.40, -.08-rise*.32),
            (.84, -.08-rise), (.99, -.07-rise), (1, -.025-rise),
            (.79, -.02-rise), (.36, -.02-rise*.25), (.05, 0)]))
        if rise:
            positive.append(box(.86, -.06-rise, .94, 0))
        mode = r.random()
        if mode < .4:
            positive.append(band((.13, -.04), (.33, -.23), u(.045, .11)))
            positive.append(band((.33, -.23), (.55, -.07-rise*.5), u(.045, .10)))
            if mode < .18:
                positive.append(band((.58, -.08-rise*.6), (.77, -.30-rise), .055))
                positive.append(band((.77, -.30-rise), (.96, -.05-rise), .05))
        else:
            ankle = u(.33, .62)+rise*.5
            positive.append(band((.22, -.055), (.90, -ankle), u(.04, .075)))
            positive.append(band((.86, -.05-rise), (.92, -ankle), u(.035, .075)))
            positive.append(band((.67, -ankle*.94), (.97, -ankle), .055))
            if mode > .74:
                positive.append(band((.14, -.04), (.45, -.20), .07))
                positive.append(band((.45, -.20), (.77, -.07-rise), .055))
    if r.random() < .5:
        def flip(p):
            if p[0] == 'polygon':
                return polygon([(1-x, y) for x, y in p[1]])
            x, y, rx, ry, power = p[1]
            return ellipse(1-x, y, rx, ry, power)
        positive = [flip(p) for p in positive]
        negative = [flip(p) for p in negative]
    return positive, negative


def render(positive, negative, r):
    # Supersampling is numerical integration of the analytic model, not data.
    b = np.array([bounds(p) for p in positive])
    low, high = b[:, :2].min(0), b[:, 2:].max(0)
    extent = max(high-low)
    scale = r.uniform(.91, 1.015)
    center = (low+high)/2
    shift = r.uniform(-.025, .025, 2)
    t = (np.arange(56)+.5)/56-.5
    x = (t[None, :]-shift[0])*extent/scale+center[0]
    y = (t[:, None]-shift[1])*extent/scale+center[1]
    a = np.zeros((56, 56), dtype=bool)
    for p in positive:
        a |= inside(p, x, y)
    for p in negative:
        a &= ~inside(p, x, y)
    return a.reshape(14, 4, 14, 4).mean((1, 3)).astype(np.float32)


def main(destination):
    net = Network()
    # A soft foreground mask combines absolute and per-image contrast cues.
    net.layer('contrast', 'norm', 'image')
    net.conv('absolute_ramps', 'image', np.array([20, 20]).reshape(2, 1, 1, 1), [-.6, -1.6])
    net.layer('absolute_positive', 'relu', 'absolute_ramps')
    net.conv('absolute_mask', 'absolute_positive', np.array([1, -1]).reshape(1, 2, 1, 1), [0])
    net.conv('relative_ramps', 'contrast', np.array([2, 2]).reshape(2, 1, 1, 1), [0, -1])
    net.layer('relative_positive', 'relu', 'relative_ramps')
    net.conv('relative_mask', 'relative_positive', np.array([1, -1]).reshape(1, 2, 1, 1), [0])
    net.conv('absolute_share', 'absolute_mask', [[[[.65]]]], [0])
    net.conv('relative_share', 'relative_mask', [[[[.35]]]], [0])
    net.layer('foreground', 'add', ['absolute_share', 'relative_share'])
    # Closing joins narrow gaps from fabric patterns, while retaining holes.
    net.conv('foreground_pad', 'foreground', [[[[1]]]], [0], padding=1)
    net.layer('dilated', 'maxpool', 'foreground_pad', size=[3, 3], stride=[1, 1])
    net.conv('complement_pad', 'dilated', [[[[-1]]]], [1], padding=1)
    net.layer('outside_dilated', 'maxpool', 'complement_pad', size=[3, 3], stride=[1, 1])
    net.conv('closed_share', 'outside_dilated', [[[[-.65]]]], [.65])
    net.conv('original_share', 'foreground', [[[[.35]]]], [0])
    net.layer('shape_mask', 'add', ['closed_share', 'original_share'])
    net.layer('coarse_shape', 'avgpool', 'shape_mask', size=[2, 2], stride=[2, 2])
    net.layer('shape_contrast', 'norm', 'coarse_shape')
    net.layer('open_shape', 'avgpool', 'foreground', size=[2, 2], stride=[2, 2])
    net.layer('open_contrast', 'norm', 'open_shape')

    count = 2048
    r = np.random.default_rng(20261119)
    filters = []
    for k in range(10):
        for _ in range(count):
            if k in (0, 2, 4, 6):
                p, n = garment(k, r)
            elif k == 1:
                p, n = trousers(r)
            elif k == 3:
                p, n = dress(r)
            elif k == 8:
                p, n = bag(r)
            else:
                p, n = footwear(k, r)
            a = render(p, n, r)
            a -= a.mean()
            a /= np.sqrt(np.mean(a*a)+1e-5)
            filters.append(a.reshape(-1)/196)
    net.dense('geometric_hypotheses', 'shape_contrast', filters, np.zeros(10*count))
    net.layer('hypotheses_by_class', 'reshape', 'geometric_hypotheses', shape=[10, count, 1])
    net.layer('best_geometric_match', 'maxpool', 'hypotheses_by_class', size=[count, 1], stride=[count, 1])
    net.layer('closed_scores', 'reshape', 'best_geometric_match', shape=[10])
    net.dense('open_hypotheses', 'open_contrast', filters, np.zeros(10*count))
    net.layer('open_by_class', 'reshape', 'open_hypotheses', shape=[10, count, 1])
    net.layer('best_open_match', 'maxpool', 'open_by_class', size=[count, 1], stride=[count, 1])
    net.layer('open_scores', 'reshape', 'best_open_match', shape=[10])
    closed_fraction = np.array([.8, .8, .8, .8, .8, .15, .8, .4, .8, .4])
    net.dense('closed_vote', 'closed_scores', np.diag(closed_fraction), np.zeros(10))
    net.dense('open_vote', 'open_scores', np.diag(1-closed_fraction), np.zeros(10))
    net.layer('geometric_scores', 'add', ['closed_vote', 'open_vote'])

    # Interior construction: a front fastening is a persistent vertical line,
    # rather than the broad, mostly seamless knitted front of a pullover.
    axis = (np.arange(28)+.5)/28
    xx, yy = np.meshgrid(axis, axis)
    def region(x0, x1, y0, y1):
        a = ((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)).astype(float)
        return a / max(a.sum(), 1)
    line_filters = []
    for offset in [-.035, 0, .035, -.18, .18]:
        for y0, y1 in [(.25, .55), (.48, .78)]:
            center = .5+offset
            w = region(center-.025, center+.025, y0, y1)
            w -= .5*region(center-.09, center-.025, y0, y1)
            w -= .5*region(center+.025, center+.09, y0, y1)
            line_filters.extend([w.ravel(), -w.ravel()])
    net.dense('front_line_contrasts', 'contrast', line_filters, np.full(len(line_filters), -.10))
    net.layer('front_line_magnitudes', 'relu', 'front_line_contrasts')
    net.layer('front_lines_by_position', 'reshape', 'front_line_magnitudes', shape=[5, 2, 2])
    net.layer('front_line_absolute', 'maxpool', 'front_lines_by_position', size=[1, 2], stride=[1, 2])
    net.layer('front_line_persistence', 'avgpool', 'front_line_absolute', size=[2, 1], stride=[2, 1])
    net.layer('line_features', 'reshape', 'front_line_persistence', shape=[5])
    net.dense('central_line_values', 'line_features', np.eye(5, dtype=np.float32)[:3], np.zeros(3))
    net.layer('central_line_stack', 'reshape', 'central_line_values', shape=[1, 3, 1])
    net.layer('strongest_center_line', 'maxpool', 'central_line_stack', size=[3, 1], stride=[3, 1])
    net.dense('side_line_baseline', 'line_features', [[0, 0, 0, -.75, -.75]], [-.08])
    net.layer('central_line_scalar', 'reshape', 'strongest_center_line', shape=[1])
    net.layer('front_opening_contrast', 'add', ['central_line_scalar', 'side_line_baseline'])
    net.layer('front_opening', 'relu', 'front_opening_contrast')
    net.dense('front_opening_ramps', 'front_opening', [[1], [1]], [0, -.45])
    net.layer('front_opening_ramp_positive', 'relu', 'front_opening_ramps')
    net.dense('bounded_front_opening', 'front_opening_ramp_positive', [[1, -1]], [0])
    seam_vote = np.zeros((10, 1))
    seam_vote[2] = -.015
    seam_vote[4] = .025
    seam_vote[6] = .13
    net.dense('construction_vote', 'bounded_front_opening', seam_vote, np.zeros(10))

    # Footwear shares an asymmetric profile, but its subclasses differ in
    # enclosure: sandals have more open space per occupied row than boots.
    shoe_select = np.eye(10)[[5, 7, 9]]
    net.dense('shoe_hypotheses', 'geometric_scores', shoe_select, np.zeros(3))
    net.layer('shoe_hypothesis_stack', 'reshape', 'shoe_hypotheses', shape=[1, 3, 1])
    net.layer('shoe_family_match', 'maxpool', 'shoe_hypothesis_stack', size=[3, 1], stride=[3, 1])
    family_vote = np.zeros((10, 1))
    family_vote[5, 0] = .80
    family_vote[7, 0] = .40
    family_vote[9, 0] = .40
    net.dense('shoe_family_vote', 'shoe_family_match', family_vote, np.zeros(10))

    net.dense('foreground_area', 'foreground', [np.ones(784)/784], [0])
    net.layer('occupied_rows', 'maxpool', 'foreground', size=[1, 28], stride=[1, 28])
    net.layer('occupied_height', 'avgpool', 'occupied_rows', size=[28, 1], stride=[28, 1])
    net.dense('height_features', 'occupied_height', [[1], [1], [.55]], np.zeros(3))
    net.dense('area_features', 'foreground_area', [[0], [0], [-1]], np.zeros(3))
    net.layer('enclosure_measures', 'add', ['height_features', 'area_features'])
    net.dense('enclosure_tests', 'enclosure_measures',
              [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 1]], [-.50, .45, -.02, -.08])
    net.layer('enclosure_evidence', 'relu', 'enclosure_tests')
    enclosure_vote = np.zeros((10, 4))
    enclosure_vote[5, 2] = 1.10
    enclosure_vote[7, 0] = -.50
    enclosure_vote[7, 3] = -.30
    enclosure_vote[9, 1] = -.50
    enclosure_vote[9, 3] = -.65
    net.dense('footwear_structure_vote', 'enclosure_evidence', enclosure_vote, np.zeros(10))

    # Fill between vertically aligned pieces. A strap and a sole enclose empty
    # space, whereas a closed shoe upper is substantially solid. The two
    # directional cumulative maxima are exact maxpool operations.
    net.conv('envelope_pad', 'foreground', [[[[1]]]], [0], padding=27)
    net.layer('vertical_envelopes', 'maxpool', 'envelope_pad', size=[28, 1], stride=[1, 1])
    selector_top = np.zeros((784, 55*82), dtype=np.float32)
    selector_bottom = np.zeros_like(selector_top)
    for iy in range(28):
        for ix in range(28):
            selector_top[iy*28+ix, iy*82+27+ix] = 1
            selector_bottom[iy*28+ix, (iy+27)*82+27+ix] = 1
    net.dense('foreground_above', 'vertical_envelopes', selector_top, np.zeros(784))
    net.dense('negative_foreground_below', 'vertical_envelopes', -selector_bottom, np.zeros(784))
    net.layer('envelope_difference', 'add', ['foreground_above', 'negative_foreground_below'])
    net.layer('positive_envelope_difference', 'relu', 'envelope_difference')
    net.dense('upper_envelope_area', 'foreground_above', [np.ones(784)/784], [0])
    net.dense('envelope_excess', 'positive_envelope_difference', [-np.ones(784)/784], [0])
    net.dense('negative_foreground_area', 'foreground_area', [[-1]], [0])
    net.layer('vertical_open_area', 'add', ['upper_envelope_area', 'envelope_excess', 'negative_foreground_area'])
    net.dense('open_space_thresholds', 'vertical_open_area', [[1], [1]], [-.035, -.12])
    net.layer('open_space_evidence', 'relu', 'open_space_thresholds')
    open_space_vote = np.zeros((10, 2))
    open_space_vote[5] = [1.8, -1.8]
    open_space_vote[7] = [-.35, .35]
    open_space_vote[9] = [-.35, .35]
    net.dense('open_space_vote', 'open_space_evidence', open_space_vote, np.zeros(10))

    # Broad garment structure: two lower legs, and a skirt wider than its waist.
    gap = .5*region(.29, .46, .62, .96)+.5*region(.54, .71, .62, .96)-region(.465, .535, .62, .96)
    shoulders = .5*region(.05, .24, .15, .45)+.5*region(.76, .95, .15, .45)
    flare = .5*region(.20, .40, .72, .92)+.5*region(.60, .80, .72, .92)
    flare -= .5*region(.20, .40, .36, .52)+.5*region(.60, .80, .36, .52)
    net.dense('garment_structure_tests', 'shape_mask',
              [(gap-.5*shoulders).ravel(), flare.ravel()], [-.15, -.12])
    net.layer('garment_structure', 'relu', 'garment_structure_tests')
    structure_vote = np.zeros((10, 2))
    structure_vote[1, 0] = .16
    structure_vote[3, 1] = .12
    net.dense('garment_structure_vote', 'garment_structure', structure_vote, np.zeros(10))

    # Fine woven/checkered detail is supporting evidence for a shirt, not a
    # sufficient classifier by itself. Edges are measured inside the torso.
    differences = np.zeros((4, 1, 3, 3))
    differences[0, 0, 1, 1] = 1
    differences[0, 0, 1, 2] = -1
    differences[1] = -differences[0]
    differences[2, 0, 1, 1] = 1
    differences[2, 0, 2, 1] = -1
    differences[3] = -differences[2]
    net.conv('fabric_differences', 'contrast', differences, np.zeros(4), padding=1)
    net.layer('fabric_edges', 'relu', 'fabric_differences')
    texture_region = region(.32, .68, .32, .72)
    net.dense('fabric_texture', 'fabric_edges', [np.tile(texture_region.ravel(), 4)*.5], [-.25])
    net.layer('fabric_texture_positive', 'relu', 'fabric_texture')
    texture_vote = np.zeros((10, 1))
    texture_vote[6] = .065
    texture_vote[2] = -.015
    net.dense('fabric_vote', 'fabric_texture_positive', texture_vote, np.zeros(10))

    # A generic short-sleeved shape has weaker class specificity than a shirt
    # with an observable front fastening. No label-frequency estimates are used.
    preference = np.zeros(10)
    preference[0] = .008
    preference[5] = -.08
    preference[6] = -.006
    scale = np.ones(10)
    scale[[5, 7, 9]] = [.20, .60, .60]
    net.dense('class_conventions', 'geometric_scores', np.diag(scale), preference)
    net.layer('scores', 'add', ['class_conventions', 'construction_vote',
                              'shoe_family_vote', 'footwear_structure_vote',
                              'garment_structure_vote', 'fabric_vote', 'open_space_vote'])

    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers': net.layers}, indent=2)+'\n')
    np.savez(out/'weights.npz', **net.weights)
    print('Wrote', len(net.layers), 'layers and', sum(a.size for a in net.weights.values()), 'float32 weights')


if __name__ == '__main__':
    main(sys.argv[1])
