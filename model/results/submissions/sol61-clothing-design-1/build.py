"""Data-free, hand-designed Fashion classifier.

No files are read.  All silhouettes below are drawings in an abstract physical
coordinate system, not samples, tracings, averages, or descriptors of data.
The seed samples nuisance parameters of those drawings; nothing is optimized.
"""
import json
import sys
from pathlib import Path
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def op(self, name, op, source, **settings):
        self.layers.append(dict(name=name, op=op,
                                inputs=[source] if isinstance(source, str) else source,
                                **settings))
        return name

    def dense(self, name, source, w, b=None):
        w = np.asarray(w, np.float32)
        self.op(name, 'dense', source, out=len(w))
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.zeros(len(w), np.float32) if b is None else np.asarray(b, np.float32)
        return name

    def conv(self, name, source, w, b=None, padding=0):
        w = np.asarray(w, np.float32)
        self.op(name, 'conv', source, out=len(w), kernel=list(w.shape[-2:]), stride=1, padding=padding)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.zeros(len(w), np.float32) if b is None else np.asarray(b, np.float32)
        return name


def polygon(points):
    return ('polygon', np.asarray(points, float))


def ellipse(cx, cy, rx, ry):
    # A polygonal ellipse makes rotating the whole drawing especially simple.
    a = np.arange(40) * (2 * np.pi / 40)
    return polygon(np.column_stack((cx + rx * np.cos(a), cy + ry * np.sin(a))))


def stroke(points, width):
    result = []
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a), np.asarray(b)
        d = b - a
        v = np.array([-d[1], d[0]]) * (width / (2 * max(np.linalg.norm(d), 1e-9)))
        result.append(polygon([a + v, b + v, b - v, a - v]))
    return result


def mirrored(points):
    return [(-x, y) for x, y in points]


def garment(r, cls):
    positive, negative = [], []
    if cls == 0:
        tank = r.random() < .18
        shoulder = r.uniform(.30, .40) if not tank else r.uniform(.18, .26)
        body = r.uniform(.25, .35)
        waist = body * r.uniform(.85, 1.10)
        hem_y = 1.
        cuff_y = r.uniform(.26, .42)
        cuff_x = shoulder + r.uniform(.12, .23)
        cuff_w = r.uniform(.12, .20)
        neck = r.uniform(.085, .15)
        neck_depth = r.uniform(.045, .14) if not tank else r.uniform(.10, .24)
        sleeved = not tank
        vneck = r.random() < .25
    elif cls == 2:
        shoulder = r.uniform(.32, .40)
        body = r.uniform(.265, .34)
        waist = body * r.uniform(.80, 1.02)
        hem_y = r.uniform(.87, .99)
        cuff_y = 1.
        cuff_x = shoulder + r.uniform(.03, .15)
        cuff_w = r.uniform(.105, .16)
        neck = r.uniform(.065, .13)
        neck_depth = r.uniform(.025, .10)
        sleeved = True
        vneck = r.random() < .12
    elif cls == 4:
        # Jackets as well as overcoats: some have a boxy, waist-length body,
        # others extend well below the cuffs. They need not all flare out.
        shoulder = r.uniform(.30, .39)
        body = r.uniform(.255, .345)
        waist = body * r.uniform(.87, 1.12)
        hem_y = r.uniform(.93, 1.)
        # A longer overcoat's skirt continues below the jacket sleeves.
        cuff_y = r.uniform(.62, .84) if r.random() < .25 else r.uniform(.84, 1.00)
        cuff_x = shoulder + r.uniform(.035, .13)
        cuff_w = r.uniform(.095, .16)
        neck = r.uniform(.07, .13)
        neck_depth = r.uniform(.045, .19)
        sleeved = True
        vneck = True
    else:
        shoulder = r.uniform(.31, .395)
        body = r.uniform(.255, .34)
        waist = body * r.uniform(.88, 1.07)
        hem_y = r.uniform(.91, 1.)
        short = r.random() < .14
        cuff_y = r.uniform(.29, .44) if short else 1.
        cuff_x = shoulder + r.uniform(.10, .20) if short else shoulder + r.uniform(.04, .14)
        cuff_w = r.uniform(.12, .18) if short else r.uniform(.085, .14)
        neck = r.uniform(.08, .14)
        neck_depth = r.uniform(.075, .18)
        sleeved = True
        vneck = r.random() < .8
        # A woven button shirt can have sleeves hanging straight down beside
        # a narrow torso, rather than the diagonal sleeve of a close-fit knit.
        if r.random() < .55:
            cuff_x = shoulder + r.uniform(-.01, .07)
            cuff_w = r.uniform(.075, .115)
            body = min(body, shoulder-r.uniform(.025, .075))
    armpit = r.uniform(.28, .38)
    # Body, shoulder, and sleeves are separate pieces as in a sewing pattern.
    tail = r.uniform(.015, .075) if cls == 6 else r.uniform(0, .025)
    torso = [(-neck, 0), (-shoulder, .055), (-body, armpit),
             (-waist, hem_y-tail), (0, hem_y), (waist, hem_y-tail),
             (body, armpit), (shoulder, .055), (neck, 0)]
    positive.append(polygon(torso))
    if sleeved:
        if cuff_y > .65:
            # An explicit elbow bends the sleeve back toward the torso. Only
            # some hanging garments have a separated cuff; it is optional.
            separation = r.uniform(0, .03)
            if cls == 6:
                separation += .008
            if r.random() < .32:
                cuff_x = max(cuff_x, max(body, waist)+cuff_w+separation)
            elbow_y = .5*(armpit+cuff_y)
            elbow_x = max(shoulder, cuff_x)+r.uniform(0, .02)
            inner_elbow = .5*(body+cuff_x-cuff_w)+separation*.4
            arm = [(shoulder, .055), (elbow_x, elbow_y), (cuff_x, cuff_y),
                   (cuff_x-cuff_w, cuff_y+.015), (inner_elbow, elbow_y), (body, armpit)]
        else:
            arm = [(shoulder, .055), (cuff_x, cuff_y),
                   (cuff_x-cuff_w, cuff_y+.015), (body, armpit)]
        positive.extend([polygon(arm), polygon(mirrored(arm))])
    if vneck:
        negative.append(polygon([(-neck, -.025), (neck, -.025), (0, neck_depth)]))
    else:
        negative.append(ellipse(0, .005, neck, neck_depth))
    # Front-opening outerwear can have a split below the fastening.
    if cls == 4 and r.random() < .32:
        opening = r.uniform(.03, .12)
        start = r.uniform(.35, .77)
        negative.append(polygon([(0, start), (-opening, 1.02), (opening, 1.02)]))
    return positive, negative


def dress(r):
    shoulder = r.uniform(.16, .28)
    chest = r.uniform(.16, .245)
    waist = r.uniform(.145, .24)
    waist_y = r.uniform(.33, .51)
    fitted = r.random() < .38
    hem = r.uniform(.23, .33) if fitted else r.uniform(.33, .55)
    neck = r.uniform(.06, .12)
    body = [(-neck, 0), (-shoulder, .045), (-chest, .23),
            (-waist, waist_y), (-hem, 1), (hem, 1), (waist, waist_y),
            (chest, .23), (shoulder, .045), (neck, 0)]
    positive = [polygon(body)]
    u = r.random()
    if u > .47:
        long = u > .86
        y = r.uniform(.65, .86) if long else r.uniform(.18, .30)
        x = shoulder + r.uniform(.055, .15)
        arm = [(shoulder, .045), (x, y), (x-r.uniform(.075, .12), y+.02), (chest, .23)]
        positive.extend([polygon(arm), polygon(mirrored(arm))])
    depth = r.uniform(.035, .13)
    negative = [ellipse(0, .005, neck, depth)]
    return positive, negative


def trousers(r):
    hip = r.uniform(.17, .285)
    fork = r.uniform(.25, .43)
    # A trouser's outer seams generally stay within its hip width; the inner
    # gap may be very narrow for parallel legs. Choose these independently of
    # leg thickness so skinny trousers are not accidentally excluded.
    inner_gap = r.uniform(.008, .075)
    outer = hip * r.uniform(.64, 1.08)
    bend = r.uniform(-.015, .015)
    right_outer, right_inner = outer+bend, inner_gap+bend
    left_outer, left_inner = outer-bend, inner_gap-bend
    p = [(-hip*.92, 0), (hip*.92, 0), (hip, .23),
         (right_outer, 1), (right_inner, 1), (0, fork),
         (-left_inner, 1), (-left_outer, 1), (-hip, .23)]
    return [polygon(p)], []


def bag(r):
    positive, negative = [], []
    height = r.uniform(.45, .90)
    handle_height = r.uniform(.11, .45)
    top = r.uniform(.32, .48)
    bottom = r.uniform(.34, .50)
    if r.random() < .15:  # a clutch / flat shoulder bag
        handle_height *= .20
        height *= .75
    rounding = r.uniform(.025, .11)
    y = handle_height
    positive.append(polygon([(-top+rounding, y), (top-rounding, y),
                             (top, y+rounding), (bottom, y+height-rounding),
                             (bottom-rounding, y+height), (-bottom+rounding, y+height),
                             (-bottom, y+height-rounding), (-top, y+rounding)]))
    handle_width = r.uniform(.13, .32)
    strap = r.uniform(.022, .055)
    if handle_height > .06:
        positive.append(ellipse(0, handle_height, handle_width, handle_height))
        negative.append(ellipse(0, handle_height-.012, max(.03, handle_width-strap), max(.02, handle_height-strap)))
        # Restrict the handle hole to the region above the bag's body.
        # The renderer applies these holes before adding the body back below.
        positive.append(polygon([(-top+rounding, y), (top-rounding, y),
                                 (top, y+rounding), (bottom, y+height-rounding),
                                 (bottom-rounding, y+height), (-bottom+rounding, y+height),
                                 (-bottom, y+height-rounding), (-top, y+rounding)]))
    return positive, negative


def sneaker(r):
    h = r.uniform(.28, .52)
    sole = r.uniform(.045, .085)
    toe = h * r.uniform(.47, .66)
    instep = r.uniform(.43, .64)
    ankle = r.uniform(.14, .36) * h
    p = [(0, h-sole*.5), (.015, toe+.055), (.10, toe), (.27, toe*.92),
         (instep, .03), (.69, ankle), (.86, max(0, ankle-.05)),
         (.97, ankle+.01), (1, h-sole), (.97, h), (.065, h)]
    return [polygon(p)], []


def boot(r):
    h = r.uniform(.58, 1.12)
    shaft_left = r.uniform(.46, .63)
    shaft_right = r.uniform(.84, .98)
    heel = r.uniform(.035, .24)
    thin = r.random() < .35
    foot_top = h-heel-r.uniform(.13, .22)
    p = [(0, h-heel-.03), (.025, foot_top+.055), (.15, foot_top),
         (.33, foot_top-.035), (shaft_left-.02, foot_top-.14),
         (shaft_left, .015), (shaft_right, 0), (shaft_right+.015, h-heel),
         (.82, h-heel+.035), (.48, h-heel+.045), (.08, h-heel+.055)]
    positive = [polygon(p)]
    if thin:
        positive.append(polygon([(.82, h-heel), (.96, h-heel), (.92, h), (.865, h)]))
        positive.extend(stroke([(.04, h-heel+.035), (.40, h-heel+.055)], .04))
    else:
        positive.append(polygon([(.05, h-heel), (.96, h-heel), (.94, h), (.07, h)]))
    return positive, []


def sandal(r):
    h = r.uniform(.20, .93)
    heel = r.uniform(.01, min(.32, h*.52))
    sole = r.uniform(.025, .06)
    ytoe = h-sole
    yheel = h-heel
    # A sole, and two crossing bands: deliberately no opaque shoe upper.
    positive = [polygon([(0, ytoe-.025), (.27, ytoe-.02), (.78, yheel-.025),
                         (1, yheel-.005), (.98, yheel+sole), (.75, yheel+sole),
                         (.26, ytoe+sole), (.015, ytoe+sole)])]
    width = r.uniform(.035, .10)
    strap_top = max(.015, ytoe-r.uniform(.12, min(.38, h*.90)))
    positive.extend(stroke([(.12, ytoe), (.29, strap_top), (.50, ytoe-.04)], width))
    if r.random() < .85:
        ankle_top = max(0., h-r.uniform(.18, min(.80, h)))
        positive.extend(stroke([(.45, ytoe-.04), (.80, ankle_top), (.94, yheel)], width))
        if r.random() < .55:
            positive.extend(stroke([(.73, ankle_top+.035), (.95, ankle_top+.045)], width*.85))
    if heel > .07:
        heelwidth = r.uniform(.035, .16)
        positive.append(polygon([(.91-heelwidth, yheel), (.98, yheel), (.96, h), (.94-heelwidth, h)]))
    # Wedge and broad-band sandals are an ordinary alternate construction.
    if r.random() < .20:
        positive.append(polygon([(.25, ytoe), (.95, yheel), (.94, h), (.20, h)]))
    return positive, []


def inside_polygon(x, y, points):
    hit = np.zeros(x.shape, bool)
    a = points[-1]
    for b in points:
        cross = ((a[1] > y) != (b[1] > y))
        den = b[1] - a[1]
        if abs(den) > 1e-12:
            boundary = (b[0]-a[0]) * (y-a[1]) / den + a[0]
            hit ^= cross & (x < boundary)
        a = b
    return hit


# Analytic antialiasing grid. These are canvas coordinates, never image readings.
GRID_Y, GRID_X = np.meshgrid((np.arange(56)+.5)/2-.5,
                            (np.arange(56)+.5)/2-.5, indexing='ij')


def draw_template(positive, negative, r, cls):
    allpoints = np.concatenate([p for _, p in positive])
    lo, hi = allpoints.min(0), allpoints.max(0)
    center = (lo+hi)/2
    scale = r.uniform(25.3, 28.2) / max(hi-lo)
    angle = r.uniform(-.065, .065)
    cs, sn = np.cos(angle), np.sin(angle)
    rot = np.array([[cs, -sn], [sn, cs]])
    flip = -1 if r.random() < .5 else 1
    # Keep the complete physical item on the canvas. A short shoe may occupy
    # a low position, but a full-height garment cannot be shifted off-screen.
    # This constraint depends only on the synthetic drawing's own geometry.
    tx = r.uniform(-1.0, 1.0)
    rotated = ((allpoints-center)*[flip, 1]) @ rot.T * scale
    halfheight = np.max(np.abs(rotated[:, 1]))
    margin = max(.10, 13.7-halfheight)
    ty = r.uniform(-margin, margin)
    if cls in (5, 7, 9) and r.random() < .6:
        ty = r.uniform(0, margin)
    if r.random() < .35:
        ty *= .3
    def transform(points):
        p = (points-center) * [flip, 1]
        return (p @ rot.T) * scale + [13.5+tx, 13.5+ty]
    mask = np.zeros((56, 56), bool)
    for _, points in positive:
        mask |= inside_polygon(GRID_X, GRID_Y, transform(points))
    for _, points in negative:
        hole = inside_polygon(GRID_X, GRID_Y, transform(points))
        if cls == 8:
            # Handle holes do not cut out the bag body.
            # Evaluate the hole in the drawing's unrotated vertical coordinate.
            qx = GRID_X-(13.5+tx)
            qy = GRID_Y-(13.5+ty)
            model_y = (-qx*sn+qy*cs)/scale+center[1]
            body_top = positive[0][1][:, 1].min()
            hole &= model_y < body_top
        mask &= ~hole
    raw28 = mask.reshape(28, 2, 28, 2).mean((1, 3)).astype(np.float32)
    raw = raw28.reshape(14, 2, 14, 2).mean((1, 3))
    # Formula-drawn masks are thickened by one pixel to define a second,
    # tolerant shape metric. This never involves input/training images.
    pad = np.pad(raw28, 1)
    thick28 = np.maximum.reduce([pad[dy:dy+28, dx:dx+28]
                                for dy in range(3) for dx in range(3)])
    thick = thick28.reshape(14, 2, 14, 2).mean((1, 3))
    opacity = r.uniform(.88, 1.)
    # Small generic nuisance prior, not a data-estimated frequency.
    prior = -.0015*(tx/1.2)**2-.003*(ty/4.8)**2
    return raw*opacity, thick*opacity, prior


def templates(per_class=2048):
    r = np.random.default_rng(981723)
    aa, bb, priors = [], [], []
    for cls in range(10):
        for _ in range(per_class):
            if cls in (0, 2, 4, 6):
                p, n = garment(r, cls)
            elif cls == 1:
                p, n = trousers(r)
            elif cls == 3:
                p, n = dress(r)
            elif cls == 5:
                p, n = sandal(r)
            elif cls == 7:
                p, n = sneaker(r)
            elif cls == 8:
                p, n = bag(r)
            else:
                p, n = boot(r)
            a, b, prior = draw_template(p, n, r, cls)
            aa.append(a.ravel())
            bb.append(b.ravel())
            priors.append(prior)
    return np.asarray(aa, np.float32), np.asarray(bb, np.float32), np.asarray(priors, np.float32)


def rectangle_weights(y0, y1, x0, x1):
    a = np.zeros((14, 14), np.float32)
    a[y0:y1, x0:x1] = 1 / ((y1-y0)*(x1-x0))
    return a


def appearance(net):
    # Signed finite differences. Absolute values are the sum of the two ReLUs.
    filters = np.zeros((8, 1, 5, 5), np.float32)
    filters[0, 0, 2, 1], filters[0, 0, 2, 3] = -.5, .5
    filters[2, 0, 2, 0], filters[2, 0, 2, 4] = -.5, .5
    filters[4, 0, 1, 2], filters[4, 0, 3, 2] = -.5, .5
    filters[6, 0, 2, 2] = 1
    for y, x in ((1, 2), (3, 2), (2, 1), (2, 3)):
        filters[6, 0, y, x] = -.25
    for i in (0, 2, 4, 6):
        filters[i+1] = -filters[i]
    net.conv('fabric_differences', 'standardized', filters, padding=2)
    net.op('fabric_rectified', 'relu', 'fabric_differences')
    net.op('fabric_local', 'avgpool', 'fabric_rectified', size=[2, 2], stride=[2, 2])
    edge_w = np.zeros((12, 8, 14, 14), np.float32)
    occ_w = np.zeros((12, 1, 14, 14), np.float32)
    # The three consecutive bands require a persistent placket, not one logo.
    for j, (y0, y1) in enumerate(((3, 6), (6, 9), (9, 12))):
        center = rectangle_weights(y0, y1, 6, 8)
        flank = .5*(rectangle_weights(y0, y1, 4, 6)+rectangle_weights(y0, y1, 8, 10))
        for c in (0, 1):
            edge_w[j, c] = center-flank
        for c in (2, 3):
            edge_w[3+j, c] = center-flank
        occ_w[9+j, 0] = flank-center
    interior = rectangle_weights(4, 11, 4, 10)
    for j in (6, 7, 8):
        edge_w[j, 6] = interior
        edge_w[j, 7] = interior
    net.dense('appearance_from_edges', 'fabric_local', edge_w.reshape(12, -1))
    net.dense('appearance_from_shape', 'foreground_cells', occ_w.reshape(12, -1))
    net.op('appearance_measurements', 'add', ['appearance_from_edges', 'appearance_from_shape'])
    start = np.array([.06]*6+[.16]*3+[.13]*3, np.float32)
    span = np.array([.28]*6+[.35]*3+[.48]*3, np.float32)
    w = np.concatenate((np.diag(1/span), np.diag(1/span)))
    b = np.concatenate((-start/span, -start/span-1))
    net.dense('bounded_evidence_pre', 'appearance_measurements', w, b)
    net.op('bounded_evidence_relu', 'relu', 'bounded_evidence_pre')
    net.dense('bounded_evidence', 'bounded_evidence_relu', np.concatenate((np.eye(12), -np.eye(12)), axis=1))
    net.op('evidence_bands', 'reshape', 'bounded_evidence', shape=[4, 1, 3])
    net.conv('negated_evidence', 'evidence_bands', -np.eye(4).reshape(4, 4, 1, 1))
    net.op('persistent_evidence', 'maxpool', 'negated_evidence', size=[1, 3], stride=[1, 3])
    # Negated minima arrive here. These coefficients express only qualitative
    # apparel knowledge: an uninterrupted opening favors a coat or shirt;
    # fine fabric texture and a narrow placket favor a woven shirt.
    rules = np.zeros((10, 4), np.float32)
    rules[2] = [.025, .015, .018, .025]
    rules[4] = [-.018, -.025, 0, -.060]
    rules[6] = [-.040, -.025, -.045, -.025]
    bias = np.zeros(10, np.float32)
    bias[2] = 0
    bias[4] = 0
    bias[6] = -.009
    net.dense('apparel_evidence', 'persistent_evidence', rules, bias)


def structural_evidence(net):
    # Whitespace separating two trouser legs: compare the lower central strip
    # against its two flanks. Averaging several strips tolerates modest shifts.
    w = np.zeros((4, 196), np.float32)
    for j, (y0, y1) in enumerate(((7, 12), (8, 13), (9, 14))):
        center = rectangle_weights(y0, y1, 6, 8)
        side = .5*(rectangle_weights(y0, y1, 4, 6)+rectangle_weights(y0, y1, 8, 10))
        w[j] = (side-center).ravel()
    # One gap on just one side can be a handbag strap or garment background;
    # a symmetric split extending through several rows is much stronger.
    net.dense('leg_split_pre', 'foreground_cells', w, [-.16, -.16, -.16, 0])
    net.op('leg_split_positive', 'relu', 'leg_split_pre')
    net.dense('leg_split_negative', 'leg_split_positive', -np.eye(4))
    net.op('leg_split_groups', 'reshape', 'leg_split_negative', shape=[1, 1, 4])
    # The fourth dummy coordinate is ignored. Use only the three leg strips.
    net.op('leg_split_min', 'maxpool', 'leg_split_groups', size=[1, 3], stride=[1, 3])
    rule = np.zeros((10, 1), np.float32)
    rule[1, 0] = -.14
    rule[3, 0] = .08
    net.dense('leg_split_scores', 'leg_split_min', rule)

    # A trouser hypothesis requires a split; without any split, a long narrow
    # uninterrupted body is instead evidence for a fitted dress or skirt.
    gap_w = np.zeros((1, 196), np.float32)
    gap_w[0] = (.5*(rectangle_weights(8, 13, 4, 6)+rectangle_weights(8, 13, 8, 10))-
                rectangle_weights(8, 13, 6, 8)).ravel()
    net.dense('unsplit_pre', 'foreground_cells', -gap_w, [.12])
    net.op('unsplit_body', 'relu', 'unsplit_pre')
    rule = np.zeros((10, 1), np.float32)
    rule[1, 0] = -.06
    net.dense('unsplit_scores', 'unsplit_body', rule)

    # Open footwear has much more air within its vertical extent than a closed
    # shoe. Row maxima are a soft height measure, not a data-derived box.
    net.op('occupied_rows', 'maxpool', 'foreground_cells', size=[1, 14], stride=[1, 14])
    net.dense('footwear_height', 'occupied_rows', np.full((1, 14), .52/14, np.float32))
    net.dense('negative_foreground_area', 'foreground_cells', np.full((1, 196), -1/196, np.float32))
    net.op('footwear_air_pre', 'add', ['footwear_height', 'negative_foreground_area'])
    net.op('footwear_air', 'relu', 'footwear_air_pre')
    rule = np.zeros((10, 1), np.float32)
    rule[5, 0] = .65
    rule[7, 0] = -.18
    rule[9, 0] = -.18
    net.dense('footwear_air_scores', 'footwear_air', rule)

    # Thin bands have a higher dilation-to-area ratio than opaque uppers.
    # Subtract a row-height term so the outside boundary of a solid boot is
    # not itself mistaken for multiple sandal straps.
    net.dense('strap_dilation_area', 'thickened_cells', np.full((1, 196), 1/196, np.float32))
    net.dense('strap_original_area', 'foreground_cells', np.full((1, 196), -1.32/196, np.float32))
    net.dense('strap_height_correction', 'occupied_rows', np.full((1, 14), -.11/14, np.float32), [-.025])
    net.op('strap_pre', 'add', ['strap_dilation_area', 'strap_original_area', 'strap_height_correction'])
    net.op('strap_evidence', 'relu', 'strap_pre')
    rule = np.zeros((10, 1), np.float32)
    rule[5, 0] = .9
    rule[7, 0] = -.1
    rule[9, 0] = -.1
    net.dense('strap_scores', 'strap_evidence', rule)

    # Outer sleeves reaching down beside the waist contradict a short-sleeved
    # tee. The response is capped, since loose sleeveless items also exist.
    lower_sleeve = .5*(rectangle_weights(6, 11, 0, 3)+rectangle_weights(6, 11, 11, 14))
    net.dense('lower_sleeve_pre', 'foreground_cells',
              np.stack([lower_sleeve.ravel(), lower_sleeve.ravel()]), [-.08, -.33])
    net.op('lower_sleeve_positive', 'relu', 'lower_sleeve_pre')
    rules = np.zeros((10, 2), np.float32)
    rules[0] = [-.07, .07]
    net.dense('sleeve_length_scores', 'lower_sleeve_positive', rules)

    # An opening has aligned edge energy but little transverse edge energy;
    # checked/striped fabric should not by itself count as an open coat.
    region = rectangle_weights(3, 12, 5, 9)
    ew = np.zeros((2, 8, 14, 14), np.float32)
    for c in (0, 1):
        ew[:, c] = region
    for c in (4, 5):
        ew[:, c] = -region
    net.dense('vertical_front_pre', 'fabric_local', ew.reshape(2, -1), [-.06, -.36])
    net.op('vertical_front_positive', 'relu', 'vertical_front_pre')
    rules = np.zeros((10, 2), np.float32)
    rules[2] = [-.025, .025]
    rules[4] = [.09, -.09]
    rules[6] = [.06, -.06]
    net.dense('front_opening_scores', 'vertical_front_positive', rules)

    # A short sleeve projects outwards in the upper body but not beside the
    # waist. Persistence across both sides rejects an isolated protrusion.
    sw = np.zeros((4, 196), np.float32)
    for j, (x0, x1) in enumerate(((0, 3), (11, 14))):
        upper = rectangle_weights(2, 6, x0, x1)
        lower = rectangle_weights(7, 11, x0, x1)
        sw[j] = (upper-lower).ravel()
        sw[j+2] = (upper-lower).ravel()
    net.dense('short_sleeve_pre', 'foreground_cells', sw, [-.12, -.12, -.62, -.62])
    net.op('short_sleeve_pos', 'relu', 'short_sleeve_pre')
    net.dense('short_sleeve_negative', 'short_sleeve_pos', -np.eye(4))
    net.op('short_sleeve_pairs', 'reshape', 'short_sleeve_negative', shape=[2, 1, 2])
    net.op('short_sleeve_min', 'maxpool', 'short_sleeve_pairs', size=[1, 2], stride=[1, 2])
    rules = np.zeros((10, 2), np.float32)
    rules[0] = [-.08, .08]
    rules[2] = [.015, -.015]
    rules[4] = [.015, -.015]
    rules[6] = [.02, -.02]
    net.dense('short_sleeve_scores', 'short_sleeve_min', rules)

    # A shirt collar consists of paired oblique edges. The response is averaged
    # over a broad neckline region, not tied to a particular input's outline.
    diagonal = np.zeros((4, 1, 3, 3), np.float32)
    diagonal[0, 0] = np.array([[-.25, -.25, 0], [-.25, 0, .25], [0, .25, .25]])
    diagonal[1] = -diagonal[0]
    diagonal[2, 0] = np.fliplr(diagonal[0, 0])
    diagonal[3] = -diagonal[2]
    net.conv('collar_diagonals', 'standardized', diagonal, padding=1)
    net.op('collar_diagonals_abs', 'relu', 'collar_diagonals')
    net.op('collar_diagonals_local', 'avgpool', 'collar_diagonals_abs', size=[2, 2], stride=[2, 2])
    cw = np.zeros((2, 4, 14, 14), np.float32)
    collar_region = rectangle_weights(1, 4, 4, 10)
    for c in range(4):
        cw[:, c] = collar_region*.5
    net.dense('collar_pre', 'collar_diagonals_local', cw.reshape(2, -1), [-.32, -.72])
    net.op('collar_positive', 'relu', 'collar_pre')
    rules = np.zeros((10, 2), np.float32)
    rules[6] = [.035, -.035]
    rules[4] = [.015, -.015]
    net.dense('collar_scores', 'collar_positive', rules)


def build(destination):
    net = Network()
    net.op('standardized', 'norm', 'image')
    net.conv('negative_standardized', 'standardized', [[[[-1]]]])
    net.op('background_offset', 'maxpool', 'negative_standardized', size=[28, 28], stride=[28, 28])
    net.dense('background_broadcast_flat', 'background_offset', np.ones((784, 1)))
    net.op('background_broadcast', 'reshape', 'background_broadcast_flat', shape=[1, 28, 28])
    net.op('contrast_scaled', 'add', ['standardized', 'background_broadcast'])
    # Since the darkest background is zero, the preceding operation is x/std.
    # A clipped ramp gives a soft occupancy independent of garment brightness.
    slope = 1/.50
    net.conv('foreground_ramps', 'contrast_scaled', np.array([slope, slope]).reshape(2, 1, 1, 1), [-.10*slope, -.10*slope-1])
    net.op('foreground_ramp_relu', 'relu', 'foreground_ramps')
    net.conv('foreground', 'foreground_ramp_relu', np.array([1, -1]).reshape(1, 2, 1, 1))
    net.op('foreground_cells', 'avgpool', 'foreground', size=[2, 2], stride=[2, 2])
    net.conv('padded_foreground', 'foreground', [[[[1]]]], padding=1)
    net.op('thickened_foreground', 'maxpool', 'padded_foreground', size=[3, 3], stride=[1, 1])
    net.op('thickened_cells', 'avgpool', 'thickened_foreground', size=[2, 2], stride=[2, 2])
    count = 2048
    a, b, prior = templates(count)
    # The shared input-squared term in squared Euclidean distance cancels.
    # Thus each generative-model score is exactly a fixed affine layer.
    alpha, beta = .40/196, .60/196
    net.dense('drawing_scores_fine', 'foreground_cells', 2*alpha*a,
              -alpha*(a*a).sum(1)+prior)
    net.dense('drawing_scores_tolerant', 'thickened_cells', 2*beta*b,
              -beta*(b*b).sum(1))
    net.op('drawing_scores', 'add', ['drawing_scores_fine', 'drawing_scores_tolerant'])
    net.op('drawings_by_class', 'reshape', 'drawing_scores', shape=[10, 1, count])
    net.op('best_drawing_per_class', 'maxpool', 'drawings_by_class', size=[1, count], stride=[1, count])
    net.op('shape_scores', 'reshape', 'best_drawing_per_class', shape=[10])
    appearance(net)
    structural_evidence(net)
    net.op('scores', 'add', ['shape_scores', 'apparel_evidence', 'leg_split_scores', 'unsplit_scores', 'footwear_air_scores', 'strap_scores', 'sleeve_length_scores', 'front_opening_scores', 'short_sleeve_scores', 'collar_scores'])
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers': net.layers}, indent=2)+'\n')
    np.savez(out/'weights.npz', **net.weights)
    print('Wrote', len(net.layers), 'layers;', sum(w.size for w in net.weights.values()), 'weights')


if __name__ == '__main__':
    build(sys.argv[1])
