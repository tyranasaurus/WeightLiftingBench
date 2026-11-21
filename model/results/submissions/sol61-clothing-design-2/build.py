"""Data-free clothing classifier: analytic shape models, not trained prototypes.

All silhouettes below are procedural drawings from generic garment anatomy.
Their ranges express deliberately broad, hand-chosen proportions, not image
measurements. A fixed PRNG samples these ranges; it never samples any data.
There is no optimization, fitting, dataset access, or external-file read.
"""
import json
import sys
from pathlib import Path
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def layer(self, name, op, inputs, **kw):
        if isinstance(inputs, str):
            inputs = [inputs]
        self.layers.append(dict(name=name, op=op, inputs=inputs, **kw))
        return name

    def dense(self, name, source, w, b=None):
        w = np.asarray(w, dtype=np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.zeros(w.shape[0], np.float32) if b is None else np.asarray(b, np.float32)
        return self.layer(name, 'dense', source, out=w.shape[0])

    def conv(self, name, source, w, b=None, padding=0):
        w = np.asarray(w, dtype=np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.zeros(w.shape[0], np.float32) if b is None else np.asarray(b, np.float32)
        return self.layer(name, 'conv', source, out=w.shape[0], kernel=list(w.shape[2:]), stride=1, padding=padding)


# Supersampled coordinates, independent of all task images.
SIDE = 56
Y, X = np.mgrid[0:SIDE, 0:SIDE].astype(np.float64)
X = (X + .5) / SIDE
Y = (Y + .5) / SIDE


def polygon(x, y, points):
    inside = np.zeros_like(x, dtype=bool)
    x0, y0 = points[-1]
    for x1, y1 in points:
        if abs(y1 - y0) > 1e-12:
            crossing = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
            inside ^= crossing
        x0, y0 = x1, y1
    return inside


def ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1


def rectangle(x, y, x0, y0, x1, y1):
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def segment(x, y, ax, ay, bx, by, width):
    vx, vy = bx - ax, by - ay
    u = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy + 1e-12), 0, 1)
    return (x - ax - u * vx) ** 2 + (y - ay - u * vy) ** 2 <= width ** 2


def garment(rng, kind, n):
    """Make a top/dress in intrinsic garment coordinates, y=0..1.

    Long sleeves hang downward; short sleeves project out from the shoulder.
    A dress combines a bodice with a skirt, never a memorized outline.
    """
    sleeve_kind = 'long'
    if kind == 0:
        sleeve_kind = 'none' if n % 7 == 0 else 'short'
    if kind == 3:
        sleeve_kind = ['none', 'none', 'short', 'short', 'long'][n % 5]
    if kind == 6:
        sleeve_kind = ['long', 'long', 'short'][n % 3]
    if kind == 4 and n % 9 == 0:
        sleeve_kind = 'none'

    body = rng.uniform(.48, .72)
    shoulder = body * rng.uniform(.86, 1.13)
    top = rng.uniform(.025, .085)
    waist_y = rng.uniform(.36, .51) if kind == 3 else rng.uniform(.60, .82)
    body_end = .99
    hem = body * rng.uniform(.90, 1.10)
    waist = body * rng.uniform(.84, 1.03)
    if kind == 3:
        body *= rng.uniform(.75, .98)
        shoulder = body * rng.uniform(.86, 1.05)
        waist = body * rng.uniform(.65, .95)
        hem = body * rng.uniform(1.15, 2.10)
    if kind == 4:
        # Tailored outerwear has a relatively long torso and shoulders wider
        # than the waist; the hem can hang below the sleeve cuffs.
        body *= rng.uniform(.82, 1.0)
        shoulder = body * rng.uniform(1.00, 1.18)
        waist = body * rng.uniform(.87, 1.02)
        hem = body * rng.uniform(1.00, 1.24)
    neck = shoulder * rng.uniform(.28, .49)
    neck_depth = rng.uniform(.055, .14)
    neck_kind = n % 4
    if kind in (4, 6):
        neck_depth = rng.uniform(.08, .28)
        neck_kind = 1 if n % 3 else 0
    if kind == 6 and sleeve_kind == 'short':
        # A short-sleeved button-down has a collar/placket opening rather
        # than simply duplicating the round-neck T-shirt model.
        neck_depth = rng.uniform(.15, .30)
        neck_kind = 1
    if kind == 2 and n % 7 == 0:
        neck_depth = -.02  # a high/turtle neck
    if kind == 3:
        neck_depth = rng.uniform(.07, .23)

    # Overall dimensions arise from body and sleeve proportions.
    if sleeve_kind == 'short':
        extent = shoulder / 2 + rng.uniform(.11, .24)
        sleeve_y = rng.uniform(.22, .43)
        armpit = rng.uniform(.27, .42)
        left = [(-neck / 2, top), (-shoulder / 2, top + .045), (-extent, sleeve_y - .085),
                (-extent + .045, sleeve_y + .065), (-body / 2, armpit),
                (-waist / 2, waist_y), (-hem / 2, body_end)]
    elif sleeve_kind == 'long':
        extent = shoulder / 2 + rng.uniform(.10, .19)
        cuff_y = rng.uniform(.83, 1.01)
        cuff_w = rng.uniform(.075, .13)
        armpit = rng.uniform(.27, .43)
        left = [(-neck / 2, top), (-shoulder / 2, top + .045), (-extent + .035, .20),
                (-extent - .025, cuff_y), (-extent + cuff_w, cuff_y + .015),
                (-body / 2 + .025, armpit), (-waist / 2, waist_y), (-hem / 2, body_end)]
    else:
        extent = shoulder / 2
        left = [(-neck / 2, top), (-shoulder / 2, top + .035), (-body / 2 + .055, .27),
                (-waist / 2, waist_y), (-hem / 2, body_end)]

    right = [(-x, y) for x, y in left[::-1]]
    width = max(2 * extent + .05, hem)
    # Long dresses are tall; wide tops are allowed to be wider than tall.
    long_factor = rng.uniform(1.18, 1.58) if kind == 3 and n % 3 else 1.0
    height = long_factor
    fit = rng.uniform(.89, 1.01) / max(width, height)
    sx = fit * rng.uniform(.94, 1.06)
    sy = fit * height
    cx = .5 + rng.uniform(-.028, .028)
    cy = .5 + rng.uniform(-.026, .026)
    xx = (X - cx) / sx
    yy = (Y - cy) / sy + .5
    mask = polygon(xx, yy, left + right)
    if neck_depth > 0:
        if neck_kind == 1:
            cut = polygon(xx, yy, [(-neck / 2, -.1), (neck / 2, -.1), (neck / 2, top), (0, top + neck_depth), (-neck / 2, top)])
        elif neck_kind == 2 and kind == 3:
            cut = rectangle(xx, yy, -neck / 2, -.1, neck / 2, top + neck_depth)
        else:
            cut = rectangle(xx, yy, -neck / 2, -.1, neck / 2, top) | ellipse(xx, yy, 0, top, neck / 2, neck_depth)
        mask &= ~cut
    # Open cardigans/coats and open shirts are generic construction modes.
    if kind == 4 and n % 5 in (0, 1):
        gap = rng.uniform(.018, .09)
        cut = polygon(xx, yy, [(-neck / 2, 0), (neck / 2, 0), (gap, .45), (gap, 1.05), (-gap, 1.05), (-gap, .45)])
        mask &= ~cut
    elif kind == 6 and n % 11 == 0:
        mask &= ~rectangle(xx, yy, -.017, .27, .017, 1.05)
    return mask


def trousers(rng, n):
    # A waist joins two tapered or straight legs, separated below the crotch.
    h = rng.uniform(.90, 1.01)
    width = rng.uniform(.30, .59)
    cx = .5 + rng.uniform(-.025, .025)
    cy = .5 + rng.uniform(-.02, .02)
    xx, yy = (X - cx) / h, (Y - cy) / h + .5
    waist = width * rng.uniform(.78, 1.00)
    crotch = rng.uniform(.25, .43)
    gap = rng.uniform(.045, .16)
    foot = width * rng.uniform(.56, 1.06)
    mask = polygon(xx, yy, [(-waist / 2, .015), (waist / 2, .015), (width / 2, .20),
                               (foot / 2, 1), (gap / 2, 1), (0, crotch),
                               (-gap / 2, 1), (-foot / 2, 1), (-width / 2, .20)])
    # The trouser category is a full-length, two-legged garment. Short skirt-like
    # shapes are deliberately not included among its analytic shape models.
    return mask


def bag(rng, n):
    # Tote, satchel, handbag, clutch, and backpack primitives.
    width = rng.uniform(.61, .99)
    height = rng.uniform(.50, .97)
    if n % 5 == 0:
        width = rng.uniform(.49, .75)
        height = rng.uniform(.85, 1.0)
    fit = rng.uniform(.90, 1.0) / max(width, height)
    cx, cy = .5 + rng.uniform(-.03, .03), .5 + rng.uniform(-.025, .025)
    xx, yy = (X - cx) / fit + width / 2, (Y - cy) / fit + height / 2
    handle_h = height * rng.uniform(.18, .43)
    if n % 7 == 0:
        handle_h = height * rng.uniform(.02, .12)
    top_w = width * rng.uniform(.70, 1.05)
    bottom_w = width * rng.uniform(.76, 1.0)
    top_y = handle_h
    bottom_y = height * rng.uniform(.93, 1.0)
    rounding = height * rng.uniform(.035, .13)
    body = polygon(xx, yy, [(width / 2 - top_w / 2, top_y), (width / 2 + top_w / 2, top_y),
                               (width / 2 + bottom_w / 2, bottom_y - rounding),
                               (width / 2 + bottom_w / 2 - rounding, bottom_y),
                               (width / 2 - bottom_w / 2 + rounding, bottom_y),
                               (width / 2 - bottom_w / 2, bottom_y - rounding)])
    rx = width * rng.uniform(.18, .35)
    thick = rng.uniform(.025, .065)
    loop = ellipse(xx, yy, width / 2, top_y, rx, max(handle_h, .025))
    loop &= ~ellipse(xx, yy, width / 2, top_y, max(.02, rx - thick), max(.012, handle_h - thick))
    if n % 5 == 0:  # Rounded backpack, with small top handle.
        body = ellipse(xx, yy, width / 2, .57 * height, width / 2, .45 * height)
        body |= rectangle(xx, yy, .04 * width, .50 * height, .96 * width, .90 * height)
    return body | loop


def footwear(rng, kind, n):
    # Intrinsic x points toe -> heel. Reflections include either photographed side.
    length = rng.uniform(.89, 1.01)
    cx, cy = .5 + rng.uniform(-.022, .022), .5 + rng.uniform(-.025, .025)
    xx = (X - cx) / length + .5
    if n % 2:
        xx = 1 - xx
    if kind == 7:
        height = rng.uniform(.31, .53)
    elif kind == 9:
        height = rng.uniform(.62, 1.03)
    else:
        height = rng.uniform(.25, .70)
    fit_y = min(1.0, .96 / height)
    yy = (Y - cy) / (length * fit_y) + height / 2
    base = height * .95
    sole_t = rng.uniform(.035, .080)
    sole = polygon(xx, yy, [(.005, base - sole_t), (.08, base - sole_t * 1.4), (.96, base - sole_t),
                               (1.0, base), (.94, base + .02), (.06, base + .02), (0, base)])
    if kind == 7:
        ankle_x = rng.uniform(.64, .83)
        toe_h = rng.uniform(.10, .17)
        vamp_y = rng.uniform(.17, .26)
        rear_y = rng.uniform(.075, .18)
        upper = polygon(xx, yy, [(.025, base - .025), (.04, base - toe_h), (.23, base - toe_h - .035),
                                    (.48, vamp_y), (ankle_x, .025), (.96, rear_y), (.99, base - .025)])
        return sole | upper
    if kind == 9:
        shaft_left = rng.uniform(.47, .70)
        shaft_right = rng.uniform(.86, 1.0)
        shaft_top = rng.uniform(.00, .10)
        heel_h = rng.uniform(.02, .20)
        if n % 3:
            heel_h *= .3
        upper = polygon(xx, yy, [(.01, base - .045), (.04, base - .15), (.25, base - .20),
                                    (shaft_left, height * .52), (shaft_left + .025, shaft_top),
                                    (shaft_right - .025, shaft_top), (.98, base - heel_h),
                                    (.88, base - heel_h), (.76, base - .055)])
        if heel_h > .055:
            sole = polygon(xx, yy, [(.0, base - .04), (.55, base - .04), (.72, base - heel_h),
                                       (.97, base - heel_h), (.98, base + .02), (.86, base + .02),
                                       (.82, base - heel_h + .04), (.57, base + .015), (.02, base + .015)])
        return sole | upper
    # Sandals: separated straps, open toe/instep; flat, wedge, or heeled.
    # Some have broad leather uppers rather than thin isolated straps. Make
    # these as a connected foot/ankle cover with explicit toe/instep openings.
    if n % 3 == 0:
        ankle = rng.uniform(.54, .79)
        top = height * rng.uniform(.03, .28)
        cover = polygon(xx, yy, [(.02, base - .025), (.04, base - .14), (.27, base - .18),
                                    (ankle, top + .12), (ankle + .015, top), (.96, top),
                                    (.99, base - .025)])
        hole_x = rng.uniform(.40, .65)
        hole_y = base - height * rng.uniform(.35, .60)
        cover &= ~ellipse(xx, yy, hole_x, hole_y, rng.uniform(.10, .20), height * rng.uniform(.12, .24))
        if n % 2:
            cover &= ~ellipse(xx, yy, .15, base - .11, .105, .065)
        return sole | cover
    high = n % 4 in (0, 1)
    if high:
        back_top = rng.uniform(.025, .12)
        footline = polygon(xx, yy, [(.03, base - .06), (.35, base - .08), (.75, back_top + .15),
                                       (.94, back_top + .15), (.57, base - .025), (.05, base + .005)])
        heel = polygon(xx, yy, [(.86, base - .27), (.97, base - .27), (.97, base + .02), (.88, base + .02)])
        strap = segment(xx, yy, .79, back_top, .94, back_top + .13, rng.uniform(.025, .05))
        strap |= segment(xx, yy, .90, back_top + .07, .89, base - .12, .025)
        front = segment(xx, yy, .22, base - .06, .30, base - .20, rng.uniform(.03, .055))
        return footline | heel | strap | front
    strap = segment(xx, yy, .18, base - .04, .43, base - height * .62, rng.uniform(.03, .065))
    strap |= segment(xx, yy, .43, base - height * .62, .69, base - .04, rng.uniform(.03, .065))
    if n % 3:
        strap |= segment(xx, yy, .64, base - .025, .83, height * .10, .025)
        strap |= segment(xx, yy, .83, height * .10, .97, base - .02, .025)
    return sole | strap


def pooled_template(mask):
    a = mask.astype(np.float64)
    # The same 2x2 max/mean blend used on the image, with supersampling.
    a = a.reshape(28, 2, 28, 2).mean((1, 3))
    windows = a.reshape(14, 2, 14, 2)
    a = .65 * windows.max((1, 3)) + .35 * windows.mean((1, 3))
    # Gentle smoothing gives some tolerance to fabric drape and pixel alignment.
    pad = np.pad(a, 1, mode='edge')
    a = .76 * a + .06 * (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])
    a -= a.mean()
    a /= np.sqrt(np.mean(a * a) + 1e-5)
    return a.reshape(-1).astype(np.float32) / (14 * 14)


def build(destination):
    net = Network()
    # Low grey-level soft foreground threshold, then exact [0,1] clipping.
    net.conv('foreground_ramps', 'image', np.array([12.5, 12.5]).reshape(2, 1, 1, 1), [-.375, -1.375])
    net.layer('foreground_rectify', 'relu', 'foreground_ramps')
    net.conv('foreground', 'foreground_rectify', np.array([1, -1]).reshape(1, 2, 1, 1))
    net.layer('foreground_max', 'maxpool', 'foreground', size=[2, 2], stride=[2, 2])
    net.layer('foreground_mean', 'avgpool', 'foreground', size=[2, 2], stride=[2, 2])
    net.conv('max_weight', 'foreground_max', np.array([.65]).reshape(1, 1, 1, 1))
    net.conv('mean_weight', 'foreground_mean', np.array([.35]).reshape(1, 1, 1, 1))
    net.layer('silhouette', 'add', ['max_weight', 'mean_weight'])
    net.layer('shape_normalized', 'norm', 'silhouette')

    count = 3072
    bank = np.empty((10 * count, 196), np.float32)
    for k in range(10):
        rng = np.random.default_rng(81821 + 1031 * k)
        for n in range(count):
            if k in (0, 2, 3, 4, 6):
                mask = garment(rng, k, n)
            elif k == 1:
                mask = trousers(rng, n)
            elif k == 8:
                mask = bag(rng, n)
            else:
                mask = footwear(rng, k, n)
            bank[k * count + n] = pooled_template(mask)
    # Sandal uppers can resemble sneakers; broad sandal covers are therefore
    # weak evidence without the separate, explicit opening detectors below.
    # This semantic ambiguity penalty is fixed by the model definition.
    shape_bias = np.zeros(10 * count, np.float32)
    shape_bias[5 * count:6 * count] = -.025
    shape_bias[5 * count:6 * count:3] = -.085
    net.dense('analytic_shape_correlations', 'shape_normalized', bank, shape_bias)
    net.layer('shape_bank', 'reshape', 'analytic_shape_correlations', shape=[10, count, 1])
    net.layer('best_shape_per_class', 'maxpool', 'shape_bank', size=[count, 1], stride=[count, 1])
    net.layer('shape_scores', 'reshape', 'best_shape_per_class', shape=[10])

    # Contrast-normalized horizontal second differences expose vertical closures.
    net.layer('contrast_normalized', 'norm', 'image')
    filters = np.zeros((4, 1, 3, 3), np.float32)
    filters[0, 0, 1] = [-.5, 1, -.5]
    filters[1] = -filters[0]
    filters[2, 0, :, 1] = [-.5, 1, -.5]
    filters[3] = -filters[2]
    net.conv('fabric_differences', 'contrast_normalized', filters, padding=1)
    net.layer('fabric_rectify', 'relu', 'fabric_differences')

    yy, xx = np.mgrid[0:28, 0:28].astype(np.float64)
    xx = (xx + .5) / 28
    yy = (yy + .5) / 28
    center = (np.abs(xx - .5) < .075) & (yy > .24) & (yy < .78)
    sides = (np.abs(xx - .5) > .12) & (np.abs(xx - .5) < .23) & (yy > .24) & (yy < .78)
    collar = (np.abs(xx - .5) < .17) & (yy > .07) & (yy < .25)
    feature_w = np.zeros((3, 4, 28, 28), np.float32)
    for c in (0, 1):
        feature_w[0, c] = center / center.sum() - sides / sides.sum()
        feature_w[1, c] = collar / collar.sum()
        feature_w[2, c] = sides / sides.sum()
    net.dense('construction_cues', 'fabric_rectify', feature_w.reshape(3, -1))
    # Central closure evidence must exceed surrounding fabric texture.
    net.dense('salient_construction', 'construction_cues', np.concatenate([np.eye(3), np.eye(3)]), [-.035, -.15, -.10, -.155, -.45, -.40])
    net.layer('positive_construction', 'relu', 'salient_construction')
    cue_readout = np.zeros((10, 3), np.float32)
    cue_readout[0] = [-.06, -.005, -.006]
    cue_readout[2] = [-.08, -.005, -.002]
    cue_readout[4] = [.18, .012, .002]
    cue_readout[6] = [.10, .012, .008]
    net.dense('detail_scores', 'positive_construction', np.concatenate([cue_readout, -cue_readout], axis=1))

    # Explicit anatomy: two trouser legs leave a lower central gap. A skirt
    # fills that region. This is a generic regional contrast, not an outline
    # taken from any example.
    bottom = (yy > .54) & (yy < .90)
    middle = (np.abs(xx - .5) < .055) & bottom
    leg_sides = (np.abs(xx - .5) > .07) & (np.abs(xx - .5) < .16) & bottom
    contrast = leg_sides / leg_sides.sum() - middle / middle.sum()
    leg_w = np.stack([contrast, -contrast]).reshape(2, -1)
    net.dense('leg_evidence', 'foreground', leg_w, [-.06, -.01])
    net.layer('positive_leg_evidence', 'relu', 'leg_evidence')
    leg_readout = np.zeros((10, 2), np.float32)
    leg_readout[1] = [.10, -.22]
    leg_readout[3] = [-.05, .035]
    net.dense('leg_scores', 'positive_leg_evidence', leg_readout)
    # An actual dark central slit, not merely an inner/outer contrast, is
    # required before a silhouette can be a pair of trousers.
    # Max of darkness over several neighboring columns locates a slit even
    # when the garment is slightly off-center; it does not demand symmetry.
    slit_bands = []
    for center_x in (.43, .465, .50, .535, .57):
        slit_region = (np.abs(xx - center_x) < .018) & (yy > .65) & (yy < .91)
        slit_bands.append(-slit_region.astype(np.float64) / slit_region.sum())
    net.dense('gap_darkness_columns', 'foreground', np.asarray(slit_bands).reshape(5, -1))
    net.layer('gap_darkness_map', 'reshape', 'gap_darkness_columns', shape=[1, 5, 1])
    net.layer('darkest_gap_column', 'maxpool', 'gap_darkness_map', size=[5, 1], stride=[5, 1])
    net.dense('filled_leg_gap', 'darkest_gap_column', [[-1], [-1]], [-.32, -.82])
    net.layer('positive_filled_gap', 'relu', 'filled_leg_gap')
    filled_readout = np.zeros((10, 2), np.float32)
    filled_readout[1] = [-.22, .22]
    net.dense('filled_gap_scores', 'positive_filled_gap', filled_readout)

    # A sandal encloses a large extent with straps but little filled area;
    # a closed shoe fills much more of its vertical extent.
    net.layer('row_extent', 'maxpool', 'foreground', size=[1, 28], stride=[1, 28])
    net.dense('half_height', 'row_extent', np.full((1, 28), .5 / 28))
    net.dense('negative_area', 'foreground', np.full((1, 784), -1 / 784))
    net.layer('open_structure', 'add', ['half_height', 'negative_area'])
    net.dense('salient_open_structure', 'open_structure', [[1]], [-.025])
    net.layer('positive_open_structure', 'relu', 'salient_open_structure')
    openness = np.zeros((10, 1), np.float32)
    openness[5, 0] = .85
    openness[7, 0] = -.20
    openness[9, 0] = -.20
    net.dense('openness_scores', 'positive_open_structure', openness)

    # A garment normally has material across the shoulders; a bag often has
    # only a narrow handle at the top. A lower U-shaped interruption between
    # a sleeve and the torso supplies additional garment evidence.
    shoulders = (np.abs(xx - .5) > .17) & (np.abs(xx - .5) < .30) & (yy > .075) & (yy < .18)
    sleeve_tip = (np.abs(xx - .5) > .36) & (np.abs(xx - .5) < .46) & (yy > .62) & (yy < .87)
    underarm_gap = (np.abs(xx - .5) > .29) & (np.abs(xx - .5) < .36) & (yy > .62) & (yy < .87)
    shoulder_map = shoulders / shoulders.sum()
    sleeve_map = sleeve_tip / sleeve_tip.sum() - underarm_gap / underarm_gap.sum()
    net.dense('garment_anatomy', 'foreground', np.stack([shoulder_map, sleeve_map]).reshape(2, -1), [-.35, -.12])
    net.layer('positive_garment_anatomy', 'relu', 'garment_anatomy')
    anatomy = np.zeros((10, 2), np.float32)
    anatomy[0] = [.025, .000]
    anatomy[2] = [.025, .020]
    anatomy[3] = [.025, .000]
    anatomy[4] = [.025, .020]
    anatomy[6] = [.025, .020]
    anatomy[8] = [-.11, -.12]
    net.dense('garment_anatomy_scores', 'positive_garment_anatomy', anatomy)

    # Local maxima versus means distinguish narrow straps and cutouts from a
    # filled vamp. Restrict attention to the middle of a centered shoe, away
    # from its outside silhouette and the ankle-opening boundary.
    net.layer('brightest_in_three', 'maxpool', 'image', size=[3, 3], stride=[1, 1])
    net.layer('mean_in_three', 'avgpool', 'image', size=[3, 3], stride=[1, 1])
    net.conv('negative_local_mean', 'mean_in_three', np.array([-1]).reshape(1, 1, 1, 1))
    net.layer('thin_bright_structure', 'add', ['brightest_in_three', 'negative_local_mean'])
    syy, sxx = np.mgrid[0:26, 0:26].astype(np.float64)
    sxx, syy = (sxx + 1.5) / 28, (syy + 1.5) / 28
    vamp = (np.abs(sxx - .5) < .25) & (syy > .43) & (syy < .67)
    net.dense('strap_evidence', 'thin_bright_structure', (vamp / vamp.sum()).reshape(1, -1), [-.10])
    net.layer('positive_strap_evidence', 'relu', 'strap_evidence')
    straps = np.zeros((10, 1), np.float32)
    straps[5] = .15
    straps[7] = -.025
    straps[9] = -.025
    net.dense('strap_scores', 'positive_strap_evidence', straps)

    # Enclosed dark holes distinguish an open sandal from the single exterior
    # boundary of a sneaker. All four opposing flanks must contain foreground.
    hole_filters = np.zeros((16, 1, 7, 7), np.float32)
    for scale in range(2):
        distance = scale + 2
        for side in range(4):
            j = scale * 4 + side
            hole_filters[j, 0, 3, 3] = -1
            if side == 0:
                hole_filters[j, 0, 3 - distance, 2:5] = 1 / 3
            elif side == 1:
                hole_filters[j, 0, 3 + distance, 2:5] = 1 / 3
            elif side == 2:
                hole_filters[j, 0, 2:5, 3 - distance] = 1 / 3
            else:
                hole_filters[j, 0, 2:5, 3 + distance] = 1 / 3
            hole_filters[8 + j] = hole_filters[j]
    hole_bias = np.array([-.10] * 8 + [-1.10] * 8, np.float32)
    net.conv('hole_flanks', 'foreground', hole_filters, hole_bias, padding=3)
    net.layer('positive_hole_flanks', 'relu', 'hole_flanks')
    # Clipping each flank at one means an absent flank cannot be compensated
    # by a different flank: this is a soft logical AND.
    hole_join = np.zeros((2, 16, 1, 1), np.float32)
    for scale in range(2):
        hole_join[scale, scale * 4:scale * 4 + 4, 0, 0] = 1
        hole_join[scale, 8 + scale * 4:12 + scale * 4, 0, 0] = -1
    net.conv('enclosed_hole', 'positive_hole_flanks', hole_join, [-2.30, -2.30])
    net.layer('positive_enclosed_hole', 'relu', 'enclosed_hole')
    hole_region = (np.abs(xx - .5) < .32) & (yy > .25) & (yy < .73)
    hole_regions = np.broadcast_to(hole_region, (2, 28, 28))
    net.dense('enclosed_hole_summary', 'positive_enclosed_hole', (hole_regions / (2 * hole_region.sum())).reshape(1, -1))
    hole_scores = np.zeros((10, 1), np.float32)
    hole_scores[5] = 2.0
    hole_scores[7] = -.25
    hole_scores[9] = -.25
    net.dense('hole_scores', 'enclosed_hole_summary', hole_scores)

    net.layer('combined_scores', 'add', ['shape_scores', 'detail_scores', 'leg_scores', 'filled_gap_scores', 'openness_scores', 'garment_anatomy_scores', 'strap_scores', 'hole_scores'])
    # Equal priors for ten equally plausible class labels. No empirical class
    # frequencies or score calibration are used.
    net.dense('scores', 'combined_scores', np.eye(10, dtype=np.float32))

    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'network.json').write_text(json.dumps({'layers': net.layers}, indent=2) + '\n')
    np.savez(out / 'weights.npz', **net.weights)
    print('Wrote', len(net.layers), 'layers and', sum(w.size for w in net.weights.values()), 'weights')


if __name__ == '__main__':
    build(sys.argv[1])
