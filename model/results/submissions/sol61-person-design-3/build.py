"""Data-free, analytic colour/shape network. Usage: python3 build.py OUTPUT.

No training, stored examples, pretrained parameters, or input-file reads.
All geometries below are generic schematic heads and shoulders, in units of
head width/height. Rasterization only integrates these analytic geometries.
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

    def op(self, name, op, inputs, **settings):
        if isinstance(inputs, str):
            inputs = [inputs]
        self.layers.append(dict(name=name, op=op, inputs=inputs, **settings))
        return name

    def conv(self, name, source, w, b=None, stride=1, padding=0):
        w = np.asarray(w, dtype=np.float32)
        if b is None:
            b = np.zeros(w.shape[0], np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.asarray(b, np.float32)
        return self.op(name, 'conv', source, out=w.shape[0],
                       kernel=list(w.shape[2:]), stride=stride, padding=padding)

    def dense(self, name, source, w, b):
        w = np.asarray(w, np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.asarray(b, np.float32)
        return self.op(name, 'dense', source, out=w.shape[0])

    def point(self, name, source, w, b=None):
        return self.conv(name, source, np.asarray(w)[:, :, None, None], b)

    def relu(self, name, source):
        return self.op(name, 'relu', source)

    def pool(self, name, source, size, stride, kind='avgpool'):
        return self.op(name, kind, source, size=list(size), stride=list(stride))


def normal_blob(k, cx, cy, sx, sy):
    """Integrate an analytic elliptical Gaussian over each unit pixel."""
    a = np.arange(k, dtype=np.float64) - (k - 1) / 2
    yy, xx = np.meshgrid(a, a, indexing='ij')
    z = np.zeros((k, k), dtype=np.float64)
    for dy in (-.375, -.125, .125, .375):
        for dx in (-.375, -.125, .125, .375):
            z += np.exp(-.5 * (((xx + dx - cx) / sx) ** 2 +
                                ((yy + dy - cy) / sy) ** 2))
    return z / z.sum()


def average(*masks):
    return sum(masks) / len(masks)


def contour(k, samples, face_y):
    """Soft, unoriented edge-normal histogram along a schematic contour.

    samples consists of (x,y,normal_x,normal_y,arc_length) tuples. The
    positive term rewards the matching normal; other normals and general
    patch texture are penalized. No image ever enters this function.
    """
    directions = np.arange(4, dtype=np.float64) * (math.pi / 4)
    pos = np.zeros((4, k, k), dtype=np.float64)
    for x, y, nx, ny, length in samples:
        spatial = normal_blob(k, x, y + face_y, .53, .53)
        norm = math.hypot(nx, ny)
        q = np.abs((nx * np.cos(directions) + ny * np.sin(directions)) / norm) ** 8
        q /= q.sum()
        pos += q[:, None, None] * spatial[None] * length
    pos /= pos.sum()
    support = pos.sum(axis=0)
    other = (support[None] - pos) / 3
    w = pos - .30 * other
    # Discourage arbitrary high-texture regions from being called outlines.
    xs = [s[0] for s in samples]
    ys = [s[1] + face_y for s in samples]
    texture = normal_blob(k, (min(xs) + max(xs)) / 2,
                          (min(ys) + max(ys)) / 2,
                          max(.7, (max(xs) - min(xs)) / 2),
                          max(.7, (max(ys) - min(ys)) / 2))
    w -= .18 * texture[None] / 4
    return w


def line_segment(x0, y0, x1, y1, count=28):
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    return [(x0 + (i + .5) * dx / count,
             y0 + (i + .5) * dy / count,
             -dy, dx, length / count) for i in range(count)]


def shape_parts(k, width, height, face_y):
    head = []
    a, b = .59 * width, .59 * height
    for t in (np.arange(96) + .5) * (2 * math.pi / 96):
        ct, st = math.cos(t), math.sin(t)
        head.append((a * ct, -.08 * height + b * st,
                     ct / a, st / b,
                     math.hypot(a * st, b * ct) * 2 * math.pi / 96))
    shoulders = []
    torso = []
    for sign in (-1, 1):
        # Neck to shoulder, and upper arm. Arms may hang or be partly cropped.
        shoulders += line_segment(sign * .35 * width, .51 * height,
                                  sign * 1.02 * width, .78 * height)
        shoulders += line_segment(sign * 1.02 * width, .78 * height,
                                  sign * 1.24 * width, 1.37 * height)
        # Torso narrows gently below the shoulder.
        torso += line_segment(sign * .91 * width, 1.02 * height,
                              sign * .76 * width, 1.72 * height)
    return [contour(k, p, face_y) for p in (head, shoulders, torso)]


def make_colour_maps(net):
    # Hinge penalties describe a broad skin-colour polytope in linear RGB.
    # A channel here is max(a.R+b.G+c.B+d,0), not a learned colour centroid.
    inequalities = [
        ((-1, 1, 0), .028, 22.0),       # not enough red relative to green
        ((-1, 0, 1), .046, 16.0),       # not enough red relative to blue
        ((0, -1, 1), -.014, 10.0),      # excessive blue relative to green
        ((1, -1.65, 0), -.025, 9.0),    # very red/orange rather than skin
        ((.38, -1, 0), -.012, 10.0),    # overly saturated red
        ((0, .40, -1), -.022, 10.0),    # yellow/orange with too little blue
        ((1, 0, -1), -.56, 6.0),        # extreme red-blue difference
        ((-1.15, 2.15, -1), -.035, 8.0), # yellow/brown hue versus red-pink skin
        ((-1, 0, 0), .13, 9.0),        # near-black is ambiguous, not skin evidence
        ((0, -1, 0), .060, 7.0),
    ]
    w = np.zeros((len(inequalities) + 1, 3), np.float64)
    b = np.zeros(len(inequalities) + 1, np.float64)
    for i, (a, d, scale) in enumerate(inequalities):
        w[i] = np.asarray(a) * scale
        b[i] = d * scale
    w[-1] = [.299, .587, .114]
    h = net.point('colour_halfspaces', 'image', w, b)
    h = net.relu('colour_violations', h)
    # Broad and conservative skin maps, luminance, and a valid-pixel mask.
    w = np.zeros((4, len(inequalities) + 1), np.float64)
    w[0, :-1] = -1
    w[1, :-1] = -1.5
    w[2, -1] = 1
    h = net.point('colour_evidence', h, w, [1, .88, 0, 1])
    h = net.relu('colour_evidence_positive', h)
    return net.pool('base_48', h, (2, 2), (2, 2))


def make_features(net, base, prefix):
    # Keep skin, conservative skin, normalized luminance, valid mask, and
    # normalized skin rarity as five colour/contrast channels.
    w = np.zeros((1, 4)); w[0, 2] = 1
    gray = net.point(prefix + '_gray', base, w)
    gray = net.op(prefix + '_gray_contrast', 'norm', gray)
    w = np.zeros((1, 4)); w[0, 0] = 1
    skin = net.point(prefix + '_skin_extract', base, w)
    skin = net.op(prefix + '_skin_rarity', 'norm', skin)
    w = np.zeros((5, 4))
    w[0, 0] = 1; w[1, 1] = 1; w[3, 3] = 1
    c0 = net.point(prefix + '_colour_embed', base, w)
    w = np.zeros((5, 1)); w[2, 0] = 1
    c1 = net.point(prefix + '_gray_embed', gray, w)
    w = np.zeros((5, 1)); w[4, 0] = 1
    c2 = net.point(prefix + '_rarity_embed', skin, w)
    colours = net.op(prefix + '_colour_features', 'add', [c0, c1, c2])

    # Sobel derivatives, followed by absolute values and selective orientation
    # energies. Four unsigned orientations suffice for heads and shoulders.
    gx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64) / 8
    gy = gx.T
    filters = [gx, (gx + gy) / math.sqrt(2), gy, (gx - gy) / math.sqrt(2)]
    w = np.zeros((8, 1, 3, 3))
    for i, f in enumerate(filters):
        w[2*i, 0] = f
        w[2*i+1, 0] = -f
    g = net.conv(prefix + '_derivatives', gray, w, padding=1)
    g = net.relu(prefix + '_signed_edges', g)
    w = np.zeros((8, 8))
    for i in range(4):
        perpendicular = (i + 2) % 4
        for row in (i, i + 4):
            w[row, 2*i:2*i+2] = 1
            w[row, 2*perpendicular:2*perpendicular+2] = -.45
    g = net.point(prefix + '_edge_selectivity', g, w, [0]*4 + [-1.2]*4)
    g = net.relu(prefix + '_edge_hinges', g)
    w = np.zeros((4, 8))
    for i in range(4):
        w[i, i] = 1; w[i, i+4] = -1
    edges = net.point(prefix + '_bounded_edges', g, w)
    return colours, edges


def directed_contour(k, samples, face_y):
    """Inside-minus-outside contrast across a generic contour.

    Unlike unsigned HOG, unrelated texture on opposite sides cannot reinforce
    this cue. Each normal has the same inside/outside sign.
    """
    out = np.zeros((k, k), np.float64)
    total = sum(s[4] for s in samples)
    for x, y, nx, ny, length in samples:
        d = math.hypot(nx, ny)
        nx, ny = nx / d, ny / d
        inside = normal_blob(k, x - .6 * nx, y + face_y - .6 * ny, .48, .48)
        outside = normal_blob(k, x + .6 * nx, y + face_y + .6 * ny, .48, .48)
        out += (inside - outside) * length / total
    return out


def signed_parts(k, width, height, face_y):
    head_left, head_right = [], []
    a, b = .59 * width, .59 * height
    for t in (np.arange(96) + .5) * (2 * math.pi / 96):
        ct, st = math.cos(t), math.sin(t)
        sample = (a * ct, -.08 * height + b * st, ct / a, st / b,
                  math.hypot(a * st, b * ct) * 2 * math.pi / 96)
        (head_left if ct < 0 else head_right).append(sample)
    body = []
    for sign in (-1, 1):
        points = [(sign * .34 * width, .50 * height),
                  (sign * 1.02 * width, .78 * height),
                  (sign * 1.24 * width, 1.37 * height),
                  (sign * .96 * width, 1.58 * height)]
        samples = []
        for p0, p1 in zip(points[:-1], points[1:]):
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            # The outward normal points left on the left silhouette.
            length = math.hypot(dx, dy)
            for i in range(24):
                t = (i + .5) / 24
                samples.append((p0[0] + t * dx, p0[1] + t * dy,
                                sign * dy, -sign * dx, length / 24))
        body.append(samples)
    return [directed_contour(k, p, face_y)
            for p in (head_left, head_right, body[0], body[1])]


def detector(net, colours, edges, prefix, widths, resolution):
    k = 13
    n = len(widths)
    # Per schematic head:
    # 0: skin compactness +1; 1,2: eye contrast and its cap;
    # 3,4: mouth contrast and cap; 5,6: left/right asymmetry;
    # 7: insufficient skin; 8,9: rare-colour evidence and cap;
    # 10,11: head contour and cap; 12,13: shoulder contour and cap;
    # 14: torso contour; 15: missing-image penalty.
    channels = 64
    cw = np.zeros((n * channels, 5, k, k), np.float64)
    cb = np.zeros(n * channels, np.float64)
    ew = np.zeros((n * channels, 4, k, k), np.float64)
    for j, width in enumerate(widths):
        height = width * 1.27
        fy = -2.0
        def blob(x, y, sx, sy):
            return normal_blob(k, x * width, y * height + fy,
                               sx * width, sy * height)
        # Generic facial regions: forehead, nose and two cheeks; negative
        # bands beside the head, above it, and in the shirt/torso region.
        forehead = blob(0, -.29, .25, .115)
        nose = blob(0, .025, .12, .21)
        left_cheek = blob(-.23, .15, .16, .15)
        right_cheek = blob(.23, .15, .16, .15)
        face = .22 * forehead + .18 * nose + .30 * left_cheek + .30 * right_cheek
        sides = average(blob(-.72, 0, .17, .34), blob(.72, 0, .17, .34))
        top = blob(0, -.72, .40, .14)
        shirt = blob(0, 1.20, .72, .31)
        core = normal_blob(k, 0, fy, .31 * width, .34 * height)
        left_eye = blob(-.21, -.12, .105, .065)
        right_eye = blob(.21, -.12, .105, .065)
        eyes = average(left_eye, right_eye)
        eye_lights = .4 * forehead + .3 * left_cheek + .3 * right_cheek
        mouth = blob(0, .32, .18, .065)
        mouth_lights = average(left_cheek, right_cheek)
        left_half = blob(-.22, .02, .15, .30)
        right_half = blob(.22, .02, .15, .30)
        o = j * channels
        # Mix the two hand-specified colour confidence levels.
        compact = face - .48 * sides - .12 * top - .24 * shirt
        cw[o, 0] = .72 * compact
        cw[o, 1] = .28 * compact
        # Penalize a putative head cut beyond the image, rather than treating
        # zero padding as a dark surround.
        cw[o, 3] = 1.5 * core
        cb[o] = -.5   # = +1 offset -1.5 missing-coverage penalty
        for r in (1, 2):
            cw[o+r, 2] = eye_lights - eyes
        cb[o+2] = -.8
        for r in (3, 4):
            cw[o+r, 2] = mouth_lights - mouth
        cb[o+4] = -.5
        cw[o+5, 2] = left_half - right_half
        cw[o+6, 2] = right_half - left_half
        cb[o+5] = cb[o+6] = -.45
        cw[o+7, 0] = -core
        cb[o+7] = .34
        cw[o+8, 4] = core
        cw[o+9, 4] = core
        cb[o+9] = -3
        parts = shape_parts(k, width, height, fy)
        for r, part in zip((10, 12, 14), parts):
            ew[o+r] = part
        ew[o+11] = parts[0]
        ew[o+13] = parts[1]
        cb[o+11] = cb[o+13] = -.8
        cw[o+15, 3] = -core
        cb[o+15] = .94
        for r, part in enumerate(signed_parts(k, width, height, fy)):
            cw[o+16+2*r, 2] = part
            cw[o+17+2*r, 2] = -part
        # Separate boundaries prevent one strongly contrasting side of an
        # orange object from compensating for its other missing boundaries.
        side_left = blob(-.72, 0, .17, .34)
        side_right = blob(.72, 0, .17, .34)
        for r, outside in enumerate((side_left, side_right, top, shirt)):
            for channel, fraction in ((0, .72), (1, .28)):
                cw[o+24+r, channel] = fraction * (outside - face)
            cb[o+24+r] = .08 if r == 2 else .15
        # Both cheeks must independently support skin, not just their mean.
        cw[o+28, 0] = -left_cheek; cb[o+28] = .44
        cw[o+29, 0] = -right_cheek; cb[o+29] = .44
        # For the frontal route, both eyes must be darker than nearby cheek.
        cw[o+30, 2] = left_eye - left_cheek; cb[o+30] = .01
        cw[o+31, 2] = right_eye - right_cheek; cb[o+31] = .01
        # Distinguish a skin head above clothing from a uniformly warm object.
        cw[o+32, 0] = shirt - face; cb[o+32] = .20
        # Missing head support and shape requirements are one-sided gates.
        cw[o+33, 0] = -core; cb[o+33] = .42
        ew[o+34] = -parts[0]; cb[o+34] = .10
        ew[o+35] = -parts[1]; cb[o+35] = .055
        # Do not reward a rare single warm pixel; require a coherent patch.
        cw[o+36, 0] = face; cb[o+36] = -.55
        cw[o+37, 0] = face; cb[o+37] = -.90
        # A meaningful neck-to-shoulder width change is needed for rear views.
        for channel, fraction in ((0, .72), (1, .28)):
            cw[o+38, channel] = fraction * (sides - face)
        cb[o+38] = .20
        cw[o+39, 3] = -core; cb[o+39] = .98
        # A separate achromatic route uses a richer frontal-face stencil:
        # two dark eyes, a dark mouth, light cheeks/nose, and bilateral balance.
        left_eye_light = average(blob(-.21, -.29, .16, .08), left_cheek)
        right_eye_light = average(blob(.21, -.29, .16, .08), right_cheek)
        cw[o+40, 2] = left_eye - left_eye_light; cb[o+40] = .035
        cw[o+41, 2] = right_eye - right_eye_light; cb[o+41] = .035
        cw[o+42, 2] = mouth - mouth_lights; cb[o+42] = .015
        cw[o+43, 2] = eyes - nose; cb[o+43] = .025
        cw[o+44, 2] = left_eye - right_eye; cb[o+44] = -.18
        cw[o+45, 2] = right_eye - left_eye; cb[o+45] = -.18
        cw[o+46, 2] = left_cheek - right_cheek; cb[o+46] = -.27
        cw[o+47, 2] = right_cheek - left_cheek; cb[o+47] = -.27
        # Light face enclosed by a darker scalp and lateral boundary.
        cw[o+48, 2] = top - forehead; cb[o+48] = .035
        cw[o+49, 2] = side_left - left_cheek; cb[o+49] = .025
        cw[o+50, 2] = side_right - right_cheek; cb[o+50] = .025
        cw[o+51, 2] = -parts[0].sum(axis=0) * 0
        # Eye contrast must not grow without bound on isolated black stripes.
        cw[o+52, 2] = eye_lights - eyes; cb[o+52] = -.65
        cw[o+53, 2] = mouth_lights - mouth; cb[o+53] = -.50
        cw[o+54, 2] = nose - eyes; cb[o+54] = -.7
        ew[o+55] = -parts[0]; cb[o+55] = .12
        # A generic shirt is approximately a coherent two-dimensional region:
        # its upper and lower halves should resemble each other, unlike a
        # spurious head on a horizontal table/food/wood edge.
        upper_shirt = blob(0, .96, .54, .17)
        lower_shirt = blob(0, 1.45, .54, .17)
        shirt_left = blob(-.44, 1.20, .24, .30)
        shirt_right = blob(.44, 1.20, .24, .30)
        for r, part in enumerate((upper_shirt - lower_shirt,
                                  lower_shirt - upper_shirt,
                                  shirt_left - shirt_right,
                                  shirt_right - shirt_left)):
            cw[o+56+r, 2] = part; cb[o+56+r] = -.38
        # Colour compactness of a head should be supported independently by
        # both cheeks; warm horizontal strips do not satisfy these contrasts.
        cw[o+60, 0] = side_left - left_cheek; cb[o+60] = .10
        cw[o+61, 0] = side_right - right_cheek; cb[o+61] = .10
        # Exact negative gates for otherwise optional shoulder/torso HOG.
        ew[o+62] = -parts[1]; cb[o+62] = .10
        ew[o+63] = -parts[2]; cb[o+63] = .065
    c = net.conv(prefix + '_face_stencils', colours, cw, cb, padding=k//2)
    e = net.conv(prefix + '_contour_stencils', edges, ew, padding=k//2)
    h = net.op(prefix + '_cue_sum', 'add', [c, e])
    h = net.relu(prefix + '_cue_hinges', h)

    # General skin head, landmark-supported face, and an independent achromatic
    # head/shoulder model. Their common score unit is a hand-set confidence
    # margin, not a fitted probability.
    w = np.zeros((n * 18, n * channels))
    b = np.zeros(n * 18)
    for j in range(n):
        i, o = j * channels, j * 18
        small = .025 if resolution == 48 and widths[j] < 3 else 0
        for row in (o, o+1):
            w[row, i] = 1
            w[row, i+8] = .020; w[row, i+9] = -.020
            w[row, i+28] = -.5; w[row, i+29] = -.5
            w[row, i+33] = -.6
            w[row, i+39] = -2
            w[row, i+56:i+60] = -.12
            w[row, i+60:i+62] = -.20
        # Generic rear/profile skin head: three-sided compactness and an
        # independently detected shoulder are all required.
        w[o, i+24:i+28] = -.65
        w[o, i+35] = -.7
        w[o, i+10] = .06; w[o, i+11] = -.06
        w[o, i+12] = .08; w[o, i+13] = -.08
        b[o] = -1.60 - small
        # Frontal route trades some colour contrast for bilateral landmarks.
        w[o+1, i+24:i+26] = -.5
        w[o+1, i+27] = -.4
        w[o+1, i+30] = -.4; w[o+1, i+31] = -.4
        w[o+1, i+1] = .28; w[o+1, i+2] = -.28
        w[o+1, i+3] = .12; w[o+1, i+4] = -.12
        w[o+1, i+5] = -.10; w[o+1, i+6] = -.10
        b[o+1] = -1.55 - small
        w[o+2, i+10] = 1
        w[o+3, i+12] = 1
        w[o+4, i+10] = 1; w[o+4, i+12] = -1
        w[o+5, i+12] = 1; w[o+5, i+14] = -1
        w[o+6, i+15] = 1
        for r, sign in ((7, 0), (11, 1)):
            w[o+r, i+16+sign] = 1
            w[o+r+1, i+16+sign] = 1
            w[o+r+1, i+18+sign] = -1
            w[o+r+2, i+20+sign] = 1
            w[o+r+3, i+20+sign] = 1
            w[o+r+3, i+22+sign] = -1
        # Colour evidence for a possible outline-supported detection.
        w[o+15, i] = 1
        w[o+15, i+24:i+28] = -.8
        w[o+15, i+28] = -.6; w[o+15, i+29] = -.6
        w[o+15, i+33] = -.7
        w[o+15, i+35] = -.7
        w[o+15, i+39] = -2
        w[o+15, i+56:i+60] = -.15
        w[o+15, i+60:i+62] = -.20
        b[o+15] = -1.48
        w[o+16, i+7] = 1
        # Strong, structured face evidence can identify grayscale faces. The
        # negative hinge gates require each landmark, not only their sum.
        row = o+17
        w[row, i+1] = .34; w[row, i+2] = -.34
        w[row, i+3] = .20; w[row, i+4] = -.20
        w[row, i+40:i+44] = -1.5
        w[row, i+44:i+48] = -.50
        w[row, i+48:i+51] = -.40
        w[row, i+52] = -.25
        w[row, i+53] = -.15
        w[row, i+54] = -.12
        w[row, i+55] = -.60
        w[row, i+62] = -.60
        w[row, i+63] = -.30
        w[row, i+56:i+60] = -.15
        w[row, i+15] = -2
        b[row] = -.13
    h = net.point(prefix + '_part_rules', h, w, b)
    h = net.relu(prefix + '_part_gates', h)
    w = np.zeros((n * 12, n * 18))
    b = np.zeros(n * 12)
    for j in range(n):
        i, o = j * 18, j * 12
        w[o, i] = 1
        w[o+1, i+1] = 1
        # HOG consistency now supports other cues rather than firing alone.
        w[o+2, i+2] = 1
        w[o+2, i+4] = -1
        w[o+2, i+3] = .30
        w[o+2, i+5] = -.30
        for r, src in ((3, 7), (5, 11)):
            w[o+r, i+src] = 1; w[o+r, i+src+1] = -1
            w[o+r+1, i+src+2] = 1; w[o+r+1, i+src+3] = -1
        w[o+7, i+15] = 1
        w[o+8, i+6] = 1
        # Exact conjunction: outline confidence cannot replace skin evidence.
        for row, src in ((o+9, 7), (o+10, 11)):
            w[row, i+src] = .16; w[row, i+src+1] = -.16
            w[row, i+src+2] = .12; w[row, i+src+3] = -.12
            w[row, i+15] = -1
        w[o+11, i+17] = 1
    h = net.point(prefix + '_bilateral_minima', h, w, b)
    h = net.relu(prefix + '_bilateral_gates', h)
    w = np.zeros((n * 5, n * 12))
    b = np.zeros(n * 5)
    for j in range(n):
        i, o = j * 12, j * 5
        w[o, i] = 1
        w[o+1, i+1] = 1
        for row, head, body, difference in ((o+2, 3, 4, 9), (o+3, 5, 6, 10)):
            # min(colour margin, bilateral contour margin).
            w[row, i+head] = .16
            w[row, i+body] = .12
            w[row, i+difference] = -1
            w[row, i+8] = -2
        w[o+4, i+11] = 1
    h = net.point(prefix + '_detector_margins', h, w, b)
    h = net.relu(prefix + '_positive_detections', h)
    # Pool over position, model, and head size without a learned readout.
    h = net.op(prefix + '_all_candidates', 'reshape', h,
               shape=[1, n * 5 * resolution, resolution])
    h = net.pool(prefix + '_strongest_detection', h,
                 (n * 5 * resolution, resolution), (1, 1), kind='maxpool')
    return h


def build(destination):
    net = Network()
    base = make_colour_maps(net)
    evidence = []
    for level, resolution in enumerate((48, 24, 12, 6)):
        prefix = 's' + str(resolution)
        if level:
            base = net.pool(prefix + '_pyramid', base, (2, 2), (2, 2))
        colours, edges = make_features(net, base, prefix)
        widths = (2.4, 3.5) if resolution > 6 else (2.4,)
        h = detector(net, colours, edges, prefix, widths, resolution)
        w = np.zeros((4, 1)); w[level, 0] = 1
        evidence.append(net.dense(prefix + '_place_evidence', h, w, np.zeros(4)))
    h = net.op('scale_evidence', 'add', evidence)
    h = net.op('scale_axis', 'reshape', h, shape=[1, 4, 1])
    h = net.pool('best_scale', h, (4, 1), (1, 1), kind='maxpool')
    net.dense('scores', h, [[0], [1]], [0, -.10])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'network.json').write_text(json.dumps({'layers': net.layers}, indent=2) + '\n')
    np.savez(destination / 'weights.npz', **net.weights)
    print('Wrote', len(net.layers), 'layers and',
          sum(a.size for a in net.weights.values()), 'analytic float32 weights.')


if __name__ == '__main__':
    build(sys.argv[1])
