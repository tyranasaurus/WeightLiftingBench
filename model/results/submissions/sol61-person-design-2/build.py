"""Build a data-free, analytically specified person-presence network.

No file is read by this program. All templates below are generated from colour
inequalities, Gaussians, and an idealized head/shoulders silhouette. Neither the
constants nor the filters are estimated from examples.
"""
import json
import sys
from pathlib import Path
import numpy as np


class Graph:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def op(self, name, op, inputs, **settings):
        if isinstance(inputs, str):
            inputs = [inputs]
        self.layers.append(dict(name=name, op=op, inputs=inputs, **settings))
        return name

    def conv(self, name, source, weight, bias=None, padding=0, stride=1):
        weight = np.asarray(weight, dtype=np.float32)
        if bias is None:
            bias = np.zeros(weight.shape[0], dtype=np.float32)
        self.weights[name + '.W'] = weight
        self.weights[name + '.b'] = np.asarray(bias, dtype=np.float32)
        return self.op(name, 'conv', source, out=weight.shape[0],
                       kernel=list(weight.shape[-2:]), stride=stride, padding=padding)

    def dense(self, name, source, weight, bias=None):
        weight = np.asarray(weight, dtype=np.float32)
        if bias is None:
            bias = np.zeros(weight.shape[0], dtype=np.float32)
        self.weights[name + '.W'] = weight
        self.weights[name + '.b'] = np.asarray(bias, dtype=np.float32)
        return self.op(name, 'dense', source, out=weight.shape[0])

    def relu(self, name, source):
        return self.op(name, 'relu', source)

    def pool(self, name, source, size, stride=None, kind='avgpool'):
        if isinstance(size, int):
            size = [size, size]
        if stride is None:
            stride = size
        if isinstance(stride, int):
            stride = [stride, stride]
        return self.op(name, kind, source, size=size, stride=stride)


def gaussian(x, y, cx, cy, sx, sy):
    z = np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    return z / z.sum()


def grid(k=9):
    a = np.arange(k, dtype=np.float64) - (k - 1) / 2
    return np.meshgrid(a, a)


def ellipse_ring(x, y, cx, cy, rx, ry, neck=True):
    d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    # Smoothly weighted annulus, excluding the neck opening at its bottom.
    z = np.exp(-((np.sqrt(d + 1e-12) - 1.45) / 0.30) ** 2)
    if neck:
        z *= (y < cy + 0.5 * ry) | (np.abs(x - cx) > 0.85 * rx)
    return z / z.sum()


def put_min_constraint(bank, biases, skin_kernel=None, gray_kernel=None,
                       threshold=0.0, penalty=1.0):
    """Produce penalty * relu(threshold - linear measurement)."""
    shape = skin_kernel.shape if skin_kernel is not None else gray_kernel.shape
    w = np.zeros((2,) + shape, dtype=np.float64)
    if skin_kernel is not None:
        w[0] = -penalty * skin_kernel
    if gray_kernel is not None:
        w[1] = -penalty * gray_kernel
    bank.append(w)
    biases.append(penalty * threshold)


def face_bank():
    xx, yy = grid()
    bank, biases, spans = [], [], []
    # Each candidate must explain a complete, co-located anatomical pattern.
    # Both cheeks, two distinct eye sockets, a nose bridge, mouth and chin
    # are tested separately; a generic dark band is not a sufficient face.
    for width in (3.2, 4.6):
        for angle in (-0.20, 0.0, 0.20):
            x = xx * np.cos(angle) + yy * np.sin(angle)
            y = -xx * np.sin(angle) + yy * np.cos(angle)
            start = len(bank)
            height = 1.32 * width
            center = gaussian(x, y, 0, 0.04 * height,
                              0.26 * width, 0.29 * height)
            ring = ellipse_ring(x, y, 0, -0.02 * height,
                                0.48 * width, 0.49 * height)
            forehead = gaussian(x, y, 0, -0.31 * height,
                                0.23 * width, max(0.28, 0.09 * height))
            cheeks = [gaussian(x, y, s * 0.25 * width, 0.12 * height,
                               max(0.29, 0.12 * width), max(0.29, 0.09 * height))
                      for s in (-1, 1)]
            eyes = [gaussian(x, y, s * 0.22 * width, -0.13 * height,
                             max(0.26, 0.10 * width), max(0.25, 0.055 * height))
                    for s in (-1, 1)]
            nose = gaussian(x, y, 0, -0.04 * height,
                            max(0.25, 0.075 * width), max(0.28, 0.12 * height))
            mouth = gaussian(x, y, 0, 0.29 * height,
                             0.20 * width, max(0.25, 0.055 * height))
            chin = gaussian(x, y, 0, 0.42 * height,
                            0.18 * width, max(0.25, 0.06 * height))
            put_min_constraint(bank, biases, skin_kernel=center,
                               threshold=0.66, penalty=3.0)
            for cheek in cheeks:
                put_min_constraint(bank, biases, skin_kernel=cheek,
                                   threshold=0.55, penalty=2.5)
            put_min_constraint(bank, biases, skin_kernel=center - ring,
                               threshold=0.24, penalty=2.0)
            for cheek, eye in zip(cheeks, eyes):
                put_min_constraint(bank, biases, gray_kernel=cheek - eye,
                                   threshold=0.25, penalty=3.0)
            eye_pair = 0.5 * (eyes[0] + eyes[1])
            put_min_constraint(bank, biases, gray_kernel=forehead - eye_pair,
                               threshold=0.20, penalty=2.0)
            put_min_constraint(bank, biases, gray_kernel=nose - eye_pair,
                               threshold=0.15, penalty=2.0)
            put_min_constraint(bank, biases,
                               gray_kernel=0.5 * (cheeks[0] + cheeks[1]) - mouth,
                               threshold=0.13, penalty=2.5)
            put_min_constraint(bank, biases, gray_kernel=chin - mouth,
                               threshold=0.08, penalty=2.0)
            # Loose bilateral symmetry constraint rejects one-sided textures.
            for sign in (-1, 1):
                put_min_constraint(bank, biases,
                                   gray_kernel=sign * (cheeks[0] - cheeks[1]),
                                   threshold=-0.65, penalty=0.8)
            spans.append((start, len(bank)))
    reduction = np.zeros((len(spans), len(bank), 1, 1), np.float32)
    for j, (start, end) in enumerate(spans):
        reduction[j, start:end, 0, 0] = -1.0
    return np.array(bank), np.array(biases), reduction, np.ones(len(spans))


def contextual_person_bank():
    """Long-range anatomy constraints centered on the head, not a crop.

    A small warm blob alone is common in objects. The same blob supported by
    bilateral shoulders and a filled torso is a much more specific event.
    Each intensity comparison gets its own hinge, so one high-contrast edge
    cannot compensate for the absence of the opposite shoulder or the torso.
    """
    x, y = grid(13)
    bank, biases, spans = [], [], []
    # Upright and gently leaning bodies; the entire coordinate frame rotates
    # around the head so that all tests still describe the same individual.
    # A second, stricter mode permits low-saturation, light heads with dark
    # clothing. It uses strong achromatic head/background contrast in place
    # of strong skin saturation.
    configurations = [(w, bw, sign, angle, False)
                      for w in (1.6, 2.4, 3.4)
                      for bw in (0.82, 1.18)
                      for sign in (-1, 1)
                      for angle in (-0.20, 0.0, 0.20)]
    configurations += [(w, 1.0, -1, 0.0, True) for w in (2.4, 3.4)]
    for head_width, body_width, sign, angle, achromatic in configurations:
        x0, y0 = grid(13)
        # Offset the head upwards within the receptive field, so
        # more of the window is available for the torso below it.
        x = x0 * np.cos(angle) + (y0 + 2.0) * np.sin(angle)
        y = -x0 * np.sin(angle) + (y0 + 2.0) * np.cos(angle) - 2.0
        start = len(bank)
        hw = head_width
        hh = 1.30 * hw
        cx, cy = 0, -2.0
        head = gaussian(x, y, cx, cy, 0.28 * hw, 0.31 * hh)
        surround = ellipse_ring(x, y, cx, cy, 0.5 * hw, 0.5 * hh)
        torso = gaussian(x, y, 0, cy + 1.55 * hh,
                         0.65 * hw * body_width, 0.62 * hh)
        if not achromatic:
            put_min_constraint(bank, biases, skin_kernel=head,
                               threshold=0.64, penalty=3.0)
            put_min_constraint(bank, biases, skin_kernel=head - surround,
                               threshold=0.26, penalty=2.0)
            put_min_constraint(bank, biases, skin_kernel=head - torso,
                               threshold=0.20, penalty=1.5)
        else:
            put_min_constraint(bank, biases, skin_kernel=head,
                               threshold=0.0, penalty=1.6)
            put_min_constraint(bank, biases, gray_kernel=head - surround,
                               threshold=0.60, penalty=2.2)
            put_min_constraint(bank, biases, gray_kernel=head - torso,
                               threshold=0.80, penalty=2.0)
        # A little exposed neck beneath the face supplies a second,
        # narrower skin patch. Its deliberately weak penalty allows
        # collars and scarves, while discouraging isolated warm dots.
        neck = gaussian(x, y, 0, cy + 0.58 * hh,
                        max(0.24, 0.14 * hw), max(0.24, 0.13 * hh))
        put_min_constraint(bank, biases, skin_kernel=neck,
                           threshold=0.18, penalty=0.50)
        # At both sides, sample background above the shoulder and
        # torso below it. Both sides must have the same polarity.
        for s in (-1, 1):
            shoulder_inside = gaussian(x, y, s * 0.70 * hw * body_width,
                                       cy + 0.96 * hh,
                                       0.26 * hw, 0.20 * hh)
            above_shoulder = gaussian(x, y, s * 0.70 * hw * body_width,
                                      cy + 0.12 * hh,
                                      0.26 * hw, 0.22 * hh)
            put_min_constraint(bank, biases,
                               gray_kernel=sign * (shoulder_inside - above_shoulder),
                               threshold=0.38, penalty=1.8)
            inside = gaussian(x, y, s * 0.55 * hw * body_width,
                              cy + 1.8 * hh, 0.25 * hw, 0.32 * hh)
            outside = gaussian(x, y, s * 1.35 * hw * body_width,
                               cy + 1.8 * hh, 0.24 * hw, 0.32 * hh)
            put_min_constraint(bank, biases,
                               gray_kernel=sign * (inside - outside),
                               threshold=0.22, penalty=1.15)
            # A genuine head is narrower than the shoulder span.
            beside_head = gaussian(x, y, s * 0.78 * hw, cy,
                                   0.23 * hw, 0.27 * hh)
            if not achromatic:
                put_min_constraint(bank, biases, skin_kernel=head - beside_head,
                                   threshold=0.28, penalty=1.5)
            else:
                put_min_constraint(bank, biases, gray_kernel=head - beside_head,
                                   threshold=0.55, penalty=1.8)
        # The torso is a region, not just two disconnected edges.
        upper_torso = gaussian(x, y, 0, cy + 1.15 * hh,
                               0.45 * hw * body_width, 0.22 * hh)
        lower_torso = gaussian(x, y, 0, cy + 2.10 * hh,
                               0.44 * hw * body_width, 0.25 * hh)
        flank = 0.5 * (gaussian(x, y, -1.3 * hw * body_width,
                               cy + 1.6 * hh, 0.25 * hw, 0.7 * hh)
                       + gaussian(x, y, 1.3 * hw * body_width,
                                  cy + 1.6 * hh, 0.25 * hw, 0.7 * hh))
        put_min_constraint(bank, biases,
                           gray_kernel=sign * (0.5 * (upper_torso + lower_torso) - flank),
                           threshold=0.40, penalty=1.8)
        spans.append((start, len(bank)))
    reduction = np.zeros((len(spans), len(bank), 1, 1), np.float32)
    for j, (start, stop) in enumerate(spans):
        reduction[j, start:stop, 0, 0] = -1
    return np.array(bank), np.array(biases), reduction, np.ones(len(spans))


def channel_max(g, name, source, channels):
    stage = 0
    while channels > 1:
        pairs = (channels + 1) // 2
        w = np.zeros((2 * pairs, channels, 1, 1), np.float32)
        for j in range(pairs):
            w[2 * j, 2 * j, 0, 0] = 1
            if 2 * j + 1 < channels:
                w[2 * j + 1, 2 * j + 1, 0, 0] = 1
                w[2 * j + 1, 2 * j, 0, 0] = -1
        a = g.conv(name + '_split' + str(stage), source, w)
        a = g.relu(name + '_hinge' + str(stage), a)
        w = np.zeros((pairs, 2 * pairs, 1, 1), np.float32)
        for j in range(pairs):
            w[j, 2 * j:2 * j + 2, 0, 0] = 1
        source = g.conv(name + '_max' + str(stage), a, w)
        channels = pairs
        stage += 1
    return source


def spatial_max(g, name, source, channels, side):
    a = g.pool(name + '_xy', source, side, kind='maxpool')
    a = g.op(name + '_as_column', 'reshape', a, shape=[1, channels, 1])
    a = g.pool(name + '_channels', a, [channels, 1], kind='maxpool')
    return g.op(name + '_scalar', 'reshape', a, shape=[1])


def build(destination):
    g = Graph()
    rgb = g.pool('antialias_half', 'image', 2)
    # Soft polygonal skin-colour membership. Coefficients express familiar
    # red dominance, rejection of purple/red/yellow paint, and a dark floor.
    # Every row is a violation, rectified before subtracting from one.
    constraints = [
        ([-0.955, 1.0, 0.0], 0.012, 25.0),
        ([-0.920, 0.0, 1.0], 0.012, 20.0),
        ([0.0, -1.0, 1.0], -0.025, 15.0),
        ([1.0, -1.85, 0.0], -0.045, 8.0),
        ([0.0, 1.0, -1.85], -0.060, 8.0),
        ([-1.0, 0.0, 0.0], 0.100, 8.0),
        ([1.0, -1.0, 0.0], -0.300, 5.0),
    ]
    w = np.array([np.array(a) * scale for a, b, scale in constraints])[:, :, None, None]
    b = [b * scale for a, b, scale in constraints]
    cv = g.conv('colour_inequalities', rgb, w, b)
    cv = g.relu('colour_violations', cv)
    w = np.zeros((2, len(constraints), 1, 1), np.float32)
    w[0, :, 0, 0] = -1
    sk = g.conv('skin_membership_affine', cv, w, [1, 0])
    sk = g.relu('skin_membership', sk)
    gray = g.conv('luminance', rgb, np.array([[[[0.299]], [[0.587]], [[0.114]]]]))
    gray = g.op('luminance_standardized', 'norm', gray)
    w = np.zeros((2, 1, 1, 1), np.float32)
    w[1, 0, 0, 0] = 1
    gr = g.conv('luminance_embedding', gray, w)
    features = g.op('skin_and_luminance', 'add', [sk, gr])

    fw, fb, fr, fc = face_bank()
    pw, pb, pr, pc = contextual_person_bank()
    summaries = []
    for level, side in enumerate((48, 24, 12, 6)):
        prefix = 'scale' + str(level)
        if level:
            features = g.pool(prefix + '_appearance_down', features, 2)
        fv = g.conv(prefix + '_face_tests', features, fw, fb, padding=4)
        fv = g.relu(prefix + '_face_shortfalls', fv)
        fs = g.conv(prefix + '_face_conjunctions', fv, fr, fc)
        fs = g.relu(prefix + '_face_candidates', fs)
        face_map = channel_max(g, prefix + '_face_choices', fs, len(fc))
        # Facial evidence is only accepted where a larger skin-supported
        # head/neck/shoulder arrangement is present at the same location.
        # Chunking keeps the interpreter's intermediate arrays modest; it
        # does not alter the bank or evaluate anything during construction.
        context_chunks = []
        per_candidate = pw.shape[0] // len(pc)
        for chunk_index, start in enumerate(range(0, len(pc), 8)):
            stop = min(start + 8, len(pc))
            i, j = start * per_candidate, stop * per_candidate
            tag = prefix + '_context' + str(chunk_index)
            pv = g.conv(tag + '_tests', features, pw[i:j], pb[i:j], padding=6)
            pv = g.relu(tag + '_shortfalls', pv)
            ps = g.conv(tag + '_conjunctions', pv, pr[start:stop, i:j], pc[start:stop])
            ps = g.relu(tag + '_candidates', ps)
            chunk_max = channel_max(g, tag + '_choices', ps, stop - start)
            context_chunks.append(chunk_max)
        context_map = context_chunks[0]
        for i, next_map in enumerate(context_chunks[1:]):
            diff = g.conv(prefix + '_context_neg' + str(i), next_map,
                          np.array([[[[-1.0]]]]))
            diff = g.op(prefix + '_context_diff' + str(i), 'add', [context_map, diff])
            diff = g.relu(prefix + '_context_hinge' + str(i), diff)
            context_map = g.op(prefix + '_context_merge' + str(i), 'add', [next_map, diff])
        shift = np.zeros((1, 1, 5, 5), np.float32)
        shift[0, 0, 0, 2] = 1
        aligned_face = g.conv(prefix + '_align_face_to_context', face_map, shift, padding=2)
        both = g.op(prefix + '_face_with_body', 'add', [aligned_face, context_map])
        both = g.conv(prefix + '_joint_anatomy', both,
                      np.array([[[[1.0]]]]), [-1.0])
        both = g.relu(prefix + '_joint_candidates', both)
        face = spatial_max(g, prefix + '_joint_max', both, 1, side)
        body = spatial_max(g, prefix + '_body_max', context_map, 1, side)
        # Embed the scalars at distinct positions, then combine with add.
        for kind, scalar in enumerate((face, body)):
            w = np.zeros((8, 1), np.float32)
            w[4 * kind + level, 0] = 1
            summaries.append(g.dense(prefix + '_embed_' + str(kind), scalar, w))

    merged = g.op('all_scale_summaries', 'add', summaries)
    merged = g.op('scale_axis', 'reshape', merged, shape=[2, 4, 1])
    merged = g.pool('best_scale', merged, [4, 1], kind='maxpool')
    # Each branch must independently explain a complete local structure.
    # Evidence at unrelated positions is never treated as a single person.
    alternatives = g.dense('person_alternatives', merged,
                           [[1, 0], [0, 1]], [-0.80, -0.93])
    alternatives = g.op('alternative_axis', 'reshape', alternatives, shape=[1, 2, 1])
    evidence = g.pool('best_person_evidence', alternatives, [2, 1], kind='maxpool')
    g.dense('scores', evidence, [[0], [1]], [0, 0])

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'network.json').write_text(json.dumps({'layers': g.layers}, indent=2) + '\n')
    np.savez(destination / 'weights.npz', **g.weights)
    total = sum(a.size for a in g.weights.values())
    print('Wrote', len(g.layers), 'layers and', total, 'float32 weights.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 build.py OUTPUT_DIRECTORY')
    build(sys.argv[1])
