"""Data-free construction of a fixed, analytic person-presence network.

All constants describe generic colours, facial proportions, contours, or fuzzy
logic rules. No data is opened, no parameters are estimated, and no optimizer
or external weights are used. The only filesystem operations write the answer.
"""
import json
import math
import sys
from pathlib import Path
import numpy as np


class Graph:
    def __init__(self):
        self.layers = []
        self.weights = {}
        self.shapes = {'image': (3, 96, 96)}

    def conv(self, name, source, w, b=None, stride=1, padding=0):
        w = np.asarray(w, dtype=np.float32)
        o, c, k, _ = w.shape
        assert self.shapes[source][0] == c
        if b is None:
            b = np.zeros(o, dtype=np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.asarray(b, dtype=np.float32)
        self.layers.append(dict(name=name, op='conv', inputs=[source], out=o,
                                kernel=[k, k], stride=stride, padding=padding))
        _, h, ww = self.shapes[source]
        self.shapes[name] = (o, (h + 2*padding-k)//stride+1,
                            (ww + 2*padding-k)//stride+1)
        return name

    def point(self, name, source, w, b=None):
        w = np.asarray(w, dtype=np.float32)
        return self.conv(name, source, w[:, :, None, None], b)

    def simple(self, name, op, source, **settings):
        ins = source if isinstance(source, list) else [source]
        self.layers.append(dict(name=name, op=op, inputs=ins, **settings))
        sh = self.shapes[ins[0]]
        if op in ('maxpool', 'avgpool'):
            c, h, w = sh
            kh, kw = settings['size']; sy, sx = settings['stride']
            sh = (c, (h-kh)//sy+1, (w-kw)//sx+1)
        elif op == 'reshape':
            sh = tuple(settings['shape'])
        self.shapes[name] = sh
        return name

    def relu(self, name, source):
        return self.simple(name, 'relu', source)

    def average(self, name, source, size=2):
        return self.simple(name, 'avgpool', source,
                           size=[size, size], stride=[size, size])

    def maximum(self, name, source):
        c, h, w = self.shapes[source]
        spatial = self.simple(name+'_space', 'maxpool', source,
                              size=[h, w], stride=[h, w])
        channels = self.simple(name+'_channels', 'reshape', spatial,
                               shape=[1, c, 1])
        return self.simple(name, 'maxpool', channels,
                           size=[c, 1], stride=[c, 1])


def gaussian(k, x0, y0, sx, sy, angle=0.0):
    """Pixel-integrated, normalized elliptical Gaussian; coordinates are generic."""
    grid = np.arange(k, dtype=np.float64) - (k-1)/2
    xx, yy = np.meshgrid(grid, grid)
    result = np.zeros((k, k), np.float64)
    ca, sa = math.cos(angle), math.sin(angle)
    # Integrating a formula over each cell prevents a small stencil from
    # depending excessively on the placement of the pixel centres.
    for dx in (-0.375, -0.125, 0.125, 0.375):
        for dy in (-0.375, -0.125, 0.125, 0.375):
            x = ca*(xx+dx) + sa*(yy+dy)
            y = -sa*(xx+dx) + ca*(yy+dy)
            result += np.exp(-0.5*((x-x0)/sx)**2 - 0.5*((y-y0)/sy)**2)
    return result / result.sum()


def colour_and_luminance(g):
    image = g.average('half_image', 'image')
    # Halfspace violations of a broad, soft skin-colour region. These are
    # colour-theoretic constraints, not an empirical skin-colour histogram.
    r = np.array([1., 0., 0.]); gr = np.array([0., 1., 0.]); b = np.array([0., 0., 1.])
    u, v, warmth = r-gr, gr-b, r-b
    lum = np.array([0.299, 0.587, 0.114])
    inequalities = [
        (-u, 0.010, 24.),
        (-warmth, 0.060, 15.),
        (-v, -0.015, 16.),
        (v-0.85*u, -0.015, 12.),
        (u-3.0*v, -0.100, 12.),
        (u, -0.280, 8.),
        (v, -0.220, 8.),
        (-lum, 0.090, 10.),
        (lum, -0.940, 6.),
    ]
    q = g.point('colour_halfspaces', image,
                np.stack([a for a, _, _ in inequalities]),
                [bb for _, bb, _ in inequalities])
    q = g.relu('colour_violations', q)
    s = g.point('skin_affinity_raw', q,
                [[-slope for _, _, slope in inequalities]], [1.])
    s = g.relu('skin_affinity', s)
    grey = g.point('luminance', image, [lum])
    grey = g.simple('contrast_reference', 'norm', grey)
    gp = g.point('face_luminance_channels', grey, [[1.], [0.], [0.]])
    sp = g.point('face_skin_channels', s, [[0.], [1.], [0.]], [0., 0., 1.])
    face = g.simple('face_input_48', 'add', [gp, sp])
    faces = {48: face}
    skins = {48: s}
    for a, z in [(48, 24), (24, 12)]:
        faces[z] = g.average('face_input_'+str(z), faces[a])
        skins[z] = g.average('skin_'+str(z), skins[a])
    return grey, faces, skins


def face_bank(g, source, height, tag, stride=1):
    # Width/height variation accommodates both normal anatomical variation
    # and the task's non-aspect-preserving image resize.
    geometries = [(0.60, 0.), (0.83, 0.), (1.06, 0.)]
    if g.shapes[source][1] < 48 or height >= 5:
        geometries += [(0.80, -0.28), (0.80, 0.28)]
    k = int(math.ceil(1.6*height))
    k += (k % 2 == 0)
    k = max(7, k)
    weights = []
    for aspect, angle in geometries:
        h, w = height, height*aspect
        def blob(x, y, sx, sy):
            return gaussian(k, x*w, y*h, sx*w, sy*h, angle)
        cheek = blob(0., 0.08, 0.29, 0.27)
        left = blob(-0.68, 0.02, 0.15, 0.24)
        right = blob(0.68, 0.02, 0.15, 0.24)
        outside = (blob(0., -0.56, 0.28, 0.09) +
                   blob(0., 0.58, 0.28, 0.10))/2
        eyes = [blob(x, -0.13, 0.12, 0.065) for x in (-0.24, 0.24)]
        references = [(blob(x, -0.32, 0.13, 0.075) +
                       blob(x, 0.12, 0.14, 0.09))/2 for x in (-0.24, 0.24)]
        nose = blob(0., -0.055, 0.10, 0.12)
        mouth = blob(0., 0.245, 0.21, 0.055)
        mouth_ref = (blob(0., 0.10, 0.24, 0.075) +
                     blob(0., 0.385, 0.20, 0.065))/2
        # Every dark anatomical band needs a brighter region on BOTH sides;
        # the average alone would accept an ordinary step edge.
        forehead = [blob(x, -0.32, .13, .075) for x in (-.24, .24)]
        cheeks = [blob(x, .12, .14, .09) for x in (-.24, .24)]
        upper_lip = blob(0., .10, .24, .075)
        chin = blob(0., .385, .20, .065)
        crown = blob(0., -.64, .27, .12)
        below = blob(0., .69, .30, .12)
        spec = [(1, cheek), (1, cheek-left), (1, cheek-right),
                (1, cheek-outside),
                (0, references[0]-eyes[0]), (0, references[1]-eyes[1]),
                (0, nose-(eyes[0]+eyes[1])/2),
                (0, mouth_ref-mouth), (2, cheek),
                (0, forehead[0]-eyes[0]), (0, cheeks[0]-eyes[0]),
                (0, forehead[1]-eyes[1]), (0, cheeks[1]-eyes[1]),
                (0, upper_lip-mouth), (0, chin-mouth),
                (1, cheek-crown), (1, cheek-below)]
        for channel, mask in spec:
            a = np.zeros((3, k, k), np.float64)
            a[channel] = mask
            weights.append(a)
    n = len(geometries)
    features = g.conv(tag+'_measurements', source, weights,
                      stride=stride, padding=k//2)
    # Independent clauses prevent skin alone, or a single high-contrast edge,
    # from standing in for a face. Each clause has a capped, nonnegative deficit.
    clauses = np.eye(17, dtype=np.float32)
    target = np.array([.62, .13, .13, .12, .19, .19, .14, .08, .97,
                       .12, .12, .12, .12, .045, .045, .12, .06])
    importance = np.array([1.3, 1.3, 1.3, .8, 1.6, 1.6, 1.6, 1.6, 5.,
                           1.8, 1.8, 1.8, 1.8, 1.6, 1.6, 1., .8])
    w = np.kron(np.eye(n, dtype=np.float32), -clauses)
    z = g.point(tag+'_shortfalls', features, w, np.tile(target, n))
    z = g.relu(tag+'_penalties', z)
    w = np.zeros((n, n*17), np.float32)
    for i in range(n):
        w[i, i*17:(i+1)*17] = -importance
    score = g.point(tag+'_rules', z, w, np.full(n, .035))
    return g.maximum(tag+'_best', score)


def orientation_maps(g, grey):
    smooth = np.array([.25, .5, .25])
    diff = np.array([-.5, 0., .5])
    dx = np.outer(smooth, diff); dy = dx.T
    filters = [dx, (dx+dy)/math.sqrt(2), dy, (dy-dx)/math.sqrt(2)]
    w = np.stack([sign*f for f in filters for sign in (1., -1.)])[:, None]
    z = g.conv('oriented_derivatives', grey, w, padding=1)
    z = g.relu('derivative_halfwaves', z)
    w = np.zeros((4, 8), np.float32)
    for i in range(4):
        w[i, 2*i:2*i+2] = 1.
        j = (i+2) % 4
        w[i, 2*j:2*j+2] -= .70
    z = g.point('orientation_selectivity', z, w, [-.015]*4)
    z = g.relu('oriented_edges', z)
    w = np.concatenate([np.eye(4), np.eye(4)], axis=0)
    z = g.point('edge_cap_arguments', z, w, [0.]*4+[-.6]*4)
    z = g.relu('edge_cap_parts', z)
    z = g.point('bounded_edges', z,
                np.concatenate([np.eye(4), -np.eye(4)], axis=1)/.6)
    return z


def contours(k, paths, width, lean, height):
    """Analytic unsigned-orientation stencil of explicitly specified paths."""
    ans = np.zeros((4, k, k), np.float64)
    grid = np.arange(k, dtype=np.float64) - (k-1)/2
    xx, yy = np.meshgrid(grid, grid)
    for path in paths:
        # Paths are supplied in units of the generic figure's height.
        p = np.array(path, np.float64)
        p[:, 0] = width*p[:, 0] + lean*p[:, 1]
        p *= height
        for start, end in zip(p[:-1], p[1:]):
            vector = end-start
            length = float(np.sqrt((vector*vector).sum()))
            if length < 1e-9:
                continue
            normal = np.array([-vector[1], vector[0]])
            angle = math.atan2(normal[1], normal[0]) % math.pi
            slot = angle/(math.pi/4)
            i = int(math.floor(slot)) % 4
            frac = slot-math.floor(slot)
            steps = max(1, int(math.ceil(length*4)))
            for t in (np.arange(steps)+.5)/steps:
                x, y = start + vector*t
                m = np.exp(-((xx-x)**2+(yy-y)**2)/(2*.55**2))
                m /= m.sum()
                ans[i] += m * length/steps * (1-frac)
                ans[(i+1) % 4] += m * length/steps * frac
    ans /= ans.sum()
    # Penalize incorrect orientations along the path, and unstructured edge
    # clutter in its neighbourhood. This is a matched contour, not an edge count.
    support = ans.sum(axis=0)
    background = np.zeros_like(support)
    for path in paths:
        for x, y in path:
            background += np.exp(-((xx-(width*x+lean*y)*height)**2 +
                                   (yy-y*height)**2)/(2*1.35**2))
    background /= background.sum()
    return ans - .55*support[None]/4 - .65*background[None]/4


def figure_paths(mode):
    if mode == 'standing':
        cy, rx, ry = -.39, .055, .073
        shoulder = []
        lower = []
        for s in (-1, 1):
            shoulder += [[(s*.035, -.32), (s*.035, -.285), (s*.13, -.25),
                          (s*.18, -.04), (s*.14, .07)]]
            lower += [[(s*.10, -.24), (s*.095, .10), (s*.08, .18), (s*.13, .48)],
                      [(s*.03, .18), (s*.06, .48)]]
    else:
        cy, rx, ry = -.30, .125, .17
        shoulder = []
        lower = []
        for s in (-1, 1):
            shoulder += [[(s*.07, -.13), (s*.07, -.06), (s*.28, .02),
                          (s*.34, .21)]]
            lower += [[(s*.34, .21), (s*.40, .40)],
                      [(s*.22, .08), (s*.21, .48)]]
    t = np.linspace(0, 2*math.pi, 41)
    head = [[(rx*math.cos(a), cy+ry*math.sin(a)) for a in t]]
    return head, shoulder, lower, (cy, rx, ry)


def body_bank(g, source, height, tag, stride=1):
    geometries = [(.80, 0.), (1.20, 0.), (1., -.16), (1., .16)]
    k = int(math.ceil(height+3))
    k += (k % 2 == 0)
    weights = []
    modes = []
    for mode in ('standing', 'upper_body'):
        head, shoulder, lower, (cy, rx, ry) = figure_paths(mode)
        for width, lean in geometries:
            modes.append(mode)
            # Channels: four edge orientations, soft skin, valid-image mask.
            for paths in (head, shoulder, lower):
                a = np.zeros((6, k, k), np.float64)
                a[:4] = contours(k, paths, width, lean, height)
                weights.append(a)
            hx = lean*cy*height
            hy = (cy+.22*ry)*height
            center = gaussian(k, hx, hy, max(.26, .70*rx*width*height),
                              max(.28, .60*ry*height))
            sides = (gaussian(k, hx-1.9*rx*width*height, hy,
                              max(.30, .48*rx*width*height), max(.35, .9*ry*height)) +
                     gaussian(k, hx+1.9*rx*width*height, hy,
                              max(.30, .48*rx*width*height), max(.35, .9*ry*height)))/2
            top = gaussian(k, hx-lean*1.6*ry*height, hy-1.6*ry*height,
                           max(.3, .65*rx*width*height), max(.3, .45*ry*height))
            bottom = gaussian(k, hx+lean*1.8*ry*height, hy+1.8*ry*height,
                              max(.3, 1.05*rx*width*height), max(.3, .45*ry*height))
            for channel, mask in ((4, center), (4, center-sides), (5, center),
                                  (4, center-top), (4, center-bottom)):
                a = np.zeros((6, k, k), np.float64)
                a[channel] = mask
                weights.append(a)
    n = len(modes)
    features = g.conv(tag+'_measurements', source, weights,
                      stride=stride, padding=k//2)
    # Two rules: modest contour support plus an isolated skin-coloured head;
    # or a much stronger complete standing silhouette, without a colour test.
    colour_target = np.array([.065, .065, .045, .65, .28, .97, .24, .28])
    colour_weight = np.array([3., 3.5, 2.5, 1.2, 1.8, 4., 1.8, 1.8])
    shape_target = np.array([.22, .22, .19, 0., 0., .97, 0., 0.])
    shape_weight = np.array([2.3, 2.4, 2.1, 0., 0., 4., 0., 0.])
    w = np.zeros((n*16, n*8), np.float32)
    bias = []
    for i in range(n):
        w[i*16:i*16+8, i*8:i*8+8] = -np.eye(8)
        w[i*16+8:i*16+16, i*8:i*8+8] = -np.eye(8)
        bias.extend(colour_target)
        bias.extend(shape_target)
    z = g.point(tag+'_shortfalls', features, w, bias)
    z = g.relu(tag+'_penalties', z)
    w = np.zeros((2*n, 16*n), np.float32)
    bias = []
    for i, mode in enumerate(modes):
        w[2*i, i*16:i*16+8] = -colour_weight
        w[2*i+1, i*16+8:i*16+16] = -shape_weight
        bias.extend([.030, .025 if mode == 'standing' else -1.])
    scores = g.point(tag+'_rules', z, w, bias)
    return g.maximum(tag+'_best', scores)


def compact_head_bank(g, source, height, tag, stride=1):
    """Small unresolved heads: compact skin above a broader coherent torso.

    Unlike the face rule, this does not demand visible eyes or a mouth. A
    consistent two-sided torso contrast is necessary: isolated warm dots alone
    must not be sufficient. Light and dark clothing are treated symmetrically.
    """
    aspects = (.65, .90, 1.15)
    k = int(math.ceil(4*height))
    k += (k % 2 == 0)
    weights = []
    for aspect in aspects:
        h, w = height, height*aspect
        def blob(x, y, sx, sy):
            return gaussian(k, x*w, y*h, sx*w, sy*h)
        core = blob(0., .06, .25, .28)
        left = blob(-.70, .03, .16, .25)
        right = blob(.70, .03, .16, .25)
        above = blob(0., -.65, .24, .13)
        below = blob(0., 1.0, .42, .18)
        left_cheek = blob(-.19, .10, .12, .17)
        right_cheek = blob(.19, .10, .12, .17)
        lower = blob(0., .28, .18, .12)
        torso = blob(0., 1.23, .53, .29)
        outside_left = blob(-1.40, 1.23, .25, .29)
        outside_right = blob(1.40, 1.23, .25, .29)
        eye_band = blob(0., -.14, .29, .09)
        forehead = blob(0., -.35, .25, .09)
        cheeks = blob(0., .16, .26, .13)
        spec = [(1, core), (1, core-left), (1, core-right),
                (1, core-above), (1, core-below),
                (1, left_cheek), (1, right_cheek), (1, lower),
                (0, torso-outside_left), (0, torso-outside_right),
                (1, core-(left+right+above+below)/4), (2, core),
                (0, forehead-eye_band), (0, cheeks-eye_band),
                (1, left_cheek-right_cheek), (1, right_cheek-left_cheek),
                (0, left_cheek-right_cheek), (0, right_cheek-left_cheek)]
        for channel, mask in spec:
            a = np.zeros((3, k, k), np.float64)
            a[channel] = mask
            weights.append(a)
    n = len(aspects)
    features = g.conv(tag+'_measurements', source, weights,
                      stride=stride, padding=k//2)
    w = np.zeros((n*36, n*18), np.float32)
    target = [.65, .22, .22, .20, .25, .53, .53, .53, .24, .24, .28, .97,
              .085, .085, -.20, -.20, -.45, -.45]
    importance = np.array([1., 1.2, 1.2, 1.2, 1.2, .6, .6, .6, .9, .9, 1.8, 5.,
                           1.5, 1.5, .8, .8, .5, .5])
    for i in range(n):
        for j, sign in enumerate((-1., 1.)):
            block = -np.eye(18, dtype=np.float32)
            block[8, 8] *= sign; block[9, 9] *= sign
            w[i*36+j*18:i*36+j*18+18, i*18:i*18+18] = block
    z = g.point(tag+'_shortfalls', features, w, np.tile(target, 2*n))
    z = g.relu(tag+'_penalties', z)
    w = np.zeros((2*n, 36*n), np.float32)
    for i in range(2*n):
        w[i, i*18:i*18+18] = -importance
    z = g.point(tag+'_rules', z, w, [.030]*(2*n))
    return g.maximum(tag+'_best', z)


def construct():
    g = Graph()
    grey, faces, skins = colour_and_luminance(g)
    evidence = []
    for resolution, heights in ((48, (3.5, 5.)), (24, (3.5, 5.)),
                                (12, (3.5, 5., 7.))):
        for j, height in enumerate(heights):
            evidence.append(face_bank(g, faces[resolution], height,
                                      f'face_{resolution}_{j}',
                                      stride=2 if resolution == 48 and height >= 5 else 1))
    for resolution, heights in ((48, (2.3, 3.3)), (24, (2.3, 3.3)),
                                (12, (2.3, 3.3))):
        for j, height in enumerate(heights):
            evidence.append(compact_head_bank(g, faces[resolution], height,
                                               f'compact_{resolution}_{j}',
                                               stride=2 if resolution == 48 else 1))
    edges = orientation_maps(g, grey)
    edges24 = g.average('edges_24', edges)
    edges12 = g.average('edges_12', edges24)
    for resolution, edges in ((24, edges24), (12, edges12)):
        w = np.zeros((6, 4), np.float32); w[:4] = np.eye(4)
        ep = g.point(f'body_edge_channels_{resolution}', edges, w)
        w = np.zeros((6, 1), np.float32); w[4, 0] = 1.
        sp = g.point(f'body_skin_channels_{resolution}', skins[resolution],
                     w, [0., 0., 0., 0., 0., 1.])
        source = g.simple(f'body_input_{resolution}', 'add', [ep, sp])
        for j, height in enumerate((7.5, 11.5)):
            evidence.append(body_bank(g, source, height, f'body_{resolution}_{j}',
                                      stride=2 if resolution == 24 and height > 10 else 1))
    # Put scalar rule margins in separate channels, then implement logical OR
    # with a maximum. There is no learned readout and no class-frequency bias.
    projections = []
    for i, e in enumerate(evidence):
        w = np.zeros((len(evidence), 1), np.float32); w[i, 0] = 1.
        projections.append(g.point('evidence_channel_'+str(i), e, w))
    combined = g.simple('all_evidence', 'add', projections)
    best = g.maximum('strongest_evidence', combined)
    # A warm surface covering much of the scene makes an accidental small
    # skin-shaped region less informative than one in a mixed-colour scene.
    coverage = g.average('warm_scene_coverage', skins[12], size=12)
    coverage = g.point('warm_scene_excess', coverage, [[1.]], [-.18])
    coverage = g.relu('warm_scene_penalty', coverage)
    coverage = g.point('warm_scene_correction', coverage, [[-.7]])
    best = g.simple('contextual_evidence', 'add', [best, coverage])
    g.layers.append(dict(name='scores', op='dense', inputs=[best], out=2))
    g.weights['scores.W'] = np.array([[0.], [1.]], dtype=np.float32)
    g.weights['scores.b'] = np.zeros(2, np.float32)
    return {'layers': g.layers}, g.weights


def main():
    destination = Path(sys.argv[1])
    destination.mkdir(parents=True, exist_ok=True)
    network, weights = construct()
    with (destination/'network.json').open('w') as f:
        json.dump(network, f, indent=2)
        f.write('\n')
    np.savez(destination/'weights.npz', **weights)
    print('Wrote', len(network['layers']), 'layers and',
          sum(a.size for a in weights.values()), 'float32 weights.')


if __name__ == '__main__':
    main()
