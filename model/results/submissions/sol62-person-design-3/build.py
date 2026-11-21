"""Compile analytic color, part and scene rules to the eight network ops.

All dimensions, thresholds, and readout coefficients are specified here. The
builder only writes files. In particular there is no training or image I/O.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np


class Graph:
    def __init__(self):
        self.layers, self.weights = [], {}
        self.shape = {'image': (3, 96, 96)}

    def op(self, name, op, inputs, result_shape, **opts):
        assert name not in self.shape
        self.layers.append(dict(name=name, op=op, inputs=inputs, **opts))
        self.shape[name] = result_shape
        return name

    def conv(self, name, src, w, b, stride=1, pad=0):
        o, i, k, l = w.shape
        c, h, v = self.shape[src]
        assert i == c and k == l
        self.weights[name + '.W'] = np.asarray(w, dtype=np.float32)
        self.weights[name + '.b'] = np.broadcast_to(b, (o,)).astype(np.float32)
        return self.op(name, 'conv', [src],
                       (o, (h + 2*pad - k)//stride + 1, (v + 2*pad - k)//stride + 1),
                       out=o, kernel=[k, k], stride=stride, padding=pad)

    def point(self, name, src, w, b=0):
        return self.conv(name, src, np.atleast_2d(w)[:, :, None, None], b)

    def dense(self, name, src, w, b=0):
        w = np.atleast_2d(w)
        assert w.shape[1] == math.prod(self.shape[src])
        self.weights[name + '.W'] = w.astype(np.float32)
        self.weights[name + '.b'] = np.broadcast_to(b, (len(w),)).astype(np.float32)
        return self.op(name, 'dense', [src], (len(w),), out=len(w))

    def pool(self, name, src, size, stride=None, op='avgpool'):
        if isinstance(size, int):
            size = (size, size)
        if stride is None:
            stride = size
        if isinstance(stride, int):
            stride = (stride, stride)
        c, h, w = self.shape[src]
        a, b = size
        u, v = stride
        return self.op(name, op, [src], (c, (h-a)//u+1, (w-b)//v+1),
                       size=list(size), stride=list(stride))

    def relu(self, name, src):
        return self.op(name, 'relu', [src], self.shape[src])

    def add(self, name, *src):
        assert all(self.shape[s] == self.shape[src[0]] for s in src)
        return self.op(name, 'add', list(src), self.shape[src[0]])

    def reshape(self, name, src, shape):
        assert math.prod(shape) == math.prod(self.shape[src])
        return self.op(name, 'reshape', [src], shape, shape=list(shape))

    def vmax(self, name, a, b):
        n, = self.shape[a]
        e = np.eye(n)
        t = self.add(name + '_delta', a, self.dense(name + '_neg', b, -e))
        return self.add(name, self.relu(name + '_win', t), b)


def diag_ramps(g, name, src, lows, highs):
    """Saturating ramps on an entire small vector."""
    ls, hs = np.asarray(lows), np.asarray(highs)
    e = np.diag(1 / (hs - ls))
    b = -ls / (hs - ls)
    r = g.relu(name + '_hinges', g.dense(name + '_thresholds', src,
               np.concatenate([e, e]), np.concatenate([b, b-1])))
    en = np.eye(len(e))
    return g.dense(name, r, np.concatenate([en, -en], axis=1))


def ramps(g, name, src, w, b, lo, hi, pad=0, stride=1):
    """A bank of spatial affine probes, each expanded to two ReLU hinges."""
    d = np.array(hi) - lo
    w = np.array(w) / d[:, None, None, None]
    b = (np.array(b) - lo) / d
    z = g.conv(name + '_probes', src, np.concatenate([w, w]),
               np.concatenate([b, b-1]), stride=stride, pad=pad)
    return g.relu(name + '_hinges', z)


def ramp_readout(m):
    m = np.atleast_2d(m)
    return np.concatenate([m, -m], axis=1)


Y = np.array([0.299, 0.587, 0.114])
A = np.array([1., -1., 0.])  # red minus green
B = np.array([0., 1., -1.])  # green minus blue
WARM = A + 0.5*B
HUE = B - 0.55*A


def face_palette(g, name, rgb):
    # A soft polygon around the warm, moderately saturated skin locus.
    # Penalize yellow/orange and magenta departures from the red-yellow axis,
    # deep shadows where chroma is unreliable, and clipped highlights.
    w = np.array([-33*WARM, 11*(WARM-0.02*Y), -26*(HUE+0.11*WARM),
                  24*(HUE-0.14*WARM), -9*Y, 11*Y, -33*A, Y,
                  6*(-A-B), 6*(-A-B), 7*(A-.65*B), 7*(A-.65*B),
                  6*(B-1.6*A), 6*(B-1.6*A)])
    b = [33*0.059, -11*0.190, -26*0.023, -24*0.034, 9*0.181,
         -11*0.933, 33*0.023, 0, -6*.055, -6*.055-1, -7*.098, -7*.098-1,
         -6*.12, -6*.12-1]
    h = g.relu(name + '_color_hinges', g.point(name + '_color_planes', rgb, w, b))
    w = np.zeros((4, len(w)))
    w[0, 7] = 1
    w[1, :7] = -1
    # Y, skin affinity, support, and a soft score for saturated dyed colors.
    # Orange and cyan cast less of a vote than blue, red, violet or turf gear.
    w[3, 8:] = [.57,-.57,.62,-.62,.29,-.29]
    return g.relu(name, g.point(name + '_palette', h, w, [0, 1, 1, 0]))


class Stencils:
    """Signed sums of spatially averaged probes, drawn from geometric parts."""
    def __init__(self, k, channels):
        self.k, self.c = k, channels
        self.w, self.lo, self.hi, self.b = [], [], [], []
        self.cache = {}

    def axis(self, center, std):
        key = center, std
        if key not in self.cache:
            u = np.arange(self.k+1) - self.k//2 - 0.5
            cdf = [math.erf((x-center)/(math.sqrt(2)*std)) for x in u]
            a = np.diff(cdf)
            self.cache[key] = a / a.sum()
        return self.cache[key]

    def patch(self, x, y, sx, sy):
        return np.outer(self.axis(y, sy), self.axis(x, sx))

    def expression(self, probes, turn=0, tx=0, ty=0):
        acc = np.zeros((self.c, self.k, self.k))
        for ch, x, y, sx, sy, gain in probes:
            acc[ch] += gain * self.patch(x+tx, y+ty+turn*x, sx, sy)
        return acc

    def add(self, probes, lo, hi, bias=0, **transform):
        idx = len(self.w)
        self.w.append(self.expression(probes, **transform))
        self.lo.append(lo)
        self.hi.append(hi)
        self.b.append(bias)
        return idx

    def compute(self, g, tag, src, stride=1, padding=None):
        pad = self.k//2 if padding is None else padding
        return ramps(g, tag, src, self.w, self.b, self.lo, self.hi, pad=pad, stride=stride)


def probe(ch, x, y, sx, sy, gain=1):
    return (ch, x, y, sx, sy, gain)


def face_bank(g, tag, src):
    """Face channels: two-eye, three-quarter, and chroma-independent evidence."""
    s = Stencils(7, 4)
    def lum(x, y, sx, sy, sign=1):
        return probe(0, x, y, sx, sy, sign)
    def skin(x, y, sx, sy, sign=1):
        return probe(1, x, y, sx, sy, sign)
    scores = [dict() for _ in range(3)]
    f, p, n = scores
    def cue(branches, probes, lo, hi):
        idx = s.add(probes, lo, hi)
        for branch, amount in branches:
            branch[idx] = amount
    for side in (-1, 1):
        # Dark eye socket between the brow/forehead and the cheek.
        cue([(f, .184), (p, .095), (n, .213)],
            [lum(side*1.13, 0.48, .66, .45, .69), lum(side*1.02, -1.93, .65, .39, .31),
             lum(side*1.18, -.77, .53, .37, -1)], .014, .119)
    cue([(f,.150),(p,.107),(n,.186)],
        [lum(0,-.29,.45,.62), lum(-1.21,-.73,.48,.42,-.5), lum(1.21,-.73,.48,.42,-.5)],
        .013,.119)
    cue([(f,.112),(p,.083),(n,.174)],
        [lum(0,.65,.73,.39,.46), lum(0,2.16,.79,.36,.36),
         lum(-1.16,1.13,.45,.45,.09), lum(1.16,1.13,.45,.45,.09),
         lum(0,1.43,.68,.29,-1)], .010,.087)
    cue([(f,.059),(p,.066),(n,.119)],
        [lum(0,-1.64,1.03,.46), lum(0,-3.12,1.22,.46,-1)], .035,.230)
    cue([(f,.234),(p,.244)], [skin(0,.12,1.28,1.46)], .29,.76)
    cue([(f,.175),(p,.235)],
        [skin(0,.10,1.15,1.36), skin(-3.09,.04,.64,1.36,-.39),
         skin(3.09,.04,.64,1.36,-.39), skin(0,-3.65,1.45,.57,-.22)], .13,.54)
    cue([(f,.079),(p,.114),(n,.085)],
        [skin(0,1.31,.93,.73,.67), skin(0,-1.5,1,.57,.33),
         skin(0,3.58,1.60,.48,-1)], .08,.51)
    # At turn angles, one eye can dominate. A positive response still needs
    # a shaded socket and illuminated nose/cheek, not just the skin blob.
    for side in (-1, 1):
        cue([(p,.081)],
            [lum(side*.93,.36,.74,.52), lum(side*1.02,-.66,.69,.44,-1)], .032,.160)
    # Do not interpret convolution padding as a dark eye or hairline.
    cue([(f,-.25),(p,-.25),(n,-.29)], [probe(2,0,0,1.34,1.7,-1)], -.963,-.70)
    h = s.compute(g, tag, src)
    m = np.zeros((len(scores), len(s.w)))
    for i, terms in enumerate(scores):
        for j, w in terms.items():
            m[i, j] = w
    return g.point(tag + '_affinities', h, ramp_readout(m))


def person_bank(g, tag, src):
    """Skin island, head above shoulders, and upright silhouette probes."""
    s = Stencils(9, 4)
    def sk(x, y, sx, sy, sign=1):
        return probe(1, .83*x, .83*y, .83*sx, .83*sy, sign)
    def lu(x, y, sx, sy, sign=1):
        return probe(0, .83*x, .83*y, .83*sx, .83*sy, sign)
    def dye(x, y, sx, sy, sign=1):
        return probe(3, .83*x, .83*y, .83*sx, .83*sy, sign)
    scores = [dict() for _ in range(4)]
    head, sil, island, clothes = scores
    def cue(branches, probes, lo, hi):
        idx = s.add(probes, lo, hi)
        for branch, amount in branches:
            branch[idx] = amount
    cue([(head,.242)], [sk(0,-3.47,.90,.91)], .33,.76)
    cue([(head,.212)],
        [sk(0,-3.47,.88,.81), sk(-2.05,-3.42,.58,.95,-.37),
         sk(2.05,-3.42,.58,.95,-.37), sk(0,-5,.94,.52,-.26)], .15,.55)
    cue([(head,.139)], [sk(0,-3.24,.79,.79), sk(0,-.43,1.29,.84,-1)], .16,.56)
    for side in (-1,1):
        shoulder = [lu(side*1.24,-.97,.64,.86), lu(side*2.96,-.95,.65,.88,-.62),
                    lu(side*1.71,-2.50,.57,.58,-.38)]
        column = [lu(side*1.06,.45,.66,1.52), lu(side*2.94,.45,.75,1.53,-1)]
        narrow_head = [lu(0,-3.63,.71,.87), lu(side*1.97,-3.56,.66,.83,-.66),
                       lu(0,-5.02,.82,.49,-.34)]
        for sign in (-1,1):
            def sign_of(ps):
                return [(*p[:-1],p[-1]*sign) for p in ps]
            cue([(head,.133),(sil,.167)], sign_of(shoulder), .036,.228)
            cue([(head,.076),(sil,.125)], sign_of(column), .042,.232)
            cue([(sil,.159)], sign_of(narrow_head), .048,.243)
    legs = [lu(0,4.28,.43,.92), lu(-1.22,3.92,.49,1.08,-.5),
            lu(1.22,3.92,.49,1.08,-.5)]
    for sign in (-1,1):
        cue([(head,.07),(sil,.148)], [(*p[:-1],p[-1]*sign) for p in legs], .028,.166)
    cue([(head,-.29),(sil,-.43),(clothes,-.22)], [probe(2,0,-1.15,1.88,2.64,-1)], -.91,-.57)
    cue([(clothes,.29)], [sk(0,-3.35,.85,.73)], .26,.73)
    cue([(clothes,.14)], [sk(0,-3.32,.88,.76), sk(0,-.7,1.16,.94,-1)], .13,.57)
    cue([(clothes,.29)], [dye(0,-.41,1.13,1.16)], .12,.53)
    cue([(clothes,.29)], [dye(0,-.39,1.0,1.15), dye(0,-3.01,.92,.76,-.42),
         dye(-2.58,-.48,.62,1.1,-.29), dye(2.58,-.48,.62,1.1,-.29)], .11,.44)
    cue([(island,.49)], [sk(0,0,.95,1.10)], .42,.85)
    surrounds = [(2.85,0),(-2.85,0),(0,3.22),(0,-3.22),
                 (2.21,2.24),(-2.21,2.24),(2.21,-2.24),(-2.21,-2.24)]
    cue([(island,.51)], [sk(0,0,.95,1.1)] +
        [sk(x,y,.75,.84,-1/len(surrounds)) for x,y in surrounds], .17,.67)
    cue([(island,-.22)], [probe(2,0,0,1.3,1.6,-1)], -.96,-.63)
    h = s.compute(g, tag, src)
    m = np.zeros((len(scores), len(s.w)))
    for i, terms in enumerate(scores):
        for j, w in terms.items():
            m[i, j] = w
    return g.point(tag + '_affinities', h, ramp_readout(m))


def scan_summary(g, tag, src, low, high, kind):
    """Pool a smooth maximum and keep a little evidence for multiple hits."""
    c, h, _ = g.shape[src]
    near = g.pool(tag + '_sectors', src, h//4, op='maxpool')
    priors = {'face': [[.86,.97,.97,.86], [.94,1,1,.94], [.89,.97,.97,.89], [.69,.79,.79,.69]],
              'body': [[.64,.73,.73,.64], [.91,.98,.98,.91], [.94,1,1,.94], [.72,.82,.82,.72]]}
    loc = np.tile(priors[kind], (c, 1, 1)).reshape(-1)
    lo = np.repeat(low, 16)
    hi = np.repeat(high, 16)
    aff = diag_ramps(g, tag + '_credible', near, lo, hi)
    weighted = g.dense(tag + '_pos_prior', aff, np.diag(loc))
    patches = g.reshape(tag + '_patches', weighted, (c,4,4))
    peak = g.pool(tag + '_peak', patches, 4, op='maxpool')
    peak = g.dense(tag + '_peak_weight', peak, .86*np.eye(c))
    avg = g.pool(tag + '_support', patches, 4)
    avg = g.dense(tag + '_support_weight', avg, .27*np.eye(c))
    return g.add(tag + '_summary', peak, avg)


def assembly(g, tag, face, body):
    """Vote for facial shading aligned with a head and torso, in one place."""
    same_grid = g.pool(tag + '_face_grid', face, 2, op='maxpool')
    # The face center is usually around three part-grid cells above the
    # shoulder/torso origin. Spreading the taps permits small pose changes.
    s = Stencils(7, 3)
    shading = s.expression([probe(2,0,-2.78,.82,.58,.53)])
    head = g.conv(tag + '_aligned_face', same_grid, shading[None], 0, pad=3)
    torso = g.point(tag + '_torso', body, [[.48,.15,0,.17]])
    joint = g.add(tag + '_match', head, torso)
    return scan_summary(g, tag, joint, [.54], [.94], 'body')


def peak_across(g, tag, src):
    y = src[0]
    for i, x in enumerate(src[1:], 1):
        y = g.vmax(f'{tag}_{i}', y, x)
    return y


def scenes(g, rgb):
    """Scene-scale color context, extended straight lines and busy detail."""
    # Each chromatic region uses a conjunction of simple soft planes.
    rules = [
        [(A, .15, .10), (B,.065,.06), (A+B,.32,.10)],                 # orange/yellow
        [(-A,.071,.074), (-B,.016,.060), (Y,.42,.15)],                 # open sky/water
        [(-A,.040,.067), (B,.065,.067), (Y,.25,.14)],                  # turf green
        [(-B,.086,.075), (-A-B,.17,.10), (-Y,-.67,.14)],               # textile blue
        [(A,.18,.084), (A+B,.14,.07), (-B,-.095,.077)],                # saturated red
        [(A,.078,.074), (-B,.059,.06)],                               # violet/pink
    ]
    pol, off, reduce = [], [], []
    for i, rule in enumerate(rules):
        for vector, thresh, fade in rule:
            pol.append(-vector/fade)
            off.append(thresh/fade)
            reduce.append(i)
    planes = g.relu('scene_color_excess', g.point('scene_color_planes', rgb, pol, off))
    w = np.zeros((len(rules), len(pol)))
    w[reduce, np.arange(len(pol))] = -1
    hue = g.relu('scene_colors', g.point('scene_color_tents', planes, w, 1))
    hue = g.pool('scene_color_cells', hue, 4)
    # Rows from top to bottom: blue sky usually lies above the subject;
    # dyes are more informative when away from the sky. Lower warm colors
    # also commonly come from food and furniture.
    region = [[.63,.89,1.18,1.30], [1.25,1.13,.96,.66], [.28,.62,1.34,1.76],
              [.37,.91,1.31,1.41], [.62,.98,1.20,1.20], [.69,1.12,1.20,.99]]
    read = np.zeros((6, 6, 6, 6))
    for i in range(6):
        # Build the spatial prior on the 6x6 pooled grid.
        vals = np.interp(np.arange(6)/5, np.arange(4)/3, region[i])
        read[i,i] = vals[:,None]/(6*vals.sum())
    color = g.dense('scene_color_context', hue, read.reshape(6,-1))
    color = diag_ramps(g, 'scene_color_strengths', color,
                       [.079,.086,.091,.010,.017,.003], [.34,.38,.38,.086,.121,.048])

    y = g.point('scene_luma', rgb, [Y])
    s = Stencils(7, 1)
    for swapped in (False, True):
        for side in (-1,1):
            def p(x,y,sx,sy,a):
                return probe(0,y,x,sy,sx,a) if swapped else probe(0,x,y,sx,sy,a)
            # Compare adjacent narrow strips across a long horizontal or
            # vertical line. The longer span suppresses isolated texture.
            s.add([p(0,-1.19,2.51,.59,side),p(0,1.19,2.51,.59,-side)], .085,.32)
    s.add([probe(0,-.95,-.95,.59,.59), probe(0,.95,.95,.59,.59),
           probe(0,-.95,.95,.59,.59,-1),probe(0,.95,-.95,.59,.59,-1)], .038,.30)
    s.add([(*p[:-1], -p[-1]) for p in [probe(0,-.95,-.95,.59,.59), probe(0,.95,.95,.59,.59),
           probe(0,-.95,.95,.59,.59,-1),probe(0,.95,-.95,.59,.59,-1)]], .038,.30)
    e = s.compute(g, 'scene_structure', y, stride=1, padding=0)
    e = g.point('scene_structure_maps', e, ramp_readout([[1,1,1,1,0,0], [0,0,0,0,1,1]]))
    e = g.pool('scene_structure_tiles', e, 6)
    # The unpadded convolution leaves an 18 x 18 interior map.
    e = g.pool('scene_structure_density', e, 3)
    e = diag_ramps(g,'scene_structure_strength',e,[.095,.093],[.32,.255])
    # Activity prior: colored clothing on a field is more persuasive than a
    # green field or blue patch by itself. These are weak context votes only.
    clr = g.dense('scene_context_votes', color,
                  [[-.355,-.174,0,.336,.165,.255],
                   [0,0,.61,.52,.40,.34]], [.0,-.60])
    clr = g.relu('scene_context_bonus', g.dense('scene_context_shift', clr,
                 [[1,0],[0,1]], [1,0]))
    clr = g.dense('scene_color_vote', clr, [[1,.34]], -1)
    edges = g.dense('scene_geometry_vote', e, [[-.28,.20]])
    return g.add('scene_vote', clr, edges)


def build(out):
    g = Graph()
    rgb = {d: g.pool('rgb_at_%d'%d, 'image', d) for d in (2,3,4)}
    palettes = {i: face_palette(g, 'palette%d'%i, rgb[i]) for i in (2,3)}
    for d, base in [(4,2),(6,3),(8,2),(12,3)]:
        palettes[d] = g.pool('palette%d'%d, palettes[base], d//base)
    faces, f_all, p_all, joints = {}, [], [], []
    for d, tensor in sorted(palettes.items()):
        f = face_bank(g, 'face%d'%d, tensor)
        faces[d] = f
        f_all.append(scan_summary(g, 'face%d_scan'%d, f, [.53,.56,.55], [.98,1.02,.94], 'face'))
        if d >= 3:
            p = person_bank(g, 'body%d'%d, tensor)
            p_all.append(scan_summary(g, 'body%d_scan'%d, p, [.51,.57,.46,.51], [.96,.99,.95,.92], 'body'))
            if d//2 in faces and d % 2 == 0:
                joints.append(assembly(g, 'assembled%d'%d, faces[d//2], p))
    f = peak_across(g, 'face_pyramid', f_all)
    p = peak_across(g, 'body_pyramid', p_all)
    j = peak_across(g, 'assembly_pyramid', joints)
    j = diag_ramps(g, 'assembly_strength', j, [.21], [.72])
    j_vote = g.dense('assembly_vote', j, [[.38]])

    f = diag_ramps(g, 'face_strengths', f, [.28,.33,.38], [.80,.85,.86])
    p = diag_ramps(g, 'body_strengths', p, [.25,.32,.30,.22], [.78,.80,.82,.77])
    # Compare the two color face variants instead of counting one face twice.
    f_common = g.relu('face_front_margin', g.dense('face_front_delta', f, [[1.18,-.85,0]]))
    f_profile = g.dense('face_profiles_and_geometry', f, [[0,.85,.37]])
    f_vote = g.add('face_vote', f_common, f_profile)
    p_vote = g.dense('body_vote', p, [[.89,.45,.54,.54]])
    c_vote = scenes(g, rgb[4])
    score = g.add('evidence', f_vote, p_vote, j_vote, c_vote)
    g.dense('scores', score, [[0],[1]], [0,-1.25])

    out.mkdir(parents=True, exist_ok=True)
    (out/'network.json').write_text(json.dumps(dict(layers=g.layers), separators=(',',':'))+'\n')
    np.savez(out/'weights.npz', **g.weights)


if __name__ == '__main__':
    build(Path(sys.argv[1]))
