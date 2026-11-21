"""Data-free ASL network: analytic contours and manually specified anatomy.

No source data, learned parameters, optimization, or file reads are used.
Coordinates below are anatomical drawing units: palm width is about one,
palm length about one, and middle-finger length about one. They are not
coordinates measured in photographs. Templates are sampled from implicit
geometric curves, not from any training image or image descriptor.
"""
import json
import sys
from pathlib import Path
import numpy as np

LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'
DIRECTIONS = 8
GRID = 14


class Hand:
    def __init__(self):
        self.parts = []
        self.creases = []
        self.anchors = []

    def bend_wrist(self, angle):
        # Finger direction is not forearm direction. For sideways and
        # downward signs, flexion keeps the forearm entering from below.
        a = np.deg2rad(angle)
        rotation = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        center = np.array([0., .56])
        end = center + np.array([-.40, -1.7]) @ rotation
        start = center + np.array([-.25, -.18]) @ rotation
        self.parts = [p for p in self.parts if p[4]]
        if angle > 120:
            # With Q, the back of the hand is above the fingers; the
            # forearm continues upward instead of hiding the fingertips.
            end = center + np.array([-.15, 1.7]) @ rotation
            start = center + np.array([-.12, .28]) @ rotation
            anchor = center + np.array([-.12, .67]) @ rotation
        self.capsule(end, start, .28, False)
        if angle <= 120:
            anchor = center + np.array([-.33, -.67]) @ rotation
        self.anchors = [tuple(anchor)]

    def capsule(self, a, b, radius, fit=True):
        self.parts.append(('cap', np.array(a, dtype=float), np.array(b, dtype=float), radius, fit))

    def ellipse(self, center, rx, ry, fit=True):
        self.parts.append(('ell', np.array(center, dtype=float), np.array([rx, ry]), 0., fit))

    def chain(self, points, radii, fit=True):
        if np.isscalar(radii):
            radii = [radii] * (len(points)-1)
        for a, b, r in zip(points[:-1], points[1:], radii):
            self.capsule(a, b, r, fit)

    def crease(self, points, strength=.45):
        for a, b in zip(points[:-1], points[1:]):
            self.creases.append((np.array(a, dtype=float), np.array(b, dtype=float), strength))

    def thumb(self, points, radius=.14, seam=.65):
        self.chain(points, [radius * (.92 ** i) for i in range(len(points)-1)])
        # The exposed edge of an overlapping thumb is a weak internal crease.
        p = np.asarray(points)
        tangent = p[-1] - p[0]
        normal = np.array([-tangent[1], tangent[0]])
        normal /= max(np.linalg.norm(normal), 1e-8)
        self.crease(p + normal[None, :] * radius * .7, seam)


def anatomy(letter, variant):
    """An anatomical alphabet, specified independently of the supplied pictures."""
    h = Hand()
    v = float(variant)
    base_angle = 0.
    if letter in 'MN':
        base_angle = -18.
    if letter == 'F':
        base_angle = 12.
    if letter in 'CO':
        # A side-on cupped palm, fingers arcing above, thumb below.
        h.chain([(-.34, -.85), (-.29, .34)], [.29], False)
        h.anchors.append((-.32, -.16))
        h.chain([(-.30, .22), (-.30, .98), (-.19, 1.34)], [.24, .22])
        arch = [(-.19, 1.30), (.04, 1.52), (.36, 1.54), (.61, 1.40)]
        if letter == 'O':
            arch += [(.68, 1.17), (.58, .96)]
            thumb = [(-.23, .35), (.12, .53), (.43, .76), (.58, .96)]
        else:
            arch[-1] = (.64, 1.43 + .06*v)
            thumb = [(-.23, .35), (.15, .48), (.61, .70 + .05*v)]
        h.chain(arch, .17)
        h.thumb(thumb, .145, .30)
        h.crease([(-.12, 1.31), (.10, 1.40), (.36, 1.41)], .22)
        return h, base_angle
    if letter in 'GH':
        # In a side view the extended index is above the folded fingers.
        # Rotating a frontal palm would incorrectly reverse this ordering.
        h.ellipse((-.10, .57), .43, .38)
        h.chain([(-.55, -1.0), (-.22, .42)], .25, False)
        h.anchors.append((-.39, -.32-.20*v))
        h.chain([(-.07, .88), (.67, 1.08), (1.38+.10*v, 1.20)], [.13, .105])
        if letter == 'H':
            h.chain([(-.02, .67), (.67, .86), (1.42+.10*v, .98)], [.13, .11])
        else:
            h.capsule((.12, .63), (.38, .63), .135)
            h.crease([(.12, .77), (.39, .74), (.44, .60)], .55)
        h.capsule((-.12, .44), (.15, .37), .14)
        h.capsule((-.25, .27), (-.01, .19), .12)
        h.crease([(-.23, .53), (.11, .45), (.21, .34)], .55)
        h.crease([(-.31, .35), (-.06, .28), (.02, .17)], .50)
        h.thumb([(-.28, .46), (.12, .59), (.48, .60)], .14, .60)
        return h, 0.
    if letter == 'Q':
        # A downward pointing index and opposed thumb, with the forearm
        # above the dorsum and the other three fingers curled under.
        h.ellipse((-.08, 1.01), .43, .44)
        h.chain([(-.28, 2.5), (-.12, 1.25)], .26, False)
        h.anchors.append((-.18, 1.83+.20*v))
        h.chain([(.24, 1.0), (.40, .35), (.36+.04*v, -.35)], [.13, .105])
        h.thumb([(-.36, 1.0), (-.41, .38), (-.16, -.17)], .14, .4)
        for x in [-.23, -.02, .16]:
            h.capsule((x, .82), (x-.12, .60), .12)
            h.crease([(x+.10, .84), (x+.01, .60), (x-.12, .56)], .5)
        return h, -12.
    if letter == 'P':
        # K's finger arrangement viewed with the palm facing the floor:
        # index forward, middle downward, thumb between them.
        h.ellipse((-.18, .82), .47, .28)
        h.chain([(-1.6, 1.02), (-.42, .88)], .25, False)
        h.anchors.append((-.73, .93))
        h.chain([(.05, .99), (.64, 1.15), (1.20, 1.10+.08*v)], [.135, .105])
        h.chain([(.12, .76), (.59, .61), (.69+.07*v, -.12)], [.14, .11])
        h.capsule((-.18, .59), (.04, .42), .16)
        h.thumb([(-.06, .65), (.36, .48), (.57, .30)], .14)
        h.crease([(-.20, .68), (.02, .61), (.17, .48)], .4)
        return h, base_angle

    palm_width = .52 * (1 + .045*v)
    if letter in 'GHQ':
        palm_width *= .84
    if letter in 'DKRU':
        # Edge-on presentations expose less of the palm's transverse width.
        palm_width *= .88
    h.ellipse((0., .48), palm_width, .55)
    h.chain([(0., -1.50), (0., .23)], .29, False)
    # The crop can include a quarter to nearly a full palm-length of wrist.
    h.anchors.append((0., -.28-.35*(v+1)))
    roots = np.array([[-.37, .99], [-.13, 1.07], [.12, 1.10], [.35, 1.04]])
    if letter in 'DKRU':
        roots[:, 0] *= .84
    if letter == 'D':
        # Curled fingers wrap toward the raised index in the side-on view.
        roots[:3, 0] = roots[:3, 0]*.72+.10
        roots[:3, 1] -= .10
    lengths = np.array([.83, 1.06, 1.17, 1.04]) * (1 + .12*v)
    radii = [.12, .13, .134, .126]
    extended = {}
    if letter == 'B': extended = {0: -.06, 1: -.025, 2: .02, 3: .06}
    if letter == 'D': extended = {3: .015}
    if letter == 'F': extended = {0: -.27, 1: -.12, 2: .045}
    if letter == 'G': extended = {3: -.02}
    if letter == 'H': extended = {2: -.025, 3: .015}
    if letter == 'I': extended = {0: -.10}
    if letter == 'K': extended = {2: -.12, 3: .15}
    if letter == 'L': extended = {3: .015}
    if letter == 'Q': extended = {3: -.04}
    if letter == 'R': extended = {2: .16, 3: -.10}
    if letter == 'U': extended = {2: -.015, 3: .015}
    if letter == 'V': extended = {2: -.25, 3: .25}
    if letter == 'W': extended = {1: -.26, 2: .01, 3: .28}
    if letter == 'Y': extended = {0: -.27}
    special = {3} if letter in 'FX' else set()
    for i in range(4):
        x, y = roots[i]
        r = radii[i]
        if i in extended:
            lean = extended[i] * (1 + .12*v)
            length = lengths[i]
            # Two phalangeal segments, tapering toward a rounded fingertip.
            p = [(x, y-.05), (x+.47*lean, y+.53*length), (x+lean, y+length)]
            h.chain(p, [r, .87*r])
            for f in [.38, .72]:
                cx, cy = x+lean*f, y+length*f
                h.crease([(cx-.62*r, cy), (cx+.62*r, cy-.015)], .16)
        elif i not in special:
            # Folded finger tips lie against the upper palm. Individual
            # lower edges remain visible even where the outer masks merge.
            low = .65 + .08*v
            high = y + .055
            dx = -.025*v
            if letter == 'E':
                low += .16
                high += .14
            if letter == 'M' and i in (1, 2, 3):
                low -= .17
                dx = -.12
            if letter == 'N' and i in (2, 3):
                low -= .16
                dx = -.12
            h.capsule((x, high), (x+dx, low), r)
            h.crease([(x-.8*r, high), (x+dx-.8*r, low),
                      (x+dx, low-.65*r), (x+dx+.8*r, low),
                      (x+.8*r, high)], .52)
    # Weak grooves where straight adjacent fingers touch.
    for i in range(3):
        if i in extended and i+1 in extended:
            if abs(extended[i+1]-extended[i]) < .12:
                x = (roots[i, 0]+roots[i+1, 0])*.5
                top = min(roots[i, 1]+lengths[i], roots[i+1, 1]+lengths[i+1])-.10
                h.crease([(x, 1.08), (x, top)], .48)
    if letter == 'A':
        h.thumb([(.43, .30), (.61, .72), (.56, 1.12+.03*v)], .14, .48)
    elif letter == 'B':
        h.thumb([(.46, .37), (.17, .58), (-.18, .64)], .135)
    elif letter == 'D':
        h.thumb([(.46, .35), (.60, .71), (.40, 1.02), (.15, .98)], .13, .50)
    elif letter == 'E':
        h.thumb([(.45, .43), (.18, .61), (-.21, .62)], .135)
    elif letter == 'F':
        h.chain([(.35, 1.04), (.60, 1.25), (.82, 1.10), (.82, .89), (.62, .74)], [.12, .11, .10, .10])
        h.thumb([(.46, .34), (.70, .56), (.62, .74)], .13, .35)
    elif letter in 'GQ':
        if letter == 'G':
            # Viewed side-on, the thumb is foreshortened and tucked below
            # the index, not a second full-length parallel finger.
            h.thumb([(.40, .34), (.65, .72), (.61, 1.07)], .13, .38)
        else:
            h.thumb([(.40, .34), (.64, .91), (.66+.03*v, 1.82)], .13, .38)
        base_angle = 73. if letter == 'G' else 164.
    elif letter == 'H':
        h.thumb([(.43, .33), (.25, .70), (-.04, .81)], .14)
        base_angle = 77.
    elif letter == 'K':
        h.thumb([(.46, .35), (.39, .93), (.20, 1.54+.05*v)], .14)
    elif letter == 'L':
        h.thumb([(.43, .32), (.83, .60), (1.20, .74+.04*v)], .14, .35)
    elif letter == 'M':
        h.thumb([(.45, .41), (.02, .68), (-.25, .67)], .12, .32)
    elif letter == 'N':
        h.thumb([(.45, .40), (.16, .69), (-.04, .76)], .12, .32)
    elif letter == 'T':
        h.thumb([(.42, .40), (.23, .86), (.23, 1.25+.04*v)], .13, .50)
    elif letter == 'X':
        h.chain([(.35, 1.02), (.40, 1.48), (.62, 1.61), (.77, 1.43-.04*v)], [.12, .115, .10])
        h.thumb([(.45, .41), (.16, .75), (-.12, .77)], .14)
    elif letter == 'Y':
        h.thumb([(.43, .32), (.82, .65), (1.13, 1.03+.06*v)], .14, .35)
    else:
        # S and the upright finger signs use an opposed, tucked thumb.
        h.thumb([(.46, .43), (.18, .86), (-.14, .81)], .14)
    # A faint thenar crease is common anatomy, not class-specific evidence.
    h.crease([(.35, .23), (.21, .43), (.23, .62)], .16)
    return h, base_angle


def frame(hand, angle, aspect, scale, mirror):
    """Project anatomical geometry into a nominal square hand crop."""
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    transform = np.diag([mirror, -1.]) @ rotation @ np.diag([aspect, 1.])
    boundary = []
    circle = np.stack([np.cos(np.arange(24)*np.pi/12), np.sin(np.arange(24)*np.pi/12)], axis=1)
    for kind, a, b, r, fit in hand.parts:
        if not fit:
            continue
        if kind == 'cap':
            boundary.extend(a + r*circle)
            boundary.extend(b + r*circle)
        else:
            boundary.extend(a + b[None, :]*circle)
    boundary.extend(hand.anchors)
    p = np.asarray(boundary) @ transform.T
    lo, hi = p.min(0), p.max(0)
    factor = 23.5 * scale / np.max(hi-lo)
    A = factor * transform
    center = (lo+hi)/2
    offset = np.array([13.5, 13.5]) - factor*center
    return A, offset


def capsule_field(q, a, b):
    v = b-a
    f = np.clip(((q-a)*v).sum(-1)/(np.dot(v, v)+1e-12), 0., 1.)
    delta = q - (a+f[..., None]*v)
    length = np.sqrt((delta*delta).sum(-1)+1e-12)
    return length, delta/length[..., None]


def geometry_fields(hand, A, offset):
    """Sample analytic signed distances and contour normals on the feature grid.

    This is direct evaluation of the anatomical equations. No images are
    loaded or passed through the network to construct these coefficients.
    """
    y, x = np.mgrid[0:GRID, 0:GRID]
    xy = np.stack([.5+2*x, .5+2*y], axis=-1)
    inv = np.linalg.inv(A)
    q = (xy-offset) @ inv.T
    dist = np.full((GRID, GRID), 1e8)
    normal = np.zeros((GRID, GRID, 2))
    for kind, a, b, r, _ in hand.parts:
        if kind == 'cap':
            length, n = capsule_field(q, a, b)
            d = length-r
        else:
            z = (q-a)/b
            rho = np.sqrt((z*z).sum(-1)+1e-12)
            g = (q-a)/(b*b)
            gn = np.sqrt((g*g).sum(-1)+1e-12)
            n = g/gn[..., None]
            d = (rho-1.)*rho/gn
        npix = n @ inv
        length = np.sqrt((npix*npix).sum(-1)+1e-12)
        dpix = d/length
        npix /= length[..., None]
        select = dpix < dist
        dist = np.where(select, dpix, dist)
        normal = np.where(select[..., None], npix, normal)
    angles = np.arange(DIRECTIONS)*np.pi/DIRECTIONS
    axes = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    perpendicular = np.stack([-np.sin(angles), np.cos(angles)], axis=-1)
    def oriented(n, strength):
        along = np.abs(np.einsum('hwc,dc->dhw', n, axes))
        across = np.abs(np.einsum('hwc,dc->dhw', n, perpendicular))
        return np.maximum(along-.65*across, 0.)*strength[None]
    # Max pooling makes a one-pixel plateau around each ideal contour;
    # Gaussian derivative filters give the smooth shoulders outside it.
    magnitude = np.exp(-.5*(np.maximum(np.abs(dist)-.35, 0.)/1.15)**2)
    edge = oriented(normal, magnitude)
    for a, b, strength in hand.creases:
        # A crease is an unsigned line, so its expected gradient is normal
        # to its tangent, irrespective of illumination polarity.
        ap, bp = A@a+offset, A@b+offset
        length, _ = capsule_field(xy, ap, bp)
        tangent = bp-ap
        n = np.array([-tangent[1], tangent[0]])
        n /= max(np.linalg.norm(n), 1e-10)
        nn = np.broadcast_to(n, (GRID, GRID, 2))
        response = strength*np.exp(-.5*(np.maximum(length-.30, 0.)/.95)**2)
        response *= (dist < 1.)
        edge = np.maximum(edge, oriented(nn, response))
    # The facing sides of extended fingers often leave a visible groove
    # even when their silhouettes overlap. Include these analytic seams.
    for kind, a, b, r, _ in hand.parts:
        if kind != 'cap' or min(a[1], b[1]) < .9:
            continue
        length, n = capsule_field(q, a, b)
        dp = length-r
        nn = n @ inv
        nl = np.sqrt((nn*nn).sum(-1)+1e-12)
        dp /= nl
        nn /= nl[..., None]
        response = .55*np.exp(-.5*(np.maximum(np.abs(dp)-.30, 0.)/.95)**2)
        response *= dist < 1.
        edge = np.maximum(edge, oriented(nn, response))
    # Low-frequency occupancy supplies a weak, polarity-free companion cue.
    silhouette = 1./(1.+np.exp(np.clip(dist/1.05, -40, 40)))
    # The lowest strip frequently contains a sleeve or an occluded wrist;
    # taper its contribution rather than treating it as a finger boundary.
    reliability = np.clip((27.-xy[..., 1])/7., .10, 1.)
    edge *= reliability[None]
    edge -= edge.mean(axis=(1, 2), keepdims=True)
    edge /= np.sqrt((edge*edge).mean(axis=(1, 2), keepdims=True)+.015)
    edge /= DIRECTIONS*GRID*GRID
    silhouette -= silhouette.mean()
    silhouette /= np.sqrt((silhouette*silhouette).mean()+.01)
    silhouette /= GRID*GRID
    return edge.astype(np.float32), silhouette.astype(np.float32)


def build(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    layers, weights = [], {}
    def layer(name, op, src, **settings):
        layers.append(dict(name=name, op=op, inputs=[src] if isinstance(src, str) else src, **settings))
        return name
    def conv(name, src, W, b=None, padding=0, stride=1):
        W = np.asarray(W, dtype=np.float32)
        weights[name+'.W'] = W
        weights[name+'.b'] = np.zeros(W.shape[0], dtype=np.float32) if b is None else np.asarray(b, dtype=np.float32)
        return layer(name, 'conv', src, out=W.shape[0], kernel=list(W.shape[2:]), padding=padding, stride=stride)

    # Replicated boundary values avoid detecting the square image border.
    pad = np.zeros((32*32, 28*28), dtype=np.float32)
    for y in range(32):
        for x in range(32):
            pad[y*32+x, np.clip(y-2, 0, 27)*28+np.clip(x-2, 0, 27)] = 1.
    weights['replicate.W'], weights['replicate.b'] = pad, np.zeros(32*32, np.float32)
    layer('replicate', 'dense', 'image', out=32*32)
    layer('padded', 'reshape', 'replicate', shape=[1, 32, 32])
    y, x = np.mgrid[-2:3, -2:3]
    g = np.exp(-(x*x+y*y)/(2*.85**2))
    gx, gy = x*g, y*g
    gx /= np.abs(gx).sum()/4
    gy /= np.abs(gy).sum()/4
    kernels = []
    for k in range(DIRECTIONS):
        a = k*np.pi/DIRECTIONS
        derivative = np.cos(a)*gx+np.sin(a)*gy
        kernels.extend([derivative, -derivative])
    conv('derivatives', 'padded', np.asarray(kernels)[:, None])
    layer('half_wave', 'relu', 'derivatives')
    absolute = np.zeros((DIRECTIONS, 2*DIRECTIONS, 1, 1), np.float32)
    for k in range(DIRECTIONS): absolute[k, 2*k:2*k+2, 0, 0] = 1.
    conv('unsigned', 'half_wave', absolute)
    sharpen = np.zeros((DIRECTIONS, DIRECTIONS, 1, 1), np.float32)
    for k in range(DIRECTIONS):
        sharpen[k, k, 0, 0] = 1.
        sharpen[k, (k+DIRECTIONS//2)%DIRECTIONS, 0, 0] = -.65
    conv('orientation_select', 'unsigned', sharpen)
    layer('edges', 'relu', 'orientation_select')
    layer('local_tolerance', 'avgpool', 'edges', size=[2, 2], stride=[2, 2])
    layer('normalized_edges', 'norm', 'local_tolerance')
    layer('coarse_image', 'avgpool', 'image', size=[2, 2], stride=[2, 2])
    layer('normalized_intensity', 'norm', 'coarse_image')

    # A fixed border reference provides polarity-independent foreground
    # evidence. It subtracts a mean computed at inference time, not a
    # training-set statistic. The lower border is excluded for the wrist.
    border = np.zeros((GRID, GRID), np.float32)
    border[:2, :] = 1.
    border[:GRID//2, :2] = 1.
    border[:GRID//2, -2:] = 1.
    border /= border.sum()
    reference = np.eye(GRID*GRID, dtype=np.float32)-border.reshape(1, -1)
    weights['border_difference.W'] = np.concatenate([reference, -reference], axis=0)
    weights['border_difference.b'] = np.zeros(2*GRID*GRID, np.float32)
    layer('border_difference', 'dense', 'coarse_image', out=2*GRID*GRID)
    layer('border_halves', 'relu', 'border_difference')
    layer('border_channels', 'reshape', 'border_halves', shape=[2, GRID, GRID])
    conv('border_contrast', 'border_channels', np.ones((1, 2, 1, 1), np.float32))
    # Saturation stops a dark sleeve or a bright patch from dominating.
    conv('clip_excess', 'border_contrast', np.ones((1, 1, 1, 1), np.float32), [-.22])
    layer('positive_excess', 'relu', 'clip_excess')
    conv('negative_excess', 'positive_excess', -np.ones((1, 1, 1, 1), np.float32))
    layer('foreground', 'add', ['border_contrast', 'negative_excess'])
    layer('normalized_foreground', 'norm', 'foreground')

    edge_templates, intensity_templates = [], []
    count = None
    # Every letter receives the same fixed nuisance-variation budget.
    rng = np.random.default_rng(28024)
    nuisance = rng.uniform(-1., 1., (420, 7))
    for letter in LETTERS:
        start = len(edge_templates)
        for v, aspect, scale, angle, mx, dx, dy in nuisance:
            hand, base = anatomy(letter, v)
            A, offset = frame(hand, base+32.*angle, 1.+.30*aspect,
                              .99+.19*scale, -1. if mx < 0 else 1.)
            offset += np.array([dx, dy])*.9
            edge, occupancy = geometry_fields(hand, A, offset)
            edge_templates.append(edge)
            intensity_templates.append(occupancy)
        n = len(edge_templates)-start
        if count is None: count = n
        assert count == n
    bank = len(edge_templates)
    conv('contour_match', 'normalized_edges', .8*np.asarray(edge_templates), padding=1)
    # Adjacent output channels are the two possible foreground polarities.
    intensity = np.asarray(intensity_templates)
    signed = np.stack([intensity, -intensity], axis=1).reshape(2*bank, 1, GRID, GRID)
    conv('occupancy_match', 'normalized_intensity', .45*signed, padding=1)
    layer('polarity_axis', 'reshape', 'occupancy_match', shape=[bank, 2, 9])
    layer('choose_polarity', 'maxpool', 'polarity_axis', size=[2, 1], stride=[2, 1])
    layer('occupancy_scores', 'reshape', 'choose_polarity', shape=[bank, 3, 3])
    conv('foreground_match', 'normalized_foreground', .55*intensity[:, None], padding=1)
    layer('joint_match', 'add', ['contour_match', 'occupancy_scores', 'foreground_match'])
    layer('class_hypotheses', 'reshape', 'joint_match', shape=[24, count, 9])
    layer('best_explanation', 'maxpool', 'class_hypotheses', size=[count, 9], stride=[count, 9])
    layer('scores', 'reshape', 'best_explanation', shape=[24])
    (destination/'network.json').write_text(json.dumps({'layers': layers}, indent=2)+'\n')
    np.savez(destination/'weights.npz', **weights)
    print('Built', bank, 'analytic hand hypotheses;', sum(a.size for a in weights.values()), 'weights.')


if __name__ == '__main__':
    build(sys.argv[1])
