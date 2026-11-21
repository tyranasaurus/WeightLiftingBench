"""Build a data-independent, analytically specified person-presence network.

Only standard Python and numpy are needed. No files are read. All colour rules,
shape geometry, filter normalisations, and decision thresholds below are fixed
from general knowledge, not estimated from any pictures or labels.
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
        self.layers.append(dict(name=name, op=op,
                                inputs=source if isinstance(source, list) else [source],
                                **settings))
        return name

    def conv(self, name, source, w, b=None, padding=0, stride=1):
        w = np.asarray(w, dtype=np.float32)
        if b is None:
            b = np.zeros(w.shape[0], dtype=np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.asarray(b, dtype=np.float32)
        return self.op(name, 'conv', source, out=w.shape[0],
                       kernel=list(w.shape[-2:]), stride=stride, padding=padding)

    def dense(self, name, source, w, b=None):
        w = np.asarray(w, dtype=np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = (np.zeros(w.shape[0], dtype=np.float32)
                                     if b is None else np.asarray(b, dtype=np.float32))
        return self.op(name, 'dense', source, out=w.shape[0])

    def relu(self, name, source):
        return self.op(name, 'relu', source)

    def pool(self, name, source, size, maximum=False, stride=None):
        if isinstance(size, int):
            size = [size, size]
        return self.op(name, 'maxpool' if maximum else 'avgpool', source,
                       size=size, stride=size if stride is None else stride)


def unit(a):
    a = np.asarray(a, dtype=np.float64)
    return a / max(float(a.sum()), 1e-12)


def bell(x, y, cx, cy, sx, sy):
    return np.exp(-0.5 * (((x-cx)/sx)**2 + ((y-cy)/sy)**2))


def smooth_inside(signed_distance, softness=0.40):
    return 1.0 / (1.0 + np.exp(np.clip(-signed_distance / softness, -60, 60)))


def ellipse(x, y, cx, cy, rx, ry):
    q = np.sqrt(((x-cx)/rx)**2 + ((y-cy)/ry)**2 + 1e-12)
    return smooth_inside((1-q) * min(rx, ry))


def capsule(x, y, ax, ay, bx, by, radius):
    dx, dy = bx-ax, by-ay
    t = np.clip(((x-ax)*dx + (y-ay)*dy) / (dx*dx + dy*dy + 1e-12), 0, 1)
    distance = np.sqrt((x-ax-t*dx)**2 + (y-ay-t*dy)**2)
    return smooth_inside(radius-distance, .32)


K = 15
Y, X = np.mgrid[-K//2+1:K//2+1, -K//2+1:K//2+1].astype(np.float64)


FEATURE_CHANNELS = 10


def skin_feature(core, ring=None, strict_fraction=.35):
    w = np.zeros((FEATURE_CHANNELS, K, K), dtype=np.float64)
    w[0] = (1-strict_fraction)*unit(core)
    w[1] = strict_fraction*unit(core)
    if ring is not None:
        w[0] -= unit(ring)
    return w


def warm_feature(core, ring=None):
    w = np.zeros((FEATURE_CHANNELS, K, K), dtype=np.float64)
    w[9] = unit(core)
    if ring is not None:
        w[9] -= unit(ring)
    return w


def luma_feature(positive, negative):
    w = np.zeros((FEATURE_CHANNELS, K, K), dtype=np.float64)
    f = unit(positive)-unit(negative)
    w[2], w[3] = f, -f
    return w


def contour_feature(occupancy, region=None, interior_penalty=.03):
    """A zero-orientation-mean, four-direction contour matched filter.

    At each model contour point, reward a measured edge normal parallel to the
    model normal and penalise perpendicular normals. Isotropic texture thus
    has zero expected reward, unlike a mere edge-energy detector.
    """
    gy, gx = np.gradient(occupancy)
    mag = np.sqrt(gx*gx + gy*gy)
    nx, ny = gx/(mag+1e-12), gy/(mag+1e-12)
    v = np.stack([np.abs(nx), np.abs(ny),
                  np.abs(nx+ny)/math.sqrt(2),
                  np.abs(nx-ny)/math.sqrt(2)])
    centered = v-v.mean(axis=0, keepdims=True)
    denom = np.sum(centered*v, axis=0)
    orient = centered / np.maximum(denom, .1)[None]
    support = mag if region is None else mag*region
    support = unit(support)
    w = np.zeros((FEATURE_CHANNELS, K, K), dtype=np.float64)
    w[4:8] = orient*support[None]
    if interior_penalty:
        inside = unit(occupancy if region is None else occupancy*region)
        w[4:8] -= interior_penalty*inside[None]/4
    # The constant channel is one inside the image and zero in padding. A
    # missing part of a contour is penalised rather than interpreted as black.
    w[8] = support
    return w


def make_head(width, aspect, cy, upper=False):
    rx, ry = width/2, width*aspect/2
    q = ((X/(rx*.78))**2 + ((Y-cy-.14*ry)/(ry*.78))**2)
    core = np.exp(-1.6*q) * (q < 1.65)
    outer = (X/rx)**2 + ((Y-cy)/ry)**2
    ring = ((outer > 1.25**2) & (outer < 1.95**2)).astype(float)
    # The neck is allowed to continue out of the bottom of a skin-coloured head.
    ring *= 1-((Y > cy+.65*ry) & (np.abs(X) < .60*rx))
    head = ellipse(X, Y, 0, cy, rx, ry)
    eye_y = cy-.18*ry
    eyes = []
    eye_brights = []
    for sign in [-1, 1]:
        ex = sign*.40*rx
        eyes.append(bell(X,Y,ex,eye_y,max(.23*rx,.35),max(.11*ry,.30)))
        eye_brights.append(
            .5*unit(bell(X,Y,ex,cy-.62*ry,max(.27*rx,.42),max(.12*ry,.32)))+
            .5*unit(bell(X,Y,ex,cy+.22*ry,max(.29*rx,.42),max(.14*ry,.33))))
    eye_pair = unit(eyes[0])+unit(eyes[1])
    light_pair = eye_brights[0]+eye_brights[1]
    nose = bell(X,Y,0,cy+.02*ry,max(.17*rx,.34),max(.25*ry,.4))
    mouth = bell(X,Y,0,cy+.52*ry,max(.38*rx,.45),max(.09*ry,.30))
    mouth_light = (
        unit(bell(X,Y,0,cy+.26*ry,max(.45*rx,.5),max(.12*ry,.32)))+
        unit(bell(X,Y,0,cy+.81*ry,max(.36*rx,.45),max(.11*ry,.30))))
    f = {
        'skin': skin_feature(core),
        'contrast': skin_feature(core, ring, strict_fraction=.20),
        'warm': warm_feature(core),
        'warm_contrast': warm_feature(core,ring),
        'warm_left': warm_feature(core,ring*(X < -.3*rx)),
        'warm_right': warm_feature(core,ring*(X > .3*rx)),
        'warm_top': warm_feature(core,ring*(Y < cy-.65*ry)),
        'skin_left': skin_feature(core,ring*(X < -.3*rx)),
        'skin_right': skin_feature(core,ring*(X > .3*rx)),
        'eye': luma_feature(light_pair, eye_pair),
        'left_eye': luma_feature(eye_brights[0], eyes[0]),
        'right_eye': luma_feature(eye_brights[1], eyes[1]),
        'nose': luma_feature(nose, eye_pair),
        'mouth': luma_feature(mouth_light, mouth),
        'head': contour_feature(head),
    }
    # Non-oriented edge energy below a head: useful for clothed shoulders even
    # when no particular arm pose or torso outline matches.
    torso_region = bell(X,Y,0,cy+2.5*ry,1.15*width,.65*width)
    torso_feature = np.zeros((FEATURE_CHANNELS,K,K))
    torso_feature[4:8] = unit(torso_region)[None]/4
    f['body_texture'] = torso_feature
    if upper:
        neck_bottom = cy+ry+.7
        half_width = 1.14*width
        top = neck_bottom + .28*np.abs(X) + .045*X*X
        side = half_width - .055*(Y-neck_bottom)-np.abs(X)
        torso = smooth_inside(Y-top)*smooth_inside(side)
        neck = capsule(X,Y,0,cy+ry-.4,0,neck_bottom+.6,.31*width)
        shoulder = np.maximum(torso, neck)
        region = smooth_inside(Y-(cy+ry-.20), .3)
        f['shoulder'] = contour_feature(shoulder, region)
    return f


def make_body(pose, stretch):
    x, y = X/stretch, Y
    head = ellipse(x,y,0,-4.65,1.02,1.28)
    torso = capsule(x,y,0,-2.7,0,-.15,1.37)
    neck = capsule(x,y,0,-3.55,0,-2.45,.47)
    if pose == 'standing':
        arms = [(-1.2,-2.55,-2.1,.35), (1.2,-2.55,2.1,.35)]
        legs = [(-.64,.25,-.92,5.3), (.64,.25,.92,5.3)]
    elif pose == 'walking':
        arms = [(-1.2,-2.55,-2.6,-.65), (1.2,-2.55,2.4,.5)]
        legs = [(-.64,.25,-2.3,5.0), (.64,.25,2.3,4.6)]
    else:
        arms = [(-1.2,-2.55,-3.4,-2.0), (1.2,-2.55,3.4,-2.0)]
        legs = [(-.64,.25,-1.7,5.0), (.64,.25,1.7,5.0)]
    body = np.maximum(torso,neck)
    for ax,ay,bx,by in arms:
        body = np.maximum(body,capsule(x,y,ax,ay,bx,by,.43))
    for ax,ay,bx,by in legs:
        body = np.maximum(body,capsule(x,y,ax,ay,bx,by,.56))
    silhouette = np.maximum(head,body)
    core = bell(x,y,0,-4.47,.68,.82)
    ring = (((x/1.9)**2+((y+4.65)/2.0)**2 < 1) &
            ((x/1.25)**2+((y+4.65)/1.55)**2 > 1)).astype(float)
    return {
        'skin': skin_feature(core),
        'head': contour_feature(silhouette, smooth_inside(-3.1-y,.25)),
        'upper': contour_feature(silhouette, smooth_inside(y+3.2,.25)*smooth_inside(.6-y,.25)),
        'lower': contour_feature(silhouette, smooth_inside(y-.55,.3)),
        'contrast': skin_feature(core, ring, strict_fraction=.2),
    }


def make_colour(net):
    image = net.pool('antialias_half', 'image', 2)
    # A small convex skin-colour cone, expressed as soft linear inequalities.
    # Each entry is (RGB coefficients, constant, transition width), with a
    # positive value meaning a violation. No colour histogram is estimated.
    broad = [
        ((-1,1,0),.015,.05),
        ((-1,0,1),.045,.07),
        ((0,-1,1),-.035,.09),
        ((1,-1.85,0),-.025,.12),
        ((-1.5,2.5,-1),-.025,.10),
        ((.48,0,-1),-.025,.09),
        ((-1,0,0),.12,.10),
        ((1,-1,0),-.38,.12),
    ]
    strict = [
        ((-1,1,0),.020,.045),
        ((-1,0,1),.045,.065),
        ((0,-1,1),-.030,.08),
        ((1,-1.70,0),-.020,.11),
        ((-.95,1.95,-1),-.025,.10),
        ((.62,0,-1),-.020,.09),
        ((-1,0,0),.16,.12),
    ]
    # A second, broader cone covers desaturated, dark, and shaded skin. It
    # never triggers alone: the shape/contrast rules must also agree.
    warm = [
        ((-1,1,0),.015,.06),
        ((-1,0,1),.040,.08),
        ((0,-1,1),-.045,.10),
        ((1,-2.0,0),-.035,.15),
        ((-1.7,2.7,-1),-.035,.12),
        ((.42,0,-1),-.035,.12),
        ((-1,0,0),.09,.12),
    ]
    specs = broad+strict+warm
    w = np.zeros((len(specs)+3,3,1,1))
    b = np.zeros(len(specs)+3)
    for i,(a,c,s) in enumerate(specs):
        w[i,:,0,0] = np.array(a)/s
        b[i] = c/s
    for c in range(3):
        w[len(specs)+c,c,0,0] = 1
    t = net.conv('colour_inequalities',image,w,b)
    t = net.relu('colour_violations',t)
    w = np.zeros((4,len(specs)+3,1,1))
    w[0,:len(broad),0,0] = -1
    w[1,len(broad):len(broad)+len(strict),0,0] = -1
    w[2,len(specs):,0,0] = [.299,.587,.114]
    w[3,len(broad)+len(strict):len(specs),0,0] = -1
    t = net.conv('skin_and_luminance',t,w,[1,1,0,1])
    return net.relu('bounded_skin_and_luminance',t)


def make_low_features(net, base, tag):
    n = net.op(tag+'norm','norm',base)
    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=float)/8
    sy = sx.T
    directions = [sx,sy,(sx+sy)/math.sqrt(2),(sx-sy)/math.sqrt(2)]
    w = np.zeros((10,4,3,3))
    w[0,2,1,1], w[1,2,1,1] = 1,-1
    for i,d in enumerate(directions):
        w[2+2*i,2] = d
        w[3+2*i,2] = -d
    d = net.conv(tag+'signed_derivatives',n,w,padding=1)
    d = net.relu(tag+'half_wave',d)
    w = np.zeros((FEATURE_CHANNELS,10,1,1))
    w[2,0,0,0],w[3,1,0,0] = 1,1
    for i in range(4):
        w[4+i,2+2*i:4+2*i,0,0] = 1
    edges = net.conv(tag+'unsigned_edges',d,w)
    w = np.zeros((FEATURE_CHANNELS,4,1,1)); w[0,0,0,0]=1; w[1,1,0,0]=1; w[9,3,0,0]=1
    b = np.zeros(FEATURE_CHANNELS); b[8] = 1
    colour = net.conv(tag+'skin_channels',base,w,b)
    return net.op(tag+'features','add',[edges,colour])


def make_templates():
    filters = []
    rules = []
    offsets = {}

    def features(fs):
        result = {}
        for key, value in fs.items():
            # Colour contrast also requires real context: absent pixels outside
            # the image must not count as a dark or non-skin surround.
            if key.startswith(('skin','warm','contrast')):
                support = np.sum(np.abs(value[[0,1,9]]),axis=0)
                if support.sum() > 0:
                    value[8] += .55*unit(support)
            result[key] = len(filters)
            offsets[len(filters)] = float(value[8].sum())
            filters.append(value)
        return result

    def rule(family, ids, requirements, decision=.76):
        # A fuzzy AND: sum deficits of independently specified part tests.
        adjusted = [(ids[k],threshold,offsets[ids[k]]) for k,threshold in requirements]
        rules.append((family,adjusted,decision))

    heads = [
        (2.5,1.35,-2.2,True),
        (3.5,1.35,-1.7,True),
        (3.8,1.20,0,False),
        (4.8,1.50,0,False),
        (5.8,1.22,0,False),
        (7.8,1.20,0,False),
        (6.7,1.55,0,False),
    ]
    for width,aspect,cy,upper in heads:
        ids = features(make_head(width,aspect,cy,upper))
        if width >= 3.5:
            rule('skin_face',ids,[('skin',.68),('contrast',.38),('eye',.50),
                                  ('nose',.24),('head',.28)],.89)
            rule('gray_face',ids,[('eye',.82),('left_eye',.55),('right_eye',.55),
                                  ('nose',.40),('mouth',.30),('head',.40)],.96)
        if upper:
            rule('skin_upper',ids,[('skin',.68),('contrast',.40),
                                   ('head',.28),('shoulder',.30)],.92)
        rule('compact_skin_head',ids,[('skin',.72),('skin_left',.52),
                                      ('skin_right',.52),('warm_top',.40),
                                      ('body_texture',.24),('head',.20)],.92)
        rule('warm_face',ids,[('warm',.76),('warm_left',.48),('warm_right',.48),
                              ('eye',.44),('nose',.22),('head',.18)],.92)
        if upper:
            rule('warm_shoulders',ids,[('warm',.74),('warm_left',.48),
                                       ('warm_right',.48),('warm_top',.40),
                                       ('shoulder',.24)],.92)

    for pose,stretch in [('standing',.82),('standing',1.15),('walking',1.0),('arms',1.0)]:
        ids = features(make_body(pose,stretch))
        rule('skin_body',ids,[('skin',.58),('contrast',.24),('head',.30),
                              ('upper',.30),('lower',.28)],.94)
        rule('gray_body',ids,[('head',.70),('upper',.60),('lower',.50)],.98)
    return np.asarray(filters),rules


def soft_skin_evidence(net,base):
    # Maps for overall skin prevalence, pink-versus-yellow balance, and
    # skin-region contrast. Unlike an exact silhouette, these tolerate crops,
    # sitting poses, and multiple people.
    w = np.zeros((5,4,1,1))
    w[0,:,0,0] = [.55,.45,0,0]
    w[1,:,0,0] = [-.55,-.45,0,0]
    w[2,1,0,0] = 1
    w[3,0,0,0] = 1
    w[4,:,0,0] = [.35,.65,0,0]
    maps = net.conv('soft_colour_maps',base,w,[0,1,0,0,0])
    kernel = 7
    yy,xx = np.mgrid[-3:4,-3:4].astype(float)
    center = unit(np.exp(-.5*(xx*xx+yy*yy)/.85**2))
    ring = unit((((xx*xx+yy*yy)>=2.2**2)&((xx*xx+yy*yy)<=3.5**2)).astype(float))
    w = np.zeros((1,5,kernel,kernel))
    w[0,4] = center-ring
    contrast = net.conv('local_skin_contrast',maps,w,padding=3)
    contrast = net.relu('positive_skin_contrast',contrast)
    contrast_mean = net.pool('average_skin_contrast',contrast,48)
    contrast_vector = net.dense('contrast_descriptor',contrast_mean,
                                [[0],[0],[0],[1]])
    # A smooth occupancy saturates rather than rewarding an entire orange wall.
    w = np.zeros((2,5,1,1)); w[0,0,0,0]=1; w[1,2,0,0]=1
    excess = net.conv('skin_above_occupancy_knee',maps,w,[-.30,-.30])
    excess = net.relu('occupancy_excess',excess)
    pooled = net.pool('mean_soft_colour',maps,48)
    overflow = net.pool('mean_occupancy_excess',excess,48)
    w = np.zeros((4,5)); w[0,0]=1; w[1,2]=1; w[2,3]=1
    means = net.dense('raw_skin_descriptors',pooled,w)
    w = np.zeros((4,2)); w[0,0]=-1; w[1,1]=-1
    subtract = net.dense('occupancy_cap_subtraction',overflow,w)
    desc = net.op('skin_descriptors','add',[means,subtract,contrast_vector])
    # This branch expresses a prior: modest amounts of concentrated,
    # desaturated warm colour are more plausible skin than a yellow field.
    return net.dense('soft_skin_score',desc,[[0,0,0,0],[2.7,2.5,-.30,1.3]],
                     [0,-.16])


def build(destination):
    net = Network()
    base = make_colour(net)
    filters,rules = make_templates()
    gate_w = []
    gate_b = []
    ranges = []
    for family,requirements,decision in rules:
        start = len(gate_w)
        for index,threshold,offset in requirements:
            row = np.zeros(len(filters))
            row[index] = -1/threshold
            gate_w.append(row)
            gate_b.append(1+offset/threshold)
        ranges.append((start,len(gate_w),decision))
    gate_w = np.asarray(gate_w)[:,:,None,None]
    rule_w = np.zeros((len(rules),len(gate_w),1,1))
    rule_b = np.zeros(len(rules))
    for i,(start,end,decision) in enumerate(ranges):
        rule_w[i,start:end,0,0] = -1
        rule_b[i] = 1-decision

    factors = [1,2,3,4,6]
    scale_vectors = []
    for si,factor in enumerate(factors):
        tag = 's%d_'%factor
        bs = base if factor == 1 else net.pool(tag+'pyramid',base,factor)
        low = make_low_features(net,bs,tag)
        raw = net.conv(tag+'analytic_templates',low,filters,padding=K//2)
        gates = net.conv(tag+'part_deficits',raw,gate_w,gate_b)
        gates = net.relu(tag+'positive_deficits',gates)
        scores = net.conv(tag+'rule_scores',gates,rule_w,rule_b)
        h = 48//factor
        scores = net.pool(tag+'location_max',scores,[h,h],maximum=True)
        # Place each scale in its own slot, without concatenation or sharing
        # any information between unrelated locations.
        w = np.zeros((len(factors)*len(rules),len(rules)))
        w[si*len(rules):(si+1)*len(rules)] = np.eye(len(rules))
        scale_vectors.append(net.dense(tag+'scale_slots',scores,w))

    all_scores = net.op('all_scale_slots','add',scale_vectors)
    positive = net.relu('positive_person_evidence',all_scores)
    w = np.zeros((len(factors),len(factors)*len(rules)))
    for si in range(len(factors)):
        w[si,si*len(rules):(si+1)*len(rules)] = 1
    support = net.dense('evidence_per_scale',positive,w)
    overflow = net.dense('evidence_over_cap',support,np.eye(len(factors)),[-.12]*len(factors))
    overflow = net.relu('excess_scale_evidence',overflow)
    negative = net.dense('subtract_scale_excess',overflow,-np.eye(len(factors)))
    capped = net.op('bounded_scale_evidence','add',[support,negative])
    w = np.zeros((2,len(factors))); w[1] = .28
    geometric = net.dense('geometric_evidence_score',capped,w)
    skin = soft_skin_evidence(net,base)
    net.op('class_scores','add',[geometric,skin])
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'network.json').write_text(json.dumps({'layers':net.layers},indent=2)+'\n')
    np.savez(destination/'weights.npz',**net.weights)
    print('Built %d layers, %d fixed weights, %d rules at %d scales.' %
          (len(net.layers),sum(v.size for v in net.weights.values()),len(rules),len(factors)))


if __name__ == '__main__':
    build(sys.argv[1])
