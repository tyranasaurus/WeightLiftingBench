"""Data-free handwritten-digit network: analytic pen models and fixed arithmetic.

No files are read. All digit geometry below describes generic writing styles,
not observations of particular examples. There is no fitting or optimization.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


# A stroke consists of line or cubic Bezier commands in a generic [-1,1] box.
def stroke(start, *commands):
    p = np.array(start, dtype=np.float64)
    result = [p]
    for cmd in commands:
        if len(cmd) == 2:
            end = np.array(cmd, dtype=np.float64)
            t = np.linspace(0, 1, 17)[1:, None]
            part = (1-t)*p + t*end
        else:
            c1, c2, end = np.array(cmd, dtype=np.float64).reshape(3, 2)
            t = np.linspace(0, 1, 33)[1:, None]
            part = (1-t)**3*p + 3*(1-t)**2*t*c1 + 3*(1-t)*t*t*c2 + t**3*end
        result.extend(part)
        p = end
    return np.array(result)


def oval(cx, cy, rx, ry, tilt=0, start=0, end=2*math.pi):
    t = np.linspace(start, end, 161)
    y = ry*np.sin(t)
    return np.stack([cx+rx*np.cos(t)+tilt*y, cy+y], axis=1)


def pen_model(digit, style):
    if digit == 0:
        t = np.linspace(0, 2*math.pi, 193)
        power = (1.0, .85, 1.15)[style % 3]
        x = np.sign(np.cos(t))*np.abs(np.cos(t))**power
        y = np.sign(np.sin(t))*np.abs(np.sin(t))**power
        return [np.stack([x*(1+.07*y), y], axis=1)]
    if digit == 1:
        forms = [
            [stroke((0,-1), (0,1))],
            [stroke((-.55,-.55), (0,-1), (0,1))],
            [stroke((-.9,-.45), (.05,-1), (.05,1))],
            [stroke((-.45,-.75), (0,-1), (.05,1))],
            [stroke((-.6,-.65), (0,-1), (0,1)), stroke((-.55,1),(.55,1))],
            [stroke((-.15,-1), (.1,-.3,.05,.5,-.2,1))],
        ]
        return forms[style % len(forms)]
    if digit == 2:
        forms = [
            [stroke((-.9,-.55),(-.6,-1.2,.65,-1.2,.85,-.65),
                    (1,-.15,-.45,.55,-.85,.97),(-.25,.85,.4,1.12,.95,.85))],
            [stroke((-.85,-.55),(-.7,-1.1,.8,-1.2,.8,-.55),
                    (.8,-.15,-.6,.65,-.85,.98),(.9,.98))],
            [stroke((-.85,-.75),(-.3,-1.15,.8,-1.15,.8,-.6),
                    (.85,-.1,-.6,.45,-.8,.85),(-.95,1.2,.25,1.05,.9,.85))],
            [stroke((-.8,-.6),(-.75,-1.1,.65,-1.2,.8,-.65),
                    (.95,-.05,-.6,.45,-.85,.9),(-1.15,1.3,-1.1,.45,-.55,.65),(.1,.85,.5,1.15,.95,1))],
            [stroke((-.8,-.85),(.15,-1.2,.9,-1,.8,-.55),(-.8,.98),(.85,.98))],
            # Cursive two: an arch, descending diagonal, closed baseline loop, exit.
            [stroke((-.8,-.6),(-.65,-1.2,.65,-1.2,.8,-.65),
                    (1,-.1,-.3,.65,-.85,.9),(-1.4,1.2,-1.1,.35,-.5,.5),
                    (0,.65,.45,1.1,.95,1))],
            [stroke((-.75,-.65),(-.4,-1.15,.7,-1.2,.7,-.65),
                    (.75,-.15,-.3,.45,-.85,.9),(-1.2,1.25,-1.15,.45,-.6,.55),
                    (-.15,.6,.35,1.1,.95,.8))],
            # A compact upper hook and a broad sweeping lower bowl.
            [stroke((-.65,-.75),(.1,-1.3,.9,-.95,.7,-.35),
                    (.35,.2,-.95,.6,-.85,.95),(-.65,1.2,.55,.85,.95,.7))],
            [stroke((-.55,-.8),(-.15,-1.15,.6,-1.15,.65,-.6),
                    (.7,-.05,-.45,.65,-.7,.95),(-.1,.65,.6,.95,1.05,.75))],
            [stroke((-.7,-.85),(.25,-1.15,.7,-1,.65,-.5),
                    (.6,0,-.45,.55,-.8,.85),(-1.1,1.25,-1.25,.7,-.6,.6),
                    (.15,.45,.55,.85,1,.85))],
        ]
        return forms[style % len(forms)]
    if digit == 3:
        forms = [
            [stroke((-.8,-.75),(-.2,-1.2,.8,-1.15,.8,-.6),
                    (.8,-.25,.25,-.04,-.15,-.03),(.6,-.2,1,.25,.8,.65),
                    (.55,1.2,-.45,1.2,-.85,.7))],
            [stroke((-.85,-1),(.75,-1),(-.1,-.05),
                    (.8,-.25,1,.3,.75,.75),(.4,1.2,-.4,1.1,-.85,.8))],
            [stroke((-.85,-.8),(-.05,-1.1,.75,-1.1,.75,-.6),
                    (.75,-.25,-.1,-.1,-.45,-.05),(.5,-.1,.85,.2,.85,.6),
                    (.8,1.05,-.25,1.3,-.85,.7))],
        ]
        return forms[style % len(forms)]
    if digit == 4:
        forms = [
            [stroke((-.3,-1),(-.85,.25),(.85,.25)),stroke((.5,-1),(.4,1))],
            [stroke((-.55,-1),(-.7,.15),(-.45,.35,.35,.25,.85,.2)),stroke((.55,-.8),(.35,1))],
            [stroke((.5,1),(.5,-1),(-.85,.3),(.9,.3))],
            [stroke((-.3,-1),(-.85,.2),(.65,.2),(.65,-1),(.55,1))],
        ]
        return forms[style % len(forms)]
    if digit == 5:
        forms = [
            [stroke((.8,-1),(-.6,-1),(-.75,-.05),
                    (-.2,-.4,.9,-.1,.85,.5),(.85,1.2,-.4,1.25,-.85,.7))],
            [stroke((.85,-1),(-.6,-.9),(-.7,-.05),
                    (.6,-.3,1,.25,.7,.75),(.4,1.2,-.6,1.1,-.85,.65))],
            [stroke((-.5,-.9),(-.8,-.1),(.3,-.45,1,-.05,.85,.55),
                    (.6,1.2,-.45,1.15,-.85,.75)),stroke((-.5,-.9),(.9,-1))],
            [stroke((.8,-1),(-.35,-1.05,-.7,-.8,-.65,-.15),
                    (.3,-.4,.95,.1,.75,.65),(.5,1.15,-.55,1.25,-.85,.65))],
        ]
        return forms[style % len(forms)]
    if digit == 6:
        forms = [
            [stroke((.65,-1),(-.15,-1.1,-.95,.05,-.7,.65),
                    (-.5,1.25,.7,1.2,.85,.55),(.95,-.15,-.3,-.4,-.75,.25))],
            [stroke((.6,-1),(-.1,-.9,-1,.3,-.65,.8),
                    (-.25,1.25,.85,1.05,.8,.45),(.75,-.25,-.65,-.15,-.75,.5))],
            [stroke((.7,-.8),(.4,-1.3,-.45,-1,-.65,-.25),
                    (-1,.65,-.35,1.25,.4,1),(.9,.85,1,.2,.45,.05),
                    (0,-.15,-.6,.1,-.65,.45))],
            [stroke((.55,-1),(-.35,-.7,-1,.1,-.7,.7),
                    (-.4,1.2,.8,1.15,.8,.45),(.8,-.2,-.45,-.3,-.7,.3))],
        ]
        return forms[style % len(forms)]
    if digit == 7:
        forms = [
            [stroke((-.9,-1),(.85,-1),(.3,-.15,-.3,.5,-.5,1))],
            [stroke((-.9,-.7),(-.8,-1),(.85,-1),(-.4,1))],
            [stroke((-.9,-.85),(-.1,-1,.5,-1,.85,-.85),(-.1,1))],
            [stroke((-.9,-1),(.85,-1),(-.4,1)),stroke((-.55,.05),(.55,.05))],
            [stroke((-.9,-1),(.85,-1),(.25,-.2,-.3,.55,-.15,1))],
        ]
        return forms[style % len(forms)]
    if digit == 8:
        if style in (0,1,4,5):
            t = np.linspace(0, 2*math.pi, 257)
            ratio = (.12,-.12,.12,.12,.32,-.32)[style]
            y = -np.cos(t)
            x = np.sin(2*t)*(.85+ratio*y)
            return [np.stack([x,y],axis=1)]
        if style == 2:
            return [oval(0,-.52,.65,.48), oval(0,.45,.85,.55)]
        if style == 6:
            return [oval(0,-.45,.85,.55), oval(0,.55,.60,.45)]
        if style == 7:
            return [stroke((.1,.15),(-1,-.25,-.7,-1.2,.2,-1),
                           (1,-.85,.65,-.2,-.1,.15),(-.9,.55,-.7,1.15,0,1),
                           (.7,.95,.8,.45,.1,.15))]
        return [stroke((.1,0),(-.95,-.45,-.65,-1.15,.15,-1),
                       (.95,-.9,.7,-.3,-.1,0),(-1,.4,-.9,1.1,0,1),
                       (.95,1.05,1,.3,.1,0))]
    if digit == 9:
        forms = [
            [oval(-.08,-.5,.78,.5),stroke((.7,-.65),(.65,.05,.4,.65,-.2,1))],
            [oval(-.08,-.5,.78,.5),stroke((.7,-.85),(.7,1))],
            [stroke((.7,-.45),(.65,-1.1,-.75,-1.2,-.85,-.5),
                    (-.85,.2,.25,.2,.7,-.45),(.8,.2,.3,.85,-.2,1))],
            [stroke((.65,-.6),(.5,-1.2,-.8,-1.05,-.85,-.45),
                    (-.85,.2,.35,.25,.65,-.5),(.75,-.05,.5,.7,.15,1))],
        ]
        return forms[style % len(forms)]
    raise ValueError(digit)


STYLE_COUNTS = [3,6,10,3,4,4,4,5,8,4]
BASE_HALF_WIDTH = [6.2,4.0,7.5,7.0,7.2,7.0,6.5,7.0,6.4,6.6]


def radical_inverse(n, base):
    f = 1.0
    value = 0.0
    while n:
        f /= base
        value += f*(n % base)
        n //= base
    return value


def model_points(digit, index):
    # Deterministic low-discrepancy geometry, not an empirically fitted distribution.
    sequence_index = index // STYLE_COUNTS[digit] + 1
    u = np.array([radical_inverse(sequence_index,p) for p in (2,3,5,7,11,13,17,19,23,29,31)])*2-1
    paths = pen_model(digit, index % STYLE_COUNTS[digit])
    width = BASE_HALF_WIDTH[digit]*(1+.43*u[0])
    height = 9.0*(1+.075*u[1])
    # Right-leaning handwriting has negative dx/dy in image coordinates.
    shear = .48*u[2] - .18
    rotation = .12*u[3]
    transformed = []
    for path in paths:
        x, y = path[:,0], path[:,1]
        xx = x*(1+.24*u[4]*y) + .14*u[5]*(y*y-.5) + .10*u[6]*np.sin(math.pi*y)
        yy = y + .22*u[7]*(1-y*y) + .085*u[8]*x
        a = width*xx + shear*height*yy
        b = height*yy
        transformed.append(np.stack([math.cos(rotation)*a-math.sin(rotation)*b,
                                     math.sin(rotation)*a+math.cos(rotation)*b],axis=1))
    # Arc-length sampling and centering are properties of the typed pen model.
    mids, lengths = [], []
    for p in transformed:
        mids.append((p[:-1]+p[1:])*.5)
        lengths.append(np.sqrt(np.sum(np.diff(p,axis=0)**2,axis=1)))
    mids, lengths = np.concatenate(mids), np.concatenate(lengths)
    center = np.sum(mids*lengths[:,None],axis=0)/lengths.sum()
    allp = np.concatenate(transformed)
    span = np.ptp(allp,axis=0).max()
    scale = min(1.0,19.0/span)
    mids = (mids-center)*scale + np.array([13.5+.9*u[9],13.5+.9*u[10]])
    vectors = np.concatenate([np.diff(p,axis=0) for p in transformed])
    angles = np.mod(np.arctan2(vectors[:,1],vectors[:,0]),math.pi)
    # Uniform arclength sampling gives every pen segment its proper contribution.
    cumulative = np.cumsum(lengths)
    positions = (np.arange(192)+.5)*cumulative[-1]/192
    ids = np.searchsorted(cumulative,positions)
    return mids[ids], angles[ids]


def template_weights(points, angles):
    # Distances to analytically specified pen paths, in original pixel units.
    y,x = np.mgrid[:14,:14].astype(np.float64)
    grid = np.stack([2*x.ravel()+.5,2*y.ravel()+.5],axis=1)
    delta = grid[:,None,:]-points[None,:,:]
    distance = np.sqrt(np.min(np.sum(delta*delta,axis=2),axis=1))
    outside = -np.clip(distance-1.0,0,5)/196
    # Integrate the image's distance-to-ink field along the expected pen path.
    p = (points-.5)/2
    lo = np.floor(p).astype(int)
    f = p-lo
    path_weights = np.zeros((14,14),dtype=np.float64)
    for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
        xx,yy = lo[:,0]+dx,lo[:,1]+dy
        a = (f[:,0] if dx else 1-f[:,0])*(f[:,1] if dy else 1-f[:,1])
        valid = (xx>=0)&(xx<14)&(yy>=0)&(yy<14)
        np.add.at(path_weights,(yy[valid],xx[valid]),a[valid]/len(points))
    # Analytic orientation votes, interpolated both spatially and angularly.
    oriented = np.zeros((8,14,14),dtype=np.float64)
    abin = angles*8/math.pi
    alo = np.floor(abin).astype(int)
    af = abin-alo
    for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
        xx,yy = lo[:,0]+dx,lo[:,1]+dy
        a = (f[:,0] if dx else 1-f[:,0])*(f[:,1] if dy else 1-f[:,1])
        valid = (xx>=0)&(xx<14)&(yy>=0)&(yy<14)
        for da in (0,1):
            weight = a*(af if da else 1-af)/len(points)
            np.add.at(oriented,((alo[valid]+da)%8,yy[valid],xx[valid]),weight[valid])
    return outside.astype(np.float32),path_weights.ravel().astype(np.float32),oriented.ravel().astype(np.float32)


def orientation_features(net):
    # Short line-average minus parallel flanks at two thicknesses.
    filters = np.zeros((16,1,9,9),dtype=np.float64)
    for o in range(8):
        theta = o*math.pi/8
        for thickness in range(2):
            separation = 2+thickness
            for along in np.linspace(-2,2,9):
                for side,weight in ((0,1),(-separation,-.5),(separation,-.5)):
                    x = 4+along*math.cos(theta)-side*math.sin(theta)
                    y = 4+along*math.sin(theta)+side*math.cos(theta)
                    ix,iy = int(math.floor(x)),int(math.floor(y))
                    fx,fy = x-ix,y-iy
                    for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
                        if 0<=ix+dx<9 and 0<=iy+dy<9:
                            filters[2*o+thickness,0,iy+dy,ix+dx] += weight*(fx if dx else 1-fx)*(fy if dy else 1-fy)/9
    net.conv('stroke_directions','ink',filters,np.zeros(16),padding=4)
    net.op('positive_directions','relu','stroke_directions')
    net.op('direction_thicknesses','reshape','positive_directions',shape=[8,2,784])
    net.op('best_thickness','maxpool','direction_thicknesses',size=[2,1],stride=[2,1])
    net.op('direction_grids','reshape','best_thickness',shape=[8,28,28])
    net.conv('direction_padding','direction_grids',np.eye(8).reshape(8,8,1,1),np.zeros(8),padding=1)
    net.op('local_directions','maxpool','direction_padding',size=[3,3],stride=[1,1])
    return net.op('coarse_directions','avgpool','local_directions',size=[2,2],stride=[2,2])


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}
    def op(self,name,op,inputs,**settings):
        if isinstance(inputs,str): inputs = [inputs]
        self.layers.append(dict(name=name,op=op,inputs=inputs,**settings))
        return name
    def conv(self,name,source,W,b,padding=0):
        W = np.asarray(W,dtype=np.float32)
        self.weights[name+'.W'] = W
        self.weights[name+'.b'] = np.asarray(b,dtype=np.float32)
        return self.op(name,'conv',source,out=W.shape[0],kernel=list(W.shape[2:]),stride=1,padding=padding)
    def dense(self,name,source,W,b):
        W = np.asarray(W,dtype=np.float32)
        self.weights[name+'.W'] = W
        self.weights[name+'.b'] = np.asarray(b,dtype=np.float32)
        return self.op(name,'dense',source,out=W.shape[0])


def normalized_ink(net,source):
    """Fixed piecewise-linear arithmetic for x / mean(x), not fitted weights."""
    net.op('ink_mean','avgpool',source,size=[14,14],stride=[14,14])
    knots = np.array([.02,.025,.03,.035,.04,.05,.06,.07,.08,.10,.12,.14,.16,.18,
                      .20,.23,.26,.30,.35,.40,.50,.65,.8,1.0],dtype=np.float64)
    vals = 1/(16*knots)
    slopes = np.diff(vals)/np.diff(knots)
    coefficients = np.diff(np.r_[0,slopes,0])
    net.dense('reciprocal_hinges','ink_mean',np.ones((len(knots),1)),-knots)
    net.op('reciprocal_relu','relu','reciprocal_hinges')
    W = np.zeros((392,len(knots)),dtype=np.float32)
    W[196:] = coefficients
    b = np.zeros(392,dtype=np.float32)
    b[196:] = vals[0]
    net.dense('reciprocal_broadcast','reciprocal_relu',W,b)
    net.op('reciprocal_grid','reshape','reciprocal_broadcast',shape=[2,14,14])
    net.conv('ink_channel',source,np.array([1,0]).reshape(2,1,1,1),[0,0])
    net.op('division_operands','add',['ink_channel','reciprocal_grid'])
    # xy = ((x+y)^2-(x-y)^2)/4. Here y=1/(16*mean(x)).
    # Linear interpolation of t^2 on a uniform grid is explicit arithmetic.
    sqknots = np.arange(-4,5.0001,.125)
    slopes = 2*sqknots[:-1]+.125
    coeff = np.r_[slopes[0],np.diff(slopes),0]
    count = len(sqknots)
    W = np.zeros((2*count,2,1,1),dtype=np.float32)
    W[:,0,0,0] = 1
    W[:count,1,0,0] = 1
    W[count:,1,0,0] = -1
    net.conv('product_hinges','division_operands',W,np.tile(-sqknots,2))
    net.op('product_relu','relu','product_hinges')
    net.conv('unit_mass_ink','product_relu',np.r_[4*coeff,-4*coeff].reshape(1,-1,1,1),[0])
    return 'unit_mass_ink'


def topology_features(net):
    """Fixed eight-neighbor flood fill of background, expressed using ReLUs.

    The small rule branch distinguishes upper/lower loops from a hollow center.
    These are qualitative digit facts, not label-derived statistics.
    """
    net.conv('topology_ramp','image',np.array([10,10]).reshape(2,1,1,1),[-2,-3])
    net.op('topology_ramp_relu','relu','topology_ramp')
    net.conv('barrier','topology_ramp_relu',np.array([1,-1]).reshape(1,2,1,1),[0])
    net.conv('free_space','barrier',-np.ones((1,1,1,1)),[1])
    border = np.zeros((28,28),dtype=np.float32)
    border[0,:] = border[-1,:] = border[:,0] = border[:,-1] = 1
    net.dense('outside_seed','ink_mean',np.zeros((784,1)),border.ravel())
    net.op('outside_grid','reshape','outside_seed',shape=[1,28,28])
    net.conv('negative_barrier','barrier',-np.ones((1,1,1,1)),[0])
    previous = 'outside_grid'
    for i in range(32):
        prefix = 'flood_'+str(i)
        net.conv(prefix+'_pad',previous,np.ones((1,1,1,1)),[0],padding=1)
        net.op(prefix+'_neighbors','maxpool',prefix+'_pad',size=[3,3],stride=[1,1])
        net.op(prefix+'_gate','add',[prefix+'_neighbors','negative_barrier'])
        previous = net.op(prefix,'relu',prefix+'_gate')
    net.conv('negative_outside',previous,-np.ones((1,1,1,1)),[0])
    net.op('enclosed_background','add',['free_space','negative_outside'])
    net.op('holes','relu','enclosed_background')
    y,x = np.mgrid[:28,:28]
    # Broad regions only: a central hole, an upper hole, and a lower hole.
    masks = [((x-13.5)/6)**2+((y-13.5)/3.5)**2<=1,
             ((x-13.5)/6)**2+((y-9)/3.5)**2<=1,
             ((x-13.5)/6)**2+((y-18)/3.5)**2<=1]
    W = np.stack([m.ravel()/12 for m in masks]).astype(np.float32)
    net.dense('hole_evidence','holes',np.concatenate([W,W]),np.r_[np.zeros(3),-np.ones(3)])
    net.op('hole_evidence_relu','relu','hole_evidence')
    # Clipped evidence in [0,1] keeps the topology branch a secondary cue.
    rules = np.zeros((10,3),dtype=np.float32)
    rules[0] = [.20,0,0]
    rules[2] = [0,-.14,0]
    rules[3] = [0,-.08,-.08]
    rules[6] = [-.14,0,.06]
    rules[8] = [-.04,.06,.06]
    rules[9] = [0,.06,-.10]
    net.dense('topology_scores','hole_evidence_relu',np.concatenate([rules,-rules],axis=1),np.zeros(10))
    return 'topology_scores'


def build(destination):
    net = Network()
    net.conv('ink_ramp','image',np.array([2.5,2.5]).reshape(2,1,1,1),[-.25,-1.25])
    net.op('ink_ramp_relu','relu','ink_ramp')
    net.conv('ink','ink_ramp_relu',np.array([1,-1]).reshape(1,2,1,1),[0])
    net.op('ink_coarse','avgpool','ink',size=[2,2],stride=[2,2])
    mass_normalized = normalized_ink(net,'ink_coarse')
    dilations = []
    previous = 'ink'
    for radius in range(5):
        if radius:
            net.conv('pad_'+str(radius),previous,np.ones((1,1,1,1)),[0],padding=1)
            previous = net.op('dilate_'+str(radius),'maxpool','pad_'+str(radius),size=[3,3],stride=[1,1])
        name = net.op('distance_scale_'+str(radius),'avgpool',previous,size=[2,2],stride=[2,2])
        scale_weight = (1.0,1.25,1.0,.75,.5)[radius]
        name = net.conv('proximity_weight_'+str(radius),name,np.array([scale_weight]).reshape(1,1,1,1),[0])
        dilations.append(name)
    net.op('ink_proximity','add',dilations)
    direction_input = orientation_features(net)
    count = 5120
    forward = np.empty((10*count,196),dtype=np.float32)
    backward = np.empty_like(forward)
    directions = np.empty((10*count,8*196),dtype=np.float32)
    for digit in range(10):
        for i in range(count):
            forward[digit*count+i],backward[digit*count+i],directions[digit*count+i] = template_weights(*model_points(digit,i))
        print('Constructed analytic models for digit',digit,flush=True)
    net.dense('extra_stroke_cost',mass_normalized,forward,np.zeros(10*count))
    net.dense('expected_stroke_support','ink_proximity',backward,np.zeros(10*count))
    net.dense('expected_stroke_direction',direction_input,2*directions,np.zeros(10*count))
    net.op('model_scores','add',['extra_stroke_cost','expected_stroke_support','expected_stroke_direction'])
    net.op('models_by_class','reshape','model_scores',shape=[10,count,1])
    net.op('best_model','maxpool','models_by_class',size=[count,1],stride=[count,1])
    net.op('shape_scores','reshape','best_model',shape=[10])
    topology_scores = topology_features(net)
    net.op('scores','add',['shape_scores',topology_scores])
    out = Path(destination)
    out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':net.layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**net.weights)
    print('Wrote',sum(w.size for w in net.weights.values()),'float32 parameters.',flush=True)


if __name__ == '__main__':
    build(sys.argv[1])
