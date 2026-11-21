"""Analytic person-presence network; no files are read and no fitting is done.

Colour cones, Gaussian region integrals, facial anatomy, and head/torso
relationships are specified from general knowledge. All parameters are
computed from the formulas below, independently of any image dataset.
"""
import json
import sys
from pathlib import Path
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}
        self.shapes = {'image': (3, 96, 96)}

    def conv(self, name, src, W, b=0.0, stride=1, padding=0):
        W = np.asarray(W, dtype=np.float32)
        if W.ndim == 2:
            W = W[:, :, None, None]
        o, c, k, _ = W.shape
        assert c == self.shapes[src][0]
        self.layers.append(dict(name=name, op='conv', inputs=[src], out=o,
                                kernel=[k,k], stride=stride, padding=padding))
        self.weights[name+'.W'] = W
        self.weights[name+'.b'] = np.broadcast_to(np.asarray(b,np.float32),(o,)).copy()
        _,h,w = self.shapes[src]
        self.shapes[name] = (o,(h+2*padding-k)//stride+1,(w+2*padding-k)//stride+1)
        return name

    def dense(self, name, src, W, b=0.0):
        W = np.asarray(W,np.float32)
        self.layers.append(dict(name=name,op='dense',inputs=[src],out=len(W)))
        self.weights[name+'.W'] = W
        self.weights[name+'.b'] = np.broadcast_to(np.asarray(b,np.float32),(len(W),)).copy()
        self.shapes[name] = (len(W),)
        return name

    def unary(self, name, op, src):
        self.layers.append(dict(name=name,op=op,inputs=[src]))
        self.shapes[name] = self.shapes[src]
        return name

    def pool(self, name, op, src, size, stride=None):
        if isinstance(size,int): size=(size,size)
        if stride is None: stride=size
        elif isinstance(stride,int): stride=(stride,stride)
        self.layers.append(dict(name=name,op=op,inputs=[src],size=list(size),stride=list(stride)))
        c,h,w = self.shapes[src]
        self.shapes[name] = (c,(h-size[0])//stride[0]+1,(w-size[1])//stride[1]+1)
        return name

    def reshape(self, name, src, shape):
        assert int(np.prod(shape)) == int(np.prod(self.shapes[src]))
        self.layers.append(dict(name=name,op='reshape',inputs=[src],shape=list(shape)))
        self.shapes[name] = tuple(shape)
        return name

    def add(self, name, sources):
        assert all(self.shapes[s]==self.shapes[sources[0]] for s in sources)
        self.layers.append(dict(name=name,op='add',inputs=sources))
        self.shapes[name] = self.shapes[sources[0]]
        return name

    def concatenate(self, name, sources):
        total = sum(self.shapes[s][0] for s in sources)
        parts, offset = [],0
        for i,s in enumerate(sources):
            c = self.shapes[s][0]
            W = np.zeros((total,c),np.float32)
            W[offset:offset+c] = np.eye(c)
            offset += c
            parts.append(self.conv(name+'_part%d'%i,s,W))
        return self.add(name,parts)

    def globalmax(self, name, src):
        c,h,w = self.shapes[src]
        z = self.reshape(name+'_flat',src,(1,c,h*w))
        return self.pool(name,'maxpool',z,(c,h*w))

    def minimum(self, name, src):
        c,h,w = self.shapes[src]
        z = self.conv(name+'_negative',src,-np.eye(c))
        z = self.reshape(name+'_flat',z,(1,c,h*w))
        z = self.pool(name+'_max','maxpool',z,(c,1),(c,1))
        z = self.reshape(name+'_map',z,(1,h,w))
        return self.conv(name,z,[[-1]])

    def absolute(self, name, src):
        c = self.shapes[src][0]
        z = self.conv(name+'_signed',src,np.concatenate([np.eye(c),-np.eye(c)]))
        z = self.unary(name+'_rectified','relu',z)
        return self.conv(name,z,np.concatenate([np.eye(c),np.eye(c)],axis=1))


def gaussian(k,x,y,sx,sy,angle=0.):
    yy,xx = np.mgrid[:k,:k]
    a = np.zeros((k,k),np.float64)
    ca,sa = np.cos(angle),np.sin(angle)
    for dy in (-.375,-.125,.125,.375):
        for dx in (-.375,-.125,.125,.375):
            X,Y = xx+dx-x,yy+dy-y
            u,v = ca*X+sa*Y,-sa*X+ca*Y
            a += np.exp(-.5*(u/sx)**2-.5*(v/sy)**2)
    return a/a.sum()


def masks(k,width,aspect=1.30,angle=0.,head_y=None):
    x = (k-1)/2
    y = (k-1)/2 if head_y is None else head_y
    h = width*aspect
    ca,sa = np.cos(angle),np.sin(angle)
    def g(dx,dy,sx,sy):
        dx,dy=dx*width,dy*h
        return gaussian(k,x+ca*dx-sa*dy,y+sa*dx+ca*dy,
                        max(.24,sx*width),max(.24,sy*h),angle)
    return g


def front_end(net):
    # Penalize departures from a broad, illumination-tolerant skin colour cone.
    rules = [
        ([-1,1,0],.025,.025), ([0,-1,1],-.015,.035),
        ([-1,0,1],.045,.035), ([1,-1,0],-.27,.075),
        ([0,1,-1],-.18,.070), ([1,-4.0,3.0],-.050,.060),
        ([-1.2,2.2,-1],-.025,.045), ([-1,0,0],.16,.080),
        ([0,-1,0],.065,.050), ([0,0,-1],.035,.040),
    ]
    W = [np.asarray(w)/soft for w,b,soft in rules]
    b = [b/soft for w,b,soft in rules]
    z = net.conv('colour_violations','image',W,b)
    z = net.unary('colour_penalties','relu',z)
    z = net.conv('skin_raw',z,-np.ones((1,len(rules))),1)
    skin = net.unary('skin','relu',z)
    grey = net.conv('luminance','image',[[.299,.587,.114]])
    return skin,grey


def face_detector(net,skin,grey,index,down,width):
    p = 'face%d_'%index
    s = net.pool(p+'skin','avgpool',skin,down)
    l = net.pool(p+'grey','avgpool',grey,down)
    l = net.unary(p+'contrast','norm',l)
    # Local high-pass energy is an illumination-independent yardstick for
    # the facial bands. Flat highlights and arbitrary high-contrast edges
    # should not look like a face simply because their raw contrast is high.
    smooth = gaussian(7,3,3,.7*width,.7*width)
    highpass = -smooth
    highpass[3,3] += 1
    local = net.conv(p+'local_contrast',l,highpass[None,None],padding=3)
    local = net.absolute(p+'local_energy',local)
    f = net.concatenate(p+'input',[s,l,local])
    k = 11
    feature_rows = []
    condition_rows = []
    # Slight tilt and differing aspect ratios accommodate camera poses and
    # the anisotropic resizing of whole photographs.
    geometries = [(1.30,-.25),(1.30,0.),(1.30,.25),(1.0,0.),(1.65,0.)]
    for aspect,angle in geometries:
        g = masks(k,width,aspect,angle)
        center = .5*(g(-.23,.12,.17,.17)+g(.23,.12,.17,.17))
        left,right = g(-.85,0,.23,.32),g(.85,0,.23,.32)
        above = g(0,-.86,.30,.18)
        eyel,eyer = g(-.23,-.19,.11,.085),g(.23,-.19,.11,.085)
        cheekl,cheekr = g(-.23,.12,.14,.11),g(.23,.12,.14,.11)
        nose,forehead = g(0,.02,.09,.12),g(0,-.39,.18,.09)
        mouth = g(0,.36,.18,.08)
        features = [(0,center),(0,left),(0,right),(0,above),
                    (1,cheekl-eyel),(1,cheekr-eyer),
                    (1,nose-eyel),(1,nose-eyer),
                    (1,.5*(cheekl+cheekr)-mouth),
                    (1,forehead-eyel),(1,forehead-eyer),
                    (2,g(0,0,.40,.40))]
        for channel,mask in features:
            row = np.zeros((3,k,k),np.float32)
            row[channel] = mask
            feature_rows.append(row)
        # Every condition must hold: a strong colour patch cannot compensate
        # for a missing eye pair, nose bridge, or mouth band.
        C = np.zeros((11,12),np.float32)
        b = np.zeros(11,np.float32)
        C[0,0],b[0] = 1/.30,-.40/.30
        C[1,0],C[1,1],b[1] = 1/.25,-1/.25,-.05/.25
        C[2,0],C[2,2],b[2] = 1/.25,-1/.25,-.05/.25
        C[3,0],C[3,3] = 1/.30,-1/.30
        for j,relative,denom in ((4,.45,.14),(5,.45,.14),
                                  (6,.25,.13),(7,.25,.13),
                                  (8,.12,.09),(9,.30,.14),(10,.30,.14)):
            C[j,j] = 1/denom
            C[j,11] = -relative/denom
        condition_rows.append((C,b))
    z = net.conv(p+'anatomy',f,feature_rows,padding=k//2)
    C = np.zeros((len(geometries)*11,len(geometries)*12),np.float32)
    b = np.zeros(len(geometries)*11,np.float32)
    for i,(c,bi) in enumerate(condition_rows):
        C[i*11:(i+1)*11,i*12:(i+1)*12] = c
        b[i*11:(i+1)*11] = bi
    z = net.conv(p+'conditions',z,C,b)
    views = []
    for i in range(len(geometries)):
        select = np.zeros((11,len(geometries)*11),np.float32)
        select[:,i*11:(i+1)*11] = np.eye(11)
        v = net.conv(p+'view%d'%i,z,select)
        v = net.minimum(p+'conjunction%d'%i,v)
        views.append(v)
    z = net.concatenate(p+'view_scores',views)
    return net.globalmax(p+'best',z)


def compact_regions(net,skin):
    # Heads and hands are isolated skin patches. Four surrounding sectors
    # distinguish them from uninterrupted walls, timber, fur, or food surfaces.
    candidates = []
    scales = [(1,2.5),(2,2.2),(2,3.4),(3,3.4),(4,3.4),(6,3.4),(8,3.4)]
    for index,(down,width) in enumerate(scales):
        p = 'compact%d_'%index
        s = skin if down == 1 else net.pool(p+'reduced','avgpool',skin,down)
        k = 13
        g = masks(k,width,1.25)
        regions = [g(0,0,.24,.26),g(-.85,0,.22,.28),
                   g(.85,0,.22,.28),g(0,-.78,.27,.20),
                   g(0,1.10,.43,.22)]
        z = net.conv(p+'regions',s,np.asarray(regions)[:,None],padding=k//2)
        W,b = np.zeros((5,5),np.float32),np.zeros(5,np.float32)
        W[0,0],b[0] = 1/.50,-.20/.50
        for j,d in ((1,.50),(2,.50),(3,.45),(4,.45)):
            W[j,0],W[j,j] = 1/d,-1/d
        z = net.conv(p+'isolation',z,W,b)
        z = net.minimum(p+'all_sides',z)
        z = net.globalmax(p+'best',z)
        candidates.append(z)
    z = net.concatenate('compact_scales',candidates)
    z = net.globalmax('compact_peak',z)
    return bounded(net,'compact_support',z)


def bounded(net,name,src):
    # clip(x, 0, 1) = relu(x) - relu(x-1).
    z = net.conv(name+'_hinges',src,[[1],[1]],[0,-1])
    z = net.unary(name+'_positive','relu',z)
    return net.conv(name,z,[[1,-1]])


def colour_presence(net,skin,grey):
    # A face or hand is a moderate-sized concentration of skin, whereas warm
    # walls, timber, and close-up foods often occupy much of the whole image.
    total = net.pool('skin_total','avgpool',skin,96)
    total = net.conv('coverage_ramp',total,[[100]],-.10)
    coverage = bounded(net,'coverage',total)
    mass = net.pool('skin_mass','avgpool',skin,96)
    excess = net.conv('excess_skin_ramp',mass,[[1/.28]],-.16/.28)
    excess = bounded(net,'excess_skin',excess)
    small = net.pool('skin_48','avgpool',skin,2)
    pooled = []
    for k in (2,3,5,8,12):
        q = net.pool('skin_patch%d'%k,'avgpool',small,k,1)
        q = net.globalmax('skin_peak%d'%k,q)
        pooled.append(q)
    # Small and medium patches matter; a one-pixel warm glint does not.
    peaks = net.concatenate('patch_peaks',pooled)
    peak = net.conv('multiscale_peak',peaks,[[.15,.25,.30,.20,.10]])
    mass_term = net.conv('mass_discount',mass,[[-1.5]])
    localization = net.add('localized_skin',[peak,mass_term])
    localization = net.conv('localization_ramp',localization,[[1/.45]],-.15/.45)
    localization = bounded(net,'localization',localization)
    fields = net.pool('broad_skin_fields','avgpool',small,24,8)
    fields = net.globalmax('broad_skin_peak',fields)
    fields = net.conv('broad_skin_ramp',fields,[[1/.40]],-.35/.40)
    fields = bounded(net,'broad_skin_penalty',fields)
    all_terms = net.concatenate('colour_summary',[coverage,localization,excess,fields])
    return net.conv('colour_presence',all_terms,[[.55,.45,-.40,-.15]])


def build():
    net = Network()
    skin,grey = front_end(net)
    colour = colour_presence(net,skin,grey)
    compact = compact_regions(net,skin)
    terms = []
    for i,(down,width) in enumerate([(2,3.2),(3,3.2),(4,3.2),(6,3.2),(8,3.2),(12,3.2)]):
        terms.append(face_detector(net,skin,grey,i,down,width))
    evidence = net.concatenate('detector_scores',terms)
    face = net.globalmax('face_evidence',evidence)
    face = net.conv('face_ramp',face,[[1]],-.35)
    face = bounded(net,'face_support',face)
    summary = net.concatenate('final_cues',[colour,compact,face,'excess_skin','broad_skin_penalty'])
    evidence = net.conv('person_evidence',summary,[[.30,.60,.15,-.15,-.10]])
    net.dense('class_scores',evidence,[[0.],[1.]],[.58,0.])
    return net


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=True)
    net = build()
    (out/'network.json').write_text(json.dumps({'layers':net.layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**net.weights)
    print('Wrote %d layers and %d float32 parameters.' %
          (len(net.layers),sum(a.size for a in net.weights.values())))


if __name__ == '__main__':
    main()
