#!/usr/bin/env python3
"""Compile a hand-written shape and appearance model into the eight scoring ops.

Only formula-generated masks and hand-set coefficients are used. There is no
runtime other than the graph in network.json, and no input file to this builder.
"""
import json
import sys
from pathlib import Path
import numpy as np


# ---- Generative silhouettes ------------------------------------------------
# Coordinates are a square of side 1, x rightward and y downward. Every random
# variable is an independent style, scale, or positioning choice, not an image
# index. Shapes are painted on a fine square and area-sampled into 14 x 14.
GRID = 42
Q = (np.arange(GRID) + .5) / GRID - .5
YX = np.meshgrid(Q, Q, indexing='xy')


def polygon(x, y, vertices):
    inside = np.zeros(x.shape, bool)
    px, py = vertices[-1]
    for nx, ny in vertices:
        if py != ny:
            inside ^= ((ny > y) != (py > y)) & (x < nx + (y-ny)*(px-nx)/(py-ny))
        px, py = nx, ny
    return inside


def oval(x, y, cx, cy, rx, ry):
    return ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1


def stroke(x, y, points, radius):
    hit = np.zeros(x.shape, bool)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx, dy = x1-x0, y1-y0
        u = np.clip(((x-x0)*dx+(y-y0)*dy)/(dx*dx+dy*dy+1.e-10), 0, 1)
        hit |= (x-x0-u*dx)**2 + (y-y0-u*dy)**2 <= radius**2
    return hit


def top_form(x, y, rng, c, style):
    # Bodies and tapered sleeves are independent pieces placed on a shared
    # shoulder. A scoop, V, or small collar opening is removed from the top.
    U = rng.uniform
    t = y + .5
    if c == 0:
        types = ('tee','tee','tee','tee','tee','tee','drop','tank','cap','tank','long','long')
    elif c == 2:
        types = ('long','long','long','long','long','long','drop','long','long','cap')
    elif c == 3:
        types = ('tank','tank','cap','tee','tee','mid','long','tank','cap','mid')
    elif c == 4:
        types = ('long','long','long','long','long','long','mid','mid','mid','long','long','mid')
    else:
        types = ('long','long','long','long','long','mid','tee','tee','cap','tank','drop','long')
    sleeve = types[style % len(types)]
    if c == 3:
        bust = U(.127,.228) if sleeve != 'long' else U(.164,.265)
        waist = bust * U(.72,.98)
        hem = np.clip(bust*U(.88,1.63)+U(-.01,.065)*(style % 3 == 1),.102,.43)
        bod_end = U(.954,1.018)
        sh_y = U(.07,.16)
        sh_x = bust*U(.86,1.17)
    else:
        bust = U(.232,.371) if c == 0 else U(.237,.355)
        if c == 4:
            bust += U(-.01,.05)
            if style % 7 in (1,2) and sleeve in ('long','mid'):
                bust = U(.166,.241)  # long overcoat; a thigh-length body and shorter arms
                sleeve = 'mid'
        if c == 6:
            bust -= U(-.013,.037)
        if sleeve == 'tank':
            bust -= U(.015,.060)
        waist = bust*U(.80,1.075)
        hem = waist*U(.92,1.24)
        sh_y = U(.049,.147) if c in (4,6) else U(.045,.129)
        sh_x = bust*U(.87,1.15)
        bod_end = U(.952,1.026)
        if sleeve == 'long':
            bod_end -= {0:U(.009,.13),2:U(.013,.148),4:U(-.012,.149),6:U(-.031,.082)}[c]
    neck_w = U(.085,.195) if c in (0,3) else U(.060,.15)
    neck_d = U(.032,.118) if c in (0,3) else U(.020,.083)
    if sleeve == 'tank':
        neck_w *= U(.88,1.19)
        neck_d += U(.025,.12)
    if c == 6 and style % 4 != 1:
        neck_w *= .74
        neck_d *= .77
    if c == 4 and style % 4 != 0:
        neck_w *= .79
    top = U(-.007,.043)
    arm_y = U(.212,.365) if sleeve == 'tank' else U(.19,.34)
    bend = U(.37,.57) if c == 3 else U(.54,.72)
    hem_curve = U(.007,.05) if c != 6 else U(.027,.145)
    if c == 3:
        hem_curve = U(.009,.049)
    collar_x = max(neck_w*.85,.056)
    right = [(0,top), (collar_x,top), (sh_x*.73, top+(sh_y-top)*.36),
             (sh_x, sh_y), (bust,arm_y), (waist,bend),
             (hem,bod_end-hem_curve), (hem*.78,bod_end-hem_curve*.32), (0,bod_end)]
    torso = polygon(x,t,right+[(-a,b) for a,b in reversed(right[1:-1])])
    for side in (-1,1):
        if sleeve == 'tank':
            continue
        z = side*x
        if sleeve in ('tee','cap','drop'):
            if sleeve == 'cap':
                end = U(.193,.338)
                reach = U(.047,.110)
            elif sleeve == 'drop':
                end = U(.395,.552)
                reach = U(.078,.148)
            else:
                end = U(.265,.439)
                reach = U(.089,.174)
            outer = min(.502,sh_x+reach)
            cuff_inner = bust*U(.84,.99)
            sleeve_pts = [(sh_x*.75,sh_y*.65), (sh_x,sh_y),
                          (outer-reach*.28,sh_y+U(.055,.099)), (outer,end-U(.00,.051)),
                          (cuff_inner,end), (bust*.77,arm_y-.033)]
        else:
            end = U(.900,1.016) if sleeve == 'long' else U(.574,.803)
            if c == 3:
                end -= U(.02,.125)
            cuff_width = U(.066,.113) if c in (3,6) else U(.081,.147)
            reach = min(.512,bust+U(.094,.197))
            if c == 4:
                reach += U(-.028,.012)
            elbow_y = U(.39,.54)
            drape = U(-.025,.025)
            sleeve_pts = [(sh_x*.71,sh_y*.69), (sh_x,sh_y), (sh_x+U(.035,.075),sh_y+.063),
                          (reach-U(.007,.067)+drape,elbow_y), (reach,end-.014),
                          (reach-cuff_width,end), (reach-cuff_width-U(.018,.058),elbow_y+.04),
                          (bust*.76,arm_y-.04)]
        torso |= polygon(z,t,sleeve_pts)
    if style % 4 == 1 and sleeve != 'tank':
        v = [(0,top+neck_d),(-neck_w,top-.026),(neck_w,top-.026)]
        torso &= ~polygon(x,t,v)
    else:
        torso &= ~oval(x,t,0,top-.023,neck_w,max(.027,neck_d+.023))
    return torso, sleeve


def trousers(x, y, rng, style):
    U = rng.uniform
    t = y+.5
    hip = U(.132,.292)
    if style % 8 == 0:
        hip += U(.02,.158)
    waist = hip*U(.79,1.03)
    fork = U(.23,.54)
    end = U(.95,1.013)
    silhouette = style % 8
    stance = U(.022,.094) if silhouette in (0,4,6) else U(-.009,.051)
    knee_split = U(.001,.048) if silhouette != 6 else U(.034,.085)
    taper = U(.46,.86) if silhouette != 0 else U(.90,1.14)
    shape = np.zeros(x.shape, bool)
    for side in (-1,1):
        cuff_outer = max(stance+.027,hip*taper+U(-.028,.028))
        knee = hip*U(.72,.99)
        foot = U(-.023,.023)
        k = U(.57,.68)
        leg = [(0,-.015), (waist,U(-.012,.018)), (hip,U(.10,.23)),
               (knee,k), (cuff_outer,end+foot), (stance+U(-.023,.026),end+foot),
               (knee_split+U(-.020,.021),k), (.006,fork), (-.029,fork-.046)]
        shape |= polygon(side*x,t,leg)
    return shape, 'legs'


def footwear(x, y, rng, c, style):
    U = rng.uniform
    # The outsole, toe box, instep and rear quarter are distinct parts.
    # Toes point to the left in these constructions (reflection is sampled).
    if c == 7:
        rise = U(.238,.408)
        if style % 10 in (0,1):
            rise += U(.055,.157)
        base = rise*.50
        tip = U(.027,.071)
        sole = U(.037,.092)
        toe_depth = U(.056,.131)
        instep = U(-.016,.182)
        heel_hi = U(.50,.89)
        notch = U(.25,.39)
        topline = [(-.493,base-tip-toe_depth*.38),(-.414,base-toe_depth),
                   (-.217,base-toe_depth*1.35),(-.098,base-rise*U(.51,.79)),
                   (instep,base-rise),(.224,base-rise*U(.56,.79)),
                   (notch,base-rise*U(.45,.65)),(.448,base-rise*heel_hi),
                   (.491,base-rise*.44),(.502,base-sole*.79),(.469,base-.005),
                   (.19,base),(-.32,base-U(-.019,.036)),(-.474,base-tip*.61)]
        fill = polygon(x,y,topline)
        return fill,'trainer'
    if c == 5 and style % 13 in (1,2,3,4):
        # A separate family of flat, thick-soled sandals. A wide forestrap or
        # slingback can approach a sneaker contour, but leaves several gaps.
        rise = U(.13,.345)
        base = rise*.5
        sole = U(.036,.100)
        baggy = style % 13 in (1,4)
        out = polygon(x,y,[(-.503,base-sole*.74),(-.443,base-.028),(-.31,base),
                            (.482,base),(.505,base-sole),(.32,base-sole*.88),(-.405,base-sole)])
        toe_x = U(-.32,-.15)
        sx = U(.12,.25) if baggy else U(.062,.147)
        strap_h = rise*U(.54,.94)
        out |= polygon(x,y,[(toe_x-sx*.65,base-sole*.85),(toe_x-sx*.20,base-strap_h),
                             (toe_x+sx*.47,base-strap_h*1.08),(toe_x+sx,base-sole)])
        if style % 13 != 2:
            rx = U(.19,.36)
            out |= stroke(x,y,[(rx-.115,base-sole),(rx-.025,base-rise*.94),
                               (.44,base-rise*U(.68,.95)),(.469,base-sole)],U(.022,.060))
            if baggy:
                out |= polygon(x,y,[(rx+.019,base-sole),(.422,base-rise*.87),(.447,base-sole)])
        return out, 'sandal'
    tall = c == 9 or style % 10 in (0,1,2,3,4)
    rise = (U(.381,.705) if c == 9 else U(.24,.67)) if tall else U(.105,.254)
    if c == 9 and style % 12 == 0:
        rise = U(.70,.939)
    heeled = (style % 9 in (0,1,2,3) if c == 9 else style % 10 in (0,1,2,3))
    base = rise*.50
    platform = U(.028,.072)
    toe_up = U(.012,.065)
    back = U(.436,.503)
    heel_rise = rise*U(.25,.53) if heeled else U(.007,.035)
    front = [(-.496,base-toe_up),(-.451,base-.004),(-.317,base+.005),(-.176,base-.017)]
    back_curve = [(.025,base-heel_rise*.50),(.225,base-heel_rise*.94),
                  (back,base-heel_rise), (back-.015,base-heel_rise-platform)]
    above = [(xx, yy-platform) for xx, yy in reversed(front)]
    under = polygon(x,y,front+back_curve+[(.215,base-heel_rise*.97-platform),(.005,base-heel_rise*.50-platform)]+above)
    if heeled:
        thick = U(.033,.148) if style % 4 != 0 else U(.13,.235)
        near = back-U(.005,.049)
        under |= polygon(x,y,[(back-thick,base-heel_rise-.047),(back,base-heel_rise-.032),
                              (near,base),(near-thick*U(.36,.90),base)])
    if c == 9:
        topline = [(-.483,base-platform*.71-toe_up),(-.436,base-platform-U(.026,.083)),
                   (-.255,base-platform-U(.036,.099)-heel_rise*.17),
                   (-.105,base-rise*U(.30,.54)),(U(-.047,.127),base-rise*U(.53,.78)),
                   (U(-.078,.161),base-rise*U(.90,1.02)),
                   (back-U(.018,.082),base-rise*U(.94,1.03)),
                   (back-U(-.008,.040),base-rise*U(.77,.93))]
        vamp = polygon(x,y,topline+[(back,base-heel_rise-.008)]+list(reversed(front+back_curve[:-1])))
        return under | vamp,'boot'
    # Sandals: straps are bands rather than an opaque vamp; wedge and
    # platform styles have a heavier sole, but retain an open upper.
    r = U(.024,.062)
    if style % 7 == 0:
        r *= .60
    top_ankle = base-rise
    sole_y = base-heel_rise*.77-platform*.8
    if heeled or tall:
        if style % 5 != 4:
            under |= stroke(x,y,[(back-.021,sole_y),(back-.052,top_ankle+.039),
                                 (back-U(.12,.27),top_ankle)],r*U(.72,1.24))
        if style % 4 != 0:
            under |= stroke(x,y,[(-.147,base-platform),(.005,base-rise*.48),
                                 (back-U(.074,.16),top_ankle+.04)],r*U(.68,1.4))
        if style % 10 in (1,4):
            under |= polygon(x,y,[(-.087,base-heel_rise*.36),(back-.09,sole_y-.05),
                                  (back,sole_y-.02),(back,base),(.11,base-.014)])
        if style % 7 == 0:
            under |= stroke(x,y,[(.08,base-rise*.35),(.167,top_ankle+.04),
                                  (back-.036,top_ankle),(.25,sole_y)],r*.82)
    else:
        mid = U(-.23,.037)
        under |= stroke(x,y,[(mid-.164,base-platform), (mid,top_ankle),
                             (mid+.135,base-platform)],r*U(.76,1.35))
        if style % 3 != 0:
            under |= stroke(x,y,[(.19,base-platform),(.405,top_ankle+.028),(.467,sole_y)],r)
    if style % 8 != 5:
        pos = U(-.408,-.24)
        under |= stroke(x,y,[(pos-.06,base-platform), (pos,base-platform-U(.021,.091)),
                             (pos+U(.068,.133),base-platform)],r*U(.70,1.39))
    if style % 11 in (1,2,3):
        # Sport sandals and open-toed pumps have a partial upper as well as
        # straps. This piece remains separate from the raised rear quarter.
        t0, t1 = U(-.33,-.17), U(-.078,.14)
        y0 = base-platform-U(.09,.15)
        under |= polygon(x,y,[(t0-.067,base-platform*.50),(t0,y0),
                              (t1,y0-U(.006,.062)),(t1+.043,base-platform-.013)])
    return under,'sandal'


def bag(x,y,rng,style):
    U = rng.uniform
    kind = style % 10
    wide = U(.406,.503) if kind != 7 else U(.253,.392)
    handle = kind in (1,2,3,6,7,8)
    tall_handle = kind in (2,8)
    body_h = U(.27,.551) if kind != 7 else U(.68,.911)
    if kind in (3,6):
        body_h += U(.05,.20)
    handle_h = U(.192,.361) if handle else 0
    if tall_handle:
        handle_h += U(.09,.18)
    if kind == 7:
        handle_h = U(.047,.12)
    bot = min(.508,(body_h+handle_h)*.5)
    top = bot-body_h
    bottom_w = wide*U(.75,1.0)
    upper_w = wide*U(.80,1.02)
    radius = U(.017,.07)
    if kind == 7:
        upper_w *= U(.78,.93)
        radius += .053
    if kind == 5:
        radius += .078
    taper = U(-.035,.035)
    side = [(0,top+taper),(upper_w*.71,top),(upper_w-radius*.48,top+radius*.29),
            (upper_w,top+radius),(wide,(top+bot)*.54),
            (bottom_w,bot-radius),(bottom_w-radius*.51,bot-radius*.21),(bottom_w-radius,bot),(0,bot)]
    fill = polygon(x,y,side+[(-a,b) for a,b in reversed(side[1:-1])])
    if handle:
        join = U(.129,.292) if kind!=7 else U(.071,.143)
        t = top-handle_h
        ww = U(.50,.87)*join
        pts = [(-join,top+.03),(-ww,t+handle_h*.31),(-ww*.61,t+handle_h*.064),
               (U(-.025,.025),t),(ww*.68,t+handle_h*.06),(ww,t+handle_h*.30),(join,top+.03)]
        if tall_handle:
            pts = [(-join,top+.02),(U(-.065,.065),t),(join,top+.02)]
        fill |= stroke(x,y,pts,U(.011,.030))
    if kind in (0,4,5,9):
        fill |= stroke(x,y,[(-wide*.87,bot-.032),(-wide-U(.013,.070),bot-body_h*.20),
                             (-wide-U(.016,.05),top+body_h*.31),(-upper_w*.85,top+.036)],U(.009,.022))
    return fill,'bag'


def template(rng,c,i):
    U = rng.uniform
    extra = (i % 7 == 0)
    dx = U(-.046,.046) if extra else U(-.021,.021)
    dy = U(-.111,.097) if c >= 5 and c != 6 else U(-.020,.020)
    if extra:
        dy += U(-.080,.083)
    angle = U(-.15,.15) if extra else U(-.048,.048)
    x0,y0 = YX[0]-dx,YX[1]-dy
    x = (x0*np.cos(angle)+y0*np.sin(angle))/U(.946,1.062)
    y = (y0*np.cos(angle)-x0*np.sin(angle))/U(.947,1.049)
    if c in (5,7,9):
        m, style = footwear(x,y,rng,c,i)
    elif c == 8:
        m, style = bag(x,y,rng,i)
    elif c == 1:
        m, style = trousers(x,y,rng,i)
    else:
        m, style = top_form(x,y,rng,c,i)
    a = m.reshape(14,3,14,3).mean(axis=(1,3))
    p = np.pad(a, 1, mode='edge')
    blur = (p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])*.25
    a = .82*a+.18*blur
    # The mask is a probabilistic expectation, not a requirement that every
    # foreground pixel be opaque. Styles get a small prior, not a veto.
    strength = {0:.91,1:.91,2:.9,3:.9,4:.9,5:.85,6:.90,7:.9,8:.9,9:.91}[c]
    if i % 6 == 0:
        strength -= .16
    if i % 17 == 0:
        strength -= .10
    prior = -.032 if extra else 0.
    if (c == 0 and style == 'long') or (c == 2 and style in ('cap','drop')):
        prior -= .085
    if c == 6 and style in ('tank','cap'):
        prior -= .077
    if c == 4 and style == 'mid':
        prior -= .046
    return .015+strength*a, prior


# ---- Graph assembly --------------------------------------------------------
class Graph:
    def __init__(self):
        self.layers = []
        self.weights = {}
        self.shapes = {'image': (1,28,28)}

    def layer(self, name, op, ins, result_shape, **kwargs):
        assert name not in self.shapes
        self.layers.append(dict(name=name,op=op,inputs=ins,**kwargs))
        self.shapes[name] = tuple(result_shape)
        return name

    def affine(self,name,op,src,w,b,shape,**settings):
        w,b = np.asarray(w,dtype=np.float32),np.asarray(b,dtype=np.float32)
        self.weights[name+'.W'], self.weights[name+'.b'] = w,b
        return self.layer(name,op,[src],shape,out=len(b),**settings)

    def dense(self,name,src,w,b):
        assert np.shape(w) == (len(b),self.size(src))
        return self.affine(name,'dense',src,w,b,(len(b),))

    def conv(self,name,src,w,b,stride=1,padding=0):
        o,i,k,s = np.shape(w)
        ci,h,ww = self.shapes[src]
        assert i == ci and k == s and len(b) == o
        return self.affine(name,'conv',src,w,b,(o,(h+2*padding-k)//stride+1,(ww+2*padding-k)//stride+1),
                           kernel=[k,k],stride=stride,padding=padding)

    def size(self,src):
        return int(np.prod(self.shapes[src]))

    def pool(self,name,op,src,size,stride):
        c,h,w = self.shapes[src]
        return self.layer(name,op,[src],(c,1+(h-size[0])//stride[0],1+(w-size[1])//stride[1]),
                          size=list(size),stride=list(stride))

    def relu(self,name,src):
        return self.layer(name,'relu',[src],self.shapes[src])

    def norm(self,name,src):
        return self.layer(name,'norm',[src],self.shapes[src])

    def reshape(self,name,src,shape):
        assert int(np.prod(shape)) == self.size(src)
        return self.layer(name,'reshape',[src],shape,shape=list(shape))

    def add(self,name,*sources):
        assert all(self.shapes[s] == self.shapes[sources[0]] for s in sources)
        return self.layer(name,'add',list(sources),self.shapes[sources[0]])

    def save(self,path):
        path.mkdir(parents=True,exist_ok=True)
        (path/'network.json').write_text(json.dumps({'layers':self.layers},indent=2)+'\n')
        np.savez(path/'weights.npz',**self.weights)


class Form:
    """A small affine expression over scalar entries of earlier layers."""
    def __init__(self,bias=0.,terms=None):
        self.bias = float(bias)
        self.terms = {} if terms is None else terms

    @staticmethod
    def make(a):
        return a if isinstance(a,Form) else Form(a)

    def __add__(self,other):
        other = Form.make(other)
        t = self.terms.copy()
        for k,v in other.terms.items():
            t[k] = t.get(k,0.)+v
        return Form(self.bias+other.bias,t)

    __radd__ = __add__

    def __neg__(self):
        return self*-1.

    def __sub__(self,a):
        return self+-Form.make(a)

    def __rsub__(self,a):
        return Form.make(a)+-self

    def __mul__(self,a):
        return Form(self.bias*a,{k:v*a for k,v in self.terms.items()})

    __rmul__ = __mul__


def entry(src,j):
    return Form(0.,{(src,j):1.})


def project(g,name,forms):
    forms = [Form.make(f) for f in forms]
    sources = dict.fromkeys(src for f in forms for src,_ in f.terms)
    if not sources:
        sources = {'image': None}
    rhs = [f.bias for f in forms]
    results = []
    for i,src in enumerate(sources):
        W = np.zeros((len(forms),g.size(src)),dtype=np.float32)
        for row,f in enumerate(forms):
            for (key,col),v in f.terms.items():
                if key == src:
                    W[row,col] = v
        results.append(g.dense(name if len(sources)==1 else f'{name}_part{i}',src,W,rhs if i==0 else np.zeros(len(forms))))
    return results[0] if len(results)==1 else g.add(name,*results)


class Hinges:
    """Compile a batch of comparisons with one dense/ReLU stage."""
    def __init__(self,g,name):
        self.g,self.name,self.items = g,name,[]

    def pos(self,f):
        j = len(self.items)
        self.items.append(f)
        return entry(self.name,j)

    def ramp(self,f,a,b):
        return (self.pos(f-a)-self.pos(f-b))*(1./(b-a))

    def absolute(self,f):
        return self.pos(f)+self.pos(-f)

    def both(self,a,b):
        return a-self.pos(a-b)

    def either(self,a,b):
        return b+self.pos(a-b)

    def done(self):
        self.g.relu(self.name,project(self.g,self.name+'_comparisons',self.items))


def localized(src,channel,y1,y2,x1,x2):
    """Area mean of a rectangle in a 14x14 grid, with fractional boundary bins."""
    b = np.arange(15)/14
    wy = np.maximum(0,np.minimum(b[1:],y2)-np.maximum(b[:-1],y1))
    wx = np.maximum(0,np.minimum(b[1:],x2+.5)-np.maximum(b[:-1],x1+.5))
    W = np.outer(wy,wx).ravel()
    W /= W.sum()
    return Form(0.,{(src,channel*196+j):v for j,v in enumerate(W) if v})


def front_gauss(src,ch,y1,y2,x0,s):
    b = np.arange(15)/14
    uy = np.maximum(0,np.minimum(b[1:],y2)-np.maximum(b[:-1],y1))
    xx = (np.arange(14)+.5)/14-.5
    ux = np.exp(-.5*((xx-x0)/s)**2)
    W = np.outer(uy,ux).ravel(); W /= W.sum()
    return Form(0.,{(src,ch*196+j):v for j,v in enumerate(W) if v>1.e-7})


def feature_maps(g):
    # Soft foreground thresholds share a local-average exposure reference.
    # Broadcast scalar offsets via dense weights, and apply channel clipping
    # as the difference of two ReLUs.
    base = g.pool('exposure_average','avgpool','image',(3,3),(1,1))
    peak = g.pool('exposure_reference','maxpool',base,(26,26),(1,1))
    gates = ((.024,.160,.17),(.062,.225,.25))  # black point, rise, exposure response
    raw = g.conv('mask_raw','image',np.array([1/w for _,w,_ in gates for _ in range(2)]).reshape(4,1,1,1),
                 [-lo/w-j for lo,w,_ in gates for j in (0,1)])
    scale = np.repeat([-q/w for _,w,q in gates for _ in range(2)],784).reshape(-1,1)
    offsets = g.dense('exposure_offsets',peak,scale,np.zeros(3136))
    offsets = g.reshape('exposure_planes',offsets,(4,28,28))
    h = g.relu('mask_thresholds',g.add('mask_signal',raw,offsets))
    masks = g.conv('soft_foregrounds',h,np.array([[1,-1,0,0],[0,0,1,-1]]).reshape(2,4,1,1),[0,0])
    m14 = g.pool('foreground_cells','avgpool',masks,(2,2),(2,2))
    ordinary = g.conv('support_base',m14,np.array([[.80,.20]]).reshape(1,2,1,1),[0])
    soft = g.conv('opaque_for_closing',masks,np.array([[1,0]]).reshape(1,2,1,1),[0])
    dilation = g.pool('neighborhood_high','maxpool',soft,(3,3),(1,1))
    neg = g.conv('negated_high',dilation,np.array([[[[-1]]]]),[0])
    erosion = g.pool('neighborhood_low','maxpool',neg,(3,3),(1,1))
    tight = g.pool('closed_inner_cells','avgpool',erosion,(2,2),(2,2))
    # Interior features use a little hole repair; keep the original at rims.
    ww = np.zeros((196,144)); bb = np.zeros(196)
    for yy in range(14):
        for xx in range(14):
            if 0<yy<13 and 0<xx<13:
                ww[yy*14+xx,(yy-1)*12+xx-1] = -.23
    close = g.dense('closed_interior_pad',tight,ww,bb)
    W = np.zeros((196,196)); np.fill_diagonal(W,.23)
    rim = np.ones((14,14),bool); rim[1:-1,1:-1] = False
    W[~rim.ravel()] = 0
    edge = g.dense('support_rim_pad',m14,np.pad(W,((0,0),(0,196))),bb)
    close = g.reshape('closed_support',g.add('closed_with_rim',close,edge),(1,14,14))
    # In the interior: .77 times raw support plus .23 times closed support.
    weak = g.conv('open_support_part',m14,np.array([[.57,.20]]).reshape(1,2,1,1),[0])
    support = g.add('shape_support',weak,close)

    # Local, signed derivatives on contrast-normalized greyscale. For the
    # appearance rules we keep magnitude, not contrast polarity.
    norm = g.norm('contrast_normalized','image')
    blur = np.array([.25,.5,.25]); d = np.array([-.5,0,.5]); r = np.array([-.5,1,-.5])
    kernels = np.array([np.outer(blur,d),np.outer(d,blur),np.outer(blur,r),
                        np.outer(r,blur),np.outer(d,d)*2.])
    signed = g.conv('local_derivatives',norm,np.concatenate((kernels,-kernels))[:,None],np.zeros(10),padding=1)
    pos = g.relu('rectified_derivatives',signed)
    mag = g.conv('edge_magnitudes',pos,np.concatenate((np.eye(5),np.eye(5)),axis=1)[:,:,None,None],np.zeros(5))
    tex = g.pool('edge_cells','avgpool',mag,(2,2),(2,2))
    grey = g.pool('contrast_cells','avgpool',norm,(2,2),(2,2))
    return support,ordinary,m14,tex,grey,soft


def shape_scores(g,support):
    # Negative squared distance to a soft silhouette factors into a dense
    # dot product and a template-only bias (the input-only term cancels).
    rng = np.random.default_rng(301903)
    count = 704
    strength = 22.
    # Softly deemphasize the outermost cell rows/columns; preprocessing and
    # very slight crop/translation errors often change only the rim.
    border = np.full((14,14),1.0)
    border[[0,-1],:] *= .83
    border[:,[0,-1]] *= .83
    scale = (strength*border/border.sum()).ravel()
    W,B = [],[]
    for c in range(10):
        for i in range(count):
            mask,prior = template(rng,c,i)
            if c in (5,7,9) and (i//2)%2:
                mask = mask[:,::-1]
            a = mask.ravel()
            W.append(2.*scale*a)
            B.append(prior-np.dot(scale,a*a))
    responses = g.dense('silhouette_log_scores',support,W,B)
    grid = g.reshape('silhouette_hypotheses',responses,(1,10,count))
    return g.pool('silhouette_best_by_class','maxpool',grid,(1,count),(1,1))


def footprint_details(g,soft,ordinary):
    # Require actual vertical AND horizontal thickness in the upper quarter.
    # A skinny ankle strap or a high sneaker tongue cannot fill a cuff patch.
    block = g.pool('upper_patch_fill','avgpool',soft,(5,6),(1,1))
    patches = []
    for a,b in ((4,9),(9,14)):
        for side in (0,1):
            x0,x1 = (1,9) if side == 0 else (14,22)
            patches += [entry(block,y*23+x) for y in range(a,b) for x in range(x0,x1)]
    z = project(g,'quarter_patch_proposals',patches)
    c = g.reshape('quarter_patch_grid',z,(1,2,len(patches)//2))
    cuff = g.pool('thickest_upper_patch','maxpool',c,(1,len(patches)//2),(1,1))
    # Use soft morphological gaps, sampled at different spans in both axes.
    W = []
    for dy,dx in ((0,1),(0,2),(0,3),(1,0),(2,0),(3,0),(1,1),(1,-1),(2,2),(2,-2)):
        a = np.zeros((7,7))
        a[3-dy,3-dx],a[3+dy,3+dx] = 1.,-1.
        W.append(a)
        b = np.zeros((7,7))
        b[3-dy,3-dx],b[3,3] = 1.,-1.
        W.append(b)
    p = g.conv('gap_end_contrasts',ordinary,np.array(W)[:,None],np.zeros(len(W)),padding=3)
    h = g.relu('positive_gap_end_imbalance',p)
    A,B = np.zeros((len(W)//2,len(W),1,1)),np.zeros((len(W)//2,len(W),1,1))
    for j in range(len(W)//2):
        A[j,2*j+1,0,0],B[j,2*j,0,0] = 1.,-1.
    left = g.conv('gap_first_end',p,A,np.full(len(A),-.065))
    lower = g.conv('gap_second_end_minimum',h,B,np.zeros(len(B)))
    gap = g.relu('supported_holes',g.add('gap_contrasts',left,lower))
    gh = sum(localized(gap,j,.31,.69,-.37,.37) for j in (0,1,2))*(1/3)
    gv = sum(localized(gap,j,.38,.66,-.375,.375) for j in (3,4,5))*(1/3)
    gd = sum(localized(gap,j,.30,.70,-.385,.385) for j in (6,7,8,9))*.25
    return dict(cuff_patch=entry(cuff,0), lower_patch=entry(cuff,1), mid_gaps=.46*gh+.32*gv+.22*gd)


def footwear_details(g,soft):
    # Partially open shoes can have occupied sides at a given height without
    # a filled middle. Measure that shape at several vertical origins.
    c = g.pool('shoe_column_reach','maxpool',soft,(1,4),(1,4))
    # Average support in a narrow horizontal run, centered at each position.
    smooth = g.conv('continuous_side_support',soft,np.array([[0,0,0],[.25,.5,.25],[0,0,0]])[None,None],
                    [0],padding=1)
    # A persistent concavity is weaker than a full run through the vamp;
    # a strap on either side of a midfoot gap still counts as upper reach.
    v = Hinges(g,'shoe_outline_comparisons')
    apertures = []
    for yy in range(8,21):
        center = sum(entry(c,yy*7+j) for j in (2,3,4))*(1/3)
        sides = (entry(c,yy*7+1)+entry(c,yy*7+5))*.5
        # Local low centers with occupied side material are characteristic of
        # V-straps, open waist and heel architecture, not a continuous vamp.
        apertures.append(v.pos(sides-center-.08))
    v.done()
    sources = {}
    for k,(r0,r1) in enumerate(((11,17),(13,19)),start=2):
        for side in (-1,1):
            xp = np.arange(28)
            center = 13.5+side*6.4
            ws = np.maximum(0,1-np.abs(xp-center)/6.6)
            ws /= ws.sum()*(r1-r0)
            sources[k,side] = Form(0.,{(smooth,yy*28+j):w for yy in range(r0,r1) for j,w in enumerate(ws) if w})
    return dict(apertures=sum(apertures)*(1/9),
                **{f'foot_band{k}{side+1}':v for (k,side),v in sources.items()})


def center_fork(g,soft):
    # Require fabric on BOTH sides in the SAME row around a persistent narrow,
    # mostly vertical split. A comparison of averages alone can mistake two
    # different edges of a tilted dress for its two 'legs'.
    shifts = ((d,s) for d in (3,5,7) for s in (-4,-2,0,2,4))
    ys = np.arange(14,27)
    xx = np.arange(28)
    forms = []
    poses = list(shifts)
    for d,s in poses:
        for y in ys:
            center = 13.5+s+(y-20)*s*.057
            for a,r in ((-d,2.1),(0,1.4),(d,2.1)):
                w = np.maximum(0,1-np.abs(xx-center-a)/r)
                w /= w.sum()
                forms.append(Form(0.,{(soft,y*28+j):v for j,v in enumerate(w) if v}))
    src = project(g,'fork_row_strips',forms)
    hinges = Hinges(g,'fork_row_pairs')
    values = []
    for j,(d,s) in enumerate(poses):
        for i,y in enumerate(ys):
            a,b,c = [entry(src,3*(j*len(ys)+i)+k) for k in range(3)]
            values.append(hinges.both(a,c)-b-.011*abs(s))
    hinges.done()
    diffs = project(g,'fork_row_valleys',values)
    v = g.relu('fork_positive_rows',diffs)
    # Require multiple rows of simultaneous support. A smooth average keeps
    # pinched legs and natural splay from requiring pixel-exact alignment.
    v = g.reshape('fork_layout',v,(1,len(poses),len(ys)))
    spans = g.pool('fork_persistence','avgpool',v,(1,8),(1,3))
    return g.pool('strongest_fork','maxpool',spans,(len(poses),2),(1,1))


def projection_cues(g,mask,ordinary,soft):
    # Row/column maxima measure spatial reach without requiring a broad area
    # to be filled. Comparisons to occupied area distinguish, for example,
    # an open strappy shoe from a boot of the same overall height.
    yr = g.pool('occupied_rows','maxpool',ordinary,(1,14),(1,1))
    xr = g.pool('occupied_columns','maxpool',ordinary,(14,1),(1,1))
    def avg(src, aa,bb):
        return sum(entry(src,j) for j in range(aa,bb))*(1/(bb-aa))
    q = Hinges(g,'projection_thresholds')
    row_reach = sum(q.ramp(entry(yr,j),.13,.64) for j in range(14))*(1/14)
    mass = localized(ordinary,0,0,1,-.5,.5)
    upper_edge = q.ramp(avg(yr,0,2),.22,.70)
    lower_edge = q.ramp(avg(yr,12,14),.23,.70)
    width_reach = q.ramp(.5*(avg(xr,0,2)+avg(xr,12,14)),.20,.68)
    q.done()
    p = Hinges(g,'reach_combinations')
    thin = p.ramp(.46*row_reach-mass,.010,.107)
    stretched = p.both(upper_edge,lower_edge)
    wide = p.both(avg(xr,0,2),avg(xr,12,14))
    broad_bottom = p.both(localized(ordinary,0,.50,.73,-.472,-.325),
                          localized(ordinary,0,.50,.73,.325,.472))
    p.done()
    return dict(thin=thin,stretch=stretched,width=width_reach,broad_bottom=broad_bottom,
                top=upper_edge,bottom=lower_edge,bilateral_width=wide,
                height=row_reach,mass=mass,fork=entry(center_fork(g,soft),0))


def torso_construction(g,grey,tex):
    # A placket or zipper is a near-vertical line, usually near the midline.
    # Stocky lapels diverge from it; a vertical stripe used as decoration is
    # less informative if other comparably strong lines cross the whole body.
    hinges = Hinges(g,'front_line_responses')
    means = []
    for p in (-.065,-.032,0,.032,.065):
        e = [front_gauss(tex,2,a,b,p,.037)-
             .5*(front_gauss(tex,2,a,b,p-.115,.037)+front_gauss(tex,2,a,b,p+.115,.037))
             for a,b in ((.23,.43),(.43,.62),(.62,.80))]
        luma = [front_gauss(grey,0,a,b,p,.031)-
                .5*(front_gauss(grey,0,a,b,p-.096,.047)+front_gauss(grey,0,a,b,p+.096,.047))
                for a,b in ((.23,.43),(.43,.62),(.62,.80))]
        v = sum(hinges.ramp(f,.012,.095) for f in e)*(1/3)
        w = sum(hinges.ramp(f,.092,.410) for f in luma)*(1/3)
        b = sum(hinges.ramp(-f,.117,.500) for f in luma)*(1/3)
        means.append(v+.35*w+.29*b-2.45*abs(p))
    hinges.done()
    t = project(g,'front_line_candidates',means)
    t = g.reshape('front_line_layout',t,(1,1,len(means)))
    return g.pool('front_line_best','maxpool',t,(1,len(means)),(1,1))


def anatomy_scores(g,mask,ordinary,raw,tex,grey,soft):
    def m(y1,y2,x1,x2): return localized(mask,0,y1,y2,x1,x2)
    def side(fun,y1,y2,inner,outer):
        return .5*(fun(y1,y2,inner,outer)+fun(y1,y2,-outer,-inner))
    def G(ch,y1,y2,x,s): return front_gauss(grey,ch,y1,y2,x,s)
    def E(ch,y1,y2,x,s): return front_gauss(tex,ch,y1,y2,x,s)
    def paired(fun,ch,y1,y2,x,s): return (fun(ch,y1,y2,x,s)+fun(ch,y1,y2,-x,s))*.5

    # Project attributes before applying hinges, so the final rule layers work
    # on a compact vector, not on a concatenation of full feature maps.
    s = projection_cues(g,mask,ordinary,soft)
    f = dict(s)
    f.update(footwear_details(g,soft))
    f.update(footprint_details(g,soft,ordinary))
    f['continuous_placket'] = entry(torso_construction(g,grey,tex),0)
    f['leg_waist'] = m(.085,.245,-.111,.111)
    f['skirt_flare'] = side(m,.77,.94,.166,.302)-side(m,.435,.635,.166,.302)
    f['shoulder_drop'] = side(m,.163,.312,.235,.389)-side(m,.49,.704,.235,.389)
    f['high_center'] = m(.034,.139,-.180,.180)
    f['high_wings'] = side(m,.18,.282,.235,.417)
    f['overcoat_half_sleeves'] = side(m,.30,.60,.233,.382)-side(m,.713,.882,.233,.382)
    f['jacket_outline'] = m(.105,.31,-.173,.173)-side(m,.105,.24,.238,.377)
    # A tee has a prominent shoulder-to-torso step on both sides; a
    # dress usually carries more of that width into the skirt or lower body.
    f['tee_width'] = side(m,.20,.36,.235,.372)-side(m,.725,.955,.245,.393)
    f['boxy_top'] = side(m,.45,.75,.151,.235)
    f['shirt_lower_lift'] = m(.851,.991,-.096,.096)-side(m,.851,.991,.135,.229)
    f['neck_side_light'] = paired(G,0,.068,.217,.103,.047)-G(0,.081,.212,0,.047)
    f['neck_open_shadow'] = paired(G,0,.171,.351,.13,.060)-G(0,.155,.311,0,.061)
    f['neck_side_edges'] = paired(E,2,.096,.267,.09,.031)-E(2,.096,.267,0,.05)
    f['strong_print'] = E(1,.25,.53,0,.126)+E(0,.25,.53,0,.126)-E(1,.54,.86,0,.118)-E(0,.54,.86,0,.118)
    f['surface_distribution'] = paired(E,3,.267,.785,.195,.041)-E(3,.267,.785,0,.143)
    f['skirt_center'] = m(.715,.94,-.106,.106)
    f['shoulders'] = side(m,.133,.292,.195,.359)
    f['upper_sleeves'] = side(m,.18,.345,.29,.455)
    f['mid_sleeves'] = side(m,.402,.554,.306,.472)
    f['cuffs'] = side(m,.725,.904,.327,.485)
    f['lower_gap'] = side(m,.633,.884,.258,.354)
    f['shirt_tail'] = (m(.88,.97,-.145,.145)-side(m,.88,.97,.20,.32))-(m(.73,.83,-.145,.145)-side(m,.73,.83,.20,.32))
    f['jacket_crop'] = side(m,.849,.945,.35,.484)-m(.862,.955,-.133,.133)
    f['top_center'] = m(.014,.127,-.098,.098)-side(m,.014,.127,.12,.244)
    f['neck_scoop'] = side(m,.055,.172,.114,.22)-m(.055,.172,-.084,.084)
    f['filled_chest'] = m(.225,.422,-.138,.138)
    f['body_width'] = side(m,.385,.745,.185,.291)
    f['shoe_skew1'] = m(.21,.48,.135,.39)-m(.21,.48,-.39,-.135)
    f['shoe_skew2'] = m(.36,.62,.144,.416)-m(.36,.62,-.416,-.144)
    f['shoe_upperL'] = m(.19,.435,-.36,-.088)
    f['shoe_upperR'] = m(.19,.435,.088,.36)
    f['shoe_upper_center'] = m(.16,.43,-.076,.076)
    f['shoe_front_continuity'] = m(.495,.648,-.30,.30)
    f['heavy_quarterL'] = localized(ordinary,0,.27,.523,-.453,-.14)
    f['heavy_quarterR'] = localized(ordinary,0,.27,.523,.14,.453)
    f['rect_body'] = side(m,.43,.66,.205,.402)
    f['handle_gap'] = side(m,.15,.435,.13,.317)-m(.15,.435,-.089,.089)
    f['bag_body'] = m(.585,.83,-.308,.308)
    f['repair'] = localized(mask,0,.14,.86,-.38,.38)-localized(ordinary,0,.14,.86,-.38,.38)
    f['weak_surface'] = localized(raw,0,.22,.79,-.28,.28)-localized(raw,1,.22,.79,-.28,.28)
    for j,(a,b) in enumerate(((.22,.411),(.405,.603),(.596,.815))):
        f['seam'+str(j)] = E(2,a,b,0,.045)-paired(E,2,a,b,.158,.038)
        f['dark'+str(j)] = paired(G,0,a,b,.135,.037)-G(0,a,b,0,.039)
        f['horizontal'+str(j)] = E(1,a,b,0,.132)
    f['hem_ribbing'] = E(1,.752,.903,0,.125)-.5*(E(1,.575,.733,0,.125)+E(0,.752,.903,0,.106))
    f['neck_horiz'] = E(1,.074,.229,0,.10)-E(0,.074,.229,0,.10)
    f['collar_diagonal'] = paired(E,4,.098,.253,.09,.044)-paired(E,4,.269,.414,.106,.052)
    f['lapel_diagonal'] = paired(E,4,.16,.422,.132,.052)-paired(E,4,.44,.66,.152,.055)
    f['layered_front'] = paired(E,0,.172,.487,.117,.061)-paired(E,0,.462,.777,.117,.061)
    f['broad_yoke'] = paired(E,1,.229,.379,.186,.073)-paired(E,1,.46,.614,.172,.07)
    f['chest_texture'] = E(0,.299,.683,0,.157)+E(1,.299,.683,0,.157)
    src = project(g,'measured_attributes',list(f.values()))
    f = {k:entry(src,j) for j,k in enumerate(f)}

    a = Hinges(g,'anatomy_thresholds')
    split = a.ramp(f['fork'],.14,.62)
    pelvis = a.ramp(f['leg_waist'],.44,.87)
    anchored = a.ramp(f['high_center'],.27,.76)
    drop = a.ramp(f['shoulder_drop'],.045,.34)
    no_flare = 1-a.ramp(f['skirt_flare'],.045,.28)
    high_wings = a.ramp(f['high_wings'],.21,.64)
    edge_width = a.ramp(f['bilateral_width'],.16,.70)
    broad_sole = a.ramp(f['broad_bottom'],.17,.66)
    dense_side = a.ramp(f['mass']-.49*f['height'],.012,.11)
    sleeve_texture = a.ramp(f['surface_distribution'],.011,.087)
    quarter = a.either(f['heavy_quarterL'],f['heavy_quarterR'])
    flare = a.ramp(f['skirt_flare'],.038,.305)
    skirt = a.ramp(f['skirt_center'],.50,.85)
    narrow = 1-a.ramp(f['shoulders'],.16,.53)
    short_sleeve = a.ramp(f['upper_sleeves']-1.02*f['mid_sleeves'],.044,.31)
    overcoat_width = a.ramp(f['overcoat_half_sleeves'],.030,.245)
    collar_light = a.ramp(.69*f['neck_side_light']+.31*f['neck_open_shadow'],.10,.68)
    chest_print = a.ramp(f['strong_print'],.038,.180)
    contrast_placket = a.ramp(f['continuous_placket'],.26,.80)
    collar_grooves = a.ramp(f['neck_side_edges'],.018,.15)
    lapel_shadow = a.ramp(f['neck_open_shadow'],.073,.58)
    neck_bulk = a.ramp(f['jacket_outline'],.24,.70)
    tee_step = a.ramp(f['tee_width'],.057,.46)
    boxy_top = a.ramp(f['boxy_top'],.48,.89)
    curved_hem = a.ramp(f['shirt_lower_lift'],.094,.47)
    foot_aperture = a.ramp(f['apertures'],.012,.106)
    tall_panel = a.ramp(f['cuff_patch'],.35,.76)
    midboot = a.ramp(.62*f['cuff_patch']+.38*f['lower_patch'],.436,.81)
    window_score = a.ramp(f['mid_gaps'],.018,.096)
    horizontal_profile = a.ramp(.62*(1-f['top'])+.38*(1-f['bottom']),.42,.87)
    ankle_by_scale = [a.either(f[f'foot_band{k}0'], f[f'foot_band{k}2']) for k in (2,3)]
    leg_outer = a.ramp(f['upper_sleeves']+.46*f['mid_sleeves'],.18,.58)
    long_sleeve = a.ramp(f['cuffs'],.13,.57)
    broad = a.ramp(f['body_width'],.45,.80)
    thin_sleeve = a.ramp(.63*f['cuffs']+.37*f['mid_sleeves']-f['lower_gap'],.020,.247)
    thick_sleeve = a.ramp(.64*f['lower_gap']+.36*f['cuffs'],.28,.72)
    tail = a.ramp(f['shirt_tail'],.049,.40)
    crop = a.ramp(f['jacket_crop'],.02,.40)
    raised = a.ramp(f['top_center'],.12,.53)
    scoop = a.ramp(f['neck_scoop'],.08,.49)
    chest = a.ramp(f['filled_chest'],.46,.89)
    asym = (.68*a.absolute(f['shoe_skew1'])+.32*a.absolute(f['shoe_skew2']))
    tall_side = a.either(f['shoe_upperL'],f['shoe_upperR'])
    box = a.ramp(f['rect_body'],.39,.82)
    handle = a.ramp(f['handle_gap'],.08,.42)
    bag_body = a.ramp(f['bag_body'],.54,.88)
    holes = a.ramp(f['repair'],.003,.049)
    weak = a.ramp(f['weak_surface'],.09,.24)
    seam = [a.ramp(f['seam'+str(j)],.007,.093) for j in range(3)]
    front_dark = [a.ramp(f['dark'+str(j)],.108,.54) for j in range(3)]
    transverse = a.ramp((f['horizontal0']+f['horizontal1']+f['horizontal2'])*(1/3),.074,.193)
    ribs = a.ramp(f['hem_ribbing'],.008,.144)
    crew = a.ramp(f['neck_horiz'],.005,.139)
    collar = a.ramp(f['collar_diagonal'],.007,.115)
    lapel = a.ramp(f['lapel_diagonal']+.47*f['layered_front'],.026,.156)
    yoke = a.ramp(f['broad_yoke'],.026,.135)
    surface = a.ramp(f['chest_texture'],.14,.38)
    a.done()

    b = Hinges(g,'anatomy_combinations')
    long_seam = b.ramp(.27*seam[0]+.35*seam[1]+.38*seam[2],.31,.73)
    dark_opening = b.ramp(.24*front_dark[0]+.38*front_dark[1]+.38*front_dark[2],.45,.92)
    collar_cues = b.ramp(.24*raised+.18*collar+.12*yoke+.29*collar_light+.17*collar_grooves,.14,.59)
    sleeve_split = b.both(long_sleeve,thin_sleeve)
    layered_lapel = b.both(.52*lapel+.48*lapel_shadow,.43*contrast_placket+.57*neck_bulk)
    no_arms = b.ramp(.67*(1-leg_outer)+.33*(1-flare),.56,.89)
    lower_volume = b.ramp(.40*ankle_by_scale[0]+.60*ankle_by_scale[1],.29,.77)
    tee_outline = b.both(tee_step,boxy_top)
    shirt_shaping = b.both(curved_hem,.5*pelvis+.5*chest)
    pants = b.both(split,.48*pelvis+.52*anchored)
    flared_skirt = b.both(flare,skirt)
    narrow_garment = b.both(narrow,chest)
    straight_drop = b.both(narrow,skirt)
    short_article = 1-b.ramp(f['height'],.59,.91)
    sleeved_top = b.both(.52*drop+.48*high_wings,no_flare)
    tapered_torso = b.both(overcoat_width,.60*anchored+.40*skirt)
    print_not_placket = b.both(chest_print,1-.72*contrast_placket)
    trouser_fallback = b.both(f['stretch'],.67*(1-f['width'])+.33*f['thin'])
    full_length = b.both(anchored,f['stretch'])
    closed_low_shoe = b.both(.54*broad_sole+.46*dense_side,1-.60*f['thin'])
    bag_handle = b.both(handle,bag_body)
    jacket_cuff = b.both(crop,long_sleeve)
    asymmetric = b.ramp(asym,.09,.40)
    boot_quarter = b.ramp(.62*tall_side+.62*quarter-.29*f['shoe_upper_center'],.29,.66)
    shoe_solid = b.ramp(f['shoe_front_continuity'],.40,.81)
    open_profile = b.ramp(.68*f['thin']+.32*weak,.20,.66)
    compact = b.ramp(f['mass']-.50*f['height'],.030,.27)
    b.done()

    d = Hinges(g,'style_conditions')
    # All expressions in a stage depend only on previous stages.
    heavy_closure = d.either(dark_opening,long_seam)
    plain_front = 1-heavy_closure
    handle_with_body = d.both(bag_handle,1-asymmetric)
    open_foot = d.either(open_profile,.66*holes)
    collared_short = d.both(short_sleeve,collar_cues)
    hemmed_sweater = d.both(long_sleeve,.5*ribs+.5*transverse)
    open_sandal = d.both(open_profile,short_article+.29)
    trouser_reach = d.both(f['stretch'],1-f['width'])
    solid_quarter = d.both(boot_quarter,1-.78*open_profile)
    bag_compact = d.both(compact,1-.76*asymmetric)
    flat_trainer = d.both(short_article,closed_low_shoe)
    zipped_coat = d.both(contrast_placket,.54*long_sleeve+.46*tapered_torso)
    collared_blouse = d.both(.64*collar_cues+.36*contrast_placket,.47*full_length+.53*(1-print_not_placket))
    narrow_opening = d.both(midboot,lower_volume)
    midfoot_window = d.both(.27*foot_aperture+.73*window_score,.50*asymmetric+.50*horizontal_profile)
    substantial_quarter = d.both(.64*tall_panel+.36*midboot,1-.70*open_profile)
    low_cut = d.both(horizontal_profile,1-.76*midboot)
    genuine_lapel = d.both(layered_lapel,.47*contrast_placket+.53*leg_outer)
    d.done()
    k = Hinges(g,'construction_cues')
    coat_panel = k.both(heavy_closure,.48*thick_sleeve+.52*lapel)
    ribbed_jersey = k.both(plain_front,.48*crew+.52*hemmed_sweater)
    tailored_coat = k.both(tapered_torso,.37*leg_outer+.32*contrast_placket+.31*genuine_lapel)
    separated_hem = k.both(trouser_fallback,1-.64*skirt)
    k.done()

    # Scores for visible construction cues beyond the silhouettes. The
    # coefficients are cue strengths; e.g. a leg split is class-defining,
    # whereas a cuff or collar is only weak evidence on its own.
    return [
        .42*short_sleeve+.15*scoop+.13*crew-.22*long_sleeve-.34*collared_short-.48*pants-.24*flared_skirt+.14*f['width']+.49*sleeved_top+.23*full_length+.32*print_not_placket-.26*contrast_placket+.37*tee_outline,
        2.04*pants+.14*trouser_reach+.13*f['thin']-.25*asymmetric+.43*separated_hem-.13,
        .29*plain_front+.22*crew+.25*hemmed_sweater+.11*broad+.118*long_sleeve-.23*tail-.18*collar_cues-.13*crop+.28*ribbed_jersey-.26*contrast_placket-.04,
        .58*flared_skirt+.245*narrow_garment+.26*straight_drop+.123*scoop-.63*pants-.14*asymmetric-.23*sleeved_top-.105*f['width']+.27*full_length+.29*no_arms-.34*tailored_coat-.13*contrast_placket+.01,
        .25*heavy_closure+.18*thick_sleeve+.15*jacket_cuff+.15*lapel+.109*raised+.09*broad+.13*long_sleeve-.21*short_sleeve-.14*tail+.28*coat_panel+.10*sleeve_texture+.49*tailored_coat+.35*zipped_coat+.34*genuine_lapel-.14*no_arms-.10,
        .65*open_foot+.37*open_sandal+.23*asymmetric+.10*weak-.21*shoe_solid-.49*substantial_quarter+.24*short_article-.29*bag_compact+.23*edge_width-1.33*full_length+.78*midfoot_window-.34*f['stretch']+.20,
        .57*collar_cues+.46*tail+.21*sleeve_split+.20*long_seam+.39*collared_short+.135*surface-.20*thick_sleeve-.25*hemmed_sweater-.20*flared_skirt-.28*pants+.15*sleeve_texture+.56*collared_blouse-.24*print_not_placket+.29*shirt_shaping-.095,
        .13*asymmetric+.37*shoe_solid-.37*substantial_quarter-.23*holes-.24*open_profile+.32*short_article+.39*flat_trainer+.17*edge_width-.24*bag_compact-.63*full_length-.53*midfoot_window+.31*low_cut+.18,
        .46*handle_with_body+.31*box+.53*bag_compact-.31*asymmetric-.39*pants-.14*f['stretch']+.40*edge_width+.07,
        .18*solid_quarter+.22*asymmetric+.11*shoe_solid-.39*open_foot-.28*short_article+.14*edge_width+.62*substantial_quarter+.23*narrow_opening-.46*midfoot_window-.25*low_cut-.045,
    ]


def main(path):
    g = Graph()
    mask,plain,raw,tex,grey,soft = feature_maps(g)
    best = shape_scores(g,mask)
    cues = anatomy_scores(g,mask,plain,raw,tex,grey,soft)
    project(g,'class_scores',[entry(best,c)+cue for c,cue in enumerate(cues)])
    g.save(path)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
