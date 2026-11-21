"""Data-free, hand-designed small-photo recognition network.

No image files are opened. Every coefficient is a colour transform, spatial
integration weight, derivative, analytic shape, or explicitly written rule.
The template bank is not trained on real or synthetic examples.
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

    def op(self, name, op, source, **settings):
        inputs = source if isinstance(source, list) else [source]
        self.layers.append(dict(name=name, op=op, inputs=inputs, **settings))
        return name

    def conv(self, name, source, w, b=None, stride=1, padding=0):
        w = np.asarray(w, np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.zeros(w.shape[0], np.float32) if b is None else np.asarray(b, np.float32)
        return self.op(name, 'conv', source, out=w.shape[0], kernel=list(w.shape[2:]), stride=stride, padding=padding)

    def dense(self, name, source, w, b=None):
        w = np.asarray(w, np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.zeros(w.shape[0], np.float32) if b is None else np.asarray(b, np.float32)
        return self.op(name, 'dense', source, out=w.shape[0])

    def relu(self, name, source):
        return self.op(name, 'relu', source)

    def pool(self, name, source, size, stride=None, maximum=False):
        if isinstance(size, int):
            size = [size, size]
        if stride is None:
            stride = size
        if isinstance(stride, int):
            stride = [stride, stride]
        return self.op(name, 'maxpool' if maximum else 'avgpool', source, size=size, stride=stride)


def regions(n):
    y, x = np.mgrid[:n, :n].astype(float)
    x = (x + .5) / n
    y = (y + .5) / n
    centre = np.exp(-(((x-.5)/.25)**4 + ((y-.51)/.28)**4))
    r = {
        'all': np.ones((n,n)),
        'top': (y < .3125).astype(float),
        'bottom': (y > .6875).astype(float),
        'centre': centre,
        'outside': 1-centre,
        'upper': np.exp(-(((x-.5)/.30)**4 + ((y-.28)/.18)**4)),
        'lower': np.exp(-(((x-.5)/.33)**4 + ((y-.72)/.18)**4)),
        'sides': (np.abs(x-.5) > .3125).astype(float),
    }
    return {k:v/v.sum() for k,v in r.items()}


def make_colours(g):
    # Capped opponent-colour signals, rather than memorized RGB colours.
    lum = np.array([.299,.587,.114])
    forms = [
        ('light', lum, 0.),
        ('green', 5*np.array([-.5,1,-.5]), -.075),
        ('blue', 4*np.array([-.65,-.35,1]), -.06),
        ('warm', 3*np.array([1,0,-1]), -.075),
        ('red', 4*np.array([1,-.5,-.5]), -.20),
        ('cyan', 4*np.array([-1,.5,.5]), -.16),
        ('bright', 2.5*lum, -1.50),
        ('dark', -2.5*lum, 1.125),
        ('rg', np.array([1,-1,0]), 0.),
        ('gr', np.array([-1,1,0]), 0.),
        ('gb', np.array([0,1,-1]), 0.),
        ('bg', np.array([0,-1,1]), 0.),
        ('rb', np.array([1,0,-1]), 0.),
        ('br', np.array([-1,0,1]), 0.),
    ]
    w = np.stack([v for _,v,_ in forms for _ in range(2)])[:,:,None,None]
    b = np.array([a for _,_,bias in forms for a in (bias,bias-1)])
    node = g.relu('colour_hinges',g.conv('colour_affine','image',w,b))
    w = np.zeros((len(forms)+3,2*len(forms),1,1))
    for i in range(len(forms)):
        w[i,2*i,0,0]=1
        w[i,2*i+1,0,0]=-1
    # Chroma, an approximate brown signal, and neutral brightness.
    for i in range(8,14):
        w[len(forms),2*i,0,0]=1.5
        w[len(forms),2*i+1,0,0]=-1.5
    w[len(forms)+1,2*3,0,0]=1
    w[len(forms)+1,2*3+1,0,0]=-1
    w[len(forms)+1,2*4,0,0]=-.55
    w[len(forms)+1,2*4+1,0,0]=.55
    w[len(forms)+1,2*6,0,0]=-.5
    w[len(forms)+1,2*6+1,0,0]=.5
    w[len(forms)+2,2*6,0,0]=1
    w[len(forms)+2,2*6+1,0,0]=-1
    for i in range(8,14):
        w[len(forms)+2,2*i,0,0]=-.7
        w[len(forms)+2,2*i+1,0,0]=.7
    node=g.relu('colours',g.conv('colour_combinations',node,w))
    return g.pool('colour_grid',node,2), [f[0] for f in forms]+['chroma','brown','white']


def derivatives(g, source, prefix, include_intensity=False):
    # Four directions, equal response to a unit linear ramp.
    s=np.array([1,2,1])/4
    d=np.array([-1,0,1])/2
    gx=s[:,None]*d[None,:]
    gy=d[:,None]*s[None,:]
    lap=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],float)/4
    base=[gx,gy,(gx+gy)/math.sqrt(2),(gx-gy)/math.sqrt(2),lap]
    names=['vertical','horizontal','diagonal_a','diagonal_b','detail']
    if include_intensity:
        ident=np.zeros((3,3)); ident[1,1]=1
        base.append(ident); names.append('intensity')
    w=np.stack([sign*k for k in base for sign in (1,-1)])[:,None]
    node=g.relu(prefix+'_positive',g.conv(prefix+'_derivatives',source,w,padding=1))
    return node,names


def make_lines(g, source):
    # Signed responses to long straight edges: average along the edge,
    # difference across it. These suppress unaligned fine texture.
    n=7
    w=[]
    for theta in [0,math.pi/2,math.pi/4,-math.pi/4]:
        y,x=np.mgrid[-3:4,-3:4]
        across=x*math.cos(theta)+y*math.sin(theta)
        along=-x*math.sin(theta)+y*math.cos(theta)
        k=across*np.exp(-.5*(across/.8)**2-.5*(along/2.5)**2)
        k=k/(np.abs(k).sum()/2)
        w.extend([k,-k])
    return g.relu('line_positive',g.conv('line_filters',source,np.array(w)[:,None],padding=3))


def integrate(g, source, channels, prefix, n, signed_pairs=False):
    masks=regions(n)
    # Padding is a computational convention, not a photographed edge.
    # Ignore every location whose derivative support crosses the image border.
    border = 3 if prefix == 'line_regions' else (1 if signed_pairs else 0)
    if border:
        valid=np.ones((n,n))
        valid[:border]=0; valid[-border:]=0
        valid[:,:border]=0; valid[:,-border:]=0
        masks={name:mask*valid/max((mask*valid).sum(),1e-9) for name,mask in masks.items()}
    names=[]; rows=[]
    for ci,ch in enumerate(channels):
        for region,mask in masks.items():
            row=np.zeros((len(channels)*(2 if signed_pairs else 1),n,n))
            if signed_pairs:
                row[2*ci]=mask; row[2*ci+1]=mask
            else:
                row[ci]=mask
            rows.append(row.reshape(-1)); names.append(ch+'_'+region)
    return g.dense(prefix,source,np.array(rows)), names


# The following geometric primitives specify idealized objects in an abstract
# coordinate system. They are not outlines or measurements of any photograph.
def ellipse(x,y,cx,cy,rx,ry,angle=0):
    co,si=math.cos(angle),math.sin(angle)
    u=(x-cx)*co+(y-cy)*si
    v=-(x-cx)*si+(y-cy)*co
    return ((u/rx)**2+(v/ry)**2 <= 1).astype(float)


def polygon(x,y,points):
    inside=np.zeros(x.shape,bool)
    j=len(points)-1
    for i,(xi,yi) in enumerate(points):
        xj,yj=points[j]
        if yi != yj:
            inside ^= ((yi>y)!=(yj>y)) & (x < (xj-xi)*(y-yi)/(yj-yi)+xi)
        j=i
    return inside.astype(float)


def shape(kind,view,x,y):
    pieces=[]; dark=[]
    def e(*args): pieces.append(ellipse(x,y,*args))
    def p(points): pieces.append(polygon(x,y,points))
    def eye(*args): dark.append(ellipse(x,y,*args))
    if kind==0:                         # aircraft: top/oblique/side
        if view<2:
            e(0,0,.13,.94)
            p([(-.95,.20),(-.95,.03),(-.13,-.24),(.13,-.24),(.95,.03),(.95,.20),(.1,.10),(-.1,.10)])
            p([(-.38,.72),(-.38,.60),(0,.48),(.38,.60),(.38,.72)])
        else:
            e(0,.04,.97,.13)
            p([(-.78,0),(-.88,-.43),(-.70,-.43),(-.50,.03)])
            p([(-.2,.05),(-.40,.35),(.38,.07)])
    elif kind==1:                       # cars: side, oblique, frontal
        if view<2:
            p([(-.94,.02),(-.74,-.08),(-.43,-.44),(.40,-.44),(.70,-.12),(.95,-.01),(.95,.28),(-.94,.28)])
            for a in [-.61,.61]:
                e(a,.29,.19,.21); eye(a,.29,.125,.15)
            if view==1: eye(.02,-.27,.35,.12)
        else:
            p([(-.72,.30),(-.78,-.04),(-.50,-.55),(.5,-.55),(.78,-.04),(.72,.30)])
            for a in [-.59,.59]: eye(a,.30,.13,.18)
            eye(0,-.30,.47,.16)
    elif kind==9:                       # cargo trucks: box, cab, frontal
        if view<2:
            p([(-.96,-.62),(.32,-.62),(.32,.24),(-.96,.24)])
            p([(.35,-.40),(.73,-.40),(.96,-.08),(.96,.25),(.35,.25)])
            for a in [-.68,.65]:
                e(a,.29,.18,.20); eye(a,.29,.12,.14)
            eye(.65,-.23,.16,.12)
        else:
            p([(-.59,-.76),(.59,-.76),(.66,.42),(-.66,.42)])
            eye(0,-.31,.47,.21)
            for a in [-.51,.49]: eye(a,.43,.13,.17)
    elif kind==8:                       # hull, cabin and mast
        p([(-.97,.20),(.97,.20),(.66,.49),(-.61,.49)])
        if view==0:
            p([(-.46,-.14),(.4,-.14),(.55,.20),(-.57,.20)])
            p([(-.17,-.35),(.15,-.35),(.15,-.13),(-.17,-.13)])
        elif view==1:
            p([(-.5,-.2),(.22,-.2),(.35,.20),(-.63,.20)])
            p([(-.15,-.72),(-.10,-.72),(-.10,.20),(-.15,.20)])
        else:
            p([(-.05,-.85),(-.05,.17),(-.63,.17)])
            p([(.02,-.72),(.02,.17),(.62,.17)])
    elif kind==2:                       # perching bird, duck, flying bird
        if view==0:
            e(0,.05,.32,.48,-.35); e(.22,-.44,.23,.23)
            p([(.39,-.47),(.68,-.37),(.39,-.34)])
            p([(-.23,.31),(-.58,.64),(-.10,.42)])
            for a in [-.10,.12]: p([(a,.40),(a+.035,.40),(a+.06,.76),(a,.76)])
        elif view==1:
            e(-.07,.19,.65,.33); e(.43,-.19,.22,.23)
            p([(.35,-.1),(.55,-.1),(.4,.30),(.16,.30)])
            p([(.56,-.23),(.90,-.15),(.56,-.07)])
        else:
            e(0,.02,.17,.46)
            p([(-.06,.08),(-.94,-.4),(-.53,-.46),(0,-.13),(.53,-.46),(.94,-.4),(.06,.08)])
    elif kind in (4,7):                 # deer and horse, side profiles
        deer=kind==4
        e(-.08,-.05,.60,.26 if deer else .33)
        p([(.27,-.08),(.45,-.65),(.66,-.64),(.57,.09)])
        e(.67,-.64,.24,.13 if deer else .17,.25)
        for a in [-.5,-.30,.23,.43]:
            slant=.1 if view==1 else -.04
            p([(a,.11),(a+.065,.11),(a+slant+.05,.78),(a+slant,.78)])
        p([(-.61,-.12),(-.88,.24),(-.80,.27),(-.53,.02)])
        for a in [.52,.69]: e(a,-.84,.055,.14,-.2)
        if view==2 and deer:
            for a in [.49,.64]:
                p([(a,-.70),(a-.18,-1.05),(a-.12,-1.08),(a+.04,-.70)])
    elif kind==6:                       # squat frog, side/front/top
        e(0,.09,.59,.36)
        e(.09,-.20,.43,.27)
        for a in [-.56,.56]:
            e(a,.23,.37,.24,-.4 if a<0 else .4)
            e(a*.95,.47,.30,.09)
        for a in [-.24,.30]:
            e(a,-.39,.16,.16); eye(a,-.41,.075,.075)
        if view==1:
            e(.53,-.07,.28,.19)
        if view==2:
            for a in [-.66,.66]: e(a,-.10,.27,.13)
    elif kind in (3,5):                 # cat/dog: faces and whole animals
        cat=kind==3
        if view<2:
            e(0,.03,.53 if cat else .47,.49)
            if cat:
                p([(-.50,-.18),(-.47,-.72),(-.10,-.39)])
                p([(.50,-.18),(.47,-.72),(.10,-.39)])
            elif view==0:
                e(-.48,-.02,.20,.48,-.2); e(.48,-.02,.20,.48,.2)
            else:
                p([(-.44,-.12),(-.43,-.75),(-.11,-.37)])
                p([(.44,-.12),(.43,-.75),(.11,-.37)])
                e(0,.31,.33,.27)
            eye(-.23,-.07,.095,.065); eye(.23,-.07,.095,.065)
            eye(0,.22,.09,.07 if cat else .10)
        else:
            e(-.12,.02,.59,.30)
            e(.48,-.32,.30,.31)
            for a in [-.50,-.25,.23,.43]: e(a,.38,.08,.35)
            if cat:
                p([(.25,-.48),(.25,-.78),(.48,-.56)])
                p([(.51,-.56),(.72,-.77),(.72,-.42)])
                e(-.70,-.1,.35,.07,.5)
            else:
                e(.34,-.2,.14,.34); e(.71,-.16,.13,.29)
                e(-.72,-.20,.30,.08,.65)
    mask=np.maximum.reduce(pieces)
    detail=np.maximum.reduce(dark) if dark else np.zeros_like(mask)
    return mask,detail


def blur(a,sigma):
    # Formula-only Gaussian integration on a generated coefficient field.
    r=int(math.ceil(3*sigma))
    v=np.exp(-np.arange(-r,r+1,dtype=float)**2/(2*sigma*sigma)); v/=v.sum()
    p=np.pad(a,((r,r),(r,r)))
    b=sum(v[i]*p[i:i+a.shape[0],:] for i in range(2*r+1))
    return sum(v[i]*b[:,i:i+a.shape[1]] for i in range(2*r+1))


def foreground_map(g):
    """Fixed inference-time colour distance to four corner backgrounds.

    The corner colours are activations computed for the current input, never
    persistent weights, prototypes, or summaries of any training image.
    """
    small=g.pool('foreground_rgb','image',4)
    h=8
    corners=[]
    for y in [0,6]:
        for x in [0,6]:
            mask=np.zeros((h,h)); mask[y:y+2,x:x+2]=.25
            corners.append(mask)
    colour_axes=np.array([[.299,.587,.114],[.7,-.7,0],[-.35,-.35,.7]])
    rows=[]
    for corner in corners:
        for axis in colour_axes:
            for sign in [1,-1]:
                for y in range(h):
                    for x in range(h):
                        spatial=-corner.copy(); spatial[y,x]+=1
                        rows.append((sign*axis[:,None,None]*spatial).reshape(-1))
    diff=g.relu('foreground_absolute',g.dense('foreground_differences',small,np.array(rows)))
    diff=g.op('foreground_distance_layout','reshape',diff,shape=[24,h,h])
    w=np.zeros((4,24,1,1))
    for i in range(4): w[i,6*i:6*i+6,0,0]=-1
    dist=g.conv('negative_corner_distances',diff,w)
    dist=g.op('corner_distance_layout','reshape',dist,shape=[1,4,h*h])
    dist=g.pool('closest_background',dist,[4,1],[4,1],maximum=True)
    dist=g.op('background_distance_map','reshape',dist,shape=[1,h,h])
    # A distance ramp rather than a brittle binary segmentation threshold.
    w=np.array([-5.,-5.]).reshape(2,1,1,1)
    dist=g.relu('foreground_mask_hinges',g.conv('foreground_mask_affine',dist,w,[-.30,-1.30]))
    mask=g.conv('foreground_mask',dist,np.array([1.,-1.]).reshape(1,2,1,1))
    norm=g.op('foreground_normalized','norm',mask)
    return mask,norm


def shape_bank(g,source,colour_source,colour_names,foreground):
    n=16; supersample=3
    yy,xx=np.mgrid[:n*supersample,:n*supersample]
    xx=(xx+.5)/(n*supersample); yy=(yy+.5)/(n*supersample)
    kernels=[]
    colour_kernels=[]
    colour_bias=[]
    foreground_kernels=[]
    counts=[]
    colour_index={name:i for i,name in enumerate(colour_names)}
    for kind in range(10):
        start=len(kernels)
        for view in range(3):
            for width in [.55,.78]:
                for dx,dy in [(0,0),(-.10,0),(.10,0),(0,-.10),(0,.10)]:
                    for angle in [-.20,.20]:
                        if kind==0 and view==1: angle+=math.pi/2
                        u=(xx-.5-dx)/(width/2); v=(yy-.5-dy)/(width/2)
                        co,si=math.cos(angle),math.sin(angle)
                        x=u*co+v*si; y=-u*si+v*co
                        m,detail=shape(kind,view,x,y)
                        m=m.reshape(n,supersample,n,supersample).mean((1,3))
                        detail=detail.reshape(n,supersample,n,supersample).mean((1,3))
                        # Equal energy, zero DC silhouette-versus-surround filter.
                        m=blur(m,.45)
                        surround=blur(m,2.2)
                        surround *= m.sum()/max(surround.sum(),1e-9)
                        contrast=m-surround
                        contrast-=contrast.mean()
                        contrast/=max(np.linalg.norm(contrast)*n,1e-9)
                        # An edge-orientation model computed from the analytic
                        # occupancy field, not from a photographed exemplar.
                        gy,gx=np.gradient(m)
                        edges=np.stack([np.abs(gx),np.abs(gy),np.abs(gx+gy)/math.sqrt(2),np.abs(gx-gy)/math.sqrt(2)])
                        edges-=np.stack([blur(e,2.) for e in edges])
                        edges-=edges.mean((1,2),keepdims=True)
                        edges/=max(np.linalg.norm(edges)*n,1e-9)
                        dw=blur(detail,1.2)-detail
                        dw-=dw.mean()
                        dw/=max(np.linalg.norm(dw)*n,1e-9)
                        # Signed inside/outside colour comparison. The zero
                        # spatial sum rejects a uniformly coloured scene.
                        in_region=m/max(m.sum(),1e-9)
                        out_region=np.maximum(blur(m,2.0)-m,0)
                        out_region/=max(out_region.sum(),1e-9)
                        difference=in_region-out_region
                        cw=np.zeros((len(colour_names),n,n))
                        def cue(name,value): cw[colour_index[name]]+=value*difference
                        if kind==0:
                            cue('blue',-.9); cue('chroma',-.15)
                        elif kind in (1,9):
                            cue('red',.7); cue('white',.3); cue('green',-.25)
                        elif kind==2:
                            cue('green',-.5); cue('warm',.25); cue('chroma',.25)
                        elif kind in (3,5):
                            cue('brown',.65); cue('green',-.25)
                        elif kind in (4,7):
                            cue('brown',.75); cue('green',-.75)
                        elif kind==6:
                            cue('green',.8); cue('brown',.2)
                        elif kind==8:
                            cue('blue',-.8); cue('white',.35); cue('green',-.2)
                        coarse=m.reshape(8,2,8,2).mean((1,3))
                        coarse-=coarse.mean()
                        coarse/=max(np.linalg.norm(coarse)*8,1e-9)
                        for polarity in [-1,1]:
                            w=np.zeros((12,n,n))
                            for orient in range(4):
                                w[2*orient]=.30*edges[orient]
                                w[2*orient+1]=.30*edges[orient]
                            signed=polarity*contrast+.15*dw
                            w[10]=signed; w[11]=-signed
                            kernels.append(w.reshape(-1))
                            colour_kernels.append(cw.reshape(-1))
                            colour_bias.append(0.)
                            foreground_kernels.append(coarse.reshape(-1))
        counts.append(len(kernels)-start)
    assert len(set(counts))==1
    bank=g.dense('geometric_matches',source,np.array(kernels))
    cb=g.dense('object_colour_matches',colour_source,np.array(colour_kernels),colour_bias)
    joint=g.op('joint_shape_colour','add',[bank,cb])
    joint=g.op('joint_layout','reshape',joint,shape=[10,1,counts[0]])
    joint=g.pool('joint_best',joint,[1,counts[0]],[1,counts[0]],maximum=True)
    joint=g.op('joint_scores','reshape',joint,shape=[10])
    bank=g.op('geometric_layout','reshape',bank,shape=[10,1,counts[0]])
    bank=g.pool('geometric_best',bank,[1,counts[0]],[1,counts[0]],maximum=True)
    fg=g.dense('foreground_shape_matches',foreground,np.array(foreground_kernels))
    fg=g.op('foreground_shape_layout','reshape',fg,shape=[10,1,counts[0]])
    fg=g.pool('foreground_shape_best',fg,[1,counts[0]],[1,counts[0]],maximum=True)
    fg=g.op('foreground_shape_scores','reshape',fg,shape=[10])
    return g.op('geometric_scores','reshape',bank,shape=[10]),joint,fg


def anatomical_parts(g, normalized):
    """Conjunctions of independently detected parts, not holistic exemplars."""
    small=g.pool('parts_luminance',normalized,2)
    y,x=np.mgrid[-3:4,-3:4]
    kernels=[]
    for sigma in [.55,.9,1.4]:
        inside=np.exp(-(x*x+y*y)/(2*sigma*sigma)); inside/=inside.sum()
        outside=np.exp(-(x*x+y*y)/(2*(sigma*1.8)**2)); outside/=outside.sum()
        dog=outside-inside
        dog/=np.abs(dog).sum()/2
        kernels.extend([dog,-dog])
    node=g.conv('spot_filters',small,np.array(kernels)[:,None],[-.12]*6,padding=3)
    # Bounded evidence: one strong part cannot substitute for its missing mate.
    w=np.zeros((12,6,1,1)); b=np.zeros(12)
    for i in range(6):
        w[2*i,i,0,0]=1.25; w[2*i+1,i,0,0]=1.25; b[2*i+1]=-1
    node=g.relu('spot_hinges',g.conv('spot_cap_affine',node,w,b))
    w=np.zeros((6,12,1,1))
    for i in range(6): w[i,2*i,0,0]=1; w[i,2*i+1,0,0]=-1
    spots=g.conv('bounded_spots',node,w)
    yy,xx=np.mgrid[:16,:16]
    def sample(channel,cx,cy,sigma=.50):
        mask=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma))
        mask/=mask.sum()
        result=np.zeros((6,16,16)); result[channel]=mask
        return result.reshape(-1)
    # Every family uses the same count of placements and scales.
    names=['wheels','front_wheels','eyes','muzzle','frog_eyes','vertical_pair']
    rows=[]; count=None
    for family in names:
        before=len(rows)
        for cx in [5.5,7.5,9.5]:
            for cy in [6.,8.,10.]:
                for size in [3.,4.5,6.]:
                    for scale in [0,1,2]:
                        ch=2*scale
                        if family=='wheels':
                            parts=[sample(ch,cx-size/2,cy+2),sample(ch,cx+size/2,cy+2)]
                        elif family=='front_wheels':
                            parts=[sample(ch,cx-size/2,cy+size/2),sample(ch,cx+size/2,cy+size/2)]
                        elif family=='eyes':
                            parts=[sample(ch,cx-size/3,cy-size/5),sample(ch,cx+size/3,cy-size/5)]
                        elif family=='muzzle':
                            # Dark eyes and nose, each required to be present.
                            parts=[sample(ch,cx-size/3,cy-size/4),sample(ch,cx+size/3,cy-size/4),sample(ch,cx,cy+size/3)]
                        elif family=='frog_eyes':
                            parts=[sample(ch,cx-size/3,cy-size/3),sample(ch,cx+size/3,cy-size/3),sample(ch+1,cx,cy+size/5)]
                        else:
                            parts=[sample(ch,cx,cy-size/2),sample(ch,cx,cy+size/2)]
                        if len(parts)==2:
                            parts.append(parts[1])
                        # Max over negatives is minus the least-supported part.
                        rows.extend([-p for p in parts])
        this=(len(rows)-before)//3
        if count is None: count=this
        assert count==this
    node=g.dense('part_placements',spots,np.array(rows))
    node=g.op('part_constraint_layout','reshape',node,shape=[len(names),count,3])
    node=g.pool('weakest_part',node,[1,3],[1,3],maximum=True)
    w=-np.eye(len(names))[:,:,None,None]
    node=g.conv('part_conjunction',node,w)
    node=g.pool('best_part_arrangement',node,[count,1],[count,1],maximum=True)
    node=g.op('part_scores','reshape',node,shape=[len(names)])
    return node,['p_'+name for name in names]


def main(folder):
    g=Graph()
    colour_grid,colour_names=make_colours(g)
    colour_stats,cn=integrate(g,colour_grid,colour_names,'colour_regions',16)
    gray=g.conv('luminance','image',np.array([.299,.587,.114])[None,:,None,None])
    normal=g.op('contrast_normalized','norm',gray)
    normalized_edges,edge_names=derivatives(g,normal,'normalized',True)
    shape_grid=g.pool('shape_grid',normalized_edges,2)
    fgmask,fgnorm=foreground_map(g)
    geo,joint,fgshape=shape_bank(g,shape_grid,colour_grid,colour_names,fgnorm)
    fgstats,fgnames=integrate(g,fgmask,['foreground'],'foreground_regions',8)
    raw_edges,texture_names=derivatives(g,gray,'raw')
    raw_grid=g.pool('texture_grid',raw_edges,2)
    texture_stats,tn=integrate(g,raw_grid,texture_names,'texture_regions',16,True)
    normalized_grid=g.pool('normalized_grid',normalized_edges,2)
    norm_stats,nn=integrate(g,normalized_grid,edge_names,'contrast_regions',16,True)
    small_gray=g.pool('small_luminance',gray,2)
    lines=make_lines(g,small_gray)
    line_stats,ln=integrate(g,lines,['vertical','horizontal','diagonal_a','diagonal_b'],'line_regions',16,True)

    parts,part_names=anatomical_parts(g,normal)
    # Pack regional measurements without using an unsupported concatenate op.
    sources=[(colour_stats,['c_'+n for n in cn]),(texture_stats,['t_'+n for n in tn]),
             (norm_stats,['n_'+n for n in nn]),(line_stats,['l_'+n for n in ln]),
             (geo,['shape_'+str(i) for i in range(10)]),(parts,part_names),
             (joint,['joint_'+str(i) for i in range(10)]),
             (fgshape,['fgshape_'+str(i) for i in range(10)]),
             (fgstats,fgnames)]
    feature_names=sum([names for _,names in sources],[])
    index={name:i for i,name in enumerate(feature_names)}
    nf=len(index)
    packed=[]; offset=0
    for i,(source,names) in enumerate(sources):
        w=np.zeros((nf,len(names)))
        w[offset:offset+len(names)]=np.eye(len(names))
        packed.append(g.dense('pack_'+str(i),source,w)); offset+=len(names)
    all_features=g.op('measurements','add',packed)

    # A hand-authored rule vocabulary. Hinge rules fire on a cue or a
    # conjunction of cues; these numbers are not calibrated against labels.
    rules=[]
    def rule(name,terms,bias=0):
        row=np.zeros(nf)
        for key,value in terms.items(): row[index[key]]=value
        rules.append((name,row,bias))
    def colour(name,channel,region='all',scale=1):
        rule(name,{'c_'+channel+'_'+region:scale})
    colour('blue_sky','blue','top',3)
    colour('blue_water','blue','bottom',3)
    colour('blue_everywhere','blue','all',3)
    colour('green_scene','green','all',3)
    colour('green_object','green','centre',3)
    colour('green_background','green','outside',3)
    colour('warm_object','warm','centre',2)
    colour('brown_object','brown','centre',3)
    colour('red_object','red','centre',3)
    colour('colourful_object','chroma','centre',1.5)
    colour('white_object','white','centre',2)
    colour('bright_background','bright','outside',2)
    colour('dark_object','dark','centre',2)
    colour('dark_background','dark','outside',2)
    colour('bright_floor','bright','bottom',2)
    # Absence of green, non-saturated backgrounds and smooth sky.
    rule('grey_scene',{'c_chroma_all':-3},.8)
    rule('green_centre_excess',{'c_green_centre':4,'c_green_outside':-3})
    rule('green_around_warm',{'c_green_outside':2.5,'c_brown_centre':2},-.25)
    rule('blue_above_not_below',{'c_blue_top':4,'c_blue_bottom':-3})
    rule('blue_below_not_above',{'c_blue_bottom':4,'c_blue_top':-2})
    rule('red_over_road',{'c_red_centre':3,'c_chroma_bottom':-1,'c_green_bottom':-1},-.04)
    rule('white_over_road',{'c_white_centre':2,'c_chroma_bottom':-1.2,'c_green_bottom':-1},-.04)
    rule('sky_smooth',{'c_blue_top':2,'t_detail_top':-8,'t_horizontal_top':-4},.20)
    rule('white_smooth',{'c_bright_outside':1.5,'t_detail_outside':-8},.15)
    rule('fine_texture',{'t_detail_all':12})
    rule('central_texture',{'t_detail_centre':12})
    rule('background_texture',{'t_detail_outside':12})
    rule('fine_centre_excess',{'t_detail_centre':15,'t_detail_outside':-12})
    rule('clutter_around',{'t_detail_outside':12,'t_detail_centre':-9})
    rule('horizontal_structure',{'l_horizontal_centre':6})
    rule('vertical_structure',{'l_vertical_centre':6})
    rule('straight_structure',{'l_horizontal_centre':3,'l_vertical_centre':3,
                               'l_diagonal_a_centre':-2,'l_diagonal_b_centre':-2})
    rule('horizontal_dominance',{'l_horizontal_centre':6,'l_vertical_centre':-4})
    rule('vertical_dominance',{'l_vertical_centre':6,'l_horizontal_centre':-4})
    rule('water_lines',{'l_horizontal_bottom':6,'l_vertical_bottom':-4})
    rule('low_horizon',{'l_horizontal_lower':5,'l_vertical_lower':-3})
    rule('thin_detail',{'n_detail_centre':2,'n_horizontal_centre':-.6,'n_vertical_centre':-.6})
    rule('object_contrast',{'n_intensity_centre':1,'n_intensity_outside':-.65})
    rule('dark_centre_excess',{'c_dark_centre':3,'c_dark_outside':-2})
    rule('paired_wheels',{'p_wheels':2,'p_vertical_pair':-1.5},-.05)
    rule('frontal_wheels',{'p_front_wheels':2,'p_vertical_pair':-1.5},-.05)
    rule('paired_eyes',{'p_eyes':2,'p_vertical_pair':-1.5},-.05)
    rule('face_parts',{'p_muzzle':2.5,'p_vertical_pair':-1.5},-.05)
    rule('frog_face',{'p_frog_eyes':2.5,'p_vertical_pair':-1.5},-.05)
    rule('wheeled_geometry',{'p_wheels':1.5,'l_horizontal_lower':2,'l_vertical_lower':-1},-.35)
    rule('angular_vehicle',{'l_horizontal_centre':4,'l_vertical_centre':3,'t_detail_centre':-8},-.25)
    rule('long_neck',{'l_vertical_upper':4,'l_vertical_lower':1,'l_horizontal_upper':-2},-.12)
    rule('cat_texture',{'n_detail_centre':2,'n_detail_outside':-1,'c_green_outside':-1},-.20)
    for k in range(10):
        # Relative evidence removes the component shared by all shape banks.
        terms={'shape_'+str(i):(-.7 if i!=k else 6.3) for i in range(10)}
        rule('shape_'+str(k),terms,.15)
        terms={'joint_'+str(i):(-.45 if i!=k else 4.05) for i in range(10)}
        rule('joint_'+str(k),terms,.15)
        terms={'fgshape_'+str(i):(-.4 if i!=k else 3.6) for i in range(10)}
        rule('fgshape_'+str(k),terms,.15)
    # Conjunctive rules: a context cue alone is insufficient to recognize an
    # object, and generic texture alone is insufficient to recognize an animal.
    rule('vehicle_context',{'p_wheels':1.5,'l_horizontal_lower':2,
         'c_red_centre':1,'c_white_centre':.5,'c_green_all':-2,
         't_detail_centre':-3,'p_vertical_pair':-1},-.15)
    rule('box_context',{'l_vertical_centre':3,'l_horizontal_centre':2,
         'c_white_centre':.8,'c_green_all':-2,'t_detail_centre':-4},-.15)
    rule('water_context',{'c_blue_bottom':3,'l_horizontal_bottom':2,
         'l_vertical_bottom':-2,'c_green_all':-2,'t_detail_bottom':-3},-.05)
    rule('green_ungulate',{'c_green_outside':2,'c_brown_centre':1,
         'c_green_centre':-.6,'n_detail_centre':-.4},-.10)
    rule('textured_animal',{'n_detail_centre':1.5,'n_detail_outside':-.5,
         'c_green_outside':-1,'l_horizontal_centre':-1.5,'l_vertical_centre':-1.5},-.05)
    rw=np.stack([v for _,v,_ in rules]); rb=np.array([b for _,_,b in rules])
    active=g.relu('active_rules',g.dense('semantic_rules',all_features,rw,rb))
    ri={name:i for i,(name,_,_) in enumerate(rules)}
    # Class evidence in the required order. These tables express ordinary
    # semantic expectations, not feature statistics from a dataset.
    tables=[
        {'blue_sky':.4,'blue_above_not_below':.25,'sky_smooth':.65,'white_smooth':.55,
         'bright_background':.15,'blue_water':-.30,'green_scene':-.35,
         'central_texture':-.20,'background_texture':-.20,
         'shape_0':.7,'joint_0':.7},
        {'red_over_road':.35,'grey_scene':.15,'horizontal_dominance':.2,
         'blue_water':-.3,'green_scene':-.35,'vertical_dominance':-.20,
         'shape_1':.7,'joint_1':.7,'vehicle_context':.7,'paired_wheels':.25},
        {'green_background':.3,'colourful_object':.15,'clutter_around':.25,
         'blue_sky':.10,'green_centre_excess':-.25,'object_contrast':.15,
         'straight_structure':-.15,'horizontal_dominance':-.10,
         'shape_2':.7,'joint_2':.7,'long_neck':.15},
        {'grey_scene':.15,'fine_centre_excess':.20,'dark_background':.15,
         'blue_sky':-.25,'blue_water':-.3,'green_background':-.20,
         'shape_3':.7,'joint_3':.7,'paired_eyes':.25,'face_parts':.25,
         'cat_texture':.25,'textured_animal':.65},
        {'green_ungulate':.7,'background_texture':.10,'clutter_around':.20,
         'green_centre_excess':-.20,'blue_water':-.4,'red_object':-.15,
         'white_object':-.15,'straight_structure':-.15,
         'shape_4':.7,'joint_4':.7,'long_neck':.15},
        {'brown_object':.20,'fine_centre_excess':.15,'dark_object':.10,
         'blue_sky':-.25,'blue_water':-.3,'green_object':-.25,
         'straight_structure':-.15,'object_contrast':.1,
         'shape_5':.7,'joint_5':.7,'face_parts':.35,'textured_animal':.25},
        {'green_object':.45,'green_centre_excess':.4,'dark_background':.20,
         'brown_object':.15,'central_texture':.15,'blue_sky':-.35,'blue_water':-.4,
         'white_object':-.25,'bright_background':-.15,'straight_structure':-.20,
         'shape_6':.7,'joint_6':.7,'frog_face':.25},
        {'green_ungulate':.55,'dark_centre_excess':.20,'object_contrast':.20,
         'central_texture':-.10,'blue_water':-.35,'green_centre_excess':-.20,
         'straight_structure':-.15,'shape_7':.7,'joint_7':.7,'long_neck':.35},
        {'blue_water':.45,'blue_below_not_above':.35,'blue_sky':.15,
         'water_context':.65,'low_horizon':.10,
         'green_scene':-.45,'warm_object':-.20,'background_texture':-.10,
         'shape_8':.7,'joint_8':.7},
        {'white_over_road':.25,'grey_scene':.20,'box_context':.7,
         'vertical_dominance':.2,'red_over_road':.15,
         'green_scene':-.3,'blue_water':-.3,'warm_object':-.15,
         'shape_9':.7,'joint_9':.7,'paired_wheels':.2,'frontal_wheels':.15},
    ]
    readout=np.zeros((10,len(rules)))
    for k,table in enumerate(tables):
        for name,value in table.items(): readout[k,ri[name]]=value
    # Equal initial class priors. Small offsets only compensate explicitly
    # positive baseline terms in the rules (not observed class frequencies).
    bias=np.array([-.20,-.12,.05,-.08,-.08,-.02,-.10,-.02,-.12,-.12])
    # Background/texture context is weak supporting evidence. Do not let
    # several correlated context rules swamp the actual object models.
    for k in range(10):
        for name in ['shape_'+str(k),'joint_'+str(k)]: readout[k,ri[name]]=0
    context=g.dense('context_scores',active,.22*readout,.22*bias)
    # Three alternative object representations: contrast silhouette,
    # colour-conditioned silhouette, and background-relative foreground.
    # A max prevents three correlated measurements from being triple-counted.
    w=np.zeros((30,len(rules)))
    for k in range(10):
        w[3*k,ri['shape_'+str(k)]]=1
        w[3*k+1,ri['joint_'+str(k)]]=.8
        w[3*k+2,ri['fgshape_'+str(k)]]=1
    evidence=g.dense('object_alternatives',active,w)
    evidence=g.op('object_alternative_layout','reshape',evidence,shape=[10,1,3])
    evidence=g.pool('best_object_evidence',evidence,[1,3],[1,3],maximum=True)
    evidence=g.op('object_evidence','reshape',evidence,shape=[10])
    g.op('class_scores','add',[context,evidence])
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    (folder/'network.json').write_text(json.dumps({'layers':g.layers},indent=2)+'\n')
    np.savez(folder/'weights.npz',**g.weights)
    print('Built',len(g.layers),'layers;',sum(a.size for a in g.weights.values()),'float32 weights.')


if __name__=='__main__':
    main(sys.argv[1])
