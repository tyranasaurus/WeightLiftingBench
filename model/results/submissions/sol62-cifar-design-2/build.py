"""Compile a hand-written visual grammar to the task's network format."""
import json
import sys
from pathlib import Path

import numpy as np


class Net:
    def __init__(self):
        self.layers, self.weights = [], {}
        self.shapes = {'image': (3, 32, 32)}

    def put(self, name, op, source, result_shape, **kw):
        assert name not in self.shapes, name
        ins = [source] if isinstance(source, str) else list(source)
        self.layers.append(dict(name=name, op=op, inputs=ins, **kw))
        self.shapes[name] = result_shape
        return name

    def conv(self, name, source, w, b, stride=1, pad=0):
        o, i, k, k2 = w.shape
        ic, h, q = self.shapes[source]
        assert i == ic and k == k2
        out = self.put(name, 'conv', source, (o, (h+2*pad-k)//stride+1, (q+2*pad-k)//stride+1),
                       out=o, kernel=[k,k], stride=stride, padding=pad)
        self.weights[out+'.W'] = np.asarray(w, dtype=np.float32)
        self.weights[out+'.b'] = np.asarray(b, dtype=np.float32)
        return out

    def dense(self, name, source, w, b):
        w = np.asarray(w, dtype=np.float32)
        assert w.shape[1] == np.prod(self.shapes[source])
        self.weights[name+'.W'] = w
        self.weights[name+'.b'] = np.asarray(b, dtype=np.float32)
        return self.put(name, 'dense', source, (len(w),), out=len(w))

    def relu(self, name, source):
        return self.put(name, 'relu', source, self.shapes[source])

    def pool(self, name, source, size, stride, op='avgpool'):
        c, h, w = self.shapes[source]
        sh, sw = size
        th, tw = stride
        return self.put(name, op, source, (c, (h-sh)//th+1, (w-sw)//tw+1), size=list(size), stride=list(stride))

    def reshape(self, name, source, shape):
        assert np.prod(self.shapes[source]) == np.prod(shape)
        return self.put(name, 'reshape', source, tuple(shape), shape=list(shape))


class E:
    """An affine expression over named nonnegative signals."""
    def __init__(self, v=None, b=0.):
        self.v = {} if v is None else v
        self.b = b

    def __add__(self, other):
        if not isinstance(other, E):
            other = E(b=other)
        v = dict(self.v)
        for k, x in other.v.items():
            v[k] = v.get(k, 0.) + x
        return E(v, self.b + other.b)

    __radd__ = __add__

    def __neg__(self):
        return self * -1.

    def __sub__(self, x):
        return self + (-x)

    def __rsub__(self, x):
        return (-self) + x

    def __mul__(self, x):
        return E({k: v*x for k, v in self.v.items()}, self.b*x)

    __rmul__ = __mul__

    def __truediv__(self, x):
        return self * (1./x)


def channels(names):
    return {k: E({k: 1.}) for k in names}


def lin(net, name, src, srcnames, exprs, relu=True):
    pos = {k: i for i, k in enumerate(srcnames)}
    w = np.zeros((len(exprs), len(pos)))
    b = []
    for i, e in enumerate(exprs.values()):
        for k, v in e.v.items():
            w[i, pos[k]] = v
        b.append(e.b)
    if len(net.shapes[src]) == 3:
        out = net.conv(name, src, w[:, :, None, None], b)
    else:
        out = net.dense(name, src, w, b)
    return (net.relu(name+'_positive', out) if relu else out), list(exprs)


def transition(net, name, src, srcnames, keep, extra):
    a = channels(srcnames)
    return lin(net, name, src, srcnames, {**{k: a[k] for k in keep}, **extra})


LUM = np.array([.299, .587, .114])


def local_appearance(net, rgb):
    # Signed differences on the pooled grid, plus unpooled detail within 2x2 blocks.
    w, names = [], []
    offsets = {'ex': (0,1), 'ey': (1,0), 'ed': (1,1), 'ea': (1,-1)}
    for i, c in enumerate('rgb'):
        a = np.zeros((3,3,3)); a[i,1,1] = 1
        w.append(a); names.append(c)
    f = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16.
    for key, (dy,dx) in offsets.items():
        a = np.zeros((3,3)); a[1+dy,1+dx] = .5; a[1-dy,1-dx] = -.5
        for sign, p in (('p',1),('n',-1)):
            w.append(p * LUM[:,None,None] * a); names.append(key+sign)
    for sign, p in (('p',1),('n',-1)):
        a = -f.copy(); a[1,1] += 1
        w.append(p * LUM[:,None,None] * a); names.append('spot'+sign)
    # A chromatic difference separates foliage texture from pure brightness texture.
    for sign, p in (('p',1),('n',-1)):
        a = -f.copy(); a[1,1] += 1
        w.append(p * np.array([-.5, 1., -.5])[:,None,None] * a); names.append('ch'+sign)
    a0 = net.relu('local_rectified', net.conv('local_filters', rgb, np.array(w), np.zeros(len(w)), pad=1))
    a = channels(names)
    r, g, b = (a[k] for k in 'rgb')
    y = .299*r + .587*g + .114*b
    h = a['eyp'] + a['eyn']
    v = a['exp'] + a['exn']
    d1 = a['edp'] + a['edn']
    d2 = a['eap'] + a['ean']
    fine = a['spotp'] + a['spotn']
    chrom = a['chp'] + a['chn']
    raw = dict(lum=y, low=.32-y, dark=.20-y, deep=.11-y, high=y-.75,
               white=y-.80, red=r-g-.04, red_b=r-b-.07, clean_r=r-1.34*g-.035,
               blue=b-r-.025, blue_g=b-g-.018, blue_v=g-r-.012, blue_nonred=.07+r-b,
               brown=r-b-.065, ochre=g-b-.025, chest=r-g-.045,
               green=g-r-.013, green_b=g-b-.02, green_v=g-.14,
               olive=r+g-2*b-.12, olive_r=.15+g-r, olive_v=y-.12,
               sepia=r-b-.042, sepia_g=.13-(r-g), sepia_rgb=.76-y,
               open=y-.36, bright=y-.53, skybrite=y-.37,
               earth_hi=.85-y, earth_lo=y-.095,
               nrg=r-g, ngr=g-r, ngb=g-b, nbg=b-g,
               hz=h, vt=v, dg1=d1, dg2=d2, lg=fine, chrom=chrom,
               smooth=.082-1.15*fine-.38*(v+h), textured=fine-.028,
               structure=(h+v+d1+d2)/4-.045,
               hline=h-.5*v-.025, vline=v-.5*h-.025,
               diag=(d1+d2)/2-.54*(h+v)-.018,
               blackedge=fine+.32*(h+v)-.072,
               edge=h+v, brightspot=a['spotp'], darkspot=a['spotn'])
    a1, n1 = lin(net, 'local_colour_and_texture', a0, names, raw)
    q = channels(n1)
    sat = q['nrg']+q['ngr']+q['ngb']+q['nbg']
    # a - relu(a-b) is min(a,b). Use this for soft pixel-set intersections.
    pairs = {
        'fol': (q['green'], q['green_b']), 'blu': (q['blue'], q['blue_g']+.55*q['blue_v']),
        'brn': (q['brown'], q['ochre']), 'olv': (.52*q['olive'], q['olive_r']),
        'neutral': (q['smooth'], .10-.58*sat),
        'orange': (q['clean_r'], q['red_b']), 'tan': (q['sepia'], q['sepia_g']),
        'cyan': (q['blue_v'], q['blue_g']+.027),
        'soil': (q['ochre'], q['olive_r']), 'grey': (.105-.65*sat,E(b=.105)),
        'flat': (q['smooth'], E(b=.077)), 'darkflat': (q['low'], q['smooth']),
        'flatopen': (q['smooth'], .7*q['open']),
        'inkwhite': (q['white'], .26-.75*sat),
        'black': (q['dark'], .24-.70*sat),
        'grassbright': (q['green'], .64*q['green_v']),
    }
    a2, n2 = transition(net, 'colour_intersection_hinges', a1, n1, n1,
                        {k+'_diff': u-v for k,(u,v) in pairs.items()})
    z = channels(n2)
    a3, n3 = transition(net, 'pixel_materials', a2, n2, n1,
                        {k: u-z[k+'_diff'] for k,(u,v) in pairs.items()})
    z = channels(n3)
    keep = ['lum','low','deep','bright','grey','flat','neutral','darkflat','flatopen','inkwhite','black',
            'hz','vt','lg','chrom','hline','vline','diag','textured','structure','blackedge','brightspot','darkspot']
    bases = {
        # Sky is often hazy or white; the blue channel provides more specific evidence.
        'skyblue': (z['blu'], 1.70*z['smooth']), 'palesky': (z['flatopen'], z['grey']+.33*z['blu']),
        'grass': (z['fol']+.27*z['olv'], .66*z['olive_v']),
        'leaf': (z['fol']+.22*z['olv'], 1.30*z['textured']+.5*z['chrom']),
        'mottle': (z['olv']+.41*z['brn']+.7*z['fol'], 1.28*z['textured']+.22*z['structure']),
        'earth': (z['brn']+.23*z['tan'], .56*z['earth_lo']),
        'furwarm': (z['tan']+.35*z['chest'], .62*z['sepia_rgb']),
        'chestnut': (z['chest'], .76*z['ochre']),
        'paintred': (z['orange'], E(b=.39)),
        'strongblue': (z['blu'], E(b=.31)),
        'bluewater': (z['blu']+.40*z['cyan'], 1.65*z['smooth']+.54*z['textured']),
        'bluegrain': (z['blu']+.35*z['cyan'], 1.3*z['textured']+.24*z['hz']),
        'pavement': (z['grey'], z['flat']+.11*z['open']),
        'linear': (z['hline']+z['vline'], z['blackedge']),
        'olivecol': (z['olv'], .6*z['olive_v']),
    }
    a4, n4 = transition(net, 'material_condition_hinges', a3, n3, n3,
                        {k+'_diff': u-v for k,(u,v) in bases.items()})
    z = channels(n4)
    return transition(net, 'materials_and_edges', a4, n4, keep,
                      {k: u-z[k+'_diff'] for k,(u,v) in bases.items()})


def yyxx(k):
    y,x = np.mgrid[:k,:k].astype(float)
    return y-k//2,x-k//2


def gauss(y,x,cy,cx,sy,sx,theta=0.):
    dy,dx=y-cy,x-cx
    ar = dx*np.cos(theta)+dy*np.sin(theta)
    cr = dy*np.cos(theta)-dx*np.sin(theta)
    return np.exp(-.5*((ar/sx)**2+(cr/sy)**2))


def unit(a):
    return a / a.sum()


def diff(p,q):
    return unit(p)-unit(q)


def gray_geometry(net, rgb):
    # First-order and line-likeness filters on a gently downsampled luminance plane.
    gray = net.conv('pooled_luminance', rgb, LUM[None,:,None,None], [0.])
    k=7; y,x=yyxx(k)
    A=lambda cy,cx,sy,sx,t=0.: gauss(y,x,cy,cx,sy,sx,t)
    w, names, bias = [], [], []
    def add(n,a,b=0.):
        w.append(a); names.append(n); bias.append(b)
    add('luma', 1.*((y==0)&(x==0)))
    for t in range(6):
        theta = t*np.pi/6
        sx,sy = -np.sin(theta),np.cos(theta)
        for d in (1,-1):
            p= A(d*.85*sy,d*.85*sx,.47,1.32,theta)
            q= A(-d*.85*sy,-d*.85*sx,.47,1.32,theta)
            add('step%d%s'%(t,'p' if d==1 else 'n'),diff(p,q),-.022)
        mid=A(0,0,.46,1.32,theta)
        outer=A(1.23*sy,1.23*sx,.59,1.38,theta)+A(-1.23*sy,-1.23*sx,.59,1.38,theta)
        add('ridge%d'%t,diff(mid,outer),-.026)
        add('valley%d'%t,diff(outer,mid),-.026)
    for i,s in enumerate((.60,1.03)):
        p=A(0,0,s,s)
        rr=np.sqrt(x*x+y*y)
        ring=np.exp(-.5*((rr-(1.66 if i==0 else 2.52))/.46)**2)
        b=diff(ring,p)
        add('blot%d'%i,b,-.027); add('light%d'%i,-b,-.031)
    ref=A(0,0,.86,.86)
    for sig in (-1,1):
        # Wheel cue: the rubber is darker than the hub and the body above it.
        rr=np.sqrt((x/.93)**2+(y/1.02)**2)
        ring=np.exp(-.5*((rr-1.30)/.39)**2)
        above=A(-2.25,0,.51,1.45); below=A(2.1,0,.55,1.75)
        add('tire%d'%sig, .57*diff(ref,ring)+.43*diff(above if sig>0 else below,ring), -.023)
    raw=net.relu('rectified_geometry_parts',net.conv('geometry_part_filters',gray,np.array(w)[:,None],bias,pad=3))
    q=channels(names)
    keep=['luma']
    exprs=dict(
        horiz=q['step0p']+q['step0n'], vert=q['step3p']+q['step3n'],
        hpos=q['step0p'], hneg=q['step0n'], vpos=q['step3p'], vneg=q['step3n'],
        dstep1=(q['step1p']+q['step1n']+q['step2p']+q['step2n'])/2,
        dstep2=(q['step4p']+q['step4n']+q['step5p']+q['step5n'])/2,
        hbar=q['ridge0']+q['valley0'], vbar=q['ridge3']+q['valley3'],
        thin_h=q['valley0'], thin_v=q['valley3'], light_h=q['ridge0'], light_v=q['ridge3'],
        # Two diagonal directions for paired ears, legs, wings, and sloping windscreens.
        slant1=.55*(q['ridge1']+q['valley1'])+.45*(q['ridge2']+q['valley2']),
        slant2=.55*(q['ridge5']+q['valley5'])+.45*(q['ridge4']+q['valley4']),
        r1=q['valley1']+q['valley2'],r2=q['valley4']+q['valley5'],
        eyespot=q['blot0'], nose=q['blot1'], cheek=q['light0'], largebright=q['light1'],
        tire=.74*q['tire1']+.26*q['tire-1'],
        grain=.5*(q['light0']+q['blot0'])+.2*(q['light1']+q['blot1']),
    )
    return transition(net,'geometric_primitives',raw,names,keep,exprs)


def geometry_templates(net, parts, pnames):
    # Each row is a soft part arrangement, not a picture of an object.
    # Width, vertical offsets and reflection are generated from general geometric ratios.
    k=11; y,x=yyxx(k)
    fs, names, bs, families = [], [], [], {}
    pos={n:i for i,n in enumerate(pnames)}
    def add(fam, taps, scale=1., mirror=1, bias=0.):
        idx=len(fs); w=np.zeros((len(pos),k,k),dtype=np.float32)
        for n, amp, yy, xx, sy, sx in taps:
            xx*=mirror
            if mirror<0:
                n={'vpos':'vneg','vneg':'vpos','slant1':'slant2','slant2':'slant1',
                   'dstep1':'dstep2','dstep2':'dstep1','r1':'r2','r2':'r1'}.get(n,n)
            p=unit(gauss(y,x,yy*scale,xx*scale,max(.34,sy*scale),max(.34,sx*scale)))
            w[pos[n]] += amp*p
        fs.append(w); bs.append(bias); names.append(fam+str(idx)); families.setdefault(fam,[]).append(idx)
    for s in (.68,1.,1.28):
        for m in (1,-1):
            add('car', [
                ('tire',.33,2.3,-2.95,.48,.54),('tire',.33,2.3,2.95,.48,.54),
                ('horiz',.27,.90,0,.52,2.55),('horiz',.24,-1.2,0,.58,2.2),
                ('dstep1',.18,-.62,-2.65,.71,.66),('dstep2',.18,-.62,2.65,.71,.66),
                ('grain',-.20,3.1,0,.54,2.3),('vert',-.16,-2.8,0,.70,2.1),
            ],s,m,-.045)
            add('truck', [
                ('vert',.28,-.62,-2.85,1.62,.58),('vert',.21,-.62,2.85,1.66,.62),
                ('horiz',.32,-2.45,-.1,.55,2.12),('horiz',.22,1.45,0,.53,2.45),
                ('tire',.30,2.7,2.1,.56,.66),('tire',.24,2.7,-2.25,.54,.72),
                ('grain',-.33,-.65,-.63,.8,1.43),('dstep2',.12,-.13,2.72,.65,.8),
            ],s,m,-.053)
            add('cab', [
                ('horiz',.30,-2.1,0,.61,1.9),('horiz',.34,1.5,0,.66,1.9),
                ('hbar',.22,.15,0,.68,1.45),('vert',.20,-.33,2.30,1.35,.47),
                ('vert',.20,-.33,-2.30,1.35,.47),('tire',.18,2.6,-1.8,.56,.83),
                ('tire',.18,2.6,1.8,.56,.83),('grain',-.17,2.8,0,.5,.81),
            ],s,m,-.056)
            add('horse', [
                ('horiz',.31,-1.15,-.4,.72,2.38),('hbar',.18,-.15,-.47,.9,1.67),
                ('vbar',.29,1.91,-2.0,1.18,.57),('vbar',.28,1.91,1.44,1.18,.57),
                ('dstep2',.25,-1.68,1.97,.89,.61),('horiz',.18,-2.63,2.73,.61,.66),
                ('vbar',-.25,2.29,-.38,.72,.76),('horiz',-.14,2.29,-.38,.72,.76),
            ],s,m,-.048)
            add('deer', [
                ('horiz',.27,-.78,-.33,.65,1.97),('thin_v',.30,1.64,-1.65,.89,.47),
                ('thin_v',.31,1.69,1.36,.86,.47),('vbar',.21,-1.72,1.97,.70,.59),
                ('slant2',.17,-2.76,2.04,.68,.72),('vbar',-.24,1.70,-.12,.76,.64),
                ('thin_h',-.12,1.42,0,.67,1.12),
            ],s,m,-.031)
            add('air', [
                ('hbar',.38,-.1,-.48,.65,2.55),('slant1',.25,-1.25,.10,.71,1.10),
                ('slant1',.25,1.30,-.39,.67,1.04),('vbar',.19,-.96,-2.85,.71,.54),
                ('horiz',-.23,2.24,0,.83,2.14),('grain',-.22,-2.85,0,.69,2.72),
            ],s,m,-.037)
            add('air', [
                ('slant1',.31,1.05,1.93,.85,1.05),('slant2',.31,1.05,-1.93,.85,1.05),
                ('vbar',.31,-.52,0,1.47,.51),('hbar',.20,.31,0,.68,2.01),
                ('grain',-.25,-1.75,-2.65,.74,.87),('grain',-.25,-1.75,2.65,.74,.87),
            ],s,m,-.035)
            add('ship', [
                ('hbar',.37,1.34,-.12,.59,2.68),('horiz',.31,.52,0,.69,2.90),
                ('slant2',.21,1.02,2.39,.61,.77),('vert',.19,-1.31,-.41,1.13,1.07),
                ('horiz',.13,-1.88,-.22,.68,1.26),('vert',-.23,3.11,0,.78,2.60),
                ('grain',-.13,3.05,0,.74,2.15),
            ],s,m,-.039)
            add('bird', [
                ('hbar',.28,.63,-.48,.82,1.62),('eyespot',.19,-1.25,1.27,.68,.69),
                ('slant2',.23,-.42,.79,.96,.77),('slant1',.21,1.39,-1.75,.78,.77),
                ('thin_v',.12,1.63,.53,.79,.53),('hbar',.15,2.52,0,.58,2.08),
                ('grain',-.23,-1.91,-1.68,.79,1.43),
            ],s,m,-.028)
        add('cat', [
            ('eyespot',.31,-.33,-1.50,.47,.48),('eyespot',.31,-.33,1.50,.47,.48),
            ('nose',.21,1.02,0,.49,.61),('cheek',.15,.91,-.96,.64,.61),('cheek',.15,.91,.96,.64,.61),
            ('dstep1',.20,-1.90,-1.96,.68,.66),('dstep2',.20,-1.90,1.96,.68,.66),
            ('eyespot',-.21,-.38,0,.56,.56),('grain',-.19,-2.63,0,.65,.86),
            ('vert',.09,-.17,-2.37,.87,.66),('vert',.09,-.17,2.37,.87,.66),
        ],s,1,-.042)
        add('dog', [
            ('eyespot',.21,-.76,-1.47,.55,.52),('eyespot',.21,-.76,1.47,.55,.52),
            ('nose',.49,.99,0,.58,.70),('largebright',.12,.78,-1.2,.71,.69),('largebright',.12,.78,1.2,.71,.69),
            ('vert',.18,-.37,-2.65,1.00,.73),('vert',.18,-.37,2.65,1.00,.73),
            ('nose',-.26,-.77,0,.60,.77),('grain',-.14,2.64,0,.69,1.26),
        ],s,1,-.047)
        add('frog', [
            ('eyespot',.25,-.98,-1.84,.53,.60),('eyespot',.25,-.98,1.84,.53,.60),
            ('horiz',.37,.58,0,.72,1.94),('hbar',.20,.57,0,.62,1.64),
            ('slant2',.18,1.05,-2.24,.74,.68),('slant1',.18,1.05,2.24,.74,.68),
            ('vbar',-.21,.82,0,.78,.74),('grain',-.19,-.90,0,.63,.90),
        ],s,1,-.042)
        add('wheelpair',[('tire',.62,0,-2.2,.46,.56),('tire',.62,0,2.2,.46,.56),
                         ('tire',-.36,0,0,.52,.93),('horiz',.26,-1.45,0,.53,2.01),
                         ('grain',-.20,1.2,0,.57,1.85)],s,1,-.029)
        add('horizon',[('horiz',.89,0,0,.43,3.18),('hbar',.25,0,0,.51,3.15),
                        ('vert',-.42,0,0,1.22,3.1),('grain',-.31,-1.29,0,.69,3.13)],s,1,-.038)
    raw=net.relu('positive_part_arrangements',net.conv('part_arrangements',parts,np.array(fs),bs,pad=5))
    # Extract a common central region with a 1x1 selector before max pooling.
    # The taper downweights clutter at the frame margin and reduces padding artifacts.
    # Use valid 6x6 pooling with nonoverlap, on positions whose centers are 3..12.
    maxima=net.pool('quadrant_template_maxima',raw,(6,6),(5,5),'maxpool')
    av=net.pool('template_support',raw,(8,8),(8,8))
    # Each channel can occur at multiple positions; central window is the most valuable.
    w=[]; keys=[]
    for i, nm in enumerate(names):
        for mode in ('center','wide','low'):
            if mode=='center':
                v=np.zeros((len(names),3,3)); v[i,1,1]=17.
            elif mode=='wide':
                v=np.zeros((len(names),3,3)); v[i] = np.array([[.55,1.15,.55],[1.1,8.2,1.1],[.54,1.25,.54]])
            else:
                v=np.zeros((len(names),3,3)); v[i] = np.array([[0,0,0],[.18,3.4,.18],[1.53,6.1,1.53]])
            keys.append(nm+'_'+mode); w.append(v.ravel())
    ds=net.dense('arrangement_region_evidence',maxima,np.array(w),np.zeros(len(w)))
    aw=[]; an=[]
    for fam, indices in families.items():
        v=np.zeros((len(names),2,2)); v[indices]=31./(len(indices)*4)
        aw.append(v.ravel()); an.append(fam+'_support')
    support=net.dense('arrangement_average_evidence',av,np.array(aw),np.zeros(len(aw)))
    return ds, keys, support, an, {k:[names[i] for i in v] for k,v in families.items()}


def append_vectors(net, sources):
    total=sum(len(keys) for _,keys in sources)
    out=[]; keys=[]; start=0
    for i,(src,kk) in enumerate(sources):
        w=np.zeros((total,len(kk))); w[start:start+len(kk)]=np.eye(len(kk)); start+=len(kk)
        out.append(net.dense('embed_vector_'+str(i),src,w,np.zeros(total)))
        keys+=kk
    assert len(keys)==len(set(keys))
    return net.put('joint_cues','add',out,(total,)),keys


def family_max(net, src, names, families):
    # Exact maxima over reflected/rescaled object arrangements (pairwise ReLU reduction).
    groups = {k+'_'+mode:[n+'_'+mode for n in ns]
              for k,ns in families.items() for mode in ('center','wide','low')}
    original = set(n for a in groups.values() for n in a)
    static = [k for k in names if k not in original]
    for j in range(4):
        a=channels(names); e={k:a[k] for k in names}; nxt={}
        for family, ns in groups.items():
            pending=[]
            for i in range(0,len(ns)-1,2):
                t=family+'_reduce_%d_%d'%(j,i)
                e[t+'_h']=a[ns[i]]-a[ns[i+1]]
                pending.append((t,ns[i+1]))
            nxt[family]=(pending,ns[-1:] if len(ns)%2 else [])
        src,names=lin(net,'max_hinges_'+str(j),src,names,e)
        a=channels(names); e={k:a[k] for k in static}
        groups={}
        for family,(pending,extra) in nxt.items():
            groups[family]=[]
            for t,b in pending:
                e[t]=a[b]+a[t+'_h']; groups[family].append(t)
            for k in extra:
                e[k]=a[k]; groups[family].append(k)
        src,names=lin(net,'max_reduce_'+str(j),src,names,e)
    assert all(len(ns)==1 for ns in groups.values())
    a=channels(names)
    return lin(net,'semantic_cues',src,names,{**{k:a[k] for k in static},
                                             **{k:a[ns[0]] for k,ns in groups.items()}})


def bounded_maps(net, src, names, name, exprs):
    # clip(z, 0, 1) = relu(z) - relu(z-1), for density-like maps.
    base, bn = lin(net, name+'_thresholds', src, names,
                   {**exprs, **{k+'_hi':v-1. for k,v in exprs.items()}})
    a=channels(bn)
    return lin(net,name,base,bn,{k:a[k]-a[k+'_hi'] for k in exprs})


def pixel_context(net, src, names):
    a=channels(names)
    v=lambda k: a[k]
    p,pn=bounded_maps(net,src,names,'soft_semantic_pixels',{
        'blue': 15*v('skyblue')-.19,
        'blue_water': 15*v('bluewater')-.27,
        'water_texture': 30*v('bluegrain')-.19,
        'open_light': 13*v('palesky')-.30,
        'open_flat': 15*v('flat')-.23,
        'green': 14*v('grass')-.22,
        'leaves': 24*v('leaf')-.20,
        'mottled': 23*v('mottle')-.27,
        'olive': 12*v('olivecol')-.35,
        'earth': 13*v('earth')-.28,
        'bronze': 14*v('furwarm')-.30,
        'chestnut': 17*v('chestnut')-.20,
        'red': 13*v('paintred')-.26,
        'white': 21*v('inkwhite')-.20,
        'black': 18*v('black')-.28,
        'road': 13*v('pavement')-.28,
        'bright': 6*v('bright')-.06,
        'fine': 22*v('lg')-.45,
        'busy': 13*v('hz')+13*v('vt')-1.08,
        'chroma_texture': 29*v('chrom')-.29,
        'round_dark': 19*v('darkspot')-.37,
        'bright_dot': 20*v('brightspot')-.45,
        'horizontal': 16*v('hline')-.34,
        'vertical': 16*v('vline')-.34,
        'diagonal': 22*v('diag')-.26,
    })
    regions={k:'A T B M O U L' for k in pn}
    return context_windows(net,'scene_fractions',p,pn,regions)


def context_windows(net,name,src,names,regions):
    _,h,w=net.shapes[src]
    y,x=np.mgrid[:h,:w].astype(float)
    x=(x+.5)*2/w-1.; y=(y+.5)*2/h-1.
    g=lambda yc,xc,ys,xs: np.exp(-.5*(((y-yc)/ys)**2+((x-xc)/xs)**2))
    rim=g(0,0,.91,.91)*((abs(x)**2+abs(y)**2)*.62)
    m=dict(A=np.ones_like(x),T=g(-.69,0,.26,.76),B=g(.68,0,.24,.78),
           M=g(0,0,.33,.48),O=rim,U=g(-.28,0,.25,.54),L=g(.37,0,.24,.56))
    valid=(abs(x)<.9)&(abs(y)<.9)
    ks=[]; ws=[]
    for i,k in enumerate(names):
        for r in regions[k].split():
            a=np.zeros((len(names),h,w),dtype=np.float32)
            a[i]=unit(m[r]*valid)
            ws.append(a.ravel()); ks.append(k+r)
    return net.dense(name,src,np.array(ws),np.zeros(len(ks))),ks


def stroke_context(net, rgb):
    # Long *coherent* steps, rather than arbitrary large horizontal/vertical gradients.
    gray = net.conv('stroke_luminance',rgb,LUM[None,:,None,None],[0.])
    k=7; y,x=yyxx(k)
    ws=[]; ns=[]; b=[]; limbs={}
    for t,theta in enumerate((0.,np.pi/2,np.pi/4,-np.pi/4)):
        ca,sa=np.cos(theta),np.sin(theta)
        for sign in (1,-1):
            c=str(t)+('p' if sign==1 else 'n')
            limbs[c]=[]
            for j,offset in enumerate((-1.5,0.,1.5)):
                def G(side):
                    return gauss(y,x,offset*sa+side*.81*ca,offset*ca-side*.81*sa,.49,.78,theta)
                ws.append(sign*diff(G(1),G(-1))); b.append(-.057)
                n='stroke_'+c+str(j); ns.append(n); limbs[c].append(n)
    raw=net.relu('positive_step_parts',net.conv('coherent_step_parts',gray,np.array(ws)[:,None],b,pad=3))
    a=channels(ns)
    u,un=transition(net,'step_agreement_pairs',raw,ns,ns,{k:a[x]-a[m] for k,(x,m,z) in limbs.items()})
    a=channels(un)
    v,vn=lin(net,'step_agreement_third_position',u,un,{**{k:a[x]-a[k] for k,(x,m,z) in limbs.items()},
                                       **{k+'agt':a[x]-a[k]-a[z] for k,(x,m,z) in limbs.items()}})
    a=channels(vn)
    val=lambda k: a[k]-a[k+'agt']
    h=val('0p')+val('0n'); vert=val('1p')+val('1n'); dia=(val('2p')+val('2n')+val('3p')+val('3n'))/2
    s,sn=bounded_maps(net,v,vn,'coherent_strokes',dict(
        long_h=9*h-.24,long_v=9*vert-.24,long_d=9*dia-.20,
        long_axis=7*(h+vert)-6*dia-.23,
        slope=8*(dia-.53*(h+vert))-.16,
        car_line=12*(h-.55*vert)-.27,
        box_line=12*(vert-.37*h)-.27,
    ))
    return context_windows(net,'spatial_stroke_density',s,sn,{k:'A T B M O U L' for k in sn})


def contrast_layout(net, rgb):
    """Compare each coarse cell to four broad corner regions in this image."""
    rgb=net.pool('layout_rgb','image',(4,4),(4,4))
    h=8
    y,x=np.mgrid[:h,:h].astype(float)
    y=(y+.5)*2/h-1; x=(x+.5)*2/h-1
    def anchor(s,t):
        return unit(np.exp(-.5*((y-.79*s)/.34)**2-.5*((x-.76*t)/.39)**2))
    anchors=[anchor(s,t) for s in (-1,1) for t in (-1,1)]
    color=[LUM,np.array([.57,-.57,0.]),np.array([-.28,-.28,.56])]
    # Store a per-site distance to each corner. Four distances, three opponent
    # coordinates per distance, and two signs implement an L1 metric.
    w=[]
    for m in anchors:
        for v in color:
            for direction in (1,-1):
                for i in range(h*h):
                    p=-m.copy(); p.ravel()[i]+=1.
                    w.append((direction*v[:,None,None]*p).ravel())
    ds=net.dense('corner_color_distances',rgb,np.array(w),np.zeros(len(w)))
    ds=net.relu('abs_corner_delta',ds)
    d=net.reshape('site_color_distances',ds,(24,8,8))
    dn=[f'd{k}{c}{s}' for k in range(4) for c in range(3) for s in range(2)]
    a=channels(dn)
    diffs=[sum(a[f'd{k}{c}{s}'] for c in range(3) for s in range(2)) for k in range(4)]
    expr={f'c{i}':v for i,v in enumerate(diffs)}
    expr.update(top_delta=diffs[0]-diffs[1],bot_delta=diffs[2]-diffs[3],
                top_dark=a['d001']+a['d101'],bot_dark=a['d201']+a['d301'],
                top_light=a['d000']+a['d100'],bot_light=a['d200']+a['d300'])
    a1,n1=lin(net,'corner_distance_metrics',d,dn,expr)
    q=channels(n1)
    top=q['c0']-q['top_delta']; bot=q['c2']-q['bot_delta']
    a2,n2=lin(net,'corner_match_intersection',a1,n1,
              dict(top=top, bot=bot, tb=top-bot,
                   dark=q['top_dark']+q['bot_dark'],light=q['top_light']+q['bot_light']))
    z=channels(n2)
    novel=z['top']-z['tb']
    maps,mn=bounded_maps(net,a2,n2,'layout_support_maps',dict(
        novelty=4.5*novel-.19,
        non_sky=3.7*z['top']-.17,
        non_ground=3.7*z['bot']-.17,
        darker_center=1.9*z['dark']-.20,
        lighter_center=1.9*z['light']-.22,
    ))
    return context_windows(net,'coarse_foreground_layout',maps,mn,{k:'A T B M O U L' for k in mn})


def physical_part_bank(net, rgb):
    """Matched filters for specific parts. Every lobe is an analytic Gaussian."""
    k=11; y,x=yyxx(k)
    W=[]; B=[]; labels=[]; families={}
    Y=LUM
    Warm=np.array([.50,.17,-.67]); Red=np.array([.64,-.57,-.07])
    def add(f,parts,scale=1.,mirror=1.,angle=0.,th=.03):
        w=np.zeros((3,k,k),dtype=float)
        ca,sa=np.cos(angle),np.sin(angle)
        for color,amt,yy,xx,sy,sx,rot in parts:
            xx*=mirror
            xo=scale*(ca*xx-sa*yy); yo=scale*(sa*xx+ca*yy)
            p=gauss(y,x,yo,xo,max(.32,sy*scale),max(.32,sx*scale),angle+mirror*rot)
            w+=amt*color[:,None,None]*unit(p)
        families.setdefault(f,[]).append(len(labels))
        labels.append(f+'_p'+str(len(labels))); W.append(w); B.append(-th)
    eye=[(Y,-.66,-.58,-1.55,.47,.48,0),(Y,-.66,-.58,1.55,.47,.48,0),
         (Y,.55,-.60,0,.56,.55,0),(Y,.38,-.30,-1.80,.89,1.06,0),(Y,.39,-.30,1.80,.89,1.06,0)]
    triangle=[(Y,-.22,-2.29,-1.86,.46,.41,-.37),(Y,-.22,-2.29,1.86,.46,.41,.37),
              (Y,.21,-2.22,0,.52,.72,0),(Y,.115,-2.35,-2.78,.58,.51,0),(Y,.115,-2.35,2.78,.58,.51,0)]
    muzzle=[(Y,-.82,.82,0,.58,.72,0),(Y,.41,.54,-1.1,.62,.67,0),(Y,.41,.54,1.1,.62,.67,0),
            (Y,-.20,1.63,0,.42,.69,0),(Y,.2,2.34,0,.70,1.06,0)]
    catnose=[(Y,-.51,.50,0,.40,.51,0),(Y,.29,.69,-.84,.54,.62,0),(Y,.29,.69,.84,.54,.62,0),
             (Y,-.17,1.35,0,.38,.54,0),(Y,.10,1.88,0,.54,.85,0)]
    paw=[(Y,-.66,1.42,-1.93,.68,.58,0),(Y,-.66,1.42,1.98,.68,.58,0),
         (Y,.72,1.46,.02,.64,.88,0),(Y,.30,1.42,-2.92,.67,.58,0),(Y,.30,1.42,2.95,.67,.58,0)]
    def wing(front,side):
        if front:
            return [(Y,-.55,0,0,.55,2.7,0),(Y,-.26,-.63,0,1.07,.57,0),
                    (Y,.26,-1.23,-2.01,.65,1.16,0),(Y,.27,-1.24,2.02,.67,1.14,0),
                    (Y,.28,1.6,-1.78,.61,1.38,0),(Y,.27,1.55,1.81,.64,1.42,0)]
        return [(Y,-.49,0,-.2,.49,2.42,0),(Y,-.29,-.53,-2.36,.96,.45,-.50),
                (Y,-.22,.90,.08,.54,.96,-.70*side),(Y,.51,-1.7,.36,.65,2.75,0),
                (Y,.49,2.05,0,.71,2.76,0)]
    for scale in (.68,1.,1.30):
        for yaw in (-.22,0.,.22):
            add('catFace',eye+triangle+catnose,scale,1.,yaw,.080)
            add('dogFace',[(c,.76*v,yy,xx,sy,sx,t) for c,v,yy,xx,sy,sx,t in eye]+muzzle,scale,1.,yaw,.086)
        for ang in (-.39,0.,.39):
            for front in (True,False):
                p=wing(front,1)
                add('airWing',p,scale,1.,ang,.069)
                add('airWing',[(c,-v,yy,xx,sy,sx,t) for c,v,yy,xx,sy,sx,t in p],scale,1.,ang,.065)
        for mirror in (1.,-1.):
            add('birdEye',[(Y,-.51,-.89,.90,.41,.45,0),(Y,.39,-1.01,.40,.66,.67,0),
                           (Y,-.22,-.70,2.10,.43,.77,.1),(Y,.22,-1.64,1.92,.65,.85,0),
                           (Y,.17,.38,.93,.75,.66,0),(Y,-.33,.57,-.49,.68,1.38,-.54),
                           (Y,.28,1.77,-.65,.68,1.08,0)],scale,mirror,0.,.071)
            add('birdEye',[(Y,-.59,-.82,.90,.48,.49,0),(Y,.56,-.95,.61,.69,.68,0),
                           (Y,.29,.43,.30,.78,1.04,.48),(Y,-.20,.72,1.91,.85,.78,0),
                           (Warm,.37,.02,.58,.81,1.01,0),(Warm,-.43,-1.91,-.72,.77,1.12,0)],scale,mirror,0.,.065)
            add('frogForm',[(Y,-.39,-.89,-1.83,.44,.53,0),(Y,-.39,-.89,1.83,.44,.53,0),
                           (Y,.29,-.55,0,.55,.99,0),(Y,-.53,.74,0,.43,1.99,0),
                           (Y,.35,.23,.05,.45,1.70,0),(Y,.34,1.31,.05,.63,1.98,0),
                           (Y,.17,-1.47,1.93,.63,.87,0),(Y,.16,-1.47,-1.93,.63,.87,0)],scale,mirror,0.,.095)
            for pol in (1,-1):
                p=[(Y,-.54,.85,0,.54,2.77,0),(Y,.51,-.02,0,.66,2.70,0),
                   (Y,.32,1.86,0,.62,2.98,0),(Y,-.29,-.81,-.46,1.52,.41,0),
                   (Y,.24,-1.54,1.46,.68,1.01,0),(Y,-.24,1.47,2.25,.74,.64,-.52)]
                add('shipHull',[(c,pol*v,yy,xx,sy,sx,t) for c,v,yy,xx,sy,sx,t in p],scale,mirror,0.,.120)
            for leg in (0,1):
                p=[(Y,-.72,-.37,-.43,.89,1.91,0),(Y,.39,-1.86,-.70,.67,1.68,0),
                   (Y,-.38,-1.02,1.65,.99,.71,-.90),(Y,.34,1.17,-.69,.47,1.63,0),
                   (Y,.32,-.74,2.90,.68,.66,0),(Y,.05,2.95,0,.53,1.44,0)]
                p+=paw if leg else [(Y,-.39,1.68,-1.64,.96,.40,-.10),(Y,-.39,1.71,1.53,.95,.42,.10),
                                    (Y,.41,1.77,-.14,.81,.78,0),(Y,.22,1.88,-2.71,.83,.51,0),
                                    (Y,.15,1.88,2.59,.82,.52,0)]
                add('ungulate',p,scale,mirror,0.,.15)
            roof=[(Y,-.53,-1.52,0,.59,1.87,0),(Y,.45,-2.62,0,.65,2.03,0),
                  (Y,.55,-.32,.18,.63,2.17,0),(Y,-.37,1.10,0,.53,2.46,0),
                  (Y,.22,2.24,0,.58,2.21,0),(Y,-.21,.95,-2.75,.76,.63,0),
                  (Y,-.11,.88,2.75,.73,.59,0)]
            profile=[(Y,-.43,-1.16,.37,.61,1.79,0),(Y,.47,-2.25,0,.65,2.01,0),
                     (Y,.39,.06,-.36,.59,2.64,0),(Y,-.40,1.49,-2.32,.56,.63,0),
                     (Y,-.40,1.49,2.33,.56,.63,0),(Y,.37,1.47,0,.53,1.06,0)]
            add('carRoof',roof,scale,mirror,0.,.124)
            add('carRoof',profile,scale,mirror,0.,.123)
            add('carRoof',[(c,-v,yy,xx,sy,sx,t) for c,v,yy,xx,sy,sx,t in profile],scale,mirror,0.,.112)
            cab=[(Y,-.55,-1.58,0,.77,1.64,0),(Y,.35,-2.79,0,.52,1.82,0),
                 (Y,.57,-.11,0,.66,1.84,0),(Y,-.43,1.00,0,.62,1.80,0),
                 (Y,.32,1.93,0,.58,1.61,0),(Y,-.17,2.69,-1.8,.54,.58,0),
                 (Y,-.17,2.69,1.8,.54,.58,0),(Y,.08,2.69,0,.56,.90,0)]
            add('truckCab',cab,scale,mirror,0.,.135)
            add('truckCab',[(Y,.30,-1.28,-1.29,1.39,1.34,0),(Y,-.32,-2.76,-1.10,.55,1.68,0),
                            (Y,-.29,-.58,-2.93,1.43,.60,0),(Y,-.27,-.99,.44,1.42,.59,0),
                            (Y,.38,-.85,1.48,1.13,.99,0),(Y,-.37,.86,-.78,.58,1.81,0),
                            (Y,.29,1.92,-.94,.52,1.65,0),(Y,-.20,1.45,2.37,.69,.74,0),
                            (Red,.20,-.98,-1.40,1.29,1.40,0),(Red,-.09,-2.52,1.21,.71,1.05,0)],scale,mirror,0.,.128)
    raw=net.relu('raw_physical_matches',net.conv('physical_part_bank',rgb,np.array(W),B,pad=0))
    _,h,w=net.shapes[raw]
    peak=net.pool('physical_part_peaks',raw,(h,w),(h,w),'maxpool')
    mean=net.pool('physical_part_mean',raw,(h,w),(h,w))
    res=[]
    for src,suf in ((peak,'peak'),(mean,'mean')):
        ww=np.zeros((len(labels),len(labels)),dtype=np.float32)
        np.fill_diagonal(ww, 2.8 if suf=='peak' else 6.0)
        z=net.dense('physical_'+suf,src,ww,np.zeros(len(labels)))
        res.append((z,[k+'_'+suf for k in labels]))
    mixed,ks=append_small(net,'physical_summaries',res)
    a=channels(ks)
    base={n:a[n+'_peak']-.32*a[n+'_mean'] for n in labels}
    q,kn=lin(net,'physical_localized_evidence',mixed,ks,base)
    # A pool over variants of each family, then retain several families independently.
    a=channels(kn)
    groups={k:[labels[i] for i in ii] for k,ii in families.items()}
    j=0
    while any(len(v)>1 for v in groups.values()):
        a=channels(kn); raw={k:a[k] for k in kn}; pending={}
        for k,items in groups.items():
            ps=[]
            for i in range(0,len(items)-1,2):
                v=k+str(j)+'r'+str(i)
                raw[v+'_diff']=a[items[i]]-a[items[i+1]]
                ps.append((v,items[i+1]))
            pending[k]=(ps,items[-1:] if len(items)%2 else [])
        q,kn=lin(net,'physical_max_hinge_'+str(j),q,kn,raw); a=channels(kn)
        new={}; groups={}
        for k,(items,odd) in pending.items():
            groups[k]=[v for v,_ in items]+odd
            for v,b in items:
                new[v]=a[b]+a[v+'_diff']
            for v in odd: new[v]=a[v]
        q,kn=lin(net,'physical_max_merge_'+str(j),q,kn,new); j+=1
    a=channels(kn)
    return lin(net,'physical_match_evidence',q,kn,{k:a[v[0]] for k,v in groups.items()})


def append_small(net, label, sources):
    n=sum(len(a) for _,a in sources)
    u=[]; ks=[]
    for i,(src,keys) in enumerate(sources):
        w=np.zeros((n,len(keys))); p=len(ks)
        w[p:p+len(keys)]=np.eye(len(keys)); ks+=keys
        u.append(net.dense(label+'_embed'+str(i),src,w,np.zeros(n)))
    return net.put(label,'add',u,(n,)),ks


def final_readout(net, src, names):
    # Competing hypotheses use capped contextual evidence and a smaller
    # contribution from individual templates. Separate class cue groups use a
    # soft AND (an affine sum followed by a hinge) rather than a bare max.
    a=channels(names)
    f=lambda k,r='A': a[k+r]
    top=f('blue','T'); bot=f('blue_water','B'); blue=f('blue','B')
    sky=f('open_light','T'); bottom_open=f('open_light','B')
    rim_flat=f('open_flat','O'); top_flat=f('open_flat','T'); bot_flat=f('open_flat','B')
    waves=f('water_texture','B'); water_middle=f('water_texture','L')
    grass=.68*f('green','B')+.32*f('green','L')
    canopy=.64*f('green','T')+.36*f('green','O')
    leaves=.58*f('leaves','O')+.42*f('leaves','T')
    red=.74*f('red','M')+.26*f('red','U')
    rimsoil=.65*f('earth','O')+.35*f('olive','O')
    corewarm=.66*f('bronze','M')+.34*f('chestnut','M')
    coreolive=.66*f('olive','M')+.34*f('mottled','M')
    browns=f('earth','M')
    roads=.73*f('road','B')+.27*f('road','L')
    fine=f('fine','M'); outside_fine=f('fine','O'); busy=f('busy','M')
    top_edge=.63*f('long_axis','U')+.37*f('long_v','U')
    bottom_edge=.64*f('long_h','L')+.36*f('car_line','L')
    outline=.69*f('long_axis','M')+.31*f('long_axis','U')
    plane_texture=.65*f('horizontal','M')+.35*f('vertical','M')
    horizon=f('long_h','B')-.63*f('long_v','B')
    chrome=f('white','M'); jetblack=f('black','M')
    marked=.55*chrome+.45*jetblack-.25*(f('white','O')+f('black','O'))
    subject=f('novelty','M')-.56*f('novelty','O')
    portrait=.48*(fine-outside_fine)+.49*subject+.32*(busy-f('busy','O'))
    softface=.51*f('round_dark','M')+.49*f('bright_dot','M')
    blackpaw=.57*f('black','L')+.43*jetblack
    chest=f('chestnut','M')-.56*f('chestnut','O')
    curving=.55*f('long_d','M')+.45*f('diagonal','M')
    true_black=jetblack-.50*f('black','O')
    wet=.61*f('bright_dot','M')+.39*f('round_dark','U')
    gd=lambda k: a[k+'_center']
    part=lambda k, rival: a[k]-.62*a[rival]
    low_wheel=a['wheelpair_low']-.77*gd('wheelpair')
    expr=dict(
        upper_sky=1.75*top-1.10*blue+.46*sky+.45*(top_flat-bot_flat)-.73*waves-.52*canopy-.12,
        still_air=1.15*top_flat+.78*sky+.44*rim_flat+.27*top-.88*waves-.55*leaves-.39,
        airplane_shape=.79*part('airWing','shipHull')+.46*rim_flat+.33*curving+.27*subject-.39*leaves-.23,
        blue_below=1.47*bot+.45*water_middle-.45*top-.29*bot_flat-.18,
        rippling=1.69*waves+.64*water_middle+.36*horizon+.40*bot-.27*top-.13,
        over_water=.77*bot+.71*waves+.58*horizon+.44*chrome+.60*part('shipHull','airWing')-.38*canopy-.33,
        pale_object=.62*chrome+.54*busy+.46*sky+.44*bottom_open-.92*canopy-.51*rimsoil-.35,
        road_context=.90*roads+.53*bot_flat+.52*(chrome+true_black)+.63*red-1.05*canopy-.83*rimsoil-.25,
        smooth_paint=1.05*red+.68*(chrome+true_black)+.65*outline+.19*plane_texture-.52*corewarm-.12,
        machine_outline=.86*outline+.57*bottom_edge+.41*top_edge+.53*busy-.45*outside_fine-.18,
        car_window=.79*part('carRoof','truckCab')+.65*bottom_edge+.43*true_black+.33*low_wheel-.22,
        tall_cab=.85*part('truckCab','carRoof')+.91*top_edge+.41*chrome+.32*red-.24,
        low_profile=.63*bottom_edge+.57*blackpaw+.39*curving-.44*top_edge-.20,
        tall_profile=.67*top_edge+.64*chrome+.42*plane_texture-.37*curving-.23,
        woodland=1.03*canopy+1.50*leaves+.56*rimsoil-.42*sky-.28,
        open_field=1.44*grass+.71*sky+.49*f('earth','B')-.84*canopy-.55*leaves-.26,
        dry_ground=.83*f('earth','B')+.54*bot_flat+.35*grass-.57*waves-.24,
        brown_ground=1.11*rimsoil+.56*f('mottled','O')+.33*leaves-.46*grass-.31,
        deer_body=.63*(gd('deer')-.67*gd('horse'))+.73*canopy+.50*subject+.36*rimsoil-.28,
        horse_body=.57*(gd('horse')-.59*gd('deer'))+.93*chest+.57*grass+.32*subject-.23,
        ungulate_part=.76*part('ungulate','carRoof')+.58*grass-.40*f('fine','L')-.20,
        wood_frog=1.20*coreolive+.72*f('mottled','M')+.43*browns+.43*wet-.53*grass-.36,
        frog_head=.69*part('frogForm','dogFace')+.51*coreolive+.53*wet+.36*f('horizontal','M')-.31,
        skin_brown=.70*browns+.50*coreolive+.40*rimsoil+.37*f('round_dark','M')-.25,
        bird_small=.86*portrait+.69*rim_flat+.40*top+.43*leaves+.22*subject-.35*outline-.13,
        bird_feather=.70*fine+.65*portrait+.51*red+.34*corewarm+.38*curving-.30*outline-.20,
        bird_head=.75*part('birdEye','airWing')+.55*subject+.37*softface+.38*canopy-.20,
        no_water=.85*rim_flat+.52*portrait+.38*softface-.79*bot-.53*waves-.32,
        indoor=.91*corewarm+.48*rim_flat+.40*portrait-.87*canopy-.75*leaves-.42*top-.21,
        soft_fur=.96*corewarm+.70*fine+.55*portrait+.30*softface-.46*outline-.27,
        feline_mask=.72*part('catFace','dogFace')+.60*softface+.44*corewarm+.30*portrait-.27,
        canine_mask=.90*part('dogFace','catFace')+.67*marked+.61*softface+.40*blackpaw-.22,
        dog_coat=.85*marked+.59*portrait+.52*true_black+.32*chrome+.27*grass-.21,
        close_pet=.86*portrait+.67*softface+.41*subject+.35*marked-.40*coreolive-.21,
        upper_green=canopy, green_field=grass, rich_red=red, bottom_water=bot, above_blue=top,
    )
    s, sn=bounded_maps(net,src,names,'visual_hypotheses',expr)
    u=channels(sn)
    # Co-occurrence gates only produce strong evidence when two independent
    # cues agree, e.g. bright metal plus a window-like shape, or a dark muzzle
    # plus a high-contrast coat. Limit each semantic group's influence.
    groups={
        'sky_plane':.78*u['upper_sky']+.68*u['still_air']+.36*u['airplane_shape']-.12,
        'water_ship':.83*u['blue_below']+.75*u['rippling']+.39*u['over_water']-.13,
        'vehicle':.74*u['smooth_paint']+.54*u['road_context']+.45*u['machine_outline']-.16,
        'car':.75*u['car_window']+.64*u['low_profile']+.31*u['road_context']-.09,
        'truck':.77*u['tall_cab']+.65*u['tall_profile']+.43*u['machine_outline']-.17,
        'bird':.75*u['bird_head']+.72*u['bird_small']+.54*u['bird_feather']-.13,
        'cat':.77*u['feline_mask']+.77*u['soft_fur']+.43*u['indoor']-.22,
        'dog':.81*u['canine_mask']+.67*u['dog_coat']+.40*u['close_pet']-.14,
        'frog':.76*u['wood_frog']+.64*u['frog_head']+.47*u['skin_brown']-.21,
        'deer':.73*u['woodland']+.65*u['deer_body']+.35*u['brown_ground']+.29*u['ungulate_part']-.13,
        'horse':.80*u['open_field']+.66*u['horse_body']+.38*u['dry_ground']+.31*u['ungulate_part']-.14,
    }
    before,bn=lin(net,'supporting_evidence',s,sn,{**u,**groups,**{k+'_excess':v-1. for k,v in groups.items()}})
    a=channels(bn)
    bounded,bk=lin(net,'bounded_support',before,bn,{**{k:a[k] for k in sn},**{k:a[k]-a[k+'_excess'] for k in groups}})
    q=channels(bk)
    def score(b,**w): return E(b=b)+sum(q[k]*v for k,v in w.items())
    out=[
        score(.04,sky_plane=1.24,pale_object=.28,airplane_shape=.29,upper_sky=.25,
              blue_below=-.30,rippling=-.96,woodland=-.39,vehicle=-.27,cat=-.22),
        score(.04,vehicle=.99,car=.90,smooth_paint=.26,rich_red=.35,pale_object=.19,
              truck=-.28,rippling=-.38,brown_ground=-.47,upper_sky=-.15,woodland=-.26),
        score(.26,bird=1.13,bird_small=.27,bird_feather=.25,no_water=.16,above_blue=.15,
              rippling=-.34,vehicle=-.42,horse=-.29,upper_sky=-.18,dog=-.19),
        score(.00,cat=1.01,indoor=.38,close_pet=.40,soft_fur=.22,no_water=.15,
              dog=-.26,vehicle=-.32,woodland=-.32,rippling=-.33,upper_sky=-.29,frog=-.21),
        score(.05,deer=1.12,woodland=.33,brown_ground=.37,ungulate_part=.25,upper_green=.22,
              horse=-.18,close_pet=-.28,rippling=-.39,vehicle=-.37,upper_sky=-.26),
        score(.22,dog=1.14,close_pet=.36,indoor=.38,canine_mask=.21,dog_coat=.22,
              cat=-.15,vehicle=-.33,woodland=-.27,rippling=-.36,upper_sky=-.26,frog=-.18),
        score(.04,frog=1.10,wood_frog=.28,skin_brown=.21,brown_ground=.30,
              open_field=-.23,vehicle=-.38,upper_sky=-.30,rippling=-.37,dog=-.15,horse=-.24),
        score(.00,horse=1.19,horse_body=.26,open_field=.25,green_field=.23,ungulate_part=.18,
              woodland=-.24,frog=-.13,vehicle=-.25,rippling=-.38,close_pet=-.26,deer=-.12),
        score(.06,water_ship=1.25,over_water=.40,rippling=.31,bottom_water=.18,pale_object=.23,
              upper_sky=-.41,woodland=-.42,close_pet=-.24,vehicle=-.28,brown_ground=-.29),
        score(.03,vehicle=.94,truck=.95,tall_profile=.25,rich_red=.30,pale_object=.17,
              car=-.13,rippling=-.35,brown_ground=-.42,upper_sky=-.18,woodland=-.25),
    ]
    lin(net,'class_scores',bounded,bk,{str(i):e for i,e in enumerate(out)},relu=False)


def main(folder):
    net=Net()
    rgb=net.pool('antialiased_colour','image',(2,2),(2,2))
    app,ak=local_appearance(net,rgb)
    context,ck=pixel_context(net,app,ak)
    strokes,sk=stroke_context(net,rgb)
    geom,gk=gray_geometry(net,rgb)
    regions,rk,support,pk,families=geometry_templates(net,geom,gk)
    contrast,lk=contrast_layout(net,rgb)
    mixed,mk=append_vectors(net,[(context,ck),(strokes,sk),(regions,rk),(support,pk),(contrast,lk)])
    cues,ck=family_max(net,mixed,mk,families)
    phys,ph=physical_part_bank(net,rgb)
    cues,ck=append_small(net,'fused_evidence',[(cues,ck),(phys,ph)])
    final_readout(net,cues,ck)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'network.json').write_text(json.dumps(dict(layers=net.layers),indent=2)+'\n')
    np.savez(folder/'weights.npz',**net.weights)


if __name__=='__main__':
    main(Path(sys.argv[1]))
