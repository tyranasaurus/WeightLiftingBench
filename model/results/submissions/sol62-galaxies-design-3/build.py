"""Generate a fixed galaxy morphology network from analytic masks and rules.

The generator needs only NumPy. It does not open any inputs. Numerical constants
are optical/geometric design choices, not estimates from the survey sample.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}
        self.serial = 0

    def layer(self, op, *inputs, name=None, **settings):
        if name is None:
            self.serial += 1
            name = f'{op}_{self.serial:03}'
        self.layers.append(dict(name=name, op=op, inputs=list(inputs), **settings))
        return name

    def affine(self, x, w, b=0, name=None, padding=0):
        w = np.asarray(w, dtype=np.float32)
        if w.ndim == 2:
            x = self.layer('dense', x, name=name, out=w.shape[0])
        else:
            x = self.layer('conv', x, name=name, out=w.shape[0],
                           kernel=list(w.shape[-2:]), stride=1, padding=padding)
        self.weights[x + '.W'] = w
        self.weights[x + '.b'] = np.broadcast_to(b, (w.shape[0],)).astype(np.float32)
        return x

    def pool(self, kind, x, n, stride, name=None):
        return self.layer(kind, x, name=name, size=[n,n], stride=[stride,stride])

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'network.json').write_text(json.dumps({'layers': self.layers}, indent=2) + '\n')
        np.savez(folder / 'weights.npz', **self.weights)


class Affine:
    """An affine expression of circuit nodes; ReLUs mark nonlinear boundaries."""
    def __init__(self, terms=None, b=0.):
        self.terms = {} if terms is None else terms
        self.b = float(b)

    @staticmethod
    def lift(x):
        return x if isinstance(x, Affine) else Affine(b=x)

    def __add__(self, other):
        other = Affine.lift(other)
        pairs = self.terms.copy()
        for k, v in other.terms.items():
            t = pairs.get(k, 0.) + v
            if t:
                pairs[k] = t
            else:
                pairs.pop(k, None)
        return Affine(pairs, self.b+other.b)

    __radd__ = __add__

    def __mul__(self, c):
        c = float(c)
        return Affine({k: c*v for k,v in self.terms.items()} if c else {}, self.b*c)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Affine.lift(other))

    def __rsub__(self, other):
        return Affine.lift(other) + (-self)

    def __truediv__(self, q):
        return self * (1/q)


class Circuit:
    """Lower the photometry/decision graph to packed dense, relu and norm layers."""
    def __init__(self):
        self.ops = []
        self.depth = []
        self.rows = []
        self.input_count = 0
        self.cache = {}

    def node(self, op, ins):
        k = len(self.ops)
        self.ops.append((op, ins))
        self.depth.append(0 if op == 'input' else
                          1 + max((self.depth[i] for x in ins for i in x.terms), default=0))
        return Affine({k: 1.})

    def read(self, row):
        assert len(self.ops) == self.input_count
        self.input_count += 1
        self.rows.append(row)
        return self.node('input', [])

    def relu(self, x):
        x = Affine.lift(x)
        if not x.terms:
            return Affine(b=max(x.b, 0))
        key = (tuple(sorted(x.terms.items())), x.b)
        if key not in self.cache:
            self.cache[key] = self.node('relu', [x])
        return self.cache[key]

    def clip(self, x, lo, hi):
        return lo + self.relu(x-lo) - self.relu(x-hi)

    def abs(self, x):
        return self.relu(x) + self.relu(-x)

    def mx(self, a, b):
        return b + self.relu(a-b)

    def mn(self, a, b):
        return a - self.relu(a-b)

    def reduce(self, op, xs):
        xs = list(xs)
        while len(xs) > 1:
            xs = [op(xs[i], xs[i+1]) if i+1 < len(xs) else xs[i]
                  for i in range(0, len(xs), 2)]
        return xs[0]

    def ratio(self, num, den):
        # A norm channel with four synthetic positions. A large denominator
        # coefficient makes this a bounded, signed approximation to num/den.
        return self.node('divide', [Affine.lift(num), Affine.lift(den)])

    def magnitude(self, a, b):
        # Average absolute projections onto a uniform set of axes, with
        # the constant that makes the continuum average equal the length.
        return (math.pi/12)*sum(self.abs(a*math.cos(t) + b*math.sin(t))
                                for t in (np.arange(6)+0.5)*math.pi/6)

    def product(self, a, b):
        # Linear interpolant of x^2; symmetric polarization gives a*b.
        # Intended for bounded features and routing weights in [-1,1].
        def sq(x):
            x = self.abs(x)
            step = .25
            return step*x + 2*step*sum(self.relu(x-j*step) for j in range(1,8))
        return (sq(a+b)-sq(a-b)) * .25

    def emit(self, net, image_features, scores):
        final_depth = 1 + max(self.depth[i] for x in scores for i in x.terms)
        last = {}
        stack = [(i, final_depth) for x in scores for i in x.terms]
        while stack:
            j, user = stack.pop()
            if j not in last:
                for x in self.ops[j][1]:
                    stack.extend((i, self.depth[j]) for i in x.terms)
            last[j] = max(last.get(j, 0), user)
        groups = defaultdict(list)
        for j in sorted(last):
            groups[self.depth[j]].append(j)
        prev = groups[0]
        # Only photometry that reaches the classifier is materialized.
        x = net.affine(image_features, np.stack([self.rows[j] for j in prev]),
                       name='analytic_photometry')

        def matrix(expressions, source):
            lookup = {j:i for i,j in enumerate(source)}
            w = np.zeros((len(expressions),len(source)), np.float32)
            b = np.zeros(len(expressions), np.float32)
            for t, e in enumerate(expressions):
                b[t] = e.b
                for j, val in e.terms.items():
                    w[t,lookup[j]] = val
            return w,b

        for d in range(1, final_depth):
            carry = [j for j in prev if last[j] > d]
            rs = [j for j in groups[d] if self.ops[j][0] == 'relu']
            ns = [j for j in groups[d] if self.ops[j][0] == 'divide']
            nxt = carry + rs + ns
            parts = []
            if carry:
                w = np.zeros((len(nxt),len(prev)), np.float32)
                lookup = {j:i for i,j in enumerate(prev)}
                for i,j in enumerate(carry):
                    w[i,lookup[j]] = 1
                parts.append(net.affine(x,w))
            if rs:
                w,b = matrix([self.ops[j][1][0] for j in rs], prev)
                y = net.layer('relu', net.affine(x,w,b))
                if rs != nxt:
                    v = np.zeros((len(nxt),len(rs)), np.float32)
                    v[len(carry)+np.arange(len(rs)),np.arange(len(rs))] = 1
                    y = net.affine(y,v)
                parts.append(y)
            if ns:
                es = []
                for j in ns:
                    num,den = self.ops[j][1]
                    es.extend([num*10,-num*10,den*120,-den*120])
                w,b = matrix(es,prev)
                y = net.affine(x,w,b)
                y = net.layer('reshape',y,shape=[len(ns),1,4])
                y = net.layer('norm',y)
                v = np.zeros((len(nxt),4*len(ns)), np.float32)
                v[len(carry)+len(rs)+np.arange(len(ns)),4*np.arange(len(ns))] = 12/math.sqrt(2)
                parts.append(net.affine(y,v))
            x = net.layer('add', *parts) if len(parts)>1 else parts[0]
            prev = nxt
        w,b = matrix(scores,prev)
        net.affine(x,w,b,name='galaxy_class_scores')


def image_planes(net):
    """Two-pixel bins, robust sky subtraction, and bounded foreground maps."""
    rgb = net.pool('avgpool', 'image', 2, 2, name='two_pixel_bins')
    cap = .235
    dark = net.affine(rgb, -np.eye(3)[:,:,None,None], cap, name='sky_winsorize')
    dark = net.layer('relu', dark)
    for j in range(6):
        dark = net.pool('avgpool', dark, 2, 2, name=f'sky_reduce_{j}')
    # Local point response, extended response, and a colour-opponent channel.
    luminance = np.array([.30,.40,.30])
    opponent = np.array([1.,0.,-1.])
    dy, dx = np.mgrid[-2:3,-2:3]
    d2 = dx*dx + dy*dy
    def gauss(s):
        g = np.exp(-.5*d2/s**2)
        return g/g.sum()
    g0, g1 = gauss(.63), gauss(1.25)
    w = np.stack([luminance[:,None,None]*g0,
                  luminance[:,None,None]*g1,
                  opponent[:,None,None]*g0])
    maps = net.affine(rgb, w, name='local_intensity_and_colour', padding=2)
    transform = np.stack([luminance,luminance,opponent])
    wsky = np.repeat(transform, 64*64, axis=0)
    bsky = np.repeat(-cap*transform.sum(axis=1), 64*64)
    sky = net.affine(dark,wsky,bsky,name='broadcast_sky_offset')
    sky = net.layer('reshape',sky,shape=[3,64,64])
    maps = net.layer('add',maps,sky,name='sky_subtracted_planes')

    # F is low-threshold light with a soft knee, H is higher isophotal
    # light. Clip colour by the local foreground envelope to suppress sky.
    w = np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[0,0,-1],
                  [0,1,0],[0,1,0],[1,0,0]])
    t = net.affine(maps,w[:,:,None,None],[-.028,-.18,-.105,-.38,0,0,-.037,-.21,-.057])
    t = net.layer('relu',t)
    f = np.array([1,-.76,0,0,0,0,0,0,0])
    h = np.array([0,0,1,-.55,0,0,0,0,0])
    col = np.array([0,0,0,0,1,-1,0,0,0])
    w = np.stack([f,h,f+col,f-col])
    pos = net.layer('relu',net.affine(t,w[:,:,None,None]))
    # F+colour and F-colour are also clipped above 2F.
    w = np.stack([np.array([0,0,1,0])-2*np.eye(4)[0],
                  np.array([0,0,0,1])-2*np.eye(4)[0]])
    overflow = net.layer('relu',net.affine(pos,w[:,:,None,None]))
    w = np.array([[1,0,0,0],[0,1,0,0],[0,0,.5,-.5],[0,0,0,0],[0,0,0,0]])
    masked = net.affine(pos,w[:,:,None,None])
    w = np.array([[0,0],[0,0],[-.5,.5],[0,0],[0,0]])
    correction = net.affine(overflow,w[:,:,None,None])
    w = np.array([[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0]])
    raw = net.affine(maps,w[:,:,None,None])
    main = net.layer('add',masked,correction,raw,name='foreground_and_core_maps')
    es = np.zeros((2,9,1,1))
    es[0,6,0,0], es[0,7,0,0], es[1,8,0,0] = 1,-.35,1
    ext = net.affine(t,es,name='extended_and_shape_maps')
    # Preserve plane order when concatenating via equal-shaped sums.
    left = np.vstack([np.eye(5),np.zeros((2,5))])
    right = np.vstack([np.zeros((5,2)),np.eye(2)])
    return net.layer('add',net.affine(main,left[:,:,None,None]),
                     net.affine(ext,right[:,:,None,None]),name='all_photometry_planes')


def design():
    net = Network()
    img = image_planes(net)
    c = Circuit()
    rel, ab, clip = c.relu, c.abs, c.clip
    mn, mx = c.mn, c.mx
    rat, mag = c.ratio, c.magnitude
    minall = lambda v: c.reduce(mn,v)
    maxall = lambda v: c.reduce(mx,v)

    p = np.arange(64)*2 - 63
    y,x = np.meshgrid(p,p,indexing='ij')
    r = np.hypot(x,y)
    theta = np.arctan2(y,x)
    rr = r*r

    def gauss(s, cx=0., cy=0.):
        g = np.exp(-((x-cx)**2+(y-cy)**2)/(2*s*s))
        return g/g.sum()

    def measure(i, mask):
        row = np.zeros((7,64,64),np.float32)
        row[i] = mask
        return c.read(row.ravel())

    def mom(i, w, harmonic):
        return (measure(i,w*np.cos(harmonic*theta)), measure(i,w*np.sin(harmonic*theta)))

    # A discrete scale-space bank. Each aperture collects flux moments,
    # outer light, and compensated log-spiral quadrature responses.
    scales = [2.25,3.25,4.75,6.85,9.9,14.25,20.5,29.5]
    banks = []
    for s in scales:
        g = gauss(s)
        g2 = gauss(2*s)
        a = (measure(0,g)-measure(0,g2))*(s/5)**.85
        den_col = measure(0,gauss(1.5*s))
        colour = measure(2,gauss(1.5*s))
        core_flux = measure(0,gauss(.60*s))
        core_high = measure(1,gauss(.60*s))
        halo_flux = measure(0,gauss(2.8*s))
        # The aperture broadens as r², but high-frequency stars and the
        # very distant field are not given arbitrary moment leverage.
        wi = gauss(.85*s)*rr
        wi /= wi.sum()
        wm = gauss(1.50*s)*rr
        wm /= wm.sum()
        wo = gauss(2.30*s)*rr*np.exp(-.5*(r/(4.2*s))**6)
        wo /= wo.sum()
        asym_mask = gauss(1.7*s)*np.minimum(r/s,1.4)
        asym_mask /= asym_mask.sum()
        qbank = [(measure(i,w),*mom(i,w,2)) for i,w in [(6,wi),(6,wm),(6,wo),(5,wm),(0,wo)]]
        # The central curvature recovers elongation for a compact source
        # whose outer quadrupole is diluted by the PSF and background.
        curv_mask = gauss(.78*s+.75)
        curv_trace = measure(6, curv_mask*(2-rr/(.78*s+.75)**2))
        curv_x = measure(6, curv_mask*(x*x-y*y)/(.78*s+.75)**2)
        curv_y = measure(6, curv_mask*(2*x*y)/(.78*s+.75)**2)
        # m=4 is robustly positive for a very thin line; it helps reject
        # broad ovals with one contaminating neighbour in the aperture.
        line4 = mom(6,wm,4)
        # Compare differently spaced isophotes: a cuspy spheroid has
        # a longer tail at fixed intermediate surface brightness than a
        # blurred Gaussian, while a disc often shows a broad shoulder.
        p0 = measure(3,gauss(max(.72,.37*s)))
        p1 = measure(3,gauss(.95*s))
        p2 = measure(3,gauss(1.80*s))
        p3 = measure(3,gauss(2.9*s))
        arm_core = measure(0,gauss(.72*s))
        # A smooth exponential or Sersic radial light profile is convex
        # outside the nucleus. Positive radial curvature in the opposite
        # sense (a local crest) flags an arm, ring, or a clump. Pairing
        # opposite spokes downweights individual foreground stars.
        ridges = []
        for height, gap in [(1.36,.43),(2.04,.58)]:
            spokes = []
            for t in np.arange(12)*math.pi/6:
                ux,uy = math.cos(t),math.sin(t)
                sm = .90 + .18*s
                def spot(d):
                    return gauss(sm,d*s*ux,d*s*uy)
                spokes.append(measure(0,spot(height)-.50*(spot(height-gap)+spot(height+gap))))
            ridges.append(spokes)
        bar_color = mom(2,wi,2)
        bar_fluxmom = mom(0,wi,2)
        bar_innercol = measure(2,wi)
        bar_innerflux = measure(0,wi)
        spiral_outer = gauss(1.45*s)*(1-np.exp(-.5*(r/(.65*s))**4))
        spiral_outer /= spiral_outer.sum()
        extended = measure(5,spiral_outer)
        knotty = measure(0,spiral_outer) - extended
        # Contrast of an intermediate annulus relative to a compact nucleus.
        core_peak = measure(3,gauss(1.25))
        mid_disk = measure(5,gauss(1.25*s))
        compact_nucleus = measure(4,gauss(1.65))
        asym = (measure(0,asym_mask),*mom(0,asym_mask,1),*mom(0,asym_mask,3))
        outer_color = measure(2,wo)
        inner_color = measure(2,gauss(.60*s))
        outer_high = measure(1,wo)

        # Spiral matched filters in log-radius. Remove any constant,
        # circular m=0 profile term and smooth fixed-angle m=2 term.
        u = np.log(r/(1.65*s))
        envelope = np.exp(-.5*(u/.42)**2 - .5*(r/(4.1*s))**6)
        envelope /= envelope.sum()
        gram_modes = np.stack([np.ones_like(u), u, u*u,
                              np.cos(2*theta), np.sin(2*theta),
                              u*np.cos(2*theta),u*np.sin(2*theta)],axis=-1).reshape(-1,7)
        eflat = envelope.ravel()
        gram = gram_modes.T @ (eflat[:,None]*gram_modes)
        # Projection of analytic filter modes only, not of any images.
        def compensate(proto):
            residual = proto.ravel() - gram_modes @ np.linalg.solve(
                gram, gram_modes.T @ (eflat*proto.ravel()))
            return (eflat*residual).reshape(64,64)
        sw = []
        for pitch in (16,24,33,48):
            for hand in (-1,1):
                phase = 2*theta - hand*2*u/math.tan(math.radians(pitch))
                sw.append((measure(0,compensate(np.cos(phase))),
                           measure(0,compensate(np.sin(phase)))))
        arm_den = measure(0,envelope)
        # Compare annular profiles after removing low angular harmonics.
        # The m=4 response of a smooth elliptical is phase-locked to its
        # m=2 response; odd modes and a non-aligned m=4 are non-elliptical.
        annulus = np.exp(-.5*((r/s-1.36)/.57)**2)
        annulus /= annulus.sum()
        hf2 = mom(0,annulus,2)
        hf3 = mom(0,annulus,3)
        hf4 = mom(0,annulus,4)
        annuli_flux = measure(0,annulus)
        smallcol = measure(2, gauss(1.0*s))
        smallflux = measure(0, gauss(1.0*s))
        # Direct inner-bar template: flat major-axis light across the
        # central region, with either a round disc or turned outer arms.
        # Compensate the m=2 terms so a fixed, uniform ellipticity cancels.
        gamma = np.exp(-.5*((r/s-1.02)/.47)**2)
        delta = np.exp(-.5*((r/s-1.95)/.60)**2)
        gamma /= gamma.sum()
        delta /= delta.sum()
        bars = mom(0,gamma-.63*delta,2)
        bar_att = measure(0,gamma)
        banks.append(dict(a=a, f=den_col, col=colour, cf=core_flux, ch=core_high, hf=halo_flux,
                          qs=qbank, aa=asym, oc=outer_color, ic=inner_color, oh=outer_high,
                          sw=sw, sd=arm_den, cc=(curv_trace,curv_x,curv_y), l4=line4,
                          extended=extended, knotty=knotty, arm_core=arm_core,
                          cp=core_peak, md=mid_disk, nuc=compact_nucleus,
                          bar=bars, ba=bar_att, profile=(p0,p1,p2,p3),
                          ridge=ridges, bc=(bar_innercol,bar_innerflux,bar_color,bar_fluxmom),
                          ell=(hf2,hf3,hf4,annuli_flux), acol=(smallcol,smallflux)))

    # The companion branch probes multiple angular directions, measuring
    # a separate maximum, an intervening dip, and extended flux at the peak.
    center = measure(3,gauss(1.10))
    center_wide = measure(4,gauss(1.45))
    companion_rows = []
    for d in (5,8,12,18,26,36):
        for t in np.arange(16)*math.pi/8:
            ux,uy = math.cos(t),math.sin(t)
            peak = measure(3,gauss(.87+.043*d,d*ux,d*uy))
            saddle = measure(3,gauss(.98+.022*d,.54*d*ux,.54*d*uy))
            wide = measure(4,gauss(.95+.065*d,d*ux,d*uy))
            companion_rows.append((d,peak,saddle,wide))

    # Turn scale responses into smooth positive votes, and pool bounded
    # per-aperture features with those votes (not with learned weights).
    top = maxall([v['a'] for v in banks] + [0.])
    gates = []
    features = defaultdict(list)
    for s,b in zip(scales,banks):
        a, f, col, cf, ch, hf = [b[k] for k in ('a','f','col','cf','ch','hf')]
        qs,aa,oc,ic,oh,sw,sd = [b[k] for k in ('qs','aa','oc','ic','oh','sw','sd')]
        # Use a broad taper rather than a hard winning aperture so
        # photometry near a boundary remains continuous.
        gate = rat(rel(a-.82*top), .18*top+.0015)
        gates.append(gate)
        fv = {}
        fv['scale'] = math.log2(s/2.25)/4
        fv['colour'] = clip(rat(col,f+.009),-.9,.9)
        fv['red_gradient'] = clip(rat(ic,cf+.012)-rat(oc,qs[4][0]+.009),-.85,.85)
        fv['core'] = clip(rat(ch,cf+.014),0.,1.)
        fv['halo'] = clip(rat(hf,cf+.010),0.,1.)
        fv['disk_brightness'] = clip(rat(oh,qs[4][0]+.009),0.,1.)
        qv = []
        for z,qx,qy in qs[:4]:
            v1,v2 = rat(qx,z+.0023),rat(qy,z+.0023)
            qv.append((v1,v2,mag(v1,v2)))
        (ix,iy,ei),(jx,jy,ej),(kx,ky,ek),(dx,dy,ed) = qv
        z, qx, qy = b['cc']
        curv = clip(mag(rat(qx,mx(z,.02)+.035),rat(qy,mx(z,.02)+.035)),0.,1.)
        fv['curvature'] = curv
        fv['inner_e'],fv['mid_e'],fv['outer_e'] = ei,ej,ek
        fv['twist'] = mag(jx-kx,jy-ky)
        fv['inner_twist'] = mag(ix-jx,iy-jy)
        fv['coherent_shape'] = mag(.26*ix+.49*jx+.25*kx,.26*iy+.49*jy+.25*ky)
        l4x,l4y = b['l4']
        fv['thin_line'] = mag(rat(l4x,qs[1][0]+.003),rat(l4y,qs[1][0]+.003))
        fv['flatness'] = .77*minall([ej,ek]) + .23*ed
        az,ax,ay,tx,ty = aa
        fv['asym'] = mag(rat(ax,az+.005),rat(ay,az+.005))
        fv['triple'] = mag(rat(tx,az+.005),rat(ty,az+.005))
        # Normalize by some core light as well as arm-region light so
        # faint isophotes and a strong point source cannot fake arms.
        arm_ref = sd + .24*b['arm_core'] + .011
        spiral = [mag(rat(a,arm_ref),rat(b,arm_ref)) for a,b in sw]
        fv['tight'] = maxall(spiral[:4])
        fv['open'] = maxall(spiral[4:])
        p0,p1,p2,p3 = b['profile']
        fv['profile'] = clip(rat(p0-p1, rel(p0)+.045),0.,1.)
        fv['middle'] = clip(rat(p1-p2, rel(p0)+.045),0.,1.)
        fv['tail'] = clip(rat(p2-p3, rel(p0)+.045),0.,1.)
        fv['extension'] = clip(rat(b['extended'],cf+.015),0.,1.)
        fv['patched'] = clip(rat(b['knotty'],b['extended']+.023),0.,1.)
        fv['cusp'] = clip(rat(b['cp']-b['nuc'],b['cp']+.11),0.,1.)
        fv['nucleus'] = clip(rat(b['nuc']-b['md'],b['nuc']+.10),0.,1.)
        bx,by = b['bar']
        fv['annular_bar'] = mag(rat(bx,b['ba']+.012),rat(by,b['ba']+.012))
        bc,bf,(bx,by),(fx,fy) = b['bc']
        local_red = clip(rat(bc,bf+.018),-.85,.85)
        fv['red_bar'] = mag(rat(bx,bf+.018)-c.product(local_red,rat(fx,bf+.018)),
                            rat(by,bf+.018)-c.product(local_red,rat(fy,bf+.018)))
        rvalues = []
        for spokes in b['ridge']:
            contrast = maxall([mn(spokes[k],spokes[k+6]) for k in range(6)]+[0.])
            rvalues.append(contrast)
        crest = .72*mx(*rvalues) + .28*mn(*rvalues)
        fv['arm_ridges'] = rat(crest, .45*b['arm_core']+.036)
        # Relative resolved outer light in the blue band, independent of
        # the nucleus's colour. Red disks can still win on arm ridges.
        fv['outer_blue'] = clip(-rat(oc,qs[4][0]+.020),-.85,.85)
        ac,af = b['acol']
        fv['central_colour'] = clip(rat(ac,af+.010),-.85,.85)
        (ax,ay),(gx,gy),(hx,hy),ellflux = b['ell']
        ax,ay = rat(ax,ellflux+.012),rat(ay,ellflux+.012)
        # The second-order angular term of an elliptical light law gives
        # a fourth harmonic proportional to the square of its quadrupole.
        aligned_x = c.product(ax,ax)-c.product(ay,ay)
        aligned_y = 2*c.product(ax,ay)
        mismatch = mag(rat(hx,ellflux+.012)-.85*aligned_x,
                       rat(hy,ellflux+.012)-.85*aligned_y)
        oddball = mag(rat(gx,ellflux+.012),rat(gy,ellflux+.012))
        fv['nonellipse'] = .65*mismatch + .35*oddball
        for k,v in fv.items():
            features[k].append(v*gate if isinstance(v,float) else c.product(v,gate))
    total = sum(gates) + .02
    d = {k:rat(sum(v),total) for k,v in features.items()}

    cands = []
    for dist, pk, dip, wide in companion_rows:
        # At larger separations require a more substantial secondary.
        frac = .25 + .006*dist
        cands.append(minall([pk-frac*center-.021, 1.8*(pk-1.06*dip)-.020,
                             2.2*(wide-.49*pk)-.013, pk-.077]))
    pair = clip(rat(maxall(cands+[0.]), center+.085),0.,1.)
    offset = clip(rat(center_wide-center, center_wide+.09),0.,.5)

    # Hand-set, hierarchical morphology readout. The piecewise ramps
    # below encode scope-limited cues rather than unconditional votes.
    ramp = lambda x,a,b: clip((x-a)/(b-a),0.,1.)
    size, warm, grad = d['scale'],d['colour'],d['red_gradient']
    ei, em, eo = d['inner_e'],d['mid_e'],d['outer_e']
    core, halo, db = d['core'],d['halo'],d['disk_brightness']
    twist, inner_twist = d['twist'],d['inner_twist']
    asym, triple = d['asym'],d['triple']
    tight, loose = d['tight'],d['open']
    arms = mx(tight,loose)
    curv, coherent, line4, flat = [d[k] for k in
                         ('curvature','coherent_shape','thin_line','flatness')]
    ext, knot, cusp, nuc, ann_bar = [d[k] for k in
                         ('extension','patched','cusp','nucleus','annular_bar')]
    profile,middle,tail = (d[k] for k in ('profile','middle','tail'))
    oval = .73*coherent + .27*mx(curv,.71*em)

    # Structural texture should be an excess above the radial profile,
    # not raw colour or total brightness. The smooth and disc branches
    # have independent confidence margins for smooth and extended light.
    blue = 1-ramp(warm,-.19,.30)
    resolved = ramp(size,.20,.63)
    red_bulge = ramp(warm,.060,.37)
    arming = ramp(arms,.044,.14)
    extension = ramp(ext,.063,.26)
    shoulder = ramp(middle-profile*.69,.0,.155)
    cuspiness = ramp(profile-.39*tail,.23,.48)
    ridged = ramp(d['arm_ridges'],.024,.12)
    cool_outer = ramp(d['outer_blue'],-.21,.18)
    # Faint blue sky residuals are not disks, and blue smooth galaxies
    # exist. A radial colour change plus a resolved shape departure is
    # better evidence for young stars on a separate extended component.
    cool_disc = mn(cool_outer, .48*resolved+.75*extension)
    central_red = ramp(d['central_colour'],.06,.44)
    nonellipse = ramp(d['nonellipse'],.027,.125)
    twist_signal = ramp(twist,.023,.115)
    shape_change = mx(nonellipse, .85*twist_signal)
    gradient = ramp(grad,.056,.31)
    compact = 1-ramp(size,.25,.59)
    coherent_oval = c.product(compact, ramp(oval,.17,.40))
    quiet = 1-mx(ramp(asym,.08,.22),shape_change)
    quiet_red = c.product(central_red, quiet)
    struct = mx(mn(ridged,resolved+.38*nonellipse), .70*arming)
    disk = (.46*(resolved-.52) + .31*(blue-.49)
            + .48*(struct-.25) + .24*(extension-.53)
            + .10*(shoulder-.48) + .23*gradient
            + .20*(cool_disc-.46) + .40*(shape_change-.21)
            + .10*ramp(triple,.065,.20)
            - .19*cuspiness - .11*ramp(cusp,.16,.33)
            - .32*coherent_oval - .16*quiet_red)
    inclination = .58*flat + .25*em + .17*coherent
    # Intercept shape in the extended light, not in the PSF-broadened
    # nucleus: a resolved uniform axial feature is an edge-on disc.
    edge = (1.14*ramp(inclination,.35,.71) + .48*ramp(size,.22,.61)
            + .38*ramp(line4,.17,.51) + .16*ramp(db,.17,.58)
            - .34*ramp(asym,.115,.31) - .70
            - .45*c.product(compact,ramp(oval,.29,.56)))
    # A small round core, a peaked axial profile and a warm central
    # excess all contribute; uniform orange disks are not bulges alone.
    bulge = (.51*(ramp(profile,.12,.52)-.51)
             + .49*(ramp(eo-ei,.01,.31)-.43)
             + .28*(ramp(grad,.035,.34)-.37) + .48*(red_bulge-.52)
             + .21*(ramp(core,.42,.85)-.55) - .10*ramp(ext,.16,.37))

    # Bar: elongated central/intermediate light embedded in a rounder
    # or twisted outer disc, not merely a flat envelope at all radii.
    bar = (.40*ramp(em,.11,.46) + .59*ramp(em-eo,-.025,.155)
           + .52*ramp(ann_bar,.058,.225) + .29*ramp(twist,.066,.24)
           + .34*gradient - .50*ramp(inclination,.47,.74)
           + .42*ramp(d['red_bar'],.019,.091)
           - .30*ramp(asym,.13,.29) + .18*shoulder
           - .36*coherent_oval - .66)
    open_arm = (3.8*(loose-tight) + .46*(blue-.44)
                + .37*(cool_outer-.51) + .34*(ramp(triple,.052,.185)-.35)
                + .18*(ramp(knot,.45,.92)-.36)
                - .13*(red_bulge-.46) - .17*(ramp(profile,.24,.51)-.45))
    mess = (1.02*ramp(asym,.084,.28) + .48*ramp(triple,.069,.21)
            + .34*ramp(twist,.134,.35) + .36*ramp(offset,.032,.235)
            + .20*blue - .27*ramp(inclination,.47,.80)
            - .22*ramp(pair,.054,.22) - .67)
    merge = (1.65*ramp(pair,.032,.23) + .43*ramp(asym,.12,.33)
             + .37*ramp(offset,.035,.235) - .17*ramp(inclination,.49,.8) - .77)

    smooth = -.76*disk - .52*rel(edge) - .19*ramp(pair,.09,.25)
    spiral = .76*disk - .84*rel(edge)
    edge_on = .07*disk + edge
    # Overlapping shape gates correspond to progressively more flattened
    # isophotes rather than allocating a single category all elongations.
    scores = [mess, merge,
              smooth + .08 - 3.10*rel(oval-.094),
              smooth + .08 - 2.43*(rel(.148-oval)+rel(oval-.32)),
              smooth + .032 - 2.25*(rel(.363-oval)+rel(oval-.74))
                     - .30*resolved + .27*coherent_oval,
              spiral + .77*bar - .095,
              spiral - .56*open_arm - .25*rel(bar) - .100,
              spiral + .58*open_arm - .24*rel(bar) - .095,
              edge_on - .57*bulge, edge_on + .57*bulge]
    c.emit(net,img,scores)
    return net


if __name__ == '__main__':
    design().save(Path(sys.argv[1]))
