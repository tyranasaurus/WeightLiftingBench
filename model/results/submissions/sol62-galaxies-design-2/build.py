"""Generate a fixed, data-free galaxy morphology network."""
import json
import sys
from pathlib import Path
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def op(self, name, op, src, **settings):
        self.layers.append(dict(name=name, op=op,
                                inputs=[src] if isinstance(src, str) else src,
                                **settings))
        return name

    def affine(self, name, src, w, b=0, stride=1, padding=0):
        w = np.asarray(w, np.float32)
        b = np.broadcast_to(np.float32(b), (len(w),)).copy()
        self.weights[name + '.W'], self.weights[name + '.b'] = w, b
        if w.ndim == 2:
            return self.op(name, 'dense', src, out=len(w))
        return self.op(name, 'conv', src, out=len(w), kernel=list(w.shape[-2:]),
                       stride=stride, padding=padding)

    def relu(self, name, src):
        return self.op(name, 'relu', src)

    def pool(self, name, src, kind, size, stride):
        return self.op(name, kind, src, size=[size, size], stride=[stride, stride])

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'network.json').write_text(json.dumps({'layers': self.layers}, indent=2))
        np.savez(path / 'weights.npz', **self.weights)


# A small straight-line compiler for scalar arithmetic. Distinct branches of the
# calculation are usually independent and are batched into the same dense layer.
# Nonlinear arithmetic uses explicit piecewise-linear formulas, not learned units.
class Value:
    def __init__(self, terms=None, b=0.):
        self.terms = terms if terms is not None else {}
        self.b = float(b)

    def __add__(self, other):
        if not isinstance(other, Value):
            return Value(self.terms, self.b + other)
        out = self.terms.copy()
        for i, v in other.terms.items():
            out[i] = out.get(i, 0.) + v
            if abs(out[i]) < 1e-14:
                del out[i]
        return Value(out, self.b + other.b)

    __radd__ = __add__

    def __mul__(self, a):
        return Value({i: a*v for i, v in self.terms.items() if a*v}, self.b*a)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other


class Circuit:
    def __init__(self, n):
        self.nodes = [None] * n
        self.depth = [0] * n
        self.inputs = [Value({i: 1.}) for i in range(n)]

    def relu(self, x):
        if not isinstance(x, Value):
            return max(0., x)
        if not x.terms:
            return max(0., x.b)
        i = len(self.nodes)
        self.nodes.append(x)
        self.depth.append(1 + max(self.depth[j] for j in x.terms))
        return Value({i: 1.})

    def clip(self, x, lo=0., hi=1.):
        return lo + self.relu(x-lo) - self.relu(x-hi)

    def abs(self, x):
        return self.relu(x) + self.relu(-x)

    def max(self, x, y):
        return y + self.relu(x-y)

    def min(self, x, y):
        return x - self.relu(x-y)

    def maximum(self, args):
        args = list(args)
        while len(args) > 1:
            args = [self.max(a, b) for a, b in zip(args[::2], args[1::2])] + (args[-1:] if len(args)%2 else [])
        return args[0]

    def norm2(self, x, y):
        # Mean absolute radial projections; isotropic to within about 1.2%.
        v = np.arange(6) * (np.pi/6)
        return sum(self.abs(float(np.cos(t))*x + float(np.sin(t))*y) for t in v) * (np.pi/12)

    def pwl(self, x, knots, vals):
        slopes = np.diff(vals)/np.diff(knots)
        a = vals[0] + slopes[0]*(x-knots[0])
        return a + sum(float(d)*self.relu(x-k) for k, d in zip(knots[1:-1], np.diff(slopes)))

    def mul01(self, x, y, steps=8):
        # Polarization identity on [-1, 1]; interpolation error <= 1/(4 steps^2).
        def sq(z):
            return -1 + 1./steps + (-2 + 1./steps)*z + sum(
                (2./steps)*self.relu(z-i/steps) for i in range(1-steps, steps))
        return sq((x+y)*.5) - sq((x-y)*.5)

    def ratio(self, a, b, floor=.1):
        # Applied after the angular/radial sums. Approximate a positive ratio
        # using two centered log splines and an exponential spline.
        t = np.geomspace(.022, 8., 24)
        log = np.log(t)
        u = self.pwl(self.clip(a, .022, 8.), t, log)
        v = self.pwl(self.clip(b, floor, 8.), t, log)
        s = np.linspace(-4., 1.1, 26)
        return self.pwl(self.clip(u-v, -4., 1.1), s, np.exp(s))

    def finish(self, net, src, outputs):
        use = set()
        until = {}
        depth = 1 + max(self.depth[i] for t in outputs for i in t.terms)

        def visit(v, at):
            for j in v.terms:
                until[j] = max(until.get(j, 0), at)
                if j not in use:
                    use.add(j)
                    if self.nodes[j] is not None:
                        visit(self.nodes[j], self.depth[j])
        for t in outputs:
            visit(t, depth)

        cols = {i: [(i, 1)] for i in range(len(self.inputs)) if i in use}
        width = len(self.inputs)

        def apply(rows, name):
            w = np.zeros((len(rows), width), dtype=np.float32)
            for o, t in enumerate(rows):
                for j, v in t.terms.items():
                    for col, sign in cols[j]:
                        w[o, col] += sign*v
            return net.affine(name, src, w, [t.b for t in rows])

        for layer in range(1, depth):
            rows = []
            new_cols = {}
            for j in sorted(use):
                d = self.depth[j]
                if d == layer:
                    new_cols[j] = [(len(rows), 1)]
                    rows.append(self.nodes[j])
                elif d < layer < until[j]:
                    new_cols[j] = [(len(rows), 1)]
                    rows.append(Value({j: 1}))
                    if d == 0:
                        new_cols[j].append((len(rows), -1))
                        rows.append(Value({j: -1}))
            name = f'geometry_{layer:02d}'
            src = net.relu(name + '_hinges', apply(rows, name))
            cols, width = new_cols, len(rows)
        apply(outputs, 'class_scores')


def photometry(net):
    """Sky rejection and pooled samples of flux, color, and small-scale structure."""
    src = net.pool('sample2', 'image', 'avgpool', 2, 2)
    # Subtract the local annular sky level. The center coefficient is one.
    y, x = np.mgrid[-8:9, -8:9]
    r = np.hypot(y, x)
    sky = ((r >= 6) & (r <= 8.5)).astype(float)
    sky /= sky.sum()
    sky[8,8] -= 1
    src = net.affine('sky_subtracted', src, -sky[None,None]*np.eye(3)[:,:,None,None])
    L = np.array([.25, .50, .25])
    # Reconstruct smooth, low- and intermediate-isophote light, blue/red excess,
    # and a sharpness map from positive local high-frequency response.
    blur = np.outer([.20,.60,.20], [.20,.60,.20])
    dog = -np.outer([.25,.50,.25], [.25,.50,.25])
    dog[1,1] += 1.
    kern = np.zeros((11, 3, 3, 3))
    kern[:6] = L[None,:,None,None]*blur
    kern[6] = np.array([-1.15,0.,1.0])[:,None,None]*blur
    kern[7] = np.array([1.0,0.,-1.17])[:,None,None]*blur
    kern[8] = np.array([-.64,-.56,1.])[:,None,None]*blur
    kern[9] = L[:,None,None]*dog
    kern[10] = -kern[9]
    src = net.relu('isophote_magnitudes',net.affine('local_contrasts',src,kern,
                  [-.014,-.045,-.095,-.18,-.32,-.55,-.007,-.008,-.006,-.012,-.012],padding=1))
    w = np.zeros((8,11))
    # Winsorized maps keep compact foreground stars from dominating moments.
    w[0,[0,3,5]] = [1.,-.76,-.24]   # soft outer surface brightness
    w[1,[1,4,5]] = [1.,-.72,-.28]   # disk / intermediate isophote
    w[2,[2,4]] = [1.,-.72]          # bright structure
    w[3,6] = 1.                     # blue excess
    w[4,7] = 1.                     # red excess
    w[5,8] = 1.                     # strongly blue excess
    w[6,[9,10]] = [.56,.44]          # clumpiness
    w[7,[0,1,2,3,4]] = [1.0,.24,.30,-1.08,-.46]
    # w[7] is a low-isophote, saturating disk map, with slightly lower weight
    # on the bright nucleus so the arms are easier to resolve.
    return net.affine('photometric_maps', src, w[:,:,None,None])


def measurements(net, src):
    """Centered annular moments, compact nuclei and multiscale disk footprints."""
    y, x = np.mgrid[:48,:48] - 23.5
    r = np.hypot(x,y)
    a = np.arctan2(y,x)
    rows = []
    keys = []

    def add(key, band, mask):
        w = np.zeros((8,48,48),np.float32)
        w[band] = mask
        keys.append(key)
        rows.append(w.ravel())

    # Radial windows overlap so arm rotation can be followed between annuli.
    for j,s in enumerate((2.3, 4.15, 7.4, 13.0)):
        g = np.exp(-.5*(np.log(r/s)/.40)**2)
        g /= g.sum()
        for band in (0,1,3,4,6):
            add((j,band,0,0),band,10*g)
        for m in (1,2,3):
            add((j,0,m,0),0,10*g*np.cos(m*a))
            add((j,0,m,1),0,10*g*np.sin(m*a))

    for j,s in enumerate((1.1, 2.4, 5.0)):
        g = np.exp(-.5*(r/s)**2)
        g /= g.sum()
        for band in (0,3,4):
            add(('gauss',j,band),band,10*g)

    # Fixed-aperture windows probe luminosity profile and its quadrupole.
    for j,s in enumerate((2.8, 5.3, 9.0, 15.0)):
        g = np.exp(-.5*(r/s)**2)
        add(('mom',j,0,1),0,.33*g*(x*x-y*y)/s**2)
        add(('mom',j,0,2),0,.33*g*(2*x*y)/s**2)
        add(('mom',j,0,3),0,.33*g*r*r/s**2)

    # Compare diametrically opposite sectors. Matching wings give more
    # reliable evidence of a long disk than a single bright neighboring knot.
    for axis in range(16):
        t = axis*(np.pi/8)
        ang = np.cos(a-t)
        for j,s in enumerate((4.0,7.0,12.0,19.0)):
            g = np.exp(-.5*(np.log(r/s)/.30)**2 + 18*(ang-1))
            g /= g.sum()
            add(('wing',axis,j),0,10*g)
    # For an edge-on disk, the extrapolated minor-axis wings distinguish a
    # central thick component from the simple taper of a uniformly thin disk.
    # These masks compare symmetric off-plane positions without pixel maxima.
    for i in range(16):
        a0 = i*np.pi/8
        u = x*np.cos(a0)+y*np.sin(a0)
        t = -x*np.sin(a0)+y*np.cos(a0)
        for j,s in enumerate((2.0,3.3,5.1)):
            for k,d in enumerate((0.,1.15,2.35)):
                mask = np.exp(-.5*((u-d*s)/(.49*s))**2 - .5*(t/(.31*s))**2)
                mask /= mask.sum()
                add(('profile',i,j,k),0,10*mask)
    # Survey pairs: seek a distinct off-center nucleus by center-surround
    # filtering. These translate across an inner grid, with no image templates.
    # The pair-color and two-sided angular evidence are in the readout.
    for axis in range(16):
        t = axis*(np.pi/8)

        def spot(d, s, offset=0.):
            dx, dy = d*np.cos(t)-offset*np.sin(t), d*np.sin(t)+offset*np.cos(t)
            sm = np.exp(-((x-dx)**2+(y-dy)**2)/(2*s**2))
            return sm / sm.sum()

        for j,d in enumerate((2.8,4.6,7.2,10.5)):
            g = spot(d,.92+.035*d)
            h = spot(d,1.95+.040*d)
            saddle = spot(max(0.,d-max(1.45,.32*d)),.98)
            flanks = .5*(spot(d,1.1,-2.15)+spot(d,1.1,2.15))
            add(('comp',axis,j,0),1,10*(g-h))
            add(('comp',axis,j,1),0,10*g)
            add(('comp',axis,j,2),0,10*h)
            add(('comp',axis,j,3),1,10*(g-saddle))
            add(('comp',axis,j,4),0,10*flanks)
            add(('comp',axis,j,5),1,10*g)
    # Phase-invariant logarithmic arm bank. In shear coordinates, arms of
    # pitch p follow phi=phi0 +/- log(r/r0)/tan(p). Subtract the zero-pitch
    # (constant-position-angle) ellipticity response within each radial band.
    for j,s in enumerate((4.0,6.8,11.0)):
        lr = np.log(r/s)
        g = np.exp(-.5*(lr/.48)**2) * (1-np.exp(-.5*(r/1.75)**2))
        g /= g.sum()
        for m,pitches in ((2,(0.,20.,38.)),(3,(0.,27.))):
            for p in pitches:
                for sign in ((0,) if p==0 else (-1,1)):
                    shear = 0 if p==0 else sign/np.tan(np.deg2rad(p))
                    h = np.exp(1j*m*(a-shear*lr))
                    if p != 0:
                        # Remove correlation with a fixed oval, with an extra
                        # exponential weight for the radial decline of a disk.
                        gw = g*np.exp(-r/(1.2*s))
                        hs = np.exp(-1j*m*shear*lr)
                        h -= np.exp(1j*m*a)*np.sum(gw*hs)/np.sum(gw)
                    add(('arm',j,m,p,sign,0),7,10*g*h.real)
                    add(('arm',j,m,p,sign,1),7,10*g*h.imag)
    out = net.affine('radial_angular_photometry', src, np.asarray(rows))
    return out, keys


def classify(net, src, keys):
    c = Circuit(len(keys))
    v = dict(zip(keys,c.inputs))
    sat = c.clip
    mag = c.norm2
    rel = c.ratio
    mn, mx = c.min, c.max
    ramp = lambda x, a, b: sat((x-a)*(1/(b-a)))
    ring = lambda j,b=0: v[j,b,0,0]
    core = lambda j,b=0: v['gauss',j,b]
    vec = lambda j,m,b=0: (v[j,b,m,0],v[j,b,m,1])
    nvec = lambda j,m,b=0: mag(*vec(j,m,b))

    # An adaptive aperture: measure elongation where there is resolved flux,
    # using smaller apertures for compact sources and large apertures for disks.
    large = ramp(ring(2,1), .22,.75)
    middle = ramp(ring(1,1), .18,.72)
    huge = ramp(ring(3,1), .26,.82)
    eccs = []
    for j in range(4):
        o = lambda k: v['mom',j,0,k]
        eccs.append(sat(rel(mag(o(1),o(2)), o(3), floor=.085)))
    ecc = eccs[0] + c.mul01(middle,eccs[1]-eccs[0]) + c.mul01(large,eccs[2]-eccs[1]) + c.mul01(huge,eccs[3]-eccs[2])
    ecc = sat(ecc)
    eq = [sat(rel(nvec(j,2),ring(j),floor=.14)) for j in range(4)]
    inner_ell = eq[0] + c.mul01(middle,eq[1]-eq[0])
    outer_ell = eq[1] + c.mul01(large,eq[2]-eq[1]) + c.mul01(huge,eq[3]-eq[2])
    quadrupole_rise = outer_ell-inner_ell
    bar_excess = mx(eq[0]-eq[1], mn(eq[1]-eq[2], ramp(ring(2),.20,.65)))
    bar_excess = mx(bar_excess, mn(eq[2]-eq[3], ramp(ring(3),.22,.65)))
    sectors = [[mn(v['wing',i,j], v['wing',i+8,j]) for i in range(8)] for j in range(4)]
    coherent = [c.maximum(mn(sectors[j][i], sectors[j+1][i]) for i in range(8)) for j in range(3)]
    edge_votes = []
    bulge_votes = []
    bar_votes = []
    for j in range(4):
        for i in range(8):
            axis = sectors[j][i]
            cross = .5*(v['wing',(i+4)%16,j]+v['wing',(i+12)%16,j])
            # Weight the minor-axis test by the brightness of both major-axis tips.
            line = mn(ramp(axis,.34,1.05),ramp(axis-3.4*cross,.08,.74))
            if j < 3:
                tip = sectors[j+1][i]
                line = mn(.67*line+.33*ramp(tip,.12,.66), ramp(axis-2.1*cross,.025,.58))
            edge_votes.append(line - (0.16 if j==0 else 0.))
            if j > 0:
                inner_side = mn(v['wing',(i+4)%16,0],v['wing',(i+12)%16,0])
                inner_axis = sectors[0][i]
                excess = inner_side - .35*inner_axis - .55*cross
                # The major axis must continue past the putative round bulge.
                bulge_votes.append(mn(ramp(excess,.005,.43),ramp(axis,.19,.83)))
            if j <= 2:
                side = .5*(v['wing',(i+4)%16,j+1]+v['wing',(i+12)%16,j+1])
                outer = sectors[j+1][i]
                narrow = ramp(axis-1.35*cross,.075,.70)
                disk_wings = ramp(side,.045,.31)
                widening = ramp(axis-1.55*cross-.54*(outer-1.3*side),.08,.47)
                bar_votes.append(mn(narrow,mn(disk_wings,widening))*(1.-.06*j))
    line = c.maximum(edge_votes)
    off_plane_bulge = c.maximum(bulge_votes)
    bar_light = c.maximum(bar_votes)
    profile_bulges = []
    aligned_disks = []
    for i in range(8):
        for j in range(3):
            p = lambda axis,k: v['profile',axis%16,j,k]
            major = mn(p(i,2),p(i+8,2))
            minor = .5*(p(i+4,2)+p(i+12,2))
            near_u = .5*(p(i,1)+p(i+8,1))
            near_v = mn(p(i+4,1),p(i+12,1))
            peak = .5*(p(i,0)+p(i+4,0))
            disk_shape = mn(ramp(major-2.0*minor,.10,.56),ramp(major,.16,.61))
            # The intermediate minor-axis isophote must be wider than a thin
            # central ridge would predict, but not a continuation of the wings.
            bump = near_v - .27*near_u - .52*minor
            bulge_ratio = rel(c.relu(bump),peak,.30)
            prof = ramp(bulge_ratio,.018,.245)
            profile_bulges.append(mn(prof, disk_shape))
            aligned_disks.append(mn(disk_shape, ramp(near_u-1.25*near_v,.025,.55)))
    profile_bulge = c.maximum(profile_bulges)
    aligned_disk = c.maximum(aligned_disks)

    # The two-fold component of an ellipse has constant phase. A spiral twists
    # between rings. Reject isolated quadrupoles by requiring signal in both.
    twists = []
    for j in range(3):
        x, y = vec(j,2); u, w = vec(j+1,2)
        l, h = nvec(j,2), nvec(j+1,2)
        # Normalize components before taking their cross product.
        def sgnratio(t,b):
            return rel(c.relu(t),b,.12) - rel(c.relu(-t),b,.12)
        x,y,u,w = (sat(sgnratio(t,b),-1.,1.) for t,b in ((x,l),(y,l),(u,h),(w,h)))
        cross = c.abs(c.mul01(x,w)-c.mul01(y,u))
        both = mn(ramp(l,.055,.27),ramp(h,.060,.26))
        twists.append(c.mul01(both, sat(cross)))
    twist = c.maximum(twists)

    def arm(j,m,p,sign):
        return mag(v['arm',j,m,p,sign,0],v['arm',j,m,p,sign,1])

    arm_votes = {20.: [],38.: [],27.: []}
    for j in range(3):
        for m,p in ((2,20.),(2,38.),(3,27.)):
            l,h = arm(j,m,p,-1),arm(j,m,p,1)
            imp = mx(l,h) - .72*mn(l,h) - .11*arm(j,m,0.,0)
            # Absolute contrast floor, then a relative arm excess. A coherent
            # handed mode is more useful than a single knot present in both.
            arm_votes[p].append(mn(ramp(imp,.022,.14),
                                    ramp(rel(c.relu(imp),ring(j+1),.16),.04,.24)))
    tight_pattern = c.maximum(arm_votes[20.])
    open_pattern = c.maximum(arm_votes[38.] + [x*.83 for x in arm_votes[27.]])
    arms = mx(tight_pattern,open_pattern)

    # Outer light gradients and color separate diffuse star-forming disks from
    # the concentrated, typically warmer light of early-type galaxies.
    disk = ring(1)+1.4*ring(2)+.6*ring(3)
    db = ring(1,3)+1.4*ring(2,3)+.6*ring(3,3)
    dr = ring(1,4)+1.4*ring(2,4)+.6*ring(3,4)
    blue = rel(db,disk,.24) - .64*rel(dr,disk,.24)
    warm = rel(core(1,4),core(1),.2) - .55*rel(core(1,3),core(1),.2)
    blue_disk = ramp(blue,.01,.28)
    is_warm = ramp(warm,.075,.39)
    gradient = ramp(blue+.95*warm,.065,.42)
    extended = ramp(.38*ring(1,1)+ring(2,1)+.85*ring(3,1),.16,.96)
    resolved = ramp(.65*ring(1)+ring(2),.31,1.12)
    concentration = rel(core(0)-.36*core(2),core(2),.28)
    bulge = ramp(concentration,.65,1.60)
    fine = rel(ring(0,6)+ring(1,6)+ring(2,6),ring(0)+ring(1)+ring(2),.4)
    clumps = ramp(fine,.012,.10)

    # The smooth-galaxy hypothesis is even under a half turn. Odd harmonics
    # capture lopsided tidal tails, displaced cores and asymmetric knots.
    a1 = rel(1.0*nvec(0,1)+1.35*nvec(1,1)+1.25*nvec(2,1),
             ring(0)+1.35*ring(1)+1.25*ring(2), .45)
    a3 = rel(.8*nvec(0,3)+1.25*nvec(1,3)+1.4*nvec(2,3),
             ring(0)+1.25*ring(1)+1.4*ring(2), .45)
    asym = ramp(a1 + .65*a3,.13,.40)
    strong_asym = ramp(a1 + a3,.23,.52)
    outer_asym = ramp(rel(nvec(2,1)+.5*nvec(2,3),ring(2),.16),.21,.60)

    pair_votes = []
    for i in range(16):
        for j in range(4):
            z = lambda k: v['comp',i,j,k]
            inv = v['comp',(i+8)%16,j,2]
            # A compact peak needs a dip toward the primary and an envelope.
            # The dip rejects a single elongated galaxy or an ordinary spiral arm.
            knot = ramp(z(0) - .024*core(0),.032,.34)
            saddle = ramp(z(3) + .10*z(0),-.09,.35)
            envelope = ramp(z(2) - .41*z(1),.02,.33)
            imbalance = ramp(z(2) - .77*inv,.018,.43)
            distinct = ramp(z(1)-z(4),.08,.47)
            vote = mn(knot,mn(saddle,mn(envelope,distinct)))
            bridge = ramp(z(5),.17,1.1)
            pair_votes.append(mn(vote, .63*imbalance+.37*bridge)*(1.-.075*j))
    companion = c.maximum(pair_votes)

    # Scores are fuzzy votes, not probabilities. The smooth classes partition
    # axis ratio; the edge-on classes partition central light/bulge contrast.
    flatten = ramp(ecc,.50,.81)
    thin = ramp(ecc,.63,.87)
    broad = ramp(.6*ring(2,1)+ring(3,1),.15,.87)
    wings = ramp(coherent[1]+.63*coherent[2],.20,.97)
    flat_core = ramp(eccs[0],.36,.72)
    edge = (.54 + 1.33*line + .65*flatten + .32*wings + .17*flat_core
            + .17*aligned_disk - .29*asym - .18*twist - .27*arms)
    # All three geometric tests see the same central light. Pool them as
    # alternative evidence instead of counting that light three times.
    bulge_shape = c.maximum([profile_bulge, off_plane_bulge,
                             .75*ramp(quadrupole_rise,.11,.49)])
    bulge_edge = (.74*bulge_shape + .21*ramp(concentration,1.33,2.30)
                  + .24*is_warm - .32*flat_core)
    # Prefer a compact, symmetric, unstructured model until there is
    # appreciable evidence for resolved disk / arms or interaction.
    smooth = (2.23 + .49*is_warm + .36*bulge - .71*blue_disk - .54*broad
              - .66*asym - .18*twist - .12*clumps - .15*gradient - .35*arms)
    spiral = (1.11 + .59*extended + .38*broad + .27*resolved + .20*twist
              + .27*gradient + .22*blue_disk + .37*arms - .53*line - .30*strong_asym)
    # An elongated inner light distribution must co-exist with a broader disk.
    bar = (1.70*bar_light + .78*ramp(bar_excess,.04,.29)
           + .26*ramp(inner_ell,.09,.27) + .13*gradient - .36*line - .25*strong_asym)
    winding = (.23*bulge + .31*is_warm + .14*gradient + .12*(1-asym)
               - .36*blue_disk - .12*clumps - .10*twist + .25*tight_pattern - .26*open_pattern)
    blue_cloud = ramp(blue,.045,.30)
    # A compact blue cloud with an irregular outer profile supports disturbed
    # over a grand-design spiral. Winding structure and a broad disk disfavor it.
    diffuse_irregular = mn(blue_cloud,mx(asym,.72*outer_asym))
    disturbed = (1.38 + 1.37*asym + .79*blue_cloud + .32*diffuse_irregular
                 + .27*clumps + .22*outer_asym - .24*bulge - .34*line
                 - .66*companion - .30*broad - .39*arms)
    merger = (1.29 + 2.27*companion + .87*strong_asym + .43*outer_asym
              + .12*clumps - .29*line - .17*blue_disk - .31*arms)

    scores = [
        disturbed,
        merger,
        smooth - 2.3*c.relu(ecc-.215),
        smooth - .06 - 1.9*c.relu(.23-ecc) - 2.4*c.relu(ecc-.515),
        smooth - .055 - 3.9*c.relu(.525-ecc) - 2.1*c.relu(ecc-.79) - .36*wings,
        spiral + bar - .34,
        spiral + winding - .10,
        spiral + .25 - .86*winding + .06*blue_disk,
        edge + .17 - .62*bulge_edge + .09*thin,
        edge - .35 + .95*bulge_edge,
    ]
    c.finish(net,src,scores)


def main(directory):
    net = Network()
    maps = photometry(net)
    src, keys = measurements(net, maps)
    classify(net, src, keys)
    net.save(Path(directory))


if __name__ == '__main__':
    main(sys.argv[1])
