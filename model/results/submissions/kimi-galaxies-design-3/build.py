"""Hand-designed galaxy morphology network; reads no files and fits nothing."""
import json, sys
from pathlib import Path
import numpy as np

N = 64
Y, X = np.mgrid[:N, :N].astype(np.float64) - (N - 1) / 2
RADII = np.geomspace(0.8, 18.0, 8)
ANGLES8 = np.arange(8) * np.pi / 8
ANGLES4 = np.arange(4) * np.pi / 4


def rotate(a):
    c, s = np.cos(a), np.sin(a)
    return c * X + s * Y, -s * X + c * Y


def rgb(bulge, disk, arms=None, bar=None, kind="spiral"):
    if kind == "smooth":
        return np.stack([bulge, .82 * bulge, .60 * bulge])
    if kind == "blue":
        return np.stack([.65 * disk, .82 * disk, disk])
    if arms is None: arms = np.zeros_like(disk)
    if bar is None: bar = np.zeros_like(disk)
    return np.stack([bulge + .75 * bar + .55 * disk + .35 * arms,
                     .82 * bulge + .85 * bar + .80 * disk + .65 * arms,
                     .58 * bulge + .72 * bar + disk + arms])


def smooth(a, q, bulge, R, profile=0):
    x, y = rotate(a)
    r = np.sqrt(x*x + (y/q)**2 + .04)
    disk = np.exp(-r/R) if profile == 0 else np.exp(-np.sqrt(r/R))
    core = np.exp(-.5*(r/(.30*R+.20))**2)
    return rgb(disk + bulge*core, disk, kind="smooth")


def edge(a, q, bulge, R):
    x, y = rotate(a)
    h = q*R + .30
    disk = np.exp(-np.abs(x)/R) * np.exp(-.5*(y/h)**2)
    core = np.exp(-.5*(x*x+y*y)/(.28*R+.25)**2)
    if bulge == 0: return rgb(None, disk, kind="blue")
    return rgb(bulge*core, disk)


def spiral(a, q, pitch, R, barred=False, variant=0):
    x, y = rotate(a)
    r = np.sqrt(x*x + (y/q)**2 + .09)
    disk = np.exp(-r/R)
    core = np.exp(-.5*(r/(.27*R+.20))**2)
    theta = np.arctan2(y/q, x)
    wave = .5 + .5*np.cos(2*theta-pitch*np.log(r/(.55*R)))
    arm = wave**5 * (r/(r+.30*R)) * np.exp(-r/(1.25*R))
    strength = 2.0 if pitch < 3 else 1.2
    bar = np.zeros_like(disk)
    if barred:
        length = R*(1.05 if variant else .75)
        width = .16*R+.25
        bar = (2.3 if variant else 1.5)*np.exp(-.5*((x/length)**6+(y/width)**2))
        arm *= .65
    return rgb(2.2*core, disk, strength*arm, bar)


def irregular(seed):
    rng = np.random.default_rng(seed)
    R = float(np.exp(rng.uniform(np.log(.9), np.log(15.0))))
    a, q = rng.uniform(0, np.pi), rng.uniform(.5, 1.0)
    x, y = rotate(a)
    r = np.sqrt(x*x+(y/q)**2+.09)
    theta = np.arctan2(y/q, x)
    disk = np.exp(-r/R)*np.maximum(.12, 1+.55*np.cos(theta+rng.uniform(0, 2*np.pi))
                                      +.30*np.cos(3*theta+rng.uniform(0, 2*np.pi)))
    core = np.exp(-.5*(r/(.25*R+.2))**2)
    clumps = np.zeros_like(disk)
    for _ in range(3):
        cx, cy = rng.normal(0, .45*R, 2)
        size = rng.uniform(.15, .40)*R+.25
        clumps += rng.uniform(.4, 1.4)*np.exp(-((X-cx)**2+(Y-cy)**2)/(2*size*size))
    return rgb(.7*core, .65*disk, 1.8*clumps)


def merger(seed):
    rng = np.random.default_rng(seed)
    R = float(np.exp(rng.uniform(np.log(.8), np.log(10.0))))
    a, sep, ratio = rng.uniform(0, np.pi), rng.uniform(.45, 1.8)*R, rng.uniform(.3, 1.0)
    u, v = rotate(a)
    out = np.zeros((3, N, N))
    for j, sign in enumerate((-1, 1)):
        xx, yy = u-sign*sep/2, v
        aa = a+rng.uniform(-1.0, 1.0)
        xp, yp = np.cos(aa-a)*xx+np.sin(aa-a)*yy, -np.sin(aa-a)*xx+np.cos(aa-a)*yy
        q = rng.uniform(.45, 1.0)
        rr = np.sqrt(xp*xp+(yp/q)**2+.09)
        size = R*(1.0 if j == 0 else ratio)
        body = np.exp(-rr/(size+.2))+1.2*np.exp(-.5*(rr/(.3*size+.2))**2)
        weight = 1.0 if j == 0 else ratio
        col = np.array([1.0, .82, .60]) if j == 0 else np.array([.55, .78, 1.0])
        out += weight*col[:, None, None]*body
    rr = np.sqrt(u*u+v*v+.1)
    th = np.arctan2(v, u)
    tail = np.exp(-rr/(1.8*R))*np.maximum(0, np.cos(th-.8*np.log(rr+.5)))**8
    return out+np.array([.25, .4, .6])[:, None, None]*tail


def unit_template(t):
    t = np.asarray(t, dtype=np.float64).copy()
    t -= t.mean(axis=(1, 2), keepdims=True)
    return (t/np.sqrt(np.sum(t*t))/(64*np.sqrt(3))).astype(np.float32)


templates = [[] for _ in range(10)]
for j in range(128): templates[0].append(irregular(91000+j))
for j in range(128): templates[1].append(merger(92000+j))
for R in np.geomspace(.65, 20.0, 16):
    for profile in range(2):
        for bulge in (.25, 1.0, 3.0, 8.0): templates[2].append(smooth(0, 1, bulge, R, profile))
for R in RADII:
    for a in ANGLES8:
        for q in (.62, .80): templates[3].append(smooth(a, q, 1.8, R))
for R in RADII:
    for a in ANGLES8:
        for q in (.27, .43): templates[4].append(smooth(a, q, 2.5, R))
for R in RADII:
    for a in ANGLES4:
        for q in (.72, 1.0):
            for v in range(2): templates[5].append(spiral(a, q, 2.0, R, True, v))
for R in RADII:
    for a in ANGLES4:
        for q in (.70, 1.0):
            for p in (4.2, 7.0): templates[6].append(spiral(a, q, p, R))
for R in RADII:
    for a in ANGLES4:
        for q in (.65, 1.0):
            for p in (1.15, 2.2): templates[7].append(spiral(a, q, p, R))
for R in RADII:
    for a in ANGLES8:
        for q in (.10, .20): templates[8].append(edge(a, q, 0, R))
for R in RADII:
    for a in ANGLES8:
        for b in (1.5, 5.0): templates[9].append(edge(a, .16, b, R))
TW = np.stack([unit_template(t) for g in templates for t in g]).reshape(1280, -1)

# ------------------- explicit image measurements -------------------
# p contains positive normalized luminance above four fixed signal levels.
# hp contains absolute responses of three isotropic high-pass filters.
names, specs = [], []
def add(source, name, channel, spatial, scale=1.0):
    names.append(name)
    specs.append((source, channel, np.asarray(spatial, dtype=np.float64)*scale))
    return len(names)-1

U, V = X/16, Y/16
RR = np.sqrt(X*X+Y*Y+.01)
zones = {
    "total": np.exp(-RR**2/(2*24**2)),
    "inner": np.exp(-RR**2/(2*4.5**2)),
    "mid": np.exp(-RR**2/(2*9**2)),
    "outer": (1-np.exp(-RR**2/(2*7**2)))*np.exp(-RR**2/(2*22**2)),
}
for j in range(4):
    for z, w in zones.items():
        add("p", f"M_{j}_{z}", j, w)
        add("p", f"R2_{j}_{z}", j, (U*U+V*V)*w)
        add("p", f"Qc_{j}_{z}", j, (U*U-V*V)*w)
        add("p", f"Qs_{j}_{z}", j, 2*U*V*w)
    for z in ("total", "outer"):
        w = zones[z]
        th = np.arctan2(V, U)
        add("p", f"X_{j}_{z}", j, U*w)
        add("p", f"Y_{j}_{z}", j, V*w)
        add("p", f"T3c_{j}_{z}", j, np.cos(3*th)*w)
        add("p", f"T3s_{j}_{z}", j, np.sin(3*th)*w)

pitches = (0.0, 1.2, 2.2, 4.5, 7.0)
for j in (1, 2):
    for ri, R in enumerate((4.0, 9.0, 18.0)):
        w = np.exp(-RR/R)*(1-np.exp(-RR**2/(.35*R)**2))
        th = np.arctan2(Y, X)
        for pitch in pitches:
            phase = 2*th-pitch*np.log(RR/(.5*R))
            tag = f"S_{j}_{ri}_{pitch}"
            add("p", "D"+tag, j, w)
            add("p", "C"+tag, j, w*np.cos(phase))
            add("p", "S"+tag, j, w*np.sin(phase))

# Three formulas on a common 9x9 support: Laplacian and two differences of
# Gaussians.  Opposite-signed pairs become exact absolute values after ReLU.
kernels = []
k = np.zeros((9, 9)); k[4, 4] = 4; k[3, 4] = k[5, 4] = k[4, 3] = k[4, 5] = -1
kernels.append(k)
for small, large in ((.7, 1.6), (1.3, 2.8)):
    yy, xx = np.mgrid[:9, :9]-4
    g1 = np.exp(-(xx*xx+yy*yy)/(2*small*small)); g1 /= g1.sum()
    g2 = np.exp(-(xx*xx+yy*yy)/(2*large*large)); g2 /= g2.sum()
    kernels.append(g2-g1)
for f in range(3):
    for z in ("total", "inner", "outer"):
        add("hp", f"HP_{f}_{z}_plus", 2*f, zones[z])
        add("hp", f"HP_{f}_{z}_minus", 2*f+1, zones[z])
for z, w in zones.items():
    for ch, cname in enumerate(("R", "G", "B")):
        add("raw", f"{cname}_{z}", ch, w, 20.0)
idx = {n:i for i, n in enumerate(names)}
F = len(names)

# Absolute values of rotated projections approximate vector magnitudes while
# retaining exact rotation covariance before the final rotational average.
derived = []
def project(name, terms):
    derived.append((name, terms)); return len(derived)-1
for j in range(4):
    for z in zones:
        for k in range(8):
            a = k*np.pi/8
            project(f"E_{j}_{z}_{k}", [(idx[f"Qc_{j}_{z}"], np.cos(2*a)),
                                         (idx[f"Qs_{j}_{z}"], np.sin(2*a))])
    for z in ("total", "outer"):
        for k in range(4):
            a = k*np.pi/4
            project(f"A_{j}_{z}_{k}", [(idx[f"X_{j}_{z}"], np.cos(a)),
                                         (idx[f"Y_{j}_{z}"], np.sin(a))])
            project(f"T_{j}_{z}_{k}", [(idx[f"T3c_{j}_{z}"], np.cos(3*a)),
                                         (idx[f"T3s_{j}_{z}"], np.sin(3*a))])
for j in (1, 2):
    for ri in range(3):
        for pitch in pitches:
            tag = f"S_{j}_{ri}_{pitch}"
            for k in range(4):
                a = k*np.pi/4
                project(f"P{tag}_{k}", [(idx["C"+tag], np.cos(a)),
                                          (idx["S"+tag], np.sin(a))])
D = len(derived)

# Conditions are homogeneous comparisons between measurements, e.g.
# quadrupole/radial moment > 0.72.  A clipped-ReLU circuit turns each into a
# scale-independent soft binary indicator without division or fitted weights.
conditions = []
def condition(name, terms):
    conditions.append((name, terms)); return len(conditions)-1

def amplitude_terms(prefix, n, scale=1.0):
    return [(F+d, scale*np.pi/(2*n)) for d in range(D) if derived[d][0].startswith(prefix+"_")]

def comparison(name, prefix, n, denominator, level, greater=True):
    terms = amplitude_terms(prefix, n)
    terms.append((idx[denominator], -level))
    if not greater: terms = [(i, -w) for i, w in terms]
    return condition(name, terms)

for j in (1, 2):
    for z in zones:
        for lev in (.22, .50, .72, .85):
            for gt in (True, False):
                comparison(f"ell_{j}_{z}_{lev}_{gt}", f"E_{j}_{z}", 8, f"R2_{j}_{z}", lev, gt)
for j in (0, 1, 2):
    for lev in (.03, .08, .18, .35):
        for gt in (True, False):
            terms = [(idx[f"M_{j}_inner"], 1), (idx[f"M_{j}_total"], -lev)]
            if not gt: terms = [(i, -w) for i, w in terms]
            condition(f"conc_{j}_{lev}_{gt}", terms)
for j in (1, 2):
    for kind, prefix in (("asym", "A"), ("three", "T")):
        for z in ("total", "outer"):
            for lev in (.04, .12, .25):
                comparison(f"{kind}_{j}_{z}_{lev}", f"{prefix}_{j}_{z}", 4, f"M_{j}_{z}", lev)
    for ri in range(3):
        for pitch in pitches:
            tag = f"S_{j}_{ri}_{pitch}"
            for lev in (.12, .25):
                for gt in (True, False):
                    comparison(f"pitch_{j}_{ri}_{pitch}_{lev}_{gt}", "P"+tag, 4, "D"+tag, lev, gt)
for f in range(3):
    for lev in (.05, .15, .30):
        for gt in (True, False):
            terms = [(idx[f"HP_{f}_outer_plus"], 1), (idx[f"HP_{f}_outer_minus"], 1),
                     (idx["M_1_total"], -lev)]
            if not gt: terms = [(i, -w) for i, w in terms]
            condition(f"texture_{f}_{lev}_{gt}", terms)
for z in zones:
    for lev in (-.10, .10, .30):
        for gt in (True, False):
            terms = [(idx[f"R_{z}"], 1-lev), (idx[f"B_{z}"], -(1+lev))]
            if not gt: terms = [(i, -w) for i, w in terms]
            condition(f"red_{z}_{lev}_{gt}", terms)
H = len(conditions)

# Rule-table readout.  Each list is averaged so that rules do not depend on
# how many redundant thresholds/scales implement the same physical concept.
RW = np.zeros((10, H), np.float64)
def rule(c, weight, selected):
    selected = list(selected)
    for h in selected: RW[c, h] += weight/len(selected)

def select(prefix):
    return [h for h, (n, _) in enumerate(conditions) if n.startswith(prefix)]

for j in (1, 2):
    rule(2, 1.2, select(f"ell_{j}_total_0.22_False"))
    rule(2, .7, select(f"ell_{j}_inner_0.22_False"))
    rule(3, .8, select(f"ell_{j}_total_0.22_True"))
    rule(3, 1.0, select(f"ell_{j}_total_0.72_False"))
    rule(4, 1.0, select(f"ell_{j}_total_0.5_True"))
    rule(4, .8, select(f"ell_{j}_total_0.85_False"))
    rule(5, .8, select(f"ell_{j}_inner_0.5_True"))
    rule(5, .8, select(f"ell_{j}_outer_0.5_False"))
    rule(6, .5, select(f"ell_{j}_total_0.5_False"))
    rule(7, .5, select(f"ell_{j}_total_0.72_False"))
    for c in (2, 3, 4, 5, 6, 7):
        rule(c, -1.5, select(f"ell_{j}_outer_0.72_True"))
    rule(8, 1.4, select(f"ell_{j}_outer_0.72_True"))
    rule(8, 1.2, select(f"ell_{j}_inner_0.5_True"))
    rule(8, .5, select(f"ell_{j}_total_0.85_True"))
    rule(9, 1.5, select(f"ell_{j}_outer_0.72_True"))
    rule(9, 1.3, select(f"ell_{j}_inner_0.5_False"))
    rule(9, -.6, select(f"ell_{j}_inner_0.72_True"))
    for c in (0, 1):
        rule(c, 1.5, select(f"asym_{j}_total_0.12"))
        rule(c, .8, select(f"three_{j}_outer_0.12"))
    rule(1, 1.2, select(f"asym_{j}_outer_0.12"))
    rule(0, .8, select(f"three_{j}_total_0.12"))
    for c in (2, 3, 4):
        rule(c, -1.0, select(f"asym_{j}_total_0.12"))
    # Arm detections at any radius; pitch separates tight and open spirals.
    allarms = [h for h, (n, _) in enumerate(conditions)
               if n.startswith(f"pitch_{j}_") and "_0.12_True" in n and "_0.0_" not in n]
    rule(5, 1.2, allarms); rule(6, 1.5, allarms); rule(7, 1.5, allarms); rule(0, .4, allarms)
    for c in (2, 3, 4, 8, 9): rule(c, -1.0, allarms)
    tight = [h for h in allarms if any(f"_{p}_" in conditions[h][0] for p in (4.5, 7.0))]
    loose = [h for h in allarms if any(f"_{p}_" in conditions[h][0] for p in (1.2, 2.2))]
    bars = [h for h, (n, _) in enumerate(conditions)
            if n.startswith(f"pitch_{j}_") and "_0.0_" in n and "_0.12_True" in n]
    rule(6, 1.4, tight); rule(6, -.7, loose)
    rule(7, 1.4, loose); rule(7, -.7, tight)
    rule(5, .8, bars)
for f in range(3):
    rule(0, .5, select(f"texture_{f}_0.15_True"))
    rule(1, .5, select(f"texture_{f}_0.15_True"))
    for c in (2, 3, 4): rule(c, -.7, select(f"texture_{f}_0.15_True"))
    rule(8, .3, select(f"texture_{f}_0.15_True"))
for z in ("inner", "total"):
    rule(7, .4, select(f"red_{z}_0.1_False")); rule(8, .6, select(f"red_{z}_0.1_False"))
    rule(2, .4, select(f"red_{z}_0.1_True")); rule(3, .3, select(f"red_{z}_0.1_True"))
    rule(4, .4, select(f"red_{z}_0.1_True")); rule(9, .8, select(f"red_{z}_0.1_True"))
for lev in (.08, .18):
    rule(9, .5, select(f"conc_1_{lev}_True")); rule(2, .3, select(f"conc_1_{lev}_True"))
    rule(8, -.4, select(f"conc_1_{lev}_True"))

# ----------------------------- network -----------------------------
layers = [
    {"name":"down", "op":"avgpool", "inputs":["image"], "size":[2,2], "stride":[2,2]},
    {"name":"standard", "op":"norm", "inputs":["down"]},
    {"name":"match", "op":"dense", "inputs":["standard"], "out":1280},
    {"name":"by_class", "op":"reshape", "inputs":["match"], "shape":[10,16,8]},
    {"name":"best", "op":"maxpool", "inputs":["by_class"], "size":[16,8], "stride":[16,8]},
    {"name":"flat", "op":"reshape", "inputs":["best"], "shape":[10]},
    {"name":"template_scores", "op":"dense", "inputs":["flat"], "out":10},
    {"name":"lum", "op":"conv", "inputs":["standard"], "out":4, "kernel":[1,1], "stride":1, "padding":0},
    {"name":"p", "op":"relu", "inputs":["lum"]},
    {"name":"hp_signed", "op":"conv", "inputs":["standard"], "out":6, "kernel":[9,9], "stride":1, "padding":4},
    {"name":"hp", "op":"relu", "inputs":["hp_signed"]},
    {"name":"feat_p", "op":"dense", "inputs":["p"], "out":F},
    {"name":"feat_hp", "op":"dense", "inputs":["hp"], "out":F},
    {"name":"feat_raw", "op":"dense", "inputs":["down"], "out":F},
    {"name":"signed", "op":"add", "inputs":["feat_p","feat_hp","feat_raw"]},
    {"name":"abs_in", "op":"dense", "inputs":["signed"], "out":2*D},
    {"name":"abs_relu", "op":"relu", "inputs":["abs_in"]},
    {"name":"raw_aug", "op":"dense", "inputs":["signed"], "out":F+D},
    {"name":"abs_aug", "op":"dense", "inputs":["abs_relu"], "out":F+D},
    {"name":"aug", "op":"add", "inputs":["raw_aug","abs_aug"]},
    {"name":"clip", "op":"dense", "inputs":["aug"], "out":2*H},
    {"name":"clip_relu", "op":"relu", "inputs":["clip"]},
    {"name":"indicators", "op":"dense", "inputs":["clip_relu"], "out":H},
    {"name":"rule_scores", "op":"dense", "inputs":["indicators"], "out":10},
    {"name":"scores", "op":"add", "inputs":["template_scores","rule_scores"]},
]
weights = {"match.W":TW, "match.b":np.zeros(1280,np.float32)}
weights["template_scores.W"] = np.eye(10,dtype=np.float32)
weights["template_scores.b"] = np.array([.05,.04,0,0,0,.04,.05,.05,.01,.01],np.float32)
weights["lum.W"] = np.repeat((np.ones((1,3,1,1))/3).astype(np.float32),4,axis=0)
weights["lum.b"] = -np.array([0,.5,1.5,3.0],np.float32)
hsw = np.zeros((6,3,9,9),np.float32)
for f,k in enumerate(kernels):
    hsw[2*f] = k[None]/3; hsw[2*f+1] = -k[None]/3
weights["hp_signed.W"] = hsw; weights["hp_signed.b"] = np.zeros(6,np.float32)
for source, shape in (("p",(4,N,N)),("hp",(6,N,N)),("raw",(3,N,N))):
    mat = np.zeros((F,int(np.prod(shape))),np.float32)
    for i,(src,ch,w) in enumerate(specs):
        if src == source:
            tmp = np.zeros(shape); tmp[ch] = w
            mat[i] = tmp.ravel()
    weights[f"feat_{source}.W"] = mat
    weights[f"feat_{source}.b"] = np.zeros(F,np.float32)
AW = np.zeros((2*D,F),np.float32)
for d,(_,terms) in enumerate(derived):
    for i,w in terms: AW[2*d,i] = w; AW[2*d+1,i] = -w
weights["abs_in.W"] = AW; weights["abs_in.b"] = np.zeros(2*D,np.float32)
RU = np.zeros((F+D,F),np.float32); RU[:F,:F] = np.eye(F)
AU = np.zeros((F+D,2*D),np.float32)
for d in range(D): AU[F+d,2*d] = AU[F+d,2*d+1] = 1
weights["raw_aug.W"] = RU; weights["raw_aug.b"] = np.zeros(F+D,np.float32)
weights["abs_aug.W"] = AU; weights["abs_aug.b"] = np.zeros(F+D,np.float32)
CW = np.zeros((2*H,F+D),np.float32); CB = np.zeros(2*H,np.float32)
K = 30.0
for h,(_,terms) in enumerate(conditions):
    for i,w in terms: CW[2*h,i] = K*w; CW[2*h+1,i] = K*w
    CB[2*h+1] = -1
weights["clip.W"] = CW; weights["clip.b"] = CB
IW = np.zeros((H,2*H),np.float32)
for h in range(H): IW[h,2*h] = 1; IW[h,2*h+1] = -1
weights["indicators.W"] = IW; weights["indicators.b"] = np.zeros(H,np.float32)
weights["rule_scores.W"] = (.5*RW).astype(np.float32)
weights["rule_scores.b"] = np.array([.125,-.875,.85,.8,.675,.6,.675,-.125,.2,.25],np.float32)

out = Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
(out/"network.json").write_text(json.dumps({"layers":layers},separators=(",",":")))
np.savez(out/"weights.npz",**weights)
