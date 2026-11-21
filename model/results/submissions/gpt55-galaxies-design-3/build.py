import json, sys, math
from pathlib import Path
import numpy as np

N = 32
ORIENT = 24
HALF = ORIENT // 2
TYPES = 12

def grid():
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    c = (N - 1) / 2.0
    return xx - c, yy - c
X, Y = grid()
R = np.sqrt(X*X + Y*Y) + 1e-6
TH = np.arctan2(Y, X)

# colour vectors are for RGB input.  They are chosen from astronomical colours:
# old stellar populations are yellow/red; star-forming arms and tidal debris are blue.
LUM = np.array([0.34, 0.33, 0.33], np.float32)
RED = np.array([0.55, 0.32, 0.13], np.float32)
BLUE = np.array([0.12, 0.34, 0.54], np.float32)
WHITE = np.array([0.34, 0.34, 0.32], np.float32)


def wrap_pi(a):
    return (a + math.pi/2) % math.pi - math.pi/2

def ell(a, q, phi, powr=1.0):
    cp, sp = math.cos(phi), math.sin(phi)
    u = X*cp + Y*sp
    v = -X*sp + Y*cp
    rr = (u/a)**2 + (v/(a*q))**2
    return np.exp(-0.5 * (rr**powr)).astype(np.float32)

def gauss_at(x0, y0, s, q=1.0, phi=0.0):
    cp, sp = math.cos(phi), math.sin(phi)
    x = X - x0; y = Y - y0
    u = x*cp + y*sp
    v = -x*sp + y*cp
    return np.exp(-0.5*((u/s)**2 + (v/(s*q))**2)).astype(np.float32)

def spiral(rad, pitch_deg, phi, hand, width=0.23, inner=1.5):
    pitch = math.radians(pitch_deg)
    phase = TH - hand*np.log(R/inner)/math.tan(pitch) - phi
    d = ((phase + math.pi/2) % math.pi) - math.pi/2
    arms = np.exp(-0.5*(d/width)**2)
    env = np.exp(-0.5*((R - 0.58*rad)/(0.32*rad))**2)
    hole = 1.0 - np.exp(-0.5*(R/(0.22*rad+0.4))**2)
    return (arms * env * hole).astype(np.float32)

def crescent(rad, phi, hand, width=0.32):
    # A single asymmetric arc/tail for disturbed systems.
    phase = TH - hand*0.85*np.log(R/2.0) - phi
    d = ((phase + math.pi) % (2*math.pi)) - math.pi
    arc = np.exp(-0.5*(d/width)**2)
    env = np.exp(-0.5*((R - 0.65*rad)/(0.30*rad))**2)
    return (arc * env).astype(np.float32)

def norm_mask(m):
    m = m.astype(np.float32)
    m = m - m.mean()
    n = float(np.sqrt((m*m).sum()) + 1e-6)
    return m / n

def row(mask, colour):
    m = norm_mask(mask)
    w = (colour[:, None, None] * m[None, :, :]).astype(np.float32)
    return w.reshape(-1)

def tmpl(cls, typ, oi):
    phi = (oi % HALF) * math.pi / HALF
    hand = 1 if oi < HALF else -1
    # repeated size sequence spanning small and large galaxies after 4x pooling
    sizes = [2.5, 3.2, 4.0, 5.0, 6.2, 7.6, 9.2, 11.0, 3.6, 5.6, 8.4, 12.5]
    s = sizes[typ]
    if cls == 2:       # round smooth: circular, centrally concentrated yellow light
        q = [1.0, .95, .9, 1.0, .95, .9, 1.0, .95, .85, .9, .95, 1.0][typ]
        m = 1.2*ell(s, q, phi) + 0.45*ell(max(1.0, s*0.45), 1.0, phi)
        return row(m, RED)
    if cls == 3:       # in-between smooth: moderate ellipticity
        q = [.62, .68, .55, .72, .60, .70, .58, .66, .50, .75, .55, .70][typ]
        m = 1.1*ell(s, q, phi) + 0.35*ell(max(1.0, s*.45), min(1.0, q+.15), phi)
        return row(m, RED)
    if cls == 4:       # cigar smooth: elongated but thicker than edge-on disks
        q = [.32, .28, .35, .25, .32, .38, .28, .34, .24, .30, .36, .27][typ]
        m = 1.2*ell(s, q, phi) + 0.25*ell(max(1.0, s*.42), .55, phi)
        return row(m, RED)
    if cls == 8:       # edge-on disk without bulge: very thin line, blue/white, central bulge suppressed
        q = [.13, .11, .16, .12, .15, .10, .13, .16, .09, .12, .15, .11][typ]
        line = ell(s*1.15, q, phi)
        core = ell(max(1.0, s*.23), .75, phi)
        m = 1.35*line - 0.38*core
        return row(m, BLUE)
    if cls == 9:       # edge-on with bulge: thin disk plus bright red/yellow centre
        q = [.12, .10, .15, .11, .14, .10, .13, .16, .09, .12, .15, .11][typ]
        line = ell(s*1.15, q, phi)
        core = ell(max(1.0, s*.24), .75, phi)
        m = 1.0*line + 0.85*core
        return row(m, RED)
    if cls == 1:       # merging pair: two bright components and a faint bridge
        ds = [2.2, 2.8, 3.4, 4.2, 5.0, 6.0, 7.0, 8.2, 3.0, 4.8, 6.8, 9.0][typ]
        ss = [1.15,1.25,1.45,1.65,1.9,2.2,2.5,2.9,1.1,1.55,2.1,2.8][typ]
        cp, sp = math.cos(phi), math.sin(phi)
        g1 = gauss_at(ds*cp, ds*sp, ss, .8, phi)
        g2 = gauss_at(-0.8*ds*cp, -0.8*ds*sp, ss*1.1, .8, phi)
        bridge = ell(max(ds,1.5), .18, phi)
        m = g1 + g2 + 0.22*bridge - 0.10*ell(max(1.0, ds*.35), 1.0, phi)
        return row(m, WHITE)
    if cls == 0:       # disturbed: lopsided blue arc/tail plus off-centre knots
        rad = [4.0,4.8,5.8,6.8,8.0,9.2,10.5,12.0,5.2,7.2,9.8,13.0][typ]
        cp, sp = math.cos(phi), math.sin(phi)
        tail = crescent(rad, phi, hand, width=[.28,.34,.40,.32,.38,.45,.30,.36,.42,.33,.39,.46][typ])
        knot = gauss_at(0.45*rad*cp, 0.45*rad*sp, max(1.0, .16*rad), .65, phi)
        cen = ell(max(1.0, .25*rad), .75, phi)
        m = 0.45*cen + 1.05*tail + 0.55*knot
        return row(m, BLUE)
    if cls == 5:       # barred spiral: straight red bar embedded in blue two-armed disk
        rad = [4.6,5.4,6.2,7.2,8.4,9.6,11.0,12.5,5.8,7.8,10.0,13.5][typ]
        pitch = [24,28,32,36,26,30,34,38,22,28,34,40][typ]
        arms = spiral(rad, pitch, phi, hand, width=.24)
        bar = ell(max(1.2, .34*rad), .24, phi)
        bulge = ell(max(1.0, .18*rad), .85, phi)
        m = 0.95*arms + 0.85*bar + 0.25*bulge
        return row(m, WHITE)
    if cls == 6:       # unbarred tight spiral: small pitch, concentrated centre
        rad = [4.5,5.2,6.0,7.0,8.2,9.4,10.8,12.0,5.6,7.6,9.8,13.0][typ]
        pitch = [12,14,16,18,20,15,17,19,13,16,18,21][typ]
        arms = spiral(rad, pitch, phi, hand, width=.20)
        bulge = ell(max(1.0, .20*rad), .9, phi)
        disk = ell(max(1.3, .48*rad), .85, phi)
        m = 1.05*arms + 0.42*bulge + 0.16*disk
        return row(m, WHITE)
    if cls == 7:       # unbarred loose spiral: open blue arms, less central dominance
        rad = [5.0,5.8,6.8,7.8,9.0,10.4,12.0,13.5,6.2,8.4,10.8,14.5][typ]
        pitch = [30,34,38,42,46,36,40,48,32,38,44,52][typ]
        arms = spiral(rad, pitch, phi, hand, width=.27)
        bulge = ell(max(1.0, .16*rad), .9, phi)
        disk = ell(max(1.5, .52*rad), .9, phi)
        m = 1.15*arms + 0.16*bulge + 0.22*disk
        return row(m, WHITE)
    raise ValueError(cls)

def make_template_weights():
    k = 10*TYPES*ORIENT
    W = np.zeros((k, 3*N*N), np.float32)
    b = np.zeros(k, np.float32)
    idx = 0
    for c in range(10):
        for t in range(TYPES):
            for o in range(ORIENT):
                W[idx] = tmpl(c, t, o)
                idx += 1
    return W, b

def ring(r0, r1):
    # soft annular mask on the pooled grid
    a = 1/(1+np.exp(-(R-r0)*2.0))
    b = 1/(1+np.exp((R-r1)*2.0))
    m = (a*b).astype(np.float32)
    return m/(m.sum()+1e-6)

def make_feature_weights():
    # Hand-coded colour/concentration cues.  Biases make ReLUs act as gentle thresholds.
    masks = []
    cols = []
    biases = []
    inner = ring(0.0, 3.2); mid = ring(3.0, 7.5); outer = ring(6.5, 13.5)
    full = ring(0.0, 13.5)
    ctr = norm_mask(ell(2.4,1.0,0.0))
    # colour contrasts in central and outer regions
    for m in [inner, mid, outer, full]:
        masks += [m, m, m]
        cols += [np.array([1.0,-0.25,-0.75],np.float32), np.array([-0.75,-0.15,1.0],np.float32), LUM]
        biases += [-0.010, -0.010, -0.018]
    # concentration: positive central excess relative to surrounding disk
    masks += [inner-mid, inner-outer, mid-inner, outer-inner, ctr]
    cols += [LUM, LUM, LUM, LUM, RED]
    biases += [-0.006, -0.006, -0.006, -0.006, 0.0]
    F = len(masks)
    W = np.zeros((F,3*N*N),np.float32)
    for i,(m,c) in enumerate(zip(masks, cols)):
        W[i] = (c[:,None,None]*m[None,:,:]).reshape(-1).astype(np.float32)*45.0
    b = np.array(biases,np.float32)*45.0
    return W,b

def make_edge_weights():
    # Oriented centre-surround ridge filters.  They respond to bright bars, arms, dust lanes,
    # and tidal filaments while mostly ignoring flat sky.  Half use total light and half use
    # blue-excess light, since star-forming spiral arms and disturbed debris are often blue.
    K = 9; c = (K-1)/2
    yy, xx = np.mgrid[0:K,0:K].astype(np.float32)
    xx -= c; yy -= c
    filters = []
    colours = [LUM, np.array([-0.45,-0.05,0.85],np.float32)]
    for col in colours:
        for su, sv in [(3.8,0.85),(5.0,1.25)]:
            for oi in range(8):
                phi = oi*math.pi/8
                cp, sp = math.cos(phi), math.sin(phi)
                u = xx*cp + yy*sp
                v = -xx*sp + yy*cp
                core = np.exp(-0.5*((u/su)**2 + (v/sv)**2))
                side = np.exp(-0.5*((u/su)**2 + (v/(sv*2.4))**2))
                m = core - 0.55*side
                m = m - m.mean()
                m = m/(np.sum(np.abs(m))+1e-6)
                filters.append((col[:,None,None]*m[None,:,:]).astype(np.float32))
    W = np.stack(filters,axis=0).astype(np.float32)
    b = np.zeros(W.shape[0],np.float32)
    return W,b

def make_edge_score(F=32):
    # Dense readout from ReLU ridge maps.  The spatial weights are annular masks on the 64x64
    # conv grid; coefficients are hand-chosen morphology rules, not fitted values.
    yy, xx = np.mgrid[0:64,0:64].astype(np.float32)
    rr = np.sqrt((xx-31.5)**2 + (yy-31.5)**2)
    def soft(r0,r1):
        a = 1/(1+np.exp(-(rr-r0)*0.6)); b = 1/(1+np.exp((rr-r1)*0.6))
        m = (a*b).astype(np.float32)
        return m/(m.sum()+1e-6)
    inner = soft(0,8); mid = soft(7,18); outer = soft(15,30); full = soft(0,30)
    W = np.zeros((10,F,64,64),np.float32); b = np.zeros(10,np.float32)
    # channel groups: 0-15 luminance ridges, 16-31 blue-excess ridges.
    lum_ch = list(range(16)); blue_ch = list(range(16,32))
    def add(cls, chs, mask, coeff):
        for ch in chs:
            W[cls,ch] += (coeff/len(chs))*mask*80.0
    # lots of structured light: spirals, edge-ons, mergers; smooth ellipticals dislike it.
    for cls in [2,3]:
        add(cls, lum_ch, mid, -0.55); add(cls, blue_ch, outer, -0.75)
    add(4, lum_ch, mid, 0.20); add(4, blue_ch, outer, -0.35)
    add(0, blue_ch, outer, 1.10); add(0, lum_ch, outer, 0.45); add(0, blue_ch, mid, 0.45)
    add(1, lum_ch, mid, 0.75); add(1, lum_ch, outer, 0.35); add(1, blue_ch, outer, 0.30)
    add(5, lum_ch, mid, 0.95); add(5, blue_ch, mid, 0.45); add(5, lum_ch, inner, 0.35)
    add(6, lum_ch, mid, 0.75); add(6, blue_ch, mid, 0.60); add(6, lum_ch, inner, 0.25)
    add(7, blue_ch, outer, 1.05); add(7, lum_ch, outer, 0.60); add(7, blue_ch, mid, 0.35)
    add(8, lum_ch, mid, 0.45); add(8, blue_ch, outer, 0.45); add(8, lum_ch, inner, -0.20)
    add(9, lum_ch, mid, 0.55); add(9, lum_ch, inner, 0.45); add(9, blue_ch, outer, -0.15)
    return W.reshape(10,-1).astype(np.float32), b

def make_feature_score(F):
    W = np.zeros((10,F),np.float32); b = np.zeros(10,np.float32)
    # feature indices per ring: red, blue, lum repeated for inner/mid/outer/full, then concentration cues
    red_inner, blue_inner, lum_inner = 0,1,2
    red_mid, blue_mid, lum_mid = 3,4,5
    red_outer, blue_outer, lum_outer = 6,7,8
    red_full, blue_full, lum_full = 9,10,11
    conc1, conc2, disk1, disk2, redcore = 12,13,14,15,16
    # Smooth systems and bulges are red/yellow and centrally concentrated.
    for c in [2,3,4,9]:
        W[c,red_inner] += .24; W[c,red_full] += .12; W[c,conc1] += .12; W[c,redcore] += .14
    # Spirals, edge-on disks without bulges, and tidal features get a modest boost from blue outskirts.
    for c in [0,5,6,7,8]:
        W[c,blue_mid] += .18; W[c,blue_outer] += .28; W[c,disk2] += .07
    W[1,blue_mid] += .10; W[1,blue_outer] += .08; W[1,lum_mid] += .06
    W[2,conc2] += .20; W[2,blue_outer] -= .10
    W[3,conc1] += .10
    W[4,red_inner] += .05; W[4,blue_outer] -= .12
    W[5,red_inner] += .04; W[5,blue_outer] += .08
    W[6,conc1] += .05; W[6,blue_mid] += .08
    W[7,blue_outer] += .14; W[7,conc1] -= .04
    W[8,blue_outer] += .13; W[8,red_inner] -= .13; W[8,conc1] -= .05
    W[9,red_inner] += .15; W[9,conc1] += .13; W[9,blue_outer] -= .04
    # Mild hand priors for visually ambiguous low-information images; not based on measured counts.
    b += np.array([-.04,-.03,.06,.03,-.10,.02,.01,.04,-.01,.02],np.float32)
    return W.astype(np.float32), b.astype(np.float32)

def main(outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    layers = [
        {"name":"p4","op":"avgpool","inputs":["image"],"size":[4,4],"stride":[4,4]},
        {"name":"n4","op":"norm","inputs":["p4"]},
        {"name":"tmpl","op":"dense","inputs":["n4"],"out":10*TYPES*ORIENT},
        {"name":"tmpl_r","op":"reshape","inputs":["tmpl"],"shape":[10,TYPES,ORIENT]},
        {"name":"tmpl_m","op":"maxpool","inputs":["tmpl_r"],"size":[TYPES,ORIENT],"stride":[TYPES,ORIENT]},
        {"name":"tmpl_v","op":"reshape","inputs":["tmpl_m"],"shape":[10]},
        {"name":"tmpl_s","op":"dense","inputs":["tmpl_v"],"out":10},
        {"name":"feat","op":"dense","inputs":["p4"],"out":17},
        {"name":"feat_relu","op":"relu","inputs":["feat"]},
        {"name":"feat_s","op":"dense","inputs":["feat_relu"],"out":10},
        {"name":"econv","op":"conv","inputs":["image"],"out":32,"kernel":[9,9],"stride":2,"padding":4},
        {"name":"erelu","op":"relu","inputs":["econv"]},
        {"name":"edge_s","op":"dense","inputs":["erelu"],"out":10},
        {"name":"scores","op":"add","inputs":["tmpl_s","feat_s","edge_s"]},
    ]
    weights = {}
    weights["tmpl.W"], weights["tmpl.b"] = make_template_weights()
    # Hand-written readout from the ten max-pooled template families.  It uses each family's
    # own response but also subtracts competing shapes: e.g. a round smooth score is reduced by
    # elongated/edge-on/spiral evidence, while spiral scores are reduced by smooth evidence.
    A = np.zeros((10,10), dtype=np.float32)
    A[0,[0,1,7,2,3,9]] = [1.65,.20,.18,-.38,-.20,-.18]
    A[1,[1,0,2,3,8,9]] = [1.55,.18,-.35,-.18,-.10,-.10]
    A[2,[2,3,4,5,6,7,8,9]] = [1.05,-.25,-.35,-.18,-.18,-.18,-.30,-.20]
    A[3,[3,2,4,5,6,7,8,9]] = [1.05,-.12,-.30,-.13,-.13,-.13,-.25,-.12]
    A[4,[4,8,9,5,6,7,2,3]] = [.72,-.38,-.30,-.20,-.20,-.20,-.08,-.06]
    A[5,[5,6,7,2,3,8,9]] = [2.65,.22,.18,-.25,-.10,-.24,-.10]
    A[6,[6,5,7,2,3,8,9]] = [2.75,.20,.16,-.24,-.10,-.20,-.08]
    A[7,[7,5,6,0,2,3,8,9]] = [2.85,.16,.16,.10,-.23,-.08,-.18,-.08]
    A[8,[8,9,4,2,3,5,6,7]] = [1.75,-.28,-.25,-.12,-.08,-.10,-.10,-.10]
    A[9,[9,8,4,2,3,5,6,7]] = [1.28,-.18,-.18,-.08,.08,-.08,-.08,-.08]
    weights["tmpl_s.W"] = A
    weights["tmpl_s.b"] = np.array([-.02,-.02,.03,.02,-.10,.10,.10,.12,.00,.03],np.float32)
    weights["feat.W"], weights["feat.b"] = make_feature_weights()
    weights["feat_s.W"], weights["feat_s.b"] = make_feature_score(17)
    weights["econv.W"], weights["econv.b"] = make_edge_weights()
    weights["edge_s.W"], weights["edge_s.b"] = make_edge_score(32)
    (out/"network.json").write_text(json.dumps({"layers":layers}, separators=(",",":")))
    np.savez(out/"weights.npz", **weights)
    (out/"method.md").write_text(
        "The network uses a 4x averaged view of each RGB image.  One branch normalizes each channel and applies a formula-generated bank of centred galaxy templates: round and elliptical Sersic-like blobs, thin edge-on disks with and without a bulge, two-component mergers, asymmetric tidal arcs, barred spirals, and logarithmic spiral arms at fixed pitches.  Responses are max-pooled over scale/orientation/chirality within each class and combined by a hand-written morphology rule table.  A second branch adds hand-coded colour and concentration cues (red central light, blue outer arms, and central excess).  A third branch uses fixed oriented centre-surround ridge filters on the original image to reward bars, arms, dust lanes, tidal filaments, and thin edge-on disks.  All weights are produced analytically in build.py from Gaussian, elliptical, annular, ridge, and logarithmic-spiral formulae; no training images or fitted statistics are used as weights.\n")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "/app/out")
