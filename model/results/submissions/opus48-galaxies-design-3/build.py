"""Build a hand-designed galaxy-morphology classifier: writes network.json and weights.npz.

Every weight comes from a formula or a hand-chosen constant grounded in general knowledge
of galaxy morphology (the standard Concentration / Asymmetry / colour / ellipticity statistics).
No training data, no fitting, no files are read. Run:  python3 build.py OUT_DIR
"""
import sys, json, os
import numpy as np

# ----------------------------------------------------------------------------
# feature standardisation (centre, scale) chosen by hand from general morphology
STD = {
    'el':  (0.40, 0.40),   # harmonic elongation  |C2|+|S2|
    'et':  (2.20, 0.60),   # total oriented energy (centre)
    'ct':  (2.45, 0.40),   # central standardized brightness
    'co':  (2.10, 0.30),   # concentration = ctrZ - ringZ
    'exr': (0.08, 0.08),   # bright area in the ring (edge-on disk extent)
    'cc':  (0.30, 0.40),   # central colour  norm(R-B)  (red positive)
    'aT':  (0.58, 0.10),   # ring oriented energy
    'asy': (2.25, 1.00),   # rotational asymmetry (net sums over mirror-pairs = half the
                           # 36-cell sum used to design the head, so centre/scale are halved)
}

# ----------------------------------------------------------------------------
def oriented_kernels(n=8, L=15, sl=4.0, sw=1.2):
    """n zero-mean elongated Gaussian (even) filters, normalized by L1 norm."""
    h = L // 2
    ay, ax = np.mgrid[-h:h+1, -h:h+1].astype(np.float64)
    ks = []
    for t in range(n):
        th = np.pi * t / n
        u = ax*np.cos(th) + ay*np.sin(th)       # along major axis
        v = -ax*np.sin(th) + ay*np.cos(th)      # perpendicular
        k = np.exp(-(u*u/(2*sl*sl) + v*v/(2*sw*sw)))
        k -= k.mean()
        k /= np.abs(k).sum()
        ks.append(k.astype(np.float32))
    return ks

# ----------------------------------------------------------------------------
def build_weights():
    W = {}
    # --- pre: 1x1 conv 3 -> 2  [L, R-B] ---------------------------------
    pre_W = np.zeros((2, 3, 1, 1), np.float32)
    pre_W[0, :, 0, 0] = [1/3, 1/3, 1/3]   # luminance L
    pre_W[1, :, 0, 0] = [1, 0, -1]        # colour R-B
    W['pre.W'] = pre_W
    W['pre.b'] = np.zeros(2, np.float32)

    # --- feat: 15x15 conv 2 -> 20 ---------------------------------------
    K = 15; C = 2; NF = 20; ctr = K // 2
    feat_W = np.zeros((NF, C, K, K), np.float32)
    feat_b = np.zeros(NF, np.float32)
    ks = oriented_kernels(8, K, 4.0, 1.2)
    for t in range(8):
        feat_W[t,   0] = ks[t]        # oriented +k_t on Zl
        feat_W[t+8, 0] = -ks[t]       # oriented -k_t on Zl
    feat_W[16, 0, ctr, ctr] = 1.0     # Zl  (brightness)
    feat_W[17, 0, ctr, ctr] = 1.0     # Zl, bias -1 -> relu(Zl-1) (extent)
    feat_b[17] = -1.0
    feat_W[18, 1, ctr, ctr] = 1.0     # Zc  (red+)
    feat_W[19, 1, ctr, ctr] = -1.0    # -Zc (blue+)
    W['feat.W'] = feat_W
    W['feat.b'] = feat_b

    # --- sfeat: dense (20*8*8=1280) -> 48 morphological atoms -------------
    NSF = 48
    sfeat_W = np.zeros((NSF, NF*8*8), np.float32)
    def idx(ch, y, x): return ch*64 + y*8 + x
    CC = [(3,3),(3,4),(4,3),(4,4)]                       # centre cells
    RC = [(2,2),(2,3),(2,4),(2,5),(3,2),(3,5),
          (4,2),(4,5),(5,2),(5,3),(5,4),(5,5)]           # ring cells
    th = np.array([np.pi*t/8 for t in range(8)])
    c2 = np.cos(2*th); s2 = np.sin(2*th)
    # rows 0..3: C2, -C2, S2, -S2  (2nd angular harmonic of centre oriented energy)
    for t in range(8):
        for (y,x) in CC:
            w = 0.25
            sfeat_W[0, idx(t,y,x)]   += c2[t]*w;  sfeat_W[0, idx(t+8,y,x)] += c2[t]*w
            sfeat_W[2, idx(t,y,x)]   += s2[t]*w;  sfeat_W[2, idx(t+8,y,x)] += s2[t]*w
    sfeat_W[1] = -sfeat_W[0]
    sfeat_W[3] = -sfeat_W[2]
    # row 4: armtot (ring oriented energy)
    for t in range(8):
        for (y,x) in RC:
            sfeat_W[4, idx(t,y,x)] += 1/12.0; sfeat_W[4, idx(t+8,y,x)] += 1/12.0
    # row 5: etot (centre oriented energy)
    for t in range(8):
        for (y,x) in CC:
            sfeat_W[5, idx(t,y,x)] += 0.25; sfeat_W[5, idx(t+8,y,x)] += 0.25
    # row 6: ctrZ, row 7: ringZ  (channel 16 = relu(Zl))
    for (y,x) in CC: sfeat_W[6, idx(16,y,x)] += 0.25
    for (y,x) in RC: sfeat_W[7, idx(16,y,x)] += 1/12.0
    # row 8: ext_c, row 9: ext_r  (channel 17 = relu(Zl-1))
    for (y,x) in CC: sfeat_W[8, idx(17,y,x)] += 0.25
    for (y,x) in RC: sfeat_W[9, idx(17,y,x)] += 1/12.0
    # row 10: colc_p (centre red+), row 11: colc_m (centre blue+)
    for (y,x) in CC: sfeat_W[10, idx(18,y,x)] += 0.25
    for (y,x) in CC: sfeat_W[11, idx(19,y,x)] += 0.25
    # rows 12..47: asymmetry atoms over central 6x6 cells of channel 16
    cent6 = [(y,x) for y in range(1,7) for x in range(1,7)]
    for i,(y,x) in enumerate(cent6):
        r = 12 + i
        sfeat_W[r, idx(16,y,x)]       += 1.0
        sfeat_W[r, idx(16,7-y,7-x)]   += -1.0
    W['sfeat.W'] = sfeat_W
    W['sfeat.b'] = np.zeros(NSF, np.float32)

    # --- standardized morphological features as linear maps over sfr (48) --
    def vec(const=0.0, terms=None):
        v = np.zeros(NSF, np.float64)
        if terms:
            for i, c in terms: v[i] += c
        return (v, const)
    A = [0,1,2,3]
    cC, sC = STD['ct'];  cEt, sEt = STD['et']; cEl, sEl = STD['el']
    cCo, sCo = STD['co']; cExr, sExr = STD['exr']; cCc, sCc = STD['cc']
    cAt, sAt = STD['aT']; cAs, sAs = STD['asy']
    F = {}
    F['el']  = (np.array([(1.0/sEl) if i in A else 0.0 for i in range(NSF)]), -cEl/sEl)
    F['et']  = vec(-cEt/sEt, [(5, 1.0/sEt)])
    F['ct']  = vec(-cC/sC,   [(6, 1.0/sC)])
    F['co']  = vec(-cCo/sCo, [(6, 1.0/sCo), (7, -1.0/sCo)])
    F['exr'] = vec(-cExr/sExr, [(9, 1.0/sExr)])
    F['cc']  = vec(-cCc/sCc, [(10, 1.0/sCc), (11, -1.0/sCc)])
    F['aT']  = vec(-cAt/sAt, [(4, 1.0/sAt)])
    asy_v = np.zeros(NSF); asy_v[12:48] = 1.0/sAs
    F['asy'] = (asy_v, -cAs/sAs)

    def combine(coefs):
        w = np.zeros(NSF, np.float64); b = 0.0
        for name, c in coefs.items():
            if name == 'const': b += c; continue
            fv, fc = F[name]; w += c*fv; b += c*fc
        return w, b

    # --- lin: linear part of each class score ---------------------------
    LIN = {
        8: {'el':2.0, 'exr':1.1, 'aT':0.3, 'ct':-0.5, 'const':-1.0},
        9: {'el':2.1, 'cc':0.8, 'et':0.4, 'exr':-1.0, 'ct':-0.4, 'const':-1.3},
        4: {'el':1.6, 'exr':-0.8, 'const':-1.9},
        2: {'cc':0.9, 'const':0.15},
        3: {'cc':0.9, 'et':0.25, 'const':0.25},
        6: {'ct':0.5, 'cc':-0.4, 'et':-1.1, 'co':0.3, 'const':0.0},
        5: {'ct':0.6, 'co':0.3, 'et':-0.9, 'const':0.55},
        7: {'cc':-1.1, 'ct':0.4, 'et':0.4, 'const':0.2},
        0: {'cc':-0.8, 'ct':-0.3, 'const':-0.6},
        1: {'asy':1.2, 'const':-0.3},
    }
    lin_W = np.zeros((10, NSF), np.float64); lin_b = np.zeros(10, np.float64)
    for k, cf in LIN.items():
        w, b = combine(cf); lin_W[k] = w; lin_b[k] = b
    W['lin.W'] = lin_W.astype(np.float32)
    W['lin.b'] = lin_b.astype(np.float32)

    # --- hid: 10 gate pre-activations -----------------------------------
    # g0=ct g1=asy-0.6 g2=-el-0.3 g3=el+0.3 g4=el-1.5 g5=asy g6=el g7=-cc g8=cc-0.8 g9=exr
    GATES = [
        {'ct':1.0},
        {'asy':1.0, 'const':-0.6},
        {'el':-1.0, 'const':-0.3},
        {'el':1.0, 'const':0.3},
        {'el':1.0, 'const':-1.5},
        {'asy':1.0},
        {'el':1.0},
        {'cc':-1.0},
        {'cc':1.0, 'const':-0.7},
        {'exr':1.0},
    ]
    hid_W = np.zeros((10, NSF), np.float64); hid_b = np.zeros(10, np.float64)
    for i, cf in enumerate(GATES):
        w, b = combine(cf); hid_W[i] = w; hid_b[i] = b
    W['hid.W'] = hid_W.astype(np.float32)
    W['hid.b'] = hid_b.astype(np.float32)

    # --- gat: gated part of each class score (from relu(hid)) -----------
    GAT = {
        4: {7:-0.6, 0:-0.5},
        2: {2:0.55, 0:-0.4, 1:-0.4},
        3: {3:0.7, 4:-0.7, 0:-0.3, 5:-0.3},
        6: {6:-0.6, 5:-0.7},
        5: {7:-0.6, 8:-0.5, 6:-0.55, 5:-0.6},
        7: {6:-0.5, 5:-0.35},
        0: {6:-0.3, 5:0.4},
        1: {9:0.5, 6:-0.6, 0:-0.3},
    }
    gat_W = np.zeros((10, 10), np.float64); gat_b = np.zeros(10, np.float64)
    for k, cf in GAT.items():
        for g, c in cf.items(): gat_W[k, g] = c
    W['gat.W'] = gat_W.astype(np.float32)
    W['gat.b'] = gat_b.astype(np.float32)
    return W

# ----------------------------------------------------------------------------
def build_network():
    return {"layers": [
        {"name":"pre",  "op":"conv",   "inputs":["image"], "out":2,  "kernel":[1,1],  "stride":1, "padding":0},
        {"name":"zn",   "op":"norm",   "inputs":["pre"]},
        {"name":"feat", "op":"conv",   "inputs":["zn"],    "out":20, "kernel":[15,15],"stride":1, "padding":7},
        {"name":"fr",   "op":"relu",   "inputs":["feat"]},
        {"name":"fp",   "op":"avgpool","inputs":["fr"],    "size":[16,16], "stride":[16,16]},
        {"name":"sfeat","op":"dense",  "inputs":["fp"],    "out":48},
        {"name":"sfr",  "op":"relu",   "inputs":["sfeat"]},
        {"name":"lin",  "op":"dense",  "inputs":["sfr"],   "out":10},
        {"name":"hid",  "op":"dense",  "inputs":["sfr"],   "out":10},
        {"name":"hr",   "op":"relu",   "inputs":["hid"]},
        {"name":"gat",  "op":"dense",  "inputs":["hr"],    "out":10},
        {"name":"out",  "op":"add",    "inputs":["lin","gat"]},
    ]}

# ----------------------------------------------------------------------------
def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    net = build_network()
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, indent=1)
    W = build_weights()
    np.savez(os.path.join(out_dir, "weights.npz"), **W)
    print("wrote network.json and weights.npz to", out_dir)

if __name__ == "__main__":
    main()
