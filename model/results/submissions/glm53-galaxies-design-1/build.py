# Builds a hand-designed galaxy classifier: network.json + weights.npz.
# Every number comes from formulas / galaxy knowledge. No data is read.
# Usage: python3 build.py OUTPUT_DIR
import sys, os, json
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else '/app/out'
os.makedirs(OUT, exist_ok=True)

layers = []
W = {}

def conv(name, inp, out, k, stride, pad):
    layers.append(dict(name=name, op='conv', inputs=[inp], out=out,
                       kernel=[k, k], stride=stride, padding=pad))

def dense(name, inp, out):
    layers.append(dict(name=name, op='dense', inputs=[inp], out=out))

def pool(name, inp, op, h, w, sh, sw):
    layers.append(dict(name=name, op=op, inputs=[inp], size=[h, w], stride=[sh, sw]))

def reshape(name, inp, shape):
    layers.append(dict(name=name, op='reshape', inputs=[inp], shape=shape))

def plain(name, op, inps):
    layers.append(dict(name=name, op=op, inputs=list(inps)))

NORI = 11

# ---------------- input processing ----------------
# lum: L=(r+g+b)/3, colour axes (b-r) and (g-(r+b)/2);  L: single-channel light
Wl = np.zeros((3, 3, 1, 1))
Wl[0, 0, 0, 0] = Wl[0, 1, 0, 0] = Wl[0, 2, 0, 0] = 1.0 / 3.0
Wl[1, 2, 0, 0] = 1.0; Wl[1, 0, 0, 0] = -1.0
Wl[2, 1, 0, 0] = 1.0; Wl[2, 0, 0, 0] = -0.5; Wl[2, 2, 0, 0] = -0.5
conv('lum', 'image', 3, 1, 1, 0)
W['lum.W'] = Wl; W['lum.b'] = np.zeros(3)

WL = np.zeros((1, 3, 1, 1)); WL[0, 0] = 1 / 3.; WL[0, 1] = 1 / 3.; WL[0, 2] = 1 / 3.
conv('L', 'image', 1, 1, 1, 0)
W['L.W'] = WL; W['L.b'] = np.zeros(1)

pool('Ld', 'L', 'avgpool', 2, 2, 2, 2)                 # (1,64,64) half-res light
pool('p8', 'lum', 'avgpool', 8, 8, 8, 8)               # (3,16,16)
pool('p4', 'lum', 'avgpool', 4, 4, 4, 4)               # (3,32,32)

# ---------------- texture bank: even Gabors, per-channel norm (brightness-free)
# fine scales sigma 1.5,3 at full res (k=13); coarse sigma 6,12 at half res
# (k=25 on the 64x64 light) -- equal filters, 9x less compute.
def gab_bank(inp, sigs, k, outsize, tag):
    pad = (k - 1) // 2
    Wg = np.zeros((11 * len(sigs), 1, k, k))
    c = (k - 1) / 2.0
    yy, xx = np.mgrid[0:k, 0:k]
    yc, xc = yy - c, xx - c
    for s, sig in enumerate(sigs):
        lam = 3.0 * sig
        for o in range(NORI):
            th = np.pi * o / NORI
            xr = xc * np.cos(th) + yc * np.sin(th)
            yr = -xc * np.sin(th) + yc * np.cos(th)
            g = np.exp(-(xr**2 + yr**2) / (2 * sig**2)) * np.cos(2 * np.pi * xr / lam)
            Wg[11 * s + o, 0] = g / np.sqrt((g**2).sum())
    conv('gab' + tag, inp, 11 * len(sigs), k, 2, pad)
    W['gab' + tag + '.W'] = Wg; W['gab' + tag + '.b'] = np.zeros(11 * len(sigs))
    plain('gn' + tag, 'norm', ['gab' + tag])
    plain('gr' + tag, 'relu', ['gn' + tag])
    pool('gE' + tag, 'gr' + tag, 'avgpool', outsize, outsize, outsize, outsize)
    reshape('gEr' + tag, 'gE' + tag, [len(sigs), 11, 1])
    pool('gEa' + tag, 'gEr' + tag, 'avgpool', 11, 1, 11, 1)
    # centre 64x64 original-pixel region: 3x3 blocks, keep the middle one
    cb = outsize // 2
    cs = outsize // 4
    pool('gE3' + tag, 'gr' + tag, 'avgpool', cb, cb, cs, cs)      # (22,3,3)
    Wc = np.zeros((22, 22, 3, 3))
    for a in range(22): Wc[a, a, 1, 1] = 1.0
    conv('gEc' + tag, 'gE3' + tag, 22, 3, 3, 0)                   # centre cell
    W['gEc' + tag + '.W'] = Wc; W['gEc' + tag + '.b'] = np.zeros(22)
    reshape('gEcr' + tag, 'gEc' + tag, [len(sigs), 11, 1])
    pool('gEca' + tag, 'gEcr' + tag, 'avgpool', 11, 1, 11, 1)     # (2,1,1)

gab_bank('L', [1.5, 3.0], 13, 64, 'f')      # (22,64,64) -> gEf (22,1,1), gEaf (2,1,1)
gab_bank('Ld', [3.0, 6.0], 25, 32, 'c')     # (22,32,32) -> gEc (22,1,1), gEac (2,1,1)

# ---------------- elongation bank: elongated Gaussians on raw light ----------------
def ax_bank(inp, sl, ss, k, gsize, tag):
    pad = (k - 1) // 2
    Wa = np.zeros((11, 1, k, k))
    c = (k - 1) / 2.0
    yy, xx = np.mgrid[0:k, 0:k]
    yc, xc = yy - c, xx - c
    for o in range(NORI):
        th = np.pi * o / NORI
        xr = xc * np.cos(th) + yc * np.sin(th)
        yr = -xc * np.sin(th) + yc * np.cos(th)
        g = np.exp(-(xr**2) / (2 * sl**2) - (yr**2) / (2 * ss**2))
        g = g - g.mean()                     # remove DC: flat sky -> 0 response
        Wa[o, 0] = g / np.sqrt((g**2).sum())
    conv('ax' + tag, inp, 11, k, 2, pad)
    W['ax' + tag + '.W'] = Wa; W['ax' + tag + '.b'] = np.zeros(11)
    plain('ar' + tag, 'relu', ['ax' + tag])                      # (11,gsize,gsize)
    reshape('ar2' + tag, 'ar' + tag, [1, 11, gsize * gsize])
    pool('axmax' + tag, 'ar2' + tag, 'maxpool', 11, 1, 11, 1)    # (1,1,gsize^2)
    pool('axavg' + tag, 'ar2' + tag, 'avgpool', 11, 1, 11, 1)
    reshape('axm' + tag, 'axmax' + tag, [1, gsize, gsize])
    reshape('axv' + tag, 'axavg' + tag, [1, gsize, gsize])
    bs = gsize // 4
    pool('axsp' + tag, 'axm' + tag, 'maxpool', bs, bs, bs, bs)   # (1,4,4) max map
    pool('axs2' + tag, 'axv' + tag, 'maxpool', bs, bs, bs, bs)   # (1,4,4) mean map
    pool('axgm' + tag, 'axmax' + tag, 'avgpool', 1, gsize * gsize, 1, gsize * gsize)
    pool('axga' + tag, 'axavg' + tag, 'avgpool', 1, gsize * gsize, 1, gsize * gsize)

ax_bank('L', 4.5, 1.5, 19, 64, 'f')    # fine: long sigma 4.5px
ax_bank('Ld', 6.0, 2.0, 25, 32, 'c')   # coarse: long sigma 12px (full-res equiv)

# ---------------- centred profile masks on the 8px grid ----------------
def gridmasks():
    y, x = np.mgrid[0:16, 0:16]
    r = y - 7.5; q = x - 7.5
    rho = np.sqrt(r**2 + q**2)
    m = []
    m.append((rho <= 1.5) * 1.0)                                  # 0 core 16px
    m.append(((rho > 1.5) & (rho <= 3.5)) * 1.0)                  # 1 ring 12-28px
    m.append(((rho > 3.5) & (rho <= 6.0)) * 1.0)                  # 2 ring 28-48px
    m.append((rho > 6.0) * 1.0)                                   # 3 outer
    m.append(((np.abs(r) <= 1.5) & (rho <= 6.0)) * 1.0)           # 4 H strip
    m.append(((np.abs(q) <= 1.5) & (rho <= 6.0)) * 1.0)           # 5 V strip
    m.append(((np.abs(r - q) / np.sqrt(2) <= 1.5) & (rho <= 6.0)) * 1.0)  # 6 diag1
    m.append(((np.abs(r + q) / np.sqrt(2) <= 1.5) & (rho <= 6.0)) * 1.0)  # 7 diag2
    m.append(((r < -2.0) & (rho <= 8.0)) * 1.0)                   # 8 upper
    m.append(((r > 2.0) & (rho <= 8.0)) * 1.0)                    # 9 lower
    m.append(((q < -2.0) & (rho <= 8.0)) * 1.0)                   # 10 left
    m.append(((q > 2.0) & (rho <= 8.0)) * 1.0)                    # 11 right
    # 8 azimuthal sectors at ring radius 12-48px (spiral arms / lopsidedness)
    ang = np.arctan2(r, q)
    for k in range(8):
        a0 = -np.pi + k * np.pi / 4
        d = np.angle(np.exp(1j * (ang - a0)))
        m.append(((np.abs(d) <= np.pi / 8) & (rho > 1.5) & (rho <= 6.0)) * 1.0)
    return m

MSK = gridmasks()
Wp = np.zeros((32, 3, 16, 16))
for j, m in enumerate(MSK[:12]):
    m = m / m.sum()
    Wp[j, 0] = m          # luminance
    Wp[12 + j, 1] = m     # colour (b-r)
for j, m in enumerate(MSK[12:]):          # sectors 24..31, luminance only
    Wp[24 + j, 0] = m / m.sum()
conv('prof', 'p8', 32, 16, 16, 0)
W['prof.W'] = Wp; W['prof.b'] = np.zeros(32)

# ---------------- light moments on the 8px grid (window sigma 40px) ----------------
y, x = np.mgrid[0:16, 0:16]
r = (y - 7.5) * 8.0; q = (x - 7.5) * 8.0          # pixel offsets from centre
rho2 = r**2 + q**2
w = np.exp(-rho2 / (2 * (40.0**2)))
Wm = np.zeros((6, 3, 16, 16))
Wm[0, 0] = w * q**2          # m1 = <L x^2>
Wm[1, 0] = w * r**2          # m2 = <L y^2>
Wm[2, 0] = w * r * q         # m3 = <L xy>
Wm[3, 0] = w                 # m4 = <L>   (total light)
Wm[4, 0] = w * rho2          # m5 = <L r^2>
Wm[5, 0] = w * rho2**2       # m6 = <L r^4>
conv('mom', 'p8', 6, 16, 16, 0)
W['mom.W'] = Wm; W['mom.b'] = np.zeros(6)
# background the moments would see on flat sky of level 1 (build-time constant,
# used to sky-subtract every moment row)
KM = np.array([(w * q**2).sum(), (w * r**2).sum(), (w * r * q).sum(),
               w.sum(), (w * rho2).sum(), (w * rho2**2).sum()])
# sky estimator: mean of the four 2x2-cell corner blocks of the 8px grid
cm = np.zeros((16, 16))
for a, b in [(0, 0), (0, 14), (14, 0), (14, 14)]:
    cm[a:a + 2, b:b + 2] = 1.0
Ws = np.zeros((1, 3, 16, 16)); Ws[0, 0] = cm / cm.sum()
conv('skyc', 'p8', 1, 16, 16, 0)
W['skyc.W'] = Ws; W['skyc.b'] = np.zeros(1)

# ---------------- fine masks on the 4px grid ----------------
y4, x4 = np.mgrid[0:32, 0:32]
r4 = y4 - 15.5; q4 = x4 - 15.5
rho4 = np.sqrt(r4**2 + q4**2)
fm = []
fm.append(((r4 < 0) & (q4 < 0) & (rho4 <= 8)) * 1.0)    # 0 Q1
fm.append(((r4 < 0) & (q4 >= 0) & (rho4 <= 8)) * 1.0)   # 1 Q2
fm.append(((r4 >= 0) & (q4 < 0) & (rho4 <= 8)) * 1.0)   # 2 Q3
fm.append(((r4 >= 0) & (q4 >= 0) & (rho4 <= 8)) * 1.0)  # 3 Q4
fm.append((rho4 <= 2) * 1.0)                            # 4 centre 16px
fm.append(((r4 < 0) & (q4 >= 0) & (rho4 > 2) & (rho4 <= 6)) * 1.0)  # 5 NE arc
fm.append(((r4 < 0) & (q4 < 0) & (rho4 > 2) & (rho4 <= 6)) * 1.0)   # 6 NW arc
fm.append(((r4 >= 0) & (q4 >= 0) & (rho4 > 2) & (rho4 <= 6)) * 1.0) # 7 SE arc
fm.append(((r4 >= 0) & (q4 < 0) & (rho4 > 2) & (rho4 <= 6)) * 1.0)  # 8 SW arc
fm.append(((rho4 > 6) & (rho4 <= 15)) * 1.0)            # 9 outer ring 24-60px
Wf = np.zeros((10, 3, 32, 32))
for j, m in enumerate(fm):
    Wf[j, 0] = m / m.sum()
conv('fine', 'p4', 10, 32, 32, 0)
W['fine.W'] = Wf; W['fine.b'] = np.zeros(10)

pool('p4mx', 'p4', 'maxpool', 32, 32, 32, 32)          # (3,1,1) brightest 4px block
# peak-to-trough contrast of the 8 azimuthal sectors (12-48px ring):
# spiral arms and lopsidedness raise it; smooth profiles and noise do not.
Wsect = np.zeros((8, 32, 1, 1)); Wsectn = np.zeros((8, 32, 1, 1))
for j in range(8):
    Wsect[j, 24 + j, 0, 0] = 1.0; Wsectn[j, 24 + j, 0, 0] = -1.0
conv('sect', 'prof', 8, 1, 1, 0)
W['sect.W'] = Wsect; W['sect.b'] = np.zeros(8)
conv('sectn', 'prof', 8, 1, 1, 0)
W['sectn.W'] = Wsectn; W['sectn.b'] = np.zeros(8)
reshape('sectr', 'sect', [1, 8, 1]); reshape('sectnr', 'sectn', [1, 8, 1])
pool('smax', 'sectr', 'maxpool', 8, 1, 8, 1)            # (1,1,1) brightest sector
pool('snmax', 'sectnr', 'maxpool', 8, 1, 8, 1)          # (1,1,1) -faintest sector
plain('rough8', 'add', ['smax', 'snmax'])               # max - min sector mean

# ---------------- hidden units ----------------
# Descriptor groups (all combined by the add BEFORE relu, so one unit can be an
# AND of conditions from different groups):
#  Tf (22), Tc (22): mean positive Gabor response per (scale,orientation),
#                    fine scales (sigma 1.5,3) / coarse (6,12), post-norm
#  Taf (2), Tac (2): per-scale mean over orientations (fine / coarse)
#  M (6): light moments, sky-corrected via S;  S (1): corner sky level
#  P (32): centred mask means on the 8px grid (12 L + 12 colour + 8 sectors)
#  F (10): centred masks on the 4px grid;  X (3): brightest 4x4 block
#  Af/Af2 (16): fine-scale block map, orientation-max / orientation-mean
#  Ac/Ac2 (16): coarse-scale ditto;  Gmf,Gaf,Gmc,Gac (1): frame means of the
#  orientation-max / orientation-mean responses, fine / coarse
# Moment rows are sky-subtracted using KM and scaled 2e-3 (elongation) /
# 5e-3 (concentration); P/F rows are scaled x3.
NU = 80
PF = 3.0
SC_E = 2e-3
SC_C = 5e-3
units = [dict(Tf=None, Tc=None, Tcf=None, Tcc=None, Taf=None, Tac=None, M=None,
              P=None, F=None, X=None, Af=None, Af2=None, Ac=None, Ac2=None,
              Gmf=None, Gaf=None, Gmc=None, Gac=None, R=None, S=None, b=0.0) for _ in range(NU)]
names = {}
i = 0

def blank():
    return dict(Tf=None, Tc=None, Tcf=None, Tcc=None, Taf=None, Tac=None, M=None,
                P=None, F=None, X=None, Af=None, Af2=None, Ac=None, Ac2=None,
                Gmf=None, Gaf=None, Gmc=None, Gac=None, R=None, S=None, b=0.0)

def mrow(a, scale):
    # moment row a (6,) -> sky-corrected (M row, S row) pair
    a = np.array(a, float)
    bgc = float((a * KM).sum())          # sky level 1 contributes bgc
    return a * scale, np.array([-bgc * scale])

def Trow(s, coef):
    v = np.zeros(22)
    for o in range(NORI):
        v[11 * s + o] = coef(np.pi * o / NORI)
    return v

# elongation tiers.  Dx/S = (1-q^2)/(1+q^2):  v=0.84 (cigar/edge-on, q<0.25),
# hi=0.60 (q<0.38), lo=0.38 (q<0.60).  Four sign combos make it position-angle
# free; each row pairs with m4 so the SIGN is brightness-independent, and each
# carries an absolute margin because sky noise puts ~100 raw units of scatter
# into the quadrupole of faint galaxies.
for t, tag, mg in [(0.84, 'v', 0.20), (0.60, 'hi', 0.16), (0.38, 'lo', 0.08)]:
    for sgn in (1, -1):
        u = blank()
        u['M'], u['S'] = mrow([sgn - t, -sgn - t, 0, 0, 0, 0], SC_E)
        u['b'] = -mg
        units[i] = u; names[f'e_{tag}_x{"p" if sgn>0 else "n"}'] = i; i += 1
        u = blank()
        u['M'], u['S'] = mrow([-t, -t, sgn * 2, 0, 0, 0], SC_E)
        u['b'] = -mg
        units[i] = u; names[f'e_{tag}_d{"p" if sgn>0 else "n"}'] = i; i += 1

def mu(row, nm, b=0.0, scale=SC_C):
    global i
    u = blank(); u['M'], u['S'] = mrow(row, scale); u['b'] = b
    units[i] = u; names[nm] = i; i += 1

# concentration / extent: <r^2> thresholds in px^2 (paired with m4, sky-corrected)
mu([0, 0, 0, 120, -1, 0], 'compact10')      # r < 11px  (condensed core)
mu([0, 0, 0, 350, -1, 0], 'compact19')      # r < 19px
mu([0, 0, 0, -700, 1, 0], 'ext30')          # r > 26px
mu([0, 0, 0, -1500, 1, 0], 'ext50')         # r > 39px  (large galaxies)

def one(group, vec, nm, b=0.0):
    global i
    u = blank()
    u[group] = np.array(vec, float) * (PF if group in ('P', 'F') else 1.0)
    u['b'] = b
    units[i] = u; names[nm] = i; i += 1
    return i - 1

def Pz(vec):
    v = np.zeros(32); v[:len(vec)] = vec; return v

one('P', Pz([1, -1.9, 0, 0]), 'core_ring', -0.09)      # core brighter than ring
one('P', Pz([0, 1, -1.2, 0]), 'steep', -0.06)          # steep falloff
one('P', Pz([0, 0, 1, -0.6]), 'env', -0.09)            # extended envelope
for a, b_, nm in [(4, 5, 'sh'), (5, 4, 'sv'), (6, 7, 'sd1'), (7, 6, 'sd2')]:
    v = np.zeros(32); v[a] = 1; v[b_] = -1
    one('P', v, nm, -0.06)                             # strip anisotropy
for a, b_, nm in [(8, 9, 'au'), (9, 8, 'al'), (10, 11, 'all'), (11, 10, 'ar')]:
    v = np.zeros(32); v[a] = 1; v[b_] = -1
    one('P', v, nm, -0.045)                            # half-image asymmetry
v = np.zeros(32); v[12] = -3; v[13] = -3
one('P', v, 'redcore', 0.06)                           # red bulge
v = np.zeros(32); v[14] = 3; v[15] = 3
one('P', v, 'blueout', -0.09)                          # blue outskirts (disk)
v = np.zeros(32); v[15] = 3; v[12] = -3
one('P', v, 'colgrad', -0.06)                          # bluer outward

def Fz(vec):
    v = np.zeros(10); v[:len(vec)] = vec; return v

one('F', Fz([0, 0, 0, 0, 1, 0, 0, 0, 0, -2.5]), 'f_core', -0.06)
vq1 = np.zeros(10); vq1[0] = 1; vq1[1] = -1; vq1[2] = -1; vq1[3] = 1
one('F', vq1, 'fq1', -0.06); one('F', -vq1, 'fq1n', -0.06)
vq2 = np.zeros(10); vq2[0] = 1; vq2[1] = 1; vq2[2] = -1; vq2[3] = -1
one('F', vq2, 'fq2', -0.06); one('F', -vq2, 'fq2n', -0.06)
va = np.zeros(10); va[5] = 1; va[6] = -1
one('F', va, 'fan1', -0.045); one('F', -va, 'fan1n', -0.045)
vb = np.zeros(10); vb[7] = 1; vb[8] = -1
one('F', vb, 'fan2', -0.045); one('F', -vb, 'fan2n', -0.045)

j = one('X', [1, 0, 0], 'offblob', -0.06)              # off-centre bright blob
units[j]['F'] = Fz([0, 0, 0, 0, -2.2, 0, 0, 0, 0, 0]) * PF
units[j]['S'] = np.array([5.6])                        # cancel flat sky
# block elongation: contrast the orientation-MAX map against the orientation-
# MEAN map so a bright round blob (max ~ mean) cannot fire these units.
Ab = np.zeros(16); Ab2 = np.zeros(16)
for by in (1, 2):
    for bx in (1, 2):
        Ab[by * 4 + bx] = 1.0; Ab2[by * 4 + bx] = -1.2
for by, bx in [(0, 0), (0, 3), (3, 0), (3, 3)]:
    Ab[by * 4 + bx] = -0.5; Ab2[by * 4 + bx] = 0.3
for by, bx in [(0, 1), (0, 2), (1, 0), (2, 0), (1, 3), (2, 3), (3, 1), (3, 2)]:
    Ab[by * 4 + bx] = -0.4; Ab2[by * 4 + bx] = 0.25
j = one('Af', Ab, 'bar', -0.6); units[j]['Af2'] = Ab2  # fine scale only
Al = np.zeros(16); Al2 = np.zeros(16)
row1 = [4, 5, 6, 7]; col1 = [1, 5, 9, 13]
diag = [0, 10, 15, 3, 12, 6, 9]; offline = [2, 8, 14, 11]
for c in set(row1) | set(col1):
    Al[c] += 0.7; Al2[c] += -0.9
for c in diag:
    Al[c] += 0.5; Al2[c] += -0.7
for c in offline:
    Al[c] += -1.0; Al2[c] += 0.3
j = one('Af', Al, 'diskline', -0.8)                    # both scales
units[j]['Af2'] = Al2
units[j]['Ac'] = Al.copy(); units[j]['Ac2'] = Al2.copy()
# thin bright structure: orientation max >> orientation mean (frame-wide, so
# scaled up to counter the dilution by empty sky positions)
j = one('Gmf', [12], 'thin', -0.08)
units[j]['Gaf'] = np.array([-15.0]); units[j]['Gmc'] = np.array([12.0])
units[j]['Gac'] = np.array([-15.0])
# azimuthal sector contrasts at 12-48px: 2-fold (spiral arms/bars) and 4-fold
for k in range(4):
    for sgn in (1, -1):
        v = np.zeros(32)
        v[24 + k] = sgn; v[24 + (k + 4) % 8] = -sgn
        one('P', v, f'az2{k}{"p" if sgn>0 else "n"}', -0.05)
for k in range(4):
    for sgn in (1, -1):
        v = np.zeros(32)
        v[24 + k] = sgn; v[24 + (k + 2) % 8] = -sgn
        one('P', v, f'az4{k}{"p" if sgn>0 else "n"}', -0.05)
# texture: orientation-Fourier anisotropy, and energy contrasts.  Taf[0] is the
# fine-scale noise floor, cancelled in the arm/coarse rows.
for s in range(2):
    one('Tf', Trow(s, lambda t: np.cos(2 * t)), f'a2c{s}', -0.06)
    one('Tf', Trow(s, lambda t: np.sin(2 * t)), f'a2s{s}', -0.06)
for s in range(2):
    one('Tc', Trow(s, lambda t: np.cos(2 * t)), f'a2c{s+2}', -0.06)
    one('Tc', Trow(s, lambda t: np.sin(2 * t)), f'a2s{s+2}', -0.06)
for s in range(2):
    one('Tf', Trow(s, lambda t: np.cos(4 * t)), f'a4c{s}', -0.05)
    one('Tf', Trow(s, lambda t: np.sin(4 * t)), f'a4s{s}', -0.05)
for s in range(2):
    one('Tc', Trow(s, lambda t: np.cos(4 * t)), f'a4c{s+2}', -0.05)
    one('Tc', Trow(s, lambda t: np.sin(4 * t)), f'a4s{s+2}', -0.05)
one('R', [1], 'rough', -0.03)                          # azimuthal sector contrast
one('Tcf', [1, 0], 'tfin', -0.45)                      # fine texture above noise
one('Tcf', [-0.9, 0], 'tarm', -0.15); units[names['tarm']]['Tcc'] = np.array([1.0, 0])
one('Tcf', [-0.9, 0], 'tbig', -0.15); units[names['tbig']]['Tcc'] = np.array([0, 1.0])
one('Tcf', [-0.75, -0.75], 'tsmooth', 1.90)
units[names['tsmooth']]['Tcc'] = np.array([-0.75, -0.75])
assert i == NU, i

# assemble group dense weights
GROUPS = [('dTf', 'Tf', 'gEf', 22), ('dTc', 'Tc', 'gEc', 22),
          ('dTcf', 'Tcf', 'gEcaf', 2), ('dTcc', 'Tcc', 'gEcac', 2),
          ('dM', 'M', 'mom', 6), ('dP', 'P', 'prof', 32), ('dF', 'F', 'fine', 10),
          ('dX', 'X', 'p4mx', 3),
          ('dAf', 'Af', 'axspf', 16), ('dAf2', 'Af2', 'axs2f', 16),
          ('dAc', 'Ac', 'axspc', 16), ('dAc2', 'Ac2', 'axs2c', 16),
          ('dGmf', 'Gmf', 'axgmf', 1), ('dGaf', 'Gaf', 'axgaf', 1),
          ('dGmc', 'Gmc', 'axgmc', 1), ('dGac', 'Gac', 'axgac', 1),
          ('dR', 'R', 'rough8', 1), ('dS', 'S', 'skyc', 1)]
for lname, key, inp, n in GROUPS:
    Mx = np.zeros((NU, n))
    for j, u in enumerate(units):
        if u[key] is not None:
            Mx[j] = u[key]
    dense(lname, inp, NU)
    W[lname + '.W'] = Mx
    W[lname + '.b'] = np.array([u['b'] for u in units]) if lname == 'dTf' else np.zeros(NU)

plain('hsum', 'add', [g[0] for g in GROUPS])
reshape('hr3', 'hsum', [NU, 1, 1])
plain('hr', 'relu', ['hr3'])
reshape('hv', 'hr', [NU])
dense('out', 'hv', 10)

# ---------------- final class weights ----------------
def Wc(entries):
    v = np.zeros(NU)
    for nm, wgt in entries:
        v[names[nm]] += wgt
    return v

E_V = ['e_v_xp', 'e_v_xn', 'e_v_dp', 'e_v_dn']
E_HI = ['e_hi_xp', 'e_hi_xn', 'e_hi_dp', 'e_hi_dn']
E_LO = ['e_lo_xp', 'e_lo_xn', 'e_lo_dp', 'e_lo_dn']
STRIP = ['sh', 'sv', 'sd1', 'sd2']
ASYM = ['au', 'al', 'all', 'ar']
FASYM = ['fq1', 'fq1n', 'fq2', 'fq2n', 'fan1', 'fan1n', 'fan2', 'fan2n']
AZ2 = [f'az2{k}{s}' for k in range(4) for s in 'pn']
AZ4 = [f'az4{k}{s}' for k in range(4) for s in 'pn']
A2N = [f'a2c{s}' for s in range(4)] + [f'a2s{s}' for s in range(4)]
A4N = [f'a4c{s}' for s in range(4)] + [f'a4s{s}' for s in range(4)]

CLS = [
 ('c0', [(n, 1.6) for n in STRIP] + [(n, 1.8) for n in ASYM] +
    [(n, 1.6) for n in FASYM] + [(n, 0.5) for n in AZ4] +
    [('tfin', 1.2), ('offblob', 1.8), ('ext50', 0.15), ('core_ring', -0.4),
     ('rough', 0.4)] +
    [(n, -0.7) for n in E_HI] + [(n, -0.9) for n in E_V] +
    [('tsmooth', -0.8), ('redcore', -0.8), ('tarm', -0.5)] +
    [(n, -1.0) for n in AZ2]),
 ('c1', [(n, 2.0) for n in ASYM] + [(n, 1.8) for n in FASYM] +
    [('offblob', 2.6), ('tfin', 1.5), ('rough', 0.2)] + [(n, 0.8) for n in AZ4] +
    [('ext30', 0.10)] + [(n, -0.3) for n in E_LO] + [(n, -0.5) for n in E_HI] +
    [(n, -0.8) for n in E_V] + [('tsmooth', -1.0), ('core_ring', -0.8),
    ('redcore', -0.5), ('bar', -0.8), ('diskline', -0.4)] + [(n, -1.0) for n in AZ2]),
 ('c2', [('f_core', 1.1), ('core_ring', 0.7), ('redcore', 0.8), ('steep', 0.8),
     ('compact19', 0.3), ('tsmooth', 0.5), ('ext30', -0.4), ('rough', -1.5)] +
    [(n, -3.0) for n in E_V] + [(n, -2.0) for n in E_HI] + [(n, -1.5) for n in E_LO] +
    [('tfin', -1.2), ('tarm', -1.0), ('tbig', -0.8)] +
    [(n, -1.0) for n in AZ2] + [(n, -0.8) for n in AZ4] +
    [(n, -1.2) for n in ASYM] + [(n, -1.0) for n in FASYM] + [(n, -0.8) for n in A2N] +
    [('bar', -0.8), ('blueout', -1.0), ('colgrad', -0.8), ('diskline', -0.6)]),
 ('c3', [(n, 1.2) for n in E_LO] + [('f_core', 0.5), ('core_ring', 0.6),
     ('compact19', 0.35), ('tsmooth', 0.5), ('steep', 0.4),
     ('redcore', 0.4), ('ext30', -0.3), ('rough', -1.2)] +
    [(n, -0.8) for n in E_HI] + [(n, -1.2) for n in E_V] +
    [('tfin', -0.8), ('tarm', -0.7), ('tbig', -0.5)] + [(n, -0.6) for n in ASYM] +
    [(n, -0.8) for n in AZ2] + [('blueout', -0.6), ('colgrad', -0.4)]),
 ('c4', [(n, 2.0) for n in E_V] + [(n, 1.5) for n in E_HI] + [(n, 0.4) for n in E_LO] +
    [(n, 1.2) for n in STRIP] + [('compact19', 0.25), ('tsmooth', 0.2),
     ('steep', 0.25), ('diskline', 1.0), ('rough', -1.2)] +
    [('tfin', -1.0), ('tarm', -0.8), ('tbig', -0.6)] + [(n, -1.0) for n in AZ2] +
    [('bar', -0.6), ('blueout', -0.4)] + [(n, -0.6) for n in ASYM]),
 ('c5', [('bar', 1.2), ('rough', 2.0), ('tarm', 0.8), ('tbig', 0.4), ('env', 0.6),
     ('ext30', 0.25), ('blueout', 1.0), ('colgrad', 0.8), ('redcore', 0.2)] +
    [(n, 1.5) for n in AZ2] + [(n, 0.8) for n in A2N] + [(n, 0.3) for n in E_LO] +
    [('tsmooth', -1.0), ('diskline', -0.8)] + [(n, -0.4) for n in FASYM] +
    [(n, -0.5) for n in STRIP]),
 ('c6', [('rough', 2.0), ('tarm', 1.0), ('tbig', 0.4), ('tfin', 0.8), ('env', 1.0),
     ('ext30', 0.4), ('blueout', 1.2), ('colgrad', 0.6)] +
    [(n, 1.0) for n in AZ4] + [(n, 1.2) for n in AZ2] + [(n, 0.6) for n in A2N] +
    [('diskline', -0.6)] + [(n, -0.5) for n in E_HI] + [(n, -0.8) for n in E_V] +
    [('tsmooth', -0.8), ('core_ring', -0.4)]),
 ('c7', [('rough', 2.0), ('tarm', 0.6), ('tbig', 1.0), ('env', 1.4), ('ext30', 0.4),
     ('blueout', 1.2), ('colgrad', 0.6)] +
    [(n, 0.6) for n in AZ4] + [('tfin', -0.8), ('tsmooth', -0.6)] +
    [(n, -0.5) for n in E_HI] + [('diskline', -0.4)] + [(n, -0.3) for n in FASYM]),
 ('c8', [(n, 2.0) for n in E_V] + [(n, 1.8) for n in E_HI] + [(n, 0.4) for n in E_LO] +
    [('diskline', 2.0), ('thin', 0.8), ('env', 0.3), ('blueout', 0.6),
     ('tsmooth', 0.4), ('offblob', -0.5), ('tfin', -1.2), ('rough', -0.8)] +
    [(n, 1.2) for n in STRIP] +
    [('tarm', -0.8), ('tbig', -0.6), ('core_ring', -1.0),
     ('redcore', -1.2), ('compact10', -0.1)] +
    [(n, -0.8) for n in AZ2] + [(n, 0.4) for n in A2N]),
 ('c9', [(n, 2.0) for n in E_V] + [(n, 1.0) for n in E_HI] +
    [('diskline', 0.9), ('thin', 0.6), ('core_ring', 0.6), ('redcore', 0.6),
     ('compact10', 0.15), ('tsmooth', 0.2)] + [(n, 0.5) for n in STRIP] +
    [('tfin', -0.8), ('tarm', -0.8), ('tbig', -0.6), ('blueout', -0.8),
     ('rough', -0.8)] +
    [(n, -1.0) for n in AZ2] + [(n, -0.8) for n in AZ4]),
]

Wout = np.zeros((10, NU))
for k, (_, entries) in enumerate(CLS):
    Wout[k] = Wc(entries)
W['out.W'] = Wout
W['out.b'] = np.zeros(10)

json.dump(dict(layers=layers), open(os.path.join(OUT, 'network.json'), 'w'))
np.savez(os.path.join(OUT, 'weights.npz'), **{k: v.astype(np.float32) for k, v in W.items()})
print('layers', len(layers), 'weights', sum(v.size for v in W.values()))
