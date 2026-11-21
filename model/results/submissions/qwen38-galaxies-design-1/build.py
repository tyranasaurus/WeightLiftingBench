"""Hand-designed galaxy morphology classifier (no training, weights from formulas).

Pipeline:
  * luminance + (B-R) colour from the 3 bands
  * multi-scale gaussian / anisotropic-gaussian / annulus / sector-pair filter bank
    at 128^2, 64^2 and 32^2, all kernels zero-sum or applied to sky-subtracted maps
  * relu ratio comparators (scale invariant: both terms scale with galaxy flux)
  * block pooling: tiny centre windows for centre-anchored shape/asymmetry stats,
    wider rings for arms / companions / stars; avgpool for signed or flux maps
  * one dense per pooled map (hand block masks) -> linear measurements, summed
  * relu -> dense rule table (10 classes), all biases 0 so scores scale with flux
    and the argmax is brightness invariant.

Usage: python3 build.py OUTDIR
"""
import sys, json, os
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ---------------------------------------------------------------- kernels
def ksize(sigma):
    return int(2 * int(np.ceil(2.5 * sigma)) + 1)

def grid(k):
    a = np.arange(k) - (k - 1) / 2.0
    yy, xx = np.meshgrid(a, a, indexing="ij")
    return xx, yy

def gauss(sigma, k=None):
    k = k or ksize(sigma)
    xx, yy = grid(k)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def aniso(sp, sq, theta_deg, k=None):
    k = k or ksize(sp)
    xx, yy = grid(k)
    t = np.deg2rad(theta_deg)
    u = xx * np.cos(t) + yy * np.sin(t)
    v = -xx * np.sin(t) + yy * np.cos(t)
    g = np.exp(-(u ** 2 / (2 * sp ** 2) + v ** 2 / (2 * sq ** 2)))
    return g / g.sum()

def annulus(r0, r1, k):
    xx, yy = grid(k)
    r = np.hypot(xx, yy)
    m = ((r >= r0) & (r < r1)).astype(float)
    return m / max(m.sum(), 1)

def sector(theta_deg, halfwidth_deg, r0, r1, k):
    xx, yy = grid(k)
    r = np.hypot(xx, yy)
    ang = np.degrees(np.arctan2(yy, xx))
    d = (ang - theta_deg + 180) % 360 - 180
    m = ((np.abs(d) <= halfwidth_deg) & (r >= r0) & (r < r1)).astype(float)
    return m / 100.0    # sum-normalised: response = sector flux / 100

def sectorpair(theta_deg, halfwidth_deg, r0, r1, k):
    """mean of two opposite sectors: responds to m=0,2,4... ; cancels m=1 (offsets)"""
    return (sector(theta_deg, halfwidth_deg, r0, r1, k) +
            sector(theta_deg + 180, halfwidth_deg, r0, r1, k))

def combine(terms):
    k = max(t[1].shape[0] for t in terms)
    out = np.zeros((k, k))
    for c, ker in terms:
        kh = ker.shape[0]
        o = (k - kh) // 2
        out[o:o + kh, o:o + kh] += c * ker
    return out

# ---------------------------------------------------------------- network
layers = []
weights = {}

def conv(name, inp, kernels, bias=None, stride=1, padding=None):
    k = kernels[0].shape[0]
    p = (k - 1) // 2 if padding is None else padding
    layers.append({"name": name, "op": "conv", "inputs": [inp],
                   "out": 1, "kernel": [k, k], "stride": stride, "padding": p})
    weights[name + ".W"] = np.stack(kernels)[None, :, :, :].astype(np.float32)
    weights[name + ".b"] = (np.zeros(1) if bias is None else np.array(bias, float)).astype(np.float32)

def relu(name, inp):
    layers.append({"name": name, "op": "relu", "inputs": [inp]})

def maxp(name, inp, s):
    layers.append({"name": name, "op": "maxpool", "inputs": [inp], "size": [s, s], "stride": [s, s]})

def avgp(name, inp, s):
    layers.append({"name": name, "op": "avgpool", "inputs": [inp], "size": [s, s], "stride": [s, s]})

def add(name, inps):
    layers.append({"name": name, "op": "add", "inputs": inps})

# ---- stage 0: luminance + colour at 128
conv("lum", "image", [np.array([[0.30]]), np.array([[0.55]]), np.array([[0.15]])], padding=0)
conv("col", "image", [np.array([[-0.5]]), np.array([[0.0]]), np.array([[0.5]])], padding=0)
avgp("L2", "lum", 2)
avgp("C2", "col", 2)

# ---- fine scale (128): stars / compact objects, smoothed colour
conv("f1", "lum", [gauss(1.2)])
conv("f2", "lum", [gauss(2.6)])
conv("starC", "lum", [combine([(1.0, gauss(1.2)), (-3.0, gauss(2.6)), (2.0, gauss(4.6))])])
conv("cbS", "col", [gauss(2.6)])
maxp("p_starC", "starC", 16)          # 8x8
avgp("p_cbS", "cbS", 16)              # 8x8

# ---- medium scale (64), sky-subtracted
conv("sky2", "L2", [combine([(-1.0, gauss(8.0))])])
add("L2s", ["L2", "sky2"])
conv("g2", "L2s", [gauss(2.0)])
conv("g4", "L2s", [gauss(4.0)])
conv("g8", "L2s", [gauss(8.0)])
conv("cb2", "C2", [gauss(2.0)])
conv("annF", "L2s", [annulus(10, 24, 49)])
avgp("L3", "L2", 2)
conv("sky3", "L3", [combine([(-1.0, gauss(5.0))])])
add("L3s", ["L3", "sky3"])
ELC = (0.30, 0.50, 0.72, 0.90)
# ellipticity: quadrupole-moment ratio comparators (offset- and brightness-robust)
def mom_kernels(sig, k):
    xx, yy = grid(k)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
    r2 = xx ** 2 + yy ** 2
    q0 = (xx ** 2 - yy ** 2) * g / g.sum()
    q45 = (2 * xx * yy) * g / g.sum()
    tr = r2 * g / g.sum()
    return q0, q45, tr

for tag, inp, sig in (("A", "L2s", 5.0), ("S", "L3s", 8.0)):
    q0, q45, tr = mom_kernels(sig, 2 * int(np.ceil(2.5 * sig)) + 1)
    conv(f"q0{tag}", inp, [q0])
    conv(f"q45{tag}", inp, [q45])
    conv(f"tr{tag}", inp, [tr])
    for ci, c in enumerate(ELC):
        parts = []
        for i, th in enumerate((0, 45, 90, 135)):
            c2, s2 = np.cos(2 * np.deg2rad(th)), np.sin(2 * np.deg2rad(th))
            conv(f"mq{tag}{ci}_{i}a", f"q0{tag}", [np.array([[c2]])], padding=0)
            conv(f"mq{tag}{ci}_{i}b", f"q45{tag}", [np.array([[s2]])], padding=0)
            conv(f"mq{tag}{ci}_{i}c", f"tr{tag}", [np.array([[-c]])], padding=0)
            add(f"mq{tag}{ci}_{i}", [f"mq{tag}{ci}_{i}a", f"mq{tag}{ci}_{i}b", f"mq{tag}{ci}_{i}c"])
            relu(f"mqr{tag}{ci}_{i}", f"mq{tag}{ci}_{i}")
            parts.append(f"mqr{tag}{ci}_{i}")
        add(f"el{tag}{ci}", parts)
        maxp(f"p_el{tag}{ci}", f"el{tag}{ci}", 2 if tag == "A" else 1)

conv("comp2", "L2s", [combine([(1.0, gauss(1.0)), (-4.0, gauss(4.0))])])
maxp("p_comp2", "comp2", 4)
conv("bulgeD2", "L2s", [combine([(1.0, gauss(2.0)), (-4.0, annulus(5, 12, 25))])])
maxp("p_bulgeD2", "bulgeD2", 4)
# m=2 projection of a radial band-pass: spiral arms (ridge vs surround) and bar
for i, th in enumerate((0, 45, 90, 135)):
    xx, yy = grid(29)
    c2 = np.cos(2 * (np.arctan2(yy, xx) - np.deg2rad(th)))
    amid, ain, aout = annulus(4, 10, 29), annulus(1.5, 4, 29), annulus(10, 15, 29)
    ker = c2 * amid - 0.5 * c2 * ain - 0.5 * c2 * aout
    for mm in (amid, ain, aout):
        ker -= (ker * mm).sum() / mm.sum() * mm * 0.0
    conv(f"armx{i}", "L2s", [ker])
    relu(f"armr{i}", f"armx{i}")
    xb, yb = grid(21)
    c2b = np.cos(2 * (np.arctan2(yb, xb) - np.deg2rad(th)))
    bmid, bin_, bout = annulus(3, 6, 21), annulus(1, 3, 21), annulus(6, 10, 21)
    kerb = c2b * bmid - 0.5 * c2b * bin_ - 0.5 * c2b * bout
    conv(f"barx{i}", "L2s", [kerb])
    relu(f"barr{i}", f"barx{i}")
conv("radB", "L2s", [annulus(4, 10, 29) - 0.5 * annulus(1.5, 4, 29) - 0.5 * annulus(10, 15, 29)])
relu("radBr", "radB")
maxp("p_radB", "radBr", 8)
add("arm2", [f"armr{i}" for i in range(4)])
maxp("p_arm2", "arm2", 8)
add("bar2", [f"barr{i}" for i in range(4)])
maxp("p_bar2", "bar2", 2)
avgp("p_cb2", "cb2", 8)
avgp("p_g4", "g4", 8)
avgp("p_g2", "g2", 8)
avgp("p_annF", "annF", 8)

# ---- coarse scale (32): asymmetry (m=1), companions
conv("h4", "L3s", [gauss(4.0)])
for tag, c in (("a", 1.8), ("b", 3.0)):
    for si, th in enumerate((0, 60, 120)):
        conv(f"asx{tag}_{si}", "L3s",
             [combine([(1.0, sector(th, 30, 3, 14, 29)), (-c, sector(th + 180, 30, 3, 14, 29))])])
        relu(f"asr{tag}_{si}", f"asx{tag}_{si}")
    add(f"asym{tag}", [f"asr{tag}_{si}" for si in range(3)])
    maxp(f"p_asym{tag}", f"asym{tag}", 2)   # 64x64, centre-anchored readout
conv("clump", "L3s", [combine([(1.0, gauss(2.0)), (-2.0, gauss(4.0)), (1.0, gauss(8.0))])])
maxp("p_clump", "clump", 4)                 # 16x16
avgp("p_h4", "h4", 4)                       # 16x16

# ---------------------------------------------------------------- readout
# pooled maps and their grid sizes
MAPS = {"p_starC": 8, "p_cbS": 8,
        "p_elA0": 32, "p_elA1": 32, "p_elA2": 32, "p_elA3": 32,
        "p_elS0": 32, "p_elS1": 32, "p_elS2": 32, "p_elS3": 32,
        "p_comp2": 16, "p_bulgeD2": 16, "p_bar2": 32,
        "p_arm2": 8, "p_radB": 8, "p_cb2": 8, "p_g4": 8, "p_g2": 8, "p_annF": 8,
        "p_asyma": 16, "p_asymb": 16, "p_clump": 8, "p_h4": 8}

def rings(n):
    """block sets on an n x n grid; distances of block centres in 128-px units.
    C0 = tiny centre window (mean), R0..R3 = centre/outward rings (sum)."""
    def d(i):
        return abs((i + 0.5) * 128.0 / n - 64.0)
    C0 = [(r, c) for r in range(n) for c in range(n) if max(d(r), d(c)) <= 8]
    R0 = [(r, c) for r in range(n) for c in range(n) if max(d(r), d(c)) <= 16]
    R1 = [(r, c) for r in range(n) for c in range(n) if 16 < max(d(r), d(c)) <= 30]
    R2 = [(r, c) for r in range(n) for c in range(n) if 30 < max(d(r), d(c)) <= 46]
    R3 = [(r, c) for r in range(n) for c in range(n) if max(d(r), d(c)) > 46]
    CC = [(r, c) for r in range(n) for c in range(n) if max(d(r), d(c)) <= 128.0 / (2 * n)]
    return {"C0": C0, "CC": CC, "R0": R0, "R1": R1, "R2": R2, "R3": R3,
            "R01": R0 + R1, "R12": R1 + R2, "R23": R2 + R3}

SETS = {n: rings(n) for n in (8, 16, 32)}

# measurement rows: (name, [(map, grid, set, coeff)], bias)
MSPEC = [
    ("flux",   [("p_g4", 8, "R0", 1.0)], 0.0),
    ("size",   [("p_annF", 8, "R0", 1.0), ("p_g2", 8, "R0", -0.12)], 0.0),
    ("el1",    [("p_elA0", 32, "CC", 1.0), ("p_elS0", 32, "CC", 1.0), ("p_g4", 8, "R0", -0.04)], 0.0),
    ("el2",    [("p_elA1", 32, "CC", 1.0), ("p_elS1", 32, "CC", 1.0), ("p_g4", 8, "R0", -0.02)], 0.0),
    ("el3",    [("p_elA2", 32, "CC", 1.0), ("p_elS2", 32, "CC", 1.0), ("p_g4", 8, "R0", -0.15)], 0.0),
    ("el4",    [("p_elA3", 32, "CC", 1.0), ("p_elS3", 32, "CC", 1.0), ("p_g4", 8, "R0", -0.15)], 0.0),
    ("comp",   [("p_comp2", 16, "C0", 1.0), ("p_g4", 8, "R0", -0.05)], 0.0),
    ("bulge",  [("p_bulgeD2", 16, "C0", 1.0), ("p_g4", 8, "R0", -0.05)], 0.0),
    ("bar",    [("p_bar2", 32, "CC", 1.0), ("p_g4", 8, "R0", -0.06)], 0.0),
    ("armIn",  [("p_arm2", 8, "R1", 1.0), ("p_radB", 8, "R1", -0.8), ("p_g4", 8, "R0", -0.02)], 0.0),
    ("armOut", [("p_arm2", 8, "R2", 1.0), ("p_radB", 8, "R2", -0.8), ("p_g4", 8, "R0", -0.02)], 0.0),
    ("arm",    [("p_arm2", 8, "R12", 1.0), ("p_radB", 8, "R12", -0.8), ("p_g4", 8, "R0", -0.04)], 0.0),
    ("tight",  [("p_arm2", 8, "R1", 1.0), ("p_arm2", 8, "R2", -1.0)], 0.0),
    ("loose",  [("p_arm2", 8, "R2", 1.0), ("p_arm2", 8, "R1", -1.0)], 0.0),
    ("color",  [("p_cb2", 8, "R01", 1.0), ("p_cbS", 8, "R01", 0.5)], 0.0),
    ("asym",   [("p_asyma", 16, "C0", 1.0), ("p_g4", 8, "R0", -0.02)], 0.0),
    ("asym2",  [("p_asymb", 16, "C0", 1.0), ("p_g4", 8, "R0", -0.01)], 0.0),
    ("clump",  [("p_clump", 8, "R2", 1.0), ("p_g4", 8, "R0", -0.06)], 0.0),
    ("clumpF", [("p_clump", 8, "R3", 1.0), ("p_g4", 8, "R0", -0.04)], 0.0),
    ("star",   [("p_starC", 8, "R23", 1.0), ("p_g4", 8, "R0", -0.02)], 0.0),
    ("star0",  [("p_starC", 8, "R0", 1.0)], 0.0),
    ("flux32", [("p_h4", 8, "R0", 1.0)], 0.0),
]
M = len(MSPEC)
b1 = np.array([bias for (_, _, bias) in MSPEC])

# rule table: score_k = sum_j a[k][j] * relu(meas_j)
MN = [s[0] for s in MSPEC]
def row(**kw):
    return np.array([kw.get(m, 0.0) for m in MN])

RULE = {
    0: row(asym=0.3, asym2=0.1, clump=-0.2),
    1: row(clump=0.15, asym2=0.1),
    2: row(color=-0.5, el3=-0.15),
    3: row(color=-0.5, el2=0.1, el3=-0.1),
    4: row(color=-0.5, el3=0.15, el2=-0.05),
    5: row(color=0.35, bar=0.2, el3=-0.2),
    6: row(color=0.35, tight=0.15, bar=-0.05, el3=-0.2),
    7: row(color=0.35, loose=0.15, el3=-0.2),
    8: row(el3=0.25, bulge=-0.05, color=0.1),
    9: row(el3=0.2, bulge=0.05),
}
PRIOR = {0: 0.03, 1: 0.05, 2: 0.12, 3: 0.07, 4: 0.012, 5: 0.06, 6: 0.05, 7: 0.09, 8: 0.04, 9: 0.05}
for k in range(10):
    RULE[k][MN.index("flux")] += PRIOR[k]
W2 = np.stack([RULE[k] for k in range(10)])
b2 = np.zeros(10)

# one dense per pooled map (block masks), summed by add -> relu -> rule table
used = sorted({m for (_, terms, _) in MSPEC for (m, g, s, c) in terms})
part_names = []
for pi, m in enumerate(used):
    n = MAPS[m]
    Wm = np.zeros((M, n * n))
    for mi, (nm, terms, bias) in enumerate(MSPEC):
        for (mm, g, s, c) in terms:
            if mm == m:
                blk = SETS[g][s]
                cc_ = c / len(blk) if s == "C0" else c
                for (r, cc) in blk:
                    Wm[mi, r * n + cc] += cc_
    layers.append({"name": "d_" + m, "op": "dense", "inputs": [m], "out": M})
    weights["d_" + m + ".W"] = Wm.astype(np.float32)
    weights["d_" + m + ".b"] = (b1 if pi == 0 else np.zeros(M, np.float32)).astype(np.float32)
    part_names.append("d_" + m)
add("measum", part_names)
relu("measr", "measum")
layers.append({"name": "scores", "op": "dense", "inputs": ["measr"], "out": 10})
weights["scores.W"] = W2.astype(np.float32)
weights["scores.b"] = b2.astype(np.float32)

# ---------------------------------------------------------------- save
os.makedirs(OUT, exist_ok=True)
json.dump({"layers": layers}, open(os.path.join(OUT, "network.json"), "w"))
np.savez(os.path.join(OUT, "weights.npz"), **weights)
tot = sum(int(np.prod(v.shape)) for v in weights.values())
print("layers:", len(layers), "weights:", tot)
