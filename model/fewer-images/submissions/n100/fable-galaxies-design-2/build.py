"""Writes network.json and weights.npz for a hand-designed galaxy classifier.  Usage: python3 build.py OUT_DIR
Reads no files.  Every weight comes from the formulas below (analytic galaxy shape models typed from general
knowledge: Sersic ellipses, thin disks with bulges, bars, logarithmic spirals, pairs of blobs).  No training.

Network:  luminance (1x1 conv) -> 2x2 average pool (64x64) -> norm (zero mean, unit variance over the frame)
          -> dense bank of model templates (each one rendered model, made zero-mean and unit-length by itself)
          -> max over the templates of each class -> gain and hand-set class biases.
The response of a template is the cosine between the frame and the model, so the largest response is the model that
fits the frame best in the least-squares sense (free brightness and free sky level)."""
import sys, json, math
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
S = 64                                    # working resolution after the 2x2 average pool
yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
xx -= (S - 1) / 2.0
yy -= (S - 1) / 2.0
SOFT = 0.7                                # seeing + pixel softening, in 64-scale pixels (about 0.7 arcsec sigma)


def rot(pa, x=xx, y=yy):
    c, s = math.cos(pa), math.sin(pa)
    return c * x + s * y, -s * x + c * y


def sersic(re, q, pa, n, x0=0.0, y0=0.0):
    """Elliptical Sersic blob: half-light radius re along the major axis, axis ratio q, angle pa, index n."""
    u, v = rot(pa, xx - x0, yy - y0)
    qe = math.sqrt(q * q + (SOFT / re) ** 2)          # the seeing widens the minor axis
    r = np.sqrt(u * u + SOFT * SOFT + (v / qe) ** 2)
    bn = 2.0 * n - 1.0 / 3.0
    return np.exp(-bn * ((r / re) ** (1.0 / n) - (SOFT / re) ** (1.0 / n)))


def disk_coords(q, pa):
    u, v = rot(pa)
    v = v / q
    return np.sqrt(u * u + v * v + SOFT * SOFT), np.arctan2(v, u)


def spiral(h, q, pa, phase, chir, pitch_deg, amp, bulge, m=2):
    """Exponential disk (scale length h) with a small bulge and m logarithmic spiral arms of the given pitch angle."""
    r, th = disk_coords(q, pa)
    k = m / math.tan(math.radians(pitch_deg))
    w = r * r / (r * r + (0.7 * h) ** 2)              # no arms inside the bulge
    arms = 1.0 + amp * w * np.cos(m * (th - phase) - chir * k * np.log(r / h))
    return bulge * np.exp(-r / (0.25 * h)) + np.exp(-r / h) * arms


def barred(h, q, pa, bar_ang, chir):
    """Bulge + bar (axis ratio 0.3) + exponential disk with two arms that start at the ends of the bar."""
    r, th = disk_coords(q, pa)
    ub, vb = rot(bar_ang)
    rb = np.sqrt(ub * ub + (vb / 0.3) ** 2 + SOFT * SOFT)
    bar = 1.5 * np.exp(-rb / (0.45 * h))
    k = 2.0 / math.tan(math.radians(20.0))
    u0, v0 = rot(pa, math.cos(bar_ang), math.sin(bar_ang))
    th_bar = math.atan2(v0 / q, u0)                    # bar direction measured in the disk plane
    w = r ** 4 / (r ** 4 + h ** 4)
    arms = 1.0 + 0.5 * w * np.cos(2.0 * (th - th_bar) - chir * k * np.log(r / h))
    return 1.5 * np.exp(-r / (0.2 * h)) + bar + 0.6 * np.exp(-r / (1.2 * h)) * arms


def angles(n, span=math.pi):
    return [span * i / n for i in range(n)]


classes = [[] for _ in range(10)]

# 2: round smooth  ------------------------------------------------------------------------------------------
for re in (0.8, 1.2, 1.8, 2.7, 4.0, 6.0):
    for n in (1.0, 2.0):
        classes[2].append(sersic(re, 1.0, 0.0, n))
        for pa in angles(6):
            classes[2].append(sersic(re, 0.9, pa, n))
# 3: in-between smooth  -------------------------------------------------------------------------------------
for re in (1.0, 1.5, 2.2, 3.3, 5.0, 7.5):
    for n in (1.0, 2.0):
        for q in (0.75, 0.6):
            for pa in angles(12):
                classes[3].append(sersic(re, q, pa, n))
# 4: cigar-shaped smooth (small, elongated, smooth)  --------------------------------------------------------
for re in (1.5, 2.2, 3.3, 5.0):
    for n in (1.0, 2.0):
        for q in (0.45, 0.35):
            for pa in angles(18):
                classes[4].append(sersic(re, q, pa, n))
# 8: edge-on disk without bulge (long, thin, flat profile)  -------------------------------------------------
for re in (3.5, 5.0, 7.5, 11.0):
    for n in (0.7, 1.0):
        for q in (0.22, 0.14):
            for pa in angles(18):
                classes[8].append(sersic(re, q, pa, n))
# 9: edge-on disk with bulge (thin disk plus a round central bulge)  ----------------------------------------
for re in (3.5, 5.0, 7.5, 11.0):
    for bfrac, bamp in ((0.22, 2.0), (0.32, 3.0)):
        for q in (0.22, 0.14):
            for pa in angles(18):
                classes[9].append(sersic(re, q, pa, 1.0) + bamp * sersic(bfrac * re, 0.8, pa, 1.0))
# 5, 6, 7: barred, tightly wound and loosely wound spirals  -------------------------------------------------
for h in (3.0, 5.0, 8.0):
    for chir in (1.0, -1.0):
        for ph in angles(18):
            classes[5].append(barred(h, 1.0, 0.0, ph, chir))
            classes[6].append(spiral(h, 1.0, 0.0, ph, chir, 12.0, 0.5, 2.0))
            classes[7].append(spiral(h, 1.0, 0.0, ph, chir, 30.0, 0.6, 1.0))
        for pa in angles(6):
            for ph in angles(6):
                classes[5].append(barred(h, 0.7, pa, pa + ph, chir))
                classes[6].append(spiral(h, 0.65, pa, ph, chir, 12.0, 0.5, 2.0))
                classes[7].append(spiral(h, 0.65, pa, ph, chir, 30.0, 0.6, 1.0))
# 1: merging galaxies (two blobs of comparable brightness close together)  ----------------------------------
for re in (1.2, 2.0, 3.0):
    for d in (3.0, 5.0, 8.0):
        for a in angles(12, 2 * math.pi):
            for ratio in (0.5, 1.0):
                classes[1].append(sersic(re, 1.0, 0.0, 1.0) + ratio * sersic(re, 1.0, 0.0, 1.0, d * math.cos(a), d * math.sin(a)))
        for a in angles(6):
            dx, dy = 0.5 * d * math.cos(a), 0.5 * d * math.sin(a)
            classes[1].append(sersic(re, 1.0, 0.0, 1.0, dx, dy) + sersic(re, 1.0, 0.0, 1.0, -dx, -dy))
# 0: disturbed (lopsided: a one-armed open spiral / tail, or a core inside an offset envelope)  -------------
for h in (3.0, 5.0, 8.0):
    for chir in (1.0, -1.0):
        for ph in angles(18, 2 * math.pi):
            classes[0].append(spiral(h, 1.0, 0.0, ph, chir, 40.0, 0.8, 1.5, m=1))
            classes[0].append(spiral(h, 0.6, ph, 0.0, chir, 40.0, 0.8, 1.5, m=1))
    for a in angles(12, 2 * math.pi):
        for off in (0.6, 1.0):
            classes[0].append(2.0 * sersic(0.5 * h, 0.8, a, 1.0) + sersic(2.0 * h, 0.7, a, 1.0, off * h * math.cos(a), off * h * math.sin(a)))

T = max(len(c) for c in classes)
bank = np.zeros((10, T, S * S), np.float64)
for k, cl in enumerate(classes):
    for i in range(T):
        t = cl[i % len(cl)].reshape(-1)                # shorter lists repeat their own templates
        t = t - t.mean()                               # each template normalised by itself alone
        t = t / math.sqrt(float((t * t).sum()))
        bank[k, i] = t / S                             # the normed frame has length S, so the response is a cosine

GAIN = 40.0
# class frequencies of this catalogue (general knowledge of the Galaxy10 DECaLS class sizes), plus a hand-set
# handicap for the two classes whose models have an extra free component and so fit any frame more easily
counts = np.array([1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873], np.float64)
bias = np.log(counts / counts.sum())
bias[0] -= 0.5
bias[1] -= 0.5

weights = {
    "lum.W": np.full((1, 3, 1, 1), 1.0 / 3.0, np.float32),
    "lum.b": np.zeros(1, np.float32),
    "tm.W": bank.reshape(10 * T, S * S).astype(np.float32),
    "tm.b": np.zeros(10 * T, np.float32),
    "out.W": (GAIN * np.eye(10)).astype(np.float32),
    "out.b": bias.astype(np.float32),
}
net = {"layers": [
    {"name": "lum", "op": "conv", "inputs": ["image"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "half", "op": "avgpool", "inputs": ["lum"], "size": [2, 2], "stride": [2, 2]},
    {"name": "nrm", "op": "norm", "inputs": ["half"]},
    {"name": "tm", "op": "dense", "inputs": ["nrm"], "out": 10 * T},
    {"name": "grp", "op": "reshape", "inputs": ["tm"], "shape": [10, T, 1]},
    {"name": "best", "op": "maxpool", "inputs": ["grp"], "size": [T, 1], "stride": [T, 1]},
    {"name": "out", "op": "dense", "inputs": ["best"], "out": 10},
]}
with open(out_dir + "/network.json", "w") as f:
    json.dump(net, f, indent=1)
np.savez(out_dir + "/weights.npz", **weights)
print("templates per class:", [len(c) for c in classes], "T =", T)
