import json, sys
from pathlib import Path
import numpy as np

# Hand-built morphology network for 128x128 colour galaxy images.
# The weights are analytic aperture, colour, elongation, spiral, and asymmetry masks.

N = 32
C = 10
SCALES = np.array([2.2, 3.0, 4.1, 5.6, 7.6, 10.2], np.float32)
O = 24
F = 17

Y0, X0 = np.mgrid[0:N, 0:N].astype(np.float32)
X = X0 - (N - 1) / 2.0
Y = Y0 - (N - 1) / 2.0
R = np.sqrt(X * X + Y * Y) + 1e-6
PHI = np.arctan2(Y, X)

LUMA = np.array([1/3, 1/3, 1/3], np.float32)
BLUE = np.array([-0.48, -0.12, 0.60], np.float32)
RED = np.array([0.55, 0.30, -0.35], np.float32)
YELLOW = np.array([0.42, 0.38, -0.20], np.float32)


def rot(x, y, th):
    c, s = np.cos(th), np.sin(th)
    return c * x + s * y, -s * x + c * y


def ell_r(a, q, th):
    xp, yp = rot(X, Y, th)
    return np.sqrt((xp / max(a, 1e-4)) ** 2 + (yp / max(a * q, 1e-4)) ** 2)


def gauss(a, q=1.0, th=0.0):
    return np.exp(-0.5 * ell_r(a, q, th) ** 2).astype(np.float32)


def expd(a, q=1.0, th=0.0):
    return np.exp(-ell_r(a, q, th)).astype(np.float32)


def unit(m):
    m = m.astype(np.float32)
    s = np.sum(np.abs(m))
    return m / max(float(s), 1e-6)


def mean_mask(m):
    m = np.maximum(m.astype(np.float32), 0)
    return m / max(float(m.sum()), 1e-6)


def contrast(pos, neg=None, neg_w=1.0):
    p = mean_mask(pos)
    if neg is None:
        return p
    return p - neg_w * mean_mask(neg)


def wrap_half(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


def spiral_mask(a, th, pitch_deg, chir=1.0, width=0.24):
    pitch = np.deg2rad(pitch_deg)
    env = np.exp(-0.5 * ((R - 1.50 * a) / max(1.05 * a, 1e-3)) ** 2) * (1 - np.exp(-(R / max(0.55 * a, 1e-3)) ** 2))
    phase = PHI - th - chir * np.log(R / max(0.72 * a, 1e-3)) / np.tan(pitch)
    d = wrap_half(phase)
    return (env * np.exp(-0.5 * (d / width) ** 2)).astype(np.float32)


def two_lobe(a, th):
    xp, yp = rot(X, Y, th)
    sep = 1.55 * a
    g1 = np.exp(-0.5 * (((xp - sep/2) / (0.45*a))**2 + (yp / (0.42*a))**2))
    g2 = np.exp(-0.5 * (((xp + sep/2) / (0.45*a))**2 + (yp / (0.42*a))**2))
    cen = np.exp(-0.5 * ((xp / (0.55*a))**2 + (yp / (0.55*a))**2))
    return contrast(g1 + g2, cen, 0.55)


def hemi(a, th):
    xp, yp = rot(X, Y, th)
    env = np.exp(-0.5 * (R / max(1.65*a, 1e-3))**2)
    return unit(np.tanh(xp / max(0.45*a, 1e-3)) * env)


def feature(f, a, oi):
    th = 2*np.pi*((oi % 12)/12.0)
    chir = 1.0 if oi < 12 else -1.0
    disk = gauss(1.55*a, 0.92, th)
    if f == 0:      # round compact central body
        m = contrast(gauss(0.82*a, 0.98, th), gauss(1.95*a, 0.98, th), 0.35); ch=LUMA; g=8.0; b=-0.020
    elif f == 1:    # red/yellow central colour
        m = contrast(gauss(0.95*a, 0.98, th)); ch=RED; g=10.0; b=-0.010
    elif f == 2:    # blue outer colour in a disk/arms
        m = contrast(gauss(1.85*a, 0.88, th), gauss(0.55*a, 0.95, th), 0.30); ch=BLUE; g=12.0; b=-0.008
    elif f == 3:    # extended disk relative to centre
        m = contrast(gauss(2.05*a, 0.90, th), gauss(0.55*a, 0.95, th), 0.80); ch=LUMA; g=8.0; b=-0.010
    elif f == 4:    # moderate elongation
        m = contrast(gauss(1.05*a, 0.55, th), gauss(1.05*a, 0.98, th), 0.78); ch=LUMA; g=12.0; b=-0.012
    elif f == 5:    # cigar elongation
        m = contrast(gauss(1.10*a, 0.32, th), gauss(1.15*a, 0.86, th), 0.62); ch=LUMA; g=13.0; b=-0.012
    elif f == 6:    # very thin edge-on disk
        m = contrast(expd(1.85*a, 0.13, th), expd(1.20*a, 0.55, th), 0.48); ch=LUMA; g=15.0; b=-0.012
    elif f == 7:    # thin blue disk with little central excess
        xp, yp = rot(X, Y, th)
        outer = np.exp(-np.abs(xp)/max(1.9*a,1e-3))*np.exp(-0.5*(yp/max(0.13*a,1e-3))**2)
        cen = gauss(0.42*a, 0.75, th)
        m = contrast(outer, cen, 0.75); ch=BLUE; g=13.0; b=-0.006
    elif f == 8:    # central bar in a broader disk
        m = contrast(gauss(0.78*a, 0.20, th), gauss(1.35*a, 0.82, th), 0.30); ch=LUMA; g=13.0; b=-0.012
    elif f == 9:    # tight two-arm spiral
        arms = spiral_mask(a, th, 13, chir, 0.18)
        m = contrast(arms, disk, 0.20); ch=np.array([0.18,0.30,0.52],np.float32); g=13.0; b=-0.010
    elif f == 10:   # loose open spiral
        arms = spiral_mask(a, th, 34, chir, 0.26)
        m = contrast(arms, gauss(1.90*a,0.95,th), 0.16); ch=np.array([0.18,0.30,0.52],np.float32); g=13.0; b=-0.010
    elif f == 11:   # double nucleus / close pair
        m = two_lobe(a, th); ch=LUMA; g=13.0; b=-0.014
    elif f == 12:   # one-sided asymmetry
        m = hemi(a, th); ch=LUMA; g=9.0; b=-0.010
    elif f == 13:   # off-centre blue clump/tail
        xp, yp = rot(X, Y, th)
        cl = np.exp(-0.5 * (((xp-1.05*a)/(0.42*a))**2 + ((yp-0.35*a)/(0.36*a))**2))
        m = contrast(cl, gauss(0.70*a,0.95,th), 0.25); ch=BLUE; g=12.0; b=-0.006
    elif f == 14:   # compact bulge/concentration
        m = contrast(gauss(0.42*a, 0.92, th), gauss(1.35*a, 0.92, th), 0.36); ch=YELLOW; g=12.0; b=-0.010
    elif f == 15:   # large low surface brightness disk
        m = contrast(gauss(2.45*a, 0.95, th), gauss(0.85*a, 0.95, th), 0.55); ch=LUMA; g=7.0; b=-0.012
    else:           # blue central/inner star-forming structure
        m = contrast(gauss(1.05*a, 0.75, th)); ch=BLUE; g=10.0; b=-0.006
    return (g * ch[:,None,None] * m[None]).astype(np.float32), np.float32(b)


def write(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = [
        {"name":"p4", "op":"avgpool", "inputs":["image"], "size":[4,4], "stride":[4,4]},
        {"name":"feat", "op":"dense", "inputs":["p4"], "out":F*len(SCALES)*O},
        {"name":"feat_relu", "op":"relu", "inputs":["feat"]},
        {"name":"feat_rs", "op":"reshape", "inputs":["feat_relu"], "shape":[F, len(SCALES), O]},
        {"name":"feat_pool", "op":"maxpool", "inputs":["feat_rs"], "size":[len(SCALES), O], "stride":[len(SCALES), O]},
        {"name":"fvec", "op":"reshape", "inputs":["feat_pool"], "shape":[F]},
        {"name":"hidden", "op":"dense", "inputs":["fvec"], "out":40},
        {"name":"hidden_relu", "op":"relu", "inputs":["hidden"]},
        {"name":"scores", "op":"dense", "inputs":["hidden_relu"], "out":C},
    ]
    W = np.zeros((F*len(SCALES)*O, 3, N, N), np.float32)
    b = np.zeros((F*len(SCALES)*O,), np.float32)
    idx = 0
    for f in range(F):
        for a in SCALES:
            for oi in range(O):
                ww, bb = feature(f, float(a), oi)
                W[idx] = ww; b[idx] = bb; idx += 1

    H = np.zeros((40, F), np.float32)
    Hb = np.zeros((40,), np.float32)
    for i in range(F):
        H[i, i] = 1.0
    combos = [
        ({0:0.8,1:0.5,14:0.5,2:-0.7,6:-0.8,11:-0.8,12:-0.6}, -0.02),
        ({4:0.8,1:0.35,14:0.25,2:-0.55,6:-0.6,11:-0.5}, -0.02),
        ({5:0.8,1:0.30,2:-0.45,6:-0.35,11:-0.4}, -0.02),
        ({6:0.9,7:0.35,14:-0.45,1:-0.25}, -0.015),
        ({6:0.65,14:0.65,1:0.35,7:-0.20}, -0.015),
        ({8:0.75,2:0.45,9:0.25,10:0.25,6:-0.55}, -0.02),
        ({9:0.85,2:0.45,14:0.20,8:-0.25,6:-0.45}, -0.02),
        ({10:0.85,2:0.55,3:0.25,14:-0.20,6:-0.40}, -0.02),
        ({12:0.75,13:0.55,2:0.25,0:-0.35,14:-0.25}, -0.02),
        ({11:0.95,12:0.30,13:0.25,6:-0.35}, -0.02),
        ({2:0.65,3:0.30,15:0.25,1:-0.20}, -0.01),
        ({1:0.55,14:0.55,2:-0.30}, -0.01),
        ({4:0.5,5:0.3,6:-0.5,2:-0.25}, -0.01),
        ({6:0.7,14:-0.35,2:0.25}, -0.015),
        ({0:0.45,15:-0.35,2:-0.30}, -0.005),
        ({16:0.45,2:0.35,9:0.15,10:0.15}, -0.005),
        ({11:0.45,12:0.45,13:0.45}, -0.01),
        ({8:0.45,14:0.25,1:0.20}, -0.01),
        ({3:0.35,15:0.35,2:0.25}, -0.01),
        ({5:0.45,6:0.20,14:0.20}, -0.01),
        ({6:0.45,1:0.25,14:0.30}, -0.01),
        ({10:0.45,13:0.25,12:0.20}, -0.01),
        ({9:0.45,8:0.20,14:0.20}, -0.01),
    ]
    for j, (d, bias) in enumerate(combos, start=17):
        for k, v in d.items(): H[j, k] = v
        Hb[j] = bias

    S = np.zeros((C, 40), np.float32)
    # Final hand calibration: use smooth/spiral classes as the ordinary cases,
    # and let the rarer cigar/interaction classes win only on stronger evidence.
    sb = np.array([0.12, 0.08, 0.70, 0.45, -0.80, 0.18, 0.28, 0.38, 0.10, 0.18], np.float32)
    def row(k, entries):
        for i, v in entries.items(): S[k, i] = v
    row(0, {12:0.70,13:0.85,16:0.28,25:1.15,33:0.55,38:0.42,11:0.22,2:0.18,0:-0.22,1:-0.16,14:-0.18,6:-0.18,5:-0.18})
    row(1, {11:1.05,12:0.32,13:0.30,26:1.25,33:0.44,0:-0.22,14:-0.14,6:-0.16,5:-0.16})
    row(2, {0:1.20,1:0.52,14:0.78,17:1.75,31:0.68,2:-0.48,4:-0.18,5:-0.42,6:-0.58,11:-0.42,12:-0.34,13:-0.30})
    row(3, {4:1.00,1:0.42,14:0.48,18:1.45,29:0.58,34:0.38,2:-0.40,5:-0.18,6:-0.46,11:-0.34,12:-0.26})
    row(4, {5:0.30,1:0.12,14:0.14,19:0.35,36:0.12,2:-0.46,9:-0.36,10:-0.36,11:-0.34,12:-0.28,6:-0.32})
    row(5, {8:0.95,2:0.42,9:0.26,10:0.20,22:1.10,34:0.22,39:0.32,6:-0.48,11:-0.22,5:-0.12})
    row(6, {9:1.22,2:0.50,14:0.28,23:1.38,39:0.46,32:0.22,8:-0.20,6:-0.42,11:-0.22})
    row(7, {10:1.18,2:0.54,3:0.24,15:0.24,24:1.35,28:0.42,38:0.38,14:-0.16,6:-0.34,11:-0.16})
    row(8, {6:1.25,7:0.75,20:1.35,30:0.68,2:0.30,14:-0.55,1:-0.32,8:-0.24,11:-0.18})
    row(9, {6:0.82,14:0.72,1:0.42,21:1.05,37:0.42,7:-0.16,2:-0.16,11:-0.14,8:-0.12})

    weights = {
        "feat.W": W.reshape((F*len(SCALES)*O, -1)), "feat.b": b,
        "hidden.W": H, "hidden.b": Hb,
        "scores.W": S, "scores.b": sb,
    }
    (out/"network.json").write_text(json.dumps({"layers":layers}, separators=(",", ":")))
    np.savez(out/"weights.npz", **{k:v.astype(np.float32) for k,v in weights.items()})

if __name__ == "__main__":
    write(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
