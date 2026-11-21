import sys, json
from pathlib import Path
import numpy as np

# Data-free analytic galaxy classifier.  Every number below is generated from
# simple geometric formulae: Gaussian blobs, ellipses, bars, rings, pairs,
# arcs, and logarithmic spirals.  No training image or statistic is read.
N = 32
O = 16
CENTER = (N - 1) / 2.0
EPS = 1e-8
RAWC = 4  # raw colour features: sky-subtracted light, blue excess, red excess, blue light

yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
x0 = xx - CENTER
y0 = yy - CENTER
rr = np.sqrt(x0 * x0 + y0 * y0) + 1e-6
phi0 = np.arctan2(y0, x0)


def rot(theta, cx=0.0, cy=0.0):
    x = x0 - cx
    y = y0 - cy
    c, s = np.cos(theta), np.sin(theta)
    u = c * x + s * y
    v = -s * x + c * y
    return u, v


def gauss(a, b=None, theta=0.0, cx=0.0, cy=0.0):
    if b is None:
        b = a
    u, v = rot(theta, cx, cy)
    return np.exp(-0.5 * ((u / a) ** 2 + (v / b) ** 2)).astype(np.float32)


def ring(r, w):
    return np.exp(-0.5 * ((rr - r) / w) ** 2).astype(np.float32)


def npos(m):
    m = np.maximum(m.astype(np.float32), 0.0)
    return m / (float(m.sum()) + EPS)


def nabs(m):
    m = m.astype(np.float32)
    return m / (float(np.abs(m).sum()) + EPS)


def contrast(pos, neg, a=1.0):
    return npos(pos) - np.float32(a) * npos(neg)


def morph_mask(kind, theta, p):
    # Masks used on the normalized luminance image.  Most have zero mean, so
    # they express shape rather than total brightness.
    if kind == 'core_conc':
        return contrast(gauss(p[0]), gauss(p[1]), p[2])
    if kind == 'outer_conc':
        return contrast(ring(p[0], p[1]), gauss(p[2]), p[3])
    if kind == 'ell':
        a, q, wide = p
        return contrast(gauss(a, a * q, theta), gauss(a, a * wide, theta), 1.0)
    if kind == 'ell_round':
        a, q, rscale = p
        return contrast(gauss(a, a * q, theta), gauss(a * rscale), 1.0)
    if kind == 'bar':
        a, q, round_w = p
        return contrast(gauss(a, a * q, theta), gauss(a * 0.85), round_w)
    if kind == 'line':
        a, q, wide, gap = p
        u, v = rot(theta)
        gate = 1.0 - np.exp(-0.5 * (u / gap) ** 2)
        pos = np.exp(-0.5 * ((u / a) ** 2 + (v / (a * q)) ** 2)) * gate
        neg = np.exp(-0.5 * ((u / a) ** 2 + (v / (a * q * wide)) ** 2)) * gate
        return contrast(pos, neg, 1.0)
    if kind == 'edge_bulge':
        a, q, wide, gap = p
        u, v = rot(theta)
        gate = 1.0 - np.exp(-0.5 * (u / gap) ** 2)
        disk = np.exp(-0.5 * ((u / a) ** 2 + (v / (a * q)) ** 2)) * gate
        flank = np.exp(-0.5 * ((u / a) ** 2 + (v / (a * q * wide)) ** 2)) * gate
        return 1.10 * contrast(disk, flank, 1.0)
    if kind == 'edge_flat':
        a, q, wide, gap, core = p
        u, v = rot(theta)
        gate = 1.0 - np.exp(-0.5 * (u / gap) ** 2)
        disk = np.exp(-0.5 * ((u / a) ** 2 + (v / (a * q)) ** 2)) * gate
        flank = np.exp(-0.5 * ((u / a) ** 2 + (v / (a * q * wide)) ** 2)) * gate
        return 1.1 * contrast(disk, flank, 1.0) - 0.18 * npos(gauss(core))
    if kind == 'spiral':
        r1, r2, beta = p
        mid = 0.5 * (r1 + r2)
        sig = 0.30 * (r2 - r1) + 1e-6
        win = np.exp(-0.5 * ((rr - mid) / sig) ** 2)
        phase = 2.0 * (phi0 - theta - beta * np.log((rr + 1.4) / 4.2))
        m = win * np.cos(phase)
        m = m - win * (float((m * win).sum()) / (float((win * win).sum()) + EPS))
        return nabs(m)
    if kind == 'm1':
        r1, r2 = p
        win = np.exp(-0.5 * ((rr - 0.5 * (r1 + r2)) / (0.35 * (r2 - r1) + 1e-6)) ** 2)
        return nabs(win * np.cos(phi0 - theta))
    if kind == 'offblob':
        rad, sig, negsig = p
        cx, cy = rad * np.cos(theta), rad * np.sin(theta)
        return contrast(gauss(sig, sig, 0.0, cx, cy), gauss(negsig), 0.55)
    if kind == 'pair':
        sep, sig, core, bridge = p
        cx, cy = 0.5 * sep * np.cos(theta), 0.5 * sep * np.sin(theta)
        pos = gauss(sig, sig, 0.0, cx, cy) + gauss(sig, sig, 0.0, -cx, -cy)
        u, v = rot(theta)
        mid = np.exp(-0.5 * ((u / bridge) ** 2 + (v / (bridge * 0.75)) ** 2))
        return contrast(pos, mid + 0.65 * gauss(core), 0.9)
    if kind == 'arc':
        rad, sig, length = p
        cx, cy = rad * np.cos(theta), rad * np.sin(theta)
        return contrast(gauss(length, sig, theta + np.pi / 2.0, cx, cy), gauss(rad + 2.0), 0.35)
    raise ValueError(kind)


def raw_mask(kind, theta, p):
    # Masks used on sky-subtracted or colour-excess images.  Positive masks
    # measure where light/colour is, while zero-sum masks measure gradients.
    if kind == 'gauss':
        return npos(gauss(p[0]))
    if kind == 'ring':
        return npos(ring(p[0], p[1]))
    if kind == 'grad':
        return contrast(ring(p[0], p[1]), gauss(p[2]), p[3])
    if kind == 'linepos':
        a, q = p
        return npos(gauss(a, a * q, theta))
    if kind == 'spiral':
        return morph_mask('spiral', theta, p)
    if kind == 'pairpos':
        sep, sig = p
        cx, cy = 0.5 * sep * np.cos(theta), 0.5 * sep * np.sin(theta)
        return npos(gauss(sig, sig, 0.0, cx, cy) + gauss(sig, sig, 0.0, -cx, -cy))
    if kind == 'offpos':
        rad, sig = p
        cx, cy = rad * np.cos(theta), rad * np.sin(theta)
        return npos(gauss(sig, sig, 0.0, cx, cy))
    raise ValueError(kind)


features = []


def add_feature(name, bias=0.0, oriented=True, morph=None, raw=None):
    mrows = []
    crows = []
    for oi in range(O):
        theta = np.pi * oi / O if oriented else 0.0
        mw = np.zeros((1, N, N), np.float32)
        cw = np.zeros((RAWC, N, N), np.float32)
        if morph is not None:
            kind, params, scale = morph
            mw[0] += np.float32(scale) * morph_mask(kind, theta, params)
        if raw is not None:
            for ch, scale, kind, params in raw:
                cw[ch] += np.float32(scale) * raw_mask(kind, theta, params)
        mrows.append(mw)
        crows.append(cw)
    features.append((name, np.float32(bias), mrows, crows))

# Raw channel convention after colour convolution:
# 0 sky-subtracted white light, 1 blue excess, 2 red/yellow excess, 3 sky-subtracted blue light.
# Radial light and colour, measured on raw intensities.
add_feature('flux_core', -0.035, False, raw=[(0, 1.0, 'gauss', (2.0,))])
add_feature('flux_mid', -0.030, False, raw=[(0, 1.0, 'gauss', (4.2,))])
add_feature('flux_disk', -0.025, False, raw=[(0, 1.0, 'gauss', (6.8,))])
add_feature('flux_outer', -0.018, False, raw=[(0, 1.0, 'ring', (8.0, 2.7))])
add_feature('flux_far', -0.012, False, raw=[(0, 1.0, 'ring', (11.3, 2.8))])
add_feature('conc_core', -0.05, False, morph=('core_conc', (1.8, 5.4, 0.85), 1.0))
add_feature('conc_strong', -0.04, False, morph=('core_conc', (1.25, 4.4, 0.90), 1.0))
add_feature('diffuse_disk', -0.03, False, morph=('outer_conc', (7.2, 2.9, 2.2, 0.75), 1.0))
add_feature('very_diffuse', -0.03, False, morph=('outer_conc', (10.5, 3.2, 3.0, 0.65), 1.0))

add_feature('blue_core', -0.010, False, raw=[(1, 4.0, 'gauss', (2.8,)), (3, 1.2, 'gauss', (2.8,))])
add_feature('blue_disk', -0.012, False, raw=[(1, 4.5, 'gauss', (5.8,)), (3, 1.0, 'gauss', (5.8,))])
add_feature('blue_outer', -0.010, False, raw=[(1, 5.0, 'ring', (7.8, 3.0)), (3, 0.8, 'ring', (7.8, 3.0))])
add_feature('blue_far', -0.008, False, raw=[(1, 5.0, 'ring', (11.2, 3.0)), (3, 0.7, 'ring', (11.2, 3.0))])
add_feature('blue_grad', -0.006, False, raw=[(1, 5.0, 'grad', (7.5, 3.0, 2.5, 0.65))])
add_feature('warm_core', -0.010, False, raw=[(2, 4.0, 'gauss', (2.7,))])
add_feature('warm_disk', -0.012, False, raw=[(2, 3.5, 'gauss', (5.8,))])
add_feature('warm_outer', -0.010, False, raw=[(2, 3.5, 'ring', (7.7, 3.0))])
add_feature('warm_grad', -0.006, False, raw=[(2, 4.0, 'grad', (2.5, 2.0, 7.5, 0.65))])

# Axis ratio, bar and edge-on structure on normalized luminance.
add_feature('round_lack_line', -0.03, False, morph=('core_conc', (3.0, 8.0, 0.55), 0.6))
add_feature('ell_mild', -0.04, True, morph=('ell_round', (5.0, 0.68, 0.92), 1.1))
add_feature('ell_large', -0.04, True, morph=('ell_round', (7.5, 0.66, 0.92), 1.1))
add_feature('cigar', -0.045, True, morph=('ell_round', (6.2, 0.40, 0.82), 1.25))
add_feature('cigar_large', -0.045, True, morph=('ell_round', (8.8, 0.36, 0.82), 1.25))
add_feature('thin', -0.035, True, morph=('line', (8.0, 0.18, 3.2, 2.1), 1.35), raw=[(0, 0.45, 'linepos', (8.0, 0.18))])
add_feature('thin_long', -0.035, True, morph=('line', (11.0, 0.14, 3.4, 2.5), 1.35), raw=[(0, 0.55, 'linepos', (11.0, 0.14))])
add_feature('ultra_thin', -0.035, True, morph=('line', (12.5, 0.10, 3.8, 2.7), 1.4), raw=[(0, 0.55, 'linepos', (12.5, 0.10))])
add_feature('edge_bulge', -0.05, True, morph=('edge_bulge', (10.0, 0.16, 3.2, 2.5), 1.25))
add_feature('edge_flat', -0.04, True, morph=('edge_flat', (10.8, 0.13, 3.5, 2.5, 2.1), 1.25))
add_feature('bar_short', -0.075, True, morph=('bar', (3.6, 0.24, 0.78), 0.70), raw=[(2, 0.25, 'linepos', (3.6, 0.24))])
add_feature('bar_mid', -0.075, True, morph=('bar', (4.9, 0.24, 0.75), 0.75), raw=[(2, 0.25, 'linepos', (4.9, 0.24))])

# Spiral texture and irregular structure.  Both handednesses are represented by the sign of beta.
for beta, lab in [(2.5, 'cw'), (-2.5, 'ccw')]:
    add_feature('spiral_tight_' + lab, -0.025, True, morph=('spiral', (2.5, 8.5, beta), 2.2), raw=[(1, 2.6, 'spiral', (2.8, 9.5, beta))])
for beta, lab in [(1.15, 'cw'), (-1.15, 'ccw'), (0.55, 'open_cw'), (-0.55, 'open_ccw')]:
    add_feature('spiral_loose_' + lab, -0.025, True, morph=('spiral', (3.2, 12.0, beta), 2.2), raw=[(1, 2.8, 'spiral', (3.3, 12.5, beta))])
add_feature('asym_lum', -0.035, True, morph=('m1', (3.8, 10.5), 1.2))
add_feature('asym_blue', -0.020, True, morph=('m1', (4.0, 11.0), 0.7), raw=[(1, 3.8, 'grad', (8.0, 3.0, 2.5, 0.6))])
add_feature('offblob_in', -0.035, True, morph=('offblob', (4.5, 1.4, 5.5), 1.1), raw=[(1, 2.0, 'offpos', (4.5, 1.4))])
add_feature('offblob_out', -0.030, True, morph=('offblob', (7.8, 1.7, 7.0), 1.15), raw=[(1, 2.4, 'offpos', (7.8, 1.7))])
add_feature('pair_close', -0.035, True, morph=('pair', (4.3, 1.4, 2.0, 1.8), 1.3), raw=[(0, 0.8, 'pairpos', (4.3, 1.4))])
add_feature('pair_mid', -0.035, True, morph=('pair', (6.8, 1.7, 2.2, 2.2), 1.3), raw=[(0, 0.8, 'pairpos', (6.8, 1.7)), (1, 1.3, 'pairpos', (6.8, 1.7))])
add_feature('pair_far', -0.030, True, morph=('pair', (9.5, 2.0, 2.5, 2.7), 1.2), raw=[(0, 0.8, 'pairpos', (9.5, 2.0)), (1, 1.2, 'pairpos', (9.5, 2.0))])
add_feature('arc_lum', -0.030, True, morph=('arc', (8.2, 1.2, 4.2), 1.15))
add_feature('arc_blue', -0.020, True, morph=('arc', (8.5, 1.3, 4.5), 0.8), raw=[(1, 2.4, 'offpos', (8.5, 1.6))])

G = len(features)
morph_W = np.zeros((G * O, 1 * N * N), np.float32)
raw_W = np.zeros((G * O, RAWC * N * N), np.float32)
morph_b = np.zeros((G * O,), np.float32)
raw_b = np.zeros((G * O,), np.float32)
for gi, (name, bias, mrows, crows) in enumerate(features):
    for oi in range(O):
        j = gi * O + oi
        morph_W[j] = mrows[oi].reshape(-1)
        raw_W[j] = crows[oi].reshape(-1)
        # Put the full threshold on the raw branch; the morph branch is unbiased.
        raw_b[j] = bias

fname = {name: i for i, (name, _, _, _) in enumerate(features)}
rule_rows = []

def terms(items):
    v = np.zeros((G,), np.float32)
    for name, coef in items:
        v[fname[name]] += np.float32(coef)
    return v

def rule(name, bias, items):
    rule_rows.append((name, np.float32(bias), terms(items)))

# Hand-written morphology propositions.
rule('round_smooth', -1.2, [
    ('conc_core',0.9),('conc_strong',0.7),('warm_core',0.45),('warm_grad',0.35),('flux_core',0.25),('flux_mid',0.18),
    ('blue_outer',-0.55),('blue_grad',-0.45),('diffuse_disk',-0.25),('very_diffuse',-0.25),
    ('ell_mild',-0.25),('cigar',-0.55),('thin_long',-0.95),('ultra_thin',-0.95),('bar_mid',-0.45),
    ('spiral_tight_cw',-0.45),('spiral_tight_ccw',-0.45),('spiral_loose_cw',-0.50),('spiral_loose_ccw',-0.50),('spiral_loose_open_cw',-0.50),('spiral_loose_open_ccw',-0.50),
    ('asym_lum',-0.55),('asym_blue',-0.55),('pair_mid',-0.75),('pair_far',-0.60),('offblob_out',-0.55),('arc_blue',-0.45)])
rule('intermediate_smooth', -1.05, [
    ('conc_core',0.35),('warm_core',0.35),('warm_disk',0.30),('ell_mild',0.95),('ell_large',0.85),('cigar',0.22),
    ('blue_outer',-0.50),('blue_grad',-0.40),('thin_long',-0.90),('ultra_thin',-0.85),('bar_mid',-0.35),
    ('spiral_loose_cw',-0.35),('spiral_loose_ccw',-0.35),('asym_blue',-0.40),('pair_mid',-0.55)])
rule('cigar_smooth', -0.95, [
    ('cigar',1.10),('cigar_large',1.10),('ell_large',0.20),('warm_disk',0.28),('warm_outer',0.22),
    ('blue_outer',-0.55),('blue_far',-0.35),('thin_long',-0.55),('ultra_thin',-0.75),('bar_mid',-0.25),
    ('spiral_loose_cw',-0.30),('spiral_loose_ccw',-0.30),('asym_blue',-0.40),('pair_mid',-0.45)])
rule('edge_no_bulge', -0.85, [
    ('thin',0.70),('thin_long',1.05),('ultra_thin',1.20),('edge_flat',1.15),('blue_disk',0.25),('blue_outer',0.35),('diffuse_disk',0.20),
    ('conc_strong',-0.50),('warm_core',-0.25),('bar_mid',-0.55),('spiral_loose_cw',-0.20),('spiral_loose_ccw',-0.20),('pair_mid',-0.35)])
rule('edge_with_bulge', -1.05, [
    ('thin',0.70),('thin_long',1.05),('ultra_thin',0.55),('edge_bulge',1.15),('conc_core',0.22),('conc_strong',0.22),('warm_core',0.25),
    ('blue_far',-0.25),('bar_mid',-0.45),('asym_blue',-0.30),('pair_mid',-0.40)])
rule('barred_spiral', -0.95, [
    ('bar_short',1.10),('bar_mid',1.25),('warm_core',0.15),('blue_disk',0.35),('blue_outer',0.35),('flux_disk',0.25),
    ('spiral_tight_cw',0.25),('spiral_tight_ccw',0.25),('spiral_loose_cw',0.22),('spiral_loose_ccw',0.22),
    ('thin_long',-1.20),('ultra_thin',-1.15),('edge_bulge',-0.55),('edge_flat',-0.55),('pair_mid',-0.55),('asym_lum',-0.25)])
rule('tight_spiral', -0.95, [
    ('spiral_tight_cw',1.05),('spiral_tight_ccw',1.05),('blue_disk',0.38),('blue_outer',0.25),('conc_core',0.25),('flux_mid',0.25),
    ('bar_mid',-0.45),('thin_long',-0.85),('ultra_thin',-0.85),('pair_mid',-0.45),('asym_blue',-0.25)])
rule('loose_spiral', -0.95, [
    ('spiral_loose_cw',0.90),('spiral_loose_ccw',0.90),('spiral_loose_open_cw',1.05),('spiral_loose_open_ccw',1.05),
    ('blue_outer',0.55),('blue_far',0.35),('blue_grad',0.35),('diffuse_disk',0.35),('very_diffuse',0.28),('flux_outer',0.25),('flux_far',0.20),
    ('bar_mid',-0.42),('thin_long',-0.70),('ultra_thin',-0.75),('pair_mid',-0.30)])
rule('merging_pair', -0.85, [
    ('pair_close',1.10),('pair_mid',1.30),('pair_far',1.05),('offblob_in',0.35),('offblob_out',0.45),('asym_lum',0.25),('asym_blue',0.30),
    ('thin_long',-0.45),('ultra_thin',-0.45),('conc_strong',-0.20)])
rule('disturbed_asym', -0.80, [
    ('asym_lum',0.90),('asym_blue',1.05),('offblob_in',0.58),('offblob_out',0.70),('arc_lum',0.70),('arc_blue',0.85),('blue_outer',0.30),('blue_far',0.25),
    ('pair_mid',-0.20),('conc_strong',-0.30),('thin_long',-0.30)])
rule('blue_disk', -0.45, [('blue_disk',0.65),('blue_outer',0.75),('blue_far',0.45),('blue_grad',0.55),('warm_core',-0.25),('warm_disk',-0.20)])
rule('warm_smooth', -0.45, [('warm_core',0.65),('warm_disk',0.45),('warm_outer',0.25),('warm_grad',0.50),('blue_outer',-0.45),('blue_grad',-0.35),('asym_blue',-0.25)])
rule('large_diffuse', -0.45, [('flux_disk',0.25),('flux_outer',0.35),('flux_far',0.25),('diffuse_disk',0.60),('very_diffuse',0.55),('conc_strong',-0.25)])
rule('compact', -0.45, [('flux_core',0.35),('conc_core',0.45),('conc_strong',0.55),('diffuse_disk',-0.25),('very_diffuse',-0.25)])

R = len(rule_rows)
rule_W = np.zeros((R, G), np.float32)
rule_b = np.zeros((R,), np.float32)
for i, (name, b, row) in enumerate(rule_rows):
    rule_W[i] = row
    rule_b[i] = b
rname = {name: i for i, (name, _, _) in enumerate(rule_rows)}
score_W = np.zeros((10, R), np.float32)
score_b = np.array([0.08, 0.46, -0.10, 0.48, 0.42, -0.35, 0.58, 0.72, 0.82, -1.00], np.float32)

def rw(n, vals):
    i = rname[n]
    for k, v in vals.items():
        score_W[k, i] += np.float32(v)

rw('round_smooth', {2:1.00, 3:0.42, 0:-0.20, 1:-0.18, 8:-0.18, 9:-0.10})
rw('intermediate_smooth', {3:1.35, 2:0.18, 4:0.35, 0:-0.12, 8:-0.15})
rw('cigar_smooth', {4:1.55, 3:0.25, 8:-0.10, 9:-0.10, 5:-0.12})
rw('edge_no_bulge', {8:1.95, 9:-0.12, 4:0.08, 5:-0.18, 2:-0.18, 3:-0.15})
rw('edge_with_bulge', {9:0.62, 8:0.12, 4:0.08, 5:-0.15, 2:-0.12, 3:-0.10})
rw('barred_spiral', {5:0.75, 6:0.18, 7:0.08, 8:-0.16, 9:-0.35, 2:-0.15, 3:-0.10})
rw('tight_spiral', {6:1.85, 5:0.12, 7:0.12, 2:-0.18, 3:-0.15, 4:-0.12, 8:-0.08, 9:-0.22})
rw('loose_spiral', {7:1.90, 5:0.06, 6:0.08, 0:0.10, 2:-0.18, 3:-0.14, 4:-0.12, 9:-0.25})
rw('merging_pair', {1:2.00, 0:0.18, 7:0.05, 2:-0.18, 3:-0.14, 4:-0.12, 8:-0.08, 9:-0.25})
rw('disturbed_asym', {0:1.30, 1:0.45, 7:0.12, 5:0.02, 2:-0.20, 3:-0.14, 4:-0.12, 9:-0.22})
rw('blue_disk', {7:0.75, 6:0.62, 5:0.20, 0:0.18, 8:0.30, 2:-0.20, 3:-0.14, 4:-0.12, 9:-0.22})
rw('warm_smooth', {2:0.22, 3:0.35, 4:0.28, 9:0.02, 0:-0.12, 7:-0.10})
rw('large_diffuse', {7:0.75, 0:0.15, 1:0.18, 8:0.25, 2:-0.12, 4:-0.05, 9:-0.18})
rw('compact', {2:0.12, 6:0.16, 9:-0.02, 8:-0.05, 7:-0.05})

network = {"layers": [
    {"name": "down", "op": "avgpool", "inputs": ["image"], "size": [4, 4], "stride": [4, 4]},
    {"name": "lum", "op": "conv", "inputs": ["down"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "nlum", "op": "norm", "inputs": ["lum"]},
    {"name": "raw", "op": "conv", "inputs": ["down"], "out": RAWC, "kernel": [1, 1], "stride": 1, "padding": 0},
    {"name": "mfeat", "op": "dense", "inputs": ["nlum"], "out": G * O},
    {"name": "cfeat", "op": "dense", "inputs": ["raw"], "out": G * O},
    {"name": "feat_sum", "op": "add", "inputs": ["mfeat", "cfeat"]},
    {"name": "feat_relu", "op": "relu", "inputs": ["feat_sum"]},
    {"name": "feat_map", "op": "reshape", "inputs": ["feat_relu"], "shape": [G, 1, O]},
    {"name": "orient_max", "op": "maxpool", "inputs": ["feat_map"], "size": [1, O], "stride": [1, O]},
    {"name": "feat_vec", "op": "reshape", "inputs": ["orient_max"], "shape": [G]},
    {"name": "rule", "op": "dense", "inputs": ["feat_vec"], "out": R},
    {"name": "rule_relu", "op": "relu", "inputs": ["rule"]},
    {"name": "score", "op": "dense", "inputs": ["rule_relu"], "out": 10}
]}

lum_W = np.zeros((1, 3, 1, 1), np.float32)
lum_W[0, :, 0, 0] = [1/3, 1/3, 1/3]
lum_b = np.zeros((1,), np.float32)
rawconv_W = np.zeros((RAWC, 3, 1, 1), np.float32)
rawconv_b = np.zeros((RAWC,), np.float32)
rawconv_W[0, :, 0, 0] = [1/3, 1/3, 1/3]
rawconv_b[0] = -0.13
rawconv_W[1, :, 0, 0] = [-0.5, -0.5, 1.0]
rawconv_W[2, :, 0, 0] = [1.0, 0.25, -0.75]
rawconv_W[3, :, 0, 0] = [0.0, 0.0, 1.0]
rawconv_b[3] = -0.13

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
out.mkdir(parents=True, exist_ok=True)
(out / "network.json").write_text(json.dumps(network, separators=(",", ":")))
np.savez(out / "weights.npz", **{
    "lum.W": lum_W, "lum.b": lum_b,
    "raw.W": rawconv_W, "raw.b": rawconv_b,
    "mfeat.W": morph_W, "mfeat.b": morph_b,
    "cfeat.W": raw_W, "cfeat.b": raw_b,
    "rule.W": rule_W, "rule.b": rule_b,
    "score.W": score_W, "score.b": score_b,
})
