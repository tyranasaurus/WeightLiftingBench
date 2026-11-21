#!/usr/bin/env python3
"""Build a fixed, analytic galaxy morphology network without data or fitting."""
import json
import math
import sys
from pathlib import Path

import numpy as np


def build(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    layers = [
        {"name": "half", "op": "avgpool", "inputs": ["image"],
         "size": [2, 2], "stride": [2, 2]},
        {"name": "phot", "op": "conv", "inputs": ["half"], "out": 18,
         "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "phot_relu", "op": "relu", "inputs": ["phot"]},
        {"name": "quarter", "op": "avgpool", "inputs": ["phot_relu"],
         "size": [2, 2], "stride": [2, 2]},
        {"name": "primitive", "op": "dense", "inputs": ["quarter"],
         "out": 774},
        {"name": "primitive_relu", "op": "relu", "inputs": ["primitive"]},
        {"name": "concept", "op": "dense", "inputs": ["primitive_relu"],
         "out": 511},
        {"name": "concept_relu", "op": "relu", "inputs": ["concept"]},
        {"name": "scores", "op": "dense", "inputs": ["concept_relu"],
         "out": 10},
    ]

    # The first bank is photometric rather than learned: soft isophotes, two
    # colour residuals, Difference-of-Gaussian residuals, and oriented ridges.
    C = np.zeros((18, 3, 5, 5), np.float32)
    cb = np.zeros(18, np.float32)
    lum = np.array([0.35, 0.45, 0.20], np.float32)
    thresholds = [.015, .025, .040, .060, .090, .140, .210, .320, .480]
    for o, t in enumerate(thresholds):
        C[o, :, 2, 2] = lum
        cb[o] = -t
    C[9, 2, 2, 2] = 1.0
    C[9, 0, 2, 2] = -.5
    C[9, 1, 2, 2] = -.5
    cb[9] = -.004
    C[10, 0, 2, 2] = 1.0
    C[10, 2, 2, 2] = -1.0
    cb[10] = -.006

    yy5, xx5 = np.mgrid[-2:3, -2:3].astype(np.float64)
    gwide = np.exp(-(xx5 * xx5 + yy5 * yy5) / (2 * 1.35 ** 2))
    gwide /= gwide.sum()
    delta = np.zeros((5, 5), np.float64)
    delta[2, 2] = 1.0
    dog = delta - gwide
    for ch in range(3):
        C[11, ch] = lum[ch] * dog
        C[12, ch] = -lum[ch] * dog
    cb[11:13] = -.0015

    for j, theta in enumerate([0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]):
        u = xx5 * math.cos(theta) + yy5 * math.sin(theta)
        v = -xx5 * math.sin(theta) + yy5 * math.cos(theta)
        core = np.exp(-.5 * ((u / 1.65) ** 2 + (v / .48) ** 2))
        surround = np.exp(-.5 * ((u / 1.75) ** 2 + (v / 1.40) ** 2))
        ridge = core / core.sum() - surround / surround.sum()
        for ch in range(3):
            C[13 + j, ch] = lum[ch] * ridge
        cb[13 + j] = -.0008
    C[17, :, 2, 2] = lum
    cb[17] = -.70

    # Coordinates are expressed in original-image pixels. Each cell at this
    # point is the average of a 4x4 region, with the optical centre at zero.
    coord = np.arange(32, dtype=np.float64) * 4.0 - 62.0
    yy, xx = np.meshgrid(coord, coord, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    phi = np.arctan2(yy, xx)

    primitive_rows = []
    primitive_meta = []
    gauss = {}
    rings = {}
    harmonics = {}
    spirals = {}

    def add_primitive(channel, mask, meta):
        row = np.zeros((18, 32, 32), np.float32)
        row[channel] = mask.astype(np.float32)
        idx = len(primitive_rows)
        primitive_rows.append(row)
        primitive_meta.append(meta)
        return idx

    scales = [3, 5, 8, 12, 18, 27, 40]
    for ch in range(18):
        for scale in scales:
            env = np.exp(-.5 * (rr / scale) ** 2)
            env /= env.sum()
            gauss[ch, scale] = add_primitive(ch, env, ("gauss", ch, scale))

    ring_channels = [0, 2, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16]
    radii = [5, 8, 12, 18, 27, 38]
    for ch in ring_channels:
        for radius in radii:
            width = 1.4 + .34 * radius
            env = np.exp(-.5 * ((rr - radius) / width) ** 2)
            env *= 1.0 - np.exp(-.5 * (rr / 2.0) ** 2)
            env /= env.sum()
            rings[ch, radius] = add_primitive(ch, env, ("ring", ch, radius))

    harmonic_radii = [6, 10, 16, 25, 36]
    phase_counts = {1: 4, 2: 8, 3: 4, 4: 8}
    for ch in [0, 2, 4]:
        for radius in harmonic_radii:
            width = 1.8 + .36 * radius
            env = np.exp(-.5 * ((rr - radius) / width) ** 2)
            env *= 1.0 - np.exp(-.5 * (rr / 2.0) ** 2)
            env /= env.sum()
            for mode in [1, 2, 3, 4]:
                ids = []
                nphase = phase_counts[mode]
                for p in range(nphase):
                    phase = 2 * math.pi * p / nphase
                    mask = env * np.cos(mode * phi - phase)
                    ids.append(add_primitive(ch, mask,
                                             ("harmonic", ch, radius, mode, p)))
                harmonics[ch, radius, mode] = ids

    for ch in [0, 9, 11]:
        for radius in [10, 18, 28]:
            # A broad log-radius aperture avoids coding a preferred galaxy size.
            logr = np.log((rr + 1.5) / radius)
            env = np.exp(-.5 * (logr / .50) ** 2)
            env *= 1.0 - np.exp(-.5 * (rr / 2.2) ** 2)
            env /= env.sum()
            for beta in [-6.0, -3.0, 3.0, 6.0]:
                ids = []
                for p in range(6):
                    phase = 2 * math.pi * p / 6
                    mask = env * np.cos(2 * phi + beta * logr - phase)
                    ids.append(add_primitive(ch, mask,
                                             ("spiral", ch, radius, beta, p)))
                spirals[ch, radius, beta] = ids

    assert len(primitive_rows) == 774
    P = np.stack(primitive_rows, axis=0)
    Pb = np.zeros(len(primitive_rows), np.float32)

    # The second dense layer turns phase samples into rotation-invariant hinge
    # tests. Every comparison is dimensionless in spirit: morphology amplitude
    # or a broad-aperture flux is compared with flux in the same neighbourhood.
    concept_rows = []
    concept_bias = []
    concept_meta = []

    def add_concept(combo, meta, bias=0.0):
        row = np.zeros(774, np.float32)
        for idx, value in combo.items():
            row[idx] += value
        concept_rows.append(row)
        concept_bias.append(bias)
        concept_meta.append(meta)

    def amp_combo(ids, factor=1.0):
        coefficient = factor * math.pi / len(ids)
        return {idx: coefficient for idx in ids}

    def merge(*items):
        result = {}
        for combo, factor in items:
            for idx, value in combo.items():
                result[idx] = result.get(idx, 0.0) + factor * value
        return result

    # Radial concentration and extension at several isophotes.
    pairs = [(3, 8), (5, 12), (8, 18), (12, 27), (18, 40)]
    for ch in [0, 2, 4]:
        for small, large in pairs:
            for k in [1.2, 1.6, 2.2, 3.2]:
                add_concept({gauss[ch, small]: 1.0, gauss[ch, large]: -k},
                            ("compact", ch, small, large, k))
            for k in [.15, .30, .50, .70]:
                add_concept({gauss[ch, large]: 1.0, gauss[ch, small]: -k},
                            ("extended", ch, small, large, k))

    # Angular m=1..4 amplitudes: elongation, lopsidedness, triangular
    # disturbance and box/bar structure respectively.
    eradii = [6, 10, 16, 25, 36]
    ethresholds = [.04, .08, .14, .22, .32]
    for ch in [0, 2, 4]:
        for radius in eradii:
            mass_radius = min(radii, key=lambda z: abs(z - radius))
            mass_idx = rings[ch, mass_radius]
            for k in ethresholds:
                combo = amp_combo(harmonics[ch, radius, 2])
                combo[mass_idx] = combo.get(mass_idx, 0.0) - k
                add_concept(combo, ("elliptic", ch, radius, k))
            for mode, label in [(1, "lopsided"), (3, "disturbed")]:
                for k in [.04, .08, .14, .24]:
                    combo = amp_combo(harmonics[ch, radius, mode])
                    combo[mass_idx] = combo.get(mass_idx, 0.0) - k
                    add_concept(combo, (label, ch, radius, k))
            for k in [.04, .09, .16, .26]:
                combo = amp_combo(harmonics[ch, radius, 4])
                combo[mass_idx] = combo.get(mass_idx, 0.0) - k
                add_concept(combo, ("boxy", ch, radius, k))

    # Logarithmic two-arm responses, made chirality-invariant by combining
    # left- and right-handed versions of the same pitch.
    for ch in [0, 9, 11]:
        for radius in [10, 18, 28]:
            mass_radius = min(radii, key=lambda z: abs(z - radius))
            mass_idx = rings[ch, mass_radius]
            for abeta in [3.0, 6.0]:
                plus = amp_combo(spirals[ch, radius, abeta])
                minus = amp_combo(spirals[ch, radius, -abeta])
                both = merge((plus, 1.0), (minus, 1.0))
                for k in [.03, .07, .12, .20]:
                    combo = dict(both)
                    combo[mass_idx] = combo.get(mass_idx, 0.0) - k
                    add_concept(combo, ("spiral", ch, radius, abeta, k))

    # Colour, residual texture, and straight-ridge evidence.
    for scale in [8, 12, 18, 27]:
        for k in [.02, .05, .10, .18]:
            add_concept({gauss[11, scale]: 1.0, gauss[0, scale]: -k},
                        ("texture", scale, k))
            add_concept({gauss[9, scale]: 1.0, gauss[0, scale]: -k},
                        ("blue", scale, k))
            add_concept({gauss[10, scale]: 1.0, gauss[0, scale]: -k},
                        ("red", scale, k))
            combo = {gauss[q, scale]: .25 for q in [13, 14, 15, 16]}
            combo[gauss[0, scale]] = -k
            add_concept(combo, ("ridge", scale, k))
    assert len(concept_rows) == 511
    Q = np.stack(concept_rows, axis=0)
    Qb = np.asarray(concept_bias, np.float32)

    # Hand-written morphology rule table. Coefficients are shared over scales
    # and isophotes; they encode astronomical definitions, not observations.
    S = np.zeros((10, 511), np.float32)
    for j, meta in enumerate(concept_meta):
        kind = meta[0]
        if kind == "compact":
            k = meta[-1]
            strength = {1.2: .03, 1.6: .06, 2.2: .11, 3.2: .18}[k] / 15
            vals = [-.02, -.01, 1.0, .65, .30, .35, .70, -.75, -.60, .85]
            S[:, j] = strength * np.asarray(vals)
        elif kind == "extended":
            k = meta[-1]
            strength = {.15: .04, .30: .07, .50: .11, .70: .16}[k] / 15
            vals = [.55, .30, -.65, -.25, -.10, .35, .50, 1.0, .75, -.45]
            S[:, j] = strength * np.asarray(vals)
        elif kind == "elliptic":
            _, ch, radius, k = meta
            level = ethresholds.index(k)
            # m=2 in the inner aperture is also the defining bar evidence.
            bar = (1.0 if radius <= 16 else -.70) * [1.0, .9, .6][[0, 2, 4].index(ch)]
            table = np.array([
                [.00, .00, -.03, -.06, -.10],
                [.02, .03, .02, -.02, -.08],
                [-.22, -.30, -.38, -.45, -.50],
                [.20, .10, -.10, -.30, -.45],
                [.05, .15, .28, .38, .42],
                [.08, .13, .12, -.04, -.22],
                [.03, .04, -.02, -.13, -.25],
                [.02, .02, -.04, -.15, -.27],
                [.08, .20, .38, .55, .70],
                [.08, .19, .37, .55, .73],
            ], dtype=np.float32)
            S[:, j] = table[:, level] / 15
            S[5, j] += (.18 * bar * [1.0, .8, .45, .15, -.30][level]) / 15
        elif kind in ("lopsided", "disturbed"):
            _, ch, radius, k = meta
            level = [.04, .08, .14, .24].index(k)
            if kind == "lopsided":
                vals = np.array([.18, .50, -.24, -.20, -.16, -.05, -.04, .00, -.13, -.13])
                vals *= [1.0, 1.1, 1.25, 1.45][level]
                if radius >= 16:
                    vals[1] *= 1.35
            else:
                vals = np.array([.55, .22, -.24, -.20, -.16, .02, .04, .10, -.10, -.10])
                vals *= [1.0, 1.1, 1.28, 1.50][level]
                if radius <= 10:
                    vals[0] *= 1.15
            S[:, j] = vals / 15
        elif kind == "boxy":
            _, ch, radius, k = meta
            level = [.04, .09, .16, .26].index(k)
            vals = np.array([.08, .02, -.10, -.05, .04, .42, .02, .01, .04, .04])
            vals *= [1.0, 1.0, .9, .75][level]
            if radius > 16:
                vals[5] *= -.35
            S[:, j] = vals / 15
        elif kind == "spiral":
            _, ch, radius, beta, k = meta
            level = [.03, .07, .12, .20].index(k)
            channel_gain = {0: 1.0, 9: .75, 11: .90}[ch]
            vals = np.array([.12, .03, -.20, -.16, -.14, .42,
                             .70 if beta == 6 else .38,
                             .72 if beta == 3 else .35, -.12, -.12])
            vals *= channel_gain * [1.0, 1.08, 1.15, 1.20][level] / 18
            if radius <= 10:
                vals[5] *= 1.25
                vals[7] *= .70
            if radius >= 28:
                vals[7] *= 1.25
                vals[5] *= .65
            S[:, j] = vals
        elif kind in ("texture", "blue", "red", "ridge"):
            k = meta[-1]
            level = [.02, .05, .10, .18].index(k)
            if kind == "texture":
                vals = [.50, .20, -.42, -.34, -.30, .30, .40, .58, .06, -.14]
            elif kind == "blue":
                vals = [.34, .18, -.32, -.20, -.10, .20, .35, .55, .38, -.28]
            elif kind == "red":
                vals = [-.06, .04, .28, .18, .12, .02, -.02, -.20, -.12, .32]
            else:
                vals = [.05, .02, -.10, -.06, .12, .10, .04, .02, .60, .54]
            S[:, j] = np.asarray(vals) * [1.0, 1.0, .9, .75][level] / 16

    # Re-centre each rule row so a class is rewarded by its defining evidence
    # rather than by the number of positive hinges in that rule.  Fixed gains
    # reflect that smoothness is an absence of structure while disturbance and
    # spiral arms need positive evidence.
    row_l1 = np.abs(S).sum(axis=1, keepdims=True)
    S /= np.maximum(row_l1, 1e-6)
    gains = np.array([1.16, 1.05, 1.24, 1.18, 1.38,
                      .84, 1.05, 1.08, 1.28, 1.48], np.float32)
    S *= gains[:, None]
    # Reserve the rare disturbed/cigar decisions for their strongest generic
    # signatures instead of increasing those classes on every image.
    for j, meta in enumerate(concept_meta):
        if meta[0] == "disturbed" and meta[-1] >= .14:
            S[0, j] *= 2.3
        if meta[0] == "lopsided" and meta[-1] >= .14:
            S[0, j] *= 1.5
        if meta[0] == "elliptic" and meta[-1] >= .22:
            S[4, j] *= 1.8
        if meta[0] == "red" and meta[-1] >= .10:
            S[4, j] *= 1.5
    # A nearly featureless central source defaults to the smooth sequence.
    Sb = np.array([.0011, .0002, .0045, .0035, .0018,
                   .0003, .0004, .0005, .0003, .0006], np.float32)

    weights = {
        "phot.W": C.astype(np.float32), "phot.b": cb.astype(np.float32),
        "primitive.W": P.reshape(774, -1).astype(np.float32),
        "primitive.b": Pb,
        "concept.W": Q.astype(np.float32), "concept.b": Qb,
        "scores.W": S.astype(np.float32), "scores.b": Sb,
    }
    (out / "network.json").write_text(json.dumps({"layers": layers}, indent=2) + "\n")
    np.savez(out / "weights.npz", **weights)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
