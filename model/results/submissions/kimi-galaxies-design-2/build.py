"""Build an analytic galaxy-morphology network.  No input files are read.
All templates are generated from geometric galaxy models and a fixed seed.
"""
import sys, os, json
import numpy as np

N_TEMPLATE = 512
N_CLASS = 10
GRID = 32


def galaxy_template(cls, variant, rng):
    """A synthetic, seeing-softened galaxy on a unit-less 32 by 32 grid."""
    z = np.linspace(-1.0, 1.0, GRID, dtype=np.float64)
    xx, yy = np.meshgrid(z, z)
    # Sixteen evenly spaced position angles are used for every class.
    angle = np.pi * (variant % 16) / 16.0
    ca, sa = np.cos(angle), np.sin(angle)
    x = ca * xx + sa * yy
    y = -sa * xx + ca * yy
    rxy = np.sqrt(xx * xx + yy * yy)
    scale = float(np.exp(rng.uniform(np.log(0.055), np.log(0.34))))

    if cls in (2, 3, 4):
        q = rng.uniform(*({2: (0.78, 1.0), 3: (0.43, 0.82),
                           4: (0.25, 0.50)}[cls]))
        r = np.sqrt(x * x + (y / q) ** 2)
        # Smooth spheroids: a range between exponential and concentrated
        # de-Vaucouleurs-like profiles, plus a small unresolved core.
        exponent = rng.uniform(0.45, 1.0)
        im = np.exp(-np.maximum(r / scale, 0) ** exponent)
        core = rng.uniform(0.05, 0.55)
        im += core * np.exp(-0.5 * (r / (0.20 * scale + 0.012)) ** 2)

    elif cls in (8, 9):
        q = rng.uniform(0.040, 0.15 if cls == 8 else 0.24)
        # A thin, nearly exponential disk.  The separate round component is
        # present only in the edge-on-with-bulge model.
        disk = np.exp(-np.abs(x) / scale) * np.exp(-np.abs(y) / (q * scale + 0.008))
        disk *= np.exp(-(np.abs(x) / (2.8 * scale)) ** 6)
        im = disk
        if cls == 9:
            bulge_width = scale * rng.uniform(0.16, 0.38) + 0.010
            im += rng.uniform(0.30, 1.0) * np.exp(-0.5 * (rxy / bulge_width) ** 2)
        else:
            im += rng.uniform(0.0, 0.10) * np.exp(-0.5 * (rxy / (0.25 * scale + 0.01)) ** 2)

    elif cls in (5, 6, 7):
        q = rng.uniform(0.43 if cls == 5 else 0.36, 1.0)
        r = np.sqrt(x * x + (y / q) ** 2) + 1.e-8
        theta = np.arctan2(y / q, x)
        if cls == 6:
            winding = rng.uniform(3.6, 6.5)       # tightly wound log spiral
        elif cls == 7:
            winding = rng.uniform(1.35, 3.25)     # openly wound log spiral
        else:
            winding = rng.uniform(1.8, 4.8)
        arm_power = rng.uniform(2.0, 6.0)
        phase = 2.0 * (theta - winding * np.log((r + 0.025) / (scale + 0.025)))
        arms = (0.5 + 0.5 * np.cos(phase)) ** arm_power
        envelope = np.exp(-r / scale) * np.exp(-(r / (2.5 * scale)) ** 4)
        disk_light = rng.uniform(0.32, 0.68)
        im = envelope * (disk_light + (1.0 - disk_light) * arms)
        bulge = rng.uniform(0.15, 0.75)
        im += bulge * np.exp(-0.5 * (r / (0.30 * scale + 0.015)) ** 2)
        if cls == 5:
            # A rounded rectangular central bar, aligned with the two arms.
            bar_length = scale * rng.uniform(0.65, 1.25)
            bar_width = scale * rng.uniform(0.12, 0.25) + 0.008
            bar = np.exp(-(np.abs(x) / bar_length) ** 4 - 0.5 * (y / bar_width) ** 2)
            im += rng.uniform(0.55, 1.25) * bar

    elif cls == 0:
        # Disturbed systems: an asymmetric host, off-axis clumps and a short
        # warped tidal feature.  These are geometric random models, not data.
        q = rng.uniform(0.35, 1.0)
        r = np.sqrt(x * x + (y / q) ** 2) + 1.e-8
        im = rng.uniform(0.25, 0.8) * np.exp(-r / scale)
        im += rng.uniform(0.3, 1.0) * np.exp(-0.5 * (r / (0.28 * scale + 0.015)) ** 2)
        theta = np.arctan2(y / q, x)
        winding = rng.uniform(1.0, 5.0)
        arms = (0.5 + 0.5 * np.cos(2 * (theta - winding * np.log((r + 0.025) / (scale + 0.025))))) ** 3
        im += rng.uniform(0.05, 0.35) * arms * np.exp(-r / scale)
        for _ in range(rng.integers(1, 4)):
            a = rng.uniform(0, 2 * np.pi)
            d = scale * rng.uniform(0.45, 1.8)
            cx, cy = d * np.cos(a), d * np.sin(a)
            sx = scale * rng.uniform(0.18, 0.55) + 0.008
            sy = sx * rng.uniform(0.35, 1.0)
            im += rng.uniform(0.10, 0.50) * np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
        # Low-frequency asymmetry mimics a distorted, lopsided light profile.
        im *= 1.0 + rng.uniform(0.15, 0.55) * np.tanh(x / (scale + 0.02))

    else:
        # Mergers have two distinct concentrations with independent shapes,
        # usually joined by a low surface-brightness bridge or tidal tail.
        sep = scale * rng.uniform(1.0, 3.4) + 0.025
        phi = rng.uniform(0, 2 * np.pi)
        dx, dy = sep * np.cos(phi), sep * np.sin(phi)
        s2 = scale * rng.uniform(0.45, 1.1)
        im = np.zeros_like(x)
        for sign, ss in ((1.0, scale), (-1.0, s2)):
            cx, cy = sign * dx * 0.5, sign * dy * 0.5
            a = rng.uniform(0, np.pi)
            u = np.cos(a) * (x - cx) + np.sin(a) * (y - cy)
            v = -np.sin(a) * (x - cx) + np.cos(a) * (y - cy)
            q = rng.uniform(0.35, 1.0)
            rr = np.sqrt(u * u + (v / q) ** 2)
            im += rng.uniform(0.55, 1.1) * np.exp(-rr / (ss + 0.012))
            im += rng.uniform(0.15, 0.65) * np.exp(-0.5 * (rr / (0.25 * ss + 0.012)) ** 2)
        bridge_width = rng.uniform(0.015, 0.055)
        im += rng.uniform(0.04, 0.20) * np.exp(-0.5 * (rxy / (0.65 * sep + 0.01)) ** 2
                                                 -0.5 * ((-np.sin(phi) * x + np.cos(phi) * y) / bridge_width) ** 2)

    # The viewer's PSF/JPEG rendering and the network's square-root tone map
    # both soften the idealized profiles.  A tiny pedestal avoids zero templates.
    im = np.sqrt(np.maximum(im, 0.0) + 0.002)
    im *= np.exp(-(rxy / 1.45) ** 8)
    return im


def normalized_templates():
    rng = np.random.default_rng(731942)
    out = np.empty((N_CLASS * N_TEMPLATE, GRID * GRID), dtype=np.float64)
    for cls in range(N_CLASS):
        for j in range(N_TEMPLATE):
            t = galaxy_template(cls, j, rng)
            t = t - t.mean()
            t = t / np.sqrt(np.mean(t * t) + 1.e-12)
            # Input norm makes RMS approximately one.  Division by the number
            # of pixels therefore makes each dot product a cosine similarity.
            out[cls * N_TEMPLATE + j] = t.reshape(-1) / (GRID * GRID)
    return out.astype(np.float32)


def add_conv(layers, weights, name, inputs, W, b, stride=1, padding=0):
    W = np.asarray(W, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    k = W.shape[2]
    layers.append({"name": name, "op": "conv", "inputs": inputs,
                   "out": W.shape[0], "kernel": [k, k],
                   "stride": stride, "padding": padding})
    weights[name + ".W"] = W
    weights[name + ".b"] = b


def build(destination):
    layers, weights = [], {}
    # Luminance uses all three survey bands equally, avoiding an assumption
    # about galaxy colour.  Seven linear pieces approximate a square root.
    add_conv(layers, weights, "luminance", ["image"],
             np.full((1, 3, 1, 1), 1.0 / 3.0), [0.0])
    knots = np.array([0.0, 0.01, 0.03, 0.07, 0.15, 0.30, 0.60])
    slopes = np.diff(np.sqrt(np.r_[knots, 1.0])) / np.diff(np.r_[knots, 1.0])
    increments = np.r_[slopes[0], np.diff(slopes)]
    W = np.ones((len(knots), 1, 1, 1), dtype=np.float32)
    add_conv(layers, weights, "tone_knots", ["luminance"], W, -knots)
    layers.append({"name": "tone_relu", "op": "relu", "inputs": ["tone_knots"]})
    add_conv(layers, weights, "tone", ["tone_relu"],
             increments.reshape(1, -1, 1, 1), [0.0])

    # Central 64 pixels at two-pixel resolution, and the full field at
    # four-pixel resolution.  Both are normalized independently per image.
    layers.append({"name": "central_pool", "op": "avgpool", "inputs": ["tone"],
                   "size": [2, 2], "stride": [2, 2]})
    crop = np.zeros((1, 1, 33, 33), dtype=np.float32)
    crop[0, 0, 16, 16] = 1.0
    add_conv(layers, weights, "central_crop", ["central_pool"], crop, [0.0])
    layers.append({"name": "central_norm", "op": "norm", "inputs": ["central_crop"]})
    layers.append({"name": "full_pool", "op": "avgpool", "inputs": ["tone"],
                   "size": [4, 4], "stride": [4, 4]})
    layers.append({"name": "full_norm", "op": "norm", "inputs": ["full_pool"]})
    # A full-resolution 32-pixel cutout retains arms in the smallest galaxies.
    tight = np.zeros((1, 1, 97, 97), dtype=np.float32)
    tight[0, 0, 48, 48] = 1.0
    add_conv(layers, weights, "tight_crop", ["tone"], tight, [0.0])
    layers.append({"name": "tight_norm", "op": "norm", "inputs": ["tight_crop"]})

    templates = normalized_templates()
    for branch in ("tight", "central", "full"):
        layers.append({"name": branch + "_match", "op": "dense",
                       "inputs": [branch + "_norm"], "out": N_CLASS * N_TEMPLATE})
        weights[branch + "_match.W"] = templates.copy()
        weights[branch + "_match.b"] = np.zeros(N_CLASS * N_TEMPLATE, dtype=np.float32)
        previous = branch + "_match"
        n = N_CLASS * N_TEMPLATE
        layers.append({"name": branch + "_layout", "op": "reshape",
                       "inputs": [previous], "shape": [1, 1, n]})
        previous = branch + "_layout"
        stage = 0
        while n > N_CLASS:
            name = branch + "_max" + str(stage)
            layers.append({"name": name, "op": "maxpool", "inputs": [previous],
                           "size": [1, 2], "stride": [1, 2]})
            previous, n, stage = name, n // 2, stage + 1
        layers.append({"name": branch + "_scores", "op": "reshape",
                       "inputs": [previous], "shape": [N_CLASS]})

    # Rotation-invariant shape tests.  At four intensity levels, oriented
    # second moments are max/min pooled over sixteen angles.  Their ratio is
    # the squared minor/major axis ratio, with no fitted ellipse parameters.
    mtW = np.ones((4, 1, 1, 1), dtype=np.float32)
    add_conv(layers, weights, "moment_thresholds", ["central_norm"],
             mtW, [0.0, -0.5, -1.0, -1.5])
    layers.append({"name": "moment_positive", "op": "relu", "inputs": ["moment_thresholds"]})
    z = np.linspace(-1.0, 1.0, GRID)
    mx, my = np.meshgrid(z, z)
    aperture_m = np.exp(-0.5 * (mx * mx + my * my) / 0.48 ** 2)
    aperture_m /= aperture_m.sum()
    maxW = np.zeros((64, 4, GRID, GRID), dtype=np.float32)
    for t in range(4):
        for j in range(16):
            a = np.pi * j / 16.0
            u = np.cos(a) * mx + np.sin(a) * my
            maxW[16 * t + j, t] = aperture_m * u * u
    for name, sign in (("moment_max", 1.0), ("moment_min", -1.0)):
        layers.append({"name": name, "op": "dense", "inputs": ["moment_positive"], "out": 64})
        weights[name + ".W"] = (sign * maxW).reshape(64, -1)
        weights[name + ".b"] = np.zeros(64, dtype=np.float32)
        layers.append({"name": name + "_layout", "op": "reshape", "inputs": [name], "shape": [1, 1, 64]})
        previous = name + "_layout"
        for stage in range(4):
            outname = name + "_pool" + str(stage)
            layers.append({"name": outname, "op": "maxpool", "inputs": [previous],
                           "size": [1, 2], "stride": [1, 2]})
            previous = outname
        layers.append({"name": name + "_flat", "op": "reshape", "inputs": [previous], "shape": [4]})

    # Six signed ratio tests: thin outer light, thin core, round core,
    # round outer light, elongated core, and a second thin-disk test.
    A = np.zeros((6, 4), dtype=np.float32)
    B = np.zeros((6, 4), dtype=np.float32)
    A[0, 0], B[0, 0] = 0.07, -1.0       # outer axis ratio below sqrt(.07)
    A[1, 2], B[1, 2] = 0.09, -1.0       # core remains thin
    A[2, 2], B[2, 2] = -0.09, 1.0       # core is at least moderately round
    A[3, 0], B[3, 0] = -0.25, 1.0       # outer light is round
    A[4, 2], B[4, 2] = 0.20, -1.0       # elongated core: a possible bar
    A[5, 1], B[5, 1] = 0.07, -1.0       # thin at an intermediate level
    # moment_min_flat is -min(moment), hence reverse all of its coefficients.
    B = -B
    for source, mat, name in (("moment_max_flat", A, "ratio_from_max"),
                              ("moment_min_flat", B, "ratio_from_min")):
        layers.append({"name": name, "op": "dense", "inputs": [source], "out": 6})
        weights[name + ".W"] = mat
        weights[name + ".b"] = np.zeros(6, dtype=np.float32)
    layers.append({"name": "ratio_differences", "op": "add", "inputs": ["ratio_from_max", "ratio_from_min"]})
    layers.append({"name": "ratio_caps", "op": "dense", "inputs": ["ratio_differences"], "out": 12})
    weights["ratio_caps.W"] = np.r_[np.eye(6, dtype=np.float32), np.eye(6, dtype=np.float32)]
    cap = 0.002
    weights["ratio_caps.b"] = np.r_[np.zeros(6), np.full(6, -cap)].astype(np.float32)
    layers.append({"name": "ratio_relu", "op": "relu", "inputs": ["ratio_caps"]})
    layers.append({"name": "shape_evidence", "op": "dense", "inputs": ["ratio_relu"], "out": 6})
    weights["shape_evidence.W"] = np.c_[np.eye(6), -np.eye(6)].astype(np.float32) / cap
    weights["shape_evidence.b"] = np.zeros(6, dtype=np.float32)
    shape_rules = np.zeros((10, 6), dtype=np.float32)
    shape_rules[8] = [1.5, 1.0, 0.0, 0.0, 0.0, 0.5]
    shape_rules[9] = [1.5, 0.0, 1.0, 0.0, 0.0, 0.5]
    shape_rules[4] = [-2.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    # The high-threshold bar test is too sensitive to sparse background noise;
    # leave barred-spiral recognition to the explicit bar templates above.
    shape_rules[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    shape_rules[3, 0] = -0.5
    shape_rules[2, 0] = -0.5
    layers.append({"name": "shape_scores", "op": "dense", "inputs": ["shape_evidence"], "out": 10})
    weights["shape_scores.W"] = shape_rules
    weights["shape_scores.b"] = np.zeros(10, dtype=np.float32)

    # Spiral and disturbed systems generally have bluer outer light than
    # smooth spheroids.  This deliberately small chromatic term uses a fixed
    # Gaussian aperture and the g-minus-z colour, not a fitted statistic.
    layers.append({"name": "colour_pool", "op": "avgpool", "inputs": ["image"],
                   "size": [4, 4], "stride": [4, 4]})
    z = np.linspace(-1.0, 1.0, GRID)
    gx, gy = np.meshgrid(z, z)
    aperture = np.exp(-0.5 * (gx * gx + gy * gy) / 0.32 ** 2)
    aperture /= aperture.sum()
    blue_weight = np.array([5.0, 1.0, -5.0, -4.0, -5.0,
                            1.5, 3.5, 5.0, 2.0, -2.0])
    colour_W = np.zeros((10, 3, GRID, GRID), dtype=np.float32)
    for cls in range(10):
        colour_W[cls, 0] = -blue_weight[cls] * aperture
        colour_W[cls, 2] = blue_weight[cls] * aperture
    layers.append({"name": "colour_scores", "op": "dense", "inputs": ["colour_pool"], "out": 10})
    weights["colour_scores.W"] = colour_W.reshape(10, -1)
    weights["colour_scores.b"] = np.zeros(10, dtype=np.float32)
    layers.append({"name": "scores", "op": "add",
                   "inputs": ["tight_scores", "central_scores", "full_scores", "colour_scores", "shape_scores"]})
    os.makedirs(destination, exist_ok=True)
    with open(os.path.join(destination, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, separators=(",", ":"))
    np.savez(os.path.join(destination, "weights.npz"), **weights)


if __name__ == "__main__":
    build(sys.argv[1])
