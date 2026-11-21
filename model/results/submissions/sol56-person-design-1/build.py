#!/usr/bin/env python3
"""Build a data-free, hand-designed person detector from analytic filters and shapes."""
import json
import math
import sys
from pathlib import Path

import numpy as np


def layer(name, op, inp, **settings):
    d = {"name": name, "op": op, "inputs": [inp] if isinstance(inp, str) else inp}
    d.update(settings)
    return d


def triangle_terms(a, b, c):
    """Coefficients of a triangular membership function made from ReLU hinges."""
    return [(a, 1.0 / (b - a)),
            (b, -1.0 / (b - a) - 1.0 / (c - b)),
            (c, 1.0 / (c - b))]


def capsule(X, Y, x0, y0, x1, y1, radius):
    vx, vy = x1 - x0, y1 - y0
    t = np.clip(((X - x0) * vx + (Y - y0) * vy) / (vx * vx + vy * vy + 1e-12), 0, 1)
    return (X - (x0 + t * vx)) ** 2 + (Y - (y0 + t * vy)) ** 2 <= radius ** 2


def person_mask(k, height, pose, bust=False, supersample=5):
    """Rasterize a generic person assembled from a head, torso and limb capsules."""
    q = supersample
    yy, xx = np.mgrid[0:k*q, 0:k*q]
    cx = k / 2.0
    top = (k - height) / 2.0
    X = (xx + 0.5) / q
    Y = ((yy + 0.5) / q - top) / height
    X = (X - cx) / height

    if bust:
        # A large head over a widening shoulder/upper-torso silhouette.
        head_y, head_r = 0.27, 0.175
        occ = (X / head_r) ** 2 + ((Y - head_y) / (head_r * 1.20)) ** 2 <= 1
        occ |= capsule(X, Y, -0.10, 0.53, -0.25, 0.86, 0.14)
        occ |= capsule(X, Y,  0.10, 0.53,  0.25, 0.86, 0.14)
        occ |= (Y >= 0.47) & (Y <= 1.03) & (np.abs(X) <= (0.11 + 0.24 * (Y - 0.47)))
    else:
        head_y, head_r = 0.105, 0.073
        occ = (X / head_r) ** 2 + ((Y - head_y) / (head_r * 1.08)) ** 2 <= 1
        # Neck and gently tapered torso.
        occ |= (Y >= 0.18) & (Y <= 0.56) & (np.abs(X) <= (0.115 - 0.045 * (Y - 0.18)))
        occ |= capsule(X, Y, -0.065, 0.54, -0.075, 0.93, 0.047)
        occ |= capsule(X, Y,  0.065, 0.54,  0.075, 0.93, 0.047)
        if pose == "walk":
            occ |= capsule(X, Y, -0.065, 0.54, -0.16, 0.94, 0.044)
            occ |= capsule(X, Y,  0.065, 0.54,  0.18, 0.92, 0.044)
            arm_ends = ((-0.18, 0.54), (0.18, 0.45))
        elif pose == "wide":
            arm_ends = ((-0.34, 0.42), (0.34, 0.42))
        elif pose == "up":
            arm_ends = ((-0.25, 0.04), (0.25, 0.04))
        else:
            arm_ends = ((-0.16, 0.58), (0.16, 0.58))
        occ |= capsule(X, Y, -0.10, 0.25, arm_ends[0][0], arm_ends[0][1], 0.038)
        occ |= capsule(X, Y,  0.10, 0.25, arm_ends[1][0], arm_ends[1][1], 0.038)

    # Area-average the supersampled silhouette into HOG cells.
    return occ.reshape(k, q, k, q).mean((1, 3)).astype(np.float32)


def hog_template(mask, bins=8):
    """Turn the boundary normals of a synthetic silhouette into an unsigned HOG template."""
    gy, gx = np.gradient(mask)
    mag = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), math.pi) * bins / math.pi
    W = np.zeros((bins,) + mask.shape, np.float32)
    lo = np.floor(angle).astype(int) % bins
    frac = angle - np.floor(angle)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            m = float(mag[y, x])
            if m > 1e-5:
                W[lo[y, x], y, x] += m * (1 - frac[y, x])
                W[(lo[y, x] + 1) % bins, y, x] += m * frac[y, x]
    s = float(W.sum())
    if s:
        W /= s
    # At silhouette boundaries, incompatible orientations are negative evidence.
    # Subtracting one eighth from every bin makes a random edge orientation
    # approximately neutral while retaining a strong correctly aligned normal.
    boundary = mag / (float(mag.sum()) + 1e-8)
    W -= 1.08 * boundary[None, :, :] / bins
    return W


def face_gray_template(k, rx, strength=0.72):
    """A generic frontal face: oval cheeks, hair cap, eye pair and mouth."""
    y, x = np.mgrid[0:k, 0:k].astype(np.float32)
    cx, cy, ry = (k - 1) / 2, (k - 1) / 2, 1.28 * rx
    xn, yn = (x - cx) / rx, (y - cy) / ry
    oval = np.exp(-1.8 * (xn*xn + yn*yn))
    ring = np.exp(-1.5 * (xn*xn + yn*yn)) - np.exp(-3.7 * (xn*xn + yn*yn))
    sig = max(0.38, 0.20 * rx)
    eye_y = cy - 0.18 * ry
    eyes = (np.exp(-((x-(cx-0.38*rx))**2 + (y-eye_y)**2)/(2*sig*sig)) +
            np.exp(-((x-(cx+0.38*rx))**2 + (y-eye_y)**2)/(2*sig*sig)))
    mouth = np.exp(-((x-cx)**2/(2*(0.38*rx)**2 + 1e-6) +
                     (y-(cy+0.34*ry))**2/(2*(0.13*ry)**2 + 1e-6)))
    hair = np.exp(-((x-cx)**2/(2*(0.66*rx)**2 + 1e-6) +
                    (y-(cy-0.72*ry))**2/(2*(0.24*ry)**2 + 1e-6)))
    t = 0.62*oval - 0.55*eyes - 0.28*mouth - 0.42*hair - 0.10*ring
    # Make the filter insensitive to local constant brightness.
    support = np.exp(-0.75 * (xn*xn + yn*yn))
    t -= support * (t.sum() / (support.sum() + 1e-8))
    t *= strength / (np.abs(t).sum() + 1e-8)
    return t.astype(np.float32)


def face_skin_template(k, rx, channels=3, strength=0.78):
    """Reward a compact elliptical skin region and weakly reject its surround."""
    y, x = np.mgrid[0:k, 0:k].astype(np.float32)
    cx, cy, ry = (k - 1) / 2, (k - 1) / 2, 1.28 * rx
    d = ((x-cx)/rx)**2 + ((y-cy)/ry)**2
    inside = np.exp(-1.1*d)
    surround = np.exp(-0.32*d) - np.exp(-0.90*d)
    t = inside/(inside.sum()+1e-8) - 0.22*surround/(surround.sum()+1e-8)
    out = np.repeat(t[None, :, :], channels, axis=0)
    out *= strength / channels
    return out.astype(np.float32)


def build(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    layers, weights = [], {}

    # Luminance, standardized per photograph.
    layers.append(layer("gray", "conv", "image", out=1, kernel=[1, 1], stride=1, padding=0))
    weights["gray.W"] = np.array([[[[0.299]], [[0.587]], [[0.114]]]], np.float32)
    weights["gray.b"] = np.zeros(1, np.float32)
    layers.append(layer("gray_norm", "norm", "gray"))

    # Eight unsigned, contrast-normalized derivative-of-Gaussian edge channels.
    kgrad = 5
    yy, xx = np.mgrid[-2:3, -2:3].astype(np.float32)
    gauss = np.exp(-(xx*xx + yy*yy)/(2*1.05**2))
    dx, dy = -xx*gauss, -yy*gauss
    gradW = []
    for j in range(8):
        th = j * math.pi / 8
        f = math.cos(th)*dx + math.sin(th)*dy
        f -= f.mean(); f /= np.sqrt((f*f).sum()) + 1e-8
        gradW.extend([f, -f])
    layers.append(layer("grad", "conv", "gray_norm", out=16, kernel=[5, 5], stride=1, padding=2))
    weights["grad.W"] = np.asarray(gradW, np.float32)[:, None, :, :]
    weights["grad.b"] = np.zeros(16, np.float32)
    layers.append(layer("grad_half", "relu", "grad"))
    layers.append(layer("grad_norm", "norm", "grad_half"))
    layers.append(layer("grad_salient", "relu", "grad_norm"))
    layers.append(layer("orient", "conv", "grad_salient", out=8, kernel=[1, 1], stride=1, padding=0))
    ow = np.zeros((8, 16, 1, 1), np.float32)
    for j in range(8): ow[j, 2*j:2*j+2, 0, 0] = 0.5
    weights["orient.W"], weights["orient.b"] = ow, np.zeros(8, np.float32)
    layers.append(layer("hog_cells", "avgpool", "orient", size=[4, 4], stride=[4, 4]))

    # Generic full-body and head/shoulder silhouettes at several scales and poses.
    body_templates = []
    for h in (5.2, 6.8, 8.6, 10.8, 13.2):
        for pose in ("down", "walk", "wide", "up"):
            body_templates.append(hog_template(person_mask(15, h, pose)))
    for h in (6.0, 8.0, 10.5, 13.0):
        body_templates.append(hog_template(person_mask(15, h, "down", bust=True)))
    bodyW = np.asarray(body_templates, np.float32)
    layers.append(layer("body_raw", "conv", "hog_cells", out=len(bodyW), kernel=[15, 15], stride=1, padding=0))
    weights["body_raw.W"] = bodyW
    weights["body_raw.b"] = np.full(len(bodyW), -0.050, np.float32)
    layers.append(layer("body_relu", "relu", "body_raw"))
    # Fold detector channels into a spatial dimension, then take one true global maximum.
    bh = len(bodyW) * 10
    layers.append(layer("body_fold", "reshape", "body_relu", shape=[1, bh, 10]))
    layers.append(layer("body_max", "maxpool", "body_fold", size=[bh, 10], stride=[bh, 10]))

    # Piecewise-linear skin-colour memberships in opponent-colour coordinates.
    profiles = [
        ((-0.06, 0.055, 0.25), (-0.07, 0.045, 0.22)),
        ((-0.07, 0.025, 0.16), (-0.08, 0.020, 0.14)),
        ((-0.01, 0.120, 0.34), (-0.04, 0.075, 0.25)),
    ]
    specs = []
    # Each hinge is represented by (linear RGB vector, threshold).
    for up, vp in profiles:
        for t, _ in triangle_terms(*up): specs.append(((1., -1., 0.), t))
        for t, _ in triangle_terms(*vp): specs.append(((0., 1., -1.), t))
    specs += [((0.299, 0.587, 0.114), 0.07), ((0.299, 0.587, 0.114), 0.24)]
    unique = []
    for s in specs:
        key = tuple(round(v, 7) for v in s[0]) + (round(s[1], 7),)
        if key not in [u[0] for u in unique]: unique.append((key, s))
    specs = [u[1] for u in unique]
    index = {(tuple(v), t): i for i, (v, t) in enumerate(specs)}
    hw = np.zeros((len(specs), 3, 1, 1), np.float32)
    hb = np.zeros(len(specs), np.float32)
    for i, (v, t) in enumerate(specs): hw[i, :, 0, 0], hb[i] = v, -t
    layers.append(layer("color_hinges", "conv", "image", out=len(specs), kernel=[1, 1], stride=1, padding=0))
    weights["color_hinges.W"], weights["color_hinges.b"] = hw, hb
    layers.append(layer("color_relu", "relu", "color_hinges"))
    sw = np.zeros((len(profiles), len(specs), 1, 1), np.float32)
    sb = np.full(len(profiles), -1.78, np.float32)
    for p, (up, vp) in enumerate(profiles):
        for t, coef in triangle_terms(*up): sw[p, index[((1., -1., 0.), t)], 0, 0] += coef
        for t, coef in triangle_terms(*vp): sw[p, index[((0., 1., -1.), t)], 0, 0] += coef
        # Brightness ramp: zero at .07 and saturated at .24.
        sw[p, index[((0.299, 0.587, 0.114), 0.07)], 0, 0] += 1/.17
        sw[p, index[((0.299, 0.587, 0.114), 0.24)], 0, 0] -= 1/.17
    layers.append(layer("skin_raw", "conv", "color_relu", out=len(profiles), kernel=[1, 1], stride=1, padding=0))
    weights["skin_raw.W"], weights["skin_raw.b"] = sw, sb
    layers.append(layer("skin", "relu", "skin_raw"))

    # Coincident colour and luminance face templates, scanned every three pixels.
    layers.append(layer("gray_32", "avgpool", "gray_norm", size=[3, 3], stride=[3, 3]))
    layers.append(layer("skin_32", "avgpool", "skin", size=[3, 3], stride=[3, 3]))
    radii = (1.15, 1.55, 2.05, 2.70, 3.55, 4.55, 5.60)
    fg, fs = [], []
    for r in radii:
        # First a colour-assisted face, then a monochrome-friendly structural face.
        fg.append(face_gray_template(15, r, 0.72))
        fs.append(face_skin_template(15, r, len(profiles), 0.82))
        fg.append(face_gray_template(15, r, 1.00))
        fs.append(np.zeros((len(profiles), 15, 15), np.float32))
    fg, fs = np.asarray(fg)[:, None, :, :], np.asarray(fs)
    layers.append(layer("face_gray", "conv", "gray_32", out=len(fg), kernel=[15, 15], stride=1, padding=0))
    layers.append(layer("face_skin", "conv", "skin_32", out=len(fs), kernel=[15, 15], stride=1, padding=0))
    weights["face_gray.W"], weights["face_gray.b"] = fg, np.zeros(len(fg), np.float32)
    weights["face_skin.W"], weights["face_skin.b"] = fs, np.tile(np.array([-0.205, -0.285], np.float32), len(radii))
    layers.append(layer("face_sum", "add", ["face_gray", "face_skin"]))
    layers.append(layer("face_relu", "relu", "face_sum"))
    fh, fw = len(fg)*18, 18
    layers.append(layer("face_fold", "reshape", "face_relu", shape=[1, fh, fw]))
    layers.append(layer("face_max", "maxpool", "face_fold", size=[fh, fw], stride=[fh, fw]))

    # A weak crowd cue: global amount of plausible skin, capped indirectly by low weight.
    layers.append(layer("skin_global", "avgpool", "skin", size=[96, 96], stride=[96, 96]))

    # Hand-set evidence fusion. Output 1 is person.
    layers.append(layer("body_vote", "dense", "body_max", out=2))
    weights["body_vote.W"] = np.array([[0.0], [0.85]], np.float32)
    weights["body_vote.b"] = np.array([0.80, 0.0], np.float32)
    layers.append(layer("face_vote", "dense", "face_max", out=2))
    weights["face_vote.W"] = np.array([[0.0], [1.0]], np.float32)
    weights["face_vote.b"] = np.zeros(2, np.float32)
    layers.append(layer("crowd_vote", "dense", "skin_global", out=2))
    weights["crowd_vote.W"] = np.array([[0., 0., 0.], [0., 0., 0.]], np.float32)
    weights["crowd_vote.b"] = np.zeros(2, np.float32)
    layers.append(layer("scores", "add", ["body_vote", "face_vote", "crowd_vote"]))

    net = {"layers": layers}
    (dest / "network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(dest / "weights.npz", **{k: np.asarray(v, np.float32) for k, v in weights.items()})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: build.py OUTPUT_FOLDER")
    build(sys.argv[1])
