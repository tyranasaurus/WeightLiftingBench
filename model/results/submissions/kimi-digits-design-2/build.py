#!/usr/bin/env python3
"""Build a data-free handwritten-digit recognizer from typed pen models.

All templates are drawn from the parametric strokes below.  No data files are
read.  The network compares normalized, blurred images with every drawn model
at two resolutions and takes the best matching model in each class.
"""
import json
import sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
MODELS = 384
SIZE = 28


def cubic(p0, p1, p2, p3, n=18):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in (p0, p1, p2, p3)]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*p3


def ellipse(cx, cy, rx, ry, n=72, wobble=0.0, phase=0.0):
    t = np.linspace(0, 2*np.pi, n)
    x = cx + rx*np.cos(t)*(1 + wobble*np.sin(3*t+phase))
    y = cy + ry*np.sin(t)*(1 + wobble*np.cos(2*t-phase))
    return np.stack([x, y], axis=1)


def line(*points):
    return np.asarray(points, dtype=float)


def paths_for(digit, rng):
    """Return pen paths in a ten by eighteen personal-coordinate frame."""
    style = rng.integers(4)
    if digit == 0:
        rx = rng.uniform(3.5, 4.6)
        ry = rng.uniform(7.4, 8.7)
        p = ellipse(5, 9, rx, ry, wobble=rng.uniform(0, .035),
                    phase=rng.uniform(0, 6.28))
        paths = [p]
        # A rare handwritten slashed zero, without making it a separate style.
        if rng.random() < .015:
            paths.append(line((2.3, 14.5), (7.8, 3.5)))
    elif digit == 1:
        topx = rng.uniform(3.8, 6.2)
        botx = topx + rng.uniform(-1.0, 1.0)
        main = cubic((topx, 1), (topx+.2, 6), (botx-.2, 12), (botx, 17.3), 24)
        paths = [main]
        if style != 0:
            paths.append(line((topx-rng.uniform(2.0, 3.5), rng.uniform(3.0, 5.0)),
                              (topx, 1)))
        if style == 3 or rng.random() < .12:
            paths.append(line((botx-rng.uniform(2.0, 3.5), 17.3),
                              (botx+rng.uniform(2.0, 3.5), 17.3)))
    elif digit == 2:
        left = rng.uniform(.5, 1.7)
        right = rng.uniform(8.3, 9.5)
        top = cubic((left, 5.2), (left-.2, .5), (right+.2, -.3), (right, 5.2), 24)
        diag = cubic((right, 5.2), (right-.2, 9.2), (4.0, 11.7),
                     (rng.uniform(.7, 1.8), 16.0), 24)[1:]
        base = cubic(diag[-1], (3.5, 17.1), (6.5, 16.5),
                     (rng.uniform(8.0, 9.7), rng.uniform(16.3, 17.4)), 12)[1:]
        paths = [np.concatenate([top, diag, base])]
        if style == 3:
            # A small returning foot is common in quick twos.
            paths = [np.concatenate([top, diag[:-5],
                     cubic(diag[-5], (0, 17.8), (3, 18.2), (4.2, 16.7), 10)[1:],
                     line((4.2, 16.7), (9.2, 16.7))])]
    elif digit == 3:
        if style == 3:
            upper = np.concatenate([
                line((.8, 1.4), (8.8, 1.4)),
                cubic((8.8, 1.4), (7.7, 4.0), (6.2, 6.5), (4.8, 8.3), 14)[1:]])
        else:
            upper = cubic((rng.uniform(.5, 1.7), rng.uniform(2.0, 3.8)),
                          (3.8, -.5), (10.0, .1), (8.4, 5.0), 20)
            mid = cubic((8.4, 5.0), (7.6, 7.0), (6.0, 7.8),
                        (rng.uniform(4.4, 5.8), 8.4), 12)[1:]
            upper = np.concatenate([upper, mid])
        lower = cubic(upper[-1], (10.8, 7.4), (11.0, 15.8), (6.0, 17.0), 24)[1:]
        end = cubic((6.0, 17.0), (3.0, 18.6), (.8, 17.0),
                    (rng.uniform(.3, 1.4), rng.uniform(14.0, 15.8)), 14)[1:]
        paths = [np.concatenate([upper, lower, end])]
    elif digit == 4:
        rightx = rng.uniform(6.0, 7.8)
        cross_y = rng.uniform(10.0, 12.2)
        if style < 2:
            # Triangular, closed four.
            left = cubic((rightx, .7), (rightx-1.7, 4.0),
                         (2.4, 8.0), (rng.uniform(.6, 1.5), cross_y), 18)
            paths = [np.concatenate([left, line(left[-1], (9.4, cross_y))[1:]]),
                     cubic((rightx, .7), (rightx+.2, 6),
                           (rightx-.2, 12), (rightx, 17.7), 22)]
        else:
            # Open four: two uprights joined by a low cross stroke.
            leftx = rng.uniform(1.2, 3.0)
            paths = [cubic((leftx, rng.uniform(.5, 2.0)), (leftx, 4),
                           (leftx-.3, 8), (leftx, cross_y), 16),
                     cubic((leftx, cross_y), (4, cross_y+.3),
                           (7, cross_y-.2), (9.2, cross_y), 12),
                     cubic((rightx, .5), (rightx+.2, 6),
                           (rightx-.2, 12), (rightx, 17.8), 22)]
    elif digit == 5:
        top_y = rng.uniform(.7, 1.8)
        leftx = rng.uniform(1.2, 2.5)
        upper = np.concatenate([line((9.0, top_y), (leftx, top_y)),
                                cubic((leftx, top_y), (leftx-.2, 3.3),
                                      (leftx-.3, 5.7), (leftx, 8.0), 12)[1:]])
        bowl = cubic((leftx, 8.0), (4.0, 5.4), (9.8, 6.0), (9.0, 11.0), 20)[1:]
        lower = cubic((9.0, 11.0), (9.0, 17.0), (3.0, 19.5),
                      (rng.uniform(.5, 1.5), rng.uniform(14.5, 16.0)), 24)[1:]
        paths = [np.concatenate([upper, bowl, lower])]
        if style == 3:
            # Slightly detached cap, as in a two-stroke five.
            paths = [line((8.9, top_y), (leftx, top_y)),
                     np.concatenate([cubic((leftx, top_y), (leftx, 3),
                                           (leftx, 6), (leftx, 8), 10), bowl, lower])]
    elif digit == 6:
        top = cubic((rng.uniform(7.0, 8.8), rng.uniform(.7, 2.0)),
                    (3.0, .0), (.6, 4.0), (1.1, 10.8), 28)
        loop = cubic((1.1, 10.8), (.4, 18.5), (8.0, 20.0), (9.0, 13.0), 26)[1:]
        close = cubic((9.0, 13.0), (10.0, 7.0), (3.0, 6.5), (1.1, 10.8), 24)[1:]
        paths = [np.concatenate([top, loop, close])]
    elif digit == 7:
        top_y = rng.uniform(.7, 1.8)
        startx = rng.uniform(.5, 1.8)
        rightx = rng.uniform(8.2, 9.5)
        endx = rng.uniform(2.0, 4.5)
        top = cubic((startx, top_y), (4, top_y-.3),
                    (6, top_y+.3), (rightx, top_y), 12)
        diag = cubic((rightx, top_y), (rightx-1.0, 5.5),
                     (endx+1.0, 12.0), (endx, 17.8), 26)[1:]
        paths = [np.concatenate([top, diag])]
        if style == 0 or rng.random() < .16:
            paths.append(line((rng.uniform(2.5, 3.8), rng.uniform(8.0, 10.0)),
                              (rng.uniform(6.5, 7.8), rng.uniform(8.0, 10.0))))
        if style == 3:
            paths.append(line((startx, top_y), (startx-.2, top_y+2.4)))
    elif digit == 8:
        split = rng.uniform(8.0, 9.7)
        upper_ry = split/2
        lower_ry = (18-split)/2
        upper_rx = rng.uniform(3.2, 4.2)
        lower_rx = rng.uniform(3.6, 4.7)
        paths = [ellipse(5, upper_ry+.2, upper_rx, upper_ry,
                         wobble=rng.uniform(0, .025), phase=rng.uniform(0, 6.28)),
                 ellipse(5, split+lower_ry-.2, lower_rx, lower_ry,
                         wobble=rng.uniform(0, .025), phase=rng.uniform(0, 6.28))]
    else:
        cx = rng.uniform(4.7, 5.3)
        rx = rng.uniform(3.4, 4.4)
        ry = rng.uniform(4.1, 5.0)
        paths = [ellipse(cx, ry+.5, rx, ry, wobble=rng.uniform(0, .025),
                         phase=rng.uniform(0, 6.28))]
        startx = cx+rx
        if style == 3:
            tail = cubic((startx, ry+1.0), (startx+.2, 8.5),
                         (startx-1.0, 14.0), (rng.uniform(3.0, 5.5), 17.8), 26)
        else:
            tail = cubic((startx, ry+1.0), (startx+.1, 10.5),
                         (rng.uniform(4.0, 6.5), 16.8),
                         (rng.uniform(1.2, 3.3), rng.uniform(16.5, 18.0)), 28)
        paths.append(tail)
    return paths


def render(digit, rng):
    paths = paths_for(digit, rng)
    # Personal slant, rotation, and proportions; all variations are generated,
    # never measured from an example.
    shear = rng.uniform(-.20, .20)
    angle = rng.uniform(-.10, .10)
    ca, sa = np.cos(angle), np.sin(angle)
    aspect = rng.uniform(.82, 1.14)
    warped = []
    for p in paths:
        q = p.copy()
        q[:, 0] = (q[:, 0]-5)*aspect + 5 + shear*(9-q[:, 1])
        x, y = q[:, 0]-5, q[:, 1]-9
        q[:, 0] = ca*x-sa*y+5
        q[:, 1] = sa*x+ca*y+9
        # A gentle non-rigid personal hand movement.
        q[:, 0] += rng.uniform(-.15, .15)*np.sin(q[:, 1]*.45+rng.uniform(0, 6.28))
        warped.append(q)
    paths = warped
    radius = rng.uniform(1.05, 1.75)
    # Fit the drawn bounding box to a twenty-pixel box, preserving proportions.
    allp = np.concatenate(paths)
    lo = allp.min(0)-radius
    hi = allp.max(0)+radius
    scale = 19.6/max(hi-lo)
    paths = [(p-(lo+hi)/2)*scale for p in paths]
    radius *= scale
    image = np.zeros((SIZE, SIZE), dtype=float)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    for path in paths:
        r = radius*rng.uniform(.91, 1.09)
        for a, b in zip(path[:-1], path[1:]):
            a = a+13.5
            b = b+13.5
            margin = r+1.2
            x0 = max(0, int(np.floor(min(a[0], b[0])-margin)))
            x1 = min(SIZE, int(np.ceil(max(a[0], b[0])+margin))+1)
            y0 = max(0, int(np.floor(min(a[1], b[1])-margin)))
            y1 = min(SIZE, int(np.ceil(max(a[1], b[1])+margin))+1)
            if x0 >= x1 or y0 >= y1:
                continue
            vx, vy = b-a
            den = vx*vx+vy*vy
            px = xx[y0:y1, x0:x1]-a[0]
            py = yy[y0:y1, x0:x1]-a[1]
            t = np.clip((px*vx+py*vy)/max(den, 1e-12), 0, 1)
            dist = np.sqrt((px-t*vx)**2+(py-t*vy)**2)
            ink = np.clip(r+.55-dist, 0, 1)
            image[y0:y1, x0:x1] = np.maximum(image[y0:y1, x0:x1], ink)
    # MNIST-style centering uses ink centre of mass, not the bounding box.
    mass = image.sum()
    if mass > 0:
        cy = float((image*yy).sum()/mass)
        cx = float((image*xx).sum()/mass)
        shifted = np.zeros_like(image)
        # Sub-pixel translation by bilinear splatting from source to target.
        dx, dy = 13.5-cx, 13.5-cy
        tx, ty = xx+dx, yy+dy
        x0 = np.floor(tx).astype(int); y0 = np.floor(ty).astype(int)
        fx, fy = tx-x0, ty-y0
        valid = (x0 >= 0) & (x0 < SIZE-1) & (y0 >= 0) & (y0 < SIZE-1)
        for oy, ox, weight in ((0, 0, (1-fy)*(1-fx)), (0, 1, (1-fy)*fx),
                               (1, 0, fy*(1-fx)), (1, 1, fy*fx)):
            target_y = y0+oy
            target_x = x0+ox
            np.add.at(shifted, (target_y[valid], target_x[valid]),
                      (image*weight)[valid])
        image = shifted
    return image


def gaussian28(images, sigma):
    radius = 2
    v = np.arange(-radius, radius+1)
    k = np.exp(-(v*v)/(2*sigma*sigma)); k /= k.sum()
    kernel = k[:, None]*k[None, :]
    padded = np.pad(images, ((0, 0), (radius, radius), (radius, radius)))
    result = np.zeros_like(images)
    for dy in range(5):
        for dx in range(5):
            result += kernel[dy, dx]*padded[:, dy:dy+SIZE, dx:dx+SIZE]
    return result, kernel


def convolve28(images, kernels):
    """Small zero-padded 3x3 convolution bank used for template features."""
    padded = np.pad(images, ((0, 0), (1, 1), (1, 1)))
    result = np.zeros((len(images), len(kernels), SIZE, SIZE), dtype=float)
    for o, kernel in enumerate(kernels):
        for dy in range(3):
            for dx in range(3):
                result[:, o] += kernel[dy, dx]*padded[:, dy:dy+SIZE, dx:dx+SIZE]
    return result


def hog_features(soft):
    gx_kernel = np.array([[0, 0, 0], [-.5, 0, .5], [0, 0, 0]])
    gy_kernel = np.array([[0, -.5, 0], [0, 0, 0], [0, .5, 0]])
    gradients = convolve28(soft, [gx_kernel, gy_kernel])
    angles = np.arange(8)*np.pi/8
    responses = []
    for angle in angles:
        response = np.abs(np.cos(angle)*gradients[:, 0] +
                          np.sin(angle)*gradients[:, 1])
        response = response.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4))
        response -= response.mean(axis=(1, 2), keepdims=True)
        response /= np.sqrt(np.mean(response*response, axis=(1, 2), keepdims=True)+1e-5)
        responses.append(response)
    maps = np.stack(responses, axis=1)
    return maps, gx_kernel, gy_kernel, angles


def normalized_rows(images):
    flat = images.reshape(len(images), -1).astype(np.float64)
    flat -= flat.mean(axis=1, keepdims=True)
    flat /= np.sqrt(np.sum(flat*flat, axis=1, keepdims=True)+1e-12)
    return flat


def pool_layer(name, op, inp, size):
    return {"name": name, "op": op, "inputs": [inp],
            "size": [1, size], "stride": [1, size]}


def main():
    templates = []
    for digit in range(10):
        rng = np.random.default_rng(1729+digit)
        templates.append(np.stack([render(digit, rng) for _ in range(MODELS)]))
    templates = np.stack(templates)  # class, model, row, column
    soft, kernel = gaussian28(templates.reshape(-1, SIZE, SIZE), 1.15)
    coarse = soft.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))
    hog, gx_kernel, gy_kernel, angles = hog_features(soft)
    fine_w = normalized_rows(soft).astype(np.float32)*.20
    coarse_w = normalized_rows(coarse).astype(np.float32)*.50
    hog_w = normalized_rows(hog).astype(np.float32)*1.50
    grad_w = np.zeros((2, 1, 3, 3), dtype=np.float32)
    grad_w[0, 0] = gx_kernel
    grad_w[1, 0] = gy_kernel
    direction = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    pos_w = np.zeros((8, 2, 1, 1), dtype=np.float32)
    pos_w[:, :, 0, 0] = direction
    weights = {
        "blur.W": kernel[None, None].astype(np.float32),
        "blur.b": np.zeros(1, np.float32),
        "grad.W": grad_w,
        "grad.b": np.zeros(2, np.float32),
        "hog_pos.W": pos_w,
        "hog_pos.b": np.zeros(8, np.float32),
        "hog_neg.W": -pos_w,
        "hog_neg.b": np.zeros(8, np.float32),
        "fine.W": fine_w,
        "fine.b": np.zeros(10*MODELS, np.float32),
        "coarse.W": coarse_w,
        "coarse.b": np.zeros(10*MODELS, np.float32),
        "hog.W": hog_w,
        "hog.b": np.zeros(10*MODELS, np.float32),
    }
    layers = [
        {"name": "blur", "op": "conv", "inputs": ["image"], "out": 1,
         "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "fine_norm", "op": "norm", "inputs": ["blur"]},
        {"name": "fine", "op": "dense", "inputs": ["fine_norm"], "out": 10*MODELS},
        {"name": "fine_grid", "op": "reshape", "inputs": ["fine"],
         "shape": [10, 1, MODELS]},
        pool_layer("fine_best", "maxpool", "fine_grid", MODELS),
        {"name": "fine_score", "op": "reshape", "inputs": ["fine_best"], "shape": [10]},
        {"name": "coarse_pool", "op": "avgpool", "inputs": ["blur"],
         "size": [2, 2], "stride": [2, 2]},
        {"name": "coarse_norm", "op": "norm", "inputs": ["coarse_pool"]},
        {"name": "coarse", "op": "dense", "inputs": ["coarse_norm"], "out": 10*MODELS},
        {"name": "coarse_grid", "op": "reshape", "inputs": ["coarse"],
         "shape": [10, 1, MODELS]},
        pool_layer("coarse_best", "maxpool", "coarse_grid", MODELS),
        {"name": "coarse_score", "op": "reshape", "inputs": ["coarse_best"], "shape": [10]},
        {"name": "grad", "op": "conv", "inputs": ["blur"], "out": 2,
         "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "hog_pos", "op": "conv", "inputs": ["grad"], "out": 8,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "hog_neg", "op": "conv", "inputs": ["grad"], "out": 8,
         "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "hog_pos_r", "op": "relu", "inputs": ["hog_pos"]},
        {"name": "hog_neg_r", "op": "relu", "inputs": ["hog_neg"]},
        {"name": "hog_abs", "op": "add", "inputs": ["hog_pos_r", "hog_neg_r"]},
        {"name": "hog_pool", "op": "avgpool", "inputs": ["hog_abs"],
         "size": [4, 4], "stride": [4, 4]},
        {"name": "hog_norm", "op": "norm", "inputs": ["hog_pool"]},
        {"name": "hog", "op": "dense", "inputs": ["hog_norm"], "out": 10*MODELS},
        {"name": "hog_grid", "op": "reshape", "inputs": ["hog"],
         "shape": [10, 1, MODELS]},
        pool_layer("hog_best", "maxpool", "hog_grid", MODELS),
        {"name": "hog_score", "op": "reshape", "inputs": ["hog_best"], "shape": [10]},
        {"name": "scores", "op": "add",
         "inputs": ["fine_score", "coarse_score", "hog_score"]},
    ]
    (OUT/"network.json").write_text(json.dumps({"layers": layers}, indent=2)+"\n")
    np.savez(OUT/"weights.npz", **weights)


if __name__ == "__main__":
    main()
