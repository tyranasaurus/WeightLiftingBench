#!/usr/bin/env python3
"""Build a data-free, analytically specified CIFAR-10 classifier."""
import json
import math
import sys
from pathlib import Path
import numpy as np


def layer(name, op, inputs, **settings):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(settings)
    return d


def gauss2(x, y, sx, sy):
    return math.exp(-0.5 * ((x / sx) ** 2 + (y / sy) ** 2))


def soft_ellipse(x, y, cx, cy, rx, ry, edge=0.16):
    q = math.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
    return max(0.0, min(1.0, (1.0 + edge - q) / (2.0 * edge)))


def line_mask(x, y, x1, y1, x2, y2, width):
    vx, vy = x2 - x1, y2 - y1
    t = ((x - x1) * vx + (y - y1) * vy) / max(vx * vx + vy * vy, 1e-9)
    t = max(0.0, min(1.0, t))
    d = math.hypot(x - (x1 + t * vx), y - (y1 + t * vy))
    return max(0.0, min(1.0, (width - d) / (0.45 * width) + 0.5))


def ring(x, y, cx, cy, r, width):
    d = abs(math.hypot(x - cx, y - cy) - r)
    return max(0.0, 1.0 - d / width)


def build(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = []
    weights = {}

    # Fixed opponent-colour and luminance channels. Every quantity is chosen
    # analytically, with no data-dependent calculation.
    colours = np.array([
        [ 0.333,  0.333,  0.333],  # luminance
        [-0.333, -0.333, -0.333],  # darkness
        [ 0.667, -0.333, -0.333],  # red
        [-0.333,  0.667, -0.333],  # green
        [-0.333, -0.333,  0.667],  # blue
        [ 0.333,  0.333, -0.667],  # yellow
        [-0.500,  0.250,  0.250],  # cyan
        [ 0.250, -0.500,  0.250],  # magenta
        [ 0.500,  0.000, -0.500],  # warm versus blue
        [ 0.000,  0.500, -0.500],  # vegetation versus blue
        [ 0.500, -0.250, -0.250],  # red dominance
        [-0.250,  0.500, -0.250],  # green dominance
        [-0.250, -0.250,  0.500],  # blue dominance
        [ 0.250,  0.250,  0.250],  # bright
        [-0.250, -0.250, -0.250],  # black/dark
        [ 0.500, -0.500,  0.000],  # red-green
    ], dtype=np.float32)
    cbias = np.array([0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, -0.47, 0.30, 0.0], np.float32)
    W = np.zeros((len(colours), 3, 1, 1), np.float32)
    W[:, :, 0, 0] = colours
    layers += [layer("colour", "conv", ["image"], out=len(colours), kernel=[1, 1], stride=1, padding=0),
               layer("colour_relu", "relu", ["colour"])]
    weights["colour.W"], weights["colour.b"] = W, cbias

    # Oriented first/second Gaussian derivatives and centre-surround filters.
    bank = []
    scales = [(0.70, 1.25), (1.15, 2.10), (1.80, 3.20)]
    for si, (sg, elong) in enumerate(scales):
        for ai in range(8):
            th = math.pi * ai / 8.0
            odd = np.zeros((7, 7), np.float64)
            even = np.zeros((7, 7), np.float64)
            for iy in range(7):
                for ix in range(7):
                    x, y = ix - 3, iy - 3
                    u = math.cos(th) * x + math.sin(th) * y
                    v = -math.sin(th) * x + math.cos(th) * y
                    g = math.exp(-0.5 * ((u / sg) ** 2 + (v / elong) ** 2))
                    odd[iy, ix] = -(u / (sg * sg)) * g
                    even[iy, ix] = ((u * u - sg * sg) / (sg ** 4)) * g
            odd -= odd.mean(); even -= even.mean()
            odd /= max(np.sum(np.abs(odd)), 1e-9)
            even /= max(np.sum(np.abs(even)), 1e-9)
            bank.extend([odd, -odd, even, -even])
    # Isotropic spots and rings at several scales.
    for s1, s2 in [(0.65, 1.25), (1.0, 2.0), (1.55, 3.0)]:
        a = np.zeros((7, 7), np.float64)
        for iy in range(7):
            for ix in range(7):
                x, y = ix - 3, iy - 3
                a[iy, ix] = math.exp(-(x*x+y*y)/(2*s1*s1)) / (s1*s1) - math.exp(-(x*x+y*y)/(2*s2*s2)) / (s2*s2)
        a -= a.mean(); a /= max(np.sum(np.abs(a)), 1e-9)
        bank.extend([a, -a])
    bank = np.asarray(bank, np.float32)
    sources = [0, 1, 8, 9]  # luminance/darkness and two broad colour contrasts
    eW = np.zeros((len(bank) * len(sources), len(colours), 7, 7), np.float32)
    n = 0
    for source in sources:
        for f in bank:
            eW[n, source] = f
            n += 1
    layers += [layer("edges", "conv", ["colour_relu"], out=n, kernel=[7, 7], stride=1, padding=3),
               layer("edge_relu", "relu", ["edges"]),
               layer("edge_pool", "maxpool", ["edge_relu"], size=[2, 2], stride=[2, 2])]
    weights["edges.W"] = eW
    weights["edges.b"] = np.full(n, -0.025, np.float32)

    # Analytic shape templates over a coarse 16x16 map. Positive lobes encode
    # expected edges; broad negative surrounds discourage unrelated clutter.
    # A 12x12 kernel on the 16x16 edge map describes a roughly 24-pixel
    # object. The response is max-pooled over all translations afterwards.
    H = Wd = 12
    yy, xx = np.mgrid[0:H, 0:Wd]
    X = (xx - (Wd - 1) / 2) / ((Wd - 1) / 2)
    Y = (yy - (H - 1) / 2) / ((H - 1) / 2)
    shape_names = [
        "air_side", "air_top", "car_side", "car_front", "bird_side", "bird_fly",
        "cat_face", "cat_body", "deer_side", "dog_face", "dog_side", "frog",
        "horse_side", "ship_side", "truck_side", "truck_front", "horizon",
        "round_face", "four_legs", "wings", "wheels", "mast", "boxy", "slender"
    ]
    masks = {}
    for name in shape_names:
        m = np.zeros((H, Wd), np.float64)
        for iy in range(H):
            for ix in range(Wd):
                x, y = X[iy, ix], Y[iy, ix]
                if name == "air_side":
                    m[iy,ix] = line_mask(x,y,-.75,.05,.75,.05,.07)+line_mask(x,y,-.05,.05,.28,-.28,.08)+line_mask(x,y,-.05,.05,.25,.33,.08)+line_mask(x,y,-.58,.05,-.40,-.20,.06)
                elif name == "air_top":
                    m[iy,ix] = line_mask(x,y,0,-.75,0,.70,.07)+line_mask(x,y,-.65,-.02,.65,-.02,.08)+line_mask(x,y,-.28,.45,.28,.45,.06)
                elif name == "car_side":
                    m[iy,ix] = line_mask(x,y,-.65,.25,.65,.25,.08)+line_mask(x,y,-.55,.18,-.30,-.20,.08)+line_mask(x,y,-.30,-.20,.28,-.20,.08)+line_mask(x,y,.28,-.20,.55,.18,.08)+ring(x,y,-.38,.28,.14,.06)+ring(x,y,.38,.28,.14,.06)
                elif name == "car_front":
                    m[iy,ix] = line_mask(x,y,-.45,.28,.45,.28,.08)+line_mask(x,y,-.38,.20,-.28,-.28,.08)+line_mask(x,y,-.28,-.28,.28,-.28,.08)+line_mask(x,y,.28,-.28,.38,.20,.08)+ring(x,y,-.27,.10,.06,.04)+ring(x,y,.27,.10,.06,.04)
                elif name == "bird_side":
                    m[iy,ix] = soft_ellipse(x,y,-.05,.10,.42,.20)-.5*soft_ellipse(x,y,-.05,.10,.28,.10)+line_mask(x,y,.28,.04,.56,-.04,.07)+line_mask(x,y,-.35,.08,-.58,.18,.06)
                elif name == "bird_fly" or name == "wings":
                    m[iy,ix] = line_mask(x,y,-.75,.30,0,-.12,.10)+line_mask(x,y,0,-.12,.75,.30,.10)+soft_ellipse(x,y,0,.10,.16,.40)
                elif name == "cat_face":
                    m[iy,ix] = ring(x,y,0,.05,.43,.09)+line_mask(x,y,-.34,-.22,-.18,-.62,.08)+line_mask(x,y,-.18,-.62,-.02,-.27,.08)+line_mask(x,y,.34,-.22,.18,-.62,.08)+line_mask(x,y,.18,-.62,.02,-.27,.08)
                elif name == "cat_body":
                    m[iy,ix] = ring(x,y,-.1,.12,.36,.10)+line_mask(x,y,.20,.28,.55,.45,.07)+line_mask(x,y,.55,.45,.62,-.05,.07)
                elif name in ("deer_side", "horse_side"):
                    m[iy,ix] = soft_ellipse(x,y,-.08,.04,.48,.25)-.55*soft_ellipse(x,y,-.08,.04,.34,.12)+line_mask(x,y,.30,-.02,.58,-.28,.08)+ring(x,y,.60,-.32,.15,.06)
                    for lx in (-.38,-.12,.18,.38): m[iy,ix] += line_mask(x,y,lx,.20,lx+(.10 if name=="deer_side" else .04),.70,.055)
                    if name == "deer_side": m[iy,ix] += line_mask(x,y,.64,-.44,.55,-.72,.04)+line_mask(x,y,.64,-.44,.76,-.68,.04)
                elif name == "dog_face":
                    m[iy,ix] = ring(x,y,0,.02,.40,.10)+line_mask(x,y,-.35,-.20,-.55,-.55,.10)+line_mask(x,y,.35,-.20,.55,-.55,.10)+soft_ellipse(x,y,0,.20,.18,.13)
                elif name == "dog_side":
                    m[iy,ix] = soft_ellipse(x,y,-.05,.05,.48,.25)-.55*soft_ellipse(x,y,-.05,.05,.34,.12)+ring(x,y,.48,-.12,.17,.06)
                    for lx in (-.34,-.08,.20,.38): m[iy,ix] += line_mask(x,y,lx,.20,lx,.62,.06)
                elif name == "frog":
                    m[iy,ix] = ring(x,y,0,.12,.38,.10)+ring(x,y,-.25,-.15,.12,.06)+ring(x,y,.25,-.15,.12,.06)+line_mask(x,y,-.25,.26,-.62,.53,.10)+line_mask(x,y,.25,.26,.62,.53,.10)
                elif name == "ship_side":
                    m[iy,ix] = line_mask(x,y,-.72,.15,.70,.15,.08)+line_mask(x,y,-.72,.15,-.45,.42,.08)+line_mask(x,y,-.45,.42,.48,.42,.08)+line_mask(x,y,.48,.42,.70,.15,.08)+line_mask(x,y,-.35,.05,.38,.05,.06)+line_mask(x,y,0,.05,0,-.55,.05)
                elif name == "truck_side":
                    m[iy,ix] = line_mask(x,y,-.72,.28,.72,.28,.08)+line_mask(x,y,-.68,.20,-.68,-.30,.08)+line_mask(x,y,-.68,-.30,.20,-.30,.08)+line_mask(x,y,.20,-.30,.60,-.05,.08)+line_mask(x,y,.60,-.05,.68,.20,.08)+ring(x,y,-.42,.28,.15,.06)+ring(x,y,.40,.28,.15,.06)
                elif name == "truck_front":
                    m[iy,ix] = line_mask(x,y,-.48,.42,.48,.42,.09)+line_mask(x,y,-.48,.42,-.48,-.48,.09)+line_mask(x,y,-.48,-.48,.48,-.48,.09)+line_mask(x,y,.48,-.48,.48,.42,.09)+line_mask(x,y,-.35,.03,.35,.03,.06)
                elif name == "horizon": m[iy,ix] = line_mask(x,y,-.9,.25,.9,.25,.07)
                elif name == "round_face": m[iy,ix] = ring(x,y,0,0,.42,.10)+ring(x,y,-.16,-.06,.05,.035)+ring(x,y,.16,-.06,.05,.035)
                elif name == "four_legs":
                    for lx in (-.42,-.16,.16,.42): m[iy,ix] += line_mask(x,y,lx,.05,lx,.65,.06)
                elif name == "wheels": m[iy,ix] = ring(x,y,-.38,.30,.15,.06)+ring(x,y,.38,.30,.15,.06)
                elif name == "mast": m[iy,ix] = line_mask(x,y,0,.30,0,-.70,.05)+line_mask(x,y,-.55,.32,.55,.32,.06)
                elif name == "boxy": m[iy,ix] = line_mask(x,y,-.55,-.40,.55,-.40,.08)+line_mask(x,y,-.55,.40,.55,.40,.08)+line_mask(x,y,-.55,-.40,-.55,.40,.08)+line_mask(x,y,.55,-.40,.55,.40,.08)
                elif name == "slender": m[iy,ix] = soft_ellipse(x,y,0,0,.20,.60)-.55*soft_ellipse(x,y,0,0,.09,.45)
        masks[name] = np.clip(m, 0, 1)

    # Map each shape mask onto all orientations/scales for the luminance edge bank.
    # Hand-selected orientation preferences are broad to tolerate pose variation.
    n_shapes = len(shape_names)
    tW = np.zeros((n_shapes, n, H, Wd), np.float32)
    # Luminance source occupies the first len(bank) channels. Use all positive
    # derivative phases; masks provide the global geometry.
    for j, name in enumerate(shape_names):
        m = masks[name]
        surround = np.exp(-((X/0.95)**2 + (Y/0.95)**2))
        spatial = m - 0.035 * surround
        # Equal L1 mass makes scores depend on layout match rather than the
        # amount of ink used to draw a particular geometric prototype.
        spatial /= max(np.sum(np.abs(spatial)), 1e-9)
        for fi in range(len(bank)):
            phase = fi % 4
            tW[j, fi] = spatial * (0.24 if phase < 2 else 0.17)
    layers += [layer("shapes", "conv", ["edge_pool"], out=n_shapes, kernel=[12, 12], stride=1, padding=0),
               layer("shape_relu_map", "relu", ["shapes"]),
               layer("shape_relu", "maxpool", ["shape_relu_map"], size=[5, 5], stride=[5, 5])]
    weights["shapes.W"] = tW
    weights["shapes.b"] = np.full(n_shapes, -0.010, np.float32)

    # Global scene colour layout. Average pooling gives a 4x4 coarse map.
    layers += [layer("scene_pool", "avgpool", ["colour_relu"], size=[8, 8], stride=[8, 8]),
               layer("scene_flat", "reshape", ["scene_pool"], shape=[256])]

    # Final hand-written semantic rule table combines templates and scene cues.
    # The two branches are projected separately and added.
    rW = np.zeros((10, n_shapes), np.float32)
    idx = {s:i for i,s in enumerate(shape_names)}
    def put(cls, terms):
        for key, val in terms.items(): rW[cls, idx[key]] = val
    put(0,{"air_side":2.6,"air_top":2.2,"wings":1.4,"slender":.5,"horizon":.25})
    put(1,{"car_side":2.5,"car_front":1.8,"wheels":1.8,"boxy":.45})
    put(2,{"bird_side":2.0,"bird_fly":2.4,"wings":1.7,"slender":.4})
    put(3,{"cat_face":2.7,"cat_body":1.5,"round_face":1.2})
    put(4,{"deer_side":2.7,"four_legs":1.2,"slender":.7})
    put(5,{"dog_face":2.3,"dog_side":2.2,"round_face":.7,"four_legs":.6})
    put(6,{"frog":3.4,"round_face":.55})
    put(7,{"horse_side":3.0,"four_legs":1.4,"slender":.5})
    put(8,{"ship_side":3.1,"horizon":1.3,"mast":1.4,"boxy":.3})
    put(9,{"truck_side":2.7,"truck_front":2.6,"wheels":1.4,"boxy":1.1})
    layers += [layer("shape_scores", "dense", ["shape_relu"], out=10)]
    weights["shape_scores.W"] = rW
    weights["shape_scores.b"] = np.array([-.16,.06,-.05,.10,.32,.06,.32,.20,-.03,-.30], np.float32)

    sW = np.zeros((10, 256), np.float32)
    def scene(cls, ch, rows, cols, val):
        for ry in rows:
            for cx in cols:
                sW[cls, ch*16 + ry*4 + cx] += val
    allc = range(4); top = range(2); bot = range(2,4); middle = range(1,3)
    # blue sky and pale upper background: aircraft; blue/grey lower half: ships
    scene(0, 12, top, allc, .70); scene(0, 4, top, allc, .35); scene(0, 11, bot, allc, -.22)
    scene(8, 4, bot, allc, .62); scene(8, 6, bot, allc, .32); scene(8, 12, top, allc, .18)
    # road vehicles favour dark/warm central lower regions and reject vegetation.
    for c in (1,9):
        scene(c, 14, middle, middle, .42); scene(c, 10, middle, middle, .18); scene(c, 11, bot, allc, -.14)
    scene(9, 13, middle, middle, .12); scene(1, 2, middle, middle, .24)
    # Cars are usually chromatic and low/sleek; large trucks often have a
    # bright neutral rectangular cargo body.
    scene(1, 2, middle, middle, .18); scene(1, 4, middle, middle, .08)
    scene(9, 13, middle, middle, .10); scene(9, 2, middle, middle, -.10)
    # natural animal scenes: green/brown surroundings; species-specific colour.
    for c in (2,3,4,5,6,7): scene(c, 11, allc, allc, .12)
    scene(6, 11, middle, middle, .45); scene(6, 5, middle, middle, .16)
    scene(4, 8, middle, middle, .26); scene(7, 8, middle, middle, .30)
    scene(3, 14, middle, middle, .22); scene(5, 8, middle, middle, .15)
    scene(2, 4, top, allc, .16); scene(2, 11, bot, allc, .16)
    layers += [layer("scene_scores", "dense", ["scene_flat"], out=10),
               layer("scores", "add", ["shape_scores", "scene_scores"])]
    weights["scene_scores.W"] = sW
    weights["scene_scores.b"] = np.array([-.08,.02,-.02,.04,.00,.05,.02,.00,-.10,.03], np.float32)

    (out / "network.json").write_text(json.dumps({"layers": layers}, separators=(",", ":")))
    np.savez(out / "weights.npz", **{k: np.asarray(v, np.float32) for k,v in weights.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
