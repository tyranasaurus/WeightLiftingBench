import json, os, sys
import numpy as np

# Data-free hand-built CIFAR-10 style feature network.
# The weights are generated from analytic filters, colour-opponent axes, and
# geometric object/background sketches.  This program reads no files.

LUM = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def smooth_masks():
    yy, xx = np.mgrid[0:8, 0:8].astype(np.float32)
    x = (xx - 3.5) / 3.5
    y = (yy - 3.5) / 3.5

    def g(cx, cy, sx, sy):
        return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2)).astype(np.float32)

    def n(m):
        m = np.maximum(m.astype(np.float32), 0.0)
        s = float(m.sum())
        return m / (s if s > 0 else 1.0)

    masks = {}
    masks["all"] = n(np.ones((8, 8), np.float32))
    masks["center"] = n(g(0, 0, .58, .58))
    masks["wide_center"] = n(g(0, 0, .92, .70))
    masks["upper"] = n(g(0, -.55, 1.0, .48))
    masks["lower"] = n(g(0, .55, 1.0, .48))
    masks["top"] = n(np.clip(-y + .10, 0, None))
    masks["bottom"] = n(np.clip(y + .10, 0, None))
    masks["sky"] = n(np.clip(-y + .35, 0, None))
    masks["ground"] = n(np.clip(y + .28, 0, None))
    masks["mid"] = n(np.exp(-0.5 * (y / .38) ** 2))
    masks["horizon"] = n(np.exp(-0.5 * ((y - .02) / .20) ** 2))
    masks["low_horizon"] = n(np.exp(-0.5 * ((y - .30) / .22) ** 2))
    masks["vehicle"] = n(g(0, .34, .95, .42))
    masks["low_center"] = n(g(0, .48, .68, .36))
    masks["animal_body"] = n(g(0, .20, .75, .55))
    masks["legs"] = n(g(-.35, .58, .18, .34) + g(.35, .58, .18, .34) + .45 * g(0, .62, .18, .30))
    masks["wheels"] = n(g(-.48, .58, .19, .19) + g(.48, .58, .19, .19))
    masks["ship_hull"] = n(g(0, .28, .92, .24))
    masks["plane_band"] = n(g(0, -.15, 1.05, .38))
    return masks


def make_conv():
    names = []
    W = []
    b = []

    def add(name, center=None, spatial=None, bias=0.0, scale=1.0):
        k = np.zeros((3, 3, 3), dtype=np.float32)
        if center is not None:
            k[:, 1, 1] = np.array(center, dtype=np.float32) * scale
        if spatial is not None:
            sp = np.array(spatial, dtype=np.float32) * scale
            for c in range(3):
                k[c] += LUM[c] * sp
        names.append(name)
        W.append(k)
        b.append(float(bias))

    for t in (.25, .45, .65, .80):
        add(f"bright{int(t*100):02d}", center=LUM, bias=-t)
    for t in (.25, .40, .55, .70):
        add(f"dark{int(t*100):02d}", center=-LUM, bias=t)

    color_specs = [
        ("red",   [1.0, -.50, -.50], (.02, .12, .25)),
        ("green", [-.50, 1.0, -.50], (.02, .12, .25)),
        ("blue",  [-.50, -.50, 1.0], (.02, .12, .25)),
        ("yellow", [.55, .55, -1.0], (.04, .16, .30)),
        ("cyan",  [-.75, .45, .75], (.04, .16)),
        ("magenta", [.70, -.80, .70], (.04, .16)),
        ("warm",  [.70, .25, -.55], (.04, .18)),
        ("earth", [.55, .35, -.65], (.02, .16)),
        ("veg",   [-.45, .85, -.25], (.03, .15)),
        ("skyblue", [-.35, .05, .85], (.06, .18)),
    ]
    for base, vec, ths in color_specs:
        for t in ths:
            add(f"{base}{int(t*100):02d}", center=vec, bias=-t)

    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 4.0
    sy = sx.T
    sd1 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], np.float32) / 4.0
    sd2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], np.float32) / 4.0
    for nm, sp in [("edgex", sx), ("edgey", sy), ("edged1", sd1), ("edged2", sd2)]:
        add(nm + "p", spatial=sp, bias=-.025)
        add(nm + "n", spatial=-sp, bias=-.025)

    hbar = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], np.float32) / 6.0
    vbar = hbar.T
    dbar1 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], np.float32) / 6.0
    dbar2 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], np.float32) / 6.0
    for nm, sp in [("lineh", hbar), ("linev", vbar), ("lined1", dbar1), ("lined2", dbar2)]:
        add(nm + "b", spatial=sp, bias=-.018)
        add(nm + "d", spatial=-sp, bias=-.018)

    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32) / 4.0
    ring = np.array([[-1, 1, -1], [1, 0, 1], [-1, 1, -1]], np.float32) / 4.0
    for nm, sp in [("spot", lap), ("ring", ring)]:
        add(nm + "b", spatial=sp, bias=-.025)
        add(nm + "d", spatial=-sp, bias=-.025)

    return names, np.stack(W).astype(np.float32), np.array(b, dtype=np.float32)


def make_readout(names):
    masks = smooth_masks()
    W = np.zeros((10, len(names), 8, 8), dtype=np.float32)
    b = np.array([-.03, -.15, .18, .20, .14, .18, .16, .05, -.12, -.22], dtype=np.float32)

    def chans(prefixes):
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        return [i for i, n in enumerate(names) if any(n.startswith(p) for p in prefixes)]

    groups = {
        "bright": chans("bright"), "dark": chans("dark"), "red": chans("red"),
        "green": chans("green"), "blue": chans("blue"), "yellow": chans("yellow"),
        "cyan": chans("cyan"), "magenta": chans("magenta"), "warm": chans("warm"),
        "earth": chans("earth"), "veg": chans("veg"), "skyblue": chans("skyblue"),
        "edge": chans(("edgex", "edgey", "edged1", "edged2")),
        "diag": chans(("edged1", "edged2", "lined1", "lined2")),
        "hline": chans(("edgey", "lineh")),
        "vline": chans(("edgex", "linev")),
        "line": chans(("lineh", "linev", "lined1", "lined2")),
        "spot": chans(("spot", "ring")),
        "straight": chans(("edgex", "edgey", "lineh", "linev")),
        "texture": chans(("edgex", "edgey", "edged1", "edged2", "spot", "ring")),
    }

    def add(cls, group, mask, coef):
        ids = groups[group] if isinstance(group, str) else group
        if not ids:
            return
        m = masks[mask].astype(np.float32)
        s = coef / (len(ids) ** 0.5)
        for idx in ids:
            W[cls, idx] += s * m

    add(0, "blue", "sky", 2.4);       add(0, "skyblue", "sky", 2.2)
    add(0, "bright", "sky", 1.0);     add(0, "diag", "plane_band", 1.4)
    add(0, "hline", "plane_band", .8); add(0, "vline", "plane_band", .35)
    add(0, "blue", "bottom", -1.0);   add(0, "green", "ground", -1.6)
    add(0, "veg", "ground", -1.4);    add(0, "earth", "ground", -1.2)
    add(0, "dark", "wheels", -1.5);   add(0, "warm", "center", -.6)

    add(1, "dark", "wheels", 2.35);   add(1, "straight", "vehicle", 1.45)
    add(1, "hline", "vehicle", 1.1);  add(1, "vline", "vehicle", .80)
    add(1, "red", "vehicle", .9);     add(1, "yellow", "vehicle", .55)
    add(1, "blue", "vehicle", .38);   add(1, "bright", "vehicle", .40)
    add(1, "green", "all", -1.0);     add(1, "veg", "all", -1.0)
    add(1, "blue", "sky", -.7);       add(1, "blue", "bottom", -1.05)
    add(1, "earth", "all", -.45);     add(1, "diag", "all", -.25)

    add(2, "texture", "center", 1.50); add(2, "diag", "center", .95)
    add(2, "green", "all", .75);      add(2, "veg", "all", .85)
    add(2, "earth", "center", .70);   add(2, "warm", "center", .60)
    add(2, "blue", "sky", .45);       add(2, "yellow", "center", .70)
    add(2, "straight", "vehicle", -1.20); add(2, "dark", "wheels", -.95)
    add(2, "blue", "bottom", -.65);   add(2, "bright", "all", -.15)

    add(3, "texture", "center", 1.65); add(3, "dark", "center", 1.30)
    add(3, "earth", "center", .95);    add(3, "warm", "center", .65)
    add(3, "spot", "center", .85);     add(3, "bright", "center", .35)
    add(3, "green", "ground", -.85);   add(3, "veg", "ground", -.95)
    add(3, "blue", "all", -1.05);      add(3, "straight", "vehicle", -1.15)
    add(3, "dark", "wheels", -.55)

    add(4, "earth", "animal_body", 1.75); add(4, "warm", "animal_body", 1.0)
    add(4, "green", "ground", 1.15);      add(4, "veg", "ground", 1.25)
    add(4, "vline", "legs", 1.1);         add(4, "texture", "animal_body", .85)
    add(4, "dark", "legs", .50);          add(4, "hline", "animal_body", .42)
    add(4, "blue", "all", -1.15);         add(4, "bright", "all", -.35)
    add(4, "straight", "vehicle", -.90);  add(4, "red", "all", -.38)

    add(5, "texture", "center", 1.72); add(5, "earth", "center", 1.05)
    add(5, "warm", "center", .85);     add(5, "dark", "center", .95)
    add(5, "bright", "center", .48);   add(5, "spot", "center", .85)
    add(5, "green", "ground", -.50);   add(5, "veg", "ground", -.55)
    add(5, "blue", "all", -1.05);      add(5, "straight", "vehicle", -1.05)
    add(5, "dark", "wheels", -.48)

    add(6, "green", "center", 2.25);   add(6, "veg", "center", 1.95)
    add(6, "green", "ground", 1.10);   add(6, "yellow", "center", 1.05)
    add(6, "warm", "center", .62);     add(6, "texture", "center", 1.05)
    add(6, "spot", "center", .62);     add(6, "blue", "all", -1.10)
    add(6, "straight", "vehicle", -1.25); add(6, "dark", "wheels", -.85)
    add(6, "bright", "sky", -.48)

    add(7, "earth", "animal_body", 1.75); add(7, "warm", "animal_body", 1.05)
    add(7, "green", "ground", 1.05);      add(7, "veg", "ground", 1.10)
    add(7, "hline", "animal_body", 1.10); add(7, "vline", "legs", 1.35)
    add(7, "dark", "legs", .70);          add(7, "texture", "animal_body", .75)
    add(7, "blue", "all", -1.05);         add(7, "straight", "vehicle", -.80)
    add(7, "red", "all", -.38)

    add(8, "blue", "bottom", 2.15);     add(8, "skyblue", "bottom", 1.80)
    add(8, "blue", "sky", 1.00);        add(8, "bright", "ship_hull", 1.05)
    add(8, "hline", "horizon", 1.45);   add(8, "hline", "ship_hull", 1.30)
    add(8, "straight", "ship_hull", .70); add(8, "dark", "ship_hull", .40)
    add(8, "green", "all", -1.20);      add(8, "veg", "all", -1.20)
    add(8, "earth", "all", -.95);       add(8, "dark", "wheels", -.85)
    add(8, "red", "vehicle", -.20)

    add(9, "straight", "vehicle", 1.75); add(9, "vline", "vehicle", 1.20)
    add(9, "hline", "vehicle", 1.00);    add(9, "dark", "wheels", 1.75)
    add(9, "bright", "vehicle", .80);    add(9, "red", "vehicle", .95)
    add(9, "yellow", "vehicle", .50);    add(9, "warm", "vehicle", .40)
    add(9, "green", "all", -.95);        add(9, "veg", "all", -.95)
    add(9, "blue", "bottom", -1.00);     add(9, "blue", "sky", -.65)
    add(9, "earth", "all", -.45);        add(9, "diag", "all", -.35)

    cal = np.array([.92, .78, .98, 1.00, .98, 1.00, 1.00, .96, .78, .74], dtype=np.float32)
    W *= cal[:, None, None, None]
    b *= cal
    return W.reshape(10, -1).astype(np.float32), b.astype(np.float32)


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names, cW, cb = make_conv()
    dW, db = make_readout(names)
    net = {"layers": [
        {"name": "p0", "op": "avgpool", "inputs": ["image"], "size": [2, 2], "stride": [2, 2]},
        {"name": "c1", "op": "conv", "inputs": ["p0"], "out": int(cW.shape[0]), "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "r1", "op": "relu", "inputs": ["c1"]},
        {"name": "p1", "op": "avgpool", "inputs": ["r1"], "size": [2, 2], "stride": [2, 2]},
        {"name": "score", "op": "dense", "inputs": ["p1"], "out": 10},
    ]}
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, separators=(",", ":"))
    np.savez(os.path.join(out_dir, "weights.npz"), **{
        "c1.W": cW.astype(np.float32), "c1.b": cb.astype(np.float32),
        "score.W": dW.astype(np.float32), "score.b": db.astype(np.float32),
    })


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
