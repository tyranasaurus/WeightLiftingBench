"""Hand-designed Fashion-MNIST classifier: data-free weight builder.

Every number below is a geometric constant describing a clothing class *in general*
(trapezoids for bodies, segments for sleeves/soles/straps, disks for necks/handle-holes).
No training data, image statistics, or fitted values are used anywhere.

Architecture (a matched-filter bank):
  image (1,28,28)
   -> norm            per-image zero-mean / unit-variance (contrast & brightness invariance)
   -> dense  (40)     correlation with 40 idealized silhouette templates (4 pose-variants x 10 classes),
                      each template zero-mean & unit-L2-norm; bias = per-class offset
   -> reshape (10,1,4)
   -> maxpool [1,4]   best-matching variant per class
   -> reshape (10)    class scores; argmax = prediction

Usage: python3 build.py OUT_DIR   (writes OUT_DIR/network.json and OUT_DIR/weights.npz)
"""
import json
import sys
import numpy as np

N = 28
yy, xx = np.meshgrid(np.arange(N, dtype=np.float64), np.arange(N, dtype=np.float64), indexing="ij")
SOFT = 1.3  # soft (anti-aliased) edge width in pixels

# ---- geometric drawing primitives -----------------------------------------

def ge(v, a, w=SOFT):
    return np.clip((v - a) / w + 0.5, 0.0, 1.0)

def le(v, a, w=SOFT):
    return np.clip((a - v) / w + 0.5, 0.0, 1.0)

def band(v, a, b, w=SOFT):
    return ge(v, a, w) * le(v, b, w)

def rect(x0, x1, y0, y1, w=SOFT):
    return band(xx, x0, x1, w) * band(yy, y0, y1, w)

def trap(y0, y1, xl0, xr0, xl1, xr1, w=SOFT):
    t = np.clip((yy - y0) / max(y1 - y0, 1e-6), 0.0, 1.0)
    xl = xl0 + (xl1 - xl0) * t
    xr = xr0 + (xr1 - xr0) * t
    return band(xx, xl, xr, w) * band(yy, y0, y1, w)

def disk(cx, cy, rx, ry, w=SOFT):
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    return le(d * rx, rx, w)

def seg(x0, y0, x1, y1, half, w=SOFT):
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L2, 0.0, 1.0)
    px, py = x0 + t * dx, y0 + t * dy
    dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    return le(dist, half, w)

def mx(*masks):
    m = masks[0]
    for a in masks[1:]:
        m = np.maximum(m, a)
    return m

def clip01(m):
    return np.clip(m, 0.0, 1.0)

# ---- per-class templates ----------------------------------------------------

# All items are CENTRED in the frame and scaled so the longest side ~ 28 px (per the task).
# Garments (tops, trouser, dress, bag) therefore fill the FULL HEIGHT; footwear is a short,
# wide shape centred VERTICALLY (empty rows at top AND bottom). Centre of frame ~ (13.5,13.5).

def t_tshirt(tank=False):
    # tall body filling its core extent (~cols 6-21), SHORT sleeves only at the top
    body = trap(1, 26, 6.0, 21.0, 7.0, 20.0)
    neck = disk(13.5, 1.5, 3.2, 2.2)
    m = clip01(body - 0.9 * neck)
    if not tank:
        sleeves = mx(trap(1, 9, 3.0, 7.0, 6.5, 7.5), trap(1, 9, 21.0, 25.0, 20.5, 25.0))
        m = mx(m, sleeves)
    # gentle negative lower-sides: no long sleeves there (vs pullover/coat/shirt)
    lower_sides = mx(rect(2, 6, 11, 24), rect(22, 26, 11, 24))
    return m - 0.9 * lower_sides

def t_trouser():
    # NARROW (core cols ~8-19), two legs separated by a firm central GAP -- the decisive cue
    waist = rect(7, 20, 0, 6)
    lleg = trap(4, 27, 7.5, 12.3, 8.0, 12.0)
    rleg = trap(4, 27, 15.7, 20.5, 16.0, 19.5)
    m = mx(waist, 1.2 * lleg, 1.2 * rleg)
    gap = rect(12.5, 15.5, 8, 27)                     # firm leg-gap over the lower 3/4
    sides = mx(rect(0, 4, 2, 27), rect(24, 28, 2, 27))
    return m - 1.3 * gap - 0.3 * sides

def t_pullover():
    # MEDIUM-width solid long-sleeve top. Its distinct feature is a SOLID centre (no placket,
    # no opening) -> it beats shirt (placket) and coat (opening) there. A negative at the wide
    # frame-sides means a bulky wide coat does NOT match a pullover.
    body = trap(1, 25, 6.0, 21.0, 7.0, 20.0)
    neck = disk(13.5, 1.0, 3.0, 2.0)
    m = clip01(body - 0.9 * neck)
    sleeves = mx(seg(7.0, 3.5, 3.5, 21.0, 2.6), seg(20.0, 3.5, 24.5, 21.0, 2.6))
    m = mx(m, 1.0 * sleeves)
    m = mx(m, 0.9 * rect(11.0, 16.5, 5, 24))          # distinct solid centre
    return m - 0.5 * mx(rect(1, 5, 4, 24), rect(23, 27, 4, 24))

def t_dress(flare=True):
    # NARROW tall garment with a SOLID skirt (no leg-gap) -> beats trouser at the centre; and
    # negatives at the WIDE sides -> loses to bulky coats/tops there, so it stays in its niche.
    if flare:
        body = trap(1, 27, 10.0, 17.5, 7.0, 21.0)     # A-line flare to the hem
    else:
        body = trap(1, 27, 10.0, 17.5, 9.0, 19.0)     # straight
    neck = disk(13.5, 0.5, 2.6, 1.8)
    m = clip01(body - 0.8 * neck)
    m = mx(m, 0.7 * rect(11.0, 16.5, 14, 27))         # solid skirt centre (trousers are empty here)
    sides = mx(rect(1, 7, 2, 24), rect(21, 27, 2, 24))
    return m - 0.6 * sides

def t_coat():
    # WIDEST top. Its distinct feature is bulk out to the frame SIDES (cols ~2-6, 22-26) over
    # the full height -- the region narrower tops (pullover/shirt/tshirt) lack -- so a wide,
    # bulky coat wins here. A faint central opening, kept weak because it is unreliable.
    body = trap(1, 26, 5.0, 23.0, 6.0, 22.0)
    neck = disk(13.5, 1.0, 3.0, 2.0)
    m = clip01(body - 0.7 * neck)
    wide = mx(rect(1.5, 6.0, 3, 24), rect(22.0, 26.5, 3, 24))
    m = mx(m, 1.4 * wide)
    opening = rect(12.8, 15.2, 5, 23)
    return m - 0.4 * opening

def t_shirt(long=True):
    # fitted long-sleeve top with an open collar neckline and a gentle central placket
    body = trap(1, 25, 6.5, 20.5, 7.5, 19.5)
    sleeves = mx(seg(7.0, 3.0, 3.0, 21.0, 2.5), seg(20.0, 3.0, 24.0, 21.0, 2.5))
    neck = disk(13.5, 2.2, 3.0, 2.4)                  # open collar: deeper/wider neckline
    m = clip01(mx(body, 1.2 * sleeves) - 0.9 * neck)
    placket = rect(12.9, 14.1, 4, 24)
    return m - 0.5 * placket

def t_sneaker(mirror=False):
    # low, wide, SOLID shoe centred vertically (~rows 9-20); flat sole, toe at the front
    sole = rect(1, 27, 16, 20.5)
    upper = trap(9, 17, 2.0, 19.0, 5.0, 24.0)
    toe = disk(22.0, 14.5, 5.5, 4.5)
    m = clip01(mx(sole, upper, toe))
    if mirror:
        m = m[:, ::-1]
    # empty top AND bottom rows (short vertical extent) -> separates from all garments
    neg = mx(rect(0, 28, 0, 7), rect(0, 28, 22, 28))
    return m - 0.9 * neg

def t_sandal(mirror=False, heel=False):
    # open footwear centred vertically; a tall slim heel at the back + a low open front
    sole = rect(3, 25, 19, 22.5)
    if heel:
        back = rect(3.0, 6.5, 8, 22.5)                # tall slim heel
        strap = seg(8, 20, 24, 9, 1.1)
        m = mx(sole, back, strap)
    else:
        back = rect(3.0, 6.5, 12, 22)                 # lower heel block
        s1 = seg(7, 20, 15, 12, 1.1)
        s2 = seg(14, 20, 24, 12, 1.1)
        m = mx(sole, back, s1, s2)
    if mirror:
        m = m[:, ::-1]
    neg = mx(rect(0, 28, 0, 7), rect(0, 28, 23, 28), 0.6 * rect(7, 22, 13, 19))
    return m - 0.7 * neg

def t_bag(handle=True):
    body = rect(2, 26, 7, 27)                         # big, wide, fills most of the frame
    m = body
    if handle:
        arch = mx(seg(7, 7, 9, 2, 1.2), seg(9, 2, 19, 2, 1.2), seg(21, 7, 19, 2, 1.2))
        m = mx(m, arch)
        hole = disk(14.0, 4.0, 4.2, 2.6)              # opening inside the handle loop
        m = clip01(m - 0.85 * hole)
    return m - 0.3 * rect(0, 28, 0, 2)

def t_boot(mirror=False):
    # tall boot centred vertically (~rows 3-25); foot along the bottom, leg rising at the back
    foot = rect(2, 26, 18, 24)
    leg = trap(3, 18, 3.0, 13.0, 3.0, 12.0)
    toe = disk(22.0, 20.5, 5.0, 4.0)
    m = clip01(mx(foot, leg, toe))
    if mirror:
        m = m[:, ::-1]
    return m - 0.6 * rect(0, 28, 0, 2) - 0.5 * rect(0, 28, 26, 28)

NPER = 2  # two variants per class; each captures only GENUINE within-class variation
          # (mirror / shape type), never another class's look -- so maxpool stays fair.

def mir(m):
    return m[:, ::-1]

def class_templates():
    ts = t_tshirt(); pu = t_pullover(); co = t_coat(); sh = t_shirt(long=True)
    return {
        0: [ts, mir(ts)],                              # T-shirt (symmetric) + mirror
        1: [t_trouser(), mir(t_trouser())],
        2: [pu, mir(pu)],
        3: [t_dress(flare=True), t_dress(flare=False)],   # A-line vs straight dress
        4: [co, mir(co)],
        5: [t_sandal(heel=True), t_sandal(heel=False)],   # heeled vs flat sandal
        6: [sh, mir(sh)],
        7: [t_sneaker(), t_sneaker(mirror=True)],         # left/right facing sneaker
        8: [t_bag(handle=True), t_bag(handle=False)],     # tote vs clutch
        9: [t_boot(), t_boot(mirror=True)],               # left/right facing boot
    }

# per-class score offsets (hand-chosen; uniform prior => 0 unless noted)
BIAS = [0.0] * 10

def build_weights():
    T = class_templates()
    raw = []
    bias = []
    for k in range(10):
        for m in T[k]:
            raw.append(m.astype(np.float64).reshape(-1))
            bias.append(BIAS[k])
    raw = np.stack(raw, 0)                        # (20, 784)
    # Normalise every template to zero-mean / unit-variance over its 784 pixels -- the SAME
    # transform the `norm` op applies to each image. Then template and image live in one space,
    # so score = template . image is an unbiased nearest-template match (no compact-filter bias,
    # no data used: this is arithmetic on my own hand-drawn shapes).
    rows = []
    for v in raw:
        v = v - v.mean()
        s = np.sqrt((v ** 2).mean() + 1e-5)
        v = v / s
        rows.append(v)
    W = np.stack(rows, 0).astype(np.float32)     # (20, 784)
    b = np.array(bias, dtype=np.float32)         # (20,)
    return W, b

def network():
    return {"layers": [
        {"name": "n1", "op": "norm", "inputs": ["image"]},
        {"name": "scores", "op": "dense", "inputs": ["n1"], "out": 10 * NPER},
        {"name": "grid", "op": "reshape", "inputs": ["scores"], "shape": [10, 1, NPER]},
        {"name": "pooled", "op": "maxpool", "inputs": ["grid"], "size": [1, NPER], "stride": [1, NPER]},
        {"name": "out", "op": "reshape", "inputs": ["pooled"], "shape": [10]},
    ]}

def main(out_dir):
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    W, b = build_weights()
    (out / "network.json").write_text(json.dumps(network(), indent=2))
    np.savez(out / "weights.npz", **{"scores.W": W, "scores.b": b})

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
    print("wrote network.json and weights.npz")
