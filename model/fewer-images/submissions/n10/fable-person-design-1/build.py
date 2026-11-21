"""Writes network.json and weights.npz for a hand-designed person / no-person network.
Every number comes from this file: colour rules for skin typed from general knowledge,
and ellipse / ring-sector filters from formulas. No data is read and nothing is fitted."""
import sys, json, numpy as np
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
layers, W = [], {}

# 1. Skin colour as a cone in RGB (hue about 4-33 degrees, saturation 0.14-0.64, not too dark).
#    Each channel of "viol" is how far one rule is broken; "skin" = relu(1 - 25 * sum of violations).
rules = [([0.06, -1.0, 0.94], 0.0),    # hue too red:      (G-B) < 0.06 (R-B)
         ([-0.55, 1.0, -0.45], 0.0),   # hue too yellow:   (G-B) > 0.55 (R-B)
         ([-0.86, 1.0 * 0, 1.0], 0.0), # too grey:         (R-B) < 0.14 R
         ([0.36, 0.0, -1.0], 0.0),     # too saturated:    (R-B) > 0.64 R
         ([-1.0, 0.0, 0.0], 0.25)]     # too dark:         R < 0.25
W["viol.W"] = np.array([r[0] for r in rules], np.float32).reshape(5, 3, 1, 1)
W["viol.b"] = np.array([r[1] for r in rules], np.float32)
layers += [{"name": "viol", "op": "conv", "inputs": ["image"], "out": 5, "kernel": [1, 1], "stride": 1, "padding": 0},
           {"name": "viol_r", "op": "relu", "inputs": ["viol"]}]
W["skin.W"] = np.full((1, 5, 1, 1), -25.0, np.float32)
W["skin.b"] = np.array([1.0], np.float32)
layers += [{"name": "skin", "op": "conv", "inputs": ["viol_r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
           {"name": "skin_r", "op": "relu", "inputs": ["skin"]}]

# 2. Face-sized blob filters: an upright ellipse (centre) and 7 ring sectors around it
#    (the sector straight below is left out: the neck and body are there).
K, A, B = 9, 1.4, 1.9          # kernel size, ellipse half width and half height in pixels
R_IN, R_OUT = 1.5, 2.35        # ring, in units of the ellipse radius
SUB = 5
def blob_kernels():
    ker = np.zeros((8, K, K))
    offs = (np.arange(SUB) + 0.5) / SUB - 0.5
    for iy in range(K):
        for ix in range(K):
            for oy in offs:
                for ox in offs:
                    x, y = ix - K // 2 + ox, iy - K // 2 + oy
                    r = np.hypot(x / A, y / B)
                    if r <= 1.0:
                        ker[0, iy, ix] += 1
                    elif R_IN <= r <= R_OUT:
                        ang = np.degrees(np.arctan2(y / B, x / A)) % 360.0   # 90 = straight down
                        s = int(((ang + 22.5) % 360.0) // 45.0)              # 0:E 1:SE 2:S 3:SW 4:W 5:NW 6:N 7:NE
                        if s == 2:
                            continue
                        ker[1 + (s if s < 2 else s - 1), iy, ix] += 1
    return ker / ker.sum(axis=(1, 2), keepdims=True)
ker = blob_kernels().astype(np.float32)
SECTOR_ALLOW, SECTOR_PEN, PEAK_MIN = 0.25, 1.2, 0.45

# 3. The same filters at 6 scales (face width about 3 to 24 pixels), then the peak over the image.
peaks = []
for f in (1, 2, 3, 4, 6, 8):
    src = "skin_r"
    if f > 1:
        layers.append({"name": f"p{f}", "op": "avgpool", "inputs": ["skin_r"], "size": [f, f], "stride": [f, f]})
        src = f"p{f}"
    stride = 2 if f == 1 else 1
    size = 96 // f // stride if f > 1 else 48
    W[f"b{f}.W"] = ker.reshape(8, 1, K, K).copy()
    W[f"b{f}.b"] = np.array([0.0] + [-SECTOR_ALLOW] * 7, np.float32)
    W[f"r{f}.W"] = np.array([1.0] + [-SECTOR_PEN] * 7, np.float32).reshape(1, 8, 1, 1)
    W[f"r{f}.b"] = np.array([-PEAK_MIN], np.float32)
    layers += [{"name": f"b{f}", "op": "conv", "inputs": [src], "out": 8, "kernel": [K, K], "stride": stride, "padding": K // 2},
               {"name": f"b{f}_r", "op": "relu", "inputs": [f"b{f}"]},
               {"name": f"r{f}", "op": "conv", "inputs": [f"b{f}_r"], "out": 1, "kernel": [1, 1], "stride": 1, "padding": 0},
               {"name": f"r{f}_r", "op": "relu", "inputs": [f"r{f}"]},
               {"name": f"m{f}", "op": "maxpool", "inputs": [f"r{f}_r"], "size": [size, size], "stride": [size, size]}]
    peaks.append(f"m{f}")

# 4. Votes from all scales are added; "person" if the total clears a small margin, else "no person".
layers.append({"name": "votes", "op": "add", "inputs": peaks})
W["out.W"] = np.array([[0.0], [10.0]], np.float32)
W["out.b"] = np.array([0.0, -0.5], np.float32)
layers.append({"name": "out", "op": "dense", "inputs": ["votes"], "out": 2})

(out / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
np.savez(out / "weights.npz", **W)
