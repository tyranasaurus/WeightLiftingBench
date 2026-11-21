"""Hand-designed 10-class photo classifier. Every number below is typed from general knowledge of the
classes (sky is blue, grass is green, animals are brown/warm, vehicles have long horizontal edges ...).
No training data, no fitting, no generated images are used. Usage: python3 build.py OUT_FOLDER"""
import sys, json, numpy as np
out = sys.argv[1] if len(sys.argv) > 1 else "."

# ---- layer 1: per-pixel colour categories + luminance edges (3x3 conv, no padding -> 30x30) ----
W = np.zeros((10, 3, 3, 3)); b = np.zeros(10)
W[0, :, 1, 1] = [-0.5, -0.5, 1.0]            # 0 blue   (sky, water)
W[1, :, 1, 1] = [-0.5, 1.0, -0.5]            # 1 green  (grass, leaves)
W[2, :, 1, 1] = [1.0, 0.0, -1.0]             # 2 warm   (brown, tan, orange, red: fur, earth)
W[3, :, 1, 1] = [1.0, -1.0, 0.0]             # 3 red    (painted vehicles)
W[4, :, 1, 1] = 1 / 3; b[4] = -0.6           # 4 bright (white sky, white paint, snow)
W[5, :, 1, 1] = -1 / 3; b[5] = 0.35          # 5 dark   (shadow, tyres, dark fur)
sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0   # Sobel, on luminance, gain 4
for c in range(3):
    W[6, c], W[7, c] = 4 * sx / 3, -4 * sx / 3              # 6,7 vertical edges (either polarity)
    W[8, c], W[9, c] = 4 * sx.T / 3, -4 * sx.T / 3          # 8,9 horizontal edges

# ---- readout: rule table. For each class and feature: weight in (top row, side of middle rows, centre, bottom row)
# of a 4x4 grid of pooled cells. Feature order: blue, green, warm, red, bright, dark, vertical edge, horizontal edge.
T = {
 0: [(2, 2, 1, 1.5), (-1, -1, -1, -.5), (-1.5, -1.5, -1, -1.5), (0, 0, 0, 0), (1, 1, .5, .5), (-.5, -.5, 0, -.5), (-1, -1, 0, -1), (-1, -1, 1, -1)],       # airplane: sky all round, smooth
 1: [(-.5, -.5, 0, -1), (0, 0, -.5, -.5), (0, 0, .5, 0), (.5, .5, 2, .5), (0, 0, .5, .3), (0, 0, .3, 1), (0, 0, .5, .5), (0, .5, 1.5, 1)],               # automobile: paint, shadow below, long edges
 2: [(.5, .5, 0, 0), (.7, .7, 0, .5), (.3, .3, .5, .3), (0, 0, 0, 0), (.3, .3, 0, 0), (0, 0, .3, 0), (-.5, -.5, .3, -.3), (-.7, -.7, 0, -.5)],            # bird: small, blurred sky/leaf background
 3: [(-1, -1, -1, -1), (-1, -1, -.5, -1), (.5, .5, .7, .5), (0, 0, 0, 0), (0, 0, .3, 0), (.3, .3, .3, .3), (.3, .3, .5, .3), (0, 0, 0, 0)],              # cat: indoors, fills frame
 4: [(-.5, -1, -1, -1), (1.5, 1.5, .5, 1.5), (.5, .5, 1, .7), (-.5, -.5, -.5, -.5), (-.3, -.3, -.3, -.3), (.2, 0, 0, 0), (.3, .3, .5, .5), (0, 0, 0, 0)],  # deer: tan body in green/brown field
 5: [(-1, -1, -1, -1), (-.3, -.3, -.5, 0), (.5, .5, 1, .5), (0, 0, 0, 0), (0, 0, .7, 0), (.2, .2, .2, .2), (0, 0, .3, .3), (0, 0, 0, 0)],                # dog: warm/white fur, indoors or lawn
 6: [(-1.5, -1.5, -1.5, -1.5), (1, 1, 1.5, 1), (.7, .7, .7, .7), (-.3, -.3, -.3, -.3), (-1, -1, -.5, -1), (.7, .7, .3, .7), (.3, .3, .3, .3), (.3, .3, .5, .3)],  # frog: green/brown, dark, textured
 7: [(.5, 0, -1, -1), (.5, .7, -.3, 1.2), (0, .3, 1.2, .7), (0, 0, 0, 0), (.3, 0, -.3, 0), (0, 0, .7, 0), (0, .3, .7, 1), (0, 0, 0, 0)],                 # horse: brown body, grass below, legs
 8: [(1.2, 1.5, 1, 2.5), (-.5, -.5, -.5, -.5), (-1, -1, -.5, -1.5), (0, 0, .3, 0), (.5, .3, .5, -.3), (-.3, 0, .3, .5), (0, 0, .3, -.5), (-.5, .7, 1, 0)],  # ship: water below, horizon
 9: [(.3, -.3, -.3, -1), (0, 0, -.5, -.5), (0, 0, .3, 0), (.3, .7, 1.5, .5), (.5, .5, 1, .3), (0, .3, .3, 1), (.5, 1, 1.5, 1), (.5, .7, 1, 1)],          # truck: big box, many edges, fills frame
}
chan = [[0], [1], [2], [3], [4], [5], [6, 7], [8, 9]]
D = np.zeros((10, 10, 4, 4))
for k, rows in T.items():
    for f, (top, side, centre, bottom) in enumerate(rows):
        g = np.zeros((4, 4)); g[0] = top; g[3] = bottom; g[1:3] = side; g[1:3, 1:3] = centre
        for c in chan[f]:
            D[k, c] = g
net = {"layers": [
    {"name": "f", "op": "conv", "inputs": ["image"], "out": 10, "kernel": [3, 3], "stride": 1, "padding": 0},
    {"name": "r", "op": "relu", "inputs": ["f"]},
    {"name": "p", "op": "avgpool", "inputs": ["r"], "size": [9, 9], "stride": [7, 7]},
    {"name": "scores", "op": "dense", "inputs": ["p"], "out": 10},
]}
json.dump(net, open(out + "/network.json", "w"), indent=1)
np.savez(out + "/weights.npz", **{"f.W": W.astype(np.float32), "f.b": b.astype(np.float32),
                                  "scores.W": D.reshape(10, 160).astype(np.float32), "scores.b": np.zeros(10, np.float32)})
