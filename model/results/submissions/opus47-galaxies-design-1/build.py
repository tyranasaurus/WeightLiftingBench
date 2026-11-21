"""Hand-designed galaxy classifier using rotation-invariant template matching.

Architecture:
  image (3,128,128)
    --conv1x1--> (luma, redness) (2,128,128)
    --avgpool 4x4--> (2,32,32)
    --norm--> standardised channels (each mean 0, var 1)
    --dense--> 10 classes * 8 orientations = 80 template cosine-similarity scores
    --reshape (10,1,8)--maxpool 1x8--> max over 8 rotations per class
    --reshape (10,)--> final class scores

Each template is a hand-designed (luma, redness) pattern rotated to 8 orientations,
with zero mean per channel and L2 unit norm so cosine similarity is comparable.

Shapes come from general galaxy-morphology knowledge:
  round smooth (2): compact circular blob, red (old stars)
  in-between round (3): mild ellipse, red
  cigar (4): elongated smooth, mildly red
  barred spiral (5): short central BAR + arm knots
  tight spiral (6): compact bulge + nearby disk, red bulge
  loose spiral (7): extended BLUE disk
  edge-on no bulge (8): long thin line, dim centre, mildly blue (disk)
  edge-on with bulge (9): thin line + red central bulge
  merging (1): two bright peaks with gap between
  disturbed (0): asymmetric lobe + blue
"""
import os, sys, json
import numpy as np


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    N_ORIENT = 8
    H = 32

    net = {"layers": [
        {"name": "color", "op": "conv", "inputs": ["image"], "out": 2, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "pool", "op": "avgpool", "inputs": ["color"], "size": [4, 4], "stride": [4, 4]},
        {"name": "norm", "op": "norm", "inputs": ["pool"]},
        {"name": "tmpl", "op": "dense", "inputs": ["norm"], "out": 10 * N_ORIENT},
        {"name": "rshp", "op": "reshape", "inputs": ["tmpl"], "shape": [10, 1, N_ORIENT]},
        {"name": "mpool", "op": "maxpool", "inputs": ["rshp"], "size": [1, N_ORIENT], "stride": [1, 1]},
        {"name": "out", "op": "reshape", "inputs": ["mpool"], "shape": [10]},
    ]}
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, indent=2)

    color_W = np.zeros((2, 3, 1, 1), dtype=np.float32)
    color_W[0, :, 0, 0] = [0.299, 0.587, 0.114]   # luma
    color_W[1, :, 0, 0] = [1.0, 0.0, -1.0]        # redness R - B
    color_b = np.zeros(2, dtype=np.float32)

    y_grid, x_grid = np.mgrid[0:H, 0:H].astype(np.float32)
    cy = cx = (H - 1) / 2.0

    def templ(k, theta):
        dy = y_grid - cy
        dx = x_grid - cx
        c, s = np.cos(theta), np.sin(theta)
        xp = c * dx + s * dy
        yp = -s * dx + c * dy
        r = np.sqrt(xp * xp + yp * yp + 1e-9)

        def el(a, b, xc=0.0, yc=0.0):
            return np.exp(-(((xp - xc) / a) ** 2 + ((yp - yc) / b) ** 2))

        def circ(sigma):
            return np.exp(-(r / sigma) ** 2)

        if k == 0:  # disturbed: asymmetric + blue
            core = circ(3.2)
            lobe = el(2.3, 2.3, 5.0, 4.0)
            lum = 0.9 * core + 0.8 * lobe
            col = -0.45 * (core + lobe)
        elif k == 1:  # merging: two peaks + central gap
            g1 = el(2.4, 2.1, 5.5, 0)
            g2 = el(2.4, 2.1, -5.5, 0)
            gap = circ(1.3)
            lum = (g1 + g2) - 0.6 * gap
            col = 0.05 * (g1 + g2)
        elif k == 2:  # round smooth: compact round, red
            shape = circ(4.8)
            lum = shape
            col = 0.55 * shape
        elif k == 3:  # in-between round smooth: mild ellipse, red
            shape = el(6.0, 4.3)
            lum = shape
            col = 0.45 * shape
        elif k == 4:  # cigar-shaped smooth: elongated smooth, mildly red
            shape = el(9.0, 2.6)
            lum = shape
            col = 0.25 * shape
        elif k == 5:  # barred spiral: short bar + offset arm knots
            bar = el(3.8, 1.3)
            arm1 = el(2.2, 2.2, 4.8, 2.3)
            arm2 = el(2.2, 2.2, -4.8, -2.3)
            lum = 1.1 * bar + 0.6 * (arm1 + arm2)
            col = 0.2 * bar - 0.2 * (arm1 + arm2)
        elif k == 6:  # unbarred tight spiral: compact bulge + modest tight disk
            bulge = circ(2.0)
            disk = circ(5.5) - bulge
            disk = np.clip(disk, 0, None)
            lum = 0.9 * bulge + 0.5 * disk
            col = 0.25 * bulge - 0.1 * disk
        elif k == 7:  # unbarred loose spiral: extended BLUE disk
            center = circ(2.5)
            disk = circ(9.0) - center
            disk = np.clip(disk, 0, None)
            lum = 0.4 * center + 1.0 * disk
            col = -0.40 * (center + disk)
        elif k == 8:  # edge-on no bulge: long thin line, dim centre, slight blue
            line = el(11.5, 1.05)
            dip = circ(1.2)
            lum = line - 0.4 * dip
            col = -0.12 * line
        elif k == 9:  # edge-on with bulge: thin line + RED central bulge (line dominates)
            line = el(11.0, 1.45)
            bulge = circ(1.9)
            lum = 1.0 * line + 0.35 * bulge   # line dominates, bulge modest
            col = 0.5 * bulge + 0.1 * line
        else:
            lum = np.zeros_like(r); col = np.zeros_like(r)
        return lum, col

    orientations = np.linspace(0, np.pi, N_ORIENT, endpoint=False)
    tmpl_W = np.zeros((10 * N_ORIENT, 2 * H * H), dtype=np.float32)
    tmpl_b = np.zeros(10 * N_ORIENT, dtype=np.float32)

    for k in range(10):
        for oi, theta in enumerate(orientations):
            lum, col = templ(k, theta)
            lum = lum - lum.mean()
            col = col - col.mean()
            v = np.concatenate([lum.flatten(), col.flatten()]).astype(np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            tmpl_W[k * N_ORIENT + oi] = v

    np.savez(os.path.join(out_dir, "weights.npz"),
             **{"color.W": color_W, "color.b": color_b,
                "tmpl.W": tmpl_W.astype(np.float32), "tmpl.b": tmpl_b.astype(np.float32)})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
