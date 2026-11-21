"""Writes network.json and weights.npz for a hand-designed person / no-person network.
Every number comes from this file: a classic RGB skin-colour rule and centre-surround (disc minus ring) blob filters.
No data is read and nothing is fitted."""
import sys, os, json
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)

layers = []
W = {}


def add(name, op, inputs, **kw):
    layers.append({"name": name, "op": op, "inputs": inputs, **kw})


# ---- 1. per-pixel skin colour rule (R > G >= B, moderate chroma, not dark, not yellow) ----------------------------
# Each unit is a violation of one linear constraint, scaled so that a violation of about 0.08 removes the pixel.
k = 12.0
viol = [
    # (weights on R, G, B), bias                     meaning
    ([-k, k, 0.0], k * 0.05),                      # R - G < 0.05 : not reddish enough
    ([k, -k, 0.0], -k * 0.38),                     # R - G > 0.38 : too saturated red / orange
    ([0.0, -k, k], -k * 0.02),                     # B > G + 0.02 : bluish / purple
    ([0.0, k, -k], -k * 0.24),                     # G - B > 0.24 : too yellow / orange (wood, cheese)
    ([-k, 0.0, 0.0], k * 0.32),                    # R < 0.32     : too dark
    ([-1.5 * k, 2.5 * k, -k], -k * 0.05),          # (G - B) > 1.5 (R - G) + 0.05 : hue too yellow
]
add("viol", "conv", ["image"], out=len(viol), kernel=[1, 1], stride=1, padding=0)
W["viol.W"] = np.array([v[0] for v in viol], np.float64).reshape(len(viol), 3, 1, 1)
W["viol.b"] = np.array([v[1] for v in viol], np.float64)
add("violr", "relu", ["viol"])
# skin = relu(1 - sum of violations), in [0, 1]
add("skinp", "conv", ["violr"], out=1, kernel=[1, 1], stride=1, padding=0)
W["skinp.W"] = -np.ones((1, len(viol), 1, 1))
W["skinp.b"] = np.ones(1)
add("skin", "relu", ["skinp"])
add("skin2", "avgpool", ["skin"], size=[2, 2], stride=[2, 2])   # 48 x 48
add("skin4", "avgpool", ["skin"], size=[4, 4], stride=[4, 4])   # 24 x 24


# ---- 2. skin blobs: mean skin in a disc minus mean skin in a ring around it (faces, hands, limbs) ------------------
def ring_kernel(K, rc):
    c = K // 2
    yy, xx = np.mgrid[0:K, 0:K]
    d = np.sqrt((yy - c) ** 2.0 + (xx - c) ** 2.0)
    cen = (d <= rc).astype(np.float64)
    sur = ((d > 1.5 * rc) & (d <= 2.3 * rc)).astype(np.float64)
    return cen / cen.sum() - sur / sur.sum()


THETA = 0.30   # a blob must be this much more skin-coloured than its surround before it counts at all


def blob_branch(tag, src, K, radii, stride):
    add("blob" + tag, "conv", [src], out=len(radii), kernel=[K, K], stride=stride, padding=K // 2)
    W["blob" + tag + ".W"] = np.stack([ring_kernel(K, r) for r in radii])[:, None]
    W["blob" + tag + ".b"] = -THETA * np.ones(len(radii))
    add("blob" + tag + "r", "relu", ["blob" + tag])


blob_branch("S", "skin", 11, [2.0], 2)                 # radius 2 px, output 48 x 48
blob_branch("A", "skin2", 13, [1.5, 2.25], 1)          # radius 3 and 4.5 px
blob_branch("B", "skin4", 19, [1.75, 2.5, 3.75], 1)    # radius 7, 10 and 15 px
add("maxS", "maxpool", ["blobSr"], size=[48, 48], stride=[48, 48])
add("maxA", "maxpool", ["blobAr"], size=[48, 48], stride=[48, 48])
add("maxB", "maxpool", ["blobBr"], size=[24, 24], stride=[24, 24])

# ---- 3. readout: person score = sum over scales of the strongest blob - 0.15; no-person score = 0 ------------------
for tag, n, bias in (("S", 1, -0.15), ("A", 2, 0.0), ("B", 3, 0.0)):
    add("d" + tag, "dense", ["max" + tag], out=2)
    w = np.zeros((2, n)); w[1, :] = 1.0
    W["d" + tag + ".W"] = w
    W["d" + tag + ".b"] = np.array([0.0, bias])
add("out", "add", ["dS", "dA", "dB"])

with open(os.path.join(out_dir, "network.json"), "w") as f:
    json.dump({"layers": layers}, f, indent=1)
np.savez(os.path.join(out_dir, "weights.npz"), **{n: v.astype(np.float32) for n, v in W.items()})
