"""Hand-built, data-free COCO person detector.

All kernels are constructed from colour-space formulas, Gaussians, Sobel
filters, and generic capsule drawings of people.  No image data is read.
"""
import json
import os
import sys
import numpy as np

net = {"layers": []}
weights = {}


def layer(name, op, inputs, **settings):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(settings)
    net["layers"].append(d)


def conv(name, source, W, b=None, stride=1, padding=None):
    W = np.asarray(W, dtype=np.float32)
    if W.ndim == 3:
        W = W[None]
    if padding is None:
        padding = W.shape[-1] // 2
    layer(name, "conv", [source], out=W.shape[0], kernel=[W.shape[-1]] * 2,
          stride=stride, padding=padding)
    weights[name + ".W"] = W
    weights[name + ".b"] = np.zeros(W.shape[0], np.float32) if b is None else np.asarray([b], np.float32) if np.isscalar(b) else np.asarray(b, np.float32)


def dense(name, source, W, b=None):
    W = np.asarray(W, dtype=np.float32)
    layer(name, "dense", [source], out=W.shape[0])
    weights[name + ".W"] = W
    weights[name + ".b"] = np.zeros(W.shape[0], np.float32) if b is None else np.asarray(b, np.float32)


def relu(name, source):
    layer(name, "relu", [source])


def add(name, sources):
    layer(name, "add", sources)


def pool(name, source, size, stride=None, average=False):
    layer(name, "avgpool" if average else "maxpool", [source], size=list(size),
          stride=list(size if stride is None else stride))


def reshape(name, source, shape):
    layer(name, "reshape", [source], shape=list(shape))


def gaussian(k, cx, cy, sx, sy=None):
    if sy is None:
        sy = sx
    yy, xx = np.mgrid[:k, :k]
    a = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    return a / max(a.sum(), 1e-12)


def ellipse(k, cx, cy, rx, ry):
    yy, xx = np.mgrid[:k, :k]
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(float)


def capsule(mask, p0, p1, radius):
    # Rasterize a generic capsule by its analytic point-to-segment distance.
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    x0, y0 = p0; x1, y1 = p1
    vx, vy = x1 - x0, y1 - y0
    t = ((xx - x0) * vx + (yy - y0) * vy) / max(vx * vx + vy * vy, 1e-9)
    t = np.clip(t, 0, 1)
    dist2 = (xx - (x0 + t * vx)) ** 2 + (yy - (y0 + t * vy)) ** 2
    mask |= dist2 <= radius * radius


def blur(a, sigma=0.65):
    # Small binomial blur, used only to tolerate pose and rasterization changes.
    r = max(1, int(np.ceil(2 * sigma)))
    z = np.arange(-r, r + 1)
    q = np.exp(-0.5 * (z / sigma) ** 2); q /= q.sum()
    b = np.pad(a, ((r, r), (r, r)), mode="edge")
    b = sum(q[i] * b[i:i + a.shape[0], r:r + a.shape[1]] for i in range(len(q)))
    b = np.pad(b, ((0, 0), (r, r)), mode="edge")
    # The horizontal pass below uses the vertically blurred image padded in x.
    b = np.pad(sum(q[i] * b[r:r + a.shape[0], i:i + a.shape[1]] for i in range(len(q))), ((0, 0), (0, 0)))
    return b


def smooth2d(a, sigma=0.65):
    r = max(1, int(np.ceil(2 * sigma)))
    z = np.arange(-r, r + 1)
    q = np.exp(-0.5 * (z / sigma) ** 2); q /= q.sum()
    p = np.pad(a, ((r, r), (r, r)), mode="edge")
    v = sum(q[i] * p[i:i + a.shape[0], :] for i in range(len(q)))
    return sum(q[i] * v[:, i:i + a.shape[1]] for i in range(len(q)))


def skin_tent(source, prefix):
    # YCbCr skin chroma.  Triangular ramps make this useful for dark and light
    # skin and reject the neutral Cb=Cr=0.5 of copied greyscale photographs.
    cb = np.array([[[[-0.168736]], [[-0.331264]], [[0.5]]]])
    cr = np.array([[[[0.5]], [[-0.418688]], [[-0.081312]]]])
    conv(prefix + "_cb", source, cb, b=0.5, padding=0)
    conv(prefix + "_cr", source, cr, b=0.5, padding=0)
    for nm, center, width in (("cb", 0.42, 0.12), ("cr", 0.60, 0.14)):
        base = prefix + "_" + nm
        conv(base + "_hi", base, [[[[1.0]]]], b=-center, padding=0)
        conv(base + "_lo", base, [[[[-1.0]]]], b=center, padding=0)
        relu(base + "_hi_r", base + "_hi"); relu(base + "_lo_r", base + "_lo")
        add(base + "_abs", [base + "_hi_r", base + "_lo_r"])
        conv(base + "_tent0", base + "_abs", [[[[-1.0 / width]]]], b=1.0, padding=0)
        relu(base + "_tent", base + "_tent0")
    add(prefix + "_chroma", [prefix + "_cb_tent", prefix + "_cr_tent"])
    conv(prefix + "_skin0", prefix + "_chroma", np.ones((1, 1, 1, 1)), b=-0.95, padding=0)
    relu(prefix + "_skin", prefix + "_skin0")
    # A little smoothing prevents JPEG chroma speckle from becoming a head.
    g = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16
    conv(prefix + "_skin_s", prefix + "_skin", g[None, None])


def gray_and_edges(source, prefix, edges=True):
    norm_name = prefix + "_normalized"
    layer(norm_name, "norm", [source])
    lum = np.array([0.299, 0.587, 0.114], dtype=float).reshape(1, 3, 1, 1)
    conv(prefix + "_gray", norm_name, lum, padding=0)
    if not edges:
        return
    sx = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
    sy = sx.T
    W = np.stack([sx, sy])[:, None]
    conv(prefix + "_signed_grad", prefix + "_gray", W)
    conv(prefix + "_signed_grad_neg", prefix + "_gray", -W)
    relu(prefix + "_grad_pos", prefix + "_signed_grad")
    relu(prefix + "_grad_neg", prefix + "_signed_grad_neg")
    add(prefix + "_grad", [prefix + "_grad_pos", prefix + "_grad_neg"])


def head_kernels(k):
    c = (k - 1) / 2
    positive = gaussian(k, c, c, 0.18 * k, 0.22 * k)
    surround = gaussian(k, c, c, 0.43 * k, 0.44 * k)
    return (positive - 0.92 * surround)[None, None]


def face_detail_kernel(k):
    c = (k - 1) / 2
    # Eyes and mouth are deliberately broad, generic dark-detail regions.
    eyes = gaussian(k, c - 0.13 * k, c - 0.08 * k, 0.085 * k, 0.055 * k)
    eyes += gaussian(k, c + 0.13 * k, c - 0.08 * k, 0.085 * k, 0.055 * k)
    mouth = gaussian(k, c, c + 0.18 * k, 0.14 * k, 0.055 * k)
    a = eyes + 0.45 * mouth
    a /= a.sum()
    return a[None, None]


def max_scalars(name, sources):
    """Max of any number of (1,1,1) maps, via one-hot vectors and pooling."""
    n = len(sources)
    projects = []
    for i, source in enumerate(sources):
        r = name + "_r" + str(i)
        reshape(r, source, [1])
        W = np.zeros((n, 1)); W[i, 0] = 1
        d = name + "_d" + str(i)
        dense(d, r, W)
        projects.append(d)
    add(name + "_vector", projects)
    reshape(name + "_row", name + "_vector", [1, 1, n])
    pool(name, name + "_row", [1, n])


# Four-level Gaussian-like area pyramid.  Templates on 48 and 24 pixel levels
# correspond to people roughly 34 and 68 pixels tall in the original photo.
levels = [("p0", "image")]
for i in range(1, 4):
    src = levels[-1][1]
    name = "p" + str(i)
    pool(name, src, [2, 2], average=True)
    levels.append((name, name))

head_maxes = []
face_maxes = []
for lev, source in levels:
    skin_tent(source, lev)
    head_k = (5, 9) if lev in ("p0", "p3") else (5, 9, 17)
    for k in head_k:
        tag = lev + "_h" + str(k)
        conv(tag + "_blob0", lev + "_skin_s", head_kernels(k), b=-0.10, padding=0)
        relu(tag + "_blob", tag + "_blob0")
        side = 96 // (2 ** int(lev[1])) - k + 1
        pool(tag + "_max", tag + "_blob", [side, side])
        head_maxes.append(tag + "_max")

        # Detail is measured on an unnormalised luminance image, so darkness
        # remains comparable across photographs.
        lum = np.array([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
        conv(tag + "_lum", source, lum, padding=0)
        g3 = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16
        conv(tag + "_lum_blur", tag + "_lum", g3[None, None])
        conv(tag + "_lum_minus", tag + "_lum", -np.ones((1, 1, 1, 1)), padding=0)
        add(tag + "_dark_sum", [tag + "_lum_blur", tag + "_lum_minus"])
        conv(tag + "_dark0", tag + "_dark_sum", np.ones((1, 1, 1, 1)), b=-0.025, padding=0)
        relu(tag + "_dark", tag + "_dark0")
        # Both complete templates are valid convolutions, so no zero-padding
        # position can masquerade as a missing (negative) surround.
        ident = np.ones((1, 1, 1, 1))
        conv(tag + "_skin_crop", tag + "_blob0", 0.70 * ident, padding=0)
        if k >= 9:
            c = (k - 1) / 2
            le = gaussian(k, c - 0.13 * k, c - 0.08 * k, 0.085 * k, 0.055 * k)
            re = gaussian(k, c + 0.13 * k, c - 0.08 * k, 0.085 * k, 0.055 * k)
            conv(tag + "_eye_l", tag + "_dark", le[None, None], padding=0)
            conv(tag + "_eye_r", tag + "_dark", re[None, None], padding=0)
            conv(tag + "_eye_l_copy", tag + "_eye_l", np.ones((1, 1, 1, 1)), padding=0)
            conv(tag + "_eye_r_neg", tag + "_eye_r", -np.ones((1, 1, 1, 1)), padding=0)
            add(tag + "_eye_diff", [tag + "_eye_l_copy", tag + "_eye_r_neg"])
            relu(tag + "_eye_drop0", tag + "_eye_diff")
            conv(tag + "_eye_drop", tag + "_eye_drop0", -np.ones((1, 1, 1, 1)), padding=0)
            add(tag + "_eye_min", [tag + "_eye_l", tag + "_eye_drop"])
            conv(tag + "_eye_pair", tag + "_eye_min", 8.0 * np.ones((1, 1, 1, 1)), padding=0)
            mo = gaussian(k, c, c + 0.18 * k, 0.14 * k, 0.055 * k)
            conv(tag + "_mouth", tag + "_dark", 2.0 * mo[None, None], padding=0)
            add(tag + "_eyes", [tag + "_eye_pair", tag + "_mouth"])
        else:
            conv(tag + "_eyes", tag + "_dark", 4.0 * face_detail_kernel(k), padding=0)
        add(tag + "_face_sum", [tag + "_skin_crop", tag + "_eyes"])
        conv(tag + "_face0", tag + "_face_sum", np.ones((1, 1, 1, 1)), b=-0.35, padding=0)
        relu(tag + "_face", tag + "_face0")
        side = (96 // (2 ** int(lev[1]))) - k + 1
        pool(tag + "_face_max", tag + "_face", [side, side])
        face_maxes.append(tag + "_face_max")

max_scalars("head_any", head_maxes)
max_scalars("face_any", face_maxes)

# Generic humanoid silhouettes: neutral standing, walking, and a raised-arm
# pose.  Matching both absolute gradient directions is the hand-built analogue
# of a small HOG template bank.
def silhouette(k, pose):
    m = np.zeros((k, k), bool)
    c = (k - 1) / 2
    u = k / 17.0
    def P(x, y): return (c + x * u, (1.0 + y) * u)
    capsule(m, P(0, 2.2), P(0, 2.2), 1.65 * u)       # head
    capsule(m, P(0, 4.0), P(0, 9.0), 2.25 * u)       # torso
    if pose == 0:
        arms = [((-1.5, 4.5), (-3.0, 10.0)), ((1.5, 4.5), (3.0, 10.0))]
        legs = [((-1.0, 8.5), (-2.0, 15.0)), ((1.0, 8.5), (2.0, 15.0))]
    elif pose == 1:
        arms = [((-1.5, 4.5), (-4.0, 8.5)), ((1.5, 4.5), (4.0, 7.0))]
        legs = [((-1.0, 8.5), (-4.0, 14.5)), ((1.0, 8.5), (4.0, 13.0))]
    else:
        arms = [((-1.5, 4.5), (-4.5, 1.5)), ((1.5, 4.5), (4.5, 1.5))]
        legs = [((-1.0, 8.5), (-2.5, 15.0)), ((1.0, 8.5), (2.5, 15.0))]
    for a, b in arms: capsule(m, P(*a), P(*b), 0.85 * u)
    for a, b in legs: capsule(m, P(*a), P(*b), 1.05 * u)
    return m


shape_maxes = []
fused_maxes = []
for lev in ("p1", "p2"):
    gray_and_edges(dict(levels)[lev], lev, edges=True)
    k = 17
    for pose_id in range(3):
        tag = lev + "_pose" + str(pose_id)
        m = silhouette(k, pose_id)
        # Boundary components aligned with horizontal and vertical gradients.
        mx = np.zeros_like(m); my = np.zeros_like(m)
        mx[:, 1:-1] = m[:, 2:] ^ m[:, :-2]
        my[1:-1, :] = m[2:, :] ^ m[:-2, :]
        tx = smooth2d(mx.astype(float), 0.8)
        ty = smooth2d(my.astype(float), 0.8)
        # Penalize unrelated texture throughout the bounding box.  Uniform
        # clutter therefore cannot obtain a high matched-filter score.
        bgx = smooth2d((~m).astype(float), 1.0)
        bgy = bgx.copy()
        def match(t, bg):
            t = t / max(t.sum(), 1e-9)
            bg = bg / max(bg.sum(), 1e-9)
            return t - 0.88 * bg
        W = np.stack([match(tx, bgx), match(ty, bgy)], axis=0)[None]
        conv(tag + "_shape0", lev + "_grad", W, b=-0.035, padding=0)
        relu(tag + "_shape", tag + "_shape0")
        side = 96 // (2 ** int(lev[1])) - k + 1
        pool(tag + "_shape_max", tag + "_shape", [side, side])
        shape_maxes.append(tag + "_shape_max")

        c = (k - 1) / 2; u = k / 17.0
        hp = gaussian(k, c, 3.2 * u, 1.5 * u, 1.8 * u)
        hs = gaussian(k, c, 3.2 * u, 3.0 * u, 3.2 * u)
        conv(tag + "_head_skin", lev + "_skin_s", 1.2 * (hp - 0.55 * hs)[None, None], b=-0.06, padding=0)
        conv(tag + "_shape_support", tag + "_shape0", 0.35 * np.ones((1, 1, 1, 1)), padding=0)
        add(tag + "_fused_sum", [tag + "_shape_support", tag + "_head_skin"])
        conv(tag + "_fused0", tag + "_fused_sum", np.ones((1, 1, 1, 1)), b=-0.18, padding=0)
        relu(tag + "_fused", tag + "_fused0")
        pool(tag + "_fused_max", tag + "_fused", [side, side])
        fused_maxes.append(tag + "_fused_max")

max_scalars("shape_any", shape_maxes)
max_scalars("body_any", fused_maxes)

# Convert the four global detector responses to class scores.  Face and fused
# head/body detections are strong evidence; an isolated colour blob or a bare
# silhouette is deliberately much weaker.  The zero class score is a fixed
# decision threshold, not a fitted bias.
for source in ("head_any", "face_any", "shape_any", "body_any"):
    reshape(source + "_flat", source, [1])
W = np.array([[0., 0., 0., 0.], [0.10, 3.0, 0.08, 3.0]])
# Project each scalar into the same four-component evidence vector.
projects = []
for i, source in enumerate(("head_any", "face_any", "shape_any", "body_any")):
    P = np.zeros((4, 1)); P[i, 0] = 1
    dense("evidence_" + source, source + "_flat", P)
    projects.append("evidence_" + source)
add("evidence", projects)
dense("scores", "evidence", W, b=[0.0, -1.20])

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "network.json"), "w") as f:
    json.dump(net, f, separators=(",", ":"))
np.savez(os.path.join(out, "weights.npz"), **weights)
print("wrote", len(net["layers"]), "layers and", sum(v.size for v in weights.values()), "weights")
