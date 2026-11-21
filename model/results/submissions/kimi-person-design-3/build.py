"""Build a hand-designed, multiscale person detector.  No data are read.

The weights are colorimetric skin rules, Sobel orientation filters, ellipse
face models, and geometrically rendered capsule/ellipse human silhouettes.
"""
import json
import sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)
layers = []
weights = {}


def put(name, W, b):
    weights[name + ".W"] = np.asarray(W, dtype=np.float32)
    weights[name + ".b"] = np.asarray(b, dtype=np.float32)


def conv(name, inp, W, b, padding=0):
    W = np.asarray(W, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    layers.append({"name": name, "op": "conv", "inputs": [inp],
                   "out": int(W.shape[0]), "kernel": [int(W.shape[2]), int(W.shape[3])],
                   "stride": 1, "padding": int(padding)})
    put(name, W, b)


def dense(name, inp, W, b):
    W = np.asarray(W, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    layers.append({"name": name, "op": "dense", "inputs": [inp], "out": int(W.shape[0])})
    put(name, W, b)


def relu(name, inp):
    layers.append({"name": name, "op": "relu", "inputs": [inp]})


def pool(name, op, inp, size, stride=None):
    if stride is None:
        stride = size
    layers.append({"name": name, "op": op, "inputs": [inp],
                   "size": list(size), "stride": list(stride)})


def add(name, inputs):
    layers.append({"name": name, "op": "add", "inputs": list(inputs)})


def reshape(name, inp, shape):
    layers.append({"name": name, "op": "reshape", "inputs": [inp], "shape": list(shape)})


# Luminance and illumination normalization.
gray_w = np.array([[[[0.299]], [[0.587]], [[0.114]]]], dtype=np.float32)
conv("gray", "image", gray_w, [0.0])
layers.append({"name": "gray_n", "op": "norm", "inputs": ["gray"]})
pool("small", "avgpool", "gray_n", [4, 4])

# Two deliberately broad YCbCr/RGB skin gates.  Each output is one where all
# margins hold and falls continuously to zero near the boundary of the rule.
def skin_rule(prefix, cr_lo, cr_hi, cb_lo, cb_hi, y_lo, rg_lo, rb_lo):
    # Cr=.5R-.418688G-.081312B+.5; Cb=-.168736R-.331264G+.5B+.5
    cr = np.array([.5, -.418688, -.081312])
    cb = np.array([-.168736, -.331264, .5])
    lum = np.array([.299, .587, .114])
    rows = [cr, -cr, cb, -cb, lum, [1., -1., 0.], [1., 0., -1.]]
    bias = [.5-cr_lo, cr_hi-.5, .5-cb_lo, cb_hi-.5, -y_lo, -rg_lo, -rb_lo]
    W = np.asarray(rows, dtype=np.float32).reshape(7, 3, 1, 1)
    conv(prefix + "_margins", "image", W, bias)
    conv(prefix + "_bad", prefix + "_margins", -100*np.eye(7)[..., None, None], np.zeros(7))
    relu(prefix + "_bad_r", prefix + "_bad")
    conv(prefix + "_pre", prefix + "_bad_r", -np.ones((1, 7, 1, 1)), [1.0])
    relu(prefix, prefix + "_pre")
    pool(prefix + "24", "avgpool", prefix, [4, 4])


skin_rule("skin", .515, .685, .305, .495, .08, .005, .020)
skin_rule("skin_broad", .500, .720, .270, .530, .05, .005, .010)
pool("skin12", "avgpool", "skin24", [2, 2])
pool("skin_broad12", "avgpool", "skin_broad24", [2, 2])
pool("tiny", "avgpool", "small", [2, 2])

# Four unsigned Sobel orientations at 24x24 and 12x12.  Two resolutions keep
# the runtime and temporary convolution windows small even for close-ups.
angles = np.arange(4) * np.pi / 4
sobel_x = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.
sobel_y = sobel_x.T
GW = []
for a in angles:
    GW.append((np.cos(a)*sobel_x + np.sin(a)*sobel_y)[None])
GW = np.asarray(GW, dtype=np.float32)
conv("grad", "small", GW, np.zeros(4), padding=1)
relu("grad_pos", "grad")
conv("grad_rev", "small", -GW, np.zeros(4), padding=1)
relu("grad_neg", "grad_rev")
add("edge", ["grad_pos", "grad_neg"])
conv("grad12", "tiny", GW, np.zeros(4), padding=1)
relu("grad12_pos", "grad12")
conv("grad12_rev", "tiny", -GW, np.zeros(4), padding=1)
relu("grad12_neg", "grad12_rev")
add("edge12", ["grad12_pos", "grad12_neg"])

# Limit local responses so one high-contrast pixel cannot win a global maximum.
def cap_spatial(name, inp, channels, ceiling):
    conv(name+"_over", inp, np.eye(channels)[..., None, None], np.full(channels, -ceiling))
    relu(name+"_over_r", name+"_over")
    conv(name+"_subtract", name+"_over_r", -np.eye(channels)[..., None, None], np.zeros(channels))
    add(name, [inp, name+"_subtract"])

cap_spatial("edge_capped", "edge", 4, .65)
cap_spatial("edge12_capped", "edge12", 4, .65)
conv("dark24_raw", "small", -np.ones((1, 1, 1, 1)), [0.])
relu("dark24_positive", "dark24_raw")
cap_spatial("dark24", "dark24_positive", 1, 1.0)
conv("dark12_raw", "tiny", -np.ones((1, 1, 1, 1)), [0.])
relu("dark12_positive", "dark12_raw")
cap_spatial("dark12", "dark12_positive", 1, 1.0)


def ellipse_mask(y, x, cx, cy, rx, ry, softness=.55):
    q = np.sqrt(((x-cx)/max(rx, .1))**2 + ((y-cy)/max(ry, .1))**2)
    return np.clip((1.-q)*max(min(rx, ry), .5)/softness + .5, 0., 1.)


def capsule_mask(y, x, ax, ay, bx, by, radius):
    vx, vy = bx-ax, by-ay
    den = vx*vx + vy*vy + 1e-9
    t = np.clip(((x-ax)*vx + (y-ay)*vy)/den, 0., 1.)
    d = np.sqrt((x-ax-t*vx)**2 + (y-ay-t*vy)**2)
    return np.clip(radius-d+.5, 0., 1.)


def contour_weights(mask, n_orient=4):
    """Matched unsigned-gradient weights on an antialiased silhouette."""
    gy, gx = np.gradient(mask.astype(np.float64))
    mag = np.sqrt(gx*gx + gy*gy)
    mag[mag < .015] = 0.
    norm = np.sqrt(gx*gx + gy*gy) + 1e-9
    out = []
    for a in np.arange(n_orient)*np.pi/n_orient:
        alignment = np.abs(gx*np.cos(a) + gy*np.sin(a))/norm
        out.append(mag*alignment)
    out = np.asarray(out)
    return out/(out.sum()+1e-9)


def pad_kernel(a, k):
    a = np.asarray(a)
    out = np.zeros(a.shape[:-2] + (k, k), dtype=np.float64)
    h, w = a.shape[-2:]
    y, x = (k-h)//2, (k-w)//2
    out[..., y:y+h, x:x+w] = a
    return out


def gaussian_spot(y, x, cx, cy, sigma):
    z = np.exp(-((x-cx)**2+(y-cy)**2)/(2*sigma*sigma))
    return z/(z.sum()+1e-9)


# Elliptical face models.  Positive skin is balanced by a surrounding ring;
# dark eye, hair and mouth regions and an elliptical boundary add specificity.
face_scales = [(24, 1.5), (24, 2.5), (24, 4.0), (24, 5.5),
               (12, 4.0), (12, 5.5)]
face_skin_scalars = []


def template_max(name, inp, n_templates, side=24):
    pool(name + "_positions", "maxpool", inp, [side, side])
    reshape(name + "_templates", name + "_positions", [1, 1, n_templates])
    pool(name + "_max", "maxpool", name + "_templates", [1, n_templates])
    return name + "_max"


for si, (side, r) in enumerate(face_scales):
    edge_input = "edge_capped" if side == 24 else "edge12_capped"
    skin_input = "skin24" if side == 24 else "skin12"
    gray_input = "dark24" if side == 24 else "dark12"
    k = int(np.ceil(3.15*r))
    k += 1-k % 2
    c = k//2
    yy, xx = np.mgrid[:k, :k].astype(np.float64)
    y, x = yy-c, xx-c
    edge_k, skin_k, gray_k, generic_edge_k = [], [], [], []
    # Narrow and wide heads, plus a slightly rotated-eye variant.
    for vi, aspect in enumerate([.76, .91, 1.06]):
        rx, ry = aspect*r, r
        mask = ellipse_mask(y, x, 0, 0, rx, ry)
        inside = ellipse_mask(y, x, 0, 0, .88*rx, .88*ry)
        inside /= inside.sum()+1e-9
        outer = ellipse_mask(y, x, 0, 0, 1.52*rx, 1.48*ry)
        surround = np.clip(outer-mask, 0., 1.)
        # The neck below a face is often skin, so penalize it less there.
        surround *= np.where(y > .65*ry, .35, 1.)
        surround /= surround.sum()+1e-9
        contour = contour_weights(mask)
        clutter = np.ones_like(contour)/contour.size
        edge_k.append(.25*contour-.06*clutter)
        generic_edge_k.append(contour-.08*clutter)
        skin_k.append(inside-.65*surround)
        eye_shift = .07*r if vi == 2 else 0.
        sigma = max(.48, .125*r)
        eyes = .5*(gaussian_spot(y, x, -.38*rx, -.13*ry+eye_shift, sigma) +
                   gaussian_spot(y, x, .38*rx, -.13*ry-eye_shift, sigma))
        hair = gaussian_spot(y, x, 0, -.69*ry, max(.55, .23*r))
        mouth = gaussian_spot(y, x, 0, .40*ry, max(.45, .11*r))
        # Facial detail is not resolved for the tiniest heads.
        detail = min(1., r/3.)
        gray_k.append(-detail*(.38*eyes+.10*hair+.05*mouth))
    n = len(edge_k)
    prefix = "face%d" % si
    conv(prefix+"_se", edge_input, np.asarray(edge_k), np.zeros(n), padding=k//2)
    conv(prefix+"_ss", skin_input, np.asarray(skin_k)[:, None], np.full(n, -.60), padding=k//2)
    conv(prefix+"_sg", gray_input, -np.asarray(gray_k)[:, None], np.zeros(n), padding=k//2)
    add(prefix+"_skin_sum", [prefix+"_se", prefix+"_ss", prefix+"_sg"])
    relu(prefix+"_skin_r", prefix+"_skin_sum")
    face_skin_scalars.append(template_max(prefix+"_skin", prefix+"_skin_r", n, side))


# Render people from general anatomical proportions.  The fixed seed only
# selects a reproducible family of poses; it has no connection to the data.
rng = np.random.default_rng(84721)


def human_template(h, upper, pose):
    k = int(np.ceil(1.16*h))
    k += 1-k % 2
    c = k//2
    yy, xx = np.mgrid[:k, :k].astype(np.float64)
    y0, x0 = yy-c, xx-c
    aspect = [.80, 1.0, 1.18][pose % 3]
    x = x0/aspect
    y = y0
    mask = np.zeros((k, k))
    def capsule(a, b, radius):
        nonlocal mask
        mask = np.maximum(mask, capsule_mask(y, x, a[0]*h, a[1]*h,
                                             b[0]*h, b[1]*h, radius*h))
    lean = [-.025, .0, .025][(pose//3) % 3]
    head_x, head_y = lean-.005, -.38
    mask = np.maximum(mask, ellipse_mask(y, x, head_x*h, head_y*h,
                                         .066*h, .080*h))
    capsule((lean, -.29), (lean, -.19), .035)
    capsule((lean, -.22), (lean*.4, .12 if not upper else .28), .088)
    capsule((lean-.115, -.205), (lean+.115, -.205), .060)
    capsule((lean*.4-.070, .10), (lean*.4+.070, .10), .068)
    # Arms: down, bent, spread, and one/two-arm raised variants.
    arm_mode = pose % 6
    for side in [-1., 1.]:
        shoulder = (lean+side*.115, -.205)
        if arm_mode == 0:
            elbow, wrist = (side*.155, -.035), (side*.16, .145)
        elif arm_mode == 1:
            elbow, wrist = (side*.21, -.07), (side*.10, .055)
        elif arm_mode == 2:
            elbow, wrist = (side*.245, -.20), (side*.34, -.12)
        elif arm_mode == 3:
            elbow = (side*.22, -.27)
            wrist = (side*.18, -.43) if side > 0 else (side*.20, -.08)
        elif arm_mode == 4:
            elbow = (side*.20, -.12)
            wrist = (side*.07, -.28) if side < 0 else (side*.20, .04)
        else:
            elbow, wrist = (side*.24, -.30), (side*.27, -.43)
        capsule(shoulder, elbow, .037)
        capsule(elbow, wrist, .030)
    if upper:
        # Cropped hips and upper legs make this a head-and-torso model.
        capsule((-.052, .16), (-.065, .49), .052)
        capsule((.052, .16), (.065, .49), .052)
    else:
        leg_mode = (pose//2) % 4
        for side in [-1., 1.]:
            hip = (side*.053, .08)
            if leg_mode == 0:
                knee, ankle = (side*.058, .28), (side*.068, .48)
            elif leg_mode == 1:
                knee = (side*.11, .25)
                ankle = (side*.17, .46 if side < 0 else .39)
            elif leg_mode == 2:
                knee = (side*.13, .20)
                ankle = (side*.14, .46)
            else:
                knee = (side*.06, .30)
                ankle = (-side*.08, .47) if side > 0 else (side*.10, .43)
            capsule(hip, knee, .050)
            capsule(knee, ankle, .037)
    head = ellipse_mask(y, x, head_x*h, head_y*h, .057*h, .070*h)
    head /= head.sum()+1e-9
    return contour_weights(mask), head, k


body_skin_scalars = []
upper_skin_scalars = []
body_scales = [("full", False, 24, h) for h in [7., 10., 14.]]
body_scales += [("full", False, 12, h) for h in [9., 12.]]
body_scales += [("upper", True, 24, h) for h in [6., 9., 13.]]
body_scales += [("upper", True, 12, h) for h in [9., 12.]]
for bi, (kind, upper, side, h) in enumerate(body_scales):
    edge_input = "edge_capped" if side == 24 else "edge12_capped"
    skin_input = "skin_broad24" if side == 24 else "skin_broad12"
    contours, heads = [], []
    ks = []
    for p in range(8):
        co, he, k = human_template(h, upper, p)
        contours.append(co); heads.append(he); ks.append(k)
    k = max(ks)
    C = np.asarray([pad_kernel(a, k) for a in contours])
    H = np.asarray([pad_kernel(a, k) for a in heads])
    n = len(C)
    prefix = "body%d" % bi
    clutter = np.ones_like(C)/(C.shape[1]*C.shape[2]*C.shape[3])
    # Require a nontrivial contour response before adding anatomical head colour.
    conv(prefix+"_shape_pre", edge_input, C-.05*clutter, np.full(n, -.20), padding=k//2)
    relu(prefix+"_shape_r", prefix+"_shape_pre")
    conv(prefix+"_skin_e", prefix+"_shape_r", 1.5*np.eye(n)[..., None, None], np.full(n, -.60))
    conv(prefix+"_skin_h", skin_input, H[:, None], np.zeros(n), padding=k//2)
    add(prefix+"_skin_sum", [prefix+"_skin_e", prefix+"_skin_h"])
    relu(prefix+"_skin_r", prefix+"_skin_sum")
    skin_scalar = template_max(prefix+"_skin", prefix+"_skin_r", n, side)
    if upper:
        upper_skin_scalars.append(skin_scalar)
    else:
        body_skin_scalars.append(skin_scalar)


# Merge scalar maps into vectors using fixed one-hot dense matrices.  This lets
# the permitted maxpool operation take a maximum across different scales.
def merge_max(name, scalars):
    n = len(scalars)
    parts = []
    for i, scalar in enumerate(scalars):
        W = np.zeros((n, 1)); W[i, 0] = 1.
        dense(name+"_slot%d" % i, scalar, W, np.zeros(n))
        parts.append(name+"_slot%d" % i)
    add(name+"_vector", parts)
    reshape(name+"_spatial", name+"_vector", [1, 1, n])
    pool(name, "maxpool", name+"_spatial", [1, n])
    return name


face_skin = merge_max("face_skin_evidence", face_skin_scalars)
body_skin = merge_max("body_skin_evidence", body_skin_scalars)
upper_skin = merge_max("upper_skin_evidence", upper_skin_scalars)

# Cap each max so one pathological response cannot overwhelm all other cues.
def cap(name, inp, ceiling=.60):
    dense(name+"_hinge", inp, [[1.], [1.]], [0., -ceiling])
    relu(name+"_hinge_r", name+"_hinge")
    dense(name, name+"_hinge_r", [[1., -1.]], [0.])
    return name


features = [cap("cap_face_skin", face_skin),
            cap("cap_body_skin", body_skin),
            cap("cap_upper_skin", upper_skin)]
gains = [1.50, 1.30, 1.30]
weighted = []
for i, (feature, gain) in enumerate(zip(features, gains)):
    dense("weighted%d" % i, feature, [[gain]], [0.])
    weighted.append("weighted%d" % i)
best = merge_max("best_evidence", weighted)
dense("scores", best, [[-1.], [1.]], [.125, -.125])

network = {"layers": layers}
(OUT/"network.json").write_text(json.dumps(network, indent=2))
np.savez(OUT/"weights.npz", **weights)
print("wrote", len(layers), "layers and", sum(v.size for v in weights.values()), "weights to", OUT)
