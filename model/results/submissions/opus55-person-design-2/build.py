"""Builds a hand-designed person / no-person network.  python3 build.py OUT_DIR
Every weight comes from formulas in this file (colour half-spaces, binomial blur kernels,
elliptical templates, hand-chosen thresholds); nothing is read from disk and nothing is fitted to data.
Output 1 - output 0 = summed person evidence - THETA; output 0 is always 0."""
import json, sys
from pathlib import Path
import numpy as np


class Net:
    def __init__(self):
        self.layers, self.w, self.shape = [], {}, {"image": (3, 96, 96)}

    def _add(self, L, shape):
        self.layers.append(L); self.shape[L["name"]] = shape; return L["name"]

    def conv(self, name, inp, W, b, stride=1, pad=0):
        W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
        c, h, w = self.shape[inp]; o, ci, k, k2 = W.shape
        assert ci == c and k == k2 and b.shape == (o,), (name, W.shape, self.shape[inp])
        self.w[name + ".W"], self.w[name + ".b"] = W, b
        ho, wo = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
        return self._add({"name": name, "op": "conv", "inputs": [inp], "out": o, "kernel": [k, k],
                          "stride": stride, "padding": pad}, (o, ho, wo))

    def dense(self, name, inp, W, b):
        W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
        assert W.shape == (len(b), int(np.prod(self.shape[inp]))), (name, W.shape, self.shape[inp])
        self.w[name + ".W"], self.w[name + ".b"] = W, b
        return self._add({"name": name, "op": "dense", "inputs": [inp], "out": len(b)}, (len(b),))

    def relu(self, name, inp):
        return self._add({"name": name, "op": "relu", "inputs": [inp]}, self.shape[inp])

    def pool(self, name, inp, op, size, stride):
        c, h, w = self.shape[inp]
        return self._add({"name": name, "op": op, "inputs": [inp], "size": list(size), "stride": list(stride)},
                         (c, (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1))

    def add(self, name, inps):
        assert all(self.shape[i] == self.shape[inps[0]] for i in inps), [self.shape[i] for i in inps]
        return self._add({"name": name, "op": "add", "inputs": list(inps)}, self.shape[inps[0]])

    def save(self, out):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        (out / "network.json").write_text(json.dumps({"layers": self.layers}, indent=1))
        np.savez(out / "weights.npz", **{k: v.astype(np.float32) for k, v in self.w.items()})


# ----------------------------------------------------------------------------------------------
# 1. Colour memberships, per pixel.  A membership is a fuzzy AND (Lukasiewicz: relu(sum - (n-1)))
#    of ramps clamp((m - lo) / (hi - lo), 0, 1), where every margin m is linear in (R, G, B).
#    Ratios such as hue and saturation give linear half-spaces because they are homogeneous.
# ----------------------------------------------------------------------------------------------
R, G, B = np.eye(3)
Y = np.array([0.299, 0.587, 0.114])


def hue_ge(deg):   # hue (red-max sector) >= deg  <=>  (G - B) - deg/60 (R - B) >= 0
    return (G - B) - deg / 60.0 * (R - B)


def hue_le(deg):
    return deg / 60.0 * (R - B) - (G - B)


def sat_ge(s):     # (R - B) / R >= s  (R is max and B min in the sectors used here)
    return (R - B) - s * R


def sat_le(s):
    return s * R - (R - B)


# each fuzzy membership: list of (margin vector, ramp low, ramp high); value = relu(sum ramps - (n-1)),
# a Lukasiewicz AND of the ramps clamp((m.x - low) / (high - low), 0, 1)
MEMBERSHIPS = {
    # typical photographed human skin: hue about 4..28 deg, saturation 0.2..0.64, not too dark
    "skin_core": [(hue_ge(4), -0.005, 0.015), (hue_le(28), 0.0, 0.02),
                  (sat_ge(0.20), 0.0, 0.03), (sat_le(0.64), 0.0, 0.04), (R, 0.20, 0.30)],
    # anything skin-like (wood, tan fur, sand, bread, blond hair...): wider hue and saturation box,
    # not dark (so that dark brown hair above a face does not count as skin-like surroundings)
    "skin_loose": [(G - B, -0.03, 0.0), (hue_le(42), 0.0, 0.02),
                   (sat_ge(0.10), 0.0, 0.03), (sat_le(0.80), 0.0, 0.03), (R, 0.22, 0.32)],
    # snow in daylight: bright, nearly grey, blue channel at least as high as red
    "snow": [(Y, 0.66, 0.74), (B - R, 0.0, 0.04), (0.16 - (B - R), 0.0, 0.04), (0.10 - (G - B), 0.0, 0.04)],
}
LINEAR = {"dark": (-Y, 1.0)}     # darkness 1 - Y (zero padding then reads as bright)


def colour_layers(net, groups_out):
    """groups_out: list of lists of output names (each list becomes one 1x1 conv output tensor).
    Fuzzy AND of ramps clamp(z_i, 0, 1):  relu(sum_i clamp(z_i) - (n-1)) == relu(1 - sum_i relu(1 - z_i)),
    so one channel u_i = relu(1 - z_i) per constraint is enough."""
    W1, b1, idx_of = [], [], {}
    for name, cons in MEMBERSHIPS.items():
        idx = []
        for (m, lo, hi) in cons:
            s = 1.0 / (hi - lo)           # z = (m.x - lo) / (hi - lo);  channel 1 - z
            W1.append(-m * s); b1.append(1.0 + lo * s); idx.append(len(W1) - 1)
        idx_of[name] = idx
    for name, (m, c) in LINEAR.items():     # values >= 0 for inputs in [0, 1], so relu keeps them
        W1.append(m); b1.append(c); idx_of[name] = [len(W1) - 1]
    W1 = np.array(W1)[:, :, None, None]
    h = net.relu("col_r", net.conv("col", "image", W1, b1))
    outs = []
    for gi, names in enumerate(groups_out):
        W2 = np.zeros((len(names), len(b1), 1, 1)); b2 = np.zeros(len(names))
        for j, name in enumerate(names):
            if name in LINEAR:
                W2[j, idx_of[name][0], 0, 0] = 1.0
            else:
                W2[j, idx_of[name], 0, 0] = -1.0; b2[j] = 1.0
        outs.append(net.relu(f"mem{gi}", net.conv(f"mem{gi}_c", h, W2, b2)))
    return outs


# ----------------------------------------------------------------------------------------------
# 2. Pyramid: anti-aliased downsampling by binomial kernels (per channel).
# ----------------------------------------------------------------------------------------------
def down(net, name, inp, factor, shifted=False):
    """Blur and subsample.  shifted=True moves the sampling grid by half an output pixel up-left in both
    directions, so that a blob straddling the corners of the plain grid is caught centred on the shifted one."""
    c = net.shape[inp][0]
    if factor == 2:   # binomial 1 3 3 1
        k1, stride, pad = (np.array([1, 3, 3, 1, 0], float), 2, 2) if shifted else (np.array([1, 3, 3, 1], float), 2, 1)
    else:             # factor 3: 1 2 3 2 1 centred on each 3x3 block; shifted: 1 2 3 3 2 1 centred 1.5 px earlier
        k1, stride, pad = (np.array([1, 2, 3, 3, 2, 1], float), 3, 3) if shifted else (np.array([1, 2, 3, 2, 1], float), 3, 1)
    k2 = np.outer(k1, k1); k2 /= k2.sum()
    W = np.zeros((c, c, len(k1), len(k1)))
    for i in range(c):
        W[i, i] = k2
    return net.conv(name, inp, W, np.zeros(c), stride, pad)


# ----------------------------------------------------------------------------------------------
# 3. Part templates at one pyramid level (inputs: A = skin_core, darkness; B = 1 - skin_core, 1 - skin_loose).
#  Head:
#    El   = mean skin_core over the lower face (face ellipse with half width a, half height 1.25 a,
#           below 0.2 half heights above its centre: cheeks, nose, mouth, chin)
#    hair = mean darkness over the top of the head (a band 0.6 .. 1.35 half heights above the centre)
#           minus mean darkness over the lower face, minus a margin, clamped to [0, HCAP]
#    T    = mean skin_core in the ring sector above the head;  L, R = mean skin_loose in the ring
#           sectors left and right of the face (neck and shoulders below are left out).  A skin
#           coloured sector means the blob is part of a larger skin-like surface, which vetoes it.
#           Sectors are read from the non-skin maps B, so the zero padding outside the photo counts
#           as skin-like (unknown) and the corners of a large skin-coloured surface are vetoed too.
#    head = relu(El + K * hair - P * sum relu(sector - t) - tau)
#  Limb (4 orientations): a bar 1 a wide and 4 a long, mean skin_core inside, minus a veto for
#    skin_loose on either long side.
#  All weights are area fractions of these shapes, computed by supersampling each pixel.
# ----------------------------------------------------------------------------------------------
def _grid(k, ss=16):
    r = (k - 1) / 2.0
    off = (np.arange(ss) + 0.5) / ss - 0.5
    for iy in range(k):
        for ix in range(k):
            yy = (iy - r) + off[:, None] + 0 * off[None, :]
            xx = (ix - r) + off[None, :] + 0 * off[:, None]
            yield iy, ix, yy, xx


def head_masks(a, k):
    bh = 1.25 * a
    out = {n: np.zeros((k, k)) for n in ("T", "L", "R", "H", "El")}
    for iy, ix, yy, xx in _grid(k):
        rho = np.sqrt((xx / a) ** 2 + (yy / bh) ** 2)
        th = np.degrees(np.arctan2(xx, -yy))            # 0 = straight up, +-180 = down
        ring = (rho >= 1.35) & (rho <= 2.1)
        out["T"][iy, ix] = np.mean(ring & (np.abs(th) < 40))
        out["L"][iy, ix] = np.mean(ring & (th <= -45) & (th > -115))
        out["R"][iy, ix] = np.mean(ring & (th >= 45) & (th < 115))
        out["H"][iy, ix] = np.mean((yy <= -0.6 * bh) & (yy >= -1.35 * bh) & (np.abs(xx) <= 0.85 * a))
        out["El"][iy, ix] = np.mean((rho <= 1.0) & (yy >= -0.2 * bh))
    return {n: m / m.sum() for n, m in out.items()}


def limb_masks(a, k, deg):
    t = np.radians(deg); hw, hl = 0.5 * a, 2.0 * a
    out = {n: np.zeros((k, k)) for n in ("In", "F1", "F2")}
    for iy, ix, yy, xx in _grid(k):
        par = xx * np.cos(t) + yy * np.sin(t); perp = -xx * np.sin(t) + yy * np.cos(t)
        along = np.abs(par) <= hl
        out["In"][iy, ix] = np.mean(along & (np.abs(perp) <= hw))
        out["F1"][iy, ix] = np.mean(along & (perp >= hw + 0.6) & (perp <= hw + 1.4))
        out["F2"][iy, ix] = np.mean(along & (perp <= -hw - 0.6) & (perp >= -hw - 1.4))
    return {n: m / m.sum() for n, m in out.items()}


HEAD = dict(a=1.1, k=7, top_t=0.2, side_t=0.3, P=2.0, Pside=1.5, hair_margin=0.02, K=1.5, hcap=0.25, tau=0.55)
LIMB = dict(a=1.1, k=7, side_t=0.25, P=2.0, tau=0.55, orient=(0, 45, 90, 135))


def part_detector(net, name, inA, inB):
    """inA: (skin_core, darkness); inB: (1 - skin_core, 1 - skin_loose).  The ring sectors and limb flanks read
    the 'non-skin' maps, so the zero padding outside the photo counts as skin-like there: a large skin-coloured
    surface cannot pass as a head or limb at the corners and edges of the photo.
    Returns a map with channels [head, limb_0, limb_45, limb_90, limb_135]."""
    p, q = HEAD, LIMB; k = p["k"]
    m = head_masks(p["a"], k)
    n = 6 + 3 * len(q["orient"])
    WA = np.zeros((n, 2, k, k)); WB = np.zeros((n, 2, k, k)); b = np.zeros(n)
    WA[0, 0] = m["El"]
    WB[1, 0] = -m["T"]; b[1] = 1.0 - p["top_t"]          # mean_T(skin_core) - t = 1 - mean_T(1 - skin_core) - t
    WB[2, 1] = -m["L"]; b[2] = 1.0 - p["side_t"]
    WB[3, 1] = -m["R"]; b[3] = 1.0 - p["side_t"]
    WA[4, 1] = m["H"] - m["El"]; b[4] = -p["hair_margin"]
    WA[5, 1] = m["H"] - m["El"]; b[5] = -p["hair_margin"] - p["hcap"]
    W2 = np.zeros((1 + len(q["orient"]), n, 1, 1)); b2 = np.zeros(len(W2))
    W2[0, :6, 0, 0] = [1.0, -p["P"], -p["Pside"], -p["Pside"], p["K"], -p["K"]]; b2[0] = -p["tau"]
    for j, deg in enumerate(q["orient"]):
        lm = limb_masks(q["a"], k, deg); c = 6 + 3 * j
        WA[c, 0] = lm["In"]; WB[c + 1, 1] = -lm["F1"]; WB[c + 2, 1] = -lm["F2"]
        b[c + 1] = b[c + 2] = 1.0 - q["side_t"]
        W2[1 + j, c:c + 3, 0, 0] = [1.0, -q["P"], -q["P"]]; b2[1 + j] = -q["tau"]
    s = net.add(name + "_s", [net.conv(name + "_pa", inA, WA, b, 1, k // 2), net.conv(name + "_pb", inB, WB, np.zeros(n), 1, k // 2)])
    h = net.relu(name + "_pr", s)
    return net.relu(name + "_r", net.conv(name, h, W2, b2))


LEVELS = [48, 32, 24, 16, 12, 8, 6, 4]
THETA = 0.10                         # evidence needed for "person" (summed over levels and parts)
BAND_W = [1.0, 1.0, 0.6]             # head evidence by third of the photo (top, middle, bottom)
LIMB_W = 0.5
SNOW_W, SNOW_T = 1.0, 0.30
GRAY_W = 0.25        # black-and-white photos (all channels equal): the colour cues cannot see skin there;
                     # such photos are mostly portraits, street scenes and old family pictures


def build():
    net = Net()
    mem, snow = colour_layers(net, [["skin_core", "skin_loose", "dark"], ["snow"]])
    # A = (skin_core, darkness);  B = (1 - skin_core, 1 - skin_loose)
    WA = np.zeros((2, 3, 1, 1)); WA[0, 0] = 1.0; WA[1, 2] = 1.0
    WB = np.zeros((2, 3, 1, 1)); WB[0, 0] = -1.0; WB[1, 1] = -1.0
    memA = net.relu("memA", net.conv("memA_c", mem, WA, [0.0, 0.0]))
    memB = net.relu("memB", net.conv("memB_c", mem, WB, [1.0, 1.0]))
    lv, lvs = {}, {}
    src_of = {48: (None, 2), 32: (None, 3), 24: (48, 2), 16: (32, 2), 12: (24, 2), 8: (16, 2), 6: (12, 2), 4: (8, 2)}
    for L in LEVELS:
        s, f = src_of[L]
        for t, base in (("A", memA), ("B", memB)):
            inp = base if s is None else lv[s, t]
            lv[L, t] = down(net, f"d{L}{t}", inp, f)
            lvs[L, t] = down(net, f"d{L}{t}s", inp, f, shifted=True)
    outs = []
    for L in LEVELS:
        g = []
        for tag, m in (("", lv), ("s", lvs)):
            fm = part_detector(net, f"part{L}{tag}", m[L, "A"], m[L, "B"])
            n = net.shape[fm][1]
            bh = n // 3 if n >= 12 else n
            g.append(net.pool(f"pmax{L}{tag}", fm, "maxpool", (bh, n), (bh, n)))     # (5, bands, 1)
        nb = net.shape[g[0]][1]; d = 5 * nb
        assert net.shape[g[1]] == net.shape[g[0]]
        # max of the plain and shifted grids: a + relu(b - a)
        dba = net.add(f"pd{L}", [net.dense(f"pd{L}a", g[0], -np.eye(d), np.zeros(d)),
                                 net.dense(f"pd{L}b", g[1], np.eye(d), np.zeros(d))])
        r = net.relu(f"pd{L}r", dba)
        wv = np.zeros((5, nb))
        wv[0] = BAND_W if nb == 3 else 1.0
        wv[1:] = LIMB_W
        Wo = np.zeros((2, d)); Wo[1] = wv.reshape(-1)
        outs.append(net.dense(f"o_part{L}", g[0], Wo, [0.0, 0.0]))
        outs.append(net.dense(f"o_part{L}x", r, Wo, [0.0, 0.0]))
    # snow: share of snow pixels in the lower two thirds of the photo (16 x 16 blocks)
    s6 = net.pool("snow6", snow, "avgpool", (16, 16), (16, 16))        # (1, 6, 6)
    Ws = np.zeros((1, 36)); Ws[0, 12:] = 1.0 / 24
    sf = net.relu("snowfrac", net.dense("snowfrac_c", s6, Ws, [-SNOW_T]))
    outs.append(net.dense("o_snow", sf, np.array([[0.0], [SNOW_W]]), [0.0, -THETA]))
    # greyscale: mean colour difference |R-G| + |G-B| over the photo is ~0 (JPEG rounding only)
    cd = np.array([R - G, G - R, G - B, B - G])[:, :, None, None]
    cdm = net.pool("chroma_m", net.relu("chroma_r", net.conv("chroma", "image", cd, np.zeros(4))), "avgpool", (96, 96), (96, 96))
    z = net.relu("gray_r", net.dense("gray_z", cdm, -np.ones((2, 4)) / 0.006, [0.012 / 0.006, 0.012 / 0.006 - 1.0]))
    outs.append(net.dense("o_gray", z, np.array([[0.0, 0.0], [GRAY_W, -GRAY_W]]), [0.0, 0.0]))
    net.add("score", outs)
    return net


if __name__ == "__main__":
    build().save(sys.argv[1] if len(sys.argv) > 1 else ".")
