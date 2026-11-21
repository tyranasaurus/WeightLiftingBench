#!/usr/bin/env python3
"""Writes network.json and weights.npz for a hand-designed person / no-person classifier.

Every number below is typed by hand from general knowledge (skin chromaticity ranges from the
skin-detection literature, which colours sit next to skin, simple soft logic).  Nothing is read
from files and nothing is fitted.  Usage: python3 build.py OUT_DIR
"""
import sys, os, json
import numpy as np


# ============================================================================ network builder
class Net:
    def __init__(self):
        self.layers, self.w, self.shapes = [], {}, {"image": (3, 96, 96)}

    def _add(self, L, shape):
        assert L["name"] not in self.shapes, L["name"]
        self.layers.append(L)
        self.shapes[L["name"]] = tuple(int(s) for s in shape)
        return L["name"]

    def conv(self, name, x, W, b, stride=1, pad=0):
        W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
        C, H, Wd = self.shapes[x]
        O, Ci, K, K2 = W.shape
        assert Ci == C and K == K2 and b.shape == (O,)
        self.w[name + ".W"], self.w[name + ".b"] = W.astype(np.float32), b.astype(np.float32)
        Ho, Wo = (H + 2 * pad - K) // stride + 1, (Wd + 2 * pad - K) // stride + 1
        return self._add(dict(name=name, op="conv", inputs=[x], out=int(O), kernel=[int(K), int(K)],
                              stride=int(stride), padding=int(pad)), (O, Ho, Wo))

    def dense(self, name, x, W, b):
        W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
        n = int(np.prod(self.shapes[x]))
        assert W.shape[1] == n and b.shape == (W.shape[0],)
        self.w[name + ".W"], self.w[name + ".b"] = W.astype(np.float32), b.astype(np.float32)
        return self._add(dict(name=name, op="dense", inputs=[x], out=int(W.shape[0])), (W.shape[0],))

    def relu(self, name, x):
        return self._add(dict(name=name, op="relu", inputs=[x]), self.shapes[x])

    def pool(self, name, x, op, size, stride):
        C, H, W = self.shapes[x]
        (sh, sw), (th, tw) = size, stride
        return self._add(dict(name=name, op=op, inputs=[x], size=[int(sh), int(sw)], stride=[int(th), int(tw)]),
                         (C, (H - sh) // th + 1, (W - sw) // tw + 1))

    def gavg(self, name, x):
        C, H, W = self.shapes[x]
        return self.pool(name, x, "avgpool", [H, W], [H, W])

    def add(self, name, xs):
        s = self.shapes[xs[0]]
        assert all(self.shapes[x] == s for x in xs)
        return self._add(dict(name=name, op="add", inputs=list(xs)), s)

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "network.json"), "w") as f:
            json.dump({"layers": self.layers}, f, indent=1)
        np.savez(os.path.join(out_dir, "weights.npz"), **self.w)


# ============================================================================ colour boxes
# Chromaticity r = R/S, g = G/S, b = B/S with S = R+G+B.  A rule "a*r + b*g + c*b >= t" is the
# brightness-independent linear rule  a*R + b*G + c*B - t*S >= 0  on the raw pixel.
def chroma_rule(a, t):
    return np.asarray(a, np.float64) - t, 0.0  # coefficients on (R, G, B), constant


def box_rules(r=None, g=None, b=None, r_minus_g=None, g_minus_b=None, min_red=0.25, soft=0.04):
    """Soft chromaticity box: list of (coef RGB, constant, softness) rules, each meaning coef.RGB + const >= 0."""
    rules = []
    for vec, rng in (([1, 0, 0], r), ([0, 1, 0], g), ([0, 0, 1], b), ([1, -1, 0], r_minus_g), ([0, 1, -1], g_minus_b)):
        if rng is None:
            continue
        lo, hi = rng
        if lo is not None:
            rules.append((*chroma_rule(vec, lo), soft))
        if hi is not None:
            rules.append((*chroma_rule(-np.asarray(vec, np.float64), -hi), soft))
    rules.append((np.array([1.0, 0.0, 0.0]), -min_red, 0.08))   # R >= min_red: colour of dark pixels is unreliable
    return rules


# Skin, typed from the published skin-colour clusters of web photos (normalised rg / YCbCr studies):
# r ~ 0.39-0.52, g ~ 0.285-0.335, b ~ 0.20-0.295, red clearly above green, green above blue.
SKIN = box_rules(r=(0.39, 0.52), g=(0.285, 0.335), b=(0.20, 0.295), r_minus_g=(0.07, None), g_minus_b=(0.02, None))
# Matched control: the warm colours just outside the skin box on its yellow side (light wood, bread,
# cardboard, golden fur), its orange / saturated-brown side (dark wood, brown fur, giraffe) and its
# red side (brick, terracotta, red-brown leather).  Materials whose colour spreads over this
# neighbourhood light up skin and control alike; real skin, whose colour cluster is tight, lights up
# the skin box only.  The control boxes start a little beyond the skin box edges and have sharper
# edges, so skin near its own edges is not counted as control.
WARM_YELLOW = box_rules(r=(0.39, 0.52), g=(0.34, 0.385), b=(0.12, None), r_minus_g=(0.03, None), soft=0.025)
WARM_ORANGE = box_rules(r=(0.42, 0.58), g=(0.27, 0.34), b=(0.10, 0.19), soft=0.025)
WARM_RED = box_rules(r=(0.50, 0.62), g=(0.22, 0.28), b=(0.12, 0.28), soft=0.025)


LUMA = np.array([0.299, 0.587, 0.114])   # Rec. 601 luminance weights

# Snow: bright (Y >= 0.65) and neutral to bluish (blue at least red, green close to red, blue not far
# above green: shaded snow is bluish, sunlit snow neutral, while indoor whites under lamps are warm).
SNOW = [
    (LUMA, -0.65, 0.08),
    (np.array([-1.0, 0.0, 1.0]), 0.01, 0.03),    # B - R >= -0.01
    (np.array([1.0, 0.0, -1.0]), 0.25, 0.05),    # B - R <= 0.25
    (np.array([-1.0, 1.0, 0.0]), 0.03, 0.03),    # G - R >= -0.03
    (np.array([1.0, -1.0, 0.0]), 0.08, 0.03),    # G - R <= 0.08
    (np.array([0.0, -1.0, 1.0]), 0.02, 0.03),    # B - G >= -0.02
    (np.array([0.0, 1.0, -1.0]), 0.16, 0.03),    # B - G <= 0.16
]


def soft_box_layers(net, name, x, boxes):
    """Memberships of several soft boxes: relu(1 - sum of relu(rule violations))."""
    rows = [(bi, coef, const, soft) for bi, bx in enumerate(boxes) for (coef, const, soft) in bx]
    C = net.shapes[x][0]
    W = np.zeros((len(rows), C, 1, 1)); b = np.zeros(len(rows))
    for i, (bi, coef, const, soft) in enumerate(rows):
        W[i, :3, 0, 0] = -coef / soft
        b[i] = -const / soft
    v = net.relu(name + "_viol_r", net.conv(name + "_viol", x, W, b))
    A = np.zeros((len(boxes), len(rows), 1, 1))
    for i, (bi, _, _, _) in enumerate(rows):
        A[bi, i, 0, 0] = -1.0
    return net.relu(name + "_r", net.conv(name, v, A, np.ones(len(boxes))))


# ============================================================================ the network
def compact_skin_kernel():
    """5x5-cell template on channels [skin, yellow, orange, red control]: net skin share of the centre
    cell, minus 2 x the warm share (skin + controls) of the ring of 16 cells two steps away, minus 0.1.
    Positive for a skin patch surrounded by non-warm colours (hair, clothes, background); zero inside
    large warm regions (walls, wood, animal bodies) and along their edges."""
    W = np.zeros((1, 4, 5, 5)); b = np.array([-0.1])
    ring = np.zeros((5, 5)); ring[[0, 4], :] = 1; ring[:, [0, 4]] = 1; ring /= ring.sum()
    W[0, 0, 2, 2] += 1.0
    W[0, 1:4, 2, 2] -= 1.0
    for c in range(4):
        W[0, c] -= 2.0 * ring
    return W, b


def build():
    net = Net()

    # ---- colour of 2x2 blocks (the JPEG files keep colour at about this resolution)
    rgb2 = net.pool("rgb2", "image", "avgpool", [2, 2], [2, 2])                   # (3, 48, 48)

    # ---- memberships [skin, warm-yellow, warm-orange, warm-red control] per block
    mem = soft_box_layers(net, "colour", rgb2, [SKIN, WARM_YELLOW, WARM_ORANGE, WARM_RED])  # (4, 48, 48)
    frac = net.gavg("mem_avg", mem)                                               # (4,1,1) shares of the photo

    # ---- local skin evidence on 8x8-pixel cells: skin share minus control share, beyond a 0.2 margin.
    # Computing it per cell keeps a face from being cancelled by wood or walls elsewhere in the photo.
    cell = net.pool("cell", mem, "avgpool", [4, 4], [4, 4])                        # (4, 12, 12)
    Wd = np.array([1.0, -1.0, -1.0, -1.0]).reshape(1, 4, 1, 1)
    net_skin = net.relu("cell_net_r", net.conv("cell_net", cell, Wd, np.array([-0.20])))
    e_local = net.gavg("net_skin_avg", net_skin)                                  # (1,1,1)

    # ---- compact skin patches at three cell sizes (4, 8, 16 px: hands and faces of ~4-40 px)
    Wk, bk = compact_skin_kernel()
    counts = []
    for cs, k in ((4, 2), (8, 4), (16, 8)):
        gname = f"cells{cs}"
        gx = net.pool(gname, mem, "avgpool", [k, k], [k, k])                      # (4, 48/k, 48/k)
        t = net.relu(gname + "_compact_r", net.conv(gname + "_compact", gx, Wk, bk, stride=1, pad=2))
        counts.append((net.gavg(gname + "_avg", t), int(np.prod(net.shapes[gx][1:]))))   # mean -> x cells = count

    # ---- snow underfoot: share of snow-coloured blocks in each quarter of the photo, top to bottom
    snow = soft_box_layers(net, "snow", rgb2, [SNOW])                             # (1, 48, 48)
    snow_q = net.pool("snow_quarters", snow, "avgpool", [12, 48], [12, 48])       # (1, 4, 1)

    # ---- black-and-white photo detector: the largest |R-G| + |G-B| of any block is near zero
    Wc = np.zeros((4, 3, 1, 1))
    for j, (a, bb) in enumerate([(0, 1), (1, 0), (1, 2), (2, 1)]):
        Wc[j, a, 0, 0] = 1.0; Wc[j, bb, 0, 0] = -1.0
    cd = net.relu("cdiff_r", net.conv("cdiff", rgb2, Wc, np.zeros(4)))
    cdsum = net.conv("cdiff_sum", cd, np.ones((1, 4, 1, 1)), np.zeros(1))
    C_, H_, W_ = net.shapes[cdsum]
    cdiff = net.pool("cdiff_max", cdsum, "maxpool", [H_, W_], [H_, W_])        # (1,1,1)

    # ---- evidence units (ramps made from relu pairs) -> person score
    U = ["local", "local_top", "compact", "compact_top", "large", "gray", "snow", "snow_top", "white_top"]
    H = len(U); ix = {u: i for i, u in enumerate(U)}
    parts = []
    # local net skin: ramp 0.1% -> 1.1% of the photo
    D = np.zeros((H, 1)); bb = np.zeros(H)
    D[ix["local"], 0] = D[ix["local_top"], 0] = 1 / 0.01; bb[ix["local"]] = -0.1; bb[ix["local_top"]] = -1.1
    parts.append(net.dense("ev_local", e_local, D, bb))
    # compact patch count n (summed over the cell sizes; the 16-px cells, where animal bodies and
    # plates of food look as compact as close-up faces, count half): ramp 0.5 -> 2.0
    for gi, (src, ncell) in enumerate(counts):
        D = np.zeros((H, 1)); bb = np.zeros(H)
        D[ix["compact"], 0] = D[ix["compact_top"], 0] = ncell / 1.5 * (0.5 if gi == 2 else 1.0)
        if gi == 0:
            bb[ix["compact"]] = -0.5 / 1.5; bb[ix["compact_top"]] = -2.0 / 1.5
        parts.append(net.dense(f"ev_compact{gi}", src, D, bb))
    # skin share above 40% of the photo (a skin-coloured wall or object rather than people)
    D = np.zeros((H, 4)); bb = np.zeros(H); D[ix["large"], 0] = 1 / 0.3; bb[ix["large"]] = -0.4 / 0.3
    parts.append(net.dense("ev_frac", frac, D, bb))
    # black-and-white photo: 1 - (largest block |R-G| + |G-B|) / 0.1
    D = np.zeros((H, 1)); bb = np.zeros(H); D[ix["gray"], 0] = -1 / 0.1; bb[ix["gray"]] = 1.0
    parts.append(net.dense("ev_gray", cdiff, D, bb))
    # snow: bottom-half snow share, ramp 0.5 -> 0.8; white_top: top-quarter snow share above 0.7
    # (an all-white frame: overcast sky, white room, close-up of a white object)
    D = np.zeros((H, 4)); bb = np.zeros(H)
    D[ix["snow"], 2:] = D[ix["snow_top"], 2:] = 0.5 / 0.3; bb[ix["snow"]] = -0.5 / 0.3; bb[ix["snow_top"]] = -0.8 / 0.3
    D[ix["white_top"], 0] = 1 / 0.15; bb[ix["white_top"]] = -0.7 / 0.15
    parts.append(net.dense("ev_snow", snow_q, D, bb))
    hid = net.relu("evidence", net.add("ev_sum", parts))
    # second stage: pass the skin units through, snow scene = max(0, snow ramp - white_top)
    V = ["local", "local_top", "compact", "compact_top", "large", "gray", "snow_scene"]
    E2 = np.zeros((len(V), H))
    for k, u in enumerate(V[:-1]):
        E2[k, ix[u]] = 1.0
    E2[-1, ix["snow"]] = 1.0; E2[-1, ix["snow_top"]] = -1.0; E2[-1, ix["white_top"]] = -1.0
    hid2 = net.relu("evidence2", net.dense("ev_combine", hid, E2, np.zeros(len(V))))
    w = {"local": 0.7, "local_top": -0.7, "compact": 1.2, "compact_top": -1.2, "large": -0.5, "gray": 1.05,
         "snow_scene": 1.2}
    Wf = np.zeros((2, len(V))); Wf[1] = [w[u] for u in V]      # class 0 score fixed at 0
    net.dense("scores", hid2, Wf, np.array([0.0, -1.0]))
    return net


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    build().save(out)
