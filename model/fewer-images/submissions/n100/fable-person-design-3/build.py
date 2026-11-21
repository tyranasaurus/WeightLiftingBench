"""Hand-built person / no-person network for 96x96 COCO photos.

Every number below is typed from general knowledge (skin-colour rules from the colour-science
literature, face and body proportions, how photos of people are composed).  Nothing is read from
data and nothing is fitted: the script reads no files and uses no randomness.

Usage: python3 build.py OUT_DIR   -> writes OUT_DIR/network.json and OUT_DIR/weights.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

IMG = 96


# --------------------------------------------------------------------------------------
# tiny graph builder
# --------------------------------------------------------------------------------------
class Net:
    def __init__(self):
        self.layers, self.w, self.shape = [], {}, {"image": (3, IMG, IMG)}

    def _add(self, name, op, inputs, oshape, **kw):
        assert name not in self.shape, name
        L = {"name": name, "op": op, "inputs": list(inputs)}
        L.update(kw)
        self.layers.append(L)
        self.shape[name] = tuple(int(v) for v in oshape)
        return name

    def conv(self, name, inp, W, b=None, stride=1, pad=0):
        W = np.asarray(W, np.float32)
        o, c, k, k2 = W.shape
        assert k == k2 and self.shape[inp][0] == c, (name, W.shape, self.shape[inp])
        b = np.zeros(o, np.float32) if b is None else np.asarray(b, np.float32).reshape(o)
        _, h, w = self.shape[inp]
        ho, wo = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
        self.w[name + ".W"], self.w[name + ".b"] = W, b
        return self._add(name, "conv", [inp], (o, ho, wo), out=int(o), kernel=[int(k), int(k)], stride=int(stride), padding=int(pad))

    def lin(self, name, inp, M, b=None):
        """1x1 conv with matrix M (out, in)."""
        M = np.asarray(M, np.float32)
        return self.conv(name, inp, M[:, :, None, None], b)

    def dense(self, name, inp, W, b=None):
        W = np.asarray(W, np.float32)
        o, n = W.shape
        assert n == int(np.prod(self.shape[inp])), (name, W.shape, self.shape[inp])
        b = np.zeros(o, np.float32) if b is None else np.asarray(b, np.float32).reshape(o)
        self.w[name + ".W"], self.w[name + ".b"] = W, b
        return self._add(name, "dense", [inp], (o,), out=int(o))

    def relu(self, name, inp):
        return self._add(name, "relu", [inp], self.shape[inp])

    def pool(self, name, inp, kind, size, stride=None):
        stride = size if stride is None else stride
        c, h, w = self.shape[inp]
        ho, wo = (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1
        return self._add(name, kind, [inp], (c, ho, wo), size=[int(size[0]), int(size[1])], stride=[int(stride[0]), int(stride[1])])

    def add(self, name, inps):
        s = self.shape[inps[0]]
        assert all(self.shape[i] == s for i in inps), (name, [self.shape[i] for i in inps])
        return self._add(name, "add", inps, s)

    def reshape(self, name, inp, shape):
        assert int(np.prod(shape)) == int(np.prod(self.shape[inp])), (name, shape, self.shape[inp])
        return self._add(name, "reshape", [inp], shape, shape=[int(v) for v in shape])

    def const_map(self, name, src, values):
        """A constant (c, h, w) map: dense layer with zero weights and the map as bias."""
        values = np.asarray(values, np.float32)
        if values.ndim == 2:
            values = values[None]
        n = int(np.prod(self.shape[src]))
        d = self.dense(name + "_d", src, np.zeros((values.size, n), np.float32), values.reshape(-1))
        return self.reshape(name, d, values.shape)

    def clip01(self, name, inp, row, bias):
        """From channels of `inp`, t = row . x + bias, return a 2-channel relu tensor [relu(t), relu(t-1)];
        clip(t, 0, 1) is channel0 - channel1."""
        row = np.asarray(row, np.float32).reshape(1, -1)
        self.lin(name + "_l", inp, np.concatenate([row, row], 0), [bias, bias - 1.0])
        return self.relu(name, name + "_l")


# --------------------------------------------------------------------------------------
# anti-aliased shape masks on a k x k kernel grid (cell (i, j) is centred at x = j - c, y = i - c)
# --------------------------------------------------------------------------------------
SS = 8  # sub-samples per cell side


def _grid(k):
    c = (k - 1) / 2.0
    t = (np.arange(k * SS) + 0.5) / SS - 0.5 - c
    return np.meshgrid(t, t)  # x, y  (y grows downward)


def _cover(mask, k):
    return mask.reshape(k, SS, k, SS).mean(axis=(1, 3))


def box_mask(k, x0, x1, y0, y1):
    x, y = _grid(k)
    return _cover(((x >= x0) & (x < x1) & (y >= y0) & (y < y1)).astype(np.float64), k)


def unit(m):
    """Scale a mask so that its entries sum to 1 (a mean over the mask)."""
    s = m.sum()
    assert s > 1e-6
    return m / s


def blob_template(k, a, b, r_in, r_out, w_bottom):
    """Face-like blob on a skin-density map: +mean over an ellipse (semi-axes a, b) minus the means over the
    top, left and right sectors of the surrounding ring, minus w_bottom * the bottom sector (neck, chest)."""
    x, y = _grid(k)
    rr = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    ang = np.degrees(np.arctan2(y / b, x / a))  # 0 = right, 90 = down, -90 = up
    centre = rr <= 1.0
    ring = (rr >= r_in) & (rr <= r_out)
    right = ring & (np.abs(ang) < 45)
    left = ring & (np.abs(ang) > 135)
    bottom = ring & (ang >= 45) & (ang <= 135)
    top = ring & (ang <= -45) & (ang >= -135)
    T = unit(_cover(centre.astype(float), k))
    neg = np.zeros_like(T)
    for m, wgt in ((top, 1.0), (left, 1.0), (right, 1.0), (bottom, w_bottom)):
        neg = neg + wgt * unit(_cover(m.astype(float), k))
    return T - neg, neg


def inside_sum(kern, n, pad):
    """Sum of the kernel weights that fall inside an n x n map, for every output position of a stride-1 conv
    with this padding (what the conv gives on an all-ones map)."""
    k = kern.shape[0]
    ones = np.pad(np.ones((n, n)), pad)
    m = n + 2 * pad - k + 1
    out = np.zeros((m, m))
    for dy in range(k):
        for dx in range(k):
            out += kern[dy, dx] * ones[dy:dy + m, dx:dx + m]
    return out


# --------------------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------------------
def build():
    net = Net()

    # ---------------- skin-colour membership (per pixel, fuzzy AND of linear rules) ----------------
    # Base gamut: the usual loose rules for skin in sRGB photos - reddish-orange hue (about -12..45 degrees),
    # some but not extreme saturation (0.08..0.80), not too dark, red above green.  Each rule is a linear form
    # u(R,G,B) that must reach a margin m; its violation is relu(1 - u/m); skin = relu(1 - largest violation).
    base_rules = [
        # (cR, cG, cB, const, margin)
        (0.20, 1.0, -1.20, 0.0, 0.06),    # hue above about -12 deg:  (G-B) + 0.20 (R-B) > 0
        (0.75, -1.0, 0.25, 0.0, 0.05),    # hue below about 45 deg:   0.75 (R-B) - (G-B) > 0
        (0.92, 0.0, -1.0, 0.0, 0.06),     # saturation above 0.08:    0.92 R - B > 0
        (-0.20, 0.0, 1.0, 0.0, 0.08),     # saturation below 0.80:    B - 0.20 R > 0
        (1.0, 0.0, 0.0, -0.20, 0.15),     # not too dark:  R > 0.20, full at 0.35
        (1.0, -1.0, 0.0, -0.03, 0.04),    # red above green:  R - G > 0.03, full at 0.07
    ]
    # Level sets: the same gamut with one extra demand each.  A face often merges with brown hair, a warm wall
    # or wood in the loose map but stands out as a blob once only the brighter / pinker / redder part is kept.
    extra_rules = [
        (1.0, 0.0, 0.0, -0.42, 0.10),     # bright:       R > 0.42, full at 0.52
        (1.0, 0.0, 0.0, -0.62, 0.10),     # very bright:  R > 0.62, full at 0.72
        (0.5, -1.0, 0.5, 0.0, 0.03),      # pinker hue (below about 30 deg):  0.5 (R-B) - (G-B) > 0
        (1.0, -1.0, 0.0, -0.13, 0.05),    # redder:       R - G > 0.13, full at 0.18
    ]
    NB, NX = len(base_rules), len(extra_rules)
    NV = 1 + NX                                                         # number of skin maps
    M = np.zeros((NB + NX, 3)); bias = np.zeros(NB + NX)
    for i, (cr, cg, cb, c0, m) in enumerate(base_rules + extra_rules):
        M[i] = [-cr / m, -cg / m, -cb / m]
        bias[i] = 1.0 - c0 / m
    net.lin("col", "image", M, bias)
    net.relu("viol", "col")                                             # (NB+NX, 96, 96) violations
    net.reshape("viol_r", "viol", (1, NB + NX, IMG * IMG))
    net.pool("bm_r", "viol_r", "maxpool", [NB, 1])                      # largest base violation (first NB rows)
    net.reshape("bm", "bm_r", (1, IMG, IMG))
    A = np.zeros((NV, NB + NX)); Bm = np.zeros((NV, 1))
    Bm[0, 0] = 1.0
    for k in range(NX):
        A[1 + k, NB + k] = 1.0; Bm[1 + k, 0] = -1.0                     # extra_k - bm
    net.lin("sv_a", "viol", A)
    net.lin("sv_b", "bm", Bm)
    net.add("sv_s", ["sv_a", "sv_b"])
    net.relu("sv", "sv_s")                                              # [bm, relu(extra_k - bm)]
    C = -np.eye(NV); C[:, 0] = -1.0
    net.lin("skin5_l", "sv", C, np.ones(NV))
    net.relu("skin5", "skin5_l")                                        # (NV, 96, 96): 1 - max(bm, extra_k)
    net.lin("skin", "skin5", np.eye(1, NV))                             # loose map alone

    # helper tensor for constant maps: any small tensor will do (weights from it are zero)
    net.pool("sfrac", "skin", "avgpool", [IMG, IMG])                      # (1,1,1) share of skin-like pixels

    # ---------------- image stacks and pyramids ----------------
    net.lin("lum", "image", [[0.299, 0.587, 0.114]])                    # luminance

    def pyr(src, prefix, l):
        nm = f"{prefix}{l}"
        if nm not in net.shape:
            if src == "x0":                                             # (4, n, n): R, G, B, loose skin - stacked after pooling
                net.pool(nm + "_i", "image", "avgpool", [l, l])
                net.pool(nm + "_k", "skin", "avgpool", [l, l])
                net.lin(nm + "_a", nm + "_i", np.eye(4, 3))
                net.lin(nm + "_b", nm + "_k", np.eye(4, 1, k=-3))
                net.add(nm, [nm + "_a", nm + "_b"])
            else:
                net.pool(nm, src, "avgpool", [l, l])
        return nm

    # ---------------- templates ----------------
    KB = 13
    T_blob, T_blob_neg = blob_template(KB, a=2.0, b=2.75, r_in=1.45, r_out=2.15, w_bottom=0.5)
    BLOB_T0, BLOB_T1 = 0.25, 0.60          # response ramp: 0 at 0.25, 1 at 0.60
    OUTSIDE_AS_SKIN = 0.5

    # eye band (on luminance, face about 4 cells wide): forehead and cheeks brighter than the eye row
    KE = 7
    eye_shifts = (-1.0, 0.0, 1.0)
    EYE_REL, EYE_FULL = 0.03, 0.05

    def eye_kernel(dy):
        fore = unit(box_mask(KE, -1.4, 1.4, -2.5 + dy, -1.5 + dy))
        eyes = unit(box_mask(KE, -1.6, 1.6, -1.35 + dy, -0.25 + dy))
        cheek = unit(box_mask(KE, -1.6, 1.6, -0.1 + dy, 1.0 + dy))
        return (0.5 * fore + (0.5 - EYE_REL) * cheek - eyes) / EYE_FULL

    # context of a head (on RGB at half the blob resolution; numbers are in blob-level cells, halved below):
    # hair/top box and shoulder box compared with the mean of the boxes left and right of the head
    KC = 9

    def ctx_kernels():
        s = 0.5
        left = unit(box_mask(KC, -6.5 * s, -3.5 * s, -2.5 * s, 2.5 * s))
        right = unit(box_mask(KC, 3.5 * s, 6.5 * s, -2.5 * s, 2.5 * s))
        top = unit(box_mask(KC, -2.0 * s, 2.0 * s, -5.0 * s, -3.0 * s))
        below = unit(box_mask(KC, -4.0 * s, 4.0 * s, 4.5 * s, 8.0 * s))
        side = 0.5 * (left + right)
        return top - side, below - side, (top, below, left, right)

    CTX_INVALID = 10.0   # subtracted (times the missing fraction) where a box leaves the picture

    CTX_T0, CTX_RANGE = 0.15, 0.30
    W_EYE, W_CTX = 0.3, 0.3
    POS_PEN, POS_START, POS_LEN = 0.3, 0.5, 0.4      # blobs low in the frame count less (tables, floors, food)

    def face_level(l, has_eye):
        n = IMG // l
        s_l = pyr("skin5", "S", l)                                      # (NV, n, n) skin densities
        g = 1.0 / (BLOB_T1 - BLOB_T0)
        parts = []
        for v in range(NV):
            net.lin(f"fs{l}_{v}", s_l, np.eye(1, NV, k=v))              # one skin map
            Wb = np.zeros((2 * NV, 1, KB, KB)); bb = np.zeros(2 * NV)
            Wb[2 * v, 0] = g * T_blob; Wb[2 * v + 1, 0] = g * T_blob
            bb[2 * v], bb[2 * v + 1] = -g * BLOB_T0, -g * BLOB_T0 - 1.0
            parts.append(net.conv(f"fb{l}_c{v}", f"fs{l}_{v}", Wb, bb, pad=KB // 2))
        out_frac = T_blob_neg.sum() - inside_sum(T_blob_neg, n, KB // 2)   # ring weight lying outside the picture
        out_frac = OUTSIDE_AS_SKIN * out_frac                              # unseen surroundings: neither clean nor skin
        net.const_map(f"fb{l}_o", "sfrac", np.stack([-g * out_frac] * (2 * NV)))
        net.add(f"fb{l}_s", parts + [f"fb{l}_o"])
        net.relu(f"fb{l}_r", f"fb{l}_s")
        D = np.zeros((NV, 2 * NV))
        for v in range(NV):
            D[v, 2 * v], D[v, 2 * v + 1] = 1.0, -1.0
        net.lin(f"fb{l}_v", f"fb{l}_r", D)                              # clipped blob response per skin map
        net.reshape(f"fb{l}_vr", f"fb{l}_v", (1, NV, n * n))
        net.pool(f"fb{l}_m", f"fb{l}_vr", "maxpool", [NV, 1])           # best skin map
        net.reshape(f"fb{l}", f"fb{l}_m", (1, n, n))                    # blob in [0, 1]
        if has_eye:
            y_l = pyr("lum", "Y", l)
            We = np.zeros((len(eye_shifts), 1, KE, KE))
            for i, dy in enumerate(eye_shifts):
                We[i, 0] = eye_kernel(dy)
            net.conv(f"fe{l}_c", y_l, We, pad=KE // 2)
            net.relu(f"fe{l}_r", f"fe{l}_c")
            net.clip01(f"fe{l}", f"fe{l}_r", np.ones(len(eye_shifts)), 0.0)
            net.lin(f"f1{l}_a", f"fb{l}", [[1.0]], [-W_EYE])
            net.lin(f"f1{l}_b", f"fe{l}", [[W_EYE, -W_EYE]])
            net.add(f"f1{l}_s", [f"f1{l}_a", f"f1{l}_b"])
        else:
            net.lin(f"f1{l}_s", f"fb{l}", [[1.0]], [-W_EYE])
        net.relu(f"f1{l}", f"f1{l}_s")
        net.pool(f"f1{l}_p", f"f1{l}", "maxpool", [2, 2])               # (1, n/2, n/2)
        # context
        x2 = pyr("x0", "X", 2 * l)
        kt, kb_, (b_top, b_below, b_left, b_right) = ctx_kernels()
        Wc = np.zeros((12, 4, KC, KC))
        for c in range(3):
            Wc[c, c] = kt; Wc[3 + c, c] = -kt; Wc[6 + c, c] = kb_; Wc[9 + c, c] = -kb_
        net.conv(f"fc{l}_c", x2, Wc, pad=KC // 2)
        m2 = n // 2
        miss = [1.0 - inside_sum(b, m2, KC // 2) for b in (b_top, b_below, b_left, b_right)]
        side_miss = np.maximum(miss[2], miss[3])
        inv_t = -CTX_INVALID * np.maximum(miss[0], side_miss)
        inv_b = -CTX_INVALID * np.maximum(miss[1], side_miss)
        net.const_map(f"fc{l}_o", "sfrac", np.stack([inv_t] * 6 + [inv_b] * 6))
        net.add(f"fc{l}_s", [f"fc{l}_c", f"fc{l}_o"])
        net.relu(f"fc{l}_r", f"fc{l}_s")
        net.clip01(f"fc{l}", f"fc{l}_r", np.ones(12) / CTX_RANGE, -CTX_T0 / CTX_RANGE)
        net.lin(f"fx{l}", f"fc{l}", [[W_CTX, -W_CTX]], [-W_CTX])
        m = n // 2
        yf = (np.arange(m) + 0.5) / m
        pen = -POS_PEN * np.clip((yf - POS_START) / POS_LEN, 0, 1)
        net.const_map(f"fp{l}", "sfrac", np.repeat(pen[:, None], m, 1))
        net.add(f"ff{l}_s", [f"f1{l}_p", f"fx{l}", f"fp{l}"])
        net.relu(f"ff{l}", f"ff{l}_s")
        return net.pool(f"ff{l}_g", f"ff{l}", "maxpool", [m, m])        # (1,1,1)

    face_levels = [(2, False), (3, True), (4, True), (6, True), (8, True)]
    face_out = [face_level(l, e) for l, e in face_levels]

    # ---------------- upright figure with skin on it, on a locally uniform background ----------------
    KF = 13
    FIG_T0, FIG_RANGE = 0.10, 0.25
    FIG_W_FLANK, FIG_W_TOP = 0.5, 1.0     # left/right background mismatch counts half, a trunk that continues upward fully
    SK_T0, SK_RANGE = 0.01, 0.05
    SK_MAX, SK_MAX_RANGE = 0.35, 0.20     # skin share of the upper trunk (clothed torso): fine up to 0.15, none above 0.35

    def fig_level(l):
        x_l = pyr("x0", "X", l)
        # boxes in cells; figure trunk is 3 wide and 10 tall (rows -3..6), top box above the head
        def bx(x0, x1, y0, y1):
            return unit(box_mask(KF, x0 - 0.5, x1 + 0.5, y0 - 0.5, y1 + 0.5))
        U, Lw = bx(-1, 1, -3, 1), bx(-1, 1, 2, 6)
        LU, RU = bx(-5, -3, -3, 1), bx(3, 5, -3, 1)
        LL, RL = bx(-5, -3, 2, 6), bx(3, 5, 2, 6)
        Tp = bx(-1, 1, -6, -5)
        diffs = [U - LU, U - RU, Lw - LL, Lw - RL, LU - RU, LL - RL, Tp - 0.5 * (LU + RU)]
        nd = len(diffs)
        W1 = np.zeros((nd * 6 + 4, 4, KF, KF)); b1 = np.zeros(nd * 6 + 4)
        for d, D in enumerate(diffs):
            for c in range(3):
                W1[d * 6 + c, c] = D
                W1[d * 6 + 3 + c, c] = -D
        inside = bx(-2, 2, -3, 6)
        flank = 0.5 * (bx(-5, -3, -3, 6) + bx(3, 5, -3, 6))
        sk = (inside - flank) / SK_RANGE
        W1[nd * 6, 3] = sk; b1[nd * 6] = -SK_T0 / SK_RANGE
        W1[nd * 6 + 1, 3] = sk; b1[nd * 6 + 1] = -SK_T0 / SK_RANGE - 1.0
        trunk = U
        W1[nd * 6 + 2, 3] = -trunk / SK_MAX_RANGE; b1[nd * 6 + 2] = SK_MAX / SK_MAX_RANGE
        W1[nd * 6 + 3, 3] = -trunk / SK_MAX_RANGE; b1[nd * 6 + 3] = SK_MAX / SK_MAX_RANGE - 1.0
        net.conv(f"g1{l}_c", x_l, W1, b1, pad=KF // 2)
        net.relu(f"g1{l}", f"g1{l}_c")
        # norms: n1..n4 contrasts, P penalties, skin clip
        def nrm(d):
            r = np.zeros(nd * 6 + 4); r[d * 6:d * 6 + 6] = 1.0; return r
        n1, n2, n3, n4 = nrm(0), nrm(1), nrm(2), nrm(3)
        P = FIG_W_FLANK * (nrm(4) + nrm(5)) + FIG_W_TOP * nrm(6)
        # skin on the figure AND trunk not all skin:  clip(a) + clip(b) - 1
        skc = np.zeros(nd * 6 + 4); skc[nd * 6:] = [1.0, -1.0, 1.0, -1.0]
        zero = np.zeros(6)
        net.lin(f"g2{l}_l", f"g1{l}", [n1, n1 - n2, n3, n3 - n4, P, skc], [0, 0, 0, 0, 0, -1.0])
        net.relu(f"g2{l}", f"g2{l}_l")                                  # [n1, (n1-n2)+, n3, (n3-n4)+, P, skin]
        # dU = min(n1,n2), dL = min(n3,n4);  out: [dU, dU - 2 dL, P, skin]
        dU = np.array([1, -1, 0, 0, 0, 0.0]); dL = np.array([0, 0, 1, -1, 0, 0.0])
        e_P = np.array([0, 0, 0, 0, 1, 0.0]); e_s = np.array([0, 0, 0, 0, 0, 1.0])
        net.lin(f"g3{l}_l", f"g2{l}", [dU, dU - 2 * dL, e_P, e_s])
        net.relu(f"g3{l}", f"g3{l}_l")
        # core = min(dU, 2 dL); fig = clip((core - P - T0)/RANGE)
        core = np.array([1, -1, 0, 0.0]); pp = np.array([0, 0, 1, 0.0]); ss = np.array([0, 0, 0, 1.0])
        f = (core - pp) / FIG_RANGE
        net.lin(f"g4{l}_l", f"g3{l}", [f, f, ss], [-FIG_T0 / FIG_RANGE, -FIG_T0 / FIG_RANGE - 1.0, 0.0])
        net.relu(f"g4{l}", f"g4{l}_l")
        net.lin(f"g5{l}_l", f"g4{l}", [[1.0, -1.0, 1.0]], [-1.0])       # fig AND skin
        net.relu(f"g5{l}", f"g5{l}_l")
        n = IMG // l
        return net.pool(f"g5{l}_g", f"g5{l}", "maxpool", [n, n])

    fig_levels = [3, 4, 6]
    fig_out = [fig_level(l) for l in fig_levels]

    # ---------------- evidence: best face level, best figure level ----------------
    def best_of(name, items):
        k = len(items)
        parts = []
        for i, it in enumerate(items):
            W = np.zeros((k, 1)); W[i, 0] = 1.0
            parts.append(net.dense(f"{name}_e{i}", it, W))
        net.add(f"{name}_v", parts)
        net.reshape(f"{name}_r", f"{name}_v", (1, 1, k))
        return net.pool(name, f"{name}_r", "maxpool", [1, k])           # (1,1,1)

    e_face = best_of("eface", face_out)
    e_fig = best_of("efig", fig_out)

    # picture dominated by skin-like colour (wood, food, warm light): colour evidence is less trustworthy
    WARM_T0, WARM_RANGE, WARM_PEN = 0.35, 0.40, 0.2
    net.clip01("warm", "sfrac", [1.0 / WARM_RANGE], -WARM_T0 / WARM_RANGE)

    W_FACE, W_FIG, THETA, GAIN = 1.0, 0.8, 0.5, 10.0
    net.dense("o_face", e_face, [[0.0], [GAIN * W_FACE]], [0.0, -GAIN * THETA])
    net.dense("o_fig", e_fig, [[0.0], [GAIN * W_FIG]])
    net.dense("o_warm", "warm", [[0.0, 0.0], [-GAIN * WARM_PEN, GAIN * WARM_PEN]])
    net.add("out", ["o_face", "o_fig", "o_warm"])
    return net


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    net = build()
    (out / "network.json").write_text(json.dumps({"layers": net.layers}, indent=1))
    np.savez(out / "weights.npz", **{k: v.astype(np.float32) for k, v in net.w.items()})


if __name__ == "__main__":
    main()
