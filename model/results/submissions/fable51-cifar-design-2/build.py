#!/usr/bin/env python3
"""Hand-designed CIFAR-10 classifier.  Usage: python3 build.py OUT_DIR

Every number below is typed from general knowledge of colours, edges and the ten
classes (no training data is read, no fitting of any kind).  The network is a
small rule system expressed as convolutions, pools and two dense layers:

  colour branch : 1x1 convs turn RGB into soft per-pixel colour indicators
                  (blue, green, warm/brown, red, bright, dark, sky, water,
                  low saturation, white, gray, vivid) -> 8x8 area fractions
  edge branch   : gray -> Sobel dx/dy + Laplacian -> per-pixel "edge", "smooth",
                  "texture" indicators -> 10x10 area fractions
  line branch   : 9-pixel straight horizontal / vertical line detectors
  blob branch   : dark round blob (wheel / eye), dark and bright thin vertical
                  bars (legs, masts) -> quadrant maxima
  readout       : named cues = region means / maxima of those maps (+ a few
                  AND-combinations), then a hand-written table of class weights.
"""
import json, os, sys
import numpy as np


# ----------------------------------------------------------------------------- builder
class Net:
    def __init__(self):
        self.layers, self.W, self.shapes = [], {}, {"image": (3, 32, 32)}

    def _add(self, name, op, inputs, shape, **kw):
        L = {"name": name, "op": op, "inputs": inputs}
        L.update(kw)
        self.layers.append(L)
        self.shapes[name] = tuple(int(v) for v in shape)
        return name

    def conv(self, name, inp, W, b, stride=1, padding=0):
        c, h, w = self.shapes[inp]
        o, i, k, k2 = W.shape
        assert i == c and k == k2
        ho, wo = (h + 2 * padding - k) // stride + 1, (w + 2 * padding - k) // stride + 1
        self.W[name + ".W"], self.W[name + ".b"] = np.asarray(W, np.float32), np.asarray(b, np.float32)
        return self._add(name, "conv", [inp], (o, ho, wo), out=o, kernel=[k, k], stride=stride, padding=padding)

    def dense(self, name, inp, W, b):
        n = int(np.prod(self.shapes[inp]))
        assert W.shape[1] == n, (W.shape, n)
        self.W[name + ".W"], self.W[name + ".b"] = np.asarray(W, np.float32), np.asarray(b, np.float32)
        return self._add(name, "dense", [inp], (W.shape[0],), out=int(W.shape[0]))

    def relu(self, name, inp):
        return self._add(name, "relu", [inp], self.shapes[inp])

    def pool(self, name, inp, op, size, stride):
        c, h, w = self.shapes[inp]
        return self._add(name, op, [inp], (c, (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1),
                         size=list(size), stride=list(stride))

    def add(self, name, inputs):
        return self._add(name, "add", list(inputs), self.shapes[inputs[0]])


# ----------------------------------------------------------------------------- helpers
def ramp_rows(a, c, t, w):
    """Two linear units whose relu difference is clip((a.x + c - t) / w, 0, 1)."""
    a = np.asarray(a, float) / w
    return [(a, (c - t) / w), (a, (c - t) / w - 1.0)]


class Lin:
    """Collects rows (weights over input channels, bias) for a 1x1 conv."""
    def __init__(self, n_in):
        self.n_in, self.rows, self.names = n_in, [], {}

    def unit(self, name, weights, bias=0.0):
        v = np.zeros(self.n_in)
        for k, val in weights.items():
            v[k] += val
        self.names[name] = len(self.rows)
        self.rows.append((v, bias))
        return self.names[name]

    def ramp(self, name, weights, t, w):
        """Two units name+'_a', name+'_b'; indicator = relu(a) - relu(b)."""
        v = np.zeros(self.n_in)
        for k, val in weights.items():
            v[k] += val
        for suffix, (a, b) in zip("ab", ramp_rows(v, 0.0, t, w)):
            self.names[name + "_" + suffix] = len(self.rows)
            self.rows.append((a, b))

    def matrices(self):
        W = np.stack([r[0] for r in self.rows])[:, :, None, None]
        b = np.array([r[1] for r in self.rows])
        return W, b


# ----------------------------------------------------------------------------- the network
def build():
    net = Net()

    # ======================= colour branch ===========================================
    R, G, B = 0, 1, 2
    c1 = Lin(3)
    c1.ramp("blue", {B: 1, R: -0.5, G: -0.5}, t=0.03, w=0.08)        # B - (R+G)/2  : sky, water
    c1.ramp("blue2", {B: 1, R: -0.5, G: -0.5}, t=0.06, w=0.10)       # clearly blue (water, not bluish shade)
    c1.ramp("green", {G: 1, R: -0.5, B: -0.5}, t=0.03, w=0.08)       # G - (R+B)/2  : grass, foliage
    c1.ramp("warm", {R: 1.5, G: -1.0, B: -0.5}, t=0.06, w=0.15)      # brown / tan / orange / red
    c1.ramp("red", {R: 1, G: -0.5, B: -0.5}, t=0.15, w=0.15)         # strong red
    c1.ramp("bright", {R: 1 / 3, G: 1 / 3, B: 1 / 3}, t=0.55, w=0.20)
    c1.ramp("bright2", {R: 1 / 3, G: 1 / 3, B: 1 / 3}, t=0.40, w=0.15)         # sky / water split
    c1.ramp("vbright", {R: 1 / 3, G: 1 / 3, B: 1 / 3}, t=0.75, w=0.15)
    c1.ramp("dark", {R: -1 / 3, G: -1 / 3, B: -1 / 3}, t=-0.30, w=0.15)   # 1 when Y <= 0.15
    for nm, (p, q) in {"rg": (R, G), "gb": (G, B), "rb": (R, B)}.items():  # |pairwise diffs| -> saturation
        c1.unit(nm + "+", {p: 1, q: -1}); c1.unit(nm + "-", {p: -1, q: 1})
    W, b = c1.matrices()
    x = net.relu("col1r", net.conv("col1", "image", W, b))
    n1 = c1.names

    def ind(lin_names, nm):  # weights that reconstruct the indicator from its two pieces
        return {lin_names[nm + "_a"]: 1.0, lin_names[nm + "_b"]: -1.0}

    c2 = Lin(len(c1.rows))
    for nm in ["blue", "green", "warm", "red", "bright", "dark", "vbright"]:
        c2.unit(nm, ind(n1, nm))
    sat = {n1[k]: 1.0 for k in ["rg+", "rg-", "gb+", "gb-", "rb+", "rb-"]}       # ~2 (max - min)
    c2.ramp("satlow", {k: -v for k, v in sat.items()}, t=-0.25, w=0.15)          # 1 when sat <= 0.10
    c2.ramp("sathigh", sat, t=0.40, w=0.40)
    sky = dict(ind(n1, "blue")); [sky.__setitem__(k, sky.get(k, 0) + v) for k, v in ind(n1, "bright2").items()]
    c2.unit("sky", sky, bias=-1.0)                                                # blue AND Y >= ~0.45
    water = dict(ind(n1, "blue2")); [water.__setitem__(k, water.get(k, 0) - v) for k, v in ind(n1, "bright2").items()]
    c2.unit("water", water)                                                       # blue AND Y <= ~0.45
    W, b = c2.matrices()
    x = net.relu("col2r", net.conv("col2", x, W, b))
    n2 = c2.names

    c3 = Lin(len(c2.rows))
    COL = []
    for nm in ["blue", "green", "warm", "red", "bright", "dark", "sky", "water"]:
        c3.unit(nm, {n2[nm]: 1.0}); COL.append(nm)
    c3.unit("satlow", ind(n2, "satlow")); COL.append("satlow")
    c3.unit("sathigh", ind(n2, "sathigh")); COL.append("sathigh")
    d = ind(n2, "satlow"); d[n2["vbright"]] = 1.0
    c3.unit("white", d, bias=-1.0); COL.append("white")                           # very bright AND unsaturated
    d = ind(n2, "satlow"); d[n2["vbright"]] = -1.0; d[n2["dark"]] = -1.0
    c3.unit("gray", d); COL.append("gray")                                        # unsaturated, mid brightness
    d = ind(n2, "sathigh"); d[n2["green"]] = -1.0; d[n2["blue"]] = -1.0
    c3.unit("vivid", d); COL.append("vivid")                                      # saturated but not green/blue
    W, b = c3.matrices()
    x = net.relu("col3r", net.conv("col3", x, W, b))
    col_pool = net.pool("col_pool", x, "avgpool", [4, 4], [4, 4])                # (13, 8, 8)
    COLI = {nm: i for i, nm in enumerate(COL)}

    # ======================= gray / edge branch ======================================
    gray = net.conv("gray", "image", np.array([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1), np.zeros(1))
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 4.0
    sy = sx.T.copy()
    lap = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], float) / 8.0
    K = np.stack([sx, -sx, sy, -sy, lap, -lap])[:, None]                           # (6,1,3,3)
    e1 = net.relu("edge1r", net.conv("edge1", gray, K, np.zeros(6)))              # 30x30
    DXP, DXM, DYP, DYM, LP, LM = range(6)

    e2 = Lin(6)
    grad = {DXP: 0.5, DXM: 0.5, DYP: 0.5, DYM: 0.5}
    e2.ramp("edge", grad, t=0.05, w=0.15)
    e2.ramp("smooth", {k: -v for k, v in grad.items()}, t=-0.04, w=0.03)          # 1 when grad <= 0.01
    e2.ramp("texup", {LP: 1.0, LM: 1.0}, t=0.01, w=0.03)                          # some local relief ...
    e2.ramp("texdown", {LP: -1.0, LM: -1.0}, t=-0.10, w=0.03)                     # ... but not a hard edge
    e2.unit("a1", {DXP: 1, DXM: 1, DYP: -1, DYM: -1})                             # relu(|dx| - |dy|)
    e2.unit("a2", {DXP: -1, DXM: -1, DYP: 1, DYM: 1})                             # relu(|dy| - |dx|)
    e2.unit("sum", {DXP: 1, DXM: 1, DYP: 1, DYM: 1})                              # |dx| + |dy|
    W, b = e2.matrices()
    x = net.relu("edge2r", net.conv("edge2", e1, W, b))
    n2e = e2.names
    e3 = Lin(len(e2.rows))
    e3.unit("edge", ind(n2e, "edge")); e3.unit("smooth", ind(n2e, "smooth"))
    d = ind(n2e, "texup"); d.update({k: v for k, v in ind(n2e, "texdown").items()})
    e3.unit("tex", d, bias=-1.0)                                                  # mild texture band
    # axis-aligned edge: | |dx|-|dy| | - 2 min(|dx|,|dy|) = 2 (a1 + a2) - sum ; diagonal: min = (sum - a1 - a2) / 2
    e3.ramp("axis", {n2e["a1"]: 2, n2e["a2"]: 2, n2e["sum"]: -1}, t=0.06, w=0.12)
    e3.ramp("diag", {n2e["a1"]: -0.5, n2e["a2"]: -0.5, n2e["sum"]: 0.5}, t=0.04, w=0.08)
    W, b = e3.matrices()
    x = net.relu("edge3r", net.conv("edge3", x, W, b))
    e4 = Lin(len(e3.rows))
    EDG = []
    for nm in ["edge", "smooth", "tex"]:
        e4.unit(nm, {e3.names[nm]: 1.0}); EDG.append(nm)
    for nm in ["axis", "diag"]:
        e4.unit(nm, ind(e3.names, nm)); EDG.append(nm)
    W, b = e4.matrices()
    x = net.relu("edge4r", net.conv("edge4", x, W, b))
    edge_pool = net.pool("edge_pool", x, "avgpool", [3, 3], [3, 3])              # (5, 10, 10), covers px 1..30
    EDGI = {nm: i for i, nm in enumerate(EDG)}

    # ---- straight lines: 11-px mean of a signed edge minus 1.5 x perpendicular energy
    KL = 11
    Kl = np.zeros((8, 6, KL, KL))
    bl = np.zeros(8)
    t_line, w_line = 0.08, 0.12
    specs = [("hl+", DYP, (DXP, DXM), "row"), ("hl-", DYM, (DXP, DXM), "row"),
             ("vl+", DXP, (DYP, DYM), "col"), ("vl-", DXM, (DYP, DYM), "col")]
    LINES = []
    for j, (nm, main, perp, orient) in enumerate(specs):
        k = np.zeros((6, KL, KL))
        mid = KL // 2
        sl = (slice(mid, mid + 1), slice(None)) if orient == "row" else (slice(None), slice(mid, mid + 1))
        k[main][sl] = 1.0 / KL
        for p in perp:
            k[p][sl] = -1.5 / KL
        Kl[2 * j], Kl[2 * j + 1] = k / w_line, k / w_line
        bl[2 * j], bl[2 * j + 1] = -t_line / w_line, -t_line / w_line - 1.0
        LINES.append(nm)
    LM_ = 30 - KL + 1                                                             # 20
    x = net.relu("line1r", net.conv("line1", e1, Kl, bl))                        # 20x20, covers px 6..25
    Wl = np.zeros((4, 8, 1, 1))
    for j in range(4):
        Wl[j, 2 * j, 0, 0], Wl[j, 2 * j + 1, 0, 0] = 1.0, -1.0
    line_map = net.relu("line2r", net.conv("line2", x, Wl, np.zeros(4)))        # (4, 20, 20)
    line_max = net.pool("line_max", line_map, "maxpool", [LM_, LM_], [LM_, LM_])  # (4, 1, 1)
    # longest line: fraction of a row (column) covered by a horizontal (vertical) line, max over rows (columns)
    line_rowmean = net.pool("line_rowmean", line_map, "avgpool", [1, LM_], [1, LM_])   # (4, 20, 1)
    line_rowmax = net.pool("line_rowmax", line_rowmean, "maxpool", [LM_, 1], [LM_, 1])  # (4, 1, 1)
    line_colmean = net.pool("line_colmean", line_map, "avgpool", [LM_, 1], [LM_, 1])   # (4, 1, 20)
    line_colmax = net.pool("line_colmax", line_colmean, "maxpool", [1, LM_], [1, LM_])  # (4, 1, 1)
    LINI = {nm: i for i, nm in enumerate(LINES)}

    # ---- blobs and bars on gray (9x9, no padding -> 24x24, covers px 4..27)
    yy, xx = np.mgrid[-4:5, -4:5]
    rr = np.sqrt(yy ** 2 + xx ** 2)
    disk, ring = (rr <= 2.2), (rr >= 3.0) & (rr <= 4.4)
    sectors = {"top": (yy < 0) & (np.abs(xx) <= np.abs(yy)), "bottom": (yy > 0) & (np.abs(xx) <= np.abs(yy)),
               "left": (xx < 0) & (np.abs(yy) < np.abs(xx)), "right": (xx > 0) & (np.abs(yy) < np.abs(xx))}
    bar_c = (xx == 0)
    Kb, bb, names1 = [], [], {}

    def blob_unit(nm, k, t, w):  # two ramp pieces of a 9x9 template response
        names1[nm + "_a"], names1[nm + "_b"] = len(Kb), len(Kb) + 1
        Kb.extend([k / w, k / w]); bb.extend([-t / w, -t / w - 1.0])

    for s, m in sectors.items():  # dark disk with a brighter ring sector on each side
        ring_s = ring & m
        blob_unit("sec_" + s, ring_s / ring_s.sum() - disk / disk.sum(), 0.12, 0.15)
    for side, m in {"L": (xx == -2), "R": (xx == 2)}.items():  # thin vertical bar: centre column vs each side column
        k = m / m.sum() - bar_c / bar_c.sum()
        blob_unit("legdark_" + side, k, 0.08, 0.15)
        blob_unit("legbright_" + side, -k, 0.08, 0.15)
    Kb = np.stack(Kb)[:, None]
    x = net.relu("blob1r", net.conv("blob1", gray, Kb, np.array(bb)))           # (16, 24, 24)
    b2 = Lin(len(bb))
    for nm in ["sec_top", "sec_bottom", "sec_left", "sec_right", "legdark_L", "legdark_R", "legbright_L", "legbright_R"]:
        b2.unit(nm, ind(names1, nm))
    W, b = b2.matrices()
    x = net.relu("blob2r", net.conv("blob2", x, W, b))
    b3 = Lin(len(b2.rows))
    secs = {b2.names["sec_" + s]: 1.0 / 1.5 for s in sectors}                     # sum of 4 sectors, ramp over [2.5, 4]
    b3.unit("wheel_a", secs, bias=-2.5 / 1.5); b3.unit("wheel_b", secs, bias=-4.0 / 1.5)
    b3.unit("legdark", {b2.names["legdark_L"]: 1.0, b2.names["legdark_R"]: 1.0}, bias=-1.0)      # AND of both sides
    b3.unit("legbright", {b2.names["legbright_L"]: 1.0, b2.names["legbright_R"]: 1.0}, bias=-1.0)
    W, b = b3.matrices()
    x = net.relu("blob3r", net.conv("blob3", x, W, b))
    b4 = Lin(len(b3.rows))
    b4.unit("blob", {b3.names["wheel_a"]: 1.0, b3.names["wheel_b"]: -1.0})
    b4.unit("legdark", {b3.names["legdark"]: 1.0}); b4.unit("legbright", {b3.names["legbright"]: 1.0})
    BLOBS = ["blob", "legdark", "legbright"]
    W, b = b4.matrices()
    blob_map = net.relu("blob4r", net.conv("blob4", x, W, b))                   # (3, 24, 24)
    blob_q = net.pool("blob_q", blob_map, "maxpool", [12, 12], [12, 12])         # (3, 2, 2) quadrant maxima
    blob_3 = net.pool("blob_3", blob_map, "maxpool", [8, 8], [8, 8])             # (3, 3, 3)
    BLI = {nm: i for i, nm in enumerate(BLOBS)}

    # ---- raw colour means
    rgb_pool = net.pool("rgb_pool", "image", "avgpool", [4, 4], [4, 4])          # (3, 8, 8)

    # ======================= cues ====================================================
    branches = {  # name -> (layer, shape, pixel offset of cell 0, cell size)
        "col": (col_pool, net.shapes[col_pool], 0.0, 4.0),
        "edge": (edge_pool, net.shapes[edge_pool], 1.0, 3.0),
        "line": (line_map, net.shapes[line_map], 6.0, 1.0),
        "lmax": (line_max, net.shapes[line_max], 0.0, 32.0),
        "lrow": (line_rowmax, net.shapes[line_rowmax], 0.0, 32.0),
        "lcol": (line_colmax, net.shapes[line_colmax], 0.0, 32.0),
        "blobq": (blob_q, net.shapes[blob_q], 4.0, 12.0),
        "blob3": (blob_3, net.shapes[blob_3], 4.0, 8.0),
        "rgb": (rgb_pool, net.shapes[rgb_pool], 0.0, 4.0),
    }
    cue_names, cue_bias, contrib = [], [], {k: [] for k in branches}

    def centers(br):
        _, (c, h, w), off, cs = branches[br]
        ys = off + cs * (np.arange(h) + 0.5)
        xs = off + cs * (np.arange(w) + 0.5)
        return np.meshgrid(ys, xs, indexing="ij")

    REG = {  # region predicates on pixel coordinates (y down, x right), 0..32
        "all": lambda y, x: np.ones_like(y, bool),
        "top": lambda y, x: y < 8, "upper": lambda y, x: y < 12,
        "bot": lambda y, x: y >= 24, "lower": lambda y, x: y >= 20,
        "mid": lambda y, x: (y >= 10) & (y < 22),
        "cen": lambda y, x: (y >= 8) & (y < 24) & (x >= 8) & (x < 24),
        "bor": lambda y, x: ~((y >= 8) & (y < 24) & (x >= 8) & (x < 24)),
        "bl": lambda y, x: (y >= 16) & (x < 16), "br": lambda y, x: (y >= 16) & (x >= 16),
        "tl": lambda y, x: (y < 16) & (x < 16), "tr": lambda y, x: (y < 16) & (x >= 16),
        "ucen": lambda y, x: (y >= 4) & (y < 20) & (x >= 6) & (x < 26),
        "botrow": lambda y, x: y >= 20,
        "sides": lambda y, x: (x < 6) | (x >= 26),
    }

    def mask(br, region):
        y, x = centers(br)
        m = REG[region](y, x).astype(float)
        assert m.sum() > 0, (br, region)
        return m / m.sum()

    def cue(name, bias=0.0):
        cue_names.append(name); cue_bias.append(bias)
        for k in contrib:
            contrib[k].append(np.zeros(branches[k][1]))
        return len(cue_names) - 1

    def term(ci, br, ch, region, weight=1.0):
        contrib[br][ci][ch] += weight * mask(br, region)

    def simple(br, chan_index, ch, region, scale=1.0, name=None):
        ci = cue(name or f"{ch}_{region}")
        term(ci, br, chan_index[ch], region, scale)
        return ci

    # --- colour area fractions
    for ch, regs in {"blue": ["all", "top", "bot", "bor", "sides"], "sky": ["all", "top", "bot", "bor"], "water": ["all", "bot", "lower"],
                     "green": ["all", "top", "bot", "cen", "bor"], "warm": ["all", "cen", "bor"],
                     "red": ["all", "cen"], "bright": ["all", "top", "bot"], "dark": ["all", "top", "bot", "cen"],
                     "white": ["all", "top", "cen"], "gray": ["all", "top", "cen", "bot"],
                     "satlow": ["all", "cen"], "vivid": ["all", "cen"], "sathigh": ["all"]}.items():
        for r in regs:
            simple("col", COLI, ch, r)
    # --- edge area fractions (edges are thin, so scaled up to a 0..1-ish range)
    for ch, regs, sc in [("edge", ["all", "cen", "bor", "top", "bot"], 3.0),
                         ("smooth", ["all", "top", "bot", "bor", "cen"], 1.0),
                         ("tex", ["all", "cen", "bor"], 2.0),
                         ("axis", ["all", "cen", "bor", "bot"], 3.0),
                         ("diag", ["all", "cen", "bor"], 3.0)]:
        for r in regs:
            simple("edge", EDGI, ch, r, sc)
    # --- lines
    ci = cue("hl_max"); term(ci, "lmax", LINI["hl+"], "all", 0.5); term(ci, "lmax", LINI["hl-"], "all", 0.5)
    ci = cue("vl_max"); term(ci, "lmax", LINI["vl+"], "all", 0.5); term(ci, "lmax", LINI["vl-"], "all", 0.5)
    ci = cue("hrow_max"); term(ci, "lrow", LINI["hl+"], "all", 1.0); term(ci, "lrow", LINI["hl-"], "all", 1.0)
    ci = cue("vcol_max"); term(ci, "lcol", LINI["vl+"], "all", 1.0); term(ci, "lcol", LINI["vl-"], "all", 1.0)
    for r in ["all", "mid", "bot", "top", "cen"]:
        ci = cue(f"hl_{r}"); term(ci, "line", LINI["hl+"], r, 3.0); term(ci, "line", LINI["hl-"], r, 3.0)
    for r in ["all", "top", "cen", "bot"]:
        ci = cue(f"vl_{r}"); term(ci, "line", LINI["vl+"], r, 3.0); term(ci, "line", LINI["vl-"], r, 3.0)
    # --- blobs / bars (quadrant and 3x3 maxima)
    ci = cue("wheel_bl"); term(ci, "blobq", BLI["blob"], "bl")
    ci = cue("wheel_br"); term(ci, "blobq", BLI["blob"], "br")
    ci = cue("blob_top"); term(ci, "blobq", BLI["blob"], "tl", 0.5); term(ci, "blobq", BLI["blob"], "tr", 0.5)
    ci = cue("blob_mean"); term(ci, "blob3", BLI["blob"], "all")
    ci = cue("legs_bot"); term(ci, "blob3", BLI["legdark"], "botrow", 0.6); term(ci, "blob3", BLI["legbright"], "botrow", 0.4)
    ci = cue("legs_all"); term(ci, "blob3", BLI["legdark"], "all", 0.6); term(ci, "blob3", BLI["legbright"], "all", 0.4)
    # --- centre vs border colour difference (|.| via +/- pairs)
    for ch, nm in enumerate("RGB"):
        for sgn, suf in [(1, "p"), (-1, "m")]:
            ci = cue(f"cb_{nm}{suf}"); term(ci, "rgb", ch, "cen", 2.0 * sgn); term(ci, "rgb", ch, "bor", -2.0 * sgn)
    # --- AND combinations (relu(a + b - 1))
    ci = cue("field", -1.0); term(ci, "col", COLI["bright"], "top"); term(ci, "col", COLI["green"], "bot")
    ci = cue("seascape", -0.8); term(ci, "col", COLI["water"], "lower"); term(ci, "col", COLI["sky"], "top")
    ci = cue("skyscene", -1.0); term(ci, "col", COLI["sky"], "all"); term(ci, "edge", EDGI["smooth"], "all")
    ci = cue("boxy", -1.0); term(ci, "lrow", LINI["hl+"], "all", 1.0); term(ci, "lrow", LINI["hl-"], "all", 1.0)
    term(ci, "lcol", LINI["vl+"], "all", 1.0); term(ci, "lcol", LINI["vl-"], "all", 1.0)
    ci = cue("furry", -0.5); term(ci, "edge", EDGI["tex"], "cen", 2.0); term(ci, "col", COLI["warm"], "cen", 0.5); term(ci, "col", COLI["gray"], "cen", 0.5)
    ci = cue("camo", -0.6); term(ci, "edge", EDGI["tex"], "bor", 2.0); term(ci, "col", COLI["green"], "all", 0.5); term(ci, "col", COLI["warm"], "all", 0.5)
    ci = cue("warm_on_blue", -1.0); term(ci, "col", COLI["blue"], "bor"); term(ci, "col", COLI["warm"], "cen")
    ci = cue("white_on_blue", -1.0); term(ci, "col", COLI["blue"], "bor"); term(ci, "col", COLI["white"], "cen")
    ci = cue("dark_on_blue", -1.0); term(ci, "col", COLI["blue"], "bor"); term(ci, "col", COLI["dark"], "cen")
    ci = cue("gray_on_blue", -1.0); term(ci, "col", COLI["blue"], "bor"); term(ci, "col", COLI["gray"], "cen")
    ci = cue("roughwater"); term(ci, "col", COLI["water"], "lower"); term(ci, "edge", EDGI["smooth"], "bot", -1.0)

    M = len(cue_names)
    parts = []
    for i, (k, (layer, shape, _, _)) in enumerate(branches.items()):
        Wd = np.stack(contrib[k]).reshape(M, -1)
        bd = np.array(cue_bias) if i == 0 else np.zeros(M)
        parts.append(net.dense("cue_" + k, layer, Wd, bd))
    cues = net.relu("cues", net.add("cue_sum", parts))

    # ======================= class rule table ========================================
    CB = {"cb_Rp": 1, "cb_Rm": 1, "cb_Gp": 1, "cb_Gm": 1, "cb_Bp": 1, "cb_Bm": 1}   # |centre - border| colour

    def cb(w):
        return {k: w for k in CB}

    def merge(*dicts):
        out = {}
        for d in dicts:
            for k, v in d.items():
                out[k] = out.get(k, 0.0) + v
        return out

    # shared "animal rather than vehicle" evidence: fine texture and curved edges, no long straight lines,
    # no vivid artificial colours, no wheels, no road, no water; natural brown / green colours
    ANIMAL = {"tex_all": 0.3, "diag_all": 0.3, "axis_all": -0.5, "hrow_max": -0.4, "vcol_max": -0.4,
              "vivid_all": -0.4, "wheel_bl": -0.2, "wheel_br": -0.2, "gray_bot": -0.3, "water_bot": -1.0,
              "warm_all": 0.5, "green_all": 0.3}
    rules = {
        "airplane": {"sky_all": 2.0, "sky_bot": 1.0, "blue_all": 0.5, "white_top": 0.8, "gray_top": 0.5,
                     "bright_all": 0.3, "smooth_all": 0.5, "smooth_bot": 0.8, "skyscene": 1.0, "white_cen": 0.5, "blue_sides": 0.3,
                     "gray_cen": 0.3, "white_on_blue": 0.8, "gray_on_blue": 0.5, "green_all": -1.5,
                     "warm_all": -1.5, "dark_bot": -0.5, "water_lower": -0.5, "warm_on_blue": -0.5, "edge_bor": -0.3},
        "ship": {"water_bot": 1.5, "roughwater": 2.0, "water_lower": 0.5, "blue_bot": 0.5, "seascape": 1.0, "sky_top": 0.5,
                 "white_top": 0.3, "hrow_max": 0.8, "hl_mid": 0.8, "white_cen": 0.6, "white_on_blue": 0.5,
                 "green_all": -1.5, "warm_all": -1.0, "sky_bot": -0.5, "vivid_cen": -0.5, "gray_bot": -0.5, "blue_sides": 0.8},
        "automobile": {"gray_bot": 1.0, "dark_bot": 0.4, "wheel_bl": 0.3, "wheel_br": 0.3, "vivid_cen": 1.2,
                       "vivid_all": 0.3, "red_cen": 0.4, "hrow_max": 0.6, "hl_all": 0.4, "hl_bot": 0.3, "axis_all": 0.4,
                       "smooth_top": 0.4, "sky_top": 0.3, "white_cen": 0.3, "satlow_cen": 0.2, "green_cen": -0.8,
                       "warm_cen": -0.5, "edge_top": -0.5, "water_bot": -1.0, "sky_all": -0.3, "blue_bot": -0.3, "blue_sides": -0.3},
        "truck": {"edge_top": 0.5, "vcol_max": 0.8, "vl_top": 0.5, "axis_bor": 0.4, "axis_all": 0.4,
                  "hrow_max": 0.3, "vivid_all": 0.8, "white_cen": 0.4, "red_cen": 0.4, "gray_bot": 0.5,
                  "wheel_bl": 0.3, "wheel_br": 0.3, "satlow_cen": 0.2, "boxy": 0.3, "green_all": -0.5,
                  "green_bot": -0.5, "warm_cen": -0.8, "smooth_top": -0.5, "water_bot": -1.0, "sky_all": -0.3, "blue_sides": -0.3},
        "bird": merge(ANIMAL, {"sky_all": 0.4, "sky_bor": 0.4, "green_all": 0.4, "warm_on_blue": 1.0,
                               "dark_on_blue": 0.8, "bright_top": 0.3, "smooth_bor": 0.3, "camo": -0.3,
                               "legs_bot": -0.3}, cb(0.4)),
        "cat": merge(ANIMAL, {"gray_cen": 0.8, "satlow_cen": 0.5, "dark_cen": 0.4, "white_cen": 0.2, "warm_cen": 0.2,
                              "blob_top": 0.3, "edge_bor": 0.2, "sky_all": -1.5, "blue_all": -0.5, "green_all": -1.0,
                              "green_bot": -0.5, "legs_bot": -0.3, "smooth_all": -0.3}),
        "dog": merge(ANIMAL, {"warm_cen": 0.9, "white_cen": 0.8, "blob_top": 0.3, "warm_all": 0.2, "satlow_cen": 0.1,
                              "sky_all": -1.5, "blue_all": -0.5, "green_all": -0.3, "legs_bot": -0.3}),
        "deer": merge(ANIMAL, {"warm_cen": 1.0, "green_bor": 0.6, "green_all": 0.3, "warm_bor": 0.3, "legs_bot": 1.0,
                               "camo": 0.5, "dark_all": 0.2, "tex_bor": 0.2, "sky_all": -0.5, "blob_top": -0.3,
                               "white_all": -0.5, "gray_all": -0.5}),
        "horse": merge(ANIMAL, {"warm_cen": 1.0, "dark_cen": 0.6, "green_bot": 1.2, "field": 1.0, "legs_bot": 0.8,
                                "sky_top": 0.5, "bright_top": 0.3, "smooth_bor": 0.3, "tex_bor": -0.3,
                                "white_all": -0.3, "blob_top": -0.2}, cb(0.2)),
        "frog": merge(ANIMAL, {"green_cen": 1.8, "warm_cen": 0.5, "camo": 0.5, "dark_all": 0.3, "sky_all": -1.5,
                               "blue_all": -0.5, "white_all": -1.0, "bright_top": -0.5, "legs_bot": -0.5,
                               "blob_top": -0.2}, cb(-0.3)),
    }
    # offsets for the baseline each class collects from generic cues on a typical photo (estimated by hand)
    class_bias = {"airplane": -0.15, "automobile": -0.2, "bird": 0.45, "cat": 0.2, "deer": -0.2,
                  "dog": 0.45, "frog": 0.45, "horse": -0.1, "ship": 0.3, "truck": -0.6}
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    idx = {nm: i for i, nm in enumerate(cue_names)}
    Wo, bo = np.zeros((10, M)), np.zeros(10)
    for k, cname in enumerate(CLASSES):
        for cue_nm, wt in rules[cname].items():
            Wo[k, idx[cue_nm]] += wt
        bo[k] = class_bias[cname]
    net.dense("scores", cues, Wo, bo)
    return net, cue_names


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    net, cue_names = build()
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump({"layers": net.layers}, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **net.W)
    print(f"{len(net.layers)} layers, {len(cue_names)} cues, "
          f"{sum(int(np.prod(v.shape)) for v in net.W.values()):,} weights -> {out}")


if __name__ == "__main__":
    main()
