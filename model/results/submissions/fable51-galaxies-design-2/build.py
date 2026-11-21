"""build.py: writes network.json and weights.npz for a hand-designed galaxy classifier.

Usage: python3 build.py OUT_DIR

Every weight below is produced from formulas and hand-chosen constants (no data is read).
Pipeline (all inside the network):
  1. sky-subtracted luminance, 3x3 box smoothing
  2. morphological geodesic reconstruction from a central seed (iterated 3x3 max-dilation clipped
     by the image): isolates the central galaxy and removes secondary peaks (companions, stars)
  3. soft isophote masks at several levels; areas, directional second moments (16 angles) -> axis ratios,
     central/total flux, annulus colours, secondary-peak ("dome") height, clumpiness, 180-degree asymmetry
  4. a Galaxy-Zoo-like decision tree written with ReLU step units
"""
import json, os, sys
import numpy as np

H = W_ = 128
CENTER = 63.5
BG = 0.125           # sky level of the JPEG cutouts (near black), same in all channels
N_ITER = 50          # geodesic reconstruction iterations (1 px per iteration)
LEVELS = [0.08, 0.25, 0.50, 0.75, 0.05]   # isophote levels above sky; the last is a loose envelope for dome detection
LEVEL_W = [0.04, 0.04, 0.04, 0.04, 0.03]  # softness of the isophote masks
N_DIR = 16           # directions for second moments
GATE = 30.0          # weight of the top-level (edge-on vs not) decision


class Net:
    def __init__(self):
        self.layers, self.weights, self.shapes, self.n = [], {}, {"image": (3, H, W_)}, 0

    def _nm(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def _emit(self, L, shape):
        self.layers.append(L)
        self.shapes[L["name"]] = shape
        return L["name"]

    def conv(self, inp, W, b=None, stride=1, pad=0, prefix="conv"):
        W = np.asarray(W, np.float32)
        o, i, k, _ = W.shape
        c, h, w = self.shapes[inp]
        assert i == c, (inp, W.shape, self.shapes[inp])
        if b is None:
            b = np.zeros(o, np.float32)
        name = self._nm(prefix)
        self.weights[name + ".W"] = W
        self.weights[name + ".b"] = np.asarray(b, np.float32).reshape(o)
        ho, wo = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
        return self._emit({"name": name, "op": "conv", "inputs": [inp], "out": o, "kernel": [k, k], "stride": stride, "padding": pad}, (o, ho, wo))

    def dense(self, inp, W, b=None, prefix="dense"):
        W = np.asarray(W, np.float32)
        o, i = W.shape
        assert i == int(np.prod(self.shapes[inp])), (inp, W.shape, self.shapes[inp])
        if b is None:
            b = np.zeros(o, np.float32)
        name = self._nm(prefix)
        self.weights[name + ".W"] = W
        self.weights[name + ".b"] = np.asarray(b, np.float32).reshape(o)
        return self._emit({"name": name, "op": "dense", "inputs": [inp], "out": o}, (o,))

    def relu(self, inp):
        return self._emit({"name": self._nm("relu"), "op": "relu", "inputs": [inp]}, self.shapes[inp])

    def pool(self, inp, size, stride, kind):
        c, h, w = self.shapes[inp]
        ho, wo = (h - size[0]) // stride[0] + 1, (w - size[1]) // stride[1] + 1
        return self._emit({"name": self._nm(kind), "op": kind, "inputs": [inp], "size": list(size), "stride": list(stride)}, (c, ho, wo))

    def maxpool(self, inp, size, stride):
        return self.pool(inp, size, stride, "maxpool")

    def avgpool(self, inp, size, stride):
        return self.pool(inp, size, stride, "avgpool")

    def add(self, *inps):
        s = self.shapes[inps[0]]
        assert all(self.shapes[i] == s for i in inps), [self.shapes[i] for i in inps]
        return self._emit({"name": self._nm("add"), "op": "add", "inputs": list(inps)}, s)

    def reshape(self, inp, shape):
        assert int(np.prod(shape)) == int(np.prod(self.shapes[inp]))
        return self._emit({"name": self._nm("reshape"), "op": "reshape", "inputs": [inp], "shape": list(shape)}, tuple(shape))

    # ----- composites -----
    def scale(self, inp, a, b=0.0):
        """per-channel affine map a*x+b on a (C,H,W) tensor (a, b scalars or per-channel)."""
        c = self.shapes[inp][0]
        a = np.broadcast_to(np.asarray(a, np.float32), (c,))
        b = np.broadcast_to(np.asarray(b, np.float32), (c,))
        W = np.zeros((c, c, 1, 1), np.float32)
        W[np.arange(c), np.arange(c), 0, 0] = a
        return self.conv(inp, W, b, prefix="scale")

    def chan_mix(self, inp, A, b=None):
        """1x1 conv with matrix A (out, in)."""
        A = np.asarray(A, np.float32)
        return self.conv(inp, A[:, :, None, None], b, prefix="mix")

    def neg(self, inp):
        return self.scale(inp, -1.0)

    def minimum(self, a, neg_b):
        """min(a, b) = a - relu(a - b), given a and -b."""
        d = self.relu(self.add(a, neg_b))
        return self.add(a, self.neg(d))

    def clip01(self, x):
        """min(x, 1) for x >= 0 : 1 - relu(1 - x)."""
        return self.scale(self.relu(self.scale(x, -1.0, 1.0)), -1.0, 1.0)

    def mask_mul(self, x, m, big=2.0):
        """x * m for m in {0,1}: relu(x - big*(1-m)) = relu(x + big*m - big)."""
        return self.relu(self.add(x, self.scale(m, big, -big)))

    def const_map(self, src_scalar, template, prefix="tmpl"):
        """a fixed (1,H,W) map from a formula template; src_scalar must have shape (1,1,1)."""
        t = np.asarray(template, np.float32)
        d = self.dense(src_scalar, np.zeros((t.size, 1), np.float32), t.reshape(-1), prefix=prefix)
        return self.reshape(d, (1,) + t.shape)

    def shifts(self, x, offsets):
        """(1,h,w) -> (len(offsets),h,w), channel j = x shifted so that out[y,x] = in[y+dy, x+dx]."""
        m = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
        k = 2 * m + 1
        W = np.zeros((len(offsets), 1, k, k), np.float32)
        for j, (dy, dx) in enumerate(offsets):
            W[j, 0, m + dy, m + dx] = 1.0
        return self.conv(x, W, pad=m, prefix="shift")

    def vec_min(self, v):
        """min over a vector (n,) -> (1,) via -max(-v)."""
        n = self.shapes[v][0]
        nv = self.dense(v, -np.eye(n, dtype=np.float32), prefix="negvec")
        mx = self.maxpool(self.reshape(nv, (1, n, 1)), [n, 1], [n, 1])
        return self.dense(self.reshape(mx, (1,)), -np.ones((1, 1), np.float32), prefix="negback")


def coords():
    yy, xx = np.mgrid[0:H, 0:W_].astype(np.float64)
    return xx - CENTER, yy - CENTER


def build():
    net = Net()
    X, Y = coords()
    R2 = X ** 2 + Y ** 2

    # ---------- 1. luminance, sky subtraction, smoothing ----------
    lum = net.conv("image", np.full((1, 3, 1, 1), 1 / 3, np.float32), [-BG], prefix="lum")
    box = np.full((1, 1, 3, 3), 1 / 9, np.float32)
    lum_s = net.conv(lum, box, pad=1, prefix="smooth")
    lpos = net.relu(lum_s)                      # L' >= 0, sky ~ 0
    neg_lpos = net.neg(lpos)

    # a (1,1,1) scalar to hang constant templates on
    gap = net.avgpool(lpos, [H, W_], [H, W_])

    # ---------- 2. geodesic reconstruction from the centre ----------
    seed_disk = (R2 <= 2.5 ** 2).astype(np.float32)
    neg_disk = net.const_map(gap, -seed_disk, prefix="seed")
    S = net.minimum(lpos, neg_disk)             # seed = L' inside the central disk, 0 outside
    ident = np.ones((1, 1, 1, 1), np.float32)
    for _ in range(N_ITER):
        P = net.conv(S, ident, pad=1, prefix="pad")           # (1,130,130), zero border
        D = net.maxpool(P, [3, 3], [1, 1])                    # 3x3 dilation
        S = net.minimum(D, neg_lpos)                          # clip by the image
    recon = S                                                 # R: central galaxy only, secondary peaks flattened

    # masked light and domes
    F = net.relu(net.scale(recon, 1.0, -0.04))                # masked light, sky noise removed
    domes = net.relu(net.add(lpos, net.neg(recon)))           # L' - R >= 0: secondary peaks

    # ---------- 3. soft isophote masks ----------
    nl = 4                                                    # levels used for shape measurements
    nm_ = len(LEVELS)
    Wm = np.array([1 / w for w in LEVEL_W], np.float32).reshape(nm_, 1, 1, 1)
    bm = np.array([-t / w for t, w in zip(LEVELS, LEVEL_W)], np.float32)
    M = net.clip01(net.relu(net.conv(recon, Wm, bm, prefix="levels")))   # (5,128,128) in [0,1]

    # directional second moments: V[l,k] = sum M_l (x cos t_k + y sin t_k)^2
    thetas = np.arange(N_DIR) * np.pi / N_DIR
    proj2 = np.stack([(X * np.cos(t) + Y * np.sin(t)) ** 2 for t in thetas])   # (16,H,W)
    Wv = np.zeros((nl * N_DIR, nm_, H, W_), np.float32)
    for l in range(nl):
        Wv[l * N_DIR:(l + 1) * N_DIR, l] = proj2
    V = net.dense(M, Wv.reshape(nl * N_DIR, -1), prefix="moments")
    lam1 = net.reshape(net.maxpool(net.reshape(V, (1, nl * N_DIR, 1)), [N_DIR, 1], [N_DIR, 1]), (nl,))
    negV = net.dense(V, -np.eye(nl * N_DIR, dtype=np.float32), prefix="negV")
    neg_lam2 = net.reshape(net.maxpool(net.reshape(negV, (1, nl * N_DIR, 1)), [N_DIR, 1], [N_DIR, 1]), (nl,))

    # areas and centroid sums per level
    Wa = np.zeros((3 * nl, nm_, H, W_), np.float32)
    for l in range(nl):
        Wa[l, l] = 1.0
        Wa[nl + l, l] = X
        Wa[2 * nl + l, l] = Y
    area_cent = net.dense(M, Wa.reshape(3 * nl, -1), prefix="areas")       # (12,)

    # fluxes: total, r<5, r<10, r<3
    Wf = np.stack([np.ones_like(R2), R2 <= 25, R2 <= 100, R2 <= 9]).astype(np.float32)
    fluxes = net.dense(F, Wf.reshape(4, -1), prefix="flux")                # (4,)

    # colours over two annuli: outer (levels 0..1) and inner (levels 1..2)
    rb = net.conv("image", np.array([[[[1]], [[0]], [[0]]], [[[0]], [[0]], [[1]]]], np.float32), [-BG, -BG], prefix="rb")   # (2,H,W): red-sky, blue-sky
    rb = net.relu(rb)
    ann_out = net.chan_mix(M, np.array([[1, 0, -1, 0, 0], [1, 0, -1, 0, 0]], np.float32))    # annulus between levels 0.08 and 0.5, duplicated for 2 channels
    ann_in = net.chan_mix(M, np.array([[0, 1, 0, -1, 0], [0, 1, 0, -1, 0]], np.float32))     # annulus between levels 0.25 and 0.75
    rb_out = net.mask_mul(rb, ann_out)
    rb_in = net.mask_mul(rb, ann_in)
    Wsum2 = np.zeros((2, 2, H, W_), np.float32)
    Wsum2[0, 0] = 1
    Wsum2[1, 1] = 1
    col_out = net.dense(rb_out, Wsum2.reshape(2, -1), prefix="colout")    # (Cr_o, Cb_o)
    col_in = net.dense(rb_in, Wsum2.reshape(2, -1), prefix="colin")       # (Cr_i, Cb_i)

    # domes inside the (3 px dilated) envelope; nuclei only within r<24 of the centre
    env = net.chan_mix(M, np.array([[0, 0, 0, 0, 1]], np.float32))         # envelope at the 0.05 level (1,H,W)
    env = net.maxpool(net.conv(env, ident, pad=3, prefix="pad"), [7, 7], [1, 1])   # dilate by 3 px
    dm_env = net.mask_mul(domes, env)
    # clumpiness: sum of clip(dome - 0.05, 0, 0.1) over the envelope (each pixel capped so one star cannot dominate)
    dc = net.relu(net.scale(dm_env, 1.0, -0.05))
    dc = net.add(dc, net.neg(net.relu(net.scale(dc, 1.0, -0.1))))
    clump = net.dense(dc, np.ones((1, H * W_), np.float32), prefix="clump")
    disk28 = net.const_map(gap, (R2 <= 28 ** 2).astype(np.float32), prefix="disk28")
    dm = net.mask_mul(dm_env, disk28)
    dome_max = net.reshape(net.maxpool(dm, [H, W_], [H, W_]), (1,))
    dome_ext = net.dense(net.relu(net.scale(dm, 1.0, -0.1)), np.ones((1, H * W_), np.float32), prefix="domeext")   # sum of (dome-0.1)+ : large for extended nuclei

    # outer-isophote centroid offset |(mx, my)| ~ max over 8 directions
    ang8 = np.arange(8) * np.pi / 4
    Woff = np.zeros((8, 3 * nl), np.float32)
    Woff[:, nl + 0] = np.cos(ang8)
    Woff[:, 2 * nl + 0] = np.sin(ang8)
    off8 = net.dense(area_cent, Woff, prefix="off8")
    offset = net.reshape(net.maxpool(net.reshape(off8, (1, 8, 1)), [8, 1], [8, 1]), (1,))

    # 180-degree asymmetry at 64x64, minimised over 3x3 centre shifts
    F64 = net.avgpool(F, [2, 2], [2, 2])
    g1 = np.exp(-np.arange(-2, 3) ** 2 / (2 * 1.2 ** 2))
    g2 = np.outer(g1, g1); g2 /= g2.sum()
    F64 = net.conv(F64, g2[None, None].astype(np.float32), pad=2, prefix="blur")   # Gaussian blur (sigma 1.2 px) before the symmetry test
    n64 = 64 * 64
    Prot = np.zeros((n64, n64), np.float32)
    idx = np.arange(n64)
    ii, jj = idx // 64, idx % 64
    Prot[(63 - ii) * 64 + (63 - jj), idx] = 1.0
    Frot = net.reshape(net.dense(F64, Prot, prefix="rot180"), (1, 64, 64))
    offs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    Frot9 = net.shifts(Frot, offs)                                          # (9,64,64)
    F9 = net.conv(F64, np.ones((9, 1, 1, 1), np.float32), prefix="bcast")  # (9,64,64)
    diff = net.add(F9, net.neg(Frot9))
    absd = net.add(net.relu(diff), net.relu(net.neg(diff)))
    Wsum9 = np.zeros((9, 9, 64, 64), np.float32)
    for j in range(9):
        Wsum9[j, j] = 1.0
    asym9 = net.dense(absd, Wsum9.reshape(9, -1), prefix="asym9")
    asym = net.vec_min(asym9)                                               # (1,)

    # ---------- assemble the feature vector ----------
    # 0-3 A_l | 4-7 lam1_l | 8-11 lam2_l | 12-15 mx_l | 16-19 my_l | 20 Ftot 21 F_r5 22 F_r10 23 F_r3 |
    # 24 Cr_o 25 Cb_o 26 Cr_i 27 Cb_i | 28 dome_max 29 clump 30 asym | 31 const 1 | 32 dome_ext 33 offset
    NF = 36

    def embed(src, rows, sign=1.0, prefix="emb", bias=None):
        n_in = int(np.prod(net.shapes[src]))
        E = np.zeros((NF, n_in), np.float32)
        for j, r in enumerate(rows):
            E[r, j] = sign
        return net.dense(src, E, bias, prefix=prefix)

    parts = [
        embed(area_cent, list(range(0, 4)) + list(range(12, 20))),
        embed(lam1, [4, 5, 6, 7]),
        embed(neg_lam2, [8, 9, 10, 11], sign=-1.0),
        embed(fluxes, [20, 21, 22, 23]),
        embed(col_out, [24, 25]),
        embed(col_in, [26, 27]),
        embed(dome_max, [28]),
        embed(clump, [29]),
        embed(asym, [30], bias=np.eye(1, NF, 31, dtype=np.float32)[0]),
        embed(dome_ext, [32]),
        embed(offset, [33]),
    ]
    feat = net.add(*parts)

    # ---------- 4. decision logic ----------
    A, L1, L2 = lambda l: l, lambda l: 4 + l, lambda l: 8 + l
    FTOT, FR5, FR10, FR3, CRO, CBO, CRI, CBI, DOME, CLUMP, ASYM, ONE, DEXT, OFF = 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33

    steps = {}   # name -> (z vector over features, width); step = clip(z / width, 0, 1)

    def step(name, terms, width, const=0.0):
        z = np.zeros(NF, np.float32)
        for idx_, coef in terms:
            z[idx_] += coef
        z[ONE] += const
        steps[name] = (z, width)

    # axis-ratio tests: q < q0  <=>  q0^2 * lam1 - lam2 > 0
    step("edge1", [(L1(1), 0.16), (L2(1), -1)], 400)      # q_mid < 0.40
    step("edge2", [(L1(0), 0.13), (L2(0), -1)], 600)      # q_out < 0.36
    step("cigar", [(L1(1), 0.25), (L2(1), -1)], 400)      # q_mid < 0.50
    step("round", [(L2(1), 1), (L1(1), -0.5625)], 400)    # q_mid > 0.75
    step("coreA", [(A(3), 1)], 8, -15)                     # A_core > 15 px (saturated core present)
    step("coreA2", [(A(3), 1)], 10, -40)                   # A_core > 40 px (core clearly larger than the PSF)
    step("inA", [(A(2), 1)], 10, -40)                      # A_in > 40 px
    step("coreround", [(L2(3), 1), (L1(3), -0.25)], 80)   # q_core > 0.50
    # bar ladder: an inner isophote (mid/in/core) much more elongated than the outer one
    BARQ = [0.35, 0.45, 0.55, 0.65]
    INNER = {"mid": 1, "in": 2, "core": 3}
    for q0 in BARQ:
        for nm_in, l in INNER.items():
            step(f"q{nm_in}_lt{q0}", [(L1(l), q0 ** 2), (L2(l), -1)], [400, 150, 80][l - 1])   # q_l < q0
        step(f"qout_gt{q0}", [(L2(0), 1), (L1(0), -(q0 + 0.15) ** 2)], 600)                 # q_out > q0 + 0.15
    # disk+bulge ladder (edge-on lenticulars): outer isophote flat, inner isophote round
    DISKQ = [0.35, 0.45, 0.55, 0.65, 0.75]
    DISKQ_EDGE = [0.35, 0.45]                               # rungs that also count as edge-on
    for q0 in DISKQ:
        step(f"qout_lt{q0}", [(L1(0), q0 ** 2), (L2(0), -1)], 600)                          # q_out < q0
        step(f"qin_gt{q0}", [(L2(2), 1), (L1(2), -(q0 + 0.25) ** 2)], 150)                   # q_in > q0 + 0.25
        step(f"qmid_lt{q0}", [(L1(1), q0 ** 2), (L2(1), -1)], 400)                          # q_mid < q0
        step(f"qcore_gt{q0}", [(L2(3), 1), (L1(3), -(q0 + 0.25) ** 2)], 80)                  # q_core > q0 + 0.25
    # colour ladder on the outer annulus: c_k = 1 if Cr/Cb < rho_k (bluer than rho_k)
    RHO = [0.67, 0.77, 0.87, 1.0, 1.15, 1.3, 1.5]
    for k, rho in enumerate(RHO):
        step(f"col{k}", [(CBO, rho), (CRO, -1)], 4)
    step("red1", [(CRO, 1), (CBO, -1.15)], 4)              # outer annulus moderately red
    step("blue", [(CBO, 1), (CRO, -1.0)], 4)               # outer annulus blue
    step("redin", [(CRI, 1), (CBI, -1.2)], 3)              # inner annulus red
    # size ladder on the outer area
    AREAS = [200, 400, 700, 1100, 1600, 2400]
    for k, a0 in enumerate(AREAS):
        step(f"size{k}", [(A(0), 1)], 0.15 * a0, -a0)
    step("tiny", [(A(1), -1)], 20, 90)                     # A_mid < 90
    # bulge-fraction ladder: A_in / A_out > f_k
    BF = [0.04, 0.07, 0.10, 0.14, 0.18]
    for k, f0 in enumerate(BF):
        step(f"bf{k}", [(A(2), 1), (A(0), -f0)], 10)
    step("conc", [(FR5, 1), (FTOT, -0.40)], 4)             # very concentrated (small bright objects)
    # structure
    step("nuc", [(DOME, 1)], 0.05, -0.20)
    step("nuc2", [(DOME, 1)], 0.05, -0.45)
    step("ext", [(DEXT, 1), (DOME, -10)], 1)               # secondary peak is extended (not a star)
    step("clumpy", [(CLUMP, 1), (FTOT, -0.01)], 1.0, -2.0)
    step("asym", [(ASYM, 1), (FTOT, -0.05)], 2.0, -2.0)
    step("asym2", [(ASYM, 1), (FTOT, -0.10)], 2.0, -2.0)
    step("off1", [(OFF, 1), (A(0), -2.5)], 200)            # outer centroid offset > 2.5 px
    step("off2", [(OFF, 1), (A(0), -5.0)], 200)            # > 5 px

    names = list(steps)
    ns = len(names)
    # layer 1: two relus per step: relu(z/w), relu(z/w - 1)
    W1 = np.zeros((2 * ns, NF), np.float32)
    b1 = np.zeros(2 * ns, np.float32)
    for j, nm in enumerate(names):
        z, w = steps[nm]
        W1[2 * j] = z / w
        W1[2 * j + 1] = z / w
        b1[2 * j + 1] = -1.0
    h1 = net.relu(net.dense(feat, W1, b1, prefix="logic1"))

    # layer 2: s_j = relu(z/w) - relu(z/w-1) in [0,1]; plus AND / NOR pre-activations
    S = {nm: j for j, nm in enumerate(names)}
    ands = [("bulgecore", ["coreround", "coreA2"], "and"),
            ("nucx", ["nuc", "ext"], "and")]
    bar_rungs, disk_rungs = [], []
    for q0 in BARQ:
        for nm_in in INNER:
            ands.append((f"bar_{nm_in}{q0}", [f"q{nm_in}_lt{q0}", f"qout_gt{q0}"] + (["coreA"] if nm_in == "core" else []), "and"))
            bar_rungs.append(f"bar_{nm_in}{q0}")
    for q0 in DISKQ:
        ands.append((f"diskedge{q0}", [f"qout_lt{q0}", f"qin_gt{q0}", "inA"], "and"))
        disk_rungs.append(f"diskedge{q0}")
        ands.append((f"diskedgeB{q0}", [f"qmid_lt{q0}", f"qcore_gt{q0}", "coreA2"], "and"))
        disk_rungs.append(f"diskedgeB{q0}")
    n2 = ns + len(ands)
    W2 = np.zeros((n2, 2 * ns), np.float32)
    b2 = np.zeros(n2, np.float32)
    for j in range(ns):
        W2[j, 2 * j] = 1.0
        W2[j, 2 * j + 1] = -1.0
    for k, (nm, ins, kind) in enumerate(ands):
        row = ns + k
        sgn = 1.0 if kind == "and" else -1.0
        for i_ in ins:
            W2[row, 2 * S[i_]] = sgn
            W2[row, 2 * S[i_] + 1] = -sgn
        b2[row] = -(len(ins) - 1.0) if kind == "and" else 1.0
        S[nm] = row
    h2 = net.relu(net.dense(h1, W2, b2, prefix="logic2"))

    # layer 3: pass-through plus OR-sums (x = min(1, sum) is realised as relu(sum) - relu(sum - 1)) and NOTs
    comps3 = [("barsum", {r: 1.0 for r in bar_rungs}, 0.0),
              ("barsum_m1", {r: 1.0 for r in bar_rungs}, -1.0),
              ("disksum", {r: 1.0 for r in disk_rungs}, 0.0),
              ("disksum_m1", {r: 1.0 for r in disk_rungs}, -1.0),
              ("notround", {"round": -1}, 1.0),
              ("face", {**{"edge1": -1.0, "edge2": -1.0}, **{f"diskedge{q0}": -1.0 for q0 in DISKQ_EDGE}, **{f"diskedgeB{q0}": -1.0 for q0 in DISKQ_EDGE}}, 1.0)]   # NOT edge-on = relu(1 - sum)
    n3 = n2 + len(comps3)
    W3 = np.zeros((n3, n2), np.float32)
    b3 = np.zeros(n3, np.float32)
    W3[:n2, :n2] = np.eye(n2)
    T = dict(S)
    for k, (nm, terms, c) in enumerate(comps3):
        row = n2 + k
        for i_, coef in terms.items():
            W3[row, S[i_]] = coef
        b3[row] = c
        T[nm] = row
    h3 = net.relu(net.dense(h2, W3, b3, prefix="logic3"))

    # layer 4: class scores
    W4 = np.zeros((10, n3), np.float32)
    b4 = np.zeros(10, np.float32)

    def addterm(cls, terms, k=1.0):
        for nm, coef in terms.items():
            W4[cls, T[nm]] += k * coef

    def const(cls, c):
        b4[cls] += c

    BAR = {"barsum": 1.0, "barsum_m1": -1.0}          # any bar rung fired
    DISKEDGE = {"disksum": 1.0, "disksum_m1": -1.0}   # any disk+bulge rung fired (bulge rounder than the outer isophote)
    COLOUR = {f"col{k}": 1.0 for k in range(len(RHO))}    # 0 (very red) .. 7 (very blue)
    SIZE = {f"size{k}": 1.0 for k in range(len(AREAS))}   # 0 (tiny) .. 6 (huge)
    BULGE = {f"bf{k}": 1.0 for k in range(len(BF))}       # 0 (disky) .. 5 (bulge dominated)

    # top-level gate: edge-on classes get GATE*(1-face), the rest GATE*face
    for c in range(10):
        if c in (8, 9):
            const(c, GATE); addterm(c, {"face": -GATE})
        else:
            addterm(c, {"face": GATE})
    # edge-on branch: bulge cues; class 9 iff at least two cues
    # shape-based cues (round bulge vs flat disk) weigh 2, non-specific cues (bright/red centre) weigh 1.2:
    # class 9 needs two shape cues, one shape cue + one other, or three non-specific cues
    addterm(9, {"red1": 1.2, "redin": 1.2, "coreA": 1.2, "bf3": 1.2, "bulgecore": 2}); addterm(9, DISKEDGE, 2); const(9, -0.4)
    const(8, 2.5)
    # featuredness axis (positive = featured disk, negative = smooth)
    PHI = {}
    def phi_add(terms, k):
        for nm, coef in terms.items():
            PHI[nm] = PHI.get(nm, 0.0) + k * coef
    phi_add(COLOUR, 0.8); phi_add({"clumpy": 1.5, "asym": 0.5}, 1.0)
    phi_add(SIZE, 0.4); phi_add(BULGE, -0.4); phi_add(BAR, 1.5); phi_add(DISKEDGE, 1.0)
    PHI_CONST = -0.8 * 3.5 - 0.4 * 3 + 0.4 * 2.5      # centre the ladders
    # merger: an extended second nucleus overrides the edge-on gate (close pairs look elongated)
    addterm(1, {"nucx": 8 + 1.5 * GATE, "nuc": 3, "nuc2": 2, "asym2": 2.0, "off2": 2.0, "off1": 1.0}); const(1, -1.0)
    # smooth classes
    addterm(2, PHI, -1); const(2, -PHI_CONST); addterm(2, {"round": 3.0, "tiny": 1.5}); const(2, 0.5)
    addterm(3, PHI, -1); const(3, -PHI_CONST); addterm(3, {"notround": 3.0, "cigar": -1.0})
    addterm(4, PHI, -1); const(4, -PHI_CONST); addterm(4, {"cigar": 3.0, "red1": 1.0}); const(4, -3.0)
    # featured classes
    FBASE = 2.5
    addterm(5, PHI, 1); const(5, PHI_CONST); addterm(5, BAR, 4.0); const(5, FBASE - 0.5)
    addterm(6, PHI, 1); const(6, PHI_CONST); addterm(6, BULGE, 0.5); addterm(6, COLOUR, -0.3); addterm(6, DISKEDGE, 1.5); addterm(6, {"redin": 0.5, "clumpy": -1.5}); const(6, FBASE + 0.5 + 0.3 * 3.5 - 0.5 * 2.5)
    addterm(7, PHI, 1); const(7, PHI_CONST); addterm(7, BULGE, -0.5); addterm(7, COLOUR, 0.3); addterm(7, {"clumpy": 1.5, "asym": 0.5}); const(7, FBASE - 0.3 * 3.5 + 0.5 * 2.5)
    # disturbed
    addterm(0, PHI, 0.5); const(0, 0.5 * PHI_CONST); addterm(0, {"asym2": 2.0, "off1": 1.5}); const(0, -2.0)

    scores = net.dense(h3, W4, b4, prefix="scores")
    return net


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    net = build()
    assert net.shapes[net.layers[-1]["name"]] == (10,)
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump({"layers": net.layers}, f)
    np.savez(os.path.join(out, "weights.npz"), **{k: v.astype(np.float32) for k, v in net.weights.items()})
    total = sum(int(v.size) for v in net.weights.values())
    print(f"{len(net.layers)} layers, {total:,} weights")


if __name__ == "__main__":
    main()
