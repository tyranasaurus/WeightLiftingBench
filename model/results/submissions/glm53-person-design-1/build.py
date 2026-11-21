"""Builds a hand-designed network that classifies 96x96 colour photos into
person / no-person.

All weights are typed from general knowledge of what everyday photos containing
people look like (and of image statistics / published skin-tone ranges); no
training data is read or used by this program.

Pipeline:
  Stage 1 (one 7x7 conv): luminance + Gaussian blurs (sigma .8/1.6/3.2) and
    colour-opponent channels u=R-G, w=R-B, v=R+G-2B, gb=G-B (raw + blurred).
  Stage 2 (1x1 convs, relu, negation convs, adds, maxpool): rectified cue maps
    (skin-hue mask, eroded flat-skin mask, edge energy, dark spots/blobs,
    vertical dark bars, dark-above-skin, blue/green/dark/bright/saturation
    fractions, horizontal-edge energy, edge-inside-skin).
  Stage 3 (one big-kernel conv per cue + add + reshape): each cue map is
    spatially averaged (uniformly or with a centre/border pattern) and scaled
    by a hand-chosen confidence weight into a 2-class score contribution.
"""
import sys, json
import numpy as np
from pathlib import Path


def gkern(k, sigma):
    ax = np.arange(k) - (k // 2)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma))
    return g / g.sum()


def build(outdir):
    L, W = [], {}
    NCH = {"image": 3}

    def conv(name, inp, out, k, w, b, stride=1, pad=0):
        L.append({"name": name, "op": "conv", "inputs": [inp], "out": out,
                  "kernel": [k, k], "stride": stride, "padding": pad})
        W[name + ".W"] = np.ascontiguousarray(w, np.float32)
        W[name + ".b"] = np.ascontiguousarray(b, np.float32)
        NCH[name] = out

    def relu(name, inp):
        L.append({"name": name, "op": "relu", "inputs": [inp]})
        NCH[name] = NCH[inp]

    def add(name, ins):
        L.append({"name": name, "op": "add", "inputs": ins})
        NCH[name] = NCH[ins[0]]

    def pool(name, inp, op, size, stride):
        L.append({"name": name, "op": op, "inputs": [inp],
                  "size": list(size), "stride": list(stride)})
        NCH[name] = NCH[inp]

    def reshape(name, inp, shape):
        L.append({"name": name, "op": "reshape", "inputs": [inp], "shape": list(shape)})
        NCH[name] = shape

    # ---- 1x1 conv helper: rows = list of (channel->coef dict, bias) ----
    def c1x1(name, inp, rows):
        nin = NCH[inp]
        w = np.zeros((len(rows), nin, 1, 1), np.float64)
        b = np.zeros(len(rows), np.float64)
        for o, (dic, bias) in enumerate(rows):
            for c, co in dic.items():
                w[o, c, 0, 0] = co
            b[o] = bias
        conv(name, inp, len(rows), 1, w, b)

    def neg(name, inp):
        """channelwise negation via 1x1 conv"""
        n = NCH[inp]
        w = np.zeros((n, n, 1, 1), np.float64)
        for i in range(n):
            w[i, i, 0, 0] = -1.0
        conv(name, inp, n, 1, w, np.zeros(n))

    def pack2(name, inp, idx):
        """1-channel map -> 2-channel [.., map at slot idx, ..]"""
        c1x1(name, inp, [({0: 1.0}, 0.0) if idx == 0 else ({}, 0.0),
                         ({0: 1.0}, 0.0) if idx == 1 else ({}, 0.0)])

    def min2(name, t):
        """t is a 2-channel tensor [A, B]; returns 2-ch tensor [min(A,B), B].
        min(A,B) = A - relu(A - B)."""
        p = name + "_d"; c1x1(p, t, [({0: 1.0, 1: -1.0}, 0.0), ({}, 0.0)])
        r = name + "_r"; relu(r, p)
        n = name + "_n"; neg(n, r)
        a = name + "_a"; add(a, [t, n])
        return a

    def pick(name, t, idx, out2=False):
        rows = [({idx: 1.0}, 0.0)]
        if out2:
            rows = [({idx: 1.0}, 0.0), ({}, 0.0)] if idx == 0 else [({}, 0.0), ({idx: 1.0}, 0.0)]
        c1x1(name, t, rows)

    def trap(name, inp, ch, coef, a, b, c, slot=None):
        """Trapezoid band of x = coef*channel(ch) in [a,b,c,d] (d=c+b-a),
        flat top normalised to 1.  Outputs 1 channel, or 2 channels with the
        value in slot `slot` (other zero) when slot is given."""
        d = c + (b - a)
        pre = name + "_pre"
        c1x1(pre, inp, [({ch: coef}, -a), ({ch: coef}, -b),
                        ({ch: coef}, -c), ({ch: coef}, -d)])
        r = name + "_r"; relu(r, pre)
        h = b - a
        row = {0: 1.0 / h, 1: -1.0 / h, 2: -1.0 / h, 3: 1.0 / h}
        if slot is None:
            c1x1(name, r, [(row, 0.0)])
        elif slot == 0:
            c1x1(name, r, [(row, 0.0), ({}, 0.0)])
        else:
            c1x1(name, r, [({}, 0.0), (row, 0.0)])

    # ================= stage 1: linear filtering =================
    g08, g16, g32 = gkern(7, 0.8), gkern(7, 1.6), gkern(7, 3.2)
    lum = [0.299, 0.587, 0.114]
    w1 = np.zeros((12, 3, 7, 7), np.float64)
    for i, lc in enumerate(lum):
        w1[0, i, 3, 3] = lc                     # 0: luminance Y
        w1[1, i] = lc * g08                     # 1: blur(Y, .8)
        w1[2, i] = lc * g16                     # 2: blur(Y, 1.6)
        w1[3, i] = lc * g32                     # 3: blur(Y, 3.2)
    w1[4, 0, 3, 3] = 1; w1[4, 1, 3, 3] = -1     # 4: u = R-G
    w1[5, 0] = g16; w1[5, 1] = -g16             # 5: blur(u)
    w1[6, 0, 3, 3] = 1; w1[6, 2, 3, 3] = -1     # 6: w = R-B
    w1[7, 0] = g16; w1[7, 2] = -g16             # 7: blur(w)
    w1[8, 0, 3, 3] = 1; w1[8, 1, 3, 3] = 1; w1[8, 2, 3, 3] = -2   # 8: v = R+G-2B
    w1[9, 0] = g16; w1[9, 1] = g16; w1[9, 2] = -2 * g16           # 9: blur(v)
    w1[10, 1, 3, 3] = 1; w1[10, 2, 3, 3] = -1   # 10: gb = G-B
    w1[11, 1] = g16; w1[11, 2] = -g16           # 11: blur(gb)
    conv("c1", "image", 12, 7, w1, np.zeros(12), pad=3)

    # ================= stage 2: rectified cue maps =================
    # E1: fine-scale contrast |blur(.8)-blur(3.2)| of luminance
    c1x1("e_pre", "c1", [({1: 1.0, 3: -1.0}, 0.0), ({3: 1.0, 1: -1.0}, 0.0)])
    relu("e_r", "e_pre")
    c1x1("E1", "e_r", [({0: 1.0, 1: 1.0}, 0.0)])

    # DKSPOT: dark pixels vs large surround
    c1x1("dkp_pre", "c1", [({3: 1.0, 0: -1.0}, -0.09)])
    relu("DKSPOT", "dkp_pre")
    # DKBLOB: blurred darkening vs small surround (figure-scale blobs)
    c1x1("dkb_pre", "c1", [({3: 1.0, 1: -1.0}, -0.06)])
    relu("DKBLOB", "dkb_pre")

    # SKIN: trapezoid bands on blur(u), blur(w), blur(Y) + channelwise min
    trap("tb_u", "c1", 5, 1.0, 0.02, 0.05, 0.30, slot=0)   # skin hue R-G
    trap("tb_w", "c1", 7, 1.0, 0.03, 0.07, 0.38, slot=1)   # skin R-B
    add("sk2", ["tb_u", "tb_w"])                            # [BU, BW]
    m1 = min2("skm1", "sk2")                                # [min(BU,BW), BW]
    pick("sk3", m1, 0, out2=True)                           # [min2, 0]
    trap("tb_y", "c1", 2, 1.0, 0.12, 0.18, 0.95, slot=1)    # skin brightness
    add("sk4", ["sk3", "tb_y"])                             # [min2, BY]
    m2 = min2("skm2", "sk4")                                # [SKIN, BY]
    pick("SKIN", m2, 0)                                     # 1-ch skin mask

    # FLAT: eroded skin (interior of large flat skin-hued expanses)
    neg("flat_n1", "SKIN")
    pool("flat_mp", "flat_n1", "maxpool", (13, 13), (1, 1))   # -min filter, 84x84
    neg("FLAT", "flat_mp")

    # DARK / BRIGHT pixel fractions (hinges normalised to height 1)
    c1x1("dark_pre", "c1", [({0: -1.0}, 0.30)])
    relu("dark_r", "dark_pre")
    c1x1("DARK", "dark_r", [({0: 1.0 / 0.30}, 0.0)])
    c1x1("bright_pre", "c1", [({0: 1.0}, -0.78)])
    relu("bright_r", "bright_pre")
    c1x1("BRIGHT", "bright_r", [({0: 1.0 / 0.22}, 0.0)])

    # BLUE: blue-dominant pixels (sky / water); GREEN: green-dominant
    trap("BLUE", "c1", 9, -1.0, 0.15, 0.25, 1.00)
    trap("tg_gb", "c1", 11, -1.0, 0.04, 0.10, 0.40, slot=0)   # blur(G-B) high
    trap("tg_u", "c1", 5, -1.0, 0.04, 0.10, 0.40, slot=1)     # blur(R-G) low
    add("gr2", ["tg_gb", "tg_u"])
    mg = min2("grm", "gr2")
    pick("GREEN", mg, 0)

    # SAT: rough saturation |blur(u)| + |blur(gb)|
    c1x1("sat_pre", "c1", [({5: 1.0}, 0.0), ({5: -1.0}, 0.0),
                           ({11: 1.0}, 0.0), ({11: -1.0}, 0.0)])
    relu("sat_r", "sat_pre")
    c1x1("SAT", "sat_r", [({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}, 0.0)])

    # HE: horizontal-edge energy (vertical difference of blur1.6 luminance)
    whe = np.zeros((1, 12, 3, 3), np.float64)
    whe[0, 2, 0, 1] = -1.0; whe[0, 2, 2, 1] = 1.0
    conv("he_d", "c1", 1, 3, whe, np.zeros(1), pad=1)
    c1x1("he_pre", "he_d", [({0: 1.0}, 0.0), ({0: -1.0}, 0.0)])
    relu("he_r", "he_pre")
    c1x1("HE", "he_r", [({0: 1.0, 1: 1.0}, 0.0)])

    # VBAR: 9-tall x 3-wide mean of DKSPOT (vertical dark figure bars)
    # VB2:  same on DKBLOB
    def vbox9x3(name, inp):
        wk = np.zeros((1, 1, 9, 9), np.float64)
        wk[0, 0, :, 3:6] = 1.0 / 27.0
        conv(name, inp, 1, 9, wk, np.zeros(1), pad=4)
    vbox9x3("vbar", "DKSPOT"); trap("VB", "vbar", 0, 1.0, 0.08, 0.18, 0.45)
    vbox9x3("vb2x", "DKBLOB"); trap("VB2", "vb2x", 0, 1.0, 0.08, 0.18, 0.45)

    # EIS: texture inside skin regions = min(10*E1, SKIN)
    c1x1("e1s", "E1", [({0: 10.0}, 0.0)])
    pack2("eis_p", "e1s", 0)
    pack2("eis_q", "SKIN", 1)
    add("eis2", ["eis_p", "eis_q"])
    me = min2("eism", "eis2")
    pick("EIS", me, 0)

    # HEAD: skin with dark (hair/hat) above it
    whd = np.zeros((1, 1, 9, 9), np.float64)
    for r in range(0, 4):
        whd[0, 0, r, 4] = 1.0        # 4 rows above centre
    conv("headab", "DKSPOT", 1, 9, whd, np.zeros(1), pad=4)
    c1x1("head_pre", "headab", [({0: 2.0}, -1.2)])   # (HEADAB - .6)/.2
    relu("head_r", "head_pre")
    pack2("head_p", "head_r", 0)
    pack2("head_q", "SKIN", 1)
    add("head2", ["head_p", "head_q"])
    mh = min2("headm", "head2")
    pick("HEAD", mh, 0)

    # ================= stage 3: readout =================
    # Each cue map -> 2-channel score contribution via one full-size conv that
    # averages the map (uniformly, or with a centre/border pattern) and applies
    # a hand-chosen confidence weight alpha.  A per-cue bias subtracts the
    # value the cue takes for a TYPICAL everyday photo (estimated from general
    # knowledge of image statistics), so only the deviation from typical counts
    # as evidence.  Hand-set signs, from general knowledge of the classes:
    #   + person photos: textured skin (faces/limbs), blue sky/water (outdoor
    #     people scenes), centre sharpness (portraits), fine texture, dark
    #     tones, vertical dark figures, heads (dark hair above skin)
    #   - no-person photos (close-ups of objects/food, flat scenes): flat
    #     skin-hued expanses (wood/sand/walls), high saturation, overall
    #     skin-hue area, greenery.
    outs = []

    def readout(name, inp, k, alpha, typ, spatial=None):
        s_ = np.ones((k, k), np.float64) if spatial is None else spatial
        wk = np.zeros((2, 1, k, k), np.float64)
        wk[0, 0] = -alpha * s_ / (k * k)
        wk[1, 0] = alpha * s_ / (k * k)
        bb = np.float32([alpha * typ, -alpha * typ])
        conv(name, inp, 2, k, wk, bb, stride=k, pad=0)
        outs.append(name)

    # bounded transforms for the raw-scale cues; each trapezoid is centred so
    # that the value typical of an everyday photo maps to ~0.5, with the
    # rising edge one typical across-image variation wide.
    trap("E1N", "E1", 0, 1.0, 0.015, 0.05, 0.20)   # fine-scale contrast
    trap("EISN", "EIS", 0, 1.0, 0.05, 0.25, 1.00)   # texture inside skin
    trap("VBN", "VB", 0, 1.0, 0.005, 0.05, 0.25)    # vertical dark bars
    trap("VB2N", "VB2", 0, 1.0, 0.005, 0.05, 0.25)
    trap("HEADN", "HEAD", 0, 1.0, 0.005, 0.08, 0.35) # dark above skin
    trap("SATN", "SAT", 0, 1.0, 0.12, 0.35, 0.95)   # saturation
    trap("HEN", "HE", 0, 1.0, 0.005, 0.025, 0.09)   # horizontal edges

    readout("ro_eis", "EISN", 96, +0.20, 0.30)   # texture inside skin regions
    readout("ro_blue", "BLUE", 96, +0.20, 0.12) # sky/water -> outdoor people
    readout("ro_e1", "E1N", 96, +0.14, 0.30)    # overall fine texture
    # centre-sharpness pattern: + centre, - border (portrait-style photos)
    sc = np.full((96, 96), -0.4, np.float64); sc[20:76, 20:76] = 1.0
    readout("ro_e1c", "E1N", 96, +0.16, 0.025, spatial=sc)
    readout("ro_dark", "DARK", 96, +0.12, 0.20) # dark tones
    readout("ro_vb", "VBN", 96, +0.07, 0.25)    # vertical dark figures
    readout("ro_vb2", "VB2N", 96, +0.05, 0.25)
    readout("ro_head", "HEADN", 96, +0.06, 0.30) # heads: dark above skin
    readout("ro_he", "HEN", 96, +0.05, 0.35)     # horizontal structure
    readout("ro_bright", "BRIGHT", 96, +0.03, 0.10)
    readout("ro_flat", "FLAT", 84, -0.12, 0.04)  # flat skin-hued expanses
    readout("ro_sat", "SATN", 96, -0.11, 0.40)   # saturation (food close-ups)
    readout("ro_skin", "SKIN", 96, -0.06, 0.30)  # overall skin-hue area
    readout("ro_green", "GREEN", 96, -0.06, 0.10)  # greenery
    # skin-hue concentrated in the bottom half (tables, floors, sand)
    sb = np.zeros((96, 96), np.float64); sb[48:, :] = 1.0
    readout("ro_skinbot", "SKIN", 96, -0.04, 0.15, spatial=sb)

    # dark detail inside bright regions: small dark figures against
    # sky / snow / water (skiers, surfers, sailors, ball sports)
    pool("br12", "BRIGHT", "avgpool", (12, 12), (12, 12))       # (1,8,8)
    pool("dk12", "DKSPOT", "avgpool", (12, 12), (12, 12))       # (1,8,8)
    c1x1("dk12s", "dk12", [({0: 12.0}, 0.0)])                   # scale up
    pack2("dbr_p", "br12", 0)
    pack2("dbr_q", "dk12s", 1)
    add("dbr2", ["dbr_p", "dbr_q"])
    mdb = min2("dbrm", "dbr2")
    pick("DKBR", mdb, 0)                                        # (1,8,8)
    readout("ro_dkbr", "DKBR", 8, +0.08, 0.06)   # dark figures in bright areas

    # small global bias (class prior is close to balanced)
    wbb = np.zeros((2, 12, 96, 96), np.float64)
    conv("ro_bias", "c1", 2, 96, wbb, np.float32([-0.04, 0.04]), stride=96, pad=0)
    outs.append("ro_bias")

    add("score", outs)
    reshape("out", "score", [2])

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "network.json").write_text(json.dumps({"layers": L}))
    np.savez(outdir / "weights.npz",
             **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ".")
