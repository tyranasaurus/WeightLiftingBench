"""Hand-designed Fashion-MNIST classifier: writes network.json and weights.npz.

Usage: python3 build.py OUT_DIR

Every number below is typed from general knowledge of how clothing, shoes and
bags look in a centred, size-normalised 28x28 product photo (longest side 28,
item light on black). Nothing is fitted to or copied from any image.

Architecture (every path from the image to the scores passes through two hard
"comparator" stages, so the network is a rule system rather than a regression):
  image
  -> conv 1x1 (3 ch) + relu     raw image, and two shifted copies whose difference is a soft binary mask
  -> conv 13x13 (21 ch) + relu  raw, mask, thin-structure detectors, brightness counts left/right/above/below,
                                edge filters, dark / bright vertical line detectors (9-row averages),
                                3-wide mask sums (for a dilated mask), 3x3 dark-block detector
  -> conv 1x1 (14 ch) + relu    combines the counts into "enclosed dark" detectors, builds the dilated mask
  -> conv 13x13 (15 ch) + relu  carries the maps and adds a 9-row vertical average of the horizontal gap map
  -> dense + relu               profile layer: row widths, column heights, band widths, region sums
  -> dense + relu               carries the profiles and adds per-row "width close to the item width" terms
  -> dense                      measurements minus hand-chosen thresholds, each paired with a 0
  -> reshape (B,1,2) -> norm    the norm of a pair (p, 0) is p / sqrt(p^2 + 4e-5): a soft sign, i.e. the
                                comparator "measurement > threshold", a bit in [-1, 1]
  -> reshape -> dense           AND-rules over bits: one rule per slot, 10 classes x J slots
  -> reshape (R,1,2) -> norm    soft sign of each rule
  -> reshape -> avgpool         average of each class's slots (weight-free vote)
  -> maxpool stride 2 -> reshape (10)   keep the first element of each (+v, -v) pair
"""
import sys, json
import numpy as np
from pathlib import Path

K_SOFT = 1.0 / np.sqrt(4e-5)   # 158.11: p = v / (K_SOFT * w) makes the norm output v / sqrt(v^2 + w^2)


def build():
    layers, weights = [], {}

    def add(**kw):
        layers.append(kw)

    # ------------------------------------------------------------------ conv1: raw + mask pieces
    # channels: 0 = x, 1 = 8x - 0.4, 2 = 8x - 1.4 ; after relu, mask m = ch1 - ch2 = clip(8x - 0.4, 0, 1)
    # (0 for x <= 0.05, 1 for x >= 0.175: the background is black; even dark garments are above this)
    W = np.zeros((3, 1, 1, 1), np.float32); b = np.zeros(3, np.float32)
    W[0, 0, 0, 0] = 1.0
    W[1, 0, 0, 0] = 8.0; b[1] = -0.4
    W[2, 0, 0, 0] = 8.0; b[2] = -1.4
    add(name="pre", op="conv", inputs=["image"], out=3, kernel=[1, 1], stride=1, padding=0)
    weights["pre.W"], weights["pre.b"] = W, b
    add(name="pre_r", op="relu", inputs=["pre"])

    # ------------------------------------------------------------------ conv2: feature maps (13x13, pad 6)
    KS, C = 13, 6
    N2 = 21
    W = np.zeros((N2, 3, KS, KS), np.float32); b = np.zeros(N2, np.float32)

    def tap_x(o, dy, dx, c):            # raw image tap
        W[o, 0, C + dy, C + dx] += c

    def tap_m(o, dy, dx, c):            # mask tap (mask = ch1 - ch2)
        W[o, 1, C + dy, C + dx] += c
        W[o, 2, C + dy, C + dx] -= c

    tap_x(0, 0, 0, 1.0)                                  # 0: raw image
    tap_m(1, 0, 0, 1.0)                                  # 1: mask
    tap_m(2, 0, 0, 1.0); tap_m(2, 0, -2, -1.0); tap_m(2, 0, 2, -1.0)   # 2: thin vertical structure (<= 3 px wide)
    tap_m(3, 0, 0, 1.0); tap_m(3, -2, 0, -1.0); tap_m(3, 2, 0, -1.0)   # 3: thin horizontal structure
    for d in range(2, 7):
        tap_m(4, 0, -d, 1.0); tap_m(5, 0, -d, 1.0)       # 4,5: bright count 2-6 px to the left, minus 1 / minus 2
        tap_m(6, 0, d, 1.0); tap_m(7, 0, d, 1.0)         # 6,7: ... to the right
        tap_m(8, -d, 0, 1.0); tap_m(9, -d, 0, 1.0)       # 8,9: ... above
        tap_m(10, d, 0, 1.0); tap_m(11, d, 0, 1.0)       # 10,11: ... below
    for o in (4, 6, 8, 10): b[o] = -1.0
    for o in (5, 7, 9, 11): b[o] = -2.0
    tap_x(12, 0, 1, 1.0); tap_x(12, 0, -1, -1.0)         # 12: horizontal derivative +
    tap_x(13, 0, 1, -1.0); tap_x(13, 0, -1, 1.0)         # 13: horizontal derivative -
    tap_x(14, 1, 0, 1.0); tap_x(14, -1, 0, -1.0)         # 14: vertical derivative +
    tap_x(15, 1, 0, -1.0); tap_x(15, -1, 0, 1.0)         # 15: vertical derivative -
    for r in range(-4, 5):                               # 16: dark vertical line (averaged over 9 rows)
        tap_x(16, r, -2, 0.5 / 9); tap_x(16, r, 2, 0.5 / 9); tap_x(16, r, 0, -1.0 / 9)
        tap_x(17, r, -2, -0.5 / 9); tap_x(17, r, 2, -0.5 / 9); tap_x(17, r, 0, 1.0 / 9)   # 17: bright vertical line
    b[16] = -0.08; b[17] = -0.08
    for d in (-1, 0, 1): tap_m(18, 0, d, 1.0); tap_m(19, 0, d, 1.0)   # 18,19: 3-wide mask sum, and minus 1
    b[19] = -1.0
    for dy in (-1, 0, 1):                                # 20: dark 3x3 block = relu(1 - sum of the 3x3 mask)
        for dx in (-1, 0, 1): tap_m(20, dy, dx, -1.0)
    b[20] = 1.0
    add(name="maps", op="conv", inputs=["pre_r"], out=N2, kernel=[KS, KS], stride=1, padding=C)
    weights["maps.W"], weights["maps.b"] = W, b
    add(name="maps_r", op="relu", inputs=["maps"])

    # ------------------------------------------------------------------ conv3: 1x1 combination
    N3 = 14
    W = np.zeros((N3, N2, 1, 1), np.float32); b = np.zeros(N3, np.float32)
    for o, i in enumerate([0, 1, 2, 3]): W[o, i] = 1.0           # 0 x, 1 mask, 2 thin_h, 3 thin_v
    # 4: horizontal gap = clip(L) + clip(R) - 1 - 1.5 m  (dark pixel with >= 2 bright pixels 2-6 px away on both sides)
    W[4, 4] = 1.0; W[4, 5] = -1.0; W[4, 6] = 1.0; W[4, 7] = -1.0; W[4, 1] = -1.5; b[4] = -1.0
    # 5: big hole = clip(U) + clip(D) + darkblock - 2 : inside a dark region >= 3 px tall enclosed above and below
    W[5, 8] = 1.0; W[5, 9] = -1.0; W[5, 10] = 1.0; W[5, 11] = -1.0; W[5, 20] = 1.0; b[5] = -2.0
    for o, i in enumerate([12, 13, 14, 15, 16, 17], start=6): W[o, i] = 1.0   # 6-9 edges, 10 dark line, 11 bright line
    W[12, 18] = 1.0; W[12, 19] = -1.0                              # 12: dilated mask d = clip(3-wide sum, 0, 1)
    W[13, 1] = 1.0                                                # 13: spare copy of the mask (unused)
    add(name="maps2", op="conv", inputs=["maps_r"], out=N3, kernel=[1, 1], stride=1, padding=0)
    weights["maps2.W"], weights["maps2.b"] = W, b
    add(name="maps2_r", op="relu", inputs=["maps2"])

    # ------------------------------------------------------------------ conv4: carry + vertically consistent gap
    NMAP = 15
    W = np.zeros((NMAP, N3, KS, KS), np.float32); b = np.zeros(NMAP, np.float32)
    for i in range(N3): W[i, i, C, C] = 1.0                        # identity for channels 0-13
    for r in range(-4, 5): W[14, 4, C + r, C] = 1.0 / 9            # 14: 9-row average of the horizontal gap map
    b[14] = -0.5                                                  # fires where a column has a gap in most of 9 rows
    add(name="maps3", op="conv", inputs=["maps2_r"], out=NMAP, kernel=[KS, KS], stride=1, padding=C)
    weights["maps3.W"], weights["maps3.b"] = W, b
    add(name="maps3_r", op="relu", inputs=["maps3"])
    MASK, DIL, GAPH, HOLE, GAPV = 1, 12, 4, 5, 14

    # ------------------------------------------------------------------ dense 1: profiles (then relu)
    slot_rows, slot_bias, slot_index = [], [], {}

    def slot(name, coef, bias=0.0):
        slot_index[name] = len(slot_rows)
        slot_rows.append(np.asarray(coef, np.float32).reshape(-1)); slot_bias.append(float(bias))

    def Z():
        return np.zeros((NMAP, 28, 28), np.float32)

    # silhouette profiles use the dilated mask (1-2 px seams between sleeve and body are closed)
    # row widths rw[y]: slots 2*rw and 2*rw - 1 (their relu difference is min(2rw, 1): row occupied)
    for y in range(28):
        c = Z(); c[DIL, y, :] = 2.0
        slot(f"rw2_{y}", c); slot(f"rw2m_{y}", c, -1.0)
    for x in range(28):
        c = Z(); c[DIL, :, x] = 2.0
        slot(f"ch2_{x}", c); slot(f"ch2m_{x}", c, -1.0)
    # full-width rows for 28-wide items: ramp of rw[y] from 23 to 26 (difference of these two relus, /3)
    for y in range(28):
        c = Z(); c[DIL, y, :] = 1.0
        slot(f"rwf_a_{y}", c, -23.0); slot(f"rwf_b_{y}", c, -26.0)
    # row-to-row width jumps going down (bag body edge under a thin handle or flat top)
    for y in range(27):
        c = Z(); c[DIL, y + 1, :] = 1.0; c[DIL, y, :] = -1.0
        slot(f"jump_{y}", c, -12.0); slot(f"jump8_{y}", c, -8.0)
    # band mean widths (rows: 0-2 top, 2-5 shoulders, 8-11 upper arm, 13-17 waist, 21-26 hem)
    for name, (y0, y1) in {"Wtop3": (0, 3), "Wsh": (2, 6), "Wup": (8, 12), "Wmid": (13, 18), "Wbot": (21, 27)}.items():
        c = Z(); c[DIL, y0:y1, :] = 1.0 / (y1 - y0); slot(name, c)
    # column-height asymmetry: mean height of right columns 18-25 minus left columns 2-9 (shaft / heel side)
    c = Z(); c[DIL, :, 18:26] = 1.0 / 8; c[DIL, :, 2:10] = -1.0 / 8
    slot("asym_p", c); slot("asym_n", -c)
    # neck notch: outer shoulder columns minus center columns over the top 8 rows (rows of V / lapel opening)
    c = Z()
    for xx in (8, 9, 10, 17, 18, 19): c[MASK, 0:8, xx] = 1.0 / 6
    for xx in (13, 14): c[MASK, 0:8, xx] = -0.5
    slot("notch_p", c); slot("notch_n", -c)
    # masses and detectors on the plain mask
    c = Z(); c[MASK] = 1.0; slot("mass", c)
    c = Z(); c[MASK, 17:28, :] = 1.0; slot("mass_lo", c)
    c = Z(); c[MASK, 20:28, 12:16] = 1.0; slot("botcenter", c)
    c = Z(); c[2] = 1.0; c[3] = 1.0; slot("thin", c)
    c = Z(); c[HOLE] = 1.0; slot("bighole", c)
    c = Z(); c[HOLE, 1:13, 4:24] = 1.0; slot("loop", c)
    c = Z(); c[GAPV, 14:28, 8:20] = 1.0; slot("gap", c)
    c = Z(); c[GAPV, 21:28, 6:22] = 1.0; slot("gap_bot", c)
    # center vertical line structure (dark + bright line responses) in the center columns, chest and below waist
    c = Z(); c[10, 4:16, 10:18] = 1.0; c[11, 4:16, 10:18] = 1.0; slot("cl_up", c)
    c = Z(); c[10, 16:27, 10:18] = 1.0; c[11, 16:27, 10:18] = 1.0; slot("cl_lo", c)
    c = Z(); c[10, 4:27, 10:18] = 1.0; c[11, 4:27, 10:18] = -1.0; slot("cl_dark_p", c); slot("cl_dark_n", -c)
    # texture: edge energy inside the body region
    c = Z(); c[6:10, 8:25, 8:20] = 1.0; slot("tex", c)

    W1 = np.stack(slot_rows); b1 = np.array(slot_bias, np.float32)
    N1 = len(slot_rows)
    add(name="prof", op="dense", inputs=["maps3_r"], out=N1)
    weights["prof.W"], weights["prof.b"] = W1, b1
    add(name="prof_r", op="relu", inputs=["prof"])

    # ------------------------------------------------------------------ dense 1b: carry + per-row "width >= 0.85 W" terms
    rows2, bias2 = [np.eye(N1, dtype=np.float32)[i] for i in range(N1)], [0.0] * N1
    slot2_index = dict(slot_index)
    wvec = np.zeros(N1, np.float32)
    for x in range(28): wvec[slot_index[f"ch2_{x}"]] += 1.0; wvec[slot_index[f"ch2m_{x}"]] -= 1.0
    for y in range(28):
        v = -0.85 * wvec; v[slot_index[f"rw2_{y}"]] += 0.5          # rw[y] - 0.85 W
        slot2_index[f"rel_a_{y}"] = len(rows2); rows2.append(v.copy()); bias2.append(1.0)   # relu(rw - 0.85W + 1)
        slot2_index[f"rel_b_{y}"] = len(rows2); rows2.append(v.copy()); bias2.append(0.0)   # relu(rw - 0.85W)
    W1b = np.stack(rows2); b1b = np.array(bias2, np.float32)
    N1b = len(rows2)
    add(name="prof2", op="dense", inputs=["prof_r"], out=N1b)
    weights["prof2.W"], weights["prof2.b"] = W1b, b1b
    add(name="prof2_r", op="relu", inputs=["prof2"])

    # ------------------------------------------------------------------ dense 2: measurements vs thresholds
    def lin(terms, const=0.0):
        v = np.zeros(N1b, np.float32)
        for k, cf in terms.items(): v[slot2_index[k]] += cf
        return v, const

    def combo(parts):
        v = np.zeros(N1b, np.float32); c = 0.0
        for s, (vv, cc) in parts: v += s * vv; c += s * cc
        return v, c

    def K(c):
        return lin({}, c)

    H = lin({**{f"rw2_{y}": 1.0 for y in range(28)}, **{f"rw2m_{y}": -1.0 for y in range(28)}})
    Wd = lin({**{f"ch2_{x}": 1.0 for x in range(28)}, **{f"ch2m_{x}": -1.0 for x in range(28)}})
    full = lin({**{f"rwf_a_{y}": 1 / 3 for y in range(28)}, **{f"rwf_b_{y}": -1 / 3 for y in range(28)}})
    fullrel = lin({**{f"rel_a_{y}": 1.0 for y in range(28)}, **{f"rel_b_{y}": -1.0 for y in range(28)}})
    jump = lin({f"jump_{y}": 1.0 for y in range(27)})
    jump8 = lin({f"jump8_{y}": 1.0 for y in range(27)})
    mass, mass_lo, botcenter = lin({"mass": 1.0}), lin({"mass_lo": 1.0}), lin({"botcenter": 1.0})
    thin, bighole, loop = lin({"thin": 1.0}), lin({"bighole": 1.0}), lin({"loop": 1.0})
    gap, gap_bot, tex = lin({"gap": 1.0}), lin({"gap_bot": 1.0}), lin({"tex": 1.0})
    Wtop3, Wsh, Wup, Wmid, Wbot = (lin({n: 1.0}) for n in ("Wtop3", "Wsh", "Wup", "Wmid", "Wbot"))
    asym = lin({"asym_p": 1.0, "asym_n": 1.0})          # |right column height - left column height|
    notch = lin({"notch_p": 1.0, "notch_n": -1.0})      # signed notch depth (rows)
    cl_up, cl_lo = lin({"cl_up": 1.0}), lin({"cl_lo": 1.0})
    cl_dark = lin({"cl_dark_p": 1.0, "cl_dark_n": -1.0})

    bits = {}

    def bit(name, expr, width):
        bits[name] = (expr, width)

    # group: the longest side is always 28, so garments are 28 tall; shoes and most bags are wider than tall
    bit("tall", combo([(1, H), (1, K(-23.0))]), 1.5)
    # wide-group cues
    bit("low", combo([(-1, H), (1, K(9.5))]), 1.0)                 # very flat: slides / flip-flops
    bit("sneakH", combo([(-1, H), (1, K(16.5))]), 1.5)             # low shoe height; ~sneakH = boot height
    bit("asym", combo([(1, asym), (-0.6, H)]), 2.5)                # one side much taller than the other (shaft, heel)
    bit("symm", combo([(-1, asym), (0.2, H)]), 2.0)                # both sides about equally tall (bag)
    # fill ratio: box area = 28 * (H + W - 28) because the longest side is 28; sparse if fill < 0.38 / 0.30
    bit("sparse", combo([(13.44, H), (13.44, Wd), (1, K(-13.44 * 28)), (-1, mass)]), 15.0)     # fill < 0.48
    bit("sparse2", combo([(10.08, H), (10.08, Wd), (1, K(-10.08 * 28)), (-1, mass)]), 15.0)    # fill < 0.36
    bit("sparseT", combo([(8.4, H), (8.4, Wd), (1, K(-8.4 * 28)), (-1, mass)]), 15.0)
    bit("thin", combo([(1, thin), (1, K(-20.0))]), 6.0)            # many thin strap-like structures
    bit("bighole", combo([(1, bighole), (1, K(-6.0))]), 3.0)       # enclosed dark openings >= 3 px tall (straps)
    bit("loop", combo([(1, loop), (1, K(-3.0))]), 1.5)             # enclosed opening in the upper part (handle loop)
    bit("full", combo([(1, full), (-0.55, H)]), 2.5)               # most rows span the full 28 width
    bit("fullrel", combo([(1, fullrel), (-0.42, H)]), 2.0)         # many rows nearly as wide as the item
    bit("fullT", combo([(1, full), (1, K(-26.0))]), 1.0)           # nearly every row full: square bag
    bit("jump", combo([(1, jump), (1, K(-4.0))]), 2.0)             # abrupt width jump going down (handle -> body)
    bit("jump8", combo([(1, jump8), (1, K(-2.0))]), 1.5)           # moderate width jump going down
    # tall-group cues (widths are from the dilated mask: about 2 px more than the plain silhouette)
    bit("narrow", combo([(-1, Wd), (1, K(18.5))]), 1.5)
    bit("vnarrow", combo([(-1, Wd), (1, K(14.5))]), 1.5)
    bit("wide", combo([(1, Wd), (1, K(-24.5))]), 2.0)
    bit("gap", combo([(1, gap), (1, K(-2.0))]), 1.0)               # vertically continuous dark gap, lower center
    bit("gap_bot", combo([(1, gap_bot), (1, K(-1.0))]), 0.7)       # the gap reaches the hem rows
    bit("botdark", combo([(-1, botcenter), (1, K(20.0))]), 4.0)    # bottom-center 8x4 block has >= 12 dark px
    bit("taper", combo([(1, Wtop3), (-1, Wbot), (1, K(-0.5))]), 1.5)       # top wider than the hem (waist > ankles)
    bit("flattop", combo([(1, Wtop3), (-0.75, Wd)]), 2.0)          # top rows nearly as wide as the item (waistband)
    bit("thintop", combo([(-1, Wtop3), (0.3, Wd)]), 1.5)           # top rows thin (handle, straps)
    bit("shwide", combo([(1, Wsh), (-0.75, Wd)]), 1.5)             # shoulder rows nearly full width (sleeves present)
    bit("flare", combo([(1, Wbot), (-1, Wmid), (1, K(-3.5))]), 1.5)        # hem clearly wider than waist
    bit("hugeflare", combo([(1, Wbot), (-1, Wmid), (1, K(-9.0))]), 2.0)    # hem much wider than waist (boot foot)
    bit("drop", combo([(1, Wup), (-1, Wmid), (1, K(-3.0))]), 1.5)          # short sleeves end between rows 11-13
    bit("shdrop", combo([(1, Wsh), (-1, Wup), (1, K(-3.0))]), 1.5)         # cap sleeves end before row 8
    bit("midwide", combo([(1, Wmid), (-0.78, Wd)]), 1.5)                   # waist rows nearly full width
    bit("botwide", combo([(1, Wbot), (-0.78, Wd)]), 1.5)                   # hem rows nearly full width
    bit("solidbot", combo([(1, mass_lo), (-0.8 * 11, Wd)]), 10.0)          # bottom 11 rows nearly filled across W
    bit("notch", combo([(1, notch), (1, K(-4.0))]), 1.0)                   # deep V / lapel opening
    bit("notch2", combo([(1, notch), (1, K(-2.5))]), 0.7)                  # collar-sized notch
    bit("cl_up", combo([(1, cl_up), (1, K(-2.0))]), 1.0)                   # center line structure, chest
    bit("cl_lo", combo([(1, cl_lo), (1, K(-1.5))]), 0.7)                   # center line structure, below waist
    bit("dark", combo([(1, cl_dark), (1, K(-1.0))]), 0.7)                  # center line predominantly dark
    bit("tex", combo([(1, tex), (1, K(-60.0))]), 20.0)                     # patterned body

    bit_names = list(bits)
    NB = len(bit_names)
    W2 = np.zeros((2 * NB, N1b), np.float32); b2 = np.zeros(2 * NB, np.float32)
    for i, name in enumerate(bit_names):
        (v, c), w = bits[name]
        s = 1.0 / (K_SOFT * w)
        W2[2 * i] = v * s; b2[2 * i] = c * s          # odd rows stay 0: the pair partner
    add(name="meas", op="dense", inputs=["prof2_r"], out=2 * NB)
    weights["meas.W"], weights["meas.b"] = W2, b2
    add(name="meas_p", op="reshape", inputs=["meas"], shape=[NB, 1, 2])
    add(name="bits", op="norm", inputs=["meas_p"])
    add(name="bits_v", op="reshape", inputs=["bits"], shape=[2 * NB])

    # ------------------------------------------------------------------ dense 3: rules (AND of bits)
    # rule = list of bit names, "~name" for negation. AND: sum(bits) - 0.92 (n - 1) > 0 only if all are about +1.
    rules = {
        0: [  # T-shirt / top: as wide as tall, shoulders + short sleeves wide, body narrower, no flare, no placket
            ["tall", "wide", "~midwide", "~flare", "shwide", "~shdrop", "~gap", "~sparse", "~asym", "~cl_lo"],
            ["tall", "wide", "drop", "~flare", "~sparse", "~cl_lo", "~notch2"],
            ["tall", "~narrow", "~midwide", "~flare", "~shdrop", "~sparse", "~jump", "~gap", "~asym", "~cl_lo"],
            ["tall", "wide", "shdrop", "~midwide", "~flare", "~gap", "~asym", "~cl_lo", "~notch2"],
            ["tall", "~narrow", "shdrop", "~midwide", "~flare", "~gap", "~asym", "~cl_lo", "~notch2", "shwide"],
        ],
        1: [  # trouser: tall and narrow, dark gap between the legs reaching the hem, flat waistband, tapering
            ["tall", "narrow", "gap", "~notch", "botdark"],
            ["tall", "vnarrow", "flattop", "botdark", "~thin"],
            ["tall", "narrow", "flattop", "~flare", "~sparseT", "~thin", "taper", "~thintop"],
            ["tall", "gap", "~wide", "~flare", "~thin", "flattop", "~notch", "botdark"],
            ["tall", "gap", "wide", "~midwide", "~thin", "~notch", "~sparse", "~asym", "botdark"],
            ["tall", "narrow", "gap_bot", "~notch", "botdark", "~flare"],
        ],
        2: [  # pullover: wide (as wide as long), rectangular with sleeves to the hem, plain round neck, no center line
            ["tall", "wide", "midwide", "botwide", "~notch2", "~cl_lo", "~fullT", "~jump", "~loop"],
            ["tall", "wide", "midwide", "~notch2", "~cl_lo", "~drop", "~fullT", "~jump", "~loop"],
            ["tall", "wide", "midwide", "botwide", "~notch2", "~dark", "~fullT", "~jump", "~loop"],
        ],
        3: [  # dress: long and narrow, no leg gap, flaring hem, thin straps, or a waist; not rectangular with shoulders
            ["tall", "~wide", "~gap", "flare", "~sparseT", "~asym", "~hugeflare"],
            ["tall", "~wide", "~gap", "thintop", "~sparseT", "~jump8", "~loop"],
            ["tall", "narrow", "shdrop", "~midwide", "~drop", "~sparseT", "~gap", "~asym"],
            ["tall", "~wide", "~gap", "~sparseT", "~jump", "~midwide", "~flattop", "~asym", "~taper"],
            ["*", "tall", "~wide", "~gap", "~sparseT", "~flattop", "~asym"],
            ["tall", "wide", "flare", "~midwide", "~drop", "~gap", "~asym", "~hugeflare"],
        ],
        4: [  # coat: long garment, rectangular with shoulders and sleeves to the hem, lapels / open dark front
            ["tall", "~vnarrow", "midwide", "notch", "~asym", "~flattop", "shwide"],
            ["tall", "~vnarrow", "midwide", "botwide", "cl_lo", "dark", "~flattop", "shwide"],
            ["tall", "~vnarrow", "midwide", "cl_up", "dark", "~drop", "~flattop", "shwide"],
            ["tall", "~vnarrow", "~wide", "midwide", "shwide", "~flare", "~jump", "~gap", "~asym", "~flattop"],
            ["tall", "~vnarrow", "~wide", "midwide", "notch2", "~flare", "~gap", "~asym", "~flattop", "~loop"],
        ],
        5: [  # sandal: sparse straps, very flat slides, thin structures, enclosed openings; heeled ones are tall
            ["~tall", "sparse"],
            ["~tall", "low"],
            ["~tall", "thin", "~fullrel", "~loop"],
            ["~tall", "bighole", "~fullrel", "~low"],
            ["tall", "sparseT", "~solidbot"],
            ["tall", "thin", "sparse", "~solidbot"],
        ],
        6: [  # shirt: button placket down the center continuing below the waist, collar without lapels, patterns
            ["tall", "~narrow", "cl_lo", "~notch", "~dark", "~gap", "~asym", "shwide"],
            ["tall", "wide", "drop", "cl_lo", "~notch"],
            ["tall", "~narrow", "midwide", "cl_lo", "~notch", "~dark", "shwide"],
            ["tall", "wide", "drop", "tex", "~notch"],
            ["tall", "wide", "midwide", "notch2", "~notch", "~dark", "~jump", "~fullT", "~loop"],
            ["tall", "wide", "drop", "notch2", "~notch", "~flare"],
        ],
        7: [  # sneaker: low solid shoe, not rectangular, no tall shaft, no big openings
            ["~tall", "sneakH", "~low", "~sparse2", "~fullrel", "~bighole"],
            ["~tall", "sneakH", "~low", "~sparse2", "~asym", "~bighole"],
            ["~tall", "sneakH", "~low", "~sparse2", "~fullrel", "~asym"],
            ["*", "~tall", "sneakH", "~low"],
        ],
        8: [  # bag: rectangular body (many rows as wide as the item), symmetric, solid; handle loop; tall bags
            ["~tall", "fullrel", "symm", "~sparse2", "~low"],
            ["~tall", "jump", "fullrel", "~sparse2", "~low"],
            ["~tall", "jump", "~sparse2", "~low", "~asym", "~sneakH"],
            ["~tall", "loop", "fullrel", "~sparse2", "~low"],
            ["tall", "thintop", "jump", "solidbot", "~flare", "~gap", "~sparseT"],
            ["tall", "jump", "midwide", "botwide", "solidbot", "~flare", "~gap", "~notch", "~narrow"],
            ["tall", "thintop", "jump8", "fullrel", "~flare", "~gap", "~sparseT", "~narrow"],
            ["tall", "loop", "solidbot", "~flare", "~gap", "~sparseT", "~narrow"],
            ["tall", "fullT"],
        ],
        9: [  # ankle boot: taller than a sneaker, solid, shaft makes one side much taller than the toe side
            ["~tall", "~sneakH", "asym", "~sparse"],
            ["~tall", "~sneakH", "~fullrel", "~sparse", "~symm"],
            ["~tall", "asym", "~sparse", "~low", "~fullrel"],
            ["tall", "asym", "~sparseT"],
            ["tall", "hugeflare", "botwide", "~sparseT"],
        ],
    }
    Kmax = max(len(r) for r in rules.values())
    J = 2 * Kmax
    R = 10 * J
    W3 = np.zeros((2 * R, 2 * NB), np.float32); b3 = np.zeros(2 * R, np.float32)
    w_rule = 0.35                     # softness of a rule vote (in units of bits): a rule missing one bit gives ~0
    s3 = 1.0 / (K_SOFT * w_rule)
    for k in range(10):
        rl = rules[k]
        for j in range(J):
            r = k * J + j
            if j < len(rl):                                  # a rule slot
                weak = rl[j][0] == "*"                       # "*": weak default rule, votes ~0.35 when satisfied
                terms = [t for t in rl[j] if t != "*"]
                n = len(terms)
                for term in terms:
                    neg = term.startswith("~"); name = term.lstrip("~")
                    col = 2 * bit_names.index(name)          # even entry = +bit, odd = -bit
                    W3[2 * r, col + (1 if neg else 0)] += s3
                # bits of clear cases sit near 0.92, so the AND threshold expects 0.92 per bit; a weak rule
                # is shifted so that a satisfied one gives pre = -0.3 (vote ~0.35 instead of ~1.9)
                b3[2 * r] = (-0.92 * (n - 1) - (1.3 if weak else 0.0)) * s3
            elif j < 2 * len(rl):                            # a constant +1 slot per rule: votes in [0, 2]
                b3[2 * r] = 100.0 * s3
            # remaining slots: pair (0, 0) -> 0
    add(name="rules", op="dense", inputs=["bits_v"], out=2 * R)
    weights["rules.W"], weights["rules.b"] = W3, b3
    add(name="rules_p", op="reshape", inputs=["rules"], shape=[R, 1, 2])
    add(name="votes", op="norm", inputs=["rules_p"])
    add(name="votes_v", op="reshape", inputs=["votes"], shape=[1, R, 2])
    add(name="pool", op="avgpool", inputs=["votes_v"], size=[J, 1], stride=[J, 1])      # (1, 10, 2)
    add(name="first", op="maxpool", inputs=["pool"], size=[1, 1], stride=[1, 2])        # (1, 10, 1)
    add(name="scores", op="reshape", inputs=["first"], shape=[10])

    info = {"bit_names": bit_names, "slot_index": slot2_index, "J": J}
    return {"layers": layers}, weights, info


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    net, weights, _ = build()
    (out / "network.json").write_text(json.dumps(net, indent=1))
    np.savez(out / "weights.npz", **{k: v.astype(np.float32) for k, v in weights.items()})
    total = sum(int(np.prod(v.shape)) for v in weights.values())
    print(f"wrote {out/'network.json'} and {out/'weights.npz'} ({total:,} weights)")


if __name__ == "__main__":
    main()
