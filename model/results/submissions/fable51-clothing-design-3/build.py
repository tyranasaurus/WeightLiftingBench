#!/usr/bin/env python3
"""Hand-designed Fashion-MNIST classifier: every weight below is set from general knowledge of how
clothing, shoes and bags are shaped, not from the training images.

Pipeline (all weights come from formulas / hand-chosen constants in this file):
  1. masks     soft silhouette of the item (grey ramp 0.10 -> 0.20) and a "deep" mask (0.03 -> 0.07) that
               only leaves out true background, used by the hole detectors.
  2. detectors hand-written convolutions: interior pixels (>= 8 of 9 lit), 1-px thin lines, horizontal /
               vertical holes at two scales (background pixel with lit pixels on both sides), a horizontal
               top-edge detector, centre valley / ridge detectors (open coat front or seam / bright shirt
               placket), a lapel detector (dark V at the neck) and an interior-gated texture map.
  3. pools     band widths (max over 4-row and 2-row bands, summed over columns), band heights (same for
               4-column bands), row / column occupancy (max over a full row / column) for height and width.
  4. features  one dense stage producing silhouette measurements (height, width, band widths/heights,
               top-third left/right masses, flare, sleeve drop, leg gap, hole counts, edge lengths ...).
  5. ramps     soft threshold indicators ind(x; t, w) = clip((x - t) / w + 0.5, 0, 1) built from relu pairs.
  6. rules     AND-gates relu(sum of indicators - (n - 1) [+ small evidence terms]) encoding hand-written
               rules per class.
  7. scores    weighted sum of rule activations plus a few linear evidence terms.

Geometry used for the constants: images are 28 x 28, the item is scaled so its longest side is 28 and
centred.  Tops / trousers / dresses therefore span the full height; shoes and most bags the full width.
Shoes in catalogue photos point left (toe left, heel right).

Usage: python3 build.py OUT_DIR   -> writes OUT_DIR/network.json and OUT_DIR/weights.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

layers = []
weights = {}


def L(name, op, inputs, **kw):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(kw)
    layers.append(d)
    return name


def conv(name, inp, W, b, padding=0, stride=1):
    W = np.asarray(W, np.float32)
    b = np.asarray(b, np.float32)
    weights[name + ".W"] = W
    weights[name + ".b"] = b
    return L(name, "conv", [inp], out=int(W.shape[0]), kernel=[int(W.shape[2]), int(W.shape[3])],
             stride=stride, padding=padding)


def dense(name, inp, W, b):
    W = np.asarray(W, np.float32)
    b = np.asarray(b, np.float32)
    weights[name + ".W"] = W
    weights[name + ".b"] = b
    return L(name, "dense", [inp], out=int(W.shape[0]))


def relu(name, inp):
    return L(name, "relu", [inp])


def maxpool(name, inp, size, stride):
    return L(name, "maxpool", [inp], size=list(size), stride=list(stride))


def add(name, inputs):
    return L(name, "add", list(inputs))


def ramp_map(name, inp, lo, hi):
    """clip((x - lo) / (hi - lo), 0, 1) applied pixel-wise to a 1-channel map, via a relu pair."""
    a = 1.0 / (hi - lo)
    conv(name + "_p", inp, np.array([[[[a]]], [[[a]]]]), [-a * lo, -a * lo - 1.0])
    relu(name + "_r", name + "_p")
    return conv(name, name + "_r", np.array([[[[1.0]], [[-1.0]]]]), [0.0])


# ----------------------------------------------------------------------------------------------
# 1. masks.  Background is exactly 0, sharpening halos stay below ~0.1, even black garments are >= ~0.2.
# ----------------------------------------------------------------------------------------------
mask = ramp_map("mask", "image", 0.10, 0.20)        # silhouette, (1, 28, 28) in [0, 1]
deep = ramp_map("deep", "image", 0.03, 0.07)        # everything that is not true background

# ----------------------------------------------------------------------------------------------
# 2. detectors
# ----------------------------------------------------------------------------------------------
# interior: 3x3 neighbourhood sum of the mask; "interior" when at least 8 of the 9 are lit.
ones33 = np.ones((1, 1, 3, 3), np.float32)
conv("nb", mask, np.concatenate([ones33, ones33], 0), [-7.0, -8.0], padding=1)
relu("nbr", "nb")
interior = conv("interior", "nbr", np.array([[[[1.0]], [[-1.0]]]]), [0.0])    # clip(sum9 - 7, 0, 1)

# thin lines: lit pixel whose two neighbours across the line are dark (1-px straps).  Two orientations.
Wth = np.zeros((2, 1, 3, 3), np.float32)
Wth[0, 0, 1, 1] = 1.0
Wth[0, 0, 0, 1] = -1.0          # horizontal line: above and below dark
Wth[0, 0, 2, 1] = -1.0
Wth[1, 0, 1, 1] = 1.0
Wth[1, 0, 1, 0] = -1.0          # vertical line: left and right dark
Wth[1, 0, 1, 2] = -1.0
conv("thin", mask, Wth, [-0.3, -0.3], padding=1)
thinr = relu("thinr", "thin")

# holes on the deep mask, small scale: centre is background while pixels 2-3 away on both sides are lit.
Wh = np.zeros((2, 1, 7, 7), np.float32)
for off in (0, 1, 5, 6):
    Wh[0, 0, 3, off] = 0.5          # horizontal neighbours at distance 2 and 3
    Wh[1, 0, off, 3] = 0.5          # vertical neighbours
Wh[0, 0, 3, 3] = -1.0
Wh[1, 0, 3, 3] = -1.0
conv("hole", deep, Wh, [-1.2, -1.2], padding=3)
holer = relu("holer", "hole")       # about 0.8 inside a 1-3 px wide gap, 0 elsewhere

# holes, larger scale: lit pixels 5-6 away on both sides (handle loops, strap openings).
Wh2 = np.zeros((2, 1, 13, 13), np.float32)
for off in (0, 1, 11, 12):
    Wh2[0, 0, 6, off] = 0.5
    Wh2[1, 0, off, 6] = 0.5
Wh2[0, 0, 6, 6] = -1.0
Wh2[1, 0, 6, 6] = -1.0
conv("hole2", deep, Wh2, [-1.2, -1.2], padding=6)
holer2 = relu("holer2", "hole2")

# horizontal top edge: the two rows below are lit, the two rows above are dark  (max 0.6 per pixel)
We = np.zeros((1, 1, 5, 5), np.float32)
We[0, 0, 2, 2] = 0.5
We[0, 0, 3, 2] = 0.5
We[0, 0, 1, 2] = -0.5
We[0, 0, 0, 2] = -0.5
conv("edge", mask, We, [-0.4], padding=2)
edger = relu("edger", "edge")

# centre valley / ridge on the grey image: pixel darker / brighter than the mean of the pixels 3-4
# columns to each side (open coat front or seam  /  bright button placket of a shirt).
Wcv = np.zeros((2, 1, 9, 9), np.float32)
for off in (0, 1, 7, 8):
    Wcv[0, 0, 4, off] = 0.25
    Wcv[1, 0, 4, off] = -0.25
Wcv[0, 0, 4, 4] = -1.0
Wcv[1, 0, 4, 4] = 1.0
conv("cv", "image", Wcv, [-0.05, -0.08], padding=4)
cvr = relu("cvr", "cv")

# lapel / deep V: pixel much darker than the mean of pixels 6-7 columns to each side
Wcv2 = np.zeros((1, 1, 15, 15), np.float32)
for off in (0, 1, 13, 14):
    Wcv2[0, 0, 7, off] = 0.25
Wcv2[0, 0, 7, 7] = -1.0
conv("cv2", "image", Wcv2, [-0.10], padding=7)
cv2r = relu("cv2r", "cv2")

# texture: |horizontal gradient| of the grey image, only counted at interior pixels.
Wdx = np.zeros((2, 1, 3, 3), np.float32)
Wdx[0, 0, 1, 2], Wdx[0, 0, 1, 1] = 1.0, -1.0
Wdx[1, 0, 1, 2], Wdx[1, 0, 1, 1] = -1.0, 1.0
conv("dx", "image", Wdx, [0.0, 0.0], padding=1)
relu("dxr", "dx")
conv("absdx", "dxr", np.array([[[[1.0]], [[1.0]]]]), [-2.1])                  # |dx| - 2.1
conv("ig", interior, np.array([[[[2.0]]]]), [0.0])                            # 2 * interior
add("texm", ["absdx", "ig"])                                                  # |dx| - 0.1 - 2 (1 - interior)
texr = relu("texr", "texm")

# ----------------------------------------------------------------------------------------------
# 3. pools
# ----------------------------------------------------------------------------------------------
bwp = maxpool("bwp", mask, [4, 1], [4, 1])          # (1, 7, 28): row band b, column c lit?
bwp2 = maxpool("bwp2", mask, [2, 1], [2, 1])        # (1, 14, 28): 2-row bands
bhp = maxpool("bhp", mask, [1, 4], [1, 4])          # (1, 28, 7): row r, column band j lit?
rowmax = maxpool("rowmax", mask, [1, 28], [1, 28])  # (1, 28, 1): row occupied?
colmax = maxpool("colmax", mask, [28, 1], [28, 1])  # (1, 1, 28): column occupied?
# vertical dilation by 7 rows (sliding max) bridges gaps between straps and sole: summing the dilated
# occupancy gives (vertical extent + 6), saturating at 22 for items taller than 16 rows.
bhp7 = maxpool("bhp7", bhp, [7, 1], [1, 1])          # (1, 22, 7)
rowmax7 = maxpool("rowmax7", rowmax, [7, 1], [1, 1])  # (1, 22, 1)

# ----------------------------------------------------------------------------------------------
# 4. feature stage: several dense layers (one per map) whose outputs are added, then relu.
# ----------------------------------------------------------------------------------------------
F = ["T", "E", "H", "Wd", "upper", "ups_p", "ups_n", "rise_p", "rise_n", "sparse_p", "empty_p",
     "flare_p", "flare2_p", "tw_p", "tw_n", "sdrop_p", "sleeve_p", "bagtop_p", "cuff_p", "waist_p",
     "cv_center", "cv_low", "cv_lapel", "rv_center", "tex_low",
     "gap_low", "hole_total", "bag_loop", "BWf0", "thin_total", "solid_p", "topfill_p"]
F += ["BW%d" % b for b in range(7)]
F += ["BH%d" % j for j in range(7)]
F += ["jump%d" % b for b in range(7)]                 # 2-row band width jumps (rows 0-15)
F += ["tv%d_p" % b for b in range(6)] + ["tv%d_n" % b for b in range(6)]   # width profile variation
F += ["edge%d" % r for r in range(28)]                # long horizontal top edge per row
fi = {n: i for i, n in enumerate(F)}
NF = len(F)


def grid(shape):
    """index helpers for a flattened (C, H, W) map"""
    C, Hh, Ww = shape
    return np.arange(C * Hh * Ww).reshape(C, Hh, Ww)


g = grid((1, 28, 28))
g2 = grid((2, 28, 28))

# --- from mask (1, 28, 28)
Wm = np.zeros((NF, 784), np.float32)
Wm[fi["T"], :] = 1.0
Wm[fi["sparse_p"], :] = 1.0                       # T - 2.5 E: large when almost nothing is interior
Wm[fi["empty_p"], :] = -1.6                       # 28 (H7 - 6) - 1.6 T: empty area of the bounding box
Wm[fi["solid_p"], :] = 1.0                        # T - 2.2 sum_j (BH7_j - 6): column bands filled over their extent
Wm[fi["topfill_p"], g[0, 0:4, :].ravel()] = 1.0   # mass of rows 0-3 - 2.8 * their width: solid top (collar),
                                                  #   negative for a thin handle arc
top = g[0, 0:10, :].ravel()                       # top third of the frame
UL = g[0, 0:10, 0:14].ravel()
UR = g[0, 0:10, 14:28].ravel()
Wm[fi["upper"], top] = 1.0
Wm[fi["ups_p"], UR] = 1.0
Wm[fi["ups_p"], UL] = -1.0
Wm[fi["ups_n"], UR] = -1.0
Wm[fi["ups_n"], UL] = 1.0
dense("f_mask", mask, Wm, np.zeros(NF))

# --- from interior (1, 28, 28)
Wi = np.zeros((NF, 784), np.float32)
Wi[fi["E"], :] = 1.0
Wi[fi["sparse_p"], :] = -2.5
dense("f_int", interior, Wi, np.zeros(NF))

# --- thin-line pixels (2 orientations)
Wthd = np.zeros((NF, 2 * 784), np.float32)
Wthd[fi["thin_total"], :] = 1.0
dense("f_thin", thinr, Wthd, np.zeros(NF))

# --- from 4-row band widths (1, 7, 28)
Wb = np.zeros((NF, 7 * 28), np.float32)
bb = np.zeros(NF, np.float32)
gb = grid((1, 7, 28))
for b in range(7):
    Wb[fi["BW%d" % b], gb[0, b, :]] = 1.0
Wb[fi["topfill_p"], gb[0, 0, :]] = -2.8
Wb[fi["flare_p"], gb[0, 6, :]] = 1.0              # hem width (rows 24-27) - width at rows 12-15
Wb[fi["flare_p"], gb[0, 3, :]] = -1.0
Wb[fi["flare2_p"], gb[0, 6, :]] = 1.0             # hem wider than the shoulder / sleeve span (rows 4-11)
Wb[fi["flare2_p"], gb[0, 1, :]] = -0.5            #   never true for a T-shirt, true for flared dresses
Wb[fi["flare2_p"], gb[0, 2, :]] = -0.5
Wb[fi["tw_p"], gb[0, 6, :]] = -0.6                # waist test: tw = BWf0 - 0.4 Wd - 0.6 BW6
Wb[fi["tw_n"], gb[0, 6, :]] = 0.6                 #   (top two rows as wide as the item, hem not wider)
Wb[fi["sdrop_p"], gb[0, 1, :]] = 1.0              # (rows 4-11 width) - (rows 16-23 width): sleeve taper
Wb[fi["sdrop_p"], gb[0, 2, :]] = 1.0
Wb[fi["sdrop_p"], gb[0, 4, :]] = -1.0
Wb[fi["sdrop_p"], gb[0, 5, :]] = -1.0
Wb[fi["sleeve_p"], gb[0, 1, :]] = 0.5             # sleeve bands wider than the band at rows 12-15
Wb[fi["sleeve_p"], gb[0, 2, :]] = 0.5
Wb[fi["sleeve_p"], gb[0, 3, :]] = -1.0
Wb[fi["bagtop_p"], gb[0, 3, :]] = 0.5             # body (rows 12-19) much wider than the top band (handle)
Wb[fi["bagtop_p"], gb[0, 4, :]] = 0.5
Wb[fi["bagtop_p"], gb[0, 0, :]] = -1.0
Wb[fi["cuff_p"], gb[0, 5, :]] = 1.0               # sleeves end above the hem: rows 20-23 wider than 24-27
Wb[fi["cuff_p"], gb[0, 6, :]] = -1.0
Wb[fi["waist_p"], gb[0, 1, :]] = 0.5              # fitted waist: rows 12-15 narrower than bust and hem
Wb[fi["waist_p"], gb[0, 6, :]] = 0.5
Wb[fi["waist_p"], gb[0, 3, :]] = -1.0
for b in range(6):                                # total variation of the width profile
    Wb[fi["tv%d_p" % b], gb[0, b, :]] = 1.0
    Wb[fi["tv%d_p" % b], gb[0, b + 1, :]] = -1.0
    Wb[fi["tv%d_n" % b], gb[0, b, :]] = -1.0
    Wb[fi["tv%d_n" % b], gb[0, b + 1, :]] = 1.0
dense("f_bw", bwp, Wb, bb)

# --- from 2-row band widths (1, 14, 28)
Wb2 = np.zeros((NF, 14 * 28), np.float32)
bb2 = np.zeros(NF, np.float32)
gb2 = grid((1, 14, 28))
Wb2[fi["BWf0"], gb2[0, 0, :]] = 1.0               # width of rows 0-1
Wb2[fi["tw_p"], gb2[0, 0, :]] = 1.0               # waistband: rows 0-1 wide (straps are thin there)
Wb2[fi["tw_n"], gb2[0, 0, :]] = -1.0
for b in range(7):                                # abrupt widening (> 14 columns) between adjacent 2-row bands
    Wb2[fi["jump%d" % b], gb2[0, b + 1, :]] = 1.0
    Wb2[fi["jump%d" % b], gb2[0, b, :]] = -1.0
    bb2[fi["jump%d" % b]] = -14.0
dense("f_bw2", bwp2, Wb2, bb2)

# --- from band heights (1, 28, 7)
Wbh = np.zeros((NF, 28 * 7), np.float32)
gh = grid((1, 28, 7))
for j in range(7):
    Wbh[fi["BH%d" % j], gh[0, :, j]] = 1.0
for j in (5, 6):                                  # heel side taller than toe side (shoes point left)
    Wbh[fi["rise_p"], gh[0, :, j]] = 1.0
    Wbh[fi["rise_n"], gh[0, :, j]] = -1.0
for j in (0, 1):
    Wbh[fi["rise_p"], gh[0, :, j]] = -1.0
    Wbh[fi["rise_n"], gh[0, :, j]] = 1.0
dense("f_bh", bhp, Wbh, np.zeros(NF))

# --- from row / column occupancy
Wr = np.zeros((NF, 28), np.float32)
Wr[fi["H"], :] = 1.0
dense("f_row", rowmax, Wr, np.zeros(NF))
Wr7 = np.zeros((NF, 22), np.float32)
br7 = np.zeros(NF, np.float32)
Wr7[fi["empty_p"], :] = 28.0                      # 28 * (H7 - 6)
br7[fi["empty_p"]] = -28.0 * 6
dense("f_row7", rowmax7, Wr7, br7)
Wbh7 = np.zeros((NF, 22 * 7), np.float32)
bbh7 = np.zeros(NF, np.float32)
Wbh7[fi["solid_p"], :] = -2.2                     # -2.2 * sum_j (BH7_j - 6)
bbh7[fi["solid_p"]] = 2.2 * 6 * 7
dense("f_bh7", bhp7, Wbh7, bbh7)
Wc = np.zeros((NF, 28), np.float32)
Wc[fi["Wd"], :] = 1.0
Wc[fi["tw_p"], :] = -0.4
Wc[fi["tw_n"], :] = 0.4
dense("f_col", colmax, Wc, np.zeros(NF))

# --- centre valley / ridge maps (2 channels): centre columns
Wcvd = np.zeros((NF, 2 * 784), np.float32)
Wcvd[fi["cv_center"], g2[0, 3:27, 13:15].ravel()] = 1.0   # seam / open front, rows 3-26
Wcvd[fi["cv_low"], g2[0, 14:28, 13:15].ravel()] = 1.0     # seam between the legs (lower half)
Wcvd[fi["rv_center"], g2[1, 6:27, 12:16].ravel()] = 1.0   # bright placket, rows 6-26, cols 12-15
dense("f_cv", cvr, Wcvd, np.zeros(NF))

# --- lapel map: centre columns, rows 2-9
Wcv2d = np.zeros((NF, 784), np.float32)
Wcv2d[fi["cv_lapel"], g[0, 2:10, 13:15].ravel()] = 1.0
dense("f_cv2", cv2r, Wcv2d, np.zeros(NF))

# --- texture map: lower part of the garment (rows 16-27), where T-shirt prints rarely reach
Wt = np.zeros((NF, 784), np.float32)
Wt[fi["tex_low"], g[0, 16:28, :].ravel()] = 1.0
dense("f_tex", texr, Wt, np.zeros(NF))

# --- hole maps
Wh1 = np.zeros((NF, 2 * 784), np.float32)
Wh1[fi["gap_low"], g2[0, 12:28, 9:19].ravel()] = 1.0      # leg gap: horizontal holes, lower centre
Wh1[fi["hole_total"], :] = 1.0
Wh1[fi["bag_loop"], g2[1, 0:12, :].ravel()] = 1.0         # vertical holes in the top rows: handle loop
dense("f_hole", holer, Wh1, np.zeros(NF))
Wh2d = np.zeros((NF, 2 * 784), np.float32)
Wh2d[fi["hole_total"], :] = 1.0
Wh2d[fi["bag_loop"], g2[1, 0:12, :].ravel()] = 1.0
dense("f_hole2", holer2, Wh2d, np.zeros(NF))

# --- top-edge map: per row, edge length beyond 16 columns (0.6 per full edge pixel -> 9.6)
Wed = np.zeros((NF, 784), np.float32)
bed = np.zeros(NF, np.float32)
for r in range(28):
    Wed[fi["edge%d" % r], g[0, r, :]] = 1.0
    bed[fi["edge%d" % r]] = -9.6
dense("f_edge", edger, Wed, bed)

add("F1", ["f_mask", "f_int", "f_thin", "f_bw", "f_bw2", "f_bh", "f_row", "f_row7", "f_bh7", "f_col", "f_cv", "f_cv2",
           "f_tex", "f_hole", "f_hole2", "f_edge"])
F1 = relu("F1r", "F1")

# ----------------------------------------------------------------------------------------------
# 5. indicator ramps  ind(x; t, w): 0 for x <= t - w/2, 1 for x >= t + w/2
#    x is a linear combination of stage-4 features: list of (feature, coefficient)
# ----------------------------------------------------------------------------------------------
IND = [
    # name,       expression,                                      t,     w
    ("tallI",     [("H", 1.0)],                                    25.0,  3.0),   # spans the full height
    ("wideI",     [("Wd", 1.0)],                                   25.0,  3.0),   # spans the full width
    ("midI",      [("Wd", 1.0)],                                   18.0,  4.0),   # wide enough for a top with sleeves
    ("midwI",     [("Wd", 1.0)],                                   21.0,  4.0),   # wide enough for a bag body
    ("vnarrowI",  [("Wd", 1.0)],                                   13.0,  4.0),   # wider than a pair of legs
    ("w23I",      [("Wd", 1.0)],                                   23.0,  4.0),   # nearly as wide as tall (boxy top)
    ("h17I",      [("H", 1.0)],                                    17.0,  4.0),   # taller than a low shoe
    ("h14I",      [("H", 1.0)],                                    14.0,  4.0),   # taller than a typical sneaker
    ("riseI",     [("rise_p", 1.0), ("rise_n", 1.0)],              8.0,   4.0),   # one end much taller: shoe profile
    ("oneI",      [("ups_p", 1.0), ("ups_n", 1.0), ("upper", -0.5)], 20.0, 10.0), # top-third mass on one side: boot shaft
    ("tallheelI", [("BH5", 1.0), ("BH6", 1.0)],                    30.0,  6.0),   # right 8 columns average >= 16 rows tall
    ("sparseI",   [("sparse_p", 1.0)],                             15.0,  15.0),  # almost no interior: straps
    ("thinI",     [("thin_total", 1.0)],                           4.0,   4.0),   # 1-px lines present (straps)
    ("solidI",    [("solid_p", 1.0)],                              10.0,  20.0),  # columns filled over their extent
    ("topfillI",  [("topfill_p", 1.0)],                            2.0,   4.0),   # solid top band (collar / shoulders)
    ("emptyI",    [("empty_p", 1.0)],                              100.0, 60.0),  # bounding box mostly empty
    ("holeI",     [("hole_total", 1.0)],                           10.0,  8.0),   # many small openings
    ("flareI",    [("flare_p", 1.0)],                              5.0,   3.0),   # hem wider than waist (dress)
    ("waistI",    [("waist_p", 1.0)],                              2.0,   2.0),   # fitted waist (dress)
    ("flare2I",   [("flare2_p", 1.0)],                             1.5,   3.0),   # hem wider than the sleeve span
    ("topwideI",  [("tw_p", 1.0), ("tw_n", -1.0)],                 0.0,   3.0),   # waistband-like top (trousers)
    ("gapI",      [("gap_low", 1.0)],                              3.0,   3.0),   # vertical gap between legs
    ("lowcvI",    [("cv_low", 1.0)],                               2.0,   2.0),   # dark seam down the lower centre
    ("sleeveI",   [("sleeve_p", 1.0)],                             4.0,   3.0),   # sleeves stick out beyond the body
    ("shortI",    [("sdrop_p", 1.0)],                              6.0,   4.0),   # width drops below the sleeves
    ("taperI",    [("sdrop_p", 1.0)],                              3.0,   3.0),   # sleeves spread / taper (pullover)
    ("cuffI",     [("cuff_p", 1.0)],                               4.0,   3.0),   # sleeves end above the hem (coat)
    ("varI",      [("tv%d_p" % b, 1.0) for b in range(6)] + [("tv%d_n" % b, 1.0) for b in range(6)], 7.0, 4.0),
    #                                                                              # width profile not uniform
    ("coatI",     [("cv_center", 1.0), ("cv_lapel", 1.5)],         1.5,   1.5),   # centre seam / deep V: coat
    ("ridgeI",    [("rv_center", 1.0)],                            2.5,   2.5),   # bright centre placket: shirt
    ("bagtopI",   [("bagtop_p", 1.0)],                             20.0,  4.0),   # narrow handle over a wide body
    ("bagfullI",  [("BWf0", 1.0)],                                 24.0,  3.0),   # top two rows span the frame: boxy bag
    ("jumpI",     [("jump%d" % b, 1.0) for b in range(7)],         3.0,   3.0),   # abrupt widening: body top edge
    ("longedgeI", [("edge%d" % r, 1.0) for r in range(28)],        1.2,   1.2),   # a long horizontal top edge
    ("texI",      [("tex_low", 1.0)],                              8.0,   6.0),   # patterned fabric (plaid / stripes)
    ("bagloopI",  [("bag_loop", 1.0)],                             2.0,   2.0),   # enclosed loop at the top (handle)
]
ind_names = [n for n, _, _, _ in IND]
W2 = np.zeros((2 * len(IND), NF), np.float32)
b2 = np.zeros(2 * len(IND), np.float32)
for k, (name, expr, t, w) in enumerate(IND):
    for fname, coef in expr:
        W2[2 * k, fi[fname]] = coef / w
        W2[2 * k + 1, fi[fname]] = coef / w
    b2[2 * k] = -t / w + 0.5
    b2[2 * k + 1] = -t / w - 0.5
dense("S2", F1, W2, b2)
relu("S2r", "S2")

W3 = np.zeros((len(IND), 2 * len(IND)), np.float32)
for k in range(len(IND)):
    W3[k, 2 * k] = 1.0
    W3[k, 2 * k + 1] = -1.0
dense("S3", "S2r", W3, np.zeros(len(IND)))
S3 = relu("S3r", "S3")
ii = {n: i for i, n in enumerate(ind_names)}

# ----------------------------------------------------------------------------------------------
# 6. rules: AND of indicators ("~x" means 1 - x), optionally with small evidence terms inside the gate:
#    unit = relu(sum gates - n + c + sum w_i ind_i), with c + sum of positive w_i <= 1 so that a closed
#    gate still switches the unit off (all gates on -> c + sum w_i ind_i).  Each rule -> (class, weight).
# ----------------------------------------------------------------------------------------------
TOP_LONG = ["tallI", "midI", "~shortI", "~oneI", "~flare2I"]    # long-sleeved top cell
LOW = ["~tallI"]                                                 # low / wide cell: shoes and most bags
RULES = [
    # trousers: tall and narrow; waistband-like top / visible leg gap / uniform width without sleeves / seam
    ("trouser_a", ["tallI", "~midI", "topwideI", "~flareI"], 1, 1.0, None),
    ("trouser_b", ["tallI", "~midI", "gapI"], 1, 1.0, None),
    ("trouser_c", ["tallI", "~vnarrowI", "~varI", "~sleeveI", "~flareI"], 1, 0.6, None),
    ("trouser_d", ["tallI", "~midI", "lowcvI", "~flareI", "~waistI", "~sleeveI"], 1, 0.5, None),
    # dresses: tall; flared hem / fitted waist / narrow top, and no leg gap; wide ones need a hem wider
    # than the shoulder span (which no T-shirt or coat has)
    ("dress_a", ["tallI", "~midI", "flareI", "~gapI"], 3, 1.2, None),
    ("dress_b", ["tallI", "~midI", "~topwideI", "~gapI"], 3, 0.8, None),
    ("dress_e", ["tallI", "~midI", "waistI", "~gapI"], 3, 0.8, None),
    ("dress_f", ["tallI", "flare2I", "~gapI", "~oneI", "~bagfullI", "~jumpI", "~bagloopI"], 3, 1.0, None),
    # T-shirts: tall, wide, short sleeves, hem not wider than the sleeves
    ("tshirt_a", ["tallI", "midI", "shortI", "~flare2I", "~oneI"], 0, 1.0, None),
    ("tshirt_b", ["tallI", "~midI", "sleeveI", "shortI", "~flareI"], 0, 0.9, None),
    # long-sleeved tops: evidence inside the gate.  Plain and boxy -> pullover; centre seam / lapel V or
    # sleeves ending above the hem -> coat; bright centre placket -> shirt.
    ("pullover", TOP_LONG, 2, 1.0,
     (0.5, [("taperI", 0.2), ("w23I", 0.1), ("cuffI", -0.2), ("coatI", -0.3), ("ridgeI", -0.2)])),
    ("coat", TOP_LONG, 4, 1.0,
     (0.3, [("coatI", 0.4), ("cuffI", 0.2), ("taperI", -0.1), ("w23I", -0.1)])),
    ("shirt", TOP_LONG, 6, 1.0,
     (0.3, [("ridgeI", 0.4), ("cuffI", 0.1), ("texI", 0.1), ("taperI", -0.1), ("w23I", -0.1)])),
    # tall bags: bag-like body width plus a boxy cue (long flat top edge / full-width top / handle loop /
    # narrow thin top over a wide body / abrupt widening); a solid top band (collar) vetoes handle cues
    ("bag_ta", ["tallI", "midwI", "longedgeI", "~sleeveI", "~oneI", "~topfillI"], 8, 1.5, None),
    ("bag_tb", ["tallI", "bagfullI"], 8, 1.5, None),
    ("bag_tc", ["tallI", "midwI", "bagloopI", "~sleeveI", "~oneI", "~topfillI"], 8, 1.0, None),
    ("bag_td", ["tallI", "midwI", "bagtopI", "~sleeveI", "~oneI", "~topfillI"], 8, 1.0, None),
    ("bag_te", ["tallI", "midwI", "jumpI", "~sleeveI", "~oneI", "~topfillI"], 8, 1.0, None),
    # tall boots (shaft reaching the top of the frame) and tall heeled sandals
    ("boot_tall", ["tallI"], 9, 1.5,
     (0.0, [("oneI", 0.6), ("tallheelI", 0.15), ("riseI", 0.15), ("sparseI", -0.3), ("thinI", -0.3)])),
    ("sandal_tall", ["tallI", "sparseI", "oneI", "~tallheelI"], 5, 1.0, None),
    # low / wide cell: evidence units.  Solid, low, heel higher than toe -> sneaker; strappy, thin, empty
    # bounding box, openings -> sandal; tall, one-sided shaft, tall heel side -> boot; symmetric, taller,
    # solid -> bag.
    ("sneaker", LOW, 7, 1.5,
     (0.5, [("solidI", 0.2), ("riseI", 0.1), ("h17I", -0.3), ("sparseI", -0.3), ("thinI", -0.3),
            ("emptyI", -0.2), ("oneI", -0.2)])),
    ("sandal", LOW, 5, 1.5,
     (0.0, [("sparseI", 0.3), ("thinI", 0.3), ("emptyI", 0.2), ("holeI", 0.2), ("solidI", -0.4)])),
    ("boot", LOW, 9, 1.5,
     (0.0, [("h17I", 0.35), ("oneI", 0.35), ("tallheelI", 0.3), ("sparseI", -0.3), ("thinI", -0.3)])),
    ("bag_low", LOW, 8, 1.5,
     (0.4, [("riseI", -0.5), ("h14I", 0.3), ("solidI", 0.2), ("thinI", -0.3), ("sparseI", -0.3),
            ("oneI", -0.3)])),
]
NR = len(RULES)
NI = len(IND)
W4 = np.zeros((NR + NI, NI), np.float32)
b4 = np.zeros(NR + NI, np.float32)
for r, (name, gates, cls, wgt, extra) in enumerate(RULES):
    bias = -(len(gates) - 1)
    for gname in gates:
        if gname.startswith("~"):
            W4[r, ii[gname[1:]]] = -1.0
            bias += 1.0
        else:
            W4[r, ii[gname]] = 1.0
    if extra is not None:
        c, terms = extra
        assert c + sum(max(w, 0.0) for _, w in terms) <= 1.0 + 1e-6, name   # a closed gate still wins
        bias += c - 1.0                           # all gates on -> output = c + sum w_i ind_i; one off -> <= 0
        for iname, w in terms:
            W4[r, ii[iname]] += w
    b4[r] = bias
for k in range(NI):                               # pass the indicators through for linear terms
    W4[NR + k, k] = 1.0
dense("S4", S3, W4, b4)
S4 = relu("S4r", "S4")

# ----------------------------------------------------------------------------------------------
# 7. class scores
# ----------------------------------------------------------------------------------------------
W5 = np.zeros((10, NR + NI), np.float32)
b5 = np.zeros(10, np.float32)
for r, (name, gates, cls, wgt, extra) in enumerate(RULES):
    W5[cls, r] = wgt
LIN = [
    (8, "bagloopI", 0.2),                                      # handle loop supports bag
    (1, "gapI", 0.3),                                          # leg gap supports trousers
]
for cls, name, wgt in LIN:
    W5[cls, NR + ii[name]] += wgt
dense("out", S4, W5, b5)

# ----------------------------------------------------------------------------------------------
out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "network.json").write_text(json.dumps({"layers": layers}, indent=1))
np.savez(out_dir / "weights.npz", **{k: v.astype(np.float32) for k, v in weights.items()})
print("wrote", len(layers), "layers,", sum(int(v.size) for v in weights.values()), "weights")
