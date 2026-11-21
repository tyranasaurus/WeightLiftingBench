#!/usr/bin/env python3
"""build.py - writes network.json and weights.npz for a hand-designed clothing classifier.

Reads no files.  Every number below is typed by hand from general knowledge of what the ten
kinds of item look like in a shop photo (their real-world proportions, scaled so the longest side
is 28 pixels and centred), or comes from a formula (running sums, corner / gap indicator logic,
a line filter).  Nothing is fitted, averaged or copied from any image.

How the network computes (all with conv / dense / relu / add / reshape / maxpool):

  1. mask      m = clip((image - T_LO) / (T_HI - T_LO), 0, 1): "is there an item at this pixel".
  2. running sums of the mask along the four directions, clipped to [0, 1]:
       PL(y, x) = "there is item at or left of x in row y", PR, PT (above), PB (below).
  3. per-pixel logic on (PL, PR, PT, PB, m) with relu, e.g.
       HH   = PL and PR                      (inside the left-right extent of the row)
       EH   = PL and PR and PT and PB, empty (a hole: handle loop, gap between sandal straps)
       LEG  = PL and PR and PT, not PB, empty (gap open to the bottom: between trouser legs, under a hem)
       NECK = PL and PR and PB, not PT, empty (gap open to the top: neckline, between straps)
       UL/UR= item above and on the inner side, nothing on the outer side (under a short sleeve)
       TLC/TRC = item below and to one side only (top corners: above a shoe's toe, beside a handle)
       HB   = PL and PR and PB, empty        (an opening above a sole; kept only where 3+ pixels tall)
  4. side branches from formulas: interior pixels (3 x 3 erosion of the mask); a "dark valley" map on the
     raw image (a pixel darker than the pixels 2 to its left and 2 to its right, averaged over 5 rows:
     blurred gaps between legs or between sleeve and body); a centre-line filter (zip / button line).
  5. sums of those maps over regions give a few dozen shape measurements (height, width, widths of the
     outline in bands of rows, sleeve overhang, leg gap, hem gap, holes, left/right asymmetry, thin sole
     columns, height of the toe end and of the heel end, ...).
  6. a hand-written rule table: each class has one or more prototypes; a prototype is a list of
     "this measurement should lie in this range" terms, each a capped piecewise-linear penalty (the
     low-sandal prototype instead adds up capped pieces of evidence).  A prototype score is at most 0;
     class score = best (max) prototype score.
"""
import json
import os
import sys

import numpy as np

S = 28
NP_ = S * S
T_LO, T_HI = 0.04, 0.10      # mask: 0 below 4% grey, 1 above 10% grey (background is black)
P_OFF, P_W = 0.25, 0.5        # running sum -> presence: clip((sum - 0.25) / 0.5, 0, 1)
KAPPA = 2.5                   # a pixel counts as "empty" for gap maps only if its mask is below 1/2.5

# ----------------------------------------------------------------------------- maps (stage 3)
# coefficients on (PL, PR, PT, PB, m) and a bias; output = relu(sum)
MAPS = [
    ("M",    (0, 0, 0, 0, 1), 0),
    ("PL",   (1, 0, 0, 0, 0), 0),
    ("PR",   (0, 1, 0, 0, 0), 0),
    ("PT",   (0, 0, 1, 0, 0), 0),
    ("PB",   (0, 0, 0, 1, 0), 0),
    ("HH",   (1, 1, 0, 0, 0), -1),
    ("VH",   (0, 0, 1, 1, 0), -1),
    ("HG",   (1, 1, 0, 0, -KAPPA), -1),
    ("VG",   (0, 0, 1, 1, -KAPPA), -1),
    ("EH",   (1, 1, 1, 1, -KAPPA), -3),
    ("LEG",  (1, 1, 1, -1, -KAPPA), -2),
    ("NECK", (1, 1, -1, 1, -KAPPA), -2),
    ("UL",   (-1, 1, 1, 0, -KAPPA), -1),
    ("UR",   (1, -1, 1, 0, -KAPPA), -1),
    ("TLC",  (-1, 1, -1, 1, -KAPPA), -1),
    ("TRC",  (1, -1, -1, 1, -KAPPA), -1),
    ("HB",   (1, 1, 0, 1, -KAPPA), -2),     # empty, item left, right and below (an opening above a sole)
]
MAP_IX = {n: i for i, (n, _, _) in enumerate(MAPS)}
K = len(MAPS)

CL_ROWS = (8, 23)             # rows over which a centre line is averaged
CL_COLS = (11, 16)            # columns where a centre line may sit
CL_KAPPA = 0.06               # line must exceed 6% of the mean brightness of the middle of the item
GAP_COLS = (9, 18)            # leg gap columns
GAP_ROW0 = 10                 # leg gap rows start here
GAP_MIN = 5.0                 # a leg gap column must be more than 5 rows tall
V_TAU = 0.25                  # valley: a pixel at least 25% darker than the pixels 2 to its left and 2 to its right
V_THETA = 0.08                # ... by at least 0.08 grey levels, then it counts as one full valley pixel
TOE_FRAC = 0.45               # left-end columns taller than 45% of the item height: a box, not the toe of a shoe
BACK_FRAC, BACK_C = 0.78, 0.4 # right-end columns taller than 78% of the item height: the back of a boot shaft
THIN_LO, THIN_HI, THIN_END = 1.0, 3.0, 4.5   # a "thin column" holds 1..3 item pixels (fading out by 4.5)
SHAFT_FRAC = 0.75             # sum over columns of relu(height - 75% of item height) = area in the top quarter
SHAFT_W = 7.0                 # ... compared with a top quarter that is 7 pixels wide on average


def region(rows=None, cols=None, val=1.0):
    a = np.zeros((S, S))
    r0, r1 = rows if rows is not None else (0, S - 1)
    c0, c1 = cols if cols is not None else (0, S - 1)
    a[r0:r1 + 1, c0:c1 + 1] = val
    return a


class F1:
    """First feature layer: units = relu(linear functional of maps / eroded mask / texture maps + bias)."""

    def __init__(self):
        self.names, self.ix = [], {}
        self.Wm, self.We, self.Wt, self.Wv, self.Wh, self.b = [], [], [], [], [], []

    def unit(self, name, maps=None, er=None, tex=None, val=None, hb=None, bias=0.0):
        wm = np.zeros((K, S, S))
        for mname, arr in (maps or {}).items():
            wm[MAP_IX[mname]] += arr
        we = np.zeros((2, S, S))
        if er is not None:                                   # interior = (relu(n - 6) - relu(n - 8)) / 2
            we[0] += 0.5 * np.array(er, float); we[1] -= 0.5 * np.array(er, float)
        wt = np.zeros((3, S, S))
        for c, arr in (tex or {}).items():
            wt[c] += arr
        wv = np.zeros((2, S, S))
        if val is not None:                                  # valley pixel count = (relu(V) - relu(V - theta)) / theta
            wv[0] += np.array(val, float) / V_THETA; wv[1] -= np.array(val, float) / V_THETA
        self.ix[name] = len(self.names)
        self.names.append(name)
        self.Wm.append(wm); self.We.append(we); self.Wt.append(wt); self.Wv.append(wv); self.b.append(bias)
        self.Wh.append(np.zeros((S, S)) if hb is None else np.array(hb, float))
        return name


def make_f1():
    f = F1()
    one = region()
    f.unit("A", {"M": one})
    rowp = np.zeros((S, S)); rowp[:, S - 1] = 1.0           # PL(y, 27) = row y has item
    colp = np.zeros((S, S)); colp[S - 1, :] = 1.0           # PT(27, x) = column x has item
    f.unit("BH", {"PL": rowp})
    f.unit("BW", {"PT": colp})
    for name, (r0, r1) in (("wT", (0, 1)), ("wU", (3, 7)), ("wM", (11, 15)), ("wL", (17, 22)), ("wB", (24, 26))):
        f.unit(name, {"HH": region(rows=(r0, r1), val=1.0 / (r1 - r0 + 1))})
    f.unit("U", {"UL": one, "UR": one})
    f.unit("NK", {"NECK": region(rows=(0, 9))})
    f.unit("EH", {"EH": one})
    f.unit("EHt", {"EH": region(rows=(0, 12))})
    f.unit("TLC", {"TLC": one})
    f.unit("TRC", {"TRC": one})
    sg = region(rows=(8, 16), cols=(2, 8)) + region(rows=(8, 16), cols=(19, 25))
    f.unit("SG", {"HG": sg})                                 # empty gap between sleeve and body, upper part
    f.unit("SV", val=region(rows=(8, 15), cols=(3, 9)) + region(rows=(8, 15), cols=(18, 24)))   # same, as dark valley
    f.unit("LSG", val=region(rows=(13, 23), cols=(2, 9)) + region(rows=(13, 23), cols=(18, 25)))  # sleeve / body valley, lower part
    for x in range(1, 7):                                    # height of the left-end columns above TOE_FRAC of the item height
        f.unit(f"toe{x}", {"VH": region(cols=(x, x)), "PL": -TOE_FRAC * rowp})
    for x in range(21, 27):                                  # height of the right-end columns above BACK_FRAC of the item height
        f.unit(f"back{x}", {"VH": region(cols=(x, x)), "PL": -BACK_FRAC * rowp})
    f.unit("LV", val=region(rows=(13, 26), cols=(11, 16)))   # dark valley between the legs
    f.unit("LEGtot", {"LEG": one})                           # any gap open to the bottom (under a heel, under a hem)
    f.unit("OPN3", hb=one)                                   # openings above a sole that are at least 3 pixels tall
    f.unit("HEMc", {"LEG": region(rows=(18, S - 1), cols=GAP_COLS)})   # empty band under the hem, between the cuffs
    f.unit("OPH", {"HG": one})
    f.unit("OPV", {"VG": one})
    f.unit("At", {"M": region(rows=(0, 6))})
    f.unit("I", er=one)                                      # number of interior pixels (3x3 neighbourhood all item)
    for x in range(S):                                       # item pixels in column x, hinged for the "thin column" test
        col = region(cols=(x, x))
        f.unit(f"cm{x}_0", {"M": col})
        f.unit(f"cm{x}_1", {"M": col}, bias=-THIN_LO)
        f.unit(f"cm{x}_a", {"M": col}, bias=-THIN_HI)
        f.unit(f"cm{x}_b", {"M": col}, bias=-THIN_END)
    # leg gap columns: relu(height of the bottom-open gap in column x - GAP_MIN)
    for x in range(GAP_COLS[0], GAP_COLS[1] + 1):
        col = region(rows=(GAP_ROW0, S - 1), cols=(x, x))
        f.unit(f"gap{x}", {"LEG": col}, bias=-GAP_MIN)
    # neck height pieces: outline width OW(y) of the top rows, hinged at 1, 9.5, 11.5
    for y in range(8):
        row = region(rows=(y, y))
        f.unit(f"ow{y}_0", {"HH": row})
        f.unit(f"ow{y}_1", {"HH": row}, bias=-1.0)
        f.unit(f"ow{y}_a", {"HH": row}, bias=-9.5)
        f.unit(f"ow{y}_b", {"HH": row}, bias=-11.5)
    # shaft columns: relu(height of column x - SHAFT_FRAC * item height)
    for x in range(S):
        f.unit(f"sh{x}", {"VH": region(cols=(x, x)), "PL": -SHAFT_FRAC * rowp})
    # centre line: tex channel 0 = +line response, 1 = -line response, 2 = raw image
    nrow = CL_ROWS[1] - CL_ROWS[0] + 1
    ibar = region(rows=CL_ROWS, cols=(10, 17), val=1.0 / (nrow * 8))
    for x in range(CL_COLS[0], CL_COLS[1] + 1):
        col = region(rows=CL_ROWS, cols=(x, x), val=1.0 / nrow)
        f.unit(f"clp{x}", tex={0: col, 1: -col, 2: -CL_KAPPA * ibar})
        f.unit(f"cln{x}", tex={0: -col, 1: col, 2: -CL_KAPPA * ibar})
    return f


def make_features(f1):
    """Scalar measurements = linear combinations of f1 units.  name -> {unit: coefficient}."""
    F = {}
    for n in ("A", "BH", "BW", "wT", "wU", "wM", "wL", "wB", "U", "NK", "EH", "EHt", "TLC", "TRC", "SG", "At"):
        F[n] = {n: 1.0}
    F["ASYM"] = {"TLC": 1.0, "TRC": -1.0}                    # > 0: empty top-left corner (shoe pointing left)
    F["CORN"] = {"TLC": 1.0, "TRC": 1.0}
    F["LGt"] = {f"gap{x}": 1.0 for x in range(GAP_COLS[0], GAP_COLS[1] + 1)}
    nh = {}
    for y in range(8):                                       # rows whose outline is 1..~10.5 wide
        nh[f"ow{y}_0"] = 1.0; nh[f"ow{y}_1"] = -1.0; nh[f"ow{y}_a"] = -0.5; nh[f"ow{y}_b"] = 0.5
    F["NH"] = nh
    # openness: gaps inside the item, thin strokes (few interior pixels)
    F["SP"] = {"OPH": 1.0, "A": 0.25, "I": -1.0}             # (gaps along rows only: a shadow strip under a shoe is not a strap)
    sh = {f"sh{x}": 1.0 for x in range(S)}
    sh["BH"] = -(1.0 - SHAFT_FRAC) * SHAFT_W                 # > 0: the top quarter is wider than SHAFT_W (a boot shaft)
    F["SHAFT"] = sh
    F["CL"] = {**{f"clp{x}": 1.0 for x in range(CL_COLS[0], CL_COLS[1] + 1)},
               **{f"cln{x}": 1.0 for x in range(CL_COLS[0], CL_COLS[1] + 1)}}
    F["dML"] = {"wM": 1.0, "wL": -1.0}
    F["dUB"] = {"wU": 1.0, "wB": -1.0}
    F["Q"] = {"wL": 1.0, "TLC": 0.1, "TRC": 0.1}
    F["EHb"] = {"EH": 1.0, "EHt": -1.0}                      # holes below the top part
    th = {}
    for x in range(S):                                       # number of columns holding only 1..3 item pixels
        th[f"cm{x}_0"] = 1.0 / THIN_LO; th[f"cm{x}_1"] = -1.0 / THIN_LO
        th[f"cm{x}_a"] = -1.0 / (THIN_END - THIN_HI); th[f"cm{x}_b"] = 1.0 / (THIN_END - THIN_HI)
    F["THINC"] = th
    F["G"] = {**F["LGt"], "LV": 0.7}                         # leg gap evidence: empty gap or dark valley
    F["SLV"] = {"SG": 1.0, "SV": 1.0}                        # upper sleeve gap evidence: empty gap or dark valley
    F["LSG"] = {"LSG": 1.0}
    F["SGA"] = {"SG": 1.0, "SV": 1.0, "LSG": 1.0}            # sleeve gap anywhere
    F["TOEH"] = {f"toe{x}": 1.0 for x in range(1, 7)}
    F["HO"] = {"EHt": 1.0, "NK": 1.0}                        # any opening in the top part (neckline, straps, handle)
    F["OPN3"] = {"OPN3": 1.0}                                # openings with item left, right and below, counted only where
                                                             # they are 3+ pixels tall (a thin dark line above a sole strip or
                                                             # a shallow dip behind the tongue belongs to a sneaker; the gaps
                                                             # between sandal straps and under an ankle strap are taller)
    bk = {f"back{x}": 1.0 for x in range(21, 27)}
    bk["BH"] = -BACK_C                                       # > 0: the back of the shoe is as tall as the whole shoe
    F["BACK"] = bk
    F["LOOP"] = {"EHt": 2.0, "EH": -1.0}                     # holes in the top part minus holes lower down
    F["HEMc"] = {"HEMc": 1.0}
    F["LEGtot"] = {"LEGtot": 1.0}
    F["dBM"] = {"wB": 1.0, "wM": -1.0}                       # hem wider than waist
    return F


# ----------------------------------------------------------------------------- rule table
def ge(feat, lo, tol, cap):
    """penalty 0 if feat >= lo, growing by 1 per `tol` below it, at most `cap`."""
    return (feat, [(lo - cap * tol, -cap), (lo, 0.0)])


def le(feat, hi, tol, cap):
    return (feat, [(hi, 0.0), (hi + cap * tol, -cap)])


def bonus(feat, a, b, w):
    """0 at feat = a, rising linearly to w at feat = b (b may be below a), constant outside."""
    return (feat, [(a, 0.0), (b, w)] if a < b else [(b, w), (a, 0.0)])


def rng(feat, lo, hi, tol, cap):
    return (feat, [(lo - cap * tol, -cap), (lo, 0.0), (hi, 0.0), (hi + cap * tol, -cap)])


def prototypes():
    """(class, name, base score, [terms]).  All numbers are in pixels of the 28 x 28 frame.

    Proportions used (item scaled so that its longest side is 28):
      tops are about as tall as the frame; a T-shirt body is roughly half as wide as it is long
      (13-18 px) and its short sleeves stick out a few pixels on each side over the top third only;
      long sleeves hang down beside the body to the hem, so the outline stays wide (19+ px) all the
      way down; trousers and dresses are 2-3 times taller than wide; a dress has a narrow waist;
      trousers have a gap between the legs; shoes are as wide as the frame, point left, and are
      low (sneaker ~0.4 of length, flat sandal less, boot more with a full-height shaft at the back);
      bags are wide boxes, symmetric, often with a thin handle loop above.
    """
    garment = [ge("BH", 24.5, 1.2, 5), rng("ASYM", -30, 30, 10, 5)]
    no_loop = [le("LOOP", 22, 6, 4)]                         # no big hole in the top part only (that would be a bag strap)
    P = []
    # 0 T-shirt / top -------------------------------------------------------------------------
    P.append((0, "tshirt", 0.0, garment + [
        ge("U", 70, 12, 6), rng("wM", 12.3, 20.5, 1.2, 6), rng("wL", 11.5, 20.5, 1.5, 5),
        ge("BW", 19.5, 1.5, 4), le("LGt", 5, 3, 4), ge("At", 60, 10, 4)]))
    P.append((0, "tank", 0.0, garment + [
        le("U", 40, 12, 5), rng("wM", 13.3, 18.5, 1.2, 6), rng("wL", 13.3, 19.0, 1.2, 6),
        rng("wB", 12.5, 20, 1.5, 4), le("BW", 20, 1.5, 5), ge("NK", 14, 4, 3), le("LGt", 5, 3, 4),
        le("EHt", 10, 5, 3)]))
    # 1 trouser -------------------------------------------------------------------------------
    P.append((1, "trouser_gap", 0.0, garment + no_loop + [
        le("BW", 16, 1.5, 6), ge("G", 6, 2, 6), le("HO", 8, 4, 3), le("U", 70, 15, 4)]))
    P.append((1, "trouser_closed", -0.5, garment + no_loop + [
        le("BW", 14.5, 1.5, 6), le("NK", 3, 2, 4), ge("wT", 7, 1.5, 4), ge("dUB", 0.5, 1.0, 4),
        ge("dML", -0.3, 0.7, 3), le("wM", 13, 1.5, 4), le("U", 60, 15, 4)]))
    # 2 pullover ------------------------------------------------------------------------------
    P.append((2, "pullover", 0.0, garment + [
        le("U", 45, 12, 6), ge("wM", 18.5, 1.5, 6), ge("wL", 19.5, 1.5, 6), ge("At", 60, 10, 4),
        le("NH", 2.0, 1.0, 1.5), le("CL", 0.02, 0.04, 1.2), ge("wL", 21.5, 2.0, 1.0), le("SLV", 14, 8, 1.0),
        ge("HEMc", 8, 4, 1.5), ge("LSG", 8, 4, 1.2)]))
    # 3 dress ---------------------------------------------------------------------------------
    P.append((3, "dress", 0.0, garment + no_loop + [
        le("wM", 12.3, 1.2, 6), le("wL", 15.5, 1.5, 6), le("BW", 20.5, 1.5, 5), le("G", 4, 3, 6),
        le("U", 120, 20, 4), le("dML", 0.5, 1.0, 1.5), le("wU", 20, 1.5, 4)]))
    P.append((3, "dress_sleeves", 0.0, garment + no_loop + [
        le("wL", 15.5, 1.5, 6), rng("wM", 10, 19, 1.5, 5), le("wB", 17, 1.5, 4), le("BW", 21.5, 1.5, 5),
        ge("dML", 1.0, 1.0, 4), le("G", 4, 3, 5), le("wU", 19.5, 1.5, 4)]))
    P.append((3, "dress_flare", 0.0, garment + no_loop + [
        le("wM", 11.5, 1.2, 6), ge("dBM", 4, 1.0, 4), le("wU", 20, 1.5, 4), le("G", 4, 3, 6), le("U", 120, 20, 4)]))
    # 4 coat ----------------------------------------------------------------------------------
    P.append((4, "coat", 0.0, garment + [
        le("U", 95, 15, 6), ge("wM", 16.5, 1.5, 6), ge("wL", 16.5, 1.5, 6), ge("At", 60, 10, 4),
        ge("NH", 2.5, 1.0, 1.2), ge("CL", 0.05, 0.04, 0.6), le("wL", 22.5, 2.0, 1.0), le("SLV", 14, 8, 1.2),
        le("NK", 5, 4, 1.0), le("LSG", 5, 4, 1.5)]))
    # 5 sandal --------------------------------------------------------------------------------
    # low sandals: evidence adds up (open gaps along rows, thin sole columns, very flat); 2 units make a full match
    P.append((5, "sandal_any", -2.0, [
        bonus("OPN3", 4, 16, 2.0), bonus("THINC", 3, 11, 2.0), bonus("BH", 11, 7, 2.0),
        ge("BW", 20, 1.5, 5), le("BH", 20, 2.0, 4), le("TOEH", 6, 3, 4)]))
    P.append((5, "sandal_heel", 0.0, [
        ge("OPN3", 10, 4, 5), ge("ASYM", 40, 10, 5), ge("BW", 17, 1.5, 5), ge("BH", 12, 1.5, 4), le("TOEH", 6, 3, 4)]))
    # 6 shirt ---------------------------------------------------------------------------------
    P.append((6, "shirt_long", -0.3, garment + [
        le("U", 45, 10, 5), ge("wM", 17, 1.5, 6), ge("wL", 17.5, 1.5, 6), ge("At", 60, 10, 4),
        le("HEMc", 4, 4, 2.5), ge("SGA", 14, 6, 2.0), rng("NH", 1.0, 3.5, 1.0, 1.5)]))
    # 7 sneaker -------------------------------------------------------------------------------
    P.append((7, "sneaker", 0.0, [
        ge("BH", 7.8, 1.0, 6), le("BH", 17.5, 2.5, 3), ge("BW", 24.5, 1.5, 5), ge("ASYM", 22, 8, 5),
        le("SP", -15, 15, 4), le("SHAFT", -2, 6, 3), le("BACK", -1, 4, 2), le("THINC", 6, 2.0, 3),
        le("LEGtot", 6, 4, 1.5), le("OPN3", 8, 4, 3)]))
    # 8 bag -----------------------------------------------------------------------------------
    P.append((8, "bag_box", 0.0, [
        rng("BH", 14, 26.0, 0.5, 6), ge("BW", 19, 1.5, 5), rng("ASYM", -22, 22, 8, 3), ge("TOEH", 8, 3, 4),
        le("SP", 10, 15, 4), le("U", 60, 15, 4), le("LGt", 6, 3, 3), le("LEGtot", 10, 5, 3)]))
    # a low wide bag (clutch, wallet) must be a plain rectangle, otherwise it is more likely a low shoe
    P.append((8, "bag_flat", 0.0, [
        rng("BH", 9, 15, 0.7, 6), ge("BW", 22, 1.5, 5), rng("ASYM", -15, 15, 6, 4), le("CORN", 20, 6, 4),
        ge("TOEH", 8, 3, 4), le("SP", 10, 15, 4), le("LEGtot", 8, 4, 3), le("THINC", 4, 2.0, 3)]))
    P.append((8, "bag_handle", 0.0, [
        ge("LOOP", 14, 4, 5), le("At", 60, 12, 5), ge("wL", 16.5, 1.5, 5), ge("Q", 20.5, 1.2, 4),
        rng("ASYM", -35, 35, 10, 5), le("U", 60, 15, 4), ge("BH", 16, 1.5, 4)]))
    P.append((8, "bag_straps", -0.5, [
        ge("NK", 30, 6, 5), ge("CORN", 50, 10, 4), le("At", 50, 10, 5), ge("wL", 17.5, 1.5, 5),
        rng("ASYM", -30, 30, 10, 5), le("U", 50, 15, 4), ge("BH", 18, 1.5, 4)]))
    P.append((8, "bag_pouch", -0.3, [
        ge("LOOP", 30, 6, 5), le("At", 45, 10, 5), ge("BH", 18, 1.5, 4), le("U", 50, 15, 4),
        rng("ASYM", -35, 35, 10, 5), ge("wL", 8, 1.5, 4)]))
    # a tall box with a flat wide top and square corners (tote or backpack with no visible handle loop)
    P.append((8, "bag_tall", 0.0, [
        ge("BH", 24, 1.5, 4), ge("BW", 18, 1.5, 4), ge("wT", 18, 1.5, 5), ge("wB", 16, 1.5, 4), le("CORN", 16, 6, 4),
        le("U", 40, 15, 4), le("LEGtot", 8, 4, 3), le("HO", 15, 6, 3)]))
    # 9 ankle boot ----------------------------------------------------------------------------
    boot = [ge("BH", 13.5, 0.8, 6), ge("BW", 17, 1.5, 5), ge("ASYM", 45, 12, 6), le("SP", -10, 15, 4),
            le("THINC", 4.5, 1.5, 4), le("TOEH", 3, 3, 4), le("OPN3", 8, 4, 4)]
    P.append((9, "boot_top", 0.0, boot + [ge("SHAFT", 2, 5, 4)]))      # wide flat top
    P.append((9, "boot_back", 0.0, boot + [ge("BACK", 1, 4, 4)]))      # full-height back (a pull tab may stick up)
    return P


# ----------------------------------------------------------------------------- network assembly
def build():
    layers, wts = [], {}

    def L(name, op, inputs, **kw):
        layers.append(dict(name=name, op=op, inputs=list(inputs), **kw))
        return name

    def conv(name, inp, W, b, padding=0):
        W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
        wts[name + ".W"], wts[name + ".b"] = W, b
        return L(name, "conv", [inp], out=int(W.shape[0]), kernel=[int(W.shape[2]), int(W.shape[3])], stride=1, padding=padding)

    def dense(name, inp, W, b):
        W = np.asarray(W, np.float32); b = np.asarray(b, np.float32)
        wts[name + ".W"], wts[name + ".b"] = W, b
        return L(name, "dense", [inp], out=int(W.shape[0]))

    # 1. mask
    conv("mk", "image", np.ones((2, 1, 1, 1)), [-T_LO, -T_HI])
    L("mkr", "relu", ["mk"])
    g = 1.0 / (T_HI - T_LO)
    conv("m", "mkr", np.array([g, -g]).reshape(1, 2, 1, 1), [0.0])

    # 2. running sums in four directions, two hinge units each, plus the mask itself
    Wc = np.zeros((9, S, S, S, S), np.float32)               # [channel, y, x, y', x']
    bc = np.zeros((9, S, S), np.float32)
    for y in range(S):
        for x in range(S):
            Wc[0, y, x, y, :x + 1] = 1; Wc[1, y, x, y, :x + 1] = 1      # left
            Wc[2, y, x, y, x:] = 1; Wc[3, y, x, y, x:] = 1              # right
            Wc[4, y, x, :y + 1, x] = 1; Wc[5, y, x, :y + 1, x] = 1      # above
            Wc[6, y, x, y:, x] = 1; Wc[7, y, x, y:, x] = 1              # below
            Wc[8, y, x, y, x] = 1
    Wc[:8] /= P_W
    for c in range(8):
        bc[c] = -P_OFF / P_W - (c % 2)
    dense("cum", "m", Wc.reshape(9 * NP_, NP_), bc.reshape(-1))
    L("cumr", "relu", ["cum"])
    L("cumm", "reshape", ["cumr"], shape=[9, S, S])

    # 3. per-pixel logic maps
    Wm = np.zeros((K, 9, 1, 1), np.float32); bm = np.zeros(K, np.float32)
    for i, (_, co, bias) in enumerate(MAPS):
        for d in range(4):
            Wm[i, 2 * d, 0, 0] = co[d]; Wm[i, 2 * d + 1, 0, 0] = -co[d]
        Wm[i, 8, 0, 0] = co[4]
        bm[i] = bias
    conv("maps", "cumm", Wm, bm)
    L("mapsr", "relu", ["maps"])

    # side branches: interior pixels (3x3 erosion of the mask) and centre-line filter on the raw image
    Wh = np.zeros((1, K, 3, 3), np.float32)                  # opening pixel whose upper and lower neighbours are opening pixels too
    Wh[0, MAP_IX["HB"], :, 1] = 1.0
    conv("hb", "mapsr", Wh, [-2.0], padding=1)
    L("hbr", "relu", ["hb"])
    conv("er", "m", np.ones((2, 1, 3, 3)), [-6.0, -8.0], padding=1)
    L("err", "relu", ["er"])
    # dark valley: vertical mean over 5 rows of ((1 - tau) * flank - centre) for the left and the right flank
    Wv = np.zeros((2, 1, 5, 5), np.float32)
    Wv[0, 0, :, 0] = (1 - V_TAU) / 5; Wv[0, 0, :, 2] = -1.0 / 5
    Wv[1, 0, :, 4] = (1 - V_TAU) / 5; Wv[1, 0, :, 2] = -1.0 / 5
    conv("vc", "image", Wv, [0, 0], padding=2)
    L("vcr", "relu", ["vc"])
    conv("vm", "vcr", np.array([[1, 0], [1, -1]], np.float32).reshape(2, 2, 1, 1), [0, 0])
    L("vmm", "relu", ["vm"])                                 # [dl, relu(dl - dr)]: V = min(dl, dr) = first - second
    conv("vb", "vmm", np.array([[1, -1], [1, -1]], np.float32).reshape(2, 2, 1, 1), [0, -V_THETA])
    L("vmr", "relu", ["vb"])                                 # [relu(V), relu(V - theta)]
    Wt = np.zeros((3, 1, 5, 5), np.float32)
    Wt[0, 0, 2] = [-0.5, 0, 1, 0, -0.5]
    Wt[1, 0, 2] = [0.5, 0, -1, 0, 0.5]
    Wt[2, 0, 2, 2] = 1.0
    conv("tc", "image", Wt, [0, 0, 0], padding=2)
    L("tcr", "relu", ["tc"])

    # 4. first feature layer
    f1 = make_f1()
    n1 = len(f1.names)
    dense("f1m", "mapsr", np.array(f1.Wm).reshape(n1, K * NP_), np.array(f1.b))
    dense("f1e", "err", np.array(f1.We).reshape(n1, 2 * NP_), np.zeros(n1))
    dense("f1t", "tcr", np.array(f1.Wt).reshape(n1, 3 * NP_), np.zeros(n1))
    dense("f1v", "vmr", np.array(f1.Wv).reshape(n1, 2 * NP_), np.zeros(n1))
    dense("f1h", "hbr", np.array(f1.Wh).reshape(n1, NP_), np.zeros(n1))
    L("f1", "add", ["f1m", "f1e", "f1t", "f1v", "f1h"])
    L("f1r", "relu", ["f1"])

    # 5. hinge units on the measurements, then prototypes
    feats = make_features(f1)
    protos = prototypes()
    hinge_ix, hinge_rows = {}, []

    def hinge(feat, theta):
        key = (feat, round(float(theta), 6))
        if key not in hinge_ix:
            row = np.zeros(n1 + 1)
            for u, cf in feats[feat].items():
                row[f1.ix[u]] += cf
            row[n1] = -theta
            hinge_ix[key] = len(hinge_rows)
            hinge_rows.append(row)
        return hinge_ix[key]

    compiled = []                                            # (class, base, {hinge: coefficient})
    for cls, _name, base, terms in protos:
        co, const = {}, base
        for feat, knots in terms:
            const += knots[0][1]
            prev = 0.0
            for k, (xk, yk) in enumerate(knots):
                slope = 0.0 if k == len(knots) - 1 else (knots[k + 1][1] - yk) / (knots[k + 1][0] - xk)
                if slope != prev:
                    h = hinge(feat, xk)
                    co[h] = co.get(h, 0.0) + (slope - prev)
                prev = slope
        compiled.append((cls, const, co))
    Hm = np.array(hinge_rows)
    dense("f2", "f1r", Hm[:, :n1], Hm[:, n1])
    L("f2r", "relu", ["f2"])

    per_class = [[c for c in compiled if c[0] == k] for k in range(10)]
    PM = max(len(v) for v in per_class)
    Wp = np.zeros((10 * PM, len(hinge_rows))); bp = np.zeros(10 * PM)
    for k in range(10):
        for j in range(PM):
            _, const, co = per_class[k][min(j, len(per_class[k]) - 1)]     # pad by repeating the last prototype
            bp[k * PM + j] = const
            for h, cf in co.items():
                Wp[k * PM + j, h] = cf
    dense("prn", "f2r", -Wp, -bp)                            # minus the prototype score
    L("prnr", "relu", ["prn"])                               # ... so that a score cannot rise above 0
    dense("pr", "prnr", -np.eye(10 * PM), np.zeros(10 * PM))
    L("prm", "reshape", ["pr"], shape=[1, 10, PM])
    L("prx", "maxpool", ["prm"], size=[1, PM], stride=[1, PM])
    L("out", "reshape", ["prx"], shape=[10])
    return {"layers": layers}, wts


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    net, wts = build()
    with open(os.path.join(out_dir, "network.json"), "w") as fh:
        json.dump(net, fh, indent=1)
    np.savez(os.path.join(out_dir, "weights.npz"), **wts)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
