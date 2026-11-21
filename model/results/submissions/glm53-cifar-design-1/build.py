"""
Hand-designed CIFAR-10 classifier (32x32 colour photos, 10 classes).

Every weight in this file comes from a formula (Gaussian / Gabor / ring
kernels, opponent-colour combinations) or from a hand-written rule about
what the classes look like in everyday photos.  No data is read, nothing
is fitted: the script only needs numpy and writes the two answer files.

Architecture (all hand-set):

  image (3,32,32)
  C1  1x1 conv    fixed colour basis: luminance, R-G, B-Y, raw R,G,B ...
  
  E1  5x5 conv    Gabor edge bank (2 scales, 4 orientations, 2 phases on
                  luminance; odd Gabors on colour-opponent axes), |response|
                  kept via +/- channel pairs                     -> 48 ch
  G1  5x5 conv    smoothed colour-category maps with hand thresholds
                  (green/red/blue/yellow/warm/dark/bright/saturation ...)
                  on the RAW colour basis                        -> 32 ch
  S1  11x11 conv  shape part detectors: long horizontal line (horizon,
                  waterline), thin vertical bars (legs, masts), dark dots
                  (eyes), dark rings (wheels), leg pairs, diagonal bars
                                                                 -> 16 ch
  relu, then pooling to an 8x8 grid (max for detections, mean for colour
  areas), the three branches summed channel-wise (they occupy disjoint
  channel slots), then a 2x2 maxpool to a 4x4 grid  F2 (96,4,4)
  D1  dense       ~105 hand-written units.  Most are "semantic detector"
                  units -- a weighted average of chosen channels over
                  chosen grid cells minus a hand threshold, then relu --
                  plus a set of bias-0 passthrough units whose value is
                  simply a region mean of a colour-category map
  D2  dense       10 class scores: a hand-written linear colour-signature
                  classifier over the passthrough means (blue sky / green
                  field / tan body / dark road / red vehicle ...), plus
                  smaller bonuses from the part detectors (wheels, eyes,
                  legs, horizon, water line) and guards against them.

The 4x4 grid cells are (row, col); row 0 = top quarter of the photo,
col 0 = left quarter.  Channels 0..47 of the merged map are edges,
48..79 colour categories, 80..95 shape parts.
"""

import json
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# kernel formulas
# --------------------------------------------------------------------------

def mesh(k):
    """Integer coordinate grids, X along columns, Y along rows, centred."""
    c = (k - 1) / 2.0
    idx = np.arange(k) - c
    X, Y = np.meshgrid(idx, idx)          # X[r, c] = c - c0, Y[r, c] = r - c0
    return X, Y


def l2n(w):
    """Normalise a kernel to unit L2 norm so responses are O(1)."""
    n = np.sqrt((w ** 2).sum())
    return w / n if n > 0 else w


def gauss(k, sigma, sum_to_one=False):
    G = np.exp(-(mesh(k)[0] ** 2 + mesh(k)[1] ** 2) / (2.0 * sigma ** 2))
    if sum_to_one:
        return G / G.sum()
    return G


def gabor(k, spar, sperp, theta_deg, freq, phase_deg):
    """Gaussian-windowed cos/sin.  spar = sigma along the modulation
    direction theta (edge normal), sperp = sigma along the edge line.
    theta 0 -> modulation along x -> vertical edge / vertical bar profile."""
    X, Y = mesh(k)
    th = np.radians(theta_deg)
    xr = X * np.cos(th) + Y * np.sin(th)
    yr = -X * np.sin(th) + Y * np.cos(th)
    env = np.exp(-(xr ** 2 / spar ** 2 + yr ** 2 / sperp ** 2) / 2.0)
    g = env * np.cos(2.0 * np.pi * freq * xr + np.radians(phase_deg))
    return l2n(g - g.mean())               # remove DC, unit energy


def wheel_kernel(k, r0, r1, r2):
    """Dark ring with lighter hub and surround: a wheel / round eye seen
    from the side.  Negative weights in the tyre annulus (dark pixels there
    give a positive response), positive in the hub and outside the tyre."""
    X, Y = mesh(k)
    r = np.sqrt(X ** 2 + Y ** 2)
    w = np.zeros((k, k))
    w[r < r0] = 0.55                       # hub
    w[(r >= r0) & (r < r1)] = -1.0         # tyre (dark ring)
    w[(r >= r1) & (r < r2)] = 0.45         # just outside the tyre
    return l2n(w - w.mean())               # zero response on flat regions


def dot_kernel(k, sigma_c, sigma_s, w_c, w_s):
    """Centre-surround: w_c * N(0,sigma_c) + w_s * N(0,sigma_s).
    With w_c < 0 < w_s it detects a dark dot on a lighter surround."""
    ker = w_c * gauss(k, sigma_c, sum_to_one=True) + \
        w_s * gauss(k, sigma_s, sum_to_one=True)
    return l2n(ker - ker.mean())           # zero response on flat regions


def twinbar_kernel(k, dx, sigma_x, sigma_y):
    """Two narrow vertical Gaussians at x = +-dx minus a centre bar chosen
    so the total integral along x cancels: the kernel then answers only to
    TWO separated thin bars (a pair of legs), not to a wide slab or to a
    single horizontal edge."""
    X, Y = mesh(k)
    one = np.exp(-((X - dx) ** 2 / sigma_x ** 2 + Y ** 2 / sigma_y ** 2) / 2.0)
    two = np.exp(-((X + dx) ** 2 / sigma_x ** 2 + Y ** 2 / sigma_y ** 2) / 2.0)
    mid = np.exp(-(X ** 2 / (sigma_x * 1.2) ** 2 + Y ** 2 / sigma_y ** 2) / 2.0)
    centre = 2.0 / 1.2      # exact integral match: 2 lobes cancel the centre
    ker = one + two - centre * mid
    return l2n(ker - ker.mean())


# --------------------------------------------------------------------------
# layer 1: fixed colour basis (raw, no normalisation)
# --------------------------------------------------------------------------
# channel: 0 L, 1 R-G, 2 B-(R+G)/2, 3 R-B, 4 R, 5 G, 6 B, 7 (R+G)/2,
#          8 L-0.4, 9 L-0.6, 10 0.45-L, 11 G-B
C1_W = np.zeros((12, 3, 1, 1), dtype=np.float64)
C1_B = np.zeros(12, dtype=np.float64)
C1_W[0, :, 0, 0] = [1 / 3, 1 / 3, 1 / 3]
C1_W[1, :, 0, 0] = [1.0, -1.0, 0.0]
C1_W[2, :, 0, 0] = [-0.5, -0.5, 1.0]
C1_W[3, :, 0, 0] = [1.0, 0.0, -1.0]
C1_W[4, :, 0, 0] = [1.0, 0.0, 0.0]
C1_W[5, :, 0, 0] = [0.0, 1.0, 0.0]
C1_W[6, :, 0, 0] = [0.0, 0.0, 1.0]
C1_W[7, :, 0, 0] = [0.5, 0.5, 0.0]
C1_W[8, :, 0, 0] = [1 / 3, 1 / 3, 1 / 3]; C1_B[8] = -0.40
C1_W[9, :, 0, 0] = [1 / 3, 1 / 3, 1 / 3]; C1_B[9] = -0.60
C1_W[10, :, 0, 0] = [-1 / 3, -1 / 3, -1 / 3]; C1_B[10] = 0.45
C1_W[11, :, 0, 0] = [0.0, 1.0, -1.0]

# input-channel indices of the basis that later kernels read
L, RG, BY, RB, R_, G_, B_, Y_, GB = 0, 1, 2, 3, 4, 5, 6, 7, 11

NCH = 96          # channels in the merged feature map

# --------------------------------------------------------------------------
# E1: Gabor edge bank on the normalised basis (channels 0..47)
# --------------------------------------------------------------------------
E1_W = np.zeros((NCH, 12, 5, 5), dtype=np.float64)
E1_B = np.zeros(NCH, dtype=np.float64)
chmap = {}        # semantic name -> merged-channel index
groups = {}       # semantic group name -> list of merged-channel indices

# (name, input channel, sigma, freq, orientation, phase)
e_specs = []
for ori in (0, 45, 90, 135):
    e_specs.append(("eL1e%d" % ori, L, 1.0, 0.25, ori, 0))     # fine, even
    e_specs.append(("eL1o%d" % ori, L, 1.0, 0.25, ori, 90))    # fine, odd
for ori in (0, 45, 90, 135):
    e_specs.append(("eL2o%d" % ori, L, 1.7, 0.13, ori, 90))    # coarse, odd
for ori in (0, 90):
    e_specs.append(("eL2e%d" % ori, L, 1.7, 0.13, ori, 0))     # coarse, even
for ori in (0, 45, 90, 135):
    e_specs.append(("eRGo%d" % ori, RG, 1.3, 0.22, ori, 90))   # red-green edges
    e_specs.append(("eBYo%d" % ori, BY, 1.3, 0.22, ori, 90))   # blue-yellow edges
for ori in (0, 90):
    e_specs.append(("eGBo%d" % ori, GB, 1.3, 0.22, ori, 90))   # green-blue edges

assert len(e_specs) * 2 == 48
groups.setdefault("EALL", [])
for j, (nm, ich, sig, f, ori, ph) in enumerate(e_specs):
    ker = gabor(5, sig, sig, ori, f, ph)
    E1_W[2 * j, ich] = ker          # +kernel
    E1_W[2 * j + 1, ich] = -ker     # -kernel  (relu of both = |response|)
    chmap[nm] = 2 * j
    chmap[nm + "n"] = 2 * j + 1
    both = [2 * j, 2 * j + 1]
    groups["EALL"] += both
    # orientation groups (theta is the edge normal): 0 = vertical edge line,
    # 90 = horizontal edge line, 45/135 = diagonals
    gname = {0: "EV", 90: "EH", 45: "ED1", 135: "ED2"}[ori]
    groups.setdefault(gname, [])
    groups[gname] += both
    if nm.startswith("eL1"):                       # fine scale, luminance
        groups.setdefault("EFINE", [])
        groups["EFINE"] += both
    if nm.startswith("eL2"):                       # coarse scale, luminance
        groups.setdefault("ECOARSE", [])
        groups["ECOARSE"] += both
    if nm[:3] in ("eRG", "eBY", "eGB"):            # colour-opponent edges
        groups.setdefault("ECOL", [])
        groups["ECOL"] += both
groups["ED"] = groups["ED1"] + groups["ED2"]


# --------------------------------------------------------------------------
# G1: smoothed colour-category maps on the RAW basis (channels 48..79)
# --------------------------------------------------------------------------
# Each kernel is a sum-to-one Gaussian (sigma 1.5) applied to a linear
# combination of basis channels; the bias is a hand-chosen threshold on
# the 0..1 colour scale, relu keeps what is above it.
G1_W = np.zeros((NCH, 12, 5, 5), dtype=np.float64)
G1_B = np.zeros(NCH, dtype=np.float64)
GSM = gauss(5, 1.5, sum_to_one=True)

# (name, {input channel: weight}, bias)
g_specs = [
    ("greenStrong", {G_: 2, R_: -1, B_: -1}, -0.12),   # 2G-R-B > .12
    ("greenMild",   {G_: 2, R_: -1, B_: -1}, -0.04),
    ("redStrong",   {R_: 2, G_: -1, B_: -1}, -0.16),
    ("warm",        {R_: 1, B_: -1}, -0.06),           # R-B > .06: tan/brown/red
    ("blueStrong",  {B_: 2, R_: -1, G_: -1}, -0.16),
    ("blueMild",    {B_: 2, R_: -1, G_: -1}, -0.05),
    ("yellow",      {R_: 1, G_: 1, B_: -2}, -0.18),    # sand / dry grass
    ("satRGp",      {R_: 1, G_: -1}, -0.06),           # saturation halves
    ("satRGn",      {R_: -1, G_: 1}, -0.06),
    ("satBYp",      {BY: 1}, -0.06),
    ("satBYn",      {BY: -1}, -0.06),
    ("bright",      {L: 1}, -0.55),
    ("brightStrong",{L: 1}, -0.65),
    ("dark",        {L: -1}, -0.42),                   # relu(0.42 - L)
    ("darkStrong",  {L: -1}, -0.30),
    ("redArea",     {R_: 1}, -0.55),
    ("greenArea",   {G_: 1}, -0.55),
    ("blueArea",    {B_: 1}, -0.55),
    ("yellowArea",  {Y_: 1}, -0.45),
    ("gbPos",       {GB: 1}, -0.06),                   # G-B: foliage vs sky/sea
    ("gbNeg",       {GB: -1}, -0.06),
    ("darkGreen",   {G_: 2, R_: -1, B_: -1, L: -0.5}, -0.10),
    ("tanDry",      {R_: 1.3, G_: 0.7, B_: -2}, -0.20),
    ("warmBright",  {R_: 1, B_: -1, L: 1}, -0.75),     # bright sand / haze
    ("coolBright",  {B_: 1, R_: -0.5, G_: -0.5, L: 1}, -0.75),
    ("brownFur",    {R_: 1, G_: -0.3, B_: -0.7}, -0.06),
    # tan / brown body colour: red clearly above green (tan, dry grass,
    # brown fur) -- olive greens and pure greens give ~0
    ("tanBody",     {R_: 3, G_: -3, B_: -0.5}, -0.06),
]
assert 48 + len(g_specs) <= 80
for j, (nm, wts, bias) in enumerate(g_specs):
    c = 48 + j
    for ich, w in wts.items():
        G1_W[c, ich] = w * GSM
    G1_B[c] = bias
    chmap[nm] = c

# absolute texture-energy channels: centre-surround (DoG) filters on the
# raw luminance, kept in + and - copies so relu+mean gives mean |contrast|
# at that scale.  Unlike the z-scored edge bank these keep the ABSOLUTE
# level, which separates fur / grass / foliage (busy) from sky, water,
# hulls and road (smooth).
def _dog(sig_c, sig_s):
    return gauss(5, sig_s, sum_to_one=True) - gauss(5, sig_c, sum_to_one=True)

for nm, ker in [("lapP", _dog(0.8, 2.0)), ("lapM", -_dog(0.8, 2.0)),
                ("lap2P", _dog(1.2, 2.6)), ("lap2M", -_dog(1.2, 2.6))]:
    c = 48 + len(g_specs)
    G1_W[c, L] = ker
    G1_B[c] = 0.0
    g_specs.append((nm, None, 0.0))
    chmap[nm] = c

# --------------------------------------------------------------------------
# S1: shape part detectors on the normalised luminance (channels 80..95)
# --------------------------------------------------------------------------
S1_W = np.zeros((NCH, 12, 11, 11), dtype=np.float64)
S1_B = np.zeros(NCH, dtype=np.float64)

# (name, kernel, [negated partner name or None])
s_specs = [
    ("hlineP",  gabor(11, 1.2, 4.5, 90, 0.09, 0),  "hlineM"),  # long horiz line
    ("vbarP",   gabor(11, 1.0, 4.0, 0, 0.09, 0),   "vbarM"),   # thin vert bar
    ("dotDark", dot_kernel(11, 0.9, 2.4, -2.2, 1.0), "dotBright"),
    ("ringD25", wheel_kernel(11, 1.6, 3.2, 5.2),  None),       # car/bug wheel
    ("ringD45", wheel_kernel(11, 2.8, 4.7, 5.5),  None),       # truck wheel
    ("hbarP",   gabor(11, 1.0, 3.0, 90, 0.09, 0),  "hbarM"),   # short horiz bar
    ("pairVP",  twinbar_kernel(11, 2.6, 1.0, 3.2), "pairVM"),
    ("diag45P", gabor(11, 1.0, 3.5, 45, 0.09, 0),  "diag45M"),
    ("diag135P", gabor(11, 1.0, 3.5, 135, 0.09, 0), "diag135M"),
]
assert 80 + sum(2 if partner else 1 for _, _, partner in s_specs) <= 96
c = 80
for nm, ker, partner in s_specs:
    S1_W[c, L] = ker
    chmap[nm] = c
    c += 1
    if partner:
        S1_W[c, L] = -ker
        chmap[partner] = c
        c += 1
groups["SALL"] = [v for v in chmap.values() if 80 <= v < 96]


# --------------------------------------------------------------------------
# grid cells of the 7x7 pooled map (row 0 = top third of the photo)
# --------------------------------------------------------------------------
def _cells(rows, cols):
    return [(r, q) for r in rows for q in cols]

ALL49 = _cells(range(7), range(7))
TOP   = _cells([0, 1, 2], range(7))            # top third
BOT   = _cells([4, 5, 6], range(7))            # bottom third
MIDR  = _cells([2, 3, 4], range(7))            # middle band
CEN   = _cells([2, 3, 4], [2, 3, 4])           # central ~12x12 px
TOPCEN = _cells([1, 2, 3], [2, 3, 4])          # upper centre: eyes / ears
BOTCEN = _cells([4, 5, 6], [2, 3, 4])
BOTL  = _cells([5, 6], [0, 1, 2])             # lower-left: wheel zone
BOTR  = _cells([5, 6], [4, 5, 6])             # lower-right: wheel zone
ROWBOT = _cells([6], range(7))                 # bottom row
ROWBOTC = _cells([6], [1, 2, 3, 4, 5])         # bottom centre: legs
BORDER = [c for c in ALL49 if c not in CEN]

# --------------------------------------------------------------------------
# D1: hand-written semantic detector units.
# (name, [(channel-or-group, cell set, total weight), ...], bias threshold)
# Colour-category channels are area means (avgpool) of maps in [0, 0.6];
# a solidly present category averages ~0.2-0.5, so those contributions use
# weights ~3-6.  Edge and shape channels are block maxima of |kernel
# response| on the RAW contrast scale: solid detections ~0.15-0.4 per
# block, noise ~0.05, so their region means carry weights ~6-12.  The bias
# is minus the hand-chosen threshold.
# --------------------------------------------------------------------------
UNITS = [
    # ---- global colour statistics (raw colour-area means) ----
    ("gGreen",   [("greenMild", ALL49, 3.0)], -0.25),
    ("gGreenStr",[("greenStrong", ALL49, 3.0)], -0.35),
    ("gRed",     [("redStrong", ALL49, 3.0)], -0.30),
    ("gWarm",    [("warm", ALL49, 3.0)], -0.25),
    ("gBlue",    [("blueMild", ALL49, 3.0)], -0.25),
    ("gBlueStr", [("blueStrong", ALL49, 3.0)], -0.35),
    ("gYellow",  [("yellow", ALL49, 3.0)], -0.35),
    ("gSat",     [("satRGp", ALL49, .75), ("satRGn", ALL49, .75),
                  ("satBYp", ALL49, .75), ("satBYn", ALL49, .75)], -0.10),
    ("gGray",    [("satRGp", ALL49, -.75), ("satRGn", ALL49, -.75),
                  ("satBYp", ALL49, -.75), ("satBYn", ALL49, -.75)], 0.08),
    ("gBright",  [("bright", ALL49, 3.0)], -0.28),
    ("gBrightStr",[("brightStrong", ALL49, 3.0)], -0.18),
    ("gDark",    [("dark", ALL49, 3.0)], -0.25),
    ("gDarkStr", [("darkStrong", ALL49, 3.0)], -0.30),
    ("gGB",      [("gbPos", ALL49, 3.0)], -0.21),
    ("gBG",      [("gbNeg", ALL49, 3.0)], -0.21),
    # ---- colour by layout ----
    ("skyBlueTop",   [("blueMild", TOP, 3.0)], -0.30),
    ("waterBlueBot", [("blueMild", BOT, 3.0)], -0.30),
    ("allBlue",      [("blueMild", ALL49, 3.6)], -0.60),
    ("waterBot",     [("blueMild", BOT, 6.0), ("bright", BOT, -6.0)], -1.30),
    ("greenTop",     [("greenMild", TOP, 3.0)], -0.30),
    ("greenBot",     [("greenMild", BOT, 3.0)], -0.30),
    ("greenAll",     [("greenMild", ALL49, 3.0)], -0.45),
    ("warmAll",      [("warm", ALL49, 3.0)], -0.30),
    ("brightTop",    [("bright", TOP, 3.0)], -0.26),
    ("darkBot",      [("dark", BOT, 3.0)], -0.28),
    ("darkTop",      [("dark", TOP, 3.0)], -0.28),
    ("cWarm",   [("warm", CEN, 3.0)], -0.36),
    ("cGreen",  [("greenMild", CEN, 3.0)], -0.36),
    ("cBlue",   [("blueMild", CEN, 3.0)], -0.36),
    ("cRed",    [("redStrong", CEN, 3.0)], -0.42),
    ("cYellow", [("yellow", CEN, 3.0)], -0.42),
    ("cDark",   [("dark", CEN, 3.0)], -0.30),
    ("cDarkStr",[("darkStrong", CEN, 3.0)], -0.32),
    ("cBright", [("bright", CEN, 3.0)], -0.30),
    ("cNeutral",[("bright", CEN, 2.5), ("satRGp", CEN, -.75), ("satRGn", CEN, -.75),
                 ("satBYp", CEN, -.75), ("satBYn", CEN, -.75)], 0.20),
    # ---- edge energy LOCALISATION (z-scored detections: a unit fires when
    #      that kind of structure is concentrated in the named region of
    #      THIS photo, relative to the photo's own level) ----
    ("eHmid",  [("EH", MIDR, 1.5)], -0.45),
    ("eVbot",  [("EV", BOT, 1.5)], -0.50),
    ("eVtop",  [("EV", TOP, 1.5)], -0.55),
    ("eBusyCen",  [("EALL", CEN, 1.5)], -0.50),
    ("busyObj",   [("EALL", CEN, 1.2), ("EALL", BORDER, -0.8)], -0.30),
    ("fineCen",   [("EFINE", CEN, 1.5)], -0.55),
    # ---- shape parts (z-scored: part far stronger than elsewhere) ----
    ("hlineMid", [("hlineP", MIDR, 1.0), ("hlineM", MIDR, 1.0)], -0.55),
    ("hlineTop", [("hlineP", TOP, 1.0), ("hlineM", TOP, 1.0)], -0.60),
    ("hbarMid",  [("hbarP", MIDR, 1.0), ("hbarM", MIDR, 1.0)], -0.55),
    ("vbarBot",  [("vbarP", BOT, 1.0), ("vbarM", BOT, 1.0)], -0.55),
    ("vbarBotCen",[("vbarP", BOTCEN, 1.2), ("vbarM", BOTCEN, 1.2)], -0.70),
    ("vbarTopCen",[("vbarP", TOPCEN, 1.2), ("vbarM", TOPCEN, 1.2)], -0.75),
    ("pairVBot", [("pairVP", BOT, 1.2), ("pairVM", BOT, 1.2)], -0.75),
    ("pairVRow3",[("pairVP", ROWBOTC, 1.5), ("pairVM", ROWBOTC, 1.5)], -0.80),
    ("dotDarkTopCen", [("dotDark", TOPCEN, 2.0)], -0.60),
    ("dotDarkCen",    [("dotDark", CEN, 2.0)], -0.65),
    ("ringBot",  [("ringD25", BOT, 1.2)], -0.25),
    ("ringBotL", [("ringD25", BOTL, 1.5)], -0.30),
    ("ringBotR", [("ringD25", BOTR, 1.5)], -0.30),
    ("twoRings", [("ringD25", BOTL, 1.2), ("ringD25", BOTR, 1.2)], -0.60),
    ("ringBig",  [("ringD45", BOTL, 1.2), ("ringD45", BOTR, 1.2)], -0.55),
    ("diagTop",  [("diag45P", TOP, 0.6), ("diag45M", TOP, 0.6),
                  ("diag135P", TOP, 0.6), ("diag135M", TOP, 0.6)], -0.50),
    # ---- scene / part conjunctions ----
    ("skyOverGreen", [("blueMild", TOP, 4.0), ("greenMild", BOT, 4.0)], -1.40),
    ("blueBrightTop",[("blueMild", TOP, 4.0), ("bright", TOP, 3.0)], -1.50),
    ("greenScene",   [("greenMild", ALL49, 3.5), ("greenStrong", ALL49, 1.5)], -0.30),
    ("warmOnGreen",  [("warm", CEN, 5.0), ("greenMild", BOT, 5.0)], -1.50),
    ("darkOnGreen",  [("dark", CEN, 6.0), ("greenMild", ALL49, 4.0)], -0.90),
    ("greenOnGreen", [("greenMild", CEN, 4.0), ("greenMild", ALL49, 3.0)], -0.90),
    ("brightOnBlue", [("bright", MIDR, 5.0), ("blueMild", ALL49, 3.5)], -1.20),
    ("darkOnBlue",   [("dark", CEN, 5.0), ("darkStrong", CEN, 3.0), ("blueMild", ALL49, 3.5)], -1.20),
    ("grayDarkBot",  [("dark", BOT, 8.0), ("darkStrong", BOT, 4.0)], -1.20),
    ("redDarkBot",   [("redStrong", CEN, 5.0), ("dark", BOT, 4.0)], -2.20),
    ("leggedGreen",  [("pairVP", ROWBOTC, 1.0), ("pairVM", ROWBOTC, 1.0),
                      ("vbarP", ROWBOTC, 0.6), ("vbarM", ROWBOTC, 0.6),
                      ("greenMild", ALL49, 3.0)], -1.70),
    ("faceLike",     [("dotDark", TOPCEN, 2.5), ("dotDark", CEN, 1.5),
                      ("EFINE", CEN, 1.5), ("warm", CEN, 6.0)], -3.30),
    ("horizon",      [("EH", MIDR, 1.2), ("hlineP", MIDR, 1.0), ("hlineM", MIDR, 1.0)], -0.90),
    ("mast",         [("vbarP", TOPCEN, 1.5), ("vbarM", TOPCEN, 1.5)], -0.85),
    ("wideWarm",     [("warm", MIDR, 6.0), ("brownFur", MIDR, 4.0)], -1.00),
    ("wideBright",   [("bright", MIDR, 5.0), ("brightStrong", MIDR, 3.0)], -0.70),
    ("bigSmooth",    [("EALL", BORDER, 1.2), ("EALL", CEN, -0.8)], -0.35),
    # ---- absolute texture level (fur / grass vs sky / water / metal) ----
    ("texFine",  [("lapP", ALL49, 4.0), ("lapM", ALL49, 4.0)], -0.60),
    ("texCen",   [("lapP", CEN, 4.0), ("lapM", CEN, 4.0)], -0.60),
    ("texMid",   [("lap2P", ALL49, 3.0), ("lap2M", ALL49, 3.0)], -0.45),
    ("faceFur",  [("dotDark", TOPCEN, 1.5), ("dotDark", CEN, 0.8),
                  ("lapP", CEN, 4.0), ("lapM", CEN, 4.0)], -1.60),
    # ---- linear colour-signature core: plain region means (bias 0, so the
    #      unit value IS the mean) of the colour-category maps.  These are
    #      on the physical 0..1 colour scale and behave the same on every
    #      photo, so the class scores below are essentially a hand-written
    #      linear colour classifier, with the detectors above as bonuses ----
    ("pGreenAll", [("greenMild", ALL49, 3.0)], 0.0),
    ("pGreenTop", [("greenMild", TOP, 3.0)], 0.0),
    ("pGreenBot", [("greenMild", BOT, 3.0)], 0.0),
    ("pGreenCen", [("greenMild", CEN, 3.0)], 0.0),
    ("pBlueAll",  [("blueMild", ALL49, 3.0)], 0.0),
    ("pBlueTop",  [("blueMild", TOP, 3.0)], 0.0),
    ("pBlueBot",  [("blueMild", BOT, 3.0)], 0.0),
    ("pWarmAll",  [("warm", ALL49, 4.0)], 0.0),
    ("pWarmCen",  [("warm", CEN, 4.0)], 0.0),
    ("pRedAll",   [("redStrong", ALL49, 3.0)], 0.0),
    ("pRedCen",   [("redStrong", CEN, 3.0)], 0.0),
    ("pYellowAll",[("yellow", ALL49, 3.0)], 0.0),
    ("pYellowCen",[("yellow", CEN, 3.0)], 0.0),
    ("pYellowBot",[("yellow", BOT, 3.0)], 0.0),
    ("pDarkAll",  [("dark", ALL49, 4.0)], 0.0),
    ("pDarkBot",  [("dark", BOT, 4.0)], 0.0),
    ("pDarkCen",  [("dark", CEN, 4.0)], 0.0),
    ("pBrightAll",[("bright", ALL49, 4.0)], 0.0),
    ("pBrightTop",[("bright", TOP, 4.0)], 0.0),
    ("pBrightCen",[("bright", CEN, 4.0)], 0.0),
    ("pSatAll",   [("satRGp", ALL49, .75), ("satRGn", ALL49, .75),
                   ("satBYp", ALL49, .75), ("satBYn", ALL49, .75)], 0.0),
    ("pGBAll",    [("gbPos", ALL49, 2.0)], 0.0),
    ("pBGAll",    [("gbNeg", ALL49, 2.0)], 0.0),
    ("pNeutralCen",[("bright", CEN, 4.0), ("satRGp", CEN, -.75), ("satRGn", CEN, -.75),
                    ("satBYp", CEN, -.75), ("satBYn", CEN, -.75)], 0.0),
    ("pTanAll",  [("tanBody", ALL49, 3.0)], 0.0),
    ("pTanCen",  [("tanBody", CEN, 3.0)], 0.0),
    ("pTanBot",  [("tanBody", BOT, 3.0)], 0.0),
]
UNIT_IDX = {u[0]: i for i, u in enumerate(UNITS)}
H = len(UNITS)


# --------------------------------------------------------------------------
# D2: hand-written class evidence.  Positive weight = the detector is
# evidence for the class, negative = evidence against.
# --------------------------------------------------------------------------
CLASS_EVIDENCE = {
 # Hand-written colour-signature classifier over region means of the
 # colour-category maps (p* units), plus smaller part-detector bonuses
 # (wheels, eyes, legs, horizon, water line).
 "airplane": [("pBlueTop", 2.2), ("pBlueAll", 1.2), ("pBrightTop", 0.8),
   ("skyBlueTop", 1.0), ("fineCen", 0.6), ("pBGAll", 0.4),
   ("pNeutralCen", 0.3),
   ("twoRings", -0.5), ("pGreenAll", -1.2), ("pGreenBot", -0.8),
   ("pWarmCen", -0.5), ("horizon", -0.4), ("pDarkBot", -0.3),
   ("pBlueBot", -0.6)],
 "automobile": [("pDarkBot", 1.8), ("twoRings", 1.2), ("ringBot", 0.9),
   ("grayDarkBot", 0.8), ("pRedCen", 0.7), ("pDarkAll", 0.5),
   ("pNeutralCen", 0.4), ("pSatAll", 0.4),
   ("pGreenAll", -1.0), ("pGreenBot", -0.8), ("pBlueTop", -1.0),
   ("pBlueAll", -0.7), ("vbarTopCen", -0.3)],
 "bird": [("pBlueTop", 0.8), ("pGreenTop", 0.8), ("dotDarkTopCen", 1.2),
   ("fineCen", 0.6), ("pSatAll", 0.8), ("pDarkCen", 0.7),
   ("texFine", 0.6), ("cGreen", 0.4),
   ("twoRings", -0.6), ("pDarkBot", -0.5), ("wideBright", -0.4),
   ("pNeutralCen", -0.3), ("pTanAll", -0.3)],
 "cat": [("pSatAll", 1.2), ("pWarmCen", 1.0), ("texFine", 1.3),
   ("faceFur", 1.2), ("pNeutralCen", 0.4), ("dotDarkCen", 0.5),
   ("pGreenAll", -1.0), ("pBlueTop", -0.9), ("pBlueAll", -0.7),
   ("pGreenBot", -0.7), ("twoRings", -0.5)],
 "deer": [("pGreenAll", 1.3), ("pGreenBot", 1.0), ("pGreenTop", 0.7),
   ("pTanAll", 0.7), ("pTanCen", 0.8), ("pWarmCen", 0.5),
   ("pGBAll", 0.5), ("skyOverGreen", 0.5), ("pairVRow3", 0.4),
   ("dotDarkTopCen", 0.3),
   ("pBlueTop", -0.9), ("pBlueAll", -0.8), ("pRedCen", -0.4),
   ("twoRings", -0.7), ("pSatAll", -0.3)],
 "dog": [("texFine", 1.3), ("faceFur", 1.3), ("pDarkCen", 1.1),
   ("pWarmCen", 0.9), ("pSatAll", 0.6), ("dotDarkCen", 0.5),
   ("pGreenAll", -1.0), ("pBlueTop", -0.9), ("pGreenBot", -0.7),
   ("twoRings", -0.5), ("pBlueAll", -0.6)],
 "frog": [("pGreenCen", 1.5), ("pGreenAll", 1.3), ("greenScene", 0.9),
   ("pGBAll", 0.8), ("pGreenBot", 0.7), ("greenOnGreen", 0.6),
   ("darkOnGreen", 0.4), ("pSatAll", 0.3), ("dotDarkTopCen", 0.3),
   ("pBlueTop", -1.0), ("pBlueAll", -0.9), ("pWarmAll", -0.6),
   ("pTanCen", -0.6), ("pBGAll", -0.5), ("twoRings", -0.5)],
 "horse": [("pGreenAll", 1.3), ("pGreenBot", 1.1), ("pWarmCen", 0.8),
   ("pDarkCen", 1.0), ("warmOnGreen", 0.8), ("pTanCen", 0.5),
   ("pTanBot", 0.6), ("pGBAll", 0.5), ("pairVRow3", 0.4),
   ("pDarkBot", 0.4),
   ("pBlueTop", -0.9), ("pBlueAll", -0.8), ("pRedCen", -0.5),
   ("pGreenCen", -0.6), ("twoRings", -0.7), ("pSatAll", -0.2)],
 "ship": [("pBlueBot", 1.8), ("horizon", 1.2), ("wideBright", 0.9),
   ("pBlueAll", 1.0), ("pBlueTop", 0.5), ("pBrightCen", 0.8),
   ("pNeutralCen", 0.5), ("hlineMid", 0.5), ("allBlue", 0.5),
   ("pBGAll", 0.4), ("fineCen", 0.4), ("twoRings", 0.4),
   ("pGreenAll", -1.2), ("pGreenBot", -1.0), ("pGBAll", -0.6),
   ("pWarmCen", -0.5)],
 "truck": [("pDarkBot", 1.3), ("twoRings", 1.1), ("pRedCen", 0.7),
   ("ringBig", 0.6), ("grayDarkBot", 0.9), ("pRedAll", 0.4),
   ("pDarkAll", 0.6), ("pNeutralCen", 0.5), ("vbarTopCen", 0.6),
   ("wideWarm", 0.4),
   ("pGreenAll", -1.0), ("pGreenBot", -0.9), ("pBlueTop", -0.9),
   ("pBlueAll", -0.6), ("faceFur", -0.4)],
}
CLASS_BIAS = {c: 0.0 for c in CLASS_EVIDENCE}

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# --------------------------------------------------------------------------
# assemble dense weights
# --------------------------------------------------------------------------
GRID = 7
DENSE_IN = NCH * GRID * GRID   # 96 channels x 7x7 grid


def build_d1():
    W = np.zeros((H, DENSE_IN), dtype=np.float64)
    b = np.zeros(H, dtype=np.float64)
    for u, (name, contribs, bias) in enumerate(UNITS):
        for key, cells, w in contribs:
            chans = chmap[key] if key in chmap else groups[key]
            if isinstance(chans, int):
                chans = [chans]
            per = w / len(chans)
            for cidx in chans:
                for (r, q) in cells:
                    W[u, cidx * GRID * GRID + r * GRID + q] += per / len(cells)
        b[u] = bias
    return W, b


def build_d2():
    """Class scores from hand-written evidence: a linear colour-signature
    core plus small part-detector bonuses."""
    W = np.zeros((10, H), dtype=np.float64)
    b = np.zeros(10, dtype=np.float64)
    for k, cls in enumerate(CLASSES):
        for uname, w in CLASS_EVIDENCE[cls]:
            W[k, UNIT_IDX[uname]] += w
        b[k] = CLASS_BIAS[cls]
    return W, b


def network_json():
    return {"layers": [
        {"name": "C1", "op": "conv", "inputs": ["image"],
         "out": 12, "kernel": [1, 1], "stride": 1, "padding": 0},
        {"name": "E1", "op": "conv", "inputs": ["C1"],
         "out": NCH, "kernel": [5, 5], "stride": 1, "padding": 0},
        {"name": "G1", "op": "conv", "inputs": ["C1"],
         "out": NCH, "kernel": [5, 5], "stride": 1, "padding": 0},
        {"name": "S1", "op": "conv", "inputs": ["C1"],
         "out": NCH, "kernel": [11, 11], "stride": 1, "padding": 0},
        {"name": "E2", "op": "relu", "inputs": ["E1"]},
        {"name": "G2", "op": "relu", "inputs": ["G1"]},
        {"name": "S2", "op": "relu", "inputs": ["S1"]},
        {"name": "EP", "op": "maxpool", "inputs": ["E2"], "size": [4, 4], "stride": [4, 4]},
        {"name": "GP", "op": "avgpool", "inputs": ["G2"], "size": [4, 4], "stride": [4, 4]},
        {"name": "SP", "op": "maxpool", "inputs": ["S2"], "size": [3, 3], "stride": [3, 3]},
        {"name": "NEP", "op": "norm", "inputs": ["EP"]},
        {"name": "NSP", "op": "norm", "inputs": ["SP"]},
        {"name": "F1", "op": "add", "inputs": ["NEP", "GP", "NSP"]},
        {"name": "D1", "op": "dense", "inputs": ["F1"], "out": H},
        {"name": "R1", "op": "relu", "inputs": ["D1"]},
        {"name": "D2", "op": "dense", "inputs": ["R1"], "out": 10},
    ]}


def build_weights():
    d1W, d1b = build_d1()
    d2W, d2b = build_d2()
    return {
        "C1.W": C1_W.astype(np.float32), "C1.b": C1_B.astype(np.float32),
        "E1.W": E1_W.astype(np.float32), "E1.b": E1_B.astype(np.float32),
        "G1.W": G1_W.astype(np.float32), "G1.b": G1_B.astype(np.float32),
        "S1.W": S1_W.astype(np.float32), "S1.b": S1_B.astype(np.float32),
        "D1.W": d1W.astype(np.float32), "D1.b": d1b.astype(np.float32),
        "D2.W": d2W.astype(np.float32), "D2.b": d2b.astype(np.float32),
    }


def main(dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    w = build_weights()
    np.savez(dest / "weights.npz", **w)
    (dest / "network.json").write_text(json.dumps(network_json()))
    total = sum(int(np.prod(v.shape)) for v in w.values())
    print("wrote %s: %d weights, hidden units %d" % (dest, total, H))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
