"""Builds the weights for a hand-designed CIFAR-10 classifier network.

Everything comes from formulas (Gabor filters, differences of Gaussians,
colour-opponent combinations with hand-set thresholds) and hand-written rule
tables that encode general knowledge about what the ten classes look like:
sky around planes, deep-blue water at the bottom of ship scenes, grass
around deer/horses, very green close-ups for frogs, warm indoor palettes
for cats, dark wheels and a shadow strip under vehicles, tall boxes with
long vertical edges for trucks, tan/brown bodies (mildly warm, not vivid
red) for animals.
Edge and blob maps are spatially normalised (norm layers) so that they
measure where structure is unusually strong FOR THIS PICTURE, not how busy
the picture is in absolute terms.
No training data is read; nothing is fitted; there is no randomness.

Usage: python3 build.py OUT_DIR   ->  OUT_DIR/network.json, OUT_DIR/weights.npz
"""
import json
import os
import sys

import numpy as np

K = 10  # classes: airplane automobile bird cat deer dog frog horse ship truck
AIR, AUTO, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK = range(K)


# ----------------------------------------------------------------------------
# Filter formulas (zero-mean where appropriate, L2-normalised so different
# filters give comparable response magnitudes).
# ----------------------------------------------------------------------------
def grid(k):
    """(y, x) offsets of a k x k kernel centred at 0."""
    c = k // 2
    ys, xs = np.mgrid[0:k, 0:k]
    return (ys - c).astype(float), (xs - c).astype(float)


def zml2(kern):
    kern = kern - kern.mean()
    n = np.sqrt((kern ** 2).sum())
    return kern / n if n > 0 else kern


def gabor(k, sac, sal, lam, theta, phase):
    """Real Gabor filter, elongated ALONG the edge line it detects.
    sac: sigma across the edge, sal: sigma along it.
    theta=0 -> stripes vary along x -> detects vertical edge lines."""
    yy, xx = grid(k)
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    env = np.exp(-(xr ** 2 / (2.0 * sac ** 2) + yr ** 2 / (2.0 * sal ** 2)))
    if phase == "even":
        ph = np.cos(2 * np.pi * xr / lam)
    else:
        ph = np.sin(2 * np.pi * xr / lam)
    return zml2(env * ph)


def gauss(k, sigma):
    yy, xx = grid(k)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def dog(k, sc, ss, sign=1.0):
    """Difference of Gaussians: positive when the centre is brighter than the
    surroundings (sign=-1: dark centre).  Unlike -LoG this does not fire on
    the dark ring around a bright blob."""
    return sign * (gauss(k, sc) - gauss(k, ss))


def d2gy(k, sx, sy, sign=1.0):
    """Elongated second derivative of a Gaussian along y: horizontal edge /
    stripe detector (water/sky horizon)."""
    yy, xx = grid(k)
    g = np.exp(-(xx ** 2 / (2.0 * sx ** 2) + yy ** 2 / (2.0 * sy ** 2)))
    return sign * zml2(g * (yy ** 2 / sy ** 4 - 1.0 / sy ** 2))


def on_lum(kern):
    """A 1-channel spatial kernel applied to luminance (R+G+B)/3 -> 3-channel."""
    return np.stack([kern / 3.0] * 3, axis=0)


ORIS = [np.deg2rad(a) for a in (0.0, 45.0, 90.0, 135.0)]  # 0: vertical edges


# ----------------------------------------------------------------------------
# Colour branch: opponent maps -> thresholded colour words.
# cA out (5): redness R-G, greenness G-R, blueness B-(R+G)/2,
#              yellowness (R+G)/2-B, luminance.
# cB out (11), per pixel:
#   0 red        1 green (scaled)   2 blue          3 yellow
#   4 warm hue   5 grey (scaled)    6 dark          7 bright (scaled)
#   8 mild warm = tan/brown band    9 strong warm = vivid red/orange band
#  10 deep blue (bluish and dark: sea water, not sky, not asphalt)
# ----------------------------------------------------------------------------
def build_colour():
    W = np.zeros((5, 3, 3, 3))
    W[0, :, 1, 1] = [1.0, -1.0, 0.0]
    W[1, :, 1, 1] = [-1.0, 1.0, 0.0]
    W[2, :, 1, 1] = [-0.5, -0.5, 1.0]
    W[3, :, 1, 1] = [0.5, 0.5, -1.0]
    W[4, :, 1, 1] = 1.0 / 3.0
    b = np.zeros(5)

    WB = np.zeros((11, 5, 3, 3))
    bB = np.zeros(11)
    rows = [
        ([1, 0, 0, 0, 0], -0.05),          # 0 red
        ([0, 2, 0, 0, 0], -0.10),          # 1 green (scaled: grass words are small)
        ([0, 0, 1, 0, 0], -0.05),          # 2 blue
        ([0, 0, 0, 1, 0], -0.04),          # 3 yellow
        ([1, 0, 0, 1, 0], -0.12),          # 4 warm hue (red/yellow; not sky or grass)
        ([-2, -2, -2, -2, 0], 0.24),       # 5 greyish (scaled)
        ([0, 0, 0, 0, -1], 0.32),          # 6 dark
        ([0, 0, 0, 0, 1.5], -0.90),        # 7 bright (scaled: sky is only just bright)
        ([1.6, 0, 0, 0.8, 0], -0.05),      # 8 mildly warm, red-dominant (tan/brown)
        ([1.6, 0, 0, 0.8, 0], -0.45),      # 9 strongly warm (same input, higher threshold)
        ([0, 0, 2.2, 0, -1.6], 0.30),      # 10 deep blue
    ]
    for o, (w, bb) in enumerate(rows):
        WB[o, :, 1, 1] = w
        bB[o] = bb
    return W, b, WB, bB


# ----------------------------------------------------------------------------
# Edge / blob filter banks.
# cE (k=5, 16ch): quadrature Gabor pairs at 4 orientations (texture energy).
# cE2 (k=11, 12ch): 2t/2t+1 +/- even long-straight-edge Gabor, 8/9 +/- horizon,
#   10 dark centre, 11 bright centre (DoGs).
# cW (k=9, 4ch): 0 small dark centre, 1 medium dark centre (wheels), 2 darkness,
#   3 small bright centre.
# ----------------------------------------------------------------------------
def build_edges():
    WE = np.zeros((16, 3, 5, 5))
    for t, th in enumerate(ORIS):
        for j, ph in enumerate(("even", "odd")):
            g = gabor(5, 1.3, 1.3, 3.6, th, ph)   # isotropic: texture energy
            WE[4 * t + 2 * j, :, :, :] = on_lum(g)
            WE[4 * t + 2 * j + 1, :, :, :] = on_lum(-g)
    # threshold: keeps real edges, kills sensor noise so that the norm layer
    # below does not amplify noise to unit variance on smooth pictures
    bE = np.zeros(16) - 0.08

    WE2 = np.zeros((12, 3, 11, 11))
    for t, th in enumerate(ORIS):
        g = gabor(11, 1.6, 4.0, 5.5, th, "even")   # long straight edge lines
        WE2[2 * t, :, :, :] = on_lum(g)
        WE2[2 * t + 1, :, :, :] = on_lum(-g)
    WE2[8, :, :, :] = on_lum(d2gy(11, 4.5, 1.1, +1.0))
    WE2[9, :, :, :] = on_lum(d2gy(11, 4.5, 1.1, -1.0))
    WE2[10, :, :, :] = on_lum(dog(11, 2.0, 5.0, -3.0))
    WE2[11, :, :, :] = on_lum(dog(11, 2.0, 5.0, +3.0))
    bE2 = np.zeros(12) - 0.05
    bE2[10] = -0.09
    bE2[11] = -0.09

    WW = np.zeros((4, 3, 9, 9))
    WW[0, :, :, :] = on_lum(dog(9, 0.8, 2.6, -3.0))
    WW[1, :, :, :] = on_lum(dog(9, 1.4, 4.0, -3.0))
    WW[2, :, :, :] = on_lum(-gauss(9, 1.2))
    WW[3, :, :, :] = on_lum(dog(9, 0.8, 2.6, +3.0))
    bW = np.zeros(4)
    bW[0] = -0.12
    bW[1] = -0.12
    bW[2] = 0.35
    bW[3] = -0.12
    return WE, bE, WE2, bE2, WW, bW


# ----------------------------------------------------------------------------
# Centre-surround convs: on the colour grid (which colour word stands out in
# the middle of the picture) and on the edge grids (edges INSIDE the object
# vs edges only around its boundary ring).
# ----------------------------------------------------------------------------
def build_cs():
    W = np.zeros((11, 11, 3, 3))
    for o in range(11):
        W[o, o, 1, 1] = 1.0
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                W[o, o, dy, dx] = -1.0 / 8.0
    b = np.zeros(11)
    return W, b


def build_cs_edge(nin):
    W = np.zeros((2, nin, 3, 3))
    for i in range(nin):
        W[0, i, 1, 1] = 1.0
        W[1, i, 1, 1] = -1.0
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                W[0, i, dy, dx] = -1.0 / 8.0
                W[1, i, dy, dx] = +1.0 / 8.0
    b = np.zeros(2)
    return W, b


# ----------------------------------------------------------------------------
# Dense readouts from rule tables.  Each rule is (class, channel, mask,
# weight).  After accumulating, every feature is made zero-sum across
# classes so features can only REDISTRIBUTE score between classes, never
# lift one class on structure that every picture has.
# ----------------------------------------------------------------------------
def build_masks4():
    """4x4 grid of 8x8-pixel cells (row 0 = sky ... row 3 = ground)."""
    return {
        "all": lambda r, c: True,
        "ctr": lambda r, c: 1 <= r <= 2 and 1 <= c <= 2,
        "ring": lambda r, c: not (1 <= r <= 2 and 1 <= c <= 2),
        "row0": lambda r, c: r == 0,
        "row1": lambda r, c: r == 1,
        "row3": lambda r, c: r == 3,
        "top2": lambda r, c: r < 2,
        "bot2": lambda r, c: r >= 2,
        "mid2all": lambda r, c: 1 <= r <= 2,
    }


def build_masks8():
    """8x8 grid of 4-pixel cells."""
    return {
        "all": lambda r, c: True,
        "ctr": lambda r, c: 2 <= r <= 5 and 2 <= c <= 5,
        "ring": lambda r, c: not (2 <= r <= 5 and 2 <= c <= 5),
        "rows0-3": lambda r, c: r < 4,
        "rows0-2": lambda r, c: r < 3,
        "rows3-5": lambda r, c: 3 <= r <= 5,
        "rows4-6": lambda r, c: 4 <= r <= 6,
        "rows5-7": lambda r, c: r >= 5,
        "rows6-7": lambda r, c: r >= 6,
        "top2": lambda r, c: r < 2,
        "row0": lambda r, c: r == 0,
    }


# ----------------------------------------------------------------------------
# Rule tables.  Channels for pB/pCS (colour words, 11):
#   0 red 1 green 2 blue 3 yellow 4 warm hue 5 grey 6 dark 7 bright
#   8 mild-warm (tan) 9 strong-warm (vivid) 10 deep blue
# ----------------------------------------------------------------------------
def colour_rules():
    """Colour-layout evidence, disciplined: only HIGH-PRECISION claims.
    Each class gets features on which it is an outlier vs all others:
    - animals: a tan/brown centre (fur fills the frame) - shared by all five;
    - vehicles: dark shadow/wheel strip at the bottom, NOT tan, NOT green ring;
    - ship: DEEP blue water below + blue above + grey/bright hull;
    - airplane: blue all around, pale centre, NOT dark below, NOT tan;
    - within animals: deer/horse see green fields, cat sees no grass,
      dog is warm-backed, frog is the most tan and least grey of all."""
    R = []
    ANIMAL = (CAT, DEER, DOG, FROG, HORSE)
    # ---- animal core: tan/brown body filling the middle ----
    for k in ANIMAL:
        R += [(k, 8, "ctr", 2.0)]
    # animal differentiators (ring-scoped, small)
    R += [(DEER, 1, "ring", 0.25), (DEER, 5, "ctr", -0.2),
          (HORSE, 6, "ctr", 0.45), (HORSE, 1, "ring", 0.3),
          (CAT, 1, "ring", -0.30), (CAT, 0, "ring", 0.25), (CAT, 3, "ring", 0.25), (CAT, 5, "ctr", 0.35),
          (DOG, 0, "ring", 0.25), (DOG, 3, "ring", 0.25), (DOG, 6, "ctr", 0.25), (DOG, 5, "ctr", 0.35), (DOG, 1, "ring", 0.2),
          (FROG, 8, "ctr", 1.6), (FROG, 5, "all", -0.35), (FROG, 7, "all", -0.10),
          (FROG, 1, "ctr", 0.60), (FROG, 3, "ring", 0.25), (FROG, 6, "ctr", 0.30)]
    # ---- vehicle core: dark strip below, body not tan, not a green scene ----
    for k in (AUTO, TRUCK):
        R += [(k, 6, "row3", 0.5), (k, 6, "bot2", 0.35),
              (k, 4, "ctr", 0.5),
              (k, 5, "bot2", 0.55), (k, 2, "row0", 0.6)]
    # deltas: the car is low (sky at row 1); the truck is a tall box
    R += [(AUTO, 2, "row1", 0.3), (AUTO, 1, "row0", 0.3), (AUTO, 1, "row3", 0.25),
          (AUTO, 5, "ctr", 0.45), (AUTO, 7, "row0", 0.3), (AUTO, 4, "ctr", 0.15), (AUTO, 9, "ctr", 0.1),
          (TRUCK, 4, "row0", 0.25), (TRUCK, 4, "row1", 0.5), (TRUCK, 5, "row1", 0.5),
          (TRUCK, 7, "row0", -0.3), (TRUCK, 7, "row1", -0.15)]
    # airplane: blue around, pale centre, least dark bottom, least tan
    R += [(AIR, 2, "row0", 1.4), (AIR, 2, "row1", 0.9), (AIR, 2, "bot2", 0.5),
          (AIR, 2, "ring", 0.6), (AIR, 5, "ctr", 0.5), (AIR, 7, "ctr", 0.6),
          (AIR, 7, "row0", 0.8), (AIR, 6, "bot2", -0.3)]
    # ship: deep blue water below, blue above, grey/bright hull, horizon
    R += [(SHIP, 10, "row3", 1.7), (SHIP, 10, "bot2", 0.9),
          (SHIP, 2, "row0", 0.6), (SHIP, 2, "ring", 0.4), (SHIP, 2, "bot2", 0.3),
          (SHIP, 5, "ctr", 0.15), (SHIP, 7, "ctr", 0.3), (SHIP, 6, "bot2", 0.15),
          (SHIP, 5, "bot2", 0.1), (SHIP, 7, "row0", 0.25),
          ]
    # bird: small tan or dark thing with sky/canopy behind
    R += [(BIRD, 2, "row0", 1.2), (BIRD, 2, "row1", 0.6), (BIRD, 1, "row0", 1.0),
          (BIRD, 6, "row0", 0.5), (BIRD, 6, "row1", 0.5), (BIRD, 8, "ctr", 0.6),
          (BIRD, 6, "row3", -0.3)]
    # channel 8/9 form a peaked band: mild warm counts, strong warm
    # (vivid red/orange, like a red car) counts against it twice over
    R += [(k, 9, m, -2 * w) for (k, ch, m, w) in R if ch == 8]
    return R


def cs_rules():
    """pCS: colour-word 'islands' (centre cell stands out from surroundings)."""
    R = []
    for k in (CAT, DEER, DOG, FROG, HORSE):   # tan body standing out
        R += [(k, 8, 1.6), (k, 6, 0.3)]
    R += [(FROG, 8, 0.3), (HORSE, 6, 0.3)]
    for k in (AUTO, TRUCK):        # coloured box on a grey road
        R += [(k, 4, 0.6), (k, 5, 0.8), (k, 8, -0.4)]
    R += [
        (AIR, 5, 1.5), (AIR, 7, 1.2), (AIR, 2, -1.0), (AIR, 10, -0.3),
        (BIRD, 6, 1.5), (BIRD, 8, 0.5),
        (SHIP, 5, 1.2), (SHIP, 7, 0.8), (SHIP, 6, 0.8), (SHIP, 2, -0.5),
        (SHIP, 10, 0.4),
    ]
    return [(k, ch, "all", w) for (k, ch, w) in R]


def edge_rules():
    """dE on pE (16ch = 4 orientations x +/- even, +/- odd; maps are
    norm-alised, so 'above-average texture for this picture').  Small
    weights; the main signal is the scale contrast: fine centre texture =
    fur, machines are smooth at small scale."""
    R = []
    def ori(t):
        return [4 * t + j for j in range(4)]
    for ch in ori(2):  # automobile: wide and low
        R += [(AUTO, ch, "ctr", 0.3)]
    for ch in ori(0):  # truck: tall box
        R += [(TRUCK, ch, "ctr", 0.5)]
    for ch in ori(2):  # ship: deck lines in the middle rows
        R += [(SHIP, ch, "mid2all", 0.2)]
    for t in range(4):  # scale contrast
        for ch in ori(t):
            R += [(CAT, ch, "ctr", 0.15), (DOG, ch, "ctr", 0.14),
                  (DEER, ch, "ctr", 0.13), (HORSE, ch, "ctr", 0.12),
                  (FROG, ch, "ctr", 0.11), (BIRD, ch, "ctr", 0.0),
                  (AUTO, ch, "ctr", -0.10), (TRUCK, ch, "ctr", -0.10),
                  (SHIP, ch, "ctr", -0.10)]
    return R


def edge2_rules():
    """dE2 on pE2 (norm-alised long-straight-edge maps): 0..7 = +/- even big
    Gabor per orientation (2t, 2t+1), 8/9 = +/- horizon, 10 = big dark
    centre, 11 = big bright centre."""
    R = []
    def pair(t):
        return [2 * t, 2 * t + 1]
    # vehicle core: long straight box edges, dark shadow mass at the bottom
    for k in (AUTO, TRUCK):
        for ch in pair(2):
            R += [(k, ch, "mid2all", 0.25)]
        for ch in pair(0):
            R += [(k, ch, "mid2all", -0.1)]
        R += [(k, 10, "row3", 0.4)]
    # truck delta: taller - more vertical edge, dark mass reaches the middle
    for ch in pair(0):
        R += [(TRUCK, ch, "mid2all", 0.2)]
    R += [(TRUCK, 10, "ctr", 0.3)]
    # car delta: some diagonal bonnet/windshield edges
    for ch in pair(1) + pair(3):
        R += [(AUTO, ch, "ctr", 0.3)]
    # ship: horizon band, dark hull at the waterline
    R += [(SHIP, 8, "mid2all", 0.07), (SHIP, 9, "mid2all", 0.07),
          (SHIP, 10, "mid2all", 0.35), (SHIP, 11, "ctr", 0.3)]
    for ch in pair(2):
        R += [(SHIP, ch, "ctr", 0.15)]
    # airplane: big pale blob, wing lines, no horizon, no dark hull
    R += [(AIR, 11, "ctr", 0.8)]
    for ch in pair(1) + pair(3):
        R += [(AIR, ch, "ctr", 0.2)]
    for ch in pair(2):
        R += [(AIR, ch, "ctr", 0.2)]
    R += [(AIR, 8, "mid2all", -0.4), (AIR, 9, "mid2all", -0.4),
          (AIR, 10, "bot2", -0.6)]
    # animals: few long straight edges anywhere
    for t in range(4):
        for ch in pair(t):
            R += [(CAT, ch, "ctr", -0.12), (DOG, ch, "ctr", -0.11),
                  (HORSE, ch, "ctr", -0.11), (DEER, ch, "ctr", -0.10),
                  (FROG, ch, "ctr", -0.09), (BIRD, ch, "ctr", -0.08)]
    R += [(HORSE, 10, "row3", 0.3), (DEER, 10, "row3", 0.3),
          (CAT, 10, "ctr", 0.3), (BIRD, 11, "row0", 0.3)]
    return R


def cs_edge_rules():
    """dCSE (small scale): ch0 = interior fine texture = fur -> animals;
    ch1 = boundary ring.  dCSE2 (long edges): ch0 = interior long straight
    structure -> vehicles; ch1 = boundary ring."""
    interior = [(CAT, 0.30), (DOG, 0.28), (DEER, 0.25), (HORSE, 0.22),
                (FROG, 0.20), (BIRD, 0.1), (AUTO, -0.2), (TRUCK, -0.2),
                (SHIP, -0.15), (AIR, 0.0)]
    ringlike = [(HORSE, 0.25), (DEER, 0.22), (DOG, 0.2), (CAT, 0.2),
                (FROG, 0.18), (BIRD, 0.2), (AUTO, -0.15), (TRUCK, -0.15),
                (SHIP, 0.0), (AIR, 0.0)]
    return ([(k, 0, "all", w) for k, w in interior]
            + [(k, 1, "all", w) for k, w in ringlike])


def cs_edge2_rules():
    interior = [(AUTO, 0.6), (TRUCK, 0.6), (SHIP, 0.4), (AIR, 0.3),
                (HORSE, -0.15), (DEER, -0.15), (CAT, -0.15), (DOG, -0.15),
                (FROG, -0.1), (BIRD, 0.0)]
    ringlike = [(HORSE, 0.25), (DEER, 0.25), (DOG, 0.2), (CAT, 0.2),
                (FROG, 0.15), (BIRD, 0.2), (AUTO, -0.2), (TRUCK, -0.2),
                (SHIP, 0.0), (AIR, 0.0)]
    return ([(k, 0, "all", w) for k, w in interior]
            + [(k, 1, "all", w) for k, w in ringlike])


def wheel_rules():
    """dW on pW (8x8 grid of 4px cells, norm-alised): 0 small dark centre,
    1 medium dark centre (wheels), 2 darkness, 3 small bright centre.
    Wheels are medium round dark centres; thin legs and eyes are small
    dark details - vehicles claim the medium ones, animals the small ones."""
    R = []
    # vehicle core: wheels / shadow strip along the bottom
    for k in (AUTO, TRUCK):
        R += [(k, 1, "rows5-7", 0.45), (k, 2, "rows6-7", 0.20),
              (k, 0, "rows5-7", 0.15)]
    R += [(TRUCK, 1, "rows5-7", 0.25), (TRUCK, 2, "rows6-7", 0.05)]
    # quadruped core: thin dark legs below, dark face/eyes in the middle
    for k in (DEER, HORSE, DOG):
        R += [(k, 2, "rows5-7", 0.35), (k, 0, "rows5-7", 0.3),
              (k, 1, "rows5-7", -0.5), (k, 0, "ctr", 0.3)]
    R += [(CAT, 2, "ctr", 0.3), (CAT, 0, "ctr", 0.3),
          (FROG, 2, "ctr", 0.7), (FROG, 0, "ctr", 1.0),
          # bird: small dark body, usually in the upper half
          (BIRD, 0, "rows0-2", 1.0), (BIRD, 2, "rows0-2", 0.7),
          # ship: dark hull at the waterline
          (SHIP, 1, "rows3-5", 0.25), (SHIP, 2, "rows4-6", 0.3),
          (SHIP, 0, "ctr", 0.2),
          # airplane: engines are small dark spots in the middle, not on top
          (AIR, 0, "ctr", 0.2), (AIR, 2, "all", -0.15), (AIR, 0, "rows0-3", -0.5),
          (AIR, 3, "ctr", 0.5)]
    return R


# ----------------------------------------------------------------------------
# Readout normalisation: the seven dense branches form one template per
# class.  Channels are rescaled by their typical magnitude (measured on
# formula-drawn calibration scenes, never on training data), each branch
# gets a hand-chosen share of the template budget, every class row is made
# zero-mean (a class must be a CONTRAST between features it likes and
# features it does not, never a blanket of positives), and finally all
# class templates are unit-length, so no class can out-score the others on
# generic input - scores are cosine matches between image and template.
# ----------------------------------------------------------------------------
IMPORTANCE = {"pB": 1.0, "pCS": 0.30, "pE": 0.10, "pE2": 0.10,
              "pW": 0.12, "pCSE": 0.04, "pCSE2": 0.03}
SCALE_FLOOR = 0.15


def _conv(x, W, b, stride, padding):
    n, c, h, w = x.shape
    o, _, kh, kw = W.shape
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho, wo = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    out = np.zeros((n, o, ho, wo))
    for dy in range(kh):
        for dx in range(kw):
            patch = x[:, :, dy:dy + stride * ho:stride, dx:dx + stride * wo:stride]
            out += np.einsum("nchw,oc->nohw", patch, W[:, :, dy, dx])
    return out + b[None, :, None, None]


def _pool(x, size, stride, reduce):
    n, c, h, w = x.shape
    (sh, sw), (th, tw) = stride, size
    ho, wo = (h - sh) // th + 1, (w - sw) // tw + 1
    out = np.zeros((n, c, ho, wo))
    for dy in range(sh):
        for dx in range(sw):
            win = x[:, :, dy:dy + th * ho:th, dx:dx + tw * wo:tw]
            out = np.maximum(out, win) if reduce == "max" else out + win
    return out / (sh * sw) if reduce != "max" else out


def _norm(x):
    mu = x.mean(axis=(2, 3), keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(axis=(2, 3), keepdims=True) + 1e-5)


def calibration_scenes():
    """Formula-drawn 32x32 scenes (sky, sea, grass, road, wall, dirt, with
    plane / ship / animal / vehicle shapes).  Used ONLY to measure typical
    feature magnitudes; no training data is involved anywhere."""
    rng = np.random.default_rng(12345)

    def canvas(c):
        return np.ones((3, 32, 32)) * np.array(c)[:, None, None]

    def grad(top, bot):
        t = (np.arange(32) / 31.0)[:, None]
        return np.stack([np.full((32, 32), bot[i]) + (top[i] - bot[i]) * (1 - t)
                         for i in range(3)])

    def blob(x, r0, r1, c0, c1, col, ellipse=True):
        ys, xs = np.mgrid[0:32, 0:32]
        inside = (ys >= r0) & (ys < r1) & (xs >= c0) & (xs < c1)
        if ellipse:
            cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
            ry, rx = max((r1 - r0) / 2, .5), max((c1 - c0) / 2, .5)
            inside = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1
        for i in range(3):
            x[i][inside] = col[i]
        return x

    def clutter(x):
        for _ in range(50):
            r0, c0 = rng.integers(0, 29), rng.integers(0, 29)
            h, w = rng.integers(2, 5), rng.integers(2, 5)
            col = x[:, r0, c0] * (0.8 + 0.4 * rng.random(3))
            x[:, r0:r0 + h, c0:c0 + w] = col[:, None, None]
        return x

    GR, RD, SK, WA = (0.33, 0.48, 0.24), (0.44, 0.44, 0.44), (0.55, 0.66, 0.88), (0.22, 0.38, 0.55)
    BR, TN, GRAY = (0.40, 0.28, 0.17), (0.56, 0.42, 0.27), (0.80, 0.80, 0.82)
    scenes = [
        blob(grad((0.62, 0.72, 0.92), (0.50, 0.60, 0.84)), 13, 19, 6, 27, GRAY),   # plane on sky
        blob(grad((0.62, 0.72, 0.92), (0.50, 0.60, 0.84)), 10, 16, 8, 25, GRAY),   # plane on sky 2
    ]
    for hz in (15, 17):                                                            # ships
        x = canvas((0.60, 0.68, 0.85)); x[:, hz:, :] = np.array(WA)[:, None, None]
        blob(x, hz - 4, hz, 10, 22, (0.88, 0.88, 0.88), False)
        blob(x, hz, hz + 6, 5, 27, (0.45, 0.46, 0.48), False)
        scenes.append(x)
    for bg, body in ((GR, BR), (GR, BR), ((0.28, 0.42, 0.20), TN), (GR, TN)):      # horse/deer
        x = clutter(canvas(bg)); r0 = 9 if body is BR else 13
        blob(x, r0, r0 + (14 if body is BR else 8), 5, 27, body)
        x[:, r0 + 12:r0 + 18, 9:12] = 0.15; x[:, r0 + 12:r0 + 18, 20:23] = 0.15
        scenes.append(x)
    scenes.append(clutter(blob(canvas((0.24, 0.34, 0.18)), 6, 28, 4, 28, (0.45, 0.35, 0.22))))  # frog
    scenes.append(clutter(blob(canvas((0.24, 0.34, 0.18)), 6, 28, 4, 28, (0.30, 0.45, 0.20))))  # green frog
    for bg in ((0.62, 0.52, 0.42), (0.45, 0.38, 0.28)):                            # cat / dog
        x = clutter(canvas(bg)); blob(x, 8, 26, 6, 26, BR)
        x[:, 10:14, 11:14] = 0.10; x[:, 10:14, 18:21] = 0.10
        scenes.append(x)
    x = canvas(SK); x[:, 14:, :] = np.array(RD)[:, None, None]; blob(x, 12, 22, 4, 28, (0.72, 0.20, 0.15))
    x[:, 22:25, 3:29] = 0.18; blob(x, 21, 26, 7, 12, (0.08, 0.08, 0.08)); blob(x, 21, 26, 22, 27, (0.08, 0.08, 0.08))
    scenes.append(clutter(x))                                                      # red automobile
    x = canvas(SK); x[:, 14:, :] = np.array(RD)[:, None, None]; blob(x, 12, 22, 4, 28, (0.55, 0.55, 0.58))
    x[:, 22:25, 3:29] = 0.18; blob(x, 21, 26, 7, 12, (0.08, 0.08, 0.08)); blob(x, 21, 26, 22, 27, (0.08, 0.08, 0.08))
    scenes.append(clutter(x))                                                      # grey automobile
    x = canvas(SK); x[:, 10:, :] = np.array(RD)[:, None, None]; blob(x, 4, 18, 6, 26, (0.82, 0.80, 0.78), False)
    blob(x, 17, 23, 8, 22, (0.35, 0.40, 0.50)); x[:, 24:27, 5:27] = 0.16
    blob(x, 23, 28, 8, 13, (0.08, 0.08, 0.08)); blob(x, 23, 28, 21, 26, (0.08, 0.08, 0.08))
    scenes.append(clutter(x))                                                      # truck
    scenes.append(clutter(blob(grad((0.60, 0.71, 0.90), (0.48, 0.58, 0.82)), 10, 15, 13, 20, (0.22, 0.20, 0.18))))  # bird
    scenes.append(clutter(blob(canvas(GR), 12, 20, 10, 22, (0.25, 0.25, 0.25))))   # gray animal on grass
    out = []
    for i, sc in enumerate(scenes):
        out.append(np.clip(sc + np.random.default_rng(100 + i).normal(0, 0.035, sc.shape), 0, 1))
    return np.stack(out).astype(np.float32)


def run_to_pooled(layers, weights, images):
    """Mini interpreter (same semantics as the scorer's blocks) that runs the
    conv/pool pipeline and returns the pooled feature maps by name."""
    vals = {"image": images}
    pooled = {}
    for L in layers:
        if L["op"] == "dense":
            break
        ins = [vals[i] for i in L["inputs"]]
        op, name = L["op"], L["name"]
        if op == "conv":
            vals[name] = _conv(ins[0], weights[name + ".W"], weights[name + ".b"],
                               L.get("stride", 1), L.get("padding", 0))
        elif op == "relu":
            vals[name] = np.maximum(ins[0], 0)
        elif op == "norm":
            vals[name] = _norm(ins[0])
        elif op in ("maxpool", "avgpool"):
            vals[name] = _pool(ins[0], L["size"], L["stride"], op[:3])
            pooled[name] = vals[name]
    return pooled


def channel_stats(layers, weights):
    """Per pooled branch: per-channel magnitude scale, and the mean feature
    vector over the calibration scenes (the 'generic scene' direction)."""
    feats = run_to_pooled(layers, weights, calibration_scenes())
    scales, means = {}, {}
    for name, x in feats.items():
        rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=(0, 2, 3)))
        scales[name] = 1.0 / np.maximum(rms, SCALE_FLOOR)
        flat = x.reshape(len(x), -1).astype(np.float64)
        means[name] = flat.mean(axis=0)
    return scales, means


def dense_from_rules(nch, gh, gw, rules, masks, biases=None):
    """Raw dense weights from the rule table (no scaling, no zero-sum)."""
    n = nch * gh * gw
    W = np.zeros((K, n))
    for (k, ch, mask, w) in rules:
        pred = masks[mask]
        for r in range(gh):
            for c in range(gw):
                if pred(r, c):
                    W[k, ch * gh * gw + r * gw + c] += w
    b = np.zeros(K)
    for k, v in (biases or {}).items():
        b[k] = v
    return W, b


# ----------------------------------------------------------------------------
# Assemble network.json and weights.npz.
# ----------------------------------------------------------------------------
def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    W, b, WB, bB = build_colour()
    WE, bE, WE2, bE2, WW, bW = build_edges()
    WCS, bCS = build_cs()
    WCSE, bCSE = build_cs_edge(16)
    WCSE2, bCSE2 = build_cs_edge(12)

    layers = [
        {"name": "cA", "op": "conv", "inputs": ["image"], "out": 5, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "rA", "op": "relu", "inputs": ["cA"]},
        {"name": "cB", "op": "conv", "inputs": ["rA"], "out": 11, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "rB", "op": "relu", "inputs": ["cB"]},
        {"name": "pB", "op": "avgpool", "inputs": ["rB"], "size": [8, 8], "stride": [8, 8]},
        {"name": "cCS", "op": "conv", "inputs": ["pB"], "out": 11, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "rCS", "op": "relu", "inputs": ["cCS"]},
        {"name": "pCS", "op": "avgpool", "inputs": ["rCS"], "size": [4, 4], "stride": [4, 4]},
        {"name": "cE", "op": "conv", "inputs": ["image"], "out": 16, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "rE", "op": "relu", "inputs": ["cE"]},
        {"name": "nE", "op": "norm", "inputs": ["rE"]},
        {"name": "qE", "op": "relu", "inputs": ["nE"]},
        {"name": "mE", "op": "maxpool", "inputs": ["qE"], "size": [2, 2], "stride": [2, 2]},
        {"name": "pE", "op": "avgpool", "inputs": ["mE"], "size": [4, 4], "stride": [4, 4]},
        {"name": "cE2", "op": "conv", "inputs": ["image"], "out": 12, "kernel": [11, 11], "stride": 1, "padding": 5},
        {"name": "rE2", "op": "relu", "inputs": ["cE2"]},
        {"name": "nE2", "op": "norm", "inputs": ["rE2"]},
        {"name": "qE2", "op": "relu", "inputs": ["nE2"]},
        {"name": "mE2", "op": "maxpool", "inputs": ["qE2"], "size": [2, 2], "stride": [2, 2]},
        {"name": "pE2", "op": "avgpool", "inputs": ["mE2"], "size": [4, 4], "stride": [4, 4]},
        {"name": "cW", "op": "conv", "inputs": ["image"], "out": 4, "kernel": [9, 9], "stride": 1, "padding": 4},
        {"name": "rW", "op": "relu", "inputs": ["cW"]},
        {"name": "mW", "op": "maxpool", "inputs": ["rW"], "size": [2, 2], "stride": [2, 2]},
        {"name": "pW", "op": "avgpool", "inputs": ["mW"], "size": [2, 2], "stride": [2, 2]},
        {"name": "cCSE", "op": "conv", "inputs": ["pE"], "out": 2, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "rCSE", "op": "relu", "inputs": ["cCSE"]},
        {"name": "pCSE", "op": "avgpool", "inputs": ["rCSE"], "size": [4, 4], "stride": [4, 4]},
        {"name": "cCSE2", "op": "conv", "inputs": ["pE2"], "out": 2, "kernel": [3, 3], "stride": 1, "padding": 1},
        {"name": "rCSE2", "op": "relu", "inputs": ["cCSE2"]},
        {"name": "pCSE2", "op": "avgpool", "inputs": ["rCSE2"], "size": [4, 4], "stride": [4, 4]},
        {"name": "dB", "op": "dense", "inputs": ["pB"], "out": 10},
        {"name": "dCS", "op": "dense", "inputs": ["pCS"], "out": 10},
        {"name": "dE", "op": "dense", "inputs": ["pE"], "out": 10},
        {"name": "dE2", "op": "dense", "inputs": ["pE2"], "out": 10},
        {"name": "dW", "op": "dense", "inputs": ["pW"], "out": 10},
        {"name": "dCSE", "op": "dense", "inputs": ["pCSE"], "out": 10},
        {"name": "dCSE2", "op": "dense", "inputs": ["pCSE2"], "out": 10},
        {"name": "scores", "op": "add", "inputs": ["dB", "dCS", "dE", "dE2", "dW", "dCSE", "dCSE2"]},
    ]

    conv_weights = {
        "cA.W": W, "cA.b": b,
        "cB.W": WB, "cB.b": bB,
        "cCS.W": WCS, "cCS.b": bCS,
        "cE.W": WE, "cE.b": bE,
        "cE2.W": WE2, "cE2.b": bE2,
        "cW.W": WW, "cW.b": bW,
        "cCSE.W": WCSE, "cCSE.b": bCSE,
        "cCSE2.W": WCSE2, "cCSE2.b": bCSE2,
    }

    m4 = build_masks4()
    m8 = build_masks8()
    allm = {"all": lambda r, c: True}
    raw = {
        "dB": dense_from_rules(11, 4, 4, colour_rules(), m4),
        "dCS": dense_from_rules(11, 1, 1, cs_rules(), allm),
        "dE": dense_from_rules(16, 4, 4, edge_rules(), m4),
        "dE2": dense_from_rules(12, 4, 4, edge2_rules(), m4),
        "dW": dense_from_rules(4, 8, 8, wheel_rules(), m8),
        "dCSE": dense_from_rules(2, 1, 1, cs_edge_rules(), allm),
        "dCSE2": dense_from_rules(2, 1, 1, cs_edge2_rules(), allm),
    }
    branch_of = {"dB": "pB", "dCS": "pCS", "dE": "pE", "dE2": "pE2",
                 "dW": "pW", "dCSE": "pCSE", "dCSE2": "pCSE2"}
    shapes = {"pB": (11, 4, 4), "pCS": (11, 1, 1), "pE": (16, 4, 4),
              "pE2": (12, 4, 4), "pW": (4, 8, 8), "pCSE": (2, 1, 1),
              "pCSE2": (2, 1, 1)}

    scales, means = channel_stats(layers, conv_weights)

    # assemble one template per class across all branches.  Each branch row
    # is magnitude-equalised, made purely redistributive (zero sum across
    # classes for every feature), and given exactly its importance budget of
    # template length, so every class template has the same total length and
    # the same branch shares, and no class can score on structure that every
    # picture has.
    sizes = [raw[name][0].shape[1] for name in raw]
    T = np.zeros((K, sum(sizes)))
    biases = {}
    off = 0
    for name, (Wb, bb) in raw.items():
        pooled = branch_of[name]
        C, H, Wd = shapes[pooled]
        col = np.repeat(scales[pooled], H * Wd)       # per-feature magnitude
        Wb = Wb * col[None, :]
        # redistribution: a feature can only move score between classes, so
        # structure present in every picture lifts nobody
        Wb = Wb - Wb.mean(axis=0, keepdims=True)
        nrm = np.linalg.norm(Wb, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Wb = Wb / nrm * IMPORTANCE[pooled]            # exact branch budget
        T[:, off:off + Wb.shape[1]] = Wb
        off += Wb.shape[1]

    weights = dict(conv_weights)
    off = 0
    for name, (Wb, bb) in raw.items():
        n = Wb.shape[1]
        weights[name + ".W"] = T[:, off:off + n]
        weights[name + ".b"] = np.zeros(K)
        off += n

    np.savez(os.path.join(out_dir, "weights.npz"),
             **{k: v.astype(np.float32) for k, v in weights.items()})
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump({"layers": layers}, f, indent=1)
    tot = sum(int(np.prod(v.shape)) for v in weights.values())
    print(f"wrote {out_dir}: {tot} weights")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
