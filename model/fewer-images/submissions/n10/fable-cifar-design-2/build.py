#!/usr/bin/env python3
"""Hand-designed 10-class photo classifier.  Usage: python3 build.py OUT_DIR

Every number below is typed from general knowledge of what skies, grass, roads, water, fur, cars ... look like,
or follows from a formula.  Nothing is read from any file and no image data is used.

Structure of the network
  1. per-pixel "colour words" (dark, greys, white, sky-blue, deep-blue, green, olive, brown, tan, red) and
     "texture words" (flat, weak texture or strong edge x horizontal / vertical / diagonal), from 3x3 / 1x1 convs + relu
  2. the words are averaged over a 4 x 4 grid of cells
  3. a table of hand-described scenes (class x outline variant x background type x object size); every scene says
     which words it expects in every cell; its score is  log prior + sum over cells of  <word fractions, log expected
     fractions>  and the class score is the maximum over its scenes (maxpool)
  4. small extra votes, each a hand-typed table of "how often does this class show this property":
     colour words that are more frequent in the centre than around it (object colours), top row against bottom row,
     long straight edges, a horizon, the extent (wide-thin / wide-thick / tall / compact / large) of the region that
     differs from the left and right image borders, and the colour words inside that region.
"""
import json
import os
import sys

import numpy as np

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CW = ['K', 'DG', 'LG', 'W', 'SKY', 'DB', 'G', 'O', 'BR', 'TAN', 'RED']      # colour words
TW = ['FLAT', 'WH', 'WV', 'WD', 'SH', 'SV', 'SD']                            # texture words
NCW, NTW = len(CW), len(TW)
LUMA = np.array([0.299, 0.587, 0.114])


def cdist(**kw):
    v = np.zeros(NCW)
    for k, x in kw.items():
        v[CW.index(k)] = x
    return v / v.sum()


# ----------------------------------------------------------------------------------------------------------------
# Hand-typed knowledge tables
# ----------------------------------------------------------------------------------------------------------------
# how often each colour word occurs in web photos overall (used only for smoothing)
GENERIC_C = cdist(K=.15, DG=.12, LG=.11, W=.07, SKY=.07, DB=.05, G=.08, O=.08, BR=.12, TAN=.11, RED=.04)
ISO = (.30, .30, .40)      # share of horizontal / vertical / diagonal edges when there is no preferred direction (bin widths)


def tex(flat, weak, strong, hvd=ISO):
    """Texture-word distribution [FLAT, WH, WV, WD, SH, SV, SD] from the shares of flat / weakly textured / strong-edge pixels
    and the split of edge directions."""
    h, v, d = np.array(hvd, dtype=np.float64) / np.sum(hvd)
    t = np.array([flat, weak * h, weak * v, weak * d, strong * h, strong * v, strong * d], dtype=np.float64)
    return t / t.sum()


GENERIC_T = tex(.42, .43, .15)

# materials: (colour-word distribution, texture-word distribution)
MAT = {
    'SKYC': (cdist(DG=.02, LG=.06, W=.07, SKY=.68, DB=.15, TAN=.01, RED=.01), tex(.84, .13, .03)),               # clear sky
    'SKYW': (cdist(W=.72, LG=.18, TAN=.04, SKY=.04, DG=.01, O=.01), tex(.90, .08, .02)),                         # blown-out sky, studio white
    'SKYG': (cdist(LG=.50, DG=.24, W=.10, SKY=.08, DB=.06, TAN=.02), tex(.80, .17, .03)),                        # overcast grey sky, grey wall
    'SKYD': (cdist(K=.12, DG=.34, LG=.08, DB=.38, SKY=.04, BR=.02, TAN=.02), tex(.80, .16, .04)),                # dusk / storm / deep-blue sky
    'SUNS': (cdist(TAN=.30, BR=.15, RED=.12, O=.10, LG=.08, DG=.08, K=.07, W=.05, SKY=.05), tex(.70, .24, .06, (.45, .20, .35))),
    'FOLI': (cdist(G=.36, O=.17, K=.22, DG=.08, BR=.07, LG=.03, TAN=.03, W=.02, SKY=.02), tex(.20, .58, .22, (.28, .34, .38))),   # trees, bushes
    'GRAS': (cdist(G=.50, O=.28, K=.05, DG=.03, BR=.05, TAN=.06, LG=.03), tex(.32, .56, .12, (.28, .36, .36))),  # lawn, field
    'BLUR': (cdist(G=.34, O=.26, BR=.08, TAN=.05, DG=.10, K=.09, LG=.06, DB=.01, SKY=.01), tex(.68, .28, .04)),  # out-of-focus greenery
    'BLRW': (cdist(BR=.24, TAN=.26, O=.14, DG=.12, LG=.12, K=.06, W=.03, G=.02, RED=.01), tex(.68, .28, .04)),   # out-of-focus / plain warm surface
    'EART': (cdist(BR=.30, TAN=.24, O=.14, DG=.10, LG=.08, K=.08, G=.03, W=.02, RED=.01), tex(.30, .55, .15, (.34, .28, .38))),   # soil, dry grass, wood, sand
    'ROAD': (cdist(DG=.35, LG=.29, W=.05, K=.06, DB=.08, SKY=.04, TAN=.06, BR=.04, O=.02, G=.01), tex(.52, .37, .11, (.50, .15, .35))),
    'WATB': (cdist(DB=.40, SKY=.22, DG=.09, K=.07, LG=.05, G=.10, W=.04, O=.03), tex(.42, .46, .12, (.55, .13, .32))),            # blue water
    'WATG': (cdist(DG=.28, LG=.20, DB=.20, W=.06, K=.08, SKY=.08, G=.05, O=.03, TAN=.02), tex(.48, .42, .10, (.55, .13, .32))),   # grey water
    'WATD': (cdist(K=.28, G=.16, O=.14, DG=.18, BR=.10, DB=.08, LG=.04, TAN=.02), tex(.52, .38, .10, (.50, .15, .35))),           # dark / green water
    'INDR': (cdist(TAN=.19, BR=.20, LG=.10, W=.07, DG=.10, K=.16, RED=.07, O=.05, DB=.03, SKY=.01, G=.02), tex(.36, .46, .18, (.33, .35, .32))),  # indoors
    'DRKB': (cdist(K=.68, DG=.14, BR=.08, DB=.05, G=.02, O=.02, RED=.01), tex(.82, .15, .03)),                   # dark background
    'URBN': (cdist(LG=.17, DG=.19, W=.09, TAN=.12, BR=.09, K=.11, RED=.05, SKY=.05, DB=.05, G=.04, O=.04), tex(.28, .50, .22, (.34, .40, .26))),  # buildings, street
    'SNOW': (cdist(W=.48, LG=.28, SKY=.12, DG=.08, DB=.02, TAN=.02), tex(.60, .33, .07)),
}


def mix(**kw):
    c = sum(w * MAT[m][0] for m, w in kw.items())
    t = sum(w * MAT[m][1] for m, w in kw.items())
    return c / c.sum(), t / t.sum()


# background types: the material mixture expected in each of the four rows of cells (top ... bottom)
BG = {
    'sky_clear_all': [mix(SKYC=1)] * 4,
    'white_all': [mix(SKYW=1)] * 4,
    'sky_grey_all': [mix(SKYG=1)] * 4,
    'sky_dark_all': [mix(SKYD=1)] * 4,
    'sky_grass': [mix(SKYC=.45, SKYW=.25, SKYG=.20, FOLI=.10), mix(SKYC=.14, SKYW=.08, SKYG=.08, FOLI=.5, GRAS=.2), mix(FOLI=.2, GRAS=.8), mix(GRAS=.85, EART=.15)],
    'trees_grass': [mix(FOLI=.75, SKYW=.10, SKYG=.05, SKYC=.10), mix(FOLI=.8, GRAS=.2), mix(FOLI=.4, GRAS=.6), mix(GRAS=.75, EART=.25)],
    'woods_mixed': [mix(FOLI=.50, EART=.30, GRAS=.20)] * 2 + [mix(FOLI=.30, EART=.40, GRAS=.30)] * 2,
    'blur_green': [mix(BLUR=1)] * 4,
    'blur_warm': [mix(BLRW=1)] * 4,
    'earth_all': [mix(EART=.55, FOLI=.35, BLUR=.10), mix(EART=.75, FOLI=.25), mix(EART=1), mix(EART=1)],
    'airfield': [mix(SKYC=.45, SKYW=.30, SKYG=.25), mix(SKYC=.32, SKYW=.20, SKYG=.18, URBN=.15, FOLI=.15), mix(ROAD=.50, GRAS=.25, SKYG=.10, URBN=.15), mix(ROAD=.65, GRAS=.35)],
    'harbour': [mix(SKYC=.25, SKYW=.20, SKYG=.20, URBN=.35), mix(URBN=.50, SKYG=.15, SKYW=.10, WATG=.25), mix(WATG=.40, WATB=.30, URBN=.30), mix(WATG=.45, WATB=.30, WATD=.25)],
    'sky_road': [mix(SKYC=.40, SKYW=.28, SKYG=.22, URBN=.10), mix(SKYC=.13, SKYW=.09, SKYG=.08, URBN=.45, FOLI=.25), mix(URBN=.30, FOLI=.15, ROAD=.55), mix(ROAD=1)],
    'urban_road': [mix(URBN=.55, FOLI=.30, SKYW=.08, SKYG=.07), mix(URBN=.7, FOLI=.3), mix(URBN=.45, FOLI=.10, ROAD=.45), mix(ROAD=.9, URBN=.1)],
    'sky_water_blue': [mix(SKYC=.55, SKYW=.25, SKYG=.20), mix(SKYC=.36, SKYW=.16, SKYG=.13, WATB=.20, URBN=.15), mix(SKYC=.14, SKYW=.06, SKYG=.05, WATB=.65, URBN=.10), mix(WATB=1)],
    'grey_sea': [mix(SKYG=.45, SKYW=.35, SKYD=.20), mix(SKYG=.30, SKYW=.22, SKYD=.13, WATG=.25, URBN=.10), mix(SKYG=.2, WATG=.7, URBN=.1), mix(WATG=1)],
    'indoor': [mix(INDR=1)] * 4,
    'dark_bg': [mix(DRKB=1)] * 4,
    'sky_earth': [mix(SKYC=.40, SKYW=.28, SKYG=.22, FOLI=.10), mix(SKYC=.13, SKYW=.09, SKYG=.08, FOLI=.35, EART=.20, URBN=.15), mix(EART=.8, FOLI=.2), mix(EART=1)],
    'water_blue_all': [mix(WATB=.65, WATG=.35)] * 4,
    'pond_dark': [mix(WATD=1)] * 4,
    'sunset_water': [mix(SUNS=1), mix(SUNS=.6, DRKB=.4), mix(WATD=.5, DRKB=.5), mix(WATD=.6, DRKB=.4)],
    'snow': [mix(SNOW=.45, FOLI=.35, SKYG=.20), mix(SNOW=.7, FOLI=.3), mix(SNOW=1), mix(SNOW=1)],
}
BGN = list(BG)

# how often each class is photographed in front of each background type (columns in CLASSES order)
BG_PRIOR = {
    #                  plane  auto   bird   cat    deer   dog    frog   horse  ship   truck
    'sky_clear_all':  [.25,  .003,  .11,   .002,  .003,  .003,  .002,  .01,   .03,   .01],
    'white_all':      [.15,  .10,   .07,   .06,   .015,  .08,   .08,   .05,   .05,   .10],
    'sky_grey_all':   [.15,  .03,   .07,   .03,   .01,   .03,   .02,   .02,   .04,   .03],
    'sky_dark_all':   [.09,  .003,  .015,  .002,  .003,  .002,  .002,  .003,  .03,   .004],
    'sky_grass':      [.05,  .04,   .03,   .005,  .09,   .04,   .003,  .17,   .004,  .06],
    'trees_grass':    [.01,  .08,   .07,   .06,   .25,   .14,   .10,   .19,   .004,  .07],
    'woods_mixed':    [.005, .01,   .05,   .02,   .17,   .02,   .08,   .05,   .002,  .01],
    'blur_green':     [.005, .01,   .20,   .03,   .12,   .04,   .16,   .03,   .002,  .004],
    'blur_warm':      [.01,  .01,   .16,   .04,   .05,   .04,   .17,   .03,   .005,  .01],
    'earth_all':      [.01,  .03,   .07,   .06,   .13,   .07,   .20,   .07,   .004,  .03],
    'airfield':       [.12,  .03,   .005,  .002,  .005,  .01,   .001,  .03,   .01,   .05],
    'harbour':        [.005, .005,  .01,   .001,  .001,  .002,  .001,  .002,  .07,   .01],
    'sky_road':       [.04,  .17,   .01,   .005,  .005,  .015,  .002,  .03,   .03,   .28],
    'urban_road':     [.04,  .32,   .02,   .05,   .015,  .06,   .005,  .11,   .02,   .25],
    'sky_water_blue': [.03,  .003,  .03,   .001,  .005,  .008,  .002,  .005,  .32,   .003],
    'grey_sea':       [.015, .003,  .02,   .001,  .003,  .004,  .002,  .004,  .20,   .003],
    'indoor':         [.02,  .08,   .03,   .46,   .005,  .37,   .06,   .05,   .01,   .03],
    'dark_bg':        [.01,  .05,   .03,   .09,   .04,   .05,   .09,   .03,   .02,   .03],
    'sky_earth':      [.02,  .05,   .015,  .003,  .04,   .02,   .003,  .13,   .01,   .10],
    'water_blue_all': [.005, .001,  .04,   .001,  .005,  .004,  .008,  .002,  .08,   .001],
    'pond_dark':      [.001, .001,  .02,   .001,  .01,   .002,  .04,   .002,  .03,   .001],
    'sunset_water':   [.02,  .003,  .005,  .001,  .003,  .001,  .001,  .004,  .05,   .002],
    'snow':           [.005, .01,   .01,   .005,  .05,   .02,   .002,  .02,   .003,  .01],
}

# object appearance per class: colour words of the object itself
OBJ_C = {
    'airplane':   cdist(K=.15, DG=.22, LG=.24, W=.23, SKY=.04, DB=.05, G=.01, O=.01, BR=.02, TAN=.02, RED=.02),   # often a dark silhouette against bright sky
    'automobile': cdist(K=.21, DG=.15, LG=.12, W=.10, SKY=.05, DB=.08, G=.02, O=.03, BR=.05, TAN=.06, RED=.13),
    'bird':       cdist(K=.18, DG=.15, LG=.10, W=.09, SKY=.02, DB=.04, G=.02, O=.07, BR=.18, TAN=.11, RED=.04),
    'cat':        cdist(K=.15, DG=.20, LG=.15, W=.10, SKY=.01, DB=.02, G=.01, O=.05, BR=.15, TAN=.13, RED=.03),
    'deer':       cdist(K=.10, DG=.09, LG=.04, W=.04, SKY=.003, DB=.01, G=.02, O=.10, BR=.36, TAN=.22, RED=.02),
    'dog':        cdist(K=.14, DG=.08, LG=.08, W=.15, SKY=.01, DB=.01, G=.01, O=.05, BR=.19, TAN=.24, RED=.05),
    'frog':       cdist(K=.12, DG=.09, LG=.05, W=.02, SKY=.005, DB=.01, G=.11, O=.16, BR=.22, TAN=.18, RED=.035),
    'horse':      cdist(K=.22, DG=.10, LG=.06, W=.08, SKY=.005, DB=.015, G=.01, O=.04, BR=.30, TAN=.13, RED=.05),
    'ship':       cdist(K=.12, DG=.17, LG=.15, W=.27, SKY=.03, DB=.07, G=.01, O=.01, BR=.05, TAN=.05, RED=.08),
    'truck':      cdist(K=.14, DG=.13, LG=.12, W=.17, SKY=.05, DB=.08, G=.03, O=.04, BR=.05, TAN=.06, RED=.13),
}
# texture words inside the object, on its outline, and on its outline in the bottom row of cells (legs / ground line)
OBJ_T = {
    'airplane':   (tex(.34, .44, .22, (.50, .15, .35)), tex(0, .3, .7, (.50, .12, .38)), None),
    'automobile': (tex(.24, .48, .28, (.45, .22, .33)), tex(0, .3, .7, (.48, .20, .32)), tex(0, .3, .7, (.55, .15, .30))),
    'bird':       (tex(.28, .52, .20, (.30, .30, .40)), tex(0, .3, .7, (.30, .28, .42)), None),
    'cat':        (tex(.24, .56, .20, (.29, .31, .40)), tex(0, .3, .7, (.28, .32, .40)), None),
    'deer':       (tex(.24, .56, .20, (.30, .33, .37)), tex(0, .3, .7, (.28, .36, .36)), tex(0, .3, .7, (.20, .52, .28))),
    'dog':        (tex(.24, .56, .20, (.29, .31, .40)), tex(0, .3, .7, (.28, .32, .40)), None),
    'frog':       (tex(.24, .58, .18, (.34, .26, .40)), tex(0, .3, .7, (.36, .22, .42)), None),
    'horse':      (tex(.24, .54, .22, (.30, .34, .36)), tex(0, .3, .7, (.30, .36, .34)), tex(0, .3, .7, (.20, .52, .28))),
    'ship':       (tex(.28, .47, .25, (.50, .20, .30)), tex(0, .3, .7, (.52, .18, .30)), None),
    'truck':      (tex(.22, .48, .30, (.40, .33, .27)), tex(0, .3, .7, (.42, .36, .22)), tex(0, .3, .7, (.55, .15, .30))),
}
# outline variants, each an ellipse: (width / height, vertical centre, how much of the ellipse the object fills, how often)
SHAPE = {
    'airplane':   [(2.4, .50, .65, .5), (1.2, .50, .50, .5)],        # side view; wings spread towards the camera
    'automobile': [(1.8, .55, .90, .6), (1.25, .55, .90, .4)],       # side view; front / three-quarter view
    'bird':       [(0.9, .50, .80, .7), (1.7, .50, .50, .3)],        # perched; flying
    'cat':        [(1.0, .50, .90, 1.)],
    'deer':       [(1.0, .52, .80, 1.)],
    'dog':        [(1.0, .50, .90, 1.)],
    'frog':       [(1.4, .52, .85, 1.)],
    'horse':      [(1.15, .52, .85, .7), (0.8, .50, .85, .3)],       # side view; head / front view
    'ship':       [(2.0, .55, .75, .75), (1.1, .50, .70, .25)],      # side view; bow view / sailing boat
    'truck':      [(1.4, .55, .90, .6), (1.0, .52, .90, .4)],        # side view; front view
}
SIZES = [0.35, 0.60, 0.85, 1.25]            # object width as a fraction of the image width: small, medium, large, fills frame
# these photos are cropped around their subject, so large is the usual case
SIZE_PRIOR = {
    'airplane': [.10, .35, .40, .15], 'automobile': [.02, .25, .50, .23], 'bird': [.12, .40, .33, .15],
    'cat': [.03, .25, .42, .30], 'deer': [.08, .37, .40, .15], 'dog': [.03, .25, .42, .30],
    'frog': [.04, .30, .42, .24], 'horse': [.03, .30, .45, .22], 'ship': [.05, .32, .43, .20],
    'truck': [.02, .20, .50, .28],
}
# weight of the evidence of one cell: the four centre cells (mostly object) count more than the twelve border cells
N_COL_CENTRE, N_COL_BORDER = 0.45, 0.25
N_TEX_CENTRE, N_TEX_BORDER = 0.28, 0.17
ETA_C, ETA_T = 0.15, 0.10  # smoothing towards the generic word frequencies
EDGE_SHARE = 0.30        # share of outline pixels in a cell that is cut by the object outline

# horizon (a level edge present at both the left and the right third of one row), per class
P_HZ_BRIGHT_ABOVE = [.18, .20, .08, .07, .10, .07, .06, .25, .45, .22]
P_HZ_DARK_ABOVE = [.06, .18, .06, .06, .08, .06, .06, .12, .06, .16]
LAM_HZ = 0.7
# extent of the region that differs from the left / right borders:
# [nothing, small compact, wide and thin, wide and thick, tall, large but sparse (thin parts), large and dense or cluttered]
P_EXTENT = {
    'airplane': [.05, .08, .30, .09, .03, .22, .23], 'automobile': [.02, .03, .03, .18, .03, .10, .61], 'bird': [.05, .20, .09, .07, .16, .14, .29],
    'cat': [.03, .05, .015, .055, .12, .13, .60], 'deer': [.04, .11, .03, .09, .13, .17, .43], 'dog': [.03, .05, .015, .055, .12, .13, .60],
    'frog': [.05, .10, .03, .12, .06, .12, .52], 'horse': [.02, .04, .02, .08, .10, .16, .58], 'ship': [.03, .06, .14, .22, .04, .14, .37],
    'truck': [.02, .02, .015, .10, .05, .09, .705],
}
LAM_EXT = 0.8
N_OBJ, CAP_OBJ, LAM_OBJ, ETA_OBJ = 8.0, 1.5, 0.5, 0.30   # (shares its evidence with the centre-vs-border vote below)
# colour words that are more frequent in the four centre cells than in the border cells count as object colours
N_CS, CAP_CS, LAM_CS = 4.0, 1.5, 0.6
# at least one long straight edge (about 13 pixels or more): level-to-diagonal / upright.  Man-made things have them.
P_LONG_H = [.40, .50, .10, .12, .10, .12, .08, .22, .55, .60]
P_LONG_V = [.08, .10, .08, .12, .13, .12, .06, .15, .12, .32]
LAM_LONG = 0.7
# top row of cells against bottom row: [same colour, top brighter, bottom brighter, other]
P_TB = {
    'airplane': [.38, .25, .22, .15], 'automobile': [.14, .40, .28, .18], 'bird': [.42, .22, .20, .16], 'cat': [.25, .28, .30, .17],
    'deer': [.25, .33, .25, .17], 'dog': [.25, .28, .30, .17], 'frog': [.42, .20, .22, .16], 'horse': [.13, .47, .22, .18],
    'ship': [.12, .55, .18, .15], 'truck': [.12, .52, .20, .16],
}
LAM_TB = 0.6


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
class Net:
    def __init__(self):
        self.layers, self.w = [], {}

    def conv(self, name, inp, W, b, stride=1, padding=0):
        W = np.asarray(W, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        self.layers.append({"name": name, "op": "conv", "inputs": [inp], "out": int(W.shape[0]),
                            "kernel": [int(W.shape[2]), int(W.shape[3])], "stride": stride, "padding": padding})
        self.w[name + ".W"], self.w[name + ".b"] = W.astype(np.float32), b.astype(np.float32)
        return name

    def dense(self, name, inp, W, b):
        W = np.asarray(W, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        self.layers.append({"name": name, "op": "dense", "inputs": [inp], "out": int(W.shape[0])})
        self.w[name + ".W"], self.w[name + ".b"] = W.astype(np.float32), b.astype(np.float32)
        return name

    def relu(self, name, inp):
        self.layers.append({"name": name, "op": "relu", "inputs": [inp]}); return name

    def pool(self, kind, name, inp, size, stride=None):
        self.layers.append({"name": name, "op": kind, "inputs": [inp], "size": list(size), "stride": list(stride or size)}); return name

    def add(self, name, inps):
        self.layers.append({"name": name, "op": "add", "inputs": list(inps)}); return name

    def reshape(self, name, inp, shape):
        self.layers.append({"name": name, "op": "reshape", "inputs": [inp], "shape": list(shape)}); return name


def ramp_pair(lin_w, lin_b, lo, hi):
    """Two pre-relu units a, b with relu(a) - relu(b) = clip((x - lo) / (hi - lo), 0, 1), x = lin_w . input + lin_b."""
    lin_w = np.asarray(lin_w, dtype=np.float64)
    s = 1.0 / (hi - lo)
    return (lin_w * s, (lin_b - lo) * s), (lin_w * s, (lin_b - lo) * s - 1.0)


def coverage(aspect, cy, fill, width):
    """Share of each of the 4 x 4 cells covered by a centred ellipse (the typed outline model), by supersampling."""
    n = 64
    t = (np.arange(4 * n) + 0.5) / (4 * n)
    yy, xx = np.meshgrid(t, t, indexing='ij')
    a, b = width / 2.0, min(width / aspect, 1.3) / 2.0
    inside = (((xx - 0.5) / a) ** 2 + ((yy - cy) / b) ** 2) <= 1.0
    return fill * inside.reshape(4, n, 4, n).mean(axis=(1, 3))


def build():
    net = Net()
    # ---------------- layer 1: 3x3 conv (valid) -> 30 x 30 ----------------
    rows = []
    R, G, B = np.eye(3)
    def centre(wb):
        w, b = wb
        k = np.zeros((3, 3, 3)); k[:, 1, 1] = w
        return k, b
    for d in (R - G, G - R, G - B, B - G, B - R, R - B):                    # 0..5  rectified channel differences
        rows.append(centre((d, 0.0)))
    for lo, hi in ((0.17, 0.25), (0.45, 0.55), (0.72, 0.82)):               # 6..11 luma ramps: not-dark, light, bright
        a, b = ramp_pair(LUMA, 0.0, lo, hi); rows += [centre(a), centre(b)]
    a, b = ramp_pair(B - R, 0.0, 0.0, 0.04); rows += [centre(a), centre(b)]      # 12,13  blue beats red
    a, b = ramp_pair(B - G, 0.0, -0.04, 0.0); rows += [centre(a), centre(b)]     # 14,15  blue at least green
    a, b = ramp_pair(G - R, 0.0, 0.01, 0.05); rows += [centre(a), centre(b)]     # 16,17  green beats red
    a, b = ramp_pair(R - G, 0.0, 0.01, 0.05); rows += [centre(a), centre(b)]     # 18,19  red beats green
    a, b = ramp_pair(R - 3 * G + 2 * B, 0.0, 0.12, 0.22); rows += [centre(a), centre(b)]   # 20,21  pure red / pink rather than orange-brown
    sob = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 4.0  # a step of height h gives h
    gx = LUMA[:, None, None] * sob[None]
    gy = LUMA[:, None, None] * sob.T[None]
    rows += [(gx, 0.0), (-gx, 0.0), (gy, 0.0), (-gy, 0.0)]                  # 22..25
    for d in (R, G, B):                                                     # 26..28 pass-through
        rows.append(centre((d, 0.0)))
    rows += [(gx + gy, 0.0), (-gx - gy, 0.0), (gx - gy, 0.0), (gy - gx, 0.0)]   # 29..32
    W1 = np.stack([r[0] for r in rows]); b1 = np.array([r[1] for r in rows])
    C1 = len(rows)
    net.conv('c1', 'image', W1, b1); net.relu('r1', 'c1')

    # ---------------- layer 2: 1x1 ----------------
    e1 = lambda i: np.eye(C1)[i]
    B1, B2 = e1(12) - e1(13), e1(14) - e1(15)
    HG, HW = e1(16) - e1(17), e1(18) - e1(19)
    U0, U1, U2 = e1(6) - e1(7), e1(8) - e1(9), e1(10) - e1(11)
    RD = e1(20) - e1(21)
    chroma = 0.5 * sum(e1(i) for i in range(6))                            # max - min of the three channels
    mag = e1(22) + e1(23) + e1(24) + e1(25)                                # |gx| + |gy|
    agx, agy = e1(22) + e1(23), e1(24) + e1(25)
    l2 = [
        (B1 + B2, -1.0),                    # 0 BLUE hue
        (HG - B2, 0.0),                     # 1 GREEN hue
        (HW - B1, 0.0),                     # 2 WARM hue
        (-HG - HW - B1, 1.0),               # 3 OLIVE / yellow hue
        (U0, 0.0), (U1, 0.0), (U2, 0.0),    # 4,5,6
    ]
    l2 += list(ramp_pair(chroma, 0.0, 0.05, 0.11))      # 7,8   chromatic
    l2 += [(RD, 0.0)]                                    # 9
    l2 += list(ramp_pair(mag, 0.0, 0.04, 0.10))         # 10,11 busy (anything above sky noise / blur)
    l2 += list(ramp_pair(mag, 0.0, 0.20, 0.32))         # 12,13 strong edge (an object-against-background step)
    l2 += list(ramp_pair(agy - 2 * agx, 0.0, 0.0, 0.03))  # 14,15 horizontal edge
    l2 += list(ramp_pair(agx - 2 * agy, 0.0, 0.0, 0.03))  # 16,17 vertical edge
    l2 += [(e1(26), 0.0), (e1(27), 0.0), (e1(28), 0.0)]  # 18..20 RGB
    l2 += list(ramp_pair(mag, 0.0, 0.12, 0.20))         # 21,22 clear edge (a step of about 0.2 in brightness)
    l2 += list(ramp_pair(e1(29) + e1(30) - e1(31) - e1(32), 0.0, -0.02, 0.02))   # 23,24 gx and gy have the same sign: the edge rises to the right
    C2 = len(l2)
    net.conv('c2', 'r1', np.stack([w for w, _ in l2])[:, :, None, None], [b for _, b in l2]); net.relu('r2', 'c2')

    # ---------------- layer 3: 1x1 -> words ----------------
    e2 = lambda i: np.eye(C2)[i]
    BLUE, GRN, WARM, OLV = e2(0), e2(1), e2(2), e2(3)
    U0, U1, U2 = e2(4), e2(5), e2(6)
    CH = e2(7) - e2(8); RD = e2(9)
    UB, US = e2(10) - e2(11), e2(12) - e2(13)
    OH, OV = e2(14) - e2(15), e2(16) - e2(17)
    UM = e2(21) - e2(22); DS = e2(23) - e2(24)
    l3 = [
        (-U0, 1.0),                              # 0  K     dark
        (U0 - U1 - CH, 0.0),                     # 1  DG    dark grey
        (U1 - U2 - CH, 0.0),                     # 2  LG    light grey
        (U2 - CH, 0.0),                          # 3  W     white
        (BLUE + CH + U1, -2.0),                  # 4  SKY   light blue
        (BLUE + CH + U0 - U1, -2.0),             # 5  DB    deep blue
        (GRN + CH + U0, -2.0),                   # 6  G     green
        (OLV + CH + U0, -2.0),                   # 7  O     olive / yellow
        (WARM + CH + U0 - U1 - RD, -2.0),        # 8  BR    brown
        (WARM + CH + U1 - RD, -2.0),             # 9  TAN   tan / beige / orange
        (WARM + CH + U0 + RD, -3.0),             # 10 RED   red / pink
        (-UB, 1.0),                              # 11 FLAT
        (UB - US + OH, -1.0),                    # 12 WH    weak texture, horizontal
        (UB - US + OV, -1.0),                    # 13 WV
        (UB - US - OH - OV, 0.0),                # 14 WD
        (US + OH, -1.0),                         # 15 SH    strong horizontal edge
        (US + OV, -1.0),                         # 16 SV    strong vertical edge
        (US - OH - OV, 0.0),                     # 17 SD    strong diagonal edge
        (e2(18), 0.0), (e2(19), 0.0), (e2(20), 0.0),   # 18..20 RGB
        (UM + OH, -1.0),                         # 21 clear horizontal edge
        (UM + OV, -1.0),                         # 22 clear vertical edge
        (UM - OV + DS, -1.0),                    # 23 clear edge, not vertical, rising to the right
        (UM - OV - DS, 0.0),                     # 24 clear edge, not vertical, falling to the right
        (UM - OH + DS, -1.0),                    # 25 clear edge, not horizontal, rising
        (UM - OH - DS, 0.0),                     # 26 clear edge, not horizontal, falling
    ]
    C3 = len(l3)
    IR = NCW + NTW                               # index of the R channel
    net.conv('c3', 'r2', np.stack([w for w, _ in l3])[:, :, None, None], [b for _, b in l3]); net.relu('words', 'c3')

    # ---------------- scene table on the 4 x 4 grid ----------------
    net.pool('avgpool', 'grid', 'words', [9, 9], [7, 7])                   # (C3, 4, 4)
    per_class = max(len(v) for v in SHAPE.values()) * len(BGN) * len(SIZES)
    scenes_W, scenes_b = [], []
    for k, cls in enumerate(CLASSES):
        oc, (ot, oe, oe_bottom) = OBJ_C[cls], OBJ_T[cls]
        mine_W, mine_b = [], []
        tot = sum(BG_PRIOR[g][k] for g in BGN)
        for aspect, cy, fill, p_shape in SHAPE[cls]:
            for bg in BGN:
                for zi, width in enumerate(SIZES):
                    cov = coverage(aspect, cy, fill, width)
                    Wm = np.zeros((C3, 4, 4))
                    for y in range(4):
                        bc, bt = BG[bg][y]
                        for x in range(4):
                            c = cov[y, x]
                            q = c * oc + (1 - c) * bc
                            q = (1 - ETA_C) * q + ETA_C * GENERIC_C
                            t = c * ot + (1 - c) * bt
                            wb = 4 * c * (1 - c) * EDGE_SHARE
                            t = (1 - wb) * t + wb * (oe_bottom if (y == 3 and oe_bottom is not None) else oe)
                            t = (1 - ETA_T) * t + ETA_T * GENERIC_T
                            is_centre = (1 <= y <= 2) and (1 <= x <= 2)
                            Wm[:NCW, y, x] = (N_COL_CENTRE if is_centre else N_COL_BORDER) * np.log(q)
                            Wm[NCW:NCW + NTW, y, x] = (N_TEX_CENTRE if is_centre else N_TEX_BORDER) * np.log(t)
                    mine_W.append(Wm.reshape(-1))
                    mine_b.append(np.log(p_shape) + np.log(BG_PRIOR[bg][k] / tot) + np.log(SIZE_PRIOR[cls][zi]))
        while len(mine_W) < per_class:           # classes with one outline variant: repeat a scene so every class has the same count (max is unchanged)
            mine_W.append(mine_W[0]); mine_b.append(mine_b[0])
        scenes_W += mine_W; scenes_b += mine_b
    net.dense('scene', 'grid', np.stack(scenes_W), scenes_b)
    net.reshape('scene_r', 'scene', [1, 10, per_class])
    net.pool('maxpool', 'scene_max', 'scene_r', [1, per_class])
    net.reshape('cls_scene', 'scene_max', [10])

    def gidx(ch, y, x):
        return ch * 16 + y * 4 + x

    # ---------------- colour words that are more frequent in the centre than around it ----------------
    Wcs = np.zeros((NCW, C3 * 16))
    for c in range(NCW):
        for y in range(4):
            for x in range(4):
                Wcs[c, gidx(c, y, x)] = 0.25 if ((1 <= y <= 2) and (1 <= x <= 2)) else -1.0 / 12
    net.dense('cs_d1', 'grid', Wcs, np.zeros(NCW)); net.relu('cs_r1', 'cs_d1')
    net.dense('cs_d2', 'cs_r1', -N_CS * np.eye(NCW), CAP_CS * np.ones(NCW)); net.relu('cs_r2', 'cs_d2')    # cap - min(N d, cap)
    Wo = np.zeros((10, NCW))
    for k, cls in enumerate(CLASSES):
        q = (1 - ETA_OBJ) * OBJ_C[cls] + ETA_OBJ * GENERIC_C
        Wo[k] = np.log(q / GENERIC_C)
    net.dense('cls_cs', 'cs_r2', -LAM_CS * Wo, LAM_CS * CAP_CS * Wo.sum(1))

    # ---------------- top row of cells against bottom row ----------------
    def row_mean(ch, y, sign=1.0):
        v = np.zeros(C3 * 16)
        for x in range(4):
            v[gidx(ch, y, x)] = sign * 0.25
        return v
    dch = [row_mean(IR + c, 0) + row_mean(IR + c, 3, -1.0) for c in range(3)]     # top minus bottom, per colour channel
    dY = sum(LUMA[c] * dch[c] for c in range(3))
    tb1 = [(dch[0], 0.0), (-dch[0], 0.0), (dch[1], 0.0), (-dch[1], 0.0), (dch[2], 0.0), (-dch[2], 0.0)]
    tb1 += list(ramp_pair(dY, 0.0, 0.05, 0.12)) + list(ramp_pair(-dY, 0.0, 0.05, 0.12))
    net.dense('tb_d1', 'grid', np.stack([w for w, _ in tb1]), [b for _, b in tb1]); net.relu('tb_r1', 'tb_d1')
    dsum = np.r_[np.ones(6), np.zeros(4)]
    tb2 = list(ramp_pair(dsum, 0.0, 0.12, 0.25)) + [(np.r_[np.zeros(6), 1, -1, 0, 0], 0.0), (np.r_[np.zeros(6), 0, 0, 1, -1], 0.0)]
    net.dense('tb_d2', 'tb_r1', np.stack([w for w, _ in tb2]), [b for _, b in tb2]); net.relu('tb_r2', 'tb_d2')   # [a, b, top-brighter, bottom-brighter]
    net.dense('tb_d3', 'tb_r2', np.array([[1, -1, 0, 0], [1, -1, 1, 0], [1, -1, 0, 1.]]), np.array([0, -1, -1.])); net.relu('tb_r3', 'tb_d3')
    # tb_r3 = [colours differ, differ and top brighter, differ and bottom brighter]
    Wt, bt_ = np.zeros((10, 3)), np.zeros(10)
    for k, cls in enumerate(CLASSES):
        Lu, Lt, Lb, Lo = np.log(np.array(P_TB[cls]) / np.sum(P_TB[cls]))
        Wt[k] = LAM_TB * np.array([Lo - Lu, Lt - Lo, Lb - Lo]); bt_[k] = LAM_TB * Lu
    net.dense('cls_tb', 'tb_r3', Wt, bt_)

    # ---------------- long straight edges ----------------
    # five edge pixels of fitting direction on one straight line, two pixels apart (a 9-pixel span), no sideways tolerance beyond the
    # two-pixel width of an edge response; random texture practically never passes (0.3 ^ 5), outlines curved tighter than ~10 px neither
    runs = []                                    # (word channel, [(dx, dy) for the five taps])
    ks = range(-2, 3)
    step = [1, 0, 0, 0, -1]
    runs.append((21, [(2 * k, 0) for k in ks]))                           # level
    runs.append((23, [(2 * k, step[k + 2]) for k in ks]))                 # rising about 14 degrees
    runs.append((24, [(2 * k, -step[k + 2]) for k in ks]))                # falling about 14 degrees
    runs.append((23, [(2 * k, -k) for k in ks]))                          # rising 27 degrees
    runs.append((24, [(2 * k, k) for k in ks]))                           # falling 27 degrees
    runs.append((23, [(2 * k, -2 * k) for k in ks]))                      # rising 45 degrees
    runs.append((24, [(2 * k, 2 * k) for k in ks]))                       # falling 45 degrees
    n_h = len(runs)
    runs.append((22, [(0, 2 * k) for k in ks]))                           # vertical
    runs.append((25, [(step[k + 2], 2 * k) for k in ks]))                 # vertical leaning 14 degrees, rising to the right
    runs.append((26, [(-step[k + 2], 2 * k) for k in ks]))
    runs.append((25, [(-k, 2 * k) for k in ks]))                          # leaning 27 degrees
    runs.append((26, [(k, 2 * k) for k in ks]))
    Wr = np.zeros((len(runs), C3, 9, 9))
    for o, (ch, taps) in enumerate(runs):
        for dx, dy in taps:
            Wr[o, ch, 4 + dy, 4 + dx] = 1
    net.conv('lg_run', 'words', Wr, -4 * np.ones(len(runs))); net.relu('lg_rr', 'lg_run')        # (12, 22, 22)
    net.pool('avgpool', 'lg_share', 'lg_rr', [22, 22])
    # "has one": the share of such positions passes 1 % .. 3 % (one 14-pixel edge gives about 2.5 %, a 20-pixel edge 5 %)
    gh, gv = np.r_[np.ones(n_h), np.zeros(len(runs) - n_h)], np.r_[np.zeros(n_h), np.ones(len(runs) - n_h)]
    lg = list(ramp_pair(gh, 0.0, 0.01, 0.03)) + list(ramp_pair(gv, 0.0, 0.01, 0.03))
    net.dense('lg_d', 'lg_share', np.stack([w for w, _ in lg]), [b for _, b in lg]); net.relu('lg_r', 'lg_d')
    ph, pv = np.array(P_LONG_H), np.array(P_LONG_V)
    lh, lv = np.log(ph / (1 - ph)), np.log(pv / (1 - pv))
    net.dense('cls_long', 'lg_r', LAM_LONG * np.stack([lh, -lh, lv, -lv], axis=1), LAM_LONG * (np.log(1 - ph) + np.log(1 - pv)))

    # ---------------- horizon ----------------
    net.pool('avgpool', 'hz_pool', 'r1', [1, 10], [1, 10])                 # (C1, 30, 3): mean over the left / middle / right third of each row
    hz = []
    hz += list(ramp_pair(e1(25) - e1(24), 0.0, 0.05, 0.13))                # bright above dark
    hz += list(ramp_pair(e1(24) - e1(25), 0.0, 0.05, 0.13))                # dark above bright
    net.conv('hz_c1', 'hz_pool', np.stack([w for w, _ in hz])[:, :, None, None], [b for _, b in hz]); net.relu('hz_r1', 'hz_c1')
    net.conv('hz_c2', 'hz_r1', np.array([[1, -1, 0, 0], [0, 0, 1, -1]], dtype=np.float64)[:, :, None, None], [0, 0])
    net.pool('maxpool', 'hz_mp', 'hz_c2', [3, 1], [1, 1])                  # (2, 28, 3) allow one row of tilt
    Wc = np.zeros((2, 2, 3, 3)); Wc[0, 0, 1, 0] = Wc[0, 0, 1, 2] = 1; Wc[1, 1, 1, 0] = Wc[1, 1, 1, 2] = 1
    net.conv('hz_c3', 'hz_mp', Wc, [-1, -1]); net.relu('hz_r3', 'hz_c3')   # (2, 26, 1): left AND right third
    band = np.zeros((52, 52))
    for p in range(2):
        for m in range(5, 22):                                             # only rows 8..24 of the image
            band[p * 26 + m, p * 26 + m] = 1
    net.dense('hz_band', 'hz_r3', band, np.zeros(52))
    net.reshape('hz_band_r', 'hz_band', [2, 26, 1])
    net.pool('maxpool', 'hz_max', 'hz_band_r', [26, 1])                    # (2, 1, 1)
    pa, pb = np.array(P_HZ_BRIGHT_ABOVE), np.array(P_HZ_DARK_ABOVE)
    Wh = LAM_HZ * np.stack([np.log(pa / (1 - pa)), np.log(pb / (1 - pb))], axis=1)
    bh = LAM_HZ * (np.log(1 - pa) + np.log(1 - pb))
    net.dense('cls_hz', 'hz_max', Wh, bh)

    # ---------------- region that differs from the left / right borders ----------------
    net.pool('avgpool', 'half', 'words', [2, 2], [2, 2])                   # (C3, 15, 15)
    Wp = np.zeros((3, C3, 1, 1)); Wp[0, IR] = Wp[1, IR + 1] = Wp[2, IR + 2] = 1
    net.conv('mk_rgb', 'half', Wp, np.zeros(3))
    n = 15
    Wd = np.zeros((12 * n * n, 3 * n * n))
    for c in range(3):
        for y in range(n):
            ys = [y]                    # border colour of the same row, so a level sky / ground boundary leaves no residue
            for x in range(n):
                i = c * n * n + y * n + x
                for side, cols in enumerate(((0, 1), (n - 2, n - 1))):
                    for sgn, ch in ((1, 6 * side + c), (-1, 6 * side + 3 + c)):
                        o = ch * n * n + y * n + x
                        Wd[o, i] += sgn
                        for yy in ys:
                            for xx in cols:
                                Wd[o, c * n * n + yy * n + xx] -= sgn / (len(ys) * 2.0)
    net.dense('mk_d1', 'mk_rgb', Wd, np.zeros(12 * n * n))
    net.reshape('mk_d1r', 'mk_d1', [12, n, n]); net.relu('mk_r1', 'mk_d1r')
    left, right = np.r_[np.ones(6), np.zeros(6)], np.r_[np.zeros(6), np.ones(6)]
    # summed |R|+|G|+|B| difference to the border colour: sky gradients and grass texture stay below ~0.15,
    # a brown animal on grass or a grey plane on sky is 0.3 or more
    mk = list(ramp_pair(left, 0.0, 0.15, 0.30)) + list(ramp_pair(right, 0.0, 0.15, 0.30))
    net.conv('mk_c2', 'mk_r1', np.stack([w for w, _ in mk])[:, :, None, None], [b for _, b in mk]); net.relu('mk_r2', 'mk_c2')
    net.conv('mk_c3', 'mk_r2', np.array([[1, -1, 1, -1]], dtype=np.float64)[:, :, None, None], [-1.0]); net.relu('mask', 'mk_c3')   # (1, 15, 15)

    # extent of the mask along both axes, and its area
    prof = list(ramp_pair(np.ones(1), 0.0, 0.02, 0.07))
    Wpf = np.stack([w for w, _ in prof])[:, :, None, None]; bpf = [b for _, b in prof]
    net.pool('avgpool', 'rowprof', 'mask', [1, n]); net.conv('row_c1', 'rowprof', Wpf, bpf); net.relu('row_r1', 'row_c1')
    net.conv('row_c2', 'row_r1', np.array([[1, -1]], dtype=np.float64)[:, :, None, None], [0.0]); net.pool('avgpool', 'h_ext', 'row_c2', [n, 1])
    net.pool('avgpool', 'colprof', 'mask', [n, 1]); net.conv('col_c1', 'colprof', Wpf, bpf); net.relu('col_r1', 'col_c1')
    net.conv('col_c2', 'col_r1', np.array([[1, -1]], dtype=np.float64)[:, :, None, None], [0.0]); net.pool('avgpool', 'w_ext', 'col_c2', [1, n])
    net.pool('avgpool', 'area', 'mask', [n, n])
    s = 1 / 0.15
    # units: [ha, hb, wa, wb, ea, eb, ta, tb, sa, sb]
    #   thin:   area < 0.22 * width extent (mean thickness under 22 % of the image height)
    #   sparse: area < 0.5 * (w + h - 0.9), a straight-line stand-in for "fills less than about half of its bounding box"
    net.dense('ext_h', 'h_ext', np.array([[s], [s], [0], [0], [5.0], [5.0], [0], [0], [0.5 / 0.04], [0.5 / 0.04]]),
              np.array([-0.4 * s, -0.4 * s - 1, 0, 0, -0.75, -1.75, 0, -1.0, -0.45 / 0.04, -0.45 / 0.04 - 1]))
    net.dense('ext_w', 'w_ext', np.array([[0], [0], [s], [s], [5.0], [5.0], [0.22 / 0.04], [0.22 / 0.04], [0.5 / 0.04], [0.5 / 0.04]]),
              np.array([0, 0, -0.4 * s, -0.4 * s - 1, 0, 0, 0, 0, 0, 0]))
    net.dense('ext_a', 'area', np.array([[0], [0], [0], [0], [0], [0], [-1 / 0.04], [-1 / 0.04], [-1 / 0.04], [-1 / 0.04]]), np.zeros(10))
    net.add('ext_sum', ['ext_h', 'ext_w', 'ext_a']); net.relu('ext_r1', 'ext_sum')
    def u10(*idx_val):
        v = np.zeros(10)
        for i, x in idx_val:
            v[i] = x
        return v
    Ht, Wdd = u10((0, 1), (1, -1)), u10((2, 1), (3, -1))
    E, Tn, Sp = u10((4, 1), (5, -1)), u10((6, 1), (7, -1)), u10((8, 1), (9, -1))
    net.dense('ext_d2', 'ext_r1', np.stack([Wdd, Ht, Wdd - Ht, Ht - Wdd, E, Tn, Sp]), np.zeros(7)); net.relu('ext_r2', 'ext_d2')
    # ext_r2 = [wide, tall, relu(wide - tall), relu(tall - wide), something_there, thin, sparse]
    W3 = np.zeros((7, 7)); b3 = np.zeros(7)
    for i in range(5):
        W3[i, i] = 1
    W3[5, 2] = W3[5, 5] = 1; b3[5] = -1                       # wide-not-tall AND thin
    W3[6, 0] = 1; W3[6, 2] = -1; W3[6, 6] = 1; b3[6] = -1     # wide-and-tall AND sparse
    net.dense('ext_d3', 'ext_r2', W3, b3); net.relu('ext_r3', 'ext_d3')
    We, be = np.zeros((10, 7)), np.zeros(10)
    for k, cls in enumerate(CLASSES):
        L0, L1, L2, L3, L4, L5, L6 = np.log(np.array(P_EXTENT[cls]) / np.sum(P_EXTENT[cls]))
        We[k] = LAM_EXT * np.array([L6, -L1, L3 - L1 - L6, L4, L1 - L0, L2 - L3, L5 - L6]); be[k] = LAM_EXT * L0
    net.dense('cls_ext', 'ext_r3', We, be)

    # colour words inside the mask, used only when the mask is not "large or cluttered"
    Wi = np.zeros((NCW, C3, 1, 1))
    for c in range(NCW):
        Wi[c, c] = 1
    net.conv('mw_words', 'half', Wi, np.zeros(NCW))
    net.conv('mw_mask', 'mask', np.ones((NCW, 1, 1, 1)), -np.ones(NCW))
    net.add('mw_sum', ['mw_words', 'mw_mask']); net.relu('mw', 'mw_sum')
    net.pool('avgpool', 'mw_hist', 'mw', [n, n])                           # (11, 1, 1) share of the image: in mask and of this colour
    net.dense('oc_d1', 'mw_hist', -N_OBJ * np.eye(NCW), CAP_OBJ * np.ones(NCW)); net.relu('oc_r1', 'oc_d1')   # cap - min(N h, cap)
    net.dense('oc_d2', 'oc_r1', -np.eye(NCW), CAP_OBJ * np.ones(NCW))                                         # min(N h, cap)
    net.dense('oc_gate', 'ext_r2', np.tile(-CAP_OBJ * np.array([1, 0, -1, 0, 0, 0, 0.]), (NCW, 1)), np.zeros(NCW))  # - cap * min(wide, tall)
    net.add('oc_sum', ['oc_d2', 'oc_gate']); net.relu('oc_r2', 'oc_sum')
    net.dense('cls_obj', 'oc_r2', LAM_OBJ * Wo, np.zeros(10))

    net.add('scores', ['cls_scene', 'cls_cs', 'cls_tb', 'cls_long', 'cls_hz', 'cls_ext', 'cls_obj'])
    return net


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(out, exist_ok=True)
    net = build()
    with open(os.path.join(out, 'network.json'), 'w') as f:
        json.dump({"layers": net.layers}, f, indent=1)
    np.savez(os.path.join(out, 'weights.npz'), **net.w)
