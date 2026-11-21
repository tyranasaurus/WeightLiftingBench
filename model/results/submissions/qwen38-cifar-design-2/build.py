"""Build a hand-designed CIFAR-10 network: weights come from formulas and hand-written
rule tables only (no data, no fitting).  Usage: python3 build.py OUT_DIR"""
import json
import sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ---------------------------------------------------------------- stage 1 filters
T = np.arange(-2, 3, dtype=float)


def g1(s):
    g = np.exp(-T ** 2 / (2 * s * s))
    return g / g.sum()


def d1(s):
    d = -T / (s * s) * np.exp(-T ** 2 / (2 * s * s))
    return d / np.linalg.norm(d)


def d2bar(s):
    """even bar detector, positive centre: -g''"""
    e = np.exp(-T ** 2 / (2 * s * s))
    b = -(T ** 2 / s ** 4 - 1 / s ** 2) * e
    return b / np.linalg.norm(b)


def g2(s):
    yy, xx = np.meshgrid(T, T, indexing="ij")
    e = np.exp(-(xx ** 2 + yy ** 2) / (2 * s * s))
    return e / e.sum()


def rot(f_perp, f_par, theta):
    """filter[y,x] = f_perp(u) * f_par(v), u along theta"""
    c, s = np.cos(theta), np.sin(theta)
    yy, xx = np.meshgrid(T, T, indexing="ij")
    u = xx * c + yy * s
    v = -xx * s + yy * c
    out = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            out[i, j] = np.interp(u[i, j], T, f_perp) * np.interp(v[i, j], T, f_par)
    return out / np.linalg.norm(out)


def dog(sc, ss):
    b = g2(sc) - g2(ss)
    return b / np.linalg.norm(b)


SE = 1.1
c1_filters = []  # (name, dict plane->5x5)
for k, th in enumerate([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    e = rot(d1(SE), g1(SE), th)
    c1_filters.append(("edge%dp" % k, {"L": e}))
    c1_filters.append(("edge%dn" % k, {"L": -e}))
for k, th in enumerate([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    b = rot(d2bar(SE), g1(SE), th)
    c1_filters.append(("bar%dp" % k, {"L": b}))
    c1_filters.append(("bar%dn" % k, {"L": -b}))
c1_filters.append(("spotp", {"L": dog(0.8, 2.0)}))
c1_filters.append(("spotn", {"L": -dog(0.8, 2.0)}))
lap = g2(0.7) * 0
e = np.exp(-(np.meshgrid(T, T, indexing="ij")[0] ** 2 + np.meshgrid(T, T, indexing="ij")[1] ** 2) / (2 * 0.49))
logg = (e * (np.meshgrid(T, T, indexing="ij")[0] ** 2 + np.meshgrid(T, T, indexing="ij")[1] ** 2) / 0.49 - 2 * e) * -1.0
logg = logg / np.linalg.norm(logg)
if logg[2, 2] < 0:
    logg = -logg
c1_filters.append(("lapp", {"L": logg}))
c1_filters.append(("lapn", {"L": -logg}))
lp = g2(1.5)
c1_filters.append(("rgp", {"R": lp, "G": -lp}))
c1_filters.append(("rgn", {"R": -lp, "G": lp}))
c1_filters.append(("byp", {"B": lp, "R": -0.5 * lp, "G": -0.5 * lp}))
c1_filters.append(("byn", {"B": -lp, "R": 0.5 * lp, "G": 0.5 * lp}))
sm = g2(2.0)
c1_filters.append(("Rs", {"R": sm}))
c1_filters.append(("Gs", {"G": sm}))
c1_filters.append(("Bs", {"B": sm}))

K1 = len(c1_filters)  # 27
W1 = np.zeros((K1, 3, 5, 5))
for o, (nm, planes) in enumerate(c1_filters):
    for pi, p in enumerate("RGB"):
        if p in planes:
            W1[o, pi] = planes[p]
B1 = np.zeros(K1)

# channel index helpers (stage 1)
ED = list(range(0, 8))        # oriented edges, rectified pairs
BR = list(range(8, 16))       # bars
SP_P, SP_N = 16, 17
LP_P, LP_N = 18, 19
RG_P, RG_N, BY_P, BY_N = 20, 21, 22, 23
RS, GS, BS = 24, 25, 26

# ---------------------------------------------------------------- stage 2 patterns (5x5 masks)
yy, xx = np.meshgrid(T, T, indexing="ij")
RR = np.sqrt(xx ** 2 + yy ** 2)


def mask_mean(m):
    m = m.astype(float)
    return m / m.sum()


PAT = {
    "CTR": mask_mean(RR < 0.5),
    "C3": mask_mean(RR < 1.5),
    "R2": mask_mean((RR >= 1.5) & (RR < 2.5)),
    "RNG": mask_mean(RR >= 1.5),
    "ALL": mask_mean(RR < 9),
    "TOP": mask_mean(yy < -1),
    "BOT": mask_mean(yy > 1),
    "LEFT": mask_mean(xx < -1),
    "RIGHT": mask_mean(xx > 1),
    "CROW": mask_mean(np.abs(yy) < 0.5),
    "CCOL": mask_mean(np.abs(xx) < 0.5),
    "TOP3": mask_mean(yy < 1),
    "BOT3": mask_mean(yy > -1),
}

# stage 2 channels: list of (c1ch, pattern, weight) plus bias
c2 = []


def C(terms, bias):
    c2.append((terms, bias))


def mean_term(chs, pat, w):
    return [(c, pat, w / len(chs)) for c in chs]


# corners: two orthogonal rectified edges meeting => both needed to cross threshold
C([(4, "TOP", 1.0), (0, "LEFT", 1.0)], -0.50)          # 0 corner TL, bright inside
C([(4, "TOP", 1.0), (1, "RIGHT", 1.0)], -0.50)         # 1 TR bright inside
C([(5, "BOT", 1.0), (0, "LEFT", 1.0)], -0.50)          # 2 BL bright inside
C([(5, "BOT", 1.0), (1, "RIGHT", 1.0)], -0.50)         # 3 BR bright inside
C([(5, "TOP", 1.0), (1, "LEFT", 1.0)], -0.50)          # 4 TL dark inside
C([(5, "TOP", 1.0), (0, "RIGHT", 1.0)], -0.50)         # 5 TR dark inside
C([(4, "BOT", 1.0), (1, "LEFT", 1.0)], -0.50)          # 6 BL dark inside
C([(4, "BOT", 1.0), (0, "RIGHT", 1.0)], -0.50)         # 7 BR dark inside
C([(SP_N, "CTR", 1.2), (LP_P, "R2", 0.2), (LP_N, "R2", 0.2), (RS, "CTR", -0.10), (GS, "CTR", -0.10), (BS, "CTR", -0.10)], -0.20)   # 8 small dark blob (wheel/eye)
C([(SP_N, "CTR", 0.8)] + mean_term(ED, "RNG", 0.5) + [(RS, "CTR", -0.08), (GS, "CTR", -0.08), (BS, "CTR", -0.08)], -0.18)            # 9 dark blob with edge ring
C([(SP_P, "CTR", 1.2), (RS, "CTR", 0.13), (GS, "CTR", 0.13), (BS, "CTR", 0.13)], -0.35)   # 10 bright highlight
C([(SP_N, "CTR", 1.2), (RS, "R2", 0.17), (GS, "R2", 0.17), (BS, "R2", 0.17), (RS, "CTR", -0.08), (GS, "CTR", -0.08), (BS, "CTR", -0.08)], -0.25)  # 11 eye: dark centre, lighter surround
C([(BY_P, "CTR", 0.8), (RS, "CTR", 0.10), (GS, "CTR", 0.10), (BS, "CTR", 0.10)] + mean_term(ED, "ALL", -0.5) + mean_term([LP_P, LP_N], "ALL", -0.4), -0.22)  # 12 smooth blue (sky)
C(mean_term(ED + BR, "ALL", -0.6) + mean_term([LP_P, LP_N], "ALL", -0.5), 0.10)           # 13 smooth flat area
C([(BY_P, "CTR", 0.5), (RG_N, "CTR", 0.4), (4, "ALL", 0.10), (5, "ALL", 0.10)] + mean_term([0, 1], "ALL", -0.12) + mean_term([LP_P, LP_N], "ALL", -0.2), -0.10)  # 14 water
C(mean_term([LP_P, LP_N], "ALL", 0.7) + mean_term(ED, "ALL", 0.4) + mean_term(BR, "ALL", -0.6), -0.16)   # 15 broadband natural texture
C([(10, "CROW", 0.5), (11, "CROW", 0.5), (4, "CROW", 0.2), (5, "CROW", 0.2)] + mean_term([0, 1], "ALL", -0.2), -0.30)  # 16 strong horizontal line
C([(8, "CCOL", 0.5), (9, "CCOL", 0.5), (0, "CCOL", 0.2), (1, "CCOL", 0.2)] + mean_term([4, 5], "ALL", -0.2), -0.30)    # 17 strong vertical line
C([(SP_N, "CTR", 0.8), (RS, "R2", 0.13), (GS, "R2", 0.13), (BS, "R2", 0.13), (RS, "CTR", -0.17), (GS, "CTR", -0.17), (BS, "CTR", -0.17)], -0.20)  # 18 dark patch, bright frame
C(mean_term([12, 13, 14, 15], "ALL", 0.8) + mean_term([2, 3, 6, 7], "ALL", 0.3) + mean_term([10, 11, 8, 9], "ALL", -0.3), -0.25)  # 19 diagonal elongated structure
C([(10, "CROW", 0.9), (RS, "CROW", 0.10), (GS, "CROW", 0.10), (BS, "CROW", 0.10)] + mean_term([LP_P, LP_N], "ALL", -0.3), -0.25)  # 20 bright horizontal band
C([(RS, "CTR", 0.23), (GS, "CTR", 0.23), (BS, "CTR", 0.23), (SP_P, "CTR", 0.3)], -0.55)   # 21 bright mass
C([(RS, "CTR", -0.30), (GS, "CTR", -0.30), (BS, "CTR", -0.30)], -0.45)                    # 22 dark mass
C([(RG_P, "CTR", 0.9), (BY_P, "CTR", -0.2)], -0.25)                                       # 23 warm red
C([(RG_P, "CTR", 0.6), (BY_N, "CTR", 0.5), (RS, "CTR", -0.07), (GS, "CTR", -0.07), (BS, "CTR", -0.07)], -0.22)  # 24 brown / tan
C([(RG_N, "CTR", 0.9), (BY_P, "CTR", -0.2)], -0.20)                                       # 25 green
C([(RG_P, "CTR", -0.25), (RG_N, "CTR", -0.25), (BY_P, "CTR", -0.25), (BY_N, "CTR", -0.25)], -0.12)  # 26 neutral grey
C([(RG_P, "CTR", -0.2), (RG_N, "CTR", -0.2), (BY_P, "CTR", -0.2), (BY_N, "CTR", -0.2), (RS, "CTR", 0.2), (GS, "CTR", 0.2), (BS, "CTR", 0.2)], -0.42)  # 27 white neutral
C([(RS, "CTR", -0.4), (GS, "CTR", -0.4), (BS, "CTR", -0.4)], -0.30)                       # 28 near black
C(mean_term([0, 1], "ALL", 0.6) + mean_term([4, 5], "ALL", -0.25), -0.10)                 # 29 dense vertical edges
C(mean_term([4, 5], "ALL", 0.6) + mean_term([0, 1], "ALL", -0.25), -0.10)                 # 30 dense horizontal edges
C([(BY_P, "CTR", 0.7), (RS, "CTR", 0.07), (GS, "CTR", 0.07), (BS, "CTR", 0.07)] + mean_term([LP_P, LP_N], "ALL", -0.3) + mean_term(ED, "ALL", -0.2), -0.15)  # 31 permissive sky

K2 = len(c2)
W2 = np.zeros((K2, K1, 5, 5))
B2 = np.zeros(K2)

# --- data-free calibration of stage-2 thresholds -------------------------------
# For every stage-1 channel group we posit (from general knowledge of photos, not from
# data) a background level and a "feature present" level of the mask-meaned rectified
# response; a stage-2 unit's bias then makes it fire when its positive evidence exceeds
# background by a fixed fraction of the feature level.
COV = {"CTR": 1.0, "C3": 0.5, "R2": 0.5, "RNG": 0.35, "ALL": 0.25, "TOP": 0.35, "BOT": 0.35,
       "LEFT": 0.35, "RIGHT": 0.35, "CROW": 0.4, "CCOL": 0.4, "TOP3": 0.3, "BOT3": 0.3}
BG_PIX = {"ED": 0.06, "BAR": 0.04, "SPOT": 0.03, "LAP": 0.06}
T_PIX = {"ED": 0.30, "BAR": 0.25, "SPOT": 0.20, "LAP": 0.25, "CHR": 0.25, "BRI": 0.35}


def grp(c):
    if c in ED:
        return "ED"
    if c in BR:
        return "BAR"
    if c in (SP_P, SP_N):
        return "SPOT"
    if c in (LP_P, LP_N):
        return "LAP"
    if c in (RG_P, RG_N, BY_P, BY_N):
        return "CHR"
    return "BRI"


def z0(g, pat):
    if g == "CHR":
        return 0.0
    if g == "BRI":
        return 0.45
    return BG_PIX[g] * COV[pat]


for o, (terms, bias) in enumerate(c2):
    for (c, pat, w) in terms:
        W2[o, c] += w * PAT[pat]
    b = 0.0
    for (c, pat, w) in terms:
        g = grp(c)
        b -= w * z0(g, pat)
        if w > 0:
            b -= 0.55 * w * T_PIX[g] * COV[pat]
    B2[o] = b

# ---------------------------------------------------------------- network graph
layers = [
    {"name": "c1", "op": "conv", "inputs": ["image"], "out": K1, "kernel": [5, 5], "stride": 1, "padding": 2},
    {"name": "r1", "op": "relu", "inputs": ["c1"]},
    {"name": "c2", "op": "conv", "inputs": ["r1"], "out": K2, "kernel": [5, 5], "stride": 1, "padding": 2},
    {"name": "r2", "op": "relu", "inputs": ["c2"]},
    {"name": "n2", "op": "norm", "inputs": ["r2"]},
    {"name": "r2n", "op": "relu", "inputs": ["n2"]},
    {"name": "pa", "op": "avgpool", "inputs": ["r1"], "size": [8, 8], "stride": [8, 8]},
    {"name": "pm", "op": "maxpool", "inputs": ["r1"], "size": [8, 8], "stride": [8, 8]},
    {"name": "qa", "op": "avgpool", "inputs": ["r1"], "size": [16, 16], "stride": [16, 16]},
    {"name": "g1", "op": "avgpool", "inputs": ["r1"], "size": [32, 32], "stride": [32, 32]},
    {"name": "p2a", "op": "avgpool", "inputs": ["r2n"], "size": [8, 8], "stride": [8, 8]},
    {"name": "p2m", "op": "maxpool", "inputs": ["r2n"], "size": [8, 8], "stride": [8, 8]},
    {"name": "q2a", "op": "avgpool", "inputs": ["r2n"], "size": [16, 16], "stride": [16, 16]},
    {"name": "g2", "op": "avgpool", "inputs": ["r2n"], "size": [32, 32], "stride": [32, 32]},
]
BRANCHES = {"pa": (K1, 16), "pm": (K1, 16), "qa": (K1, 4), "g1": (K1, 1),
            "p2a": (K2, 16), "p2m": (K2, 16), "q2a": (K2, 4), "g2": (K2, 1)}

# ---------------------------------------------------------------- concept atoms
# term: (branch, channel-or-macro, cells, weight); cells: list of (r,c) or None for global
MACRO1 = {"BRI": [(RS, 1 / 3), (GS, 1 / 3), (BS, 1 / 3)],
          "EDG": [(c, 1 / 8) for c in ED],
          "BAR": [(c, 1 / 8) for c in BR],
          "LAP": [(LP_P, 0.5), (LP_N, 0.5)],
}
MACRO2 = {"CORN": [(c, 1 / 8) for c in range(8)]}

# baseline of pooled relu features, used to auto-set atom biases
def base1(ch, pool):
    if ch in ED:
        return {"avg8": 0.06, "max8": 0.25, "avg16": 0.06, "g": 0.06}[pool]
    if ch in BR:
        return {"avg8": 0.04, "max8": 0.20, "avg16": 0.04, "g": 0.04}[pool]
    if ch in (SP_P, SP_N):
        return {"avg8": 0.03, "max8": 0.15, "avg16": 0.03, "g": 0.03}[pool]
    if ch in (LP_P, LP_N):
        return {"avg8": 0.06, "max8": 0.25, "avg16": 0.06, "g": 0.06}[pool]
    if ch in (RG_P, RG_N, BY_P, BY_N):
        return 0.0
    return {"avg8": 0.45, "max8": 0.80, "avg16": 0.45, "g": 0.45}[pool]


POOLK = {"pa": "avg8", "pm": "max8", "qa": "avg16", "g1": "g",
         "p2a": "zavg", "p2m": "zmax", "q2a": "zavg", "g2": "zavg"}
ZBASE = {"zavg": 0.40, "zmax": 1.90}

atoms = []  # (name, terms, margin)


def A(name, terms, margin=0.25):
    atoms.append((name, terms, margin))


def c4(rows, cols=(0, 1, 2, 3)):
    return [(r, c) for r in rows for c in cols]


CTR4 = [(1, 1), (1, 2), (2, 1), (2, 2)]
BORD4 = [(r, c) for r in range(4) for c in range(4) if r in (0, 3) or c in (0, 3)]
Q2 = [(0, 0), (0, 1), (1, 0), (1, 1)]
R0 = [(0, 0), (0, 1)]
R1 = [(1, 0), (1, 1)]
# neutrality = low chroma: rectified chroma channels are >= 0, so "achromatic" must be
# written with NEGATIVE weights on them and a negative margin (fires when they are small)
NEU1 = [(RG_P, -0.3), (RG_N, -0.3), (BY_P, -0.3), (BY_N, -0.3)]


def neu(br, cells, w):
    return [(br, c, cells, w * cw) for (c, cw) in NEU1]


# Every atom is a within-image contrast (row vs row, centre vs border, channel vs
# channel); margins are hand-set excesses over the auto-computed feature-less baseline.
A("sky_top", [("qa", BY_P, R0, 2.0), ("qa", BY_P, R1, -1.0), ("q2a", 12, R0, 1.2),
              ("q2a", 12, R1, -0.6), ("q2a", 31, R0, 0.8), ("q2a", 31, R1, -0.4),
              ("g1", BY_P, None, -0.5)], 0.20)
A("sky_all", [("g1", BY_P, None, 3.0), ("g1", RG_N, None, -0.5)], 0.30)
A("water_bot", [("qa", BY_P, R1, 2.0), ("qa", BY_P, R0, -1.0), ("qa", RG_N, R1, 1.5),
                ("qa", RG_N, R0, -0.75), ("pa", "BRI", c4([2, 3]), -0.8),
                ("pa", "BRI", c4([0, 1]), 0.4), ("q2a", 14, R1, 1.5), ("q2a", 14, R0, -0.75),
                ("q2a", 30, R1, 0.8), ("q2a", 30, R0, -0.4)], 0.25)
A("water_any", [("q2a", 14, R1, 1.2), ("q2a", 14, R0, 0.6), ("g1", BY_P, None, 0.5),
                ("g1", RG_N, None, 0.4)], 0.25)
A("green_bg", [("g1", RG_N, None, 2.0), ("g1", RG_P, None, -1.0), ("q2a", 25, R0, 0.8),
               ("q2a", 25, R1, -0.4)], 0.20)
A("green_bot", [("q2a", 25, R1, 1.4), ("q2a", 25, R0, -0.7), ("qa", RG_N, R1, 1.2),
                ("qa", RG_N, R0, -0.6)], 0.20)
A("road_bot", [("q2a", 26, R1, 1.4), ("q2a", 26, R0, -0.7)] + neu("qa", R1, 1.0) +
              neu("qa", R0, -0.5) + [("q2a", 30, R1, 0.6), ("q2a", 30, R0, -0.3),
                                     ("q2a", 25, R1, -0.8), ("q2a", 25, R0, 0.4)], 0.20)
A("grey_all", neu("g1", None, 1.5) + [("q2a", 26, R1, 0.8), ("q2a", 26, R0, -0.4)], -0.10)
A("white_bg", [("g1", "BRI", None, 1.0)] + neu("g1", None, 0.8) +
              [("g1", "LAP", None, -1.0), ("g1", "EDG", None, 0.6), ("q2a", 27, R0, 0.8),
               ("q2a", 27, R1, -0.4)], 0.15)
A("dark_bg", [("g1", "BRI", None, -1.0)], 0.20)
A("fur", [("g1", "LAP", None, 8.0), ("g1", "EDG", None, -4.8), ("q2a", 13, R1, -1.0),
          ("q2a", 13, R0, 0.5), ("q2a", 16, R1, -0.8), ("q2a", 16, R0, 0.4)], 0.10)
A("smooth_all", [("g1", "EDG", None, -6.0), ("g1", "LAP", None, 3.0)], -0.10)
A("manmade", [("q2a", 16, R1, 1.0), ("q2a", 16, R0, -0.5), ("q2a", 17, R0, 1.0),
              ("q2a", 17, R1, -0.5), ("q2a", "CORN", R0, 0.8), ("q2a", "CORN", R1, -0.4)], 0.35)
A("wheels", [("q2a", 8, R1, 1.4), ("q2a", 8, R0, -0.7), ("q2a", 9, R1, 1.0), ("q2a", 9, R0, -0.5),
             ("p2m", 8, c4([2, 3]), 0.8), ("p2m", 8, c4([0, 1]), -0.4),
             ("q2a", 28, R1, 0.6), ("q2a", 28, R0, -0.3),
             ("pa", 17, c4([2, 3]), 0.8), ("pa", 17, c4([0, 1]), -0.4)], 0.30)
A("windows", [("q2a", 18, Q2, 1.3), ("p2m", 18, c4([1, 2]), 0.8), ("p2m", 18, c4([3]), -0.4)], 0.50)
A("glossy", [("q2a", 13, R1, 1.3), ("q2a", 13, R0, -0.6), ("q2a", 10, R1, 0.8),
             ("q2a", 10, R0, -0.4), ("q2a", 27, R1, 0.6), ("q2a", 27, R0, -0.3)], 0.45)
A("hbar_mid", [("q2a", 20, R1, 1.4), ("q2a", 20, R0, -0.7), ("qa", 10, R1, 1.0),
               ("qa", 10, R0, -0.5)], 0.40)
A("boxy", [("q2a", "CORN", R0, 1.3), ("q2a", "CORN", R1, -0.6),
           ("p2m", "CORN", c4([0, 1]), 0.8), ("p2m", "CORN", c4([2, 3]), -0.4)], 0.50)
A("face_ctr", [("p2m", 11, CTR4, 1.0), ("p2m", 11, BORD4, -0.6), ("p2a", 15, CTR4, 0.6),
               ("p2a", 15, BORD4, -0.3), ("q2a", 16, R1, -0.5), ("q2a", 16, R0, 0.25),
               ("q2a", 8, R1, -0.4), ("q2a", 8, R0, 0.2)], 1.30)
A("eyes_pair", [("p2m", 11, [(1, 1), (1, 2)], 1.0), ("p2m", 11, c4([3]), -0.5)], 1.60)
A("dark_bot", [("pa", "BRI", c4([2, 3]), -1.5), ("pa", "BRI", c4([0, 1]), 0.8),
               ("pa", 17, c4([2, 3]), 1.0), ("pa", 17, c4([0, 1]), -0.5)], 0.05)
A("ears_top", [("q2a", 19, R0, 1.3), ("q2a", 19, R1, -0.6), ("q2a", "CORN", R0, 0.8),
               ("q2a", "CORN", R1, -0.4)], 0.40)
A("snout", [("p2m", 17, [(2, 1), (2, 2)], 0.9), ("p2m", 17, c4([0]), -0.45),
            ("q2a", 22, Q2, 0.6)], 0.70)
A("legs", [("q2a", 29, R1, 1.3), ("q2a", 29, R0, -0.6), ("q2a", 17, R1, 0.8),
           ("q2a", 17, R0, -0.4), ("p2a", 29, c4([2, 3]), 0.6), ("p2a", 29, c4([0, 1]), -0.3)], 0.40)
A("body_hbar", [("q2a", 16, R1, 1.3), ("q2a", 16, R0, -0.6), ("q2a", 20, R1, 0.8),
                ("q2a", 20, R0, -0.4), ("qa", 10, R1, 0.8), ("qa", 10, R0, -0.4)], 0.40)
A("brown", [("g1", RG_P, None, 1.6), ("g1", BY_N, None, 1.2), ("g1", RG_N, None, -0.8),
            ("g1", "BRI", None, -0.6), ("qa", RG_P, R1, 0.6), ("qa", RG_P, R0, -0.3)], 0.15)
A("greenbody", [("g1", RG_N, None, 2.0), ("g1", RG_P, None, -1.0), ("q2a", 25, R1, 0.8),
                ("q2a", 25, R0, 0.4)], 0.20)
A("whitebody", [("q2a", 27, R1, 1.3), ("q2a", 27, R0, -0.6), ("g1", "BRI", None, 0.8)] +
               neu("g1", None, 0.6), 0.25)
A("darkbody", [("q2a", 22, R1, 1.3), ("q2a", 22, R0, -0.6), ("q2a", 28, R1, 0.8),
               ("q2a", 28, R0, -0.4)], 0.45)
A("redbody", [("g1", RG_P, None, 2.0), ("g1", RG_N, None, -1.0), ("g1", BY_N, None, -0.5)], 0.15)
A("orange", [("g1", RG_P, None, 1.5), ("g1", BY_N, None, 1.0), ("g1", RG_N, None, -1.0),
             ("g1", "BRI", None, 0.3)], 0.15)
A("small_obj", [("pa", "EDG", CTR4, 1.5), ("pa", "EDG", BORD4, -1.0)], 0.04)
A("fills", [("pa", "EDG", BORD4, 3.0), ("pa", "EDG", CTR4, -1.5)], 0.02)
A("horizon", [("p2m", 16, c4([2]), 1.0), ("p2m", 16, c4([0]), -0.5), ("q2a", 16, R1, 0.8),
              ("q2a", 16, R0, -0.4)], 0.60)
A("sky_obj", [("qa", BY_P, R0, 1.5), ("qa", BY_P, R1, -0.8), ("qa", "EDG", R1, 1.0),
              ("qa", "EDG", R0, -0.5)], 0.20)
A("sail", [("q2a", 27, R0, 1.0), ("q2a", 27, R1, -0.5), ("q2a", 19, Q2, 1.0),
           ("q2a", 14, R1, 0.8), ("q2a", 14, R0, -0.4)], 0.45)
A("spec", [("p2m", 10, c4([1, 2]), 1.0), ("p2m", 10, c4([0, 3]), -0.5)], 1.00)
A("hull", [("pa", "BRI", c4([2, 3]), -1.2), ("pa", "BRI", c4([0, 1]), 0.6),
           ("q2a", 22, R1, 1.3), ("q2a", 22, R0, -0.6), ("q2a", 16, R1, 0.8),
           ("q2a", 16, R0, -0.4), ("q2a", 14, R1, 0.8), ("q2a", 14, R0, -0.4),
           ("q2a", 25, R1, -0.6), ("q2a", 25, R0, 0.3)], 0.30)
A("truckbox", [("q2a", 17, R0, 1.3), ("q2a", 17, R1, -0.6), ("q2a", 16, R0, 0.8),
               ("q2a", 16, R1, -0.4), ("q2a", "CORN", R0, 0.8), ("q2a", "CORN", R1, -0.4),
               ("q2a", 23, R0, 0.5), ("q2a", 23, R1, -0.25)], 0.40)
A("beak", [("p2m", 23, CTR4, 0.9), ("p2m", 23, BORD4, -0.45), ("p2m", 24, CTR4, 0.5),
           ("p2m", 24, BORD4, -0.25)], 0.70)
A("big_side", [("qa", "EDG", R1, 1.5), ("qa", "EDG", R0, -0.7), ("q2a", 16, R1, 0.8),
               ("q2a", 16, R0, -0.4), ("q2a", 25, R1, -0.5), ("q2a", 25, R0, 0.25)], 0.25)
A("indoor", [("g1", "BRI", None, 0.8), ("g1", "LAP", None, -3.0), ("g1", "EDG", None, 1.8),
             ("g1", RG_N, None, -1.5), ("q2a", 13, R0, 0.6), ("q2a", 13, R1, -0.3)], 0.10)
A("head_big", [("p2m", 11, c4([1]), 1.0), ("p2m", 11, c4([3]), -0.5),
               ("p2m", 17, c4([1, 2]), 0.6), ("p2m", 17, c4([0, 3]), -0.3),
               ("pa", "EDG", CTR4, 0.8), ("pa", "EDG", BORD4, -0.4)], 0.70)

H = len(atoms)

# expand atoms into per-branch dense weight matrices.
# A term (branch, ch, cells, w) spreads w/len(cells) over the named grid cells of channel ch.
# The atom bias cancels the expected "baseline" response (what the pooled terms give on a
# feature-less image) plus a hand-set margin, so an atom fires only on real evidence.
Wd = {b: np.zeros((H, BRANCHES[b][0] * BRANCHES[b][1])) for b in BRANCHES}
Bd = {b: np.zeros(H) for b in BRANCHES}


def chans_of(ch):
    if isinstance(ch, str):
        return MACRO1[ch] if ch in MACRO1 else MACRO2[ch]
    return [(ch, 1.0)]


for h, (name, terms, margin) in enumerate(atoms):
    base = 0.0
    for (br, ch, cells, w) in terms:
        pk = POOLK[br]
        for (c, cw) in chans_of(ch):
            ww = w * cw
            if br in ("g1", "g2"):
                Wd[br][h, c] += ww
            else:
                gs = int(round(np.sqrt(BRANCHES[br][1])))
                assert all(0 <= r < gs and 0 <= cc < gs for r, cc in cells), (br, cells)
                for (r, cc) in cells:
                    Wd[br][h, c * BRANCHES[br][1] + r * gs + cc] += ww / len(cells)
            # expected response of this term on a feature-less image
            if pk == "zavg":
                base += ww * 0.45
            elif pk == "zmax":
                base += ww * 2.80
            else:
                base += ww * base1(c, pk)
    Bd["pa"][h] = -base - margin  # single bias per atom, carried by the first branch

for b, nm in [("pa", "dpa"), ("pm", "dpm"), ("qa", "dqa"), ("g1", "dg1"),
              ("p2a", "dp2a"), ("p2m", "dp2m"), ("q2a", "dq2a"), ("g2", "dg2")]:
    layers.append({"name": nm, "op": "dense", "inputs": [b], "out": H})
layers.append({"name": "hadd", "op": "add",
               "inputs": ["dpa", "dpm", "dqa", "dg1", "dp2a", "dp2m", "dq2a", "dg2"]})
layers.append({"name": "hrelu", "op": "relu", "inputs": ["hadd"]})
layers.append({"name": "dout", "op": "dense", "inputs": ["hrelu"], "out": 10})

AN = {nm: i for i, (nm, _, _) in enumerate(atoms)}


def rule(pairs, bias=None):
    row = np.zeros(H)
    pos = 0.0
    for nm, w in pairs:
        row[AN[nm]] = w
        if w > 0:
            pos += w
    row *= 6.0 / pos  # equal positive mass so the common bias is fair across classes
    b = -2.10 if bias is None else bias
    return row, b


RULES = [
    [("sky_all", 1.6), ("sky_top", 1.0), ("hbar_mid", 1.1), ("manmade", 0.8), ("whitebody", 0.6),
     ("sky_obj", 0.8), ("smooth_all", 0.4), ("green_bg", -0.8), ("water_bot", -0.6),
     ("wheels", -0.6), ("face_ctr", -0.8), ("fur", -0.6)],
    [("wheels", 1.5), ("dark_bot", 1.0), ("windows", 1.2), ("glossy", 1.0), ("road_bot", 1.2),
     ("grey_all", 0.7), ("spec", 0.5), ("sky_all", -0.8), ("green_bg", -0.6), ("truckbox", -0.9),
     ("face_ctr", -0.6), ("fur", -0.6)],
    [("small_obj", 1.5), ("beak", 1.2), ("face_ctr", 0.7), ("sky_top", 0.6), ("green_bg", 0.5),
     ("white_bg", 0.4), ("sky_obj", 0.5), ("fills", -0.8), ("wheels", -0.8), ("fur", -0.5),
     ("dark_bot", -0.3)],
    [("face_ctr", 1.5), ("eyes_pair", 0.9), ("ears_top", 1.0), ("fur", 1.2), ("orange", 0.8),
     ("fills", 0.7), ("green_bg", -0.7), ("wheels", -0.6), ("dark_bot", -0.4)],
    [("brown", 1.8), ("green_bg", 1.2), ("legs", 1.0), ("body_hbar", 0.7), ("green_bot", 0.4),
     ("face_ctr", -0.6), ("wheels", -0.5), ("fills", -0.3), ("big_side", -0.4)],
    [("face_ctr", 1.3), ("fur", 1.1), ("indoor", 0.9), ("whitebody", 0.7), ("eyes_pair", 0.6),
     ("snout", 0.6), ("green_bg", -0.6), ("wheels", -0.6), ("dark_bot", -0.4), ("orange", -0.4)],
    [("greenbody", 2.2), ("fills", 1.0), ("face_ctr", 0.6), ("green_bg", 0.5), ("eyes_pair", 0.4),
     ("smooth_all", 0.5), ("legs", -0.6), ("body_hbar", -0.5), ("wheels", -0.4)],
    [("body_hbar", 1.3), ("legs", 1.2), ("green_bot", 1.2), ("big_side", 1.0), ("brown", 0.7),
     ("darkbody", 0.6), ("whitebody", 0.5), ("head_big", 0.5), ("wheels", -0.7),
     ("grey_all", -0.3)],
    [("water_bot", 1.8), ("hull", 1.4), ("water_any", 0.8), ("horizon", 0.8), ("sail", 0.8),
     ("whitebody", 0.5), ("sky_top", 0.3), ("green_bg", -1.0), ("face_ctr", -0.6), ("fur", -0.4)],
    [("truckbox", 1.6), ("wheels", 1.3), ("dark_bot", 0.8), ("redbody", 0.9), ("road_bot", 0.8),
     ("boxy", 0.8), ("whitebody", 0.5), ("windows", -0.7), ("glossy", -0.4)],
]
Wo = np.zeros((10, H))
Bo = np.zeros(10)
for k, pairs in enumerate(RULES):
    Wo[k], Bo[k] = rule(pairs)

# ---------------------------------------------------------------- write files
import os
os.makedirs(OUT, exist_ok=True)
weights = {"c1.W": W1.astype(np.float32), "c1.b": B1.astype(np.float32),
           "c2.W": W2.astype(np.float32), "c2.b": B2.astype(np.float32)}
for b, nm in [("pa", "dpa"), ("pm", "dpm"), ("qa", "dqa"), ("g1", "dg1"),
              ("p2a", "dp2a"), ("p2m", "dp2m"), ("q2a", "dq2a"), ("g2", "dg2")]:
    weights[nm + ".W"] = Wd[b].astype(np.float32)
    weights[nm + ".b"] = Bd[b].astype(np.float32)
weights["dout.W"] = Wo.astype(np.float32)
weights["dout.b"] = Bo.astype(np.float32)
np.savez(os.path.join(OUT, "weights.npz"), **weights)
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump({"layers": layers}, f)
print("wrote", OUT, "atoms", H, "weights", sum(v.size for v in weights.values()))
