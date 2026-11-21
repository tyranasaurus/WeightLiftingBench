"""Writes network.json and weights.npz for a hand-designed person / no-person classifier.

Every number below is chosen by hand from general knowledge about skin colour, faces and
photographs; nothing is read from files and nothing is fitted to data.
Usage: python3 build.py OUT_DIR
"""
import json, os, sys
import numpy as np

LAYERS, WEIGHTS = [], {}
H = W_ = 96


def _add_layer(**kw):
    LAYERS.append(kw)
    return kw["name"]


def conv(name, inp, W, b, stride=1, padding=0):
    W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
    WEIGHTS[name + ".W"], WEIGHTS[name + ".b"] = W, b
    return _add_layer(name=name, op="conv", inputs=[inp], out=int(W.shape[0]),
                      kernel=[int(W.shape[2]), int(W.shape[3])], stride=stride, padding=padding)


def dense(name, inp, W, b):
    W = np.asarray(W, np.float64); b = np.asarray(b, np.float64)
    WEIGHTS[name + ".W"], WEIGHTS[name + ".b"] = W, b
    return _add_layer(name=name, op="dense", inputs=[inp], out=int(W.shape[0]))


def relu(name, inp):
    return _add_layer(name=name, op="relu", inputs=[inp])


def avgpool(name, inp, size, stride):
    return _add_layer(name=name, op="avgpool", inputs=[inp], size=list(size), stride=list(stride))


def maxpool(name, inp, size, stride):
    return _add_layer(name=name, op="maxpool", inputs=[inp], size=list(size), stride=list(stride))


def add(name, inputs):
    return _add_layer(name=name, op="add", inputs=list(inputs))


def reshape(name, inp, shape):
    return _add_layer(name=name, op="reshape", inputs=[inp], shape=list(shape))


# ==============================================================================================
# Stage 1: per-pixel colour analysis.
# ==============================================================================================
# Piecewise-linear natural log of each channel (knots a factor ~1.5 apart), so colour ratios
# become differences:  u = ln(R/G) (redness),  v = ln(G/B) (yellowness).  Brightness-free.
KNOTS = [0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.85]
NK = len(KNOTS) + 1          # per channel: the raw value + one hinge per knot


def log_coeffs():
    """ln(c) ~= ln(K0) + sum_i d_i * relu(c - K_i) for c in [K0, 1]."""
    ks = KNOTS + [1.0]
    slopes = [(np.log(ks[i + 1]) - np.log(ks[i])) / (ks[i + 1] - ks[i]) for i in range(len(KNOTS))]
    d = [slopes[0]] + [slopes[i] - slopes[i - 1] for i in range(1, len(slopes))]
    return np.log(KNOTS[0]), np.array(d)


c1W = np.zeros((3 * NK, 3, 1, 1)); c1b = np.zeros(3 * NK)
for ch in range(3):
    c1W[ch * NK, ch] = 1.0
    for i, t in enumerate(KNOTS):
        c1W[ch * NK + 1 + i, ch] = 1.0; c1b[ch * NK + 1 + i] = -t
col1 = relu("col1r", conv("col1", "image", c1W, c1b))
LN0, LND = log_coeffs()

# A linear expression over col1 channels is stored as (w over channels, constant, cu, cv):
# cu / cv record how much of u / v it contains, so the white-balance shift can be applied.
ZERO = np.zeros(3 * NK)


def E(w=None, c=0.0, cu=0.0, cv=0.0):
    return (ZERO.copy() if w is None else w, c, cu, cv)


def comb(*terms):
    w = ZERO.copy(); c = cu = cv = 0.0
    for k, (tw, tc, tu, tv) in terms:
        w = w + k * tw; c += k * tc; cu += k * tu; cv += k * tv
    return (w, c, cu, cv)


def _log(ch):
    w = ZERO.copy(); w[ch * NK + 1: ch * NK + 1 + len(KNOTS)] = LND
    return (w, LN0, 0.0, 0.0)


def _raw(ch):
    w = ZERO.copy(); w[ch * NK] = 1.0
    return (w, 0.0, 0.0, 0.0)


R, G, B = _raw(0), _raw(1), _raw(2)
_u = comb((1, _log(0)), (-1, _log(1))); U = (_u[0], _u[1], 1.0, 0.0)
_v = comb((1, _log(1)), (-1, _log(2))); V = (_v[0], _v[1], 0.0, 1.0)
Y = comb((0.299, R), (0.587, G), (0.114, B))

# --- partial grey-world white balance ------------------------------------------------------
# The photo's mean u and v are measured, and every pixel's u, v are shifted by
# -WB_ALPHA * (mean - typical).  This removes most of a warm (tungsten) or cool (shade, snow)
# colour cast, so skin keeps its usual hue; in photos that are warm all over (wooden rooms,
# food, desert) it also makes warm surfaces look less skin-like.
WB_ALPHA, U_TYP, V_TYP, OFF = 0.5, 0.05, 0.05, 3.6
uvW = np.zeros((2, 3 * NK, 1, 1)); uvb = np.zeros(2)
uvW[0, :, 0, 0] = U[0]; uvb[0] = U[1] + OFF
uvW[1, :, 0, 0] = V[0]; uvb[1] = V[1] + OFF
uv = relu("uvr", conv("uv", col1, uvW, uvb))                      # u + 3.6, v + 3.6 (positive)
uvmean = avgpool("uvmean", uv, [H, W_], [1, 1])                    # (2, 1, 1)

# --- per-pixel ramps ------------------------------------------------------------------------
# Skin core (general knowledge: skin under white light has R > G > B, R/G ~ 1.2-1.65 and is
# "pinker" than wood, bread, sand, tan fur, dirt or bark: its ln(G/B) is about 0.55-0.8 of its
# ln(R/G), while those materials have ln(G/B) >= ln(R/G)).
#   (expression, value where the ramp is 0, value where it is 1)
SKIN = [
    (U, 0.12, 0.18),                             # R/G >= ~1.2
    (comb((-1, U)), -0.58, -0.45),               # R/G <= ~1.57 (not red / orange / cooked meat)
    (V, 0.00, 0.05),                             # G > B (not pink / magenta)
    (comb((-1, V)), -0.40, -0.30),               # G/B <= ~1.35 (not yellow)
    (R, 0.12, 0.22),                             # bright enough for a reliable hue
    (comb((0.8, U), (-1, V)), -0.07, -0.01),     # ln(G/B) <= 0.8 ln(R/G)  (pink-ish, not tan)
]
# Broad "warm" locus: skin plus browns, tans, oranges, wood, food, fur.
WARM = [
    (U, 0.05, 0.10),
    (comb((-1, U)), -1.10, -0.90),
    (V, 0.05, 0.10),
    (comb((-1, V)), -1.60, -1.30),
    (R, 0.10, 0.18),
]
DARK = (comb((-1, Y)), -0.32, -0.16)             # luma below 0.16 -> dark (black / brown hair, shadow)
# Scene cues, measured on the camera's own colours (not white-balanced):
_u_raw = (U[0], U[1], 0.0, 0.0); _v_raw = (V[0], V[1], 0.0, 0.0)
SNOW = [                                         # bright, slightly bluish white (G >= R, B >= G)
    (Y, 0.55, 0.65),
    (comb((-1, _u_raw)), -0.005, 0.015), (_u_raw, -0.22, -0.15),
    (comb((-1, _v_raw)), 0.0, 0.02), (_v_raw, -0.28, -0.20),
]
YELLOW = [                                       # saturated yellow / orange / gold: food, wood, fur
    (_v_raw, 0.40, 0.50), (_u_raw, -0.05, 0.0), (R, 0.25, 0.35),
    (comb((1, _v_raw), (-1, _u_raw)), -0.10, 0.0),  # yellow at least as strong as red (not clay courts)
]

rows = []                                        # (w, c, cu, cv) pre-activations of col2


def ramp(expr, zero_at, one_at):
    w, c, cu, cv = expr
    k = 1.0 / (one_at - zero_at)
    a = (k * w, k * (c - zero_at), k * cu, k * cv)
    rows.append(a); rows.append((a[0], a[1] - 1.0, a[2], a[3]))  # relu(a) - relu(a-1) = clip(a,0,1)


GROUP = {}                                       # name -> (first row, number of ramps)


def ramps(name, lst):
    GROUP[name] = (len(rows), len(lst))
    for r in lst:
        ramp(*r)


ramps("skin", SKIN)
ramps("warm", WARM)
ramps("dark", [DARK])
ramps("snow", SNOW)
ramps("yellow", YELLOW)
J_CHROMA = len(rows)
for expr in (comb((1, R), (-1, G)), comb((-1, R), (1, G)), comb((1, G), (-1, B)), comb((-1, G), (1, B))):
    rows.append(expr)                            # |R-G| + |G-B| chroma (0 for black-and-white)
NR = len(rows)
c2W = np.array([r[0] for r in rows])[:, :, None, None]; c2b = np.array([r[1] for r in rows])
col2 = conv("col2", col1, c2W, c2b)

# white-balance shift for every col2 channel, broadcast to all pixels by a dense layer
cu = np.array([r[2] for r in rows]); cv = np.array([r[3] for r in rows])
wbW = np.zeros((NR, H * W_, 2)); wbb = np.zeros((NR, H * W_))
wbW[:, :, 0] = -WB_ALPHA * cu[:, None]; wbW[:, :, 1] = -WB_ALPHA * cv[:, None]
wbb[:, :] = (WB_ALPHA * (cu * (U_TYP + OFF) + cv * (V_TYP + OFF)))[:, None]
wb = reshape("wbmap", dense("wb", uvmean, wbW.reshape(NR * H * W_, 2), wbb.reshape(-1)), [NR, H, W_])
col2r = relu("col2r", add("col2wb", [col2, wb]))

c3W = np.zeros((7, NR, 1, 1)); c3b = np.zeros(7)
for out_ch, name in ((0, "skin"), (1, "warm"), (2, "dark"), (5, "snow"), (6, "yellow")):
    j0, n = GROUP[name]                          # fuzzy AND: relu(sum of clipped ramps - (n - 1))
    for j in range(n):
        c3W[out_ch, j0 + 2 * j] = 1.0; c3W[out_ch, j0 + 2 * j + 1] = -1.0
    c3b[out_ch] = -(n - 1)
c3W[3, J_CHROMA: J_CHROMA + 4] = 1.0             # chroma magnitude
c3b[4] = 1.0                                     # constant 1 inside the photo (0 in padding)
base0 = relu("base0", conv("col3", col2r, c3W, c3b))   # skin, warm, dark, chroma, inside, snow, yellow
# warm colour that is neither skin nor dark: wood, tan fur, beige walls, bread, dirt
c4W = np.eye(7)[:, :, None, None].copy(); c4W[1, :, 0, 0] = [-1, 1, -1, 0, 0, 0, 0]
base = relu("base", conv("col4", base0, c4W, np.zeros(7)))
CH_S, CH_Q, CH_D, CH_C, CH_I, CH_SNOW, CH_YEL = range(7)
NB = 7

# ==============================================================================================
# Stage 2: multi-scale head / skin-blob detectors.
# For a face (or hand) about c pixels wide: the c x c cell is mostly skin AND the cells to its
# left, right and above are mostly not skin (background, hair, clothes); for "hair heads" the
# cell above is also partly dark.  Cells are average-pooled at half-cell stride, so a 5x5 kernel
# with taps two steps apart compares neighbouring, non-overlapping cells.  Outside the photo
# counts as 80% "not skin" (via the constant channel, which is 0 in the zero padding).
# Corners and edges of large skin-coloured areas (walls, tables, sand) fail the AND.
# ==============================================================================================
SCALES = [(6, 0.8), (8, 1.0), (12, 1.0), (16, 1.0), (24, 1.0), (32, 0.8)]   # (cell size, weight)
OUT_NONSKIN = 0.8
F, LFT, RGT, UP = (2, 2), (2, 0), (2, 4), (0, 2)


def cond(terms, const, zero_at, one_at):
    k = 1.0 / (one_at - zero_at)
    w = np.zeros((NB, 5, 5))
    for ch, (dy, dx), coef in terms:
        w[ch, dy, dx] += k * coef
    return w, k * (const - zero_at)


def nonskin(tap, q=0.6):
    # 'neutral' surroundings: not skin and (for q > 0) not other warm colours; outside the photo: 0.8
    return [(CH_S, tap, -1.0), (CH_Q, tap, -q), (CH_I, tap, 1.0 - OUT_NONSKIN)], OUT_NONSKIN


CONDS = [cond([(CH_S, F, 1.0)], 0.0, 0.35, 0.65)]                       # face cell mostly skin
for tap in (LFT, RGT):
    t, c0 = nonskin(tap); CONDS.append(cond(t, c0, 0.50, 0.80))           # sides not skin / tan
t, c0 = nonskin(UP); CONDS.append(cond(t, c0, 0.50, 0.80))                # above not skin / tan
CONDS.append(cond([(CH_D, UP, 1.0)], 0.0, 0.10, 0.35))                   # above partly dark
for tap in (LFT, RGT):
    t, c0 = nonskin(tap, q=0.0); CONDS.append(cond(t, c0, 0.50, 0.80))    # sides not skin (tan allowed)

NS = len(SCALES)
NFEAT = 2 * NS + 6
I_SKIN, I_WARM, I_CMEAN, I_CMAX, I_YEL, I_SNOW = range(2 * NS, 2 * NS + 6)
branch_vecs = []
for si, (c, wgt) in enumerate(SCALES):
    cells = avgpool(f"cell{c}", base, [c, c], [c // 2, c // 2])
    n = (H - c) // (c // 2) + 1
    tW = np.zeros((2 * len(CONDS), NB, 5, 5)); tb = np.zeros(2 * len(CONDS))
    for j, (w, b0) in enumerate(CONDS):
        tW[2 * j] = w; tW[2 * j + 1] = w; tb[2 * j] = b0; tb[2 * j + 1] = b0 - 1.0
    t = relu(f"tpl{c}r", conv(f"tpl{c}", cells, tW, tb, stride=1, padding=2))
    hW = np.zeros((2, 2 * len(CONDS), 1, 1)); hb = np.array([-3.0, -3.0])
    for j in (0, 1, 2, 3):                       # blob = AND(skin cell, left, right, above)
        hW[0, 2 * j] = 1.0; hW[0, 2 * j + 1] = -1.0
    for j in (0, 5, 6, 4):                       # hair head = AND(skin cell, left, right, dark above)
        hW[1, 2 * j] = 1.0; hW[1, 2 * j + 1] = -1.0
    head = relu(f"head{c}r", conv(f"head{c}", t, hW, hb))
    hmean = avgpool(f"hmean{c}", head, [n, n], [1, 1])
    Wv = np.zeros((NFEAT, 2))
    Wv[si, 0] = n * n / 2.0                      # ~ number of blobs (a blob fires ~2 positions)
    Wv[NS + si, 1] = n * n / 2.0
    branch_vecs.append(dense(f"vcnt{c}", hmean, Wv, np.zeros(NFEAT)))

gmean = avgpool("gmean", base, [H, W_], [1, 1])
gmax = maxpool("gmax", base, [H, W_], [1, 1])
Wv = np.zeros((NFEAT, NB)); Wv[I_SKIN, CH_S] = 1.0; Wv[I_WARM, CH_Q] = 1.0; Wv[I_CMEAN, CH_C] = 1.0
Wv[I_YEL, CH_YEL] = 1.0
branch_vecs.append(dense("vmean", gmean, Wv, np.zeros(NFEAT)))
Wv = np.zeros((NFEAT, NB)); Wv[I_CMAX, CH_C] = 1.0
branch_vecs.append(dense("vmax", gmax, Wv, np.zeros(NFEAT)))
low = avgpool("lowpart", base, [64, W_], [32, 1])               # rows 0-63 and 32-95 -> (7, 2, 1)
Wv = np.zeros((NFEAT, NB, 2, 1)); Wv[I_SNOW, CH_SNOW, 1, 0] = 1.0  # snow share of the lower 2/3
branch_vecs.append(dense("vlow", low, Wv.reshape(NFEAT, -1), np.zeros(NFEAT)))
feat = add("feat", branch_vecs)

# ==============================================================================================
# Stage 3: hand-set evidence combination (log-odds style; positive favours "person").
# ==============================================================================================
hid_W, hid_b, out_terms = [], [], []


def capped(wvec, scale, cap, weight):
    """adds weight * min(max(wvec . f, 0) / scale, cap) to the person score."""
    w = np.asarray(wvec, float) / scale
    hid_W.append(w); hid_b.append(0.0); out_terms.append((len(hid_W) - 1, weight))
    hid_W.append(w); hid_b.append(-cap); out_terms.append((len(hid_W) - 1, -weight))


def unit(i, k=1.0):
    v = np.zeros(NFEAT); v[i] = k
    return v


wblob = np.zeros(NFEAT); whair = np.zeros(NFEAT)
for si, (c, wgt) in enumerate(SCALES):
    wblob[si] = wgt; whair[NS + si] = wgt
capped(unit(I_SKIN), 0.004, 1.5, 0.6)                       # skin pixels, per 0.4% of the photo
capped(wblob, 1.5, 1.5, 0.7)                                # compact skin blobs on neutral ground
capped(whair, 1.0, 1.5, 1.0)                                # ... with dark hair above
# skin colour over more than a quarter of the photo: walls / tables / close-up animals
hid_W.append(unit(I_SKIN, 1.0 / 0.25)); hid_b.append(-1.0); out_terms.append((len(hid_W) - 1, -1.5))
# a lot of warm non-skin colour (wood, fur, beige rooms, dirt): discount
hid_W.append(unit(I_WARM, 1.0 / 0.3)); hid_b.append(-1.0); out_terms.append((len(hid_W) - 1, -0.6))
hid_W.append(unit(I_WARM, 1.0 / 0.3)); hid_b.append(-2.0); out_terms.append((len(hid_W) - 1, 0.6))
# saturated yellow / orange over a large area: food close-ups, wood, giraffes, teddy bears
hid_W.append(unit(I_YEL, 1.0 / 0.12)); hid_b.append(-0.08 / 0.12); out_terms.append((len(hid_W) - 1, -0.8))
hid_W.append(unit(I_YEL, 1.0 / 0.12)); hid_b.append(-0.20 / 0.12); out_terms.append((len(hid_W) - 1, 0.8))
# snow covering the lower part of the photo: skiing / snowboarding scenes
hid_W.append(unit(I_SNOW, 1.0 / 0.2)); hid_b.append(-0.25 / 0.2); out_terms.append((len(hid_W) - 1, 1.0))
hid_W.append(unit(I_SNOW, 1.0 / 0.2)); hid_b.append(-0.45 / 0.2); out_terms.append((len(hid_W) - 1, -1.0))
# black-and-white photo (no chroma anywhere): skin cannot be seen; prior towards person
hid_W.append(unit(I_CMAX, -1.0 / 0.04)); hid_b.append(1.0); out_terms.append((len(hid_W) - 1, 1.7))
BIAS = -1.6

hid = relu("hidr", dense("hid", feat, np.array(hid_W), np.array(hid_b)))
out_w = np.zeros((2, len(hid_W))); out_b = np.array([0.0, BIAS])
for i, wgt in out_terms:
    out_w[1, i] += wgt
dense("scores", hid, out_w, out_b)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump({"layers": LAYERS}, f, indent=1)
    np.savez(os.path.join(out, "weights.npz"), **{k: v.astype(np.float32) for k, v in WEIGHTS.items()})


if __name__ == "__main__":
    main()
