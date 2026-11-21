"""Hand-designed classifier for 28x28 greyscale clothing photos (10 classes).

No training of any kind: every weight comes from formulas (Gaussian-derivative
filters), from geometric shape models typed from general knowledge of garments,
shoes and bags, and from hand-chosen rule tables.  build.py reads no files.

Pipeline
  imgn = norm(image)                  scale-invariant base (contrast varies wildly)
  econv = 12 zero-sum 5x5 filters     oriented edges +- and thin-line detectors +-
  ep   = 4x4 avgpool of relu(econv)   edge/line energy per cell
  sp   = 4x4 avgpool of imgn          silhouette occupancy (relative brightness)
  sp2  = 2x2 avgpool of imgn          finer silhouette
  rm/cm= row / column max profiles    vertical / horizontal extent
  tc   = 13 zero-sum 28x28 templates  hand-drawn shape models, translation search
  tm   = maxpool over translations    translation-tolerant match score
  spars= mean positive deviation      sparsity (strappy vs solid items)
  8 dense layers (one per branch) -> add -> relu  = hand-written concept units
  out  = dense -> 10                  hand-written vote table
"""
import sys, json
import numpy as np

# ---------------------------------------------------------------- utilities
def zsum(K):
    K = np.asarray(K, float)
    return K - K.mean()

def l2s(K, s):
    K = zsum(K)
    return s * K / np.sqrt((K * K).sum())

# ---------------------------------------------------------------- edge filters (5x5, sigma 1)
SIG = 1.0
yy, xx = np.mgrid[-2:3, -2:3].astype(float)
G = np.exp(-(xx * xx + yy * yy) / (2 * SIG * SIG))
gy_raw  = -yy * G / SIG**2                      # d/dy  -> horizontal edges
gx_raw  = -xx * G / SIG**2                      # d/dx  -> vertical edges
d45_raw  = (gx_raw + gy_raw) / np.sqrt(2)       # diagonal edges
d135_raw = (gx_raw - gy_raw) / np.sqrt(2)
hline_raw = (1.0/SIG**2 - yy*yy/SIG**4) * G     # -d2/dy2: bright thin horizontal bar
vline_raw = (1.0/SIG**2 - xx*xx/SIG**4) * G     # -d2/dx2: bright thin vertical bar
ESC, VSC = 2.5, 3.0                             # response scale of edge / line filters
FILT = np.stack([
    l2s(gy_raw,  ESC),  l2s(-gy_raw,  ESC),     # 0,1  horizontal edges
    l2s(gx_raw,  ESC),  l2s(-gx_raw,  ESC),     # 2,3  vertical edges
    l2s(d45_raw, ESC),  l2s(-d45_raw, ESC),     # 4,5  diagonal edges
    l2s(d135_raw,ESC),  l2s(-d135_raw,ESC),     # 6,7  diagonal edges
    l2s(hline_raw,VSC), l2s(-hline_raw,VSC),    # 8,9  bright/dark thin horizontal line
    l2s(vline_raw,VSC), l2s(-vline_raw,VSC),    # 10,11 bright/dark thin vertical line
])
NEC = FILT.shape[0]

# ---------------------------------------------------------------- template primitives
def Tp_new():
    return np.zeros((28, 28), float)

def Tp_box(m, x0, x1, y0, y1, v=1.0):
    m[max(0,y0):min(28,y1+1), max(0,x0):min(28,x1+1)] += v

def Tp_trap(m, y0, y1, xt0, xt1, xb0, xb1, v=1.0):
    for y in range(y0, y1+1):
        f = (y - y0) / max(1, (y1 - y0))
        a = int(round(xt0 + f*(xb0-xt0))); b = int(round(xt1 + f*(xb1-xt1)))
        m[y, max(0,a):min(28,b+1)] += v

def Tp_diag(m, x0, y0, x1, y1, w, v=1.0):
    n = max(abs(x1-x0), abs(y1-y0)) + 1
    for k in range(n):
        f = k / max(1, n-1)
        cx = int(round(x0 + f*(x1-x0))); cy = int(round(y0 + f*(y1-y0)))
        for o in range(-(w//2), w - w//2):
            if 0 <= cx+o < 28 and 0 <= cy < 28: m[cy, cx+o] += v

def Tp_negfull(m, v=-0.6):
    """every cell not part of the shape expects darkness (full-canvas penalty:
    no translation can hide extra mass outside a bounding box)"""
    m[m == 0] += v

def Tp_finish(m, negw=0.6):
    m = zsum(m)                       # exactly zero-sum: norm-translation invariance
    P = np.clip(m, 0, None).sum(); N = np.clip(-m, 0, None).sum()
    return m / max(1e-9, P + negw * N)

def T_mirror(m):
    return m[:, ::-1]

# ---------------------------------------------------------------- shape templates
# Drawn from general garment/shoe/bag geometry on a 28x28 canvas where the item's
# longest side spans the frame: tops fill the height with sleeves at the sides,
# trousers are tall and narrow with a dark gap between the legs, a dress flares
# from narrow shoulders to a wide hem, shoes sit low with a bright sole bar,
# a bag has thin handle straps and a dark hole above a big body, a boot has a
# shaft rising on one side over a wide foot and sole.
def build_templates():
    T = []

    def add(m): T.append(Tp_finish(m))

    # 0 t-shirt: short sleeves high up, boxy body, neck notch, hem near the bottom
    m = Tp_new()
    Tp_box(m, 12, 15, 1, 2, -1.0)     # neck notch (dark dip, unlike a shirt collar)
    Tp_box(m, 8, 19, 2, 6)            # shoulders
    Tp_box(m, 3, 8, 4, 10); Tp_box(m, 19, 24, 4, 10)   # short sleeves
    Tp_box(m, 10, 17, 7, 26)          # body
    Tp_negfull(m); add(m)

    # 1 trouser: waistband bar, two legs with a dark gap between them
    m = Tp_new()
    Tp_box(m, 10, 17, 1, 3)           # waistband
    Tp_box(m, 9, 12, 4, 26); Tp_box(m, 15, 18, 4, 26)  # legs
    Tp_box(m, 13, 14, 6, 26, -0.8)    # dark gap between legs
    Tp_negfull(m); add(m)

    # 2 pullover: shoulders, long sleeves down the sides, body to the hem, neck notch
    m = Tp_new()
    Tp_box(m, 12, 15, 1, 2, -1.0)   # neck notch (dark dip at the top centre)
    Tp_box(m, 8, 19, 2, 5)
    Tp_box(m, 3, 8, 5, 23); Tp_box(m, 19, 24, 5, 23)   # long sleeves
    Tp_box(m, 9, 18, 5, 25)           # body
    Tp_negfull(m); add(m)

    # 3 dress: narrow shoulders flaring to a wide hem, solid skirt centre
    m = Tp_new()
    Tp_box(m, 13, 14, 10, 24, 1.5)  # skirt centre is solid (no leg gap)
    Tp_trap(m, 2, 26, 11, 16, 7, 20)
    Tp_negfull(m); add(m)

    # 4 coat: like pullover but longer, with a dark front opening down the middle
    m = Tp_new()
    Tp_box(m, 8, 19, 2, 5)
    Tp_box(m, 3, 8, 5, 24); Tp_box(m, 19, 24, 5, 24)
    Tp_box(m, 9, 18, 5, 27)
    Tp_box(m, 13, 14, 8, 22, -0.8)    # front opening (dark line)
    Tp_negfull(m); add(m)

    # 5 sandal: low sole bar with thin crossing straps, nothing solid above
    m = Tp_new()
    Tp_box(m, 4, 23, 17, 21)                      # sole
    Tp_diag(m, 9, 15, 17, 9, 2, 0.7)              # strap /
    Tp_diag(m, 17, 15, 9, 9, 2, 0.7)              # strap backslash
    Tp_box(m, 7, 20, 8, 17, -1.4)                 # mostly dark between straps (not a solid upper)
    Tp_negfull(m, -0.6); add(m)

    # 6 shirt: collar band, long sleeves, button placket down the centre
    m = Tp_new()
    Tp_box(m, 11, 16, 1, 3, 1.8)     # collar (bright band, unlike a neck notch)
    Tp_box(m, 8, 19, 3, 6)            # yoke/shoulders
    Tp_box(m, 3, 8, 4, 22); Tp_box(m, 19, 24, 4, 22)   # long sleeves
    Tp_box(m, 9, 18, 4, 26)           # body
    Tp_box(m, 13, 14, 5, 25, 1.4)     # button placket (the shirt's signature)
    Tp_negfull(m); add(m)

    # 7 sneaker: thick bright sole, solid upper, and nothing above row 8
    m = Tp_new()
    Tp_box(m, 5, 22, 2, 8, -0.7)      # empty air above the upper
    Tp_box(m, 2, 25, 18, 21)          # sole
    Tp_box(m, 3, 11, 14, 17, 0.8)     # toe
    Tp_box(m, 11, 21, 9, 17, 0.8)     # upper
    Tp_box(m, 18, 23, 9, 13, 0.6)     # heel collar
    Tp_negfull(m); add(m)

    # 8 bag: two thin handle straps + dark hole above a big body
    m = Tp_new()
    Tp_box(m, 4, 23, 12, 26)          # body
    Tp_box(m, 11, 12, 3, 11); Tp_box(m, 15, 16, 3, 11) # handle straps
    Tp_box(m, 11, 16, 2, 4)           # handle top
    Tp_box(m, 13, 14, 4, 11, -0.8)    # dark hole between straps
    Tp_negfull(m); add(m)

    # 9 boot (shaft on the right): tall shaft, wide foot, sole bar
    m = Tp_new()
    Tp_box(m, 15, 21, 2, 13)          # shaft
    Tp_box(m, 8, 21, 13, 18, 0.9)     # foot
    Tp_box(m, 5, 22, 19, 23)          # sole
    Tp_negfull(m); add(m)

    # 10 boot (shaft on the left): mirror image of the right-shaft boot
    add(Tp_finish(T_mirror(m)))      # m still holds the right-shaft boot

    # 11 sandal, crossing straps: sole bar lower, two diagonal straps
    m = Tp_new()
    Tp_box(m, 5, 22, 15, 19)
    Tp_diag(m, 8, 14, 18, 8, 3, 0.6)
    Tp_diag(m, 18, 14, 8, 8, 3, 0.6)
    Tp_box(m, 7, 20, 8, 15, -1.4)
    Tp_negfull(m, -0.5); add(m)

    # 12 dress with short sleeves: shoulders + sleeves then a flaring skirt
    m = Tp_new()
    Tp_box(m, 9, 18, 2, 5)
    Tp_box(m, 5, 8, 3, 8); Tp_box(m, 19, 22, 3, 8)
    Tp_trap(m, 6, 26, 11, 16, 7, 20)
    Tp_negfull(m); add(m)

    return T

# ---------------------------------------------------------------- network assembly
class Net:
    def __init__(self):
        self.layers = []; self.W = {}

    def layer(self, name, op, inputs, **kw):
        d = {"name": name, "op": op, "inputs": inputs}; d.update(kw)
        self.layers.append(d)

    def conv(self, name, inp, W, stride=1, padding=0):
        W = np.asarray(W, float)
        self.W[name + ".W"] = W.astype(np.float32)
        self.W[name + ".b"] = np.zeros(W.shape[0], np.float32)
        self.layer(name, "conv", [inp], out=int(W.shape[0]),
                   kernel=[int(W.shape[2]), int(W.shape[3])], stride=stride, padding=padding)

    def dense(self, name, inp, W, b):
        self.W[name + ".W"] = np.asarray(W, float).astype(np.float32)
        self.W[name + ".b"] = np.asarray(b, float).astype(np.float32)
        self.layer(name, "dense", [inp], out=int(W.shape[0]))

def spread(mat, row, idxs, wtot):
    n = len(idxs)
    if n == 0: return
    for k in idxs: mat[row, k] += wtot / n

def ep_idx(ch, i, j):   return ch * 49 + i * 7 + j      # ep   (12,7,7)
def sp_idx(i, j):       return i * 7 + j                # sp   (1,7,7)
def sp2_idx(i, j):      return i * 14 + j               # sp2  (1,14,14)

# ---------------------------------------------------------------- concept units
# Every unit is a hand-written rule over interpretable features:
#   dS/dS2  occupancy cells (7x7 / 14x14) of relu(imgn): empty cell = 0 exactly,
#           partial cell ~ 0.3..0.8, full cell ~ 1.1..2.0 (sparser items larger)
#   dR/dC   row / column max occupancy: empty band = 0, present ~ 1.2..2.0
#   dE      edge/line energy cells; a thin line or edge in a cell ~ 0.5..1.5
#   dT      template match scores ~ 0..1
#   dX      fill scalar sqrt(fill*(1-fill)): sparse items ~0.35, solid ~0.48
# Sleeve notes: a top's widest dimension is its sleeve span, so after scaling to
# 28 px the sleeves hug columns 0-3 and 24-27; trouser/dress/bag stay narrow there.
UNITS = [
 dict(name="top_center",   b=-0.42,   # bright block at the very top centre (collar / handle top)
      dS =[("rect",0,0,2,4, 0.9)], dS2=[("rect",0,1,5,8, 0.5)]),
 dict(name="top_wide",     b=-0.40,   # shoulders / waistband across the top
      dS =[("rect",0,0,1,5, 0.7)], dS2=[("rect",1,2,3,10, 0.6)]),
 dict(name="empty_top",    b=+0.45,   # nothing in the top two row bands (shoes)
      dR =[("cell",0,-0.70),("cell",1,-0.45)]),
 dict(name="fullheight",   b=-0.90,   # content in the first and last bands and the middle
      dR =[("cell",1,0.50),("cell",6,0.50),("bands",2,5,0.50)]),
 dict(name="fill_lo",      b=+2.50,   # low fill fraction (sandal / sneaker)
      dX =[("x", -6.0)]),
 dict(name="top_fill",     b=+0.00,   # graded: how much content sits in the top two bands
      dR =[("cell",0,0.50),("cell",1,0.50)]),
 dict(name="sole_B4",      b=-0.90,   # bright wide bar with line/edge energy at rows 16-19
      dS =[("rect",4,4,2,4, 0.8)],
      dE =[("ch",8,4,4,2,4, 0.60),("ch",0,4,4,0,6, 0.25),("ch",1,4,4,0,6, 0.25)]),
 dict(name="sole_B5",      b=-0.90,   # same at rows 20-23
      dS =[("rect",5,5,2,4, 0.8)],
      dE =[("ch",8,5,5,2,4, 0.60),("ch",0,5,5,0,6, 0.25),("ch",1,5,5,0,6, 0.25)]),
 dict(name="sole_B6",      b=-1.00,   # same at rows 24-27
      dS =[("rect",6,6,1,5, 0.8)],
      dE =[("ch",8,6,6,2,4, 0.40),("ch",0,6,6,0,6, 0.20),("ch",1,6,6,0,6, 0.20)]),
 dict(name="upper_solid",  b=-0.60,   # solid bright mass in the middle rows (sneaker upper)
      dS =[("rect",3,4,2,4, 1.0)]),
 dict(name="leggap",       b=-0.35,   # dark slot at the bottom centre, bright on both sides (trouser legs)
      dS =[("cell",6,3,-0.90),("cell",5,3,-0.50),("cell",4,3,-0.50),
           ("cell",4,2, 0.30),("cell",4,4, 0.30),("cell",5,2, 0.30),("cell",5,4, 0.30)],
      dE =[("ch",2,4,6,3,3, 0.25),("ch",3,4,6,3,3, 0.25),("ch",11,4,6,3,3, 0.40)],
      dS2=[("rect",12,13,6,7, -0.70),("rect",10,13,6,7, -0.40)]),
 dict(name="sleeve_up",    b=-0.75,   # bright at both far sides, rows 4-11 (shoulders/sleeve tops)
      dS =[("cell",1,0,1.00),("cell",2,0,1.00),("cell",1,6,1.00),("cell",2,6,1.00)],
      dS2=[("rect",2,5,0,1, 1.6),("rect",2,5,12,13, 1.6)]),
 dict(name="sleeve_low",   b=-1.60,   # long sleeves: tall bright columns at both far sides, rows 8-23
      dS =[("cell",2,0,0.80),("cell",3,0,0.50),("cell",4,0,0.40),("cell",5,0,0.30),
           ("cell",2,6,0.80),("cell",3,6,0.50),("cell",4,6,0.40),("cell",5,6,0.30)],
      dS2=[("rect",4,11,0,1, 0.8),("rect",4,11,12,13, 0.8)]),
 dict(name="narrow_top",   b=-0.35,   # rows 4-11: bright only in the centre (bodice/waist/handle)
      dS =[("cell",1,1,-0.50),("cell",1,5,-0.50),("cell",2,1,-0.50),("cell",2,5,-0.50),
           ("cell",1,3, 0.50),("cell",2,3, 0.50),
           ("cell",1,2, 0.20),("cell",2,2, 0.20),("cell",1,4, 0.20),("cell",2,4, 0.20)],
      dS2=[("rect",2,5,2,3, -0.5),("rect",2,5,10,11, -0.5),("rect",2,5,5,8, 0.5)]),
 dict(name="flare",        b=-0.35,   # sides dark high up, bright low down (skirt flare)
      dS =[("cell",5,1,0.20),("cell",5,5,0.20),("cell",6,1,0.20),("cell",6,5,0.20),
           ("cell",1,1,-0.20),("cell",1,5,-0.20),("cell",2,1,-0.20),("cell",2,5,-0.20)],
      dS2=[("rect",10,13,4,4, 0.50),("rect",10,13,9,9, 0.50),
           ("rect",1,4,4,4, -0.30),("rect",1,4,9,9, -0.30),
           ("rect",1,4,3,3, -0.20),("rect",1,4,10,10, -0.20)]),
 dict(name="handle_straps",b=-0.65,   # two thin bright straps with a dark slot between, dark flanks
      dE =[("ch",10,1,2,2,4, 0.30),("ch",11,1,1,3,3, 0.35)],
      dS2=[("rect",2,5,4,4, -0.5),("rect",2,5,9,9, -0.5),
           ("rect",2,5,5,5, 0.5),("rect",2,5,8,8, 0.5)]),
 dict(name="handle_hole",  b=-0.55,   # dark slot below a bright handle bar (bag)
      dE =[("ch",11,1,2,2,4, 0.8)],
      dS2=[("rect",2,5,6,7, -0.6),("rect",0,1,5,8, 0.4)]),
 dict(name="centerline",   b=-0.55,   # bright vertical line down a bright centre (buttons/opening)
      dE =[("ch",10,2,5,3,3, 0.30)],
      dS2=[("rect",4,11,6,7, 1.1)]),
 dict(name="body_high",    b=-0.55,   # solid body at rows 8-11 centre (tops/dress, not a bag hole)
      dS =[("rect",2,2,3,3, 1.2)], dS2=[("rect",4,5,6,7, 0.5)]),
 dict(name="shaft_R",      b=-0.75,   # tall bright column upper right, dark upper left (boot shaft)
      dS =[("cell",0,4,0.40),("cell",0,5,0.40),("cell",1,4,0.40),("cell",1,5,0.40),
           ("cell",2,4,0.25),("cell",2,5,0.25),
           ("cell",0,1,-0.35),("cell",0,2,-0.35),("cell",1,1,-0.35),("cell",1,2,-0.35),
           ("cell",2,1,-0.20),("cell",2,2,-0.20)],
      dS2=[("rect",0,4,9,13, 0.5),("rect",0,4,1,4, -0.4)]),
 dict(name="shaft_L",      b=-0.75,   # mirror of shaft_R
      dS =[("cell",0,2,0.40),("cell",0,1,0.40),("cell",1,2,0.40),("cell",1,1,0.40),
           ("cell",2,2,0.25),("cell",2,1,0.25),
           ("cell",0,5,-0.35),("cell",0,4,-0.35),("cell",1,5,-0.35),("cell",1,4,-0.35),
           ("cell",2,5,-0.20),("cell",2,4,-0.20)],
      dS2=[("rect",0,4,0,4, 0.5),("rect",0,4,9,13, -0.4)]),
 dict(name="hem_center",   b=-0.75,   # bright block at bottom centre (coat/shirt hem, bag base)
      dS =[("cell",6,3, 0.80),("cell",6,2, 0.35),("cell",6,4, 0.35)],
      dS2=[("rect",12,13,6,7, 0.5)]),
 dict(name="strap_thin",   b=-0.50,   # line energy high but occupancy low in the middle (sandal straps)
      dE =[("ch",8,2,4,1,5, 0.80),("ch",10,2,4,1,5, 0.80)],
      dS =[("rect",2,4,1,5, -1.0)]),
]
TEMPLATE_THETA = 0.42   # relu gate for template pass-through units

# ---------------------------------------------------------------- per-class evidence
# Second hidden layer: one conjunction unit per class.  Every rule unit X also
# exists as a clone Xc whose gate is raised by CAP, so a contribution
# w*(relu(X) - relu(Xc)) equals w*min(activation, CAP): no hand-tuned rule can
# run away on real data, whatever its absolute scale.  Templates are naturally
# bounded (self-normalised) and are the primary evidence.
CAP = 0.8
UNITS2 = [
 dict(name="S0", b=-0.85,   # t-shirt: short sleeves only, no skirt flare
      h =[("Tt0",1.2),("sleeve_up",0.8),("top_wide",0.3),("top_fill",0.25),("body_high",0.15),
          ("sleeve_low",-0.7),("flare",-0.4),("leggap",-0.4),("empty_top",-0.6),
          ("centerline",-0.15),("narrow_top",-0.3),("handle_straps",-0.3)]),
 dict(name="S1", b=-0.85,   # trouser: leg gap, narrow waist, no flare
      h =[("Tt1",1.2),("leggap",1.0),("narrow_top",0.5),("top_fill",0.25),("body_high",0.15),
          ("flare",-0.4),("sleeve_up",-0.4),("sleeve_low",-0.4),("handle_straps",-0.4),
          ("handle_hole",-0.3),("hem_center",-0.2),("centerline",-0.3)]),
 dict(name="S2", b=-0.85,   # pullover: long sleeves, plain front, dim hem centre
      h =[("Tt2",1.1),("sleeve_low",0.9),("sleeve_up",0.4),("top_wide",0.25),("top_fill",0.25),
          ("body_high",0.15),("centerline",-0.4),("hem_center",-0.35),("leggap",-0.4),
          ("empty_top",-0.6),("narrow_top",-0.3),("handle_straps",-0.3),("flare",-0.2)]),
 dict(name="S3", b=-0.85,   # dress: skirt flare, narrow bodice
      h =[("Tt3",1.1),("flare",0.8),("narrow_top",0.45),("Tt12",0.3),("hem_center",0.2),
          ("top_fill",0.25),("body_high",0.15),("sleeve_up",-0.4),("sleeve_low",-0.4),
          ("leggap",-0.3),("handle_straps",-0.4)]),
 dict(name="S4", b=-0.85,   # coat: long sleeves + opening line + solid low hem
      h =[("Tt4",1.1),("sleeve_low",0.85),("centerline",0.5),("hem_center",0.45),
          ("sleeve_up",0.4),("top_fill",0.25),("body_high",0.15),("leggap",-0.4),
          ("empty_top",-0.5),("flare",-0.3),("narrow_top",-0.3),("handle_hole",-0.3)]),
 dict(name="S5", b=-0.75,   # sandal: little at top, thin straps, sole bars, sparse
      h =[("Tt5",1.1),("Tt11",0.5),("sole_B4",0.6),("sole_B5",0.6),("sole_B6",0.3),
          ("strap_thin",0.6),("fill_lo",0.35),("empty_top",0.3),("upper_solid",-0.5),
          ("fullheight",-0.4),("sleeve_up",-0.35),("sleeve_low",-0.25),("top_center",-0.35),
          ("top_fill",-0.5),("body_high",-0.3),("leggap",-0.2)]),
 dict(name="S6", b=-0.85,   # shirt: bright placket line AND long sleeves AND collar
      h =[("Tt6",1.1),("centerline",0.6),("sleeve_low",0.6),("top_center",0.35),
          ("hem_center",0.3),("sleeve_up",0.25),("top_fill",0.25),("body_high",0.15),
          ("flare",-0.35),("empty_top",-0.6),("leggap",-0.3),("narrow_top",-0.3),
          ("handle_hole",-0.3),("shaft_R",-0.2),("shaft_L",-0.2),("sole_B5",-0.3)]),
 dict(name="S7", b=-0.85,   # sneaker: empty top, solid upper, sole bars
      h =[("Tt7",1.15),("empty_top",0.7),("upper_solid",0.65),("sole_B4",0.35),("sole_B5",0.45),
          ("sole_B6",0.2),("fill_lo",0.15),("fullheight",-0.5),("sleeve_up",-0.35),
          ("sleeve_low",-0.25),("shaft_R",-0.25),("shaft_L",-0.25),("handle_straps",-0.3),
          ("strap_thin",-0.25),("top_center",-0.25),("top_fill",-0.4),("body_high",-0.2)]),
 dict(name="S8", b=-0.85,   # bag: handle straps + hole, no high body mass
      h =[("Tt8",1.1),("handle_straps",0.9),("handle_hole",0.7),("narrow_top",0.4),
          ("hem_center",0.2),("fullheight",0.2),("strap_thin",0.1),("body_high",-0.45),
          ("sleeve_up",-0.4),("sleeve_low",-0.3),("empty_top",-0.35),("leggap",-0.35),
          ("flare",-0.2),("top_fill",-0.2)]),
 dict(name="S9a", b=-0.80,  # boot with right shaft
      h =[("Tt9",1.0),("shaft_R",0.9),("sole_B5",0.35),("sole_B6",0.45),("upper_solid",0.25),
          ("empty_top",-0.25),("sleeve_up",-0.25),("handle_straps",-0.2)]),
 dict(name="S9b", b=-0.80,  # boot with left shaft
      h =[("Tt10",1.0),("shaft_L",0.9),("sole_B5",0.35),("sole_B6",0.45),("upper_solid",0.25),
          ("empty_top",-0.25),("sleeve_up",-0.25),("handle_straps",-0.2)]),
]
OUT_MAP = {0:["S0"],1:["S1"],2:["S2"],3:["S3"],4:["S4"],5:["S5"],6:["S6"],
           7:["S7"],8:["S8"],9:["S9a","S9b"]}

# ---------------------------------------------------------------- build the network
def build_network(templates):
    net = Net()
    Ft = len(templates)

    net.layer("imgn", "norm", ["image"])
    net.layer("imgp", "relu", ["imgn"])
    net.layer("spars", "avgpool", ["imgp"], size=[28, 28], stride=[28, 28])
    net.layer("peak", "maxpool", ["imgn"], size=[28, 28], stride=[28, 28])
    net.conv("econv", "imgn", FILT[:, None, :, :], stride=1, padding=2)
    net.layer("er", "relu", ["econv"])
    net.layer("ep", "avgpool", ["er"], size=[4, 4], stride=[4, 4])
    net.layer("sp", "avgpool", ["imgp"], size=[4, 4], stride=[4, 4])
    net.layer("sp2", "avgpool", ["imgp"], size=[2, 2], stride=[2, 2])
    net.layer("rm", "maxpool", ["imgp"], size=[4, 28], stride=[4, 28])
    net.layer("cm", "maxpool", ["imgp"], size=[28, 4], stride=[28, 4])
    net.conv("tc", "imgn", np.stack(templates)[:, None, :, :], stride=1, padding=8)
    net.layer("tm", "maxpool", ["tc"], size=[17, 17], stride=[17, 17])

    units = list(UNITS) + [dict(name="Tt%d" % t, b=-TEMPLATE_THETA, dT=[("t", t, 1.0)])
                           for t in range(Ft)]
    # capped clones of every rule unit (gate lowered by CAP):
    # relu(X) - relu(X - CAP) = min(activation, CAP)
    clones = [dict(u, name=u["name"] + "c", b=u.get("b", 0.0) - CAP) for u in UNITS]
    units = units + clones
    H = len(units)
    UIDX = {u["name"]: i for i, u in enumerate(units)}

    branches = [("dE", "ep", NEC * 49), ("dS", "sp", 49), ("dS2", "sp2", 196),
                ("dR", "rm", 7), ("dC", "cm", 7), ("dT", "tm", Ft),
                ("dX", "spars", 1), ("dP", "peak", 1)]
    mats = {b[0]: np.zeros((H, b[2])) for b in branches}
    bias = {b[0]: np.zeros(H) for b in branches}

    for u, spec in enumerate(units):
        for it in spec.get("dE", []):
            if it[0] == "cell":
                mats["dE"][u, ep_idx(it[1], it[2], it[3])] += it[4]
            else:
                ch, i0, i1, j0, j1, wt = it[1:7]
                spread(mats["dE"], u, [ep_idx(ch, i, j) for i in range(i0, i1 + 1)
                                       for j in range(j0, j1 + 1)], wt)
        for key, dim, ix in (("dS", 7, sp_idx), ("dS2", 14, sp2_idx)):
            for it in spec.get(key, []):
                if it[0] == "cell":
                    mats[key][u, ix(it[1], it[2])] += it[3]
                else:
                    i0, i1, j0, j1, wt = it[1:6]
                    spread(mats[key], u, [ix(i, j) for i in range(i0, i1 + 1)
                                          for j in range(j0, j1 + 1)], wt)
        for it in spec.get("dR", []):
            if it[0] == "cell": mats["dR"][u, it[1]] += it[2]
            else: spread(mats["dR"], u, list(range(it[1], it[2] + 1)), it[3])
        for it in spec.get("dC", []):
            if it[0] == "cell": mats["dC"][u, it[1]] += it[2]
            else: spread(mats["dC"], u, list(range(it[1], it[2] + 1)), it[3])
        for it in spec.get("dT", []): mats["dT"][u, it[1]] += it[2]
        for it in spec.get("dX", []): mats["dX"][u, 0] += it[1]
        for it in spec.get("dP", []): mats["dP"][u, 0] += it[1]
        bias["dE"][u] = spec.get("b", 0.0)          # the unit's gate lives in the first branch

    for name, inp, dim in branches:
        net.dense(name, inp, mats[name], bias[name])
    net.layer("hsum", "add", [b[0] for b in branches])
    net.layer("hid", "relu", ["hsum"])

    H2 = len(UNITS2)
    U2IDX = {u["name"]: i for i, u in enumerate(UNITS2)}
    W2 = np.zeros((H2, H)); b2 = np.zeros(H2)
    for u, spec in enumerate(UNITS2):
        for nm, w in spec.get("h", []):
            W2[u, UIDX[nm]] += w
            if nm + "c" in UIDX:              # capped rule: subtract the clone
                W2[u, UIDX[nm + "c"]] -= w
        b2[u] = spec.get("b", 0.0)
    net.dense("dH2", "hid", W2, b2)
    net.layer("ev", "relu", ["dH2"])

    Wout = np.zeros((10, H2)); bout = np.zeros(10)
    for c, names in OUT_MAP.items():
        for nm in names: Wout[c, U2IDX[nm]] += 1.0
    net.dense("out", "ev", Wout, bout)
    return net

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/app/out"
    net = build_network(build_templates())
    np.savez(out + "/weights.npz", **net.W)
    with open(out + "/network.json", "w") as f:
        json.dump({"layers": net.layers}, f, indent=1)
    print("wrote", out + "/network.json and weights.npz",
          "| layers:", len(net.layers), "| weights:",
      sum(int(np.prod(v.shape)) for v in net.W.values()))

if __name__ == "__main__":
    main()
