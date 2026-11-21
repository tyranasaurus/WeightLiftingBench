"""Builds a hand-designed clothing classifier: network.json + weights.npz.

No data is read.  Every number comes from geometry typed from general knowledge
of what the ten clothing / shoe / bag classes look like.

Idea.  The input is lightly blurred and a small bias is subtracted so that a
relu leaves a background~0 item mask r (the garment is light on a black ground).
For each class c we hand-draw two region masks from geometric primitives:
  P_c  where that class characteristically HAS mass, and
  N_c  where its most-confusable neighbours have mass but this class does NOT
       (e.g. a dress's N is the near-sides a wide top fills; a trouser's N is
        the centre gap a filled torso covers; a bag's N is the strip just above
        its flat top where a garment's shoulders sit; a shoe's N is the whole
        empty upper half).
The dense weight row is  P_c/|P_c| - N_c/|N_c| , so the score is
  score_c = mean_{P_c}(r) - mean_{N_c}(r) .
This is balanced (a region filled equally in P and N scores ~0, so no class is
"greedy") and scale free (r scales -> all scores scale, argmax unchanged).  Two
pose variants per class (as-drawn and left-right mirror) are max-pooled so shoes
facing either way match, and the class with the largest score is the prediction.
"""
import sys, os, json
import numpy as np

ys, xs = np.mgrid[0:28, 0:28].astype(float)   # ys = row (top->down), xs = col
EPS = 1e-8

# ----------------------------------------------------------------------------
# drawing primitives (all parametric, from general shape knowledge)
# ----------------------------------------------------------------------------
def blank():
    return np.zeros((28, 28), float)

def rect(m, y0, y1, x0, x1, v=1.0):
    y0 = max(0, y0); y1 = min(27, y1); x0 = max(0, x0); x1 = min(27, x1)
    if y1 >= y0 and x1 >= x0:
        m[y0:y1 + 1, x0:x1 + 1] = v

def clear(m, y0, y1, x0, x1):
    rect(m, y0, y1, x0, x1, 0.0)

# ----------------------------------------------------------------------------
# POSITIVE (P) and NEGATIVE (N) regions for each class.
# ----------------------------------------------------------------------------
def tmpl_tshirt():                                   # 0  short sleeves, short body
    P = blank()
    rect(P, 6, 10, 4, 23)                            # wide shoulders / short sleeves
    rect(P, 10, 21, 8, 19)                           # body
    clear(P, 6, 8, 12, 15)                           # neck
    N = blank()
    rect(N, 13, 23, 1, 6)                            # long-sleeve zone (left)  -> pullover/coat/shirt
    rect(N, 13, 23, 21, 26)                          # long-sleeve zone (right)
    rect(N, 22, 27, 8, 19)                           # below a short hem        -> dress/coat
    return P, N

def tmpl_trouser():                                  # 1  two legs, centre gap
    P = blank()
    rect(P, 4, 8, 9, 18)                             # waistband
    rect(P, 8, 27, 9, 12)                            # left leg
    rect(P, 8, 27, 16, 19)                           # right leg
    N = blank()
    rect(N, 12, 27, 13, 15)                          # centre gap   -> filled torso / dress
    rect(N, 4, 12, 2, 7)                             # sleeve/shoulder zone -> tops
    rect(N, 4, 12, 21, 26)
    return P, N

def tmpl_pullover():                                 # 2  wide, long sleeves, solid, short
    P = blank()
    rect(P, 6, 22, 5, 22)                            # wide solid body (shortest top)
    rect(P, 9, 21, 2, 6)                             # left long sleeve
    rect(P, 9, 21, 21, 25)                           # right long sleeve
    rect(P, 4, 6, 9, 18)                             # rounded top
    N = blank()
    rect(N, 0, 3, 4, 23)                             # above shoulders
    rect(N, 24, 27, 5, 22)                           # below hem -> shirt/coat/dress are longer
    return P, N

def tmpl_dress():                                    # 3  narrow, long
    P = blank()
    rect(P, 5, 8, 9, 18)                             # narrow shoulders
    rect(P, 8, 27, 9, 18)                            # long slim body
    N = blank()
    rect(N, 8, 24, 3, 8)                             # near-left side  -> wide tops fill this
    rect(N, 8, 24, 19, 24)                           # near-right side
    rect(N, 5, 9, 1, 8)                              # wide shoulders of tops
    rect(N, 5, 9, 19, 26)
    return P, N

def tmpl_coat():                                     # 4  wide, long sleeves, longest body
    P = blank()
    rect(P, 6, 27, 5, 22)                            # longest body (outerwear)
    rect(P, 9, 26, 2, 6)                             # left sleeve
    rect(P, 9, 26, 21, 25)                           # right sleeve
    N = blank()
    rect(N, 0, 3, 4, 23)                             # above shoulders
    return P, N

def tmpl_sandal():                                   # 5  thin, open, bottom
    P = blank()
    rect(P, 20, 26, 4, 24)                           # sole
    rect(P, 15, 25, 20, 24)                          # heel block (right)
    N = blank()
    rect(N, 0, 15, 2, 25)                            # whole upper is empty -> garments
    rect(N, 18, 21, 9, 15)                           # open mid-foot -> sneaker is solid here
    return P, N

def tmpl_shirt():                                    # 6  wide, long sleeves, open collar
    P = blank()
    rect(P, 6, 23, 5, 22)                            # body (medium length)
    rect(P, 9, 22, 2, 6)                             # left sleeve
    rect(P, 9, 22, 21, 25)                           # right sleeve
    clear(P, 5, 8, 11, 16)                           # collar opening
    N = blank()
    rect(N, 5, 8, 11, 16)                            # collar opening -> pullover is solid there
    rect(N, 0, 4, 5, 22)                             # above shoulders
    rect(N, 25, 27, 5, 22)                           # below hem -> shorter than a coat
    return P, N

def tmpl_sneaker():                                  # 7  low, solid, bottom
    P = blank()
    rect(P, 17, 26, 3, 24)                           # solid low shoe
    N = blank()
    rect(N, 0, 14, 2, 25)                            # empty upper -> garments
    rect(N, 8, 16, 14, 24)                           # no tall ankle shaft -> boot
    return P, N

def tmpl_bag():                                      # 8  rectangular block, flat top
    P = blank()
    rect(P, 9, 25, 5, 22)                            # block
    N = blank()
    rect(N, 4, 8, 5, 22)                             # strip above flat top -> garment shoulders
    rect(N, 6, 15, 0, 4)                             # corners -> garment sleeves
    rect(N, 6, 15, 23, 27)
    rect(N, 26, 27, 5, 22)
    return P, N

def tmpl_boot():                                     # 9  foot + tall ankle shaft (L shape)
    P = blank()
    rect(P, 18, 27, 4, 23)                           # foot
    rect(P, 8, 18, 14, 23)                           # ankle shaft (right)
    N = blank()
    rect(N, 0, 17, 0, 10)                            # empty above the toe
    rect(N, 0, 7, 11, 27)                            # empty very top
    return P, N

BUILDERS = [tmpl_tshirt, tmpl_trouser, tmpl_pullover, tmpl_dress, tmpl_coat,
            tmpl_sandal, tmpl_shirt, tmpl_sneaker, tmpl_bag, tmpl_boot]

# ----------------------------------------------------------------------------
# small separable Gaussian blur (from a formula), reflect padding
# ----------------------------------------------------------------------------
def gauss1d(sigma, radius):
    k = np.arange(-radius, radius + 1)
    g = np.exp(-(k ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def blur2d(m, sigma=1.0, radius=2):
    g = gauss1d(sigma, radius)
    p = np.pad(m, radius, mode="reflect")
    tmp = np.zeros_like(m, float)
    for i, w in enumerate(g):
        tmp += w * p[i:i + 28, radius:radius + 28]
    p2 = np.pad(tmp, radius, mode="reflect")
    out = np.zeros_like(m, float)
    for i, w in enumerate(g):
        out += w * p2[radius:radius + 28, i:i + 28]
    return out

def discriminant(P, N):
    """P/|P| - N/|N|, blurred -> score = mean_P(r) - mean_N(r)."""
    w = P / (P.sum() + EPS) - N / (N.sum() + EPS)
    return blur2d(w, sigma=1.0, radius=2)

def make_templates():
    K = 2                                            # as-drawn and left-right mirror
    rows = []
    for build in BUILDERS:
        P, N = build()
        w = discriminant(P, N)
        rows.append(w.ravel())
        rows.append(w[:, ::-1].ravel())              # mirror
    return np.stack(rows, axis=0), K                 # (20, 784)

# ----------------------------------------------------------------------------
# assemble network + weights
# ----------------------------------------------------------------------------
def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    W, K = make_templates()
    nclass = 10
    ntmpl = nclass * K

    # input blur (5x5, sigma 0.8) and a small negative bias so relu removes the
    # near-black background (bg ~ 0 from the stated data range) while preserving
    # the full item silhouette and thin features (trouser gap, straps).
    B0 = 0.06
    g = gauss1d(0.8, 2)
    gk = np.outer(g, g)
    blurW = gk[None, None, :, :].astype(np.float32)
    blurB = np.full(1, -B0, np.float32)

    weights = {
        "blur.W": blurW,
        "blur.b": blurB,
        "tmpl.W": W.astype(np.float32),
        "tmpl.b": np.zeros(ntmpl, np.float32),
    }

    net = {"layers": [
        {"name": "blur", "op": "conv", "inputs": ["image"],
         "out": 1, "kernel": [5, 5], "stride": 1, "padding": 2},
        {"name": "pos", "op": "relu", "inputs": ["blur"]},
        {"name": "flat", "op": "reshape", "inputs": ["pos"], "shape": [784]},
        {"name": "tmpl", "op": "dense", "inputs": ["flat"], "out": ntmpl},
        {"name": "grp", "op": "reshape", "inputs": ["tmpl"], "shape": [nclass, 1, K]},
        {"name": "mx", "op": "maxpool", "inputs": ["grp"], "size": [1, K], "stride": [1, K]},
        {"name": "out", "op": "reshape", "inputs": ["mx"], "shape": [10]},
    ]}

    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    build(out)
