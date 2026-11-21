"""Builds a hand-designed nearest-prototype digit classifier (no training data used).

Architecture:
  image -> norm -> conv(8 oriented ridge filters + 1 lowpass, 9x9) -> relu
        -> avgpool(4x4) -> reshape -> norm -> dense(prototype match)
        -> reshape -> maxpool(over the prototypes of each class) -> reshape(10)

Every weight comes from this source alone:
  * ridge.W holds 8 oriented ridge (bar) detectors and 1 Gaussian lowpass, all from
    closed-form formulas (2nd-derivative-of-Gaussian across the stroke).  They turn an
    image into a map of "which stroke orientation is present where".
  * match.W rows are idealised digit templates drawn from general knowledge of how the
    ten digits look (strokes along typed pen paths, loops, bars).  Each template is run
    through the SAME norm+conv+relu+avgpool pipeline, then the feature vector is
    standardised (the second norm) and L2-normalised, so each unit computes the cosine
    similarity between the input's orientation-feature descriptor and the template's.
    The per-class max over its prototypes, then argmax, classifies by the best-matching
    idealised prototype (match.b = 0).
No image, statistic, or label from the training set enters any number here.  The five
allowed checks were used only to confirm the network runs and to read overall accuracy.
"""
import sys, os, json
import numpy as np

# ===================== geometry helpers (canonical coords: x right, y down) =====================
def arc(cx, cy, rx, ry, d0, d1, n=60):
    t = np.linspace(np.radians(d0), np.radians(d1), n)
    return list(zip(cx + rx*np.cos(t), cy + ry*np.sin(t)))

def line(p0, p1, n=20):
    t = np.linspace(0, 1, n)
    return list(zip(p0[0]+(p1[0]-p0[0])*t, p0[1]+(p1[1]-p0[1])*t))

def ellipse(cx, cy, rx, ry, n=80):
    return arc(cx, cy, rx, ry, 0, 360, n)

def _smooth(pts, n=14):
    pts = np.array(pts, float)
    if len(pts) < 3:
        return line(tuple(pts[0]), tuple(pts[-1]), 20)
    out = []
    ext = np.vstack([pts[0], pts, pts[-1]])
    for i in range(1, len(ext)-2):
        p0,p1,p2,p3 = ext[i-1],ext[i],ext[i+1],ext[i+2]
        t = np.linspace(0,1,n)[:,None]
        a = 0.5*((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t**2 + (-p0+3*p1-3*p2+p3)*t**3)
        out.extend([tuple(r) for r in a])
    return out

# ===================== the ten digit shape models (4 styles each) =====================
def d0():
    return [[ellipse(0.0,0.0,0.52,0.82)],
            [ellipse(0.0,0.0,0.44,0.84)],
            [ellipse(0.02,0.0,0.60,0.80)],
            [ellipse(0.0,0.0,0.50,0.80), line((-0.3,0.35),(0.3,-0.35))]]
def d1():
    stem = line((0.0,-0.85),(0.0,0.86),40); flag = line((-0.30,-0.50),(0.02,-0.85),15)
    foot = line((-0.34,0.85),(0.36,0.85),15)
    return [[stem],[stem,flag],[stem,flag,foot],[line((0.08,-0.85),(-0.04,0.86),40),flag]]
def d2():
    top = arc(0.0,-0.40,0.46,0.42,195,430,50); diag = line((0.20,-0.02),(-0.5,0.72),30)
    base = line((-0.5,0.74),(0.56,0.70),25)
    s0 = [top,diag,base]
    s1 = [arc(0.0,-0.42,0.42,0.44,185,420,50), line((0.26,-0.08),(-0.46,0.72),30), line((-0.46,0.74),(0.54,0.72),25)]
    s2 = [top, line((0.20,-0.02),(-0.35,0.55),22), arc(-0.18,0.62,0.22,0.18,270,470,20), base]
    s3 = [arc(0.0,-0.44,0.40,0.40,200,420,45), line((0.18,-0.05),(-0.5,0.74),30), line((-0.5,0.76),(0.52,0.74),25)]
    return [s0,s1,s2,s3]
def d3():
    up = arc(0.0,-0.40,0.42,0.42,250,470,45); lo = arc(0.0,0.40,0.44,0.44,250,470,45)
    s0 = [up,lo]
    s1 = [line((-0.35,-0.78),(0.4,-0.78),15), arc(0.1,-0.35,0.33,0.38,290,470,30), arc(0.0,0.40,0.46,0.46,250,470,45)]
    s2 = [arc(0.05,-0.40,0.40,0.42,255,460,45), arc(0.05,0.40,0.47,0.45,250,475,45)]
    s3 = [arc(0.0,-0.38,0.38,0.40,250,470,45), arc(0.0,0.42,0.42,0.44,250,470,45)]
    return [s0,s1,s2,s3]
def d4():
    diag = line((0.18,-0.82),(-0.50,0.18),30); bar = line((-0.50,0.18),(0.52,0.18),25)
    vert = line((0.30,-0.82),(0.30,0.86),40)
    s0 = [diag,bar,vert]
    s1 = [line((0.30,-0.82),(-0.50,0.18),30), bar, vert]
    s2 = [line((0.12,-0.82),(-0.48,0.22),30), line((-0.48,0.22),(0.50,0.22),25), line((0.26,-0.60),(0.26,0.86),35)]
    s3 = [diag, bar, line((0.30,-0.82),(0.34,0.86),40)]
    return [s0,s1,s2,s3]
def d5():
    topbar = line((-0.42,-0.80),(0.50,-0.80),25); lvert = line((-0.44,-0.80),(-0.44,0.02),25)
    belly = arc(0.0,0.32,0.47,0.47,250,470,45); conn = line((-0.44,0.02),(-0.17,-0.11),8)
    s0 = [topbar,lvert,conn,belly]
    s1 = [line((-0.40,-0.78),(0.46,-0.84),25), lvert, conn, belly]
    s2 = [topbar, line((-0.44,-0.80),(-0.44,0.05),25), arc(0.02,0.34,0.46,0.45,255,460,45)]
    s3 = [line((-0.42,-0.78),(0.44,-0.78),25), line((-0.42,-0.78),(-0.46,0.06),25), arc(0.0,0.34,0.44,0.45,250,470,45)]
    return [s0,s1,s2,s3]
def d6():
    cane = _smooth([(0.40,-0.74),(0.16,-0.84),(-0.16,-0.66),(-0.40,-0.26),(-0.48,0.12),(-0.47,0.30)])
    s0 = [cane, ellipse(-0.02,0.42,0.46,0.40)]
    s1 = [_smooth([(0.34,-0.72),(0.06,-0.82),(-0.22,-0.58),(-0.42,-0.1),(-0.44,0.25)]), ellipse(0.0,0.44,0.44,0.38)]
    s2 = [_smooth([(0.42,-0.70),(0.1,-0.84),(-0.26,-0.5),(-0.45,0.05)]), ellipse(-0.02,0.42,0.48,0.44)]
    s3 = [_smooth([(0.30,-0.76),(0.0,-0.8),(-0.3,-0.45),(-0.44,0.1),(-0.42,0.3)]), ellipse(0.0,0.45,0.45,0.40)]
    return [s0,s1,s2,s3]
def d7():
    topbar = line((-0.50,-0.80),(0.50,-0.78),25); diag = line((0.50,-0.78),(-0.24,0.86),35)
    cross = line((-0.06,0.06),(0.30,0.04),12)
    s0 = [topbar,diag]
    s1 = [topbar,diag,cross]
    s2 = [line((-0.48,-0.78),(0.52,-0.80),25), _smooth([(0.52,-0.80),(0.2,0.0),(-0.10,0.86)])]
    s3 = [topbar, line((-0.42,-0.80),(-0.42,-0.52),10), diag]
    return [s0,s1,s2,s3]
def d8():
    s0 = [ellipse(0.0,-0.42,0.40,0.42), ellipse(0.0,0.42,0.48,0.44)]
    s1 = [ellipse(0.0,-0.44,0.36,0.40), ellipse(0.0,0.42,0.44,0.44)]
    s2 = [ellipse(0.02,-0.40,0.42,0.42), ellipse(-0.02,0.44,0.50,0.42)]
    s3 = [ellipse(0.0,-0.42,0.38,0.44), ellipse(0.0,0.44,0.46,0.42)]
    return [s0,s1,s2,s3]
def d9():
    s0 = [ellipse(0.02,-0.40,0.44,0.42), _smooth([(0.46,-0.40),(0.44,0.0),(0.30,0.45),(0.08,0.86)])]
    s1 = [ellipse(0.0,-0.42,0.42,0.40), line((0.42,-0.40),(0.20,0.86),30)]
    s2 = [ellipse(0.02,-0.40,0.46,0.44), _smooth([(0.48,-0.38),(0.46,0.1),(0.34,0.5),(0.18,0.86)])]
    s3 = [ellipse(0.0,-0.44,0.40,0.42), _smooth([(0.40,-0.42),(0.40,0.1),(0.26,0.5),(0.06,0.86)])]
    return [s0,s1,s2,s3]
STYLES = {0:d0,1:d1,2:d2,3:d3,4:d4,5:d5,6:d6,7:d7,8:d8,9:d9}

# ===================== rendering =====================
SCALE = 11.0; CEN = 13.5
_GY, _GX = np.mgrid[0:28, 0:28]
_GRID = np.stack([_GX.ravel(), _GY.ravel()], 1).astype(float)

def _shift(img, dy, dx):
    out = np.zeros_like(img); h, w = img.shape
    ys0,ys1 = max(0,dy),min(h,h+dy); xs0,xs1 = max(0,dx),min(w,w+dx)
    yd0,yd1 = max(0,-dy),min(h,h-dy); xd0,xd1 = max(0,-dx),min(w,w-dx)
    out[ys0:ys1, xs0:xs1] = img[yd0:yd1, xd0:xd1]; return out

def render(polylines, slant=0.0, width=1.0, thick=0.55, aa=0.8):
    pts = []
    for pl in polylines:
        P = np.array(pl, float)
        xs = P[:,0]*width + slant*(-P[:,1])
        X = CEN + xs*SCALE; Y = CEN + P[:,1]*SCALE
        for i in range(len(P)-1):
            d = np.hypot(X[i+1]-X[i], Y[i+1]-Y[i]); m = max(2, int(d/0.35)+1)
            tt = np.linspace(0,1,m)
            pts.append(np.stack([X[i]+(X[i+1]-X[i])*tt, Y[i]+(Y[i+1]-Y[i])*tt],1))
    pts = np.concatenate(pts,0)
    d2 = (_GRID[:,None,0]-pts[None,:,0])**2 + (_GRID[:,None,1]-pts[None,:,1])**2
    dist = np.sqrt(d2.min(1)).reshape(28,28)
    img = np.clip(1.0 - (dist-thick)/aa, 0.0, 1.0)
    s = img.sum()
    if s > 1e-6:
        cy = (img*_GY).sum()/s; cx = (img*_GX).sum()/s
        img = _shift(img, int(round(CEN-cy)), int(round(CEN-cx)))
    return img

SLANTS = [-0.18, -0.05, 0.08, 0.20, 0.33]; WIDTHS = [0.85, 1.0, 1.15]; THICKS = [0.25, 0.55, 0.95]
PER_CLASS = len(STYLES[0]())*len(SLANTS)*len(WIDTHS)*len(THICKS)  # 64

def build_prototypes():
    protos, labels = [], []
    for dgt in range(10):
        for st in STYLES[dgt]():
            for sl in SLANTS:
                for wd in WIDTHS:
                    for th in THICKS:
                        protos.append(render(st, sl, wd, th)); labels.append(dgt)
    return np.array(protos, np.float64), np.array(labels, np.int64)

# ===================== oriented ridge filter bank (from formulas) =====================
KS = 9; PAD = KS//2; NORI = 8; SIG_ALONG = 2.5; SIG_PERP = 1.3
POOL = 4; GRID = (28 - POOL)//POOL + 1          # 7
NORM_EPS = 1e-5

def filter_bank():
    ax = np.arange(KS) - PAD
    DX, DY = np.meshgrid(ax, ax)
    filts = []
    for k in range(NORI):
        th = np.pi*k/NORI
        along = DX*np.cos(th) + DY*np.sin(th)
        perp  = -DX*np.sin(th) + DY*np.cos(th)
        val = (1 - (perp/SIG_PERP)**2) * np.exp(-(perp**2)/(2*SIG_PERP**2) - (along**2)/(2*SIG_ALONG**2))
        val = val - val.mean()
        val = val / np.sqrt((val**2).sum())
        filts.append(val)
    g = np.exp(-(DX**2+DY**2)/(2*1.2**2)); g = g/g.sum()
    filts.append(g)
    return np.array(filts)                       # (9,9,9)

def conv_same(img, filt):
    xp = np.pad(img, PAD); out = np.zeros((28,28))
    for dy in range(KS):
        for dx in range(KS):
            out += filt[dy,dx]*xp[dy:dy+28, dx:dx+28]
    return out

def avgpool(x):
    return x.reshape(GRID, POOL, GRID, POOL).mean(axis=(1,3))

def feature_vec(img, F):
    mu = img.mean(); v = ((img-mu)**2).mean()
    x = (img-mu)/np.sqrt(v+NORM_EPS)
    chans = [avgpool(np.maximum(conv_same(x, F[k]), 0.0)) for k in range(len(F))]
    return np.concatenate([c.ravel() for c in chans])

# ===================== assemble weights and network.json =====================
def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    F = filter_bank(); NF = len(F)
    protos, labels = build_prototypes()
    N = len(protos); FD = NF*GRID*GRID
    W = np.zeros((N, FD), np.float64); b = np.zeros(N, np.float64)
    for i in range(N):
        v = feature_vec(protos[i], F)
        t = v - v.mean(); t = t/np.sqrt((t*t).mean()+NORM_EPS)   # match the feature-norm op
        nrm = np.sqrt((t*t).sum())
        if nrm > 1e-9: t = t/nrm                                 # unit norm -> cosine matching
        W[i] = t
    convW = F.reshape(NF, 1, KS, KS).astype(np.float32)
    weights = {
        "ridge.W": convW,
        "ridge.b": np.zeros(NF, np.float32),
        "match.W": W.astype(np.float32),
        "match.b": b.astype(np.float32),
    }
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)
    net = {"layers": [
        {"name":"nrm",  "op":"norm",    "inputs":["image"]},
        {"name":"ridge","op":"conv",    "inputs":["nrm"], "out": int(NF), "kernel":[KS,KS], "stride":1, "padding":int(PAD)},
        {"name":"rect", "op":"relu",    "inputs":["ridge"]},
        {"name":"feat", "op":"avgpool", "inputs":["rect"], "size":[POOL,POOL], "stride":[POOL,POOL]},
        {"name":"fflat","op":"reshape", "inputs":["feat"], "shape":[1, int(NF*GRID*GRID), 1]},
        {"name":"fnrm", "op":"norm",    "inputs":["fflat"]},
        {"name":"match","op":"dense",   "inputs":["fnrm"], "out": int(N)},
        {"name":"grid", "op":"reshape", "inputs":["match"], "shape":[1, 10, int(PER_CLASS)]},
        {"name":"pool", "op":"maxpool", "inputs":["grid"], "size":[1, int(PER_CLASS)], "stride":[1, int(PER_CLASS)]},
        {"name":"out",  "op":"reshape", "inputs":["pool"], "shape":[10]},
    ]}
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
