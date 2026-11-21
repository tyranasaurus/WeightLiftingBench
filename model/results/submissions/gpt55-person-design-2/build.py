import json, sys
from pathlib import Path
import numpy as np

Y = np.array([0.299, 0.587, 0.114], np.float32)


def add_center(W, b, o, coeff, bias):
    W[o, :, 1, 1] = np.asarray(coeff, np.float32)
    b[o] = bias


def make_base():
    W = np.zeros((32, 3, 3, 3), np.float32)
    b = np.zeros(32, np.float32)
    # Colour half-spaces.  ReLU turns these into soft margins.
    add_center(W, b, 0,  [1, 0, -1],     -0.04)   # red above blue
    add_center(W, b, 1,  [0, 1, -1],     -0.01)   # green above blue
    add_center(W, b, 2,  [-1, 1, 0],      0.28)   # red not too far above green
    add_center(W, b, 3,  [1, -1, 0],      0.22)   # green not too far above red
    add_center(W, b, 4,  Y,              -0.14)   # not black
    add_center(W, b, 5, -Y,               0.90)   # not white
    add_center(W, b, 6,  [1, -1, 0],     -0.18)   # too red/orange for normal skin
    add_center(W, b, 7,  [-1, 1, 0],     -0.24)   # too green
    add_center(W, b, 8,  [-.5, -.5, 1],  -0.05)   # too blue
    add_center(W, b, 9, -Y,               0.45)   # dark
    add_center(W, b, 10, Y,              -0.55)   # bright
    add_center(W, b, 11, [1, 0, 0],      -0.45)   # red amount
    add_center(W, b, 12, [0, 1, 0],      -0.45)   # green amount
    add_center(W, b, 13, [0, 0, 1],      -0.45)   # blue amount

    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32) / 4.0
    sy = sx.T
    sd1 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]], np.float32) / 4.0
    sd2 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]], np.float32) / 4.0
    for o, K, sgn in [(14,sx,1),(15,sx,-1),(16,sy,1),(17,sy,-1),
                      (18,sd1,1),(19,sd1,-1),(20,sd2,1),(21,sd2,-1)]:
        for c in range(3):
            W[o, c] = sgn * Y[c] * K
        b[o] = -0.018
    # Local contrast, useful for eyes, hair, limb edges and textured clothing.
    ring = np.ones((3,3), np.float32) / 8.0
    ring[1,1] = 0.0
    for c in range(3):
        W[22, c] = Y[c] * ring;  W[22, c,1,1] -= Y[c]
        W[23, c] = -Y[c] * ring; W[23, c,1,1] += Y[c]
    b[22] = b[23] = -0.01
    add_center(W, b, 24, Y,              -0.72)   # white/highlight
    add_center(W, b, 25, -Y,              0.28)   # black/deep shadow
    add_center(W, b, 26, [1, -.5, -.5],  -0.12)   # saturated red/orange
    add_center(W, b, 27, [1, .8, -1.2],  -0.28)   # warm non-blue material
    add_center(W, b, 28, [0, 0, -1],      0.50)   # blue low
    add_center(W, b, 29, [0, -1, 0],      0.55)   # green low
    add_center(W, b, 30, [-1, 0, 1],     -0.02)   # blue above red
    add_center(W, b, 31, [0, 0, 0],       1.00)   # constant
    return W, b


def make_mix():
    W = np.zeros((16, 32, 1, 1), np.float32)
    b = np.zeros(16, np.float32)
    def m(o, terms, bias=0.0):
        for i,a in terms:
            W[o,i,0,0] = a
        b[o] = bias
    # Two skin scores: one stricter hue/brightness score and one looser shadow score.
    m(0, [(0,1.0),(1,1.0),(2,5.0),(3,1.2),(4,.7),(5,.5),(28,.25),
          (6,-6.0),(7,-2.0),(8,-2.0),(24,-.6),(26,-.8)], -1.05)
    m(1, [(0,.75),(1,.65),(2,2.5),(3,1.0),(4,.45),(5,.45),(28,.25),
          (6,-3.0),(7,-1.5),(8,-1.4),(24,-.35)], -0.62)
    m(2, [(9,1.0),(25,.6)], 0.0)                                      # dark
    m(3, [(10,1.0),(24,.55)], 0.0)                                     # bright
    m(4, [(14,.35),(15,.35),(16,.35),(17,.35),(18,.25),(19,.25),(20,.25),(21,.25),(22,.35),(23,.35)], -0.015)
    m(5, [(14,1.0),(15,1.0),(22,.2),(23,.2)], -0.015)                 # vertical boundary energy
    m(6, [(16,1.0),(17,1.0),(22,.2),(23,.2)], -0.015)                 # horizontal boundary energy
    m(7, [(22,1.0),(25,.25)], 0.0)                                    # dark spot
    m(8, [(23,1.0),(24,.25)], 0.0)                                    # bright spot
    m(9, [(27,1.0),(11,.35),(26,.25)], 0.0)                           # warm/red material
    m(10,[(12,1.0),(7,.5)], 0.0)                                      # green material
    m(11,[(13,1.0),(8,.45),(30,.35)], 0.0)                            # blue material
    m(12,[(6,1.0),(26,.8)], 0.0)                                      # saturated red/orange penalty cue
    m(13,[(14,.3),(15,.3),(16,.3),(17,.3),(18,.22),(19,.22),(20,.22),(21,.22),(22,.7),(23,.7)], -0.02)
    m(14,[(2,.25),(3,.2),(4,.2),(5,.15),(9,.35),(10,.25),(14,.18),(15,.18),(16,.18),(17,.18)], -0.10)
    m(15,[(31,1.0)], 0.0)                                             # constant
    return W, b


def pts_rect(y0, y1, x0, x1):
    return [(y,x) for y in range(y0, y1+1) for x in range(x0, x1+1)]


def ellipse_pts(k, cy, cx, ry, rx):
    return [(y,x) for y in range(k) for x in range(k) if ((y-cy)/ry)**2 + ((x-cx)/rx)**2 <= 1.0]


def add(W, o, ch, pts, val):
    k = W.shape[2]
    for y,x in pts:
        if 0 <= y < k and 0 <= x < k:
            W[o,ch,y,x] += val

S, SL, D, BR, E, V, H, DS, BS, WARM, GREEN, BLUE, REDX, TEX, OBJ, C = range(16)


def face_template(W, b, o, k, loose=False, side=0, bias=None):
    cy = (k-1)/2 + 0.2
    cx = (k-1)/2 + side
    ell = ellipse_pts(k, cy, cx, k*.42, k*.31)
    hair = pts_rect(0, max(1,k//3), max(1,k//2-2+side), min(k-2,k//2+2+side))
    eyes = [(k//2, k//2-1+side), (k//2, k//2+1+side)]
    eye_band = [(k//2, x) for x in range(max(1,k//2-2+side), min(k-1,k//2+3+side))]
    sides = [(y, max(0,k//2-3+side)) for y in range(k//3,k-1)] + [(y, min(k-1,k//2+3+side)) for y in range(k//3,k-1)]
    add(W,o,S,ell, .17 if not loose else .08)
    add(W,o,SL,ell, .08 if not loose else .17)
    add(W,o,D,hair, .13)
    add(W,o,D,eyes, .25)
    add(W,o,DS,eyes, .20)
    add(W,o,H,eye_band, .08)
    add(W,o,V,sides, .07)
    add(W,o,E,ell, .018)
    add(W,o,GREEN,ell, -.08); add(W,o,BLUE,ell, -.08); add(W,o,REDX,ell, -.10)
    b[o] = (-2.40 if not loose else -2.25) if bias is None else bias


def make_templates(scale, k, n):
    W = np.zeros((n, 16, k, k), np.float32)
    b = np.zeros(n, np.float32)
    c = k//2
    if scale == 2:
        # Faces and heads at about 8--24 pixels in the original image.
        face_template(W,b,0,k,False,0,-2.45)
        face_template(W,b,1,k,True,0,-2.25)
        face_template(W,b,2,k,False,-1,-2.25)
        face_template(W,b,3,k,False,1,-2.25)
        # Close-up face fragments: skin with a dark upper band and eye line.
        add(W,4,S,ellipse_pts(k,c,c,k*.48,k*.35),.13); add(W,4,SL,ellipse_pts(k,c,c,k*.48,k*.35),.08)
        add(W,4,D,pts_rect(0,2,1,k-2),.12); add(W,4,DS,[(c,c-1),(c,c+1)],.22); add(W,4,H,[(c,x) for x in range(1,k-1)],.07)
        add(W,4,REDX,ellipse_pts(k,c,c,k*.48,k*.35),-.08); b[4] = -2.25
        # Head over shoulders: skin/dark head, wider lower body, vertical sides.
        head = pts_rect(0,2,c-1,c+1); torso = pts_rect(3,k-1,c-2,c+2)
        add(W,5,S,head,.22); add(W,5,SL,head,.12); add(W,5,D,pts_rect(0,1,c-2,c+2),.10)
        add(W,5,OBJ,torso,.10); add(W,5,V,pts_rect(2,k-1,c-3,c-3)+pts_rect(2,k-1,c+3,c+3),.12); add(W,5,H,pts_rect(3,3,c-3,c+3),.08); b[5] = -1.85
        # Skin blobs in several places: groups of people or hands/faces.
        for o, blobs, bb in [
            (6, [(1,1),(1,3),(2,5),(4,2),(5,4)], -1.30),
            (7, [(1,2),(1,4),(3,1),(3,5),(5,3)], -1.28)]:
            for y,x in blobs:
                add(W,o,S,pts_rect(y,y,x,x),.32); add(W,o,SL,pts_rect(y,y,x,x),.18); add(W,o,D,[(max(0,y-1),x)],.10)
            add(W,o,E,pts_rect(0,k-1,0,k-1),.02); b[o] = bb
        # Small upright body with skin/hat head.
        for o, sk, bb in [(8,S,-1.70),(9,SL,-1.55)]:
            add(W,o,sk,pts_rect(0,1,c-1,c+1),.22); add(W,o,D,pts_rect(0,0,c-1,c+1),.10)
            add(W,o,OBJ,pts_rect(2,k-1,c-1,c+1),.13); add(W,o,V,pts_rect(1,k-1,c-2,c-2)+pts_rect(1,k-1,c+2,c+2),.13)
            add(W,o,H,pts_rect(2,2,c-2,c+2),.08); b[o] = bb
        # Diagonal limb/body cues with a head-like blob.
        diag1 = [(1,2),(2,3),(3,4),(4,5),(2,1),(3,2),(4,3),(5,4)]
        diag2 = [(1,4),(2,3),(3,2),(4,1),(2,5),(3,4),(4,3),(5,2)]
        for o, diag in [(10,diag1),(11,diag2)]:
            add(W,o,E,diag,.13); add(W,o,OBJ,diag,.05); add(W,o,SL,[(1,c)],.24); add(W,o,D,[(0,c)],.10); b[o] = -1.15
    elif scale == 4:
        # Medium faces.
        face_template(W,b,0,k,False,0,-2.35)
        face_template(W,b,1,k,True,0,-2.15)
        # Upright standing/seated figures with a head above a torso.
        head = pts_rect(0,1,c-1,c+1); body = pts_rect(2,k-1,c-1,c+1)
        legs = pts_rect(k-3,k-1,c-2,c-2) + pts_rect(k-3,k-1,c+2,c+2)
        for o, sk, bb in [(2,S,-1.65),(3,SL,-1.50),(4,D,-1.55),(5,OBJ,-1.45)]:
            add(W,o,sk,head,.20); add(W,o,D,pts_rect(0,0,c-1,c+1),.10)
            add(W,o,OBJ,body,.13); add(W,o,V,pts_rect(1,k-1,c-2,c-2)+pts_rect(1,k-1,c+2,c+2),.14)
            add(W,o,H,pts_rect(2,2,c-2,c+2),.08); add(W,o,E,legs,.08); b[o] = bb
        # Head and shoulders / bust.
        for o, sk, bb in [(6,S,-1.80),(7,SL,-1.62)]:
            add(W,o,sk,pts_rect(0,2,c-1,c+1),.18); add(W,o,D,pts_rect(0,0,c-2,c+2),.12)
            add(W,o,OBJ,pts_rect(3,k-1,c-2,c+2),.10); add(W,o,V,pts_rect(2,k-1,c-3,c-3)+pts_rect(2,k-1,c+3,c+3),.12)
            add(W,o,H,pts_rect(3,3,c-3,c+3),.09); b[o] = bb
        # Athletes/children: diagonal or horizontal bodies with a skin head/hand.
        for o, diag, bb in [(8,[(1,1),(2,2),(3,3),(4,4),(5,5),(2,1),(3,2),(4,3)],-1.20),
                            (9,[(1,5),(2,4),(3,3),(4,2),(5,1),(2,5),(3,4),(4,3)],-1.20),
                            (10,[(3,x) for x in range(1,6)] + [(4,x) for x in range(1,6)],-1.25)]:
            add(W,o,E,diag,.13); add(W,o,OBJ,diag,.07); add(W,o,SL,[(1,c),(2,c-1)],.18); add(W,o,D,[(0,c)],.08); b[o] = bb
        # Groups/crowds: repeated heads with surrounding edge clutter.
        for o, blobs, bb in [(11,[(1,1),(1,3),(1,5),(3,2),(3,4)],-1.25),
                             (12,[(1,2),(1,4),(2,1),(2,5),(4,3)],-1.25)]:
            for y,x in blobs:
                add(W,o,S,[(y,x)],.24); add(W,o,SL,[(y,x)],.16); add(W,o,D,[(max(0,y-1),x)],.09)
            add(W,o,E,pts_rect(0,k-1,0,k-1),.025); b[o] = bb
        # Large skin-plus-eye evidence.
        add(W,13,S,ellipse_pts(k,c,c,k*.48,k*.36),.12); add(W,13,SL,ellipse_pts(k,c,c,k*.48,k*.36),.07)
        add(W,13,D,pts_rect(0,2,1,k-2),.11); add(W,13,DS,[(c,c-1),(c,c+1)],.20); add(W,13,H,pts_rect(c,c,1,k-2),.07); b[13] = -2.10
        # Conservative no-skin silhouette, kept weak in the readout.
        add(W,14,V,pts_rect(0,k-1,c-2,c-2)+pts_rect(0,k-1,c+2,c+2),.16); add(W,14,OBJ,pts_rect(1,k-1,c-1,c+1),.12); add(W,14,H,pts_rect(1,2,c-2,c+2),.08); b[14] = -1.45
        add(W,15,E,pts_rect(0,k-1,c-1,c+1),.12); add(W,15,DS,pts_rect(0,1,c-1,c+1),.12); add(W,15,OBJ,pts_rect(2,k-1,c-1,c+1),.08); b[15] = -1.35
    else:
        # Coarse 8x scale: large people and many-person scenes.
        face_template(W,b,0,k,False,0,-1.55)
        face_template(W,b,1,k,True,0,-1.45)
        head = pts_rect(0,1,c-1,c+1); torso = pts_rect(2,k-1,c-1,c+1)
        for o, sk, bb in [(2,S,-1.20),(3,SL,-1.10),(4,OBJ,-1.05)]:
            add(W,o,sk,head,.20); add(W,o,D,pts_rect(0,0,c-1,c+1),.10); add(W,o,OBJ,torso,.13)
            add(W,o,V,pts_rect(1,k-1,c-2,c-2)+pts_rect(1,k-1,c+2,c+2),.12); add(W,o,H,pts_rect(2,2,c-2,c+2),.08); b[o] = bb
        # Crowds / multiple people.
        for o, blobs, bb in [(5,[(1,1),(1,3),(2,2),(3,1),(3,3)],-1.05),
                             (6,[(0,2),(1,1),(1,3),(3,2),(4,1),(4,3)],-1.10)]:
            for y,x in blobs:
                add(W,o,S,[(y,x)],.22); add(W,o,SL,[(y,x)],.15); add(W,o,D,[(max(0,y-1),x)],.08)
            add(W,o,E,pts_rect(0,k-1,0,k-1),.025); b[o] = bb
        # Broad upper-body/portrait fragments.
        add(W,7,S,pts_rect(0,k-1,1,k-2),.11); add(W,7,SL,pts_rect(0,k-1,1,k-2),.07); add(W,7,D,pts_rect(0,1,0,k-1),.10); add(W,7,H,pts_rect(c,c,0,k-1),.06); b[7] = -1.45
        add(W,8,SL,pts_rect(0,k-1,0,k-1),.07); add(W,8,E,pts_rect(0,k-1,0,k-1),.045); add(W,8,D,pts_rect(0,1,0,k-1),.06); b[8] = -1.35
        add(W,9,V,pts_rect(0,k-1,1,1)+pts_rect(0,k-1,k-2,k-2),.14); add(W,9,OBJ,pts_rect(1,k-1,c,c),.12); add(W,9,DS,pts_rect(0,1,c,c),.12); b[9] = -1.10
    return W, b


def make(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    layers = []
    weights = {}
    branch_specs = [(2, 48, 7, 12), (4, 24, 7, 16), (8, 12, 5, 10)]
    score_names = []
    for scale, dim, k, nt in branch_specs:
        suf = str(scale)
        layers += [
            {"name":"p"+suf, "op":"avgpool", "inputs":["image"], "size":[scale,scale], "stride":[scale,scale]},
            {"name":"base"+suf, "op":"conv", "inputs":["p"+suf], "out":32, "kernel":[3,3], "stride":1, "padding":1},
            {"name":"base"+suf+"r", "op":"relu", "inputs":["base"+suf]},
            {"name":"mix"+suf, "op":"conv", "inputs":["base"+suf+"r"], "out":16, "kernel":[1,1], "stride":1, "padding":0},
            {"name":"mix"+suf+"r", "op":"relu", "inputs":["mix"+suf]},
            {"name":"templ"+suf, "op":"conv", "inputs":["mix"+suf+"r"], "out":nt, "kernel":[k,k], "stride":1, "padding":k//2},
            {"name":"templ"+suf+"r", "op":"relu", "inputs":["templ"+suf]},
            {"name":"max"+suf, "op":"maxpool", "inputs":["templ"+suf+"r"], "size":[dim,dim], "stride":[dim,dim]},
            {"name":"avg"+suf, "op":"avgpool", "inputs":["templ"+suf+"r"], "size":[dim,dim], "stride":[dim,dim]},
            {"name":"scorem"+suf, "op":"dense", "inputs":["max"+suf], "out":2},
            {"name":"scorea"+suf, "op":"dense", "inputs":["avg"+suf], "out":2},
        ]
        score_names += ["scorem"+suf, "scorea"+suf]
        BW, Bb = make_base(); MW, Mb = make_mix(); TW, Tb = make_templates(scale, k, nt)
        weights["base"+suf+".W"] = BW; weights["base"+suf+".b"] = Bb
        weights["mix"+suf+".W"] = MW; weights["mix"+suf+".b"] = Mb
        weights["templ"+suf+".W"] = TW; weights["templ"+suf+".b"] = Tb
        # Readout: default class is no-person; only positive handmade evidence votes for person.
        Wm = np.zeros((2, nt), np.float32); bm = np.zeros(2, np.float32)
        Wa = np.zeros((2, nt), np.float32); ba = np.zeros(2, np.float32)
        if scale == 2:
            Wm[1,:5] = [0.82,0.72,0.62,0.62,0.58]
            Wm[1,5:10] = [0.50,0.36,0.36,0.38,0.34]
            Wm[1,10:12] = [0.24,0.24]
            Wa[1,6:8] = [2.0,2.0]
        elif scale == 4:
            Wm[1,:2] = [0.62,0.56]
            Wm[1,2:8] = [0.42,0.38,0.28,0.26,0.42,0.38]
            Wm[1,8:14] = [0.22,0.22,0.20,0.30,0.30,0.54]
            Wm[1,14:16] = [0.13,0.13]
            Wa[1,11:13] = [1.8,1.8]
        else:
            Wm[1,:2] = [0.45,0.40]
            Wm[1,2:5] = [0.28,0.25,0.22]
            Wm[1,5:7] = [0.34,0.34]
            Wm[1,7:10] = [0.38,0.30,0.18]
            Wa[1,5:7] = [1.4,1.4]
            Wa[1,8] = 0.45
        weights["scorem"+suf+".W"] = Wm; weights["scorem"+suf+".b"] = bm
        weights["scorea"+suf+".W"] = Wa; weights["scorea"+suf+".b"] = ba

    # A single negative prior/threshold for the person logit.  Class 0 remains zero.
    layers.append({"name":"prior", "op":"dense", "inputs":["max2"], "out":2})
    PW = np.zeros((2, 12), np.float32); Pb = np.array([0.0, -1.55], np.float32)
    weights["prior.W"] = PW; weights["prior.b"] = Pb
    layers.append({"name":"scores", "op":"add", "inputs":score_names + ["prior"]})

    (outdir / "network.json").write_text(json.dumps({"layers":layers}, separators=(",",":")))
    np.savez(outdir / "weights.npz", **weights)


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
