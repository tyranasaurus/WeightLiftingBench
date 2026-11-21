"""Hand-designed MNIST digit net. No training data is read or used.

Weights come from: (a) fixed formula filters (Gabor / Gaussian), and
(b) prototypes of each digit rendered from typed pen-path stroke models
(general knowledge of how digits are written), passed through the same
fixed frontend; the readout is a nearest-prototype table in feature space.
"""
import sys, json, math, os
import numpy as np

# ---------------------------------------------------------------- pen paths
# Each digit: list of writing forms; a form is a list of strokes; a stroke is
# a polyline of (x, y) control points, y in [0,20] top->bottom, x centred on 0.
DIG = {
 0: [[(0,.6),(5.2,2.2),(7.2,6),(7.4,10),(7.,14),(5.,17.8),(0,19.4),(-5.,17.8),(-7.,14),(-7.4,10),(-7.2,6),(-5.2,2.2),(0,.6)],
      [(0,.6),(4.3,2.2),(6.,6),(6.1,10),(5.8,14),(4.1,17.8),(0,19.4),(-4.1,17.8),(-5.8,14),(-6.1,10),(-6.,6),(-4.3,2.2),(0,.6)],
      [(0,.5),(4.5,2),(6.5,5.5),(6.8,10),(6.2,14.5),(4.,18),(0,19.5),(-4.2,18),(-6.4,14.5),(-6.8,10),(-6.4,5.5),(-4.4,2),(0,.5)],
      [(0,.5),(4.8,2.5),(6.6,6),(6.8,10),(6.2,14.5),(3.8,18),(0,19.6),(-3.8,18),(-6.2,14.5),(-6.8,10),(-6.6,6),(-4.8,2.5),(0,.5)],
      [(0,.8),(5.6,2.4),(7.6,6),(7.8,10),(7.4,14),(5.4,17.6),(0,19.2),(-5.4,17.6),(-7.4,14),(-7.8,10),(-7.6,6),(-5.6,2.4),(0,.8)]],
 1: [[(-3.5,4.5),(0,1),(.3,19)],
      [(0,1),(0,19)],
      [(-4,4),(0,.8),(.2,19),(-4.5,19),(4.5,19)],
      [(-3,4.5),(.5,1),(0,19)],
      [(-2.5,3.5),(.5,1),(1.5,19)]],
 2: [[(-6,5),(-5,2.5),(-2,1),(1.5,1),(4.5,2.5),(6,5),(5.8,7.5),(4,10),(0,13.5),(-4,16.5),(-6.5,19),(-1,18.8),(3,19),(6.5,19)],
      [(-6.5,4),(-4,1.5),(0,1),(4,2),(6,4.5),(5.5,7.5),(3,10.5),(-1,14),(-5,17),(-6.5,19),(0,19),(6,19)],
      [(-6,2),(0,1.5),(5.5,2),(6,4.5),(4.5,8),(0,12.5),(-4.5,16.5),(-6.5,19),(0,19.2),(6.5,19)],
      [(-6,5),(-5,2.5),(-2,1),(1.5,1),(4.5,2.5),(6,5),(5.8,7.5),(4,10),(0,13.5),(-3.5,16.5),(-5,18.5),(-2,19.6),(1.5,19),(4,17.5)],
      [(-5.5,2.5),(-.5,1.8),(5.5,2.2),(6,4.2),(-5.5,18.5),(0,19),(6,19)]],
 3: [[(-5.5,3),(-2.5,1),(1.5,1),(4.5,2.5),(5.5,5),(4.5,7.5),(1.5,9),(-1,9.5),(2,10),(4.8,11.5),(6,14),(5.5,16.5),(3,18.5),(0,19.3),(-3.5,18.7),(-6,17)],
      [(-6,2),(0,1.5),(4.5,2),(5.5,4.5),(4,7.5),(.5,9.3),(-1.5,9.6),(2,10.3),(5,12),(6,14.5),(5,17),(2.5,18.8),(-1,19.2),(-4.5,18),(-6.5,16)],
      [(-4,3.5),(-1,1.5),(2.5,1.5),(5,3),(5.8,5.5),(4.5,8),(1.5,9.4),(-.5,9.8),(2.5,10.5),(5.2,12.5),(6,15),(4.8,17.5),(2,19),(-1.5,19.2),(-4.5,18),(-6.5,16.2)],
      [(-4.5,4),(-2,1.5),(1.5,1.2),(4.2,2.6),(5.2,5),(4.2,7.4),(1.5,8.9),(-.8,9.4),(1.8,10.2),(4.6,11.8),(5.6,14.2),(5,16.6),(2.6,18.6),(-.6,19.2),(-3.8,18.4),(-6,16.6)],
      [(-6,1.8),(-.5,1.4),(4.8,2),(5.8,4.5),(4.5,7.5),(1,9.2),(-1.5,9.6),(2,10.3),(5,12),(6,14.5),(5,17),(2.5,18.8),(-1,19.2),(-4.5,18),(-6.5,16)]],
 4: [[(4.5,.8),(4.3,19)],[(4.,2.5),(-1.5,9),(-6.8,13.8),(7.,13.5)]],
 41: [[(4.8,1),(4.6,19)],[(4.8,1.5),(-.5,7.5),(-6,13.5),(6.8,13.2)]],
 42: [[(3.8,.8),(3.6,19)],[(3.5,2.8),(-2,9.5),(-7,14.2),(6.5,13.8)]],
 43: [[(4.2,.8),(4.,19)],[(3.8,3),(-1.5,9.2),(-6.2,13.6),(5.5,13.3)]],
 44: [[(5.2,.9),(4.6,19)],[(4.4,3),(-1,9.8),(-6.5,14.6),(6.2,14.2)]],
 5: [[(5.5,1.2),(-3.5,1.6),(-5,8),(-1.5,6.8),(1.5,7),(4.2,8.5),(5.6,11),(5.4,14),(4,16.8),(1,18.8),(-2.5,19.2),(-5.5,17.8)],
      [(4.5,2),(-3,1.2),(-4.5,7.5),(-1,6.5),(2,6.8),(4.6,8.5),(5.8,11.5),(5.2,14.5),(3.5,17.2),(.5,19),(-3,19),(-5.8,17.5)],
      [(5,1),(-4,1.5),(-5.5,8.5),(-2,7),(1,7),(4,8.3),(5.5,10.8),(5.5,13.8),(4.2,16.6),(1.5,18.6),(-2,19),(-5,17.6),(-6.5,15.5)],
      [(4.8,1.4),(-3,1.8),(-4.2,7),(-1.2,6.2),(1.6,6.6),(4.4,8.2),(5.8,11),(5.6,14.2),(4.2,17),(1.2,19),(-2.4,19.3),(-5.4,17.8)],
      [[(5.8,1.2),(-4.5,1.4)],[(-4.2,1.6),(-5,8),(-1.5,6.8),(1.5,7),(4.2,8.5),(5.6,11),(5.4,14),(4,16.8),(1,18.8),(-2.5,19.2),(-5.5,17.8)]]],
 6: [[(4.5,1),(1,1.5),(-2.5,4),(-5,8),(-6,12),(-5.5,15.5),(-3.5,18),(0,19.2),(3,18.4),(5,16.2),(5.3,13.2),(4,10.8),(1.2,9.6),(-1.8,9.9),(-4.3,11.6),(-5.6,14)],
      [(3.5,.8),(.5,4),(-3,8),(-5.2,12),(-5.5,15.5),(-3.5,18.2),(0,19.3),(3.2,18.3),(5.2,16),(5.4,13),(4,10.6),(1,9.5),(-2,9.8),(-4.5,11.5),(-5.7,14)],
      [(4.8,1.2),(1.5,1.8),(-2,4.5),(-4.5,8.5),(-5.6,12.5),(-5,16),(-2.8,18.4),(.5,19.2),(3.5,18.2),(5.3,15.8),(5.4,12.8),(4,10.5),(1,9.4),(-2,9.8),(-4.6,11.8)],
      [(4.2,1),(1,1.6),(-2.4,4.2),(-4.8,8.2),(-5.8,12.2),(-5.3,15.7),(-3.3,18.1),(0,19.3),(3,18.5),(5,16.3),(5.3,13.3),(4,10.9),(1.2,9.7),(-1.8,10),(-4.3,11.7),(-5.5,14.1)]],
 7: [[(-7,2),(0,1.5),(7,1.9),(.5,19)],
      [(-6.5,2),(0,1.5),(6.5,1.8),(.3,19)],[(-2.5,10),(3,9.6)],
      [(-6.5,2.3),(-.5,1.5),(6.5,2),(4.5,6),(1.8,11),(.4,15),(-.2,19)],
      [(-4.5,2),(1,1.6),(6.5,2),(.5,19)],
      [(-6.8,2.2),(-.5,1.5),(6.6,2),(2.5,8),(1,13),(.2,19)]],
 8: [[(.5,9.5),(-3,8.6),(-4.8,6.6),(-4.4,4),(-2.4,2),(.4,1.3),(3.2,2),(4.8,4),(4.6,6.6),(2.8,8.6),(-.2,9.8),(-3.4,11.4),(-5.2,13.8),(-5,16.4),(-3,18.4),(.2,19.2),(3.4,18.5),(5.2,16.2),(5.2,13.6),(3.2,11.2),(.5,9.5)],
      [(-.2,9.2),(-2.8,8.4),(-4.2,6.6),(-3.9,4.2),(-2.1,2.3),(.3,1.5),(2.8,2.2),(4.2,4.1),(4.1,6.5),(2.5,8.5),(-.4,9.9),(-3.6,11.5),(-5.4,14),(-5.2,16.6),(-3.1,18.6),(.2,19.3),(3.5,18.6),(5.3,16.3),(5.3,13.7),(3.3,11.3),(0,9.6)],
      [(-4.5,3),(-2,1.3),(1.5,1.2),(4,2.5),(4.8,4.8),(3.6,7.2),(.5,9.2),(-3,11),(-5,13.6),(-4.8,16.3),(-2.8,18.4),(.4,19.2),(3.6,18.4),(5.3,16),(5.2,13.4),(3,11),(0,9.4),(-3,8.4),(-4.6,6.4)],
      [(.4,9.4),(-2.4,8.5),(-3.8,6.6),(-3.5,4.1),(-1.9,2.2),(.4,1.4),(2.7,2.2),(4,4.1),(3.9,6.6),(2.3,8.6),(-.2,9.8),(-2.9,11.4),(-4.4,13.8),(-4.2,16.4),(-2.5,18.4),(.2,19.2),(2.9,18.5),(4.4,16.2),(4.4,13.6),(2.7,11.2),(.4,9.4)]],
 9: [[(4.6,7),(4.9,4.4),(4,2.2),(1.8,1),(-.8,1),(-3.2,2.2),(-4.5,4.4),(-4.6,7),(-3.6,9.2),(-1.2,10.4),(1.4,10.2),(3.6,9),(4.7,6.8),(4.9,11),(4.6,15),(3,17.8),(.5,19)],
      [(4.6,7),(4.9,4.4),(4,2.2),(1.8,1),(-.8,1),(-3.2,2.2),(-4.5,4.4),(-4.6,7),(-3.6,9.2),(-1.2,10.4),(1.4,10.2),(3.6,9),(4.6,7)],[(4.7,7),(4.6,19)],
      [(5.2,8.5),(5.5,5.5),(4.5,2.8),(2,1.2),(-1,1),(-3.6,2.4),(-5,5),(-5.2,8),(-4,10.6),(-1.4,12),(1.6,11.8),(4,10.4),(5.3,8),(5.5,12),(4.8,16),(2.5,18.6),(-.8,19.4)],
      [(4.8,7.5),(5.1,4.8),(4.2,2.4),(2,1.1),(-.8,1),(-3.3,2.3),(-4.6,4.6),(-4.7,7.4),(-3.7,9.6),(-1.2,10.8),(1.5,10.6),(3.7,9.3),(4.8,7),(5.1,11),(4.6,14),(3.2,16.4),(1.2,17.8)]],
}
# forms for class 4 live under keys 4/41/42 and for 7 the tick is a 2nd stroke
def forms(k):
    if k == 4: return [DIG[4], DIG[41], DIG[42], DIG[43], DIG[44]]
    if k == 7: return [[DIG[7][0]], [DIG[7][1], DIG[7][2]], [DIG[7][3]], [DIG[7][4]], [DIG[7][5]]]
    if k == 9: return [[DIG[9][0]], [DIG[9][1], DIG[9][2]], [DIG[9][3]], [DIG[9][4]]]
    if k == 5: return [[s] for s in DIG[5][:4]] + [DIG[5][4]]
    return [[s] for s in DIG[k]]

def cr(pts, spacing=.35):
    P = np.asarray(pts, float)
    if len(P) == 2:
        n = max(2, int(np.hypot(*(P[1]-P[0]))/spacing)+1)
        return P[0][None]*(1-np.linspace(0,1,n))[:,None] + P[1][None]*np.linspace(0,1,n)[:,None]
    Q = np.vstack([P[0], P, P[-1]]); out = []
    for i in range(1, len(Q)-2):
        p0,p1,p2,p3 = Q[i-1],Q[i],Q[i+1],Q[i+2]
        t = np.linspace(0,1,10,endpoint=False)[:,None]
        out.append(.5*(2*p1+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t**3))
    out.append(P[-1][None]); R = np.vstack(out)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(R,axis=0).T))]
    m = np.linspace(0, d[-1], max(2,int(d[-1]/spacing)+1))
    return np.stack([np.interp(m, d, R[:,0]), np.interp(m, d, R[:,1])], 1)

def render(strokes, slant=0., aspect=1., rot=0., sw=1., shift=(0.,0.), size=28, sigma=.95, spacing=.35, seed=0):
    parts = []
    for si, st in enumerate(strokes):
        R = cr(st)
        u = np.linspace(0, 1, len(R))
        tap = 1. + .18*np.sin(np.pi*u)
        rng = np.random.default_rng(9000 + seed*7 + si)
        ph, fr = rng.uniform(0, 6.283), rng.uniform(.8, 1.6)
        tang = np.gradient(R, axis=0)
        nl = np.hypot(tang[:,0], tang[:,1]) + 1e-9
        nrm = np.stack([-tang[:,1]/nl, tang[:,0]/nl], 1)
        wob = .35*np.sin(2*np.pi*fr*u + ph)
        R = R + nrm*wob[:,None]
        parts.append((R, tap))
    pts = np.vstack([q for q, _ in parts])
    amps = np.concatenate([a for _, a in parts])
    pts[:,0] *= aspect
    pts[:,0] += slant*(pts[:,1]-10.)
    if rot:
        a = math.radians(rot); c,s = math.cos(a), math.sin(a)
        pts = np.stack([c*pts[:,0]-s*(pts[:,1]-10.), s*pts[:,0]+c*(pts[:,1]-10.)+10.],1)
    ext = np.array([np.ptp(pts[:,0]), np.ptp(pts[:,1])]) + 2*sigma*sw
    sc = 20./max(ext)
    pts = pts*sc
    com = pts.mean(0)                      # ink COM ~ point COM (uniform stroke)
    Y, X = np.mgrid[0:size,0:size].astype(float)
    oy, ox = 14.-com[1]+shift[0], 14.-com[0]+shift[1]
    img = np.zeros((size,size))
    px, py = pts[:,0]+ox, pts[:,1]+oy
    r = int(3*sigma*sw*sc)+1
    for i in range(len(px)):
        y0,y1 = int(py[i])-r, int(py[i])+r+1; x0,x1 = int(px[i])-r, int(px[i])+r+1
        yy0,xx0 = max(0,y0),max(0,x0); yy1,xx1 = min(size,y1),min(size,x1)
        if yy1<=yy0 or xx1<=xx0: continue
        gy, gx = Y[yy0:yy1,xx0:xx1], X[yy0:yy1,xx0:xx1]
        se = sigma*sw*sc
        img[yy0:yy1,xx0:xx1] += amps[i]*(spacing/(se*2.5066))*np.exp(-((gy-py[i])**2+(gx-px[i])**2)/(2*se*se))
    return np.clip(img, 0, 1).astype(np.float32)

# ---------------------------------------------------------------- frontend
def filters():
    F = []
    y, x = np.mgrid[-2:3,-2:3].astype(float)
    for i in range(8):
        th = i*math.pi/8
        u = x*math.cos(th)+y*math.sin(th); v = -x*math.sin(th)+y*math.cos(th)
        G = np.exp(-(u*u+v*v)/(2*1.3**2))
        for ph in (0,1):
            f = (np.cos(2*math.pi*u/4.6) if ph==0 else np.sin(2*math.pi*u/4.6))*G
            f = f - f.mean()
            F.append(f/np.linalg.norm(f))
    for sg in (1.0, 2.0):
        f = np.exp(-(x*x+y*y)/(2*sg*sg)); F.append(f/np.linalg.norm(f))
    return np.asarray(F)[:,None]          # (18,1,5,5)

def conv18(imgs, W):
    n = len(imgs); K = W.shape[2]; pad = K//2
    xp = np.pad(imgs, ((0,0),(0,0),(pad,pad),(pad,pad)))
    acc = np.zeros((n, len(W), 28, 28), np.float32)
    for o in range(len(W)):
        w = W[o,0]
        for dy in range(K):
            for dx in range(K):
                if w[dy,dx]: acc[:,o] += w[dy,dx]*xp[:,0,dy:dy+28,dx:dx+28]
    return acc

def pool(x, size, stride):
    n,c,h,w = x.shape; sh,sw = size; th,tw = stride
    ho, wo = (h-sh)//th+1, (w-sw)//tw+1
    win = np.stack([x[:,:,dy:dy+th*ho:th,dx:dx+tw*wo:tw] for dy in range(sh) for dx in range(sw)],0)
    return win.max(0)

def frontend(imgs, W):
    r = np.maximum(conv18(imgs, W), 0)
    p2 = pool(r,(3,3),(2,2)); p4 = pool(p2,(3,3),(2,2)); p8 = pool(p4,(2,2),(2,2))
    return [p2.reshape(len(imgs),-1), p4.reshape(len(imgs),-1), p8.reshape(len(imgs),-1)]

# ---------------------------------------------------------------- build
SLANTS = (-.22, -.08, .06); WIDTHS = (.78, 1.0, 1.25); ASPECTS = (.9, 1.1)
SHIFTS = ((0,0),(.8,0),(-.8,0),(0,.8),(0,-.8))
SCALE = (.75, 1.0, .75)

def prototypes():
    per = {}; seed = 0
    for k in range(10):
        v = []
        for f in forms(k):
            for sl in SLANTS:
                for wv in WIDTHS:
                    for asp in ASPECTS:
                        for sh in SHIFTS:
                            v.append(render(f, slant=sl, sw=wv, aspect=asp, shift=sh, seed=seed))
                            seed += 1
        per[k] = v
    M = max(len(v) for v in per.values())
    imgs, lab = [], []
    for k in range(10):
        v = per[k]
        while len(v) < M: v = v + v[:M-len(v)]
        imgs += v; lab += [k]*M
    return np.asarray(imgs)[:,None], np.asarray(lab), M

def main(out):
    W = filters()
    P, lab, M = prototypes()
    feats = frontend(P, W)
    layers = [{"name":"e1","op":"conv","inputs":["image"],"out":18,"kernel":[5,5],"stride":1,"padding":2},
              {"name":"r1","op":"relu","inputs":["e1"]},
              {"name":"p2","op":"maxpool","inputs":["r1"],"size":[3,3],"stride":[2,2]},
              {"name":"p4","op":"maxpool","inputs":["p2"],"size":[3,3],"stride":[2,2]},
              {"name":"p8","op":"maxpool","inputs":["p4"],"size":[2,2],"stride":[2,2]}]
    weights = {"e1.W": W.astype(np.float32), "e1.b": np.zeros(18, np.float32)}
    names = []
    for li,(feat,w) in enumerate(zip(feats, SCALE)):
        mu = feat.mean(0)
        Q = feat - mu
        Qn = Q/np.maximum(np.linalg.norm(Q,axis=1,keepdims=True),1e-9)
        sc = w
        tag = "ABC"[li]
        rows = np.empty((10*M, Q.shape[1]), np.float32); bs = np.empty(10*M, np.float32)
        for k in range(10):
            idx = np.nonzero(lab==k)[0]
            rows[k*M:(k+1)*M] = Qn[idx]*sc
            bs[k*M:(k+1)*M] = -sc*(mu@Qn[idx].T)
        weights[f"d{tag}.W"] = rows; weights[f"d{tag}.b"] = bs
        layers += [{"name":f"d{tag}","op":"dense","inputs":["p2p4p8"[li*2:li*2+2]],"out":10*M},
                   {"name":f"r{tag}","op":"reshape","inputs":[f"d{tag}"],"shape":[10,M,1]},
                   {"name":f"m{tag}","op":"maxpool","inputs":[f"r{tag}"],"size":[M,1],"stride":[M,1]},
                   {"name":f"s{tag}","op":"reshape","inputs":[f"m{tag}"],"shape":[10]}]
        names.append(f"s{tag}")
    layers.append({"name":"scores","op":"add","inputs":names})
    if os.environ.get("DIAG"):
        S = np.zeros((len(P), 10*M))
        for li in range(3):
            tag = "ABC"[li]
            S += feats[li]@weights[f"d{tag}.W"].T + weights[f"d{tag}.b"]
        Sm = S.reshape(len(P), 10, M).max(2)
        pr = Sm.argmax(1)
        print("self-acc on synthetic prototypes:", (pr == lab).mean())
        cm = np.zeros((10,10),int)
        np.add.at(cm, (lab, pr), 1)
        print(cm)
        print("per-class self acc:", np.round([ (pr[lab==k]==k).mean() for k in range(10)],2))
        print("norm F per class:", np.round([np.linalg.norm(feats[0][lab==k],axis=1).mean() for k in range(10)],1))
    os.makedirs(out, exist_ok=True)
    json.dump({"layers":layers}, open(os.path.join(out,"network.json"),"w"))
    np.savez(os.path.join(out,"weights.npz"), **weights)
    print("weights:", sum(v.size for v in weights.values()), "M per class:", M)

if __name__ == "__main__":
    main(sys.argv[1])
