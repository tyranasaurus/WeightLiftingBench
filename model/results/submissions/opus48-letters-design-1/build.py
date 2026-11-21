"""Training-free ASL fingerspelling classifier: translation- and polarity-invariant matching of synthetic handshapes.

Every weight comes from synthetic handshape drawings built from geometric primitives (palm disks, finger capsules)
using general knowledge of the ASL alphabet -- no training data, image statistics, or fitting of any kind. The 24
drawings are turned into zero-mean, unit-norm 28x28 correlation templates. The network normalizes an input image
(contrast invariance), correlates it against every template and its negation (polarity invariance) at all positions,
takes the best-matching position (maxpool, translation invariance), and the class whose template matches best wins.
build.py reads no files.

Usage: python3 build.py OUT_DIR   ->  writes OUT_DIR/network.json and OUT_DIR/weights.npz
"""
import sys, os, json
import numpy as np

SIZE = 28

# ----------------------------------------------------------------------------- synthetic handshape drawings
YY, XX = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)  # row, col

def _seg_dist(y0, x0, y1, x1):
    vy, vx = y1 - y0, x1 - x0
    L2 = vy * vy + vx * vx
    if L2 < 1e-9:
        return np.sqrt((YY - y0) ** 2 + (XX - x0) ** 2)
    t = ((YY - y0) * vy + (XX - x0) * vx) / L2
    t = np.clip(t, 0.0, 1.0)
    py, px = y0 + t * vy, x0 + t * vx
    return np.sqrt((YY - py) ** 2 + (XX - px) ** 2)

def _soft(dist, half):
    return np.clip(half + 0.5 - dist, 0.0, 1.0)

class Canvas:
    def __init__(self):
        self.a = np.zeros((SIZE, SIZE), np.float64)
    def capsule(self, y0, x0, y1, x1, width):
        self.a = np.maximum(self.a, _soft(_seg_dist(y0, x0, y1, x1), width / 2.0))
        return self
    def disk(self, y, x, r):
        self.a = np.maximum(self.a, _soft(np.sqrt((YY - y) ** 2 + (XX - x) ** 2), r))
        return self
    def ellipse(self, y, x, ry, rx):
        d = np.sqrt(((YY - y) / ry) ** 2 + ((XX - x) / rx) ** 2)
        self.a = np.maximum(self.a, _soft(d * min(ry, rx), min(ry, rx)))
        return self
    def ring(self, y, x, rout, rin, amin=None, amax=None):
        d = np.sqrt((YY - y) ** 2 + (XX - x) ** 2)
        band = np.clip(d - rin + 0.5, 0, 1) * np.clip(rout - d + 0.5, 0, 1)
        if amin is not None:
            ang = (np.degrees(np.arctan2(YY - y, XX - x)) + 360) % 360
            lo, hi = amin % 360, amax % 360
            mask = (ang >= lo) & (ang <= hi) if lo <= hi else (ang >= lo) | (ang <= hi)
            band = band * mask
        self.a = np.maximum(self.a, band)
        return self
    def hole(self, y, x, r):
        self.a = self.a * (1 - _soft(np.sqrt((YY - y) ** 2 + (XX - x) ** 2), r))
        return self

def _wrist(cv):
    return cv.capsule(27, 13.5, 22.5, 13.5, 5.5)

def build_templates():
    """24 synthetic greyscale handshapes, drawn hand-bright-on-dark; the network also uses each negated, so the real
    hand being lighter or darker than its background is handled either way."""
    T = {}
    def up_finger(cv, col, tip, base=12.5, w=2.2):
        return cv.capsule(base, col, tip, col, w)

    # 0 A : fist, thumb up the left side
    cv = Canvas().disk(16.5, 13.5, 6.0).capsule(19, 9.5, 12.5, 9.0, 2.6); T[0] = _wrist(cv).a
    # 1 B : four fingers up together, thumb folded on palm
    cv = Canvas().ellipse(18.5, 13.5, 5.0, 6.0)
    for c in (9.5, 12.0, 14.5, 17.0): up_finger(cv, c, 3.5, base=13, w=2.1)
    T[1] = _wrist(cv).a
    # 2 C : crescent opening to the right
    cv = Canvas().ring(14.5, 17.0, 9.5, 4.5, amin=55, amax=305); T[2] = _wrist(cv).a
    # 3 D : single index finger up (central), round base
    cv = Canvas().disk(18.5, 13.5, 5.3); up_finger(cv, 12.5, 3.5, base=14, w=2.4); cv.ring(17.5, 15.0, 3.5, 1.6)
    T[3] = _wrist(cv).a
    # 4 E : curled fist (compact)
    cv = Canvas().ellipse(17.5, 13.5, 5.6, 5.3).capsule(14.0, 9.0, 14.0, 18.0, 1.6); T[4] = _wrist(cv).a
    # 5 F : index+thumb circle, three fingers up on the right
    cv = Canvas().disk(18.5, 14.0, 5.0)
    for c in (13.5, 16.0, 18.5): up_finger(cv, c, 4.0, base=13, w=2.0)
    cv.ring(12.0, 10.0, 3.2, 1.4); T[5] = _wrist(cv).a
    # 6 G : index pointing left (horizontal)
    cv = Canvas().disk(15.0, 19.5, 5.2).capsule(15.0, 17.0, 15.0, 4.5, 2.4); T[6] = _wrist(cv).a
    # 7 H : index+middle pointing left (horizontal)
    cv = Canvas().disk(15.5, 20.0, 5.0).capsule(13.5, 18.0, 13.5, 4.5, 2.1).capsule(16.5, 18.0, 16.5, 4.5, 2.1)
    T[7] = _wrist(cv).a
    # 8 I : pinky up on the left, fist
    cv = Canvas().disk(18.5, 14.5, 5.6).capsule(16.0, 11.0, 6.0, 11.0, 1.8); T[8] = _wrist(cv).a
    # 9 K : index+middle up in a V, thumb between at base
    cv = Canvas().disk(19.0, 13.5, 5.0).capsule(18, 12.0, 5.0, 9.5, 2.1).capsule(18, 14.5, 5.0, 16.5, 2.1)
    cv.capsule(18.5, 13.0, 13.0, 12.5, 2.0); T[9] = _wrist(cv).a
    # 10 L : index up, thumb out to the left (right angle)
    cv = Canvas().disk(18.5, 15.0, 5.0).capsule(16.5, 12.5, 3.5, 13.5, 2.3).capsule(19.5, 12.0, 19.5, 4.5, 2.6)
    T[10] = _wrist(cv).a
    # 11 M : fist, three fingertips draped down in front
    cv = Canvas().disk(15.5, 13.5, 6.2)
    for c in (10.5, 13.5, 16.5): cv.capsule(19.5, c, 24.0, c, 2.0)
    T[11] = _wrist(cv).a
    # 12 N : fist, two fingertips draped down
    cv = Canvas().disk(15.5, 13.5, 6.2)
    for c in (11.5, 15.0): cv.capsule(19.5, c, 24.0, c, 2.2)
    T[12] = _wrist(cv).a
    # 13 O : ring with a hole
    cv = Canvas().disk(16.5, 14.0, 7.0).hole(16.5, 14.0, 3.4); T[13] = _wrist(cv).a
    # 14 P : index down, middle out, palm high (points down)
    cv = Canvas().disk(10.5, 13.5, 5.0).capsule(12.0, 12.0, 26.0, 10.5, 2.3).capsule(12.5, 15.0, 21.0, 20.0, 2.1)
    cv.capsule(0, 13.5, 5.0, 13.5, 5.0); T[14] = cv.a
    # 15 Q : index+thumb pointing down from a high palm
    cv = Canvas().disk(9.5, 14.0, 5.0).capsule(11.5, 13.0, 26.0, 12.0, 2.3).capsule(11.5, 15.5, 24.0, 17.5, 2.1)
    cv.capsule(0, 14.0, 4.5, 14.0, 5.0); T[15] = cv.a
    # 16 R : index+middle up, crossed, narrow
    cv = Canvas().disk(19.0, 13.5, 5.0).capsule(18, 12.5, 5.0, 14.0, 1.9).capsule(18, 14.0, 5.0, 12.5, 1.9)
    T[16] = _wrist(cv).a
    # 17 S : compact round fist, thumb across front
    cv = Canvas().disk(16.5, 13.5, 6.3).capsule(17.5, 10.0, 17.5, 17.0, 1.8); T[17] = _wrist(cv).a
    # 18 T : fist, thumb tip between index/middle (bump at top centre)
    cv = Canvas().disk(17.0, 13.5, 6.0).disk(11.0, 13.0, 1.9); T[18] = _wrist(cv).a
    # 19 U : index+middle up together (narrow bar)
    cv = Canvas().disk(19.0, 13.5, 5.0).capsule(18, 12.0, 4.0, 12.0, 2.1).capsule(18, 15.0, 4.0, 15.0, 2.1)
    T[19] = _wrist(cv).a
    # 20 V : index+middle up spread
    cv = Canvas().disk(19.0, 13.5, 5.0).capsule(18, 12.5, 4.0, 9.0, 2.1).capsule(18, 14.5, 4.0, 18.0, 2.1)
    T[20] = _wrist(cv).a
    # 21 W : three fingers up spread
    cv = Canvas().disk(19.0, 13.5, 5.0).capsule(18, 12.0, 4.0, 8.5, 2.0).capsule(18, 13.5, 3.5, 13.5, 2.0).capsule(18, 15.0, 4.0, 18.5, 2.0)
    T[21] = _wrist(cv).a
    # 22 X : index hooked up (short, bent)
    cv = Canvas().disk(18.0, 13.5, 5.6).capsule(17.0, 12.5, 10.0, 12.5, 2.2).capsule(10.0, 12.5, 11.5, 15.5, 2.1)
    T[22] = _wrist(cv).a
    # 23 Y : thumb left, pinky up-right, fist
    cv = Canvas().disk(18.0, 13.5, 5.6).capsule(18.5, 11.5, 20.5, 4.5, 2.6).capsule(16.0, 16.0, 6.0, 19.0, 1.9)
    T[23] = _wrist(cv).a

    arr = np.stack([np.clip(T[k], 0, 1) for k in range(24)])[:, None, :, :]
    return arr

# ----------------------------------------------------------------------------- assemble weights
PAD = 6
KOUT = SIZE + 2 * PAD - SIZE + 1           # scan positions per axis

def build():
    base = build_templates()[:, 0]                     # (24, 28, 28) clean silhouettes
    T = base.reshape(24, -1)
    T = T - T.mean(axis=1, keepdims=True)              # zero-mean -> match shape/contrast, not absolute brightness
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)   # unit-norm -> comparable match across classes
    T = T.reshape(24, SIZE, SIZE)
    W1 = np.zeros((48, 1, SIZE, SIZE), np.float64)
    for k in range(24):
        W1[k, 0] = T[k]
        W1[k + 24, 0] = -T[k]                          # negated copy handles bright- vs dark-hand polarity
    b1 = np.zeros(48, np.float64)
    W2 = np.zeros((24, 48), np.float64)                # score[k] = max_pos relu(+corr) + max_pos relu(-corr)
    for k in range(24):
        W2[k, k] = 1.0
        W2[k, k + 24] = 1.0
    b2 = np.zeros(24, np.float64)

    weights = {
        "tmatch.W": W1.astype(np.float32), "tmatch.b": b1.astype(np.float32),
        "cls.W": W2.astype(np.float32),    "cls.b": b2.astype(np.float32),
    }
    net = {"layers": [
        {"name": "n1", "op": "norm", "inputs": ["image"]},
        {"name": "tmatch", "op": "conv", "inputs": ["n1"], "out": 48, "kernel": [SIZE, SIZE], "stride": 1, "padding": PAD},
        {"name": "r1", "op": "relu", "inputs": ["tmatch"]},
        {"name": "mp", "op": "maxpool", "inputs": ["r1"], "size": [KOUT, KOUT], "stride": [KOUT, KOUT]},
        {"name": "f", "op": "reshape", "inputs": ["mp"], "shape": [48]},
        {"name": "cls", "op": "dense", "inputs": ["f"], "out": 24},
    ]}
    return net, weights

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    net, weights = build()
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f, indent=2)
    np.savez(os.path.join(out_dir, "weights.npz"), **weights)

if __name__ == "__main__":
    main()
