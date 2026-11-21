"""Build the recognizer using only the pen descriptions below. No source data files."""
import math
from pathlib import Path
import sys
import numpy as np

# Shared, per-image feature extractor. This text is also included in the emitted program.
FEATURES = '''import numpy as np

side = 32
grid = np.arange(side, dtype=np.float32) - (side - 1) / 2
sourcegrid = np.arange(28, dtype=np.float32)

def quantilepos(hist, q):
    cumul = np.cumsum(hist, axis=1)
    target = cumul[:, -1:] * q
    k = np.minimum((cumul < target).sum(1), hist.shape[1] - 1)
    rows = np.arange(len(hist))
    delta = (target[:, 0] - cumul[rows, k]) / np.maximum(hist[rows, k], .00001)
    return (k + .5 + delta).astype(np.float32)

def sample(im, y, x):
    n = len(im)
    iy = np.floor(y).astype(np.int32)
    ix = np.floor(x).astype(np.int32)
    dy = (y - iy).astype(np.float32)
    dx = (x - ix).astype(np.float32)
    rows = np.arange(n)[:, None, None]
    ya = np.clip(iy, 0, 27)
    yb = np.clip(iy + 1, 0, 27)
    xa = np.clip(ix, 0, 27)
    xb = np.clip(ix + 1, 0, 27)
    a = im[rows, ya, xa] * (1 - dx) + im[rows, ya, xb] * dx
    b = im[rows, yb, xa] * (1 - dx) + im[rows, yb, xb] * dx
    return ((1 - dy) * a + dy * b) * ((y > -.5) & (y < 27.5) & (x > -.5) & (x < 27.5))

def normalize(im):
    # Estimate slant from row moments, then use robust bounds in the deskewed frame.
    h = im.sum(2)
    w = im.sum(1)
    mass = np.maximum(h.sum(1), 1e-5)
    cy = (h * sourcegrid).sum(1) / mass
    cx = (w * sourcegrid).sum(1) / mass
    yy = sourcegrid - cy[:, None]
    xx = sourcegrid - cx[:, None]
    slope = ((im * xx[:, None, :]).sum(2) * yy).sum(1) / np.maximum((h * yy**2).sum(1), 1e-5)
    slope = np.clip(.96 * slope, -1.65, 1.65)
    # Project directly onto the de-slanted horizontal axis in a padded histogram.
    pos = sourcegrid[None, None, :] + 24 - slope[:, None, None] * yy[:, :, None]
    pos = np.clip(pos, 0, 78)
    k = np.floor(pos).astype(np.int32)
    f = (pos - k).astype(np.float32)
    offset = (np.arange(len(im), dtype=np.int32) * 80)[:, None, None]
    k += offset
    bins = np.bincount(k.ravel(), (im * (1 - f)).ravel(), minlength=len(im)*80)
    bins += np.bincount((k+1).ravel(), (im*f).ravel(), minlength=len(im)*80)
    proj = bins.reshape(-1, 80).astype(np.float32)
    x0 = quantilepos(proj, .007) - 24
    x1 = quantilepos(proj, .993) - 24
    y0 = quantilepos(h, .007)
    y1 = quantilepos(h, .993)
    ht = np.maximum(y1 - y0, 2)
    wd = np.maximum(x1 - x0, 1.5)
    aspect = wd / ht
    # Moderate width normalization: preserve narrow bars without penalizing wide handwriting.
    sx = (np.maximum(wd, ht*.34)/22) * np.clip(.74 / np.maximum(aspect,.34), .30, 3)**.18
    sy = ht / 25
    y = ((y0+y1)[:, None, None]*.5 + sy[:, None, None]*grid[None, :, None])
    x = (x0+x1)[:, None, None]*.5 + sx[:, None, None]*grid[None, None, :] + slope[:, None, None]*(y-cy[:, None, None])
    image = sample(im, y, x)
    return image / np.maximum(image.max((1,2), keepdims=True), .001), aspect

def blur(a):
    # Separable binomial filter, zero extension.
    b = .5*a
    b[:,1:] += .25*a[:,:-1]
    b[:,:-1] += .25*a[:,1:]
    c = .5*b
    c[:,:,1:] += .25*b[:,:,:-1]
    c[:,:,:-1] += .25*b[:,:,1:]
    return c

def coarse(a):
    return a.reshape(len(a),8,4,8,4).mean((2,4))

def cavity(im, threshold):
    # Fill the exterior by four-connected propagation. What is left is enclosed paper.
    paper = im < threshold
    reach = paper.copy()
    reach[:,1:-1,1:-1] = False
    for step in range(100):
        old = reach.copy()
        reach[:,1:] |= old[:,:-1]
        reach[:,:-1] |= old[:,1:]
        reach[:,:,1:] |= old[:,:,:-1]
        reach[:,:,:-1] |= old[:,:,1:]
        reach &= paper
        if np.array_equal(reach, old):
            break
    return (paper & ~reach).astype(np.float32)

# Two simultaneous-deletion passes, preserving connectivity and endpoints (Zhang-Suen).
# Compile both eight-neighbor truth tables directly from the thinning conditions.
pattern = np.arange(256)
bits = ((pattern[:,None] >> np.arange(8)) & 1).astype(bool)
transitions = ((~bits) & np.roll(bits,-1,axis=1)).sum(1)
eligible = (bits.sum(1)>=2) & (bits.sum(1)<=6) & (transitions==1)
north,east,south,west = bits[:,0],bits[:,2],bits[:,4],bits[:,6]
remove = np.stack((eligible & ~(north & east & south) & ~(east & south & west),
                   eligible & ~(north & east & west) & ~(north & south & west)))

def neighbors(s):
    p = np.pad(s.astype(np.uint8),((0,0),(1,1),(1,1)))
    return (p[:,:-2,1:-1] + 2*p[:,:-2,2:] + 4*p[:,1:-1,2:] + 8*p[:,2:,2:]
           + 16*p[:,2:,1:-1] + 32*p[:,2:,:-2] + 64*p[:,1:-1,:-2] + 128*p[:,:-2,:-2])

def strokes(a):
    s = a > .37
    for it in range(24):
        old = s.copy()
        for passnum in range(2):
            s &= ~remove[passnum,neighbors(s)]
        if np.array_equal(s,old):
            break
    hood = neighbors(s)
    ends = s & (bits.sum(1)[hood] == 1)
    joints = s & (transitions[hood] >= 3)
    # Distance coverage of the thin line and its salient points. Coarse coverage is
    # insensitive to moving a pen lift or the joining point by a few pixels.
    def spread(mask,steps):
        wave = mask.astype(np.float32)
        for i in range(steps):
            pre = wave.copy()
            wave[:,1:] = np.maximum(wave[:,1:],pre[:,:-1]-.92/steps)
            wave[:,:-1] = np.maximum(wave[:,:-1],pre[:,1:]-.92/steps)
            wave[:,:,1:] = np.maximum(wave[:,:,1:],pre[:,:,:-1]-.92/steps)
            wave[:,:,:-1] = np.maximum(wave[:,:,:-1],pre[:,:,1:]-.92/steps)
        return coarse(blur(wave))
    line = spread(s,4)
    line /= np.maximum(np.sqrt((line**2).sum((1,2),keepdims=True)),.001)
    end = spread(ends,7)
    joint = spread(joints,7)
    end /= np.maximum(np.sqrt((end**2).sum((1,2),keepdims=True)),1.7)
    joint /= np.maximum(np.sqrt((joint**2).sum((1,2),keepdims=True)),1.7)
    return np.stack((line,end*.4,joint*.4),axis=3)

def fields(im):
    a, aspect = normalize(im)
    smooth = blur(a)
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:,:,1:-1] = smooth[:,:,2:] - smooth[:,:,:-2]
    gy[:,1:-1] = smooth[:,2:] - smooth[:,:-2]
    angles = np.arange(8) * (np.pi/4)
    mag = np.sqrt(gx*gx+gy*gy+.000001)
    maps = []
    for angle in angles:
        # Soft positive projections raised to the third power give orientation selectivity.
        p = np.maximum(gx*np.float32(np.cos(angle))+gy*np.float32(np.sin(angle)), 0) / mag
        maps.append(coarse(blur(blur(p*p*p*mag))))
    maps = np.stack(maps, axis=3)
    # Per-image local contrast compression prevents a heavy pen from dominating the match.
    maps /= np.sqrt(.045 + maps.sum(axis=3,keepdims=True))
    maps /= np.maximum(np.sqrt((maps*maps).sum((1,2,3),keepdims=True)), .0001)
    ink = coarse(blur(smooth))
    ink /= np.maximum(np.sqrt((ink*ink).sum((1,2),keepdims=True)), .0001)
    shape = np.concatenate((maps*np.float32(np.sqrt(.79)),ink[:,:,:,None]*np.float32(np.sqrt(.21))),axis=3)
    holes = .65*cavity(a,.23) + .35*cavity(a,.53)
    holerows = holes.sum(2)
    bands = []
    for center in (6, 12, 19, 25):
        window = np.exp(-.5*((grid+15.5-center)/4.5)**2)
        bands.append(1-np.exp(-(holerows*window).sum(1)/20))
    geometry = np.stack([np.log(np.maximum(aspect,.05))]+bands,axis=1)
    return shape.astype(np.float32), geometry.astype(np.float32), strokes(smooth)

def packedfield(f):
    return np.rint(f*1200).clip(0,255).astype(np.uint8)
'''

MATCH = '''
shapes = CONSTANTS['forms']
geometry = CONSTANTS['geom']
graphs = CONSTANTS['strokes']
# Compact first pass: signed-edge histograms on a 4x4 grid, plus 8x8 ink coverage.
def indexvec(f):
    pooled = f.reshape(len(f),4,2,4,2,9).sum((2,4))*.5
    v = np.concatenate((pooled.reshape(len(f),-1), f[:,:,:,8].reshape(len(f),-1)*.8),axis=1)
    return v / np.maximum(np.sqrt((v*v).sum(1,keepdims=True)),1e-7)

index = np.empty((208,len(shapes)),dtype=np.float32)
for start in range(0,len(shapes),512):
    group = shapes[start:start+512].astype(np.float32)/1200
    index[:,start:start+len(group)] = indexvec(group).T

# Fixed matrix sizes and internal padding also make the dot-product path independent of the caller's batch shape.
inbatch = 24
keep = 48
perclass = len(shapes)//10

def selectforms(f, g):
    sim = indexvec(f) @ index
    for j in range(5):
        d = np.abs(g[:,j:j+1]-geometry[None,:,j])
        if j == 0:
            sim -= .028*np.minimum(d,2)**1.5
        else:
            sim -= .021*d*d
    sim = sim.reshape(inbatch,10,perclass)
    return np.argpartition(sim,-keep,axis=2)[:,:,-keep:] + (np.arange(10)*perclass)[None,:,None]

def slide(a,dy,dx):
    b = np.zeros_like(a)
    ya,yb = max(0,-dy),min(8,8-dy)
    xa,xb = max(0,-dx),min(8,8-dx)
    b[:,ya:yb,xa:xb] = a[:,ya+dy:yb+dy,xa+dx:xb+dx]
    return b

def blocks(a):
    # Twenty-five patch sums, including small windows centered on the border.
    p = np.pad(a,((0,0),(1,1),(1,1)))
    b = p.reshape(len(p),5,2,5,2).sum((2,4))
    return b

def compare(a,g,graph,choices):
    cand = choices.ravel()
    b = shapes[cand].astype(np.float32)/1200
    # Half-cell motions of each local patch, with capped deformation freedom.
    # Penalize a mismatch twice, in both coordinate frames (forms and observation).
    ds = ((0,0),(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,1),(1,-1),(-1,-1))
    a = a[None,:,:,:]
    def sqerror(q,r):
        diff = q-r
        return (diff*diff).sum(axis=3)
    direct = sqerror(a,b)
    Best = blocks(direct)
    Rev = Best.copy()
    for dy,dx in ds[1:]:
        sa = slide(a,dy,dx)
        # At the halfway point between cells, compare a linearly blended map. In the
        # reverse frame use the same unblended cell comparisons and the exact variance identity.
        error = sqerror(sa,b)
        forward = .5*(direct+error) - .25*sqerror(sa,a)
        shifted = slide(error[:,:,:,None],-dy,-dx)[:,:,:,0]
        sb = slide(b,-dy,-dx)
        reverse = .5*(direct+shifted) - .25*sqerror(b,sb)
        # Where the shared comparison was shifted out, compare to the zero padding.
        if dy != 0:
            edge = 0 if dy == 1 else 7
            reverse[:,edge,:] += .5*(a[:,edge,:]**2).sum(2)
        if dx != 0:
            col = 0 if dx == 1 else 7
            correction = .5*(a[:,:,col]**2).sum(2)
            if dy != 0:
                correction[:,edge] = 0
            reverse[:,:,col] += correction
        penalty = .00032*(abs(dx)+abs(dy))
        Best = np.minimum(Best, blocks(forward)+penalty)
        Rev = np.minimum(Rev, blocks(reverse)+penalty)
    geo = np.abs(geometry[cand] - g)
    spatial = .43*(Best+Rev).sum((1,2))+.14*direct.sum((1,2))
    lineerror = np.square(graphs[cand].astype(np.float32)/750 - graph).sum((1,2,3))
    score = spatial + .14*lineerror + .043*np.minimum(geo[:,0], 2)**1.5 + .038*(geo[:,1:]**2).sum(1)
    # Use several individual matches, without estimating any parameters from the template bank.
    best = np.partition(score.reshape(10,keep),2,axis=1)[:,:3]
    return np.argmin(.78*best.min(1)+.22*best.mean(1))

def predict(x):
    out = np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),inbatch):
        part = x[start:start+inbatch,0]
        buf = np.zeros((inbatch,28,28),dtype=np.float32)
        buf[:len(part)] = part
        f,g,graph = fields(buf)
        graph = np.rint(graph*750).clip(0,255)/750
        f = packedfield(f).astype(np.float32)/1200
        choice = selectforms(f,g)
        for row in range(len(part)):
            out[start+row] = compare(f[row],g[row],graph[row],choice[row])
    return out
'''

# Pen contours in an arbitrary writing square: x increases rightward, y downward.
# M starts a new stroke, L is straight, Q and C are quadratic and cubic Beziers.
# Coordinates, including all alternative allographs, are authored here, not sampled.
paths = [
# 0: elliptical, egg-shaped, quadrangular, open seam, overshoot, slashed.
[
'M 61 5 C 34 -5 8 23 8 62 C 7 93 32 110 60 91 C 91 69 96 5 61 5',
'M 46 2 C 18 3 10 35 13 68 C 17 99 49 110 72 86 C 92 66 80 1 46 2',
'M 55 4 C 35 -1 19 27 15 52 Q 4 103 44 96 Q 79 94 84 46 Q 87 5 55 4',
'M 70 13 Q 45 -4 26 19 Q 10 39 9 76 Q 8 104 37 96 Q 65 89 79 57 Q 97 13 63 5',
'M 49 5 Q 25 3 18 41 L 12 71 Q 8 96 38 97 L 56 92 Q 80 83 83 41 Q 85 3 49 5',
'M 64 4 C 33 -3 10 34 12 72 C 15 106 57 103 76 73 C 91 49 82 11 69 10',
'M 53 15 Q 46 0 34 10 Q 12 30 13 70 C 16 104 55 105 73 79 Q 93 46 75 13 Q 65 -3 48 6 Q 41 11 40 20',
'M 70 3 C 38 1 15 36 15 69 C 16 105 49 103 67 81 C 92 50 85 10 63 4 M 26 86 L 68 16',
],
# 1: straight stroke, fine side flick, ascending lead-in, occasional baseline.
[
'M 52 5 L 48 96',
'M 53 5 Q 45 53 50 96',
'M 55 4 Q 59 40 49 74 Q 45 89 47 97',
'M 43 19 L 54 5 Q 54 37 45 97',
'M 32 30 Q 40 26 53 5 L 46 96',
'M 23 36 Q 38 26 51 5 Q 56 51 49 96',
'M 49 5 Q 53 10 50 37 L 45 86 Q 43 97 53 95',
'M 27 30 L 51 5 L 49 95 M 24 96 Q 47 91 77 94',
],
# 2: print two, cursive loop at the left tip of the foot, swan-neck, bottom hook.
[
'M 12 25 C 24 -2 65 -1 69 19 C 77 44 33 67 12 94 Q 44 84 87 94',
'M 14 32 Q 20 4 47 5 Q 83 4 69 38 Q 59 57 13 91 Q 44 95 85 89',
'M 17 20 Q 43 -4 64 13 Q 83 33 47 67 L 18 94 L 86 94',
'M 23 21 Q 13 13 26 8 C 57 -4 75 15 65 40 Q 54 62 20 86 Q 0 103 14 82 Q 23 74 43 84 Q 69 98 90 84',
'M 18 30 C 8 0 63 -8 67 20 Q 74 49 28 82 C -4 112 1 68 29 76 C 48 81 64 103 90 85',
'M 16 23 Q 32 0 48 4 C 92 12 62 56 25 82 Q 5 99 15 99 Q 24 98 35 92 Q 55 80 85 89',
'M 14 14 Q 53 -1 65 18 Q 81 39 45 68 Q 16 91 16 95 Q 49 93 87 98',
'M 18 25 Q 35 6 55 5 Q 83 2 73 24 L 19 88 Q 43 81 67 91 Q 82 96 90 91',
'M 13 33 C 13 14 22 3 41 4 Q 71 5 61 38 Q 53 65 24 87 Q 8 97 8 85 Q 9 68 25 73 Q 47 81 53 95 Q 67 104 87 88',
'M 16 20 C 42 -8 75 1 71 25 C 67 42 19 72 17 88 Q 16 99 43 94 L 85 83',
],
# 3: round, angular flat cap, crossed centre/central loop, narrow or open lower lobe.
[
'M 17 17 C 39 -3 78 0 75 20 Q 75 41 42 48 C 92 38 90 78 61 94 Q 34 108 13 86',
'M 18 13 Q 58 -4 74 12 Q 88 29 40 48 Q 73 42 79 62 C 90 94 40 111 15 87',
'M 14 8 Q 42 10 76 6 L 39 47 C 70 32 89 49 79 73 Q 61 110 18 91',
'M 14 14 L 75 6 Q 68 22 39 48 Q 80 39 79 66 Q 81 94 48 98 Q 32 99 19 90',
'M 20 22 C 30 4 62 -3 72 12 Q 86 33 41 50 Q 27 51 38 43 C 64 26 93 60 74 84 Q 53 107 18 89',
'M 18 13 C 51 -7 90 13 60 37 Q 55 42 43 48 Q 92 51 74 78 Q 47 112 13 88',
'M 25 13 Q 55 -2 74 14 Q 91 32 50 49 Q 82 58 75 78 C 66 103 22 108 17 81',
'M 17 12 Q 68 -6 72 16 Q 74 36 34 51 Q 23 57 30 46 Q 45 33 64 43 C 93 59 74 99 40 97 Q 25 95 17 87',
'M 20 18 Q 47 7 67 5 Q 83 8 62 29 L 40 48 Q 62 47 73 59 Q 92 79 62 91 Q 42 100 16 94',
],
# 4: open right angle, cupped and crossed versions, triangular closed four.
[
'M 31 5 L 14 55 Q 41 59 86 54 M 65 5 L 59 97',
'M 29 6 Q 15 37 16 54 Q 27 67 72 49 M 71 5 Q 62 53 62 97',
'M 27 6 Q 4 66 30 62 Q 50 60 72 36 M 73 11 L 60 97',
'M 27 4 L 17 48 Q 18 59 35 59 L 78 53 M 69 19 L 61 98',
'M 26 5 Q 10 41 14 51 Q 14 70 37 61 Q 63 47 68 12 Q 67 43 57 97',
'M 24 5 Q 16 41 18 53 Q 34 63 72 52 L 68 97 M 75 8 L 72 52',
'M 68 5 L 12 63 L 87 63 M 67 5 L 60 97',
'M 65 5 Q 49 17 14 59 Q 36 63 82 55 M 66 8 Q 60 48 64 97',
'M 57 8 L 11 59 Q 44 56 81 59 M 69 15 L 59 97',
'M 38 4 Q 29 27 15 51 Q 7 61 28 60 L 85 51 M 70 26 L 64 97',
],
# 5: the bar may be separate or attached, rising or falling, and of different lengths.
[
'M 77 6 Q 50 11 27 7 L 22 44 C 65 24 89 52 73 80 C 59 105 28 103 12 85',
'M 79 5 L 24 6 L 20 47 Q 48 31 67 46 C 102 70 58 116 19 91',
'M 81 7 Q 63 14 27 9 Q 27 26 20 43 Q 58 25 73 52 Q 92 84 53 96 Q 32 104 14 86',
'M 27 8 Q 26 26 16 43 Q 44 36 61 46 C 98 67 62 111 26 93 M 23 9 Q 48 14 80 3',
'M 34 7 Q 23 25 20 41 C 45 33 82 48 75 71 C 70 103 24 102 15 89 M 36 12 L 80 7',
'M 82 6 Q 49 14 29 6 Q 16 26 22 39 Q 25 45 46 47 C 88 54 80 90 46 96 Q 26 100 14 89',
'M 78 5 Q 62 16 29 11 Q 10 24 27 38 Q 36 44 51 47 C 91 60 75 93 44 96 C 25 103 7 92 16 81',
'M 73 8 Q 43 13 26 8 L 22 45 Q 68 35 77 56 Q 88 81 51 91 Q 33 95 15 95',
'M 79 8 L 20 10 Q 20 34 28 47 C 54 36 78 43 78 64 C 77 92 36 106 19 88',
'M 32 7 Q 12 29 20 41 Q 22 46 48 42 C 85 37 83 77 53 92 Q 25 107 15 87 M 37 8 Q 54 12 80 4',
],
# 6: tapered flag into a small lower bowl, round-backed six, upper hook, open bowl.
[
'M 69 5 Q 37 11 19 54 C 0 99 50 110 72 82 C 96 51 57 29 24 64',
'M 65 3 C 40 10 10 55 17 81 C 26 110 73 99 77 71 C 79 40 37 37 17 71',
'M 61 5 Q 23 31 20 69 Q 18 100 49 97 C 80 96 91 59 66 53 Q 45 46 23 71',
'M 61 5 Q 36 18 16 64 C 4 97 44 110 69 88 C 94 65 70 42 45 54 Q 29 62 21 80',
'M 76 15 Q 69 -2 53 8 C 28 20 12 63 21 84 C 37 110 80 98 79 65 Q 73 35 39 54 Q 26 63 24 77',
'M 68 5 L 28 55 Q 1 93 36 97 Q 66 105 78 78 Q 89 50 63 48 Q 44 47 25 68',
'M 68 4 C 43 25 23 47 22 68 Q 15 96 43 98 Q 79 101 77 68 Q 72 45 52 53 Q 41 56 35 65',
'M 57 4 C 33 22 10 73 28 91 C 52 113 92 84 72 61 C 57 44 31 60 25 81 Q 25 92 35 95',
'M 55 5 C 44 12 28 41 20 60 Q 8 100 49 96 C 87 94 89 51 58 49 Q 28 50 18 72 L 11 83',
],
# 7: level, dished or wavy cap; pointed knee, bowed downstroke, optional crossbar.
[
'M 13 8 Q 46 9 79 6 Q 57 45 43 97',
'M 13 10 Q 41 3 79 9 L 39 97',
'M 12 21 L 15 9 Q 45 7 78 13 Q 54 56 45 97',
'M 12 7 Q 38 22 77 7 Q 57 43 51 75 L 44 97',
'M 12 18 Q 22 1 43 9 Q 65 20 79 8 Q 59 44 43 96',
'M 13 12 Q 50 4 79 10 Q 55 42 45 97 M 28 52 L 70 49',
'M 14 22 L 17 7 L 81 10 Q 59 43 48 96 M 39 55 Q 53 51 69 55',
'M 14 8 C 24 3 38 4 54 11 Q 69 18 77 7 Q 72 37 57 69 Q 48 91 48 97',
'M 13 16 Q 34 2 53 8 L 79 13 L 48 92 Q 46 99 40 94',
],
# 8: continuous figure-eight with a diagonal crossing, double bowls, open seams.
[
'M 55 6 C 26 -5 9 19 26 35 C 44 52 76 58 76 79 C 76 106 22 103 19 81 C 13 58 58 45 73 25 C 87 7 67 1 55 6',
'M 51 49 C 24 41 9 15 35 6 C 66 -6 93 12 69 32 C 52 48 23 59 19 78 C 11 104 69 106 75 82 Q 81 65 51 49',
'M 48 47 C 15 43 16 8 44 5 C 83 1 83 32 48 47 C 88 52 89 89 63 96 C 18 108 -1 66 48 47',
'M 55 7 C 20 -3 14 26 35 41 Q 45 47 58 55 C 99 81 63 105 37 96 C 3 85 15 66 43 47 Q 78 26 70 14 Q 65 5 55 7',
'M 65 8 C 40 -7 8 12 25 33 Q 35 45 60 62 C 97 90 52 109 27 91 Q 6 75 34 56 C 56 40 81 24 69 12',
'M 53 3 C 31 2 15 19 24 33 Q 33 47 57 55 C 86 70 78 94 53 98 C 21 104 7 80 22 64 Q 38 49 61 42 Q 85 27 72 11 Q 66 5 57 7',
'M 47 45 C 10 37 20 2 49 4 C 84 3 81 37 47 45 M 47 45 C 78 43 93 76 71 90 C 37 113 4 87 22 61 Q 29 47 47 45',
'M 64 5 C 44 -3 17 7 23 27 Q 31 45 56 59 C 94 83 67 108 35 96 C 2 84 22 64 44 46 Q 67 28 72 16',
'M 41 11 C 26 11 19 27 32 39 Q 48 53 63 62 Q 94 90 59 98 C 27 106 5 85 26 63 Q 48 42 64 32 C 88 9 60 -4 40 11',
],
# 9: little upper bowl and long leg; continuous single stroke, angular or loose bowl,
# loopless seam, hook back left, some print nines with a curved descending flag.
[
'M 71 22 C 61 -5 30 1 18 22 C 1 53 41 66 61 38 Q 71 24 73 9 Q 72 47 57 97',
'M 69 14 Q 50 -3 28 12 C 4 29 13 56 37 53 Q 58 51 70 23 L 65 96',
'M 67 8 C 46 -3 17 17 18 38 Q 20 58 44 50 Q 65 39 74 7 Q 70 47 60 96',
'M 68 13 C 38 -6 8 20 20 44 C 34 65 65 41 69 18 Q 81 57 48 96',
'M 69 9 Q 43 1 25 20 Q 7 41 22 48 Q 39 57 66 32 L 73 14 Q 66 52 62 97',
'M 61 6 C 37 0 13 18 18 41 Q 27 61 57 43 Q 72 32 69 11 Q 83 45 68 79 Q 58 99 36 97',
'M 57 8 Q 28 3 15 33 Q 5 57 33 52 Q 60 44 68 10 M 71 10 Q 63 55 63 96',
'M 55 8 C 31 -1 7 29 19 46 Q 33 61 54 45 Q 69 33 72 17 Q 66 55 62 96',
'M 66 7 L 47 5 Q 22 10 16 33 Q 11 48 27 49 Q 44 50 73 25 M 73 10 Q 70 51 55 97',
'M 71 19 C 61 -4 35 1 22 17 C 6 38 19 58 39 51 Q 61 44 69 22 Q 78 62 56 88 Q 48 99 36 95',
]
]


def parse(s):
    tok = iter(s.split())
    out = []
    def pt(): return np.array([float(next(tok)),float(next(tok))])
    for cmd in tok:
        if cmd == 'M':
            p = pt()
            out.append([p])
            continue
        q = pt()
        if cmd == 'L':
            end = q
            t = np.linspace(0,1,25)[1:,None]
            line = (1-t)*p+t*q
        elif cmd == 'Q':
            end = pt()
            t = np.linspace(0,1,32)[1:,None]
            line = (1-t)**2*p+2*t*(1-t)*q+t*t*end
        elif cmd == 'C':
            r = pt(); end = pt()
            t = np.linspace(0,1,40)[1:,None]
            line = (1-t)**3*p+3*t*(1-t)**2*q+3*t*t*(1-t)*r+t**3*end
        else:
            raise ValueError(cmd)
        out[-1].extend(line); p = end
    return np.concatenate([np.array(st) for st in out])


def render(points, rng, rate):
    # Sample an elastic writing field (smooth offsets with a bend around the waist).
    # All distributions are set a priori in units of the writing square.
    a = points / 100
    x,y = a.T
    u = x-.5; v = y-.5
    field = np.stack([4*u*v, 1-4*v*v, 1-4*u*u, np.sin(2*np.pi*y), np.sin(2*np.pi*x), np.cos(2*np.pi*y)*u*2, np.cos(2*np.pi*x)*v*2],1)
    size = np.array([.115,.12,.075,.06,.045,.065,.055])[:,None]
    coeff = rng.uniform(-1,1,(7,2)) * size * rate
    q = a+field@coeff
    # Occasional more triangular nib, changing pressure along the line.
    q[:,0] *= rng.uniform(.74,1.17)
    q[:,0] += rng.uniform(-.50,.31)*(q[:,1]-.5)
    theta = rng.uniform(-.23,.23)
    co,si = np.cos(theta),np.sin(theta)
    q = q@np.array([[co,si],[-si,co]])
    base = rng.uniform(.047,.115)
    pressure = base*(1 + rng.uniform(-.24,.24)*np.sin(y*np.pi+2*np.pi*rng.random())+rng.uniform(-.17,.17)*np.cos(2*np.pi*x+6*rng.random()))
    shape = np.exp(rng.uniform(-.30,.22))
    rx = pressure*shape; ry = pressure/shape
    left = (q[:,0]-rx).min(); right = (q[:,0]+rx).max()
    top = (q[:,1]-ry).min(); bottom = (q[:,1]+ry).max()
    scale = 20 / max(right-left, bottom-top)
    q = (q-[(right+left)*.5,(top+bottom)*.5])*scale + 13.5
    q += rng.uniform(-.48,.48,(1,2))
    rx *= scale; ry *= scale
    # Stamp overlapping elliptical nibs evaluated as anti-aliased signed distances.
    span = int(math.ceil(max(rx.max(),ry.max())+1.5))
    delta = np.arange(-span,span+1)
    ix = q[:,0].astype(int)[:,None]+delta
    iy = q[:,1].astype(int)[:,None]+delta
    dx = ((ix-q[:,0:1])/rx[:,None])**2
    dy = ((iy-q[:,1:2])/ry[:,None])**2
    dist = np.sqrt(dx[:,None,:]+dy[:,:,None])
    z = (.5 + (1-dist)*np.sqrt(rx*ry)[:,None,None]).clip(0,1)
    result = np.zeros(784,dtype=np.float32)
    idx = np.clip(iy,0,27)[:,:,None]*28 + np.clip(ix,0,27)[:,None,:]
    np.maximum.at(result,idx.reshape(-1),z.reshape(-1).astype(np.float32))
    return result.reshape(28,28)


def build(folder):
    # Every render contributes exactly one row. No comparisons between renders,
    # data selection, feature calibration, or bank-level normalization.
    rng = np.random.default_rng(310519)
    scope = {}
    exec(FEATURES,scope)
    count = 3456
    forms = np.empty((10,count,8,8,9),dtype=np.uint8)
    geom = np.empty((10,count,5),dtype=np.float32)
    graphs = np.empty((10,count,8,8,3),dtype=np.uint8)
    repeats = [
        [6,4,4,3,3,3,2,1], [10,8,6,5,4,2,3,1], [5,5,4,3,4,3,2,2,2,4],
        [5,4,4,3,2,4,2,2,3], [4,4,4,4,4,3,4,3,2,4], [4,3,4,4,3,3,2,3,3,3],
        [5,5,4,4,3,3,3,3,2], [6,6,4,3,4,3,2,3,3], [5,5,3,4,3,4,2,3,3],
        [5,5,5,4,4,3,3,3,3,2],
    ]
    for lab,descs in enumerate(paths):
        ps = [parse(s) for s in descs]
        choice = np.repeat(np.arange(len(ps)), repeats[lab])
        for i in range(0,count,48):
            batch = np.stack([render(ps[choice[j%len(choice)]],rng, .4+1.15*rng.random()) for j in range(i,i+48)])
            f,g,graph = scope['fields'](batch)
            forms[lab,i:i+48] = scope['packedfield'](f)
            geom[lab,i:i+48] = g
            graphs[lab,i:i+48] = np.rint(graph*750).clip(0,255).astype(np.uint8)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'solution.py').write_text(FEATURES + MATCH)
    np.savez(folder/'constants.npz',forms=forms.reshape(-1,8,8,9),geom=geom.reshape(-1,5),strokes=graphs.reshape(-1,8,8,3))

if __name__ == '__main__':
    build(Path(sys.argv[1]))
