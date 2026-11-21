"""Construct an OCR program from a small pen-path language. No input files."""
from pathlib import Path
import sys
import math
import numpy as np

# Shared by the renderer's single-glyph descriptor and by the classifier.
features = '''import numpy as np

def masslimits(p, start):
    # Quantile of a piecewise uniform density (one bin per pixel).
    a = np.cumsum(p, axis=1)
    m = a[:, -1:]
    out = []
    for q in (.006, .994):
        t = m*q
        i = np.minimum(np.sum(a < t, axis=1), p.shape[1]-1)
        r = np.arange(len(p))
        f = 1 - (a[r,i] - t[:,0]) / np.maximum(p[r,i], 1.e-8)
        out.append(start + i - .5 + np.clip(f, 0, 1))
    return out

def canonical(img):
    img = np.asarray(img, dtype=np.float32)
    b = len(img)
    g = np.arange(28, dtype=np.float32)
    v = g[None,:,None]
    u = g[None,None,:]
    p = img.sum(axis=2)
    m = np.maximum(p.sum(axis=1), 1.e-6)
    ym = (p * g).sum(axis=1)/m
    xm = (img.sum(axis=1) * g).sum(axis=1)/m
    yc = v - ym[:,None,None]
    xc = u - xm[:,None,None]
    tilt = np.clip((img*yc*xc).sum(axis=(1,2)) / np.maximum((img*yc*yc).sum(axis=(1,2)), 1.e-6), -1.15, 1.15)
    shear = .96*tilt
    ux = u - shear[:,None,None]*yc
    # Project ink onto the de-slanted x axis, splatting between adjacent bins.
    h = 33 + ux
    ix = h.astype(np.int32)
    f = h-ix
    pos = ix + 94*np.arange(b)[:,None,None]
    hist = np.bincount(pos.ravel(), (img*(1-f)).ravel(), minlength=b*94)
    hist += np.bincount((pos+1).ravel(), (img*f).ravel(), minlength=b*94)
    l,r = masslimits(hist.reshape(b,94), -33)
    t,d = masslimits(p, 0)
    w = np.maximum(r-l, 1.)
    ht = np.maximum(d-t, 1.)
    aspect = np.clip(w/ht, .06, 3.)
    # Partial rather than full aspect normalization keeps a bare 1 narrow.
    tgtw = np.minimum(21.5, np.minimum(20.5*aspect*1.55, 16*(aspect/.72)**.46))
    sy = (v-13.5) * (ht/20.5)[:,None,None] + ((t+d)/2)[:,None,None]
    sx = (u-13.5) * (w/tgtw)[:,None,None] + ((r+l)/2)[:,None,None] + (sy-ym[:,None,None])*shear[:,None,None]
    ix = np.floor(sx).astype(np.int32)
    iy = np.floor(sy).astype(np.int32)
    fx = (sx-ix).astype(np.float32)
    fy = (sy-iy).astype(np.float32)
    j = np.arange(b)[:,None,None]
    def sample(yy,xx):
        return img[j,np.clip(yy,0,27),np.clip(xx,0,27)]*((xx>=0)&(xx<28)&(yy>=0)&(yy<28))
    z = (1-fy)*((1-fx)*sample(iy,ix)+fx*sample(iy,ix+1)) + fy*((1-fx)*sample(iy+1,ix)+fx*sample(iy+1,ix+1))
    z = np.minimum(1., z / np.maximum(z.max(axis=(1,2),keepdims=True), 1.e-6))
    geom = np.stack((145*tilt, 155*np.log(aspect)),axis=1)
    return z, np.rint(geom).astype(np.int16)

def ringbits(sk):
    p = np.pad(sk,((0,0),(1,1),(1,1)),mode='constant')
    return (p[:,:-2,1:-1]+2*p[:,:-2,2:]+4*p[:,1:-1,2:]+8*p[:,2:,2:]
            +16*p[:,2:,1:-1]+32*p[:,2:,:-2]+64*p[:,1:-1,:-2]+128*p[:,:-2,:-2])

def make_tables():
    bits = (np.arange(256)[:,None] >> np.arange(8)[None,:]) & 1
    nextbits = np.roll(bits,-1,axis=1)
    arcs = np.sum((bits==0)&(nextbits==1),axis=1)
    n = bits.sum(axis=1)
    removable = (arcs==1)&(n>=2)&(n<=6)
    top,right,bot,left = bits[:,::2].T
    keep1 = ~(removable & ((top*right*bot)==0) & ((right*bot*left)==0))
    keep2 = ~(removable & ((top*right*left)==0) & ((top*bot*left)==0))
    return keep1,keep2,n,arcs

tables = make_tables()

def ridgefields(z):
    # Parallel topology-preserving thinning. Compute at two isophotes so a
    # nearly joined stroke isn't forced into an irrevocable topology decision.
    total = np.zeros((len(z),7,28,28),np.float32)
    n,arcs = tables[2:]
    m = np.arange(28)
    smooth = np.exp(-.5*((m[:,None]-m[None,:])/1.25)**2).astype(np.float32)
    for threshold in (.30,.58):
        sk = (z>threshold).astype(np.uint8)
        for step in range(20):
            old = sk
            sk = sk*tables[0][ringbits(sk)]
            sk = sk*tables[1][ringbits(sk)]
            if np.array_equal(sk,old):
                break
        bits = ringbits(sk)
        # Estimate tangent from a tensor of links on the lattice skeleton.
        # Use diagonal links only if not redundant with an axial neighbour.
        vertical = ((bits&1)>0).astype(np.float32) + ((bits&16)>0)
        horiz = ((bits&4)>0).astype(np.float32) + ((bits&64)>0)
        desc = ((bits&28)==8).astype(np.float32) + ((bits&193)==128)
        asc = ((bits&7)==2).astype(np.float32) + ((bits&112)==32)
        tx = smooth @ ((horiz-vertical)*sk) @ smooth.T
        ty = smooth @ ((desc-asc)*sk*1.41421356) @ smooth.T
        disp = smooth @ ((horiz+vertical+(desc+asc)*1.41421356)*sk) @ smooth.T
        # cos(2*theta), sin(2*theta) parameterize unoriented tangents.
        fac = sk / np.maximum(np.abs(tx)+np.abs(ty),.25*disp+1.e-5)
        dirs = [np.maximum(0,v)*fac for v in (tx,ty,-tx,-ty)]
        ends = (n[bits]==1)*sk
        forks = (arcs[bits]>=3)*sk
        total += np.stack(dirs+[sk,ends,forks],axis=1)
    return total*.5

def fields(z):
    # Blur perpendicular to each derivative as in a Sobel filter.
    p = np.pad(z,((0,0),(1,1),(1,1)),mode='constant')
    v = p[:,:-2,:] + 2*p[:,1:-1,:] + p[:,2:,:]
    h = p[:,:,:-2] + 2*p[:,:,1:-1] + p[:,:,2:]
    gx = (v[:,:,2:] - v[:,:,:-2])*.25
    gy = (h[:,2:,:] - h[:,:-2,:])*.25
    diagon = .70710678
    a = (gx+gy)*diagon
    b = (gy-gx)*diagon
    grads = []
    for dot, cross in ((gx,gy),(a,b),(gy,gx),(b,a),(-gx,gy),(-a,b),(-gy,gx),(-b,a)):
        grads.append(np.maximum(dot-np.abs(cross),0))
    # Distinguish inside a ring from inside a bay facing each compass point.
    ink = np.minimum(1., 1.4*z)
    l = np.maximum.accumulate(ink,axis=2)
    r = np.maximum.accumulate(ink[:,:,::-1],axis=2)[:,:,::-1]
    t = np.maximum.accumulate(ink,axis=1)
    d = np.maximum.accumulate(ink[:,::-1,:],axis=1)[:,::-1,:]
    lr = np.minimum(l,r)
    td = np.minimum(t,d)
    mid = np.minimum(lr,td)
    bg = 1-ink
    bay = [(a-ink)*bg for a in (mid,lr,td)]
    bay += [(a-mid)*bg for a in (np.minimum(td,r),np.minimum(td,l),np.minimum(lr,d),np.minimum(lr,t))]
    # Soft flood fill: the value is the least ink one has to cross to escape.
    cover = np.minimum(1., z*1.5)
    fill = np.ones_like(z)
    fill[:,0,:] = cover[:,0,:]
    fill[:,-1,:] = cover[:,-1,:]
    fill[:,:,0] = cover[:,:,0]
    fill[:,:,-1] = cover[:,:,-1]
    for step in range(48):
        n = np.minimum(fill[:,2:,1:-1],fill[:,:-2,1:-1])
        n = np.minimum(n,np.minimum(fill[:,1:-1,:-2],fill[:,1:-1,2:]))
        n = np.maximum(cover[:,1:-1,1:-1],n)
        if step >= 16 and step%4 == 0 and np.array_equal(n,fill[:,1:-1,1:-1]):
            break
        fill[:,1:-1,1:-1] = n
    hole = np.maximum(fill-cover,0.)
    return np.concatenate((np.stack(grads+[z]+bay+[hole],axis=1),ridgefields(z)),axis=1)

def poolmat(shift):
    ds = (np.arange(6)[:,None]*4+3.5+shift) - np.arange(28)[None,:]
    a = np.exp(-.5*(ds/1.85)**2)
    return (a/a.sum(axis=1,keepdims=True)).astype(np.float32)

def norms(parts):
    return [np.sqrt((q*q).sum(axis=(1,2,3),keepdims=True)+floor) for q,floor in zip(parts,(.1,.1,.1,1.,1.1,.05,.05,.10,.13))]

def packed(maps,py,px,den):
    a = (py @ maps) @ px.T
    signed = np.sqrt(np.maximum(a[:,:8],0.))
    orient = np.sqrt(np.maximum(a[:,:4]+a[:,4:8],0.))
    # Keep some magnitude information for enclosed spaces, including absence.
    parts = [signed,orient,np.sqrt(a[:,8:9]),np.maximum(a[:,9:16],0.)**.8,np.sqrt(np.maximum(a[:,16:17],0.)),
             np.sqrt(a[:,17:21]),np.sqrt(a[:,21:22]),np.sqrt(a[:,22:23]),np.sqrt(a[:,23:24])]
    if den is None:
        den = norms(parts)
    weights = (.69,.33,.39,.435,.28,.44,.28,.17,.16)
    blocks = [a*((1024*w)/d) for a,d,w in zip(parts,den,weights)]
    out = np.concatenate(blocks,axis=1).transpose(0,2,3,1)
    return np.ascontiguousarray(np.rint(out).astype(np.int16)),den
'''

infer = '''
# Each row is ONE rendered glyph. Its entries and norm depend on no other glyph.
local = CONSTANTS['glyphs'].reshape(-1,6,6,28)
meta = CONSTANTS['geometry'].reshape(-1,2)
full = np.concatenate((local.reshape(len(local),-1),meta),axis=1).astype(np.float32)
lengths = np.sum(full*full,axis=1)
stylecost = CONSTANTS['stylecost'].reshape(-1)

def classify(img):
    z,g = canonical(img)
    maps = fields(z)
    center = poolmat(0)
    a,den = packed(maps,center,center,None)
    q = np.concatenate((a.reshape(len(a),-1),g),axis=1).astype(np.float32)
    # Dot products are small integers exactly representable as float32.
    dist = lengths[None,:] - 2*(q @ full.T) + stylecost[None,:]
    dclass = dist.reshape(len(a),10,-1)
    k = 16
    order = np.argpartition(dclass,k-1,axis=2)[:,:,:k]
    idx = (order + np.arange(10)[None,:,None]*dclass.shape[2]).reshape(len(a),10*k)
    selected = local[idx].astype(np.float32)
    # Elastic match. Consider four placements of the boundaries between
    # patches so a pen tip at a boundary is not arbitrarily cut in two.
    fixed = None
    best = None
    for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
        if dx == 0 and dy == 0:
            shifted = a
        else:
            shifted,unused = packed(maps,poolmat(1.65*dy),poolmat(1.65*dx),den)
        dif = selected - shifted[:,None,:,:,:]
        cell = np.einsum('bijhc,bijhc->bijh',dif,dif)
        # Sum over every overlapping 2x2 patch, including clipped ones
        # around the border. Each cell participates exactly four times.
        p = np.pad(cell,((0,0),(0,0),(1,1),(1,1)),mode='constant')
        region = p[:,:,:-1,:-1]+p[:,:,1:,:-1]+p[:,:,:-1,1:]+p[:,:,1:,1:]
        if fixed is None:
            fixed = cell.sum(axis=(2,3))
            best = region
        else:
            area = np.array([1,2,2,2,2,2,1],np.float32)
            best = np.minimum(best, region + 300*(dx*dx+dy*dy)*(area[:,None]*area))
    final = .25*fixed + (.75/4)*np.sum(best,axis=(2,3))
    delta = meta[idx].astype(np.float32)-g[:,None,:]
    final += np.sum(delta*delta,axis=2) + stylecost[idx]
    ranks = np.sort(final.reshape(len(a),10,k),axis=2)
    return np.argmin(.65*ranks[:,:,0]+.25*ranks[:,:,1]+.10*ranks[:,:,2],axis=1)

def predict(x):
    labels = np.empty(len(x), dtype=np.int64)
    for j in range(0,len(x),48):
        labels[j:j+48] = classify(x[j:j+48,0])
    return labels
'''

# The coordinates below are typographic design units: a cap is ten units
# high, and the nominal width is ten before the nib and aspect transform.
# M = move, L = line, Q = quadratic and C = cubic. Independent paths mean
# that the writer can lift the pen. The numbers at the left are style weights.
styles = [[] for j in range(10)]
def glyph(d, freq, path):
    styles[d].append((freq,path))

# Zero: ovals, squared/pointed ovals, seams in different places, overshoots.
glyph(0,8,'M 6 .2 C 3 -.8 .2 1 .1 5 C -.3 8.4 2 10 5.3 10 C 9 10 10.2 7 10 4.4 C 9.8 1.3 8 0 5.4 .3')
glyph(0,6,'M 8 1 C 6 -1 3 -.3 1.3 2.4 C -.7 5.4 -.5 9.2 3.2 9.9 C 7.7 11 10 7.4 10 4 C 10 2 9 .6 7.1 .3')
glyph(0,4,'M 3 1.1 C .3 1.4 -.3 4.4 .3 7 C .7 9.5 2.6 10.4 5.6 10 C 8.8 9.5 10.2 7.5 10 4.1 C 9.8 .6 7 -.6 4.1 .2 Q 1.8 .9 1.5 3')
glyph(0,3,'M .5 6 Q .1 9.6 4.1 10 Q 8.8 10.8 9.8 6.3 Q 11 .7 6 .1 Q 1 -.7 .3 4.7 L .4 7.2')
glyph(0,3,'M 5 0 C 2 .4 -.4 4 .1 7.5 Q .7 10 4 10 C 8.2 10.2 10.7 7.5 9.7 3.8 Q 8.9 .2 5 0')
glyph(0,1.5,'M 6.8 2.2 Q 5 -.5 3.7 .3 C .9 1.2 -1 6.4 .8 9 Q 4.3 11.4 7.7 9.2 C 11 6.2 10.6 2.5 8.7 1 Q 6.8 -.5 4.9 .4')
glyph(0,.5,'M 8.5 1.8 C 6.9 -1.9 1.6 -.3 .3 4.6 C -1.1 9.2 3 11 6.5 9.8 Q 11 8.3 9.6 3.8')

# One: the plain downstroke, with optional entry flick and/or baseline.
glyph(1,10,'M 5 .1 L 5 10')
glyph(1,6,'M 5.1 .1 Q 4.7 4.9 5.2 10')
glyph(1,4,'M 4.1 1.4 L 5.3 .1 L 5 10')
glyph(1,4,'M 1.3 3 L 5.6 .1 Q 5.1 5.3 5.3 10')
glyph(1,2,'M .4 3.1 Q 3.5 2.2 5.4 0 L 5.3 10')
glyph(1,1,'M 3 1.4 L 5.3 .1 L 5.2 9.9 M 1.4 10 L 8.8 10')
glyph(1,.65,'M 5 .1 L 5 9.8 M 1.9 10 L 8 10')
glyph(1,.8,'M 2.1 2.8 L 5.5 0 L 5 9.6 Q 5.5 10 7.4 9.9')

# Two: arch, falling diagonal, then a baseline. The returning pen can
# cross the diagonal to produce a loop or turn back without crossing.
glyph(2,7,'M .3 2.1 C 3 -1.2 8.2 -.6 8.2 2.2 C 8.5 4.2 4.6 6.6 .6 9.7 Q 4.8 9 10 9.8')
glyph(2,5,'M .3 2.6 Q 2.9 -.5 5.2 .1 C 9.7 .1 9.2 3.6 7 5.4 L .4 9.6 Q 3.2 9 6.7 9.6 Q 9.4 10.1 10.4 9.4')
glyph(2,5,'M .9 2.1 C 1.5 -.7 6 -.7 7.2 1.7 C 9.5 5.2 5.3 8.4 1.4 9.7 Q -.7 10.6 .4 8.8 C 2.2 6.6 6.2 10.4 10 9.6')
glyph(2,4,'M 1.2 1.7 Q 4.8 -1.9 6.9 1.3 C 9.7 5.9 4.1 10.6 .7 9.9 C -1.5 9.1 .9 6.5 3.7 7.5 Q 6.9 8.4 10.1 10')
glyph(2,3,'M .1 3.5 C .1 1.1 2.7 -.9 5.6 .2 C 9.3 1.5 8 4.8 4.2 7.3 L .9 9.7 L 10 9.7')
glyph(2,2,'M 1.1 1.4 Q 7.9 -2.8 7.3 2.6 Q 7.4 5.3 .3 9 Q .6 9.9 5.7 9.8 Q 9.4 10.1 10.1 8.1')
glyph(2,1.5,'M .4 1.5 Q 4.6 -.4 7.4 .3 Q 10.1 1.2 7.6 3.6 L .5 9.6 Q 5.4 9.3 10.1 10')
glyph(2,1,'M .8 3.8 Q -.5 .5 3.3 .1 C 7.4 -.7 9.5 2.4 7.8 4.6 Q 6.7 6.5 .4 9.6 L 10 9.8')

# Three: cursive double arc, pointed waist, or angular top as in a 3
# written with the same first two strokes as a 7.
glyph(3,7,'M .7 1.6 C 3.7 -.5 8.5 -.7 8.6 1.9 Q 8.7 3.8 3.6 4.9 C 10.5 3.9 11.8 7.6 7.6 9.4 Q 3.7 11.2 .3 8.5')
glyph(3,5,'M 1 .4 Q 9.5 -1.3 8.2 1.9 Q 7.7 3.5 3.5 4.6 Q 8.9 4 9.6 6.5 C 11 9.6 5.1 11.6 .5 9.1')
glyph(3,4,'M .6 1.1 Q 4.8 -.2 9.2 .1 L 3.5 4.7 C 6.8 3.9 9.8 5.1 9.5 7 C 9.2 9.7 3.3 11.4 .3 8.4')
glyph(3,4,'M 1.1 1.4 C 6.4 -1.9 11.5 .7 7.5 3.5 L 3.4 5.1 Q 8.1 3.9 9.4 6.3 Q 11 9.7 1 10')
glyph(3,3,'M 1.2 1.1 Q 7.2 -1.1 8.2 .9 Q 9.4 3.1 2.8 5.2 Q 1.1 5.6 2 4.6 C 5.5 3 10 4.9 9.8 7.3 Q 9.6 10.3 1.1 9.7')
glyph(3,2,'M .4 .6 L 8.8 .2 L 3.7 4.6 Q 8.9 3.9 9.8 6.6 Q 11.3 10.9 .7 9.8')
glyph(3,2,'M .4 2 Q 4 -1.4 7.6 .5 Q 11.5 2.3 5.4 4.9 C 11 5.7 10.7 8.6 7 9.7 Q 3.1 10.9 .5 8.9')

# Four: left upright plus the crossbar, and a downstroke; also joined,
# triangular, round-bottomed or with the pen retracing the right upright.
glyph(4,7,'M 1.3 .4 L .4 5.8 Q .4 6.3 2 6.2 L 9.8 5.8 M 7.8 .1 L 7.5 10')
glyph(4,6,'M 2.4 .4 Q -.2 4.3 .6 5.7 Q 1.6 7 7.8 5.2 M 7.9 .2 L 7.6 10')
glyph(4,5,'M 1.4 .1 Q -.3 3.2 .4 5.3 C 1.4 7.4 6.7 5.6 7.9 3.5 L 8 .6 L 7.7 10')
glyph(4,4,'M 6.9 .2 L .3 6 L 10.1 6 M 7.6 .4 L 7.6 10')
glyph(4,3,'M 3 .1 L .3 5.9 L 7.7 5.4 M 7.8 1.1 Q 7.4 5.6 8.1 10')
glyph(4,2.5,'M 7.8 10 L 7.6 .1 L .3 6.4 L 10.2 5.9')
glyph(4,2,'M 1.5 .2 L .4 5.1 Q .3 6.1 2 6 L 7.8 4.9 L 8.3 .3 L 7.8 10')
glyph(4,1.4,'M 3.9 .1 Q 1.9 2.8 .2 5.7 Q 3.5 7.2 9.9 5.9 M 7.8 2.3 L 7.8 10')

# Five: open bowl and roof. Vary which parts touch and which are drawn
# in one stroke. Include the continuous, S-like handwritten form.
glyph(5,6,'M 9.4 .3 L 1.9 .3 L 1.1 4.9 C 4.3 3.4 9.6 4 9.5 6.8 C 9.5 9.4 4.3 11.4 .2 8.6')
glyph(5,5,'M 2.3 .5 Q 2 2.8 .8 4.5 C 4.9 3.5 10.1 4.8 9.2 7.4 Q 8.2 10.4 .7 9.8 M 1.9 .4 Q 6 1 9.9 .1')
glyph(5,4,'M 9.8 .2 Q 4.6 .9 2 .3 Q 1.7 2.8 .9 4.5 C 5.3 4.1 10.8 5.5 8.9 8.3 Q 7.1 10.6 .7 9.5')
glyph(5,4,'M 2.2 .3 Q -.9 3.7 1.7 4.6 C 4.1 5.1 9.8 4.8 9.4 7.3 Q 9.4 10.7 .4 9.8 M 1.4 .5 L 9.7 .2')
glyph(5,3,'M 9.6 .4 C 6.2 -.2 1.7 -.1 1.4 2.1 C .8 4.6 4.3 4.1 7.7 5.1 C 12 6.7 8.2 10.3 3.8 10 Q 1.4 9.9 .3 8.9')
glyph(5,2.5,'M 8.9 .5 Q 2.1 -1.3 .8 2.1 Q -.5 3.9 4.8 5.1 C 10.6 6.6 11 9.6 5.2 9.9 Q 2.1 10.4 .3 8.9')
glyph(5,2,'M 9.9 .2 L 2.3 .5 L 1.5 3.9 Q 7.7 3.6 9.1 6.5 C 10.7 9.8 3 11.5 .5 8.3')
glyph(5,1,'M 2.1 .8 L .7 5 C 7.5 2.1 12.5 6 7.9 8.9 Q 4.4 11.1 .9 9.8 M 3.2 .5 L 10 .2')

# Six: clockwise return through the bowl or ending within the bowl;
# short/long entry arc with or without a curved hook at the top.
glyph(6,7,'M 7.7 .2 C 4.3 1.3 .1 5.5 .6 7.8 C .9 10.8 7.7 10.9 9.3 8 C 10.9 5.2 6.4 4 3 5.8 Q 1.6 6.6 1.1 8')
glyph(6,5,'M 7.1 .1 Q 1.4 2.2 .7 6.5 C -.3 10.4 5.2 11.2 8.3 9.3 C 12.2 6.5 8.7 3.3 4.9 5 Q 2.2 6.1 1.3 8.4')
glyph(6,4,'M 9.4 1.1 C 6.2 -1.6 1.4 1.5 .6 6.3 C -.2 10.1 4.9 11.3 8.2 9.2 C 11.9 6.5 8.4 3.5 4.5 5.1 Q 2.1 5.9 .7 7.7')
glyph(6,4,'M 5.9 0 C 2.1 2.2 -1.4 7.2 1.4 9 Q 5.5 11.7 9.3 8.6 C 12.4 5.2 6.6 4.1 3.7 6.3')
glyph(6,3,'M 6.2 .1 Q 2.3 2.6 .7 6.8 Q -.8 10.6 5.6 9.9 Q 10.1 9.6 9.8 6.9 Q 9.1 4 5.6 5.2 Q 2.6 6.1 .5 8.8')
glyph(6,2,'M 8.6 .1 C 2.7 .5 -1.1 5.6 1.3 8.6 C 3.7 11.6 9.6 9.7 9.5 7 C 9.5 4.6 5.7 4.3 3.4 5.8 Q 1.4 7.5 3.7 8')
glyph(6,1.2,'M 4.8 0 Q .7 3.8 .8 7.6 Q .9 10.9 6 9.9 Q 10.8 9 9 6.9 Q 6.8 5.2 2.2 7.1')

# Seven: roof, optional initial serif, downstroke, optional cross-stroke.
glyph(7,7,'M .3 .6 Q 4.6 .1 9.7 .3 L 3.4 10')
glyph(7,6,'M .3 .3 Q 4.6 1.1 9.5 .2 Q 7.3 4.9 4.2 10')
glyph(7,5,'M .4 .6 Q 2.7 -.3 4.6 .3 Q 7.6 1.5 9.5 .6 Q 5.5 6.3 4.8 10')
glyph(7,3,'M .5 2.6 L .9 .3 L 9.6 .2 L 4.4 10')
glyph(7,3,'M .2 .2 Q 4.9 .7 9.8 .4 L 3.8 10 M 2.6 5.1 L 9.1 5.2')
glyph(7,2,'M .2 .7 Q 2.7 -.4 5.7 .4 L 9.6 .8 Q 6 5.2 5 10 M 3.4 5.4 L 8.7 5.2')
glyph(7,2,'M .2 1.3 Q 2.2 -.1 6.6 .2 Q 10.6 .5 9.2 2.2 Q 5 7 4.8 10')
glyph(7,1,'M .4 .3 L 9.8 .3 Q 7.7 4.3 4.9 8.1 Q 3.8 9.5 2.4 10')

# Eight: crossed ribbon, two joined ovals, different seam locations,
# open seams and unequal loops. Some writers begin at the crossing.
glyph(8,7,'M 7.7 .6 C 4.3 -1.2 -.8 .6 1.5 3 Q 2.7 4.3 6.7 5.9 C 12.1 8.3 8.6 10.6 4.3 9.9 C -.3 9.6 -.8 7.1 3.6 5.3 C 9 3.3 10.6 1.8 7.7 .6')
glyph(8,6,'M 5.3 4.7 C .3 3.5 -.6 .7 3.9 .1 C 9.8 -.7 11.1 2.5 6.7 4.5 C 3.4 6 .1 6.6 .4 8.3 C 1.1 10.9 8.8 10.8 9.7 8.3 C 10.3 6.6 7.6 5.4 5.3 4.7')
glyph(8,4,'M 8.7 1.8 C 7.2 -.7 1.6 -.5 .9 1.4 C .1 3.3 5.7 5 8.2 6.7 C 12.3 9.8 4.8 11.4 1.3 9 C -1.8 6.5 6.4 4.6 8.5 2.8 Q 10.2 .5 7.5 .1')
glyph(8,4,'M 6.5 .3 C 1.9 -.7 -.5 1.8 2.6 3.9 L 7.8 6.2 C 11.9 8.3 8.4 11.2 3.8 9.8 C -.9 8.6 1.2 6.3 4.3 4.7 Q 11.3 1.6 7.3 .4')
glyph(8,3,'M 4.8 4.6 C 9.4 4.7 11 1.4 7.6 .4 C 2.7 -1.4 -1.4 1.9 2.1 3.8 Q 3.3 4.6 5.2 4.6 C -.5 4.8 -1.3 8.9 3 9.9 C 7.8 11.1 12.4 8.2 8.7 5.9 Q 7.5 4.8 4.8 4.6')
glyph(8,3,'M 8.4 2 C 10.4 .2 6.7 -.8 3.6 .3 C -.5 1.8 1.2 3.9 5.7 5.4 Q 12.7 8.1 8.2 9.6 C 4.1 11.3 -.9 9.2 1.1 7 C 2.7 5.2 8.9 3.9 8.9 1.4')
glyph(8,2,'M 7.1 1 C 4.3 -.9 .6 .4 .8 2.2 Q .6 4.4 5.7 5.1 Q 10.2 6.4 9.2 8.5 C 8.2 10.9 1.4 10.6 .6 8.5 Q -.1 6.7 5.1 4.8 Q 10.2 2.3 8.6 .6')

# Nine: an oval followed by a descender, or a return down through the
# right side of the oval. The tail may be straight, bent or deeply hooked.
glyph(9,7,'M 8.6 1.6 C 7 -.7 2.7 -.4 .9 1.9 C -1.6 5.6 3.6 6.7 6.9 4.7 Q 8.8 3.3 9.1 1.1 L 8.2 10')
glyph(9,6,'M 8.8 2.3 C 8 .3 4.8 -.8 2.1 1 C -1.5 3.5 .6 6.2 4.1 5.6 C 7.1 5.3 8.8 2.8 8.8 .8 Q 8.1 5.5 9.1 10')
glyph(9,4,'M 8.1 .8 C 4.3 -1.2 .1 .9 .4 3.2 C .5 6.5 7.5 6.1 8.9 2.8 Q 9.8 .2 8.5 .3 Q 9.5 6.4 5.8 10')
glyph(9,4,'M 8.8 1.5 C 4.9 -1.6 -.5 .9 .4 3.8 C 1.3 6.6 6.5 5.3 8.9 1.6 C 10.2 5.8 8.8 10.2 3.4 10')
glyph(9,3,'M 8.8 1 C 5.7 -1 .9 .3 .5 2.9 C .2 5.3 4.5 6.5 7.4 4 Q 9.2 2.6 9.1 1 M 9.1 .5 L 7.8 10')
glyph(9,2,'M 8.9 1.2 C 4.4 -.7 .4 .5 .4 3.1 C .3 6.2 7.2 5.8 8.8 2.9 Q 9.2 6.2 8.5 8.1 Q 7.4 11.4 2.8 9.3')
glyph(9,2,'M 6.5 .4 C 3.1 -1.1 -1.3 2.2 .8 4.6 C 2.8 6.8 8.4 3.5 9 .6 Q 8.4 5.4 8.9 10')
glyph(9,1,'M 8.7 .2 Q 3 -.8 .6 2.5 C -1 5.9 5.6 5.7 8.6 2.2 L 6.1 10')


def parse(path):
    s = iter(path.split()); substrokes = []; segments = None
    for letter in s:
        pts = [(float(next(s)),float(next(s))) for j in range({'M':1,'L':1,'Q':2,'C':3}[letter])]
        if letter == 'M':
            segments = []; substrokes.append(segments)
        else:
            segments.append(np.array([last]+pts))
        last = pts[-1]
    return substrokes


def render(path, count, rng):
    # All random quantities describe nib, hand movement or proportions. No
    # test is performed on a rendered digit to accept or reject the drawing.
    def n(shape):
        return np.clip(rng.standard_normal(shape),-2.4,2.4).astype(np.float32)
    def u(lo,hi,shape):
        return rng.uniform(lo,hi,shape).astype(np.float32)
    c=count; col=(c,1)
    pts=[]; pressures=[]
    angle = n(col)*.19
    slant = n(col)*.33-.10
    width = np.exp(n(col)*.25)*.70
    waist = n(col)*.90; waistx=n(col)*.70
    asym=n(col)*.35; diag=n(col)*.32; tilt=n(col)*.39
    warp=n((c,2,3,3))*.53
    baseNib = u(1.,4.2,col) + u(0,1,col)**4*1.25
    nibx = np.exp(n(col)*.17)
    nibly = np.exp(n(col)*.12)
    for segs in path:
        xs=[]; prev=None
        drift=n((c,1,2))*.32
        spread=u(.16,.78,(c,1,1))
        for seg in segs:
            k=len(seg)-1
            # Use closely spaced Bezier evaluations even along straight lines.
            length=np.hypot(*np.diff(seg,axis=0).T).sum()
            steps=max(3, math.ceil(length*5))
            t=np.linspace(0,1,steps+1,dtype=np.float32)[:,None]
            b=np.concatenate([math.comb(k,j)*t**j*(1-t)**(k-j) for j in range(k+1)],axis=1)
            cp=seg[None,:,:].astype(np.float32)+drift+n((c,k+1,2))*spread
            if prev is not None:
                cp[:,0,:]=prev
            prev=cp[:,-1,:]
            xs.append(b @ cp)
        arr=np.concatenate(xs,axis=1)
        # Small variation in the free starting and ending point, including
        # whether a loop has a gap. Freeze points beyond the two endpoints.
        ticks=np.broadcast_to(np.arange(arr.shape[1])[None,:],arr.shape[:2])
        i0=(u(0,1,col)**3*.11*arr.shape[1]).astype(int)
        i1=arr.shape[1]-1-(u(0,1,col)**3*.11*arr.shape[1]).astype(int)
        idx=np.minimum(np.maximum(ticks,i0),i1)
        arr=arr[np.arange(c)[:,None],idx]
        x=(arr[:,:,0]-5)/5; y=(arr[:,:,1]-5)/5
        wx=1-np.minimum(1,x*x); wy=1-np.minimum(1,y*y)
        xx=(5*x*(1+asym*.63*y)+waistx*wy)*width
        yy=5*y + waist*wy + tilt*x*y + diag*wx
        bx=np.stack(((1-x)**2/4,(1-x*x)/2,(1+x)**2/4),axis=2)
        by=np.stack(((1-y)**2/4,(1-y*y)/2,(1+y)**2/4),axis=2)
        xx+=np.einsum('bsi,bij,bsj->bs',bx,warp[:,0],by)
        yy+=np.einsum('bsi,bij,bsj->bs',bx,warp[:,1],by)
        xx=xx + slant*yy
        px=np.cos(angle)*xx-np.sin(angle)*yy
        py=np.sin(angle)*xx+np.cos(angle)*yy
        pts.append(np.stack((px,py),axis=2))
        tt=idx.astype(np.float32)/max(1,arr.shape[1]-1)
        brush=1 + .11*n(col) + .14*n(col)*np.cos(np.pi*tt) + .12*n(col)*np.sin(2*np.pi*tt) + .09*n(col)*np.cos(2*np.pi*tt)
        taper=1-u(.03,.45,col)*np.exp(-14*(tt-i0/arr.shape[1]))-u(.03,.45,col)*np.exp(-14*(i1/arr.shape[1]-tt))
        pressures.append(np.maximum(.37,brush*taper))
    pts=np.concatenate(pts,axis=1)
    # These bounds normalize only this single drawing to the specified size.
    ext=pts.max(axis=1)-pts.min(axis=1)
    ctr=.5*(pts.max(axis=1)+pts.min(axis=1))
    scale=(20-baseNib)/np.maximum(ext.max(axis=1,keepdims=True),1.)
    pts=(pts-ctr[:,None,:])*scale[:,None,:]+13.5 + u(-.5,.5,(c,1,2))
    radii=np.clip(.5*baseNib*np.concatenate(pressures,axis=1),.4,3.6)
    x=pts[:,:,0]; y=pts[:,:,1]; ix=np.rint(x).astype(np.int32); iy=np.rint(y).astype(np.int32)
    dx=ix.astype(np.float32)-x; dy=iy.astype(np.float32)-y
    img=np.zeros((c,28,28),np.float32); dest=img.reshape(-1)
    row=np.arange(c,dtype=np.int32)[:,None]*784
    blur=u(.68,1.4,col); bhalf=.5*blur
    # Fixed support from the nib bound 3.6, blur bound 1.4,
    # and the analytic stretch bounds exp(2.4*.17), exp(2.4*.12).
    xmax,ymax=7,6
    for j in range(-ymax,ymax+1):
        y2=((dy+j)/nibly)**2
        line=row+28*np.clip(iy+j,0,27)
        for i in range(-xmax,xmax+1):
            distance=np.sqrt(((dx+i)/nibx)**2 + y2)
            alpha=(radii+bhalf-distance)/blur
            pos=(line+np.clip(ix+i,0,27))
            use=alpha>0
            np.maximum.at(dest,pos[use],alpha[use])
    return np.minimum(img,1.)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    (out/'solution.py').write_text(features+infer)
    env={}; exec(features,env)
    py=env['poolmat'](0)
    # Same number of exemplars per class. Allocate by the design weights,
    # not by any measurement of how a glyph renders or how it scores.
    perclass=11264
    bank=np.empty((10,perclass,6,6,28),np.int16)
    geo=np.empty((10,perclass,2),np.int16)
    costs=np.empty((10,perclass),np.int32)
    for d in range(10):
        rs=styles[d]; total=sum(w for w,p in rs); upto=0; ws=0
        for h,(w,p) in enumerate(rs):
            ws+=w; end=round(ws*perclass/total)
            rng=np.random.default_rng(14503+d*127+h)
            spec=parse(p)
            for start in range(upto,end,64):
                c=min(64,end-start)
                z,g=env['canonical'](render(spec,c,rng))
                fields1=env['fields'](z)
                desc,unused=env['packed'](fields1,py,py,None)
                bank[d,start:start+c]=desc
                geo[d,start:start+c]=g
            # A small description-length charge for a less usual construction.
            costs[d,upto:end]=int(1550*math.log(10/w))
            upto=end
        print('constructed digit',d,flush=True)
    np.savez(out/'constants.npz',glyphs=bank,geometry=geo,stylecost=costs)

if __name__=='__main__':
    main()
