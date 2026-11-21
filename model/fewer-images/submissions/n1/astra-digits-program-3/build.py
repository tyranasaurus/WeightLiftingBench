"""Build from geometric handwriting models only. No input files or fonts."""
import sys
from pathlib import Path
import numpy as np

# This block is used verbatim both to describe a template and in the recognizer.
features = '''import numpy as np

side = 28
mid = 13.5
axis = np.arange(side, dtype=np.float32)
planeY, planeX = np.meshgrid(axis, axis, indexing="ij")


def convaxis(a, k, ax):
    r = len(k) // 2
    padding = [(0, 0)] * a.ndim
    padding[ax] = (r, r)
    b = np.pad(a, padding)
    v = np.zeros_like(a)
    sl = [slice(None)] * a.ndim
    for j in range(len(k)):
        sl[ax] = slice(j, j + a.shape[ax])
        v += k[j] * b[tuple(sl)]
    return v


def blur(a, s):
    k = np.arange(-int(np.ceil(2.8*s)), int(np.ceil(2.8*s))+1, dtype=np.float32)
    k = np.exp(-.5*(k/s)**2)
    k /= k.sum()
    return convaxis(convaxis(a, k, -1), k, -2)


def sample(a, y, x):
    # The extra zero border ensures affine sampling outside the frame reads black.
    y = np.clip(y+1, 0, side+.999)
    x = np.clip(x+1, 0, side+.999)
    iy = y.astype(np.int32)
    ix = x.astype(np.int32)
    fy = (y-iy).astype(np.float32)
    fx = (x-ix).astype(np.float32)
    b = np.pad(a, ((0,0),(1,1),(1,1)))
    row = np.arange(len(a))[:,None,None]
    return ((1-fy)*((1-fx)*b[row,iy,ix]+fx*b[row,iy,ix+1]) +
            fy*((1-fx)*b[row,iy+1,ix]+fx*b[row,iy+1,ix+1]))


def extent(a):
    # Relative grey-level contour, with interpolation between successive pixels.
    h = a.max(axis=1,keepdims=True)*.22
    c = a > h
    l = c.argmax(axis=1)
    r = a.shape[1]-1 - c[:,::-1].argmax(axis=1)
    row = np.arange(len(a))
    v = a[row,l]
    w = a[row,np.maximum(0,l-1)]
    left = l - np.clip((v-h[:,0])/np.maximum(v-w,1e-5),0,1)
    v = a[row,r]
    w = a[row,np.minimum(a.shape[1]-1,r+1)]
    right = r + np.clip((v-h[:,0])/np.maximum(v-w,1e-5),0,1)
    return left.astype(np.float32), right.astype(np.float32)


def canonical(a):
    # Remove predominant slant and part of the variation in height and width.
    mass = np.maximum(a.sum(axis=(1,2)),1e-6)
    mx = (a.sum(axis=1)*axis).sum(axis=1)/mass
    my = (a.sum(axis=2)*axis).sum(axis=1)/mass
    y = planeY - my[:,None,None]
    x = planeX - mx[:,None,None]
    shear = np.clip((a*x*y).sum(axis=(1,2))/np.maximum((a*y*y).sum(axis=(1,2)),1e-5),-1.3,1.3)
    shear *= .94
    shift = shear[:,None,None]*y
    flat = sample(a,planeY,planeX+shift)
    l,r = extent(flat.max(axis=1))
    t,b = extent(a.max(axis=2))
    height = np.maximum(2.5,b-t)
    width = np.maximum(1.5,r-l)
    aspect = width/height
    h = 22*np.minimum(1,1.36/np.maximum(aspect,.001))**.4
    w = 18*(np.maximum(aspect,.05)/.76)**.49
    w = np.clip(w,5.3,24.0)
    h = np.clip(h,14.0,22.0)
    cx = .76*(l+r)/2+.24*mx
    cy = .76*(t+b)/2+.24*my
    yy = (planeY-mid)*(height/h)[:,None,None]+cy[:,None,None]
    xx = (planeX-mid)*(width/w)[:,None,None]+cx[:,None,None]
    xx += shear[:,None,None]*(yy-my[:,None,None])
    img = sample(a,yy,xx)
    img /= np.maximum(img.max(axis=(1,2),keepdims=True),1e-5)
    # As with shear, ignore a modest change of aspect when comparing model to image.
    return img,np.stack([np.log(np.maximum(aspect,.035)),shear],axis=1)


def channels(a):
    # Signed edge directions. A short smoothing step reduces pixel noise but
    # keeps the two sides of a thin line separate.
    a = blur(a,.62)
    gx = convaxis(a,[-.5,0,.5],-1)
    gy = convaxis(a,[-.5,0,.5],-2)
    mag = np.sqrt(gx*gx+gy*gy)
    v = []
    for k in range(8):
        ang = k*(np.pi/4)
        p = gx*np.float32(np.cos(ang))+gy*np.float32(np.sin(ang))
        v.append(np.maximum(p-.68*mag,0)*3.125)
    v.append(a*.75)
    return np.stack(v,axis=1)


def spatial(a, step, n, sigma):
    centers = mid + (np.arange(n,dtype=np.float32)-(n-1)*.5)*step
    k = np.exp(-.5*((centers[:,None]-axis[None,:])/sigma)**2).astype(np.float32)
    k /= k.sum(axis=1,keepdims=True)
    return (k @ a) @ k.T


def unit(v, axes):
    return v/np.sqrt(np.maximum((v*v).sum(axis=axes,keepdims=True),1e-9))


def inside(mask):
    # Propagate exterior background from the four sides. Anything not reached
    # belongs to an enclosed counter (hole), with four-connected background.
    allowed = ~mask
    seen = np.zeros_like(mask)
    seen[:,0] = allowed[:,0]
    seen[:,-1] = allowed[:,-1]
    seen[:,:,0] = allowed[:,:,0]
    seen[:,:,-1] = allowed[:,:,-1]
    for j in range(2*side+4):
        nxt = seen.copy()
        nxt[:,1:] |= seen[:,:-1]
        nxt[:,:-1] |= seen[:,1:]
        nxt[:,:,1:] |= seen[:,:,:-1]
        nxt[:,:,:-1] |= seen[:,:,1:]
        nxt &= allowed
        if np.array_equal(nxt,seen):
            break
        seen = nxt
    return allowed & ~seen


def topology(a):
    # Three levels of persistent enclosure. A very small hole casts a weaker vote.
    scaled = blur(a,.4)
    scaled /= np.maximum(scaled.max(axis=(1,2),keepdims=True),1e-6)
    records = []
    for level in (.20,.44,.68):
        holes = inside(scaled>=level)
        rows = holes.sum(axis=2)
        r = np.arange(side)
        upper = np.minimum(1, np.maximum(0,(17-r)/6))
        lower = 1-upper
        center = np.minimum(1, np.maximum(0,(8-np.abs(r-mid))/4))
        areas = np.stack([rows@upper,rows@lower,rows@center,rows.sum(axis=1)],axis=1)
        presence = (1-np.exp(-areas/8))
        size = np.sqrt(areas/100)
        # Occupancy on a coarse grid encodes the location, not just the size.
        loc = np.sqrt(np.maximum(0,spatial(holes.astype(np.float32),5,5,2.5)))
        rec = np.concatenate([presence,size,loc.reshape(len(a),25)],axis=1)
        records.append(rec)
    return np.stack(records,axis=1).astype(np.float32)


def describe(a):
    a,geometry = canonical(a)
    v = channels(a)
    # Power compression and per-block normalisation stop one heavily inked
    # stroke from drowning out all the other strokes.
    coarse = []
    for step,n,sigma in ((3.5,7,1.85),(5.5,5,2.9)):
        grid = np.maximum(spatial(v,step,n,sigma),0)
        part = grid[:,:8]**.76
        energy = (part*part).sum(axis=1,keepdims=True)
        part /= (energy+.003)**.23
        part = unit(part.reshape(len(a),-1),(1,))
        ink = unit(grid[:,8].reshape(len(a),-1),(1,))
        if n==7:
            coarse.extend([part*.78,ink*.29])
        else:
            coarse.extend([part*.48,ink*.25])
    desc = unit(np.concatenate(coarse,axis=1),(1,))
    # SIFT-like patches centred on a 5x5 grid. Keep a one-step halo for
    # bounded local displacement in the elastic verification stage.
    grid = np.maximum(spatial(v,2,14,1.2),0)
    grid[:,:8] = grid[:,:8]**.8
    grid[:,8] = grid[:,8]**.85*.52
    pads = np.pad(grid,((0,0),(0,0),(2,2),(2,2)))
    coords = np.arange(3,12,2)
    patches = []
    for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
        feat = []
        for iy,ix in ((-1,-1),(-1,1),(1,-1),(1,1),(0,0)):
            y = coords+iy+dy+2
            x = coords+ix+dx+2
            feat.append(pads[:,:,y[:,None],x].transpose(0,2,3,1))
        feat = np.concatenate(feat,axis=-1)
        norm = np.sqrt((feat*feat).sum(axis=-1,keepdims=True))
        # Introduce an explicit blank component, so aligning ink to blank
        # costs something even when one of the patches has no gradient.
        blank = np.full_like(norm,.11)
        feat = unit(np.concatenate([feat,blank],axis=-1),(-1,))
        patches.append(feat.reshape(len(a),25,-1))
    return desc, np.stack(patches,axis=1), topology(a), geometry
'''

recognizer = '''
coarseModel = CONSTANTS["global"].astype(np.float32)/255
coarseModel = unit(coarseModel,(1,))
patchModel = CONSTANTS["patch"]
holeModel = CONSTANTS["hole"]
geometryModel = CONSTANTS["geo"]
cuts = CONSTANTS["cuts"]
variantCost = CONSTANTS["cost"]


def batch(a):
    d,patch,hole,geo = describe(a)
    # Stage 1: compare whole shapes independently. Coarse screening is
    # separate for each class, so every class gets an elastic comparison.
    similarity = d @ coarseModel.T
    scores = np.zeros((len(a),10),dtype=np.float32)
    rows = np.arange(len(a))[:,None]
    disp = np.array([0,.020,.020,.020,.020,.032,.032,.032,.032],dtype=np.float32)
    for digit in range(10):
        start = int(cuts[digit])
        end = int(cuts[digit+1])
        gap = 1-similarity[:,start:end]
        delta = np.abs(geo[:,None,:]-geometryModel[None,start:end,:])
        geoCost = .017*np.minimum(delta[:,:,0],1.7)**1.4 + .010*np.minimum(delta[:,:,1],1.7)**1.4
        gap = gap + geoCost + variantCost[None,start:end]
        # Check topology on the closest whole shapes, using weak alignment
        # tolerance in grey level for variations in stroke pressure.
        take = np.argpartition(gap,39,axis=1)[:,:40]
        idx = start+take
        cost = gap[rows,take]
        ht = holeModel[idx]
        diff = np.abs(ht-hole[:,None,:,:])
        hd = (.047*diff[:,:,:,:4].mean(axis=(2,3)) + .031*diff[:,:,:,4:8].mean(axis=(2,3))
              +.082*diff[:,:,:,8:].mean(axis=(2,3)))
        cost += hd
        pick = np.argpartition(cost,7,axis=1)[:,:8]
        idx = idx[rows,pick]
        cost = cost[rows,pick]
        template = patchModel[idx].astype(np.float32)*(1/255)
        # Each model patch can move by one fine-grid cell; non-zero moves
        # have a small deformation cost. Only the input has to be shifted.
        best = np.full(template.shape[:3],2,dtype=np.float32)
        for j in range(9):
            ds = np.einsum("bkpv,bpv->bkp",template,patch[:,j],optimize=False)
            np.minimum(best,1-ds+disp[j],out=best)
        # Use both the mean patch error and the worst discrepancies. A
        # misplaced crossbar or an extra tail must matter to the decision.
        elastic = .64*best.mean(axis=2)+.36*np.sqrt((best*best).mean(axis=2))
        final = cost + 1.10*elastic
        scores[:,digit] = -final.min(axis=1)
    return scores.argmax(axis=1)


def predict(x):
    ans = np.empty(len(x),dtype=np.int64)
    for start in range(0,len(x),48):
        # Fixed internal batch size, including the final chunk, for identical
        # reduction kernels on singleton and full-batch calls.
        end = min(start+48,len(x))
        a = np.zeros((48,side,side),dtype=np.float32)
        a[:end-start] = x[start:end,0]
        ans[start:end] = batch(a)[:end-start]
    return ans
'''

# A small pen-plotter language. Consecutive knots have a smooth tangent
# (centripetal chord weights); ! is a corner, / lifts the pen. Each entry
# is one ordinary way to write a digit on a nominal 10-by-14 writing grid.
styles = [[],[],[],[],[],[],[],[],[],[]]
def style(d,weight,path):
    styles[d].append((weight,path))

# 0 -- the oval, egg, squared oval, overlaps at top and bottom, pointed apex.
style(0,5,'5,0 8,.8 9.8,4 10,7 9,11.5 6.5,14 3.5,13.8 .9,11 0,7 1,3 3,.6 5,0 6,.1')
style(0,3,'6,.3 3,.2 .7,3 0,8 .8,12 3.2,14 6.3,13.3 9.5,10 10,6 9.3,2 7.4,0 5.2,.4')
style(0,2,'6.8,.2 4.4,.5 1,4.5 0,9 .8,12.4 3.3,14 6.3,13.8 9,11.7 10,7.9 9.7,4.5 8,1.1 6,.2')
style(0,2,'6.8,0 3,.5 .8,2.8 .2,6 0,11 1.5,13.6 5.2,14 8.6,12.2 10,8 10,4 9.4,1.6 6.8,0')
style(0,2,'6.4,1.5 5.6,.2 3.2,1.6 .8,5.5 0,10 1.4,13.5 4.7,14 8,11.4 10,7.4 9.8,3.3 8,.4 5.7,0 3.7,2')
style(0,1,'1,5 2.1,1.6 5,0 8.8,1.8 10,6.3 8.7,11.5 5.8,14 2.5,13.6 .4,10.5 .1,6.3 1.9,3.8')
style(0,1,'4,.4 1.5,2.3 0,6.4 .1,10 2.1,13.7 5.6,14 9.3,11.5 10,7 9.4,2.7 7.4,0 5.7,.2')
style(0,.20,'5,0 8.8,1.5 10,6 8.5,12 5,14 1.4,12.5 0,7 1.6,2.5 5,0 / 1,13 !9,1')
# 1 -- simple descending pen strokes; upstroke/hook and pedestal variants.
style(1,6,'5.5,0 5.2,4.5 4.9,9.5 4.8,14')
style(1,4,'4.8,0 5.5,2 5.6,6 5,10.8 4,14')
style(1,3,'6,0 5.4,5.5 5.2,11.8 5.7,14')
style(1,2,'4.6,1.8 5.7,0 6.1,.4 5.5,5 4.9,10 4.8,14')
style(1,3,'1.5,4.6 3.9,2.8 !6,0 5.6,6.3 5.3,14')
style(1,1,'1.5,6 3.2,3.4 5.1,.2 5.8,.4 6,6.5 5.8,14')
style(1,.9,'2.4,3.3 !5.6,0 5.5,6.7 !5.4,13.7 / 1.4,14 5.5,13.8 9.2,14')
style(1,.7,'5,0 4.9,6.1 !4.7,13.8 6.5,14 8.8,13.8')
style(1,.45,'3.3,2.3 !5.3,0 5.1,7 !4.8,13.7 !1.3,14 !9,13.8')
# 2 -- curved upper bowl with diagonal downstroke and a baseline. The
# baseline may be straight, concave, or start as a cursive loop.
style(2,5,'0,3.8 1.5,1.5 4.7,0 7.7,.5 9,2.5 8.3,5.4 5.5,8.6 2,12 !.6,13.9 5.8,13.4 10.5,13.6')
style(2,3,'0,2.1 2.4,.4 5.7,0 8.7,1.1 9.5,3.6 8.1,6.6 5,9.5 !.5,13.5 3.9,13.9 7.4,14 10.8,12.7')
style(2,2,'.3,5 .5,2.8 2.4,.6 4.7,0 7.1,.8 8.4,3.1 7.9,5.5 5.3,9 2.4,12 !1.1,13.8 4.5,13 7.7,13.5 10,14')
style(2,2,'1.3,3.3 2.5,1.1 5.1,0 7.2,.7 8.1,2.6 7.4,5.3 4.6,9.7 2.5,13 .6,14 -.2,13.4 .4,12.2 2.1,11.7 5,12.4 8,13.3 10.9,13.3')
style(2,2,'1.6,2.5 3,.4 5.1,0 7.5,1.8 7.9,4.1 7,7.5 5.1,11.1 2.6,13.8 .7,14 0,13 .6,11.5 2.5,10.8 5.3,11 8.5,13.1 10.5,13.9')
style(2,1.5,'1.4,.9 3.4,0 5.7,.7 7.1,3.3 6.6,6.7 4.4,10.6 1.8,13.6 0,14 .3,12.5 2.1,11.1 4.6,11.6 7.8,13.1 10.2,12.8')
style(2,1,'0,2.7 2,1.1 4.2,0 6,.4 6.8,2.5 6.1,6.2 4.6,10 3.5,13.5 !4.1,14 7.8,13.5 10,13.5')
style(2,1,'0,1.9 2.8,.4 6.1,0 8.8,.6 !9.2,1.9 6.6,5.9 3,10.5 !.5,13.8 4.9,13.5 10,13.7')
style(2,.7,'.8,5.5 .2,3.8 1.3,1.4 3.9,0 6.7,.4 8.5,2.5 8.3,5.3 6.3,7.8 2.5,11.3 !.3,13.7 5.8,13.8 10,13.2')
# 3 -- a pair of right-hand lobes or a numeral with an angular top bar.
style(3,5,'0,2.1 2.3,.5 5.8,0 8.3,.8 8.8,2.5 7.4,4.9 4,6.8 !3.5,7.1 6.4,6.7 8.7,7.6 9.8,9.8 9,12 6.6,13.7 3.1,14 .1,12.4')
style(3,3,'0,1.2 3.5,0 6.6,.2 8.4,1.4 7.9,3.7 5.1,6.1 !3.5,7.2 6.8,7.4 9.1,8.9 9.5,11 8.2,12.9 5.6,14 2.3,13.7 .5,12.9')
style(3,2,'1,0 4.3,.1 7.6,.4 9,1.8 8.1,4.3 5,7.1 !4.1,7.6 6.3,7.4 8.4,8.3 9.3,10.4 8.5,12.7 5.8,14 3.2,13.8 .2,12.4')
style(3,2,'0,1.6 3.9,.8 !9.4,0 6.3,3.7 !3.3,6.8 5.9,6.6 8.7,7.4 10,9.7 9.3,12.3 6.7,13.9 3.7,14 .3,12.4')
style(3,1.4,'0,.7 4.5,.1 !9.3,0 6.9,3.4 !4.6,6.4 7.1,7 9.3,8.6 9.8,11 8.6,13.1 5.6,14 2.3,13.4 .3,11.7')
style(3,1,'1,1.4 3.5,0 6.2,.2 8.2,1.5 8.2,3.7 6.4,6.1 3.7,7 3,6.6 4.5,6 7.4,6.7 9.4,8.8 9.1,11.5 7.1,13.3 4.1,14 1.5,13.5 0,12')
style(3,1,'1.6,.6 4.5,0 7.4,1.1 8.9,3 8.2,5.2 !5.4,7 8.1,7.6 9.7,9.2 9.6,11.4 7.8,13.5 5,14 2.2,13.5 0,12.3')
style(3,1,'1,2.8 2.9,1.1 5.3,0 7.4,.2 8.5,1.9 8.3,3.8 6.2,5.8 !3.6,7.2 6.7,7.4 8.9,8.8 9.4,10.9 8.1,12.8 5.6,14 2.6,13.6 .9,12.2 .4,10.9')
# 4 -- open uprights, a continuous U with descending second stem, and
# closed angular or rounded triangular forms. Keep the actual pen lifts.
style(4,5,'2.8,0 1.5,3.8 !.5,7.8 4.5,8 9.8,7.4 / 7.8,.5 7.3,5.5 7.2,10 7.1,14')
style(4,3,'2.4,0 1.1,3.8 .3,6.8 1.1,7.7 3.6,8 6.4,7.4 8,6.4 / 8.8,1.2 8,6.5 7.5,10.3 7.4,14')
style(4,3,'1,.1 .3,3.8 .1,6.7 1,8.1 3.4,8.1 6,7.1 !8.1,5 !8.9,1 8.4,5.4 7.9,9.6 7.4,14')
style(4,2,'2.6,.2 1.2,3.6 !0,7.3 4.1,7.6 7.2,7 !9,6.2 / 8,2.3 7.6,7 7.4,10.5 8,14')
style(4,2,'4.4,0 2.6,3.8 !0,8.8 4,8.5 9.6,8.1 / 7.4,.6 7.4,5.8 7.3,10 7.6,14')
style(4,2,'7.4,0 4.1,3.8 !0,8.6 4.3,8.4 9.8,8.4 / 7.4,0 7.4,6.3 7.3,10 7.3,14')
style(4,1.3,'7.5,14 7.1,8.5 !7.4,0 4,3.9 !.1,8.8 4.9,8.3 9.2,8')
style(4,1,'6.3,.9 3.3,3.8 1.3,6.1 .7,7.5 1.4,8.1 4.5,8.1 8.1,7.6 / 7.4,0 7.2,5.6 7,9.6 7.5,14')
style(4,1,'1.2,.5 .3,3.6 .3,6 1.9,7.9 4.4,8.1 7.4,7.2 / 8.7,.4 8,4.6 7.6,9.1 6.9,14')
# 5 -- separate or connected cap, vertical/oblique back, lower bowl.
style(5,5,'9.6,.2 6.2,.5 !2,0 1.6,3.3 !.9,6.7 3.1,5.4 5.6,5.3 8.1,6.7 9.1,9.3 8.5,11.7 6.5,13.5 3.6,14 1,13 0,11.9')
style(5,4,'1.8,.4 1.4,3.4 !1,6.2 3.3,5.9 5.9,6.2 8.4,7.7 9.3,9.8 9,11.7 7,13.4 4.1,14 1.5,13.7 .1,12.7 / 1.3,1 5.3,.4 9.4,0')
style(5,3,'9,0 5.9,.4 !2.5,.8 1.3,3.8 !.4,6.3 2.7,5.7 5.5,5.6 8.1,7.1 9.3,9.5 8.4,12.4 5.9,14 2.4,13.7 0,12')
style(5,2,'9.9,.1 7,.9 3.7,1 !1.5,.7 1.2,3.7 1.2,5.3 2.1,6.3 4.6,6.9 7.5,8.3 8.7,10.5 7.8,12.7 5.3,14 2.6,13.7 .4,12.3')
style(5,2,'1.3,.2 .5,2.7 .3,4.9 1.3,6.2 4.1,7.1 7.2,8.1 9,10.2 8.6,12.1 6.2,13.8 3.3,14 .5,12.6 / .8,1 4.1,1 7.2,.3 9.8,0')
style(5,1.4,'9,1 7.2,.3 4.6,0 2.5,.4 .8,2 .5,3.9 1.6,5.5 4,6.7 6.5,7.6 8.6,9.2 9,11 7.7,12.9 5.1,14 2.3,13.6 .2,12.3')
style(5,1,'2.1,1 1.2,3.7 !.7,5.5 3.9,5.2 6.7,5.8 8.8,7.6 9,10.1 7.6,12.6 5.1,14 2.4,13.7 .5,12 / 2.8,.3 6,.5 9.8,0')
style(5,1,'9.6,0 6.6,1.1 !3.2,1.1 1.7,3.5 !.3,6.4 3.5,6.6 6.9,7.5 8.9,9.7 8.6,11.8 6.5,13.5 3.3,14 .1,13.3')
style(5,.7,'8.5,0 5.2,.5 !2.3,.8 .9,3.3 !.1,5.5 2.9,5.3 5,5.9 7,7.4 7.8,9.7 6.9,11.8 4.2,13.5 1.6,14 .3,13.4 .9,12.1 2.6,11.3')
# 6 -- rising hook optional; varying amounts of closure and overshoot.
style(6,5,'7.9,0 5.3,1.8 2.9,4.8 1,8 .6,10.5 1.6,12.9 4,14 6.9,13.6 8.9,11.9 9.3,9.6 8.2,7.8 5.9,7.1 3.4,7.6 1.4,9.3')
style(6,3,'7.1,0 4.9,2.1 2.6,5.8 1.3,9.1 1.3,11.8 2.8,13.6 5.3,14 7.7,12.7 9.2,10.5 9.1,8.5 7.6,7.4 5.7,7.5 3.5,8.9 1.8,10.4')
style(6,2.5,'6,0 4,2.3 2.1,6.1 1.1,10.2 1.8,12.9 4.1,14 7,13.4 9.1,11.2 9,9.3 7.2,8.5 4.9,8.8 2.4,10.3')
style(6,2,'9.2,1.8 8,.5 6.2,0 4.1,1.6 2.2,4.9 .8,8.9 1.1,11.6 2.8,13.6 5.5,14 8.3,12.4 9.3,10 8.8,7.8 7,6.6 4.8,6.5 2.5,7.8 1.3,9.3')
style(6,2,'6.6,0 4.1,2.8 2,6.6 .6,10.9 1.1,13.1 3.1,14 6.1,13.4 8.4,11.8 9.4,9.6 8.7,8.1 7,7.7 5.2,8.6 3.6,10')
style(6,1.5,'7.1,0 4.8,2.9 3.2,6.4 2.2,9.8 2.3,12.3 3.9,13.9 6.6,14 8.8,12.5 9.2,10.3 8.3,8.4 6.3,7.7 4.3,8.3 2.4,9.9 1.2,11.1')
style(6,1,'6.8,0 4.4,2.3 2.4,5.4 1.3,8.4 1.5,11.4 3.3,13.5 5.7,14 8.1,12.9 9.2,10.6 8.4,8.8 6.5,8.2 4.9,8.7')
style(6,.8,'6.1,0 4.1,2.2 2.7,5.1 1.7,8.3 1.8,11.2 3.1,13.3 5,14 7.4,13.7 9.2,12.1 9.4,10.1 8.4,8.5 6.4,7.2 4.7,6 4.1,5.3')
# 7 -- top cap to a descending straight or curving oblique; optional flag/crossbar.
style(7,5,'0,.5 3.4,.2 6.8,.1 !10,0 8.6,3.2 6.6,7.6 4.8,11.6 3.6,14')
style(7,4,'0,.3 2.6,0 5.8,.7 !9.6,1.1 7.7,4.3 5.8,8.6 4.4,12.3 3.8,14')
style(7,3,'0,1.5 3.4,.7 6.9,0 9.3,.2 !9.8,1.2 8.4,4.6 6.4,8.4 4.5,11.9 3.4,14')
style(7,2,'0,2.8 !.6,.3 3.4,.2 6.3,.6 !9.8,.2 8,4.3 6.1,8.2 4.4,12 3.9,14')
style(7,2,'0,0 3.2,.5 6.8,1 !9.8,.5 7.8,4.9 5.9,9.4 4.5,14 / 2.5,7.3 5.8,7.6 8.8,7.4')
style(7,1.5,'0,2.8 !.5,.3 3.5,.5 6.1,.6 !9.2,.1 7.1,5.3 5.5,9.1 4.4,14 / 3.1,7.2 5.7,7 8.7,7.2')
style(7,1.4,'0,1.2 1.6,0 3.8,.4 6.5,1.8 !9.5,1.4 8.1,4.5 6.7,8.1 5.6,11.8 5.2,14')
style(7,1,'0,0 3.9,.5 7.5,.6 !9.8,.2 8.6,3.3 7.6,7.2 7.1,10.8 7.3,14')
style(7,1,'0,1.1 2.7,.3 5.7,0 8.3,.2 !9,.9 8.6,3.5 7,7.1 4.2,11.3 2.5,14')
# 8 -- two crossing pen arcs with alternative waist positions and open knots.
style(8,5,'5.6,6.7 2.7,5.4 1.4,3.6 1.8,1.5 4.1,.2 6.7,0 8.6,1.2 8.8,3.2 7.5,5.2 5,7.1 2,9.2 .8,11.3 1.4,13.2 3.8,14 6.6,13.8 8.8,12.5 9,10.5 7.6,8.6 5.3,6.9')
style(8,3,'5.1,.1 2.4,.7 1.3,2.6 2.1,4.6 4.4,6.2 7.2,7.9 8.7,10 8.7,12 7.1,13.5 4.7,14 2.2,13.5 1,12 1.3,10.2 3,8.3 5.7,6.4 8,4.5 8.8,2.5 8,1 5.9,0 4.4,.1')
style(8,2,'5.8,1.1 4.2,.3 2.2,.7 1.1,2.4 1.8,4.5 4.1,6.4 6.7,8.4 8.6,10.7 8.4,12.7 6.1,14 3.8,13.9 1.8,12.8 1.2,11 2.4,8.8 5,6.5 7.6,4.8 9.2,2.7 9.1,1.1 7.3,.1 5.3,0 3.9,.8')
style(8,2,'6.6,0 4.1,.3 1.6,1.9 1.4,3.6 2.8,5.4 5.2,7 7.8,9 8.9,11.2 8.3,13.1 6.2,14 3.5,13.8 1.6,12.5 1.2,10.6 2.9,8.6 5.3,6.7 7.4,4.9 8.9,2.7 8.6,1.1 7.6,.5')
style(8,1.5,'4.7,6 2.1,5.2 1.4,3.1 2.1,1.2 4.8,0 7.1,.4 8.6,2 8.1,4.1 6.1,5.6 3.8,6.6 1.5,8.2 .7,10.4 1.5,12.7 3.7,14 6.6,13.9 8.7,12.4 9.1,10.1 8.3,8 6.5,6.6 4.5,6')
style(8,1.5,'4.8,7.8 2.2,6.4 1,4.2 1.1,2 2.9,.4 5.4,0 7.9,1.1 8.9,3 8.3,5.1 6.2,7 3.6,8.5 1.8,10.3 1.6,12.2 3.1,13.7 5.4,14 7.8,13.3 8.8,11.5 8,9.5 6.2,8.4 4.7,7.8')
style(8,1,'5.7,0 3.5,.1 1.6,1.5 1.5,3.5 3.1,5.4 5.3,7.4 7.7,9.8 8.4,12 7.3,13.6 5.1,14 2.7,13.5 1.6,11.9 2.1,9.8 3.7,7.9 5.8,5.8 7.9,3.3 8.1,1.7')
style(8,.8,'4.7,7.1 2.4,6.2 1.5,4.4 1.9,2.1 4.2,.4 6.8,0 8.5,1.2 8.7,3.1 7.4,5.2 5.6,7.2 3.3,9.2 1.9,11.3 2.4,13 4.4,14 6.7,13.6 8.1,12.3 8.1,10.3 6.8,8.2')
# 9 -- loop with vertical, diagonal, hooked, or cursive descending stem.
style(9,5,'7.9,2.4 7.1,1 5.4,0 3.1,.3 1.4,1.9 .8,4.4 1.5,6.5 3.8,7.4 6,6.4 7.6,4.2 !8.6,1.3 8.1,5.5 7.6,9.1 7,14')
style(9,3,'8.1,1.5 6.4,.1 3.9,.1 1.5,1.8 .7,4.2 1.2,6.4 3.1,7.6 5.4,7.1 7.5,5 8.2,2.8 8.1,1.1 / 8,.8 8.3,5 8.6,9 8.7,14')
style(9,2.5,'8.4,2.5 7.1,.7 5.3,0 2.8,.5 1.1,2.2 .7,4.2 1.8,6 4.1,6.6 6.7,5.4 8.4,3.5 !9.4,.7 8.2,5.3 6.7,9.9 5.1,14')
style(9,2,'8.3,2 6.8,.4 4.6,0 2.3,1 1,3.1 1.1,5.3 2.9,6.9 5,7.1 7,5.9 8.3,3.5 8.1,1.6 / 8.4,1 8.3,4.8 8.7,8.5 9.7,12 10.9,14')
style(9,2,'8.5,2.4 7.5,.5 5.5,0 3,.4 1.2,2.1 .6,4.3 1.4,6.3 3.6,7 5.9,6.2 7.8,4.4 8.5,2.1 8.4,5.4 7.5,9.4 6.2,12.6 4.6,14')
style(9,1.5,'8.1,1.1 6.2,0 3.9,.1 1.8,1.4 .8,3.7 1.1,6.2 2.9,7.5 5.2,7.2 7.4,5.3 8.4,2.9 !8.4,1 8.9,5 9,8.8 8.4,11.7 6.6,13.6 4.6,14 2.6,13')
style(9,1.5,'8,.9 6.3,0 4.1,.4 2,2.1 .8,4.7 .8,6.7 2.2,7.7 4.4,7.2 6.8,5.2 !8.8,1.5 8.2,5.5 7.9,9.8 8.2,14')
style(9,1,'6.5,.6 5.1,0 3.3,.3 1.5,2.1 .9,4.1 1.2,6.1 3.1,7.1 5.6,6.3 7.5,4.5 !8.9,1.9 8.4,6.4 7.7,10.5 7.3,14')
style(9,1,'8.1,1.7 6.3,.2 3.8,0 1.5,1.7 .9,3.6 1.5,5.6 3.5,6.5 5.8,6 7.8,4.5 8.6,2.3 / 8.4,.7 9,4.4 8.6,8.8 7.1,12 5,13.9 3.3,14')

# Interpolate smooth cubic segments through the knots, with one-sided
# endpoint tangents and straight entry/exit tangents at hard corners.
def parse(path):
    result=[]
    for stroke in path.split('/'):
        points=[]; corners=[]
        for tok in stroke.split():
            corners.append(tok.startswith('!'))
            points.append([float(z) for z in tok.lstrip('!').split(',')])
        p=np.array(points)
        dif=np.diff(p,axis=0)
        leng=np.sqrt((dif*dif).sum(axis=1)).clip(.001)
        tang=dif/leng[:,None]
        t=np.vstack([tang[0],(tang[:-1]+tang[1:])*.5,tang[-1]])
        allp=[]
        for j,L in enumerate(leng):
            a=(tang[j] if corners[j] else t[j])*L
            b=(tang[j] if corners[j+1] else t[j+1])*L
            u=np.linspace(0,1,max(3,int(np.ceil(L*5))),endpoint=False)[:,None]
            seg=p[j]*(1-3*u*u+2*u**3)+p[j+1]*(3*u*u-2*u**3)+a*(u-2*u*u+u**3)+b*(-u*u+u**3)
            allp.append(seg)
        result.append(np.concatenate(allp+[p[-1:]]))
    return result

# Stateless templates apart from the seeded pseudorandom draws. No choices
# made by comparing renders: every scheduled instance becomes one template.
def render(strokes,rng,canonical_style):
    # Smooth geometric perturbations, not independent shaking of the knots:
    # successive passes of the pen around a loop still meet.
    f = .15 if canonical_style else 1.
    z=lambda : float(np.clip(rng.normal(),-2.3,2.3))*f
    ax=np.exp(.22*z()); ay=np.exp(.07*z())
    sh=-.12+.29*z(); tilt=.09*z()
    xb=.072*z(); yb=.072*z(); wtop=.15*z(); xt=.064*z()
    yside=.055*z(); yshort=.06*z(); xc=.045*z()
    # Slender/wide loops and off-centre waists, with independent deformation
    # at the top, waist and bottom, and modest vertical bow of cross-strokes.
    coef=rng.normal(size=(2,3,2)).clip(-2,2)*f
    scales=.38 + rng.uniform()*.35
    width=np.exp(.30*z())*1.63
    widths=[]; pp=[]
    for p in strokes:
        x=(p[:,0]-5)/10; y=(p[:,1]-7)/14
        b1=np.cos(np.pi*y); b2=np.sin(2*np.pi*y)
        qx=(x*(1+wtop*2*y)+xb*b1+xt*b2+xc*(4*x*x-.5))*10
        qy=(y+yb*b1+yside*(1-4*x*x)+yshort*np.sin(2*np.pi*x))*14
        for h,cy in enumerate((-.5,0,.5)):
            k=np.exp(-.5*((y-cy)/.24)**2)
            qx+=k*(coef[0,h,0]+coef[0,h,1]*x*2)*.63
            qy+=k*(coef[1,h,0]+coef[1,h,1]*x*2)*.54
        if len(strokes)>1:
            qx+=.39*z(); qy+=.45*z()
        Y=ay*qy+tilt*qx
        X=ax*qx+sh*Y
        pp.append(np.stack([X,Y],axis=1))
        # Pressure variation and pen taper at endpoints; avoid a uniform tube.
        u=np.linspace(0,1,len(p)); phase=rng.uniform(-np.pi,np.pi)
        local=1+.11*z()*np.sin(2*np.pi*u+phase)+.08*z()*np.cos(np.pi*u)
        local*=1-rng.uniform(0,.36)*np.exp(-u/.055)-rng.uniform(0,.32)*np.exp(-(1-u)/.055)
        widths.append(width*local.clip(.45,1.65))
    p=np.concatenate(pp); thick=np.concatenate(widths)
    scl=20/max((p+thick[:,None]/2).max(0)-(p-thick[:,None]/2).min(0))
    p=p*scl; rad=thick*scl*.5
    p=p-(p.max(0)+p.min(0))/2+13.5+rng.uniform(-.48,.48,2)
    # Render round pen footprints with a soft pixel aperture. Only a small
    # neighbourhood of each sample is needed. Min signed distance to ink.
    pad=int(np.ceil(rad.max()+scales))+1
    off=np.arange(-pad,pad+1)
    yy,xx=np.meshgrid(off,off,indexing='ij')
    ix=np.floor(p[:,0]).astype(int)[:,None]+xx.ravel()[None,:]
    iy=np.floor(p[:,1]).astype(int)[:,None]+yy.ravel()[None,:]
    valid=(ix>=0)&(iy>=0)&(ix<28)&(iy<28)
    sd=np.sqrt((ix-p[:,0,None])**2+(iy-p[:,1,None])**2)-rad[:,None]
    dist=np.full(784,15,np.float32)
    np.minimum.at(dist,(iy*28+ix)[valid],sd[valid])
    ink=np.clip(.5-dist.reshape(28,28)/(2*scales),0,1)
    return ink*ink*(3-2*ink)


def build(target):
    target.mkdir(exist_ok=True,parents=True)
    (target/'solution.py').write_text(features+recognizer)
    context={}
    exec(features,context)
    describe=context['describe']
    rng=np.random.default_rng(617091)
    gl=[]; pa=[]; ho=[]; ge=[]; co=[]; cuts=[0]
    for digit in range(10):
        imgs=[]
        for weight,path in styles[digit]:
            strokes=parse(path)
            count=round(150*weight**.65)
            for j in range(count):
                imgs.append(render(strokes,rng,j==0))
                co.append(.004*max(0,np.log(2./weight)))
        for j in range(0,len(imgs),48):
            g,p,h,s=describe(np.array(imgs[j:j+48],dtype=np.float32))
            # Byte encoding of unit-vector components, only for storage.
            gl.append(np.uint8(np.rint(np.minimum(g,1)*255)))
            pa.append(np.uint8(np.rint(np.minimum(p[:,0],1)*255)))
            ho.append(h); ge.append(s)
        cuts.append(len(co))
    np.savez(target/'constants.npz',**{'global':np.concatenate(gl),'patch':np.concatenate(pa),'hole':np.concatenate(ho),'geo':np.concatenate(ge),'cost':np.array(co,np.float32),'cuts':np.array(cuts,np.int32)})

if __name__=='__main__':
    build(Path(sys.argv[1]))
