import numpy as np


def smooth(a, axis):
    v = np.swapaxes(a, axis, -1)
    left = np.concatenate((v[..., :1], v[..., :-1]), axis=-1)
    right = np.concatenate((v[..., 1:], v[..., -1:]), axis=-1)
    return np.swapaxes((left + 2 * v + right) * .25, axis, -1)


def blur(a):
    return smooth(smooth(a, -1), -2)


def diff(a, axis):
    v = np.swapaxes(a, axis, -1)
    d = np.concatenate((v[..., 1:], v[..., -1:]), -1) - np.concatenate((v[..., :1], v[..., :-1]), -1)
    return np.swapaxes(d, axis, -1)


def poolmu(count, sigma):
    p = (np.arange(count) + .5) * 28 / count - .5
    w = np.exp(-.5 * ((p[:, None] - np.arange(28)) / sigma) ** 2)
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


wide = poolmu(7, 1.85)
fine = poolmu(14, 1.12)


def pool(a, mat):
    return (mat @ a) @ mat.T


def unit(a):
    a = a.reshape(len(a), -1)
    return a / np.sqrt(np.sum(a*a, 1, keepdims=True) + 1e-9)


def fgmap(a):
    # The surrounding background often forms an almost planar illumination
    # field. Estimate it in narrow border patches, allowing a hand/obstacle
    # to obscure one edge. No parameters are shared between photographs.
    xx = np.linspace(0, 1, 28, dtype=np.float32)[None,None,:]
    yy = xx.transpose(0,2,1)
    v = np.stack((a[:,:4,:4].mean((1,2)), a[:,:4,-4:].mean((1,2)),
                  a[:,-4:,:4].mean((1,2)), a[:,-4:,-4:].mean((1,2))), axis=1)
    grad = blur(np.abs(diff(a,1)) + np.abs(diff(a,2)))
    anchor = np.stack((grad[:,:4,:4].mean((1,2)),grad[:,:4,-4:].mean((1,2)),
                       grad[:,-4:,:4].mean((1,2)),grad[:,-4:,-4:].mean((1,2))), axis=1)
    # Extrapolate a plane from each triplet of corners. Use a robust
    # agreement test on the rim: a bad corner should not spoil the mask.
    hs = []
    es = []
    for k in range(4):
        c = v.copy()
        c[:, k] = c[:, k ^ 1] + c[:, k ^ 2] - c[:, k ^ 3]
        c = c[:,:,None,None]
        h = c[:,0] + (xx*27 - 1.5)/24 * (c[:,1] - c[:,0]) + (yy*27 - 1.5)/24 * (c[:,2] - c[:,0])
        d = np.abs(a - h)
        rim = np.concatenate((d[:,:2,:].reshape(len(a),-1),d[:,:,0],d[:,:,-1]),axis=1)
        es.append(np.mean(np.minimum(rim, .12), axis=1) + .30*(np.sum(anchor,1)-anchor[:,k]))
        hs.append(h)
    e = np.stack(es,1)
    w = np.exp(-(e - e.min(1,keepdims=True)) / .018)
    w /= w.sum(1,keepdims=True)
    h = sum(w[:,k,None,None] * hs[k] for k in range(4))
    residual = np.abs(a-h)
    # Include texture as evidence for the hand even at a brightness match.
    text = grad
    for j in range(3):
        text = blur(text)
    m = np.maximum(residual - .014, 0) + .70 * np.maximum(text - .026,0)
    photometric = m / (m + .069)
    # A second cue needs no assumption about the color of the background:
    # a path from the frame to the palm has to cross an occluding contour.
    # Propagate the least upper bound on edge strength along such paths.
    # Leave the middle of the bottom edge unseeded (the wrist crosses it).
    barrier = .58*grad + .42*(np.abs(diff(a,1))+np.abs(diff(a,2)))
    flood = np.full_like(a,2)
    flood[:,0,:] = barrier[:,0,:]
    flood[:,-1,:5] = barrier[:,-1,:5]
    flood[:,-1,-5:] = barrier[:,-1,-5:]
    flood[:,:,0] = barrier[:,:,0]
    flood[:,:,-1] = barrier[:,:,-1]
    for j in range(40):
        v = flood.copy()
        v[:,1:] = np.minimum(v[:,1:],flood[:,:-1])
        v[:,:-1] = np.minimum(v[:,:-1],flood[:,1:])
        v[:,:,1:] = np.minimum(v[:,:,1:],flood[:,:,:-1])
        v[:,:,:-1] = np.minimum(v[:,:,:-1],flood[:,:,1:])
        flood = np.maximum(v,barrier)
    trapped = np.maximum(blur(flood) - .018,0)
    return .55*photometric + .45*trapped/(trapped+.065)


def channels(pic, mask):
    a = blur(pic)
    gx = diff(a,2)
    gy = diff(a,1)
    mag = np.sqrt(gx*gx + gy*gy + 1e-10)
    ca, sa = gx / mag, gy / mag
    strength = mag / (mag + .075)
    # Separate true contrast changes from small texture/noise gradients.
    strength *= np.sqrt(mag / (mag + .045))
    maps = []
    for k in range(6):
        angle = k * np.pi / 6
        c = np.abs(ca * np.cos(angle) + sa * np.sin(angle))
        maps.append((strength * np.maximum(c*c - .54, 0)).astype(np.float32))
    if mask is None:
        mask = fgmap(a)
    # Signed local contrast adds the bright phalanges and dark valleys that
    # are invisible to a silhouette (notably in the closed handshapes).
    surround = a
    for j in range(10):
        surround = blur(surround)
    relief = a - surround
    maps.append((mask * relief / (np.abs(relief) + .060)).astype(np.float32))
    return np.stack(maps,1), mask.astype(np.float32)


def descriptor(edges, mask, mat):
    pooled = pool(edges, mat)
    ed = pooled[:,:6]
    relief = pooled[:,6]
    # Remove the common isotropic response but keep some edge mass.
    ed = ed - .62 * ed.mean(1,keepdims=True)
    pos = ed.shape[-1]
    row = np.arange(pos,dtype=np.float32)
    weight = (1 - .32 * (row / pos)**2)[None,None,:,None]
    ed = ed * weight
    f = pool(mask, mat)
    f = f - f.mean((1,2),keepdims=True)
    vec = np.concatenate((unit(ed) * .815, unit(f * weight[:,0]) * .513, unit(relief * weight[:,0]) * .270),axis=1)
    return vec.astype(np.float32)


coarsebank = np.ascontiguousarray(CONSTANTS['coarse'].astype(np.float32) / 512)
finebank = CONSTANTS['fine']
variants = CONSTANTS['coarse'].shape[0] // 24
picks = 16
stride = 16


def block(x):
    # A fixed row count gives the matrix products identical reduction order
    # regardless of the number or order of other pictures in the call.
    edge, mask = channels(x, None)
    coarse = descriptor(edge, mask, wide)
    finevec = descriptor(edge, mask, fine)
    s = (coarse @ coarsebank.T).reshape(stride,24,variants)
    best = np.argpartition(s, variants-picks, axis=2)[:,:,-picks:]
    out = np.empty((stride,24), dtype=np.float32)
    labels = np.arange(24)[:,None] * variants
    for j in range(stride):
        idx = labels + best[j]
        f = finebank[idx].astype(np.float32)
        t = np.sum(f * finevec[j,None,None,:],axis=2) / 512
        z = .70*t + .30*np.take_along_axis(s[j],best[j],axis=1)
        # A max with a small near-neighbour vote dampens isolated accidents.
        z.sort(axis=1)
        out[j] = .58 * z[:,-1] + .28*z[:,-2] + .14*z[:,-3]
    return np.argmax(out,1)


def predict(x):
    n = len(x)
    y = np.empty(n, dtype=np.int64)
    for start in range(0,n,stride):
        chunk = x[start:start+stride,0]
        a = np.empty((stride,28,28),dtype=np.float32)
        a[:] = chunk[0]
        a[:len(chunk)] = chunk
        y[start:start+len(chunk)] = block(a)[:len(chunk)]
    return y
