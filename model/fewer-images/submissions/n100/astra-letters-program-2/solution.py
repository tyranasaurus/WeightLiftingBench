import numpy as np


def neighbor(a, d, axis):
    n = a.shape[axis]
    return np.take(a, np.clip(np.arange(n) + d, 0, n-1), axis=axis)


def smooth(a):
    for axis in (1, 2):
        a = (neighbor(a, -1, axis) + 2*a + neighbor(a, 1, axis)) * .25
    return a


def normrows(a):
    a = a.reshape(len(a), -1)
    return a / np.sqrt((a*a).sum(axis=1, keepdims=True) + 1e-10)


def describe(a):
    # Unsigned gradients: the hand/background contrast can change sign.
    a = smooth(a)
    gx = (neighbor(a, 1, 2)-neighbor(a, -1, 2)) * .5
    gy = (neighbor(a, 1, 1)-neighbor(a, -1, 1)) * .5
    mag = np.sqrt(gx*gx + gy*gy + 1e-12)
    n = len(a)
    # A smooth illumination ramp is not a contour. Remove that background
    # response before any compression, and cap strong clutter edges locally.
    rms = np.sqrt(np.mean(mag*mag, axis=(1,2), keepdims=True) + 1e-8)
    residual = np.maximum(0, mag / rms - .27)
    weight = np.minimum(residual, 2.8)
    field = []
    for k in range(6):
        t = k * (np.pi/6)
        proj = np.abs((np.cos(t)*gx + np.sin(t)*gy) / mag)
        field.append((weight * np.maximum(0, 1 - (1-proj)*7.5)).astype(np.float32))
    field = np.stack(field, axis=3)
    # Four-pixel cells, softly overlap their immediate neighbors.
    f = smooth(field).reshape(n,7,4,7,4,6).mean(axis=(2,4))
    density = f.sum(axis=-1, keepdims=True)
    loc = .22 + smooth(density)
    hog = f / np.sqrt(loc)
    # Separately preserve orientation selectivity and occupancy. Uniform dirt
    # producing energy in every orientation should not resemble a finger.
    selective = hog - hog.mean(axis=3, keepdims=True)
    def center(v): return v-v.mean(axis=(1,2),keepdims=True)
    edge = normrows(center(hog))
    occ = normrows(center(density / np.sqrt(loc)))
    orient = normrows(selective)
    desc = np.concatenate([edge*.70, occ*.55, orient*.45], axis=1)
    # Piecewise smooth hand-surface shading, without a broad illumination ramp.
    broad = a
    for stride in (1, 2, 3):
        for axis in (1, 2):
            broad = (neighbor(broad,-stride,axis)+2*broad+neighbor(broad,stride,axis))*.25
    hp = a-broad
    texture = np.maximum(-2.5, np.minimum(2.5, hp/(2*rms+.006)))
    texture = smooth(texture).reshape(n,14,2,14,2).mean(axis=(2,4))
    signed = normrows(texture)
    # Difference of local averages responds to the two flanks of a narrow digit,
    # even if one of those flanks has almost the same brightness as the wall.
    fine=[]
    for dy, dx in ((0,2),(2,0),(1,1),(1,-1)):
        ridge=np.abs(a-.5*(shifted(a,dy,dx)+shifted(a,-dy,-dx))) / (rms+.004)
        ridge=np.clip(ridge-.22,0,2.5)
        fine.append(smooth(ridge).reshape(n,14,2,14,2).mean(axis=(2,4)))
    bar=normrows(center(np.stack(fine,axis=3)))
    # Hypothesize a backdrop varying smoothly between the two side margins.
    # If a margin contains a strong edge, favor the less textured margin.
    def border(v):
        for stride in (1,2,3):
            v=(neighbor(v,-stride,1)+2*v+neighbor(v,stride,1))*.25
        return v
    left=border(np.median(a[:,:,:3],axis=2,keepdims=True))
    right=border(np.median(a[:,:,-3:],axis=2,keepdims=True))
    ltex=border(mag[:,:,:3].mean(axis=2,keepdims=True))
    rtex=border(mag[:,:,-3:].mean(axis=2,keepdims=True))
    pos=np.linspace(.06,.94,28,dtype=np.float32)[None,None,:]
    wl=(1-pos)/(ltex/rms+.24)**2
    wr=pos/(rtex/rms+.24)**2
    wall=(wl*left+wr*right)/(wl+wr)
    mat=np.minimum(1,np.abs(a-wall)/(2.4*rms+.015))
    coverage=smooth(mat).reshape(n,14,2,14,2).mean(axis=(2,4))
    matte=normrows(center(coverage))
    desc=np.concatenate([desc*.70, signed*.37, bar*.36, matte*.47],axis=1)
    # Emphasize the articulated hand, not the sleeve, and retain a separate
    # orientation signature restricted to the finger region. Weighting after
    # centering does not introduce an outside brightness reference.
    def focus(h,w):
        y,x=np.mgrid[0:h,0:w]
        y=(y+.5)/h; x=(x+.5)/w
        return (.22+.78*np.exp(-.5*((y-.33)/.27)**2-.5*((x-.50)/.40)**2)).astype(np.float32)
    fmask=focus(7,7)[None,:,:,None]
    upper=normrows(fmask*center(hog))
    dmask=focus(14,14)[None,:,:,None]
    upperridge=normrows(dmask*center(np.stack(fine,axis=3)))
    desc=np.concatenate([desc*.83, upper*.39, upperridge*.39],axis=1)
    # Integer fixed point lets every dot product be exactly representable in float32.
    return np.rint(desc*1536).astype(np.int16)


def shifted(a, y, x):
    return neighbor(neighbor(a, y, 1), x, 2)


def predict(x):
    bank = CONSTANTS['shape'].astype(np.float32)
    coarse=np.ascontiguousarray(np.concatenate([bank[:,:833],bank[:,1617:2107]],axis=1))
    nmodel = bank.shape[0] // 24
    out = np.zeros(len(x), dtype=np.int64)
    for start in range(0, len(x), 40):
        a = x[start:start+40,0]
        views = []
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                views.append(describe(shifted(a, dy, dx)))
        q = np.stack(views, axis=1).reshape(-1,bank.shape[1]).astype(np.float32)
        # First align the entire hand rigidly. Use the inexpensive pooled
        # gradients, matte and shading to shortlist individual render/shift
        # pairs; only those need the full-resolution comparison.
        cols=list(range(833))+list(range(1617,2107))
        matrix=q[:,cols]@coarse.T
        allscore=matrix.reshape(len(a),9,24,nmodel).transpose(0,2,1,3).reshape(len(a),24,-1)
        nshort=48
        ids=np.argpartition(allscore,9*nmodel-nshort,axis=2)[:,:,-nshort:]
        tiny=np.array([[0,0],[-1,0],[1,0],[0,-1],[0,1]])
        for j in range(len(a)):
            templates=ids[j]%nmodel+np.arange(24)[:,None]*nmodel
            aligns=ids[j]//nmodel
            row=q[j*9:j*9+9]
            resc=(row[aligns]*bank[templates]).sum(axis=2)
            which=np.argpartition(resc,nshort-6,axis=1)[:,-6:]
            letter=np.arange(24)[:,None]
            indices=templates[letter,which].ravel()
            bestv=aligns[letter,which].ravel()
            globalmatch=resc[letter,which].ravel()/(1536.*1536.)
            def patchscore(first,last,side,channels,off):
                # Soft quadrants overlap at the palm. Moving one quadrant by
                # one fine cell allows small changes in digit length/spread.
                t=bank[indices,first:last].reshape(-1,side,side,channels)
                image=q[9*j:9*j+9,first:last].reshape(9,side,side,channels)
                h=(side+1)//2+1
                centergrid=np.arange(side)
                total=np.zeros(len(t))
                for d in (0,side-h):
                    for e in (0,side-h):
                        sub=t[:,d:d+h,e:e+h]
                        best=np.full(len(t),-1.)
                        for dy,dx in tiny:
                            yy=np.clip(centergrid[d:d+h]+dy,0,side-1)
                            xx=np.clip(centergrid[e:e+h]+dx,0,side-1)
                            match=image[:,yy[:,None],xx[None,:]][bestv]
                            dot=(sub*match).sum(axis=(1,2,3))
                            den=np.sqrt((sub*sub).sum(axis=(1,2,3))*(match*match).sum(axis=(1,2,3))+1.)
                            best=np.maximum(best,dot/den-off*(abs(dy)+abs(dx)))
                        total+=best*.25
                return total
            # Fine-scale second derivative and shading maps share a 14x14 grid.
            micro=patchscore(833,1617,14,4,.045)
            light=patchscore(637,833,14,1,.05)
            local=.57*micro+.43*light
            sc=(globalmatch*.73+local*.27).reshape(24,-1)
            out[start+j]=(.70*sc.max(axis=1)+.30*sc.mean(axis=1)).argmax()
    return out
