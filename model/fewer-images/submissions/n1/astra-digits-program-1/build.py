"""Build a handwritten stroke grammar and an elastic image matcher, without data."""
import sys
from pathlib import Path
import numpy as np

# Shared by the template builder and the inference program. All measurements here
# are local to ONE image; no calibration of any kind is done on the library.
common = '''import numpy as np

nside = 28
a = np.arange(nside, dtype=np.float32)
xx = a[None, :]
yy = a[:, None]

def edges(v):
    c = np.cumsum(v, axis=1)
    total = c[:, -1:]
    q = total * .008
    ans = []
    for target in (q, total-q):
        i = np.minimum(np.sum(c < target, axis=1), 27)
        n = np.arange(len(v))
        f = (c[n, i]-target[:,0]) / np.maximum(v[n,i], 1.e-7)
        ans.append(i+.5-np.clip(f, 0, 1))
    return ans

def sample(im, x, y):
    x, y = np.broadcast_arrays(x, y)
    valid = (x > -1) & (x < 28) & (y > -1) & (y < 28)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = (x-x0).astype(np.float32), (y-y0).astype(np.float32)
    n = np.arange(len(im))[:,None,None]
    l, r = np.clip(x0,0,27), np.clip(x0+1,0,27)
    t, b = np.clip(y0,0,27), np.clip(y0+1,0,27)
    return valid*((1-fy)*((1-fx)*im[n,t,l]+fx*im[n,t,r])+fy*((1-fx)*im[n,b,l]+fx*im[n,b,r]))

def normalize(im):
    mass = np.maximum(im.sum(axis=(1,2)), 1.e-5)
    row, col = im.sum(axis=2), im.sum(axis=1)
    mx, my = (col*a).sum(axis=1)/mass, (row*a).sum(axis=1)/mass
    dy = yy-my[:,None,None]
    cov = (im*dy*(xx-mx[:,None,None])).sum(axis=(1,2))
    v = (im*dy*dy).sum(axis=(1,2))
    s = (.93*np.clip(cov/np.maximum(v,1.e-5),-1.25,1.25))[:,None,None]
    upright = sample(im,xx+s*dy, np.broadcast_to(yy, im.shape))
    l,r = edges(upright.sum(axis=1))
    t,b = edges(row)
    w,h = np.maximum(1.,r-l),np.maximum(1.,b-t)
    rat = np.clip(w/h,.08,3.)
    # Partial rather than full aspect normalization: a bare '1' must stay narrow.
    sx = w/(17*np.clip((rat/.73)**.38,.44,1.28))
    sy = h/(21*np.clip((1.35/np.maximum(rat,1.35))**.45,.68,1.))
    y = ((t+b)/2)[:,None,None]+(yy-13.5)*sy[:,None,None]
    x = ((l+r)/2)[:,None,None]+(xx-13.5)*sx[:,None,None]+s*(y-my[:,None,None])
    canon = sample(im,x,y)
    rawl,rawr = edges(col)
    rawratio = np.clip((rawr-rawl)/h,.05,3.)
    return canon, np.log(rat).astype(np.float32), s[:,0,0], np.log(rawratio).astype(np.float32)

def spread(x, radius, sig):
    k = np.exp(-.5*(np.arange(-radius,radius+1)/sig)**2).astype(np.float32)
    k = k/k.sum()
    n,h,w = x.shape[:3]
    pad = np.zeros((n,h+2*radius,w)+x.shape[3:], np.float32)
    pad[:,radius:radius+h] = x
    v = sum(pad[:,i:i+h]*k[i] for i in range(len(k)))
    pad = np.zeros((n,h,w+2*radius)+x.shape[3:], np.float32)
    pad[:,:,radius:radius+w] = v
    return sum(pad[:,:,i:i+w]*k[i] for i in range(len(k)))

def directions(im):
    smooth = spread(im,1,.70)
    gy = (np.roll(smooth,-1,axis=1)-np.roll(smooth,1,axis=1))*.5
    gx = (np.roll(smooth,-1,axis=2)-np.roll(smooth,1,axis=2))*.5
    mag = np.sqrt(gx*gx+gy*gy)
    ang = np.arctan2(gy,gx).astype(np.float32)*np.float32(4/np.pi)
    # Four unsigned normal orientations, with linear angular voting.
    channels = []
    for j in range(4):
        d = np.abs((ang-j+2)%4-2)
        channels.append(mag*np.maximum(0,1-d))
    return np.stack(channels,axis=3)

def features(im):
    ink = im[:,:,:,None]
    maps = np.concatenate([ink, directions(im)],axis=3)
    # Grid centres at 1.5,5.5,...,25.5. A coarser description is used only
    # to look up templates on which to do the deformable fine comparison.
    pool = spread(maps,4,1.65)
    c = .25*(pool[:,1::4,1::4]+pool[:,2::4,1::4]+pool[:,1::4,2::4]+pool[:,2::4,2::4])
    vecs = []
    for k,weight in ((c[:,:,:,0:1],.49),(c[:,:,:,1:],.51)):
        f = np.sqrt(np.maximum(k.reshape(len(im),-1),0))
        vecs.append(f * (np.sqrt(weight)/np.maximum(np.sqrt((f*f).sum(axis=1,keepdims=True)),1.e-7)))
    coarse = np.concatenate(vecs,axis=1).astype(np.float32)
    g = spread(maps,2,1.0)
    g = g.reshape(len(im),14,2,14,2,5).mean(axis=(2,4))
    # Square-root densities suppress differences in pen darkness. Keeping the
    # two channel families separately normalized avoids a pen-width prior.
    parts = []
    for k,weight in ((g[:,:,:,0:1],.52),(g[:,:,:,1:],.48)):
        f = np.sqrt(np.maximum(k,0))
        norm = np.sqrt((f*f).sum(axis=(1,2,3),keepdims=True))
        parts.append(f*(np.sqrt(weight)/np.maximum(norm,1.e-7)))
    fine = np.concatenate(parts,axis=3).astype(np.float32)
    return coarse, fine

def enclosed(bg):
    # Flood from all four borders in parallel. Four-connected white regions
    # respect diagonal pen strokes. What remains is enclosed background.
    f = bg.copy()
    f[:,1:-1,1:-1] = False
    for step in range(64):
        p = f.copy()
        p[:,1:] |= f[:,:-1]
        p[:,:-1] |= f[:,1:]
        p[:,:,1:] |= f[:,:,:-1]
        p[:,:,:-1] |= f[:,:,1:]
        p &= bg
        if np.array_equal(p,f):
            break
        f = p
    return (bg & ~f).astype(np.float32)

def topology(im):
    ref = im.max(axis=(1,2),keepdims=True)
    hole = .5*(enclosed(im < .28*ref)+enclosed(im < .57*ref))
    # Soft vertical location of closed counters; a tiny pinhole provides
    # less evidence than an easily visible counter. A tall hole in a zero
    # can activate both halves; the image terms distinguish it from an eight.
    levels = [np.clip((16-yy)/5,0,1), np.clip((yy-11)/5,0,1), np.clip((yy-9)/3,0,1)*np.clip((18-yy)/3,0,1)]
    ans = []
    for w in levels:
        vol = (hole*w).sum(axis=(1,2))
        ans.append(1-np.exp(-vol/8))
    return np.stack(ans,axis=1)
'''

infer = '''
ct = CONSTANTS['coarse'].astype(np.float32)
ft = CONSTANTS['fine']
rt = CONSTANTS['aspect']
st = CONSTANTS['slope']
rawt = CONSTANTS['rawratio']
ht = CONSTANTS['loops']
styles = CONSTANTS['styles'].astype(np.int32)
num = ct.shape[0]//10

# Little overlapping windows on a 7 by 7 displacement grid. The 14 by 14
# descriptor field has two nodes per grid cell in each direction. For each
# translation compute local squared feature distance, then pool and minimize.
def elastic(q,t):
    # q is one image, t is a modest shortlist of templates.
    pad = np.pad(q,((2,2),(2,2),(0,0)))
    small = np.full((len(t),7,7), 100.,np.float32)
    rigid = np.zeros(len(t),np.float32)
    for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1),(-2,0),(2,0),(0,-2),(0,2)):
        v = pad[2+dy:16+dy,2+dx:16+dx]
        dif = t-v
        d = np.einsum('nhwf,nhwf->nhw',dif,dif)
        cost = d.reshape(len(t),7,2,7,2).sum(axis=(2,4))
        # Neighbouring blocks vote on a shift, to discourage incompatible
        # motion of immediately adjacent strokes. Sum-preserving at borders.
        c = .5*cost
        c[:,1:] += .125*cost[:,:-1]
        c[:,:-1] += .125*cost[:,1:]
        c[:,:,1:] += .125*cost[:,:,:-1]
        c[:,:,:-1] += .125*cost[:,:,1:]
        c[:,0] += .125*cost[:,0]
        c[:,-1] += .125*cost[:,-1]
        c[:,:,0] += .125*cost[:,:,0]
        c[:,:,-1] += .125*cost[:,:,-1]
        if dx==0 and dy==0:
            rigid = c.sum(axis=(1,2))
        small = np.minimum(small,c+.00015*(dx*dx+dy*dy))
    return .83*small.sum(axis=(1,2))+.17*rigid

def block(x):
    im,r,s,raw = normalize(x[:,0].astype(np.float32))
    coarse,fine = features(im)
    loop = topology(im)
    # Exact integer inner products give the same first-stage decisions even
    # when the BLAS microkernel or the outer batch size changes. Feature
    # norms are below 2**24 after quantization, so every sum is exact.
    corr = np.rint(coarse*1100).astype(np.float32) @ ct.T
    cq = (np.rint(coarse*1100)**2).sum(axis=1)
    approx = (cq[:,None]+cn[None,:]-2*corr)/1210000
    dist = .017*np.minimum((r[:,None]-rt)**2,3.)
    dist += .032*(s[:,None]-st)**2+.013*np.minimum((raw[:,None]-rawt)**2,3.)
    dist += .015*((loop[:,0:1]-ht[:,0])**2+(loop[:,1:2]-ht[:,1])**2)+.008*(loop[:,2:3]-ht[:,2])**2
    approx = approx+dist
    res = np.empty(len(im),np.int32)
    for i in range(len(im)):
        allcost = approx[i].reshape(10,num)
        # Keep three distinct close variants and a fourth that may have a
        # different stroke construction from the closest variant.
        k = 14
        ix = np.argpartition(allcost,k,axis=1)[:,:k]
        vals = np.take_along_axis(allcost,ix,axis=1)
        ix = np.take_along_axis(ix,np.argsort(vals,axis=1,kind='stable'),axis=1)+np.arange(10)[:,None]*num
        first = ix[:,:3]
        diffstyle = styles[ix] != styles[ix[:,:1]]
        diffstyle[:,:3] = False
        loc = np.where(diffstyle.any(axis=1),diffstyle.argmax(axis=1),3)
        idx = np.concatenate([first,ix[np.arange(10),loc][:,None]],axis=1).ravel()
        d = elastic(fine[i],ft[idx].astype(np.float32)/1800.)
        d += .55*dist[i,idx]+.22*approx[i,idx]
        ordered = np.sort(d.reshape(10,4),axis=1)
        scores = .88*ordered[:,0]+.12*ordered[:,1]
        res[i] = np.argmin(scores)
    return res

cn = (ct*ct).sum(axis=1)

def predict(x):
    out = np.empty(len(x),np.int32)
    for start in range(0,len(x),48):
        out[start:start+48] = block(x[start:start+48])
    return out
'''

# Cursor data are in an imagined 100-by-100 handwriting square. Each
# elementary style is a procedure, not an image or a transcription. Its
# dimensions are drawn afresh. Endpoints get a common freehand warping to
# preserve junctions while allowing local variation. All random draws have a
# fixed seed; the sampler never inspects or compares completed templates.
rng = np.random.default_rng(873216)
def u(lo,hi):
    return rng.uniform(lo,hi)

def pick(*values):
    return values[rng.integers(len(values))]

class Pen:
    def __init__(self):
        self.strokes=[]
    def m(self,x,y):
        self.xy=np.array([x,y]); self.strokes.append([self.xy])
        return self
    def l(self,x,y):
        xy=np.array([x,y]); t=np.linspace(0,1,17)[1:,None]
        self.strokes[-1].extend(self.xy*(1-t)+xy*t); self.xy=xy
        return self
    def c(self,x1,y1,x2,y2,x,y):
        xy=np.array([x,y]); t=np.linspace(0,1,23)[1:,None]; s=1-t
        self.strokes[-1].extend(s**3*self.xy+3*s*s*t*[x1,y1]+3*s*t*t*[x2,y2]+t**3*xy)
        self.xy=xy
        return self
    def q(self,xc,yc,x,y):
        xy=np.array([x,y]); t=np.linspace(0,1,21)[1:,None]; s=1-t
        self.strokes[-1].extend(s*s*self.xy+2*s*t*[xc,yc]+t*t*xy); self.xy=xy
        return self

def zero(p,s):
    # Rotating the start angle lets the closure, gap, or overshoot occur
    # anywhere in the upper half, instead of at one canonical seam.
    rot = u(-2.9,.15) if s!=4 else u(-.5,1.6)
    gap = u(-.40,.14) if s not in (3,4) else u(.06,.62)
    turn = np.linspace(rot,rot+2*np.pi-gap,150)
    c,si=np.cos(turn),np.sin(turn)
    power=u(.83,1.16) if s!=1 else u(.62,.86)
    px=50+48*np.sign(c)*np.abs(c)**power*(1+u(-.22,.22)*si)
    py=50+50*np.sign(si)*np.abs(si)**power+u(-8,8)*c*c
    p.strokes.append(list(np.stack([px,py],axis=1)))
    if s==2:
        p.xy=np.array([px[-1],py[-1]])
        p.q(px[-1]-4,py[-1]+2,px[-1]+u(-25,20),py[-1]+u(3,20))
    if s==5:
        p.m(15,86).l(83,13)

def one(p,s):
    left=u(15,55); top=u(50,71); foot=top+u(-7,6)
    if s<4:
        p.m(top+u(-6,6),0)
        p.c(top+u(-7,8),32,foot+u(-7,7),78,foot,100)
        if s==3: p.q(foot+2,99,foot+u(5,15),u(88,100))
    else:
        h=u(12,48) if s!=9 else u(40,69)
        p.m(left-u(6,22),h)
        p.q(top-7,h*.37,top,0).q(top+u(-4,7),50,foot,100)
    if s in (6,7):
        ww=u(16,34)
        p.m(foot-ww,100+u(-2,2)).q(foot,98,foot+ww,100+u(-3,3))
    if s==8:
        p.l(foot+u(14,34),98)

def two(p,s):
    a=u(16,40); peak=u(34,64); end=u(82,107); top=u(62,101)
    bend=u(18,43); bx=u(0,19); by=u(85,100)
    if s==1:
        p.m(u(10,32),a+u(8,18)).c(-13,a,4,8,peak*.52,4).q(top,-12,top,bend)
    elif s==3:
        p.m(u(0,12),a).q(peak*.55,0,peak,0).q(top+15,-7,top,bend)
    else:
        p.m(u(-3,22),a).c(peak*.3,-2,top-3,-16,top,bend*.62).q(top+4,bend,top-8,bend+12)
    if s in (2,4,5):
        # A cursive two: the downstroke forms a small anticlockwise
        # counter at the left of the baseline on its return to the right.
        x=u(2,22); ly=u(65,84); wx=u(16,46)
        p.c(65,54,x+wx,92,x+7,96).c(x-16,106,x-20,ly+1,x-7,ly)
        p.c(x+11,ly-9,54,103,end,u(83,103))
    elif s==6:
        p.l(bx,by).l(end,u(79,101))
    else:
        p.c(75,u(43,62),bx+8,by-18,bx,by)
        if s==7: p.q(35,by-18,60,by-6).q(83,105,end,u(82,98))
        else: p.c(27,by+u(-7,8),78,by+u(-10,9),end,by+u(-12,5))

def three(p,s):
    h=u(40,59); cx=u(28,55); l=u(0,24); r=u(77,101); b=u(69,104)
    if s in (1,4):
        p.m(l,u(1,17)).q(44,u(-4,9),r,0).q(69,23,cx,h)
    else:
        p.m(l,u(8,27)).c(44,u(-13,2),r+4,u(-8,2),r,u(13,25))
        p.c(r-1,34,cx+14,h-1,cx,h)
    if s in (2,5):
        p.c(cx-u(18,33),h+9,cx-u(12,25),h-15,cx+5,h-5)
    if s==3:
        p.q(b+13,h-1,b,h+22).q(b-3,111,u(7,24),94)
    else:
        p.c(cx+26,h-11,b+11,h+7,b,h+27)
        p.c(b-8,105,23,110,u(-4,17),u(77,99))

def four(p,s):
    h=u(44,68); mid=u(54,79); base=mid+u(-12,10); rx=mid+u(6,27)
    if s in (2,5,8):
        p.m(mid,u(-3,6)).l(u(-5,10),h).l(rx,h+u(-9,3))
        p.m(mid+u(-4,4),u(-3,4)).q(mid+u(-5,8),51,base,100)
    elif s==9:
        # The square corner can be replaced with a continuous cup; its
        # rising arm is retraced as the descending stem starts.
        p.m(u(4,29),u(-2,6)).c(-12,h+14,mid-17,h+21,mid,h-20)
        p.q(mid+3,h*.3,mid+4,u(0,12)).q(mid+1,58,base,100)
    else:
        ax=u(9,44); end=u(2,18)
        p.m(ax,u(-4,9))
        if s in (1,4):
            p.c(ax-6,h*.46,end-14,h+6,end+7,h).q(47,h+3,rx,h+u(-10,2))
        elif s in (6,7):
            p.q(end-5,h+26,mid,h+u(-8,0))
        else:
            p.l(end,h).q(43,h+u(-4,5),rx,h+u(-9,2))
        p.m(mid+u(-5,9),u(-2,22)).c(mid,34,base+3,74,base,100)

def five(p,s):
    h=u(35,56); bx=u(0,19); ax=u(8,32); rx=u(77,103)
    if s==6:
        # Round, unbroken top half: approaches the shape of a capital S.
        p.m(rx,u(0,10)).c(47,-7,-4,9,8,h-8).c(17,h+8,104,h-7,90,h+28)
        p.c(82,101,29,111,u(-2,12),u(83,100))
        return
    # Many writers add the top arm as a separate final pen stroke.
    if s==4:
        p.m(ax-4,u(-5,4)).q(45,12,rx,u(-4,5))
    else:
        p.m(rx,u(-7,10)).q(45,u(-6,12),ax+u(-5,3),u(-3,7))
    if s in (1,3): p.m(ax+u(-8,6),u(3,15))
    else: p.m(ax,u(-1,5))
    p.q(ax-4,h*.53,bx,h)
    if s==2:
        p.c(56,h-9,85,h-10,89,h+10).q(115,89,53,99).q(18,108,0,82)
    elif s==5:
        p.c(68,h-27,113,h+14,74,83).c(50,107,12,104,u(-2,8),u(86,98))
    else:
        p.c(46,h-u(4,22),98,h-u(6,20),92,h+20)
        p.c(89,94,34,117,u(-4,21),u(78,100))

def six(p,s):
    ax=u(53,94); lx=u(0,15); ly=u(59,78); cy=u(40,62); rx=u(75,99)
    if s in (1,4):
        p.m(ax+5,u(10,25)).q(ax+1,-7,ax-13,1).c(31,8,lx,42,lx,ly)
    else:
        p.m(ax,u(-2,6)).c(ax-u(14,33),7,lx-2,42,lx,ly)
    p.c(lx-1,107,rx-23,113,rx,79).c(rx+20,cy-8,rx-24,cy-15,rx-42,cy-3)
    if s==3: p.q(36,cy-1,lx+u(8,25),ly-3)
    elif s==2: p.q(24,cy-3,lx+u(-6,4),ly+u(-5,6)).q(lx-3,ly+4,lx-9,ly+9)
    else: p.c(38,cy-1,lx+7,ly-u(5,17),lx+u(-3,8),ly+u(-5,5))

def seven(p,s):
    ax=u(-3,21); rx=u(76,101); by=100; bx=u(11,44); bend=u(-9,12)
    if s in (2,5): p.m(ax-2,u(13,32)).q(ax-3,6,ax+1,2)
    else: p.m(ax,u(-2,11))
    if s in (1,4): p.c(31,-14,69,15,rx,u(0,9))
    else: p.q(46,bend,rx,u(0,10))
    if s==7: p.q(48,53,bx+3,84).q(bx-4,101,bx-13,97)
    else: p.q((rx+bx)/2+u(-12,15),49,bx,by)
    if s in (3,4,5):
        mx=(rx+bx)*.5; h=u(42,63)
        p.m(mx-u(19,34),h).q(mx,h-1,mx+u(13,27),h+u(-5,3))

def eight(p,s):
    cy=u(38,60); cx=u(38,61); top=u(69,103); bot=u(69,104)
    lt=u(0,25); lb=u(-3,23)
    if s in (7,8):
        # A writer can start at a side of the upper loop rather than at its
        # waist. The pen can lift before closing this first loop on return.
        sx = lt if s==7 else top
        op = top if s==7 else lt
        p.m(sx,u(21,34)).c(sx-4,1,(op+sx)*.5,-9,op,8)
        p.c(op+(op-sx)*.12,cy*.48,op-2,cy*.70,cx,cy)
        p.c(2*cx-op,cy+18,2*cx-op-6,106,cx,100)
        p.c(op+16,100,op+11,cy+18,cx,cy)
        p.q((cx+sx)*.5,cy-7,sx+u(-5,5),u(8,33))
    elif s in (0,1,4,6):
        # Figure-eight: middle -> upper right -> upper left -> lower right
        # -> lower left -> middle. Variations include a gap or an overlap.
        p.m(cx+u(-3,5),cy+u(-3,5))
        p.c(top,cy-19,top+12,3,top-11,1)
        p.c(lt+1,-8,lt-13,cy*.55,lt+7,cy*.70)
        p.c(lt+29,cy-1,bot+16,cy+10,bot-1,cy+33)
        p.c(bot-13,111,lb-12,105,lb+3,cy+27)
        if s==4: p.q(lb+13,cy+15,cx-10,cy+10)
        elif s==6: p.q(lb+12,cy+11,cx+u(9,22),cy-u(2,12))
        else: p.q(lb+9,cy+14,cx+u(-2,5),cy+u(-3,3))
    elif s==3:
        p.m(cx,cy).c(lt-18,cy-5,lt-7,1,cx,0).c(top+15,-1,top+10,cy-7,cx,cy)
        p.c(lb-8,cy+3,lb-13,103,cx,100).c(bot+15,102,bot+19,cy+3,cx,cy)
    else:
        # Reverse diagonal through the crossing.
        p.m(cx,cy).c(lt-19,cy*.48,lt+2,-2,cx,0).c(top+23,1,top+2,cy-20,cx,cy)
        p.c(lb-25,cy+29,lb+2,106,cx,100).c(bot+22,99,bot+16,cy+22,cx+u(-4,8),cy+u(-2,6))

def nine(p,s):
    rx=u(66,92); lx=u(-2,17); h=u(40,63); by=100
    # Loop joins the high end of a descending stem. Either the loop is
    # closed by the stem itself, or it is closed before the stem is drawn.
    p.m(rx+u(-5,4),u(8,20))
    p.c(rx-7,-9,lx+11,-4,lx+1,h*.46)
    p.c(lx-17,h*.99,lx+7,h+12,lx+30,h-2)
    if s in (2,6):
        p.q(rx-12,h-7,rx-8,u(22,34))
        p.m(rx+u(-3,5),u(8,18))
    else:
        p.q(rx-4,h-13,rx+u(-3,5),u(4,19))
    bx=rx+u(-25,9)
    if s in (1,4):
        p.c(rx-2,53,rx-4,87,bx-12,98).q(bx-27,109,bx-42,u(85,95))
    elif s==3:
        p.q(rx+u(2,16),61,bx-16,98)
    elif s==5:
        p.c(rx,52,rx+2,82,bx+9,95).q(bx+14,102,bx+22,96)
    else:
        p.q(rx-2,58,bx,by)

models=[zero,one,two,three,four,five,six,seven,eight,nine]
# Repetitions of a style token set the intended frequency of common and
# uncommon writing conventions in the model, not a measured frequency.
menu=[
    [0,0,0,0,0,1,1,2,2,3,4,5],
    [0,0,0,1,2,3,4,4,5,5,6,7,8,9],
    [0,0,0,1,2,2,3,4,5,6,7,7],
    [0,0,0,1,1,2,2,3,3,4,5],
    [0,0,0,1,1,2,2,3,3,4,5,6,7,8,9],
    [0,0,0,1,1,2,3,4,5,6],
    [0,0,0,1,2,3,3,4],
    [0,0,0,1,1,2,3,4,5,6,7],
    [0,0,0,1,2,2,3,4,5,6,7,8],
    [0,0,0,1,2,2,3,4,5,6],
]

def render(digit,style):
    p=Pen(); models[digit](p,style)
    width=u(.50,1.03) if digit!=1 else u(.60,1.05)
    if digit==0: width=u(.43,.92)
    if digit==8: width=u(.53,.93)
    slant=u(-.52,.29) if digit!=1 else u(-.62,.25)
    roll=u(-.18,.18); bow=u(-.10,.10); pinch=u(-.19,.19)
    warp=rng.uniform(-1,1,(2,4))*[6,6,3.5,3.5]
    phases=rng.uniform(0,2*np.pi,(2,4))
    radius=u(4.0,10.7)
    ell=u(.80,1.14); pressure=u(.02,.22); speed=u(.8,1.4); offset=u(0,5)
    strokes=[]; radii=[]; bx=[]; by=[]
    for path in p.strokes:
        z=np.array(path); a=(z[:,0]-50)/100; b=(z[:,1]-50)/100
        modes=np.stack([a+.3*b, .3*a-b, a+b, b-.6*a],axis=1)*[np.pi,np.pi,1.8*np.pi,2.3*np.pi]
        dx=(np.sin(modes+phases[0])*warp[0]).sum(axis=1)
        dy=(np.sin(modes+phases[1])*warp[1]).sum(axis=1)
        x=(100*a*(1+pinch*b)+dx)*width+100*b*slant+bow*400*b*b
        y=100*b+roll*100*a*width+dy
        z=np.stack([x,y],axis=1)
        dist=np.sqrt(np.sum(np.diff(z,axis=0)**2,axis=1))
        t=np.r_[0.,dist.cumsum()]; length=max(t[-1],1.e-5)
        pos=np.arange(0,length+1.15,1.15).clip(0,length)
        z=np.stack([np.interp(pos,t,x),np.interp(pos,t,y)],axis=1)
        r=radius*(1+pressure*np.sin(pos/length*speed*2*np.pi+offset))
        r*=1-u(0,.33)*np.exp(-pos/6)-u(0,.33)*np.exp((pos-length)/6)
        strokes.append(z); radii.append(r)
        bx.extend([np.min(z[:,0]-ell*r),np.max(z[:,0]+ell*r)])
        by.extend([np.min(z[:,1]-r),np.max(z[:,1]+r)])
    x0,x1=min(bx),max(bx); y0,y1=min(by),max(by)
    scale=20/max(x1-x0,y1-y0)
    # Paint overlapping disks on a 4x subpixel grid. Radius variation
    # belongs to the pen, not to an intensity prototype.
    image=np.zeros((112,112),np.uint8)
    for z,r in zip(strokes,radii):
        z=(z-np.array([(x1+x0)/2,(y1+y0)/2]))*(4*scale)+55.5
        r=r*(4*scale)
        ext=int(np.ceil(max(1,ell)*r.max()+.5))
        k=np.arange(-ext,ext+1)
        base=np.rint(z).astype(int); f=z-base
        disk=(((k[None,:,None]-f[:,1,None,None]) / r[:,None,None])**2+
              ((k[None,None,:]-f[:,0,None,None])/(ell*r[:,None,None]))**2<=1)
        v,iy,ix=np.nonzero(disk)
        image[base[v,1]+k[iy],base[v,0]+k[ix]]=1
    return image.reshape(28,4,28,4).mean(axis=(1,3),dtype=np.float32)

def main():
    dest=Path(sys.argv[1]); dest.mkdir(parents=True,exist_ok=True)
    (dest/'solution.py').write_text(common+infer)
    fun={}; exec(common,fun)
    n=6144
    bank={}
    for digit in range(10):
        for start in range(0,n,48):
            xs=[]; ss=[]
            for j in range(start,start+48):
                style=menu[digit][j%len(menu[digit])]
                ss.append(style); xs.append(render(digit,style))
            im,asp,slant,raw=fun['normalize'](np.stack(xs))
            co,fi=fun['features'](im); ht=fun['topology'](im)
            pack=dict(coarse=np.rint(co*1100).astype(np.uint16),
                      fine=np.rint(fi*1800).astype(np.uint16),
                      loops=ht,aspect=asp,slope=slant,rawratio=raw,styles=np.array(ss,np.uint8))
            sl=slice(n*digit+start,n*digit+start+48)
            for key,val in pack.items():
                if key not in bank:
                    bank[key]=np.empty((n*10,)+val.shape[1:],val.dtype)
                bank[key][sl]=val
        print('Constructed digit',digit,flush=True)
    np.savez(dest/'constants.npz',**bank)

if __name__=='__main__':
    main()
