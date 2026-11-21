"""Data-free construction of a deformable handwritten numeral dictionary."""
import os
import sys
import numpy as np

# These functions are used both for one-template-at-a-time normalization at
# build time (vectorized for speed) and for transforming test images.
features = '''import numpy as np

axis28 = np.arange(28, dtype=np.float32)
axis24 = np.arange(24, dtype=np.float32) - 11.5

def ink_limits(proj):
    cs = proj.cumsum(1)
    total = cs[:, -1:]
    bounds = []
    for frac in (.012, .988):
        q = frac * total
        i = np.minimum((cs < q).sum(1), proj.shape[1]-1)
        n = np.arange(len(i))
        bounds.append(i.astype(np.float32) + .5 - (cs[n,i] - q[:,0]) / np.maximum(proj[n,i], 1.e-7))
    return bounds

def bilinear(im, y, x):
    y = np.clip(y, 0, im.shape[1]-1.0001)
    x = np.clip(x, 0, im.shape[2]-1.0001)
    j = y.astype(np.int32)
    i = x.astype(np.int32)
    b = y-j.astype(np.float32)
    a = x-i.astype(np.float32)
    ind = np.arange(len(im))[:,None,None]
    return (1-b)*((1-a)*im[ind,j,i]+a*im[ind,j,i+1]) + b*((1-a)*im[ind,j+1,i]+a*im[ind,j+1,i+1])

def canonical(im):
    # Undo horizontal shear using ink moments, without rotating the baseline.
    yproj = im.sum(2)
    mass = np.maximum(yproj.sum(1), 1.e-6)
    cy = (yproj*axis28).sum(1) / mass
    dy = axis28[None,:] - cy[:,None]
    slope = ((im*axis28).sum(2)*dy).sum(1) / np.maximum((yproj*dy*dy).sum(1),1.e-5)
    slope = np.clip(slope,-1.2,1.2)
    offsets = np.arange(44, dtype=np.float32) - 8
    xx = offsets[None,None,:] + dy[:,:,None]*slope[:,None,None]
    sh = bilinear(im, axis28[None,:,None],xx)
    left,right = ink_limits(sh.sum(1))
    top,bottom = ink_limits(yproj)
    w = np.maximum(right-left,1.)
    h = np.maximum(bottom-top,1.)
    # Partial rather than full width normalization keeps a 1 narrow.
    targetw = np.clip(16.*(w/h/.75)**.45,5.,21.)
    xs = (left+right)[:,None,None]*.5 - 8. + axis24[None,None,:]*(w/targetw)[:,None,None]
    ys = (top+bottom)[:,None,None]*.5 + axis24[None,:,None]*(h/19.)[:,None,None]
    xs = xs + (ys-cy[:,None,None])*slope[:,None,None]
    norm = bilinear(im,ys,xs)
    return norm, w/h

def center_features(im):
    p = np.pad(im,((0,0),(1,1),(1,1)))
    smooth = (p[:,:-2,1:-1]+p[:,2:,1:-1]+p[:,1:-1,:-2]+p[:,1:-1,2:]+4*im)*.125
    p = np.pad(smooth,((0,0),(1,1),(1,1)))
    gx = (p[:,1:-1,2:]-p[:,1:-1,:-2])*.5
    gy = (p[:,2:,1:-1]-p[:,:-2,1:-1])*.5
    mag = np.sqrt(gx*gx + gy*gy)
    # A modest compression of contrast lets weak pen marks contribute too.
    mag = np.power(mag, .8)
    angle = np.arctan2(gy,gx) * (4./np.pi)
    cells = []
    for i in range(8):
        a = mag*np.maximum(0., 1.-np.abs((angle-i+4.)%8.-4.))
        cells.append(a.reshape(len(im),12,2,12,2).sum((2,4)))
    return np.stack(cells,axis=1), smooth

def unit(v):
    return v / np.sqrt(np.maximum((v*v).sum(1,keepdims=True),1.e-10))

def windowmat(n, count, sigma):
    centers = (np.arange(count)+.5)*n/count - .5
    a = np.exp(-.5*((np.arange(n)[None,:]-centers[:,None])/sigma)**2)
    return a.astype(np.float32)

poolgrad = windowmat(12,6,1.)
poolink = windowmat(24,8,1.35)
poolspace = windowmat(24,5,1.8)

# The shadow on each side of a stroke encodes concavities that a local
# gradient cannot see (e.g. the open space between the stems of a 4).
def negative_space(smooth):
    left = np.maximum.accumulate(smooth,axis=2)
    right = np.maximum.accumulate(smooth[:,:,::-1],axis=2)[:,:,::-1]
    above = np.maximum.accumulate(smooth,axis=1)
    below = np.maximum.accumulate(smooth[:,::-1],axis=1)[:,::-1]
    horizontal = np.minimum(left,right)-smooth
    vertical = np.minimum(above,below)-smooth
    concavities = np.stack((horizontal,vertical,np.minimum(horizontal,vertical)),axis=1)
    pooled = pool(concavities,poolspace)/16.
    v = (np.sqrt(pooled+.05)-np.float32(np.sqrt(.05))).reshape(len(smooth),-1)
    return unit(np.concatenate((v,np.ones((len(v),1),dtype=np.float32)),axis=1))

def localmap(smooth):
    ink = smooth.reshape(len(smooth),12,2,12,2).mean((2,4))
    p = np.pad(ink,((0,0),(1,1),(1,1)))
    dx = p[:,1:-1,2:]-p[:,1:-1,:-2]
    dy = p[:,2:,1:-1]-p[:,:-2,1:-1]
    grid = np.stack((.85*ink,dx,dy),axis=1)
    grid /= np.sqrt(np.maximum((grid*grid).sum((1,2,3),keepdims=True),1.e-8))
    return np.clip(np.rint(384*grid),-127,127).astype(np.int8)

def pool(grid, win):
    # Explicit order of the short contractions, independent of batch layout.
    tmp = np.einsum('py,ncyx->ncpx',win,grid,optimize=False)
    return np.einsum('qx,ncpx->ncpq',win,tmp,optimize=False)

def descriptors(im):
    canon,ratio = canonical(im)
    cells, smooth = center_features(canon)
    hist = pool(cells,poolgrad)
    hist = np.sqrt(hist+.012) - np.float32(np.sqrt(.012))
    # Normalize both over the local patch and over the glyph.
    hist /= (.45+(hist*hist).sum(1,keepdims=True))**.19
    occ = pool(smooth[:,None],poolink)[:,0]
    occ = np.sqrt(occ+.10)-np.float32(np.sqrt(.10))
    hg = unit(hist.reshape(len(im),-1))
    hk = unit(occ.reshape(len(im),-1))
    space = negative_space(smooth)
    full = np.concatenate((np.sqrt(np.float32(.67))*hg,np.sqrt(np.float32(.21))*hk,np.sqrt(np.float32(.12))*space),axis=1)
    return full, canon, ratio, localmap(smooth)
'''

matcher = '''
def components(im, threshold):
    # Flood exterior background, then measure the largest few enclosed regions.
    v = im < threshold
    v[0] = True
    v[-1] = True
    v[:,0] = True
    v[:,-1] = True
    def grow(seed):
        old = -1
        for k in range(96):
            a = seed.copy()
            a[1:] |= seed[:-1]
            a[:-1] |= seed[1:]
            a[:,1:] |= seed[:,:-1]
            a[:,:-1] |= seed[:,1:]
            seed = a & v
            now = seed.sum()
            if old == now:
                break
            old = now
        return seed
    outside = np.zeros_like(v)
    outside[0] = True
    outside[-1] = True
    outside[:,0] = True
    outside[:,-1] = True
    v &= ~grow(outside)
    holes = []
    for k in range(8):
        if not v.any():
            break
        seed = np.zeros_like(v)
        seed.flat[np.argmax(v)] = True
        part = grow(seed)
        v &= ~part
        ys,xs = np.nonzero(part)
        if len(ys)>=2:
            holes.append((len(ys),float(ys.mean())))
    return holes

def topology(im):
    # Soft evidence: no missing-hole penalty (loops can be open or filled in).
    evidence = np.zeros(10,dtype=np.float32)
    for t in (.25,.54):
        holes = components(im,t)
        if not holes:
            continue
        holes = sorted(holes,reverse=True)
        area,yy = holes[0]
        power = np.clip((area-2.)/13.,0,1)
        # A large central counter is typical of 0; counters on 6 and 9
        # occupy the lower and upper zones respectively.
        d = (yy-11.5)/8.
        small = np.clip(1.-(area-6.)/28.,0,1)
        e = np.array([.012-.018*abs(d),-.052,-.030,-.031,-.012,
                      -.027,.010+.025*d,-.044,.002,.010-.025*d], dtype=np.float32)
        e[0] += .016*(1.-small)-.014*small
        e[4] += .015*np.clip(-d,0,1)
        e[2] += .014*np.clip(d,0,1)
        e[8] += .008*small
        if len(holes)>1:
            a2,y2 = holes[1]
            two = np.clip((a2-1.)/8.,0,1)*np.clip((abs(yy-y2)-2.)/4.,0,1)
            e -= .015*two
            e[8] += .085*two
        evidence += .5*power*e
    return evidence

# Integer-valued descriptors and dots make shortlist selection independent of
# floating point BLAS summation order. The exact scores fit inside 24 bits.
bank = CONSTANTS['glyph'].astype(np.float32)
logaspect = CONSTANTS['aspect'].astype(np.float32)
count = len(bank)//10

def match_block(im):
    # Keep even the sizes of elementwise math operations in preprocessing
    # independent of the number and positions of neighboring query images.
    rows = [descriptors(np.ascontiguousarray(im[j:j+1])) for j in range(len(im))]
    desc,canon,ratios,local = [np.concatenate(a,axis=0) for a in zip(*rows)]
    q = np.rint(desc*4096.).astype(np.float32)
    # Templates quantized separately with scale 1024; all dots are positive.
    scores = q @ bank.T
    scores *= (1./4194304.)
    scores -= .018*np.minimum(1.6,np.abs(np.log(ratios)[:,None]-logaspect[None,:]))**1.4
    scores = scores.reshape(len(im),10,count)
    shortlist = np.argpartition(scores,count-6,axis=2)[:,:,-6:]
    s = np.take_along_axis(scores,shortlist,axis=2)
    picks = shortlist + np.arange(10)[None,:,None]*count
    best = np.empty((len(im),10),dtype=np.float32)
    for n in range(len(im)):
        templates = CONSTANTS['local'][picks[n].ravel()].astype(np.float32)
        loss = elastic(local[n].astype(np.float32),templates).reshape(10,6)
        z = np.sort(s[n]-.19*loss,axis=1)
        best[n] = z[:,-1]*.68 + z[:,-3:].sum(1)*(.32/3.) + topology(canon[n])
    return best.argmax(1)

def elastic(q,t):
    # Squared edge/ink mismatch after moving each 4x4 patch at most one
    # sample. Compare both directions so a template cannot hide extra ink.
    def patchsum(a):
        return a.reshape(len(a),3,4,3,4).sum((2,4))
    qt = q[None,:,:,:]-t
    center = patchsum((qt*qt).sum(1))
    best1 = center.copy()
    best2 = center.copy()
    qp = np.pad(q,((0,0),(1,1),(1,1)))
    tp = np.pad(t,((0,0),(0,0),(1,1),(1,1)))
    for dy,dx in ((0,1),(0,-1),(-1,0),(1,0),(-1,-1),(1,1),(-1,1),(1,-1)):
        y,x = 1+dy,1+dx
        d1 = t-qp[None,:,y:y+12,x:x+12]
        d2 = q[None]-tp[:,:,y:y+12,x:x+12]
        cost = 360*(abs(dy)+abs(dx))
        best1 = np.minimum(best1,patchsum((d1*d1).sum(1))+cost)
        best2 = np.minimum(best2,patchsum((d2*d2).sum(1))+cost)
    return (.17*center.sum((1,2))+.415*(best1+best2).sum((1,2)))/(384.*384.)

def predict(x):
    out = np.empty(len(x),dtype=np.int64)
    for i in range(0,len(x),96):
        out[i:i+96] = match_block(x[i:i+96,0])
    return out
'''

# The glyph grammar. Coordinates are arbitrary pen units on a y-down page,
# with a typical capital about ten units tall. C is a cubic Bezier, Q is a
# quadratic Bezier, L is a straight segment, M lifts the pen.
class Pen:
    def __init__(self):
        self.lines=[]
    def M(self,x,y):
        self.pos=np.array([x,y])
        self.lines.append([self.pos[None,:]])
        return self
    def L(self,x,y):
        b=np.array([x,y]); a=self.pos
        t=np.linspace(0,1,30)[1:,None]
        self.lines[-1].append(a*(1-t)+b*t); self.pos=b
        return self
    def Q(self,x1,y1,x,y):
        a=self.pos;b=np.array([x1,y1]);c=np.array([x,y]);t=np.linspace(0,1,30)[1:,None];u=1-t
        self.lines[-1].append(a*(u*u)+b*(2*u*t)+c*(t*t));self.pos=c
        return self
    def C(self,x1,y1,x2,y2,x,y):
        a=self.pos;b=np.array([x1,y1]);c=np.array([x2,y2]);d=np.array([x,y]);t=np.linspace(0,1,46)[1:,None];u=1-t
        self.lines[-1].append(a*(u*u*u)+b*(3*u*u*t)+c*(3*u*t*t)+d*(t*t*t));self.pos=d
        return self
    def points(self):
        return [np.concatenate(line) for line in self.lines]

def glyph(label,rng):
    U=rng.uniform
    p=Pen();style=U(); coin=U
    if label==0:
        # An oval can be egg-shaped, pinched, slightly open, or overtraced.
        rad=U(3.1,5.); c=rad
        angle=U(-2.2,-.6) if coin()<.85 else U(-3.5,0)
        gap=U(-.32,.22) if coin()<.86 else U(.20,.78)
        t=np.linspace(angle,angle+2*np.pi-gap,235)
        e=U(.74,1.22); s=np.sin(t);co=np.cos(t)
        a=c+rad*np.sign(co)*abs(co)**e*(1+U(-.20,.20)*s)+U(-.45,.45)*(s*s)
        b=5+5*np.sign(s)*abs(s)**e
        if style<.18:
            a[:35]+=np.linspace(U(-1.8,1.8),0,35);b[:35]+=np.linspace(U(.3,1.8),0,35)
        return [np.stack([a,b],1)]
    if label==1:
        x=4.;bx=x+U(-1.4,1.4); by=10
        if style<.56:
            # A plain pen stroke, sometimes with just a flick at the ends.
            p.M(x+U(-.55,.55),U(-.25,.25)).Q(x+U(-.85,.85),U(3.,7.),bx,by)
        else:
            lead=U(.6,3.9); tiph=U(.65,3.95)
            p.M(x-lead,tiph).Q(x-.48*lead,U(.15,.85)*tiph,x,0).Q(x+U(-.55,.55),5,bx,by)
        if style>.84:
            p.M(bx-U(.8,3.6),by+U(-.45,.45)).Q(bx,by+U(-.45,.45),bx+U(1.,3.2),by+U(-.45,.45))
        elif coin()<.14:
            p.Q(bx+U(-.1,.4),10.3,bx+U(.35,1.15),U(9.5,10.2))
    if label==2 and style>.92:
        # A narrow upright version: a tall hooked opening and a long exit.
        p.M(U(.1,1.),U(2.8,4.7)).C(-1.2,1.35,.8,-1.15,2.5,.25)
        p.C(4.25,1.35,3.05,5.4,2.15,7.05).Q(.35,10.4,-.6,9.5)
        p.C(-1.45,8.15,.45,8.25,2.25,9.05).Q(4.55,10.25,U(6.0,8.4),U(8.2,9.85))
        return p.points()
    if label==2:
        # Arched top; descend diagonally into a baseline. The cursive baseline
        # can curl back through the diagonal and make a lower-left counter.
        w=U(5.8,8.8); start=U(1.1,4.1); apex=U(1.,3.1)
        if coin()<.13:
            p.M(-.2,start+U(-1.2,.1)).Q(.1,start+U(.4,1.15),.65,start)
        else:p.M(0,start)
        p.C(U(.4,1.4),-.4,w*U(.73,.99),U(-2.,-.5),w,apex)
        foot=U(-.15,1.25)
        if style<.59:
            p.C(w+U(-.6,.45),U(4.,6.), U(.8,3.7),U(7.1,9.), foot, U(9.25,10.2))
            p.Q(U(2.4,5.1),U(8.65,11.3), w+U(-.2,2.),U(8.7,10.4))
        elif style<.90:
            y=U(6.7,8.7); a=U(2.1,4.8)
            p.C(w+U(-.35,.65),5.1,a,9.05,foot+.4,9.85)
            p.C(foot-2.6,11.,foot-2.45,y-.4,foot,y)
            p.C(1.7,y-.45,w*.64,U(10.3,11.),w+U(.3,2.2),U(8.6,10.2))
        else:
            p.Q(w,5.0,foot,9.7).Q(2.7,9.2,w*.8,9.65).Q(w+1.5,10.2,w+1.0,U(8.,9.35))
    if label==3:
        w=U(5.8,8.5); waist=U(3.85,5.85); mid=U(.9,3.65); lx=U(-.75,1.2)
        p.M(lx,U(.3,2.6))
        if style<.64:
            p.C(w*.35,U(-1.,-.1),w+U(.3,1.4),U(-.65,.8),w-.3,U(1.15,2.5))
            p.Q(w-.4,U(3.4,4.3),mid,waist)
        else:
            p.Q(w*.46,U(-.4,.8),w,U(-.1,1.0)).Q(w*.69,waist*.55,mid,waist)
        if style>.91:
            p.Q(mid-1.5,waist+1.,mid-1.05,waist-.15).Q(mid-.55,waist-.9,mid+.9,waist-.55)
        p.C(w+U(.0,2.1),waist-U(.7,1.9),w+U(.1,2.4),8.3,w*U(.65,.90),9.25)
        p.Q(w*.33,11.1,lx+U(-.7,1.1),U(7.4,9.75))
    if label==4:
        # Two-stroke open 4 (including the round U form), or a triangular 4.
        cross=U(4.95,7.0); stem=U(5.1,7.3); end=stem+U(-1.,1.)
        lt=U(.45,3.6); sTop=U(-.3,2.5)
        if style<.65:
            p.M(lt,U(-.3,1.55))
            if style<.33:
                p.Q(U(-.3,1.9),cross*.57,.1,cross).Q(stem*.5,cross+U(-.9,.5),stem+U(.25,2.8),cross+U(-.85,.65))
            else:
                p.C(U(-1.,.45),cross*.84,U(-.6,1.1),cross+2.,stem*.63,cross+U(-.2,.55))
                p.Q(stem-.65,cross-.15,stem+U(-.3,1.1),cross-U(.3,1.8))
            p.M(stem+U(-.45,.8),sTop).Q(stem+U(-.45,.45),5.45,end,10)
        else:
            point=stem+U(-.8,.7); tip=U(-.25,.7)
            p.M(point,tip).Q(stem*.52,cross*.49,-.15,cross)
            p.Q(stem*.63,cross+U(-.8,.55),stem+U(.15,2.45),cross+U(-.65,.7))
            p.M(point+U(-.65,.7),tip+U(-.35,1.55)).Q(stem+U(-.65,.7),5.,end,10.)
    if label==5:
        w=U(5.6,8.15);y=U(3.65,5.7);corner=U(.3,2.5);left=U(-.55,1.1)
        start=U(-.2,1.3); cap=U(-.75,1.0)
        if style<.66:
            p.M(w+U(-.15,1.2),cap).Q(w*.48,U(-.55,1.15),corner,start)
        else:
            p.M(corner,start)
        if coin()<.32:
            p.C(corner+U(-1.,.6),2.15,left-.75,y+.4,left,y+.35)
        else:p.L(left,y+.35)
        p.C(w*.40,y-U(1.2,2.45),w+U(.2,1.1),y-U(1.25,2.0),w,y+U(1.05,2.35))
        p.C(w+U(-.4,.7),9.25,w*.36,11.35,left+U(-.85,1.35),U(8.25,10.0))
        if style>=.66:
            p.M(corner+U(-1.35,.5),start+U(-.5,.55)).Q(w*.59,U(-.85,.95),w+U(-.1,1.45),cap)
    if label==5 and style>.83:
        # In cursive writing the corner tends to round off and the base can
        # finish in a short upward flick rather than a blunt bottom serif.
        w=U(5.2,8.3);yc=U(2.7,5.3);p=Pen()
        p.M(w,U(-.5,.6)).Q(w*.6,U(.7,1.65),2.1,.6)
        p.C(.8,-.15,-.55,yc-.65,1.4,yc)
        p.C(3.8,yc+.25,w+.45,yc-.15,w,7.15).Q(w-.15,10.5,1.7,9.8)
        p.Q(.0,9.65,-.15,U(7.6,9.25))
    if label==6:
        w=U(4.95,8.15); mid=U(3.7,7.15); upper=U(.6,w*.97);left=U(-.25,.5)
        p.M(upper,U(-.2,.5))
        p.C(upper-U(1.,3.),U(.75,2.3),left-.65,mid-.55,left,mid+U(.9,2.1))
        p.C(left-.2,9.65,w*.25,10.85,w*.66,9.7)
        p.C(w+U(.6,1.85),8.65,w+.5,mid+.05,w*.80,mid-.05)
        if style<.75:
            p.C(w*.52,mid-1.15,w*.20,mid+.1,left+U(-.55,.75),mid+U(.65,2.25))
        else:
            p.Q(w*.62,mid-1.05,w*U(.20,.55),mid+U(-.15,.85))
        if style<.10:p.Q(left+.3,mid+3.,w*.40,9.6)
    if label==7:
        w=U(5.8,9.35); top=U(-.25,1.2); foot=U(.05,w*.71)
        if style<.20:
            p.M(U(-.5,.25),U(1.55,3.15)).L(0,U(-.25,.8))
        else:p.M(0,U(-.25,1.25))
        p.Q(w*.54,U(-.8,1.8),w,top)
        p.Q(w*.55+foot*.45+U(-1.4,1.2),4.7,foot,10.)
        if .21<style<.45:
            x=.5*(w+foot);y=U(4.2,6.65)
            p.M(x-U(1.5,3.45),y+U(-.6,.5)).Q(x,y+.25,x+U(.85,2.6),y+U(-.45,.55))
    if label==8:
        mid=U(4.1,5.8);x=U(2.7,4.35);upper=U(2.45,4.1);lower=U(2.9,4.55)
        skew=U(-1.2,1.2);bottom=10.;upx=x+skew;lowx=x+U(-.6,.6)
        if style>.89:
            # One unfinished bowl: start from a loose end on the upper left,
            # cross the waist, close the bottom and hook back across the top.
            p.M(upx-upper*U(.60,1.15),mid*U(.16,.57))
            p.Q(upx-upper,mid*.62,x,mid)
            p.C(lowx+1.3*lower,mid+2.1,lowx+1.35*lower,10.2,lowx+.2,10.)
            p.C(lowx-1.28*lower,10.25,lowx-1.29*lower,mid+2.55,x,mid)
            p.C(upx+1.3*upper,mid-1.95,upx+1.3*upper,-.20,upx+.3,.12)
            p.Q(upx-.48*upper,-.3,upx-upper*U(.6,1.2),mid*U(.03,.37))
        elif style<.78:
            # Figure eight. Trace (in either direction) from the crossing,
            # around one bowl, diagonally through it and around the other.
            side=1 if coin()<.53 else -1
            p.M(x+U(-.55,.55),mid+U(-.55,.55))
            p.C(x-side*upper*.70,mid*.76,upx-side*1.43*upper,mid*.24,upx-side*.68*upper,.37)
            p.C(upx+side*.23*upper,-1.15,upx+side*upper*1.21,-.05,upx+side*upper,mid*.43)
            p.C(upx+side*upper*.89,mid*.72,x+side*.85,mid-.42,x,mid)
            p.C(x-side*lower*.65,mid+1.05,lowx-side*lower*1.4,mid+2.95,lowx-side*lower*.8,9.35)
            p.C(lowx-side*lower*.29,bottom+1.05,lowx+side*lower*1.20,bottom+.45,lowx+side*lower,bottom-1.9)
            p.C(lowx+side*lower*.94,mid+1.7,x+side*1.2,mid+.45,x+U(-.65,.65),mid+U(-.75,.7))
        else:
            # Stacked ovals sharing a short stroke at the waist.
            for cx,cy,rx,ry,phi in [(upx,.5*mid,upper,.52*mid,np.pi/2),(lowx,.5*(mid+bottom),lower,.51*(bottom-mid),-np.pi/2)]:
                theta=np.linspace(phi,phi+2*np.pi+U(-.2,.16),130)
                pts=np.stack([cx+rx*np.cos(theta),cy+ry*np.sin(theta)],axis=1)
                p.lines.append([pts]);p.pos=pts[-1]
    if label==9:
        w=U(5.1,8.05);mid=U(3.45,7.0); side=w+U(-.65,.5); start=U(.7,2.45)
        # A closed upper bowl or an open hook, followed by the main descender.
        p.M(w-U(-.1,1.15),start)
        p.C(w*.73,-1.25,w*.22,-.45,w*.06,mid*.33)
        p.C(-1.0,mid*.62,-.30,mid+1.,w*.33,mid+.1)
        p.Q(w*.68,mid+.3,side,mid*U(.28,.72))
        stemtop=U(-.25,1.6)
        if style<.69:p.Q(side+.30,stemtop,side+U(-.1,.5),stemtop+.35)
        elif style<.85:p.M(side+U(-.2,.6),stemtop)
        bx=side+U(-2.3,1.45)
        if style<.79:
            p.Q(side+U(-.45,1.05),mid+1.1,bx,10.0)
        else:
            p.C(side+.8,mid+1.55,side+.4,10.75,bx-U(1.1,2.7),9.65)
        if style<.09:p.Q(bx-.7,10.55,bx-1.6,9.45)
    return p.points()

def draw(label,rng):
    U=rng.uniform
    paths=glyph(label,rng)
    # All deformations, pen widths and orientations are specified a priori.
    # Only points from the current drawing are used in this rendering.
    coords=np.concatenate(paths).astype(np.float64)
    x,y=coords.T.copy(); x-=.5*(x.max()+x.min());y-=.5*(y.max()+y.min())
    x*=np.exp(U(-.29,.28));y*=np.exp(U(-.13,.13))
    wave=U(-.68,.68)*np.cos(y*.48+U(-np.pi,np.pi))
    wave+=U(-.42,.42)*np.sin(x*.55+y*.53+U(-np.pi,np.pi))
    newy=y+U(-.7,.7)*np.cos(x*.57+U(-np.pi,np.pi))+x*U(-.23,.23)
    x=x+wave+y*U(-.64,.43);y=newy
    radius=U(.38,1.17) if U()<.86 else U(.95,1.50)
    rx=radius*U(.83,1.16);ry=radius*U(.83,1.16)
    scale=20./max(x.max()-x.min()+2*rx,y.max()-y.min()+2*ry)
    x=(x-.5*(x.min()+x.max()))*scale+13.5+U(-.40,.40)
    y=(y-.5*(y.min()+y.max()))*scale+13.5+U(-.40,.40)
    rx*=scale;ry*=scale
    # Rasterize at triple resolution. Variable pen pressure gives thinner tips
    # and gently varying width along the writing direction.
    res=3;n=28*res
    xx=(x+.5)*res-.5;yy=(y+.5)*res-.5
    pressure=1.+U(-.23,.23)*np.sin(y*.34+x*.18+U(-np.pi,np.pi))
    off=0
    for path in paths:
        length=len(path);tt=np.linspace(0,1,length)
        pressure[off:off+length]*=1.-U(.0,.41)*np.exp(-13*tt)-U(.0,.43)*np.exp(-12*(1-tt));off+=length
    # Splat small disks, grouping radii to keep the cost independent of canvas size.
    rxi=np.maximum(1,np.rint(rx*res*pressure).astype(int));ryi=np.maximum(1,np.rint(ry*res*pressure).astype(int))
    ix=np.rint(xx).astype(int);iy=np.rint(yy).astype(int);img=np.zeros(n*n+1,np.uint8)
    for a,b in set(zip(rxi.tolist(),ryi.tolist())):
        pts=(rxi==a)&(ryi==b);foot=brush(a,b,n)
        base=ix[pts]+n*iy[pts]
        indices=(base[:,None]+foot[None,:]).ravel()
        img[np.clip(indices,0,n*n)]=1
    img=img[:n*n].reshape(28,res,28,res).mean((1,3)).astype(np.float32)
    return img

brushes={}
def brush(a,b,n):
    key=(a,b,n)
    if key not in brushes:
        dy,dx=np.mgrid[-b:b+1,-a:a+1]
        ok=(dx/(a+.15))**2+(dy/(b+.15))**2<=1
        brushes[key]=(dx[ok]+n*dy[ok])
    return brushes[key]

def build(dest):
    global descriptors
    env={};exec(features,env);descriptors=env['descriptors']
    os.makedirs(dest,exist_ok=True)
    with open(os.path.join(dest,'solution.py'),'w') as stream:
        stream.write(features+matcher)
    rng=np.random.default_rng(45017)
    blocks=[];aspects=[];local=[]
    for lab in range(10):
        for block in range(144):
            images=np.stack([draw(lab,rng) for i in range(64)])
            vectors,can,ratio,loc=descriptors(images)
            blocks.append(np.rint(vectors*1024).astype(np.uint16));aspects.append(np.log(ratio).astype(np.float32));local.append(loc)
    np.savez(os.path.join(dest,'constants.npz'),glyph=np.concatenate(blocks),aspect=np.concatenate(aspects),local=np.concatenate(local))

if __name__=='__main__':
    build(sys.argv[1])
