"""Build a fixed quadrature table of a hand-specified ten-way stroke grammar.
No fonts, examples, saved descriptors, or other input files are used.
"""
import numpy as np
from pathlib import Path
import sys

# The following numpy geometry is shared verbatim by builder and recognizer.
geometry = '''import numpy as np

axis28 = np.arange(28, dtype=np.float32)
axis32 = np.arange(32, dtype=np.float32)

def blur(a):
    b = np.pad(a, ((0,0),(1,1),(1,1)))
    c = (b[:,2:] + 2*b[:,1:-1] + b[:,:-2])*.25
    return (c[:,:,2:] + 2*c[:,:,1:-1] + c[:,:,:-2])*.25

def bounds(mass):
    mass = mass / np.maximum(mass.sum(1,keepdims=True),1e-8)
    # A soft extent: exceptionally light incidental marks do not fix the scale.
    cs = np.cumsum(mass,axis=1)
    idx = np.minimum((cs < .007).sum(1),27)
    lo = idx-.5 + 1 - np.clip((cs[np.arange(len(idx)),idx]-.007) / np.maximum(1e-8,mass[np.arange(len(idx)),idx]),0,1)
    idx = np.minimum((cs < .993).sum(1),27)
    hi = idx-.5 + 1 - np.clip((cs[np.arange(len(idx)),idx]-.993) / np.maximum(1e-8,mass[np.arange(len(idx)),idx]),0,1)
    return lo.astype(np.float32),hi.astype(np.float32)

def normalize(a):
    n = len(a)
    r = np.arange(n)[:,None,None]
    den = np.maximum(a.sum((1,2)),1e-6)
    rows = a.sum(2)
    my = (rows*axis28).sum(1)/den
    dy = axis28[None,:]-my[:,None]
    xy = (a * (axis28[None,None,:]-13.5)).sum(2)
    cov = (xy*dy).sum(1)
    var = (rows*dy**2).sum(1)
    shear = .87*np.clip(cov/np.maximum(var,1e-6),-1.35,1.35)
    # Re-sample just to measure the horizontal extent after shearing. The
    # final image is taken directly from the input, not from this interpolant.
    sx = axis28[None,None,:] + (shear[:,None]*dy)[:,:,None]
    ix = np.floor(sx).astype(np.int32)
    f = sx-ix
    ry = np.arange(28)[None,:,None]
    z = a[r,ry,np.clip(ix,0,27)]*(1-f) + a[r,ry,np.clip(ix+1,0,27)]*f
    lx,hx = bounds(z.sum(1))
    ly,hy = bounds(rows)
    w,h = np.maximum(2,hx-lx),np.maximum(2,hy-ly)
    sy = h/24
    # Partial aspect normalization: allow width to vary less than in the input,
    # without stretching a vertical pen stroke into the shape of a block.
    sw = np.sqrt(np.maximum(w/18.0,sy*.33)*sy)
    yy = ((ly+hy)*.5)[:,None] + sy[:,None]*(axis32-15.5)
    xx = ((lx+hx)*.5)[:,None,None] + sw[:,None,None]*(axis32-15.5) + (shear[:,None]*(yy-my[:,None]))[:,:,None]
    iy = np.floor(yy).astype(np.int32)[:,:,None]
    ix = np.floor(xx).astype(np.int32)
    fy = (yy[:,:,None]-iy).astype(np.float32)
    fx = (xx-ix).astype(np.float32)
    b = np.pad(a,((0,0),(1,1),(1,1)))
    yl = np.clip(iy+1,0,29)
    xl = np.clip(ix+1,0,29)
    yh = np.clip(iy+2,0,29)
    xh = np.clip(ix+2,0,29)
    return ((b[r,yl,xl]*(1-fx)+b[r,yl,xh]*fx)*(1-fy)+(b[r,yh,xl]*(1-fx)+b[r,yh,xh]*fx)*fy).astype(np.float32),np.stack((shear,w/h),1).astype(np.float32)

def bits(a):
    b = blur(a)
    p = np.pad(b,((0,0),(1,1),(1,1)))
    gx = p[:,1:-1,2:]-p[:,1:-1,:-2]
    gy = p[:,2:,1:-1]-p[:,:-2,1:-1]
    return b,gx,gy

def pooled(a,s):
    return a.reshape(len(a),32//s,s,32//s,s).mean((2,4))

def unit(a):
    a = a.reshape(len(a),-1)
    return a/np.sqrt(1e-8+(a*a).sum(1,keepdims=True))

def describe(a,extra):
    b,gx,gy = bits(a)
    v = []
    for j in range(8):
        t = np.pi*j/4
        d = np.cos(t)*gx+np.sin(t)*gy
        # A narrow signed-edge orientation channel.
        e = np.maximum(0,d-.51*np.sqrt(gx*gx+gy*gy))
        v.append(np.sqrt(pooled(blur(blur(e.astype(np.float32))),4)))
    ori = np.stack(v,1)
    img = np.sqrt(pooled(b,2))
    return np.concatenate((unit(ori)*.80,unit(img)*.60),axis=1).astype(np.float32)

def counters(a):
    # Where is white paper inaccessible from the frame? Repeat at several
    # ink levels so a gap just one grey pixel wide is not a hard decision.
    maps = []
    peak = a.max((1,2),keepdims=True)
    for lev in (.17,.40,.64):
        paper = a < lev*peak
        exterior = np.zeros_like(paper)
        exterior[:,0,:] = paper[:,0,:]
        exterior[:,-1,:] = paper[:,-1,:]
        exterior[:,:,0] = paper[:,:,0]
        exterior[:,:,-1] = paper[:,:,-1]
        for step in range(65):
            nxt = exterior.copy()
            nxt[:,1:] |= exterior[:,:-1]
            nxt[:,:-1] |= exterior[:,1:]
            nxt[:,:,1:] |= exterior[:,:,:-1]
            nxt[:,:,:-1] |= exterior[:,:,1:]
            nxt &= paper
            if np.array_equal(nxt,exterior):
                break
            exterior = nxt
        rows = (paper & ~exterior).sum(2).astype(np.float32)
        centers = np.arange(7,dtype=np.float32)*4+3.5
        win = np.maximum(0,1-np.abs(axis32[:,None]-centers[None,:])/5)
        areas = (rows[:,:,None]*win).sum(1)
        maps.append(areas/(6+areas))
    return (maps[0]+maps[1]+maps[2])/3

def localfields(a):
    b,gx,gy = bits(a)
    v = [np.maximum(gx,0),np.maximum(-gx,0),np.maximum(gy,0),np.maximum(-gy,0),b*.59]
    return np.stack([pooled(blur(ch),2) for ch in v],axis=-1)
'''

# Pen notation is a polyline through knots in an imaginary 10 by 10 em.
# ~ gives a smooth tangent through the preceding knot; space gives a corner;
# | lifts the pen. The units have no relationship to a particular sample.
styles = []
def glyph(d, weight, s):
    styles.append((d,weight,s))

glyph(0, 5, '5,0~ 8,1~ 9.5,4~ 8.8,7.7~ 6.3,10~ 3.5,9.8~ 1.1,7.6~ .5,4.8~ 1.6,2~ 5,0~ 7.4,.7')
glyph(0, 3, '6.6,.6~ 4.3,0~ 1.7,1.5~ .5,4.5~ 1.1,7.9~ 3.4,9.8~ 6.4,9.7~ 8.7,7.4~ 9.2,3.5~ 8.2,1~ 6.4,.2')
glyph(0, 1, '5.5,0~ 2.6,1~ .5,5.5~ 1.7,9~ 5,10~ 7.8,8.7~ 9.5,5.5~ 9.1,2.4~ 7.4,.4~ 5.4,.8~ 4.1,2.8')
glyph(0, 1, '5.1,.8~ 2.9,.2~ 1.3,2~ .6,6.5~ 2.0,9.6~ 5.1,10~ 8.2,8.2~ 9.5,4.2~ 8.9,1.3~ 7.3,.1~ 5.8,.2')

glyph(1, 6, '5.2,0~ 4.9,3~ 5.1,6~ 5,10')
glyph(1, 2.2, '4.2,1.7 5.5,.1~ 5.3,4~ 5.2,7~ 5.3,10')
glyph(1, 1.6, '1.8,3.3 5.6,0 5.5,10')
glyph(1, .35, '2.3,2.9 5.4,0 5.3,9.9 | 2.4,10 8.3,10')
glyph(1, .55, '5.2,.2 4.8,8.8~ 5.4,9.9~ 6.8,9.6')

glyph(2, 4, '.8,2.1~ 3,.4~ 6.1,0~ 8,1.1~ 8.2,3.1~ 6.2,5.4~ 2.8,8.1~ .8,9.8 4.8,9.7~ 9.7,9.7')
glyph(2, 2, '.8,3.4~ 1.5,1.3~ 4,.1~ 7,.4~ 8.2,2.1~ 7,4.8~ 3.5,7.9~ 1.4,9.9~ 4.7,9.2~ 7.3,9.7~ 9.6,9.9')
glyph(2, 3, '.5,1.6~ 2.7,.1~ 5.5,.6~ 6.5,2.8~ 5.8,6.1~ 3.9,9.1~ 1.8,10~ .8,9.2~ 2,8~ 4.7,8.1~ 7.6,9.6~ 10,9.7')
glyph(2, 1.7, '.6,2.5~ .8,.7~ 3.5,.1~ 5.9,1.2~ 6.4,3.3~ 5.2,6.6~ 2.7,9.6~ 1.4,9.8~ 2,8.8~ 5.3,8.6~ 8.4,9.1~ 10,8.9')
glyph(2, .9, '.5,1~ 3.4,.2~ 7,.3~ 8.5,1.1~ 7.8,3~ 3.5,7.1~ 1,9.7 9.6,10')
glyph(2, .7, '.8,.5~ 3.1,.4~ 5.4,1.9~ 5.8,4.7~ 4.4,8~ 2.3,10~ 1.3,9~ 2.2,7.9~ 5.7,7.7~ 9.7,8.5')

glyph(3, 4, '.8,1.2~ 3.8,.1~ 7.3,.3~ 8.7,1.5~ 7.6,3.3~ 4.4,4.8 7,4.8~ 9,6~ 9,7.9~ 7,9.6~ 3.6,10~ .6,8.8')
glyph(3, 2.5, '.7,.6~ 4.5,0~ 8.4,.5 4.2,4.3~ 3,5.2~ 6.2,4.8~ 8.4,5.7~ 8.7,7.9~ 6.8,9.6~ 3.5,9.9~ .5,9.6')
glyph(3, 2, '1.5,.5~ 4.3,.1~ 7,1~ 7.5,2.5~ 6.3,4~ 3.4,5.1~ 3.3,4.5~ 6.1,4.6~ 8.5,6.2~ 8.6,8.2~ 6.7,9.7~ 3.3,10~ 1,8.4')
glyph(3, 1.1, '.5,.4~ 4,.1~ 7.8,.3~ 9,.8~ 6.4,2.9~ 3.8,4.5 7.7,6.4~ 8.6,8.1~ 6.5,9.7~ 3.2,9.9~ .6,9')
glyph(3, .8, '1.6,1.9~ 2.5,.5~ 5.7,0~ 8.1,1.4~ 7.8,3.5~ 5.9,4.8 8.4,6.5~ 8.3,8.6~ 6.2,10~ 3.6,9.7~ 1.2,8')

glyph(4, 4, '2.3,.3~ 1.2,3.8~ .8,5.7~ 1.5,6.4~ 5.6,6~ 9.1,5.8 | 7.8,.9~ 7.3,4.7~ 7.3,10')
glyph(4, 2.5, '1.4,0~ .9,2.7~ .8,4.5~ 1.9,5.3~ 4.8,5.4~ 8,4.7 | 7.4,.5~ 7.1,4.6~ 7.5,10')
glyph(4, 2.3, '6.8,.1 .7,6.3 9.7,6.1 | 7.6,.2 7.4,10')
glyph(4, 1.3, '3.7,0 .8,5.9 7.5,5.8 7.6,1.2 7.3,10')
glyph(4, .9, '1.8,0~ .8,2.6~ 1.4,5.1~ 3.3,5.8~ 6.3,5.5~ 7.7,3.4~ 7.7,.9 7.5,5.7~ 7.2,10')
glyph(4, .7, '5.7,.1 1,5.6 9.5,5.3 | 7.5,2.4 7,10')

glyph(5, 3, '9,.1~ 5.4,.3~ 2.1,.1 1.5,4.6 4.6,4.1~ 7.1,4.7~ 8.4,6.6~ 8,8.3~ 6,9.7~ 3.2,10~ .6,9.1')
glyph(5, 2.3, '9.7,-.3~ 6.3,.6~ 2,.8 1.2,3.4~ 2.7,4.5~ 6,5.2~ 7.4,6.8~ 7,8.5~ 5.1,9.7~ 2,9.8~ .6,9.2')
glyph(5, 2.1, '2.5,.5~ 1.1,2.4~ 1.3,4.1~ 4.3,4.6~ 7,5.8~ 7.9,7.8~ 6.3,9.6~ 3.5,10~ .6,9 | 1.7,.7~ 5.5,.5~ 9.6,0')
glyph(5, 1.2, '8.6,.1~ 5.1,.5~ 2.3,1.1~ 1.4,2.5~ 2.4,3.7~ 6.5,5.5~ 8.1,7.6~ 7,9.3~ 4.3,10~ 1.5,9.6~ .4,8')
glyph(5, 1.0, '1.5,.6 1.4,4.7~ 3.8,3.6~ 6.4,3.7~ 8.4,5.1~ 8.7,7.6~ 7,9.4~ 3.9,10~ .8,9.4 | 2.9,.4~ 5.9,.4~ 9.5,0')
glyph(5, .55, '8.8,.2~ 5.1,.6~ 2.1,.6 1.1,3.9~ 3.5,4.6~ 6.6,6.1~ 7.1,8~ 5.4,9.6~ 2.7,9.9~ 1.4,9~ 2.6,7.7~ 4.3,7.3')

glyph(6, 4, '7.7,.2~ 5.7,1~ 3.3,3.4~ 1.7,6.4~ 1.7,8.5~ 3.4,9.9~ 6.2,9.6~ 8.1,8.1~ 8.1,6.3~ 6.3,5.3~ 3.9,5.7~ 1.7,7.4')
glyph(6, 2, '6.5,.1~ 4.2,1.2~ 2.3,3.9~ 1.8,7.1~ 2.5,9.2~ 4.4,10~ 7.2,9.4~ 8.3,7.5~ 7.4,5.9~ 5.4,5.7~ 3.1,7~ 2.6,8.2')
glyph(6, 1.4, '8.4,.7~ 6.8,.2~ 4.9,1~ 2.5,4.1~ 1.5,7.3~ 2.4,9.3~ 4.8,10~ 7.5,9.2~ 8.4,7.3~ 7.7,5.3~ 5.8,4.7~ 3.5,5.4~ 1.8,6.9')
glyph(6, 1.5, '7.4,.1~ 4.9,2.2~ 2.5,5.8~ 1.9,8.4~ 2.8,9.7~ 5.1,10~ 7.7,9.2~ 8.4,7.7~ 7.4,6.4~ 5.5,6.4~ 3.7,7.5')
glyph(6, .65, '7.2,.2~ 4.3,2.2~ 1.9,5.8~ 2.1,8.8~ 4,10~ 6.8,9.7~ 8.4,8.5~ 8.3,6.6~ 6.9,5.5~ 5.6,5.6')

glyph(7, 4.5, '.5,.8~ 3.7,.2~ 7.3,.2~ 9.3,.3 6.6,4.4~ 3.8,10')
glyph(7, 2.1, '.8,1.7~ 1.5,.5~ 4.1,.2~ 7.2,.9~ 9,.6 6,4.9~ 4.6,10')
glyph(7, 1.6, '.3,.3~ 3.5,1.1~ 6.7,1.2~ 8.8,.6 6.8,4.9~ 4.6,10')
glyph(7, 1.2, '.7,.6~ 4,.2~ 8.9,.2 6.4,5.1~ 4.4,10 | 3.9,5.2 8.2,5.1')
glyph(7, .5, '.5,1.9 1.2,.2 8.7,.5 5.7,5.1~ 3.9,10 | 3.3,5.3 8.1,5.5')
glyph(7, .5, '.4,.2~ 4.8,.3~ 8,.5~ 8.5,1.8~ 6.1,5.7~ 4.2,10')

glyph(8, 3.5, '5,4.9~ 2.3,3.5~ 1.8,1.9~ 3.3,.4~ 5.6,.1~ 7.9,.9~ 8.1,2.5~ 6.4,4.1~ 3.1,6.2~ 1.6,8~ 2.5,9.5~ 5.1,10~ 7.6,9.3~ 8.4,7.6~ 7,6~ 4.9,4.8')
glyph(8, 2.5, '6.9,1.1~ 5.1,.2~ 3,.5~ 1.7,1.9~ 2.5,3.8~ 5.6,5.1~ 8.1,6.9~ 8.5,8.4~ 7.1,9.7~ 4.4,10~ 2.2,9.1~ 1.8,7.4~ 3.4,5.6~ 6.8,3.7~ 8,2.1~ 7.3,.8~ 5.7,.3')
glyph(8, 1.5, '4.2,.2~ 2.2,.8~ 1.6,2.4~ 2.7,4~ 6.7,5.7~ 8.2,7.4~ 7.8,9~ 5.9,10~ 3.5,9.6~ 2.1,8~ 2.7,6.1~ 6.9,3.4~ 7.8,1.6~ 6.7,.3~ 5.5,.2')
glyph(8, 1.4, '5.2,4.8~ 2.8,4.1~ 1.8,2.3~ 2.7,.7~ 5.2,.1~ 7.6,.8~ 8.2,2.6~ 7.1,4.1~ 4.8,5~ 2.6,6.2~ 1.6,8~ 2.8,9.6~ 5.5,9.9~ 7.9,8.9~ 8.2,7.2~ 6.8,5.7~ 4.7,4.8')
glyph(8, .8, '6.8,.7~ 4.8,.2~ 2.6,.8~ 2.2,2.5~ 3.6,4.4~ 6.8,6.2~ 7.8,8.5~ 6.3,9.9~ 3.9,9.6~ 2.5,7.9~ 3.1,5.9~ 5.5,3.5~ 6.9,1.8~ 6.2,.5')

glyph(9, 3, '7.6,2~ 6.6,.5~ 4.6,.1~ 2.2,1.2~ 1.2,3.2~ 1.7,4.9~ 3.3,5.6~ 5.7,4.6~ 7.7,2.2 7.3,5.6~ 7.4,10')
glyph(9, 2.5, '7.9,1.7~ 6.6,.3~ 4,.1~ 1.9,1.5~ 1.5,3.4~ 2.8,4.9~ 5,5.2~ 7,4.1~ 7.9,2.1~ 7.9,.9 7.5,5.7~ 7.4,10')
glyph(9, 1.6, '7.3,1.4~ 5.5,.1~ 3.2,.3~ 1.4,1.8~ 1,3.8~ 2.6,5.2~ 5.4,5~ 7.4,3.3~ 7.7,1.1 7.9,4.4~ 7.5,7.8~ 6.4,10')
glyph(9, 1.2, '8.3,2~ 7.4,.5~ 5.1,.1~ 2.9,1~ 1.7,2.9~ 2.2,4.7~ 4,5.5~ 6.1,4.8~ 7.9,2.6 8,5.3~ 7.2,8.4~ 5.2,10~ 3.5,9.6')
glyph(9, 1.3, '7.9,.2~ 5.8,0~ 3,.9~ 1.3,2.8~ 1.4,4.8~ 3.1,5.5~ 5.7,4.3~ 8.2,1.3 7.3,5.7~ 7.6,10')
glyph(9, .75, '5.8,.2~ 3.5,.4~ 1.7,2~ 1.5,4.2~ 3.1,5.4~ 5.5,4.8~ 7.4,2.4~ 7.3,.4~ 6.2,0 | 7.6,1.3~ 7.7,5.3~ 8.5,10')

# More ductus families: a two can start with a rising upstroke, a five
# can curl back through its bowl, and the ends of a figure eight need not meet.
glyph(2, 1.8, '1.4,4.7~ .9,3.3~ 1.3,1.3~ 2.9,.1~ 5.4,.2~ 7.5,1.5~ 7.8,3.8~ 6.3,6.4~ 3.3,9.4~ 2.2,10~ 3,9.1~ 5.3,8.6~ 8.1,9.1~ 10,9.2')
glyph(2, 1.25, '1.7,4~ .9,2.4~ 1.4,.9~ 3.1,.1~ 5.2,.7~ 6.5,2.9~ 6.3,6~ 4.7,8.9~ 2.7,10~ 1.1,9.3~ 1.2,7.5~ 3.1,6.6~ 5.8,7~ 8.3,8.4~ 10.3,8.2')
glyph(2, 1.4, '.5,1.6~ 2,.1~ 4.1,.5~ 5.3,2.8~ 5.1,5.9~ 3.4,8.9~ 2,9.9~ .7,9.5~ .5,8~ 1.6,6.1~ 3.3,5.7~ 5.1,6.4~ 7.8,8.7~ 10,9.5')
glyph(2, .8, '1.1,3.6~ 1.5,1.3~ 3.5,0~ 6,.6~ 7.3,2.4~ 7.1,4.6~ 4.6,7.3~ 2.2,10 5.3,9.5~ 8.1,9.1~ 10,7.9')
glyph(2, .8, '.9,.4~ 3,.1~ 5.1,1~ 6.5,3.3~ 6.1,6.3~ 4.4,9.3~ 3,10~ 2.2,9.2~ 3.7,7.8~ 6.3,7.3~ 9.9,7.5')
glyph(3, .8, '.6,.9~ 3.6,.3~ 6.9,.5~ 7.3,1.9~ 5.3,3.8~ 2.7,5.1~ 2.5,4.2~ 4.8,3.8~ 7.5,4.8~ 9,7.1~ 8.4,9~ 6.4,10~ 3.3,9.6~ 1,8')
glyph(5, 1.15, '9.8,-.2~ 6,.6~ 2.4,1 1.5,4.9~ 3.9,4.3~ 6.1,4.8~ 7.3,6.4~ 6.9,8.4~ 5.1,9.7~ 2.7,10~ 1.1,9.2~ 1.4,7.9~ 3.2,7.1~ 5.2,7.6')
glyph(5, 1, '1.9,1.1~ 1,3~ 1.5,4.7~ 3.4,5~ 6.7,5~ 8.3,6.4~ 8.1,8.5~ 6.1,9.8~ 3.1,10~ 1.1,8.8~ 1.4,7.1 | 3.7,1.5~ 6.2,1~ 9.8,.2')
glyph(5, 1.1, '8.7,.3~ 6.2,.4~ 3.8,1.2~ 2,3.5~ 1,5.4~ 3.2,4.7~ 5.7,4.9~ 7.3,6.1~ 7.3,8~ 5.7,9.6~ 3.2,10~ .6,9.4')
glyph(5, .6, '2.2,.4 1.6,3.1~ 2.8,4.2~ 5.2,4.7~ 6.4,6.4~ 6.3,8.3~ 4.8,9.8~ 2.6,9.8~ 1.1,8.4 | 4.1,1.1 10,-.2')
glyph(6, .65, '6.5,.1~ 4.2,3.1~ 3.2,6.4~ 3.4,8.7~ 4.8,10~ 7.1,9.7~ 8.3,8~ 7.9,5.7~ 6.6,4.4~ 4.4,4.9~ 3.2,6.6')
glyph(7, .9, '.3,1.5~ 2.1,.6~ 5,.2~ 7.1,.7~ 8.3,1.9~ 8,3.5~ 6,6.9~ 4.7,10')
glyph(7, .7, '1,1.1~ 2.9,.7~ 5.2,1.1~ 7,1.8~ 8.9,1.3~ 9,.4 8,3.6~ 7.4,6.9~ 7.5,10')
glyph(7, .55, '1,3.2~ .9,1.4~ 2.4,.2~ 5.1,.1~ 8.7,.3 6.7,4.5~ 4.1,10')
glyph(8, 1, '6.6,.5~ 4.3,.1~ 2.1,1.7~ 2.3,3.7~ 5.2,5.7~ 7.3,7.8~ 7.2,9.3~ 5.2,10~ 3.4,9~ 3.3,6.8~ 5.7,4.2~ 8.1,2.2~ 8.5,1')
glyph(8, .95, '2.9,1.6~ 3.9,.4~ 6.2,0~ 8,1.3~ 7.9,3.1~ 5.8,5.3~ 3.1,7.1~ 2.7,8.9~ 4,10~ 6.1,9.7~ 7.3,8.2~ 6.5,6.2~ 3.3,3.9~ 2.3,2.7')
glyph(8, .65, '7.8,2.2~ 6.4,.3~ 4.5,.2~ 3.4,1.5~ 3.6,3.6~ 5.8,5.9~ 7.9,7.7~ 8.2,9~ 6.8,10~ 4.1,9.6~ 2.7,8~ 3.1,6.3~ 6.4,4.4~ 8.3,2.9~ 8.3,1.3')
glyph(9, .9, '6.2,.2~ 3.7,.1~ 1.5,1.4~ 1.4,3.5~ 3.5,4.9~ 6,4.6~ 7.6,2.9~ 7.8,.7 7.7,5.4~ 8.3,8.8~ 9.5,10')
glyph(9, .7, '8.2,.9~ 5.7,.2~ 2.9,1~ 1.2,3.3~ 1.9,5.7~ 4.3,6.2~ 6.4,4.5~ 7.9,1.2 7,5.9~ 7.5,8.6~ 8.5,10')

# Compose several continuous transformations of the em: independently move the
# waist vertically, make upper/lower bowls wider, tilt/bend the axis, and jitter
# the knots. Smooth, spatially-correlated perturbations keep crossings together.
def makepaths(text, digit, count, rng):
    def normal(shape):
        return np.clip(rng.normal(size=shape),-2.5,2.5)
    m=count
    sx=np.exp(normal(m)*.23)*(.65 if digit==1 else .83)
    shear=.22+.34*normal(m)
    tilt=.10*normal(m)
    bend=.85*normal(m)
    slide=.60*normal(m)
    waist=.16*normal(m)
    wide=.19*normal(m)
    pinch=.115*normal(m)
    ybow=.48*normal(m)
    knots=normal((m,3,3,2))*.53
    off=[]; po=[]; draws=[]
    for stroke in text.split('|'):
        entries=stroke.split()
        u=np.array([list(map(float,e.rstrip('~').split(','))) for e in entries])
        v=np.array([e.endswith('~') for e in entries])
        p=np.tile(u,(m,1,1))
        x,y=u[:,0],u[:,1]
        # Displace all knots by one interpolated low-frequency vector field.
        for k in range(3):
            for j in range(3):
                b=np.maximum(0,1-np.abs(x*.2-k))*np.maximum(0,1-np.abs(y*.2-j))
                p+=knots[:,k,j,None,:]*b[None,:,None]
        x,y=p[:,:,0].copy(),p[:,:,1].copy()
        level=(y-5)*.2; side=(x-5)*.2
        arch=1-level**2
        p[:,:,0]=((x-5)*(1+wide[:,None]*level+pinch[:,None]*arch)+bend[:,None]*arch+slide[:,None]*np.sin(2*np.pi*y/10))*sx[:,None]+(5-y)*shear[:,None]
        p[:,:,1]=y+waist[:,None]*5*arch+(x-5)*tilt[:,None]+ybow[:,None]*(1-side**2)
        p+=normal(p.shape)*(.22 if digit==1 else .28)
        a=p[:,:-1]; b=p[:,1:]; ab=b-a
        diff=np.diff(p,axis=1); lens=np.maximum(.05,np.sqrt((diff*diff).sum(-1,keepdims=True)))
        direction=diff/lens
        tang=np.concatenate((direction[:,:1],.5*(direction[:,:-1]+direction[:,1:]),direction[:,-1:]),axis=1)
        # Cubic Hermite interpolants with a straight tangent at sharp corners.
        ma=np.where(v[None,:-1,None],tang[:,:-1]*lens,ab)
        mb=np.where(v[None,1:,None],tang[:,1:]*lens,ab)
        nv=24
        t=(np.arange(nv)/nv)[None,None,:,None]
        d=(a[:,:,None]*(2*t**3-3*t*t+1)+b[:,:,None]*(3*t*t-2*t**3)+ma[:,:,None]*(t**3-2*t*t+t)+mb[:,:,None]*(t**3-t*t))
        d=np.concatenate((d.reshape(m,-1,2),p[:,-1:]),axis=1)
        po.append(d)
        # The width can drift along each stroke and taper at either endpoint.
        t2=np.linspace(0,1,d.shape[1])[None,:]
        w=(1+.10*normal((m,1))*np.sin(np.pi*t2)+.095*normal((m,1))*(2*t2-1))
        w*=1-rng.uniform(0,.42,(m,1))*np.exp(-t2/.075)-rng.uniform(0,.4,(m,1))*np.exp((t2-1)/.075)
        draws.append(w)
    return np.concatenate(po,axis=1),np.concatenate(draws,axis=1)

def render(text,digit,m,rng):
    path,variation=makepaths(text,digit,m,rng)
    mn=path.min(1); mx=path.max(1)
    radius=np.clip(np.exp(rng.normal(-.20,.37,m)),.32,1.75)
    radiusx=radius*np.exp(rng.normal(0,.11,m)); radiusy=radius*np.exp(rng.normal(0,.11,m))
    scale=19.8/np.maximum(mx[:,0]-mn[:,0]+2*radiusx,mx[:,1]-mn[:,1]+2*radiusy)
    path=(path-(mn+mx)[:,None,:]*.5)*scale[:,None,None]+13.5
    rx=radiusx[:,None]*scale[:,None]*variation
    ry=radiusy[:,None]*scale[:,None]*variation
    # Stamp the moving pen at subpixel positions. max-at merges overlapping
    # passes without turning retraced lines into extra dark patches.
    ix=np.rint(path[:,:,0]).astype(int); iy=np.rint(path[:,:,1]).astype(int)
    xx=path[:,:,0]-ix; yy=path[:,:,1]-iy
    rad=np.sqrt(rx*ry); soft=np.clip(rng.normal(.86,.15,(m,1)),.6,1.3)
    rr=np.zeros(m*28*28,dtype=np.float32); index=(np.arange(m)*784)[:,None]+iy*28+ix
    margin=int(np.ceil(max(rx.max(),ry.max())+.9))
    for dy in range(-margin,margin+1):
        for dx in range(-margin,margin+1):
            if dx*dx+dy*dy>(margin+.7)**2: continue
            val=np.clip((1-np.sqrt(((dx-xx)/rx)**2+((dy-yy)/ry)**2))*rad/soft+.5,0,1).astype(np.float32)
            use=val>0
            np.maximum.at(rr,np.clip(index[use]+dy*28+dx,0,rr.size-1),val[use])
    img=rr.reshape(m,28,28)
    # Center of the em rather than a single knot is used above. Normalization
    # is invariant to the translation used for rasterization.
    return img

runtime = '''
def elastic(q,templates):
    # A correspondence lattice. Each patch may move a cell up/down and
    # left/right; a displacement has a cost, as does a discontinuity between
    # neighboring patches. The final distance uses the full pairwise energy.
    n,k = templates.shape[:2]
    temp = templates.reshape(-1,16,16,5).astype(np.float32)*(1/255)
    temp = np.pad(temp,((0,0),(1,1),(1,1),(0,0)))
    q = q[:,None,:,:, :]
    unary = []
    for dy in range(-1,2):
        for dx in range(-1,2):
            tv = temp[:,dy+1:dy+17,dx+1:dx+17].reshape(n,k,16,16,5)
            df = (tv-q).reshape(n,k,4,4,4,4,5)
            unary.append((df*df).sum((3,5,6))+.07*(dy*dy+dx*dx))
    cost = np.stack(unary,axis=-1)
    # Two alternating checkerboard relaxations, with starts given by the
    # independent-patch optima. Only non-neighbors move in each half-sweep.
    labels = np.argmin(cost,axis=-1)
    grid = np.arange(9)
    pot = .092*((grid[:,None]//3-grid//3)**2+(grid[:,None]%3-grid%3)**2)
    parity = (np.arange(4)[:,None]+np.arange(4))%2
    for phase in range(4):
        edges = pot[labels].astype(np.float32)
        u = cost.copy()
        u[:,:,1:] += edges[:,:,:-1]
        u[:,:,:-1] += edges[:,:,1:]
        u[:,:,:,1:] += edges[:,:,:,:-1]
        u[:,:,:,:-1] += edges[:,:,:,1:]
        labels = np.where(parity==phase%2,u.argmin(-1),labels)
    energy = np.take_along_axis(cost,labels[:,:,:,:,None],axis=-1).sum((2,3,4))
    energy += pot[labels[:,:,1:],labels[:,:,:-1]].sum((2,3))
    energy += pot[labels[:,:,:,1:],labels[:,:,:,:-1]].sum((2,3))
    norm = (q*q).sum((2,3,4))+1e-5
    return energy/norm

def predict(x):
    result = np.empty(len(x),np.int64)
    # Use fixed-size microkernels. Quantizing the unit query vector to a
    # power-of-two grid makes the coarse dot products exact in float32:
    # the sum of their nonnegative integer numerators is under 2**24.
    tile = 4 if len(x)<8 else 32
    for start in range(0,len(x),tile):
        n = min(tile,len(x)-start)
        inp = np.zeros((tile,28,28),np.float32)
        inp[:n] = x[start:start+n,0]
        img,extra = normalize(inp)
        f = np.rint(16384*describe(img,extra))
        scores = []
        shortlist = []
        cq = counters(img)
        qfields = localfields(img)
        for d in range(10):
            dot = (f @ features[d].T)*(1/(16384*240))
            dw = (extra[:,1:2]-aspects[d][None,:]) / (.32+.25*extra[:,1:2])
            ds = (extra[:,0:1]-slants[d][None,:]) / .65
            dist = 1-dot + .009*np.minimum(3,dw**2) + .006*np.minimum(3,ds**2)
            sel = np.argpartition(dist,64,axis=1)[:,:64]
            first = np.take_along_axis(dist,sel,axis=1)
            gap = cq[:,None,:]-countermaps[d,sel]
            first += .010*(gap*gap).sum(2)
            idx = np.argsort(first,axis=1)[:,:12]
            e = np.take_along_axis(first,idx,axis=1)
            scores.append(e)
            shortlist.append(np.take_along_axis(sel,idx,axis=1))
        e = np.stack(scores,1)
        which = np.stack(shortlist,1)
        coarse = .70*e[:,:,0]+.30*e.mean(2)
        # Use the displacement lattice on the three plausible explanations.
        rivals = np.argsort(coarse,axis=1)[:,:3]
        qrow = np.arange(tile)[:,None]
        score = []
        for rj in range(3):
            ds = rivals[:,rj:rj+1]
            ix = which[qrow,ds][:,0,:]
            de = elastic(qfields,fieldmaps[ds,ix])
            combined = e[qrow,ds][:,0,:] + .22*de
            best = np.sort(combined,axis=1)
            score.append(.74*best[:,0]+.26*best[:,:5].mean(1))
        labels = rivals[np.arange(tile),np.argmin(np.stack(score,1),axis=1)]
        result[start:start+n] = labels[:n]
    return result

features = [np.ascontiguousarray(a,dtype=np.float32) for a in CONSTANTS['directions']]
aspects = CONSTANTS['attributes'][:,:,1]
slants = CONSTANTS['attributes'][:,:,0]
countermaps = CONSTANTS['counters'].astype(np.float32)*(1/255)
fieldmaps = CONSTANTS['fields']
'''

# Execute only our literal geometric formulas, not code taken from a file.
exec(geometry)

def main():
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    (out/'solution.py').write_text(geometry+runtime)
    rng=np.random.default_rng(50265)
    total=5632
    features=np.empty((10,total,768),np.uint8)
    attrs=np.empty((10,total,2),np.float32)
    fields=np.empty((10,total,16,16,5),np.uint8)
    ctr=np.empty((10,total,7),np.uint8)
    for digit in range(10):
        group=[(w,s) for d,w,s in styles if d==digit]
        ends=np.rint(np.cumsum([w for w,s in group])*total/sum(w for w,s in group)).astype(int)
        start=0
        for (weight,text),end in zip(group,ends):
            for i in range(start,end,64):
                n=min(64,end-i)
                img=render(text,digit,n,rng)
                img,extra=normalize(img)
                f=describe(img,extra)
                features[digit,i:i+n]=np.rint(f*240).astype(np.uint8)
                attrs[digit,i:i+n]=extra
                ctr[digit,i:i+n]=np.rint(255*counters(img)).astype(np.uint8)
                fields[digit,i:i+n]=np.rint(255*np.clip(localfields(img),0,1)).astype(np.uint8)
            start=end
        print('built geometry for',digit,flush=True)
    np.savez(out/'constants.npz',directions=features,attributes=attrs,counters=ctr,fields=fields)

if __name__=='__main__':
    main()
