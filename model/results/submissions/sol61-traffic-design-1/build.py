"""Data-free traffic-sign classifier: all parameters are analytic geometry.

No data, fonts, fitted coefficients, or external assets are read. Coordinates
below describe ideal road-sign artwork in a dimensionless, canonical frame.
"""
import json
import sys
from pathlib import Path
import numpy as np


class Art:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.ink = np.zeros_like(x)
        self.redink = np.zeros_like(x)

    def line(self, points, width=.07, red=False):
        points = np.asarray(points, dtype=float)
        d2 = np.full_like(self.x, 100.)
        for a, b in zip(points[:-1], points[1:]):
            vx, vy = b-a
            t = np.clip(((self.x-a[0])*vx+(self.y-a[1])*vy)/(vx*vx+vy*vy+1e-20), 0, 1)
            d2 = np.minimum(d2, (self.x-a[0]-t*vx)**2+(self.y-a[1]-t*vy)**2)
        self.put(d2 <= (width/2)**2, red)

    def put(self, mask, red=False):
        target = self.redink if red else self.ink
        np.maximum(target, np.asarray(mask, dtype=float), out=target)

    def poly(self, points, red=False):
        self.put(polygon(self.x, self.y, points), red)

    def ellipse(self, cx, cy, rx, ry=None, red=False):
        if ry is None:
            ry = rx
        self.put(((self.x-cx)/rx)**2+((self.y-cy)/ry)**2 <= 1, red)

    def box(self, x0, y0, x1, y1, red=False):
        self.put((self.x >= x0)&(self.x <= x1)&(self.y >= y0)&(self.y <= y1), red)

    def curve(self, control, width=.07, red=False):
        self.line(bezier(control), width, red)


def polygon(x, y, points):
    p = np.asarray(points, dtype=float)
    inside = np.zeros_like(x, dtype=bool)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        test = ((a[1] > y) != (b[1] > y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1]+1e-30)+a[0])
        inside ^= test
    return inside


def bezier(p, n=18):
    p = np.asarray(p, dtype=float)
    t = np.linspace(0, 1, n)[:, None]
    if len(p) == 3:
        return (1-t)**2*p[0]+2*t*(1-t)*p[1]+t*t*p[2]
    return (1-t)**3*p[0]+3*t*(1-t)**2*p[1]+3*t*t*(1-t)*p[2]+t**3*p[3]


def join(*parts):
    return np.concatenate(parts)


def digit_path(d):
    # Ordinary rounded, monoline road-sign numerals, not a font file.
    if d == '0':
        t = np.linspace(0, 2*np.pi, 49)
        return np.stack((.5+.39*np.cos(t), .5+.44*np.sin(t)), 1)
    if d == '1':
        return np.array([[.18,.23],[.57,.055],[.57,.95]])
    if d == '2':
        return join(bezier([[.10,.23],[.14,-.02],[.88,-.02],[.90,.25]]),
                    bezier([[.90,.25],[.91,.42],[.30,.71],[.10,.94]]), np.array([[.91,.94]]))
    if d == '3':
        return join(bezier([[.11,.16],[.46,-.08],[.95,.05],[.89,.28]]),
                    bezier([[.89,.28],[.86,.43],[.62,.49],[.40,.49]]),
                    bezier([[.40,.49],[1.10,.41],[1.02,1.10],[.12,.87]]))
    if d == '5':
        return join(np.array([[.88,.06],[.17,.06],[.14,.47]]),
                    bezier([[.14,.47],[.98,.18],[1.12,1.12],[.12,.88]]))
    if d == '6':
        return join(bezier([[.82,.08],[.24,-.13],[.07,.40],[.11,.70]]),
                    bezier([[.11,.70],[.16,1.07],[.91,1.07],[.90,.69]]),
                    bezier([[.90,.69],[.90,.35],[.20,.33],[.11,.70]]))
    if d == '7':
        return np.array([[.08,.06],[.93,.06],[.34,.95]])
    if d == '8':
        t = np.linspace(-np.pi/2, 3*np.pi/2, 37)
        top = np.stack((.5+.34*np.cos(t), .28+.23*np.sin(t)), 1)
        bot = np.stack((.5+.40*np.cos(t), .72+.24*np.sin(t)), 1)
        return None, top, bot
    raise ValueError(d)


def number(a, text, style=0):
    h = (1.02 if len(text) == 2 else 1.00)*((1.,1.12,.90)[style%3])
    w = (.52 if len(text) == 2 else .37)*((1.,1.06,.90)[style%3])
    gap = .065 if len(text) == 2 else .045
    total = len(text)*w+(len(text)-1)*gap
    sw = h*.12
    for j, d in enumerate(text):
        cx = -total/2+w/2+j*(w+gap)
        paths = digit_path(d)
        if isinstance(paths, tuple):
            paths = paths[1:]
        else:
            paths = [paths]
        for p in paths:
            q = p.copy()
            q[:,0] = (q[:,0]-.5)*w+cx
            q[:,1] = (q[:,1]-.5)*h
            a.line(q, sw)


def arrow(a, points, width=.18, head=.31):
    p = np.array(points, dtype=float)
    a.line(p, width)
    v = p[-1]-p[-2]
    v /= np.sqrt((v*v).sum())
    u = np.array([-v[1], v[0]])
    tip = p[-1]+v*.07
    base = p[-1]-v*head
    a.poly([tip, base+u*head*.90, base-u*head*.90])


def car(a, cx, cy, s=.35, red=False, frontal=False):
    if frontal:
        a.poly([(cx-s*.62,cy-s*.25),(cx-s*.44,cy-s*.72),(cx+s*.44,cy-s*.72),
                (cx+s*.62,cy-s*.25),(cx+s*.68,cy+s*.43),(cx-s*.68,cy+s*.43)], red)
        a.box(cx-s*.58,cy+s*.20,cx-s*.33,cy+s*.68,red)
        a.box(cx+s*.33,cy+s*.20,cx+s*.58,cy+s*.68,red)
    else:
        a.poly([(cx-s*.95,cy+s*.18),(cx-s*.95,cy-s*.13),(cx-s*.62,cy-s*.23),
                (cx-s*.38,cy-s*.62),(cx+s*.32,cy-s*.62),(cx+s*.64,cy-s*.23),
                (cx+s*.94,cy-s*.10),(cx+s*.94,cy+s*.18)],red)
        for t in [-.58,.58]:
            a.ellipse(cx+s*t,cy+s*.22,s*.19,red=red)


def truck(a, cx, cy, s=.7, red=False):
    a.box(cx-s*.80,cy-s*.38,cx+s*.26,cy+s*.23,red)
    a.poly([(cx+s*.28,cy-s*.25),(cx+s*.63,cy-s*.25),(cx+s*.87,cy+s*.02),
            (cx+s*.87,cy+s*.25),(cx+s*.28,cy+s*.25)],red)
    for t in [-.56,.56]:
        a.ellipse(cx+s*t,cy+s*.27,s*.16,red=red)


def person(a, cx, cy, s=1., walking=True, tilt=0):
    a.ellipse(cx+tilt*s,cy-.32*s,.105*s)
    a.line([(cx+tilt*s,cy-.18*s),(cx-.04*s,cy+.10*s)],.14*s)
    a.line([(cx-.04*s,cy+.10*s),(cx-.23*s,cy+.38*s)],.10*s)
    a.line([(cx-.04*s,cy+.10*s),(cx+.22*s,cy+.36*s)],.10*s)
    a.line([(cx,cy-.12*s),(cx-.22*s,cy+.02*s),(cx-.32*s,cy-.04*s)],.075*s)
    a.line([(cx+.01*s,cy-.10*s),(cx+.19*s,cy+.04*s),(cx+.30*s,cy-.05*s)],.075*s)


def bicycle(a, cx=0., cy=.22, s=.90):
    for xx in [-.38,.38]:
        t = np.linspace(0,2*np.pi,33)
        a.line(np.stack((cx+s*(xx+.235*np.cos(t)),cy+s*.235*np.sin(t)),1),.045*s)
    def p(v): return [(cx+s*x,cy+s*y) for x,y in v]
    a.line(p([[-.38,0],[-.14,-.38],[.06,0],[-.38,0]]),.047*s)
    a.line(p([[-.14,-.38],[.25,-.38],[.06,0]]),.047*s)
    a.line(p([[.38,0],[.22,-.51],[.37,-.51]]),.047*s)
    a.line(p([[-.25,-.43],[-.07,-.43]]),.055*s)


def stop_letters(a):
    # Simple geometric sans-serif capitals, in the standard word STOP.
    letters = {
        'S': [[(.9,.05),(.2,.05),(.05,.25),(.2,.47),(.8,.53),(.95,.75),(.8,.95),(.1,.95)]],
        'T': [[(.0,.05),(1.,.05)],[(.5,.05),(.5,.95)]],
        'O': [[(.2,.05),(.8,.05),(.95,.23),(.95,.77),(.8,.95),(.2,.95),(.05,.77),(.05,.23),(.2,.05)]],
        'P': [[(.1,.95),(.1,.05),(.75,.05),(.95,.2),(.95,.4),(.75,.52),(.1,.52)]]}
    w, h, gap = .32,.60,.045
    total = 4*w+3*gap
    for j,c in enumerate('STOP'):
        for path in letters[c]:
            p = np.asarray(path,dtype=float)
            p[:,0] = -total/2+j*(w+gap)+w*p[:,0]
            p[:,1] = (p[:,1]-.5)*h
            a.line(p,.065)


def icon(a, c, style):
    nums = {0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',6:'80',7:'100',8:'120'}
    if c in nums:
        number(a,nums[c],style)
    elif c in (9,10,41,42):
        if c in (10,42):
            # The overtaking prohibition uses a frontal truck, unlike the
            # side-view truck on the separate vehicle prohibition sign.
            a.box(-.58,-.43,-.08,.20,red=c==10)
            a.box(-.54,.12,-.41,.40,red=c==10)
            a.box(-.25,.12,-.12,.40,red=c==10)
            car(a,.34,.06,.38,frontal=True)
        else:
            car(a,-.32,.06,.38,red=c==9,frontal=True)
            car(a,.32,.06,.38,frontal=True)
        if c in (9,10):
            a.ink=np.maximum(a.ink,a.redink*.62)
    elif c == 11:
        arrow(a,[(0,.56),(0,-.44)],.17,.21)
        a.line([(-.46,.17),(0,-.06),(.46,.17)],.085)
    elif c == 14:
        stop_letters(a)
    elif c == 16:
        truck(a,0,.04,.69)
    elif c == 17:
        a.box(-.69,-.15,.69,.15)
    elif c == 18:
        a.poly([(-.08,-.48),(.08,-.48),(.055,.17),(-.055,.17)])
        a.ellipse(0,.39,.095)
    elif c in (19,20):
        sg = -1 if c == 19 else 1
        p = join(np.array([[0,.53],[0,.08]]),bezier([[0,.08],[0,-.21],[sg*.17,-.33],[sg*.43,-.33]]))
        a.line(p,.17)
    elif c == 21:
        p = join(bezier([[.13,.55],[.13,.30],[-.30,.26],[-.30,.0]]),
                 bezier([[-.30,.0],[-.30,-.27],[.28,-.25],[.28,-.49]]))
        a.line(p,.14)
    elif c == 22:
        p = join(np.array([[-.58,.42]]),bezier([[-.58,.42],[-.36,-.17],[-.12,-.17],[.02,.35]]),
                 bezier([[.02,.35],[.18,-.15],[.43,-.15],[.60,.42]]),np.array([[.60,.54],[-.58,.54]]))
        a.poly(p)
    elif c == 23:
        car(a,0,-.12,.40,frontal=True)
        for x in [-.20,.20]:
            a.line([(x,.16),(x-.14,.28),(x+.07,.42),(x-.08,.57)],.055)
    elif c == 24:
        a.line([[-.33,-.49],[-.33,.55]],.13)
        a.line([[.12,-.49],[.12,-.22],[.32,.19],[.32,.55]],.13)
    elif c == 25:
        a.ellipse(-.05,-.34,.095)
        a.line([[-.06,-.19],[-.24,.04],[-.07,.14]],.14)
        a.line([[-.08,.12],[-.34,.51]],.09)
        a.line([[-.08,.12],[.11,.50]],.09)
        a.line([[-.13,-.10],[.12,.08],[.30,.08]],.075)
        a.line([[.27,-.08],[.43,.47]],.045)
        a.poly([(.22,.53),(.48,.27),(.64,.53)])
    elif c == 26:
        a.box(-.17,-.51,.17,.54)
        # Holes in the housing form the familiar three signal lamps.
        for yy in [-.32,.01,.34]:
            hole = (a.x*a.x+(a.y-yy)**2)<.112**2
            a.ink[hole] = 0
        a.redink[(a.x*a.x+(a.y+.32)**2)<.10**2] = 1
    elif c == 27:
        person(a,0,.06,.95,tilt=.06)
        for yy in [.37,.48,.59]:
            a.line([[-.53,yy],[.53,yy]],.035)
    elif c == 28:
        person(a,-.21,.05,.84,tilt=.10)
        person(a,.28,.22,.62,tilt=.10)
    elif c == 29:
        bicycle(a,0,.22,.84)
    elif c == 30:
        for angle in [0,np.pi/3,2*np.pi/3]:
            v = np.array([np.cos(angle),np.sin(angle)])
            u = np.array([-v[1],v[0]])
            a.line([-v*.49,v*.49],.065)
            for sg in [-1,1]:
                joint = sg*v*.26
                for side in [-1,1]:
                    a.line([joint+sg*v*.14+side*u*.13,joint],.050)
    elif c == 31:
        # Leaping deer: horizontal body, long legs, upright neck and antlers.
        a.ellipse(-.04,.01,.30,.13)
        a.line([[.18,-.02],[.28,-.32],[.39,-.38]],.11)
        a.ellipse(.36,-.38,.13,.07)
        a.line([[.30,-.41],[.24,-.60],[.15,-.66]],.035)
        a.line([[.27,-.51],[.35,-.61]],.030)
        a.line([[-.25,.04],[-.39,.29],[-.57,.41]],.060)
        a.line([[-.18,.09],[-.15,.31],[-.34,.48]],.055)
        a.line([[.16,.07],[.35,.26],[.53,.27]],.055)
        a.line([[.20,.03],[.47,.08],[.53,.20]],.050)
        a.line([[-.28,-.03],[-.43,-.14]],.045)
    elif c in (33,34):
        sg = 1 if c == 33 else -1
        arrow(a,[(0,.60),(0,-.13),(sg*.52,-.13)],.19,.30)
    elif c == 35:
        arrow(a,[(0,.62),(0,-.61)],.19,.33)
    elif c in (36,37):
        sg = 1 if c == 36 else -1
        arrow(a,[(0,.61),(0,-.60)],.17,.29)
        arrow(a,[(0,.10),(sg*.54,.10)],.17,.28)
    elif c in (38,39):
        sg = 1 if c == 38 else -1
        arrow(a,[(-sg*.43,-.48),(sg*.44,.48)],.21,.33)
    elif c == 40:
        for rot in [0,2*np.pi/3,4*np.pi/3]:
            t = np.linspace(rot+.20,rot+1.65,18)
            p = np.stack((.43*np.cos(t),.43*np.sin(t)),1)
            arrow(a,p,.125,.22)
    # 12,13,15 and32 have no central pictogram.
    if c in (6,32,41,42):
        # Cancellation is five parallel black diagonals, rising to the right.
        for b in np.linspace(-.46,.46,5):
            a.put((np.abs((a.x+a.y)/np.sqrt(2)-b)<.027)&(a.x*a.x+a.y*a.y<.91**2))


def reduce2(x):
    return x.reshape(28,2,28,2).mean((1,3))


def dilate(x, radius):
    p = np.pad(x, radius)
    return np.maximum.reduce([p[dy:dy+28,dx:dx+28] for dy in range(2*radius+1) for dx in range(2*radius+1)])


def meanmask(x):
    return x/(x.sum()+1e-12)


def sobel_bank():
    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)/8
    return np.stack([np.cos(t)*sx+np.sin(t)*sx.T for t in np.arange(8)*np.pi/4])


def analytic_edges(luminance, region):
    # The same fixed derivative and integration formulas used in the network.
    p = np.pad(luminance,1,mode='edge')
    edges=[]
    for k in sobel_bank():
        e=sum(k[dy,dx]*p[dy:dy+28,dx:dx+28] for dy in range(3) for dx in range(3))
        e=np.maximum(e,0)
        q=np.pad(e,1)
        e=sum(q[dy:dy+28,dx:dx+28] for dy in range(3) for dx in range(3))/9
        edges.append(e)
    edges=np.stack(edges)
    edges-=edges.mean(0,keepdims=True)
    edges*=region[None]
    return edges/(np.sqrt(np.sum(edges*edges))+1e-12)


def stencil(c, radius, aspect, angle, style, family_only=False):
    # Evaluate ideal artwork at 2x sampling; integrate each 2x2 subpixel cell.
    q = (np.arange(56)+.5)/2-14
    xx, yy = np.meshgrid(q,q)
    t = angle*np.pi/180
    x = (np.cos(t)*xx+np.sin(t)*yy)/(radius*aspect)
    y = (-np.sin(t)*xx+np.cos(t)*yy)/radius
    if c == 11 or 18 <= c <= 31:
        a = Art(x/.92,(y-.12)/.92)
    else:
        a = Art(x,y)
    icon(a,c,style)
    rr = np.sqrt(x*x+y*y)
    if c in tuple(range(0,11))+(15,16,17,32,41,42) or 33 <= c <= 40:
        outer = rr<=1
        inner = rr<(.78 if 33 <= c <= 40 else .72)
        ring = outer & ~inner
        outside = (rr>1.02)&(rr<1.18)
    elif c == 12:
        d = np.abs(x)+np.abs(y)
        outer = d<1.18
        inner = d<.88
        ring = outer & ~inner
        outside = (d>1.20)&(d<1.42)
    elif c == 14:
        outer = (np.maximum(np.abs(x),np.abs(y))<.94)&((np.abs(x)+np.abs(y))<1.34)
        inner = (np.maximum(np.abs(x),np.abs(y))<.75)&((np.abs(x)+np.abs(y))<1.10)
        ring = outer & ~inner
        outside = ~outer & (np.maximum(np.abs(x),np.abs(y))<1.14)
    else:
        sy = -y if c == 13 else y
        outer = polygon(x,sy,[(0,-1),(-1,.85),(1,.85)])
        inner = polygon(x,sy,[(0,-.67),(-.72,.66),(.72,.66)])
        ring = outer & ~inner
        big = polygon(x,sy,[(0,-1.18),(-1.19,1.02),(1.19,1.02)])
        outside = big & ~outer
    ink = reduce2(a.ink*inner)
    redink = reduce2(a.redink*inner)
    outer, inner, ring, outside = map(reduce2,[outer,inner,ring,outside])
    W = np.zeros((13,28,28),dtype=np.float64)
    bias = 0.
    # Channels: clipped luminance, red, blue, yellow, 3x3 dark dilation.
    if c in (6,32,41,42):
        # Achromatic discs. Explicitly disfavor saturated red/blue paint.
        W[1] -= .58*meanmask(outer)
        W[2] -= .58*meanmask(outer)
        W[3] -= .20*meanmask(outer)
        W[0] += .16*meanmask(inner)-.16*meanmask(ring)
        bias += .02
    elif 33 <= c <= 40:
        W[2] += 1.02*meanmask(outer*(1-ink))-.43*meanmask(outside)-.20*meanmask(ink)
        W[1] -= .35*meanmask(outer)
        W[3] -= .20*meanmask(outer)
    elif c == 12:
        W[3] += 1.15*meanmask(inner)-.48*meanmask(outside)
        W[1] -= .35*meanmask(outer)
        W[2] -= .35*meanmask(outer)
        W[0] += .22*meanmask(ring)
        bias -= .10
    elif c in (14,17):
        W[1] += 1.02*meanmask(outer*(1-ink))-.46*meanmask(outside)-.20*meanmask(ink)
        W[2] -= .34*meanmask(outer)
    else:
        W[1] += 1.02*meanmask(ring)-.48*meanmask(inner*(1-redink))-.36*meanmask(outside)
        W[2] -= .32*meanmask(outer)
        if redink.sum() > 0:
            W[1] += .20*meanmask(redink)
    if family_only:
        return W.astype(np.float32), np.float32(bias)
    if ink.sum() > .25:
        # Balanced foreground/background correlation avoids a sparse-symbol
        # preference. A small halo allows an imprecise ideal stroke width.
        white = inner*(1-dilate(ink,1))*(1-redink)
        if white.sum() < 1:
            white = inner*(1-ink)*(1-redink)
        inverted = (33 <= c <= 40) or c in (14,17)
        signed = meanmask(ink)-meanmask(white)
        signed /= np.sqrt(np.sum(signed*signed)+1e-12)*radius
        if not inverted: signed = -signed
        W[0] += 1.38*signed
        lum = ink if inverted else 1-ink
        W[5:] = (5.0/radius)*analytic_edges(lum,inner)
        # Pigment itself is evidence for the left red car/truck.
        if redink.sum() > .25:
            W[1] += .48*meanmask(redink)
    elif c in (13,15):
        W[0] += .32*meanmask(inner)-.24*meanmask(ring)
        bias -= .05
    return W.astype(np.float32), np.float32(bias)


def build(folder):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    layers=[]
    weights={}
    def layer(name,op,source,**kw):
        layers.append(dict(name=name,op=op,inputs=[source] if isinstance(source,str) else source,**kw))
        return name
    def conv(name,source,W,b,pad=0,stride=1):
        W=np.asarray(W,dtype=np.float32)
        layer(name,'conv',source,out=W.shape[0],kernel=list(W.shape[2:]),padding=pad,stride=stride)
        weights[name+'.W']=W
        weights[name+'.b']=np.asarray(b,dtype=np.float32)
        return name
    # A single normalization over all RGB values preserves colour differences.
    layer('rgb_sheet','reshape','image',shape=[1,96,32])
    layer('illumination','norm','rgb_sheet')
    layer('rgb','reshape','illumination',shape=[3,32,32])
    axes=np.array([[.299,.587,.114],[1,-.5,-.5],[-.5,-.5,1],[.5,.5,-1]],np.float32)
    axes*=np.array([.70,1.25,1.25,1.05],np.float32)[:,None]
    lo=np.array([.5,-.10,-.08,-.12],np.float32)
    threshold=np.repeat(axes,2,axis=0)
    bias=np.repeat(lo,2)-np.tile([0,1],4)
    conv('thresholds','rgb',threshold[:,:,None,None],bias)
    layer('hinges','relu','thresholds')
    diff=np.zeros((4,8,1,1),np.float32)
    for j in range(4):
        diff[j,2*j,0,0]=1
        diff[j,2*j+1,0,0]=-1
    conv('features','hinges',diff,np.zeros(4))
    # Keep spatially aligned raw colour/luminance and a tolerant ink detector.
    raw=np.zeros((5,4,3,3),np.float32)
    for j in range(4): raw[j,j,1,1]=1
    conv('cropped_features','features',raw,np.zeros(5))
    dark=np.zeros((1,4,1,1),np.float32)
    dark[0,0,0,0]=-1
    conv('dark','features',dark,[1])
    layer('nearby_dark','maxpool','dark',size=[3,3],stride=[1,1])
    expand=np.zeros((5,1,1,1),np.float32)
    expand[4,0,0,0]=1
    conv('ink_channel','nearby_dark',expand,np.zeros(5))
    layer('base_descriptor','add',['cropped_features','ink_channel'])
    grad=np.zeros((8,4,3,3),np.float32)
    grad[:,0]=sobel_bank()
    conv('directions','features',grad,np.zeros(8))
    layer('positive_directions','relu','directions')
    smooth=np.zeros((13,8,3,3),np.float32)
    for j in range(8): smooth[5+j,j]=1/9
    conv('integrated_edges','positive_directions',smooth,np.zeros(13),pad=1)
    lift=np.zeros((13,5,1,1),np.float32)
    for j in range(5): lift[j,j,0,0]=1
    conv('appearance_channels','base_descriptor',lift,np.zeros(13))
    layer('descriptor','add',['appearance_channels','integrated_edges'])
    variants=[(r,a,t,s) for r in (8.,9.5,11.,12.5,14.) for a in (.91,1.06) for t in (-9.,0.,9.) for s in (0,1,2)]
    # A separate family branch gives a common coarse-shape score to classes
    # whose borders are identical. It cannot learn or memorize any examples.
    families=[0,11,12,13,14,17,32,35]
    ffilters=[]; fbias=[]
    geom=[(r,a,t) for r in (9.,11.,13.) for a in (.91,1.06) for t in (-9.,0.,9.)]
    for c in families:
        for r,a,t in geom:
            W,b=stencil(c,r,a,t,0,True)
            ffilters.append(W); fbias.append(b)
    conv('family_matches','descriptor',np.stack(ffilters),fbias,pad=3,stride=1)
    layer('family_positions','maxpool','family_matches',size=[9,9],stride=[9,9])
    layer('family_geometry','reshape','family_positions',shape=[len(families),len(geom),1])
    layer('family_best','maxpool','family_geometry',size=[len(geom),1],stride=[len(geom),1])
    gate=np.zeros((43,len(families)),np.float32)
    for c in range(43):
        f=0
        if c==11 or 18<=c<=31: f=1
        elif c==12: f=2
        elif c==13: f=3
        elif c==14: f=4
        elif c==17: f=5
        elif c in (6,32,41,42): f=6
        elif 33<=c<=40: f=7
        gate[c,f]=3.0
    layer('family_scores','dense','family_best',out=43)
    weights['family_scores.W']=gate
    weights['family_scores.b']=np.zeros(43,np.float32)
    filters=[]; biases=[]
    for c in range(43):
        for r,a,t,s in variants:
            W,b=stencil(c,r,a,t,s)
            filters.append(W); biases.append(b)
    conv('geometric_matches','descriptor',np.stack(filters),biases,pad=3,stride=1)
    layer('translation_invariance','maxpool','geometric_matches',size=[9,9],stride=[9,9])
    layer('class_variants','reshape','translation_invariance',shape=[43,len(variants),1])
    layer('artwork_invariance','maxpool','class_variants',size=[len(variants),1],stride=[len(variants),1])
    layer('artwork_scores','reshape','artwork_invariance',shape=[43])
    layer('scores','add',['artwork_scores','family_scores'])
    (folder/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(folder/'weights.npz',**weights)
    print('Built',len(layers),'layers;',sum(v.size for v in weights.values()),'analytic weights;',len(variants),'artwork variants per class')


if __name__ == '__main__':
    build(sys.argv[1])
