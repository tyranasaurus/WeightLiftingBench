#!/usr/bin/env python3
"""Data-free compiler for a geometric ASL network. No input files are used."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ABC = 'ABCDEFGHIKLMNOPQRSTUVWXY'
N = 28


class Net:
    def __init__(self):
        self.layers = []
        self.w = {}

    def op(self, name, op, *inputs, **settings):
        self.layers.append(dict(name=name, op=op, inputs=list(inputs), **settings))
        return name

    def affine(self, name, inp, w, b=0, padding=0):
        w = np.asarray(w, dtype=np.float32)
        self.w[name + '.W'] = w
        self.w[name + '.b'] = np.broadcast_to(b, (len(w),)).astype(np.float32)
        if w.ndim == 2:
            return self.op(name, 'dense', inp, out=len(w))
        return self.op(name, 'conv', inp, out=len(w), kernel=list(w.shape[-2:]),
                       stride=1, padding=padding)

    def relu(self, name, inp):
        return self.op(name, 'relu', inp)

    def shape(self, name, inp, *shape):
        return self.op(name, 'reshape', inp, shape=list(shape))

    def pool(self, name, inp, kind, size, stride):
        return self.op(name, kind + 'pool', inp, size=list(size), stride=list(stride))


def derivatives():
    y, x = np.mgrid[-2:3, -2:3].astype(float)
    gx = x * np.exp(-(x*x+y*y)/(2*.88**2))
    gx /= gx[gx > 0].sum()
    return gx, gx.T


ANGLES = np.arange(4)*math.pi/4
COS, SIN = np.cos(ANGLES), np.sin(ANGLES)


def strip_filters():
    y,x = np.mgrid[-2:3,-2:3]
    kernels = []
    for c,s in zip(COS,SIN):
        v,t = c*x+s*y, -s*x+c*y
        a = np.exp(-.5*(t/1.32)**2)
        mid = np.exp(-.5*(v/.65)**2)*a
        mid /= mid.sum()
        halves = []
        for shift in (-1.83,1.83):
            side = np.exp(-.5*((v-shift)/.60)**2)*a
            halves.append(mid-side/side.sum())
        kernels.append(halves)
    w = np.asarray(kernels)
    return np.concatenate([w,-w])


def background_matrix():
    # Look across scanlines: subtract a linear illumination field defined by
    # softly averaged side strips. Slope and intercept are input activations.
    w = np.eye(N*N, dtype=np.float32).reshape(N, N, N, N)
    a = ((np.arange(N)-3.)/(N-7)).clip(0, 1)
    for y in range(N):
        for x in range(N):
            anchor_left = min(2., max(0., x-5.))
            anchor_right = min(2., max(0., N-1-x-5.))
            for j, p in enumerate((.20, .60, .20)):
                l = int(anchor_left)+j
                r = N-1-int(anchor_right)-j
                w[y, x, y, l] -= (1-a[x])*p
                w[y, x, y, r] -= a[x]*p
    return w.reshape(N*N, N*N)


def frontend(net):
    g = net.op('brightness_invariant', 'norm', 'image')
    gx, gy = derivatives()
    d = COS[:, None, None]*gx + SIN[:, None, None]*gy
    e = net.affine('directional_derivatives', g, np.concatenate([d, -d])[:, None])
    e = net.relu('split_edge_polarities', e)
    # Rectify signed derivatives, then suppress response to orthogonal edges.
    mixing = np.zeros((8, 8, 1, 1))
    for k in range(4):
        p = (k+2) % 4
        mixing[k, [k, k+4], 0, 0] = 1
        mixing[k, [p, p+4], 0, 0] = -.40
    mixing[4:] = mixing[:4]
    e = net.affine('edge_floor_and_ceiling', e, mixing, [-.10]*4+[-1.05]*4)
    e = net.relu('bounded_edge_hinges', e)
    w = np.zeros((10, 8, 1, 1))
    wx = np.zeros_like(w)
    for k in range(4):
        w[k+2, [k, k+4], 0, 0] = (1, -1)
        wx[k+2, [k, k+4], 0, 0] = (.38, -.38)
    emax = net.affine('peak_oriented_edges', e, w*.62)
    emax = net.pool('edges_with_local_slack', emax, 'max', (2, 2), (2, 2))
    em = net.affine('mean_oriented_edges', e, wx)
    em = net.pool('cell_average_edges', em, 'avg', (2, 2), (2, 2))
    e = net.op('oriented_edge_cells','add',emax,em)

    # A slender digit has two edges with opposing gradients. Require similar
    # contrast on BOTH sides of the central strip, not just a strong step.
    pairs = strip_filters()
    d = net.affine('opposite_sides_of_strips', g, pairs.reshape(16,1,5,5))
    d = net.relu('positive_strip_halves', d)
    z = np.zeros((16,16,1,1))
    for i in range(8):
        z[i, 2*i,0,0] = 1
        z[8+i, 2*i:2*i+2,0,0] = (1,-1)
    delta = net.affine('strip_half_differences', d, z)
    delta = net.relu('one_sided_strip_response', delta)
    z = np.zeros((10,16,1,1))
    for i in range(4):
        z[i+6, [i,i+4,i+8,i+12],0,0] = (1,1,-1,-1)
    q = net.affine('two_sided_ridge_response', delta, z, [-.075]*10)
    q = net.relu('ridge_contrast_floor',q)
    q = net.pool('ridge_cells', q, 'max', (2,2), (2,2))
    # Reject dots and corners: pool only along the tangent to a strip. A
    # partly broken ridge still contributes, but two adjacent cells agree.
    k = np.zeros((10,10,3,3),dtype=np.float32)
    k[np.arange(10),np.arange(10),1,1] = 1.
    for i,(dy,dx) in enumerate(((1,0),(1,-1),(0,1),(1,1)),6):
        k[i,i,1,1] = .55
        k[i,i,1+dy,1+dx] = .225
        k[i,i,1-dy,1-dx] = .225
    q = net.affine('tangential_ridge_support',q,k,padding=1)

    m = net.affine('subtract_scanline_background', g, background_matrix())
    m = net.shape('contrast_field', m, 1, N, N)
    yy, xx = np.mgrid[-2:3, -2:3]
    sm = np.exp(-(xx*xx+yy*yy)/(2*.65**2))
    sm /= sm.sum()
    m = net.affine('contrast_floor_and_ceiling', m, np.stack([sm, -sm, sm, -sm])[:, None],
                   [-.19, -.19, -1.50, -1.50])
    m = net.relu('bounded_contrast_hinges', m)
    w = np.zeros((10, 4, 1, 1))
    w[0, :, 0, 0] = (1, 1, -1, -1)  # even under change of contrast polarity
    w[1, :, 0, 0] = (1, -1, -1, 1)
    m = net.affine('even_and_odd_contrast', m, w)
    m = net.pool('regional_contrast', m, 'avg', (2, 2), (2, 2))
    return net.op('hand_evidence', 'add', m, e, q)


# Coordinates below are palm units: radial is +x, toward the fingers is +y.
# A row of finger joints is a projected skeleton, not a picture or contour.
# Depth is only used to decide which overlapping surface is visible.
FINGER_X = (-.36, -.14, .085, .30)  # little, ring, middle, index
KNUCKLE_Y = (.34, .45, .52, .48)
LENGTH = (.68, .91, 1.00, .88)
GIRTH = (.100, .111, .120, .118)


class Hand:
    def __init__(self, width=.465, height=.535, center=(0., -.045), wrist=(-.18, -1.0)):
        self.palm = (center, width, height)
        self.paths = []
        self.wrist = wrist
        self.topology = None

    def chain(self, pts, radii, depth=.2):
        self.paths.append((np.asarray(pts, dtype=float), np.broadcast_to(radii, (len(pts),)), depth))

    def extended(self, i, angle, length=1., shift=0., flare=0., depth=.20):
        x, y, l, r = FINGER_X[i]+shift, KNUCKLE_Y[i], LENGTH[i]*length, GIRTH[i]
        pts = [(x, y-.12)]
        for t in (.35, .69, 1.):
            a = math.radians(angle+flare*t)
            pts.append((x+l*t*math.sin(a), y-.10+l*t*math.cos(a)))
        self.chain(pts, (1.04*r, .98*r, .84*r, .77*r), depth)

    def curled(self, i, rise=0., drop=.20, lean=0., depth=.255, gain=1.):
        x, y, r = FINGER_X[i], KNUCKLE_Y[i]+rise, GIRTH[i]*gain
        self.chain([(x, y-.23), (x+lean*.45, y+.015), (x+lean, y+.045)], (r, r, .98*r), depth)
        # The visible front face ends at the folded fingertip, below the knuckle.
        self.chain([(x+lean, y+.044), (x+lean+.012, y-.05), (x+lean+.035, y-drop)],
                   (r*.96, r*.98, r*.80), depth+.079)

    def thumb(self, pts, r=(.19, .146, .105), depth=.41):
        self.chain(pts, r, depth)


def posture(letter, u, v, w, family):
    """Four generic views per letter, plus continuous articulation parameters."""
    tilt = 0.
    spread = 1. + .24*u
    if letter in 'AEMNS':
        h = Hand(width=.465+.032*v, height=.527, wrist=(-.17+.16*w, -1.))
        for i in range(4):
            rise = .026*v*(i-1.5)
            drop, lean, d = .175, .014*u, .26
            if letter == 'S':
                rise -= .020 + .013*i
            if letter == 'A':
                rise += .022*i
                drop = .18+.035*w
            if letter == 'E':
                rise += .045
                drop = .33+.060*u
            if letter in 'MN':
                # Spreading the pertinent knuckle interval follows the thumb's exit.
                gap = 1 if letter == 'M' else 2
                rise += .028 if i >= gap else -.02
                lean += .045 if i >= gap else -.035
                drop = .218+.026*v
                d = .32 if i >= gap else .25
            if letter == 'S':
                lean -= .031
                drop = .22
            h.curled(i, rise, drop, lean, d)
        if letter == 'A':
            h.thumb([(.23,-.34), (.435,-.055), (.468+.05*u,.45+.095*v)], (.208,.151,.117))
        elif letter == 'E':
            h.thumb([(.29,-.34), (.20,-.082), (-.18+.06*v,.105+.055*u)], (.199,.142,.108), .392)
        elif letter in 'MN':
            x = -.22 if letter == 'M' else -.038
            h.thumb([(.26,-.35), (.13,-.077), (x+.050*u,.37+.06*v)], (.205,.151,.092), .346)
        elif letter == 'S':
            if family in (1,3):
                h.thumb([(.30,-.32), (.192,-.008), (-.07+.054*u,.49+.056*v)], (.192,.15,.104), .49)
            else:
                h.thumb([(.32,-.28), (.174,.049), (-.166+.071*u,.305+.087*v)], (.193,.158,.111), .49)
        tilt = -13+17*w
        return h, tilt

    if letter in 'CO':
        h = Hand(width=.34+.040*v, height=.53, center=(-.20, -.09), wrist=(-.49+.15*w, -1.))
        if letter == 'C':
            gap = .12*v
            h.chain([(-.37,.18), (-.30,.68), (-.05,.88), (.32,.90), (.58+.08*u,.77+gap)],
                    (.24,.225,.204,.172,.123), .243)
            h.chain([(-.24,.25), (-.13,.53), (.08,.72), (.36,.73), (.515+.06*u,.64+gap)],
                    (.21,.189,.167,.139,.104), .338)
            h.thumb([(-.205,-.42), (.20,-.25), (.61+.09*u,-.02-.11*v)], (.234,.171,.12))
        else:
            h.chain([(-.36,.17), (-.31,.54), (-.095,.779), (.26,.765), (.49+.035*u,.57), (.55+.055*u,.36)],
                    (.238,.227,.192,.161,.125,.109), .25)
            h.chain([(-.266,.20), (-.126,.52), (.064,.68), (.32,.63), (.493+.06*u,.40)],
                    (.21,.165,.147,.119,.103), .33)
            h.thumb([(-.24,-.43), (.18,-.27), (.49+.05*u,-.009), (.551+.052*u,.358)],
                    (.239,.194,.144,.117))
        tilt = -8+11*w
        return h, tilt

    if letter in 'GHT':
        h = Hand(width=.465, height=.423+.048*v, center=(-.12,-.02), wrist=(-.87+.20*v, -1.0))
        t = -.038+.12*u+(.16 if family == 1 else 0.)
        if letter == 'G':
            h.chain([(-.40,-.02), (-.23,.23), (.07,.335), (.33,.36+t*.48), (.73,.38+t), (1.06+.11*v,.36+t)],
                    (.251,.216,.153,.106,.088,.075), .303)
        else:
            h.chain([(-.40,-.02), (-.23,.29), (.07,.38), (.33,.41+t*.48), (.69,.41+t), (.94,.40+t)],
                    (.251,.222,.162,.125,.105,.085), .303)
        if letter == 'G':
            h.thumb([(-.25,-.23), (.077,-.075), (.49+.30*v,.16+t*.7)], (.188,.138,.094))
            h.chain([(-.25,-.28), (.013,-.24), (.20+.07*u,-.141)], (.160,.145,.111), .34)
        elif letter == 'H':
            h.chain([(-.35,.079), (.077,.207), (.40,.221+t*.52), (.71,.213+t), (.925+.042*v,.205+t)],
                    (.193,.156,.122,.101,.084), .381)
            h.thumb([(-.23,-.308), (.092,-.16), (.347+.14*v,.042)], (.179,.123,.106), .425)
        else:
            # A side-on tucked-thumb fist: the bent finger row is foreshortened
            # and the thumb rises on the radial side of the row.
            h.paths[0][0][-2:, 0] -= (.07, .14)
            h.thumb([(-.273,-.10), (-.068,.31), (.038+.072*v,.665+.10*u)], (.187,.130,.085), .346)
            h.chain([(-.22,-.32), (.026,-.22), (.32,.032)], (.177,.153,.121), .373)
        tilt = -6+21*w
        return h, tilt

    if letter == 'P':
        h = Hand(width=.40+.035*v, height=.459, center=(.04,-.056), wrist=(.63+.21*w,-1.))
        dy = .22*u+(.18 if family >= 2 else -.08)
        h.chain([(.13,.15), (-.17,.382), (-.45,.50+dy*.4), (-.74,.58+dy*.78), (-1.04,.65+dy)],
                (.187,.139,.118,.096,.073), .294)
        h.chain([(.07,.16), (-.17,.088), (-.32,-.13), (-.49-.11*v,-.37+.13*u)],
                (.170,.138,.108,.085), .32)
        h.thumb([(.27,-.23), (.015,-.16), (-.267-.1*v,.001)], (.163,.128,.112), .448)
        h.chain([(.25,.16), (.38,.016), (.311,-.19)], (.13,.14,.11), .359)
        tilt = 10*w
        return h, tilt

    if letter == 'Q':
        side = -1 if family == 2 else 1
        h = Hand(width=.428, height=.407, center=(0,.26), wrist=(-.81*side+.12*w,-1.))
        h.chain([(-.15*side,.37), (.24*side,.39), (.41*side,.13), ((.52+.15*u)*side,-.20), ((.68+.10*u)*side,-.42+.105*v)],
                (.277,.207,.163,.126,.089), .28)
        h.chain([(-.19*side,.29), (-.22*side,.018), (-.36*side,-.21)], (.177,.144,.103), .35)
        h.thumb([(-.043*side,.23), (.02*side,-.039), ((.11-.085*v)*side,-.348-.10*u)],
                (.181,.124,.089), .414)
        tilt = 6+16*w
        return h, tilt

    if letter == 'X':
        h = Hand(width=.39+.042*v, height=.493, center=(-.04,-.036), wrist=(-.4+.17*w,-1.))
        for i in range(3):
            h.curled(i, -.03, .177, .035*u, .255)
        h.chain([(.21,.24), (.144,.56), (.176,.829), (.354+.09*u,.938+.078*v), (.519+.12*u,.820+.078*v)],
                (.151,.128,.11,.1,.078), .358)
        h.thumb([(.22,-.325), (.323,-.03), (.403+.04*v,.328)], (.19,.148,.104), .439)
        tilt = -10+12*w
        return h, tilt

    # Upright handshapes, based on which digits are extended and the placement
    # of the thumb (opposed across the palm, in contact, or abducted).
    narrow = letter in 'DRUKIF'
    h = Hand(width=(.404 if narrow else .455)+.036*v,
             height=.521, wrist=(-.15+.15*w,-1.0))
    ext = dict(B={0:-8,1:-3,2:1,3:6}, D={3:4}, F={0:-25,1:-9,2:9}, I={0:-6},
               K={2:-29,3:7}, L={3:3}, R={2:14,3:-16}, U={2:3,3:-2},
               V={2:-18,3:19}, W={1:-20,2:-2,3:19}, Y={0:-20})[letter]
    for i in ext:
        a = ext[i]*spread + 3*v
        lng, shift, flare, depth = 1.+.071*w, 0., 2*u, .25
        if letter == 'K' and i == 2:
            lng *= .62+.225*v+(.12 if family == 2 else 0.)
        if letter == 'U':
            shift = .018 if i == 2 else -.016
        if letter == 'R':
            # Tips can cross in projection or appear nearly superimposed.
            if family in (1,3):
                a *= .54
                shift += .040 if i == 2 else -.027
            else:
                lng *= .91 if i == 3 else 1.
            flare += -2 if i == 3 else 3
            depth += .052 if i == 3 else 0.
        if letter in 'VW' and i == min(ext):
            lng *= 1.01+.101*u
        if letter == 'W' and family in (1,3):
            a += 6*(i-2)
        h.extended(i, a, lng, shift, flare, depth)
    for i in range(4):
        if i not in ext and letter != 'F':
            if letter == 'D':
                bend = .07*v if family in (0,2) else (-.07+.04*v)
                h.chain([(FINGER_X[i],.13), (FINGER_X[i]+.013,.417+.06*i+bend),
                         (FINGER_X[i]+.155,.616+.018*i+bend), (FINGER_X[i]+.26,.584+.017*i+bend)],
                        (GIRTH[i]*1.2,GIRTH[i],GIRTH[i]*.91,GIRTH[i]*.79), .268+i*.019)
            else:
                h.curled(i, -.015+.038*v, .187+.03*u, .018*w, .27)
    if letter == 'L':
        h.thumb([(.24,-.35), (.52,-.205), (.97+.143*v,.062+.107*u)], (.20,.158,.105))
    elif letter == 'Y':
        h.thumb([(.21,-.35), (.50,-.05), (.984+.132*v,.48+.147*u)], (.207,.149,.091))
    elif letter == 'F':
        h.chain([(.16,.21), (.318,.473), (.547,.606+.078*w), (.717+.086*v,.486+.068*w),
                 (.718+.086*v,.329+.045*w)], (.143,.121,.105,.091,.081), .33)
        h.thumb([(.153,-.331), (.434,-.135), (.661+.086*v,.089), (.718+.086*v,.34+.045*w)],
                (.198,.146,.119,.091))
    elif letter == 'D':
        h.thumb([(.242,-.326), (.351,-.047), (.248,.277), (.106+.064*u,.504+.052*v)],
                (.204,.163,.13,.098), .42)
    elif letter == 'K':
        h.thumb([(.243,-.317), (.265,.099), (.103+.056*u,.596+.102*v)], (.203,.148,.109), .418)
    elif letter == 'B':
        h.thumb([(.238,-.352), (.223,-.063), (-.14+.11*u,.244+.090*v)], (.206,.14,.098), .425)
    else:
        h.thumb([(.245,-.348), (.173,-.041), (-.135+.074*u,.245+.086*v)], (.201,.153,.105), .435)
    tilt = (-5 if letter in 'DKRUVW' else 0)+13*w
    return h, tilt


GRID_Y, GRID_X = np.mgrid[:N, :N].astype(float)
FY, FX = np.mgrid[2.5:26.5:2., 2.5:26.5:2.]
# The crop edge and wrist exits are unreliable places to compare contours.
TRUST = (1.-.61*np.clip((FY-19.)/7., 0., 1.)) * (1.-.29*(np.abs(FX-13.5)/11.)**3)
TRUST /= TRUST.sum()


def bounding_box(h, stretch):
    (x, y), a,b = h.palm
    lo = np.array([(x-a)*stretch, y-b]); hi = np.array([(x+a)*stretch, y+b])
    for pts, radii, _ in h.paths:
        rr = radii[:, None] * np.array([stretch, 1.])[None]
        p = pts * [stretch, 1.]
        lo = np.minimum(lo, (p-rr).min(0))
        hi = np.maximum(hi, (p+rr).max(0))
    return lo, hi


def render(h, angle, stretch, zoom, dx, dy):
    # Orthographic projection with crop translation, roll and foreshortening.
    lo, hi = bounding_box(h, stretch)
    span = hi-lo
    scale = zoom * min(22./span[0], 21.7/span[1])
    top = max(1.55, (27.-span[1]*scale)*.41-1.05)
    if span[0] > 1.40*span[1] and h.wrist[0] < -.60:
        top -= .8
    angle = math.radians(angle)
    c, s = math.cos(angle), math.sin(angle)
    X, Y = (GRID_X-13.5-dx)/scale, -(GRID_Y-top-dy)/scale+span[1]/2
    x = (c*X-s*Y+(lo[0]+hi[0])/2)/stretch
    y = s*X+c*Y+(lo[1]+hi[1])/2
    (cx,cy), a,b = h.palm
    q = np.sqrt(((x-cx)/a)**2+((y-cy)/b)**2)
    sd = (1.-q)*min(a,b)
    antialias = .86/scale
    # Depth and local diffuse illumination of ellipsoids and tapered capsules.
    # Difference of surfaces inside the silhouette gives crease/ridge cues.
    zmax = np.where(sd>-antialias, .056+.103*np.sqrt(np.maximum(0.,1.-q*q)), -10.)
    shade = .47+.31*np.sqrt(np.maximum(0.,1.-q*q))-.12*(x-cx)/a+.055*(y-cy)/b

    palm = np.array([cx,cy])
    exit = np.asarray(h.wrist)
    # The wrist is a separate taper; it is not used to set hand crop scale.
    wrist_pts = np.stack([palm+.38*(exit-palm), exit, palm+2.55*(exit-palm)])
    for pts, rr, depth in [(wrist_pts, (.334,.305,.337), .043)] + h.paths:
        for k in range(len(pts)-1):
            vx, vy = pts[k+1]-pts[k]
            ox, oy = x-pts[k,0], y-pts[k,1]
            t = np.clip((vx*ox+vy*oy)/(vx*vx+vy*vy), 0., 1.)
            ox, oy = ox-t*vx, oy-t*vy
            d = np.sqrt(ox*ox+oy*oy)
            r = rr[k]+(rr[k+1]-rr[k])*t
            local_sd = r-d
            nr = np.sqrt(np.maximum(0., 1.-(d/r)**2))
            z = depth + .20*r*nr
            vis = (local_sd>-antialias*.40) & (z>zmax)
            zmax = np.where(vis, z, zmax)
            paint = .48 + .29*nr + .103*ox/r + .073*oy/r
            shade = np.where(vis, paint, shade)
            sd = np.maximum(local_sd, sd)

    alpha = np.clip(.5+.50*sd/antialias, 0., 1.)
    return alpha, alpha*np.clip(shade, .15, .95)


def blocks2(a, fn):
    return fn(fn(a[..., 2:-2, 2:-2].reshape(*a.shape[:-2], 12, 2, 12, 2), axis=-1), axis=-2)


def edges_for(field):
    patches = np.lib.stride_tricks.sliding_window_view(field, (5, 5))
    d = np.einsum('ijab,kab->kij', patches, np.asarray(derivatives()))
    dx,dy = d
    l = np.abs(COS[:,None,None]*dx+SIN[:,None,None]*dy)
    l = np.maximum(l-.40*l[[2,3,0,1]],0.).reshape(4,12,2,12,2)
    return .62*l.max(axis=(2,4)) + .38*l.mean(axis=(2,4))


def ridges_for(field):
    patches = np.lib.stride_tricks.sliding_window_view(field, (5,5))
    pair = np.einsum('yxij,kij->kyx', patches, STRIPS)
    pos = np.maximum(pair.reshape(4,2,24,24).min(axis=1),0.)
    neg = np.maximum(-pair.reshape(4,2,24,24).max(axis=1),0.)
    return (pos+neg).reshape(4,12,2,12,2).max(axis=(2,4))


STRIPS = strip_filters()[:4].reshape(8,5,5)


def smear(a):
    p = np.pad(a, [(0,0)]*(a.ndim-2)+[(1,1),(1,1)], mode='edge')
    return (.44*p[...,1:-1,1:-1]+.11*(p[...,1:-1,:-2]+p[...,1:-1,2:]
                                         +p[...,:-2,1:-1]+p[...,2:,1:-1])
            +.03*(p[...,:-2,:-2]+p[...,2:,2:]+p[...,:-2,2:]+p[...,2:,:-2]))


def free_space_continuation(mask):
    """Search for an unmodeled digit just outside the plausible silhouette.

    A fixed narrow receptive halo ignores objects far from the hand. A
    digit-shaped response is negative evidence only if the hand model
    leaves adequate free space across the strip at that point.
    """
    wide = smear(smear(smear(mask)))
    empty = np.clip((.53-smear(mask))*2.9,0.,1.)
    near = np.clip((wide-.023)*6.1,0.,1.)
    f = near*empty*TRUST
    tangent = np.exp(-.5*((FX-13.5)/8.0)**4)
    f *= tangent*np.clip((22.5-FY)/5.7,.13,1.)
    return f / max(.18, f.sum())


def matched_weights(alpha, shade):
    mask = blocks2(alpha, np.mean)
    mean = float((TRUST*mask).sum())
    m = 2.*TRUST*(mask-mean)/math.sqrt(max(.06, mean*(1.-mean)))
    # The model needs a domain outside as well as inside the fingers. This
    # term reacts to stray structures near gaps, rather than to the borders.
    free = free_space_continuation(mask)
    in_hand = TRUST*mask
    in_hand /= max(.07,in_hand.sum())
    m += .37*(in_hand-free)

    # Pools on a real image give the dominant oriented edge in each cell.
    # Contrast a prediction with its blurred surroundings, so a response to
    # texture everywhere is not direct evidence for a contour at this spot.
    contour = edges_for(alpha)
    detail = edges_for(shade)
    detail *= np.clip((mask-.64)*3.0, 0., 1.)
    ec = .79*contour+.21*smear(contour)
    ed = .80*detail+.20*smear(detail)
    ep = (ec - .70*smear(smear(ec)))*TRUST
    ip = (ed - .83*smear(smear(ed)))*TRUST
    e = ep/max(.036, (ec*TRUST).sum())
    e += .24*ip/max(.034, (ed*TRUST).sum())

    return m,e,contour,detail,mask,ridges_for(alpha)


def radical_inverse(j, base):
    result, f = 0., 1./base
    while j:
        j,r = divmod(j,base)
        result += r*f
        f /= base
    return result


def projection(h,angle,stretch,zoom,dx,dy):
    lo,hi = bounding_box(h,stretch)
    span = hi-lo
    scale = zoom*min(22./span[0],21.7/span[1])
    top = max(1.55, (27.-span[1]*scale)*.41-1.05)
    if span[0] > 1.40*span[1] and h.wrist[0] < -.60:
        top -= .8
    a = math.radians(angle)
    c,s = math.cos(a), math.sin(a)
    def forward(p):
        x,y = p*np.array([stretch,1.])-(lo+hi)/2
        return np.array([13.5+dx+scale*(c*x+s*y), top+dy+scale*(span[1]/2+s*x-c*y)])
    return forward


def landmark_parts(letter,h,project):
    """Distinct contact regions or exposed extensions with separate votes."""
    out = []
    def part(i, t=.78, a=1.8, kind='tip'):
        pts = h.paths[i][0]
        p0,p1 = project(pts[0]), project(pts[-1])
        out.append((p0*(1-t)+t*p1,a,kind))
    def pos(x,y,a=2.6,kind='outline'):
        out.append((project(np.array([x,y])),a,kind))
    if letter in 'AEMNS':
        part(-1,.84,1.55,'interior')
        pos(-.13,.47,2.4)
        pos(.42,.22,1.85)
    elif letter == 'C':
        part(0,.86,2.0)
        part(-1,.90,2.1)
        pos(-.46,.28,2.1)
    elif letter == 'O':
        pos(.20,.42,2.7,'hole')
        part(-1,.90,2.0,'outline')
        pos(-.44,.34,2.5)
    elif letter in 'GHT':
        part(0,.88,2.35)
        part(1,.88,2.1,'tip' if letter=='T' else 'outline')
        pos(-.09,-.13,2.7)
    elif letter == 'P':
        part(0,.80,2.2)
        part(1,.88,2.1)
        pos(.18,.09,2.6)
    elif letter == 'Q':
        part(0,.90,2.45)
        part(-1,.88,2.2)
        pos(.05,.48,2.5)
    elif letter == 'X':
        part(-2,.89,2.1,'tip')
        pos(.21,.69,2.0,'outline')
        pos(-.23,.37,2.8)
    elif letter == 'F':
        part(0,.83,1.8)
        part(2,.83,1.8)
        pos(.55,.34,2.6,'hole')
    elif letter in 'BW':
        n = 4 if letter == 'B' else 3
        for i in (0, (n-1)//2, n-1):
            part(i,.83,1.9)
    elif letter == 'V':
        part(0,.84,1.9)
        part(1,.85,1.9)
        pos(.08,.53,2.5)
    elif letter in 'LY':
        part(0,.80,2.0)
        part(-1,.85,2.15)
        pos(-.08,.21,2.7)
    elif letter in 'KRU':
        part(0,.82,1.95)
        part(1,.85,2.)
        pos(.07,.66,2.15, 'interior' if letter in 'KR' else 'outline')
    else:  # D, I: one vertical extension, plus the curled digits and thumb
        part(0,.80,2.0)
        pos(-.14,.44,2.2)
        part(-1,.88,1.9,'interior' if letter=='D' else 'outline')
    return out


def part_weights(part, contour, detail, mask, ridge):
    (px,py), sigma, kind = part
    u = np.exp(-.5*(((FX-px)/sigma)**2+((FY-py)/sigma)**2))*TRUST
    u /= max(u.sum(),1e-8)
    w = np.zeros((10,12,12))
    m = u*(mask-(u*mask).sum())
    w[0] = .25*m/max(.125,np.abs(m).sum())
    ec = (.72*contour+.28*smear(contour))*u
    if kind == 'interior':
        ev = (.77*detail+.23*smear(detail))*u
        w[2:6] = .44*ec/max(.038,ec.sum()) + .44*ev/max(.063,ev.sum()) - .025*u
    else:
        w[2:6] = (.69 if kind=='tip' else .78)*ec/max(.038,ec.sum()) - .025*u
    # Opposite edges of a ridge count as a single part. Flat skin is not an
    # expected ridge; give edge evidence preference there instead.
    if kind in ('tip','hole'):
        rv = (.74*ridge+.26*smear(ridge))*u
        w[6:] = .32*rv/max(.034,rv.sum())
    return w


def templates(net, inp):
    # A fixed low-discrepancy cubature of plausible pose and crop coordinates,
    # shared by every sign. Not selected or fitted to any collection of images.
    variants, parts = 816, 4
    w = np.empty((24,variants,parts,10,12,12), dtype=np.float32)
    b = np.empty((24,variants,parts), dtype=np.float32)
    for k, letter in enumerate(ABC):
        for j in range(variants):
            coords = [2*radical_inverse(j+1,p)-1 for p in (3,5,7,11,13,17,19,23)]
            u,v,r,z,x,y,wr,f = coords
            family = j % 4
            h, angle = posture(letter,u,v,r,family)
            angle += (12. if letter in 'GHPT' else 7.)*f
            stretch,zoom,dx,dy = 1+.20*wr,1+.22*z,4.2*x,4.1*y
            alpha, shade = render(h,angle,stretch,zoom,dx,dy)
            m,e,contour,detail,mask,ridge = matched_weights(alpha, shade)
            w[k,j,0,0] = .72*m
            w[k,j,0,1] = 0
            w[k,j,0,2:6] = 1.05*e
            # Coverage term: missing modeled ridges cost a small amount and
            # extra ridges in the neighboring predicted free space cost more.
            # A shared scale keeps flash- or sleeve-induced edges bounded.
            rmap = (.60*ridge+.40*smear(ridge))*TRUST
            rmap /= max(.024,rmap.sum())
            free = free_space_continuation(mask)
            w[k,j,0,6:] = .42*rmap - 1.14*free[None]/4.
            b[k,j,0] = -.031*(x*x+y*y)-.028*z*z-.014*r*r
            project = projection(h,angle,stretch,zoom,dx,dy)
            for t,part in enumerate(landmark_parts(letter,h,project),1):
                w[k,j,t] = part_weights(part,contour,detail,mask,ridge)
                b[k,j,t] = -.21
    # Saturation limits a very strong local match to one vote, so it cannot
    # substitute for an absent second finger, curled-finger group or thumb.
    g = net.affine('whole_hand_match', inp, w[:,:,0].reshape(-1,1440), b[:,:,0].ravel())
    g = net.shape('whole_hand_by_pose', g, 24, variants, 1)
    v = w[:,:,1:].reshape(-1,1440)
    bias = b[:,:,1:].ravel()
    a = net.affine('landmark_evidence', inp, v, bias)
    a = net.shape('landmarks_by_pose',a,1,24,variants*3)
    hh = net.affine('landmark_hinges',a,np.ones((2,1,1,1)),[0.,-.30])
    hh = net.relu('rectified_landmark_hinges',hh)
    clip = net.affine('bounded_landmark_votes',hh,np.array([1.85,-1.85]).reshape(1,2,1,1))
    avg = net.pool('pose_votes',clip,'avg',(1,3),(1,3))
    avg = net.shape('local_votes_by_pose',avg,24,variants,1)
    n = net.op('aggregate_pose_evidence','add',g,avg)
    n = net.pool('best_compatible_pose',n,'max',(variants,1),(variants,1))
    return net.shape('letter_scores',n,24)


def main(directory):
    net = Net()
    features = frontend(net)
    templates(net, features)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':net.layers}, indent=2)+'\n')
    np.savez(out/'weights.npz', **net.w)


if __name__ == '__main__':
    main(sys.argv[1])
