"""Compile analytic road-sign detectors to the eight permitted network primitives.

All geometry below is dimensionless, in units of the sign's half-width. No
images, fonts, numerical fitting, or other external resources are used.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np


class Network:
    def __init__(self):
        self.layers, self.weights = [], {}

    def op(self, name, op, *inputs, **settings):
        self.layers.append(dict(name=name, op=op, inputs=list(inputs), **settings))
        return name

    def linear(self, name, inp, w, b=0, stride=1, padding=0):
        w = np.asarray(w, dtype=np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.broadcast_to(b, (len(w),)).astype(np.float32).copy()
        if w.ndim == 2:
            return self.op(name, 'dense', inp, out=len(w))
        return self.op(name, 'conv', inp, out=len(w), kernel=list(w.shape[-2:]), stride=stride, padding=padding)

    def mix(self, name, inp, w, b=0):
        return self.linear(name, inp, np.atleast_2d(w)[:, :, None, None], b)

    def shape(self, name, inp, *shape):
        return self.op(name, 'reshape', inp, shape=list(shape))

    def relu(self, name, inp):
        return self.op(name, 'relu', inp)

    def pool(self, name, inp, kind, h, w, sh, sw):
        return self.op(name, kind, inp, size=[h, w], stride=[sh, sw])

    def write(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'network.json').write_text(json.dumps({'layers': self.layers}, indent=2) + '\n')
        np.savez(path / 'weights.npz', **self.weights)


@lru_cache(maxsize=None)
def gaussian1(sigma):
    n = int(np.ceil(2.7 * sigma))
    t = np.arange(-n, n + 1)
    u = np.exp(-.5 * (t / sigma)**2)
    return u / u.sum()


def blur(a, sigma):
    u = gaussian1(sigma)
    p = len(u)//2
    h,w = a.shape[-2:]
    if a.ndim != 2:
        raise ValueError('blur expects a 2-D mask')
    x = np.pad(a, ((0,0),(p,p)))
    x = sum(g*x[:,i:i+w] for i,g in enumerate(u))
    y = np.pad(x, ((p,p),(0,0)))
    return sum(g*y[i:i+h,:] for i,g in enumerate(u))


def feature_layers(n):
    # Piecewise-linear log of RGB: exposure acts mainly as an additive offset.
    k = np.array([0., .012, .025, .045, .075, .119, .18, .265, .38, .53, .73, 1.])
    v = np.log(k + .019)
    slopes = np.diff(v) / np.diff(k)
    a = np.repeat(np.eye(3), len(slopes), axis=0)
    t = n.relu('log_hinges_positive', n.mix('log_hinges', 'image', a, -np.tile(k[:-1], 3)))
    lrgb = n.mix('log_rgb', t, a.T * np.tile(np.diff(slopes, prepend=0), 3), v[0])
    # Red vs neutral, blue vs neutral, yellow vs neutral, and log luminance.
    mat = np.array([[.33,.48,.19], [1.,-.68,-.32], [-1.,.27,.73], [.48,.52,-1.]])
    colors = n.mix('log_opponents', lrgb, mat)
    lum = n.mix('log_luminance', colors, [1,0,0,0])

    # Difference of Gaussians with zero DC. Repeat copies at the margin are
    # unnecessary: the ink masks are all well inside the crop.
    wide = gaussian1(2.05)
    narrow = gaussian1(.58)
    narrow = np.pad(narrow, ((len(wide)-len(narrow))//2,)*2)
    dog = np.outer(narrow,narrow) - np.outer(wide,wide)
    h = n.linear('center_surround', lum, np.stack([dog, -dog])[:,None], padding=len(wide)//2)
    hn = n.relu('contrast_halves', h)
    e = n.pool('local_contrast', hn, 'avgpool', 7,7,1,1)
    # Align a 7x7 local mean with the original 32x32 coordinates.
    p = np.zeros((1,2,7,7))
    p[0,:,3,3] = .5
    e = n.linear('centered_local_contrast', e, p, padding=6)

    # A soft histogram of signed contrast relative to nearby contrast plus a
    # floor. Each bin is a pair of rectifiers implementing a clipped ramp.
    # h is decomposed as h - t*e using two parallel projections.
    thresholds = [0., .38, .76, 1.22, 1.82, 2.62, 3.6]
    strengths = [.12, .15, .16, .17, .16, .13, .11]
    A, B, bias, coefficients = [], [], [], []
    for sign in (1,-1):
        for t, s in zip(thresholds, strengths):
            for offset, c in ((0,s),(-1,-s)):
                A.append([sign*24, 0])
                B.append([-24*t])
                bias.append(offset - 24*(.006 + .022*t))
                coefficients.append(sign*c)
    num = n.mix('contrast_bin_numerator', h, A, bias)
    den = n.mix('contrast_bin_reference', e, B)
    bins = n.relu('contrast_bins', n.op('contrast_bin_inputs', 'add', num, den))
    ink = n.mix('signed_ink_contrast', bins, coefficients)
    # Shape scoring sees both signed chromaticity and one-sided evidence.
    M = np.vstack([np.eye(4), -np.eye(4), np.eye(4)[1:]])
    t = n.relu('color_halves', n.mix('color_projections', colors, M, [0]*8+[-.13,-.14,-.14]))
    P = np.zeros((8,11))
    P[:4,:4] = np.eye(4); P[:4,4:8] = -np.eye(4)
    P[4:7,8:] = np.eye(3)
    P[7,[1,5,2,6]] = [.5]*4
    colors_full = n.mix('geometry_channels', t, P)
    g = n.pool('geometry_grid', colors_full, 'avgpool', 2,2,2,2)

    # Robust impression of chromatic ink and white cutouts. Use a bounded
    # red-opponent channel so that extreme hue shifts cannot dominate shape.
    u = n.relu('color_ink_positive', n.mix('color_ink_hinges', colors,
            [[0,4,0,0]]*4, [.24,-.76,-.44,-1.44]))
    red = n.mix('bounded_red_ink', u, [.68,-.68,.32,-.32])

    # When the surface is white, the upper envelope of nearby luminance is
    # a better reference for black ink than the arithmetic local mean. A
    # narrow Gaussian before the max rejects single-pixel sampling noise.
    a = gaussian1(.51)
    smooth = n.linear('smoothed_luminance',lum,np.outer(a,a)[None,None],padding=len(a)//2)
    hi = n.pool('white_envelope_window',smooth,'maxpool',7,7,1,1)
    pad = np.zeros((1,1,7,7)); pad[0,0,3,3] = 1
    # Logs may have either sign. Pad with a low luminance reference.
    hi = n.mix('shift_envelope',hi,[[1]],4.)
    hi = n.linear('white_envelope_centered',hi,pad,-4.,padding=6)
    d = n.op('ink_depth','add',hi,n.mix('negative_smoothed_luminance',smooth,[[-1]]))
    cuts = [.055,.19,.40,.69,.98]
    gains = [.24,.47,.22,.06,.01]
    wd = [[1/.16]]*(2*len(cuts)); bd = sum(([-t/.16,-t/.16-1] for t in cuts),[])
    dr = n.relu('ink_depth_hinges',n.mix('ink_depth_inputs',d,wd,bd))
    dark = n.mix('black_ink',dr,sum(([u,-u] for u in gains),[]))
    red_grid = n.pool('red_sector_grid',red,'avgpool',2,2,2,2)
    return g, ink, red, dark, red_grid


# --------- Vector artwork: geometric primitives and dimensionless drawings ---------

N = 192
SPAN = 1.26
uv = ((np.arange(N) + .5) / N * 2 - 1) * SPAN
Y, X = np.meshgrid(uv, uv, indexing='ij')


def rect(x1, y1, x2, y2):
    return (X>=x1)&(X<=x2)&(Y>=y1)&(Y<=y2)


def ellipse(cx, cy, rx, ry):
    return ((X-cx)/rx)**2 + ((Y-cy)/ry)**2 <= 1


def poly(points):
    mask = np.zeros_like(X, dtype=bool)
    for (a,b),(c,d) in zip(points, points[1:] + points[:1]):
        if b != d:
            mask ^= ((Y>b)!=(Y>d)) & (X < a + (Y-b)*(c-a)/(d-b))
    return mask


def stroke(points, w):
    mask = np.zeros_like(X, dtype=bool)
    for (a,b),(c,d) in zip(points, points[1:]):
        s, t = c-a, d-b
        if s == 0 and t == 0:
            continue
        alpha = np.clip(((X-a)*s + (Y-b)*t)/(s*s+t*t),0,1)
        mask |= (X-a-alpha*s)**2 + (Y-b-alpha*t)**2 <= (.5*w)**2
    return mask


def bez(p, *curves):
    """Join straight segments and cubic Beziers for monoline strokes."""
    pts = [tuple(p)]
    t = np.linspace(0.,1.,13)[1:,None]
    for q in curves:
        if len(q) == 2:
            pts.append(tuple(q)); continue
        a = np.array(pts[-1]); b,c,d = np.array(q).reshape(3,2)
        pts.extend(map(tuple,(1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t*t*c + t**3*d))
    return pts


def transform(path, cx, cy, xscale=1., yscale=1.):
    return [(cx+x*xscale,cy+y*yscale) for x,y in path]


# A simplified DIN-style monoline alphabet. Cap height is one em; these
# coordinates describe stroke centre lines in an abstract type design.
DIGITS = {
    '0': [bez((0,-.438), (.17,-.438,.246,-.34,.246,-.18), (.246,.18), (.246,.34,.17,.438,0,.438), (-.17,.438,-.246,.34,-.246,.18), (-.246,-.18), (-.246,-.34,-.17,-.438,0,-.438))],
    '1': [[(-.173,-.285),(.045,-.436),(.045,.444)]],
    '1_base': [[(-.167,-.285),(.033,-.432),(.033,.432)],[(-.164,.43),(.203,.43)]],
    '2': [bez((-.235,-.26), (-.23,-.385,-.14,-.447,.003,-.447), (.22,-.447,.31,-.295,.199,-.13), (.14,-.04,-.10,.235,-.234,.425), (.25,.425))],
    '3': [bez((-.222,-.328), (-.177,-.436,-.049,-.452,.052,-.445), (.222,-.437,.250,-.328,.241,-.241), (.231,-.104,.153,-.047,.008,-.041), (-.081,-.041)), bez((.008,-.041), (.172,-.042,.258,.068,.258,.207), (.258,.472,-.117,.538,-.245,.326))],
    '3_flat': [bez((-.228,-.428), (.211,-.428), (-.04,-.082), (.136,-.076,.259,.013,.255,.193), (.247,.472,-.114,.519,-.241,.324))],
    '5': [bez((.221,-.427), (-.190,-.427), (-.216,-.024), (-.13,-.076,-.073,-.122,.034,-.117), (.165,-.114,.247,-.019,.246,.153), (.246,.509,-.107,.51,-.243,.321))],
    '6': [bez((.176,-.390), (.027,-.52,-.144,-.395,-.207,-.209), (-.257,-.071,-.254,.117,-.234,.241), (-.217,.378,-.14,.448,.013,.448), (.184,.448,.245,.333,.245,.182), (.245,.014,.154,-.071,.029,-.071), (-.10,-.071,-.21,.003,-.243,.117))],
    '7': [[(-.246,-.429),(.237,-.429),(-.123,.442)]],
    '8': [bez((0,-.019), (-.178,-.019,-.224,-.127,-.224,-.231), (-.224,-.36,-.135,-.444,0,-.444), (.135,-.444,.224,-.36,.224,-.231), (.224,-.127,.178,-.019,0,-.019), (-.17,-.019,-.258,.089,-.258,.219), (-.258,.352,-.176,.446,0,.446), (.176,.446,.258,.352,.258,.219), (.258,.089,.17,-.019,0,-.019))],
    '8_round': [bez((0,-.052),(-.175,-.052,-.246,-.155,-.246,-.251),(-.246,-.373,-.142,-.442,0,-.442),(.142,-.442,.246,-.373,.246,-.251),(.246,-.155,.175,-.052,0,-.052),(-.206,-.052,-.273,.083,-.273,.207),(-.273,.371,-.162,.45,0,.45),(.162,.45,.273,.371,.273,.207),(.273,.083,.206,-.052,0,-.052))],
    'S': [bez((.235,-.335), (.065,-.53,-.244,-.445,-.242,-.238), (-.242,-.10,-.105,-.063,.048,-.019), (.25,.049,.291,.167,.232,.298), (.166,.481,-.127,.495,-.255,.334))],
    'T': [[(-.27,-.436),(.27,-.436)],[(0,-.436),(0,.437)]],
    'O': [bez((0,-.438), (.17,-.438,.246,-.34,.246,-.18), (.246,.18), (.246,.34,.17,.438,0,.438), (-.17,.438,-.246,.34,-.246,.18), (-.246,-.18), (-.246,-.34,-.17,-.438,0,-.438))],
    'P': [bez((-.205,.436), (-.205,-.436), (.015,-.436), (.335,-.436,.34,.03,.015,.03), (-.205,.03))],
}


def lettering(text, height, width_ratio, track=.72, y=0., thickness=.13, style=0):
    widths = [((.82 if style in (0,3) else .71) if c == '1' else 1.)*track for c in text]
    x = -sum(widths)/2
    letters = []
    for c, w in zip(text, widths):
        letter = np.zeros_like(X, dtype=bool)
        d = ('3_flat' if c=='3' and style==4 else
             '1_base' if c=='1' and style==2 else
             '8_round' if c=='8' and style in (1,2,3) else c)
        for p in DIGITS[d]:
            letter |= stroke(transform(p, (x+w/2)*height*width_ratio, y,
                                       height*width_ratio, height), height*thickness)
        letters.append(letter)
        x += w
    return letters


def number(k,v):
    text = TEXT[k]
    height = (.75,.66,.81,.70,.79)[v]
    width = ((1.,1.06,1.02,1.17,.81) if len(text)==2 else (.96,.94,.83,1.11,.76))[v]
    return lettering(text,height,width,track=(.76 if len(text)==3 else .72),
                    y=(0,.020,-.016,-.023,.01)[v],thickness=(.126,.15,.121,.154,.112)[v],style=v)


def car(cx, cy, s=1.):
    def p(pts): return poly(transform(pts,cx,cy,s,s))
    outline = p([(-.24,.077),(-.24,-.033),(-.18,-.088),(-.136,-.177),(.128,-.177),(.181,-.091),(.24,-.036),(.24,.077),(.178,.077),(.166,.151),(.096,.151),(.074,.079),(-.078,.079),(-.094,.151),(-.165,.151),(-.176,.077)])
    window = p([(-.133,-.090),(-.097,-.145),(.095,-.145),(.127,-.090)])
    return outline & ~window


def front_truck(cx, cy, s=1.):
    def p(pts): return poly(transform(pts,cx,cy,s,s))
    return p([(-.232,-.288),(.231,-.288),(.231,.167),(.185,.167),(.185,.232),(.100,.232),(.100,.164),(-.101,.164),(-.101,.23),(-.191,.23),(-.191,.167),(-.232,.167)])


def side_truck():
    return (poly([(-.63,.13),(-.63,-.02),(-.53,-.07),(-.47,-.24),(-.27,-.24),(-.27,.12)])
            & ~poly([(-.51,-.05),(-.46,-.18),(-.32,-.18),(-.32,-.05)])) | rect(-.242,-.265,.615,.127) | ellipse(-.434,.13,.105,.105) | ellipse(.394,.131,.096,.096)


def arrow(tip, angle, end, shaft=.15, headw=.52, headl=.34):
    d = np.array([np.sin(angle),-np.cos(angle)])
    q = np.array([d[1],-d[0]])
    a = np.array(tip); b = a-d*headl
    neck = a-d*(headl+.065)
    return poly([tuple(v) for v in (a,b+q*headw/2,b+q*shaft/2,neck+q*shaft/2,
                np.array(end)+q*shaft/2,np.array(end)-q*shaft/2,neck-q*shaft/2,
                b-q*shaft/2,b-q*headw/2)])


def hooked(right):
    s = 1 if right else -1
    p = bez((-.19*s,.61),(-.19*s,.03),(-.19*s,-.26,.00*s,-.37,.24*s,-.37),(.43*s,-.37))
    return stroke(p,.175) | arrow((.66*s,-.37),s*np.pi/2,(.17*s,-.37),.175,.47,.31)


def roundabout():
    out = np.zeros_like(X, dtype=bool)
    for a in np.radians([15,135,255]):
        # Three counterclockwise arrows on a shared circle.
        t = np.linspace(a+.44,a-.96,20)
        r = .61
        path = list(zip(r*np.cos(t),r*np.sin(t)))
        tip = (.62*np.cos(a-1.16),.62*np.sin(a-1.16))
        dx,dy = tip[0]-path[-5][0],tip[1]-path[-5][1]
        out |= stroke(path,.108) | arrow(tip,np.arctan2(dx,-dy),path[-3],.108,.315,.20)
    return out


def human(cx, cy, s=1., mirror=1):
    def st(p,w): return stroke(transform(p,cx,cy,s*mirror,s),w*s)
    return (ellipse(cx-.092*s*mirror,cy-.376*s,.091*s,.092*s)
            | st([(-.06,-.245),(-.008,-.045),(.025,.135)],.109)
            | st([(-.045,-.202),(-.198,-.006),(-.238,.137)],.061)
            | st([(-.033,-.223),(.119,-.064),(.229,-.038)],.06)
            | st([(.020,.101),(-.099,.291),(-.198,.455)],.082)
            | st([(.035,.106),(.167,.274),(.255,.44)],.078))


def traffic_lenses(v):
    if v:
        return [rect(-.112,y-.104,.112,y+.104) for y in (-.209,.151,.485)]
    return [ellipse(0,y,.118,.116) for y in (-.245,.096,.427)]


def triangle_ink(k, v):
    if k == 11:  # major road and minor road at right angles
        return poly([(-.123,.63),(-.123,.355),(-.337,.355),(-.337,.237),(-.123,.237),(-.106,-.084),(-.177,-.084),(0,-.304),(.177,-.084),(.106,-.084),(.123,.237),(.337,.237),(.337,.355),(.123,.355),(.123,.63)])
    if k == 18:
        return poly([(-.081,-.342),(.080,-.342),(.045,.347),(-.045,.347)]) | rect(-.071,.457,.071,.603)
    if k in (19,20):
        s = 1 if k==19 else -1
        if v==0:
            p=bez((.169,.601),(.169,.284),(.169,.079,.074,-.018,-.19,-.018))
        elif v==1:
            p=bez((.185,.585),(.185,.326),(.185,.121,.093,.045,-.027,.042),(-.213,.04))
        else:
            p=bez((.203,.595),(.203,.245),(.204,.086,.098,-.059,-.09,-.065),(-.275,-.065))
        return stroke(transform(p,0,0,s,1),(.142,.19,.175)[v])
    if k == 21:
        # Reverse bends, rising from the lower right to the upper left.
        if v:
            return stroke([(.181,.608),(.181,.339),(.099,.238),(-.101,.238),(-.194,.142),(-.194,-.207)],.158)
        return stroke(bez((.203,.607),(.203,.426),(.211,.316,.201,.191,.118,.185),(.01,.176),(-.124,.148,-.158,.116,-.158,-.03),(-.158,-.176)),.137)
    if k == 22:
        return rect(-.46,.488,.462,.616) | ellipse(-.231,.493,.183,.134) | ellipse(.193,.495,.17,.126)
    if k == 23:
        # Rear silhouette of a skidding car; parallel S-shaped tire trails.
        dy = (-.035,.032,.012)[v]
        s = (.99,1.11,.78)[v]
        a = [(-.273,.120),(-.295,-.032),(-.208,-.137),(-.13,-.198),(.101,-.171),(.242,-.045),(.283,.025),(.275,.182),(.193,.17),(.167,.123),(-.161,.102),(-.181,.164)]
        p = poly(transform(a,0,dy,s,s))
        if v==2:
            p |= stroke([(-.180,.012),(.167,.070)],.11)
        for d in (-.143,.17):
            c = bez((d+.019,.215),(d+.213,.342,d-.209,.373,d-.124,.425),(d-.05,.472,d+.129,.499,d+.158,.595))
            p |= stroke(transform(c,0,dy,s,1),.058 if v else .047)
        return p
    if k == 24:
        if v==2:
            return stroke([(-.108,.615),(-.108,-.271)],.103) | stroke([(.277,.615),(.277,.203),(.10,-.071),(.10,-.27)],.11)
        if v==1:
            return stroke([(-.15,.60),(-.15,-.22)],.10) | stroke([(.249,.61),(.249,.24),(.105,.069),(.105,-.21)],.102)
        return stroke([(-.197,.603),(-.197,-.205)],.079) | stroke(bez((.284,.61),(.284,.356),(.284,.213,.165,.13,.125,-.031),(.125,-.202)),.081)
    if k == 25:
        worker = (ellipse(-.105,-.118,.074,.079) | stroke([(-.148,-.019),(-.195,.145),(-.177,.339),(-.282,.593)],.095)
                | stroke([(-.174,.079),(-.029,.195),(.093,.177)],.075)
                | stroke([(-.177,.327),(-.079,.474),(-.027,.594)],.078)
                | stroke([(-.095,.087),(.198,.518)],.052))
        pile = poly([(.061,.61),(.151,.472),(.207,.458),(.267,.323),(.331,.348),(.477,.611)])
        if v:
            worker = (ellipse(-.101,-.173,.077,.084) | stroke([(-.143,-.065),(-.239,.107),(-.216,.298),(-.363,.566)],.103)
                | stroke([(-.211,.014),(-.070,.175),(.066,.178)],.078)
                | stroke([(-.225,.29),(-.110,.458),(-.116,.589)],.076)
                | stroke([(-.097,.051),(.197,.495)],.057))
        return worker | pile
    if k == 26:
        lenses = traffic_lenses(v)
        return .64*lenses[0] + .36*lenses[1] + .68*lenses[2]
    if k == 27:
        return human(0,.17,.94,-1 if v else 1)
    if k == 28:
        if v:
            return human(-.199,.265,.58,1) | human(.163,.207,.77,1) | stroke([(-.037,.188),(.057,.145)],.042)
        return human(-.233,.294,.50,-1) | human(.151,.236,.76,-1) | stroke([(-.14,.291),(.009,.118)],.040)
    if k == 29:
        # A side-on bicycle, with open wheels and a diamond frame.
        dy = (.0,-.059,.045)[v]
        c = .365+dy
        size = (.175,.153,.191)[v]
        hole = size-(.070 if v else .055)
        w = (ellipse(-.276,c,size,size) & ~ellipse(-.276,c,hole,hole)) | (ellipse(.281,c,size,size) & ~ellipse(.281,c,hole,hole))
        return (w | stroke(transform([(-.276,.365),(-.155,.146),(.003,.373),(-.283,.373)],0,dy),.070)
                | stroke(transform([(.003,.373),(.197,.146),(-.155,.146)],0,dy),.064)
                | stroke(transform([(.279,.365),(.156,.033),(.211,-.008)],0,dy),.057)
                | stroke(transform([(-.245,.080),(-.113,.080)],0,dy),.069))
    if k == 30:
        p = np.zeros_like(X,dtype=bool)
        for a in np.radians(np.array([0,60,120,180,240,300])+30*v):
            u, w = np.array([np.sin(a),np.cos(a)]), np.array([np.cos(a),-np.sin(a)])
            o = np.array([0,.252+.021*v]); j = o+.213*u
            p |= stroke([o,o+.366*u],.047+.012*v) | stroke([j+.093*u+.092*w,j,j+.093*u-.092*w],.045+.012*v)
        return p
    if k == 31:
        # Leaping deer, head and antlers to the left.
        if v:
            return (stroke([(-.21,-.157),(-.185,.014),(-.127,.146),(.094,.201),(.260,.25)],.135)
                | stroke([(-.208,-.111),(-.307,-.049)],.073)
                | stroke([(-.207,-.173),(-.278,-.264),(-.258,-.348)],.052)
                | stroke([(-.218,-.21),(-.155,-.317)],.044)
                | stroke([(-.082,.15),(-.155,.358),(-.260,.425)],.060)
                | stroke([(-.061,.173),(.07,.393)],.065)
                | stroke([(.247,.241),(.34,.389),(.457,.493)],.071)
                | stroke([(.26,.218),(.385,.314),(.498,.365)],.061))
        return (poly([(-.315,.083),(-.331,.023),(-.221,-.018),(-.172,-.081),(-.105,-.068),(-.094,.052),(.076,.132),(.281,.162),(.315,.245),(.223,.322),(.116,.28),(-.080,.192),(-.146,.104),(-.240,.122)])
                | stroke([(-.17,-.036),(-.167,-.19),(-.228,-.289),(-.223,-.337)],.040)
                | stroke([(-.165,-.139),(-.086,-.219),(-.089,-.276)],.035)
                | stroke([(-.03,.167),(-.059,.284),(-.222,.416)],.060)
                | stroke([(-.041,.188),(.096,.305),(.045,.437)],.045)
                | stroke([(.249,.251),(.351,.368),(.426,.485)],.066)
                | stroke([(.235,.205),(.354,.269),(.486,.347)],.053))
    raise ValueError(k)


RING, TRI, YIELD, DIAMOND, STOP, ENTRY, BLUE, END = range(8)
FAMILIES = [RING]*6 + [END] + [RING]*4 + [TRI,DIAMOND,YIELD,STOP,RING,RING,ENTRY] + [TRI]*14 + [END] + [BLUE]*8 + [END,END]
TEXT = {0:'20',1:'30',2:'50',3:'60',4:'70',5:'80',7:'100',8:'120',6:'80'}


def glyph(k, v):
    """Return ink (real-valued for translucent cancellation stripes)."""
    if k in TEXT:
        ink = np.logical_or.reduce(number(k,v))
        if k != 6:
            return ink
        if v==2:
            return .98*ink + .09*np.exp(-.5*((X+Y)/.21)**6) * (np.hypot(X,Y)<.93)
    elif k in (9,10,41,42):
        ink = (front_truck(-.292,.010,.95) if k in (10,42) else car(-.30,.028,1.)) | car(.305,.028,1.)
        if k in (9,10):
            return ink
    elif k==16:
        return side_truck()
    elif k==14:
        return np.logical_or.reduce(lettering('STOP', (.514,.462)[v], (.96,1.06)[v], track=.705, thickness=.135))
    elif k==17:
        return rect(-.74,(-.147,-.185)[v],.74,(.147,.185)[v])
    elif FAMILIES[k]==TRI:
        return triangle_ink(k,v)
    elif k in (33,34):
        return hooked(k==33)
    elif k==35:
        return arrow((0,-.735),0,(0,.646),.171,.49,.337)
    elif k in (36,37):
        s = 1 if k==36 else -1
        if v:
            return (arrow((-.215*s,-.69),0,(-.215*s,.625),.145,.426,.30)
                    | stroke(bez((-.214*s,.416),(-.21*s,.14,-.064*s,.071,.22*s,.079),(.49*s,.079)),.145)
                    | arrow((.66*s,.079),s*np.pi/2,(.201*s,.079),.145,.41,.272))
        return (arrow((-.189*s,-.726),0,(-.189*s,.635),.150,.424,.291)
                | stroke(bez((-.189*s,.416),(-.188*s,.118,-.099*s,-.076,.18*s,-.093),(.49*s,-.093)),.140)
                | arrow((.659*s,-.093),s*np.pi/2,(.190*s,-.093),.141,.414,.258))
    elif k in (38,39):
        s = 1 if k==38 else -1
        a = np.pi*(.75 if s==1 else -.75)
        return arrow((.514*s,.51),a,(-.515*s,-.515),.178,.562,.399)
    elif k==40:
        return roundabout()
    elif k in (12,13,15):
        return None
    else:
        ink = np.zeros_like(X)
    # Cancellation signs retain the diagonal overlay even when the original
    # pictogram is absent. Closely spaced black stripes merge at low resolution.
    angle = np.radians((45,49,40)[v])
    d = X*np.sin(angle)+Y*np.cos(angle)
    stripe = np.exp(-.5*(d/(.155 if v!=1 else .193))**6)
    stripe *= np.clip((.99-np.hypot(X,Y))/.065,0,1)
    return np.maximum((.85 if k==6 else .82)*ink, stripe*(1. if k==32 else .73))


def soft_band(u, low, high, edge):
    return np.minimum(np.clip((u-low)/edge,0,1),np.clip((high-u)/edge,0,1))


def pose_parameters():
    # Translation, scale, roll and two mild projective/skew directions. The
    # bounds express sampling and pose tolerances, not a fitted distribution.
    warps = [(0,0,0),(8,0,0),(-8,0,0),(0,.105,0),(0,-.105,0),(0,0,.13),(0,0,-.13)]
    p = []
    for r in (8.,9.20,10.4,11.6,12.8,14.0):
        for dy in (-1.45,0,1.45):
            for dx in (-1.45,0,1.45):
                for rot,asp,skew in warps:
                    p.append((r,dx,dy,rot,asp,skew))
                if r>=9.2 and dx==0 and dy==0:
                    p.append((r,dx,dy,0,.20,0))
                    p.append((r,dx,dy,0,-.20,0))
        for dx,dy in ((-2.65,0),(2.65,0),(0,-2.65),(0,2.65)):
            p.append((r,dx,dy,0,0,0))
    return p


def coord(pose, family, stride=1, supersample=1):
    r,dx,dy,angle,asp,skew=pose
    mid = (np.arange(32//stride)+.5)*stride-16
    fine = ((np.arange(supersample)+.5)/supersample-.5)*stride
    grid = (mid[:,None]+fine).reshape(-1)
    yy, xx = np.meshgrid(grid-dy,grid-dx,indexing='ij')
    a = np.radians(angle)
    x = (xx*np.cos(a)+yy*np.sin(a))/(r*(1+asp))
    y = (yy*np.cos(a)-xx*np.sin(a))/(r*(1-asp))
    # The bounding box of an equilateral triangle is 2 wide by sqrt(3)
    # high. Warn and yield crops are usually close to square after resize.
    if family in (TRI,YIELD):
        y /= .92
    x = x-skew*y
    y = y*(1+.045*np.sign(skew)*x)
    return x,y


def down(a, s):
    n = a.shape[-1]//s
    return a.reshape(*a.shape[:-2],n,s,n,s).mean(axis=(-1,-3))


def unit(mask):
    return mask/max(float(mask.sum()),1e-6)


def geometry_template(pose, family):
    x,y = coord(pose,family,2,3)
    r = np.hypot(x,y)
    if family == STOP:
        r = np.maximum(np.maximum(abs(x),abs(y)),(abs(x)+abs(y))/np.sqrt(2))
    if family == DIAMOND:
        r = abs(x)+abs(y)
    if family in (TRI,YIELD):
        yy = y if family==TRI else -y
        border_dist = np.minimum(1-yy,(yy+1-2*abs(x))/np.sqrt(5))
        edge = soft_band(border_dist,.018,.182,.05)
        inside = soft_band(border_dist,.185,.38,.065)
        ext = soft_band(border_dist,-.22,-.010,.055)
    else:
        edge = soft_band(r,.777,1.004,.065)
        inside = soft_band(r,.53,.779,.085)
        ext = soft_band(r,1.035,1.273,.078)
    edge,inside,ext = [unit(down(a,3)) for a in (edge,inside,ext)]
    plate = unit(down(np.clip((.95-r)/.19,0,1),3))
    rim = unit(down(soft_band(r,.86,.992,.046),3))
    mat = np.zeros((8,16,16))
    if family in (RING,TRI,YIELD):
        mat[1] = .96*edge-.75*inside-.21*ext
        mat[4] = .23*edge
        mat[0] = .24*(inside-edge)
        # A broad achromatic inside border is expected.
        mat[7] = -.12*inside
        if family == YIELD:
            field = unit(down(np.clip((border_dist-.206)/.088,0,1),3))
            mat[1] += -.28*(field-edge)
            mat[4] += -.23*field
            mat[0] += .17*(field-edge)
        if family == RING:
            field = unit(down(np.clip((.72-r)/.12,0,1),3))
            mat[1] += -.20*(field-edge)
    elif family in (STOP,ENTRY,BLUE):
        c = 2 if family==BLUE else 1
        response = .57*plate+.43*edge
        mat[c] = (.62 if c==2 else .56)*response - (.41 if c==2 else .31)*ext
        mat[5 if c==2 else 4] = .46*response
        # Suppress neutral signs under a color cast and blue sky surrounding
        # a gray plate; an actual blue sign has a blue rim as well as center.
        mat[4 if c==2 else 5] = -.31*plate
        if family==STOP:
            mat[1] += .12*(rim-ext)
    elif family == DIAMOND:
        core = unit(down(np.clip((.67-r)/.15,0,1),3))
        rim = unit(down(soft_band(r,.69,.998,.079),3))
        mat[3] = .94*(core-.73*rim-.27*ext)
        mat[6] = .43*core
        mat[7] = -.16*rim
        mat[0] = .22*(rim-core)+.19*(rim-ext)
    elif family == END:
        mat[0] = .19*(inside-rim)+.11*(inside-ext)
        mat[7] = -.15*plate-.12*edge
        mat[4] = -.12*edge
        mat[1] = -.36*(edge-inside)
        mat[2] = -.27*(plate-ext)
        mat[0] += .12*(plate-ext)
    return mat


def sampler(x,y):
    u = np.clip((x/SPAN+1)*N/2-.5,0,N-1.00001)
    v = np.clip((y/SPAN+1)*N/2-.5,0,N-1.00001)
    a,b = u.astype(int),v.astype(int)
    c,d = u-a,v-b
    q = b*N+a
    m = (abs(x)<SPAN)&(abs(y)<SPAN)
    return (q, q+1, q+N, q+N+1, m*(1-c)*(1-d), m*c*(1-d), m*(1-c)*d, m*c*d)


def sample(image, s):
    q0,q1,q2,q3,w0,w1,w2,w3 = s
    a = image.ravel()
    return w0*a[q0]+w1*a[q1]+w2*a[q2]+w3*a[q3]


def normalized_minority(a, scale=.18):
    # Boost interior relative to edge pixels without imposing a hard edge.
    return unit(a*np.clip(a/scale,.35,1))


def digit_match(y, r, support, t, letters, s):
    # Each character casts one vote, including the thin leading 1 in
    # three-digit speed limits. Layout mismatch is penalized by the gaps.
    glyphs = [blur(down(sample(a,s),2),.46)*support for a in letters]
    on = sum(normalized_minority(z) for z in glyphs)/len(glyphs)
    blank = support * np.clip((.81-down(r,2))/.085,0,1) * (1-np.clip(2.15*t,0,1))
    row = down(np.clip((.52-abs(y))/.16,0,1),2)
    blank = unit(blank*(.31+.69*row))
    return on, blank


def glyph_template(image, x, y, s, family, k, pose, letters=None):
    r = np.hypot(x,y)
    if family == TRI:
        margin = np.minimum(1-y,(y+1-2*abs(x))/np.sqrt(5))
        support = np.clip((margin-.178)/.082,0,1)
    else:
        support = np.clip(((1.0 if family==END else .817)-r)/.115,0,1)
    support = down(support,2)
    if image is None:
        return np.zeros((32,32),dtype=np.float32)
    t = blur(down(sample(image,s),2),.49)
    # Penalize missing strokes and dark marks where the sign should be
    # blank. Normalizing foreground and background separately prevents a
    # large dense symbol from winning simply because of its area.
    halo = blur(t,1.36)
    blank = support * (1-np.clip(1.55*t,0,1))
    flanks = unit(blank*halo)
    on = unit(t*support)
    coeff = on - .42*flanks - .58*unit(blank)
    if letters is not None and family==RING:
        on,blank = digit_match(y,r,support,t,letters,s)
        coeff = on-.20*flanks-.80*blank
    strength = 1. if family==TRI else 1.04
    if family in (BLUE,ENTRY,STOP):
        coeff = -coeff
    if family==END:
        strength *= 1.06
        if letters is not None:
            stripe_free = down(np.clip((abs(x+y)-.23)/.15,0,1),2)
            text = sum(unit(blur(down(sample(a,s),2),.48)*support*stripe_free) for a in letters)/len(letters)
            ref = unit(stripe_free*support*(1-np.clip(t*2.1,0,1)))
            coeff = .74*coeff + .22*(text-ref)
    return (-strength*coeff).astype(np.float32)


def black_template(image, x, y, s, family, k, pose, letters=None):
    if family == TRI:
        margin = np.minimum(1-y,(y+1-2*abs(x))/np.sqrt(5))
        support = np.clip((margin-.22)/.10,0,1)
    else:
        r = np.hypot(x,y)
        support = np.clip(((.98 if family==END else .76)-r)/.105,0,1)
    support = down(support,2)
    if image is None:
        return -.77*unit(support)
    t = blur(down(sample(image,s),2),.40)
    dt = unit(t*support)
    blank = unit(support * (1-np.clip(2.4*blur(t,.71),0,1)))
    # An ink likelihood corresponds to explaining BOTH the dark glyph and
    # the absence of ink elsewhere. Use extra stroke evidence for a glyph
    # with more covered area (but never add a reward that scales with area).
    amount = np.sqrt(np.sum((t*support)**2)) / pose[0]
    a = .36 + .50*np.clip(amount,.10,.50)
    if k in (9,10):
        a *= .73  # the red car does not have to be as dark as the black car
    if family==TRI:
        a = .49
    if letters is not None and family==RING:
        dt,blank = digit_match(y,r,support,t,letters,s)
        return .55*dt-.47*blank
    if letters is not None and family==END:
        stripe_free = down(np.clip((abs(x+y)-.20)/.15,0,1),2)
        text = sum(unit(blur(down(sample(a,s),2),.44)*support*stripe_free) for a in letters)/len(letters)
        return .94*a*dt + .13*text - .47*blank
    return a*dt - .37*blank


def sector_template(pose,family):
    x,y = coord(pose,family,2,3)
    if family in (TRI,YIELD):
        yy = y if family==TRI else -y
        d = np.stack([1-yy,(yy+1-2*x)/np.sqrt(5),(yy+1+2*x)/np.sqrt(5)])
        axis = np.argmin(d,axis=0)
        dist = np.min(d,axis=0)
        groups = [axis==i for i in range(3)]
        band = soft_band(dist,.008,.190,.051)
        inner = soft_band(dist,.204,.397,.06)
    else:
        a = np.arctan2(y,x)
        rr = np.hypot(x,y)
        band = soft_band(rr,.77,1.02,.07)
        inner = soft_band(rr,.48,.75,.07)
        groups = [np.cos(a-i*2*np.pi/3)>=.5 for i in range(3)]
    return np.stack([unit(down(group*band,3))-unit(down(group*inner,3)) for group in groups])


def label_color_template(k,v,x,y,symbol,s):
    w = np.zeros((32,32),dtype=np.float32)
    if k in (9,10,41,42):
        left = front_truck(-.292,.010,.95) if k in (10,42) else car(-.30,.028,1.)
        left = unit(down(sample(left,s),2))
        right = unit(down(sample(car(.305,.028,1.),s),2))
        if k in (9,10):
            w[:] = .52*(left-right)
        else:
            w[:] = -.21*(left-right)
    elif k in (14,17):
        t = down(sample(symbol,s),2)
        radius = np.hypot(x,y) if k==17 else np.maximum(np.maximum(abs(x),abs(y)),(abs(x)+abs(y))/np.sqrt(2))
        blank = down(np.clip((.85-radius)/.12,0,1),2)*(1-np.clip(1.6*t,0,1))
        w[:] = .85*(unit(blank)-unit(t))
    elif k==26:
        # Red and green lenses give opposite-signed red opponency.
        top,mid,bot = [unit(down(sample(a,s),2)) for a in traffic_lenses(v)]
        t = traffic_lenses(v)[0]
        surround = down(sample(ellipse(0,-.232,.31,.177)&~t,s),2)
        w[:] = .87*top-.37*bot-.41*unit(surround)-.09*mid
    return w


def build(path):
    n = Network()
    geo_input, contrast, red, dark, red_grid = feature_layers(n)
    poses = pose_parameters()
    P = len(poses)
    style_counts = [5 if k in TEXT and k!=6 else (3 if k in (6,19,20,23,24,29) else (2 if k in (14,17,21,25,26,27,28,30,31,32,36,37,41,42) else 1)) for k in range(43)]
    styles = [(i,v) for i,c in enumerate(style_counts) for v in range(c)]
    color_ids = [s for s,(k,v) in enumerate(styles) if k in (9,10,14,17,26,41,42)]
    black_ids = [s for s,(k,v) in enumerate(styles) if FAMILIES[k] in (RING,TRI,END)]
    S = len(styles)
    G = np.empty((8,P,8,16,16),dtype=np.float32)
    T = np.empty((S,P,32,32),dtype=np.float32)
    C = np.empty((len(color_ids),P,32,32),dtype=np.float32)
    D = np.empty((len(black_ids),P,32,32),dtype=np.float32)
    E = np.empty((3,3,P,16,16),dtype=np.float32)
    prior = np.empty((8,P),dtype=np.float32)
    artwork = [glyph(k,v) for k,v in styles]
    letters = [number(k,v) if k in TEXT else None for k,v in styles]
    for j, p in enumerate(poses):
        r,dx,dy,ang,asp,skew = p
        for f in range(8):
            G[f,j] = geometry_template(p,f)
            prior[f,j] = -.0027*(dx*dx+dy*dy)-.0013*(r-12.25)**2-.00009*ang**2-.12*(asp**2+skew**2)
        maps = {}
        for f in (TRI,END,RING):
            x,y=coord(p,f,1,2); maps[f]=(x,y,sampler(x,y))
        for f in (STOP,ENTRY,BLUE,DIAMOND,YIELD):
            maps[f] = maps[RING]
        for f in (RING,TRI,YIELD):
            E[f,:,j] = sector_template(p,f)
        for s,(k,v) in enumerate(styles):
            f=FAMILIES[k]
            x,y,q=maps[f]
            T[s,j] = glyph_template(artwork[s],x,y,q,f,k,p,letters[s])
        for c,s in enumerate(color_ids):
            k,v=styles[s]; x,y,q=maps[FAMILIES[k]]
            C[c,j] = label_color_template(k,v,x,y,artwork[s],q)
        for d,s in enumerate(black_ids):
            k,v=styles[s]; f=FAMILIES[k]; x,y,q=maps[f]
            D[d,j] = black_template(artwork[s],x,y,q,f,k,p,letters[s])
    geo = n.linear('shape_and_color_matches', geo_input, G.reshape(8*P,-1), prior.ravel())
    g = n.shape('family_pose_grid',geo,8,1,P)

    # All three pieces of a real border should be supported. Contrast just
    # one edge of a triangle must not masquerade as a red circle.
    edges = n.linear('red_border_side_matches',red_grid,E.reshape(9*P,-1))
    edges = n.shape('red_border_sides',edges,3,3,P)
    neg = n.mix('negative_side_scores',edges,-np.eye(3))
    low = n.pool('negative_weakest_side',neg,'maxpool',3,1,3,1)
    mean = n.pool('mean_border_side',edges,'avgpool',3,1,3,1)
    edge_mismatch = n.relu('inconsistent_border',n.op('side_score_spread','add',low,mean))
    ef = np.zeros((8,3)); ef[:3]=-.84*np.eye(3)
    g = n.op('consistent_shape_scores','add',g,n.mix('border_consistency_penalty',edge_mismatch,ef))

    # Prefer the pose that explains the plate, not one that happens to
    # put a few overlay strokes on a nearby tree, edge, or another sign.
    best = n.pool('best_plate_pose',g,'maxpool',1,P,1,P)
    # A small linear layer broadcasts each family maximum over its own
    # poses without a spatial transpose operation.
    w = np.eye(8)[:,:,None]*np.ones((1,1,P))
    peak_flat = n.linear('broadcast_family_peaks', best, w.transpose(0,2,1).reshape(8*P,8))
    peak = n.shape('family_peak_grid',peak_flat,8,1,P)
    excess = n.relu('pose_mismatch',n.op('pose_gap','add',
        n.mix('negative_family_scores',g,-np.eye(8),-.065),peak))
    g = n.op('anchored_family_scores','add',g,n.mix('subtract_pose_mismatch',excess,-1.35*np.eye(8)))
    # Symbol scores should not override decisive plate evidence. Only
    # penalize alternatives behind by more than a broad ambiguity margin.
    best_shape = n.pool('best_plate_family',n.shape('plate_family_axis',best,1,1,8),'maxpool',1,8,1,8)
    fg = n.linear('family_score_deficits',best,-np.eye(8),-.155)
    bg = n.mix('broadcast_best_family',best_shape,np.ones((8,1)))
    gap = n.relu('family_deficit',n.op('family_gap','add',
        n.shape('family_peak_deficits',fg,8,1,1),bg))
    adj = n.linear('family_confidence_penalty',gap,-.93*w.transpose(0,2,1).reshape(8*P,8))
    g = n.op('confidence_adjusted_plates','add',g,n.shape('family_penalty_grid',adj,8,1,P))
    fan = np.eye(8)[[FAMILIES[k] for k,v in styles]]
    margins = {12:.065,13:.26,15:.30,32:-.105,41:-.055,42:-.055,6:-.095}
    geometry = n.mix('route_families_to_symbols',g,fan,[margins.get(k,-.03 if v==0 else -.042) for k,v in styles])
    t = n.linear('pictogram_matches', contrast, T.reshape(S*P,-1))
    t = n.shape('pictogram_pose_grid',t,S,1,P)
    col = n.shape('symbol_color_grid',n.linear('symbol_color_matches',red,C.reshape(len(color_ids)*P,-1)),len(color_ids),1,P)
    col = n.mix('route_symbol_color',col,np.eye(S)[:,color_ids])
    depth = n.shape('dark_symbol_grid',n.linear('dark_symbol_matches',dark,D.reshape(len(black_ids)*P,-1)),len(black_ids),1,P)
    depth = n.mix('route_dark_symbol',depth,np.eye(S)[:,black_ids])
    joint = n.op('joint_pose_scores','add',t,geometry,col,depth)
    m = n.pool('best_pose_per_symbol',joint,'maxpool',1,P,1,P)
    v = max(style_counts)
    routing = np.zeros((43*v,S))
    biases = np.full(43*v,-10.)
    for j,(k,s) in enumerate(styles):
        routing[k*v+s,j]=1
        biases[k*v+s]=0
    z = n.linear('route_symbol_scores',m,routing,biases)
    z = n.shape('class_symbol_grid',z,1,43,v)
    z = n.pool('best_symbol_per_class',z,'maxpool',1,v,1,v)
    n.shape('scores',z,43)
    n.write(path)


if __name__ == '__main__':
    build(Path(sys.argv[1]))
