"""Data-free, formula-built clothing recognizer. No input files are opened.

All geometric dimensions describe generic manufactured garments, not examples.
Templates are rasterizations of polygons, ellipses and line segments. There is
no optimization, parameter estimation, sampling of data, or learned readout.
"""
import json
import sys
from pathlib import Path
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.weights = {}

    def op(self, name, op, source, **kw):
        self.layers.append(dict(name=name, op=op,
                                inputs=[source] if isinstance(source, str) else source, **kw))
        return name

    def dense(self, name, source, w, b=0):
        w = np.asarray(w, np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.broadcast_to(np.asarray(b, np.float32), (len(w),)).copy()
        return self.op(name, 'dense', source, out=len(w))

    def conv(self, name, source, w, b=0, padding=0):
        w = np.asarray(w, np.float32)
        self.weights[name + '.W'] = w
        self.weights[name + '.b'] = np.broadcast_to(np.asarray(b, np.float32), (len(w),)).copy()
        return self.op(name, 'conv', source, out=len(w), kernel=list(w.shape[-2:]),
                       stride=1, padding=padding)


# Analytic geometry is evaluated on a supersampled unit square.
S = 56
Y, X = np.mgrid[:S, :S].astype(float)
X = (X + .5) / S
Y = (Y + .5) / S


def polygon(vertices):
    inside = np.zeros_like(X, dtype=bool)
    x0, y0 = vertices[-1]
    for x1, y1 in vertices:
        if abs(y1 - y0) > 1e-12:
            cross = ((y0 > Y) != (y1 > Y)) & (X < (x1-x0)*(Y-y0)/(y1-y0)+x0)
            inside ^= cross
        x0, y0 = x1, y1
    return inside


def ellipse(cx, cy, rx, ry):
    return ((X-cx)/rx)**2 + ((Y-cy)/ry)**2 <= 1


def segment(x0, y0, x1, y1, radius):
    t = np.clip(((X-x0)*(x1-x0)+(Y-y0)*(y1-y0))/((x1-x0)**2+(y1-y0)**2+1e-12), 0, 1)
    return (X-x0-t*(x1-x0))**2 + (Y-y0-t*(y1-y0))**2 < radius**2


def close_and_reduce(mask):
    # A soft edge represents uncertainty in the idealized manufactured outline.
    a = mask.astype(float).reshape(28, 2, 28, 2).mean((1, 3))
    p = np.pad(a, 2)
    d = np.maximum.reduce([p[i:i+30, j:j+30] for i in range(3) for j in range(3)])
    e = np.minimum.reduce([d[i:i+28, j:j+28] for i in range(3) for j in range(3)])
    a = (.75*e+.25*a).reshape(14, 2, 14, 2).mean((1, 3))
    # Small uncertainty near an idealized outline, rather than tracing detail.
    p = np.pad(a, 1)
    a = .70*a + .075*(p[1:-1,:-2]+p[1:-1,2:]+p[:-2,1:-1]+p[2:,1:-1])
    return a


def garment(body, hem, sleeve, spread, neck, bottom=.95, top=.045, neckstyle='round'):
    shoulder = body + .025
    armpit_y = top + .27
    cuff_y = min(top+sleeve, .98)
    cuff_width = .085 if sleeve > .45 else .12
    side = [(neck, top), (shoulder, top+.04),
            (spread, cuff_y-.055), (spread-cuff_width, cuff_y+.035),
            (body, armpit_y), (hem, bottom)]
    vertices = [(.5-x,y) for x,y in side] + [(.5+x,y) for x,y in reversed(side)]
    m = polygon(vertices)
    if neckstyle == 'round':
        m &= ~ellipse(.5, top-.015, neck, neck*.75)
    elif neckstyle == 'v':
        m &= ~polygon([(.5-neck, top-.02),(.5+neck,top-.02),(.5,top+neck*1.7)])
    return m


def dress(waist, hem, shoulder, sleeve, waist_y, neck):
    top, bottom = .04, .96
    side = [(neck, top), (shoulder, top+.055)]
    if sleeve:
        side += [(shoulder+sleeve*.35,top+sleeve),
                 (shoulder+sleeve*.35-.10,top+sleeve+.06),
                 (shoulder-.025,top+.26)]
    else:
        side += [(shoulder-.055,top+.24)]
    side += [(waist,waist_y),(hem,bottom)]
    m = polygon([(.5-x,y) for x,y in side]+[(.5+x,y) for x,y in reversed(side)])
    return m & ~ellipse(.5,top-.025,neck,neck)


def trousers(hip, ankle, gap, crotch, lean=0):
    bottom=.97
    v=[(.5-hip,.035),(.5+hip,.035),(.5+hip*1.05,.23),
       (.5+gap+2*ankle+lean,bottom),(.5+gap+lean,bottom),
       (.5,crotch),(.5-gap+lean,bottom),(.5-gap-2*ankle+lean,bottom),
       (.5-hip*1.05,.23)]
    return polygon(v)


def bag(width, body_top, bottom, taper, handle, handle_width=.21):
    m = polygon([(.5-width,body_top),(.5+width,body_top),
                 (.5+width*taper,bottom),(.5-width*taper,bottom)])
    if handle == 'loop':
        ry=max(.08,(body_top-.035)*.57)
        cy=body_top-ry*.65
        outer=ellipse(.5,cy,handle_width,ry)
        inner=ellipse(.5,cy,max(.03,handle_width-.035),max(.03,ry-.035))
        m |= outer & ~inner
    elif handle == 'strap':
        m |= segment(.5-width*.85,body_top+.04,.5,.035,.018)
        m |= segment(.5, .035,.5+width*.85,body_top+.04,.018)
    elif handle == 'backpack':
        m |= ellipse(.5,body_top+.06,width,.17)
        m |= ellipse(.5,body_top-.06,.085,.09) & ~ellipse(.5,body_top-.06,.05,.06)
    return m


def shoe(height, bottom, kind, thick=.055, shaft=.32, heel=.0, band=.065):
    # Side-view geometry: rounded toe to the left, heel to the right.
    if kind == 'sneaker':
        v=[(.02,bottom-.035),(.07,bottom-height*.32),(.34,bottom-height*.43),
           (.59,bottom-height),(.69,bottom-height*.77),(.82,bottom-height*.65),
           (.94,bottom-height*.89),(.98,bottom-.07),(.95,bottom),(.10,bottom)]
        return polygon(v)
    if kind == 'boot':
        v=[(.02,bottom-.035),(.08,bottom-.12),(.36,bottom-.18),
           (.57,bottom-.29),(.94-shaft,bottom-height),(.92,bottom-height),
           (.96,bottom-.10),(.95,bottom),(.82,bottom),(.78,bottom-heel),
           (.57,bottom-heel),(.33,bottom),(.08,bottom)]
        return polygon(v)
    # Sandal: sole/footbed, open toe straps and optional ankle fastening.
    foot_y = bottom-height*.30
    m=polygon([(.025,bottom-.03),(.20,bottom-.07),(.52,bottom-.10),
               (.90,foot_y),(.96,foot_y+.035),(.91,foot_y+thick),
               (.50,bottom-.035),(.16,bottom+.005),(.035,bottom)])
    if heel:
        m |= polygon([(.82,foot_y),(.94,foot_y),(.91,bottom),(.85,bottom)])
    m |= segment(.20,bottom-.055,.29,bottom-height*.48,band*.5)
    m |= segment(.29,bottom-height*.48,.47,bottom-.075,band*.5)
    m |= segment(.46,bottom-.075,.77,bottom-height*.80,band*.5)
    m |= segment(.77,bottom-height*.80,.88,foot_y,band*.5)
    if height>.30:
        m |= segment(.73,bottom-height*.82,.94,bottom-height,band*.40)
        m |= segment(.94,bottom-height,.92,foot_y,band*.40)
    return m


def make_templates():
    t=[[] for _ in range(10)]
    def put(c,m,prior=0,flip=False):
        a=close_and_reduce(m)
        t[c].append((a,prior))
        if flip:
            t[c].append((a[:,::-1],prior))

    # Tops: sleeve length, torso fit, neckline and hem describe construction.
    for body in (.25,.29,.33,.37):
        for hem_ratio in (.85,1.,1.08):
            for spread in (.44,.49):
                for neck in (.075,.12):
                    for sl in (.23,.32):
                        put(0,garment(body,body*hem_ratio,sl,spread,neck))
                    for sl in (.65,.78,.90):
                        put(0,garment(body,body*hem_ratio,sl,spread,neck),-.30)
                        # Sweater cuffs normally reach their ribbed hem;
                        # an overcoat's body often extends below its cuffs.
                        put(2,garment(body,body*hem_ratio,sl,spread,neck),
                            -.25 if sl < .70 else 0)
                        put(4,garment(body,body*hem_ratio,sl,spread,neck,neckstyle='v'),
                            -.14 if sl > .85 else -.02)
                        put(6,garment(body,body*hem_ratio,sl,spread,neck,neckstyle='v'),-.13)
                    for sl in (.23,.34):
                        put(6,garment(body,body*hem_ratio,sl,spread,neck,neckstyle='v'),-.13)
    # Sleeveless undershirts, camisoles, and vest-like tops.
    for width in (.20,.26,.32):
        for neckline in (.10,.15,.20):
            m=polygon([(.5-width,.05),(.5+width,.05),(.5+width*.78,.35),
                       (.5+width,.95),(.5-width,.95),(.5-width*.78,.35)])
            m &= ~ellipse(.5,.025,neckline,.19)
            put(0,m,-.08)
            put(6,m,-.24)
    # Long open outer garments can extend below their sleeves.
    for body in (.27,.32,.37):
        for sl in (.66,.78):
            for neck in (.08,.14,.20):
                put(4,garment(body,body*1.08,sl,.47,neck,neckstyle='v'),-.02)
    # Trousers, including tapered, straight and wide legs.
    for hip in (.13,.18,.23,.29):
        for ankle in (.035,.055,.080,.11):
            for gap in (.015,.05,.085):
                if gap+2*ankle > hip*1.5:
                    continue
                for crotch in (.35,.48,.60):
                    for lean in (-.035,0,.035):
                        put(1,trousers(hip,ankle,gap,crotch,lean))
    # Fitted bodice and a skirt, with independently variable sleeve type.
    for waist in (.12,.17,.22):
        for hem in (.19,.27,.36,.44):
            for shoulder in (.18,.24,.30):
                for sl in (0,.18,.30,.68):
                    for wy in (.38,.51):
                        put(3,dress(waist,hem,shoulder,sl,wy,.09),-.18)
    # Bags: tote, handbag, shoulder strap, clutch, and backpack.
    for width in (.19,.28,.37,.46):
        for bt in (.20,.36,.52):
            for bottom in (.83,.96):
                for taper in (.78,1.,1.12):
                    for handle in ('loop','strap','backpack','none'):
                        put(8,bag(width,bt,bottom,taper,handle,min(.22,width*.68)))
    # Both orientations are included, without measuring the dataset's pose.
    for bottom in (.75,.86,.97):
        for height in (.24,.33,.42,.51):
            put(7,shoe(height,bottom,'sneaker'),flip=True)
        for height in (.39,.51,.65,.80,.92):
            if bottom-height < -.025:
                continue
            for shaft in (.25,.34,.43):
                for heel in (0,.10):
                    put(9,shoe(height,bottom,'boot',shaft=shaft,heel=heel),flip=True)
        for height in (.20,.32,.44,.56,.68):
            for band in (.045,.085,.13):
                for heel in (0,1):
                    put(5,shoe(height,bottom,'sandal',heel=heel,band=band),flip=True)
    return t


def rect(n,x0,x1,y0,y1):
    yy,xx=np.mgrid[:n,:n]
    xx=(xx+.5)/n; yy=(yy+.5)/n
    a=((xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)).astype(float)
    return a/max(1,a.sum())


def build(destination):
    net=Network()
    # The signed spatial duplication has exactly zero mean. Instance norm
    # therefore supplies RMS contrast normalization, without data statistics.
    net.conv('signed_image','image',np.array([1.,-1.]).reshape(2,1,1,1))
    net.op('signed_plane','reshape','signed_image',shape=[1,56,28])
    net.op('rms_normalized','norm','signed_plane')
    net.op('signed_channels','reshape','rms_normalized',shape=[2,28,28])
    net.conv('relative_intensity','signed_channels',np.array([1.,0.]).reshape(1,2,1,1))
    # Two fixed soft foreground ramps, expressed in relative intensity units.
    net.conv('thresholds','relative_intensity',np.array([10.,10.,4.,4.]).reshape(4,1,1,1),
             [-.20,-1.20,-.20,-1.20])
    net.op('threshold_relu','relu','thresholds')
    net.conv('foreground','threshold_relu',np.array([.65,-.65,.35,-.35]).reshape(1,4,1,1))
    net.conv('pad_foreground','foreground',np.ones((1,1,1,1)),padding=2)
    net.op('dilate','maxpool','pad_foreground',size=[3,3],stride=[1,1])
    net.conv('negative_dilation','dilate',-np.ones((1,1,1,1)))
    net.op('negative_closing','maxpool','negative_dilation',size=[3,3],stride=[1,1])
    net.conv('closed','negative_closing',-np.ones((1,1,1,1)))
    net.conv('closed_weight','closed',np.array([.75]).reshape(1,1,1,1))
    net.conv('foreground_weight','foreground',np.array([.25]).reshape(1,1,1,1))
    net.op('shape_occupancy','add',['closed_weight','foreground_weight'])
    net.op('shape_image','avgpool','shape_occupancy',size=[2,2],stride=[2,2])

    templates=make_templates()
    count=max(map(len,templates))
    w=np.zeros((10*count,196),np.float32)
    b=np.full(10*count,-100.,np.float32)
    for c,ts in enumerate(templates):
        for j,(a,prior) in enumerate(ts):
            v=2*a.ravel()-1
            # A weighted Hamming compatibility, uncertain at analytic edges.
            scale=16/196
            w[c*count+j]=v*scale
            b[c*count+j]=-np.maximum(v,0).sum()*scale+prior
            # Long narrow objects need actual leg separation to be trousers.
            if c == 1:
                b[c*count+j] -= .05
    net.dense('geometric_compatibility','shape_image',w,b)
    net.op('geometric_grid','reshape','geometric_compatibility',shape=[10,1,count])
    net.op('best_geometry','maxpool','geometric_grid',size=[1,count],stride=[1,count])
    net.op('shape_scores','reshape','best_geometry',shape=[10])

    # Appearance cues are contrast measurements, not a fitted classifier.
    net.op('contrast_normalized','norm','image')
    seam=[]
    for x in (.46,.50,.54):
        for width in (.036,.072):
            for polarity in (-1,1):
                for y0,y1 in ((.22,.40),(.40,.60),(.60,.80)):
                    center=rect(28,x-width/2,x+width/2,y0,y1)
                    sides=.5*(rect(28,x-width*1.5,x-width/2,y0,y1)+
                               rect(28,x+width/2,x+width*1.5,y0,y1))
                    seam.append(((center-sides)*polarity).ravel())
    net.dense('placket_contrast','contrast_normalized',seam)
    net.op('positive_placket','relu','placket_contrast')
    net.op('placket_grid','reshape','positive_placket',shape=[1,12,3])
    net.conv('negative_placket','placket_grid',-np.ones((1,1,1,1)))
    net.op('weakest_placket_band','maxpool','negative_placket',size=[1,3],stride=[1,3])
    net.conv('continuous_placket','weakest_placket_band',-np.ones((1,1,1,1)))
    net.op('best_placket','maxpool','continuous_placket',size=[12,1],stride=[12,1])
    net.dense('placket_threshold','best_placket',[[1.]],[-.12])
    net.op('clear_placket','relu','placket_threshold')
    net.dense('opening_evidence','clear_placket',np.array([-.25,0,-.40,0,1.20,0,1.35,0,0,0])[:,None])

    # Auxiliary tests of garment construction. Each row is an explicitly
    # chosen regional contrast; each rule is a fixed piecewise-linear gate.
    region_specs=[
        (.45,.55,.65,.95),  # crotch/leg separation, below the middle
        (.32,.43,.65,.95),  # left leg
        (.57,.68,.65,.95),  # right leg
        (.15,.85,.08,.40),  # shoulders or upper handle space
        (.10,.90,.52,.84),  # lower object body
        (.30,.70,.40,.78),  # central torso / closed shoe upper
        (.06,.27,.48,.82),  # left sleeve
        (.73,.94,.48,.82),  # right sleeve
        (.30,.70,.10,.35),  # central upper region
        (.05,.30,.10,.35),  # upper left
        (.70,.95,.10,.35),  # upper right
        (.08,.28,.46,.73),  # outer lower sleeve, left
        (.72,.92,.46,.73),  # outer lower sleeve, right
        (.05,.28,.13,.36),  # short sleeve, left
        (.72,.95,.13,.36),  # short sleeve, right
        (.30,.70,.79,.95),  # hem centre
        (.12,.29,.75,.94),  # outer skirt left
        (.71,.88,.75,.94),  # outer skirt right
        (.12,.29,.37,.57),  # waist exterior left
        (.71,.88,.37,.57),  # waist exterior right
    ]
    regions=np.array([rect(28,*r).ravel() for r in region_specs])
    net.dense('construction_regions','foreground',regions)
    rules=[]; rb=[]; outputs=[]
    def rule(coefs, bias, contribution):
        row=np.zeros(len(regions))
        for i,v in coefs.items(): row[i]=v
        rules.append(row); rb.append(bias); outputs.append(contribution)
    # A visible gap with material on both sides is evidence for two legs.
    rule({0:-1,1:.5,2:.5},-.10,{1:2.0,3:-.5})
    # Conversely, a continuous skirt or torso should not be called trousers.
    rule({0:1,1:-.35,2:-.35},-.15,{1:-1.8})
    # A bag body beneath empty upper space differs from a solid shouldered top.
    rule({3:-1,4:1},-.30,{8:.65})
    rule({8:1,9:.15,10:.15},-.65,{8:-1.2})
    # Open footwear lacks the filled-in vamp of a sneaker.
    rule({5:-1},.28,{5:.8,7:-.5,9:-.25})
    # Material at both sides high up, but not beside the waist, suggests a tee.
    rule({13:.5,14:.5,11:-.5,12:-.5},-.12,{0:1.0,6:.25,3:-.45})
    # Long sleeves distinguish an upper garment from a simple skirt profile.
    rule({11:.5,12:.5},-.42,{2:.35,4:.35,6:.35,3:-.30})
    # A widening skirt is distinct from the straight torso of a shirt.
    rule({16:.5,17:.5,18:-.5,19:-.5},-.12,{3:.75})
    # A narrow shoulder span and filled hem support a dress, not a wide tee.
    rule({13:-.5,14:-.5,15:.25},.05,{3:1.5,0:-.3})
    net.dense('construction_tests','construction_regions',rules,rb)
    net.op('construction_gates','relu','construction_tests')
    ow=np.zeros((10,len(rules)))
    for i,d in enumerate(outputs):
        for c,v in d.items(): ow[c,i]=v
    net.dense('construction_evidence','construction_gates',ow)

    # A collar is modeled as a V opening with cloth on either side. Separate
    # short shirt collars from the longer lapels of outerwear. Contrasting
    # cloth around a dark opening supplies the evidence. No observed patches.
    collars=[]
    yy,xx=np.mgrid[:28,:28]
    xx=(xx+.5)/28; yy=(yy+.5)/28
    for length in (.14,.25):
        for top in (.02,.07,.12):
            for halfwidth in (.09,.14,.19):
                for shift in (-.035,0,.035):
                    xdist=np.abs(xx-.5-shift)
                    half=halfwidth*(1-(yy-top)/length)
                    valid=(yy>top+.03)&(yy<top+length)
                    inner=valid & (xdist<half)
                    outer=valid & (xdist>=half) & (xdist<half+.065)
                    f=outer/max(1,outer.sum())-inner/max(1,inner.sum())
                    collars.append(f.ravel())
    net.dense('collar_contrasts','contrast_normalized',collars,-.22)
    net.op('collar_relu','relu','collar_contrasts')
    net.op('collar_grid','reshape','collar_relu',shape=[2,1,27])
    net.op('best_collar','maxpool','collar_grid',size=[1,27],stride=[1,27])
    # A positive score is evidence only; absence is not a collar decision.
    cw=np.zeros((10,2))
    cw[0]=[-.08,-.06]; cw[2]=[-.08,-.10]
    cw[4]=[.05,.30]; cw[6]=[.18,.08]
    net.dense('collar_evidence','best_collar',cw)

    # A side-view shoe is horizontally elongated. Compare its occupied area
    # with its bounding span to separate a closed vamp from narrow straps.
    net.op('horizontal_extent','maxpool','foreground',size=[28,1],stride=[28,1])
    net.op('vertical_extent','maxpool','foreground',size=[1,28],stride=[1,28])
    net.dense('footwear_span','vertical_extent',np.ones((1,28))/28)
    net.dense('footwear_area','foreground',np.ones((1,784))/784)
    net.dense('span_density_term','footwear_span',[[.72]])
    net.dense('area_density_term','footwear_area',[[-1.]])
    net.op('openness','add',['span_density_term','area_density_term'])
    net.dense('open_strap_test','openness',[[1.]],[-.020])
    net.op('open_strap_gate','relu','open_strap_test')
    net.dense('open_strap_evidence','open_strap_gate',
              np.array([0,0,0,0,0,5.5,0,-2.,0,-1.])[:,None])
    net.dense('closed_vamp_test','openness',[[-1.]],[-.015])
    net.op('closed_vamp_gate','relu','closed_vamp_test')
    net.dense('closed_vamp_evidence','closed_vamp_gate',
              np.array([0,0,0,0,0,-1.0,0,.65,0,.45])[:,None])
    net.op('class_scores','add',['shape_scores','opening_evidence','construction_evidence',
                                 'collar_evidence','open_strap_evidence','closed_vamp_evidence'])
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'network.json').open('w') as f:
        json.dump({'layers':net.layers},f,indent=2)
    np.savez(destination/'weights.npz',**net.weights)
    print('Analytic templates per class:',[len(x) for x in templates])
    print('Weight entries:',sum(v.size for v in net.weights.values()))


if __name__=='__main__':
    build(sys.argv[1])
