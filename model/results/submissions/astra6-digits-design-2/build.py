"""Data-free handwritten digit shape matcher.

Only mathematical constants, typed generic pen paths, and seeded geometric
variations produce the weights. No files are read, and there is no training,
optimization, fitted readout, or imported font/image asset.
"""
import sys
import json
from pathlib import Path
import numpy as np

PER_CLASS = 8192
SEED = 19670423


def bezier(a, b, c, d, n=18):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1-t)**3*np.array(a) + 3*(1-t)**2*t*np.array(b)
            + 3*(1-t)*t*t*np.array(c) + t**3*np.array(d))


def line(*points):
    return np.asarray(points, dtype=np.float64)


def chain(*parts):
    return np.concatenate(parts)


def oval(cx, cy, rx, ry, power=2.0, n=73):
    a = np.linspace(0, 2*np.pi, n)
    c, s = np.cos(a), np.sin(a)
    return np.stack((cx + rx*np.sign(c)*abs(c)**(2/power),
                     cy + ry*np.sign(s)*abs(s)**(2/power)), axis=1)


def paths(k, style, rng):
    """Unit-coordinate pen paths, specified from ordinary digit anatomy."""
    u = rng.uniform
    if k == 0:
        p = oval(.5, .5, .5, .5, u(1.8, 2.6))
        p[:, 0] = .5 + (p[:, 0]-.5)*(1+u(-.22, .22)*(p[:, 1]-.5))
        return [p]
    if k == 1:
        top = u(.48, .65)
        bottom = u(.35, .57)
        p = bezier((top, 0), (top+u(-.06,.06), .30),
                   (bottom+u(-.07,.07), .75), (bottom, 1))
        result = [p]
        if style in (2, 3, 4, 6, 7):
            result.append(line((top-u(.18,.55), u(.12,.32)), (top,0)))
        if style in (5, 6, 7):
            result.append(bezier((bottom-u(.2,.45),1), (bottom-.10,.98),
                                 (bottom+.12,.98), (bottom+u(.2,.45),1)))
        return result
    if k == 2:
        a = (u(0,.13), u(.15,.30))
        b = (1., u(.15,.28))
        bend = u(.44,.62)
        top = bezier(a, (u(-.05,.10),u(-.08,.03)),
                     (u(.75,.98),u(-.10,.02)), b)
        middle = bezier(b, (1.02,bend-.15), (u(.20,.55),bend+.18),
                        (u(-.03,.09),u(.90,1.0)))
        end = tuple(middle[-1])
        if style in (0,1,2,3):
            bottom = bezier(end, (.30,u(.87,1.05)), (.72,u(.90,1.06)),
                            (1,u(.88,1.03)))
        elif style in (4,5):
            bottom = bezier(end, (-.12,1.18), (.62,1.10), (1,.85))
        else:
            # Cursive 2: descending diagonal curls into a lower-left loop,
            # then crosses it on the way to the right-hand terminal.
            middle = bezier(b, (1.15,.52), (.25,1.06), (.03,.99))
            loop = bezier((.03,.99), (-.35,.92), (.12,.58), (.40,.69))
            terminal = bezier((.40,.69), (.65,.80), (.82,.94), (1.13,.95))
            return [chain(top,middle,loop,terminal)]
        return [chain(top,middle,bottom)]
    if k == 3:
        waist = (u(.12,.52),u(.36,.57))
        if style in (0,1,2,3,4,5):
            top = chain(bezier((0,u(.06,.19)), (.28,u(-.12,.02)),
                               (.95,u(-.10,.05)), (1,u(.14,.24))),
                        bezier((1,.20), (1,.34), (.72,waist[1]), waist))
        else:
            top = line((0,.03),(1,0),waist)
        bottom = chain(bezier(waist, (.92,waist[1]-.09),
                              (1.04,.60), (1,.77)),
                       bezier((1,.77), (.96,1.08), (.25,1.11),
                              (0,u(.86,1.00))))
        result = chain(top,bottom)
        balance = u(-.36,.36)
        result[:,0] = .5 + (result[:,0]-.5)*(1+balance*np.tanh((result[:,1]-waist[1])*9))
        return [result]
    if k == 4:
        cross = u(.48,.68)
        stem = u(.68,.86)
        if style in (0,1,2,3):
            lefttop = (u(.08,.38),0)
        else:
            lefttop = (stem,u(-.04,.08))
        left = bezier(lefttop, (lefttop[0]-.12,.19),
                      (u(-.06,.08),cross-.08), (0,cross))
        crossbar = line((0,cross),(1,u(cross-.04,cross+.04)))
        right = bezier((stem,u(-.04,.12)), (stem+u(-.06,.06),.31),
                       (stem+u(-.07,.07),.73), (stem+u(-.12,.05),1))
        return [chain(left,crossbar),right]
    if k == 5:
        shoulder = u(.30,.56)
        corner = (u(.02,.30),u(-.02,.12))
        turn = (u(-.03,.22),shoulder)
        top = bezier((1,u(-.08,.06)), (.75,.02), (.35,.02), corner)
        left = bezier(corner,(corner[0]-.04,.18),(turn[0]-.12,shoulder-.08),turn)
        bowl = chain(bezier(turn, (.53,shoulder-u(.08,.22)),
                            (1.03,shoulder+.03), (1,.73)),
                     bezier((1,.73), (1.02,1.08), (.23,1.16), (0,u(.77,1.0))))
        if style in (4,5,6,7):
            # Freely written 5: a softer, oblique spine and a deep bowl.
            turn = (u(.03,.28),shoulder)
            left = bezier(corner,(-.10,.28),(.30,.32),turn)
            bowl = chain(bezier(turn,(.78,.55),(1.05,.63),(.85,.84)),
                         bezier((.85,.84),(.60,1.12),(-.03,1.08),(0,.80)))
        return [top,chain(left,bowl)]
    if k == 6:
        shoulder = u(.40,.69)
        leftmeet = (u(0,.08),u(.68,.78))
        if style in (0,1,2,3,4,5):
            back = bezier((u(.70,1.05),0), (.23,u(-.14,.02)),
                          (-.10,.47), leftmeet)
        else:
            back = bezier((u(.70,1.05),0), (.48,.24), (.02,.41), leftmeet)
        loop = chain(bezier(leftmeet, (.05,1.09), (1.08,1.13), (1,.75)),
                     bezier((1,.75), (.98,shoulder-.11), (.16,shoulder-.13), leftmeet))
        return [chain(back,loop)]
    if k == 7:
        top = bezier((0,u(0,.10)), (.3,u(-.06,.08)), (.7,u(-.07,.07)), (1,0))
        foot = u(.15,.47)
        stem = bezier((1,0), (u(.72,1),.32), (foot+u(.02,.18),.70), (foot,1))
        result = [chain(top,stem)]
        if style in (3,4):
            result.append(line((0,.18),(0,.04)))
        if style in (5,6,7):
            yy = u(.42,.60)
            result.append(line((.24,yy+.02),(.88,yy-.02)))
        return result
    if k == 8:
        mid = u(.34,.58)
        wx = u(.42,.58)
        if style in (0,1,2,3):
            top = oval(wx,mid/2,u(.35,.47),mid/2,u(1.8,2.4))
            bottom = oval(.5,(mid+1)/2,.5,(1-mid)/2,u(1.8,2.5))
            return [top,bottom]
        waist = (wx,mid)
        return [chain(bezier(waist,(u(-.12,.08),.13),(.20,-.12),(.70,.01)),
                      bezier((.70,.01),(1.22,.12),(.79,mid-.01),waist),
                      bezier(waist,(-.16,mid+.12),(-.04,1.13),(.58,1.0)),
                      bezier((.58,1.0),(1.20,.94),(1.13,mid+.23),waist))]
    if k == 9:
        loopbottom = u(.28,.62)
        loop = oval(.46,loopbottom/2,.46,loopbottom/2,u(1.8,2.7))
        start = (.90,loopbottom*u(.25,.65))
        if style in (0,1,2,3,4,5):
            tail = bezier(start, (.96,.46), (u(.30,.98),.82), (u(.05,.98),1))
        else:
            tail = chain(bezier(start,(1,.50),(1.01,.82),(.75,.97)),
                         bezier((.75,.97),(.54,1.10),(.19,1.09),(.10,.92)))
        return [loop,tail]
    raise ValueError(k)


WIDTHS = [(7.5,17.0),(8.0,13.0),(9.5,18.5),(7.0,16.5),(9.0,18.0),
          (8.0,17.0),(7.0,15.5),(9.0,18.5),(5.5,15.5),(6.0,15.5)]


def geometric_model(k, style, rng):
    pp = paths(k,style,rng)
    allp = np.concatenate(pp)
    lo, hi = allp.min(0), allp.max(0)
    # '1' retains the narrow natural extent of its simple pen stroke.
    for p in pp:
        p[:,1] = (p[:,1]-lo[1])/(hi[1]-lo[1])
        if k != 1:
            p[:,0] = (p[:,0]-lo[0])/(hi[0]-lo[0])
    width = rng.uniform(*WIDTHS[k])
    height = rng.uniform(16.5,19.5)
    slant = rng.uniform(-.45,.75)
    bow = rng.uniform(-.13,.13)
    taper = rng.uniform(-.34,.34)
    tilt = rng.uniform(-.10,.10)
    thickness = rng.uniform(1.4,3.5)
    if k in (0,6,8,9):
        # A loop model must leave space for a visible interior.
        thickness = min(thickness,.29*width)
    for p in pp:
        x, y = p[:,0]-.5, p[:,1]-.5
        xx = width*(x*(1+taper*y)+bow*(4*y*y-1)) - slant*height*y
        yy = height*(y+tilt*x)
        p[:,0],p[:,1] = xx,yy
    allp = np.concatenate(pp)
    extent = np.ptp(allp,axis=0)
    scale = min(1.0,(20.5-thickness)/max(extent))
    pp = [p*scale for p in pp]
    # Centre of mass of the abstract constant-width pen paths, not of data.
    mids, lens = [], []
    for p in pp:
        ll = np.linalg.norm(np.diff(p,axis=0),axis=1)
        mids.append((p[1:]+p[:-1])/2)
        lens.append(ll)
    lens = np.concatenate(lens)
    centre = (np.concatenate(mids)*lens[:,None]).sum(0)/lens.sum()
    centre -= np.array([13.5,13.5]) + rng.uniform(-.50,.50,size=2)
    # Uniformly spaced samples approximate the mathematical pen path.
    samples = []
    for p in pp:
        p = p-centre
        ll = np.linalg.norm(np.diff(p,axis=0),axis=1)
        tt = np.r_[0,np.cumsum(ll)]
        keep = np.r_[True,np.diff(tt)>1e-9]
        tt,p = tt[keep],p[keep]
        t = np.linspace(0,tt[-1],max(2,int(np.ceil(tt[-1]/.32))+1))
        samples.append(np.stack([np.interp(t,tt,p[:,j]) for j in range(2)],axis=1))
    return np.concatenate(samples),thickness


def analytic_holes(raw):
    """Enclosed complement of a formula-generated pen raster, no data."""
    background = raw < .15
    seen = np.zeros_like(background,dtype=bool)
    todo = [(y,x) for y in range(28) for x in range(28)
            if (y in (0,27) or x in (0,27)) and background[y,x]]
    for y,x in todo:
        seen[y,x] = True
    pos = 0
    while pos < len(todo):
        y,x=todo[pos];pos+=1
        for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            yy,xx=y+dy,x+dx
            if 0<=yy<28 and 0<=xx<28 and background[yy,xx] and not seen[yy,xx]:
                seen[yy,xx]=True;todo.append((yy,xx))
    return (background & ~seen).astype(np.float64)


def build(out_dir):
    layers,weights = [],{}
    def op(name,kind,inputs,**settings):
        layers.append(dict(name=name,op=kind,inputs=inputs,**settings))
        return name
    def conv(name,source,W,b,padding=0,stride=1):
        W = np.asarray(W,dtype=np.float32)
        op(name,'conv',[source],out=len(W),kernel=list(W.shape[-2:]),
           stride=stride,padding=padding)
        weights[name+'.W'],weights[name+'.b'] = W,np.asarray(b,dtype=np.float32)
        return name
    def dense(name,source,W,b):
        W = np.asarray(W,dtype=np.float32)
        op(name,'dense',[source],out=len(W))
        weights[name+'.W'],weights[name+'.b'] = W,np.asarray(b,dtype=np.float32)
        return name
    def pool(name,source,size,kind='avgpool',stride=None):
        return op(name,kind,[source],size=list(size),stride=list(stride or size))

    x = np.arange(-2,3,dtype=np.float64)
    gaussian = np.exp(-(x[:,None]**2+x[None,:]**2)/(2*.8**2))
    gaussian /= gaussian.sum()
    conv('gaussian','image',gaussian[None,None], [0],padding=2)
    pool('smooth','gaussian',(2,2))
    op('contrast','norm',['smooth'])

    # Eight signed Sobel orientations, followed by local histogram pooling.
    # This encodes stroke direction with more positional tolerance than pixels.
    sx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float64)/8
    sy=sx.T
    angles=np.arange(8)*np.pi/4
    oriented=np.stack([np.cos(a)*sx+np.sin(a)*sy for a in angles])
    conv('orientation','gaussian',oriented[:,None],np.zeros(8),padding=1)
    op('positive_orientation','relu',['orientation'])
    pool('orientation_histograms','positive_orientation',(4,4))
    op('histogram_plane','reshape',['orientation_histograms'],shape=[1,8,49])
    op('normalized_histograms','norm',['histogram_plane'])

    # Saturated ink is min(2I,1), expressed exactly by ReLUs.
    conv('ink_ramps','image',np.array([2.,2.]).reshape(2,1,1,1),[0,-1])
    op('ink_positive','relu',['ink_ramps'])
    conv('ink','ink_positive',np.array([1.,-1.]).reshape(1,2,1,1),[0])
    pool('coarse_ink','ink',(2,2))
    conv('padded_ink','ink',np.ones((1,1,1,1)),[0],padding=1)
    pool('near_ink','padded_ink',(3,3),'maxpool',(1,1))
    pool('proximity','near_ink',(2,2))

    # Background reachability is a fixed recurrent morphological circuit.
    # The unrolled graph has no fitted parameters and uses 8-neighbour steps.
    conv('background_ramps','image',np.array([-10.,-10.]).reshape(2,1,1,1),[2.,1.])
    op('background_positive','relu',['background_ramps'])
    conv('background','background_positive',np.array([1.,-1.]).reshape(1,2,1,1),[0])
    border = np.zeros((1,28,28),dtype=np.float32)
    border[:,0,:]=border[:,-1,:]=border[:,:,0]=border[:,:,-1]=1
    pool('image_mean','image',(28,28))
    op('border_seed','reshape',[dense('border_constants','image_mean',np.zeros((784,1)),border.ravel())],shape=[1,28,28])
    reach = 'border_seed'
    identity = np.ones((1,1,1,1),dtype=np.float32)
    conv('negative_background','background',-identity,[0])
    for step in range(32):
        stem = 'flood_%02d_' % step
        conv(stem+'pad',reach,identity,[0],padding=1)
        pool(stem+'expanded',stem+'pad',(3,3),'maxpool',(1,1))
        op(stem+'difference','add',[stem+'expanded','negative_background'])
        op(stem+'excess','relu',[stem+'difference'])
        conv(stem+'subtract',stem+'excess',-identity,[0])
        reach=op(stem+'reachable','add',[stem+'expanded',stem+'subtract'])
    conv('negative_reachable',reach,-identity,[0])
    op('holes','add',['background','negative_reachable'])
    pool('hole_features','holes',(2,2))

    n = 10*PER_CLASS
    correlation = np.empty((n,196),dtype=np.float32)
    coverage = np.empty_like(correlation)
    excess = np.empty_like(correlation)
    loop_evidence = np.empty_like(correlation)
    direction_correlation = np.empty((n,392),dtype=np.float32)
    model_bias = np.empty(n,dtype=np.float32)
    axis = np.arange(14)*2+.5
    yy,xx = np.meshgrid(axis,axis,indexing='ij')
    grid = np.stack([xx.ravel(),yy.ravel()],axis=1)
    yy_full,xx_full = np.mgrid[:28,:28]
    grid_full = np.stack([xx_full.ravel(),yy_full.ravel()],axis=1)
    for k in range(10):
        rng = np.random.default_rng(SEED+k*104729)
        for j in range(PER_CLASS):
            pts,thickness = geometric_model(k,j%8,rng)
            d2 = ((grid[:,None,:]-pts[None,:,:])**2).sum(2).min(1)
            d2_full = ((grid_full[:,None,:]-pts[None,:,:])**2).sum(2).min(1)
            # Render the analytic pen on the pixel grid, then apply exactly
            # the fixed blur used by the image branch. This is evaluation
            # of a formula, not fitting a model to images.
            raw = np.clip(thickness/2+.5-np.sqrt(d2_full),0,1).reshape(28,28)
            padded = np.pad(raw,2)
            blurred = sum(gaussian[dy,dx]*padded[dy:dy+28,dx:dx+28]
                          for dy in range(5) for dx in range(5))
            q = blurred.reshape(14,2,14,2).mean(axis=(1,3)).ravel()
            q -= q.mean()
            row = k*PER_CLASS+j
            correlation[row] = (.8/14)*q/max(np.linalg.norm(q),1e-8)
            bp=np.pad(blurred,1)
            gx=sum(sx[dy,dx]*bp[dy:dy+28,dx:dx+28] for dy in range(3) for dx in range(3))
            gy=sum(sy[dy,dx]*bp[dy:dy+28,dx:dx+28] for dy in range(3) for dx in range(3))
            hist=np.maximum(np.cos(angles)[:,None,None]*gx+np.sin(angles)[:,None,None]*gy,0)
            hist=hist.reshape(8,7,4,7,4).mean(axis=(2,4)).ravel()
            hist-=hist.mean()
            direction_correlation[row]=(.35/np.sqrt(392))*hist/max(np.linalg.norm(hist),1e-8)
            skeleton = np.exp(-d2/(2*.65**2))
            coverage[row] = .5*skeleton/skeleton.sum()
            expected_ink = np.minimum(2*raw,1).sum()/4
            penalty = np.minimum(np.maximum(np.sqrt(d2)-.55*thickness,0)/3,1.5)
            excess[row] = -.5*penalty/max(expected_ink,1.)
            hh = analytic_holes(raw)
            hh = hh.reshape(14,2,14,2).mean(axis=(1,3)).ravel()
            # Positive evidence only: a visible enclosed region should
            # coincide with a model interior. Absence is not evidence
            # against a loop because handwriting often leaves a gap.
            area = max(hh.sum(),10.)
            loop_evidence[row] = .25*(2*hh-1)/area
            model_bias[row] = 0.
        print('constructed analytic models for digit',k,flush=True)
    zeros = np.zeros(n,dtype=np.float32)
    dense('shape_correlation','contrast',correlation,zeros)
    dense('stroke_coverage','proximity',coverage,zeros)
    dense('unexplained_ink','coarse_ink',excess,zeros)
    dense('loop_agreement','hole_features',loop_evidence,model_bias)
    dense('stroke_directions','normalized_histograms',direction_correlation,zeros)
    op('model_scores','add',['shape_correlation','stroke_coverage','unexplained_ink','loop_agreement','stroke_directions'])
    op('by_class','reshape',['model_scores'],shape=[10,1,PER_CLASS])
    pool('best_model','by_class',(1,PER_CLASS),'maxpool')
    op('scores','reshape',['best_model'],shape=[10])
    out = Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    (out/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(out/'weights.npz',**weights)
    print('weights:',sum(w.size for w in weights.values()),flush=True)


if __name__ == '__main__':
    build(sys.argv[1])
