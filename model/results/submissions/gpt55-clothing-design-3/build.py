#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np

# Data-free generator for a hand-written Fashion-MNIST-style classifier.
# All numbers are formulaic geometric constants; no files are read.

N = 28
xs = (np.arange(N, dtype=np.float32) + 0.5 - N/2) / (N/2)
ys = (np.arange(N, dtype=np.float32) + 0.5 - N/2) / (N/2)
X, Y = np.meshgrid(xs, ys)


def sig(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def rect(cx, cy, w, h, e=0.035):
    return sig((w/2 - np.abs(X-cx))/e) * sig((h/2 - np.abs(Y-cy))/e)


def ellipse(cx, cy, rx, ry, e=0.055):
    r = 1.0 - ((X-cx)/rx)**2 - ((Y-cy)/ry)**2
    return sig(r/e)


def trap(y0, y1, w0, w1, cx0=0.0, cx1=0.0, e=0.035):
    t = np.clip((Y-y0)/(y1-y0), 0, 1)
    hw = (w0*(1-t) + w1*t)/2
    cx = cx0*(1-t) + cx1*t
    inside_y = sig((Y-y0)/e) * sig((y1-Y)/e)
    inside_x = sig((hw - np.abs(X-cx))/e)
    return inside_x * inside_y


def line(x1, y1, x2, y2, th=0.045, e=0.018):
    vx, vy = x2-x1, y2-y1
    wx, wy = X-x1, Y-y1
    denom = vx*vx + vy*vy + 1e-12
    t = np.clip((wx*vx + wy*vy)/denom, 0, 1)
    px, py = x1 + t*vx, y1 + t*vy
    d = np.sqrt((X-px)**2 + (Y-py)**2)
    return sig((th-d)/e)


def ering(cx, cy, rx, ry, th=0.12, e=0.035, top_only=False, bottom_only=False):
    r = np.sqrt(((X-cx)/rx)**2 + ((Y-cy)/ry)**2)
    m = sig((th - np.abs(r-1.0))/e)
    if top_only:
        m *= sig((cy - Y)/0.035)
    if bottom_only:
        m *= sig((Y - cy)/0.035)
    return m


def add(a, *parts):
    out = np.array(a, dtype=np.float32)
    for p in parts:
        out = np.maximum(out, p.astype(np.float32))
    return out


def cut(a, *holes):
    out = np.array(a, dtype=np.float32)
    for h in holes:
        out = out * (1.0 - 0.95*np.clip(h, 0, 1))
    return np.clip(out, 0, 1)


def blurish(a):
    # A tiny fixed smoothing, not learned: average with neighbours.
    b = a.copy()
    b += np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)
    return np.clip(b/5.0, 0, 1)


def translate(a, dx=0, dy=0):
    out = np.zeros_like(a)
    ys0 = max(0, -dy); ys1 = min(N, N-dy)
    xs0 = max(0, -dx); xs1 = min(N, N-dx)
    yd0 = max(0, dy); yd1 = yd0 + (ys1-ys0)
    xd0 = max(0, dx); xd1 = xd0 + (xs1-xs0)
    if ys1 > ys0 and xs1 > xs0:
        out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
    return out


def norm_template(t):
    t = blurish(np.clip(t.astype(np.float32), 0, 1))
    t = t - t.mean()
    s = float(np.sqrt((t*t).mean()) + 1e-6)
    return (t / (s * (N*N))).astype(np.float32)


def tshirt(short=0.35, body_w=0.62, hem_w=0.54, length=0.72, tank=0.0, neck=0.18):
    y0, y1 = -0.66, length
    m = trap(y0, y1, body_w, hem_w)
    # shoulder cap and short sleeves
    m = add(m, rect(0, -0.47, body_w+0.12, 0.18))
    if tank < 0.5:
        m = add(m, trap(-0.55, -0.12+0.15*short, 0.25, 0.16, -body_w/2-0.10, -body_w/2-short))
        m = add(m, trap(-0.55, -0.12+0.15*short, 0.25, 0.16, body_w/2+0.10, body_w/2+short))
    else:
        m = cut(m, rect(-body_w/2+0.08, -0.36, 0.16, 0.42), rect(body_w/2-0.08, -0.36, 0.16, 0.42))
    m = cut(m, ellipse(0, -0.64, neck, 0.16), line(-0.11, -0.60, 0, -0.42, 0.035), line(0.11, -0.60, 0, -0.42, 0.035))
    return m


def pullover(w=0.68, length=0.74, sleeve=0.68, hood=0.0):
    m = trap(-0.62, length, w, w*0.86)
    m = add(m, rect(0, -0.45, w+0.08, 0.18))
    m = add(m, trap(-0.54, length-0.04, 0.22, 0.18, -w/2-0.08, -w/2-sleeve*0.28))
    m = add(m, trap(-0.54, length-0.04, 0.22, 0.18, w/2+0.08, w/2+sleeve*0.28))
    if hood > 0.5:
        m = add(m, ering(0, -0.60, 0.24, 0.25, 0.20, top_only=False))
    m = cut(m, ellipse(0, -0.63, 0.16, 0.12))
    # cuffs/bottom ribbing as bright horizontal bands
    m = add(m, line(-w*0.33, length-0.03, w*0.33, length-0.03, 0.035))
    return m


def shirt(w=0.64, length=0.74, sleeve=0.56, short=0.0):
    m = trap(-0.64, length, w, w*0.82)
    m = add(m, rect(0, -0.47, w+0.08, 0.16))
    yend = 0.18 if short > 0.5 else length-0.02
    m = add(m, trap(-0.54, yend, 0.20, 0.16, -w/2-0.07, -w/2-sleeve*0.30))
    m = add(m, trap(-0.54, yend, 0.20, 0.16, w/2+0.07, w/2+sleeve*0.30))
    # open collar and placket: dark vertical centre and V neck
    m = cut(m, line(0, -0.58, 0, length-0.10, 0.030), line(-0.16, -0.63, 0, -0.43, 0.045), line(0.16, -0.63, 0, -0.43, 0.045))
    # small collar wings remain bright
    m = add(m, trap(-0.62, -0.44, 0.16, 0.08, -0.13, -0.04), trap(-0.62, -0.44, 0.16, 0.08, 0.13, 0.04))
    return m


def coat(w=0.72, length=0.90, sleeve=0.72, flare=0.08, hood=0.0):
    m = trap(-0.66, length, w, w+flare)
    m = add(m, rect(0, -0.46, w+0.12, 0.20))
    m = add(m, trap(-0.55, length-0.03, 0.24, 0.18, -w/2-0.08, -w/2-sleeve*0.28))
    m = add(m, trap(-0.55, length-0.03, 0.24, 0.18, w/2+0.08, w/2+sleeve*0.28))
    # front opening and lapels
    m = cut(m, line(0, -0.58, 0, length, 0.035), line(-0.20, -0.60, 0, -0.32, 0.050), line(0.20, -0.60, 0, -0.32, 0.050))
    if hood > 0.5:
        m = add(m, ering(0, -0.62, 0.28, 0.28, 0.22, top_only=False))
        m = cut(m, ellipse(0, -0.61, 0.16, 0.18))
    return m


def dress(length=0.94, skirt=0.70, waist=0.34, topw=0.56, sleeve=0.10, straight=0.0):
    m = trap(-0.66, -0.08, topw, waist)
    if straight > 0.5:
        m = add(m, trap(-0.08, length, waist+0.10, skirt*0.72))
    else:
        m = add(m, trap(-0.10, length, waist, skirt))
    m = add(m, rect(0, -0.47, topw+0.06, 0.15))
    if sleeve > 0.04:
        m = add(m, trap(-0.55, -0.18, 0.17, 0.13, -topw/2-0.06, -topw/2-sleeve))
        m = add(m, trap(-0.55, -0.18, 0.17, 0.13, topw/2+0.06, topw/2+sleeve))
    m = cut(m, ellipse(0, -0.64, 0.15, 0.13))
    return m


def trouser(width=0.42, legw=0.22, flare=0.02, rise=-0.62, length=0.96, gap=0.13):
    m = rect(0, rise, width+0.10, 0.14)
    # two legs, usually a little tapered or flared
    m = add(m, trap(rise+0.02, length, legw, legw+flare, -gap-legw/2, -gap-legw/2-0.02))
    m = add(m, trap(rise+0.02, length, legw, legw+flare, gap+legw/2, gap+legw/2+0.02))
    m = cut(m, trap(-0.36, length, 0.12, 0.20))
    return m


def sneaker(height=0.32, toe=0.72, heel=0.42, sole=0.08, mirror=0.0):
    # side-view low filled shoe
    toe_x = toe
    heel_x = -0.72
    m = trap(0.34, 0.74, 0.35, 1.35, cx0=-0.28, cx1=0.02)  # crude upper/sole body
    m = cut(m, rect(-0.78, 0.30, 0.20, 0.30))
    m = add(m, ellipse(0.48, 0.56, 0.32, 0.18), rect(-0.15, 0.63, 0.95, 0.18))
    # low top line and sole line
    m = add(m, line(-0.66, 0.55-height*0.65, -0.10, 0.38-height*0.35, 0.045), line(-0.68, 0.72, 0.70, 0.72, sole))
    for off in [-0.16, -0.03, 0.10]:
        m = add(m, line(off, 0.40, off+0.20, 0.55, 0.020))
    if mirror > 0.5:
        m = m[:, ::-1]
    return m


def boot(shaft=0.72, heel=0.16, mirror=0.0, slim=0.0):
    foot = add(rect(0.05, 0.62, 1.15, 0.26), ellipse(0.48, 0.57, 0.30, 0.18))
    sw = 0.38 if slim < 0.5 else 0.28
    shaftm = trap(-0.60, 0.58, sw, sw*0.88, -0.25, -0.24)
    m = add(foot, shaftm, line(-0.60, 0.74, 0.58, 0.74, 0.06))
    if heel > 0.05:
        m = add(m, rect(-0.55, 0.82, 0.16, heel))
    m = cut(m, ellipse(-0.24, -0.62, sw*0.36, 0.10))
    if shaft < 0.55:
        m = m * (1 - 0.6*sig((-0.30-Y)/0.05))
    if mirror > 0.5:
        m = m[:, ::-1]
    return m


def sandal(heel=0.0, high=0.0, mirror=0.0):
    m = np.zeros_like(X)
    # sole and footbed
    m = add(m, line(-0.78, 0.72, 0.72, 0.70, 0.045), line(-0.70, 0.62, 0.62, 0.66, 0.030))
    # open straps
    m = add(m, line(-0.36, 0.60, -0.08, 0.18, 0.035), line(-0.10, 0.62, 0.20, 0.25, 0.035), line(0.18, 0.64, 0.46, 0.35, 0.030))
    m = add(m, ering(-0.18, 0.30, 0.34, 0.22, 0.10, top_only=True))
    if heel > 0.4:
        m = add(m, line(-0.62, 0.70, -0.74, 0.98, 0.045), line(-0.55, 0.66, -0.34, 0.18, 0.030))
    if high > 0.5:
        m = add(m, line(-0.18, 0.20, -0.12, -0.35, 0.030), ering(-0.12, -0.36, 0.18, 0.10, 0.10))
    if mirror > 0.5:
        m = m[:, ::-1]
    return np.clip(m, 0, 1)


def bag(kind=0, wide=0.70, tall=0.55, handle=1.0):
    if kind == 0:  # tote/handbag
        m = rect(0, 0.28, wide, tall)
        m = cut(m, ellipse(0, 0.02, wide*0.38, 0.22))
        if handle > 0.2:
            m = add(m, ering(0, -0.18, wide*0.36, 0.36, 0.13, top_only=True))
            m = cut(m, ellipse(0, -0.15, wide*0.24, 0.26))
        m = add(m, line(-wide*0.46, 0.02, wide*0.46, 0.02, 0.035))
    elif kind == 1:  # rectangular clutch/briefcase
        m = rect(0, 0.22, wide*1.10, tall*0.58)
        m = add(m, line(-wide*0.48, 0.06, wide*0.48, 0.06, 0.035))
        if handle > 0.5:
            m = add(m, ering(0, -0.12, wide*0.22, 0.20, 0.12, top_only=True))
    else:  # shoulder bag/rounded purse
        m = ellipse(0, 0.30, wide*0.55, tall*0.55)
        m = cut(m, ellipse(0, 0.00, wide*0.24, 0.18))
        m = add(m, ering(0, -0.12, wide*0.42, 0.38, 0.10, top_only=True))
    return np.clip(m, 0, 1)


def make_templates():
    per = 160
    cls = {i: [] for i in range(10)}
    # 0 tops: short sleeves, tanks, crop/long tops
    for short in [0.22,0.32,0.42,0.52]:
        for bw in [0.56,0.64,0.72]:
            for length in [0.58,0.70,0.80]:
                cls[0].append(tshirt(short,bw,bw*0.86,length,0.0,0.16))
    for bw in [0.45,0.55,0.65]:
        for length in [0.65,0.80]:
            cls[0].append(tshirt(0.15,bw,bw*0.82,length,1.0,0.18))
    # 1 trousers
    for width in [0.34,0.42,0.50]:
        for legw in [0.16,0.20,0.25]:
            for flare in [-0.04,0.00,0.05,0.10]:
                for gap in [0.08,0.13]:
                    cls[1].append(trouser(width,legw,flare,-0.62,0.98,gap))
    # 2 pullovers
    for w in [0.58,0.68,0.78]:
        for length in [0.66,0.76,0.84]:
            for sleeve in [0.52,0.68,0.82]:
                for hood in [0.0,1.0]:
                    cls[2].append(pullover(w,length,sleeve,hood))
    # 3 dresses
    for length in [0.80,0.92,0.98]:
        for skirt in [0.48,0.62,0.78,0.92]:
            for waist in [0.24,0.34,0.44]:
                for sleeve in [0.00,0.14,0.28]:
                    cls[3].append(dress(length,skirt,waist,0.54,sleeve,0.0))
    for length in [0.86,0.98]:
        for skirt in [0.44,0.58]:
            cls[3].append(dress(length,skirt,0.38,0.50,0.04,1.0))
    # 4 coats
    for w in [0.62,0.72,0.82]:
        for length in [0.78,0.90,0.98]:
            for sleeve in [0.58,0.74,0.90]:
                for flare in [-0.02,0.08,0.18]:
                    cls[4].append(coat(w,length,sleeve,flare,0.0))
    for w in [0.66,0.78]:
        for length in [0.84,0.96]:
            cls[4].append(coat(w,length,0.74,0.08,1.0))
    # 5 sandals
    for heel in [0.0,0.5,1.0]:
        for high in [0.0,1.0]:
            for mir in [0.0,1.0]:
                for dummy in range(8):
                    cls[5].append(sandal(heel,high,mir))
    # 6 shirts
    for w in [0.54,0.64,0.74]:
        for length in [0.62,0.74,0.84]:
            for sleeve in [0.42,0.58,0.74]:
                for short in [0.0,1.0]:
                    cls[6].append(shirt(w,length,sleeve,short))
    # 7 sneakers
    for h in [0.24,0.32,0.42]:
        for sole in [0.05,0.08,0.11]:
            for mir in [0.0,1.0]:
                for dummy in range(6):
                    cls[7].append(sneaker(h,0.72,0.42,sole,mir))
    # 8 bags
    for kind in [0,1,2]:
        for wide in [0.56,0.72,0.88,1.04]:
            for tall in [0.42,0.56,0.70]:
                for handle in [0.0,1.0]:
                    cls[8].append(bag(kind,wide,tall,handle))
    # 9 ankle boots
    for shaft in [0.46,0.62,0.78,0.94]:
        for heel in [0.08,0.16,0.26]:
            for mir in [0.0,1.0]:
                for slim in [0.0,1.0]:
                    cls[9].append(boot(shaft,heel,mir,slim))
    # Force equal counts by deterministic cycling through the hand-drawn variants.
    # Include small hand-specified translations because the item centering is not exact.
    rows=[]; labs=[]; canon=[]
    shifts=[(0,0),(-2,0),(2,0),(0,-1),(0,1),(-1,0),(1,0),(0,-2),(0,2),(0,0),(0,0),(0,0)]
    for k in range(10):
        variants = cls[k]
        if len(variants) == 0:
            raise RuntimeError(k)
        canon.append(norm_template(variants[0]))
        for i in range(per):
            base = variants[i % len(variants)]
            dx,dy = shifts[(i // len(variants)) % len(shifts)]
            rows.append(norm_template(translate(base, dx, dy)))
            labs.append(k)
    return np.stack(rows,0).reshape(10*per, -1), np.array(labs, dtype=np.int64), np.stack(canon,0).reshape(10,-1)


def mask_rect(x0,x1,y0,y1):
    m = ((X>=x0)&(X<=x1)&(Y>=y0)&(Y<=y1)).astype(np.float32)
    s = m.sum()
    return m/(s if s>0 else 1.0)


def mask_ellipse(cx,cy,rx,ry):
    m = (((X-cx)/rx)**2 + ((Y-cy)/ry)**2 <= 1).astype(np.float32)
    s=m.sum(); return m/(s if s>0 else 1.0)


def signed(mpos, mneg, sp=1.0, sn=1.0):
    return sp*mpos - sn*mneg


def make_features():
    feats=[]; biases=[]; impacts=[]
    def addf(mask,bias,imp):
        feats.append(mask.astype(np.float32).reshape(-1)); biases.append(float(bias)); impacts.append(np.array(imp,dtype=np.float32))
    allm = mask_rect(-1,1,-1,1)
    top = mask_rect(-1,1,-1,-0.32); high = mask_rect(-1,1,-1,0.10)
    mid = mask_rect(-1,1,-0.25,0.35); low = mask_rect(-1,1,0.18,1)
    foot = mask_rect(-0.95,0.95,0.42,0.95); upper = mask_rect(-0.95,0.95,-0.85,0.05)
    sole = mask_rect(-0.95,0.95,0.60,0.93); shoe_top = mask_rect(-0.80,0.80,0.18,0.52)
    center_low = mask_rect(-0.16,0.16,-0.10,0.95)
    sides_low = 0.5*(mask_rect(-0.55,-0.18,-0.10,0.95)+mask_rect(0.18,0.55,-0.10,0.95))
    skirt_low = mask_rect(-0.42,0.42,0.22,0.95)
    waist = mask_rect(-0.30,0.30,-0.22,0.18)
    shoulder_side = 0.5*(mask_rect(-0.95,-0.42,-0.68,0.02)+mask_rect(0.42,0.95,-0.68,0.02))
    sleeve_low = 0.5*(mask_rect(-0.92,-0.42,0.02,0.78)+mask_rect(0.42,0.92,0.02,0.78))
    handle_side = 0.5*(mask_rect(-0.60,-0.18,-0.66,-0.18)+mask_rect(0.18,0.60,-0.66,-0.18))
    handle_hole = mask_rect(-0.16,0.16,-0.58,-0.08)
    bag_body = mask_rect(-0.72,0.72,-0.10,0.66)
    bag_bottom = mask_rect(-0.70,0.70,0.25,0.72)
    boot_shaft_l = mask_rect(-0.55,0.05,-0.62,0.25)
    boot_shaft_r = mask_rect(-0.05,0.55,-0.62,0.25)
    toe_l = mask_rect(-0.85,-0.10,0.38,0.82); toe_r = mask_rect(0.10,0.85,0.38,0.82)
    # Average-brightness thresholds and contrasts. Biases are hand-chosen semantic cut points.
    addf(-allm, 0.20, [-0.2,-0.2,-0.4,-0.2,-0.4,3.2,-0.3,0.5,-0.1,-0.5])       # sparse line drawing: sandals
    addf(allm, -0.24, [0.1,0.1,0.5,0.1,0.7,-1.5,0.2,-0.7,0.3,0.3])              # filled fabric/leather
    addf(signed(low, high), -0.03, [-1.2,-0.5,-1.2,-0.6,-0.8,2.0,-1.1,2.5,-0.1,2.0])
    addf(signed(high, low), -0.02, [0.9,1.2,0.9,0.7,0.8,-1.4,0.9,-1.3,-0.3,-0.7])
    addf(-top, 0.055, [-0.4,-0.2,-0.6,-0.4,-0.6,0.8,-0.5,1.2,0.0,0.4])          # empty top margin: low footwear
    addf(signed(sole, upper), -0.035, [-1.0,-0.5,-1.2,-0.8,-0.9,1.6,-1.0,2.5,-0.5,1.4])
    addf(signed(foot, top), -0.035, [-1.0,-0.4,-1.1,-0.5,-0.8,1.8,-1.0,2.0,0.1,1.8])
    addf(signed(sides_low, center_low), -0.025, [-0.3,2.5,-0.3,-1.0,-0.5,0.0,-0.2,-0.3,-0.8,-0.3]) # trouser leg gap
    addf(signed(center_low, sides_low), -0.015, [0.0,-1.2,0.3,1.8,0.7,-0.4,0.0,-0.2,0.2,0.2])      # skirt/coat solid lower middle
    addf(signed(skirt_low, waist), -0.015, [-0.1,-1.1,-0.1,2.3,0.5,-0.4,-0.1,-0.3,0.1,0.1])
    addf(signed(sleeve_low, shoulder_side), -0.015, [-0.6,0.0,1.6,-0.2,1.5,-0.4,1.2,-0.3,-0.4,0.1])
    addf(signed(shoulder_side, sleeve_low), -0.015, [1.4,0.0,-0.7,0.4,-0.5,0.0,0.3,0.0,0.0,0.0])
    addf(signed(handle_side, handle_hole), -0.005, [-0.4,-0.5,-0.4,-0.3,-0.3,-0.5,-0.4,-0.4,3.0,-0.3])
    addf(bag_body, -0.20, [-0.3,-0.6,-0.2,0.0,0.0,-0.8,-0.2,-0.8,0.45,-0.3])
    addf(signed(bag_body, sole), -0.02, [-0.2,-0.6,0.0,0.2,0.2,-0.6,-0.1,-0.7,0.35,-0.4])
    addf(signed(boot_shaft_l, toe_r), -0.01, [-0.4,0.0,0.1,0.0,0.3,-0.4,0.0,-0.8,-0.3,2.2])
    addf(signed(boot_shaft_r, toe_l), -0.01, [-0.4,0.0,0.1,0.0,0.3,-0.4,0.0,-0.8,-0.3,2.2])
    addf(signed(toe_r, boot_shaft_l), -0.02, [-0.2,-0.2,-0.4,-0.3,-0.3,0.2,-0.3,1.4,-0.2,-0.7])
    addf(signed(toe_l, boot_shaft_r), -0.02, [-0.2,-0.2,-0.4,-0.3,-0.3,0.2,-0.3,1.4,-0.2,-0.7])
    addf(shoe_top, -0.18, [-0.3,-0.2,-0.4,-0.2,-0.2,-0.1,-0.3,1.2,-0.2,1.1])
    addf(mid, -0.23, [0.1,-0.1,0.5,0.2,0.5,-0.7,0.2,-0.6,0.5,0.1])
    addf(bag_bottom, -0.25, [-0.2,-0.4,-0.1,0.2,0.1,-0.5,-0.1,-0.5,1.2,-0.1])
    return np.stack(feats,0), np.array(biases,dtype=np.float32), np.stack(impacts,1)


def main(outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    T, labs, canon = make_templates()
    H = T.shape[0]
    # Template matcher: correlation against each individual drawn template, then ReLU.
    match_W = T.astype(np.float32)
    match_b = np.full((H,), -0.035, dtype=np.float32)
    per = H//10

    # Direct canonical correlations: one hand-drawn exemplar for each class, not an average.
    lin_W = (0.15 * canon).astype(np.float32)
    lin_b = np.zeros(10, dtype=np.float32)

    feat_W, feat_b, feat_imp = make_features()
    fs_W = (0.22 * feat_imp).astype(np.float32)
    fs_b = np.array([0.00, 0.02, 0.00, 0.00, -0.02, 0.02, -0.02, 0.03, 0.03, 0.03], dtype=np.float32)

    # Max over each class's hand-written template bank, followed by a small
    # fixed one-vs-rest competition.
    ms_W = np.full((10,10), -0.025, dtype=np.float32)
    for k in range(10):
        ms_W[k, k] = 1.0
    ms_b = np.array([0.00,0.02,0.00,0.00,-0.02,0.02,-0.02,0.03,0.01,0.02], dtype=np.float32)

    layers = [
        {"name":"n", "op":"norm", "inputs":["image"]},
        {"name":"match", "op":"dense", "inputs":["n"], "out": int(H)},
        {"name":"mr", "op":"relu", "inputs":["match"]},
        {"name":"matchmap", "op":"reshape", "inputs":["mr"], "shape":[10, int(per), 1]},
        {"name":"classmax", "op":"maxpool", "inputs":["matchmap"], "size":[int(per),1], "stride":[int(per),1]},
        {"name":"classvec", "op":"reshape", "inputs":["classmax"], "shape":[10]},
        {"name":"template_scores", "op":"dense", "inputs":["classvec"], "out": 10},
        {"name":"raw_features", "op":"dense", "inputs":["image"], "out": int(feat_W.shape[0])},
        {"name":"fr", "op":"relu", "inputs":["raw_features"]},
        {"name":"feature_scores", "op":"dense", "inputs":["fr"], "out": 10},
        {"name":"linear_scores", "op":"dense", "inputs":["n"], "out": 10},
        {"name":"scores", "op":"add", "inputs":["template_scores","feature_scores","linear_scores"]}
    ]
    net = {"layers": layers}
    (out/"network.json").write_text(json.dumps(net, separators=(",", ":")))
    np.savez(out/"weights.npz",
             **{"match.W":match_W, "match.b":match_b,
                "template_scores.W":ms_W, "template_scores.b":ms_b,
                "raw_features.W":feat_W.astype(np.float32), "raw_features.b":feat_b,
                "feature_scores.W":fs_W, "feature_scores.b":fs_b,
                "linear_scores.W":lin_W, "linear_scores.b":lin_b})
    method = (
        "Hand-written classifier using data-free geometric templates and semantic region rules. "
        "The input is channel-normalised, correlated with individual formula-drawn clothing, shoe and bag silhouettes, "
        "rectified, and averaged by class; additional ReLU features encode broad cues such as low shoes, trouser leg gaps, "
        "bag handles, long sleeves and skirt-like solid lower regions. All weights are generated from analytic shapes and "
        "hand-chosen constants in build.py; no training images or labels are read when producing the submitted weights.\n"
    )
    (out/"method.md").write_text(method)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "/app/out")
