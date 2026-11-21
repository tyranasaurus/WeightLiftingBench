#!/usr/bin/env python3
"""Build a data-free, hand-designed geometric template network for Fashion-MNIST."""
import json, os, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/app/out"
os.makedirs(OUT, exist_ok=True)
V = 256
S = 4
N = 14
rng = np.random.default_rng(24051984)

# A supersampled normalized drawing surface.  All shapes below are idealized
# clothing silhouettes, not measurements or traces from dataset images.
yy, xx = np.mgrid[0:N*S, 0:N*S].astype(np.float64)
X = (xx + .5) / (N*S) * 2.0 - 1.0
Y = (yy + .5) / (N*S)

def ellipse(x, y, cx, cy, rx, ry):
    return (((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1).astype(float)

def capsule(x, y, ax, ay, bx, by, r):
    vx, vy = bx-ax, by-ay
    q = np.clip(((x-ax)*vx + (y-ay)*vy)/(vx*vx+vy*vy+1e-12), 0, 1)
    return (((x-(ax+q*vx))**2 + (y-(ay+q*vy))**2) <= r*r).astype(float)

def band(x, y, ax, ay, bx, by, r):
    return capsule(x, y, ax, ay, bx, by, r)

def rect(x, y, x0, x1, y0, y1):
    return ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)).astype(float)

def down(a):
    return a.reshape(N, S, N, S).mean((1,3))

def coords():
    # Mild size and position diversity models centering/crop variation.
    sx = rng.uniform(.90, 1.10)
    sy = rng.uniform(.93, 1.06)
    dx = rng.uniform(-.055, .055)
    dy = rng.uniform(-.025, .025)
    return (X-dx)/sx, np.clip((Y-dy-.5)/sy+.5, -.2, 1.2)

def neck_cut(t, x, y, cy=.115, rx=.105, ry=.075):
    return t * (1.0 - ellipse(x,y,0,cy,rx,ry))

def short_sleeves(x,y, shoulder, top, amount=1.0):
    r = rng.uniform(.105,.15)
    drop = rng.uniform(.07,.16)
    reach = rng.uniform(.43,.61)
    a = capsule(x,y, shoulder,top+.06, reach,top+drop,r)
    b = capsule(x,y,-shoulder,top+.06,-reach,top+drop,r)
    return np.maximum(a,b)*amount

def long_sleeves(x,y, shoulder=.29, top=.16, cuff=.42, bottom=.86, thick=.135):
    bend = rng.uniform(-.04,.04)
    a = capsule(x,y, shoulder,top, cuff+bend,bottom,thick)
    b = capsule(x,y,-shoulder,top,-cuff+bend,bottom,thick)
    return np.maximum(a,b)

def torso(x,y, top, bottom, wt, wb, waist=None):
    q = np.clip((y-top)/(bottom-top),0,1)
    if waist is None:
        w = wt*(1-q)+wb*q
    else:
        # Narrow at the middle and broaden toward both ends.
        w = np.where(q < .55, wt+(waist-wt)*(q/.55), waist+(wb-waist)*((q-.55)/.45))
    return ((y>=top)&(y<=bottom)&(np.abs(x)<=w)).astype(float)

def detail_dark(t, line, strength):
    return t * (1.0-strength*line)

def make_template(c):
    x,y = coords()
    t = np.zeros_like(x)
    if c == 0:                         # T-shirt / top
        top = rng.uniform(.105,.17); bot=rng.uniform(.79,.94)
        wt=rng.uniform(.29,.37); wb=rng.uniform(.25,.34)
        t = torso(x,y,top,bot,wt,wb,rng.uniform(.23,.31))
        if rng.random() > .10:
            t=np.maximum(t,short_sleeves(x,y,wt,top))
        t=neck_cut(t,x,y,top+rng.uniform(-.005,.025),rng.uniform(.08,.14),rng.uniform(.055,.09))
        if rng.random()<.18:
            ln=band(x,y,-wb,bot-.02,wb,bot-.02,.012)
            t=detail_dark(t,ln,.45)
    elif c == 1:                       # trousers
        top=rng.uniform(.045,.11); crotch=rng.uniform(.38,.54); bot=rng.uniform(.94,1.03)
        hip=rng.uniform(.27,.36); ankle=rng.uniform(.12,.21)
        q=np.clip((y-top)/(bot-top),0,1)
        outer=hip*(1-q)+ankle*q
        gap=np.clip((y-crotch)/(bot-crotch),0,1)*rng.uniform(.045,.105)
        t=((y>=top)&(y<=bot)&(np.abs(x)<=outer)&((y<crotch)|(np.abs(x)>=gap))).astype(float)
        t=np.maximum(t,rect(x,y,-hip,hip,top,top+rng.uniform(.05,.10)))
        if rng.random()<.5:
            t=detail_dark(t,band(x,y,-hip*.9,top+.07,hip*.9,top+.07,.008),.3)
    elif c == 2:                       # pullover
        top=rng.uniform(.09,.15); bot=rng.uniform(.82,.94)
        wt=rng.uniform(.28,.36); wb=rng.uniform(.27,.36)
        t=torso(x,y,top,bot,wt,wb,rng.uniform(.25,.32))
        t=np.maximum(t,long_sleeves(x,y,wt*.92,top+.06,rng.uniform(.35,.48),rng.uniform(.82,.94),rng.uniform(.11,.16)))
        t=neck_cut(t,x,y,top+.005,rng.uniform(.085,.145),rng.uniform(.045,.085))
        if rng.random()<.28:
            for fy in (.35,.55,.74):
                if rng.random()<.55:
                    t=detail_dark(t,band(x,y,-wb*.9,fy,wb*.9,fy,.010),rng.uniform(.15,.35))
    elif c == 3:                       # dress
        top=rng.uniform(.08,.16); waist_y=rng.uniform(.34,.49); bot=rng.uniform(.91,1.02)
        ws=rng.uniform(.23,.34); ww=rng.uniform(.14,.23); hem=rng.uniform(.32,.52)
        q=np.clip((y-top)/(waist_y-top),0,1); upperw=ws*(1-q)+ww*q
        upper=((y>=top)&(y<=waist_y)&(np.abs(x)<=upperw)).astype(float)
        q2=np.clip((y-waist_y)/(bot-waist_y),0,1); skirtw=ww*(1-q2)+hem*q2
        skirt=((y>=waist_y)&(y<=bot)&(np.abs(x)<=skirtw)).astype(float)
        t=np.maximum(upper,skirt)
        r=rng.random()
        if r<.48: t=np.maximum(t,short_sleeves(x,y,ws,top))
        elif r<.62: t=np.maximum(t,long_sleeves(x,y,ws*.9,top+.05,rng.uniform(.28,.39),rng.uniform(.60,.88),rng.uniform(.08,.12)))
        t=neck_cut(t,x,y,top+.005,rng.uniform(.07,.13),rng.uniform(.045,.085))
        if rng.random()<.45:
            t=detail_dark(t,band(x,y,-ww,waist_y,ww,waist_y,.012),.35)
    elif c == 4:                       # coat
        top=rng.uniform(.075,.14); bot=rng.uniform(.90,1.02)
        wt=rng.uniform(.29,.37); wb=rng.uniform(.30,.43)
        t=torso(x,y,top,bot,wt,wb,rng.uniform(.27,.34))
        t=np.maximum(t,long_sleeves(x,y,wt*.92,top+.06,rng.uniform(.36,.48),rng.uniform(.84,.98),rng.uniform(.115,.165)))
        t=neck_cut(t,x,y,top,rng.uniform(.07,.115),rng.uniform(.04,.07))
        # Front opening, lapels, and buttons are generic coat construction cues.
        t=detail_dark(t,band(x,y,0,top+.12,rng.uniform(-.025,.025),bot,.010),rng.uniform(.35,.75))
        if rng.random()<.75:
            t=detail_dark(t,np.maximum(band(x,y,-.02,top+.11,-.16,top+.31,.018),band(x,y,.02,top+.11,.16,top+.31,.018)),rng.uniform(.25,.6))
        for by in (.39,.55,.71):
            if rng.random()<.55: t=detail_dark(t,ellipse(x,y,rng.choice([-.045,.045]),by,.014,.014),.8)
    elif c == 5:                       # sandal
        flip=-1 if rng.random()<.5 else 1
        u=x*flip
        sole=rect(u,y,-.82,.82,rng.uniform(.82,.88),rng.uniform(.89,.95))
        sole=np.maximum(sole,ellipse(u,y,-.72,.865,.15,.06)); sole=np.maximum(sole,ellipse(u,y,.70,.865,.18,.065))
        t=sole
        kind=rng.random()
        if kind<.72:
            t=np.maximum(t,band(u,y,-.45,.84,-.15,rng.uniform(.53,.70),rng.uniform(.025,.055)))
            t=np.maximum(t,band(u,y,.42,.84,.10,rng.uniform(.57,.72),rng.uniform(.025,.055)))
            if rng.random()<.75: t=np.maximum(t,band(u,y,-.20,.60,.30,.64,rng.uniform(.025,.05)))
            if rng.random()<.35: t=np.maximum(t,band(u,y,-.15,.61,-.28,.43,.025))
        else: # generic high-heeled sandal
            t=np.maximum(t,band(u,y,-.48,.84,-.20,.40,.045))
            t=np.maximum(t,band(u,y,-.20,.40,.18,.65,.035))
            t=np.maximum(t,band(u,y,-.52,.84,-.57,.98,.035))
    elif c == 6:                       # shirt
        top=rng.uniform(.09,.17); bot=rng.uniform(.78,.95)
        wt=rng.uniform(.30,.39); wb=rng.uniform(.25,.36)
        t=torso(x,y,top,bot,wt,wb,rng.uniform(.23,.31))
        r=rng.random()
        if r<.72: t=np.maximum(t,short_sleeves(x,y,wt,top))
        elif r<.91: t=np.maximum(t,long_sleeves(x,y,wt*.9,top+.06,rng.uniform(.34,.46),rng.uniform(.78,.93),rng.uniform(.10,.145)))
        t=neck_cut(t,x,y,top,rng.uniform(.075,.13),rng.uniform(.04,.075))
        # Collar/front placket distinguish a constructed shirt from a plain tee.
        if rng.random()<.82:
            collar=np.maximum(band(x,y,-.02,top+.04,-.13,top+.16,.018),band(x,y,.02,top+.04,.13,top+.16,.018))
            t=detail_dark(t,collar,rng.uniform(.3,.7))
        if rng.random()<.72: t=detail_dark(t,band(x,y,0,top+.10,0,bot,.008),rng.uniform(.25,.6))
    elif c == 7:                       # sneaker
        flip=-1 if rng.random()<.5 else 1; u=x*flip
        sole_y=rng.uniform(.86,.92)
        t=rect(u,y,-.84,.82,sole_y,sole_y+rng.uniform(.055,.085))
        # heel at left, toe at right in u coordinates
        q=np.clip((u+.68)/1.35,0,1)
        roof=(rng.uniform(.68,.76)*(1-q)+rng.uniform(.76,.82)*q)
        upper=((u>=-.72)&(u<=.78)&(y>=roof)&(y<=sole_y)).astype(float)
        upper=np.maximum(upper,ellipse(u,y,.68,sole_y-.08,.20,.10))
        t=np.maximum(t,upper)
        hole=ellipse(u,y,-.37,rng.uniform(.67,.73),rng.uniform(.16,.23),rng.uniform(.035,.065))
        t=t*(1-.50*hole)
        for d in (-.18,-.02,.14):
            if rng.random()<.8: t=detail_dark(t,band(u,y,d,.67,d+.25,.75,.009),.45)
    elif c == 8:                       # bag
        top=rng.uniform(.32,.48); bot=rng.uniform(.86,.98); half=rng.uniform(.43,.72)
        hb=half*rng.uniform(.82,1.08)
        q=np.clip((y-top)/(bot-top),0,1); w=half*(1-q)+hb*q
        t=((y>=top)&(y<=bot)&(np.abs(x)<=w)).astype(float)
        if rng.random()<.82:
            hx=rng.uniform(.18,.34); crown=rng.uniform(.05,.21); thick=rng.uniform(.025,.065)
            outer=ellipse(x,y,0,top,hx+thick,top-crown+thick)
            inner=ellipse(x,y,0,top,hx,top-crown)
            handle=np.maximum(outer-inner,0)*(y<=top+.025)
            t=np.maximum(t,handle)
        if rng.random()<.5: t=detail_dark(t,band(x,y,-half*.9,top+.06,half*.9,top+.06,.012),.4)
    else:                              # ankle boot
        flip=-1 if rng.random()<.5 else 1; u=x*flip
        sole_y=rng.uniform(.88,.93)
        t=rect(u,y,-.78,.83,sole_y,sole_y+rng.uniform(.055,.085))
        # Filled foot plus a tall shaft at heel side.
        q=np.clip((u+.55)/1.28,0,1); roof=rng.uniform(.55,.68)*(1-q)+rng.uniform(.73,.80)*q
        foot=((u>=-.65)&(u<=.78)&(y>=roof)&(y<=sole_y)).astype(float)
        shaft_top=rng.uniform(.12,.40); shaft_right=rng.uniform(-.10,.20)
        shaft=((u>=-.62)&(u<=shaft_right)&(y>=shaft_top)&(y<=sole_y-.05)).astype(float)
        t=np.maximum(t,np.maximum(foot,shaft))
        t=np.maximum(t,ellipse(u,y,.67,sole_y-.07,.20,.10))
        if rng.random()<.55: t=detail_dark(t,band(u,y,-.55,shaft_top+.12,shaft_right,shaft_top+.13,.012),.35)
        if rng.random()<.35: t=np.maximum(t,rect(u,y,-.62,-.52,sole_y+.04,.99))
    return down(np.clip(t,0,1))

# Fixed analytic filter bank.  The positive channel preserves absolute support;
# signed first/second derivatives add contour and construction-line evidence.
F = np.zeros((5,1,3,3),np.float32)
F[0,0,1,1]=1
F[1,0]=np.ones((3,3),float)/9
F[2,0]=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)/8
F[3,0]=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],float)/8
F[4,0]=np.array([[0,1,0],[1,-4,1],[0,1,0]],float)/4

def filt(a):
    p=np.pad(a,1)
    out=np.zeros((5,N,N),float)
    for dy in range(3):
        for dx in range(3):
            out += F[:,0,dy,dx,None,None]*p[dy:dy+N,dx:dx+N]
    return out

W=np.empty((10*V,5*N*N),np.float32)
b=np.zeros(10*V,np.float32)
beta=np.array([.65,.35,.18,.18,.06],float)
# Hand-set class offsets express only general ambiguity between similar apparel.
prior=np.array([.05,.04,.03,.05,-.03,.08,-.02,.09,.00,.04],np.float32)
for c in range(10):
    for v in range(V):
        a=make_template(c)
        ft=filt(a)
        den=np.sqrt(sum(beta[k]*(ft[k]**2).sum() for k in range(5)))+1e-8
        wt=np.stack([beta[k]*ft[k]/den for k in range(5)])
        W[c*V+v]=wt.reshape(-1).astype(np.float32)
        b[c*V+v]=prior[c]

layers=[
 {"name":"small","op":"avgpool","inputs":["image"],"size":[2,2],"stride":[2,2]},
 {"name":"shape_features","op":"conv","inputs":["small"],"out":5,"kernel":[3,3],"stride":1,"padding":1},
 {"name":"matches","op":"dense","inputs":["shape_features"],"out":10*V},
 {"name":"match_grid","op":"reshape","inputs":["matches"],"shape":[10,V,1]},
 {"name":"best_match","op":"maxpool","inputs":["match_grid"],"size":[V,1],"stride":[V,1]},
 {"name":"scores","op":"reshape","inputs":["best_match"],"shape":[10]}
]
with open(os.path.join(OUT,"network.json"),"w") as f: json.dump({"layers":layers},f,indent=2)
np.savez(os.path.join(OUT,"weights.npz"), **{
 "shape_features.W":F,"shape_features.b":np.zeros(5,np.float32),
 "matches.W":W,"matches.b":b})
