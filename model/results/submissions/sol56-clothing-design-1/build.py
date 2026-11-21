import json, sys
from pathlib import Path
import numpy as np

# This model is a bank of silhouettes drawn from geometric clothing primitives.
# All randomness below only varies those synthetic drawings and has a fixed seed.

def polygon(X, Y, pts):
    z = np.zeros(X.shape, bool)
    x0, y0 = pts[-1]
    for x1, y1 in pts:
        hit = ((y1 > Y) != (y0 > Y)) & (X < (x0-x1)*(Y-y1)/(y0-y1+1e-20)+x1)
        z ^= hit
        x0, y0 = x1, y1
    return z

def ellipse(X,Y,cx,cy,rx,ry):
    return ((X-cx)/rx)**2 + ((Y-cy)/ry)**2 <= 1

def stroke(X,Y,x1,y1,x2,y2,r):
    vx,vy=x2-x1,y2-y1
    t=np.clip(((X-x1)*vx+(Y-y1)*vy)/(vx*vx+vy*vy+1e-12),0,1)
    return (X-x1-t*vx)**2+(Y-y1-t*vy)**2 <= r*r

def render(kind, rng, S=4):
    q=(np.arange(28*S)+.5)/(14*S)-1
    X,Y=np.meshgrid(q,q)
    # Small generic centering and scale variations, not measurements from data.
    X=(X-rng.uniform(-.055,.055))/rng.uniform(.92,1.04)
    Y=(Y-rng.uniform(-.045,.045))/rng.uniform(.94,1.04)
    M=np.zeros(X.shape,bool)
    if kind==0: # T-shirt / top
        top=rng.uniform(-.84,-.68); bot=rng.uniform(.62,.88)
        wt=rng.uniform(.30,.43); wb=rng.uniform(.30,.47)
        style=rng.random()
        if style < .22: # sleeveless top
            M=polygon(X,Y,[(-wt,top),(wt,top),(wt*.92,top+.25),(wb,bot),(-wb,bot),(-wt*.92,top+.25)])
        else:
            ex=rng.uniform(.60,.82); sy=rng.uniform(-.28,.04)
            pts=[(-wt,top),(-ex,top+.14),(-ex*.92,sy),(-wt,sy-.03),(-wb,bot),(wb,bot),(wt,sy-.03),(ex*.92,sy),(ex,top+.14),(wt,top)]
            M=polygon(X,Y,pts)
        neck=ellipse(X,Y,0,top,rng.uniform(.10,.22),rng.uniform(.07,.15)) & (Y<top+.16)
        M &= ~neck
    elif kind==1: # trousers
        top=rng.uniform(-.91,-.78); bot=rng.uniform(.80,.94)
        hip=rng.uniform(.32,.47); waist=hip*rng.uniform(.82,1.02)
        crotch=rng.uniform(-.05,.25); inner=rng.uniform(.045,.13); cuff=rng.uniform(.10,.21)
        left=polygon(X,Y,[(-waist,top),(0,top),(inner,crotch),(-inner*.45,bot),(-inner*.45-cuff,bot),(-hip,crotch)])
        right=polygon(X,Y,[(0,top),(waist,top),(hip,crotch),(inner*.45+cuff,bot),(inner*.45,bot),(-inner,crotch)])
        M=left|right
    elif kind==2: # pullover
        top=rng.uniform(-.84,-.68); bot=rng.uniform(.60,.86)
        wt=rng.uniform(.31,.43); wb=rng.uniform(.31,.46)
        ex=rng.uniform(.57,.80); cuffy=rng.uniform(.42,.79); cuff=rng.uniform(.10,.18)
        torso=polygon(X,Y,[(-wt,top),(wt,top),(wb,bot),(-wb,bot)])
        left=polygon(X,Y,[(-wt,top),(-ex,top+.18),(-ex+cuff*.15,cuffy),(-ex+cuff,cuffy+.03),(-wt*.92,-.05)])
        right=polygon(X,Y,[(wt,top),(ex,top+.18),(ex-cuff*.15,cuffy),(ex-cuff,cuffy+.03),(wt*.92,-.05)])
        M=torso|left|right
        M &= ~ellipse(X,Y,0,top,rng.uniform(.10,.20),rng.uniform(.06,.13))
    elif kind==3: # dress
        top=rng.uniform(-.88,-.68); waisty=rng.uniform(-.18,.16); bot=rng.uniform(.72,.93)
        sh=rng.uniform(.21,.38); waist=rng.uniform(.18,.31); hem=rng.uniform(.38,.66)
        M=polygon(X,Y,[(-sh,top),(sh,top),(waist,waisty),(hem,bot),(-hem,bot),(-waist,waisty)])
        if rng.random()<.48:
            ex=rng.uniform(.42,.67); sleeve=rng.uniform(-.36,-.05)
            M |= polygon(X,Y,[(-sh,top),(-ex,top+.15),(-ex*.92,sleeve),(-sh,sleeve-.03)])
            M |= polygon(X,Y,[(sh,top),(ex,top+.15),(ex*.92,sleeve),(sh,sleeve-.03)])
        M &= ~ellipse(X,Y,0,top,rng.uniform(.08,.18),rng.uniform(.06,.13))
    elif kind==4: # coat
        top=rng.uniform(-.88,-.70); bot=rng.uniform(.68,.93)
        wt=rng.uniform(.32,.45); wb=rng.uniform(.34,.53); ex=rng.uniform(.58,.79)
        cuffy=rng.uniform(.52,.84); cuff=rng.uniform(.10,.17)
        torso=polygon(X,Y,[(-wt,top),(wt,top),(wb,bot),(-wb,bot)])
        M=torso
        M|=polygon(X,Y,[(-wt,top),(-ex,top+.17),(-ex+cuff*.1,cuffy),(-ex+cuff,cuffy+.03),(-wt*.90,-.02)])
        M|=polygon(X,Y,[(wt,top),(ex,top+.17),(ex-cuff*.1,cuffy),(ex-cuff,cuffy+.03),(wt*.90,-.02)])
        # Open front or narrow central seam is a common coat construction.
        if rng.random()<.72:
            gap=rng.uniform(.012,.035)
            M &= ~((np.abs(X)<gap)&(Y>rng.uniform(-.48,-.22)))
        if rng.random()<.55:
            lap=polygon(X,Y,[(-.02,top+.38),(-wt*.65,top),(-.08,top+.05),(0,top+.22),(.08,top+.05),(wt*.65,top),(.02,top+.38)])
            M &= ~lap
    elif kind==6: # shirt: short/long sleeves, collar and often an opening
        top=rng.uniform(-.86,-.67); bot=rng.uniform(.58,.88)
        wt=rng.uniform(.30,.44); wb=rng.uniform(.30,.47); long=rng.random()<.45
        torso=polygon(X,Y,[(-wt,top),(wt,top),(wb,bot),(-wb,bot)])
        ex=rng.uniform(.59,.82)
        if long:
            ey=rng.uniform(.42,.79); cuff=rng.uniform(.11,.18)
            sleeves=polygon(X,Y,[(-wt,top),(-ex,top+.17),(-ex+cuff*.1,ey),(-ex+cuff,ey+.03),(-wt*.9,-.02)])
            sleeves|=polygon(X,Y,[(wt,top),(ex,top+.17),(ex-cuff*.1,ey),(ex-cuff,ey+.03),(wt*.9,-.02)])
        else:
            ey=rng.uniform(-.34,.05)
            sleeves=polygon(X,Y,[(-wt,top),(-ex,top+.15),(-ex*.91,ey),(-wt,ey-.03)])
            sleeves|=polygon(X,Y,[(wt,top),(ex,top+.15),(ex*.91,ey),(wt,ey-.03)])
        M=torso|sleeves
        M &= ~ellipse(X,Y,0,top,rng.uniform(.08,.17),rng.uniform(.05,.12))
        if rng.random()<.65:
            M &= ~((np.abs(X)<rng.uniform(.010,.026))&(Y>top+.15))
    elif kind==5: # sandal: sole plus open straps, including heeled sandals
        flip=-1 if rng.random()<.5 else 1; X=flip*X
        base=rng.uniform(.55,.77); sole=rng.uniform(.035,.075)
        if rng.random()<.42: # high-heeled/open shoe
            M=stroke(X,Y,-.72,base,.78,base-rng.uniform(.02,.12),sole)
            M|=stroke(X,Y,-.58,base,-.48,rng.uniform(-.15,.12),rng.uniform(.035,.075))
            M|=stroke(X,Y,-.48,rng.uniform(-.15,.12),.18,rng.uniform(.15,.35),rng.uniform(.045,.09))
            M|=stroke(X,Y,.18,rng.uniform(.15,.35),.65,base-.08,rng.uniform(.035,.08))
            M|=stroke(X,Y,-.55,base,-.62,.82,rng.uniform(.025,.055))
        else:
            M=stroke(X,Y,-.82,base,.82,base-rng.uniform(.00,.08),sole)
            for _ in range(rng.integers(2,5)):
                x1=rng.uniform(-.58,.42); x2=x1+rng.uniform(.18,.52)
                M|=stroke(X,Y,x1,base-.03,x2,rng.uniform(.08,.40),rng.uniform(.035,.075))
            if rng.random()<.45:
                M|=stroke(X,Y,-.62,base-.02,-.62,rng.uniform(-.05,.28),rng.uniform(.03,.06))
    elif kind==7: # sneaker
        flip=-1 if rng.random()<.5 else 1; X=flip*X
        bot=rng.uniform(.55,.76); heel=rng.uniform(-.78,-.58); toe=rng.uniform(.70,.91)
        top=rng.uniform(-.08,.20); ankle=rng.uniform(-.48,-.20)
        pts=[(heel,bot),(heel,top+.20),(ankle,top),(rng.uniform(-.05,.20),rng.uniform(.16,.34)),(toe-.18,bot-.25),(toe,bot-.10),(toe,bot),(0,bot+.04)]
        M=polygon(X,Y,pts)
        M|=stroke(X,Y,heel,bot+.015,toe,bot-.02,rng.uniform(.035,.065))
        M &= ~stroke(X,Y,ankle+.04,top+.10,rng.uniform(.22,.48),bot-.22,rng.uniform(.025,.055))
        for yy in np.linspace(top+.19,bot-.27,3):
            M &= ~stroke(X,Y,ankle+.11,yy,rng.uniform(.18,.43),yy+.02,rng.uniform(.012,.026))
    elif kind==8: # bag
        top=rng.uniform(-.34,.05); bot=rng.uniform(.60,.88)
        wt=rng.uniform(.43,.70); wb=wt*rng.uniform(.82,1.10)
        M=polygon(X,Y,[(-wt,top),(wt,top),(wb,bot),(-wb,bot)])
        if rng.random()<.88:
            outerx=wt*rng.uniform(.38,.72); outery=rng.uniform(.35,.62); cy=top+rng.uniform(-.02,.10)
            thick=rng.uniform(.045,.095)
            outer=ellipse(X,Y,0,cy,outerx,outery)&(Y<top+.06)
            inner=ellipse(X,Y,0,cy,maximum(outerx-thick,.03),maximum(outery-thick,.03))&(Y<top+.06)
            M|=outer&~inner
        # Common product bags have a flap, zipper, or top opening.
        if rng.random()<.70:
            M &= ~stroke(X,Y,-wt*.72,top+rng.uniform(.08,.26),wt*.72,top+rng.uniform(.08,.26),rng.uniform(.012,.03))
    elif kind==9: # ankle boot
        flip=-1 if rng.random()<.5 else 1; X=flip*X
        bot=rng.uniform(.58,.79); heel=rng.uniform(-.73,-.52); toe=rng.uniform(.67,.91)
        shaft_top=rng.uniform(-.80,-.38); shaft_r=rng.uniform(.19,.34)
        foot=polygon(X,Y,[(heel,bot),(heel,shaft_top),(heel+2*shaft_r,shaft_top),(heel+2*shaft_r+rng.uniform(-.02,.13),.22),(toe-.12,bot-.23),(toe,bot-.08),(toe,bot)])
        M=foot|stroke(X,Y,heel,bot+.015,toe,bot-.01,rng.uniform(.035,.07))
        if rng.random()<.58:
            M &= ~stroke(X,Y,heel+shaft_r*1.05,shaft_top+.10,heel+shaft_r*1.05,.28,rng.uniform(.012,.028))
    # Area average gives anti-aliased 28x28 synthetic photography silhouettes.
    return M.reshape(28,S,28,S).mean((1,3)).astype(np.float32)

def maximum(a,b): return a if a>b else b

def correlate(m,k):
    p=np.pad(m,1)
    out=np.zeros_like(m)
    for y in range(3):
        for x in range(3): out += k[y,x]*p[y:y+28,x:x+28]
    return out

def features(m, raw=None):
    ks=[]
    z=np.zeros((3,3),np.float32); z[1,1]=1; ks.append(z)
    z=np.zeros((3,3),np.float32); z[1,0]=-.5; z[1,2]=.5; ks += [z,-z]
    z=np.zeros((3,3),np.float32); z[0,1]=-.5; z[2,1]=.5; ks += [z,-z]
    z=np.array([[0,-.25,0],[-.25,1,-.25],[0,-.25,0]],np.float32); ks += [z,-z]
    a=np.stack([np.maximum(correlate(m,k),0) for k in ks])
    a=a.reshape(7,7,4,7,4).mean((2,4))
    if raw is not None:
        # Match the graph's add of seven broadcast copies of the raw 7x7 image.
        a += 3*raw.reshape(7,4,7,4).mean((1,3))[None]
    return a

def build(dst):
    dst=Path(dst); dst.mkdir(parents=True,exist_ok=True)
    layers=[
      {"name":"raw_coarse","op":"avgpool","inputs":["image"],"size":[4,4],"stride":[4,4]},
      {"name":"clip_pre","op":"conv","inputs":["image"],"out":2,"kernel":[1,1],"stride":1,"padding":0},
      {"name":"clip_relu","op":"relu","inputs":["clip_pre"]},
      {"name":"occupancy","op":"conv","inputs":["clip_relu"],"out":1,"kernel":[1,1],"stride":1,"padding":0},
      {"name":"shape_features","op":"conv","inputs":["occupancy"],"out":7,"kernel":[3,3],"stride":1,"padding":1},
      {"name":"positive_features","op":"relu","inputs":["shape_features"]},
      {"name":"coarse_features","op":"avgpool","inputs":["positive_features"],"size":[4,4],"stride":[4,4]},
      {"name":"raw_project","op":"conv","inputs":["raw_coarse"],"out":7,"kernel":[1,1],"stride":1,"padding":0},
      {"name":"all_features","op":"add","inputs":["coarse_features","raw_project"]},
      {"name":"template_scores","op":"dense","inputs":["all_features"],"out":3840},
      {"name":"class_templates","op":"reshape","inputs":["template_scores"],"shape":[10,384,1]},
      {"name":"class_scores","op":"maxpool","inputs":["class_templates"],"size":[384,1],"stride":[384,1]},
      {"name":"scores","op":"reshape","inputs":["class_scores"],"shape":[10]}
    ]
    weights={}
    t=.18
    weights['clip_pre.W']=np.array([[[[1/t]]],[[[1/t]]]],np.float32)
    weights['clip_pre.b']=np.array([0,-1],np.float32)
    weights['occupancy.W']=np.array([[[[1]], [[-1]]]],np.float32)
    # Preserve midtone textile variation while clipping the bright end: this keeps
    # seams, openings, and texture that distinguish coats/shirts/bags.
    weights['occupancy.b']=np.array([-.18],np.float32)
    K=np.zeros((7,1,3,3),np.float32)
    K[0,0,1,1]=1
    K[1,0,1,0]=-.5; K[1,0,1,2]=.5; K[2]=-K[1]
    K[3,0,0,1]=-.5; K[3,0,2,1]=.5; K[4]=-K[3]
    K[5,0]=np.array([[0,-.25,0],[-.25,1,-.25],[0,-.25,0]])
    K[6]=-K[5]
    weights['shape_features.W']=K
    weights['shape_features.b']=np.zeros(7,np.float32)
    weights['raw_project.W']=np.full((7,1,1,1),3,np.float32)
    weights['raw_project.b']=np.zeros(7,np.float32)
    rng=np.random.default_rng(7281979)
    W=np.empty((3840,7*7*7),np.float32); b=np.empty(3840,np.float32)
    cw=np.array([1.6,.45,.45,.45,.45,.28,.28],np.float32)[:,None,None]
    # Priors express semantic specificity: generic open straps and coats match many
    # unrelated silhouettes, while trousers and sneakers are highly distinctive.
    prior=[.015,.045,.010,.020,-.045,-.050,-.010,.050,.005,.020]
    n=7*7
    for c in range(10):
        for j in range(384):
            R=render(c,rng)
            F=features(np.maximum(R-.18,0),R)
            row=c*384+j
            W[row]=(2*cw*F/n).reshape(-1)
            b[row]=-float((cw*F*F).sum()/n)+prior[c]
    weights['template_scores.W']=W
    weights['template_scores.b']=b
    (dst/'network.json').write_text(json.dumps({'layers':layers},separators=(',',':')))
    np.savez(dst/'weights.npz',**{k:v.astype(np.float32) for k,v in weights.items()})

if __name__=='__main__':
    build(sys.argv[1] if len(sys.argv)>1 else '.')
