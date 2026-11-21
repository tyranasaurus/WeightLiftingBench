"""Build a data-free Fashion-MNIST classifier from geometric clothing models.

Every template below is drawn from general clothing geometry.  No data files are
read.  The network compares fixed Gaussian/derivative image features with these
models using cosine similarity, followed by a within-class maximum.
"""
import sys, json
from pathlib import Path
import numpy as np

NCLASS = 10
NVAR = 192
GROUP = 24
SIZE = 28
rng = np.random.default_rng(731905)

# Supersampled drawing grid.  Coordinates in the drawing routines are pixels.
S = 2
yy, xx = np.mgrid[0:SIZE*S, 0:SIZE*S].astype(float) / S + 0.25

def blank():
    return np.zeros((SIZE*S, SIZE*S), float)

def paint(a, mask, value=0.78):
    a[mask] = value

def rect(a, x0, y0, x1, y1, value=0.78):
    paint(a, (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1), value)

def ellipse(a, cx, cy, rx, ry, value=0.78):
    paint(a, ((xx-cx)/rx)**2 + ((yy-cy)/ry)**2 <= 1, value)

def polygon(a, points, value=0.78):
    inside = np.zeros(a.shape, bool)
    p = list(points)
    for (x1,y1),(x2,y2) in zip(p, p[1:]+p[:1]):
        cross = ((y1 > yy) != (y2 > yy))
        xhit = (x2-x1)*(yy-y1)/(y2-y1+1e-12)+x1
        inside ^= cross & (xx < xhit)
    paint(a, inside, value)

def line(a, p, q, width=0.8, value=0.35):
    x1,y1=p; x2,y2=q
    dx=x2-x1; dy=y2-y1
    t=((xx-x1)*dx+(yy-y1)*dy)/(dx*dx+dy*dy+1e-12)
    t=np.clip(t,0,1)
    d2=(xx-x1-t*dx)**2+(yy-y1-t*dy)**2
    paint(a,d2 <= (width/2)**2,value)

def polyline(a, pts, width=0.8, value=0.35):
    for p,q in zip(pts[:-1],pts[1:]): line(a,p,q,width,value)

def mirror(a):
    return a[:, ::-1].copy()

def neck_cut(a, cx, top, width, depth, round_neck=True):
    if round_neck:
        ellipse(a,cx,top,width/2,depth,0.0)
    else:
        polygon(a,[(cx-width/2,top-0.2),(cx+width/2,top-0.2),(cx,top+depth)],0.0)

def fabric(a, kind, fill):
    # Generic fabric treatments, not textures taken from any garment image.
    a[a > 0] = fill
    if kind == 1:
        m=(np.sin(yy*2.7)+np.sin(xx*0.15) > 1.15) & (a>0)
        a[m] *= 0.72
    elif kind == 2:
        m=((np.sin(xx*2.1)*np.sin(yy*2.1)) > 0.72) & (a>0)
        a[m] *= 0.75

def top_model(cls):
    a=blank(); cx=14+rng.uniform(-0.6,0.6)
    top=rng.uniform(1.3,4.2); bottom=rng.uniform(24.0,27.0)
    bw=rng.uniform(5.0,7.3)                 # half width of torso
    nw=rng.uniform(2.0,3.6)
    long = cls in (2,4,6) and not (cls==6 and rng.random()<0.23)
    sleeveless = cls==0 and rng.random()<0.18
    sw=bw+rng.uniform(2.5,4.8)
    shoulder=top+rng.uniform(1.0,2.5)
    armpit=top+rng.uniform(8.0,11.0)
    if long:
        cuff_y=bottom-rng.uniform(0.2,3.0)
        cuff_x=bw+rng.uniform(0.2,2.0)
        left=[(cx-nw,top),(cx-bw,shoulder),(cx-sw,top+4),
              (cx-cuff_x-1.2,cuff_y),(cx-cuff_x+1.2,cuff_y),
              (cx-bw+0.3,armpit),(cx-bw+rng.uniform(-0.5,0.7),bottom)]
        right=[(2*cx-x,y) for x,y in left[::-1]]
        polygon(a,left+right)
    elif sleeveless:
        left=[(cx-nw,top),(cx-bw,shoulder),(cx-bw+0.5,armpit),
              (cx-bw+rng.uniform(-0.5,0.5),bottom)]
        polygon(a,left+[(2*cx-x,y) for x,y in left[::-1]])
    else:
        sl=rng.uniform(6.0,10.0)
        left=[(cx-nw,top),(cx-bw,shoulder),(cx-sw,top+3.7),
              (cx-sw+0.7,top+sl),(cx-bw,top+sl-0.8),
              (cx-bw+rng.uniform(-0.5,0.5),bottom)]
        polygon(a,left+[(2*cx-x,y) for x,y in left[::-1]])
    fill=rng.uniform(0.58,0.94)
    kind=rng.choice(3,p=[0.76,0.13,0.11]); fabric(a,kind,fill)
    round_neck = cls in (0,2) or rng.random()<0.35
    neck_cut(a,cx,top,nw*1.25,rng.uniform(1.4,2.8),round_neck)
    # Neck binding is a generic dark seam around the opening.
    if cls in (0,2):
        pts=[(cx-nw*0.75,top+0.7),(cx-nw*0.4,top+2.0),(cx,top+2.6),
             (cx+nw*0.4,top+2.0),(cx+nw*0.75,top+0.7)]
        polyline(a,pts,0.65,fill*0.5)
    if cls==2:
        # Sweatshirt hem/cuffs, and occasionally a hood or pouch.
        line(a,(cx-bw+0.5,bottom-1),(cx+bw-0.5,bottom-1),0.8,fill*0.65)
        if rng.random()<0.22:
            ellipse(a,cx,top+2.5,nw*1.15,3.2,fill*0.65)
            ellipse(a,cx,top+2.0,nw*0.75,2.0,0.0)
        if rng.random()<0.18:
            polygon(a,[(cx-3,bottom-6),(cx+3,bottom-6),(cx+2.4,bottom-1.8),(cx-2.4,bottom-1.8)],fill*0.8)
    if cls in (4,6):
        # Shirt collar or coat lapels and a central front opening.
        collar=rng.uniform(1.6,2.6) if cls==6 else rng.uniform(2.2,3.4)
        polygon(a,[(cx-nw,top),(cx-0.2,top+3.0),(cx-collar,top+4.8),(cx-nw-0.8,top+1.4)],fill*0.52)
        polygon(a,[(cx+nw,top),(cx+0.2,top+3.0),(cx+collar,top+4.8),(cx+nw+0.8,top+1.4)],fill*0.52)
        line(a,(cx,top+2.8),(cx,bottom),0.65 if cls==6 else 0.95,fill*0.3)
        if cls==4:
            line(a,(cx-nw,top+1),(cx-1.0,top+8.5),1.0,fill*0.35)
            line(a,(cx+nw,top+1),(cx+1.0,top+8.5),1.0,fill*0.35)
            if rng.random()<0.55:
                line(a,(cx-3,bottom-7),(cx-0.8,bottom-7),0.8,fill*0.5)
                line(a,(cx+0.8,bottom-7),(cx+3,bottom-7),0.8,fill*0.5)
        else:
            for by in np.arange(top+7,bottom-2,4.0): ellipse(a,cx+0.45,by,0.3,0.3,fill*1.15)
    return a

def trousers_model():
    a=blank(); cx=14+rng.uniform(-0.5,0.5)
    top=rng.uniform(0.8,3.0); bottom=rng.uniform(25.0,27.0)
    hw=rng.uniform(5.7,8.0); gap=rng.uniform(0.7,2.5)
    crotch=rng.uniform(10.0,14.0); taper=rng.uniform(-1.2,1.8)
    left=[(cx-hw,top),(cx-gap/2,top),(cx-gap/2,crotch),
          (cx-gap/2-1.0+taper,bottom),(cx-hw+taper,bottom)]
    right=[(2*cx-x,y) for x,y in left]
    polygon(a,left); polygon(a,right)
    rect(a,cx-hw,top,cx+hw,top+rng.uniform(1.2,2.3),0.75)
    fabric(a,rng.choice(3,p=[0.85,0.05,0.10]),rng.uniform(0.55,0.9))
    line(a,(cx-hw,top+2),(cx+hw,top+2),0.55,0.35)
    line(a,(cx,top+2),(cx,crotch),0.55,0.3)
    return a

def dress_model():
    a=blank(); cx=14+rng.uniform(-0.5,0.5)
    top=rng.uniform(1.0,3.5); bottom=rng.uniform(25.0,27.0)
    waist_y=rng.uniform(10.0,14.0); ww=rng.uniform(3.2,5.2)
    hw=rng.uniform(5.0,12.0)                # hem half width
    bw=rng.uniform(4.5,6.5); nw=rng.uniform(1.6,2.8)
    style=rng.integers(0,3)
    if style==0:                            # sleeveless, flared or straight
        left=[(cx-nw,top),(cx-bw,top+1.5),(cx-ww,waist_y),(cx-hw,bottom)]
        polygon(a,left+[(2*cx-x,y) for x,y in left[::-1]])
    elif style==1:                          # short sleeves
        sw=bw+rng.uniform(1.8,3.5)
        left=[(cx-nw,top),(cx-bw,top+1.5),(cx-sw,top+4),
              (cx-sw+0.5,top+8),(cx-bw,top+7),(cx-ww,waist_y),(cx-hw,bottom)]
        polygon(a,left+[(2*cx-x,y) for x,y in left[::-1]])
    else:                                   # long narrow sleeves
        left=[(cx-nw,top),(cx-bw,top+1.5),(cx-bw-2.5,top+4),
              (cx-bw-1.5,bottom-3),(cx-bw+0.5,bottom-3),
              (cx-bw,top+9),(cx-ww,waist_y),(cx-hw,bottom)]
        polygon(a,left+[(2*cx-x,y) for x,y in left[::-1]])
    fabric(a,rng.choice(3,p=[0.75,0.12,0.13]),rng.uniform(0.55,0.92))
    neck_cut(a,cx,top,nw*1.3,rng.uniform(1.3,3.2),rng.random()<0.6)
    if rng.random()<0.7: line(a,(cx-ww,waist_y),(cx+ww,waist_y),0.65,0.3)
    if hw>8:
        for d in np.linspace(-hw+2,hw-2,4):
            line(a,(cx+d*0.35,waist_y),(cx+d,bottom),0.45,0.48)
    return a

def bag_model():
    a=blank(); cx=14+rng.uniform(-0.5,0.5)
    left=rng.uniform(1.5,7.0); right=rng.uniform(21.0,26.5)
    top=rng.uniform(8.0,15.0); bottom=rng.uniform(24.0,27.0)
    style=rng.integers(0,4)
    if style==0:
        polygon(a,[(left+2,top),(right-2,top),(right,bottom),(left,bottom)])
    elif style==1:
        rect(a,left,top,right,bottom)
    elif style==2:
        ellipse(a,cx,(top+bottom)/2,(right-left)/2,(bottom-top)/2+1)
        rect(a,left,top+2,right,bottom-2)
    else:
        polygon(a,[(left,top+2),(left+3,top),(right-3,top),(right,top+2),
                   (right-1,bottom),(left+1,bottom)])
    fill=rng.uniform(0.58,0.9); fabric(a,0,fill)
    if rng.random()<0.82:
        hw=(right-left)*rng.uniform(0.20,0.33)
        htop=rng.uniform(0.7,5.0)
        pts=[]
        for t in np.linspace(0,np.pi,17):
            pts.append((cx+hw*np.cos(t),top-(top-htop)*np.sin(t)))
        polyline(a,pts,rng.uniform(0.65,1.15),fill)
        if rng.random()<0.35:
            pts2=[(x,y+1.6) for x,y in pts]; polyline(a,pts2,0.6,fill*0.8)
    if rng.random()<0.65:
        fy=top+rng.uniform(2.0,5.0)
        polyline(a,[(left+1,fy-1),(cx,fy+1),(right-1,fy-1)],0.7,fill*0.5)
    if rng.random()<0.35: rect(a,cx-0.7,top+3,cx+0.7,top+4.3,fill*1.15)
    return a

def shoe_model(cls):
    a=blank()
    toe=rng.uniform(0.8,2.8); heel=rng.uniform(24.0,26.5)
    sole_y=rng.uniform(22.0,24.0); bottom=min(27.0,sole_y+rng.uniform(1.0,2.0))
    fill=rng.uniform(0.58,0.9)
    if cls==7:                              # low trainer / sneaker
        ankle=rng.uniform(10.0,15.0)
        pts=[(toe,sole_y),(toe+0.5,20),(5.5,17.5),(10.0,14.0),
             (14.0,ankle),(18.0,ankle+1),(20.0,16.0),(heel,17.5),
             (heel,sole_y),(heel,bottom),(toe,bottom)]
        polygon(a,pts,fill)
        rect(a,toe,sole_y,heel,bottom,fill*0.9)
        for k in range(4):
            x=11.5+k*1.7
            line(a,(x,ankle+2.5),(x-3.0,ankle+5.0),0.6,fill*0.35)
        line(a,(toe+1,sole_y-0.8),(heel,sole_y-0.8),0.55,fill*0.35)
    elif cls==9:                            # ankle-height boot
        top=rng.uniform(2.0,8.0); shaft_l=rng.uniform(14.0,18.0)
        pts=[(toe,sole_y),(toe+0.5,20),(8.0,18.5),(shaft_l,15.0),
             (shaft_l,top),(heel-1,top+0.5),(heel,sole_y),(heel,bottom),(toe,bottom)]
        polygon(a,pts,fill)
        rect(a,toe,sole_y,heel,bottom,fill*0.85)
        line(a,(shaft_l+0.8,top+1),(shaft_l+0.8,17.0),0.65,fill*0.4)
        if rng.random()<0.35:
            # A raised heel leaves a characteristic notch below the sole.
            rect(a,toe+2,bottom,heel-5,bottom+1.3,0.0)
            rect(a,heel-3,bottom,heel,bottom+2.0,fill)
    else:                                   # open sandal, sometimes high heel
        rect(a,toe,sole_y,heel,bottom,fill)
        high=rng.random()<0.42
        if high:
            line(a,(toe+3,sole_y),(12,12),2.0,fill)
            line(a,(12,12),(19,7),1.6,fill)
            line(a,(19,7),(23,sole_y),1.8,fill)
            line(a,(23,sole_y),(23,bottom+2.0),1.0,fill)
            line(a,(toe+5,20),(18,20),1.4,fill)
        else:
            line(a,(toe+2,sole_y),(9,17),2.0,fill)
            line(a,(9,17),(17,sole_y),2.0,fill)
            line(a,(13,sole_y),(21,16.5),1.7,fill)
            if rng.random()<0.5: line(a,(20,16.5),(23,sole_y),1.0,fill)
    if rng.random()<0.5: a=mirror(a)
    return a

def make_model(cls):
    if cls==1: a=trousers_model()
    elif cls==3: a=dress_model()
    elif cls==8: a=bag_model()
    elif cls in (5,7,9): a=shoe_model(cls)
    else: a=top_model(cls)
    # Antialias the geometric drawing and include small generic shifts/scale changes.
    a=a.reshape(SIZE,S,SIZE,S).mean((1,3))
    if cls not in (5,7,9) and rng.random()<0.18: a=mirror(a)
    return np.clip(a,0,1).astype(np.float32)

def filters():
    # Binomial Gaussian and Gaussian derivatives, all fixed formulas.
    r=np.arange(-2,3); g=np.exp(-0.5*(r/1.15)**2); g=g/g.sum()
    gauss=np.outer(g,g)
    gx=np.outer(g,(-r)*g); gx=gx/(np.abs(gx).sum()+1e-12)*2
    gy=gx.T
    return np.stack([gauss,gauss,gauss,gauss,gx,gy]).astype(np.float32)[:,None]

def conv_features(images):
    f=filters(); pad=2
    x=np.pad(images,((0,0),(0,0),(pad,pad),(pad,pad)))
    # 5x5 stride two: output is 14x14, exactly as in network.json.
    out=np.zeros((len(images),6,14,14),np.float32)
    for o in range(6):
        for dy in range(5):
            for dx in range(5):
                out[:,o]+=f[o,0,dy,dx]*x[:,0,dy:dy+28:2,dx:dx+28:2]
    mu=out.mean((2,3),keepdims=True)
    out=(out-mu)/np.sqrt(((out-mu)**2).mean((2,3),keepdims=True)+1e-5)
    return out

def geometry_weights():
    """Hand-set regional and moment measurements on the positive shape channel."""
    q=np.arange(14,dtype=float)
    u=(q-6.5)/6.5
    v=(q-6.5)/6.5
    U,V=np.meshgrid(u,v)
    F=[]
    def add(a,scale): F.append(a/scale)
    add(np.ones((14,14)),200)                                      # 0 area
    add(U*U,200)                                                  # 1 horizontal spread
    add(V*V,200)                                                  # 2 vertical spread
    add(np.exp(-(U/.34)**2-((V-.72)/.38)**2),20)                 # 3 bottom centre
    add(np.exp(-((np.abs(U)-.47)/.22)**2-((V-.72)/.40)**2),20)  # 4 two lower sides
    add(np.exp(-((np.abs(U)-.91)/.24)**2-((V-.48)/.50)**2),20)  # 5 long sleeves
    add(np.exp(-(U/.34)**2-((V+.88)/.27)**2),15)                 # 6 top centre / handle
    add(np.exp(-((V+.48)/.52)**4),100)                            # 7 upper mass
    add(np.exp(-((V-.48)/.52)**4),100)                            # 8 lower mass
    add(np.exp(-((np.abs(U)-.34)/.20)**2-((V-.15)/.60)**4)
        -np.exp(-(U/.13)**2-((V-.15)/.60)**4),20)                # 9 dark front opening
    add(np.exp(-((np.abs(U)-.30)/.20)**2-((V+.68)/.28)**2)
        -np.exp(-(U/.12)**2-((V+.68)/.28)**2),15)                # 10 open collar
    add(np.exp(-((np.abs(U)-.87)/.28)**2-((V+.05)/.50)**4),30)  # 11 middle sides
    add(np.exp(-((np.abs(U)-.87)/.28)**2-((V-.75)/.35)**2),20)  # 12 flared hem / wide sole
    add(np.exp(-((np.abs(U)-.68)/.30)**2-((V+.72)/.40)**2),20)  # 13 upper side / boot shaft
    add(np.exp(-((V-.85)/.20)**2),30)                             # 14 sole / hem line
    add(np.exp(-(U/.30)**2-((V-.05)/.60)**4),30)                 # 15 solid central body
    W=np.zeros((len(F),1,14,14),np.float32)
    W[:,0]=np.stack(F)
    return W.reshape(len(F),-1),len(F)

def geometry_rules():
    # Columns are the measurements above.  These are category definitions, not fitted coefficients.
    R=np.zeros((10,16),np.float32)
    R[0,3]=.10; R[0,5]=-.22; R[0,9]=-.05; R[0,12]=-.04       # short-sleeve filled torso
    R[1,3]=-.34; R[1,4]=.22; R[1,2]=.08; R[1,12]=-.08        # separated trouser legs
    R[2,5]=.16; R[2,9]=-.30; R[2,10]=-.10; R[2,13]=-.04     # unopened sweater front
    R[3,3]=.24; R[3,4]=-.18; R[3,12]=.13; R[3,9]=-.06        # filled, often flared skirt
    R[4,5]=.20; R[4,9]=.25; R[4,10]=.10; R[4,15]=.03        # coat lapels/opening
    R[5,0]=-.80; R[5,1]=5.0; R[5,2]=-5.0; R[5,6]=-.45; R[5,7]=-.15; R[5,14]=.16
    R[6,5]=.10; R[6,9]=.15; R[6,10]=.15                       # collared button shirt
    R[7,1]=5.0; R[7,2]=-5.0; R[7,6]=-.45; R[7,7]=-.15; R[7,14]=.16
    R[8,1]=-3.0; R[8,2]=3.0; R[8,3]=.08; R[8,6]=.50; R[8,5]=-.45; R[8,9]=-.35
    R[9,1]=-.50; R[9,2]=.70; R[9,13]=.18; R[9,6]=-.10; R[9,14]=.05
    return R


def build(dest):
    models=np.stack([make_model(c) for c in range(NCLASS) for _ in range(NVAR)])[None]
    models=models.transpose(1,0,2,3)
    feat=conv_features(models)
    occupancy=np.clip((models-.15)*12.0,0,1)
    occupancy=occupancy.reshape(len(occupancy),1,14,2,14,2).mean((3,5))
    mu=occupancy.mean((2,3),keepdims=True)
    occupancy=(occupancy-mu)/np.sqrt(((occupancy-mu)**2).mean((2,3),keepdims=True)+1e-5)
    OW=(occupancy/np.sqrt(196)).reshape(NCLASS*NVAR,-1).astype(np.float32)
    # Each channel has unit variance; division makes each complete template unit norm.
    W=(feat/np.sqrt(np.prod(feat.shape[1:]))).reshape(NCLASS*NVAR,-1).astype(np.float32)
    layers=[
      {"name":"features","op":"conv","inputs":["image"],"out":6,"kernel":[5,5],"stride":2,"padding":2},
      {"name":"normalized","op":"norm","inputs":["features"]},
      {"name":"match","op":"dense","inputs":["normalized"],"out":NCLASS*NVAR},
      {"name":"byclass","op":"reshape","inputs":["match"],"shape":[NCLASS,NVAR,1]},
      {"name":"localbest","op":"maxpool","inputs":["byclass"],"size":[GROUP,1],"stride":[GROUP,1]},
      {"name":"best","op":"avgpool","inputs":["localbest"],"size":[NVAR//GROUP,1],"stride":[NVAR//GROUP,1]},
      {"name":"classvector","op":"reshape","inputs":["best"],"shape":[10]},
      {"name":"scores","op":"dense","inputs":["classvector"],"out":10},
      {"name":"occupancy_hi","op":"conv","inputs":["image"],"out":1,"kernel":[1,1],"stride":1,"padding":0},
      {"name":"occupancy_hi_relu","op":"relu","inputs":["occupancy_hi"]},
      {"name":"occupancy_lo","op":"conv","inputs":["image"],"out":1,"kernel":[1,1],"stride":1,"padding":0},
      {"name":"occupancy_lo_relu","op":"relu","inputs":["occupancy_lo"]},
      {"name":"occupancy_negative","op":"conv","inputs":["occupancy_lo_relu"],"out":1,"kernel":[1,1],"stride":1,"padding":0},
      {"name":"occupancy_clip","op":"add","inputs":["occupancy_hi_relu","occupancy_negative"]},
      {"name":"occupancy_pool","op":"avgpool","inputs":["occupancy_clip"],"size":[2,2],"stride":[2,2]},
      {"name":"geometry","op":"dense","inputs":["occupancy_pool"],"out":16},
      {"name":"geometry_scores","op":"dense","inputs":["geometry"],"out":10},
      {"name":"occupancy_norm","op":"norm","inputs":["occupancy_pool"]},
      {"name":"occupancy_match","op":"dense","inputs":["occupancy_norm"],"out":NCLASS*NVAR},
      {"name":"occupancy_byclass","op":"reshape","inputs":["occupancy_match"],"shape":[NCLASS,NVAR,1]},
      {"name":"occupancy_localbest","op":"maxpool","inputs":["occupancy_byclass"],"size":[GROUP,1],"stride":[GROUP,1]},
      {"name":"occupancy_best","op":"avgpool","inputs":["occupancy_localbest"],"size":[NVAR//GROUP,1],"stride":[NVAR//GROUP,1]},
      {"name":"occupancy_vector","op":"reshape","inputs":["occupancy_best"],"shape":[10]},
      {"name":"occupancy_scores","op":"dense","inputs":["occupancy_vector"],"out":10},
      {"name":"answer","op":"add","inputs":["scores","geometry_scores","occupancy_scores"]}
    ]
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    (dest/"network.json").write_text(json.dumps({"layers":layers},separators=(",",":")))
    GW,ng=geometry_weights()
    np.savez(dest/"weights.npz",**{"features.W":filters(),"features.b":np.zeros(6,np.float32),
                                   "match.W":W,"match.b":np.zeros(NCLASS*NVAR,np.float32),
                                   "scores.W":np.eye(10,dtype=np.float32)*.25,"scores.b":np.zeros(10,np.float32),
                                   "geometry.W":GW,"geometry.b":np.zeros(ng,np.float32),
                                   "geometry_scores.W":geometry_rules()*.35,"geometry_scores.b":np.zeros(10,np.float32),
                                   "occupancy_hi.W":np.full((1,1,1,1),12,np.float32),"occupancy_hi.b":np.array([-1.8],np.float32),
                                   "occupancy_lo.W":np.full((1,1,1,1),12,np.float32),"occupancy_lo.b":np.array([-2.8],np.float32),
                                   "occupancy_negative.W":np.full((1,1,1,1),-1,np.float32),"occupancy_negative.b":np.zeros(1,np.float32),
                                   "occupancy_match.W":OW,"occupancy_match.b":np.zeros(NCLASS*NVAR,np.float32),
                                   "occupancy_scores.W":np.eye(10,dtype=np.float32),
                                   "occupancy_scores.b":np.array([0,-.02,.12,-.02,0,.03,.03,.04,-.05,0],np.float32)})

if __name__=="__main__":
    build(sys.argv[1])
