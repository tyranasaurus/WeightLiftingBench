import json, os, sys
import numpy as np

# All geometry below is expressed in the 28 by 28 design coordinate system.  It
# is a collection of ordinary, idealized clothing silhouettes rather than data.
N = 28
yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)

def poly(points, x=xx, y=yy):
    inside = np.zeros_like(x, dtype=bool)
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = points[i]; xj, yj = points[j]
        cross = ((yi > y) != (yj > y)) & (x < (xj-xi)*(y-yi)/(yj-yi + 1e-9) + xi)
        inside ^= cross
        j = i
    return inside.astype(np.float32)

def ellipse(cx, cy, rx, ry, x=xx, y=yy):
    return ((((x-cx)/rx)**2 + ((y-cy)/ry)**2) <= 1).astype(np.float32)

def line(x1, y1, x2, y2, width, x=xx, y=yy):
    vx, vy = x2-x1, y2-y1
    q = np.clip(((x-x1)*vx + (y-y1)*vy)/(vx*vx+vy*vy+1e-9), 0, 1)
    return (((x-(x1+q*vx))**2 + (y-(y1+q*vy))**2) <= width*width).astype(np.float32)

def union(*a):
    return np.maximum.reduce(a)

def tshirt(v, x, y):
    long = (v % 4) * .45
    shoulder = 7.0 + (v % 3)*.35
    bottom = 25.8 - (v % 2)*.7
    body = poly([(8.2,shoulder),(19.8,shoulder),(19.0,bottom),(9.0,bottom)],x,y)
    sl = poly([(8.4,shoulder),(5.8,shoulder+.4),(2.7,11.8+long),(6.8,14.2+long),(8.7,11.8)],x,y)
    sr = poly([(19.6,shoulder),(22.2,shoulder+.4),(25.3,11.8+long),(21.2,14.2+long),(19.3,11.8)],x,y)
    m = union(body,sl,sr)
    neck = ellipse(14,shoulder-.15,2.2+(v%2)*.35,1.9,x,y)
    return np.clip(m-.85*neck,0,1)

def trouser(v,x,y):
    wide = 0.25*(v%4)
    crotch = 11.5 + .7*(v%3)
    top = 2.3 + .4*(v%2)
    waist = poly([(7.4-wide,top),(20.6+wide,top),(20.1,11.2),(7.9,11.2)],x,y)
    left = poly([(7.9,8.5),(13.9,8.5),(12.3-wide*.25,26.7),(7.0-wide,26.7)],x,y)
    right = poly([(14.1,8.5),(20.1,8.5),(21.0+wide,26.7),(15.7+wide*.25,26.7)],x,y)
    gap = poly([(13.2,crotch),(14.8,crotch),(15.7,27),(12.3,27)],x,y)
    return np.clip(union(waist,left,right)-gap,0,1)

def pullover(v,x,y):
    shoulder=6.0+.35*(v%3); bottom=26.2-.4*(v%2)
    body=poly([(7.8,shoulder),(20.2,shoulder),(20.0,bottom),(8.0,bottom)],x,y)
    sl=poly([(8.1,shoulder),(5.5,7.0),(3.1,23.2),(7.2,24.7),(9.0,10.0)],x,y)
    sr=poly([(19.9,shoulder),(22.5,7.0),(24.9,23.2),(20.8,24.7),(19.0,10.0)],x,y)
    m=union(body,sl,sr)
    neck=ellipse(14,shoulder-.1,2.0+.3*(v%2),1.55,x,y)
    if v%5==4: # a generic hood
        m=union(m, ellipse(14,5.6,3.4,3.5,x,y)); neck=ellipse(14,5.6,1.8,2.0,x,y)
    return np.clip(m-.72*neck,0,1)

def dress(v,x,y):
    top=4.0+.45*(v%3); waist=12.0+.4*(v%2); flare=4.3+.55*(v%4)
    bod=poly([(10.1,top),(17.9,top),(18.1,waist),(9.9,waist)],x,y)
    skirt=poly([(9.9,waist),(18.1,waist),(18.1+flare,26.5),(9.9-flare,26.5)],x,y)
    m=union(bod,skirt)
    if v%3==0:
        m=union(m,poly([(10.2,top+.6),(7.6,6.3),(5.5,11.5),(9.2,12.4)],x,y),poly([(17.8,top+.6),(20.4,6.3),(22.5,11.5),(18.8,12.4)],x,y))
    elif v%3==1:
        m=union(m,poly([(10.3,top+.5),(7.8,6.2),(7.0,9.5),(10.0,9.8)],x,y),poly([(17.7,top+.5),(20.2,6.2),(21.0,9.5),(18.0,9.8)],x,y))
    return np.clip(m-.55*ellipse(14,top-.1,1.8,1.4,x,y),0,1)

def coat(v,x,y):
    shoulder=5.3+.45*(v%3); flare=.5*(v%4)
    body=poly([(7.8,shoulder),(20.2,shoulder),(20.3+flare,26.5),(7.7-flare,26.5)],x,y)
    sl=poly([(8.1,shoulder),(5.4,6.8),(2.9,23.0),(7.0,24.7),(9.0,10.0)],x,y)
    sr=poly([(19.9,shoulder),(22.6,6.8),(25.1,23.0),(21.0,24.7),(19.0,10.0)],x,y)
    m=union(body,sl,sr)
    m=np.clip(m-.24*line(14,8.0,14,26.5,.45,x,y),0,1)
    # broad lapels and a small neck opening
    m=np.clip(m-.35*poly([(11.0,shoulder),(14,11.8),(14,7.0)],x,y)-.35*poly([(17.0,shoulder),(14,11.8),(14,7.0)],x,y),0,1)
    return m

def sandal(v,x,y):
    sole=poly([(2.2,22.2),(23.8,21.5),(25.5,23.6),(5.2,25.2),(2.0,24.1)],x,y)
    if v%4==0:
        m=union(sole,line(6,21.7,10,12.2,1.05,x,y),line(9,13,17,21.9,1.05,x,y),line(13,20.5,20,15,1.1,x,y))
    elif v%4==1:
        m=union(sole,line(6,21.8,11,15.0,1.3,x,y),line(10,15,18,21.7,1.2,x,y),ellipse(16,18.5,5,1.4,x,y))
    elif v%4==2:
        heel=line(5,21.5,4.5,26,1.0,x,y)
        m=union(sole,heel,line(6,21,11,10,1.0,x,y),line(10,11,14,20,1.0,x,y),line(10,11,19,8,1.0,x,y))
    else:
        m=union(sole,line(4,21.8,8,14,1,x,y),line(7,14,18,20.5,1,x,y),line(9,20,20,14,1,x,y))
    return m

def shirt(v,x,y):
    shoulder=6.1+.45*(v%3); sleeve=(v%4)
    body=poly([(8.4,shoulder),(19.6,shoulder),(19.4,26),(8.6,26)],x,y)
    if sleeve<2:
        sl=poly([(8.7,shoulder),(5.8,6.8),(2.9,12.8),(7.5,15.0),(9.2,10.3)],x,y)
        sr=poly([(19.3,shoulder),(22.2,6.8),(25.1,12.8),(20.5,15.0),(18.8,10.3)],x,y)
    else:
        sl=poly([(8.7,shoulder),(5.8,6.8),(3.1,23.5),(7.1,24.5),(9.2,10.3)],x,y)
        sr=poly([(19.3,shoulder),(22.2,6.8),(24.9,23.5),(20.9,24.5),(18.8,10.3)],x,y)
    m=union(body,sl,sr)
    # V collar and button placket are characteristic but kept subtle.
    m=np.clip(m-.50*poly([(11.4,shoulder),(14,10.8),(14,6.4)],x,y)-.50*poly([(16.6,shoulder),(14,10.8),(14,6.4)],x,y),0,1)
    m=np.clip(m-.16*line(14,10,14,25,.35,x,y),0,1)
    return m

def sneaker(v,x,y):
    # Generic low, filled shoe profile; generated in one direction, then mirrored outside.
    top=poly([(3.0,20.0),(8.0,18.8),(11.0,13.0-(v%3)*.5),(15.0,18.0),(23.0,20.0),(25.5,22.7),(24.5,24.2),(4.0,24.2),(2.0,22.5)],x,y)
    sole=poly([(2.2,23.2),(25.5,23.0),(25.0,25.0),(4.0,25.4),(2.0,24.4)],x,y)
    return union(top,sole)

def bag(v,x,y):
    if v%5==4: # clutch / soft rectangular handbag
        body=poly([(3.0,13),(25.0,13),(24.0,23.5),(4.0,23.5)],x,y)
        return body
    top=10.2+.6*(v%3); side=.5*(v%4)
    body=poly([(4.3+side,top),(23.7-side,top),(24.8,25.2),(3.2,25.2)],x,y)
    outer=ellipse(14,10.3,6.3-(v%2)*.7,7.0,x,y)
    inner=ellipse(14,10.8,4.2-(v%2)*.5,5.3,x,y)
    handle=np.clip(outer-inner,0,1)*(y<=top+2.0)
    return union(body,handle)

def boot(v,x,y):
    ankle = 10.5 + .7*(v%4)
    shaft=poly([(7.5,3.0+(v%2)),(16.0,3.0+(v%2)),(16.8,20.2),(8.0,20.2)],x,y)
    foot=poly([(7.8,17.0),(16.6,17.0),(18.3,20.0),(24.8,21.4),(25.7,24.5),(7.0,24.8)],x,y)
    if v%3==2: # shorter ankle boot
        shaft=poly([(8.0,ankle),(16.2,ankle),(17.2,20.2),(8.0,20.2)],x,y)
    return union(shaft,foot)

makers=[tshirt,trouser,pullover,dress,coat,sandal,shirt,sneaker,bag,boot]

# Exactly the same number of analytic views is made for every class.  The views
# cover modest generic changes of width, centering, and handedness.
protos=[]
for cls, make in enumerate(makers):
    for v in range(10):
        for sx in (.94,1.06):
            for dx in (-.7,.7):
                X=(xx-dx-14)/sx+14
                Y=yy
                t=make(v,X,Y)
                if cls in (5,7,9) and ((v + (dx>0)) % 2):
                    t=t[:,::-1].copy()
                protos.append((cls,t))

levels=3
# Four hinge levels turn each prototype correlation into a steep but smooth
# approximation of "best matching silhouette".
hinges=(.22,.38,.52,.65)
Hproto=len(protos)*len(hinges)
rules=[]

def rect(x0,x1,y0,y1):
    return ((xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)).astype(np.float32)

def add_rule(wmap, threshold, votes, gain=1.0, channel=0, normed=False):
    # Rules use absolute multilevel occupancy unless explicitly marked as a
    # normalized contrast.
    rules.append((wmap.astype(np.float32)*gain, -threshold*gain, votes, channel, normed))

# Broad, semantic part tests.  Each one is deliberately a generic clothing rule.
# Votes are additions to the ten class scores when its half-space is active.
# A split extending below the crotch is strong trouser evidence.
legs=(rect(6,12,15,27)+rect(16,22,15,27))/144
lower_gap=rect(12,16,17,27)/40
add_rule(legs-1.4*lower_gap, .12, {1:5.0,3:-1.2,4:-.6}, gain=10.0)
add_rule(1.6*lower_gap-legs, .01, {3:.35,1:-2.5}, gain=5.0)
# A dress flares into the lower outer quadrants without sleeve mass in the
# corresponding upper outer quadrants.
skirt=(rect(4,24,17,27)/200)
waist=rect(8,20,8,15)/84
side_skirt=(rect(3,8,18,27)+rect(20,25,18,27))/90
lower_outer=(rect(3,9,17,27)+rect(19,25,17,27))/120
upper_outer=(rect(2,9,7,15)+rect(19,26,7,15))/112
add_rule(lower_outer-1.05*upper_outer, .035, {3:3.0,1:-1.0,4:-.4}, gain=8.0)
# Horizontal footwear has ink low down and very little in the upper half.
shoe_band=rect(2,26,14,26)/288
upper=rect(4,24,2,13)/220
add_rule(shoe_band-1.2*upper, .11, {5:1.5,7:2.0,9:1.2,0:-1,1:-1,2:-1,3:-1,4:-1,6:-1,8:-.5}, gain=8.0)
# A tall central shaft distinguishes boots from low shoes.
shaft=rect(7,18,4,17)/143
outer_upper=(rect(2,7,4,17)+rect(18,26,4,17))/169
add_rule(shaft-.7*outer_upper, .10, {9:2.0,7:-1.5,5:-1.5}, gain=6.0)
# Solid low shoe versus sparse sandal straps.
lowfill=rect(3,25,18,25)/154
midfill=rect(5,23,12,19)/126
add_rule(lowfill+.35*midfill, .28, {7:3.0,5:-2.0}, gain=5.0)
add_rule(.65*shoe_band-lowfill-.25*midfill, -.08, {5:3.0,7:-1.5}, gain=5.0)
# Bag body and handle arch, with empty upper corners.
body=rect(3,25,11,26)/330
handle_sides=(rect(7,11,4,13)+rect(17,21,4,13))/72
handle_hole=rect(11,17,5,11)/36
add_rule(body+.55*handle_sides-.35*handle_hole-.7*rect(2,26,0,5)/120, .20, {8:2.0,0:-.5,2:-.5,4:-.5,6:-.5}, gain=5.0)
# Short sleeves and long sleeves are useful soft cues among upper garments.
shortarms=(rect(2,8,7,15)+rect(20,26,7,15))/96
longarms=(rect(2,8,15,25)+rect(20,26,15,25))/120
centerbody=rect(8,20,8,25)/204
add_rule(shortarms-.55*longarms+.25*centerbody, .04, {0:2.2,6:.8,2:-.7,4:-.7}, gain=5.0)
add_rule(longarms-.35*shortarms+.2*centerbody, .05, {2:1.2,4:1.2,6:.5,0:-1.5}, gain=5.0)
# A central opening/collar gives modest coat or shirt evidence.
placket=(rect(13,15,10,26)/32)
body_sides=(rect(9,13,10,26)+rect(15,19,10,26))/128
add_rule(body_sides-1.8*placket, .04, {4:1.1,6:1.0,2:-1.0}, gain=5.0)
H=Hproto+len(rules)

Wp=np.zeros((H,levels*N*N),np.float32)
bp=np.zeros(H,np.float32)
Wr=np.zeros_like(Wp); br=np.zeros(H,np.float32)
Wo=np.zeros((10,H),np.float32); bo=np.zeros(10,np.float32)
level_mix=(.62,.28,.10)
# Equal class normalization prevents the prototype count from becoming a prior.
per_class=sum(1 for c,t in protos if c==0)
for pi,(cls,t) in enumerate(protos):
    sd=float(t.std())
    if sd < 1e-5: continue
    z=(t-t.mean())/sd
    for hi,th in enumerate(hinges):
        row=pi*len(hinges)+hi
        for ch,mix in enumerate(level_mix):
            Wp[row,ch*N*N:(ch+1)*N*N]=z.reshape(-1)*(mix/(N*N))
        bp[row]=-th
        # Successive hinges make high correlation disproportionately valuable.
        Wo[cls,row]=(1.0,1.4,2.1,3.0)[hi]*5.5/per_class
for j,(wm,bias,votes,ch,normed) in enumerate(rules):
    row=Hproto+j
    target=Wp if normed else Wr
    target[row,ch*N*N:(ch+1)*N*N]=wm.reshape(-1)
    if normed: bp[row]=bias
    else: br[row]=bias
    for cls,val in votes.items(): Wo[cls,row]=val
# Hand priors compensate only for generic silhouette-family ambiguity.
bo[:]=np.array([.55,.60,.55,-.65,.55,.55,.80,.65,.45,.50],np.float32)

layers=[
 {"name":"level","op":"conv","inputs":["image"],"out":levels,"kernel":[1,1],"stride":1,"padding":0},
 {"name":"level_relu","op":"relu","inputs":["level"]},
 {"name":"level_norm","op":"norm","inputs":["level_relu"]},
 {"name":"prototype_pre","op":"dense","inputs":["level_norm"],"out":H},
 {"name":"rule_pre","op":"dense","inputs":["level_relu"],"out":H},
 {"name":"combined","op":"add","inputs":["prototype_pre","rule_pre"]},
 {"name":"evidence","op":"relu","inputs":["combined"]},
 {"name":"scores","op":"dense","inputs":["evidence"],"out":10}
]
weights={
 "level.W":np.ones((levels,1,1,1),np.float32),
 "level.b":np.array([0,-.15,-.45],np.float32),
 "prototype_pre.W":Wp,"prototype_pre.b":bp,
 "rule_pre.W":Wr,"rule_pre.b":br,
 "scores.W":Wo,"scores.b":bo
}
out=sys.argv[1] if len(sys.argv)>1 else '.'
os.makedirs(out,exist_ok=True)
with open(os.path.join(out,'network.json'),'w') as f: json.dump({"layers":layers},f,separators=(',',':'))
np.savez(os.path.join(out,'weights.npz'),**weights)
