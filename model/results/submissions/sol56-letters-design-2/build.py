#!/usr/bin/env python3
"""Build a data-independent, synthetic-shape ASL classifier."""
import json, sys
from pathlib import Path
import numpy as np

# The synthetic drawings below are deliberately schematic.  Their dimensions are
# conventional drawing coordinates, not measurements from any supplied image.
S = 4                         # antialiasing grid scale
H = W = 28
Y, X = np.mgrid[0:H*S, 0:W*S].astype(np.float32) / S


def segdist(x1, y1, x2, y2):
    dx, dy = x2-x1, y2-y1
    q = np.clip(((X-x1)*dx + (Y-y1)*dy) / (dx*dx+dy*dy+1e-6), 0, 1)
    return np.sqrt((X-(x1+q*dx))**2 + (Y-(y1+q*dy))**2)


def capsule(mask, a, b, r):
    np.maximum(mask, np.clip(1.35-(segdist(*a, *b)/r), 0, 1), out=mask)


def ellipse(mask, cx, cy, rx, ry, amount=1.0):
    d = np.sqrt(((X-cx)/rx)**2 + ((Y-cy)/ry)**2)
    np.maximum(mask, amount*np.clip(1.25-d*1.25, 0, 1), out=mask)


def stroke(lines, pts, r=.38, amount=1.0):
    for a,b in zip(pts[:-1],pts[1:]):
        np.maximum(lines, amount*np.clip(1.3-segdist(*a,*b)/r,0,1), out=lines)


def transform_point(p, rot=0., scale=1., dx=0., dy=0.):
    a=np.deg2rad(rot); c,s=np.cos(a),np.sin(a)
    x,y=(p[0]-14)*scale,(p[1]-15)*scale
    return (14+c*x-s*y+dx, 15+s*x+c*y+dy)


def render(cls, rot=0., scale=1., style=0, polarity=1):
    """Draw a generic hand diagram for one ASL letter."""
    m=np.zeros_like(X); line=np.zeros_like(X); holes=np.zeros_like(X)
    wide=1.0 + (-.10,.08,.0)[style%3]
    spread=(-.45,0,.45)[style%3]
    def P(p): return transform_point(p,rot,scale)
    def cap(a,b,r): capsule(m,P(a),P(b),r*scale*wide)
    def ell(cx,cy,rx,ry):
        # ellipse is represented by several broad crossing capsules so it rotates too
        cap((cx-rx*.70,cy),(cx+rx*.70,cy),ry*.72)
        cap((cx,cy-ry*.70),(cx,cy+ry*.70),rx*.72)
    def ln(pts,r=.34,amt=1): stroke(line,[P(q) for q in pts],r*scale,amt)
    def hole(cx,cy,rx,ry):
        # Holes are antialiased ellipses in transformed coordinates.
        cc=P((cx,cy)); a=np.deg2rad(rot); ca,sa=np.cos(a),np.sin(a)
        xx=(X-cc[0])*ca+(Y-cc[1])*sa; yy=-(X-cc[0])*sa+(Y-cc[1])*ca
        d=np.sqrt((xx/(rx*scale))**2+(yy/(ry*scale))**2)
        np.maximum(holes,np.clip(1.35-d*1.35,0,1),out=holes)
    def wrist(cx=14): cap((cx-2.8,29),(cx,20),3.5)
    def palm(cx=14,cy=19,rx=5.8,ry=7.0): ell(cx,cy,rx,ry); wrist(cx)
    def finger(x,tipy,basey=15,r=1.35,lean=0):
        cap((x,basey),(x+lean*.45,(basey+tipy)/2),r)
        cap((x+lean*.45,(basey+tipy)/2),(x+lean,tipy),r*.91)
    def fist():
        palm(14,19,6.0,6.6)
        for j,x in enumerate((9.7,12.5,15.4,18.2)):
            ell(x,11.8+(j==3)*.5,2.0,2.5)
        ln([(8.8,14),(18.8,14)],.36)
        for x in (11.2,14.1,17.0): ln([(x,10.5),(x-.25,14)],.30)
    # 0 A
    if cls==0:
        fist(); cap((8.7,20),(8.0,14),1.65); ln([(9.2,19),(8.8,14)],.30)
    # 1 B
    elif cls==1:
        palm()
        for j,(x,t) in enumerate(zip((9.5,12.3,15.2,18.0),(4.3,2.5,3.0,5.3))): finger(x,t,15,1.28,(-.4,-.1,.15,.45)[j]+spread*.18)
        cap((8.5,18),(17.3,16.1),1.25); ln([(9,18.3),(17.4,16.2)],.32)
        for x in (10.9,13.8,16.6): ln([(x,5.4),(x,14.0)],.25)
    # 2 C: a side-on curved hand, open to the right
    elif cls==2:
        wrist(10.5); cap((10,22),(8.2,13.2),3.8)
        cap((8.3,13),(11,7.0),2.5); cap((11,7),(17.7,6.0),1.9)
        cap((9.7,20),(13.2,22.5),2.7); cap((13.2,22.5),(18.5,20.5),1.65)
        for y in (8.0,9.2,10.4): ln([(10,y),(17.2,y-1.1)],.24,.65)
    # 3 D
    elif cls==3:
        palm(); finger(10.7,2.4,15,1.45,-.3)
        cap((12.0,15.5),(17.5,12.2),2.0); cap((17.5,12.2),(18.2,17),1.6)
        cap((18.2,17),(14.2,18.0),1.45); cap((14.2,18),(11.8,15.5),1.25)
        hole(15.0,15.0,2.15,2.0); ln([(13,12.4),(18,14.0)],.28)
    # 4 E
    elif cls==4:
        fist();
        for j,y in enumerate((10.2,11.1,12.0,12.8)): cap((9.2+j*2.7,y),(10.1+j*2.5,16.3),1.25)
        ln([(9.2,16),(18.2,16)],.42)
    # 5 F
    elif cls==5:
        palm()
        for j,(x,t) in enumerate(((12.9,3.2),(15.7,2.4),(18.4,4.2))): finger(x,t,14.8,1.22,(j-1)*.35+spread*.3)
        cap((9.1,17.8),(10.7,12.2),1.5); cap((10.7,12.2),(14.0,12.0),1.35)
        cap((14.0,12.0),(16.3,16.8),1.25); cap((16.3,16.8),(11.2,18.2),1.25)
        hole(13.2,15.0,1.65,1.8)
    # 6 G
    elif cls==6:
        wrist(10); ell(10.5,19,4.8,5.4); cap((9.5,15),(21.8,12.5),1.55)
        cap((10.5,18.0),(18.2,19.2),1.45); ln([(10,15.3),(19.8,13.2)],.28)
    # 7 H
    elif cls==7:
        wrist(9.5); ell(10.2,19,4.7,5.4)
        cap((9.5,14.4),(22.0,11.7),1.35); cap((9.8,16.4),(22.1,14.1),1.30)
        ln([(10.2,15.5),(21.5,13.0)],.27)
    # 8 I
    elif cls==8:
        fist(); finger(18.6,4.2,13.5,1.25,.45); ln([(17.1,12),(18.3,6)],.27)
    # 9 K
    elif cls==9:
        palm(); finger(11.1,3.0,15,1.35,-1.2-spread); finger(15.2,2.8,15,1.35,1.8+spread)
        cap((8.6,19.2),(15.1,12.5),1.38); ln([(12.8,16),(15.4,12)],.3)
    # 10 L
    elif cls==10:
        palm(); finger(11.0,2.4,15,1.45,-.2); cap((10.2,18),(21.7,17.0),1.55); ln([(11.5,17.6),(19.5,17.1)],.28)
    # 11 M
    elif cls==11:
        fist(); cap((8.4,18.7),(16.0,14.9),1.38)
        for j in range(3): ln([(9.2+j*2.8,10.6),(10.1+j*2.6,16.3)],.42)
    # 12 N
    elif cls==12:
        fist(); cap((8.5,18.7),(14.7,14.5),1.45)
        for j in range(2): ln([(10.0+j*3.2,10.5),(10.8+j*2.9,16.2)],.45)
        ln([(16.2,11.2),(17.0,15)],.25)
    # 13 O
    elif cls==13:
        wrist(12.0); cap((11.2,22),(8.5,14.0),3.8); cap((8.8,13.5),(12.1,7.4),2.5)
        cap((12.1,7.4),(18.5,9.0),2.0); cap((18.5,9),(20.0,15.5),1.8)
        cap((20,15.5),(16.8,20.2),1.8); cap((16.8,20.2),(11.5,21),2.0)
        hole(14.7,14.0,3.25,4.4)
    # 14 P (K directed down)
    elif cls==14:
        wrist(11.0); ell(11.5,14.0,5.0,5.2)
        cap((12.5,15.5),(17.0,25.2),1.45); cap((14.1,14.5),(21.0,21.0),1.35)
        cap((8.5,13.5),(15.1,18.2),1.25); ln([(13.4,16),(18.5,22)],.28)
    # 15 Q (G directed down)
    elif cls==15:
        wrist(10.0); ell(10.5,14.5,4.8,5.2)
        cap((11.7,16.5),(16.8,25.5),1.5); cap((9.0,16.0),(8.2,22.8),1.35); ln([(12,17),(16,24)],.28)
    # 16 R
    elif cls==16:
        palm(); cap((11.0,15),(15.0,2.8),1.38); cap((15.2,15),(11.5,3.2),1.35)
        ln([(12.0,13.5),(14.3,5.0)],.3)
    # 17 S
    elif cls==17:
        fist(); cap((8.8,18.3),(17.5,14.6),1.55); ln([(9.3,18.5),(17.5,14.7)],.38)
    # 18 T
    elif cls==18:
        fist(); cap((9.0,19.0),(13.4,13.0),1.45); ln([(12.0,10.7),(13.3,16.2)],.48); ln([(15.2,10.8),(15.8,15)],.28)
    # 19 U
    elif cls==19:
        palm(); finger(11.8,2.8,15,1.42,.15); finger(14.6,2.6,15,1.42,-.05)
        ln([(13.2,4.0),(13.2,14.3)],.25)
    # 20 V
    elif cls==20:
        palm(); finger(11.5,2.8,15,1.35,-2.0-spread); finger(15.0,2.8,15,1.35,2.1+spread)
        ln([(12.5,13),(11.2,5)],.25); ln([(14,13),(16.3,5)],.25)
    # 21 W
    elif cls==21:
        palm(); finger(10.2,4.0,15,1.23,-1.6-spread); finger(13.5,2.2,15,1.30,0); finger(16.6,3.3,15,1.22,1.65+spread)
        ln([(11.4,13),(10,6)],.23); ln([(14.9,13),(16.8,5.5)],.23)
    # 22 X
    elif cls==22:
        fist(); cap((10.8,14.0),(11.0,6.1),1.48); cap((11.0,6.1),(14.5,7.2),1.4); cap((14.5,7.2),(13.6,10.6),1.28)
        ln([(11.5,10.2),(14,9.5)],.3)
    # 23 Y
    elif cls==23:
        fist(); cap((9.0,18.4),(3.2,14.2),1.48); finger(18.2,4.0,14,1.30,.65); ln([(17.1,12),(18.5,5.5)],.28)
    # Add generic joint creases and flexion lines. These internal parts matter
    # especially for compact letters whose outer silhouettes are very similar.
    if cls not in (2,6,7,13,14,15):
        ln([(10.0,21.0),(12.0,20.4)],.20,.45); ln([(16.0,20.4),(18.0,20.8)],.20,.4)
    if cls in (0,4,11,12,17,18):
        for j in range(4):
            x=9.8+j*2.75
            ln([(x,11.0),(x+.15,14.2)],.26,.72)
            ln([(x-.8,13.0),(x+.8,13.2)],.22,.55)
    # Many real front-view signs show dark inter-finger clefts. Model these from
    # anatomical knowledge as tapered creases, not as traced image contours.
    if cls in (1,3,5,8,9,10,16,19,20,21,23):
        xs={1:(10.9,13.8,16.6),3:(11.8,),5:(13.8,16.6),8:(17.0,),9:(12.7,14.8),10:(11.8,),16:(12.8,14.1),19:(13.2,),20:(13.2,),21:(11.9,14.8),23:(17.0,)}[cls]
        for x in xs: ln([(x,8.2),(x,14.6)],.34,.88)
    m *= (1.0-holes)
    # A broad, nonuniform background and shaded hand.  Polarity variants model
    # either a light or a dark hand without using any data-derived color values.
    bg=.50 + .035*(X/28-.5) - .025*(Y/28-.5)
    shade=.70 + .12*(X/28) - .08*(Y/28)
    hand=bg + polarity*.36*shade
    im=bg*(1-m)+hand*m
    im -= polarity*.25*line*m
    im=np.clip(im,0,1)
    # exact area downsampling from the 4x antialiasing grid
    return im.reshape(28,S,28,S).mean((1,3)).astype(np.float32)


def filters():
    yy,xx=np.mgrid[-2:3,-2:3].astype(np.float32)
    g=np.exp(-(xx*xx+yy*yy)/3.2)
    fs=[]
    for q in range(8):
        a=np.pi*q/8; f=(xx*np.cos(a)+yy*np.sin(a))*g; f-=f.mean(); f/=np.sum(np.abs(f))
        fs += [f,-f]
    lap=(xx*xx+yy*yy-2.25)*g; lap-=lap.mean(); lap/=np.sum(np.abs(lap)); fs += [lap,-lap]
    blur=g/g.sum(); fs += [blur,-blur]
    return np.asarray(fs,np.float32)[:,None]


def features(im, F, mix):
    # Same fixed front end as network: padded convolution, ReLU, 1x1 mixing,
    # 2x2 averaging, then exact per-channel spatial normalization.
    mu=im.mean(); im=(im-mu)/np.sqrt(((im-mu)**2).mean()+1e-5)
    p=np.pad(im,2)
    a=np.empty((len(F),28,28),np.float32)
    for k,f in enumerate(F[:,0]):
        z=np.zeros((28,28),np.float32)
        for y in range(5):
            for x in range(5): z += f[y,x]*p[y:y+28,x:x+28]
        a[k]=np.maximum(z,0)
    b=np.einsum('oi,ihw->ohw',mix[:,:,0,0],a)
    b=b.reshape(len(mix),14,2,14,2).mean((2,4))
    mu=b.mean((1,2),keepdims=True); va=((b-mu)**2).mean((1,2),keepdims=True)
    return (b-mu)/np.sqrt(va+1e-5)


def build(dst):
    dst=Path(dst); dst.mkdir(parents=True,exist_ok=True)
    F=filters(); mix=np.zeros((10,20,1,1),np.float32)
    for q in range(8): mix[q,2*q:2*q+2,0,0]=1
    mix[8,16:18,0,0]=1
    mix[9,18,0,0]=1; mix[9,19,0,0]=-1
    # Equal numbers of hand-drawn nuisance variants for every class. Their
    # correlation maps are pooled both over translation and over variant.
    variants=[]
    settings=[]
    for polarity in (1,-1):
        for style in range(3):
            for rot in (-5,5):
                settings.append((rot,style,polarity))
    V=len(settings)
    # A 10x10 template on the 14x14 feature map leaves a 5x5 translation range,
    # and tolerates harmless variation at the crop boundary.
    for c in range(24):
        for rot,style,pol in settings:
            im=render(c,rot=rot,scale=(.91,1.0,1.09)[style],style=style,polarity=pol)
            f=features(im,F,mix)[:,2:12,2:12].copy()
            # Cosine-style template normalization avoids favoring a drawing just
            # because it contains more ink or more contour length.
            f /= np.sqrt(np.sum(f*f)+1e-6)
            variants.append(f)
    T=np.asarray(variants,np.float32)
    layers=[
      {'name':'inorm','op':'norm','inputs':['image']},
      {'name':'oriented','op':'conv','inputs':['inorm'],'out':20,'kernel':[5,5],'stride':1,'padding':2},
      {'name':'halfedges','op':'relu','inputs':['oriented']},
      {'name':'features','op':'conv','inputs':['halfedges'],'out':10,'kernel':[1,1],'stride':1,'padding':0},
      {'name':'cells','op':'avgpool','inputs':['features'],'size':[2,2],'stride':[2,2]},
      {'name':'fnorm','op':'norm','inputs':['cells']},
      {'name':'templates','op':'conv','inputs':['fnorm'],'out':24*V,'kernel':[10,10],'stride':1,'padding':0},
      {'name':'all_matches','op':'reshape','inputs':['templates'],'shape':[24,V*5,5]},
      {'name':'best_match','op':'maxpool','inputs':['all_matches'],'size':[V*5,5],'stride':[V*5,5]},
      {'name':'scores','op':'reshape','inputs':['best_match'],'shape':[24]},
    ]
    weights={
      'oriented.W':F.astype(np.float32),'oriented.b':np.zeros(20,np.float32),
      'features.W':mix.astype(np.float32),'features.b':np.zeros(10,np.float32),
      'templates.W':T.astype(np.float32),'templates.b':np.zeros(24*V,np.float32),
    }
    (dst/'network.json').write_text(json.dumps({'layers':layers},indent=2))
    np.savez(dst/'weights.npz',**weights)

if __name__=='__main__':
    build(sys.argv[1] if len(sys.argv)>1 else '.')
