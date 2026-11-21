#!/usr/bin/env python3
"""Build a data-free, hand-designed multiscale person/face shape detector."""
import json, math, sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)
L = []
W = {}

def layer(name, op, inputs, **settings):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(settings); L.append(d)

def conv(name, inp, weight, bias, stride=1, padding=0):
    weight=np.asarray(weight,np.float32); bias=np.asarray(bias,np.float32)
    layer(name,"conv",[inp],out=int(weight.shape[0]),kernel=[int(weight.shape[2]),int(weight.shape[3])],stride=stride,padding=padding)
    W[name+".W"]=weight; W[name+".b"]=bias

def dense(name, inp, weight, bias):
    weight=np.asarray(weight,np.float32); bias=np.asarray(bias,np.float32)
    layer(name,"dense",[inp],out=int(weight.shape[0])); W[name+".W"]=weight; W[name+".b"]=bias

# Luminance is normalized before differentiation, making edge thresholds largely
# independent of exposure and contrast.
lw=np.array([[[[0.299]],[[0.587]],[[0.114]]]],np.float32)
conv("luminance","image",lw,[0.0])
layer("lumnorm","norm",["luminance"])
layer("lum48","avgpool",["lumnorm"],size=[2,2],stride=[2,2])

# Eight signed Gaussian-derivative/Sobel filters, followed by rectification and
# combination of opposite signs into four unsigned HOG-like orientation maps.
sx=np.array([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])/8.0
sy=sx.T
angs=np.arange(8)*math.pi/4
edge=np.zeros((8,1,3,3),np.float32)
for j,a in enumerate(angs): edge[j,0]=math.cos(a)*sx+math.sin(a)*sy
conv("signed_edges","lum48",edge,np.zeros(8),padding=1)
layer("edge_relu","relu",["signed_edges"])
comb=np.zeros((4,8,1,1),np.float32)
for j in range(4): comb[j,j,0,0]=comb[j,j+4,0,0]=1.0
conv("orient_edges","edge_relu",comb,np.zeros(4))
layer("edge24","avgpool",["orient_edges"],size=[2,2],stride=[2,2])
layer("edgenorm","norm",["edge24"])
epad=np.zeros((8,4,1,1),np.float32); epad[:4,:,0,0]=np.eye(4,dtype=np.float32)
conv("edge_features","edgenorm",epad,np.zeros(8))

# Three deliberately broad skin-colour tents.  They encode chromatic order and
# brightness ranges rather than a race-specific RGB prototype.  Each tent is a
# sum of hinge penalties for falling outside general skin-colour ranges.
models=[
    # bounds for R-G, G-B, R-B, and luminance
    ((0.015,0.34),(-0.045,0.27),(0.035,0.52),(0.13,1.01)),
    ((-0.005,0.22),(-0.045,0.18),(0.012,0.34),(0.045,0.68)),
    ((0.005,0.27),(-0.10,0.16),(0.018,0.38),(0.28,1.01)),
]
vecs=[np.array([1.,-1.,0.]),np.array([0.,1.,-1.]),np.array([1.,0.,-1.]),np.array([.299,.587,.114])]
constraints=[]; cb=[]
for bounds in models:
    for v,(lo,hi) in zip(vecs,bounds):
        constraints.extend([-v,v]); cb.extend([lo,-hi])
constraints=np.asarray(constraints,np.float32)[:,:,None,None]
conv("skin_limits","image",constraints,np.asarray(cb,np.float32))
layer("skin_penalty","relu",["skin_limits"])
skw=np.zeros((3,24,1,1),np.float32)
for m in range(3): skw[m,8*m:8*m+8,0,0]=-5.0
conv("skin_tents0","skin_penalty",skw,np.ones(3))
layer("skin_tents","relu",["skin_tents0"])
layer("skin24","avgpool",["skin_tents"],size=[4,4],stride=[4,4])
layer("skinnorm","norm",["skin24"])
spad=np.zeros((8,3,1,1),np.float32)
for j in range(3): spad[4+j,j,0,0]=1.0
conv("skin_features","skinnorm",spad,np.zeros(8))

# The last feature is normalized coarse luminance.  Keeping all branches at
# eight channels permits legal elementwise merging with the available ops.
layer("lum24raw","avgpool",["lumnorm"],size=[4,4],stride=[4,4])
layer("lum24","norm",["lum24raw"])
lpad=np.zeros((8,1,1,1),np.float32); lpad[7,0,0,0]=1.0
conv("light_features","lum24",lpad,np.zeros(8))
layer("features","add",["edge_features","skin_features","light_features"])

# Template drawing utilities. Coordinates are generic normalized body-model
# coordinates, not measurements from any image.
def splat(a,ch,x,y,val=1.0):
    k=a.shape[-1]
    px=(x+1)*(k-1)/2; py=(y+1)*(k-1)/2
    x0=math.floor(px); y0=math.floor(py)
    for yy in (y0,y0+1):
      for xx in (x0,x0+1):
        if 0<=xx<k and 0<=yy<k:
          a[ch,yy,xx]+=val*max(0,1-abs(px-xx))*max(0,1-abs(py-yy))

def line(a,pts,val=1.0):
    # Draw an oriented polyline into unsigned gradient bins.  The expected
    # gradient is normal to the outline tangent, as in a HOG silhouette.
    for (x0,y0),(x1,y1) in zip(pts[:-1],pts[1:]):
      dx=x1-x0; dy=y1-y0
      n=max(2,int(math.ceil(math.hypot(dx,dy)*(a.shape[-1]-1)*1.4)))
      normal=(math.atan2(dx,-dy)%math.pi)/(math.pi/4)
      q=int(math.floor(normal))%4; f=normal-math.floor(normal)
      for t in np.linspace(0,1,n):
        x=x0+t*dx; y=y0+t*dy
        splat(a,q,x,y,val*(1-f)/n*5)
        splat(a,(q+1)%4,x,y,val*f/n*5)

def ellipse(a,cx,cy,rx,ry,val=1.0):
    n=max(12,int(2*math.pi*max(rx,ry)*(a.shape[-1]-1)*1.1))
    p=[(cx+rx*math.cos(t),cy+ry*math.sin(t)) for t in np.linspace(0,2*math.pi,n+1)]
    line(a,p,val)

def skin_fill(a,ch,cx,cy,rx,ry,val):
    k=a.shape[-1]
    for yy in range(k):
      y=2*yy/(k-1)-1
      for xx in range(k):
        x=2*xx/(k-1)-1
        d=((x-cx)/rx)**2+((y-cy)/ry)**2
        if d<=1: a[ch,yy,xx]+=val*(1-d*.35)

def finish(a):
    # L1 normalization gives a common, interpretable response scale across
    # template sizes and numbers of rasterized outline points.
    s=np.abs(a).sum()
    return a/max(s,1e-6)

def full_body(k,width,arm,lean,headskin):
    a=np.zeros((8,k,k),np.float32)
    hx=lean*.65; rx=.115+width*.025; ry=.125
    ellipse(a,hx,-.73,rx,ry,1.15)
    shoulder=.24+.15*width; waist=.14+.10*width
    # shoulders and tapered torso
    line(a,[(-shoulder+lean*.45,-.50),(hx,-.46),(shoulder+lean*.45,-.50)],1.1)
    line(a,[(-shoulder+lean*.45,-.48),(-waist+lean*.15,.12)],1.0)
    line(a,[( shoulder+lean*.45,-.48),( waist+lean*.15,.12)],1.0)
    # arms: down, out, or bent/action pose
    if arm==0:
      line(a,[(-shoulder+lean*.45,-.43),(-shoulder-.04+lean*.2,.25)],.8)
      line(a,[( shoulder+lean*.45,-.43),( shoulder+.04+lean*.2,.25)],.8)
    elif arm==1:
      line(a,[(-shoulder+lean*.45,-.43),(-.62+lean*.2,-.20)],.8)
      line(a,[( shoulder+lean*.45,-.43),( .62+lean*.2,-.20)],.8)
    else:
      line(a,[(-shoulder+lean*.45,-.43),(-.48+lean*.2,-.12),(-.30+lean*.1,.16)],.75)
      line(a,[( shoulder+lean*.45,-.43),( .48+lean*.2,-.12),( .30+lean*.1,.16)],.75)
    # separated legs, allowing a walking stance
    hip=.10+.07*width
    spread=.13 if arm==2 else .06
    line(a,[(-waist+lean*.15,.08),(-hip+lean*.05,.30),(-hip-spread,.92)],1.0)
    line(a,[( waist+lean*.15,.08),( hip+lean*.05,.30),( hip+spread,.92)],1.0)
    line(a,[(-.025+lean*.12,.12),(-.025-spread*.25,.88)],.55)
    line(a,[( .025+lean*.12,.12),( .025+spread*.25,.88)],.55)
    if headskin>=0: skin_fill(a,4+headskin,hx,-.73,rx*.78,ry*.78,.30)
    return finish(a)

def upper_body(k,width,lean,headskin,back=False):
    a=np.zeros((8,k,k),np.float32)
    hx=lean*.55; rx=.14+.035*width; ry=.16
    ellipse(a,hx,-.52,rx,ry,1.2)
    sh=.36+.15*width
    line(a,[(-sh+lean*.25,-.12),(-.18+lean*.4,-.23),(hx,-.25),(.18+lean*.4,-.23),(sh+lean*.25,-.12)],1.15)
    line(a,[(-sh+lean*.25,-.12),(-.25+lean*.1,.88)],.9)
    line(a,[( sh+lean*.25,-.12),( .25+lean*.1,.88)],.9)
    # neck/face-side details and a generic eye/mouth band
    line(a,[(hx-rx*.65,-.39),(hx-rx*.55,-.18)],.45)
    line(a,[(hx+rx*.65,-.39),(hx+rx*.55,-.18)],.45)
    if not back:
      line(a,[(hx-rx*.55,-.55),(hx+rx*.55,-.55)],.38)
      line(a,[(hx-rx*.38,-.45),(hx+rx*.38,-.45)],.25)
    if headskin>=0: skin_fill(a,4+headskin,hx,-.51,rx*.82,ry*.82,.42)
    return finish(a)

def face(k,aspect,skin,turn):
    a=np.zeros((8,k,k),np.float32)
    cx=.10*turn; rx=.55*aspect; ry=.68
    ellipse(a,cx,0,rx,ry,1.0)
    # Nose/profile and horizontal eye/mouth structure.
    if turn==0:
      line(a,[(cx-rx*.62,-.20),(cx+rx*.62,-.20)],.48)
      line(a,[(cx-rx*.42,.30),(cx+rx*.42,.30)],.34)
      # Haar-like luminance pattern: forehead/cheeks over eye and mouth bands.
      for x in np.linspace(cx-rx*.48,cx+rx*.48,4):
        splat(a,7,x,-.43,.12); splat(a,7,x,.05,.09)
        splat(a,7,x,-.18,-.13); splat(a,7,x,.30,-.08)
    else:
      side=1 if turn>0 else -1
      line(a,[(cx,-.42),(cx+side*rx*.75,-.15),(cx+side*rx*.92,.05),(cx+side*rx*.58,.18)],.5)
    if skin>=0: skin_fill(a,4+skin,cx,0,rx*.80,ry*.82,.62)
    return finish(a)

def make_bank(k):
    bank=[]; bias=[]
    # Full figures are useful from roughly 20 pixels high upward; the smallest
    # grid templates retain fewer variants because their outlines quantize.
    widths=(.45,.75,1.05) if k>=7 else (.55,.95)
    arms=(0,1,2) if k>=7 else (0,1)
    leans=(-.16,0,.16) if k>=9 else (0,)
    for wi in widths:
      for ar in arms:
       for le in leans:
        bank.append(full_body(k,wi,ar,le,-1)); bias.append(.68 if k>=9 else .78)
    # Head-and-shoulder models include colour-free backs of heads plus broad
    # skin variants for front/profile views.
    for wi in (.45,.85,1.20):
      for le in ((-.14,0,.14) if k>=9 else (0,)):
        bank.append(upper_body(k,wi,le,-1,True)); bias.append(.66 if k>=9 else .76)
        for sk in range(3):
          bank.append(upper_body(k,wi,le,sk,False)); bias.append(.62 if k>=9 else .72)
    # Face models are omitted at 5 cells only when their internal pattern would
    # collapse; even small head blobs are covered by upper-body templates.
    if k>=7:
      for asp in (.78,1.0,1.22):
       for sk in range(3):
        bank.append(face(k,asp,sk,0)); bias.append(.58)
      for sk in range(3):
       bank.append(face(k,.92,sk,-1)); bias.append(.60)
       bank.append(face(k,.92,sk,1)); bias.append(.60)
    return np.stack(bank),np.asarray(bias,np.float32)

# Independent scale maxima provide weak evidence which can add across, while a
# single very clear face or body may itself cross the final decision margin.
scale_outputs=[]
for k in (5,7,9,11,13,15,19,23):
    bank,bias=make_bank(k)
    n="shapes%d"%k
    conv(n,"features",bank,-bias)
    layer(n+"_relu","relu",[n])
    side=24-k+1
    layer(n+"_flat","reshape",[n+"_relu"],shape=[1,int(len(bank)*side),side])
    layer(n+"_max","maxpool",[n+"_flat"],size=[int(len(bank)*side),side],stride=[int(len(bank)*side),side])
    dense(n+"_vote",n+"_max",[[0.0],[1.0]],[0.0,0.0])
    scale_outputs.append(n+"_vote")

# A fixed margin compensates for maximizing many candidate locations.  This is
# a hand-set detector threshold, not a statistic fitted from supplied images.
dense("decision_bias","shapes5_max",[[0.0],[0.0]],[0.0,-7.45])
layer("scores","add",scale_outputs+["decision_bias"])

(OUT/"network.json").write_text(json.dumps({"layers":L},indent=2))
np.savez(OUT/"weights.npz",**{k:v.astype(np.float32) for k,v in W.items()})
