#!/usr/bin/env python3
"""Build a formula-derived galaxy morphology network without reading any data."""
import json, math, sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/out")
OUT.mkdir(parents=True, exist_ok=True)
layers = []
weights = {}

def layer(name, op, inputs, **settings):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(settings); layers.append(d)

def dense(name, inp, W, b=None):
    W = np.asarray(W, np.float32)
    W = W.reshape(W.shape[0], -1)
    if b is None: b = np.zeros(W.shape[0], np.float32)
    layer(name, "dense", [inp], out=int(W.shape[0]))
    weights[name+".W"] = W
    weights[name+".b"] = np.asarray(b, np.float32)

def conv(name, inp, W, b=None, stride=1, padding=0):
    W = np.asarray(W, np.float32)
    if b is None: b = np.zeros(W.shape[0], np.float32)
    layer(name, "conv", [inp], out=int(W.shape[0]), kernel=[int(W.shape[2]),int(W.shape[3])], stride=stride, padding=padding)
    weights[name+".W"] = W
    weights[name+".b"] = np.asarray(b, np.float32)

def contrast_rows(targets, n, strength=1.0):
    A=np.empty((10,n),np.float32)
    for k,t in enumerate(targets):
        a=np.full(n,-1.0/n,np.float32)
        for j in t: a[j]+=1.0/len(t)
        A[k]=strength*a
    return A

# Hand-set formula-scale offsets balance the independent expert banks.
CAL=np.array([-1.50,-1.88,.46,-.40,-.32,-1.79,-.92,-1.87,2.74,1.05],np.float32)

# Coordinates are fixed by the stated image geometry, not by examples.
N=32
Y,X=np.mgrid[0:N,0:N].astype(np.float64)
X=X+0.5-N/2; Y=Y+0.5-N/2
R=np.sqrt(X*X+Y*Y); TH=np.arctan2(Y,X)
def unit(a, zero_mean=True):
    a=np.asarray(a,np.float64)
    if zero_mean: a=a-a.mean()
    n=np.sqrt(np.sum(a*a))
    return (a/max(n,1e-12)).astype(np.float32)
def ellipse(scale,q,ang,power=1.0):
    u=X*np.cos(ang)+Y*np.sin(ang); v=-X*np.sin(ang)+Y*np.cos(ang)
    rr=np.sqrt((u/scale)**2+(v/(scale*q))**2+1e-8)
    return np.exp(-(rr**power))
def oriented_gauss(sx,sy,ang,cx=0.,cy=0.):
    xx=X-cx; yy=Y-cy
    u=xx*np.cos(ang)+yy*np.sin(ang); v=-xx*np.sin(ang)+yy*np.cos(ang)
    return np.exp(-.5*((u/sx)**2+(v/sy)**2))

# Normalize each displayed colour channel, pool away pixel noise, then form luminance.
layer("n0","norm",["image"])
layer("p0","avgpool",["n0"],size=[4,4],stride=[4,4])
Wlum=np.array([[[[0.36]],[[0.44]],[[0.20]]]],np.float32)
conv("lum","p0",Wlum)
layer("lumn","norm",["lum"])

# Elliptical isophote bank. Max over position angle and scale yields a robust axis-ratio signature.
qs=[1.0,.82,.66,.50,.36,.25,.17,.11]
scales=[1.5,2.25,3.4,5.1,7.5]
angs=[math.pi*j/12 for j in range(12)]
EW=[]
for q in qs:
    for s in scales:
        for a in angs: EW.append(unit(ellipse(s,q,a,power=.82)))
dense("ell_d","lumn",EW)
layer("ell_rs","reshape",["ell_d"],shape=[len(qs),len(scales),len(angs)])
layer("ell_o","maxpool",["ell_rs"],size=[1,len(angs)],stride=[1,len(angs)])
layer("ell_q","maxpool",["ell_o"],size=[len(scales),1],stride=[len(scales),1])
layer("ell_v","reshape",["ell_q"],shape=[len(qs)])
axis_targets=[(1,2,3),(1,2),(0,),(1,2),(3,4),(1,2),(0,1,2),(0,1,2),(6,7),(5,6,7)]
dense("axis_score","ell_v",contrast_rows(axis_targets,len(qs),.42))

# Whole-galaxy generative templates: smooth ellipsoids, thin disks, bulged disks, bars and ordinary disks.
# Each family is searched over scale and orientation; scores are relative across families.
families=7; MW=[]
for fam in range(families):
    for s in [1.8,2.8,4.2,6.3]:
        for a in angs:
            if fam==0: m=ellipse(s,1.0,a,.72)                                      # round smooth
            elif fam==1: m=ellipse(s,.64,a,.78)                                   # intermediate
            elif fam==2: m=ellipse(s,.34,a,.82)                                   # cigar
            elif fam==3: m=ellipse(s,.105,a,1.0)                                  # flat, no bulge
            elif fam==4: m=ellipse(s,.105,a,1.0)+.75*ellipse(.30*s+.35,.72,a,.8)   # flat + round bulge
            elif fam==5: m=.58*ellipse(s,.88,a,.85)+.72*ellipse(.48*s+.2,.12,a,1.)# round disk + bar
            else: m=ellipse(s,.82,a,.78)+.25*ellipse(.28*s+.3,.75,a,.8)            # ordinary disk
            MW.append(unit(m))
dense("model_d","lumn",MW)
layer("model_rs","reshape",["model_d"],shape=[families,4,len(angs)])
layer("model_m","maxpool",["model_rs"],size=[4,len(angs)],stride=[4,len(angs)])
layer("model_v","reshape",["model_m"],shape=[families])
model_targets=[(1,6),(1,6),(0,),(1,),(2,),(5,6),(6,),(6,),(3,),(4,)]
dense("model_score","model_v",contrast_rows(model_targets,families,.34))

# Log-polar m=2 matched filters detect spiral arms and estimate their pitch independently of rotation,
# handedness and apparent size.
pitches=[11.,18.,28.,40.]
spir_scales=[2.2,3.3,4.8,6.8]
phases=[math.pi*j/12 for j in range(12)]
SW=[]
for pitch in pitches:
    for s in spir_scales:
        env=np.exp(-.5*((R-1.05*s)/(.62*s+.2))**2)*(1-np.exp(-(R/.8)**2))
        for hand in [-1.,1.]:
            rate=hand/math.tan(math.radians(pitch))
            for ph in phases:
                pat=env*np.cos(2*(TH-rate*np.log(R+.55)-ph))
                SW.append(unit(pat))
dense("spir_d","lumn",SW)
layer("spir_rs","reshape",["spir_d"],shape=[len(pitches),len(spir_scales),24])
layer("spir_m","maxpool",["spir_rs"],size=[len(spir_scales),24],stride=[len(spir_scales),24])
layer("spir_v","reshape",["spir_m"],shape=[len(pitches)])
# Absolute arm evidence raises disk classes; pitch contrast separates tight and loose arms.
Sp=np.zeros((10,4),np.float32)
Sp[0]=[.025,.035,.055,.065]; Sp[1]=[.015,.025,.035,.035]
Sp[5]=[.06,.09,.09,.06]; Sp[6]=[.15,.17,.02,-.05]; Sp[7]=[-.04,.02,.17,.20]
Sp[2:5]=-.018; Sp[8:10]=-.025
dense("spir_score","spir_v",Sp)

# A double-core bank responds to two nuclei rather than a single central concentration.
PW=[]
for sig in [.55,.9]:
    for sep in [1.5,2.5,3.8]:
        for a in angs:
            dx=.5*sep*math.cos(a); dy=.5*sep*math.sin(a)
            pair=oriented_gauss(sig,sig,0,dx,dy)+oriented_gauss(sig,sig,0,-dx,-dy)
            single=oriented_gauss(math.sqrt(sig*sig+(sep*.5)**2),sig, a)
            PW.append(unit(pair-.72*single))
dense("pair_d","lumn",PW)
layer("pair_rs","reshape",["pair_d"],shape=[6,1,len(angs)])
layer("pair_m","maxpool",["pair_rs"],size=[1,len(angs)],stride=[1,len(angs)])
layer("pair_v","reshape",["pair_m"],shape=[6])
Pair=np.zeros((10,6),np.float32); Pair[1]=.016; Pair[0]=.006; Pair[5]=.003
Pair[2:5]=-.002; Pair[8:10]=-.001
dense("pair_score","pair_v",Pair)

# Rotational asymmetry: signed differences of opposite, formula-defined apertures, rectified both ways.
AW=[]
for rad in [2.5,4.5,7.0]:
    sig=.65+.10*rad
    for j in range(12):
        a=2*math.pi*j/12; cx=rad*math.cos(a); cy=rad*math.sin(a)
        d=oriented_gauss(sig,sig,0,cx,cy)-oriented_gauss(sig,sig,0,-cx,-cy)
        d=unit(d,False); AW.extend([d,-d])
dense("asym_d","lumn",AW)
layer("asym_r","relu",["asym_d"])
As=np.zeros((10,len(AW)),np.float32)
As[0]=.030/len(AW); As[1]=.023/len(AW); As[7]=.009/len(AW); As[5:7]=.004/len(AW)
As[2:5]=-.012/len(AW); As[8:10]=-.010/len(AW)
dense("asym_score","asym_r",As)

# Fixed zero-mean Gabor filters measure clumpiness and oriented texture in the central galaxy.
K=9; yy,xx=np.mgrid[0:K,0:K].astype(np.float64); xx-=K//2; yy-=K//2
GW=[]
for si,(sg,wl) in enumerate([(1.05,2.7),(1.65,4.3),(2.4,6.8)]):
    for j in range(8):
        a=math.pi*j/8; u=xx*np.cos(a)+yy*np.sin(a); v=-xx*np.sin(a)+yy*np.cos(a)
        g=np.exp(-.5*((u/sg)**2+(v/(.72*sg))**2))*np.cos(2*math.pi*u/wl)
        g=unit(g)
        for sign in [1,-1]: GW.append((sign*g)[None,:,:])
conv("gabor","lumn",np.asarray(GW),padding=4)
layer("gabor_r","relu",["gabor"])
# Sum both contrast polarities with a central Gaussian aperture.
TW=np.zeros((24,48*N*N),np.float32)
for si in range(3):
    aperture=np.exp(-.5*(R/(5.5+1.3*si))**2); aperture/=aperture.sum()
    for j in range(8):
        row=si*8+j
        for sign in range(2):
            ch=(si*8+j)*2+sign
            TW[row,ch*N*N:(ch+1)*N*N]=aperture.ravel()
dense("tex_d","gabor_r",TW)
layer("tex_rs","reshape",["tex_d"],shape=[3,1,8])
layer("tex_a","avgpool",["tex_rs"],size=[1,8],stride=[1,8])
layer("tex_av","reshape",["tex_a"],shape=[3])
Ta=np.zeros((10,3),np.float32)
Ta[0]=[.065,.050,.025]; Ta[1]=[.035,.035,.02]; Ta[5]=[.035,.045,.025]
Ta[6]=[.025,.04,.02]; Ta[7]=[.055,.05,.02]
Ta[2:5]=[-.025,-.02,-.008]; Ta[8:10]=[-.018,-.012,-.004]
dense("tex_ascore","tex_av",Ta)
layer("tex_x","maxpool",["tex_rs"],size=[1,8],stride=[1,8])
layer("tex_xv","reshape",["tex_x"],shape=[3])
Tx=np.zeros((10,3),np.float32)
Tx[4]=[.012,.025,.025]; Tx[5]=[.025,.035,.025]; Tx[8]=[.025,.045,.04]; Tx[9]=[.02,.04,.035]
Tx[2]=[-.015,-.018,-.012]
dense("tex_xscore","tex_xv",Tx)

# Raw-colour aperture expert. Blue outer light favours star-forming structure; red light favours smooth/bulged systems.
# p0 is normalized per channel, so this uses relative spatial colour contrast, not survey zero points.
CW=np.zeros((10,3*N*N),np.float32)
ann=np.exp(-.5*(R/5.0)**2)-.55*np.exp(-.5*(R/1.4)**2); ann=ann/(np.sum(np.abs(ann))+1e-9)
blue=np.zeros(3*N*N,np.float32); blue[2*N*N:3*N*N]=ann.ravel(); blue[0:N*N]=-ann.ravel()
red=-blue
for k in [0,1,5,6,7]: CW[k]+=.32*blue
for k in [2,3,4,9]: CW[k]+=.24*red
CW[8]+=.10*red
dense("color_score","p0",CW)

# Add fixed class offsets in a final identity readout.
layer("sum0","add",["axis_score","model_score","spir_score","pair_score","asym_score","tex_ascore","tex_xscore","color_score"])
# Fixed gain factors compensate differing analytic expert dynamic ranges.
GAIN=np.array([1.0,.8,.55,.75,2.5,.75,.7,.8,.5,.7],np.float32)
TIE=np.array([0.,0.,0.,.15,0.,.15,.15,0.,0.,.05],np.float32)
dense("scores","sum0",np.diag(GAIN),CAL*GAIN+TIE)

(OUT/"network.json").write_text(json.dumps({"layers":layers},separators=(",",":")))
np.savez(OUT/"weights.npz",**{k:np.asarray(v,np.float32) for k,v in weights.items()})
