"""Data-free builder for a hand-designed person / no-person network.

Usage:  python3 build.py OUT_DIR
Writes OUT_DIR/network.json and OUT_DIR/weights.npz.  Reads no files; every number
comes from the formulas and hand-chosen constants in this source.

DESIGN (reasoned from general knowledge of people in photos, not fitted to any data):
The cue for "person" is visible HUMAN SKIN (face + hands/arms).  Raw warm colour is
useless here (wood, brick, meat, bread, tan surfaces, warm light are everywhere), so skin
is made specific by two general properties that separate skin from reddish OBJECTS:
  (a) COLOUR  -- a blood-red tint: R/G ~1.3, unlike neutral beige/grey (R/G ~1.05); not too
                 orange or yellow; not near-black or near-white.  Brightness-invariant
                 *ratio* half-planes, so all skin tones (pale..dark) pass and beige/grey/
                 snow/green/sky/orange are rejected.
  (b) TEXTURE -- skin is SMOOTH while wood/brick/meat/bread are textured.  A 3x3 high-pass
                 measures local texture; a smooth gate keeps low-texture pixels.
Skin pixel = colour AND smooth.  A person shows a solid patch of such pixels, so the feature
is the fraction of the photo that is SOLID smooth-skin (6x6 density, clamped to a 0/1 "solid"
indicator so scattered specks and thin edges don't count).  Decision: person if that
fraction exceeds a hand threshold (~"a modestly visible face").

Primitive tricks (only conv/relu/avgpool/add/reshape/dense allowed):
  violation v_i = relu(linear) is 0 when a skin rule holds; skin = relu(1 - sum v_i).
  |x| = relu(x)+relu(-x);  AND(a,b)=relu(a+b-1);  clamp(x,0,1)=relu(x)-relu(x-1) (the two
  relu pieces are recombined with +1/-1 weights in a later linear layer).
"""
import sys, json, os
import numpy as np

SKIN_K    = 25.0
BETA      = 6.0
CLAMP_M   = 12.0
SOLID_T   = 0.40
OUT_SCALE = 60.0
THRESHOLD = 0.030

SKIN_COND = np.array([
    [ 1.0, -1.13,  0.0 , 0.0 ],   # R >= 1.13 G   (red tint vs neutral beige/grey)
    [ 0.0,  1.0 , -1.02, 0.0 ],   # G >= 1.02 B
    [-1.0,  1.85,  0.0 , 0.0 ],   # R <= 1.85 G   (not too orange / red)
    [ 0.0, -1.0 ,  1.5 , 0.0 ],   # G <= 1.5 B    (not too yellow: dried grass)
    [ 1.0,  0.0 ,  0.0 ,-0.20],   # R >= 0.20     (not near-black)
    [-1.0, -1.0 , -1.0 , 2.80],   # R+G+B <= 2.80 (not near-white)
])

def build():
    layers=[]; W={}
    def add(name,op,inputs,**kw):
        d={"name":name,"op":op,"inputs":inputs}; d.update(kw); layers.append(d); return name

    # (a) skin colour -> S
    k=SKIN_K; C=SKIN_COND.shape[0]
    vW=np.zeros((C,3,1,1)); vb=np.zeros(C)
    for i in range(C):
        vW[i,:,0,0]=-k*SKIN_COND[i,:3]; vb[i]=-k*SKIN_COND[i,3]
    add("vcond","conv",["image"],out=C,kernel=[1,1],stride=1,padding=0); W["vcond.W"],W["vcond.b"]=vW,vb
    add("vrelu","relu",["vcond"])
    add("skinlin","conv",["vrelu"],out=1,kernel=[1,1],stride=1,padding=0); W["skinlin.W"],W["skinlin.b"]=-np.ones((1,C,1,1)),np.array([1.0])
    add("skin","relu",["skinlin"])

    # (b) smoothness gate -> smooth
    add("Y","conv",["image"],out=1,kernel=[1,1],stride=1,padding=0)
    W["Y.W"]=np.array([0.299,0.587,0.114]).reshape(1,3,1,1); W["Y.b"]=np.zeros(1)
    hp=-np.ones((3,3))/9.0; hp[1,1]+=1.0
    add("hf","conv",["Y"],out=2,kernel=[3,3],stride=1,padding=1)
    W["hf.W"]=np.stack([hp,-hp]).reshape(2,1,3,3); W["hf.b"]=np.zeros(2)
    add("hfr","relu",["hf"])
    add("smlin","conv",["hfr"],out=1,kernel=[1,1],stride=1,padding=0)
    W["smlin.W"]=(-BETA*np.ones((1,2,1,1))); W["smlin.b"]=np.array([1.0])
    add("smooth","relu",["smlin"])

    # skin AND smooth
    add("ssl","add",["skin","smooth"])
    add("sslin","conv",["ssl"],out=1,kernel=[1,1],stride=1,padding=0)
    W["sslin.W"]=np.ones((1,1,1,1)); W["sslin.b"]=np.array([-1.0])
    add("ss","relu",["sslin"])

    # solid smooth-skin coverage fraction
    add("dens","avgpool",["ss"],size=[6,6],stride=[6,6])
    M=CLAMP_M
    cW=np.zeros((2,1,1,1)); cb=np.zeros(2)
    cW[0,0,0,0]=M; cb[0]=-M*SOLID_T
    cW[1,0,0,0]=M; cb[1]=-M*SOLID_T-1.0
    add("clip","conv",["dens"],out=2,kernel=[1,1],stride=1,padding=0); W["clip.W"],W["clip.b"]=cW,cb
    add("clipr","relu",["clip"])
    add("gap","avgpool",["clipr"],size=[16,16],stride=[16,16])
    add("flat","reshape",["gap"],shape=[2])

    dW=np.array([[0.0,0.0],[OUT_SCALE,-OUT_SCALE]])
    db=np.array([OUT_SCALE*THRESHOLD, 0.0])
    add("out","dense",["flat"],out=2); W["out.W"],W["out.b"]=dW,db
    return {"layers":layers}, W


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    net, W = build()
    with open(os.path.join(out_dir, "network.json"), "w") as f:
        json.dump(net, f)
    np.savez(os.path.join(out_dir, "weights.npz"),
             **{k: v.astype(np.float32) for k, v in W.items()})


if __name__ == "__main__":
    main()
