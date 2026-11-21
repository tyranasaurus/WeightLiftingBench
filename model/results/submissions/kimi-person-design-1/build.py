"""Build a hand-designed, multi-scale person detector.  No input files are read."""
import sys, json, math
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
layers, weights = [], {}

def add_layer(name, op, inputs, **settings):
    d = {"name": name, "op": op, "inputs": inputs}
    d.update(settings)
    layers.append(d)

def conv(name, inp, W, b=None, stride=1, padding=0):
    W = np.asarray(W, dtype=np.float32)
    if b is None: b = np.zeros(W.shape[0], np.float32)
    weights[name+".W"], weights[name+".b"] = W, np.asarray(b, np.float32)
    add_layer(name, "conv", [inp], out=W.shape[0], kernel=[W.shape[2], W.shape[3]], stride=stride, padding=padding)

def dense(name, inp, W, b=None):
    W = np.asarray(W, dtype=np.float32)
    if b is None: b = np.zeros(W.shape[0], np.float32)
    weights[name+".W"], weights[name+".b"] = W, np.asarray(b, np.float32)
    add_layer(name, "dense", [inp], out=W.shape[0])

def pool(name, op, inp, size, stride):
    add_layer(name, op, [inp], size=[size,size], stride=[stride,stride])

def relu(name, inp): add_layer(name, "relu", [inp])
def reshape(name, inp, n): add_layer(name, "reshape", [inp], shape=[n])
def add(name, ins): add_layer(name, "add", ins)

def gauss(k, sigma):
    z = np.arange(k, dtype=np.float64) - (k-1)/2
    x,y = np.meshgrid(z,z)
    g = np.exp(-(x*x+y*y)/(2*sigma*sigma))
    return g/g.sum()

# Illumination-normalized intensity for shape measurements.
W = np.zeros((1,3,1,1)); W[0,:,0,0] = [0.299,0.587,0.114]
conv("gray", "image", W)
add_layer("gray_norm", "norm", ["gray"])

C = 28
# Channels 2:18 are signed first Gaussian derivatives in 16 directions.
k=5; g=gauss(k,0.95); z=np.arange(k)-(k-1)/2; x,y=np.meshgrid(z,z)
W = np.zeros((C,1,k,k))
for j in range(16):
    a=j*math.pi/8
    d=-(x*math.cos(a)+y*math.sin(a))/(0.95**2)*g
    d *= 1.5/max(d.__abs__().sum(),1e-12)
    W[2+j,0] = d
# Two center/surround channels are useful for tiny heads.
lo=(x*x+y*y-2*0.95**2)/(0.95**4)*g
lo/=max(np.abs(lo).sum(),1e-12)
W[26,0],W[27,0] = lo,-lo
conv("edge_affine", "gray_norm", W, stride=2, padding=2)
relu("edge_relu", "edge_affine")

# Raw colour and brightness, smoothed while reducing to 48 x 48.
k=3; g=gauss(k,0.65)
W=np.zeros((C,3,k,k)); b=np.zeros(C)
for ch,co in [(18,[0.299,0.587,0.114]),(20,[1,-1,0]),(21,[-1,1,0]),
              (22,[0,1,-1]),(23,[0,-1,1]),(24,[1,0,-1]),(25,[-1,0,1])]:
    for q in range(3): W[ch,q]=co[q]*g
for q,co in enumerate([0.299,0.587,0.114]): W[19,q]=-co*g
b[19]=1
conv("colour_affine", "image", W, b, stride=2, padding=1)
relu("colour_relu", "colour_affine")

# Broad and strict skin likelihoods from the usual YCbCr colour box.
# Each input to the second 1x1 convolution is a nonnegative hinge violation.
W=np.zeros((12,3,3,3)); b=np.zeros(12)
g=gauss(3,0.65)
cb=np.array([-0.168736,-0.331264,0.5]); cr=np.array([0.5,-0.418688,-0.081312]); yy=np.array([0.299,0.587,0.114])
def put_hinge(row, rgb, constant):
    for q in range(3): W[row,q]=rgb[q]*g
    b[row]=constant
put_hinge(0,cb,0.5-0.50)       # Cb above strict upper limit
put_hinge(1,-cb,0.28-0.5)      # Cb below strict lower limit
put_hinge(2,cr,0.5-0.72)       # Cr above strict upper limit
put_hinge(3,-cr,0.515-0.5)     # Cr below strict lower limit
put_hinge(4,-yy,0.08)
put_hinge(5,yy,-0.995)
put_hinge(6,cb,0.5-0.54)
put_hinge(7,-cb,0.22-0.5)
put_hinge(8,cr,0.5-0.78)
put_hinge(9,-cr,0.50-0.5)
put_hinge(10,-yy,0.035)
put_hinge(11,yy,-1.0)
conv("skin_hinges", "image", W, b, stride=2, padding=1)
relu("skin_hinges_relu", "skin_hinges")
W=np.zeros((2,12,1,1))
W[0,:,0,0]=[-10,-10,-8,-10,-1.5,-2,0,0,0,0,0,0]
W[1,:,0,0]=[0,0,0,0,0,0,-5,-5,-5,-5,-1,-1]
conv("skin_affine", "skin_hinges_relu", W, [1.0,1.0])
relu("skin_likelihood", "skin_affine")
W=np.zeros((C,2,1,1)); W[0,0,0,0]=1; W[1,1,0,0]=1
conv("skin_pack", "skin_likelihood", W)
add("features", ["edge_relu","colour_relu","skin_pack"])
pool("features24", "avgpool", "features", 2,2)
pool("features12", "avgpool", "features24", 2,2)

# Rasterize generic geometric sketches into unsigned oriented-edge templates.
def smooth2(a, sigma):
    if sigma<=0: return a
    r=max(1,int(math.ceil(2.5*sigma))); z=np.arange(-r,r+1)
    g=np.exp(-z*z/(2*sigma*sigma)); g/=g.sum()
    out=np.zeros_like(a)
    for c in range(a.shape[0]):
        tmp=np.zeros((a.shape[1]+2*r,a.shape[2]))
        tmp[r:-r]=a[c]
        a[c]=np.sum(tmp[np.arange(a.shape[1])[:,None]+np.arange(2*r+1)]*g[None,:,None],axis=1)
        tmp=np.zeros((a.shape[1],a.shape[2]+2*r)); tmp[:,r:-r]=a[c]
        a[c]=np.sum(tmp[:,np.arange(a.shape[2])[:,None]+np.arange(2*r+1)]*g[None,None,:],axis=2)
    return a

def ellipse(cx,cy,rx,ry,n=48):
    return [(cx+rx*math.cos(2*math.pi*i/n),cy+ry*math.sin(2*math.pi*i/n)) for i in range(n+1)]

class Sketch:
    def __init__(self,k):
        self.k=k; self.e=np.zeros((8,k,k)); self.skin=np.zeros((k,k)); self.dark=np.zeros((k,k))
    def splat(self,arr,p,v):
        k=self.k; xx=(p[0]+1)*(k-1)/2; yy=(p[1]+1)*(k-1)/2
        ix=int(math.floor(xx)); iy=int(math.floor(yy)); fx=xx-ix; fy=yy-iy
        for dy in [0,1]:
            for dx in [0,1]:
                j=iy+dy; i=ix+dx
                if 0<=j<k and 0<=i<k: arr[j,i]+=v*(fy if dy else 1-fy)*(fx if dx else 1-fx)
    def stroke(self,points,weight=1.0):
        pts=[]
        for a,b in zip(points[:-1],points[1:]):
            length=math.hypot(b[0]-a[0],b[1]-a[1]); n=max(2,int(length*self.k*2)+1)
            for t in np.linspace(0,1,n,endpoint=False): pts.append((a[0]*(1-t)+b[0]*t,a[1]*(1-t)+b[1]*t))
        pts.append(points[-1])
        for a,b in zip(pts[:-1],pts[1:]):
            dx=b[0]-a[0]; dy=b[1]-a[1]
            if dx*dx+dy*dy<1e-12: continue
            # The gradient is normal to the drawn boundary.  Orientation is unsigned.
            ang=(math.atan2(dy,dx)+math.pi/2)%math.pi
            u=ang/(math.pi/8); j=int(math.floor(u))%8; f=u-math.floor(u)
            self.splat(self.e[j],a,weight/len(pts)*(1-f))
            self.splat(self.e[(j+1)%8],a,weight/len(pts)*f)
    def region(self,kind,cx,cy,rx,ry,value):
        z=np.arange(self.k)-(self.k-1)/2; x,y=np.meshgrid(z,z)
        x=x/((self.k-1)/2); y=y/((self.k-1)/2)
        m=np.clip(1-((x-cx)/rx)**2-((y-cy)/ry)**2,0,1)
        if m.sum()>0: m/=m.sum()
        a=self.skin if kind=="skin" else self.dark
        a+=value*m
    def weights(self,skin_gain=0.0,dark_gain=0.0,clutter=0.2):
        e=smooth2(self.e,self.k/20)
        if e.sum()>0: e/=e.sum()
        W=np.zeros((C,self.k,self.k)); pos=0.0
        for j in range(8):
            W[2+j]=e[j]; W[2+j+8]=e[j]
            pos+=2*e[j].sum()
        W[2:18]-=clutter/(16*self.k*self.k)
        W[0]=skin_gain*self.skin
        W[19]=dark_gain*self.dark
        return W

# Each entry is a generic pose, not an image-derived template.
def face_sketch(k,aspect=1.0,skin=True,profile=False):
    s=Sketch(k); rx=0.64*aspect; ry=0.82
    s.stroke(ellipse(0,0,rx,ry),1.0)
    if not profile:
        s.stroke([(-0.42*aspect,-0.16),(-0.12*aspect,-0.16)],0.35)
        s.stroke([(0.12*aspect,-0.16),(0.42*aspect,-0.16)],0.35)
        s.stroke([(-0.22,0.38),(0,0.42),(0.22,0.38)],0.18)
    else:
        s.stroke([(0.1,-0.12),(0.4,-0.12)],0.3)
        s.stroke([(0.5,0.05),(0.62,0.18),(0.42,0.22)],0.25)
    if skin:
        s.region("skin",0,0.03,rx*0.88,ry*0.88,1.0)
        s.region("skin",0,0,rx*1.35,ry*1.25,-0.80)
        s.region("dark",0,-0.57,rx*0.8,0.28,0.22)
        s.region("dark",-0.26*aspect,-0.15,0.16,0.09,0.18)
        s.region("dark",0.26*aspect,-0.15,0.16,0.09,0.18)
    return s

def shoulders_sketch(k,skin=True,wide=1.0):
    s=Sketch(k)
    s.stroke(ellipse(0,-0.34,0.31,0.36),1.0)
    s.stroke([(-0.18,-0.04),(-0.48*wide,0.13),(-0.84*wide,0.45),(-0.91*wide,0.87)],0.8)
    s.stroke([(0.18,-0.04),(0.48*wide,0.13),(0.84*wide,0.45),(0.91*wide,0.87)],0.8)
    s.stroke([(-0.48*wide,0.15),(0,0.29),(0.48*wide,0.15)],0.28)
    if skin:
        s.region("skin",0,-0.32,0.27,0.31,0.85)
        s.region("skin",0,-0.32,0.52,0.55,-0.18)
        s.region("dark",0,-0.57,0.26,0.15,0.15)
    return s

def body_sketch(k,pose="stand",wide=1.0):
    s=Sketch(k)
    s.stroke(ellipse(0,-0.70,0.16,0.18),0.8)
    if pose=="stand":
        lines=[([(-.10,-.49),(-.32,-.37),(-.25,.16)],1),
               ([(.10,-.49),(.32,-.37),(.25,.16)],1),
               ([(-.31,-.34),(-.46,.05),(-.40,.36)],.65),
               ([(.31,-.34),(.46,.05),(.40,.36)],.65),
               ([(-.25,.14),(-.20,.51),(-.24,.91)],.9),
               ([(.25,.14),(.20,.51),(.24,.91)],.9),
               ([(-.03,.17),(-.04,.52),(-.06,.88)],.55),
               ([(.03,.17),(.04,.52),(.06,.88)],.55)]
    elif pose=="walk":
        lines=[([(-.09,-.49),(-.29,-.34),(-.20,.15)],1),
               ([(.09,-.49),(.29,-.34),(.20,.15)],1),
               ([(-.27,-.32),(-.52,-.02),(-.68,.17)],.65),
               ([(.27,-.32),(.48,-.05),(.32,.28)],.65),
               ([(-.18,.12),(-.34,.48),(-.65,.85)],.9),
               ([(.17,.12),(.34,.47),(.62,.81)],.9),
               ([(0,.15),(.02,.49),(-.19,.91)],.7)]
    elif pose=="spread":
        lines=[([(-.09,-.49),(-.27,-.34),(-.22,.17)],1),
               ([(.09,-.49),(.27,-.34),(.22,.17)],1),
               ([(-.25,-.34),(-.57,-.43),(-.88,-.58)],.85),
               ([(.25,-.34),(.57,-.43),(.88,-.58)],.85),
               ([(-.20,.14),(-.39,.48),(-.60,.88)],.9),
               ([(.20,.14),(.39,.48),(.60,.88)],.9)]
    elif pose=="sit":
        lines=[([(-.10,-.49),(-.35,-.30),(-.28,.17)],1),
               ([(.10,-.49),(.35,-.30),(.28,.17)],1),
               ([(-.30,-.25),(-.52,.03),(-.17,.25)],.6),
               ([(.30,-.25),(.52,.03),(.17,.25)],.6),
               ([(-.25,.15),(-.48,.33),(-.83,.33)],.9),
               ([(.20,.17),(.48,.32),(.82,.32)],.9),
               ([(-.79,.34),(-.76,.67),(-.82,.91)],.8),
               ([(.78,.34),(.72,.67),(.78,.91)],.8)]
    else: # compact upper body, including people cut off by the photograph
        return shoulders_sketch(k,True,wide)
    for pts,w in lines:
        s.stroke([(x*wide,y) for x,y in pts],w)
    s.region("skin",0,-0.70,0.14,0.16,0.28)
    s.region("skin",0,-0.70,0.28,0.29,-0.07)
    return s

# Descriptions are shared at all scales so detections can be maximized over scale.
families={
 "face5":(5,["face5"],48,1),
 "head9":(9,["head9"],48,2),
 "body9":(9,["body9"],48,2),
 "body17":(17,["body17"],48,2)
}
sketches={}
sketches["face5"]=[face_sketch(5,a,True,False) for a in [0.78,1.0,1.22]]+[face_sketch(5,a,False,False) for a in [0.82,1.08,1.28]]
sketches["head9"]=[face_sketch(9,.9,True,False),face_sketch(9,1.15,True,False),face_sketch(9,1.0,True,True),
                    shoulders_sketch(9,True,.85),shoulders_sketch(9,True,1.15),
                    shoulders_sketch(9,False,.8),shoulders_sketch(9,False,1.0),shoulders_sketch(9,False,1.25)]
sketches["body9"]=[body_sketch(9,p,w) for p,w in [("stand",.8),("stand",1.15),("walk",.9),("walk",1.2),("spread",.85),("spread",1.15),("sit",1.0),("upper",1.0)]]
sketches["body17"]=[body_sketch(17,p,w) for p,w in [("stand",.75),("stand",1.0),("stand",1.3),("walk",.85),("walk",1.2),("spread",.9),("sit",1.0),("upper",1.15)]]
params={
 "face5":dict(skin=.95,dark=.65,clutter=1.6,bias=[.46,.46,.46,.20,.20,.20],thr=.025,weight=[1.15]*6),
 "head9":dict(skin=.70,dark=.45,clutter=1.5,bias=[.40,.40,.38,.32,.32,.20,.20,.20],thr=.018,weight=[1.0]*8),
 "body9":dict(skin=.65,dark=0,clutter=1.4,bias=[.17,.17,.18,.18,.18,.18,.20,.20],thr=.012,weight=[.82]*8),
 "body17":dict(skin=.65,dark=0,clutter=1.4,bias=[.18,.18,.18,.19,.19,.19,.21,.20],thr=.012,weight=[.86]*8)
}

# Exact per-channel maximum over an arbitrary list of vectors.
def maximum_vectors(name,vecs,n):
    cur=vecs[0]
    for q,v in enumerate(vecs[1:]):
        a=np.eye(n,dtype=np.float32)
        dense(name+"_old"+str(q),cur,a)
        dense(name+"_newneg"+str(q),v,-a)
        add(name+"_difference"+str(q),[name+"_old"+str(q),name+"_newneg"+str(q)])
        relu(name+"_positive"+str(q),name+"_difference"+str(q))
        dense(name+"_new"+str(q),v,a)
        add(name+"_maximum"+str(q),[name+"_positive"+str(q),name+"_new"+str(q)])
        cur=name+"_maximum"+str(q)
    return cur

family_scores=[]
for fi,(family,(kk,_,_,_)) in enumerate(families.items()):
    ss=sketches[family]; n=len(ss); pp=params[family]
    W=np.stack([s.weights(pp["skin"],pp["dark"],pp["clutter"]) for s in ss])
    maxima=[]; means=[]
    for si,(src,size,stride) in enumerate([("features",48,2 if family!="face5" else 1),("features24",24,1),("features12",12,1)]):
        nm=family+"_"+str(size)
        conv(nm+"_raw",src,W,-np.asarray(pp["bias"]),stride=stride,padding=kk//2)
        relu(nm+"_relu",nm+"_raw")
        outsize=size//stride
        pool(nm+"_maxpool","maxpool",nm+"_relu",outsize,1)
        reshape(nm+"_maxvec",nm+"_maxpool",n)
        maxima.append(nm+"_maxvec")
        pool(nm+"_meanpool","avgpool",nm+"_relu",outsize,1)
        reshape(nm+"_meanvec",nm+"_meanpool",n)
        means.append(nm+"_meanvec")
    mv=maximum_vectors(family+"_scale",maxima,n)
    dense(family+"_threshold",mv,np.eye(n),-np.full(n,pp["thr"]))
    relu(family+"_evidence",family+"_threshold")
    # Take the strongest pose rather than adding all the poses that weakly match.
    pose_vectors=[]
    for j,wj in enumerate(pp["weight"]):
        Wo=np.zeros((1,n)); Wo[0,j]=wj
        dense(family+"_pose"+str(j),family+"_evidence",Wo)
        pose_vectors.append(family+"_pose"+str(j))
    family_max=maximum_vectors(family+"_posemax",pose_vectors,1)
    # A small mean term rewards coherent detections but cannot dominate the max.
    mean_score_names=[]
    for si,v in enumerate(means):
        Wo=np.zeros((1,n)); Wo[0]=np.asarray(pp["weight"])*0.25/len(means)
        dense(family+"_meanscore"+str(si),v,Wo)
        mean_score_names.append(family+"_meanscore"+str(si))
    add(family+"_meanscore",mean_score_names)
    add(family+"_score",[family_max,family+"_meanscore"])
    family_scores.append(family+"_score")

# Combine shape with explicit skin-area evidence.  The local statistic responds to
# a compact face/hand-sized region, while the global one also helps close portraits.
best=maximum_vectors("all_detectors",family_scores,1)
dense("shape_score",best,np.array([[0.0],[1.0]]),[0.0,-0.65])
pool("skin_area_pool","avgpool","skin_likelihood",48,1)
W=np.zeros((2,2)); W[1]=[2.0,0.5]
dense("skin_area_score","skin_area_pool",W,[0.0,-0.05])
pool("skin_neighbourhood","avgpool","skin_likelihood",5,1)
pool("skin_local_pool","maxpool","skin_neighbourhood",44,1)
W=np.zeros((2,2)); W[1]=[0.5,0.0]
dense("skin_local_score","skin_local_pool",W)
add("scores",["shape_score","skin_area_score","skin_local_score"])

(OUT/"network.json").write_text(json.dumps({"layers":layers},separators=(",",":")))
np.savez(OUT/"weights.npz",**weights)
print("wrote",len(layers),"layers and",sum(v.size for v in weights.values()),"weights")
