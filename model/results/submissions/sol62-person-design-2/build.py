import sys, json
from pathlib import Path
import numpy as np

p = Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=True)
L=[]; W={}
def layer(name,op,inputs,**kw):
    L.append(dict(name=name,op=op,inputs=inputs,**kw)); return name
def conv(n,x,w,b,s=1,pad=0):
    w=np.asarray(w,np.float32)
    if w.ndim==2: w=w[:,:,None,None]
    W[n+'.W']=w; W[n+'.b']=np.asarray(b,np.float32)
    return layer(n,'conv',[x],out=len(w),kernel=list(w.shape[2:]),stride=s,padding=pad)
def rel(n,x): return layer(n,'relu',[x])
def pool(n,x,k,s,op='avgpool'):
    return layer(n,op,[x],size=[k,k],stride=[s,s])

# Opponent colors are bounded by a polyhedral skin locus. Include both
# brighter low-chroma and darker skin with the same luminance-relative tests.
a=rel('violations',conv('opponents','image',[
    [-1,1,0],[0,-1,1],[1,-1,0],[-.52,1,-.61],
    [1,0,-1],[-.299,-.587,-.114],[.299,.587,.114],
    [.299,.587,.114],[1,-1,0],[0,1,-1]],
    [.021,.002,-.24,-.015,-.39,.12,-.96,0,-.045,-.055],2))
b=rel('skin_and_light',conv('locus',a,[
    [-7,-6,-4,-3,-4,-6,-5,0,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [-7,-6,-4,-3,-4,-6,-5,0,2,1]], [1,0,.65]))
# Average before the spatial detectors, to soften JPEG and sampling noise.
b=pool('smooth',b,2,2)
features=[]
for k in [5,7,9,13,17,23]:
    y,x=np.mgrid[-k//2+1:k//2+1,-k//2+1:k//2+1]/(k//2)
    def g(cx,cy,sx,sy):
        z=np.exp(-.5*((x-cx)**2/sx**2+(y-cy)**2/sy**2)); return z/z.sum()
    weights=[]; biases=[]
    for dx in [-.18,0,.18]:
        core=g(dx,-.27,.22,.30)
        side=(g(dx-.57,-.25,.18,.32)+g(dx+.57,-.25,.18,.32))/2
        below=g(dx,.58,.36,.22)
        cheeks=(g(dx-.16,-.07,.12,.12)+g(dx+.16,-.07,.12,.12))/2
        eyes=(g(dx-.17,-.34,.09,.08)+g(dx+.17,-.34,.09,.08))/2
        brow=g(dx,-.48,.20,.12)
        mouth=g(dx,.09,.17,.08)
        chin=g(dx,.24,.18,.12)
        contrast=.62*cheeks+.38*brow-.77*eyes-.23*mouth
        for cs in [.60,.95]:
            weights.append([.63*core-.34*side-.25*below,cs*contrast+.24*(chin-below),.35*cheeks-.13*side-.08*below]); biases.append(-.27)
    # Only score complete windows: zero padding would otherwise look
    # like a dark surround and produce artificial face contrast.
    f=rel(f'heads{k}',conv(f'template{k}',b,weights,biases))
    features.append(pool(f'peak{k}',f,25-k,25-k,'maxpool'))
    features.append(pool(f'extent{k}',f,25-k,25-k))

# The same modest vote can come from larger exposed skin areas, not only faces.
raw=pool('color_coverage',b,24,24)
parts=[]
for i,f in enumerate(features):
    if i%2==0:
        # Pairwise reduction implements maximum over the pose channels.
        n=6
        while n>1:
            u=[]; v=[]; merge=[]
            for j in range(0,n-1,2):
                row=np.zeros(n); row[j]=1; row[j+1]=-1; u.append(row)
            d=rel(f'diff{i}_{n}',conv(f'delta{i}_{n}',f,u,[0]*len(u)))
            for j in range(n//2):
                row=np.zeros(n); row[2*j+1]=1; v.append(row)
            if n%2:
                row=np.zeros(n); row[-1]=1; v.append(row)
            base=conv(f'base{i}_{n}',f,v,[0]*len(v))
            if n%2:
                d=conv(f'pad{i}_{n}',d,np.vstack([np.eye(n//2),np.zeros((1,n//2))]),[0]*len(v))
            f=layer(f'max{i}_{n}','add',[d,base]); n=len(v)
        weight=[.21,.25,.26,.25,.21,.15][i//2]
        q=conv(f'vote{i}',f,[[weight]],[0])
    else:
        q=conv(f'vote{i}',f,[[.16]*6],[0])
    parts.append(q)
parts.append(conv('coverage_vote',raw,[[.12,0,.17]],[0]))
f=layer('evidence','add',parts)
W['scores.W']=np.array([[0],[1]],np.float32); W['scores.b']=np.array([0,-.245],np.float32)
layer('scores','dense',[f],out=2)
(p/'network.json').write_text(json.dumps(dict(layers=L),indent=2)+'\n')
np.savez(p/'weights.npz',**W)
