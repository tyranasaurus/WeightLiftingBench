import sys, json
from pathlib import Path
import numpy as np

# Each string describes one pen stroke on an abstract 0..9 grid.
FORMS = [
 ['46 19 07 04 21 60 92 95 78 46', '48 18 06 03 20 60 92 96 78 48', '49 19 06 02 30 71 93 97 69 49'],
 ['59 50', '18 76 70', '26 78 60', '16 59 58 50', '17 59 50/20 90'],
 ['18 39 77 86 64 32 10 59'.replace('59','90'), '08 29 69 88 86 64 32 10 90', '17 29 78 87 65 32 11 30 90', '18 49 88 86 53 11 30 90'],
 ['18 49 88 85 55 36 65 92 81 50 11', '09 79 47 35 65 93 82 60 10', '18 39 78 87 65 45 76 93 81 40 11'],
 ['79 36 05 94/77 70', '59 18 05 93/77 70', '68 37 15 45 75 70', '69 47 24 04 93/69 70', '69 47 25 04 74 69 70', '69 37 16 04 74 67 69 70'],
 ['89 29 16 46 76 94 92 70 40 11', '89 19 17 15 55 86 94 91 60 20', '89 39 19 16 46 84 92 71 40 10'],
 ['78 59 37 15 03 12 31 61 92 94 65 35 14', '89 58 35 13 11 40 80 92 84 54 24 13', '69 47 24 12 20 60 91 94 74 43 13'],
 ['09 99 76 53 40', '08 49 99 76 53 40/25 85', '19 89 87 66 54 32 30', '18 49 99 78 56 43 40'],
 ['55 27 28 49 79 88 77 55 23 12 21 50 81 92 83 55', '55 25 16 27 49 79 88 77 55 23 12 21 50 81 92 83 55', '54 24 16 27 59 89 97 74 43 12 21 50 81 92 83 54'],
 ['84 88 69 39 18 16 35 65 86 88/86 74 62 40', '78 59 28 16 34 64 86 88 76 65 63 60', '88 58 29 17 25 55 76 88 75 64 52 40', '84 87 78 49 28 16 25 55 85 86 73 51 20', '86 78 59 28 16 25 55 86 88/88 85 73 61 60']]

def path(s):
    p = np.array([[int(v[0]),-int(v[1])] for v in s.split()],float)
    q=np.pad(p,((1,1),(0,0)),mode='edge')
    t=np.linspace(0,1,24)[:,None]
    return np.concatenate([.5*(2*b+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t) for a,b,c,d in zip(q,q[1:],q[2:],q[3:])])

def main(dest):
    rng=np.random.default_rng(716)
    layers=[]; weights={}
    def layer(name,op,inputs,**kw):
        layers.append(dict(name=name,op=op,inputs=inputs,**kw)); return name
    def conv(name,inp,w,b):
        layer(name,'conv',[inp],out=len(b),kernel=[1,1],stride=1,padding=0)
        weights[name+'.W']=np.asarray(w,np.float32).reshape(len(b),-1,1,1)
        weights[name+'.b']=np.asarray(b,np.float32)
    # Two channels: clipped ink and its immediate spatial neighborhood.
    conv('clip_parts','image',[[1],[1]],[0,-.65])
    layer('positive','relu',['clip_parts'])
    conv('ink','positive',[[1/.65,-1/.65]],[0])
    layer('near','maxpool',['ink'],size=[3,3],stride=[1,1])
    # Pooling is cropped by one pixel along each edge, so use a 26x26 grid for this channel.
    y,x=np.mgrid[:28,:28]; grid=np.stack([x,y],axis=-1)
    exact=[]; near=[]; biases=[]
    count=2048
    for digit, forms in enumerate(FORMS):
        paths=[np.concatenate([path(stroke) for stroke in s.split('/')]) for s in forms]
        for j in range(count):
            p=paths[j%len(paths)].copy()
            p=(p-(p.max(0)+p.min(0))/2)/np.maximum(np.ptp(p,axis=0),.01)
            width=rng.uniform(9,18) if digit!=1 else rng.uniform(1,8) if j%len(paths)<3 else rng.uniform(6,14)
            p*= [width,rng.uniform(16,20)]
            p[:,0]+=rng.uniform(-.43,.43)*p[:,1]+rng.uniform(-.028,.028)*p[:,1]**2
            p[:,1]+=rng.uniform(-.23,.23)*p[:,0]
            # Center at the approximate center of mass of the stroked curve.
            step=np.linalg.norm(np.diff(p,axis=0),axis=1)
            step=np.where(step<3,step,0)
            p-=np.average((p[1:]+p[:-1])/2,axis=0,weights=step)
            p+=13.5+rng.uniform(-1.15,1.15,2)
            d=np.sqrt(np.min(np.sum((grid[:,:,None,:]-p[None,None,:,:])**2,axis=-1),axis=2))
            radius=rng.uniform(.65,1.7)
            # Penalize image ink far from the curve; reward coverage of the curve.
            penalty=np.clip((d-radius-.45)/3.8,0,1)
            target=np.clip(1.65-d,0,1)
            target/=target.sum()
            exact.append((.43*target-penalty/57).ravel())
            near.append((.72*target[1:-1,1:-1]).ravel())
            biases.append(-.015*abs(radius-1.15))
    for name,inp,w,b in [('match','ink',exact,biases),('cover','near',near,np.zeros(len(near)))]:
        layer(name,'dense',[inp],out=len(w)); weights[name+'.W']=np.asarray(w,np.float32); weights[name+'.b']=np.asarray(b,np.float32)
    layer('similarity','add',['match','cover'])
    layer('grouped','reshape',['similarity'],shape=[10,count,1])
    layer('best','maxpool',['grouped'],size=[count,1],stride=[count,1])
    layer('scores','reshape',['best'],shape=[10])
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    (dest/'network.json').write_text(json.dumps({'layers':layers},indent=2))
    np.savez(dest/'weights.npz',**weights)

if __name__=='__main__':
    main(sys.argv[1])
