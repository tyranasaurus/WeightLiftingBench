import json
import sys
from pathlib import Path

import numpy as np

# All digit geometry below is typed as idealized pen paths in a 10 by 16
# writing box.  Nothing in this program reads an image or any other file.
PATHS = {
    0: [
        [[(5,0),(8,1),(9.5,5),(9.3,11),(7.5,15),(5,16),(2.2,15),(0.7,11),(0.5,5),(2,1),(5,0)]],
        [[(5,0),(7.8,0.8),(9,4),(9,11),(7.5,15),(4.8,16),(2,14.8),(0.8,11),(1,4),(2.5,1),(5,0)]],
        [[(4.5,0),(7.5,1),(9,5),(8.8,12),(6.5,15.5),(3.5,15.8),(1,13),(0.5,7),(1.5,2),(4.5,0)]],
    ],
    1: [
        [[(5,0),(5,16)]],
        [[(2,3),(5,0),(5,16)]],
        [[(2,3),(5,0),(5,16)],[(2,16),(8,16)]],
        [[(1,4),(4.8,0),(5.3,16)],[(2,16),(8,16)]],
        [[(3,2),(5,0),(4.5,16)]],
        [[(5,0),(5,16)],[(2.5,15.8),(7.5,16)]],
    ],
    2: [
        [[(1,4),(2,1.5),(5,0.5),(8,1.5),(9,4),(8,7),(5,10),(1.5,15.5),(9,15.5)]],
        [[(1,3),(3,0.5),(7,0.5),(9,3),(9,6),(1,15),(9,15)]],
        [[(1,4),(2.5,1),(5.5,0.5),(8.5,2),(9,5),(7.5,8),(4,11),(1,14),(1.2,15.5),(4,15),(9,15.5)]],
        [[(0.8,3),(2.5,0.8),(6,0.5),(8.8,2),(9,5),(7,8),(3,12),(1,15.5)],[(1,15.5),(4,14.8),(9,15.5)]],
        [[(1,2.5),(4,0.5),(7.5,1),(9,3.5),(8.5,6),(5,9),(1,13),(1,15.5),(9,15.5)]],
    ],
    3: [
        [[(1,1.5),(4,0.5),(7,1),(8.5,3.5),(7.5,7),(4.5,8),(7.5,8.5),(9,11),(8,14.5),(5,16),(1.5,15)]],
        [[(1,1),(8,1),(5,7),(7.5,7.5),(9,10.5),(8,14),(5,16),(1,15)]],
        [[(1,2),(3,0.5),(6.5,0.8),(8.5,3),(8,6),(5,8)],[(5,8),(8,8.5),(9,11.5),(7.5,15),(4,16),(1,14.5)]],
        [[(1,1),(5,0.5),(8,2),(8.5,5),(6.5,7.5),(4,8),(7,8),(9,10),(8.5,14),(5.5,16),(1.5,15)]],
    ],
    4: [
        [[(7,16),(7,0),(1,10),(9,10)]],
        [[(7,0),(7,16)],[(7,0),(1,10),(9,10)]],
        [[(7,0),(7,16)],[(6,0),(1,9),(1,11),(9,11)]],
        [[(7,0),(7,16)],[(6.5,0),(2,8),(2,10),(9,10)]],
        [[(3,0),(3,9),(1,11),(9,11)],[(7,0),(7,16)]],
        [[(7,16),(7,0)],[(6,1),(2,7),(1,10),(2,11),(9,11)]],
    ],
    5: [
        [[(8.5,1),(2,1),(1.5,7),(4,5.5),(7,6),(9,8.5),(8.5,12.5),(6,15.5),(2.5,15.5),(1,14)]],
        [[(8,0.8),(2,1),(1.5,7),(3.5,6),(7,6),(9,8),(9,12),(7,15),(3.5,16),(1,14.5)]],
        [[(2,1),(8,1)],[(2,1),(1.5,7),(4,5.5),(7,6),(9,9),(8,13),(5,16),(1.5,15)]],
        [[(8.5,1),(3,0.5),(1.8,2),(1.5,7),(4,6),(7,6.5),(8.8,9),(8.5,13),(5.5,16),(2,15),(1,13.5)]],
        [[(8,1),(2,1),(1.5,6.5),(4,6),(7,6.5),(9,9),(8.5,13),(6,15.5),(2,16),(1,14)]],
    ],
    6: [
        [[(8,1),(5,0.5),(2.5,3),(1,7),(1.2,12),(3,15),(6,16),(8.5,14),(9,10.5),(7.5,7.5),(4.5,7),(1.2,9)]],
        [[(8,1),(5,0.5),(2,3),(1,8),(1.5,13),(4,15.5),(7,15.5),(9,13),(8.5,9.5),(6,7.5),(3.5,8),(1.2,10)]],
        [[(7.5,0.5),(4.5,1),(2,4),(1,9),(1.5,13),(4,15.5),(7,15),(8.8,12.5),(8,9),(5.5,7.5),(2.5,8.5),(1.2,11)]],
        [[(8,1),(5,0.5),(2.5,2.5),(1.2,6),(1,11),(2.5,14.5),(5.5,16),(8,14.5),(9,11),(7.5,8),(4,7.5),(1.2,9.5)]],
    ],
    7: [
        [[(1,1),(9,1),(4,16)]],
        [[(1,2),(2,1),(9,1),(4,16)]],
        [[(1,1),(9,1),(7,5),(5,10),(4.5,16)]],
        [[(1,1),(9,1),(4,16)],[(3,8),(7,8)]],
        [[(1,2),(2,1),(9,1),(6,6),(4.5,11),(4,16)],[(2.5,8),(7,8)]],
        [[(1,1),(1,3)],[(1,1),(9,1),(4,16)]],
        [[(0.5,2),(3,0.8),(9,1),(6,6),(5,11),(5.5,16)]],
    ],
    8: [
        [[(5,8),(2.5,6.5),(1.5,3.5),(3,1),(5.5,0.5),(8,2),(8.5,4.5),(7,7),(5,8),(2.5,9),(1.2,12),(2,15),(5,16),(8,15),(9,12),(8,9),(5,8)]],
        [[(5,7.5),(2.5,6),(1.5,3),(3,0.8),(6,0.5),(8.5,2.5),(8,5.5),(5,7.5),(2.5,9),(1.2,12),(2,15),(5,16),(8,15),(9,12),(8,9),(5,7.5)]],
        [[(5,0.5),(2.5,1),(1.5,3.5),(2.5,6),(5,7.5),(7.5,6),(8.5,3.5),(7.5,1),(5,0.5)],[(5,7.5),(2,9),(1,12),(2,15),(5,16),(8,15),(9,12),(8,9),(5,7.5)]],
        [[(4.5,0.5),(2,2),(1.5,4.5),(3,7),(5,8),(2,10),(1.5,13),(3,15.5),(6,16),(8.5,14),(9,11),(7,8.5),(5,8),(7,6),(8.5,3.5),(7,1),(4.5,0.5)]],
    ],
    9: [
        [[(8.5,7),(6,8.5),(3,8),(1.2,5.5),(1.5,2.5),(4,0.5),(7,1),(8.8,3.5),(9,8),(8,13),(6,16),(3,16)]],
        [[(8.5,7),(5.5,8.5),(2.5,7.5),(1.2,5),(2,2),(4.5,0.5),(7.5,1),(9,3.5),(8.5,9),(8.5,16)]],
        [[(8.5,6),(6,8),(3,7.5),(1.5,5),(1.5,2.5),(4,0.5),(7,1),(8.5,3),(8.5,16)],[(8.5,11),(7,15),(4,16)]],
        [[(8.5,7),(6,8.5),(3,8),(1.2,5.5),(1.5,2.5),(4,0.5),(7,1),(8.8,3.5),(8.5,9),(7,14),(4,16),(2,16)]],
        [[(8.5,6),(5.5,8),(2.5,7),(1.2,4.5),(2,1.5),(5,0.5),(8,2),(9,5),(8.5,10),(8.5,16)]],
    ],
}


def sample_path(points, rng, smooth=True):
    p = np.asarray(points, dtype=np.float64)
    # Small independent pen-control perturbations, not image-derived variation.
    p = p + rng.normal(0.0, 0.18, p.shape)
    p[0] = 0.65 * p[0] + 0.35 * np.asarray(points[0])
    p[-1] = 0.65 * p[-1] + 0.35 * np.asarray(points[-1])
    if len(p) == 2 or not smooth:
        ts = np.linspace(0, 1, max(3, int(4 * np.linalg.norm(p[1] - p[0]))))
        return p[0][None] * (1-ts[:, None]) + p[1][None] * ts[:, None]
    q = np.vstack([p[0], p, p[-1]])
    out = []
    for j in range(1, len(q)-2):
        a,b,c,d = q[j-1:j+3]
        for t in np.linspace(0, 1, 6, endpoint=False):
            out.append(0.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
    out.append(p[-1])
    return np.asarray(out)


def make_template(digit, index, rng):
    variants = PATHS[digit]
    strokes = variants[index % len(variants)]
    paths = [sample_path(s, rng) for s in strokes]
    allp = np.concatenate(paths)
    # Natural independent changes in proportion, slant, rotation and smooth warp.
    sx = rng.uniform(0.90, 1.20) * (1.0,1.0,1.25,1.20,1.35,1.25,1.20,1.25,1.20,1.15)[digit]
    sy = rng.uniform(0.90, 1.10)
    slant = rng.uniform(-0.22, 0.24)
    angle = rng.uniform(-0.075, 0.075)
    wavex = rng.uniform(-0.32, 0.32)
    wavey = rng.uniform(-0.20, 0.20)
    phase = rng.uniform(0, 2*np.pi)
    warped = []
    for p in paths:
        x = p[:,0] * sx
        y = p[:,1] * sy
        x = x - slant * (y-8) + wavex*np.sin(y*np.pi/16 + phase)
        y = y + wavey*np.sin(x*np.pi/10 + phase)
        xr = x*np.cos(angle) - (y-8)*np.sin(angle)
        yr = x*np.sin(angle) + (y-8)*np.cos(angle) + 8
        warped.append(np.stack([xr,yr],axis=1))
    allp = np.concatenate(warped)
    lo, hi = allp.min(0), allp.max(0)
    scale = 20.0 / max(hi-lo) * rng.uniform(0.88, 1.02)
    mid = (lo+hi)/2
    paths = [(p-mid)*scale + 13.5 for p in warped]
    width = rng.uniform(1.00, 1.80)
    gain = rng.uniform(0.82, 1.0)
    yy,xx = np.mgrid[0:28,0:28].astype(np.float64)
    shift = np.zeros(2)
    image = None
    for iteration in range(2):
        dist = np.full((28,28), 100.0)
        for p in paths:
            p = p + shift
            a = p[:-1]
            ab = p[1:] - a
            den = np.sum(ab*ab,axis=1) + 1e-12
            for start,vec,dd in zip(a,ab,den):
                rx = xx-start[0]; ry = yy-start[1]
                t = np.clip((rx*vec[0]+ry*vec[1])/dd,0,1)
                d2 = (rx-t*vec[0])**2 + (ry-t*vec[1])**2
                dist = np.minimum(dist,d2)
        image = gain*np.clip(1.15-np.sqrt(dist)/width,0,1)
        if iteration == 0:
            total = image.sum()
            if total > 0:
                cy,cx = (image[:,:,None]*np.stack([yy,xx],axis=2)).sum((0,1))/total
                shift += np.array([13.5-cx,13.5-cy])
    return image.astype(np.float32)


def make_part_template(digit,index,rng,parts=4):
    variants=PATHS[digit]
    paths=[sample_path(s,rng) for s in variants[index%len(variants)]]
    sx=rng.uniform(0.90,1.20)*(1.0,1.0,1.25,1.20,1.35,1.25,1.20,1.25,1.20,1.15)[digit]
    sy=rng.uniform(0.90,1.10)
    slant=rng.uniform(-0.22,0.24)
    angle=rng.uniform(-0.075,0.075)
    wavex=rng.uniform(-0.32,0.32); wavey=rng.uniform(-0.20,0.20)
    phase=rng.uniform(0,2*np.pi)
    warped=[]
    for p in paths:
        x=p[:,0]*sx; y=p[:,1]*sy
        x=x-slant*(y-8)+wavex*np.sin(y*np.pi/16+phase)
        y=y+wavey*np.sin(x*np.pi/10+phase)
        xr=x*np.cos(angle)-(y-8)*np.sin(angle)
        yr=x*np.sin(angle)+(y-8)*np.cos(angle)+8
        warped.append(np.stack([xr,yr],axis=1))
    allp=np.concatenate(warped); lo=allp.min(0); hi=allp.max(0)
    scale=20.0/max(hi-lo)*rng.uniform(0.88,1.02)
    paths=[(p-(lo+hi)/2)*scale+13.5 for p in warped]
    segments=[]
    for p in paths:
        for a,b in zip(p[:-1],p[1:]):
            v=b-a; length=np.linalg.norm(v)
            segments.append([a,v,float(length)])
    total=sum(s[2] for s in segments)+1e-12
    cumulative=0.0; grouped=[]
    for a,v,length in segments:
        group=min(parts-1,int(parts*(cumulative+0.5*length)/total))
        grouped.append((group,a,v))
        cumulative+=length
    width=rng.uniform(1.0,1.8); gain=rng.uniform(0.82,1.0)
    yy,xx=np.mgrid[0:28,0:28].astype(np.float64)
    shift=np.zeros(2)
    for iteration in range(2):
        dist=np.full((parts,28,28),100.0)
        for group,a,v in grouped:
            a=a+shift
            rx=xx-a[0]; ry=yy-a[1]
            t=np.clip((rx*v[0]+ry*v[1])/(np.dot(v,v)+1e-12),0,1)
            d2=(rx-t*v[0])**2+(ry-t*v[1])**2
            dist[group]=np.minimum(dist[group],d2)
        image=gain*np.clip(1.15-np.sqrt(dist.min(0))/width,0,1)
        if iteration==0:
            total_ink=image.sum()
            if total_ink>0:
                cy=(image*yy).sum()/total_ink; cx=(image*xx).sum()/total_ink
                shift+=np.array([13.5-cx,13.5-cy])
    return (gain*np.clip(1.15-np.sqrt(dist)/width,0,1)).astype(np.float32)


def blur5(x):
    g=np.array([1.,4.,6.,4.,1.])/16
    p=np.pad(x,((0,0),(2,2),(2,2)))
    y=sum(g[i]*p[:,i:i+28,:] for i in range(5))
    p=np.pad(y,((0,0),(0,0),(2,2)))
    return sum(g[i]*p[:,:,i:i+28] for i in range(5))


def feature_kernels():
    yy,xx = np.mgrid[-4:5,-4:5].astype(np.float64)
    kernels=[]
    for sigma in (0.65,1.0,1.5,2.3,3.5):
        k=np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma))
        kernels.append(k/k.sum())
    for theta in np.arange(8)*np.pi/8:
        u=xx*np.cos(theta)+yy*np.sin(theta)
        v=-xx*np.sin(theta)+yy*np.cos(theta)
        g=np.exp(-(u*u/(2*2.4*2.4)+v*v/(2*0.85*0.85)))
        k=(1-v*v/(0.85*0.85))*g
        k=k/(np.linalg.norm(k)+1e-12)
        kernels.append(0.45*k)
    return np.asarray(kernels,dtype=np.float32)[:,None]


def describe(images,kernels):
    # Exactly the operations used by the submitted network.
    k=kernels[:,0].astype(np.float64)
    p=np.pad(images,((0,0),(4,4),(4,4)))
    out=np.empty((len(images),len(k),28,28),np.float64)
    for c in range(len(k)):
        out[:,c]=0
        for dy in range(9):
            for dx in range(9):
                out[:,c]+=k[c,dy,dx]*p[:,dy:dy+28,dx:dx+28]
    out=np.maximum(out,0)
    return out.reshape(len(images),len(k),7,4,7,4).mean((3,5))


def ring_kernels():
    yy,xx=np.mgrid[-7:8,-7:8].astype(np.float64)
    rr=np.sqrt(xx*xx+yy*yy)
    aa=np.arctan2(yy,xx)
    kernels=[]
    for radius in (3.0,4.5,6.0):
        for sector in range(8):
            angle=sector*np.pi/4
            da=(aa-angle+np.pi)%(2*np.pi)-np.pi
            k=np.exp(-((rr-radius)/1.15)**2/2)*np.exp(-(da/(np.pi/5))**2/2)
            k*=4.0/k.sum()
            kernels.append(k)
    return np.asarray(kernels,dtype=np.float32)[:,None]


def proximity(images,radius=4):
    x=np.clip(20*images-3,0,1)
    p=x.copy()
    total=x.copy()
    for _ in range(radius):
        q=np.pad(p,((0,0),(1,1),(1,1)))
        p=np.maximum.reduce([q[:,dy:dy+28,dx:dx+28] for dy in range(3) for dx in range(3)])
        total+=p
    return x,total


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(20261119)
    per_class = 1200
    kernels=feature_kernels()
    rows=[]
    biases=[]
    for d in range(10):
        templates = np.stack([make_template(d,i,rng) for i in range(per_class)])
        features = describe(templates,kernels)
        flat = features.reshape(per_class,-1)
        flat = flat / (np.linalg.norm(flat,axis=1,keepdims=True)+1e-12)
        rows.append(flat.astype(np.float32))
        biases.append(np.zeros(per_class,np.float32))
    W = np.concatenate(rows).astype(np.float32)
    b = np.concatenate(biases).astype(np.float32)
    per_parts=80; part_count=4
    part_rows=[]
    for d in range(10):
        for v in range(per_parts):
            p=make_part_template(d,v,rng,part_count)
            p=blur5(p.reshape(-1,28,28)).reshape(part_count,28,28)[:,2:25,2:25]
            p=p.reshape(part_count,-1)
            p=p-p.mean(axis=1,keepdims=True)
            p=0.08*p/(np.linalg.norm(p,axis=1,keepdims=True)+1e-12)
            part_rows.append(p)
    part_W=np.concatenate(part_rows).astype(np.float32).reshape(10*per_parts*part_count,1,23,23)
    g=np.array([1.,4.,6.,4.,1.],np.float32)/16
    class_bias=np.array([0.01,0.01,0.035,0.025,-0.04,0.035,0.02,-0.025,-0.03,-0.025],np.float32)
    weights = {
        'smooth.W':kernels,
        'smooth.b':np.zeros(len(kernels),np.float32),
        'match.W':W,
        'match.b':b,
        'adjust.W':np.eye(10,dtype=np.float32),
        'adjust.b':class_bias,
        'ink_blur.W':(g[:,None]*g[None,:])[None,None],
        'ink_blur.b':np.zeros(1,np.float32),
        'parts.W':part_W,
        'parts.b':np.full(10*per_parts*part_count,-0.02,np.float32),
        'part_neg.W':-np.eye(10*per_parts*part_count,dtype=np.float32)[:,:,None,None],
        'part_neg.b':np.zeros(10*per_parts*part_count,np.float32),
        'part_pos.W':-0.65*np.eye(10*per_parts,dtype=np.float32)[:,:,None,None],
        'part_pos.b':np.zeros(10*per_parts,np.float32),
    }
    np.savez(out/'weights.npz',**weights)
    net={'layers':[
        {'name':'smooth','op':'conv','inputs':['image'],'out':len(kernels),'kernel':[9,9],'stride':1,'padding':4},
        {'name':'respond','op':'relu','inputs':['smooth']},
        {'name':'cells','op':'avgpool','inputs':['respond'],'size':[4,4],'stride':[4,4]},
        {'name':'match','op':'dense','inputs':['cells'],'out':10*per_class},
        {'name':'by_class','op':'reshape','inputs':['match'],'shape':[10,per_class,1]},
        {'name':'best','op':'maxpool','inputs':['by_class'],'size':[per_class,1],'stride':[per_class,1]},
        {'name':'base_scores','op':'reshape','inputs':['best'],'shape':[10]},
        {'name':'adjust','op':'dense','inputs':['base_scores'],'out':10},
        {'name':'ink_blur','op':'conv','inputs':['image'],'out':1,'kernel':[5,5],'stride':1,'padding':2},
        {'name':'parts','op':'conv','inputs':['ink_blur'],'out':10*per_parts*part_count,'kernel':[23,23],'stride':1,'padding':0},
        {'name':'part_best','op':'maxpool','inputs':['parts'],'size':[6,6],'stride':[6,6]},
        {'name':'part_neg','op':'conv','inputs':['part_best'],'out':10*per_parts*part_count,'kernel':[1,1],'stride':1,'padding':0},
        {'name':'part_groups','op':'reshape','inputs':['part_neg'],'shape':[10*per_parts,part_count,1]},
        {'name':'part_min','op':'maxpool','inputs':['part_groups'],'size':[part_count,1],'stride':[part_count,1]},
        {'name':'part_pos','op':'conv','inputs':['part_min'],'out':10*per_parts,'kernel':[1,1],'stride':1,'padding':0},
        {'name':'part_classes','op':'reshape','inputs':['part_pos'],'shape':[10,per_parts,1]},
        {'name':'part_max','op':'maxpool','inputs':['part_classes'],'size':[per_parts,1],'stride':[per_parts,1]},
        {'name':'part_scores','op':'reshape','inputs':['part_max'],'shape':[10]},
        {'name':'scores','op':'add','inputs':['adjust','part_scores']},
    ]}
    (out/'network.json').write_text(json.dumps(net,separators=(',',':')))


if __name__ == '__main__':
    main()
