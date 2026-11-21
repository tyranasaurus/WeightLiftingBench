import json, sys
from pathlib import Path
import numpy as np

# Pen paths on an abstract (roughly 0..10) writing grid. Separate strokes use '/'.
STYLES = [
[
'5,0 2,1 0,5 0,8 2,10 5,10 8,8 9,4 8,1 5,0',
'6,0 3,0 1,2 0,6 1,9 4,10 7,9 9,6 9,2 6,0',
'5,0 2,1 1,4 1,8 3,10 6,10 8,7 8,3 7,0 5,0',
'7,1 4,0 2,2 0,6 1,9 4,10 7,8 9,5 9,2 7,1',
], [
'5,0 5,10', '2,3 6,0 5,10', '1,4 5,0 5,10 / 1,10 9,10',
'2,2 6,0 5,6 5,10', '6,0 5,5 3,10',
], [
'0,2 2,0 6,0 8,2 8,4 5,6 0,10 9,10',
'1,2 3,0 7,0 9,2 8,4 5,7 1,10 6,9 10,10',
'1,3 1,1 4,0 7,1 8,3 7,5 3,8 1,10 9,9',
'1,1 4,0 7,1 8,3 6,6 2,8 1,9 3,10 9,10',
'1,2 4,0 8,0 9,2 6,5 2,7 0,9 2,10 9,9',
], [
'0,1 3,0 7,0 9,2 7,4 4,5 7,5 9,7 8,9 5,10 1,10 0,9',
'1,0 9,0 5,4 3,5 7,5 9,7 8,9 5,10 2,10 0,8',
'1,2 2,0 6,0 8,1 8,3 6,5 4,5 7,5 9,7 8,9 6,10 2,10 1,8',
'1,1 4,0 7,1 7,3 5,5 8,6 9,8 7,10 2,10 0,9',
], [
'3,0 0,6 10,6 / 7,0 6,10',
'6,0 0,7 10,7 / 7,2 7,10',
'7,0 0,6 8,6 7,0 6,10',
'1,0 0,5 2,6 9,5 / 7,1 6,10',
'7,0 1,5 0,7 9,6 / 8,1 7,10',
], [
'9,0 2,0 1,5 4,4 7,5 9,7 8,9 5,10 1,10 0,9',
'9,0 3,1 1,4 5,4 8,5 9,7 7,9 4,10 0,9',
'8,0 1,0 1,4 4,4 7,5 8,7 7,9 4,10 1,9',
'9,1 4,0 2,1 1,4 4,4 8,6 8,8 6,10 2,10 0,8',
], [
'7,0 4,1 1,4 0,7 1,9 4,10 7,9 8,7 7,5 4,4 1,6',
'8,0 5,0 2,3 0,7 1,9 4,10 7,9 9,7 8,5 5,5 1,7',
'6,0 3,2 1,5 1,8 3,10 6,10 8,8 8,6 6,5 3,5 1,7',
'8,1 6,0 3,1 1,4 0,7 2,10 6,10 8,8 7,6 4,5 1,6',
], [
'0,0 9,0 6,4 3,10',
'0,1 3,0 9,0 6,5 5,10 / 2,5 9,5',
'0,2 1,0 9,0 8,2 5,6 4,10',
'0,0 4,1 10,0 6,4 3,10',
], [
'5,5 2,3 2,1 4,0 7,0 9,2 8,4 5,5 2,6 0,8 1,10 5,10 8,9 8,7 5,5',
'5,5 2,4 1,2 3,0 6,0 8,1 8,3 5,5 2,7 1,9 3,10 6,10 8,9 8,7 5,5',
'5,5 1,3 1,1 4,0 7,1 8,3 5,5 2,6 0,8 2,10 6,10 9,8 8,6 5,5',
'6,5 3,4 2,2 4,0 7,0 9,1 8,3 6,5 2,7 1,9 3,10 6,10 8,9 8,7 6,5',
], [
'8,4 6,5 3,5 1,4 1,2 3,0 6,0 8,1 9,3 8,6 6,9 4,10',
'8,4 5,5 2,4 1,2 3,0 6,0 8,1 8,4 7,7 7,10',
'9,3 7,0 4,0 1,1 0,3 2,5 5,5 8,3 8,6 7,9 5,10 2,10',
'8,4 5,6 2,5 0,3 1,1 4,0 7,1 8,3 8,10',
]]

# Additional common pen constructions, specified independently of any images.
STYLES[2] += ['0,2 2,0 6,0 8,2 7,4 4,6 1,8 0,10 2,10 4,8 6,8 10,9',
              '0,3 1,1 3,0 6,0 8,2 8,4 6,6 1,9 0,10 5,10 10,10']
STYLES[4] += ['5,0 1,5 0,6 4,6 9,6 / 6,1 6,10',
              '8,0 3,4 0,7 7,7 8,0 / 7,6 7,10']
STYLES[5] += ['1,0 0,5 3,4 6,4 8,6 8,8 6,10 2,10 0,9 / 2,0 9,0',
              '8,0 3,0 2,2 1,5 4,4 7,4 9,6 8,8 6,10 2,10 0,8']
STYLES[9] += ['7,6 3,6 1,4 1,2 3,0 6,0 8,1 8,4 7,10',
              '7,4 5,5 2,5 0,3 1,1 4,0 7,0 8,2 8,5 6,8 5,10',
              '9,0 8,4 5,5 2,5 0,3 1,1 4,0 7,0 9,1 8,5 7,10']

def points(style):
    curves=[]
    for s in style.split('/'):
        p=np.array([list(map(float,t.split(','))) for t in s.split()])
        q=np.pad(p,((1,1),(0,0)),mode='edge')
        t=np.linspace(0,1,18)[:,None]
        for a,b,c,d in zip(q,q[1:],q[2:],q[3:]):
            curves.append(b + .5*((c-a)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t**3))
    return np.concatenate(curves)

def build(folder):
    layers=[]
    weights={}
    def op(name,kind,inputs,**settings):
        layers.append(dict(name=name,op=kind,inputs=inputs,**settings))
    def linear(name,kind,src,w,b,**kw):
        weights[name+'.W']=np.asarray(w,np.float32)
        weights[name+'.b']=np.asarray(b,np.float32)
        op(name,kind,[src],out=len(b),**kw)
    # Soft saturation reduces differences in pen pressure.
    linear('knees','conv','image',np.ones((2,1,1,1)),[0,-.6],kernel=[1,1])
    op('positive','relu',['knees'])
    linear('ink','conv','positive',np.array([1,-1]).reshape(1,2,1,1)/.6,[0],kernel=[1,1])
    op('coarse','avgpool',['ink'],size=[2,2],stride=[2,2])
    op('near','maxpool',['coarse'],size=[3,3],stride=[1,1])
    op('raw_grid','reshape',['coarse'],shape=[1,1,196])
    op('near_grid','reshape',['near'],shape=[1,1,144])
    rng=np.random.default_rng(813)
    Y,X=np.mgrid[0:28,0:28]
    W=[]
    B=[]
    count=1024
    for digit,styles in enumerate(STYLES):
        paths=[points(s) for s in styles]
        for idx in range(count):
            p=paths[idx%len(paths)].copy()
            p-= (p.max(0)+p.min(0))/2
            if digit!=1:
                p[:,0]*=rng.uniform(.68,1.12)
            else:
                p[:,0]*=rng.uniform(.45,1.05)
            p[:,0]+=rng.uniform(-.29,.40)*p[:,1]
            p[:,1]+=rng.uniform(-.16,.16)*p[:,0]
            p *= rng.uniform(16.2,19.0)/np.ptp(p,axis=0).max()
            p[:,0]+=rng.uniform(-.017,.017)*p[:,1]**2
            p-=p.mean(0)
            p+=13.5
            def distance():
                return np.sqrt(np.min((X[:,:,None]-p[:,0])**2+(Y[:,:,None]-p[:,1])**2,axis=2))
            width=rng.uniform(.78,1.9)
            ink=np.clip(width+.65-distance(),0,1)
            p+=13.5-np.array([(ink*X).sum(),(ink*Y).sum()])/ink.sum()+rng.uniform(-1.25,1.25,2)
            d=distance()
            t=np.clip(width+.65-d,0,1).reshape(14,2,14,2).mean((1,3))
            # Weighted, two-sided soft chamfer similarity plus a sharper overlap term.
            dd=d.reshape(14,2,14,2).min((1,3))
            outside=np.clip((dd-width-.10)/2.2,0,1)
            tt=t[1:-1,1:-1].ravel()
            raw=(.84*t-.96*outside)/t.sum()
            near=.67*tt/tt.sum()
            W.append(np.r_[raw.ravel(),near])
            B.append(-.42*(t*t).sum()/t.sum()-.61)
    W=np.array(W,np.float32)
    # Separate affine projections let us combine two grids of different sizes.
    linear('local_match','dense','raw_grid',W[:,:196],B)
    linear('tolerant_match','dense','near_grid',W[:,196:],np.zeros(len(B)))
    op('match','add',['local_match','tolerant_match'])
    op('families','reshape',['match'],shape=[10,1,count])
    op('best','maxpool',['families'],size=[1,count],stride=[1,count])
    op('scores','reshape',['best'],shape=[10])
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'network.json').write_text(json.dumps(dict(layers=layers),indent=2)+'\n')
    np.savez(folder/'weights.npz',**weights)

if __name__=='__main__':
    build(sys.argv[1])
