"""Compile an anatomical contour matcher into the permitted network operations."""
import json
import sys
from pathlib import Path
import numpy as np

# Hand-local coordinates in palm widths, x right, y up. Each stroke is
# (half-width, polyline). The palm is an ellipse; wrists are capsules.
I = (.098, [(.22,.19),(.23,.65),(.23,1.10)])
M = (.096, [(.02,.20),(.015,.70),(.00,1.20)])
R = (.088, [(-.17,.19),(-.20,.67),(-.21,1.04)])
P = (.077, [(-.31,.15),(-.36,.54),(-.40,.84)])

def stroke(r,*p): return (r,p)
def aim(f,x,y):
    r,p=f
    a=np.array(p[0]); b=np.array([x,y]); return stroke(r,a,.5*(a+b),b)

def fold(f,h=.42):
    r,p=f; x,y=p[0]; return stroke(r,(x,y-.08),(x-.035,h),(x+.02,h-.09))

def pose(k,v):
    fingers=[I,M,R,P]; palm=(0,0,.38,.35); wrist=[(0,-.19),(-.12,-1.12)]; holes=[]
    folded=[fold(f,.46-.055*j) for j,f in enumerate(fingers)]
    t=stroke(.12,(.28,-.17),(.32,.08),(.15,.25))
    s=folded+[t]
    if k==0: s=folded+[stroke(.115,(.26,-.22),(.38,.04),(.39,.41))]
    if k==1: s=fingers+[stroke(.125,(.25,-.17),(.22,.03),(-.03,.25))]
    if k in (2,13):
        palm=(-.10,-.025,.29,.35)
        wrist=[(-.14,-.23),(-.46,-1.03)]
        tip=(.37,.43) if k==13 else (.30,.65)
        s=[stroke(.156,(-.24,.06),(-.24,.39),(-.08,.62),(.13,.67),tip),
           stroke(.122,(-.05,-.23),(.21,-.17),(.39,.02),(.44,.31 if k==13 else .19))]
    if k==3: s=[I]+folded[1:]+[stroke(.107,(.24,-.16),(.37,.10),(.19,.37),(.04,.46))]; holes=[(.17,.29,.072,.092)]
    if k==4: s=[fold(f,.42-.026*j) for j,f in enumerate(fingers)]+[stroke(.115,(.26,-.14),(.10,.10),(-.19,.14))]
    if k==5:
        s=[aim(M,.14,1.14),aim(R,-.19,1.24),aim(P,-.49,1.00),stroke(.10,(.10,.18),(.24,.49),(.46,.47),(.48,.27)),stroke(.10,(.18,-.21),(.46,-.04),(.50,.24))]
        holes=[(.35,.23,.09,.12)]
    if k in (6,7,14,15,18):
        tips={6:(.96,.54),7:(1.02,.62),14:(-.84,-.37),15:(.13,-.89),18:(.92,.43)}
        x,y=tips[k]
        if k==15:
            palm=(-.05,.04,.38,.36); wrist=[(.10,.19),(.58,.79)]
            s=[stroke(.11,(-.27,.09),(-.35,-.26),(-.30,-.78)),stroke(.12,(.13,.09),(.20,-.29),(.17,-.57))]
        elif k==14:
            wrist=[(.18,-.19),(.65,-.92)]
            s=[stroke(.11,(0,.26),(-.48,.28),(-.91,.12)),stroke(.10,(-.08,.13),(-.43,-.07),(x,y)),stroke(.12,(.23,-.22),(-.04,-.18),(-.26,-.23))]
        else:
            wrist=[(-.13,-.17),(-.53,-.99)]
            s=[stroke(.10,(-.01,.22),(.39,y-.05),(x,y))]
            if k==7: s+=[stroke(.094,(0,.11),(.48,y-.22),(x-.04,y-.18))]
            if k==6: s+=[stroke(.108,(.16,-.22),(.36,-.05),(.61,.05))]
            if k==18: s+=[stroke(.12,(-.32,-.15),(-.22,.14),(.15,.25))]
    if k==8: s=folded[:3]+[aim(P,-.43,1.13),t]
    if k==9: s=[aim(I,.23,1.28),aim(M,-.43,.91)]+folded[2:]+[stroke(.125,(.24,-.20),(.07,.20),(-.10,.56))]
    if k==10: s=[I]+folded[1:]+[stroke(.114,(.19,-.20),(.55,-.07),(.90,.14))]
    if k in (11,12,17):
        s=[fold(f,.37-.036*j) for j,f in enumerate(fingers)]
        thumb={11:-.23,12:-.08,17:.12}[k]; s+=[stroke(.13,(.25,-.22),(.29,.03),(thumb,.29 if k!=17 else .46))]
    if k==16: s=[aim(I,-.032,1.14),aim(M,.22,1.23)]+folded[2:]+[t]
    if k==19: s=[aim(I,.11,1.12),aim(M,-.062,1.21)]+folded[2:]+[t]
    if k==20: s=[aim(I,.37,1.07),aim(M,-.25,1.22)]+folded[2:]+[t]
    if k==21: s=[aim(I,.45,1.08),M,aim(R,-.38,1.07),folded[3],t]
    if k==22: s=[stroke(.10,(.21,.05),(.32,.53),(.26,.75),(.03,.71),(-.01,.56))]+folded[1:]+[t]
    if k==23: s=folded[:3]+[aim(P,-.56,1.01),stroke(.115,(.19,-.20),(.52,.01),(.75,.25))]
    if v:
        # Alternative degree of flexion and wrist lean.
        s=[(r,[(x+.10*y,y*(1.08 if y>0 else 1)) for x,y in p]) for r,p in s]
        wrist=[(wrist[0][0],wrist[0][1]),(wrist[1][0]-.19,wrist[1][1])]
    return palm,wrist,s,holes

# Gaussian smoothing perpendicular to centered differences, two polarities
# and four directions. Absolute responses discard foreground polarity.
G=np.array([1,4,6,4,1],float); G/=G.sum()
D=np.array([-1,-2,0,2,1],float)/6
X=np.outer(G,D); Y=X.T
K=np.array([X,Y,(X+Y)*.70710678,(X-Y)*.70710678])

def render(model,ang,size,dx,dy,aspect):
    palm,wrist,s,holes=model
    cx,cy,rx,ry=palm
    # Fit a hand-centered crop to the articulated geometry, not to examples.
    a=np.deg2rad(ang); c=np.cos(a); t=np.sin(a)
    mat=np.array([[c*aspect,-t],[t*aspect,c]])
    pts=np.array([(cx+rx*np.cos(a),cy+ry*np.sin(a)) for a in np.arange(24)*np.pi/12]+[p for r,ps in s for p in ps]+wrist)
    box=pts@mat.T
    lo=box.min(0)-.12; hi=box.max(0)+.12
    center=(lo+hi)*.5
    scale=size/max(hi-lo)
    y,x=np.mgrid[:28,:28]; coords=np.stack([x-13.5-dx,13.5-y-dy],-1)/scale+center
    u,w=(coords@np.linalg.inv(mat).T).transpose(2,0,1)
    field=(1-np.hypot((u-cx)/rx,(w-cy)/ry))*min(rx,ry)
    for r,ps in s+[(.235,wrist)]:
        for (x0,y0),(x1,y1) in zip(ps,ps[1:]):
            vx=x1-x0; vy=y1-y0
            h=np.clip(((u-x0)*vx+(w-y0)*vy)/(vx*vx+vy*vy),0,1)
            field=np.maximum(field,r-np.hypot(u-x0-h*vx,w-y0-h*vy))
    for x0,y0,r1,r2 in holes:
        field=np.minimum(field,(np.hypot((u-x0)/r1,(w-y0)/r2)-1)*min(r1,r2))
    return np.clip(.5+scale*field/1.65,0,1)

def descriptor(im):
    # The same fixed edge filters and local pooling used at inference.
    win=np.lib.stride_tricks.sliding_window_view(im,(5,5))
    a=np.abs(np.einsum('yxij,kij->kyx',win,K))
    return np.lib.stride_tricks.sliding_window_view(a,(4,4),axis=(1,2))[:,::2,::2].max(axis=(-1,-2))

def main(folder):
    folder.mkdir(parents=True,exist_ok=True)
    layers=[]; weights={}
    def layer(name,op,inputs,**kw): layers.append(dict(name=name,op=op,inputs=inputs,**kw))
    def affine(name,op,src,W,b,**kw):
        layer(name,op,[src],out=len(W),**kw); weights[name+'.W']=np.asarray(W,np.float32); weights[name+'.b']=np.asarray(b,np.float32)
    layer('contrast','norm',['image'])
    affine('derivatives','conv','contrast',np.concatenate([K,-K])[:,None],np.zeros(8),kernel=[5,5],stride=1,padding=0)
    layer('positive_edges','relu',['derivatives'])
    affine('unsigned_edges','conv','positive_edges',np.concatenate([np.eye(4),np.eye(4)],axis=1)[:,:,None,None],np.zeros(4),kernel=[1,1],stride=1,padding=0)
    layer('local_edges','maxpool',['unsigned_edges'],size=[4,4],stride=[2,2])
    bank=[]
    for k in range(24):
        for v in range(2):
            model=pose(k,v)
            for a in (-24,-12,0,12,24):
                for size in (22,26,30,34):
                    for dx in (-3,0,3):
                        for dy in (-3,0,3):
                            for aspect in (.84,1.12):
                                f=descriptor(render(model,a,size,dx,dy,aspect))
                                # Remove the spatially constant edge baseline, then
                                # equalize template energy for score comparability.
                                f-=f.mean(axis=(1,2),keepdims=True)
                                bank.append((f/np.linalg.norm(f)).ravel())
    n=len(bank)//24
    affine('contour_matches','dense','local_edges',np.array(bank),np.zeros(len(bank)))
    layer('by_letter','reshape',['contour_matches'],shape=[1,24,n])
    layer('best_pose','maxpool',['by_letter'],size=[1,n],stride=[1,n])
    layer('scores','reshape',['best_pose'],shape=[24])
    (folder/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(folder/'weights.npz',**weights)

if __name__=='__main__': main(Path(sys.argv[1]))
