"""Build a bank of analytic hand-shape correlators, without reading any files."""
import json
import sys
from pathlib import Path
import numpy as np

layers=[]
weights={}
def layer(name,op,inputs,**kw):
    layers.append(dict(name=name,op=op,inputs=[inputs] if isinstance(inputs,str) else inputs,**kw))
    return name

def affine(name,op,src,w,b,**kw):
    weights[name+'.W']=np.asarray(w,np.float32)
    weights[name+'.b']=np.asarray(b,np.float32)
    return layer(name,op,src,out=len(w),**kw)

# Four axes; both polarities are accepted. Derivatives are smoothed
# perpendicular to the gradient to reduce sensitivity to pixel noise.
u=np.arange(-2,3,dtype=float)
x,y=np.meshgrid(u,u)
grad=np.stack([(x*np.cos(t)+y*np.sin(t))*np.exp(-(x*x+y*y)/2.6)/5 for t in np.arange(4)*np.pi/4])

# A palm is an ellipse; each digit is a chain of tapered capsules.
# Coordinates are in palm-width units, with positive y toward the wrist.
def finger(radius,*points): return radius,points

def fist(width=1):
    return [finger(.135,(x*width,-.21),(x*width,-.49+abs(x)*.26),(x*width+.06,-.31)) for x in (-.35,-.12,.12,.34)]
def upright(x,tip,dx=0):
    return finger(.119,(x,-.21),(x+dx*.48,tip*.64),(x+dx,tip))
def thumb(x,y): return finger(.145,(.27,.20),(.45,.01),(x,y))

def model(k):
    p=(0,0,.47,.48); wrist=(-.14,1.06)
    f=fist()
    if k==0: f+=[thumb(-.02,-.24)]
    if k==1: f=[upright(x,t,d) for x,t,d in [(-.32,-.96,-.06),(-.12,-1.22,-.04),(.10,-1.29,-.01),(.31,-1.12,.02)]]+[thumb(-.16,-.10)]
    if k in (2,13):
        p=(-.20,.08,.35,.45); wrist=(-.55,.98)
        f=[finger(.17,(-.34,-.03),(-.27,-.57),(.07,-.84),(.40,-.77),(.58,-.55 if k==2 else -.34)),finger(.155,(-.04,.25),(.32,.23),(.58,.03),(.58,-.30 if k==13 else .03))]
    if k==3: f=f[:3]+[upright(.14,-1.28,-.02),thumb(.25,-.28)]
    if k==4: f=[finger(.15,(x,-.15),(x-.04,-.53),(x+.08,-.37)) for x in (-.34,-.12,.10,.30)]+[thumb(-.26,-.19)]
    if k==5: f=[upright(-.32,-1.05,-.27),upright(-.13,-1.28,-.08),upright(.08,-1.20,.23),finger(.115,(.18,-.17),(.52,-.59),(.75,-.46),(.64,-.24)),thumb(.64,-.24)]
    if k in (6,7):
        p=(-.19,.13,.39,.40); wrist=(-.58,.90)
        f=[finger(.125,(-.35,-.03),(.10,-.39),(.78,-.55 if k==6 else -.39)),finger(.14,(-.02,.34),(.28,.28),(.31,.02))]
        f += [finger(.115,(-.18,.01),(.21,-.20),(.77,-.26))] if k==7 else [finger(.13,(-.21,-.04),(.18,-.18),(.64,-.20))]
    if k==8: f=f[1:]+[upright(-.34,-1.11,-.12),thumb(-.13,-.17)]
    if k==9: f=f[:2]+[upright(-.04,-1.23,-.20),upright(.18,-.97,.32),thumb(.09,-.62)]
    if k==10: f=f[:2]+[upright(.02,-1.30,-.18),thumb(.91,-.19)]
    if k==11: f=fist()+[thumb(-.33,-.02)]
    if k==12: f=fist(.89)+[thumb(-.19,-.07)]
    if k==14:
        p=(.17,.04,.42,.36); wrist=(.45,.95)
        f=[finger(.13,(.29,-.15),(-.11,-.43),(-.78,-.53)),finger(.125,(.02,-.03),(-.38,.13),(-.50,.55)),finger(.14,(.32,.25),(.06,.29),(-.23,.04))]
    if k==15:
        p=(.04,-.23,.48,.36); wrist=(.57,.45)
        f=[finger(.14,(.14,-.42),(-.34,-.19),(-.63,.53)),finger(.14,(.37,-.08),(.19,.24),(.08,.56)),finger(.15,(-.37,-.17),(-.29,.08),(-.12,.12))]
    if k==16: f=f[:2]+[upright(.03,-1.28,.20),upright(.22,-1.20,-.13),thumb(-.04,-.15)]
    if k==17: f=fist()+[finger(.16,(.32,.13),(.16,-.22),(-.27,-.33))]
    if k==18:
        f=[finger(.145,(-.30,.20),(-.34,-.23),(.22,-.31),(.74,-.29)),finger(.13,(-.02,-.08),(.07,-.68)),finger(.15,(-.22,.12),(.28,-.03))]
    if k==19: f=f[:2]+[upright(.03,-1.24,-.02),upright(.25,-1.24,-.01),thumb(-.10,-.18)]
    if k==20: f=f[:2]+[upright(.02,-1.21,-.25),upright(.24,-1.25,.28),thumb(-.10,-.15)]
    if k==21: f=f[:1]+[upright(-.19,-1.09,-.22),upright(.03,-1.25,-.04),upright(.24,-1.12,.27),thumb(-.21,-.15)]
    if k==22: f=f[:2]+[finger(.135,(.13,-.18),(.19,-.73),(-.04,-.91),(-.25,-.76)),thumb(.17,-.31)]
    if k==23: f=f[1:3]+[upright(-.33,-1.03,-.14),thumb(.87,-.38)]
    return p,wrist,f

coords=(np.arange(28)-13.5)/10
XX,YY=np.meshgrid(coords,coords)
def render(k,angle,sx,sy,dx,dy,pose,mirror):
    (px,py,a,b),wrist, fingers=model(k)
    # Adjust framing for the hand's overall pointing direction.
    cy=-.29 if k in (1,3,5,8,9,10,16,19,20,21,22,23) else -.05
    X=((XX-dx)*np.cos(angle)+(YY-dy)*np.sin(angle))*mirror/sx
    Y=(-(XX-dx)*np.sin(angle)+(YY-dy)*np.cos(angle))/sy+cy
    palm=1-np.sqrt(((X-px)/a)**2+((Y-py)/b)**2)
    mask=palm.copy()
    relief=np.maximum(palm,0)*.28
    def chain(radius,points):
        nonlocal mask, relief
        for j,((ax,ay),(bx,by)) in enumerate(zip(points,points[1:])):
            vx,vy=bx-ax,by-ay
            t=np.clip(((X-ax)*vx+(Y-ay)*vy)/max(vx*vx+vy*vy,1e-12),0,1)
            d=np.hypot(X-ax-t*vx,Y-ay-t*vy)
            width=radius*(1-.11*t)
            mask=np.maximum(mask,(width-d)*2.5)
            relief=np.maximum(relief,.42*np.maximum(1-(d/width)**2,0))
    chain(.29,[(px,py+.18),wrist,(wrist[0]-.06,2.4)])
    for rad,pts in fingers:
        ax,ay=pts[0]
        chain(rad,[(ax+(bx-ax)*(1+pose),ay+(by-ay)*(1+pose)) for bx,by in pts])
    return np.clip(.5+mask*4,0,1)*(.60+relief)

def feature(im):
    windows=np.lib.stride_tricks.sliding_window_view(np.pad(im,2),(5,5))
    d=np.einsum('kyx,abyx->kab',grad,windows,optimize=False)
    d=np.abs(d)
    d=np.maximum(0,d-.65*np.roll(d,2,axis=0))
    return d.reshape(4,14,2,14,2).mean(axis=(2,4))

def main(dest):
    # Opposite-polarity responses merge before directional sharpening.
    a=affine('derivative','conv','image',np.concatenate([grad,-grad])[:,None],np.zeros(8),kernel=[5,5],stride=1,padding=2)
    a=layer('half_wave','relu',a)
    w=np.zeros((4,8,1,1))
    for i in range(4):
        w[i,[i,i+4],0,0]=1
        j=(i+2)%4
        w[i,[j,j+4],0,0]=-.65
    a=affine('orientation','conv',a,w,np.zeros(4),kernel=[1,1],stride=1,padding=0)
    a=layer('oriented_edges','relu',a)
    a=layer('soft_spatial_bins','avgpool',a,size=[2,2],stride=[2,2])
    a=layer('contrast_equalization','norm',a)
    rng=np.random.default_rng(371)
    n=512
    # Quasi-uniform nuisance coverage via independently shuffled strata.
    q=np.stack([(rng.permutation(n)+.5)/n for _ in range(7)],axis=1)*2-1
    bank=[]
    # Suppress evidence from the padded border and the dangling wrist.
    taper=np.ones((14,14)); taper[[0,-1],:]=0; taper[:,[0,-1]]=0; taper[-2,:]*=.38
    for k in range(24):
        for r,s,t,x,y,p,m in q:
            z=feature(render(k,r*.43, .96+s*.24,1.02+t*.21,x*.30,y*.27,p*.24, -1 if m>.75 else 1))
            z*=taper
            # One-bin tolerance around the contour, with more weight on the core.
            h=np.pad(z,((0,0),(1,1),(1,1)))
            z=.54*z+.115*(h[:,2:,1:-1]+h[:,:-2,1:-1]+h[:,1:-1,2:]+h[:,1:-1,:-2])
            z-=z.mean(axis=(1,2),keepdims=True)
            z/=max(np.linalg.norm(z),1e-6)
            bank.append(z.ravel())
    a=affine('shape_similarity','dense',a,bank,np.zeros(len(bank)))
    a=layer('class_by_pose','reshape',a,shape=[24,1,n])
    a=layer('best_pose','maxpool',a,size=[1,n],stride=[1,n])
    layer('scores','reshape',a,shape=[24])
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    (dest/'network.json').write_text(json.dumps({'layers':layers},indent=2)+'\n')
    np.savez(dest/'weights.npz',**weights)

if __name__=='__main__': main(sys.argv[1])
